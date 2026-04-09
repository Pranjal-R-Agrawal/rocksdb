//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "table/cuckoo/cuckoo_table_reader.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "memory/arena.h"
#include "options/cf_options.h"
#include "rocksdb/iterator.h"
#include "rocksdb/table.h"
#include "table/cuckoo/cuckoo_table_factory.h"
#include "table/get_context.h"
#include "table/internal_iterator.h"
#include "table/meta_blocks.h"
#include "util/coding.h"

namespace ROCKSDB_NAMESPACE {
namespace {
const uint64_t CACHE_LINE_MASK = ~((uint64_t)CACHE_LINE_SIZE - 1);
const uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();
}  // namespace

extern const uint64_t kCuckooTableMagicNumber;

CuckooTableReader::CuckooTableReader(
    const ImmutableOptions& ioptions,
    std::unique_ptr<RandomAccessFileReader>&& file, uint64_t file_size,
    const Comparator* comparator,
    uint64_t (*get_slice_hash)(const Slice&, uint32_t, uint64_t))
    : file_(std::move(file)),
      is_last_level_(false),
      identity_as_first_hash_(false),
      use_module_hash_(false),
      num_hash_func_(0),

      key_length_(0),
      user_key_length_(0),
      value_length_(0),
      bucket_length_(0),
      cuckoo_block_size_(0),
      cuckoo_block_bytes_minus_one_(0),
      table_size_(0),
      kv_data_offset_(0),
      ucomp_(comparator),
      get_slice_hash_(get_slice_hash) {
  if (!ioptions.allow_mmap_reads) {
    status_ = Status::InvalidArgument("File is not mmaped");
    return;
  }
  {
    std::unique_ptr<TableProperties> props;
    // TODO: plumb Env::IOActivity, Env::IOPriority
    const ReadOptions read_options;
    status_ =
        ReadTableProperties(file_.get(), file_size, kCuckooTableMagicNumber,
                            ioptions, read_options, &props);
    if (!status_.ok()) {
      return;
    }
    table_props_ = std::move(props);
  }
  auto& user_props = table_props_->user_collected_properties;
  auto hash_funs = user_props.find(CuckooTablePropertyNames::kNumHashFunc);
  if (hash_funs == user_props.end()) {
    status_ = Status::Corruption("Number of hash functions not found");
    return;
  }
  num_hash_func_ = *reinterpret_cast<const uint32_t*>(hash_funs->second.data());

  key_length_ = static_cast<uint32_t>(table_props_->fixed_key_len);
  bucket_length_ = static_cast<uint32_t>(CuckooTableBucket::size());

  auto hash_table_size =
      user_props.find(CuckooTablePropertyNames::kHashTableSize);
  if (hash_table_size == user_props.end()) {
    status_ = Status::Corruption("Hash table size not found");
    return;
  }
  table_size_ =
      *reinterpret_cast<const uint64_t*>(hash_table_size->second.data());

  auto is_last_level = user_props.find(CuckooTablePropertyNames::kIsLastLevel);
  if (is_last_level == user_props.end()) {
    status_ = Status::Corruption("Is last level not found");
    return;
  }
  is_last_level_ = *reinterpret_cast<const bool*>(is_last_level->second.data());

  auto identity_as_first_hash =
      user_props.find(CuckooTablePropertyNames::kIdentityAsFirstHash);
  if (identity_as_first_hash == user_props.end()) {
    status_ = Status::Corruption("identity as first hash not found");
    return;
  }
  identity_as_first_hash_ =
      *reinterpret_cast<const bool*>(identity_as_first_hash->second.data());

  auto use_module_hash =
      user_props.find(CuckooTablePropertyNames::kUseModuleHash);
  if (use_module_hash == user_props.end()) {
    status_ = Status::Corruption("hash type is not found");
    return;
  }
  use_module_hash_ =
      *reinterpret_cast<const bool*>(use_module_hash->second.data());
  auto cuckoo_block_size =
      user_props.find(CuckooTablePropertyNames::kCuckooBlockSize);
  if (cuckoo_block_size == user_props.end()) {
    status_ = Status::Corruption("Cuckoo block size not found");
    return;
  }
  cuckoo_block_size_ =
      *reinterpret_cast<const uint32_t*>(cuckoo_block_size->second.data());
  cuckoo_block_bytes_minus_one_ = cuckoo_block_size_ * bucket_length_ - 1;
  kv_data_offset_ =
      (table_size_ + cuckoo_block_size_ - 1) * bucket_length_;
  // TODO: rate limit reads of whole cuckoo tables.
  status_ = file_->Read(IOOptions(), 0, static_cast<size_t>(file_size),
                        &file_data_, nullptr, nullptr);
}

Status CuckooTableReader::Get(const ReadOptions& /*readOptions*/,
                              const Slice& key, GetContext* get_context,
                              const SliceTransform* /* prefix_extractor */,
                              bool /*skip_filters*/) {
  const Slice user_key = ExtractUserKey(key);
  const uint64_t fp =
      CuckooFingerPrint(user_key, num_hash_func_);
  for (uint32_t hash_cnt = 0; hash_cnt < num_hash_func_; ++hash_cnt) {
    uint64_t offset =
        bucket_length_ * CuckooHash(user_key, hash_cnt, use_module_hash_,
                                    table_size_, identity_as_first_hash_,
                                    get_slice_hash_);
    const char* bucket_ptr = &file_data_.data()[offset];
    for (uint32_t block_idx = 0; block_idx < cuckoo_block_size_;
         ++block_idx, bucket_ptr += bucket_length_) {
      const auto [fingerprint, address] = CuckooTableBucket::decode(bucket_ptr);
      if (fingerprint == fp) {
        const char* kv_ptr = &file_data_.data()[address];
        const uint32_t key_len = DecodeFixed32(kv_ptr);
        const uint32_t val_len = DecodeFixed32(kv_ptr + sizeof(uint32_t));
        const Slice found_key(kv_ptr + 2 * sizeof(uint32_t), key_len);
        if (ucomp_->Equal(user_key, ExtractUserKey(found_key))) {
          const Slice value(kv_ptr + 2 * sizeof(uint32_t) + key_len, val_len);
          ParsedInternalKey found_ikey;
          Status s = ParseInternalKey(found_key, &found_ikey,
                                      false /* log_err_key */);
          if (!s.ok()) {
            return s;
          }
          bool dont_care;
          get_context->SaveValue(found_ikey, value, &dont_care, &s);
          if (!s.ok()) {
            return s;
          }
          return Status::OK();
        }
      }
    }
  }
  return Status::OK();
}

void CuckooTableReader::Prepare(const Slice& key) {
  // Prefetch the first Cuckoo Block.
  Slice user_key = ExtractUserKey(key);
  uint64_t addr =
      reinterpret_cast<uint64_t>(file_data_.data()) +
      bucket_length_ * CuckooHash(user_key, 0, use_module_hash_, table_size_,
                                  identity_as_first_hash_, nullptr);
  uint64_t end_addr = addr + cuckoo_block_bytes_minus_one_;
  for (addr &= CACHE_LINE_MASK; addr < end_addr; addr += CACHE_LINE_SIZE) {
    PREFETCH(reinterpret_cast<const char*>(addr), 0, 3);
  }
}

class CuckooTableIterator : public InternalIterator {
 public:
  explicit CuckooTableIterator(CuckooTableReader* reader);
  // No copying allowed
  CuckooTableIterator(const CuckooTableIterator&) = delete;
  void operator=(const Iterator&) = delete;
  ~CuckooTableIterator() override = default;
  bool Valid() const override;
  void SeekToFirst() override;
  void SeekToLast() override;
  void Seek(const Slice& target) override;
  void SeekForPrev(const Slice& target) override;
  void Next() override;
  void Prev() override;
  Slice key() const override;
  Slice value() const override;
  Status status() const override { return Status::OK(); }
  void InitIfNeeded();

 private:
  struct KVOffsetComparator {
    KVOffsetComparator(const Slice& file_data, const Comparator* ucomp,
                       const Slice& target = Slice())
        : file_data_(file_data), ucomp_(ucomp), target_(target) {}
    bool operator()(const uint32_t first, const uint32_t second) const {
      auto get_user_key = [&](uint32_t offset) {
        if (offset == kInvalidIndex) {
          return target_;
        }
        const char* kv_ptr = &file_data_.data()[offset];
        const uint32_t key_len = DecodeFixed32(kv_ptr);
        const Slice key(kv_ptr + 2 * sizeof(uint32_t), key_len);
        return ExtractUserKey(key);
      };
      return ucomp_->Compare(get_user_key(first), get_user_key(second)) < 0;
    }

   private:
    const Slice file_data_;
    const Comparator* ucomp_;
    const Slice target_;
  };

  const KVOffsetComparator kv_offset_comparator_;
  void PrepareKVAtCurrIdx();
  CuckooTableReader* reader_;
  bool initialized_;
  // Contains offsets to KV pairs in the KV section, in sorted order.
  std::vector<uint32_t> kv_offsets_;
  // We assume that the number of items can be stored in uint32 (4 Billion).
  uint32_t curr_key_idx_;
  Slice curr_value_;
  IterKey curr_key_;
};

CuckooTableIterator::CuckooTableIterator(CuckooTableReader* reader)
    : kv_offset_comparator_(reader->file_data_, reader->ucomp_),
      reader_(reader),
      initialized_(false),
      curr_key_idx_(kInvalidIndex) {
  kv_offsets_.clear();
  curr_value_.clear();
  curr_key_.Clear();
}

void CuckooTableIterator::InitIfNeeded() {
  if (initialized_) {
    return;
  }
  uint32_t num_entries =
      static_cast<uint32_t>(reader_->GetTableProperties()->num_entries);
  kv_offsets_.reserve(num_entries);

  const char* kv_ptr = reader_->file_data_.data() + reader_->kv_data_offset_;
  for (uint32_t i = 0; i < num_entries; ++i) {
    kv_offsets_.push_back(
        static_cast<uint32_t>(kv_ptr - reader_->file_data_.data()));
    uint32_t key_len = DecodeFixed32(kv_ptr);
    uint32_t val_len = DecodeFixed32(kv_ptr + sizeof(uint32_t));
    kv_ptr += 2 * sizeof(uint32_t) + key_len + val_len;
  }
  curr_key_idx_ = kInvalidIndex;
  initialized_ = true;
}

void CuckooTableIterator::SeekToFirst() {
  InitIfNeeded();
  curr_key_idx_ = 0;
  PrepareKVAtCurrIdx();
}

void CuckooTableIterator::SeekToLast() {
  InitIfNeeded();
  curr_key_idx_ = static_cast<uint32_t>(kv_offsets_.size()) - 1;
  PrepareKVAtCurrIdx();
}

void CuckooTableIterator::Seek(const Slice& target) {
  InitIfNeeded();
  const KVOffsetComparator seek_comparator(reader_->file_data_, reader_->ucomp_,
                                           ExtractUserKey(target));
  const auto seek_it = std::lower_bound(kv_offsets_.begin(), kv_offsets_.end(),
                                        kInvalidIndex, seek_comparator);
  curr_key_idx_ =
      static_cast<uint32_t>(std::distance(kv_offsets_.begin(), seek_it));
  PrepareKVAtCurrIdx();
}

void CuckooTableIterator::SeekForPrev(const Slice& /*target*/) {
  // Not supported
  assert(false);
}

bool CuckooTableIterator::Valid() const {
  return curr_key_idx_ < kv_offsets_.size();
}

void CuckooTableIterator::PrepareKVAtCurrIdx() {
  if (!Valid()) {
    curr_value_.clear();
    curr_key_.Clear();
    return;
  }
  const uint32_t offset = kv_offsets_[curr_key_idx_];
  const char* kv_ptr = reader_->file_data_.data() + offset;
  const uint32_t key_len = DecodeFixed32(kv_ptr);
  const uint32_t val_len = DecodeFixed32(kv_ptr + sizeof(uint32_t));
  const Slice key_slice(kv_ptr + 2 * sizeof(uint32_t), key_len);
  const Slice value_slice(kv_ptr + 2 * sizeof(uint32_t) + key_len, val_len);

  curr_key_.SetInternalKey(key_slice, true /* copy */);
  curr_value_ = value_slice;
}

void CuckooTableIterator::Next() {
  if (!Valid()) {
    curr_value_.clear();
    curr_key_.Clear();
    return;
  }
  ++curr_key_idx_;
  PrepareKVAtCurrIdx();
}

void CuckooTableIterator::Prev() {
  if (curr_key_idx_ == 0) {
    curr_key_idx_ = static_cast<uint32_t>(kv_offsets_.size());
  }
  if (!Valid()) {
    curr_value_.clear();
    curr_key_.Clear();
    return;
  }
  --curr_key_idx_;
  PrepareKVAtCurrIdx();
}

Slice CuckooTableIterator::key() const {
  assert(Valid());
  return curr_key_.GetKey();
}

Slice CuckooTableIterator::value() const {
  assert(Valid());
  return curr_value_;
}

InternalIterator* CuckooTableReader::NewIterator(
    const ReadOptions& /*read_options*/,
    const SliceTransform* /* prefix_extractor */, Arena* arena,
    bool /*skip_filters*/, TableReaderCaller /*caller*/,
    size_t /*compaction_readahead_size*/, bool /* allow_unprepared_value */) {
  if (!status().ok()) {
    return NewErrorInternalIterator<Slice>(
        Status::Corruption("CuckooTableReader status is not okay."), arena);
  }
  CuckooTableIterator* iter;
  if (arena == nullptr) {
    iter = new CuckooTableIterator(this);
  } else {
    auto iter_mem = arena->AllocateAligned(sizeof(CuckooTableIterator));
    iter = new (iter_mem) CuckooTableIterator(this);
  }
  return iter;
}

size_t CuckooTableReader::ApproximateMemoryUsage() const { return 0; }

}  // namespace ROCKSDB_NAMESPACE
