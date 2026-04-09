//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "table/cuckoo/cuckoo_table_builder.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <string>
#include <vector>

#include "db/dbformat.h"
#include "file/writable_file_writer.h"
#include "rocksdb/env.h"
#include "rocksdb/table.h"
#include "table/block_based/block_builder.h"
#include "table/cuckoo/cuckoo_table_factory.h"
#include "table/format.h"
#include "table/meta_blocks.h"
#include "util/autovector.h"
#include "util/random.h"
#include "util/string_util.h"

namespace ROCKSDB_NAMESPACE {
const std::string CuckooTablePropertyNames::kNumHashFunc =
    "rocksdb.cuckoo.hash.num";
const std::string CuckooTablePropertyNames::kHashTableSize =
    "rocksdb.cuckoo.hash.size";
const std::string CuckooTablePropertyNames::kValueLength =
    "rocksdb.cuckoo.value.length";
const std::string CuckooTablePropertyNames::kIsLastLevel =
    "rocksdb.cuckoo.file.islastlevel";
const std::string CuckooTablePropertyNames::kCuckooBlockSize =
    "rocksdb.cuckoo.hash.cuckooblocksize";
const std::string CuckooTablePropertyNames::kIdentityAsFirstHash =
    "rocksdb.cuckoo.hash.identityfirst";
const std::string CuckooTablePropertyNames::kUseModuleHash =
    "rocksdb.cuckoo.hash.usemodule";

// Obtained by running echo rocksdb.table.cuckoo | sha1sum
const uint64_t kCuckooTableMagicNumber = 0x926789d0c5f17873ull;

CuckooTableBuilder::CuckooTableBuilder(
    WritableFileWriter* file, double max_hash_table_ratio,
    uint32_t max_num_hash_table, uint32_t max_search_depth,
    const Comparator* user_comparator, uint32_t cuckoo_block_size,
    bool use_module_hash, bool identity_as_first_hash,
    uint64_t (*get_slice_hash)(const Slice&, uint32_t, uint64_t),
    uint32_t column_family_id, const std::string& column_family_name,
    const std::string& db_id, const std::string& db_session_id,
    uint64_t file_number)
    : num_hash_func_(2),
      file_(file),
      max_hash_table_ratio_(max_hash_table_ratio),
      max_num_hash_func_(max_num_hash_table),
      max_search_depth_(max_search_depth),
      cuckoo_block_size_(std::max(1U, cuckoo_block_size)),
      hash_table_size_(use_module_hash ? 0 : 2),
      is_last_level_file_(false),
      has_seen_first_key_(false),
      num_entries_(0),
      num_values_(0),
      ucomp_(user_comparator),
      use_module_hash_(use_module_hash),
      identity_as_first_hash_(identity_as_first_hash),
      get_slice_hash_(get_slice_hash),
      closed_(false) {
  // Data is in a huge block.
  properties_.num_data_blocks = 1;
  properties_.index_size = 0;
  properties_.filter_size = 0;
  properties_.column_family_id = column_family_id;
  properties_.column_family_name = column_family_name;
  properties_.db_id = db_id;
  properties_.db_session_id = db_session_id;
  properties_.orig_file_number = file_number;
  status_.PermitUncheckedError();
  io_status_.PermitUncheckedError();
}

void CuckooTableBuilder::Add(const Slice& key, const Slice& value) {
  if (num_entries_ >= kMaxVectorIdx - 1) {
    status_ = Status::NotSupported("Number of keys in a file must be < 2^32-1");
    return;
  }
  ParsedInternalKey ikey;
  Status pik_status =
      ParseInternalKey(key, &ikey, false /* log_err_key */);  // TODO
  if (!pik_status.ok()) {
    status_ = Status::Corruption("Unable to parse key into internal key. ",
                                 pik_status.getState());
    return;
  }
  if (ikey.type != kTypeDeletion && ikey.type != kTypeValue) {
    status_ = Status::NotSupported("Unsupported key type " +
                                   std::to_string(ikey.type));
    return;
  }

  // Determine if we can ignore the sequence number and value type from
  // internal keys by looking at sequence number from first key. We assume
  // that if first key has a zero sequence number, then all the remaining
  // keys will have zero seq. no.
  if (!has_seen_first_key_) {
    is_last_level_file_ = ikey.sequence == 0;
    has_seen_first_key_ = true;
  }

  const Slice& key_slice = key;
  const Slice& value_slice = ikey.type == kTypeValue ? value : Slice();
  kv_offsets_.emplace_back(kvs_.size(), key_slice.size(), value_slice.size());
  kvs_.append(key_slice.data(), key_slice.size());
  kvs_.append(value_slice.data(), value_slice.size());
  ++num_entries_;
  if (ikey.type == kTypeValue) ++num_values_;

  // In order to fill the empty buckets in the hash table, we identify a
  // key which is not used so far (unused_user_key). We determine this by
  // maintaining smallest and largest keys inserted so far in bytewise order
  // and use them to find a key outside this range in Finish() operation.
  // Note that this strategy is independent of user comparator used here.
  if (!use_module_hash_) {
    if (hash_table_size_ < num_entries_ / max_hash_table_ratio_) {
      hash_table_size_ *= 2;
    }
  }
}

Slice CuckooTableBuilder::GetKey(uint64_t idx) const {
  assert(closed_);
  const auto& [kv_idx, key_len, _] = kv_offsets_[idx];
  return Slice{&kvs_[kv_idx], key_len};
}

Slice CuckooTableBuilder::GetUserKey(uint64_t idx) const {
  assert(closed_);
  return ExtractUserKey(GetKey(idx));
}

Slice CuckooTableBuilder::GetValue(uint64_t idx) const {
  assert(closed_);
  const auto& [kv_idx, key_len, val_len] = kv_offsets_[idx];
  return Slice{&kvs_[kv_idx+key_len], val_len};
}

Status CuckooTableBuilder::MakeHashTable(std::vector<CuckooBucket>* buckets) {
  buckets->resize(
      static_cast<size_t>(hash_table_size_ + cuckoo_block_size_ - 1));
  uint32_t make_space_for_key_call_id = 0;
  for (uint32_t vector_idx = 0; vector_idx < num_entries_; vector_idx++) {
    uint64_t bucket_id = 0;
    bool bucket_found = false;
    autovector<uint64_t> hash_vals;
    Slice user_key = GetUserKey(vector_idx);
    for (uint32_t hash_cnt = 0; hash_cnt < num_hash_func_ && !bucket_found;
         ++hash_cnt) {
      uint64_t hash_val =
          CuckooHash(user_key, hash_cnt, use_module_hash_, hash_table_size_,
                     identity_as_first_hash_, get_slice_hash_);
      // If there is a collision, check next cuckoo_block_size_ locations for
      // empty locations. While checking, if we reach end of the hash table,
      // stop searching and proceed for next hash function.
      for (uint32_t block_idx = 0; block_idx < cuckoo_block_size_;
           ++block_idx, ++hash_val) {
        if ((*buckets)[static_cast<size_t>(hash_val)].vector_idx ==
            kMaxVectorIdx) {
          bucket_id = hash_val;
          bucket_found = true;
          break;
        } else {
          if (ucomp_->Compare(
                  user_key, GetUserKey((*buckets)[static_cast<size_t>(hash_val)]
                                           .vector_idx)) == 0) {
            return Status::NotSupported("Same key is being inserted again.");
          }
          hash_vals.push_back(hash_val);
        }
      }
    }
    while (!bucket_found &&
           !MakeSpaceForKey(hash_vals, ++make_space_for_key_call_id, buckets,
                            &bucket_id)) {
      // Rehash by increashing number of hash tables.
      if (num_hash_func_ >= max_num_hash_func_) {
        return Status::NotSupported("Too many collisions. Unable to hash.");
      }
      // We don't really need to rehash the entire table because old hashes are
      // still valid and we only increased the number of hash functions.
      uint64_t hash_val = CuckooHash(user_key, num_hash_func_, use_module_hash_,
                                     hash_table_size_, identity_as_first_hash_,
                                     get_slice_hash_);
      ++num_hash_func_;
      for (uint32_t block_idx = 0; block_idx < cuckoo_block_size_;
           ++block_idx, ++hash_val) {
        if ((*buckets)[static_cast<size_t>(hash_val)].vector_idx ==
            kMaxVectorIdx) {
          bucket_found = true;
          bucket_id = hash_val;
          break;
        } else {
          hash_vals.push_back(hash_val);
        }
      }
    }
    (*buckets)[static_cast<size_t>(bucket_id)].vector_idx = vector_idx;
  }
  return Status::OK();
}

Status CuckooTableBuilder::Finish() {
  assert(!closed_);
  closed_ = true;
  std::vector<CuckooBucket> buckets;
  if (num_entries_ > 0) {
    // Calculate the real hash size if module hash is enabled.
    if (use_module_hash_) {
      hash_table_size_ =
          static_cast<uint64_t>(num_entries_ / max_hash_table_ratio_);
    }
    status_ = MakeHashTable(&buckets);
    if (!status_.ok()) {
      return status_;
    }
  }

  properties_.num_entries = num_entries_;
  properties_.num_deletions = num_entries_ - num_values_;

  // Fingerprint and Address
  constexpr uint64_t bucket_size = CuckooTableBucket::size();
  char unused_bucket[bucket_size], encoded_bucket[bucket_size];
  CuckooTableBucket().encode(unused_bucket);
  uint32_t base_address = static_cast<uint32_t>(buckets.size() * bucket_size);
  // Write the table.
  properties_.raw_value_size = properties_.raw_key_size = 0;
  const IOOptions opts;
  for (auto& bucket : buckets) {
    if (bucket.vector_idx == kMaxVectorIdx) {
      io_status_ = file_->Append(opts, Slice(unused_bucket, bucket_size));
    } else {
      const auto& [kv_idx, key_len, val_len] = kv_offsets_[bucket.vector_idx];
      CuckooTableBucket {
        CuckooFingerPrint(GetUserKey(bucket.vector_idx), num_hash_func_),
        static_cast<uint32_t>(base_address + kv_idx + bucket.vector_idx * sizeof(uint32_t) * 2)
      }.encode(encoded_bucket);
      io_status_ = file_->Append(opts, Slice(encoded_bucket, bucket_size));
      properties_.raw_key_size += key_len;
      properties_.raw_value_size += val_len;
    }
    if (!io_status_.ok()) {
      status_ = io_status_;
      return status_;
    }
  }

  for (const auto& [kv_idx, key_len, val_len] : kv_offsets_) {
    char buf[sizeof(uint32_t)];
    EncodeFixed32(buf, key_len);
    file_->Append(opts, Slice(buf, sizeof(uint32_t)));
    EncodeFixed32(buf, val_len);
    file_->Append(opts, Slice(buf, sizeof(uint32_t)));
    file_->Append(opts, Slice(&kvs_[kv_idx], key_len));
    if (val_len)
      file_->Append(opts, Slice(&kvs_[kv_idx + key_len], val_len));
  }

  // TODO: Fix Endianness

  uint64_t offset = base_address + kvs_.size() + num_entries_ * sizeof(uint32_t) * 2;
  properties_.data_size = buckets.size() * bucket_size + kvs_.size() + num_entries_ * sizeof(uint32_t) * 2;
  properties_.user_collected_properties[CuckooTablePropertyNames::kNumHashFunc]
      .assign(reinterpret_cast<char*>(&num_hash_func_), sizeof(num_hash_func_));

  properties_
      .user_collected_properties[CuckooTablePropertyNames::kHashTableSize]
      .assign(reinterpret_cast<const char*>(&hash_table_size_),
              sizeof(hash_table_size_));
  properties_.user_collected_properties[CuckooTablePropertyNames::kIsLastLevel]
      .assign(reinterpret_cast<const char*>(&is_last_level_file_),
              sizeof(is_last_level_file_));
  properties_
      .user_collected_properties[CuckooTablePropertyNames::kCuckooBlockSize]
      .assign(reinterpret_cast<const char*>(&cuckoo_block_size_),
              sizeof(cuckoo_block_size_));
  properties_
      .user_collected_properties[CuckooTablePropertyNames::kIdentityAsFirstHash]
      .assign(reinterpret_cast<const char*>(&identity_as_first_hash_),
              sizeof(identity_as_first_hash_));
  properties_
      .user_collected_properties[CuckooTablePropertyNames::kUseModuleHash]
      .assign(reinterpret_cast<const char*>(&use_module_hash_),
              sizeof(use_module_hash_));

  // Write meta blocks.
  MetaIndexBuilder meta_index_builder;
  PropertyBlockBuilder property_block_builder;

  property_block_builder.AddTableProperty(properties_);
  property_block_builder.Add(properties_.user_collected_properties);
  Slice property_block = property_block_builder.Finish();
  BlockHandle property_block_handle;
  property_block_handle.set_offset(offset);
  property_block_handle.set_size(property_block.size());
  io_status_ = file_->Append(opts, property_block);
  offset += property_block.size();
  if (!io_status_.ok()) {
    status_ = io_status_;
    return status_;
  }

  meta_index_builder.Add(kPropertiesBlockName, property_block_handle);
  Slice meta_index_block = meta_index_builder.Finish();

  BlockHandle meta_index_block_handle;
  meta_index_block_handle.set_offset(offset);
  meta_index_block_handle.set_size(meta_index_block.size());
  io_status_ = file_->Append(opts, meta_index_block);
  if (!io_status_.ok()) {
    status_ = io_status_;
    return status_;
  }

  FooterBuilder footer;
  Status s = footer.Build(kCuckooTableMagicNumber, /* format_version */ 1,
                          offset, kNoChecksum, meta_index_block_handle);
  if (!s.ok()) {
    status_ = s;
    return status_;
  }
  io_status_ = file_->Append(opts, footer.GetSlice());
  status_ = io_status_;
  return status_;
}

void CuckooTableBuilder::Abandon() {
  assert(!closed_);
  closed_ = true;
}

uint64_t CuckooTableBuilder::NumEntries() const { return num_entries_; }

uint64_t CuckooTableBuilder::FileSize() const {
  if (closed_) {
    return file_->GetFileSize();
  } else if (num_entries_ == 0) {
    return 0;
  }

  uint64_t expected_hash_table_size = hash_table_size_;
  if (!use_module_hash_ &&
      expected_hash_table_size < (num_entries_ + 1) / max_hash_table_ratio_) {
    expected_hash_table_size *= 2;
  }

  return (expected_hash_table_size + cuckoo_block_size_ - 1) *
             CuckooTableBucket::size() +
         kvs_.size() + num_entries_ * sizeof(uint32_t) * 2;
}

// This method is invoked when there is no place to insert the target key.
// It searches for a set of elements that can be moved to accommodate target
// key. The search is a BFS graph traversal with first level (hash_vals)
// being all the buckets target key could go to.
// Then, from each node (curr_node), we find all the buckets that curr_node
// could go to. They form the children of curr_node in the tree.
// We continue the traversal until we find an empty bucket, in which case, we
// move all elements along the path from first level to this empty bucket, to
// make space for target key which is inserted at first level (*bucket_id).
// If tree depth exceedes max depth, we return false indicating failure.
bool CuckooTableBuilder::MakeSpaceForKey(
    const autovector<uint64_t>& hash_vals,
    const uint32_t make_space_for_key_call_id,
    std::vector<CuckooBucket>* buckets, uint64_t* bucket_id) {
  struct CuckooNode {
    uint64_t bucket_id;
    uint32_t depth;
    uint32_t parent_pos;
    CuckooNode(uint64_t _bucket_id, uint32_t _depth, int _parent_pos)
        : bucket_id(_bucket_id), depth(_depth), parent_pos(_parent_pos) {}
  };
  // This is BFS search tree that is stored simply as a vector.
  // Each node stores the index of parent node in the vector.
  std::vector<CuckooNode> tree;
  // We want to identify already visited buckets in the current method call so
  // that we don't add same buckets again for exploration in the tree.
  // We do this by maintaining a count of current method call in
  // make_space_for_key_call_id, which acts as a unique id for this invocation
  // of the method. We store this number into the nodes that we explore in
  // current method call.
  // It is unlikely for the increment operation to overflow because the maximum
  // no. of times this will be called is <= max_num_hash_func_ + num_entries_.
  for (uint32_t hash_cnt = 0; hash_cnt < num_hash_func_; ++hash_cnt) {
    uint64_t bid = hash_vals[hash_cnt];
    (*buckets)[static_cast<size_t>(bid)].make_space_for_key_call_id =
        make_space_for_key_call_id;
    tree.emplace_back(bid, 0, 0);
  }
  bool null_found = false;
  uint32_t curr_pos = 0;
  while (!null_found && curr_pos < tree.size()) {
    CuckooNode& curr_node = tree[curr_pos];
    uint32_t curr_depth = curr_node.depth;
    if (curr_depth >= max_search_depth_) {
      break;
    }
    CuckooBucket& curr_bucket =
        (*buckets)[static_cast<size_t>(curr_node.bucket_id)];
    for (uint32_t hash_cnt = 0; hash_cnt < num_hash_func_ && !null_found;
         ++hash_cnt) {
      uint64_t child_bucket_id = CuckooHash(
          GetUserKey(curr_bucket.vector_idx), hash_cnt, use_module_hash_,
          hash_table_size_, identity_as_first_hash_, get_slice_hash_);
      // Iterate inside Cuckoo Block.
      for (uint32_t block_idx = 0; block_idx < cuckoo_block_size_;
           ++block_idx, ++child_bucket_id) {
        if ((*buckets)[static_cast<size_t>(child_bucket_id)]
                .make_space_for_key_call_id == make_space_for_key_call_id) {
          continue;
        }
        (*buckets)[static_cast<size_t>(child_bucket_id)]
            .make_space_for_key_call_id = make_space_for_key_call_id;
        tree.emplace_back(child_bucket_id, curr_depth + 1, curr_pos);
        if ((*buckets)[static_cast<size_t>(child_bucket_id)].vector_idx ==
            kMaxVectorIdx) {
          null_found = true;
          break;
        }
      }
    }
    ++curr_pos;
  }

  if (null_found) {
    // There is an empty node in tree.back(). Now, traverse the path from this
    // empty node to top of the tree and at every node in the path, replace
    // child with the parent. Stop when first level is reached in the tree
    // (happens when 0 <= bucket_to_replace_pos < num_hash_func_) and return
    // this location in first level for target key to be inserted.
    uint32_t bucket_to_replace_pos = static_cast<uint32_t>(tree.size()) - 1;
    while (bucket_to_replace_pos >= num_hash_func_) {
      CuckooNode& curr_node = tree[bucket_to_replace_pos];
      (*buckets)[static_cast<size_t>(curr_node.bucket_id)] =
          (*buckets)[static_cast<size_t>(tree[curr_node.parent_pos].bucket_id)];
      bucket_to_replace_pos = curr_node.parent_pos;
    }
    *bucket_id = tree[bucket_to_replace_pos].bucket_id;
  }
  return null_found;
}

std::string CuckooTableBuilder::GetFileChecksum() const {
  if (file_ != nullptr) {
    return file_->GetFileChecksum();
  } else {
    return kUnknownFileChecksum;
  }
}

const char* CuckooTableBuilder::GetFileChecksumFuncName() const {
  if (file_ != nullptr) {
    return file_->GetFileChecksumFuncName();
  } else {
    return kUnknownFileChecksumFuncName;
  }
}

}  // namespace ROCKSDB_NAMESPACE
