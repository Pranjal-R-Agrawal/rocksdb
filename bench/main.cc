#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "db_stress_tool/db_stress_common.h"
#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"

using namespace rocksdb;

static constexpr uint64_t kSeed = 0xC0FFEE123456789ULL;
static std::mt19937_64 g_rng(kSeed);

class ZipfianGenerator {
  uint64_t n;
  mutable std::uniform_real_distribution<double> dist_;

 public:
  ZipfianGenerator(uint64_t n, uint64_t /*seed*/, double theta = 0.99)
      : n(n), dist_(0.0, 1.0) {
    ROCKSDB_NAMESPACE::InitializeHotKeyGenerator(theta);
  }

  uint64_t next() const {
    return ROCKSDB_NAMESPACE::GetOneHotKeyID(dist_(g_rng), n);
  }
};

class UniformSizeSampler {
  mutable std::uniform_int_distribution<uint64_t> dist_;

 public:
  UniformSizeSampler(uint64_t min_size, uint64_t max_size, uint64_t /*seed*/)
      : dist_(min_size, max_size) {}

  uint64_t next() const { return dist_(g_rng); }
};

class ZipfSizeSampler {
 public:
  ZipfSizeSampler(uint64_t min_size, uint64_t max_size, uint64_t /*seed*/)
      : min_size_(min_size),
        range_(max_size - min_size + 1),
        zipf_(range_, 0) {}

  uint64_t next() const { return min_size_ + (zipf_.next() % range_); }

 private:
  uint64_t min_size_;
  uint64_t range_;
  ZipfianGenerator zipf_;
};

struct BenchData {
  std::vector<std::string> keys;
  std::vector<std::string> values;
};

BenchData GenerateData(uint64_t num_keys, const std::string& data_dist) {
  BenchData data;
  data.keys.reserve(num_keys);
  data.values.reserve(num_keys);

  const ZipfSizeSampler zipf_key_pad(0, 64, kSeed ^ 0x11111111ULL);
  const UniformSizeSampler uniform_key_pad(0, 64, kSeed ^ 0x11111111ULL);

  const ZipfSizeSampler zipf_value_size(100, 1000, kSeed ^ 0x22222222ULL);
  const UniformSizeSampler uniform_value_size(100, 1000, kSeed ^ 0x22222222ULL);

  for (uint64_t i = 0; i < num_keys; ++i) {
    char buf[32];
    snprintf(buf, sizeof(buf), "key_%010llu", (unsigned long long)i);

    uint64_t suffix_len = 0;
    uint64_t value_len = 0;

    if (data_dist == "zipf") {
      suffix_len = zipf_key_pad.next();
      value_len = zipf_value_size.next();
    } else {
      suffix_len = uniform_key_pad.next();
      value_len = uniform_value_size.next();
    }

    data.keys.push_back(std::string(buf) + std::string(suffix_len, 'a'));
    data.values.emplace_back(value_len, 'v');
  }

  return data;
}

struct BenchmarkConfig {
  bool compact = false;
  std::string mode_name;
};

Options BuildOptions(bool is_cuckoo, const BenchmarkConfig& cfg) {
  Options options;
  options.create_if_missing = true;
  options.error_if_exists = false;
  options.paranoid_checks = true;
  options.allow_mmap_reads = true;
  options.allow_mmap_writes = false;
  options.use_direct_reads = false;
  options.use_direct_io_for_flush_and_compaction = false;

  options.write_buffer_size = 1073741824ULL;
  options.max_write_buffer_number = 2;
  options.level0_slowdown_writes_trigger = 16;
  options.level0_stop_writes_trigger = 24;
  options.delete_obsolete_files_period_micros = 300000000ULL;
  options.compression = kNoCompression;
  options.compression_opts = CompressionOptions();
  options.max_background_compactions = 20;
  options.max_bytes_for_level_base = 4294967296ULL;
  options.target_file_size_base = 201327616ULL;
  options.level0_file_num_compaction_trigger = 10;
  options.num_levels = 7;
  options.compaction_style =
      cfg.compact ? kCompactionStyleUniversal : kCompactionStyleNone;
  options.max_background_flushes = 1;
  options.level_compaction_dynamic_level_bytes = false;

  if (is_cuckoo) {
    constexpr CuckooTableOptions cuckoo_opts;
    options.table_factory.reset(NewCuckooTableFactory(cuckoo_opts));
  } else {
    BlockBasedTableOptions block_opts;
    block_opts.filter_policy.reset(NewBloomFilterPolicy(10));
    block_opts.no_block_cache = true;
    options.table_factory.reset(NewBlockBasedTableFactory(block_opts));
  }

  return options;
}

void RunLoadAndMaybeCompact(DB* db, uint64_t num_keys,
                            const std::string& bench_mode,
                            const std::vector<std::string>& keys,
                            const std::vector<std::string>& values,
                            bool shuffle_keys) {
  WriteOptions wo;
  wo.disableWAL = true;

  std::vector<uint64_t> insert_order(num_keys);
  std::iota(insert_order.begin(), insert_order.end(), 0);
  if (shuffle_keys) {
    std::mt19937_64 shuffle_rng(kSeed ^ 0x66666666ULL);
    std::ranges::shuffle(insert_order, shuffle_rng);
  }

  const uint64_t flush_chunk_size = std::max<uint64_t>(1, num_keys / 8);

  for (uint64_t base = 0; base < num_keys; base += flush_chunk_size) {
    constexpr uint64_t batch_size = 1000;
    uint64_t end = std::min<uint64_t>(num_keys, base + flush_chunk_size);

    for (uint64_t batch_base = base; batch_base < end;
         batch_base += batch_size) {
      const uint64_t batch_end =
          std::min<uint64_t>(end, batch_base + batch_size);
      WriteBatch wb;
      for (uint64_t i = batch_base; i < batch_end; ++i) {
        const uint64_t idx = insert_order[i];
        wb.Put(keys[idx], values[idx]);
      }
      db->Write(wo, &wb);
    }

    db->Flush(FlushOptions());
  }

  if (bench_mode == "compact") {
    CompactRangeOptions cro;
    cro.change_level = true;
    db->CompactRange(cro, nullptr, nullptr);
  }
}

void run_bench(uint64_t num_keys, bool is_cuckoo, const std::string& bench_mode,
               const std::string& data_dist, const std::string& access_dist,
               double duration_sec, double miss_ratio, bool shuffle_keys) {
  BenchmarkConfig cfg;
  cfg.compact = (bench_mode == "compact");
  cfg.mode_name = bench_mode;

  Options options = BuildOptions(is_cuckoo, cfg);

  std::string path = "/tmp/rocks_adv_" +
                     std::string(is_cuckoo ? "cuckoo" : "block") + "_" +
                     bench_mode;
  DestroyDB(path, options);

  std::unique_ptr<DB> db;
  if (Status s = DB::Open(options, path, &db); !s.ok()) {
    std::cerr << "DB::Open failed: " << s.ToString() << std::endl;
    return;
  }

  const auto [keys, values] = GenerateData(num_keys, data_dist);

  const auto t0 = std::chrono::high_resolution_clock::now();
  RunLoadAndMaybeCompact(&*db, num_keys, bench_mode, keys, values,
                         shuffle_keys);
  const auto t1 = std::chrono::high_resolution_clock::now();
  double build_time_sec = std::chrono::duration<double>(t1 - t0).count();

  db->WaitForCompact(WaitForCompactOptions());

  ReadOptions ro;
  ro.verify_checksums = false;

  PinnableSlice val;
  std::mt19937_64 access_rng(kSeed ^ 0x33333333ULL);
  std::uniform_int_distribution<uint64_t> uniform_key_dist(0, num_keys - 1);
  ZipfianGenerator zipf(num_keys, kSeed ^ 0x44444444ULL);

  uint64_t miss_threshold = 0;
  if (miss_ratio > 0.0) {
    miss_threshold = static_cast<uint64_t>(
                         miss_ratio * static_cast<double>(access_rng.max() -
                                                          access_rng.min())) +
                     access_rng.min();
  }
  std::string dummy_miss_key = "key_MISS_0000000000";

  auto run_read_benchmark = [&](auto next_index_fn) -> double {
    for (int i = 0; i < 15000; i++) {
      val.Reset();
      if (miss_ratio > 0.0 && access_rng() < miss_threshold) {
        db->Get(ro, db->DefaultColumnFamily(), dummy_miss_key, &val);
      } else {
        db->Get(ro, db->DefaultColumnFamily(), keys[next_index_fn()], &val);
      }
    }

    const auto start = std::chrono::high_resolution_clock::now();
    uint64_t total_iters = 0;

    while (true) {
      for (int i = 0; i < 7500; i++) {
        val.Reset();
        if (miss_ratio > 0.0 && access_rng() < miss_threshold) {
          db->Get(ro, db->DefaultColumnFamily(), dummy_miss_key, &val);
        } else {
          db->Get(ro, db->DefaultColumnFamily(), keys[next_index_fn()], &val);
        }
      }
      total_iters += 7500;

      const auto now = std::chrono::high_resolution_clock::now();
      if (std::chrono::duration<double>(now - start).count() >= duration_sec) {
        const double total_us =
            std::chrono::duration<double, std::micro>(now - start).count();
        return total_us / total_iters;
      }
    }
  };

  auto run_with_median = [&](auto next_index_fn) -> double {
    std::vector<double> results;
    for (int i = 0; i < 3; i++) {
      results.push_back(run_read_benchmark(next_index_fn));
    }
    std::sort(results.begin(), results.end());
    return results[1];  // Return the median of 3 runs
  };

  double avg = 0.0;
  if (access_dist == "zipf") {
    avg = run_with_median([&]() { return zipf.next(); });
  } else {
    avg = run_with_median([&]() { return uniform_key_dist(access_rng); });
  }

  uint64_t size_bytes = 0;
  db->GetIntProperty("rocksdb.total-sst-files-size", &size_bytes);

  std::cout << size_bytes / 1024 << "," << avg << "," << build_time_sec
            << std::endl;
}

int main(const int argc, const char** argv) {
  if (argc < 9) return 1;

  const uint64_t n = std::stoull(argv[1]);
  const bool cuckoo = std::string(argv[2]) == "cuckoo";
  const std::string bench_mode = argv[3];
  const std::string data_dist = argv[4];
  const std::string access_dist = argv[5];
  const double duration_sec = std::stod(argv[6]);
  const double miss_ratio = std::stod(argv[7]);
  const bool shuffle_keys =
      std::string(argv[8]) == "true" || std::string(argv[8]) == "1";

  run_bench(n, cuckoo, bench_mode, data_dist, access_dist, duration_sec,
            miss_ratio, shuffle_keys);
  return 0;
}