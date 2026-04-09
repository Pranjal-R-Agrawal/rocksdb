#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "db_stress_tool/db_stress_common.h"

using namespace rocksdb;

static constexpr uint64_t kSeed = 0xC0FFEE123456789ULL;
static std::mt19937_64 g_rng(kSeed);

class ZipfianGenerator {
    uint64_t n;

public:
    ZipfianGenerator(uint64_t n, uint64_t /*seed*/, double theta = 0.99)
        : n(n) {
        ROCKSDB_NAMESPACE::InitializeHotKeyGenerator(theta);
    }

    uint64_t next() const {
        const double float_rand = (static_cast<double>(g_rng() % n)) / n;
        return ROCKSDB_NAMESPACE::GetOneHotKeyID(float_rand, n);
    }
};

uint64_t SampleUniformSize(uint64_t min_size, uint64_t max_size) {
    std::uniform_int_distribution<uint64_t> dist(min_size, max_size);
    return dist(g_rng);
}

class ZipfSizeSampler {
public:
    ZipfSizeSampler(uint64_t min_size, uint64_t max_size, uint64_t /*seed*/)
        : min_size_(min_size),
          range_(max_size - min_size + 1),
          zipf_(range_, 0) {}

    uint64_t next() {
        return min_size_ + (zipf_.next() % range_);
    }

private:
    uint64_t min_size_;
    uint64_t range_;
    ZipfianGenerator zipf_;
};

class NormalSizeSampler {
public:
    NormalSizeSampler(uint64_t min_size, uint64_t max_size, uint64_t seed)
        : min_size_(min_size),
          max_size_(max_size),
          mean_((min_size + max_size) / 2.0),
          stddev_((max_size - min_size) / 6.0),
          gen_(seed),
          dist_(mean_, stddev_) {}

    uint64_t next() {
        double v = dist_(gen_);
        if (v < static_cast<double>(min_size_)) return min_size_;
        if (v > static_cast<double>(max_size_)) return max_size_;
        return static_cast<uint64_t>(v);
    }

private:
    uint64_t min_size_;
    uint64_t max_size_;
    double mean_;
    double stddev_;
    std::mt19937_64 gen_;
    std::normal_distribution<double> dist_;
};

struct BenchData {
    std::vector<std::string> keys;
    std::vector<std::string> values;
};

BenchData GenerateData(uint64_t num_keys, const std::string& data_dist) {
    BenchData data;
    data.keys.reserve(num_keys);
    data.values.reserve(num_keys);

    ZipfSizeSampler zipf_key_pad(0, 64, kSeed ^ 0x11111111ULL);
    ZipfSizeSampler zipf_value_size(100, 1000, kSeed ^ 0x22222222ULL);
    NormalSizeSampler normal_key_pad(0, 64, kSeed ^ 0x33333333ULL);
    NormalSizeSampler normal_value_size(100, 1000, kSeed ^ 0x44444444ULL);

    for (uint64_t i = 0; i < num_keys; ++i) {
        char buf[32];
        snprintf(buf, sizeof(buf), "key_%010llu", (unsigned long long)i);

        uint64_t suffix_len = 0;
        uint64_t value_len = 0;

        if (data_dist == "zipf") {
            suffix_len = zipf_key_pad.next();
            value_len = zipf_value_size.next();
        } else if (data_dist == "normal") {
            suffix_len = normal_key_pad.next();
            value_len = normal_value_size.next();
        } else {
            suffix_len = SampleUniformSize(0, 64);
            value_len = SampleUniformSize(100, 1000);
        }

        data.keys.push_back(std::string(buf) + std::string(suffix_len, 'a'));
        data.values.push_back(std::string(value_len, 'v'));
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
    options.compaction_style = cfg.compact? kCompactionStyleUniversal : kCompactionStyleNone;
    options.max_background_flushes = 1;
    options.level_compaction_dynamic_level_bytes = false;

    if (is_cuckoo) {
        CuckooTableOptions cuckoo_opts;
        options.table_factory.reset(NewCuckooTableFactory(cuckoo_opts));
    } else {
        BlockBasedTableOptions bbto;
        bbto.filter_policy.reset(NewBloomFilterPolicy(10));
        options.table_factory.reset(NewBlockBasedTableFactory(bbto));
    }

    return options;
}

void RunLoadAndMaybeCompact(DB* db, uint64_t num_keys, const std::string& bench_mode,
                            const std::vector<std::string>& keys,
                            const std::vector<std::string>& values) {
    WriteOptions wo;
    wo.disableWAL = true;

    const uint64_t batch_size = 1000;
    const uint64_t flush_chunk_size = std::max<uint64_t>(1, num_keys / 8);

    for (uint64_t base = 0; base < num_keys; base += flush_chunk_size) {
        uint64_t end = std::min<uint64_t>(num_keys, base + flush_chunk_size);

        for (uint64_t batch_base = base; batch_base < end; batch_base += batch_size) {
            uint64_t batch_end = std::min<uint64_t>(end, batch_base + batch_size);
            WriteBatch wb;
            for (uint64_t i = batch_base; i < batch_end; ++i) {
                wb.Put(keys[i], values[i]);
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
               const std::string& data_dist, const std::string& access_dist) {
    BenchmarkConfig cfg;
    cfg.compact = (bench_mode == "compact");
    cfg.mode_name = bench_mode;

    Options options = BuildOptions(is_cuckoo, cfg);

    std::string path = "/tmp/rocks_adv_" + std::string(is_cuckoo ? "cuckoo" : "block") +
                       "_" + bench_mode;
    DestroyDB(path, options);

    std::unique_ptr<DB> db;
    if (Status s = DB::Open(options, path, &db); !s.ok()) {
        std::cerr << "DB::Open failed: " << s.ToString() << std::endl;
        return;
    }

    BenchData data = GenerateData(num_keys, data_dist);

    const auto t0 = std::chrono::high_resolution_clock::now();
    RunLoadAndMaybeCompact(&*db, num_keys, bench_mode, data.keys, data.values);
    const auto t1 = std::chrono::high_resolution_clock::now();
    double build_time_sec = std::chrono::duration<double>(t1 - t0).count();

    ReadOptions ro;
    ro.verify_checksums = true;
    ro.fill_cache = false;

    std::string val;
    std::mt19937_64 access_rng(kSeed ^ 0x33333333ULL);
    std::uniform_int_distribution<uint64_t> uniform_key_dist(0, num_keys - 1);
    ZipfianGenerator zipf(num_keys, kSeed ^ 0x44444444ULL);
    NormalSizeSampler normal_access(0, num_keys - 1, kSeed ^ 0x55555555ULL);

    auto next_index = [&]() -> uint64_t {
        if (access_dist == "zipf") {
            return zipf.next();
        } else if (access_dist == "normal") {
            return normal_access.next();
        } else {
            return uniform_key_dist(access_rng);
        }
    };

    for (int i = 0; i < 15000; i++) {
        uint64_t idx = next_index();
        db->Get(ro, data.keys[idx], &val);
    }

    std::vector<double> latencies;
    const int iterations = 40000;

    for (int i = 0; i < iterations; i++) {
        uint64_t k_idx = next_index();
        auto start = std::chrono::high_resolution_clock::now();
        db->Get(ro, data.keys[k_idx], &val);
        auto end = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(latencies.begin(), latencies.end());

    uint64_t size_bytes = 0;
    db->GetIntProperty("rocksdb.total-sst-files-size", &size_bytes);

    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double p99 = latencies[static_cast<int>(0.99 * latencies.size())];

    std::cout << size_bytes / 1024 << "," << avg << "," << p99 << "," << build_time_sec << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 6) return 1;

    uint64_t n = std::stoull(argv[1]);
    bool cuckoo = std::string(argv[2]) == "cuckoo";
    std::string bench_mode = argv[3];
    std::string data_dist = argv[4];
    std::string access_dist = argv[5];

    run_bench(n, cuckoo, bench_mode, data_dist, access_dist);
    return 0;
}