// include/strata/ngram/hy4_engram_reader.hpp - 4-Module Native 8-Bit Engram (PLE) Reader for Hy4-Flash
// Reads Hy4-Flash-Engram-Q8.strata (67.58 GB / 62.94 GiB, 40 rows per 4 KiB NVMe page, zero page straddling)
// backed by a 64 MiB static Section 2 seed + 836 MiB dynamic host-RAM session LRU (900 MiB total).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::ngram {

constexpr uint32_t HY4_ENGRAM_MODULES = 4;
constexpr uint32_t HY4_ENGRAM_HEADS_PER_MOD = 16;  // 8 bigram + 8 trigram
constexpr uint32_t HY4_ENGRAM_ROW_DIM = 96;
constexpr uint32_t HY4_ENGRAM_ROW_BYTES = 100;     // 4 B float32 scale + 96 B F8_E4M3
constexpr uint32_t HY4_ENGRAM_ROWS_PER_PAGE = 40;  // floor(4096 / 100) = 40 whole rows/page
constexpr uint32_t HY4_ENGRAM_PAGE_BYTES = 4096;
constexpr float    HY4_ENGRAM_EPS = 1.1920929e-7f; // FLT_EPSILON exact parity with tencent/Hy4-flash

struct Hy4EngramStats {
    uint64_t total_lookups = 0;
    uint64_t seed_hits = 0;     // served from 64 MiB Section 2 static hot seed
    uint64_t lru_hits = 0;      // served from 836 MiB dynamic host-RAM session LRU
    uint64_t ssd_page_reads = 0;
    uint64_t ssd_bytes_read = 0;
    double   hit_rate() const noexcept {
        return total_lookups ? static_cast<double>(seed_hits + lru_hits) / static_cast<double>(total_lookups) : 0.0;
    }
};

class Hy4EngramReader {
public:
    Hy4EngramReader() = default;
    ~Hy4EngramReader() { close(); }
    Hy4EngramReader(const Hy4EngramReader&) = delete;
    Hy4EngramReader& operator=(const Hy4EngramReader&) = delete;

    // Open Hy4-Flash-Engram-Q8.strata and optionally bind the 64 MiB Section 2 static seed blob.
    // lru_capacity_rows default ~6,000,000 (~836 MiB dynamic session LRU).
    bool open(
        const std::string& shard_path,
        const uint8_t* section2_seed,
        size_t section2_bytes,
        size_t lru_capacity_rows,
        std::string& err);

    void close();
    bool is_open() const noexcept { return fp_ != nullptr; }

    // Fetch n global row IDs into out_f32 [n * 96] (dequantized from F8_E4M3 * f32_scale).
    // Deduplicates 4 KiB NVMe page reads across all 4 modules in a single batch!
    bool fetch_rows_f32(const uint64_t* global_row_ids, size_t n, float* out_f32, std::string& err);

    const Hy4EngramStats& stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_ = {}; }

    // Exact FP8 E4M3 -> float32 decode helper (matches PyTorch torch.float8_e4m3fn)
    static float fp8_e4m3_to_f32(uint8_t b) noexcept;

private:
    bool lookup_seed(uint64_t row_id, uint8_t out_raw[HY4_ENGRAM_ROW_BYTES]) const noexcept;
    void decode_raw_row(const uint8_t raw[HY4_ENGRAM_ROW_BYTES], float* dst96) const noexcept;

    std::FILE* fp_ = nullptr;
    uint64_t sec4_offset_ = 0;
    uint8_t shard_sha256_[32]{};

    // Section 2 static seed table (sorted {uint64_t row_id, uint32_t slot} + raw 100 B rows)
    struct SeedIdx {
        uint64_t row_id;
        uint32_t slot;
    };
    std::vector<SeedIdx> seed_index_;
    std::vector<uint8_t> seed_rows_;

    // Dynamic host-RAM LRU cache (clock-eviction ring buffer of 100 B rows)
    size_t lru_cap_ = 0;
    size_t lru_clock_ = 0;
    std::vector<uint64_t> lru_keys_;
    std::vector<uint8_t>  lru_ref_;
    std::vector<uint8_t>  lru_data_;  // lru_cap_ * 100 B
    std::unordered_map<uint64_t, uint32_t> lru_map_;

    Hy4EngramStats stats_{};
};

}  // namespace strata::ngram
