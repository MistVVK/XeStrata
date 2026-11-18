// src/ngram/hy4_engram_reader.cpp - 4-Module Native 8-Bit Engram (PLE) Reader for Hy4-Flash
#include "strata/ngram/hy4_engram_reader.hpp"
#include "strata/artifact/strata_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata::ngram {
namespace {

bool seek_abs(std::FILE* fp, uint64_t off) noexcept {
#if defined(_WIN32)
    return _fseeki64(fp, static_cast<__int64>(off), SEEK_SET) == 0;
#else
    return fseeko(fp, static_cast<off_t>(off), SEEK_SET) == 0;
#endif
}

struct Fp8E4M3Lut {
    float v[256]{};
    Fp8E4M3Lut() {
        for (int b = 0; b < 256; ++b) {
            const int sign = (b >> 7) & 1;
            const int exp  = (b >> 3) & 0xF;
            const int man  = b & 0x7;
            float val = 0.0f;
            if (exp == 0) {
                val = std::ldexp(static_cast<float>(man) / 8.0f, -6);
            } else if (exp == 15 && man == 7) {
                val = 0.0f;  // NaN guard
            } else {
                val = std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
            }
            v[b] = sign ? -val : val;
        }
    }
};
const Fp8E4M3Lut kFp8Lut{};

}  // namespace

float Hy4EngramReader::fp8_e4m3_to_f32(uint8_t b) noexcept {
    return kFp8Lut.v[b];
}

void Hy4EngramReader::close() {
    if (fp_) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
    seed_index_.clear();
    seed_rows_.clear();
    lru_keys_.clear();
    lru_ref_.clear();
    lru_data_.clear();
    lru_map_.clear();
    lru_cap_ = 0;
    lru_clock_ = 0;
}

bool Hy4EngramReader::open(
    const std::string& shard_path,
    const uint8_t* section2_seed,
    size_t section2_bytes,
    size_t lru_capacity_rows,
    std::string& err) {
    close();

    // Validate shard header & locate Section 4 (4 KiB page-framed Engram table)
    strata::artifact::StrataFileReader hdr_reader;
    if (!hdr_reader.open(shard_path, err)) return false;
    if (hdr_reader.header().profile != strata::artifact::StrataProfile::EngramShard) {
        err = "Expected EngramShard profile in " + shard_path;
        return false;
    }
    sec4_offset_ = hdr_reader.header().sections[4].file_offset;
    std::memcpy(shard_sha256_, hdr_reader.header().sections[4].sha256, 32);
    hdr_reader.close();

    fp_ = std::fopen(shard_path.c_str(), "rb");
    if (!fp_) {
        err = "Failed to open Engram shard: " + shard_path;
        return false;
    }

    // Parse optional 64 MiB Section 2 static seed table
    if (section2_seed && section2_bytes >= 64 && std::memcmp(section2_seed, "ENGR", 4) == 0) {
        uint32_t fmt_id = 0, n_rows = 0, row_b = 0;
        std::memcpy(&fmt_id, section2_seed + 4, 4);
        std::memcpy(&n_rows, section2_seed + 8, 4);
        std::memcpy(&row_b, section2_seed + 12, 4);
        // Verify seed SHA-256 matches the mounted shard (Cache Hit == Cache Miss invariant!)
        if (fmt_id == 8 && row_b == HY4_ENGRAM_ROW_BYTES &&
            std::memcmp(section2_seed + 16, shard_sha256_, 32) == 0) {
            const size_t idx_bytes = static_cast<size_t>(n_rows) * 12;
            const size_t data_off = ((64 + idx_bytes + 63) / 64) * 64;
            const size_t data_bytes = static_cast<size_t>(n_rows) * HY4_ENGRAM_ROW_BYTES;
            if (data_off + data_bytes <= section2_bytes) {
                seed_index_.resize(n_rows);
                for (uint32_t i = 0; i < n_rows; ++i) {
                    const uint8_t* p = section2_seed + 64 + static_cast<size_t>(i) * 12;
                    std::memcpy(&seed_index_[i].row_id, p, 8);
                    std::memcpy(&seed_index_[i].slot, p + 8, 4);
                }
                seed_rows_.assign(section2_seed + data_off, section2_seed + data_off + data_bytes);
            }
        }
    }

    if (lru_capacity_rows > 0) {
        lru_cap_ = lru_capacity_rows;
        lru_keys_.assign(lru_cap_, UINT64_MAX);
        lru_ref_.assign(lru_cap_, 0);
        lru_data_.resize(lru_cap_ * HY4_ENGRAM_ROW_BYTES);
        lru_map_.reserve(lru_cap_);
    }
    return true;
}

bool Hy4EngramReader::lookup_seed(uint64_t row_id, uint8_t out_raw[HY4_ENGRAM_ROW_BYTES]) const noexcept {
    if (seed_index_.empty()) return false;
    size_t lo = 0, hi = seed_index_.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (seed_index_[mid].row_id < row_id) lo = mid + 1;
        else hi = mid;
    }
    if (lo < seed_index_.size() && seed_index_[lo].row_id == row_id) {
        const uint32_t slot = seed_index_[lo].slot;
        std::memcpy(out_raw, seed_rows_.data() + static_cast<size_t>(slot) * HY4_ENGRAM_ROW_BYTES, HY4_ENGRAM_ROW_BYTES);
        return true;
    }
    return false;
}

void Hy4EngramReader::decode_raw_row(const uint8_t raw[HY4_ENGRAM_ROW_BYTES], float* dst96) const noexcept {
    float scale = 0.0f;
    std::memcpy(&scale, raw, 4);
    const int8_t* q = reinterpret_cast<const int8_t*>(raw + 4);
    for (uint32_t i = 0; i < HY4_ENGRAM_ROW_DIM; ++i) {
        dst96[i] = static_cast<float>(q[i]) * scale;
    }
}

bool Hy4EngramReader::fetch_rows_f32(const uint64_t* global_row_ids, size_t n, float* out_f32, std::string& err) {
    if (!fp_) {
        err = "Hy4EngramReader not open";
        return false;
    }
    struct Miss {
        uint64_t page_id;
        uint32_t in_page;
        size_t   out_idx;
        uint64_t row_id;
    };
    std::vector<Miss> misses;
    misses.reserve(n);
    uint8_t tmp[HY4_ENGRAM_ROW_BYTES];

    for (size_t i = 0; i < n; ++i) {
        const uint64_t rid = global_row_ids[i];
        ++stats_.total_lookups;
        if (lookup_seed(rid, tmp)) {
            ++stats_.seed_hits;
            decode_raw_row(tmp, out_f32 + i * HY4_ENGRAM_ROW_DIM);
            continue;
        }
        if (lru_cap_ > 0) {
            auto it = lru_map_.find(rid);
            if (it != lru_map_.end()) {
                ++stats_.lru_hits;
                lru_ref_[it->second] = 1;
                decode_raw_row(lru_data_.data() + static_cast<size_t>(it->second) * HY4_ENGRAM_ROW_BYTES,
                               out_f32 + i * HY4_ENGRAM_ROW_DIM);
                continue;
            }
        }
        misses.push_back({rid / HY4_ENGRAM_ROWS_PER_PAGE,
                          static_cast<uint32_t>(rid % HY4_ENGRAM_ROWS_PER_PAGE),
                          i, rid});
    }

    if (misses.empty()) return true;

    // Sort misses by physical 4 KiB page for sequential/deduplicated NVMe reads
    std::sort(misses.begin(), misses.end(), [](const Miss& a, const Miss& b) { return a.page_id < b.page_id; });

    uint8_t page_buf[HY4_ENGRAM_PAGE_BYTES];
    uint64_t cur_page = UINT64_MAX;

    for (const auto& m : misses) {
        if (m.page_id != cur_page) {
            if (!seek_abs(fp_, sec4_offset_ + m.page_id * HY4_ENGRAM_PAGE_BYTES) ||
                std::fread(page_buf, 1, HY4_ENGRAM_PAGE_BYTES, fp_) != HY4_ENGRAM_PAGE_BYTES) {
                err = "Engram SSD 4 KiB page read failed at page " + std::to_string(m.page_id);
                return false;
            }
            cur_page = m.page_id;
            ++stats_.ssd_page_reads;
            stats_.ssd_bytes_read += HY4_ENGRAM_PAGE_BYTES;
        }
        const uint8_t* raw_row = page_buf + static_cast<size_t>(m.in_page) * HY4_ENGRAM_ROW_BYTES;
        decode_raw_row(raw_row, out_f32 + m.out_idx * HY4_ENGRAM_ROW_DIM);

        // Insert into clock-eviction LRU
        if (lru_cap_ > 0 && lru_map_.find(m.row_id) == lru_map_.end()) {
            while (lru_ref_[lru_clock_]) {
                lru_ref_[lru_clock_] = 0;
                lru_clock_ = (lru_clock_ + 1) % lru_cap_;
            }
            const uint32_t slot = static_cast<uint32_t>(lru_clock_);
            if (lru_keys_[slot] != UINT64_MAX) lru_map_.erase(lru_keys_[slot]);
            lru_keys_[slot] = m.row_id;
            lru_ref_[slot] = 1;
            std::memcpy(lru_data_.data() + static_cast<size_t>(slot) * HY4_ENGRAM_ROW_BYTES, raw_row, HY4_ENGRAM_ROW_BYTES);
            lru_map_.emplace(m.row_id, slot);
            lru_clock_ = (lru_clock_ + 1) % lru_cap_;
        }
    }
    return true;
}

}  // namespace strata::ngram
