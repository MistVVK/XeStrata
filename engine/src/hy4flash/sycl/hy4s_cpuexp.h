// hy4s_cpuexp.h - CPU routed-expert engine of the Hy4-Flash Intel Arc lane (HY4-SYCL): the 41 MoE layers' routed experts
// (.strata Section 4, S_K square [1536,1536] experts) are computed on the host while the GPU holds the trunk.
// Plain C++17 + OpenMP + AVX2 (+ engine/src/quant/sk_codecs.cpp); no SYCL in this module.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "strata/quant/sk_codecs.hpp"

namespace hy4s {

constexpr int CX_K = 1536;        // expert width (columns of every expert matrix)
constexpr int CX_NG = CX_K / 64;  // 24 Q8_1 groups per activation vector
constexpr int CX_NTOP = 16;       // routed experts per token

// One quantized activation vector (K = 1536) in the "planar" layout the AVX2 kernels read: for every pair of 64-weight groups p = g/2,
// bytes [p*128 + j*32 + (g&1)*16 + m] = qs_g[4*m + j]  (j = 0..3, m = 0..15).  d, s, sq = block scale, d*sum(qs) (as BlockQ8_1_G64), sum(qs).
struct ActPlan {
    alignas(32) int8_t p[CX_K];
    float d[CX_NG];
    float s[CX_NG];
    int32_t sq[CX_NG];
};

void plan_from_blocks(const strata::quant::sk::BlockQ8_1_G64* blocks /*[24]*/, ActPlan& out);
void plan_from_floats(const float* x /*[1536]*/, ActPlan& out);  // quantize_q8_1_g64 (engine) + plan_from_blocks

// Fast AVX2 kernels for the codes of the SQ-32/64 experts. W: nrows rows of 1536 columns in the S_K wire format (row = 6 super-blocks), acts: M <= 16 plans.
// out[t*ldo + r] = dot(row r, act t), same arithmetic as dot_sk_q8_1_row (integer dots exact; float accumulation in the same order).
// Supported codes: 0 (S1.25), 2 (S2_K), 3 (S3_K). Buffers must be readable 8 bytes past the last row (padding).  Returns false for another code.
bool cx_rows_q8(int code, const uint8_t* W, int nrows, const ActPlan* const* acts, int M, float* out, size_t ldo);

struct CpuExpertStats {
    double seconds = 0;     // wall time inside run()
    uint64_t calls = 0;
    uint64_t bytes = 0;     // expert bytes touched (sq + gate_up + down, per distinct expert per call)
    uint64_t pairs = 0;     // (token, expert) evaluations
    double code_sec[9] = {};  // thread-seconds in the gate_up / down matrices by code id (0..8)
    double t_prep = 0, t_gu = 0, t_dn = 0, t_red = 0;  // wall seconds per phase (summed over calls)
};

class CpuExperts {
public:
    CpuExperts();
    ~CpuExperts();
    CpuExperts(const CpuExperts&) = delete;
    CpuExperts& operator=(const CpuExperts&) = delete;

    // Opens the .strata, reads the expert directory and maps Section 4 read-only (the page cache is the RAM copy: no second copy).
    // max_tokens: largest T handled in one pass (larger calls are split).
    bool open(const std::string& model_path, int nthreads, std::string& err, int max_tokens = 512);
    // Test pool without a file: 512 random SQ-32-like expert bodies (ranks 0-31: gu S2/dn S3, 32-175: S2/S2, 176-511: S1.25/S1.25), aliased in all 41 layers.
    bool open_synthetic(int nthreads, std::string& err, int max_tokens = 512, uint32_t seed = 1);
    void close();
    bool is_open() const { return base_ != nullptr; }

    // Brings all of Section 4 into the page cache (parallel sequential reads) and maps it. Returns read GB/s; *seconds gets the total time.
    double warm(int nthreads, double* seconds = nullptr);
    // Drops Section 4 from the page cache (clean pages, unprivileged) for cold-cache tests.
    void drop_cache();
    // Asynchronous read-ahead hints.
    void prefetch_layer(int moe_layer);
    // Expert body -> dst (4 KiB aligned): memcpy from the page cache when resident, else a large pread (buffered; O_DIRECT when the experts are far larger than the RAM, HY4S_DIRECT=0|1 overrides).
    void copy_body(const uint8_t* body, uint32_t bytes, void* dst) const;
    uint64_t direct_read_bytes() const;
    uint64_t cached_read_bytes() const;
    uint64_t miss_read_bytes() const;  // bodies read with a buffered pread because their pages were not cached
    void prefetch_experts(int moe_layer, const int32_t* idx, int n);

    // One MoE layer (moe_layer = il-1, 0..40). z fp32 [T][1536] = routed_expert_down_proj output; idx int32 [T][16] expert ids (0..511);
    // w float [T][16]; zout fp32 [T][1536] (assigned). act_mode 0 = float activations (dequantized rows, the oracle's math),
    // 1 = Q8_1_G64 activations (integer kernels, the engine's fast path).
    void run(int moe_layer, int T, const float* z, const int32_t* idx, const float* w, float* zout, int act_mode);

    const CpuExpertStats& stats() const { return st_; }
    void reset_stats() { st_ = CpuExpertStats(); }
    int nthreads() const { return nth_; }
    uint64_t section4_bytes() const { return sec4_bytes_; }
    bool streaming() const { return streaming_; }  // true when the experts do not fit the page cache (warm() read only a share): reads come from the SSD
    // expert body lookup (for tests): pointer into the mapping, directory fields
    bool expert_info(int moe_layer, int expert_id, const uint8_t** body, int* gu_code, int* dn_code, uint32_t* dn_off, uint32_t* body_bytes) const;

private:
    struct Impl;
    std::unique_ptr<Impl> im_;
    const uint8_t* base_ = nullptr;
    uint64_t sec4_bytes_ = 0;
    int nth_ = 1;
    bool streaming_ = false, bypass_ = false;
    CpuExpertStats st_;
};

}  // namespace hy4s
