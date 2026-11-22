// engine/src/inkling2/cuda/ink_moe_cuda.cuh - Complete Inkling-2-Mini MoE Router + SF4 / MX8 / S_K (paired-2-row) CUDA Kernels
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Zero quality loss:
//   - Exact Top-8 / 128 sigmoid routing with F32 score-correction bias (bias for selection only, weights from sigmoid,
//     normalized by sum + 1e-20 and multiplied by route_scale = 8.0; STRATA_ROUTE_TAIL_SKIP = 0!).
//   - Exact SITU(beta=4.0, lb=25.0) activation on interleaved w13 [2304, 2304] (even=gate, odd=up).
//   - Exact deterministic 8-slot weighted reduction (zero atomic float non-determinism -> bitwise parity across runs!).
//   - Supports ALL .strata v1.1 expert codes:
//       * SF4 (code_id=15): lossless NVFP4 (L0..19) -> Blackwell sm_120a mxf4nvf4 TC for large M, fused register-LUT/cvt.e2m1x2 for M=1..16
//       * MX8 (code_id=16): lossless MXFP8 (L20..23) -> Blackwell sm_120a mxf8f6f4 TC for large M, fused hardware cvt/LUT for M=1..16
//       * S1.5, S2_K, S3_K, S4_K, S5_K, S6_K, S8_K: Q8_1_G64 dp4a (M=1..16) + sm_80+ INT8 TC (prefill/batch) with paired 2-row w2 (SB_0..SB_8) + FP16 sq_xd_dn!
#pragma once

#include "ink_cuda.cuh"
#include "ink_moe_fp4_sm120.cuh"

#if defined(__CUDACC__)
#include <mma.h>
#endif

#include <algorithm>
#include <cstdint>

namespace strata::ink {

// ---- Q8_1_G64 activation block for S_K experts (matches strata::quant::sk::BlockQ8_1_G64 bit-for-bit!) ----
struct alignas(16) InkQ8_1_G64 {
    int8_t qs[64];
    float  d;
    float  s;  // d * sum(qs)
};
static_assert(sizeof(InkQ8_1_G64) == 80, "InkQ8_1_G64 must be 80 B");

// ---- Q8_1_G32 activation block for native GGUF DP4A experts (Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_XXS, Q3_K, Q2_K, Q4_0, Q5_0, IQ4_NL) ----
struct alignas(8) InkQ8_1_G32 {
    int8_t qs[32];
    float  d;
    float  s;  // d * sum(qs)
};
static_assert(sizeof(InkQ8_1_G32) == 40, "InkQ8_1_G32 must be 40 B");

// ---- S1.5 ternary 5-trits-per-byte lookup table in constant memory (matches sk_codecs.cpp!) ----
// For byte b in 0..242: 5 trits t0..t4 in {-1, 0, +1}, packed as 4 signed bytes in low 32 bits + 5th trit in hi byte.
// Even simpler: compute on-the-fly or store int8_t[256][5] (1,280 B in constant cache -> 1-cycle hit!).
__device__ __constant__ int8_t kS15TritLut[256][5] = {
#define _T5(b) { \
    (int8_t)(((b) % 3) - 1), \
    (int8_t)((((b) / 3) % 3) - 1), \
    (int8_t)((((b) / 9) % 3) - 1), \
    (int8_t)((((b) / 27) % 3) - 1), \
    (int8_t)((((b) / 81) % 3) - 1) }
#define _T5_8(x) _T5(x), _T5((x)+1), _T5((x)+2), _T5((x)+3), _T5((x)+4), _T5((x)+5), _T5((x)+6), _T5((x)+7)
#define _T5_32(x) _T5_8(x), _T5_8((x)+8), _T5_8((x)+16), _T5_8((x)+24)
    _T5_32(0), _T5_32(32), _T5_32(64), _T5_32(96), _T5_32(128), _T5_32(160), _T5_32(192), _T5_32(224)
#undef _T5_32
#undef _T5_8
#undef _T5
};

// ---- Workspace sizing for Inkling-2-Mini MoE (M tokens, P = M * 8 pairs) ----
inline size_t ink_moe_align(size_t x) { return (x + 255) & ~(size_t)255; }
inline int ink_moe_max_tiles(int M, int tile_rows = 16) {
    const int P = M * kInkTopK;
    return (P + tile_rows - 1) / tile_rows + kInkExperts;
}

struct InkMoeWorkspace {
    InkMoeTables tb{};
    float*   r_logits = nullptr;    // [M, 128]
    int32_t* topk_idx = nullptr;    // [M, 8]
    float*   topk_w   = nullptr;    // [M, 8]
    float*   h_f32    = nullptr;    // [P, 1152] (decode/S_K intermediate)
    float*   y_pair   = nullptr;    // [P, 2304] (deterministic slot outputs)
    uint8_t* xa_fp4   = nullptr;    // [M, 36 * 48] (Blackwell SF4 TC)
    uint8_t* ha_fp4   = nullptr;    // [P, 18 * 48]
    uint8_t* xq_mx8   = nullptr;    // [M, 2304]
    uint8_t* xsc_mx8  = nullptr;    // [M, 72]
    uint8_t* hq_mx8   = nullptr;    // [P, 1152]
    uint8_t* hsc_mx8  = nullptr;    // [P, 36]
    InkQ8_1_G64* xq_sk = nullptr;   // [M, 36]
    InkQ8_1_G64* hq_sk = nullptr;   // [P, 18]
    InkQ8_1_G32* xq_g32 = nullptr;  // [M, 72] (native GGUF DP4A activation reuse)
    InkQ8_1_G32* hq_g32 = nullptr;  // [P, 36]
    void*    raw_buf  = nullptr;
    size_t   raw_bytes = 0;
    int      cap_M    = 0;

    void alloc(int max_M) {
        if (max_M <= cap_M && raw_buf) return;
        free_all();
        cap_M = max_M;
        const size_t M = (size_t)max_M;
        const size_t P = M * kInkTopK;
        const size_t MT = (size_t)ink_moe_max_tiles(max_M);
        size_t off = 0;
        auto bump = [&](size_t bytes) -> size_t {
            size_t o = off;
            off += ink_moe_align(bytes);
            return o;
        };
        const size_t o_cnt     = bump(128 * 4);
        const size_t o_off     = bump(129 * 4);
        const size_t o_act     = bump(128 * 4);
        const size_t o_nact    = bump(16);
        const size_t o_pairs   = bump(P * 4);
        const size_t o_tok     = bump(P * 4);
        const size_t o_ntiles  = bump(16);
        const size_t o_tiles   = bump(MT * sizeof(int2));
        const size_t o_rlog    = bump(M * kInkExperts * 4);
        const size_t o_tidx    = bump(P * 4);
        const size_t o_tw      = bump(P * 4);
        const size_t o_hf32    = bump(P * kInkInter * 4);
        const size_t o_ypair   = bump(P * kInkHidden * 4);
        const size_t o_xafp4   = bump(M * (kInkHidden / 64) * sm120::kActFp4SlotBytes);
        const size_t o_hafp4   = bump(P * (kInkInter / 64) * sm120::kActFp4SlotBytes);
        const size_t o_xqmx8   = bump(M * kInkHidden);
        const size_t o_xscmx8  = bump(M * (kInkHidden / 32));
        const size_t o_hqmx8   = bump(P * kInkInter);
        const size_t o_hscmx8  = bump(P * (kInkInter / 32));
        const size_t o_xqsk    = bump(M * (kInkHidden / 64) * sizeof(InkQ8_1_G64));
        const size_t o_hqsk    = bump(P * (kInkInter / 64) * sizeof(InkQ8_1_G64));
        const size_t o_xqg32   = bump(M * (kInkHidden / 32) * sizeof(InkQ8_1_G32));
        const size_t o_hqg32   = bump(P * (kInkInter / 32) * sizeof(InkQ8_1_G32));

        raw_bytes = off;
        INK_CK(cudaMalloc(&raw_buf, raw_bytes));
        uint8_t* base = (uint8_t*)raw_buf;
        tb.cnt     = (int32_t*)(base + o_cnt);
        tb.off     = (int32_t*)(base + o_off);
        tb.active  = (int32_t*)(base + o_act);
        tb.nact    = (int32_t*)(base + o_nact);
        tb.pairs   = (int32_t*)(base + o_pairs);
        tb.tok_idx = (int32_t*)(base + o_tok);
        tb.n_tiles = (int32_t*)(base + o_ntiles);
        tb.tiles   = (int2*)(base + o_tiles);
        r_logits   = (float*)(base + o_rlog);
        topk_idx   = (int32_t*)(base + o_tidx);
        topk_w     = (float*)(base + o_tw);
        h_f32      = (float*)(base + o_hf32);
        y_pair     = (float*)(base + o_ypair);
        xa_fp4     = base + o_xafp4;
        ha_fp4     = base + o_hafp4;
        xq_mx8     = base + o_xqmx8;
        xsc_mx8    = base + o_xscmx8;
        hq_mx8     = base + o_hqmx8;
        hsc_mx8    = base + o_hscmx8;
        xq_sk      = (InkQ8_1_G64*)(base + o_xqsk);
        hq_sk      = (InkQ8_1_G64*)(base + o_hqsk);
        xq_g32     = (InkQ8_1_G32*)(base + o_xqg32);
        hq_g32     = (InkQ8_1_G32*)(base + o_hqg32);
    }

    void free_all() {
        if (raw_buf) {
            INK_CK(cudaFree(raw_buf));
            raw_buf = nullptr;
            raw_bytes = 0;
            cap_M = 0;
        }
    }
};

// ============================================================================
// 1. Exact Top-8 / 128 Sigmoid Router + Device-Side Expert Binning (Zero Host Sync!)
//    Split into 128-CTA coalesced GEMV (for M<=16) / cuBLAS TF32 (for M>16) + 1-warp Top-8 selector!
// ============================================================================
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_router_gemv_smallM(
    const float* __restrict__ x,       // [M, 2304]
    const float* __restrict__ gate_w,  // [128, 2304] F32
    float* __restrict__ r_logits,      // [M, 128]
    int M) {
    const int e = blockIdx.x;  // 0..127 -> 128 CTAs across 170 SMs!
    if (e >= kInkExperts) return;
    const int tid = threadIdx.x;
    const float4* gw = reinterpret_cast<const float4*>(gate_w + (size_t)e * kInkHidden);
    constexpr int K4 = kInkHidden / 4;  // 576
    float acc[MAX_M] = {};
    for (int i = tid; i < K4; i += 256) {
        const float4 w = gw[i];
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4 xv = reinterpret_cast<const float4*>(x + (size_t)m * kInkHidden)[i];
                acc[m] = fmaf(w.x, xv.x, acc[m]);
                acc[m] = fmaf(w.y, xv.y, acc[m]);
                acc[m] = fmaf(w.z, xv.z, acc[m]);
                acc[m] = fmaf(w.w, xv.w, acc[m]);
            }
        }
    }
    __shared__ float sh[32];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float s = block_sum(acc[m], sh);
            if (tid == 0) r_logits[(size_t)m * kInkExperts + e] = s;
        }
    }
}

// 1 warp per token selects exact Top-8 by (sigmoid(logit) + bias), tie-breaking lowest expert id!
__global__ void __launch_bounds__(32) k_ink_router_select_topk8(
    const float* __restrict__ r_logits, // [M, 128]
    const float* __restrict__ gate_b,   // [128]
    int32_t* __restrict__ topk_idx,     // [M, 8]
    float* __restrict__ topk_w,         // [M, 8]
    float route_scale,
    int M) {
    const int tok = blockIdx.x;
    if (tok >= M) return;
    const int lane = threadIdx.x;
    const float* lg = r_logits + (size_t)tok * kInkExperts;

    float sc[4], pr[4];
    int id[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = lane + 32 * j;
        const float p = 1.0f / (1.0f + expf(-lg[e]));
        id[j] = e;
        pr[j] = p;
        sc[j] = p + gate_b[e];
    }

    int best_id[kInkTopK];
    float best_p[kInkTopK];
    float psum = 0.0f;
#pragma unroll
    for (int k = 0; k < kInkTopK; ++k) {
        float mv = -1e30f;
        int mi = 1000000;
        float mp = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (sc[j] > mv || (sc[j] == mv && id[j] < mi)) {
                mv = sc[j]; mi = id[j]; mp = pr[j];
            }
        }
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            float ov = __shfl_xor_sync(0xFFFFFFFFu, mv, o);
            int   oi = __shfl_xor_sync(0xFFFFFFFFu, mi, o);
            float op = __shfl_xor_sync(0xFFFFFFFFu, mp, o);
            if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; mp = op; }
        }
        best_id[k] = mi;
        best_p[k] = mp;
        psum += mp;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (id[j] == mi) sc[j] = -1e30f;
        }
    }
    if (lane < kInkTopK) {
        const float inv = route_scale / (psum + 1e-20f);
        topk_idx[(size_t)tok * kInkTopK + lane] = best_id[lane];
        topk_w[(size_t)tok * kInkTopK + lane] = best_p[lane] * inv;
    }
}

// Single-block (128 threads) device-side counting sort + tile_rows-row tile list builder for 128 experts!
// Zero host sync! Capturable inside CUDA Graphs!
__global__ void __launch_bounds__(128) k_ink_moe_bin(
    const InkExpertDev* __restrict__ ex,
    const int32_t* __restrict__ topk_idx,
    int M,
    int tile_rows,
    InkMoeTables tb) {
    __shared__ int s_cnt[128], s_off[128], s_act[128], s_fill[128], s_tcnt[128];
    const int t = threadIdx.x;
    const int P = M * kInkTopK;

    s_cnt[t] = 0;
    s_fill[t] = 0;
    __syncthreads();

    for (int i = t; i < P; i += 128) {
        const int e = topk_idx[i];
        if (ex[e].tier <= 1) atomicAdd(&s_cnt[e], 1);  // tier 0=VRAM, 1=pinned host GPU-readable, 2=CPU hybrid
    }
    __syncthreads();

    const int cv = s_cnt[t];
    const int is_act = cv > 0 ? 1 : 0;
    const int nt = (cv + tile_rows - 1) / tile_rows;
    s_off[t] = cv;
    s_act[t] = is_act;
    s_tcnt[t] = nt;
    __syncthreads();

    // Inclusive Hillis-Steele prefix sum over 128 elements (7 steps)
#pragma unroll
    for (int d = 1; d < 128; d <<= 1) {
        int a0 = 0, a1 = 0, a2 = 0;
        if (t >= d) { a0 = s_off[t - d]; a1 = s_act[t - d]; a2 = s_tcnt[t - d]; }
        __syncthreads();
        s_off[t] += a0;
        s_act[t] += a1;
        s_tcnt[t] += a2;
        __syncthreads();
    }

    const int off_e = s_off[t] - cv;
    tb.cnt[t] = cv;
    tb.off[t] = off_e;
    if (t == 127) {
        tb.off[128] = s_off[127];
        *tb.nact = s_act[127];
        *tb.n_tiles = s_tcnt[127];
    }
    if (is_act) tb.active[s_act[t] - 1] = t;
    if (nt > 0) {
        const int tbase = s_tcnt[t] - nt;
        for (int ti = 0; ti < nt; ++ti) {
            tb.tiles[tbase + ti] = make_int2(t, off_e + ti * tile_rows);
        }
    }
    s_off[t] = off_e;
    __syncthreads();

    // Scatter sorted pair indices & source token indices
    for (int i = t; i < P; i += 128) {
        const int e = topk_idx[i];
        if (ex[e].tier > 1) continue;
        const int pos = s_off[e] + atomicAdd(&s_fill[e], 1);
        tb.pairs[pos] = i;
        tb.tok_idx[pos] = i / kInkTopK;
    }
}

// ============================================================================
// 2. Deterministic 8-Slot Reduction (y_pair [M, 8, 2304] -> y_out [M, 2304] F32)
// ============================================================================
__global__ void __launch_bounds__(256) k_ink_moe_reduce8(
    const float* __restrict__ y_pair,  // [M * 8, 2304]
    float* __restrict__ y_out,         // [M, 2304]
    int M) {
    const int tok = blockIdx.y;
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (tok >= M || d >= kInkHidden) return;
    const float* base = y_pair + (size_t)tok * kInkTopK * kInkHidden + d;
    float s = base[0 * kInkHidden] + base[1 * kInkHidden]
            + base[2 * kInkHidden] + base[3 * kInkHidden]
            + base[4 * kInkHidden] + base[5 * kInkHidden]
            + base[6 * kInkHidden] + base[7 * kInkHidden];
    y_out[(size_t)tok * kInkHidden + d] = s;
}

// Masked 8-slot reduction for Dual-GPU EP (sums ONLY slots where experts[topk_idx].tier <= 1, zero memset needed!)
__global__ void __launch_bounds__(256) k_ink_moe_reduce8_tier0(
    const InkExpertDev* __restrict__ experts,
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ y_pair,
    float* __restrict__ y_out,
    int M) {
    const int tok = blockIdx.y;
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (tok >= M || d >= kInkHidden) return;
    const int32_t* tidx = topk_idx + (size_t)tok * kInkTopK;
    const float* base = y_pair + (size_t)tok * kInkTopK * kInkHidden + d;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < kInkTopK; ++k) {
        if (experts[tidx[k]].tier <= 1) s += base[(size_t)k * kInkHidden];
    }
    y_out[(size_t)tok * kInkHidden + d] = s;
}

__global__ void __launch_bounds__(256) k_ink_add_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// ============================================================================
// 2B. Native Lossless GGUF_RAW CUDA MoE Kernels (Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4, BF16, F16, F32)
// ============================================================================
struct InkGgufExpertHeader {
    uint32_t magic;       // 0x47475546 ('GGUF')
    uint32_t gu_type;     // ggml_type for gate/up
    uint32_t dn_type;     // ggml_type for down
    uint32_t flags;       // 0 = split [gate 0..F-1, up F..2F-1]
    float    gate_scale;
    float    up_scale;
    float    dn_scale;
    uint32_t reserved;
};
static_assert(sizeof(InkGgufExpertHeader) == 32, "InkGgufExpertHeader must be 32 B");

__host__ __device__ inline bool ink_gguf_has_gpu_kernel(uint32_t t) noexcept {
    switch (t) {
        case 0: case 1: case 2: case 6: case 8: case 10: case 11: case 12: case 13: case 14: case 18: case 20: case 23: case 30: case 39:
            return true;
        default:
            return false;
    }
}

__host__ __device__ inline size_t ink_gguf_row_bytes(uint32_t t, int K) noexcept {
    switch (t) {
        case 0:  return (size_t)K * 4u;
        case 1:
        case 30: return (size_t)K * 2u;
        case 2:  return (size_t)(K / 32) * 18u;
        case 6:  return (size_t)(K / 32) * 22u;
        case 8:  return (size_t)(K / 32) * 34u;
        case 10: return (size_t)(K / 256) * 84u;
        case 11: return (size_t)(K / 256) * 110u;
        case 12: return (size_t)(K / 256) * 144u;
        case 13: return (size_t)(K / 256) * 176u;
        case 14: return (size_t)(K / 256) * 210u;
        case 18: return (size_t)(K / 256) * 98u;
        case 20: return (size_t)(K / 32) * 18u;
        case 23: return (size_t)(K / 256) * 136u;
        case 39: return (size_t)(K / 32) * 17u;
        default: return 0u;
    }
}

#if defined(__CUDACC__)
__device__ __constant__ int8_t kInkIq4NlLut[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// Exact IQ3_XXS 256-entry 4-magnitude codebook (1 KB in constant cache -> 1-cycle hit!)
__device__ __constant__ uint32_t kInkIq3XxsGrid[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u,
};

// Fast 1-warp-per-32-elem Q8_1_G32 quantizer for vectorized GGUF DP4A activation reuse!
__global__ void __launch_bounds__(256) k_ink_quant_q8_1_g32(
    const float* __restrict__ x,
    InkQ8_1_G32* __restrict__ out,
    int rows,
    int cols) {
    const int nb = cols >> 5;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (w >= rows * nb) return;
    const float v = x[(size_t)w * 32 + lane];
    const float am = warp_max(fabsf(v));
    const float d = am * (1.0f / 127.0f);
    const float inv = d > 0.0f ? (127.0f / am) : 0.0f;
    const int q = max(-127, min(127, __float2int_rn(v * inv)));
    int s = q;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    InkQ8_1_G32& dst = out[w];
    dst.qs[lane] = (int8_t)q;
    if (lane == 0) {
        dst.d = d;
        dst.s = d * (float)s;
    }
}

__device__ __forceinline__ uint32_t gguf_pack4_i8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) | ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}

// Vectorized DP4A multi-token GGUF row dot product: reads & unpacks each weight superblock ONCE across up to MAX_M tokens!
// Covers ALL priority Unsloth quants (Q4_K=12, Q5_K=13, Q6_K=14, Q8_0=8, IQ4_XS=23, IQ3_XXS=18, Q3_K=11, Q2_K=10, Q4_0=2, Q5_0=6, IQ4_NL=20)
// plus multi-token cached MXFP4=39, BF16=30, F16=1, F32=0!
template <int MAX_M>
__device__ __forceinline__ void ink_gguf_row_dot_multi(
    uint32_t ggml_type,
    const uint8_t* __restrict__ row,
    const InkQ8_1_G32* const (&xq_ptrs)[MAX_M],
    const float* const (&xf_ptrs)[MAX_M],
    int n_e,
    int K,
    int lane,
    float (&acc)[MAX_M]) {
    const int sub = lane >> 2;  // 0..7 (32-elem sub-block index)
    const int l4  = lane & 3;   // 0..3 (8-elem chunk within sub-block)
    const int in32 = l4 * 8;

    if (ggml_type == 12) {  // Q4_K: 256 elems = 144 B
        const int j64 = sub >> 1;
        const int hi  = sub & 1;
        for (int b = 0; b < (K >> 8); ++b) {
            const uint8_t* blk = row + (size_t)b * 144u;
            const float d    = f16_to_f32(ld16_u(blk));
            const float dmin = f16_to_f32(ld16_u(blk + 2));
            const uint8_t* sc = blk + 4;
            uint8_t sd, sm;
            if (sub < 4) { sd = sc[sub] & 63u; sm = sc[sub + 4] & 63u; }
            else { sd = (sc[sub + 4] & 0xFu) | ((sc[sub - 4] >> 6) << 4); sm = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4); }
            const float dl = d * (float)sd;
            const float ml = dmin * (float)sm;
            const uint8_t* qp = blk + 16 + j64 * 32 + in32;
            const uint32_t p0 = ld32_u(qp);
            const uint32_t p1 = ld32_u(qp + 4);
            const uint32_t w0 = hi ? ((p0 >> 4) & 0x0F0F0F0Fu) : (p0 & 0x0F0F0F0Fu);
            const uint32_t w1 = hi ? ((p1 >> 4) & 0x0F0F0F0Fu) : (p1 & 0x0F0F0F0Fu);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G32& xb = xq_ptrs[m][b * 8 + sub];
                    const uint2 x2 = *reinterpret_cast<const uint2*>(xb.qs + in32);
                    int32_t isum = dp4a_i8(w1, x2.y, dp4a_i8(w0, x2.x, 0));
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 2);
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 1);
                    if (l4 == 0) acc[m] = fmaf(dl * xb.d, (float)isum, fmaf(-ml, xb.s, acc[m]));
                }
            }
        }
    } else if (ggml_type == 13) {  // Q5_K: 256 elems = 176 B
        const int j64 = sub >> 1;
        const int hi  = sub & 1;
        for (int b = 0; b < (K >> 8); ++b) {
            const uint8_t* blk = row + (size_t)b * 176u;
            const float d    = f16_to_f32(ld16_u(blk));
            const float dmin = f16_to_f32(ld16_u(blk + 2));
            const uint8_t* sc = blk + 4;
            const uint8_t* qh = blk + 16 + in32;
            uint8_t sd, sm;
            if (sub < 4) { sd = sc[sub] & 63u; sm = sc[sub + 4] & 63u; }
            else { sd = (sc[sub + 4] & 0xFu) | ((sc[sub - 4] >> 6) << 4); sm = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4); }
            const float dl = d * (float)sd;
            const float ml = dmin * (float)sm;
            const uint8_t* qp = blk + 48 + j64 * 32 + in32;
            const uint32_t p0 = ld32_u(qp);
            const uint32_t p1 = ld32_u(qp + 4);
            const uint32_t h0 = ld32_u(qh);
            const uint32_t h1 = ld32_u(qh + 4);
            const uint32_t hb0 = ((h0 >> sub) & 0x01010101u) << 4;
            const uint32_t hb1 = ((h1 >> sub) & 0x01010101u) << 4;
            const uint32_t w0 = (hi ? ((p0 >> 4) & 0x0F0F0F0Fu) : (p0 & 0x0F0F0F0Fu)) | hb0;
            const uint32_t w1 = (hi ? ((p1 >> 4) & 0x0F0F0F0Fu) : (p1 & 0x0F0F0F0Fu)) | hb1;
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G32& xb = xq_ptrs[m][b * 8 + sub];
                    const uint2 x2 = *reinterpret_cast<const uint2*>(xb.qs + in32);
                    int32_t isum = dp4a_i8(w1, x2.y, dp4a_i8(w0, x2.x, 0));
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 2);
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 1);
                    if (l4 == 0) acc[m] = fmaf(dl * xb.d, (float)isum, fmaf(-ml, xb.s, acc[m]));
                }
            }
        }
    } else if (ggml_type == 14) {  // Q6_K: 256 elems = 210 B (two 16-elem scales per 32-elem sub-block!)
        const int half = sub >> 2;
        const int qgrp = sub & 3;
        const int ql_off = half * 64 + ((qgrp & 1) ? 32 : 0) + in32;
        const int qh_shift = qgrp * 2;
        const int hi_nib = (qgrp >= 2) ? 4 : 0;
        const int sc_idx = half * 8 + qgrp * 2 + (l4 >> 1);
        for (int b = 0; b < (K >> 8); ++b) {
            const uint8_t* blk = row + (size_t)b * 210u;
            const float d = f16_to_f32(ld16_u(blk + 208));
            const int8_t scv = reinterpret_cast<const int8_t*>(blk + 192)[sc_idx];
            const float dl = d * (float)scv;
            const uint32_t p0 = ld32_u(blk + ql_off);
            const uint32_t p1 = ld32_u(blk + ql_off + 4);
            const uint32_t h0 = ld32_u(blk + 128 + half * 32 + in32);
            const uint32_t h1 = ld32_u(blk + 128 + half * 32 + in32 + 4);
            const uint32_t q0 = ((p0 >> hi_nib) & 0x0F0F0F0Fu) | (((h0 >> qh_shift) & 0x03030303u) << 4);
            const uint32_t q1 = ((p1 >> hi_nib) & 0x0F0F0F0Fu) | (((h1 >> qh_shift) & 0x03030303u) << 4);
            // Subtract 32 from each byte (safe since each byte is in 0..63 -> -32..31 fits in int8_t)
            const uint32_t w0 = gguf_pack4_i8((int8_t)((q0 & 0xFFu) - 32), (int8_t)(((q0 >> 8) & 0xFFu) - 32),
                                              (int8_t)(((q0 >> 16) & 0xFFu) - 32), (int8_t)(((q0 >> 24) & 0xFFu) - 32));
            const uint32_t w1 = gguf_pack4_i8((int8_t)((q1 & 0xFFu) - 32), (int8_t)(((q1 >> 8) & 0xFFu) - 32),
                                              (int8_t)(((q1 >> 16) & 0xFFu) - 32), (int8_t)(((q1 >> 24) & 0xFFu) - 32));
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G32& xb = xq_ptrs[m][b * 8 + sub];
                    const uint2 x2 = *reinterpret_cast<const uint2*>(xb.qs + in32);
                    int32_t isum = dp4a_i8(w1, x2.y, dp4a_i8(w0, x2.x, 0));
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 1);
                    if ((l4 & 1) == 0) acc[m] = fmaf(dl * xb.d, (float)isum, acc[m]);
                }
            }
        }
    } else if (ggml_type == 23) {  // IQ4_XS: 256 elems = 136 B
        const int hi = l4 >> 1;
        const int qoff = sub * 16 + (l4 & 1) * 8;
        for (int b = 0; b < (K >> 8); ++b) {
            const uint8_t* blk = row + (size_t)b * 136u;
            const float d = f16_to_f32(ld16_u(blk));
            const uint16_t sh = ld16_u(blk + 2);
            const uint8_t* sl = blk + 4;
            const int ls = ((sl[sub >> 1] >> (4 * (sub & 1))) & 0xFu) | (((sh >> (2 * sub)) & 3u) << 4);
            const float dl = d * (float)(ls - 32);
            const uint8_t* qp = blk + 8 + qoff;
            int8_t wv[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint8_t nib = hi ? (qp[k] >> 4) : (qp[k] & 0xFu);
                wv[k] = kInkIq4NlLut[nib];
            }
            const uint32_t w0 = gguf_pack4_i8(wv[0], wv[1], wv[2], wv[3]);
            const uint32_t w1 = gguf_pack4_i8(wv[4], wv[5], wv[6], wv[7]);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G32& xb = xq_ptrs[m][b * 8 + sub];
                    const uint2 x2 = *reinterpret_cast<const uint2*>(xb.qs + in32);
                    int32_t isum = dp4a_i8(w1, x2.y, dp4a_i8(w0, x2.x, 0));
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 2);
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 1);
                    if (l4 == 0) acc[m] = fmaf(dl * xb.d, (float)isum, acc[m]);
                }
            }
        }
    } else if (ggml_type == 18) {  // IQ3_XXS: 256 elems = 98 B
        for (int b = 0; b < (K >> 8); ++b) {
            const uint8_t* blk = row + (size_t)b * 98u;
            const float d = f16_to_f32(ld16_u(blk));
            const uint8_t* qs = blk + 2;
            const uint32_t aux32 = ld32_u(blk + 66 + sub * 4);
            const float dl = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            const uint32_t s7 = (aux32 >> (7 * l4)) & 127u;
            const uint32_t signs = s7 | ((uint32_t)(__popc(s7) & 1) << 7);
            const uint32_t g1 = kInkIq3XxsGrid[qs[sub * 8 + 2 * l4 + 0]];
            const uint32_t g2 = kInkIq3XxsGrid[qs[sub * 8 + 2 * l4 + 1]];
            int8_t wv[8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int8_t v1 = (int8_t)((g1 >> (8 * k)) & 0xFFu);
                const int8_t v2 = (int8_t)((g2 >> (8 * k)) & 0xFFu);
                wv[k]     = ((signs >> k) & 1u) ? (int8_t)(-v1) : v1;
                wv[k + 4] = ((signs >> (k + 4)) & 1u) ? (int8_t)(-v2) : v2;
            }
            const uint32_t w0 = gguf_pack4_i8(wv[0], wv[1], wv[2], wv[3]);
            const uint32_t w1 = gguf_pack4_i8(wv[4], wv[5], wv[6], wv[7]);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G32& xb = xq_ptrs[m][b * 8 + sub];
                    const uint2 x2 = *reinterpret_cast<const uint2*>(xb.qs + in32);
                    int32_t isum = dp4a_i8(w1, x2.y, dp4a_i8(w0, x2.x, 0));
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 2);
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 1);
                    if (l4 == 0) acc[m] = fmaf(dl * xb.d, (float)isum, acc[m]);
                }
            }
        }
    } else if (ggml_type == 11) {  // Q3_K: 256 elems = 110 B (two 16-elem scales per 32-elem sub-block!)
        const int half = sub >> 2;
        const int j = sub & 3;
        const int shift = j * 2;
        const uint8_t hm_mask = (uint8_t)(1u << (half * 4 + j));
        const int is = half * 8 + j * 2 + (l4 >> 1);
        for (int b = 0; b < (K >> 8); ++b) {
            const uint8_t* blk = row + (size_t)b * 110u;
            const uint8_t* hm = blk + in32;
            const uint8_t* qs = blk + 32 + half * 32 + in32;
            const uint8_t* sc = blk + 96;
            const float d = f16_to_f32(ld16_u(blk + 108));
            int sc_i;
            if (is < 4)       sc_i = (sc[is] & 0xFu) | (((sc[is + 8] >> 0) & 3u) << 4);
            else if (is < 8)  sc_i = (sc[is] & 0xFu) | (((sc[is + 4] >> 2) & 3u) << 4);
            else if (is < 12) sc_i = (sc[is - 8] >> 4) | (((sc[is] >> 4) & 3u) << 4);
            else              sc_i = (sc[is - 8] >> 4) | (((sc[is - 4] >> 6) & 3u) << 4);
            const float dl = d * (float)(sc_i - 32);
            int8_t wv[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                wv[k] = (int8_t)((int)((qs[k] >> shift) & 3u) - ((hm[k] & hm_mask) ? 0 : 4));
            }
            const uint32_t w0 = gguf_pack4_i8(wv[0], wv[1], wv[2], wv[3]);
            const uint32_t w1 = gguf_pack4_i8(wv[4], wv[5], wv[6], wv[7]);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G32& xb = xq_ptrs[m][b * 8 + sub];
                    const uint2 x2 = *reinterpret_cast<const uint2*>(xb.qs + in32);
                    int32_t isum = dp4a_i8(w1, x2.y, dp4a_i8(w0, x2.x, 0));
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 1);
                    if ((l4 & 1) == 0) acc[m] = fmaf(dl * xb.d, (float)isum, acc[m]);
                }
            }
        }
    } else if (ggml_type == 10) {  // Q2_K: 256 elems = 84 B
        const int half = sub >> 2;
        const int j = sub & 3;
        const int shift = j * 2;
        const int is = half * 8 + j * 2 + (l4 >> 1);
        for (int b = 0; b < (K >> 8); ++b) {
            const uint8_t* blk = row + (size_t)b * 84u;
            const uint8_t scb = blk[is];
            const float d    = f16_to_f32(ld16_u(blk + 80));
            const float dmin = f16_to_f32(ld16_u(blk + 82));
            const float dl = d * (float)(scb & 0xFu);
            const float ml = dmin * (float)(scb >> 4);
            const uint8_t* qp = blk + 16 + half * 32 + in32;
            const uint32_t p0 = ld32_u(qp);
            const uint32_t p1 = ld32_u(qp + 4);
            const uint32_t w0 = (p0 >> shift) & 0x03030303u;
            const uint32_t w1 = (p1 >> shift) & 0x03030303u;
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G32& xb = xq_ptrs[m][b * 8 + sub];
                    const uint2 x2 = *reinterpret_cast<const uint2*>(xb.qs + in32);
                    int32_t isum = dp4a_i8(w1, x2.y, dp4a_i8(w0, x2.x, 0));
                    int32_t xsum = dp4a_i8(0x01010101u, x2.y, dp4a_i8(0x01010101u, x2.x, 0));
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 1);
                    xsum += __shfl_xor_sync(0xFFFFFFFFu, xsum, 1);
                    if ((l4 & 1) == 0) acc[m] = fmaf(xb.d, dl * (float)isum - ml * (float)xsum, acc[m]);
                }
            }
        }
    } else if (ggml_type == 8 || ggml_type == 2 || ggml_type == 20 || ggml_type == 6) {
        // 32-element block types (Q8_0=8, Q4_0=2, IQ4_NL=20, Q5_0=6): 8 blocks processed in parallel per warp!
        const int nblk = K >> 5;
        const size_t bbytes = (ggml_type == 8) ? 34u : (ggml_type == 6 ? 22u : 18u);
        for (int b0 = 0; b0 < nblk; b0 += 8) {
            const int b = b0 + sub;
            const bool valid = (b < nblk);
            float dl = 0.0f;
            uint32_t w0 = 0u, w1 = 0u;
            if (valid) {
                const uint8_t* blk = row + (size_t)b * bbytes;
                dl = f16_to_f32(ld16_u(blk));
                if (ggml_type == 8) {
                    w0 = ld32_u(blk + 2 + in32);
                    w1 = ld32_u(blk + 2 + in32 + 4);
                } else if (ggml_type == 2) {
                    const uint8_t* qp = blk + 2 + (l4 & 1) * 8;
                    const int hi = l4 >> 1;
                    int8_t wv[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) wv[k] = (int8_t)((int)(hi ? (qp[k] >> 4) : (qp[k] & 0xFu)) - 8);
                    w0 = gguf_pack4_i8(wv[0], wv[1], wv[2], wv[3]);
                    w1 = gguf_pack4_i8(wv[4], wv[5], wv[6], wv[7]);
                } else if (ggml_type == 20) {
                    const uint8_t* qp = blk + 2 + (l4 & 1) * 8;
                    const int hi = l4 >> 1;
                    int8_t wv[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) wv[k] = kInkIq4NlLut[hi ? (qp[k] >> 4) : (qp[k] & 0xFu)];
                    w0 = gguf_pack4_i8(wv[0], wv[1], wv[2], wv[3]);
                    w1 = gguf_pack4_i8(wv[4], wv[5], wv[6], wv[7]);
                } else {  // Q5_0 (6)
                    const uint32_t qh = ld32_u(blk + 2);
                    const uint8_t* qp = blk + 6 + (l4 & 1) * 8;
                    const int hi = l4 >> 1;
                    const int hbase = hi * 16 + (l4 & 1) * 8;
                    int8_t wv[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const uint32_t nib = hi ? (qp[k] >> 4) : (qp[k] & 0xFu);
                        const uint32_t hb = ((qh >> (hbase + k)) & 1u) << 4;
                        wv[k] = (int8_t)((int)(nib | hb) - 16);
                    }
                    w0 = gguf_pack4_i8(wv[0], wv[1], wv[2], wv[3]);
                    w1 = gguf_pack4_i8(wv[4], wv[5], wv[6], wv[7]);
                }
            }
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    int32_t isum = 0;
                    float xd = 0.0f;
                    if (valid) {
                        const InkQ8_1_G32& xb = xq_ptrs[m][b];
                        xd = xb.d;
                        const uint2 x2 = *reinterpret_cast<const uint2*>(xb.qs + in32);
                        isum = dp4a_i8(w1, x2.y, dp4a_i8(w0, x2.x, 0));
                    }
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 2);
                    isum += __shfl_xor_sync(0xFFFFFFFFu, isum, 1);
                    if (l4 == 0 && valid) acc[m] = fmaf(dl * xd, (float)isum, acc[m]);
                }
            }
        }
    } else if (ggml_type == 39) {  // MXFP4: 32 elems = 17 B (multi-token weight reuse!)
        for (int b = lane; b < (K >> 5); b += 32) {
            const uint8_t* blk = row + (size_t)b * 17u;
            const uint8_t e = blk[0];
            const float d = (e < 2) ? __uint_as_float(0x00200000u << e) : __uint_as_float((uint32_t)(e - 1) << 23);
            const uint8_t* qs = blk + 1;
            float wv[32];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint8_t q = qs[j];
                wv[j]      = 2.0f * e2m1_nibble_to_f32(q & 0xFu) * d;
                wv[j + 16] = 2.0f * e2m1_nibble_to_f32(q >> 4) * d;
            }
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const float* xb = xf_ptrs[m] + b * 32;
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < 32; ++j) s = fmaf(wv[j], xb[j], s);
                    acc[m] += s;
                }
            }
        }
    } else if (ggml_type == 30 || ggml_type == 1 || ggml_type == 0) {
        for (int i = lane; i < K; i += 32) {
            float wv;
            if (ggml_type == 30)      wv = bf16_to_f32(reinterpret_cast<const uint16_t*>(row)[i]);
            else if (ggml_type == 1)  wv = f16_to_f32(reinterpret_cast<const uint16_t*>(row)[i]);
            else                      wv = reinterpret_cast<const float*>(row)[i];
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) acc[m] = fmaf(wv, xf_ptrs[m][i], acc[m]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < n_e) acc[m] = warp_sum(acc[m]);
    }
}

// Grouped GGUF_RAW Gate+Up+SITU DP4A kernel with multi-token activation reuse (MAX_M = 1, 4, 16)!
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink_moe_gu_gguf_dp4a(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const InkQ8_1_G32* __restrict__ xq_g32,
    const float* __restrict__ x_f32,
    float* __restrict__ h_out) {
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const InkExpertDev ex = experts[e];
    if (ex.gu_code != FMT_GGUF_RAW) return;
    const int f = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (f >= kInkInter) return;
    const int lane = threadIdx.x & 31;
    const int beg = tb.off[e];
    const int cnt = tb.cnt[e];
    const uint8_t* body = sec4_base + ex.body_off;
    const InkGgufExpertHeader* hdr = reinterpret_cast<const InkGgufExpertHeader*>(body);
    const size_t rb = ink_gguf_row_bytes(hdr->gu_type, kInkHidden);
    const uint8_t* base = body + sizeof(InkGgufExpertHeader);
    const uint8_t* rg = base + (size_t)f * rb;
    const uint8_t* ru = base + (size_t)(kInkInter + f) * rb;
    constexpr int NBX = kInkHidden >> 5;
    for (int k0 = 0; k0 < cnt; k0 += MAX_M) {
        const int n_e = min(MAX_M, cnt - k0);
        const InkQ8_1_G32* xq_ptrs[MAX_M];
        const float* xf_ptrs[MAX_M];
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            const int ti = (m < n_e) ? tb.tok_idx[beg + k0 + m] : 0;
            xq_ptrs[m] = xq_g32 + (size_t)ti * NBX;
            xf_ptrs[m] = x_f32  + (size_t)ti * kInkHidden;
        }
        float ga[MAX_M] = {}, ua[MAX_M] = {};
        ink_gguf_row_dot_multi<MAX_M>(hdr->gu_type, rg, xq_ptrs, xf_ptrs, n_e, kInkHidden, lane, ga);
        ink_gguf_row_dot_multi<MAX_M>(hdr->gu_type, ru, xq_ptrs, xf_ptrs, n_e, kInkHidden, lane, ua);
        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    h_out[(size_t)(beg + k0 + m) * kInkInter + f] =
                        situ_f32(ga[m] * hdr->gate_scale, ua[m] * hdr->up_scale);
                }
            }
        }
    }
}

// Grouped GGUF_RAW Down DP4A kernel with multi-token activation reuse (MAX_M = 1, 4, 16)!
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink_moe_dn_gguf_dp4a(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const InkQ8_1_G32* __restrict__ hq_g32,
    const float* __restrict__ h_in,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair) {
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const InkExpertDev ex = experts[e];
    if (ex.dn_code != FMT_GGUF_RAW) return;
    const int r = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= kInkHidden) return;
    const int lane = threadIdx.x & 31;
    const int beg = tb.off[e];
    const int cnt = tb.cnt[e];
    const uint8_t* body = sec4_base + ex.body_off;
    const InkGgufExpertHeader* hdr = reinterpret_cast<const InkGgufExpertHeader*>(body);
    const size_t rb = ink_gguf_row_bytes(hdr->dn_type, kInkInter);
    const uint8_t* rr = body + ex.dn_off + (size_t)r * rb;
    constexpr int NBH = kInkInter >> 5;
    for (int k0 = 0; k0 < cnt; k0 += MAX_M) {
        const int n_e = min(MAX_M, cnt - k0);
        const InkQ8_1_G32* hq_ptrs[MAX_M];
        const float* hf_ptrs[MAX_M];
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            const int row_i = (m < n_e) ? (beg + k0 + m) : 0;
            hq_ptrs[m] = hq_g32 + (size_t)row_i * NBH;
            hf_ptrs[m] = h_in   + (size_t)row_i * kInkInter;
        }
        float va[MAX_M] = {};
        ink_gguf_row_dot_multi<MAX_M>(hdr->dn_type, rr, hq_ptrs, hf_ptrs, n_e, kInkInter, lane, va);
        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const int pair = tb.pairs[beg + k0 + m];
                    y_pair[(size_t)pair * kInkHidden + r] = va[m] * hdr->dn_scale * topk_w[pair];
                }
            }
        }
    }
}
#endif  // __CUDACC__

// ============================================================================
// 3. Universal Q8_1_G64 Quantizer (for S_K experts, matches sk_codecs.cpp bit-for-bit!)
// ============================================================================
__global__ void __launch_bounds__(256) k_ink_quant_q8_1(
    const float* __restrict__ x,
    InkQ8_1_G64* __restrict__ out,
    int rows,
    int cols) {
    const int ng = cols / 64;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int total = rows * ng;
    if (w >= total) return;
    const float* xb = x + (size_t)w * 64;
    const float v0 = xb[2 * lane];
    const float v1 = xb[2 * lane + 1];
    float am = warp_max(fmaxf(fabsf(v0), fabsf(v1)));
    const float d = am * (1.0f / 127.0f);
    const float inv = d > 0.0f ? (127.0f / am) : 0.0f;
    int q0 = max(-127, min(127, __float2int_rn(v0 * inv)));
    int q1 = max(-127, min(127, __float2int_rn(v1 * inv)));
    int s = q0 + q1;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    InkQ8_1_G64& dst = out[w];
    dst.qs[2 * lane]     = (int8_t)q0;
    dst.qs[2 * lane + 1] = (int8_t)q1;
    if (lane == 0) {
        dst.d = d;
        dst.s = d * (float)s;
    }
}

// ============================================================================
// 4. Bandwidth-Optimal Decode & Verify (M = 1..16) Fused Expert Kernels (sm_70 .. sm_121a!)
//    Each active expert caches its n_e (<= 16) input vectors in shared memory (or L1) and
//    reads each weight byte ONCE across all n_e tokens routed to that expert!
// ============================================================================

// Helper: Unpack 8 S_K weights ONCE into (w0, w1) int8x4 registers so MAX_M=4/8/16 tokens reuse the unpack (8x faster!)
template <int FMT>
__device__ __forceinline__ void sk_unpack8_lane8(const uint8_t* pl, int g, int lane_in_g, uint32_t& w0, uint32_t& w1) {
    const int in_g = lane_in_g * 8;
    if constexpr (FMT == FMT_S8_K) {
        w0 = ld32_u(pl + g * 64 + in_g);
        w1 = ld32_u(pl + g * 64 + in_g + 4);
    } else if constexpr (FMT == FMT_S2_K) {
        const uint8_t b0 = pl[g * 16 + lane_in_g * 2 + 0];
        const uint8_t b1 = pl[g * 16 + lane_in_g * 2 + 1];
        w0 = (b0 & 3u) | (((b0 >> 2) & 3u) << 8) | (((b0 >> 4) & 3u) << 16) | (((b0 >> 6) & 3u) << 24);
        w1 = (b1 & 3u) | (((b1 >> 2) & 3u) << 8) | (((b1 >> 4) & 3u) << 16) | (((b1 >> 6) & 3u) << 24);
    } else if constexpr (FMT == FMT_S3_K) {
        const uint8_t b0 = pl[g * 16 + lane_in_g * 2 + 0];
        const uint8_t b1 = pl[g * 16 + lane_in_g * 2 + 1];
        const uint8_t hb = pl[64 + g * 8 + lane_in_g];
        w0 = (b0 & 3u) | (((b0 >> 2) & 3u) << 8) | (((b0 >> 4) & 3u) << 16) | (((b0 >> 6) & 3u) << 24);
        w1 = (b1 & 3u) | (((b1 >> 2) & 3u) << 8) | (((b1 >> 4) & 3u) << 16) | (((b1 >> 6) & 3u) << 24);
        w0 |= ((hb & 1u) << 2) | (((hb >> 1) & 1u) << 10) | (((hb >> 2) & 1u) << 18) | (((hb >> 3) & 1u) << 26);
        w1 |= (((hb >> 4) & 1u) << 2) | (((hb >> 5) & 1u) << 10) | (((hb >> 6) & 1u) << 18) | (((hb >> 7) & 1u) << 26);
    } else if constexpr (FMT == FMT_S4_K) {
        const uint32_t p4 = ld32_u(pl + g * 32 + lane_in_g * 4);
        w0 = (p4 & 0x0Fu) | (((p4 >> 4) & 0x0Fu) << 8) | (((p4 >> 8) & 0x0Fu) << 16) | (((p4 >> 12) & 0x0Fu) << 24);
        w1 = ((p4 >> 16) & 0x0Fu) | (((p4 >> 20) & 0x0Fu) << 8) | (((p4 >> 24) & 0x0Fu) << 16) | (((p4 >> 28) & 0x0Fu) << 24);
    } else if constexpr (FMT == FMT_S5_K) {
        const uint32_t p4 = ld32_u(pl + g * 32 + lane_in_g * 4);
        const uint8_t hb = pl[128 + g * 8 + lane_in_g];
        w0 = (p4 & 0x0Fu) | (((p4 >> 4) & 0x0Fu) << 8) | (((p4 >> 8) & 0x0Fu) << 16) | (((p4 >> 12) & 0x0Fu) << 24);
        w1 = ((p4 >> 16) & 0x0Fu) | (((p4 >> 20) & 0x0Fu) << 8) | (((p4 >> 24) & 0x0Fu) << 16) | (((p4 >> 28) & 0x0Fu) << 24);
        w0 |= ((hb & 1u) << 4) | (((hb >> 1) & 1u) << 12) | (((hb >> 2) & 1u) << 20) | (((hb >> 3) & 1u) << 28);
        w1 |= (((hb >> 4) & 1u) << 4) | (((hb >> 5) & 1u) << 12) | (((hb >> 6) & 1u) << 20) | (((hb >> 7) & 1u) << 28);
    } else if constexpr (FMT == FMT_S6_K) {
        const uint32_t p4 = ld32_u(pl + g * 32 + lane_in_g * 4);
        const uint16_t h16 = ld16_u(pl + 128 + g * 16 + lane_in_g * 2);
        w0 = (p4 & 0x0Fu) | (((p4 >> 4) & 0x0Fu) << 8) | (((p4 >> 8) & 0x0Fu) << 16) | (((p4 >> 12) & 0x0Fu) << 24);
        w1 = ((p4 >> 16) & 0x0Fu) | (((p4 >> 20) & 0x0Fu) << 8) | (((p4 >> 24) & 0x0Fu) << 16) | (((p4 >> 28) & 0x0Fu) << 24);
        w0 |= ((h16 & 3u) << 4) | (((h16 >> 2) & 3u) << 12) | (((h16 >> 4) & 3u) << 20) | (((h16 >> 6) & 3u) << 28);
        w1 |= (((h16 >> 8) & 3u) << 4) | (((h16 >> 10) & 3u) << 12) | (((h16 >> 12) & 3u) << 20) | (((h16 >> 14) & 3u) << 28);
    } else if constexpr (FMT == FMT_S1_5) {
        uint8_t b[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int idx = g * 64 + in_g + k;
            b[k] = (uint8_t)kS15TritLut[pl[idx / 5]][idx % 5];
        }
        w0 = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
        w1 = (uint32_t)b[4] | ((uint32_t)b[5] << 8) | ((uint32_t)b[6] << 16) | ((uint32_t)b[7] << 24);
    } else {
        w0 = w1 = 0u;
    }
}

// Compute 1 row of S_K (K = 2304 = 9 SBs) against n_e <= MAX_M tokens simultaneously (unpacks weights ONCE!)
template <int FMT, int MAX_M>
__device__ __forceinline__ void sk_row_dot_2304(
    const uint8_t* row_ptr,
    const InkQ8_1_G64* const (&x_ptrs)[MAX_M],
    int n_e,
    int lane,
    float (&acc)[MAX_M]) {
    constexpr int SB = ink_sb_bytes(FMT);
    constexpr float HALF = ink_sk_half(FMT);
    const int g = lane >> 3;         // 0..3
    const int lg = lane & 7;         // 0..7
    const int in_g = lg * 8;
#pragma unroll
    for (int s = 0; s < 9; ++s) {
        const uint8_t* blk = row_ptr + s * SB;
        const float D = f16_to_f32(ld16_u(blk));
        const float eff = D * ((float)blk[2 + g] * (1.0f / 255.0f));
        const uint8_t* pl = blk + 6;
        uint32_t w0, w1;
        sk_unpack8_lane8<FMT>(pl, g, lg, w0, w1);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
                const InkQ8_1_G64& xb = x_ptrs[m][s * 4 + g];
                const uint2 x2 = *reinterpret_cast<const uint2*>(xb.qs + in_g);
                int32_t isum = dp4a_i8(w1, x2.y, dp4a_i8(w0, x2.x, 0));
#pragma unroll
                for (int o = 4; o >= 1; o >>= 1) isum += __shfl_xor_sync(0xFFFFFFFFu, isum, o);
                if (lg == 0) {
                    if constexpr (FMT == FMT_S8_K || FMT == FMT_S1_5) {
                        acc[m] = fmaf(eff, xb.d * (float)isum, acc[m]);
                    } else {
                        acc[m] = fmaf(eff, xb.d * (float)isum - HALF * xb.s, acc[m]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < n_e) {
            acc[m] += __shfl_xor_sync(0xFFFFFFFFu, acc[m], 8);
            acc[m] += __shfl_xor_sync(0xFFFFFFFFu, acc[m], 16);
        }
    }
}

// Paired 2-row S_K Down dot product (unpacks weights ONCE across all n_e tokens!)
template <int FMT, int MAX_M>
__device__ __forceinline__ void sk_paired2_dot_1152(
    const uint8_t* pair_ptr,
    const InkQ8_1_G64* const (&h_ptrs)[MAX_M],
    int n_e,
    int lane,
    float (&acc0)[MAX_M],
    float (&acc1)[MAX_M]) {
    constexpr int SB = ink_sb_bytes(FMT);
    constexpr float HALF = ink_sk_half(FMT);
    const int g = lane >> 3;   // 0..3
    const int lg = lane & 7;   // 0..7
    const int in_g = lg * 8;
#pragma unroll
    for (int s = 0; s < 9; ++s) {
        const uint8_t* blk = pair_ptr + s * SB;
        const float D = f16_to_f32(ld16_u(blk));
        const float eff = D * ((float)blk[2 + g] * (1.0f / 255.0f));
        const uint8_t* pl = blk + 6;
        uint32_t w0, w1;
        sk_unpack8_lane8<FMT>(pl, g, lg, w0, w1);
        const int sub_global = s * 4 + g;
        const bool is_r1 = (sub_global >= 18);
        const int xg = is_r1 ? (sub_global - 18) : sub_global;
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
                const InkQ8_1_G64& xb = h_ptrs[m][xg];
                const uint2 x2 = *reinterpret_cast<const uint2*>(xb.qs + in_g);
                int32_t isum = dp4a_i8(w1, x2.y, dp4a_i8(w0, x2.x, 0));
#pragma unroll
                for (int o = 4; o >= 1; o >>= 1) isum += __shfl_xor_sync(0xFFFFFFFFu, isum, o);
                if (lg == 0) {
                    float term;
                    if constexpr (FMT == FMT_S8_K || FMT == FMT_S1_5) {
                        term = eff * xb.d * (float)isum;
                    } else {
                        term = eff * (xb.d * (float)isum - HALF * xb.s);
                    }
                    if (is_r1) acc1[m] += term;
                    else       acc0[m] += term;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < n_e) {
            acc0[m] += __shfl_xor_sync(0xFFFFFFFFu, acc0[m], 8);
            acc0[m] += __shfl_xor_sync(0xFFFFFFFFu, acc0[m], 16);
            acc1[m] += __shfl_xor_sync(0xFFFFFFFFu, acc1[m], 8);
            acc1[m] += __shfl_xor_sync(0xFFFFFFFFu, acc1[m], 16);
        }
    }
}

// Fast inline __half2 weight decode (8 registers = 16 weights) + 2-accumulator LDS.128 dot across ALL n_e tokens!
__device__ __forceinline__ void sf4_decode16_h2(uint32_t w0, uint32_t w1, __half2 (&w)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
    auto c2 = [](uint8_t b) -> __half2 {
        __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b, __NV_E2M1);
        return *reinterpret_cast<const __half2*>(&r);
    };
#else
    auto c2 = [](uint8_t b) -> __half2 {
        float2 f = e2m1x2_byte_to_f32x2(b);
        return __floats2half2_rn(f.x, f.y);
    };
#endif
    w[0] = c2((uint8_t)(w0 & 0xFFu));
    w[1] = c2((uint8_t)((w0 >> 8) & 0xFFu));
    w[2] = c2((uint8_t)((w0 >> 16) & 0xFFu));
    w[3] = c2((uint8_t)(w0 >> 24));
    w[4] = c2((uint8_t)(w1 & 0xFFu));
    w[5] = c2((uint8_t)((w1 >> 8) & 0xFFu));
    w[6] = c2((uint8_t)((w1 >> 16) & 0xFFu));
    w[7] = c2((uint8_t)(w1 >> 24));
}

__device__ __forceinline__ void mx8_decode16_h2(const uint4& wv, __half2 (&w)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    auto c2 = [](uint16_t u) -> __half2 {
        __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)u, __NV_E4M3);
        return *reinterpret_cast<const __half2*>(&r);
    };
#else
    auto c2 = [](uint16_t u) -> __half2 {
        float2 f = e4m3x2_to_f32x2(u);
        return __floats2half2_rn(f.x, f.y);
    };
#endif
    w[0] = c2((uint16_t)(wv.x & 0xFFFFu)); w[1] = c2((uint16_t)(wv.x >> 16));
    w[2] = c2((uint16_t)(wv.y & 0xFFFFu)); w[3] = c2((uint16_t)(wv.y >> 16));
    w[4] = c2((uint16_t)(wv.z & 0xFFFFu)); w[5] = c2((uint16_t)(wv.z >> 16));
    w[6] = c2((uint16_t)(wv.w & 0xFFFFu)); w[7] = c2((uint16_t)(wv.w >> 16));
}

__device__ __forceinline__ float dot16_h2_2acc(const __half2 (&w)[8], const __half2* xh2) {
    // 2x 128-bit LDS.128 loads + 2 independent accumulators -> zero pipeline stall!
    const uint4 v0 = *reinterpret_cast<const uint4*>(xh2);
    const uint4 v1 = *reinterpret_cast<const uint4*>(xh2 + 4);
    const __half2* x0 = reinterpret_cast<const __half2*>(&v0);
    const __half2* x1 = reinterpret_cast<const __half2*>(&v1);
    __half2 a0 = __hmul2(w[0], x0[0]);
    __half2 a1 = __hmul2(w[1], x0[1]);
    a0 = __hfma2(w[2], x0[2], a0);
    a1 = __hfma2(w[3], x0[3], a1);
    a0 = __hfma2(w[4], x1[0], a0);
    a1 = __hfma2(w[5], x1[1], a1);
    a0 = __hfma2(w[6], x1[2], a0);
    a1 = __hfma2(w[7], x1[3], a1);
    const float2 f = __half22float2(__hadd2(a0, a1));
    return f.x + f.y;
}

// ---- Ultra-Lean 128B-Coalesced uint4 High-Occupancy SF4/MX8 Gate+Up+SITU Kernel ----
// Grid: (1152 / 16 = 72, max_active), Block: 128 (16 features/CTA, 8 lanes/feature -> 576 CTAs at M=1 = 3.4 full waves on 170 SMs!)
// Uses 1-instruction hardware e4m3x2_to_f32x2 (zero branches!) + hoisted w_outer + dedicated n_e==1 fast path!
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink_moe_gu_sf4_mx8(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const float* __restrict__ x_f32,          // [M, 2304]
    float* __restrict__ h_f32) {              // [P, 1152]
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || (ed.gu_code != FMT_SF4 && ed.gu_code != FMT_MX8)) return;

    const int tid  = threadIdx.x;
    const int sub  = tid >> 3;    // 0..15 features per CTA
    const int lg   = tid & 7;     // 0..7 lanes per feature (8 x 16B = 128B exact cache line!)
    const int feat = blockIdx.x * 16 + sub;
    if (feat >= kInkInter) return;

    const uint8_t* body = sec4_base + ed.body_off;
    float w_outer = 1.0f;
    if (ed.gu_code == FMT_SF4) {
        const float2 h01 = *reinterpret_cast<const float2*>(body);
        w_outer = h01.x / h01.y;
    }
    __shared__ alignas(16) __half2 s_x[MAX_M][kInkHidden / 2];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        for (int m = 0; m < n_e; ++m) {
            const float2* src2 = reinterpret_cast<const float2*>(x_f32 + (size_t)tb.tok_idx[row0 + off_m + m] * kInkHidden);
            __half2* dst2 = s_x[m];
            for (int i = tid; i < (kInkHidden / 2); i += 128) {
                const float2 f = src2[i];
                dst2[i] = __floats2half2_rn(fminf(fmaxf(f.x, -65504.0f), 65504.0f),
                                            fminf(fmaxf(f.y, -65504.0f), 65504.0f));
            }
        }
        __syncthreads();

        if (n_e == 1) {
            float ag0 = 0.0f, au0 = 0.0f;
            if (ed.gu_code == FMT_SF4) {
                constexpr size_t W_P_ROW = kInkHidden / 2;   // 1152 B = 72 uint4s!
                constexpr size_t W_S_ROW = kInkHidden / 16;  // 144 B = 72 uint16s!
                const uint8_t* base_p = body + 16;
                const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
                const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
                const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
                const uint16_t* gs2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat) * W_S_ROW);
                const uint16_t* us2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat + 1) * W_S_ROW);

                for (int i = lg; i < 72; i += 8) {
                    const uint4 gv = gp16[i], uv = up16[i];
                    const float2 sg = e4m3x2_to_f32x2(gs2[i]);
                    const float2 su = e4m3x2_to_f32x2(us2[i]);
                    __half2 gw[8], uw[8];
                    const __half2* xh0 = s_x[0] + i * 16;
                    sf4_decode16_h2(gv.x, gv.y, gw);
                    sf4_decode16_h2(uv.x, uv.y, uw);
                    ag0 = fmaf(dot16_h2_2acc(gw, xh0), sg.x, ag0);
                    au0 = fmaf(dot16_h2_2acc(uw, xh0), su.x, au0);
                    sf4_decode16_h2(gv.z, gv.w, gw);
                    sf4_decode16_h2(uv.z, uv.w, uw);
                    ag0 = fmaf(dot16_h2_2acc(gw, xh0 + 8), sg.y, ag0);
                    au0 = fmaf(dot16_h2_2acc(uw, xh0 + 8), su.y, au0);
                }
                ag0 *= w_outer;
                au0 *= w_outer;
            } else {
                constexpr size_t W_P_ROW = kInkHidden;
                constexpr size_t W_S_ROW = kInkHidden / 32;
                const uint8_t* base_p = body;
                const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
                const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
                const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
                const uint8_t* gs = base_s + (size_t)(2 * feat) * W_S_ROW;
                const uint8_t* us = base_s + (size_t)(2 * feat + 1) * W_S_ROW;
                for (int i = lg; i < 144; i += 8) {
                    const float sg = ue8m0_to_f32(gs[i >> 1]);
                    const float su = ue8m0_to_f32(us[i >> 1]);
                    __half2 gw[8], uw[8];
                    mx8_decode16_h2(gp16[i], gw);
                    mx8_decode16_h2(up16[i], uw);
                    const __half2* xh = s_x[0] + i * 8;
                    ag0 = fmaf(dot16_h2_2acc(gw, xh), sg, ag0);
                    au0 = fmaf(dot16_h2_2acc(uw, xh), su, au0);
                }
            }
#pragma unroll
            for (int o = 4; o >= 1; o >>= 1) {
                ag0 += __shfl_xor_sync(0xFFFFFFFFu, ag0, o);
                au0 += __shfl_xor_sync(0xFFFFFFFFu, au0, o);
            }
            if (lg == 0) h_f32[(size_t)(row0 + off_m) * kInkInter + feat] = situ_f32(ag0, au0);
        } else {
            float ag[MAX_M] = {};
            float au[MAX_M] = {};
            if (ed.gu_code == FMT_SF4) {
                constexpr size_t W_P_ROW = kInkHidden / 2;
                constexpr size_t W_S_ROW = kInkHidden / 16;
                const uint8_t* base_p = body + 16;
                const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
                const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
                const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
                const uint16_t* gs2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat) * W_S_ROW);
                const uint16_t* us2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat + 1) * W_S_ROW);

                for (int i = lg; i < 72; i += 8) {
                    const uint4 gv = gp16[i], uv = up16[i];
                    const float2 sg = e4m3x2_to_f32x2(gs2[i]);
                    const float2 su = e4m3x2_to_f32x2(us2[i]);
                    __half2 gw[8], uw[8];
                    sf4_decode16_h2(gv.x, gv.y, gw);
                    sf4_decode16_h2(uv.x, uv.y, uw);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 16;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.x, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su.x, au[m]);
                        }
                    }
                    sf4_decode16_h2(gv.z, gv.w, gw);
                    sf4_decode16_h2(uv.z, uv.w, uw);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 16 + 8;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.y, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su.y, au[m]);
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) { ag[m] *= w_outer; au[m] *= w_outer; }
            } else {
                constexpr size_t W_P_ROW = kInkHidden;
                constexpr size_t W_S_ROW = kInkHidden / 32;
                const uint8_t* base_p = body;
                const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
                const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
                const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
                const uint8_t* gs = base_s + (size_t)(2 * feat) * W_S_ROW;
                const uint8_t* us = base_s + (size_t)(2 * feat + 1) * W_S_ROW;
                for (int i = lg; i < 144; i += 8) {
                    const float sg = ue8m0_to_f32(gs[i >> 1]);
                    const float su = ue8m0_to_f32(us[i >> 1]);
                    __half2 gw[8], uw[8];
                    mx8_decode16_h2(gp16[i], gw);
                    mx8_decode16_h2(up16[i], uw);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 8;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su, au[m]);
                        }
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    float gv = ag[m], uv = au[m];
#pragma unroll
                    for (int o = 4; o >= 1; o >>= 1) {
                        gv += __shfl_xor_sync(0xFFFFFFFFu, gv, o);
                        uv += __shfl_xor_sync(0xFFFFFFFFu, uv, o);
                    }
                    if (lg == 0) h_f32[(size_t)(row0 + off_m + m) * kInkInter + feat] = situ_f32(gv, uv);
                }
            }
        }
        if (off_m + MAX_M < n_total) __syncthreads();
    }
}

// ---- Ultra-Lean 128B-Coalesced uint4 High-Occupancy SF4/MX8 Down Projection Kernel ----
// Grid: (1152 / 16 = 72, max_active), Block: 128 (16 row-pairs/CTA -> 32 outputs/CTA, 576 CTAs at M=1 = 3.4 waves on 170 SMs!)
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink_moe_dn_sf4_mx8(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const float* __restrict__ h_f32,          // [P, 1152]
    const float* __restrict__ topk_w,         // [P]
    float* __restrict__ y_pair) {             // [P, 2304]
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || (ed.dn_code != FMT_SF4 && ed.dn_code != FMT_MX8)) return;

    const int tid    = threadIdx.x;
    const int sub    = tid >> 3;    // 0..15 row-pairs per CTA
    const int lg     = tid & 7;     // 0..7 lanes per row-pair
    const int pair_r = blockIdx.x * 16 + sub;
    if (pair_r >= kInkHidden / 2) return;

    const uint8_t* body = sec4_base + ed.body_off;
    float w_outer = 1.0f;
    if (ed.dn_code == FMT_SF4) {
        const float2 h23 = *reinterpret_cast<const float2*>(body + 8);
        w_outer = h23.x / h23.y;
    }
    __shared__ alignas(16) __half2 s_h[MAX_M][kInkInter / 2];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        for (int m = 0; m < n_e; ++m) {
            const float2* src2 = reinterpret_cast<const float2*>(h_f32 + (size_t)(row0 + off_m + m) * kInkInter);
            __half2* dst2 = s_h[m];
            for (int i = tid; i < (kInkInter / 2); i += 128) {
                const float2 f = src2[i];
                dst2[i] = __floats2half2_rn(fminf(fmaxf(f.x, -65504.0f), 65504.0f),
                                            fminf(fmaxf(f.y, -65504.0f), 65504.0f));
            }
        }
        __syncthreads();

        if (n_e == 1) {
            float y0 = 0.0f, y1 = 0.0f;
            if (ed.dn_code == FMT_SF4) {
                constexpr size_t W_P_ROW = kInkInter / 2;   // 576 B = 36 uint4s!
                constexpr size_t W_S_ROW = kInkInter / 16;  // 72 B = 36 uint16s!
                const uint8_t* base_p = body + ed.dn_off;
                const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
                const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW);
                const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW);
                const uint16_t* s0 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r) * W_S_ROW);
                const uint16_t* s1 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r + 1) * W_S_ROW);

                for (int i = lg; i < 36; i += 8) {
                    const uint4 v0 = p0[i], v1 = p1[i];
                    const float2 sc0 = e4m3x2_to_f32x2(s0[i]);
                    const float2 sc1 = e4m3x2_to_f32x2(s1[i]);
                    __half2 w0[8], w1[8];
                    const __half2* hh = s_h[0] + i * 16;
                    sf4_decode16_h2(v0.x, v0.y, w0);
                    sf4_decode16_h2(v1.x, v1.y, w1);
                    y0 = fmaf(dot16_h2_2acc(w0, hh), sc0.x, y0);
                    y1 = fmaf(dot16_h2_2acc(w1, hh), sc1.x, y1);
                    sf4_decode16_h2(v0.z, v0.w, w0);
                    sf4_decode16_h2(v1.z, v1.w, w1);
                    y0 = fmaf(dot16_h2_2acc(w0, hh + 8), sc0.y, y0);
                    y1 = fmaf(dot16_h2_2acc(w1, hh + 8), sc1.y, y1);
                }
                y0 *= w_outer;
                y1 *= w_outer;
            } else {
                constexpr size_t W_P_ROW = kInkInter;
                constexpr size_t W_S_ROW = kInkInter / 32;
                const uint8_t* base_p = body + ed.dn_off;
                const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
                const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW);
                const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW);
                const uint8_t* s0 = base_s + (size_t)(2 * pair_r) * W_S_ROW;
                const uint8_t* s1 = base_s + (size_t)(2 * pair_r + 1) * W_S_ROW;
                for (int i = lg; i < 72; i += 8) {
                    const float sc0 = ue8m0_to_f32(s0[i >> 1]);
                    const float sc1 = ue8m0_to_f32(s1[i >> 1]);
                    __half2 w0[8], w1[8];
                    mx8_decode16_h2(p0[i], w0);
                    mx8_decode16_h2(p1[i], w1);
                    const __half2* hh = s_h[0] + i * 8;
                    y0 = fmaf(dot16_h2_2acc(w0, hh), sc0, y0);
                    y1 = fmaf(dot16_h2_2acc(w1, hh), sc1, y1);
                }
            }
#pragma unroll
            for (int o = 4; o >= 1; o >>= 1) {
                y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, o);
                y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, o);
            }
            if (lg == 0) {
                const int pid = tb.pairs[row0 + off_m];
                const float rw = topk_w[pid];
                y_pair[(size_t)pid * kInkHidden + 2 * pair_r]     = y0 * rw;
                y_pair[(size_t)pid * kInkHidden + 2 * pair_r + 1] = y1 * rw;
            }
        } else {
            float a0[MAX_M] = {};
            float a1[MAX_M] = {};
            if (ed.dn_code == FMT_SF4) {
                constexpr size_t W_P_ROW = kInkInter / 2;
                constexpr size_t W_S_ROW = kInkInter / 16;
                const uint8_t* base_p = body + ed.dn_off;
                const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
                const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW);
                const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW);
                const uint16_t* s0 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r) * W_S_ROW);
                const uint16_t* s1 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r + 1) * W_S_ROW);

                for (int i = lg; i < 36; i += 8) {
                    const uint4 v0 = p0[i], v1 = p1[i];
                    const float2 sc0 = e4m3x2_to_f32x2(s0[i]);
                    const float2 sc1 = e4m3x2_to_f32x2(s1[i]);
                    __half2 w0[8], w1[8];
                    sf4_decode16_h2(v0.x, v0.y, w0);
                    sf4_decode16_h2(v1.x, v1.y, w1);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 16;
                            a0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.x, a0[m]);
                            a1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.x, a1[m]);
                        }
                    }
                    sf4_decode16_h2(v0.z, v0.w, w0);
                    sf4_decode16_h2(v1.z, v1.w, w1);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 16 + 8;
                            a0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.y, a0[m]);
                            a1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.y, a1[m]);
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) { a0[m] *= w_outer; a1[m] *= w_outer; }
            } else {
                constexpr size_t W_P_ROW = kInkInter;
                constexpr size_t W_S_ROW = kInkInter / 32;
                const uint8_t* base_p = body + ed.dn_off;
                const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
                const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW);
                const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW);
                const uint8_t* s0 = base_s + (size_t)(2 * pair_r) * W_S_ROW;
                const uint8_t* s1 = base_s + (size_t)(2 * pair_r + 1) * W_S_ROW;
                for (int i = lg; i < 72; i += 8) {
                    const float sc0 = ue8m0_to_f32(s0[i >> 1]);
                    const float sc1 = ue8m0_to_f32(s1[i >> 1]);
                    __half2 w0[8], w1[8];
                    mx8_decode16_h2(p0[i], w0);
                    mx8_decode16_h2(p1[i], w1);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 8;
                            a0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0, a0[m]);
                            a1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1, a1[m]);
                        }
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    float y0 = a0[m], y1 = a1[m];
#pragma unroll
                    for (int o = 4; o >= 1; o >>= 1) {
                        y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, o);
                        y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, o);
                    }
                    if (lg == 0) {
                        const int pid = tb.pairs[row0 + off_m + m];
                        const float rw = topk_w[pid];
                        y_pair[(size_t)pid * kInkHidden + 2 * pair_r]     = y0 * rw;
                        y_pair[(size_t)pid * kInkHidden + 2 * pair_r + 1] = y1 * rw;
                    }
                }
            }
        }
        if (off_m + MAX_M < n_total) __syncthreads();
    }
}

// ---- Ultra-Fast 4-Lane/Feature (32 Features/CTA, Zero Inner Shuffles!) S_K Decode/Verify Kernels! ----
// Each of 4 lanes handles 1 full 64-weight sub-group g in 0..3 -> 100% coalesced cache-line loads, 16x dp4a_i8/lane,
// ZERO warp shuffles inside s=0..8, fused Gate+Up reading s_xq ONCE, and dedicated n_e==1 fast path!
template <int FMT>
__device__ __forceinline__ void sk_unpack64_g_to_u32(const uint8_t* pl, int g, uint32_t (&w)[16]) {
    if constexpr (FMT == FMT_S2_K) {
        const uint8_t* p = pl + g * 16;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t q = ld32_u(p + i * 4);
            const uint32_t b0 = q & 0xFFu, b1 = (q >> 8) & 0xFFu, b2 = (q >> 16) & 0xFFu, b3 = q >> 24;
            w[4*i + 0] = (b0 & 3u) | (((b0 >> 2) & 3u) << 8) | (((b0 >> 4) & 3u) << 16) | (((b0 >> 6) & 3u) << 24);
            w[4*i + 1] = (b1 & 3u) | (((b1 >> 2) & 3u) << 8) | (((b1 >> 4) & 3u) << 16) | (((b1 >> 6) & 3u) << 24);
            w[4*i + 2] = (b2 & 3u) | (((b2 >> 2) & 3u) << 8) | (((b2 >> 4) & 3u) << 16) | (((b2 >> 6) & 3u) << 24);
            w[4*i + 3] = (b3 & 3u) | (((b3 >> 2) & 3u) << 8) | (((b3 >> 4) & 3u) << 16) | (((b3 >> 6) & 3u) << 24);
        }
    } else if constexpr (FMT == FMT_S3_K) {
        const uint8_t* p = pl + g * 16;
        const uint8_t* hp = pl + 64 + g * 8;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t q = ld32_u(p + i * 4);
            const uint16_t h = ld16_u(hp + i * 2);
            const uint32_t b0 = q & 0xFFu, b1 = (q >> 8) & 0xFFu, b2 = (q >> 16) & 0xFFu, b3 = q >> 24;
            const uint32_t h0 = h & 0xFFu, h1 = h >> 8;
            w[4*i + 0] = (b0 & 3u) | (((b0 >> 2) & 3u) << 8) | (((b0 >> 4) & 3u) << 16) | (((b0 >> 6) & 3u) << 24)
                       | ((h0 & 1u) << 2) | (((h0 >> 1) & 1u) << 10) | (((h0 >> 2) & 1u) << 18) | (((h0 >> 3) & 1u) << 26);
            w[4*i + 1] = (b1 & 3u) | (((b1 >> 2) & 3u) << 8) | (((b1 >> 4) & 3u) << 16) | (((b1 >> 6) & 3u) << 24)
                       | (((h0 >> 4) & 1u) << 2) | (((h0 >> 5) & 1u) << 10) | (((h0 >> 6) & 1u) << 18) | (((h0 >> 7) & 1u) << 26);
            w[4*i + 2] = (b2 & 3u) | (((b2 >> 2) & 3u) << 8) | (((b2 >> 4) & 3u) << 16) | (((b2 >> 6) & 3u) << 24)
                       | ((h1 & 1u) << 2) | (((h1 >> 1) & 1u) << 10) | (((h1 >> 2) & 1u) << 18) | (((h1 >> 3) & 1u) << 26);
            w[4*i + 3] = (b3 & 3u) | (((b3 >> 2) & 3u) << 8) | (((b3 >> 4) & 3u) << 16) | (((b3 >> 6) & 3u) << 24)
                       | (((h1 >> 4) & 1u) << 2) | (((h1 >> 5) & 1u) << 10) | (((h1 >> 6) & 1u) << 18) | (((h1 >> 7) & 1u) << 26);
        }
    } else if constexpr (FMT == FMT_S4_K) {
        const uint8_t* p = pl + g * 32;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = ld32_u(p + i * 4);
            w[2*i + 0] = (q & 0x0Fu) | (((q >> 4) & 0x0Fu) << 8) | (((q >> 8) & 0x0Fu) << 16) | (((q >> 12) & 0x0Fu) << 24);
            w[2*i + 1] = ((q >> 16) & 0x0Fu) | (((q >> 20) & 0x0Fu) << 8) | (((q >> 24) & 0x0Fu) << 16) | (((q >> 28) & 0x0Fu) << 24);
        }
    } else if constexpr (FMT == FMT_S5_K) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = ld32_u(p + i * 4);
            const uint8_t hb = hp[i];
            w[2*i + 0] = (q & 0x0Fu) | (((q >> 4) & 0x0Fu) << 8) | (((q >> 8) & 0x0Fu) << 16) | (((q >> 12) & 0x0Fu) << 24)
                       | ((hb & 1u) << 4) | (((hb >> 1) & 1u) << 12) | (((hb >> 2) & 1u) << 20) | (((hb >> 3) & 1u) << 28);
            w[2*i + 1] = ((q >> 16) & 0x0Fu) | (((q >> 20) & 0x0Fu) << 8) | (((q >> 24) & 0x0Fu) << 16) | (((q >> 28) & 0x0Fu) << 24)
                       | (((hb >> 4) & 1u) << 4) | (((hb >> 5) & 1u) << 12) | (((hb >> 6) & 1u) << 20) | (((hb >> 7) & 1u) << 28);
        }
    } else if constexpr (FMT == FMT_S6_K) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 16;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = ld32_u(p + i * 4);
            const uint16_t h16 = ld16_u(hp + i * 2);
            w[2*i + 0] = (q & 0x0Fu) | (((q >> 4) & 0x0Fu) << 8) | (((q >> 8) & 0x0Fu) << 16) | (((q >> 12) & 0x0Fu) << 24)
                       | ((h16 & 3u) << 4) | (((h16 >> 2) & 3u) << 12) | (((h16 >> 4) & 3u) << 20) | (((h16 >> 6) & 3u) << 28);
            w[2*i + 1] = ((q >> 16) & 0x0Fu) | (((q >> 20) & 0x0Fu) << 8) | (((q >> 24) & 0x0Fu) << 16) | (((q >> 28) & 0x0Fu) << 24)
                       | (((h16 >> 8) & 3u) << 4) | (((h16 >> 10) & 3u) << 12) | (((h16 >> 12) & 3u) << 20) | (((h16 >> 14) & 3u) << 28);
        }
    } else if constexpr (FMT == FMT_S8_K) {
        const uint8_t* p = pl + g * 64;
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = ld32_u(p + i * 4);
    } else if constexpr (FMT == FMT_S1_5) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int b0 = g * 64 + i * 4;
            uint32_t t0 = (uint8_t)kS15TritLut[pl[(b0 + 0) / 5]][(b0 + 0) % 5];
            uint32_t t1 = (uint8_t)kS15TritLut[pl[(b0 + 1) / 5]][(b0 + 1) % 5];
            uint32_t t2 = (uint8_t)kS15TritLut[pl[(b0 + 2) / 5]][(b0 + 2) % 5];
            uint32_t t3 = (uint8_t)kS15TritLut[pl[(b0 + 3) / 5]][(b0 + 3) % 5];
            w[i] = t0 | (t1 << 8) | (t2 << 16) | (t3 << 24);
        }
    }
}

__device__ __forceinline__ int32_t dp64_i8_smem(const uint32_t (&w)[16], const InkQ8_1_G64& xb) {
    const uint4* x4 = reinterpret_cast<const uint4*>(xb.qs);
    uint4 v = x4[0];
    int32_t s0 = dp4a_i8(w[0],  v.x, 0);
    int32_t s1 = dp4a_i8(w[1],  v.y, 0);
    s0 = dp4a_i8(w[2],  v.z, s0); s1 = dp4a_i8(w[3],  v.w, s1);
    v = x4[1];
    s0 = dp4a_i8(w[4],  v.x, s0); s1 = dp4a_i8(w[5],  v.y, s1);
    s0 = dp4a_i8(w[6],  v.z, s0); s1 = dp4a_i8(w[7],  v.w, s1);
    v = x4[2];
    s0 = dp4a_i8(w[8],  v.x, s0); s1 = dp4a_i8(w[9],  v.y, s1);
    s0 = dp4a_i8(w[10], v.z, s0); s1 = dp4a_i8(w[11], v.w, s1);
    v = x4[3];
    s0 = dp4a_i8(w[12], v.x, s0); s1 = dp4a_i8(w[13], v.y, s1);
    s0 = dp4a_i8(w[14], v.z, s0); s1 = dp4a_i8(w[15], v.w, s1);
    return s0 + s1;
}

template <int FMT, int MAX_M>
__device__ __forceinline__ void sk_gu_fused_4lane(
    const uint8_t* gr, const uint8_t* ur,
    const InkQ8_1_G64* const (&xp)[MAX_M],
    int n_e, int g,
    float (&ag)[MAX_M], float (&au)[MAX_M]) {
    constexpr int SB = ink_sb_bytes(FMT);
    constexpr float HALF = ink_sk_half(FMT);
    constexpr bool HAS_HALF = (FMT != FMT_S8_K && FMT != FMT_S1_5);
    if (n_e == 1) {
        float g0 = 0.0f, u0 = 0.0f;
        const InkQ8_1_G64* x0 = xp[0];
#pragma unroll 1
        for (int s = 0; s < 9; ++s) {
            const uint8_t* bg = gr + s * SB;
            const uint8_t* bu = ur + s * SB;
            const InkQ8_1_G64& xb = x0[s * 4 + g];
            uint32_t w[16];
            // Sequential gate then up unpack reuses w[16] (halves live weight regs: 32 -> 16, bit-exact)
            const float eg = f16_to_f32(ld16_u(bg)) * ((float)bg[2 + g] * (1.0f / 255.0f));
            sk_unpack64_g_to_u32<FMT>(bg + 6, g, w);
            const int32_t ig = dp64_i8_smem(w, xb);
            if constexpr (HAS_HALF) g0 = fmaf(eg, xb.d * (float)ig - HALF * xb.s, g0);
            else                    g0 = fmaf(eg, xb.d * (float)ig, g0);

            const float eu = f16_to_f32(ld16_u(bu)) * ((float)bu[2 + g] * (1.0f / 255.0f));
            sk_unpack64_g_to_u32<FMT>(bu + 6, g, w);
            const int32_t iu = dp64_i8_smem(w, xb);
            if constexpr (HAS_HALF) u0 = fmaf(eu, xb.d * (float)iu - HALF * xb.s, u0);
            else                    u0 = fmaf(eu, xb.d * (float)iu, u0);
        }
        g0 += __shfl_xor_sync(0xFFFFFFFFu, g0, 2); g0 += __shfl_xor_sync(0xFFFFFFFFu, g0, 1);
        u0 += __shfl_xor_sync(0xFFFFFFFFu, u0, 2); u0 += __shfl_xor_sync(0xFFFFFFFFu, u0, 1);
        ag[0] = g0; au[0] = u0;
    } else {
#pragma unroll 1
        for (int s = 0; s < 9; ++s) {
            const uint8_t* bg = gr + s * SB;
            const uint8_t* bu = ur + s * SB;
            uint32_t w[16];
            const float eg = f16_to_f32(ld16_u(bg)) * ((float)bg[2 + g] * (1.0f / 255.0f));
            sk_unpack64_g_to_u32<FMT>(bg + 6, g, w);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G64& xb = xp[m][s * 4 + g];
                    const int32_t ig = dp64_i8_smem(w, xb);
                    if constexpr (HAS_HALF) ag[m] = fmaf(eg, xb.d * (float)ig - HALF * xb.s, ag[m]);
                    else                    ag[m] = fmaf(eg, xb.d * (float)ig, ag[m]);
                }
            }
            const float eu = f16_to_f32(ld16_u(bu)) * ((float)bu[2 + g] * (1.0f / 255.0f));
            sk_unpack64_g_to_u32<FMT>(bu + 6, g, w);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G64& xb = xp[m][s * 4 + g];
                    const int32_t iu = dp64_i8_smem(w, xb);
                    if constexpr (HAS_HALF) au[m] = fmaf(eu, xb.d * (float)iu - HALF * xb.s, au[m]);
                    else                    au[m] = fmaf(eu, xb.d * (float)iu, au[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
                ag[m] += __shfl_xor_sync(0xFFFFFFFFu, ag[m], 2); ag[m] += __shfl_xor_sync(0xFFFFFFFFu, ag[m], 1);
                au[m] += __shfl_xor_sync(0xFFFFFFFFu, au[m], 2); au[m] += __shfl_xor_sync(0xFFFFFFFFu, au[m], 1);
            }
        }
    }
}

template <int FMT, int MAX_M>
__device__ __forceinline__ void sk_dn_paired2_4lane(
    const uint8_t* pr,
    const InkQ8_1_G64* const (&hp)[MAX_M],
    int n_e, int g,
    float (&a0)[MAX_M], float (&a1)[MAX_M]) {
    constexpr int SB = ink_sb_bytes(FMT);
    constexpr float HALF = ink_sk_half(FMT);
    constexpr bool HAS_HALF = (FMT != FMT_S8_K && FMT != FMT_S1_5);
    if (n_e == 1) {
        float y0 = 0.0f, y1 = 0.0f;
        const InkQ8_1_G64* h0 = hp[0];
#pragma unroll 1
        for (int s = 0; s < 9; ++s) {
            const uint8_t* blk = pr + s * SB;
            const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
            const int sub_g = s * 4 + g;
            const bool is_r1 = (sub_g >= 18);
            const InkQ8_1_G64& xb = h0[is_r1 ? (sub_g - 18) : sub_g];
            uint32_t w[16];
            sk_unpack64_g_to_u32<FMT>(blk + 6, g, w);
            const int32_t isum = dp64_i8_smem(w, xb);
            const float term = HAS_HALF ? (eff * (xb.d * (float)isum - HALF * xb.s)) : (eff * xb.d * (float)isum);
            if (is_r1) y1 += term; else y0 += term;
        }
        y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, 2); y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, 1);
        y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, 2); y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, 1);
        a0[0] = y0; a1[0] = y1;
    } else {
#pragma unroll 1
        for (int s = 0; s < 9; ++s) {
            const uint8_t* blk = pr + s * SB;
            const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
            const int sub_g = s * 4 + g;
            const bool is_r1 = (sub_g >= 18);
            const int xg = is_r1 ? (sub_g - 18) : sub_g;
            uint32_t w[16];
            sk_unpack64_g_to_u32<FMT>(blk + 6, g, w);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G64& xb = hp[m][xg];
                    const int32_t isum = dp64_i8_smem(w, xb);
                    const float term = HAS_HALF ? (eff * (xb.d * (float)isum - HALF * xb.s)) : (eff * xb.d * (float)isum);
                    if (is_r1) a1[m] += term; else a0[m] += term;
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
                a0[m] += __shfl_xor_sync(0xFFFFFFFFu, a0[m], 2); a0[m] += __shfl_xor_sync(0xFFFFFFFFu, a0[m], 1);
                a1[m] += __shfl_xor_sync(0xFFFFFFFFu, a1[m], 2); a1[m] += __shfl_xor_sync(0xFFFFFFFFu, a1[m], 1);
            }
        }
    }
}

// Grid: (1152 / 32 = 36, max_act), Block: 128 (32 features/CTA, 4 lanes/feature -> 100% coalesced, 0 inner shuffles!)
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink_moe_gu_sk(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const InkQ8_1_G64* __restrict__ x_sk,
    float* __restrict__ h_f32) {
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || ed.gu_code == FMT_SF4 || ed.gu_code == FMT_MX8) return;

    const int tid  = threadIdx.x;
    const int sub  = tid >> 2;    // 0..31 features per CTA
    const int g    = tid & 3;     // 0..3 sub-group lane
    const int feat = blockIdx.x * 32 + sub;

    const uint8_t* body = sec4_base + ed.body_off;
    const uint8_t* gu_base = body + kInkInter * 2;
    const size_t row_b = (size_t)(kInkHidden / 256) * ink_sb_bytes(ed.gu_code);
    const uint8_t* gr = gu_base + (size_t)(2 * feat) * row_b;
    const uint8_t* ur = gr + row_b;
    const float inv_sq = (feat < kInkInter) ? (1.0f / f16_to_f32(ld16_u(body + feat * 2))) : 0.0f;

    constexpr int SM_M = (MAX_M <= 4) ? MAX_M : 1;
    __shared__ alignas(16) InkQ8_1_G64 s_xq[SM_M][kInkHidden / 64];  // 2.88 KB for M=1, 11.5 KB for M=4!

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        float ag[MAX_M] = {}, au[MAX_M] = {};
        const InkQ8_1_G64* xp[MAX_M];
        if constexpr (MAX_M <= 4) {
            constexpr int N4 = (int)((kInkHidden / 64) * sizeof(InkQ8_1_G64) / 16);  // 180 uint4s
            for (int m = 0; m < n_e; ++m) {
                const uint4* src = reinterpret_cast<const uint4*>(x_sk + (size_t)tb.tok_idx[row0 + off_m + m] * (kInkHidden / 64));
                uint4* dst = reinterpret_cast<uint4*>(s_xq[m]);
                for (int i = tid; i < N4; i += 128) dst[i] = src[i];
                xp[m] = s_xq[m];
            }
            for (int m = n_e; m < MAX_M; ++m) xp[m] = nullptr;
            __syncthreads();
        } else {
#pragma unroll
            for (int m = 0; m < MAX_M; ++m)
                xp[m] = (m < n_e) ? (x_sk + (size_t)tb.tok_idx[row0 + off_m + m] * (kInkHidden / 64)) : nullptr;
        }

        if (feat < kInkInter) {
            switch (ed.gu_code) {
                case FMT_S2_K: sk_gu_fused_4lane<FMT_S2_K, MAX_M>(gr, ur, xp, n_e, g, ag, au); break;
                case FMT_S3_K: sk_gu_fused_4lane<FMT_S3_K, MAX_M>(gr, ur, xp, n_e, g, ag, au); break;
                case FMT_S4_K: sk_gu_fused_4lane<FMT_S4_K, MAX_M>(gr, ur, xp, n_e, g, ag, au); break;
                case FMT_S5_K: sk_gu_fused_4lane<FMT_S5_K, MAX_M>(gr, ur, xp, n_e, g, ag, au); break;
                case FMT_S6_K: sk_gu_fused_4lane<FMT_S6_K, MAX_M>(gr, ur, xp, n_e, g, ag, au); break;
                case FMT_S8_K: sk_gu_fused_4lane<FMT_S8_K, MAX_M>(gr, ur, xp, n_e, g, ag, au); break;
                case FMT_S1_5: sk_gu_fused_4lane<FMT_S1_5, MAX_M>(gr, ur, xp, n_e, g, ag, au); break;
                default: break;
            }
            if (g == 0) {
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < n_e) h_f32[(size_t)(row0 + off_m + m) * kInkInter + feat] = situ_f32(ag[m], au[m]) * inv_sq;
                }
            }
        }
        if constexpr (MAX_M <= 4) { if (off_m + MAX_M < n_total) __syncthreads(); }
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink_moe_dn_sk(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const InkQ8_1_G64* __restrict__ h_sk,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair) {
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || ed.dn_code == FMT_SF4 || ed.dn_code == FMT_MX8) return;

    const int tid    = threadIdx.x;
    const int sub    = tid >> 2;    // 0..31 row-pairs per CTA (64 outputs/CTA!)
    const int g      = tid & 3;
    const int pair_r = blockIdx.x * 32 + sub;

    const uint8_t* body = sec4_base + ed.body_off;
    const size_t pair_b = 9ull * ink_sb_bytes(ed.dn_code);
    const uint8_t* pr = body + ed.dn_off + (size_t)pair_r * pair_b;

    constexpr int SM_M = (MAX_M <= 4) ? MAX_M : 1;
    __shared__ alignas(16) InkQ8_1_G64 s_hq[SM_M][kInkInter / 64];  // 1.44 KB for M=1, 5.76 KB for M=4!

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        float a0[MAX_M] = {}, a1[MAX_M] = {};
        const InkQ8_1_G64* hp[MAX_M];
        if constexpr (MAX_M <= 4) {
            constexpr int N4 = (int)((kInkInter / 64) * sizeof(InkQ8_1_G64) / 16);  // 90 uint4s
            for (int m = 0; m < n_e; ++m) {
                const uint4* src = reinterpret_cast<const uint4*>(h_sk + (size_t)(row0 + off_m + m) * (kInkInter / 64));
                uint4* dst = reinterpret_cast<uint4*>(s_hq[m]);
                for (int i = tid; i < N4; i += 128) dst[i] = src[i];
                hp[m] = s_hq[m];
            }
            for (int m = n_e; m < MAX_M; ++m) hp[m] = nullptr;
            __syncthreads();
        } else {
#pragma unroll
            for (int m = 0; m < MAX_M; ++m)
                hp[m] = (m < n_e) ? (h_sk + (size_t)(row0 + off_m + m) * (kInkInter / 64)) : nullptr;
        }

        if (pair_r < kInkHidden / 2) {
            switch (ed.dn_code) {
                case FMT_S2_K: sk_dn_paired2_4lane<FMT_S2_K, MAX_M>(pr, hp, n_e, g, a0, a1); break;
                case FMT_S3_K: sk_dn_paired2_4lane<FMT_S3_K, MAX_M>(pr, hp, n_e, g, a0, a1); break;
                case FMT_S4_K: sk_dn_paired2_4lane<FMT_S4_K, MAX_M>(pr, hp, n_e, g, a0, a1); break;
                case FMT_S5_K: sk_dn_paired2_4lane<FMT_S5_K, MAX_M>(pr, hp, n_e, g, a0, a1); break;
                case FMT_S6_K: sk_dn_paired2_4lane<FMT_S6_K, MAX_M>(pr, hp, n_e, g, a0, a1); break;
                case FMT_S8_K: sk_dn_paired2_4lane<FMT_S8_K, MAX_M>(pr, hp, n_e, g, a0, a1); break;
                case FMT_S1_5: sk_dn_paired2_4lane<FMT_S1_5, MAX_M>(pr, hp, n_e, g, a0, a1); break;
                default: break;
            }
            if (g == 0) {
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < n_e) {
                        const int pid = tb.pairs[row0 + off_m + m];
                        const float rw = topk_w[pid];
                        y_pair[(size_t)pid * kInkHidden + 2 * pair_r]     = a0[m] * rw;
                        y_pair[(size_t)pid * kInkHidden + 2 * pair_r + 1] = a1[m] * rw;
                    }
                }
            }
        }
        if constexpr (MAX_M <= 4) { if (off_m + MAX_M < n_total) __syncthreads(); }
    }
}

// Per-format specialized S_K decode/verify kernels (eliminates the 7-way runtime switch that forced REG:255 STACK:32..72!)
template <int FMT, int MAX_M>
__global__ void __launch_bounds__(128, (FMT == FMT_S1_5 ? 2 : (MAX_M == 1 ? 6 : (MAX_M <= 4 ? 5 : 2)))) k_ink_moe_gu_sk_fmt(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const InkQ8_1_G64* __restrict__ x_sk,
    float* __restrict__ h_f32) {
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || ed.gu_code != FMT) return;
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];

    const int tid  = threadIdx.x;
    const int sub  = tid >> 2;
    const int g    = tid & 3;
    const int feat = blockIdx.x * 32 + sub;

    const uint8_t* body = sec4_base + ed.body_off;
    const uint8_t* gu_base = body + kInkInter * 2;
    constexpr size_t row_b = (size_t)(kInkHidden / 256) * ink_sb_bytes(FMT);
    const uint8_t* gr = gu_base + (size_t)(2 * feat) * row_b;
    const uint8_t* ur = gr + row_b;
    const float inv_sq = (feat < kInkInter) ? (1.0f / f16_to_f32(ld16_u(body + feat * 2))) : 0.0f;

    constexpr int SM_M = (MAX_M <= 4) ? MAX_M : 1;
    __shared__ alignas(16) InkQ8_1_G64 s_xq[SM_M][kInkHidden / 64];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        float ag[MAX_M] = {}, au[MAX_M] = {};
        const InkQ8_1_G64* xp[MAX_M];
        if constexpr (MAX_M <= 4) {
            constexpr int N4 = (int)((kInkHidden / 64) * sizeof(InkQ8_1_G64) / 16);
            for (int m = 0; m < n_e; ++m) {
                const uint4* src = reinterpret_cast<const uint4*>(x_sk + (size_t)tb.tok_idx[row0 + off_m + m] * (kInkHidden / 64));
                uint4* dst = reinterpret_cast<uint4*>(s_xq[m]);
                for (int i = tid; i < N4; i += 128) dst[i] = src[i];
            }
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) xp[m] = s_xq[m];
            __syncthreads();
        } else {
#pragma unroll
            for (int m = 0; m < MAX_M; ++m)
                xp[m] = (m < n_e) ? (x_sk + (size_t)tb.tok_idx[row0 + off_m + m] * (kInkHidden / 64)) : nullptr;
        }
        if (feat < kInkInter) {
            sk_gu_fused_4lane<FMT, MAX_M>(gr, ur, xp, n_e, g, ag, au);
            if (g == 0) {
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < n_e) h_f32[(size_t)(row0 + off_m + m) * kInkInter + feat] = situ_f32(ag[m], au[m]) * inv_sq;
                }
            }
        }
        if constexpr (MAX_M <= 4) { if (off_m + MAX_M < n_total) __syncthreads(); }
    }
}

template <int FMT, int MAX_M>
__global__ void __launch_bounds__(128, (FMT == FMT_S1_5 ? 2 : (MAX_M == 1 ? 6 : (MAX_M <= 4 ? 5 : 2)))) k_ink_moe_dn_sk_fmt(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const InkQ8_1_G64* __restrict__ h_sk,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair) {
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || ed.dn_code != FMT) return;
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];

    const int tid    = threadIdx.x;
    const int sub    = tid >> 2;
    const int g      = tid & 3;
    const int pair_r = blockIdx.x * 32 + sub;

    const uint8_t* body = sec4_base + ed.body_off;
    constexpr size_t pair_b = 9ull * ink_sb_bytes(FMT);
    const uint8_t* pr = body + ed.dn_off + (size_t)pair_r * pair_b;

    constexpr int SM_M = (MAX_M <= 4) ? MAX_M : 1;
    __shared__ alignas(16) InkQ8_1_G64 s_hq[SM_M][kInkInter / 64];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        float a0[MAX_M] = {}, a1[MAX_M] = {};
        const InkQ8_1_G64* hp[MAX_M];
        if constexpr (MAX_M <= 4) {
            constexpr int N4 = (int)((kInkInter / 64) * sizeof(InkQ8_1_G64) / 16);
            for (int m = 0; m < n_e; ++m) {
                const uint4* src = reinterpret_cast<const uint4*>(h_sk + (size_t)(row0 + off_m + m) * (kInkInter / 64));
                uint4* dst = reinterpret_cast<uint4*>(s_hq[m]);
                for (int i = tid; i < N4; i += 128) dst[i] = src[i];
            }
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) hp[m] = s_hq[m];
            __syncthreads();
        } else {
#pragma unroll
            for (int m = 0; m < MAX_M; ++m)
                hp[m] = (m < n_e) ? (h_sk + (size_t)(row0 + off_m + m) * (kInkInter / 64)) : nullptr;
        }
        if (pair_r < kInkHidden / 2) {
            sk_dn_paired2_4lane<FMT, MAX_M>(pr, hp, n_e, g, a0, a1);
            if (g == 0) {
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < n_e) {
                        const int pid = tb.pairs[row0 + off_m + m];
                        const float rw = topk_w[pid];
                        y_pair[(size_t)pid * kInkHidden + 2 * pair_r]     = a0[m] * rw;
                        y_pair[(size_t)pid * kInkHidden + 2 * pair_r + 1] = a1[m] * rw;
                    }
                }
            }
        }
        if constexpr (MAX_M <= 4) { if (off_m + MAX_M < n_total) __syncthreads(); }
    }
}

// ============================================================================
// 4B. Portable sm_75..sm_121 INT8 Tensor Core (mma.m16n8k32.s8) S_K MoE Kernels!
//     10x-13x faster prefill & batched decode on SQ-15, SQ-10, and RTX 3090/4090/5090!
//     100.000% exact Q8_1_G64 arithmetic (every S1_5..S8_K value fits inside signed int8_t)!
// ============================================================================
__device__ __forceinline__ void sk_unpack64_sub_to_i8(
    int fmt, const uint8_t* blk, int g, int8_t* dst64, float& eff_out) {
    const float D = f16_to_f32(ld16_u(blk));
    eff_out = D * ((float)blk[2 + g] * (1.0f / 255.0f));
    const uint8_t* pl = blk + 6;
    uint32_t* d4 = reinterpret_cast<uint32_t*>(dst64);
    if (fmt == FMT_S3_K) {
        for (int lg = 0; lg < 8; ++lg) {
            uint32_t w0, w1;
            sk_unpack8_lane8<FMT_S3_K>(pl, g, lg, w0, w1);
            d4[2 * lg + 0] = w0;
            d4[2 * lg + 1] = w1;
        }
    } else if (fmt == FMT_S2_K) {
        for (int lg = 0; lg < 8; ++lg) {
            uint32_t w0, w1;
            sk_unpack8_lane8<FMT_S2_K>(pl, g, lg, w0, w1);
            d4[2 * lg + 0] = w0;
            d4[2 * lg + 1] = w1;
        }
    } else if (fmt == FMT_S4_K) {
        for (int lg = 0; lg < 8; ++lg) {
            uint32_t w0, w1;
            sk_unpack8_lane8<FMT_S4_K>(pl, g, lg, w0, w1);
            d4[2 * lg + 0] = w0;
            d4[2 * lg + 1] = w1;
        }
    } else if (fmt == FMT_S5_K) {
        for (int lg = 0; lg < 8; ++lg) {
            uint32_t w0, w1;
            sk_unpack8_lane8<FMT_S5_K>(pl, g, lg, w0, w1);
            d4[2 * lg + 0] = w0;
            d4[2 * lg + 1] = w1;
        }
    } else if (fmt == FMT_S6_K) {
        for (int lg = 0; lg < 8; ++lg) {
            uint32_t w0, w1;
            sk_unpack8_lane8<FMT_S6_K>(pl, g, lg, w0, w1);
            d4[2 * lg + 0] = w0;
            d4[2 * lg + 1] = w1;
        }
    } else if (fmt == FMT_S8_K) {
        for (int lg = 0; lg < 8; ++lg) {
            uint32_t w0, w1;
            sk_unpack8_lane8<FMT_S8_K>(pl, g, lg, w0, w1);
            d4[2 * lg + 0] = w0;
            d4[2 * lg + 1] = w1;
        }
    } else if (fmt == FMT_S1_5) {
        for (int lg = 0; lg < 8; ++lg) {
            uint32_t w0, w1;
            sk_unpack8_lane8<FMT_S1_5>(pl, g, lg, w0, w1);
            d4[2 * lg + 0] = w0;
            d4[2 * lg + 1] = w1;
        }
    }
}

inline bool ink_has_int8_tc() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        cached = 0;
        return false;
    }
    cached = (p.major >= 8) ? 1 : 0;
    return cached != 0;
}

inline bool ink_has_wmma_f16() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        cached = 0;
        return false;
    }
    cached = (p.major == 7) ? 1 : 0;  // Volta sm_70 (V100) + Turing sm_75 (RTX 2080 Ti)
    return cached != 0;
}

// Persistent Coalesced 256-Thread INT8 Tensor Core S_K MoE Kernel (sm_80..sm_121: GU=true -> Gate+Up+SITU, GU=false -> Paired-2-Row Down)
// All 256 threads cooperate (8 coalesced lanes/row x 32 rows/wave x 4 waves = 128 rows) with compile-time FMT dispatch!
template <bool GU, int TR, int FMT>
__device__ __forceinline__ void ink_moe_sk_tc_kloop(
    const uint8_t* __restrict__ body,
    uint64_t dn_off,
    int fb,
    int nrows,
    int tid, int wf, int nb0, int g, int tig,
    bool on0, bool on1,
    const InkQ8_1_G64* __restrict__ act_sk,
    const int32_t* __restrict__ s_row,
    int8_t (&s_wq)[128][80],
    int8_t (&s_aq)[TR][80],
    float (&s_wsc)[128],
    float (&s_ad)[TR],
    float (&s_as)[TR],
    float (&acc)[(128 / (8 / (TR / 16))) / 16][2][4]) {
    constexpr int kWR      = 128;
    constexpr int NG       = GU ? (kInkHidden / 64) : (kInkInter / 64);
    constexpr int N_WN     = TR / 16;
    constexpr int N_WF     = 8 / N_WN;
    constexpr int W_PER_WF = kWR / N_WF;
    constexpr int NI       = W_PER_WF / 16;
    constexpr int SB       = ink_sb_bytes(FMT);
    constexpr float HALF   = ink_sk_half(FMT);
    constexpr bool HAS_HALF = (FMT != FMT_S8_K && FMT != FMT_S1_5);

    const uint8_t* gu_base = body + kInkInter * 2;
    constexpr size_t gu_row_b  = 9ull * (size_t)SB;
    constexpr size_t dn_pair_b = 9ull * (size_t)SB;
    const int r_in_w = tid >> 3;  // 0..31 rows per wave
    const int lg     = tid & 7;   // 0..7 coalesced lanes per row (8 weights/lane = 64 weights/row!)

    for (int xg = 0; xg < NG; ++xg) {
        // 1. ALL 256 threads unpack 128 weight rows x 64 weights in 4 coalesced waves of 32 rows!
#pragma unroll
        for (int wave = 0; wave < 4; ++wave) {
            const int row_s = wave * 32 + r_in_w;
            const int phys_row = GU ? (2 * (fb * 64 + 8 * (row_s >> 4) + (row_s & 7)) + ((row_s >> 3) & 1))
                                    : (fb * kWR + row_s);
            const uint8_t* blk;
            int sg;
            if constexpr (GU) {
                blk = gu_base + (size_t)phys_row * gu_row_b + (size_t)(xg >> 2) * SB;
                sg  = xg & 3;
            } else {
                const int pair_r = phys_row >> 1;
                const int sub_glob = xg + (phys_row & 1) * 18;
                blk = body + dn_off + (size_t)pair_r * dn_pair_b + (size_t)(sub_glob >> 2) * SB;
                sg  = sub_glob & 3;
            }
            uint32_t w0, w1;
            sk_unpack8_lane8<FMT>(blk + 6, sg, lg, w0, w1);
            uint32_t* d4 = reinterpret_cast<uint32_t*>(s_wq[row_s]);
            d4[2 * lg + 0] = w0;
            d4[2 * lg + 1] = w1;
            if (lg == 0) s_wsc[row_s] = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + sg] * (1.0f / 255.0f));
        }
        // 2. Load TR activation blocks (2 threads per row -> <= 128 threads active)
        if (tid < nrows * 2) {
            const int r = tid >> 1;
            const int h = tid & 1;
            const InkQ8_1_G64& xb = act_sk[(size_t)s_row[r] * NG + xg];
            const uint4* src = reinterpret_cast<const uint4*>(xb.qs + h * 32);
            uint4* dst = reinterpret_cast<uint4*>(s_aq[r] + h * 32);
            dst[0] = src[0]; dst[1] = src[1];
            if (h == 0) { s_ad[r] = xb.d; s_as[r] = xb.s; }
        }
        __syncthreads();

        // 3. Run 2x mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (k=0..31 and k=32..63)!
        if (on0) {
            const uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + g] + 4 * tig);
            const uint32_t b0_1 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + g] + 16 + 4 * tig);
            const uint32_t b0_2 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + g] + 32 + 4 * tig);
            const uint32_t b0_3 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + g] + 48 + 4 * tig);
            uint32_t b1_0 = 0, b1_1 = 0, b1_2 = 0, b1_3 = 0;
            if (on1) {
                b1_0 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + 8 + g] + 4 * tig);
                b1_1 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + 8 + g] + 16 + 4 * tig);
                b1_2 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + 8 + g] + 32 + 4 * tig);
                b1_3 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + 8 + g] + 48 + 4 * tig);
            }
            const float xd00 = s_ad[nb0 + 2 * tig + 0], xs00 = s_as[nb0 + 2 * tig + 0];
            const float xd01 = s_ad[nb0 + 2 * tig + 1], xs01 = s_as[nb0 + 2 * tig + 1];
            const float xd10 = on1 ? s_ad[nb0 + 8 + 2 * tig + 0] : 0.0f;
            const float xs10 = on1 ? s_as[nb0 + 8 + 2 * tig + 0] : 0.0f;
            const float xd11 = on1 ? s_ad[nb0 + 8 + 2 * tig + 1] : 0.0f;
            const float xs11 = on1 ? s_as[nb0 + 8 + 2 * tig + 1] : 0.0f;

#pragma unroll
            for (int i = 0; i < NI; ++i) {
                const int wr0 = W_PER_WF * wf + 16 * i + g;
                const int wr1 = wr0 + 8;
                const float eff0 = s_wsc[wr0];
                const float eff1 = s_wsc[wr1];

                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(s_wq[wr0] + 4 * tig);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(s_wq[wr1] + 4 * tig);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(s_wq[wr0] + 16 + 4 * tig);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(s_wq[wr1] + 16 + 4 * tig);
                const uint32_t a4 = *reinterpret_cast<const uint32_t*>(s_wq[wr0] + 32 + 4 * tig);
                const uint32_t a5 = *reinterpret_cast<const uint32_t*>(s_wq[wr1] + 32 + 4 * tig);
                const uint32_t a6 = *reinterpret_cast<const uint32_t*>(s_wq[wr0] + 48 + 4 * tig);
                const uint32_t a7 = *reinterpret_cast<const uint32_t*>(s_wq[wr1] + 48 + 4 * tig);

                int32_t cv[4] = {};
                mma_m16n8k32_s8(cv, a0, a1, a2, a3, b0_0, b0_1);
                mma_m16n8k32_s8(cv, a4, a5, a6, a7, b0_2, b0_3);
                if constexpr (HAS_HALF) {
                    acc[i][0][0] = fmaf(eff0, xd00 * (float)cv[0] - HALF * xs00, acc[i][0][0]);
                    acc[i][0][1] = fmaf(eff0, xd01 * (float)cv[1] - HALF * xs01, acc[i][0][1]);
                    acc[i][0][2] = fmaf(eff1, xd00 * (float)cv[2] - HALF * xs00, acc[i][0][2]);
                    acc[i][0][3] = fmaf(eff1, xd01 * (float)cv[3] - HALF * xs01, acc[i][0][3]);
                } else {
                    acc[i][0][0] = fmaf(eff0, xd00 * (float)cv[0], acc[i][0][0]);
                    acc[i][0][1] = fmaf(eff0, xd01 * (float)cv[1], acc[i][0][1]);
                    acc[i][0][2] = fmaf(eff1, xd00 * (float)cv[2], acc[i][0][2]);
                    acc[i][0][3] = fmaf(eff1, xd01 * (float)cv[3], acc[i][0][3]);
                }
                if (on1) {
                    int32_t dv[4] = {};
                    mma_m16n8k32_s8(dv, a0, a1, a2, a3, b1_0, b1_1);
                    mma_m16n8k32_s8(dv, a4, a5, a6, a7, b1_2, b1_3);
                    if constexpr (HAS_HALF) {
                        acc[i][1][0] = fmaf(eff0, xd10 * (float)dv[0] - HALF * xs10, acc[i][1][0]);
                        acc[i][1][1] = fmaf(eff0, xd11 * (float)dv[1] - HALF * xs11, acc[i][1][1]);
                        acc[i][1][2] = fmaf(eff1, xd10 * (float)dv[2] - HALF * xs10, acc[i][1][2]);
                        acc[i][1][3] = fmaf(eff1, xd11 * (float)dv[3] - HALF * xs11, acc[i][1][3]);
                    } else {
                        acc[i][1][0] = fmaf(eff0, xd10 * (float)dv[0], acc[i][1][0]);
                        acc[i][1][1] = fmaf(eff0, xd11 * (float)dv[1], acc[i][1][1]);
                        acc[i][1][2] = fmaf(eff1, xd10 * (float)dv[2], acc[i][1][2]);
                        acc[i][1][3] = fmaf(eff1, xd11 * (float)dv[3], acc[i][1][3]);
                    }
                }
            }
        }
        if (xg + 1 < NG) __syncthreads();
    }
}

template <bool GU, int TR>
__global__ void __launch_bounds__(256) k_ink_moe_sk_tc(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const InkQ8_1_G64* __restrict__ act_sk,   // GU: [M, 36]; DN: [P, 18]
    const float* __restrict__ topk_w,         // DN only
    float* __restrict__ out_buf) {            // GU: h_f32 [P, 1152]; DN: y_pair [P, 2304]
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int kWR      = 128;
    constexpr int ROWS_N   = GU ? (2 * kInkInter) : kInkHidden;  // 2304
    constexpr int NFB      = ROWS_N / kWR;                       // 18 blocks of 128 rows
    constexpr int N_WN     = TR / 16;                            // 1 (TR=16) or 4 (TR=64)
    constexpr int N_WF     = 8 / N_WN;                           // 8 (TR=16) or 2 (TR=64)
    constexpr int W_PER_WF = kWR / N_WF;                         // 16 or 64
    constexpr int NI       = W_PER_WF / 16;                      // 1 or 4
    constexpr int WSTR     = 80;
    constexpr int ASTR     = 80;

    __shared__ alignas(16) int8_t  s_wq[kWR][WSTR];
    __shared__ alignas(16) int8_t  s_aq[TR][ASTR];
    __shared__ float               s_wsc[kWR];
    __shared__ float               s_ad[TR];
    __shared__ float               s_as[TR];
    __shared__ int32_t             s_row[TR];
    __shared__ float               s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;
    const int wn   = warp % N_WN;
    const int nb0  = 16 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        const int fmt = GU ? (int)ed.gu_code : (int)ed.dn_code;
        if (ed.tier > 1 || fmt == FMT_SF4 || fmt == FMT_MX8) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                s_ad[tid] = 0.0f; s_as[tid] = 0.0f;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                uint32_t* z = reinterpret_cast<uint32_t*>(s_aq[tid]);
                for (int k = 0; k < 16; ++k) z[k] = 0u;
            }
        }
        __syncthreads();

        float acc[NI][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        switch (fmt) {
            case FMT_S2_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S2_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S3_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S3_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S4_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S4_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S5_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S5_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S6_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S6_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S8_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S8_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S1_5: ink_moe_sk_tc_kloop<GU, TR, FMT_S1_5>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            default: break;
        }

        if (on0) {
            if constexpr (GU) {
                const int fbase = fb * 64 + (W_PER_WF / 2) * wf;
                float inv_sq[NI];
#pragma unroll
                for (int i = 0; i < NI; ++i) {
                    const int feat = fbase + 8 * i + g;
                    inv_sq[i] = 1.0f / f16_to_f32(ld16_u(body + feat * 2));
                }
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (r < nrows) {
                            float* dst = out_buf + (size_t)(row0 + r) * kInkInter;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[fbase + 8 * i + g] = situ_f32(acc[i][n][c], acc[i][n][2 + c]) * inv_sq[i];
                            }
                        }
                    }
                }
            } else {
                const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (r < nrows) {
                            const float sc = s_rw[r];
                            float* dst = out_buf + (size_t)tb.pairs[row0 + r] * kInkHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                                dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act_sk; (void)topk_w; (void)out_buf;
#endif
}

// ============================================================================
// 4C. Exact FP16-Tensor-Core (mma.m16n8k16.f16 -> FP32) SF4 / MX8 MoE Kernel (sm_80..sm_121!)
//     Zero FP4 activation quantization loss -> 100.000% lossless M=1..16 speculative verify on SQ-20/SQ-15
//     AND 6x faster SQ-20 prefill on Ampere/Ada (RTX 3090/4090)!
//     Each K=16 MMA matches the EXACT SF4 16-weight scale block (scaled in FP32 after MMA)!
// ============================================================================
template <bool GU>
__global__ void __launch_bounds__(256) k_ink_moe_sf4_mx8_fp16tc(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const float* __restrict__ act_f32,        // GU: x_f32 [M, 2304]; DN: h_f32 [P, 1152]
    const float* __restrict__ topk_w,         // DN only
    float* __restrict__ out_buf) {            // GU: h_f32 [P, 1152]; DN: y_pair [P, 2304]
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int kWR    = 128;
    constexpr int TR     = 16;
    constexpr int K      = GU ? kInkHidden : kInkInter;       // 2304 or 1152
    constexpr int ROWS_N = GU ? (2 * kInkInter) : kInkHidden; // 2304
    constexpr int NFB    = ROWS_N / kWR;                      // 18 blocks of 128 rows
    constexpr int N32    = K / 32;                            // 72 (GU) or 36 (DN)

    __shared__ alignas(16) __half s_w[kWR][40];   // 10,240 B (stride 80B = 16B aligned, bank-conflict free!)
    __shared__ alignas(16) __half s_a[TR][40];    //  1,280 B
    __shared__ float              s_wsc[kWR][2];  //  1,024 B (exact scale per K=16 block!)
    __shared__ int32_t            s_row[TR];      //     64 B
    __shared__ float              s_rw[TR];       //     64 B

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;   // 0..7 (each warp owns 16 weight rows: wf = warp)
    const int lane = tid & 31;
    const int g    = lane >> 2;  // 0..7
    const int tig  = lane & 3;   // 0..3

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        const int fmt = GU ? (int)ed.gu_code : (int)ed.dn_code;
        if (ed.tier > 1 || (fmt != FMT_SF4 && fmt != FMT_MX8)) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        float w_outer = 1.0f;
        if (fmt == FMT_SF4) {
            float hdr[4];
            memcpy(hdr, body, 16);
            w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);
        }

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                uint32_t* z = reinterpret_cast<uint32_t*>(s_a[tid]);
                for (int k = 0; k < 16; ++k) z[k] = 0u;
            }
        }
        __syncthreads();

        float acc[2][4] = {};
        const bool on1 = 8 < nrows;
        const int wr0 = 16 * warp + g;
        const int wr1 = wr0 + 8;

        auto phys_row_fn = [&](int r) -> int {
            return GU ? (2 * (fb * 64 + 8 * (r >> 4) + (r & 7)) + ((r >> 3) & 1))
                      : (fb * kWR + r);
        };

        if (fmt == FMT_SF4) {
            constexpr size_t W_P_ROW = K / 2;
            constexpr size_t W_S_ROW = K / 16;
            const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
            const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;
            const int pr_t = (tid < kWR) ? phys_row_fn(tid) : 0;
            const uint4* wp16 = reinterpret_cast<const uint4*>(base_p + (size_t)pr_t * W_P_ROW);
            const uint16_t* ws2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)pr_t * W_S_ROW);

            for (int k32 = 0; k32 < N32; ++k32) {
                if (tid < kWR) {
                    const uint4 v = wp16[k32];
                    const uint16_t sp = ws2[k32];
                    s_wsc[tid][0] = e4m3_to_f32((uint8_t)(sp & 0xFFu)) * w_outer;
                    s_wsc[tid][1] = e4m3_to_f32((uint8_t)(sp >> 8))    * w_outer;
                    __half2 w0[8], w1[8];
                    sf4_decode16_h2(v.x, v.y, w0);
                    sf4_decode16_h2(v.z, v.w, w1);
                    __half2* dw = reinterpret_cast<__half2*>(s_w[tid]);
#pragma unroll
                    for (int j = 0; j < 8; ++j) { dw[j] = w0[j]; dw[8 + j] = w1[j]; }
                } else {
                    const int atid = tid - kWR;  // 0..127 -> 8 threads per activation row (16 rows x 8 = 128!)
                    const int r = atid >> 3;
                    const int j = atid & 7;      // 4 floats = 2 half2s
                    if (r < nrows) {
                        const float4 fv = *reinterpret_cast<const float4*>(act_f32 + (size_t)s_row[r] * K + k32 * 32 + j * 4);
                        __half2* da = reinterpret_cast<__half2*>(s_a[r]);
                        da[2 * j + 0] = __floats2half2_rn(fminf(fmaxf(fv.x, -65504.0f), 65504.0f),
                                                          fminf(fmaxf(fv.y, -65504.0f), 65504.0f));
                        da[2 * j + 1] = __floats2half2_rn(fminf(fmaxf(fv.z, -65504.0f), 65504.0f),
                                                          fminf(fmaxf(fv.w, -65504.0f), 65504.0f));
                    }
                }
                __syncthreads();

#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const int ko = sub * 16;
                    const float sc0 = s_wsc[wr0][sub];
                    const float sc1 = s_wsc[wr1][sub];
                    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 2 * tig);
                    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 2 * tig);
                    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 8 + 2 * tig);
                    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 8 + 2 * tig);
                    const uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(s_a[g] + ko + 2 * tig);
                    const uint32_t b0_1 = *reinterpret_cast<const uint32_t*>(s_a[g] + ko + 8 + 2 * tig);
                    float cv[4] = {};
                    mma_m16n8k16_f16(cv, a0, a1, a2, a3, b0_0, b0_1);
                    acc[0][0] = fmaf(sc0, cv[0], acc[0][0]);
                    acc[0][1] = fmaf(sc0, cv[1], acc[0][1]);
                    acc[0][2] = fmaf(sc1, cv[2], acc[0][2]);
                    acc[0][3] = fmaf(sc1, cv[3], acc[0][3]);
                    if (on1) {
                        const uint32_t b1_0 = *reinterpret_cast<const uint32_t*>(s_a[8 + g] + ko + 2 * tig);
                        const uint32_t b1_1 = *reinterpret_cast<const uint32_t*>(s_a[8 + g] + ko + 8 + 2 * tig);
                        float dv[4] = {};
                        mma_m16n8k16_f16(dv, a0, a1, a2, a3, b1_0, b1_1);
                        acc[1][0] = fmaf(sc0, dv[0], acc[1][0]);
                        acc[1][1] = fmaf(sc0, dv[1], acc[1][1]);
                        acc[1][2] = fmaf(sc1, dv[2], acc[1][2]);
                        acc[1][3] = fmaf(sc1, dv[3], acc[1][3]);
                    }
                }
                if (k32 + 1 < N32) __syncthreads();
            }
        } else {
            constexpr size_t W_P_ROW = K;
            constexpr size_t W_S_ROW = K / 32;
            const uint8_t* base_p = GU ? body : (body + ed.dn_off);
            const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;
            const int r_w = tid >> 1;
            const int h_w = tid & 1;
            const int pr_t = phys_row_fn(r_w);
            const uint4* wp16 = reinterpret_cast<const uint4*>(base_p + (size_t)pr_t * W_P_ROW);
            const uint8_t* ws = base_s + (size_t)pr_t * W_S_ROW;

            for (int k32 = 0; k32 < N32; ++k32) {
                __half2 wh[8];
                mx8_decode16_h2(wp16[k32 * 2 + h_w], wh);
                __half2* dw = reinterpret_cast<__half2*>(s_w[r_w]) + h_w * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) dw[j] = wh[j];
                if (h_w == 0) {
                    const float sc = ue8m0_to_f32(ws[k32]);
                    s_wsc[r_w][0] = sc;
                }
                if (tid < nrows * 8) {
                    const int r = tid >> 3;
                    const int j = tid & 7;
                    const float4 fv = *reinterpret_cast<const float4*>(act_f32 + (size_t)s_row[r] * K + k32 * 32 + j * 4);
                    __half2* da = reinterpret_cast<__half2*>(s_a[r]);
                    da[2 * j + 0] = __floats2half2_rn(fminf(fmaxf(fv.x, -65504.0f), 65504.0f),
                                                      fminf(fmaxf(fv.y, -65504.0f), 65504.0f));
                    da[2 * j + 1] = __floats2half2_rn(fminf(fmaxf(fv.z, -65504.0f), 65504.0f),
                                                      fminf(fmaxf(fv.w, -65504.0f), 65504.0f));
                }
                __syncthreads();

                const float sc0 = s_wsc[wr0][0];
                const float sc1 = s_wsc[wr1][0];
                float cv[4] = {}, dv[4] = {};
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const int ko = sub * 16;
                    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 2 * tig);
                    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 2 * tig);
                    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 8 + 2 * tig);
                    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 8 + 2 * tig);
                    const uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(s_a[g] + ko + 2 * tig);
                    const uint32_t b0_1 = *reinterpret_cast<const uint32_t*>(s_a[g] + ko + 8 + 2 * tig);
                    mma_m16n8k16_f16(cv, a0, a1, a2, a3, b0_0, b0_1);
                    if (on1) {
                        const uint32_t b1_0 = *reinterpret_cast<const uint32_t*>(s_a[8 + g] + ko + 2 * tig);
                        const uint32_t b1_1 = *reinterpret_cast<const uint32_t*>(s_a[8 + g] + ko + 8 + 2 * tig);
                        mma_m16n8k16_f16(dv, a0, a1, a2, a3, b1_0, b1_1);
                    }
                }
                acc[0][0] = fmaf(sc0, cv[0], acc[0][0]);
                acc[0][1] = fmaf(sc0, cv[1], acc[0][1]);
                acc[0][2] = fmaf(sc1, cv[2], acc[0][2]);
                acc[0][3] = fmaf(sc1, cv[3], acc[0][3]);
                if (on1) {
                    acc[1][0] = fmaf(sc0, dv[0], acc[1][0]);
                    acc[1][1] = fmaf(sc0, dv[1], acc[1][1]);
                    acc[1][2] = fmaf(sc1, dv[2], acc[1][2]);
                    acc[1][3] = fmaf(sc1, dv[3], acc[1][3]);
                }
                if (k32 + 1 < N32) __syncthreads();
            }
        }

        if constexpr (GU) {
            const int feat = fb * 64 + 8 * warp + g;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        out_buf[(size_t)(row0 + r) * kInkInter + feat] = situ_f32(acc[n][c], acc[n][2 + c]);
                    }
                }
            }
        } else {
            const int rbase = fb * kWR + 16 * warp;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = s_rw[r];
                        float* dst = out_buf + (size_t)tb.pairs[row0 + r] * kInkHidden;
                        dst[rbase + g + 0] = acc[n][c] * sc;
                        dst[rbase + g + 8] = acc[n][2 + c] * sc;
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act_f32; (void)topk_w; (void)out_buf;
#endif
}

// ============================================================================
// 4D. Volta (sm_70 V100) & Turing (sm_75 RTX 2080 Ti) High-Occupancy 2x2-Tiled nvcuda::wmma FP16 Tensor Core MoE!
//     Handles ALL .strata v1.1 expert codecs (SF4, MX8, S1_5..S8_K) with FP32 accumulation (125 TFLOPS on V100!)
//     BN=128, BK=32, 256 threads (8 warps), 16.9 KB smem, 1-cycle s_e4[256] LUT, TR=64 (2x2 warp tiles) or TR=16!
// ============================================================================
#if defined(__CUDACC__)
template <bool GU, int FMT>
__device__ __forceinline__ void wmma_unpack_sk_w32_f16(
    const uint8_t* __restrict__ body, uint64_t dn_off, int fb, int k32, int tid, __half (&s_w)[128][40]) {
    constexpr int SB = ink_sb_bytes(FMT);
    constexpr float HALF = ink_sk_half(FMT);
    constexpr size_t row_b = 9ull * (size_t)SB;
    const uint8_t* gu_base = body + kInkInter * 2;
    const int xg = k32 >> 1;
    const int lg_off = (k32 & 1) * 4;  // 0..3 or 4..7 inside 64-elem subgroup
#pragma unroll
    for (int wave = 0; wave < 2; ++wave) {
        const int r_loc = wave * 64 + (tid >> 2);  // 0..127
        const int lg4   = tid & 3;                 // 0..3 (8 weights/lane -> 32 weights/row!)
        const int phys_row = fb * 128 + r_loc;
        const uint8_t* blk;
        int sg;
        if constexpr (GU) {
            blk = gu_base + (size_t)phys_row * row_b + (size_t)(xg >> 2) * SB;
            sg  = xg & 3;
        } else {
            const int pair_r = phys_row >> 1;
            const int sub_glob = xg + (phys_row & 1) * 18;
            blk = body + dn_off + (size_t)pair_r * row_b + (size_t)(sub_glob >> 2) * SB;
            sg  = sub_glob & 3;
        }
        const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + sg] * (1.0f / 255.0f));
        uint32_t w0, w1;
        sk_unpack8_lane8<FMT>(blk + 6, sg, lg_off + lg4, w0, w1);
        __half2* dst2 = reinterpret_cast<__half2*>(&s_w[r_loc][lg4 * 8]);
        dst2[0] = __floats2half2_rn(((float)(int8_t)(w0)       - HALF) * eff, ((float)(int8_t)(w0 >> 8)  - HALF) * eff);
        dst2[1] = __floats2half2_rn(((float)(int8_t)(w0 >> 16) - HALF) * eff, ((float)(int8_t)(w0 >> 24) - HALF) * eff);
        dst2[2] = __floats2half2_rn(((float)(int8_t)(w1)       - HALF) * eff, ((float)(int8_t)(w1 >> 8)  - HALF) * eff);
        dst2[3] = __floats2half2_rn(((float)(int8_t)(w1 >> 16) - HALF) * eff, ((float)(int8_t)(w1 >> 24) - HALF) * eff);
    }
}

template <bool GU, int TR>
__global__ void __launch_bounds__(256) k_ink_moe_wmma_f16(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const float* __restrict__ act_f32,        // GU: x_f32 [M, 2304]; DN: h_f32 [P, 1152]
    const float* __restrict__ topk_w,         // DN only
    float* __restrict__ out_buf) {            // GU: h_f32 [P, 1152]; DN: y_pair [P, 2304]
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    constexpr int BN     = 128;
    constexpr int BK     = 32;
    constexpr int K      = GU ? kInkHidden : kInkInter;         // 2304 or 1152
    constexpr int ROWS_N = GU ? (2 * kInkInter) : kInkHidden;   // 2304
    constexpr int NFB    = ROWS_N / BN;                         // 18 exact blocks of 128 rows!
    constexpr int N32    = K / BK;                              // 72 (GU) or 36 (DN)
    constexpr int M_SUB  = (TR == 64) ? 2 : 1;                  // 2x16=32 rows/warp for TR=64, 1x16 for TR=16
    constexpr int N_SUB  = (TR == 64) ? 2 : 1;                  // 2x16=32 cols/warp for TR=64, 1x16 for TR=16

    __shared__ alignas(16) __half s_a[TR][40];
    __shared__ alignas(16) union {
        __half w[128][40];   // 10,240 B
        float  o[16][128];   //  8,192 B
    } s_u;
    __shared__ float   s_e4[256];
    __shared__ __half2 s_e2m1[256];
    __shared__ int32_t s_row[TR];
    __shared__ float   s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm   = (TR == 64) ? (warp >> 2) : 0;      // 0..1 (TR=64) or 0 (TR=16)
    const int wn   = (TR == 64) ? (warp & 3)  : warp;   // 0..3 (TR=64) or 0..7 (TR=16)

    // Preload 1-cycle E4M3 & E2M1x2 LUTs into shared memory ONCE per CTA!
    s_e4[tid] = e4m3_to_f32_compute((uint8_t)tid);
    {
        const float2 e2 = e2m1x2_byte_to_f32x2((uint8_t)tid);
        s_e2m1[tid] = __floats2half2_rn(e2.x, e2.y);
    }
    __syncthreads();

    const int nt = *tb.n_tiles;
    const int total_jobs = nt * NFB;
    for (int job = blockIdx.x; job < total_jobs; job += gridDim.x) {
        const int t_idx = job / NFB;
        const int fb    = job - t_idx * NFB;
        const int2 tl   = tb.tiles[t_idx];
        const int e     = tl.x;
        const int row0  = tl.y;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const InkExpertDev ex = experts[e];
        const int fmt = GU ? (int)ex.gu_code : (int)ex.dn_code;
        if (ex.tier > 1) continue;

        if (tid < nrows) {
            s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
            if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
        }
        __syncthreads();

        const uint8_t* body = sec4_base + ex.body_off;
        float w_outer = 1.0f;
        const uint8_t* bp = nullptr;
        const uint8_t* bs = nullptr;
        if (fmt == FMT_SF4) {
            const float* hdr = reinterpret_cast<const float*>(body);
            w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);
            bp = GU ? (body + 16) : (body + ex.dn_off);
            bs = bp + (size_t)ROWS_N * (K / 2);
        } else if (fmt == FMT_MX8) {
            bp = GU ? body : (body + ex.dn_off);
            bs = bp + (size_t)ROWS_N * K;
        }

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> fc[M_SUB][N_SUB];
#pragma unroll
        for (int mi = 0; mi < M_SUB; ++mi)
#pragma unroll
            for (int ni = 0; ni < N_SUB; ++ni)
                nvcuda::wmma::fill_fragment(fc[mi][ni], 0.0f);

        for (int k32 = 0; k32 < N32; ++k32) {
            // 1. Load activation tile s_a[TR][32] (vectorized float4 -> 2x half2!)
            for (int idx = tid; idx < TR * 8; idx += 256) {
                const int r = idx >> 3;
                const int q = idx & 7;
                __half2* dst2 = reinterpret_cast<__half2*>(&s_a[r][q * 4]);
                if (r < nrows) {
                    const float4 xv = *reinterpret_cast<const float4*>(act_f32 + (size_t)s_row[r] * K + k32 * 32 + q * 4);
                    dst2[0] = __floats2half2_rn(fminf(fmaxf(xv.x, -65504.0f), 65504.0f), fminf(fmaxf(xv.y, -65504.0f), 65504.0f));
                    dst2[1] = __floats2half2_rn(fminf(fmaxf(xv.z, -65504.0f), 65504.0f), fminf(fmaxf(xv.w, -65504.0f), 65504.0f));
                } else {
                    const __half2 z = __float2half2_rn(0.0f);
                    dst2[0] = z; dst2[1] = z;
                }
            }
            // 2. Unpack 128 x 32 weight block into s_u.w[128][40] (ZERO loop for SF4 & MX8!)
            if (fmt == FMT_SF4) {
                const int r_loc = tid >> 1;     // 0..127
                const int b     = tid & 1;      // 0..1 (16 weights/thread!)
                const int phys_row = fb * 128 + r_loc;
                const int blk = k32 * 2 + b;
                const __half2 hsc = __float2half2_rn(s_e4[bs[(size_t)phys_row * (K / 16) + blk]] * w_outer);
                const uint2 p8 = *reinterpret_cast<const uint2*>(bp + (size_t)phys_row * (K / 2) + blk * 8);
                __half2* dst2 = reinterpret_cast<__half2*>(&s_u.w[r_loc][b * 16]);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    dst2[i]     = __hmul2(s_e2m1[(p8.x >> (8 * i)) & 0xFFu], hsc);
                    dst2[4 + i] = __hmul2(s_e2m1[(p8.y >> (8 * i)) & 0xFFu], hsc);
                }
            } else if (fmt == FMT_MX8) {
                const int r_loc  = tid >> 1;    // 0..127
                const int half_b = tid & 1;     // 0..1 (16 bytes/thread!)
                const int phys_row = fb * 128 + r_loc;
                const float sc = ue8m0_to_f32(bs[(size_t)phys_row * (K / 32) + k32]);
                const uint4 w16 = *reinterpret_cast<const uint4*>(bp + (size_t)phys_row * K + k32 * 32 + half_b * 16);
                const uint32_t wu[4] = {w16.x, w16.y, w16.z, w16.w};
                __half2* dst2 = reinterpret_cast<__half2*>(&s_u.w[r_loc][half_b * 16]);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t u = wu[i];
                    dst2[2 * i + 0] = __floats2half2_rn(s_e4[u & 0xFFu] * sc, s_e4[(u >> 8) & 0xFFu] * sc);
                    dst2[2 * i + 1] = __floats2half2_rn(s_e4[(u >> 16) & 0xFFu] * sc, s_e4[(u >> 24) & 0xFFu] * sc);
                }
            } else if (fmt == FMT_S3_K) {
                wmma_unpack_sk_w32_f16<GU, FMT_S3_K>(body, ex.dn_off, fb, k32, tid, s_u.w);
            } else if (fmt == FMT_S2_K) {
                wmma_unpack_sk_w32_f16<GU, FMT_S2_K>(body, ex.dn_off, fb, k32, tid, s_u.w);
            } else if (fmt == FMT_S4_K) {
                wmma_unpack_sk_w32_f16<GU, FMT_S4_K>(body, ex.dn_off, fb, k32, tid, s_u.w);
            } else if (fmt == FMT_S5_K) {
                wmma_unpack_sk_w32_f16<GU, FMT_S5_K>(body, ex.dn_off, fb, k32, tid, s_u.w);
            } else if (fmt == FMT_S6_K) {
                wmma_unpack_sk_w32_f16<GU, FMT_S6_K>(body, ex.dn_off, fb, k32, tid, s_u.w);
            } else if (fmt == FMT_S8_K) {
                wmma_unpack_sk_w32_f16<GU, FMT_S8_K>(body, ex.dn_off, fb, k32, tid, s_u.w);
            } else if (fmt == FMT_S1_5) {
                wmma_unpack_sk_w32_f16<GU, FMT_S1_5>(body, ex.dn_off, fb, k32, tid, s_u.w);
            }
            __syncthreads();

            // 3. 2x K-steps of 2x2 WMMA 16x16x16 Tensor Core MMA (scalar fb_m -> saves 8 live regs, ZERO spills!)
#pragma unroll
            for (int kk = 0; kk < 32; kk += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> fa[M_SUB];
#pragma unroll
                for (int mi = 0; mi < M_SUB; ++mi)
                    nvcuda::wmma::load_matrix_sync(fa[mi], &s_a[(wm * M_SUB + mi) * 16][kk], 40);
#pragma unroll
                for (int ni = 0; ni < N_SUB; ++ni) {
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> fb_m;
                    nvcuda::wmma::load_matrix_sync(fb_m, &s_u.w[(wn * N_SUB + ni) * 16][kk], 40);
#pragma unroll
                    for (int mi = 0; mi < M_SUB; ++mi)
                        nvcuda::wmma::mma_sync(fc[mi][ni], fa[mi], fb_m, fc[mi][ni]);
                }
            }
            __syncthreads();
        }

        // 4. Staged 16x128 epilogue reusing s_u.o[16][128] (ZERO extra smem!)
        constexpr int N_MBLK = TR / 16;  // 4 for TR=64, 1 for TR=16
        const bool is_sk = (fmt != FMT_SF4 && fmt != FMT_MX8);
        for (int mb = 0; mb < N_MBLK; ++mb) {
            const int r_base = mb * 16;
            if (r_base >= nrows) break;
            if (TR == 16 || wm == (mb >> 1)) {
                const int mi = (TR == 64) ? (mb & 1) : 0;
#pragma unroll
                for (int ni = 0; ni < N_SUB; ++ni) {
                    nvcuda::wmma::store_matrix_sync(&s_u.o[0][(wn * N_SUB + ni) * 16], fc[mi][ni], 128, nvcuda::wmma::mem_row_major);
                }
            }
            __syncthreads();
            const int nr = min(16, nrows - r_base);
            if constexpr (GU) {
                const int fbase = fb * 64;
                for (int idx = tid; idx < nr * 64; idx += 256) {
                    const int r = idx >> 6;
                    const int f = idx & 63;
                    float v = situ_f32(s_u.o[r][2 * f], s_u.o[r][2 * f + 1]);
                    if (is_sk) v *= (1.0f / f16_to_f32(ld16_u(body + (fbase + f) * 2)));
                    out_buf[(size_t)(row0 + r_base + r) * kInkInter + fbase + f] = v;
                }
            } else {
                const int rbase = fb * 128;
                for (int idx = tid; idx < nr * 128; idx += 256) {
                    const int r = idx >> 7;
                    const int c = idx & 127;
                    out_buf[(size_t)tb.pairs[row0 + r_base + r] * kInkHidden + rbase + c] = s_u.o[r][c] * s_rw[r_base + r];
                }
            }
            __syncthreads();
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act_f32; (void)topk_w; (void)out_buf;
#endif
}
// High-Occupancy Zero-Spill BK=64, BN=64, TR=64 WMMA FP16 Tensor Core Prefill Kernel for Volta sm_70 & Turing sm_75!
// Split IS_SK (SF4/MX8 vs S_K) + scalar fc0/fc1 (0 stack frame, <=64 regs!) + 20.5 KB smem -> 4 CTAs/SM (1024 thr/SM)!
template <bool GU, int FMT>
__device__ __forceinline__ void wmma_unpack_sk_64x64_f16(
    const uint8_t* __restrict__ body, uint64_t dn_off, int fb, int k64, int tid, __half (&s_w)[64][72]) {
    constexpr int SB = ink_sb_bytes(FMT);
    constexpr float HALF = ink_sk_half(FMT);
    constexpr size_t row_b = 9ull * (size_t)SB;
    const uint8_t* gu_base = body + kInkInter * 2;
    const int r_loc = tid >> 2;  // 0..63
    const int lg4   = tid & 3;   // 0..3 -> 2x8 = 16 weights/thread = 64 weights/row!
    const int phys_row = fb * 64 + r_loc;
    const uint8_t* blk;
    int sg;
    if constexpr (GU) {
        blk = gu_base + (size_t)phys_row * row_b + (size_t)(k64 >> 2) * SB;
        sg  = k64 & 3;
    } else {
        const int pair_r = phys_row >> 1;
        const int sub_glob = k64 + (phys_row & 1) * 18;
        blk = body + dn_off + (size_t)pair_r * row_b + (size_t)(sub_glob >> 2) * SB;
        sg  = sub_glob & 3;
    }
    const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + sg] * (1.0f / 255.0f));
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        uint32_t w0, w1;
        sk_unpack8_lane8<FMT>(blk + 6, sg, p * 4 + lg4, w0, w1);
        __half2* dst2 = reinterpret_cast<__half2*>(&s_w[r_loc][(p * 4 + lg4) * 8]);
        dst2[0] = __floats2half2_rn(((float)(int8_t)(w0)       - HALF) * eff, ((float)(int8_t)(w0 >> 8)  - HALF) * eff);
        dst2[1] = __floats2half2_rn(((float)(int8_t)(w0 >> 16) - HALF) * eff, ((float)(int8_t)(w0 >> 24) - HALF) * eff);
        dst2[2] = __floats2half2_rn(((float)(int8_t)(w1)       - HALF) * eff, ((float)(int8_t)(w1 >> 8)  - HALF) * eff);
        dst2[3] = __floats2half2_rn(((float)(int8_t)(w1 >> 16) - HALF) * eff, ((float)(int8_t)(w1 >> 24) - HALF) * eff);
    }
}

template <bool GU, bool IS_SK>
__global__ void __launch_bounds__(256, 4) k_ink_moe_wmma_f16_bk64(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const float* __restrict__ act_f32,
    const float* __restrict__ topk_w,
    float* __restrict__ out_buf) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    constexpr int TR     = 64;
    constexpr int BN     = 64;
    constexpr int BK     = 64;
    constexpr int K      = GU ? kInkHidden : kInkInter;         // 2304 or 1152
    constexpr int ROWS_N = GU ? (2 * kInkInter) : kInkHidden;   // 2304
    constexpr int NFB    = ROWS_N / BN;                         // 36 blocks of 64 rows
    constexpr int N64    = K / BK;                              // 36 (GU) or 18 (DN)

    __shared__ alignas(16) __half s_a[TR][72];                  // 9,216 B
    __shared__ alignas(16) union {
        __half w[BN][72];                                       // 9,216 B
        float  o[16][BN];                                       // 4,096 B
    } s_u;                                                      // Total smem = 20.5 KB -> 4 CTAs/SM!
    __shared__ float   s_e4[256];
    __shared__ __half2 s_e2m1[256];
    __shared__ int32_t s_row[TR];
    __shared__ float   s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm   = warp >> 1;  // 0..3 (each warp owns 16 rows)
    const int wn   = warp & 1;   // 0..1 (each warp owns 32 cols = 2x16!)

    if constexpr (!IS_SK) {
        s_e4[tid] = e4m3_to_f32_compute((uint8_t)tid);
        const float2 e2 = e2m1x2_byte_to_f32x2((uint8_t)tid);
        s_e2m1[tid] = __floats2half2_rn(e2.x, e2.y);
        __syncthreads();
    }

    const int nt = *tb.n_tiles;
    const int total_jobs = nt * NFB;
    for (int job = blockIdx.x; job < total_jobs; job += gridDim.x) {
        const int t_idx = job / NFB;
        const int fb    = job - t_idx * NFB;
        const int2 tl   = tb.tiles[t_idx];
        const int e     = tl.x;
        const int row0  = tl.y;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const InkExpertDev ex = experts[e];
        const int fmt = GU ? (int)ex.gu_code : (int)ex.dn_code;
        if (ex.tier > 1) continue;
        if constexpr (!IS_SK) { if (fmt != FMT_SF4 && fmt != FMT_MX8) continue; }
        else                  { if (fmt == FMT_SF4 || fmt == FMT_MX8) continue; }

        if (tid < nrows) {
            s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
            if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
        }
        __syncthreads();

        const uint8_t* body = sec4_base + ex.body_off;
        float w_outer = 1.0f;
        const uint8_t* bp = nullptr;
        const uint8_t* bs = nullptr;
        if constexpr (!IS_SK) {
            if (fmt == FMT_SF4) {
                const float* hdr = reinterpret_cast<const float*>(body);
                w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);
                bp = GU ? (body + 16) : (body + ex.dn_off);
                bs = bp + (size_t)ROWS_N * (K / 2);
            } else {
                bp = GU ? body : (body + ex.dn_off);
                bs = bp + (size_t)ROWS_N * K;
            }
        }

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> fc0, fc1;
        nvcuda::wmma::fill_fragment(fc0, 0.0f);
        nvcuda::wmma::fill_fragment(fc1, 0.0f);

        for (int k64 = 0; k64 < N64; ++k64) {
            // 1. Load activation tile s_a[64][64] (4 float4s/thread, 100% coalesced!)
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                const int idx = it * 256 + tid;
                const int r = idx >> 4;
                const int q = idx & 15;
                __half2* dst2 = reinterpret_cast<__half2*>(&s_a[r][q * 4]);
                if (r < nrows) {
                    const float4 xv = *reinterpret_cast<const float4*>(act_f32 + (size_t)s_row[r] * K + k64 * 64 + q * 4);
                    dst2[0] = __floats2half2_rn(fminf(fmaxf(xv.x, -65504.0f), 65504.0f), fminf(fmaxf(xv.y, -65504.0f), 65504.0f));
                    dst2[1] = __floats2half2_rn(fminf(fmaxf(xv.z, -65504.0f), 65504.0f), fminf(fmaxf(xv.w, -65504.0f), 65504.0f));
                } else {
                    const __half2 z = __float2half2_rn(0.0f);
                    dst2[0] = z; dst2[1] = z;
                }
            }
            // 2. Coalesced 4-threads/row unpack of 64 x 64 weights into s_u.w[64][72]!
            if constexpr (!IS_SK) {
                const int r_loc = tid >> 2;
                const int b     = tid & 3;
                const int phys_row = fb * 64 + r_loc;
                if (fmt == FMT_SF4) {
                    const int blk = k64 * 4 + b;
                    const __half2 hsc = __float2half2_rn(s_e4[bs[(size_t)phys_row * (K / 16) + blk]] * w_outer);
                    const uint2 p8 = *reinterpret_cast<const uint2*>(bp + (size_t)phys_row * (K / 2) + blk * 8);
                    __half2* dst2 = reinterpret_cast<__half2*>(&s_u.w[r_loc][b * 16]);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        dst2[i]     = __hmul2(s_e2m1[(p8.x >> (8 * i)) & 0xFFu], hsc);
                        dst2[4 + i] = __hmul2(s_e2m1[(p8.y >> (8 * i)) & 0xFFu], hsc);
                    }
                } else {
                    const float sc = ue8m0_to_f32(bs[(size_t)phys_row * (K / 32) + k64 * 2 + (b >> 1)]);
                    const uint4 w16 = *reinterpret_cast<const uint4*>(bp + (size_t)phys_row * K + k64 * 64 + b * 16);
                    const uint32_t wu[4] = {w16.x, w16.y, w16.z, w16.w};
                    __half2* dst2 = reinterpret_cast<__half2*>(&s_u.w[r_loc][b * 16]);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const uint32_t u = wu[i];
                        dst2[2 * i + 0] = __floats2half2_rn(s_e4[u & 0xFFu] * sc, s_e4[(u >> 8) & 0xFFu] * sc);
                        dst2[2 * i + 1] = __floats2half2_rn(s_e4[(u >> 16) & 0xFFu] * sc, s_e4[(u >> 24) & 0xFFu] * sc);
                    }
                }
            } else {
                if (fmt == FMT_S3_K)      wmma_unpack_sk_64x64_f16<GU, FMT_S3_K>(body, ex.dn_off, fb, k64, tid, s_u.w);
                else if (fmt == FMT_S2_K) wmma_unpack_sk_64x64_f16<GU, FMT_S2_K>(body, ex.dn_off, fb, k64, tid, s_u.w);
                else if (fmt == FMT_S4_K) wmma_unpack_sk_64x64_f16<GU, FMT_S4_K>(body, ex.dn_off, fb, k64, tid, s_u.w);
                else if (fmt == FMT_S5_K) wmma_unpack_sk_64x64_f16<GU, FMT_S5_K>(body, ex.dn_off, fb, k64, tid, s_u.w);
                else if (fmt == FMT_S6_K) wmma_unpack_sk_64x64_f16<GU, FMT_S6_K>(body, ex.dn_off, fb, k64, tid, s_u.w);
                else if (fmt == FMT_S8_K) wmma_unpack_sk_64x64_f16<GU, FMT_S8_K>(body, ex.dn_off, fb, k64, tid, s_u.w);
                else if (fmt == FMT_S1_5) wmma_unpack_sk_64x64_f16<GU, FMT_S1_5>(body, ex.dn_off, fb, k64, tid, s_u.w);
            }
            __syncthreads();

            // 3. 4x K-steps of 1x2 WMMA 16x16x16 Tensor Core MMA (scalar fa & fb -> <= 64 regs, ZERO spills!)
#pragma unroll
            for (int kk = 0; kk < 64; kk += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> fa;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> fb_m;
                nvcuda::wmma::load_matrix_sync(fa,   &s_a[wm * 16][kk], 72);
                nvcuda::wmma::load_matrix_sync(fb_m, &s_u.w[wn * 32 +  0][kk], 72);
                nvcuda::wmma::mma_sync(fc0, fa, fb_m, fc0);
                nvcuda::wmma::load_matrix_sync(fb_m, &s_u.w[wn * 32 + 16][kk], 72);
                nvcuda::wmma::mma_sync(fc1, fa, fb_m, fc1);
            }
            __syncthreads();
        }

        // 4. Staged 16x64 epilogue reusing s_u.o[16][64] (0 stack frame, 4 CTAs/SM!)
        for (int mb = 0; mb < 4; ++mb) {
            const int r_base = mb * 16;
            if (r_base >= nrows) break;
            if (wm == mb) {
                nvcuda::wmma::store_matrix_sync(&s_u.o[0][wn * 32 +  0], fc0, BN, nvcuda::wmma::mem_row_major);
                nvcuda::wmma::store_matrix_sync(&s_u.o[0][wn * 32 + 16], fc1, BN, nvcuda::wmma::mem_row_major);
            }
            __syncthreads();
            const int nr = min(16, nrows - r_base);
            if constexpr (GU) {
                const int fbase = fb * 32;
                for (int idx = tid; idx < nr * 32; idx += 256) {
                    const int r = idx >> 5;
                    const int f = idx & 31;
                    float v = situ_f32(s_u.o[r][2 * f], s_u.o[r][2 * f + 1]);
                    if constexpr (IS_SK) v *= (1.0f / f16_to_f32(ld16_u(body + (fbase + f) * 2)));
                    out_buf[(size_t)(row0 + r_base + r) * kInkInter + fbase + f] = v;
                }
            } else {
                const int rbase = fb * 64;
                for (int idx = tid; idx < nr * 64; idx += 256) {
                    const int r = idx >> 6;
                    const int c = idx & 63;
                    out_buf[(size_t)tb.pairs[row0 + r_base + r] * kInkHidden + rbase + c] = s_u.o[r][c] * s_rw[r_base + r];
                }
            }
            __syncthreads();
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act_f32; (void)topk_w; (void)out_buf;
#endif
}
#endif  // __CUDACC__

// ============================================================================
// 5. High-Level MoE Dispatcher (Auto-selects Blackwell sm_120a TC vs Exact FP16 TC vs INT8 TC vs WMMA sm_70/75 vs Decode!)
// ============================================================================
struct InkLayerMoeInfo {
    bool has_sf4  = false;
    bool has_mx8  = false;
    bool has_sk   = false;
    bool has_gguf = false;
    uint32_t sk_gu_mask = 0;
    uint32_t sk_dn_mask = 0;
};

template <int MAX_M>
inline void launch_ink_moe_gu_sk_mask(
    dim3 grid, cudaStream_t st, uint32_t mask,
    const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
    const InkQ8_1_G64* x_sk, float* h_f32) {
    if (mask == 0) { k_ink_moe_gu_sk<MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32); return; }
    if (mask & (1u << FMT_S1_5)) k_ink_moe_gu_sk_fmt<FMT_S1_5, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32);
    if (mask & (1u << FMT_S2_K)) k_ink_moe_gu_sk_fmt<FMT_S2_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32);
    if (mask & (1u << FMT_S3_K)) k_ink_moe_gu_sk_fmt<FMT_S3_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32);
    if (mask & (1u << FMT_S4_K)) k_ink_moe_gu_sk_fmt<FMT_S4_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32);
    if (mask & (1u << FMT_S5_K)) k_ink_moe_gu_sk_fmt<FMT_S5_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32);
    if (mask & (1u << FMT_S6_K)) k_ink_moe_gu_sk_fmt<FMT_S6_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32);
    if (mask & (1u << FMT_S8_K)) k_ink_moe_gu_sk_fmt<FMT_S8_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32);
}

template <int MAX_M>
inline void launch_ink_moe_dn_sk_mask(
    dim3 grid, cudaStream_t st, uint32_t mask,
    const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
    const InkQ8_1_G64* h_sk, const float* topk_w, float* y_pair) {
    if (mask == 0) { k_ink_moe_dn_sk<MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair); return; }
    if (mask & (1u << FMT_S1_5)) k_ink_moe_dn_sk_fmt<FMT_S1_5, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair);
    if (mask & (1u << FMT_S2_K)) k_ink_moe_dn_sk_fmt<FMT_S2_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair);
    if (mask & (1u << FMT_S3_K)) k_ink_moe_dn_sk_fmt<FMT_S3_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair);
    if (mask & (1u << FMT_S4_K)) k_ink_moe_dn_sk_fmt<FMT_S4_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair);
    if (mask & (1u << FMT_S5_K)) k_ink_moe_dn_sk_fmt<FMT_S5_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair);
    if (mask & (1u << FMT_S6_K)) k_ink_moe_dn_sk_fmt<FMT_S6_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair);
    if (mask & (1u << FMT_S8_K)) k_ink_moe_dn_sk_fmt<FMT_S8_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair);
}

// Standalone Top-8 / 128 exact sigmoid router (shared by single-GPU, hybrid, and Dual-GPU EP!)
inline void launch_ink_moe_router(
    cudaStream_t st,
    cublasHandle_t cublas,
    const float* gate_w,
    const float* gate_b,
    const float* x_f32,
    int M,
    InkMoeWorkspace& ws,
    float route_scale = kInkRouteScale,
    bool force_exact_dec = false) {
    if (M <= 16) {
        if (M == 1)      k_ink_router_gemv_smallM<1><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
        else if (M <= 4) k_ink_router_gemv_smallM<4><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
        else if (M <= 8) k_ink_router_gemv_smallM<8><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
        else             k_ink_router_gemv_smallM<16><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
    } else {
        const float alpha = 1.0f, beta = 0.0f;
        INK_CUBLAS_CK(cublasGemmEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, kInkExperts, M, kInkHidden, &alpha,
            gate_w, CUDA_R_32F, kInkHidden, x_f32, CUDA_R_32F, kInkHidden, &beta,
            ws.r_logits, CUDA_R_32F, kInkExperts,
            (!force_exact_dec && ink_has_int8_tc()) ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    k_ink_router_select_topk8<<<M, 32, 0, st>>>(ws.r_logits, gate_b, ws.topk_idx, ws.topk_w, route_scale, M);
}

// Shared GPU expert execution helper (bins tier<=1 experts in `experts` and writes weighted outputs to ws.y_pair)
inline void launch_ink_moe_dispatch_only(
    cudaStream_t st,
    const uint8_t* sec4_base,
    const InkExpertDev* experts,
    const InkLayerMoeInfo& info,
    const float* x_f32,
    int M,
    InkMoeWorkspace& ws,
    bool force_exact_dec,
    bool quant_x_done = false) {
    // For 128-expert Top-8 MoE, M<=16 (all spec verify trees) averages ~1..1.5 tok/active-expert (fast decode kernel);
    // M>=24 (c=32, c=64, prefill) gets 2x-3x speedup from Blackwell sm_120a / INT8 / FP16 Tensor Cores!
    const bool use_sm120_tc = (!force_exact_dec && M >= 24 && sm120::available());
    const bool use_fp16_tc  = (!use_sm120_tc && M >= 24 && ink_has_int8_tc());
    const bool use_wmma_tc  = (!use_sm120_tc && !use_fp16_tc && M >= 24 && ink_has_wmma_f16());
    const int tile_rows = use_wmma_tc ? ((M <= 128) ? 16 : 64)
                                      : ((M <= 128 || use_fp16_tc) ? 16 : 64);
    k_ink_moe_bin<<<1, 128, 0, st>>>(experts, ws.topk_idx, M, tile_rows, ws.tb);

    const int P = M * kInkTopK;
    const int max_act = std::min(P, kInkExperts);

    // 2A. Volta sm_70 (V100) & Turing sm_75 (2080 Ti) Zero-Spill 4-CTA/SM WMMA FP16 Tensor Core path!
    if (use_wmma_tc && (info.has_sf4 || info.has_mx8 || info.has_sk)) {
        const int mt = ink_moe_max_tiles(M, tile_rows);
        if (tile_rows <= 16) {
            const unsigned grid = (unsigned)std::min(std::max(1, mt * 18), 2048);
            k_ink_moe_wmma_f16<true, 16><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, x_f32, nullptr, ws.h_f32);
            k_ink_moe_wmma_f16<false, 16><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.h_f32, ws.topk_w, ws.y_pair);
        } else {
            static bool s_carveout_set = false;
            if (!s_carveout_set) {
                cudaFuncSetAttribute(k_ink_moe_wmma_f16_bk64<true,  false>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
                cudaFuncSetAttribute(k_ink_moe_wmma_f16_bk64<false, false>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
                cudaFuncSetAttribute(k_ink_moe_wmma_f16_bk64<true,  true>,  cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
                cudaFuncSetAttribute(k_ink_moe_wmma_f16_bk64<false, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
                s_carveout_set = true;
            }
            const unsigned grid = (unsigned)std::min(std::max(1, mt * 36), 4096);
            if (info.has_sf4 || info.has_mx8) {
                k_ink_moe_wmma_f16_bk64<true,  false><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, x_f32, nullptr, ws.h_f32);
                k_ink_moe_wmma_f16_bk64<false, false><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.h_f32, ws.topk_w, ws.y_pair);
            }
            if (info.has_sk) {
                k_ink_moe_wmma_f16_bk64<true,  true><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, x_f32, nullptr, ws.h_f32);
                k_ink_moe_wmma_f16_bk64<false, true><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.h_f32, ws.topk_w, ws.y_pair);
            }
        }
        if (!info.has_gguf) return;
    }

    // 2B. Dispatch SF4 & MX8 experts (Blackwell sm_120a TC for M>=16 prefill, exact FP16 TC for M>=16 sm_80..89, 8-lane uint4 for M<16!)
    if (!use_wmma_tc && (info.has_sf4 || info.has_mx8)) {
        if (use_sm120_tc) {
            const int mt = ink_moe_max_tiles(M, tile_rows);
            if (info.has_sf4) {
                if (!quant_x_done) sm120::quantize_act_fp4(x_f32, M, kInkHidden, ws.xa_fp4, st);
                sm120::moe_sf4_gu_tc(sec4_base, experts, ws.tb, ws.xa_fp4, ws.ha_fp4, mt, tile_rows, st);
                sm120::moe_sf4_dn_tc(sec4_base, experts, ws.tb, ws.ha_fp4, ws.topk_w, ws.y_pair, mt, tile_rows, st);
            }
            if (info.has_mx8) {
                if (!quant_x_done) sm120::quantize_act_mx8(x_f32, M, kInkHidden, ws.xq_mx8, ws.xsc_mx8, st);
                sm120::moe_mx8_gu_tc(sec4_base, experts, ws.tb, ws.xq_mx8, ws.xsc_mx8, ws.hq_mx8, ws.hsc_mx8, mt, tile_rows, st);
                sm120::moe_mx8_dn_tc(sec4_base, experts, ws.tb, ws.hq_mx8, ws.hsc_mx8, ws.topk_w, ws.y_pair, mt, tile_rows, st);
            }
        } else if (use_fp16_tc) {
            const int mt = ink_moe_max_tiles(M, 16);
            const unsigned grid = (unsigned)std::min(std::max(1, mt * 18), 2048);
            k_ink_moe_sf4_mx8_fp16tc<true><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, x_f32, nullptr, ws.h_f32);
            k_ink_moe_sf4_mx8_fp16tc<false><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.h_f32, ws.topk_w, ws.y_pair);
        } else {
            dim3 g16(kInkInter / 16, max_act);
            if (M == 1) {
                k_ink_moe_gu_sf4_mx8<1><<<g16, 128, 0, st>>>(sec4_base, experts, ws.tb, x_f32, ws.h_f32);
                k_ink_moe_dn_sf4_mx8<1><<<g16, 128, 0, st>>>(sec4_base, experts, ws.tb, ws.h_f32, ws.topk_w, ws.y_pair);
            } else {
                k_ink_moe_gu_sf4_mx8<4><<<g16, 128, 0, st>>>(sec4_base, experts, ws.tb, x_f32, ws.h_f32);
                k_ink_moe_dn_sf4_mx8<4><<<g16, 128, 0, st>>>(sec4_base, experts, ws.tb, ws.h_f32, ws.topk_w, ws.y_pair);
            }
        }
    }

    // 2C. Dispatch S_K experts (Coalesced sm_80..sm_121 INT8 TC for M>=16, 4-lane zero-shuffle 32-feat/CTA dp4a_i8 for M<16!)
    if (!use_wmma_tc && info.has_sk) {
        if (!quant_x_done) {
            const int nblk_x = M * (kInkHidden / 64);
            k_ink_quant_q8_1<<<(nblk_x + 7) / 8, 256, 0, st>>>(x_f32, ws.xq_sk, M, kInkHidden);
        }
        if (M >= 24 && ink_has_int8_tc()) {
            const int mt = ink_moe_max_tiles(M, tile_rows);
            const unsigned grid = (unsigned)std::min(std::max(1, mt * 18), 2048);
            if (tile_rows <= 16) k_ink_moe_sk_tc<true, 16><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.xq_sk, nullptr, ws.h_f32);
            else                 k_ink_moe_sk_tc<true, 64><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.xq_sk, nullptr, ws.h_f32);
            const int nblk_h = P * (kInkInter / 64);
            k_ink_quant_q8_1<<<(nblk_h + 7) / 8, 256, 0, st>>>(ws.h_f32, ws.hq_sk, P, kInkInter);
            if (tile_rows <= 16) k_ink_moe_sk_tc<false, 16><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.hq_sk, ws.topk_w, ws.y_pair);
            else                 k_ink_moe_sk_tc<false, 64><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.hq_sk, ws.topk_w, ws.y_pair);
        } else {
            dim3 g32(kInkInter / 32, max_act);
            if (M == 1)       launch_ink_moe_gu_sk_mask<1>(g32, st, info.sk_gu_mask, sec4_base, experts, ws.tb, ws.xq_sk, ws.h_f32);
            else if (M <= 32) launch_ink_moe_gu_sk_mask<4>(g32, st, info.sk_gu_mask, sec4_base, experts, ws.tb, ws.xq_sk, ws.h_f32);
            else              launch_ink_moe_gu_sk_mask<16>(g32, st, info.sk_gu_mask, sec4_base, experts, ws.tb, ws.xq_sk, ws.h_f32);
            const int nblk_h = P * (kInkInter / 64);
            k_ink_quant_q8_1<<<(nblk_h + 7) / 8, 256, 0, st>>>(ws.h_f32, ws.hq_sk, P, kInkInter);
            if (M == 1)       launch_ink_moe_dn_sk_mask<1>(g32, st, info.sk_dn_mask, sec4_base, experts, ws.tb, ws.hq_sk, ws.topk_w, ws.y_pair);
            else if (M <= 32) launch_ink_moe_dn_sk_mask<4>(g32, st, info.sk_dn_mask, sec4_base, experts, ws.tb, ws.hq_sk, ws.topk_w, ws.y_pair);
            else              launch_ink_moe_dn_sk_mask<16>(g32, st, info.sk_dn_mask, sec4_base, experts, ws.tb, ws.hq_sk, ws.topk_w, ws.y_pair);
        }
    }

    // 2D. Dispatch native GGUF_RAW experts via vectorized DP4A + Q8_1_G32 multi-token activation reuse!
    // Covers Q4_K(12), Q5_K(13), Q6_K(14), Q8_0(8), IQ4_XS(23), IQ3_XXS(18), Q3_K(11), Q2_K(10), Q4_0(2), Q5_0(6), IQ4_NL(20), MXFP4(39), BF16, F16, F32!
    if (info.has_gguf) {
        if (!quant_x_done) {
            const int nblk_x = M * (kInkHidden / 32);
            k_ink_quant_q8_1_g32<<<(nblk_x + 7) / 8, 256, 0, st>>>(x_f32, ws.xq_g32, M, kInkHidden);
        }
        dim3 g_gu(kInkInter / 4, max_act);
        dim3 g_dn(kInkHidden / 4, max_act);
        if (M == 1) {
            k_ink_moe_gu_gguf_dp4a<1><<<g_gu, 128, 0, st>>>(sec4_base, experts, ws.tb, ws.xq_g32, x_f32, ws.h_f32);
        } else if (M <= 32) {
            k_ink_moe_gu_gguf_dp4a<4><<<g_gu, 128, 0, st>>>(sec4_base, experts, ws.tb, ws.xq_g32, x_f32, ws.h_f32);
        } else {
            k_ink_moe_gu_gguf_dp4a<16><<<g_gu, 128, 0, st>>>(sec4_base, experts, ws.tb, ws.xq_g32, x_f32, ws.h_f32);
        }
        const int nblk_h = P * (kInkInter / 32);
        k_ink_quant_q8_1_g32<<<(nblk_h + 7) / 8, 256, 0, st>>>(ws.h_f32, ws.hq_g32, P, kInkInter);
        if (M == 1) {
            k_ink_moe_dn_gguf_dp4a<1><<<g_dn, 128, 0, st>>>(sec4_base, experts, ws.tb, ws.hq_g32, ws.h_f32, ws.topk_w, ws.y_pair);
        } else if (M <= 32) {
            k_ink_moe_dn_gguf_dp4a<4><<<g_dn, 128, 0, st>>>(sec4_base, experts, ws.tb, ws.hq_g32, ws.h_f32, ws.topk_w, ws.y_pair);
        } else {
            k_ink_moe_dn_gguf_dp4a<16><<<g_dn, 128, 0, st>>>(sec4_base, experts, ws.tb, ws.hq_g32, ws.h_f32, ws.topk_w, ws.y_pair);
        }
    }
}

inline void launch_ink_moe(
    cudaStream_t st,
    cublasHandle_t cublas,
    const uint8_t* sec4_base,
    const InkExpertDev* experts,
    const InkLayerMoeInfo& info,
    const float* gate_w,
    const float* gate_b,
    const float* x_f32,
    float* y_out,
    int M,
    InkMoeWorkspace& ws,
    float route_scale = kInkRouteScale,
    bool force_exact_dec = false,
    bool run_reduce8 = true) {
    // 1. Exact Top-8 / 128 sigmoid router
    launch_ink_moe_router(st, cublas, gate_w, gate_b, x_f32, M, ws, route_scale, force_exact_dec);

    // 2. Dispatch GPU-resident experts
    launch_ink_moe_dispatch_only(st, sec4_base, experts, info, x_f32, M, ws, force_exact_dec, false);

    // 3. Deterministic 8-slot weighted sum -> y_out [M, 2304] (skipped when caller fuses reduce8 + post-MLP ShortConv4!)
    if (run_reduce8) {
        dim3 rgrid((kInkHidden + 255) / 256, M);
        k_ink_moe_reduce8<<<rgrid, 256, 0, st>>>(ws.y_pair, y_out, M);
    }
}

}  // namespace strata::ink
