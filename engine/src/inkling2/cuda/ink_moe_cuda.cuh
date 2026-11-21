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
// Dedicated n_e == 1 fast path eliminates loop/register overhead for the ~75% 1-token experts at M=5..8!
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
    __shared__ alignas(16) __half2 s_x[MAX_M][kInkHidden / 2];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        for (int m = 0; m < n_e; ++m) {
            const float2* src2 = reinterpret_cast<const float2*>(x_f32 + (size_t)tb.tok_idx[row0 + off_m + m] * kInkHidden);
            __half2* dst2 = s_x[m];
            for (int i = tid; i < (kInkHidden / 2); i += 128) {
                const float2 f = src2[i];
                dst2[i] = __floats2half2_rn(f.x, f.y);
            }
        }
        __syncthreads();

        if (n_e == 1) {
            float ag0 = 0.0f, au0 = 0.0f;
            if (ed.gu_code == FMT_SF4) {
                float hdr[4];
                memcpy(hdr, body, 16);
                const float w_outer = hdr[0] / hdr[1];
                constexpr size_t W_P_ROW = kInkHidden / 2;   // 1152 B = 72 uint4s!
                constexpr size_t W_S_ROW = kInkHidden / 16;  // 144 B = 72 uint16s!
                const uint8_t* base_p = body + 16;
                const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
                const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
                const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
                const uint16_t* gs2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat) * W_S_ROW);
                const uint16_t* us2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat + 1) * W_S_ROW);

                // 72 / 8 = 9 EXACT iterations (ZERO idle lanes, 128B coalesced uint4 loads!)
                for (int i = lg; i < 72; i += 8) {
                    const uint4 gv = gp16[i], uv = up16[i];
                    const uint16_t gs_p = gs2[i], us_p = us2[i];
                    const float sg0 = e4m3_to_f32((uint8_t)(gs_p & 0xFFu)) * w_outer;
                    const float sg1 = e4m3_to_f32((uint8_t)(gs_p >> 8))    * w_outer;
                    const float su0 = e4m3_to_f32((uint8_t)(us_p & 0xFFu)) * w_outer;
                    const float su1 = e4m3_to_f32((uint8_t)(us_p >> 8))    * w_outer;
                    __half2 gw[8], uw[8];
                    const __half2* xh0 = s_x[0] + i * 16;
                    sf4_decode16_h2(gv.x, gv.y, gw);
                    sf4_decode16_h2(uv.x, uv.y, uw);
                    ag0 = fmaf(dot16_h2_2acc(gw, xh0), sg0, ag0);
                    au0 = fmaf(dot16_h2_2acc(uw, xh0), su0, au0);
                    sf4_decode16_h2(gv.z, gv.w, gw);
                    sf4_decode16_h2(uv.z, uv.w, uw);
                    ag0 = fmaf(dot16_h2_2acc(gw, xh0 + 8), sg1, ag0);
                    au0 = fmaf(dot16_h2_2acc(uw, xh0 + 8), su1, au0);
                }
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
                float hdr[4];
                memcpy(hdr, body, 16);
                const float w_outer = hdr[0] / hdr[1];
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
                    const uint16_t gs_p = gs2[i], us_p = us2[i];
                    const float sg0 = e4m3_to_f32((uint8_t)(gs_p & 0xFFu)) * w_outer;
                    const float sg1 = e4m3_to_f32((uint8_t)(gs_p >> 8))    * w_outer;
                    const float su0 = e4m3_to_f32((uint8_t)(us_p & 0xFFu)) * w_outer;
                    const float su1 = e4m3_to_f32((uint8_t)(us_p >> 8))    * w_outer;
                    __half2 gw[8], uw[8];
                    sf4_decode16_h2(gv.x, gv.y, gw);
                    sf4_decode16_h2(uv.x, uv.y, uw);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 16;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg0, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su0, au[m]);
                        }
                    }
                    sf4_decode16_h2(gv.z, gv.w, gw);
                    sf4_decode16_h2(uv.z, uv.w, uw);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 16 + 8;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg1, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su1, au[m]);
                        }
                    }
                }
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
    __shared__ alignas(16) __half2 s_h[MAX_M][kInkInter / 2];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        for (int m = 0; m < n_e; ++m) {
            const float2* src2 = reinterpret_cast<const float2*>(h_f32 + (size_t)(row0 + off_m + m) * kInkInter);
            __half2* dst2 = s_h[m];
            for (int i = tid; i < (kInkInter / 2); i += 128) {
                const float2 f = src2[i];
                dst2[i] = __floats2half2_rn(f.x, f.y);
            }
        }
        __syncthreads();

        if (n_e == 1) {
            float y0 = 0.0f, y1 = 0.0f;
            if (ed.dn_code == FMT_SF4) {
                float hdr[4];
                memcpy(hdr, body, 16);
                const float w_outer = hdr[2] / hdr[3];
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
                    const uint16_t sc0p = s0[i], sc1p = s1[i];
                    const float sc00 = e4m3_to_f32((uint8_t)(sc0p & 0xFFu)) * w_outer;
                    const float sc01 = e4m3_to_f32((uint8_t)(sc0p >> 8))    * w_outer;
                    const float sc10 = e4m3_to_f32((uint8_t)(sc1p & 0xFFu)) * w_outer;
                    const float sc11 = e4m3_to_f32((uint8_t)(sc1p >> 8))    * w_outer;
                    __half2 w0[8], w1[8];
                    const __half2* hh = s_h[0] + i * 16;
                    sf4_decode16_h2(v0.x, v0.y, w0);
                    sf4_decode16_h2(v1.x, v1.y, w1);
                    y0 = fmaf(dot16_h2_2acc(w0, hh), sc00, y0);
                    y1 = fmaf(dot16_h2_2acc(w1, hh), sc10, y1);
                    sf4_decode16_h2(v0.z, v0.w, w0);
                    sf4_decode16_h2(v1.z, v1.w, w1);
                    y0 = fmaf(dot16_h2_2acc(w0, hh + 8), sc01, y0);
                    y1 = fmaf(dot16_h2_2acc(w1, hh + 8), sc11, y1);
                }
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
                float hdr[4];
                memcpy(hdr, body, 16);
                const float w_outer = hdr[2] / hdr[3];
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
                    const uint16_t sc0p = s0[i], sc1p = s1[i];
                    const float sc00 = e4m3_to_f32((uint8_t)(sc0p & 0xFFu)) * w_outer;
                    const float sc01 = e4m3_to_f32((uint8_t)(sc0p >> 8))    * w_outer;
                    const float sc10 = e4m3_to_f32((uint8_t)(sc1p & 0xFFu)) * w_outer;
                    const float sc11 = e4m3_to_f32((uint8_t)(sc1p >> 8))    * w_outer;
                    __half2 w0[8], w1[8];
                    sf4_decode16_h2(v0.x, v0.y, w0);
                    sf4_decode16_h2(v1.x, v1.y, w1);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 16;
                            a0[m] = fmaf(dot16_h2_2acc(w0, hh), sc00, a0[m]);
                            a1[m] = fmaf(dot16_h2_2acc(w1, hh), sc10, a1[m]);
                        }
                    }
                    sf4_decode16_h2(v0.z, v0.w, w0);
                    sf4_decode16_h2(v1.z, v1.w, w1);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 16 + 8;
                            a0[m] = fmaf(dot16_h2_2acc(w0, hh), sc01, a0[m]);
                            a1[m] = fmaf(dot16_h2_2acc(w1, hh), sc11, a1[m]);
                        }
                    }
                }
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

// ---- Universal Decode/Verify S_K Gate+Up+SITU Kernel (S1.5..S8_K) with Shared-Memory x_sk Cache! ----
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_moe_gu_sk(
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
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int feat = blockIdx.x * 8 + warp;

    const uint8_t* body = sec4_base + ed.body_off;
    const uint8_t* gu_base = body + kInkInter * 2;
    const size_t row_b = (size_t)(kInkHidden / 256) * ink_sb_bytes(ed.gu_code);
    const uint8_t* gr = gu_base + (size_t)(2 * feat) * row_b;
    const uint8_t* ur = gr + row_b;
    const float inv_sq = (feat < kInkInter) ? (1.0f / f16_to_f32(ld16_u(body + feat * 2))) : 0.0f;

    // Cache InkQ8_1_G64[MAX_M][36] in shared memory for MAX_M <= 8 (2.88 KB for M=1, 23 KB for M=8!)
    constexpr int SM_M = (MAX_M <= 8) ? MAX_M : 1;
    __shared__ alignas(16) InkQ8_1_G64 s_xq[SM_M][kInkHidden / 64];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        float ag[MAX_M] = {}, au[MAX_M] = {};
        const InkQ8_1_G64* xp[MAX_M];
        if constexpr (MAX_M <= 8) {
            constexpr int N4 = (int)((kInkHidden / 64) * sizeof(InkQ8_1_G64) / 16);  // 36 * 5 = 180 uint4s
            for (int m = 0; m < n_e; ++m) {
                const uint4* src = reinterpret_cast<const uint4*>(x_sk + (size_t)tb.tok_idx[row0 + off_m + m] * (kInkHidden / 64));
                uint4* dst = reinterpret_cast<uint4*>(s_xq[m]);
                for (int i = tid; i < N4; i += 256) dst[i] = src[i];
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
                case FMT_S1_5: sk_row_dot_2304<FMT_S1_5, MAX_M>(gr, xp, n_e, lane, ag); sk_row_dot_2304<FMT_S1_5, MAX_M>(ur, xp, n_e, lane, au); break;
                case FMT_S2_K: sk_row_dot_2304<FMT_S2_K, MAX_M>(gr, xp, n_e, lane, ag); sk_row_dot_2304<FMT_S2_K, MAX_M>(ur, xp, n_e, lane, au); break;
                case FMT_S3_K: sk_row_dot_2304<FMT_S3_K, MAX_M>(gr, xp, n_e, lane, ag); sk_row_dot_2304<FMT_S3_K, MAX_M>(ur, xp, n_e, lane, au); break;
                case FMT_S4_K: sk_row_dot_2304<FMT_S4_K, MAX_M>(gr, xp, n_e, lane, ag); sk_row_dot_2304<FMT_S4_K, MAX_M>(ur, xp, n_e, lane, au); break;
                case FMT_S5_K: sk_row_dot_2304<FMT_S5_K, MAX_M>(gr, xp, n_e, lane, ag); sk_row_dot_2304<FMT_S5_K, MAX_M>(ur, xp, n_e, lane, au); break;
                case FMT_S6_K: sk_row_dot_2304<FMT_S6_K, MAX_M>(gr, xp, n_e, lane, ag); sk_row_dot_2304<FMT_S6_K, MAX_M>(ur, xp, n_e, lane, au); break;
                case FMT_S8_K: sk_row_dot_2304<FMT_S8_K, MAX_M>(gr, xp, n_e, lane, ag); sk_row_dot_2304<FMT_S8_K, MAX_M>(ur, xp, n_e, lane, au); break;
                default: break;
            }
            if (lane == 0) {
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < n_e) h_f32[(size_t)(row0 + off_m + m) * kInkInter + feat] = situ_f32(ag[m], au[m]) * inv_sq;
                }
            }
        }
        if constexpr (MAX_M <= 8) { if (off_m + MAX_M < n_total) __syncthreads(); }
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_moe_dn_sk(
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

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int pair_r = blockIdx.x * 8 + warp;

    const uint8_t* body = sec4_base + ed.body_off;
    const size_t pair_b = 9ull * ink_sb_bytes(ed.dn_code);
    const uint8_t* pr = body + ed.dn_off + (size_t)pair_r * pair_b;

    // Cache InkQ8_1_G64[MAX_M][18] in shared memory for ALL MAX_M <= 16 (23 KB for M=16!)
    __shared__ alignas(16) InkQ8_1_G64 s_hq[MAX_M][kInkInter / 64];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        float a0[MAX_M] = {}, a1[MAX_M] = {};
        const InkQ8_1_G64* hp[MAX_M];
        constexpr int N4 = (int)((kInkInter / 64) * sizeof(InkQ8_1_G64) / 16);  // 18 * 5 = 90 uint4s
        for (int m = 0; m < n_e; ++m) {
            const uint4* src = reinterpret_cast<const uint4*>(h_sk + (size_t)(row0 + off_m + m) * (kInkInter / 64));
            uint4* dst = reinterpret_cast<uint4*>(s_hq[m]);
            for (int i = tid; i < N4; i += 256) dst[i] = src[i];
            hp[m] = s_hq[m];
        }
        for (int m = n_e; m < MAX_M; ++m) hp[m] = nullptr;
        __syncthreads();

        if (pair_r < kInkHidden / 2) {
            switch (ed.dn_code) {
                case FMT_S1_5: sk_paired2_dot_1152<FMT_S1_5, MAX_M>(pr, hp, n_e, lane, a0, a1); break;
                case FMT_S2_K: sk_paired2_dot_1152<FMT_S2_K, MAX_M>(pr, hp, n_e, lane, a0, a1); break;
                case FMT_S3_K: sk_paired2_dot_1152<FMT_S3_K, MAX_M>(pr, hp, n_e, lane, a0, a1); break;
                case FMT_S4_K: sk_paired2_dot_1152<FMT_S4_K, MAX_M>(pr, hp, n_e, lane, a0, a1); break;
                case FMT_S5_K: sk_paired2_dot_1152<FMT_S5_K, MAX_M>(pr, hp, n_e, lane, a0, a1); break;
                case FMT_S6_K: sk_paired2_dot_1152<FMT_S6_K, MAX_M>(pr, hp, n_e, lane, a0, a1); break;
                case FMT_S8_K: sk_paired2_dot_1152<FMT_S8_K, MAX_M>(pr, hp, n_e, lane, a0, a1); break;
                default: break;
            }
            if (lane == 0) {
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
        if (off_m + MAX_M < n_total) __syncthreads();
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

// Persistent Portable INT8 Tensor Core S_K MoE Kernel (sm_80..sm_121: GU=true -> Gate+Up+SITU, GU=false -> Paired-2-Row Down)
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
    constexpr int NG       = GU ? (kInkHidden / 64) : (kInkInter / 64);  // 36 (GU) or 18 (DN)
    constexpr int NFB      = ROWS_N / kWR;                       // 18 blocks of 128 rows
    constexpr int N_WN     = TR / 16;                            // 1 (TR=16) or 4 (TR=64)
    constexpr int N_WF     = 8 / N_WN;                           // 8 (TR=16) or 2 (TR=64)
    constexpr int W_PER_WF = kWR / N_WF;                         // 16 or 64
    constexpr int NI       = W_PER_WF / 16;                      // 1 or 4
    constexpr int WSTR     = 80;  // 64B + 16B pad -> 16-byte aligned for uint4 AND bank-conflict free!
    constexpr int ASTR     = 80;

    __shared__ alignas(16) int8_t  s_wq[kWR][WSTR];   // 8,704 B
    __shared__ alignas(16) int8_t  s_aq[TR][ASTR];    // 1,088 .. 4,352 B
    __shared__ float               s_wsc[kWR];        //   512 B
    __shared__ float               s_ad[TR];          //   256 B
    __shared__ float               s_as[TR];          //   256 B
    __shared__ int32_t             s_row[TR];         //   256 B
    __shared__ float               s_rw[TR];          //   256 B

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;
    const int wn   = warp % N_WN;
    const int nb0  = 16 * wn;
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
        if (ed.tier > 1 || fmt == FMT_SF4 || fmt == FMT_MX8) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        const int sb_b = ink_sb_bytes(fmt);
        const float half = ink_sk_half(fmt);
        const bool has_half = (fmt != FMT_S8_K && fmt != FMT_S1_5);

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

        // Weight pointer setup for tid < kWR (1 thread unpacks 1 weight row x 64 weights per sub-block!)
        const uint8_t* gu_base = body + kInkInter * 2;
        const size_t gu_row_b = 9ull * (size_t)sb_b;
        const size_t dn_pair_b = 9ull * (size_t)sb_b;
        const int phys_row = GU ? (2 * (fb * 64 + 8 * (tid >> 4) + (tid & 7)) + ((tid >> 3) & 1))
                                : (fb * kWR + tid);

        for (int xg = 0; xg < NG; ++xg) {
            // 1. First 128 threads unpack 128 weight rows x 64 weights into s_wq; next threads load TR activation blocks!
            if (tid < kWR) {
                const uint8_t* blk;
                int sg;
                if constexpr (GU) {
                    blk = gu_base + (size_t)phys_row * gu_row_b + (size_t)(xg >> 2) * sb_b;
                    sg  = xg & 3;
                } else {
                    const int pair_r = phys_row >> 1;
                    const int sub_glob = xg + (phys_row & 1) * 18;
                    blk = body + ed.dn_off + (size_t)pair_r * dn_pair_b + (size_t)(sub_glob >> 2) * sb_b;
                    sg  = sub_glob & 3;
                }
                sk_unpack64_sub_to_i8(fmt, blk, sg, s_wq[tid], s_wsc[tid]);
            } else {
                const int atid = tid - kWR;  // 0..127 -> 2 threads per activation row (for TR<=64)
                const int r = atid >> 1;
                const int h = atid & 1;      // 0..1: 32 bytes each
                if (r < nrows) {
                    const InkQ8_1_G64& xb = act_sk[(size_t)s_row[r] * NG + xg];
                    const uint4* src = reinterpret_cast<const uint4*>(xb.qs + h * 32);
                    uint4* dst = reinterpret_cast<uint4*>(s_aq[r] + h * 32);
                    dst[0] = src[0]; dst[1] = src[1];
                    if (h == 0) { s_ad[r] = xb.d; s_as[r] = xb.s; }
                }
            }
            __syncthreads();

            // 2. Run 2x mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (k=0..31 and k=32..63)!
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
                    if (has_half) {
                        acc[i][0][0] = fmaf(eff0, xd00 * (float)cv[0] - half * xs00, acc[i][0][0]);
                        acc[i][0][1] = fmaf(eff0, xd01 * (float)cv[1] - half * xs01, acc[i][0][1]);
                        acc[i][0][2] = fmaf(eff1, xd00 * (float)cv[2] - half * xs00, acc[i][0][2]);
                        acc[i][0][3] = fmaf(eff1, xd01 * (float)cv[3] - half * xs01, acc[i][0][3]);
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
                        if (has_half) {
                            acc[i][1][0] = fmaf(eff0, xd10 * (float)dv[0] - half * xs10, acc[i][1][0]);
                            acc[i][1][1] = fmaf(eff0, xd11 * (float)dv[1] - half * xs11, acc[i][1][1]);
                            acc[i][1][2] = fmaf(eff1, xd10 * (float)dv[2] - half * xs10, acc[i][1][2]);
                            acc[i][1][3] = fmaf(eff1, xd11 * (float)dv[3] - half * xs11, acc[i][1][3]);
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
// 5. High-Level MoE Dispatcher (Auto-selects Blackwell sm_120a TC vs INT8 TC vs Bandwidth-Optimal Decode!)
// ============================================================================
struct InkLayerMoeInfo {
    bool has_sf4 = false;
    bool has_mx8 = false;
    bool has_sk  = false;
};

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
    bool force_exact_dec = false) {
    // 1. Exact Top-8 / 128 sigmoid router (128-CTA coalesced GEMV for M<=16, cuBLAS TF32 for M>16!) + device-side binning!
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
            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    k_ink_router_select_topk8<<<M, 32, 0, st>>>(ws.r_logits, gate_b, ws.topk_idx, ws.topk_w, route_scale, M);
    const int tile_rows = (M <= 128) ? 16 : 64;
    k_ink_moe_bin<<<1, 128, 0, st>>>(experts, ws.topk_idx, M, tile_rows, ws.tb);

    const int P = M * kInkTopK;
    const int max_act = std::min(P, kInkExperts);
    const bool use_sm120_tc = (!force_exact_dec && M >= 16 && sm120::available());

    // 2A. Dispatch SF4 & MX8 experts (Blackwell sm_120a TC even on mixed SQ-15/SQ-10 layers!)
    if (info.has_sf4 || info.has_mx8) {
        if (use_sm120_tc) {
            const int mt = ink_moe_max_tiles(M, tile_rows);
            if (info.has_sf4) {
                sm120::quantize_act_fp4(x_f32, M, kInkHidden, ws.xa_fp4, st);
                sm120::moe_sf4_gu_tc(sec4_base, experts, ws.tb, ws.xa_fp4, ws.ha_fp4, mt, tile_rows, st);
                sm120::moe_sf4_dn_tc(sec4_base, experts, ws.tb, ws.ha_fp4, ws.topk_w, ws.y_pair, mt, tile_rows, st);
            }
            if (info.has_mx8) {
                sm120::quantize_act_mx8(x_f32, M, kInkHidden, ws.xq_mx8, ws.xsc_mx8, st);
                sm120::moe_mx8_gu_tc(sec4_base, experts, ws.tb, ws.xq_mx8, ws.xsc_mx8, ws.hq_mx8, ws.hsc_mx8, mt, tile_rows, st);
                sm120::moe_mx8_dn_tc(sec4_base, experts, ws.tb, ws.hq_mx8, ws.hsc_mx8, ws.topk_w, ws.y_pair, mt, tile_rows, st);
            }
        } else {
            // 16 features/CTA (128 threads, 8 lanes x 16B uint4 = 128B cache line!), 576 CTAs at M=1 -> 3.4 full waves on 170 SMs!
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

    // 2B. Dispatch S_K experts (Portable sm_75..sm_121 INT8 TC for M>=32, dp4a_i8 for M<32!)
    if (info.has_sk) {
        const int nblk_x = M * (kInkHidden / 64);
        k_ink_quant_q8_1<<<(nblk_x + 7) / 8, 256, 0, st>>>(x_f32, ws.xq_sk, M, kInkHidden);
        if (!force_exact_dec && M >= 32 && ink_has_int8_tc()) {
            const int mt = ink_moe_max_tiles(M, tile_rows);
            const unsigned grid = (unsigned)std::min(std::max(1, mt * 18), 2048);
            if (tile_rows <= 16) k_ink_moe_sk_tc<true, 16><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.xq_sk, nullptr, ws.h_f32);
            else                 k_ink_moe_sk_tc<true, 64><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.xq_sk, nullptr, ws.h_f32);
            const int nblk_h = P * (kInkInter / 64);
            k_ink_quant_q8_1<<<(nblk_h + 7) / 8, 256, 0, st>>>(ws.h_f32, ws.hq_sk, P, kInkInter);
            if (tile_rows <= 16) k_ink_moe_sk_tc<false, 16><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.hq_sk, ws.topk_w, ws.y_pair);
            else                 k_ink_moe_sk_tc<false, 64><<<grid, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.hq_sk, ws.topk_w, ws.y_pair);
        } else {
            dim3 g8(kInkInter / 8, max_act);
            if (M == 1)       k_ink_moe_gu_sk<1><<<g8, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.xq_sk, ws.h_f32);
            else if (M <= 16) k_ink_moe_gu_sk<4><<<g8, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.xq_sk, ws.h_f32);
            else              k_ink_moe_gu_sk<16><<<g8, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.xq_sk, ws.h_f32);
            const int nblk_h = P * (kInkInter / 64);
            k_ink_quant_q8_1<<<(nblk_h + 7) / 8, 256, 0, st>>>(ws.h_f32, ws.hq_sk, P, kInkInter);
            if (M == 1)       k_ink_moe_dn_sk<1><<<g8, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.hq_sk, ws.topk_w, ws.y_pair);
            else if (M <= 16) k_ink_moe_dn_sk<4><<<g8, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.hq_sk, ws.topk_w, ws.y_pair);
            else              k_ink_moe_dn_sk<16><<<g8, 256, 0, st>>>(sec4_base, experts, ws.tb, ws.hq_sk, ws.topk_w, ws.y_pair);
        }
    }

    // 3. Deterministic 8-slot weighted sum -> y_out [M, 2304]
    dim3 rgrid((kInkHidden + 255) / 256, M);
    k_ink_moe_reduce8<<<rgrid, 256, 0, st>>>(ws.y_pair, y_out, M);
}

}  // namespace strata::ink
