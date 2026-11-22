// Copyright (c) 2026 Strata contributors
// SPDX-License-Identifier: MIT
//
// src/kernels/hip/hip_unsloth_dp4a.cu - Clean-room AMD Wave32 DP4A decode GEMV, fused EXP_MODE=8
// gate+up+SwiGLU+q8_1, and LDS-staged down kernels for RDNA2 (gfx1030), RDNA3/3.5 (gfx1100..1151),
// and RDNA4 (gfx1200/1201).
//
// Supports all 7 primary Unsloth quant formats with hoisted decode-once HipSplit<TY>:
//   Q4_K (12), Q5_K (13), Q6_K (14), Q8_0 (8), IQ4_XS (23), IQ3_XXS (18), NVFP4 (40).

#include "strata/kernels/hip_unsloth_dp4a.hpp"
#include "strata/kernels/hip_unsloth_wmma.hpp"

#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#include "strata/quant/strata-quant-common.h"

#include <algorithm>
#include <type_traits>

namespace strata::kernels::hip_unsloth {
namespace {

constexpr int kLdsNt = 4;     // entries staged simultaneously in LDS
constexpr int kLdsRb = 64;    // down rows per block
constexpr int kMaxGridY = 65535;

// Branchless 6-bit scale/min extraction for Q4_K / Q5_K
__device__ __forceinline__ void hip_k_scale_min(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const auto* scales = reinterpret_cast<const uint16_t*>(scales8);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = static_cast<uint32_t>(-static_cast<int32_t>(j >= 2));
    aux[0] = static_cast<uint16_t>(((s0 & 0x3f3fu) & ~hi) |
                                   ((((s4 >> 0) & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2)) & hi));
    aux[1] = static_cast<uint16_t>(((s2 & 0x3f3fu) & ~hi) |
                                   ((((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2)) & hi));
}

// ===========================================================================
// Unified hoisted-decode HipSplit<TY> traits (decode once, apply to NC tokens)
// ===========================================================================
template <int TY>
struct HipSplit;

// Q4_K (12): QK=256, 16 items per block (step=2)
template <>
struct HipSplit<12> {
    static constexpr int qk = 256, ipb = 16, step = 2;
    static constexpr bool has_grid = false;
    struct W {
        int v0[2], v1[2];
        float2 dm;
        uint8_t sc[2], m[2];
    };
    __device__ static __forceinline__ W load(const uint8_t* row, int kbx, int iqs, const uint32_t*) {
        const auto* b = reinterpret_cast<const block_q4_K*>(row) + kbx;
        const int bq8_off = 2 * ((iqs / 2) / 4);
        const int* q4 = reinterpret_cast<const int*>(b->qs + 16 * bq8_off + 4 * ((iqs / 2) % 4));
        const int raw0 = q4[0], raw1 = q4[4];
        uint16_t aux[2];
        hip_k_scale_min(b->scales, bq8_off, aux);
        W r;
        r.v0[0] = (raw0 >> 0) & 0x0F0F0F0F;
        r.v1[0] = (raw1 >> 0) & 0x0F0F0F0F;
        r.v0[1] = (raw0 >> 4) & 0x0F0F0F0F;
        r.v1[1] = (raw1 >> 4) & 0x0F0F0F0F;
        r.dm = __half22float2(b->dm);
        r.sc[0] = static_cast<uint8_t>(aux[0] & 0xFFu);
        r.sc[1] = static_cast<uint8_t>(aux[0] >> 8);
        r.m[0]  = static_cast<uint8_t>(aux[1] & 0xFFu);
        r.m[1]  = static_cast<uint8_t>(aux[1] >> 8);
        return r;
    }
    __device__ static __forceinline__ float apply(const W& r, const block_q8_1* bq8, int iqs) {
        const int bq8_off = 2 * ((iqs / 2) / 4);
        const int q8_sub = (iqs / 2) % 4;
        float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const block_q8_1* bi = bq8 + bq8_off + i;
            const float d8 = __low2float(bi->ds);
            const int* q8 = reinterpret_cast<const int*>(bi->qs) + q8_sub;
            const int u0 = q8[0], u1 = q8[4];
            const int dot1 = hip_dp4a(r.v1[i], u1, hip_dp4a(r.v0[i], u0, 0));
            const int dot2 = hip_dp4a(0x01010101, u1, hip_dp4a(0x01010101, u0, 0));
            sumf_d += d8 * static_cast<float>(dot1 * static_cast<int>(r.sc[i]));
            sumf_m += d8 * static_cast<float>(dot2 * static_cast<int>(r.m[i]));
        }
        return r.dm.x * sumf_d - r.dm.y * sumf_m;
    }
};

// Q5_K (13): QK=256, 16 items per block (step=2)
template <>
struct HipSplit<13> {
    static constexpr int qk = 256, ipb = 16, step = 2;
    static constexpr bool has_grid = false;
    struct W {
        int v0[2], v1[2];
        float2 dm;
        uint8_t sc[2], m[2];
    };
    __device__ static __forceinline__ W load(const uint8_t* row, int kbx, int iqs, const uint32_t*) {
        const auto* b = reinterpret_cast<const block_q5_K*>(row) + kbx;
        const int bq8_off = 2 * ((iqs / 2) / 4);
        const int qsub = (iqs / 2) % 4;
        const int* ql = reinterpret_cast<const int*>(b->qs + 16 * bq8_off + 4 * qsub);
        const int* qh = reinterpret_cast<const int*>(b->qh + 4 * qsub);
        const int vl0 = ql[0], vl1 = ql[4];
        const int vh0 = qh[0] >> bq8_off, vh1 = qh[4] >> bq8_off;
        uint16_t aux[2];
        hip_k_scale_min(b->scales, bq8_off, aux);
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            r.v0[i] = ((vl0 >> (4 * i)) & 0x0F0F0F0F) | (((vh0 >> i) << 4) & 0x10101010);
            r.v1[i] = ((vl1 >> (4 * i)) & 0x0F0F0F0F) | (((vh1 >> i) << 4) & 0x10101010);
        }
        r.dm = __half22float2(b->dm);
        r.sc[0] = static_cast<uint8_t>(aux[0] & 0xFFu);
        r.sc[1] = static_cast<uint8_t>(aux[0] >> 8);
        r.m[0]  = static_cast<uint8_t>(aux[1] & 0xFFu);
        r.m[1]  = static_cast<uint8_t>(aux[1] >> 8);
        return r;
    }
    __device__ static __forceinline__ float apply(const W& r, const block_q8_1* bq8, int iqs) {
        const int bq8_off = 2 * ((iqs / 2) / 4);
        const int q8_sub = (iqs / 2) % 4;
        float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const block_q8_1* bi = bq8 + bq8_off + i;
            const float d8 = __low2float(bi->ds);
            const int* q8 = reinterpret_cast<const int*>(bi->qs) + q8_sub;
            const int u0 = q8[0], u1 = q8[4];
            const int dot1 = hip_dp4a(r.v0[i], u0, hip_dp4a(r.v1[i], u1, 0));
            const int dot2 = hip_dp4a(0x01010101, u0, hip_dp4a(0x01010101, u1, 0));
            sumf_d += d8 * static_cast<float>(dot1 * static_cast<int>(r.sc[i]));
            sumf_m += d8 * static_cast<float>(dot2 * static_cast<int>(r.m[i]));
        }
        return r.dm.x * sumf_d - r.dm.y * sumf_m;
    }
};

// Q6_K (14): QK=256, 32 items per block (step=1). Hoisted decode + 2-op hip_sub32_bytes!
template <>
struct HipSplit<14> {
    static constexpr int qk = 256, ipb = 32, step = 1;
    static constexpr bool has_grid = false;
    struct W {
        int vi[2];
        int sc[2];
        float d;
    };
    __device__ static __forceinline__ W load(const uint8_t* row, int kbx, int iqs, const uint32_t*) {
        const auto* b = reinterpret_cast<const block_q6_K*>(row) + kbx;
        const int scale_off = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const uint32_t vl = static_cast<uint32_t>(hip_ld_b2(b->ql, iqs));
        const uint32_t vh = static_cast<uint32_t>(hip_ld_b2(b->qh, 8 * (iqs / 16) + (iqs % 8))) >> vh_shift;
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t vil = (vl >> (4 * i)) & 0x0F0F0F0Fu;
            const uint32_t vih = ((vh >> (4 * i)) << 4) & 0x30303030u;
            r.vi[i] = hip_sub32_bytes(vil | vih);
            r.sc[i] = static_cast<int>(b->scales[scale_off + 4 * i]);
        }
        r.d = __half2float(b->d);
        return r;
    }
    __device__ static __forceinline__ float apply(const W& r, const block_q8_1* bq8, int iqs) {
        const int bq8_off = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int q8_idx = iqs & 7;
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const block_q8_1* bi = bq8 + bq8_off + 2 * i;
            const int u = hip_ld_b4(bi->qs, q8_idx);
            sumf += __low2float(bi->ds) * static_cast<float>(hip_dp4a(r.vi[i], u, 0) * r.sc[i]);
        }
        return r.d * sumf;
    }
};

// Q8_0 (8): QK=32, 4 items per block (step=2, 8 values per call)
template <>
struct HipSplit<8> {
    static constexpr int qk = 32, ipb = 4, step = 2;
    static constexpr bool has_grid = false;
    struct W {
        int q[2];
        float d;
    };
    __device__ static __forceinline__ W load(const uint8_t* row, int kbx, int iqs, const uint32_t*) {
        const auto* b = reinterpret_cast<const block_q8_0*>(row) + kbx;
        W r;
        r.q[0] = hip_ld_b2(b->qs, iqs + 0);
        r.q[1] = hip_ld_b2(b->qs, iqs + 1);
        r.d = __half2float(b->d);
        return r;
    }
    __device__ static __forceinline__ float apply(const W& r, const block_q8_1* bq8, int iqs) {
        int sumi = hip_dp4a(r.q[0], hip_ld_b4(bq8->qs, iqs + 0), 0);
        sumi = hip_dp4a(r.q[1], hip_ld_b4(bq8->qs, iqs + 1), sumi);
        return r.d * __low2float(bq8->ds) * static_cast<float>(sumi);
    }
};

// IQ4_XS (23): QK=256, 8 items per block (step=4, 32 values per call). 6-op v_perm_b32 LUT!
template <>
struct HipSplit<23> {
    static constexpr int qk = 256, ipb = 8, step = 4;
    static constexpr bool has_grid = false;
    struct W {
        int2 v[4];
        float dl;
    };
    __device__ static __forceinline__ W load(const uint8_t* row, int kbx, int iqs, const uint32_t*) {
        const auto* b = reinterpret_cast<const block_iq4_xs*>(row) + kbx;
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            r.v[j] = hip_iq4_lut16(static_cast<uint32_t>(hip_ld_b4(b->qs, iqs + j)));
        }
        const int ls = ((b->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) |
                       (((b->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dl = __half2float(b->d) * static_cast<float>(ls - 32);
        return r;
    }
    __device__ static __forceinline__ float apply(const W& r, const block_q8_1* bq8, int iqs) {
        const block_q8_1* bi = bq8 + (iqs / 4);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = hip_dp4a(r.v[j].x, hip_ld_b4(bi->qs, j + 0), sumi);
            sumi = hip_dp4a(r.v[j].y, hip_ld_b4(bi->qs, j + 4), sumi);
        }
        return r.dl * __low2float(bi->ds) * static_cast<float>(sumi);
    }
};

// IQ3_XXS (18): QK=256, 8 items per block (step=2, 32 values per call). LDS grid + 3-op sign negate!
template <>
struct HipSplit<18> {
    static constexpr int qk = 256, ipb = 8, step = 2;
    static constexpr bool has_grid = true;
    struct W {
        int g[8];
        int ls;
        float d;
    };
    __device__ static __forceinline__ W load(const uint8_t* row, int kbx, int iqs, const uint32_t* sgrid) {
        const auto* b = reinterpret_cast<const block_iq3_xxs*>(row) + kbx;
        const int2 q3p = make_int2(hip_ld_b2(b->qs, iqs), hip_ld_b2(b->qs, iqs + 1));
        const auto* q3 = reinterpret_cast<const uint8_t*>(&q3p);
        const uint32_t aux32 = static_cast<uint32_t>(hip_ld_b2(b->qs, 16 + iqs / 2));
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t g0 = sgrid ? sgrid[q3[l0 + 0]] : iq3xxs_grid[q3[l0 + 0]];
            const uint32_t g1 = sgrid ? sgrid[q3[l0 + 1]] : iq3xxs_grid[q3[l0 + 1]];
            const uint32_t v7 = (aux32 >> (7 * (l0 >> 1))) & 0x7Fu;
            const uint32_t s8 = v7 | ((__popc(v7) & 1u) << 7);
            r.g[l0 + 0] = hip_iq_sign_negate(g0, s8 & 0xFu);
            r.g[l0 + 1] = hip_iq_sign_negate(g1, s8 >> 4);
        }
        r.ls = static_cast<int>(aux32 >> 28);
        r.d = __half2float(b->d);
        return r;
    }
    __device__ static __forceinline__ float apply(const W& r, const block_q8_1* bq8, int iqs) {
        const block_q8_1* bi = bq8 + (iqs / 2);
        int sumi = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; ++l0) {
            sumi = hip_dp4a(r.g[l0], hip_ld_b4(bi->qs, l0), sumi);
        }
        sumi = (r.ls * sumi + sumi / 2) / 2;
        return r.d * __low2float(bi->ds) * static_cast<float>(sumi);
    }
};

// NVFP4 (40): QK=64, 2 items per block (step=4, 32 values per call). 6-op v_perm_b32 LUT!
template <>
struct HipSplit<40> {
    static constexpr int qk = 64, ipb = 2, step = 4;
    static constexpr bool has_grid = false;
    struct W {
        int2 v0[2], v1[2];
        float sc[2];
    };
    __device__ static __forceinline__ W load(const uint8_t* row, int kbx, int iqs, const uint32_t*) {
        const auto* b = reinterpret_cast<const block_nvfp4*>(row) + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int iqs0 = iqs + 2 * i;
            const int iqs1 = iqs0 + 1;
            const int is = iqs0 >> 1;
            r.v0[i] = hip_fp4_lut16(static_cast<uint32_t>(hip_ld_b4(b->qs, iqs0)));
            r.v1[i] = hip_fp4_lut16(static_cast<uint32_t>(hip_ld_b4(b->qs, iqs1)));
            r.sc[i] = hip_ue4m3_to_f32_half(b->d[is]);
        }
        return r;
    }
    __device__ static __forceinline__ float apply(const W& r, const block_q8_1* bq8, int iqs) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int is = (iqs + 2 * i) >> 1;
            const block_q8_1* bi = bq8 + (is >> 1);
            const int i8 = (is & 1) << 2;
            int sumi = hip_dp4a(r.v0[i].x, hip_ld_b4(bi->qs, i8 + 0), 0);
            sumi = hip_dp4a(r.v0[i].y, hip_ld_b4(bi->qs, i8 + 2), sumi);
            sumi = hip_dp4a(r.v1[i].x, hip_ld_b4(bi->qs, i8 + 1), sumi);
            sumi = hip_dp4a(r.v1[i].y, hip_ld_b4(bi->qs, i8 + 3), sumi);
            sum += r.sc[i] * __low2float(bi->ds) * static_cast<float>(sumi);
        }
        return sum;
    }
};

// ===========================================================================
// 1. Wave32 DP4A Multi-Column Decode GEMV (8 waves/block = 8 rows/block)
// ===========================================================================
template <int TY, int NC>
__global__ void __launch_bounds__(256) hip_dp4a_mmvq_kernel(
    const uint8_t* __restrict__ W, std::size_t row_bytes,
    const block_q8_1* __restrict__ X, float* __restrict__ Y,
    int n_in, int n_out, float global_scale) {
    using S = HipSplit<TY>;
    __shared__ uint32_t sgrid[S::has_grid ? 256 : 1];
    if constexpr (S::has_grid) {
        sgrid[threadIdx.x] = iq3xxs_grid[threadIdx.x];
        hip_lds_barrier();
    }
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = static_cast<int>(blockIdx.x) * 8 + wave;
    if (row >= n_out) return;

    const int nb = n_in / S::qk;
    const int n_items = nb * S::ipb;
    const int xb = n_in / 32;
    const uint8_t* wr = W + static_cast<std::size_t>(row) * row_bytes;

    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;

    for (int k = lane; k < n_items; k += 32) {
        const int kbx = k / S::ipb;
        const int iqs = S::step * (k % S::ipb);
        const typename S::W rw = S::load(wr, kbx, iqs, S::has_grid ? sgrid : nullptr);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[c] += S::apply(rw, X + static_cast<std::size_t>(c) * xb + kbx * (S::qk / 32), iqs);
        }
    }

#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = hip_wsum32(acc[c]) * global_scale;
        if (lane == 0) Y[static_cast<std::size_t>(c) * n_out + row] = v;
    }
}

template <int TY>
void launch_dp4a_mmvq_ty(const uint8_t* W, std::size_t rb, const block_q8_1* X, float* Y,
                         int n_in, int n_out, int ncols, float gscale, hipStream_t s) {
    const unsigned blocks = static_cast<unsigned>((n_out + 7) / 8);
    const int xs = n_in / 32;
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = (ncols - c0) < 8 ? (ncols - c0) : 8;
        const block_q8_1* xc = X + static_cast<std::size_t>(c0) * xs;
        float* yc = Y + static_cast<std::size_t>(c0) * n_out;
        switch (nc) {
            case 1: hip_dp4a_mmvq_kernel<TY, 1><<<blocks, 256, 0, s>>>(W, rb, xc, yc, n_in, n_out, gscale); break;
            case 2: hip_dp4a_mmvq_kernel<TY, 2><<<blocks, 256, 0, s>>>(W, rb, xc, yc, n_in, n_out, gscale); break;
            case 3: hip_dp4a_mmvq_kernel<TY, 3><<<blocks, 256, 0, s>>>(W, rb, xc, yc, n_in, n_out, gscale); break;
            case 4: hip_dp4a_mmvq_kernel<TY, 4><<<blocks, 256, 0, s>>>(W, rb, xc, yc, n_in, n_out, gscale); break;
            case 5: hip_dp4a_mmvq_kernel<TY, 5><<<blocks, 256, 0, s>>>(W, rb, xc, yc, n_in, n_out, gscale); break;
            case 6: hip_dp4a_mmvq_kernel<TY, 6><<<blocks, 256, 0, s>>>(W, rb, xc, yc, n_in, n_out, gscale); break;
            case 7: hip_dp4a_mmvq_kernel<TY, 7><<<blocks, 256, 0, s>>>(W, rb, xc, yc, n_in, n_out, gscale); break;
            default: hip_dp4a_mmvq_kernel<TY, 8><<<blocks, 256, 0, s>>>(W, rb, xc, yc, n_in, n_out, gscale); break;
        }
    }
}

// ===========================================================================
// 2. Fused EXP_MODE=8 gate + up + SwiGLU + q8_1 (all 7 Unsloth quant types)
//    - 1 block computes 32 features (32 gate + 32 up rows) = 1 q8_1 output block per entry
//    - 65535-safe gridDim.y group striding
//    - Exact NVFP4 s_gate / s_up tail scales
// ===========================================================================
template <int TG>
__global__ void __launch_bounds__(256) hip_exp8_gu_fused_kernel(
    const unsigned long long* __restrict__ grp_ptr,
    const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups,
    const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq,
    HipExpertLayout L,
    block_q8_1* __restrict__ hq) {
    using S = HipSplit<TG>;
    __shared__ uint32_t sgrid[S::has_grid ? 256 : 1];
    __shared__ float res[kLdsNt][64];
    extern __shared__ int sx_raw[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if constexpr (S::has_grid) {
        sgrid[tid] = iq3xxs_grid[tid];
    }

    const int ng = *n_groups;
    const int nb = static_cast<int>(L.n_embd / S::qk);
    const int n_items = nb * S::ipb;
    const int xb = static_cast<int>(L.n_embd / 32);
    const int hb = static_cast<int>(L.n_ff / 32);
    const int r0 = static_cast<int>(blockIdx.x) * 32;
    const auto* sx = reinterpret_cast<const block_q8_1*>(sx_raw);

    for (int g = static_cast<int>(blockIdx.y); g < ng; g += static_cast<int>(gridDim.y)) {
        const int e0 = grp_start[g];
        const int e1 = grp_start[g + 1];
        if (e0 >= e1) continue;

        const auto* blob = reinterpret_cast<const uint8_t*>(grp_ptr[g]);
        const float s_gate = (TG == 40 && L.tail_off != 0u) ? reinterpret_cast<const float*>(blob + L.tail_off)[0] : 1.0f;
        const float s_up   = (TG == 40 && L.tail_off != 0u) ? reinterpret_cast<const float*>(blob + L.tail_off)[1] : 1.0f;

        auto wrow = [&](int i) -> const uint8_t* {
            return blob + (i < 32 ? std::size_t(0) : L.up_off) + static_cast<std::size_t>(r0 + (i & 31)) * L.gu_row;
        };

        for (int c0 = e0; c0 < e1; c0 += kLdsNt) {
            const int cn = min(kLdsNt, e1 - c0);
            hip_lds_barrier();
            for (int j = 0; j < cn; ++j) {
                const int* src = reinterpret_cast<const int*>(xq + static_cast<std::size_t>(ent_tok[c0 + j]) * xb);
                for (int i = tid; i < xb * 9; i += 256) sx_raw[j * xb * 9 + i] = src[i];
            }
            hip_lds_barrier();

            for (int p = 0; p < 4; ++p) {
                const int i0 = p * 16 + warp;
                const int i1 = i0 + 8;
                const uint8_t* w0 = wrow(i0);
                const uint8_t* w1 = wrow(i1);

                auto pass = [&](auto ntc_tag) {
                    constexpr int NTC = decltype(ntc_tag)::value;
                    float s0[NTC], s1[NTC];
#pragma unroll
                    for (int j = 0; j < NTC; ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
                    for (int k = lane; k < n_items; k += 32) {
                        const int kbx = k / S::ipb;
                        const int iqs = S::step * (k % S::ipb);
                        const typename S::W rw0 = S::load(w0, kbx, iqs, S::has_grid ? sgrid : nullptr);
                        const typename S::W rw1 = S::load(w1, kbx, iqs, S::has_grid ? sgrid : nullptr);
#pragma unroll
                        for (int j = 0; j < NTC; ++j) {
                            const block_q8_1* xk = sx + static_cast<std::size_t>(j) * xb + kbx * (S::qk / 32);
                            s0[j] += S::apply(rw0, xk, iqs);
                            s1[j] += S::apply(rw1, xk, iqs);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < NTC; ++j) {
                        const float a = hip_wsum32(s0[j]);
                        const float b = hip_wsum32(s1[j]);
                        if (lane == 0) { res[j][i0] = a; res[j][i1] = b; }
                    }
                };
                switch (cn) {
                    case 1: pass(std::integral_constant<int, 1>{}); break;
                    case 2: pass(std::integral_constant<int, 2>{}); break;
                    case 3: pass(std::integral_constant<int, 3>{}); break;
                    default: pass(std::integral_constant<int, 4>{}); break;
                }
            }

            hip_lds_barrier();
            if (warp < cn) {
                const float gg = res[warp][lane] * s_gate;
                const float uu = res[warp][32 + lane] * s_up;
                const float xi = (gg / (1.0f + __expf(-gg))) * uu;
                const float amax = hip_wmax32(fabsf(xi));
                const float sum = hip_wsum32(xi);
                const float d = hip_q8_1_finite(amax / 127.0f);
                const int8_t q = hip_q8_1_quant(xi, d, amax);
                block_q8_1* y = hq + static_cast<std::size_t>(c0 + warp) * hb + blockIdx.x;
                y->qs[lane] = q;
                if (lane == 0) y->ds = hip_q8_1_ds(d, sum);
            }
        }
    }
}

// ===========================================================================
// 3. LDS-Staged Down Projection (all 7 Unsloth quant types)
//    - 8-lane sub-warp path when n_items in {16, 20, 40} (zero idle lanes for n_ff = 512 / 640!)
//    - Exact NVFP4 s_down tail scale
//    - 65535-safe gridDim.y group striding
// ===========================================================================
template <int TD>
__global__ void __launch_bounds__(256) hip_down_lds_kernel(
    const unsigned long long* __restrict__ grp_ptr,
    const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups,
    const int32_t* __restrict__ ent_dst,
    const block_q8_1* __restrict__ hq,
    HipExpertLayout L,
    float* __restrict__ out) {
    using S = HipSplit<TD>;
    __shared__ uint32_t sgrid[S::has_grid ? 256 : 1];
    extern __shared__ int sh_raw[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if constexpr (S::has_grid) {
        sgrid[tid] = iq3xxs_grid[tid];
    }

    const int ng = *n_groups;
    const int nb = static_cast<int>(L.n_ff / S::qk);
    const int hb = static_cast<int>(L.n_ff / 32);
    const int n_items = nb * S::ipb;
    const int nrow = static_cast<int>(L.n_embd);
    const auto* sh = reinterpret_cast<const block_q8_1*>(sh_raw);
    const bool use_sub8 = (n_items == 16 || n_items == 20 || n_items == 40);

    for (int g = static_cast<int>(blockIdx.y); g < ng; g += static_cast<int>(gridDim.y)) {
        const int e0 = grp_start[g];
        const int e1 = grp_start[g + 1];
        if (e0 >= e1) continue;

        const auto* blob = reinterpret_cast<const uint8_t*>(grp_ptr[g]);
        const float sd = (TD == 40 && L.tail_off != 0u) ? reinterpret_cast<const float*>(blob + L.tail_off)[2] : 1.0f;

        for (int c0 = e0; c0 < e1; c0 += kLdsNt) {
            const int cn = min(kLdsNt, e1 - c0);
            hip_lds_barrier();
            {
                const int* src = reinterpret_cast<const int*>(hq + static_cast<std::size_t>(c0) * hb);
                for (int i = tid; i < cn * hb * 9; i += 256) sh_raw[i] = src[i];
            }
            hip_lds_barrier();

            if (use_sub8) {
                const int sub_row = tid >> 3;  // 0..31
                const int slane = tid & 7;     // 0..7
                for (int p = 0; p < 2; ++p) {
                    const int r = static_cast<int>(blockIdx.x) * kLdsRb + p * 32 + sub_row;
                    if (r >= nrow) break;
                    const uint8_t* w = blob + L.down_off + static_cast<std::size_t>(r) * L.d_row;
                    auto pass8 = [&](auto ntc_tag) {
                        constexpr int NTC = decltype(ntc_tag)::value;
                        float s[NTC];
#pragma unroll
                        for (int j = 0; j < NTC; ++j) s[j] = 0.0f;
                        for (int k = slane; k < n_items; k += 8) {
                            const int kbx = k / S::ipb;
                            const int iqs = S::step * (k % S::ipb);
                            const typename S::W rw = S::load(w, kbx, iqs, S::has_grid ? sgrid : nullptr);
#pragma unroll
                            for (int j = 0; j < NTC; ++j) {
                                const block_q8_1* xk = sh + static_cast<std::size_t>(j) * hb + kbx * (S::qk / 32);
                                s[j] += S::apply(rw, xk, iqs);
                            }
                        }
#pragma unroll
                        for (int j = 0; j < NTC; ++j) {
                            float v = s[j];
#pragma unroll
                            for (int o = 4; o > 0; o >>= 1) v += __shfl_xor(v, o, 8);
                            if (slane == 0) {
                                out[static_cast<std::size_t>(ent_dst[c0 + j]) * L.n_embd + r] = v * sd;
                            }
                        }
                    };
                    switch (cn) {
                        case 1: pass8(std::integral_constant<int, 1>{}); break;
                        case 2: pass8(std::integral_constant<int, 2>{}); break;
                        case 3: pass8(std::integral_constant<int, 3>{}); break;
                        default: pass8(std::integral_constant<int, 4>{}); break;
                    }
                }
            } else {
                for (int p = 0; p < kLdsRb / 16; ++p) {
                    const int r0 = static_cast<int>(blockIdx.x) * kLdsRb + p * 16 + warp;
                    const int r1 = r0 + 8;
                    if (r0 >= nrow) break;
                    const bool two = r1 < nrow;
                    const uint8_t* w0 = blob + L.down_off + static_cast<std::size_t>(r0) * L.d_row;
                    const uint8_t* w1 = blob + L.down_off + static_cast<std::size_t>(two ? r1 : r0) * L.d_row;
                    auto pass = [&](auto ntc_tag) {
                        constexpr int NTC = decltype(ntc_tag)::value;
                        float s0[NTC], s1[NTC];
#pragma unroll
                        for (int j = 0; j < NTC; ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
                        for (int k = lane; k < n_items; k += 32) {
                            const int kbx = k / S::ipb;
                            const int iqs = S::step * (k % S::ipb);
                            const typename S::W rw0 = S::load(w0, kbx, iqs, S::has_grid ? sgrid : nullptr);
                            const typename S::W rw1 = S::load(w1, kbx, iqs, S::has_grid ? sgrid : nullptr);
#pragma unroll
                            for (int j = 0; j < NTC; ++j) {
                                const block_q8_1* xk = sh + static_cast<std::size_t>(j) * hb + kbx * (S::qk / 32);
                                s0[j] += S::apply(rw0, xk, iqs);
                                s1[j] += S::apply(rw1, xk, iqs);
                            }
                        }
#pragma unroll
                        for (int j = 0; j < NTC; ++j) {
                            const float a = hip_wsum32(s0[j]) * sd;
                            const float b = hip_wsum32(s1[j]) * sd;
                            if (lane == 0) {
                                out[static_cast<std::size_t>(ent_dst[c0 + j]) * L.n_embd + r0] = a;
                                if (two) out[static_cast<std::size_t>(ent_dst[c0 + j]) * L.n_embd + r1] = b;
                            }
                        }
                    };
                    switch (cn) {
                        case 1: pass(std::integral_constant<int, 1>{}); break;
                        case 2: pass(std::integral_constant<int, 2>{}); break;
                        case 3: pass(std::integral_constant<int, 3>{}); break;
                        default: pass(std::integral_constant<int, 4>{}); break;
                    }
                }
            }
        }
    }
}

}  // namespace

// ===========================================================================
// Public Host API
// ===========================================================================

bool hip_unsloth_dp4a_mmvq(int ggml_type, const void* W_quant, const void* X_q8_1, float* Y,
                           int n_in, int n_out, int ncols, float global_scale, void* stream) {
    if (!W_quant || !X_q8_1 || !Y || n_in <= 0 || n_out <= 0 || ncols < 1) return false;
    const std::size_t rb = hip_quant_row_bytes(ggml_type, n_in);
    if (rb == 0u) return false;

    const auto* wb = static_cast<const uint8_t*>(W_quant);
    const auto* xq = static_cast<const block_q8_1*>(X_q8_1);
    hipStream_t s = static_cast<hipStream_t>(stream);

    switch (ggml_type) {
        case 8:  launch_dp4a_mmvq_ty<8>(wb, rb, xq, Y, n_in, n_out, ncols, global_scale, s); break;
        case 12: launch_dp4a_mmvq_ty<12>(wb, rb, xq, Y, n_in, n_out, ncols, global_scale, s); break;
        case 13: launch_dp4a_mmvq_ty<13>(wb, rb, xq, Y, n_in, n_out, ncols, global_scale, s); break;
        case 14: launch_dp4a_mmvq_ty<14>(wb, rb, xq, Y, n_in, n_out, ncols, global_scale, s); break;
        case 18: launch_dp4a_mmvq_ty<18>(wb, rb, xq, Y, n_in, n_out, ncols, global_scale, s); break;
        case 23: launch_dp4a_mmvq_ty<23>(wb, rb, xq, Y, n_in, n_out, ncols, global_scale, s); break;
        case 40: launch_dp4a_mmvq_ty<40>(wb, rb, xq, Y, n_in, n_out, ncols, global_scale, s); break;
        default: return false;
    }
    return hipGetLastError() == hipSuccess;
}

bool hip_unsloth_exp8_gu_fused(const HipExpertLayout& L, const unsigned long long* grp_ptr,
                               const int32_t* grp_start, const int32_t* n_groups,
                               const int32_t* ent_tok, int64_t cap_groups,
                               const void* x_q8_1, void* h_q8_1, void* stream) {
    // CRITICAL GUARD: Hadamard-folded down weights MUST NOT fuse SwiGLU+q8_1 directly without rotation!
    if (L.had_block != 0) return false;
    if (cap_groups <= 0 || L.n_ff <= 0 || (L.n_ff % 32) != 0) return false;
    if (hip_quant_row_bytes(L.gu_type, L.n_embd) == 0u) return false;
    if (!grp_ptr || !grp_start || !n_groups || !ent_tok || !x_q8_1 || !h_q8_1) return false;

    const unsigned gy = static_cast<unsigned>(std::min<int64_t>(cap_groups, kMaxGridY));
    const dim3 grid(static_cast<unsigned>(L.n_ff / 32), gy, 1);
    const std::size_t sh = static_cast<std::size_t>(kLdsNt) * static_cast<std::size_t>(L.n_embd / 32) * sizeof(block_q8_1);
    // Keep total LDS (dynamic + static res + optional sgrid) within 64 KiB RDNA limit.
    if (sh + kLdsNt * 64u * sizeof(float) + 1024u > 65536u) return false;

    hipStream_t s = static_cast<hipStream_t>(stream);
    const auto* xq = static_cast<const block_q8_1*>(x_q8_1);
    auto* hq = static_cast<block_q8_1*>(h_q8_1);

    switch (L.gu_type) {
        case 8:  hip_exp8_gu_fused_kernel<8><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, xq, L, hq); break;
        case 12: hip_exp8_gu_fused_kernel<12><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, xq, L, hq); break;
        case 13: hip_exp8_gu_fused_kernel<13><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, xq, L, hq); break;
        case 14: hip_exp8_gu_fused_kernel<14><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, xq, L, hq); break;
        case 18: hip_exp8_gu_fused_kernel<18><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, xq, L, hq); break;
        case 23: hip_exp8_gu_fused_kernel<23><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, xq, L, hq); break;
        case 40: hip_exp8_gu_fused_kernel<40><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, xq, L, hq); break;
        default: return false;
    }
    return hipGetLastError() == hipSuccess;
}

bool hip_unsloth_down_lds(const HipExpertLayout& L, const unsigned long long* grp_ptr,
                          const int32_t* grp_start, const int32_t* n_groups,
                          const int32_t* ent_dst, int64_t cap_groups,
                          const void* h_q8_1, float* out, void* stream) {
    if (cap_groups <= 0 || L.n_embd <= 0 || L.n_ff <= 0) return false;
    if (hip_quant_row_bytes(L.d_type, L.n_ff) == 0u) return false;
    if (!grp_ptr || !grp_start || !n_groups || !ent_dst || !h_q8_1 || !out) return false;

    const unsigned gy = static_cast<unsigned>(std::min<int64_t>(cap_groups, kMaxGridY));
    const dim3 grid(static_cast<unsigned>((L.n_embd + kLdsRb - 1) / kLdsRb), gy, 1);
    const std::size_t sh = static_cast<std::size_t>(kLdsNt) * static_cast<std::size_t>(L.n_ff / 32) * sizeof(block_q8_1);
    if (sh + 1024u > 65536u) return false;

    hipStream_t s = static_cast<hipStream_t>(stream);
    const auto* hq = static_cast<const block_q8_1*>(h_q8_1);

    switch (L.d_type) {
        case 8:  hip_down_lds_kernel<8><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 12: hip_down_lds_kernel<12><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 13: hip_down_lds_kernel<13><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 14: hip_down_lds_kernel<14><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 18: hip_down_lds_kernel<18><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 23: hip_down_lds_kernel<23><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 40: hip_down_lds_kernel<40><<<grid, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        default: return false;
    }
    return hipGetLastError() == hipSuccess;
}

}  // namespace strata::kernels::hip_unsloth
