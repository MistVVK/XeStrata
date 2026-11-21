// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// src/kernels/cuda/rdna_mmvq.cu - RDNA3/RDNA3.5 (gfx1100..gfx1151) Wave32 multi-column decode GEMV & T=8..16 WMMA verify:
//   - rdna_q6k_kernel<NC>: Q6_K x q8_1 decode GEMV (one Wave32 per row, DPP reduction, NC=1..8, sliced for 9..16)
//   - rdna_kq_kernel<FMT, NC>: Q4_K (12), Q5_K (13), IQ4_XS (23), IQ3_S (21) using strata-quant-common.h iq3s_grid
//   - rdna_q80_wmma_t8_kernel: T=8..16 Q8_0 x q8_1 direct Wave32 INT8 WMMA (__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
// Enabled via STRATA_RDNA_Q6K=1 or STRATA_HIP_FAST_DEC=1 on gfx1100..gfx1151 (zero rocWMMA dependency).
#include "strata/kernels/gfx_arch.hpp"
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "strata/quant/strata-quant-common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

bool rdna_q6k_enabled();
bool rdna_q6k_mmvq(const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream);
bool rdna_q6k_mmvq2(const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream,
                    const void* w2, float* y2, int n_out2);
bool rdna_kq_mmvq(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream);
bool rdna_kq_mmvq2(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream,
                   const void* w2, float* y2, int n_out2);
bool rdna_q80_wmma_mmvq(const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream);

#if defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_HIP_GFX906)
namespace {

struct Q81 {
    half2 ds;
    int8_t qs[32];
};
struct Q80 {
    half d;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36 && sizeof(Q80) == 34);

constexpr int QK = 256;
constexpr int BB_Q6K = 210;

__device__ __forceinline__ float wsum32(float v) {
#define STRATA_DPP6(val, c) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(val), c, 0xf, 0xf, false))
    v += STRATA_DPP6(v, 0x161);
    v += STRATA_DPP6(v, 0x162);
    v += STRATA_DPP6(v, 0x164);
    v += STRATA_DPP6(v, 0x168);
    v += __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210, 0xfedcba98, false, false));
#undef STRATA_DPP6
    return v;
}

__device__ __forceinline__ uint32_t ld32u(const uint8_t* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

__device__ __forceinline__ int dot4_s8(int w, int a, int acc) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, w, true, a, acc, false);
#else
    return __builtin_amdgcn_sdot4(w, a, acc, false);
#endif
}

__device__ __forceinline__ int dot2(int w0, int w1, const int* q, int acc) {
    acc = dot4_s8(w0, q[0], acc);
    return dot4_s8(w1, q[1], acc);
}

template <int NC, int UNR = (NC <= 2 ? 5 : NC <= 4 ? 3 : 2)>
__global__ void __launch_bounds__(256) rdna_q6k_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x,
                                                       float* __restrict__ y, int n_in, int n_out,
                                                       const uint8_t* __restrict__ w2, float* __restrict__ y2, int n_out2) {
    const int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (w2 != nullptr && row >= n_out) {
        row -= n_out;
        w = w2;
        y = y2;
        n_out = n_out2;
    }
    if (row >= n_out) return;
    const int nsb = n_in / QK, xs = n_in / 32;
    const int n = lane >> 4, qd = (lane >> 2) & 3, l = (lane & 3) * 8;
    const int ql_off = n * 64 + (qd & 1) * 32 + l;
    const int qh_off = 128 + n * 32 + l;
    const int sc_off = 192 + n * 8 + (l >> 4) + 2 * qd;
    const int xb = n * 4 + qd, xi = l >> 2;
    const int hs = 2 * qd, ns = (qd >= 2) ? 4 : 0;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    const uint8_t* wr = w + (size_t) row * nsb * BB_Q6K;
#pragma unroll UNR
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* b = wr + (size_t) sb * BB_Q6K;
        const uint32_t l0 = ld32u(b + ql_off), l1 = ld32u(b + ql_off + 4);
        const uint32_t h0 = ld32u(b + qh_off), h1 = ld32u(b + qh_off + 4);
        const int sc = (int) (int8_t) b[sc_off];
        const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
        const uint32_t lo0 = (l0 >> ns) & 0x0f0f0f0fu, lo1 = (l1 >> ns) & 0x0f0f0f0fu;
        const uint32_t hi0 = ((h0 >> hs) & 0x03030303u) << 4, hi1 = ((h1 >> hs) & 0x03030303u) << 4;
        const int w0 = (int) ((((lo0 | hi0) | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
        const int w1 = (int) ((((lo1 | hi1) | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
        const float ds = d * (float) sc;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const Q81* xb8 = x + (size_t) c * xs + sb * 8 + xb;
            const int* q = reinterpret_cast<const int*>(xb8->qs) + xi;
            int s = dot4_s8(w0, q[0], 0);
            s = dot4_s8(w1, q[1], s);
            acc[c] = fmaf(ds * __low2float(xb8->ds), (float) s, acc[c]);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = wsum32(acc[c]);
        if (lane == 0) y[(size_t) c * n_out + row] = v;
    }
}

__device__ __forceinline__ uint32_t iq4_lut4(uint32_t idx) {
    const uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    const uint32_t sel = idx & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(t1, t0, sel), hi = __builtin_amdgcn_perm(t3, t2, sel);
    const uint32_t msk = ((idx >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~msk) | (hi & msk);
}

__device__ __forceinline__ int neg_by_signbits4(uint32_t g, uint32_t mask4) {
    // mask4 bits 0..3 negate bytes 0..3 of g (each byte in g is small positive magnitude 1..15)
    const uint32_t m0 = (mask4 & 1u) ? 0x000000FFu : 0u;
    const uint32_t m1 = (mask4 & 2u) ? 0x0000FF00u : 0u;
    const uint32_t m2 = (mask4 & 4u) ? 0x00FF0000u : 0u;
    const uint32_t m3 = (mask4 & 8u) ? 0xFF000000u : 0u;
    const uint32_t sg = m0 | m1 | m2 | m3;
    // two's complement per byte without cross-byte borrow (no byte is 0x80)
    const uint32_t a = g ^ sg;
    const uint32_t b = (sg & 0x01010101u);
    return (int) (a + b);
}

template <int FMT> struct KFmt;
template <> struct KFmt<12> { static constexpr int B = 144; };
template <> struct KFmt<13> { static constexpr int B = 176; };
template <> struct KFmt<23> { static constexpr int B = 136; };
template <> struct KFmt<21> { static constexpr int B = 110; };

template <int FMT, int NC, int UNR = (FMT == 21 ? 2 : (NC <= 2 ? 5 : NC <= 4 ? 3 : 2))>
__global__ void __launch_bounds__(256) rdna_kq_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x,
                                                      float* __restrict__ y, int n_in, int n_out,
                                                      const uint8_t* __restrict__ w2, float* __restrict__ y2, int n_out2) {
    constexpr int BB_ = KFmt<FMT>::B;
    const int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    __shared__ uint32_t s_grid[FMT == 21 ? 512 : 1];
    if constexpr (FMT == 21) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) s_grid[i] = iq3s_grid[i];
        __syncthreads();
    }
    if (w2 != nullptr && row >= n_out) {
        row -= n_out;
        w = w2;
        y = y2;
        n_out = n_out2;
    }
    if (row >= n_out) return;
    const int nsb = n_in / QK, xs = n_in / 32;
    const int j = lane >> 2, l0 = (lane & 3) * 8;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    const uint8_t* wr = w + (size_t) row * nsb * BB_;
#pragma unroll UNR
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* b = wr + (size_t) sb * BB_;
        int w0, w1;
        float f1, f2 = 0.0f;
        if constexpr (FMT == 21) {
            const int l = lane & 3;
            const uint8_t* qs = b + 2 + 8 * j + 2 * l;
            const int qh = b[66 + j];
            const uint32_t sg = b[74 + 4 * j + l];
            const uint32_t g1 = s_grid[qs[0] | ((qh << (8 - 2 * l)) & 0x100)];
            const uint32_t g2 = s_grid[qs[1] | ((qh << (7 - 2 * l)) & 0x100)];
            w0 = neg_by_signbits4(g1, sg & 0x0Fu);
            w1 = neg_by_signbits4(g2, (sg >> 4) & 0x0Fu);
            const int sc = (b[106 + j / 2] >> (4 * (j & 1))) & 0x0f;
            f1 = __half2float(*reinterpret_cast<const __half*>(b)) * (float) (1 + 2 * sc);
        } else if constexpr (FMT == 23) {
            const int hl = l0 >= 16 ? 4 : 0;
            const uint32_t q0 = ld32u(b + 8 + 16 * j + (l0 & 15)), q1 = ld32u(b + 8 + 16 * j + (l0 & 15) + 4);
            w0 = (int) iq4_lut4((q0 >> hl) & 0x0f0f0f0fu);
            w1 = (int) iq4_lut4((q1 >> hl) & 0x0f0f0f0fu);
            const uint16_t sh = *reinterpret_cast<const uint16_t*>(b + 2);
            const int ls = ((b[4 + j / 2] >> (4 * (j & 1))) & 0xf) | (((sh >> (2 * j)) & 3) << 4);
            f1 = __half2float(*reinterpret_cast<const __half*>(b)) * (float) (ls - 32);
        } else {
            const int qo = (FMT == 12 ? 16 : 48) + (j >> 1) * 32 + l0, ns = 4 * (j & 1);
            uint32_t v0 = (ld32u(b + qo) >> ns) & 0x0f0f0f0fu, v1 = (ld32u(b + qo + 4) >> ns) & 0x0f0f0f0fu;
            if constexpr (FMT == 13) {
                v0 |= ((ld32u(b + 16 + l0) >> j) & 0x01010101u) << 4;
                v1 |= ((ld32u(b + 16 + l0 + 4) >> j) & 0x01010101u) << 4;
            }
            w0 = (int) v0;
            w1 = (int) v1;
            const uint32_t s0 = ld32u(b + 4), s1 = ld32u(b + 8), s2 = ld32u(b + 12);
            auto sb8 = [&](int k) -> int {
                const uint32_t wd = k < 4 ? s0 : (k < 8 ? s1 : s2);
                return (int) ((wd >> (8 * (k & 3))) & 0xffu);
            };
            int sc, m;
            if (j < 4) { sc = sb8(j) & 63; m = sb8(j + 4) & 63; }
            else { sc = (sb8(j + 4) & 0xF) | ((sb8(j - 4) >> 6) << 4); m = (sb8(j + 4) >> 4) | ((sb8(j) >> 6) << 4); }
            const float2 dm = __half22float2(*reinterpret_cast<const half2*>(b));
            f1 = dm.x * (float) sc;
            f2 = dm.y * (float) m;
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const Q81* xb8 = x + (size_t) c * xs + sb * 8 + j;
            const int* q = reinterpret_cast<const int*>(xb8->qs) + (l0 >> 2);
            const float d8 = __low2float(xb8->ds);
            const int s1 = dot2(w0, w1, q, 0);
            if constexpr (FMT == 23 || FMT == 21) {
                acc[c] = fmaf(d8 * f1, (float) s1, acc[c]);
            } else {
                const int s2 = dot2(0x01010101, 0x01010101, q, 0);
                acc[c] = fmaf(d8, f1 * (float) s1 - f2 * (float) s2, acc[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = wsum32(acc[c]);
        if (lane == 0) y[(size_t) c * n_out + row] = v;
    }
}

template <int FMT>
void kq_launch_slice(const uint8_t* wb, const Q81* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s,
                     const uint8_t* w2, float* y2, int n_out2) {
    const unsigned blocks = (unsigned) ((n_out + 7) / 8 + (w2 ? (n_out2 + 7) / 8 : 0));
    switch (ncols) {
        case 1: rdna_kq_kernel<FMT, 1><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 2: rdna_kq_kernel<FMT, 2><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 3: rdna_kq_kernel<FMT, 3><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 4: rdna_kq_kernel<FMT, 4><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 5: rdna_kq_kernel<FMT, 5><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 6: rdna_kq_kernel<FMT, 6><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 7: rdna_kq_kernel<FMT, 7><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        default: rdna_kq_kernel<FMT, 8><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
    }
}

// ---- T=8..16 Wave32 INT8 WMMA target-verify GEMM for Q8_0 x Q8_1 (direct __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32) ----
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_RDNA_WMMA_I8 1
#else
#define STRATA_RDNA_WMMA_I8 0
#endif

using v4i = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using v8i = __attribute__((__vector_size__(8 * sizeof(int)))) int;

// 1 block = 4 waves (128 threads), 64 output rows (16 per wave) x up to 16 token columns.
// Each K step processes one 32-element Q8_0/Q8_1 block via two 16x16x16 INT8 WMMA ops, scaling per-block in FP32.
__global__ void __launch_bounds__(128) rdna_q80_wmma_t8_kernel(const Q80* __restrict__ w, const Q81* __restrict__ x,
                                                               float* __restrict__ y, int n_in, int n_out, int ncols) {
#if STRATA_RDNA_WMMA_I8
    const int tid = threadIdx.x;
    const int wv = tid >> 5;
    const int lane = tid & 31;
    const int r16 = lane & 15;
    const int half = lane >> 4;
    const int row0 = (blockIdx.x * 4 + wv) * 16;
    const int nb = n_in / 32;

    __shared__ int s_x[16][8];      // [col 0..15][8 int32 = 32 int8]
    __shared__ float s_xd[16];      // [col 0..15] q8_1 scale

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int blk = 0; blk < nb; ++blk) {
        // Cooperatively stage ncols x 32 int8 activations + scales into LDS (128 threads = 16 cols x 8 int32)
        const int sc = tid >> 3;
        const int si = tid & 7;
        if (sc < ncols) {
            const Q81* xb = x + (size_t) sc * nb + blk;
            s_x[sc][si] = reinterpret_cast<const int*>(xb->qs)[si];
            if (si == 0) s_xd[sc] = __low2float(xb->ds);
        } else {
            s_x[sc][si] = 0;
            if (si == 0) s_xd[sc] = 0.0f;
        }
        __syncthreads();

        // Load 16 rows x 32 int8 weights for this wave: lane (half, r16) loads row (row0 + r16)
        const int row = row0 + r16;
        v4i a0 = {0, 0, 0, 0}, a1 = {0, 0, 0, 0};
        float wd = 0.0f;
        if (row < n_out) {
            const Q80* wb = w + (size_t) row * nb + blk;
            wd = __half2float(wb->d);
            const auto* q16 = reinterpret_cast<const uint16_t*>(wb->qs);
            auto ld32 = [&](int idx) -> int {
                return (int) q16[2 * idx] | ((int) q16[2 * idx + 1] << 16);
            };
            a0[0] = ld32(0); a0[1] = ld32(1); a0[2] = ld32(2); a0[3] = ld32(3);
            a1[0] = ld32(4); a1[1] = ld32(5); a1[2] = ld32(6); a1[3] = ld32(7);
        }
        v4i b0 = {s_x[r16][0], s_x[r16][1], s_x[r16][2], s_x[r16][3]};
        v4i b1 = {s_x[r16][4], s_x[r16][5], s_x[r16][6], s_x[r16][7]};
        v8i ci = {0, 0, 0, 0, 0, 0, 0, 0};
        ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, ci, false);
        ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, ci, false);
        // Output layout on Wave32 WMMA: lane (half, col=r16) holds rows (2*k + half) for k=0..7 at column r16!
        // Fetch wd for row (row0 + 2*k + half) via lane shuffle from lane (2*k + half)
        const float xd = s_xd[r16];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float wdk = __shfl_sync(0xffffffffu, wd, 2 * k + half);
            acc[k] = fmaf(wdk * xd, (float) ci[k], acc[k]);
        }
        __syncthreads();
    }

    if (r16 < ncols) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int out_row = row0 + 2 * k + half;
            if (out_row < n_out) y[(size_t) r16 * n_out + out_row] = acc[k];
        }
    }
#else
    (void) w; (void) x; (void) y; (void) n_in; (void) n_out; (void) ncols;
#endif
}

bool arch_is_rdna3_or_4() {
    static int ok[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (ok[dev] == 0) {
        cudaDeviceProp p{};
        ok[dev] = (cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                   (gfx_arch_is_gfx11_wmma(p.gcnArchName) || gfx_arch_is_gfx12(p.gcnArchName))) ? 2 : 1;
    }
    return ok[dev] == 2;
}

bool arch_is_gfx11_wmma() {
    static int ok[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (ok[dev] == 0) {
        cudaDeviceProp p{};
        ok[dev] = (cudaGetDeviceProperties(&p, dev) == cudaSuccess && gfx_arch_is_gfx11_wmma(p.gcnArchName)) ? 2 : 1;
    }
    return ok[dev] == 2;
}

}  // namespace

bool rdna_q6k_enabled() {
    static const int env = [] {
        if (const char* v = std::getenv("STRATA_RDNA_Q6K")) return v[0] == '1' ? 1 : 0;
        if (const char* f = std::getenv("STRATA_HIP_FAST_DEC")) return f[0] == '1' ? 2 : 0;
        return 0;
    }();
    if (env == 0) return false;
    return env == 1 ? true : arch_is_rdna3_or_4();
}

bool rdna_kq_mmvq(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream) {
    return rdna_kq_mmvq2(type, w, xq, y, n_in, n_out, ncols, stream, nullptr, nullptr, 0);
}

bool rdna_kq_mmvq2(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream,
                   const void* w2v, float* y2, int n_out2) {
    if (!rdna_q6k_enabled() || n_in % QK != 0 || ncols < 1 || ncols > 16) return false;
    if (w2v != nullptr && n_out % 8 != 0) return false;
    const auto* wb = static_cast<const uint8_t*>(w);
    const auto* w2 = static_cast<const uint8_t*>(w2v);
    const auto* x = static_cast<const Q81*>(xq);
    cudaStream_t s = (cudaStream_t) stream;
    const int xs = n_in / 32;
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = (ncols - c0) < 8 ? (ncols - c0) : 8;
        const Q81* xc = x + (size_t) c0 * xs;
        float* yc = y + (size_t) c0 * n_out;
        float* y2c = y2 ? (y2 + (size_t) c0 * n_out2) : nullptr;
        switch (type) {
            case 12:
                if (ncols < 4 && std::getenv("STRATA_RDNA_KQ_ALL") == nullptr) return false;
                kq_launch_slice<12>(wb, xc, yc, n_in, n_out, nc, s, w2, y2c, n_out2);
                break;
            case 13:
                if (ncols < 4 && std::getenv("STRATA_RDNA_KQ_ALL") == nullptr) return false;
                kq_launch_slice<13>(wb, xc, yc, n_in, n_out, nc, s, w2, y2c, n_out2);
                break;
            case 23:
                kq_launch_slice<23>(wb, xc, yc, n_in, n_out, nc, s, w2, y2c, n_out2);
                break;
            case 21:
                kq_launch_slice<21>(wb, xc, yc, n_in, n_out, nc, s, w2, y2c, n_out2);
                break;
            default:
                return false;
        }
    }
    return true;
}

bool rdna_q6k_mmvq(const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream) {
    return rdna_q6k_mmvq2(w, xq, y, n_in, n_out, ncols, stream, nullptr, nullptr, 0);
}

bool rdna_q6k_mmvq2(const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream,
                    const void* w2v, float* y2, int n_out2) {
    if (!rdna_q6k_enabled() || n_in % QK != 0 || ncols < 1 || ncols > 16) return false;
    if (w2v != nullptr && n_out % 8 != 0) return false;
    const auto* wb = static_cast<const uint8_t*>(w);
    const auto* w2 = static_cast<const uint8_t*>(w2v);
    const auto* x = static_cast<const Q81*>(xq);
    const unsigned blocks = (unsigned) ((n_out + 7) / 8 + (w2 ? (n_out2 + 7) / 8 : 0));
    cudaStream_t s = (cudaStream_t) stream;
    const int xs = n_in / 32;
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = (ncols - c0) < 8 ? (ncols - c0) : 8;
        const Q81* xc = x + (size_t) c0 * xs;
        float* yc = y + (size_t) c0 * n_out;
        float* y2c = y2 ? (y2 + (size_t) c0 * n_out2) : nullptr;
        switch (nc) {
            case 1: rdna_q6k_kernel<1><<<blocks, 256, 0, s>>>(wb, xc, yc, n_in, n_out, w2, y2c, n_out2); break;
            case 2: rdna_q6k_kernel<2><<<blocks, 256, 0, s>>>(wb, xc, yc, n_in, n_out, w2, y2c, n_out2); break;
            case 3: rdna_q6k_kernel<3><<<blocks, 256, 0, s>>>(wb, xc, yc, n_in, n_out, w2, y2c, n_out2); break;
            case 4: rdna_q6k_kernel<4><<<blocks, 256, 0, s>>>(wb, xc, yc, n_in, n_out, w2, y2c, n_out2); break;
            case 5: rdna_q6k_kernel<5><<<blocks, 256, 0, s>>>(wb, xc, yc, n_in, n_out, w2, y2c, n_out2); break;
            case 6: rdna_q6k_kernel<6><<<blocks, 256, 0, s>>>(wb, xc, yc, n_in, n_out, w2, y2c, n_out2); break;
            case 7: rdna_q6k_kernel<7><<<blocks, 256, 0, s>>>(wb, xc, yc, n_in, n_out, w2, y2c, n_out2); break;
            default: rdna_q6k_kernel<8><<<blocks, 256, 0, s>>>(wb, xc, yc, n_in, n_out, w2, y2c, n_out2); break;
        }
    }
    return true;
}

bool rdna_q80_wmma_mmvq(const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!rdna_q6k_enabled() || !arch_is_gfx11_wmma()) return false;
    static const bool wmma_dec_off = [] {
        const char* v = std::getenv("STRATA_RDNA_Q80_WMMA");
        return v != nullptr && v[0] == '0';
    }();
    if (wmma_dec_off) return false;
    if (ncols < 8 || ncols > 16 || (n_in % 32) != 0 || n_out < 64) return false;
    const unsigned blocks = (unsigned) ((n_out + 63) / 64);
    rdna_q80_wmma_t8_kernel<<<blocks, 128, 0, (cudaStream_t) stream>>>(
        static_cast<const Q80*>(w), static_cast<const Q81*>(xq), y, n_in, n_out, ncols);
    return true;
}

#else

bool rdna_q6k_enabled() { return false; }
bool rdna_q6k_mmvq(const void*, const void*, float*, int, int, int, void*) { return false; }
bool rdna_q6k_mmvq2(const void*, const void*, float*, int, int, int, void*, const void*, float*, int) { return false; }
bool rdna_kq_mmvq(int, const void*, const void*, float*, int, int, int, void*) { return false; }
bool rdna_kq_mmvq2(int, const void*, const void*, float*, int, int, int, void*, const void*, float*, int) { return false; }
bool rdna_q80_wmma_mmvq(const void*, const void*, float*, int, int, int, void*) { return false; }

#endif

}  // namespace strata::kernels
