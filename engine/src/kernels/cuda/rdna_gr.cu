// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// src/kernels/cuda/rdna_gr.cu - RDNA3/RDNA3.5 (gfx1100..gfx1151) Wave32 DPP hyper-connection read:
// split-K down + partial-folding up (opt-in via STRATA_GR_RDNA=1 or STRATA_HIP_FAST_DEC=1 on gfx11).
// Ported from MIT Strata-rdna3 with CU-count-safe persistent fusion gating and WDDM-safe wall_clock64() spin guards
// (rdna3_wddm_guard.hpp), operating directly on 0.2.18 FusedGrArgs (BF16 and s_down/s_up per-32 FP32-scale int8).
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/kernels/rdna3_wddm_guard.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

bool rdna_gr_enabled();
void rdna_gr_read_multi(const FusedGrArgs* a, int n_tok, float* scratch, void* stream,
                        unsigned long long* sb = nullptr, int si = 0);

#if defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_HIP_GFX906)
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, PR = LR + HC;
constexpr int H = N / 2;
constexpr int TH = 256, WV = TH / 32;
constexpr int HQ = H / 8 / 32;
constexpr int UC = 16;
constexpr int UB = N / UC;

unsigned* g_gr_bar[64] = {};
int g_gr_can_fuse[64] = {};   // 0 unknown, 1 no, 2 yes

struct RdnaGr {
    FusedGrArgs a[kFusedGrMaxT];
    float* part = nullptr;
    float* ss = nullptr;
    unsigned* bar = nullptr;
};

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint4 ntload(const uint4* p) {
    const u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p));
    return make_uint4(v.x, v.y, v.z, v.w);
}
__device__ __forceinline__ uint2 ntload2(const uint2* p) {
    const u32x2 v = __builtin_nontemporal_load(reinterpret_cast<const u32x2*>(p));
    return make_uint2(v.x, v.y);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }

#define STRATA_GR_DPP(v, ctrl) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, 0xf, 0xf, false))

__device__ __forceinline__ float wsum(float v) {
    v += STRATA_GR_DPP(v, 0xB1);
    v += STRATA_GR_DPP(v, 0x4E);
    v += STRATA_GR_DPP(v, 0x141);
    v += STRATA_GR_DPP(v, 0x140);
    v += __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210, 0xfedcba98, false, false));
    return v;
}

__device__ __forceinline__ float max8(float v) {
    v = fmaxf(v, STRATA_GR_DPP(v, 0x161));
    v = fmaxf(v, STRATA_GR_DPP(v, 0x162));
    v = fmaxf(v, STRATA_GR_DPP(v, 0x164));
    return v;
}

__device__ __forceinline__ void quant4(float4 x, uint32_t* q, float* d, bool first) {
    const float amax = max8(fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w))));
    const float dd = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int a = __float2int_rn(x.x * id), b = __float2int_rn(x.y * id);
    const int c = __float2int_rn(x.z * id), e = __float2int_rn(x.w * id);
    *q = (uint32_t) (a & 0xff) | ((uint32_t) (b & 0xff) << 8) | ((uint32_t) (c & 0xff) << 16) | ((uint32_t) (e & 0xff) << 24);
    if (first) *d = dd;
}

__device__ __forceinline__ int dot8i(uint2 w, uint2 a) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, (int) w.y, true, (int) a.y,
                                   __builtin_amdgcn_sudot4(true, (int) w.x, true, (int) a.x, 0, false), false);
#else
    return __builtin_amdgcn_sdot4((int) w.y, (int) a.y, __builtin_amdgcn_sdot4((int) w.x, (int) a.x, 0, false), false);
#endif
}

__device__ __forceinline__ float dot8(const uint4 w, const float4 x0, const float4 x1, float acc) {
    acc = fmaf(__uint_as_float(w.x << 16), x0.x, acc); acc = fmaf(__uint_as_float(w.x & 0xffff0000u), x0.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), x0.z, acc); acc = fmaf(__uint_as_float(w.y & 0xffff0000u), x0.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), x1.x, acc); acc = fmaf(__uint_as_float(w.z & 0xffff0000u), x1.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), x1.z, acc); acc = fmaf(__uint_as_float(w.w & 0xffff0000u), x1.w, acc);
    return acc;
}

__device__ __forceinline__ float dot8q(const uint2 w, float d, const float4 x0, const float4 x1, float acc) {
    float s = 0.0f;
    s = fmaf((float) (int8_t) (w.x), x0.x, s);       s = fmaf((float) (int8_t) (w.x >> 8), x0.y, s);
    s = fmaf((float) (int8_t) (w.x >> 16), x0.z, s); s = fmaf((float) (int8_t) (w.x >> 24), x0.w, s);
    s = fmaf((float) (int8_t) (w.y), x1.x, s);       s = fmaf((float) (int8_t) (w.y >> 8), x1.y, s);
    s = fmaf((float) (int8_t) (w.y >> 16), x1.z, s); s = fmaf((float) (int8_t) (w.y >> 24), x1.w, s);
    return fmaf(d, s, acc);
}

template <int T, int RW>
__global__ void __launch_bounds__(TH) rdna_gr_down(RdnaGr m) {
    __shared__ __align__(16) float xs[T][H];
    __shared__ float sq[WV][T];
    const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
    const int rg = blockIdx.x, slice = blockIdx.y, c = slice >> 1, h = slice & 1;
    uint4 wv4[RW][HQ];
    bool act[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int row = (rg * WV + wv) * RW + r;
        act[r] = row < LR || (row < PR && m.a[0].w_inject != nullptr);
        const uint16_t* wbase = row < LR ? m.a[0].w_down + (size_t) row * D : m.a[0].w_inject + (size_t) (row - LR) * D;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + c * N + h * H);
        if (act[r]) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv4[r][q] = ntload(w4 + lane + 32 * q);
        }
    }
    const int e0 = c * N + h * H;
#pragma unroll 4
    for (int it = t; it < T * (H / 4); it += TH) {
        const int k = it / (H / 4), i = (it - k * (H / 4)) * 4;
        const FusedGrArgs& a = m.a[k];
        float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
        if (a.apply) {
            const float gw = 2.0f * sigm(a.inj_prev[c] / (float) HC);
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + e0 + i);
        *reinterpret_cast<float4*>(&xs[k][i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    if (rg == 0) {
#pragma unroll 1
        for (int k = 0; k < T; ++k) {
            const FusedGrArgs& a = m.a[k];
            const float gw = a.apply ? 2.0f * sigm(a.inj_prev[c] / (float) HC) : 0.0f;
            float s = 0.0f;
            for (int i = t * 4; i < H; i += TH * 4) {
                float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
                if (a.apply) {
                    const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
                    r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
                }
                s = fmaf(r.w, r.w, fmaf(r.z, r.z, fmaf(r.y, r.y, fmaf(r.x, r.x, s))));
            }
            const float v = wsum(s);
            if (lane == 0) sq[wv][k] = v;
        }
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WV; ++w) s += sq[w][t];
        m.ss[t * 2 * HC + slice] = s;
    }
#pragma unroll 1
    for (int k0 = 0; k0 < T; k0 += 2) {
        const bool two = k0 + 1 < T;
        float acc0[RW], acc1[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
#pragma unroll
        for (int q = 0; q < HQ; ++q) {
            const int j = (lane + 32 * q) * 8;
            const float4 a0 = *reinterpret_cast<const float4*>(&xs[k0][j]);
            const float4 a1 = *reinterpret_cast<const float4*>(&xs[k0][j + 4]);
            float4 b0 = make_float4(0.f, 0.f, 0.f, 0.f), b1 = b0;
            if (two) {
                b0 = *reinterpret_cast<const float4*>(&xs[k0 + 1][j]);
                b1 = *reinterpret_cast<const float4*>(&xs[k0 + 1][j + 4]);
            }
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                acc0[r] = dot8(wv4[r][q], a0, a1, acc0[r]);
                acc1[r] = dot8(wv4[r][q], b0, b1, acc1[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int row = (rg * WV + wv) * RW + r;
            const float v0 = wsum(acc0[r]), v1 = wsum(acc1[r]);
            if (act[r] && lane == 0) m.part[((size_t) k0 * 2 * HC + slice) * PR + row] = v0;
            if (act[r] && lane == 1 && two) m.part[((size_t) (k0 + 1) * 2 * HC + slice) * PR + row] = v1;
        }
    }
}

template <int T>
__device__ __forceinline__ void inject_slice(const RdnaGr& m, int slice) {
    const int t = threadIdx.x, lane = t & 31, wv = t >> 5, c = slice >> 1, h = slice & 1, e0 = c * N + h * H;
    for (int it = wv; it < (PR - LR) * T; it += WV) {
        const int row = it / T, k = it - row * T;
        const FusedGrArgs& a = m.a[k];
        const uint4* w4 = reinterpret_cast<const uint4*>(a.w_inject + (size_t) row * D + e0);
        const float gw = a.apply ? 2.0f * sigm(a.inj_prev[c] / (float) HC) : 0.0f;
        uint4 wb[HQ];
#pragma unroll
        for (int q = 0; q < HQ; ++q) wb[q] = ntload(w4 + lane + 32 * q);
        float acc = 0.0f;
#pragma unroll
        for (int q = 0; q < HQ; ++q) {
            const int i = (lane + 32 * q) * 8;
            float4 v[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i + 4 * u);
                if (a.apply) {
                    const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i + 4 * u);
                    r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
                }
                const float4 g = *reinterpret_cast<const float4*>(a.w_norm + e0 + i + 4 * u);
                v[u] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
            }
            acc = dot8(wb[q], v[0], v[1], acc);
        }
        const float s = wsum(acc);
        if (lane == 0) m.part[((size_t) k * 2 * HC + slice) * PR + LR + row] = s;
    }
}

// Per-32 FP32-scale int8 weights (0.2.18 FusedGrArgs: w_down holds int8 codes, s_down holds float scales).
template <int T, int RW>
__device__ __forceinline__ void down_a8_body(const RdnaGr& m, const int bx, const int by) {
    __shared__ __align__(16) uint32_t xq[T][H / 4];
    __shared__ float xd[T][H / 32];
    __shared__ float sq[WV][T];
    const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
    const int rg = bx, slice = by, c = slice >> 1, h = slice & 1;
    if (rg == LR / (RW * WV)) {
        if (m.a[0].w_inject != nullptr) inject_slice<T>(m, slice);
        return;
    }
    const int8_t* qd = reinterpret_cast<const int8_t*>(m.a[0].w_down);
    const float* sd = m.a[0].s_down;
    uint2 wq[RW][HQ];
    float wd[RW][HQ];
    bool act[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int row = (rg * WV + wv) * RW + r;
        act[r] = row < LR;
        if (!act[r]) continue;
        const uint2* w2 = reinterpret_cast<const uint2*>(qd + (size_t) row * D + c * N + h * H);
        const float* s2 = sd + (size_t) row * (D / 32) + (c * N + h * H) / 32;
#pragma unroll
        for (int q = 0; q < HQ; ++q) {
            wq[r][q] = ntload2(w2 + lane + 32 * q);
            wd[r][q] = s2[(lane + 32 * q) >> 2];
        }
    }
    const int e0 = c * N + h * H;
#pragma unroll 4
    for (int it = t; it < T * (H / 4); it += TH) {
        const int k = it / (H / 4), i = (it - k * (H / 4)) * 4;
        const FusedGrArgs& a = m.a[k];
        float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
        if (a.apply) {
            const float gw = 2.0f * sigm(a.inj_prev[c] / (float) HC);
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + e0 + i);
        const float4 v = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        quant4(v, &xq[k][i >> 2], &xd[k][i >> 5], (i & 31) == 0);
    }
    if (rg == 0) {
#pragma unroll 1
        for (int k = 0; k < T; ++k) {
            const FusedGrArgs& a = m.a[k];
            const float gw = a.apply ? 2.0f * sigm(a.inj_prev[c] / (float) HC) : 0.0f;
            float s = 0.0f;
            for (int i = t * 4; i < H; i += TH * 4) {
                float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
                if (a.apply) {
                    const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
                    r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
                }
                s = fmaf(r.w, r.w, fmaf(r.z, r.z, fmaf(r.y, r.y, fmaf(r.x, r.x, s))));
            }
            const float v = wsum(s);
            if (lane == 0) sq[wv][k] = v;
        }
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WV; ++w) s += sq[w][t];
        m.ss[t * 2 * HC + slice] = s;
    }
#pragma unroll 1
    for (int k = 0; k < T; ++k) {
        float acc[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) acc[r] = 0.0f;
#pragma unroll
        for (int q = 0; q < HQ; ++q) {
            const int ch = lane + 32 * q;
            const uint2 av = *reinterpret_cast<const uint2*>(&xq[k][ch * 2]);
            const float ad = xd[k][ch >> 2];
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                acc[r] = fmaf(wd[r][q] * ad, (float) dot8i(wq[r][q], av), acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int row = (rg * WV + wv) * RW + r;
            const float v = wsum(acc[r]);
            if (act[r] && lane == 0) m.part[((size_t) k * 2 * HC + slice) * PR + row] = v;
        }
    }
}

template <int T, int RW>
__global__ void __launch_bounds__(TH) rdna_gr_down_a8(RdnaGr m) {
    down_a8_body<T, RW>(m, blockIdx.x, blockIdx.y);
}

struct RdnaQ81 {
    half2 ds;
    int8_t qs[32];
};

template <int T>
__device__ __forceinline__ void q8_tail(const RdnaGr& m, int d0) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* c = m.a[0].q8_cnt + d0 / 32;
        last = atomicAdd(c, 1u) == 1u;
        if (last) atomicExch(c, 0u);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (wave >= T) return;
    const int c0 = (d0 / 32) * 32;
    const float xi = *reinterpret_cast<const volatile float*>(m.a[wave].mixed + c0 + lane);
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    RdnaQ81* y = reinterpret_cast<RdnaQ81*>(m.a[wave].q8_mixed) + c0 / 32;
    y->qs[lane] = q;
    if (lane == 0) y->ds = q8_1_ds(d, sum);
}

template <int T, bool Q8 = false, bool A8 = false>
__device__ __forceinline__ void up_body(const RdnaGr& m, const int bx) {
    __shared__ __align__(16) uint32_t loq[A8 ? T : 1][LR / 4];
    __shared__ float lod[A8 ? T : 1][LR / 32];
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rs[T][HC];
    __shared__ float g[T][HC][UC];
    const int t = threadIdx.x;
    const int rr = t >> 2, q = t & 3;
    const int c = rr / UC, dd = rr - c * UC, d = bx * UC + dd, i = c * N + d;
    uint4 w[10];
    uint2 w8[10];
    float d8[10];
    if constexpr (Q8) {
        const int8_t* qu = reinterpret_cast<const int8_t*>(m.a[0].w_up);
        const float* su = m.a[0].s_up;
        const uint2* w2 = reinterpret_cast<const uint2*>(qu + (size_t) i * LR);
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            w8[u] = ntload2(w2 + q + 4 * u);
            d8[u] = su[(size_t) i * (LR / 32) + ((q + 4 * u) >> 2)];
        }
    } else {
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
#pragma unroll
        for (int u = 0; u < 10; ++u) w[u] = ntload(w4 + q + 4 * u);
    }
    float rv[T], wn = m.a[0].w_norm[i];
#pragma unroll
    for (int k = 0; k < T; ++k) rv[k] = m.a[k].R[i];
    if (t < T * HC) {
        const int k = t / HC, cs = t - k * HC;
        const float s = m.ss[k * 2 * HC + 2 * cs] + m.ss[k * 2 * HC + 2 * cs + 1];
        rs[k][cs] = rsqrtf(s / (float) N + m.a[k].eps);
        if (bx == 0) m.a[k].rs[cs] = rs[k][cs];
    }
    __syncthreads();
    for (int idx = t; idx < T * PR; idx += TH) {
        const int k = idx / PR, j = idx - k * PR;
        const float* p = m.part + (size_t) k * 2 * HC * PR + j;
        float s = 0.0f;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) s = fmaf(rs[k][cc], p[(2 * cc) * PR] + p[(2 * cc + 1) * PR], s);
        if (j < LR) {
            const float x = s / (float) HC;
            lo[k][j] = x / (1.0f + __expf(-x));
            if (bx == 0) m.a[k].lo[j] = lo[k][j];
        } else if (bx == 0 && m.a[0].w_inject != nullptr) {
            m.a[k].inject_out[j - LR] = s;
        }
    }
    __syncthreads();
    if constexpr (A8) {
        for (int idx = t; idx < T * (LR / 4); idx += TH) {
            const int k = idx / (LR / 4), f = idx - k * (LR / 4);
            const float4 v = *reinterpret_cast<const float4*>(&lo[k][4 * f]);
            quant4(v, &loq[k][f], &lod[k][f >> 3], (f & 7) == 0);
        }
        __syncthreads();
    }
    float acc[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
        float a = 0.0f;
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            const int j = (q + 4 * u) * 8;
            if constexpr (A8) a = fmaf(d8[u] * lod[k][j >> 5], (float) dot8i(w8[u], *reinterpret_cast<const uint2*>(&loq[k][j >> 2])), a);
            else if constexpr (Q8) a = dot8q(w8[u], d8[u], *reinterpret_cast<const float4*>(&lo[k][j]), *reinterpret_cast<const float4*>(&lo[k][j + 4]), a);
            else a = dot8(w[u], *reinterpret_cast<const float4*>(&lo[k][j]), *reinterpret_cast<const float4*>(&lo[k][j + 4]), a);
        }
        a += __shfl_xor_sync(0xffffffffu, a, 1);
        a += __shfl_xor_sync(0xffffffffu, a, 2);
        acc[k] = a;
    }
    if (q == 0) {
#pragma unroll
        for (int k = 0; k < T; ++k) {
            const FusedGrArgs& a = m.a[k];
            float r = rv[k];
            if (a.apply) {
                r = fmaf(a.bo_prev[d], 2.0f * sigm(a.inj_prev[c] / (float) HC), r);
                a.R_out[i] = r;
            }
            g[k][c][dd] = r * wn * rs[k][c] * sigm(acc[k]);
        }
    }
    __syncthreads();
    if (t < T * UC) {
        const int k = t / UC, col = t - k * UC;
        float s = 0.0f;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) s += g[k][cc][col];
        m.a[k].mixed[bx * UC + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) q8_tail<T>(m, bx * UC);
}

template <int T, bool Q8 = false, bool A8 = false>
__global__ void __launch_bounds__(TH) rdna_gr_up(RdnaGr m) {
    up_body<T, Q8, A8>(m, blockIdx.x);
}

__device__ __forceinline__ bool grid_barrier_bounded(unsigned* bar, unsigned nblocks, uint64_t t0) {
    __shared__ int ok;
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned* gen = bar + 1;
        const unsigned g = *gen;
        __threadfence();
        int done = 1;
        if (atomicAdd(bar, 1u) == nblocks - 1) {
            atomicExch(bar, 0u);
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            uint32_t iter = 0;
            while (*gen == g) {
                __builtin_amdgcn_s_sleep(1);
                if (rdna3_spin_expired(t0, iter, nullptr, 1u)) {
                    done = 0;
                    break;
                }
            }
        }
        ok = done;
    }
    __syncthreads();
    __threadfence();
    return ok != 0;
}

template <int T>
__global__ void __launch_bounds__(TH) rdna_gr_fused(RdnaGr m) {
    constexpr int NBX = LR / (2 * WV) + 1;
    const uint64_t t0 = (uint64_t) wall_clock64();
    const int b = (int) blockIdx.x;
    down_a8_body<T, 2>(m, b % NBX, b / NBX);
    if (!grid_barrier_bounded(m.bar, gridDim.x, t0)) return;
    if (b < UB) up_body<T, true, true>(m, b);
}

template <int T>
void launch_gr(const RdnaGr& m, cudaStream_t st, unsigned long long* sb, int si, int dev) {
    if (sb) gpu_stamp(sb, si, st);
    if (m.a[0].s_down != nullptr && m.a[0].s_up != nullptr) {
        static const bool fused_req = [] {
            const char* v = std::getenv("STRATA_GR_FUSED");
            return v != nullptr && v[0] == '1';
        }();
        constexpr int kFusedBlocks = (LR / (2 * WV) + 1) * 2 * HC;  // 168 blocks
        if (fused_req && m.bar != nullptr && sb == nullptr && dev >= 0 && dev < 64) {
            if (g_gr_can_fuse[dev] == 0) {
                int cus = 0, per = 0;
                cudaDeviceGetAttribute(&cus, cudaDevAttrMultiProcessorCount, dev);
                (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, rdna_gr_fused<8>, TH, 0);
#if defined(_WIN32)
                constexpr bool kWin = true;
#else
                constexpr bool kWin = false;
#endif
                const PersistentGridPlan plan = compute_safe_persistent_grid(cus, per > 0 ? per : 1, kFusedBlocks, kWin);
                g_gr_can_fuse[dev] = plan.can_fuse_fixed ? 2 : 1;
            }
            if (g_gr_can_fuse[dev] == 2) {
                rdna_gr_fused<T><<<kFusedBlocks, TH, 0, st>>>(m);
                return;
            }
        }
        rdna_gr_down_a8<T, 2><<<dim3(LR / (2 * WV) + 1, 2 * HC), TH, 0, st>>>(m);
        if (sb) gpu_stamp(sb, si + 1, st);
        rdna_gr_up<T, true, true><<<UB, TH, 0, st>>>(m);
        return;
    }
    rdna_gr_down<T, 2><<<dim3((PR + 2 * WV - 1) / (2 * WV), 2 * HC), TH, 0, st>>>(m);
    if (sb) gpu_stamp(sb, si + 1, st);
    rdna_gr_up<T><<<UB, TH, 0, st>>>(m);
}

}  // namespace

bool rdna_gr_enabled() {
    static const int env = [] {
        if (const char* v = std::getenv("STRATA_GR_RDNA")) return v[0] == '1' ? 1 : 0;
        if (const char* f = std::getenv("STRATA_HIP_FAST_DEC")) return f[0] == '1' ? 2 : 0;
        return 0;
    }();
    if (env == 0) return false;
    if (env == 1) return true;
    static int arch_ok[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (arch_ok[dev] == 0) {
        cudaDeviceProp p{};
        arch_ok[dev] = (cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                        (gfx_arch_is_gfx11_wmma(p.gcnArchName) || gfx_arch_is_gfx12(p.gcnArchName))) ? 2 : 1;
    }
    return arch_ok[dev] == 2;
}

void rdna_gr_read_multi(const FusedGrArgs* a, int n_tok, float* scratch, void* stream,
                        unsigned long long* sb, int si) {
    RdnaGr m;
    for (int t = 0; t < n_tok; ++t) m.a[t] = a[t];
    m.part = scratch;
    m.ss = scratch + (size_t) n_tok * 2 * HC * PR;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64) {
        if (g_gr_bar[dev] == nullptr && std::getenv("STRATA_GR_FUSED") != nullptr) {
            if (cudaMalloc(&g_gr_bar[dev], 64) == cudaSuccess) cudaMemset(g_gr_bar[dev], 0, 64);
        }
        m.bar = g_gr_bar[dev];
    }
    cudaStream_t st = (cudaStream_t) stream;
    switch (n_tok) {
        case 1: launch_gr<1>(m, st, sb, si, dev); break;
        case 2: launch_gr<2>(m, st, sb, si, dev); break;
        case 3: launch_gr<3>(m, st, sb, si, dev); break;
        case 4: launch_gr<4>(m, st, sb, si, dev); break;
        case 5: launch_gr<5>(m, st, sb, si, dev); break;
        case 6: launch_gr<6>(m, st, sb, si, dev); break;
        case 7: launch_gr<7>(m, st, sb, si, dev); break;
        default: launch_gr<8>(m, st, sb, si, dev); break;
    }
}

#else

bool rdna_gr_enabled() { return false; }
void rdna_gr_read_multi(const FusedGrArgs*, int, float*, void*, unsigned long long*, int) {}

#endif

}  // namespace strata::kernels
