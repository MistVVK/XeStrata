// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// src/kernels/cuda/megakernel_gdn.cu - Per-layer persistent GDN dense chain for RDNA3/4 (STRATA_MEGAKERNEL=1).
// Ported from MIT Strata-rdna3 (nicejunjie/Strata-rdna3) with CU-count-safe persistent grid sizing (12..96 CUs)
// and WDDM-safe 150 ms wall_clock64() spin guards (rdna3_wddm_guard.hpp).
#include "strata/kernels/megakernel.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/kernels/rdna3_wddm_guard.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

#ifndef MK_TH
#define MK_TH 512
#endif
constexpr int TH = MK_TH;
constexpr int S = 128;
constexpr int RG = 4;
constexpr int RPG = S / RG;
constexpr int WARP = 32;
constexpr int WARPS = 4;
constexpr int Q8K = 32;
constexpr int QUANT_THREADS = 256;

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q80Block {
    half d;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36 && sizeof(Q80Block) == 34);

__device__ __forceinline__ float warp_sum(float x) {
#if defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_HIP_GFX906)
#define MK_DPP(val, c) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(val), c, 0xf, 0xf, false))
    x += MK_DPP(x, 0x161);
    x += MK_DPP(x, 0x162);
    x += MK_DPP(x, 0x164);
    x += MK_DPP(x, 0x168);
    x += __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(x), 0x76543210, 0xfedcba98, false, false));
#undef MK_DPP
    return x;
#else
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset, WARP);
    return x;
#endif
}
__device__ __forceinline__ float warp_max(float x) {
#if defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_HIP_GFX906)
#define MK_DPP(val, c) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(val), c, 0xf, 0xf, false))
    x = fmaxf(x, MK_DPP(x, 0x161));
    x = fmaxf(x, MK_DPP(x, 0x162));
    x = fmaxf(x, MK_DPP(x, 0x164));
    x = fmaxf(x, MK_DPP(x, 0x168));
    x = fmaxf(x, __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(x), 0x76543210, 0xfedcba98, false, false)));
#undef MK_DPP
    return x;
#else
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, offset, WARP));
    return x;
#endif
}
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__device__ __forceinline__ void quant_body(const float* __restrict__ x, Q81Block* __restrict__ y, int n_in, int vb, int vtid) {
    const int i = vb * QUANT_THREADS + vtid;
    if (i >= n_in) return;
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

struct Q80Traits {
    static constexpr int DIV = 32;
    static constexpr int T = 4;
    static constexpr int KBY = 1;
    static constexpr int BPI = 2 * WARPS * WARP / 8;
    __device__ static int kqs(int tid) { return 2 * (tid % 4); }
    struct W {
        int v[2];
        float d0;
    };
    __device__ static W load(const Q80Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = load_int_b2(w->qs, iqs + i);
        r.d0 = __half2float(w->d);
        return r;
    }
};

template <int NCOLS, bool MAP = false>
__device__ __forceinline__ void gemv_q80_body(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x,
                                              float* __restrict__ y, int n_in, int n_out, int vb, int vtid, int xs_zero = 0,
                                              const MkMat* __restrict__ mm = nullptr) {
    using F = Q80Traits;
    const int lane = vtid & 31;
    const int row = vb * 8 + vtid / 32;
    if (row >= n_out) return;
    const int blocks_per_row = n_in / F::DIV;
    int pitch = blocks_per_row, prow = row, cshift = 30, cgap = 0, cadd = 0;
    if (MAP && mm != nullptr && mm->pitch > 0) {
        pitch = mm->pitch;
        int add = mm->radd[0];
#pragma unroll
        for (int k = 1; k < 8; ++k)
            if (k < mm->nr && row >= mm->rlo[k]) add = mm->radd[k];
        prow = row + add;
        cadd = mm->cadd;
        if (mm->cw > 0) {
            cshift = __ffs(mm->cw) - 1;
            cgap = mm->cgap;
        }
    }
    const Q80Block* __restrict__ wrow = w + std::size_t(prow) * pitch;
    const int x_stride = xs_zero ? 0 : n_in / Q8K;
    float tmp[NCOLS][WARPS] = {};
#pragma unroll
    for (int g = 0; g < WARPS; ++g) {
        const int tid = lane + g * WARP;
        const int kqs = F::kqs(tid);
        for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
            const F::W wv = F::load(MAP ? wrow + kbx + cadd + (kbx >> cshift) * cgap : w + std::size_t(row) * blocks_per_row + kbx, kqs);
            int u[NCOLS][2];
            float d1[NCOLS];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * x_stride + kbx * F::KBY;
#pragma unroll
                for (int i = 0; i < 2; ++i) u[j][i] = reinterpret_cast<const int*>(xb->qs)[kqs + i];
                d1[j] = __low2float(xb->ds);
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 2; ++i) sumi = STRATA_DP4A(wv.v[i], u[j][i], sumi);
                tmp[j][g] += wv.d0 * d1[j] * float(sumi);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = tmp[j][0];
#pragma unroll
        for (int g = 1; g < WARPS; ++g) v += tmp[j][g];
        v = warp_sum(v);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}

__device__ __forceinline__ void conv_body(float* part, int bx, int by, const float* hist, const float* __restrict__ qkv,
                                          const float* __restrict__ w, float* __restrict__ h, int C, int qk_heads, float eps,
                                          int vtid) {
    const int t = by;
    const int c = bx * S + vtid;
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    float sq = y * y;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((vtid & 31) == 0) part[vtid >> 5] = sq;
    __syncthreads();
    if (bx < qk_heads) {
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

template <int MAX_T>
__device__ __forceinline__ void ab_body(int bx, const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                        const uint16_t* __restrict__ wb, const float* __restrict__ dt,
                                        const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                        int n, int h_v, int vtid) {
    const int row = bx * 2 + (vtid >> 5), lane = vtid & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    float acc[MAX_T];
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
        const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
        const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
        const float w4f = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
        const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
#pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            const float* xt = x + (size_t) t * n;
            const float4 xa = __ldg(reinterpret_cast<const float4*>(xt + j * 8));
            const float4 xb = __ldg(reinterpret_cast<const float4*>(xt + j * 8 + 4));
            float a = acc[t];
            a = fmaf(w0, xa.x, a); a = fmaf(w1, xa.y, a);
            a = fmaf(w2, xa.z, a); a = fmaf(w3, xa.w, a);
            a = fmaf(w4f, xb.x, a); a = fmaf(w5, xb.y, a);
            a = fmaf(w6, xb.z, a); a = fmaf(w7, xb.w, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        float a = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t) t * h_v + r] = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

struct GdnQ81 {
    half2 ds;
    int8_t qs[32];
};
__device__ __forceinline__ void gdn_q8_1_store(GdnQ81* __restrict__ xq, size_t idx, float xi) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    xq[idx / 32].qs[idx % 32] = q;
    if (idx % 32 == 0) xq[idx / 32].ds = q8_1_ds(d, sum);
}

struct StepShared {
    float sk[2][S];
    float sq[2][S];
    float red_kv[RG][S];
    float red_o[RG][S];
    float wsum[S / 32];
};

__device__ __forceinline__ void step_body(StepShared& sh, int head, const float* __restrict__ state,
                                          const float* __restrict__ hbuf, int C, const float* __restrict__ gate,
                                          const float* __restrict__ beta, const float* __restrict__ z,
                                          const float* __restrict__ gamma, float eps, float* __restrict__ y, int h_k, int h_v,
                                          int T, GdnQ81* __restrict__ xq, int tid) {
    constexpr int RGT = TH / S, RPT = RG / RGT;
    static_assert(RG % RGT == 0, "row groups must tile the block");
    const int col = tid % S;
    const int rg0 = tid / S;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int value_dim = S * h_v;
    const int n = T;
    if (n > 0) {
        if (rg0 == 0) sh.sk[0][col] = hbuf[qk + qh * S + col];
        else if (rg0 == 1) sh.sq[0][col] = hbuf[qh * S + col];
    }
    const float gam = (rg0 == 0) ? gamma[col] : 0.0f;
    float s[RPT][RPG];
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int j = 0; j < RPT; ++j) {
        const int rg = rg0 + j * RGT;
        const float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[j][r] = __ldg(&base[r * row_stride]);
    }
    for (int t = 0; t < n; ++t) {
        const int cur = t & 1, nxt = (t + 1) & 1;
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();
        if (t + 1 < n) {
            const float* ht_next = ht + C;
            if (rg0 == 0) sh.sk[nxt][col] = ht_next[qk + qh * S + col];
            else if (rg0 == 1) sh.sq[nxt][col] = ht_next[qh * S + col];
        }
        const float g = __expf(gate[(size_t) t * h_v + head]);
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            const int rg = rg0 + j * RGT;
            float kv = 0.0f;
#pragma unroll
            for (int r = 0; r < RPG; ++r) kv = fmaf(s[j][r], sh.sk[cur][rg * RPG + r], kv);
            sh.red_kv[rg][col] = kv;
        }
        __syncthreads();
        const float kv_col = sh.red_kv[0][col] + sh.red_kv[1][col] + sh.red_kv[2][col] + sh.red_kv[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            const int rg = rg0 + j * RGT;
            float o = 0.0f;
#pragma unroll
            for (int r = 0; r < RPG; ++r) {
                s[j][r] = fmaf(g, s[j][r], sh.sk[cur][rg * RPG + r] * delta);
                o = fmaf(s[j][r], sh.sq[cur][rg * RPG + r], o);
            }
            sh.red_o[rg][col] = o;
        }
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg0 == 0) {
            oc = (sh.red_o[0][col] + sh.red_o[1][col] + sh.red_o[2][col] + sh.red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if (rg0 == 0 && (col & 31) == 0) sh.wsum[col >> 5] = sq_part;
        __syncthreads();
        if (rg0 == 0) {
            const float ss = sh.wsum[0] + sh.wsum[1] + sh.wsum[2] + sh.wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            const float yv = oc * scale * gam * (1.0f / (1.0f + __expf(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
            gdn_q8_1_store(xq, (size_t) t * value_dim + head * S + col, yv);
        }
    }
}

#if defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_HIP_GFX906)

__device__ __forceinline__ bool wave0() { return threadIdx.x < WARP; }
__device__ __forceinline__ unsigned one0() { return threadIdx.x == 0 ? 1u : 0u; }

__device__ __forceinline__ int grab(unsigned* next, int* slot, unsigned k) {
    if (wave0()) *slot = (int) __builtin_amdgcn_readfirstlane(atomicAdd(next, threadIdx.x == 0 ? k : 0u));
    __syncthreads();
    const int it = __builtin_amdgcn_readfirstlane(*slot);
    __syncthreads();
    return it;
}

__device__ __forceinline__ void phase_wait(unsigned* done, unsigned items, unsigned* diag, unsigned phase,
                                           unsigned* stall, uint64_t kernel_t0) {
    if (wave0()) {
        uint32_t iter = 0;
        while (__builtin_amdgcn_readfirstlane(__hip_atomic_load(done, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) < items) {
            __builtin_amdgcn_s_sleep(1);
            if ((iter & 4095u) == 4095u && stall != nullptr &&
                __builtin_amdgcn_readfirstlane(__hip_atomic_load(stall, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)) != 0)
                break;
            if (rdna3_spin_expired(kernel_t0, iter, stall, phase + 1)) {
                atomicMax(diag, phase + 1);
                break;
            }
        }
    }
    __syncthreads();
    __threadfence();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

__device__ __forceinline__ void items_done(unsigned* done, unsigned n, bool all_fence = true) {
    __syncthreads();
    if (all_fence) __threadfence();
    if (wave0()) __hip_atomic_fetch_add(done, threadIdx.x == 0 ? n : 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

template <class Body>
__device__ __forceinline__ void run_phase(const MkGdnArgs& a, int k, unsigned n, unsigned chunk, int* slot, bool sync_between,
                                          Body body) {
    unsigned* done = a.counters + k;
    unsigned* next = a.counters + 8 + k;
    if (a.diag & 1) chunk = 1;
    if (a.chunk > 0) chunk = (unsigned) a.chunk;
    if (!(a.diag & (256 | 1)) && a.chunk == 0) {
        unsigned mine = 0;
        for (int it = (int) blockIdx.x; it < (int) n; it += (int) gridDim.x) {
            body(it);
            if (sync_between) __syncthreads();
            ++mine;
        }
        items_done(done, mine, !(a.diag & 128));
        return;
    }
    for (;;) {
        const int it0 = grab(next, slot, chunk);
        if (it0 >= (int) n) break;
        const int e = min(it0 + (int) chunk, (int) n);
        for (int it = it0; it < e; ++it) {
            body(it);
            if (sync_between) __syncthreads();
        }
        items_done(done, (unsigned) (e - it0), !(a.diag & 128));
    }
}

__device__ __forceinline__ void mk_exit(const MkGdnArgs& a) {
    __syncthreads();
    __threadfence();
    if (wave0()) {
        const unsigned out = __builtin_amdgcn_readfirstlane(atomicAdd(a.counters + 14, one0()));
        if (out + 1 == gridDim.x) {
            __threadfence();
            if (threadIdx.x < 15) __hip_atomic_store(a.counters + threadIdx.x, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

template <int NCOLS, bool MAP>
__global__ void __launch_bounds__(TH) mk_gdn_kernel(MkGdnArgs a) {
    __shared__ int slot;
    __shared__ float conv_part[TH / S][S / 32];
    __shared__ StepShared step_sh;
    const uint64_t kernel_t0 = (uint64_t) wall_clock64();
    const int tid = (int) threadIdx.x;
    const int N = a.n_embd, C = a.conv_channels, ZV = a.value_dim, HK = a.h_k, HV = a.h_v;
    const int T = NCOLS;
    unsigned* done = a.counters;
    const Q81Block* xq = (const Q81Block*) a.xq;
    const Q81Block* yq = (const Q81Block*) a.yq;
    if (wave0()) atomicAdd(a.counters + 6, one0());
    if (a.phases <= 0) { mk_exit(a); return; }

    // Phase 0: q8_1 input quantization
    const unsigned n0 = (unsigned) ((T * N + (TH / QUANT_THREADS) * QUANT_THREADS - 1) / ((TH / QUANT_THREADS) * QUANT_THREADS));
    if (!(a.diag & 32)) {
        run_phase(a, 0, n0, 1, &slot, false, [&](int it) {
            quant_body(a.x, (Q81Block*) a.xq, T * N, it * (TH / QUANT_THREADS) + tid / QUANT_THREADS, tid % QUANT_THREADS);
        });
    }
    if (blockIdx.x == 0 && wave0()) atomicMax(a.counters + 5, 1u);
    if (a.phases <= 1) { mk_exit(a); return; }

    // Phase 1: qkv + z Q8_0 projections and alpha/beta BF16 gates
    const unsigned n_qkv = (unsigned) ((C / 8 + TH / 256 - 1) / (TH / 256));
    const unsigned n_z = (unsigned) ((ZV / 8 + TH / 256 - 1) / (TH / 256));
    const unsigned n_ab = (unsigned) (((2 * HV + 1) / 2 + TH / 64 - 1) / (TH / 64));
    const unsigned n1 = n_qkv + n_z + n_ab;
    if (!(a.diag & 32)) phase_wait(done + 0, n0, a.counters + 15, 0, a.stall, kernel_t0);
    run_phase(a, 1, (a.diag & 8) ? n_qkv : n1, 8, &slot, (a.diag & 4) != 0, [&](int it) {
        if (it < (int) n_qkv) {
            gemv_q80_body<NCOLS, MAP>((const Q80Block*) a.w_qkv, xq, a.qkv, N, C, it * (TH / 256) + tid / 256, tid % 256,
                                      a.diag & 16, &a.m_qkv);
        } else if (it < (int) (n_qkv + n_z)) {
            gemv_q80_body<NCOLS, MAP>((const Q80Block*) a.w_z, xq, a.z, N, ZV, (it - (int) n_qkv) * (TH / 256) + tid / 256,
                                      tid % 256, 0, &a.m_z);
        } else {
            ab_body<NCOLS>((it - (int) (n_qkv + n_z)) * (TH / 64) + tid / 64, a.x, a.w_alpha, a.w_beta, a.dt, a.ssm_a,
                           a.gate, a.beta, N, HV, tid % 64);
        }
    });
    if (blockIdx.x == 0 && wave0()) atomicMax(a.counters + 5, 2u);
    if (a.phases <= 2) { mk_exit(a); return; }

    // Phase 2: conv1d + L2 norm
    const unsigned n_conv_vb = (unsigned) ((C / S) * T);
    const unsigned n2 = (n_conv_vb + TH / S - 1) / (TH / S);
    phase_wait(done + 1, (a.diag & 8) ? n_qkv : n1, a.counters + 15, 1, a.stall, kernel_t0);
    run_phase(a, 2, n2, 2, &slot, true, [&](int it) {
        const int vb = it * (TH / S) + tid / S;
        if (vb < (int) n_conv_vb)
            conv_body(conv_part[tid / S], vb % (C / S), vb / (C / S), a.hist, a.qkv, a.w_conv, a.h, C, 2 * HK, a.eps, tid % S);
        else
            __syncthreads();
    });
    if (blockIdx.x == 0 && wave0()) atomicMax(a.counters + 5, 3u);
    if (a.phases <= 3) { mk_exit(a); return; }

    // Phase 3: GDN recurrence + output RMSNorm + q8_1 store
    phase_wait(done + 2, n2, a.counters + 15, 2, a.stall, kernel_t0);
    run_phase(a, 3, (unsigned) HV, 1, &slot, true, [&](int it) {
        step_body(step_sh, it, a.state, a.h, C, a.gate, a.beta, a.z, a.gamma, a.eps, a.y, HK, HV, T, (GdnQ81*) a.yq, tid);
    });
    if (blockIdx.x == 0 && wave0()) atomicMax(a.counters + 5, 4u);
    if (a.phases <= 4) { mk_exit(a); return; }

    // Phase 4: Q8_0 out projection
    const unsigned n4 = (unsigned) ((N / 8 + TH / 256 - 1) / (TH / 256));
    phase_wait(done + 3, (unsigned) HV, a.counters + 15, 3, a.stall, kernel_t0);
    run_phase(a, 4, n4, 4, &slot, (a.diag & 4) != 0, [&](int it) {
        gemv_q80_body<NCOLS, MAP>((const Q80Block*) a.w_out, yq, a.bo, ZV, N, it * (TH / 256) + tid / 256, tid % 256, 0,
                                  &a.m_out);
    });
    mk_exit(a);
}

int g_grid[64][2][9] = {};

template <int NCOLS, bool MAP>
void launch_m(const MkGdnArgs& a, cudaStream_t s) {
    int dev = 0;
    cudaGetDevice(&dev);
    const int d = (dev >= 0 && dev < 64) ? dev : 0;
    int& g_grid_n = g_grid[d][MAP ? 1 : 0][NCOLS];
    if (g_grid_n == 0) {
        int cus = 0, per = 0;
        cudaDeviceGetAttribute(&cus, cudaDevAttrMultiProcessorCount, dev);
        (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, mk_gdn_kernel<NCOLS, MAP>, TH, 0);
#if defined(_WIN32)
        constexpr bool kWin = true;
#else
        constexpr bool kWin = false;
#endif
        const PersistentGridPlan plan = compute_safe_persistent_grid(cus, per > 0 ? per : 1, 0, kWin);
        g_grid_n = plan.safe_blocks > 0 ? plan.safe_blocks : 8;
    }
    mk_gdn_kernel<NCOLS, MAP><<<a.grid > 0 ? a.grid : g_grid_n, TH, 0, s>>>(a);
}

template <int NCOLS>
void launch(const MkGdnArgs& a, cudaStream_t s) {
    if (a.m_qkv.pitch > 0 || a.m_z.pitch > 0 || a.m_out.pitch > 0) launch_m<NCOLS, true>(a, s);
    else launch_m<NCOLS, false>(a, s);
}

#endif  // __HIP_PLATFORM_AMD__ && !STRATA_HIP_GFX906

template <int NCOLS>
__global__ void __launch_bounds__(256) q80_mapped_kernel(const Q80Block* w, MkMat m, const Q81Block* x, float* y,
                                                         int n_in, int n_out) {
    gemv_q80_body<NCOLS, true>(w, x, y, n_in, n_out, (int) blockIdx.x, (int) threadIdx.x, 0, &m);
}

}  // namespace

void q80_mmvq_mapped(const void* w, const MkMat& m, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    const unsigned g = (unsigned) ((n_out + 7) / 8);
    const Q80Block* W = (const Q80Block*) w;
    const Q81Block* X = (const Q81Block*) xq;
    switch (ncols) {
        case 1: q80_mapped_kernel<1><<<g, 256, 0, s>>>(W, m, X, y, n_in, n_out); break;
        case 2: q80_mapped_kernel<2><<<g, 256, 0, s>>>(W, m, X, y, n_in, n_out); break;
        case 3: q80_mapped_kernel<3><<<g, 256, 0, s>>>(W, m, X, y, n_in, n_out); break;
        case 4: q80_mapped_kernel<4><<<g, 256, 0, s>>>(W, m, X, y, n_in, n_out); break;
        case 5: q80_mapped_kernel<5><<<g, 256, 0, s>>>(W, m, X, y, n_in, n_out); break;
        case 6: q80_mapped_kernel<6><<<g, 256, 0, s>>>(W, m, X, y, n_in, n_out); break;
        case 7: q80_mapped_kernel<7><<<g, 256, 0, s>>>(W, m, X, y, n_in, n_out); break;
        case 8: q80_mapped_kernel<8><<<g, 256, 0, s>>>(W, m, X, y, n_in, n_out); break;
        default: std::fprintf(stderr, "q80_mmvq_mapped: %d columns\n", ncols); std::exit(1);
    }
}

unsigned* mk_counters_alloc() {
    unsigned* p = nullptr;
    if (cudaMalloc((void**) &p, 64) != cudaSuccess) return nullptr;
    cudaMemset(p, 0, 64);
    return p;
}

int mk_gdn_grid(int n_tok) {
#if defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_HIP_GFX906)
    int dev = 0;
    cudaGetDevice(&dev);
    const int d = (dev >= 0 && dev < 64) ? dev : 0;
    return g_grid[d][0][n_tok < 1 ? 1 : (n_tok > 8 ? 8 : n_tok)];
#else
    (void) n_tok;
    return 0;
#endif
}

void mk_gdn_dense(const MkGdnArgs& a, int n_tok, void* stream) {
#if defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_HIP_GFX906)
    if (n_tok < 1 || n_tok > 8 || a.counters == nullptr) {
        std::fprintf(stderr, "mk_gdn_dense: invalid arguments\n");
        std::exit(1);
    }
    cudaStream_t s = (cudaStream_t) stream;
    switch (n_tok) {
        case 1: launch<1>(a, s); break;
        case 2: launch<2>(a, s); break;
        case 3: launch<3>(a, s); break;
        case 4: launch<4>(a, s); break;
        case 5: launch<5>(a, s); break;
        case 6: launch<6>(a, s); break;
        case 7: launch<7>(a, s); break;
        default: launch<8>(a, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mk_gdn_dense: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
#else
    (void) a; (void) n_tok; (void) stream;
    std::fprintf(stderr, "mk_gdn_dense: RDNA3/4 HIP build required\n");
    std::exit(1);
#endif
}

}  // namespace strata::kernels
