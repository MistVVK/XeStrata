// src/image/dit/qi21_kernels.cu - see qi21_kernels.hpp.  The kernels from tools/image/sdcpp_native/qi21-native.cu
// (k_res_ln_mod, k_qkv_post_w, k_swiglu, k_attn_causal, k_attn_mma, k_attn_wmma) with the ggml headers taken out:
// the bodies and launch shapes are unchanged, so a step computes what the native runner's kernels computed.  New here:
// the timestep sinusoid, SiLU, GELU and txt_in's RMSNorm, written as ggml's CUDA kernels compute them (same formulas,
// float maths); their reductions are fixed-order, so a run gives the same bytes every time.
#include "qi21_kernels.hpp"

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#define QI_HIP 1
#include "strata/hip_compat/cuda_runtime.h"
#include "strata/hip_compat/cuda_fp16.h"
#include "strata/hip_compat/intrinsics.hpp"
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::image::qi21 {
namespace {

constexpr int QI_D = 128;

__device__ __forceinline__ float qi_block_sum(float v, float* red) {
    // blockDim.x is a multiple of 32 and <= 1024
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off, 32);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    float s = (int) threadIdx.x < nw ? red[threadIdx.x] : 0.f;
    if (w == 0) {
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off, 32);
        if (l == 0) red[0] = s;
    }
    __syncthreads();
    return red[0];
}

template <bool RES, bool LN>
__global__ void __launch_bounds__(256) k_res_ln_mod(float* __restrict__ x, const float* __restrict__ h,
                                                   const float* __restrict__ g, float* __restrict__ out,
                                                   const float* __restrict__ m, int C, float eps) {
    __shared__ float red[32];
    const size_t row = blockIdx.x;
    float* xr = x + row * C;
    float v[32];
    int n = 0;
    float s = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float a = xr[c];
        if (RES) {
            a += h[row * C + c] * tanhf(g[c]);
            xr[c] = a;
        }
        v[n++] = a;
        s += a;
    }
    if (!LN) return;
    const float mean = qi_block_sum(s, red) / C;
    float q = 0.f;
    for (int i = 0; i < n; ++i) q += (v[i] - mean) * (v[i] - mean);
    const float inv = rsqrtf(qi_block_sum(q, red) / C + eps);
    n = 0;
    for (int c = threadIdx.x; c < C; c += blockDim.x) out[row * C + c] = (v[n++] - mean) * inv * (1.f + m[c]);
}

template <typename QT>
__device__ __forceinline__ void qi_store4(QT* dst, float a, float b, float c, float d);
template <>
__device__ __forceinline__ void qi_store4<float>(float* dst, float a, float b, float c, float d) {
    *reinterpret_cast<float4*>(dst) = make_float4(a, b, c, d);
}
template <>
__device__ __forceinline__ void qi_store4<half>(half* dst, float a, float b, float c, float d) {
    half2 lo = __floats2half2_rn(a, b), hi2 = __floats2half2_rn(c, d);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&lo);
    u.y = *reinterpret_cast<uint32_t*>(&hi2);
    *reinterpret_cast<uint2*>(dst) = u;
}

// warp per head: grid (T, H/8), block 256; lane holds dims 4*lane..+3 (two RoPE pairs), no shared memory
template <typename QT>
__global__ void __launch_bounds__(256) k_qkv_post_w(const float* __restrict__ q, const float* __restrict__ k,
                                                   const float* __restrict__ v, int ld, const float* __restrict__ wq,
                                                   const float* __restrict__ wk, const float* __restrict__ pe,
                                                   QT* __restrict__ qo, int q_tstride, half* __restrict__ ko,
                                                   half* __restrict__ vo, int kv_tstride, int kv_off, float eps) {
    const int t = blockIdx.x, lane = threadIdx.x & 31, hh = blockIdx.y * 8 + (threadIdx.x >> 5);
    const int d = lane * 4;
    const size_t src = (size_t) t * ld + hh * QI_D + d;
    const float4 p0 = *reinterpret_cast<const float4*>(pe + ((size_t) t * (QI_D / 2) + lane * 2) * 4);
    const float4 p1 = *reinterpret_cast<const float4*>(pe + ((size_t) t * (QI_D / 2) + lane * 2 + 1) * 4);
    for (int which = 0; which < 2; ++which) {
        const float4 x = *reinterpret_cast<const float4*>((which == 0 ? q : k) + src);
        float ss = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off, 32);
        const float inv = rsqrtf(ss / QI_D + eps);
        const float4 w = *reinterpret_cast<const float4*>((which == 0 ? wq : wk) + d);
        const float y0 = x.x * inv * w.x, y1 = x.y * inv * w.y, y2 = x.z * inv * w.z, y3 = x.w * inv * w.w;
        const float r0 = p0.x * y0 + p0.y * y1, r1 = p0.z * y0 + p0.w * y1;
        const float r2 = p1.x * y2 + p1.y * y3, r3 = p1.z * y2 + p1.w * y3;
        if (which == 0)
            qi_store4<QT>(qo + ((size_t) hh * q_tstride + t) * QI_D + d, r0, r1, r2, r3);
        else
            qi_store4<half>(ko + ((size_t) hh * kv_tstride + kv_off + t) * QI_D + d, r0, r1, r2, r3);
    }
    const float4 vv = *reinterpret_cast<const float4*>(v + src);
    qi_store4<half>(vo + ((size_t) hh * kv_tstride + kv_off + t) * QI_D + d, vv.x, vv.y, vv.z, vv.w);
}

__global__ void k_swiglu(const float* __restrict__ gu, float* __restrict__ out, int F, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t t = i / F, c = i % F;
    const float g = gu[t * 2 * F + c], u = gu[t * 2 * F + F + c];
    out[i] = u * (g / (1.f + expf(-g)));
}

// grid (Lq, H), block 128, dynamic smem Lk floats.  causal: query i sees keys 0..i.
__global__ void __launch_bounds__(QI_D) k_attn_causal(const float* __restrict__ q, int q_tstride,
                                                     const half* __restrict__ k, const half* __restrict__ v,
                                                     int kv_tstride, float* __restrict__ out, int H, int Lk,
                                                     int causal, float scale) {
    extern __shared__ float s[];
    __shared__ float red[32];
    __shared__ float qs[QI_D];
    const int i = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    qs[d] = q[((size_t) hh * q_tstride + i) * QI_D + d];
    __syncthreads();
    const int n = causal ? i + 1 : Lk;
    float mx = -INFINITY;
    for (int j = d; j < n; j += QI_D) {
        const half* kr = k + ((size_t) hh * kv_tstride + j) * QI_D;
        float dot = 0.f;
        for (int c = 0; c < QI_D; ++c) dot += qs[c] * __half2float(kr[c]);
        s[j] = dot * scale;
        mx = fmaxf(mx, s[j]);
    }
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, 32));
    __syncthreads();
    if ((d & 31) == 0) red[d >> 5] = mx;
    __syncthreads();
    mx = red[0];
    for (int w = 1; w < QI_D / 32; ++w) mx = fmaxf(mx, red[w]);
    __syncthreads();
    float sum = 0.f;
    for (int j = d; j < n; j += QI_D) {
        s[j] = expf(s[j] - mx);
        sum += s[j];
    }
    sum = qi_block_sum(sum, red);
    float acc = 0.f;
    for (int j = 0; j < n; ++j) acc += s[j] * __half2float(v[((size_t) hh * kv_tstride + j) * QI_D + d]);
    out[(size_t) i * H * QI_D + hh * QI_D + d] = acc / sum;
}

#if defined(QI_HIP)
// WMMA fragments: gfx11 (RDNA3) lanes hold 16 k-values of their row/column (lanes 16..31 duplicate 0..15) and C row
// 2i + (lane >> 4); gfx12 (RDNA4) lanes hold 8 k-values at offset 8 * (lane >> 4) and C row 8 * (lane >> 4) + i.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define QI_RDNA4 1
#define QI_KL 8
#define QI_KOFF(hi) ((hi) * 8)
#define QI_ROW(i, hi) ((hi) * 8 + (i))
#define QI_WMMA __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define QI_RDNA3 1
#endif
#if !defined(QI_KL)
#define QI_KL 16
#define QI_KOFF(hi) 0
#define QI_ROW(i, hi) (2 * (i) + (hi))
#define QI_WMMA __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
#endif
typedef _Float16 qi_h16 __attribute__((ext_vector_type(QI_KL)));
typedef float qi_f8 __attribute__((ext_vector_type(8)));
constexpr int FA_BQ = 128, FA_BKV = 64, FA_NW = 8, FA_KLD = QI_D + 8, FA_VLD = FA_BKV + 8, FA_PLD = FA_BKV + 8;

__device__ __forceinline__ qi_h16 qi_ld16(const _Float16* p, int hi) {
#if QI_KL == 8
    return __builtin_bit_cast(qi_h16, *reinterpret_cast<const uint4*>(p + QI_KOFF(hi)));
#else
    (void) hi;
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(qi_h16, w);
#endif
}

__global__ void __launch_bounds__(FA_NW * 32) k_attn_wmma(const _Float16* __restrict__ Q, int q_tstride,
                                                         const _Float16* __restrict__ K, const _Float16* __restrict__ V,
                                                         int kv_tstride, float* __restrict__ O, int H, int nq, int nkv,
                                                         float scale_log2) {
#if defined(QI_RDNA3) || defined(QI_RDNA4)
    __shared__ __align__(16) _Float16 sK[FA_BKV][FA_KLD];
    __shared__ __align__(16) _Float16 sVt[QI_D][FA_VLD];
    __shared__ __align__(16) _Float16 sP[FA_NW][16][FA_PLD];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int head = blockIdx.y, q0 = blockIdx.x * FA_BQ + wave * 16;
    const _Float16* Qh = Q + (size_t) head * q_tstride * QI_D;
    const _Float16* Kh = K + (size_t) head * kv_tstride * QI_D;
    const _Float16* Vh = V + (size_t) head * kv_tstride * QI_D;
    qi_h16 qa[QI_D / 16];
    {
        const int qr = min(q0 + l16, nq - 1);
        for (int c = 0; c < QI_D / 16; ++c) qa[c] = qi_ld16(Qh + (size_t) qr * QI_D + c * 16, hi);
    }
    qi_f8 o[QI_D / 16];
    for (int c = 0; c < QI_D / 16; ++c) o[c] = qi_f8{};
    float m[8], l[8];
    for (int i = 0; i < 8; ++i) { m[i] = -INFINITY; l[i] = 0.f; }
    for (int k0 = 0; k0 < nkv; k0 += FA_BKV) {
        __syncthreads();
        for (int e = tid * 8; e < FA_BKV * QI_D; e += FA_NW * 32 * 8) {
            const int r = e / QI_D, c = e % QI_D, key = k0 + r;
            uint4 kv = make_uint4(0, 0, 0, 0);
            if (key < nkv) kv = *reinterpret_cast<const uint4*>(Kh + (size_t) key * QI_D + c);
            *reinterpret_cast<uint4*>(&sK[r][c]) = kv;
        }
        for (int e = tid; e < (FA_BKV / 4) * (QI_D / 8); e += FA_NW * 32) {
            const int r4 = (e / (QI_D / 8)) * 4, c = (e % (QI_D / 8)) * 8;
            _Float16 vv[4][8];
            for (int j = 0; j < 4; ++j) {
                uint4 x = make_uint4(0, 0, 0, 0);
                if (k0 + r4 + j < nkv) x = *reinterpret_cast<const uint4*>(Vh + (size_t) (k0 + r4 + j) * QI_D + c);
                *reinterpret_cast<uint4*>(vv[j]) = x;
            }
            for (int dd = 0; dd < 8; ++dd) {
                _Float16 w4[4] = {vv[0][dd], vv[1][dd], vv[2][dd], vv[3][dd]};
                *reinterpret_cast<uint2*>(&sVt[c + dd][r4]) = *reinterpret_cast<uint2*>(w4);
            }
        }
        __syncthreads();
        qi_f8 s[FA_BKV / 16];
        for (int t = 0; t < FA_BKV / 16; ++t) {
            s[t] = qi_f8{};
            for (int c = 0; c < QI_D / 16; ++c) s[t] = QI_WMMA(qa[c], qi_ld16(&sK[t * 16 + l16][c * 16], hi), s[t]);
        }
        float alpha[8];
        for (int i = 0; i < 8; ++i) {
            float mx = -INFINITY;
            for (int t = 0; t < FA_BKV / 16; ++t) {
                const float val = k0 + t * 16 + l16 < nkv ? s[t][i] * scale_log2 : -INFINITY;
                s[t][i] = val;
                mx = fmaxf(mx, val);
            }
            for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
            const float mnew = fmaxf(m[i], mx);
            alpha[i] = exp2f(m[i] - mnew);
            float sum = 0.f;
            for (int t = 0; t < FA_BKV / 16; ++t) {
                const float pv = exp2f(s[t][i] - mnew);
                s[t][i] = pv;
                sum += pv;
            }
            for (int off = 1; off < 16; off <<= 1) sum += __shfl_xor(sum, off, 32);
            l[i] = l[i] * alpha[i] + sum;
            m[i] = mnew;
        }
        for (int c = 0; c < QI_D / 16; ++c)
            for (int i = 0; i < 8; ++i) o[c][i] *= alpha[i];
        for (int t = 0; t < FA_BKV / 16; ++t)
            for (int i = 0; i < 8; ++i) sP[wave][QI_ROW(i, hi)][t * 16 + l16] = (_Float16) s[t][i];
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
        for (int kc = 0; kc < FA_BKV / 16; ++kc) {
            const qi_h16 pa = qi_ld16(&sP[wave][l16][kc * 16], hi);
            for (int c = 0; c < QI_D / 16; ++c) o[c] = QI_WMMA(pa, qi_ld16(&sVt[c * 16 + l16][kc * 16], hi), o[c]);
        }
    }
    for (int c = 0; c < QI_D / 16; ++c)
        for (int i = 0; i < 8; ++i) {
            const int row = q0 + QI_ROW(i, hi);
            if (row < nq) O[(size_t) row * H * QI_D + head * QI_D + c * 16 + l16] = o[c][i] / l[i];
        }
#else
    __builtin_trap();
#endif
}
#else   // NVIDIA
// sm_80+: mma.sync m16n8k16 f16 -> f32 flash attention.  8 warps x 16 queries per block, 64-key tiles of K and V
// (row-major) double-buffered with cp.async, ldmatrix (.trans for V) fragments, fp32 online softmax, P in fp16.
constexpr int MF_NW = 8, MF_BQ = MF_NW * 16, MF_BKV = 64, MF_LD = QI_D + 8;
constexpr size_t MF_SMEM = (size_t) 4 * MF_BKV * MF_LD * sizeof(half);

__device__ __forceinline__ void qi_mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void) c; (void) a; (void) b;
    __trap();
#endif
}
__device__ __forceinline__ uint32_t qi_pack2(float x, float y) {
    half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void qi_ldsm_x4(uint32_t* r, const half* p, bool trans) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    if (trans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#else
    (void) r; (void) p; (void) trans;
#endif
}
__device__ __forceinline__ void qi_cp16(half* dst, const half* src, bool ok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(dst);
    const int n = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(a), "l"(src), "r"(n));
#else
    (void) dst; (void) src; (void) ok;
#endif
}

__global__ void __launch_bounds__(MF_NW * 32) k_attn_mma(const half* __restrict__ Q, int q_tstride,
                                                        const half* __restrict__ K, const half* __restrict__ V,
                                                        int kv_tstride, float* __restrict__ O, int H, int nq, int nkv,
                                                        float scale_log2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) half qi_smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r = lane >> 2, c2 = (lane & 3) * 2, lr = lane & 7, lq = lane >> 3;
    const int head = blockIdx.y, q0 = blockIdx.x * MF_BQ + warp * 16;
    const half* Qh = Q + (size_t) head * q_tstride * QI_D;
    const half* Kh = K + (size_t) head * kv_tstride * QI_D;
    const half* Vh = V + (size_t) head * kv_tstride * QI_D;
    auto tile = [&](int st, int kv) { return qi_smem + ((st * 2 + kv) * MF_BKV) * MF_LD; };
    auto load = [&](int st, int k0) {
        for (int e = tid * 8; e < MF_BKV * QI_D; e += MF_NW * 32 * 8) {
            const int rr = e / QI_D, cc = e % QI_D, key = k0 + rr;
            const bool ok = key < nkv;
            const size_t off = (size_t) (ok ? key : 0) * QI_D + cc;
            qi_cp16(tile(st, 0) + rr * MF_LD + cc, Kh + off, ok);
            qi_cp16(tile(st, 1) + rr * MF_LD + cc, Vh + off, ok);
        }
        asm volatile("cp.async.commit_group;\n");
    };
    uint32_t qa[QI_D / 16][4];
    {
        const int r0 = min(q0 + r, nq - 1), r1 = min(q0 + r + 8, nq - 1);
        for (int kc = 0; kc < QI_D / 16; ++kc) {
            qa[kc][0] = *reinterpret_cast<const uint32_t*>(Qh + (size_t) r0 * QI_D + kc * 16 + c2);
            qa[kc][1] = *reinterpret_cast<const uint32_t*>(Qh + (size_t) r1 * QI_D + kc * 16 + c2);
            qa[kc][2] = *reinterpret_cast<const uint32_t*>(Qh + (size_t) r0 * QI_D + kc * 16 + c2 + 8);
            qa[kc][3] = *reinterpret_cast<const uint32_t*>(Qh + (size_t) r1 * QI_D + kc * 16 + c2 + 8);
        }
    }
    float o[QI_D / 8][4];
    for (int n = 0; n < QI_D / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const int ntiles = (nkv + MF_BKV - 1) / MF_BKV;
    load(0, 0);
    for (int it = 0; it < ntiles; ++it) {
        const int st = it & 1, k0 = it * MF_BKV;
        if (it + 1 < ntiles) {
            load(st ^ 1, k0 + MF_BKV);
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();
        const half* sK = tile(st, 0);
        const half* sV = tile(st, 1);
        float s[MF_BKV / 8][4];
        for (int t = 0; t < MF_BKV / 8; ++t) {
            s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.f;
            for (int kc = 0; kc < QI_D / 16; kc += 2) {
                uint32_t b[4];
                qi_ldsm_x4(b, sK + (t * 8 + lr) * MF_LD + kc * 16 + lq * 8, false);
                qi_mma16816(s[t], qa[kc], b);
                qi_mma16816(s[t], qa[kc + 1], b + 2);
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
        for (int t = 0; t < MF_BKV / 8; ++t)
            for (int j = 0; j < 2; ++j) {
                const bool ok = k0 + t * 8 + c2 + j < nkv;
                s[t][j] = ok ? s[t][j] * scale_log2 : -INFINITY;
                s[t][2 + j] = ok ? s[t][2 + j] * scale_log2 : -INFINITY;
                mx0 = fmaxf(mx0, s[t][j]);
                mx1 = fmaxf(mx1, s[t][2 + j]);
            }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, off));
        }
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float a0 = exp2f(m0 - mn0), a1 = exp2f(m1 - mn1);
        float sum0 = 0.f, sum1 = 0.f;
        for (int t = 0; t < MF_BKV / 8; ++t)
            for (int j = 0; j < 2; ++j) {
                s[t][j] = exp2f(s[t][j] - mn0);
                s[t][2 + j] = exp2f(s[t][2 + j] - mn1);
                sum0 += s[t][j];
                sum1 += s[t][2 + j];
            }
        for (int off = 1; off < 4; off <<= 1) {
            sum0 += __shfl_xor_sync(0xffffffff, sum0, off);
            sum1 += __shfl_xor_sync(0xffffffff, sum1, off);
        }
        l0 = l0 * a0 + sum0;
        l1 = l1 * a1 + sum1;
        m0 = mn0;
        m1 = mn1;
        for (int n = 0; n < QI_D / 8; ++n) {
            o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1;
        }
        for (int kc = 0; kc < MF_BKV / 16; ++kc) {
            uint32_t pa[4];
            pa[0] = qi_pack2(s[2 * kc][0], s[2 * kc][1]);
            pa[1] = qi_pack2(s[2 * kc][2], s[2 * kc][3]);
            pa[2] = qi_pack2(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[3] = qi_pack2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
            for (int n = 0; n < QI_D / 8; n += 2) {
                uint32_t b[4];
                qi_ldsm_x4(b, sV + (kc * 16 + (lq & 1) * 8 + lr) * MF_LD + n * 8 + (lq >> 1) * 8, true);
                qi_mma16816(o[n], pa, b);
                qi_mma16816(o[n + 1], pa, b + 2);
            }
        }
        __syncthreads();
    }
    const int row0 = q0 + r, row1 = q0 + r + 8;
    for (int n = 0; n < QI_D / 8; ++n) {
        if (row0 < nq)
            *reinterpret_cast<float2*>(O + (size_t) row0 * H * QI_D + head * QI_D + n * 8 + c2) =
                make_float2(o[n][0] / l0, o[n][1] / l0);
        if (row1 < nq)
            *reinterpret_cast<float2*>(O + (size_t) row1 * H * QI_D + head * QI_D + n * 8 + c2) =
                make_float2(o[n][2] / l1, o[n][3] / l1);
    }
#else
    __trap();
#endif
}
#endif  // QI_HIP

// ---- the steps ggml did for the native runner (ggml-cuda's formulas, float maths)
__global__ void k_timestep(const float* __restrict__ t, float* __restrict__ emb, int dim, float max_period) {
    const int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x, half_dim = dim / 2;
    if (j >= half_dim) return;
    const float freq = expf(-logf(max_period) * j / half_dim);
    const float arg = t[i] * freq;
    emb[(size_t) i * dim + j] = cosf(arg);
    emb[(size_t) i * dim + j + half_dim] = sinf(arg);
}

__global__ void k_silu(float* __restrict__ x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.f + expf(-x[i]));
}

__global__ void k_gelu(float* __restrict__ x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float GELU_COEF_A = 0.044715f, SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const float v = x[i];
    x[i] = 0.5f * v * (1.f + tanhf(SQRT_2_OVER_PI * v * (1.f + GELU_COEF_A * v * v)));
}

__global__ void __launch_bounds__(256) k_rms_w1(const float* __restrict__ x, const float* __restrict__ w,
                                               float* __restrict__ out, int C, float eps) {
    __shared__ float red[32];
    const size_t row = blockIdx.x;
    float s = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) s += x[row * C + c] * x[row * C + c];
    const float inv = rsqrtf(qi_block_sum(s, red) / C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) out[row * C + c] = x[row * C + c] * inv * (w[c] + 1.f);
}

inline cudaStream_t qs(void* s) { return static_cast<cudaStream_t>(s); }
inline int launched() { return cudaGetLastError() == cudaSuccess ? 0 : 2; }
inline unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

int res_ln_mod(void* stream, float* x, const float* h, const float* g, float* out, const float* m, int T, int C, float eps) {
    if (C > 32 * 256 || T <= 0) return T <= 0 ? 0 : 1;
    if (h != nullptr && out != nullptr)
        k_res_ln_mod<true, true><<<T, 256, 0, qs(stream)>>>(x, h, g, out, m, C, eps);
    else if (h != nullptr)
        k_res_ln_mod<true, false><<<T, 256, 0, qs(stream)>>>(x, h, g, out, m, C, eps);
    else
        k_res_ln_mod<false, true><<<T, 256, 0, qs(stream)>>>(x, h, g, out, m, C, eps);
    return launched();
}

int qkv_post(void* stream, const float* q, const float* k, const float* v, int ld, const float* wq, const float* wk,
             const float* pe, void* qo, int q_f16, int q_tstride, void* ko, void* vo, int kv_tstride, int kv_off, int T,
             int H, float eps) {
    if (H % 8 != 0 || ld % 4 != 0) return 1;
    const dim3 gw(T, H / 8);
    if (q_f16)
        k_qkv_post_w<half><<<gw, 256, 0, qs(stream)>>>(q, k, v, ld, wq, wk, pe, static_cast<half*>(qo), q_tstride,
                                                       static_cast<half*>(ko), static_cast<half*>(vo), kv_tstride, kv_off,
                                                       eps);
    else
        k_qkv_post_w<float><<<gw, 256, 0, qs(stream)>>>(q, k, v, ld, wq, wk, pe, static_cast<float*>(qo), q_tstride,
                                                        static_cast<half*>(ko), static_cast<half*>(vo), kv_tstride, kv_off,
                                                        eps);
    return launched();
}

int swiglu(void* stream, const float* gu, float* out, int T, int F) {
    const int64_t n = (int64_t) T * F;
    k_swiglu<<<blocks(n), 256, 0, qs(stream)>>>(gu, out, F, n);
    return launched();
}

int attn_exact(void* stream, const float* q, int q_tstride, const void* k, const void* v, int kv_tstride, float* out,
               int H, int Lq, int Lk, int causal) {
    const size_t smem = (size_t) Lk * sizeof(float);
    if (smem > 48 * 1024) return 1;
    k_attn_causal<<<dim3(Lq, H), QI_D, smem, qs(stream)>>>(q, q_tstride, static_cast<const half*>(k),
                                                           static_cast<const half*>(v), kv_tstride, out, H, Lk, causal,
                                                           1.f / sqrtf((float) QI_D));
    return launched();
}

int has_fast_attn(int device) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
#if defined(QI_HIP)
    const char* arch = prop.gcnArchName;
    for (const char* p : {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152"})
        if (std::strncmp(arch, p, 7) == 0) return 1;
    const char* g12 = std::getenv("STRATA_QI_GFX12_ATTN");
    const bool rdna4 = std::strncmp(arch, "gfx1200", 7) == 0 || std::strncmp(arch, "gfx1201", 7) == 0;
    return rdna4 && g12 != nullptr && g12[0] == '1' ? 1 : 0;
#else
    return prop.major >= 8 ? 1 : 0;
#endif
}

int attn_fast(void* stream, const void* q, int q_tstride, const void* k, const void* v, int kv_tstride, float* out, int H,
              int nq, int nkv) {
    const float sl = (float) (1.0 / std::sqrt((double) QI_D) * 1.4426950408889634);
#if defined(QI_HIP)
    k_attn_wmma<<<dim3((nq + FA_BQ - 1) / FA_BQ, H), FA_NW * 32, 0, qs(stream)>>>(
        static_cast<const _Float16*>(q), q_tstride, static_cast<const _Float16*>(k), static_cast<const _Float16*>(v),
        kv_tstride, out, H, nq, nkv, sl);
#else
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(k_attn_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) MF_SMEM) != cudaSuccess)
            return 2;
        attr = true;
    }
    k_attn_mma<<<dim3((nq + MF_BQ - 1) / MF_BQ, H), MF_NW * 32, MF_SMEM, qs(stream)>>>(
        static_cast<const half*>(q), q_tstride, static_cast<const half*>(k), static_cast<const half*>(v), kv_tstride, out,
        H, nq, nkv, sl);
#endif
    return launched();
}

int timestep_embedding(void* stream, const float* t, float* emb, int n, int dim, float max_period) {
    if (dim % 2 != 0) return 1;
    k_timestep<<<dim3((dim / 2 + 255) / 256, n), 256, 0, qs(stream)>>>(t, emb, dim, max_period);
    return launched();
}

int silu(void* stream, float* x, int64_t n) {
    k_silu<<<blocks(n), 256, 0, qs(stream)>>>(x, n);
    return launched();
}

int gelu_tanh(void* stream, float* x, int64_t n) {
    k_gelu<<<blocks(n), 256, 0, qs(stream)>>>(x, n);
    return launched();
}

int rms_norm_w1(void* stream, const float* x, const float* w, float* out, int T, int C, float eps) {
    if (T <= 0) return 0;
    k_rms_w1<<<T, 256, 0, qs(stream)>>>(x, w, out, C, eps);
    return launched();
}

}  // namespace strata::image::qi21
