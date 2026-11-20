// hy4_vision.hip.h - the Hy4-Flash image path on the GPU (HIP, portable to CUDA through hy4_gpu.h): the 27-layer ViT + rmsnorm_mlp_adapter of Hy4-Flash-Vision.strata, and the small
// kernels the runner needs to put the image rows into a prefill (embedding scatter, 3D rope positions for the MLA / indexer rope).
//
// VisionEncoder: load() streams the 934 MB shard into device memory (on demand, not at start), encode() takes a resized RGB picture and produces the image rows [(H/32)*(W/32+1)][3072]
// fp32 in the LM's embedding space (the rows replace the embeddings of the image-pad tokens; the last row of every merged row is the learned `image_newline`).
// Numerics: fp16 weights, fp32 residual stream and accumulation, fp16 GEMM inputs (like the LM runner). All kernels are plain (no matrix cores) so every backend can build them;
// the GEMM and the attention are written for correctness first (see HY4-VISION-DESIGN.md for the measured speed).
#pragma once
#include "hy4_gpu.h"

#include <atomic>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "hy4_gpu_model.hip.h"
#include "hy4_kernels.hip.h"
#include "hy4_vision.h"

namespace hy4 {
namespace vit {

__device__ __forceinline__ float gelu_tanh_f(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
__device__ __forceinline__ float gelu_erf_f(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }

// ---- patchify: rgb u8 [h][w][3] -> fp16 [N][768] = (C, 16, 16) flattened per patch, (v/255 - 0.5)/0.5, patches in raster order -----------------------------------------------
__global__ void k_patchify(const uint8_t* __restrict__ rgb, __half* __restrict__ pix, int w, int gw) {
    const int pidx = blockIdx.x;  // patch
    const int py = pidx / gw, px = pidx % gw;
    for (int e = threadIdx.x; e < 768; e += blockDim.x) {
        const int c = e >> 8, dy = (e >> 4) & 15, dx = e & 15;
        const float v = (float)rgb[((size_t)(py * 16 + dy) * w + px * 16 + dx) * 3 + c] * (1.0f / 255.0f);
        pix[(size_t)pidx * 768 + e] = __float2half((v - 0.5f) / 0.5f);
    }
}

// ---- learned position grid (row 0 = unused cls, rows 1.. = 128x128 patch grid, fp16) resampled bilinearly to gh x gw like torch F.interpolate(scale_factor=(gh+0.1)/128, align_corners=False)
// sy / sx = 1/scale_factor (float, as torch computes it)
__global__ void k_posemb(const __half* __restrict__ tab, float* __restrict__ out, int gh, int gw, int edge, float sy, float sx, int D) {
    const int p = blockIdx.x;
    const int y = p / gw, x = p % gw;
    float fy = sy * ((float)y + 0.5f) - 0.5f, fx = sx * ((float)x + 0.5f) - 0.5f;
    if (fy < 0.f) fy = 0.f;
    if (fx < 0.f) fx = 0.f;
    int y0 = (int)fy, x0 = (int)fx;
    const int y1 = y0 < edge - 1 ? y0 + 1 : y0, x1 = x0 < edge - 1 ? x0 + 1 : x0;
    if (y0 > edge - 1) y0 = edge - 1;
    if (x0 > edge - 1) x0 = edge - 1;
    const float ly = fy - (float)y0, lx = fx - (float)x0;
    const __half* t = tab + (size_t)D;  // skip the cls row
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        const float a = __half2float(t[((size_t)y0 * edge + x0) * D + c]), b = __half2float(t[((size_t)y0 * edge + x1) * D + c]);
        const float cc = __half2float(t[((size_t)y1 * edge + x0) * D + c]), d = __half2float(t[((size_t)y1 * edge + x1) * D + c]);
        out[(size_t)p * D + c] = (1.f - ly) * ((1.f - lx) * a + lx * b) + ly * ((1.f - lx) * cc + lx * d);
    }
}

// ---- LayerNorm (biased variance, affine): fp32 [N][D] -> fp16 [N][D] -----------------------------------------------------------------------------------------------------
__global__ void k_ln(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, __half* __restrict__ out, int D, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t)blockIdx.x * D;
    float s = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s, sh) / (float)D;
    float v = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float inv = rsqrtf(block_sum(v, sh) / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[(size_t)blockIdx.x * D + i] = __float2half((xr[i] - mean) * inv * w[i] + b[i]);
}

// ---- RMSNorm of the ViT output (perceive.norm) written straight into the 2x2-merge matrix A [(gh/2)*(gw/2)][1152*4], column c*4 + dy*2 + dx (the conv weight's (C, dy, dx) order)
__global__ void k_rms_merge(const float* __restrict__ x, const float* __restrict__ w, __half* __restrict__ A, int gw, int D, float eps) {
    __shared__ float sh[32];
    const int p = blockIdx.x;
    const int y = p / gw, xx = p % gw;
    const float* xr = x + (size_t)p * D;
    float ss = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(ss, sh) / (float)D + eps);
    const size_t cell = (size_t)(y >> 1) * (gw >> 1) + (xx >> 1);
    const int sub = (y & 1) * 2 + (xx & 1);
    for (int i = threadIdx.x; i < D; i += blockDim.x) A[cell * (size_t)(D * 4) + (size_t)i * 4 + sub] = __float2half(xr[i] * inv * w[i]);
}

// ---- GEMM Y = X[M][K] (fp16) * W[N][K]^T (fp16) + bias, 64x64 tile, BK 16, 256 threads (4x4 per thread), fp32 accumulation -------------------------------------------------------
constexpr int VA_HD = 72;  // head dimension (1152 / 16)
enum { EPI_H = 0, EPI_GELU_TANH = 1, EPI_GELU_ERF = 2, EPI_RES_ADD = 3, EPI_RES_SET_POS = 4, EPI_F32_ROWMAP = 5, EPI_VT = 6 };

// the epilogue shared by both GEMMs: one output element (row m, column n of an [M][N] result)
template <int EPI>
__device__ __forceinline__ void store_epi(float v, int m, int n, int N, int ow, __half* Yh, float* Yf, const float* pe) {
    if (EPI == EPI_H) Yh[(size_t)m * N + n] = __float2half(v);
    else if (EPI == EPI_GELU_TANH) Yh[(size_t)m * N + n] = __float2half(gelu_tanh_f(v));
    else if (EPI == EPI_GELU_ERF) Yh[(size_t)m * N + n] = __float2half(gelu_erf_f(v));
    else if (EPI == EPI_RES_ADD) Yf[(size_t)m * N + n] += v;
    else if (EPI == EPI_RES_SET_POS) Yf[(size_t)m * N + n] = v + pe[(size_t)m * N + n];
    else if (EPI == EPI_VT) Yh[(size_t)n * ow + m] = __float2half(v);  // transposed [N][ow]: ow = the key pitch of the attention kernel
    else Yf[(size_t)(m + m / ow) * N + n] = v;                      // EPI_F32_ROWMAP: merged cell m = oy*ow + ox -> row oy*(ow+1) + ox
}

template <int EPI>
__global__ void __launch_bounds__(256) k_gemm(const __half* __restrict__ X, const __half* __restrict__ W, const float* __restrict__ bias, __half* __restrict__ Yh, float* __restrict__ Yf,
                                              const float* __restrict__ pe, int M, int N, int K, int ow) {
    __shared__ float As[16][65], Bs[16][65];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const int ty = tid >> 4, tx = tid & 15;
    const int lr = tid >> 2, lk = (tid & 3) * 4;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    const bool am = (m0 + lr) < M, bn = (n0 + lr) < N;
    const __half* xp = X + (size_t)(m0 + lr) * K + lk;
    const __half* wp = W + (size_t)(n0 + lr) * K + lk;
    for (int k0 = 0; k0 < K; k0 += 16) {
        float a4[4] = {0, 0, 0, 0}, b4[4] = {0, 0, 0, 0};
        if (am) {
            const uint2 u = *reinterpret_cast<const uint2*>(xp + k0);
            float x0, x1, x2, x3;
            h2_unpack(u.x, x0, x1);
            h2_unpack(u.y, x2, x3);
            a4[0] = x0; a4[1] = x1; a4[2] = x2; a4[3] = x3;
        }
        if (bn) {
            const uint2 u = *reinterpret_cast<const uint2*>(wp + k0);
            float x0, x1, x2, x3;
            h2_unpack(u.x, x0, x1);
            h2_unpack(u.y, x2, x3);
            b4[0] = x0; b4[1] = x1; b4[2] = x2; b4[3] = x3;
        }
        __syncthreads();
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            As[lk + e][lr] = a4[e];
            Bs[lk + e][lr] = b4[e];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx + 16 * j;
            if (n >= N) continue;
            store_epi<EPI>(acc[i][j] + (bias ? bias[n] : 0.f), m, n, N, ow, Yh, Yf, pe);
        }
    }
}

#if !HY4_CUDA
// ---- Wave32 fdot2 GEMM for RDNA2 (and non-WMMA HIP): 64x64 tile, BK 16 (8 __half2 pairs), 256 threads (4x4 per thread), fp32 accumulation
template <int EPI>
__global__ void __launch_bounds__(256) k_gemm_fdot2(const __half* __restrict__ X, const __half* __restrict__ W, const float* __restrict__ bias, __half* __restrict__ Yh, float* __restrict__ Yf,
                                                    const float* __restrict__ pe, int M, int N, int K, int ow) {
    __shared__ uint32_t As[8][65], Bs[8][65];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const int ty = tid >> 4, tx = tid & 15;
    const int lr = tid >> 2, lk = (tid & 3) * 4;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    const bool am = (m0 + lr) < M, bn = (n0 + lr) < N;
    const __half* xp = X + (size_t)(m0 + lr) * K + lk;
    const __half* wp = W + (size_t)(n0 + lr) * K + lk;
    for (int k0 = 0; k0 < K; k0 += 16) {
        uint2 ua = {0u, 0u}, ub = {0u, 0u};
        if (am) ua = *reinterpret_cast<const uint2*>(xp + k0);
        if (bn) ub = *reinterpret_cast<const uint2*>(wp + k0);
        __syncthreads();
        const int kk0 = lk >> 1;
        As[kk0][lr] = ua.x;
        As[kk0 + 1][lr] = ua.y;
        Bs[kk0][lr] = ub.x;
        Bs[kk0 + 1][lr] = ub.y;
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            uint32_t a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = h2_dot_f32(a[i], b[j], acc[i][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx + 16 * j;
            if (n >= N) continue;
            store_epi<EPI>(acc[i][j] + (bias ? bias[n] : 0.f), m, n, N, ow, Yh, Yf, pe);
        }
    }
}

// ---- Wave32 fdot2 Flash Attention for RDNA2 (4 threads per query, 32 queries per 128-thread block, head_dim 72 = 36 half2 pairs, transposed V in LDS)
__global__ void __launch_bounds__(128) k_attn_fdot2(const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v, __half* __restrict__ o, int N, int D, float scale) {
    __shared__ uint32_t Ks[32][37];
    __shared__ uint32_t Vst[16][VA_HD + 1];
    const int tid = threadIdx.x;
    const int q_loc = tid >> 2, lane4 = tid & 3;
    const int h = blockIdx.y;
    const int qi = blockIdx.x * 32 + q_loc;
    uint32_t qh[9];
    if (qi < N) {
        const uint32_t* qp = reinterpret_cast<const uint32_t*>(q + (size_t)qi * D + h * VA_HD) + lane4 * 9;
#pragma unroll
        for (int d = 0; d < 9; ++d) qh[d] = qp[d];
    } else {
#pragma unroll
        for (int d = 0; d < 9; ++d) qh[d] = 0u;
    }
    float acc[18];
#pragma unroll
    for (int d = 0; d < 18; ++d) acc[d] = 0.f;
    float mrun = -3.0e38f, lrun = 0.f;
    for (int kb = 0; kb < N; kb += 32) {
        __syncthreads();
        for (int e = tid; e < 32 * 36; e += 128) {
            const int j = e / 36, d2 = e % 36;
            const int kj = kb + j;
            Ks[j][d2] = kj < N ? reinterpret_cast<const uint32_t*>(k + (size_t)kj * D + h * VA_HD)[d2] : 0u;
        }
        for (int e = tid; e < 16 * VA_HD; e += 128) {
            const int jp = e / VA_HD, d = e % VA_HD;
            const int k0 = kb + 2 * jp, k1 = k0 + 1;
            const uint16_t h0 = k0 < N ? *reinterpret_cast<const uint16_t*>(v + (size_t)k0 * D + h * VA_HD + d) : 0u;
            const uint16_t h1 = k1 < N ? *reinterpret_cast<const uint16_t*>(v + (size_t)k1 * D + h * VA_HD + d) : 0u;
            Vst[jp][d] = (uint32_t)h0 | ((uint32_t)h1 << 16);
        }
        __syncthreads();
        const int nj = min(32, N - kb);
        float s[32];
        float mt = -3.0e38f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            float dot = 0.f;
#pragma unroll
            for (int d = 0; d < 9; ++d) dot = h2_dot_f32(qh[d], Ks[j][lane4 * 9 + d], dot);
            dot += __shfl_xor(dot, 1, 4);
            dot += __shfl_xor(dot, 2, 4);
            s[j] = j < nj ? dot * scale : -3.0e38f;
            mt = fmaxf(mt, s[j]);
        }
        const float mn = fmaxf(mrun, mt);
        const float corr = __expf(mrun - mn);
        lrun *= corr;
#pragma unroll
        for (int d = 0; d < 18; ++d) acc[d] *= corr;
#pragma unroll
        for (int jp = 0; jp < 16; ++jp) {
            const float p0 = (2 * jp < nj) ? __expf(s[2 * jp] - mn) : 0.f;
            const float p1 = (2 * jp + 1 < nj) ? __expf(s[2 * jp + 1] - mn) : 0.f;
            lrun += p0 + p1;
            const uint32_t p2 = h2_pack(p0, p1);
#pragma unroll
            for (int d = 0; d < 18; ++d) acc[d] = h2_dot_f32(p2, Vst[jp][lane4 * 18 + d], acc[d]);
        }
        mrun = mn;
    }
    if (qi < N) {
        const float inv = 1.0f / lrun;
        __half* op = o + (size_t)qi * D + h * VA_HD + lane4 * 18;
#pragma unroll
        for (int d = 0; d < 18; ++d) op[d] = __float2half(acc[d] * inv);
    }
}
#endif  // !HY4_CUDA

#if HY4_MMA && !HY4_CUDA
// ---- gfx11 / gfx12 WMMA versions (v_wmma_f32_16x16x16_f16, wave32).
// gfx11: lane l holds all 16 k of row/col (l&15); acc row = 2*i + (l>>4).
// gfx12: lane l holds 8 k at offset 8*(l>>4) of row/col (l&15); acc row = 8*(l>>4) + i.
#if HY4_GFX12
typedef _Float16 hv_t __attribute__((ext_vector_type(8)));
typedef float fv8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ hv_t ld_hv(const void* p, int hi) {
    const uint4 a = *reinterpret_cast<const uint4*>(reinterpret_cast<const _Float16*>(p) + hi * 8);
    return __builtin_bit_cast(hv_t, a);
}
// 8 valid halves (16 bytes at p) followed by 8 zeros: the 72-wide head dimension padded to 80
__device__ __forceinline__ hv_t ld_hv_half(const void* p, int hi) {
    if (hi == 0) return __builtin_bit_cast(hv_t, *reinterpret_cast<const uint4*>(p));
    return hv_t{0, 0, 0, 0, 0, 0, 0, 0};
}
__device__ __forceinline__ fv8 vit_wmma(hv_t a, hv_t b, fv8 c) {
#if defined(__HIP_DEVICE_COMPILE__) && !HY4_DEV_WMMA
    (void)a; (void)b;
    return c;
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#endif
}
__device__ __forceinline__ int vit_acc_m(int i, int hi) { return 8 * hi + i; }
#else
typedef _Float16 hv16 __attribute__((ext_vector_type(16)));
typedef hv16 hv_t;
typedef float fv8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ hv_t ld_hv(const void* p, int /*hi*/) {
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *(reinterpret_cast<const uint4*>(p) + 1);
    u32x8 v = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(hv_t, v);
}
// 8 valid halves (16 bytes at p) followed by 8 zeros: the 72-wide head dimension padded to 80
__device__ __forceinline__ hv_t ld_hv_half(const void* p, int /*hi*/) {
    const uint4 a = *reinterpret_cast<const uint4*>(p);
    u32x8 v = {a.x, a.y, a.z, a.w, 0u, 0u, 0u, 0u};
    return __builtin_bit_cast(hv_t, v);
}
__device__ __forceinline__ fv8 vit_wmma(hv_t a, hv_t b, fv8 c) {
#if defined(__HIP_DEVICE_COMPILE__) && !HY4_DEV_WMMA
    (void)a; (void)b;
    return c;
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
}
__device__ __forceinline__ int vit_acc_m(int i, int hi) { return 2 * i + hi; }
#endif

// Y = X * W^T + bias, block 64x64 (4 waves, each 32x32 = 2x2 WMMA tiles), operands straight from global memory (rows clamped; the store is guarded). K % 16 == 0.
template <int EPI>
__global__ void __launch_bounds__(128) k_gemm_w(const __half* __restrict__ X, const __half* __restrict__ W, const float* __restrict__ bias, __half* __restrict__ Yh, float* __restrict__ Yf,
                                                const float* __restrict__ pe, int M, int N, int K, int ow) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int m0 = blockIdx.y * 64 + wm * 32, n0 = blockIdx.x * 64 + wn * 32;
    const __half* xr[2];
    const __half* wr[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        xr[t] = X + (size_t)min(m0 + t * 16 + l16, M - 1) * K;
        wr[t] = W + (size_t)min(n0 + t * 16 + l16, N - 1) * K;
    }
    fv8 acc[2][2];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b) acc[a][b] = fv8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 4
    for (int k = 0; k < K; k += 16) {
        const hv_t a0 = ld_hv(xr[0] + k, hi), a1 = ld_hv(xr[1] + k, hi), b0 = ld_hv(wr[0] + k, hi), b1 = ld_hv(wr[1] + k, hi);
        acc[0][0] = vit_wmma(a0, b0, acc[0][0]);
        acc[0][1] = vit_wmma(a0, b1, acc[0][1]);
        acc[1][0] = vit_wmma(a1, b0, acc[1][0]);
        acc[1][1] = vit_wmma(a1, b1, acc[1][1]);
    }
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int n = n0 + b * 16 + l16;
            if (n >= N) continue;
            const float bv = bias ? bias[n] : 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int m = m0 + a * 16 + vit_acc_m(i, hi);
                if (m < M) store_epi<EPI>(acc[a][b][i] + bv, m, n, N, ow, Yh, Yf, pe);
            }
        }
}

// Flash attention on WMMA: block = 4 waves, wave = 16 queries of one head, loop over key tiles of 16. S = Q K^T (head dim 72 padded to 80: 4 full k-steps + a half one), online softmax in fp32,
// P goes through a wave-private LDS tile into the A operand of P V. V arrives TRANSPOSED (vt[n = 72 h + d][key], row pitch ldv, so a column of V is a contiguous run of keys).
constexpr int WA_PP = 72;   // LDS pitch (halves) of a wave's 16 x 64 probability tile
constexpr int WA_KB = 64;   // keys per loop iteration: the row max / correction / oacc rescale are done once per 64 keys, the row sums are kept per lane and reduced at the end
__global__ void __launch_bounds__(128) k_attn_w(const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ vt, __half* __restrict__ o, int N, int D, int ldv,
                                                float scale) {
    __shared__ __align__(16) __half sP[4][16 * WA_PP];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int l16 = lane & 15, hi = lane >> 4;
    const int h = blockIdx.y;
    const int q0 = blockIdx.x * 64 + wave * 16;
    hv_t qf[5];
    {
        const __half* qp = q + (size_t)min(q0 + l16, N - 1) * D + h * VA_HD;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) qf[ks] = ld_hv(qp + ks * 16, hi);
        qf[4] = ld_hv_half(qp + 64, hi);
    }
    fv8 oacc[5];
#pragma unroll
    for (int t = 0; t < 5; ++t) oacc[t] = fv8{0, 0, 0, 0, 0, 0, 0, 0};
    float mrun[8], lpart[8];  // lpart: this lane's share of the row sum (its column of every 16-key tile)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        mrun[i] = -1e30f;
        lpart[i] = 0.f;
    }
    __half* sp = sP[wave];
    for (int kb = 0; kb < N; kb += WA_KB) {
        fv8 s[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int key = min(kb + j * 16 + l16, N - 1);
            const __half* kp = k + (size_t)key * D + h * VA_HD;
            fv8 a = fv8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) a = vit_wmma(qf[ks], ld_hv(kp + ks * 16, hi), a);
            s[j] = vit_wmma(qf[4], ld_hv_half(kp + 64, hi), a);
        }
        float corr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float mx = -1e30f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool kvalid = kb + j * 16 + l16 < N;
                s[j][i] = kvalid ? s[j][i] * scale : -1e30f;
                mx = fmaxf(mx, s[j][i]);
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 16));
            const float nm = fmaxf(mrun[i], mx);
            corr[i] = __expf(mrun[i] - nm);
            mrun[i] = nm;
            float ps = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool kvalid = kb + j * 16 + l16 < N;
                const float p = kvalid ? __expf(s[j][i] - nm) : 0.f;
                ps += p;
                sp[vit_acc_m(i, hi) * WA_PP + j * 16 + l16] = __float2half(p);
            }
            lpart[i] = lpart[i] * corr[i] + ps;
        }
#pragma unroll
        for (int t = 0; t < 5; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) oacc[t][i] *= corr[i];
        hy4_sync_lds();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const hv_t pf = ld_hv(sp + l16 * WA_PP + j * 16, hi);
#pragma unroll
            for (int t = 0; t < 5; ++t) {
                const int d = min(t * 16 + l16, VA_HD - 1);
                oacc[t] = vit_wmma(pf, ld_hv(vt + (size_t)(h * VA_HD + d) * ldv + kb + j * 16, hi), oacc[t]);
            }
        }
        hy4_sync_lds();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float l = lpart[i];
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) l += __shfl_xor(l, off, 16);
        const int qi = q0 + vit_acc_m(i, hi);
        if (qi >= N) continue;
        const float il = 1.0f / l;
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const int d = t * 16 + l16;
            if (d < VA_HD) o[(size_t)qi * D + h * VA_HD + d] = __float2half(oacc[t][i] * il);
        }
    }
}
#endif  // HY4_MMA && !HY4_CUDA

// ---- attention: full (non-causal) softmax attention per head, head_dim 72, fp16 q/k/v [N][1152] (head h at columns 72h..), online softmax in fp32, one thread per query ------------------
constexpr int VA_KT = 32;
__global__ void __launch_bounds__(64) k_attn(const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v, __half* __restrict__ o, int N, int D, float scale) {
    __shared__ float Ks[VA_KT][VA_HD], Vs[VA_KT][VA_HD];
    const int h = blockIdx.y;
    const int qi = blockIdx.x * 64 + threadIdx.x;
    float qv[VA_HD], acc[VA_HD];
    if (qi < N) {
        const __half* qp = q + (size_t)qi * D + h * VA_HD;
#pragma unroll
        for (int d = 0; d < VA_HD; ++d) qv[d] = __half2float(qp[d]) * scale;
    } else {
#pragma unroll
        for (int d = 0; d < VA_HD; ++d) qv[d] = 0.f;
    }
#pragma unroll
    for (int d = 0; d < VA_HD; ++d) acc[d] = 0.f;
    float mrun = -3.0e38f, lrun = 0.f;
    for (int kb = 0; kb < N; kb += VA_KT) {
        __syncthreads();
        for (int e = threadIdx.x; e < VA_KT * VA_HD; e += 64) {
            const int j = e / VA_HD, d = e % VA_HD;
            const int kj = kb + j;
            Ks[j][d] = kj < N ? __half2float(k[(size_t)kj * D + h * VA_HD + d]) : 0.f;
            Vs[j][d] = kj < N ? __half2float(v[(size_t)kj * D + h * VA_HD + d]) : 0.f;
        }
        __syncthreads();
        const int nj = min(VA_KT, N - kb);
        float s[VA_KT];
        float mt = -3.0e38f;
#pragma unroll
        for (int j = 0; j < VA_KT; ++j) {
            float dot = 0.f;
#pragma unroll
            for (int d = 0; d < VA_HD; ++d) dot = fmaf(qv[d], Ks[j][d], dot);
            s[j] = j < nj ? dot : -3.0e38f;
            mt = fmaxf(mt, s[j]);
        }
        const float mn = fmaxf(mrun, mt);
        const float corr = __expf(mrun - mn);
        lrun *= corr;
#pragma unroll
        for (int d = 0; d < VA_HD; ++d) acc[d] *= corr;
#pragma unroll
        for (int j = 0; j < VA_KT; ++j) {
            const float p = j < nj ? __expf(s[j] - mn) : 0.f;
            lrun += p;
#pragma unroll
            for (int d = 0; d < VA_HD; ++d) acc[d] = fmaf(p, Vs[j][d], acc[d]);
        }
        mrun = mn;
    }
    if (qi < N) {
        const float inv = 1.0f / lrun;
        __half* op = o + (size_t)qi * D + h * VA_HD;
#pragma unroll
        for (int d = 0; d < VA_HD; ++d) op[d] = __float2half(acc[d] * inv);
    }
}

__global__ void k_fill_newline(float* __restrict__ rows, const float* __restrict__ nl, int oh, int ow, int D) {
    const int r = blockIdx.x;  // merged row
    float* dst = rows + ((size_t)r * (ow + 1) + ow) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) dst[i] = nl[i];
}

}  // namespace vit

// =============================================================================================================================================================================
struct VisionEncoder {
    VisionShardInfo vi;
    uint8_t* d_w = nullptr;  // Section 1 of the shard
    bool loaded = false;
    double t_load = 0;
    // debug: dump the residual stream after the embedding and every layer (fp32 [N][1152]) as .npy into dump_dir
    bool dump = false;
    std::string dump_dir = ".";
    bool gelu_tanh_ffn = true;
    bool use_wmma = false;  // gfx11 matrix-core GEMM / attention (set in load(); HY4_VIT_NOWMMA=1 forces non-WMMA)
    bool use_fdot2 = false; // Wave32 fdot2 GEMM / attention for RDNA2 & non-WMMA HIP (HY4_VIT_PLAIN=1 forces plain)
    const std::atomic<bool>* cancel = nullptr;  // set by serve: STOP during the encode ends it at the next layer

    const __half* wh(const std::string& n) const { return reinterpret_cast<const __half*>(d_w + vi.get(n).off); }
    const float* wf(const std::string& n) const { return reinterpret_cast<const float*>(d_w + vi.get(n).off); }

    bool load(const std::string& path, std::string& err) {
        if (loaded) return true;
        if (!vision_probe(path, vi, err)) return false;
        gelu_tanh_ffn = vi.c.gelu_tanh_ffn;
        if (const char* g = getenv("HY4_VIT_GELU")) gelu_tanh_ffn = std::string(g) != "erf";
        const uint64_t bytes = (vi.sec1_len + 4095) & ~(uint64_t)4095;
        hipError_t e = hipMalloc((void**)&d_w, bytes);
        if (e != hipSuccess) {
            (void)hipGetLastError();
            err = "cannot allocate " + std::to_string(bytes / 1048576) + " MiB for the vision weights";
            return false;
        }
        stream_to_device(path, vi.sec1_off, vi.sec1_len, d_w, &t_load, "vision");
#if HY4_MMA && !HY4_CUDA
        {  // matrix-core kernels on gfx11 (RDNA3 / 3.5) and gfx12 (RDNA4: RX 9060/9070, R9700); everything else takes fdot2 / plain kernels
            hipDeviceProp_t dp;
            HCK(hipGetDeviceProperties(&dp, 0));
            const std::string arch(dp.gcnArchName);
            use_wmma = (arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0) && !getenv("HY4_VIT_NOWMMA");
        }
#endif
#if !HY4_CUDA
        use_fdot2 = !use_wmma && !getenv("HY4_VIT_PLAIN");
#endif
        loaded = true;
        fprintf(stderr, "[vision] shard loaded to the GPU: %.2f GB in %.2f s (%s)\n", vi.sec1_len / 1e9, t_load,
                use_wmma ? "wmma" : (use_fdot2 ? "fdot2" : "plain"));
        return true;
    }
    void unload() {
        if (d_w) HCK(hipFree(d_w));
        d_w = nullptr;
        loaded = false;
    }

    void gemm(hipStream_t st, int epi, const __half* X, const __half* W, const float* bias, __half* Yh, float* Yf, const float* pe, int M, int N, int K, int ow = 1) {
        dim3 grid((N + 63) / 64, (M + 63) / 64), blk(256);
#if HY4_MMA && !HY4_CUDA
        if (use_wmma) {  // gfx11 matrix cores
            dim3 bw(128);
            switch (epi) {
                case vit::EPI_H: hipLaunchKernelGGL((vit::k_gemm_w<vit::EPI_H>), grid, bw, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
                case vit::EPI_VT: hipLaunchKernelGGL((vit::k_gemm_w<vit::EPI_VT>), grid, bw, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
                case vit::EPI_GELU_TANH: hipLaunchKernelGGL((vit::k_gemm_w<vit::EPI_GELU_TANH>), grid, bw, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
                case vit::EPI_GELU_ERF: hipLaunchKernelGGL((vit::k_gemm_w<vit::EPI_GELU_ERF>), grid, bw, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
                case vit::EPI_RES_ADD: hipLaunchKernelGGL((vit::k_gemm_w<vit::EPI_RES_ADD>), grid, bw, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
                case vit::EPI_RES_SET_POS: hipLaunchKernelGGL((vit::k_gemm_w<vit::EPI_RES_SET_POS>), grid, bw, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
                default: hipLaunchKernelGGL((vit::k_gemm_w<vit::EPI_F32_ROWMAP>), grid, bw, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            }
            return;
        }
#endif
#if !HY4_CUDA
        if (use_fdot2) {
            switch (epi) {
                case vit::EPI_H: hipLaunchKernelGGL((vit::k_gemm_fdot2<vit::EPI_H>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
                case vit::EPI_GELU_TANH: hipLaunchKernelGGL((vit::k_gemm_fdot2<vit::EPI_GELU_TANH>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
                case vit::EPI_GELU_ERF: hipLaunchKernelGGL((vit::k_gemm_fdot2<vit::EPI_GELU_ERF>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
                case vit::EPI_RES_ADD: hipLaunchKernelGGL((vit::k_gemm_fdot2<vit::EPI_RES_ADD>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
                case vit::EPI_RES_SET_POS: hipLaunchKernelGGL((vit::k_gemm_fdot2<vit::EPI_RES_SET_POS>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
                default: hipLaunchKernelGGL((vit::k_gemm_fdot2<vit::EPI_F32_ROWMAP>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            }
            return;
        }
#endif
        switch (epi) {
            case vit::EPI_H: hipLaunchKernelGGL((vit::k_gemm<vit::EPI_H>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            case vit::EPI_GELU_TANH: hipLaunchKernelGGL((vit::k_gemm<vit::EPI_GELU_TANH>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            case vit::EPI_GELU_ERF: hipLaunchKernelGGL((vit::k_gemm<vit::EPI_GELU_ERF>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            case vit::EPI_RES_ADD: hipLaunchKernelGGL((vit::k_gemm<vit::EPI_RES_ADD>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            case vit::EPI_RES_SET_POS: hipLaunchKernelGGL((vit::k_gemm<vit::EPI_RES_SET_POS>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
            default: hipLaunchKernelGGL((vit::k_gemm<vit::EPI_F32_ROWMAP>), grid, blk, 0, st, X, W, bias, Yh, Yf, pe, M, N, K, ow); break;
        }
    }

    void dump_x(hipStream_t st, const std::string& name, const float* dx, int N, int D) {
        HCK(hipStreamSynchronize(st));
        std::vector<float> h((size_t)N * D);
        HCK(hipMemcpy(h.data(), dx, h.size() * 4, hipMemcpyDeviceToHost));
        save_npy_f32_vis(dump_dir + "/" + name + ".npy", h.data(), {N, D});
    }
    static void save_npy_f32_vis(const std::string& path, const float* d, const std::vector<int64_t>& shape) {
        std::string sh = "(";
        for (size_t i = 0; i < shape.size(); ++i) sh += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
        sh += ")";
        std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
        size_t tot = 10 + hdr.size() + 1, pad = (64 - tot % 64) % 64;
        hdr += std::string(pad, ' ') + "\n";
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return;
        fwrite("\x93NUMPY", 1, 6, f);
        unsigned char ver[2] = {1, 0};
        fwrite(ver, 1, 2, f);
        uint16_t hl = (uint16_t)hdr.size();
        fwrite(&hl, 2, 1, f);
        fwrite(hdr.data(), 1, hdr.size(), f);
        size_t n = 1;
        for (auto s : shape) n *= (size_t)s;
        fwrite(d, 4, n, f);
        fclose(f);
    }

    // rgb: host RGB u8 [h][w][3] (h, w multiples of 32). rows_out: device fp32 [(h/32)*(w/32+1)][out_hidden], filled on stream st. Allocates and frees its own workspace.
    bool encode(hipStream_t st, const uint8_t* rgb, int h, int w, float* rows_out, std::string& err) {
        if (!loaded) {
            err = "vision shard not loaded";
            return false;
        }
        const VitCfg& c = vi.c;
        const int P = c.patch, D = c.hidden, NH = c.heads, FF = c.ffn, OD = c.out_hidden;
        if (h % (P * c.merge) || w % (P * c.merge) || h < 32 || w < 32) {
            err = "picture size must be a multiple of 32";
            return false;
        }
        const int gh = h / P, gw = w / P, N = gh * gw, oh = gh / 2, ow = gw / 2, M2 = oh * ow;
        if (D / NH != vit::VA_HD) {
            err = "unsupported head dim";
            return false;
        }
        uint8_t* d_rgb = nullptr;
        __half *d_pix = nullptr, *d_h = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_ff = nullptr, *d_A = nullptr, *d_mid = nullptr;
        float *d_x = nullptr, *d_pe = nullptr;
        const size_t bN = (size_t)N * D;
        auto free_ws = [&]() {
            for (void* q : {(void*)d_rgb, (void*)d_pix, (void*)d_x, (void*)d_pe, (void*)d_h, (void*)d_q, (void*)d_k, (void*)d_v, (void*)d_ff, (void*)d_A, (void*)d_mid})
                if (q) HCK(hipFree(q));
        };
        auto fail = [&](const char* what) {
            free_ws();
            (void)hipGetLastError();
            err = std::string("out of device memory for the vision workspace (") + what + ")";
            return false;
        };
        if (hipMalloc((void**)&d_rgb, (size_t)h * w * 3) != hipSuccess) return fail("rgb");
        if (hipMalloc((void**)&d_pix, (size_t)N * 768 * 2) != hipSuccess) return fail("pixels");
        if (hipMalloc((void**)&d_x, bN * 4) != hipSuccess) return fail("x");
        if (hipMalloc((void**)&d_pe, bN * 4) != hipSuccess) return fail("pe");
        if (hipMalloc((void**)&d_h, bN * 2) != hipSuccess) return fail("h");
        if (hipMalloc((void**)&d_q, bN * 2) != hipSuccess) return fail("q");
        if (hipMalloc((void**)&d_k, bN * 2) != hipSuccess) return fail("k");
        const int ldv = (N + 63) & ~63;  // key pitch of the transposed V of the matrix-core attention (a multiple of its 64-key block)
        if (hipMalloc((void**)&d_v, use_wmma ? (size_t)D * ldv * 2 : bN * 2) != hipSuccess) return fail("v");
        if (use_wmma) HCK(hipMemsetAsync(d_v, 0, (size_t)D * ldv * 2, st));
        if (hipMalloc((void**)&d_ff, (size_t)N * FF * 2) != hipSuccess) return fail("ffn");
        if (hipMalloc((void**)&d_A, (size_t)M2 * D * 4 * 2) != hipSuccess) return fail("merge");
        if (hipMalloc((void**)&d_mid, (size_t)M2 * D * 4 * 2) != hipSuccess) return fail("mid");
        HCK(hipMemcpyAsync(d_rgb, rgb, (size_t)h * w * 3, hipMemcpyHostToDevice, st));
        hipLaunchKernelGGL(vit::k_patchify, dim3(N), dim3(256), 0, st, d_rgb, d_pix, w, gw);
        // position grid resampled to gh x gw
        const float sy = (float)(1.0 / (((double)gh + 0.1) / (double)c.pos_edge)), sx = (float)(1.0 / (((double)gw + 0.1) / (double)c.pos_edge));
        hipLaunchKernelGGL(vit::k_posemb, dim3(N), dim3(256), 0, st, wh("vit.embeddings.position_embedding.weight"), d_pe, gh, gw, c.pos_edge, sy, sx, D);
        gemm(st, vit::EPI_RES_SET_POS, d_pix, wh("vit.embeddings.patch_embedding.weight"), wf("vit.embeddings.patch_embedding.bias"), nullptr, d_x, d_pe, N, D, 768);
        if (dump) dump_x(st, "emb0", d_x, N, D);
        const float scale = 1.0f / sqrtf((float)(D / NH));
        // HY4_VIT_PROF=1: wall time per stage (synchronises after every stage: only for measuring)
        static const bool prof = getenv("HY4_VIT_PROF") != nullptr;
        std::map<std::string, double> pacc;
        double tlap = now_s();
        auto lap = [&](const char* what) {
            if (!prof) return;
            HCK(hipStreamSynchronize(st));
            const double t = now_s();
            pacc[what] += t - tlap;
            tlap = t;
        };
        for (int il = 0; il < c.layers; ++il) {
            if (cancel && cancel->load()) {
                err = "cancelled";
                HCK(hipStreamSynchronize(st));
                free_ws();
                return false;
            }
            const std::string p = "vit.layers." + std::to_string(il) + ".";
            hipLaunchKernelGGL(vit::k_ln, dim3(N), dim3(256), 0, st, d_x, wf(p + "input_layernorm.weight"), wf(p + "input_layernorm.bias"), d_h, D, c.eps);
            lap("ln");
            gemm(st, vit::EPI_H, d_h, wh(p + "self_attn.q_proj.weight"), wf(p + "self_attn.q_proj.bias"), d_q, nullptr, nullptr, N, D, D);
            gemm(st, vit::EPI_H, d_h, wh(p + "self_attn.k_proj.weight"), wf(p + "self_attn.k_proj.bias"), d_k, nullptr, nullptr, N, D, D);
            if (use_wmma) {
#if HY4_MMA && !HY4_CUDA
                gemm(st, vit::EPI_VT, d_h, wh(p + "self_attn.v_proj.weight"), wf(p + "self_attn.v_proj.bias"), d_v, nullptr, nullptr, N, D, D, ldv);
                lap("qkv gemm");
                hipLaunchKernelGGL(vit::k_attn_w, dim3((N + 63) / 64, NH), dim3(128), 0, st, d_q, d_k, d_v, d_h, N, D, ldv, scale);
#endif
            } else {
                gemm(st, vit::EPI_H, d_h, wh(p + "self_attn.v_proj.weight"), wf(p + "self_attn.v_proj.bias"), d_v, nullptr, nullptr, N, D, D);
                lap("qkv gemm");
#if !HY4_CUDA
                if (use_fdot2)
                    hipLaunchKernelGGL(vit::k_attn_fdot2, dim3((N + 31) / 32, NH), dim3(128), 0, st, d_q, d_k, d_v, d_h, N, D, scale);
                else
#endif
                    hipLaunchKernelGGL(vit::k_attn, dim3((N + 63) / 64, NH), dim3(64), 0, st, d_q, d_k, d_v, d_h, N, D, scale);
            }
            lap("attention");
            gemm(st, vit::EPI_RES_ADD, d_h, wh(p + "self_attn.o_proj.weight"), wf(p + "self_attn.o_proj.bias"), nullptr, d_x, nullptr, N, D, D);
            lap("o_proj gemm");
            hipLaunchKernelGGL(vit::k_ln, dim3(N), dim3(256), 0, st, d_x, wf(p + "post_attention_layernorm.weight"), wf(p + "post_attention_layernorm.bias"), d_h, D, c.eps);
            gemm(st, gelu_tanh_ffn ? vit::EPI_GELU_TANH : vit::EPI_GELU_ERF, d_h, wh(p + "mlp.dense_h_to_4h.weight"), wf(p + "mlp.dense_h_to_4h.bias"), d_ff, nullptr, nullptr, N, FF, D);
            lap("fc1 gemm+ln");
            gemm(st, vit::EPI_RES_ADD, d_ff, wh(p + "mlp.dense_4h_to_h.weight"), wf(p + "mlp.dense_4h_to_h.bias"), nullptr, d_x, nullptr, N, D, FF);
            lap("fc2 gemm");
            if (dump) dump_x(st, "L" + std::to_string(il), d_x, N, D);
        }
        // adapter
        hipLaunchKernelGGL(vit::k_rms_merge, dim3(N), dim3(256), 0, st, d_x, wf("vit.perceive.norm.weight"), d_A, gw, D, c.eps);
        gemm(st, vit::EPI_GELU_ERF, d_A, wh("vit.perceive.proj.0.weight"), wf("vit.perceive.proj.0.bias"), d_mid, nullptr, nullptr, M2, D * 4, D * 4);
        gemm(st, vit::EPI_F32_ROWMAP, d_mid, wh("vit.perceive.proj.2.weight"), wf("vit.perceive.proj.2.bias"), nullptr, rows_out, nullptr, M2, OD, D * 4, ow);
        hipLaunchKernelGGL(vit::k_fill_newline, dim3(oh), dim3(256), 0, st, rows_out, wf("vit.perceive.image_newline"), oh, ow, OD);
        HCK(hipStreamSynchronize(st));
        if (prof) {
            lap("adapter");
            std::string line = "[vit prof] ";
            for (auto& kv : pacc) line += kv.first + " " + std::to_string(kv.second).substr(0, 5) + " s, ";
            fprintf(stderr, "%s\n", line.c_str());
        }
        free_ws();
        return true;
    }
};

// =============================================================================================================================================================================
// LM side: what a prefill with pictures needs. Everything here is reached from the runner through a few clearly marked hooks ("HY4-VISION hook"); a request without pictures has no VisionCtx
// and takes none of this.
namespace vit {

// image rows -> the residual stream: x [T][4][3072] fp32 = the 4 iHC streams all start as the embedding (what k_embed_hc does for a text token)
__global__ void k_scatter_rows(float* __restrict__ x, const float* __restrict__ rows, const int32_t* __restrict__ map, int T) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const int r = map[t];
    if (r < 0) return;
    for (int d = threadIdx.x; d < HID; d += blockDim.x) {
        const float v = rows[(size_t)r * HID + d];
#pragma unroll
        for (int h = 0; h < NHC; ++h) x[((size_t)t * NHC + h) * HID + d] = v;
    }
}

struct RopeDev {
    const int32_t* tab;  // [3][stride]: (t, h, w) position of token index < tail_from
    int stride, tail_from, delta;
};
__device__ __forceinline__ int rope_pos(const RopeDev& r, int idx, int axis) { return idx < r.tail_from ? r.tab[(size_t)axis * r.stride + idx] : idx + r.delta; }
// same arithmetic as mla_attn.hip.cpp rope_pair (interleaved pair i of 32, fp32)
__device__ __forceinline__ void rope_pair_v(float x0, float x1, int pos, int i, float theta, float& o0, float& o1) {
    const float freq = 1.0f / powf(theta, (float)(2 * i) / 64.0f);
    const float ang = (float)pos * freq;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    o0 = x0 * cs - x1 * sn;
    o1 = x0 * sn + x1 * cs;
}

// MLA query: qa[t][h][512 + 2i ..] = rope(q_b[t][h][192 + 2i ..], mrope position of token p0 + t)
__global__ void k_fix_q(const float* __restrict__ q_b, __half* __restrict__ qa, int NH, int p0, float theta, RopeDev rd) {
    const int t = blockIdx.x / NH, h = blockIdx.x % NH, i = threadIdx.x;  // 32 threads
    const float* qp = q_b + ((size_t)t * NH + h) * 256 + 192 + 2 * i;
    float o0, o1;
    rope_pair_v(qp[0], qp[1], rope_pos(rd, p0 + t, mrope_axis(i)), i, theta, o0, o1);
    *reinterpret_cast<__half2*>(qa + ((size_t)t * NH + h) * 576 + 512 + 2 * i) = __floats2half2_rn(o0, o1);
}

// MLA key: cache[slot][512 + 2i ..] = rope(kv_a[t][512 + 2i ..], mrope position of token p0 + t); slot = (p0 + t) % ring, or p0 + t without a ring
__global__ void k_fix_k(const float* __restrict__ kv_a, __half* __restrict__ cache, int p0, int ring, float theta, RopeDev rd) {
    const int t = blockIdx.x, i = threadIdx.x;  // 32 threads
    const int pos = p0 + t;
    const int slot = ring > 0 ? pos % ring : pos;
    float o0, o1;
    rope_pair_v(kv_a[(size_t)t * 576 + 512 + 2 * i], kv_a[(size_t)t * 576 + 512 + 2 * i + 1], rope_pos(rd, pos, mrope_axis(i)), i, theta, o0, o1);
    *reinterpret_cast<__half2*>(cache + (size_t)slot * 576 + 512 + 2 * i) = __floats2half2_rn(o0, o1);
}

// indexer query: out[t][h][d] = iq[t][h][d], dims 64..127 roped at the mrope position of token p0 + t (instead of idx::k_idx_q_rope)
__global__ void k_fix_iq(const float* __restrict__ iq, __half* __restrict__ out, int p0, float theta, RopeDev rd) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;  // 128 threads
    const float* q = iq + (size_t)t * 2048 + h * 128;
    float v = q[d];
    if (d >= 64) {
        const int i = (d - 64) >> 1;
        float o0, o1;
        rope_pair_v(q[64 + 2 * i], q[64 + 2 * i + 1], rope_pos(rd, p0 + t, mrope_axis(i)), i, theta, o0, o1);
        v = (d & 1) ? o1 : o0;
    }
    out[(size_t)t * 2048 + h * 128 + d] = __float2half(v);
}

// pooled indexer keys were roped at 4g (classic); rotate pair i of dims 64..127 by (mrope position of token 4g - 4g) * freq_i so they sit at the picture-aware position
__global__ void k_fix_pool(__half* __restrict__ pooled, int g_lo, float theta, RopeDev rd) {
    const int g = g_lo + blockIdx.x, i = threadIdx.x;  // 32 threads
    const int pn = rope_pos(rd, 4 * g, mrope_axis(i));
    if (pn == 4 * g) return;
    const float freq = 1.0f / powf(theta, (float)(2 * i) / 64.0f);
    const float ang = (float)(pn - 4 * g) * freq;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    __half* p = pooled + (size_t)g * 128 + 64 + 2 * i;
    const float x0 = __half2float(p[0]), x1 = __half2float(p[1]);
    p[0] = __float2half(x0 * cs - x1 * sn);
    p[1] = __float2half(x0 * sn + x1 * cs);
}

}  // namespace vit

// Per-request picture state: image rows on the device, the spans of the prompt they fill, the (t, h, w) position table. Owned by serve / main, pointed to by Runner::vis.
struct VisionCtx {
    std::vector<ImageSpan> spans;
    RopePos pos;
    std::vector<uint8_t> mask;  // per token index of the prompt: 1 where the token is an image row (Engram contribution is off there)
    float* d_rows = nullptr;    // [R][3072] fp32
    int n_rows = 0;
    int32_t* d_tab = nullptr;   // [3][tab_n]
    int tab_n = 0;
    int32_t* d_map = nullptr;
    int map_cap = 0;
    double t_encode = 0;
    int image_token = 120120;

    ~VisionCtx() { release(); }
    void release() {
        if (d_rows) hipFree(d_rows);
        if (d_tab) hipFree(d_tab);
        if (d_map) hipFree(d_map);
        d_rows = nullptr;
        d_tab = nullptr;
        d_map = nullptr;
        n_rows = tab_n = map_cap = 0;
        spans.clear();
        mask.clear();
        pos = RopePos();
    }
    bool active() const { return !spans.empty(); }

    // ids = the whole prompt, recs = its pictures in order. Encodes every picture (enc must be loaded) into d_rows.
    bool setup(VisionEncoder& enc, hipStream_t st, const std::vector<int32_t>& ids, const std::vector<ImageRec>& recs, int max_chunk, std::string& err) {
        release();
        image_token = enc.vi.c.image_token;
        if (!build_image_spans(ids, image_token, recs, spans, pos, err)) return false;
        int total = 0;
        for (const ImageSpan& s : spans) total += s.n;
        n_rows = total;
        if (hipMalloc((void**)&d_rows, (size_t)total * HID * 4) != hipSuccess) {
            (void)hipGetLastError();
            d_rows = nullptr;
            err = "out of device memory for the image rows";
            return false;
        }
        double t0 = now_s();
        for (size_t k = 0; k < recs.size(); ++k) {
            if (!enc.encode(st, recs[k].rgb.data(), recs[k].h, recs[k].w, d_rows + (size_t)spans[k].row0 * HID, err)) return false;
        }
        t_encode = now_s() - t0;
        mask.assign(ids.size(), 0);
        for (const ImageSpan& s : spans)
            for (int j = 0; j < s.n; ++j) mask[s.start + j] = 1;
        tab_n = (int)ids.size();
        HCK(hipMalloc((void**)&d_tab, (size_t)3 * tab_n * 4));
        for (int a = 0; a < 3; ++a) HCK(hipMemcpy(d_tab + (size_t)a * tab_n, pos.p[a].data(), (size_t)tab_n * 4, hipMemcpyHostToDevice));
        map_cap = max_chunk;
        HCK(hipMalloc((void**)&d_map, (size_t)map_cap * 4));
        return true;
    }

    vit::RopeDev rd() const { return vit::RopeDev{d_tab, tab_n, tab_n, pos.delta}; }

    // hook 1: after the text embedding of chunk [p0, p0 + T): overwrite the image positions with the image rows
    void embed(hipStream_t st, float* x, int p0, int T) {
        if (!active()) return;
        std::vector<int32_t> map(T, -1);
        bool any = false;
        for (const ImageSpan& s : spans) {
            const int a = std::max(p0, s.start), b = std::min(p0 + T, s.start + s.n);
            for (int i = a; i < b; ++i) {
                map[i - p0] = s.row0 + (i - s.start);
                any = true;
            }
        }
        if (!any) return;
        HCK(hipMemcpyAsync(d_map, map.data(), (size_t)T * 4, hipMemcpyHostToDevice, st));
        hipLaunchKernelGGL(vit::k_scatter_rows, dim3(T), dim3(256), 0, st, x, (const float*)d_rows, (const int32_t*)d_map, T);
    }
    // hook 2: positions of the chunk are not index + 0 any more (a picture is in the context): redo the rope of q (qa rope part) and of the new keys with the 3D positions
    void fix_q(hipStream_t st, const float* q_b, __half* qa, int M, int NH, int p0, float theta) {
        hipLaunchKernelGGL(vit::k_fix_q, dim3(M * NH), dim3(32), 0, st, q_b, qa, NH, p0, theta, rd());
    }
    void fix_k(hipStream_t st, const float* kv_a, __half* cache, int M, int p0, int ring, float theta) {
        hipLaunchKernelGGL(vit::k_fix_k, dim3(M), dim3(32), 0, st, kv_a, cache, p0, ring, theta, rd());
    }
    void iq_rope(hipStream_t st, const float* iq, __half* out, int T, int p0, float theta) {
        hipLaunchKernelGGL(vit::k_fix_iq, dim3(T, 16), dim3(128), 0, st, iq, out, p0, theta, rd());
    }
    void pool_fix(hipStream_t st, __half* pooled, int g_lo, int n, float theta) { hipLaunchKernelGGL(vit::k_fix_pool, dim3(n), dim3(32), 0, st, pooled, g_lo, theta, rd()); }
    // Engram: the contribution is masked at image positions (HY_V4_FLASH_ENGRAM_DEAD: every column invalid)
    void mask_engram(uint8_t* valid /*[T][16]*/, int p0, int T) const {
        for (int t = 0; t < T; ++t)
            if (p0 + t < (int)mask.size() && mask[p0 + t]) memset(valid + (size_t)t * 16, 0, 16);
    }
    // positions of the generated tokens follow index + delta (RopeDev.tail_from / delta); true when the 3D rope path is needed for this chunk
    bool needs_fix(int p0, int T) const { return active() && (pos.delta != 0 || p0 < tab_n); }
};

}  // namespace hy4

