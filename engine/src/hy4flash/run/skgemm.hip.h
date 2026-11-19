// skgemm.hip.h - HIP linear-algebra primitives for the Hy4-Flash runner on gfx1151 (wave32, RDNA 3.5)
//
// Phase 1 (correctness): straightforward wave-per-row kernels that dequantize S_K / Q8ROW weights on the fly.
//   Y[m][n] = sum_k W[n][k] * X[m][k]   (X: __half [M][K], Y: float [M][N], fp32 accumulation)
//   w = (q - H) * D * (u/255)  per 64-weight sub-group, H = 1.5/3.5/7.5/15.5/31.5 (S2..S6), 0 for S8 (q signed int8)
// Header-only (one translation unit per program).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace hy4 {

enum Fmt { FMT_S2 = 2, FMT_S3 = 3, FMT_S4 = 4, FMT_S5 = 5, FMT_S6 = 6, FMT_S8 = 8, FMT_Q8ROW = 100, FMT_F32 = 101 };

constexpr int sk_sb_bytes(int f) {
    return f == FMT_S2 ? 70 : f == FMT_S3 ? 102 : f == FMT_S4 ? 134 : f == FMT_S5 ? 166 : f == FMT_S6 ? 198 : f == FMT_S8 ? 262 : 0;
}
// bytes of a weight [N rows][K cols] in the .strata trunk/expert wire format
constexpr size_t weight_bytes(int fmt, int N, int K) {
    return fmt == FMT_Q8ROW ? (size_t)N * 4 + (size_t)N * K
         : fmt == FMT_F32 ? (size_t)N * K * 4
         : (size_t)N * (K / 256) * sk_sb_bytes(fmt);
}

namespace skd {

__device__ __forceinline__ uint16_t ld16(const uint8_t* p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
__device__ __forceinline__ float h2f_bits(uint16_t b) { return __half2float(__ushort_as_half(b)); }

template <int FMT> constexpr float sk_half() {
    return FMT == FMT_S2 ? 1.5f : FMT == FMT_S3 ? 3.5f : FMT == FMT_S4 ? 7.5f : FMT == FMT_S5 ? 15.5f : FMT == FMT_S6 ? 31.5f : 0.0f;
}

// quantized integer of weight i (0..255) of a super-block payload `pl` (blk + 6)
template <int FMT> __device__ __forceinline__ int sk_q(const uint8_t* pl, int i) {
    if (FMT == FMT_S2) return (pl[i >> 2] >> ((i & 3) * 2)) & 3;
    if (FMT == FMT_S3) return ((pl[i >> 2] >> ((i & 3) * 2)) & 3) | (((pl[64 + (i >> 3)] >> (i & 7)) & 1) << 2);
    if (FMT == FMT_S4) return (pl[i >> 1] >> ((i & 1) * 4)) & 15;
    if (FMT == FMT_S5) return ((pl[i >> 1] >> ((i & 1) * 4)) & 15) | (((pl[128 + (i >> 3)] >> (i & 7)) & 1) << 4);
    if (FMT == FMT_S6) return ((pl[i >> 1] >> ((i & 1) * 4)) & 15) | (((pl[128 + (i >> 2)] >> ((i & 3) * 2)) & 3) << 4);
    return (int)(int8_t)pl[i];  // S8
}

// one weight of super-block `blk` at index i (0..255)
template <int FMT> __device__ __forceinline__ float sk_weight(const uint8_t* blk, int i) {
    const float D = h2f_bits(ld16(blk));
    const int g = i >> 6;
    const float eff = D * ((float)blk[2 + g] / 255.0f);
    return ((float)sk_q<FMT>(blk + 6, i) - sk_half<FMT>()) * eff;
}

// ---- Phase 1 GEMV/GEMM: one wave computes one output row for up to MT tokens per pass
template <int FMT, int MT>
__global__ void __launch_bounds__(256) k_linear_sk(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X,
                                                    float* __restrict__ Y, int M) {
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= N) return;
    constexpr int SB = sk_sb_bytes(FMT);
    const int nsb = K >> 8;
    const uint8_t* rp = W + (size_t)row * nsb * SB;
    for (int m0 = 0; m0 < M; m0 += MT) {
        float acc[MT];
#pragma unroll
        for (int i = 0; i < MT; ++i) acc[i] = 0.f;
        for (int unit = lane; unit < nsb * 32; unit += 32) {
            const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
            const uint8_t* blk = rp + (size_t)sbi * SB;
            const float D = h2f_bits(ld16(blk));
            const float eff = D * ((float)blk[2 + g] / 255.0f);
            float w[8];
#pragma unroll
            for (int t = 0; t < 8; ++t) w[t] = ((float)sk_q<FMT>(blk + 6, j + t) - sk_half<FMT>()) * eff;
#pragma unroll
            for (int mm = 0; mm < MT; ++mm) {
                if (m0 + mm < M) {
                    const __half* xp = X + (size_t)(m0 + mm) * K + sbi * 256 + j;
                    float a = 0.f;
#pragma unroll
                    for (int t = 0; t < 8; ++t) a += w[t] * __half2float(xp[t]);
                    acc[mm] += a;
                }
            }
        }
#pragma unroll
        for (int mm = 0; mm < MT; ++mm) {
            float v = acc[mm];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0 && m0 + mm < M) Y[(size_t)(m0 + mm) * N + row] = v;
        }
    }
}

// Q8ROW: [N f32 scales][N*K int8]
template <int MT>
__global__ void __launch_bounds__(256) k_linear_q8row(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X,
                                                       float* __restrict__ Y, int M) {
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= N) return;
    const float scale = ((const float*)W)[row];
    const int8_t* q = (const int8_t*)(W + (size_t)N * 4) + (size_t)row * K;
    for (int m0 = 0; m0 < M; m0 += MT) {
        float acc[MT];
#pragma unroll
        for (int i = 0; i < MT; ++i) acc[i] = 0.f;
        for (int k = lane; k < K; k += 32) {
            const float w = (float)q[k];
#pragma unroll
            for (int mm = 0; mm < MT; ++mm)
                if (m0 + mm < M) acc[mm] += w * __half2float(X[(size_t)(m0 + mm) * K + k]);
        }
#pragma unroll
        for (int mm = 0; mm < MT; ++mm) {
            float v = acc[mm];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0 && m0 + mm < M) Y[(size_t)(m0 + mm) * N + row] = v * scale;
        }
    }
}

// F32 weights, F32 activations
template <int MT>
__global__ void __launch_bounds__(256) k_linear_f32(const float* __restrict__ W, int N, int K, const float* __restrict__ X,
                                                     float* __restrict__ Y, int M) {
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= N) return;
    const float* wp = W + (size_t)row * K;
    for (int m0 = 0; m0 < M; m0 += MT) {
        float acc[MT];
#pragma unroll
        for (int i = 0; i < MT; ++i) acc[i] = 0.f;
        for (int k = lane; k < K; k += 32) {
            const float w = wp[k];
#pragma unroll
            for (int mm = 0; mm < MT; ++mm)
                if (m0 + mm < M) acc[mm] += w * X[(size_t)(m0 + mm) * K + k];
        }
#pragma unroll
        for (int mm = 0; mm < MT; ++mm) {
            float v = acc[mm];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0 && m0 + mm < M) Y[(size_t)(m0 + mm) * N + row] = v;
        }
    }
}

template <int FMT>
__global__ void k_embed_sk(const uint8_t* __restrict__ W, int K, const int32_t* __restrict__ ids, float* __restrict__ out) {
    const int t = blockIdx.x;
    const int64_t id = ids[t];
    constexpr int SB = sk_sb_bytes(FMT);
    const uint8_t* rp = W + (size_t)id * (K >> 8) * SB;
    for (int k = threadIdx.x; k < K; k += blockDim.x) out[(size_t)t * K + k] = sk_weight<FMT>(rp + (size_t)(k >> 8) * SB, k & 255);
}

__global__ void k_embed_q8row(const uint8_t* __restrict__ W, int N, int K, const int32_t* __restrict__ ids, float* __restrict__ out) {
    const int t = blockIdx.x;
    const int64_t id = ids[t];
    const float scale = ((const float*)W)[id];
    const int8_t* q = (const int8_t*)(W + (size_t)N * 4) + (size_t)id * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) out[(size_t)t * K + k] = (float)q[k] * scale;
}

}  // namespace skd

#define HY4_LAUNCH_SK(F)                                                                                                    \
    case F: hipLaunchKernelGGL((skd::k_linear_sk<F, 8>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M); break

// Y[m][n] = sum_k W[n][k] * X[m][k]. W: device pointer to the raw .strata bytes of the [N,K] weight.
// reference (phase 1) kernels; the production dispatcher linear_h() is in skhy4.hip.h
inline void linear_h_ref(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    dim3 grid((N + 7) / 8), block(256);
    switch (fmt) {
        HY4_LAUNCH_SK(FMT_S2);
        HY4_LAUNCH_SK(FMT_S3);
        HY4_LAUNCH_SK(FMT_S4);
        HY4_LAUNCH_SK(FMT_S5);
        HY4_LAUNCH_SK(FMT_S6);
        HY4_LAUNCH_SK(FMT_S8);
        case FMT_Q8ROW: hipLaunchKernelGGL((skd::k_linear_q8row<8>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M); break;
        default: fprintf(stderr, "linear_h: unsupported fmt %d\n", fmt); abort();
    }
}

// F32 weights with fp32 activations: Y[m][n] = sum_k Wf[n][k] * X[m][k]
inline void linear_f32(hipStream_t s, const float* W, int N, int K, const float* X, float* Y, int M) {
    dim3 grid((N + 7) / 8), block(256);
    hipLaunchKernelGGL((skd::k_linear_f32<8>), grid, block, 0, s, W, N, K, X, Y, M);
}

// Row gather + dequant: out[t][:] = fp32 row ids[t] of W [N,K]
inline void embed_rows(hipStream_t s, int fmt, const void* W, int N, int K, const int32_t* ids, float* out, int T) {
    dim3 grid(T), block(256);
    const uint8_t* w = (const uint8_t*)W;
    switch (fmt) {
        case FMT_S2: hipLaunchKernelGGL((skd::k_embed_sk<FMT_S2>), grid, block, 0, s, w, K, ids, out); break;
        case FMT_S3: hipLaunchKernelGGL((skd::k_embed_sk<FMT_S3>), grid, block, 0, s, w, K, ids, out); break;
        case FMT_S4: hipLaunchKernelGGL((skd::k_embed_sk<FMT_S4>), grid, block, 0, s, w, K, ids, out); break;
        case FMT_S5: hipLaunchKernelGGL((skd::k_embed_sk<FMT_S5>), grid, block, 0, s, w, K, ids, out); break;
        case FMT_S6: hipLaunchKernelGGL((skd::k_embed_sk<FMT_S6>), grid, block, 0, s, w, K, ids, out); break;
        case FMT_S8: hipLaunchKernelGGL((skd::k_embed_sk<FMT_S8>), grid, block, 0, s, w, K, ids, out); break;
        case FMT_Q8ROW: hipLaunchKernelGGL((skd::k_embed_q8row), grid, block, 0, s, w, N, K, ids, out); break;
        default: fprintf(stderr, "embed_rows: unsupported fmt %d\n", fmt); abort();
    }
}

}  // namespace hy4
