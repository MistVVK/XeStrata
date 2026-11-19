// skgemm.hip.h - HIP linear-algebra primitives for the Hy4-Flash runner on gfx1151 (wave32, RDNA 3.5)
//
// Phase 1 (correctness): straightforward wave-per-row kernels that dequantize S_K / Q8ROW weights on the fly.
//   Y[m][n] = sum_k W[n][k] * X[m][k]   (X: __half [M][K], Y: float [M][N], fp32 accumulation)
//   w = (q - H) * D * (u/255)  per 64-weight sub-group, H = 1.5/3.5/7.5/15.5/31.5 (S2..S6), 0 for S8 (q signed int8)
// Header-only (one translation unit per program).
#pragma once
#include "hy4_gpu.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace hy4 {

// Fmt values for S_K are the .strata code ids (0 = S1.25, 1 = S1.5, 2..6 = S2_K..S6_K, 8 = S8_K)
enum Fmt { FMT_S1_25 = 0, FMT_S1_5 = 1, FMT_S2 = 2, FMT_S3 = 3, FMT_S4 = 4, FMT_S5 = 5, FMT_S6 = 6, FMT_S8 = 8, FMT_Q8ROW = 100, FMT_F32 = 101 };

constexpr int sk_sb_bytes(int f) {
    return f == FMT_S1_25 ? 48 : f == FMT_S1_5 ? 58 : f == FMT_S2 ? 70 : f == FMT_S3 ? 102 : f == FMT_S4 ? 134 : f == FMT_S5 ? 166 : f == FMT_S6 ? 198
         : f == FMT_S8 ? 262 : 0;
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

// ---- the two sub-2-bit codes (S1.25 = lattice codebook + fine nibbles, S1.5 = base-3 ternary), exact as in src/quant/sk_codecs.cpp
__constant__ int8_t kS125CB[32][4] = {
    {-1, -1, -1, -1}, {-1, -1, -1,  1}, {-1, -1,  1, -1}, {-1, -1,  1,  1},
    {-1,  1, -1, -1}, {-1,  1, -1,  1}, {-1,  1,  1, -1}, {-1,  1,  1,  1},
    { 1, -1, -1, -1}, { 1, -1, -1,  1}, { 1, -1,  1, -1}, { 1, -1,  1,  1},
    { 1,  1, -1, -1}, { 1,  1, -1,  1}, { 1,  1,  1, -1}, { 1,  1,  1,  1},
    { 2,  0,  0,  0}, {-2,  0,  0,  0}, { 0,  2,  0,  0}, { 0, -2,  0,  0},
    { 0,  0,  2,  0}, { 0,  0, -2,  0}, { 0,  0,  0,  2}, { 0,  0,  0, -2},
    { 0,  0,  0,  0}, { 2,  2,  0,  0}, {-2, -2,  0,  0}, { 0,  0,  2,  2},
    { 0,  0, -2, -2}, { 2, -2,  0,  0}, { 0,  0,  2, -2}, { 2,  0,  2,  0},
};
constexpr uint32_t s125_hb(int v) { return v == 1 ? 0x3C00u : v == -1 ? 0xBC00u : v == 2 ? 0x4000u : v == -2 ? 0xC000u : 0u; }
constexpr uint32_t s125_h2(int a, int b) { return s125_hb(a) | (s125_hb(b) << 16); }
#define HY4_CB(a, b, c, d) {s125_h2(a, b), s125_h2(c, d)}
// the same codebook as packed fp16 pairs: .x = (c0, c1), .y = (c2, c3)
__constant__ uint2 kS125H2[32] = {
    HY4_CB(-1, -1, -1, -1), HY4_CB(-1, -1, -1, 1), HY4_CB(-1, -1, 1, -1), HY4_CB(-1, -1, 1, 1),
    HY4_CB(-1, 1, -1, -1),  HY4_CB(-1, 1, -1, 1),  HY4_CB(-1, 1, 1, -1),  HY4_CB(-1, 1, 1, 1),
    HY4_CB(1, -1, -1, -1),  HY4_CB(1, -1, -1, 1),  HY4_CB(1, -1, 1, -1),  HY4_CB(1, -1, 1, 1),
    HY4_CB(1, 1, -1, -1),   HY4_CB(1, 1, -1, 1),   HY4_CB(1, 1, 1, -1),   HY4_CB(1, 1, 1, 1),
    HY4_CB(2, 0, 0, 0),     HY4_CB(-2, 0, 0, 0),   HY4_CB(0, 2, 0, 0),    HY4_CB(0, -2, 0, 0),
    HY4_CB(0, 0, 2, 0),     HY4_CB(0, 0, -2, 0),   HY4_CB(0, 0, 0, 2),    HY4_CB(0, 0, 0, -2),
    HY4_CB(0, 0, 0, 0),     HY4_CB(2, 2, 0, 0),    HY4_CB(-2, -2, 0, 0),  HY4_CB(0, 0, 2, 2),
    HY4_CB(0, 0, -2, -2),   HY4_CB(2, -2, 0, 0),   HY4_CB(0, 0, 2, -2),   HY4_CB(2, 0, 2, 0),
};
#undef HY4_CB

// LDS copy of kS125H2 (one per kernel). s125_lut_init() must be followed by a __syncthreads() before the first use.
__device__ __forceinline__ uint2* s125_lut() {
    __shared__ uint2 lut[32];
    return lut;
}
__device__ __forceinline__ void s125_lut_init() {
    if (threadIdx.x < 32) s125_lut()[threadIdx.x] = kS125H2[threadIdx.x];
}

// trit d (0..4) of a base-3 byte, minus 1 (bytes >= 243 decode to 0 like the CPU table)
__device__ __forceinline__ int s15_trit(uint32_t b, int d) {
    if (b >= 243u) return 0;
    const uint32_t m = d == 0 ? 65536u : d == 1 ? 21846u : d == 2 ? 7282u : d == 3 ? 2428u : 810u;
    const uint32_t x = (b * m) >> 16;  // floor(b / 3^d), exact for b < 243
    const uint32_t x3 = (x * 21846u) >> 16;
    return (int)(x - 3u * x3) - 1;
}

// S1.25: the 5-bit codebook index of weight group n (4 weights each, n = 0..63) of payload pl
__device__ __forceinline__ uint32_t s125_index(const uint8_t* pl, int n) {
    const int G = n >> 3, j = n & 7;
    const int bit = 5 * j;
    const uint8_t* p = pl + G * 5 + (bit >> 3);
    const uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8);  // bit + 5 <= 40, so 2 bytes always cover the field
    return (v >> (bit & 7)) & 31u;
}
__device__ __forceinline__ float s125_fine(const uint8_t* pl, int g) {
    const uint32_t fb = pl[40 + (g >> 1)];
    return 0.5f + (float)((fb >> ((g & 1) * 4)) & 15u) / 15.0f;
}

// one weight of super-block `blk` at index i (0..255)
template <int FMT> __device__ __forceinline__ float sk_weight(const uint8_t* blk, int i) {
    const float D = h2f_bits(ld16(blk));
    const int g = i >> 6;
    if constexpr (FMT == FMT_S1_25) {
        const float eff = D * ((float)blk[2 + g] / 255.0f) * s125_fine(blk + 6, g);
        return (float)kS125CB[s125_index(blk + 6, i >> 2)][i & 3] * eff;
    } else if constexpr (FMT == FMT_S1_5) {
        const float eff = D * ((float)blk[2 + g] / 255.0f);
        return (float)s15_trit(blk[6 + i / 5], i % 5) * eff;
    } else {
        const float eff = D * ((float)blk[2 + g] / 255.0f);
        return ((float)sk_q<FMT>(blk + 6, i) - sk_half<FMT>()) * eff;
    }
}

// sub-group scale used by the fast kernels: D * u[g]/255 (and * fine_g for S1.25). The S2..S8 expression is unchanged from before.
template <int FMT> __device__ __forceinline__ float sk_eff(const uint8_t* blk, int g) {
    const float D = h2f_bits(ld16(blk));
    float eff = D * ((float)blk[2 + g] * (1.0f / 255.0f));
    if constexpr (FMT == FMT_S1_25) eff *= s125_fine(blk + 6, g);
    return eff;
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
            float w[8];
            if constexpr (FMT == FMT_S1_25 || FMT == FMT_S1_5) {
#pragma unroll
                for (int t = 0; t < 8; ++t) w[t] = sk_weight<FMT>(blk, j + t);
            } else {
                const float D = h2f_bits(ld16(blk));
                const float eff = D * ((float)blk[2 + g] / 255.0f);
#pragma unroll
                for (int t = 0; t < 8; ++t) w[t] = ((float)sk_q<FMT>(blk + 6, j + t) - sk_half<FMT>()) * eff;
            }
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
        HY4_LAUNCH_SK(FMT_S1_25);
        HY4_LAUNCH_SK(FMT_S1_5);
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
        case FMT_S1_25: hipLaunchKernelGGL((skd::k_embed_sk<FMT_S1_25>), grid, block, 0, s, w, K, ids, out); break;
        case FMT_S1_5: hipLaunchKernelGGL((skd::k_embed_sk<FMT_S1_5>), grid, block, 0, s, w, K, ids, out); break;
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
