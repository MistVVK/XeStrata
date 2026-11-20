// sy_linear.h - SYCL linear-algebra primitives of the Hy4-Flash runner (Arc A750).
//   Y[m][n] = sum_k W[n][k] * X[m][k]   (X: half [M][K], Y: float [M][N], fp32 accumulation)
//   w = (q - H) * D * (u/255)  per 64-weight sub-group, H = 1.5/3.5/7.5/15.5/31.5 (S2..S6), 0 for S8 (q signed int8); Q8ROW = [N f32 scales][N*K int8]
// Phase 1: one sub-group (32 lanes) per output row, MT tokens per pass; correctness reference for later optimised kernels.
#pragma once
#include "sy_common.h"

namespace hy4s {

template <int FMT, int MT>
inline void k_linear_sk(const uint8_t* W, int N, int K, const half* X, float* Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    Q().parallel_for(NR((N + 7) / 8, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int wave = (int)it.get_local_id(0) >> 5, lane = (int)it.get_local_id(0) & 31;
        const int row = (int)it.get_group(0) * 8 + wave;
        if (row >= N) return;
        const int nsb = K >> 8;
        const uint8_t* rp = W + (size_t)row * nsb * SB;
        for (int m0 = 0; m0 < M; m0 += MT) {
            float acc[MT];
            for (int i = 0; i < MT; ++i) acc[i] = 0.f;
            for (int unit = lane; unit < nsb * 32; unit += 32) {
                const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                const uint8_t* blk = rp + (size_t)sbi * SB;
                const float D = h2f_bits(ld16(blk));
                const float eff = D * ((float)blk[2 + g] / 255.0f);
                float w[8];
                for (int t = 0; t < 8; ++t) w[t] = ((float)sk_q<FMT>(blk + 6, j + t) - sk_half<FMT>()) * eff;
                for (int mm = 0; mm < MT; ++mm) {
                    if (m0 + mm < M) {
                        const half* xp = X + (size_t)(m0 + mm) * K + sbi * 256 + j;
                        float a = 0.f;
                        for (int t = 0; t < 8; ++t) a += w[t] * (float)xp[t];
                        acc[mm] += a;
                    }
                }
            }
            for (int mm = 0; mm < MT; ++mm) {
                const float v = sg_sum(it, acc[mm]);
                if (lane == 0 && m0 + mm < M) Y[(size_t)(m0 + mm) * N + row] = v;
            }
        }
    });
}

template <int MT>
inline void k_linear_q8row(const uint8_t* W, int N, int K, const half* X, float* Y, int M) {
    Q().parallel_for(NR((N + 7) / 8, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int wave = (int)it.get_local_id(0) >> 5, lane = (int)it.get_local_id(0) & 31;
        const int row = (int)it.get_group(0) * 8 + wave;
        if (row >= N) return;
        const float scale = ((const float*)W)[row];
        const int8_t* q = (const int8_t*)(W + (size_t)N * 4) + (size_t)row * K;
        for (int m0 = 0; m0 < M; m0 += MT) {
            float acc[MT];
            for (int i = 0; i < MT; ++i) acc[i] = 0.f;
            for (int k = lane; k < K; k += 32) {
                const float w = (float)q[k];
                for (int mm = 0; mm < MT; ++mm)
                    if (m0 + mm < M) acc[mm] += w * (float)X[(size_t)(m0 + mm) * K + k];
            }
            for (int mm = 0; mm < MT; ++mm) {
                const float v = sg_sum(it, acc[mm]);
                if (lane == 0 && m0 + mm < M) Y[(size_t)(m0 + mm) * N + row] = v * scale;
            }
        }
    });
}

// F32 weights, F32 activations
template <int MT>
inline void k_linear_f32(const float* W, int N, int K, const float* X, float* Y, int M) {
    Q().parallel_for(NR((N + 7) / 8, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int wave = (int)it.get_local_id(0) >> 5, lane = (int)it.get_local_id(0) & 31;
        const int row = (int)it.get_group(0) * 8 + wave;
        if (row >= N) return;
        const float* wp = W + (size_t)row * K;
        for (int m0 = 0; m0 < M; m0 += MT) {
            float acc[MT];
            for (int i = 0; i < MT; ++i) acc[i] = 0.f;
            for (int k = lane; k < K; k += 32) {
                const float w = wp[k];
                for (int mm = 0; mm < MT; ++mm)
                    if (m0 + mm < M) acc[mm] += w * X[(size_t)(m0 + mm) * K + k];
            }
            for (int mm = 0; mm < MT; ++mm) {
                const float v = sg_sum(it, acc[mm]);
                if (lane == 0 && m0 + mm < M) Y[(size_t)(m0 + mm) * N + row] = v;
            }
        }
    });
}

// Tiled Q8ROW GEMM for prefill (M >= 16): BM=32, BN=32, BK=32, 128 threads (16x8 -> each thread computes 4x2 = 8 outputs).
// Avoids 2-WG stall on small-N / small-K indexer projections (key_gate [128,128], key_value [128,128], head_weight [16,3072]).
inline void k_gemm_q8row(const uint8_t* W, int N, int K, const half* X, float* Y, int M) {
    const int nbm = (M + 31) / 32, nbn = (N + 31) / 32;
    const float* scales = reinterpret_cast<const float*>(W);
    const int8_t* Qw = reinterpret_cast<const int8_t*>(W + (size_t)N * 4);
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sA(32 * 33, h);
        sycl::local_accessor<float, 1> sB(32 * 33, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nbm * nbn * 128, 128), [=](sycl::nd_item<1> it) {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            const int tm = (tid >> 4) * 4;  // 0,4,...,28 (8 rows of 4)
            const int tn = (tid & 15) * 2;  // 0,2,...,30 (16 cols of 2)
            float c[4][2] = {};
            for (int k0 = 0; k0 < K; k0 += 32) {
                for (int idx = tid; idx < 32 * 32; idx += 128) {
                    const int r = idx >> 5, k = idx & 31;
                    const int gm = bm * 32 + r;
                    sA[r * 33 + k] = (gm < M && k0 + k < K) ? (float)X[(size_t)gm * K + k0 + k] : 0.f;
                    const int gn = bn * 32 + r;
                    sB[r * 33 + k] = (gn < N && k0 + k < K) ? (float)Qw[(size_t)gn * K + k0 + k] : 0.f;
                }
                sycl::group_barrier(it.get_group());
                for (int k = 0; k < 32; ++k) {
                    const float a0 = sA[(tm + 0) * 33 + k], a1 = sA[(tm + 1) * 33 + k];
                    const float a2 = sA[(tm + 2) * 33 + k], a3 = sA[(tm + 3) * 33 + k];
                    const float b0 = sB[(tn + 0) * 33 + k], b1 = sB[(tn + 1) * 33 + k];
                    c[0][0] += a0 * b0; c[0][1] += a0 * b1;
                    c[1][0] += a1 * b0; c[1][1] += a1 * b1;
                    c[2][0] += a2 * b0; c[2][1] += a2 * b1;
                    c[3][0] += a3 * b0; c[3][1] += a3 * b1;
                }
                sycl::group_barrier(it.get_group());
            }
            const int gn0 = bn * 32 + tn, gn1 = gn0 + 1;
            const float sc0 = gn0 < N ? scales[gn0] : 0.f;
            const float sc1 = gn1 < N ? scales[gn1] : 0.f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int gm = bm * 32 + tm + r;
                if (gm < M) {
                    if (gn0 < N) Y[(size_t)gm * N + gn0] = c[r][0] * sc0;
                    if (gn1 < N) Y[(size_t)gm * N + gn1] = c[r][1] * sc1;
                }
            }
        });
    });
}

// Tiled exact-FP32 GEMM for prefill router (W [N,K] f32, X [M,K] f32 -> Y [M,N] f32, M >= 16).
// BM=32, BN=64, BK=16, 128 threads (8x16 -> each thread computes 4x4 = 16 outputs). Zero FP16 rounding!
inline void k_gemm_f32(const float* W, int N, int K, const float* X, float* Y, int M) {
    const int nbm = (M + 31) / 32, nbn = (N + 63) / 64;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sA(32 * 17, h);
        sycl::local_accessor<float, 1> sB(64 * 17, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nbm * nbn * 128, 128), [=](sycl::nd_item<1> it) {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            const int tm = (tid >> 4) * 4;   // 0,4,...,28 (8 rows of 4)
            const int tn = (tid & 15) * 4;   // 0,4,...,60 (16 cols of 4)
            float c[4][4] = {};
            for (int k0 = 0; k0 < K; k0 += 16) {
                // Load 32x16 of X (512 floats = 4/thread) and 64x16 of W (1024 floats = 8/thread)
                for (int idx = tid; idx < 32 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15;
                    const int gm = bm * 32 + r;
                    sA[r * 17 + k] = (gm < M && k0 + k < K) ? X[(size_t)gm * K + k0 + k] : 0.f;
                }
                for (int idx = tid; idx < 64 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15;
                    const int gn = bn * 64 + r;
                    sB[r * 17 + k] = (gn < N && k0 + k < K) ? W[(size_t)gn * K + k0 + k] : 0.f;
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    const float a0 = sA[(tm + 0) * 17 + k], a1 = sA[(tm + 1) * 17 + k];
                    const float a2 = sA[(tm + 2) * 17 + k], a3 = sA[(tm + 3) * 17 + k];
                    const float b0 = sB[(tn + 0) * 17 + k], b1 = sB[(tn + 1) * 17 + k];
                    const float b2 = sB[(tn + 2) * 17 + k], b3 = sB[(tn + 3) * 17 + k];
                    c[0][0] += a0 * b0; c[0][1] += a0 * b1; c[0][2] += a0 * b2; c[0][3] += a0 * b3;
                    c[1][0] += a1 * b0; c[1][1] += a1 * b1; c[1][2] += a1 * b2; c[1][3] += a1 * b3;
                    c[2][0] += a2 * b0; c[2][1] += a2 * b1; c[2][2] += a2 * b2; c[2][3] += a2 * b3;
                    c[3][0] += a3 * b0; c[3][1] += a3 * b1; c[3][2] += a3 * b2; c[3][3] += a3 * b3;
                }
                sycl::group_barrier(it.get_group());
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int gm = bm * 32 + tm + r;
                if (gm < M) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int gn = bn * 64 + tn + j;
                        if (gn < N) Y[(size_t)gm * N + gn] = c[r][j];
                    }
                }
            }
        });
    });
}

inline void linear_h(int fmt, const void* Wv, int N, int K, const half* X, float* Y, int M) {
    const uint8_t* W = (const uint8_t*)Wv;
    switch (fmt) {
#define HY4S_LIN(F)                                                                                  \
    case F:                                                                                          \
        if (M <= 8) k_linear_sk<F, 8>(W, N, K, X, Y, M); else k_linear_sk<F, 16>(W, N, K, X, Y, M);   \
        break
        HY4S_LIN(FMT_S2);
        HY4S_LIN(FMT_S3);
        HY4S_LIN(FMT_S4);
        HY4S_LIN(FMT_S5);
        HY4S_LIN(FMT_S6);
        HY4S_LIN(FMT_S8);
#undef HY4S_LIN
        case FMT_Q8ROW:
            if (M >= 16) k_gemm_q8row(W, N, K, X, Y, M);
            else k_linear_q8row<8>(W, N, K, X, Y, M);
            break;
        default: fprintf(stderr, "linear_h: unsupported fmt %d\n", fmt); abort();
    }
}

inline void linear_f32(const float* W, int N, int K, const float* X, float* Y, int M) {
    if (M >= 16) k_gemm_f32(W, N, K, X, Y, M);
    else k_linear_f32<8>(W, N, K, X, Y, M);
}

// kv_b_proj [NH*448][512] (any S_K fmt) -> wk_b [NH][192][512] and wv_b [NH][256][512] fp16 (absorbed MLA weights)
template <int FMT>
inline void k_split_kvb(const uint8_t* W, half* wk, half* wv, int NH) {
    Q().parallel_for(NR((size_t)NH * 448, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int r = (int)it.get_group(0), h = r / 448, j = r % 448;
        constexpr int SB = sk_sb_bytes(FMT);
        const uint8_t* rp = W + (size_t)r * 2 * SB;
        half* dst = j < 192 ? wk + ((size_t)h * 192 + j) * 512 : wv + ((size_t)h * 256 + (j - 192)) * 512;
        for (int k = (int)it.get_local_id(0); k < 512; k += 128) dst[k] = (half)sk_weight<FMT>(rp + (size_t)(k >> 8) * SB, k & 255);
    });
}
inline bool split_kvb(int fmt, const void* W, half* wk, half* wv, int NH) {
    const uint8_t* w = (const uint8_t*)W;
    switch (fmt) {
        case FMT_S4: k_split_kvb<FMT_S4>(w, wk, wv, NH); return true;
        case FMT_S5: k_split_kvb<FMT_S5>(w, wk, wv, NH); return true;
        case FMT_S6: k_split_kvb<FMT_S6>(w, wk, wv, NH); return true;
        case FMT_S8: k_split_kvb<FMT_S8>(w, wk, wv, NH); return true;
        default: return false;
    }
}

}  // namespace hy4s
