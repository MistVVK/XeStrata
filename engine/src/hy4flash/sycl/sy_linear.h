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
            if (M <= 8) k_linear_q8row<8>(W, N, K, X, Y, M); else k_linear_q8row<16>(W, N, K, X, Y, M);
            break;
        default: fprintf(stderr, "linear_h: unsupported fmt %d\n", fmt); abort();
    }
}

inline void linear_f32(const float* W, int N, int K, const float* X, float* Y, int M) {
    if (M <= 8) k_linear_f32<8>(W, N, K, X, Y, M); else k_linear_f32<16>(W, N, K, X, Y, M);
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
