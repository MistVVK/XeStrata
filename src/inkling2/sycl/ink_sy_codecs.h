// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/sycl/ink_sy_codecs.h - Inkling-2-Mini's linear and expert kernels on 16 logical lanes
// (ink_sy_common.h) for the .strata v1.1 formats:
//   - dp4a Q8_ROW linear with Q8 activations (sycl::ext::oneapi::dot_acc)
//   - SF4 (NVFP4 as it is), MX8 (MXFP8 as it is), S_K (S2_K..S8_K and the paired 2-row w2 SB_0..SB_8)
//   - a tiled FP16-free GEMM for 16 rows and more where the device has no matrix engines (ink_sy_xmx.h otherwise)
//   - the experts binned on the GPU (k_moe_plan) + fused w13 SITU(4.0, 25.0) + w2 + the 8-way reduce
#pragma once

#include <algorithm>
#include <cstdint>

#ifdef INK_SYCL_USE_ONEMKL
#include <oneapi/mkl/blas.hpp>
#endif

#include "ink_sy_common.h"

namespace ink_sycl {

template <int F>
inline void sk_q8_sy(const uint8_t* pl, int j, int* q) {
    if constexpr (F == FMT_S2) {
        const uint8_t b0 = pl[j >> 2], b1 = pl[(j >> 2) + 1];
        q[0] = b0 & 3; q[1] = (b0 >> 2) & 3; q[2] = (b0 >> 4) & 3; q[3] = (b0 >> 6) & 3;
        q[4] = b1 & 3; q[5] = (b1 >> 2) & 3; q[6] = (b1 >> 4) & 3; q[7] = (b1 >> 6) & 3;
    } else if constexpr (F == FMT_S3) {
        const uint8_t b0 = pl[j >> 2], b1 = pl[(j >> 2) + 1], h = pl[64 + (j >> 3)];
        q[0] = (b0 & 3) | (((h >> 0) & 1) << 2); q[1] = ((b0 >> 2) & 3) | (((h >> 1) & 1) << 2);
        q[2] = ((b0 >> 4) & 3) | (((h >> 2) & 1) << 2); q[3] = ((b0 >> 6) & 3) | (((h >> 3) & 1) << 2);
        q[4] = (b1 & 3) | (((h >> 4) & 1) << 2); q[5] = ((b1 >> 2) & 3) | (((h >> 5) & 1) << 2);
        q[6] = ((b1 >> 4) & 3) | (((h >> 6) & 1) << 2); q[7] = ((b1 >> 6) & 3) | (((h >> 7) & 1) << 2);
    } else if constexpr (F == FMT_S4) {
        const uint8_t b0 = pl[(j >> 1) + 0], b1 = pl[(j >> 1) + 1], b2 = pl[(j >> 1) + 2], b3 = pl[(j >> 1) + 3];
        q[0] = b0 & 15; q[1] = b0 >> 4; q[2] = b1 & 15; q[3] = b1 >> 4;
        q[4] = b2 & 15; q[5] = b2 >> 4; q[6] = b3 & 15; q[7] = b3 >> 4;
    } else if constexpr (F == FMT_S5) {
        const uint8_t b0 = pl[(j >> 1) + 0], b1 = pl[(j >> 1) + 1], b2 = pl[(j >> 1) + 2], b3 = pl[(j >> 1) + 3];
        const uint8_t h = pl[128 + (j >> 3)];
        q[0] = (b0 & 15) | (((h >> 0) & 1) << 4); q[1] = (b0 >> 4) | (((h >> 1) & 1) << 4);
        q[2] = (b1 & 15) | (((h >> 2) & 1) << 4); q[3] = (b1 >> 4) | (((h >> 3) & 1) << 4);
        q[4] = (b2 & 15) | (((h >> 4) & 1) << 4); q[5] = (b2 >> 4) | (((h >> 5) & 1) << 4);
        q[6] = (b3 & 15) | (((h >> 6) & 1) << 4); q[7] = (b3 >> 4) | (((h >> 7) & 1) << 4);
    } else if constexpr (F == FMT_S6) {
        const uint8_t b0 = pl[(j >> 1) + 0], b1 = pl[(j >> 1) + 1], b2 = pl[(j >> 1) + 2], b3 = pl[(j >> 1) + 3];
        const uint8_t h0 = pl[128 + (j >> 2)], h1 = pl[128 + (j >> 2) + 1];
        q[0] = (b0 & 15) | (((h0 >> 0) & 3) << 4); q[1] = (b0 >> 4) | (((h0 >> 2) & 3) << 4);
        q[2] = (b1 & 15) | (((h0 >> 4) & 3) << 4); q[3] = (b1 >> 4) | (((h0 >> 6) & 3) << 4);
        q[4] = (b2 & 15) | (((h1 >> 0) & 3) << 4); q[5] = (b2 >> 4) | (((h1 >> 2) & 3) << 4);
        q[6] = (b3 & 15) | (((h1 >> 4) & 3) << 4); q[7] = (b3 >> 4) | (((h1 >> 6) & 3) << 4);
    } else {
        const int8_t* qp = reinterpret_cast<const int8_t*>(pl + j);
        for (int i = 0; i < 8; ++i) q[i] = (int)qp[i];
    }
}

// ============================================================================
// 1. Dense Trunk & DSpark Linear Kernels (SG16: BF16, Q8_ROW w/ dp4a, F32, S_K)
// ============================================================================

// BF16 SG16 GEMV (16 SGs = 256 threads/WG -> 16 rows/WG)
inline void k_linear_bf16_sg16(const uint16_t* W, int N, int K, const float* X, float* Y, int M) {
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int sg = (int)it.get_local_id(0) >> 4;
        const int lane = (int)it.get_local_id(0) & 15;
        const int row_any = (int)it.get_group(0) * 16 + sg;
        const bool live = row_any < N;   // no early return: both halves of a 32-wide sub-group reach the sums
        const int row = live ? row_any : N - 1;
        const uint16_t* wp = W + (size_t)row * K;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            float a[4] = {0, 0, 0, 0};
            for (int k = lane * 4; k < K; k += 64) {
                const float w0 = bf16_to_f32(wp[k + 0]), w1 = bf16_to_f32(wp[k + 1]);
                const float w2 = bf16_to_f32(wp[k + 2]), w3 = bf16_to_f32(wp[k + 3]);
                for (int i = 0; i < mm; ++i) {
                    const float* xp = X + (size_t)(m0 + i) * K + k;
                    a[i] += w0 * xp[0] + w1 * xp[1] + w2 * xp[2] + w3 * xp[3];
                }
            }
            for (int i = 0; i < mm; ++i) {
                const float v = sg_sum(it, a[i]);
                if (lane == 0 && live) Y[(size_t)(m0 + i) * N + row] = v;
            }
        }
    });
}

// GGUF Q8_0 rows (FMT_Q8_0: the N * K/32 block scales, then the int8 weights) for any M: a sub-group per output
// row, four weights a lane from one 32-bit load, FP32 products (the activations are not quantized)
inline void k_linear_q8_0_sg16(const uint8_t* W, int N, int K, const float* X, float* Y, int M, int N_total) {
    const int nb = K >> 5, K4 = K >> 2;
    const float* scales = reinterpret_cast<const float*>(W);
    const int32_t* qw4 = reinterpret_cast<const int32_t*>(W + (size_t)N_total * nb * 4);
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int sg = (int)it.get_local_id(0) >> 4;
        const int lane = (int)it.get_local_id(0) & 15;
        const int row_any = (int)it.get_group(0) * 16 + sg;
        const bool live = row_any < N;   // no early return: every lane reaches the sums
        const int row = live ? row_any : N - 1;
        const float* sc = scales + (size_t)row * nb;
        const int32_t* wp = qw4 + (size_t)row * K4;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            float a[4] = {0, 0, 0, 0};
            for (int k = lane; k < K4; k += 16) {
                const int32_t v = wp[k];
                const float d = sc[k >> 3];
                const float w0 = (float)(int8_t)(v & 0xff) * d, w1 = (float)(int8_t)((v >> 8) & 0xff) * d;
                const float w2 = (float)(int8_t)((v >> 16) & 0xff) * d, w3 = (float)(int8_t)(v >> 24) * d;
                for (int i = 0; i < mm; ++i) {
                    const float* xp = X + (size_t)(m0 + i) * K + 4 * k;
                    a[i] += w0 * xp[0] + w1 * xp[1] + w2 * xp[2] + w3 * xp[3];
                }
            }
            for (int i = 0; i < mm; ++i) {
                const float s = sg_sum(it, a[i]);
                if (lane == 0 && live) Y[(size_t)(m0 + i) * N + row] = s;
            }
        }
    });
}

// BF16 SG16 Tiled GEMM for M >= 16 (BM=32, BN=64, BK=16, 128 threads = 8 SG16)
inline void k_gemm_bf16_sg16(const uint16_t* W, int N, int K, const float* X, float* Y, int M) {
    const int nbm = (M + 31) / 32, nbn = (N + 63) / 64;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sA(32 * 17, h);
        sycl::local_accessor<float, 1> sB(64 * 17, h);
        h.parallel_for(NR((size_t)nbm * nbn, 128), [=](sycl::nd_item<1> it) INK_REQD_SG {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            const int tm = (tid >> 4) * 4, tn = (tid & 15) * 4;
            float c[4][4] = {};
            for (int k0 = 0; k0 < K; k0 += 16) {
                for (int idx = tid; idx < 32 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15, gm = bm * 32 + r;
                    sA[r * 17 + k] = (gm < M && k0 + k < K) ? X[(size_t)gm * K + k0 + k] : 0.f;
                }
                for (int idx = tid; idx < 64 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15, gn = bn * 64 + r;
                    sB[r * 17 + k] = (gn < N && k0 + k < K) ? bf16_to_f32(W[(size_t)gn * K + k0 + k]) : 0.f;
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
            for (int r = 0; r < 4; ++r) {
                const int gm = bm * 32 + tm + r;
                if (gm < M) {
                    for (int j = 0; j < 4; ++j) {
                        const int gn = bn * 64 + tn + j;
                        if (gn < N) Y[(size_t)gm * N + gn] = c[r][j];
                    }
                }
            }
        });
    });
}

// Quantize X [M, K] -> Q8_ROW [M*K int8] + [M float scales] for deterministic dp4a dot_acc!
inline void k_quantize_x_q8_sg16(const float* X, int M, int K, int8_t* xq, float* xsc) {
    Q().parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const float* xp = X + (size_t)m * K;
        float amax = 0.0f;
        for (int k = tid; k < K; k += 256) amax = sycl::fmax(amax, sycl::fabs(xp[k]));
        amax = lane16_max(it, amax);
        const int w = group16(it), l = lane16(it);
        sycl::group_barrier(it.get_group());
        if (l == 0) sh[w] = amax;
        sycl::group_barrier(it.get_group());
        float gmax = sh[l];   // 256 work-items: 16 logical sub-groups
        gmax = lane16_max(it, gmax);
        const float d = gmax * (1.0f / 127.0f);
        const float inv = gmax > 0.0f ? (127.0f / gmax) : 0.0f;
        if (tid == 0) xsc[m] = d;
        int8_t* dst = xq + (size_t)m * K;
        for (int k = tid; k < K; k += 256) dst[k] = (int8_t)sycl::clamp((int)sycl::rint(xp[k] * inv), -127, 127);
    });
}

// Q8_ROW SG16 dp4a GEMV (deterministic dot_acc on int8x4!)
inline void k_linear_q8row_dp4a_sg16(const uint8_t* W, int N, int K, const int8_t* xq, const float* xsc, float* Y, int M, int N_total = -1) {
    const int n_scales = (N_total > 0) ? N_total : N;
    const float* scales = reinterpret_cast<const float*>(W);
    const int32_t* qw4 = reinterpret_cast<const int32_t*>(W + (size_t)n_scales * 4);
    const int32_t* xq4 = reinterpret_cast<const int32_t*>(xq);
    const int K4 = K >> 2;
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int sg = (int)it.get_local_id(0) >> 4, lane = (int)it.get_local_id(0) & 15;
        const int row_any = (int)it.get_group(0) * 16 + sg;
        const bool live = row_any < N;   // no early return: both halves of a 32-wide sub-group reach the sums
        const int row = live ? row_any : N - 1;
        const float wsc = scales[row];
        const int32_t* wp = qw4 + (size_t)row * K4;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            int32_t acc[4] = {0, 0, 0, 0};
            for (int k = lane; k < K4; k += 16) {
                const int32_t wv = wp[k];
                for (int i = 0; i < mm; ++i) {
                    acc[i] = dp4a_s8(wv, xq4[(size_t)(m0 + i) * K4 + k], acc[i]);
                }
            }
            for (int i = 0; i < mm; ++i) {
                const int32_t s = lane16_sum(it, acc[i]);
                if (lane == 0 && live) Y[(size_t)(m0 + i) * N + row] = (float)s * (wsc * xsc[m0 + i]);
            }
        }
    });
}

// Q8_ROW SG16 exact FP32-act fallback & tiled GEMM for M >= 16
inline void k_gemm_q8row_sg16(const uint8_t* W, int N, int K, const float* X, float* Y, int M, int N_total = -1) {
    const int nbm = (M + 31) / 32, nbn = (N + 63) / 64;
    const int n_scales = (N_total > 0) ? N_total : N;
    const float* scales = reinterpret_cast<const float*>(W);
    const int8_t* qw = reinterpret_cast<const int8_t*>(W + (size_t)n_scales * 4);
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sA(32 * 17, h);
        sycl::local_accessor<float, 1> sB(64 * 17, h);
        h.parallel_for(NR((size_t)nbm * nbn, 128), [=](sycl::nd_item<1> it) INK_REQD_SG {
            const int gid = (int)it.get_group(0), bn = gid % nbn, bm = gid / nbn, tid = (int)it.get_local_id(0);
            const int tm = (tid >> 4) * 4, tn = (tid & 15) * 4;
            float c[4][4] = {};
            for (int k0 = 0; k0 < K; k0 += 16) {
                for (int idx = tid; idx < 32 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15, gm = bm * 32 + r;
                    sA[r * 17 + k] = (gm < M && k0 + k < K) ? X[(size_t)gm * K + k0 + k] : 0.f;
                }
                for (int idx = tid; idx < 64 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15, gn = bn * 64 + r;
                    sB[r * 17 + k] = (gn < N && k0 + k < K) ? (float)qw[(size_t)gn * K + k0 + k] : 0.f;
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
            for (int r = 0; r < 4; ++r) {
                const int gm = bm * 32 + tm + r;
                if (gm < M) {
                    for (int j = 0; j < 4; ++j) {
                        const int gn = bn * 64 + tn + j;
                        if (gn < N) Y[(size_t)gm * N + gn] = c[r][j] * scales[gn];
                    }
                }
            }
        });
    });
}

// F32 SG16 Linear (exact FP32 for routers, controllers, dynamic sconv weight_proj, attn_sink_proj)
inline void k_linear_f32_sg16(const float* W, int N, int K, const float* X, float* Y, int M) {
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int sg = (int)it.get_local_id(0) >> 4, lane = (int)it.get_local_id(0) & 15;
        const int row_any = (int)it.get_group(0) * 16 + sg;
        const bool live = row_any < N;   // no early return: both halves of a 32-wide sub-group reach the sums
        const int row = live ? row_any : N - 1;
        const float* wp = W + (size_t)row * K;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            float a[4] = {0, 0, 0, 0};
            for (int k = lane; k < K; k += 16) {
                const float w = wp[k];
                for (int i = 0; i < mm; ++i) a[i] += w * X[(size_t)(m0 + i) * K + k];
            }
            for (int i = 0; i < mm; ++i) {
                const float v = sg_sum(it, a[i]);
                if (lane == 0 && live) Y[(size_t)(m0 + i) * N + row] = v;
            }
        }
    });
}

// S_K SG16 Linear (S5_K DSpark, S6_K unembed)
template <int F>
inline void k_linear_sk_sg16(const uint8_t* W, int N, int K, const float* X, float* Y, int M) {
    constexpr int SB = sk_sb_bytes(F);
    constexpr float H = sk_half(F);
    const int nsb = K >> 8;
    const size_t rowb = (size_t)nsb * SB;
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int sg = (int)it.get_local_id(0) >> 4, lane = (int)it.get_local_id(0) & 15;
        const int row_any = (int)it.get_group(0) * 16 + sg;
        const bool live = row_any < N;   // no early return: both halves of a 32-wide sub-group reach the sums
        const int row = live ? row_any : N - 1;
        const uint8_t* rp = W + (size_t)row * rowb;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            float acc[4] = {0, 0, 0, 0};
            for (int unit = lane; unit < nsb * 32; unit += 16) {
                const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                const uint8_t* blk = rp + (size_t)sbi * SB;
                const float eff = h2f_bits(ld16(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
                int q[8];
                sk_q8_sy<F>(blk + 6, j, q);
                float w[8];
                for (int t = 0; t < 8; ++t) w[t] = ((float)q[t] - H) * eff;
                for (int i = 0; i < mm; ++i) {
                    const float* xp = X + (size_t)(m0 + i) * K + sbi * 256 + j;
                    float s = 0.0f;
                    for (int t = 0; t < 8; ++t) s += w[t] * xp[t];
                    acc[i] += s;
                }
            }
            for (int i = 0; i < mm; ++i) {
                const float v = sg_sum(it, acc[i]);
                if (lane == 0 && live) Y[(size_t)(m0 + i) * N + row] = v;
            }
        }
    });
}

// ============================================================================
// 2. Inkling-2-Mini MoE Kernels (SG16: SF4, MX8, S2_K..S8_K + Paired 2-Row w2)
// ============================================================================

// GPU-side expert binning (1 WG of 128 threads = 8 SG16, 1 thread per expert e in 0..127)
inline void k_moe_plan_sg16(const int* topk_idx, int P, int* uexp, int* uoff, int* ucnt, int* pair_list, int* n_uniq_dev) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> sg_u(8, h);
        sycl::local_accessor<int, 1> sg_p(8, h);
        h.parallel_for(NR(1, 128), [=](sycl::nd_item<1> it) INK_REQD_SG {
            const int e = (int)it.get_local_id(0);
            const int sgi = group16(it), lane = lane16(it);
            int c = 0;
            for (int i = 0; i < P; ++i) if (topk_idx[i] == e) ++c;
            const int has = c > 0 ? 1 : 0;
            const int ex_u = lane16_exclusive_sum(it, has);
            const int ex_p = lane16_exclusive_sum(it, c);
            if (lane == 15) { sg_u[sgi] = ex_u + has; sg_p[sgi] = ex_p + c; }
            sycl::group_barrier(it.get_group());
            int bu = 0, bp = 0;
            for (int w = 0; w < sgi; ++w) { bu += sg_u[w]; bp += sg_p[w]; }
            const int pu = bu + ex_u, pp = bp + ex_p;
            if (has) {
                uexp[pu] = e; uoff[pu] = pp; ucnt[pu] = c;
                int k = 0;
                for (int i = 0; i < P; ++i) if (topk_idx[i] == e) pair_list[pp + k++] = i;
            }
            if (e == 127) n_uniq_dev[0] = pu + has;
        });
    });
}

// Fused w13 + SITU(4.0, 25.0) + 1/sq_xd_dn (grid: 72 x max_uniq, 256 threads = 16 SG16 -> 16 inter channels/WG)
inline void k_moe_gu_situ_sg16(
    const InkSyExpertDev* ex_layer, const float* x_in,
    const int* uexp, const int* uoff, const int* ucnt, const int* pair_list, const int* n_uniq_dev,
    int max_uniq, float* h_act) {
    // 1152 / 16 = 72 WGs along X, max_uniq along Y
    Q().parallel_for(NR2(72, (size_t)max_uniq, 256), [=](sycl::nd_item<2> it) INK_REQD_SG {
        const int y = (int)it.get_group(0);
        if (y >= n_uniq_dev[0]) return;
        const int sg = (int)it.get_local_id(1) >> 4, lane = (int)it.get_local_id(1) & 15;
        const int r = (int)it.get_group(1) * 16 + sg;
        if (r >= 1152) return;

        const int e = uexp[y], off = uoff[y], n = ucnt[y];
        const InkSyExpertDev ed = ex_layer[e];
        const uint8_t* body = ed.body_ptr;
        const bool is_sk_dn = (ed.dn_code != FMT_SF4 && ed.dn_code != FMT_MX8 && ed.dn_code != FMT_GGUF_RAW);
        const float inv_sq = is_sk_dn ? (1.0f / h2f_bits(ld16(body + 2 * r))) : 1.0f;

        for (int t0 = 0; t0 < n; t0 += 4) {
            const int mm = sycl::min(4, n - t0);
            int pid[4], tk[4];
            for (int i = 0; i < mm; ++i) { pid[i] = pair_list[off + t0 + i]; tk[i] = pid[i] >> 3; }
            float ag[4] = {0, 0, 0, 0}, au[4] = {0, 0, 0, 0};

            if (ed.gu_code == FMT_SF4) {
                const float* hdr = reinterpret_cast<const float*>(body);
                const float outer = hdr[0] / hdr[1];
                const uint8_t* pb = body + 16;
                const uint8_t* sb = pb + 2654208;
                const uint8_t* pg = pb + (size_t)(2 * r) * 1152;
                const uint8_t* sg_p = sb + (size_t)(2 * r) * 144;
                const uint8_t* pu = pb + (size_t)(2 * r + 1) * 1152;
                const uint8_t* su_p = sb + (size_t)(2 * r + 1) * 144;
                for (int b = lane; b < 144; b += 16) {
                    const float scg = e4m3_to_f32(sg_p[b]) * outer;
                    const float scu = e4m3_to_f32(su_p[b]) * outer;
                    const uint8_t* bg = pg + b * 8;
                    const uint8_t* bu = pu + b * 8;
                    float wg[16], wu[16];
                    for (int j = 0; j < 8; ++j) {
                        wg[2*j] = e2m1_lut(bg[j] & 15u); wg[2*j+1] = e2m1_lut(bg[j] >> 4);
                        wu[2*j] = e2m1_lut(bu[j] & 15u); wu[2*j+1] = e2m1_lut(bu[j] >> 4);
                    }
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + b * 16;
                        float dg = 0.0f, du = 0.0f;
                        for (int j = 0; j < 16; ++j) { dg += wg[j] * xp[j]; du += wu[j] * xp[j]; }
                        ag[i] += dg * scg; au[i] += du * scu;
                    }
                }
            } else if (ed.gu_code == FMT_MX8) {
                const uint8_t* wb = body;
                const uint8_t* sb = body + 5308416;
                const uint8_t* wg = wb + (size_t)(2 * r) * 2304;
                const uint8_t* sg_p = sb + (size_t)(2 * r) * 72;
                const uint8_t* wu = wb + (size_t)(2 * r + 1) * 2304;
                const uint8_t* su_p = sb + (size_t)(2 * r + 1) * 72;
                for (int b = lane; b < 72; b += 16) {
                    const float scg = ue8m0_to_f32(sg_p[b]);
                    const float scu = ue8m0_to_f32(su_p[b]);
                    const uint8_t* bg = wg + b * 32;
                    const uint8_t* bu = wu + b * 32;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + b * 32;
                        float dg = 0.0f, du = 0.0f;
                        for (int j = 0; j < 32; ++j) { dg += e4m3_to_f32(bg[j]) * xp[j]; du += e4m3_to_f32(bu[j]) * xp[j]; }
                        ag[i] += dg * scg; au[i] += du * scu;
                    }
                }
            } else if (ed.gu_code == FMT_GGUF_RAW) {
                const auto* hdr = reinterpret_cast<const strata::quant::sk::GgufExpertHeader*>(body);
                const uint8_t* gu = body + 32;
                constexpr size_t kRowBytes = 72u * 34u;
                const uint8_t* rg = gu + (size_t)(2 * r) * kRowBytes;
                const uint8_t* ru = gu + (size_t)(2 * r + 1) * kRowBytes;
                for (int b = lane; b < 72; b += 16) {
                    const float dg = h2f_bits(ld16(rg + b * 34)) * hdr->gate_scale;
                    const float du = h2f_bits(ld16(ru + b * 34)) * hdr->up_scale;
                    const uint8_t* pg = rg + b * 34 + 2;
                    const uint8_t* pu = ru + b * 34 + 2;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + b * 32;
                        float vg = 0.0f, vu = 0.0f;
                        for (int j = 0; j < 32; ++j) {
                            vg += (float)(int8_t)pg[j] * xp[j];
                            vu += (float)(int8_t)pu[j] * xp[j];
                        }
                        ag[i] += vg * dg; au[i] += vu * du;
                    }
                }
            } else {
                const int F = ed.gu_code;
                const int SB = sk_sb_bytes(F);
                const float H = sk_half(F);
                const size_t rowb = 9ul * SB;
                const uint8_t* gu = body + 2304;
                const uint8_t* rg = gu + (size_t)(2 * r) * rowb;
                const uint8_t* ru = gu + (size_t)(2 * r + 1) * rowb;
                for (int unit = lane; unit < 288; unit += 16) {
                    const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                    const uint8_t* bg = rg + (size_t)sbi * SB;
                    const uint8_t* bu = ru + (size_t)sbi * SB;
                    const float effg = h2f_bits(ld16(bg)) * ((float)bg[2 + g] * (1.0f / 255.0f));
                    const float effu = h2f_bits(ld16(bu)) * ((float)bu[2 + g] * (1.0f / 255.0f));
                    int qg[8], qu[8];
                    if (F == FMT_S2) { sk_q8_sy<FMT_S2>(bg + 6, j, qg); sk_q8_sy<FMT_S2>(bu + 6, j, qu); }
                    else if (F == FMT_S3) { sk_q8_sy<FMT_S3>(bg + 6, j, qg); sk_q8_sy<FMT_S3>(bu + 6, j, qu); }
                    else if (F == FMT_S4) { sk_q8_sy<FMT_S4>(bg + 6, j, qg); sk_q8_sy<FMT_S4>(bu + 6, j, qu); }
                    else if (F == FMT_S5) { sk_q8_sy<FMT_S5>(bg + 6, j, qg); sk_q8_sy<FMT_S5>(bu + 6, j, qu); }
                    else if (F == FMT_S6) { sk_q8_sy<FMT_S6>(bg + 6, j, qg); sk_q8_sy<FMT_S6>(bu + 6, j, qu); }
                    else { sk_q8_sy<FMT_S8>(bg + 6, j, qg); sk_q8_sy<FMT_S8>(bu + 6, j, qu); }
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + sbi * 256 + j;
                        float dg = 0.0f, du = 0.0f;
                        for (int t = 0; t < 8; ++t) { dg += ((float)qg[t] - H) * xp[t]; du += ((float)qu[t] - H) * xp[t]; }
                        ag[i] += dg * effg; au[i] += du * effu;
                    }
                }
            }

            for (int i = 0; i < mm; ++i) {
                const float g = sg_sum(it, ag[i]), u = sg_sum(it, au[i]);
                if (lane == 0) h_act[(size_t)pid[i] * 1152 + r] = situ_act(g, u) * inv_sq;
            }
        }
    });
}

// Down projection w2 (SF4 / MX8 / paired 2-row S_K SB_0..SB_8) * router weight -> y_pairs [P, 2304]
// Grid: 72 x max_uniq, 256 threads = 16 SG16 (each SG handles 1 row-pair = 2 output rows!)
inline void k_moe_dn_sg16(
    const InkSyExpertDev* ex_layer, const float* topk_w,
    const int* uexp, const int* uoff, const int* ucnt, const int* pair_list, const int* n_uniq_dev,
    int max_uniq, const float* h_act, float* y_pairs) {
    Q().parallel_for(NR2(72, (size_t)max_uniq, 256), [=](sycl::nd_item<2> it) INK_REQD_SG {
        const int yy = (int)it.get_group(0);
        if (yy >= n_uniq_dev[0]) return;
        const int sg = (int)it.get_local_id(1) >> 4, lane = (int)it.get_local_id(1) & 15;
        const int rp = (int)it.get_group(1) * 16 + sg;
        if (rp >= 1152) return;

        const int e = uexp[yy], off = uoff[yy], n = ucnt[yy];
        const InkSyExpertDev ed = ex_layer[e];
        const uint8_t* body = ed.body_ptr;

        for (int t0 = 0; t0 < n; t0 += 4) {
            const int mm = sycl::min(4, n - t0);
            int pid[4];
            for (int i = 0; i < mm; ++i) pid[i] = pair_list[off + t0 + i];
            float a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};

            if (ed.dn_code == FMT_SF4) {
                const float* hdr = reinterpret_cast<const float*>(body);
                const float outer = hdr[2] / hdr[3];
                const uint8_t* pb = body + ed.dn_off;
                const uint8_t* sb = pb + 1327104;
                const uint8_t* p0 = pb + (size_t)(2 * rp) * 576;
                const uint8_t* s0 = sb + (size_t)(2 * rp) * 72;
                const uint8_t* p1 = pb + (size_t)(2 * rp + 1) * 576;
                const uint8_t* s1 = sb + (size_t)(2 * rp + 1) * 72;
                for (int b = lane; b < 72; b += 16) {
                    const float sc0 = e4m3_to_f32(s0[b]) * outer;
                    const float sc1 = e4m3_to_f32(s1[b]) * outer;
                    const uint8_t* b0 = p0 + b * 8;
                    const uint8_t* b1 = p1 + b * 8;
                    float w0[16], w1[16];
                    for (int j = 0; j < 8; ++j) {
                        w0[2*j] = e2m1_lut(b0[j] & 15u); w0[2*j+1] = e2m1_lut(b0[j] >> 4);
                        w1[2*j] = e2m1_lut(b1[j] & 15u); w1[2*j+1] = e2m1_lut(b1[j] >> 4);
                    }
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + b * 16;
                        float d0 = 0.0f, d1 = 0.0f;
                        for (int j = 0; j < 16; ++j) { d0 += w0[j] * xp[j]; d1 += w1[j] * xp[j]; }
                        a0[i] += d0 * sc0; a1[i] += d1 * sc1;
                    }
                }
            } else if (ed.dn_code == FMT_MX8) {
                const uint8_t* wb = body + ed.dn_off;
                const uint8_t* sb = wb + 2654208;
                const uint8_t* w0 = wb + (size_t)(2 * rp) * 1152;
                const uint8_t* s0 = sb + (size_t)(2 * rp) * 36;
                const uint8_t* w1 = wb + (size_t)(2 * rp + 1) * 1152;
                const uint8_t* s1 = sb + (size_t)(2 * rp + 1) * 36;
                for (int b = lane; b < 36; b += 16) {
                    const float sc0 = ue8m0_to_f32(s0[b]);
                    const float sc1 = ue8m0_to_f32(s1[b]);
                    const uint8_t* b0 = w0 + b * 32;
                    const uint8_t* b1 = w1 + b * 32;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + b * 32;
                        float d0 = 0.0f, d1 = 0.0f;
                        for (int j = 0; j < 32; ++j) { d0 += e4m3_to_f32(b0[j]) * xp[j]; d1 += e4m3_to_f32(b1[j]) * xp[j]; }
                        a0[i] += d0 * sc0; a1[i] += d1 * sc1;
                    }
                }
            } else if (ed.dn_code == FMT_GGUF_RAW) {
                const auto* hdr = reinterpret_cast<const strata::quant::sk::GgufExpertHeader*>(body);
                const uint8_t* dnb = body + ed.dn_off;
                constexpr size_t kRowBytes = 36u * 34u;
                const uint8_t* r0 = dnb + (size_t)(2 * rp) * kRowBytes;
                const uint8_t* r1 = dnb + (size_t)(2 * rp + 1) * kRowBytes;
                for (int b = lane; b < 36; b += 16) {
                    const float s0 = h2f_bits(ld16(r0 + b * 34)) * hdr->dn_scale;
                    const float s1 = h2f_bits(ld16(r1 + b * 34)) * hdr->dn_scale;
                    const uint8_t* p0 = r0 + b * 34 + 2;
                    const uint8_t* p1 = r1 + b * 34 + 2;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + b * 32;
                        float v0 = 0.0f, v1 = 0.0f;
                        for (int j = 0; j < 32; ++j) {
                            v0 += (float)(int8_t)p0[j] * xp[j];
                            v1 += (float)(int8_t)p1[j] * xp[j];
                        }
                        a0[i] += v0 * s0; a1[i] += v1 * s1;
                    }
                }
            } else {
                const int F = ed.dn_code;
                const int SB = sk_sb_bytes(F);
                const float H = sk_half(F);
                const uint8_t* pr = body + ed.dn_off + (size_t)rp * 9ul * SB;
                for (int unit = lane; unit < 288; unit += 16) {
                    const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                    const uint8_t* blk = pr + (size_t)sbi * SB;
                    const float eff = h2f_bits(ld16(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
                    int q[8];
                    if (F == FMT_S2) sk_q8_sy<FMT_S2>(blk + 6, j, q);
                    else if (F == FMT_S3) sk_q8_sy<FMT_S3>(blk + 6, j, q);
                    else if (F == FMT_S4) sk_q8_sy<FMT_S4>(blk + 6, j, q);
                    else if (F == FMT_S5) sk_q8_sy<FMT_S5>(blk + 6, j, q);
                    else if (F == FMT_S6) sk_q8_sy<FMT_S6>(blk + 6, j, q);
                    else sk_q8_sy<FMT_S8>(blk + 6, j, q);
                    const bool is_r1 = (unit >= 144);
                    const int col = (is_r1 ? (unit - 144) : unit) << 3;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + col;
                        float s = 0.0f;
                        for (int t = 0; t < 8; ++t) s += ((float)q[t] - H) * xp[t];
                        if (is_r1) a1[i] += s * eff; else a0[i] += s * eff;
                    }
                }
            }

            for (int i = 0; i < mm; ++i) {
                const float v0 = sg_sum(it, a0[i]), v1 = sg_sum(it, a1[i]);
                if (lane == 0) {
                    const float rw = topk_w[pid[i]];
                    y_pairs[(size_t)pid[i] * 2304 + 2 * rp] = v0 * rw;
                    y_pairs[(size_t)pid[i] * 2304 + 2 * rp + 1] = v1 * rw;
                }
            }
        }
    });
}

inline void k_moe_reduce8_sg16(const float* y_pairs, float* moe_out, int M) {
    Q().parallel_for(NR2(9, (size_t)M, 256), [=](sycl::nd_item<2> it) INK_REQD_SG {
        const int m = (int)it.get_group(0);
        const int d = (int)it.get_group(1) * 256 + (int)it.get_local_id(1);
        if (d >= 2304) return;
        float s = 0.0f;
        for (int k = 0; k < 8; ++k) s += y_pairs[((size_t)m * 8 + k) * 2304 + d];
        moe_out[(size_t)m * 2304 + d] = s;
    });
}

}  // namespace ink_sycl
