// Copyright (c) 2026 Strata contributors
// SPDX-License-Identifier: MIT
//
// src/kernels/hip/hip_unsloth_wmma.cu - Clean-room AMD Wave32 WMMA kernels for RDNA3 (gfx1100..1151)
// and RDNA4 (gfx1200/1201):
//   1. FP16 & BF16 dense prefill GEMM:
//        - 16x16 single-wave (zero LDS, small T/N)
//        - 64x64 4-wave double-buffered LDS (medium T/N)
//        - 128x128 BK=64 8-wave register-prefetched grouped-tile (large prefill)
//   2. INT8 WMMA Q8_0 x Q8_1 verify GEMM (T = 1..16, gfx11* AND gfx12*)
//   3. Fused dequant-to-LDS FP16 + Wave32 WMMA prefill GEMM directly from quantized Unsloth weights:
//        Q4_K (12), Q5_K (13), Q6_K (14), Q8_0 (8), IQ4_XS (23), IQ3_XXS (18), NVFP4 (40).
//
// Compiles cleanly across --offload-arch=gfx1030, gfx1100, gfx1151, gfx1200, gfx1201.

#include "strata/kernels/hip_unsloth_wmma.hpp"
#include "strata/kernels/hip_unsloth_dp4a.hpp"

#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#include "strata/quant/strata-quant-common.h"

#include <atomic>
#include <cstdlib>

namespace strata::kernels::hip_unsloth {

namespace {

bool hip_device_is_wmma(int dev) {
    static std::atomic<int> cache[64];  // 0 unknown, 1 yes, 2 no
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load(std::memory_order_acquire);
    if (c == 0) {
        hipDeviceProp_t prop{};
        const bool ok = (hipGetDeviceProperties(&prop, dev) == hipSuccess) && gcn_has_wmma(prop.gcnArchName);
        c = ok ? 1 : 2;
        cache[dev].store(c, std::memory_order_release);
    }
    return c == 1;
}

// ===========================================================================
// 1. Dense FP16 / BF16 Wave32 WMMA GEMM (supports BOTH gfx11* and gfx12*)
//    Y[t, n] = beta * Y[t, n] + sum_k X[t, k] * W[n, k]
// ===========================================================================

template <typename ElemT>
__global__ void __launch_bounds__(32) hip_wmma_gemm_16x16_1w(
    const ElemT* __restrict__ X, int64_t ldx,
    const ElemT* __restrict__ W, int64_t ldw,
    float* __restrict__ Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {
#if STRATA_HIP_UNSLOTH_HAS_WMMA
    using Traits = WmmaWave32<ElemT>;
    using frag_t = typename Traits::frag_t;

    const int m_tile = static_cast<int>(blockIdx.y) * 16;
    const int n_tile = static_cast<int>(blockIdx.x) * 16;
    if (m_tile >= T || n_tile >= N) return;

    const int lane = threadIdx.x;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int k_sub = wmma_k_offset(hi);

    f32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const int m_row = m_tile + l16;
    const int n_row = n_tile + l16;

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        frag_t a_frag = Traits::zero_frag();
        frag_t b_frag = Traits::zero_frag();
        if (m_row < T) a_frag = Traits::load_frag(X + static_cast<int64_t>(m_row) * ldx + k_tile + k_sub);
        if (n_row < N) b_frag = Traits::load_frag(W + static_cast<int64_t>(n_row) * ldw + k_tile + k_sub);
        acc = Traits::mma(a_frag, b_frag, acc);
    }

    const int out_n = n_tile + l16;
    if (out_n < N) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_m = m_tile + wmma_d_row(i, hi);
            if (out_m < T) {
                float* dst = Y + static_cast<int64_t>(out_m) * ldy + out_n;
                *dst = (beta == 0.0f) ? acc[i] : (beta * (*dst) + acc[i]);
            }
        }
    }
#else
    (void) X; (void) ldx; (void) W; (void) ldw; (void) Y;
    (void) T; (void) N; (void) K; (void) ldy; (void) beta;
#endif
}

template <typename ElemT>
__global__ void __launch_bounds__(128) hip_wmma_gemm_64x64_4w(
    const ElemT* __restrict__ X, int64_t ldx,
    const ElemT* __restrict__ W, int64_t ldw,
    float* __restrict__ Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {
#if STRATA_HIP_UNSLOTH_HAS_WMMA
    using Traits = WmmaWave32<ElemT>;
    using frag_t = typename Traits::frag_t;

    const int m_tile = static_cast<int>(blockIdx.y) * 64;
    const int n_tile = static_cast<int>(blockIdx.x) * 64;
    if (m_tile >= T || n_tile >= N) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int k_sub = wmma_k_offset(hi);

    f32x8_t c0 = {0, 0, 0, 0, 0, 0, 0, 0};
    f32x8_t c1 = {0, 0, 0, 0, 0, 0, 0, 0};
    f32x8_t c2 = {0, 0, 0, 0, 0, 0, 0, 0};
    f32x8_t c3 = {0, 0, 0, 0, 0, 0, 0, 0};

    // Double-buffered LDS tile for W: 64 rows x 16 K (padded to 24 for zero bank conflicts)
    __shared__ ElemT b_lds[2][64][24] __attribute__((aligned(16)));

    const int r_ld = tid >> 1;
    const int k_ld = (tid & 1) << 3;
    const int actual_n = n_tile + r_ld;

    auto load_w = [&](int buf, int k_tile) {
        uint4 v = make_uint4(0, 0, 0, 0);
        if (actual_n < N && k_tile + k_ld < K) {
            v = *reinterpret_cast<const uint4*>(W + static_cast<int64_t>(actual_n) * ldw + k_tile + k_ld);
        }
        *reinterpret_cast<uint4*>(&b_lds[buf][r_ld][k_ld]) = v;
    };

    load_w(0, 0);
    hip_lds_barrier();

    const int m_row = m_tile + wave * 16 + l16;
    int buf = 0;

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        const int nb = buf ^ 1;
        if (k_tile + 16 < K) load_w(nb, k_tile + 16);

        frag_t a_frag = Traits::zero_frag();
        if (m_row < T) a_frag = Traits::load_frag(X + static_cast<int64_t>(m_row) * ldx + k_tile + k_sub);

        const frag_t b0 = Traits::load_frag(&b_lds[buf][0 + l16][k_sub]);
        const frag_t b1 = Traits::load_frag(&b_lds[buf][16 + l16][k_sub]);
        const frag_t b2 = Traits::load_frag(&b_lds[buf][32 + l16][k_sub]);
        const frag_t b3 = Traits::load_frag(&b_lds[buf][48 + l16][k_sub]);

        c0 = Traits::mma(a_frag, b0, c0);
        c1 = Traits::mma(a_frag, b1, c1);
        c2 = Traits::mma(a_frag, b2, c2);
        c3 = Traits::mma(a_frag, b3, c3);

        hip_lds_barrier();
        buf = nb;
    }

    const int m_base = m_tile + wave * 16;
    auto store_c = [&](const f32x8_t& acc, int n_base) {
        const int out_n = n_base + l16;
        if (out_n < N) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int out_m = m_base + wmma_d_row(i, hi);
                if (out_m < T) {
                    float* dst = Y + static_cast<int64_t>(out_m) * ldy + out_n;
                    *dst = (beta == 0.0f) ? acc[i] : (beta * (*dst) + acc[i]);
                }
            }
        }
    };
    store_c(c0, n_tile + 0);
    store_c(c1, n_tile + 16);
    store_c(c2, n_tile + 32);
    store_c(c3, n_tile + 48);
#else
    (void) X; (void) ldx; (void) W; (void) ldw; (void) Y;
    (void) T; (void) N; (void) K; (void) ldy; (void) beta;
#endif
}

// High-throughput 128 x 128, BK = 64, 8-wave (256 threads), grouped-tile order, register-prefetched
// prefill GEMM supporting BOTH gfx11* and gfx12* for FP16 and BF16.
template <typename ElemT>
__global__ void __launch_bounds__(256) hip_wmma_gemm_128x128_bk64(
    const ElemT* __restrict__ X, int64_t ldx,
    const ElemT* __restrict__ W, int64_t ldw,
    float* __restrict__ Y, int M, int N, int K, int64_t ldy, int accumulate) {
#if STRATA_HIP_UNSLOTH_HAS_WMMA
    using Traits = WmmaWave32<ElemT>;
    using frag_t = typename Traits::frag_t;

    constexpr int BM = 128, BN = 128, BK = 64, LDK = BK + 8, GM = 8;
    __shared__ ElemT sA[BM][LDK] __attribute__((aligned(16)));
    __shared__ ElemT sB[BN][LDK] __attribute__((aligned(16)));

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int k_sub = wmma_k_offset(hi);
    // 8 waves = 2 along M (64 rows each = 4x16) x 4 along N (32 cols each = 2x16)
    const int wm = wave & 1;
    const int wn = wave >> 1;

    const int num_m = (M + BM - 1) / BM;
    const int num_n = (N + BN - 1) / BN;
    const int b = static_cast<int>(blockIdx.x);
    const int group = b / (GM * num_n);
    const int first_m = group * GM;
    const int gsize = min(GM, num_m - first_m);
    const int rem = b % (GM * num_n);
    const int m0 = (first_m + rem % gsize) * BM;
    const int n0 = (rem / gsize) * BN;

    const int sr = tid >> 3;  // 0..31
    const int sq = tid & 7;   // 0..7 (8 x 8 = 64 elems)

    auto ldA = [&](int k0, int r) -> uint4 {
        const int row = m0 + r;
        if (row < M) return *reinterpret_cast<const uint4*>(X + static_cast<int64_t>(row) * ldx + k0 + 8 * sq);
        return make_uint4(0, 0, 0, 0);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        const int row = n0 + r;
        if (row < N) return *reinterpret_cast<const uint4*>(W + static_cast<int64_t>(row) * ldw + k0 + 8 * sq);
        return make_uint4(0, 0, 0, 0);
    };

    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3;
    auto load_tile = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
    };
    auto store_tile = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][8 * sq]) = ra0;
        *reinterpret_cast<uint4*>(&sA[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[sr + 64][8 * sq]) = ra2;
        *reinterpret_cast<uint4*>(&sA[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[sr][8 * sq]) = rb0;
        *reinterpret_cast<uint4*>(&sB[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[sr + 64][8 * sq]) = rb2;
        *reinterpret_cast<uint4*>(&sB[sr + 96][8 * sq]) = rb3;
    };

    f32x8_t acc[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) acc[i][j] = f32x8_t{0, 0, 0, 0, 0, 0, 0, 0};

    load_tile(0);
    store_tile();
    hip_lds_barrier();

    const int ar = 64 * wm + l16;
    const int br = 32 * wn + l16;

    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = (k0 + BK) < K;
        if (more) load_tile(k0 + BK);

#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            const frag_t a0 = Traits::load_frag(&sA[ar + 0][ks + k_sub]);
            const frag_t a1 = Traits::load_frag(&sA[ar + 16][ks + k_sub]);
            const frag_t a2 = Traits::load_frag(&sA[ar + 32][ks + k_sub]);
            const frag_t a3 = Traits::load_frag(&sA[ar + 48][ks + k_sub]);
            const frag_t b0 = Traits::load_frag(&sB[br + 0][ks + k_sub]);
            const frag_t b1 = Traits::load_frag(&sB[br + 16][ks + k_sub]);

            acc[0][0] = Traits::mma(a0, b0, acc[0][0]);
            acc[1][0] = Traits::mma(a1, b0, acc[1][0]);
            acc[2][0] = Traits::mma(a2, b0, acc[2][0]);
            acc[3][0] = Traits::mma(a3, b0, acc[3][0]);
            acc[0][1] = Traits::mma(a0, b1, acc[0][1]);
            acc[1][1] = Traits::mma(a1, b1, acc[1][1]);
            acc[2][1] = Traits::mma(a2, b1, acc[2][1]);
            acc[3][1] = Traits::mma(a3, b1, acc[3][1]);
        }

        if (more) {
            hip_lds_barrier();
            store_tile();
            hip_lds_barrier();
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int n = n0 + 32 * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + wmma_d_row(e, hi);
                if (m < M) {
                    float* dst = Y + static_cast<int64_t>(m) * ldy + n;
                    *dst = accumulate ? (*dst + acc[i][j][e]) : acc[i][j][e];
                }
            }
        }
    }
#else
    (void) X; (void) ldx; (void) W; (void) ldw; (void) Y;
    (void) M; (void) N; (void) K; (void) ldy; (void) accumulate;
#endif
}

template <typename ElemT>
bool dispatch_dense_wmma(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                         float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                         void* stream) {
    if (!X || !W || !Y || T <= 0 || N <= 0 || K <= 0) return false;
    if (!hip_wmma_device_available()) return false;
    if ((K % 16) != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    if (ldx <= 0) ldx = K;
    if (ldw <= 0) ldw = K;
    if (ldy <= 0) ldy = N;
    if (ldx < K || ldw < K || ldy < N || (ldx % 8) != 0 || (ldw % 8) != 0) return false;

    hipStream_t s = static_cast<hipStream_t>(stream);
    const auto* Xe = reinterpret_cast<const ElemT*>(X);
    const auto* We = reinterpret_cast<const ElemT*>(W);

    if (T >= 64 && N >= 128 && (K % 64) == 0 && T <= (1LL << 30) && N <= (1LL << 30)) {
        const unsigned grid = static_cast<unsigned>(((T + 127) / 128) * ((N + 127) / 128));
        hip_wmma_gemm_128x128_bk64<ElemT><<<grid, 256, 0, s>>>(
            Xe, ldx, We, ldw, Y, static_cast<int>(T), static_cast<int>(N), static_cast<int>(K),
            ldy, beta == 1.0f ? 1 : 0);
    } else if (T >= 32 && N >= 32) {
        dim3 grid(static_cast<unsigned>((N + 63) / 64), static_cast<unsigned>((T + 63) / 64), 1);
        hip_wmma_gemm_64x64_4w<ElemT><<<grid, 128, 0, s>>>(Xe, ldx, We, ldw, Y, T, N, K, ldy, beta);
    } else {
        dim3 grid(static_cast<unsigned>((N + 15) / 16), static_cast<unsigned>((T + 15) / 16), 1);
        hip_wmma_gemm_16x16_1w<ElemT><<<grid, 32, 0, s>>>(Xe, ldx, We, ldw, Y, T, N, K, ldy, beta);
    }
    return hipGetLastError() == hipSuccess;
}

// ===========================================================================
// 2. Wave32 INT8 WMMA Q8_0 x Q8_1 verify GEMM (T = 1..16, BOTH gfx11* and gfx12*)
//    Y[c * n_out + row] = sum_k W_q8_0[row, k] * X_q8_1[c, k]
// ===========================================================================
__global__ void __launch_bounds__(128) hip_wmma_q80_mmvq_kernel(
    const block_q8_0* __restrict__ W, const block_q8_1* __restrict__ X,
    float* __restrict__ Y, int n_in, int n_out, int ncols) {
#if STRATA_HIP_UNSLOTH_HAS_WMMA
    using Traits = WmmaWave32<int8_t>;
    using frag_t = typename Traits::frag_t;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int row0 = (static_cast<int>(blockIdx.x) * 4 + wave) * 16;
    const int nb = n_in / 32;

    __shared__ int s_x[16][8] __attribute__((aligned(16)));
    __shared__ float s_xd[16];

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int blk = 0; blk < nb; ++blk) {
        const int sc = tid >> 3;
        const int si = tid & 7;
        if (sc < ncols) {
            const block_q8_1* xb = X + static_cast<std::size_t>(sc) * nb + blk;
            s_x[sc][si] = reinterpret_cast<const int*>(xb->qs)[si];
            if (si == 0) s_xd[sc] = __low2float(xb->ds);
        } else {
            s_x[sc][si] = 0;
            if (si == 0) s_xd[sc] = 0.0f;
        }
        hip_lds_barrier();

        const int row = row0 + l16;
        float wd = 0.0f;
        i32x8_t ci = {0, 0, 0, 0, 0, 0, 0, 0};

#if STRATA_HIP_UNSLOTH_WMMA_GFX12
        // gfx12: lane (hi, l16) holds 8 int8 (2 int32) at k = 8*hi .. 8*hi+7 for each 16-wide WMMA
        frag_t a0 = {0, 0}, a1 = {0, 0};
        if (row < n_out) {
            const block_q8_0* wb = W + static_cast<std::size_t>(row) * nb + blk;
            wd = __half2float(wb->d);
            const int base = 2 * hi;
            a0[0] = hip_ld_b2(wb->qs, base + 0);
            a0[1] = hip_ld_b2(wb->qs, base + 1);
            a1[0] = hip_ld_b2(wb->qs, 4 + base + 0);
            a1[1] = hip_ld_b2(wb->qs, 4 + base + 1);
        }
        const int xb_base = 2 * hi;
        const frag_t b0 = {s_x[l16][xb_base + 0], s_x[l16][xb_base + 1]};
        const frag_t b1 = {s_x[l16][4 + xb_base + 0], s_x[l16][4 + xb_base + 1]};
        ci = Traits::mma(a0, b0, ci);
        ci = Traits::mma(a1, b1, ci);
#elif STRATA_HIP_UNSLOTH_WMMA_GFX11
        // gfx11: lane (hi, l16) holds all 16 int8 (4 int32), duplicated across halves
        frag_t a0 = {0, 0, 0, 0}, a1 = {0, 0, 0, 0};
        if (row < n_out) {
            const block_q8_0* wb = W + static_cast<std::size_t>(row) * nb + blk;
            wd = __half2float(wb->d);
            a0[0] = hip_ld_b2(wb->qs, 0); a0[1] = hip_ld_b2(wb->qs, 1);
            a0[2] = hip_ld_b2(wb->qs, 2); a0[3] = hip_ld_b2(wb->qs, 3);
            a1[0] = hip_ld_b2(wb->qs, 4); a1[1] = hip_ld_b2(wb->qs, 5);
            a1[2] = hip_ld_b2(wb->qs, 6); a1[3] = hip_ld_b2(wb->qs, 7);
        }
        const frag_t b0 = {s_x[l16][0], s_x[l16][1], s_x[l16][2], s_x[l16][3]};
        const frag_t b1 = {s_x[l16][4], s_x[l16][5], s_x[l16][6], s_x[l16][7]};
        ci = Traits::mma(a0, b0, ci);
        ci = Traits::mma(a1, b1, ci);
#endif

        const float xd = s_xd[l16];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int d_row = wmma_d_row(k, hi);
            const float wdk = __shfl(wd, d_row, 32);
            acc[k] = fmaf(wdk * xd, static_cast<float>(ci[k]), acc[k]);
        }
        hip_lds_barrier();
    }

    if (l16 < ncols) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int out_row = row0 + wmma_d_row(k, hi);
            if (out_row < n_out) {
                Y[static_cast<std::size_t>(l16) * n_out + out_row] = acc[k];
            }
        }
    }
#else
    (void) W; (void) X; (void) Y; (void) n_in; (void) n_out; (void) ncols;
#endif
}

// ===========================================================================
// 3. Fused Dequant-to-LDS FP16 + Wave32 WMMA Prefill GEMM
//    Covers Q4_K (12), Q5_K (13), Q6_K (14), Q8_0 (8), IQ4_XS (23), IQ3_XXS (18), NVFP4 (40)
//    Block: 64 M-rows (tokens) x 64 N-rows (weights), BK = 64, 4 waves (128 threads).
//    Every thread dequantizes 32 consecutive FP16 weight values per K-stage (128 * 32 = 64 * 64).
// ===========================================================================

template <int TY>
__device__ __forceinline__ void dequant32_to_f16(
    const uint8_t* __restrict__ row_ptr, int k_stage, int sub_half,
    _Float16* __restrict__ dst32, const uint32_t* __restrict__ s_iq3xxs) {
    // k_stage indexes 64-element stages; sub_half in {0, 1} selects 32-element half within stage.
    const int sb32 = k_stage * 2 + sub_half;  // 32-element sub-block index along K
    const int kbx256 = sb32 >> 3;             // 256-element super-block index
    const int j = sb32 & 7;                   // 0..7 sub-block within 256 super-block

    if constexpr (TY == 12) {
        // Q4_K (144 B / 256)
        const auto* b = reinterpret_cast<const block_q4_K*>(row_ptr) + kbx256;
        const float2 dm = __half22float2(b->dm);
        int sc, m;
        if (j < 4) {
            sc = b->scales[j] & 63;
            m = b->scales[j + 4] & 63;
        } else {
            sc = (b->scales[j + 4] & 0xF) | ((b->scales[j - 4] >> 6) << 4);
            m = (b->scales[j + 4] >> 4) | ((b->scales[j] >> 6) << 4);
        }
        const float d_sc = dm.x * static_cast<float>(sc);
        const float m_sc = dm.y * static_cast<float>(m);
        const uint8_t* qs = b->qs + (j >> 1) * 32;
        const int shift = (j & 1) * 4;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int q = (qs[i] >> shift) & 0xF;
            dst32[i] = static_cast<_Float16>(fmaf(d_sc, static_cast<float>(q), -m_sc));
        }
    } else if constexpr (TY == 13) {
        // Q5_K (176 B / 256)
        const auto* b = reinterpret_cast<const block_q5_K*>(row_ptr) + kbx256;
        const float2 dm = __half22float2(b->dm);
        int sc, m;
        if (j < 4) {
            sc = b->scales[j] & 63;
            m = b->scales[j + 4] & 63;
        } else {
            sc = (b->scales[j + 4] & 0xF) | ((b->scales[j - 4] >> 6) << 4);
            m = (b->scales[j + 4] >> 4) | ((b->scales[j] >> 6) << 4);
        }
        const float d_sc = dm.x * static_cast<float>(sc);
        const float m_sc = dm.y * static_cast<float>(m);
        const uint8_t* qs = b->qs + (j >> 1) * 32;
        const int shift = (j & 1) * 4;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int q = ((qs[i] >> shift) & 0xF) | (((b->qh[i] >> j) & 1) << 4);
            dst32[i] = static_cast<_Float16>(fmaf(d_sc, static_cast<float>(q), -m_sc));
        }
    } else if constexpr (TY == 14) {
        // Q6_K (210 B / 256): n = j >> 2 (0..1), qd = j & 3 (0..3)
        const auto* b = reinterpret_cast<const block_q6_K*>(row_ptr) + kbx256;
        const float d = __half2float(b->d);
        const int n = j >> 2;
        const int qd = j & 3;
        const uint8_t* ql = b->ql + n * 64 + (qd & 1) * 32;
        const uint8_t* qh = b->qh + n * 32;
        const int ns = (qd >= 2) ? 4 : 0;
        const int hs = 2 * qd;
        const float d0 = d * static_cast<float>(b->scales[n * 8 + 2 * qd + 0]);
        const float d1 = d * static_cast<float>(b->scales[n * 8 + 2 * qd + 1]);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q0 = ((ql[i] >> ns) & 0xF) | (((qh[i] >> hs) & 3) << 4);
            const int q1 = ((ql[i + 16] >> ns) & 0xF) | (((qh[i + 16] >> hs) & 3) << 4);
            dst32[i] = static_cast<_Float16>(d0 * static_cast<float>(q0 - 32));
            dst32[i + 16] = static_cast<_Float16>(d1 * static_cast<float>(q1 - 32));
        }
    } else if constexpr (TY == 8) {
        // Q8_0 (34 B / 32)
        const auto* b = reinterpret_cast<const block_q8_0*>(row_ptr) + sb32;
        const float d = __half2float(b->d);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            dst32[i] = static_cast<_Float16>(d * static_cast<float>(b->qs[i]));
        }
    } else if constexpr (TY == 23) {
        // IQ4_XS (136 B / 256)
        const auto* b = reinterpret_cast<const block_iq4_xs*>(row_ptr) + kbx256;
        const int ls = ((b->scales_l[j >> 1] >> (4 * (j & 1))) & 0xF) |
                       (((b->scales_h >> (2 * j)) & 3) << 4);
        const float dl = __half2float(b->d) * static_cast<float>(ls - 32);
        const uint8_t* qs = b->qs + 16 * j;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t q4 = static_cast<uint32_t>(hip_ld_b4(qs, i));
            const int2 v = hip_iq4_lut16(q4);
            const auto* lo = reinterpret_cast<const int8_t*>(&v.x);
            const auto* hi = reinterpret_cast<const int8_t*>(&v.y);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                dst32[4 * i + k] = static_cast<_Float16>(dl * static_cast<float>(lo[k]));
                dst32[16 + 4 * i + k] = static_cast<_Float16>(dl * static_cast<float>(hi[k]));
            }
        }
    } else if constexpr (TY == 18) {
        // IQ3_XXS (98 B / 256)
        const auto* b = reinterpret_cast<const block_iq3_xxs*>(row_ptr) + kbx256;
        const uint32_t aux32 = static_cast<uint32_t>(hip_ld_b2(b->qs, 16 + j));
        const int ls = static_cast<int>(aux32 >> 28);
        const float dl = __half2float(b->d) * (0.5f + static_cast<float>(ls)) * 0.5f;
        const uint8_t* qs = b->qs + 8 * j;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t g0 = s_iq3xxs[qs[l0 + 0]];
            const uint32_t g1 = s_iq3xxs[qs[l0 + 1]];
            const uint32_t v7 = (aux32 >> (7 * (l0 >> 1))) & 0x7Fu;
            const uint32_t s8 = v7 | ((__popc(v7) & 1u) << 7);
            const int w0 = hip_iq_sign_negate(g0, s8 & 0xFu);
            const int w1 = hip_iq_sign_negate(g1, s8 >> 4);
            const auto* b0 = reinterpret_cast<const int8_t*>(&w0);
            const auto* b1 = reinterpret_cast<const int8_t*>(&w1);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                dst32[4 * l0 + k] = static_cast<_Float16>(dl * static_cast<float>(b0[k]));
                dst32[4 * l0 + 4 + k] = static_cast<_Float16>(dl * static_cast<float>(b1[k]));
            }
        }
    } else if constexpr (TY == 40) {
        // NVFP4 (36 B / 64): k_stage is block_nvfp4 index; sub_half selects sub-blocks (0,1) vs (2,3)
        const auto* b = reinterpret_cast<const block_nvfp4*>(row_ptr) + k_stage;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int is = sub_half * 2 + s;
            const float sc = hip_ue4m3_to_f32_half(b->d[is]);
            const int2 v0 = hip_fp4_lut16(static_cast<uint32_t>(hip_ld_b4(b->qs, 2 * is + 0)));
            const int2 v1 = hip_fp4_lut16(static_cast<uint32_t>(hip_ld_b4(b->qs, 2 * is + 1)));
            const auto* x0 = reinterpret_cast<const int8_t*>(&v0.x);
            const auto* y0 = reinterpret_cast<const int8_t*>(&v0.y);
            const auto* x1 = reinterpret_cast<const int8_t*>(&v1.x);
            const auto* y1 = reinterpret_cast<const int8_t*>(&v1.y);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                dst32[16 * s + 0 + k] = static_cast<_Float16>(sc * static_cast<float>(x0[k]));
                dst32[16 * s + 4 + k] = static_cast<_Float16>(sc * static_cast<float>(x1[k]));
                dst32[16 * s + 8 + k] = static_cast<_Float16>(sc * static_cast<float>(y0[k]));
                dst32[16 * s + 12 + k] = static_cast<_Float16>(sc * static_cast<float>(y1[k]));
            }
        }
    }
    (void) s_iq3xxs;
}

template <int TY>
__global__ void __launch_bounds__(128) hip_wmma_dequant_prefill_kernel(
    const uint8_t* __restrict__ W, std::size_t row_bytes,
    const _Float16* __restrict__ X, int64_t ldx,
    float* __restrict__ Y, int T, int N, int K, int64_t ldy, float global_scale) {
#if STRATA_HIP_UNSLOTH_HAS_WMMA
    using Traits = WmmaWave32<_Float16>;
    using frag_t = typename Traits::frag_t;

    constexpr int BM = 64, BN = 64, BK = 64, LDK = BK + 8;
    __shared__ _Float16 sW[BN][LDK] __attribute__((aligned(16)));
    __shared__ _Float16 sX[BM][LDK] __attribute__((aligned(16)));
    __shared__ uint32_t s_grid[TY == 18 ? 256 : 1];

    const int tid = threadIdx.x;
    if constexpr (TY == 18) {
        for (int i = tid; i < 256; i += 128) s_grid[i] = iq3xxs_grid[i];
    }

    const int m0 = static_cast<int>(blockIdx.y) * BM;
    const int n0 = static_cast<int>(blockIdx.x) * BN;
    if (m0 >= T || n0 >= N) return;

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int k_sub = wmma_k_offset(hi);

    // Weight dequant assignment: 128 threads -> 64 rows x 2 halves of 32 elements each
    const int w_r = tid >> 1;
    const int w_h = tid & 1;
    const int global_n = n0 + w_r;
    const uint8_t* w_row_ptr = (global_n < N) ? (W + static_cast<std::size_t>(global_n) * row_bytes) : nullptr;

    // Activation load assignment: 128 threads -> 16 rows/pass x 8 uint4 (64 halves), 4 passes = 64 rows
    const int x_r = tid >> 3;  // 0..15
    const int x_q = tid & 7;   // 0..7

    f32x8_t acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = f32x8_t{0, 0, 0, 0, 0, 0, 0, 0};

    if constexpr (TY == 18) hip_lds_barrier();

    const int n_stages = K / BK;
    for (int st = 0; st < n_stages; ++st) {
        // 1. Cooperative dequantize 64 x 64 weight tile into sW
        _Float16* wdst = &sW[w_r][w_h * 32];
        if (w_row_ptr != nullptr) {
            dequant32_to_f16<TY>(w_row_ptr, st, w_h, wdst, s_grid);
        } else {
#pragma unroll
            for (int i = 0; i < 32; ++i) wdst[i] = static_cast<_Float16>(0);
        }

        // 2. Cooperative load 64 x 64 FP16 activation tile into sX
        const int k0 = st * BK;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int r = p * 16 + x_r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (m0 + r < T) {
                v = *reinterpret_cast<const uint4*>(X + static_cast<int64_t>(m0 + r) * ldx + k0 + 8 * x_q);
            }
            *reinterpret_cast<uint4*>(&sX[r][8 * x_q]) = v;
        }
        hip_lds_barrier();

        // 3. Each wave computes 16 M-rows x 64 N-rows across 4 x 16 K-steps
        const int ar = wave * 16 + l16;
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            const frag_t a = Traits::load_frag(&sX[ar][ks + k_sub]);
            const frag_t b0 = Traits::load_frag(&sW[0 + l16][ks + k_sub]);
            const frag_t b1 = Traits::load_frag(&sW[16 + l16][ks + k_sub]);
            const frag_t b2 = Traits::load_frag(&sW[32 + l16][ks + k_sub]);
            const frag_t b3 = Traits::load_frag(&sW[48 + l16][ks + k_sub]);
            acc[0] = Traits::mma(a, b0, acc[0]);
            acc[1] = Traits::mma(a, b1, acc[1]);
            acc[2] = Traits::mma(a, b2, acc[2]);
            acc[3] = Traits::mma(a, b3, acc[3]);
        }
        hip_lds_barrier();
    }

    const int m_base = m0 + wave * 16;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int out_n = n0 + 16 * j + l16;
        if (out_n >= N) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_m = m_base + wmma_d_row(i, hi);
            if (out_m < T) {
                Y[static_cast<int64_t>(out_m) * ldy + out_n] = acc[j][i] * global_scale;
            }
        }
    }
#else
    (void) W; (void) row_bytes; (void) X; (void) ldx; (void) Y;
    (void) T; (void) N; (void) K; (void) ldy; (void) global_scale;
#endif
}

}  // namespace

// ===========================================================================
// Public Host API
// ===========================================================================

bool hip_wmma_device_available() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return false;
    return hip_device_is_wmma(dev);
}

bool hip_exp8_auto_enabled(int dev) {
    static std::atomic<int> cache[64];  // 0 unknown, 1 no, 2 yes
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load(std::memory_order_acquire);
    if (c == 0) {
        const char* v_all = std::getenv("STRATA_HIP_EXP8");
        const char* v1151 = std::getenv("STRATA_GFX1151_EXP8");
        const char* v11 = std::getenv("STRATA_GFX11_EXP8");
        const char* v12 = std::getenv("STRATA_GFX12_EXP8");
        const char* off11 = std::getenv("STRATA_GFX11_DEFAULTS");
        const char* v1030 = std::getenv("STRATA_GFX1030_EXP8");

        hipDeviceProp_t prop{};
        bool yes = false;
        if (hipGetDeviceProperties(&prop, dev) == hipSuccess) {
            const bool global_off = (v_all != nullptr && v_all[0] == '0');
            if (!global_off) {
                if (gcn_is_gfx12_wmma(prop.gcnArchName)) {
                    yes = !(v12 != nullptr && v12[0] == '0');
                } else if (gcn_is(prop.gcnArchName, "gfx1151")) {
                    yes = !(v1151 != nullptr && v1151[0] == '0') && !(v11 != nullptr && v11[0] == '0');
                } else if (gcn_is_gfx11_wmma(prop.gcnArchName)) {
                    yes = !(v1151 != nullptr && v1151[0] == '0') && !(v11 != nullptr && v11[0] == '0') &&
                          !(off11 != nullptr && off11[0] == '0');
                } else if (gcn_is_rdna2(prop.gcnArchName)) {
                    yes = (v1030 != nullptr && v1030[0] == '1') || (v_all != nullptr && v_all[0] == '1');
                }
            }
        }
        c = yes ? 2 : 1;
        cache[dev].store(c, std::memory_order_release);
    }
    return c == 2;
}

bool hip_unsloth_wmma_gemm_f16(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                               float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                               void* stream) {
    return dispatch_dense_wmma<_Float16>(X, ldx, W, ldw, Y, T, N, K, ldy, beta, stream);
}

bool hip_unsloth_wmma_gemm_bf16(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                                float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                                void* stream) {
    return dispatch_dense_wmma<__bf16>(X, ldx, W, ldw, Y, T, N, K, ldy, beta, stream);
}

bool hip_unsloth_wmma_q80_mmvq(const void* W_q8_0, const void* X_q8_1, float* Y,
                               int n_in, int n_out, int ncols, void* stream) {
    if (!W_q8_0 || !X_q8_1 || !Y || n_in <= 0 || n_out <= 0) return false;
    if ((n_in % 32) != 0 || ncols < 1 || ncols > 16) return false;
    if (!hip_wmma_device_available()) return false;

    const unsigned blocks = static_cast<unsigned>((n_out + 63) / 64);
    hip_wmma_q80_mmvq_kernel<<<blocks, 128, 0, static_cast<hipStream_t>(stream)>>>(
        static_cast<const block_q8_0*>(W_q8_0), static_cast<const block_q8_1*>(X_q8_1),
        Y, n_in, n_out, ncols);
    return hipGetLastError() == hipSuccess;
}

bool hip_unsloth_wmma_dequant_prefill(int ggml_type, const void* W_quant, const uint16_t* X_f16,
                                      int64_t ldx, float* Y, int64_t T, int64_t N, int64_t K,
                                      int64_t ldy, float global_scale, void* stream) {
    if (!W_quant || !X_f16 || !Y || T <= 0 || N <= 0 || K <= 0) return false;
    if (!hip_wmma_device_available()) return false;
    if (ldx <= 0) ldx = K;
    if (ldy <= 0) ldy = N;
    if (ldx < K || ldy < N || (ldx % 8) != 0 || (K % 64) != 0) return false;
    const std::size_t rb = hip_quant_row_bytes(ggml_type, K);
    if (rb == 0u) return false;

    dim3 grid(static_cast<unsigned>((N + 63) / 64), static_cast<unsigned>((T + 63) / 64), 1);
    if (grid.y > 65535u) return false;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const auto* wb = static_cast<const uint8_t*>(W_quant);
    const auto* xf = reinterpret_cast<const _Float16*>(X_f16);

    switch (ggml_type) {
        case 8:
            hip_wmma_dequant_prefill_kernel<8><<<grid, 128, 0, s>>>(
                wb, rb, xf, ldx, Y, static_cast<int>(T), static_cast<int>(N), static_cast<int>(K), ldy, global_scale);
            break;
        case 12:
            hip_wmma_dequant_prefill_kernel<12><<<grid, 128, 0, s>>>(
                wb, rb, xf, ldx, Y, static_cast<int>(T), static_cast<int>(N), static_cast<int>(K), ldy, global_scale);
            break;
        case 13:
            hip_wmma_dequant_prefill_kernel<13><<<grid, 128, 0, s>>>(
                wb, rb, xf, ldx, Y, static_cast<int>(T), static_cast<int>(N), static_cast<int>(K), ldy, global_scale);
            break;
        case 14:
            hip_wmma_dequant_prefill_kernel<14><<<grid, 128, 0, s>>>(
                wb, rb, xf, ldx, Y, static_cast<int>(T), static_cast<int>(N), static_cast<int>(K), ldy, global_scale);
            break;
        case 18:
            hip_wmma_dequant_prefill_kernel<18><<<grid, 128, 0, s>>>(
                wb, rb, xf, ldx, Y, static_cast<int>(T), static_cast<int>(N), static_cast<int>(K), ldy, global_scale);
            break;
        case 23:
            hip_wmma_dequant_prefill_kernel<23><<<grid, 128, 0, s>>>(
                wb, rb, xf, ldx, Y, static_cast<int>(T), static_cast<int>(N), static_cast<int>(K), ldy, global_scale);
            break;
        case 40:
            hip_wmma_dequant_prefill_kernel<40><<<grid, 128, 0, s>>>(
                wb, rb, xf, ldx, Y, static_cast<int>(T), static_cast<int>(N), static_cast<int>(K), ldy, global_scale);
            break;
        default:
            return false;
    }
    return hipGetLastError() == hipSuccess;
}

}  // namespace strata::kernels::hip_unsloth
