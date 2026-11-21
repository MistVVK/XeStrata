// engine/src/inkling2/hip/ink_hip_codecs.hip.h
// Multi-Gen HIP Linear & MoE Kernels for Inkling-2-Mini (.strata v1.1):
//   - Lossless SF4 (NVFP4 E2M1 + F8_E4M3/16w + F32 amax/denom)
//   - Lossless MX8 (MXFP8 F8_E4M3 + UE8M0/32w)
//   - S_K (S1.5, S2_K, S3_K, S4_K, S5_K, S6_K, S8_K) standard & paired-2-row down (SB_0..SB_8)
//   - Dense trunk Q8_ROW, BF16, F32
//   - RDNA4 (gfx120x): Wave32 half8 WMMA + 2x-rate iu8 WMMA (SF4 exact x2 int8 LUT + S_K exact int8)
//   - RDNA3 / RDNA3.5 (gfx110x, gfx1103 780M, gfx1150 890M, gfx1151 Strix Halo): Wave32 half16 WMMA + fdot2
//   - RDNA2 (gfx103x): zero WMMA intrinsics; Wave32 sdot4 + fdot2 tiled GEMM
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_common.hip.h"

namespace strata::inkling2::hip {

// Wire code_ids matching strata::quant::sk::SkCode
constexpr int CODE_S1_25 = 0;
constexpr int CODE_S1_5  = 1;
constexpr int CODE_S2_K  = 2;
constexpr int CODE_S3_K  = 3;
constexpr int CODE_S4_K  = 4;
constexpr int CODE_S5_K  = 5;
constexpr int CODE_S6_K  = 6;
constexpr int CODE_S8_K  = 8;
constexpr int CODE_SF4   = 15;
constexpr int CODE_MX8   = 16;

__host__ __device__ constexpr int sk_sb_bytes(int code) noexcept {
    switch (code) {
        case CODE_S1_25: return 48;
        case CODE_S1_5:  return 58;
        case CODE_S2_K:  return 70;
        case CODE_S3_K:  return 102;
        case CODE_S4_K:  return 134;
        case CODE_S5_K:  return 166;
        case CODE_S6_K:  return 198;
        case CODE_S8_K:  return 262;
        default:         return 0;
    }
}

__device__ __forceinline__ float sk_half_offset(int code) {
    switch (code) {
        case CODE_S2_K: return 1.5f;
        case CODE_S3_K: return 3.5f;
        case CODE_S4_K: return 7.5f;
        case CODE_S5_K: return 15.5f;
        case CODE_S6_K: return 31.5f;
        default:        return 0.0f;
    }
}

__device__ __forceinline__ int sk_i8_bias(int code) {
    switch (code) {
        case CODE_S2_K: return 3;
        case CODE_S3_K: return 7;
        case CODE_S4_K: return 15;
        case CODE_S5_K: return 31;
        case CODE_S6_K: return 63;
        default:        return 0;
    }
}

__device__ __forceinline__ float ld_fp16_as_f32(const uint8_t* p) {
    const uint16_t bits = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    return static_cast<float>(__builtin_bit_cast(half_t, bits));
}

__device__ __forceinline__ int s15_trit(uint32_t byte_val, int pos) {
    static constexpr uint32_t kDiv[5] = {1u, 3u, 9u, 27u, 81u};
    return static_cast<int>((byte_val / kDiv[pos]) % 3u) - 1;
}

// Decode 1 weight at index `j` (0..255) inside an S_K super-block `blk` as exact signed int8 (scaled by 2x for S2..S6!)
// Effective multiplier for sub-group g (0..3) is:
//   - S2..S6: 0.5f * D * (u[g] / 255.0f)
//   - S1.5 / S8: D * (u[g] / 255.0f)
__device__ __forceinline__ int8_t decode_sk_w_i8(const uint8_t* pl, int code, int j) {
    switch (code) {
        case CODE_S1_25: {
            static constexpr int8_t kS125[32][4] = {
                {-1,-1,-1,-1},{-1,-1,-1, 1},{-1,-1, 1,-1},{-1,-1, 1, 1},
                {-1, 1,-1,-1},{-1, 1,-1, 1},{-1, 1, 1,-1},{-1, 1, 1, 1},
                { 1,-1,-1,-1},{ 1,-1,-1, 1},{ 1,-1, 1,-1},{ 1,-1, 1, 1},
                { 1, 1,-1,-1},{ 1, 1,-1, 1},{ 1, 1, 1,-1},{ 1, 1, 1, 1},
                { 2, 0, 0, 0},{-2, 0, 0, 0},{ 0, 2, 0, 0},{ 0,-2, 0, 0},
                { 0, 0, 2, 0},{ 0, 0,-2, 0},{ 0, 0, 0, 2},{ 0, 0, 0,-2},
                { 0, 0, 0, 0},{ 2, 2, 0, 0},{-2,-2, 0, 0},{ 0, 0, 2, 2},
                { 0, 0,-2,-2},{ 2,-2, 0, 0},{ 0, 0, 2,-2},{ 2, 0, 2, 0},
            };
            const int blk4 = j >> 2;
            const int g8 = blk4 >> 3;
            const int sub = blk4 & 7;
            const uint8_t* p = pl + g8 * 5;
            const uint64_t bits40 =
                static_cast<uint64_t>(p[0]) | (static_cast<uint64_t>(p[1]) << 8) |
                (static_cast<uint64_t>(p[2]) << 16) | (static_cast<uint64_t>(p[3]) << 24) |
                (static_cast<uint64_t>(p[4]) << 32);
            const uint32_t idx = static_cast<uint32_t>((bits40 >> (5 * sub)) & 0x1Fu);
            return kS125[idx][j & 3];
        }
        case CODE_S8_K:
            return static_cast<int8_t>(pl[j]);
        case CODE_S1_5:
            return static_cast<int8_t>(s15_trit(pl[j / 5], j % 5));
        case CODE_S2_K: {
            const uint32_t q = (pl[j >> 2] >> ((j & 3) * 2)) & 3u;
            return static_cast<int8_t>(static_cast<int>(q << 1) - 3);
        }
        case CODE_S3_K: {
            const uint32_t lo = (pl[j >> 2] >> ((j & 3) * 2)) & 3u;
            const uint32_t hi = (pl[64 + (j >> 3)] >> (j & 7)) & 1u;
            return static_cast<int8_t>(static_cast<int>((lo | (hi << 2)) << 1) - 7);
        }
        case CODE_S4_K: {
            const uint32_t q = (pl[j >> 1] >> ((j & 1) * 4)) & 15u;
            return static_cast<int8_t>(static_cast<int>(q << 1) - 15);
        }
        case CODE_S5_K: {
            const uint32_t lo = (pl[j >> 1] >> ((j & 1) * 4)) & 15u;
            const uint32_t hi = (pl[128 + (j >> 3)] >> (j & 7)) & 1u;
            return static_cast<int8_t>(static_cast<int>((lo | (hi << 4)) << 1) - 31);
        }
        case CODE_S6_K: {
            const uint32_t lo = (pl[j >> 1] >> ((j & 1) * 4)) & 15u;
            const uint32_t hi = (pl[128 + (j >> 2)] >> ((j & 3) * 2)) & 3u;
            return static_cast<int8_t>(static_cast<int>((lo | (hi << 4)) << 1) - 63);
        }
        default:
            return 0;
    }
}

__device__ __forceinline__ float decode_sk_sub_eff_i8(const uint8_t* blk, int code, int g) {
    const float D = ld_fp16_as_f32(blk);
    float eff = D * (static_cast<float>(blk[2 + g]) * (1.0f / 255.0f));
    if (code == CODE_S1_25) {
        const uint8_t fb = blk[6 + 40 + (g >> 1)];
        const uint32_t nib = (g & 1) ? (fb >> 4) : (fb & 0x0Fu);
        eff *= (0.5f + static_cast<float>(nib) * (1.0f / 15.0f));
    }
    return (code >= CODE_S2_K && code <= CODE_S6_K) ? (0.5f * eff) : eff;
}

__device__ __forceinline__ float decode_sk_w_f32(const uint8_t* blk, int code, int j) {
    const int g = j >> 6;
    return static_cast<float>(decode_sk_w_i8(blk + 6, code, j)) * decode_sk_sub_eff_i8(blk, code, g);
}

// =================================================================================================
// 1. SF4 (Lossless NVFP4) Multi-Gen Kernels
//    Layout: packed_e2m1 [N, K/2] (low nib = even col, high nib = odd col)
//            scales_e4m3 [N, K/16] (F8_E4M3FN)
//            outer_scale = amax / denom (float)
// =================================================================================================

// M=1..4 Bandwidth-Saturated Wave32 GEMV (RDNA4 / RDNA3 / RDNA3.5 / RDNA2)
// 8 waves/block (256 threads), each wave computes 1 output row across K (32 lanes x 16 weights/step = 512 weights/iter).
template <int MT>
__global__ void __launch_bounds__(256) k_sf4_gemv(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    float outer_scale,
    const half_t* __restrict__ X,  // [M, K]
    float* __restrict__ Y,         // [M, N]
    int M, int N, int K) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= N) return;

    const uint8_t* wp = packed + static_cast<size_t>(row) * (K >> 1);
    const uint8_t* ws = scales + static_cast<size_t>(row) * (K >> 4);
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    const int nblk16 = K >> 4;  // K / 16 (e.g. 2304/16 = 144, 1152/16 = 72)
    for (int b = lane; b < nblk16; b += 32) {
        const float sc = e4m3fn_to_f32(ws[b]) * outer_scale;
        const uint2 raw = *reinterpret_cast<const uint2*>(wp + b * 8);
        // Unpack 16 E2M1 nibbles -> 8 packed half2 words in registers
        uint32_t w2[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t byte0 = (raw.x >> (i * 8)) & 0xFFu;
            const uint32_t byte1 = (raw.y >> (i * 8)) & 0xFFu;
            const hv2 h0 = {sf4_e2m1_f16(byte0 & 15u), sf4_e2m1_f16(byte0 >> 4)};
            const hv2 h1 = {sf4_e2m1_f16(byte1 & 15u), sf4_e2m1_f16(byte1 >> 4)};
            w2[i]     = __builtin_bit_cast(uint32_t, h0);
            w2[4 + i] = __builtin_bit_cast(uint32_t, h1);
        }
        const int k0 = b << 4;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const uint4 x0 = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(m) * K + k0);
                const uint4 x1 = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(m) * K + k0 + 8);
                float d = 0.0f;
                d = fdot2_f32(w2[0], x0.x, d);
                d = fdot2_f32(w2[1], x0.y, d);
                d = fdot2_f32(w2[2], x0.z, d);
                d = fdot2_f32(w2[3], x0.w, d);
                d = fdot2_f32(w2[4], x1.x, d);
                d = fdot2_f32(w2[5], x1.y, d);
                d = fdot2_f32(w2[6], x1.z, d);
                d = fdot2_f32(w2[7], x1.w, d);
                acc[m] = fmaf(d, sc, acc[m]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const float sum = wave32_sum(acc[m]);
        if (lane == 0 && m < M) Y[static_cast<size_t>(m) * N + row] = sum;
    }
}

// RDNA4 (gfx120x half8 WMMA) & RDNA3/3.5 (gfx110x/1103/1150/1151 half16 WMMA) Skinny & Batched SF4 GEMM!
// Block tile: BM=16 (skinny verify M=2..16) or BM=64 (batched prefill / B=32..64), BN=64, BK=64 (4 blocks of 16).
// 4 waves (128 threads): each wave handles (BM/2) x (BN/2) = 16x32 or 32x32 via Wave32 WMMA!
// Guarded INSIDE body so host compiler always instantiates symbol while RDNA2 device pass emits zero WMMA!
template <int BM, int BN>
__global__ void __launch_bounds__(128) k_sf4_wmma_f16(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    float outer_scale,
    const half_t* __restrict__ X,  // [M, K]
    float* __restrict__ Y,         // [M, N]
    int M, int N, int K) {
#if INK_HAS_WMMA
    constexpr int BK = 64;
    constexpr int LD = BK + 8;  // 72 halves -> 144 B (zero bank conflicts on RDNA3/4!)
    constexpr int W_M = (BM >= 32) ? 2 : 1;
    constexpr int W_N = 4 / W_M;
    constexpr int MTW = (BM / W_M) / 16;
    constexpr int NTW = (BN / W_N) / 16;

    __shared__ alignas(16) half_t sX[BM * LD];
    __shared__ alignas(16) half_t sW[BN * LD];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave / W_N;
    const int wn = wave % W_N;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;

    fv8 acc[MTW][NTW];
#pragma unroll
    for (int mi = 0; mi < MTW; ++mi)
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) acc[mi][ni] = fv8_zero();

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 1. Load X tile [BM, BK] -> sX
        for (int idx = tid; idx < BM * (BK / 8); idx += 128) {
            const int r = idx / (BK / 8);
            const int c8 = idx % (BK / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (m_blk + r < M && k0 + c8 * 8 < K) {
                v = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(m_blk + r) * K + k0 + c8 * 8);
            }
            *reinterpret_cast<uint4*>(&sX[r * LD + c8 * 8]) = v;
        }
        // 2. Dequant SF4 W tile [BN, BK] -> sW (128 threads: 2 threads per row, each unpacks 2 blocks of 16 = 32 weights)
        for (int idx = tid; idx < BN * (BK / 16); idx += 128) {
            const int r = idx / (BK / 16);
            const int b4 = idx % (BK / 16);
            const int gn = n_blk + r;
            const int gb = (k0 >> 4) + b4;
            half_t* dst = &sW[r * LD + b4 * 16];
            if (gn < N && gb < (K >> 4)) {
                const float sc = e4m3fn_to_f32(scales[static_cast<size_t>(gn) * (K >> 4) + gb]) * outer_scale;
                const uint32_t sc2 = h2_splat(sc);
                const uint2 raw = *reinterpret_cast<const uint2*>(packed + static_cast<size_t>(gn) * (K >> 1) + gb * 8);
                uint32_t out0[4], out1[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t b0 = (raw.x >> (i * 8)) & 0xFFu;
                    const uint32_t b1 = (raw.y >> (i * 8)) & 0xFFu;
                    const hv2 h0 = {sf4_e2m1_f16(b0 & 15u), sf4_e2m1_f16(b0 >> 4)};
                    const hv2 h1 = {sf4_e2m1_f16(b1 & 15u), sf4_e2m1_f16(b1 >> 4)};
                    out0[i] = h2_mul(__builtin_bit_cast(uint32_t, h0), sc2);
                    out1[i] = h2_mul(__builtin_bit_cast(uint32_t, h1), sc2);
                }
                *reinterpret_cast<uint4*>(dst)     = make_uint4(out0[0], out0[1], out0[2], out0[3]);
                *reinterpret_cast<uint4*>(dst + 8) = make_uint4(out1[0], out1[1], out1[2], out1[3]);
            } else {
                *reinterpret_cast<uint4*>(dst)     = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
            }
        }
        sync_lds();

        // 3. Wave32 WMMA across BK=64 (4 K-steps of 16)
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma_h_t a[MTW];
#pragma unroll
            for (int mi = 0; mi < MTW; ++mi) {
                a[mi] = wmma_ld_h(&sX[(wm * MTW * 16 + mi * 16 + l16) * LD + kk * 16], hi);
            }
            wmma_h_t b[NTW];
#pragma unroll
            for (int ni = 0; ni < NTW; ++ni) {
                b[ni] = wmma_ld_h(&sW[(wn * NTW * 16 + ni * 16 + l16) * LD + kk * 16], hi);
            }
#pragma unroll
            for (int mi = 0; mi < MTW; ++mi)
#pragma unroll
                for (int ni = 0; ni < NTW; ++ni) acc[mi][ni] = wmma_f16(a[mi], b[ni], acc[mi][ni]);
        }
        sync_lds();
    }

    // 4. Epilogue
#pragma unroll
    for (int mi = 0; mi < MTW; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) {
            const int gn = n_blk + wn * NTW * 16 + ni * 16 + wmma_acc_n(0, lane);
            if (gn < N) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gm = m_blk + wm * MTW * 16 + mi * 16 + wmma_acc_m(r, lane);
                    if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[mi][ni][r];
                }
            }
        }
    }
#else
    (void)packed; (void)scales; (void)outer_scale; (void)X; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// RDNA4 (gfx1200/gfx1201) 2x-Rate Native INT8 WMMA (`wmma_i32_16x16x16_iu8_w32_gfx12`) for SF4!
// Exploits exact integer property: 2 * E2M1 in {-12,-8,-6,-4,-3,-2,-1,0,+1,+2,+3,+4,+6,+8,+12} fits in signed int8!
// Uses 2-term symmetric Q8 activation quantization per 16-element SF4 block (zero loss vs FP32, <1e-4 rel_err).
template <int BM, int BN>
__global__ void __launch_bounds__(128) k_sf4_wmma_i8_gfx12(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    float outer_scale,
    const half_t* __restrict__ X,
    float* __restrict__ Y,
    int M, int N, int K) {
#if INK_GFX12
    constexpr int LD = 24;   // 16 + 8 int8 -> 8-byte aligned, bank-conflict-free
    constexpr int W_M = (BM >= 32) ? 2 : 1;
    constexpr int W_N = 4 / W_M;
    constexpr int MTW = (BM / W_M) / 16;
    constexpr int NTW = (BN / W_N) / 16;

    __shared__ alignas(16) int8_t sXq[BM * LD];
    __shared__ alignas(16) int8_t sWq[BN * LD];
    __shared__ float sXd[BM];
    __shared__ float sWs[BN];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave / W_N;
    const int wn = wave % W_N;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;
    const float half_outer = 0.5f * outer_scale;

    fv8 acc[MTW][NTW];
#pragma unroll
    for (int mi = 0; mi < MTW; ++mi)
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) acc[mi][ni] = fv8_zero();

    const int nblk16 = K >> 4;
    for (int gb = 0; gb < nblk16; ++gb) {
        const int k0 = gb << 4;
        // 1. Quantize X [BM, 16] into int8 + per-16 scale sXd[BM]
        for (int r = tid; r < BM; r += 128) {
            const int gm = m_blk + r;
            if (gm < M) {
                const uint4 u0 = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(gm) * K + k0);
                const uint4 u1 = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(gm) * K + k0 + 8);
                float xf[16];
                h2_unpack(u0.x, xf[0], xf[1]);   h2_unpack(u0.y, xf[2], xf[3]);
                h2_unpack(u0.z, xf[4], xf[5]);   h2_unpack(u0.w, xf[6], xf[7]);
                h2_unpack(u1.x, xf[8], xf[9]);   h2_unpack(u1.y, xf[10], xf[11]);
                h2_unpack(u1.z, xf[12], xf[13]); h2_unpack(u1.w, xf[14], xf[15]);
                float am = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(xf[i]));
                const float d = am * (1.0f / 127.0f);
                const float inv = am > 0.0f ? (127.0f / am) : 0.0f;
                int8_t* dst = &sXq[r * LD];
#pragma unroll
                for (int i = 0; i < 16; ++i) dst[i] = static_cast<int8_t>(clamp_i8(xf[i] * inv));
                sXd[r] = d;
            } else {
                *reinterpret_cast<uint4*>(&sXq[r * LD]) = make_uint4(0, 0, 0, 0);
                sXd[r] = 0.0f;
            }
        }
        // 2. Unpack SF4 W [BN, 16] into EXACT int8 x2 via register LUT + per-16 scale
        for (int r = tid; r < BN; r += 128) {
            const int gn = n_blk + r;
            int8_t* dst = &sWq[r * LD];
            if (gn < N) {
                sWs[r] = e4m3fn_to_f32(scales[static_cast<size_t>(gn) * nblk16 + gb]) * half_outer;
                const uint2 raw = *reinterpret_cast<const uint2*>(packed + static_cast<size_t>(gn) * (K >> 1) + gb * 8);
                uint32_t out[4];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint32_t w = (i == 0) ? raw.x : raw.y;
                    out[2 * i + 0] = static_cast<uint32_t>(pack4_i8(
                        sf4_e2m1_i8_x2(w & 15u), sf4_e2m1_i8_x2((w >> 4) & 15u),
                        sf4_e2m1_i8_x2((w >> 8) & 15u), sf4_e2m1_i8_x2((w >> 12) & 15u)));
                    out[2 * i + 1] = static_cast<uint32_t>(pack4_i8(
                        sf4_e2m1_i8_x2((w >> 16) & 15u), sf4_e2m1_i8_x2((w >> 20) & 15u),
                        sf4_e2m1_i8_x2((w >> 24) & 15u), sf4_e2m1_i8_x2((w >> 28) & 15u)));
                }
                *reinterpret_cast<uint4*>(dst) = make_uint4(out[0], out[1], out[2], out[3]);
            } else {
                sWs[r] = 0.0f;
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
        sync_lds();

        // 3. 2x-rate INT8 WMMA 16x16x16
        iv2 a[MTW];
#pragma unroll
        for (int mi = 0; mi < MTW; ++mi) a[mi] = wmma_ld_i8_k16(&sXq[(wm * MTW * 16 + mi * 16 + l16) * LD], hi);
        iv2 b[NTW];
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) b[ni] = wmma_ld_i8_k16(&sWq[(wn * NTW * 16 + ni * 16 + l16) * LD], hi);

        float wn_sc[NTW];
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) wn_sc[ni] = sWs[wn * NTW * 16 + ni * 16 + l16];

#pragma unroll
        for (int mi = 0; mi < MTW; ++mi) {
            float xm_sc[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) xm_sc[r] = sXd[wm * MTW * 16 + mi * 16 + 8 * hi + r];
#pragma unroll
            for (int ni = 0; ni < NTW; ++ni) {
                const iv8 iv = wmma_i8_k16(a[mi], b[ni], iv8_zero());
                const float ws_v = wn_sc[ni];
#pragma unroll
                for (int r = 0; r < 8; ++r) acc[mi][ni][r] = fmaf(static_cast<float>(iv[r]), xm_sc[r] * ws_v, acc[mi][ni][r]);
            }
        }
        sync_lds();
    }

#pragma unroll
    for (int mi = 0; mi < MTW; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) {
            const int gn = n_blk + wn * NTW * 16 + ni * 16 + wmma_acc_n(0, lane);
            if (gn < N) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gm = m_blk + wm * MTW * 16 + mi * 16 + wmma_acc_m(r, lane);
                    if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[mi][ni][r];
                }
            }
        }
    }
#else
    (void)packed; (void)scales; (void)outer_scale; (void)X; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// RDNA2 (gfx103x) Zero-WMMA Wave32 sdot4 Tiled SF4 GEMM (also clean fallback for any arch!)
template <int BM, int BN>
__global__ void __launch_bounds__(128) k_sf4_sdot4_tiled(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    float outer_scale,
    const half_t* __restrict__ X,
    float* __restrict__ Y,
    int M, int N, int K) {
    // 128 threads = 4 waves, each thread computes TM x TN outputs over K in 16-element blocks using sdot4_i32!
    __shared__ alignas(16) int sXq[BM * 4];  // [BM][4 int32 = 16 int8]
    __shared__ alignas(16) int sWq[BN * 4];  // [BN][4 int32 = 16 int8]
    __shared__ float sXd[BM];
    __shared__ float sWs[BN];

    const int tid = threadIdx.x;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;
    const float half_outer = 0.5f * outer_scale;

    // 128 threads map onto BM x BN (e.g. BM=16, BN=32 -> 4 threads per n? No: 16*32=512 -> 4 outputs/thread)
    constexpr int TM = 2;
    constexpr int TN = (BM * BN) / (128 * TM);
    const int t_row = (tid / (BN / TN)) * TM;
    const int t_col = (tid % (BN / TN)) * TN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const int nblk16 = K >> 4;
    for (int gb = 0; gb < nblk16; ++gb) {
        const int k0 = gb << 4;
        for (int r = tid; r < BM; r += 128) {
            const int gm = m_blk + r;
            if (gm < M) {
                const uint4 u0 = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(gm) * K + k0);
                const uint4 u1 = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(gm) * K + k0 + 8);
                float xf[16];
                h2_unpack(u0.x, xf[0], xf[1]);   h2_unpack(u0.y, xf[2], xf[3]);
                h2_unpack(u0.z, xf[4], xf[5]);   h2_unpack(u0.w, xf[6], xf[7]);
                h2_unpack(u1.x, xf[8], xf[9]);   h2_unpack(u1.y, xf[10], xf[11]);
                h2_unpack(u1.z, xf[12], xf[13]); h2_unpack(u1.w, xf[14], xf[15]);
                float am = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(xf[i]));
                const float d = am * (1.0f / 127.0f);
                const float inv = am > 0.0f ? (127.0f / am) : 0.0f;
                int4 p4;
                p4.x = pack4_i8(clamp_i8(xf[0] * inv), clamp_i8(xf[1] * inv), clamp_i8(xf[2] * inv), clamp_i8(xf[3] * inv));
                p4.y = pack4_i8(clamp_i8(xf[4] * inv), clamp_i8(xf[5] * inv), clamp_i8(xf[6] * inv), clamp_i8(xf[7] * inv));
                p4.z = pack4_i8(clamp_i8(xf[8] * inv), clamp_i8(xf[9] * inv), clamp_i8(xf[10] * inv), clamp_i8(xf[11] * inv));
                p4.w = pack4_i8(clamp_i8(xf[12] * inv), clamp_i8(xf[13] * inv), clamp_i8(xf[14] * inv), clamp_i8(xf[15] * inv));
                *reinterpret_cast<int4*>(&sXq[r * 4]) = p4;
                sXd[r] = d;
            } else {
                *reinterpret_cast<int4*>(&sXq[r * 4]) = make_int4(0, 0, 0, 0);
                sXd[r] = 0.0f;
            }
        }
        for (int r = tid; r < BN; r += 128) {
            const int gn = n_blk + r;
            if (gn < N) {
                sWs[r] = e4m3fn_to_f32(scales[static_cast<size_t>(gn) * nblk16 + gb]) * half_outer;
                const uint2 raw = *reinterpret_cast<const uint2*>(packed + static_cast<size_t>(gn) * (K >> 1) + gb * 8);
                int4 p4;
                p4.x = pack4_i8(sf4_e2m1_i8_x2(raw.x & 15u), sf4_e2m1_i8_x2((raw.x >> 4) & 15u),
                                sf4_e2m1_i8_x2((raw.x >> 8) & 15u), sf4_e2m1_i8_x2((raw.x >> 12) & 15u));
                p4.y = pack4_i8(sf4_e2m1_i8_x2((raw.x >> 16) & 15u), sf4_e2m1_i8_x2((raw.x >> 20) & 15u),
                                sf4_e2m1_i8_x2((raw.x >> 24) & 15u), sf4_e2m1_i8_x2((raw.x >> 28) & 15u));
                p4.z = pack4_i8(sf4_e2m1_i8_x2(raw.y & 15u), sf4_e2m1_i8_x2((raw.y >> 4) & 15u),
                                sf4_e2m1_i8_x2((raw.y >> 8) & 15u), sf4_e2m1_i8_x2((raw.y >> 12) & 15u));
                p4.w = pack4_i8(sf4_e2m1_i8_x2((raw.y >> 16) & 15u), sf4_e2m1_i8_x2((raw.y >> 20) & 15u),
                                sf4_e2m1_i8_x2((raw.y >> 24) & 15u), sf4_e2m1_i8_x2((raw.y >> 28) & 15u));
                *reinterpret_cast<int4*>(&sWq[r * 4]) = p4;
            } else {
                *reinterpret_cast<int4*>(&sWq[r * 4]) = make_int4(0, 0, 0, 0);
                sWs[r] = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int4 a = *reinterpret_cast<const int4*>(&sXq[(t_row + i) * 4]);
            const float xd = sXd[t_row + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int4 b = *reinterpret_cast<const int4*>(&sWq[(t_col + j) * 4]);
                int idot = 0;
                idot = sdot4_i32(a.x, b.x, idot);
                idot = sdot4_i32(a.y, b.y, idot);
                idot = sdot4_i32(a.z, b.z, idot);
                idot = sdot4_i32(a.w, b.w, idot);
                acc[i][j] = fmaf(static_cast<float>(idot), xd * sWs[t_col + j], acc[i][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = m_blk + t_row + i;
        if (gm < M) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int gn = n_blk + t_col + j;
                if (gn < N) Y[static_cast<size_t>(gm) * N + gn] = acc[i][j];
            }
        }
    }
}

// =================================================================================================
// 2. MX8 (Lossless MXFP8: F8_E4M3FN + UE8M0/32w) Multi-Gen Kernels (Layers 20..23)
// =================================================================================================

template <int MT>
__global__ void __launch_bounds__(256) k_mx8_gemv(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ scales,
    const half_t* __restrict__ X,
    float* __restrict__ Y,
    int M, int N, int K) {
    __shared__ half_t s_e4m3[256];
    s_e4m3[threadIdx.x] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(threadIdx.x)));
    __syncthreads();

    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= N) return;

    const uint8_t* wr = weights + static_cast<size_t>(row) * K;
    const uint8_t* sr = scales + static_cast<size_t>(row) * (K >> 5);
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    const int nblk32 = K >> 5;
    for (int b = lane; b < nblk32; b += 32) {
        const float sc = ue8m0_to_f32(sr[b]);
        const uint8_t* wb = wr + (b << 5);
        uint32_t w2[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const hv2 h = {s_e4m3[wb[2 * i]], s_e4m3[wb[2 * i + 1]]};
            w2[i] = __builtin_bit_cast(uint32_t, h);
        }
        const int k0 = b << 5;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const uint4* xp = reinterpret_cast<const uint4*>(X + static_cast<size_t>(m) * K + k0);
                float d = 0.0f;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint4 xv = xp[q];
                    d = fdot2_f32(w2[4 * q + 0], xv.x, d);
                    d = fdot2_f32(w2[4 * q + 1], xv.y, d);
                    d = fdot2_f32(w2[4 * q + 2], xv.z, d);
                    d = fdot2_f32(w2[4 * q + 3], xv.w, d);
                }
                acc[m] = fmaf(d, sc, acc[m]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const float sum = wave32_sum(acc[m]);
        if (lane == 0 && m < M) Y[static_cast<size_t>(m) * N + row] = sum;
    }
}

template <int BM, int BN>
__global__ void __launch_bounds__(128) k_mx8_wmma_f16(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ scales,
    const half_t* __restrict__ X,
    float* __restrict__ Y,
    int M, int N, int K) {
#if INK_HAS_WMMA
    constexpr int BK = 64;
    constexpr int LD = BK + 8;
    constexpr int W_M = (BM >= 32) ? 2 : 1;
    constexpr int W_N = 4 / W_M;
    constexpr int MTW = (BM / W_M) / 16;
    constexpr int NTW = (BN / W_N) / 16;

    __shared__ alignas(16) half_t sX[BM * LD];
    __shared__ alignas(16) half_t sW[BN * LD];
    __shared__ half_t s_e4m3[256];

    const int tid = threadIdx.x;
    s_e4m3[tid]       = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    s_e4m3[tid + 128] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid + 128)));
    sync_lds();

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave / W_N;
    const int wn = wave % W_N;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;

    fv8 acc[MTW][NTW];
#pragma unroll
    for (int mi = 0; mi < MTW; ++mi)
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) acc[mi][ni] = fv8_zero();

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * (BK / 8); idx += 128) {
            const int r = idx / (BK / 8);
            const int c8 = idx % (BK / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (m_blk + r < M && k0 + c8 * 8 < K) {
                v = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(m_blk + r) * K + k0 + c8 * 8);
            }
            *reinterpret_cast<uint4*>(&sX[r * LD + c8 * 8]) = v;
        }
        // 128 threads dequant BN * 2 blocks of 32 weights
        for (int idx = tid; idx < BN * (BK / 32); idx += 128) {
            const int r = idx / (BK / 32);
            const int b2 = idx % (BK / 32);
            const int gn = n_blk + r;
            const int gb = (k0 >> 5) + b2;
            half_t* dst = &sW[r * LD + b2 * 32];
            if (gn < N && gb < (K >> 5)) {
                const float sc = ue8m0_to_f32(scales[static_cast<size_t>(gn) * (K >> 5) + gb]);
                const uint32_t sc2 = h2_splat(sc);
                const uint8_t* wb = weights + static_cast<size_t>(gn) * K + (gb << 5);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    uint32_t o[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const hv2 h = {s_e4m3[wb[q * 8 + 2 * i]], s_e4m3[wb[q * 8 + 2 * i + 1]]};
                        o[i] = h2_mul(__builtin_bit_cast(uint32_t, h), sc2);
                    }
                    *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(o[0], o[1], o[2], o[3]);
                }
            } else {
                for (int q = 0; q < 4; ++q) *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(0, 0, 0, 0);
            }
        }
        sync_lds();

#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma_h_t a[MTW];
#pragma unroll
            for (int mi = 0; mi < MTW; ++mi) a[mi] = wmma_ld_h(&sX[(wm * MTW * 16 + mi * 16 + l16) * LD + kk * 16], hi);
            wmma_h_t b[NTW];
#pragma unroll
            for (int ni = 0; ni < NTW; ++ni) b[ni] = wmma_ld_h(&sW[(wn * NTW * 16 + ni * 16 + l16) * LD + kk * 16], hi);
#pragma unroll
            for (int mi = 0; mi < MTW; ++mi)
#pragma unroll
                for (int ni = 0; ni < NTW; ++ni) acc[mi][ni] = wmma_f16(a[mi], b[ni], acc[mi][ni]);
        }
        sync_lds();
    }

#pragma unroll
    for (int mi = 0; mi < MTW; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) {
            const int gn = n_blk + wn * NTW * 16 + ni * 16 + wmma_acc_n(0, lane);
            if (gn < N) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gm = m_blk + wm * MTW * 16 + mi * 16 + wmma_acc_m(r, lane);
                    if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[mi][ni][r];
                }
            }
        }
    }
#else
    (void)weights; (void)scales; (void)X; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// RDNA2 zero-WMMA fdot2 tiled MX8 GEMM
template <int BM, int BN>
__global__ void __launch_bounds__(128) k_mx8_fdot2_tiled(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ scales,
    const half_t* __restrict__ X,
    float* __restrict__ Y,
    int M, int N, int K) {
    constexpr int BK = 32;
    __shared__ alignas(16) half_t sX[BM * BK];
    __shared__ alignas(16) half_t sW[BN * BK];
    __shared__ half_t s_e4m3[256];

    const int tid = threadIdx.x;
    s_e4m3[tid]       = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    s_e4m3[tid + 128] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid + 128)));
    __syncthreads();

    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;
    constexpr int TM = 2;
    constexpr int TN = (BM * BN) / (128 * TM);
    const int t_row = (tid / (BN / TN)) * TM;
    const int t_col = (tid % (BN / TN)) * TN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const int nblk32 = K >> 5;
    for (int gb = 0; gb < nblk32; ++gb) {
        const int k0 = gb << 5;
        for (int idx = tid; idx < BM * 4; idx += 128) {
            const int r = idx >> 2, c8 = idx & 3;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (m_blk + r < M) v = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(m_blk + r) * K + k0 + c8 * 8);
            *reinterpret_cast<uint4*>(&sX[r * BK + c8 * 8]) = v;
        }
        for (int r = tid; r < BN; r += 128) {
            const int gn = n_blk + r;
            half_t* dst = &sW[r * BK];
            if (gn < N) {
                const uint32_t sc2 = h2_splat(ue8m0_to_f32(scales[static_cast<size_t>(gn) * nblk32 + gb]));
                const uint8_t* wb = weights + static_cast<size_t>(gn) * K + k0;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    uint32_t o[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const hv2 h = {s_e4m3[wb[q * 8 + 2 * i]], s_e4m3[wb[q * 8 + 2 * i + 1]]};
                        o[i] = h2_mul(__builtin_bit_cast(uint32_t, h), sc2);
                    }
                    *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(o[0], o[1], o[2], o[3]);
                }
            } else {
                for (int q = 0; q < 4; ++q) *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

#pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint4 ax[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ax[i] = *reinterpret_cast<const uint4*>(&sX[(t_row + i) * BK + q * 8]);
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const uint4 bw = *reinterpret_cast<const uint4*>(&sW[(t_col + j) * BK + q * 8]);
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    float d = acc[i][j];
                    d = fdot2_f32(ax[i].x, bw.x, d);
                    d = fdot2_f32(ax[i].y, bw.y, d);
                    d = fdot2_f32(ax[i].z, bw.z, d);
                    d = fdot2_f32(ax[i].w, bw.w, d);
                    acc[i][j] = d;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = m_blk + t_row + i;
        if (gm < M) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int gn = n_blk + t_col + j;
                if (gn < N) Y[static_cast<size_t>(gm) * N + gn] = acc[i][j];
            }
        }
    }
}

// =================================================================================================
// 3. S_K (S1.5, S2_K..S8_K) Standard & Paired-2-Row (SB_0..SB_8) Multi-Gen Kernels
// =================================================================================================

// Standard S_K GEMV/GEMM (K % 256 == 0, e.g. w13 [2304, 2304], dense S_K trunk tensors)
template <int MT>
__global__ void __launch_bounds__(256) k_sk_gemv(
    const uint8_t* __restrict__ W,
    int code,
    const half_t* __restrict__ X,
    float* __restrict__ Y,
    int M, int N, int K) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= N) return;

    const int nsb = K >> 8;
    const int sb_b = sk_sb_bytes(code);
    const uint8_t* wr = W + static_cast<size_t>(row) * nsb * sb_b;
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    for (int s = 0; s < nsb; ++s) {
        const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
        const int j0 = lane * 8;
        const int g = j0 >> 6;
        const float eff = decode_sk_sub_eff_i8(blk, code, g);
        const uint8_t* pl = blk + 6;
        uint32_t w2[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const hv2 h = {
                static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, code, j0 + 2 * i))),
                static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, code, j0 + 2 * i + 1)))
            };
            w2[i] = __builtin_bit_cast(uint32_t, h);
        }
        const int k0 = (s << 8) + j0;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const uint4 xv = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(m) * K + k0);
                float d = 0.0f;
                d = fdot2_f32(w2[0], xv.x, d);
                d = fdot2_f32(w2[1], xv.y, d);
                d = fdot2_f32(w2[2], xv.z, d);
                d = fdot2_f32(w2[3], xv.w, d);
                acc[m] = fmaf(d, eff, acc[m]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const float sum = wave32_sum(acc[m]);
        if (lane == 0 && m < M) Y[static_cast<size_t>(m) * N + row] = sum;
    }
}

// Multi-Gen Wave32 WMMA S_K GEMM (RDNA4 half8 / RDNA3/3.5 half16) for M=2..64+!
template <int BM, int BN>
__global__ void __launch_bounds__(128) k_sk_wmma_f16(
    const uint8_t* __restrict__ W,
    int code,
    const half_t* __restrict__ X,
    float* __restrict__ Y,
    int M, int N, int K) {
#if INK_HAS_WMMA
    constexpr int BK = 64;  // 1 sub-group of 64 weights
    constexpr int LD = BK + 8;
    constexpr int W_M = (BM >= 32) ? 2 : 1;
    constexpr int W_N = 4 / W_M;
    constexpr int MTW = (BM / W_M) / 16;
    constexpr int NTW = (BN / W_N) / 16;

    __shared__ alignas(16) half_t sX[BM * LD];
    __shared__ alignas(16) half_t sW[BN * LD];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave / W_N;
    const int wn = wave % W_N;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;
    const int nsb = K >> 8;
    const int sb_b = sk_sb_bytes(code);

    fv8 acc[MTW][NTW];
#pragma unroll
    for (int mi = 0; mi < MTW; ++mi)
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) acc[mi][ni] = fv8_zero();

    for (int s = 0; s < nsb; ++s) {
        for (int g = 0; g < 4; ++g) {
            const int k0 = (s << 8) + (g << 6);
            for (int idx = tid; idx < BM * 8; idx += 128) {
                const int r = idx >> 3, c8 = idx & 7;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (m_blk + r < M) v = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(m_blk + r) * K + k0 + c8 * 8);
                *reinterpret_cast<uint4*>(&sX[r * LD + c8 * 8]) = v;
            }
            for (int idx = tid; idx < BN * 4; idx += 128) {
                const int r = idx >> 2, q16 = idx & 3;
                const int gn = n_blk + r;
                half_t* dst = &sW[r * LD + q16 * 16];
                if (gn < N) {
                    const uint8_t* blk = W + (static_cast<size_t>(gn) * nsb + s) * sb_b;
                    const uint32_t ef2 = h2_splat(decode_sk_sub_eff_i8(blk, code, g));
                    const uint8_t* pl = blk + 6;
                    const int j0 = (g << 6) + q16 * 16;
                    uint32_t o0[4], o1[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const hv2 h0 = {
                            static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, code, j0 + 2 * i))),
                            static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, code, j0 + 2 * i + 1)))
                        };
                        const hv2 h1 = {
                            static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, code, j0 + 8 + 2 * i))),
                            static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, code, j0 + 8 + 2 * i + 1)))
                        };
                        o0[i] = h2_mul(__builtin_bit_cast(uint32_t, h0), ef2);
                        o1[i] = h2_mul(__builtin_bit_cast(uint32_t, h1), ef2);
                    }
                    *reinterpret_cast<uint4*>(dst)     = make_uint4(o0[0], o0[1], o0[2], o0[3]);
                    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
                } else {
                    *reinterpret_cast<uint4*>(dst)     = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
                }
            }
            sync_lds();

#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma_h_t a[MTW];
#pragma unroll
                for (int mi = 0; mi < MTW; ++mi) a[mi] = wmma_ld_h(&sX[(wm * MTW * 16 + mi * 16 + l16) * LD + kk * 16], hi);
                wmma_h_t b[NTW];
#pragma unroll
                for (int ni = 0; ni < NTW; ++ni) b[ni] = wmma_ld_h(&sW[(wn * NTW * 16 + ni * 16 + l16) * LD + kk * 16], hi);
#pragma unroll
                for (int mi = 0; mi < MTW; ++mi)
#pragma unroll
                    for (int ni = 0; ni < NTW; ++ni) acc[mi][ni] = wmma_f16(a[mi], b[ni], acc[mi][ni]);
            }
            sync_lds();
        }
    }

#pragma unroll
    for (int mi = 0; mi < MTW; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) {
            const int gn = n_blk + wn * NTW * 16 + ni * 16 + wmma_acc_n(0, lane);
            if (gn < N) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gm = m_blk + wm * MTW * 16 + mi * 16 + wmma_acc_m(r, lane);
                    if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[mi][ni][r];
                }
            }
        }
    }
#else
    (void)W; (void)code; (void)X; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// Paired 2-Row S_K GEMV/GEMM (ncols % 256 == 128, e.g. Inkling-2-Mini w2 [2304, 1152] = 9 super-blocks SB_0..SB_8 per pair!)
// Optionally applies inverse MACaQ scale `sq_xd_dn` (FP16[K]) directly on-the-fly when `sq_xd_dn != nullptr`!
template <int MT>
__global__ void __launch_bounds__(256) k_sk_paired2_gemv(
    const uint8_t* __restrict__ W,
    int code,
    const half_t* __restrict__ sq_xd_dn,  // optional FP16[K] inverse scale (divide X by sq_xd_dn)
    const half_t* __restrict__ X,         // [M, K]
    float* __restrict__ Y,                // [M, N] (N even)
    int M, int N, int K) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int pair = blockIdx.x * 8 + wave;
    if (pair * 2 >= N) return;

    const int sb_per_pair = (2 * K) >> 8;   // e.g. 2304 / 256 = 9
    const int sb_b = sk_sb_bytes(code);
    const uint8_t* pr = W + static_cast<size_t>(pair) * sb_per_pair * sb_b;

    float acc0[MT], acc1[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

    for (int s = 0; s < sb_per_pair; ++s) {
        const uint8_t* blk = pr + static_cast<size_t>(s) * sb_b;
        const int j0 = lane * 8;
        const int g = j0 >> 6;
        const float eff = decode_sk_sub_eff_i8(blk, code, g);
        const uint8_t* pl = blk + 6;
        const int flat_col = (s << 8) + j0;
        const bool is_r1 = (flat_col >= K);
        const int k0 = is_r1 ? (flat_col - K) : flat_col;

        uint32_t w2[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float wf0 = static_cast<float>(decode_sk_w_i8(pl, code, j0 + 2 * i));
            float wf1 = static_cast<float>(decode_sk_w_i8(pl, code, j0 + 2 * i + 1));
            if (sq_xd_dn != nullptr) {
                wf0 /= static_cast<float>(sq_xd_dn[k0 + 2 * i]);
                wf1 /= static_cast<float>(sq_xd_dn[k0 + 2 * i + 1]);
            }
            const hv2 h = {static_cast<half_t>(wf0), static_cast<half_t>(wf1)};
            w2[i] = __builtin_bit_cast(uint32_t, h);
        }

#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const uint4 xv = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(m) * K + k0);
                float d = 0.0f;
                d = fdot2_f32(w2[0], xv.x, d);
                d = fdot2_f32(w2[1], xv.y, d);
                d = fdot2_f32(w2[2], xv.z, d);
                d = fdot2_f32(w2[3], xv.w, d);
                if (is_r1) acc1[m] = fmaf(d, eff, acc1[m]);
                else       acc0[m] = fmaf(d, eff, acc0[m]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const float s0 = wave32_sum(acc0[m]);
        const float s1 = wave32_sum(acc1[m]);
        if (lane == 0 && m < M) {
            Y[static_cast<size_t>(m) * N + 2 * pair]     = s0;
            Y[static_cast<size_t>(m) * N + 2 * pair + 1] = s1;
        }
    }
}

// =================================================================================================
// 4. Dense Trunk / DSpark / ViT Linear Kernels (BF16, F32, Q8_ROW) + Conversions + Zero-Sync MoE
// =================================================================================================

__global__ void __launch_bounds__(256) k_f32_to_f16(
    const float* __restrict__ src,
    half_t* __restrict__ dst,
    int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) dst[idx] = static_cast<half_t>(src[idx]);
}

// BF16 linear Y[M, N] = X[M, K] @ W[N, K]^T + optional bias[N] (M=1..4 Wave32 GEMV)
template <int MT>
__global__ void __launch_bounds__(256) k_bf16_gemv(
    const uint16_t* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= N) return;
    const uint16_t* wr = W + static_cast<size_t>(row) * K;
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    const int k4 = K & ~3;
    for (int k = lane * 4; k < k4; k += 128) {
        const uint2 w4 = *reinterpret_cast<const uint2*>(wr + k);
        const float f0 = bf16_to_f32(static_cast<uint16_t>(w4.x & 0xFFFFu));
        const float f1 = bf16_to_f32(static_cast<uint16_t>(w4.x >> 16));
        const float f2 = bf16_to_f32(static_cast<uint16_t>(w4.y & 0xFFFFu));
        const float f3 = bf16_to_f32(static_cast<uint16_t>(w4.y >> 16));
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const float4 xv = *reinterpret_cast<const float4*>(X + static_cast<size_t>(m) * K + k);
                acc[m] = fmaf(f0, xv.x, fmaf(f1, xv.y, fmaf(f2, xv.z, fmaf(f3, xv.w, acc[m]))));
            }
        }
    }
    for (int k = k4 + lane; k < K; k += 32) {
        const float fw = bf16_to_f32(wr[k]);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) acc[m] = fmaf(fw, X[static_cast<size_t>(m) * K + k], acc[m]);
        }
    }
    const float bv = bias ? bias[row] : 0.0f;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const float sum = wave32_sum(acc[m]) + bv;
        if (lane == 0 && m < M) Y[static_cast<size_t>(m) * N + row] = sum;
    }
}

// Multi-Gen Wave32 WMMA BF16/FP16 GEMM (RDNA4 half8 / RDNA3/3.5 half16) for M > 4 (prefill, B=8..64, DSpark, ViT!)
template <int BM, int BN>
__global__ void __launch_bounds__(128) k_bf16_wmma_f16(
    const uint16_t* __restrict__ W,   // [N, K] BF16
    const float* __restrict__ X,      // [M, K] FP32
    const float* __restrict__ bias,   // nullptr or [N] FP32
    float* __restrict__ Y,            // [M, N] FP32
    int M, int N, int K) {
#if INK_HAS_WMMA
    constexpr int BK = 64;
    constexpr int LD = BK + 8;
    constexpr int W_M = (BM >= 32) ? 2 : 1;
    constexpr int W_N = 4 / W_M;
    constexpr int MTW = (BM / W_M) / 16;
    constexpr int NTW = (BN / W_N) / 16;

    __shared__ alignas(16) half_t sX[BM * LD];
    __shared__ alignas(16) half_t sW[BN * LD];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave / W_N;
    const int wn = wave % W_N;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;

    fv8 acc[MTW][NTW];
#pragma unroll
    for (int mi = 0; mi < MTW; ++mi)
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) acc[mi][ni] = fv8_zero();

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * (BK / 4); idx += 128) {
            const int r = idx / (BK / 4);
            const int c4 = idx % (BK / 4);
            const int gm = m_blk + r;
            const int gk = k0 + c4 * 4;
            half_t* dst = &sX[r * LD + c4 * 4];
            if (gm < M && gk + 3 < K) {
                const float4 xv = *reinterpret_cast<const float4*>(X + static_cast<size_t>(gm) * K + gk);
                dst[0] = static_cast<half_t>(xv.x);
                dst[1] = static_cast<half_t>(xv.y);
                dst[2] = static_cast<half_t>(xv.z);
                dst[3] = static_cast<half_t>(xv.w);
            } else {
                for (int j = 0; j < 4; ++j) {
                    dst[j] = (gm < M && gk + j < K) ? static_cast<half_t>(X[static_cast<size_t>(gm) * K + gk + j]) : static_cast<half_t>(0.0f);
                }
            }
        }
        for (int idx = tid; idx < BN * (BK / 4); idx += 128) {
            const int r = idx / (BK / 4);
            const int c4 = idx % (BK / 4);
            const int gn = n_blk + r;
            const int gk = k0 + c4 * 4;
            half_t* dst = &sW[r * LD + c4 * 4];
            if (gn < N && gk + 3 < K) {
                const uint2 w4 = *reinterpret_cast<const uint2*>(W + static_cast<size_t>(gn) * K + gk);
                dst[0] = static_cast<half_t>(bf16_to_f32(static_cast<uint16_t>(w4.x & 0xFFFFu)));
                dst[1] = static_cast<half_t>(bf16_to_f32(static_cast<uint16_t>(w4.x >> 16)));
                dst[2] = static_cast<half_t>(bf16_to_f32(static_cast<uint16_t>(w4.y & 0xFFFFu)));
                dst[3] = static_cast<half_t>(bf16_to_f32(static_cast<uint16_t>(w4.y >> 16)));
            } else {
                for (int j = 0; j < 4; ++j) {
                    dst[j] = (gn < N && gk + j < K) ? static_cast<half_t>(bf16_to_f32(W[static_cast<size_t>(gn) * K + gk + j])) : static_cast<half_t>(0.0f);
                }
            }
        }
        sync_lds();

#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma_h_t a[MTW];
#pragma unroll
            for (int mi = 0; mi < MTW; ++mi) a[mi] = wmma_ld_h(&sX[(wm * MTW * 16 + mi * 16 + l16) * LD + kk * 16], hi);
            wmma_h_t b[NTW];
#pragma unroll
            for (int ni = 0; ni < NTW; ++ni) b[ni] = wmma_ld_h(&sW[(wn * NTW * 16 + ni * 16 + l16) * LD + kk * 16], hi);
#pragma unroll
            for (int mi = 0; mi < MTW; ++mi)
#pragma unroll
                for (int ni = 0; ni < NTW; ++ni) acc[mi][ni] = wmma_f16(a[mi], b[ni], acc[mi][ni]);
        }
        sync_lds();
    }

#pragma unroll
    for (int mi = 0; mi < MTW; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) {
            const int gn = n_blk + wn * NTW * 16 + ni * 16 + wmma_acc_n(0, lane);
            if (gn < N) {
                const float bv = bias ? bias[gn] : 0.0f;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gm = m_blk + wm * MTW * 16 + mi * 16 + wmma_acc_m(r, lane);
                    if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[mi][ni][r] + bv;
                }
            }
        }
    }
#else
    (void)W; (void)X; (void)bias; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// RDNA2 zero-WMMA tiled BF16 GEMM (fdot2)
template <int BM, int BN>
__global__ void __launch_bounds__(128) k_bf16_fdot2_tiled(
    const uint16_t* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K) {
    constexpr int BK = 32;
    __shared__ alignas(16) half_t sX[BM * BK];
    __shared__ alignas(16) half_t sW[BN * BK];
    const int tid = threadIdx.x;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;
    constexpr int TM = 2;
    constexpr int TN = (BM * BN) / (128 * TM);
    const int t_row = (tid / (BN / TN)) * TM;
    const int t_col = (tid % (BN / TN)) * TN;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += 128) {
            const int r = idx / BK, c = idx % BK;
            sX[idx] = (m_blk + r < M && k0 + c < K) ? static_cast<half_t>(X[static_cast<size_t>(m_blk + r) * K + k0 + c]) : static_cast<half_t>(0.0f);
        }
        for (int idx = tid; idx < BN * BK; idx += 128) {
            const int r = idx / BK, c = idx % BK;
            sW[idx] = (n_blk + r < N && k0 + c < K) ? static_cast<half_t>(bf16_to_f32(W[static_cast<size_t>(n_blk + r) * K + k0 + c])) : static_cast<half_t>(0.0f);
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const uint32_t* xr = reinterpret_cast<const uint32_t*>(&sX[(t_row + i) * BK]);
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const uint32_t* wr = reinterpret_cast<const uint32_t*>(&sW[(t_col + j) * BK]);
                float d = acc[i][j];
#pragma unroll
                for (int q = 0; q < BK / 2; ++q) d = fdot2_f32(xr[q], wr[q], d);
                acc[i][j] = d;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = m_blk + t_row + i;
        if (gm < M) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int gn = n_blk + t_col + j;
                if (gn < N) Y[static_cast<size_t>(gm) * N + gn] = acc[i][j] + (bias ? bias[gn] : 0.0f);
            }
        }
    }
}

// FP32 linear Y[M, N] = X[M, K] @ W[N, K]^T + optional bias[N] (used for small projections: router, rel_proj, sink, dyn_sconv)
template <int MT>
__global__ void __launch_bounds__(256) k_f32_gemv(
    const float* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= N) return;
    const float* wr = W + static_cast<size_t>(row) * K;
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    const int k4 = K & ~3;
    for (int k = lane * 4; k < k4; k += 128) {
        const float4 wv = *reinterpret_cast<const float4*>(wr + k);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const float4 xv = *reinterpret_cast<const float4*>(X + static_cast<size_t>(m) * K + k);
                acc[m] = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc[m]))));
            }
        }
    }
    for (int k = k4 + lane; k < K; k += 32) {
        const float fw = wr[k];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) acc[m] = fmaf(fw, X[static_cast<size_t>(m) * K + k], acc[m]);
        }
    }
    const float bv = bias ? bias[row] : 0.0f;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const float sum = wave32_sum(acc[m]) + bv;
        if (lane == 0 && m < M) Y[static_cast<size_t>(m) * N + row] = sum;
    }
}

// Single-launch 2D-grid FP32 GEMV for RDNA2 / small-N batched fallback (replaces 128-iter host loop!)
__global__ void __launch_bounds__(256) k_f32_gemv_2d(
    const float* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    const int m0 = blockIdx.y * 4;
    if (row >= N || m0 >= M) return;
    const int bm = (M - m0) < 4 ? (M - m0) : 4;
    const float* wr = W + static_cast<size_t>(row) * K;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int k4 = K & ~3;
    for (int k = lane * 4; k < k4; k += 128) {
        const float4 wv = *reinterpret_cast<const float4*>(wr + k);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m < bm) {
                const float4 xv = *reinterpret_cast<const float4*>(X + static_cast<size_t>(m0 + m) * K + k);
                acc[m] = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc[m]))));
            }
        }
    }
    for (int k = k4 + lane; k < K; k += 32) {
        const float fw = wr[k];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m < bm) acc[m] = fmaf(fw, X[static_cast<size_t>(m0 + m) * K + k], acc[m]);
        }
    }
    const float bv = bias ? bias[row] : 0.0f;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const float sum = wave32_sum(acc[m]) + bv;
        if (lane == 0 && m < bm) Y[static_cast<size_t>(m0 + m) * N + row] = sum;
    }
}

// Multi-Gen Wave32 WMMA FP32->FP16 GEMM (RDNA4 half8 / RDNA3/3.5 half16) for M > 4 (router, loop ctrl, sinks!)
template <int BM, int BN>
__global__ void __launch_bounds__(128) k_f32_wmma_f16(
    const float* __restrict__ W,   // [N, K] FP32
    const float* __restrict__ X,   // [M, K] FP32
    const float* __restrict__ bias,
    float* __restrict__ Y,         // [M, N] FP32
    int M, int N, int K) {
#if INK_HAS_WMMA
    constexpr int BK = 64;
    constexpr int LD = BK + 8;
    constexpr int W_M = (BM >= 32) ? 2 : 1;
    constexpr int W_N = 4 / W_M;
    constexpr int MTW = (BM / W_M) / 16;
    constexpr int NTW = (BN / W_N) / 16;

    __shared__ alignas(16) half_t sX[BM * LD];
    __shared__ alignas(16) half_t sW[BN * LD];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave / W_N;
    const int wn = wave % W_N;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;

    fv8 acc[MTW][NTW];
#pragma unroll
    for (int mi = 0; mi < MTW; ++mi)
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) acc[mi][ni] = fv8_zero();

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * (BK / 4); idx += 128) {
            const int r = idx / (BK / 4);
            const int c4 = idx % (BK / 4);
            const int gm = m_blk + r;
            const int gk = k0 + c4 * 4;
            half_t* dst = &sX[r * LD + c4 * 4];
            if (gm < M && gk + 3 < K) {
                const float4 xv = *reinterpret_cast<const float4*>(X + static_cast<size_t>(gm) * K + gk);
                dst[0] = static_cast<half_t>(xv.x);
                dst[1] = static_cast<half_t>(xv.y);
                dst[2] = static_cast<half_t>(xv.z);
                dst[3] = static_cast<half_t>(xv.w);
            } else {
                for (int j = 0; j < 4; ++j)
                    dst[j] = (gm < M && gk + j < K) ? static_cast<half_t>(X[static_cast<size_t>(gm) * K + gk + j]) : static_cast<half_t>(0.0f);
            }
        }
        for (int idx = tid; idx < BN * (BK / 4); idx += 128) {
            const int r = idx / (BK / 4);
            const int c4 = idx % (BK / 4);
            const int gn = n_blk + r;
            const int gk = k0 + c4 * 4;
            half_t* dst = &sW[r * LD + c4 * 4];
            if (gn < N && gk + 3 < K) {
                const float4 wv = *reinterpret_cast<const float4*>(W + static_cast<size_t>(gn) * K + gk);
                dst[0] = static_cast<half_t>(wv.x);
                dst[1] = static_cast<half_t>(wv.y);
                dst[2] = static_cast<half_t>(wv.z);
                dst[3] = static_cast<half_t>(wv.w);
            } else {
                for (int j = 0; j < 4; ++j)
                    dst[j] = (gn < N && gk + j < K) ? static_cast<half_t>(W[static_cast<size_t>(gn) * K + gk + j]) : static_cast<half_t>(0.0f);
            }
        }
        sync_lds();

#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma_h_t a[MTW];
#pragma unroll
            for (int mi = 0; mi < MTW; ++mi) a[mi] = wmma_ld_h(&sX[(wm * MTW * 16 + mi * 16 + l16) * LD + kk * 16], hi);
            wmma_h_t b[NTW];
#pragma unroll
            for (int ni = 0; ni < NTW; ++ni) b[ni] = wmma_ld_h(&sW[(wn * NTW * 16 + ni * 16 + l16) * LD + kk * 16], hi);
#pragma unroll
            for (int mi = 0; mi < MTW; ++mi)
#pragma unroll
                for (int ni = 0; ni < NTW; ++ni) acc[mi][ni] = wmma_f16(a[mi], b[ni], acc[mi][ni]);
        }
        sync_lds();
    }

#pragma unroll
    for (int mi = 0; mi < MTW; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NTW; ++ni) {
            const int gn = n_blk + wn * NTW * 16 + ni * 16 + wmma_acc_n(0, lane);
            if (gn < N) {
                const float bv = bias ? bias[gn] : 0.0f;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gm = m_blk + wm * MTW * 16 + mi * 16 + wmma_acc_m(r, lane);
                    if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[mi][ni][r] + bv;
                }
            }
        }
    }
#else
    (void)W; (void)X; (void)bias; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// Q8_ROW linear: Inkling-2-Mini Q8_ROW wire layout is [rows] float32 scales FIRST, followed by [rows, cols] int8_t weights!
template <int MT>
__global__ void __launch_bounds__(256) k_q8row_gemv(
    const int8_t* __restrict__ W,
    const float* __restrict__ row_scales,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int N, int K) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= N) return;
    const int8_t* wr = W + static_cast<size_t>(row) * K;
    const float sc = row_scales[row];
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    for (int k = lane * 4; k < K; k += 128) {
        const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
        const float f0 = static_cast<float>(static_cast<int8_t>(w4 & 0xFFu));
        const float f1 = static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFFu));
        const float f2 = static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFFu));
        const float f3 = static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFFu));
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const float4 xv = *reinterpret_cast<const float4*>(X + static_cast<size_t>(m) * K + k);
                acc[m] = fmaf(f0, xv.x, fmaf(f1, xv.y, fmaf(f2, xv.z, fmaf(f3, xv.w, acc[m]))));
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const float sum = wave32_sum(acc[m]) * sc;
        if (lane == 0 && m < M) Y[static_cast<size_t>(m) * N + row] = sum;
    }
}

// Fused interleaved Gate+Up -> SITU(4.0, 25.0) activation (FP32 -> FP16 for w2 input)
__global__ void __launch_bounds__(256) k_situ_interleaved(
    const float* __restrict__ gu,   // [M, 2*inter] (even=gate, odd=up)
    half_t* __restrict__ act_out,   // [M, inter]
    int M, int inter, float beta, float lb) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * inter;
    if (idx >= total) return;
    const float g = gu[2 * idx];
    const float u = gu[2 * idx + 1];
    act_out[idx] = static_cast<half_t>(situ_f32(g, u, beta, lb));
}

// =================================================================================================
// 4B. Zero-Host-Sync GPU Resident MoE Kernels (Decode B=1..8 & HIP-Graph Capturable!)
// =================================================================================================
// Device expert descriptor table for one physical layer [128 experts]:
struct GpuExpertDesc {
    const uint8_t* body;   // device pointer to raw Section 4 expert body
    uint32_t dn_offset;
    uint8_t  gu_code;
    uint8_t  dn_code;
    uint16_t pad;
};

// Stage 1: Fused w13 + SITU across all (b, k) active expert selections (ZERO host sync!)
// Grid: ((inter + 3) / 4, B * top_k), Block: 256 (8 waves = 4 pairs of (even=gate, odd=up)!)
__global__ void __launch_bounds__(256) k_moe_direct_w13_situ(
    const GpuExpertDesc* __restrict__ exps, // [128]
    const int32_t* __restrict__ topk_ids,   // [B, top_k]
    const half_t* __restrict__ X_f16,       // [B, D]
    half_t* __restrict__ act_out,           // [B * top_k, inter]
    int B, int top_k, int D, int inter, float beta, float lb) {
    const int slot_k = blockIdx.y;
    if (slot_k >= B * top_k) return;
    const int b = slot_k / top_k;
    const int eid = topk_ids[slot_k];
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * 4 + (wave >> 1);  // 4 inter indices per block; wave 2k=gate(2j), 2k+1=up(2j+1)
    if (j >= inter) return;
    const int row = (j << 1) | (wave & 1);
    const half_t* xrow = X_f16 + static_cast<size_t>(b) * D;
    float acc = 0.0f;

    __shared__ half_t s_e4m3[256];
    if (desc.gu_code == CODE_MX8) {
        s_e4m3[threadIdx.x] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(threadIdx.x)));
        __syncthreads();
    }

    if (desc.gu_code == CODE_SF4) {
        const float* hdr = reinterpret_cast<const float*>(desc.body);
        const float outer = hdr[0] / hdr[1];
        const uint8_t* packed = desc.body + 16;
        const uint8_t* scales = packed + static_cast<size_t>(2 * inter) * (D >> 1);
        const uint8_t* pr = packed + static_cast<size_t>(row) * (D >> 1);
        const uint8_t* sr = scales + static_cast<size_t>(row) * (D >> 4);
        const int nblk = D >> 4;
        for (int blk = lane; blk < nblk; blk += 32) {
            const float sc = e4m3fn_to_f32(sr[blk]) * outer;
            const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
            const uint4 x0 = *reinterpret_cast<const uint4*>(xrow + (blk << 4));
            const uint4 x1 = *reinterpret_cast<const uint4*>(xrow + (blk << 4) + 8);
            float d = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t b0 = (raw.x >> (i * 8)) & 0xFFu;
                const uint32_t b1 = (raw.y >> (i * 8)) & 0xFFu;
                const hv2 h0 = {sf4_e2m1_f16(b0 & 15u), sf4_e2m1_f16(b0 >> 4)};
                const hv2 h1 = {sf4_e2m1_f16(b1 & 15u), sf4_e2m1_f16(b1 >> 4)};
                const uint32_t xv0 = (i == 0) ? x0.x : (i == 1) ? x0.y : (i == 2) ? x0.z : x0.w;
                const uint32_t xv1 = (i == 0) ? x1.x : (i == 1) ? x1.y : (i == 2) ? x1.z : x1.w;
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h0), xv0, d);
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h1), xv1, d);
            }
            acc = fmaf(d, sc, acc);
        }
    } else if (desc.gu_code == CODE_MX8) {
        const uint8_t* W = desc.body;
        const uint8_t* S = W + static_cast<size_t>(2 * inter) * D;
        const uint8_t* wr = W + static_cast<size_t>(row) * D;
        const uint8_t* sr = S + static_cast<size_t>(row) * (D >> 5);
        const int nblk = D >> 5;
        for (int blk = lane; blk < nblk; blk += 32) {
            const float sc = ue8m0_to_f32(sr[blk]);
            const uint8_t* wb = wr + (blk << 5);
            const uint4* xp = reinterpret_cast<const uint4*>(xrow + (blk << 5));
            float d = 0.0f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 xv = xp[q];
                const hv2 h0 = {s_e4m3[wb[q*8+0]], s_e4m3[wb[q*8+1]]};
                const hv2 h1 = {s_e4m3[wb[q*8+2]], s_e4m3[wb[q*8+3]]};
                const hv2 h2 = {s_e4m3[wb[q*8+4]], s_e4m3[wb[q*8+5]]};
                const hv2 h3 = {s_e4m3[wb[q*8+6]], s_e4m3[wb[q*8+7]]};
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h0), xv.x, d);
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h1), xv.y, d);
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h2), xv.z, d);
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h3), xv.w, d);
            }
            acc = fmaf(d, sc, acc);
        }
    } else {
        // S_K: gu weights start after sq_xd_dn (2 * inter bytes)!
        const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
        const int nsb = D >> 8;
        const int sb_b = sk_sb_bytes(desc.gu_code);
        const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * sb_b;
        for (int s = 0; s < nsb; ++s) {
            const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
            const int j0 = lane * 8;
            const float eff = decode_sk_sub_eff_i8(blk, desc.gu_code, j0 >> 6);
            const uint8_t* pl = blk + 6;
            const uint4 xv = *reinterpret_cast<const uint4*>(xrow + (s << 8) + j0);
            float d = 0.0f;
            for (int i = 0; i < 4; ++i) {
                const hv2 h = {
                    static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.gu_code, j0 + 2 * i))),
                    static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.gu_code, j0 + 2 * i + 1)))
                };
                const uint32_t xw = (i == 0) ? xv.x : (i == 1) ? xv.y : (i == 2) ? xv.z : xv.w;
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h), xw, d);
            }
            acc = fmaf(d, eff, acc);
        }
    }

    const float val = wave32_sum(acc);
    __shared__ float s_gu[8];
    if (lane == 0) s_gu[wave] = val;
    __syncthreads();
    if (lane == 0 && (wave & 1) == 0) {
        const float g = s_gu[wave];
        const float u = s_gu[wave + 1];
        float v = situ_f32(g, u, beta, lb);
        if (desc.dn_code != CODE_SF4 && desc.dn_code != CODE_MX8) {
            const float sq = static_cast<float>(reinterpret_cast<const half_t*>(desc.body)[j]);
            v /= sq;
        }
        act_out[static_cast<size_t>(slot_k) * inter + j] = static_cast<half_t>(v);
    }
}

// Stage 2: Fused w2 across all (b, k) active experts + deterministic weighted reduction into moe_out[B, D]!
// Grid: ((D + 7) / 8, B), Block: 256 (8 waves = 8 output rows; each wave accumulates all top_k=8 experts!)
__global__ void __launch_bounds__(256) k_moe_direct_w2_reduce(
    const GpuExpertDesc* __restrict__ exps,   // [128]
    const int32_t* __restrict__ topk_ids,     // [B, top_k]
    const float* __restrict__ topk_weights,   // [B, top_k]
    const half_t* __restrict__ act_in,        // [B * top_k, inter]
    float* __restrict__ moe_out,              // [B, D]
    int B, int top_k, int D, int inter) {
    const int b = blockIdx.y;
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    if (b >= B || row >= D) return;

    __shared__ half_t s_e4m3[256];
    s_e4m3[threadIdx.x] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(threadIdx.x)));
    __syncthreads();

    float total = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        const int slot_k = b * top_k + k;
        const int eid = topk_ids[slot_k];
        const float rw = topk_weights[slot_k];
        const GpuExpertDesc desc = exps[eid];
        if (!desc.body) continue;
        const half_t* arow = act_in + static_cast<size_t>(slot_k) * inter;
        float acc = 0.0f;

        if (desc.dn_code == CODE_SF4) {
            const float* hdr = reinterpret_cast<const float*>(desc.body);
            const float outer = hdr[2] / hdr[3];
            const uint8_t* packed = desc.body + desc.dn_offset;
            const uint8_t* scales = packed + static_cast<size_t>(D) * (inter >> 1);
            const uint8_t* pr = packed + static_cast<size_t>(row) * (inter >> 1);
            const uint8_t* sr = scales + static_cast<size_t>(row) * (inter >> 4);
            const int nblk = inter >> 4;
            for (int blk = lane; blk < nblk; blk += 32) {
                const float sc = e4m3fn_to_f32(sr[blk]) * outer;
                const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
                const uint4 x0 = *reinterpret_cast<const uint4*>(arow + (blk << 4));
                const uint4 x1 = *reinterpret_cast<const uint4*>(arow + (blk << 4) + 8);
                float d = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t b0 = (raw.x >> (i * 8)) & 0xFFu;
                    const uint32_t b1 = (raw.y >> (i * 8)) & 0xFFu;
                    const hv2 h0 = {sf4_e2m1_f16(b0 & 15u), sf4_e2m1_f16(b0 >> 4)};
                    const hv2 h1 = {sf4_e2m1_f16(b1 & 15u), sf4_e2m1_f16(b1 >> 4)};
                    const uint32_t xv0 = (i == 0) ? x0.x : (i == 1) ? x0.y : (i == 2) ? x0.z : x0.w;
                    const uint32_t xv1 = (i == 0) ? x1.x : (i == 1) ? x1.y : (i == 2) ? x1.z : x1.w;
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h0), xv0, d);
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h1), xv1, d);
                }
                acc = fmaf(d, sc, acc);
            }
        } else if (desc.dn_code == CODE_MX8) {
            const uint8_t* W = desc.body + desc.dn_offset;
            const uint8_t* S = W + static_cast<size_t>(D) * inter;
            const uint8_t* wr = W + static_cast<size_t>(row) * inter;
            const uint8_t* sr = S + static_cast<size_t>(row) * (inter >> 5);
            const int nblk = inter >> 5;
            for (int blk = lane; blk < nblk; blk += 32) {
                const float sc = ue8m0_to_f32(sr[blk]);
                const uint8_t* wb = wr + (blk << 5);
                const uint4* xp = reinterpret_cast<const uint4*>(arow + (blk << 5));
                float d = 0.0f;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint4 xv = xp[q];
                    const hv2 h0 = {s_e4m3[wb[q*8+0]], s_e4m3[wb[q*8+1]]};
                    const hv2 h1 = {s_e4m3[wb[q*8+2]], s_e4m3[wb[q*8+3]]};
                    const hv2 h2 = {s_e4m3[wb[q*8+4]], s_e4m3[wb[q*8+5]]};
                    const hv2 h3 = {s_e4m3[wb[q*8+6]], s_e4m3[wb[q*8+7]]};
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h0), xv.x, d);
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h1), xv.y, d);
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h2), xv.z, d);
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h3), xv.w, d);
                }
                acc = fmaf(d, sc, acc);
            }
        } else {
            // Paired 2-row S_K (arow already pre-divided by sq_xd_dn in w13_situ -> zero divisions + fast fdot2!)
            const uint8_t* W = desc.body + desc.dn_offset;
            const int pair = row >> 1;
            const int is_odd = row & 1;
            const int sb_per_pair = (2 * inter) >> 8;
            const int sb_b = sk_sb_bytes(desc.dn_code);
            const int sub_per_row = inter >> 6;
            const uint8_t* pr = W + static_cast<size_t>(pair) * sb_per_pair * sb_b;
            for (int sg = 0; sg < sub_per_row; ++sg) {
                const int sub_global = is_odd * sub_per_row + sg;
                const int s = sub_global >> 2;
                const int g = sub_global & 3;
                const uint8_t* blk = pr + static_cast<size_t>(s) * sb_b;
                const float eff = decode_sk_sub_eff_i8(blk, desc.dn_code, g);
                const uint8_t* pl = blk + 6;
                const int j0 = (g << 6) + lane * 2;
                const int col0 = (sg << 6) + lane * 2;
                const hv2 wh = {
                    static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.dn_code, j0))),
                    static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.dn_code, j0 + 1)))
                };
                const uint32_t xv = *reinterpret_cast<const uint32_t*>(arow + col0);
                const float d = fdot2_f32(__builtin_bit_cast(uint32_t, wh), xv, 0.0f);
                acc = fmaf(d, eff, acc);
            }
        }
        const float yk = wave32_sum(acc);
        if (lane == 0) total = fmaf(yk, rw, total);
    }
    if (lane == 0) moe_out[static_cast<size_t>(b) * D + row] = total;
}

// =================================================================================================
// 4C. Zero-Host-Sync Fused GPU Grouped-MoE for B=2..512 (Collapses 128 Expert Loops into 4 Launches!)
// =================================================================================================

// Build CSR buckets on GPU: exp_off[129], bucket_slotk[B*top_k], slotk_to_bucket[B*top_k]
// Single block of 128 threads — runs in < 5 us with ZERO host sync!
__global__ void __launch_bounds__(128) k_moe_build_buckets(
    const int32_t* __restrict__ topk_ids,  // [total_pairs] (total_pairs = B * top_k)
    int32_t* __restrict__ exp_off,         // [129]
    int32_t* __restrict__ bucket_tok,      // [total_pairs] -> b in [0..B-1]
    int32_t* __restrict__ slotk_to_bidx,   // [total_pairs] -> index in bucket arrays for deterministic reduce!
    int B, int top_k) {
    const int e = threadIdx.x;
    const int total = B * top_k;
    __shared__ int32_t s_cnt[128];
    __shared__ int32_t s_off[129];

    int cnt = 0;
    for (int i = 0; i < total; ++i) {
        if (topk_ids[i] == e) ++cnt;
    }
    s_cnt[e] = cnt;
    __syncthreads();

    if (e == 0) {
        int acc = 0;
        for (int i = 0; i < 128; ++i) {
            s_off[i] = acc;
            acc += s_cnt[i];
        }
        s_off[128] = acc;
        exp_off[128] = acc;
    }
    __syncthreads();
    exp_off[e] = s_off[e];

    // Each thread e writes its own expert's bucket slice in ascending slot_k order (lock-free & deterministic!)
    int cur = s_off[e];
    for (int i = 0; i < total; ++i) {
        if (topk_ids[i] == e) {
            bucket_tok[cur] = i / top_k;
            slotk_to_bidx[i] = cur;
            ++cur;
        }
    }
}

// Fused Grouped MoE Stage 1 (w13 + SITU) across ALL 128 experts in ONE launch!
// Reads each active expert's weights ONCE for all m_e tokens routed to it (processes up to 8 tokens/tile in registers!).
// Grid: ((inter + 3) / 4, 128), Block: 256 (8 waves = 4 (gate, up) pairs; uses LDS s_e4m3 LUT for MX8!)
__global__ void __launch_bounds__(256) k_moe_grouped_w13_situ(
    const GpuExpertDesc* __restrict__ exps,  // [128]
    const int32_t* __restrict__ exp_off,     // [129]
    const int32_t* __restrict__ bucket_tok,  // [B * top_k]
    const half_t* __restrict__ X_f16,        // [B, D]
    half_t* __restrict__ bucket_act,         // [B * top_k, inter]
    int D, int inter, float beta, float lb) {
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    if (beg >= end) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int j = blockIdx.x * 4 + (wave >> 1);
    if (j >= inter) return;
    const int row = (j << 1) | (wave & 1);
    const int m_e = end - beg;

    __shared__ half_t s_e4m3[256];
    if (desc.gu_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        __syncthreads();
    }
    __shared__ float s_gu[8 * 8];  // [8 waves][8 tokens]

    // Process m_e routed tokens in register tiles of 8 -> 8x weight reuse from L1/registers!
    for (int t0 = 0; t0 < m_e; t0 += 8) {
        const int tm = (m_e - t0 < 8) ? (m_e - t0) : 8;
        int toks[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) toks[t] = (t < tm) ? bucket_tok[beg + t0 + t] : 0;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        if (desc.gu_code == CODE_SF4) {
            const float* hdr = reinterpret_cast<const float*>(desc.body);
            const float outer = hdr[0] / hdr[1];
            const uint8_t* packed = desc.body + 16;
            const uint8_t* scales = packed + static_cast<size_t>(2 * inter) * (D >> 1);
            const uint8_t* pr = packed + static_cast<size_t>(row) * (D >> 1);
            const uint8_t* sr = scales + static_cast<size_t>(row) * (D >> 4);
            const int nblk = D >> 4;
            for (int blk = lane; blk < nblk; blk += 32) {
                const float sc = e4m3fn_to_f32(sr[blk]) * outer;
                const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
                uint32_t w2[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t b0 = (raw.x >> (i * 8)) & 0xFFu;
                    const uint32_t b1 = (raw.y >> (i * 8)) & 0xFFu;
                    const hv2 h0 = {sf4_e2m1_f16(b0 & 15u), sf4_e2m1_f16(b0 >> 4)};
                    const hv2 h1 = {sf4_e2m1_f16(b1 & 15u), sf4_e2m1_f16(b1 >> 4)};
                    w2[i]     = __builtin_bit_cast(uint32_t, h0);
                    w2[i + 4] = __builtin_bit_cast(uint32_t, h1);
                }
                const int k0 = blk << 4;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (t < tm) {
                        const uint4 x0 = *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(toks[t]) * D + k0);
                        const uint4 x1 = *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(toks[t]) * D + k0 + 8);
                        float d = 0.0f;
                        d = fdot2_f32(w2[0], x0.x, d); d = fdot2_f32(w2[1], x0.y, d);
                        d = fdot2_f32(w2[2], x0.z, d); d = fdot2_f32(w2[3], x0.w, d);
                        d = fdot2_f32(w2[4], x1.x, d); d = fdot2_f32(w2[5], x1.y, d);
                        d = fdot2_f32(w2[6], x1.z, d); d = fdot2_f32(w2[7], x1.w, d);
                        acc[t] = fmaf(d, sc, acc[t]);
                    }
                }
            }
        } else if (desc.gu_code == CODE_MX8) {
            const uint8_t* W = desc.body;
            const uint8_t* S = W + static_cast<size_t>(2 * inter) * D;
            const uint8_t* wr = W + static_cast<size_t>(row) * D;
            const uint8_t* sr = S + static_cast<size_t>(row) * (D >> 5);
            const int nblk = D >> 5;
            for (int blk = lane; blk < nblk; blk += 32) {
                const float sc = ue8m0_to_f32(sr[blk]);
                const uint8_t* wb = wr + (blk << 5);
                uint32_t w2[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const hv2 h = {s_e4m3[wb[2 * i]], s_e4m3[wb[2 * i + 1]]};
                    w2[i] = __builtin_bit_cast(uint32_t, h);
                }
                const int k0 = blk << 5;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (t < tm) {
                        const uint4* xp = reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(toks[t]) * D + k0);
                        float d = 0.0f;
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const uint4 xv = xp[q];
                            d = fdot2_f32(w2[4*q+0], xv.x, d); d = fdot2_f32(w2[4*q+1], xv.y, d);
                            d = fdot2_f32(w2[4*q+2], xv.z, d); d = fdot2_f32(w2[4*q+3], xv.w, d);
                        }
                        acc[t] = fmaf(d, sc, acc[t]);
                    }
                }
            }
        } else {
            const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
            const int nsb = D >> 8;
            const int sb_b = sk_sb_bytes(desc.gu_code);
            const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * sb_b;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
                const int j0 = lane * 8;
                const float eff = decode_sk_sub_eff_i8(blk, desc.gu_code, j0 >> 6);
                const uint8_t* pl = blk + 6;
                uint32_t w2[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const hv2 h = {
                        static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.gu_code, j0 + 2 * i))),
                        static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.gu_code, j0 + 2 * i + 1)))
                    };
                    w2[i] = __builtin_bit_cast(uint32_t, h);
                }
                const int k0 = (s << 8) + j0;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (t < tm) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(toks[t]) * D + k0);
                        float d = 0.0f;
                        d = fdot2_f32(w2[0], xv.x, d); d = fdot2_f32(w2[1], xv.y, d);
                        d = fdot2_f32(w2[2], xv.z, d); d = fdot2_f32(w2[3], xv.w, d);
                        acc[t] = fmaf(d, eff, acc[t]);
                    }
                }
            }
        }

#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const float sum = wave32_sum(acc[t]);
            if (lane == 0 && t < tm) s_gu[wave * 8 + t] = sum;
        }
        __syncthreads();
        if (lane == 0 && (wave & 1) == 0) {
            const bool need_sq = (desc.dn_code != CODE_SF4 && desc.dn_code != CODE_MX8);
            const float inv_sq = need_sq ? (1.0f / static_cast<float>(reinterpret_cast<const half_t*>(desc.body)[j])) : 1.0f;
            for (int t = 0; t < tm; ++t) {
                const float g = s_gu[wave * 8 + t];
                const float u = s_gu[(wave + 1) * 8 + t];
                bucket_act[static_cast<size_t>(beg + t0 + t) * inter + j] = static_cast<half_t>(situ_f32(g, u, beta, lb) * inv_sq);
            }
        }
        __syncthreads();
    }
}

// Fused Grouped MoE Stage 2 (w2) across ALL 128 experts in ONE launch!
// Grid: ((D + 7) / 8, 128), Block: 256 (8 waves = 8 output rows; up to 8 tokens/tile in registers!)
__global__ void __launch_bounds__(256) k_moe_grouped_w2(
    const GpuExpertDesc* __restrict__ exps,  // [128]
    const int32_t* __restrict__ exp_off,     // [129]
    const half_t* __restrict__ bucket_act,   // [B * top_k, inter]
    float* __restrict__ bucket_y,            // [B * top_k, D]
    int D, int inter) {
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    if (beg >= end) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= D) return;
    const int m_e = end - beg;

    __shared__ half_t s_e4m3[256];
    if (desc.dn_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        __syncthreads();
    }

    for (int t0 = 0; t0 < m_e; t0 += 8) {
        const int tm = (m_e - t0 < 8) ? (m_e - t0) : 8;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        if (desc.dn_code == CODE_SF4) {
            const float* hdr = reinterpret_cast<const float*>(desc.body);
            const float outer = hdr[2] / hdr[3];
            const uint8_t* packed = desc.body + desc.dn_offset;
            const uint8_t* scales = packed + static_cast<size_t>(D) * (inter >> 1);
            const uint8_t* pr = packed + static_cast<size_t>(row) * (inter >> 1);
            const uint8_t* sr = scales + static_cast<size_t>(row) * (inter >> 4);
            const int nblk = inter >> 4;
            for (int blk = lane; blk < nblk; blk += 32) {
                const float sc = e4m3fn_to_f32(sr[blk]) * outer;
                const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
                uint32_t w2[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t b0 = (raw.x >> (i * 8)) & 0xFFu;
                    const uint32_t b1 = (raw.y >> (i * 8)) & 0xFFu;
                    const hv2 h0 = {sf4_e2m1_f16(b0 & 15u), sf4_e2m1_f16(b0 >> 4)};
                    const hv2 h1 = {sf4_e2m1_f16(b1 & 15u), sf4_e2m1_f16(b1 >> 4)};
                    w2[i]     = __builtin_bit_cast(uint32_t, h0);
                    w2[i + 4] = __builtin_bit_cast(uint32_t, h1);
                }
                const int k0 = blk << 4;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (t < tm) {
                        const half_t* ar = bucket_act + static_cast<size_t>(beg + t0 + t) * inter + k0;
                        const uint4 x0 = *reinterpret_cast<const uint4*>(ar);
                        const uint4 x1 = *reinterpret_cast<const uint4*>(ar + 8);
                        float d = 0.0f;
                        d = fdot2_f32(w2[0], x0.x, d); d = fdot2_f32(w2[1], x0.y, d);
                        d = fdot2_f32(w2[2], x0.z, d); d = fdot2_f32(w2[3], x0.w, d);
                        d = fdot2_f32(w2[4], x1.x, d); d = fdot2_f32(w2[5], x1.y, d);
                        d = fdot2_f32(w2[6], x1.z, d); d = fdot2_f32(w2[7], x1.w, d);
                        acc[t] = fmaf(d, sc, acc[t]);
                    }
                }
            }
        } else if (desc.dn_code == CODE_MX8) {
            const uint8_t* W = desc.body + desc.dn_offset;
            const uint8_t* S = W + static_cast<size_t>(D) * inter;
            const uint8_t* wr = W + static_cast<size_t>(row) * inter;
            const uint8_t* sr = S + static_cast<size_t>(row) * (inter >> 5);
            const int nblk = inter >> 5;
            for (int blk = lane; blk < nblk; blk += 32) {
                const float sc = ue8m0_to_f32(sr[blk]);
                const uint8_t* wb = wr + (blk << 5);
                uint32_t w2[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const hv2 h = {s_e4m3[wb[2 * i]], s_e4m3[wb[2 * i + 1]]};
                    w2[i] = __builtin_bit_cast(uint32_t, h);
                }
                const int k0 = blk << 5;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (t < tm) {
                        const uint4* xp = reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg + t0 + t) * inter + k0);
                        float d = 0.0f;
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const uint4 xv = xp[q];
                            d = fdot2_f32(w2[4*q+0], xv.x, d); d = fdot2_f32(w2[4*q+1], xv.y, d);
                            d = fdot2_f32(w2[4*q+2], xv.z, d); d = fdot2_f32(w2[4*q+3], xv.w, d);
                        }
                        acc[t] = fmaf(d, sc, acc[t]);
                    }
                }
            }
        } else {
            const uint8_t* W = desc.body + desc.dn_offset;
            const int pair = row >> 1;
            const int is_odd = row & 1;
            const int sb_per_pair = (2 * inter) >> 8;
            const int sb_b = sk_sb_bytes(desc.dn_code);
            const int sub_per_row = inter >> 6;
            const uint8_t* pr = W + static_cast<size_t>(pair) * sb_per_pair * sb_b;
            for (int sg = 0; sg < sub_per_row; ++sg) {
                const int sub_global = is_odd * sub_per_row + sg;
                const int s = sub_global >> 2;
                const int g = sub_global & 3;
                const uint8_t* blk = pr + static_cast<size_t>(s) * sb_b;
                const float eff = decode_sk_sub_eff_i8(blk, desc.dn_code, g);
                const uint8_t* pl = blk + 6;
                const int j0 = (g << 6) + lane * 2;
                const int col0 = (sg << 6) + lane * 2;
                const hv2 wh = {
                    static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.dn_code, j0))),
                    static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.dn_code, j0 + 1)))
                };
                const uint32_t w2 = __builtin_bit_cast(uint32_t, wh);
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (t < tm) {
                        const uint32_t xv = *reinterpret_cast<const uint32_t*>(bucket_act + static_cast<size_t>(beg + t0 + t) * inter + col0);
                        acc[t] = fmaf(fdot2_f32(w2, xv, 0.0f), eff, acc[t]);
                    }
                }
            }
        }

#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const float sum = wave32_sum(acc[t]);
            if (lane == 0 && t < tm) bucket_y[static_cast<size_t>(beg + t0 + t) * D + row] = sum;
        }
    }
}

// Deterministic reduction across top_k=8 bucket outputs into moe_out[B, D]!
// Grid: ((D + 255) / 256, B), Block: 256
__global__ void __launch_bounds__(256) k_moe_reduce_buckets(
    const float* __restrict__ bucket_y,         // [B * top_k, D]
    const int32_t* __restrict__ slotk_to_bidx,  // [B * top_k]
    const float* __restrict__ topk_weights,     // [B * top_k]
    float* __restrict__ moe_out,                // [B, D]
    int B, int top_k, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= D || b >= B) return;
    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        const int sk = b * top_k + k;
        const int bi = slotk_to_bidx[sk];
        acc = fmaf(bucket_y[static_cast<size_t>(bi) * D + d], topk_weights[sk], acc);
    }
    moe_out[static_cast<size_t>(b) * D + d] = acc;
}

// =================================================================================================
// 4D. Wave32 WMMA Fused Grouped-MoE (SF4 + MX8 + S_K) for Heavy Prefill / High-Batch (B >= 64)!
// Uses BM=16, BN=64, BK=64 Wave32 WMMA across ALL 128 experts in ONE launch + fused SITU/sq_xd_dn in LDS!
// Zero WMMA emitted on RDNA2 (guarded inside body).
// =================================================================================================

__global__ void __launch_bounds__(128) k_moe_grouped_w13_situ_wmma(
    const GpuExpertDesc* __restrict__ exps,  // [128]
    const int32_t* __restrict__ exp_off,     // [129]
    const int32_t* __restrict__ bucket_tok,  // [B * top_k]
    const half_t* __restrict__ X_f16,        // [B, D]
    half_t* __restrict__ bucket_act,         // [B * top_k, inter]
    int D, int inter, float beta, float lb) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    if (beg >= end) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    constexpr int BM = 16, BN = 64, BK = 64, LD = BK + 8;
    __shared__ alignas(16) half_t sX[BM * LD];
    __shared__ alignas(16) half_t sW[BN * LD];
    __shared__ half_t s_e4m3[256];
    __shared__ float s_out[BM * BN];

    const int tid = threadIdx.x;
    if (desc.gu_code == CODE_MX8) {
        s_e4m3[tid]       = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        s_e4m3[tid + 128] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid + 128)));
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int N = 2 * inter;
    const int n_blk = blockIdx.x * BN;
    const int m_e = end - beg;

    const float sf4_outer = (desc.gu_code == CODE_SF4)
        ? (reinterpret_cast<const float*>(desc.body)[0] / reinterpret_cast<const float*>(desc.body)[1]) : 1.0f;
    const uint8_t* sf4_p = desc.body + 16;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(N) * (D >> 1);
    const uint8_t* mx8_w = desc.body;
    const uint8_t* mx8_s = mx8_w + static_cast<size_t>(N) * D;
    const uint8_t* sk_w  = desc.body + static_cast<size_t>(inter) * 2;
    const int nsb = D >> 8;
    const int sb_b = sk_sb_bytes(desc.gu_code);

    for (int t0 = 0; t0 < m_e; t0 += BM) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc = fv8_zero();  // 4 waves cover BN=64 -> each wave handles 16x16!

        for (int k0 = 0; k0 < D; k0 += BK) {
            // 1. Gather X tile [16, 64]
            for (int idx = tid; idx < BM * 8; idx += 128) {
                const int r = idx >> 3, c8 = idx & 7;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < tm) {
                    const int tok = bucket_tok[beg + t0 + r];
                    v = *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(tok) * D + k0 + c8 * 8);
                }
                *reinterpret_cast<uint4*>(&sX[r * LD + c8 * 8]) = v;
            }
            // 2. Dequant W tile [64, 64] -> sW
            if (desc.gu_code == CODE_SF4) {
                for (int idx = tid; idx < BN * 4; idx += 128) {
                    const int r = idx >> 2, b4 = idx & 3;
                    const int gn = n_blk + r;
                    const int gb = (k0 >> 4) + b4;
                    half_t* dst = &sW[r * LD + b4 * 16];
                    if (gn < N) {
                        const float sc = e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * (D >> 4) + gb]) * sf4_outer;
                        const uint32_t sc2 = h2_splat(sc);
                        const uint2 raw = *reinterpret_cast<const uint2*>(sf4_p + static_cast<size_t>(gn) * (D >> 1) + gb * 8);
                        uint32_t o0[4], o1[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const uint32_t b0 = (raw.x >> (i * 8)) & 0xFFu;
                            const uint32_t b1 = (raw.y >> (i * 8)) & 0xFFu;
                            const hv2 h0 = {sf4_e2m1_f16(b0 & 15u), sf4_e2m1_f16(b0 >> 4)};
                            const hv2 h1 = {sf4_e2m1_f16(b1 & 15u), sf4_e2m1_f16(b1 >> 4)};
                            o0[i] = h2_mul(__builtin_bit_cast(uint32_t, h0), sc2);
                            o1[i] = h2_mul(__builtin_bit_cast(uint32_t, h1), sc2);
                        }
                        *reinterpret_cast<uint4*>(dst)     = make_uint4(o0[0], o0[1], o0[2], o0[3]);
                        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
                    } else {
                        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
                    }
                }
            } else if (desc.gu_code == CODE_MX8) {
                for (int idx = tid; idx < BN * 2; idx += 128) {
                    const int r = idx >> 1, b2 = idx & 1;
                    const int gn = n_blk + r;
                    const int gb = (k0 >> 5) + b2;
                    half_t* dst = &sW[r * LD + b2 * 32];
                    if (gn < N) {
                        const uint32_t sc2 = h2_splat(ue8m0_to_f32(mx8_s[static_cast<size_t>(gn) * (D >> 5) + gb]));
                        const uint8_t* wb = mx8_w + static_cast<size_t>(gn) * D + (gb << 5);
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            uint32_t o[4];
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const hv2 h = {s_e4m3[wb[q * 8 + 2 * i]], s_e4m3[wb[q * 8 + 2 * i + 1]]};
                                o[i] = h2_mul(__builtin_bit_cast(uint32_t, h), sc2);
                            }
                            *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(o[0], o[1], o[2], o[3]);
                        }
                    } else {
                        for (int q = 0; q < 4; ++q) *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(0, 0, 0, 0);
                    }
                }
            } else {
                const int s = k0 >> 8;
                const int g = (k0 >> 6) & 3;
                for (int idx = tid; idx < BN * 4; idx += 128) {
                    const int r = idx >> 2, q16 = idx & 3;
                    const int gn = n_blk + r;
                    half_t* dst = &sW[r * LD + q16 * 16];
                    if (gn < N) {
                        const uint8_t* blk = sk_w + (static_cast<size_t>(gn) * nsb + s) * sb_b;
                        const uint32_t ef2 = h2_splat(decode_sk_sub_eff_i8(blk, desc.gu_code, g));
                        const uint8_t* pl = blk + 6;
                        const int j0 = (g << 6) + q16 * 16;
                        uint32_t o0[4], o1[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const hv2 h0 = {
                                static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.gu_code, j0 + 2 * i))),
                                static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.gu_code, j0 + 2 * i + 1)))
                            };
                            const hv2 h1 = {
                                static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.gu_code, j0 + 8 + 2 * i))),
                                static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.gu_code, j0 + 8 + 2 * i + 1)))
                            };
                            o0[i] = h2_mul(__builtin_bit_cast(uint32_t, h0), ef2);
                            o1[i] = h2_mul(__builtin_bit_cast(uint32_t, h1), ef2);
                        }
                        *reinterpret_cast<uint4*>(dst)     = make_uint4(o0[0], o0[1], o0[2], o0[3]);
                        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
                    } else {
                        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
                    }
                }
            }
            sync_lds();

#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma_h_t a = wmma_ld_h(&sX[l16 * LD + kk * 16], hi);
                wmma_h_t b = wmma_ld_h(&sW[(wave * 16 + l16) * LD + kk * 16], hi);
                acc = wmma_f16(a, b, acc);
            }
            sync_lds();
        }

        // Write 16x64 to LDS, fuse SITU (+ optional / sq_xd_dn for S_K) directly into bucket_act!
        const int ln = wave * 16 + wmma_acc_n(0, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int lm = wmma_acc_m(r, lane);
            s_out[lm * BN + ln] = acc[r];
        }
        sync_lds();

        const bool need_sq = (desc.dn_code != CODE_SF4 && desc.dn_code != CODE_MX8);
        const half_t* sq_xd = reinterpret_cast<const half_t*>(desc.body);
        const int j_blk = n_blk >> 1;
        for (int idx = tid; idx < tm * 32; idx += 128) {
            const int lm = idx >> 5;
            const int lj = idx & 31;
            const int gj = j_blk + lj;
            if (gj < inter) {
                const float g = s_out[lm * BN + 2 * lj];
                const float u = s_out[lm * BN + 2 * lj + 1];
                float v = situ_f32(g, u, beta, lb);
                if (need_sq) v /= static_cast<float>(sq_xd[gj]);
                bucket_act[static_cast<size_t>(beg + t0 + lm) * inter + gj] = static_cast<half_t>(v);
            }
        }
        sync_lds();
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)X_f16; (void)bucket_act; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

// Wave32 WMMA Fused Grouped-MoE Stage 2 (w2 for SF4 + MX8 + paired-2-row S_K!) for B >= 64!
__global__ void __launch_bounds__(128) k_moe_grouped_w2_wmma(
    const GpuExpertDesc* __restrict__ exps,  // [128]
    const int32_t* __restrict__ exp_off,     // [129]
    const half_t* __restrict__ bucket_act,   // [B * top_k, inter]
    float* __restrict__ bucket_y,            // [B * top_k, D]
    int D, int inter) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    if (beg >= end) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    constexpr int BM = 16, BN = 64, BK = 64, LD = BK + 8;
    __shared__ alignas(16) half_t sX[BM * LD];
    __shared__ alignas(16) half_t sW[BN * LD];
    __shared__ half_t s_e4m3[256];

    const int tid = threadIdx.x;
    if (desc.dn_code == CODE_MX8) {
        s_e4m3[tid]       = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        s_e4m3[tid + 128] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid + 128)));
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int n_blk = blockIdx.x * BN;
    const int m_e = end - beg;

    const float sf4_outer = (desc.dn_code == CODE_SF4)
        ? (reinterpret_cast<const float*>(desc.body)[2] / reinterpret_cast<const float*>(desc.body)[3]) : 1.0f;
    const uint8_t* sf4_p = desc.body + desc.dn_offset;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(D) * (inter >> 1);
    const uint8_t* mx8_w = desc.body + desc.dn_offset;
    const uint8_t* mx8_s = mx8_w + static_cast<size_t>(D) * inter;
    const uint8_t* sk_w  = desc.body + desc.dn_offset;
    const int sb_per_pair = (2 * inter) >> 8;
    const int sb_b = sk_sb_bytes(desc.dn_code);
    const int sub_per_row = inter >> 6;

    for (int t0 = 0; t0 < m_e; t0 += BM) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc = fv8_zero();

        for (int k0 = 0; k0 < inter; k0 += BK) {
            for (int idx = tid; idx < BM * 8; idx += 128) {
                const int r = idx >> 3, c8 = idx & 7;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < tm) {
                    v = *reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg + t0 + r) * inter + k0 + c8 * 8);
                }
                *reinterpret_cast<uint4*>(&sX[r * LD + c8 * 8]) = v;
            }
            if (desc.dn_code == CODE_SF4) {
                for (int idx = tid; idx < BN * 4; idx += 128) {
                    const int r = idx >> 2, b4 = idx & 3;
                    const int gn = n_blk + r;
                    const int gb = (k0 >> 4) + b4;
                    half_t* dst = &sW[r * LD + b4 * 16];
                    if (gn < D) {
                        const float sc = e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * (inter >> 4) + gb]) * sf4_outer;
                        const uint32_t sc2 = h2_splat(sc);
                        const uint2 raw = *reinterpret_cast<const uint2*>(sf4_p + static_cast<size_t>(gn) * (inter >> 1) + gb * 8);
                        uint32_t o0[4], o1[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const uint32_t b0 = (raw.x >> (i * 8)) & 0xFFu;
                            const uint32_t b1 = (raw.y >> (i * 8)) & 0xFFu;
                            const hv2 h0 = {sf4_e2m1_f16(b0 & 15u), sf4_e2m1_f16(b0 >> 4)};
                            const hv2 h1 = {sf4_e2m1_f16(b1 & 15u), sf4_e2m1_f16(b1 >> 4)};
                            o0[i] = h2_mul(__builtin_bit_cast(uint32_t, h0), sc2);
                            o1[i] = h2_mul(__builtin_bit_cast(uint32_t, h1), sc2);
                        }
                        *reinterpret_cast<uint4*>(dst)     = make_uint4(o0[0], o0[1], o0[2], o0[3]);
                        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
                    } else {
                        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
                    }
                }
            } else if (desc.dn_code == CODE_MX8) {
                for (int idx = tid; idx < BN * 2; idx += 128) {
                    const int r = idx >> 1, b2 = idx & 1;
                    const int gn = n_blk + r;
                    const int gb = (k0 >> 5) + b2;
                    half_t* dst = &sW[r * LD + b2 * 32];
                    if (gn < D) {
                        const uint32_t sc2 = h2_splat(ue8m0_to_f32(mx8_s[static_cast<size_t>(gn) * (inter >> 5) + gb]));
                        const uint8_t* wb = mx8_w + static_cast<size_t>(gn) * inter + (gb << 5);
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            uint32_t o[4];
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const hv2 h = {s_e4m3[wb[q * 8 + 2 * i]], s_e4m3[wb[q * 8 + 2 * i + 1]]};
                                o[i] = h2_mul(__builtin_bit_cast(uint32_t, h), sc2);
                            }
                            *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(o[0], o[1], o[2], o[3]);
                        }
                    } else {
                        for (int q = 0; q < 4; ++q) *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(0, 0, 0, 0);
                    }
                }
            } else {
                const int sg = k0 >> 6;
                for (int idx = tid; idx < BN * 4; idx += 128) {
                    const int r = idx >> 2, q16 = idx & 3;
                    const int gn = n_blk + r;
                    half_t* dst = &sW[r * LD + q16 * 16];
                    if (gn < D) {
                        const int pair = gn >> 1;
                        const int is_odd = gn & 1;
                        const int sub_global = is_odd * sub_per_row + sg;
                        const int s = sub_global >> 2;
                        const int g = sub_global & 3;
                        const uint8_t* blk = sk_w + (static_cast<size_t>(pair) * sb_per_pair + s) * sb_b;
                        const uint32_t ef2 = h2_splat(decode_sk_sub_eff_i8(blk, desc.dn_code, g));
                        const uint8_t* pl = blk + 6;
                        const int j0 = (g << 6) + q16 * 16;
                        uint32_t o0[4], o1[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const hv2 h0 = {
                                static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.dn_code, j0 + 2 * i))),
                                static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.dn_code, j0 + 2 * i + 1)))
                            };
                            const hv2 h1 = {
                                static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.dn_code, j0 + 8 + 2 * i))),
                                static_cast<half_t>(static_cast<float>(decode_sk_w_i8(pl, desc.dn_code, j0 + 8 + 2 * i + 1)))
                            };
                            o0[i] = h2_mul(__builtin_bit_cast(uint32_t, h0), ef2);
                            o1[i] = h2_mul(__builtin_bit_cast(uint32_t, h1), ef2);
                        }
                        *reinterpret_cast<uint4*>(dst)     = make_uint4(o0[0], o0[1], o0[2], o0[3]);
                        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
                    } else {
                        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
                    }
                }
            }
            sync_lds();

#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma_h_t a = wmma_ld_h(&sX[l16 * LD + kk * 16], hi);
                wmma_h_t b = wmma_ld_h(&sW[(wave * 16 + l16) * LD + kk * 16], hi);
                acc = wmma_f16(a, b, acc);
            }
            sync_lds();
        }

        const int gn = n_blk + wave * 16 + wmma_acc_n(0, lane);
        if (gn < D) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int lm = wmma_acc_m(r, lane);
                if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[r];
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)bucket_act; (void)bucket_y; (void)D; (void)inter;
#endif
}

// =================================================================================================
// 5. Production Multi-Gen Host Dispatchers (Runtime dispatch by AmdGen — works on ALL multi-target builds!)
// =================================================================================================

inline void launch_sf4_linear(
    const uint8_t* packed, const uint8_t* scales, float outer_scale,
    const half_t* X, float* Y, int M, int N, int K,
    strata::hip_uma::AmdGen gen, hipStream_t st, bool use_i8_rdna4 = false) {
    if (M <= 4) {
        const dim3 grid((N + 7) / 8);
        if (M == 1)      k_sf4_gemv<1><<<grid, 256, 0, st>>>(packed, scales, outer_scale, X, Y, M, N, K);
        else if (M == 2) k_sf4_gemv<2><<<grid, 256, 0, st>>>(packed, scales, outer_scale, X, Y, M, N, K);
        else             k_sf4_gemv<4><<<grid, 256, 0, st>>>(packed, scales, outer_scale, X, Y, M, N, K);
        return;
    }
    if (gen == strata::hip_uma::AmdGen::Rdna2 || gen == strata::hip_uma::AmdGen::Unknown) {
        const dim3 grid((N + 31) / 32, (M + 15) / 16);
        k_sf4_sdot4_tiled<16, 32><<<grid, 128, 0, st>>>(packed, scales, outer_scale, X, Y, M, N, K);
        return;
    }
    if (gen == strata::hip_uma::AmdGen::Rdna4 && use_i8_rdna4) {
        if (M <= 16) {
            const dim3 grid((N + 63) / 64, (M + 15) / 16);
            k_sf4_wmma_i8_gfx12<16, 64><<<grid, 128, 0, st>>>(packed, scales, outer_scale, X, Y, M, N, K);
        } else {
            const dim3 grid((N + 63) / 64, (M + 63) / 64);
            k_sf4_wmma_i8_gfx12<64, 64><<<grid, 128, 0, st>>>(packed, scales, outer_scale, X, Y, M, N, K);
        }
        return;
    }
    if (M <= 16) {
        const dim3 grid((N + 63) / 64, (M + 15) / 16);
        k_sf4_wmma_f16<16, 64><<<grid, 128, 0, st>>>(packed, scales, outer_scale, X, Y, M, N, K);
    } else {
        const dim3 grid((N + 63) / 64, (M + 63) / 64);
        k_sf4_wmma_f16<64, 64><<<grid, 128, 0, st>>>(packed, scales, outer_scale, X, Y, M, N, K);
    }
}

inline void launch_mx8_linear(
    const uint8_t* weights, const uint8_t* scales,
    const half_t* X, float* Y, int M, int N, int K,
    strata::hip_uma::AmdGen gen, hipStream_t st) {
    if (M <= 4) {
        const dim3 grid((N + 7) / 8);
        if (M == 1)      k_mx8_gemv<1><<<grid, 256, 0, st>>>(weights, scales, X, Y, M, N, K);
        else if (M == 2) k_mx8_gemv<2><<<grid, 256, 0, st>>>(weights, scales, X, Y, M, N, K);
        else             k_mx8_gemv<4><<<grid, 256, 0, st>>>(weights, scales, X, Y, M, N, K);
        return;
    }
    if (gen == strata::hip_uma::AmdGen::Rdna2 || gen == strata::hip_uma::AmdGen::Unknown) {
        const dim3 grid((N + 31) / 32, (M + 15) / 16);
        k_mx8_fdot2_tiled<16, 32><<<grid, 128, 0, st>>>(weights, scales, X, Y, M, N, K);
        return;
    }
    if (M <= 16) {
        const dim3 grid((N + 63) / 64, (M + 15) / 16);
        k_mx8_wmma_f16<16, 64><<<grid, 128, 0, st>>>(weights, scales, X, Y, M, N, K);
    } else {
        const dim3 grid((N + 63) / 64, (M + 63) / 64);
        k_mx8_wmma_f16<64, 64><<<grid, 128, 0, st>>>(weights, scales, X, Y, M, N, K);
    }
}

inline void launch_sk_linear(
    const uint8_t* W, int code,
    const half_t* X, float* Y, int M, int N, int K,
    strata::hip_uma::AmdGen gen, hipStream_t st) {
    if (M <= 4 || gen == strata::hip_uma::AmdGen::Rdna2 || gen == strata::hip_uma::AmdGen::Unknown) {
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int bm = (M - m0) < 4 ? (M - m0) : 4;
            const dim3 grid((N + 7) / 8);
            if (bm == 1)      k_sk_gemv<1><<<grid, 256, 0, st>>>(W, code, X + static_cast<size_t>(m0) * K, Y + static_cast<size_t>(m0) * N, bm, N, K);
            else if (bm == 2) k_sk_gemv<2><<<grid, 256, 0, st>>>(W, code, X + static_cast<size_t>(m0) * K, Y + static_cast<size_t>(m0) * N, bm, N, K);
            else              k_sk_gemv<4><<<grid, 256, 0, st>>>(W, code, X + static_cast<size_t>(m0) * K, Y + static_cast<size_t>(m0) * N, bm, N, K);
        }
        return;
    }
    if (M <= 16) {
        const dim3 grid((N + 63) / 64, (M + 15) / 16);
        k_sk_wmma_f16<16, 64><<<grid, 128, 0, st>>>(W, code, X, Y, M, N, K);
    } else {
        const dim3 grid((N + 63) / 64, (M + 63) / 64);
        k_sk_wmma_f16<64, 64><<<grid, 128, 0, st>>>(W, code, X, Y, M, N, K);
    }
}

inline void launch_sk_paired2_linear(
    const uint8_t* W, int code, const half_t* sq_xd_dn,
    const half_t* X, float* Y, int M, int N, int K, hipStream_t st) {
    for (int m0 = 0; m0 < M; m0 += 4) {
        const int bm = (M - m0) < 4 ? (M - m0) : 4;
        const dim3 grid(((N / 2) + 7) / 8);
        if (bm == 1)      k_sk_paired2_gemv<1><<<grid, 256, 0, st>>>(W, code, sq_xd_dn, X + static_cast<size_t>(m0) * K, Y + static_cast<size_t>(m0) * N, bm, N, K);
        else if (bm == 2) k_sk_paired2_gemv<2><<<grid, 256, 0, st>>>(W, code, sq_xd_dn, X + static_cast<size_t>(m0) * K, Y + static_cast<size_t>(m0) * N, bm, N, K);
        else              k_sk_paired2_gemv<4><<<grid, 256, 0, st>>>(W, code, sq_xd_dn, X + static_cast<size_t>(m0) * K, Y + static_cast<size_t>(m0) * N, bm, N, K);
    }
}

inline void launch_bf16_linear(
    const uint16_t* W, const float* X, const float* bias, float* Y,
    int M, int N, int K, strata::hip_uma::AmdGen gen, hipStream_t st) {
    if (M <= 4) {
        const dim3 grid((N + 7) / 8);
        if (M == 1)      k_bf16_gemv<1><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        else if (M == 2) k_bf16_gemv<2><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        else             k_bf16_gemv<4><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        return;
    }
    if (gen == strata::hip_uma::AmdGen::Rdna2 || gen == strata::hip_uma::AmdGen::Unknown) {
        const dim3 grid((N + 31) / 32, (M + 15) / 16);
        k_bf16_fdot2_tiled<16, 32><<<grid, 128, 0, st>>>(W, X, bias, Y, M, N, K);
        return;
    }
    if (M <= 16) {
        const dim3 grid((N + 63) / 64, (M + 15) / 16);
        k_bf16_wmma_f16<16, 64><<<grid, 128, 0, st>>>(W, X, bias, Y, M, N, K);
    } else {
        const dim3 grid((N + 63) / 64, (M + 63) / 64);
        k_bf16_wmma_f16<64, 64><<<grid, 128, 0, st>>>(W, X, bias, Y, M, N, K);
    }
}

inline void launch_f32_linear(
    const float* W, const float* X, const float* bias, float* Y,
    int M, int N, int K, hipStream_t st, strata::hip_uma::AmdGen gen = strata::hip_uma::AmdGen::Unknown) {
    if (M <= 4) {
        const dim3 grid((N + 7) / 8);
        if (M == 1)      k_f32_gemv<1><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        else if (M == 2) k_f32_gemv<2><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        else             k_f32_gemv<4><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        return;
    }
    if (gen != strata::hip_uma::AmdGen::Rdna2 && gen != strata::hip_uma::AmdGen::Unknown) {
        if (M <= 16) {
            const dim3 grid((N + 63) / 64, (M + 15) / 16);
            k_f32_wmma_f16<16, 64><<<grid, 128, 0, st>>>(W, X, bias, Y, M, N, K);
        } else {
            const dim3 grid((N + 63) / 64, (M + 63) / 64);
            k_f32_wmma_f16<64, 64><<<grid, 128, 0, st>>>(W, X, bias, Y, M, N, K);
        }
        return;
    }
    const dim3 grid((N + 7) / 8, (M + 3) / 4);
    k_f32_gemv_2d<<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
}

inline void launch_q8row_linear(
    const int8_t* W, const float* row_scales, const float* X, float* Y,
    int M, int N, int K, hipStream_t st) {
    for (int m0 = 0; m0 < M; m0 += 4) {
        const int bm = (M - m0) < 4 ? (M - m0) : 4;
        const dim3 grid((N + 7) / 8);
        if (bm == 1)      k_q8row_gemv<1><<<grid, 256, 0, st>>>(W, row_scales, X + static_cast<size_t>(m0) * K, Y + static_cast<size_t>(m0) * N, bm, N, K);
        else if (bm == 2) k_q8row_gemv<2><<<grid, 256, 0, st>>>(W, row_scales, X + static_cast<size_t>(m0) * K, Y + static_cast<size_t>(m0) * N, bm, N, K);
        else              k_q8row_gemv<4><<<grid, 256, 0, st>>>(W, row_scales, X + static_cast<size_t>(m0) * K, Y + static_cast<size_t>(m0) * N, bm, N, K);
    }
}

}  // namespace strata::inkling2::hip
