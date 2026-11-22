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

__device__ __forceinline__ uint16_t ld_u16_unaligned(const uint8_t* p) {
    uint16_t v;
    __builtin_memcpy(&v, p, 2);
    return v;
}

__device__ __forceinline__ uint32_t ld_u32_unaligned(const uint8_t* p) {
    uint32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

// Fast 8-weight aligned S_K unpack to 4 packed FP16x2 words (unscaled integer values in [-128..127]):
// Switches ONCE per 8 weights and uses 32-bit loads (eliminates 8x runtime switch + 8x byte loads!)
__device__ __forceinline__ uint4 decode_sk_8_unscaled_f16(const uint8_t* pl, int code, int j0) {
    uint32_t w2[4];
    switch (code) {
        case CODE_S2_K: {
            const uint16_t raw = ld_u16_unaligned(pl + (j0 >> 2));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int a = static_cast<int>(((raw >> (4 * i)) & 3u) << 1) - 3;
                const int b = static_cast<int>(((raw >> (4 * i + 2)) & 3u) << 1) - 3;
                const hv2 h = {static_cast<half_t>(a), static_cast<half_t>(b)};
                w2[i] = __builtin_bit_cast(uint32_t, h);
            }
            break;
        }
        case CODE_S3_K: {
            const uint16_t lo = ld_u16_unaligned(pl + (j0 >> 2));
            const uint8_t  hi = pl[64 + (j0 >> 3)];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t q0 = ((lo >> (4 * i)) & 3u) | (((hi >> (2 * i)) & 1u) << 2);
                const uint32_t q1 = ((lo >> (4 * i + 2)) & 3u) | (((hi >> (2 * i + 1)) & 1u) << 2);
                const hv2 h = {static_cast<half_t>(static_cast<int>(q0 << 1) - 7),
                               static_cast<half_t>(static_cast<int>(q1 << 1) - 7)};
                w2[i] = __builtin_bit_cast(uint32_t, h);
            }
            break;
        }
        case CODE_S4_K: {
            const uint32_t raw = ld_u32_unaligned(pl + (j0 >> 1));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t by = (raw >> (8 * i)) & 0xFFu;
                const hv2 h = {static_cast<half_t>(static_cast<int>((by & 15u) << 1) - 15),
                               static_cast<half_t>(static_cast<int>((by >> 4) << 1) - 15)};
                w2[i] = __builtin_bit_cast(uint32_t, h);
            }
            break;
        }
        case CODE_S5_K: {
            const uint32_t lo = ld_u32_unaligned(pl + (j0 >> 1));
            const uint8_t  hi = pl[128 + (j0 >> 3)];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t by = (lo >> (8 * i)) & 0xFFu;
                const uint32_t q0 = (by & 15u) | (((hi >> (2 * i)) & 1u) << 4);
                const uint32_t q1 = (by >> 4)  | (((hi >> (2 * i + 1)) & 1u) << 4);
                const hv2 h = {static_cast<half_t>(static_cast<int>(q0 << 1) - 31),
                               static_cast<half_t>(static_cast<int>(q1 << 1) - 31)};
                w2[i] = __builtin_bit_cast(uint32_t, h);
            }
            break;
        }
        case CODE_S6_K: {
            const uint32_t lo = ld_u32_unaligned(pl + (j0 >> 1));
            const uint16_t hi = ld_u16_unaligned(pl + 128 + (j0 >> 2));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t by = (lo >> (8 * i)) & 0xFFu;
                const uint32_t q0 = (by & 15u) | (((hi >> (4 * i)) & 3u) << 4);
                const uint32_t q1 = (by >> 4)  | (((hi >> (4 * i + 2)) & 3u) << 4);
                const hv2 h = {static_cast<half_t>(static_cast<int>(q0 << 1) - 63),
                               static_cast<half_t>(static_cast<int>(q1 << 1) - 63)};
                w2[i] = __builtin_bit_cast(uint32_t, h);
            }
            break;
        }
        case CODE_S8_K: {
            const uint32_t r0 = ld_u32_unaligned(pl + j0);
            const uint32_t r1 = ld_u32_unaligned(pl + j0 + 4);
            const hv2 h0 = {static_cast<half_t>(static_cast<int8_t>(r0 & 0xFFu)),
                            static_cast<half_t>(static_cast<int8_t>((r0 >> 8) & 0xFFu))};
            const hv2 h1 = {static_cast<half_t>(static_cast<int8_t>((r0 >> 16) & 0xFFu)),
                            static_cast<half_t>(static_cast<int8_t>((r0 >> 24) & 0xFFu))};
            const hv2 h2 = {static_cast<half_t>(static_cast<int8_t>(r1 & 0xFFu)),
                            static_cast<half_t>(static_cast<int8_t>((r1 >> 8) & 0xFFu))};
            const hv2 h3 = {static_cast<half_t>(static_cast<int8_t>((r1 >> 16) & 0xFFu)),
                            static_cast<half_t>(static_cast<int8_t>((r1 >> 24) & 0xFFu))};
            w2[0] = __builtin_bit_cast(uint32_t, h0);
            w2[1] = __builtin_bit_cast(uint32_t, h1);
            w2[2] = __builtin_bit_cast(uint32_t, h2);
            w2[3] = __builtin_bit_cast(uint32_t, h3);
            break;
        }
        default: {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const hv2 h = {static_cast<half_t>(static_cast<int>(decode_sk_w_i8(pl, code, j0 + 2 * i))),
                               static_cast<half_t>(static_cast<int>(decode_sk_w_i8(pl, code, j0 + 2 * i + 1)))};
                w2[i] = __builtin_bit_cast(uint32_t, h);
            }
            break;
        }
    }
    return make_uint4(w2[0], w2[1], w2[2], w2[3]);
}

__device__ __forceinline__ void decode_sk_16_f16(
    const uint8_t* pl, int code, int j0, uint32_t eff2, uint4& o0, uint4& o1) {
    const uint4 u0 = decode_sk_8_unscaled_f16(pl, code, j0);
    const uint4 u1 = decode_sk_8_unscaled_f16(pl, code, j0 + 8);
    o0 = make_uint4(h2_mul(u0.x, eff2), h2_mul(u0.y, eff2), h2_mul(u0.z, eff2), h2_mul(u0.w, eff2));
    o1 = make_uint4(h2_mul(u1.x, eff2), h2_mul(u1.y, eff2), h2_mul(u1.z, eff2), h2_mul(u1.w, eff2));
}

// Fast 16-weight aligned S_K unpack directly to 16 exact signed int8 bytes (int4) for 2x-rate iu8 WMMA!
// ZERO floating-point conversions, ZERO h2_mul — pure integer bitshifts/loads!
__device__ __forceinline__ int4 decode_sk_16_i8(const uint8_t* pl, int code, int j0) {
    int out[4];
    switch (code) {
        case CODE_S2_K: {
            const uint32_t raw = ld_u32_unaligned(pl + (j0 >> 2));
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t b = (raw >> (w * 8)) & 0xFFu;
                out[w] = pack4_i8(
                    static_cast<int>((b & 3u) << 1) - 3,
                    static_cast<int>(((b >> 2) & 3u) << 1) - 3,
                    static_cast<int>(((b >> 4) & 3u) << 1) - 3,
                    static_cast<int>(((b >> 6) & 3u) << 1) - 3);
            }
            return make_int4(out[0], out[1], out[2], out[3]);
        }
        case CODE_S3_K: {
            const uint32_t lo = ld_u32_unaligned(pl + (j0 >> 2));
            const uint16_t hi = ld_u16_unaligned(pl + 64 + (j0 >> 3));
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t b = (lo >> (w * 8)) & 0xFFu;
                const uint32_t h = (hi >> (w * 4)) & 0xFu;
                out[w] = pack4_i8(
                    static_cast<int>(((b & 3u) | ((h & 1u) << 2)) << 1) - 7,
                    static_cast<int>((((b >> 2) & 3u) | (((h >> 1) & 1u) << 2)) << 1) - 7,
                    static_cast<int>((((b >> 4) & 3u) | (((h >> 2) & 1u) << 2)) << 1) - 7,
                    static_cast<int>((((b >> 6) & 3u) | (((h >> 3) & 1u) << 2)) << 1) - 7);
            }
            return make_int4(out[0], out[1], out[2], out[3]);
        }
        case CODE_S4_K: {
            const uint32_t r0 = ld_u32_unaligned(pl + (j0 >> 1));
            const uint32_t r1 = ld_u32_unaligned(pl + (j0 >> 1) + 4);
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t src = (w < 2) ? (r0 >> ((w & 1) * 16)) : (r1 >> ((w & 1) * 16));
                out[w] = pack4_i8(
                    static_cast<int>((src & 15u) << 1) - 15,
                    static_cast<int>(((src >> 4) & 15u) << 1) - 15,
                    static_cast<int>(((src >> 8) & 15u) << 1) - 15,
                    static_cast<int>(((src >> 12) & 15u) << 1) - 15);
            }
            return make_int4(out[0], out[1], out[2], out[3]);
        }
        case CODE_S5_K: {
            const uint32_t r0 = ld_u32_unaligned(pl + (j0 >> 1));
            const uint32_t r1 = ld_u32_unaligned(pl + (j0 >> 1) + 4);
            const uint16_t hi = ld_u16_unaligned(pl + 128 + (j0 >> 3));
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t src = (w < 2) ? (r0 >> ((w & 1) * 16)) : (r1 >> ((w & 1) * 16));
                const uint32_t h = (hi >> (w * 4)) & 0xFu;
                out[w] = pack4_i8(
                    static_cast<int>(((src & 15u) | ((h & 1u) << 4)) << 1) - 31,
                    static_cast<int>((((src >> 4) & 15u) | (((h >> 1) & 1u) << 4)) << 1) - 31,
                    static_cast<int>((((src >> 8) & 15u) | (((h >> 2) & 1u) << 4)) << 1) - 31,
                    static_cast<int>((((src >> 12) & 15u) | (((h >> 3) & 1u) << 4)) << 1) - 31);
            }
            return make_int4(out[0], out[1], out[2], out[3]);
        }
        case CODE_S6_K: {
            const uint32_t r0 = ld_u32_unaligned(pl + (j0 >> 1));
            const uint32_t r1 = ld_u32_unaligned(pl + (j0 >> 1) + 4);
            const uint32_t hi = ld_u32_unaligned(pl + 128 + (j0 >> 2));
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t src = (w < 2) ? (r0 >> ((w & 1) * 16)) : (r1 >> ((w & 1) * 16));
                const uint32_t h = (hi >> (w * 8)) & 0xFFu;
                out[w] = pack4_i8(
                    static_cast<int>(((src & 15u) | ((h & 3u) << 4)) << 1) - 63,
                    static_cast<int>((((src >> 4) & 15u) | (((h >> 2) & 3u) << 4)) << 1) - 63,
                    static_cast<int>((((src >> 8) & 15u) | (((h >> 4) & 3u) << 4)) << 1) - 63,
                    static_cast<int>((((src >> 12) & 15u) | (((h >> 6) & 3u) << 4)) << 1) - 63);
            }
            return make_int4(out[0], out[1], out[2], out[3]);
        }
        case CODE_S8_K: {
            int4 v;
            __builtin_memcpy(&v, pl + j0, 16);
            return v;
        }
        default: {
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                out[w] = pack4_i8(
                    decode_sk_w_i8(pl, code, j0 + w * 4 + 0),
                    decode_sk_w_i8(pl, code, j0 + w * 4 + 1),
                    decode_sk_w_i8(pl, code, j0 + w * 4 + 2),
                    decode_sk_w_i8(pl, code, j0 + w * 4 + 3));
            }
            return make_int4(out[0], out[1], out[2], out[3]);
        }
    }
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
    (void)hi;
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
    (void)hi;
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
    (void)hi;
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
    (void)hi;
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

// LDS-cached M=1 BF16 GEMV (K <= 4096, K % 8 == 0): loads X[K] ONCE into 16 KB LDS for all 8 waves
// with 1 single __syncthreads() + 2x K-unrolled 128-bit uint4 W loads -> 100% DRAM bus dedicated to W!
__global__ void __launch_bounds__(256) k_bf16_gemv1_lds(
    const uint16_t* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int N, int K) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;

    __shared__ alignas(16) float sX[4096];
    const int k4 = K >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(X);
    float4* sx4 = reinterpret_cast<float4*>(sX);
    for (int i = tid; i < k4; i += 256) sx4[i] = x4[i];
    __syncthreads();

    if (row >= N) return;
    const uint16_t* wr = W + static_cast<size_t>(row) * K;
    float acc0 = 0.0f, acc1 = 0.0f;
    int k = lane * 8;
    for (; k + 256 < K; k += 512) {
        const uint4 wa = *reinterpret_cast<const uint4*>(wr + k);
        const uint4 wb = *reinterpret_cast<const uint4*>(wr + k + 256);
        const float4 xa0 = *reinterpret_cast<const float4*>(sX + k);
        const float4 xa1 = *reinterpret_cast<const float4*>(sX + k + 4);
        const float4 xb0 = *reinterpret_cast<const float4*>(sX + k + 256);
        const float4 xb1 = *reinterpret_cast<const float4*>(sX + k + 260);
        acc0 = fmaf(bf16_to_f32(wa.x & 0xFFFFu), xa0.x, fmaf(bf16_to_f32(wa.x >> 16), xa0.y,
               fmaf(bf16_to_f32(wa.y & 0xFFFFu), xa0.z, fmaf(bf16_to_f32(wa.y >> 16), xa0.w, acc0))));
        acc0 = fmaf(bf16_to_f32(wa.z & 0xFFFFu), xa1.x, fmaf(bf16_to_f32(wa.z >> 16), xa1.y,
               fmaf(bf16_to_f32(wa.w & 0xFFFFu), xa1.z, fmaf(bf16_to_f32(wa.w >> 16), xa1.w, acc0))));
        acc1 = fmaf(bf16_to_f32(wb.x & 0xFFFFu), xb0.x, fmaf(bf16_to_f32(wb.x >> 16), xb0.y,
               fmaf(bf16_to_f32(wb.y & 0xFFFFu), xb0.z, fmaf(bf16_to_f32(wb.y >> 16), xb0.w, acc1))));
        acc1 = fmaf(bf16_to_f32(wb.z & 0xFFFFu), xb1.x, fmaf(bf16_to_f32(wb.z >> 16), xb1.y,
               fmaf(bf16_to_f32(wb.w & 0xFFFFu), xb1.z, fmaf(bf16_to_f32(wb.w >> 16), xb1.w, acc1))));
    }
    if (k < K) {
        const uint4 wa = *reinterpret_cast<const uint4*>(wr + k);
        const float4 xa0 = *reinterpret_cast<const float4*>(sX + k);
        const float4 xa1 = *reinterpret_cast<const float4*>(sX + k + 4);
        acc0 = fmaf(bf16_to_f32(wa.x & 0xFFFFu), xa0.x, fmaf(bf16_to_f32(wa.x >> 16), xa0.y,
               fmaf(bf16_to_f32(wa.y & 0xFFFFu), xa0.z, fmaf(bf16_to_f32(wa.y >> 16), xa0.w, acc0))));
        acc0 = fmaf(bf16_to_f32(wa.z & 0xFFFFu), xa1.x, fmaf(bf16_to_f32(wa.z >> 16), xa1.y,
               fmaf(bf16_to_f32(wa.w & 0xFFFFu), xa1.z, fmaf(bf16_to_f32(wa.w >> 16), xa1.w, acc0))));
    }
    const float sum = wave32_sum(acc0 + acc1) + (bias ? bias[row] : 0.0f);
    if (lane == 0) Y[row] = sum;
}

// Wave-6C/6F Fused M=1 QKVRG + Attn-Sink BF16/F32 GEMV + Direct Scatter into Q, R, Gate, (optional K, V), Sinks:
// Deletes 48 k_split_qkvrg + 48 attn_sink_proj launches + 48 cat_out DRAM round-trips per token!
// Grid: ((tot_n + n_sink + 7) / 8), Block: 256
__global__ void __launch_bounds__(256) k_bf16_gemv1_qkvrg(
    const uint16_t* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Q,
    float* __restrict__ R,
    float* __restrict__ G,
    float* __restrict__ K_out,
    float* __restrict__ V_out,
    const float* __restrict__ W_sink,   // nullptr or [n_sink, K] FP32
    float* __restrict__ Sinks_out,      // nullptr or [n_sink] FP32
    int q_dim, int r_dim, int kv_dim, int tot_n, int n_sink, int K) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;

    __shared__ alignas(16) float sX[4096];
    const int k4 = K >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(X);
    float4* sx4 = reinterpret_cast<float4*>(sX);
    for (int i = tid; i < k4; i += 256) sx4[i] = x4[i];
    __syncthreads();

    if (row >= tot_n + n_sink) return;
    if (row >= tot_n) {
        const int sr = row - tot_n;
        const float4* w4 = reinterpret_cast<const float4*>(W_sink + static_cast<size_t>(sr) * K);
        float acc = 0.0f;
        for (int i = lane; i < k4; i += 32) {
            const float4 wv = w4[i];
            const float4 xv = sx4[i];
            acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
        }
        const float sum = wave32_sum(acc);
        if (lane == 0) Sinks_out[sr] = sum;
        return;
    }

    const uint16_t* wr = W + static_cast<size_t>(row) * K;
    float acc0 = 0.0f, acc1 = 0.0f;
    int k = lane * 8;
    for (; k + 256 < K; k += 512) {
        const uint4 wa = *reinterpret_cast<const uint4*>(wr + k);
        const uint4 wb = *reinterpret_cast<const uint4*>(wr + k + 256);
        const float4 xa0 = *reinterpret_cast<const float4*>(sX + k);
        const float4 xa1 = *reinterpret_cast<const float4*>(sX + k + 4);
        const float4 xb0 = *reinterpret_cast<const float4*>(sX + k + 256);
        const float4 xb1 = *reinterpret_cast<const float4*>(sX + k + 260);
        acc0 = fmaf(bf16_to_f32(wa.x & 0xFFFFu), xa0.x, fmaf(bf16_to_f32(wa.x >> 16), xa0.y,
               fmaf(bf16_to_f32(wa.y & 0xFFFFu), xa0.z, fmaf(bf16_to_f32(wa.y >> 16), xa0.w, acc0))));
        acc0 = fmaf(bf16_to_f32(wa.z & 0xFFFFu), xa1.x, fmaf(bf16_to_f32(wa.z >> 16), xa1.y,
               fmaf(bf16_to_f32(wa.w & 0xFFFFu), xa1.z, fmaf(bf16_to_f32(wa.w >> 16), xa1.w, acc0))));
        acc1 = fmaf(bf16_to_f32(wb.x & 0xFFFFu), xb0.x, fmaf(bf16_to_f32(wb.x >> 16), xb0.y,
               fmaf(bf16_to_f32(wb.y & 0xFFFFu), xb0.z, fmaf(bf16_to_f32(wb.y >> 16), xb0.w, acc1))));
        acc1 = fmaf(bf16_to_f32(wb.z & 0xFFFFu), xb1.x, fmaf(bf16_to_f32(wb.z >> 16), xb1.y,
               fmaf(bf16_to_f32(wb.w & 0xFFFFu), xb1.z, fmaf(bf16_to_f32(wb.w >> 16), xb1.w, acc1))));
    }
    if (k < K) {
        const uint4 wa = *reinterpret_cast<const uint4*>(wr + k);
        const float4 xa0 = *reinterpret_cast<const float4*>(sX + k);
        const float4 xa1 = *reinterpret_cast<const float4*>(sX + k + 4);
        acc0 = fmaf(bf16_to_f32(wa.x & 0xFFFFu), xa0.x, fmaf(bf16_to_f32(wa.x >> 16), xa0.y,
               fmaf(bf16_to_f32(wa.y & 0xFFFFu), xa0.z, fmaf(bf16_to_f32(wa.y >> 16), xa0.w, acc0))));
        acc0 = fmaf(bf16_to_f32(wa.z & 0xFFFFu), xa1.x, fmaf(bf16_to_f32(wa.z >> 16), xa1.y,
               fmaf(bf16_to_f32(wa.w & 0xFFFFu), xa1.z, fmaf(bf16_to_f32(wa.w >> 16), xa1.w, acc0))));
    }
    const float sum = wave32_sum(acc0 + acc1);
    if (lane == 0) {
        if (row < q_dim) Q[row] = sum;
        else if (row < q_dim + r_dim) R[row - q_dim] = sum;
        else if (row < 2 * q_dim + r_dim) G[row - (q_dim + r_dim)] = sum;
        else if (K_out && row < 2 * q_dim + r_dim + kv_dim) K_out[row - (2 * q_dim + r_dim)] = sum;
        else if (V_out) V_out[row - (2 * q_dim + r_dim + kv_dim)] = sum;
    }
}

// Wave-6F Fused M=1 wo_ud BF16 GEMV + Post-Attn ShortConv4 + Residual Add / Block-Boundary Snapshot!
// Deletes 48 sconv launches + 24 boundary D2D copies + 48 oproj/asc DRAM round-trips per token!
// Grid: ((D + 7) / 8), Block: 256
__global__ void __launch_bounds__(256) k_bf16_gemv1_wo_sconv(
    const uint16_t* __restrict__ W,        // [D, K] BF16 (wo_ud)
    const float* __restrict__ X,           // [K] FP32 (attn_out)
    const float* __restrict__ static_w,    // [D, 4] FP32
    const int32_t* __restrict__ slot_ids,  // [1]
    uint16_t* __restrict__ hist_pool,      // [max_slots, 3, D] BF16
    float* __restrict__ cur,               // [D] FP32 (updated in-place!)
    float* __restrict__ blk_save,          // nullptr (non-boundary: cur += y) or [D] (boundary: blk_save=cur, cur=y)
    int D, int K) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;

    __shared__ alignas(16) float sX[4096];
    const int k4 = K >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(X);
    float4* sx4 = reinterpret_cast<float4*>(sX);
    for (int i = tid; i < k4; i += 256) sx4[i] = x4[i];
    __syncthreads();

    if (row >= D) return;
    const uint16_t* wr = W + static_cast<size_t>(row) * K;
    float acc0 = 0.0f, acc1 = 0.0f;
    int k = lane * 8;
    for (; k + 256 < K; k += 512) {
        const uint4 wa = *reinterpret_cast<const uint4*>(wr + k);
        const uint4 wb = *reinterpret_cast<const uint4*>(wr + k + 256);
        const float4 xa0 = *reinterpret_cast<const float4*>(sX + k);
        const float4 xa1 = *reinterpret_cast<const float4*>(sX + k + 4);
        const float4 xb0 = *reinterpret_cast<const float4*>(sX + k + 256);
        const float4 xb1 = *reinterpret_cast<const float4*>(sX + k + 260);
        acc0 = fmaf(bf16_to_f32(wa.x & 0xFFFFu), xa0.x, fmaf(bf16_to_f32(wa.x >> 16), xa0.y,
               fmaf(bf16_to_f32(wa.y & 0xFFFFu), xa0.z, fmaf(bf16_to_f32(wa.y >> 16), xa0.w, acc0))));
        acc0 = fmaf(bf16_to_f32(wa.z & 0xFFFFu), xa1.x, fmaf(bf16_to_f32(wa.z >> 16), xa1.y,
               fmaf(bf16_to_f32(wa.w & 0xFFFFu), xa1.z, fmaf(bf16_to_f32(wa.w >> 16), xa1.w, acc0))));
        acc1 = fmaf(bf16_to_f32(wb.x & 0xFFFFu), xb0.x, fmaf(bf16_to_f32(wb.x >> 16), xb0.y,
               fmaf(bf16_to_f32(wb.y & 0xFFFFu), xb0.z, fmaf(bf16_to_f32(wb.y >> 16), xb0.w, acc1))));
        acc1 = fmaf(bf16_to_f32(wb.z & 0xFFFFu), xb1.x, fmaf(bf16_to_f32(wb.z >> 16), xb1.y,
               fmaf(bf16_to_f32(wb.w & 0xFFFFu), xb1.z, fmaf(bf16_to_f32(wb.w >> 16), xb1.w, acc1))));
    }
    if (k < K) {
        const uint4 wa = *reinterpret_cast<const uint4*>(wr + k);
        const float4 xa0 = *reinterpret_cast<const float4*>(sX + k);
        const float4 xa1 = *reinterpret_cast<const float4*>(sX + k + 4);
        acc0 = fmaf(bf16_to_f32(wa.x & 0xFFFFu), xa0.x, fmaf(bf16_to_f32(wa.x >> 16), xa0.y,
               fmaf(bf16_to_f32(wa.y & 0xFFFFu), xa0.z, fmaf(bf16_to_f32(wa.y >> 16), xa0.w, acc0))));
        acc0 = fmaf(bf16_to_f32(wa.z & 0xFFFFu), xa1.x, fmaf(bf16_to_f32(wa.z >> 16), xa1.y,
               fmaf(bf16_to_f32(wa.w & 0xFFFFu), xa1.z, fmaf(bf16_to_f32(wa.w >> 16), xa1.w, acc0))));
    }
    const float x0 = wave32_sum(acc0 + acc1);
    if (lane == 0) {
        const int slot = slot_ids[0];
        uint16_t* h_slot = hist_pool + static_cast<size_t>(slot) * 3 * D;
        const float h0 = bf16_to_f32(h_slot[0 * D + row]);
        const float h1 = bf16_to_f32(h_slot[1 * D + row]);
        const float h2 = bf16_to_f32(h_slot[2 * D + row]);
        const float4 sw = *reinterpret_cast<const float4*>(static_w + static_cast<size_t>(row) * 4);
        const float y = (x0 + sw.x * h0 + sw.y * h1 + sw.z * h2 + sw.w * x0) * 0.44721359549995793f;
        h_slot[0 * D + row] = h_slot[1 * D + row];
        h_slot[1 * D + row] = h_slot[2 * D + row];
        h_slot[2 * D + row] = f32_to_bf16(x0);
        if (blk_save != nullptr) {
            blk_save[row] = cur[row];
            cur[row] = y;
        } else {
            cur[row] += y;
        }
    }
}

// Wave-6C High-Occupancy Small-N FP32 GEMV (1 block of 256 threads per output row -> 128 blocks for router, 24 for sink!)
// Saturates ALL 40-64 CUs on R9700 & Strix Halo (8x higher occupancy than 16-block k_f32_gemv!)
// Grid: (N, M), Block: 256
__global__ void __launch_bounds__(256) k_f32_gemv_hiocc(
    const float* __restrict__ W,    // [N, K]
    const float* __restrict__ X,    // [M, K]
    const float* __restrict__ bias, // nullptr or [N]
    float* __restrict__ Y,          // [M, N]
    int M, int N, int K) {
    const int row = blockIdx.x;
    const int m = blockIdx.y;
    if (row >= N || m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const float* wr = W + static_cast<size_t>(row) * K;
    const float* xr = X + static_cast<size_t>(m) * K;
    float acc = 0.0f;
    const int k4 = K >> 2;
    const float4* w4 = reinterpret_cast<const float4*>(wr);
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    for (int i = tid; i < k4; i += 256) {
        const float4 wv = w4[i];
        const float4 xv = x4[i];
        acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
    }
    acc = wave32_sum(acc);
    __shared__ float s_w[8];
    if (lane == 0) s_w[wave] = acc;
    __syncthreads();
    if (tid == 0) {
        float tot = s_w[0] + s_w[1] + s_w[2] + s_w[3] + s_w[4] + s_w[5] + s_w[6] + s_w[7];
        if (bias) tot += bias[row];
        Y[static_cast<size_t>(m) * N + row] = tot;
    }
}

// BF16 linear Y[M, N] = X[M, K] @ W[N, K]^T + optional bias[N] (M=1..8 Wave32 GEMV, 128-bit uint4 W loads!)
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

    const int k8 = K & ~7;
    for (int k = lane * 8; k < k8; k += 256) {
        const uint4 w8 = *reinterpret_cast<const uint4*>(wr + k);
        const float f0 = bf16_to_f32(static_cast<uint16_t>(w8.x & 0xFFFFu));
        const float f1 = bf16_to_f32(static_cast<uint16_t>(w8.x >> 16));
        const float f2 = bf16_to_f32(static_cast<uint16_t>(w8.y & 0xFFFFu));
        const float f3 = bf16_to_f32(static_cast<uint16_t>(w8.y >> 16));
        const float f4 = bf16_to_f32(static_cast<uint16_t>(w8.z & 0xFFFFu));
        const float f5 = bf16_to_f32(static_cast<uint16_t>(w8.z >> 16));
        const float f6 = bf16_to_f32(static_cast<uint16_t>(w8.w & 0xFFFFu));
        const float f7 = bf16_to_f32(static_cast<uint16_t>(w8.w >> 16));
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const float* xp = X + static_cast<size_t>(m) * K + k;
                const float4 x0 = *reinterpret_cast<const float4*>(xp);
                const float4 x1 = *reinterpret_cast<const float4*>(xp + 4);
                acc[m] = fmaf(f0, x0.x, fmaf(f1, x0.y, fmaf(f2, x0.z, fmaf(f3, x0.w, acc[m]))));
                acc[m] = fmaf(f4, x1.x, fmaf(f5, x1.y, fmaf(f6, x1.z, fmaf(f7, x1.w, acc[m]))));
            }
        }
    }
    for (int k = k8 + lane; k < K; k += 32) {
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

// LDS-cached packed fdot2 BF16 GEMV for M=2..8 (e.g. B=8 DSpark verify & c=4/8 parallel decode!):
// 4 KB LDS sX[MT][256] shared across all 8 waves/block + 128-bit uint4 W loads + native v_dot2_f32_f16!
// High occupancy (384 blocks for N=3072, 288 for N=2304, 128 for N=1024) across ALL 11 AMD GPUs incl. RDNA2!
template <int MT>
__global__ void __launch_bounds__(256) k_bf16_gemv_lds_fdot2(
    const uint16_t* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;
    const uint16_t* wr = (row < N) ? (W + static_cast<size_t>(row) * K) : nullptr;

    __shared__ alignas(16) half_t sX[MT * 256];
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    const int k256 = K & ~255;
    for (int k0 = 0; k0 < k256; k0 += 256) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) sX[m * 256 + tid] = static_cast<half_t>(X[static_cast<size_t>(m) * K + k0 + tid]);
        }
        __syncthreads();
        if (wr) {
            const uint4 w8 = *reinterpret_cast<const uint4*>(wr + k0 + lane * 8);
            const hv2 h0 = {static_cast<half_t>(bf16_to_f32(w8.x & 0xFFFFu)), static_cast<half_t>(bf16_to_f32(w8.x >> 16))};
            const hv2 h1 = {static_cast<half_t>(bf16_to_f32(w8.y & 0xFFFFu)), static_cast<half_t>(bf16_to_f32(w8.y >> 16))};
            const hv2 h2 = {static_cast<half_t>(bf16_to_f32(w8.z & 0xFFFFu)), static_cast<half_t>(bf16_to_f32(w8.z >> 16))};
            const hv2 h3 = {static_cast<half_t>(bf16_to_f32(w8.w & 0xFFFFu)), static_cast<half_t>(bf16_to_f32(w8.w >> 16))};
            const uint32_t p0 = __builtin_bit_cast(uint32_t, h0);
            const uint32_t p1 = __builtin_bit_cast(uint32_t, h1);
            const uint32_t p2 = __builtin_bit_cast(uint32_t, h2);
            const uint32_t p3 = __builtin_bit_cast(uint32_t, h3);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < M) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(&sX[m * 256 + lane * 8]);
                    float d = acc[m];
                    d = fdot2_f32(p0, xv.x, d);
                    d = fdot2_f32(p1, xv.y, d);
                    d = fdot2_f32(p2, xv.z, d);
                    d = fdot2_f32(p3, xv.w, d);
                    acc[m] = d;
                }
            }
        }
        __syncthreads();
    }
    if (wr) {
        for (int k = k256 + lane; k < K; k += 32) {
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
}

// Skinny Native Hardware BF16 WMMA (BM=16, BN=32 or 64, BK=64) for M=9..16 or small-N controllers (N=64)!
// BN=64 uses 128 threads (4 waves), BN=32 uses 64 threads (2 waves) -> saturates 32-40 CUs even for N=1024!
template <int BN>
__global__ void __launch_bounds__((BN == 64) ? 128 : 64) k_bf16_wmma_bf16_16xN(
    const uint16_t* __restrict__ W,   // [N, K] BF16
    const float* __restrict__ X,      // [M, K] FP32
    const float* __restrict__ bias,   // nullptr or [N] FP32
    float* __restrict__ Y,            // [M, N] FP32
    int M, int N, int K) {
#if INK_HAS_WMMA
    constexpr int BM = 16, BK = 64, LD = BK + 8;
    constexpr int NTHREADS = (BN == 64) ? 128 : 64;
    __shared__ alignas(16) uint16_t sX[BM * LD];
    __shared__ alignas(16) uint16_t sW[BN * LD];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;
    const int tm = (M - m_blk < BM) ? (M - m_blk) : BM;

    fv8 acc = fv8_zero();

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * (BK / 4); idx += NTHREADS) {
            const int r = idx >> 4;
            const int c4 = idx & 15;
            const int gm = m_blk + r;
            const int gk = k0 + c4 * 4;
            uint16_t* dst = &sX[r * LD + c4 * 4];
            if (r < tm && gk + 3 < K) {
                const float4 xv = *reinterpret_cast<const float4*>(X + static_cast<size_t>(gm) * K + gk);
                const uint32_t b0 = static_cast<uint32_t>(f32_to_bf16(xv.x)) | (static_cast<uint32_t>(f32_to_bf16(xv.y)) << 16);
                const uint32_t b1 = static_cast<uint32_t>(f32_to_bf16(xv.z)) | (static_cast<uint32_t>(f32_to_bf16(xv.w)) << 16);
                *reinterpret_cast<uint2*>(dst) = make_uint2(b0, b1);
            } else if (r < tm) {
                for (int j = 0; j < 4; ++j)
                    dst[j] = (gk + j < K) ? f32_to_bf16(X[static_cast<size_t>(gm) * K + gk + j]) : 0u;
            } else {
                *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
            }
        }
        for (int idx = tid; idx < BN * (BK / 8); idx += NTHREADS) {
            const int r = idx >> 3;
            const int c8 = idx & 7;
            const int gn = n_blk + r;
            const int gk = k0 + c8 * 8;
            uint16_t* dst = &sW[r * LD + c8 * 8];
            if (gn < N && gk + 7 < K) {
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(W + static_cast<size_t>(gn) * K + gk);
            } else {
                for (int j = 0; j < 8; ++j)
                    dst[j] = (gn < N && gk + j < K) ? W[static_cast<size_t>(gn) * K + gk + j] : 0u;
            }
        }
        sync_lds();

#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma_bf_t a0 = wmma_ld_bf(&sX[l16 * LD + kk * 16], hi);
            wmma_bf_t b0 = wmma_ld_bf(&sW[(wave * 16 + l16) * LD + kk * 16], hi);
            acc = wmma_bf16(a0, b0, acc);
        }
        sync_lds();
    }

    const int gn = n_blk + wave * 16 + wmma_acc_n(0, lane);
    if (gn < N) {
        const float bv = bias ? bias[gn] : 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gm = m_blk + wmma_acc_m(r, lane);
            if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[r] + bv;
        }
    }
#else
    (void)W; (void)X; (void)bias; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// Multi-Gen Double-Buffered L2/MALL-Swizzled BM=128, BN=128, BK=32 BF16 WMMA GEMM (40 KB LDS, 1 sync/iter!)
// Overlaps global uint4 loads for k0+BK with 16 hardware WMMAs of k0, streams b fragment (saves 24 VGPRs on gfx11!),
// and uses GROUP_N=4 M-first swizzle so W[512, K] (~2.3 MB) + X_bf16 stay 100% resident in 32 MB L2/MALL!
__global__ void __launch_bounds__(256) k_bf16_wmma_bf16_128x128_swiz(
    const uint16_t* __restrict__ W,      // [N, K] BF16
    const uint16_t* __restrict__ X_bf16, // [M, K] BF16
    const float* __restrict__ bias,      // nullptr or [N] FP32
    float* __restrict__ Y,               // [M, N] FP32
    int M, int N, int K) {
#if INK_HAS_WMMA
    constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 8;  // 40 uint16 = 80 B/row
    __shared__ alignas(16) uint16_t sX[2][BM * LD];          // 20,480 B
    __shared__ alignas(16) uint16_t sW[2][BN * LD];          // 20,480 B -> 40,960 B (double buffered!)

    const int num_n = (N + BN - 1) / BN;
    const int num_m = (M + BM - 1) / BM;
    const int bid = blockIdx.x;
    if (bid >= num_m * num_n) return;

    constexpr int GROUP_N = 4;
    const int grp_sz = GROUP_N * num_m;
    const int grp = bid / grp_sz;
    const int rem = bid - grp * grp_sz;
    const int gn0 = grp * GROUP_N;
    const int cur_gn = (num_n - gn0 < GROUP_N) ? (num_n - gn0) : GROUP_N;
    const int tile_m = rem / cur_gn;
    const int tile_n = gn0 + (rem - tile_m * cur_gn);

    const int m_blk = tile_m * BM;
    const int n_blk = tile_n * BN;
    const int tm = (M - m_blk < BM) ? (M - m_blk) : BM;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 1;
    const int wn = wave & 1;

    const int r0 = tid >> 2, c8_0 = tid & 3;
    const int r1 = (tid + 256) >> 2, c8_1 = (tid + 256) & 3;
    const int gm0 = m_blk + r0, gm1 = m_blk + r1;
    const int gn_0 = n_blk + r0, gn_1 = n_blk + r1;

    auto ld_x = [&](int gm, int gk) -> uint4 {
        if (gm < M && gk + 7 < K) return *reinterpret_cast<const uint4*>(X_bf16 + static_cast<size_t>(gm) * K + gk);
        uint16_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (gm < M) for (int j = 0; j < 8; ++j) if (gk + j < K) t[j] = X_bf16[static_cast<size_t>(gm) * K + gk + j];
        return *reinterpret_cast<const uint4*>(t);
    };
    auto ld_w = [&](int gn, int gk) -> uint4 {
        if (gn < N && gk + 7 < K) return *reinterpret_cast<const uint4*>(W + static_cast<size_t>(gn) * K + gk);
        uint16_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (gn < N) for (int j = 0; j < 8; ++j) if (gk + j < K) t[j] = W[static_cast<size_t>(gn) * K + gk + j];
        return *reinterpret_cast<const uint4*>(t);
    };

    // Prologue: load tile 0 into buf 0
    *reinterpret_cast<uint4*>(&sX[0][r0 * LD + c8_0 * 8]) = ld_x(gm0, c8_0 * 8);
    *reinterpret_cast<uint4*>(&sX[0][r1 * LD + c8_1 * 8]) = ld_x(gm1, c8_1 * 8);
    *reinterpret_cast<uint4*>(&sW[0][r0 * LD + c8_0 * 8]) = ld_w(gn_0, c8_0 * 8);
    *reinterpret_cast<uint4*>(&sW[0][r1 * LD + c8_1 * 8]) = ld_w(gn_1, c8_1 * 8);
    sync_lds();

    fv8 acc[2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int kn = k0 + BK;
        uint4 rx0, rx1, rw0, rw1;
        if (kn < K) {
            rx0 = ld_x(gm0, kn + c8_0 * 8);
            rx1 = ld_x(gm1, kn + c8_1 * 8);
            rw0 = ld_w(gn_0, kn + c8_0 * 8);
            rw1 = ld_w(gn_1, kn + c8_1 * 8);
        }
        if (wm * 32 < tm) {
            const bool do_m1 = (wm * 32 + 16 < tm);
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                wmma_bf_t a0 = wmma_ld_bf(&sX[buf][(wm * 32 + l16) * LD + kk * 16], hi);
                wmma_bf_t a1 = do_m1 ? wmma_ld_bf(&sX[buf][(wm * 32 + 16 + l16) * LD + kk * 16], hi) : a0;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    wmma_bf_t b = wmma_ld_bf(&sW[buf][(wn * 64 + ni * 16 + l16) * LD + kk * 16], hi);
                    acc[0][ni] = wmma_bf16(a0, b, acc[0][ni]);
                    if (do_m1) acc[1][ni] = wmma_bf16(a1, b, acc[1][ni]);
                }
            }
        }
        if (kn < K) {
            const int nb = buf ^ 1;
            *reinterpret_cast<uint4*>(&sX[nb][r0 * LD + c8_0 * 8]) = rx0;
            *reinterpret_cast<uint4*>(&sX[nb][r1 * LD + c8_1 * 8]) = rx1;
            *reinterpret_cast<uint4*>(&sW[nb][r0 * LD + c8_0 * 8]) = rw0;
            *reinterpret_cast<uint4*>(&sW[nb][r1 * LD + c8_1 * 8]) = rw1;
            sync_lds();
            buf = nb;
        }
    }

    if (wm * 32 < tm) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int gn = n_blk + wn * 64 + ni * 16 + wmma_acc_n(0, lane);
                if (gn < N) {
                    const float bv = bias ? bias[gn] : 0.0f;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int gm = m_blk + wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                        if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[mi][ni][r] + bv;
                    }
                }
            }
        }
    }
#else
    (void)W; (void)X_bf16; (void)bias; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// Multi-Gen Native Hardware BF16 WMMA GEMM (RDNA4 bf8 / RDNA3/3.5 bf16v) BM=64, BN=128, 256 threads (8 waves)!
// Loads BF16 W directly via 128-bit uint4 into LDS with ZERO BF16->FP32->FP16 conversion!
__global__ void __launch_bounds__(256) k_bf16_wmma_bf16_64x128(
    const uint16_t* __restrict__ W,   // [N, K] BF16
    const float* __restrict__ X,      // [M, K] FP32
    const float* __restrict__ bias,   // nullptr or [N] FP32
    float* __restrict__ Y,            // [M, N] FP32
    int M, int N, int K) {
#if INK_HAS_WMMA
    constexpr int BM = 64, BN = 128, BK = 64, LD = BK + 8;
    __shared__ alignas(16) uint16_t sX[BM * LD];
    __shared__ alignas(16) uint16_t sW[BN * LD];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 2;   // 0..1 (32 M rows)
    const int wn = wave & 3;    // 0..3 (32 N cols)
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;
    const int tm = (M - m_blk < BM) ? (M - m_blk) : BM;

    fv8 acc[2][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) acc[mi][ni] = fv8_zero();

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load X [64, 64] FP32 -> BF16 (4 iterations per thread!)
        for (int idx = tid; idx < BM * (BK / 4); idx += 256) {
            const int r = idx >> 4;
            const int c4 = idx & 15;
            const int gm = m_blk + r;
            const int gk = k0 + c4 * 4;
            uint16_t* dst = &sX[r * LD + c4 * 4];
            if (gm < M && gk + 3 < K) {
                const float4 xv = *reinterpret_cast<const float4*>(X + static_cast<size_t>(gm) * K + gk);
                const uint32_t b0 = static_cast<uint32_t>(f32_to_bf16(xv.x)) | (static_cast<uint32_t>(f32_to_bf16(xv.y)) << 16);
                const uint32_t b1 = static_cast<uint32_t>(f32_to_bf16(xv.z)) | (static_cast<uint32_t>(f32_to_bf16(xv.w)) << 16);
                *reinterpret_cast<uint2*>(dst) = make_uint2(b0, b1);
            } else {
                for (int j = 0; j < 4; ++j)
                    dst[j] = (gm < M && gk + j < K) ? f32_to_bf16(X[static_cast<size_t>(gm) * K + gk + j]) : 0u;
            }
        }
        // Load W [128, 64] BF16 directly via 128-bit uint4 -> ZERO conversion! (4 iterations per thread!)
        for (int idx = tid; idx < BN * (BK / 8); idx += 256) {
            const int r = idx >> 3;
            const int c8 = idx & 7;
            const int gn = n_blk + r;
            const int gk = k0 + c8 * 8;
            uint16_t* dst = &sW[r * LD + c8 * 8];
            if (gn < N && gk + 7 < K) {
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(W + static_cast<size_t>(gn) * K + gk);
            } else {
                for (int j = 0; j < 8; ++j)
                    dst[j] = (gn < N && gk + j < K) ? W[static_cast<size_t>(gn) * K + gk + j] : 0u;
            }
        }
        sync_lds();

        if (wm * 32 < tm) {
            const bool do_m1 = (wm * 32 + 16 < tm);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma_bf_t b0 = wmma_ld_bf(&sW[(wn * 32 + l16) * LD + kk * 16], hi);
                wmma_bf_t b1 = wmma_ld_bf(&sW[(wn * 32 + 16 + l16) * LD + kk * 16], hi);
                wmma_bf_t a0 = wmma_ld_bf(&sX[(wm * 32 + l16) * LD + kk * 16], hi);
                acc[0][0] = wmma_bf16(a0, b0, acc[0][0]);
                acc[0][1] = wmma_bf16(a0, b1, acc[0][1]);
                if (do_m1) {
                    wmma_bf_t a1 = wmma_ld_bf(&sX[(wm * 32 + 16 + l16) * LD + kk * 16], hi);
                    acc[1][0] = wmma_bf16(a1, b0, acc[1][0]);
                    acc[1][1] = wmma_bf16(a1, b1, acc[1][1]);
                }
            }
        }
        sync_lds();
    }

    if (wm * 32 < tm) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int gn = n_blk + wn * 32 + ni * 16 + wmma_acc_n(0, lane);
                if (gn < N) {
                    const float bv = bias ? bias[gn] : 0.0f;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int gm = m_blk + wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                        if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[mi][ni][r] + bv;
                    }
                }
            }
        }
    }
#else
    (void)W; (void)X; (void)bias; (void)Y; (void)M; (void)N; (void)K;
#endif
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
    (void)hi;
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
    (void)hi;
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
// Wave-6: LDS-cached sX[D] (4.6 KB shared by all 8 waves) + 1 KB SF4 LUT + zero-switch S2_K/S3_K hot loops!
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

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int j = blockIdx.x * 4 + (wave >> 1);  // 4 inter indices per block; wave 2k=gate(2j), 2k+1=up(2j+1)
    const int row = (j << 1) | (wave & 1);
    const half_t* xrow = X_f16 + static_cast<size_t>(b) * D;
    float acc = 0.0f;

    __shared__ alignas(16) half_t sX[2304];  // 4,608 B
    __shared__ half_t s_e4m3[256];           //   512 B
    __shared__ uint32_t s_sf4_pair[256];     // 1,024 B
    __shared__ float s_sf4_sc[256];          // 1,024 B
    const bool use_sx = (D <= 2304 && (D & 7) == 0);
    if (use_sx) {
        const int d8 = D >> 3;
        const uint4* x8 = reinterpret_cast<const uint4*>(xrow);
        uint4* sx8 = reinterpret_cast<uint4*>(sX);
        for (int i = tid; i < d8; i += 256) sx8[i] = x8[i];
    }
    if (desc.gu_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    } else if (desc.gu_code == CODE_SF4) {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc[tid] = e4m3fn_to_f32(static_cast<uint8_t>(tid));
    }
    __syncthreads();
    if (j >= inter) return;
    const half_t* xsrc = use_sx ? sX : xrow;

    if (desc.gu_code == CODE_SF4) {
        const float* hdr = reinterpret_cast<const float*>(desc.body);
        const float outer = hdr[0] / hdr[1];
        const uint8_t* packed = desc.body + 16;
        const uint8_t* scales = packed + static_cast<size_t>(2 * inter) * (D >> 1);
        const uint8_t* pr = packed + static_cast<size_t>(row) * (D >> 1);
        const uint8_t* sr = scales + static_cast<size_t>(row) * (D >> 4);
        const int nblk = D >> 4;
        for (int blk = lane; blk < nblk; blk += 32) {
            const float sc = s_sf4_sc[sr[blk]] * outer;
            const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
            const uint4 x0 = *reinterpret_cast<const uint4*>(xsrc + (blk << 4));
            const uint4 x1 = *reinterpret_cast<const uint4*>(xsrc + (blk << 4) + 8);
            float d = fdot2_f32(s_sf4_pair[raw.x & 0xFFu], x0.x, 0.0f);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 8) & 0xFFu], x0.y, d);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 16) & 0xFFu], x0.z, d);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 24) & 0xFFu], x0.w, d);
            d = fdot2_f32(s_sf4_pair[raw.y & 0xFFu], x1.x, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 8) & 0xFFu], x1.y, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 16) & 0xFFu], x1.z, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 24) & 0xFFu], x1.w, d);
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
            const uint4* xp = reinterpret_cast<const uint4*>(xsrc + (blk << 5));
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
    } else if (desc.gu_code == CODE_S2_K) {
        const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
        const int nsb = D >> 8;
        const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * 70;
        const int j0 = lane * 8;
        const int sub = j0 >> 6;
        for (int s = 0; s < nsb; ++s) {
            const uint8_t* blk = wr + static_cast<size_t>(s) * 70;
            const float eff = decode_sk_sub_eff_i8(blk, CODE_S2_K, sub);
            const uint4 xv = *reinterpret_cast<const uint4*>(xsrc + (s << 8) + j0);
            const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S2_K, j0);
            float d = fdot2_f32(wv.x, xv.x, 0.0f);
            d = fdot2_f32(wv.y, xv.y, d);
            d = fdot2_f32(wv.z, xv.z, d);
            d = fdot2_f32(wv.w, xv.w, d);
            acc = fmaf(d, eff, acc);
        }
    } else if (desc.gu_code == CODE_S3_K) {
        const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
        const int nsb = D >> 8;
        const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * 102;
        const int j0 = lane * 8;
        const int sub = j0 >> 6;
        for (int s = 0; s < nsb; ++s) {
            const uint8_t* blk = wr + static_cast<size_t>(s) * 102;
            const float eff = decode_sk_sub_eff_i8(blk, CODE_S3_K, sub);
            const uint4 xv = *reinterpret_cast<const uint4*>(xsrc + (s << 8) + j0);
            const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S3_K, j0);
            float d = fdot2_f32(wv.x, xv.x, 0.0f);
            d = fdot2_f32(wv.y, xv.y, d);
            d = fdot2_f32(wv.z, xv.z, d);
            d = fdot2_f32(wv.w, xv.w, d);
            acc = fmaf(d, eff, acc);
        }
    } else {
        const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
        const int nsb = D >> 8;
        const int sb_b = sk_sb_bytes(desc.gu_code);
        const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * sb_b;
        const int j0 = lane * 8;
        const int sub = j0 >> 6;
        for (int s = 0; s < nsb; ++s) {
            const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
            const float eff = decode_sk_sub_eff_i8(blk, desc.gu_code, sub);
            const uint4 xv = *reinterpret_cast<const uint4*>(xsrc + (s << 8) + j0);
            const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, desc.gu_code, j0);
            float d = fdot2_f32(wv.x, xv.x, 0.0f);
            d = fdot2_f32(wv.y, xv.y, d);
            d = fdot2_f32(wv.z, xv.z, d);
            d = fdot2_f32(wv.w, xv.w, d);
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

// Stage 2A (Wave-6): Parallel-Expert w2 across all (b, k) active experts -> 8x more blocks (2,304 at B=1!),
// LDS-cached sA[1024] (2 KB shared by all 8 waves), 1 KB SF4 LUT, zero-switch S2_K/S3_K hot loops!
// Grid: ((D + 7) / 8, B * top_k), Block: 256
__global__ void __launch_bounds__(256) k_moe_direct_w2_par(
    const GpuExpertDesc* __restrict__ exps,   // [128]
    const int32_t* __restrict__ topk_ids,     // [B, top_k]
    const half_t* __restrict__ act_in,        // [B * top_k, inter]
    float* __restrict__ y_part,               // [B * top_k, D] (reuses sc_bucket_y_!)
    int B, int top_k, int D, int inter) {
    const int slot_k = blockIdx.y;
    if (slot_k >= B * top_k) return;
    const int eid = topk_ids[slot_k];
    const GpuExpertDesc desc = exps[eid];
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;
    if (!desc.body) {
        if (lane == 0 && row < D) y_part[static_cast<size_t>(slot_k) * D + row] = 0.0f;
        return;
    }

    const half_t* arow = act_in + static_cast<size_t>(slot_k) * inter;
    __shared__ alignas(16) half_t sA[1024];  // 2,048 B
    __shared__ half_t s_e4m3[256];           //   512 B
    __shared__ uint32_t s_sf4_pair[256];     // 1,024 B
    __shared__ float s_sf4_sc[256];          // 1,024 B
    const bool use_sa = (inter <= 1024 && (inter & 7) == 0);
    if (use_sa) {
        const int i8 = inter >> 3;
        const uint4* a8 = reinterpret_cast<const uint4*>(arow);
        uint4* sa8 = reinterpret_cast<uint4*>(sA);
        for (int i = tid; i < i8; i += 256) sa8[i] = a8[i];
    }
    if (desc.dn_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    } else if (desc.dn_code == CODE_SF4) {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc[tid] = e4m3fn_to_f32(static_cast<uint8_t>(tid));
    }
    __syncthreads();
    if (row >= D) return;
    const half_t* asrc = use_sa ? sA : arow;
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
            const float sc = s_sf4_sc[sr[blk]] * outer;
            const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
            const uint4 x0 = *reinterpret_cast<const uint4*>(asrc + (blk << 4));
            const uint4 x1 = *reinterpret_cast<const uint4*>(asrc + (blk << 4) + 8);
            float d = fdot2_f32(s_sf4_pair[raw.x & 0xFFu], x0.x, 0.0f);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 8) & 0xFFu], x0.y, d);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 16) & 0xFFu], x0.z, d);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 24) & 0xFFu], x0.w, d);
            d = fdot2_f32(s_sf4_pair[raw.y & 0xFFu], x1.x, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 8) & 0xFFu], x1.y, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 16) & 0xFFu], x1.z, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 24) & 0xFFu], x1.w, d);
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
            const uint4* xp = reinterpret_cast<const uint4*>(asrc + (blk << 5));
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
    } else if (desc.dn_code == CODE_S2_K) {
        const uint8_t* W = desc.body + desc.dn_offset;
        const int pair = row >> 1;
        const int is_odd = row & 1;
        const int sb_per_pair = (2 * inter) >> 8;
        const int sub_per_row = inter >> 6;
        const uint8_t* pr = W + static_cast<size_t>(pair) * sb_per_pair * 70;
        const int sub_lane = lane >> 3;
        const int l8 = lane & 7;
        for (int sg0 = 0; sg0 < sub_per_row; sg0 += 4) {
            const int sg = sg0 + sub_lane;
            if (sg < sub_per_row) {
                const int sub_global = is_odd * sub_per_row + sg;
                const uint8_t* blk = pr + static_cast<size_t>(sub_global >> 2) * 70;
                const int g = sub_global & 3;
                const float eff = decode_sk_sub_eff_i8(blk, CODE_S2_K, g);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S2_K, (g << 6) + l8 * 8);
                const uint4 xv = *reinterpret_cast<const uint4*>(asrc + (sg << 6) + l8 * 8);
                float d = fdot2_f32(wv.x, xv.x, 0.0f);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                d = fdot2_f32(wv.w, xv.w, d);
                acc = fmaf(d, eff, acc);
            }
        }
    } else if (desc.dn_code == CODE_S3_K) {
        const uint8_t* W = desc.body + desc.dn_offset;
        const int pair = row >> 1;
        const int is_odd = row & 1;
        const int sb_per_pair = (2 * inter) >> 8;
        const int sub_per_row = inter >> 6;
        const uint8_t* pr = W + static_cast<size_t>(pair) * sb_per_pair * 102;
        const int sub_lane = lane >> 3;
        const int l8 = lane & 7;
        for (int sg0 = 0; sg0 < sub_per_row; sg0 += 4) {
            const int sg = sg0 + sub_lane;
            if (sg < sub_per_row) {
                const int sub_global = is_odd * sub_per_row + sg;
                const uint8_t* blk = pr + static_cast<size_t>(sub_global >> 2) * 102;
                const int g = sub_global & 3;
                const float eff = decode_sk_sub_eff_i8(blk, CODE_S3_K, g);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S3_K, (g << 6) + l8 * 8);
                const uint4 xv = *reinterpret_cast<const uint4*>(asrc + (sg << 6) + l8 * 8);
                float d = fdot2_f32(wv.x, xv.x, 0.0f);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                d = fdot2_f32(wv.w, xv.w, d);
                acc = fmaf(d, eff, acc);
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
        const int sub_lane = lane >> 3;
        const int l8 = lane & 7;
        for (int sg0 = 0; sg0 < sub_per_row; sg0 += 4) {
            const int sg = sg0 + sub_lane;
            if (sg < sub_per_row) {
                const int sub_global = is_odd * sub_per_row + sg;
                const uint8_t* blk = pr + static_cast<size_t>(sub_global >> 2) * sb_b;
                const int g = sub_global & 3;
                const float eff = decode_sk_sub_eff_i8(blk, desc.dn_code, g);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, desc.dn_code, (g << 6) + l8 * 8);
                const uint4 xv = *reinterpret_cast<const uint4*>(asrc + (sg << 6) + l8 * 8);
                float d = fdot2_f32(wv.x, xv.x, 0.0f);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                d = fdot2_f32(wv.w, xv.w, d);
                acc = fmaf(d, eff, acc);
            }
        }
    }

    const float yk = wave32_sum(acc);
    if (lane == 0) y_part[static_cast<size_t>(slot_k) * D + row] = yk;
}

// Stage 2B (Wave-6): Deterministic weighted reduction across top_k=8 partial expert outputs in L2!
// Grid: ((D + 255) / 256, B), Block: 256
__global__ void __launch_bounds__(256) k_moe_direct_reduce(
    const float* __restrict__ y_part,       // [B * top_k, D]
    const float* __restrict__ topk_weights, // [B, top_k]
    float* __restrict__ moe_out,            // [B, D]
    int B, int top_k, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (b >= B || d >= D) return;
    float sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        sum = fmaf(y_part[static_cast<size_t>(b * top_k + k) * D + d], topk_weights[b * top_k + k], sum);
    }
    moe_out[static_cast<size_t>(b) * D + d] = sum;
}

// Stage 2B-Fused (Wave-6B): Fuses deterministic MoE weighted reduce + 4-tap MLP ShortConv + residual add into cur!
// Eliminates 2 launches/visit (96/tok) + moe_out & msc DRAM round-trips completely!
// Grid: ((D + 255) / 256, B), Block: 256
__global__ void __launch_bounds__(256) k_moe_reduce_sconv_add(
    const float* __restrict__ y_part,       // [B * top_k, D]
    const float* __restrict__ topk_weights, // [B, top_k]
    const float* __restrict__ static_w,     // [D, 4]
    const int32_t* __restrict__ slot_ids,   // [B]
    uint16_t* __restrict__ hist_pool,       // [max_slots, 3, D]
    float* __restrict__ cur,                // [B, D] (updated in-place!)
    int B, int top_k, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (b >= B || d >= D) return;
    float x0 = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        x0 = fmaf(y_part[static_cast<size_t>(b * top_k + k) * D + d], topk_weights[b * top_k + k], x0);
    }
    const int slot = slot_ids[b];
    uint16_t* h_slot = hist_pool + static_cast<size_t>(slot) * 3 * D;
    const float h0 = bf16_to_f32(h_slot[0 * D + d]);
    const float h1 = bf16_to_f32(h_slot[1 * D + d]);
    const float h2 = bf16_to_f32(h_slot[2 * D + d]);
    const float4 sw = *reinterpret_cast<const float4*>(static_w + static_cast<size_t>(d) * 4);
    const float y = (x0 + sw.x * h0 + sw.y * h1 + sw.z * h2 + sw.w * x0) * 0.44721359549995793f;
    h_slot[0 * D + d] = h_slot[1 * D + d];
    h_slot[1 * D + d] = h_slot[2 * D + d];
    h_slot[2 * D + d] = f32_to_bf16(x0);
    cur[static_cast<size_t>(b) * D + d] += y;
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
            // Paired 2-row S_K: 4 subgroups (256 weights) per wave iter via 8-weight fast 32-bit unpack + uint4 loads!
            const uint8_t* W = desc.body + desc.dn_offset;
            const int pair = row >> 1;
            const int is_odd = row & 1;
            const int sb_per_pair = (2 * inter) >> 8;
            const int sb_b = sk_sb_bytes(desc.dn_code);
            const int sub_per_row = inter >> 6;
            const uint8_t* pr = W + static_cast<size_t>(pair) * sb_per_pair * sb_b;
            const int sub_lane = lane >> 3;  // 0..3 (4 subgroups in parallel per wave!)
            const int l8 = lane & 7;         // 0..7 (8 weights per lane -> 64 weights/subgroup!)
            for (int sg0 = 0; sg0 < sub_per_row; sg0 += 4) {
                const int sg = sg0 + sub_lane;
                if (sg < sub_per_row) {
                    const int sub_global = is_odd * sub_per_row + sg;
                    const int s = sub_global >> 2;
                    const int g = sub_global & 3;
                    const uint8_t* blk = pr + static_cast<size_t>(s) * sb_b;
                    const float eff = decode_sk_sub_eff_i8(blk, desc.dn_code, g);
                    const int j0 = (g << 6) + l8 * 8;
                    const int col0 = (sg << 6) + l8 * 8;
                    const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, desc.dn_code, j0);
                    const uint4 xv = *reinterpret_cast<const uint4*>(arow + col0);
                    float d = fdot2_f32(wv.x, xv.x, 0.0f);
                    d = fdot2_f32(wv.y, xv.y, d);
                    d = fdot2_f32(wv.z, xv.z, d);
                    d = fdot2_f32(wv.w, xv.w, d);
                    acc = fmaf(d, eff, acc);
                }
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

// Parallel Multi-Block MoE Bucket Builder for B >= 32 (replaces 1-CU serial k_moe_build_buckets -> 90x faster at B=2048!)
__global__ void __launch_bounds__(256) k_moe_count_buckets(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ exp_cnt,  // [129] pre-zeroed
    int total) {
    __shared__ int32_t s_cnt[128];
    const int tid = threadIdx.x;
    if (tid < 128) s_cnt[tid] = 0;
    __syncthreads();
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < total; i += stride) {
        const int e = topk_ids[i];
        if (e >= 0 && e < 128) atomicAdd(&s_cnt[e], 1);
    }
    __syncthreads();
    if (tid < 128 && s_cnt[tid] > 0) atomicAdd(&exp_cnt[tid], s_cnt[tid]);
}

__global__ void __launch_bounds__(128) k_moe_prefix_buckets(
    const int32_t* __restrict__ exp_cnt,
    int32_t* __restrict__ exp_off,  // [129]
    int32_t* __restrict__ exp_cur)  // [128]
{
    __shared__ int32_t s_off[129];
    const int e = threadIdx.x;
    s_off[e] = exp_cnt[e];
    __syncthreads();
    if (e == 0) {
        int acc = 0;
        for (int i = 0; i < 128; ++i) {
            const int c = s_off[i];
            s_off[i] = acc;
            acc += c;
        }
        s_off[128] = acc;
        exp_off[128] = acc;
    }
    __syncthreads();
    exp_off[e] = s_off[e];
    exp_cur[e] = s_off[e];
}

__global__ void __launch_bounds__(256) k_moe_scatter_buckets(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ exp_cur,
    int32_t* __restrict__ bucket_tok,
    int32_t* __restrict__ slotk_to_bidx,
    int total, int top_k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int e = topk_ids[i];
    const int pos = atomicAdd(&exp_cur[e], 1);
    bucket_tok[pos] = i / top_k;
    slotk_to_bidx[i] = pos;
}

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
    const int row = (j << 1) | (wave & 1);
    const int m_e = end - beg;

    __shared__ alignas(16) half_t sX[2304];  // 4,608 B (used when m_e == 1!)
    __shared__ half_t s_e4m3[256];           //   512 B
    __shared__ uint32_t s_sf4_pair[256];     // 1,024 B
    __shared__ float s_sf4_sc[256];          // 1,024 B
    __shared__ float s_gu[8 * 8];            //   256 B ([8 waves][8 tokens])

    const bool single_tok = (m_e == 1);
    const bool use_sx = (single_tok && D <= 2304 && (D & 7) == 0);
    const int tok0 = single_tok ? bucket_tok[beg] : 0;
    if (use_sx) {
        const int d8 = D >> 3;
        const uint4* x8 = reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(tok0) * D);
        uint4* sx8 = reinterpret_cast<uint4*>(sX);
        for (int i = tid; i < d8; i += 256) sx8[i] = x8[i];
    }
    if (desc.gu_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    } else if (desc.gu_code == CODE_SF4) {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc[tid] = e4m3fn_to_f32(static_cast<uint8_t>(tid));
    }
    __syncthreads();
    if (j >= inter) return;

    // Uniform fast path when m_e == 1 (~70% of active experts in tree verify / B=8 decode!):
    // Zero multi-token loop/branch overhead + LDS-cached sX[D] -> 100% DRAM bandwidth!
    if (single_tok) {
        const half_t* xsrc = use_sx ? sX : (X_f16 + static_cast<size_t>(tok0) * D);
        float acc = 0.0f;
        if (desc.gu_code == CODE_SF4) {
            const float* hdr = reinterpret_cast<const float*>(desc.body);
            const float outer = hdr[0] / hdr[1];
            const uint8_t* packed = desc.body + 16;
            const uint8_t* scales = packed + static_cast<size_t>(2 * inter) * (D >> 1);
            const uint8_t* pr = packed + static_cast<size_t>(row) * (D >> 1);
            const uint8_t* sr = scales + static_cast<size_t>(row) * (D >> 4);
            const int nblk = D >> 4;
            for (int blk = lane; blk < nblk; blk += 32) {
                const float sc = s_sf4_sc[sr[blk]] * outer;
                const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
                const uint4 x0 = *reinterpret_cast<const uint4*>(xsrc + (blk << 4));
                const uint4 x1 = *reinterpret_cast<const uint4*>(xsrc + (blk << 4) + 8);
                float d = fdot2_f32(s_sf4_pair[raw.x & 0xFFu], x0.x, 0.0f);
                d = fdot2_f32(s_sf4_pair[(raw.x >> 8) & 0xFFu], x0.y, d);
                d = fdot2_f32(s_sf4_pair[(raw.x >> 16) & 0xFFu], x0.z, d);
                d = fdot2_f32(s_sf4_pair[(raw.x >> 24) & 0xFFu], x0.w, d);
                d = fdot2_f32(s_sf4_pair[raw.y & 0xFFu], x1.x, d);
                d = fdot2_f32(s_sf4_pair[(raw.y >> 8) & 0xFFu], x1.y, d);
                d = fdot2_f32(s_sf4_pair[(raw.y >> 16) & 0xFFu], x1.z, d);
                d = fdot2_f32(s_sf4_pair[(raw.y >> 24) & 0xFFu], x1.w, d);
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
                const uint4* xp = reinterpret_cast<const uint4*>(xsrc + (blk << 5));
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
        } else if (desc.gu_code == CODE_S2_K) {
            const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
            const int nsb = D >> 8;
            const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * 70;
            const int j0 = lane * 8;
            const int sub = j0 >> 6;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * 70;
                const float eff = decode_sk_sub_eff_i8(blk, CODE_S2_K, sub);
                const uint4 xv = *reinterpret_cast<const uint4*>(xsrc + (s << 8) + j0);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S2_K, j0);
                float d = fdot2_f32(wv.x, xv.x, 0.0f);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                d = fdot2_f32(wv.w, xv.w, d);
                acc = fmaf(d, eff, acc);
            }
        } else if (desc.gu_code == CODE_S3_K) {
            const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
            const int nsb = D >> 8;
            const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * 102;
            const int j0 = lane * 8;
            const int sub = j0 >> 6;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * 102;
                const float eff = decode_sk_sub_eff_i8(blk, CODE_S3_K, sub);
                const uint4 xv = *reinterpret_cast<const uint4*>(xsrc + (s << 8) + j0);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S3_K, j0);
                float d = fdot2_f32(wv.x, xv.x, 0.0f);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                d = fdot2_f32(wv.w, xv.w, d);
                acc = fmaf(d, eff, acc);
            }
        } else {
            const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
            const int nsb = D >> 8;
            const int sb_b = sk_sb_bytes(desc.gu_code);
            const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * sb_b;
            const int j0 = lane * 8;
            const int sub = j0 >> 6;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
                const float eff = decode_sk_sub_eff_i8(blk, desc.gu_code, sub);
                const uint4 xv = *reinterpret_cast<const uint4*>(xsrc + (s << 8) + j0);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, desc.gu_code, j0);
                float d = fdot2_f32(wv.x, xv.x, 0.0f);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                d = fdot2_f32(wv.w, xv.w, d);
                acc = fmaf(d, eff, acc);
            }
        }
        const float sum = wave32_sum(acc);
        if (lane == 0) s_gu[wave * 8] = sum;
        __syncthreads();
        if (lane == 0 && (wave & 1) == 0) {
            const bool need_sq = (desc.dn_code != CODE_SF4 && desc.dn_code != CODE_MX8);
            const float inv_sq = need_sq ? (1.0f / static_cast<float>(reinterpret_cast<const half_t*>(desc.body)[j])) : 1.0f;
            bucket_act[static_cast<size_t>(beg) * inter + j] =
                static_cast<half_t>(situ_f32(s_gu[wave * 8], s_gu[(wave + 1) * 8], beta, lb) * inv_sq);
        }
        return;
    }

    // Multi-token path (m_e >= 2): stream expert weights ONCE per 8 tokens in low-VGPR register tiles!
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
                const float sc = s_sf4_sc[sr[blk]] * outer;
                const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
                const uint32_t w0 = s_sf4_pair[raw.x & 0xFFu];
                const uint32_t w1 = s_sf4_pair[(raw.x >> 8) & 0xFFu];
                const uint32_t w2 = s_sf4_pair[(raw.x >> 16) & 0xFFu];
                const uint32_t w3 = s_sf4_pair[(raw.x >> 24) & 0xFFu];
                const uint32_t w4 = s_sf4_pair[raw.y & 0xFFu];
                const uint32_t w5 = s_sf4_pair[(raw.y >> 8) & 0xFFu];
                const uint32_t w6 = s_sf4_pair[(raw.y >> 16) & 0xFFu];
                const uint32_t w7 = s_sf4_pair[(raw.y >> 24) & 0xFFu];
                const int k0 = blk << 4;
                for (int t = 0; t < tm; ++t) {
                    const half_t* xp = X_f16 + static_cast<size_t>(toks[t]) * D + k0;
                    const uint4 x0 = *reinterpret_cast<const uint4*>(xp);
                    const uint4 x1 = *reinterpret_cast<const uint4*>(xp + 8);
                    float d = fdot2_f32(w0, x0.x, 0.0f);
                    d = fdot2_f32(w1, x0.y, d); d = fdot2_f32(w2, x0.z, d); d = fdot2_f32(w3, x0.w, d);
                    d = fdot2_f32(w4, x1.x, d); d = fdot2_f32(w5, x1.y, d); d = fdot2_f32(w6, x1.z, d); d = fdot2_f32(w7, x1.w, d);
                    acc[t] = fmaf(d, sc, acc[t]);
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
                const int k0 = blk << 5;
                float dt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const hv2 h0 = {s_e4m3[wb[q*8+0]], s_e4m3[wb[q*8+1]]};
                    const hv2 h1 = {s_e4m3[wb[q*8+2]], s_e4m3[wb[q*8+3]]};
                    const hv2 h2 = {s_e4m3[wb[q*8+4]], s_e4m3[wb[q*8+5]]};
                    const hv2 h3 = {s_e4m3[wb[q*8+6]], s_e4m3[wb[q*8+7]]};
                    const uint32_t w0 = __builtin_bit_cast(uint32_t, h0);
                    const uint32_t w1 = __builtin_bit_cast(uint32_t, h1);
                    const uint32_t w2 = __builtin_bit_cast(uint32_t, h2);
                    const uint32_t w3 = __builtin_bit_cast(uint32_t, h3);
                    for (int t = 0; t < tm; ++t) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(toks[t]) * D + k0 + q * 8);
                        float d = dt[t];
                        d = fdot2_f32(w0, xv.x, d); d = fdot2_f32(w1, xv.y, d);
                        d = fdot2_f32(w2, xv.z, d); d = fdot2_f32(w3, xv.w, d);
                        dt[t] = d;
                    }
                }
                for (int t = 0; t < tm; ++t) acc[t] = fmaf(dt[t], sc, acc[t]);
            }
        } else if (desc.gu_code == CODE_S2_K) {
            const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
            const int nsb = D >> 8;
            const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * 70;
            const int j0 = lane * 8;
            const int sub = j0 >> 6;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * 70;
                const float eff = decode_sk_sub_eff_i8(blk, CODE_S2_K, sub);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S2_K, j0);
                const int k0 = (s << 8) + j0;
                for (int t = 0; t < tm; ++t) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(toks[t]) * D + k0);
                    float d = fdot2_f32(wv.x, xv.x, 0.0f);
                    d = fdot2_f32(wv.y, xv.y, d); d = fdot2_f32(wv.z, xv.z, d); d = fdot2_f32(wv.w, xv.w, d);
                    acc[t] = fmaf(d, eff, acc[t]);
                }
            }
        } else if (desc.gu_code == CODE_S3_K) {
            const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
            const int nsb = D >> 8;
            const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * 102;
            const int j0 = lane * 8;
            const int sub = j0 >> 6;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * 102;
                const float eff = decode_sk_sub_eff_i8(blk, CODE_S3_K, sub);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S3_K, j0);
                const int k0 = (s << 8) + j0;
                for (int t = 0; t < tm; ++t) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(toks[t]) * D + k0);
                    float d = fdot2_f32(wv.x, xv.x, 0.0f);
                    d = fdot2_f32(wv.y, xv.y, d); d = fdot2_f32(wv.z, xv.z, d); d = fdot2_f32(wv.w, xv.w, d);
                    acc[t] = fmaf(d, eff, acc[t]);
                }
            }
        } else {
            const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
            const int nsb = D >> 8;
            const int sb_b = sk_sb_bytes(desc.gu_code);
            const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * sb_b;
            const int j0 = lane * 8;
            const int sub = j0 >> 6;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
                const float eff = decode_sk_sub_eff_i8(blk, desc.gu_code, sub);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, desc.gu_code, j0);
                const int k0 = (s << 8) + j0;
                for (int t = 0; t < tm; ++t) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(toks[t]) * D + k0);
                    float d = fdot2_f32(wv.x, xv.x, 0.0f);
                    d = fdot2_f32(wv.y, xv.y, d); d = fdot2_f32(wv.z, xv.z, d); d = fdot2_f32(wv.w, xv.w, d);
                    acc[t] = fmaf(d, eff, acc[t]);
                }
            }
        }

        for (int t = 0; t < tm; ++t) {
            const float sum = wave32_sum(acc[t]);
            if (lane == 0) s_gu[wave * 8 + t] = sum;
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
// Wave-7D: uniform m_e==1 LDS-cached fast path + zero-switch S2_K/S3_K 4-subgroup/wave + 1 KB SF4 LUT!
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
    const int m_e = end - beg;

    __shared__ alignas(16) half_t sA[1024];  // 2,048 B (used when m_e == 1!)
    __shared__ half_t s_e4m3[256];           //   512 B
    __shared__ uint32_t s_sf4_pair[256];     // 1,024 B
    __shared__ float s_sf4_sc[256];          // 1,024 B

    const bool single_tok = (m_e == 1);
    const bool use_sa = (single_tok && inter <= 1024 && (inter & 7) == 0);
    if (use_sa) {
        const int i8 = inter >> 3;
        const uint4* a8 = reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg) * inter);
        uint4* sa8 = reinterpret_cast<uint4*>(sA);
        for (int i = tid; i < i8; i += 256) sa8[i] = a8[i];
    }
    if (desc.dn_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    } else if (desc.dn_code == CODE_SF4) {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc[tid] = e4m3fn_to_f32(static_cast<uint8_t>(tid));
    }
    __syncthreads();
    if (row >= D) return;

    // Uniform fast path when m_e == 1 (exact k_moe_direct_w2_par speed!)
    if (single_tok) {
        const half_t* asrc = use_sa ? sA : (bucket_act + static_cast<size_t>(beg) * inter);
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
                const float sc = s_sf4_sc[sr[blk]] * outer;
                const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
                const uint4 x0 = *reinterpret_cast<const uint4*>(asrc + (blk << 4));
                const uint4 x1 = *reinterpret_cast<const uint4*>(asrc + (blk << 4) + 8);
                float d = fdot2_f32(s_sf4_pair[raw.x & 0xFFu], x0.x, 0.0f);
                d = fdot2_f32(s_sf4_pair[(raw.x >> 8) & 0xFFu], x0.y, d);
                d = fdot2_f32(s_sf4_pair[(raw.x >> 16) & 0xFFu], x0.z, d);
                d = fdot2_f32(s_sf4_pair[(raw.x >> 24) & 0xFFu], x0.w, d);
                d = fdot2_f32(s_sf4_pair[raw.y & 0xFFu], x1.x, d);
                d = fdot2_f32(s_sf4_pair[(raw.y >> 8) & 0xFFu], x1.y, d);
                d = fdot2_f32(s_sf4_pair[(raw.y >> 16) & 0xFFu], x1.z, d);
                d = fdot2_f32(s_sf4_pair[(raw.y >> 24) & 0xFFu], x1.w, d);
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
                const uint4* xp = reinterpret_cast<const uint4*>(asrc + (blk << 5));
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
        } else if (desc.dn_code == CODE_S2_K) {
            const uint8_t* W = desc.body + desc.dn_offset;
            const int pair = row >> 1;
            const int is_odd = row & 1;
            const int sb_per_pair = (2 * inter) >> 8;
            const int sub_per_row = inter >> 6;
            const uint8_t* pr = W + static_cast<size_t>(pair) * sb_per_pair * 70;
            const int sub_lane = lane >> 3;
            const int l8 = lane & 7;
            for (int sg0 = 0; sg0 < sub_per_row; sg0 += 4) {
                const int sg = sg0 + sub_lane;
                if (sg < sub_per_row) {
                    const int sub_global = is_odd * sub_per_row + sg;
                    const uint8_t* blk = pr + static_cast<size_t>(sub_global >> 2) * 70;
                    const int g = sub_global & 3;
                    const float eff = decode_sk_sub_eff_i8(blk, CODE_S2_K, g);
                    const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S2_K, (g << 6) + l8 * 8);
                    const uint4 xv = *reinterpret_cast<const uint4*>(asrc + (sg << 6) + l8 * 8);
                    float d = fdot2_f32(wv.x, xv.x, 0.0f);
                    d = fdot2_f32(wv.y, xv.y, d); d = fdot2_f32(wv.z, xv.z, d); d = fdot2_f32(wv.w, xv.w, d);
                    acc = fmaf(d, eff, acc);
                }
            }
        } else if (desc.dn_code == CODE_S3_K) {
            const uint8_t* W = desc.body + desc.dn_offset;
            const int pair = row >> 1;
            const int is_odd = row & 1;
            const int sb_per_pair = (2 * inter) >> 8;
            const int sub_per_row = inter >> 6;
            const uint8_t* pr = W + static_cast<size_t>(pair) * sb_per_pair * 102;
            const int sub_lane = lane >> 3;
            const int l8 = lane & 7;
            for (int sg0 = 0; sg0 < sub_per_row; sg0 += 4) {
                const int sg = sg0 + sub_lane;
                if (sg < sub_per_row) {
                    const int sub_global = is_odd * sub_per_row + sg;
                    const uint8_t* blk = pr + static_cast<size_t>(sub_global >> 2) * 102;
                    const int g = sub_global & 3;
                    const float eff = decode_sk_sub_eff_i8(blk, CODE_S3_K, g);
                    const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S3_K, (g << 6) + l8 * 8);
                    const uint4 xv = *reinterpret_cast<const uint4*>(asrc + (sg << 6) + l8 * 8);
                    float d = fdot2_f32(wv.x, xv.x, 0.0f);
                    d = fdot2_f32(wv.y, xv.y, d); d = fdot2_f32(wv.z, xv.z, d); d = fdot2_f32(wv.w, xv.w, d);
                    acc = fmaf(d, eff, acc);
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
            const int sub_lane = lane >> 3;
            const int l8 = lane & 7;
            for (int sg0 = 0; sg0 < sub_per_row; sg0 += 4) {
                const int sg = sg0 + sub_lane;
                if (sg < sub_per_row) {
                    const int sub_global = is_odd * sub_per_row + sg;
                    const uint8_t* blk = pr + static_cast<size_t>(sub_global >> 2) * sb_b;
                    const int g = sub_global & 3;
                    const float eff = decode_sk_sub_eff_i8(blk, desc.dn_code, g);
                    const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, desc.dn_code, (g << 6) + l8 * 8);
                    const uint4 xv = *reinterpret_cast<const uint4*>(asrc + (sg << 6) + l8 * 8);
                    float d = fdot2_f32(wv.x, xv.x, 0.0f);
                    d = fdot2_f32(wv.y, xv.y, d); d = fdot2_f32(wv.z, xv.z, d); d = fdot2_f32(wv.w, xv.w, d);
                    acc = fmaf(d, eff, acc);
                }
            }
        }
        const float sum = wave32_sum(acc);
        if (lane == 0) bucket_y[static_cast<size_t>(beg) * D + row] = sum;
        return;
    }

    // Multi-token path (m_e >= 2): stream expert weights ONCE per 8 tokens in low-VGPR register tiles!
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
                const float sc = s_sf4_sc[sr[blk]] * outer;
                const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
                const uint32_t w0 = s_sf4_pair[raw.x & 0xFFu];
                const uint32_t w1 = s_sf4_pair[(raw.x >> 8) & 0xFFu];
                const uint32_t w2 = s_sf4_pair[(raw.x >> 16) & 0xFFu];
                const uint32_t w3 = s_sf4_pair[(raw.x >> 24) & 0xFFu];
                const uint32_t w4 = s_sf4_pair[raw.y & 0xFFu];
                const uint32_t w5 = s_sf4_pair[(raw.y >> 8) & 0xFFu];
                const uint32_t w6 = s_sf4_pair[(raw.y >> 16) & 0xFFu];
                const uint32_t w7 = s_sf4_pair[(raw.y >> 24) & 0xFFu];
                const int k0 = blk << 4;
                for (int t = 0; t < tm; ++t) {
                    const half_t* ar = bucket_act + static_cast<size_t>(beg + t0 + t) * inter + k0;
                    const uint4 x0 = *reinterpret_cast<const uint4*>(ar);
                    const uint4 x1 = *reinterpret_cast<const uint4*>(ar + 8);
                    float d = fdot2_f32(w0, x0.x, 0.0f);
                    d = fdot2_f32(w1, x0.y, d); d = fdot2_f32(w2, x0.z, d); d = fdot2_f32(w3, x0.w, d);
                    d = fdot2_f32(w4, x1.x, d); d = fdot2_f32(w5, x1.y, d); d = fdot2_f32(w6, x1.z, d); d = fdot2_f32(w7, x1.w, d);
                    acc[t] = fmaf(d, sc, acc[t]);
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
                const int k0 = blk << 5;
                float dt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const hv2 h0 = {s_e4m3[wb[q*8+0]], s_e4m3[wb[q*8+1]]};
                    const hv2 h1 = {s_e4m3[wb[q*8+2]], s_e4m3[wb[q*8+3]]};
                    const hv2 h2 = {s_e4m3[wb[q*8+4]], s_e4m3[wb[q*8+5]]};
                    const hv2 h3 = {s_e4m3[wb[q*8+6]], s_e4m3[wb[q*8+7]]};
                    const uint32_t w0 = __builtin_bit_cast(uint32_t, h0);
                    const uint32_t w1 = __builtin_bit_cast(uint32_t, h1);
                    const uint32_t w2 = __builtin_bit_cast(uint32_t, h2);
                    const uint32_t w3 = __builtin_bit_cast(uint32_t, h3);
                    for (int t = 0; t < tm; ++t) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg + t0 + t) * inter + k0 + q * 8);
                        float d = dt[t];
                        d = fdot2_f32(w0, xv.x, d); d = fdot2_f32(w1, xv.y, d);
                        d = fdot2_f32(w2, xv.z, d); d = fdot2_f32(w3, xv.w, d);
                        dt[t] = d;
                    }
                }
                for (int t = 0; t < tm; ++t) acc[t] = fmaf(dt[t], sc, acc[t]);
            }
        } else if (desc.dn_code == CODE_S2_K) {
            const uint8_t* W = desc.body + desc.dn_offset;
            const int pair = row >> 1;
            const int is_odd = row & 1;
            const int sb_per_pair = (2 * inter) >> 8;
            const int sub_per_row = inter >> 6;
            const uint8_t* pr = W + static_cast<size_t>(pair) * sb_per_pair * 70;
            const int sub_lane = lane >> 3;
            const int l8 = lane & 7;
            for (int sg0 = 0; sg0 < sub_per_row; sg0 += 4) {
                const int sg = sg0 + sub_lane;
                if (sg < sub_per_row) {
                    const int sub_global = is_odd * sub_per_row + sg;
                    const uint8_t* blk = pr + static_cast<size_t>(sub_global >> 2) * 70;
                    const int g = sub_global & 3;
                    const float eff = decode_sk_sub_eff_i8(blk, CODE_S2_K, g);
                    const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S2_K, (g << 6) + l8 * 8);
                    const int col0 = (sg << 6) + l8 * 8;
                    for (int t = 0; t < tm; ++t) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg + t0 + t) * inter + col0);
                        float d = fdot2_f32(wv.x, xv.x, 0.0f);
                        d = fdot2_f32(wv.y, xv.y, d); d = fdot2_f32(wv.z, xv.z, d); d = fdot2_f32(wv.w, xv.w, d);
                        acc[t] = fmaf(d, eff, acc[t]);
                    }
                }
            }
        } else if (desc.dn_code == CODE_S3_K) {
            const uint8_t* W = desc.body + desc.dn_offset;
            const int pair = row >> 1;
            const int is_odd = row & 1;
            const int sb_per_pair = (2 * inter) >> 8;
            const int sub_per_row = inter >> 6;
            const uint8_t* pr = W + static_cast<size_t>(pair) * sb_per_pair * 102;
            const int sub_lane = lane >> 3;
            const int l8 = lane & 7;
            for (int sg0 = 0; sg0 < sub_per_row; sg0 += 4) {
                const int sg = sg0 + sub_lane;
                if (sg < sub_per_row) {
                    const int sub_global = is_odd * sub_per_row + sg;
                    const uint8_t* blk = pr + static_cast<size_t>(sub_global >> 2) * 102;
                    const int g = sub_global & 3;
                    const float eff = decode_sk_sub_eff_i8(blk, CODE_S3_K, g);
                    const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, CODE_S3_K, (g << 6) + l8 * 8);
                    const int col0 = (sg << 6) + l8 * 8;
                    for (int t = 0; t < tm; ++t) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg + t0 + t) * inter + col0);
                        float d = fdot2_f32(wv.x, xv.x, 0.0f);
                        d = fdot2_f32(wv.y, xv.y, d); d = fdot2_f32(wv.z, xv.z, d); d = fdot2_f32(wv.w, xv.w, d);
                        acc[t] = fmaf(d, eff, acc[t]);
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
            const int sub_lane = lane >> 3;
            const int l8 = lane & 7;
            for (int sg0 = 0; sg0 < sub_per_row; sg0 += 4) {
                const int sg = sg0 + sub_lane;
                if (sg < sub_per_row) {
                    const int sub_global = is_odd * sub_per_row + sg;
                    const uint8_t* blk = pr + static_cast<size_t>(sub_global >> 2) * sb_b;
                    const int g = sub_global & 3;
                    const float eff = decode_sk_sub_eff_i8(blk, desc.dn_code, g);
                    const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, desc.dn_code, (g << 6) + l8 * 8);
                    const int col0 = (sg << 6) + l8 * 8;
                    for (int t = 0; t < tm; ++t) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg + t0 + t) * inter + col0);
                        float d = fdot2_f32(wv.x, xv.x, 0.0f);
                        d = fdot2_f32(wv.y, xv.y, d); d = fdot2_f32(wv.z, xv.z, d); d = fdot2_f32(wv.w, xv.w, d);
                        acc[t] = fmaf(d, eff, acc[t]);
                    }
                }
            }
        }

        for (int t = 0; t < tm; ++t) {
            const float sum = wave32_sum(acc[t]);
            if (lane == 0) bucket_y[static_cast<size_t>(beg + t0 + t) * D + row] = sum;
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
// 4D. Wave32 WMMA BM=64, BN=128 Fused Grouped-MoE (SF4 + MX8 + S_K) for Prefill & Batched Decode (B >= 5)!
// 256 threads = 8 waves (W_M=2, W_N=4 -> 2x2 WMMA tiles/wave) + 3D blockIdx.z M-tile fan-out for hot experts!
// Uses Wave32 __shfl_xor(val, 1) register-partner SITU (ZERO s_out LDS -> 27.5 KB < 32 KB -> 2 WG/CU occupancy!).
// Zero WMMA emitted on RDNA2 (guarded inside body).
// =================================================================================================

__global__ void __launch_bounds__(256) k_moe_grouped_w13_situ_wmma(
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
    const int m_e = end - beg;
    constexpr int BM = 64, BN = 128, BK = 64, LD = BK + 8;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) half_t sX[BM * LD];   // 9,216 B
    __shared__ alignas(16) half_t sW[BN * LD];   // 18,432 B
    __shared__ half_t s_e4m3[256];               // 512 B -> 28,160 B (27.5 KB < 32 KB -> 2 WG/CU!)

    const int tid = threadIdx.x;
    if (desc.gu_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 2;   // 0..1 (32 M rows)
    const int wn = wave & 3;    // 0..3 (32 N cols -> 4*32 = 128!)
    const int N = 2 * inter;
    const int n_blk = blockIdx.x * BN;

    const float sf4_outer = (desc.gu_code == CODE_SF4)
        ? (reinterpret_cast<const float*>(desc.body)[0] / reinterpret_cast<const float*>(desc.body)[1]) : 1.0f;
    const uint8_t* sf4_p = desc.body + 16;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(N) * (D >> 1);
    const uint8_t* mx8_w = desc.body;
    const uint8_t* mx8_s = mx8_w + static_cast<size_t>(N) * D;
    const uint8_t* sk_w  = desc.body + static_cast<size_t>(inter) * 2;
    const int nsb = D >> 8;
    const int sb_b = sk_sb_bytes(desc.gu_code);
    const bool need_sq = (desc.dn_code != CODE_SF4 && desc.dn_code != CODE_MX8);
    const half_t* sq_xd = reinterpret_cast<const half_t*>(desc.body);
    const float out_scale = (desc.gu_code == CODE_SF4) ? (sf4_outer * 16.0f) : 1.0f;
    const int t_stride = gridDim.z * BM;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc[2][2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < D; k0 += BK) {
            const int x_iters = ((tm + 15) & ~15) * 8;
            for (int idx = tid; idx < x_iters; idx += 256) {
                const int r = idx >> 3, c8 = idx & 7;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < tm) {
                    const int tok = bucket_tok[beg + t0 + r];
                    v = *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(tok) * D + k0 + c8 * 8);
                }
                *reinterpret_cast<uint4*>(&sX[r * LD + c8 * 8]) = v;
            }
            if (desc.gu_code == CODE_SF4) {
                for (int idx = tid; idx < BN * 4; idx += 256) {
                    const int r = idx >> 2, b4 = idx & 3;
                    const int gn = n_blk + r;
                    const int gb = (k0 >> 4) + b4;
                    half_t* dst = &sW[r * LD + b4 * 16];
                    if (gn < N) {
                        const float sc = e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * (D >> 4) + gb]) * 0.0625f;
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
                for (int idx = tid; idx < BN * 2; idx += 256) {
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
                for (int idx = tid; idx < BN * 4; idx += 256) {
                    const int r = idx >> 2, q16 = idx & 3;
                    const int gn = n_blk + r;
                    half_t* dst = &sW[r * LD + q16 * 16];
                    if (gn < N) {
                        const uint8_t* blk = sk_w + (static_cast<size_t>(gn) * nsb + s) * sb_b;
                        const uint32_t ef2 = h2_splat(decode_sk_sub_eff_i8(blk, desc.gu_code, g));
                        uint4 o0, o1;
                        decode_sk_16_f16(blk + 6, desc.gu_code, (g << 6) + q16 * 16, ef2, o0, o1);
                        *reinterpret_cast<uint4*>(dst)     = o0;
                        *reinterpret_cast<uint4*>(dst + 8) = o1;
                    } else {
                        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
                    }
                }
            }
            sync_lds();

            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    wmma_h_t b0 = wmma_ld_h(&sW[(wn * 32 + l16) * LD + kk * 16], hi);
                    wmma_h_t b1 = wmma_ld_h(&sW[(wn * 32 + 16 + l16) * LD + kk * 16], hi);
                    wmma_h_t a0 = wmma_ld_h(&sX[(wm * 32 + l16) * LD + kk * 16], hi);
                    acc[0][0] = wmma_f16(a0, b0, acc[0][0]);
                    acc[0][1] = wmma_f16(a0, b1, acc[0][1]);
                    if (do_m1) {
                        wmma_h_t a1 = wmma_ld_h(&sX[(wm * 32 + 16 + l16) * LD + kk * 16], hi);
                        acc[1][0] = wmma_f16(a1, b0, acc[1][0]);
                        acc[1][1] = wmma_f16(a1, b1, acc[1][1]);
                    }
                }
            }
            sync_lds();
        }

        // Direct Wave32 register-shuffle SITU epilogue: even lane=gate, odd lane=up -> ZERO s_out LDS, ZERO barrier!
        if (wm * 32 < tm) {
            const bool is_even = ((lane & 1) == 0);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int gn = n_blk + wn * 32 + ni * 16 + wmma_acc_n(0, lane);
                    const int gj = gn >> 1;
                    const float inv_sq = (is_even && need_sq && gj < inter) ? (1.0f / static_cast<float>(sq_xd[gj])) : 1.0f;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float g_val = acc[mi][ni][r] * out_scale;
                        const float u_val = __shfl_xor(g_val, 1, 32);
                        const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                        if (is_even && lm < tm && gj < inter) {
                            const float v = situ_f32(g_val, u_val, beta, lb) * inv_sq;
                            bucket_act[static_cast<size_t>(beg + t0 + lm) * inter + gj] = static_cast<half_t>(v);
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)X_f16; (void)bucket_act; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

// Wave32 WMMA BM=64, BN=128 Fused Grouped-MoE Stage 2 (w2 for SF4 + MX8 + paired-2-row S_K!) for B >= 5!
// 256 threads = 8 waves, 27.5 KB LDS (< 32 KB -> 2 WG/CU!) + 3D blockIdx.z M-tile fan-out!
__global__ void __launch_bounds__(256) k_moe_grouped_w2_wmma(
    const GpuExpertDesc* __restrict__ exps,  // [128]
    const int32_t* __restrict__ exp_off,     // [129]
    const half_t* __restrict__ bucket_act,   // [B * top_k, inter]
    float* __restrict__ bucket_y,            // [B * top_k, D]
    int D, int inter) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 64, BN = 128, BK = 64, LD = BK + 8;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) half_t sX[BM * LD];
    __shared__ alignas(16) half_t sW[BN * LD];
    __shared__ half_t s_e4m3[256];

    const int tid = threadIdx.x;
    if (desc.dn_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 2;
    const int wn = wave & 3;
    const int n_blk = blockIdx.x * BN;

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
    const float out_scale = (desc.dn_code == CODE_SF4) ? (sf4_outer * 16.0f) : 1.0f;
    const int t_stride = gridDim.z * BM;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc[2][2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < inter; k0 += BK) {
            const int x_iters = ((tm + 15) & ~15) * 8;
            for (int idx = tid; idx < x_iters; idx += 256) {
                const int r = idx >> 3, c8 = idx & 7;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < tm) {
                    v = *reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg + t0 + r) * inter + k0 + c8 * 8);
                }
                *reinterpret_cast<uint4*>(&sX[r * LD + c8 * 8]) = v;
            }
            if (desc.dn_code == CODE_SF4) {
                for (int idx = tid; idx < BN * 4; idx += 256) {
                    const int r = idx >> 2, b4 = idx & 3;
                    const int gn = n_blk + r;
                    const int gb = (k0 >> 4) + b4;
                    half_t* dst = &sW[r * LD + b4 * 16];
                    if (gn < D) {
                        const float sc = e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * (inter >> 4) + gb]) * 0.0625f;
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
                for (int idx = tid; idx < BN * 2; idx += 256) {
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
                for (int idx = tid; idx < BN * 4; idx += 256) {
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
                        uint4 o0, o1;
                        decode_sk_16_f16(blk + 6, desc.dn_code, (g << 6) + q16 * 16, ef2, o0, o1);
                        *reinterpret_cast<uint4*>(dst)     = o0;
                        *reinterpret_cast<uint4*>(dst + 8) = o1;
                    } else {
                        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
                    }
                }
            }
            sync_lds();

            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    wmma_h_t b0 = wmma_ld_h(&sW[(wn * 32 + l16) * LD + kk * 16], hi);
                    wmma_h_t b1 = wmma_ld_h(&sW[(wn * 32 + 16 + l16) * LD + kk * 16], hi);
                    wmma_h_t a0 = wmma_ld_h(&sX[(wm * 32 + l16) * LD + kk * 16], hi);
                    acc[0][0] = wmma_f16(a0, b0, acc[0][0]);
                    acc[0][1] = wmma_f16(a0, b1, acc[0][1]);
                    if (do_m1) {
                        wmma_h_t a1 = wmma_ld_h(&sX[(wm * 32 + 16 + l16) * LD + kk * 16], hi);
                        acc[1][0] = wmma_f16(a1, b0, acc[1][0]);
                        acc[1][1] = wmma_f16(a1, b1, acc[1][1]);
                    }
                }
            }
            sync_lds();
        }

        if (wm * 32 < tm) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int gn = n_blk + wn * 32 + ni * 16 + wmma_acc_n(0, lane);
                    if (gn < D) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                            if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[mi][ni][r] * out_scale;
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)bucket_act; (void)bucket_y; (void)D; (void)inter;
#endif
}

// =================================================================================================
// Wave-3.5 2x-Rate INT8 WMMA Fused Grouped-MoE for S_K (SQ-10 / SQ-15) & SF4 (SQ-20 / SQ-15)!
//   - RDNA4 (gfx120x): 16x16x32 iu8 WMMA (96 TOPS = 2x FP16!) for S_K & 16x16x16 iu8 WMMA for SF4!
//   - RDNA3/3.5 (gfx110x/1103/1150/1151): 2x-rate 16x16x16 iu8 WMMA!
//   - ZERO FP16 weight conversion/multiply! Only 15.75 KB LDS (4 WG/CU occupancy on S_K)!
//   - Zero WMMA emitted on RDNA2 (guarded inside body).
// =================================================================================================

__global__ void __launch_bounds__(256) k_moe_grouped_w13_situ_wmma_sk_i8(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int32_t* __restrict__ bucket_tok,
    const half_t* __restrict__ X_f16,
    half_t* __restrict__ bucket_act,
    int D, int inter, float beta, float lb) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 64, BN = 128, BK = 64, LDI = BK + 16;  // 80 B stride (16B aligned, bank-free!)
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) int8_t sXq[BM * LDI];   //  5,120 B
    __shared__ alignas(16) int8_t sWq[BN * LDI];   // 10,240 B
    __shared__ float sXd[BM];                      //    256 B
    __shared__ float sWs[BN];                      //    512 B -> 16,128 B (15.75 KB!)

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 2;
    const int wn = wave & 3;
    const int N = 2 * inter;
    const int n_blk = blockIdx.x * BN;
    const uint8_t* sk_w = desc.body + static_cast<size_t>(inter) * 2;
    const int nsb = D >> 8;
    const int code = desc.gu_code;
    const int sb_b = sk_sb_bytes(code);
    const half_t* sq_xd = reinterpret_cast<const half_t*>(desc.body);
    const int t_stride = gridDim.z * BM;

    const int xr = tid >> 2;  // 0..63
    const int xs = tid & 3;   // 0..3 (16 elements each -> 4 threads = 64 elements!)

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc[2][2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < D; k0 += BK) {
            // 1. 0-loop parallel Q8_0 quantization of sX[tm, 64] -> sXq[64, 80] + sXd[64]
            if (xr < tm) {
                const int tok = bucket_tok[beg + t0 + xr];
                const half_t* xp = X_f16 + static_cast<size_t>(tok) * D + k0 + xs * 16;
                const uint4 u0 = *reinterpret_cast<const uint4*>(xp);
                const uint4 u1 = *reinterpret_cast<const uint4*>(xp + 8);
                float xf[16];
                h2_unpack(u0.x, xf[0], xf[1]);   h2_unpack(u0.y, xf[2], xf[3]);
                h2_unpack(u0.z, xf[4], xf[5]);   h2_unpack(u0.w, xf[6], xf[7]);
                h2_unpack(u1.x, xf[8], xf[9]);   h2_unpack(u1.y, xf[10], xf[11]);
                h2_unpack(u1.z, xf[12], xf[13]); h2_unpack(u1.w, xf[14], xf[15]);
                float am = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(xf[i]));
                am = fmaxf(am, __shfl_xor(am, 1, 32));
                am = fmaxf(am, __shfl_xor(am, 2, 32));
                const float d = am * (1.0f / 127.0f);
                const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
                int4 p4;
                p4.x = pack4_i8(clamp_i8(xf[0] * inv), clamp_i8(xf[1] * inv), clamp_i8(xf[2] * inv), clamp_i8(xf[3] * inv));
                p4.y = pack4_i8(clamp_i8(xf[4] * inv), clamp_i8(xf[5] * inv), clamp_i8(xf[6] * inv), clamp_i8(xf[7] * inv));
                p4.z = pack4_i8(clamp_i8(xf[8] * inv), clamp_i8(xf[9] * inv), clamp_i8(xf[10] * inv), clamp_i8(xf[11] * inv));
                p4.w = pack4_i8(clamp_i8(xf[12] * inv), clamp_i8(xf[13] * inv), clamp_i8(xf[14] * inv), clamp_i8(xf[15] * inv));
                *reinterpret_cast<int4*>(&sXq[xr * LDI + xs * 16]) = p4;
                if (xs == 0) sXd[xr] = d;
            } else {
                *reinterpret_cast<int4*>(&sXq[xr * LDI + xs * 16]) = make_int4(0, 0, 0, 0);
                if (xs == 0) sXd[xr] = 0.0f;
            }

            // 2. Direct integer S_K unpack -> sWq[128, 80] + sWs[128] (ZERO FP16 conversions!)
            const int s = k0 >> 8;
            const int g = (k0 >> 6) & 3;
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, q16 = idx & 3;
                const int gn = n_blk + r;
                if (gn < N) {
                    const uint8_t* blk = sk_w + (static_cast<size_t>(gn) * nsb + s) * sb_b;
                    *reinterpret_cast<int4*>(&sWq[r * LDI + q16 * 16]) = decode_sk_16_i8(blk + 6, code, (g << 6) + q16 * 16);
                    if (q16 == 0) sWs[r] = decode_sk_sub_eff_i8(blk, code, g);
                } else {
                    *reinterpret_cast<int4*>(&sWq[r * LDI + q16 * 16]) = make_int4(0, 0, 0, 0);
                    if (q16 == 0) sWs[r] = 0.0f;
                }
            }
            sync_lds();

            // 3. 2x-Rate INT8 WMMA across full BK=64 sub-group (1 single scale FMA per 64 K-elements!)
            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
                iv8 iacc[2][2];
                iacc[0][0] = iv8_zero(); iacc[0][1] = iv8_zero();
                iacc[1][0] = iv8_zero(); iacc[1][1] = iv8_zero();
#if INK_GFX12
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    iv4 b0 = wmma_ld_i8_k32(&sWq[(wn * 32 + l16) * LDI + kk * 32], hi);
                    iv4 b1 = wmma_ld_i8_k32(&sWq[(wn * 32 + 16 + l16) * LDI + kk * 32], hi);
                    iv4 a0 = wmma_ld_i8_k32(&sXq[(wm * 32 + l16) * LDI + kk * 32], hi);
                    iacc[0][0] = wmma_i8_k32(a0, b0, iacc[0][0]);
                    iacc[0][1] = wmma_i8_k32(a0, b1, iacc[0][1]);
                    if (do_m1) {
                        iv4 a1 = wmma_ld_i8_k32(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 32], hi);
                        iacc[1][0] = wmma_i8_k32(a1, b0, iacc[1][0]);
                        iacc[1][1] = wmma_i8_k32(a1, b1, iacc[1][1]);
                    }
                }
#elif INK_GFX11
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    iv4 b0 = wmma_ld_i8_k16_g11(&sWq[(wn * 32 + l16) * LDI + kk * 16]);
                    iv4 b1 = wmma_ld_i8_k16_g11(&sWq[(wn * 32 + 16 + l16) * LDI + kk * 16]);
                    iv4 a0 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + l16) * LDI + kk * 16]);
                    iacc[0][0] = wmma_i8_k16_g11(a0, b0, iacc[0][0]);
                    iacc[0][1] = wmma_i8_k16_g11(a0, b1, iacc[0][1]);
                    if (do_m1) {
                        iv4 a1 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16]);
                        iacc[1][0] = wmma_i8_k16_g11(a1, b0, iacc[1][0]);
                        iacc[1][1] = wmma_i8_k16_g11(a1, b1, iacc[1][1]);
                    }
                }
#endif
                const int col16 = wmma_acc_n(0, lane);
                const float ws0 = sWs[wn * 32 + col16];
                const float ws1 = sWs[wn * 32 + 16 + col16];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float xd0 = sXd[wm * 32 + wmma_acc_m(r, lane)];
                    acc[0][0][r] = fmaf(static_cast<float>(iacc[0][0][r]), xd0 * ws0, acc[0][0][r]);
                    acc[0][1][r] = fmaf(static_cast<float>(iacc[0][1][r]), xd0 * ws1, acc[0][1][r]);
                }
                if (do_m1) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd1 = sXd[wm * 32 + 16 + wmma_acc_m(r, lane)];
                        acc[1][0][r] = fmaf(static_cast<float>(iacc[1][0][r]), xd1 * ws0, acc[1][0][r]);
                        acc[1][1][r] = fmaf(static_cast<float>(iacc[1][1][r]), xd1 * ws1, acc[1][1][r]);
                    }
                }
            }
            sync_lds();
        }

        // Direct Wave32 register-shuffle SITU + sq_xd_dn epilogue
        if (wm * 32 < tm) {
            const bool is_even = ((lane & 1) == 0);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int gn = n_blk + wn * 32 + ni * 16 + wmma_acc_n(0, lane);
                    const int gj = gn >> 1;
                    const float inv_sq = (is_even && gj < inter) ? (1.0f / static_cast<float>(sq_xd[gj])) : 1.0f;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float g_val = acc[mi][ni][r];
                        const float u_val = __shfl_xor(g_val, 1, 32);
                        const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                        if (is_even && lm < tm && gj < inter) {
                            bucket_act[static_cast<size_t>(beg + t0 + lm) * inter + gj] =
                                static_cast<half_t>(situ_f32(g_val, u_val, beta, lb) * inv_sq);
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)X_f16; (void)bucket_act; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

__global__ void __launch_bounds__(256) k_moe_grouped_w2_wmma_sk_i8(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const half_t* __restrict__ bucket_act,
    float* __restrict__ bucket_y,
    int D, int inter) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 64, BN = 128, BK = 64, LDI = BK + 16;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) int8_t sXq[BM * LDI];
    __shared__ alignas(16) int8_t sWq[BN * LDI];
    __shared__ float sXd[BM];
    __shared__ float sWs[BN];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 2;
    const int wn = wave & 3;
    const int n_blk = blockIdx.x * BN;
    const uint8_t* sk_w = desc.body + desc.dn_offset;
    const int code = desc.dn_code;
    const int sb_per_pair = (2 * inter) >> 8;
    const int sb_b = sk_sb_bytes(code);
    const int sub_per_row = inter >> 6;
    const int t_stride = gridDim.z * BM;

    const int xr = tid >> 2;
    const int xs = tid & 3;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc[2][2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < inter; k0 += BK) {
            if (xr < tm) {
                const half_t* xp = bucket_act + static_cast<size_t>(beg + t0 + xr) * inter + k0 + xs * 16;
                const uint4 u0 = *reinterpret_cast<const uint4*>(xp);
                const uint4 u1 = *reinterpret_cast<const uint4*>(xp + 8);
                float xf[16];
                h2_unpack(u0.x, xf[0], xf[1]);   h2_unpack(u0.y, xf[2], xf[3]);
                h2_unpack(u0.z, xf[4], xf[5]);   h2_unpack(u0.w, xf[6], xf[7]);
                h2_unpack(u1.x, xf[8], xf[9]);   h2_unpack(u1.y, xf[10], xf[11]);
                h2_unpack(u1.z, xf[12], xf[13]); h2_unpack(u1.w, xf[14], xf[15]);
                float am = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(xf[i]));
                am = fmaxf(am, __shfl_xor(am, 1, 32));
                am = fmaxf(am, __shfl_xor(am, 2, 32));
                const float d = am * (1.0f / 127.0f);
                const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
                int4 p4;
                p4.x = pack4_i8(clamp_i8(xf[0] * inv), clamp_i8(xf[1] * inv), clamp_i8(xf[2] * inv), clamp_i8(xf[3] * inv));
                p4.y = pack4_i8(clamp_i8(xf[4] * inv), clamp_i8(xf[5] * inv), clamp_i8(xf[6] * inv), clamp_i8(xf[7] * inv));
                p4.z = pack4_i8(clamp_i8(xf[8] * inv), clamp_i8(xf[9] * inv), clamp_i8(xf[10] * inv), clamp_i8(xf[11] * inv));
                p4.w = pack4_i8(clamp_i8(xf[12] * inv), clamp_i8(xf[13] * inv), clamp_i8(xf[14] * inv), clamp_i8(xf[15] * inv));
                *reinterpret_cast<int4*>(&sXq[xr * LDI + xs * 16]) = p4;
                if (xs == 0) sXd[xr] = d;
            } else {
                *reinterpret_cast<int4*>(&sXq[xr * LDI + xs * 16]) = make_int4(0, 0, 0, 0);
                if (xs == 0) sXd[xr] = 0.0f;
            }

            const int sg = k0 >> 6;
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, q16 = idx & 3;
                const int gn = n_blk + r;
                if (gn < D) {
                    const int pair = gn >> 1;
                    const int is_odd = gn & 1;
                    const int sub_global = is_odd * sub_per_row + sg;
                    const int s = sub_global >> 2;
                    const int g = sub_global & 3;
                    const uint8_t* blk = sk_w + (static_cast<size_t>(pair) * sb_per_pair + s) * sb_b;
                    *reinterpret_cast<int4*>(&sWq[r * LDI + q16 * 16]) = decode_sk_16_i8(blk + 6, code, (g << 6) + q16 * 16);
                    if (q16 == 0) sWs[r] = decode_sk_sub_eff_i8(blk, code, g);
                } else {
                    *reinterpret_cast<int4*>(&sWq[r * LDI + q16 * 16]) = make_int4(0, 0, 0, 0);
                    if (q16 == 0) sWs[r] = 0.0f;
                }
            }
            sync_lds();

            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
                iv8 iacc[2][2];
                iacc[0][0] = iv8_zero(); iacc[0][1] = iv8_zero();
                iacc[1][0] = iv8_zero(); iacc[1][1] = iv8_zero();
#if INK_GFX12
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    iv4 b0 = wmma_ld_i8_k32(&sWq[(wn * 32 + l16) * LDI + kk * 32], hi);
                    iv4 b1 = wmma_ld_i8_k32(&sWq[(wn * 32 + 16 + l16) * LDI + kk * 32], hi);
                    iv4 a0 = wmma_ld_i8_k32(&sXq[(wm * 32 + l16) * LDI + kk * 32], hi);
                    iacc[0][0] = wmma_i8_k32(a0, b0, iacc[0][0]);
                    iacc[0][1] = wmma_i8_k32(a0, b1, iacc[0][1]);
                    if (do_m1) {
                        iv4 a1 = wmma_ld_i8_k32(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 32], hi);
                        iacc[1][0] = wmma_i8_k32(a1, b0, iacc[1][0]);
                        iacc[1][1] = wmma_i8_k32(a1, b1, iacc[1][1]);
                    }
                }
#elif INK_GFX11
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    iv4 b0 = wmma_ld_i8_k16_g11(&sWq[(wn * 32 + l16) * LDI + kk * 16]);
                    iv4 b1 = wmma_ld_i8_k16_g11(&sWq[(wn * 32 + 16 + l16) * LDI + kk * 16]);
                    iv4 a0 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + l16) * LDI + kk * 16]);
                    iacc[0][0] = wmma_i8_k16_g11(a0, b0, iacc[0][0]);
                    iacc[0][1] = wmma_i8_k16_g11(a0, b1, iacc[0][1]);
                    if (do_m1) {
                        iv4 a1 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16]);
                        iacc[1][0] = wmma_i8_k16_g11(a1, b0, iacc[1][0]);
                        iacc[1][1] = wmma_i8_k16_g11(a1, b1, iacc[1][1]);
                    }
                }
#endif
                const int col16 = wmma_acc_n(0, lane);
                const float ws0 = sWs[wn * 32 + col16];
                const float ws1 = sWs[wn * 32 + 16 + col16];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float xd0 = sXd[wm * 32 + wmma_acc_m(r, lane)];
                    acc[0][0][r] = fmaf(static_cast<float>(iacc[0][0][r]), xd0 * ws0, acc[0][0][r]);
                    acc[0][1][r] = fmaf(static_cast<float>(iacc[0][1][r]), xd0 * ws1, acc[0][1][r]);
                }
                if (do_m1) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd1 = sXd[wm * 32 + 16 + wmma_acc_m(r, lane)];
                        acc[1][0][r] = fmaf(static_cast<float>(iacc[1][0][r]), xd1 * ws0, acc[1][0][r]);
                        acc[1][1][r] = fmaf(static_cast<float>(iacc[1][1][r]), xd1 * ws1, acc[1][1][r]);
                    }
                }
            }
            sync_lds();
        }

        if (wm * 32 < tm) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int gn = n_blk + wn * 32 + ni * 16 + wmma_acc_n(0, lane);
                    if (gn < D) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                            if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[mi][ni][r];
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)bucket_act; (void)bucket_y; (void)D; (void)inter;
#endif
}

// Wave-3.5 2x-Rate INT8 WMMA Fused Grouped-MoE for SF4 (SQ-20 Layers 0..19 & SQ-15 SF4 layers)!
// Exact 2x int8 LUT (sf4_e2m1_i8_x2) + FP32 per-16 scales (ZERO FP16 underflow, ZERO S_K/MX8 switch bloat!)
__global__ void __launch_bounds__(256) k_moe_grouped_w13_situ_wmma_sf4_i8(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int32_t* __restrict__ bucket_tok,
    const half_t* __restrict__ X_f16,
    half_t* __restrict__ bucket_act,
    int D, int inter, float beta, float lb) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 64, BN = 128, BK = 64, LDI = BK + 16;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) int8_t sXq[BM * LDI];    //  5,120 B
    __shared__ alignas(16) int8_t sWq[BN * LDI];    // 10,240 B
    __shared__ float sXd[BM * 4];                   //  1,024 B
    __shared__ float sWs[BN * 4];                   //  2,048 B -> 18,432 B (18 KB < 32 KB -> 2+ WG/CU!)

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 2;
    const int wn = wave & 3;
    const int N = 2 * inter;
    const int n_blk = blockIdx.x * BN;
    const float half_outer = 0.5f * (reinterpret_cast<const float*>(desc.body)[0] / reinterpret_cast<const float*>(desc.body)[1]);
    const uint8_t* sf4_p = desc.body + 16;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(N) * (D >> 1);
    const int t_stride = gridDim.z * BM;

    const int xr = tid >> 2;
    const int xs = tid & 3;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc[2][2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < D; k0 += BK) {
            // 1. Per-16-block activation Q8_0 (0-loop, 0 shuffles!)
            if (xr < tm) {
                const int tok = bucket_tok[beg + t0 + xr];
                const half_t* xp = X_f16 + static_cast<size_t>(tok) * D + k0 + xs * 16;
                const uint4 u0 = *reinterpret_cast<const uint4*>(xp);
                const uint4 u1 = *reinterpret_cast<const uint4*>(xp + 8);
                float xf[16];
                h2_unpack(u0.x, xf[0], xf[1]);   h2_unpack(u0.y, xf[2], xf[3]);
                h2_unpack(u0.z, xf[4], xf[5]);   h2_unpack(u0.w, xf[6], xf[7]);
                h2_unpack(u1.x, xf[8], xf[9]);   h2_unpack(u1.y, xf[10], xf[11]);
                h2_unpack(u1.z, xf[12], xf[13]); h2_unpack(u1.w, xf[14], xf[15]);
                float am = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(xf[i]));
                const float d = am * (1.0f / 127.0f);
                const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
                int4 p4;
                p4.x = pack4_i8(clamp_i8(xf[0] * inv), clamp_i8(xf[1] * inv), clamp_i8(xf[2] * inv), clamp_i8(xf[3] * inv));
                p4.y = pack4_i8(clamp_i8(xf[4] * inv), clamp_i8(xf[5] * inv), clamp_i8(xf[6] * inv), clamp_i8(xf[7] * inv));
                p4.z = pack4_i8(clamp_i8(xf[8] * inv), clamp_i8(xf[9] * inv), clamp_i8(xf[10] * inv), clamp_i8(xf[11] * inv));
                p4.w = pack4_i8(clamp_i8(xf[12] * inv), clamp_i8(xf[13] * inv), clamp_i8(xf[14] * inv), clamp_i8(xf[15] * inv));
                *reinterpret_cast<int4*>(&sXq[xr * LDI + xs * 16]) = p4;
                sXd[xr * 4 + xs] = d;
            } else {
                *reinterpret_cast<int4*>(&sXq[xr * LDI + xs * 16]) = make_int4(0, 0, 0, 0);
                sXd[xr * 4 + xs] = 0.0f;
            }

            // 2. Exact 2x int8 SF4 unpack -> sWq[128, 80] + FP32 sWs[128, 4]
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, b4 = idx & 3;
                const int gn = n_blk + r;
                const int gb = (k0 >> 4) + b4;
                if (gn < N) {
                    sWs[r * 4 + b4] = e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * (D >> 4) + gb]) * half_outer;
                    const uint2 raw = *reinterpret_cast<const uint2*>(sf4_p + static_cast<size_t>(gn) * (D >> 1) + gb * 8);
                    int4 p4;
                    p4.x = pack4_i8(sf4_e2m1_i8_x2(raw.x & 15u), sf4_e2m1_i8_x2((raw.x >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 8) & 15u), sf4_e2m1_i8_x2((raw.x >> 12) & 15u));
                    p4.y = pack4_i8(sf4_e2m1_i8_x2((raw.x >> 16) & 15u), sf4_e2m1_i8_x2((raw.x >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 24) & 15u), sf4_e2m1_i8_x2((raw.x >> 28) & 15u));
                    p4.z = pack4_i8(sf4_e2m1_i8_x2(raw.y & 15u), sf4_e2m1_i8_x2((raw.y >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 8) & 15u), sf4_e2m1_i8_x2((raw.y >> 12) & 15u));
                    p4.w = pack4_i8(sf4_e2m1_i8_x2((raw.y >> 16) & 15u), sf4_e2m1_i8_x2((raw.y >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 24) & 15u), sf4_e2m1_i8_x2((raw.y >> 28) & 15u));
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = p4;
                } else {
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = make_int4(0, 0, 0, 0);
                    sWs[r * 4 + b4] = 0.0f;
                }
            }
            sync_lds();

            // 3. 4x 16x16x16 INT8 WMMA with per-16 FP32 scale accumulation
            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
                const int col16 = wmma_acc_n(0, lane);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
#if INK_GFX12
                    iv2 b0 = wmma_ld_i8_k16(&sWq[(wn * 32 + l16) * LDI + kk * 16], hi);
                    iv2 b1 = wmma_ld_i8_k16(&sWq[(wn * 32 + 16 + l16) * LDI + kk * 16], hi);
                    iv2 a0 = wmma_ld_i8_k16(&sXq[(wm * 32 + l16) * LDI + kk * 16], hi);
                    iv8 i00 = wmma_i8_k16(a0, b0, iv8_zero());
                    iv8 i01 = wmma_i8_k16(a0, b1, iv8_zero());
#elif INK_GFX11
                    iv4 b0 = wmma_ld_i8_k16_g11(&sWq[(wn * 32 + l16) * LDI + kk * 16]);
                    iv4 b1 = wmma_ld_i8_k16_g11(&sWq[(wn * 32 + 16 + l16) * LDI + kk * 16]);
                    iv4 a0 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + l16) * LDI + kk * 16]);
                    iv8 i00 = wmma_i8_k16_g11(a0, b0, iv8_zero());
                    iv8 i01 = wmma_i8_k16_g11(a0, b1, iv8_zero());
#endif
                    const float ws0 = sWs[(wn * 32 + col16) * 4 + kk];
                    const float ws1 = sWs[(wn * 32 + 16 + col16) * 4 + kk];
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd0 = sXd[(wm * 32 + wmma_acc_m(r, lane)) * 4 + kk];
                        acc[0][0][r] = fmaf(static_cast<float>(i00[r]), xd0 * ws0, acc[0][0][r]);
                        acc[0][1][r] = fmaf(static_cast<float>(i01[r]), xd0 * ws1, acc[0][1][r]);
                    }
                    if (do_m1) {
#if INK_GFX12
                        iv2 a1 = wmma_ld_i8_k16(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16], hi);
                        iv8 i10 = wmma_i8_k16(a1, b0, iv8_zero());
                        iv8 i11 = wmma_i8_k16(a1, b1, iv8_zero());
#elif INK_GFX11
                        iv4 a1 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16]);
                        iv8 i10 = wmma_i8_k16_g11(a1, b0, iv8_zero());
                        iv8 i11 = wmma_i8_k16_g11(a1, b1, iv8_zero());
#endif
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float xd1 = sXd[(wm * 32 + 16 + wmma_acc_m(r, lane)) * 4 + kk];
                            acc[1][0][r] = fmaf(static_cast<float>(i10[r]), xd1 * ws0, acc[1][0][r]);
                            acc[1][1][r] = fmaf(static_cast<float>(i11[r]), xd1 * ws1, acc[1][1][r]);
                        }
                    }
                }
            }
            sync_lds();
        }

        if (wm * 32 < tm) {
            const bool is_even = ((lane & 1) == 0);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int gn = n_blk + wn * 32 + ni * 16 + wmma_acc_n(0, lane);
                    const int gj = gn >> 1;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float g_val = acc[mi][ni][r];
                        const float u_val = __shfl_xor(g_val, 1, 32);
                        const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                        if (is_even && lm < tm && gj < inter) {
                            bucket_act[static_cast<size_t>(beg + t0 + lm) * inter + gj] =
                                static_cast<half_t>(situ_f32(g_val, u_val, beta, lb));
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)X_f16; (void)bucket_act; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

__global__ void __launch_bounds__(256) k_moe_grouped_w2_wmma_sf4_i8(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const half_t* __restrict__ bucket_act,
    float* __restrict__ bucket_y,
    int D, int inter) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 64, BN = 128, BK = 64, LDI = BK + 16;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) int8_t sXq[BM * LDI];
    __shared__ alignas(16) int8_t sWq[BN * LDI];
    __shared__ float sXd[BM * 4];
    __shared__ float sWs[BN * 4];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 2;
    const int wn = wave & 3;
    const int n_blk = blockIdx.x * BN;
    const float half_outer = 0.5f * (reinterpret_cast<const float*>(desc.body)[2] / reinterpret_cast<const float*>(desc.body)[3]);
    const uint8_t* sf4_p = desc.body + desc.dn_offset;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(D) * (inter >> 1);
    const int t_stride = gridDim.z * BM;

    const int xr = tid >> 2;
    const int xs = tid & 3;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc[2][2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < inter; k0 += BK) {
            if (xr < tm) {
                const half_t* xp = bucket_act + static_cast<size_t>(beg + t0 + xr) * inter + k0 + xs * 16;
                const uint4 u0 = *reinterpret_cast<const uint4*>(xp);
                const uint4 u1 = *reinterpret_cast<const uint4*>(xp + 8);
                float xf[16];
                h2_unpack(u0.x, xf[0], xf[1]);   h2_unpack(u0.y, xf[2], xf[3]);
                h2_unpack(u0.z, xf[4], xf[5]);   h2_unpack(u0.w, xf[6], xf[7]);
                h2_unpack(u1.x, xf[8], xf[9]);   h2_unpack(u1.y, xf[10], xf[11]);
                h2_unpack(u1.z, xf[12], xf[13]); h2_unpack(u1.w, xf[14], xf[15]);
                float am = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(xf[i]));
                const float d = am * (1.0f / 127.0f);
                const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
                int4 p4;
                p4.x = pack4_i8(clamp_i8(xf[0] * inv), clamp_i8(xf[1] * inv), clamp_i8(xf[2] * inv), clamp_i8(xf[3] * inv));
                p4.y = pack4_i8(clamp_i8(xf[4] * inv), clamp_i8(xf[5] * inv), clamp_i8(xf[6] * inv), clamp_i8(xf[7] * inv));
                p4.z = pack4_i8(clamp_i8(xf[8] * inv), clamp_i8(xf[9] * inv), clamp_i8(xf[10] * inv), clamp_i8(xf[11] * inv));
                p4.w = pack4_i8(clamp_i8(xf[12] * inv), clamp_i8(xf[13] * inv), clamp_i8(xf[14] * inv), clamp_i8(xf[15] * inv));
                *reinterpret_cast<int4*>(&sXq[xr * LDI + xs * 16]) = p4;
                sXd[xr * 4 + xs] = d;
            } else {
                *reinterpret_cast<int4*>(&sXq[xr * LDI + xs * 16]) = make_int4(0, 0, 0, 0);
                sXd[xr * 4 + xs] = 0.0f;
            }

#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, b4 = idx & 3;
                const int gn = n_blk + r;
                const int gb = (k0 >> 4) + b4;
                if (gn < D) {
                    sWs[r * 4 + b4] = e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * (inter >> 4) + gb]) * half_outer;
                    const uint2 raw = *reinterpret_cast<const uint2*>(sf4_p + static_cast<size_t>(gn) * (inter >> 1) + gb * 8);
                    int4 p4;
                    p4.x = pack4_i8(sf4_e2m1_i8_x2(raw.x & 15u), sf4_e2m1_i8_x2((raw.x >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 8) & 15u), sf4_e2m1_i8_x2((raw.x >> 12) & 15u));
                    p4.y = pack4_i8(sf4_e2m1_i8_x2((raw.x >> 16) & 15u), sf4_e2m1_i8_x2((raw.x >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 24) & 15u), sf4_e2m1_i8_x2((raw.x >> 28) & 15u));
                    p4.z = pack4_i8(sf4_e2m1_i8_x2(raw.y & 15u), sf4_e2m1_i8_x2((raw.y >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 8) & 15u), sf4_e2m1_i8_x2((raw.y >> 12) & 15u));
                    p4.w = pack4_i8(sf4_e2m1_i8_x2((raw.y >> 16) & 15u), sf4_e2m1_i8_x2((raw.y >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 24) & 15u), sf4_e2m1_i8_x2((raw.y >> 28) & 15u));
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = p4;
                } else {
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = make_int4(0, 0, 0, 0);
                    sWs[r * 4 + b4] = 0.0f;
                }
            }
            sync_lds();

            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
                const int col16 = wmma_acc_n(0, lane);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
#if INK_GFX12
                    iv2 b0 = wmma_ld_i8_k16(&sWq[(wn * 32 + l16) * LDI + kk * 16], hi);
                    iv2 b1 = wmma_ld_i8_k16(&sWq[(wn * 32 + 16 + l16) * LDI + kk * 16], hi);
                    iv2 a0 = wmma_ld_i8_k16(&sXq[(wm * 32 + l16) * LDI + kk * 16], hi);
                    iv8 i00 = wmma_i8_k16(a0, b0, iv8_zero());
                    iv8 i01 = wmma_i8_k16(a0, b1, iv8_zero());
#elif INK_GFX11
                    iv4 b0 = wmma_ld_i8_k16_g11(&sWq[(wn * 32 + l16) * LDI + kk * 16]);
                    iv4 b1 = wmma_ld_i8_k16_g11(&sWq[(wn * 32 + 16 + l16) * LDI + kk * 16]);
                    iv4 a0 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + l16) * LDI + kk * 16]);
                    iv8 i00 = wmma_i8_k16_g11(a0, b0, iv8_zero());
                    iv8 i01 = wmma_i8_k16_g11(a0, b1, iv8_zero());
#endif
                    const float ws0 = sWs[(wn * 32 + col16) * 4 + kk];
                    const float ws1 = sWs[(wn * 32 + 16 + col16) * 4 + kk];
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd0 = sXd[(wm * 32 + wmma_acc_m(r, lane)) * 4 + kk];
                        acc[0][0][r] = fmaf(static_cast<float>(i00[r]), xd0 * ws0, acc[0][0][r]);
                        acc[0][1][r] = fmaf(static_cast<float>(i01[r]), xd0 * ws1, acc[0][1][r]);
                    }
                    if (do_m1) {
#if INK_GFX12
                        iv2 a1 = wmma_ld_i8_k16(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16], hi);
                        iv8 i10 = wmma_i8_k16(a1, b0, iv8_zero());
                        iv8 i11 = wmma_i8_k16(a1, b1, iv8_zero());
#elif INK_GFX11
                        iv4 a1 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16]);
                        iv8 i10 = wmma_i8_k16_g11(a1, b0, iv8_zero());
                        iv8 i11 = wmma_i8_k16_g11(a1, b1, iv8_zero());
#endif
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float xd1 = sXd[(wm * 32 + 16 + wmma_acc_m(r, lane)) * 4 + kk];
                            acc[1][0][r] = fmaf(static_cast<float>(i10[r]), xd1 * ws0, acc[1][0][r]);
                            acc[1][1][r] = fmaf(static_cast<float>(i11[r]), xd1 * ws1, acc[1][1][r]);
                        }
                    }
                }
            }
            sync_lds();
        }

        if (wm * 32 < tm) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int gn = n_blk + wn * 32 + ni * 16 + wmma_acc_n(0, lane);
                    if (gn < D) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                            if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[mi][ni][r];
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)bucket_act; (void)bucket_y; (void)D; (void)inter;
#endif
}

// =================================================================================================
// Wave-5 Pre-Quantized Activation Q8 Helpers + BM=128, BN=128 Zero-Redundant-Dequant Grouped-MoE!
//   - Pre-quantizes X ONCE per MoE stage (eliminating 144x redundant FP16->FP32->Q8 quant inside GEMM!)
//   - BM=128, BN=128, m_tiles=1: unpacks W STRICTLY ONCE for 95%+ of experts at B=2048!
//   - 4x2 waves (2x4 = 8 WMMAs/wave per kk), 20.5-26.0 KB LDS (< 32 KB -> 2-3 WG/CU)!
// =================================================================================================

// Quantize FP32 [M, K] -> Q8_64 (per-64-element int8 + float scale). Grid: (K / 64, M), Block: 64 (1 elem/thr? No: 16 thr x float4 -> Grid: ((M*(K/64) + 15)/16,), Block: 256!)
__global__ void __launch_bounds__(256) k_quant_f32_to_q8_64(
    const float* __restrict__ src,
    int8_t* __restrict__ dst_q,
    float* __restrict__ dst_d,
    int n_blks64) {
    const int blk = blockIdx.x * 16 + (threadIdx.x >> 4);  // 16 blocks of 64 per 256-thread WG
    const int sub = threadIdx.x & 15;                       // 16 threads x 4 floats = 64 floats
    if (blk >= n_blks64) return;
    const float4 v = *reinterpret_cast<const float4*>(src + static_cast<size_t>(blk) * 64 + sub * 4);
    float am = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    am = fmaxf(am, __shfl_xor(am, 1, 32));
    am = fmaxf(am, __shfl_xor(am, 2, 32));
    am = fmaxf(am, __shfl_xor(am, 4, 32));
    am = fmaxf(am, __shfl_xor(am, 8, 32));
    const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
    *reinterpret_cast<int32_t*>(dst_q + static_cast<size_t>(blk) * 64 + sub * 4) =
        pack4_i8(clamp_i8(v.x * inv), clamp_i8(v.y * inv), clamp_i8(v.z * inv), clamp_i8(v.w * inv));
    if (sub == 0) dst_d[blk] = am * (1.0f / 127.0f);
}

__global__ void __launch_bounds__(256) k_quant_f16_to_q8_64(
    const half_t* __restrict__ src,
    int8_t* __restrict__ dst_q,
    float* __restrict__ dst_d,
    int n_blks64) {
    const int blk = blockIdx.x * 16 + (threadIdx.x >> 4);
    const int sub = threadIdx.x & 15;
    if (blk >= n_blks64) return;
    const uint2 u = *reinterpret_cast<const uint2*>(src + static_cast<size_t>(blk) * 64 + sub * 4);
    float f0, f1, f2, f3;
    h2_unpack(u.x, f0, f1);
    h2_unpack(u.y, f2, f3);
    float am = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
    am = fmaxf(am, __shfl_xor(am, 1, 32));
    am = fmaxf(am, __shfl_xor(am, 2, 32));
    am = fmaxf(am, __shfl_xor(am, 4, 32));
    am = fmaxf(am, __shfl_xor(am, 8, 32));
    const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
    *reinterpret_cast<int32_t*>(dst_q + static_cast<size_t>(blk) * 64 + sub * 4) =
        pack4_i8(clamp_i8(f0 * inv), clamp_i8(f1 * inv), clamp_i8(f2 * inv), clamp_i8(f3 * inv));
    if (sub == 0) dst_d[blk] = am * (1.0f / 127.0f);
}

__global__ void __launch_bounds__(256) k_quant_f32_to_q8_16(
    const float* __restrict__ src,
    int8_t* __restrict__ dst_q,
    float* __restrict__ dst_d,
    int n_blks16) {
    const int blk = blockIdx.x * 64 + (threadIdx.x >> 2);  // 64 blocks of 16 per 256-thread WG
    const int sub = threadIdx.x & 3;                        // 4 threads x 4 floats = 16 floats
    if (blk >= n_blks16) return;
    const float4 v = *reinterpret_cast<const float4*>(src + static_cast<size_t>(blk) * 16 + sub * 4);
    float am = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    am = fmaxf(am, __shfl_xor(am, 1, 32));
    am = fmaxf(am, __shfl_xor(am, 2, 32));
    const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
    *reinterpret_cast<int32_t*>(dst_q + static_cast<size_t>(blk) * 16 + sub * 4) =
        pack4_i8(clamp_i8(v.x * inv), clamp_i8(v.y * inv), clamp_i8(v.z * inv), clamp_i8(v.w * inv));
    if (sub == 0) dst_d[blk] = am * (1.0f / 127.0f);
}

__global__ void __launch_bounds__(256) k_quant_f16_to_q8_16(
    const half_t* __restrict__ src,
    int8_t* __restrict__ dst_q,
    float* __restrict__ dst_d,
    int n_blks16) {
    const int blk = blockIdx.x * 64 + (threadIdx.x >> 2);
    const int sub = threadIdx.x & 3;
    if (blk >= n_blks16) return;
    const uint2 u = *reinterpret_cast<const uint2*>(src + static_cast<size_t>(blk) * 16 + sub * 4);
    float f0, f1, f2, f3;
    h2_unpack(u.x, f0, f1);
    h2_unpack(u.y, f2, f3);
    float am = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
    am = fmaxf(am, __shfl_xor(am, 1, 32));
    am = fmaxf(am, __shfl_xor(am, 2, 32));
    const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
    *reinterpret_cast<int32_t*>(dst_q + static_cast<size_t>(blk) * 16 + sub * 4) =
        pack4_i8(clamp_i8(f0 * inv), clamp_i8(f1 * inv), clamp_i8(f2 * inv), clamp_i8(f3 * inv));
    if (sub == 0) dst_d[blk] = am * (1.0f / 127.0f);
}

// BM=128, BN=128, BK=64 Pre-Q8 S_K 2x-Rate INT8 WMMA Grouped-MoE w13 + SITU (22 KB LDS, parallel iacc[2][4], m_tiles=8!)
__global__ void __launch_bounds__(256) k_moe_grouped_w13_situ_wmma_sk_i8_128(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int32_t* __restrict__ bucket_tok,
    const int8_t* __restrict__ X_q8,      // [B, D] pre-quantized int8
    const float* __restrict__ X_d64,      // [B, D/64] float scales
    half_t* __restrict__ bucket_act,      // [B * top_k, inter]
    int D, int inter, float beta, float lb) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 128, BN = 128, BK = 64, LDI = BK + 16;  // 80 B/row
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) int8_t sXq[BM * LDI];   // 10,240 B
    __shared__ alignas(16) int8_t sWq[BN * LDI];   // 10,240 B
    __shared__ float sXd[BM];                      //    512 B
    __shared__ float sWs[BN];                      //    512 B
    __shared__ int32_t sTok[BM];                   //    512 B -> 22.0 KB

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 1;  // 0..3 (32 M rows/wave)
    const int wn = wave & 1;   // 0..1 (64 N cols/wave)
    const int N = 2 * inter;
    const int n_blk = blockIdx.x * BN;
    const uint8_t* sk_w = desc.body + static_cast<size_t>(inter) * 2;
    const int nsb = D >> 8;
    const int nblk64 = D >> 6;
    const int code = desc.gu_code;
    const int sb_b = sk_sb_bytes(code);
    const half_t* sq_xd = reinterpret_cast<const half_t*>(desc.body);
    const int t_stride = gridDim.z * BM;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        if (tid < BM) sTok[tid] = (tid < tm) ? bucket_tok[beg + t0 + tid] : 0;
        sync_lds();

        fv8 acc[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < D; k0 += BK) {
            const int kb64 = k0 >> 6;
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, q16 = idx & 3;
                if (r < tm) {
                    const int tok = sTok[r];
                    *reinterpret_cast<int4*>(&sXq[r * LDI + q16 * 16]) =
                        *reinterpret_cast<const int4*>(X_q8 + static_cast<size_t>(tok) * D + k0 + q16 * 16);
                    if (q16 == 0) sXd[r] = X_d64[static_cast<size_t>(tok) * nblk64 + kb64];
                } else {
                    *reinterpret_cast<int4*>(&sXq[r * LDI + q16 * 16]) = make_int4(0, 0, 0, 0);
                    if (q16 == 0) sXd[r] = 0.0f;
                }
            }
            const int s = k0 >> 8;
            const int g = (k0 >> 6) & 3;
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, q16 = idx & 3;
                const int gn = n_blk + r;
                if (gn < N) {
                    const uint8_t* blk = sk_w + (static_cast<size_t>(gn) * nsb + s) * sb_b;
                    *reinterpret_cast<int4*>(&sWq[r * LDI + q16 * 16]) = decode_sk_16_i8(blk + 6, code, (g << 6) + q16 * 16);
                    if (q16 == 0) sWs[r] = decode_sk_sub_eff_i8(blk, code, g);
                } else {
                    *reinterpret_cast<int4*>(&sWq[r * LDI + q16 * 16]) = make_int4(0, 0, 0, 0);
                    if (q16 == 0) sWs[r] = 0.0f;
                }
            }
            sync_lds();

            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
                iv8 iacc[2][4];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni) iacc[mi][ni] = iv8_zero();
#if INK_GFX12
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    iv4 b0 = wmma_ld_i8_k32(&sWq[(wn * 64 +  0 + l16) * LDI + kk * 32], hi);
                    iv4 b1 = wmma_ld_i8_k32(&sWq[(wn * 64 + 16 + l16) * LDI + kk * 32], hi);
                    iv4 b2 = wmma_ld_i8_k32(&sWq[(wn * 64 + 32 + l16) * LDI + kk * 32], hi);
                    iv4 b3 = wmma_ld_i8_k32(&sWq[(wn * 64 + 48 + l16) * LDI + kk * 32], hi);
                    iv4 a0 = wmma_ld_i8_k32(&sXq[(wm * 32 + l16) * LDI + kk * 32], hi);
                    iacc[0][0] = wmma_i8_k32(a0, b0, iacc[0][0]);
                    iacc[0][1] = wmma_i8_k32(a0, b1, iacc[0][1]);
                    iacc[0][2] = wmma_i8_k32(a0, b2, iacc[0][2]);
                    iacc[0][3] = wmma_i8_k32(a0, b3, iacc[0][3]);
                    if (do_m1) {
                        iv4 a1 = wmma_ld_i8_k32(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 32], hi);
                        iacc[1][0] = wmma_i8_k32(a1, b0, iacc[1][0]);
                        iacc[1][1] = wmma_i8_k32(a1, b1, iacc[1][1]);
                        iacc[1][2] = wmma_i8_k32(a1, b2, iacc[1][2]);
                        iacc[1][3] = wmma_i8_k32(a1, b3, iacc[1][3]);
                    }
                }
#elif INK_GFX11
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    iv4 b0 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 +  0 + l16) * LDI + kk * 16]);
                    iv4 b1 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 16 + l16) * LDI + kk * 16]);
                    iv4 b2 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 32 + l16) * LDI + kk * 16]);
                    iv4 b3 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 48 + l16) * LDI + kk * 16]);
                    iv4 a0 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + l16) * LDI + kk * 16]);
                    iacc[0][0] = wmma_i8_k16_g11(a0, b0, iacc[0][0]);
                    iacc[0][1] = wmma_i8_k16_g11(a0, b1, iacc[0][1]);
                    iacc[0][2] = wmma_i8_k16_g11(a0, b2, iacc[0][2]);
                    iacc[0][3] = wmma_i8_k16_g11(a0, b3, iacc[0][3]);
                    if (do_m1) {
                        iv4 a1 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16]);
                        iacc[1][0] = wmma_i8_k16_g11(a1, b0, iacc[1][0]);
                        iacc[1][1] = wmma_i8_k16_g11(a1, b1, iacc[1][1]);
                        iacc[1][2] = wmma_i8_k16_g11(a1, b2, iacc[1][2]);
                        iacc[1][3] = wmma_i8_k16_g11(a1, b3, iacc[1][3]);
                    }
                }
#endif
                const int col16 = wmma_acc_n(0, lane);
                const float ws0 = sWs[wn * 64 +  0 + col16];
                const float ws1 = sWs[wn * 64 + 16 + col16];
                const float ws2 = sWs[wn * 64 + 32 + col16];
                const float ws3 = sWs[wn * 64 + 48 + col16];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float xd0 = sXd[wm * 32 + wmma_acc_m(r, lane)];
                    acc[0][0][r] = fmaf(static_cast<float>(iacc[0][0][r]), xd0 * ws0, acc[0][0][r]);
                    acc[0][1][r] = fmaf(static_cast<float>(iacc[0][1][r]), xd0 * ws1, acc[0][1][r]);
                    acc[0][2][r] = fmaf(static_cast<float>(iacc[0][2][r]), xd0 * ws2, acc[0][2][r]);
                    acc[0][3][r] = fmaf(static_cast<float>(iacc[0][3][r]), xd0 * ws3, acc[0][3][r]);
                }
                if (do_m1) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd1 = sXd[wm * 32 + 16 + wmma_acc_m(r, lane)];
                        acc[1][0][r] = fmaf(static_cast<float>(iacc[1][0][r]), xd1 * ws0, acc[1][0][r]);
                        acc[1][1][r] = fmaf(static_cast<float>(iacc[1][1][r]), xd1 * ws1, acc[1][1][r]);
                        acc[1][2][r] = fmaf(static_cast<float>(iacc[1][2][r]), xd1 * ws2, acc[1][2][r]);
                        acc[1][3][r] = fmaf(static_cast<float>(iacc[1][3][r]), xd1 * ws3, acc[1][3][r]);
                    }
                }
            }
            sync_lds();
        }

        if (wm * 32 < tm) {
            const bool is_even = ((lane & 1) == 0);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int gn = n_blk + wn * 64 + ni * 16 + wmma_acc_n(0, lane);
                    const int gj = gn >> 1;
                    const float inv_sq = (is_even && gj < inter) ? (1.0f / static_cast<float>(sq_xd[gj])) : 1.0f;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float g_val = acc[mi][ni][r];
                        const float u_val = __shfl_xor(g_val, 1, 32);
                        const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                        if (is_even && lm < tm && gj < inter) {
                            bucket_act[static_cast<size_t>(beg + t0 + lm) * inter + gj] =
                                static_cast<half_t>(situ_f32(g_val, u_val, beta, lb) * inv_sq);
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)X_q8; (void)X_d64; (void)bucket_act; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

// BM=128, BN=128, BK=64 Pre-Q8 S_K 2x-Rate INT8 WMMA Grouped-MoE w2 (21.5 KB LDS, parallel iacc[2][4], m_tiles=8!)
__global__ void __launch_bounds__(256) k_moe_grouped_w2_wmma_sk_i8_128(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int8_t* __restrict__ act_q8,    // [B * top_k, inter]
    const float* __restrict__ act_d64,    // [B * top_k, inter/64]
    float* __restrict__ bucket_y,         // [B * top_k, D]
    int D, int inter) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 128, BN = 128, BK = 64, LDI = BK + 16;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) int8_t sXq[BM * LDI];
    __shared__ alignas(16) int8_t sWq[BN * LDI];
    __shared__ float sXd[BM];
    __shared__ float sWs[BN];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 1;
    const int wn = wave & 1;
    const int n_blk = blockIdx.x * BN;
    const uint8_t* sk_w = desc.body + desc.dn_offset;
    const int code = desc.dn_code;
    const int sb_per_pair = (2 * inter) >> 8;
    const int sb_b = sk_sb_bytes(code);
    const int sub_per_row = inter >> 6;
    const int t_stride = gridDim.z * BM;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < inter; k0 += BK) {
            const int sg = k0 >> 6;
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, q16 = idx & 3;
                if (r < tm) {
                    const size_t row_i = static_cast<size_t>(beg + t0 + r);
                    *reinterpret_cast<int4*>(&sXq[r * LDI + q16 * 16]) =
                        *reinterpret_cast<const int4*>(act_q8 + row_i * inter + k0 + q16 * 16);
                    if (q16 == 0) sXd[r] = act_d64[row_i * sub_per_row + sg];
                } else {
                    *reinterpret_cast<int4*>(&sXq[r * LDI + q16 * 16]) = make_int4(0, 0, 0, 0);
                    if (q16 == 0) sXd[r] = 0.0f;
                }
            }
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, q16 = idx & 3;
                const int gn = n_blk + r;
                if (gn < D) {
                    const int pair = gn >> 1;
                    const int is_odd = gn & 1;
                    const int sub_global = is_odd * sub_per_row + sg;
                    const int s = sub_global >> 2;
                    const int g = sub_global & 3;
                    const uint8_t* blk = sk_w + (static_cast<size_t>(pair) * sb_per_pair + s) * sb_b;
                    *reinterpret_cast<int4*>(&sWq[r * LDI + q16 * 16]) = decode_sk_16_i8(blk + 6, code, (g << 6) + q16 * 16);
                    if (q16 == 0) sWs[r] = decode_sk_sub_eff_i8(blk, code, g);
                } else {
                    *reinterpret_cast<int4*>(&sWq[r * LDI + q16 * 16]) = make_int4(0, 0, 0, 0);
                    if (q16 == 0) sWs[r] = 0.0f;
                }
            }
            sync_lds();

            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
                iv8 iacc[2][4];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni) iacc[mi][ni] = iv8_zero();
#if INK_GFX12
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    iv4 b0 = wmma_ld_i8_k32(&sWq[(wn * 64 +  0 + l16) * LDI + kk * 32], hi);
                    iv4 b1 = wmma_ld_i8_k32(&sWq[(wn * 64 + 16 + l16) * LDI + kk * 32], hi);
                    iv4 b2 = wmma_ld_i8_k32(&sWq[(wn * 64 + 32 + l16) * LDI + kk * 32], hi);
                    iv4 b3 = wmma_ld_i8_k32(&sWq[(wn * 64 + 48 + l16) * LDI + kk * 32], hi);
                    iv4 a0 = wmma_ld_i8_k32(&sXq[(wm * 32 + l16) * LDI + kk * 32], hi);
                    iacc[0][0] = wmma_i8_k32(a0, b0, iacc[0][0]);
                    iacc[0][1] = wmma_i8_k32(a0, b1, iacc[0][1]);
                    iacc[0][2] = wmma_i8_k32(a0, b2, iacc[0][2]);
                    iacc[0][3] = wmma_i8_k32(a0, b3, iacc[0][3]);
                    if (do_m1) {
                        iv4 a1 = wmma_ld_i8_k32(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 32], hi);
                        iacc[1][0] = wmma_i8_k32(a1, b0, iacc[1][0]);
                        iacc[1][1] = wmma_i8_k32(a1, b1, iacc[1][1]);
                        iacc[1][2] = wmma_i8_k32(a1, b2, iacc[1][2]);
                        iacc[1][3] = wmma_i8_k32(a1, b3, iacc[1][3]);
                    }
                }
#elif INK_GFX11
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    iv4 b0 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 +  0 + l16) * LDI + kk * 16]);
                    iv4 b1 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 16 + l16) * LDI + kk * 16]);
                    iv4 b2 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 32 + l16) * LDI + kk * 16]);
                    iv4 b3 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 48 + l16) * LDI + kk * 16]);
                    iv4 a0 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + l16) * LDI + kk * 16]);
                    iacc[0][0] = wmma_i8_k16_g11(a0, b0, iacc[0][0]);
                    iacc[0][1] = wmma_i8_k16_g11(a0, b1, iacc[0][1]);
                    iacc[0][2] = wmma_i8_k16_g11(a0, b2, iacc[0][2]);
                    iacc[0][3] = wmma_i8_k16_g11(a0, b3, iacc[0][3]);
                    if (do_m1) {
                        iv4 a1 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16]);
                        iacc[1][0] = wmma_i8_k16_g11(a1, b0, iacc[1][0]);
                        iacc[1][1] = wmma_i8_k16_g11(a1, b1, iacc[1][1]);
                        iacc[1][2] = wmma_i8_k16_g11(a1, b2, iacc[1][2]);
                        iacc[1][3] = wmma_i8_k16_g11(a1, b3, iacc[1][3]);
                    }
                }
#endif
                const int col16 = wmma_acc_n(0, lane);
                const float ws0 = sWs[wn * 64 +  0 + col16];
                const float ws1 = sWs[wn * 64 + 16 + col16];
                const float ws2 = sWs[wn * 64 + 32 + col16];
                const float ws3 = sWs[wn * 64 + 48 + col16];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float xd0 = sXd[wm * 32 + wmma_acc_m(r, lane)];
                    acc[0][0][r] = fmaf(static_cast<float>(iacc[0][0][r]), xd0 * ws0, acc[0][0][r]);
                    acc[0][1][r] = fmaf(static_cast<float>(iacc[0][1][r]), xd0 * ws1, acc[0][1][r]);
                    acc[0][2][r] = fmaf(static_cast<float>(iacc[0][2][r]), xd0 * ws2, acc[0][2][r]);
                    acc[0][3][r] = fmaf(static_cast<float>(iacc[0][3][r]), xd0 * ws3, acc[0][3][r]);
                }
                if (do_m1) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd1 = sXd[wm * 32 + 16 + wmma_acc_m(r, lane)];
                        acc[1][0][r] = fmaf(static_cast<float>(iacc[1][0][r]), xd1 * ws0, acc[1][0][r]);
                        acc[1][1][r] = fmaf(static_cast<float>(iacc[1][1][r]), xd1 * ws1, acc[1][1][r]);
                        acc[1][2][r] = fmaf(static_cast<float>(iacc[1][2][r]), xd1 * ws2, acc[1][2][r]);
                        acc[1][3][r] = fmaf(static_cast<float>(iacc[1][3][r]), xd1 * ws3, acc[1][3][r]);
                    }
                }
            }
            sync_lds();
        }

        if (wm * 32 < tm) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int gn = n_blk + wn * 64 + ni * 16 + wmma_acc_n(0, lane);
                    if (gn < D) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                            if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[mi][ni][r];
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)act_q8; (void)act_d64; (void)bucket_y; (void)D; (void)inter;
#endif
}

// BM=128, BN=128, BK=64 Pre-Q8 SF4 2x-Rate INT8 WMMA Grouped-MoE w13 + SITU (26.5 KB LDS < 32 KB -> 2 WG/CU!)
__global__ void __launch_bounds__(256) k_moe_grouped_w13_situ_wmma_sf4_i8_128(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int32_t* __restrict__ bucket_tok,
    const int8_t* __restrict__ X_q8,      // [B, D] pre-quantized int8 (per-16)
    const float* __restrict__ X_d16,      // [B, D/16] float scales
    half_t* __restrict__ bucket_act,      // [B * top_k, inter]
    int D, int inter, float beta, float lb) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 128, BN = 128, BK = 64, LDI = BK + 16;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) int8_t sXq[BM * LDI];   // 10,240 B
    __shared__ alignas(16) int8_t sWq[BN * LDI];   // 10,240 B
    __shared__ float sXd[BM * 4];                  //  2,048 B
    __shared__ float sWs[BN * 4];                  //  2,048 B -> 24,576 B
    __shared__ int32_t sTok[BM];                   //    512 B -> 25.0 KB (< 32 KB -> 2 WG/CU!)

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 1;
    const int wn = wave & 1;
    const int N = 2 * inter;
    const int n_blk = blockIdx.x * BN;
    const float half_outer = 0.5f * (reinterpret_cast<const float*>(desc.body)[0] / reinterpret_cast<const float*>(desc.body)[1]);
    const uint8_t* sf4_p = desc.body + 16;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(N) * (D >> 1);
    const int nblk16 = D >> 4;
    const int t_stride = gridDim.z * BM;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        if (tid < BM) sTok[tid] = (tid < tm) ? bucket_tok[beg + t0 + tid] : 0;
        sync_lds();

        fv8 acc[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < D; k0 += BK) {
            const int kb16 = k0 >> 4;
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, b4 = idx & 3;
                if (r < tm) {
                    const int tok = sTok[r];
                    *reinterpret_cast<int4*>(&sXq[r * LDI + b4 * 16]) =
                        *reinterpret_cast<const int4*>(X_q8 + static_cast<size_t>(tok) * D + k0 + b4 * 16);
                    sXd[r * 4 + b4] = X_d16[static_cast<size_t>(tok) * nblk16 + kb16 + b4];
                } else {
                    *reinterpret_cast<int4*>(&sXq[r * LDI + b4 * 16]) = make_int4(0, 0, 0, 0);
                    sXd[r * 4 + b4] = 0.0f;
                }
            }
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, b4 = idx & 3;
                const int gn = n_blk + r;
                const int gb = kb16 + b4;
                if (gn < N) {
                    sWs[r * 4 + b4] = e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * nblk16 + gb]) * half_outer;
                    const uint2 raw = *reinterpret_cast<const uint2*>(sf4_p + static_cast<size_t>(gn) * (D >> 1) + gb * 8);
                    int4 p4;
                    p4.x = pack4_i8(sf4_e2m1_i8_x2(raw.x & 15u), sf4_e2m1_i8_x2((raw.x >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 8) & 15u), sf4_e2m1_i8_x2((raw.x >> 12) & 15u));
                    p4.y = pack4_i8(sf4_e2m1_i8_x2((raw.x >> 16) & 15u), sf4_e2m1_i8_x2((raw.x >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 24) & 15u), sf4_e2m1_i8_x2((raw.x >> 28) & 15u));
                    p4.z = pack4_i8(sf4_e2m1_i8_x2(raw.y & 15u), sf4_e2m1_i8_x2((raw.y >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 8) & 15u), sf4_e2m1_i8_x2((raw.y >> 12) & 15u));
                    p4.w = pack4_i8(sf4_e2m1_i8_x2((raw.y >> 16) & 15u), sf4_e2m1_i8_x2((raw.y >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 24) & 15u), sf4_e2m1_i8_x2((raw.y >> 28) & 15u));
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = p4;
                } else {
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = make_int4(0, 0, 0, 0);
                    sWs[r * 4 + b4] = 0.0f;
                }
            }
            sync_lds();

            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
                const int col16 = wmma_acc_n(0, lane);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
#if INK_GFX12
                    iv2 b0 = wmma_ld_i8_k16(&sWq[(wn * 64 +  0 + l16) * LDI + kk * 16], hi);
                    iv2 b1 = wmma_ld_i8_k16(&sWq[(wn * 64 + 16 + l16) * LDI + kk * 16], hi);
                    iv2 b2 = wmma_ld_i8_k16(&sWq[(wn * 64 + 32 + l16) * LDI + kk * 16], hi);
                    iv2 b3 = wmma_ld_i8_k16(&sWq[(wn * 64 + 48 + l16) * LDI + kk * 16], hi);
                    iv2 a0 = wmma_ld_i8_k16(&sXq[(wm * 32 + l16) * LDI + kk * 16], hi);
                    iv8 i00 = wmma_i8_k16(a0, b0, iv8_zero());
                    iv8 i01 = wmma_i8_k16(a0, b1, iv8_zero());
                    iv8 i02 = wmma_i8_k16(a0, b2, iv8_zero());
                    iv8 i03 = wmma_i8_k16(a0, b3, iv8_zero());
#elif INK_GFX11
                    iv4 b0 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 +  0 + l16) * LDI + kk * 16]);
                    iv4 b1 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 16 + l16) * LDI + kk * 16]);
                    iv4 b2 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 32 + l16) * LDI + kk * 16]);
                    iv4 b3 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 48 + l16) * LDI + kk * 16]);
                    iv4 a0 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + l16) * LDI + kk * 16]);
                    iv8 i00 = wmma_i8_k16_g11(a0, b0, iv8_zero());
                    iv8 i01 = wmma_i8_k16_g11(a0, b1, iv8_zero());
                    iv8 i02 = wmma_i8_k16_g11(a0, b2, iv8_zero());
                    iv8 i03 = wmma_i8_k16_g11(a0, b3, iv8_zero());
#endif
                    const float ws0 = sWs[(wn * 64 +  0 + col16) * 4 + kk];
                    const float ws1 = sWs[(wn * 64 + 16 + col16) * 4 + kk];
                    const float ws2 = sWs[(wn * 64 + 32 + col16) * 4 + kk];
                    const float ws3 = sWs[(wn * 64 + 48 + col16) * 4 + kk];
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd0 = sXd[(wm * 32 + wmma_acc_m(r, lane)) * 4 + kk];
                        acc[0][0][r] = fmaf(static_cast<float>(i00[r]), xd0 * ws0, acc[0][0][r]);
                        acc[0][1][r] = fmaf(static_cast<float>(i01[r]), xd0 * ws1, acc[0][1][r]);
                        acc[0][2][r] = fmaf(static_cast<float>(i02[r]), xd0 * ws2, acc[0][2][r]);
                        acc[0][3][r] = fmaf(static_cast<float>(i03[r]), xd0 * ws3, acc[0][3][r]);
                    }
                    if (do_m1) {
#if INK_GFX12
                        iv2 a1 = wmma_ld_i8_k16(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16], hi);
                        iv8 i10 = wmma_i8_k16(a1, b0, iv8_zero());
                        iv8 i11 = wmma_i8_k16(a1, b1, iv8_zero());
                        iv8 i12 = wmma_i8_k16(a1, b2, iv8_zero());
                        iv8 i13 = wmma_i8_k16(a1, b3, iv8_zero());
#elif INK_GFX11
                        iv4 a1 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16]);
                        iv8 i10 = wmma_i8_k16_g11(a1, b0, iv8_zero());
                        iv8 i11 = wmma_i8_k16_g11(a1, b1, iv8_zero());
                        iv8 i12 = wmma_i8_k16_g11(a1, b2, iv8_zero());
                        iv8 i13 = wmma_i8_k16_g11(a1, b3, iv8_zero());
#endif
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float xd1 = sXd[(wm * 32 + 16 + wmma_acc_m(r, lane)) * 4 + kk];
                            acc[1][0][r] = fmaf(static_cast<float>(i10[r]), xd1 * ws0, acc[1][0][r]);
                            acc[1][1][r] = fmaf(static_cast<float>(i11[r]), xd1 * ws1, acc[1][1][r]);
                            acc[1][2][r] = fmaf(static_cast<float>(i12[r]), xd1 * ws2, acc[1][2][r]);
                            acc[1][3][r] = fmaf(static_cast<float>(i13[r]), xd1 * ws3, acc[1][3][r]);
                        }
                    }
                }
            }
            sync_lds();
        }

        if (wm * 32 < tm) {
            const bool is_even = ((lane & 1) == 0);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int gn = n_blk + wn * 64 + ni * 16 + wmma_acc_n(0, lane);
                    const int gj = gn >> 1;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float g_val = acc[mi][ni][r];
                        const float u_val = __shfl_xor(g_val, 1, 32);
                        const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                        if (is_even && lm < tm && gj < inter) {
                            bucket_act[static_cast<size_t>(beg + t0 + lm) * inter + gj] =
                                static_cast<half_t>(situ_f32(g_val, u_val, beta, lb));
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)X_q8; (void)X_d16; (void)bucket_act; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

// BM=128, BN=128, BK=64 Pre-Q8 SF4 2x-Rate INT8 WMMA Grouped-MoE w2 (24.5 KB LDS < 32 KB -> 2 WG/CU!)
__global__ void __launch_bounds__(256) k_moe_grouped_w2_wmma_sf4_i8_128(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int8_t* __restrict__ act_q8,
    const float* __restrict__ act_d16,
    float* __restrict__ bucket_y,
    int D, int inter) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 128, BN = 128, BK = 64, LDI = BK + 16;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) int8_t sXq[BM * LDI];
    __shared__ alignas(16) int8_t sWq[BN * LDI];
    __shared__ float sXd[BM * 4];
    __shared__ float sWs[BN * 4];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 1;
    const int wn = wave & 1;
    const int n_blk = blockIdx.x * BN;
    const float half_outer = 0.5f * (reinterpret_cast<const float*>(desc.body)[2] / reinterpret_cast<const float*>(desc.body)[3]);
    const uint8_t* sf4_p = desc.body + desc.dn_offset;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(D) * (inter >> 1);
    const int nblk16 = inter >> 4;
    const int t_stride = gridDim.z * BM;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < inter; k0 += BK) {
            const int kb16 = k0 >> 4;
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, b4 = idx & 3;
                if (r < tm) {
                    const size_t row_i = static_cast<size_t>(beg + t0 + r);
                    *reinterpret_cast<int4*>(&sXq[r * LDI + b4 * 16]) =
                        *reinterpret_cast<const int4*>(act_q8 + row_i * inter + k0 + b4 * 16);
                    sXd[r * 4 + b4] = act_d16[row_i * nblk16 + kb16 + b4];
                } else {
                    *reinterpret_cast<int4*>(&sXq[r * LDI + b4 * 16]) = make_int4(0, 0, 0, 0);
                    sXd[r * 4 + b4] = 0.0f;
                }
            }
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, b4 = idx & 3;
                const int gn = n_blk + r;
                const int gb = kb16 + b4;
                if (gn < D) {
                    sWs[r * 4 + b4] = e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * nblk16 + gb]) * half_outer;
                    const uint2 raw = *reinterpret_cast<const uint2*>(sf4_p + static_cast<size_t>(gn) * (inter >> 1) + gb * 8);
                    int4 p4;
                    p4.x = pack4_i8(sf4_e2m1_i8_x2(raw.x & 15u), sf4_e2m1_i8_x2((raw.x >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 8) & 15u), sf4_e2m1_i8_x2((raw.x >> 12) & 15u));
                    p4.y = pack4_i8(sf4_e2m1_i8_x2((raw.x >> 16) & 15u), sf4_e2m1_i8_x2((raw.x >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 24) & 15u), sf4_e2m1_i8_x2((raw.x >> 28) & 15u));
                    p4.z = pack4_i8(sf4_e2m1_i8_x2(raw.y & 15u), sf4_e2m1_i8_x2((raw.y >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 8) & 15u), sf4_e2m1_i8_x2((raw.y >> 12) & 15u));
                    p4.w = pack4_i8(sf4_e2m1_i8_x2((raw.y >> 16) & 15u), sf4_e2m1_i8_x2((raw.y >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 24) & 15u), sf4_e2m1_i8_x2((raw.y >> 28) & 15u));
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = p4;
                } else {
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = make_int4(0, 0, 0, 0);
                    sWs[r * 4 + b4] = 0.0f;
                }
            }
            sync_lds();

            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
                const int col16 = wmma_acc_n(0, lane);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
#if INK_GFX12
                    iv2 b0 = wmma_ld_i8_k16(&sWq[(wn * 64 +  0 + l16) * LDI + kk * 16], hi);
                    iv2 b1 = wmma_ld_i8_k16(&sWq[(wn * 64 + 16 + l16) * LDI + kk * 16], hi);
                    iv2 b2 = wmma_ld_i8_k16(&sWq[(wn * 64 + 32 + l16) * LDI + kk * 16], hi);
                    iv2 b3 = wmma_ld_i8_k16(&sWq[(wn * 64 + 48 + l16) * LDI + kk * 16], hi);
                    iv2 a0 = wmma_ld_i8_k16(&sXq[(wm * 32 + l16) * LDI + kk * 16], hi);
                    iv8 i00 = wmma_i8_k16(a0, b0, iv8_zero());
                    iv8 i01 = wmma_i8_k16(a0, b1, iv8_zero());
                    iv8 i02 = wmma_i8_k16(a0, b2, iv8_zero());
                    iv8 i03 = wmma_i8_k16(a0, b3, iv8_zero());
#elif INK_GFX11
                    iv4 b0 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 +  0 + l16) * LDI + kk * 16]);
                    iv4 b1 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 16 + l16) * LDI + kk * 16]);
                    iv4 b2 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 32 + l16) * LDI + kk * 16]);
                    iv4 b3 = wmma_ld_i8_k16_g11(&sWq[(wn * 64 + 48 + l16) * LDI + kk * 16]);
                    iv4 a0 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + l16) * LDI + kk * 16]);
                    iv8 i00 = wmma_i8_k16_g11(a0, b0, iv8_zero());
                    iv8 i01 = wmma_i8_k16_g11(a0, b1, iv8_zero());
                    iv8 i02 = wmma_i8_k16_g11(a0, b2, iv8_zero());
                    iv8 i03 = wmma_i8_k16_g11(a0, b3, iv8_zero());
#endif
                    const float ws0 = sWs[(wn * 64 +  0 + col16) * 4 + kk];
                    const float ws1 = sWs[(wn * 64 + 16 + col16) * 4 + kk];
                    const float ws2 = sWs[(wn * 64 + 32 + col16) * 4 + kk];
                    const float ws3 = sWs[(wn * 64 + 48 + col16) * 4 + kk];
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd0 = sXd[(wm * 32 + wmma_acc_m(r, lane)) * 4 + kk];
                        acc[0][0][r] = fmaf(static_cast<float>(i00[r]), xd0 * ws0, acc[0][0][r]);
                        acc[0][1][r] = fmaf(static_cast<float>(i01[r]), xd0 * ws1, acc[0][1][r]);
                        acc[0][2][r] = fmaf(static_cast<float>(i02[r]), xd0 * ws2, acc[0][2][r]);
                        acc[0][3][r] = fmaf(static_cast<float>(i03[r]), xd0 * ws3, acc[0][3][r]);
                    }
                    if (do_m1) {
#if INK_GFX12
                        iv2 a1 = wmma_ld_i8_k16(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16], hi);
                        iv8 i10 = wmma_i8_k16(a1, b0, iv8_zero());
                        iv8 i11 = wmma_i8_k16(a1, b1, iv8_zero());
                        iv8 i12 = wmma_i8_k16(a1, b2, iv8_zero());
                        iv8 i13 = wmma_i8_k16(a1, b3, iv8_zero());
#elif INK_GFX11
                        iv4 a1 = wmma_ld_i8_k16_g11(&sXq[(wm * 32 + 16 + l16) * LDI + kk * 16]);
                        iv8 i10 = wmma_i8_k16_g11(a1, b0, iv8_zero());
                        iv8 i11 = wmma_i8_k16_g11(a1, b1, iv8_zero());
                        iv8 i12 = wmma_i8_k16_g11(a1, b2, iv8_zero());
                        iv8 i13 = wmma_i8_k16_g11(a1, b3, iv8_zero());
#endif
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float xd1 = sXd[(wm * 32 + 16 + wmma_acc_m(r, lane)) * 4 + kk];
                            acc[1][0][r] = fmaf(static_cast<float>(i10[r]), xd1 * ws0, acc[1][0][r]);
                            acc[1][1][r] = fmaf(static_cast<float>(i11[r]), xd1 * ws1, acc[1][1][r]);
                            acc[1][2][r] = fmaf(static_cast<float>(i12[r]), xd1 * ws2, acc[1][2][r]);
                            acc[1][3][r] = fmaf(static_cast<float>(i13[r]), xd1 * ws3, acc[1][3][r]);
                        }
                    }
                }
            }
            sync_lds();
        }

        if (wm * 32 < tm) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int gn = n_blk + wn * 64 + ni * 16 + wmma_acc_n(0, lane);
                    if (gn < D) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                            if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[mi][ni][r];
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)act_q8; (void)act_d16; (void)bucket_y; (void)D; (void)inter;
#endif
}

// BM=128, BN=128, BK=32 FP16 WMMA Grouped-MoE w13 & w2 (21.5 KB LDS -> 2-3 WG/CU; handles MX8 + SF4 + S_K!)
__global__ void __launch_bounds__(256) k_moe_grouped_w13_situ_wmma_128(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int32_t* __restrict__ bucket_tok,
    const half_t* __restrict__ X_f16,
    half_t* __restrict__ bucket_act,
    int D, int inter, float beta, float lb) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 8;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) half_t sX[BM * LD];   // 10,240 B
    __shared__ alignas(16) half_t sW[BN * LD];   // 10,240 B
    __shared__ half_t s_e4m3[256];               //    512 B
    __shared__ int32_t sTok[BM];                 //    512 B -> 21.5 KB

    const int tid = threadIdx.x;
    if (desc.gu_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 1;
    const int wn = wave & 1;
    const int N = 2 * inter;
    const int n_blk = blockIdx.x * BN;
    const float sf4_outer = (desc.gu_code == CODE_SF4)
        ? (reinterpret_cast<const float*>(desc.body)[0] / reinterpret_cast<const float*>(desc.body)[1]) : 1.0f;
    const uint8_t* sf4_p = desc.body + 16;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(N) * (D >> 1);
    const uint8_t* mx8_w = desc.body;
    const uint8_t* mx8_s = mx8_w + static_cast<size_t>(N) * D;
    const uint8_t* sk_w  = desc.body + static_cast<size_t>(inter) * 2;
    const int nsb = D >> 8;
    const int sb_b = sk_sb_bytes(desc.gu_code);
    const bool need_sq = (desc.dn_code != CODE_SF4 && desc.dn_code != CODE_MX8);
    const half_t* sq_xd = reinterpret_cast<const half_t*>(desc.body);
    const float out_scale = (desc.gu_code == CODE_SF4) ? (sf4_outer * 16.0f) : 1.0f;
    const int t_stride = gridDim.z * BM;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        if (tid < BM) sTok[tid] = (tid < tm) ? bucket_tok[beg + t0 + tid] : 0;
        sync_lds();

        fv8 acc[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < D; k0 += BK) {
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, c8 = idx & 3;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < tm) v = *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(sTok[r]) * D + k0 + c8 * 8);
                *reinterpret_cast<uint4*>(&sX[r * LD + c8 * 8]) = v;
            }
            if (desc.gu_code == CODE_SF4) {
                const int r = tid >> 1, b2 = tid & 1;
                const int gn = n_blk + r;
                const int gb = (k0 >> 4) + b2;
                half_t* dst = &sW[r * LD + b2 * 16];
                if (gn < N) {
                    const uint32_t sc2 = h2_splat(e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * (D >> 4) + gb]) * 0.0625f);
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
            } else if (desc.gu_code == CODE_MX8) {
                const int r = tid >> 1, q16 = tid & 1;  // 2 threads per row x 16 weights = 32 weights!
                const int gn = n_blk + r;
                const int gb = k0 >> 5;
                half_t* dst = &sW[r * LD + q16 * 16];
                if (gn < N) {
                    const uint32_t sc2 = h2_splat(ue8m0_to_f32(mx8_s[static_cast<size_t>(gn) * (D >> 5) + gb]));
                    const uint8_t* wb = mx8_w + static_cast<size_t>(gn) * D + (gb << 5) + q16 * 16;
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        uint32_t o[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const hv2 h = {s_e4m3[wb[q * 8 + 2 * i]], s_e4m3[wb[q * 8 + 2 * i + 1]]};
                            o[i] = h2_mul(__builtin_bit_cast(uint32_t, h), sc2);
                        }
                        *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(o[0], o[1], o[2], o[3]);
                    }
                } else {
                    *reinterpret_cast<uint4*>(dst)     = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
                }
            } else {
                const int s = k0 >> 8;
                const int g = (k0 >> 6) & 3;
                const int half_off = (k0 & 32);
                const int r = tid >> 1, q16 = tid & 1;
                const int gn = n_blk + r;
                half_t* dst = &sW[r * LD + q16 * 16];
                if (gn < N) {
                    const uint8_t* blk = sk_w + (static_cast<size_t>(gn) * nsb + s) * sb_b;
                    const uint32_t ef2 = h2_splat(decode_sk_sub_eff_i8(blk, desc.gu_code, g));
                    uint4 o0, o1;
                    decode_sk_16_f16(blk + 6, desc.gu_code, (g << 6) + half_off + q16 * 16, ef2, o0, o1);
                    *reinterpret_cast<uint4*>(dst)     = o0;
                    *reinterpret_cast<uint4*>(dst + 8) = o1;
                } else {
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
                }
            }
            sync_lds();

            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    wmma_h_t b0 = wmma_ld_h(&sW[(wn * 64 +  0 + l16) * LD + kk * 16], hi);
                    wmma_h_t b1 = wmma_ld_h(&sW[(wn * 64 + 16 + l16) * LD + kk * 16], hi);
                    wmma_h_t b2 = wmma_ld_h(&sW[(wn * 64 + 32 + l16) * LD + kk * 16], hi);
                    wmma_h_t b3 = wmma_ld_h(&sW[(wn * 64 + 48 + l16) * LD + kk * 16], hi);
                    wmma_h_t a0 = wmma_ld_h(&sX[(wm * 32 + l16) * LD + kk * 16], hi);
                    acc[0][0] = wmma_f16(a0, b0, acc[0][0]);
                    acc[0][1] = wmma_f16(a0, b1, acc[0][1]);
                    acc[0][2] = wmma_f16(a0, b2, acc[0][2]);
                    acc[0][3] = wmma_f16(a0, b3, acc[0][3]);
                    if (do_m1) {
                        wmma_h_t a1 = wmma_ld_h(&sX[(wm * 32 + 16 + l16) * LD + kk * 16], hi);
                        acc[1][0] = wmma_f16(a1, b0, acc[1][0]);
                        acc[1][1] = wmma_f16(a1, b1, acc[1][1]);
                        acc[1][2] = wmma_f16(a1, b2, acc[1][2]);
                        acc[1][3] = wmma_f16(a1, b3, acc[1][3]);
                    }
                }
            }
            sync_lds();
        }

        if (wm * 32 < tm) {
            const bool is_even = ((lane & 1) == 0);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int gn = n_blk + wn * 64 + ni * 16 + wmma_acc_n(0, lane);
                    const int gj = gn >> 1;
                    const float inv_sq = (is_even && need_sq && gj < inter) ? (1.0f / static_cast<float>(sq_xd[gj])) : 1.0f;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float g_val = acc[mi][ni][r] * out_scale;
                        const float u_val = __shfl_xor(g_val, 1, 32);
                        const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                        if (is_even && lm < tm && gj < inter) {
                            bucket_act[static_cast<size_t>(beg + t0 + lm) * inter + gj] =
                                static_cast<half_t>(situ_f32(g_val, u_val, beta, lb) * inv_sq);
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)X_f16; (void)bucket_act; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

__global__ void __launch_bounds__(256) k_moe_grouped_w2_wmma_128(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const half_t* __restrict__ bucket_act,
    float* __restrict__ bucket_y,
    int D, int inter) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 8;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) half_t sX[BM * LD];
    __shared__ alignas(16) half_t sW[BN * LD];
    __shared__ half_t s_e4m3[256];

    const int tid = threadIdx.x;
    if (desc.dn_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 1;
    const int wn = wave & 1;
    const int n_blk = blockIdx.x * BN;
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
    const float out_scale = (desc.dn_code == CODE_SF4) ? (sf4_outer * 16.0f) : 1.0f;
    const int t_stride = gridDim.z * BM;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

        for (int k0 = 0; k0 < inter; k0 += BK) {
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, c8 = idx & 3;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < tm) v = *reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg + t0 + r) * inter + k0 + c8 * 8);
                *reinterpret_cast<uint4*>(&sX[r * LD + c8 * 8]) = v;
            }
            if (desc.dn_code == CODE_SF4) {
                const int r = tid >> 1, b2 = tid & 1;
                const int gn = n_blk + r;
                const int gb = (k0 >> 4) + b2;
                half_t* dst = &sW[r * LD + b2 * 16];
                if (gn < D) {
                    const uint32_t sc2 = h2_splat(e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * (inter >> 4) + gb]) * 0.0625f);
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
            } else if (desc.dn_code == CODE_MX8) {
                const int r = tid >> 1, q16 = tid & 1;
                const int gn = n_blk + r;
                const int gb = k0 >> 5;
                half_t* dst = &sW[r * LD + q16 * 16];
                if (gn < D) {
                    const uint32_t sc2 = h2_splat(ue8m0_to_f32(mx8_s[static_cast<size_t>(gn) * (inter >> 5) + gb]));
                    const uint8_t* wb = mx8_w + static_cast<size_t>(gn) * inter + (gb << 5) + q16 * 16;
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        uint32_t o[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const hv2 h = {s_e4m3[wb[q * 8 + 2 * i]], s_e4m3[wb[q * 8 + 2 * i + 1]]};
                            o[i] = h2_mul(__builtin_bit_cast(uint32_t, h), sc2);
                        }
                        *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(o[0], o[1], o[2], o[3]);
                    }
                } else {
                    *reinterpret_cast<uint4*>(dst)     = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
                }
            } else {
                const int sg = k0 >> 6;
                const int half_off = (k0 & 32);
                const int r = tid >> 1, q16 = tid & 1;
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
                    uint4 o0, o1;
                    decode_sk_16_f16(blk + 6, desc.dn_code, (g << 6) + half_off + q16 * 16, ef2, o0, o1);
                    *reinterpret_cast<uint4*>(dst)     = o0;
                    *reinterpret_cast<uint4*>(dst + 8) = o1;
                } else {
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
                }
            }
            sync_lds();

            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    wmma_h_t b0 = wmma_ld_h(&sW[(wn * 64 +  0 + l16) * LD + kk * 16], hi);
                    wmma_h_t b1 = wmma_ld_h(&sW[(wn * 64 + 16 + l16) * LD + kk * 16], hi);
                    wmma_h_t b2 = wmma_ld_h(&sW[(wn * 64 + 32 + l16) * LD + kk * 16], hi);
                    wmma_h_t b3 = wmma_ld_h(&sW[(wn * 64 + 48 + l16) * LD + kk * 16], hi);
                    wmma_h_t a0 = wmma_ld_h(&sX[(wm * 32 + l16) * LD + kk * 16], hi);
                    acc[0][0] = wmma_f16(a0, b0, acc[0][0]);
                    acc[0][1] = wmma_f16(a0, b1, acc[0][1]);
                    acc[0][2] = wmma_f16(a0, b2, acc[0][2]);
                    acc[0][3] = wmma_f16(a0, b3, acc[0][3]);
                    if (do_m1) {
                        wmma_h_t a1 = wmma_ld_h(&sX[(wm * 32 + 16 + l16) * LD + kk * 16], hi);
                        acc[1][0] = wmma_f16(a1, b0, acc[1][0]);
                        acc[1][1] = wmma_f16(a1, b1, acc[1][1]);
                        acc[1][2] = wmma_f16(a1, b2, acc[1][2]);
                        acc[1][3] = wmma_f16(a1, b3, acc[1][3]);
                    }
                }
            }
            sync_lds();
        }

        if (wm * 32 < tm) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int gn = n_blk + wn * 64 + ni * 16 + wmma_acc_n(0, lane);
                    if (gn < D) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                            if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[mi][ni][r] * out_scale;
                        }
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)bucket_act; (void)bucket_y; (void)D; (void)inter;
#endif
}

// Codec-Specialized Double-Buffered BM=128, BN=128, BK=32 FP16 WMMA Grouped-MoE (0=S_K, 1=SF4, 2=MX8!)
// Eliminates cross-codec VGPR bloat + overlaps global loads with 16 WMMAs + streams b (saves 24 VGPRs on gfx11!)
template <int CODEC_KIND>
__global__ void __launch_bounds__(256) k_moe_grouped_w13_situ_wmma_128_spec(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int32_t* __restrict__ bucket_tok,
    const half_t* __restrict__ X_f16,
    half_t* __restrict__ bucket_act,
    int D, int inter, float beta, float lb) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 8;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) half_t sX[2][BM * LD];  // 20,480 B
    __shared__ alignas(16) half_t sW[2][BN * LD];  // 20,480 B -> 40,960 B (double buffered!)
    __shared__ half_t s_e4m3[CODEC_KIND == 2 ? 256 : 1];
    __shared__ uint32_t s_sf4_pair[CODEC_KIND == 1 ? 256 : 1];
    __shared__ uint32_t s_sf4_sc2[CODEC_KIND == 1 ? 256 : 1];
    __shared__ int32_t sTok[BM];

    const int tid = threadIdx.x;
    if constexpr (CODEC_KIND == 2) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        sync_lds();
    } else if constexpr (CODEC_KIND == 1) {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc2[tid]  = h2_splat(e4m3fn_to_f32(static_cast<uint8_t>(tid)) * 0.0625f);
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 1;
    const int wn = wave & 1;
    const int N = 2 * inter;
    const int n_blk = blockIdx.x * BN;
    const float sf4_outer = (CODEC_KIND == 1)
        ? (reinterpret_cast<const float*>(desc.body)[0] / reinterpret_cast<const float*>(desc.body)[1]) : 1.0f;
    const uint8_t* sf4_p = desc.body + 16;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(N) * (D >> 1);
    const uint8_t* mx8_w = desc.body;
    const uint8_t* mx8_s = mx8_w + static_cast<size_t>(N) * D;
    const uint8_t* sk_w  = desc.body + static_cast<size_t>(inter) * 2;
    const int nsb = D >> 8;
    const int sb_b = (CODEC_KIND == 0) ? sk_sb_bytes(desc.gu_code) : 0;
    const bool need_sq = (CODEC_KIND == 0) && (desc.dn_code != CODE_SF4 && desc.dn_code != CODE_MX8);
    const half_t* sq_xd = reinterpret_cast<const half_t*>(desc.body);
    const float out_scale = (CODEC_KIND == 1) ? (sf4_outer * 16.0f) : 1.0f;
    const int t_stride = gridDim.z * BM;

    const int xr0 = tid >> 2, xc0 = tid & 3;
    const int xr1 = (tid + 256) >> 2, xc1 = (tid + 256) & 3;
    const int wr = tid >> 1, wq = tid & 1;
    const int wgn = n_blk + wr;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        if (tid < BM) sTok[tid] = (tid < tm) ? bucket_tok[beg + t0 + tid] : 0;
        sync_lds();

        auto load_x_tile = [&](int k0, uint4& x0, uint4& x1) {
            x0 = (xr0 < tm) ? *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(sTok[xr0]) * D + k0 + xc0 * 8) : make_uint4(0, 0, 0, 0);
            x1 = (xr1 < tm) ? *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(sTok[xr1]) * D + k0 + xc1 * 8) : make_uint4(0, 0, 0, 0);
        };
        auto load_w_tile = [&](int k0, uint4& o0, uint4& o1) {
            if (wgn >= N) { o0 = make_uint4(0, 0, 0, 0); o1 = make_uint4(0, 0, 0, 0); return; }
            if constexpr (CODEC_KIND == 1) {
                const int gb = (k0 >> 4) + wq;
                const uint32_t sc2 = s_sf4_sc2[sf4_s[static_cast<size_t>(wgn) * (D >> 4) + gb]];
                const uint2 raw = *reinterpret_cast<const uint2*>(sf4_p + static_cast<size_t>(wgn) * (D >> 1) + gb * 8);
                uint32_t a0[4], a1[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    a0[i] = h2_mul(s_sf4_pair[(raw.x >> (i * 8)) & 0xFFu], sc2);
                    a1[i] = h2_mul(s_sf4_pair[(raw.y >> (i * 8)) & 0xFFu], sc2);
                }
                o0 = make_uint4(a0[0], a0[1], a0[2], a0[3]);
                o1 = make_uint4(a1[0], a1[1], a1[2], a1[3]);
            } else if constexpr (CODEC_KIND == 2) {
                const int gb = k0 >> 5;
                const uint32_t sc2 = h2_splat(ue8m0_to_f32(mx8_s[static_cast<size_t>(wgn) * (D >> 5) + gb]));
                const uint8_t* wb = mx8_w + static_cast<size_t>(wgn) * D + (gb << 5) + wq * 16;
                uint32_t a0[4], a1[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const hv2 h0 = {s_e4m3[wb[2 * i]], s_e4m3[wb[2 * i + 1]]};
                    const hv2 h1 = {s_e4m3[wb[8 + 2 * i]], s_e4m3[wb[8 + 2 * i + 1]]};
                    a0[i] = h2_mul(__builtin_bit_cast(uint32_t, h0), sc2);
                    a1[i] = h2_mul(__builtin_bit_cast(uint32_t, h1), sc2);
                }
                o0 = make_uint4(a0[0], a0[1], a0[2], a0[3]);
                o1 = make_uint4(a1[0], a1[1], a1[2], a1[3]);
            } else {
                const int s = k0 >> 8;
                const int g = (k0 >> 6) & 3;
                const int half_off = (k0 & 32);
                const uint8_t* blk = sk_w + (static_cast<size_t>(wgn) * nsb + s) * sb_b;
                const uint32_t ef2 = h2_splat(decode_sk_sub_eff_i8(blk, desc.gu_code, g));
                decode_sk_16_f16(blk + 6, desc.gu_code, (g << 6) + half_off + wq * 16, ef2, o0, o1);
            }
        };

        uint4 rx0, rx1, rw0, rw1;
        load_x_tile(0, rx0, rx1);
        load_w_tile(0, rw0, rw1);
        *reinterpret_cast<uint4*>(&sX[0][xr0 * LD + xc0 * 8]) = rx0;
        *reinterpret_cast<uint4*>(&sX[0][xr1 * LD + xc1 * 8]) = rx1;
        *reinterpret_cast<uint4*>(&sW[0][wr * LD + wq * 16])     = rw0;
        *reinterpret_cast<uint4*>(&sW[0][wr * LD + wq * 16 + 8]) = rw1;
        sync_lds();

        fv8 acc[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

        int buf = 0;
        for (int k0 = 0; k0 < D; k0 += BK) {
            const int kn = k0 + BK;
            if (kn < D) {
                load_x_tile(kn, rx0, rx1);
                load_w_tile(kn, rw0, rw1);
            }
            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    wmma_h_t a0 = wmma_ld_h(&sX[buf][(wm * 32 + l16) * LD + kk * 16], hi);
                    wmma_h_t a1 = do_m1 ? wmma_ld_h(&sX[buf][(wm * 32 + 16 + l16) * LD + kk * 16], hi) : a0;
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni) {
                        wmma_h_t b = wmma_ld_h(&sW[buf][(wn * 64 + ni * 16 + l16) * LD + kk * 16], hi);
                        acc[0][ni] = wmma_f16(a0, b, acc[0][ni]);
                        if (do_m1) acc[1][ni] = wmma_f16(a1, b, acc[1][ni]);
                    }
                }
            }
            if (kn < D) {
                const int nb = buf ^ 1;
                *reinterpret_cast<uint4*>(&sX[nb][xr0 * LD + xc0 * 8]) = rx0;
                *reinterpret_cast<uint4*>(&sX[nb][xr1 * LD + xc1 * 8]) = rx1;
                *reinterpret_cast<uint4*>(&sW[nb][wr * LD + wq * 16])     = rw0;
                *reinterpret_cast<uint4*>(&sW[nb][wr * LD + wq * 16 + 8]) = rw1;
                sync_lds();
                buf = nb;
            }
        }

        if (wm * 32 < tm) {
            const bool is_even = ((lane & 1) == 0);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int gn = n_blk + wn * 64 + ni * 16 + wmma_acc_n(0, lane);
                    const int gj = gn >> 1;
                    const float inv_sq = (is_even && need_sq && gj < inter) ? (1.0f / static_cast<float>(sq_xd[gj])) : 1.0f;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float g_val = acc[mi][ni][r] * out_scale;
                        const float u_val = __shfl_xor(g_val, 1, 32);
                        const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                        if (is_even && lm < tm && gj < inter) {
                            bucket_act[static_cast<size_t>(beg + t0 + lm) * inter + gj] =
                                static_cast<half_t>(situ_f32(g_val, u_val, beta, lb) * inv_sq);
                        }
                    }
                }
            }
        }
        if (t0 + t_stride < m_e) sync_lds();
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)X_f16; (void)bucket_act; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

template <int CODEC_KIND>
__global__ void __launch_bounds__(256) k_moe_grouped_w2_wmma_128_spec(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const half_t* __restrict__ bucket_act,
    float* __restrict__ bucket_y,
    int D, int inter) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 8;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) half_t sX[2][BM * LD];
    __shared__ alignas(16) half_t sW[2][BN * LD];
    __shared__ half_t s_e4m3[CODEC_KIND == 2 ? 256 : 1];
    __shared__ uint32_t s_sf4_pair[CODEC_KIND == 1 ? 256 : 1];
    __shared__ uint32_t s_sf4_sc2[CODEC_KIND == 1 ? 256 : 1];

    const int tid = threadIdx.x;
    if constexpr (CODEC_KIND == 2) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        sync_lds();
    } else if constexpr (CODEC_KIND == 1) {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc2[tid]  = h2_splat(e4m3fn_to_f32(static_cast<uint8_t>(tid)) * 0.0625f);
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int wm = wave >> 1;
    const int wn = wave & 1;
    const int n_blk = blockIdx.x * BN;
    const float sf4_outer = (CODEC_KIND == 1)
        ? (reinterpret_cast<const float*>(desc.body)[2] / reinterpret_cast<const float*>(desc.body)[3]) : 1.0f;
    const uint8_t* sf4_p = desc.body + desc.dn_offset;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(D) * (inter >> 1);
    const uint8_t* mx8_w = desc.body + desc.dn_offset;
    const uint8_t* mx8_s = mx8_w + static_cast<size_t>(D) * inter;
    const uint8_t* sk_w  = desc.body + desc.dn_offset;
    const int sb_per_pair = (2 * inter) >> 8;
    const int sb_b = (CODEC_KIND == 0) ? sk_sb_bytes(desc.dn_code) : 0;
    const int sub_per_row = inter >> 6;
    const float out_scale = (CODEC_KIND == 1) ? (sf4_outer * 16.0f) : 1.0f;
    const int t_stride = gridDim.z * BM;

    const int xr0 = tid >> 2, xc0 = tid & 3;
    const int xr1 = (tid + 256) >> 2, xc1 = (tid + 256) & 3;
    const int wr = tid >> 1, wq = tid & 1;
    const int wgn = n_blk + wr;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;

        auto load_x_tile = [&](int k0, uint4& x0, uint4& x1) {
            x0 = (xr0 < tm) ? *reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg + t0 + xr0) * inter + k0 + xc0 * 8) : make_uint4(0, 0, 0, 0);
            x1 = (xr1 < tm) ? *reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg + t0 + xr1) * inter + k0 + xc1 * 8) : make_uint4(0, 0, 0, 0);
        };
        auto load_w_tile = [&](int k0, uint4& o0, uint4& o1) {
            if (wgn >= D) { o0 = make_uint4(0, 0, 0, 0); o1 = make_uint4(0, 0, 0, 0); return; }
            if constexpr (CODEC_KIND == 1) {
                const int gb = (k0 >> 4) + wq;
                const uint32_t sc2 = s_sf4_sc2[sf4_s[static_cast<size_t>(wgn) * (inter >> 4) + gb]];
                const uint2 raw = *reinterpret_cast<const uint2*>(sf4_p + static_cast<size_t>(wgn) * (inter >> 1) + gb * 8);
                uint32_t a0[4], a1[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    a0[i] = h2_mul(s_sf4_pair[(raw.x >> (i * 8)) & 0xFFu], sc2);
                    a1[i] = h2_mul(s_sf4_pair[(raw.y >> (i * 8)) & 0xFFu], sc2);
                }
                o0 = make_uint4(a0[0], a0[1], a0[2], a0[3]);
                o1 = make_uint4(a1[0], a1[1], a1[2], a1[3]);
            } else if constexpr (CODEC_KIND == 2) {
                const int gb = k0 >> 5;
                const uint32_t sc2 = h2_splat(ue8m0_to_f32(mx8_s[static_cast<size_t>(wgn) * (inter >> 5) + gb]));
                const uint8_t* wb = mx8_w + static_cast<size_t>(wgn) * inter + (gb << 5) + wq * 16;
                uint32_t a0[4], a1[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const hv2 h0 = {s_e4m3[wb[2 * i]], s_e4m3[wb[2 * i + 1]]};
                    const hv2 h1 = {s_e4m3[wb[8 + 2 * i]], s_e4m3[wb[8 + 2 * i + 1]]};
                    a0[i] = h2_mul(__builtin_bit_cast(uint32_t, h0), sc2);
                    a1[i] = h2_mul(__builtin_bit_cast(uint32_t, h1), sc2);
                }
                o0 = make_uint4(a0[0], a0[1], a0[2], a0[3]);
                o1 = make_uint4(a1[0], a1[1], a1[2], a1[3]);
            } else {
                const int sg = k0 >> 6;
                const int half_off = (k0 & 32);
                const int pair = wgn >> 1;
                const int is_odd = wgn & 1;
                const int sub_global = is_odd * sub_per_row + sg;
                const int s = sub_global >> 2;
                const int g = sub_global & 3;
                const uint8_t* blk = sk_w + (static_cast<size_t>(pair) * sb_per_pair + s) * sb_b;
                const uint32_t ef2 = h2_splat(decode_sk_sub_eff_i8(blk, desc.dn_code, g));
                decode_sk_16_f16(blk + 6, desc.dn_code, (g << 6) + half_off + wq * 16, ef2, o0, o1);
            }
        };

        uint4 rx0, rx1, rw0, rw1;
        load_x_tile(0, rx0, rx1);
        load_w_tile(0, rw0, rw1);
        *reinterpret_cast<uint4*>(&sX[0][xr0 * LD + xc0 * 8]) = rx0;
        *reinterpret_cast<uint4*>(&sX[0][xr1 * LD + xc1 * 8]) = rx1;
        *reinterpret_cast<uint4*>(&sW[0][wr * LD + wq * 16])     = rw0;
        *reinterpret_cast<uint4*>(&sW[0][wr * LD + wq * 16 + 8]) = rw1;
        sync_lds();

        fv8 acc[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

        int buf = 0;
        for (int k0 = 0; k0 < inter; k0 += BK) {
            const int kn = k0 + BK;
            if (kn < inter) {
                load_x_tile(kn, rx0, rx1);
                load_w_tile(kn, rw0, rw1);
            }
            if (wm * 32 < tm) {
                const bool do_m1 = (wm * 32 + 16 < tm);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    wmma_h_t a0 = wmma_ld_h(&sX[buf][(wm * 32 + l16) * LD + kk * 16], hi);
                    wmma_h_t a1 = do_m1 ? wmma_ld_h(&sX[buf][(wm * 32 + 16 + l16) * LD + kk * 16], hi) : a0;
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni) {
                        wmma_h_t b = wmma_ld_h(&sW[buf][(wn * 64 + ni * 16 + l16) * LD + kk * 16], hi);
                        acc[0][ni] = wmma_f16(a0, b, acc[0][ni]);
                        if (do_m1) acc[1][ni] = wmma_f16(a1, b, acc[1][ni]);
                    }
                }
            }
            if (kn < inter) {
                const int nb = buf ^ 1;
                *reinterpret_cast<uint4*>(&sX[nb][xr0 * LD + xc0 * 8]) = rx0;
                *reinterpret_cast<uint4*>(&sX[nb][xr1 * LD + xc1 * 8]) = rx1;
                *reinterpret_cast<uint4*>(&sW[nb][wr * LD + wq * 16])     = rw0;
                *reinterpret_cast<uint4*>(&sW[nb][wr * LD + wq * 16 + 8]) = rw1;
                sync_lds();
                buf = nb;
            }
        }

        if (wm * 32 < tm) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (wm * 32 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int gn = n_blk + wn * 64 + ni * 16 + wmma_acc_n(0, lane);
                    if (gn < D) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const int lm = wm * 32 + mi * 16 + wmma_acc_m(r, lane);
                            if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[mi][ni][r] * out_scale;
                        }
                    }
                }
            }
        }
        if (t0 + t_stride < m_e) sync_lds();
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
    if (M == 1) {
        const dim3 grid((N + 7) / 8);
        if (K <= 4096 && (K & 7) == 0) {
            k_bf16_gemv1_lds<<<grid, 256, 0, st>>>(W, X, bias, Y, N, K);
        } else {
            k_bf16_gemv<1><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        }
        return;
    }
    if (M <= 8) {
        const dim3 grid((N + 7) / 8);
        if (K >= 256) {
            if (M == 2)      k_bf16_gemv_lds_fdot2<2><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
            else if (M <= 4) k_bf16_gemv_lds_fdot2<4><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
            else if (M <= 6) k_bf16_gemv_lds_fdot2<6><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
            else             k_bf16_gemv_lds_fdot2<8><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        } else {
            if (M == 2)      k_bf16_gemv<2><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
            else if (M <= 4) k_bf16_gemv<4><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
            else if (M <= 6) k_bf16_gemv<6><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
            else             k_bf16_gemv<8><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        }
        return;
    }
    if (gen == strata::hip_uma::AmdGen::Rdna2 || gen == strata::hip_uma::AmdGen::Unknown) {
        const dim3 grid((N + 31) / 32, (M + 15) / 16);
        k_bf16_fdot2_tiled<16, 32><<<grid, 128, 0, st>>>(W, X, bias, Y, M, N, K);
        return;
    }
    if (M <= 16 || N <= 128) {
        // Use BM=16 for small M or small N (e.g. rank-64 loop_ctrl down_weight -> 32 blocks at B=512 instead of 8!)
        if (N >= 64) {
            const dim3 grid((N + 63) / 64, (M + 15) / 16);
            k_bf16_wmma_bf16_16xN<64><<<grid, 128, 0, st>>>(W, X, bias, Y, M, N, K);
        } else {
            const dim3 grid((N + 31) / 32, (M + 15) / 16);
            k_bf16_wmma_bf16_16xN<32><<<grid, 64, 0, st>>>(W, X, bias, Y, M, N, K);
        }
    } else if (N >= 256) {
        const dim3 grid((N + 127) / 128, (M + 63) / 64);
        k_bf16_wmma_bf16_64x128<<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
    } else {
        const dim3 grid((N + 63) / 64, (M + 63) / 64);
        k_bf16_wmma_f16<64, 64><<<grid, 128, 0, st>>>(W, X, bias, Y, M, N, K);
    }
}

inline void launch_f32_linear(
    const float* W, const float* X, const float* bias, float* Y,
    int M, int N, int K, hipStream_t st, strata::hip_uma::AmdGen gen = strata::hip_uma::AmdGen::Unknown) {
    if (M <= 12 && N <= 128 && (K & 3) == 0 && K >= 256) {
        k_f32_gemv_hiocc<<<dim3(N, M), 256, 0, st>>>(W, X, bias, Y, M, N, K);
        return;
    }
    if (M <= 8) {
        const dim3 grid((N + 7) / 8);
        if (M == 1)      k_f32_gemv<1><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        else if (M == 2) k_f32_gemv<2><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        else if (M <= 4) k_f32_gemv<4><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        else             k_f32_gemv<8><<<grid, 256, 0, st>>>(W, X, bias, Y, M, N, K);
        return;
    }
    if (gen != strata::hip_uma::AmdGen::Rdna2 && gen != strata::hip_uma::AmdGen::Unknown && N >= 64) {
        // Use BM=16 when M<=16 OR N<=128 (e.g. MoE router N=128 -> 64 blocks at B=512 instead of 16!)
        if (M <= 16 || N <= 128) {
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

// =================================================================================================
// 6. Wave-3 L3-Resident Expert Unpack-Once Kernels (SF4, MX8, Compile-Time Specialized S_K) + Trunk Splitters
//    Unpacks ONE expert's w13 (11.8 MB) or w2 (5.9 MB) directly into L3/MALL cache -> hipBLASLt Tensile FP16 GEMM!
// =================================================================================================

// Unpack SF4 [N, K] into FP16 with 1/16 tile norm (caller passes alpha = sf4_outer * 16.0f to hipBLASLt -> zero FP16 underflow!)
// Grid: ((N * (K / 16) + 255) / 256,), Block: 256 (1 thread = 16 weights = 1 SF4 block -> 2 coalesced uint4 stores!)
__global__ void __launch_bounds__(256) k_unpack_sf4_f16(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    half_t* __restrict__ out_f16,
    int N, int K) {
    __shared__ float s_e4m3_norm[256];
    s_e4m3_norm[threadIdx.x] = e4m3fn_to_f32(static_cast<uint8_t>(threadIdx.x)) * 0.0625f;
    __syncthreads();

    const int nblk = K >> 4;
    const int tot_blks = N * nblk;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tot_blks) return;

    const uint32_t sc2 = h2_splat(s_e4m3_norm[scales[idx]]);
    const uint2 raw = *reinterpret_cast<const uint2*>(packed + static_cast<size_t>(idx) * 8);
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
    half_t* dst = out_f16 + static_cast<size_t>(idx) * 16;
    *reinterpret_cast<uint4*>(dst)     = make_uint4(o0[0], o0[1], o0[2], o0[3]);
    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
}

// Unpack MX8 [N, K] into FP16 (1 thread = 32 weights = 1 MX8 block -> 4 coalesced uint4 stores!)
__global__ void __launch_bounds__(256) k_unpack_mx8_f16(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ scales,
    half_t* __restrict__ out_f16,
    int N, int K) {
    __shared__ half_t s_e4m3[256];
    s_e4m3[threadIdx.x] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(threadIdx.x)));
    __syncthreads();

    const int nblk = K >> 5;
    const int tot_blks = N * nblk;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tot_blks) return;

    const uint32_t sc2 = h2_splat(ue8m0_to_f32(scales[idx]));
    const uint8_t* wb = weights + static_cast<size_t>(idx) * 32;
    half_t* dst = out_f16 + static_cast<size_t>(idx) * 32;
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
}

// Compile-time specialized S_K [N, K] -> FP16 unpacker (ZERO runtime switch! 1 thread = 16 weights = 2 uint4 stores!)
template <int CODE>
__global__ void __launch_bounds__(256) k_unpack_sk_f16(
    const uint8_t* __restrict__ W,
    half_t* __restrict__ out_f16,
    int N, int K) {
    const int q16_per_row = K >> 4;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * q16_per_row) return;
    const int row = idx / q16_per_row;
    const int q16 = idx % q16_per_row;
    const int s = q16 >> 4;
    const int j0 = (q16 & 15) * 16;
    const int g = j0 >> 6;
    const int nsb = K >> 8;
    const int sb_b = sk_sb_bytes(CODE);
    const uint8_t* blk = W + (static_cast<size_t>(row) * nsb + s) * sb_b;
    const uint32_t ef2 = h2_splat(decode_sk_sub_eff_i8(blk, CODE, g));
    uint4 o0, o1;
    decode_sk_16_f16(blk + 6, CODE, j0, ef2, o0, o1);
    half_t* dst = out_f16 + static_cast<size_t>(idx) * 16;
    *reinterpret_cast<uint4*>(dst)     = o0;
    *reinterpret_cast<uint4*>(dst + 8) = o1;
}

// Compile-time specialized paired-2-row S_K w2 [D, inter] -> FP16 unpacker (ZERO runtime switch!)
template <int CODE>
__global__ void __launch_bounds__(256) k_unpack_sk_paired2_f16(
    const uint8_t* __restrict__ W,
    half_t* __restrict__ out_f16,
    int D, int inter) {
    const int q16_per_row = inter >> 4;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= D * q16_per_row) return;
    const int row = idx / q16_per_row;
    const int q16 = idx % q16_per_row;
    const int sg = q16 >> 2;
    const int sub_per_row = inter >> 6;
    const int pair = row >> 1;
    const int is_odd = row & 1;
    const int sub_global = is_odd * sub_per_row + sg;
    const int s = sub_global >> 2;
    const int g = sub_global & 3;
    const int sb_per_pair = (2 * inter) >> 8;
    const int sb_b = sk_sb_bytes(CODE);
    const uint8_t* blk = W + (static_cast<size_t>(pair) * sb_per_pair + s) * sb_b;
    const uint32_t ef2 = h2_splat(decode_sk_sub_eff_i8(blk, CODE, g));
    uint4 o0, o1;
    decode_sk_16_f16(blk + 6, CODE, (g << 6) + (q16 & 3) * 16, ef2, o0, o1);
    half_t* dst = out_f16 + static_cast<size_t>(idx) * 16;
    *reinterpret_cast<uint4*>(dst)     = o0;
    *reinterpret_cast<uint4*>(dst + 8) = o1;
}

// Fused SITU + optional sq_xd_dn divide directly on hipBLASLt FP32 gu[m_e, 2*inter] -> act[m_e, inter] FP16!
__global__ void __launch_bounds__(256) k_situ_interleaved_sq(
    const float* __restrict__ gu,
    const half_t* __restrict__ sq_xd_dn,  // nullptr for SF4/MX8, or [inter] FP16 for S_K
    half_t* __restrict__ act_out,
    int m_e, int inter, float beta, float lb) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m_e * inter) return;
    const float g = gu[static_cast<size_t>(idx) * 2];
    const float u = gu[static_cast<size_t>(idx) * 2 + 1];
    float v = situ_f32(g, u, beta, lb);
    if (sq_xd_dn) v /= static_cast<float>(sq_xd_dn[idx % inter]);
    act_out[idx] = static_cast<half_t>(v);
}

inline void launch_unpack_sk_f16_dispatch(const uint8_t* W, int code, half_t* dst, int N, int K, hipStream_t st) {
    const int blks = (N * (K >> 4) + 255) / 256;
    switch (code) {
        case 0: k_unpack_sk_f16<0><<<blks, 256, 0, st>>>(W, dst, N, K); break;
        case 1: k_unpack_sk_f16<1><<<blks, 256, 0, st>>>(W, dst, N, K); break;
        case 2: k_unpack_sk_f16<2><<<blks, 256, 0, st>>>(W, dst, N, K); break;
        case 3: k_unpack_sk_f16<3><<<blks, 256, 0, st>>>(W, dst, N, K); break;
        case 4: k_unpack_sk_f16<4><<<blks, 256, 0, st>>>(W, dst, N, K); break;
        case 5: k_unpack_sk_f16<5><<<blks, 256, 0, st>>>(W, dst, N, K); break;
        case 6: k_unpack_sk_f16<6><<<blks, 256, 0, st>>>(W, dst, N, K); break;
        case 7: k_unpack_sk_f16<7><<<blks, 256, 0, st>>>(W, dst, N, K); break;
        default: k_unpack_sk_f16<8><<<blks, 256, 0, st>>>(W, dst, N, K); break;
    }
}

inline void launch_unpack_sk_paired2_f16_dispatch(const uint8_t* W, int code, half_t* dst, int D, int inter, hipStream_t st) {
    const int blks = (D * (inter >> 4) + 255) / 256;
    switch (code) {
        case 0: k_unpack_sk_paired2_f16<0><<<blks, 256, 0, st>>>(W, dst, D, inter); break;
        case 1: k_unpack_sk_paired2_f16<1><<<blks, 256, 0, st>>>(W, dst, D, inter); break;
        case 2: k_unpack_sk_paired2_f16<2><<<blks, 256, 0, st>>>(W, dst, D, inter); break;
        case 3: k_unpack_sk_paired2_f16<3><<<blks, 256, 0, st>>>(W, dst, D, inter); break;
        case 4: k_unpack_sk_paired2_f16<4><<<blks, 256, 0, st>>>(W, dst, D, inter); break;
        case 5: k_unpack_sk_paired2_f16<5><<<blks, 256, 0, st>>>(W, dst, D, inter); break;
        case 6: k_unpack_sk_paired2_f16<6><<<blks, 256, 0, st>>>(W, dst, D, inter); break;
        case 7: k_unpack_sk_paired2_f16<7><<<blks, 256, 0, st>>>(W, dst, D, inter); break;
        default: k_unpack_sk_paired2_f16<8><<<blks, 256, 0, st>>>(W, dst, D, inter); break;
    }
}

// Fast 1-launch splitter for concatenated trunk GEMM output [B, total_n] -> (q, r, gate_a [, k, v])!
// Layout of packed rows: [q (q_dim) | r (r_dim) | gate_a (q_dim) | k (kv_dim) | v (kv_dim)]
__global__ void __launch_bounds__(256) k_split_qkvrg(
    const float* __restrict__ src,  // [B, total_n]
    float* __restrict__ q_out,
    float* __restrict__ r_out,
    float* __restrict__ g_out,
    float* __restrict__ k_out,      // nullptr when total_n == 2*q_dim + r_dim
    float* __restrict__ v_out,      // nullptr when total_n == 2*q_dim + r_dim
    int B, int q_dim, int r_dim, int kv_dim, int total_n) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    const int c = c4 * 4;
    if (b >= B || c >= total_n) return;
    const float4 v = *reinterpret_cast<const float4*>(src + static_cast<size_t>(b) * total_n + c);
    const int o_r = q_dim;
    const int o_g = q_dim + r_dim;
    const int o_k = o_g + q_dim;
    const int o_v = o_k + kv_dim;
    if (c < o_r) {
        *reinterpret_cast<float4*>(q_out + static_cast<size_t>(b) * q_dim + c) = v;
    } else if (c < o_g) {
        *reinterpret_cast<float4*>(r_out + static_cast<size_t>(b) * r_dim + (c - o_r)) = v;
    } else if (c < o_k) {
        *reinterpret_cast<float4*>(g_out + static_cast<size_t>(b) * q_dim + (c - o_g)) = v;
    } else if (k_out && c < o_v) {
        *reinterpret_cast<float4*>(k_out + static_cast<size_t>(b) * kv_dim + (c - o_k)) = v;
    } else if (v_out) {
        *reinterpret_cast<float4*>(v_out + static_cast<size_t>(b) * kv_dim + (c - o_v)) = v;
    }
}

}  // namespace strata::inkling2::hip
