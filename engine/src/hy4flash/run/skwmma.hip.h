// skwmma.hip.h - prefill GEMM (M >= 16) for S_K weights with gfx11 WMMA (v_wmma_f32_16x16x16_f16, wave32)
//
// Y[m][n] = sum_k W[n][k] X[m][k];  W S_K wire format [N][K] (K % 256 == 0), X __half [M][K], Y float [M][N] (row-major).
// Block tile 128 tokens x 64 weight rows, 8 waves (4 x 2), wave tile 32 x 32 (2 x 2 WMMA tiles). Per super-block (256 k) the 64 weight rows are
// dequantized once to fp16 in LDS (w = (q-H) * D*u/255 computed in fp32, rounded to fp16) and reused by all 128 tokens.
// A operand (X) is read straight from global memory (L1/L2 reuse), B operand (W tile) from padded LDS.
// gemm_mainloop is shared with the grouped expert kernels (skmoe.hip.h).
#pragma once
#include "skfast.hip.h"
#if HY4_MMA

namespace hy4 {
namespace skd {

#if !HY4_CUDA
typedef half_t hv16 __attribute__((ext_vector_type(16)));
typedef float fv8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ fv8 fv8_zero() { return fv8{0, 0, 0, 0, 0, 0, 0, 0}; }

__device__ __forceinline__ hv16 ld_hv16(const void* p) {
    const uint4 a = *(const uint4*)p, b = *((const uint4*)p + 1);
    typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
    u32x8 v = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(hv16, v);
}
// accumulator element r (0..7) of a lane in a 16x16 output tile: row / column within the tile
__device__ __forceinline__ int acc_m(int r, int lane) { return 2 * r + (lane >> 4); }
__device__ __forceinline__ int acc_n(int r, int lane) { return lane & 15; }
#else
#endif

// dequantize the 8 weights 8u..8u+7 of super-block `blk` scaled by eff into 4 packed half2 words
template <int FMT>
__device__ __forceinline__ void dequant8_h(const uint8_t* blk, int u, float eff, uint4& out) {
    uint32_t wp[4];
    decode_w8<FMT>(blk + 6, u, wp);
    uint32_t r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a, b;
        h2_unpack(wp[i], a, b);
        r[i] = h2_pack(a * eff, b * eff);
    }
    out = make_uint4(r[0], r[1], r[2], r[3]);
}

constexpr int GEMM_BM = 128, GEMM_BN = 64, GEMM_LDW = 264;

// One block's K loop. wr: this thread's dequant row (tid>>2) pointer of the weight tile (row start, nsb super-blocks).
// xrow0/xrow1: A-operand row pointers (fp16, K elements) for this lane's two m-tiles (row = lane & 15). b0row/b1row: LDS tile row bases of the two n-tiles
// of this wave. ws: LDS [64][GEMM_LDW].

// ---- raw sub-group loader: the 64 weights of sub-group dg of a super-block are fetched with 4-byte aligned dword loads (+ byte realignment),
// one stage ahead of their use, so the global-memory latency hides behind the WMMA loop of the previous super-block.
template <int FMT>
struct RawSub {
    static constexpr int NLO = FMT == FMT_S1_25 ? 3 : (FMT == FMT_S2 || FMT == FMT_S3 || FMT == FMT_S1_5) ? 4 : (FMT == FMT_S8 ? 16 : 8);
    static constexpr int NHI = FMT == FMT_S3 ? 2 : FMT == FMT_S5 ? 2 : FMT == FMT_S6 ? 4 : 1;
    uint32_t lo[NLO];
    uint32_t hi[NHI];
    uint32_t hdr[2];
};

template <int N>
__device__ __forceinline__ void load_bytes(const uint8_t* p, uint32_t (&o)[N]) {
    const uintptr_t a = (uintptr_t)p;
    const uint32_t ofs = (uint32_t)(a & 3);
    const uint32_t* q = (const uint32_t*)(a - ofs);
    uint32_t d[N + 1];
#pragma unroll
    for (int i = 0; i < N; ++i) d[i] = q[i];
    d[N] = ofs ? q[N] : 0u;
#pragma unroll
    for (int i = 0; i < N; ++i) o[i] = byte_funnel(d[i + 1], d[i], ofs);
}

template <int FMT>
__device__ __forceinline__ void load_raw(const uint8_t* blk, int dg, RawSub<FMT>& r) {
    load_bytes<2>(blk, r.hdr);
    const uint8_t* pl = blk + 6;
    if constexpr (FMT == FMT_S1_25) { load_bytes<3>(pl + dg * 10, r.lo); r.hi[0] = pl[40 + (dg >> 1)]; }  // 16 indices (10 bytes) + fine byte
    else if constexpr (FMT == FMT_S1_5) { load_bytes<4>(pl + (64 * dg) / 5, r.lo); r.hi[0] = 64u * dg; }  // trits 64dg..64dg+63 (<= 14 bytes)
    else if constexpr (FMT == FMT_S2) load_bytes<4>(pl + dg * 16, r.lo);
    else if constexpr (FMT == FMT_S3) { load_bytes<4>(pl + dg * 16, r.lo); load_bytes<2>(pl + 64 + dg * 8, r.hi); }
    else if constexpr (FMT == FMT_S4) load_bytes<8>(pl + dg * 32, r.lo);
    else if constexpr (FMT == FMT_S5) { load_bytes<8>(pl + dg * 32, r.lo); load_bytes<2>(pl + 128 + dg * 8, r.hi); }
    else if constexpr (FMT == FMT_S6) { load_bytes<8>(pl + dg * 32, r.lo); load_bytes<4>(pl + 128 + dg * 16, r.hi); }
    else load_bytes<16>(pl + dg * 64, r.lo);
}

template <int FMT>
__device__ __forceinline__ float raw_eff(const RawSub<FMT>& r, int dg) {
    const float D = h2f_bits((uint16_t)(r.hdr[0] & 0xFFFFu));
    const uint32_t u = dg < 2 ? (r.hdr[0] >> (16 + 8 * dg)) & 0xFFu : (r.hdr[1] >> (8 * (dg - 2))) & 0xFFu;
    float eff = D * ((float)u * (1.0f / 255.0f));
    if constexpr (FMT == FMT_S1_25) eff *= 0.5f + (float)((r.hi[0] >> ((dg & 1) * 4)) & 15u) / 15.0f;
    return eff;
}

__device__ __forceinline__ uint32_t byte_at4(const uint32_t (&w)[4], int o) {
    const int d = o >> 2;
    const uint32_t v = d == 0 ? w[0] : d == 1 ? w[1] : d == 2 ? w[2] : w[3];
    return (v >> ((o & 3) * 8)) & 0xFFu;
}

// q[0..7] of unit u8 (weights 8*u8 .. 8*u8+7 of the sub-group); S8 gives byte ^ 0x80
template <int FMT, int U8>
__device__ __forceinline__ void raw_q8(const RawSub<FMT>& r, uint32_t (&q)[8]) {
    if constexpr (FMT == FMT_S2 || FMT == FMT_S3) {
        const uint32_t w = (r.lo[U8 >> 1] >> (16 * (U8 & 1))) & 0xFFFFu;
        uint32_t hb = 0;
        if constexpr (FMT == FMT_S3) hb = (r.hi[U8 >> 2] >> (8 * (U8 & 3))) & 0xFFu;
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = ((w >> (2 * k)) & 3u) | (FMT == FMT_S3 ? (((hb >> k) & 1u) << 2) : 0u);
    } else if constexpr (FMT == FMT_S4 || FMT == FMT_S5 || FMT == FMT_S6) {
        const uint32_t w = r.lo[U8];
        uint32_t hb = 0, h16 = 0;
        if constexpr (FMT == FMT_S5) hb = (r.hi[U8 >> 2] >> (8 * (U8 & 3))) & 0xFFu;
        if constexpr (FMT == FMT_S6) h16 = (r.hi[U8 >> 1] >> (16 * (U8 & 1))) & 0xFFFFu;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t v = (w >> (4 * k)) & 15u;
            if constexpr (FMT == FMT_S5) v |= ((hb >> k) & 1u) << 4;
            if constexpr (FMT == FMT_S6) v |= ((h16 >> (2 * k)) & 3u) << 4;
            q[k] = v;
        }
    } else {
        const uint32_t d0 = r.lo[2 * U8], d1 = r.lo[2 * U8 + 1];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            q[k] = ((d0 >> (8 * k)) & 0xFFu) ^ 0x80u;
            q[4 + k] = ((d1 >> (8 * k)) & 0xFFu) ^ 0x80u;
        }
    }
}

template <int FMT, int U8>
__device__ __forceinline__ void dequant_unit(const RawSub<FMT>& r, float eff, half_t* dst) {
    uint32_t o[4];
    const uint32_t ef2 = h2_splat(eff);
    if constexpr (FMT == FMT_S1_25) {
        // unit U8 = codebook groups 2*U8, 2*U8+1 of this sub-group: a 10-bit field at bit 10*U8 of the 10-byte window
        constexpr int p = 10 * U8, wi = p >> 5, sh = p & 31;
        uint32_t bits;
        if constexpr (sh + 10 <= 32) bits = (r.lo[wi] >> sh) & 0x3FFu;
        else bits = ((r.lo[wi] >> sh) | (r.lo[wi + 1] << (32 - sh))) & 0x3FFu;
        const uint2* lut = s125_lut();
        const uint2 a = lut[bits & 31u], b = lut[bits >> 5];
        o[0] = h2_mul(a.x, ef2);
        o[1] = h2_mul(a.y, ef2);
        o[2] = h2_mul(b.x, ef2);
        o[3] = h2_mul(b.y, ef2);
    } else if constexpr (FMT == FMT_S1_5) {
        const uint32_t t0 = r.hi[0], byte0 = t0 / 5u;
        const uint32_t mag = h2_splat(1025.0f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t ti0 = t0 + 8 * U8 + 2 * i, ti1 = ti0 + 1;
            const uint32_t v0 = (uint32_t)(s15_trit(byte_at4(r.lo, (int)(ti0 / 5u - byte0)), (int)(ti0 % 5u)) + 1);
            const uint32_t v1 = (uint32_t)(s15_trit(byte_at4(r.lo, (int)(ti1 / 5u - byte0)), (int)(ti1 % 5u)) + 1);
            o[i] = h2_mul(h2_sub(0x64006400u | v0 | (v1 << 16), mag), ef2);
        }
    } else {
        uint32_t q[8];
        raw_q8<FMT, U8>(r, q);
        const uint32_t mag = h2_splat(FMT == FMT_S8 ? 1152.0f : 1024.0f);
        const uint32_t hh = h2_splat(sk_half<FMT>());
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t pat = 0x64006400u | q[2 * i] | (q[2 * i + 1] << 16);
            uint32_t w = h2_sub(pat, mag);
            if constexpr (FMT != FMT_S8) w = h2_sub(w, hh);
            o[i] = h2_mul(w, ef2);
        }
    }
    *(uint4*)dst = make_uint4(o[0], o[1], o[2], o[3]);
}

template <int FMT>
__device__ __forceinline__ void dequant_sub(const RawSub<FMT>& r, float eff, half_t* dst) {
    dequant_unit<FMT, 0>(r, eff, dst + 0);
    dequant_unit<FMT, 1>(r, eff, dst + 8);
    dequant_unit<FMT, 2>(r, eff, dst + 16);
    dequant_unit<FMT, 3>(r, eff, dst + 24);
    dequant_unit<FMT, 4>(r, eff, dst + 32);
    dequant_unit<FMT, 5>(r, eff, dst + 40);
    dequant_unit<FMT, 6>(r, eff, dst + 48);
    dequant_unit<FMT, 7>(r, eff, dst + 56);
}

#if !HY4_CUDA
// ---- subgroup-scaled BK=64 WMMA loader (gfx11 Wave32):
// Dequantizes 1 subgroup (64 weights) at a time as exact (q - H) in fp16 (zero fp32 unpack/mul/pack per weight!),
// accumulates 4 WMMAs (64 k-steps) into `sub`, and scales by fp32 `eff` ONCE per subgroup (acc_n = lane & 15).
// Cuts LDS from 33.8 KB (1 blk/CU) to 9.5-27.9 KB (2-3 blk/CU) and eliminates idle waves in small-M / MoE kernels.
constexpr int SG_BK = 64, SG_LD = 72;  // 64 + 8 halves = 144 B (36 dwords -> zero bank conflicts for ds_read_b128 on RDNA3)

template <int FMT>
struct RawQ16 {
    static constexpr int NLO = FMT == FMT_S8 ? 4 : (FMT == FMT_S4 || FMT == FMT_S5 || FMT == FMT_S6) ? 2 : 1;
    uint32_t lo[NLO];
    uint32_t hi;
    float eff;
};

template <int FMT>
__device__ __forceinline__ void load_q16(const uint8_t* blk, int g, int q4, RawQ16<FMT>& r) {
    const uint8_t* pl = blk + 6;
    if constexpr (FMT == FMT_S1_25) {
        load_bytes<1>(pl + g * 10 + ((20 * q4) >> 3), r.lo);
        r.hi = 0;
    } else if constexpr (FMT == FMT_S1_5) {
        const int t0 = g * 64 + q4 * 16;
        load_bytes<1>(pl + t0 / 5, r.lo);
        r.hi = (uint32_t)t0;
    } else if constexpr (FMT == FMT_S2) {
        load_bytes<1>(pl + g * 16 + q4 * 4, r.lo);
        r.hi = 0;
    } else if constexpr (FMT == FMT_S3) {
        load_bytes<1>(pl + g * 16 + q4 * 4, r.lo);
        r.hi = ld16(pl + 64 + g * 8 + q4 * 2);
    } else if constexpr (FMT == FMT_S4) {
        load_bytes<2>(pl + g * 32 + q4 * 8, r.lo);
        r.hi = 0;
    } else if constexpr (FMT == FMT_S5) {
        load_bytes<2>(pl + g * 32 + q4 * 8, r.lo);
        r.hi = ld16(pl + 128 + g * 8 + q4 * 2);
    } else if constexpr (FMT == FMT_S6) {
        load_bytes<2>(pl + g * 32 + q4 * 8, r.lo);
        uint32_t tmp[1];
        load_bytes<1>(pl + 128 + g * 16 + q4 * 4, tmp);
        r.hi = tmp[0];
    } else {
        load_bytes<4>(pl + g * 64 + q4 * 16, r.lo);
        r.hi = 0;
    }
    if (q4 == 0) r.eff = sk_eff<FMT>(blk, g);
}

template <int FMT>
__device__ __forceinline__ void dequant_q16(const RawQ16<FMT>& r, int q4, half_t* dst) {
    if constexpr (FMT == FMT_S1_25) {
        const uint32_t bits = (r.lo[0] >> ((q4 & 1) * 4)) & 0xFFFFFu;
        const uint2* lut = s125_lut();
        const uint2 a = lut[bits & 31u], b = lut[(bits >> 5) & 31u], c = lut[(bits >> 10) & 31u], d = lut[(bits >> 15) & 31u];
        *(uint4*)(dst + 0) = make_uint4(a.x, a.y, b.x, b.y);
        *(uint4*)(dst + 8) = make_uint4(c.x, c.y, d.x, d.y);
        return;
    }
    if constexpr (FMT == FMT_S1_5) {
        const uint32_t t0 = r.hi, byte0 = t0 / 5u;
        const uint32_t mag = h2_splat(1025.0f);
        uint32_t o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t ti0 = t0 + 2 * i, ti1 = ti0 + 1;
            const uint32_t b0 = (r.lo[0] >> ((ti0 / 5u - byte0) * 8)) & 0xFFu;
            const uint32_t b1 = (r.lo[0] >> ((ti1 / 5u - byte0) * 8)) & 0xFFu;
            const uint32_t v0 = (uint32_t)(s15_trit(b0, (int)(ti0 % 5u)) + 1);
            const uint32_t v1 = (uint32_t)(s15_trit(b1, (int)(ti1 % 5u)) + 1);
            o[i] = h2_sub(0x64006400u | v0 | (v1 << 16), mag);
        }
        *(uint4*)(dst + 0) = make_uint4(o[0], o[1], o[2], o[3]);
        *(uint4*)(dst + 8) = make_uint4(o[4], o[5], o[6], o[7]);
        return;
    }
    uint32_t q[16];
    if constexpr (FMT == FMT_S2 || FMT == FMT_S3) {
        const uint32_t w = r.lo[0], hb = FMT == FMT_S3 ? r.hi : 0u;
#pragma unroll
        for (int k = 0; k < 16; ++k) q[k] = ((w >> (2 * k)) & 3u) | (FMT == FMT_S3 ? (((hb >> k) & 1u) << 2) : 0u);
    } else if constexpr (FMT == FMT_S4 || FMT == FMT_S5 || FMT == FMT_S6) {
        const uint32_t w0 = r.lo[0], w1 = r.lo[1], hb = FMT == FMT_S5 ? r.hi : 0u, h2 = FMT == FMT_S6 ? r.hi : 0u;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t v0 = (w0 >> (4 * k)) & 15u, v1 = (w1 >> (4 * k)) & 15u;
            if constexpr (FMT == FMT_S5) { v0 |= ((hb >> k) & 1u) << 4; v1 |= ((hb >> (k + 8)) & 1u) << 4; }
            if constexpr (FMT == FMT_S6) { v0 |= ((h2 >> (2 * k)) & 3u) << 4; v1 |= ((h2 >> (2 * (k + 8))) & 3u) << 4; }
            q[k] = v0;
            q[k + 8] = v1;
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t d = r.lo[j];
#pragma unroll
            for (int k = 0; k < 4; ++k) q[4 * j + k] = ((d >> (8 * k)) & 0xFFu) ^ 0x80u;
        }
    }
    const uint32_t mag = h2_splat(FMT == FMT_S8 ? 1152.0f : 1024.0f);
    const uint32_t hh = h2_splat(sk_half<FMT>());
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t w = h2_sub(0x64006400u | q[2 * i] | (q[2 * i + 1] << 16), mag);
        if constexpr (FMT != FMT_S8) w = h2_sub(w, hh);
        o[i] = w;
    }
    *(uint4*)(dst + 0) = make_uint4(o[0], o[1], o[2], o[3]);
    *(uint4*)(dst + 8) = make_uint4(o[4], o[5], o[6], o[7]);
}

// 128-row BK=128 WMMA mainloop (gfx11 Wave32): loads 128 weight rows x 128 k (2 subgroups) into ws[128][136] (34.8 KB) with packed v_pk_mul_f16 dequant,
// global A register-ring prefetch (PD=4), zero sub-accumulators, 2x fewer barriers than BK=64, and ALL 8 waves active (for BN=128 small-M / MoE BM=16,32).
constexpr int BK128_LD = 136;

template <int FMT>
__device__ __forceinline__ void dequant_q16_scaled(const RawQ16<FMT>& r, int q4, float eff, half_t* dst) {
    const uint32_t ef2 = h2_splat(eff);
    if constexpr (FMT == FMT_S1_25) {
        const uint32_t bits = (r.lo[0] >> ((q4 & 1) * 4)) & 0xFFFFFu;
        const uint2* lut = s125_lut();
        const uint2 a = lut[bits & 31u], b = lut[(bits >> 5) & 31u], c = lut[(bits >> 10) & 31u], d = lut[(bits >> 15) & 31u];
        *(uint4*)(dst + 0) = make_uint4(h2_mul(a.x, ef2), h2_mul(a.y, ef2), h2_mul(b.x, ef2), h2_mul(b.y, ef2));
        *(uint4*)(dst + 8) = make_uint4(h2_mul(c.x, ef2), h2_mul(c.y, ef2), h2_mul(d.x, ef2), h2_mul(d.y, ef2));
        return;
    }
    if constexpr (FMT == FMT_S1_5) {
        const uint32_t t0 = r.hi, byte0 = t0 / 5u;
        const uint32_t mag = h2_splat(1025.0f);
        uint32_t o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t ti0 = t0 + 2 * i, ti1 = ti0 + 1;
            const uint32_t b0 = (r.lo[0] >> ((ti0 / 5u - byte0) * 8)) & 0xFFu;
            const uint32_t b1 = (r.lo[0] >> ((ti1 / 5u - byte0) * 8)) & 0xFFu;
            const uint32_t v0 = (uint32_t)(s15_trit(b0, (int)(ti0 % 5u)) + 1);
            const uint32_t v1 = (uint32_t)(s15_trit(b1, (int)(ti1 % 5u)) + 1);
            o[i] = h2_mul(h2_sub(0x64006400u | v0 | (v1 << 16), mag), ef2);
        }
        *(uint4*)(dst + 0) = make_uint4(o[0], o[1], o[2], o[3]);
        *(uint4*)(dst + 8) = make_uint4(o[4], o[5], o[6], o[7]);
        return;
    }
    uint32_t q[16];
    if constexpr (FMT == FMT_S2 || FMT == FMT_S3) {
        const uint32_t w = r.lo[0], hb = FMT == FMT_S3 ? r.hi : 0u;
#pragma unroll
        for (int k = 0; k < 16; ++k) q[k] = ((w >> (2 * k)) & 3u) | (FMT == FMT_S3 ? (((hb >> k) & 1u) << 2) : 0u);
    } else if constexpr (FMT == FMT_S4 || FMT == FMT_S5 || FMT == FMT_S6) {
        const uint32_t w0 = r.lo[0], w1 = r.lo[1], hb = FMT == FMT_S5 ? r.hi : 0u, h2 = FMT == FMT_S6 ? r.hi : 0u;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t v0 = (w0 >> (4 * k)) & 15u, v1 = (w1 >> (4 * k)) & 15u;
            if constexpr (FMT == FMT_S5) { v0 |= ((hb >> k) & 1u) << 4; v1 |= ((hb >> (k + 8)) & 1u) << 4; }
            if constexpr (FMT == FMT_S6) { v0 |= ((h2 >> (2 * k)) & 3u) << 4; v1 |= ((h2 >> (2 * (k + 8))) & 3u) << 4; }
            q[k] = v0;
            q[k + 8] = v1;
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t d = r.lo[j];
#pragma unroll
            for (int k = 0; k < 4; ++k) q[4 * j + k] = ((d >> (8 * k)) & 0xFFu) ^ 0x80u;
        }
    }
    const uint32_t mag = h2_splat(FMT == FMT_S8 ? 1152.0f : 1024.0f);
    const uint32_t hh = h2_splat(sk_half<FMT>());
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t w = h2_sub(0x64006400u | q[2 * i] | (q[2 * i + 1] << 16), mag);
        if constexpr (FMT != FMT_S8) w = h2_sub(w, hh);
        o[i] = h2_mul(w, ef2);
    }
    *(uint4*)(dst + 0) = make_uint4(o[0], o[1], o[2], o[3]);
    *(uint4*)(dst + 8) = make_uint4(o[4], o[5], o[6], o[7]);
}

// 128-row BK=128 mainloop: 256 threads load 128 rows x 2 subgroups (1 thread loads 1 subgroup of 64 weights = 1 RawSub<FMT>!).
// Wait: 128 rows * 2 subgroups = 256 RawSub! 1 thread = 1 (row, subgroup) -> EXACTLY load_raw<FMT> + dequant_sub<FMT> (zero wasted header loads, 100% coalesced 32-64 B loads)!
template <int FMT, int MTW, int NTW>
__device__ __forceinline__ void gemm_mainloop_128(const uint8_t* wr, int nsb, const half_t* const (&xrow)[MTW], half_t* ws, fv8 (&acc)[MTW][NTW], const int (&brow)[NTW]) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int tid = threadIdx.x, lane = tid & 31, l16 = lane & 15;
    const int drow = tid >> 1, dg_lo = tid & 1;  // 128 rows (0..127), 2 subgroups (0,1 then 2,3)
    if constexpr (FMT == FMT_S1_25) {
        s125_lut_init();
        __syncthreads();
    }
    RawSub<FMT> cur, nxt;
    load_raw<FMT>(wr, dg_lo, cur);
    constexpr int PD = 4;
    const int total = nsb * 16;
    hv16 aq[MTW][PD];
#pragma unroll
    for (int i = 0; i < PD; ++i)
#pragma unroll
        for (int m = 0; m < MTW; ++m) aq[m][i] = ld_hv16(xrow[m] + i * 16);
    const int nhalf = nsb * 2;
    for (int hs = 0; hs < nhalf; ++hs) {
        const int sb = hs >> 1, dg = (hs & 1) * 2 + dg_lo;
        dequant_sub<FMT>(cur, raw_eff<FMT>(cur, dg), &ws[drow * BK128_LD + dg_lo * 64]);
        if (hs + 1 < nhalf) {
            const int nsb_i = (hs + 1) >> 1, ndg = ((hs + 1) & 1) * 2 + dg_lo;
            load_raw<FMT>(wr + (size_t)nsb_i * SB, ndg, nxt);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int g = hs * 8 + kk;
            hv16 a[MTW];
#pragma unroll
            for (int m = 0; m < MTW; ++m) {
                a[m] = aq[m][kk & (PD - 1)];
                if (g + PD < total) aq[m][kk & (PD - 1)] = ld_hv16(xrow[m] + (g + PD) * 16);
            }
            hv16 b[NTW];
#pragma unroll
            for (int j = 0; j < NTW; ++j) b[j] = ld_hv16(&ws[(brow[j] + l16) * BK128_LD + kk * 16]);
#pragma unroll
            for (int m = 0; m < MTW; ++m)
#pragma unroll
                for (int j = 0; j < NTW; ++j) acc[m][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[m], b[j], acc[m][j]);
        }
        __syncthreads();
        cur = nxt;
    }
}

template <int FMT>
__device__ __forceinline__ void gemm_mainloop_old(const uint8_t* wr, int nsb, const half_t* xrow0, const half_t* xrow1, half_t* ws, fv8 (&acc)[2][2],
                                              int b0row, int b1row) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int tid = threadIdx.x, lane = tid & 31;
    const int drow = tid >> 2, dg = tid & 3;
    RawSub<FMT> cur, nxt;
    load_raw<FMT>(wr, dg, cur);
    // A operand (activations) is streamed from global memory with a prefetch distance of 4 k-steps (register ring), so the loads never stall a WMMA
    constexpr int PD = 4;
    const int total = nsb * 16;
    hv16 aq0[PD], aq1[PD];
#pragma unroll
    for (int i = 0; i < PD; ++i) {
        aq0[i] = ld_hv16(xrow0 + i * 16);
        aq1[i] = ld_hv16(xrow1 + i * 16);
    }
    for (int sb = 0; sb < nsb; ++sb) {
        dequant_sub<FMT>(cur, raw_eff<FMT>(cur, dg), &ws[drow * GEMM_LDW + dg * 64]);
        if (sb + 1 < nsb) load_raw<FMT>(wr + (size_t)(sb + 1) * SB, dg, nxt);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            const int g = sb * 16 + kk;
            const hv16 a0 = aq0[kk & (PD - 1)], a1 = aq1[kk & (PD - 1)];
            if (g + PD < total) {
                aq0[kk & (PD - 1)] = ld_hv16(xrow0 + (g + PD) * 16);
                aq1[kk & (PD - 1)] = ld_hv16(xrow1 + (g + PD) * 16);
            }
            hv16 b0 = ld_hv16(&ws[(b0row + (lane & 15)) * GEMM_LDW + kk * 16]);
            hv16 b1 = ld_hv16(&ws[(b1row + (lane & 15)) * GEMM_LDW + kk * 16]);
            acc[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc[0][0]);
            acc[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, acc[0][1]);
            acc[1][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, acc[1][0]);
            acc[1][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc[1][1]);
        }
        __syncthreads();
        cur = nxt;
    }
}
#endif  // !HY4_CUDA

template <int A, int B>
__device__ __forceinline__ void acc_zero(fv8 (&acc)[A][B]) {
#pragma unroll
    for (int i = 0; i < A; ++i)
#pragma unroll
        for (int j = 0; j < B; ++j) acc[i][j] = fv8_zero();
}

// Generic tile main loop: MTW m-tiles x NTW n-tiles of 16 per wave. xrow[i]: A-operand row (this lane: token row (lane&15) of m-tile i);
// brow[j]: LDS tile row base of n-tile j. Waves with active == false only take part in the dequantization and the barriers.
#if !HY4_CUDA
template <int FMT, int MTW, int NTW>
__device__ __forceinline__ void gemm_mainloop_g(const uint8_t* wr, int nsb, const half_t* const (&xrow)[MTW], half_t* ws, fv8 (&acc)[MTW][NTW],
                                                const int (&brow)[NTW], bool active) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int tid = threadIdx.x, lane = tid & 31;
    const int drow = tid >> 2, dg = tid & 3;
    if constexpr (FMT == FMT_S1_25) {
        s125_lut_init();
        __syncthreads();
    }
    RawSub<FMT> cur, nxt;
    load_raw<FMT>(wr, dg, cur);
    constexpr int PD = 4;
    const int total = nsb * 16;
    hv16 aq[MTW][PD];
    if (active) {
#pragma unroll
        for (int i = 0; i < PD; ++i)
#pragma unroll
            for (int m = 0; m < MTW; ++m) aq[m][i] = ld_hv16(xrow[m] + i * 16);
    }
    for (int sb = 0; sb < nsb; ++sb) {
        dequant_sub<FMT>(cur, raw_eff<FMT>(cur, dg), &ws[drow * GEMM_LDW + dg * 64]);
        if (sb + 1 < nsb) load_raw<FMT>(wr + (size_t)(sb + 1) * SB, dg, nxt);
        __syncthreads();
        if (active) {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) {
                const int g = sb * 16 + kk;
                hv16 a[MTW];
#pragma unroll
                for (int m = 0; m < MTW; ++m) {
                    a[m] = aq[m][kk & (PD - 1)];
                    if (g + PD < total) aq[m][kk & (PD - 1)] = ld_hv16(xrow[m] + (g + PD) * 16);
                }
                hv16 b[NTW];
#pragma unroll
                for (int j = 0; j < NTW; ++j) b[j] = ld_hv16(&ws[(brow[j] + (lane & 15)) * GEMM_LDW + kk * 16]);
#pragma unroll
                for (int m = 0; m < MTW; ++m)
#pragma unroll
                    for (int j = 0; j < NTW; ++j) acc[m][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[m], b[j], acc[m][j]);
            }
        }
        __syncthreads();
        cur = nxt;
    }
}

#else  // HY4_CUDA
// Same contract as the gfx11 loop: xrow[m] = this lane's A row (row = lane & 15 of m-tile m); brow[j] = LDS tile row base of n-tile j (16 weight rows).
// A rows g = lane>>2 and g+8 are taken from lanes g and g+8 by pointer shuffle; per 16-k step a lane loads 8 bytes of each A row and 8 bytes of each B row (n = g of the n8 sub-tile).
template <int FMT, int MTW, int NTW>
__device__ __forceinline__ void gemm_mainloop_g(const uint8_t* wr, int nsb, const half_t* const (&xrow)[MTW], half_t* ws, fv8 (&acc)[MTW][NTW], const int (&brow)[NTW], bool active) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int tid = threadIdx.x, lane = tid & 31;
    const int drow = tid >> 2, dg = tid & 3;
    const int g = lane >> 2, c4 = (lane & 3) * 4;
    if constexpr (FMT == FMT_S1_25) {
        s125_lut_init();
        __syncthreads();
    }
    RawSub<FMT> cur, nxt;
    load_raw<FMT>(wr, dg, cur);
    constexpr int PD = 4;
    const int total = nsb * 16;
    const half_t* rlo[MTW];
    const half_t* rhi[MTW];
    uint2 alo[MTW][PD], ahi[MTW][PD];
#pragma unroll
    for (int m = 0; m < MTW; ++m) {
        rlo[m] = (const half_t*)shfl_ptr(xrow[m], g);
        rhi[m] = (const half_t*)shfl_ptr(xrow[m], g + 8);
    }
    if (active) {
#pragma unroll
        for (int i = 0; i < PD; ++i)
#pragma unroll
            for (int m = 0; m < MTW; ++m) {
                alo[m][i] = *(const uint2*)(rlo[m] + i * 16 + c4);
                ahi[m][i] = *(const uint2*)(rhi[m] + i * 16 + c4);
            }
    }
    for (int sb = 0; sb < nsb; ++sb) {
        dequant_sub<FMT>(cur, raw_eff<FMT>(cur, dg), &ws[drow * GEMM_LDW + dg * 64]);
        if (sb + 1 < nsb) load_raw<FMT>(wr + (size_t)(sb + 1) * SB, dg, nxt);
        __syncthreads();
        if (active) {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) {
                const int gi = sb * 16 + kk;
                uint2 a_lo[MTW], a_hi[MTW];
#pragma unroll
                for (int m = 0; m < MTW; ++m) {
                    a_lo[m] = alo[m][kk & (PD - 1)];
                    a_hi[m] = ahi[m][kk & (PD - 1)];
                    if (gi + PD < total) {
                        alo[m][kk & (PD - 1)] = *(const uint2*)(rlo[m] + (gi + PD) * 16 + c4);
                        ahi[m][kk & (PD - 1)] = *(const uint2*)(rhi[m] + (gi + PD) * 16 + c4);
                    }
                }
#pragma unroll
                for (int j = 0; j < NTW; ++j)
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        const uint2 b = *(const uint2*)&ws[(brow[j] + s * 8 + g) * GEMM_LDW + kk * 16 + c4];
#pragma unroll
                        for (int m = 0; m < MTW; ++m) mma_16816(&acc[m][j].v[s * 4], a_lo[m].x, a_hi[m].x, a_lo[m].y, a_hi[m].y, b.x, b.y);
                    }
            }
        }
        __syncthreads();
        cur = nxt;
    }
}
#endif

template <int FMT>
__device__ __forceinline__ void gemm_mainloop(const uint8_t* wr, int nsb, const half_t* xrow0, const half_t* xrow1, half_t* ws, fv8 (&acc)[2][2],
                                              int b0row, int b1row) {
    const half_t* xr[2] = {xrow0, xrow1};
    const int br[2] = {b0row, b1row};
    gemm_mainloop_g<FMT, 2, 2>(wr, nsb, xr, ws, acc, br, true);
}

// small-M linear kernel (M <= 16 tokens per block row):
// On gfx11 (HIP): 128 weight rows per block, ALL 8 waves active (16 rows each), BK=128 (2 subgroups/step, 1 thread = 1 subgroup = 64 weights, v_pk_mul_f16).
#if !HY4_CUDA
constexpr int GEMMS_BN = 128;
template <int FMT>
__global__ void __launch_bounds__(256) k_gemm_s(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    __shared__ __attribute__((aligned(16))) half_t ws[GEMMS_BN * BK128_LD];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int bn = blockIdx.x * GEMMS_BN, bm = blockIdx.y * 16;
    const int nsb = K >> 8;
    int wrow = bn + (tid >> 1);
    wrow = wrow < N ? wrow : N - 1;
    int row = bm + (lane & 15);
    row = row < M ? row : M - 1;
    const half_t* xrow[1] = {(const half_t*)X + (size_t)row * K};
    fv8 acc[1][1];
    acc_zero(acc);
    const int brow[1] = {wave * 16};
    gemm_mainloop_128<FMT, 1, 1>(W + (size_t)wrow * nsb * SB, nsb, xrow, ws, acc, brow);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int n = bn + wave * 16 + acc_n(r, lane);
        const int m = bm + acc_m(r, lane);
        if (m < M && n < N) Y[(size_t)m * N + n] = acc[0][0][r];
    }
}

template <int FMT>
__global__ void __launch_bounds__(256) k_gemm(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    __shared__ __attribute__((aligned(16))) half_t ws[GEMM_BN * GEMM_LDW];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave & 3, wn = wave >> 2;
    const int bn = blockIdx.x * GEMM_BN, bm = blockIdx.y * GEMM_BM;
    const int nsb = K >> 8;
    fv8 acc[2][2];
    acc_zero(acc);
    const half_t* xrow[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int row = bm + wm * 32 + i * 16 + (lane & 15);
        row = row < M ? row : M - 1;
        xrow[i] = (const half_t*)X + (size_t)row * K;
    }
    int wrow = bn + (tid >> 2);
    wrow = wrow < N ? wrow : N - 1;
    gemm_mainloop<FMT>(W + (size_t)wrow * nsb * SB, nsb, xrow[0], xrow[1], ws, acc, wn * 32, wn * 32 + 16);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int n = bn + wn * 32 + j * 16 + acc_n(r, lane);
                const int m = bm + wm * 32 + i * 16 + acc_m(r, lane);
                if (m < M && n < N) Y[(size_t)m * N + n] = acc[i][j][r];
            }
        }
}
#else  // HY4_CUDA
constexpr int GEMMS_BN = GEMM_BN;
template <int FMT>
__global__ void __launch_bounds__(256) k_gemm_s(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    __shared__ __attribute__((aligned(16))) half_t ws[GEMM_BN * GEMM_LDW];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int bn = blockIdx.x * GEMM_BN, bm = blockIdx.y * 16;
    const int nsb = K >> 8;
    fv8 acc[1][1];
    acc_zero(acc);
    int row = bm + (lane & 15);
    row = row < M ? row : M - 1;
    const half_t* xrow[1] = {(const half_t*)X + (size_t)row * K};
    int wrow = bn + (tid >> 2);
    wrow = wrow < N ? wrow : N - 1;
    const int brow[1] = {wave * 16};
    gemm_mainloop_g<FMT, 1, 1>(W + (size_t)wrow * nsb * SB, nsb, xrow, ws, acc, brow, wave < 4);
    if (wave < 4) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int n = bn + wave * 16 + acc_n(r, lane);
            const int m = bm + acc_m(r, lane);
            if (m < M && n < N) Y[(size_t)m * N + n] = acc[0][0][r];
        }
    }
}

template <int FMT>
__global__ void __launch_bounds__(256) k_gemm(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    __shared__ __attribute__((aligned(16))) half_t ws[GEMM_BN * GEMM_LDW];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave & 3, wn = wave >> 2;
    const int bn = blockIdx.x * GEMM_BN, bm = blockIdx.y * GEMM_BM;
    const int nsb = K >> 8;
    fv8 acc[2][2];
    acc_zero(acc);
    const half_t* xrow[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int row = bm + wm * 32 + i * 16 + (lane & 15);
        row = row < M ? row : M - 1;
        xrow[i] = (const half_t*)X + (size_t)row * K;
    }
    int wrow = bn + (tid >> 2);
    wrow = wrow < N ? wrow : N - 1;
    gemm_mainloop<FMT>(W + (size_t)wrow * nsb * SB, nsb, xrow[0], xrow[1], ws, acc, wn * 32, wn * 32 + 16);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int n = bn + wn * 32 + j * 16 + acc_n(r, lane);
                const int m = bm + wm * 32 + i * 16 + acc_m(r, lane);
                if (m < M && n < N) Y[(size_t)m * N + n] = acc[i][j][r];
            }
        }
}
#endif

}  // namespace skd

// small-M WMMA entry (M <= 16 per call; larger M loops over 16-token blocks inside the grid): S_K weights, same requirements as linear_h_gemm
inline void linear_h_gemm_s(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    dim3 grid((N + skd::GEMMS_BN - 1) / skd::GEMMS_BN, (M + 15) / 16), block(256);
#define HY4_GEMMS_CASE(F) case F: hipLaunchKernelGGL((skd::k_gemm_s<F>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M); break
    switch (fmt) {
        HY4_GEMMS_CASE(FMT_S1_25);
        HY4_GEMMS_CASE(FMT_S1_5);
        HY4_GEMMS_CASE(FMT_S2);
        HY4_GEMMS_CASE(FMT_S3);
        HY4_GEMMS_CASE(FMT_S4);
        HY4_GEMMS_CASE(FMT_S5);
        HY4_GEMMS_CASE(FMT_S6);
        HY4_GEMMS_CASE(FMT_S8);
        default: fprintf(stderr, "linear_h_gemm_s: unsupported fmt %d\n", fmt); abort();
    }
#undef HY4_GEMMS_CASE
}

// production prefill entry: S_K weights, M >= 1 (best for M >= 16), K % 256 == 0, X rows 32-byte aligned (K % 16 == 0 and 16-byte aligned base)
inline void linear_h_gemm(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    dim3 grid((N + skd::GEMM_BN - 1) / skd::GEMM_BN, (M + skd::GEMM_BM - 1) / skd::GEMM_BM), block(256);
#define HY4_GEMM_CASE(F) case F: hipLaunchKernelGGL((skd::k_gemm<F>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M); break
    switch (fmt) {
        HY4_GEMM_CASE(FMT_S1_25);
        HY4_GEMM_CASE(FMT_S1_5);
        HY4_GEMM_CASE(FMT_S2);
        HY4_GEMM_CASE(FMT_S3);
        HY4_GEMM_CASE(FMT_S4);
        HY4_GEMM_CASE(FMT_S5);
        HY4_GEMM_CASE(FMT_S6);
        HY4_GEMM_CASE(FMT_S8);
        default: fprintf(stderr, "linear_h_gemm: unsupported fmt %d\n", fmt); abort();
    }
#undef HY4_GEMM_CASE
}

}  // namespace hy4
#endif  // HY4_MMA
