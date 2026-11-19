// skwmma.hip.h - prefill GEMM (M >= 16) for S_K weights with gfx11 WMMA (v_wmma_f32_16x16x16_f16, wave32)
//
// Y[m][n] = sum_k W[n][k] X[m][k];  W S_K wire format [N][K] (K % 256 == 0), X __half [M][K], Y float [M][N] (row-major).
// Block tile 128 tokens x 64 weight rows, 8 waves (4 x 2), wave tile 32 x 32 (2 x 2 WMMA tiles). Per super-block (256 k) the 64 weight rows are
// dequantized once to fp16 in LDS (w = (q-H) * D*u/255 computed in fp32, rounded to fp16) and reused by all 128 tokens.
// A operand (X) is read straight from global memory (L1/L2 reuse), B operand (W tile) from padded LDS.
// gemm_mainloop is shared with the grouped expert kernels (skmoe.hip.h).
#pragma once
#include "skfast.hip.h"

namespace hy4 {
namespace skd {

typedef _Float16 hv16 __attribute__((ext_vector_type(16)));
typedef float fv8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ hv16 ld_hv16(const void* p) {
    const uint4 a = *(const uint4*)p, b = *((const uint4*)p + 1);
    typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
    u32x8 v = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(hv16, v);
}

__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    hv2 h = {(_Float16)a, (_Float16)b};
    return __builtin_bit_cast(uint32_t, h);
}
__device__ __forceinline__ void unpack_h2(uint32_t w, float& a, float& b) {
    hv2 h = __builtin_bit_cast(hv2, w);
    a = (float)h[0];
    b = (float)h[1];
}

// dequantize the 8 weights 8u..8u+7 of super-block `blk` scaled by eff into 4 packed half2 words
template <int FMT>
__device__ __forceinline__ void dequant8_h(const uint8_t* blk, int u, float eff, uint4& out) {
    uint32_t wp[4];
    decode_w8<FMT>(blk + 6, u, wp);
    uint32_t r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a, b;
        unpack_h2(wp[i], a, b);
        r[i] = pack_h2(a * eff, b * eff);
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
    static constexpr int NLO = (FMT == FMT_S2 || FMT == FMT_S3) ? 4 : (FMT == FMT_S8 ? 16 : 8);
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
    for (int i = 0; i < N; ++i) o[i] = __builtin_amdgcn_alignbyte(d[i + 1], d[i], ofs);
}

template <int FMT>
__device__ __forceinline__ void load_raw(const uint8_t* blk, int dg, RawSub<FMT>& r) {
    load_bytes<2>(blk, r.hdr);
    const uint8_t* pl = blk + 6;
    if constexpr (FMT == FMT_S2) load_bytes<4>(pl + dg * 16, r.lo);
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
    return D * ((float)u * (1.0f / 255.0f));
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
__device__ __forceinline__ void dequant_unit(const RawSub<FMT>& r, float eff, _Float16* dst) {
    uint32_t q[8];
    raw_q8<FMT, U8>(r, q);
    const hv2 mag = FMT == FMT_S8 ? hv2{(_Float16)1152.0f, (_Float16)1152.0f} : hv2{(_Float16)1024.0f, (_Float16)1024.0f};
    const hv2 hh = hv2{(_Float16)sk_half<FMT>(), (_Float16)sk_half<FMT>()};
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t pat = 0x64006400u | q[2 * i] | (q[2 * i + 1] << 16);
        hv2 w = __builtin_bit_cast(hv2, pat) - mag;
        if constexpr (FMT != FMT_S8) w = w - hh;
        o[i] = pack_h2((float)w[0] * eff, (float)w[1] * eff);
    }
    *(uint4*)dst = make_uint4(o[0], o[1], o[2], o[3]);
}

template <int FMT>
__device__ __forceinline__ void dequant_sub(const RawSub<FMT>& r, float eff, _Float16* dst) {
    dequant_unit<FMT, 0>(r, eff, dst + 0);
    dequant_unit<FMT, 1>(r, eff, dst + 8);
    dequant_unit<FMT, 2>(r, eff, dst + 16);
    dequant_unit<FMT, 3>(r, eff, dst + 24);
    dequant_unit<FMT, 4>(r, eff, dst + 32);
    dequant_unit<FMT, 5>(r, eff, dst + 40);
    dequant_unit<FMT, 6>(r, eff, dst + 48);
    dequant_unit<FMT, 7>(r, eff, dst + 56);
}

template <int FMT>
__device__ __forceinline__ void gemm_mainloop_old(const uint8_t* wr, int nsb, const _Float16* xrow0, const _Float16* xrow1, _Float16* ws, fv8 (&acc)[2][2],
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


template <int A, int B>
__device__ __forceinline__ void acc_zero(fv8 (&acc)[A][B]) {
#pragma unroll
    for (int i = 0; i < A; ++i)
#pragma unroll
        for (int j = 0; j < B; ++j) acc[i][j] = fv8{0, 0, 0, 0, 0, 0, 0, 0};
}

// Generic tile main loop: MTW m-tiles x NTW n-tiles of 16 per wave. xrow[i]: A-operand row (this lane: token row (lane&15) of m-tile i);
// brow[j]: LDS tile row base of n-tile j. Waves with active == false only take part in the dequantization and the barriers.
template <int FMT, int MTW, int NTW>
__device__ __forceinline__ void gemm_mainloop_g(const uint8_t* wr, int nsb, const _Float16* const (&xrow)[MTW], _Float16* ws, fv8 (&acc)[MTW][NTW],
                                                const int (&brow)[NTW], bool active) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int tid = threadIdx.x, lane = tid & 31;
    const int drow = tid >> 2, dg = tid & 3;
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

template <int FMT>
__device__ __forceinline__ void gemm_mainloop(const uint8_t* wr, int nsb, const _Float16* xrow0, const _Float16* xrow1, _Float16* ws, fv8 (&acc)[2][2],
                                              int b0row, int b1row) {
    const _Float16* xr[2] = {xrow0, xrow1};
    const int br[2] = {b0row, b1row};
    gemm_mainloop_g<FMT, 2, 2>(wr, nsb, xr, ws, acc, br, true);
}

// small-M linear kernel (M <= 16 tokens per block row): 64 weight rows per block, 4 active waves (16 rows each), one 16-token m-tile
template <int FMT>
__global__ void __launch_bounds__(256) k_gemm_s(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    __shared__ __attribute__((aligned(16))) _Float16 ws[GEMM_BN * GEMM_LDW];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int bn = blockIdx.x * GEMM_BN, bm = blockIdx.y * 16;
    const int nsb = K >> 8;
    fv8 acc[1][1];
    acc_zero(acc);
    int row = bm + (lane & 15);
    row = row < M ? row : M - 1;
    const _Float16* xrow[1] = {(const _Float16*)X + (size_t)row * K};
    int wrow = bn + (tid >> 2);
    wrow = wrow < N ? wrow : N - 1;
    const int brow[1] = {wave * 16};
    gemm_mainloop_g<FMT, 1, 1>(W + (size_t)wrow * nsb * SB, nsb, xrow, ws, acc, brow, wave < 4);
    if (wave < 4) {
        const int n = bn + wave * 16 + (lane & 15);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int m = bm + 2 * r + (lane >> 4);
            if (m < M && n < N) Y[(size_t)m * N + n] = acc[0][0][r];
        }
    }
}

template <int FMT>
__global__ void __launch_bounds__(256) k_gemm(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    __shared__ __attribute__((aligned(16))) _Float16 ws[GEMM_BN * GEMM_LDW];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave & 3, wn = wave >> 2;
    const int bn = blockIdx.x * GEMM_BN, bm = blockIdx.y * GEMM_BM;
    const int nsb = K >> 8;
    fv8 acc[2][2];
    acc_zero(acc);
    const _Float16* xrow[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int row = bm + wm * 32 + i * 16 + (lane & 15);
        row = row < M ? row : M - 1;
        xrow[i] = (const _Float16*)X + (size_t)row * K;
    }
    int wrow = bn + (tid >> 2);
    wrow = wrow < N ? wrow : N - 1;
    gemm_mainloop<FMT>(W + (size_t)wrow * nsb * SB, nsb, xrow[0], xrow[1], ws, acc, wn * 32, wn * 32 + 16);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int n = bn + wn * 32 + j * 16 + (lane & 15);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int m = bm + wm * 32 + i * 16 + 2 * r + (lane >> 4);
                if (m < M && n < N) Y[(size_t)m * N + n] = acc[i][j][r];
            }
        }
}

}  // namespace skd

// small-M WMMA entry (M <= 16 per call; larger M loops over 16-token blocks inside the grid): S_K weights, same requirements as linear_h_gemm
inline void linear_h_gemm_s(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    dim3 grid((N + skd::GEMM_BN - 1) / skd::GEMM_BN, (M + 15) / 16), block(256);
#define HY4_GEMMS_CASE(F) case F: hipLaunchKernelGGL((skd::k_gemm_s<F>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M); break
    switch (fmt) {
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
