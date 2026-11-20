// sk_int8_cuda.hip.h - Native CUDA INT8 Tensor Core (mma.sync.m16n8k32.s32.s8.s8.s32) and __dp4a S_K x Q8_1_G64 kernels.
//
// Exact identity for S2_K..S6_K (q in [0..63] fits in signed int8) and S8_K (q in [-127..127], H = 0):
//   w_k = (q_k - H) * eff_g
//   dot_g = eff_g * (d_a * sum(q_k * qa_k) + d_b * sum(q_k * qb_k) - H * s_g)
// where s_g = d_a * sum(qa_k) [+ d_b * sum(qb_k) in 2-term hi mode].
//
// Advantages over FP16 mma_16816:
//   - Zero floating-point weight dequantization (pure integer bit unpacking into int8 LDS tile).
//   - Half the LDS weight footprint (18.5 KB vs 34 KB -> higher occupancy).
//   - 1-term Q8_1 mode: 2x Tensor Core TOPS on sm_80+ (Ampere/Ada/Blackwell).
//   - 2-term HI mode: ~3e-5 relative accuracy (more accurate than FP16) while still avoiding FP16 weight dequant ALU and halving LDS.
//   - __dp4a GEMV path (sm_61+): 100% batch-invariant for M = 1..8 decode & verify, quantizes X on-the-fly in LDS.
#pragma once
#include "skwmma.hip.h"

#if HY4_CUDA && HY4_MMA

namespace hy4 {
namespace skd {

// ---- GPU activation quantizer: __half [M][K] -> Q8_1_G64 (1-term or 2-term hi)
// Layout:
//   Xq  : int8_t [M][K]          (primary int8 activations, 64-byte aligned per sub-group)
//   Xds : float2 [M][K/64]       (.x = da, .y = s = da*sum(qa) [+ db*sum(qb) if HI])
//   Xq2 : int8_t [M][K]          (residual int8 activations, only when HI)
//   Xd2 : float  [M][K/64]       (db, only when HI)
template <bool HI>
__global__ void __launch_bounds__(256) k_quant_x_q8(const __half* __restrict__ X, int M, int K,
                                                    int8_t* __restrict__ Xq, float2* __restrict__ Xds,
                                                    int8_t* __restrict__ Xq2, float* __restrict__ Xd2) {
    // 8 warps per block, each warp quantizes one 64-element sub-group (2 elements per lane)
    const int ng = K >> 6;
    const int warp_global = (int)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int m = blockIdx.y;
    if (m >= M || warp_global >= ng) return;
    const int g = warp_global;
    const __half2 h2 = *(const __half2*)(X + (size_t)m * K + g * 64 + lane * 2);
    const float2 f = __half22float2(h2);
    float amax = fmaxf(fabsf(f.x), fabsf(f.y));
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o));
    const float da = amax * (1.0f / 127.0f);
    const float inva = da > 0.f ? (127.0f / amax) : 0.f;
    int q0 = __float2int_rn(f.x * inva);
    int q1 = __float2int_rn(f.y * inva);
    q0 = max(-127, min(127, q0));
    q1 = max(-127, min(127, q1));
    int s0 = q0 + q1;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) s0 += __shfl_xor(s0, o);
    int8_t* dst = Xq + (size_t)m * K + g * 64;
    dst[lane * 2] = (int8_t)q0;
    dst[lane * 2 + 1] = (int8_t)q1;
    float s_tot = da * (float)s0;
    if constexpr (HI) {
        const float r0 = f.x - da * (float)q0;
        const float r1 = f.y - da * (float)q1;
        float rmax = fmaxf(fabsf(r0), fabsf(r1));
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) rmax = fmaxf(rmax, __shfl_xor(rmax, o));
        const float db = rmax * (1.0f / 127.0f);
        const float invb = db > 0.f ? (127.0f / rmax) : 0.f;
        int qb0 = __float2int_rn(r0 * invb);
        int qb1 = __float2int_rn(r1 * invb);
        qb0 = max(-127, min(127, qb0));
        qb1 = max(-127, min(127, qb1));
        int sb = qb0 + qb1;
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) sb += __shfl_xor(sb, o);
        int8_t* dst2 = Xq2 + (size_t)m * K + g * 64;
        dst2[lane * 2] = (int8_t)qb0;
        dst2[lane * 2 + 1] = (int8_t)qb1;
        s_tot += db * (float)sb;
        if (lane == 0) Xd2[(size_t)m * ng + g] = db;
    }
    if (lane == 0) Xds[(size_t)m * ng + g] = make_float2(da, s_tot);
}

// ---- Pure-integer S_K sub-group unpack (64 weights -> 16 uint32_t of packed int8_t in LDS).
// Zero FP instructions! S2..S6 produce u8 in [0..63] (valid int8_t); S8 copies signed int8_t directly.
template <int FMT>
__device__ __forceinline__ void unpack_sub_i8(const RawSub<FMT>& r, uint32_t* dst16) {
    if constexpr (FMT == FMT_S8) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            ((uint4*)dst16)[i] = make_uint4(r.lo[4 * i], r.lo[4 * i + 1], r.lo[4 * i + 2], r.lo[4 * i + 3]);
    } else if constexpr (FMT == FMT_S2 || FMT == FMT_S3) {
        // r.lo[0..3]: 4 dwords = 16 bytes = 64 x 2-bit weights; r.hi[0..1]: 8 bytes = 64 x 1-bit high bits (S3)
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const uint32_t w = r.lo[wi];
            uint32_t h16 = 0;
            if constexpr (FMT == FMT_S3) h16 = (r.hi[wi >> 1] >> (16 * (wi & 1))) & 0xFFFFu;
            uint32_t o[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint32_t by = (w >> (8 * b)) & 0xFFu;
                uint32_t p = (by & 0x03u) | ((by & 0x0Cu) << 6) | ((by & 0x30u) << 12) | ((by & 0xC0u) << 18);
                if constexpr (FMT == FMT_S3) {
                    const uint32_t hb = (h16 >> (4 * b)) & 0xFu;
                    p |= ((hb & 1u) << 2) | ((hb & 2u) << 9) | ((hb & 4u) << 16) | ((hb & 8u) << 23);
                }
                o[b] = p;
            }
            ((uint4*)dst16)[wi] = make_uint4(o[0], o[1], o[2], o[3]);
        }
    } else {
        // S4, S5, S6: r.lo[0..7] = 32 bytes = 64 x 4-bit nibbles
        // S5: r.hi[0..1] = 8 bytes = 64 x 1-bit high bits; S6: r.hi[0..3] = 16 bytes = 64 x 2-bit high bits
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint32_t o[4];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int wi = 2 * q + k;
                const uint32_t w = r.lo[wi];  // 8 nibbles -> 2 packed uint32_t (o[2k], o[2k+1])
                uint32_t lo4 = (w & 0x0000000Fu) | ((w & 0x000000F0u) << 4) | ((w & 0x00000F00u) << 8) | ((w & 0x0000F000u) << 12);
                uint32_t hi4 = ((w >> 16) & 0x0000000Fu) | (((w >> 16) & 0x000000F0u) << 4) | (((w >> 16) & 0x00000F00u) << 8) | (((w >> 16) & 0x0000F000u) << 12);
                if constexpr (FMT == FMT_S5) {
                    const uint32_t hb = (r.hi[wi >> 2] >> (8 * (wi & 3))) & 0xFFu;
                    lo4 |= ((hb & 0x01u) << 4) | ((hb & 0x02u) << 11) | ((hb & 0x04u) << 18) | ((hb & 0x08u) << 25);
                    hi4 |= ((hb & 0x10u) << 0) | ((hb & 0x20u) << 7) | ((hb & 0x40u) << 14) | ((hb & 0x80u) << 21);
                } else if constexpr (FMT == FMT_S6) {
                    const uint32_t h16 = (r.hi[wi >> 1] >> (16 * (wi & 1))) & 0xFFFFu;
                    const uint32_t hlo = h16 & 0xFFu, hhi = h16 >> 8;
                    lo4 |= ((hlo & 0x03u) << 4) | ((hlo & 0x0Cu) << 10) | ((hlo & 0x30u) << 16) | ((hlo & 0xC0u) << 22);
                    hi4 |= ((hhi & 0x03u) << 4) | ((hhi & 0x0Cu) << 10) | ((hhi & 0x30u) << 16) | ((hhi & 0xC0u) << 22);
                }
                o[2 * k] = lo4;
                o[2 * k + 1] = hi4;
            }
            ((uint4*)dst16)[q] = make_uint4(o[0], o[1], o[2], o[3]);
        }
    }
}

// ---- Native INT8 Tensor Core GEMM mainloop (mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32)
// BN_W * BK == 16384:
//   (64, 256)  -> LDW32 = 68 (272 B/row), ws_q = 17.4 KB, ws_eff = 1.0 KB  (total 18.4 KB)
//   (128, 128) -> LDW32 = 36 (144 B/row), ws_q = 18.4 KB, ws_eff = 1.0 KB  (total 19.4 KB)
//   (256, 64)  -> LDW32 = 20 ( 80 B/row), ws_q = 20.5 KB, ws_eff = 1.0 KB  (total 21.5 KB)
template <int FMT, int MTW, int NTW, int BN_W, int BK, bool HI>
__device__ __forceinline__ void gemm_mainloop_i8(const uint8_t* wr, int nsb,
                                                 const int8_t* const (&xq_row)[MTW], const float2* const (&xds_row)[MTW],
                                                 const int8_t* const (&xq2_row)[MTW], const float* const (&xd2_row)[MTW],
                                                 uint32_t* ws_q, float* ws_eff,
                                                 fv8 (&acc)[MTW][NTW], const int (&brow)[NTW]) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int STAGES = 256 / BK;
    constexpr int G_PER_ROW = BK / 64;
    constexpr int LDW32 = (BK >> 2) + 4;  // uint32_t stride per weight row (16-byte aligned, bank-conflict free)
    constexpr float H = sk_half<FMT>();
    const int tid = threadIdx.x, lane = tid & 31;
    const int drow = tid / G_PER_ROW, gl = tid % G_PER_ROW;
    const int g = lane >> 2, c = lane & 3;
    const int nstages = nsb * STAGES;

    RawSub<FMT> cur, nxt;
    load_raw<FMT>(wr, gl, cur);

    const uint32_t* alo_p[MTW];
    const uint32_t* ahi_p[MTW];
    const float2* dlo_p[MTW];
    const float2* dhi_p[MTW];
    const uint32_t* b2lo_p[MTW];
    const uint32_t* b2hi_p[MTW];
    const float* d2lo_p[MTW];
    const float* d2hi_p[MTW];
#pragma unroll
    for (int m = 0; m < MTW; ++m) {
        alo_p[m] = (const uint32_t*)shfl_ptr(xq_row[m], g);
        ahi_p[m] = (const uint32_t*)shfl_ptr(xq_row[m], g + 8);
        dlo_p[m] = (const float2*)shfl_ptr(xds_row[m], g);
        dhi_p[m] = (const float2*)shfl_ptr(xds_row[m], g + 8);
        if constexpr (HI) {
            b2lo_p[m] = (const uint32_t*)shfl_ptr(xq2_row[m], g);
            b2hi_p[m] = (const uint32_t*)shfl_ptr(xq2_row[m], g + 8);
            d2lo_p[m] = (const float*)shfl_ptr(xd2_row[m], g);
            d2hi_p[m] = (const float*)shfl_ptr(xd2_row[m], g + 8);
        }
    }

    for (int st = 0; st < nstages; ++st) {
        const int dg = (st % STAGES) * G_PER_ROW + gl;
        unpack_sub_i8<FMT>(cur, &ws_q[drow * LDW32 + gl * 16]);
        ws_eff[drow * G_PER_ROW + gl] = raw_eff<FMT>(cur, dg);
        if (st + 1 < nstages) {
            const int nst = st + 1;
            const int nsb_i = nst / STAGES;
            const int ndg = (nst % STAGES) * G_PER_ROW + gl;
            load_raw<FMT>(wr + (size_t)nsb_i * SB, ndg, nxt);
        }
        __syncthreads();

#pragma unroll
        for (int sg = 0; sg < G_PER_ROW; ++sg) {
            const int g_global = st * G_PER_ROW + sg;
            // Load A fragments (64 bytes = 2 x m16n8k32 steps) and scales for all MTW m-tiles once
            uint32_t a0_0[MTW], a1_0[MTW], a2_0[MTW], a3_0[MTW];
            uint32_t a0_1[MTW], a1_1[MTW], a2_1[MTW], a3_1[MTW];
            float2 ds_lo[MTW], ds_hi[MTW];
#pragma unroll
            for (int m = 0; m < MTW; ++m) {
                const uint32_t* p0 = alo_p[m] + g_global * 16;
                const uint32_t* p1 = ahi_p[m] + g_global * 16;
                a0_0[m] = p0[c];      a2_0[m] = p0[4 + c];
                a1_0[m] = p1[c];      a3_0[m] = p1[4 + c];
                a0_1[m] = p0[8 + c];  a2_1[m] = p0[12 + c];
                a1_1[m] = p1[8 + c];  a3_1[m] = p1[12 + c];
                ds_lo[m] = dlo_p[m][g_global];
                ds_hi[m] = dhi_p[m][g_global];
            }
            uint32_t q2_00[MTW], q2_10[MTW], q2_20[MTW], q2_30[MTW];
            uint32_t q2_01[MTW], q2_11[MTW], q2_21[MTW], q2_31[MTW];
            float d2_lo[MTW], d2_hi[MTW];
            if constexpr (HI) {
#pragma unroll
                for (int m = 0; m < MTW; ++m) {
                    const uint32_t* p0 = b2lo_p[m] + g_global * 16;
                    const uint32_t* p1 = b2hi_p[m] + g_global * 16;
                    q2_00[m] = p0[c];      q2_20[m] = p0[4 + c];
                    q2_10[m] = p1[c];      q2_30[m] = p1[4 + c];
                    q2_01[m] = p0[8 + c];  q2_21[m] = p0[12 + c];
                    q2_11[m] = p1[8 + c];  q2_31[m] = p1[12 + c];
                    d2_lo[m] = d2lo_p[m][g_global];
                    d2_hi[m] = d2hi_p[m][g_global];
                }
            }
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const uint32_t* bp = &ws_q[(brow[j] + s * 8 + g) * LDW32 + sg * 16];
                    const uint32_t b0_0 = bp[c],     b1_0 = bp[4 + c];
                    const uint32_t b0_1 = bp[8 + c], b1_1 = bp[12 + c];
                    const float e0 = ws_eff[(brow[j] + s * 8 + 2 * c) * G_PER_ROW + sg];
                    const float e1 = ws_eff[(brow[j] + s * 8 + 2 * c + 1) * G_PER_ROW + sg];
#pragma unroll
                    for (int m = 0; m < MTW; ++m) {
                        int32_t id[4] = {0, 0, 0, 0};
                        mma_16832_s8s8(id, a0_0[m], a1_0[m], a2_0[m], a3_0[m], b0_0, b1_0);
                        mma_16832_s8s8(id, a0_1[m], a1_1[m], a2_1[m], a3_1[m], b0_1, b1_1);
                        float v0 = ds_lo[m].x * (float)id[0];
                        float v1 = ds_lo[m].x * (float)id[1];
                        float v2 = ds_hi[m].x * (float)id[2];
                        float v3 = ds_hi[m].x * (float)id[3];
                        if constexpr (HI) {
                            int32_t id2[4] = {0, 0, 0, 0};
                            mma_16832_s8s8(id2, q2_00[m], q2_10[m], q2_20[m], q2_30[m], b0_0, b1_0);
                            mma_16832_s8s8(id2, q2_01[m], q2_11[m], q2_21[m], q2_31[m], b0_1, b1_1);
                            v0 = fmaf(d2_lo[m], (float)id2[0], v0);
                            v1 = fmaf(d2_lo[m], (float)id2[1], v1);
                            v2 = fmaf(d2_hi[m], (float)id2[2], v2);
                            v3 = fmaf(d2_hi[m], (float)id2[3], v3);
                        }
                        if constexpr (FMT != FMT_S8) {
                            const float hs_lo = H * ds_lo[m].y;
                            const float hs_hi = H * ds_hi[m].y;
                            v0 -= hs_lo; v1 -= hs_lo;
                            v2 -= hs_hi; v3 -= hs_hi;
                        }
                        acc[m][j].v[s * 4 + 0] = fmaf(e0, v0, acc[m][j].v[s * 4 + 0]);
                        acc[m][j].v[s * 4 + 1] = fmaf(e1, v1, acc[m][j].v[s * 4 + 1]);
                        acc[m][j].v[s * 4 + 2] = fmaf(e0, v2, acc[m][j].v[s * 4 + 2]);
                        acc[m][j].v[s * 4 + 3] = fmaf(e1, v3, acc[m][j].v[s * 4 + 3]);
                    }
                }
            }
        }
        __syncthreads();
        cur = nxt;
    }
}

// ---- Small-M INT8 Tensor Core GEMM (M <= 16, BN = 128, BK = 128, all 8 warps active, 19.4 KB LDS)
template <int FMT, bool HI>
__global__ void __launch_bounds__(256) k_gemm_i8_s(const uint8_t* __restrict__ W, int N, int K,
                                                   const int8_t* __restrict__ Xq, const float2* __restrict__ Xds,
                                                   const int8_t* __restrict__ Xq2, const float* __restrict__ Xd2,
                                                   float* __restrict__ Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int BN_W = 128, BK = 128, LDW32 = 36, GPR = 2;
    __shared__ __align__(16) uint32_t ws_q[BN_W * LDW32];
    __shared__ float ws_eff[BN_W * GPR];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int bn = blockIdx.x * BN_W, bm = blockIdx.y * 16;
    const int nsb = K >> 8, ng = K >> 6;
    fv8 acc[1][1];
    acc_zero(acc);
    int row = bm + (lane & 15);
    row = row < M ? row : M - 1;
    const int8_t* xqr[1] = {Xq + (size_t)row * K};
    const float2* xdsr[1] = {Xds + (size_t)row * ng};
    const int8_t* xq2r[1] = {HI ? Xq2 + (size_t)row * K : nullptr};
    const float* xd2r[1] = {HI ? Xd2 + (size_t)row * ng : nullptr};
    int wrow = bn + (tid >> 1);
    wrow = wrow < N ? wrow : N - 1;
    const int brow[1] = {wave * 16};
    gemm_mainloop_i8<FMT, 1, 1, BN_W, BK, HI>(W + (size_t)wrow * nsb * SB, nsb, xqr, xdsr, xq2r, xd2r, ws_q, ws_eff, acc, brow);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int n = bn + wave * 16 + acc_n(r, lane);
        const int m = bm + acc_m(r, lane);
        if (m < M && n < N) Y[(size_t)m * N + n] = acc[0][0][r];
    }
}

// ---- Large-M INT8 Tensor Core GEMM (BM = 128, BN = 64, BK = 256, all 8 warps active, 18.4 KB LDS)
template <int FMT, bool HI>
__global__ void __launch_bounds__(256) k_gemm_i8(const uint8_t* __restrict__ W, int N, int K,
                                                 const int8_t* __restrict__ Xq, const float2* __restrict__ Xds,
                                                 const int8_t* __restrict__ Xq2, const float* __restrict__ Xd2,
                                                 float* __restrict__ Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int BN_W = 64, BK = 256, LDW32 = 68, GPR = 4;
    __shared__ __align__(16) uint32_t ws_q[BN_W * LDW32];
    __shared__ float ws_eff[BN_W * GPR];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave & 3, wn = wave >> 2;
    const int bn = blockIdx.x * BN_W, bm = blockIdx.y * GEMM_BM;
    const int nsb = K >> 8, ng = K >> 6;
    fv8 acc[2][2];
    acc_zero(acc);
    const int8_t* xqr[2];
    const float2* xdsr[2];
    const int8_t* xq2r[2];
    const float* xd2r[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int row = bm + wm * 32 + i * 16 + (lane & 15);
        row = row < M ? row : M - 1;
        xqr[i] = Xq + (size_t)row * K;
        xdsr[i] = Xds + (size_t)row * ng;
        xq2r[i] = HI ? Xq2 + (size_t)row * K : nullptr;
        xd2r[i] = HI ? Xd2 + (size_t)row * ng : nullptr;
    }
    int wrow = bn + (tid >> 2);
    wrow = wrow < N ? wrow : N - 1;
    const int brow[2] = {wn * 32, wn * 32 + 16};
    gemm_mainloop_i8<FMT, 2, 2, BN_W, BK, HI>(W + (size_t)wrow * nsb * SB, nsb, xqr, xdsr, xq2r, xd2r, ws_q, ws_eff, acc, brow);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int n = bn + wn * 32 + j * 16 + acc_n(r, lane);
                const int m = bm + wm * 32 + i * 16 + acc_m(r, lane);
                if (m < M && n < N) Y[(size_t)m * N + n] = acc[i][j][r];
            }
}

// ---- __dp4a batch-invariant GEMV (M = 1..8) with on-the-fly Q8_1_G64 activation quantization in LDS (no extra launch!).
// Works on sm_61+ (hardware dp4a) and sm_60 (scalar fallback). Bit-identical per token across M = 1..8!
template <int FMT, int MT, int RW, bool HI>
__global__ void __launch_bounds__(256) k_gemv_dp4a(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr float H = sk_half<FMT>();
    // Chunk KC = 2048 elements (32 sub-groups of 64) per LDS pass for MT <= 8:
    //   xq_s : MT * 512 uint32_t (2 KB * MT -> <= 16 KB)
    //   xq2_s: MT * 512 uint32_t (only if HI -> <= 16 KB)
    //   xds_s: MT * 32 float2    (256 B * MT -> <= 2 KB)
    //   xd2_s: MT * 32 float     (128 B * MT -> <= 1 KB)
    // Total <= 35 KB even for MT=8, HI=true!
    constexpr int KC = 2048;
    constexpr int NG_C = KC / 64;  // 32 sub-groups per chunk
    __shared__ uint32_t xq_s[MT * (KC / 4)];
    __shared__ float2 xds_s[MT * NG_C];
    __shared__ uint32_t xq2_s[HI ? MT * (KC / 4) : 1];
    __shared__ float xd2_s[HI ? MT * NG_C : 1];
    __shared__ int8_t tmp_q[8][64];
    __shared__ int8_t tmp_q2[HI ? 8 : 1][64];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int row0 = (blockIdx.x * 8 + wave) * RW;
    const int nsb = K >> 8;
    float acc[RW][MT];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;
    const uint8_t* rp[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) rp[r] = W + (size_t)(row0 + r < N ? row0 + r : N - 1) * nsb * SB;

    for (int k0 = 0; k0 < K; k0 += KC) {
        const int kc = (K - k0) < KC ? (K - k0) : KC;
        const int ngc = kc >> 6;
        // Cooperatively quantize MT * ngc sub-groups: 8 warps loop over sub-groups
        for (int idx = wave; idx < MT * ngc; idx += 8) {
            const int mm = idx / ngc, sg = idx - mm * ngc;
            float2 f = make_float2(0.f, 0.f);
            if (mm < M) {
                const __half2 h2 = *(const __half2*)(X + (size_t)mm * K + k0 + sg * 64 + lane * 2);
                f = __half22float2(h2);
            }
            float amax = fmaxf(fabsf(f.x), fabsf(f.y));
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o));
            const float da = amax * (1.0f / 127.0f);
            const float inva = da > 0.f ? (127.0f / amax) : 0.f;
            int q0 = max(-127, min(127, __float2int_rn(f.x * inva)));
            int q1 = max(-127, min(127, __float2int_rn(f.y * inva)));
            int s0 = q0 + q1;
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) s0 += __shfl_xor(s0, o);
            tmp_q[wave][lane * 2] = (int8_t)q0;
            tmp_q[wave][lane * 2 + 1] = (int8_t)q1;
            float s_tot = da * (float)s0;
            if constexpr (HI) {
                const float r0 = f.x - da * (float)q0, r1 = f.y - da * (float)q1;
                float rmax = fmaxf(fabsf(r0), fabsf(r1));
#pragma unroll
                for (int o = 16; o >= 1; o >>= 1) rmax = fmaxf(rmax, __shfl_xor(rmax, o));
                const float db = rmax * (1.0f / 127.0f);
                const float invb = db > 0.f ? (127.0f / rmax) : 0.f;
                int qb0 = max(-127, min(127, __float2int_rn(r0 * invb)));
                int qb1 = max(-127, min(127, __float2int_rn(r1 * invb)));
                int sb = qb0 + qb1;
#pragma unroll
                for (int o = 16; o >= 1; o >>= 1) sb += __shfl_xor(sb, o);
                tmp_q2[wave][lane * 2] = (int8_t)qb0;
                tmp_q2[wave][lane * 2 + 1] = (int8_t)qb1;
                s_tot += db * (float)sb;
                if (lane == 0) xd2_s[mm * NG_C + sg] = db;
            }
            __syncwarp();
            if (lane < 16) {
                xq_s[mm * (KC / 4) + sg * 16 + lane] = ((const uint32_t*)tmp_q[wave])[lane];
                if constexpr (HI) xq2_s[mm * (KC / 4) + sg * 16 + lane] = ((const uint32_t*)tmp_q2[wave])[lane];
            }
            if (lane == 0) xds_s[mm * NG_C + sg] = make_float2(da, s_tot);
        }
        __syncthreads();

        const int sb0 = k0 >> 8, nsbc = kc >> 8;
        const int g_in_sb = lane >> 3;  // 0..3
        const bool is_g0 = (lane & 7) == 0;
        for (int s = 0; s < nsbc; ++s) {
            const int sg_idx = s * 4 + g_in_sb;
            uint2 xv[MT];
            float2 ds[MT];
            uint2 xv2[MT];
            float d2[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                xv[m] = *(const uint2*)&xq_s[m * (KC / 4) + s * 64 + lane * 2];
                ds[m] = xds_s[m * NG_C + sg_idx];
                if constexpr (HI) {
                    xv2[m] = *(const uint2*)&xq2_s[m * (KC / 4) + s * 64 + lane * 2];
                    d2[m] = xd2_s[m * NG_C + sg_idx];
                }
            }
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                const uint8_t* blk = rp[r] + (size_t)(sb0 + s) * SB;
                const float eff = sk_eff<FMT>(blk, g_in_sb);
                uint32_t w0, w1;
                if constexpr (FMT == FMT_S8) {
                    w0 = ld32u(blk + 6 + 8 * lane);
                    w1 = ld32u(blk + 6 + 8 * lane + 4);
                } else {
                    uint32_t qa, qb;
                    load_q_raw<FMT>(blk + 6, lane, qa, qb);
                    if constexpr (FMT == FMT_S2 || FMT == FMT_S3) {
                        const uint32_t b0 = qa & 0xFFu, b1 = (qa >> 8) & 0xFFu;
                        w0 = (b0 & 0x03u) | ((b0 & 0x0Cu) << 6) | ((b0 & 0x30u) << 12) | ((b0 & 0xC0u) << 18);
                        w1 = (b1 & 0x03u) | ((b1 & 0x0Cu) << 6) | ((b1 & 0x30u) << 12) | ((b1 & 0xC0u) << 18);
                        if constexpr (FMT == FMT_S3) {
                            w0 |= ((qb & 0x01u) << 2) | ((qb & 0x02u) << 9) | ((qb & 0x04u) << 16) | ((qb & 0x08u) << 23);
                            w1 |= ((qb & 0x10u) >> 2) | ((qb & 0x20u) << 5) | ((qb & 0x40u) << 12) | ((qb & 0x80u) << 19);
                        }
                    } else {
                        w0 = (qa & 0x0000000Fu) | ((qa & 0x000000F0u) << 4) | ((qa & 0x00000F00u) << 8) | ((qa & 0x0000F000u) << 12);
                        w1 = ((qa >> 16) & 0x0000000Fu) | (((qa >> 16) & 0x000000F0u) << 4) | (((qa >> 16) & 0x00000F00u) << 8) | (((qa >> 16) & 0x0000F000u) << 12);
                        if constexpr (FMT == FMT_S5) {
                            w0 |= ((qb & 0x01u) << 4) | ((qb & 0x02u) << 11) | ((qb & 0x04u) << 18) | ((qb & 0x08u) << 25);
                            w1 |= ((qb & 0x10u) << 0) | ((qb & 0x20u) << 7) | ((qb & 0x40u) << 14) | ((qb & 0x80u) << 21);
                        } else if constexpr (FMT == FMT_S6) {
                            const uint32_t hlo = qb & 0xFFu, hhi = qb >> 8;
                            w0 |= ((hlo & 0x03u) << 4) | ((hlo & 0x0Cu) << 10) | ((hlo & 0x30u) << 16) | ((hlo & 0xC0u) << 22);
                            w1 |= ((hhi & 0x03u) << 4) | ((hhi & 0x0Cu) << 10) | ((hhi & 0x30u) << 16) | ((hhi & 0xC0u) << 22);
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    int32_t id = dp4a_s8(w0, xv[m].x, 0);
                    id = dp4a_s8(w1, xv[m].y, id);
                    float v = ds[m].x * (float)id;
                    if constexpr (HI) {
                        int32_t id2 = dp4a_s8(w0, xv2[m].x, 0);
                        id2 = dp4a_s8(w1, xv2[m].y, id2);
                        v = fmaf(d2[m], (float)id2, v);
                    }
                    if constexpr (FMT != FMT_S8) {
                        if (is_g0) v -= H * ds[m].y;
                    }
                    acc[r][m] = fmaf(eff, v, acc[r][m]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float v = acc[r][m];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0 && m < M && row0 + r < N) Y[(size_t)m * N + row0 + r] = v;
        }
}

// Reusable scratch workspace for quantized activations (allocated lazily, zero per-call malloc)
struct Int8ActBuf {
    int8_t* xq = nullptr;
    float2* xds = nullptr;
    int8_t* xq2 = nullptr;
    float* xd2 = nullptr;
    size_t cap_elems = 0;
    void ensure(size_t elems, bool hi) {
        if (elems <= cap_elems && (!hi || xq2 != nullptr)) return;
        if (xq) { (void)cudaFree(xq); (void)cudaFree(xds); }
        if (xq2) { (void)cudaFree(xq2); (void)cudaFree(xd2); xq2 = nullptr; xd2 = nullptr; }
        cap_elems = elems + (elems >> 2);
        const size_t ng = (cap_elems + 63) / 64;
        if (cudaMalloc(&xq, cap_elems) != cudaSuccess) abort();
        if (cudaMalloc(&xds, ng * sizeof(float2)) != cudaSuccess) abort();
        if (hi) {
            if (cudaMalloc(&xq2, cap_elems) != cudaSuccess) abort();
            if (cudaMalloc(&xd2, ng * sizeof(float)) != cudaSuccess) abort();
        }
    }
};
inline Int8ActBuf& int8_act_buf() { static Int8ActBuf b; return b; }

template <bool HI>
inline void quantize_x_gpu(hipStream_t s, const __half* X, int M, int K, Int8ActBuf& b) {
    b.ensure((size_t)M * K, HI);
    const int ng = K >> 6;
    dim3 grid((ng + 7) / 8, M), block(256);
    hipLaunchKernelGGL((k_quant_x_q8<HI>), grid, block, 0, s, X, M, K, b.xq, b.xds, b.xq2, b.xd2);
}

template <int FMT, bool HI>
inline void launch_gemm_i8_fmt(hipStream_t s, const void* W, int N, int K, const __half* X, float* Y, int M) {
    Int8ActBuf& b = int8_act_buf();
    quantize_x_gpu<HI>(s, X, M, K, b);
    if (M <= 16) {
        dim3 grid((N + 127) / 128, (M + 15) / 16), block(256);
        hipLaunchKernelGGL((k_gemm_i8_s<FMT, HI>), grid, block, 0, s, (const uint8_t*)W, N, K, b.xq, b.xds, b.xq2, b.xd2, Y, M);
    } else {
        dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM), block(256);
        hipLaunchKernelGGL((k_gemm_i8<FMT, HI>), grid, block, 0, s, (const uint8_t*)W, N, K, b.xq, b.xds, b.xq2, b.xd2, Y, M);
    }
}

template <int FMT, bool HI>
inline void launch_gemv_dp4a_fmt(hipStream_t s, const void* W, int N, int K, const __half* X, float* Y, int M) {
    constexpr int RW = 2;
    dim3 grid((N + 8 * RW - 1) / (8 * RW)), block(256);
    if (M <= 1) hipLaunchKernelGGL((k_gemv_dp4a<FMT, 1, RW, HI>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M);
    else if (M <= 2) hipLaunchKernelGGL((k_gemv_dp4a<FMT, 2, RW, HI>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M);
    else if (M <= 4) hipLaunchKernelGGL((k_gemv_dp4a<FMT, 4, RW, HI>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M);
    else hipLaunchKernelGGL((k_gemv_dp4a<FMT, 8, RW, HI>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M);
}

}  // namespace skd

// Public CUDA INT8 Tensor Core GEMM entry (S2..S8_K; falls back to FP16 MMA for S1.25/S1.5).
// hi = false: 1-term Q8_1 (2x Tensor Core TOPS); hi = true: 2-term Q8_1 (~3e-5 rel err, better than FP16).
inline void linear_h_gemm_i8(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M, bool hi = true) {
    if (K < 256 || (K & 255) != 0) { linear_h_ref(s, fmt, W, N, K, X, Y, M); return; }
#define HY4_I8G(F) case F: if (hi) skd::launch_gemm_i8_fmt<F, true>(s, W, N, K, X, Y, M); else skd::launch_gemm_i8_fmt<F, false>(s, W, N, K, X, Y, M); break
    switch (fmt) {
        HY4_I8G(FMT_S2);
        HY4_I8G(FMT_S3);
        HY4_I8G(FMT_S4);
        HY4_I8G(FMT_S5);
        HY4_I8G(FMT_S6);
        HY4_I8G(FMT_S8);
        default:
            if (M <= 16) linear_h_gemm_s(s, fmt, W, N, K, X, Y, M);
            else linear_h_gemm(s, fmt, W, N, K, X, Y, M);
            break;
    }
#undef HY4_I8G
}

// Public CUDA __dp4a GEMV entry (M = 1..8, S2..S8_K; falls back to FP16 skfast for S1.25/S1.5 or M > 8).
inline void linear_h_gemv_dp4a(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M, bool hi = true) {
    if (M > 8 || K < 256 || (K & 255) != 0) { linear_h_gemv(s, fmt, W, N, K, X, Y, M); return; }
#define HY4_DP4A(F) case F: if (hi) skd::launch_gemv_dp4a_fmt<F, true>(s, W, N, K, X, Y, M); else skd::launch_gemv_dp4a_fmt<F, false>(s, W, N, K, X, Y, M); break
    switch (fmt) {
        HY4_DP4A(FMT_S2);
        HY4_DP4A(FMT_S3);
        HY4_DP4A(FMT_S4);
        HY4_DP4A(FMT_S5);
        HY4_DP4A(FMT_S6);
        HY4_DP4A(FMT_S8);
        default: linear_h_gemv(s, fmt, W, N, K, X, Y, M); break;
    }
#undef HY4_DP4A
}

}  // namespace hy4
#endif  // HY4_CUDA && HY4_MMA
