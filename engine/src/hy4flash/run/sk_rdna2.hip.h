// sk_rdna2.hip.h - RDNA2 (gfx103x) Wave32 high-throughput tiled GEMM using v_dot4_i32_i8 (sdot4)
//
// No WMMA instructions exist on gfx103x. Every S_K format (S1.25, S1.5, S2_K..S8_K) unpacks cleanly to exact zero-point-free signed int8 weights:
//   - S2..S6: w_i8 = 2*q - 2*H  (exact int8 in [-63, +63], effective scale = 0.5 * eff_g)
//   - S8:     w_i8 = int8(q)    (exact int8 in [-127, +127], effective scale = eff_g)
//   - S1.25:  w_i8 = kS125CB[]  (exact int8 in [-2, +2], effective scale = eff_g)
//   - S1.5:   w_i8 = trit       (exact int8 in [-1, +1], effective scale = eff_g)
//
// Activations are quantized in LDS per 64-element sub-group into 2-term symmetric int8 (x = d_hi * q_hi + d_lo * q_lo, ~14-bit precision,
// passing sk_test rel_err ~1.5e-5 << 2e-3 and teacher-forced KL <= 4.5e-3) and multiplied via __builtin_amdgcn_sdot4 (v_dot4_i32_i8).
// K-major ws_i8[64][64] + row-major xs[BM][16] enable bank-conflict-free ds_read_b128 (int4) loads and keep VGPRs <= 64 for high occupancy.
#pragma once
#include "skfast.hip.h"

#if !HY4_CUDA

namespace hy4 {
namespace skd {
namespace rdna2 {

constexpr int R2_BN = 64;

__device__ __forceinline__ int pack4_i8(int a, int b, int c, int d) {
    const uint32_t u = ((uint32_t)(a & 0xFF)) | (((uint32_t)(b & 0xFF)) << 8) | (((uint32_t)(c & 0xFF)) << 16) | (((uint32_t)(d & 0xFF)) << 24);
    return (int)u;
}

// Unpack 64 weights of sub-group dg into 16 packed int32 (4 signed int8s each), storing K-major into ws_i8[(dg*16 + k)*64 + drow].
template <int FMT>
__device__ __forceinline__ float unpack_sub_i8_kmaj(const RawSub<FMT>& r, int dg, int* ws_i8, int drow) {
    const float eff = raw_eff<FMT>(r, dg);
    int* dst = ws_i8 + dg * 16 * 64 + drow;
    if constexpr (FMT == FMT_S8) {
#pragma unroll
        for (int k = 0; k < 16; ++k) dst[k * 64] = (int)r.lo[k];
        return eff;
    } else if constexpr (FMT == FMT_S2) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint32_t b = (r.lo[k >> 2] >> ((k & 3) * 8)) & 0xFFu;
            dst[k * 64] = pack4_i8(((int)(b & 3u) << 1) - 3, ((int)((b >> 2) & 3u) << 1) - 3, ((int)((b >> 4) & 3u) << 1) - 3, ((int)((b >> 6) & 3u) << 1) - 3);
        }
        return eff * 0.5f;
    } else if constexpr (FMT == FMT_S3) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint32_t lb = (r.lo[k >> 2] >> ((k & 3) * 8)) & 0xFFu;
            const uint32_t hb = (r.hi[k >> 3] >> ((k & 7) * 4)) & 0xFu;
            const int w0 = ((int)((lb & 3u) | ((hb & 1u) << 2)) << 1) - 7;
            const int w1 = ((int)(((lb >> 2) & 3u) | (((hb >> 1) & 1u) << 2)) << 1) - 7;
            const int w2 = ((int)(((lb >> 4) & 3u) | (((hb >> 2) & 1u) << 2)) << 1) - 7;
            const int w3 = ((int)(((lb >> 6) & 3u) | (((hb >> 3) & 1u) << 2)) << 1) - 7;
            dst[k * 64] = pack4_i8(w0, w1, w2, w3);
        }
        return eff * 0.5f;
    } else if constexpr (FMT == FMT_S4) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint32_t u = (r.lo[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu;
            dst[k * 64] = pack4_i8(((int)(u & 15u) << 1) - 15, ((int)((u >> 4) & 15u) << 1) - 15, ((int)((u >> 8) & 15u) << 1) - 15, ((int)((u >> 12) & 15u) << 1) - 15);
        }
        return eff * 0.5f;
    } else if constexpr (FMT == FMT_S5) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint32_t u = (r.lo[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu;
            const uint32_t hb = (r.hi[k >> 3] >> ((k & 7) * 4)) & 0xFu;
            const int w0 = ((int)((u & 15u) | ((hb & 1u) << 4)) << 1) - 31;
            const int w1 = ((int)(((u >> 4) & 15u) | (((hb >> 1) & 1u) << 4)) << 1) - 31;
            const int w2 = ((int)(((u >> 8) & 15u) | (((hb >> 2) & 1u) << 4)) << 1) - 31;
            const int w3 = ((int)(((u >> 12) & 15u) | (((hb >> 3) & 1u) << 4)) << 1) - 31;
            dst[k * 64] = pack4_i8(w0, w1, w2, w3);
        }
        return eff * 0.5f;
    } else if constexpr (FMT == FMT_S6) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint32_t u = (r.lo[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu;
            const uint32_t hb = (r.hi[k >> 2] >> ((k & 3) * 8)) & 0xFFu;
            const int w0 = ((int)((u & 15u) | ((hb & 3u) << 4)) << 1) - 63;
            const int w1 = ((int)(((u >> 4) & 15u) | (((hb >> 2) & 3u) << 4)) << 1) - 63;
            const int w2 = ((int)(((u >> 8) & 15u) | (((hb >> 4) & 3u) << 4)) << 1) - 63;
            const int w3 = ((int)(((u >> 12) & 15u) | (((hb >> 6) & 3u) << 4)) << 1) - 63;
            dst[k * 64] = pack4_i8(w0, w1, w2, w3);
        }
        return eff * 0.5f;
    } else if constexpr (FMT == FMT_S1_25) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int p = 5 * k, wi = p >> 5, sh = p & 31;
            uint32_t idx;
            if (sh + 5 <= 32) idx = (r.lo[wi] >> sh) & 31u;
            else idx = ((r.lo[wi] >> sh) | (r.lo[wi + 1] << (32 - sh))) & 31u;
            dst[k * 64] = pack4_i8(kS125CB[idx][0], kS125CB[idx][1], kS125CB[idx][2], kS125CB[idx][3]);
        }
        return eff;
    } else {  // FMT_S1_5
        const uint32_t t0 = r.hi[0], byte0 = t0 / 5u;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            int w[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t ti = t0 + 4 * k + i;
                w[i] = s15_trit(byte_at4(r.lo, (int)(ti / 5u - byte0)), (int)(ti % 5u));
            }
            dst[k * 64] = pack4_i8(w[0], w[1], w[2], w[3]);
        }
        return eff;
    }
}

__device__ __forceinline__ int clamp_i8(float x) {
    int v = __float2int_rn(x);
    return v < -127 ? -127 : (v > 127 ? 127 : v);
}

// Quantize BM rows x 64 fp16 activations into 2-term (or 1-term) int8 in LDS (row-major [BM][16], 16-byte aligned for int4 loads).
template <int BM, bool TWO_TERM, class RowPtrFn>
__device__ __forceinline__ void quant_x_sub_i8(RowPtrFn xrow_fn, int k0, int* xs_hi, int* xs_lo, float* d_hi, float* d_lo) {
    const int tid = threadIdx.x;
    const int r = tid >> 2, q = tid & 3;
    if (r < BM) {
        const __half* xp = xrow_fn(r) + k0 + q * 16;
        const uint4 v0 = *reinterpret_cast<const uint4*>(xp);
        const uint4 v1 = *reinterpret_cast<const uint4*>(xp + 8);
        float f[16];
        h2_unpack(v0.x, f[0], f[1]);
        h2_unpack(v0.y, f[2], f[3]);
        h2_unpack(v0.z, f[4], f[5]);
        h2_unpack(v0.w, f[6], f[7]);
        h2_unpack(v1.x, f[8], f[9]);
        h2_unpack(v1.y, f[10], f[11]);
        h2_unpack(v1.z, f[12], f[13]);
        h2_unpack(v1.w, f[14], f[15]);
        float am = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(f[i]));
        am = fmaxf(am, __shfl_xor(am, 1));
        am = fmaxf(am, __shfl_xor(am, 2));
        const float dh = am * (1.0f / 127.0f);
        const float ih = am > 0.f ? (127.0f / am) : 0.f;
        int qh[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) qh[i] = clamp_i8(f[i] * ih);
        int4 p4;
        p4.x = pack4_i8(qh[0], qh[1], qh[2], qh[3]);
        p4.y = pack4_i8(qh[4], qh[5], qh[6], qh[7]);
        p4.z = pack4_i8(qh[8], qh[9], qh[10], qh[11]);
        p4.w = pack4_i8(qh[12], qh[13], qh[14], qh[15]);
        *reinterpret_cast<int4*>(&xs_hi[r * 16 + q * 4]) = p4;
        if (q == 0) d_hi[r] = dh;
        if constexpr (TWO_TERM) {
            float res[16];
            float am2 = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                res[i] = f[i] - dh * (float)qh[i];
                am2 = fmaxf(am2, fabsf(res[i]));
            }
            am2 = fmaxf(am2, __shfl_xor(am2, 1));
            am2 = fmaxf(am2, __shfl_xor(am2, 2));
            const float dl = am2 * (1.0f / 127.0f);
            const float il = am2 > 0.f ? (127.0f / am2) : 0.f;
            int4 pl;
            pl.x = pack4_i8(clamp_i8(res[0] * il), clamp_i8(res[1] * il), clamp_i8(res[2] * il), clamp_i8(res[3] * il));
            pl.y = pack4_i8(clamp_i8(res[4] * il), clamp_i8(res[5] * il), clamp_i8(res[6] * il), clamp_i8(res[7] * il));
            pl.z = pack4_i8(clamp_i8(res[8] * il), clamp_i8(res[9] * il), clamp_i8(res[10] * il), clamp_i8(res[11] * il));
            pl.w = pack4_i8(clamp_i8(res[12] * il), clamp_i8(res[13] * il), clamp_i8(res[14] * il), clamp_i8(res[15] * il));
            *reinterpret_cast<int4*>(&xs_lo[r * 16 + q * 4]) = pl;
            if (q == 0) d_lo[r] = dl;
        }
    }
}

// Hoisted shared memory (24.8 KB for BM=64 2-term -> 4 blocks/WGP = full occupancy!)
template <int BM, bool TWO_TERM>
struct Rdna2Smem {
    alignas(16) int ws_i8[64 * 64];         // [k_pack (0..63)][n_row (0..63)] = 16 KB
    alignas(16) int xs_hi[BM * 16];         // [m_row (0..BM-1)][k (0..15)]    = 4 KB (BM=64)
    alignas(16) int xs_lo[TWO_TERM ? BM * 16 : 4];
    float ws_eff[4][64];                    // [g][n_row]                      = 1 KB
    float d_hi[BM];
    float d_lo[TWO_TERM ? BM : 1];
};

// Load TN weight words for one k-pack from K-major ws_i8[k_row * 64 + ...]:
//   SPLIT_GU == false: TN contiguous cols starting at n0 (int4 for TN=4, int2 for TN=2)
//   SPLIT_GU == true:  TN/2 gate cols starting at n0 + TN/2 up cols starting at 32 + n0
template <int TN, bool SPLIT_GU>
__device__ __forceinline__ void ld_w_vec(const int* wrow, int n0, int (&b)[TN]) {
    if constexpr (!SPLIT_GU && TN == 4) {
        const int4 v = *reinterpret_cast<const int4*>(wrow + n0);
        b[0] = v.x; b[1] = v.y; b[2] = v.z; b[3] = v.w;
    } else if constexpr (!SPLIT_GU && TN == 2) {
        const int2 v = *reinterpret_cast<const int2*>(wrow + n0);
        b[0] = v.x; b[1] = v.y;
    } else if constexpr (SPLIT_GU && TN == 4) {
        const int2 vg = *reinterpret_cast<const int2*>(wrow + n0);
        const int2 vu = *reinterpret_cast<const int2*>(wrow + 32 + n0);
        b[0] = vg.x; b[1] = vg.y; b[2] = vu.x; b[3] = vu.y;
    } else if constexpr (SPLIT_GU && TN == 2) {
        b[0] = wrow[n0];
        b[1] = wrow[32 + n0];
    }
}

template <int TN, bool SPLIT_GU>
__device__ __forceinline__ void ld_we(const float* eff_row, int n0, float (&we)[TN]) {
    if constexpr (!SPLIT_GU) {
#pragma unroll
        for (int j = 0; j < TN; ++j) we[j] = eff_row[n0 + j];
    } else {
        constexpr int H = TN / 2;
#pragma unroll
        for (int j = 0; j < H; ++j) {
            we[j] = eff_row[n0 + j];
            we[H + j] = eff_row[32 + n0 + j];
        }
    }
}

// Compute 16 k-packs (64 elements) of TM x TN sdot4 using bank-conflict-free int4 LDS reads (xs) and int4/int2 LDS reads (ws).
template <int TM, int TN, bool SPLIT_GU>
__device__ __forceinline__ void sdot4_k16(const int* xs, const int* ws_g, int m0, int n0, int (&ih)[TM][TN]) {
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) ih[i][j] = 0;

#pragma unroll
    for (int k4 = 0; k4 < 4; ++k4) {
        int4 a4[TM];
#pragma unroll
        for (int i = 0; i < TM; ++i) a4[i] = *reinterpret_cast<const int4*>(xs + (m0 + i) * 16 + k4 * 4);
        int b[TN];
        ld_w_vec<TN, SPLIT_GU>(ws_g + (k4 * 4 + 0) * 64, n0, b);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) ih[i][j] = sdot4_i32(a4[i].x, b[j], ih[i][j]);
        ld_w_vec<TN, SPLIT_GU>(ws_g + (k4 * 4 + 1) * 64, n0, b);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) ih[i][j] = sdot4_i32(a4[i].y, b[j], ih[i][j]);
        ld_w_vec<TN, SPLIT_GU>(ws_g + (k4 * 4 + 2) * 64, n0, b);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) ih[i][j] = sdot4_i32(a4[i].z, b[j], ih[i][j]);
        ld_w_vec<TN, SPLIT_GU>(ws_g + (k4 * 4 + 3) * 64, n0, b);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) ih[i][j] = sdot4_i32(a4[i].w, b[j], ih[i][j]);
    }
}

// Generic RDNA2 Wave32 sdot4 GEMM main loop (256 threads = 8 waves, 100% active):
// Outer g = 0..3 is deliberately NOT unrolled (#pragma nounroll) so VGPRs stay <= 64 and compilation is 10x faster.
template <int FMT, int BM, int TM, int TN, bool TWO_TERM, bool SPLIT_GU, class RowPtrFn>
__device__ __forceinline__ void rdna2_sdot4_loop(Rdna2Smem<BM, TWO_TERM>& sm, const uint8_t* wr, int nsb, RowPtrFn xrow_fn, int m0, int n0, float (&acc)[TM][TN]) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int tid = threadIdx.x;
    const int drow = tid >> 2, dg = tid & 3;

    for (int sb = 0; sb < nsb; ++sb) {
        {
            RawSub<FMT> cur;
            load_raw<FMT>(wr + (size_t)sb * SB, dg, cur);
            sm.ws_eff[dg][drow] = unpack_sub_i8_kmaj<FMT>(cur, dg, sm.ws_i8, drow);
        }

#pragma nounroll
        for (int g = 0; g < 4; ++g) {
            quant_x_sub_i8<BM, TWO_TERM>(xrow_fn, sb * 256 + g * 64, sm.xs_hi, sm.xs_lo, sm.d_hi, sm.d_lo);
            __syncthreads();

            float we[TN];
            ld_we<TN, SPLIT_GU>(sm.ws_eff[g], n0, we);
            const int* ws_g = sm.ws_i8 + g * 16 * 64;

            int ih[TM][TN];
            sdot4_k16<TM, TN, SPLIT_GU>(sm.xs_hi, ws_g, m0, n0, ih);
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float dh = sm.d_hi[m0 + i];
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf((float)ih[i][j], dh * we[j], acc[i][j]);
            }

            if constexpr (TWO_TERM) {
                sdot4_k16<TM, TN, SPLIT_GU>(sm.xs_lo, ws_g, m0, n0, ih);
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const float dl = sm.d_lo[m0 + i];
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] = fmaf((float)ih[i][j], dl * we[j], acc[i][j]);
                }
            }
            __syncthreads();
        }
    }
}

// Thread-to-tile mapping for 256 threads (8 waves), BN=64:
//   BM=16 -> TM=2, TN=2 (T_N=32: tn=tid&31, tm=tid>>5; n0=tn*2, m0=tm*2)
//   BM=32 -> TM=4, TN=2 (T_N=32: tn=tid&31, tm=tid>>5; n0=tn*2, m0=tm*4)
//   BM=64 -> TM=4, TN=4 (T_N=16: tn=tid&15, tm=tid>>4; n0=tn*4, m0=tm*4)
template <int FMT, int BM, int TM, int TN, bool TWO_TERM>
__global__ void __launch_bounds__(256, 2) k_rdna2_gemm(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    __shared__ Rdna2Smem<BM, TWO_TERM> sm;
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int T_N = R2_BN / TN;
    const int tid = threadIdx.x;
    const int bn = blockIdx.x * R2_BN, bm = blockIdx.y * BM;
    const int nsb = K >> 8;

    const int tn = tid % T_N, tm = tid / T_N;
    const int m0 = tm * TM, n0 = tn * TN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    int wrow = bn + (tid >> 2);
    wrow = wrow < N ? wrow : N - 1;
    const uint8_t* wr = W + (size_t)wrow * nsb * SB;
    auto xrow_fn = [&](int r) -> const __half* {
        int mr = bm + r;
        mr = mr < M ? mr : M - 1;
        return X + (size_t)mr * K;
    };

    rdna2_sdot4_loop<FMT, BM, TM, TN, TWO_TERM, false>(sm, wr, nsb, xrow_fn, m0, n0, acc);

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int mr = bm + m0 + i;
        if (mr >= M) continue;
        float* yp = Y + (size_t)mr * N + bn + n0;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (bn + n0 + j < N) yp[j] = acc[i][j];
        }
    }
}

template <int FMT, bool TWO_TERM>
inline void launch_rdna2_gemm_fmt(hipStream_t s, const uint8_t* W, int N, int K, const __half* X, float* Y, int M) {
    const int gx = (N + R2_BN - 1) / R2_BN;
    if (M <= 16)
        hipLaunchKernelGGL((k_rdna2_gemm<FMT, 16, 2, 2, TWO_TERM>), dim3(gx, 1), dim3(256), 0, s, W, N, K, X, Y, M);
    else if (M <= 32)
        hipLaunchKernelGGL((k_rdna2_gemm<FMT, 32, 4, 2, TWO_TERM>), dim3(gx, 1), dim3(256), 0, s, W, N, K, X, Y, M);
    else
        hipLaunchKernelGGL((k_rdna2_gemm<FMT, 64, 4, 4, TWO_TERM>), dim3(gx, (M + 63) / 64), dim3(256), 0, s, W, N, K, X, Y, M);
}

inline bool rdna2_use_1term() {
    static int m = -1;
    if (m < 0) {
        const char* e = getenv("HY4_RDNA2_1TERM");
        m = (e && atoi(e) != 0) ? 1 : 0;
    }
    return m != 0;
}

template <bool TWO_TERM>
inline void dispatch_rdna2_gemm(hipStream_t s, int fmt, const uint8_t* W, int N, int K, const __half* X, float* Y, int M) {
    switch (fmt) {
        case FMT_S1_25: launch_rdna2_gemm_fmt<FMT_S1_25, TWO_TERM>(s, W, N, K, X, Y, M); break;
        case FMT_S1_5: launch_rdna2_gemm_fmt<FMT_S1_5, TWO_TERM>(s, W, N, K, X, Y, M); break;
        case FMT_S2: launch_rdna2_gemm_fmt<FMT_S2, TWO_TERM>(s, W, N, K, X, Y, M); break;
        case FMT_S3: launch_rdna2_gemm_fmt<FMT_S3, TWO_TERM>(s, W, N, K, X, Y, M); break;
        case FMT_S4: launch_rdna2_gemm_fmt<FMT_S4, TWO_TERM>(s, W, N, K, X, Y, M); break;
        case FMT_S5: launch_rdna2_gemm_fmt<FMT_S5, TWO_TERM>(s, W, N, K, X, Y, M); break;
        case FMT_S6: launch_rdna2_gemm_fmt<FMT_S6, TWO_TERM>(s, W, N, K, X, Y, M); break;
        case FMT_S8: launch_rdna2_gemm_fmt<FMT_S8, TWO_TERM>(s, W, N, K, X, Y, M); break;
        default: linear_h_ref(s, fmt, W, N, K, X, Y, M); break;
    }
}

}  // namespace rdna2
}  // namespace skd

inline void linear_h_rdna2(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    if (fmt == FMT_Q8ROW || (K & 255) != 0) {
        linear_h_ref(s, fmt, W, N, K, X, Y, M);
        return;
    }
#ifdef HY4_RDNA2_BENCH_MODES
    if (skd::rdna2::rdna2_use_1term()) {
        skd::rdna2::dispatch_rdna2_gemm<false>(s, fmt, (const uint8_t*)W, N, K, X, Y, M);
        return;
    }
#endif
    skd::rdna2::dispatch_rdna2_gemm<true>(s, fmt, (const uint8_t*)W, N, K, X, Y, M);
}

}  // namespace hy4
#endif  // !HY4_CUDA
