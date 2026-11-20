// sk_dp4a_cuda.hip.h - CUDA __dp4a INT8 S_K x two-term Q8_1 decode/verify GEMV & MoE kernels (sm_61+ / sm_70 / sm_75 / sm_80+)
//
// Quantizes each 64-element sub-group of X in LDS to two-term Q8_1 (x ~= da*qa + db*qb, ~14-bit precision: <1e-4 rel_err,
// matching FP16/FP32 reference KL parity) in even/odd packed int8x4 layout, then evaluates S2..S8 weight dots with __dp4a.
// Every token m is quantized and evaluated independently of M, so M=1..16 rows are bitwise batch-invariant (DFly2 == plain).
// Included from skfast.hip.h after LinJob / LinMulti / load_q_raw are defined.
#pragma once
#if HY4_CUDA

namespace hy4 {
namespace skd {

__device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// Unpack 8 weights (8*u .. 8*u+7) of super-block payload `pl` into two signed-int8x4 words:
//   we = [q0, q2, q4, q6],  wo = [q1, q3, q5, q7]
// For S2..S6, q in 0..63 (fits in signed int8); for S8, q in -128..127 directly.
template <int FMT>
__device__ __forceinline__ void decode_eo_raw(uint32_t a, uint32_t b, int& we, int& wo) {
    if constexpr (FMT == FMT_S2) {
        we = (int)((a & 0x0003u) | ((a & 0x0030u) << 4) | ((a & 0x0300u) << 8) | ((a & 0x3000u) << 12));
        wo = (int)(((a & 0x000Cu) >> 2) | ((a & 0x00C0u) << 2) | ((a & 0x0C00u) << 6) | ((a & 0xC000u) << 10));
    } else if constexpr (FMT == FMT_S3) {
        const uint32_t le = (a & 0x0003u) | ((a & 0x0030u) << 4) | ((a & 0x0300u) << 8) | ((a & 0x3000u) << 12);
        const uint32_t lo = ((a & 0x000Cu) >> 2) | ((a & 0x00C0u) << 2) | ((a & 0x0C00u) << 6) | ((a & 0xC000u) << 10);
        const uint32_t he = ((b & 0x01u) << 2) | ((b & 0x04u) << 8) | ((b & 0x10u) << 14) | ((b & 0x40u) << 20);
        const uint32_t ho = ((b & 0x02u) << 1) | ((b & 0x08u) << 7) | ((b & 0x20u) << 13) | ((b & 0x80u) << 19);
        we = (int)(le | he);
        wo = (int)(lo | ho);
    } else if constexpr (FMT == FMT_S4) {
        we = (int)(a & 0x0F0F0F0Fu);
        wo = (int)((a >> 4) & 0x0F0F0F0Fu);
    } else if constexpr (FMT == FMT_S5) {
        const uint32_t he = ((b & 0x01u) << 4) | ((b & 0x04u) << 10) | ((b & 0x10u) << 16) | ((b & 0x40u) << 22);
        const uint32_t ho = ((b & 0x02u) << 3) | ((b & 0x08u) << 9) | ((b & 0x20u) << 15) | ((b & 0x80u) << 21);
        we = (int)((a & 0x0F0F0F0Fu) | he);
        wo = (int)(((a >> 4) & 0x0F0F0F0Fu) | ho);
    } else if constexpr (FMT == FMT_S6) {
        const uint32_t he = ((b & 0x0003u) << 4) | ((b & 0x0030u) << 8) | ((b & 0x0300u) << 12) | ((b & 0x3000u) << 16);
        const uint32_t ho = ((b & 0x000Cu) << 2) | ((b & 0x00C0u) << 6) | ((b & 0x0C00u) << 10) | ((b & 0xC000u) << 14);
        we = (int)((a & 0x0F0F0F0Fu) | he);
        wo = (int)(((a >> 4) & 0x0F0F0F0Fu) | ho);
    } else {  // FMT_S8: signed int8 bytes directly
        we = (int)__byte_perm(a, b, 0x6420u);
        wo = (int)__byte_perm(a, b, 0x7531u);
    }
}

template <int FMT>
__device__ __forceinline__ void decode_eo8(const uint8_t* pl, int u, int& we, int& wo) {
    uint32_t a, b;
    load_q_raw<FMT>(pl, u, a, b);
    decode_eo_raw<FMT>(a, b, we, wo);
}

// Quantize 8 floats x[0..7] owned by one lane of an 8-lane (64-element) sub-group into 2-term Q8_1.
// All 8 lanes (lane & 7 == 0..7) must be converged. Outputs:
//   qa = {qa_even, qa_odd}, qb = {qb_even, qb_odd}, da, db, and sx8 = 0.125f * sum_64(da*qa + db*qb).
__device__ __forceinline__ void quant8_sub64(const float (&x)[8], int2& qa_out, int2& qb_out, float& da_out, float& db_out, float& sx8_out) {
    float am = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) am = fmaxf(am, fabsf(x[i]));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1, 8));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2, 8));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4, 8));
    const float da = am * (1.0f / 127.0f);
    const float inva = da > 0.f ? (1.0f / da) : 0.f;
    int qa[8];
    float r[8];
    int sqa = 0;
    float amb = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int q = __float2int_rn(x[i] * inva);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        qa[i] = q;
        sqa += q;
        r[i] = x[i] - da * (float)q;
        amb = fmaxf(amb, fabsf(r[i]));
    }
    amb = fmaxf(amb, __shfl_xor_sync(0xffffffffu, amb, 1, 8));
    amb = fmaxf(amb, __shfl_xor_sync(0xffffffffu, amb, 2, 8));
    amb = fmaxf(amb, __shfl_xor_sync(0xffffffffu, amb, 4, 8));
    const float db = amb * (1.0f / 127.0f);
    const float invb = db > 0.f ? (1.0f / db) : 0.f;
    int qb[8];
    int sqb = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int q = __float2int_rn(r[i] * invb);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        qb[i] = q;
        sqb += q;
    }
    sqa += __shfl_xor_sync(0xffffffffu, sqa, 1, 8);
    sqa += __shfl_xor_sync(0xffffffffu, sqa, 2, 8);
    sqa += __shfl_xor_sync(0xffffffffu, sqa, 4, 8);
    sqb += __shfl_xor_sync(0xffffffffu, sqb, 1, 8);
    sqb += __shfl_xor_sync(0xffffffffu, sqb, 2, 8);
    sqb += __shfl_xor_sync(0xffffffffu, sqb, 4, 8);
    qa_out = make_int2(pack_i8x4(qa[0], qa[2], qa[4], qa[6]), pack_i8x4(qa[1], qa[3], qa[5], qa[7]));
    qb_out = make_int2(pack_i8x4(qb[0], qb[2], qb[4], qb[6]), pack_i8x4(qb[1], qb[3], qb[5], qb[7]));
    da_out = da;
    db_out = db;
    sx8_out = 0.125f * (da * (float)sqa + db * (float)sqb);
}

__device__ __forceinline__ void load_half8_f32(const __half* p, float (&x)[8]) {
    const uint4 u = *(const uint4*)p;
    h2_unpack(u.x, x[0], x[1]);
    h2_unpack(u.y, x[2], x[3]);
    h2_unpack(u.z, x[4], x[5]);
    h2_unpack(u.w, x[6], x[7]);
}

// ------------------------------------------------------------------ __dp4a GEMV (M = 1..16, S2..S8)
// KC = 8192 / MT elements per K-tile: LDS = 16 KB (qa+qb) + 1.5 KB (da,db,sx8) = 17.5 KB (high occupancy on sm_75 64 KB/SM!).
template <int FMT, int MT, int RW>
__device__ __forceinline__ void gemv_dp4a_body(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M, int bidx) {
    constexpr int KC = (8192 / MT) & ~255;
    static_assert(KC >= 256 && (KC % 256) == 0, "KC must be a multiple of 256");
    constexpr int N8_MAX = KC / 8;
    constexpr int NG_MAX = KC / 64;
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr float HH = sk_half<FMT>();
    __shared__ int2 qa_s[MT * N8_MAX];
    __shared__ int2 qb_s[MT * N8_MAX];
    __shared__ float da_s[MT * NG_MAX];
    __shared__ float db_s[MT * NG_MAX];
    __shared__ float sx_s[MT * NG_MAX];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int row0 = (bidx * 8 + wave) * RW;
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
        const int n8 = kc >> 3;  // multiple of 32 (since K % 256 == 0 and KC >= 512)
        // Quantize X[0..MT-1][k0..k0+kc) in aligned 8-thread sub-groups (all 8 lanes converged).
        const int total8 = MT * n8;
        for (int i = tid; i < total8; i += 256) {
            const int mm = i / n8, c = i - mm * n8;
            float xv[8];
            if (mm < M) load_half8_f32(X + (size_t)mm * K + k0 + c * 8, xv);
            else {
#pragma unroll
                for (int j = 0; j < 8; ++j) xv[j] = 0.f;
            }
            int2 qa, qb;
            float da, db, sx8;
            quant8_sub64(xv, qa, qb, da, db, sx8);
            qa_s[mm * N8_MAX + c] = qa;
            qb_s[mm * N8_MAX + c] = qb;
            if ((c & 7) == 0) {
                const int gidx = mm * NG_MAX + (c >> 3);
                da_s[gidx] = da;
                db_s[gidx] = db;
                sx_s[gidx] = sx8;
            }
        }
        __syncthreads();
        const int sb0 = k0 >> 8, nsbc = kc >> 8;
        const int lg = lane >> 3;
#pragma unroll 2
        for (int s = 0; s < nsbc; ++s) {
            int2 qav[MT], qbv[MT];
            float dav[MT], dbv[MT], sxv[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const int ci = m * N8_MAX + s * 32 + lane;
                const int gi = m * NG_MAX + s * 4 + lg;
                qav[m] = qa_s[ci];
                qbv[m] = qb_s[ci];
                dav[m] = da_s[gi];
                dbv[m] = db_s[gi];
                sxv[m] = sx_s[gi];
            }
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                const uint8_t* blk = rp[r] + (size_t)(sb0 + s) * SB;
                const float eff = sk_eff<FMT>(blk, lg);
                int we, wo;
                decode_eo8<FMT>(blk + 6, lane, we, wo);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const int ia = hy4_dp4a(we, qav[m].x, hy4_dp4a(wo, qav[m].y, 0));
                    const int ib = hy4_dp4a(we, qbv[m].x, hy4_dp4a(wo, qbv[m].y, 0));
                    float dot = fmaf(dav[m], (float)ia, dbv[m] * (float)ib);
                    if constexpr (FMT != FMT_S8) dot = fmaf(-HH, sxv[m], dot);
                    acc[r][m] = fmaf(eff, dot, acc[r][m]);
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

template <int FMT, int MT, int RW>
__global__ void __launch_bounds__(256) k_gemv_dp4a(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    gemv_dp4a_body<FMT, MT, RW>(W, N, K, X, Y, M, blockIdx.x);
}

template <int FMT, int MT, int RW>
__global__ void __launch_bounds__(256) k_gemv_multi_dp4a(const LinJob* __restrict__ jobs, int njobs, int K, const __half* __restrict__ X, int M) {
    int j = 0;
    while (j + 1 < njobs && (int)blockIdx.x >= jobs[j + 1].blk0) ++j;
    const LinJob jb = jobs[j];
    gemv_dp4a_body<FMT, MT, RW>(jb.W, jb.N, K, X, jb.Y, M, (int)blockIdx.x - jb.blk0);
}

// Runtime toggle: HY4_DP4A=1 forces __dp4a GEMV; HY4_DP4A=0 forces fp16 h2_dot_f32 GEMV.
// Default: enabled on pre-Ampere (sm_70/sm_75) where __dp4a + 17.5 KB LDS beats fp16->fp32 FMA; optional on sm_80+.
inline bool use_dp4a_gemv() {
    static int cached = -1;
    if (cached < 0) {
        if (const char* e = getenv("HY4_DP4A")) {
            cached = atoi(e) != 0 ? 1 : 0;
        } else {
            hipDeviceProp_t dp{};
            cached = (hipGetDeviceProperties(&dp, 0) == hipSuccess && dp.major < 8) ? 1 : 0;
        }
    }
    return cached != 0;
}

template <int FMT, int MT>
inline void launch_gemv_dp4a(hipStream_t s, const void* W, int N, int K, const __half* X, float* Y, int M) {
    constexpr int RW = 2;
    dim3 grid((N + 8 * RW - 1) / (8 * RW)), block(256);
    hipLaunchKernelGGL((k_gemv_dp4a<FMT, MT, RW>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M);
}

template <int FMT>
inline void gemv_dp4a_by_m(hipStream_t s, const void* W, int N, int K, const __half* X, float* Y, int M) {
    if (M <= 1) launch_gemv_dp4a<FMT, 1>(s, W, N, K, X, Y, M);
    else if (M <= 2) launch_gemv_dp4a<FMT, 2>(s, W, N, K, X, Y, M);
    else if (M <= 4) launch_gemv_dp4a<FMT, 4>(s, W, N, K, X, Y, M);
    else if (M <= 8) launch_gemv_dp4a<FMT, 8>(s, W, N, K, X, Y, M);
    else launch_gemv_dp4a<FMT, 16>(s, W, N, K, X, Y, M);
}

inline bool linear_h_gemv_dp4a(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    if (!use_dp4a_gemv() || (K & 255) != 0) return false;
    switch (fmt) {
        case FMT_S2: gemv_dp4a_by_m<FMT_S2>(s, W, N, K, X, Y, M); return true;
        case FMT_S3: gemv_dp4a_by_m<FMT_S3>(s, W, N, K, X, Y, M); return true;
        case FMT_S4: gemv_dp4a_by_m<FMT_S4>(s, W, N, K, X, Y, M); return true;
        case FMT_S5: gemv_dp4a_by_m<FMT_S5>(s, W, N, K, X, Y, M); return true;
        case FMT_S6: gemv_dp4a_by_m<FMT_S6>(s, W, N, K, X, Y, M); return true;
        case FMT_S8: gemv_dp4a_by_m<FMT_S8>(s, W, N, K, X, Y, M); return true;
        default: return false;
    }
}

template <int FMT>
inline void multi_dp4a_by_m(hipStream_t s, const LinMulti& d, const __half* X, int M) {
#define HY4_MULTI_DP4A(MT) hipLaunchKernelGGL((k_gemv_multi_dp4a<FMT, MT, 2>), dim3(d.total_blocks), dim3(256), 0, s, d.d_jobs, d.njobs, d.K, X, M)
    if (M <= 1) { HY4_MULTI_DP4A(1); }
    else if (M <= 2) { HY4_MULTI_DP4A(2); }
    else if (M <= 4) { HY4_MULTI_DP4A(4); }
    else if (M <= 8) { HY4_MULTI_DP4A(8); }
    else { HY4_MULTI_DP4A(16); }
#undef HY4_MULTI_DP4A
}

inline bool linear_h_multi_dp4a(hipStream_t s, const LinMulti& d, const __half* X, int M) {
    if (!use_dp4a_gemv() || (d.K & 255) != 0) return false;
    switch (d.fmt) {
        case FMT_S2: multi_dp4a_by_m<FMT_S2>(s, d, X, M); return true;
        case FMT_S3: multi_dp4a_by_m<FMT_S3>(s, d, X, M); return true;
        case FMT_S4: multi_dp4a_by_m<FMT_S4>(s, d, X, M); return true;
        case FMT_S5: multi_dp4a_by_m<FMT_S5>(s, d, X, M); return true;
        case FMT_S6: multi_dp4a_by_m<FMT_S6>(s, d, X, M); return true;
        case FMT_S8: multi_dp4a_by_m<FMT_S8>(s, d, X, M); return true;
        default: return false;
    }
}

}  // namespace skd
}  // namespace hy4
#endif  // HY4_CUDA
