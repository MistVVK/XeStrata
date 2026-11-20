// sk_int8_rdna4.hip.h - RDNA4 (gfx1200/gfx1201) Native INT8 WMMA kernels for S_K weights (v_wmma_i32_16x16x16_iu8_w32_gfx12)
//
// Every S_K weight format maps with ZERO rounding error and ZERO zero-point offset to signed int8:
//   S2..S6: wi = 2*q - (2^b - 1) in [-63, 63],  sub-group scale = 0.5f * eff
//   S8:     wi = (int8_t)payload,               sub-group scale = eff
//   S1.25:  wi = kS125CB[idx][j] in [-2, 2],    sub-group scale = eff
//   S1.5:   wi = trit in [-1, 1],               sub-group scale = eff
// Weight LDS is int8_t (17-19 KB instead of 34-37 KB for fp16), enabling 2+ workgroups/CU on gfx12 (64 KB LDS/CU) and
// zero FP16 weight conversion overhead. Activations X (fp16) are quantized per 64-element sub-group on the fly into sXq[BMT][72]
// (<= 9.2 KB LDS, total LDS <= 27.5 KB).
//
// Modes (env HY4_INT8_WMMA):
//   0 = off (use all-8-wave FP16 WMMA in skwmma.hip.h)
//   1 = 1-term Q8_G64 INT8 WMMA (2x WMMA issue rate, ~4e-3 uniform rel_err)
//   2 = 2-term Q8_G64 INT8 WMMA (x = d0*q0 + d1*q1: ~3e-5 exact precision, passes <2e-3 unit tests & exact KL)
#pragma once
#include "skwmma.hip.h"
#include <cstring>
#include <type_traits>

#if HY4_MMA && !HY4_CUDA

namespace hy4 {
namespace skd {

typedef int iv2 __attribute__((ext_vector_type(2)));
typedef int iv8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ iv8 iv8_zero() { return iv8{0, 0, 0, 0, 0, 0, 0, 0}; }

__device__ __forceinline__ iv2 ld_iv2(const int8_t* p, int hi) {
    const uint2 a = *(const uint2*)(p + hi * 8);
    return __builtin_bit_cast(iv2, a);
}

__device__ __forceinline__ iv8 wmma_i8(iv2 a, iv2 b, iv8 c) {
#if HY4_GFX12
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    (void)a; (void)b;
    return c;
#endif
}

inline bool is_gfx12_device() {
    static bool v = []() {
        int dev = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return false;
        return strncmp(p.gcnArchName, "gfx12", 5) == 0;
    }();
    return v;
}

// Runtime knob (gfx12 only): 0 = FP16 WMMA, 1 = 1-term INT8 WMMA, 2 = 2-term INT8 WMMA.
inline int& int8_wmma_mode() {
    static int v = []() {
        if (!is_gfx12_device()) return 0;
        const char* e = getenv("HY4_INT8_WMMA");
        return e ? atoi(e) : 0;
    }();
    return v;
}

// LDS int8 codebook for S1.25 (32 entries x 4 int8 = 32 uint32 = 128 B)
__device__ __forceinline__ uint32_t* s125_i8_lut() {
    __shared__ uint32_t lut[32];
    return lut;
}
__device__ __forceinline__ void s125_i8_lut_init() {
    if (threadIdx.x < 32) {
        const int i = threadIdx.x;
        const uint32_t b0 = (uint8_t)kS125CB[i][0];
        const uint32_t b1 = (uint8_t)kS125CB[i][1];
        const uint32_t b2 = (uint8_t)kS125CB[i][2];
        const uint32_t b3 = (uint8_t)kS125CB[i][3];
        s125_i8_lut()[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}

template <int FMT>
__device__ __forceinline__ float raw_eff_i8(const RawSub<FMT>& r, int dg) {
    const float eff = raw_eff<FMT>(r, dg);
    if constexpr (FMT >= FMT_S2 && FMT <= FMT_S6) return 0.5f * eff;
    else return eff;
}

template <int FMT, int U8>
__device__ __forceinline__ void dequant_unit_i8(const RawSub<FMT>& r, int8_t* dst) {
    if constexpr (FMT == FMT_S8) {
        *(uint2*)dst = make_uint2(r.lo[2 * U8], r.lo[2 * U8 + 1]);
    } else if constexpr (FMT == FMT_S1_25) {
        constexpr int p = 10 * U8, wi = p >> 5, sh = p & 31;
        uint32_t bits;
        if constexpr (sh + 10 <= 32) bits = (r.lo[wi] >> sh) & 0x3FFu;
        else bits = ((r.lo[wi] >> sh) | (r.lo[wi + 1] << (32 - sh))) & 0x3FFu;
        const uint32_t* lut = s125_i8_lut();
        *(uint2*)dst = make_uint2(lut[bits & 31u], lut[bits >> 5]);
    } else if constexpr (FMT == FMT_S1_5) {
        const uint32_t t0 = r.hi[0], byte0 = t0 / 5u;
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t ti0 = t0 + 8 * U8 + k;
            const uint32_t ti1 = t0 + 8 * U8 + 4 + k;
            const int v0 = s15_trit(byte_at4(r.lo, (int)(ti0 / 5u - byte0)), (int)(ti0 % 5u));
            const int v1 = s15_trit(byte_at4(r.lo, (int)(ti1 / 5u - byte0)), (int)(ti1 % 5u));
            w0 |= ((uint32_t)(uint8_t)(int8_t)v0) << (8 * k);
            w1 |= ((uint32_t)(uint8_t)(int8_t)v1) << (8 * k);
        }
        *(uint2*)dst = make_uint2(w0, w1);
    } else {
        constexpr int BIAS = FMT == FMT_S2 ? 3 : FMT == FMT_S3 ? 7 : FMT == FMT_S4 ? 15 : FMT == FMT_S5 ? 31 : 63;
        uint32_t q[8];
        raw_q8<FMT, U8>(r, q);
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int8_t a = (int8_t)((int)(q[k] << 1) - BIAS);
            const int8_t b = (int8_t)((int)(q[4 + k] << 1) - BIAS);
            w0 |= ((uint32_t)(uint8_t)a) << (8 * k);
            w1 |= ((uint32_t)(uint8_t)b) << (8 * k);
        }
        *(uint2*)dst = make_uint2(w0, w1);
    }
}

template <int FMT>
__device__ __forceinline__ void dequant_sub_i8(const RawSub<FMT>& r, int8_t* dst) {
    if constexpr (FMT == FMT_S8) {
        uint4* d4 = (uint4*)dst;
        d4[0] = make_uint4(r.lo[0], r.lo[1], r.lo[2], r.lo[3]);
        d4[1] = make_uint4(r.lo[4], r.lo[5], r.lo[6], r.lo[7]);
        d4[2] = make_uint4(r.lo[8], r.lo[9], r.lo[10], r.lo[11]);
        d4[3] = make_uint4(r.lo[12], r.lo[13], r.lo[14], r.lo[15]);
    } else {
        dequant_unit_i8<FMT, 0>(r, dst + 0);
        dequant_unit_i8<FMT, 1>(r, dst + 8);
        dequant_unit_i8<FMT, 2>(r, dst + 16);
        dequant_unit_i8<FMT, 3>(r, dst + 24);
        dequant_unit_i8<FMT, 4>(r, dst + 32);
        dequant_unit_i8<FMT, 5>(r, dst + 40);
        dequant_unit_i8<FMT, 6>(r, dst + 48);
        dequant_unit_i8<FMT, 7>(r, dst + 56);
    }
}

// Quantize BMT activation rows x 64 halves (at column offset k0) into sXq[BMT][72] and sXd[BMT].
// All 256 threads cooperate: TPR = 256 / BMT threads per row (2, 4, 8, or 16), EPT = 64 / TPR elements per thread (32, 16, 8, or 4).
// Zero barriers needed inside this function (uses warp-local shfl_xor across TPR lanes!).
// When HI == true, also computes the residual term (q1, d1) into caller registers for zero extra global loads.
template <int BMT, bool HI>
__device__ __forceinline__ void quant_x_sg64(const half_t* xptr, int k0, int8_t* sXq, float* sXd, uint32_t* q1_out, float& d1_out) {
    constexpr int TPR = 256 / BMT;    // threads per row: 2 (BM=128), 4 (BM=64), 8 (BM=32), 16 (BM=16)
    constexpr int EPT = 64 / TPR;     // elements per thread: 32, 16, 8, 4
    constexpr int NW4 = EPT / 4;      // int8 packed uint32 words per thread: 8, 4, 2, 1
    const int tid = threadIdx.x;
    const int r = tid / TPR;
    const int sub = tid & (TPR - 1);
    const half_t* src = xptr + k0 + sub * EPT;

    float xf[EPT];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < NW4; ++i) {
        const uint2 u = *(const uint2*)(src + i * 4);
        float a, b, c, d;
        h2_unpack(u.x, a, b);
        h2_unpack(u.y, c, d);
        xf[4 * i + 0] = a;
        xf[4 * i + 1] = b;
        xf[4 * i + 2] = c;
        xf[4 * i + 3] = d;
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a), fabsf(b)), fmaxf(fabsf(c), fabsf(d))));
    }
#pragma unroll
    for (int off = TPR >> 1; off >= 1; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off));

    const float d0 = amax * (1.0f / 127.0f);
    const float inv0 = amax > 0.f ? (127.0f / amax) : 0.f;
    uint32_t q0w[NW4];
    float amax1 = 0.f;
#pragma unroll
    for (int i = 0; i < NW4; ++i) {
        int v0 = __float2int_rn(xf[4 * i + 0] * inv0);
        int v1 = __float2int_rn(xf[4 * i + 1] * inv0);
        int v2 = __float2int_rn(xf[4 * i + 2] * inv0);
        int v3 = __float2int_rn(xf[4 * i + 3] * inv0);
        v0 = max(-127, min(127, v0));
        v1 = max(-127, min(127, v1));
        v2 = max(-127, min(127, v2));
        v3 = max(-127, min(127, v3));
        q0w[i] = ((uint32_t)(uint8_t)(int8_t)v0) | (((uint32_t)(uint8_t)(int8_t)v1) << 8) | (((uint32_t)(uint8_t)(int8_t)v2) << 16) | (((uint32_t)(uint8_t)(int8_t)v3) << 24);
        if constexpr (HI) {
            xf[4 * i + 0] -= d0 * (float)v0;
            xf[4 * i + 1] -= d0 * (float)v1;
            xf[4 * i + 2] -= d0 * (float)v2;
            xf[4 * i + 3] -= d0 * (float)v3;
            amax1 = fmaxf(amax1, fmaxf(fmaxf(fabsf(xf[4 * i + 0]), fabsf(xf[4 * i + 1])), fmaxf(fabsf(xf[4 * i + 2]), fabsf(xf[4 * i + 3]))));
        }
    }
    uint32_t* dst = (uint32_t*)(sXq + r * 72 + sub * EPT);
#pragma unroll
    for (int i = 0; i < NW4; ++i) dst[i] = q0w[i];
    if (sub == 0) sXd[r] = d0;

    if constexpr (HI) {
#pragma unroll
        for (int off = TPR >> 1; off >= 1; off >>= 1) amax1 = fmaxf(amax1, __shfl_xor(amax1, off));
        d1_out = amax1 * (1.0f / 127.0f);
        const float inv1 = amax1 > 0.f ? (127.0f / amax1) : 0.f;
#pragma unroll
        for (int i = 0; i < NW4; ++i) {
            int v0 = max(-127, min(127, __float2int_rn(xf[4 * i + 0] * inv1)));
            int v1 = max(-127, min(127, __float2int_rn(xf[4 * i + 1] * inv1)));
            int v2 = max(-127, min(127, __float2int_rn(xf[4 * i + 2] * inv1)));
            int v3 = max(-127, min(127, __float2int_rn(xf[4 * i + 3] * inv1)));
            q1_out[i] = ((uint32_t)(uint8_t)(int8_t)v0) | (((uint32_t)(uint8_t)(int8_t)v1) << 8) | (((uint32_t)(uint8_t)(int8_t)v2) << 16) | (((uint32_t)(uint8_t)(int8_t)v3) << 24);
        }
    }
}

template <int BMT>
__device__ __forceinline__ void store_x_sg64_hi(int8_t* sXq, float* sXd, const uint32_t* q1, float d1) {
    constexpr int TPR = 256 / BMT;
    constexpr int EPT = 64 / TPR;
    constexpr int NW4 = EPT / 4;
    const int tid = threadIdx.x;
    const int r = tid / TPR;
    const int sub = tid & (TPR - 1);
    uint32_t* dst = (uint32_t*)(sXq + r * 72 + sub * EPT);
#pragma unroll
    for (int i = 0; i < NW4; ++i) dst[i] = q1[i];
    if (sub == 0) sXd[r] = d1;
}

// Generic RDNA4 Native INT8 WMMA main loop.
// BN_W * BK == 16384: (64, 256), (128, 128), (256, 64).
// x_qptr: activation row pointer for this thread's quantize row r = tid / (256 / BMT) (start of K, fp16).
// wm_base: starting m-row inside sXq for this wave (wm * MTW * 16).
// brow[NTW]: starting n-row inside sWq for each n-tile of this wave.
template <int FMT, int BMT, int MTW, int NTW, int BN_W, int BK, bool HI>
__device__ __forceinline__ void gemm_mainloop_i8_rdna4(const uint8_t* wr, int nsb, const half_t* x_qptr, int wm_base, const int (&brow)[NTW],
                                                       int8_t* sWq, float* sWs, int8_t* sXq, float* sXd, fv8 (&acc)[MTW][NTW]) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int G = BK / 64;          // 4 (BK=256), 2 (BK=128), 1 (BK=64)
    constexpr int ST = 4 / G;           // steps per super-block: 1, 2, 4
    constexpr int LDW = BK + 8;         // 264, 136, 72 (8-byte aligned, bank-conflict-free)
    constexpr int NW4 = (64 / (256 / BMT)) / 4;

    const int tid = threadIdx.x, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int drow = tid / G, sg_t = tid & (G - 1);

    if constexpr (FMT == FMT_S1_25) {
        s125_i8_lut_init();
        hy4_sync_lds();
    }

    RawSub<FMT> cur, nxt;
    load_raw<FMT>(wr, sg_t, cur);

    const int niter = nsb * ST;
    for (int it = 0; it < niter; ++it) {
        const int sb = it / ST;
        const int sub = it % ST;
        const int dg = sub * G + sg_t;

        // Dequant this tile's weights to int8 + store per-group scale
        dequant_sub_i8<FMT>(cur, &sWq[drow * LDW + sg_t * 64]);
        sWs[drow * G + sg_t] = raw_eff_i8<FMT>(cur, dg);

        // Prefetch next weight tile while computing this tile
        if (it + 1 < niter) {
            const int sb_n = (it + 1) / ST;
            const int dg_n = ((it + 1) % ST) * G + sg_t;
            load_raw<FMT>(wr + (size_t)sb_n * SB, dg_n, nxt);
        }

        // Process the G sub-groups of 64 k-elements inside this BK tile
#pragma unroll
        for (int sg = 0; sg < G; ++sg) {
            const int k0 = sb * 256 + (sub * G + sg) * 64;
            uint32_t q1[HI ? NW4 : 1];
            float d1 = 0.f;
            quant_x_sg64<BMT, HI>(x_qptr, k0, sXq, sXd, q1, d1);
            hy4_sync_lds();

            // Term 0: 4 INT8 WMMAs covering K=64
            iv8 iacc[MTW][NTW];
#pragma unroll
            for (int m = 0; m < MTW; ++m)
#pragma unroll
                for (int j = 0; j < NTW; ++j) iacc[m][j] = iv8_zero();

#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                iv2 a[MTW];
#pragma unroll
                for (int m = 0; m < MTW; ++m) a[m] = ld_iv2(&sXq[(wm_base + m * 16 + l16) * 72 + kk * 16], hi);
                iv2 b[NTW];
#pragma unroll
                for (int j = 0; j < NTW; ++j) b[j] = ld_iv2(&sWq[(brow[j] + l16) * LDW + sg * 64 + kk * 16], hi);
#pragma unroll
                for (int m = 0; m < MTW; ++m)
#pragma unroll
                    for (int j = 0; j < NTW; ++j) iacc[m][j] = wmma_i8(a[m], b[j], iacc[m][j]);
            }

            // Rescale int32 accumulators into fp32 acc
            float ws_n[NTW];
#pragma unroll
            for (int j = 0; j < NTW; ++j) ws_n[j] = sWs[(brow[j] + l16) * G + sg];
#pragma unroll
            for (int m = 0; m < MTW; ++m) {
                float xd_m[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) xd_m[r] = sXd[wm_base + m * 16 + 8 * hi + r];
#pragma unroll
                for (int j = 0; j < NTW; ++j) {
                    const float wn = ws_n[j];
#pragma unroll
                    for (int r = 0; r < 8; ++r) acc[m][j][r] += (float)iacc[m][j][r] * (xd_m[r] * wn);
                }
            }

            if constexpr (HI) {
                hy4_sync_lds();
                store_x_sg64_hi<BMT>(sXq, sXd, q1, d1);
                hy4_sync_lds();

                iv8 iacc1[MTW][NTW];
#pragma unroll
                for (int m = 0; m < MTW; ++m)
#pragma unroll
                    for (int j = 0; j < NTW; ++j) iacc1[m][j] = iv8_zero();

#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    iv2 a[MTW];
#pragma unroll
                    for (int m = 0; m < MTW; ++m) a[m] = ld_iv2(&sXq[(wm_base + m * 16 + l16) * 72 + kk * 16], hi);
                    iv2 b[NTW];
#pragma unroll
                    for (int j = 0; j < NTW; ++j) b[j] = ld_iv2(&sWq[(brow[j] + l16) * LDW + sg * 64 + kk * 16], hi);
#pragma unroll
                    for (int m = 0; m < MTW; ++m)
#pragma unroll
                        for (int j = 0; j < NTW; ++j) iacc1[m][j] = wmma_i8(a[m], b[j], iacc1[m][j]);
                }
#pragma unroll
                for (int m = 0; m < MTW; ++m) {
                    float xd_m[8];
#pragma unroll
                    for (int r = 0; r < 8; ++r) xd_m[r] = sXd[wm_base + m * 16 + 8 * hi + r];
#pragma unroll
                    for (int j = 0; j < NTW; ++j) {
                        const float wn = ws_n[j];
#pragma unroll
                        for (int r = 0; r < 8; ++r) acc[m][j][r] += (float)iacc1[m][j][r] * (xd_m[r] * wn);
                    }
                }
            }

            hy4_sync_lds();
        }
        cur = nxt;
    }
}

// ---- Dense linear INT8 WMMA kernels (RDNA4) ----
template <int FMT, bool HI>
__global__ void __launch_bounds__(256) k_gemm_i8_rdna4(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int BN_W = 64, BK = 256, G = 4, LDW = BK + 8, BMT = 128;
    __shared__ __attribute__((aligned(16))) int8_t sWq[BN_W * LDW];   // 16,896 B
    __shared__ __attribute__((aligned(16))) float sWs[BN_W * G];      //  1,024 B
    __shared__ __attribute__((aligned(16))) int8_t sXq[BMT * 72];     //  9,216 B
    __shared__ __attribute__((aligned(16))) float sXd[BMT];           //    512 B  (total = 27.6 KB -> 2 WG/CU!)

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave & 3, wn = wave >> 2;
    const int bn = blockIdx.x * BN_W, bm = blockIdx.y * BMT;
    const int nsb = K >> 8;

    int qrow = bm + (tid >> 1);
    qrow = qrow < M ? qrow : M - 1;
    const half_t* x_qptr = (const half_t*)X + (size_t)qrow * K;

    int wrow = bn + (tid >> 2);
    wrow = wrow < N ? wrow : N - 1;
    const int brow[2] = {wn * 32, wn * 32 + 16};

    fv8 acc[2][2];
    acc_zero(acc);
    gemm_mainloop_i8_rdna4<FMT, BMT, 2, 2, BN_W, BK, HI>(W + (size_t)wrow * nsb * SB, nsb, x_qptr, wm * 32, brow, sWq, sWs, sXq, sXd, acc);

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

template <int FMT, bool HI>
__global__ void __launch_bounds__(256) k_gemm_s_i8_rdna4(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int BN_W = 128, BK = 128, G = 2, LDW = BK + 8, BMT = 16;
    __shared__ __attribute__((aligned(16))) int8_t sWq[BN_W * LDW];   // 17,408 B
    __shared__ __attribute__((aligned(16))) float sWs[BN_W * G];      //  1,024 B
    __shared__ __attribute__((aligned(16))) int8_t sXq[BMT * 72];     //  1,152 B
    __shared__ __attribute__((aligned(16))) float sXd[BMT];           //     64 B  (total = 19.6 KB -> 3 WG/CU!)

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int bn = blockIdx.x * BN_W, bm = blockIdx.y * BMT;
    const int nsb = K >> 8;

    int qrow = bm + (tid >> 4);
    qrow = qrow < M ? qrow : M - 1;
    const half_t* x_qptr = (const half_t*)X + (size_t)qrow * K;

    int wrow = bn + (tid >> 1);
    wrow = wrow < N ? wrow : N - 1;
    const int brow[1] = {wave * 16};

    fv8 acc[1][1];
    acc_zero(acc);
    gemm_mainloop_i8_rdna4<FMT, BMT, 1, 1, BN_W, BK, HI>(W + (size_t)wrow * nsb * SB, nsb, x_qptr, 0, brow, sWq, sWs, sXq, sXd, acc);

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int n = bn + wave * 16 + acc_n(r, lane);
        const int m = bm + acc_m(r, lane);
        if (m < M && n < N) Y[(size_t)m * N + n] = acc[0][0][r];
    }
}

template <bool HI>
inline void launch_gemm_i8_rdna4(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    dim3 grid((N + 63) / 64, (M + 127) / 128), block(256);
#define HY4_I8_CASE(F) case F: hipLaunchKernelGGL((k_gemm_i8_rdna4<F, HI>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M); break
    switch (fmt) {
        HY4_I8_CASE(FMT_S1_25);
        HY4_I8_CASE(FMT_S1_5);
        HY4_I8_CASE(FMT_S2);
        HY4_I8_CASE(FMT_S3);
        HY4_I8_CASE(FMT_S4);
        HY4_I8_CASE(FMT_S5);
        HY4_I8_CASE(FMT_S6);
        HY4_I8_CASE(FMT_S8);
        default: fprintf(stderr, "launch_gemm_i8_rdna4: unsupported fmt %d\n", fmt); abort();
    }
#undef HY4_I8_CASE
}

template <bool HI>
inline void launch_gemm_s_i8_rdna4(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
    dim3 grid((N + 127) / 128, (M + 15) / 16), block(256);
#define HY4_I8S_CASE(F) case F: hipLaunchKernelGGL((k_gemm_s_i8_rdna4<F, HI>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M); break
    switch (fmt) {
        HY4_I8S_CASE(FMT_S1_25);
        HY4_I8S_CASE(FMT_S1_5);
        HY4_I8S_CASE(FMT_S2);
        HY4_I8S_CASE(FMT_S3);
        HY4_I8S_CASE(FMT_S4);
        HY4_I8S_CASE(FMT_S5);
        HY4_I8S_CASE(FMT_S6);
        HY4_I8S_CASE(FMT_S8);
        default: fprintf(stderr, "launch_gemm_s_i8_rdna4: unsupported fmt %d\n", fmt); abort();
    }
#undef HY4_I8S_CASE
}

}  // namespace skd
}  // namespace hy4

#endif  // HY4_MMA && !HY4_CUDA
