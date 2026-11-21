// engine/src/inkling2/cuda/ink_moe_fp4_sm120.cu - Clean-room MIT Blackwell (sm_120a / sm_121a) native mxf4nvf4 & mxf8f6f4 MoE kernels
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Targets NVIDIA PTX ISA 8.7+ block-scaled Tensor Core MMA instructions on GeForce RTX 50-series / RTX PRO Blackwell (sm_120a)
// and DGX Spark GB10 (sm_121a):
//   - SF4 (NVFP4, L0..19 = 44/48 visits):
//       mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//   - MX8 (MXFP8, L20..23 = 4/48 visits):
//       mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
//
// Zero bit-reordering needed! .strata v1.1 SF4 stores sequential E2M1x2 nibbles (lo=even, hi=odd, 16-byte aligned rows:
// 2304/2 = 1152 B and 1152/2 = 576 B, both multiples of 32 B!) with contiguous 4-byte-aligned F8_E4M3 scales per 16 weights.
//
// Activation FP4 slot (48 B per 64 values, 16-byte aligned for cp.async):
//   bytes [ 0..31]: uint8_t qs[32]   (64 sequential E2M1 values, 2 per byte: lo=2j, hi=2j+1)
//   bytes [32..35]: uint8_t d[4]     (4 UE4M3 sub-block scales for sub = 0..3)
//   bytes [36..39]: float   g_blk    (per-64-element FP32 global scale = amax_64 / (6 * 448))
//   bytes [40..47]: padding
#include "ink_moe_fp4_sm120.cuh"
#include "ink_cuda.cuh"

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
#include <cuda_fp4.h>
#endif
#include <cuda_fp8.h>

#include <algorithm>

namespace strata::ink::sm120 {
namespace {

constexpr int kHidden = kInkHidden;  // 2304 = 36 * 64 = 72 * 32
constexpr int kInter  = kInkInter;   // 1152 = 18 * 64 = 36 * 32
constexpr int kAB     = kActFp4SlotBytes;  // 48 B per 64 values

constexpr int kNT = 256;    // 8 warps: wf in 0..1 (64 weight rows each), wn in 0..3 (16 token rows each)
constexpr int kTR = 64;     // routed token rows per tile
constexpr int kWR = 128;    // weight rows per CTA (GU: 64 interleaved gate+up pairs = 64 SITU outputs; DN: 128 outputs)
[[maybe_unused]] constexpr int kWStride4 = 48; // 32B qs + 16B pad (bank-conflict free ldmatrix.m8n8.x4)
[[maybe_unused]] constexpr int kAStride4 = 48; // 32B qs + 16B pad

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && (__CUDA_ARCH__ < 1300)
#define INK_SM120_DEV 1
#endif

#if defined(INK_SM120_DEV)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp16_async(void* dst_smem, const void* src_glob) {
    const uint32_t d = smem_u32(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src_glob));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait0()  { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_wait1()  { asm volatile("cp.async.wait_group 1;\n"); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* src_smem) {
    const uint32_t a = smem_u32(src_smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void mma_nvfp4_m16n8k64(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                                   uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

__device__ __forceinline__ void mma_mxfp8_m16n8k32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                                   uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

__device__ __forceinline__ uint8_t encode_ue4m3(float x, float& decoded) {
    if (!(x > 0.0f)) {
        decoded = 0.0f;
        return 0;
    }
    const float clamped = fminf(x, 448.0f);
    const __nv_fp8_e4m3 fp8(clamped);
    decoded = (float)fp8;
    return fp8.__x;
}

__device__ __forceinline__ uint8_t pack_e2m1x2(float v_lo, float v_hi) {
    const __nv_fp4x2_e2m1 q2(make_float2(v_lo, v_hi));
    return q2.__x;
}

__device__ __forceinline__ uint8_t encode_e4m3_signed(float x) {
    const float clamped = fminf(fmaxf(x, -448.0f), 448.0f);
    const __nv_fp8_e4m3 fp8(clamped);
    return fp8.__x;
}

// Compute UE8M0 exponent byte e so 2^(e-127) >= amax / 448.0f (smallest power-of-2 scale covering amax in E4M3).
__device__ __forceinline__ uint8_t encode_ue8m0(float amax, float& inv_scale) {
    if (!(amax > 0.0f)) {
        inv_scale = 0.0f;
        return 255u;  // zero scale sentinel
    }
    const float need = amax * (1.0f / 448.0f);
    uint32_t bits = __float_as_uint(need);
    // Round up to next power of 2 if mantissa != 0
    if (bits & 0x007FFFFFu) bits = (bits & 0xFF800000u) + 0x00800000u;
    int exp = (int)((bits >> 23) & 0xFFu);
    if (exp < 1) exp = 1;
    if (exp > 254) exp = 254;
    const float sc = __uint_as_float((uint32_t)exp << 23);
    inv_scale = 1.0f / sc;
    return (uint8_t)exp;
}

#endif  // INK_SM120_DEV

// ---- Activation FP4 quantizer: 1 warp per 64 FP32 values -> 48-byte slot (sequential E2M1x2 matching .strata SF4!)
__global__ void k_quant_act_fp4(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
#if defined(INK_SM120_DEV)
    const int64_t w = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= nblk) return;
    const int lane = threadIdx.x & 31;
    const int sub = lane >> 3;   // 0..3 (four 16-element sub-blocks)
    const int j   = lane & 7;    // 0..7 -> elements 2*j and 2*j + 1 inside sub-block!

    const float* xb = x + w * 64;
    const float v0 = xb[16 * sub + 2 * j];
    const float v1 = xb[16 * sub + 2 * j + 1];

    float am_sub = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) am_sub = fmaxf(am_sub, __shfl_xor_sync(0xFFFFFFFFu, am_sub, o));

    float am_64 = fmaxf(am_sub, __shfl_xor_sync(0xFFFFFFFFu, am_sub, 8));
    am_64 = fmaxf(am_64, __shfl_xor_sync(0xFFFFFFFFu, am_64, 16));

    const float g_blk = am_64 * (1.0f / (6.0f * 448.0f));
    float s_sub = 0.0f;
    const uint8_t d_code = encode_ue4m3(g_blk > 0.0f ? (am_sub / (g_blk * 6.0f)) : 0.0f, s_sub);
    const float denom = g_blk * s_sub;
    const float inv = denom > 0.0f ? (1.0f / denom) : 0.0f;

    uint8_t* out = xa + w * kAB;
    out[lane] = pack_e2m1x2(v0 * inv, v1 * inv);
    if (j == 0) out[32 + sub] = d_code;
    if (lane == 0) *(float*)(out + 36) = g_blk;
#else
    (void)x; (void)nblk; (void)xa;
#endif
}

// ---- Activation MX8 quantizer: 1 warp per 32 FP32 values -> 32B E4M3 + 1B UE8M0
__global__ void k_quant_act_mx8(const float* __restrict__ x, int64_t nblk,
                                uint8_t* __restrict__ xq, uint8_t* __restrict__ xsc) {
#if defined(INK_SM120_DEV)
    const int64_t w = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= nblk) return;
    const int lane = threadIdx.x & 31;
    const float v = x[w * 32 + lane];
    float am = fabsf(v);
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, o));
    float inv_sc = 0.0f;
    const uint8_t sc_byte = encode_ue8m0(am, inv_sc);
    xq[w * 32 + lane] = encode_e4m3_signed(v * inv_sc);
    if (lane == 0) xsc[w] = sc_byte;
#else
    (void)x; (void)nblk; (void)xq; (void)xsc;
#endif
}

// ---- Persistent Blackwell sm_120a mxf4nvf4 MoE kernel (TR in {16, 32, 64}; GU=true: Gate+Up+SITU+FP4-requant; GU=false: Down)
// TR=16 (n_wf=8, n_wn=1): 100% warp utilization for c=16..128 batched decode!
// TR=32 (n_wf=4, n_wn=2): 100% warp utilization for M=129..768 prefill (avg 32 tok/expert at M=512)!
// TR=64 (n_wf=2, n_wn=4): max weight reuse for M>768 prefill!
template <bool GU, int TR>
__global__ void __launch_bounds__(kNT) k_moe_sf4_tc(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ act,
    uint8_t* __restrict__ out_ha,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair) {
#if defined(INK_SM120_DEV)
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;  // 2304
    constexpr int ROWS_K   = GU ? kHidden : kInter;        // 2304 (GU) or 1152 (DN)
    constexpr int NS       = ROWS_K / 64;                  // 36 (GU) or 18 (DN)
    constexpr int NFB      = ROWS_N / kWR;                 // 18 blocks of 128 rows
    constexpr int N_WN     = TR / 16;                      // 1, 2, or 4
    constexpr int N_WF     = 8 / N_WN;                     // 8, 4, or 2
    constexpr int W_PER_WF = kWR / N_WF;                   // 16, 32, or 64
    constexpr int NI       = W_PER_WF / 16;                // 1, 2, or 4
    constexpr size_t W_P_ROW   = ROWS_K / 2;
    constexpr size_t W_S_ROW   = ROWS_K / 16;
    constexpr size_t ACT_ROW_B = (size_t)NS * kAB;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kAStride4];
    __shared__ uint32_t s_wsc[2][kWR];
    __shared__ uint32_t s_asc[2][TR];
    __shared__ float    s_agb[2][TR];
    __shared__ int32_t  s_row[TR];
    __shared__ float    s_am[N_WF][TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;  // 0..N_WF-1
    const int wn   = warp % N_WN;  // 0..N_WN-1
    const int nb0  = 16 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1;
    const int uj = tid & 1;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_SF4) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        float hdr[4];
        memcpy(hdr, body, 16);
        const float w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);

        const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;

        const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                                : (fb * kWR + ur);
        const uint8_t* wrow_p = base_p + (size_t)phys_row * W_P_ROW;
        const uint8_t* wrow_s = base_s + (size_t)phys_row * W_S_ROW;

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
                s_agb[0][tid] = 0.0f; s_agb[1][tid] = 0.0f;
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int st, int buf) {
            cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
            if (uj == 0) s_wsc[buf][ur] = *(const uint32_t*)(wrow_s + (size_t)st * 4);
            if (tid < 2 * TR) {
                const int r = tid >> 1;
                const int h = tid & 1;
                if (r < nrows) {
                    const uint8_t* ap = act + (size_t)s_row[r] * ACT_ROW_B + (size_t)st * kAB;
                    cp16_async(s_aqs[buf] + r * kAStride4 + 16 * h, ap + 16 * h);
                }
            } else if (tid < 3 * TR) {
                const int r = tid - 2 * TR;
                if (r < nrows) {
                    const uint8_t* ap = act + (size_t)s_row[r] * ACT_ROW_B + (size_t)st * kAB;
                    const uint2 meta = *(const uint2*)(ap + 32);
                    s_asc[buf][r] = meta.x;
                    s_agb[buf][r] = __uint_as_float(meta.y);
                }
            }
            cp_commit();
        };

        issue_stage(0, 0);

        float acc[NI][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int st = 0; st < NS; ++st) {
            const int buf = st & 1;
            const int nxt = buf ^ 1;
            if (st + 1 < NS) {
                issue_stage(st + 1, nxt);
                cp_wait1();
            } else {
                cp_wait0();
            }
            __syncthreads();

            if (on0) {
                uint32_t bq[4];
                ldsm4(bq, s_aqs[buf] + b_off);
                const uint32_t sfb0 = s_asc[buf][nb0 + g];
                const uint32_t sfb1 = on1 ? s_asc[buf][nb0 + 8 + g] : 0u;
                const float gb00 = s_agb[buf][nb0 + 2 * tig + 0];
                const float gb01 = s_agb[buf][nb0 + 2 * tig + 1];
                const float gb10 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 0] : 0.0f;
                const float gb11 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 1] : 0.0f;

#pragma unroll
                for (int i = 0; i < NI; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                    const uint32_t sfa = s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a];

                    float d0[4] = {};
                    mma_nvfp4_m16n8k64(d0, a, bq[0], bq[1], sfa, sfb0);
                    acc[i][0][0] = fmaf(d0[0], gb00, acc[i][0][0]);
                    acc[i][0][1] = fmaf(d0[1], gb01, acc[i][0][1]);
                    acc[i][0][2] = fmaf(d0[2], gb00, acc[i][0][2]);
                    acc[i][0][3] = fmaf(d0[3], gb01, acc[i][0][3]);

                    if (on1) {
                        float d1[4] = {};
                        mma_nvfp4_m16n8k64(d1, a, bq[2], bq[3], sfa, sfb1);
                        acc[i][1][0] = fmaf(d1[0], gb10, acc[i][1][0]);
                        acc[i][1][1] = fmaf(d1[1], gb11, acc[i][1][1]);
                        acc[i][1][2] = fmaf(d1[2], gb10, acc[i][1][2]);
                        acc[i][1][3] = fmaf(d1[3], gb11, acc[i][1][3]);
                    }
                }
            }
            if (st + 1 < NS) __syncthreads();
        }

        if constexpr (GU) {
            float h_val[2][2][NI] = {};
            float am_sub0[2][2] = {}, am_sub1[2][2] = {};
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            const float gt = acc[i][n][c] * w_outer;
                            const float up = acc[i][n][2 + c] * w_outer;
                            h_val[n][c][i] = situ_f32(gt, up);
                        }
                        float a0 = fabsf(h_val[n][c][0]);
                        if constexpr (NI >= 2) a0 = fmaxf(a0, fabsf(h_val[n][c][1]));
                        float a1 = 0.0f;
                        if constexpr (NI == 4) a1 = fmaxf(fabsf(h_val[n][c][2]), fabsf(h_val[n][c][3]));
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) {
                            a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, o));
                            if constexpr (NI == 4) a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, o));
                        }
                        am_sub0[n][c] = a0;
                        am_sub1[n][c] = a1;
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (g == 0 && r < TR) s_am[wf][r] = (NI == 4) ? fmaxf(a0, a1) : a0;
                    }
                }
            }
            __syncthreads();
            if (on0) {
                constexpr size_t H_ROW_B = (size_t)(kInter / 64) * kAB;
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        const bool valid = (r < nrows);
                        float am_64 = 0.0f;
                        if (valid) {
#pragma unroll
                            for (int k = 0; k < N_WF; ++k) am_64 = fmaxf(am_64, s_am[k][r]);
                        }
                        const float g_blk = am_64 * (1.0f / (6.0f * 448.0f));
                        float s0 = 0.0f, s1 = 0.0f;
                        float sub0_am = am_sub0[n][c];
                        if constexpr (NI == 1) {
                            sub0_am = valid ? fmaxf(s_am[wf & ~1][r], s_am[wf | 1][r]) : 0.0f;
                        }
                        const uint8_t d0 = encode_ue4m3(g_blk > 0.0f ? (sub0_am / (g_blk * 6.0f)) : 0.0f, s0);
                        const uint8_t d1 = (NI == 4) ? encode_ue4m3(g_blk > 0.0f ? (am_sub1[n][c] / (g_blk * 6.0f)) : 0.0f, s1) : 0u;
                        const float inv0 = (g_blk * s0) > 0.0f ? (1.0f / (g_blk * s0)) : 0.0f;
                        const float inv1 = (NI == 4 && (g_blk * s1) > 0.0f) ? (1.0f / (g_blk * s1)) : 0.0f;

                        // Unconditional __shfl_xor_sync across all 32 lanes (deadlock-free when nrows % 8 != 0)!
                        float qa[NI], qb[NI];
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            qa[i] = h_val[n][c][i] * ((i < 2) ? inv0 : inv1);
                            qb[i] = __shfl_xor_sync(0xFFFFFFFFu, qa[i], 4);
                        }
                        if (valid) {
                            uint8_t* dst_slot = out_ha + (size_t)(row0 + r) * H_ROW_B + (size_t)fb * kAB;
                            if ((g & 1) == 0) {
                                const int bo = (4 * NI) * wf + (g >> 1);
#pragma unroll
                                for (int i = 0; i < NI; ++i) dst_slot[bo + 4 * i] = pack_e2m1x2(qa[i], qb[i]);
                            }
                            if (g == 0) {
                                if constexpr (NI == 4) {
                                    dst_slot[32 + 2 * wf + 0] = d0;
                                    dst_slot[32 + 2 * wf + 1] = d1;
                                } else if constexpr (NI == 2) {
                                    dst_slot[32 + wf] = d0;
                                } else {
                                    if ((wf & 1) == 0) dst_slot[32 + (wf >> 1)] = d0;
                                }
                                if (wf == 0) *(float*)(dst_slot + 36) = g_blk;
                            }
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = w_outer * s_rw[r];
                        float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                            dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act; (void)out_ha; (void)topk_w; (void)y_pair;
#endif
}

// ---- Persistent Blackwell sm_120a mxf8f6f4 MoE kernel (TR in {16, 32, 64}; L20..23 MX8 experts, 32-element UE8M0 blocks)
template <bool GU, int TR>
__global__ void __launch_bounds__(kNT) k_moe_mx8_tc(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ xq,
    const uint8_t* __restrict__ xsc,
    uint8_t* __restrict__ hq,
    uint8_t* __restrict__ hsc,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair) {
#if defined(INK_SM120_DEV)
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 32;
    constexpr int NFB      = ROWS_N / kWR;
    constexpr int N_WN     = TR / 16;
    constexpr int N_WF     = 8 / N_WN;
    constexpr int W_PER_WF = kWR / N_WF;
    constexpr int NI       = W_PER_WF / 16;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kAStride4];
    __shared__ uint8_t  s_wsc[2][kWR];
    __shared__ uint8_t  s_asc[2][TR];
    __shared__ int32_t  s_row[TR];
    __shared__ float    s_am[N_WF][TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;
    const int wn   = warp % N_WN;
    const int nb0  = 16 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1;
    const int uj = tid & 1;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_MX8) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        const uint8_t* base_p = GU ? body : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * ROWS_K;

        const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                                : (fb * kWR + ur);
        const uint8_t* wrow_p = base_p + (size_t)phys_row * ROWS_K;
        const uint8_t* wrow_s = base_s + (size_t)phys_row * NS;

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int st, int buf) {
            cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
            if (uj == 0) s_wsc[buf][ur] = wrow_s[st];
            if (tid < 2 * TR) {
                const int r = tid >> 1;
                const int h = tid & 1;
                if (r < nrows) {
                    cp16_async(s_aqs[buf] + r * kAStride4 + 16 * h,
                               xq + (size_t)s_row[r] * ROWS_K + (size_t)st * 32 + 16 * h);
                }
            } else if (tid < 3 * TR) {
                const int r = tid - 2 * TR;
                if (r < nrows) s_asc[buf][r] = xsc[(size_t)s_row[r] * NS + st];
            }
            cp_commit();
        };

        issue_stage(0, 0);

        float acc[NI][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int st = 0; st < NS; ++st) {
            const int buf = st & 1;
            const int nxt = buf ^ 1;
            if (st + 1 < NS) {
                issue_stage(st + 1, nxt);
                cp_wait1();
            } else {
                cp_wait0();
            }
            __syncthreads();

            if (on0) {
                uint32_t bq[4];
                ldsm4(bq, s_aqs[buf] + b_off);
                const uint32_t sfb0 = (uint32_t)s_asc[buf][nb0 + g];
                const uint32_t sfb1 = on1 ? (uint32_t)s_asc[buf][nb0 + 8 + g] : 0u;
#pragma unroll
                for (int i = 0; i < NI; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                    const uint32_t sfa = (uint32_t)s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a];
                    mma_mxfp8_m16n8k32(acc[i][0], a, bq[0], bq[1], sfa, sfb0);
                    if (on1) mma_mxfp8_m16n8k32(acc[i][1], a, bq[2], bq[3], sfa, sfb1);
                }
            }
            if (st + 1 < NS) __syncthreads();
        }

        if constexpr (GU) {
            // Fused SITU(4.0, 25.0) + 32-element MX8 quantization (2 blocks of 32 feats per CTA!)
            float hv[2][2][NI] = {};
            float am_loc[2][2] = {};
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        float am = 0.0f;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            hv[n][c][i] = situ_f32(acc[i][n][c], acc[i][n][2 + c]);
                            am = fmaxf(am, fabsf(hv[n][c][i]));
                        }
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, o));
                        am_loc[n][c] = am;
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (g == 0 && r < TR) s_am[wf][r] = am;
                    }
                }
            }
            if constexpr (NI < 4) __syncthreads();
            if (on0) {
                constexpr int WF_PER_BLK32 = N_WF / 2;  // 1 (TR=64), 2 (TR=32), 4 (TR=16)
                const int half_id = wf / WF_PER_BLK32;  // 0 or 1
                const int sub_in  = wf % WF_PER_BLK32;  // 0..WF_PER_BLK32-1
                const int blk32   = 2 * fb + half_id;
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (r < nrows) {
                            float am32 = am_loc[n][c];
                            if constexpr (NI < 4) {
                                am32 = 0.0f;
#pragma unroll
                                for (int k = 0; k < WF_PER_BLK32; ++k) am32 = fmaxf(am32, s_am[half_id * WF_PER_BLK32 + k][r]);
                            }
                            float inv_sc = 0.0f;
                            const uint8_t sc_b = encode_ue8m0(am32, inv_sc);
                            uint8_t* dst_q = hq + (size_t)(row0 + r) * kInter + blk32 * 32 + sub_in * (8 * NI);
#pragma unroll
                            for (int i = 0; i < NI; ++i) dst_q[8 * i + g] = encode_e4m3_signed(hv[n][c][i] * inv_sc);
                            if (g == 0 && sub_in == 0) hsc[(size_t)(row0 + r) * (kInter / 32) + blk32] = sc_b;
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = s_rw[r];
                        float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                            dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)xq; (void)xsc; (void)hq; (void)hsc; (void)topk_w; (void)y_pair;
#endif
}

// ---- Standalone single-matrix test kernels for FP64 parity & TFLOPS microbenchmarks ----
__global__ void __launch_bounds__(kNT) k_test_sf4_gemm(
    const uint8_t* __restrict__ w_u8, const uint8_t* __restrict__ w_sc, float w_outer,
    const uint8_t* __restrict__ xa, float* __restrict__ y, int M, int N, int K) {
#if defined(INK_SM120_DEV)
    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][kTR * kAStride4];
    __shared__ uint32_t s_wsc[2][kWR];
    __shared__ uint32_t s_asc[2][kTR];
    __shared__ float    s_agb[2][kTR];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wf = warp >> 2, wn = warp & 3, nb0 = 16 * wn;
    const int g = lane >> 2, tig = lane & 3, lm = lane >> 3, l8 = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1, uj = tid & 1;

    const int rbase = blockIdx.x * kWR;
    const int row0 = blockIdx.y * kTR;
    const int nrows = min(kTR, M - row0);
    if (nrows <= 0 || rbase >= N) return;
    const int NS = K / 64;
    const size_t W_P_ROW = (size_t)(K / 2);
    const size_t W_S_ROW = (size_t)(K / 16);
    const size_t ACT_ROW_B = (size_t)NS * kAB;
    const int wrow_i = min(rbase + ur, N - 1);
    const uint8_t* wrow_p = w_u8 + (size_t)wrow_i * W_P_ROW;
    const uint8_t* wrow_s = w_sc + (size_t)wrow_i * W_S_ROW;

    if (tid < kTR && tid >= nrows) {
        s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
        s_agb[0][tid] = 0.0f; s_agb[1][tid] = 0.0f;
        for (int b = 0; b < 2; ++b) {
            uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
            z[0] = make_uint4(0u, 0u, 0u, 0u);
            z[1] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    __syncthreads();

    auto issue = [&](int st, int buf) {
        cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
        if (uj == 0) s_wsc[buf][ur] = *(const uint32_t*)(wrow_s + (size_t)st * 4);
        if (tid < 2 * kTR) {
            const int r = tid >> 1, h = tid & 1;
            if (r < nrows) cp16_async(s_aqs[buf] + r * kAStride4 + 16 * h, xa + (size_t)(row0 + r) * ACT_ROW_B + (size_t)st * kAB + 16 * h);
        } else if (tid < 3 * kTR) {
            const int r = tid - 2 * kTR;
            if (r < nrows) {
                const uint2 meta = *(const uint2*)(xa + (size_t)(row0 + r) * ACT_ROW_B + (size_t)st * kAB + 32);
                s_asc[buf][r] = meta.x;
                s_agb[buf][r] = __uint_as_float(meta.y);
            }
        }
        cp_commit();
    };

    issue(0, 0);
    float acc[4][2][4] = {};
    const bool on0 = nb0 < nrows;
    const bool on1 = nb0 + 8 < nrows;

    for (int st = 0; st < NS; ++st) {
        const int buf = st & 1, nxt = buf ^ 1;
        if (st + 1 < NS) { issue(st + 1, nxt); cp_wait1(); } else cp_wait0();
        __syncthreads();
        if (on0) {
            uint32_t bq[4];
            ldsm4(bq, s_aqs[buf] + b_off);
            const uint32_t sfb0 = s_asc[buf][nb0 + g];
            const uint32_t sfb1 = on1 ? s_asc[buf][nb0 + 8 + g] : 0u;
            const float gb00 = s_agb[buf][nb0 + 2 * tig + 0];
            const float gb01 = s_agb[buf][nb0 + 2 * tig + 1];
            const float gb10 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 0] : 0.0f;
            const float gb11 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 1] : 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t a[4];
                ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                const uint32_t sfa = s_wsc[buf][64 * wf + 16 * i + tidx_a];
                float d0[4] = {};
                mma_nvfp4_m16n8k64(d0, a, bq[0], bq[1], sfa, sfb0);
                acc[i][0][0] = fmaf(d0[0], gb00, acc[i][0][0]);
                acc[i][0][1] = fmaf(d0[1], gb01, acc[i][0][1]);
                acc[i][0][2] = fmaf(d0[2], gb00, acc[i][0][2]);
                acc[i][0][3] = fmaf(d0[3], gb01, acc[i][0][3]);
                if (on1) {
                    float d1[4] = {};
                    mma_nvfp4_m16n8k64(d1, a, bq[2], bq[3], sfa, sfb1);
                    acc[i][1][0] = fmaf(d1[0], gb10, acc[i][1][0]);
                    acc[i][1][1] = fmaf(d1[1], gb11, acc[i][1][1]);
                    acc[i][1][2] = fmaf(d1[2], gb10, acc[i][1][2]);
                    acc[i][1][3] = fmaf(d1[3], gb11, acc[i][1][3]);
                }
            }
        }
        if (st + 1 < NS) __syncthreads();
    }

    if (on0) {
        const int rb = rbase + 64 * wf;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            if (n == 1 && !on1) break;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int r = nb0 + 8 * n + 2 * tig + c;
                if (r < nrows) {
                    float* dst = y + (size_t)(row0 + r) * N;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int c0 = rb + 16 * i + g;
                        if (c0 < N) dst[c0] = acc[i][n][c] * w_outer;
                        if (c0 + 8 < N) dst[c0 + 8] = acc[i][n][2 + c] * w_outer;
                    }
                }
            }
        }
    }
#else
    (void)w_u8; (void)w_sc; (void)w_outer; (void)xa; (void)y; (void)M; (void)N; (void)K;
#endif
}

__global__ void __launch_bounds__(kNT) k_test_mx8_gemm(
    const uint8_t* __restrict__ w_u8, const uint8_t* __restrict__ w_sc,
    const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xsc,
    float* __restrict__ y, int M, int N, int K) {
#if defined(INK_SM120_DEV)
    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][kTR * kAStride4];
    __shared__ uint8_t s_wsc[2][kWR];
    __shared__ uint8_t s_asc[2][kTR];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wf = warp >> 2, wn = warp & 3, nb0 = 16 * wn;
    const int g = lane >> 2, tig = lane & 3, lm = lane >> 3, l8 = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1, uj = tid & 1;

    const int rbase = blockIdx.x * kWR;
    const int row0 = blockIdx.y * kTR;
    const int nrows = min(kTR, M - row0);
    if (nrows <= 0 || rbase >= N) return;
    const int NS = K / 32;
    const int wrow_i = min(rbase + ur, N - 1);
    const uint8_t* wrow_p = w_u8 + (size_t)wrow_i * K;
    const uint8_t* wrow_s = w_sc + (size_t)wrow_i * NS;

    if (tid < kTR && tid >= nrows) {
        s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
        for (int b = 0; b < 2; ++b) {
            uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
            z[0] = make_uint4(0u, 0u, 0u, 0u);
            z[1] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    __syncthreads();

    auto issue = [&](int st, int buf) {
        cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
        if (uj == 0) s_wsc[buf][ur] = wrow_s[st];
        if (tid < 2 * kTR) {
            const int r = tid >> 1, h = tid & 1;
            if (r < nrows) cp16_async(s_aqs[buf] + r * kAStride4 + 16 * h, xq + (size_t)(row0 + r) * K + (size_t)st * 32 + 16 * h);
        } else if (tid < 3 * kTR) {
            const int r = tid - 2 * kTR;
            if (r < nrows) s_asc[buf][r] = xsc[(size_t)(row0 + r) * NS + st];
        }
        cp_commit();
    };

    issue(0, 0);
    float acc[4][2][4] = {};
    const bool on0 = nb0 < nrows;
    const bool on1 = nb0 + 8 < nrows;

    for (int st = 0; st < NS; ++st) {
        const int buf = st & 1, nxt = buf ^ 1;
        if (st + 1 < NS) { issue(st + 1, nxt); cp_wait1(); } else cp_wait0();
        __syncthreads();
        if (on0) {
            uint32_t bq[4];
            ldsm4(bq, s_aqs[buf] + b_off);
            const uint32_t sfb0 = (uint32_t)s_asc[buf][nb0 + g];
            const uint32_t sfb1 = on1 ? (uint32_t)s_asc[buf][nb0 + 8 + g] : 0u;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t a[4];
                ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                const uint32_t sfa = (uint32_t)s_wsc[buf][64 * wf + 16 * i + tidx_a];
                mma_mxfp8_m16n8k32(acc[i][0], a, bq[0], bq[1], sfa, sfb0);
                if (on1) mma_mxfp8_m16n8k32(acc[i][1], a, bq[2], bq[3], sfa, sfb1);
            }
        }
        if (st + 1 < NS) __syncthreads();
    }

    if (on0) {
        const int rb = rbase + 64 * wf;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            if (n == 1 && !on1) break;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int r = nb0 + 8 * n + 2 * tig + c;
                if (r < nrows) {
                    float* dst = y + (size_t)(row0 + r) * N;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int c0 = rb + 16 * i + g;
                        if (c0 < N) dst[c0] = acc[i][n][c];
                        if (c0 + 8 < N) dst[c0 + 8] = acc[i][n][2 + c];
                    }
                }
            }
        }
    }
#else
    (void)w_u8; (void)w_sc; (void)xq; (void)xsc; (void)y; (void)M; (void)N; (void)K;
#endif
}

}  // namespace

bool available() {
#if !defined(CUDART_VERSION) || (CUDART_VERSION < 12080)
    return false;
#else
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        cached = 0;
        return false;
    }
    if (p.major != 12 || (p.minor != 0 && p.minor != 1)) {
        cached = 0;
        return false;
    }
    cudaFuncAttributes attr{};
    cached = (cudaFuncGetAttributes(&attr, (const void*)k_moe_sf4_tc<true, 64>) == cudaSuccess &&
              attr.binaryVersion >= 120) ? 1 : 0;
    return cached != 0;
#endif
}

void quantize_act_fp4(const float* x, int rows, int cols, uint8_t* xa, cudaStream_t stream) {
    const int64_t nblk = (int64_t)rows * (cols / 64);
    const unsigned blocks = (unsigned)std::max<int64_t>(1, (nblk + 7) / 8);
    k_quant_act_fp4<<<blocks, 256, 0, stream>>>(x, nblk, xa);
}

void quantize_act_mx8(const float* x, int rows, int cols, uint8_t* xq, uint8_t* xsc, cudaStream_t stream) {
    const int64_t nblk = (int64_t)rows * (cols / 32);
    const unsigned blocks = (unsigned)std::max<int64_t>(1, (nblk + 7) / 8);
    k_quant_act_mx8<<<blocks, 256, 0, stream>>>(x, nblk, xq, xsc);
}

void moe_sf4_gu_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* xa, uint8_t* ha, int max_tiles, int tile_rows, cudaStream_t stream) {
    const unsigned grid = (unsigned)std::min(std::max(1, max_tiles * (2 * kInter / kWR)), 2048);
    if (tile_rows <= 16)      k_moe_sf4_tc<true, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr);
    else if (tile_rows <= 32) k_moe_sf4_tc<true, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr);
    else                      k_moe_sf4_tc<true, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr);
}

void moe_sf4_dn_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* ha, const float* topk_w, float* y_pair, int max_tiles, int tile_rows, cudaStream_t stream) {
    const unsigned grid = (unsigned)std::min(std::max(1, max_tiles * (kHidden / kWR)), 2048);
    if (tile_rows <= 16)      k_moe_sf4_tc<false, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair);
    else if (tile_rows <= 32) k_moe_sf4_tc<false, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair);
    else                      k_moe_sf4_tc<false, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair);
}

void moe_mx8_gu_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* xq, const uint8_t* xsc, uint8_t* hq, uint8_t* hsc,
                   int max_tiles, int tile_rows, cudaStream_t stream) {
    const unsigned grid = (unsigned)std::min(std::max(1, max_tiles * (2 * kInter / kWR)), 2048);
    if (tile_rows <= 16)      k_moe_mx8_tc<true, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xq, xsc, hq, hsc, nullptr, nullptr);
    else if (tile_rows <= 32) k_moe_mx8_tc<true, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xq, xsc, hq, hsc, nullptr, nullptr);
    else                      k_moe_mx8_tc<true, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xq, xsc, hq, hsc, nullptr, nullptr);
}

void moe_mx8_dn_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* hq, const uint8_t* hsc, const float* topk_w, float* y_pair,
                   int max_tiles, int tile_rows, cudaStream_t stream) {
    const unsigned grid = (unsigned)std::min(std::max(1, max_tiles * (kHidden / kWR)), 2048);
    if (tile_rows <= 16)      k_moe_mx8_tc<false, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, y_pair);
    else if (tile_rows <= 32) k_moe_mx8_tc<false, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, y_pair);
    else                      k_moe_mx8_tc<false, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, y_pair);
}

void test_sf4_gemm_tc(const uint8_t* w_u8, const uint8_t* w_sc, float w_outer,
                      const float* x_f32, float* y_f32, int M, int N, int K, cudaStream_t stream) {
    uint8_t* d_xa = dalloc<uint8_t>((size_t)M * (K / 64) * kAB);
    quantize_act_fp4(x_f32, M, K, d_xa, stream);
    dim3 grid((N + kWR - 1) / kWR, (M + kTR - 1) / kTR);
    k_test_sf4_gemm<<<grid, kNT, 0, stream>>>(w_u8, w_sc, w_outer, d_xa, y_f32, M, N, K);
    INK_CK(cudaStreamSynchronize(stream));
    INK_CK(cudaFree(d_xa));
}

void test_mx8_gemm_tc(const uint8_t* w_u8, const uint8_t* w_sc,
                      const float* x_f32, float* y_f32, int M, int N, int K, cudaStream_t stream) {
    uint8_t* d_xq = dalloc<uint8_t>((size_t)M * K);
    uint8_t* d_xsc = dalloc<uint8_t>((size_t)M * (K / 32));
    quantize_act_mx8(x_f32, M, K, d_xq, d_xsc, stream);
    dim3 grid((N + kWR - 1) / kWR, (M + kTR - 1) / kTR);
    k_test_mx8_gemm<<<grid, kNT, 0, stream>>>(w_u8, w_sc, d_xq, d_xsc, y_f32, M, N, K);
    INK_CK(cudaStreamSynchronize(stream));
    INK_CK(cudaFree(d_xq));
    INK_CK(cudaFree(d_xsc));
}

}  // namespace strata::ink::sm120
