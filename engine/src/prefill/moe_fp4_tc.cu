// src/prefill/moe_fp4_tc.cu - clean-room MIT Blackwell (sm_120a / sm_121a) mxf4nvf4 Tensor Core prompt experts.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Derived from NVIDIA's public PTX ISA 8.7+ specification for block-scaled FP4 matrix multiply-accumulate:
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
// and Strata's own persistent tile scheduler / interleaved gate-up epilogue layout (src/prefill/moe_fused_iq.cu).
//
// Activation 80-byte slot layout (one slot per 64 values, fitting in place inside m.Xq and m.H):
//   bytes [ 0..31]: uint8_t qs[32]  (split-half E2M1 nibbles: qs[8*sub + j] packs elem 16*sub + j in bits 3..0 and
//                                    elem 16*sub + 8 + j in bits 7..4, matching block_nvfp4's qs[32] byte for byte)
//   bytes [32..35]: uint8_t d[4]    (packed UE4M3 sub-block scales for sub = 0..3)
//   bytes [36..39]: float   g_blk   (per-64-element FP32 scale amax_64 / (6 * 448), preserving full UE4M3 dynamic range)
//   bytes [40..79]: unused padding (keeps fused::act_bytes stride AB = 80 unchanged)
#include "strata/prefill/moe_fp4_tc.hpp"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "strata/core/emulate.hpp"
#include "strata/prefill/nvfp4_prefill_mode.hpp"
#include "strata/quant/types.hpp"

namespace strata::prefill::fp4_tc {
namespace {

using strata::prefill::fused::Batch;
using strata::prefill::fused::NativeGeom;
using strata::prefill::fused::kMoeEmbd;
using strata::prefill::fused::kMoeFF;
using strata::prefill::fused::kTileRows;

constexpr int T_NVFP4 = (int) strata::quant::Type::NVFP4;
constexpr int AB = 80;                         // bytes per 64 activation values in m.Xq / m.H
constexpr int K_STEP = 64;                     // values per block_nvfp4 and per m16n8k64 MMA
constexpr int BS_NVFP4 = 36;                   // sizeof(block_nvfp4): 4B UE4M3 scales + 32B E2M1 nibbles
constexpr int NT = 256;                        // 8 warps: wf in 0..1 (weight row half), wn in 0..3 (16 routed rows each)
constexpr int TR = 64;                         // routed rows per tile (kTileRows)
constexpr int WR = 128;                        // weight rows per CTA (GU: 64 gate + 64 up = 64 SwiGLU cols; Down: 128 cols)
constexpr int W_STRIDE = 48;                   // 32B qs + 16B pad (12 banks: conflict-free ldmatrix.m8n8.x4)
constexpr int A_STRIDE = 48;                   // 32B qs + 16B pad

constexpr int GU_ROWS_N = 2 * kMoeFF;
constexpr int GU_ROWS_K = kMoeEmbd;
constexpr int D_ROWS_N = kMoeEmbd;
constexpr int D_ROWS_K = kMoeFF;
constexpr size_t X_ROW_BYTES = (size_t) (GU_ROWS_K / K_STEP) * AB;
constexpr size_t H_ROW_BYTES = (size_t) (D_ROWS_K / K_STEP) * AB;

static_assert(TR == kTileRows, "tile rows match fused::group");
constexpr bool kShapeOk = (GU_ROWS_K % K_STEP == 0) && (D_ROWS_K % K_STEP == 0) &&
                          (GU_ROWS_N % WR == 0) && (D_ROWS_N % WR == 0);

struct Tables {
    int32_t *cnt, *off, *fill, *ts;
    int2* tiles;
};

size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }

Tables tables_of(const void* scratch, int n_expert) {
    int32_t* p = (int32_t*) const_cast<void*>(scratch);
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1,
            (int2*) ((uint8_t*) const_cast<void*>(scratch) + tiles_at(n_expert))};
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

unsigned blocks_for(int64_t items, int per_block) {
    const int64_t b = (items + per_block - 1) / per_block;
    return (unsigned) std::max<int64_t>(1, std::min<int64_t>(b, 65535));
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && (__CUDA_ARCH__ < 1300)
#define STRATA_FP4_TC_DEV 1
#endif

// Unaligned-safe 32-bit load from 2-byte-aligned GGUF / Strata expert blobs.
__device__ __forceinline__ uint32_t ld32_blob(const uint8_t* p) {
    if (((uintptr_t) p & 3u) == 0u) return *(const uint32_t*) p;
    const uint16_t lo = *(const uint16_t*) p;
    const uint16_t hi = *(const uint16_t*) (p + 2);
    return (uint32_t) lo | ((uint32_t) hi << 16);
}

#if defined(STRATA_FP4_TC_DEV)
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp16_async(void* dst_smem, const void* src_glob) {
    const uint32_t d = smem_u32(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src_glob));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;\n"); }

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

__device__ __forceinline__ uint8_t encode_ue4m3(float x, float& decoded_scale) {
    if (!(x > 0.0f)) {
        decoded_scale = 0.0f;
        return 0;
    }
    const float clamped = fminf(x, 448.0f);
    const __nv_fp8_e4m3 fp8(clamped);
    decoded_scale = (float) fp8;
    return fp8.__x;
}

__device__ __forceinline__ uint8_t pack_e2m1x2(float v_lo, float v_hi) {
    const __nv_fp4x2_e2m1 q2(make_float2(v_lo, v_hi));
    return q2.__x;
}
#endif

// ---- Activation quantizer: 1 warp (32 threads) per 64 FP32 values -> 80-byte AB slot.
__global__ void quant_act_fp4_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
#if defined(STRATA_FP4_TC_DEV)
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= nblk) return;
    const int lane = threadIdx.x & 31;
    const int sub = lane >> 3;   // 0..3 (four 16-element sub-blocks)
    const int j = lane & 7;      // 0..7 (low-nibble index j and high-nibble index j + 8)

    const float* xb = x + w * K_STEP;
    const float v0 = xb[16 * sub + j];
    const float v1 = xb[16 * sub + 8 + j];

    float am_sub = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) am_sub = fmaxf(am_sub, __shfl_xor_sync(0xffffffffu, am_sub, o));

    float am_64 = fmaxf(am_sub, __shfl_xor_sync(0xffffffffu, am_sub, 8));
    am_64 = fmaxf(am_64, __shfl_xor_sync(0xffffffffu, am_64, 16));

    const float g_blk = am_64 * (1.0f / (6.0f * 448.0f));
    float s_sub = 0.0f;
    const uint8_t d_code = encode_ue4m3(g_blk > 0.0f ? (am_sub / (g_blk * 6.0f)) : 0.0f, s_sub);
    const float denom = g_blk * s_sub;
    const float inv = denom > 0.0f ? (1.0f / denom) : 0.0f;

    uint8_t* out = xa + w * AB;
    out[lane] = pack_e2m1x2(v0 * inv, v1 * inv);
    if (j == 0) out[32 + sub] = d_code;
    if (lane == 0) *(float*) (out + 36) = g_blk;
#else
    (void) x; (void) nblk; (void) xa;
#endif
}

// ---- Persistent Blackwell mxf4nvf4 MoE kernel (GU = true: gate+up + SwiGLU + FP4 H requant; GU = false: down).
template <bool GU>
__global__ void __launch_bounds__(NT) fp4_tc_kernel(Batch b, NativeGeom geo, Tables tb, const uint8_t* __restrict__ act,
                                                    const int32_t* __restrict__ src, uint8_t* __restrict__ out,
                                                    float* __restrict__ dm) {
#if defined(STRATA_FP4_TC_DEV)
    if constexpr (!kShapeOk) return;
    constexpr int ROWS_N = GU ? GU_ROWS_N : D_ROWS_N;
    constexpr int ROWS_K = GU ? GU_ROWS_K : D_ROWS_K;
    constexpr int NS = ROWS_K / K_STEP;
    constexpr int NFB = ROWS_N / WR;
    constexpr size_t ACT_ROW_BYTES = GU ? X_ROW_BYTES : H_ROW_BYTES;

    __shared__ alignas(16) uint8_t s_wqs[2][WR * W_STRIDE];     // 2 * 6,144 = 12,288 B
    __shared__ alignas(16) uint8_t s_aqs[2][TR * A_STRIDE];     // 2 * 3,072 =  6,144 B
    __shared__ uint32_t s_wsc[2][WR];                           // 2 *   512 =  1,024 B
    __shared__ uint32_t s_asc[2][TR];                           // 2 *   256 =    512 B
    __shared__ float s_agb[2][TR];                              // 2 *   256 =    512 B
    __shared__ int32_t s_row[TR];                               //                256 B
    __shared__ float s_am32[2][TR];                             //                512 B (GU epilogue 64-elem max exchange)

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf = warp >> 2;          // 0..1: 64 weight rows per wf (GU: 32 gate+up features)
    const int wn = warp & 3;           // 0..3: 16 routed rows per wn (nb0 = 16 * wn)
    const int nb0 = 16 * wn;
    const int g = lane >> 2;           // 0..7
    const int tig = lane & 3;          // 0..3
    const int lm = lane >> 3;          // 0..3
    const int l8 = lane & 7;           // 0..7
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * W_STRIDE + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * A_STRIDE + 16 * (lm & 1);

    // 256 threads load 128 weight rows per stage: ur = tid >> 1 (0..127), uj = tid & 1 (0: d + qs[0..15], 1: qs[16..31]).
    const int ur = tid >> 1;
    const int uj = tid & 1;

    const int t0 = tb.ts[b.e0];
    const int nwork = (tb.ts[b.e1] - t0) * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * WR;

        // Per-expert FP32 scales {s_gate, s_up, s_down, 0} in the blob's 16-byte tail.
        const uint8_t* tail = blob + geo.down_off + geo.d_row * (size_t) D_ROWS_N;
        float s_gate = 1.0f, s_up = 1.0f, s_down = 1.0f;
        if constexpr (GU) {
            s_gate = __uint_as_float(ld32_blob(tail));
            s_up = __uint_as_float(ld32_blob(tail + 4));
        } else {
            s_down = __uint_as_float(ld32_blob(tail + 8));
        }

        // Interleave 8 gate rows then 8 up rows per 16-row tile so each lane holds gate & up of the same feature.
        const uint8_t* wrow = GU ? blob + ((ur & 8) ? geo.up_off : 0) +
                                       (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;

        __syncthreads();
        if (tid < TR) {
            s_row[tid] = tid < nrows ? (GU ? src[row0 + tid] : (row0 + tid)) : -1;
            if (tid >= nrows) {
                s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
                s_agb[0][tid] = 0.0f; s_agb[1][tid] = 0.0f;
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    uint4* z = (uint4*) (s_aqs[b] + tid * A_STRIDE);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto fetch_w_regs = [&](int st, uint32_t (&wq)[4], uint32_t& wd) {
            const uint8_t* bp = wrow + (size_t) st * BS_NVFP4;
            if (uj == 0) wd = ld32_blob(bp);
            const uint8_t* qp = bp + 4 + 16 * uj;
#pragma unroll
            for (int k = 0; k < 4; ++k) wq[k] = ld32_blob(qp + 4 * k);
        };

        auto store_w_smem = [&](int buf, const uint32_t (&wq)[4], uint32_t wd) {
            *(uint4*) (s_wqs[buf] + ur * W_STRIDE + 16 * uj) = make_uint4(wq[0], wq[1], wq[2], wq[3]);
            if (uj == 0) s_wsc[buf][ur] = wd;
        };

        auto issue_act = [&](int st, int buf) {
            // 128 threads copy 64 rows * 32B qs via 16B cp.async; 64 threads load d (4B) and g_blk (4B).
            if (tid < 2 * TR) {
                const int r = tid >> 1;
                const int h = tid & 1;
                if (r < nrows) {
                    const uint8_t* ap = act + (size_t) s_row[r] * ACT_ROW_BYTES + (size_t) st * AB;
                    cp16_async(s_aqs[buf] + r * A_STRIDE + 16 * h, ap + 16 * h);
                }
            } else if (tid < 3 * TR) {
                const int r = tid - 2 * TR;
                if (r < nrows) {
                    const uint8_t* ap = act + (size_t) s_row[r] * ACT_ROW_BYTES + (size_t) st * AB;
                    const uint2 meta = *(const uint2*) (ap + 32);
                    s_asc[buf][r] = meta.x;
                    s_agb[buf][r] = __uint_as_float(meta.y);
                }
            }
            cp_commit();
        };

        // Stage 0 prologue.
        uint32_t wq_reg[4], wd_reg = 0;
        fetch_w_regs(0, wq_reg, wd_reg);
        store_w_smem(0, wq_reg, wd_reg);
        issue_act(0, 0);

        float acc[4][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int st = 0; st < NS; ++st) {
            const int buf = st & 1;
            const int nxt = buf ^ 1;
            // Prefetch stage st + 1 into registers / async copy while stage st waits.
            if (st + 1 < NS) {
                fetch_w_regs(st + 1, wq_reg, wd_reg);
                issue_act(st + 1, nxt);
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
                for (int i = 0; i < 4; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * W_STRIDE);
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

            if (st + 1 < NS) {
                __syncthreads();
                store_w_smem(nxt, wq_reg, wd_reg);
            }
        }

        if constexpr (GU) {
            // Fused SwiGLU + 64-feature NVFP4 quantization directly into `out` (m.H).
            // Warp (wf = 0, wn) holds features 0..31 and warp (wf = 1, wn) holds features 32..63 of block `fb`
            // for routed rows [16*wn .. 16*wn + 15].
            float h_val[2][2][4] = {};
            float am_sub0[2][2] = {}, am_sub1[2][2] = {};
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float gt = acc[i][n][c] * s_gate;
                            const float up = acc[i][n][2 + c] * s_up;
                            h_val[n][c][i] = (gt / (1.0f + __expf(-gt))) * up;
                        }
                        float a0 = fmaxf(fabsf(h_val[n][c][0]), fabsf(h_val[n][c][1]));
                        float a1 = fmaxf(fabsf(h_val[n][c][2]), fabsf(h_val[n][c][3]));
                        // Reduce across g = 0..7 (lanes with identical tig = lane & 3).
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) {
                            a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
                            a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
                        }
                        am_sub0[n][c] = a0;
                        am_sub1[n][c] = a1;
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (g == 0 && r < TR) s_am32[wf][r] = fmaxf(a0, a1);
                    }
                }
            }
            __syncthreads();
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (r < nrows) {
                            const float am_64 = fmaxf(s_am32[0][r], s_am32[1][r]);
                            const float g_blk = am_64 * (1.0f / (6.0f * 448.0f));
                            float s0 = 0.0f, s1 = 0.0f;
                            const uint8_t d0 = encode_ue4m3(g_blk > 0.0f ? (am_sub0[n][c] / (g_blk * 6.0f)) : 0.0f, s0);
                            const uint8_t d1 = encode_ue4m3(g_blk > 0.0f ? (am_sub1[n][c] / (g_blk * 6.0f)) : 0.0f, s1);
                            const float inv0 = (g_blk * s0) > 0.0f ? (1.0f / (g_blk * s0)) : 0.0f;
                            const float inv1 = (g_blk * s1) > 0.0f ? (1.0f / (g_blk * s1)) : 0.0f;

                            uint8_t* dst_slot = out + (size_t) (row0 + r) * H_ROW_BYTES + (size_t) fb * AB;
                            dst_slot[16 * wf + 0 + g] = pack_e2m1x2(h_val[n][c][0] * inv0, h_val[n][c][1] * inv0);
                            dst_slot[16 * wf + 8 + g] = pack_e2m1x2(h_val[n][c][2] * inv1, h_val[n][c][3] * inv1);
                            if (g == 0) {
                                dst_slot[32 + 2 * wf + 0] = d0;
                                dst_slot[32 + 2 * wf + 1] = d1;
                                if (wf == 0) *(float*) (dst_slot + 36) = g_blk;
                            }
                        }
                    }
                }
            }
        } else if (on0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows) {
                            dm[(size_t) (row0 + r) * kMoeEmbd + rbase + 64 * wf + 16 * i + 8 * (q >> 1) + g] =
                                acc[i][n][q] * s_down;
                        }
                    }
                }
            }
        }
    }
#else
    (void) b; (void) geo; (void) tb; (void) act; (void) src; (void) out; (void) dm;
#endif
}

struct DevState {
    bool ok = false;
    int sms = 0;
    int occ_gu = 1;
    int occ_d = 1;
};

const DevState& dev_state() {
    static std::mutex mu;
    static DevState per[32];
    static bool done[32] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) {
        static const DevState none{};
        return none;
    }
    std::lock_guard<std::mutex> lock(mu);
    if (!done[dev]) {
        DevState st;
        int major = 0, minor = 0, sms = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess) {
            const int mj = strata::cc_major_of(major);
            const int mn = strata::cc_minor_of(minor);
            if (kShapeOk && mj == 12 && (mn == 0 || mn == 1) && sms > 0) {
                cudaFuncAttributes attr_gu{}, attr_d{};
                int o_gu = 0, o_d = 0;
                if (cudaFuncGetAttributes(&attr_gu, fp4_tc_kernel<true>) == cudaSuccess &&
                    cudaFuncGetAttributes(&attr_d, fp4_tc_kernel<false>) == cudaSuccess &&
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o_gu, fp4_tc_kernel<true>, NT, 0) == cudaSuccess &&
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o_d, fp4_tc_kernel<false>, NT, 0) == cudaSuccess &&
                    o_gu > 0 && o_d > 0) {
                    st.ok = true;
                    st.sms = sms;
                    st.occ_gu = o_gu;
                    st.occ_d = o_d;
                } else {
                    (void) cudaGetLastError();
                }
            }
        } else {
            (void) cudaGetLastError();
        }
        per[dev] = st;
        done[dev] = true;
    }
    return per[dev];
}

}  // namespace

bool built() { return true; }

bool available() { return dev_state().ok; }

bool wanted() {
    if (!available()) return false;
    static const Nvfp4PrefillPath p = resolve_nvfp4_prefill_path(std::getenv("STRATA_PREFILL_NVFP4"), true, false);
    return p == Nvfp4PrefillPath::FusedW4A4;
}

void quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    if (cols % K_STEP != 0) {
        std::fprintf(stderr, "fp4_tc::quantize_act: cols=%lld is not a multiple of 64\n", (long long) cols);
        std::exit(1);
    }
    const int64_t nblk = rows * (cols / K_STEP);
    quant_act_fp4_kernel<<<blocks_for(nblk, 8), 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xa);
    ck(cudaGetLastError(), "fp4_tc::quantize_act");
}

void experts(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
             const int32_t* src, void* ha, float* dm, void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevState& d = dev_state();
    if (!d.ok || g.gu_type != T_NVFP4 || g.d_type != T_NVFP4) {
        std::fprintf(stderr, "fp4_tc::experts: unavailable on this device or non-NVFP4 types (%d / %d)\n",
                     g.gu_type, g.d_type);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables_of(scratch, n_expert);
    const int64_t tiles = (n + TR - 1) / TR + (b.e1 - b.e0);
    const unsigned grid_gu = (unsigned) std::max<int64_t>(1, std::min<int64_t>(tiles * (GU_ROWS_N / WR), (int64_t) d.sms * d.occ_gu));
    const unsigned grid_d = (unsigned) std::max<int64_t>(1, std::min<int64_t>(tiles * (D_ROWS_N / WR), (int64_t) d.sms * d.occ_d));
    fp4_tc_kernel<true><<<grid_gu, NT, 0, s>>>(b, g, tb, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr);
    fp4_tc_kernel<false><<<grid_d, NT, 0, s>>>(b, g, tb, (const uint8_t*) ha, src, nullptr, dm);
    ck(cudaGetLastError(), "fp4_tc::experts");
}

}  // namespace strata::prefill::fp4_tc
