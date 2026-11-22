// src/prefill/moe_fp4_tc.cu - clean-room MIT Blackwell (sm_120a / sm_121a) mxf4nvf4 Tensor Core prompt experts.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Derived from NVIDIA's public PTX ISA 8.7+ specification for block-scaled FP4 matrix multiply-accumulate:
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
// and Strata's own persistent tile scheduler / interleaved gate-up epilogue layout (src/prefill/moe_fused_iq.cu).
//
// v3 High-Bandwidth Coalesced Pipeline (NT=256, TR=64 single-pass, WR=128):
//   - 100% coalesced 32B/row weight loads: 8 consecutive lanes load 8 contiguous uint32 words of each 36B NVFP4 block,
//     eliminating 32-way strided L2/DRAM over-fetch;
//   - Single-pass TR=64 (8 warps = 2 weight-row halves x 4 routed-row quarters) eliminates double weight reads at T=2048;
//   - True software-pipelined register prefetch overlaps global weight loads + cp.async activations with mxf4nvf4 MMA
//     using only a single __syncthreads() per K=64 step (~21.2 KB smem, 0 spills).
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
constexpr int AB = 80;
constexpr int K_STEP = 64;
constexpr int BS_NVFP4 = 36;
constexpr int NT = 512;          // 16 warps: wf in 0..3 (64 weight rows each -> 256 rows), wn in 0..3 (16 routed rows each -> 64 rows/pass)
constexpr int TR = 64;           // fused::group tile rows (single pass: zero double weight reads!)
constexpr int WR = 256;          // GU: 128 gate + 128 up = 128 SwiGLU features (2 NVFP4 blocks/fb); Down: 256 cols -> 2x fewer tiles & activation reads!
constexpr int W_STRIDE = 48;     // 32B qs + 16B pad (conflict-free ldmatrix)
constexpr int A_STRIDE = 48;

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

__device__ __forceinline__ uint32_t ld32_unaligned(const uint8_t* p) {
    const uint16_t lo = __ldg((const uint16_t*) p);
    const uint16_t hi = __ldg((const uint16_t*) (p + 2));
    return (uint32_t) lo | ((uint32_t) hi << 16);
}

__device__ __forceinline__ uint32_t ld32_fast(const uint8_t* p, bool aligned4) {
    return aligned4 ? __ldg((const uint32_t*) p) : ld32_unaligned(p);
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
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p)); }

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

__global__ void quant_act_fp4_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
#if defined(STRATA_FP4_TC_DEV)
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= nblk) return;
    const int lane = threadIdx.x & 31;
    const int sub = lane >> 3;
    const int j = lane & 7;

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

// Persistent Blackwell mxf4nvf4 MoE kernel (v4):
//   - 512 threads (16 warps): wf = warp >> 2 (0..3: 64 weight rows each = 256 rows), wn = warp & 3 (0..3: 16 routed rows each = 64 rows).
//   - Halves CTA tile count (5 GU tiles, 10 Down tiles) and halves global activation traffic vs WR=128 (~35 KB static smem, 0 spills).
template <bool GU>
__global__ void __launch_bounds__(NT, 1) fp4_tc_kernel(Batch b, NativeGeom geo, Tables tb, const uint8_t* __restrict__ act,
                                                       const int32_t* __restrict__ src, uint8_t* __restrict__ out,
                                                       float* __restrict__ dm) {
#if defined(STRATA_FP4_TC_DEV)
    if constexpr (!kShapeOk) return;
    constexpr int ROWS_N = GU ? GU_ROWS_N : D_ROWS_N;
    constexpr int ROWS_K = GU ? GU_ROWS_K : D_ROWS_K;
    constexpr int NS = ROWS_K / K_STEP;
    constexpr int NFB = ROWS_N / WR;
    constexpr size_t ACT_ROW_BYTES = GU ? X_ROW_BYTES : H_ROW_BYTES;

    __shared__ alignas(16) uint8_t s_wqs[2][WR * W_STRIDE]; // 2 * 12,288 = 24,576 B
    __shared__ alignas(16) uint8_t s_aqs[2][TR * A_STRIDE]; // 2 *  3,072 =  6,144 B
    __shared__ uint32_t s_wsc[2][WR];                       // 2 *  1,024 =  2,048 B
    __shared__ uint32_t s_asc[2][TR];                       // 2 *    256 =    512 B
    __shared__ float s_agb[2][TR];                          // 2 *    256 =    512 B
    __shared__ int32_t s_row[TR];                           //                 256 B
    __shared__ float s_am32[4][TR];                         //               1,024 B (total 35,072 B)

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf = warp >> 2;          // 0..3: 64 weight rows per wf (256 weight rows/CTA)
    const int wn = warp & 3;           // 0..3: 16 routed rows per wn
    const int nb0 = 16 * wn;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int lm = lane >> 3;
    const int l8 = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int rl = tid >> 3;           // 0..63: row offset within 64-row quarter
    const int lc = tid & 7;            // 0..7: 8 consecutive lanes load 32 contiguous bytes of qs!

    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * W_STRIDE + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * A_STRIDE + 16 * (lm & 1);

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
        const bool aligned4 = (((uintptr_t) blob & 3u) == 0u);
        const int rbase = fb * WR;

        const uint8_t* tail = blob + geo.down_off + geo.d_row * (size_t) D_ROWS_N;
        float s_gate = 1.0f, s_up = 1.0f, s_down = 1.0f;
        if constexpr (GU) {
            s_gate = __uint_as_float(ld32_fast(tail, aligned4));
            s_up = __uint_as_float(ld32_fast(tail + 4, aligned4));
        } else {
            s_down = __uint_as_float(ld32_fast(tail + 8, aligned4));
        }

        // Precompute the 4 row pointers owned by lane-group `rl` (rows rl, rl+64, rl+128, rl+192).
        auto row_ptr = [&](int row) -> const uint8_t* {
            if constexpr (GU) {
                return blob + ((row & 8) ? geo.up_off : 0) +
                       (size_t) (fb * (WR / 2) + 8 * (row >> 4) + (row & 7)) * geo.gu_row;
            } else {
                return blob + geo.down_off + (size_t) (rbase + row) * geo.d_row;
            }
        };
        const uint8_t* const wr0 = row_ptr(rl);
        const uint8_t* const wr1 = row_ptr(rl + 64);
        const uint8_t* const wr2 = row_ptr(rl + 128);
        const uint8_t* const wr3 = row_ptr(rl + 192);
        if (lc == 0) {
            pf_l2(wr0); pf_l2(wr1); pf_l2(wr2); pf_l2(wr3);
            if (NS > 3) {
                pf_l2(wr0 + 3 * BS_NVFP4); pf_l2(wr1 + 3 * BS_NVFP4);
                pf_l2(wr2 + 3 * BS_NVFP4); pf_l2(wr3 + 3 * BS_NVFP4);
            }
        }

        __syncthreads();
        if (tid < TR) {
            s_row[tid] = tid < nrows ? (GU ? src[row0 + tid] : (row0 + tid)) : -1;
            if (tid >= nrows) {
                s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
                s_agb[0][tid] = 0.0f; s_agb[1][tid] = 0.0f;
                uint4* z0 = (uint4*) (s_aqs[0] + tid * A_STRIDE);
                uint4* z1 = (uint4*) (s_aqs[1] + tid * A_STRIDE);
                z0[0] = make_uint4(0u, 0u, 0u, 0u); z0[1] = make_uint4(0u, 0u, 0u, 0u);
                z1[0] = make_uint4(0u, 0u, 0u, 0u); z1[1] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        auto issue_act = [&](int st, int buf) {
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

        auto load_w_regs = [&](int st, uint32_t& q0, uint32_t& q1, uint32_t& q2, uint32_t& q3, uint32_t& wsc) {
            const size_t sb = (size_t) st * BS_NVFP4;
            const size_t qo = sb + 4 + (size_t) lc * 4;
            q0 = ld32_fast(wr0 + qo, aligned4);
            q1 = ld32_fast(wr1 + qo, aligned4);
            q2 = ld32_fast(wr2 + qo, aligned4);
            q3 = ld32_fast(wr3 + qo, aligned4);
            if (lc == 0) wsc = ld32_fast(wr0 + sb, aligned4);
            else if (lc == 1) wsc = ld32_fast(wr1 + sb, aligned4);
            else if (lc == 2) wsc = ld32_fast(wr2 + sb, aligned4);
            else if (lc == 3) wsc = ld32_fast(wr3 + sb, aligned4);
            else wsc = 0u;
        };

        auto store_w_smem = [&](int buf, uint32_t q0, uint32_t q1, uint32_t q2, uint32_t q3, uint32_t wsc) {
            *(uint32_t*) (s_wqs[buf] + (rl + 0) * W_STRIDE + lc * 4) = q0;
            *(uint32_t*) (s_wqs[buf] + (rl + 64) * W_STRIDE + lc * 4) = q1;
            *(uint32_t*) (s_wqs[buf] + (rl + 128) * W_STRIDE + lc * 4) = q2;
            *(uint32_t*) (s_wqs[buf] + (rl + 192) * W_STRIDE + lc * 4) = q3;
            if (lc < 4) s_wsc[buf][lc * 64 + rl] = wsc;
        };

        // Stage 0 prologue.
        uint32_t wq0 = 0u, wq1 = 0u, wq2 = 0u, wq3 = 0u, wsc = 0u;
        issue_act(0, 0);
        load_w_regs(0, wq0, wq1, wq2, wq3, wsc);
        store_w_smem(0, wq0, wq1, wq2, wq3, wsc);

        float acc[4][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int st = 0; st < NS; ++st) {
            const int buf = st & 1;
            const int nxt = buf ^ 1;
            cp_wait0();
            __syncthreads();

            // Issue stage st+1 async activation DMA + non-blocking global weight loads into registers BEFORE MMA!
            if (st + 1 < NS) {
                issue_act(st + 1, nxt);
                load_w_regs(st + 1, wq0, wq1, wq2, wq3, wsc);
                if (lc == 0 && ((st & 3) == 0) && st + 4 < NS) {
                    const size_t pf = (size_t) (st + 4) * BS_NVFP4;
                    pf_l2(wr0 + pf); pf_l2(wr1 + pf); pf_l2(wr2 + pf); pf_l2(wr3 + pf);
                }
            }

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

            // Store prefetched stage st+1 weights into `nxt` smem AFTER MMA finishes reading `buf`!
            if (st + 1 < NS) {
                store_w_smem(nxt, wq0, wq1, wq2, wq3, wsc);
            }
        }

        if constexpr (GU) {
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
                const int wf_pair = wf >> 1;
                const int wf_half = wf & 1;
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (r < nrows) {
                            const float am_64 = fmaxf(s_am32[2 * wf_pair + 0][r], s_am32[2 * wf_pair + 1][r]);
                            const float g_blk = am_64 * (1.0f / (6.0f * 448.0f));
                            float s0 = 0.0f, s1 = 0.0f;
                            const uint8_t d0 = encode_ue4m3(g_blk > 0.0f ? (am_sub0[n][c] / (g_blk * 6.0f)) : 0.0f, s0);
                            const uint8_t d1 = encode_ue4m3(g_blk > 0.0f ? (am_sub1[n][c] / (g_blk * 6.0f)) : 0.0f, s1);
                            const float inv0 = (g_blk * s0) > 0.0f ? (1.0f / (g_blk * s0)) : 0.0f;
                            const float inv1 = (g_blk * s1) > 0.0f ? (1.0f / (g_blk * s1)) : 0.0f;

                            uint8_t* dst_slot = out + (size_t) (row0 + r) * H_ROW_BYTES + (size_t) (2 * fb + wf_pair) * AB;
                            dst_slot[16 * wf_half + 0 + g] = pack_e2m1x2(h_val[n][c][0] * inv0, h_val[n][c][1] * inv0);
                            dst_slot[16 * wf_half + 8 + g] = pack_e2m1x2(h_val[n][c][2] * inv1, h_val[n][c][3] * inv1);
                            if (g == 0) {
                                dst_slot[32 + 2 * wf_half + 0] = d0;
                                dst_slot[32 + 2 * wf_half + 1] = d1;
                                if (wf_half == 0) *(float*) (dst_slot + 36) = g_blk;
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
