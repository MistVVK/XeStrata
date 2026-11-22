// src/prefill/moe_fp8_mxf_tc.cu - Clean-room MIT Blackwell (sm_120a / sm_121a) mxf8f6f4 Tensor Core MoE prefill.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// v2 High-Throughput Edition (NT=512, TR=128 single-pass, WR=128):
//   - TR=128 (16 warps = 2 weight halves x 8 routed-row eighths) halves expert weight re-reads on 2K-8K prompts;
//   - 4 threads per weight row (16 weights/thread) with 32-bit coalesced loads + branchless E4M3 XOR sign bit;
//   - Hardware mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0;
//   - Zero spills, ~44.5 KB smem (2 CTAs/SM = 32 warps/SM on sm_120!).
#include "strata/prefill/moe_fp8_mxf_tc.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "strata/core/emulate.hpp"
#include "strata/quant/types.hpp"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace strata::prefill::fp8_mxf_tc {
namespace {

using strata::prefill::fused::Batch;
using strata::prefill::fused::NativeGeom;
using strata::prefill::fused::kMoeEmbd;
using strata::prefill::fused::kMoeFF;
using strata::prefill::fused::kTileRows;

constexpr int T_Q5_1 = (int) strata::quant::Type::Q5_1;
constexpr int T_Q8_0 = (int) strata::quant::Type::Q8_0;
constexpr int T_Q4_K = (int) strata::quant::Type::Q4_K;
constexpr int T_Q5_K = (int) strata::quant::Type::Q5_K;
constexpr int T_IQ3_XXS = (int) strata::quant::Type::IQ3_XXS;
constexpr int T_IQ4_NL = (int) strata::quant::Type::IQ4_NL;
constexpr int T_IQ3_S = (int) strata::quant::Type::IQ3_S;
constexpr int T_IQ2_S = (int) strata::quant::Type::IQ2_S;
constexpr int T_IQ4_XS = (int) strata::quant::Type::IQ4_XS;

constexpr int AB = 80;       // 64 E4M3 bytes + 2 UE8M0 bytes + 14 pad (matches fused::act_bytes!)
constexpr int K_STEP = 64;   // 2 blocks of 32 per stage
constexpr int NT = 512;      // 16 warps: wf in 0..1 (64 weight rows each), wn in 0..7 (16 routed rows each -> TR=128!)
constexpr int TR = 128;      // 2x kTileRows: covers up to 128 routed rows per expert in ONE weight pass!
constexpr int WR = 128;      // GU: 64 gate + 64 up = 64 SwiGLU features; Down: 128 cols
constexpr int W_STRIDE = 80; // 64B E4M3 + 16B pad (16B aligned, conflict-free ldmatrix.m8n8.x4)
constexpr int A_STRIDE = 80;

constexpr int GU_ROWS_N = 2 * kMoeFF;
constexpr int GU_ROWS_K = kMoeEmbd;
constexpr int D_ROWS_N = kMoeEmbd;
constexpr int D_ROWS_K = kMoeFF;
constexpr size_t X_ROW_BYTES = (size_t) (GU_ROWS_K / K_STEP) * AB;
constexpr size_t H_ROW_BYTES = (size_t) (D_ROWS_K / K_STEP) * AB;

static_assert(TR == 2 * kTileRows, "TR is two fused::group tiles");
constexpr bool kShapeOk = (GU_ROWS_K % 256 == 0) && (D_ROWS_K % K_STEP == 0) &&
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
        std::fprintf(stderr, "fp8_mxf_tc %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && (__CUDA_ARCH__ < 1300)
#define STRATA_FP8_MXF_DEV 1
#endif

#if defined(STRATA_FP8_MXF_DEV)
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp16_async(void* dst_smem, const void* src_glob) {
    const uint32_t d = smem_u32(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src_glob));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* src_smem) {
    const uint32_t a = smem_u32(src_smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
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

__device__ __forceinline__ uint16_t ld16_g(const uint8_t* p) {
    return __ldg(reinterpret_cast<const uint16_t*>(p));
}

__device__ __forceinline__ uint32_t ld32_g(const uint8_t* p) {
    return (uint32_t) ld16_g(p) | ((uint32_t) ld16_g(p + 2) << 16);
}

__device__ __forceinline__ uint32_t pack_e4m3x4(float f0, float f1, float f2, float f3) {
    const __nv_fp8x2_e4m3 p0(make_float2(f0, f1));
    const __nv_fp8x2_e4m3 p1(make_float2(f2, f3));
    return (uint32_t) p0.__x | ((uint32_t) p1.__x << 16);
}

// Expand 4 sign bits (in bits 0..3) into bit 7 of each byte: bit j -> 0x80 << (8*j) (branchless E4M3 negate mask!).
__device__ __forceinline__ uint32_t sign4_to_e4m3_mask(uint32_t s4) {
    return ((s4 & 1u) << 7) | ((s4 & 2u) << 14) | ((s4 & 4u) << 21) | ((s4 & 8u) << 28);
}

__device__ __forceinline__ void amax_to_ue8m0(float amax, uint8_t& ue8m0, float& inv) {
    if (amax > 0.0f) {
        const float req = amax * (1.0f / 448.0f);
        const uint32_t u = __float_as_uint(req);
        int e = (int) ((u >> 23) & 0xFFu) - 127;
        if ((u & 0x007FFFFFu) != 0u) ++e;
        e = max(-126, min(127, e));
        ue8m0 = (uint8_t) (e + 127);
        inv = __uint_as_float((uint32_t) (127 - e) << 23);
    } else {
        ue8m0 = 1u;
        inv = 0.0f;
    }
}

__device__ __forceinline__ void split_scale_ue8m0(float scale, uint8_t& ue8m0, float& m_d) {
    const float a = fabsf(scale);
    const uint32_t u = __float_as_uint(a);
    const uint32_t exp_b = (u >> 23) & 0xFFu;
    if (exp_b == 0u || exp_b == 255u) {
        ue8m0 = 1u;
        m_d = 0.0f;
        return;
    }
    ue8m0 = (uint8_t) exp_b;
    const uint32_t sign_bit = __float_as_uint(scale) & 0x80000000u;
    m_d = __uint_as_float(sign_bit | (u & 0x007FFFFFu) | 0x3F800000u);
}

template <int WT>
__host__ __device__ constexpr int grid_words() {
    if constexpr (WT == T_IQ3_S) return 512;
    else if constexpr (WT == T_IQ3_XXS) return 256;
    else if constexpr (WT == T_IQ2_S) return 2048;
    else return 4;
}

template <int WT>
__device__ __forceinline__ void load_grid(uint32_t* s_grid, int tid) {
    if constexpr (WT == T_IQ3_S) {
        if (tid < 512) s_grid[tid] = iq3s_grid[tid];
    } else if constexpr (WT == T_IQ3_XXS) {
        if (tid < 256) s_grid[tid] = iq3xxs_grid[tid];
    } else if constexpr (WT == T_IQ2_S) {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(iq2s_grid);
        for (int i = tid; i < 2048; i += NT) s_grid[i] = src[i];
    }
}

__device__ __forceinline__ void k4_scale_min_ldg(const uint8_t* sc, int j, uint8_t& d_out, uint8_t& m_out) {
    if (j < 4) {
        d_out = __ldg(&sc[j]) & 63u;
        m_out = __ldg(&sc[j + 4]) & 63u;
    } else {
        const uint8_t s4 = __ldg(&sc[j + 4]);
        d_out = (s4 & 0xFu) | ((__ldg(&sc[j - 4]) >> 6) << 4);
        m_out = (s4 >> 4) | ((__ldg(&sc[j]) >> 6) << 4);
    }
}

// Decode HALF (16 weights: half=0 -> 0..15, half=1 -> 16..31) of 32-weight sub-block `ib_global` into 4 uint32s (16 E4M3 bytes)!
// 4 threads per row -> 2x faster weight decode, 32-bit loads, branchless E4M3 XOR sign!
template <int WT>
__device__ __forceinline__ void decode_half16_mxfp8(const uint8_t* row, int ib_global, int half, const uint32_t* s_grid,
                                                    uint32_t (&dst4)[4], uint8_t& out_ue8m0) {
    if constexpr (WT == T_IQ3_S) {
        const block_iq3_s* b = reinterpret_cast<const block_iq3_s*>(row) + (ib_global >> 3);
        const int ib = ib_global & 7;
        const float d = __half2float(__ushort_as_half(ld16_g(reinterpret_cast<const uint8_t*>(&b->d))));
        const uint8_t sc_b = __ldg(&b->scales[ib >> 1]);
        const int sc = (sc_b >> (4 * (ib & 1))) & 0xF;
        float m_d = 0.0f;
        split_scale_ue8m0(d * (float) (1 + 2 * sc), out_ue8m0, m_d);
        const uint32_t qh = __ldg(&b->qh[ib]);
        const uint32_t q4 = ld32_g(b->qs + 8 * ib + 4 * half);
        const uint32_t sg2 = ld16_g(b->signs + 4 * ib + 2 * half);
#pragma unroll
        for (int ll = 0; ll < 2; ++ll) {
            const int l = 2 * half + ll;
            const uint32_t q0 = (q4 >> (16 * ll)) & 0xFFu;
            const uint32_t q1 = (q4 >> (16 * ll + 8)) & 0xFFu;
            const uint32_t g0 = s_grid[q0 | ((qh << (8 - 2 * l)) & 256u)];
            const uint32_t g1 = s_grid[q1 | ((qh << (7 - 2 * l)) & 256u)];
            const uint32_t sg = (sg2 >> (8 * ll)) & 0xFFu;
            const uint32_t e0 = pack_e4m3x4(m_d * (float) (g0 & 0xFFu), m_d * (float) ((g0 >> 8) & 0xFFu),
                                            m_d * (float) ((g0 >> 16) & 0xFFu), m_d * (float) ((g0 >> 24) & 0xFFu));
            const uint32_t e1 = pack_e4m3x4(m_d * (float) (g1 & 0xFFu), m_d * (float) ((g1 >> 8) & 0xFFu),
                                            m_d * (float) ((g1 >> 16) & 0xFFu), m_d * (float) ((g1 >> 24) & 0xFFu));
            dst4[2 * ll + 0] = e0 ^ sign4_to_e4m3_mask(sg);
            dst4[2 * ll + 1] = e1 ^ sign4_to_e4m3_mask(sg >> 4);
        }
    } else if constexpr (WT == T_IQ3_XXS) {
        const block_iq3_xxs* b = reinterpret_cast<const block_iq3_xxs*>(row) + (ib_global >> 3);
        const int ib = ib_global & 7;
        const float d = __half2float(__ushort_as_half(ld16_g(reinterpret_cast<const uint8_t*>(&b->d))));
        const uint32_t aux32 = ld32_g(b->qs + QK_K / 4 + 4 * ib);
        float m_d = 0.0f;
        split_scale_ue8m0(d * (0.5f + (float) (aux32 >> 28)) * 0.5f, out_ue8m0, m_d);
        const uint32_t q4 = ld32_g(b->qs + 8 * ib + 4 * half);
#pragma unroll
        for (int ll = 0; ll < 2; ++ll) {
            const int l = 2 * half + ll;
            const uint32_t g0 = s_grid[(q4 >> (16 * ll)) & 0xFFu];
            const uint32_t g1 = s_grid[(q4 >> (16 * ll + 8)) & 0xFFu];
            const uint32_t sg = ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
            const uint32_t e0 = pack_e4m3x4(m_d * (float) (g0 & 0xFFu), m_d * (float) ((g0 >> 8) & 0xFFu),
                                            m_d * (float) ((g0 >> 16) & 0xFFu), m_d * (float) ((g0 >> 24) & 0xFFu));
            const uint32_t e1 = pack_e4m3x4(m_d * (float) (g1 & 0xFFu), m_d * (float) ((g1 >> 8) & 0xFFu),
                                            m_d * (float) ((g1 >> 16) & 0xFFu), m_d * (float) ((g1 >> 24) & 0xFFu));
            dst4[2 * ll + 0] = e0 ^ sign4_to_e4m3_mask(sg);
            dst4[2 * ll + 1] = e1 ^ sign4_to_e4m3_mask(sg >> 4);
        }
    } else if constexpr (WT == T_IQ2_S) {
        const block_iq2_s* b = reinterpret_cast<const block_iq2_s*>(row) + (ib_global >> 3);
        const int ib = ib_global & 7;
        const float d = __half2float(__ushort_as_half(ld16_g(reinterpret_cast<const uint8_t*>(&b->d))));
        const uint8_t sc = __ldg(&b->scales[ib]);
        const float d0 = d * (0.5f + (float) (sc & 0xF)) * 0.25f;
        const float d1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
        const float amax = fmaxf(fabsf(d0), fabsf(d1)) * 43.0f;
        float inv = 0.0f;
        amax_to_ue8m0(amax, out_ue8m0, inv);
        const float m = (half == 0 ? d0 : d1) * inv;
        const uint32_t qh = __ldg(&b->qh[ib]);
        const uint32_t q2 = ld16_g(b->qs + 4 * ib + 2 * half);
        const uint32_t sg2 = ld16_g(b->qs + QK_K / 8 + 4 * ib + 2 * half);
        const uint2* g2 = reinterpret_cast<const uint2*>(s_grid);
#pragma unroll
        for (int ll = 0; ll < 2; ++ll) {
            const int l = 2 * half + ll;
            const uint32_t idx = ((q2 >> (8 * ll)) & 0xFFu) | ((qh << (8 - 2 * l)) & 0x300u);
            const uint2 gv = g2[idx];
            const uint32_t sg = (sg2 >> (8 * ll)) & 0xFFu;
            const uint32_t e0 = pack_e4m3x4(m * (float) (gv.x & 0xFFu), m * (float) ((gv.x >> 8) & 0xFFu),
                                            m * (float) ((gv.x >> 16) & 0xFFu), m * (float) ((gv.x >> 24) & 0xFFu));
            const uint32_t e1 = pack_e4m3x4(m * (float) (gv.y & 0xFFu), m * (float) ((gv.y >> 8) & 0xFFu),
                                            m * (float) ((gv.y >> 16) & 0xFFu), m * (float) ((gv.y >> 24) & 0xFFu));
            dst4[2 * ll + 0] = e0 ^ sign4_to_e4m3_mask(sg);
            dst4[2 * ll + 1] = e1 ^ sign4_to_e4m3_mask(sg >> 4);
        }
    } else if constexpr (WT == T_IQ4_XS) {
        const block_iq4_xs* b = reinterpret_cast<const block_iq4_xs*>(row) + (ib_global >> 3);
        const int ib = ib_global & 7;
        const float d = __half2float(__ushort_as_half(ld16_g(reinterpret_cast<const uint8_t*>(&b->d))));
        const uint16_t sh = ld16_g(reinterpret_cast<const uint8_t*>(&b->scales_h));
        const uint8_t sl = __ldg(&b->scales_l[ib >> 1]);
        const int ls = ((sl >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        float m_d = 0.0f;
        split_scale_ue8m0(d * (float) (ls - 32), out_ue8m0, m_d);
        const uint8_t* qs = b->qs + 16 * ib;
        const int shift = half ? 4 : 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t raw = ld32_g(qs + 4 * k);
            dst4[k] = pack_e4m3x4(m_d * (float) kvalues_iq4nl[(raw >> shift) & 15u],
                                  m_d * (float) kvalues_iq4nl[(raw >> (8 + shift)) & 15u],
                                  m_d * (float) kvalues_iq4nl[(raw >> (16 + shift)) & 15u],
                                  m_d * (float) kvalues_iq4nl[(raw >> (24 + shift)) & 15u]);
        }
    } else if constexpr (WT == T_IQ4_NL) {
        const block_iq4_nl* b = reinterpret_cast<const block_iq4_nl*>(row) + ib_global;
        const float d = __half2float(__ushort_as_half(ld16_g(reinterpret_cast<const uint8_t*>(&b->d))));
        float m_d = 0.0f;
        split_scale_ue8m0(d, out_ue8m0, m_d);
        const int shift = half ? 4 : 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t raw = ld32_g(b->qs + 4 * k);
            dst4[k] = pack_e4m3x4(m_d * (float) kvalues_iq4nl[(raw >> shift) & 15u],
                                  m_d * (float) kvalues_iq4nl[(raw >> (8 + shift)) & 15u],
                                  m_d * (float) kvalues_iq4nl[(raw >> (16 + shift)) & 15u],
                                  m_d * (float) kvalues_iq4nl[(raw >> (24 + shift)) & 15u]);
        }
    } else if constexpr (WT == T_Q8_0) {
        const block_q8_0* b = reinterpret_cast<const block_q8_0*>(row) + ib_global;
        const float d = __half2float(__ushort_as_half(ld16_g(reinterpret_cast<const uint8_t*>(&b->d))));
        float m_d = 0.0f;
        split_scale_ue8m0(d, out_ue8m0, m_d);
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(b->qs) + 16 * half;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t q4 = ld32_g(qs + 4 * k);
            dst4[k] = pack_e4m3x4(m_d * (float) (int8_t) (q4 & 0xFFu),
                                  m_d * (float) (int8_t) ((q4 >> 8) & 0xFFu),
                                  m_d * (float) (int8_t) ((q4 >> 16) & 0xFFu),
                                  m_d * (float) (int8_t) ((q4 >> 24) & 0xFFu));
        }
    } else if constexpr (WT == T_Q4_K) {
        const block_q4_K* b = reinterpret_cast<const block_q4_K*>(row) + (ib_global >> 3);
        const int ib = ib_global & 7;
        const uint8_t* p_dm = reinterpret_cast<const uint8_t*>(&b->dm);
        const float d_sup = __half2float(__ushort_as_half(ld16_g(p_dm)));
        const float m_sup = __half2float(__ushort_as_half(ld16_g(p_dm + 2)));
        uint8_t sc = 0, mn = 0;
        k4_scale_min_ldg(b->scales, ib, sc, mn);
        const float d = d_sup * (float) sc;
        const float m = m_sup * (float) mn;
        const float amax = fmaxf(fabsf(15.0f * d - m), fabsf(m));
        float inv = 0.0f;
        amax_to_ue8m0(amax, out_ue8m0, inv);
        const float ds = d * inv;
        const float ms = -m * inv;
        const uint8_t* qs = b->qs + 32 * (ib >> 1) + 16 * half;
        const int sh = (ib & 1) ? 4 : 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t raw = ld32_g(qs + 4 * k);
            dst4[k] = pack_e4m3x4(fmaf(ds, (float) ((raw >> sh) & 15u), ms),
                                  fmaf(ds, (float) ((raw >> (8 + sh)) & 15u), ms),
                                  fmaf(ds, (float) ((raw >> (16 + sh)) & 15u), ms),
                                  fmaf(ds, (float) ((raw >> (24 + sh)) & 15u), ms));
        }
    } else if constexpr (WT == T_Q5_K) {
        const block_q5_K* b = reinterpret_cast<const block_q5_K*>(row) + (ib_global >> 3);
        const int ib = ib_global & 7;
        const uint8_t* p_dm = reinterpret_cast<const uint8_t*>(&b->dm);
        const float d_sup = __half2float(__ushort_as_half(ld16_g(p_dm)));
        const float m_sup = __half2float(__ushort_as_half(ld16_g(p_dm + 2)));
        uint8_t sc = 0, mn = 0;
        k4_scale_min_ldg(b->scales, ib, sc, mn);
        const float d = d_sup * (float) sc;
        const float m = m_sup * (float) mn;
        const float amax = fmaxf(fabsf(31.0f * d - m), fabsf(m));
        float inv = 0.0f;
        amax_to_ue8m0(amax, out_ue8m0, inv);
        const float ds = d * inv;
        const float ms = -m * inv;
        const uint8_t* qs = b->qs + 32 * (ib >> 1) + 16 * half;
        const uint8_t* qh = b->qh + 16 * half;
        const int sh = (ib & 1) ? 4 : 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t raw = ld32_g(qs + 4 * k);
            const uint32_t h4 = ld32_g(qh + 4 * k);
            const int q0 = (int) ((raw >> sh) & 15u) | (int) (((h4 >> ib) & 1u) << 4);
            const int q1 = (int) ((raw >> (8 + sh)) & 15u) | (int) ((((h4 >> 8) >> ib) & 1u) << 4);
            const int q2 = (int) ((raw >> (16 + sh)) & 15u) | (int) ((((h4 >> 16) >> ib) & 1u) << 4);
            const int q3 = (int) ((raw >> (24 + sh)) & 15u) | (int) ((((h4 >> 24) >> ib) & 1u) << 4);
            dst4[k] = pack_e4m3x4(fmaf(ds, (float) q0, ms), fmaf(ds, (float) q1, ms),
                                  fmaf(ds, (float) q2, ms), fmaf(ds, (float) q3, ms));
        }
    } else if constexpr (WT == T_Q5_1) {
        const block_q5_1* b = reinterpret_cast<const block_q5_1*>(row) + ib_global;
        const uint8_t* p_dm = reinterpret_cast<const uint8_t*>(&b->dm);
        const float d = __half2float(__ushort_as_half(ld16_g(p_dm)));
        const float m = __half2float(__ushort_as_half(ld16_g(p_dm + 2)));
        const float amax = fmaxf(fabsf(31.0f * d + m), fabsf(m));
        float inv = 0.0f;
        amax_to_ue8m0(amax, out_ue8m0, inv);
        const float ds = d * inv;
        const float ms = m * inv;
        const uint32_t qh = ld32_g(b->qh);
        const int shift = half ? 4 : 0;
        const int hbase = half ? 16 : 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t raw = ld32_g(b->qs + 4 * k);
            const int j0 = hbase + 4 * k;
            const int q0 = (int) ((raw >> shift) & 15u) | (int) (((qh >> (j0 + 0)) & 1u) << 4);
            const int q1 = (int) ((raw >> (8 + shift)) & 15u) | (int) (((qh >> (j0 + 1)) & 1u) << 4);
            const int q2 = (int) ((raw >> (16 + shift)) & 15u) | (int) (((qh >> (j0 + 2)) & 1u) << 4);
            const int q3 = (int) ((raw >> (24 + shift)) & 15u) | (int) (((qh >> (j0 + 3)) & 1u) << 4);
            dst4[k] = pack_e4m3x4(fmaf(ds, (float) q0, ms), fmaf(ds, (float) q1, ms),
                                  fmaf(ds, (float) q2, ms), fmaf(ds, (float) q3, ms));
        }
    }
}
#endif

__global__ void __launch_bounds__(256) quant_act_mxfp8_kernel(const float* __restrict__ x, int64_t nblk,
                                                              uint8_t* __restrict__ xa) {
#if defined(STRATA_FP8_MXF_DEV)
    const int lane = threadIdx.x & 31;
    const int64_t stride = (int64_t) gridDim.x * 8;
    for (int64_t w = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5); w < nblk; w += stride) {
        const float* xb = x + w * K_STEP;
        const float r0 = xb[lane];
        const float r1 = xb[32 + lane];
        const float v0 = isfinite(r0) ? r0 : 0.0f;
        const float v1 = isfinite(r1) ? r1 : 0.0f;
        float a0 = fabsf(v0);
        float a1 = fabsf(v1);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
            a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
        }
        uint8_t sc0 = 1u, sc1 = 1u;
        float inv0 = 0.0f, inv1 = 0.0f;
        amax_to_ue8m0(a0, sc0, inv0);
        amax_to_ue8m0(a1, sc1, inv1);
        uint8_t* out = xa + w * AB;
        const float q0f = v0 * inv0;
        const float q1f = v1 * inv1;
        const __nv_fp8_e4m3 q0(isfinite(q0f) ? q0f : 0.0f);
        const __nv_fp8_e4m3 q1(isfinite(q1f) ? q1f : 0.0f);
        out[lane] = q0.__x;
        out[32 + lane] = q1.__x;
        if (lane == 0) {
            out[64] = sc0;
            out[65] = sc1;
        }
    }
#else
    (void) x; (void) nblk; (void) xa;
#endif
}

// Persistent TR=128 single-pass kernel (NT=512 = 16 warps):
// Pairs consecutive 64-row tiles of the same expert (tb.tiles[t] and tb.tiles[t+1] when (row0 & 64) == 0)
// so experts with 65..128 rows read & decode their weights ONLY ONCE!
template <int WT, bool GU>
__global__ void __launch_bounds__(NT, 1) fp8_mxf_tc_kernel(Batch b, NativeGeom geo, Tables tb,
                                                           const uint8_t* __restrict__ act,
                                                           const int32_t* __restrict__ src, uint8_t* __restrict__ out,
                                                           float* __restrict__ dm) {
#if defined(STRATA_FP8_MXF_DEV)
    if constexpr (!kShapeOk) return;
    constexpr int ROWS_N = GU ? GU_ROWS_N : D_ROWS_N;
    constexpr int ROWS_K = GU ? GU_ROWS_K : D_ROWS_K;
    constexpr int NS = ROWS_K / K_STEP;
    constexpr int NFB = ROWS_N / WR;
    constexpr size_t ACT_ROW_BYTES = GU ? X_ROW_BYTES : H_ROW_BYTES;

    __shared__ alignas(16) uint8_t s_wqs[2][WR * W_STRIDE]; // 2 * 10,240 = 20,480 B
    __shared__ alignas(16) uint8_t s_aqs[2][TR * A_STRIDE]; // 2 * 10,240 = 20,480 B
    __shared__ uint8_t s_wsc[2][2][WR];                     //                 512 B
    __shared__ uint8_t s_asc[2][2][TR];                     //                 512 B
    __shared__ int32_t s_row[TR];                           //                 512 B
    __shared__ alignas(16) uint32_t s_grid[grid_words<WT>()];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf = warp >> 3;          // 0..1: 64 weight rows each (GU: 32 gate + 32 up)
    const int wn = warp & 7;           // 0..7: 16 routed rows each -> TR=128!
    const int nb0 = 16 * wn;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int lm = lane >> 3;
    const int l8 = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int dec_r = tid >> 2;        // 0..127: 128 weight rows
    const int dec_sb = (tid >> 1) & 1; // 0..1: 2 sub-blocks of 32
    const int dec_h = tid & 1;         // 0..1: 2 halves of 16 weights per sub-block (4 threads/row!)

    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * W_STRIDE + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * A_STRIDE + 16 * (lm & 1);

    load_grid<WT>(s_grid, tid);
    __syncthreads();

    const int t0 = tb.ts[b.e0];
    const int nwork = (tb.ts[b.e1] - t0) * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int e = tl.x;
        const int row0 = tl.y;
        // Pair two 64-row tiles into one 128-row work item: skip odd-numbered tiles of the same expert!
        if (((row0 - tb.off[e]) & kTileRows) != 0) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const int fb = w % NFB;
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * WR;
        const uint8_t* const my_wrow = GU
            ? (blob + ((dec_r & 8) ? geo.up_off : 0) +
               (size_t) (fb * (WR / 2) + 8 * (dec_r >> 4) + (dec_r & 7)) * geo.gu_row)
            : (blob + geo.down_off + (size_t) (rbase + dec_r) * geo.d_row);

        __syncthreads();
        if (tid < TR) {
            s_row[tid] = tid < nrows ? (GU ? src[row0 + tid] : (row0 + tid)) : -1;
            if (tid >= nrows) {
                s_asc[0][0][tid] = 1u; s_asc[0][1][tid] = 1u;
                s_asc[1][0][tid] = 1u; s_asc[1][1][tid] = 1u;
                uint4* z0 = reinterpret_cast<uint4*>(s_aqs[0] + tid * A_STRIDE);
                uint4* z1 = reinterpret_cast<uint4*>(s_aqs[1] + tid * A_STRIDE);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    z0[q] = make_uint4(0u, 0u, 0u, 0u);
                    z1[q] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_act = [&](int st, int buf) {
            // 512 threads copy 128 rows x 64 bytes = 512 x 16-byte chunks via cp.async (1 chunk/thread!)
            const int r = tid >> 2;
            const int h = tid & 3;
            if (r < nrows) {
                const uint8_t* ap = act + (size_t) s_row[r] * ACT_ROW_BYTES + (size_t) st * AB;
                cp16_async(s_aqs[buf] + r * A_STRIDE + 16 * h, ap + 16 * h);
                if (h == 0) {
                    s_asc[buf][0][r] = ap[64];
                    s_asc[buf][1][r] = ap[65];
                }
            }
            cp_commit();
        };

        // Stage 0 prologue.
        issue_act(0, 0);
        {
            uint32_t q4[4];
            uint8_t sc = 1u;
            decode_half16_mxfp8<WT>(my_wrow, dec_sb, dec_h, s_grid, q4, sc);
            *reinterpret_cast<uint4*>(s_wqs[0] + dec_r * W_STRIDE + dec_sb * 32 + dec_h * 16) =
                make_uint4(q4[0], q4[1], q4[2], q4[3]);
            if (dec_h == 0) s_wsc[0][dec_sb][dec_r] = sc;
        }

        float acc[4][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int st = 0; st < NS; ++st) {
            const int buf = st & 1;
            const int nxt = buf ^ 1;
            cp_wait0();
            __syncthreads();

            uint32_t nxt_q[4];
            uint8_t nxt_sc = 1u;
            if (st + 1 < NS) {
                issue_act(st + 1, nxt);
                decode_half16_mxfp8<WT>(my_wrow, (st + 1) * 2 + dec_sb, dec_h, s_grid, nxt_q, nxt_sc);
            }

            if (on0) {
                uint32_t bq0[4];
                ldsm4(bq0, s_aqs[buf] + b_off);
                const uint32_t sfb0_0 = (uint32_t) s_asc[buf][0][nb0 + g];
                const uint32_t sfb1_0 = on1 ? (uint32_t) s_asc[buf][0][nb0 + 8 + g] : 1u;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * W_STRIDE);
                    const uint32_t sfa = (uint32_t) s_wsc[buf][0][64 * wf + 16 * i + tidx_a];
                    mma_mxfp8_m16n8k32(acc[i][0], a, bq0[0], bq0[1], sfa, sfb0_0);
                    if (on1) mma_mxfp8_m16n8k32(acc[i][1], a, bq0[2], bq0[3], sfa, sfb1_0);
                }

                uint32_t bq1[4];
                ldsm4(bq1, s_aqs[buf] + b_off + 32);
                const uint32_t sfb0_1 = (uint32_t) s_asc[buf][1][nb0 + g];
                const uint32_t sfb1_1 = on1 ? (uint32_t) s_asc[buf][1][nb0 + 8 + g] : 1u;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * W_STRIDE + 32);
                    const uint32_t sfa = (uint32_t) s_wsc[buf][1][64 * wf + 16 * i + tidx_a];
                    mma_mxfp8_m16n8k32(acc[i][0], a, bq1[0], bq1[1], sfa, sfb0_1);
                    if (on1) mma_mxfp8_m16n8k32(acc[i][1], a, bq1[2], bq1[3], sfa, sfb1_1);
                }
            }

            if (st + 1 < NS) {
                *reinterpret_cast<uint4*>(s_wqs[nxt] + dec_r * W_STRIDE + dec_sb * 32 + dec_h * 16) =
                    make_uint4(nxt_q[0], nxt_q[1], nxt_q[2], nxt_q[3]);
                if (dec_h == 0) s_wsc[nxt][dec_sb][dec_r] = nxt_sc;
            }
        }

        if constexpr (GU) {
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        float hv[4];
                        float am = 0.0f;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float gt = acc[i][n][c];
                            const float up = acc[i][n][2 + c];
                            const float hv_raw = (gt / (1.0f + __expf(-gt))) * up;
                            hv[i] = isfinite(hv_raw) ? hv_raw : 0.0f;
                            am = fmaxf(am, fabsf(hv[i]));
                        }
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                        if (r < nrows) {
                            uint8_t sc = 1u;
                            float inv = 0.0f;
                            amax_to_ue8m0(am, sc, inv);
                            uint8_t* dst_slot = out + (size_t) (row0 + r) * H_ROW_BYTES + (size_t) fb * AB;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const float qf = hv[i] * inv;
                                const __nv_fp8_e4m3 q(isfinite(qf) ? qf : 0.0f);
                                dst_slot[32 * wf + 8 * i + g] = q.__x;
                            }
                            if (g == 0) dst_slot[64 + wf] = sc;
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
                                acc[i][n][q];
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
    int occ = 1;
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
                int occ = 0;
                if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fp8_mxf_tc_kernel<T_IQ3_S, true>, NT, 0) ==
                        cudaSuccess &&
                    occ > 0) {
                    st.ok = true;
                    st.sms = sms;
                    st.occ = occ;
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

bool gu_covered(int t) {
    return t == T_IQ3_S || t == T_IQ3_XXS || t == T_IQ2_S || t == T_IQ4_XS ||
           t == T_Q4_K || t == T_Q5_K || t == T_Q8_0;
}

bool d_covered(int t) {
    return t == T_IQ4_NL || t == T_Q5_1 || t == T_Q8_0;
}

}  // namespace

bool built() { return true; }

bool available() { return dev_state().ok; }

bool wanted(int gu_type, int d_type) {
    static const int env = [] {
        const char* v = std::getenv("STRATA_PF_MXFP8_MOE");
        if (v == nullptr || v[0] == '\0') return 0;   // opt-in: TR=128 wins at >=8K prompts; keep lossless W4A8 default for 2K prompts
        return std::atoi(v) != 0 ? 1 : 0;
    }();
    if (env == 0) return false;
    return available() && gu_covered(gu_type) && d_covered(d_type);
}

void quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (cols < 0 || cols % K_STEP != 0) {
        std::fprintf(stderr, "fp8_mxf_tc::quantize_act: cols=%lld must be a non-negative multiple of %d\n",
                     (long long) cols, K_STEP);
        std::exit(1);
    }
    if (rows <= 0 || cols == 0) return;
    const int64_t blocks_per_row = cols / K_STEP;
    static constexpr int64_t kMaxGridX = 0x7fffffffLL;
    if (rows > (kMaxGridX * 8) / blocks_per_row) {
        std::fprintf(stderr, "fp8_mxf_tc::quantize_act: rows=%lld cols=%lld exceeds CUDA grid.x limit\n",
                     (long long) rows, (long long) cols);
        std::exit(1);
    }
    const int64_t nblk = rows * blocks_per_row;
    // Each warp handles one 64-value block. Large prefill chunks need more than 65535 CTAs in grid.x.
    const unsigned grid = (unsigned) ((nblk + 7) / 8);
    quant_act_mxfp8_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xa);
    ck(cudaGetLastError(), "quantize_act");
}

void experts(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
             const int32_t* src, void* ha, float* dm, void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevState& d = dev_state();
    if (!d.ok || !gu_covered(g.gu_type) || !d_covered(g.d_type)) {
        std::fprintf(stderr, "fp8_mxf_tc::experts: unsupported types %d / %d\n", g.gu_type, g.d_type);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables_of(scratch, n_expert);
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned grid_gu = (unsigned) std::max<int64_t>(1, std::min<int64_t>(tiles * (GU_ROWS_N / WR), (int64_t) d.sms * d.occ));
    const unsigned grid_d = (unsigned) std::max<int64_t>(1, std::min<int64_t>(tiles * (D_ROWS_N / WR), (int64_t) d.sms * d.occ));

    switch (g.gu_type) {
        case T_IQ3_S:   fp8_mxf_tc_kernel<T_IQ3_S, true><<<grid_gu, NT, 0, s>>>(b, g, tb, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr); break;
        case T_IQ3_XXS: fp8_mxf_tc_kernel<T_IQ3_XXS, true><<<grid_gu, NT, 0, s>>>(b, g, tb, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr); break;
        case T_IQ2_S:   fp8_mxf_tc_kernel<T_IQ2_S, true><<<grid_gu, NT, 0, s>>>(b, g, tb, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr); break;
        case T_IQ4_XS:  fp8_mxf_tc_kernel<T_IQ4_XS, true><<<grid_gu, NT, 0, s>>>(b, g, tb, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr); break;
        case T_Q4_K:    fp8_mxf_tc_kernel<T_Q4_K, true><<<grid_gu, NT, 0, s>>>(b, g, tb, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr); break;
        case T_Q5_K:    fp8_mxf_tc_kernel<T_Q5_K, true><<<grid_gu, NT, 0, s>>>(b, g, tb, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr); break;
        default:        fp8_mxf_tc_kernel<T_Q8_0, true><<<grid_gu, NT, 0, s>>>(b, g, tb, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr); break;
    }
    switch (g.d_type) {
        case T_IQ4_NL:  fp8_mxf_tc_kernel<T_IQ4_NL, false><<<grid_d, NT, 0, s>>>(b, g, tb, (const uint8_t*) ha, src, nullptr, dm); break;
        case T_Q5_1:    fp8_mxf_tc_kernel<T_Q5_1, false><<<grid_d, NT, 0, s>>>(b, g, tb, (const uint8_t*) ha, src, nullptr, dm); break;
        default:        fp8_mxf_tc_kernel<T_Q8_0, false><<<grid_d, NT, 0, s>>>(b, g, tb, (const uint8_t*) ha, src, nullptr, dm); break;
    }
    ck(cudaGetLastError(), "experts");
}

}  // namespace strata::prefill::fp8_mxf_tc
