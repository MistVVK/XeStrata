// src/kernels/cuda/dense_i8_tc.cu - Clean-room MIT INT8 Tensor Core dense GEMM (sm_80+: RTX 3090 & RTX 5090).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Derived strictly from NVIDIA PTX ISA mma.sync.aligned.m16n8k32 / m16n8k16 (.s32.s8.s8.s32, sm_80+) and standard
// GGUF block layouts (Q8_0=8, Q4_K=12, Q5_K=13, Q6_K=14, IQ4_XS=23).
//   - Decode / verify (`dense_i8_tc_mmvq`): K-chunked shared-memory activation caching (K_CHUNK=256 -> only ~5 KB smem
//     per CTA for T <= 16, high occupancy across all K = 256..10240) + warp-level INT8 Tensor Core accumulation.
//   - Prompt prefill (`dense_i8_tc_prefill`): quantizes FP16 activations -> block_q8_1 into caller's scratch, then runs
//     64-row x 32-token (or 64-token) INT8 Tensor Core tiles reading quantized GGUF weight blocks directly (no global
//     FP16 weight dequantization round-trip).
#include "strata/kernels/dense_i8_tc.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int kTypeQ8_0 = 8;
constexpr int kTypeQ4_K = 12;
constexpr int kTypeQ5_K = 13;
constexpr int kTypeQ6_K = 14;
constexpr int kTypeIQ4_XS = 23;

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36);

#if !defined(__HIPCC__) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define STRATA_DENSE_I8_SM80 1
#else
#define STRATA_DENSE_I8_SM80 0
#endif

// Standard IQ4_NL / IQ4_XS non-linear int8 codebook packed as 4 little-endian uint32 immediates:
//   {-127, -104, -83, -65}, {-49, -35, -22, -10}, {1, 13, 25, 38}, {53, 69, 89, 113}
constexpr uint32_t kIq4NlW0 = 0xBFAD9881u;
constexpr uint32_t kIq4NlW1 = 0xF6EADDCFu;
constexpr uint32_t kIq4NlW2 = 0x26190D01u;
constexpr uint32_t kIq4NlW3 = 0x71594535u;

// Unaligned-safe 32-bit load from 2-byte-aligned GGUF blocks (Q8_0=34B, Q6_K=210B, IQ4_XS=136B).
__device__ __forceinline__ uint32_t ld32_b2(const uint8_t* p) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t) __ldg(p16) | ((uint32_t) __ldg(p16 + 1) << 16);
}

__device__ __forceinline__ float ld_fp16_b2(const uint8_t* p) {
    const uint16_t bits = __ldg(reinterpret_cast<const uint16_t*>(p));
    return __half2float(__ushort_as_half(bits));
}

// Fast 16-entry signed int8 LUT via __byte_perm: maps 4 low nibbles -> .x and 4 high nibbles -> .y.
__device__ __forceinline__ uint2 iq4nl_lut8(uint32_t q4_bytes) {
    const uint32_t sel = 0x32103210u | ((q4_bytes & 0x88888888u) >> 1);
    const uint32_t lo0 = __byte_perm(kIq4NlW0, kIq4NlW1, q4_bytes);
    const uint32_t hi0 = __byte_perm(kIq4NlW2, kIq4NlW3, q4_bytes);
    const uint32_t t0 = __byte_perm(lo0, hi0, sel);
    const uint32_t lo1 = __byte_perm(kIq4NlW0, kIq4NlW1, q4_bytes >> 16);
    const uint32_t hi1 = __byte_perm(kIq4NlW2, kIq4NlW3, q4_bytes >> 16);
    const uint32_t t1 = __byte_perm(lo1, hi1, sel >> 16);
    return make_uint2(__byte_perm(t0, t1, 0x6420), __byte_perm(t0, t1, 0x7531));
}

// Decode Q4_K / Q5_K 6-bit scale and min for sub-block sb in 0..7.
__device__ __forceinline__ void decode_k4_scale_min(const uint8_t* scales, int sb, float& sc, float& mn) {
    uint8_t s, m;
    if (sb < 4) {
        s = scales[sb] & 63u;
        m = scales[sb + 4] & 63u;
    } else {
        s = (scales[sb + 4] & 0x0Fu) | ((scales[sb - 4] >> 6) << 4);
        m = (scales[sb + 4] >> 4) | ((scales[sb] >> 6) << 4);
    }
    sc = (float) s;
    mn = (float) m;
}

#if STRATA_DENSE_I8_SM80
__device__ __forceinline__ void mma_s8_m16n8k32(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8_m16n8k16(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}
#endif

// ---- FP16 -> block_q8_1 quantizer for prefill activations (1 warp = 32 elements = 1 block_q8_1) ----
__global__ void quantize_f16_to_q8_1_kernel(const uint16_t* __restrict__ x_f16, Q81Block* __restrict__ out,
                                            int64_t total_blocks) {
    const int64_t b = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (b >= total_blocks) return;
    const int lane = threadIdx.x & 31;
    const float v = __half2float(__ushort_as_half(x_f16[b * 32 + lane]));
    float amax = fabsf(v);
    float sum = v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.0f;
    const int q = d > 0.0f ? __float2int_rn(v / d) : 0;
    const int qc = max(-127, min(127, q));
    out[b].qs[lane] = (int8_t) qc;
    if (lane == 0) out[b].ds = __floats2half2_rn(d, sum);
}

// ---- Core quantized weight-tile x activation-slab INT8 TC evaluator ----
// Evaluates one 16-row weight tile (rows r0 = r_base + g, r1 = r_base + 8 + g) against NTILES*8 activation columns
// stored in shared memory for a K-chunk of 256 elements (8 block_q8_1s; or fewer on Q8_0 tail).
template <int GTYPE, int NTILES>
__device__ __forceinline__ void eval_kchunk_tc(
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
    bool v0, bool v1, int nb_here, int k_blk0,
    const int8_t* __restrict__ s_xq, const float* __restrict__ s_dx, const float* __restrict__ s_xs,
    int g, int tig, float (&acc)[NTILES][4]) {
#if STRATA_DENSE_I8_SM80
    if constexpr (GTYPE == kTypeQ8_0) {
        for (int b = 0; b < nb_here; ++b) {
            const uint8_t* blk0 = w0 + (size_t) (k_blk0 + b) * 34u;
            const uint8_t* blk1 = w1 + (size_t) (k_blk0 + b) * 34u;
            const float dw0 = v0 ? ld_fp16_b2(blk0) : 0.0f;
            const float dw1 = v1 ? ld_fp16_b2(blk1) : 0.0f;
            const uint32_t a0 = v0 ? ld32_b2(blk0 + 2 + tig * 4) : 0u;
            const uint32_t a2 = v0 ? ld32_b2(blk0 + 2 + 16 + tig * 4) : 0u;
            const uint32_t a1 = v1 ? ld32_b2(blk1 + 2 + tig * 4) : 0u;
            const uint32_t a3 = v1 ? ld32_b2(blk1 + 2 + 16 + tig * 4) : 0u;

#pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                const int tg = nt * 8 + g;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                int32_t d[4] = {0, 0, 0, 0};
                mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1);

                const int t0 = nt * 8 + 2 * tig;
                const int t1 = t0 + 1;
                const float dx0 = s_dx[t0 * 8 + b];
                const float dx1 = s_dx[t1 * 8 + b];
                acc[nt][0] = fmaf(dw0 * dx0, (float) d[0], acc[nt][0]);
                acc[nt][1] = fmaf(dw0 * dx1, (float) d[1], acc[nt][1]);
                acc[nt][2] = fmaf(dw1 * dx0, (float) d[2], acc[nt][2]);
                acc[nt][3] = fmaf(dw1 * dx1, (float) d[3], acc[nt][3]);
            }
        }
    } else if constexpr (GTYPE == kTypeQ4_K) {
        const int sb_idx = k_blk0 >> 3;
        const uint8_t* blk0 = w0 + (size_t) sb_idx * 144u;
        const uint8_t* blk1 = w1 + (size_t) sb_idx * 144u;
        const half2 dm0 = v0 ? *reinterpret_cast<const half2*>(blk0) : __floats2half2_rn(0.0f, 0.0f);
        const half2 dm1 = v1 ? *reinterpret_cast<const half2*>(blk1) : __floats2half2_rn(0.0f, 0.0f);
        const float d0_all = __low2float(dm0), m0_all = __high2float(dm0);
        const float d1_all = __low2float(dm1), m1_all = __high2float(dm1);
        const uint8_t* sc0_p = blk0 + 4;
        const uint8_t* sc1_p = blk1 + 4;
        const uint8_t* qs0_p = blk0 + 16;
        const uint8_t* qs1_p = blk1 + 16;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            // Load 32 bytes for pair of sub-blocks (2*j = low nibbles, 2*j+1 = high nibbles) ONCE!
            const uint32_t raw0_lo = v0 ? __ldg(reinterpret_cast<const uint32_t*>(qs0_p + 32 * j + tig * 4)) : 0u;
            const uint32_t raw0_hi = v0 ? __ldg(reinterpret_cast<const uint32_t*>(qs0_p + 32 * j + 16 + tig * 4)) : 0u;
            const uint32_t raw1_lo = v1 ? __ldg(reinterpret_cast<const uint32_t*>(qs1_p + 32 * j + tig * 4)) : 0u;
            const uint32_t raw1_hi = v1 ? __ldg(reinterpret_cast<const uint32_t*>(qs1_p + 32 * j + 16 + tig * 4)) : 0u;

#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                const int b = 2 * j + hi;
                float s0 = 0.0f, mn0 = 0.0f, s1 = 0.0f, mn1 = 0.0f;
                if (v0) decode_k4_scale_min(sc0_p, b, s0, mn0);
                if (v1) decode_k4_scale_min(sc1_p, b, s1, mn1);
                const float ds0 = d0_all * s0, ms0 = m0_all * mn0;
                const float ds1 = d1_all * s1, ms1 = m1_all * mn1;

                const uint32_t a0 = (hi == 0) ? (raw0_lo & 0x0F0F0F0Fu) : ((raw0_lo >> 4) & 0x0F0F0F0Fu);
                const uint32_t a2 = (hi == 0) ? (raw0_hi & 0x0F0F0F0Fu) : ((raw0_hi >> 4) & 0x0F0F0F0Fu);
                const uint32_t a1 = (hi == 0) ? (raw1_lo & 0x0F0F0F0Fu) : ((raw1_lo >> 4) & 0x0F0F0F0Fu);
                const uint32_t a3 = (hi == 0) ? (raw1_hi & 0x0F0F0F0Fu) : ((raw1_hi >> 4) & 0x0F0F0F0Fu);

#pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    const int tg = nt * 8 + g;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                    int32_t d[4] = {0, 0, 0, 0};
                    mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1);

                    const int t0 = nt * 8 + 2 * tig;
                    const int t1 = t0 + 1;
                    acc[nt][0] = fmaf(ds0 * s_dx[t0 * 8 + b], (float) d[0], fmaf(-ms0, s_xs[t0 * 8 + b], acc[nt][0]));
                    acc[nt][1] = fmaf(ds0 * s_dx[t1 * 8 + b], (float) d[1], fmaf(-ms0, s_xs[t1 * 8 + b], acc[nt][1]));
                    acc[nt][2] = fmaf(ds1 * s_dx[t0 * 8 + b], (float) d[2], fmaf(-ms1, s_xs[t0 * 8 + b], acc[nt][2]));
                    acc[nt][3] = fmaf(ds1 * s_dx[t1 * 8 + b], (float) d[3], fmaf(-ms1, s_xs[t1 * 8 + b], acc[nt][3]));
                }
            }
        }
    } else if constexpr (GTYPE == kTypeQ5_K) {
        const int sb_idx = k_blk0 >> 3;
        const uint8_t* blk0 = w0 + (size_t) sb_idx * 176u;
        const uint8_t* blk1 = w1 + (size_t) sb_idx * 176u;
        const half2 dm0 = v0 ? *reinterpret_cast<const half2*>(blk0) : __floats2half2_rn(0.0f, 0.0f);
        const half2 dm1 = v1 ? *reinterpret_cast<const half2*>(blk1) : __floats2half2_rn(0.0f, 0.0f);
        const float d0_all = __low2float(dm0), m0_all = __high2float(dm0);
        const float d1_all = __low2float(dm1), m1_all = __high2float(dm1);
        const uint8_t* sc0_p = blk0 + 4;
        const uint8_t* sc1_p = blk1 + 4;
        const uint32_t qh0_lo = v0 ? __ldg(reinterpret_cast<const uint32_t*>(blk0 + 16 + tig * 4)) : 0u;
        const uint32_t qh0_hi = v0 ? __ldg(reinterpret_cast<const uint32_t*>(blk0 + 16 + 16 + tig * 4)) : 0u;
        const uint32_t qh1_lo = v1 ? __ldg(reinterpret_cast<const uint32_t*>(blk1 + 16 + tig * 4)) : 0u;
        const uint32_t qh1_hi = v1 ? __ldg(reinterpret_cast<const uint32_t*>(blk1 + 16 + 16 + tig * 4)) : 0u;
        const uint8_t* qs0_p = blk0 + 48;
        const uint8_t* qs1_p = blk1 + 48;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t raw0_lo = v0 ? __ldg(reinterpret_cast<const uint32_t*>(qs0_p + 32 * j + tig * 4)) : 0u;
            const uint32_t raw0_hi = v0 ? __ldg(reinterpret_cast<const uint32_t*>(qs0_p + 32 * j + 16 + tig * 4)) : 0u;
            const uint32_t raw1_lo = v1 ? __ldg(reinterpret_cast<const uint32_t*>(qs1_p + 32 * j + tig * 4)) : 0u;
            const uint32_t raw1_hi = v1 ? __ldg(reinterpret_cast<const uint32_t*>(qs1_p + 32 * j + 16 + tig * 4)) : 0u;

#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                const int b = 2 * j + hi;
                float s0 = 0.0f, mn0 = 0.0f, s1 = 0.0f, mn1 = 0.0f;
                if (v0) decode_k4_scale_min(sc0_p, b, s0, mn0);
                if (v1) decode_k4_scale_min(sc1_p, b, s1, mn1);
                const float ds0 = d0_all * s0, ms0 = m0_all * mn0;
                const float ds1 = d1_all * s1, ms1 = m1_all * mn1;

                const uint32_t q4_0l = (hi == 0) ? (raw0_lo & 0x0F0F0F0Fu) : ((raw0_lo >> 4) & 0x0F0F0F0Fu);
                const uint32_t q4_0h = (hi == 0) ? (raw0_hi & 0x0F0F0F0Fu) : ((raw0_hi >> 4) & 0x0F0F0F0Fu);
                const uint32_t q4_1l = (hi == 0) ? (raw1_lo & 0x0F0F0F0Fu) : ((raw1_lo >> 4) & 0x0F0F0F0Fu);
                const uint32_t q4_1h = (hi == 0) ? (raw1_hi & 0x0F0F0F0Fu) : ((raw1_hi >> 4) & 0x0F0F0F0Fu);

                const uint32_t a0 = q4_0l | (((qh0_lo >> b) & 0x01010101u) << 4);
                const uint32_t a2 = q4_0h | (((qh0_hi >> b) & 0x01010101u) << 4);
                const uint32_t a1 = q4_1l | (((qh1_lo >> b) & 0x01010101u) << 4);
                const uint32_t a3 = q4_1h | (((qh1_hi >> b) & 0x01010101u) << 4);

#pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    const int tg = nt * 8 + g;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                    int32_t d[4] = {0, 0, 0, 0};
                    mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1);

                    const int t0 = nt * 8 + 2 * tig;
                    const int t1 = t0 + 1;
                    acc[nt][0] = fmaf(ds0 * s_dx[t0 * 8 + b], (float) d[0], fmaf(-ms0, s_xs[t0 * 8 + b], acc[nt][0]));
                    acc[nt][1] = fmaf(ds0 * s_dx[t1 * 8 + b], (float) d[1], fmaf(-ms0, s_xs[t1 * 8 + b], acc[nt][1]));
                    acc[nt][2] = fmaf(ds1 * s_dx[t0 * 8 + b], (float) d[2], fmaf(-ms1, s_xs[t0 * 8 + b], acc[nt][2]));
                    acc[nt][3] = fmaf(ds1 * s_dx[t1 * 8 + b], (float) d[3], fmaf(-ms1, s_xs[t1 * 8 + b], acc[nt][3]));
                }
            }
        }
    } else if constexpr (GTYPE == kTypeQ6_K) {
        const int sb_idx = k_blk0 >> 3;
        const uint8_t* blk0 = w0 + (size_t) sb_idx * 210u;
        const uint8_t* blk1 = w1 + (size_t) sb_idx * 210u;
        const float d0_all = v0 ? ld_fp16_b2(blk0 + 208) : 0.0f;
        const float d1_all = v1 ? ld_fp16_b2(blk1 + 208) : 0.0f;
        const int8_t* sc0_p = reinterpret_cast<const int8_t*>(blk0 + 192);
        const int8_t* sc1_p = reinterpret_cast<const int8_t*>(blk1 + 192);

#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int half16 = 0; half16 < 2; ++half16) {
                const int l0 = half16 * 16;
                const uint32_t qh0 = v0 ? ld32_b2(blk0 + 128 + 32 * n + l0 + tig * 4) : 0u;
                const uint32_t qh1 = v1 ? ld32_b2(blk1 + 128 + 32 * n + l0 + tig * 4) : 0u;
                const uint32_t ql0_a = v0 ? ld32_b2(blk0 + 64 * n + l0 + tig * 4) : 0u;
                const uint32_t ql0_b = v0 ? ld32_b2(blk0 + 64 * n + 32 + l0 + tig * 4) : 0u;
                const uint32_t ql1_a = v1 ? ld32_b2(blk1 + 64 * n + l0 + tig * 4) : 0u;
                const uint32_t ql1_b = v1 ? ld32_b2(blk1 + 64 * n + 32 + l0 + tig * 4) : 0u;

#pragma unroll
                for (int qid = 0; qid < 4; ++qid) {
                    const int sub = 8 * n + 2 * qid + half16;
                    const int b32 = sub >> 1;
                    const float dw0 = v0 ? (d0_all * (float) __ldg(sc0_p + sub)) : 0.0f;
                    const float dw1 = v1 ? (d1_all * (float) __ldg(sc1_p + sub)) : 0.0f;

                    const uint32_t src0 = (qid & 1) ? ql0_b : ql0_a;
                    const uint32_t src1 = (qid & 1) ? ql1_b : ql1_a;
                    const uint32_t lo0 = (qid < 2) ? (src0 & 0x0F0F0F0Fu) : ((src0 >> 4) & 0x0F0F0F0Fu);
                    const uint32_t lo1 = (qid < 2) ? (src1 & 0x0F0F0F0Fu) : ((src1 >> 4) & 0x0F0F0F0Fu);
                    const uint32_t a0 = __vsub4(lo0 | (((qh0 >> (2 * qid)) & 0x03030303u) << 4), 0x20202020u);
                    const uint32_t a1 = __vsub4(lo1 | (((qh1 >> (2 * qid)) & 0x03030303u) << 4), 0x20202020u);

#pragma unroll
                    for (int nt = 0; nt < NTILES; ++nt) {
                        const int tg = nt * 8 + g;
                        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                            s_xq + (tg * 8 + b32) * 32 + half16 * 16 + tig * 4);
                        int32_t d[4] = {0, 0, 0, 0};
                        mma_s8_m16n8k16(d, a0, a1, b0);

                        const int t0 = nt * 8 + 2 * tig;
                        const int t1 = t0 + 1;
                        acc[nt][0] = fmaf(dw0 * s_dx[t0 * 8 + b32], (float) d[0], acc[nt][0]);
                        acc[nt][1] = fmaf(dw0 * s_dx[t1 * 8 + b32], (float) d[1], acc[nt][1]);
                        acc[nt][2] = fmaf(dw1 * s_dx[t0 * 8 + b32], (float) d[2], acc[nt][2]);
                        acc[nt][3] = fmaf(dw1 * s_dx[t1 * 8 + b32], (float) d[3], acc[nt][3]);
                    }
                }
            }
        }
    } else if constexpr (GTYPE == kTypeIQ4_XS) {
        const int sb_idx = k_blk0 >> 3;
        const uint8_t* blk0 = w0 + (size_t) sb_idx * 136u;
        const uint8_t* blk1 = w1 + (size_t) sb_idx * 136u;
        const float d0_all = v0 ? ld_fp16_b2(blk0) : 0.0f;
        const float d1_all = v1 ? ld_fp16_b2(blk1) : 0.0f;
        const uint16_t sh0 = v0 ? __ldg(reinterpret_cast<const uint16_t*>(blk0 + 2)) : 0u;
        const uint16_t sh1 = v1 ? __ldg(reinterpret_cast<const uint16_t*>(blk1 + 2)) : 0u;
        const uint32_t sl0 = v0 ? ld32_b2(blk0 + 4) : 0u;
        const uint32_t sl1 = v1 ? ld32_b2(blk1 + 4) : 0u;

#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int ls0 = (int) (((sl0 >> (4 * b)) & 0x0Fu) | (((sh0 >> (2 * b)) & 3u) << 4)) - 32;
            const int ls1 = (int) (((sl1 >> (4 * b)) & 0x0Fu) | (((sh1 >> (2 * b)) & 3u) << 4)) - 32;
            const float dw0 = v0 ? (d0_all * (float) ls0) : 0.0f;
            const float dw1 = v1 ? (d1_all * (float) ls1) : 0.0f;

            const uint32_t raw0 = v0 ? ld32_b2(blk0 + 8 + b * 16 + tig * 4) : 0u;
            const uint32_t raw1 = v1 ? ld32_b2(blk1 + 8 + b * 16 + tig * 4) : 0u;
            const uint2 w0_lh = iq4nl_lut8(raw0);
            const uint2 w1_lh = iq4nl_lut8(raw1);

#pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                const int tg = nt * 8 + g;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                int32_t d[4] = {0, 0, 0, 0};
                mma_s8_m16n8k32(d, w0_lh.x, w1_lh.x, w0_lh.y, w1_lh.y, b0, b1);

                const int t0 = nt * 8 + 2 * tig;
                const int t1 = t0 + 1;
                acc[nt][0] = fmaf(dw0 * s_dx[t0 * 8 + b], (float) d[0], acc[nt][0]);
                acc[nt][1] = fmaf(dw0 * s_dx[t1 * 8 + b], (float) d[1], acc[nt][1]);
                acc[nt][2] = fmaf(dw1 * s_dx[t0 * 8 + b], (float) d[2], acc[nt][2]);
                acc[nt][3] = fmaf(dw1 * s_dx[t1 * 8 + b], (float) d[3], acc[nt][3]);
            }
        }
    }
#else
    (void) w0; (void) w1; (void) v0; (void) v1; (void) nb_here; (void) k_blk0;
    (void) s_xq; (void) s_dx; (void) s_xs; (void) g; (void) tig; (void) acc;
#endif
}

__device__ __forceinline__ size_t row_bytes_of(int ggml_type, int n_in) {
    switch (ggml_type) {
        case kTypeQ8_0: return (size_t) (n_in / 32) * 34u;
        case kTypeQ4_K: return (size_t) (n_in / 256) * 144u;
        case kTypeQ5_K: return (size_t) (n_in / 256) * 176u;
        case kTypeQ6_K: return (size_t) (n_in / 256) * 210u;
        case kTypeIQ4_XS: return (size_t) (n_in / 256) * 136u;
        default: return 0;
    }
}

// Byte count of one K<=256 chunk for a single weight row, padded up to a multiple of 4 bytes.
template <int GTYPE>
__device__ __host__ constexpr int chunk_u32_of() {
    if constexpr (GTYPE == kTypeQ8_0) return 68;     // 8 * 34 = 272 B
    else if constexpr (GTYPE == kTypeQ4_K) return 36; // 144 B
    else if constexpr (GTYPE == kTypeQ5_K) return 44; // 176 B
    else if constexpr (GTYPE == kTypeQ6_K) return 53; // 210 B -> 212 B
    else return 34;                                   // IQ4_XS: 136 B
}

// 100%-coalesced warp-wide weight stager: all 32 lanes read contiguous words along each of 16 rows in turn.
template <int GTYPE>
__device__ __forceinline__ void stage_dense_rows16(
    const uint8_t* __restrict__ weights, int r_base, int n_out, size_t rbytes,
    int k_blk0, int nb_here, int lane, uint32_t (&s_w)[16][chunk_u32_of<GTYPE>()]) {
    constexpr int MAX_U32 = chunk_u32_of<GTYPE>();
    const size_t blk_off = (GTYPE == kTypeQ8_0)
                               ? (size_t) k_blk0 * 34u
                               : (size_t) (k_blk0 >> 3) * (GTYPE == kTypeQ4_K ? 144u :
                                                           GTYPE == kTypeQ5_K ? 176u :
                                                           GTYPE == kTypeQ6_K ? 210u : 136u);
    const int n_bytes = (GTYPE == kTypeQ8_0)
                            ? nb_here * 34
                            : (GTYPE == kTypeQ4_K ? 144 :
                               GTYPE == kTypeQ5_K ? 176 :
                               GTYPE == kTypeQ6_K ? 210 : 136);
    const int full_u32 = n_bytes >> 2;
    const bool has_tail2 = (n_bytes & 2) != 0;

#pragma unroll
    for (int rr = 0; rr < 16; ++rr) {
        if (r_base + rr < n_out) {
            const uint8_t* p = weights + (size_t) (r_base + rr) * rbytes + blk_off;
            if constexpr (GTYPE == kTypeQ4_K || GTYPE == kTypeQ5_K) {
                const uint32_t* p32 = reinterpret_cast<const uint32_t*>(p);
                for (int u = lane; u < full_u32; u += 32) s_w[rr][u] = __ldg(p32 + u);
            } else {
                for (int u = lane; u < full_u32; u += 32) {
                    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p + u * 4);
                    s_w[rr][u] = (uint32_t) __ldg(p16) | ((uint32_t) __ldg(p16 + 1) << 16);
                }
                if (has_tail2 && lane == 0) {
                    s_w[rr][full_u32] = (uint32_t) __ldg(reinterpret_cast<const uint16_t*>(p + full_u32 * 4));
                }
            }
        } else {
            for (int u = lane; u < MAX_U32; u += 32) s_w[rr][u] = 0u;
        }
    }
    __syncwarp();
}

// Fast shared-memory byte/half/u32 readers for staged rows in s_w (already 4-byte aligned at row start!).
__device__ __forceinline__ uint32_t smem_ld32_b2(const uint8_t* p) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t) p16[0] | ((uint32_t) p16[1] << 16);
}

__device__ __forceinline__ float smem_ld_fp16_b2(const uint8_t* p) {
    return __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(p)));
}

template <int GTYPE, int NTILES>
__device__ __forceinline__ void eval_kchunk_tc_smem(
    const uint32_t (&s_w)[16][chunk_u32_of<GTYPE>()],
    bool v0, bool v1, int nb_here,
    const int8_t* __restrict__ s_xq, const float* __restrict__ s_dx, const float* __restrict__ s_xs,
    int g, int tig, float (&acc)[NTILES][4]) {
#if STRATA_DENSE_I8_SM80
    const uint8_t* blk0 = reinterpret_cast<const uint8_t*>(s_w[g]);
    const uint8_t* blk1 = reinterpret_cast<const uint8_t*>(s_w[8 + g]);

    if constexpr (GTYPE == kTypeQ8_0) {
        for (int b = 0; b < nb_here; ++b) {
            const uint8_t* p0 = blk0 + b * 34;
            const uint8_t* p1 = blk1 + b * 34;
            const float dw0 = v0 ? smem_ld_fp16_b2(p0) : 0.0f;
            const float dw1 = v1 ? smem_ld_fp16_b2(p1) : 0.0f;
            const uint32_t a0 = v0 ? smem_ld32_b2(p0 + 2 + tig * 4) : 0u;
            const uint32_t a2 = v0 ? smem_ld32_b2(p0 + 18 + tig * 4) : 0u;
            const uint32_t a1 = v1 ? smem_ld32_b2(p1 + 2 + tig * 4) : 0u;
            const uint32_t a3 = v1 ? smem_ld32_b2(p1 + 18 + tig * 4) : 0u;

#pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                const int tg = nt * 8 + g;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                int32_t d[4] = {0, 0, 0, 0};
                mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1);

                const int t0 = nt * 8 + 2 * tig;
                const int t1 = t0 + 1;
                const float dx0 = s_dx[t0 * 8 + b];
                const float dx1 = s_dx[t1 * 8 + b];
                acc[nt][0] = fmaf(dw0 * dx0, (float) d[0], acc[nt][0]);
                acc[nt][1] = fmaf(dw0 * dx1, (float) d[1], acc[nt][1]);
                acc[nt][2] = fmaf(dw1 * dx0, (float) d[2], acc[nt][2]);
                acc[nt][3] = fmaf(dw1 * dx1, (float) d[3], acc[nt][3]);
            }
        }
    } else if constexpr (GTYPE == kTypeQ4_K) {
        const half2 dm0 = v0 ? *reinterpret_cast<const half2*>(blk0) : __floats2half2_rn(0.0f, 0.0f);
        const half2 dm1 = v1 ? *reinterpret_cast<const half2*>(blk1) : __floats2half2_rn(0.0f, 0.0f);
        const float d0_all = __low2float(dm0), m0_all = __high2float(dm0);
        const float d1_all = __low2float(dm1), m1_all = __high2float(dm1);
        const uint8_t* sc0_p = blk0 + 4;
        const uint8_t* sc1_p = blk1 + 4;
        const uint8_t* qs0_p = blk0 + 16;
        const uint8_t* qs1_p = blk1 + 16;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t raw0_lo = v0 ? *reinterpret_cast<const uint32_t*>(qs0_p + 32 * j + tig * 4) : 0u;
            const uint32_t raw0_hi = v0 ? *reinterpret_cast<const uint32_t*>(qs0_p + 32 * j + 16 + tig * 4) : 0u;
            const uint32_t raw1_lo = v1 ? *reinterpret_cast<const uint32_t*>(qs1_p + 32 * j + tig * 4) : 0u;
            const uint32_t raw1_hi = v1 ? *reinterpret_cast<const uint32_t*>(qs1_p + 32 * j + 16 + tig * 4) : 0u;

#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                const int b = 2 * j + hi;
                float s0 = 0.0f, mn0 = 0.0f, s1 = 0.0f, mn1 = 0.0f;
                if (v0) decode_k4_scale_min(sc0_p, b, s0, mn0);
                if (v1) decode_k4_scale_min(sc1_p, b, s1, mn1);
                const float ds0 = d0_all * s0, ms0 = m0_all * mn0;
                const float ds1 = d1_all * s1, ms1 = m1_all * mn1;

                const uint32_t a0 = (hi == 0) ? (raw0_lo & 0x0F0F0F0Fu) : ((raw0_lo >> 4) & 0x0F0F0F0Fu);
                const uint32_t a2 = (hi == 0) ? (raw0_hi & 0x0F0F0F0Fu) : ((raw0_hi >> 4) & 0x0F0F0F0Fu);
                const uint32_t a1 = (hi == 0) ? (raw1_lo & 0x0F0F0F0Fu) : ((raw1_lo >> 4) & 0x0F0F0F0Fu);
                const uint32_t a3 = (hi == 0) ? (raw1_hi & 0x0F0F0F0Fu) : ((raw1_hi >> 4) & 0x0F0F0F0Fu);

#pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    const int tg = nt * 8 + g;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                    int32_t d[4] = {0, 0, 0, 0};
                    mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1);

                    const int t0 = nt * 8 + 2 * tig;
                    const int t1 = t0 + 1;
                    acc[nt][0] = fmaf(ds0 * s_dx[t0 * 8 + b], (float) d[0], fmaf(-ms0, s_xs[t0 * 8 + b], acc[nt][0]));
                    acc[nt][1] = fmaf(ds0 * s_dx[t1 * 8 + b], (float) d[1], fmaf(-ms0, s_xs[t1 * 8 + b], acc[nt][1]));
                    acc[nt][2] = fmaf(ds1 * s_dx[t0 * 8 + b], (float) d[2], fmaf(-ms1, s_xs[t0 * 8 + b], acc[nt][2]));
                    acc[nt][3] = fmaf(ds1 * s_dx[t1 * 8 + b], (float) d[3], fmaf(-ms1, s_xs[t1 * 8 + b], acc[nt][3]));
                }
            }
        }
    } else if constexpr (GTYPE == kTypeQ5_K) {
        const half2 dm0 = v0 ? *reinterpret_cast<const half2*>(blk0) : __floats2half2_rn(0.0f, 0.0f);
        const half2 dm1 = v1 ? *reinterpret_cast<const half2*>(blk1) : __floats2half2_rn(0.0f, 0.0f);
        const float d0_all = __low2float(dm0), m0_all = __high2float(dm0);
        const float d1_all = __low2float(dm1), m1_all = __high2float(dm1);
        const uint8_t* sc0_p = blk0 + 4;
        const uint8_t* sc1_p = blk1 + 4;
        const uint32_t qh0_lo = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 16 + tig * 4) : 0u;
        const uint32_t qh0_hi = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 32 + tig * 4) : 0u;
        const uint32_t qh1_lo = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 16 + tig * 4) : 0u;
        const uint32_t qh1_hi = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 32 + tig * 4) : 0u;
        const uint8_t* qs0_p = blk0 + 48;
        const uint8_t* qs1_p = blk1 + 48;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t raw0_lo = v0 ? *reinterpret_cast<const uint32_t*>(qs0_p + 32 * j + tig * 4) : 0u;
            const uint32_t raw0_hi = v0 ? *reinterpret_cast<const uint32_t*>(qs0_p + 32 * j + 16 + tig * 4) : 0u;
            const uint32_t raw1_lo = v1 ? *reinterpret_cast<const uint32_t*>(qs1_p + 32 * j + tig * 4) : 0u;
            const uint32_t raw1_hi = v1 ? *reinterpret_cast<const uint32_t*>(qs1_p + 32 * j + 16 + tig * 4) : 0u;

#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                const int b = 2 * j + hi;
                float s0 = 0.0f, mn0 = 0.0f, s1 = 0.0f, mn1 = 0.0f;
                if (v0) decode_k4_scale_min(sc0_p, b, s0, mn0);
                if (v1) decode_k4_scale_min(sc1_p, b, s1, mn1);
                const float ds0 = d0_all * s0, ms0 = m0_all * mn0;
                const float ds1 = d1_all * s1, ms1 = m1_all * mn1;

                const uint32_t q4_0l = (hi == 0) ? (raw0_lo & 0x0F0F0F0Fu) : ((raw0_lo >> 4) & 0x0F0F0F0Fu);
                const uint32_t q4_0h = (hi == 0) ? (raw0_hi & 0x0F0F0F0Fu) : ((raw0_hi >> 4) & 0x0F0F0F0Fu);
                const uint32_t q4_1l = (hi == 0) ? (raw1_lo & 0x0F0F0F0Fu) : ((raw1_lo >> 4) & 0x0F0F0F0Fu);
                const uint32_t q4_1h = (hi == 0) ? (raw1_hi & 0x0F0F0F0Fu) : ((raw1_hi >> 4) & 0x0F0F0F0Fu);

                const uint32_t a0 = q4_0l | (((qh0_lo >> b) & 0x01010101u) << 4);
                const uint32_t a2 = q4_0h | (((qh0_hi >> b) & 0x01010101u) << 4);
                const uint32_t a1 = q4_1l | (((qh1_lo >> b) & 0x01010101u) << 4);
                const uint32_t a3 = q4_1h | (((qh1_hi >> b) & 0x01010101u) << 4);

#pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    const int tg = nt * 8 + g;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                    int32_t d[4] = {0, 0, 0, 0};
                    mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1);

                    const int t0 = nt * 8 + 2 * tig;
                    const int t1 = t0 + 1;
                    acc[nt][0] = fmaf(ds0 * s_dx[t0 * 8 + b], (float) d[0], fmaf(-ms0, s_xs[t0 * 8 + b], acc[nt][0]));
                    acc[nt][1] = fmaf(ds0 * s_dx[t1 * 8 + b], (float) d[1], fmaf(-ms0, s_xs[t1 * 8 + b], acc[nt][1]));
                    acc[nt][2] = fmaf(ds1 * s_dx[t0 * 8 + b], (float) d[2], fmaf(-ms1, s_xs[t0 * 8 + b], acc[nt][2]));
                    acc[nt][3] = fmaf(ds1 * s_dx[t1 * 8 + b], (float) d[3], fmaf(-ms1, s_xs[t1 * 8 + b], acc[nt][3]));
                }
            }
        }
    } else if constexpr (GTYPE == kTypeQ6_K) {
        const float d0_all = v0 ? smem_ld_fp16_b2(blk0 + 208) : 0.0f;
        const float d1_all = v1 ? smem_ld_fp16_b2(blk1 + 208) : 0.0f;
        const int8_t* sc0_p = reinterpret_cast<const int8_t*>(blk0 + 192);
        const int8_t* sc1_p = reinterpret_cast<const int8_t*>(blk1 + 192);

#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int half16 = 0; half16 < 2; ++half16) {
                const int l0 = half16 * 16;
                const uint32_t qh0 = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 128 + 32 * n + l0 + tig * 4) : 0u;
                const uint32_t qh1 = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 128 + 32 * n + l0 + tig * 4) : 0u;
                const uint32_t ql0_a = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 64 * n + l0 + tig * 4) : 0u;
                const uint32_t ql0_b = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 64 * n + 32 + l0 + tig * 4) : 0u;
                const uint32_t ql1_a = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 64 * n + l0 + tig * 4) : 0u;
                const uint32_t ql1_b = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 64 * n + 32 + l0 + tig * 4) : 0u;

#pragma unroll
                for (int qid = 0; qid < 4; ++qid) {
                    const int sub = 8 * n + 2 * qid + half16;
                    const int b32 = sub >> 1;
                    const float dw0 = v0 ? (d0_all * (float) sc0_p[sub]) : 0.0f;
                    const float dw1 = v1 ? (d1_all * (float) sc1_p[sub]) : 0.0f;

                    const uint32_t src0 = (qid & 1) ? ql0_b : ql0_a;
                    const uint32_t src1 = (qid & 1) ? ql1_b : ql1_a;
                    const uint32_t lo0 = (qid < 2) ? (src0 & 0x0F0F0F0Fu) : ((src0 >> 4) & 0x0F0F0F0Fu);
                    const uint32_t lo1 = (qid < 2) ? (src1 & 0x0F0F0F0Fu) : ((src1 >> 4) & 0x0F0F0F0Fu);
                    const uint32_t a0 = __vsub4(lo0 | (((qh0 >> (2 * qid)) & 0x03030303u) << 4), 0x20202020u);
                    const uint32_t a1 = __vsub4(lo1 | (((qh1 >> (2 * qid)) & 0x03030303u) << 4), 0x20202020u);

#pragma unroll
                    for (int nt = 0; nt < NTILES; ++nt) {
                        const int tg = nt * 8 + g;
                        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                            s_xq + (tg * 8 + b32) * 32 + half16 * 16 + tig * 4);
                        int32_t d[4] = {0, 0, 0, 0};
                        mma_s8_m16n8k16(d, a0, a1, b0);

                        const int t0 = nt * 8 + 2 * tig;
                        const int t1 = t0 + 1;
                        acc[nt][0] = fmaf(dw0 * s_dx[t0 * 8 + b32], (float) d[0], acc[nt][0]);
                        acc[nt][1] = fmaf(dw0 * s_dx[t1 * 8 + b32], (float) d[1], acc[nt][1]);
                        acc[nt][2] = fmaf(dw1 * s_dx[t0 * 8 + b32], (float) d[2], acc[nt][2]);
                        acc[nt][3] = fmaf(dw1 * s_dx[t1 * 8 + b32], (float) d[3], acc[nt][3]);
                    }
                }
            }
        }
    } else if constexpr (GTYPE == kTypeIQ4_XS) {
        const float d0_all = v0 ? smem_ld_fp16_b2(blk0) : 0.0f;
        const float d1_all = v1 ? smem_ld_fp16_b2(blk1) : 0.0f;
        const uint16_t sh0 = v0 ? *reinterpret_cast<const uint16_t*>(blk0 + 2) : 0u;
        const uint16_t sh1 = v1 ? *reinterpret_cast<const uint16_t*>(blk1 + 2) : 0u;
        const uint32_t sl0 = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 4) : 0u;
        const uint32_t sl1 = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 4) : 0u;

#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int ls0 = (int) (((sl0 >> (4 * b)) & 0x0Fu) | (((sh0 >> (2 * b)) & 3u) << 4)) - 32;
            const int ls1 = (int) (((sl1 >> (4 * b)) & 0x0Fu) | (((sh1 >> (2 * b)) & 3u) << 4)) - 32;
            const float dw0 = v0 ? (d0_all * (float) ls0) : 0.0f;
            const float dw1 = v1 ? (d1_all * (float) ls1) : 0.0f;

            const uint32_t raw0 = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 8 + b * 16 + tig * 4) : 0u;
            const uint32_t raw1 = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 8 + b * 16 + tig * 4) : 0u;
            const uint2 w0_lh = iq4nl_lut8(raw0);
            const uint2 w1_lh = iq4nl_lut8(raw1);

#pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                const int tg = nt * 8 + g;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                int32_t d[4] = {0, 0, 0, 0};
                mma_s8_m16n8k32(d, w0_lh.x, w1_lh.x, w0_lh.y, w1_lh.y, b0, b1);

                const int t0 = nt * 8 + 2 * tig;
                const int t1 = t0 + 1;
                acc[nt][0] = fmaf(dw0 * s_dx[t0 * 8 + b], (float) d[0], acc[nt][0]);
                acc[nt][1] = fmaf(dw0 * s_dx[t1 * 8 + b], (float) d[1], acc[nt][1]);
                acc[nt][2] = fmaf(dw1 * s_dx[t0 * 8 + b], (float) d[2], acc[nt][2]);
                acc[nt][3] = fmaf(dw1 * s_dx[t1 * 8 + b], (float) d[3], acc[nt][3]);
            }
        }
    }
#else
    (void) s_w; (void) v0; (void) v1; (void) nb_here;
    (void) s_xq; (void) s_dx; (void) s_xs; (void) g; (void) tig; (void) acc;
#endif
}

// ---- High-Occupancy 8-Warp K-Split Decode / Verify Kernel: 16 output rows x (NTILES*8) tokens per 256-thread CTA ----
// All 8 warps cooperate on ONE 16-row tile per K=256 superblock:
//   - Load phase: each warp loads ONLY 2 rows coalesced into shared s_w[16][U32] (~3.4 KB total for the whole CTA!),
//     while cooperatively loading T*8 activation blocks into s_xq (~2.0 KB). Total smem ~6-9 KB -> 6+ CTAs/SM = 48 warps/SM!
//   - Compute phase: warp w (0..7) owns sub-block b = w (32 elements of the 256-elem superblock), executing just ONE
//     mma.m16n8k32.s8 (or two m16n8k16 for Q6_K) per 8-token tile!
//   - Epilogue: single warp-reduction across the 8 warps at the very end of the K loop!
template <int GTYPE, int NTILES>
__global__ void __launch_bounds__(256) dense_i8_tc_decode_ksplit_kernel(
    const uint8_t* __restrict__ weights, const Q81Block* __restrict__ xq, float* __restrict__ y,
    int n_in, int n_out, int T) {
#if STRATA_DENSE_I8_SM80
    constexpr int MAX_T = NTILES * 8;
    constexpr int MAX_U32 = chunk_u32_of<GTYPE>();
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;   // 0..7 -> sub-block b = warp within each K=256 chunk!
    const int lane = tid & 31;
    const int g = lane >> 2;     // 0..7
    const int tig = lane & 3;    // 0..3

    const int r_base = (int) blockIdx.x * 16;
    const int r0 = r_base + g;
    const int r1 = r_base + 8 + g;
    const bool v0 = r0 < n_out;
    const bool v1 = r1 < n_out;

    const int nb32 = n_in >> 5;
    const size_t rbytes = row_bytes_of(GTYPE, n_in);
    constexpr size_t kChunkBytes = (GTYPE == kTypeQ8_0) ? 272u :
                                   (GTYPE == kTypeQ4_K) ? 144u :
                                   (GTYPE == kTypeQ5_K) ? 176u :
                                   (GTYPE == kTypeQ6_K) ? 210u : 136u;

    __shared__ alignas(16) int8_t s_xq[MAX_T * 256];
    __shared__ float s_dx[MAX_T * 8];
    __shared__ float s_xs[MAX_T * 8];
    __shared__ alignas(16) uint32_t s_w[16][MAX_U32];

    float acc[NTILES][4];
#pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {
        acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f;
    }

    const uint8_t* blk0 = reinterpret_cast<const uint8_t*>(s_w[g]);
    const uint8_t* blk1 = reinterpret_cast<const uint8_t*>(s_w[8 + g]);

    for (int k_blk0 = 0; k_blk0 < nb32; k_blk0 += 8) {
        const int nb_here = min(8, nb32 - k_blk0);
        const int n_bytes = (GTYPE == kTypeQ8_0) ? (nb_here * 34) : (int) kChunkBytes;
        const int full_u32 = n_bytes >> 2;
        const bool has_tail2 = (n_bytes & 2) != 0;
        const size_t blk_off = (GTYPE == kTypeQ8_0) ? ((size_t) k_blk0 * 34u)
                                                    : ((size_t) (k_blk0 >> 3) * kChunkBytes);

        // 256 threads load <= 128 activation blocks (at most 1 block per 2 threads -> 1 cycle!).
        if (tid < MAX_T * 8) {
            const int t = tid >> 3;
            const int b = tid & 7;
            if (t < T && b < nb_here) {
                const Q81Block& blk = xq[(size_t) t * nb32 + k_blk0 + b];
                const float dx = __low2float(blk.ds);
                s_dx[tid] = dx;
                const uint32_t* src32 = reinterpret_cast<const uint32_t*>(blk.qs);
                const uint32_t x0 = src32[0], x1 = src32[1], x2 = src32[2], x3 = src32[3];
                const uint32_t x4 = src32[4], x5 = src32[5], x6 = src32[6], x7 = src32[7];
                if constexpr (GTYPE == kTypeQ4_K || GTYPE == kTypeQ5_K) {
                    int qsum = __dp4a(0x01010101, (int) x0, 0);
                    qsum = __dp4a(0x01010101, (int) x1, qsum);
                    qsum = __dp4a(0x01010101, (int) x2, qsum);
                    qsum = __dp4a(0x01010101, (int) x3, qsum);
                    qsum = __dp4a(0x01010101, (int) x4, qsum);
                    qsum = __dp4a(0x01010101, (int) x5, qsum);
                    qsum = __dp4a(0x01010101, (int) x6, qsum);
                    qsum = __dp4a(0x01010101, (int) x7, qsum);
                    s_xs[tid] = dx * (float) qsum;
                }
                uint4* dst4 = reinterpret_cast<uint4*>(s_xq + tid * 32);
                dst4[0] = make_uint4(x0, x1, x2, x3);
                dst4[1] = make_uint4(x4, x5, x6, x7);
            } else {
                s_dx[tid] = 0.0f;
                if constexpr (GTYPE == kTypeQ4_K || GTYPE == kTypeQ5_K) s_xs[tid] = 0.0f;
                uint4* dst4 = reinterpret_cast<uint4*>(s_xq + tid * 32);
                dst4[0] = make_uint4(0u, 0u, 0u, 0u);
                dst4[1] = make_uint4(0u, 0u, 0u, 0u);
            }
        }

        // 8 warps cooperatively load 16 weight rows (ONLY 2 rows per warp, 100% coalesced 128B transactions!).
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int lr = warp * 2 + rr;
            if (r_base + lr < n_out) {
                const uint8_t* p = weights + (size_t) (r_base + lr) * rbytes + blk_off;
                if constexpr (GTYPE == kTypeQ4_K || GTYPE == kTypeQ5_K) {
                    const uint32_t* p32 = reinterpret_cast<const uint32_t*>(p);
                    for (int u = lane; u < full_u32; u += 32) s_w[lr][u] = __ldg(p32 + u);
                } else {
                    const bool al4 = (((uintptr_t) p & 3u) == 0u);
                    for (int u = lane; u < full_u32; u += 32) {
                        if (al4) {
                            s_w[lr][u] = __ldg(reinterpret_cast<const uint32_t*>(p) + u);
                        } else {
                            const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p + u * 4);
                            s_w[lr][u] = (uint32_t) __ldg(p16) | ((uint32_t) __ldg(p16 + 1) << 16);
                        }
                    }
                    if (has_tail2 && lane == 0) {
                        s_w[lr][full_u32] = (uint32_t) __ldg(reinterpret_cast<const uint16_t*>(p + full_u32 * 4));
                    }
                }
            }
        }
        __syncthreads();

        // Each warp w (0..7) evaluates ONLY sub-block b = warp (32 elements)!
        const int b = warp;
        if (b < nb_here) {
            if constexpr (GTYPE == kTypeQ8_0) {
                const uint8_t* p0 = blk0 + b * 34;
                const uint8_t* p1 = blk1 + b * 34;
                const float dw0 = v0 ? smem_ld_fp16_b2(p0) : 0.0f;
                const float dw1 = v1 ? smem_ld_fp16_b2(p1) : 0.0f;
                const uint32_t a0 = v0 ? smem_ld32_b2(p0 + 2 + tig * 4) : 0u;
                const uint32_t a2 = v0 ? smem_ld32_b2(p0 + 18 + tig * 4) : 0u;
                const uint32_t a1 = v1 ? smem_ld32_b2(p1 + 2 + tig * 4) : 0u;
                const uint32_t a3 = v1 ? smem_ld32_b2(p1 + 18 + tig * 4) : 0u;
#pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    const int tg = nt * 8 + g;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                    int32_t d[4] = {0, 0, 0, 0};
                    mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1);
                    const int t0 = nt * 8 + 2 * tig;
                    const int t1 = t0 + 1;
                    acc[nt][0] = fmaf(dw0 * s_dx[t0 * 8 + b], (float) d[0], acc[nt][0]);
                    acc[nt][1] = fmaf(dw0 * s_dx[t1 * 8 + b], (float) d[1], acc[nt][1]);
                    acc[nt][2] = fmaf(dw1 * s_dx[t0 * 8 + b], (float) d[2], acc[nt][2]);
                    acc[nt][3] = fmaf(dw1 * s_dx[t1 * 8 + b], (float) d[3], acc[nt][3]);
                }
            } else if constexpr (GTYPE == kTypeQ4_K) {
                const half2 dm0 = v0 ? *reinterpret_cast<const half2*>(blk0) : __floats2half2_rn(0.0f, 0.0f);
                const half2 dm1 = v1 ? *reinterpret_cast<const half2*>(blk1) : __floats2half2_rn(0.0f, 0.0f);
                float s0 = 0.0f, mn0 = 0.0f, s1 = 0.0f, mn1 = 0.0f;
                if (v0) decode_k4_scale_min(blk0 + 4, b, s0, mn0);
                if (v1) decode_k4_scale_min(blk1 + 4, b, s1, mn1);
                const float ds0 = __low2float(dm0) * s0, ms0 = __high2float(dm0) * mn0;
                const float ds1 = __low2float(dm1) * s1, ms1 = __high2float(dm1) * mn1;
                const int j = b >> 1, hi = b & 1;
                const uint32_t raw0_lo = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 16 + 32 * j + tig * 4) : 0u;
                const uint32_t raw0_hi = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 16 + 32 * j + 16 + tig * 4) : 0u;
                const uint32_t raw1_lo = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 16 + 32 * j + tig * 4) : 0u;
                const uint32_t raw1_hi = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 16 + 32 * j + 16 + tig * 4) : 0u;
                const uint32_t a0 = hi ? ((raw0_lo >> 4) & 0x0F0F0F0Fu) : (raw0_lo & 0x0F0F0F0Fu);
                const uint32_t a2 = hi ? ((raw0_hi >> 4) & 0x0F0F0F0Fu) : (raw0_hi & 0x0F0F0F0Fu);
                const uint32_t a1 = hi ? ((raw1_lo >> 4) & 0x0F0F0F0Fu) : (raw1_lo & 0x0F0F0F0Fu);
                const uint32_t a3 = hi ? ((raw1_hi >> 4) & 0x0F0F0F0Fu) : (raw1_hi & 0x0F0F0F0Fu);
#pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    const int tg = nt * 8 + g;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                    int32_t d[4] = {0, 0, 0, 0};
                    mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1);
                    const int t0 = nt * 8 + 2 * tig;
                    const int t1 = t0 + 1;
                    acc[nt][0] = fmaf(ds0 * s_dx[t0 * 8 + b], (float) d[0], fmaf(-ms0, s_xs[t0 * 8 + b], acc[nt][0]));
                    acc[nt][1] = fmaf(ds0 * s_dx[t1 * 8 + b], (float) d[1], fmaf(-ms0, s_xs[t1 * 8 + b], acc[nt][1]));
                    acc[nt][2] = fmaf(ds1 * s_dx[t0 * 8 + b], (float) d[2], fmaf(-ms1, s_xs[t0 * 8 + b], acc[nt][2]));
                    acc[nt][3] = fmaf(ds1 * s_dx[t1 * 8 + b], (float) d[3], fmaf(-ms1, s_xs[t1 * 8 + b], acc[nt][3]));
                }
            } else if constexpr (GTYPE == kTypeQ5_K) {
                const half2 dm0 = v0 ? *reinterpret_cast<const half2*>(blk0) : __floats2half2_rn(0.0f, 0.0f);
                const half2 dm1 = v1 ? *reinterpret_cast<const half2*>(blk1) : __floats2half2_rn(0.0f, 0.0f);
                float s0 = 0.0f, mn0 = 0.0f, s1 = 0.0f, mn1 = 0.0f;
                if (v0) decode_k4_scale_min(blk0 + 4, b, s0, mn0);
                if (v1) decode_k4_scale_min(blk1 + 4, b, s1, mn1);
                const float ds0 = __low2float(dm0) * s0, ms0 = __high2float(dm0) * mn0;
                const float ds1 = __low2float(dm1) * s1, ms1 = __high2float(dm1) * mn1;
                const int j = b >> 1, hi = b & 1;
                const uint32_t qh0_lo = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 16 + tig * 4) : 0u;
                const uint32_t qh0_hi = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 32 + tig * 4) : 0u;
                const uint32_t qh1_lo = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 16 + tig * 4) : 0u;
                const uint32_t qh1_hi = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 32 + tig * 4) : 0u;
                const uint32_t raw0_lo = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 48 + 32 * j + tig * 4) : 0u;
                const uint32_t raw0_hi = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 48 + 32 * j + 16 + tig * 4) : 0u;
                const uint32_t raw1_lo = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 48 + 32 * j + tig * 4) : 0u;
                const uint32_t raw1_hi = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 48 + 32 * j + 16 + tig * 4) : 0u;
                const uint32_t q4_0l = hi ? ((raw0_lo >> 4) & 0x0F0F0F0Fu) : (raw0_lo & 0x0F0F0F0Fu);
                const uint32_t q4_0h = hi ? ((raw0_hi >> 4) & 0x0F0F0F0Fu) : (raw0_hi & 0x0F0F0F0Fu);
                const uint32_t q4_1l = hi ? ((raw1_lo >> 4) & 0x0F0F0F0Fu) : (raw1_lo & 0x0F0F0F0Fu);
                const uint32_t q4_1h = hi ? ((raw1_hi >> 4) & 0x0F0F0F0Fu) : (raw1_hi & 0x0F0F0F0Fu);
                const uint32_t a0 = q4_0l | (((qh0_lo >> b) & 0x01010101u) << 4);
                const uint32_t a2 = q4_0h | (((qh0_hi >> b) & 0x01010101u) << 4);
                const uint32_t a1 = q4_1l | (((qh1_lo >> b) & 0x01010101u) << 4);
                const uint32_t a3 = q4_1h | (((qh1_hi >> b) & 0x01010101u) << 4);
#pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    const int tg = nt * 8 + g;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                    int32_t d[4] = {0, 0, 0, 0};
                    mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1);
                    const int t0 = nt * 8 + 2 * tig;
                    const int t1 = t0 + 1;
                    acc[nt][0] = fmaf(ds0 * s_dx[t0 * 8 + b], (float) d[0], fmaf(-ms0, s_xs[t0 * 8 + b], acc[nt][0]));
                    acc[nt][1] = fmaf(ds0 * s_dx[t1 * 8 + b], (float) d[1], fmaf(-ms0, s_xs[t1 * 8 + b], acc[nt][1]));
                    acc[nt][2] = fmaf(ds1 * s_dx[t0 * 8 + b], (float) d[2], fmaf(-ms1, s_xs[t0 * 8 + b], acc[nt][2]));
                    acc[nt][3] = fmaf(ds1 * s_dx[t1 * 8 + b], (float) d[3], fmaf(-ms1, s_xs[t1 * 8 + b], acc[nt][3]));
                }
            } else if constexpr (GTYPE == kTypeQ6_K) {
                const float d0_all = v0 ? smem_ld_fp16_b2(blk0 + 208) : 0.0f;
                const float d1_all = v1 ? smem_ld_fp16_b2(blk1 + 208) : 0.0f;
                const int8_t* sc0_p = reinterpret_cast<const int8_t*>(blk0 + 192);
                const int8_t* sc1_p = reinterpret_cast<const int8_t*>(blk1 + 192);
                const int n = b >> 2;
                const int qid = b & 3;
                const int ql_off = 64 * n + ((qid & 1) ? 32 : 0);
#pragma unroll
                for (int half16 = 0; half16 < 2; ++half16) {
                    const int l0 = half16 * 16;
                    const int sub = 2 * b + half16;
                    const float dw0 = v0 ? (d0_all * (float) sc0_p[sub]) : 0.0f;
                    const float dw1 = v1 ? (d1_all * (float) sc1_p[sub]) : 0.0f;
                    const uint32_t qh0 = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 128 + 32 * n + l0 + tig * 4) : 0u;
                    const uint32_t qh1 = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 128 + 32 * n + l0 + tig * 4) : 0u;
                    const uint32_t src0 = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + ql_off + l0 + tig * 4) : 0u;
                    const uint32_t src1 = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + ql_off + l0 + tig * 4) : 0u;
                    const uint32_t lo0 = (qid < 2) ? (src0 & 0x0F0F0F0Fu) : ((src0 >> 4) & 0x0F0F0F0Fu);
                    const uint32_t lo1 = (qid < 2) ? (src1 & 0x0F0F0F0Fu) : ((src1 >> 4) & 0x0F0F0F0Fu);
                    const uint32_t a0 = __vsub4(lo0 | (((qh0 >> (2 * qid)) & 0x03030303u) << 4), 0x20202020u);
                    const uint32_t a1 = __vsub4(lo1 | (((qh1 >> (2 * qid)) & 0x03030303u) << 4), 0x20202020u);
#pragma unroll
                    for (int nt = 0; nt < NTILES; ++nt) {
                        const int tg = nt * 8 + g;
                        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                            s_xq + (tg * 8 + b) * 32 + l0 + tig * 4);
                        int32_t d[4] = {0, 0, 0, 0};
                        mma_s8_m16n8k16(d, a0, a1, b0);
                        const int t0 = nt * 8 + 2 * tig;
                        const int t1 = t0 + 1;
                        acc[nt][0] = fmaf(dw0 * s_dx[t0 * 8 + b], (float) d[0], acc[nt][0]);
                        acc[nt][1] = fmaf(dw0 * s_dx[t1 * 8 + b], (float) d[1], acc[nt][1]);
                        acc[nt][2] = fmaf(dw1 * s_dx[t0 * 8 + b], (float) d[2], acc[nt][2]);
                        acc[nt][3] = fmaf(dw1 * s_dx[t1 * 8 + b], (float) d[3], acc[nt][3]);
                    }
                }
            } else if constexpr (GTYPE == kTypeIQ4_XS) {
                const float d0_all = v0 ? smem_ld_fp16_b2(blk0) : 0.0f;
                const float d1_all = v1 ? smem_ld_fp16_b2(blk1) : 0.0f;
                const uint16_t sh0 = v0 ? *reinterpret_cast<const uint16_t*>(blk0 + 2) : 0u;
                const uint16_t sh1 = v1 ? *reinterpret_cast<const uint16_t*>(blk1 + 2) : 0u;
                const uint32_t sl0 = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 4) : 0u;
                const uint32_t sl1 = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 4) : 0u;
                const int ls0 = (int) (((sl0 >> (4 * b)) & 0x0Fu) | (((sh0 >> (2 * b)) & 3u) << 4)) - 32;
                const int ls1 = (int) (((sl1 >> (4 * b)) & 0x0Fu) | (((sh1 >> (2 * b)) & 3u) << 4)) - 32;
                const float dw0 = v0 ? (d0_all * (float) ls0) : 0.0f;
                const float dw1 = v1 ? (d1_all * (float) ls1) : 0.0f;
                const uint32_t raw0 = v0 ? *reinterpret_cast<const uint32_t*>(blk0 + 8 + b * 16 + tig * 4) : 0u;
                const uint32_t raw1 = v1 ? *reinterpret_cast<const uint32_t*>(blk1 + 8 + b * 16 + tig * 4) : 0u;
                const uint2 w0_lh = iq4nl_lut8(raw0);
                const uint2 w1_lh = iq4nl_lut8(raw1);
#pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    const int tg = nt * 8 + g;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                    int32_t d[4] = {0, 0, 0, 0};
                    mma_s8_m16n8k32(d, w0_lh.x, w1_lh.x, w0_lh.y, w1_lh.y, b0, b1);
                    const int t0 = nt * 8 + 2 * tig;
                    const int t1 = t0 + 1;
                    acc[nt][0] = fmaf(dw0 * s_dx[t0 * 8 + b], (float) d[0], acc[nt][0]);
                    acc[nt][1] = fmaf(dw0 * s_dx[t1 * 8 + b], (float) d[1], acc[nt][1]);
                    acc[nt][2] = fmaf(dw1 * s_dx[t0 * 8 + b], (float) d[2], acc[nt][2]);
                    acc[nt][3] = fmaf(dw1 * s_dx[t1 * 8 + b], (float) d[3], acc[nt][3]);
                }
            }
        }
        __syncthreads();
    }

    // Reduce the 8 warps' partial sums using s_w as scratch (8 * MAX_T * 16 floats <= 8 * 16 * 16 * 4 = 8 KB;
    // use a dedicated small 4 KB / 8 KB smem buffer).
    __shared__ float s_red[8][MAX_T][16];
#pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {
        const int t0 = nt * 8 + 2 * tig;
        const int t1 = t0 + 1;
        s_red[warp][t0][g] = acc[nt][0];
        s_red[warp][t1][g] = acc[nt][1];
        s_red[warp][t0][8 + g] = acc[nt][2];
        s_red[warp][t1][8 + g] = acc[nt][3];
    }
    __syncthreads();

    for (int idx = tid; idx < T * 16; idx += 256) {
        const int tk = idx >> 4;
        const int lr = idx & 15;
        const int row = r_base + lr;
        if (row < n_out) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) sum += s_red[w][tk][lr];
            y[(size_t) tk * n_out + row] = sum;
        }
    }
#else
    (void) weights; (void) xq; (void) y; (void) n_in; (void) n_out; (void) T;
#endif
}

// ---- Decode / Verify Kernel (legacy multi-warp, unused when ksplit runs) ----
template <int GTYPE, int NTILES, int WARPS = 2>
__global__ void __launch_bounds__(WARPS * 32) dense_i8_tc_decode_kernel(
    const uint8_t* __restrict__ weights, const Q81Block* __restrict__ xq, float* __restrict__ y,
    int n_in, int n_out, int T) {
#if STRATA_DENSE_I8_SM80
    constexpr int MAX_T = NTILES * 8;
    constexpr int NTHREADS = WARPS * 32;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int tig = lane & 3;

    const int r_base = (int) blockIdx.x * (WARPS * 16) + warp * 16;
    const int r0 = r_base + g;
    const int r1 = r_base + 8 + g;
    const bool v0 = r0 < n_out;
    const bool v1 = r1 < n_out;

    const int nb32 = n_in >> 5;
    const size_t rbytes = row_bytes_of(GTYPE, n_in);

    __shared__ alignas(16) int8_t s_xq[MAX_T * 256];
    __shared__ float s_dx[MAX_T * 8];
    __shared__ float s_xs[MAX_T * 8];
    __shared__ alignas(16) uint32_t s_w[WARPS][16][chunk_u32_of<GTYPE>()];

    float acc[NTILES][4];
#pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {
        acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f;
    }

    for (int k_blk0 = 0; k_blk0 < nb32; k_blk0 += 8) {
        const int nb_here = min(8, nb32 - k_blk0);
        for (int idx = tid; idx < MAX_T * 8; idx += NTHREADS) {
            const int t = idx >> 3;
            const int b = idx & 7;
            if (t < T && b < nb_here) {
                const Q81Block& blk = xq[(size_t) t * nb32 + k_blk0 + b];
                const float dx = __low2float(blk.ds);
                s_dx[idx] = dx;
                const uint32_t* src32 = reinterpret_cast<const uint32_t*>(blk.qs);
                const uint32_t x0 = src32[0], x1 = src32[1], x2 = src32[2], x3 = src32[3];
                const uint32_t x4 = src32[4], x5 = src32[5], x6 = src32[6], x7 = src32[7];
                if constexpr (GTYPE == kTypeQ4_K || GTYPE == kTypeQ5_K) {
                    int qsum = __dp4a(0x01010101, (int) x0, 0);
                    qsum = __dp4a(0x01010101, (int) x1, qsum);
                    qsum = __dp4a(0x01010101, (int) x2, qsum);
                    qsum = __dp4a(0x01010101, (int) x3, qsum);
                    qsum = __dp4a(0x01010101, (int) x4, qsum);
                    qsum = __dp4a(0x01010101, (int) x5, qsum);
                    qsum = __dp4a(0x01010101, (int) x6, qsum);
                    qsum = __dp4a(0x01010101, (int) x7, qsum);
                    s_xs[idx] = dx * (float) qsum;
                }
                uint4* dst4 = reinterpret_cast<uint4*>(s_xq + idx * 32);
                dst4[0] = make_uint4(x0, x1, x2, x3);
                dst4[1] = make_uint4(x4, x5, x6, x7);
            } else {
                s_dx[idx] = 0.0f;
                if constexpr (GTYPE == kTypeQ4_K || GTYPE == kTypeQ5_K) s_xs[idx] = 0.0f;
                uint4* dst4 = reinterpret_cast<uint4*>(s_xq + idx * 32);
                dst4[0] = make_uint4(0u, 0u, 0u, 0u);
                dst4[1] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        stage_dense_rows16<GTYPE>(weights, r_base, n_out, rbytes, k_blk0, nb_here, lane, s_w[warp]);
        __syncthreads();

        if (r_base < n_out) {
            eval_kchunk_tc_smem<GTYPE, NTILES>(s_w[warp], v0, v1, nb_here, s_xq, s_dx, s_xs, g, tig, acc);
        }
        __syncthreads();
    }

    if (v0) {
#pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            const int t0 = nt * 8 + 2 * tig;
            const int t1 = t0 + 1;
            if (t0 < T) y[(size_t) t0 * n_out + r0] = acc[nt][0];
            if (t1 < T) y[(size_t) t1 * n_out + r0] = acc[nt][1];
        }
    }
    if (v1) {
#pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            const int t0 = nt * 8 + 2 * tig;
            const int t1 = t0 + 1;
            if (t0 < T) y[(size_t) t0 * n_out + r1] = acc[nt][2];
            if (t1 < T) y[(size_t) t1 * n_out + r1] = acc[nt][3];
        }
    }
#else
    (void) weights; (void) xq; (void) y; (void) n_in; (void) n_out; (void) T;
#endif
}

// ---- Prompt Prefill Kernel: 8 warps (256 threads) = 64 output rows x 64 tokens per CTA ----
// Warp layout: wm = warp >> 1 (0..3 -> 16 rows each = 64 rows), wn = warp & 1 (0..1 -> 32 tokens = 4 N-tiles each).
// Double-buffered 64-token x 256-K activation slab = 2 * (64*256 + 64*8*8) = ~41 KB smem -> 2 CTAs/SM!
template <int GTYPE>
__global__ void __launch_bounds__(256) dense_i8_tc_prefill_kernel(
    const uint8_t* __restrict__ weights, const Q81Block* __restrict__ xq, float* __restrict__ y,
    int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {
#if STRATA_DENSE_I8_SM80
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int NTILES_W = 4;  // 32 tokens per warp
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp >> 1;    // 0..3
    const int wn = warp & 1;     // 0..1
    const int g = lane >> 2;
    const int tig = lane & 3;

    const int m0 = (int) blockIdx.x * BM;
    const int n0 = (int) blockIdx.y * BN;
    const int r0 = m0 + wm * 16 + g;
    const int r1 = r0 + 8;
    const bool v0 = r0 < N;
    const bool v1 = r1 < N;

    const int nb32 = (int) (K >> 5);
    const size_t rbytes = row_bytes_of(GTYPE, (int) K);
    const uint8_t* w0 = weights + (size_t) (v0 ? r0 : 0) * rbytes;
    const uint8_t* w1 = weights + (size_t) (v1 ? r1 : 0) * rbytes;

    __shared__ alignas(16) int8_t s_xq[BN * 256];
    __shared__ float s_dx[BN * 8];
    __shared__ float s_xs[BN * 8];

    float acc[NTILES_W][4];
#pragma unroll
    for (int nt = 0; nt < NTILES_W; ++nt) {
        acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f;
    }

    const int tok_base = wn * 32;
    const int8_t* my_xq = s_xq + tok_base * 256;
    const float* my_dx = s_dx + tok_base * 8;
    const float* my_xs = s_xs + tok_base * 8;

    for (int k_blk0 = 0; k_blk0 < nb32; k_blk0 += 8) {
        const int nb_here = min(8, nb32 - k_blk0);
        // 256 threads load 64*8 = 512 Q81Blocks (2 blocks per thread) cooperatively.
        for (int idx = tid; idx < BN * 8; idx += 256) {
            const int t_loc = idx >> 3;
            const int b = idx & 7;
            const int t_glob = n0 + t_loc;
            if (t_glob < T && b < nb_here) {
                const Q81Block& blk = xq[(size_t) t_glob * nb32 + k_blk0 + b];
                const float dx = __low2float(blk.ds);
                s_dx[idx] = dx;
                const uint32_t* src32 = reinterpret_cast<const uint32_t*>(blk.qs);
                const uint32_t w0 = src32[0], w1 = src32[1], w2 = src32[2], w3 = src32[3];
                const uint32_t w4 = src32[4], w5 = src32[5], w6 = src32[6], w7 = src32[7];
                int qsum = __dp4a(0x01010101, (int) w0, 0);
                qsum = __dp4a(0x01010101, (int) w1, qsum);
                qsum = __dp4a(0x01010101, (int) w2, qsum);
                qsum = __dp4a(0x01010101, (int) w3, qsum);
                qsum = __dp4a(0x01010101, (int) w4, qsum);
                qsum = __dp4a(0x01010101, (int) w5, qsum);
                qsum = __dp4a(0x01010101, (int) w6, qsum);
                qsum = __dp4a(0x01010101, (int) w7, qsum);
                s_xs[idx] = dx * (float) qsum;
                uint4* dst4 = reinterpret_cast<uint4*>(s_xq + idx * 32);
                dst4[0] = make_uint4(w0, w1, w2, w3);
                dst4[1] = make_uint4(w4, w5, w6, w7);
            } else {
                s_dx[idx] = 0.0f;
                s_xs[idx] = 0.0f;
                uint4* dst4 = reinterpret_cast<uint4*>(s_xq + idx * 32);
                dst4[0] = make_uint4(0u, 0u, 0u, 0u);
                dst4[1] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        if ((v0 || v1) && (n0 + tok_base < T)) {
            eval_kchunk_tc<GTYPE, NTILES_W>(w0, w1, v0, v1, nb_here, k_blk0, my_xq, my_dx, my_xs, g, tig, acc);
        }
        __syncthreads();
    }

    auto store = [&](int row, int tok, float val) {
        if (tok < T) {
            float* dst = y + (size_t) tok * (size_t) ldy + (size_t) row;
            *dst = (beta == 0.0f) ? val : fmaf(beta, *dst, val);
        }
    };
    if (v0) {
#pragma unroll
        for (int nt = 0; nt < NTILES_W; ++nt) {
            const int t0 = n0 + tok_base + nt * 8 + 2 * tig;
            store(r0, t0, acc[nt][0]);
            store(r0, t0 + 1, acc[nt][1]);
        }
    }
    if (v1) {
#pragma unroll
        for (int nt = 0; nt < NTILES_W; ++nt) {
            const int t0 = n0 + tok_base + nt * 8 + 2 * tig;
            store(r1, t0, acc[nt][2]);
            store(r1, t0 + 1, acc[nt][3]);
        }
    }
#else
    (void) weights; (void) xq; (void) y; (void) T; (void) N; (void) K; (void) ldy; (void) beta;
#endif
}

int current_cc() {
#if defined(__HIPCC__)
    return 0;
#else
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) { (void) cudaGetLastError(); return 0; }
    int mj = 0, mn = 0;
    if (cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&mn, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return 0;
    }
    return mj * 10 + mn;
#endif
}

}  // namespace

bool dense_i8_tc_supported() {
    return current_cc() >= 80;
}

bool dense_i8_tc_type_ok(int ggml_type) noexcept {
    return ggml_type == kTypeQ8_0 || ggml_type == kTypeQ4_K || ggml_type == kTypeQ5_K ||
           ggml_type == kTypeQ6_K || ggml_type == kTypeIQ4_XS;
}

bool dense_i8_tc_decode_enabled() {
    static const int env = [] {
        const char* v = std::getenv("STRATA_DENSE_I8_TC");
        if (v == nullptr || v[0] == '\0') return -1;
        return std::atoi(v) != 0 ? 1 : 0;
    }();
    if (env == 0) return false;
    return dense_i8_tc_supported();
}

bool dense_i8_tc_prefill_enabled() {
    static const int env = [] {
        const char* v = std::getenv("STRATA_PREFILL_DENSE_I8_TC");
        if (v == nullptr || v[0] == '\0') return -1;
        return std::atoi(v) != 0 ? 1 : 0;
    }();
    if (env == 0) return false;
    return dense_i8_tc_supported();
}

bool dense_i8_tc_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int T, void* stream) {
    if (!dense_i8_tc_supported() || !dense_i8_tc_type_ok(ggml_type)) return false;
    if (weights == nullptr || x_q8_1 == nullptr || y == nullptr || T < 1 || T > 16 || n_out <= 0 || n_in <= 0)
        return false;
    const int align_k = (ggml_type == kTypeQ8_0) ? 32 : 256;
    if ((n_in % align_k) != 0) return false;

    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const unsigned grid = (unsigned) ((n_out + 15) / 16);
    const auto* w = reinterpret_cast<const uint8_t*>(weights);
    const auto* xq = reinterpret_cast<const Q81Block*>(x_q8_1);

    auto launch = [&](auto type_tag) {
        constexpr int GT = decltype(type_tag)::value;
        if (T <= 8) {
            dense_i8_tc_decode_ksplit_kernel<GT, 1><<<grid, 256, 0, s>>>(w, xq, y, n_in, n_out, T);
        } else {
            dense_i8_tc_decode_ksplit_kernel<GT, 2><<<grid, 256, 0, s>>>(w, xq, y, n_in, n_out, T);
        }
    };

    switch (ggml_type) {
        case kTypeQ8_0: launch(std::integral_constant<int, kTypeQ8_0>{}); break;
        case kTypeQ4_K: launch(std::integral_constant<int, kTypeQ4_K>{}); break;
        case kTypeQ5_K: launch(std::integral_constant<int, kTypeQ5_K>{}); break;
        case kTypeQ6_K: launch(std::integral_constant<int, kTypeQ6_K>{}); break;
        case kTypeIQ4_XS: launch(std::integral_constant<int, kTypeIQ4_XS>{}); break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

size_t dense_i8_tc_prefill_scratch_bytes(int64_t T, int64_t K) {
    return (size_t) T * (size_t) (K / 32) * sizeof(Q81Block);
}

bool dense_i8_tc_prefill(const uint16_t* X_f16, int ggml_type, const void* W_blocks, float* Y,
                         int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                         void* scratch, size_t scratch_bytes, void* stream) {
    if (!dense_i8_tc_supported() || !dense_i8_tc_type_ok(ggml_type)) return false;
    if (X_f16 == nullptr || W_blocks == nullptr || Y == nullptr || scratch == nullptr) return false;
    if (T <= 0 || N <= 0 || K <= 0) return false;
    const int align_k = (ggml_type == kTypeQ8_0) ? 32 : 256;
    if ((K % align_k) != 0) return false;
    if (ldy <= 0) ldy = N;

    const size_t need = dense_i8_tc_prefill_scratch_bytes(T, K);
    if (scratch_bytes < need) return false;

    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    auto* xq = reinterpret_cast<Q81Block*>(scratch);
    const int64_t total_blocks = T * (K / 32);
    const unsigned q_grid = (unsigned) ((total_blocks + 7) / 8);
    quantize_f16_to_q8_1_kernel<<<q_grid, 256, 0, s>>>(X_f16, xq, total_blocks);

    const dim3 grid((unsigned) ((N + 63) / 64), (unsigned) ((T + 63) / 64));
    const auto* w = reinterpret_cast<const uint8_t*>(W_blocks);

    switch (ggml_type) {
        case kTypeQ8_0: dense_i8_tc_prefill_kernel<kTypeQ8_0><<<grid, 256, 0, s>>>(w, xq, Y, T, N, K, ldy, beta); break;
        case kTypeQ4_K: dense_i8_tc_prefill_kernel<kTypeQ4_K><<<grid, 256, 0, s>>>(w, xq, Y, T, N, K, ldy, beta); break;
        case kTypeQ5_K: dense_i8_tc_prefill_kernel<kTypeQ5_K><<<grid, 256, 0, s>>>(w, xq, Y, T, N, K, ldy, beta); break;
        case kTypeQ6_K: dense_i8_tc_prefill_kernel<kTypeQ6_K><<<grid, 256, 0, s>>>(w, xq, Y, T, N, K, ldy, beta); break;
        case kTypeIQ4_XS: dense_i8_tc_prefill_kernel<kTypeIQ4_XS><<<grid, 256, 0, s>>>(w, xq, Y, T, N, K, ldy, beta); break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace strata::kernels
