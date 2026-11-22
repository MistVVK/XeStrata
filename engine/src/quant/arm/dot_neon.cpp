// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// src/quant/arm/dot_neon.cpp - the Neon contract: ggml-cpu's dot products + ARM64 NEON dotprod vector-accumulated
// and L1-prefetched optimizations for Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, and IQ3_XXS (DGX Spark GB10 & Apple Silicon).
// Compiled with the flags of cmake/backends/quant.cmake (armv8.2-a+dotprod+fp16+i8mm, no SVE).
#include "tier_common.hpp"

#include "../generic.hpp"
#include "../registry.hpp"
#include "strata/quant/dot.hpp"

namespace strata::quant {
namespace neon {

#include "arm_helpers.inc"
#include "arm_tables.inc"
#include "arm_dots.inc"

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

// Vector-accumulated NEON dotprod Q4_K x Q8_K: eliminates 7 inner-loop vaddvq_s32 stalls per superblock (bit-exact).
void dot_q4_K_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_q4_K* STRATA_RESTRICT x = (const block_q4_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const int32x4_t mzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            __builtin_prefetch(&x[i + 2], 0, 0);
            __builtin_prefetch(&y[i + 2], 0, 0);
        }
        const float d = y[i].d * h2f(x[i].d);
        const float dmin = y[i].d * h2f(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));
        std::memcpy(utmp, x[i].scales, 12);

        uint32x2_t mins8 = {0, 0};
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t* scales = (const uint8_t*) utmp;
        const uint8_t* STRATA_RESTRICT q4 = x[i].qs;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;

        int32x4_t vsum1 = mzero;
        int32x4_t vsum2 = mzero;
        for (int j = 0; j < QK_K / 64; ++j) {
            const uint8x16x2_t q4bits = vld1q_u8_x2(q4);
            q4 += 32;
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;

            const int8x16_t q4l0 = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
            const int8x16_t q4l1 = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));
            const int8x16_t q4h0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            const int8x16_t q4h1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4l0, q8b.val[0]), q4l1, q8b.val[1]);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4h0, q8b.val[2]), q4h1, q8b.val[3]);
            vsum1 = vmlaq_n_s32(vsum1, p1, (int32_t) scales[2 * j + 0]);
            vsum2 = vmlaq_n_s32(vsum2, p2, (int32_t) scales[2 * j + 1]);
        }
        sumf += d * (float) (vaddvq_s32(vsum1) + vaddvq_s32(vsum2));
    }
    *s = sumf;
}

// Vector-accumulated NEON dotprod Q5_K x Q8_K (bit-exact).
void dot_q5_K_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_q5_K* STRATA_RESTRICT x = (const block_q5_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const uint8x16_t mone = vdupq_n_u8(1);
    const uint8x16_t mtwo = vdupq_n_u8(2);
    const int32x4_t mzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            __builtin_prefetch(&x[i + 2], 0, 0);
            __builtin_prefetch(&y[i + 2], 0, 0);
        }
        const float d = y[i].d * h2f(x[i].d);
        const float dmin = y[i].d * h2f(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8x8_t mins8 = vld1_u8((const uint8_t*) utmp + 8);
        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(mins8));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        const int32_t sumi_mins = vaddvq_s32(prod);

        const uint8_t* scales = (const uint8_t*) utmp;
        const uint8_t* STRATA_RESTRICT q5 = x[i].qs;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;
        uint8x16x2_t qhbits = vld1q_u8_x2(x[i].qh);

        int32x4_t vsum1 = mzero;
        int32x4_t vsum2 = mzero;
        for (int j = 0; j < QK_K / 64; ++j) {
            const uint8x16x2_t q5bits = vld1q_u8_x2(q5);
            q5 += 32;
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;

            const uint8x16_t h0 = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
            const uint8x16_t h1 = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
            const uint8x16_t h2 = vshlq_n_u8(vandq_u8(mtwo, qhbits.val[0]), 3);
            const uint8x16_t h3 = vshlq_n_u8(vandq_u8(mtwo, qhbits.val[1]), 3);
            qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 2);
            qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 2);

            const int8x16_t q5_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[0], m4b), h0));
            const int8x16_t q5_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[1], m4b), h1));
            const int8x16_t q5_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[0], 4), h2));
            const int8x16_t q5_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[1], 4), h3));

            const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q5_0, q8b.val[0]), q5_1, q8b.val[1]);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q5_2, q8b.val[2]), q5_3, q8b.val[3]);
            vsum1 = vmlaq_n_s32(vsum1, p1, (int32_t) scales[2 * j + 0]);
            vsum2 = vmlaq_n_s32(vsum2, p2, (int32_t) scales[2 * j + 1]);
        }
        const int32_t sumi = vaddvq_s32(vaddq_s32(vsum1, vsum2));
        sumf += d * (float) sumi - dmin * (float) sumi_mins;
    }
    *s = sumf;
}

// Vector-accumulated NEON dotprod IQ4_XS x Q8_K (bit-exact).
void dot_iq4_xs_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_iq4_xs* STRATA_RESTRICT x = (const block_iq4_xs*) vx;
    const block_q8_K*   STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    const int8x16_t values = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const int32x4_t mzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (int ibl = 0; ibl < nb; ++ibl) {
        if (ibl + 2 < nb) {
            __builtin_prefetch(&x[ibl + 2], 0, 0);
            __builtin_prefetch(&y[ibl + 2], 0, 0);
        }
        const int8_t*  STRATA_RESTRICT q8 = y[ibl].qs;
        const uint8_t* STRATA_RESTRICT q4 = x[ibl].qs;
        uint16_t h = x[ibl].scales_h;

        int32x4_t vsum1 = mzero;
        int32x4_t vsum2 = mzero;
        for (int ib = 0; ib < QK_K / 64; ++ib) {
            const uint8x16x2_t q4bits = vld1q_u8_x2(q4);
            q4 += 32;
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;

            const int8x16_t q4_0 = vqtbl1q_s8(values, vandq_u8(q4bits.val[0], m4b));
            const int8x16_t q4_1 = vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[0], 4));
            const int8x16_t q4_2 = vqtbl1q_s8(values, vandq_u8(q4bits.val[1], m4b));
            const int8x16_t q4_3 = vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4_0, q8b.val[0]), q4_1, q8b.val[1]);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4_2, q8b.val[2]), q4_3, q8b.val[3]);

            const int32_t ls1 = ((x[ibl].scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            const int32_t ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
            h >>= 4;
            vsum1 = vmlaq_n_s32(vsum1, p1, ls1);
            vsum2 = vmlaq_n_s32(vsum2, p2, ls2);
        }
        sumf += h2f(x[ibl].d) * y[ibl].d * (float) (vaddvq_s32(vsum1) + vaddvq_s32(vsum2));
    }
    *s = sumf;
}

// Prefetched NEON dotprod Q8_0 x Q8_0 (bit-exact).
void dot_q8_0_q8_0_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK8_0 == 0);
    const int nb = n / QK8_0;
    const block_q8_0* STRATA_RESTRICT x = (const block_q8_0*) vx;
    const block_q8_0* STRATA_RESTRICT y = (const block_q8_0*) vy;

    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);
    const int32x4_t mzero = vdupq_n_s32(0);

    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        if (ib + 4 < nb) {
            __builtin_prefetch(&x[ib + 4], 0, 0);
            __builtin_prefetch(&y[ib + 4], 0, 0);
        }
        const block_q8_0* STRATA_RESTRICT x0 = &x[ib + 0];
        const block_q8_0* STRATA_RESTRICT x1 = &x[ib + 1];
        const block_q8_0* STRATA_RESTRICT y0 = &y[ib + 0];
        const block_q8_0* STRATA_RESTRICT y1 = &y[ib + 1];

        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(vdotq_s32(mzero, x0_0, y0_0), vdotq_s32(mzero, x0_1, y0_1))),
                            h2f(x0->d) * h2f(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(vdotq_s32(mzero, x1_0, y1_0), vdotq_s32(mzero, x1_1, y1_1))),
                            h2f(x1->d) * h2f(y1->d));
    }
    float sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; ++j) sumi += x[ib].qs[j] * y[ib].qs[j];
        sumf += sumi * (h2f(x[ib].d) * h2f(y[ib].d));
    }
    *s = sumf;
}

#endif  // __ARM_NEON && __ARM_FEATURE_DOTPROD

}  // namespace neon

namespace {
#define ENTRY(T, ACT, NAME) {neon::dot_##NAME, Type::T, Type::ACT, Arith::Neon, Isa::Neon, "neon_" #NAME}
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define ENTRY_OPT(T, ACT, NAME) {neon::dot_##NAME##_opt, Type::T, Type::ACT, Arith::Neon, Isa::Neon, "neon_" #NAME}
#else
#define ENTRY_OPT(T, ACT, NAME) ENTRY(T, ACT, NAME)
#endif
const DotKernel kTable[] = {
    ENTRY(Q2_0, Q8_0, q2_0_q8_0),     ENTRY(Q4_0, Q8_0, q4_0_q8_0),     ENTRY(Q4_1, Q8_1, q4_1_q8_1),     ENTRY(Q5_0, Q8_0, q5_0_q8_0),
    ENTRY(Q5_1, Q8_1, q5_1_q8_1),     ENTRY_OPT(Q8_0, Q8_0, q8_0_q8_0), ENTRY(IQ4_NL, Q8_0, iq4_nl_q8_0), ENTRY(MXFP4, Q8_0, mxfp4_q8_0),
    ENTRY(NVFP4, Q8_0, nvfp4_q8_0),   ENTRY(Q2_K, Q8_K, q2_K_q8_K),     ENTRY(Q3_K, Q8_K, q3_K_q8_K),     ENTRY_OPT(Q4_K, Q8_K, q4_K_q8_K),
    ENTRY_OPT(Q5_K, Q8_K, q5_K_q8_K), ENTRY(Q6_K, Q8_K, q6_K_q8_K),     ENTRY(IQ1_S, Q8_K, iq1_s_q8_K),   ENTRY(IQ1_M, Q8_K, iq1_m_q8_K),
    ENTRY(IQ2_XXS, Q8_K, iq2_xxs_q8_K), ENTRY(IQ2_XS, Q8_K, iq2_xs_q8_K), ENTRY(IQ2_S, Q8_K, iq2_s_q8_K),
    ENTRY(IQ3_XXS, Q8_K, iq3_xxs_q8_K), ENTRY(IQ3_S, Q8_K, iq3_s_q8_K), ENTRY_OPT(IQ4_XS, Q8_K, iq4_xs_q8_K),
};
#undef ENTRY_OPT
#undef ENTRY
}  // namespace

DotTable neon_dot_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }

}  // namespace strata::quant
