// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// src/quant/x86/dot_avx2.cpp - the Avx2 contract: ggml-cpu's dot products + bit-exact optimized AVX2 kernels for
// Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_XXS (dual accumulators, unrolled shifts, bit-3 sign LUT, L1 prefetching,
// and 4-row shared-activation AVX2 dot_x4 kernels). Strictly AVX2+FMA+F16C only: 0 EVEX/ZMM instructions.
#include "tier_common.hpp"

#include "../generic.hpp"
#include "../registry.hpp"
#include "strata/quant/dot.hpp"

namespace strata::quant {
namespace avx2 {

#include "x86_helpers.inc"
#include "x86_dots.inc"

// ---- Bit-exact optimized AVX2 cold-expert dots (0 EVEX/ZMM instructions, 0 ULP diff vs x86_dots.inc) ----

void dot_q8_0_q8_0_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK8_0 == 0);
    const int nb = n / QK8_0;
    const block_q8_0* STRATA_RESTRICT x = (const block_q8_0*) vx;
    const block_q8_0* STRATA_RESTRICT y = (const block_q8_0*) vy;

    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    int ib = 0;

    for (; ib + 1 < nb; ib += 2) {
        if (ib + 4 < nb) {
            _mm_prefetch((const char*) &x[ib + 4], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[ib + 4], _MM_HINT_T0);
        }
        const __m256 d0 = _mm256_set1_ps(h2f(x[ib + 0].d) * h2f(y[ib + 0].d));
        const __m256 d1 = _mm256_set1_ps(h2f(x[ib + 1].d) * h2f(y[ib + 1].d));

        const __m256i qx0 = _mm256_loadu_si256((const __m256i*) x[ib + 0].qs);
        const __m256i qy0 = _mm256_loadu_si256((const __m256i*) y[ib + 0].qs);
        const __m256i qx1 = _mm256_loadu_si256((const __m256i*) x[ib + 1].qs);
        const __m256i qy1 = _mm256_loadu_si256((const __m256i*) y[ib + 1].qs);

        const __m256i p16_0 = _mm256_maddubs_epi16(_mm256_sign_epi8(qx0, qx0), _mm256_sign_epi8(qy0, qx0));
        const __m256i p16_1 = _mm256_maddubs_epi16(_mm256_sign_epi8(qx1, qx1), _mm256_sign_epi8(qy1, qx1));
        const __m256 q0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, p16_0));
        const __m256 q1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, p16_1));

        acc = _mm256_fmadd_ps(d0, q0, acc);
        acc = _mm256_fmadd_ps(d1, q1, acc);
    }
    for (; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(h2f(x[ib].d) * h2f(y[ib].d));
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(qx, qx), _mm256_sign_epi8(qy, qx));
        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, p16)), acc);
    }
    *s = hsum_float_8(acc);
}

void dot_q4_K_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_q4_K* STRATA_RESTRICT x = (const block_q4_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const float d = y[i].d * h2f(x[i].d);
        const float dmin = -y[i].d * h2f(x[i].dmin);

        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* STRATA_RESTRICT q4 = x[i].qs;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_castsi256_si128(mins_and_scales);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
            const __m256i q8h = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            q8 += 64;

            const __m256i p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));
            const __m256i p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));
            sumi0 = _mm256_add_epi32(sumi0, p16l);
            sumi1 = _mm256_add_epi32(sumi1, p16h);
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi0, sumi1)), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

void dot_q5_K_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_q5_K* STRATA_RESTRICT x = (const block_q5_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m16 = _mm256_set1_epi8(16);
    const __m128i mzero = _mm_setzero_si128();
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const uint8_t* STRATA_RESTRICT q5 = x[i].qs;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;

        const float d = y[i].d * h2f(x[i].d);
        const float dmin = -y[i].d * h2f(x[i].dmin);

        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128 = _mm256_castsi256_si128(mins_and_scales);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();

#define Q5K_AVX2_STEP(J, QH0, QH1) do { \
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * (J) + 0)); \
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * (J) + 1)); \
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) (q5 + 32 * (J))); \
            const __m256i q5_0 = _mm256_or_si256(_mm256_and_si256(q5bits, m4), (QH0)); \
            const __m256i q5_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4), (QH1)); \
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) (q8 + 64 * (J))); \
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 64 * (J) + 32)); \
            sumi0 = _mm256_add_epi32(sumi0, _mm256_madd_epi16(scale_0, _mm256_maddubs_epi16(q5_0, q8_0))); \
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(scale_1, _mm256_maddubs_epi16(q5_1, q8_1))); \
        } while (0)

        Q5K_AVX2_STEP(0, _mm256_and_si256(_mm256_slli_epi16(hbits, 4), m16), _mm256_and_si256(_mm256_slli_epi16(hbits, 3), m16));
        Q5K_AVX2_STEP(1, _mm256_and_si256(_mm256_slli_epi16(hbits, 2), m16), _mm256_and_si256(_mm256_slli_epi16(hbits, 1), m16));
        Q5K_AVX2_STEP(2, _mm256_and_si256(hbits, m16),                        _mm256_and_si256(_mm256_srli_epi16(hbits, 1), m16));
        Q5K_AVX2_STEP(3, _mm256_and_si256(_mm256_srli_epi16(hbits, 2), m16), _mm256_and_si256(_mm256_srli_epi16(hbits, 3), m16));
#undef Q5K_AVX2_STEP

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi0, sumi1)), acc);
    }

    *s = hsum_float_8(acc) + summs;
}

void dot_q6_K_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_q6_K* STRATA_RESTRICT x = (const block_q6_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m192 = _mm256_set1_epi8(-64);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const float d = y[i].d * h2f(x[i].d);
        const uint8_t* STRATA_RESTRICT q4 = x[i].ql;
        const uint8_t* STRATA_RESTRICT qh = x[i].qh;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i*) x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*) q4);
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*) (q4 + 32));
            q4 += 64;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) qh);
            qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m12), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, m48);
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, m192), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) (q8 +  0));
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 64));
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) (q8 + 96));
            q8 += 128;

            const __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            const __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            const __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            const __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(_mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0),
                                                             _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1)));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(_mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2),
                                                             _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3)));
        }

        const __m256i sumi = _mm256_sub_epi32(_mm256_add_epi32(sumi0, sumi1), q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);
}

void dot_iq4_xs_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_iq4_xs* STRATA_RESTRICT x = (const block_iq4_xs*) vx;
    const block_q8_K*   STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    // Unsigned |kvalues_iq4nl[k]| LUT + bit-3 sign mask eliminates both vpsignb(x,x) and dual 128-bit shuffles.
    static constexpr uint8_t kAbsIq4nl[16] = {127, 104, 83, 65, 49, 35, 22, 10, 1, 13, 25, 38, 53, 69, 89, 113};
    const __m256i abs256 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kAbsIq4nl));
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i bit3 = _mm256_set1_epi8(0x08);

    __m256 accum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        if (ibl + 2 < nb) {
            _mm_prefetch((const char*) &x[ibl + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[ibl + 2], _MM_HINT_T0);
        }
        const uint8_t* STRATA_RESTRICT qs = x[ibl].qs;
        const int8_t*  STRATA_RESTRICT q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*) qs);
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*) (qs + 16));
            qs += 32;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*) q8);
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            q8 += 64;

            const __m256i nib_1 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bits_1, 4), q4bits_1), m4b);
            const __m256i nib_2 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bits_2, 4), q4bits_2), m4b);
            const __m256i ax_1 = _mm256_shuffle_epi8(abs256, nib_1);
            const __m256i ax_2 = _mm256_shuffle_epi8(abs256, nib_2);
            const __m256i neg_1 = _mm256_cmpgt_epi8(bit3, nib_1);
            const __m256i neg_2 = _mm256_cmpgt_epi8(bit3, nib_2);
            const __m256i sy_1 = _mm256_sub_epi8(_mm256_xor_si256(q8b_1, neg_1), neg_1);
            const __m256i sy_2 = _mm256_sub_epi8(_mm256_xor_si256(q8b_2, neg_2), neg_2);

            const __m256i p16_1 = _mm256_maddubs_epi16(ax_1, sy_1);
            const __m256i p16_2 = _mm256_maddubs_epi16(ax_2, sy_2);

            const int16_t ls1 = (int16_t) (((x[ibl].scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const int16_t ls2 = (int16_t) (((x[ibl].scales_l[ib / 2] >>  4) | ((sh << 2) & 0x30)) - 32);
            sh >>= 4;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2)));
        }
        accum = _mm256_fmadd_ps(_mm256_set1_ps(h2f(x[ibl].d) * y[ibl].d),
                                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum);
    }

    *s = hsum_float_8(accum);
}

void dot_iq3_xxs_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_iq3_xxs* STRATA_RESTRICT x = (const block_iq3_xxs*) vx;
    const block_q8_K*    STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    const uint64_t* signs64 = (const uint64_t*) keven_signs_q2xs;
    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const float d = h2f(x[i].d) * y[i].d;
        const uint8_t* STRATA_RESTRICT q3 = x[i].qs;
        const uint8_t* STRATA_RESTRICT gas = x[i].qs + QK_K / 4;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            q8 += 64;
            const __m256i q2_1 = _mm256_set_epi32(
                (int) iq3xxs_grid[q3[ 7]], (int) iq3xxs_grid[q3[ 6]], (int) iq3xxs_grid[q3[ 5]], (int) iq3xxs_grid[q3[ 4]],
                (int) iq3xxs_grid[q3[ 3]], (int) iq3xxs_grid[q3[ 2]], (int) iq3xxs_grid[q3[ 1]], (int) iq3xxs_grid[q3[ 0]]);
            const __m256i q2_2 = _mm256_set_epi32(
                (int) iq3xxs_grid[q3[15]], (int) iq3xxs_grid[q3[14]], (int) iq3xxs_grid[q3[13]], (int) iq3xxs_grid[q3[12]],
                (int) iq3xxs_grid[q3[11]], (int) iq3xxs_grid[q3[10]], (int) iq3xxs_grid[q3[ 9]], (int) iq3xxs_grid[q3[ 8]]);
            q3 += 16;
            std::memcpy(aux32, gas, 8);
            gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(
                (long long) signs64[(aux32[0] >> 21) & 127], (long long) signs64[(aux32[0] >> 14) & 127],
                (long long) signs64[(aux32[0] >>  7) & 127], (long long) signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(
                (long long) signs64[(aux32[1] >> 21) & 127], (long long) signs64[(aux32[1] >> 14) & 127],
                (long long) signs64[(aux32[1] >>  7) & 127], (long long) signs64[(aux32[1] >>  0) & 127]);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1));
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2));
            const int16_t ls1 = (int16_t) (2 * (aux32[0] >> 28) + 1);
            const int16_t ls2 = (int16_t) (2 * (aux32[1] >> 28) + 1);
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2)));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    *s = 0.25f * hsum_float_8(accumf);
}

// ---- 4-row shared-activation AVX2 kernels (zero EVEX/ZMM instructions; Alder/Raptor Lake safe) ----

void dot_q4_K_q8_K_x4_avx2(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;
    const uint8_t* wb = (const uint8_t*) w_base;
    const block_q4_K* rows[4] = {
        (const block_q4_K*) (wb + 0 * row_bytes),
        (const block_q4_K*) (wb + 1 * row_bytes),
        (const block_q4_K*) (wb + 2 * row_bytes),
        (const block_q4_K*) (wb + 3 * row_bytes),
    };
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    __m128 acc_m[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][i + 2], _MM_HINT_T0);
        }
        const float yd = y[i].d;
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m256i q8v[8] = {
            _mm256_loadu_si256((const __m256i*) (y[i].qs +   0)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs +  32)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs +  64)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs +  96)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs + 128)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs + 160)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs + 192)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs + 224)),
        };

        for (int r = 0; r < 4; ++r) {
            const block_q4_K& bx = rows[r][i];
            const float d = yd * h2f(bx.d);
            const float dmin = -yd * h2f(bx.dmin);
            uint32_t utmp[4];
            std::memcpy(utmp, bx.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
            const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m[r] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[r]);

            const __m128i sc128 = _mm256_castsi256_si128(mins_and_scales);
            const __m256i scales = MM256_SET_M128I(sc128, sc128);
            __m256i sumi0 = _mm256_setzero_si256();
            __m256i sumi1 = _mm256_setzero_si256();
            const uint8_t* q4 = bx.qs;
            for (int j = 0; j < 4; ++j) {
                const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
                const __m256i q4bits = _mm256_loadu_si256((const __m256i*) (q4 + 32 * j));
                const __m256i q4l = _mm256_and_si256(q4bits, m4);
                const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                sumi0 = _mm256_add_epi32(sumi0, _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8v[2 * j + 0])));
                sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8v[2 * j + 1])));
            }
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi0, sumi1)), acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) {
        acc_m[r] = _mm_add_ps(acc_m[r], _mm_movehl_ps(acc_m[r], acc_m[r]));
        acc_m[r] = _mm_add_ss(acc_m[r], _mm_movehdup_ps(acc_m[r]));
        out[r] = hsum_float_8(acc[r]) + _mm_cvtss_f32(acc_m[r]);
    }
}

void dot_iq4_xs_q8_K_x4_avx2(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;
    const uint8_t* wb = (const uint8_t*) w_base;
    const block_iq4_xs* rows[4] = {
        (const block_iq4_xs*) (wb + 0 * row_bytes),
        (const block_iq4_xs*) (wb + 1 * row_bytes),
        (const block_iq4_xs*) (wb + 2 * row_bytes),
        (const block_iq4_xs*) (wb + 3 * row_bytes),
    };
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;

    static constexpr uint8_t kAbsIq4nl[16] = {127, 104, 83, 65, 49, 35, 22, 10, 1, 13, 25, 38, 53, 69, 89, 113};
    const __m256i abs256 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kAbsIq4nl));
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i bit3 = _mm256_set1_epi8(0x08);

    __m256 accum[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int ibl = 0; ibl < nb; ++ibl) {
        if (ibl + 2 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][ibl + 2], _MM_HINT_T0);
        }
        const float yd = y[ibl].d;
        const __m256i q8v[8] = {
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs +   0)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs +  32)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs +  64)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs +  96)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs + 128)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs + 160)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs + 192)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs + 224)),
        };
        for (int r = 0; r < 4; ++r) {
            const block_iq4_xs& bx = rows[r][ibl];
            const uint8_t* qs = bx.qs;
            uint16_t sh = bx.scales_h;
            __m256i sumi1 = _mm256_setzero_si256();
            __m256i sumi2 = _mm256_setzero_si256();
            for (int ib = 0; ib < QK_K / 32; ib += 2) {
                const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*) qs);
                const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*) (qs + 16));
                qs += 32;
                const __m256i nib_1 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bits_1, 4), q4bits_1), m4b);
                const __m256i nib_2 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bits_2, 4), q4bits_2), m4b);
                const __m256i ax_1 = _mm256_shuffle_epi8(abs256, nib_1);
                const __m256i ax_2 = _mm256_shuffle_epi8(abs256, nib_2);
                const __m256i neg_1 = _mm256_cmpgt_epi8(bit3, nib_1);
                const __m256i neg_2 = _mm256_cmpgt_epi8(bit3, nib_2);
                const __m256i sy_1 = _mm256_sub_epi8(_mm256_xor_si256(q8v[ib + 0], neg_1), neg_1);
                const __m256i sy_2 = _mm256_sub_epi8(_mm256_xor_si256(q8v[ib + 1], neg_2), neg_2);
                const int16_t ls1 = (int16_t) (((bx.scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32);
                const int16_t ls2 = (int16_t) (((bx.scales_l[ib / 2] >>  4) | ((sh << 2) & 0x30)) - 32);
                sh >>= 4;
                sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(_mm256_maddubs_epi16(ax_1, sy_1), _mm256_set1_epi16(ls1)));
                sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(_mm256_maddubs_epi16(ax_2, sy_2), _mm256_set1_epi16(ls2)));
            }
            accum[r] = _mm256_fmadd_ps(_mm256_set1_ps(h2f(bx.d) * yd),
                                       _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = hsum_float_8(accum[r]);
}

void dot_q8_0_q8_0_x4_avx2(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
    assert(n % QK8_0 == 0);
    const int nb = n / QK8_0;
    const uint8_t* wb = (const uint8_t*) w_base;
    const block_q8_0* rows[4] = {
        (const block_q8_0*) (wb + 0 * row_bytes),
        (const block_q8_0*) (wb + 1 * row_bytes),
        (const block_q8_0*) (wb + 2 * row_bytes),
        (const block_q8_0*) (wb + 3 * row_bytes),
    };
    const block_q8_0* STRATA_RESTRICT y = (const block_q8_0*) vy;

    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};

    for (int ib = 0; ib < nb; ++ib) {
        if (ib + 4 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][ib + 4], _MM_HINT_T0);
        }
        const float dy = h2f(y[ib].d);
        const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);
        for (int r = 0; r < 4; ++r) {
            const __m256i qx = _mm256_loadu_si256((const __m256i*) rows[r][ib].qs);
            const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(qx, qx), _mm256_sign_epi8(qy, qx));
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(h2f(rows[r][ib].d) * dy),
                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, p16)), acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = hsum_float_8(acc[r]);
}

}  // namespace avx2

namespace {
#define ENTRY(T, ACT, NAME)     {avx2::dot_##NAME,         Type::T, Type::ACT, Arith::Avx2, Isa::Avx2, "avx2_" #NAME}
#define ENTRY_OPT(T, ACT, NAME) {avx2::dot_##NAME##_opt,   Type::T, Type::ACT, Arith::Avx2, Isa::Avx2, "avx2_" #NAME}
#define ENTRY_X4(T, ACT, NAME)  {avx2::dot_##NAME##_x4_avx2, Type::T, Type::ACT, Arith::Avx2, Isa::Avx2, "avx2_" #NAME "_x4"}
const DotKernel kTable[] = {
    ENTRY(Q4_0, Q8_0, q4_0_q8_0),   ENTRY(Q4_1, Q8_1, q4_1_q8_1),   ENTRY(Q5_0, Q8_0, q5_0_q8_0),   ENTRY(Q5_1, Q8_1, q5_1_q8_1),
    ENTRY_OPT(Q8_0, Q8_0, q8_0_q8_0), ENTRY(IQ4_NL, Q8_0, iq4_nl_q8_0), ENTRY(MXFP4, Q8_0, mxfp4_q8_0), ENTRY(NVFP4, Q8_0, nvfp4_q8_0),
    ENTRY(Q2_K, Q8_K, q2_K_q8_K),   ENTRY(Q3_K, Q8_K, q3_K_q8_K),   ENTRY_OPT(Q4_K, Q8_K, q4_K_q8_K), ENTRY_OPT(Q5_K, Q8_K, q5_K_q8_K),
    ENTRY_OPT(Q6_K, Q8_K, q6_K_q8_K), ENTRY(IQ1_S, Q8_K, iq1_s_q8_K), ENTRY(IQ1_M, Q8_K, iq1_m_q8_K), ENTRY(IQ2_XXS, Q8_K, iq2_xxs_q8_K),
    ENTRY(IQ2_XS, Q8_K, iq2_xs_q8_K), ENTRY(IQ2_S, Q8_K, iq2_s_q8_K), ENTRY_OPT(IQ3_XXS, Q8_K, iq3_xxs_q8_K), ENTRY(IQ3_S, Q8_K, iq3_s_q8_K),
    ENTRY_OPT(IQ4_XS, Q8_K, iq4_xs_q8_K),
    {generic::dot_q2_0_q8_0_generic, Type::Q2_0, Type::Q8_0, Arith::Avx2, Isa::Scalar, "avx2_q2_0_generic"},
};
const DotX4Kernel kX4Table[] = {
    ENTRY_X4(Q8_0, Q8_0, q8_0_q8_0),
    ENTRY_X4(Q4_K, Q8_K, q4_K_q8_K),
    ENTRY_X4(IQ4_XS, Q8_K, iq4_xs_q8_K),
};
#undef ENTRY_X4
#undef ENTRY_OPT
#undef ENTRY
}  // namespace

DotTable avx2_dot_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }
DotX4Table avx2_dot_x4_table() noexcept { return {kX4Table, (int) (sizeof kX4Table / sizeof kX4Table[0])}; }

}  // namespace strata::quant
