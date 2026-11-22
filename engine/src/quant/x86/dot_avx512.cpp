// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// src/quant/x86/dot_avx512.cpp - the Avx2 contract on CPUs with AVX-512 F/DQ/BW/VL + VNNI:
//   1. Ported ggml x86 dots whose mul_sum_us8_pairs_float helper takes VPDPBUSD (`__AVX512VNNI__ && __AVX512VL__`), and
//   2. Dedicated 512-bit ZMM AVX512BW + AVX512_VNNI dots for Q4_K, IQ4_XS, IQ3_S, and IQ3_XXS (`*_zmm`), preserving the exact
//      8-lane int32 sums and float superblock accumulation order of dot_avx2.cpp (0 ULP diff against Arith::Avx2, zero VBMI instructions).
// Compiled with the AVX-512 flags of cmake/backends/quant.cmake; only ever called when caps() says AVX-512 F/DQ/BW/VL + VNNI (Isa::Avx512).
#include "tier_common.hpp"

#include "../generic.hpp"
#include "../registry.hpp"
#include "strata/quant/dot.hpp"

// MSVC cl.exe under /arch:AVX512 exposes _mm256_dpbusd_epi32 and _mm512_dpbusd_epi32 in <immintrin.h> without predefining
// __AVX512VNNI__; define it for this TU (which is gated at runtime by caps() checking CPUID AVX512_VNNI).
#if defined(_MSC_VER) && !defined(__clang__) && defined(__AVX512VL__) && !defined(__AVX512VNNI__)
#define __AVX512VNNI__ 1
#endif

#if !defined(_MSC_VER) && (!defined(__AVX512VNNI__) || !defined(__AVX512VL__) || !defined(__AVX512BW__))
#error "dot_avx512.cpp needs AVX-512 BW, VL, and VNNI"
#endif

namespace strata::quant {
namespace avx512 {

#include "x86_helpers.inc"
#include "x86_dots.inc"

// ---- 512-bit ZMM AVX512BW + AVX512_VNNI bit-exact kernels (Arith::Avx2 float order, zero VBMI instructions) ----

void dot_q4_K_q8_K_zmm(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);

    const block_q4_K* STRATA_RESTRICT x = (const block_q4_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const __m512i m4 = _mm512_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
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

        const __m512i scales512 = _mm512_broadcast_i32x4(_mm256_castsi256_si128(mins_and_scales));
        __m512i sumi512 = _mm512_setzero_si512();

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m512i shuf = _mm512_inserti64x4(_mm512_castsi256_si512(get_scale_shuffle_k4(2 * j + 0)),
                                                    get_scale_shuffle_k4(2 * j + 1), 1);
            const __m512i scale_lh = _mm512_shuffle_epi8(scales512, shuf);

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            const __m512i q4_lh = _mm512_and_si512(
                _mm512_inserti64x4(_mm512_castsi256_si512(q4bits), _mm256_srli_epi16(q4bits, 4), 1), m4);
            const __m512i q8_lh = _mm512_loadu_si512((const __m512i*) q8);
            q8 += 64;

            const __m512i p16 = _mm512_maddubs_epi16(q4_lh, q8_lh);
            sumi512 = _mm512_add_epi32(sumi512, _mm512_madd_epi16(scale_lh, p16));
        }

        const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

void dot_q5_K_q8_K_zmm(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);

    const block_q5_K* STRATA_RESTRICT x = (const block_q5_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i m16 = _mm256_set1_epi8(16);

    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;

    for (int i = 0; i < nb; ++i) {
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

        const __m512i scales512 = _mm512_broadcast_i32x4(_mm256_castsi256_si128(mins_and_scales));
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);
        __m512i sumi512 = _mm512_setzero_si512();

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m512i shuf = _mm512_inserti64x4(_mm512_castsi256_si512(get_scale_shuffle_k4(2 * j + 0)),
                                                    get_scale_shuffle_k4(2 * j + 1), 1);
            const __m512i scale_lh = _mm512_shuffle_epi8(scales512, shuf);

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) q5);
            q5 += 32;
            const __m512i q5_lo = _mm512_and_si512(
                _mm512_inserti64x4(_mm512_castsi256_si512(q5bits), _mm256_srli_epi16(q5bits, 4), 1), m4);

            const __m256i hb0 = _mm256_and_si256(_mm256_slli_epi16(_mm256_srli_epi16(hbits, 2 * j + 0), 4), m16);
            const __m256i hb1 = _mm256_and_si256(_mm256_slli_epi16(_mm256_srli_epi16(hbits, 2 * j + 1), 4), m16);
            const __m512i q5_lh = _mm512_or_si512(q5_lo, _mm512_inserti64x4(_mm512_castsi256_si512(hb0), hb1, 1));

            const __m512i q8_lh = _mm512_loadu_si512((const __m512i*) q8);
            q8 += 64;

            const __m512i p16 = _mm512_maddubs_epi16(q5_lh, q8_lh);
            sumi512 = _mm512_add_epi32(sumi512, _mm512_madd_epi16(scale_lh, p16));
        }

        const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc) + summs;
}

void dot_q6_K_q8_K_zmm(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);

    const block_q6_K* STRATA_RESTRICT x = (const block_q6_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    const __m512i m15 = _mm512_set1_epi8(15);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m192 = _mm256_set1_epi8((char) 0xC0);

    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * h2f(x[i].d);
        const uint8_t* STRATA_RESTRICT q4 = x[i].ql;
        const uint8_t* STRATA_RESTRICT qh = x[i].qh;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i*) x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m512i sumi512 = _mm512_setzero_si512();
        int is = 0;
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m512i q4bits12 = _mm512_loadu_si512((const __m512i*) q4);
            q4 += 64;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) qh);
            qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m12), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, m48);
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, m192), 2);

            const __m512i q4_01 = _mm512_or_si512(_mm512_and_si512(q4bits12, m15),
                                                  _mm512_inserti64x4(_mm512_castsi256_si512(q4h_0), q4h_1, 1));
            const __m512i q4_23 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(q4bits12, 4), m15),
                                                  _mm512_inserti64x4(_mm512_castsi256_si512(q4h_2), q4h_3, 1));

            const __m512i q8_01 = _mm512_loadu_si512((const __m512i*) q8);
            const __m512i q8_23 = _mm512_loadu_si512((const __m512i*) (q8 + 64));
            q8 += 128;

            const __m512i p16_01 = _mm512_maddubs_epi16(q4_01, q8_01);
            const __m512i p16_23 = _mm512_maddubs_epi16(q4_23, q8_23);

            const __m256i sc_01 = _mm256_inserti128_si256(
                _mm256_castsi128_si256(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 0))),
                _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1)), 1);
            const __m256i sc_23 = _mm256_inserti128_si256(
                _mm256_castsi128_si256(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 2))),
                _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3)), 1);
            is += 4;

            sumi512 = _mm512_add_epi32(sumi512, _mm512_madd_epi16(_mm512_cvtepi8_epi16(sc_01), p16_01));
            sumi512 = _mm512_add_epi32(sumi512, _mm512_madd_epi16(_mm512_cvtepi8_epi16(sc_23), p16_23));
        }

        __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);
}

void dot_iq4_xs_q8_K_zmm(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);

    const block_iq4_xs* STRATA_RESTRICT x = (const block_iq4_xs*) vx;
    const block_q8_K*   STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    // |kvalues_iq4nl[k]|: entries 0..7 are negative (-127..-10), entries 8..15 are positive (+1..+113).
    // Testing bit 3 of the 4-bit nibble directly gives the sign mask in parallel with the abs-LUT shuffle,
    // eliminating _mm512_abs_epi8 + _mm512_movepi8_mask; _mm512_maddubs_epi16 + _mm512_madd_epi16 eliminates vpmulld.
    static constexpr uint8_t kAbsIq4nl[16] = {127, 104, 83, 65, 49, 35, 22, 10, 1, 13, 25, 38, 53, 69, 89, 113};
    const __m512i abs_values512 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kAbsIq4nl));
    const __m512i m4b = _mm512_set1_epi8(0x0F);
    const __m512i bit3 = _mm512_set1_epi8(0x08);
    const __m512i zero = _mm512_setzero_si512();

    __m256 accum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t* STRATA_RESTRICT qs = x[ibl].qs;
        const int8_t*  STRATA_RESTRICT q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m512i sumi512 = zero;

        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*) qs);
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*) (qs + 16));
            qs += 32;

            const __m256i nib_1 = MM256_SET_M128I(_mm_srli_epi16(q4bits_1, 4), q4bits_1);
            const __m256i nib_2 = MM256_SET_M128I(_mm_srli_epi16(q4bits_2, 4), q4bits_2);
            const __m512i nib512 = _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(nib_1), nib_2, 1), m4b);
            const __m512i ax = _mm512_shuffle_epi8(abs_values512, nib512);
            const __mmask64 neg = _mm512_testn_epi8_mask(nib512, bit3);

            const __m512i q8b = _mm512_loadu_si512((const __m512i*) q8);
            q8 += 64;

            const __m512i sy = _mm512_mask_sub_epi8(q8b, neg, zero, q8b);
            const __m512i p16 = _mm512_maddubs_epi16(ax, sy);

            const int16_t ls1 = (int16_t) (((x[ibl].scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const int16_t ls2 = (int16_t) (((x[ibl].scales_l[ib / 2] >>  4) | ((sh << 2) & 0x30)) - 32);
            sh >>= 4;

            const __m512i sc16 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(ls1)),
                                                    _mm256_set1_epi16(ls2), 1);
            sumi512 = _mm512_add_epi32(sumi512, _mm512_madd_epi16(p16, sc16));
        }

        const __m256i sum12 = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
        accum = _mm256_fmadd_ps(_mm256_set1_ps(h2f(x[ibl].d) * y[ibl].d), _mm256_cvtepi32_ps(sum12), accum);
    }

    *s = hsum_float_8(accum);
}

void dot_iq3_xxs_q8_K_zmm(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);

    const block_iq3_xxs* STRATA_RESTRICT x = (const block_iq3_xxs*) vx;
    const block_q8_K*    STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    const uint64_t* signs64 = (const uint64_t*) keven_signs_q2xs;
    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = h2f(x[i].d) * y[i].d;
        const uint8_t* STRATA_RESTRICT q3 = x[i].qs;
        const uint8_t* STRATA_RESTRICT gas = x[i].qs + QK_K / 4;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;
        __m512i sumi512 = _mm512_setzero_si512();

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            q8 += 64;

            const __m256i q2_lo = _mm256_set_epi32(
                (int) iq3xxs_grid[q3[ 7]], (int) iq3xxs_grid[q3[ 6]], (int) iq3xxs_grid[q3[ 5]], (int) iq3xxs_grid[q3[ 4]],
                (int) iq3xxs_grid[q3[ 3]], (int) iq3xxs_grid[q3[ 2]], (int) iq3xxs_grid[q3[ 1]], (int) iq3xxs_grid[q3[ 0]]);
            const __m256i q2_hi = _mm256_set_epi32(
                (int) iq3xxs_grid[q3[15]], (int) iq3xxs_grid[q3[14]], (int) iq3xxs_grid[q3[13]], (int) iq3xxs_grid[q3[12]],
                (int) iq3xxs_grid[q3[11]], (int) iq3xxs_grid[q3[10]], (int) iq3xxs_grid[q3[ 9]], (int) iq3xxs_grid[q3[ 8]]);
            const __m512i q2_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q2_lo), q2_hi, 1);
            q3 += 16;

            std::memcpy(aux32, gas, 8);
            gas += 8;
            // Use the 64-bit vector sign table (4 loads per 32-elem half, no scalar 8-shift chain or GPR->kmovq stall)
            const __m256i s2_1 = _mm256_set_epi64x(
                (long long) signs64[(aux32[0] >> 21) & 127], (long long) signs64[(aux32[0] >> 14) & 127],
                (long long) signs64[(aux32[0] >>  7) & 127], (long long) signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(
                (long long) signs64[(aux32[1] >> 21) & 127], (long long) signs64[(aux32[1] >> 14) & 127],
                (long long) signs64[(aux32[1] >>  7) & 127], (long long) signs64[(aux32[1] >>  0) & 127]);
            const __m512i q8s_512 = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_sign_epi8(q8_1, s2_1)),
                _mm256_sign_epi8(q8_2, s2_2), 1);

            const __m512i p16 = _mm512_maddubs_epi16(q2_512, q8s_512);

            const int16_t s1 = (int16_t) (2 * (aux32[0] >> 28) + 1);
            const int16_t s2 = (int16_t) (2 * (aux32[1] >> 28) + 1);
            const __m512i sc16 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(s1)),
                                                    _mm256_set1_epi16(s2), 1);
            sumi512 = _mm512_add_epi32(sumi512, _mm512_madd_epi16(p16, sc16));
        }

        const __m256i sum12 = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum12), accumf);
    }

    *s = 0.25f * hsum_float_8(accumf);
}

void dot_iq3_s_q8_K_zmm(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);

    const block_iq3_s* STRATA_RESTRICT x = (const block_iq3_s*) vx;
    const block_q8_K*  STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    const __m512i zero = _mm512_setzero_si512();

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = h2f(x[i].d) * y[i].d;
        const uint8_t* STRATA_RESTRICT qs = x[i].qs;
        const uint8_t* STRATA_RESTRICT qh = x[i].qh;
        const uint8_t* STRATA_RESTRICT signs = x[i].signs;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;
        __m512i sumi512 = zero;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m512i q8_512 = _mm512_loadu_si512((const __m512i*) q8);
            q8 += 64;

            const uint32_t h0 = qh[ib32 + 0];
            const uint32_t h1 = qh[ib32 + 1];
            const __m256i q2_lo = _mm256_set_epi32(
                (int) iq3s_grid[qs[ 7] | ((h0 << 1) & 256u)], (int) iq3s_grid[qs[ 6] | ((h0 << 2) & 256u)],
                (int) iq3s_grid[qs[ 5] | ((h0 << 3) & 256u)], (int) iq3s_grid[qs[ 4] | ((h0 << 4) & 256u)],
                (int) iq3s_grid[qs[ 3] | ((h0 << 5) & 256u)], (int) iq3s_grid[qs[ 2] | ((h0 << 6) & 256u)],
                (int) iq3s_grid[qs[ 1] | ((h0 << 7) & 256u)], (int) iq3s_grid[qs[ 0] | ((h0 << 8) & 256u)]);
            const __m256i q2_hi = _mm256_set_epi32(
                (int) iq3s_grid[qs[15] | ((h1 << 1) & 256u)], (int) iq3s_grid[qs[14] | ((h1 << 2) & 256u)],
                (int) iq3s_grid[qs[13] | ((h1 << 3) & 256u)], (int) iq3s_grid[qs[12] | ((h1 << 4) & 256u)],
                (int) iq3s_grid[qs[11] | ((h1 << 5) & 256u)], (int) iq3s_grid[qs[10] | ((h1 << 6) & 256u)],
                (int) iq3s_grid[qs[ 9] | ((h1 << 7) & 256u)], (int) iq3s_grid[qs[ 8] | ((h1 << 8) & 256u)]);
            const __m512i q2_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q2_lo), q2_hi, 1);
            qs += 16;

            uint64_t s64;
            std::memcpy(&s64, signs, 8);
            signs += 8;

            const __m512i q8s_512 = _mm512_mask_sub_epi8(q8_512, _cvtu64_mask64(s64), zero, q8_512);
            const __m512i p16 = _mm512_maddubs_epi16(q2_512, q8s_512);

            const int16_t s1 = (int16_t) (2 * (x[i].scales[ib32 / 2] & 0xf) + 1);
            const int16_t s2 = (int16_t) (2 * (x[i].scales[ib32 / 2] >>  4) + 1);
            const __m512i sc16 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(s1)),
                                                    _mm256_set1_epi16(s2), 1);
            sumi512 = _mm512_add_epi32(sumi512, _mm512_madd_epi16(p16, sc16));
        }

        const __m256i sum12 = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum12), accumf);
    }

    *s = hsum_float_8(accumf);
}

}  // namespace avx512

namespace {
#define ENTRY(T, ACT, NAME)     {avx512::dot_##NAME,     Type::T, Type::ACT, Arith::Avx2, Isa::Avx512, "avx512_" #NAME}
#define ENTRY_ZMM(T, ACT, NAME) {avx512::dot_##NAME##_zmm, Type::T, Type::ACT, Arith::Avx2, Isa::Avx512, "avx512_" #NAME}
const DotKernel kTable[] = {
    ENTRY(Q4_0, Q8_0, q4_0_q8_0), ENTRY(Q4_1, Q8_1, q4_1_q8_1), ENTRY(Q5_0, Q8_0, q5_0_q8_0), ENTRY(Q5_1, Q8_1, q5_1_q8_1),
    ENTRY(Q8_0, Q8_0, q8_0_q8_0), ENTRY(IQ4_NL, Q8_0, iq4_nl_q8_0), ENTRY(MXFP4, Q8_0, mxfp4_q8_0),
    ENTRY(NVFP4, Q8_0, nvfp4_q8_0),
    ENTRY_ZMM(Q4_K, Q8_K, q4_K_q8_K), ENTRY_ZMM(Q5_K, Q8_K, q5_K_q8_K), ENTRY_ZMM(Q6_K, Q8_K, q6_K_q8_K),
    ENTRY_ZMM(IQ4_XS, Q8_K, iq4_xs_q8_K), ENTRY_ZMM(IQ3_S, Q8_K, iq3_s_q8_K),
};
#undef ENTRY_ZMM
#undef ENTRY
}  // namespace

DotTable avx512_dot_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }

}  // namespace strata::quant
