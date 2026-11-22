// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// src/quant/x86/dot_avx512.cpp - the Avx2 contract on CPUs with AVX-512 F/DQ/BW/VL + VNNI:
//   1. Ported ggml x86 dots whose mul_sum_us8_pairs_float helper takes VPDPBUSD (`__AVX512VNNI__ && __AVX512VL__`), and
//   2. Dedicated 512-bit ZMM AVX512BW + AVX512_VNNI dots for Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_XXS, and IQ3_S (`*_zmm`),
//      using VPDPWSSD/VPDPBUSD, dual accumulators, precomputed 64B scale shuffles, and L1 prefetching while preserving
//      the exact 8-lane int32 sums and float superblock accumulation order of dot_avx2.cpp (0 ULP diff, zero VBMI).
//   3. 4-row shared-activation VPDPBUSD/VPDPWSSD (`dot_*_x4`) kernels for Q4_0, MXFP4, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_XXS.
// Compiled with the AVX-512 flags of cmake/backends/quant.cmake; only ever called when caps() says AVX-512 F/DQ/BW/VL + VNNI (Isa::Avx512).
#include "tier_common.hpp"

#include "../generic.hpp"
#include "../registry.hpp"
#include "strata/quant/dot.hpp"

// MSVC cl.exe under /arch:AVX512 exposes _mm256_dpbusd_epi32, _mm512_dpbusd_epi32, and _mm512_dpwssd_epi32 in <immintrin.h>
// without predefining __AVX512VNNI__; define it for this TU (gated at runtime by caps() checking CPUID AVX512_VNNI).
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

// Precomputed 64-byte aligned ZMM scale-shuffle masks for Q4_K / Q5_K (eliminates two 256b loads + vinserti64x4 per iter).
alignas(64) static constexpr uint8_t kScaleShuffleK4Zmm[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,  2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,  4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,  6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,  8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11, 10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13, 12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15, 14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
};

// Precomputed 64-byte high-bit test masks for Q5_K (low 32B tests bit 2*j, high 32B tests bit 2*j+1).
static inline __m512i q5k_hbit_mask_zmm(int j) {
    const __m256i lo = _mm256_set1_epi8((char) (1u << (2 * j + 0)));
    const __m256i hi = _mm256_set1_epi8((char) (1u << (2 * j + 1)));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

// ---- 512-bit ZMM AVX512BW + AVX512_VNNI bit-exact kernels (Arith::Avx2 float order, zero VBMI instructions) ----

void dot_q8_0_q8_0_zmm(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK8_0 == 0);
    const int nb = n / QK8_0;
    const block_q8_0* STRATA_RESTRICT x = (const block_q8_0*) vx;
    const block_q8_0* STRATA_RESTRICT y = (const block_q8_0*) vy;

    // 2-way unrolled 256-bit AVX-VNNI: avoids 512-bit cross-lane insert/extract and k-mask transition penalties on Zen 4
    // while keeping exact per-block float accumulation order (0 ULP diff vs dot_avx2.cpp).
    const __m256i zero256 = _mm256_setzero_si256();
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

        const __m256 q0 = _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(zero256, _mm256_sign_epi8(qx0, qx0), _mm256_sign_epi8(qy0, qx0)));
        const __m256 q1 = _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(zero256, _mm256_sign_epi8(qx1, qx1), _mm256_sign_epi8(qy1, qx1)));

        acc = _mm256_fmadd_ps(d0, q0, acc);
        acc = _mm256_fmadd_ps(d1, q1, acc);
    }
    for (; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(h2f(x[ib].d) * h2f(y[ib].d));
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);
        const __m256 q = _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(zero256, _mm256_sign_epi8(qx, qx), _mm256_sign_epi8(qy, qx)));
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    *s = hsum_float_8(acc);
}

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
    const __m512i shuf0 = _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm +   0));
    const __m512i shuf1 = _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm +  64));
    const __m512i shuf2 = _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm + 128));
    const __m512i shuf3 = _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm + 192));
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

        const __m512i scales512 = _mm512_broadcast_i32x4(_mm256_castsi256_si128(mins_and_scales));
        __m512i sumi0 = _mm512_setzero_si512();
        __m512i sumi1 = _mm512_setzero_si512();

#define Q4K_ZMM_STEP(J, SHUF, ACC) do { \
            const __m512i scale_lh = _mm512_shuffle_epi8(scales512, (SHUF)); \
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) (q4 + 32 * (J))); \
            const __m512i q4_lh = _mm512_and_si512( \
                _mm512_inserti64x4(_mm512_castsi256_si512(q4bits), _mm256_srli_epi16(q4bits, 4), 1), m4); \
            const __m512i q8_lh = _mm512_loadu_si512((const __m512i*) (q8 + 64 * (J))); \
            const __m512i p16 = _mm512_maddubs_epi16(q4_lh, q8_lh); \
            (ACC) = _mm512_dpwssd_epi32((ACC), scale_lh, p16); \
        } while (0)
        Q4K_ZMM_STEP(0, shuf0, sumi0);
        Q4K_ZMM_STEP(1, shuf1, sumi1);
        Q4K_ZMM_STEP(2, shuf2, sumi0);
        Q4K_ZMM_STEP(3, shuf3, sumi1);
#undef Q4K_ZMM_STEP

        const __m512i sumi512 = _mm512_add_epi32(sumi0, sumi1);
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
    const __m512i m16 = _mm512_set1_epi8(16);
    const __m128i mzero = _mm_setzero_si128();
    const __m512i shuf0 = _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm +   0));
    const __m512i shuf1 = _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm +  64));
    const __m512i shuf2 = _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm + 128));
    const __m512i shuf3 = _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm + 192));
    const __m512i hm0 = q5k_hbit_mask_zmm(0);
    const __m512i hm1 = q5k_hbit_mask_zmm(1);
    const __m512i hm2 = q5k_hbit_mask_zmm(2);
    const __m512i hm3 = q5k_hbit_mask_zmm(3);

    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;

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
        summs = _mm_cvtss_f32(_mm_add_ss(_mm_set_ss(summs),
                              _mm_mul_ss(_mm_set_ss(dmin), _mm_cvtsi32_ss(_mm_setzero_ps(), _mm_extract_epi32(hsum, 0)))));

        const __m512i scales512 = _mm512_broadcast_i32x4(_mm256_castsi256_si128(mins_and_scales));
        const __m512i hbits512 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) x[i].qh));
        __m512i sumi0 = _mm512_setzero_si512();
        __m512i sumi1 = _mm512_setzero_si512();

#define Q5K_ZMM_STEP(J, SHUF, HMASK, ACC) do { \
            const __m512i scale_lh = _mm512_shuffle_epi8(scales512, (SHUF)); \
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) (q5 + 32 * (J))); \
            const __m512i q5_lo = _mm512_and_si512( \
                _mm512_inserti64x4(_mm512_castsi256_si512(q5bits), _mm256_srli_epi16(q5bits, 4), 1), m4); \
            const __mmask64 hb = _mm512_test_epi8_mask(hbits512, (HMASK)); \
            const __m512i q5_lh = _mm512_mask_add_epi8(q5_lo, hb, q5_lo, m16); \
            const __m512i q8_lh = _mm512_loadu_si512((const __m512i*) (q8 + 64 * (J))); \
            const __m512i p16 = _mm512_maddubs_epi16(q5_lh, q8_lh); \
            (ACC) = _mm512_dpwssd_epi32((ACC), scale_lh, p16); \
        } while (0)
        Q5K_ZMM_STEP(0, shuf0, hm0, sumi0);
        Q5K_ZMM_STEP(1, shuf1, hm1, sumi1);
        Q5K_ZMM_STEP(2, shuf2, hm2, sumi0);
        Q5K_ZMM_STEP(3, shuf3, hm3, sumi1);
#undef Q5K_ZMM_STEP

        const __m512i sumi512 = _mm512_add_epi32(sumi0, sumi1);
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

        __m512i sumi0 = _mm512_setzero_si512();
        __m512i sumi1 = _mm512_setzero_si512();
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

            sumi0 = _mm512_dpwssd_epi32(sumi0, _mm512_cvtepi8_epi16(sc_01), p16_01);
            sumi1 = _mm512_dpwssd_epi32(sumi1, _mm512_cvtepi8_epi16(sc_23), p16_23);
        }

        const __m512i sumi512 = _mm512_add_epi32(sumi0, sumi1);
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

    static constexpr uint8_t kAbsIq4nl[16] = {127, 104, 83, 65, 49, 35, 22, 10, 1, 13, 25, 38, 53, 69, 89, 113};
    const __m512i abs_values512 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kAbsIq4nl));
    const __m512i m4b = _mm512_set1_epi8(0x0F);
    const __m512i bit3 = _mm512_set1_epi8(0x08);
    const __m512i zero = _mm512_setzero_si512();

    __m256 accum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        if (ibl + 2 < nb) {
            _mm_prefetch((const char*) &x[ibl + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[ibl + 2], _MM_HINT_T0);
        }
        const uint8_t* STRATA_RESTRICT qs = x[ibl].qs;
        const int8_t*  STRATA_RESTRICT q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m512i sumi0 = zero;
        __m512i sumi1 = zero;

#define IQ4XS_ZMM_STEP(IB, ACC) do { \
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*) (qs + 16 * (IB))); \
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*) (qs + 16 * (IB) + 16)); \
            const __m256i nib_1 = MM256_SET_M128I(_mm_srli_epi16(q4bits_1, 4), q4bits_1); \
            const __m256i nib_2 = MM256_SET_M128I(_mm_srli_epi16(q4bits_2, 4), q4bits_2); \
            const __m512i nib512 = _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(nib_1), nib_2, 1), m4b); \
            const __m512i ax = _mm512_shuffle_epi8(abs_values512, nib512); \
            const __mmask64 neg = _mm512_testn_epi8_mask(nib512, bit3); \
            const __m512i q8b = _mm512_loadu_si512((const __m512i*) (q8 + 32 * (IB))); \
            const __m512i sy = _mm512_mask_sub_epi8(q8b, neg, zero, q8b); \
            const __m512i p16 = _mm512_maddubs_epi16(ax, sy); \
            const int16_t ls1 = (int16_t) (((x[ibl].scales_l[(IB) / 2] & 0xf) | ((sh << 4) & 0x30)) - 32); \
            const int16_t ls2 = (int16_t) (((x[ibl].scales_l[(IB) / 2] >>  4) | ((sh << 2) & 0x30)) - 32); \
            sh >>= 4; \
            const __m512i sc16 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(ls1)), \
                                                    _mm256_set1_epi16(ls2), 1); \
            (ACC) = _mm512_dpwssd_epi32((ACC), p16, sc16); \
        } while (0)
        IQ4XS_ZMM_STEP(0, sumi0);
        IQ4XS_ZMM_STEP(2, sumi1);
        IQ4XS_ZMM_STEP(4, sumi0);
        IQ4XS_ZMM_STEP(6, sumi1);
#undef IQ4XS_ZMM_STEP

        const __m512i sumi512 = _mm512_add_epi32(sumi0, sumi1);
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
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const float d = h2f(x[i].d) * y[i].d;
        const uint8_t* STRATA_RESTRICT q3 = x[i].qs;
        const uint8_t* STRATA_RESTRICT gas = x[i].qs + QK_K / 4;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;
        __m512i sumi0 = _mm512_setzero_si512();
        __m512i sumi1 = _mm512_setzero_si512();

#define IQ3XXS_ZMM_STEP(IB32, ACC) do { \
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32 * (IB32))); \
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32 * (IB32) + 32)); \
            const uint8_t* p3 = q3 + 8 * (IB32); \
            const __m256i q2_lo = _mm256_set_epi32( \
                (int) iq3xxs_grid[p3[ 7]], (int) iq3xxs_grid[p3[ 6]], (int) iq3xxs_grid[p3[ 5]], (int) iq3xxs_grid[p3[ 4]], \
                (int) iq3xxs_grid[p3[ 3]], (int) iq3xxs_grid[p3[ 2]], (int) iq3xxs_grid[p3[ 1]], (int) iq3xxs_grid[p3[ 0]]); \
            const __m256i q2_hi = _mm256_set_epi32( \
                (int) iq3xxs_grid[p3[15]], (int) iq3xxs_grid[p3[14]], (int) iq3xxs_grid[p3[13]], (int) iq3xxs_grid[p3[12]], \
                (int) iq3xxs_grid[p3[11]], (int) iq3xxs_grid[p3[10]], (int) iq3xxs_grid[p3[ 9]], (int) iq3xxs_grid[p3[ 8]]); \
            const __m512i q2_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q2_lo), q2_hi, 1); \
            std::memcpy(aux32, gas + 4 * (IB32), 8); \
            const __m256i s2_1 = _mm256_set_epi64x( \
                (long long) signs64[(aux32[0] >> 21) & 127], (long long) signs64[(aux32[0] >> 14) & 127], \
                (long long) signs64[(aux32[0] >>  7) & 127], (long long) signs64[(aux32[0] >>  0) & 127]); \
            const __m256i s2_2 = _mm256_set_epi64x( \
                (long long) signs64[(aux32[1] >> 21) & 127], (long long) signs64[(aux32[1] >> 14) & 127], \
                (long long) signs64[(aux32[1] >>  7) & 127], (long long) signs64[(aux32[1] >>  0) & 127]); \
            const __m512i q8s_512 = _mm512_inserti64x4( \
                _mm512_castsi256_si512(_mm256_sign_epi8(q8_1, s2_1)), \
                _mm256_sign_epi8(q8_2, s2_2), 1); \
            const __m512i p16 = _mm512_maddubs_epi16(q2_512, q8s_512); \
            const int16_t s1 = (int16_t) (2 * (aux32[0] >> 28) + 1); \
            const int16_t s2 = (int16_t) (2 * (aux32[1] >> 28) + 1); \
            const __m512i sc16 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(s1)), \
                                                    _mm256_set1_epi16(s2), 1); \
            (ACC) = _mm512_dpwssd_epi32((ACC), p16, sc16); \
        } while (0)
        IQ3XXS_ZMM_STEP(0, sumi0);
        IQ3XXS_ZMM_STEP(2, sumi1);
        IQ3XXS_ZMM_STEP(4, sumi0);
        IQ3XXS_ZMM_STEP(6, sumi1);
#undef IQ3XXS_ZMM_STEP

        const __m512i sumi512 = _mm512_add_epi32(sumi0, sumi1);
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
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const float d = h2f(x[i].d) * y[i].d;
        const uint8_t* STRATA_RESTRICT qs = x[i].qs;
        const uint8_t* STRATA_RESTRICT qh = x[i].qh;
        const uint8_t* STRATA_RESTRICT signs = x[i].signs;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;
        __m512i sumi0 = zero;
        __m512i sumi1 = zero;

#define IQ3S_ZMM_STEP(IB32, ACC) do { \
            const __m512i q8_512 = _mm512_loadu_si512((const __m512i*) (q8 + 32 * (IB32))); \
            const uint8_t* ps = qs + 8 * (IB32); \
            const uint32_t h0 = qh[(IB32) + 0]; \
            const uint32_t h1 = qh[(IB32) + 1]; \
            const __m256i q2_lo = _mm256_set_epi32( \
                (int) iq3s_grid[ps[ 7] | ((h0 << 1) & 256u)], (int) iq3s_grid[ps[ 6] | ((h0 << 2) & 256u)], \
                (int) iq3s_grid[ps[ 5] | ((h0 << 3) & 256u)], (int) iq3s_grid[ps[ 4] | ((h0 << 4) & 256u)], \
                (int) iq3s_grid[ps[ 3] | ((h0 << 5) & 256u)], (int) iq3s_grid[ps[ 2] | ((h0 << 6) & 256u)], \
                (int) iq3s_grid[ps[ 1] | ((h0 << 7) & 256u)], (int) iq3s_grid[ps[ 0] | ((h0 << 8) & 256u)]); \
            const __m256i q2_hi = _mm256_set_epi32( \
                (int) iq3s_grid[ps[15] | ((h1 << 1) & 256u)], (int) iq3s_grid[ps[14] | ((h1 << 2) & 256u)], \
                (int) iq3s_grid[ps[13] | ((h1 << 3) & 256u)], (int) iq3s_grid[ps[12] | ((h1 << 4) & 256u)], \
                (int) iq3s_grid[ps[11] | ((h1 << 5) & 256u)], (int) iq3s_grid[ps[10] | ((h1 << 6) & 256u)], \
                (int) iq3s_grid[ps[ 9] | ((h1 << 7) & 256u)], (int) iq3s_grid[ps[ 8] | ((h1 << 8) & 256u)]); \
            const __m512i q2_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q2_lo), q2_hi, 1); \
            uint64_t s64; \
            std::memcpy(&s64, signs + 4 * (IB32), 8); \
            const __m512i q8s_512 = _mm512_mask_sub_epi8(q8_512, _cvtu64_mask64(s64), zero, q8_512); \
            const __m512i p16 = _mm512_maddubs_epi16(q2_512, q8s_512); \
            const int16_t s1 = (int16_t) (2 * (x[i].scales[(IB32) / 2] & 0xf) + 1); \
            const int16_t s2 = (int16_t) (2 * (x[i].scales[(IB32) / 2] >>  4) + 1); \
            const __m512i sc16 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(s1)), \
                                                    _mm256_set1_epi16(s2), 1); \
            (ACC) = _mm512_dpwssd_epi32((ACC), p16, sc16); \
        } while (0)
        IQ3S_ZMM_STEP(0, sumi0);
        IQ3S_ZMM_STEP(2, sumi1);
        IQ3S_ZMM_STEP(4, sumi0);
        IQ3S_ZMM_STEP(6, sumi1);
#undef IQ3S_ZMM_STEP

        const __m512i sumi512 = _mm512_add_epi32(sumi0, sumi1);
        const __m256i sum12 = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum12), accumf);
    }

    *s = hsum_float_8(accumf);
}

// ---- 4-row shared-activation VPDPBUSD/VPDPWSSD (`dot_*_x4`) kernels (MIT, bit-exact to Arith::Avx2 1-row dots) ----

void dot_q4_0_q8_0_x4(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
    assert(n % QK8_0 == 0);
    const int nb = n / QK8_0;
    const uint8_t* wb = (const uint8_t*) w_base;
    const block_q4_0* x0 = (const block_q4_0*) (wb + 0 * row_bytes);
    const block_q4_0* x1 = (const block_q4_0*) (wb + 1 * row_bytes);
    const block_q4_0* x2 = (const block_q4_0*) (wb + 2 * row_bytes);
    const block_q4_0* x3 = (const block_q4_0*) (wb + 3 * row_bytes);
    const block_q8_0* STRATA_RESTRICT y = (const block_q8_0*) vy;

    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i bias8 = _mm256_set1_epi8(8);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
        if (ib + 4 < nb) {
            _mm_prefetch((const char*) &x0[ib + 4], _MM_HINT_T0);
            _mm_prefetch((const char*) &x1[ib + 4], _MM_HINT_T0);
            _mm_prefetch((const char*) &x2[ib + 4], _MM_HINT_T0);
            _mm_prefetch((const char*) &x3[ib + 4], _MM_HINT_T0);
        }
        const float dy = h2f(y[ib].d);
        const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);
        const __m256i ybias = _mm256_dpbusd_epi32(_mm256_setzero_si256(), bias8, qy);

        auto step_row = [&](const block_q4_0& bx, __m256& acc) {
            const __m128i raw = _mm_loadu_si128((const __m128i*) bx.qs);
            const __m256i u = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(raw, 4), raw), m4);
            const __m256i prod = _mm256_sub_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), u, qy), ybias);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(h2f(bx.d) * dy), _mm256_cvtepi32_ps(prod), acc);
        };
        step_row(x0[ib], acc0);
        step_row(x1[ib], acc1);
        step_row(x2[ib], acc2);
        step_row(x3[ib], acc3);
    }
    out[0] = hsum_float_8(acc0);
    out[1] = hsum_float_8(acc1);
    out[2] = hsum_float_8(acc2);
    out[3] = hsum_float_8(acc3);
}

void dot_q8_0_q8_0_x4(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
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

    const __m512i zero512 = _mm512_setzero_si512();
    const __m256i zero256 = _mm256_setzero_si256();
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};

    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        if (ib + 4 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][ib + 4], _MM_HINT_T0);
        }
        const float dy0 = h2f(y[ib + 0].d);
        const float dy1 = h2f(y[ib + 1].d);
        const __m512i qy = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) y[ib + 0].qs)),
            _mm256_loadu_si256((const __m256i*) y[ib + 1].qs), 1);

        for (int r = 0; r < 4; ++r) {
            const __m512i qx = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) rows[r][ib + 0].qs)),
                _mm256_loadu_si256((const __m256i*) rows[r][ib + 1].qs), 1);
            const __m512i ax = _mm512_abs_epi8(qx);
            const __mmask64 neg = _mm512_movepi8_mask(qx);
            const __m512i sy = _mm512_mask_sub_epi8(qy, neg, zero512, qy);
            const __m512i p32 = _mm512_dpbusd_epi32(zero512, ax, sy);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(h2f(rows[r][ib + 0].d) * dy0), _mm256_cvtepi32_ps(_mm512_castsi512_si256(p32)), acc[r]);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(h2f(rows[r][ib + 1].d) * dy1), _mm256_cvtepi32_ps(_mm512_extracti64x4_epi64(p32, 1)), acc[r]);
        }
    }
    for (; ib < nb; ++ib) {
        const float dy = h2f(y[ib].d);
        const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);
        for (int r = 0; r < 4; ++r) {
            const __m256i qx = _mm256_loadu_si256((const __m256i*) rows[r][ib].qs);
            const __m256i ax = _mm256_sign_epi8(qx, qx);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(zero256, ax, sy));
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(h2f(rows[r][ib].d) * dy), q, acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = hsum_float_8(acc[r]);
}

void dot_mxfp4_q8_0_x4(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
    assert(n % QK_MXFP4 == 0);
    const int nb = n / QK_MXFP4;
    const uint8_t* wb = (const uint8_t*) w_base;
    const block_mxfp4* x0 = (const block_mxfp4*) (wb + 0 * row_bytes);
    const block_mxfp4* x1 = (const block_mxfp4*) (wb + 1 * row_bytes);
    const block_mxfp4* x2 = (const block_mxfp4*) (wb + 2 * row_bytes);
    const block_mxfp4* x3 = (const block_mxfp4*) (wb + 3 * row_bytes);
    const block_q8_0* STRATA_RESTRICT y = (const block_q8_0*) vy;

    static constexpr uint8_t kFp4Plus12[16] = {
        12, 13, 14, 15, 16, 18, 20, 24,
        12, 11, 10,  9,  8,  6,  4,  0,
    };
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kFp4Plus12));
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i c12 = _mm256_set1_epi8(12);

    __m256 acc1[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    __m256 acc2[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};

    for (int ib = 0; ib + 1 < nb; ib += 2) {
        if (ib + 4 < nb) {
            _mm_prefetch((const char*) &x0[ib + 4], _MM_HINT_T0);
            _mm_prefetch((const char*) &x1[ib + 4], _MM_HINT_T0);
            _mm_prefetch((const char*) &x2[ib + 4], _MM_HINT_T0);
            _mm_prefetch((const char*) &x3[ib + 4], _MM_HINT_T0);
        }
        const float dy0 = h2f(y[ib + 0].d);
        const float dy1 = h2f(y[ib + 1].d);
        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) y[ib + 0].qs);
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) y[ib + 1].qs);
        const __m256i bias0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), c12, q8_0);
        const __m256i bias1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), c12, q8_1);

        auto step_row = [&](const block_mxfp4* xr, int r) {
            const __m128i raw0 = _mm_loadu_si128((const __m128i*) xr[ib + 0].qs);
            const __m128i raw1 = _mm_loadu_si128((const __m128i*) xr[ib + 1].qs);
            const __m256i nib0 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(raw0, 4), raw0), m4b);
            const __m256i nib1 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(raw1, 4), raw1), m4b);
            const __m256i u0 = _mm256_shuffle_epi8(lut, nib0);
            const __m256i u1 = _mm256_shuffle_epi8(lut, nib1);
            const __m256i p0 = _mm256_sub_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), u0, q8_0), bias0);
            const __m256i p1 = _mm256_sub_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), u1, q8_1), bias1);
            const __m256 sc0 = _mm256_set1_ps(dy0 * e8m0_to_fp32_half(xr[ib + 0].e));
            const __m256 sc1 = _mm256_set1_ps(dy1 * e8m0_to_fp32_half(xr[ib + 1].e));
            acc1[r] = _mm256_fmadd_ps(sc0, _mm256_cvtepi32_ps(p0), acc1[r]);
            acc2[r] = _mm256_fmadd_ps(sc1, _mm256_cvtepi32_ps(p1), acc2[r]);
        };
        step_row(x0, 0);
        step_row(x1, 1);
        step_row(x2, 2);
        step_row(x3, 3);
    }
    for (int r = 0; r < 4; ++r) out[r] = hsum_float_8(_mm256_add_ps(acc1[r], acc2[r]));
}

void dot_q4_K_q8_K_x4(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
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
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i shufs[4] = {
        _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm +   0)),
        _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm +  64)),
        _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm + 128)),
        _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm + 192)),
    };
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    __m128 acc_m[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][i + 2], _MM_HINT_T0);
        }
        const float yd = y[i].d;
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m512i q8v[4] = {
            _mm512_loadu_si512((const __m512i*) (y[i].qs +   0)),
            _mm512_loadu_si512((const __m512i*) (y[i].qs +  64)),
            _mm512_loadu_si512((const __m512i*) (y[i].qs + 128)),
            _mm512_loadu_si512((const __m512i*) (y[i].qs + 192)),
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

            const __m512i scales512 = _mm512_broadcast_i32x4(_mm256_castsi256_si128(mins_and_scales));
            __m512i sumi0 = _mm512_setzero_si512();
            __m512i sumi1 = _mm512_setzero_si512();
            const uint8_t* q4 = bx.qs;
            for (int j = 0; j < 4; ++j) {
                const __m512i scale_lh = _mm512_shuffle_epi8(scales512, shufs[j]);
                const __m256i q4bits = _mm256_loadu_si256((const __m256i*) (q4 + 32 * j));
                const __m512i q4_lh = _mm512_and_si512(
                    _mm512_inserti64x4(_mm512_castsi256_si512(q4bits), _mm256_srli_epi16(q4bits, 4), 1), m4);
                const __m512i p16 = _mm512_maddubs_epi16(q4_lh, q8v[j]);
                if (j & 1) sumi1 = _mm512_dpwssd_epi32(sumi1, scale_lh, p16);
                else       sumi0 = _mm512_dpwssd_epi32(sumi0, scale_lh, p16);
            }
            const __m512i sumi512 = _mm512_add_epi32(sumi0, sumi1);
            const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) {
        acc_m[r] = _mm_add_ps(acc_m[r], _mm_movehl_ps(acc_m[r], acc_m[r]));
        acc_m[r] = _mm_add_ss(acc_m[r], _mm_movehdup_ps(acc_m[r]));
        out[r] = hsum_float_8(acc[r]) + _mm_cvtss_f32(acc_m[r]);
    }
}

void dot_q5_K_q8_K_x4(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;
    const uint8_t* wb = (const uint8_t*) w_base;
    const block_q5_K* rows[4] = {
        (const block_q5_K*) (wb + 0 * row_bytes),
        (const block_q5_K*) (wb + 1 * row_bytes),
        (const block_q5_K*) (wb + 2 * row_bytes),
        (const block_q5_K*) (wb + 3 * row_bytes),
    };
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i m16 = _mm512_set1_epi8(16);
    const __m128i mzero = _mm_setzero_si128();
    const __m512i shufs[4] = {
        _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm +   0)),
        _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm +  64)),
        _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm + 128)),
        _mm512_load_si512((const __m512i*) (kScaleShuffleK4Zmm + 192)),
    };
    const __m512i hms[4] = {q5k_hbit_mask_zmm(0), q5k_hbit_mask_zmm(1), q5k_hbit_mask_zmm(2), q5k_hbit_mask_zmm(3)};

    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    float summs[4] = {0.f, 0.f, 0.f, 0.f};

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][i + 2], _MM_HINT_T0);
        }
        const float yd = y[i].d;
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m512i q8v[4] = {
            _mm512_loadu_si512((const __m512i*) (y[i].qs +   0)),
            _mm512_loadu_si512((const __m512i*) (y[i].qs +  64)),
            _mm512_loadu_si512((const __m512i*) (y[i].qs + 128)),
            _mm512_loadu_si512((const __m512i*) (y[i].qs + 192)),
        };

        for (int r = 0; r < 4; ++r) {
            const block_q5_K& bx = rows[r][i];
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
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[r] = _mm_cvtss_f32(_mm_add_ss(_mm_set_ss(summs[r]),
                                     _mm_mul_ss(_mm_set_ss(dmin), _mm_cvtsi32_ss(_mm_setzero_ps(), _mm_extract_epi32(hsum, 0)))));

            const __m512i scales512 = _mm512_broadcast_i32x4(_mm256_castsi256_si128(mins_and_scales));
            const __m512i hbits512 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) bx.qh));
            __m512i sumi0 = _mm512_setzero_si512();
            __m512i sumi1 = _mm512_setzero_si512();
            const uint8_t* q5 = bx.qs;
            for (int j = 0; j < 4; ++j) {
                const __m512i scale_lh = _mm512_shuffle_epi8(scales512, shufs[j]);
                const __m256i q5bits = _mm256_loadu_si256((const __m256i*) (q5 + 32 * j));
                const __m512i q5_lo = _mm512_and_si512(
                    _mm512_inserti64x4(_mm512_castsi256_si512(q5bits), _mm256_srli_epi16(q5bits, 4), 1), m4);
                const __mmask64 hb = _mm512_test_epi8_mask(hbits512, hms[j]);
                const __m512i q5_lh = _mm512_mask_add_epi8(q5_lo, hb, q5_lo, m16);
                const __m512i p16 = _mm512_maddubs_epi16(q5_lh, q8v[j]);
                if (j & 1) sumi1 = _mm512_dpwssd_epi32(sumi1, scale_lh, p16);
                else       sumi0 = _mm512_dpwssd_epi32(sumi0, scale_lh, p16);
            }
            const __m512i sumi512 = _mm512_add_epi32(sumi0, sumi1);
            const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = hsum_float_8(acc[r]) + summs[r];
}

void dot_q6_K_q8_K_x4(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;
    const uint8_t* wb = (const uint8_t*) w_base;
    const block_q6_K* rows[4] = {
        (const block_q6_K*) (wb + 0 * row_bytes),
        (const block_q6_K*) (wb + 1 * row_bytes),
        (const block_q6_K*) (wb + 2 * row_bytes),
        (const block_q6_K*) (wb + 3 * row_bytes),
    };
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;

    const __m512i m15 = _mm512_set1_epi8(15);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m192 = _mm256_set1_epi8((char) 0xC0);

    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][i + 2], _MM_HINT_T0);
        }
        const float yd = y[i].d;
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m512i q8v[4] = {
            _mm512_loadu_si512((const __m512i*) (y[i].qs +   0)),
            _mm512_loadu_si512((const __m512i*) (y[i].qs +  64)),
            _mm512_loadu_si512((const __m512i*) (y[i].qs + 128)),
            _mm512_loadu_si512((const __m512i*) (y[i].qs + 192)),
        };

        for (int r = 0; r < 4; ++r) {
            const block_q6_K& bx = rows[r][i];
            const float d = yd * h2f(bx.d);
            const uint8_t* q4 = bx.ql;
            const uint8_t* qh = bx.qh;
            const __m128i scales = _mm_loadu_si128((const __m128i*) bx.scales);
            const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
            const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

            __m512i sumi0 = _mm512_setzero_si512();
            __m512i sumi1 = _mm512_setzero_si512();
            int is = 0;
            for (int j = 0; j < 2; ++j) {
                const __m512i q4bits12 = _mm512_loadu_si512((const __m512i*) (q4 + 64 * j));
                const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) (qh + 32 * j));
                const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
                const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m12), 2);
                const __m256i q4h_2 = _mm256_and_si256(q4bitsH, m48);
                const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, m192), 2);
                const __m512i q4_01 = _mm512_or_si512(_mm512_and_si512(q4bits12, m15),
                                                      _mm512_inserti64x4(_mm512_castsi256_si512(q4h_0), q4h_1, 1));
                const __m512i q4_23 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(q4bits12, 4), m15),
                                                      _mm512_inserti64x4(_mm512_castsi256_si512(q4h_2), q4h_3, 1));
                const __m512i p16_01 = _mm512_maddubs_epi16(q4_01, q8v[2 * j + 0]);
                const __m512i p16_23 = _mm512_maddubs_epi16(q4_23, q8v[2 * j + 1]);
                const __m256i sc_01 = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 0))),
                    _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1)), 1);
                const __m256i sc_23 = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 2))),
                    _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3)), 1);
                is += 4;
                sumi0 = _mm512_dpwssd_epi32(sumi0, _mm512_cvtepi8_epi16(sc_01), p16_01);
                sumi1 = _mm512_dpwssd_epi32(sumi1, _mm512_cvtepi8_epi16(sc_23), p16_23);
            }
            const __m512i sumi512 = _mm512_add_epi32(sumi0, sumi1);
            __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
            sumi = _mm256_sub_epi32(sumi, q8sclsub);
            acc[r] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = hsum_float_8(acc[r]);
}

void dot_iq4_xs_q8_K_x4(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
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
    const __m512i abs_values512 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kAbsIq4nl));
    const __m512i m4b = _mm512_set1_epi8(0x0F);
    const __m512i bit3 = _mm512_set1_epi8(0x08);
    const __m512i zero = _mm512_setzero_si512();

    __m256 accum[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int ibl = 0; ibl < nb; ++ibl) {
        if (ibl + 2 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][ibl + 2], _MM_HINT_T0);
        }
        const float yd = y[ibl].d;
        const __m512i q8v[4] = {
            _mm512_loadu_si512((const __m512i*) (y[ibl].qs +   0)),
            _mm512_loadu_si512((const __m512i*) (y[ibl].qs +  64)),
            _mm512_loadu_si512((const __m512i*) (y[ibl].qs + 128)),
            _mm512_loadu_si512((const __m512i*) (y[ibl].qs + 192)),
        };
        for (int r = 0; r < 4; ++r) {
            const block_iq4_xs& bx = rows[r][ibl];
            const uint8_t* qs = bx.qs;
            uint16_t sh = bx.scales_h;
            __m512i sumi0 = zero;
            __m512i sumi1 = zero;
            for (int ib = 0; ib < QK_K / 32; ib += 2) {
                const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*) qs);
                const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*) (qs + 16));
                qs += 32;
                const __m256i nib_1 = MM256_SET_M128I(_mm_srli_epi16(q4bits_1, 4), q4bits_1);
                const __m256i nib_2 = MM256_SET_M128I(_mm_srli_epi16(q4bits_2, 4), q4bits_2);
                const __m512i nib512 = _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(nib_1), nib_2, 1), m4b);
                const __m512i ax = _mm512_shuffle_epi8(abs_values512, nib512);
                const __mmask64 neg = _mm512_testn_epi8_mask(nib512, bit3);
                const __m512i q8b = q8v[ib / 2];
                const __m512i sy = _mm512_mask_sub_epi8(q8b, neg, zero, q8b);
                const __m512i p16 = _mm512_maddubs_epi16(ax, sy);
                const int16_t ls1 = (int16_t) (((bx.scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32);
                const int16_t ls2 = (int16_t) (((bx.scales_l[ib / 2] >>  4) | ((sh << 2) & 0x30)) - 32);
                sh >>= 4;
                const __m512i sc16 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(ls1)),
                                                        _mm256_set1_epi16(ls2), 1);
                if (ib & 2) sumi1 = _mm512_dpwssd_epi32(sumi1, p16, sc16);
                else        sumi0 = _mm512_dpwssd_epi32(sumi0, p16, sc16);
            }
            const __m512i sumi512 = _mm512_add_epi32(sumi0, sumi1);
            const __m256i sum12 = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
            accum[r] = _mm256_fmadd_ps(_mm256_set1_ps(h2f(bx.d) * yd), _mm256_cvtepi32_ps(sum12), accum[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = hsum_float_8(accum[r]);
}

void dot_iq3_xxs_q8_K_x4(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;
    const uint8_t* wb = (const uint8_t*) w_base;
    const block_iq3_xxs* rows[4] = {
        (const block_iq3_xxs*) (wb + 0 * row_bytes),
        (const block_iq3_xxs*) (wb + 1 * row_bytes),
        (const block_iq3_xxs*) (wb + 2 * row_bytes),
        (const block_iq3_xxs*) (wb + 3 * row_bytes),
    };
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const uint64_t* signs64 = (const uint64_t*) keven_signs_q2xs;

    __m256 accumf[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][i + 2], _MM_HINT_T0);
        }
        const float yd = y[i].d;
        const __m256i q8_h[8] = {
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
            const block_iq3_xxs& bx = rows[r][i];
            const uint8_t* q3 = bx.qs;
            const uint8_t* gas = bx.qs + QK_K / 4;
            uint32_t aux32[2];
            __m512i sumi0 = _mm512_setzero_si512();
            __m512i sumi1 = _mm512_setzero_si512();
            for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
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
                const __m256i s2_1 = _mm256_set_epi64x(
                    (long long) signs64[(aux32[0] >> 21) & 127], (long long) signs64[(aux32[0] >> 14) & 127],
                    (long long) signs64[(aux32[0] >>  7) & 127], (long long) signs64[(aux32[0] >>  0) & 127]);
                const __m256i s2_2 = _mm256_set_epi64x(
                    (long long) signs64[(aux32[1] >> 21) & 127], (long long) signs64[(aux32[1] >> 14) & 127],
                    (long long) signs64[(aux32[1] >>  7) & 127], (long long) signs64[(aux32[1] >>  0) & 127]);
                const __m512i q8s_512 = _mm512_inserti64x4(
                    _mm512_castsi256_si512(_mm256_sign_epi8(q8_h[ib32 + 0], s2_1)),
                    _mm256_sign_epi8(q8_h[ib32 + 1], s2_2), 1);

                const __m512i p16 = _mm512_maddubs_epi16(q2_512, q8s_512);
                const int16_t s1 = (int16_t) (2 * (aux32[0] >> 28) + 1);
                const int16_t s2 = (int16_t) (2 * (aux32[1] >> 28) + 1);
                const __m512i sc16 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(s1)),
                                                        _mm256_set1_epi16(s2), 1);
                if (ib32 & 2) sumi1 = _mm512_dpwssd_epi32(sumi1, p16, sc16);
                else          sumi0 = _mm512_dpwssd_epi32(sumi0, p16, sc16);
            }
            const __m512i sumi512 = _mm512_add_epi32(sumi0, sumi1);
            const __m256i sum12 = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
            accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(h2f(bx.d) * yd), _mm256_cvtepi32_ps(sum12), accumf[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = 0.25f * hsum_float_8(accumf[r]);
}

}  // namespace avx512

namespace {
#define ENTRY(T, ACT, NAME)     {avx512::dot_##NAME,       Type::T, Type::ACT, Arith::Avx2, Isa::Avx512, "avx512_" #NAME}
#define ENTRY_ZMM(T, ACT, NAME) {avx512::dot_##NAME##_zmm, Type::T, Type::ACT, Arith::Avx2, Isa::Avx512, "avx512_" #NAME}
#define ENTRY_X4(T, ACT, NAME)  {avx512::dot_##NAME##_x4,  Type::T, Type::ACT, Arith::Avx2, Isa::Avx512, "avx512_" #NAME "_x4"}
const DotKernel kTable[] = {
    ENTRY(Q4_0, Q8_0, q4_0_q8_0), ENTRY(Q4_1, Q8_1, q4_1_q8_1), ENTRY(Q5_0, Q8_0, q5_0_q8_0), ENTRY(Q5_1, Q8_1, q5_1_q8_1),
    ENTRY_ZMM(Q8_0, Q8_0, q8_0_q8_0), ENTRY(IQ4_NL, Q8_0, iq4_nl_q8_0), ENTRY(MXFP4, Q8_0, mxfp4_q8_0),
    ENTRY(NVFP4, Q8_0, nvfp4_q8_0),
    ENTRY_ZMM(Q4_K, Q8_K, q4_K_q8_K), ENTRY_ZMM(Q5_K, Q8_K, q5_K_q8_K), ENTRY_ZMM(Q6_K, Q8_K, q6_K_q8_K),
    ENTRY_ZMM(IQ4_XS, Q8_K, iq4_xs_q8_K), ENTRY_ZMM(IQ3_XXS, Q8_K, iq3_xxs_q8_K), ENTRY_ZMM(IQ3_S, Q8_K, iq3_s_q8_K),
};
const DotX4Kernel kX4Table[] = {
    ENTRY_X4(Q4_0, Q8_0, q4_0_q8_0),
    ENTRY_X4(Q8_0, Q8_0, q8_0_q8_0),
    ENTRY_X4(MXFP4, Q8_0, mxfp4_q8_0),
    ENTRY_X4(Q4_K, Q8_K, q4_K_q8_K),
    ENTRY_X4(Q5_K, Q8_K, q5_K_q8_K),
    ENTRY_X4(Q6_K, Q8_K, q6_K_q8_K),
    ENTRY_X4(IQ4_XS, Q8_K, iq4_xs_q8_K),
    ENTRY_X4(IQ3_XXS, Q8_K, iq3_xxs_q8_K),
};
#undef ENTRY_X4
#undef ENTRY_ZMM
#undef ENTRY
}  // namespace

DotTable avx512_dot_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }
DotX4Table avx512_dot_x4_table() noexcept { return {kX4Table, (int) (sizeof kX4Table / sizeof kX4Table[0])}; }

}  // namespace strata::quant
