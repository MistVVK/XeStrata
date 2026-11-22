// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// src/quant/x86/act_avx512.cpp - isolated AVX-512 activation quantizers for the Avx2 contract (Q8_0, Q8_1, Q8_K).
// Compiled strictly in this AVX-512 TU so act_avx2.cpp emits 0 EVEX/ZMM instructions (safe for Alder/Raptor Lake).
// Runtime-gated behind Isa::Avx512 (F/DQ/BW/VL + VNNI); bit-exact to act_avx2.cpp / quantize_q8_K_ref.
#include "tier_common.hpp"

#include "../registry.hpp"
#include "strata/quant/act.hpp"

#if defined(_MSC_VER) && !defined(__clang__) && defined(__AVX512VL__) && !defined(__AVX512VNNI__)
#define __AVX512VNNI__ 1
#endif

#if !defined(_MSC_VER) && (!defined(__AVX512F__) || !defined(__AVX512BW__) || !defined(__AVX512VL__))
#error "act_avx512.cpp needs AVX-512 F, BW, and VL"
#endif

namespace strata::quant {
namespace avx512 {

static inline int q8k_ctz32_avx512(unsigned int m) {
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward(&idx, m);
    return (int) idx;
#else
    return __builtin_ctz(m);
#endif
}

void quantize_q8_0_avx512(const float* STRATA_RESTRICT x, void* STRATA_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int64_t nb = k / QK8_0;
    block_q8_0* STRATA_RESTRICT y = (block_q8_0*) vy;
    const __m512 absm = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));

    for (int64_t i = 0; i < nb; ++i, x += 32) {
        if (i + 4 < nb) _mm_prefetch((const char*) (x + 128), _MM_HINT_T0);
        const __m512 v0 = _mm512_loadu_ps(x);
        const __m512 v1 = _mm512_loadu_ps(x + 16);

        const __m512 maxAbs = _mm512_max_ps(_mm512_and_ps(v0, absm), _mm512_and_ps(v1, absm));
        const __m256 m8 = _mm256_max_ps(_mm512_castps512_ps256(maxAbs), _mm512_extractf32x8_ps(maxAbs, 1));
        __m128 m4 = _mm_max_ps(_mm256_extractf128_ps(m8, 1), _mm256_castps256_ps128(m8));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float maxScalar = _mm_cvtss_f32(m4);

        const float d = maxScalar / 127.f;
        y[i].d = f2h(d);
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m512 mul = _mm512_set1_ps(id);

        const __m512 r0 = _mm512_roundscale_ps(_mm512_mul_ps(v0, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m512 r1 = _mm512_roundscale_ps(_mm512_mul_ps(v1, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m128i b0 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(r0));
        const __m128i b1 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(r1));
        _mm_storeu_si128((__m128i*) (y[i].qs + 0), b0);
        _mm_storeu_si128((__m128i*) (y[i].qs + 16), b1);
    }
}

void quantize_q8_1_avx512(const float* STRATA_RESTRICT x, void* STRATA_RESTRICT vy, int64_t k) {
    assert(QK8_1 == 32);
    assert(k % QK8_1 == 0);
    const int64_t nb = k / QK8_1;
    block_q8_1* STRATA_RESTRICT y = (block_q8_1*) vy;
    const __m512 absm = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));

    for (int64_t i = 0; i < nb; ++i, x += 32) {
        if (i + 4 < nb) _mm_prefetch((const char*) (x + 128), _MM_HINT_T0);
        const __m512 v0 = _mm512_loadu_ps(x);
        const __m512 v1 = _mm512_loadu_ps(x + 16);

        const __m512 maxAbs = _mm512_max_ps(_mm512_and_ps(v0, absm), _mm512_and_ps(v1, absm));
        const __m256 m8 = _mm256_max_ps(_mm512_castps512_ps256(maxAbs), _mm512_extractf32x8_ps(maxAbs, 1));
        __m128 m4 = _mm_max_ps(_mm256_extractf128_ps(m8, 1), _mm256_castps256_ps128(m8));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float maxScalar = _mm_cvtss_f32(m4);

        const float d = maxScalar / 127.f;
        y[i].d = f2h(d);
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m512 mul = _mm512_set1_ps(id);

        const __m512i i0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(v0, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        const __m512i i1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(v1, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        const int sumi = _mm512_reduce_add_epi32(_mm512_add_epi32(i0, i1));
        y[i].s = f2h(d * (float) sumi);
        _mm_storeu_si128((__m128i*) (y[i].qs + 0), _mm512_cvtsepi32_epi8(i0));
        _mm_storeu_si128((__m128i*) (y[i].qs + 16), _mm512_cvtsepi32_epi8(i1));
    }
}

void quantize_q8_K_avx512(const float* STRATA_RESTRICT x, void* STRATA_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q8_K* STRATA_RESTRICT y = (block_q8_K*) vy;
    const int64_t nb = k / QK_K;

    const __m512 absm = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const __m512 magic = _mm512_set1_ps(12582912.f);
    const __m512i mant = _mm512_set1_epi32(0x007fffff);
    const __m512i off = _mm512_set1_epi32(0x00400000);
    const __m512i c127 = _mm512_set1_epi32(127);
    const __m512i sign80 = _mm512_set1_epi8((char) 0x80);
    const __m512i zero = _mm512_setzero_si512();

    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        if (i + 1 < nb) {
            _mm_prefetch((const char*) (x + QK_K), _MM_HINT_T0);
            _mm_prefetch((const char*) (x + QK_K + 64), _MM_HINT_T0);
            _mm_prefetch((const char*) (x + QK_K + 128), _MM_HINT_T0);
            _mm_prefetch((const char*) (x + QK_K + 192), _MM_HINT_T0);
        }

        // 4x ZMM max(abs(x)): 4 iterations cover all 256 floats
        __m512 m0 = _mm512_setzero_ps();
        __m512 m1 = _mm512_setzero_ps();
        __m512 m2 = _mm512_setzero_ps();
        __m512 m3 = _mm512_setzero_ps();
        for (int j = 0; j < QK_K; j += 64) {
            m0 = _mm512_max_ps(m0, _mm512_and_ps(_mm512_loadu_ps(x + j +  0), absm));
            m1 = _mm512_max_ps(m1, _mm512_and_ps(_mm512_loadu_ps(x + j + 16), absm));
            m2 = _mm512_max_ps(m2, _mm512_and_ps(_mm512_loadu_ps(x + j + 32), absm));
            m3 = _mm512_max_ps(m3, _mm512_and_ps(_mm512_loadu_ps(x + j + 48), absm));
        }
        const __m512 m = _mm512_max_ps(_mm512_max_ps(m0, m1), _mm512_max_ps(m2, m3));
        const float amax = _mm512_reduce_max_ps(m);
        if (!amax) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }

        // Exact first-occurrence sign scan matching q8_K_ref (`if (ax > amax)`)
        float mx = 0.f;
        const __m512 va = _mm512_set1_ps(amax);
        for (int j = 0; j < QK_K; j += 32) {
            const unsigned int m0k = (unsigned int) _mm512_cmp_ps_mask(_mm512_and_ps(_mm512_loadu_ps(x + j +  0), absm), va, _CMP_EQ_OQ);
            const unsigned int m1k = (unsigned int) _mm512_cmp_ps_mask(_mm512_and_ps(_mm512_loadu_ps(x + j + 16), absm), va, _CMP_EQ_OQ);
            const unsigned int msk = m0k | (m1k << 16);
            if (msk) {
                mx = x[j + q8k_ctz32_avx512(msk)];
                break;
            }
        }

        const float iscale = -127.f / mx;
        const __m512 vis = _mm512_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 64) {
            __m128i b[4];
            for (int q = 0; q < 4; ++q) {
                __m512 p = _mm512_mul_ps(vis, _mm512_loadu_ps(x + j + 16 * q));
#if defined(__GNUC__)
                // Keep multiply rounded before add to match scalar nearest_int(iscale * x) bit-for-bit
                __asm__("" : "+v"(p));
#endif
                const __m512 t = _mm512_add_ps(p, magic);
                const __m512i vi = _mm512_min_epi32(_mm512_sub_epi32(_mm512_and_si512(_mm512_castps_si512(t), mant), off), c127);
                // VPMOVDB truncates each int32 to its low 8 bits in exact order (matches int8_t qs = MIN(127, v))
                b[q] = _mm512_cvtepi32_epi8(vi);
            }
            const __m256i p256_lo = _mm256_inserti128_si256(_mm256_castsi128_si256(b[0]), b[1], 1);
            const __m256i p256_hi = _mm256_inserti128_si256(_mm256_castsi128_si256(b[2]), b[3], 1);
            const __m512i p512 = _mm512_inserti64x4(_mm512_castsi256_si512(p256_lo), p256_hi, 1);
            _mm512_storeu_si512((__m512i*) (y[i].qs + j), p512);

            // Compute all 4 16-byte signed sub-block sums in one ZMM VPSADBW
            const __m512i sad = _mm512_sad_epu8(_mm512_xor_si512(p512, sign80), zero);
            const __m256i slo = _mm512_castsi512_si256(sad);
            const __m256i shi = _mm512_extracti64x4_epi64(sad, 1);
            const __m128i s0 = _mm256_castsi256_si128(slo);
            const __m128i s1 = _mm256_extracti128_si256(slo, 1);
            const __m128i s2 = _mm256_castsi256_si128(shi);
            const __m128i s3 = _mm256_extracti128_si256(shi, 1);
            y[i].bsums[j / 16 + 0] = (int16_t) (_mm_cvtsi128_si32(s0) + _mm_extract_epi32(s0, 2) - 2048);
            y[i].bsums[j / 16 + 1] = (int16_t) (_mm_cvtsi128_si32(s1) + _mm_extract_epi32(s1, 2) - 2048);
            y[i].bsums[j / 16 + 2] = (int16_t) (_mm_cvtsi128_si32(s2) + _mm_extract_epi32(s2, 2) - 2048);
            y[i].bsums[j / 16 + 3] = (int16_t) (_mm_cvtsi128_si32(s3) + _mm_extract_epi32(s3, 2) - 2048);
        }
        y[i].d = 1 / iscale;
    }
}

}  // namespace avx512

namespace {
const ActKernel kTable[] = {
    {avx512::quantize_q8_0_avx512, Type::Q8_0, Arith::Avx2, Isa::Avx512, "avx512_q8_0"},
    {avx512::quantize_q8_1_avx512, Type::Q8_1, Arith::Avx2, Isa::Avx512, "avx512_q8_1"},
    {avx512::quantize_q8_K_avx512, Type::Q8_K, Arith::Avx2, Isa::Avx512, "avx512_q8_K"},
};
}  // namespace

ActTable avx512_act_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }

}  // namespace strata::quant
