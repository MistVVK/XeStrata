// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// q8k_avx2.inc - bit-exact AVX2 activation quantizer for Q8_K (zero EVEX/ZMM instructions).
// Matches quantize_q8_K_ref byte-for-byte on all inputs including exact ties and degenerate scales.

static inline int q8k_ctz32_avx2(unsigned int m) {
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward(&idx, m);
    return (int) idx;
#else
    return __builtin_ctz(m);
#endif
}

void quantize_q8_K_avx2(const float* STRATA_RESTRICT x, void* STRATA_RESTRICT vy, int64_t k) {
    block_q8_K* STRATA_RESTRICT y = (block_q8_K*) vy;
    const int64_t nb = k / QK_K;
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 magic = _mm256_set1_ps(12582912.f);
    const __m256i mant = _mm256_set1_epi32(0x007fffff);
    const __m256i off = _mm256_set1_epi32(0x00400000);
    const __m256i c127 = _mm256_set1_epi32(127);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i sign80 = _mm256_set1_epi8((char) 0x80);
    const __m256i zero = _mm256_setzero_si256();

    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        if (i + 1 < nb) {
            _mm_prefetch((const char*) (x + QK_K), _MM_HINT_T0);
            _mm_prefetch((const char*) (x + QK_K + 64), _MM_HINT_T0);
        }

        // 4-way unrolled max(abs(x)) to break the vmaxps dependency chain
        __m256 m0 = _mm256_setzero_ps();
        __m256 m1 = _mm256_setzero_ps();
        __m256 m2 = _mm256_setzero_ps();
        __m256 m3 = _mm256_setzero_ps();
        for (int j = 0; j < QK_K; j += 32) {
            m0 = _mm256_max_ps(m0, _mm256_and_ps(_mm256_loadu_ps(x + j +  0), absm));
            m1 = _mm256_max_ps(m1, _mm256_and_ps(_mm256_loadu_ps(x + j +  8), absm));
            m2 = _mm256_max_ps(m2, _mm256_and_ps(_mm256_loadu_ps(x + j + 16), absm));
            m3 = _mm256_max_ps(m3, _mm256_and_ps(_mm256_loadu_ps(x + j + 24), absm));
        }
        const __m256 m = _mm256_max_ps(_mm256_max_ps(m0, m1), _mm256_max_ps(m2, m3));
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_movehdup_ps(h));
        const float amax = _mm_cvtss_f32(h);
        if (!amax) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }

        // Exact first-occurrence sign scan matching q8_K_ref (`if (ax > amax)`)
        float mx = 0.f;
        const __m256 va = _mm256_set1_ps(amax);
        for (int j = 0; j < QK_K; j += 32) {
            const unsigned int msk0 = (unsigned int) _mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x + j +  0), absm), va, _CMP_EQ_OQ));
            const unsigned int msk1 = (unsigned int) _mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x + j +  8), absm), va, _CMP_EQ_OQ));
            const unsigned int msk2 = (unsigned int) _mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x + j + 16), absm), va, _CMP_EQ_OQ));
            const unsigned int msk3 = (unsigned int) _mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x + j + 24), absm), va, _CMP_EQ_OQ));
            const unsigned int msk = msk0 | (msk1 << 8) | (msk2 << 16) | (msk3 << 24);
            if (msk) {
                mx = x[j + q8k_ctz32_avx2(msk)];
                break;
            }
        }

        const float iscale = -127.f / mx;
        const __m256 vis = _mm256_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 32) {
            __m256i v[4];
            for (int q = 0; q < 4; ++q) {
                __m256 p = _mm256_mul_ps(vis, _mm256_loadu_ps(x + j + 8 * q));
#if defined(__GNUC__)
                // Keep multiply rounded before add to match scalar nearest_int(iscale * x) bit-for-bit
                __asm__("" : "+x"(p));
#endif
                const __m256 t = _mm256_add_ps(p, magic);
                v[q] = _mm256_min_epi32(_mm256_sub_epi32(_mm256_and_si256(_mm256_castps_si256(t), mant), off), c127);
                // Sign-extend low byte so even degenerate inf/NaN iscale matches int8_t truncation in q8_K_ref
                v[q] = _mm256_srai_epi32(_mm256_slli_epi32(v[q], 24), 24);
            }
            const __m256i p16a = _mm256_packs_epi32(v[0], v[1]);
            const __m256i p16b = _mm256_packs_epi32(v[2], v[3]);
            const __m256i p8 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p16a, p16b), perm);
            _mm256_storeu_si256((__m256i*) (y[i].qs + j), p8);

            // Fast exact 16-byte signed sums via psadbw on (byte ^ 0x80) = (byte + 128) in [0, 255]
            const __m256i sad = _mm256_sad_epu8(_mm256_xor_si256(p8, sign80), zero);
            const __m128i slo = _mm256_castsi256_si128(sad);
            const __m128i shi = _mm256_extracti128_si256(sad, 1);
            y[i].bsums[j / 16 + 0] = (int16_t) (_mm_cvtsi128_si32(slo) + _mm_extract_epi32(slo, 2) - 2048);
            y[i].bsums[j / 16 + 1] = (int16_t) (_mm_cvtsi128_si32(shi) + _mm_extract_epi32(shi, 2) - 2048);
        }
        y[i].d = 1 / iscale;
    }
}
