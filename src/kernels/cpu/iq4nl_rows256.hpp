// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/iq4nl_rows256.hpp - the IQ4_NL (type 20) down rows against Q8_0 activations, several tokens at once,
// for iq_avx2.cpp (AVX2) and iq_avxvnni.cpp (AVX-VNNI); `Dot` as in rows256.hpp.
//
// ggml-cpu's own AVX-2 dot (ggml_vec_dot_iq4_nl_q8_0) is single-token: every token redoes the nibble ->
// kvalues_iq4nl pshufb decode and the |w| half of the signed x signed product.  Here both are computed once per
// block; every token then costs a sign, the integer dot and an fmadd.  The arithmetic is ggml's - only the order of
// the float additions differs.
//
// ggml's code is MIT: Copyright (c) 2023-2026 The ggml authors (third_party/main/ggml/LICENSE).
#pragma once

#include "rows256.hpp"

// ggml's block layouts and the IQ4_NL codebook (its tables are static: a copy in each including file)
#if !defined(GGML_COMMON_DECL_CPP)
#define GGML_COMMON_DECL_CPP
#endif
#if !defined(GGML_COMMON_IMPL_CPP)
#define GGML_COMMON_IMPL_CPP
#endif
#include "ggml-common.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu::rows256 {

// hadd twice: the AVX2 IQ kernels' horizontal sum (iq_avx2.cpp), kept so the rows stay bitwise the same
inline float hsum8_hadd(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

template <class Dot, int NT>
void iq4nl_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out, int r0,
                int r1) {
    const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const int nb = n / QK4_NL;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
            const __m128i lo = _mm_and_si128(bits, m4b);                      // values 0..15
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);    // values 16..31
            const __m256i q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)),
                                                       _mm_shuffle_epi8(values, hi), 1);
            const __m256i aq = _mm256_sign_epi8(q4, q4);   // |w|: the unsigned operand of the dot
            const float dx = h2f(blk);
            for (int t = 0; t < NT; ++t) {
                const block_q8_0& b = y[t][ib];
                const __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.qs));
                const __m256i p = Dot::dot(aq, _mm256_sign_epi8(q8, q4));
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * h2f((const uint8_t*) &b.d)), _mm256_cvtepi32_ps(p),
                                          accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8_hadd(accf[t]);
    }
}

template <class Dot>
void iq4nl_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                     int r0, int r1) {
    const block_q8_0* y[8];
    for (int t = 0; t < nt && t < 8; ++t) y[t] = (const block_q8_0*) hq[t];
    switch (nt) {
        case 1: iq4nl_rows<Dot, 1>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: iq4nl_rows<Dot, 2>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: iq4nl_rows<Dot, 3>(w, row_bytes, n, y, out, r0, r1); break;
        case 4: iq4nl_rows<Dot, 4>(w, row_bytes, n, y, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            iq4nl_down_rows<Dot>(w, row_bytes, n, hq + t0, k, out + t0, r0, r1);
        }
    }
}

}  // namespace strata::kernels::cpu::rows256
