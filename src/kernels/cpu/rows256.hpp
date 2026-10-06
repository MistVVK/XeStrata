// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/rows256.hpp - the 256-bit multi-token rows whose integer dot is a plain sum of byte products: Q2_0
// (q2_avx2.cpp) and the IQ4_NL down rows (iq_avx2.cpp).  `Dot::dot(u, s)` is that sum per 32-bit lane: AVX2's
// madd(maddubs(u, s), 1), or AVX-VNNI's vpdpbusd in the files compiled with -mavxvnni (q2_avxvnni.cpp,
// iq_avxvnni.cpp).  maddubs does not saturate here (|w| <= 127 against int8 activations; Q2_0 codes 0..3), so both
// give the same integers and the rows are bitwise the same.  Included only by files compiled with AVX2.
#pragma once

#include "strata/kernels/cpu/expert.hpp"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::kernels::cpu::rows256 {

inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

inline float hsum8(__m256 v) {
    const __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 s = _mm_add_ps(h, _mm_movehl_ps(h, h));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}

// 16 bytes of 2-bit codes (value i in byte i/4, bits 2*(i%4)) -> 64 codes in value order, two 32-byte vectors
inline void unpack64(const uint8_t* codes, __m256i& lo, __m256i& hi) {
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes));
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);   // bytes 0..7
    const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);   // bytes 8..15
    lo = _mm256_set_m128i(_mm_unpackhi_epi16(a0, a1), _mm_unpacklo_epi16(a0, a1));  // values 0..31
    hi = _mm256_set_m128i(_mm_unpackhi_epi16(b0, b1), _mm_unpacklo_epi16(b0, b1));  // values 32..63
}

// Q2_0: codes 0..3 against the int8 activation per 32-value chunk, times the weight scale and the chunk scale, minus
// the weight scale times the chunk's `hx` (the -1 code offset).
template <class Dot, int NT>
inline void q2_row(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    __m256 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.f; }
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float d = h2f(blk);
        __m256i lo, hi;
        unpack64(blk + 2, lo, hi);
        const ptrdiff_t c = 2 * (ptrdiff_t) b;   // the block's two 32-value chunks
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->q + 32 * c;
            const __m256i s0 = Dot::dot(lo, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q)));
            const __m256i s1 = Dot::dot(hi, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32)));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[c]), _mm256_cvtepi32_ps(s0), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[c + 1]), _mm256_cvtepi32_ps(s1), acc[t]);
            corr[t] += d * (a[t]->hx[c] + a[t]->hx[c + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(acc[t]) - corr[t];
}

template <class Dot, int NT>
void q2_rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        q2_row<Dot, NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <class Dot>
void q2_rows_nt(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                int r0, int r1) {
    switch (nt) {
        case 1: q2_rows<Dot, 1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: q2_rows<Dot, 2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: q2_rows<Dot, 3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: q2_rows<Dot, 4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                q2_rows_nt<Dot>(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

}  // namespace strata::kernels::cpu::rows256
