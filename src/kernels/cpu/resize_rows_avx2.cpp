// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/resize_rows_avx2.cpp - the vector copies of Pillow's rows' pass (include/strata/kernels/cpu/
// resize_rows.hpp).  Compiled with -mavx2 like i8_gemm_vnni.cpp; the AVX-VNNI copy's functions alone carry
// target("avxvnni") and the AVX-512 one's target("avx512f,avx512vl,avx512vnni"), and resize_rows.cpp reaches each only
// after its CPU probe.  This file includes no header with inline functions another file also compiles.
//
// FOUR ROWS AT A TIME: an output's weights are loaded once for four rows, whose four running totals are folded into
// one vector at the end (two hadd, an add of the halves), shifted, clipped and stored a byte to each row.
//
// THE AVX2 COPY widens eight bytes to int32 (vpmovzxbd) and multiplies them by eight weights (vpmulld): one step per
// eight taps, which is all a bicubic's output takes up to a shrink to a half.
//
// THE VNNI COPIES take sixteen taps a step in int16: the bytes widened to int16 (vpmovzxbw), each weight split into
// kh = k >> 15 and kl = k & 0x7fff, so both fit int16, and vpdpwssd summing the pairs of products into int32: the total
// is (sum kh x) 2^15 + sum kl x, which wraps in int32 to Pillow's exact total (its own fits int32).  Two vpdpwssd for
// sixteen taps where the AVX2 copy takes two vpmulld (four micro-ops on Intel's cores) for eight.  llvm-mca, cycles a
// step of four rows (2026-11-12): sixteen taps 5.0 by VNNI against 11.4 (Alder Lake) and 10.9 (Ice Lake) by vpmulld,
// eight taps 5.0 against 5.7 and 5.4; Zen 4's AVX-512 VNNI 6.0 against 6.3.  A zmm vpmulld copy was slower than
// vpmulld on eight taps (7.7 against 5.4 on Ice Lake): not taken.  The Ryzen 7 5700X (Zen 3: AVX2 alone) ran the
// AVX2 copy 2.1 to 2.5 times as fast as the scalar pass.

#include "strata/kernels/cpu/resize_rows.hpp"

#include <immintrin.h>

#if !defined(STRATA_AVXVNNI)
#define STRATA_AVXVNNI 0
#endif

namespace strata::kernels::cpu {

namespace {

// four rows' totals (one vector each) -> + half, >> bits, clipped to 0 .. 255, a byte to each row
inline void put4(__m256i a0, __m256i a1, __m256i a2, __m256i a3, int bits, uint8_t* d, int64_t stride) {
    const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
    s = _mm_srai_epi32(_mm_add_epi32(s, _mm_set1_epi32(1 << (bits - 1))), bits);
    s = _mm_min_epi32(_mm_max_epi32(s, _mm_setzero_si128()), _mm_set1_epi32(255));
    d[0] = (uint8_t) _mm_extract_epi32(s, 0);
    d[stride] = (uint8_t) _mm_extract_epi32(s, 1);
    d[2 * stride] = (uint8_t) _mm_extract_epi32(s, 2);
    d[3 * stride] = (uint8_t) _mm_extract_epi32(s, 3);
}

// one row's total -> its byte
inline uint8_t put1(__m256i a, int bits) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    const int32_t v = (_mm_cvtsi128_si32(s) + (1 << (bits - 1))) >> bits;
    return (uint8_t) (v < 0 ? 0 : v > 255 ? 255 : v);
}

inline __m256i load32(const void* p) { return _mm256_loadu_si256(static_cast<const __m256i*>(p)); }
inline __m256i bytes8(const uint8_t* p) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}
inline __m256i bytes16(const uint8_t* p) {
    return _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// the VNNI copies' loop, `DOT(acc, x, w)` their vpdpwssd
#define STRATA_RESIZE_ROWS_PAIRS(DOT)                                                                           \
    int64_t l = 0;                                                                                              \
    for (; l + 4 <= lines; l += 4) {                                                                            \
        const uint8_t* r = src + l * n_in;                                                                      \
        for (int64_t o = 0; o < n_out; ++o) {                                                                   \
            const uint8_t* p = r + t.lo[o];                                                                     \
            const int16_t* hh = t.kh + o * t.stride;                                                            \
            const int16_t* ll = t.kl + o * t.stride;                                                            \
            __m256i h0 = _mm256_setzero_si256(), h1 = h0, h2 = h0, h3 = h0, q0 = h0, q1 = h0, q2 = h0, q3 = h0; \
            for (int32_t j = 0; j < t.n[o]; j += 16) {                                                          \
                const __m256i wh = load32(hh + j);                                                              \
                const __m256i wl = load32(ll + j);                                                              \
                const __m256i x0 = bytes16(p + j), x1 = bytes16(p + n_in + j);                                  \
                const __m256i x2 = bytes16(p + 2 * n_in + j), x3 = bytes16(p + 3 * n_in + j);                   \
                h0 = DOT(h0, x0, wh);                                                                           \
                q0 = DOT(q0, x0, wl);                                                                           \
                h1 = DOT(h1, x1, wh);                                                                           \
                q1 = DOT(q1, x1, wl);                                                                           \
                h2 = DOT(h2, x2, wh);                                                                           \
                q2 = DOT(q2, x2, wl);                                                                           \
                h3 = DOT(h3, x3, wh);                                                                           \
                q3 = DOT(q3, x3, wl);                                                                           \
            }                                                                                                   \
            put4(_mm256_add_epi32(_mm256_slli_epi32(h0, 15), q0), _mm256_add_epi32(_mm256_slli_epi32(h1, 15), q1), \
                 _mm256_add_epi32(_mm256_slli_epi32(h2, 15), q2), _mm256_add_epi32(_mm256_slli_epi32(h3, 15), q3), \
                 bits, dst + l * n_out + o, n_out);                                                             \
        }                                                                                                       \
    }                                                                                                           \
    for (; l < lines; ++l)                                                                                      \
        for (int64_t o = 0; o < n_out; ++o) {                                                                   \
            const uint8_t* p = src + l * n_in + t.lo[o];                                                        \
            __m256i h = _mm256_setzero_si256(), q = h;                                                          \
            for (int32_t j = 0; j < t.n[o]; j += 16) {                                                          \
                const __m256i x = bytes16(p + j);                                                               \
                h = DOT(h, x, load32(t.kh + o * t.stride + j));                                                 \
                q = DOT(q, x, load32(t.kl + o * t.stride + j));                                                 \
            }                                                                                                   \
            dst[l * n_out + o] = put1(_mm256_add_epi32(_mm256_slli_epi32(h, 15), q), bits);                    \
        }

#if STRATA_AVXVNNI
__attribute__((target("avxvnni"))) inline __m256i dot_avxvnni(__m256i a, __m256i x, __m256i w) {
    return _mm256_dpwssd_avx_epi32(a, x, w);
}
#endif
__attribute__((target("avx512f,avx512vl,avx512vnni"))) inline __m256i dot_avx512(__m256i a, __m256i x, __m256i w) {
    return _mm256_dpwssd_epi32(a, x, w);
}

}  // namespace

void resize_rows_avx2(const uint8_t* src, int64_t lines, int64_t n_in, int64_t n_out, const ResizeRowsTaps& t,
                      int bits, uint8_t* dst) {
    int64_t l = 0;
    for (; l + 4 <= lines; l += 4) {
        const uint8_t* r = src + l * n_in;
        for (int64_t o = 0; o < n_out; ++o) {
            const uint8_t* p = r + t.lo[o];
            const int32_t* k = t.k + o * t.stride;
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
            for (int32_t j = 0; j < t.n[o]; j += 8) {
                const __m256i w = load32(k + j);
                a0 = _mm256_add_epi32(a0, _mm256_mullo_epi32(w, bytes8(p + j)));
                a1 = _mm256_add_epi32(a1, _mm256_mullo_epi32(w, bytes8(p + n_in + j)));
                a2 = _mm256_add_epi32(a2, _mm256_mullo_epi32(w, bytes8(p + 2 * n_in + j)));
                a3 = _mm256_add_epi32(a3, _mm256_mullo_epi32(w, bytes8(p + 3 * n_in + j)));
            }
            put4(a0, a1, a2, a3, bits, dst + l * n_out + o, n_out);
        }
    }
    for (; l < lines; ++l)
        for (int64_t o = 0; o < n_out; ++o) {
            const uint8_t* p = src + l * n_in + t.lo[o];
            const int32_t* k = t.k + o * t.stride;
            __m256i a = _mm256_setzero_si256();
            for (int32_t j = 0; j < t.n[o]; j += 8)
                a = _mm256_add_epi32(a, _mm256_mullo_epi32(load32(k + j), bytes8(p + j)));
            dst[l * n_out + o] = put1(a, bits);
        }
}

#if STRATA_AVXVNNI
__attribute__((target("avxvnni"))) void resize_rows_avxvnni(const uint8_t* src, int64_t lines, int64_t n_in,
                                                            int64_t n_out, const ResizeRowsTaps& t, int bits,
                                                            uint8_t* dst) {
    STRATA_RESIZE_ROWS_PAIRS(dot_avxvnni)
}
#endif

__attribute__((target("avx512f,avx512vl,avx512vnni"))) void resize_rows_avx512(const uint8_t* src, int64_t lines,
                                                                               int64_t n_in, int64_t n_out,
                                                                               const ResizeRowsTaps& t, int bits,
                                                                               uint8_t* dst) {
    STRATA_RESIZE_ROWS_PAIRS(dot_avx512)
}

}  // namespace strata::kernels::cpu
