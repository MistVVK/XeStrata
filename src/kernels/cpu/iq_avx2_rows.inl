// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/cpu/iq_avx2_rows.inl - the row kernels of iq_avx2.cpp.  Not a header: iq_avx2.cpp includes it twice,
//
//   STRATA_ROWS_VNNI 0   the AVX2 forms (namespace plain);
//   STRATA_ROWS_VNNI 1   the same loops with AVX-VNNI (namespace vnni, built with STRATA_AVXVNNI): vpdpwssd for each
//                        madd + add, vpdpbusd for maddubs + madd(ones).  Integer sums, so the same bits.
//
// Every function here is STRATA_ROWS_FN: nothing in the plain copy, GCC/Clang's target("avxvnni") in the VNNI copy, so
// the AVX-VNNI intrinsics compile in these functions only and the plain copy stays AVX2 (-mavxvnni on the whole file
// would let the compiler use the extension anywhere).  MSVC takes the intrinsics under /arch:AVX2.  The VNNI copy is
// called only where cpu_avxvnni_ok().

// acc + madd(p16, sc).  vpdpwssd is the madd and the add in one uop; both wrap the same way.
STRATA_ROWS_FN inline __m256i madd_add(__m256i acc, __m256i p16, __m256i sc) {
#if STRATA_ROWS_VNNI
    return _mm256_dpwssd_avx_epi32(acc, p16, sc);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p16, sc));
#endif
}

// Four u8 x s8 products per int32 lane.  vpdpbusd does not saturate where maddubs would, but maddubs cannot here: the
// u8 side is an IQ4_NL |w| <= 127, so a pair is at most 2 * 127 * 128.
STRATA_ROWS_FN inline __m256i dot4u(__m256i u8, __m256i s8) {
#if STRATA_ROWS_VNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u8, s8);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u8, s8), _mm256_set1_epi16(1));
#endif
}

template <int TY, int NT> STRATA_ROWS_FN
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    const int pf = prefetch_distance();
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt32<TY>::bytes;
        rows_ahead(blk, pf);
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m256i g, sgn, sc;
                Fmt32<TY>::decode(blk, j, half, g, sgn, sc);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(g, ys), sc);
                }
            }
        }
        const float dx = h2f(u16(blk)) * Fmt32<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

// The (block k0, block k0 + 1) sums of tokens t and t + 1 (8 int32 lanes each: e, o, e2, o2) in one go:
// [token t's even, its odd, token t + 1's even, its odd] (exact).
STRATA_ROWS_FN inline __m128 pair_sums2(__m256i e, __m256i o, __m256i e2, __m256i o2) {
    const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(e, o), _mm256_hadd_epi32(e2, o2));
    return _mm_cvtepi32_ps(_mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)));
}

// Token u's (0 or 1) row of a token pair's sums [t's even blocks, t's odd blocks, u's even, u's odd]: even + odd.
STRATA_ROWS_FN inline float canon_pair_sum(__m128 a, int u) {
    const __m128 v = u ? _mm_movehl_ps(a, a) : a;
    return _mm_cvtss_f32(v) + _mm_cvtss_f32(_mm_shuffle_ps(v, v, 1));
}

// One token's: [its even block's sum, its odd block's, the same again] (exact).
STRATA_ROWS_FN inline __m128 pair_sums1(__m256i e, __m256i o) {
    const __m256i h = _mm256_hadd_epi32(e, o);
    const __m128i f = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
    return _mm_cvtepi32_ps(_mm_hadd_epi32(f, f));
}

// The exact sum of 8 int32 lanes.
STRATA_ROWS_FN inline int32_t hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// ---- IQ2_XS (17) with ggml's vectorized sign decode (arch/x86/quants.c, ggml_vec_dot_iq2_xs_q8_K):
// the 7-bit sign indices never touch ksigns_iq2xs - the 8th sign bit is reconstructed by parity (two
// shifts, a xor and ggml's bit_helper pshufb) and the per-half sign vectors come out of shuffles of one
// 32-byte load.  The 8 scale bytes become all 16 half-scales with ggml's unpack trick.  The grid lookups
// stay scalar loads into set_epi64x (ggml does the same).  Every token still only pays a load, a sign,
// a maddubs, a madd and an add per half, into alternating accumulators.
// C: the canonical rows (row_dot_canon below) against xq / xd instead of y: a block's integer sum, and two blocks'
// (d x the scale x 0.125) x sum into the even and the odd lane with one fma; with fx (exact.hpp) instead each block's
// integer sum times its integer scale (canon_math.hpp's fixed-point sums, row_dot_canon_fx's) added in int64.
template <bool C, int NT> STRATA_ROWS_FN
inline void row_dot_iq2xs_impl(const uint8_t* row, int nblocks, const block_q8_K* const* y, const int8_t* const* xq,
                               const float* const* xd, float* res, bool fx = false) {
    static const uint8_t bit_sel[32] = {
        1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80, 1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80,
        1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80, 1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80 };
    static const char shuf_even[32] = {   // sign bits of u16 0,1,2,3 (even bytes of the sign register)
        0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
        4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6 };
    static const char shuf_odd[32] = {    // sign bits of u16 4,5,6,7
        8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10,
        12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14 };
    static const uint8_t bit_help[32] = {  // 0x80 when the 4-bit index has odd popcount (parity bit 7)
        0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80,
        (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00,
        0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80,
        (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00 };
    static const uint8_t k_sc_shuffle[128] = {  // half h -> scale bytes 2h (lanes 0-7), 2h+1 (lanes 8-15)
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15 };
    const __m256i bsel = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bit_sel));
    const __m256i shev = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(shuf_even));
    const __m256i shod = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(shuf_odd));
    const __m256i bhelp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bit_help));
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf), m1 = _mm_set1_epi8(1);
    const __m256i one8 = _mm256_set1_epi8(1);
    const int pf = prefetch_distance();

    __m256 accf[NT];
    constexpr int NP = (NT + 1) / 2;
    __m128 accc[NP];   // C: token pairs, lanes 0, 1 the first token's even and odd blocks, 2, 3 the second's
    alignas(32) int32_t even[NT][8];   // C: the even block's sums, in memory until the odd one's (one token: even1)
    __m256i even1 = _mm256_setzero_si256();
    for (int p = 0; p < NP; ++p) accc[p] = _mm_setzero_ps();
    int64_t fxs[NT];   // C with fx: the fixed-point sums, and each token's row top
    int top[NT];
    for (int t = 0; t < NT; ++t) {
        accf[t] = _mm256_setzero_ps();
        fxs[t] = 0;
        top[t] = canon::kFxNone;
    }
    if (C && fx) {
        int topw = canon::kFxNone;
        for (int i = 0; i < nblocks; ++i) topw = std::max(topw, canon::fx_exp(h2f(u16(row + (size_t) i * 74))));
        topw = canon::fx_row_top(canon::fx_exp(h2f(u16(row))), topw);
        for (int t = 0; t < NT; ++t) {
            for (int i = 0; i < nblocks; ++i) top[t] = std::max(top[t], canon::fx_exp(xd[t][i]));
            top[t] = canon::fx_row_top(canon::fx_exp(xd[t][0]), top[t]) + topw;
        }
    }
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 74;
        rows_ahead(blk, pf);
        // the 8 scale bytes -> 16 half-scales of 2*s+1, interleaved [a0, b0, a1, b1, ...] (ggml's unpack)
        __m128i st = _mm_set1_epi64x((long long) u64(blk + 66));
        st = _mm_unpacklo_epi8(_mm_and_si128(st, m4), _mm_and_si128(_mm_srli_epi16(st, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(st, 1), m1);
        __m256i acc[NT][2];
        for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = _mm256_setzero_si256();
        for (int jj = 0; jj < 2; ++jj) {   // 128 values (16 u16) per step
            const __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 2 + 32 * (ptrdiff_t) jj));
            const __m256i p7 = _mm256_srli_epi16(q2, 9);              // 7 sign bits per u16, even bytes
            const __m256i p3 = _mm256_srli_epi16(q2, 13);             // sign index bits 4-6
            const __m256i fsb = _mm256_or_si256(p7,
                _mm256_shuffle_epi8(bhelp, _mm256_xor_si256(p7, p3)));  // + the parity sign bit 7
            const __m256i s01 = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(fsb)); // u16 0-7
            const __m256i s23 = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(fsb, 1)); // u16 8-15
            alignas(32) uint16_t gi[16];
            _mm256_store_si256(reinterpret_cast<__m256i*>(gi), _mm256_and_si256(q2, m511));
            for (int h = 0; h < 4; ++h) {   // the four 32-value halves of this group
                const int H = 4 * jj + h;   // global half index -> scales byte pair
                const __m256i g = _mm256_set_epi64x((long long) iq2xs_grid[gi[4 * h + 3]],
                                                   (long long) iq2xs_grid[gi[4 * h + 2]],
                                                   (long long) iq2xs_grid[gi[4 * h + 1]],
                                                   (long long) iq2xs_grid[gi[4 * (size_t) h]]);
                const __m256i sb = _mm256_shuffle_epi8(h < 2 ? s01 : s23, (h & 1) ? shod : shev);
                const __m256i sgn = _mm256_or_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(sb, bsel), bsel), one8);
                const __m256i sc = _mm256_cvtepi8_epi16(
                    _mm_shuffle_epi8(scales, _mm_loadu_si128(reinterpret_cast<const __m128i*>(k_sc_shuffle) + H)));
                const int off = 128 * jj + 32 * h;
                for (int t = 0; t < NT; ++t) {
                    const int8_t* yq = C ? xq[t] + 256 * (ptrdiff_t) i + off : y[t][i].qs + off;
                    const __m256i yv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yq));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acc[t][h & 1] = madd_add(acc[t][h & 1], _mm256_maddubs_epi16(g, ys), sc);
                }
            }
        }
        if constexpr (C) {
            if (fx) {
                const canon::FxPart w = canon::fx_part(h2f(u16(blk)));
                for (int t = 0; t < NT; ++t)
                    fxs[t] += (int64_t) canon::fx_scale(canon::fx_part(xd[t][i]), w, top[t]) *
                              (int64_t) hsum_i32(_mm256_add_epi32(acc[t][0], acc[t][1]));
            } else {
                if ((i & 1) == 0) {
                    if constexpr (NT == 1) even1 = _mm256_add_epi32(acc[0][0], acc[0][1]);
                    else
                        for (int t = 0; t < NT; ++t)
                            _mm256_store_si256(reinterpret_cast<__m256i*>(even[t]), _mm256_add_epi32(acc[t][0], acc[t][1]));
                } else if constexpr (NT == 1) {
                    const __m128 cw = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, 0, 0, (short) u16(blk), (short) u16(blk - 74)));
                    const __m128 T = pair_sums1(even1, _mm256_add_epi32(acc[0][0], acc[0][1]));
                    const __m128 dx = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(xd[0] + i - 1)));
                    accc[0] = _mm_fmadd_ps(_mm_mul_ps(_mm_mul_ps(cw, dx), _mm_set1_ps(0.125f)), T, accc[0]);
                } else {
                    const short h0 = (short) u16(blk - 74), h1 = (short) u16(blk);
                    const __m128 cw = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, h1, h0, h1, h0));
                    __m256i odd[NT];
                    for (int t = 0; t < NT; ++t) odd[t] = _mm256_add_epi32(acc[t][0], acc[t][1]);
                    for (int p = 0; p < NP; ++p) {
                        const int t = 2 * p, u = t + 1 < NT ? t + 1 : t;
                        const __m128 T = pair_sums2(_mm256_load_si256(reinterpret_cast<const __m256i*>(even[t])), odd[t],
                                                    _mm256_load_si256(reinterpret_cast<const __m256i*>(even[u])), odd[u]);
                        const __m128 dx = _mm_castpd_ps(
                            _mm_unpacklo_pd(_mm_load_sd(reinterpret_cast<const double*>(xd[t] + i - 1)),
                                            _mm_load_sd(reinterpret_cast<const double*>(xd[u] + i - 1))));
                        accc[p] = _mm_fmadd_ps(_mm_mul_ps(_mm_mul_ps(cw, dx), _mm_set1_ps(0.125f)), T, accc[p]);
                    }
                }
            }
        } else {
            const float dx = h2f(u16(blk)) * 0.125f;
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                    _mm256_cvtepi32_ps(_mm256_add_epi32(acc[t][0], acc[t][1])), accf[t]);
        }
    }
    if constexpr (C) {
        if (fx)
            for (int t = 0; t < NT; ++t) res[t] = canon::fx_value(fxs[t], top[t], -3);
        else
            for (int p = 0; p < NP; ++p) {
                res[2 * (size_t) p] = canon_pair_sum(accc[p], 0);
                if (2 * p + 1 < NT) res[2 * p + 1] = canon_pair_sum(accc[p], 1);
            }
    } else {
        for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
    }
}

template <int NT> STRATA_ROWS_FN
inline void row_dot_iq2xs(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    row_dot_iq2xs_impl<false, NT>(row, nblocks, y, nullptr, nullptr, res);
}

// ---- Q4_K (12) and Q5_K (13): d, dmin, 12 bytes of 6-bit scales and mins, [Q5_K: qh[32]], qs[128].  ggml's
// ggml_vec_dot_q4_K_q8_K / _q5_K_q8_K: x = d * sc * q - dmin * m, so a block is d * y.d * sum(sc * q.y) minus
// dmin * y.d * sum(m * (the activation's 32-value sums, two bsums each)).  The nibbles (and Q5_K's fifth bits) are
// unpacked once per block for all tokens; a token pays a load, a maddubs and a madd per 32 values, and one madd of
// its bsums against the mins per block.
template <bool Q5, int NT> STRATA_ROWS_FN
inline void row_dot_q45k(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    constexpr int bytes = Q5 ? 176 : 144, qs_off = Q5 ? 48 : 16;
    constexpr uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const int pf = prefetch_distance();
    __m256 accf[NT], accm[NT];
    for (int t = 0; t < NT; ++t) accf[t] = accm[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * bytes;
        rows_ahead(blk, pf);
        // ggml's unpacking of the 12 scale bytes: u[0..1] the eight scales, u[2..3] the eight mins
        uint32_t u[4];
        std::memcpy(u, blk + 4, 12);
        u[3] = ((u[2] >> 4) & kmask2) | (((u[1] >> 6) & kmask3) << 4);
        const uint32_t um = u[1] & kmask1;
        u[1] = (u[2] & kmask2) | (((u[0] >> 6) & kmask3) << 4);
        u[2] = um;
        u[0] &= kmask1;
        uint8_t sc[8];
        std::memcpy(sc, u, 8);
        const __m128i m8 = _mm_set_epi32(0, 0, (int) u[3], (int) u[2]);
        const __m256i mins = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(m8, m8));   // m0 m0 m1 m1 ...: one per bsum
        __m256i hb = _mm256_setzero_si256();
        if constexpr (Q5) hb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 16));
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + qs_off + 32 * (ptrdiff_t) j));
            for (int half = 0; half < 2; ++half) {
                const int H = 2 * j + half;   // values 32H..32H+31: the low nibbles, then the high ones
                __m256i g = half ? _mm256_and_si256(_mm256_srli_epi16(q, 4), m4) : _mm256_and_si256(q, m4);
                if constexpr (Q5) {   // bit H of each qh byte is the fifth bit of that value
                    const __m256i hm = _mm256_set1_epi8((char) (1 << H));
                    g = _mm256_or_si256(g, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(hb, hm), hm),
                                                            _mm256_set1_epi8(0x10)));
                }
                const __m256i s = sc32(sc[H]);
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[t][i].qs + 32 * (ptrdiff_t) H));
                    acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(g, yv), s);
                }
            }
        }
        const float d = h2f(u16(blk)), dmin = h2f(u16(blk + 2));
        for (int t = 0; t < NT; ++t) {
            const __m256i bs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[t][i].bsums));
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
            accm[t] = _mm256_fmadd_ps(_mm256_set1_ps(dmin * y[t][i].d), _mm256_cvtepi32_ps(_mm256_madd_epi16(bs, mins)),
                                      accm[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]) - hsum8(accm[t]);
}

// ---- Q6_K (14): ql[128], qh[64], 16 signed scales (one per 16 values), d.  ggml's ggml_vec_dot_q6_K_q8_K:
// x = d * sc * (q - 32) with q in 0..63, so a block is d * y.d * (sum(sc * q.y) - 32 * sum(sc * bsums)).  The
// 6-bit values are assembled once per block for all tokens; the offset is one madd of the token's bsums.
template <int NT> STRATA_ROWS_FN
inline void row_dot_q6k(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    const __m256i m4 = _mm256_set1_epi8(0x0f), m3 = _mm256_set1_epi8(0x03);
    const __m256i mc = _mm256_set1_epi8(0x0c), m30 = _mm256_set1_epi8(0x30);
    const int pf = prefetch_distance();
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 210;
        rows_ahead(blk, pf);
        const int8_t* scs = reinterpret_cast<const int8_t*>(blk + 192);
        const __m256i sc_all = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(scs)));
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int j2 = 0; j2 < 2; ++j2) {   // two 128-value chunks: 64 ql bytes and 32 qh bytes each
            const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 64 * (ptrdiff_t) j2));
            const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 64 * (ptrdiff_t) j2 + 32));
            const __m256i hh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 128 + 32 * (ptrdiff_t) j2));
            __m256i g[4];   // values 128*j2 + 32k .. +31: qh bits 2k, 2k+1 above the nibbles
            g[0] = _mm256_or_si256(_mm256_and_si256(l0, m4), _mm256_slli_epi16(_mm256_and_si256(hh, m3), 4));
            g[1] = _mm256_or_si256(_mm256_and_si256(l1, m4), _mm256_slli_epi16(_mm256_and_si256(hh, mc), 2));
            g[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4), _mm256_and_si256(hh, m30));
            g[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4),
                                   _mm256_and_si256(_mm256_srli_epi16(hh, 2), m30));
            for (int k = 0; k < 4; ++k) {
                const int s = 8 * j2 + 2 * k;   // the scales of the two 16-value halves
                const __m256i sc = sc16(scs[s], scs[s + 1]);
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(y[t][i].qs + 128 * (ptrdiff_t) j2 + 32 * (ptrdiff_t) k));
                    acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(g[k], yv), sc);
                }
            }
        }
        const float d = h2f(u16(blk + 208));
        for (int t = 0; t < NT; ++t) {
            const __m256i bs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[t][i].bsums));
            const __m256i off = _mm256_slli_epi32(_mm256_madd_epi16(sc_all, bs), 5);   // 32 * sum(sc * bsums)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * y[t][i].d), _mm256_cvtepi32_ps(_mm256_sub_epi32(acci[t], off)),
                                      accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

// ---- Q2_K (10): scales[16] (a 4-bit scale and a 4-bit min a 16 values), qs[64], d, dmin.  ggml's
// ggml_vec_dot_q2_K_q8_K: x = d * sc * q - dmin * m with q in 0..3, so a block is d * y.d * sum(sc * q.y) minus
// dmin * y.d * sum(m * bsums) (a bsum a 16 values, as the scales).  Value 128k + 32j + l is bits 2j of qs[32k + l].
template <int NT> STRATA_ROWS_FN
inline void row_dot_q2k(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    const __m256i m3 = _mm256_set1_epi8(0x03);
    const int pf = prefetch_distance();
    __m256 accf[NT], accm[NT];
    for (int t = 0; t < NT; ++t) accf[t] = accm[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 84;
        rows_ahead(blk, pf);
        const __m128i sc8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk));
        const __m256i mins = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sc8, 4), _mm_set1_epi8(0x0f)));
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int k = 0; k < 2; ++k) {
            const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 16 + 32 * (ptrdiff_t) k));
            for (int j = 0; j < 4; ++j) {
                const __m256i g = _mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m3);
                const int s = 8 * k + 2 * j;   // the scales of the two 16-value halves
                const __m256i sc = sc16(blk[s] & 0xf, blk[s + 1] & 0xf);
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(y[t][i].qs + 128 * (ptrdiff_t) k + 32 * (ptrdiff_t) j));
                    acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(g, yv), sc);
                }
            }
        }
        const float d = h2f(u16(blk + 80)), dmin = h2f(u16(blk + 82));
        for (int t = 0; t < NT; ++t) {
            const __m256i bs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[t][i].bsums));
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
            accm[t] = _mm256_fmadd_ps(_mm256_set1_ps(dmin * y[t][i].d), _mm256_cvtepi32_ps(_mm256_madd_epi16(bs, mins)),
                                      accm[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]) - hsum8(accm[t]);
}

// ---- Q3_K (11): hmask[32], qs[64], 12 bytes of 6-bit scales (less 32), d.  ggml's ggml_vec_dot_q3_K_q8_K:
// x = d * sc * (q - 4 * (1 - h)) with q in 0..3 and h the hmask bit, so with q' = q + 4h in 0..7 a block is
// d * y.d * (sum(sc * q'.y) - 4 * sum(sc * bsums)).  Value 128k + 32j + l: bits 2j of qs[32k + l], bit 4k + j of
// hmask[l].
template <int NT> STRATA_ROWS_FN
inline void row_dot_q3k(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    constexpr uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const __m256i m3 = _mm256_set1_epi8(0x03), four = _mm256_set1_epi8(0x04);
    const int pf = prefetch_distance();
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 110;
        rows_ahead(blk, pf);
        // ggml's unpacking of the 12 scale bytes into 16 6-bit scales
        uint32_t a[4];
        std::memcpy(a, blk + 96, 12);
        const uint32_t tmp = a[2];
        a[2] = ((a[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        a[3] = ((a[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        a[0] = (a[0] & kmask2) | ((tmp & kmask1) << 4);
        a[1] = (a[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int8_t scs[16];
        std::memcpy(scs, a, 16);
        for (int s = 0; s < 16; ++s) scs[s] = (int8_t) (scs[s] - 32);
        const __m256i sc_all = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(scs)));
        const __m256i hm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk));
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int k = 0; k < 2; ++k) {
            const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 32 + 32 * (ptrdiff_t) k));
            for (int j = 0; j < 4; ++j) {
                const __m256i bit = _mm256_set1_epi8((char) (1 << (4 * k + j)));
                const __m256i h = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(hm, bit), bit), four);
                const __m256i g = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m3), h);
                const int s = 8 * k + 2 * j;
                const __m256i sc = sc16(scs[s], scs[s + 1]);
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(y[t][i].qs + 128 * (ptrdiff_t) k + 32 * (ptrdiff_t) j));
                    acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(g, yv), sc);
                }
            }
        }
        const float d = h2f(u16(blk + 108));
        for (int t = 0; t < NT; ++t) {
            const __m256i bs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[t][i].bsums));
            const __m256i off = _mm256_slli_epi32(_mm256_madd_epi16(sc_all, bs), 2);   // 4 * sum(sc * bsums)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * y[t][i].d), _mm256_cvtepi32_ps(_mm256_sub_epi32(acci[t], off)),
                                      accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

template <int TY, int NT> STRATA_ROWS_FN
inline void row_dot_any(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    if constexpr (TY == 17)      row_dot_iq2xs<NT>(row, nblocks, y, res);
    else if constexpr (TY == 12) row_dot_q45k<false, NT>(row, nblocks, y, res);
    else if constexpr (TY == 13) row_dot_q45k<true, NT>(row, nblocks, y, res);
    else if constexpr (TY == 14) row_dot_q6k<NT>(row, nblocks, y, res);
    else if constexpr (TY == 10) row_dot_q2k<NT>(row, nblocks, y, res);
    else if constexpr (TY == 11) row_dot_q3k<NT>(row, nblocks, y, res);
    else                         row_dot<TY, NT>(row, nblocks, y, res);
}

template <int TY, int NT> STRATA_ROWS_FN
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    const float lim = expert_swiglu_limit();
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot_any<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = swiglu_host(g[t], u[t], lim);
    }
}

template <int TY, int NT> STRATA_ROWS_FN
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY> STRATA_ROWS_FN
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

// ---- canonical rows (canon_expert.cpp, iq_kernels.cpp's seg_dot_canon): the activation's int8 values with one scale
// for each 256 (xd, a float each), a block's eight 32-value units' (2 ls + 1)-weighted integer sums added first, then
// (d x the activation's scale) x K x that sum with one fma into the even or the odd blocks' sum, two blocks a step;
// the two sums added at the end.  The same bits as the GPU's row.
template <int TY, int NT> STRATA_ROWS_FN
inline void row_dot_canon_gen(const uint8_t* row, int nblocks, const int8_t* const* xq, const float* const* xd,
                              float* res) {
    constexpr int NP = (NT + 1) / 2;   // token pairs: lanes 0, 1 the first token's even and odd sums, 2, 3 the second's
    const int pf = prefetch_distance();
    __m128 acc[NP];
    for (int p = 0; p < NP; ++p) acc[p] = _mm_setzero_ps();
    for (int k0 = 0; k0 < nblocks; k0 += 2) {   // nblocks even (canon_layer)
        alignas(32) int32_t ev[NT][8];   // the even block's sums, in memory: registers stay for the decode
        __m256i ev1 = _mm256_setzero_si256(), od[NT];   // one token: ev1
        for (int b = 0; b < 2; ++b) {
            const int i = k0 + b;
            const uint8_t* blk = row + (size_t) i * Fmt32<TY>::bytes;
            rows_ahead(blk, pf);
            __m256i acci[NT];
            for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
            for (int j = 0; j < 4; ++j)
                for (int half = 0; half < 2; ++half) {
                    __m256i g, sgn, sc;
                    Fmt32<TY>::decode(blk, j, half, g, sgn, sc);
                    const int off = 256 * i + 64 * j + 32 * half;
                    for (int t = 0; t < NT; ++t) {
                        const __m256i yv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[t] + off));
                        acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(g, _mm256_sign_epi8(yv, sgn)), sc);
                    }
                }
            for (int t = 0; t < NT; ++t) {
                if (b == 1) od[t] = acci[t];
                else if constexpr (NT == 1) ev1 = acci[t];
                else _mm256_store_si256(reinterpret_cast<__m256i*>(ev[t]), acci[t]);
            }
        }
        const uint8_t* b0 = row + (size_t) k0 * Fmt32<TY>::bytes;
        const short h0 = (short) u16(b0), h1 = (short) u16(b0 + Fmt32<TY>::bytes);
        const __m128 cw = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, h1, h0, h1, h0));
        for (int p = 0; p < NP; ++p) {
            const int t = 2 * p, u = t + 1 < NT ? t + 1 : t;
            __m128 T;
            if constexpr (NT == 1) T = pair_sums1(ev1, od[0]);
            else T = pair_sums2(_mm256_load_si256(reinterpret_cast<const __m256i*>(ev[t])), od[t],
                                _mm256_load_si256(reinterpret_cast<const __m256i*>(ev[u])), od[u]);
            const __m128 dx = _mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd(reinterpret_cast<const double*>(xd[t] + k0)),
                                                            _mm_load_sd(reinterpret_cast<const double*>(xd[u] + k0))));
            acc[p] = _mm_fmadd_ps(_mm_mul_ps(_mm_mul_ps(cw, dx), _mm_set1_ps(Fmt32<TY>::K)), T, acc[p]);
        }
    }
    for (int p = 0; p < NP; ++p) {
        res[2 * (size_t) p] = canon_pair_sum(acc[p], 0);
        if (2 * p + 1 < NT) res[2 * p + 1] = canon_pair_sum(acc[p], 1);
    }
}

// The fixed-point canonical rows (canon_math.hpp's fx_*; iq_kernels.cpp's seg_dot_canon and iq_mmq.cpp's FX bodies take
// the same steps).  A block's eight 32-value units' (2 ls + 1)-weighted integer sums added first, as above,
// then that sum times the block's integer scale (the activation's and the block's d at the row's exponent) added in
// int64; the row's value from its sum.  Every addition is exact, so no order is the GPU's to follow.  IQ2_XXS, IQ3_XXS
// and IQ3_S (and their gathered decodes).
template <int TY> constexpr bool canon_fx() {
    return TY == 16 || TY == 18 || TY == 118 || TY == 21 || TY == 121 || TY == 22 || TY == 122;
}
template <int TY> constexpr int fx_k_log2() {
    return Fmt32<TY>::K == 0.125f ? -3 : Fmt32<TY>::K == 0.25f ? -2 : Fmt32<TY>::K == 0.5f ? -1 : 0;
}
// the activations' blocks a row of these rows takes at most (canon_layer: n_embd's values within kNativeActBytes)
constexpr int kFxBlocks = 32;

// xp[t][i]: token t's block i's scale as fx parts, topa[t] its largest exponent
template <int TY, int NT> STRATA_ROWS_FN
inline void row_dot_canon_fx(const uint8_t* row, int nblocks, const int8_t* const* xq,
                             const canon::FxPart (*xp)[kFxBlocks], const int* topa, float* res) {
    const int pf = prefetch_distance();
    int topw = canon::kFxNone;
    for (int i = 0; i < nblocks; ++i) {
        const int e = canon::fx_exp(h2f(u16(row + (size_t) i * Fmt32<TY>::bytes)));
        topw = e > topw ? e : topw;
    }
    topw = canon::fx_row_top(canon::fx_exp(h2f(u16(row))), topw);
    int64_t acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = 0;
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt32<TY>::bytes;
        rows_ahead(blk, pf);
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j)
            for (int half = 0; half < 2; ++half) {
                __m256i g, sgn, sc;
                Fmt32<TY>::decode(blk, j, half, g, sgn, sc);
                const int off = 256 * i + 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[t] + off));
                    acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(g, _mm256_sign_epi8(yv, sgn)), sc);
                }
            }
        const canon::FxPart w = canon::fx_part(h2f(u16(blk)));
        for (int t = 0; t < NT; ++t)
            acc[t] += (int64_t) canon::fx_scale(xp[t][i], w, topa[t] + topw) * (int64_t) hsum_i32(acci[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = canon::fx_value(acc[t], topa[t] + topw, fx_k_log2<TY>());
}

template <int TY, int NT> STRATA_ROWS_FN
inline void row_dot_canon(const uint8_t* row, int nblocks, const int8_t* const* xq, const float* const* xd,
                          float* res) {
    if constexpr (TY == 17) row_dot_iq2xs_impl<true, NT>(row, nblocks, nullptr, xq, xd, res, exact_sums());
    else row_dot_canon_gen<TY, NT>(row, nblocks, xq, xd, res);
}

template <int TY, int NT> STRATA_ROWS_FN
void canon_rows(const uint8_t* w, size_t row_bytes, int n, const int8_t* const* xq, const float* const* xd,
                float* const* out, int r0, int r1) {
    float res[NT];
    if (canon_fx<TY>() && exact_sums()) {   // the activations' parts once for every row
        const int nb = n / QK_K;
        canon::FxPart xp[NT][kFxBlocks];
        int topa[NT];
        for (int t = 0; t < NT; ++t) {
            topa[t] = canon::kFxNone;
            for (int i = 0; i < nb; ++i) {
                xp[t][i] = canon::fx_part(xd[t][i]);
                topa[t] = xp[t][i].e > topa[t] ? xp[t][i].e : topa[t];
            }
            topa[t] = canon::fx_row_top(xp[t][0].e, topa[t]);
        }
        for (int r = r0; r < r1; ++r) {
            row_dot_canon_fx<TY, NT>(w + (size_t) r * row_bytes, nb, xq, xp, topa, res);
            for (int t = 0; t < NT; ++t) out[t][r] = res[t];
        }
        return;
    }
    for (int r = r0; r < r1; ++r) {
        row_dot_canon<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, xq, xd, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY> STRATA_ROWS_FN
void canon_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const int8_t* const* xq, const float* const* xd,
                   float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: canon_rows<TY, 1>(w, row_bytes, n, xq, xd, out, r0, r1); break;
        case 2: canon_rows<TY, 2>(w, row_bytes, n, xq, xd, out, r0, r1); break;
        case 3: canon_rows<TY, 3>(w, row_bytes, n, xq, xd, out, r0, r1); break;
        case 4: canon_rows<TY, 4>(w, row_bytes, n, xq, xd, out, r0, r1); break;
        case 5: canon_rows<TY, 5>(w, row_bytes, n, xq, xd, out, r0, r1); break;
        case 6: canon_rows<TY, 6>(w, row_bytes, n, xq, xd, out, r0, r1); break;
        case 7: canon_rows<TY, 7>(w, row_bytes, n, xq, xd, out, r0, r1); break;
        default: canon_rows<TY, 8>(w, row_bytes, n, xq, xd, out, r0, r1); break;
    }
}

template <bool G> STRATA_ROWS_FN
void canon_type(int type, int nt, const uint8_t* w, size_t row_bytes, int n, const int8_t* const* xq,
                const float* const* xd, float* const* out, int r0, int r1) {
    switch (type) {
        case 16: canon_rows_nt<16>(nt, w, row_bytes, n, xq, xd, out, r0, r1); break;
        case 17: canon_rows_nt<17>(nt, w, row_bytes, n, xq, xd, out, r0, r1); break;
        case 18: canon_rows_nt<G ? 118 : 18>(nt, w, row_bytes, n, xq, xd, out, r0, r1); break;
        case 21: canon_rows_nt<G ? 121 : 21>(nt, w, row_bytes, n, xq, xd, out, r0, r1); break;
        case 22: canon_rows_nt<G ? 122 : 22>(nt, w, row_bytes, n, xq, xd, out, r0, r1); break;
        case 23: canon_rows_nt<23>(nt, w, row_bytes, n, xq, xd, out, r0, r1); break;
        default: break;
    }
}

template <int TY> STRATA_ROWS_FN
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

// The decode of each format for a variant: the gathered one (kIq256Gather) for IQ3_XXS, IQ3_S and IQ2_S.  The K-quants
// (10-14, kq256_*) have one decode.
template <bool G> STRATA_ROWS_FN
void gu_type(int type, int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
             float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<G ? 118 : 18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 19: gu_rows_nt<19>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<G ? 121 : 21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<G ? 122 : 22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 23: gu_rows_nt<23>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 12: gu_rows_nt<12>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 13: gu_rows_nt<13>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 14: gu_rows_nt<14>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 10: gu_rows_nt<10>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 11: gu_rows_nt<11>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

template <bool G> STRATA_ROWS_FN
void dot_type(int type, int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out,
              int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<G ? 118 : 18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 19: dot_rows_nt<19>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<G ? 121 : 21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<G ? 122 : 22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 23: dot_rows_nt<23>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 12: dot_rows_nt<12>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 13: dot_rows_nt<13>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 14: dot_rows_nt<14>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 10: dot_rows_nt<10>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 11: dot_rows_nt<11>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

// ---- IQ4_NL (type 20): 32-value blocks of 18 bytes (f16 d + 16 nibble bytes) against Q8_0 activations.
// ggml-cpu's own AVX-2 dot (ggml_vec_dot_iq4_nl_q8_0) is single-token: every token redoes the nibble ->
// kvalues_iq4nl pshufb decode and the |w| half of the signed x signed product.  Here both are computed once
// per block; every token then costs a sign, a maddubs, a madd and an fmadd.  The arithmetic is ggml's - only
// the order of the float additions differs.
template <int NT> STRATA_ROWS_FN
void iq4nl_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out,
                int r0, int r1) {
    const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const int nb = n / QK4_NL;
    const int pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            rows_ahead(blk, pf);
            const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
            const __m128i lo = _mm_and_si128(bits, m4b);                      // values 0..15
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);    // values 16..31
            const __m256i q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)),
                                                       _mm_shuffle_epi8(values, hi), 1);
            const __m256i aq = _mm256_sign_epi8(q4, q4);   // |w|: the unsigned operand of maddubs
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                const block_q8_0& b = y[t][ib];
                const __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.qs));
                const __m256i p = dot4u(aq, _mm256_sign_epi8(q8, q4));
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * h2f(b.d)), _mm256_cvtepi32_ps(p), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8(accf[t]);
    }
}

STRATA_ROWS_FN void iq4nl_nt(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, int nt,
                             float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: iq4nl_rows<1>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: iq4nl_rows<2>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: iq4nl_rows<3>(w, row_bytes, n, y, out, r0, r1); break;
        case 4: iq4nl_rows<4>(w, row_bytes, n, y, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            iq4nl_nt(w, row_bytes, n, y + t0, k, out + t0, r0, r1);
        }
    }
}

// ---- Q8_0 (type 8): 32-value blocks of 34 bytes (f16 d + 32 int8) against Q8_0 activations.  ggml-cpu's dot
// (ggml_vec_dot_q8_0_q8_0) is single-token and takes |w| and w's sign for every token; here once per block, and a
// token costs a sign, the dot4u and an fmadd.  |w| <= 127 for every block ggml's quantizer writes.
template <int NT> STRATA_ROWS_FN
inline void q80_row(const uint8_t* row, int nb, const block_q8_0* const* y, float* res) {
    const int pf = prefetch_distance();
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t* blk = row + (size_t) ib * sizeof(block_q8_0);
        rows_ahead(blk, pf);
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 2));
        const __m256i aq = _mm256_sign_epi8(q, q);
        const float dx = h2f(u16(blk));
        for (int t = 0; t < NT; ++t) {
            const block_q8_0& b = y[t][ib];
            const __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.qs));
            const __m256i p = dot4u(aq, _mm256_sign_epi8(q8, q));
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * h2f(b.d)), _mm256_cvtepi32_ps(p), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

template <int NT> STRATA_ROWS_FN
void q80_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        q80_row<NT>(w + (size_t) r * row_bytes, n / QK8_0, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int NT> STRATA_ROWS_FN
void q80_gu(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const block_q8_0* const* y, float* const* ff,
            int r0, int r1) {
    float g[NT], u[NT];
    const float lim = expert_swiglu_limit();
    for (int r = r0; r < r1; ++r) {
        q80_row<NT>(blob + (size_t) r * gu_row, n / QK8_0, y, g);
        q80_row<NT>(blob + up_off + (size_t) r * gu_row, n / QK8_0, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = swiglu_host(g[t], u[t], lim);
    }
}

STRATA_ROWS_FN void q80_rows_nt(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, int nt,
                                float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: q80_rows<1>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: q80_rows<2>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: q80_rows<3>(w, row_bytes, n, y, out, r0, r1); break;
        case 4: q80_rows<4>(w, row_bytes, n, y, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            q80_rows_nt(w, row_bytes, n, y + t0, k, out + t0, r0, r1);
        }
    }
}

STRATA_ROWS_FN void q80_gu_nt(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const block_q8_0* const* y,
                              int nt, float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: q80_gu<1>(blob, gu_row, up_off, n, y, ff, r0, r1); break;
        case 2: q80_gu<2>(blob, gu_row, up_off, n, y, ff, r0, r1); break;
        case 3: q80_gu<3>(blob, gu_row, up_off, n, y, ff, r0, r1); break;
        case 4: q80_gu<4>(blob, gu_row, up_off, n, y, ff, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            q80_gu_nt(blob, gu_row, up_off, n, y + t0, k, ff + t0, r0, r1);
        }
    }
}

#undef STRATA_ROWS_VNNI
#undef STRATA_ROWS_FN
