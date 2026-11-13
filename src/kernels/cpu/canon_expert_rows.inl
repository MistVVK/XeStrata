// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/canon_expert_rows.inl - the AVX2 rows of canon_expert.cpp.  Not a header: canon_expert.cpp includes
// it twice, STRATA_ROWS_VNNI 0 (namespace plain) and 1 (namespace vnni, AVX-VNNI's vpdpwssd and vpdpbusd for the
// integer sums: the same bits), as iq_avx2_rows.inl.

// the weights `pf` bytes ahead (iq_avx2's rows_ahead: the rows stream from DRAM)
STRATA_ROWS_FN inline void canon_ahead(const uint8_t* p, int pf) {
    if (pf <= 0) return;
    _mm_prefetch((const char*) p + pf, _MM_HINT_T0);
    _mm_prefetch((const char*) p + pf + 64, _MM_HINT_T0);
}

// Four blocks' 16 code bytes (at b0, b0 + stride, ...) as the calls' first and second 4 bytes a lane: w0 = [b0's
// dwords 0, 2, b1's 0, 2 | b2's 0, 2, b3's 0, 2], w1 the same with dwords 1, 3 (two loads a lane, one shufps each)
STRATA_ROWS_FN inline void call_dwords(const uint8_t* b0, ptrdiff_t stride, __m256i& w0, __m256i& w1) {
    const __m256 a = _mm256_castsi256_ps(_mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b0))),
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(b0 + 2 * stride)), 1));
    const __m256 b = _mm256_castsi256_ps(_mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b0 + stride))),
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(b0 + 3 * stride)), 1));
    w0 = _mm256_castps_si256(_mm256_shuffle_ps(a, b, 0x88));
    w1 = _mm256_castps_si256(_mm256_shuffle_ps(a, b, 0xDD));
}

// acc + madd(p16, sc)
STRATA_ROWS_FN inline __m256i madd_add(__m256i acc, __m256i p16, __m256i sc) {
#if STRATA_ROWS_VNNI
    return _mm256_dpwssd_avx_epi32(acc, p16, sc);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p16, sc));
#endif
}

// four u8 x s8 products per int32 lane (u8 an |IQ4_NL value| <= 127: maddubs does not saturate either)
STRATA_ROWS_FN inline __m256i dot4u(__m256i u8, __m256i s8) {
#if STRATA_ROWS_VNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u8, s8);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u8, s8), _mm256_set1_epi16(1));
#endif
}

// four int32 vectors -> their four lane sums, in order (exact)
STRATA_ROWS_FN inline __m128i hsum4_i32(__m256i a, __m256i b, __m256i c, __m256i d) {
    const __m256i ab = _mm256_hadd_epi32(a, b), cd = _mm256_hadd_epi32(c, d);
    const __m256i abcd = _mm256_hadd_epi32(ab, cd);   // [a0+a1+a2+a3 b.. c.. d.. | a4+..+a7 b.. c.. d..]
    return _mm_add_epi32(_mm256_castsi256_si128(abcd), _mm256_extracti128_si256(abcd, 1));
}

// acc + four u8 x s8 products per int32 lane
STRATA_ROWS_FN inline __m256i dot4u_add(__m256i acc, __m256i u8, __m256i s8) {
#if STRATA_ROWS_VNNI
    return _mm256_dpbusd_avx_epi32(acc, u8, s8);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u8, s8), _mm256_set1_epi16(1)));
#endif
}

// canon_math.hpp's fixed-point sums 8 lanes at a time: the same integer steps as fx_part and fx_scale, so the same A.
// The fp16 values of 8 lanes held as floats (db: their bits) as fx parts: mantissa (0 for 0), exponent, sign (-1).
struct Fx8 {
    __m256i m, e, neg;
};
STRATA_ROWS_FN inline Fx8 fx_parts8(__m256i db) {
    const __m256i ef = _mm256_and_si256(_mm256_srli_epi32(db, 23), _mm256_set1_epi32(0xFF));
    const __m256i live = _mm256_xor_si256(_mm256_cmpeq_epi32(ef, _mm256_setzero_si256()), _mm256_set1_epi32(-1));
    const __m256i m = _mm256_and_si256(
        live, _mm256_srli_epi32(_mm256_or_si256(_mm256_and_si256(db, _mm256_set1_epi32(0x7FFFFF)),
                                                _mm256_set1_epi32(0x800000)), 13));
    return {m, _mm256_sub_epi32(ef, _mm256_set1_epi32(137)), _mm256_srai_epi32(db, 31)};
}
// fx_scale of 8 lanes: the activation's mantissas and exponents (am, ae; never negative), the weights' w, and
// kFxHead - the row's top (shift0): (ma mw) 2^s, s = ea + ew - top + kFxHead, rounded half up where s < 0 (0 far
// below, as fx_scale's)
STRATA_ROWS_FN inline __m256i fx_scale8(__m256i am, __m256i ae, const Fx8& w, __m256i shift0) {
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i mm = _mm256_mullo_epi32(am, w.m);
    const __m256i sh = _mm256_add_epi32(_mm256_add_epi32(ae, w.e), shift0);
    const __m256i ns = _mm256_sub_epi32(_mm256_setzero_si256(), sh);
    const __m256i half = _mm256_sllv_epi32(one, _mm256_sub_epi32(ns, one));
    const __m256i right = _mm256_srlv_epi32(_mm256_add_epi32(mm, half), ns);
    const __m256i left = _mm256_sllv_epi32(mm, sh);
    const __m256i a = _mm256_blendv_epi8(right, left, _mm256_cmpgt_epi32(sh, _mm256_set1_epi32(-1)));
    return _mm256_sub_epi32(_mm256_xor_si256(a, w.neg), w.neg);
}
// acc (4 int64 lanes) + the 8 lanes' products A T
STRATA_ROWS_FN inline __m256i fx_mac8(__m256i acc, __m256i A, __m256i T) {
    const __m256i odd = _mm256_mul_epi32(_mm256_srli_epi64(A, 32), _mm256_srli_epi64(T, 32));
    return _mm256_add_epi64(acc, _mm256_add_epi64(_mm256_mul_epi32(A, T), odd));
}
STRATA_ROWS_FN inline int64_t hsum_i64(__m256i v) {
    alignas(32) int64_t l[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(l), v);
    return l[0] + l[1] + l[2] + l[3];
}
// 4 ints at p, each twice: [p0, p0, p1, p1, p2, p2, p3, p3]
STRATA_ROWS_FN inline __m256i dup4_i32(const int32_t* p) {
    return _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))),
                                       _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
}

// one IQ4_NL row against NT tokens' activations in IQ4_NL's call order (canon_quant's iq4nl_calls: four blocks' 128
// values as 4 vectors, lane k = 2 b + c of vector m holding block b's call c's m-th 4 values), with a scale for each
// block (fa: its fx parts).  The weights are decoded into the same order (block b's bytes 0-3 and 8-11 give vectors 0
// and 2, 4-7 and 12-15 vectors 1 and 3), so the 4 products add vertically into the 8 calls' integer sums; each call's
// sum times its block's integer scale (canon_math.hpp's fx_scale, 8 lanes at once: the same integer steps) is added in
// int64, and the row's value is canon_math.hpp's fx_value - the GPU's (iq_kernels.cpp's seg_dot_canon_fx, iq_mmq.cpp's
// FX bodies) in any order.
template <int NT> STRATA_ROWS_FN void iq4nl_rows_nt(const uint8_t* row, int n, const int8_t* const* x, const FxAct* fa,
                                                    float* res) {
    const __m256i kv = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl)));
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i dup = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    const __m256i zero = _mm256_setzero_si256();
    const int nb = n / QK4_NL;   // a multiple of 4 (canon_layer)
    const auto* wb = reinterpret_cast<const block_iq4_nl*>(row);
    int topw = canon::kFxNone;
    for (int k = 0; k < nb; ++k) topw = std::max(topw, canon::fx_exp(_cvtsh_ss(wb[k].d)));
    topw = canon::fx_row_top(canon::fx_exp(_cvtsh_ss(wb[0].d)), topw);
    __m256i acc[NT];   // int64 x 4
    __m256i shift0[NT];   // kFxHead - the row's top, a lane each
    for (int t = 0; t < NT; ++t) {
        acc[t] = zero;
        shift0[t] = _mm256_set1_epi32(canon::kFxHead - (fa[t].top + topw));
    }
    const int pf = iq_prefetch_bytes();
    for (int k0 = 0; k0 < nb; k0 += 4) {
        const block_iq4_nl* w = wb + k0;
        canon_ahead(reinterpret_cast<const uint8_t*>(w), pf);
        __m256i b02, b13;
        call_dwords(w[0].qs, (ptrdiff_t) sizeof(block_iq4_nl), b02, b13);
        __m256i wv[4], wa[4];
        wv[0] = _mm256_shuffle_epi8(kv, _mm256_and_si256(b02, m4));
        wv[1] = _mm256_shuffle_epi8(kv, _mm256_and_si256(b13, m4));
        wv[2] = _mm256_shuffle_epi8(kv, _mm256_and_si256(_mm256_srli_epi16(b02, 4), m4));
        wv[3] = _mm256_shuffle_epi8(kv, _mm256_and_si256(_mm256_srli_epi16(b13, 4), m4));
        for (int m = 0; m < 4; ++m) wa[m] = _mm256_sign_epi8(wv[m], wv[m]);
        // the blocks' d as fx parts, [d0, d0, d1, d1, d2, d2, d3, d3]: a block's two calls share them
        const __m256i db = _mm256_permutevar8x32_epi32(
            _mm256_castsi128_si256(_mm_castps_si128(_mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, (short) w[3].d,
                                                                              (short) w[2].d, (short) w[1].d,
                                                                              (short) w[0].d)))), dup);
        const Fx8 wp = fx_parts8(db);
        for (int t = 0; t < NT; ++t) {
            const int8_t* xb = x[t] + 32 * (ptrdiff_t) k0;
            __m256i T = _mm256_setzero_si256();
            for (int m = 0; m < 4; ++m) {
                const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb + 32 * (ptrdiff_t) m));
                T = dot4u_add(T, wa[m], _mm256_sign_epi8(y, wv[m]));
            }
            acc[t] = fx_mac8(acc[t], fx_scale8(dup4_i32(fa[t].m + k0), dup4_i32(fa[t].e + k0), wp, shift0[t]), T);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = canon::fx_value(hsum_i64(acc[t]), fa[t].top + topw, 0);
}

// Q2_0, four blocks (8 calls of 32 values) at b0: c[2 p + h] the codes of plane p of the calls' first (h 0) or second
// 4 bytes, lane k call k; dw [d0, d0, d1, d1, d2, d2, d3, d3]
STRATA_ROWS_FN inline void q20_step_codes(const uint8_t* b0, __m256i (&c)[8], __m256& dw) {
    const __m256i m3 = _mm256_set1_epi8(3);
    __m256i w0, w1;
    call_dwords(b0 + 2, 18, w0, w1);
    for (int p = 0; p < 4; ++p) {
        c[2 * (size_t) p] = _mm256_and_si256(_mm256_srli_epi16(w0, 2 * p), m3);
        c[2 * (size_t) p + 1] = _mm256_and_si256(_mm256_srli_epi16(w1, 2 * p), m3);
    }
    uint16_t h[4];
    for (int b = 0; b < 4; ++b) std::memcpy(&h[b], b0 + 18 * (ptrdiff_t) b, 2);
    dw = _mm256_permutevar8x32_ps(
        _mm256_castps128_ps256(_mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, (short) h[3], (short) h[2], (short) h[1],
                                                          (short) h[0]))), _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
}

// the 8 calls' exact (code - 1) x q sums against 256 values in Q2_0's order at xb (qs: their 8 sums of q)
STRATA_ROWS_FN inline __m256i q20_step_sums(const __m256i (&c)[8], const int8_t* xb, const int32_t* qs) {
#if STRATA_ROWS_VNNI
    __m256i T = _mm256_setzero_si256();
    for (int m = 0; m < 8; ++m)
        T = _mm256_dpbusd_avx_epi32(T, c[m], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb + 32 * (ptrdiff_t) m)));
#else
    __m256i s16[2];   // two chains of int16 sums (|sum| <= 4 x 762 each)
    for (int m = 0; m < 2; ++m)
        s16[m] = _mm256_maddubs_epi16(c[m], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb + 32 * (ptrdiff_t) m)));
    for (int m = 2; m < 8; ++m)
        s16[m & 1] = _mm256_add_epi16(s16[m & 1], _mm256_maddubs_epi16(c[m], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb + 32 * (ptrdiff_t) m))));
    const __m256i T = _mm256_madd_epi16(_mm256_add_epi16(s16[0], s16[1]), _mm256_set1_epi16(1));
#endif
    return _mm256_sub_epi32(T, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs)));
}

// A Q2_0 row's part of its top (canon_math.hpp's fx_row_top), nb blocks of 64
STRATA_ROWS_FN inline int q20_topw(const uint8_t* row, int nb) {
    int top = canon::kFxNone;
    for (int b = 0; b < nb; ++b) {
        uint16_t h;
        std::memcpy(&h, row + 18 * (ptrdiff_t) b, 2);
        top = std::max(top, canon::fx_exp(_cvtsh_ss(h)));
    }
    uint16_t h0;
    std::memcpy(&h0, row, 2);
    return canon::fx_row_top(canon::fx_exp(_cvtsh_ss(h0)), top);
}

// one Q2_0 row against NT tokens' activations in Q2_0's order (canon_quant's q2_planes), with a scale for each 32
// (fa: their fx parts) and their 32-value integer sums (qs32): each call of 32 values is (code - 1) x q summed as
// code x q - sum(q), exact, times its integer scale (canon_math.hpp's fx_scale: the call's activation scale and its
// block's d) added in int64.
//   Four blocks (8 calls) a step: 8 vectors (plane p of the calls' first or second 4 bytes), lane k of each holding
// call k's codes, so the 8 products add vertically into the 8 calls' sums (without VNNI in int16, then one madd).
// A last pair of blocks (n / 64 not a multiple of 4) as two 64-value plane blocks (value 4 k + p at 16 p + k).
template <int NT> STRATA_ROWS_FN void q20_rows_nt(const uint8_t* row, int n, const int8_t* const* x, const FxAct* fa,
                                                  const int32_t* const* qs32, float* res) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const int nb = n / 64;   // even (canon_layer: n a multiple of 128)
    const int topw = q20_topw(row, nb);
    __m256i acc[NT], shift0[NT];
    int64_t tail[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm256_setzero_si256();
        shift0[t] = _mm256_set1_epi32(canon::kFxHead - (fa[t].top + topw));
        tail[t] = 0;
    }
    const int pf = iq_prefetch_bytes();
    int k0 = 0;
    for (; k0 + 4 <= nb; k0 += 4) {
        const uint8_t* b0 = row + (size_t) k0 * 18;
        canon_ahead(b0, pf);
        __m256i c[8];
        __m256 dw;
        q20_step_codes(b0, c, dw);
        const Fx8 wp = fx_parts8(_mm256_castps_si256(dw));
        for (int t = 0; t < NT; ++t) {
            const __m256i T = q20_step_sums(c, x[t] + 64 * (ptrdiff_t) k0, qs32[t] + 2 * (ptrdiff_t) k0);
            // calls 2 k0 .. 2 k0 + 7 (int, not ptrdiff_t: the wider product changed these rows' code)
            const int c0 = 2 * k0;
            const __m256i am = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fa[t].m + c0));
            const __m256i ae = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fa[t].e + c0));
            acc[t] = fx_mac8(acc[t], fx_scale8(am, ae, wp, shift0[t]), T);
        }
    }
    if (k0 < nb) {   // two blocks: plane p of a block's 16 bytes, bytes 0-7 the first call's, 8-15 the second's
        const __m256i sh01 = _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2), sh23 = _mm256_setr_epi32(4, 4, 4, 4, 6, 6, 6, 6);
        const uint8_t* b0 = row + (size_t) k0 * 18;
        __m256i ca[2], cb[2];
        for (int b = 0; b < 2; ++b) {
            const __m256i q = _mm256_broadcastsi128_si256(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b0 + 18 * (ptrdiff_t) b + 2)));
            ca[b] = _mm256_and_si256(_mm256_srlv_epi32(q, sh01), m3);   // planes 0 | 1
            cb[b] = _mm256_and_si256(_mm256_srlv_epi32(q, sh23), m3);   // planes 2 | 3
        }
        uint16_t h[2];
        std::memcpy(&h[0], b0, 2);
        std::memcpy(&h[1], b0 + 18, 2);
        const canon::FxPart wpart[2] = {canon::fx_part(_cvtsh_ss(h[0])), canon::fx_part(_cvtsh_ss(h[1]))};
        for (int t = 0; t < NT; ++t) {
            const int8_t* xb = x[t] + 64 * (ptrdiff_t) k0;
            __m256i sb[2];
            for (int b = 0; b < 2; ++b) {
                const __m256i xa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb + 64 * (ptrdiff_t) b));
                const __m256i xc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb + 64 * (ptrdiff_t) b + 32));
                sb[b] = _mm256_add_epi32(dot4u(ca[b], xa), dot4u(cb[b], xc));
            }
            // lanes 0, 1, 4, 5 a block's first call, 2, 3, 6, 7 its second
            const __m256i hh = _mm256_hadd_epi32(sb[0], sb[1]);
            const __m128i ti = _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(hh), _mm256_extracti128_si256(hh, 1)),
                                             _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs32[t] + 2 * (ptrdiff_t) k0)));
            alignas(16) int32_t tv[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(tv), ti);
            const int top = fa[t].top + topw;
            for (int c = 0; c < 4; ++c) {   // call 2 k0 + c, block k0 + c / 2
                const canon::FxPart ap{(uint32_t) fa[t].m[2 * k0 + c], fa[t].e[2 * k0 + c], 0u};
                tail[t] += (int64_t) canon::fx_scale(ap, wpart[c / 2], top) * tv[c];
            }
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = canon::fx_value(hsum_i64(acc[t]) + tail[t], fa[t].top + topw, 0);
}

// one Q2_0 gate/up row against NT tokens' activations in Q2_0's order with one scale for each 256 (fa: their fx parts)
// and their 32-value sums (qs32): each call (32 values) its exact sum times its integer scale (canon_math.hpp's
// fx_scale) added in int64.  n a multiple of 256.
template <int NT> STRATA_ROWS_FN void q20_gu_rows_nt(const uint8_t* row, int n, const int8_t* const* x, const FxAct* fa,
                                                     const int32_t* const* qs32, float* res) {
    const int nb = n / 64;
    const int topw = q20_topw(row, nb);
    __m256i acc[NT], shift0[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm256_setzero_si256();
        shift0[t] = _mm256_set1_epi32(canon::kFxHead - (fa[t].top + topw));
    }
    const int pf = iq_prefetch_bytes();
    for (int k0 = 0; k0 < nb; k0 += 4) {
        const uint8_t* b0 = row + (size_t) k0 * 18;
        canon_ahead(b0, pf);
        __m256i c[8];
        __m256 dw;
        q20_step_codes(b0, c, dw);
        const Fx8 wp = fx_parts8(_mm256_castps_si256(dw));
        for (int t = 0; t < NT; ++t) {
            const __m256i T = q20_step_sums(c, x[t] + 64 * (ptrdiff_t) k0, qs32[t] + 2 * (ptrdiff_t) k0);
            const __m256i am = _mm256_set1_epi32(fa[t].m[k0 >> 2]), ae = _mm256_set1_epi32(fa[t].e[k0 >> 2]);
            acc[t] = fx_mac8(acc[t], fx_scale8(am, ae, wp, shift0[t]), T);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = canon::fx_value(hsum_i64(acc[t]), fa[t].top + topw, 0);
}

// ---- the float canonical rows (exact.hpp's default): each call's or block's scale product times its integer sum in
// the GPU's lane order, the sums added as the GPU's butterfly

// one IQ4_NL row against NT tokens' activations in IQ4_NL's call order (canon_quant's iq4nl_calls: four blocks' 128
// values as 4 vectors, lane k = 2 b + c of vector m holding block b's call c's m-th 4 values), with a scale for each
// block (dx).  The GPU's 8 lanes a row: call k (16 values: block k / 2's values 0-7 and 16-23, or 8-15 and 24-31) on
// lane k % 8, its scale product x its integer sum added with one fma.  The weights are decoded into the same order
// (block b's bytes 0-3 and 8-11 give vectors 0 and 2, 4-7 and 12-15 vectors 1 and 3), so the 4 products add
// vertically into the 8 calls' sums; at the end the GPU's butterfly, ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)).
template <int NT> STRATA_ROWS_FN void iq4nl_rows_nt_f(const uint8_t* row, int n, const int8_t* const* x, const float* const* dx,
                                     float* res) {
    STRATA_CANON_NO_CONTRACT
    const __m256i kv = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl)));
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i dup = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    const int nb = n / QK4_NL;   // a multiple of 4 (canon_layer)
    const int pf = iq_prefetch_bytes();
    for (int k0 = 0; k0 < nb; k0 += 4) {
        const auto* w = reinterpret_cast<const block_iq4_nl*>(row) + k0;
        canon_ahead(reinterpret_cast<const uint8_t*>(w), pf);
        __m256i b02, b13;
        call_dwords(w[0].qs, (ptrdiff_t) sizeof(block_iq4_nl), b02, b13);
        __m256i wv[4], wa[4];
        wv[0] = _mm256_shuffle_epi8(kv, _mm256_and_si256(b02, m4));
        wv[1] = _mm256_shuffle_epi8(kv, _mm256_and_si256(b13, m4));
        wv[2] = _mm256_shuffle_epi8(kv, _mm256_and_si256(_mm256_srli_epi16(b02, 4), m4));
        wv[3] = _mm256_shuffle_epi8(kv, _mm256_and_si256(_mm256_srli_epi16(b13, 4), m4));
        for (int m = 0; m < 4; ++m) wa[m] = _mm256_sign_epi8(wv[m], wv[m]);
        // [d0, d0, d1, d1, d2, d2, d3, d3]: a block's two calls share its scale
        const __m256 dw = _mm256_cvtph_ps(_mm_set_epi16((short) w[3].d, (short) w[3].d, (short) w[2].d, (short) w[2].d,
                                                        (short) w[1].d, (short) w[1].d, (short) w[0].d, (short) w[0].d));
        for (int t = 0; t < NT; ++t) {
            const int8_t* xb = x[t] + 32 * (ptrdiff_t) k0;
            __m256i T = _mm256_setzero_si256();
            for (int m = 0; m < 4; ++m) {
                const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb + 32 * (ptrdiff_t) m));
                T = dot4u_add(T, wa[m], _mm256_sign_epi8(y, wv[m]));
            }
            const __m256 xs = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(_mm_loadu_ps(dx[t] + k0)), dup);
            acc[t] = _mm256_fmadd_ps(_mm256_mul_ps(dw, xs), _mm256_cvtepi32_ps(T), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m128 v = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));   // l + (l + 4)
        const __m128 u = _mm_add_ps(v, _mm_movehl_ps(v, v));                                              // + (l + 2)
        res[t] = _mm_cvtss_f32(u) + _mm_cvtss_f32(_mm_shuffle_ps(u, u, 1));
    }
}

// one Q2_0 row against NT tokens' activations in Q2_0's order (canon_quant's q2_planes), with a scale for each 32
// (dx) and their 32-value integer sums (qs32): each call of 32 values is (code - 1) x q summed as code x q - sum(q),
// exact, then d x the scale x that sum into the sum of the call index modulo 4 (the GPU's 4 lanes a row); at the end
// (lane 0 + lane 2) + (lane 1 + lane 3).
//   Four blocks (8 calls) a step: 8 vectors (plane p of the calls' first or second 4 bytes), lane k of each holding
// call k's codes, so the 8 products add vertically into the 8 calls' sums (without VNNI in int16, then one madd).
// A last pair of blocks (n / 64 not a multiple of 4) as two 64-value plane blocks (value 4 k + p at 16 p + k): the
// 4-call form in 128 bits was 8% slower there with AVX-VNNI.
template <int NT> STRATA_ROWS_FN void q20_rows_nt_f(const uint8_t* row, int n, const int8_t* const* x,
                                                  const float* const* dx, const int32_t* const* qs32, float* res) {
    STRATA_CANON_NO_CONTRACT
    const __m256i m3 = _mm256_set1_epi8(3);
    __m128 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm_setzero_ps();
    const int nb = n / 64;   // even (canon_layer: n a multiple of 128)
    const int pf = iq_prefetch_bytes();
    int k0 = 0;
    for (; k0 + 4 <= nb; k0 += 4) {
        const uint8_t* b0 = row + (size_t) k0 * 18;
        canon_ahead(b0, pf);
        __m256i c[8];
        __m256 dw;
        q20_step_codes(b0, c, dw);
        for (int t = 0; t < NT; ++t) {
            const __m256i T = q20_step_sums(c, x[t] + 64 * (ptrdiff_t) k0, qs32[t] + 2 * (ptrdiff_t) k0);
            const __m256 cT = _mm256_mul_ps(dw, _mm256_loadu_ps(dx[t] + 2 * (ptrdiff_t) k0));   // calls 2 k0 .. + 7
            const __m256 Tf = _mm256_cvtepi32_ps(T);
            acc[t] = _mm_fmadd_ps(_mm256_castps256_ps128(cT), _mm256_castps256_ps128(Tf), acc[t]);
            acc[t] = _mm_fmadd_ps(_mm256_extractf128_ps(cT, 1), _mm256_extractf128_ps(Tf, 1), acc[t]);
        }
    }
    if (k0 < nb) {   // two blocks: plane p of a block's 16 bytes, bytes 0-7 the first call's, 8-15 the second's
        const __m256i sh01 = _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2), sh23 = _mm256_setr_epi32(4, 4, 4, 4, 6, 6, 6, 6);
        const uint8_t* b0 = row + (size_t) k0 * 18;
        __m256i ca[2], cb[2];
        for (int b = 0; b < 2; ++b) {
            const __m256i q = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b0 + 18 * (ptrdiff_t) b + 2)));
            ca[b] = _mm256_and_si256(_mm256_srlv_epi32(q, sh01), m3);   // planes 0 | 1
            cb[b] = _mm256_and_si256(_mm256_srlv_epi32(q, sh23), m3);   // planes 2 | 3
        }
        uint16_t h0, h1;
        std::memcpy(&h0, b0, 2);
        std::memcpy(&h1, b0 + 18, 2);
        const __m128 dw = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, (short) h1, (short) h1, (short) h0, (short) h0));
        for (int t = 0; t < NT; ++t) {
            const int8_t* xb = x[t] + 64 * (ptrdiff_t) k0;
            __m256i sb[2];
            for (int b = 0; b < 2; ++b) {
                const __m256i xa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb + 64 * (ptrdiff_t) b));
                const __m256i xc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb + 64 * (ptrdiff_t) b + 32));
                sb[b] = _mm256_add_epi32(dot4u(ca[b], xa), dot4u(cb[b], xc));
            }
            // lanes 0, 1, 4, 5 a block's first call, 2, 3, 6, 7 its second
            const __m256i hh = _mm256_hadd_epi32(sb[0], sb[1]);
            const __m128i ti = _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(hh), _mm256_extracti128_si256(hh, 1)),
                                             _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs32[t] + 2 * (ptrdiff_t) k0)));
            acc[t] = _mm_fmadd_ps(_mm_mul_ps(dw, _mm_loadu_ps(dx[t] + 2 * (ptrdiff_t) k0)), _mm_cvtepi32_ps(ti), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m128 v = _mm_add_ps(acc[t], _mm_movehl_ps(acc[t], acc[t]));
        res[t] = _mm_cvtss_f32(v) + _mm_cvtss_f32(_mm_shuffle_ps(v, v, 1));
    }
}

// one Q2_0 gate/up row against NT tokens' activations in Q2_0's order with one scale for each 256 (dx) and their
// 32-value sums (qs32): each call (32 values) d x the scale x its exact sum into the sum of the call index modulo 16
// (the GPU's 16 lanes a gate/up row; two 8-lane sums, the steps of 8 calls in turn); at the end the GPU's
// butterfly, ((l + l + 8) + (l + 4)) + ..., in four steps.  n a multiple of 256.
template <int NT> STRATA_ROWS_FN void q20_gu_rows_nt_f(const uint8_t* row, int n, const int8_t* const* x,
                                                     const float* const* dx, const int32_t* const* qs32, float* res) {
    STRATA_CANON_NO_CONTRACT
    __m256 acc[NT][2];
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = _mm256_setzero_ps();
    const int nb = n / 64;
    const int pf = iq_prefetch_bytes();
    for (int k0 = 0; k0 < nb; k0 += 4) {
        const uint8_t* b0 = row + (size_t) k0 * 18;
        canon_ahead(b0, pf);
        __m256i c[8];
        __m256 dw;
        q20_step_codes(b0, c, dw);
        const int h = (k0 >> 2) & 1;   // calls 2 k0 .. + 7 are lanes 0-7 or 8-15 of the 16
        for (int t = 0; t < NT; ++t) {
            const __m256i T = q20_step_sums(c, x[t] + 64 * (ptrdiff_t) k0, qs32[t] + 2 * (ptrdiff_t) k0);
            const __m256 cT = _mm256_mul_ps(dw, _mm256_set1_ps(dx[t][k0 >> 2]));
            acc[t][h] = _mm256_fmadd_ps(cT, _mm256_cvtepi32_ps(T), acc[t][h]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m256 w = _mm256_add_ps(acc[t][0], acc[t][1]);                                   // l + (l + 8)
        const __m128 u = _mm_add_ps(_mm256_castps256_ps128(w), _mm256_extractf128_ps(w, 1));    // + (l + 4)
        const __m128 v = _mm_add_ps(u, _mm_movehl_ps(u, u));                                    // + (l + 2)
        res[t] = _mm_cvtss_f32(v) + _mm_cvtss_f32(_mm_shuffle_ps(v, v, 1));                     // + (l + 1)
    }
}
