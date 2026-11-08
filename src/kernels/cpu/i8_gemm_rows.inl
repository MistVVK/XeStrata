// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/i8_gemm_rows.inl - the row kernels of i8_gemm_vnni.cpp.  Not a header: i8_gemm_vnni.cpp includes it
// four times,
//
//   STRATA_I8_ROWS_VNNI 0, STRATA_I8_ROWS_U8 0   the AVX2 forms (namespace avx2): vpmaddubsw + vpmaddwd on the
//                           signed levels, for eight-bit weights;
//   STRATA_I8_ROWS_VNNI 0, STRATA_I8_ROWS_U8 1   the AVX2 forms for seven-bit weights (namespace avx2_w7): vpmaddubsw
//                           + vpmaddwd on the unsigned activation as it is, the row-sum correction taken off;
//   STRATA_I8_ROWS_VNNI 1, STRATA_I8_ROWS_U8 1   the same loops with AVX-VNNI (namespace vnni, built with
//                           STRATA_AVXVNNI): vpdpbusd on the unsigned activation, the correction taken off;
//   STRATA_I8_ROWS_VNNI 2, STRATA_I8_ROWS_U8 0   with AVX-VNNI-INT8 (namespace vnni_int8, built with
//                           STRATA_AVXVNNIINT8): vpdpbssd on the activation quantized signed (i8_gemm_quantize for
//                           I8GemmIsa::vnni_int8), no correction.
//
// Integer sums, so the same bits.
// Every function here is STRATA_I8_ROWS_FN: nothing in the AVX2 copy, GCC/Clang's target("avxvnni") in the VNNI copy,
// so the AVX-VNNI intrinsics compile in these functions only and the AVX2 copy stays AVX2.

/// One quad's products into the eight lanes: `a` the activation's four bytes (with their +128) in every lane, `w`
/// the panel's 32 bytes.  The VNNI and the seven-bit copies add sum (qa + 128) * qw; the eight-bit AVX2 copy adds
/// sum qa * qw.
STRATA_I8_ROWS_FN inline __m256i dot4(__m256i acc, __m256i a, __m256i w) {
#if STRATA_I8_ROWS_VNNI == 1
    return _mm256_dpbusd_avx_epi32(acc, a, w);
#elif STRATA_I8_ROWS_VNNI == 2
    return _mm256_dpbssd_epi32(acc, a, w);
#elif STRATA_I8_ROWS_U8
    // |qw| <= 63: a pair is at most 2 * 255 * 63 = 32130, so vpmaddubsw's int16 cannot saturate (the header)
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), _mm256_set1_epi16(1)));
#else
    const __m256i s = _mm256_xor_si256(a, _mm256_set1_epi8((char) 0x80));   // the signed levels
    const __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(s), _mm256_sign_epi8(w, s));
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
#endif
}

/// One panel against `ROWS` token rows (1..8): `xq` is the first of them and `xs` its scales (`nb` a row), `y` the
/// outputs' row for lane j at `base + j`.  `cols` is how many of the panel's eight lanes are real (the last panel of
/// a matrix whose n_out is not a multiple of eight); the padding lanes compute zero times zero and are not stored.
/// `ws` and `wc` are the panel's scales and corrections, `[block][lane]`.
template <int ROWS>
STRATA_I8_ROWS_FN inline void tile(const uint8_t* wp, int quads, int nb, int bq, const uint8_t* xq, size_t ldq,
                                   const float* xs, float* y, size_t ldy, int base, int cols, const float* ws,
                                   const int32_t* wc, const float* bias) {
    __m256 bv = _mm256_setzero_ps();
    if (bias != nullptr) {
        if (cols == kI8Panel) {
            bv = _mm256_loadu_ps(bias + base);
        } else {
            float bpad[kI8Panel];
            for (int j = 0; j < kI8Panel; ++j) bpad[j] = j < cols ? bias[base + j] : 0.0f;
            bv = _mm256_loadu_ps(bpad);
        }
    }
    __m256 f[ROWS];
    for (int b = 0; b < nb; ++b) {
        __m256i acc[ROWS];
        for (int r = 0; r < ROWS; ++r) acc[r] = _mm256_setzero_si256();
        const int q1 = std::min(quads, (b + 1) * bq);
        for (int kq = b * bq; kq < q1; ++kq) {
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + (size_t) kq * kI8QuadBytes));
            const uint8_t* a = xq + (size_t) kq * kI8Quad;
            for (int r = 0; r < ROWS; ++r)
                acc[r] = dot4(acc[r], _mm256_set1_epi32((int) quad_bytes(a + (size_t) r * ldq)), w);
        }
        // Both side arrays are sized in panel lanes, so a partial panel's pad lanes read the 0 the packer left.
        const __m256 wsv = _mm256_loadu_ps(ws + (size_t) b * kI8Panel);
        const __m256i cv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wc + (size_t) b * kI8Panel));
        for (int r = 0; r < ROWS; ++r) {
            const __m256 d = _mm256_cvtepi32_ps(STRATA_I8_ROWS_U8 ? _mm256_sub_epi32(acc[r], cv) : acc[r]);
            const __m256 s = _mm256_mul_ps(_mm256_set1_ps(xs[(size_t) r * nb + b]), wsv);
            if (b == 0) f[r] = bias != nullptr ? _mm256_fmadd_ps(s, d, bv) : _mm256_mul_ps(s, d);
            else f[r] = _mm256_fmadd_ps(s, d, f[r]);
        }
    }
    for (int r = 0; r < ROWS; ++r) {
        float* yr = y + (size_t) r * ldy + base;
        if (cols == kI8Panel) {
            _mm256_storeu_ps(yr, f[r]);
        } else {
            alignas(32) float out[kI8Panel];
            _mm256_store_ps(out, f[r]);
            std::memcpy(yr, out, (size_t) cols * sizeof(float));
        }
    }
}

/// Two whole panels side by side (16 outputs) against `ROWS` token rows, with more than one scale block: each
/// activation broadcast feeds both panels' products, so a quad costs two weight loads and ROWS broadcasts for 2 * ROWS
/// products where one panel's costs one and ROWS for ROWS.  The arithmetic of each panel is tile's, so the same bits.
template <int ROWS>
STRATA_I8_ROWS_FN inline void tile2(const uint8_t* wp, int quads, int nb, int bq, const uint8_t* xq, size_t ldq,
                                    const float* xs, float* y, size_t ldy, int base, const float* ws,
                                    const int32_t* wc, const float* bias) {
    const uint8_t* wp1 = wp + (size_t) quads * kI8QuadBytes;
    const float* ws1 = ws + (size_t) nb * kI8Panel;
    const int32_t* wc1 = wc + (size_t) nb * kI8Panel;
    const __m256 bv0 = bias != nullptr ? _mm256_loadu_ps(bias + base) : _mm256_setzero_ps();
    const __m256 bv1 = bias != nullptr ? _mm256_loadu_ps(bias + base + kI8Panel) : _mm256_setzero_ps();
    __m256 f0[ROWS], f1[ROWS];
    for (int b = 0; b < nb; ++b) {
        __m256i a0[ROWS], a1[ROWS];
        for (int r = 0; r < ROWS; ++r) a0[r] = a1[r] = _mm256_setzero_si256();
        const int q1 = std::min(quads, (b + 1) * bq);
        for (int kq = b * bq; kq < q1; ++kq) {
            const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + (size_t) kq * kI8QuadBytes));
            const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp1 + (size_t) kq * kI8QuadBytes));
            const uint8_t* a = xq + (size_t) kq * kI8Quad;
            for (int r = 0; r < ROWS; ++r) {
                const __m256i x = _mm256_set1_epi32((int) quad_bytes(a + (size_t) r * ldq));
                a0[r] = dot4(a0[r], x, w0);
                a1[r] = dot4(a1[r], x, w1);
            }
        }
        const __m256 s0 = _mm256_loadu_ps(ws + (size_t) b * kI8Panel), s1 = _mm256_loadu_ps(ws1 + (size_t) b * kI8Panel);
        const __m256i c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wc + (size_t) b * kI8Panel));
        const __m256i c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wc1 + (size_t) b * kI8Panel));
        for (int r = 0; r < ROWS; ++r) {
            const __m256 xsv = _mm256_set1_ps(xs[(size_t) r * nb + b]);
            const __m256 d0 = _mm256_cvtepi32_ps(STRATA_I8_ROWS_U8 ? _mm256_sub_epi32(a0[r], c0) : a0[r]);
            const __m256 d1 = _mm256_cvtepi32_ps(STRATA_I8_ROWS_U8 ? _mm256_sub_epi32(a1[r], c1) : a1[r]);
            const __m256 t0 = _mm256_mul_ps(xsv, s0), t1 = _mm256_mul_ps(xsv, s1);
            if (b == 0) {
                f0[r] = bias != nullptr ? _mm256_fmadd_ps(t0, d0, bv0) : _mm256_mul_ps(t0, d0);
                f1[r] = bias != nullptr ? _mm256_fmadd_ps(t1, d1, bv1) : _mm256_mul_ps(t1, d1);
            } else {
                f0[r] = _mm256_fmadd_ps(t0, d0, f0[r]);
                f1[r] = _mm256_fmadd_ps(t1, d1, f1[r]);
            }
        }
    }
    for (int r = 0; r < ROWS; ++r) {
        _mm256_storeu_ps(y + (size_t) r * ldy + base, f0[r]);
        _mm256_storeu_ps(y + (size_t) r * ldy + base + kI8Panel, f1[r]);
    }
}

/// Two whole panels' token rows, kPairRows at a time, the last few through tile2 of their count.
STRATA_I8_ROWS_FN inline void rows_of_pair(const uint8_t* wp, int quads, int nb, int bq, const uint8_t* xq, size_t ldq,
                                           const float* xs, float* y, size_t ldy, int base, int rows, const float* ws,
                                           const int32_t* wc, const float* bias) {
    int r = 0;
    for (; r + kPairRows <= rows; r += kPairRows)
        tile2<kPairRows>(wp, quads, nb, bq, xq + (size_t) r * ldq, ldq, xs + (size_t) r * nb, y + (size_t) r * ldy, ldy,
                         base, ws, wc, bias);
    const int left = rows - r;
    const uint8_t* xr = xq + (size_t) r * ldq;
    const float* sr = xs + (size_t) r * nb;
    float* yr = y + (size_t) r * ldy;
    if (left == 4) tile2<4>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, ws, wc, bias);
    else if (left == 3) tile2<3>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, ws, wc, bias);
    else if (left == 2) tile2<2>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, ws, wc, bias);
    else if (left == 1) tile2<1>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, ws, wc, bias);
    static_assert(kPairRows <= 5, "the last rows below a group are at most four");
}

/// The token groups of one block of rows: `group` at a time (eight with one scale block, kBlockRows with more), then
/// the last few.
STRATA_I8_ROWS_FN inline void rows_of_panel(const uint8_t* wp, int quads, int nb, int bq, const uint8_t* xq,
                                            size_t ldq, const float* xs, float* y, size_t ldy, int base, int cols,
                                            int rows, const float* ws, const int32_t* wc, const float* bias) {
    const int group = nb == 1 ? 8 : kBlockRows;
    int r = 0;
    for (; r + group <= rows; r += group) {
        const uint8_t* xr = xq + (size_t) r * ldq;
        const float* sr = xs + (size_t) r * nb;
        float* yr = y + (size_t) r * ldy;
        if (group == 8) tile<8>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, cols, ws, wc, bias);
        else tile<kBlockRows>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, cols, ws, wc, bias);
    }
    const int left = rows - r;
    if (left <= 0) return;
    const uint8_t* xr = xq + (size_t) r * ldq;
    const float* sr = xs + (size_t) r * nb;
    float* yr = y + (size_t) r * ldy;
    switch (left) {
        case 7: tile<7>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, cols, ws, wc, bias); break;
        case 6: tile<6>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, cols, ws, wc, bias); break;
        case 5: tile<5>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, cols, ws, wc, bias); break;
        case 4: tile<4>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, cols, ws, wc, bias); break;
        case 3: tile<3>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, cols, ws, wc, bias); break;
        case 2: tile<2>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, cols, ws, wc, bias); break;
        default: tile<1>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, cols, ws, wc, bias); break;
    }
}

STRATA_I8_ROWS_FN void rows_x86(const uint8_t* panel, const float* wscale, const int32_t* wcorr, int n_in, int n_out,
                                const uint8_t* xq, size_t ldq, const float* xs, float* y, size_t ldy, int n_tok,
                                int p0, int p1, const float* bias, int block) {
    const int quads = i8_gemm_quads(n_in);
    const int nb = i8_gemm_blocks(n_in, block);
    const int bq = block <= 0 ? quads : block / kI8Quad;
    const int tb = bf16_gemm_token_block(n_in);
    for (int t0 = 0; t0 < n_tok; t0 += tb) {
        const int rows = std::min(tb, n_tok - t0);
        const uint8_t* xb = xq + (size_t) t0 * ldq;
        const float* sb = xs + (size_t) t0 * nb;
        float* yb = y + (size_t) t0 * ldy;
        for (int p = p0; p < p1;) {
            const int base = p * kI8Panel;
            const int cols = std::min(kI8Panel, n_out - base);
            const uint8_t* wp = panel + (size_t) p * (size_t) quads * (size_t) kI8QuadBytes;
            if (nb > 1 && p + 1 < p1 && base + 2 * kI8Panel <= n_out) {   // two whole panels together
                rows_of_pair(wp, quads, nb, bq, xb, ldq, sb, yb, ldy, base, rows, wscale + (size_t) p * nb * kI8Panel,
                             wcorr + (size_t) p * nb * kI8Panel, bias);
                p += 2;
                continue;
            }
            rows_of_panel(wp, quads, nb, bq, xb, ldq, sb, yb, ldy, base, cols, rows,
                          wscale + (size_t) p * nb * kI8Panel, wcorr + (size_t) p * nb * kI8Panel, bias);
            ++p;
        }
    }
}

#undef STRATA_I8_ROWS_VNNI
#undef STRATA_I8_ROWS_U8
#undef STRATA_I8_ROWS_FN
