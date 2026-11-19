// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Included in tq_avx2.cpp for AVX2, AVX-VNNI and AVX-512 VNNI.  All three keep the scalar kernel's four
// independent Q8_0 sums and the same ordered float operations.

#if BIH3_MODE != 2
// The AVX2 / AVX-VNNI copies spend their width on 16 states at once and keep the weights weight-major: the
// four 32-bit products become pairs of 16-bit halves (VPMULLW / VPMULHUW, three 16-bit products per constant:
// s is 15 bits, so s*C = mullo(s,C_lo) + (mulhi_u(s,C_lo) + mullo(s,C_hi))<<16), the three byte planes are two
// 16-bit shifts and one AND per product, and the weights are 16-bit sums - no 32-bit planes, no VPMULLD, no
// pack to activation order.  The dot reads the activations transposed to weight-major order (qtrans, once per
// call and block, amortised over the rows); two VPHADDD and one interleave leave the four exact Q8_0 sub-block
// sums in lane order (llvm-mca, cycles per 128-weight block: the old byte-plane decode8 + three-VPHADDD chain
// took ~88 on alderlake and ~94 on znver3/znver5 at NT=1, this form ~50 / ~50 / ~46).
constexpr uint8_t bih3_qshuf[4][32] = {
    {0, 1, 8, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
     0, 1, 8, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80},
    {2, 3, 10, 11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
     2, 3, 10, 11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80},
    {4, 5, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
     4, 5, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80},
    {6, 7, 14, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
     6, 7, 14, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80}};

// 64 activations -> four weight-major registers: qT[g] word j = q[4j + g].
BIH3_FN inline void qtrans(const int16_t* q, __m256i* qT) {
    const __m256i Q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
    const __m256i Q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16));
    const __m256i Q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
    const __m256i Q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 48));
    for (int g = 0; g < 4; ++g) {
        const __m256i m = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bih3_qshuf[g]));
        const __m256i t0 = _mm256_shuffle_epi8(Q0, m), t1 = _mm256_shuffle_epi8(Q1, m);
        const __m256i t2 = _mm256_shuffle_epi8(Q2, m), t3 = _mm256_shuffle_epi8(Q3, m);
        const __m256i u = _mm256_unpacklo_epi32(t0, t1), v = _mm256_unpacklo_epi32(t2, t3);
        const __m256i w = _mm256_unpacklo_epi64(u, v);
        const __m256i ws = _mm256_permute2x128_si256(w, w, 0x01);
        const __m256i r0 = _mm256_unpacklo_epi32(w, ws), r1 = _mm256_unpackhi_epi32(w, ws);
        qT[g] = _mm256_permute2x128_si256(r0, r1, 0x20);
    }
}

// 16 states as 16 unsigned 15-bit words.
BIH3_FN inline __m256i states16(const uint8_t* padded, int k, int step) {
    if (k == 8) {
        const __m128i cur = _mm_loadu_si128(reinterpret_cast<const __m128i*>(padded + step));
        const __m128i prev = step == 0 ? _mm_or_si128(_mm_slli_si128(cur, 1), _mm_cvtsi32_si128(padded[31]))
                                      : _mm_loadu_si128(reinterpret_cast<const __m128i*>(padded + step - 1));
        const __m256i pairs = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(cur, prev)),
                                                      _mm_unpackhi_epi8(cur, prev), 1);
        return _mm256_and_si256(pairs, _mm256_set1_epi16(0x7fff));
    }
    alignas(32) uint16_t st[16];
    for (int j = 0; j < 16; ++j) st[j] = (uint16_t) bih3_state_padded(padded, k, step + j);
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(st));
}

// 16 states yield w0..w3: weight g of the 16 states, signed 16-bit, activation value (sum - 384).
BIH3_FN inline void weights16(__m256i s, __m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3) {
    const __m256i la = _mm256_mullo_epi16(s, _mm256_set1_epi16((short) 0x79b1));
    const __m256i ha = _mm256_add_epi16(_mm256_mulhi_epu16(s, _mm256_set1_epi16((short) 0x79b1)),
                                        _mm256_mullo_epi16(s, _mm256_set1_epi16((short) 0x9e37)));
    const __m256i lb = _mm256_mullo_epi16(s, _mm256_set1_epi16((short) 0xca6b));
    const __m256i hb = _mm256_add_epi16(_mm256_mulhi_epu16(s, _mm256_set1_epi16((short) 0xca6b)),
                                        _mm256_mullo_epi16(s, _mm256_set1_epi16((short) 0x85eb)));
    const __m256i lc = _mm256_mullo_epi16(s, _mm256_set1_epi16((short) 0xae35));
    const __m256i hc = _mm256_add_epi16(_mm256_mulhi_epu16(s, _mm256_set1_epi16((short) 0xae35)),
                                        _mm256_mullo_epi16(s, _mm256_set1_epi16((short) 0xc2b2)));
    const __m256i ld = _mm256_mullo_epi16(s, _mm256_set1_epi16((short) 0xeb2f));
    const __m256i hd = _mm256_add_epi16(_mm256_mulhi_epu16(s, _mm256_set1_epi16((short) 0xeb2f)),
                                        _mm256_mullo_epi16(s, _mm256_set1_epi16((short) 0x27d4)));
    const __m256i mask = _mm256_set1_epi16(255), centre = _mm256_set1_epi16(384);
    const __m256i a1 = _mm256_srli_epi16(la, 8), a2 = _mm256_and_si256(ha, mask), a3 = _mm256_srli_epi16(ha, 8);
    const __m256i b1 = _mm256_srli_epi16(lb, 8), b2 = _mm256_and_si256(hb, mask), b3 = _mm256_srli_epi16(hb, 8);
    const __m256i c1 = _mm256_srli_epi16(lc, 8), c2 = _mm256_and_si256(hc, mask), c3 = _mm256_srli_epi16(hc, 8);
    const __m256i d1 = _mm256_srli_epi16(ld, 8), d2 = _mm256_and_si256(hd, mask), d3 = _mm256_srli_epi16(hd, 8);
    w0 = _mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(a1, b2), c3), centre);
    w1 = _mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(a2, b3), d1), centre);
    w2 = _mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(a3, c1), d2), centre);
    w3 = _mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(b1, c2), d3), centre);
}
#else
// The AVX-512 copy decodes 16 states at once and spends its width on the expansion, not the product: the
// byte planes and the unpack/pack chain of the old AVX2 decode8 were shuffle-port bound (llvm-mca, one
// 128-weight block, NT=1: ~90 shuffle-port cycles of ~107 total on icelake-server).  Here two masked VPERMT2B
// gather each weight's three product bytes straight into weight order, VPDPBUSD sums them (unsigned bytes
// against +1, from -384), and one VPACKSSDW per 32 weights needs no lane fix afterwards: gather group g holds
// weights 32*(g/2) + 8*(j/4) + (j%4) + 4*(g%2) in dword lane j, the interleave VPACKSSDW performs per 128-bit
// lane.  The products stay 32-bit (four VPMULLD per 16 states, half the count of decode8 per weight).
struct BiH3Gather { uint8_t ab[4][64], cd[4][64]; };
constexpr BiH3Gather bih3_gather_tables() {
    // weight i of a state: three (product, byte plane) pairs, products 0..3 = a..d, planes in bits / 8
    constexpr int src[4][3][2] = {{{0, 1}, {1, 2}, {2, 3}}, {{0, 2}, {1, 3}, {3, 1}},
                                  {{0, 3}, {2, 1}, {3, 2}}, {{1, 1}, {2, 2}, {3, 3}}};
    BiH3Gather t{};
    for (int g = 0; g < 4; ++g) for (int j = 0; j < 16; ++j) {
        const int s = 8 * (g >> 1) + 2 * (j >> 2) + (g & 1), i = j & 3;
        for (int p = 0; p < 3; ++p) {
            const int byte = 4 * s + src[i][p][1];
            if (src[i][p][0] < 2) t.ab[g][4 * j + p] = (uint8_t) (64 * src[i][p][0] + byte);
            else t.cd[g][4 * j + p] = (uint8_t) (64 * (src[i][p][0] - 2) + byte);
        }
    }
    return t;
}
constexpr BiH3Gather bih3_gather = bih3_gather_tables();
// per dword lane: which of the four bytes come from a/b (kAb) and from c/d (kCd); byte 3 stays zero
constexpr __mmask64 bih3_k_ab = 0x1133113311331133ull, bih3_k_cd = 0x6644664466446644ull;

// 16 states yield 64 signed 16-bit weights in activation order, two ZMM registers.  `padded` carries the
// block's last byte before its first, so a K=8 state reads its byte and the previous one interleaved.
BIH3_FN inline void decode16(const uint8_t* padded, int k, int step, __m512i& w0, __m512i& w1) {
    __m512i s;
    if (k == 8) {
        const __m128i cur = _mm_loadu_si128(reinterpret_cast<const __m128i*>(padded + 1 + step));
        const __m128i prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(padded + step));
        const __m256i pairs = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(cur, prev)),
                                                      _mm_unpackhi_epi8(cur, prev), 1);
        s = _mm512_cvtepu16_epi32(_mm256_and_si256(pairs, _mm256_set1_epi16(0x7fff)));
    } else {
        alignas(64) int32_t states[16];
        for (int j = 0; j < 16; ++j) states[j] = (int32_t) bih3_state_padded(padded + 1, k, step + j);
        s = _mm512_load_si512(reinterpret_cast<const __m512i*>(states));
    }
    const __m512i a = _mm512_mullo_epi32(s, _mm512_set1_epi32((int) 0x9e3779b1u));
    const __m512i b = _mm512_mullo_epi32(s, _mm512_set1_epi32((int) 0x85ebca6bu));
    const __m512i c = _mm512_mullo_epi32(s, _mm512_set1_epi32((int) 0xc2b2ae35u));
    const __m512i d = _mm512_mullo_epi32(s, _mm512_set1_epi32((int) 0x27d4eb2fu));
    const __m512i ones = _mm512_set1_epi32(0x01010101);
    const __m512i centre = _mm512_set1_epi32(-384);
    __m512i sums[4];
    for (int g = 0; g < 4; ++g) {
        const __m512i ab = _mm512_maskz_permutex2var_epi8(bih3_k_ab, a,
                                                          _mm512_loadu_si512(bih3_gather.ab[g]), b);
        const __m512i cd = _mm512_maskz_permutex2var_epi8(bih3_k_cd, c,
                                                          _mm512_loadu_si512(bih3_gather.cd[g]), d);
        sums[g] = _mm512_dpbusd_epi32(centre, _mm512_or_si512(ab, cd), ones);
    }
    w0 = _mm512_packs_epi32(sums[0], sums[1]);
    w1 = _mm512_packs_epi32(sums[2], sums[3]);
}

BIH3_FN inline __m256i dot32(__m256i lo, __m256i hi, const int16_t* q) {
    // Keep the accumulation: clang can split the intrinsic into a separate madd and add.  The AVX-512 copy
    // needs the EVEX encoding (VEX AVX-VNNI is not valid on an AVX-512 VNNI chip without AVX-VNNI, SDE -icx).
    __m256i sum;
    __asm__("vpmaddwd %3, %1, %0\n\tvpdpwssd %4, %2, %0"
            : "=&x"(sum) : "x"(hi), "x"(lo), "m"(*reinterpret_cast<const __m256i_u*>(q + 16)),
              "m"(*reinterpret_cast<const __m256i_u*>(q)));
    return sum;
}
#endif

template <int NT> BIH3_FN
void rows_t(int type, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
            float* const* out, int r0, int r1, const TqR1q2Scales& scales) {
    constexpr int nt = NT;
    const int k = type - 180, nb = n / 128;
#if BIH3_MODE != 2
    if (r0 >= r1) return;
    alignas(32) __m256i qT[NT][kTqMaxN / 128][2][4];
    for (int t = 0; t < nt; ++t) for (int b = 0; b < nb; ++b) for (int c = 0; c < 2; ++c)
        qtrans(a[t]->q + (size_t) b * 128 + c * 64, qT[t][b][c]);
#endif
    for (int r = r0; r < r1; ++r) {
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        const uint8_t* qr = scales.q + (size_t) r * scales.qbytes;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* code = w + (size_t) r * row_bytes + (size_t) b * 4 * (size_t) k;
            const size_t bytes = (size_t) 4 * (size_t) k;
#if BIH3_MODE == 2
            uint8_t padded[72];
            padded[0] = code[bytes - 1];
            std::memcpy(padded + 1, code, bytes);
            padded[bytes + 1] = code[0]; padded[bytes + 2] = code[1];
#else
            uint8_t padded[58];
            std::memcpy(padded, code, bytes);
            padded[bytes] = code[0]; padded[bytes + 1] = code[1];
#endif
            const float scale = tq_half_to_float(tq_ld16(scales.colscale + 2 * (size_t) b)) *
                                scales.p2[(qr[b / 4] >> (2 * (b % 4))) & 3];
#if BIH3_MODE == 2
            __m256i sums[NT][4];
            for (int h = 0; h < 2; ++h) {
                __m512i w0, w1;
                decode16(padded, k, h * 16, w0, w1);
                const __m256i w00 = _mm512_castsi512_si256(w0), w01 = _mm512_extracti64x4_epi64(w0, 1);
                const __m256i w10 = _mm512_castsi512_si256(w1), w11 = _mm512_extracti64x4_epi64(w1, 1);
                for (int t = 0; t < nt; ++t) {
                    const int16_t* q = a[t]->q + (size_t) b * 128 + (size_t) h * 64;
                    sums[t][2 * h] = dot32(w00, w01, q);
                    sums[t][2 * h + 1] = dot32(w10, w11, q + 32);
                }
            }
#else
            __m256i sums[NT][2];
            for (int c = 0; c < 2; ++c) {
                const __m256i s = states16(padded, k, c * 16);
                __m256i w0, w1, w2, w3;
                weights16(s, w0, w1, w2, w3);
                for (int t = 0; t < nt; ++t) {
                    const __m256i* qt = qT[t][b][c];
#if BIH3_MODE == 1
                    // Keep the accumulation fused and VEX-encoded (the target has no AVX-512VL).
                    __asm__("vpmaddwd %5, %1, %0\n\t"
                            "%{vex%} vpdpwssd %6, %2, %0\n\t"
                            "%{vex%} vpdpwssd %7, %3, %0\n\t"
                            "%{vex%} vpdpwssd %8, %4, %0"
                            : "=&x"(sums[t][c])
                            : "x"(w0), "x"(w1), "x"(w2), "x"(w3),
                              "m"(qt[0]), "m"(qt[1]), "m"(qt[2]), "m"(qt[3]));
#else
                    sums[t][c] = _mm256_add_epi32(
                        _mm256_add_epi32(_mm256_madd_epi16(w0, qt[0]), _mm256_madd_epi16(w1, qt[1])),
                        _mm256_add_epi32(_mm256_madd_epi16(w2, qt[2]), _mm256_madd_epi16(w3, qt[3])));
#endif
                }
            }
#endif
            for (int t = 0; t < nt; ++t) {
#if BIH3_MODE == 2
                const __m256i pairs = _mm256_hadd_epi32(_mm256_hadd_epi32(sums[t][0], sums[t][1]),
                                                       _mm256_hadd_epi32(sums[t][2], sums[t][3]));
                const __m128i dot = _mm_slli_epi32(_mm_add_epi32(_mm256_castsi256_si128(pairs),
                                                                _mm256_extracti128_si256(pairs, 1)), 6);
#else
                // Q8_0 sub-block k in lane k: the first hadd pairs the dwords of each 16-state half, the second
                // sums the pairs (lane 0: sub-blocks 0,2; lane 1: 1,3), and the interleave puts them in order.
                const __m256i pairs = _mm256_hadd_epi32(sums[t][0], sums[t][1]);
                const __m256i quad = _mm256_hadd_epi32(pairs, pairs);
                const __m256i ordered = _mm256_unpacklo_epi32(quad, _mm256_permute2x128_si256(quad, quad, 0x01));
                const __m128i dot = _mm_slli_epi32(_mm256_castsi256_si128(ordered), 6);
#endif
                const __m128 sc = _mm_mul_ps(_mm_set1_ps(scale), _mm_loadu_ps(a[t]->d + (size_t) b * 4));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(dot), sc, acc[t]);
            }
        }
        const float rowbase = tq_half_to_float(tq_ld16(scales.rowbase + 2 * (size_t) r));
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.f / 8192.f) * rowbase;
        }
    }
}

BIH3_FN void rows(int type, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                  float* const* out, int r0, int r1, const TqR1q2Scales& scales) {
    switch (nt) {
    case 1: rows_t<1>(type, w, row_bytes, n, a, out, r0, r1, scales); break;
    case 2: rows_t<2>(type, w, row_bytes, n, a, out, r0, r1, scales); break;
    case 3: rows_t<3>(type, w, row_bytes, n, a, out, r0, r1, scales); break;
    case 4: rows_t<4>(type, w, row_bytes, n, a, out, r0, r1, scales); break;
    case 5: rows_t<5>(type, w, row_bytes, n, a, out, r0, r1, scales); break;
    case 6: rows_t<6>(type, w, row_bytes, n, a, out, r0, r1, scales); break;
    case 7: rows_t<7>(type, w, row_bytes, n, a, out, r0, r1, scales); break;
    case 8: rows_t<8>(type, w, row_bytes, n, a, out, r0, r1, scales); break;
    default: std::abort();
    }
}
