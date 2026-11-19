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
// The AVX-512 copy takes the same 16-bit-word form as the AVX2 / AVX-VNNI copies, 32 states (one whole
// 128-weight block) at a time in ZMM: three VPMULLW/VPMULHUW per constant, the byte planes as 16-bit shifts
// and ANDs, the weights weight-major, the activations transposed once per call and block (qtrans512, three
// VPERMT2W per weight kind), and two in-lane permute+add folds plus one VPERMD leave the four Q8_0
// sub-block sums in lane order (AVX-512 has no VPHADDD).
// No VBMI: the condition is AVX-512 F/BW/VL/VNNI (cpu_avx512_latent_ok), so AVX-512 CPUs without VBMI
// run it too.  v17 (the VBMI byte gather into activation order) measured slower than the new ymm copies
// on every model (llvm-mca -mcpu=sapphirerapids, 2026-11-19: 62.87/68.45/82.31/114.28 against
// 51.61/58.84/75.32/108.28 at NT 1/2/4/8).

// qT[g] word j = q[4j + g], j = 0..31: two VPERMT2W gather the stride-4 words of each 64-word half, one joins.
constexpr uint16_t bih3_qidx_lo[4][32] = {
    {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
constexpr uint16_t bih3_qidx_cat[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};

BIH3_FN inline void qtrans512(const int16_t* q, __m512i* qT) {
    const __m512i Q0 = _mm512_loadu_si512(q);
    const __m512i Q1 = _mm512_loadu_si512(q + 32);
    const __m512i Q2 = _mm512_loadu_si512(q + 64);
    const __m512i Q3 = _mm512_loadu_si512(q + 96);
    const __m512i cat = _mm512_loadu_si512(bih3_qidx_cat);
    for (int g = 0; g < 4; ++g) {
        const __m512i idx = _mm512_loadu_si512(bih3_qidx_lo[g]);
        const __m512i t01 = _mm512_permutex2var_epi16(Q0, idx, Q1);
        const __m512i t23 = _mm512_permutex2var_epi16(Q2, idx, Q3);
        qT[g] = _mm512_permutex2var_epi16(t01, cat, t23);
    }
}

// 32 states as 32 unsigned 15-bit words.  `padded` carries the block's last byte before its first.
BIH3_FN inline __m512i states32(const uint8_t* padded, int k) {
    if (k == 8) {
        const __m512i cur = _mm512_loadu_si512(padded + 1);
        const __m512i prev = _mm512_loadu_si512(padded);
        const __m256i c = _mm512_castsi512_si256(cur), p = _mm512_castsi512_si256(prev);
        const __m256i lo = _mm256_unpacklo_epi8(c, p), hi = _mm256_unpackhi_epi8(c, p);
        // the 256-bit unpacks interleave within lanes: [s0-7 | s16-23] and [s8-15 | s24-31]; the two
        // vperm2i128 put the four 8-state groups back in order
        const __m256i l01 = _mm256_permute2x128_si256(lo, hi, 0x20);
        const __m256i l23 = _mm256_permute2x128_si256(lo, hi, 0x31);
        const __m512i pairs = _mm512_inserti64x4(_mm512_castsi256_si512(l01), l23, 1);
        return _mm512_and_si512(pairs, _mm512_set1_epi16(0x7fff));
    }
    alignas(64) uint16_t st[32];
    for (int j = 0; j < 32; ++j) st[j] = (uint16_t) bih3_state_padded(padded + 1, k, j);
    return _mm512_load_si512(st);
}

// 32 states yield w0..w3: weight g of the 32 states, signed 16-bit, activation value (sum - 384).
BIH3_FN inline void weights32(__m512i s, __m512i& w0, __m512i& w1, __m512i& w2, __m512i& w3) {
    const __m512i la = _mm512_mullo_epi16(s, _mm512_set1_epi16((short) 0x79b1));
    const __m512i ha = _mm512_add_epi16(_mm512_mulhi_epu16(s, _mm512_set1_epi16((short) 0x79b1)),
                                        _mm512_mullo_epi16(s, _mm512_set1_epi16((short) 0x9e37)));
    const __m512i lb = _mm512_mullo_epi16(s, _mm512_set1_epi16((short) 0xca6b));
    const __m512i hb = _mm512_add_epi16(_mm512_mulhi_epu16(s, _mm512_set1_epi16((short) 0xca6b)),
                                        _mm512_mullo_epi16(s, _mm512_set1_epi16((short) 0x85eb)));
    const __m512i lc = _mm512_mullo_epi16(s, _mm512_set1_epi16((short) 0xae35));
    const __m512i hc = _mm512_add_epi16(_mm512_mulhi_epu16(s, _mm512_set1_epi16((short) 0xae35)),
                                        _mm512_mullo_epi16(s, _mm512_set1_epi16((short) 0xc2b2)));
    const __m512i ld = _mm512_mullo_epi16(s, _mm512_set1_epi16((short) 0xeb2f));
    const __m512i hd = _mm512_add_epi16(_mm512_mulhi_epu16(s, _mm512_set1_epi16((short) 0xeb2f)),
                                        _mm512_mullo_epi16(s, _mm512_set1_epi16((short) 0x27d4)));
    const __m512i mask = _mm512_set1_epi16(255), centre = _mm512_set1_epi16(384);
    const __m512i a1 = _mm512_srli_epi16(la, 8), a2 = _mm512_and_si512(ha, mask), a3 = _mm512_srli_epi16(ha, 8);
    const __m512i b1 = _mm512_srli_epi16(lb, 8), b2 = _mm512_and_si512(hb, mask), b3 = _mm512_srli_epi16(hb, 8);
    const __m512i c1 = _mm512_srli_epi16(lc, 8), c2 = _mm512_and_si512(hc, mask), c3 = _mm512_srli_epi16(hc, 8);
    const __m512i d1 = _mm512_srli_epi16(ld, 8), d2 = _mm512_and_si512(hd, mask), d3 = _mm512_srli_epi16(hd, 8);
    w0 = _mm512_sub_epi16(_mm512_add_epi16(_mm512_add_epi16(a1, b2), c3), centre);
    w1 = _mm512_sub_epi16(_mm512_add_epi16(_mm512_add_epi16(a2, b3), d1), centre);
    w2 = _mm512_sub_epi16(_mm512_add_epi16(_mm512_add_epi16(a3, c1), d2), centre);
    w3 = _mm512_sub_epi16(_mm512_add_epi16(_mm512_add_epi16(b1, c2), d3), centre);
}
#endif

template <int NT> BIH3_FN
void rows_t(int type, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
            float* const* out, int r0, int r1, const TqR1q2Scales& scales) {
    constexpr int nt = NT;
    const int k = type - 180, nb = n / 128;
    if (r0 >= r1) return;
#if BIH3_MODE == 2
    alignas(64) __m512i qT[NT][kTqMaxN / 128][4];
    for (int t = 0; t < nt; ++t) for (int b = 0; b < nb; ++b)
        qtrans512(a[t]->q + (size_t) b * 128, qT[t][b]);
#else
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
            const __m512i s = states32(padded, k);
            __m512i w0, w1, w2, w3;
            weights32(s, w0, w1, w2, w3);
            // lane k of the 16 dword sums is Q8_0 sub-block k (states 8k..8k+7).  AVX-512 has no vphaddd:
            // two in-lane permute+add folds sum each lane's four dwords, one vpermd takes lane k's dword 0
            // into lane k of the xmm.
            const __m512i adj = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
            const __m512i pair = _mm512_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
            const __m512i fold = _mm512_setr_epi32(0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            for (int t = 0; t < nt; ++t) {
                const __m512i* qt = qT[t][b];
                // Two fused EVEX chains (VEX AVX-VNNI is not valid on an AVX-512 VNNI chip without it, SDE -icx).
                __m512i lo, hi;
                __asm__("vpmaddwd %3, %1, %0\n\tvpdpwssd %4, %2, %0"
                        : "=&x"(lo) : "x"(w0), "x"(w1), "m"(qt[0]), "m"(qt[1]));
                __asm__("vpmaddwd %3, %1, %0\n\tvpdpwssd %4, %2, %0"
                        : "=&x"(hi) : "x"(w2), "x"(w3), "m"(qt[2]), "m"(qt[3]));
                const __m512i sums = _mm512_add_epi32(lo, hi);
                const __m512i pairs = _mm512_add_epi32(sums, _mm512_permutexvar_epi32(adj, sums));
                const __m512i quad = _mm512_add_epi32(pairs, _mm512_permutexvar_epi32(pair, pairs));
                const __m128i dot = _mm_slli_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi32(fold, quad)), 6);
                const __m128 sc = _mm_mul_ps(_mm_set1_ps(scale), _mm_loadu_ps(a[t]->d + (size_t) b * 4));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(dot), sc, acc[t]);
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
                    // Two fused chains instead of one serial madd + 3 dpwssd: at NT <= 4 the chain's latency,
                    // not the ports, sets the time on the dev machine (i7-14700: the serial form measured
                    // 0.3289/0.3463/0.4031 us/row at NT 1/2/4 against the AVX2 tree's 0.2936/0.3345/0.3904).
                    // Keep the VEX encoding (the target has no AVX-512VL).
                    __m256i lo, hi;
                    __asm__("vpmaddwd %3, %1, %0\n\t%{vex%} vpdpwssd %4, %2, %0"
                            : "=&x"(lo) : "x"(w0), "x"(w1), "m"(qt[0]), "m"(qt[1]));
                    __asm__("vpmaddwd %3, %1, %0\n\t%{vex%} vpdpwssd %4, %2, %0"
                            : "=&x"(hi) : "x"(w2), "x"(w3), "m"(qt[2]), "m"(qt[3]));
                    sums[t][c] = _mm256_add_epi32(lo, hi);
#else
                    sums[t][c] = _mm256_add_epi32(
                        _mm256_add_epi32(_mm256_madd_epi16(w0, qt[0]), _mm256_madd_epi16(w1, qt[1])),
                        _mm256_add_epi32(_mm256_madd_epi16(w2, qt[2]), _mm256_madd_epi16(w3, qt[3])));
#endif
                }
            }
            for (int t = 0; t < nt; ++t) {
                // Q8_0 sub-block k in lane k: the first hadd pairs the dwords of each 16-state half, the second
                // sums the pairs (lane 0: sub-blocks 0,2; lane 1: 1,3), and the interleave puts them in order.
                const __m256i pairs = _mm256_hadd_epi32(sums[t][0], sums[t][1]);
                const __m256i quad = _mm256_hadd_epi32(pairs, pairs);
                const __m256i ordered = _mm256_unpacklo_epi32(quad, _mm256_permute2x128_si256(quad, quad, 0x01));
                const __m128i dot = _mm_slli_epi32(_mm256_castsi256_si128(ordered), 6);
                const __m128 sc = _mm_mul_ps(_mm_set1_ps(scale), _mm_loadu_ps(a[t]->d + (size_t) b * 4));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(dot), sc, acc[t]);
            }
#endif
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
