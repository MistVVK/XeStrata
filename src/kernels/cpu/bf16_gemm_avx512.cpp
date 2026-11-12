// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/bf16_gemm_avx512.cpp - the AVX-512 copy of the BF16 x FP32 matrix product
// (include/strata/kernels/cpu/bf16_gemm.hpp).  Compiled with -mavx512f (and the AVX2 flags), and reached only behind
// bf16_gemm_isa_ok(Bf16GemmIsa::avx512), so nothing here can fault on an older CPU.
//
// The packed form is the AVX2 copy's (panels of eight outputs); this copy takes two neighbouring panels in one zmm,
// sixteen outputs, so one vfmadd does the work of two there.  A lane is still one output and its accumulation still
// one chain of vfmadd in ascending k with the bias added after it: the scalar reference's chain, bit for bit.
#include "strata/kernels/cpu/bf16_gemm.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cstddef>

namespace strata::kernels::cpu::bf16_gemm_detail {
namespace {

/// Eight BF16 weights of panel `a` and eight of `b` (null: none, zeros) to sixteen FP32: exact, as the AVX2 copy's.
inline __m512 upconvert2(const uint16_t* a, const uint16_t* b) {
    const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    const __m128i hi = b != nullptr ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(b)) : _mm_setzero_si128();
    const __m256i h = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}

/// Panels `wa` and `wb` (null: `wa` is the last) against `ROWS` token rows (1..8); `cols` real outputs of the sixteen
/// from `base`, the rest neither read (the bias) nor stored.
template <int ROWS>
void tile(const uint16_t* wa, const uint16_t* wb, int n_in, const float* x, size_t ldx, float* y, size_t ldy, int base,
          int cols, const float* bias) {
    __m512 acc[ROWS];
    for (int r = 0; r < ROWS; ++r) acc[r] = _mm512_setzero_ps();
    for (int k = 0; k < n_in; ++k) {
        const __m512 w = upconvert2(wa + (size_t) k * kBf16Panel, wb != nullptr ? wb + (size_t) k * kBf16Panel : nullptr);
        for (int r = 0; r < ROWS; ++r)
            acc[r] = _mm512_fmadd_ps(w, _mm512_set1_ps(x[(size_t) r * ldx + k]), acc[r]);
    }
    const __mmask16 m = (__mmask16) ((1u << cols) - 1u);
    const __m512 bv = bias != nullptr ? _mm512_maskz_loadu_ps(m, bias + base) : _mm512_setzero_ps();
    for (int r = 0; r < ROWS; ++r) {
        __m512 v = acc[r];
        if (bias != nullptr) v = _mm512_add_ps(v, bv);
        _mm512_mask_storeu_ps(y + (size_t) r * ldy + base, m, v);
    }
}

inline void rows_of_pair(const uint16_t* wa, const uint16_t* wb, int n_in, const float* x, size_t ldx, float* y,
                         size_t ldy, int base, int cols, int rows, const float* bias) {
    int r = 0;
    for (; r + 8 <= rows; r += 8)
        tile<8>(wa, wb, n_in, x + (size_t) r * ldx, ldx, y + (size_t) r * ldy, ldy, base, cols, bias);
    const int left = rows - r;
    if (left <= 0) return;
    const float* xr = x + (size_t) r * ldx;
    float* yr = y + (size_t) r * ldy;
    switch (left) {
        case 7: tile<7>(wa, wb, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        case 6: tile<6>(wa, wb, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        case 5: tile<5>(wa, wb, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        case 4: tile<4>(wa, wb, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        case 3: tile<3>(wa, wb, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        case 2: tile<2>(wa, wb, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        default: tile<1>(wa, wb, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
    }
}

}  // namespace

void bf16_gemm_rows_avx512(const uint16_t* panel, int n_in, int n_out, const float* x, size_t ldx, float* y,
                           size_t ldy, int n_tok, int p0, int p1, const float* bias) {
    const int tb = bf16_gemm_token_block(n_in);
    const size_t step = (size_t) n_in * (size_t) kBf16Panel;
    for (int t0 = 0; t0 < n_tok; t0 += tb) {
        const int rows = std::min(tb, n_tok - t0);
        const float* xb = x + (size_t) t0 * ldx;
        float* yb = y + (size_t) t0 * ldy;
        for (int p = p0; p < p1; p += 2) {
            const int base = p * kBf16Panel;
            const bool pair = p + 1 < p1;
            const int cols = std::min(pair ? 2 * kBf16Panel : kBf16Panel, n_out - base);
            const uint16_t* wa = panel + (size_t) p * step;
            rows_of_pair(wa, pair ? wa + step : nullptr, n_in, xb, ldx, yb, ldy, base, cols, rows, bias);
        }
    }
}

}  // namespace strata::kernels::cpu::bf16_gemm_detail
