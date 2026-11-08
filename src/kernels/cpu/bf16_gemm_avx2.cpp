// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/bf16_gemm_avx2.cpp - the AVX2 copy of the BF16 x FP32 matrix product
// (include/strata/kernels/cpu/bf16_gemm.hpp).  Compiled with -mavx2 -mfma and reached only behind
// bf16_gemm_isa_ok(Bf16GemmIsa::avx2), so nothing here can fault on an older CPU.
//
// THE SHAPE OF THE INNER LOOP: one panel (eight outputs of the weight matrix, contiguous in the packed form) against
// up to eight token rows.  The eight lanes are eight DIFFERENT outputs, so a lane is never reduced with another and
// the accumulation of one output is one chain of vfmadd in ascending k - exactly the chain the scalar reference
// runs, which is why the two agree bit for bit (the header).  A weight arrives as eight BF16 (16 bytes, one load)
// and becomes FP32 by widening to eight 32-bit lanes and shifting them up: BF16 is the top half of FP32, so the
// shift is exact and costs no rounding.
//
// WHY EIGHT OUTPUTS AND NOT SIXTEEN OR THIRTY-TWO, which is the choice the plan left open ("重みを 32 列のパネルに
// 詰め直し"): a panel of eight costs one 16-byte load and one shift per k and feeds eight accumulators, so a group of
// four tokens needs eight FMAs against four broadcasts and two non-FMA instructions - about the balance the two FMA
// ports and the three load ports of a Golden Cove or Zen 3 P-core are happy with.  Wider panels need more
// accumulators than the sixteen vector registers leave room for once the token group is over two, and the packed
// panel would have to skip the k values a store would not write.  Measured numbers are in the record's `s2/`.
#include "strata/kernels/cpu/bf16_gemm.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace strata::kernels::cpu::bf16_gemm_detail {
namespace {

/// Eight BF16 weights, as bits, to the same eight values as FP32.  Exact: the shift fills the low mantissa bits.
inline __m256 upconvert(__m128i h) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

/// One panel against `ROWS` token rows (1..8): `x` is the first of them, `y` the outputs' row for lane j at
/// `base + j`.  `cols` is how many of the panel's eight lanes are real (the last panel of a matrix whose n_out is
/// not a multiple of eight); the padding lanes compute zero and are not stored.
template <int ROWS>
void tile(const uint16_t* wp, int n_in, const float* x, size_t ldx, float* y, size_t ldy, int base, int cols,
          const float* bias) {
    __m256 acc[ROWS];
    for (int r = 0; r < ROWS; ++r) acc[r] = _mm256_setzero_ps();
    for (int k = 0; k < n_in; ++k) {
        const __m256 w = upconvert(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wp + (size_t) k * kBf16Panel)));
        for (int r = 0; r < ROWS; ++r)
            acc[r] = _mm256_fmadd_ps(w, _mm256_broadcast_ss(x + (size_t) r * ldx + k), acc[r]);
    }
    // The bias is added after the reduction, never as the chain's first value, so every path ends the same way the
    // reference does.  A partial panel's bias lanes past n_out are not read: the array has n_out of them.
    __m256 bv = _mm256_setzero_ps();
    float bpad[kBf16Panel];
    if (bias != nullptr) {
        if (cols == kBf16Panel) {
            bv = _mm256_loadu_ps(bias + base);
        } else {
            for (int j = 0; j < kBf16Panel; ++j) bpad[j] = j < cols ? bias[base + j] : 0.0f;
            bv = _mm256_loadu_ps(bpad);
        }
    }
    for (int r = 0; r < ROWS; ++r) {
        __m256 v = acc[r];
        if (bias != nullptr) v = _mm256_add_ps(v, bv);
        float* yr = y + (size_t) r * ldy + base;
        if (cols == kBf16Panel) {
            _mm256_storeu_ps(yr, v);
        } else {
            alignas(32) float out[kBf16Panel];
            _mm256_store_ps(out, v);
            std::memcpy(yr, out, (size_t) cols * sizeof(float));
        }
    }
}

/// The token groups of one block: eight at a time, then the last 1..7 (the block's size is a multiple of eight, so
/// only its last group is short).  At ROWS below four the loop is one vfmadd per four cycles - the chain's latency,
/// not a port: an embedding is never one token (the shortest reference case is eight), and a product that is one
/// token streams its weights at a fraction of the bandwidth.  Fixing that is a panel unroll, not a new order.
inline void rows_of_panel(const uint16_t* wp, int n_in, const float* x, size_t ldx, float* y, size_t ldy, int base,
                          int cols, int rows, const float* bias) {
    int r = 0;
    for (; r + 8 <= rows; r += 8)
        tile<8>(wp, n_in, x + (size_t) r * ldx, ldx, y + (size_t) r * ldy, ldy, base, cols, bias);
    const int left = rows - r;
    if (left <= 0) return;
    const float* xr = x + (size_t) r * ldx;
    float* yr = y + (size_t) r * ldy;
    switch (left) {
        case 7: tile<7>(wp, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        case 6: tile<6>(wp, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        case 5: tile<5>(wp, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        case 4: tile<4>(wp, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        case 3: tile<3>(wp, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        case 2: tile<2>(wp, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
        default: tile<1>(wp, n_in, xr, ldx, yr, ldy, base, cols, bias); break;
    }
}

}  // namespace

void bf16_gemm_rows_avx2(const uint16_t* panel, int n_in, int n_out, const float* x, size_t ldx, float* y, size_t ldy,
                         int n_tok, int p0, int p1, const float* bias) {
    const int tb = bf16_gemm_token_block(n_in);
    for (int t0 = 0; t0 < n_tok; t0 += tb) {
        const int rows = std::min(tb, n_tok - t0);
        const float* xb = x + (size_t) t0 * ldx;
        float* yb = y + (size_t) t0 * ldy;
        for (int p = p0; p < p1; ++p) {
            const int base = p * kBf16Panel;
            const int cols = std::min(kBf16Panel, n_out - base);
            const uint16_t* wp = panel + (size_t) p * (size_t) n_in * (size_t) kBf16Panel;
            rows_of_panel(wp, n_in, xb, ldx, yb, ldy, base, cols, rows, bias);
        }
    }
}

}  // namespace strata::kernels::cpu::bf16_gemm_detail
