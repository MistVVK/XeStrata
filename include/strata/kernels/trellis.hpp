// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/kernels/trellis.hpp - the state of a trellis code's step and its table lookup, the one place for the
// readers that decode a bit stream through a state table: the .yskn matrices (artifact/yskn.cpp's unit_state, the GPU
// kernels' state_of) and, in the same shape, the KV trellis code (Y5's kv_tcq_parts.hpp).
//
// A step j of K-bit steps holds the L bits of the stream that end at bit K (j + 1): the state picks an entry of the
// step's table.  The bits below the stream's start are zero (start "zero").  The words are little-endian, the stream
// least-significant bit first (YASAKANI's lnord.cu); the window reads w[word] and, where it crosses, w[word + 1].
//
// The KV trellis (Y5) reads its window most-significant bit first out of a 64-bit read (kv_tcq_parts.hpp's bits64):
// the same role, another bit order, kept beside this until the two bundles meet.  Plain C++ with no library call: the
// host compiler and every SYCL device compile it the same.
#pragma once

#include <cstdint>

namespace strata::kernels::trellis {

/// The L-bit state of step j (0-based) of K-bit steps, zero below the stream's start.  `w` holds the unit's words
/// (4 K of them for L 12); the window stays inside them
inline uint32_t state(const uint32_t* w, int K, int j, int L) {
    const int e = K * (j + 1);
    if (e < L) return (w[0] & ((1u << e) - 1u)) << (L - e);
    const int start = e - L, word = start >> 5, sh = start & 31;
    uint32_t v = w[word] >> sh;
    if (sh + L > 32) v |= w[word + 1] << (32 - sh);
    return v & ((1u << L) - 1u);
}

/// The same with the state's width and the step's K compile-time (the GPU kernels unroll the steps)
template<int K, int L = 12> inline uint32_t state_of(const uint32_t* w, int j) {
    const int e = K * (j + 1);
    if (e < L) return (w[0] & ((1u << e) - 1u)) << (L - e);
    const int start = e - L, word = start >> 5, sh = start & 31;
    uint32_t v = w[word] >> sh;
    if (sh + L > 32) v |= w[word + 1] << (32 - sh);
    return v & ((1u << L) - 1u);
}

/// The int8 pair a state picks from the table (the low byte the first value): the .yskn weights' lut8
inline uint32_t pair8(const uint16_t* lut8, uint32_t st) { return lut8[st]; }
/// The FP16 pair a state picks (the .yskn prompt's dequantization, lut16): the two halves as one word
inline uint32_t pair16(const uint16_t* lut16, uint32_t st) {
    return (uint32_t) lut16[2 * st] | ((uint32_t) lut16[2 * st + 1] << 16);
}

}  // namespace strata::kernels::trellis