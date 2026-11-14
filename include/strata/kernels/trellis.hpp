// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/kernels/trellis.hpp - the state of a trellis code's step and its table lookup, the one place for the
// readers that decode a bit stream through a state table: the .yskn matrices (artifact/yskn.cpp's unit_state, the GPU
// kernels' state_of) and the KV trellis code (Y5's kv_tcq_parts.hpp).  The two bit orders are the parameters: the
// state's width and whether the stream reads least- or most-significant bit first.
//
// LSB first (the .yskn matrices, YASAKANI's lnord.cu): a step j of K-bit steps holds the L bits of the stream that end
// at bit K (j + 1), the words little-endian, the bits below the stream's start zero (start "zero").  The state picks an
// entry of the step's table (the int8 pair lut8, the FP16 pair lut16).
//
// MSB first (the KV trellis, YASAKANI's kvx_codec.cuh): a step's state is the bits that end at `end`, not below the
// chain's first bit `first`, read out of a 64-bit big-endian window (window_msb); the width varies with the step's
// place in its chain (state_msb).  The state picks a codebook value (kv_tcq_parts.hpp's code_val).
//
// Plain C++ with no library call: the host compiler and every SYCL device compile it the same.
#pragma once

#include <cstdint>

namespace strata::kernels::trellis {

/// The stream's bit order: the .yskn matrices read least-significant first, the KV trellis most-significant first
enum class Order { Lsb, Msb };

/// The L-bit state of step j (0-based) of K-bit steps, zero below the stream's start (Order::Lsb).  `w` holds the
/// unit's words (4 K of them for L 12); the window stays inside them
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

/// 64 bits of the stream from bit `at` on, most significant first (Order::Msb): the words big-endian, the read staying
/// inside the row (the code's last word is followed by the scale's)
inline uint64_t window_msb(const uint32_t* w, int at) {
    const int wi = at >> 5, o = at & 31;
    const uint64_t z = ((uint64_t) w[wi] << 32) | w[wi + 1];
    return o == 0 ? z : (z << o) | (w[wi + 2] >> (32 - o));
}

/// The state of a step whose bits end at `end`, not below the chain's first bit `first` (Order::Msb): the width is
/// end - lo, at most 8 (a 64-bit window holds it and the bits before)
inline uint32_t state_msb(const uint32_t* w, int end, int first) {
    const int lo = end - 8 > first ? end - 8 : first, n = end - lo;
    const int wi = lo >> 5, o = lo & 31;
    const uint64_t z = ((uint64_t) w[wi] << 32) | w[wi + 1];
    return (uint32_t) (z >> (64 - o - n)) & ((1u << n) - 1u);
}

/// The int8 pair a state picks from the table (the low byte the first value): the .yskn weights' lut8
inline uint32_t pair8(const uint16_t* lut8, uint32_t st) { return lut8[st]; }
/// The FP16 pair a state picks (the .yskn prompt's dequantization, lut16): the two halves as one word
inline uint32_t pair16(const uint16_t* lut16, uint32_t st) {
    return (uint32_t) lut16[2 * st] | ((uint32_t) lut16[2 * st + 1] << 16);
}

}  // namespace strata::kernels::trellis