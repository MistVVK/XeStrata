// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/kernels/trellis.hpp - the state of a trellis code's step, the one place for the readers that decode a
// bit stream through a state table: the KV trellis code (Y5's kv_tcq_parts.hpp), after YASAKANI's kvx_codec.cuh.
//
// MSB first: a step's state is the bits that end at `end`, not below the chain's first bit `first`, read out of a
// 64-bit big-endian window (window_msb); the width varies with the step's place in its chain (state_msb).  The state
// picks a codebook value (kv_tcq_parts.hpp's code_val).
//
// Plain C++ with no library call: the host compiler and every SYCL device compile it the same.
#pragma once

#include <cstdint>

namespace strata::kernels::trellis {

/// 64 bits of the stream from bit `at` on, most significant first: the words big-endian, the read staying inside the
/// row (the code's last word is followed by the scale's)
inline uint64_t window_msb(const uint32_t* w, int at) {
    const int wi = at >> 5, o = at & 31;
    const uint64_t z = ((uint64_t) w[wi] << 32) | w[wi + 1];
    return o == 0 ? z : (z << o) | (w[wi + 2] >> (32 - o));
}

/// The state of a step whose bits end at `end`, not below the chain's first bit `first`: the width is end - lo, at
/// most 8 (a 64-bit window holds it and the bits before)
inline uint32_t state_msb(const uint32_t* w, int end, int first) {
    const int lo = end - 8 > first ? end - 8 : first, n = end - lo;
    const int wi = lo >> 5, o = lo & 31;
    const uint64_t z = ((uint64_t) w[wi] << 32) | w[wi + 1];
    return (uint32_t) (z >> (64 - o - n)) & ((1u << n) - 1u);
}

}  // namespace strata::kernels::trellis
