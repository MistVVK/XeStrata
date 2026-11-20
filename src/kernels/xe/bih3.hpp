// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 hiruhiru2005
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/bih3.hpp - the byte planes of four bih3 values, shared by the vector and matrix products.
#pragma once

#include <cstdint>

namespace strata::kernels::xe {

struct Bih3Planes { uint32_t v[3]; };

// A 15-bit state's four values as three byte planes.
inline Bih3Planes bih3_state_planes(uint32_t state) {
    state &= 0x7fffu;
    const uint32_t a = state * 0x9e3779b1u, b = state * 0x85ebca6bu;
    const uint32_t c = state * 0xc2b2ae35u, d = state * 0x27d4eb2fu;
    return {{(((a >> 8) & 0x00ffffffu) | ((b & 0x0000ff00u) << 16)) ^ 0x80808080u,
             (((b >> 16) & 0x0000ffffu) | ((c & 0x00ffff00u) << 8)) ^ 0x80808080u,
             (((c >> 24) & 0xffu) | (d & 0xffffff00u)) ^ 0x80808080u}};
}

// One plane of bih3_state_planes, with only the two products it reads.
inline uint32_t bih3_state_plane(uint32_t state, int p) {
    state &= 0x7fffu;
    if (p == 0) {
        const uint32_t a = state * 0x9e3779b1u, b = state * 0x85ebca6bu;
        return (((a >> 8) & 0x00ffffffu) | ((b & 0x0000ff00u) << 16)) ^ 0x80808080u;
    }
    if (p == 1) {
        const uint32_t b = state * 0x85ebca6bu, c = state * 0xc2b2ae35u;
        return (((b >> 16) & 0x0000ffffu) | ((c & 0x00ffff00u) << 8)) ^ 0x80808080u;
    }
    const uint32_t c = state * 0xc2b2ae35u, d = state * 0x27d4eb2fu;
    return (((c >> 24) & 0xffu) | (d & 0xffffff00u)) ^ 0x80808080u;
}

// Step `step`'s state of a block's codes (4K bytes).  Each byte of its planes is signed: adding the three bytes gives
// the value's integer numerator (the weight is that / 128).  K=8 stores each state in this order rather than as the
// circular K-bit stream used by the other widths.
inline uint32_t bih3_state(const uint8_t* code, int k, int step) {
    if (k == 8) return ((uint32_t) code[(step + 31) % 32] << 8) | code[step];
    const int bit = (31 - step) * k, bytes = 4 * k, i = bit / 8;
    const uint32_t window = (uint32_t) code[i] | ((uint32_t) code[(i + 1) % bytes] << 8) |
                            ((uint32_t) code[(i + 2) % bytes] << 16);
    return window >> (bit % 8);
}

// The same state with the codes read as 32-bit words (the block is 4K bytes, 4-byte aligned in every blob): two word
// loads a state instead of three byte loads.  K=8's byte order goes through the byte form.
inline uint32_t bih3_state_w(const uint8_t* code, int k, int step) {
    if (k == 8) return bih3_state(code, k, step);
    const auto* words = reinterpret_cast<const uint32_t*>(code);
    const int bit = (31 - step) * k, w = bit >> 5, sh = bit & 31;
    const uint64_t pair = (uint64_t) words[w] | ((uint64_t) words[w + 1 == k ? 0 : w + 1] << 32);
    return (uint32_t) (pair >> sh);
}

inline Bih3Planes bih3_planes(const uint8_t* code, int k, int step) { return bih3_state_planes(bih3_state(code, k, step)); }
inline Bih3Planes bih3_planes_w(const uint8_t* code, int k, int step) {
    return bih3_state_planes(bih3_state_w(code, k, step));
}

}  // namespace strata::kernels::xe
