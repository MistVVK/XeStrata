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

// The state's four values' integer numerators (the planes' byte sums) read from its four products: the planes' bytes
// are bytes 1-3 of a b c d (value 0 = a1 + b2 + c3, 1 = a2 + b3 + d1, 2 = a3 + c1 + d2, 3 = b1 + c2 + d3), each less
// 128, so the sums need no planes put together (fifteen operations a state on the B70).
inline void bih3_state_values(uint32_t state, int (&v)[4]) {
    state &= 0x7fffu;
    const uint32_t a = state * 0x9e3779b1u, b = state * 0x85ebca6bu;
    const uint32_t c = state * 0xc2b2ae35u, d = state * 0x27d4eb2fu;
    v[0] = (int) (((a >> 8) & 0xffu) + ((b >> 16) & 0xffu) + (c >> 24)) - 384;
    v[1] = (int) (((a >> 16) & 0xffu) + (b >> 24) + ((d >> 8) & 0xffu)) - 384;
    v[2] = (int) ((a >> 24) + ((c >> 8) & 0xffu) + ((d >> 16) & 0xffu)) - 384;
    v[3] = (int) (((b >> 8) & 0xffu) + ((c >> 16) & 0xffu) + (d >> 24)) - 384;
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

// The 32 bits of hi:lo from bit sh (0 <= sh < 32) in 32-bit operations: a 64-bit shift is several instructions a
// lane on the B70 (its integer pipe splits 64-bit operands, the decode rows' states spent half their moves there).
inline uint32_t bih3_funnel(uint32_t lo, uint32_t hi, int sh) { return (lo >> sh) | ((hi << 1) << (31 - sh)); }

// The same state with the codes read as 32-bit words (the block is 4K bytes, 4-byte aligned in every blob): two word
// loads a state instead of three byte loads.  K=8's byte order goes through the byte form.
inline uint32_t bih3_state_w(const uint8_t* code, int k, int step) {
    if (k == 8) return bih3_state(code, k, step);
    const auto* words = reinterpret_cast<const uint32_t*>(code);
    const int bit = (31 - step) * k, w = bit >> 5;
    return bih3_funnel(words[w], words[w + 1 == k ? 0 : w + 1], bit & 31);
}

inline Bih3Planes bih3_planes(const uint8_t* code, int k, int step) { return bih3_state_planes(bih3_state(code, k, step)); }
inline Bih3Planes bih3_planes_w(const uint8_t* code, int k, int step) {
    return bih3_state_planes(bih3_state_w(code, k, step));
}

// The eight states of a block's 32-value part q (steps 8q .. 8q + 7) as planes, their K-bit windows read in the
// stream's order with two words held: about K / 4 + 2 word loads where bih3_planes_w takes two a state.
inline void bih3_block_planes(const uint8_t* code, int k, int q, Bih3Planes (&out)[8]) {
    if (k == 8) {
        for (int j = 0; j < 8; ++j) out[j] = bih3_planes(code, k, q * 8 + j);
        return;
    }
    const auto* words = reinterpret_cast<const uint32_t*>(code);
    int off = (24 - 8 * q) * k, w = off >> 5;   // step 8q + 7's window, the lowest of the part's
    uint32_t cur = words[w], nxt = words[w + 1 == k ? 0 : w + 1];
    for (int m = 0; m < 8; ++m, off += k) {
        if ((off >> 5) != w) {
            ++w;
            cur = nxt;
            nxt = words[w + 1 == k ? 0 : w + 1];
        }
        out[7 - m] = bih3_state_planes(bih3_funnel(cur, nxt, off & 31));
    }
}

}  // namespace strata::kernels::xe
