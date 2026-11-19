// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 hiruhiru2005
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/bih3.hpp - the byte planes of four bih3 values, shared by the vector and matrix products.
#pragma once

#include <cstdint>

namespace strata::kernels::xe {

struct Bih3Planes { uint32_t v[3]; };

// Each byte is signed: adding the three bytes gives the value's integer numerator (the weight is that / 128).
// K=8 stores each state in this order rather than as the circular K-bit stream used by the other widths.
inline Bih3Planes bih3_planes(const uint8_t* code, int k, int step) {
    const int bit = (31 - step) * k, bytes = 4 * k, i = bit / 8;
    uint32_t state;
    if (k == 8) state = ((uint32_t) code[(step + 31) % 32] << 8) | code[step];
    else {
        const uint32_t window = (uint32_t) code[i] | ((uint32_t) code[(i + 1) % bytes] << 8) |
                                ((uint32_t) code[(i + 2) % bytes] << 16);
        state = window >> (bit % 8);
    }
    state &= 0x7fffu;
    const uint32_t a = state * 0x9e3779b1u, b = state * 0x85ebca6bu;
    const uint32_t c = state * 0xc2b2ae35u, d = state * 0x27d4eb2fu;
    return {{(((a >> 8) & 0x00ffffffu) | ((b & 0x0000ff00u) << 16)) ^ 0x80808080u,
             (((b >> 16) & 0x0000ffffu) | ((c & 0x00ffff00u) << 8)) ^ 0x80808080u,
             (((c >> 24) & 0xffu) | (d & 0xffffff00u)) ^ 0x80808080u}};
}

}  // namespace strata::kernels::xe
