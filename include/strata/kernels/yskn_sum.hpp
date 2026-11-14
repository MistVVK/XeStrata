// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/kernels/yskn_sum.hpp - how the .yskn products add their groups' sums: the one place, for the CPU's
// reference (artifact/yskn.cpp) and the GPU kernels (kernels/xe/yskn.cpp) alike.
//
// After YASAKANI's lnord.cu (lno_xsm_rm, lno_xsm_A, lno_xsm_us) and a8_prologue.cuh (xsum_y): each 512-group g of a row
// gives an exact int32 dot of int8 weights and int8 activations; its activation scale alpha_g is rounded once to a
// fixed-point integer A_g = round(alpha_g * 2^24 / max_g alpha_g), the products dot_g * A_g are added in int64, and the
// sum is scaled back by max alpha * 2^-24 (and the row's tau).  Integers add the same in any order and any split, so a
// row's result does not depend on how many rows a window has or how the columns are divided among the work-items.
// Plain C++ with no library call: the host compiler and every SYCL device compile it the same.
#pragma once

#include <cstdint>

namespace strata::kernels::yskn {

/// 2^24 / max alpha (0 for a row of zeros), the factor that takes each group's alpha to its fixed-point integer
inline float fixed_factor(float alpha_max) { return alpha_max > 0.f ? 16777216.f / alpha_max : 0.f; }

/// round(alpha * factor), to nearest with ties to even (CUDA's __float2int_rn), for alpha * factor in [0, 2^24]
inline int64_t fixed_scale(float alpha, float factor) {
    const float v = alpha * factor;
    int64_t i = (int64_t) v;   // v >= 0: the truncation is the floor, and exact below 2^24
    const float frac = v - (float) i;
    if (frac > 0.5f || (frac == 0.5f && (i & 1) != 0)) ++i;
    return i;
}

/// one group's contribution
inline int64_t group_term(int32_t dot, int64_t fixed) { return (int64_t) dot * fixed; }

/// the row's value in the rotated basis: the int64 sum times max alpha * 2^-24 and the row's tau
inline float finish(int64_t acc, float alpha_max, float tau) {
    return ((float) acc * (alpha_max * (1.f / 16777216.f))) * tau;
}

}  // namespace strata::kernels::yskn
