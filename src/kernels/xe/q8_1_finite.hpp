// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/q8_1_finite.hpp - #606: the q8_1 activation block's scale and sum, kept finite.
//
// A q8_1 block stores its scale d = amax / 127 and the sum of its 32 raw activations as fp16.  A sum past 65504 (one
// massive-activation dimension among 32 values is enough) rounds to inf, and d does past amax = 8.3M; the dot products
// then read inf * 0 = NaN, and a NaN in the residual ends as one token repeated (ggml-org/llama.cpp#23606 is the same
// defect).  Clamped to the largest finite half, such a block gives an approximate product instead of NaN, and the
// values are held to the int8 range they then exceed.  Every block that was finite before is stored bit for bit as
// before: values in (65504, 65520) round to 65504 anyway, and a finite block's |x / d| is at most 127.  A NaN stays
// NaN (the comparisons are false for it).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels::xe {

constexpr float kHalfMax = 65504.0f;

inline float q8_1_finite(const float v) { return sycl::fabs(v) > kHalfMax ? sycl::copysign(kHalfMax, v) : v; }

/// The block's quantized value of `xi` for the (finite) scale `d`; `amax` = 0 is an all-zero block.
inline int8_t q8_1_quant(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = sycl::round(xi / d);
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

}  // namespace strata::kernels::xe
