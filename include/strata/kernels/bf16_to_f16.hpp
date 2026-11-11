// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// BF16 tensors to FP16, with each caller's treatment of values outside FP16's range.
#pragma once

#include "strata/kernels/f16_gpu.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::kernels {

enum class F16Range { native, finite_saturate, clamp };

template<F16Range Range>
inline void bf16_to_f16(sycl::queue& q, const uint16_t* x, uint16_t* y, int64_t n) {
    if (n <= 0) return;
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        float f = sycl::bit_cast<float>((uint32_t) x[i] << 16);
        if constexpr (Range == F16Range::finite_saturate) {
            if (!sycl::isinf(f) && !sycl::isnan(f)) f = sycl::fmin(sycl::fmax(f, -65504.f), 65504.f);
        } else if constexpr (Range == F16Range::clamp) {
            f = sycl::clamp(f, -65504.f, 65504.f);
        }
        y[i] = f16_gpu::f16_from_f32(f);
    });
}

}  // namespace strata::kernels
