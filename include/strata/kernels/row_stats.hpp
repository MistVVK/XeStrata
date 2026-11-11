// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Row normalization statistics, shared by the image and text kernels.
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <algorithm>

namespace strata::kernels::row_stats {

// A 16-wide device can use its own reduction (the B70 measurements in native_qsa.cpp); the 32-wide form
// remains available for devices without it and for the pinned bit-exact native path.
inline int subgroup_width(const sycl::device& d) {
    const auto sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    return std::find(sizes.begin(), sizes.end(), 16) != sizes.end() ? 16 : 32;
}

enum class Kind { rms, variance, moments };
struct Stats { float mean = 0.f, inv = 1.f; };

// Fixed is the per-item register capacity; callers must ensure it covers every channel.
// On B70, Florence-2 vision was 0.0350 s with a dynamic loop and 0.0336 s with 8 unrolled values
// (warm medians, 3 runs); the fixed loop keeps its values in registers.
template<Kind K, bool Rsqrt = false, int Fixed = 0, class Read, class Reduce>
inline Stats calculate(int64_t lane, int64_t step, int64_t cols, float eps, Read read, Reduce reduce) {
    float sum = 0.f, sum2 = 0.f;
    if constexpr (Fixed > 0) {
#pragma unroll
        for (int64_t k = 0; k < Fixed; ++k) {
            const int64_t c = lane + k * step;
            if (c < cols) {
                const float x = read(c);
                if constexpr (K != Kind::rms) sum += x;
                if constexpr (K != Kind::variance) sum2 += x * x;
            }
        }
    } else {
        for (int64_t c = lane; c < cols; c += step) {
            const float x = read(c);
            if constexpr (K != Kind::rms) sum += x;
            if constexpr (K != Kind::variance) sum2 += x * x;
        }
    }
    float mean = 0.f;
    if constexpr (K != Kind::rms) mean = reduce(sum) / (float) cols;
    if constexpr (K == Kind::variance) {
        if constexpr (Fixed > 0) {
#pragma unroll
            for (int64_t k = 0; k < Fixed; ++k) {
                const int64_t c = lane + k * step;
                if (c < cols) { const float d = read(c) - mean; sum2 += d * d; }
            }
        } else {
            for (int64_t c = lane; c < cols; c += step) {
                const float d = read(c) - mean;
                sum2 += d * d;
            }
        }
    }
    float var = reduce(sum2) / (float) cols;
    if constexpr (K == Kind::moments) var = sycl::fmax(var - mean * mean, 0.f);
    const float inv = Rsqrt ? sycl::rsqrt(var + eps) : 1.f / sycl::sqrt(var + eps);
    return {mean, inv};
}

}  // namespace strata::kernels::row_stats
