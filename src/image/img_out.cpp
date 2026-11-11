// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/img_out.cpp - see img_out.hpp.

#include "img_out.hpp"

#include "strata/image/gpu_buf.hpp"

#include <sycl/sycl.hpp>

namespace strata::image {

void patch_to_tokens(const float* in, int64_t N, int64_t L, float* out, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (N * 4 * L)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / (4 * L), j = i - t * 4 * L, p = j / L, c = j - p * L;
        out[t * 4 * L + c * 4 + p] = in[i];
    });
}

void to_rgba(const float* x, int64_t HW, int64_t C, TilePlanes layout, float* rgba, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (4 * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / HW, p = i - c * HW;
        const float v = layout == TilePlanes::Planes ? x[c * HW + p] : x[p * C + c];
        rgba[i] = c < C ? sycl::clamp((v + 1.f) * 0.5f, 0.f, 1.f) : 1.f;
    });
}

}  // namespace strata::image
