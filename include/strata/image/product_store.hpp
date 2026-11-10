// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace strata::image {

enum class ImageAct { none, relu, gelu, silu, sigmoid, relu6, leaky };

inline float image_activate(ImageAct act, float v, float slope = 0.f) {
    switch (act) {
    case ImageAct::relu: return sycl::fmax(v, 0.f);
    case ImageAct::gelu: return 0.5f * v * (1.f + sycl::erf(v * 0.70710678118654752f));
    case ImageAct::silu: return v / (1.f + sycl::exp(-v));
    case ImageAct::sigmoid: return 1.f / (1.f + sycl::exp(-v));
    case ImageAct::relu6: return sycl::clamp(v, 0.f, 6.f);
    case ImageAct::leaky: return v > 0.f ? v : v * slope;
    default: return v;
    }
}

// Rows of a product: optional bias, activation, then residual. Strides and offsets are in elements.
// Exactly one input and at least one output is set. The 16-bit input is FP16; bf selects BF16 for output/residual.
struct ProductStore {
    const float* in = nullptr;
    const uint16_t* in16 = nullptr;
    const float* bias = nullptr;
    const uint16_t* residual = nullptr;
    uint16_t* out16 = nullptr;
    uint16_t* out2 = nullptr;
    float* out = nullptr;
    int64_t rows = 0, cols = 0, in_stride = 0, out_stride = 0, offset = 0, from2 = 0;
    ImageAct act = ImageAct::none;
    float slope = 0.f;
    bool bf = false, add_zero = true;
};
void product_store(const ProductStore& p, void* stream);

}  // namespace strata::image
