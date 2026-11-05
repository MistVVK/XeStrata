// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/gelu.hpp - the image engine's two GELUs, for every model's kernels (device code: SYCL files
// only).
#pragma once

#include <sycl/sycl.hpp>

namespace strata::image {

/// GELU, the tanh form (PyTorch's approximate="tanh"; Qwen-Image's and Mage-Flow's DiTs)
inline float gelu_tanh(float v) {
    return 0.5f * v * (1.f + sycl::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

/// GELU, the erf form (PyTorch's nn.GELU(); the vision reader, Anima, the VAEs)
inline float gelu_erf(float v) { return 0.5f * v * (1.f + sycl::erf(v * 0.70710678118654752f)); }

}  // namespace strata::image
