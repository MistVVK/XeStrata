// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Internal cuDNN entry points shared by image inference and training.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::image::sdpa {

/// `batch` attentions of the same shape one after another in each tensor (q0 0 where there are more than one).
bool cudnn(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1,
           int64_t T, int64_t heads, int64_t hd, uint16_t* out16, int64_t batch = 1);
/// A training step's attention through cuDNN (no mask, kv_heads = heads): the forward (bwd false) into out16 with the
/// statistics [heads][rows] FP32, or the backward from o16, do16 and the statistics into FP16 dq16, dk16 and dv16
/// (kernels::aotriton_sdpa_train's shapes and units)
bool cudnn_train(sycl::queue& q, bool bwd, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16,
                 const uint16_t* o16, const uint16_t* do16, float* stats, int64_t rows, int64_t keys, int64_t heads,
                 int64_t hd, uint16_t* out16, uint16_t* dq16, uint16_t* dk16, uint16_t* dv16);


}  // namespace strata::image::sdpa
