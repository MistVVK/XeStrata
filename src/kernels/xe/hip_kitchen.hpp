// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/hip_kitchen.hpp - comfy-kitchen's HIP kernels (src/kernels/hip/kitchen_module.cpp, opened by
// i8_gemm.cpp on an AMD GPU) for the other kernels' files: its int8 attention.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels::hip_kitchen {

/// Its int8 attention on q's GPU: query rows r0 .. T - 1 of T keys, FP16 [rows][heads * hd], out16's row 0 query row
/// `first`; false where the GPU does not take it (nothing written).
bool attn(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
          int64_t hd, int64_t r0, int64_t first, uint16_t* out16);

}  // namespace strata::kernels::hip_kitchen
