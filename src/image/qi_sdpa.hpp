// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_sdpa.hpp - qi_sdpa's GPU makers' kernels (qi_sdpa.cpp chooses), each in a file of its own: their
// headers do not go together (CUDA's and HIP's define the same vector types).  Each takes the attention of query rows
// q0 .. q1 - 1 over all T keys on q's device, false where it does not (a failure: the engine's own kernel).  aotriton's
// is in a module of its own (strata_sdpa_aotriton), which qi_sdpa.cpp opens at run time; oneDNN's is the kernels'
// (sdpa_onednn.hpp), which the plain models' prompt shares.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::image::sdpa {

bool cudnn(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1,
           int64_t T, int64_t heads, int64_t hd, uint16_t* out16);
bool aotriton(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1,
              int64_t T, int64_t heads, int64_t hd, uint16_t* out16, bool causal);

}  // namespace strata::image::sdpa
