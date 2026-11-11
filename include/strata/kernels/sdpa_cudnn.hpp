// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/sdpa_cudnn.hpp - cuDNN's SDPA on NVIDIA GPUs (src/kernels/xe/sdpa_cudnn.cpp, built with
// STRATA_CUDNN in the contrib-llvm mode, cuDNN opened at the first use): the plain models' causal prompt (prompt_attn),
// as the image attention's.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The attention of `rows` query rows over `keys` keys on the stream's GPU through cuDNN's SDPA, onednn_sdpa's layouts
/// and causal mask (q16 and out16 [rows][heads * hd], k16 and v16 [keys][kv_heads * hd], FP16; row t sees the keys up
/// to keys - rows + t, with a `window` the last `window` of them).  A plan is built once a shape and kept.  False where
/// cuDNN does not take it (not an NVIDIA GPU, cuDNN not installed, no plan: the caller's own kernel).
bool cudnn_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys, int64_t heads,
                int64_t kv_heads, int64_t hd, uint16_t* out16, void* stream, bool causal, int64_t window = 0);

}  // namespace strata::kernels
