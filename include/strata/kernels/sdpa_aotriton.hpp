// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/sdpa_aotriton.hpp - aotriton's flash attention on AMD GPUs (src/kernels/xe/sdpa_aotriton.cpp
// opens its module, src/image/qi_sdpa_aotriton.cpp, where the build has it): the image attention (qi_sdpa) and the
// plain models' causal prompt (prompt_attn).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The attention of `rows` query rows over `keys` keys on the stream's GPU through aotriton: q16 and out16
/// [rows][heads * hd], k16 and v16 [keys][kv_heads * hd], all FP16 (query head h reads KV head h / (heads /
/// kv_heads)); scale 1 / sqrt(hd).  causal: row t sees the keys up to keys - rows + t.  False where it does not take it
/// (not an AMD GPU, a build or a machine without the module, a failure: the caller's own kernel).
bool aotriton_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys,
                   int64_t heads, int64_t kv_heads, int64_t hd, uint16_t* out16, void* stream, bool causal = false);
/// aotriton_sdpa can run on the stream's GPU (an AMD GPU and the module opened).
bool aotriton_sdpa_ready(void* stream);

}  // namespace strata::kernels
