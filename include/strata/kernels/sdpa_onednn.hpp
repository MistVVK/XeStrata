// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/sdpa_onednn.hpp - oneDNN's SDPA on Intel GPUs (src/kernels/xe/sdpa_onednn.cpp, built with
// STRATA_ONEDNN): the image attention's unmasked rows (qi_sdpa) and the plain models' causal prompt (prompt_attn).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The attention of `rows` query rows over `keys` keys on the stream's GPU through oneDNN's fused SDPA: q16 and out16
/// [rows][heads * hd], k16 and v16 [keys][kv_heads * hd], all FP16 (query head h reads KV head h / (heads /
/// kv_heads)); scale 1 / sqrt(hd).  causal: row t sees the keys up to keys - rows + t (the rows the last of the keys'
/// positions).  `batch` attentions of that shape one after another in each tensor (a ViT's windows), in one call.  A
/// partition is compiled once a shape and the last 64 kept.  False where oneDNN does not take it (not an Intel GPU, a
/// failure: the caller's own kernel).
bool onednn_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys,
                 int64_t heads, int64_t kv_heads, int64_t hd, uint16_t* out16, void* stream, bool causal = false,
                 int64_t batch = 1);
/// Drops the causal partitions compiled on the stream's GPU and what oneDNN holds for them (a timing's shapes).
void onednn_sdpa_drop_causal(void* stream);

}  // namespace strata::kernels
