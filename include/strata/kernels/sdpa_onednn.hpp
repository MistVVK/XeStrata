// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/sdpa_onednn.hpp - oneDNN's SDPA on Intel GPUs (src/kernels/xe/sdpa_onednn.cpp, built with
// STRATA_ONEDNN): the image attention's unmasked rows (qi_sdpa), the plain models' causal prompt (prompt_attn) and a
// training step's attention (qi_attention_train, qi_attention_bwd).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The attention of `rows` query rows over `keys` keys on the stream's GPU through oneDNN's fused SDPA: q16 and out16
/// [rows][heads * hd], k16 and v16 [keys][kv_heads * hd], all FP16 (query head h reads KV head h / (heads /
/// kv_heads)); scale 1 / sqrt(hd).  causal: row t sees the keys up to keys - rows + t (the rows the last of the keys'
/// positions), with a `window` (causal only) the last `window` of them (an explicit mask: a sliding window's layers).
/// `batch` attentions of that shape one after another in each tensor (a ViT's windows), in one call.  A
/// partition is compiled once a shape and the last 64 kept.  False where oneDNN does not take it (not an Intel GPU, a
/// failure: the caller's own kernel).
bool onednn_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys,
                 int64_t heads, int64_t kv_heads, int64_t hd, uint16_t* out16, void* stream, bool causal = false,
                 int64_t batch = 1, int64_t window = 0);
/// A training step's attention through oneDNN's SDPA for training (no mask, kv_heads = heads): the forward with the
/// softmax's statistics, lse [heads][rows] FP32 (head h, row t at lse[h * rows + t]: the log-sum-exp of the row's
/// scaled scores; dense: its fused backward read them so whatever their strides, 2026-11-09), the output FP16; the
/// strides as onednn_sdpa's.
bool onednn_sdpa_train_fwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys,
                           int64_t heads, int64_t hd, uint16_t* out16, float* lse, void* stream);
/// Its backward: o16 the forward's output and do16 its gradient (FP16, in any one unit: the gradients come out in it),
/// lse the forward's; dq16 [rows][heads * hd], dk16 and dv16 [keys][heads * hd] FP16, written.  oneDNN takes FP16
/// gradients only (FP32 ones left its fused kernel for one five times slower on the B70): the caller keeps them in
/// range.
bool onednn_sdpa_train_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const uint16_t* o16,
                           const float* lse, const uint16_t* do16, int64_t rows, int64_t keys, int64_t heads,
                           int64_t hd, uint16_t* dq16, uint16_t* dk16, uint16_t* dv16, void* stream);
/// Drops the causal partitions compiled on the stream's GPU and what oneDNN holds for them (a timing's shapes).
void onednn_sdpa_drop_causal(void* stream);

}  // namespace strata::kernels
