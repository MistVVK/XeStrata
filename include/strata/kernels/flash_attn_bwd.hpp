// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/flash_attn_bwd.hpp - the attention's backward on the GPU's FP16 matrix engines
// (src/kernels/xe/flash_attn_bwd.cpp), FlashAttention-2's: P made again a chunk at a time from the scores and the
// forward's log-sum-exp, nothing of rows x rows kept.  Two kernels walk the queries for a sub-group's keys (dK, dV),
// a third the keys for its queries (dQ), on the forward's tiles (attn_parts.hpp: Intel's XMX, NVIDIA's tensor cores,
// RDNA's WMMA).  A training step's attention takes it (qi_attention_bwd).
#pragma once

#include <cstdint>

namespace strata::kernels {

struct FlashAttnBwdArgs {
    /// FP16 [rows][n_head * head_dim]: row t, head h at x + t * ld + h * head_dim; the rows past `rows` up to a
    /// multiple of 64 there and zero (the chunks are read whole).  dO is the output's gradient divided by its head's
    /// g (below): its values, and the score gradients made from them, then sit within FP16's range.
    const uint16_t *q = nullptr, *k = nullptr, *v = nullptr, *dO = nullptr;
    int64_t ld = 0;
    const float* lse = nullptr;   ///< [rows][n_head]: the forward's log-sum-exp of each row's scaled scores
    const float* D = nullptr;     ///< [n_head][rows]: dO . O of each row (the same g's units)
    const float* g = nullptr;     ///< [n_head]: the head's factor, the outputs multiplied by it
    int64_t rows = 0, n_head = 0, head_dim = 0;
    int64_t n_causal = 0;         ///< rows below it see the keys up to themselves (the forward's rule)
    float scale = 0.0f;           ///< the scores' scale (1 / sqrt(head_dim))
    float *dq = nullptr, *dk = nullptr, *dv = nullptr;   ///< FP32 [rows][n_head * head_dim]
    bool accumulate = false;      ///< dK and dV added into (dQ is always written)
    /// The query rows taken (0: every row): dK and dV of the queries below it alone, dQ of those rows alone (the
    /// rest the caller's: a training step's causal text rows beside oneDNN's image rows)
    int64_t q_rows = 0;
};

/// Launches the backward on `stream`, not waiting for it; false, nothing launched, where the GPU has no FP16 matrix
/// engines (flash_attn_path -1) or the shape is not one it takes (head_dim 128): the caller's products take it then.
/// Every GPU with them runs the same kernel: Intel's XMX, NVIDIA's tensor cores, RDNA3's and RDNA4's WMMA.
bool flash_attn_bwd(const FlashAttnBwdArgs& a, void* stream);

}  // namespace strata::kernels
