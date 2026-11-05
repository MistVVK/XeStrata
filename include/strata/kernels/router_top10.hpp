// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/router_top10.hpp - the MoE router, host-callable (P2.S2).
//
// logits (n_tokens, n_expert) -> the top-k experts per token with renormalised weights: softmax over ALL experts,
// stable descending argsort (ties break by index), gather, then renormalise with ggml's 2**-14 lower clamp.  The
// native router's kernel (router.cpp) at any width: in FP32, as ggml, not ref/moe.py's double.
//
// The BF16 GEMV that PRODUCES the logits is a separate kernel and is not here; this handles the routing only,
// which is why `logits` is taken as an f32 buffer the caller has already filled.
#pragma once

#include <cstdint>

namespace strata::kernels {

// `stream` may be null, in which case the call synchronises before returning.  `ids` is (n_tokens, k) int32 and
// `weights` is (n_tokens, k) f32, both device pointers.  k must be <= 32
// and below n_expert, n_expert <= 1,024.
void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream);

}  // namespace strata::kernels
