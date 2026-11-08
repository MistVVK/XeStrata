// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/sampler_host.hpp - the sampler chain of sampler.hpp on the host, for a forward that runs on
// the CPU: the same chain (bias and mask, penalties, top_k, top_p, min_p, temperature, one Philox draw or the
// Gumbel-max pick), the same selection order (the penalised value descending, the id ascending; NaN and -inf never
// taken) and the same draws keyed by (seed, counter + row), in double as the GPUs with FP64 compute it.  So a seed
// gives the CPU the tokens the GPU picks from the same logits.
#pragma once

#include "strata/kernels/sampler.hpp"

#include <cstdint>

namespace strata::kernels {

/// sample_tokens with HOST pointers: `logits` (n_tokens x n_vocab), `history` (n_tokens rows of history_len, or
/// null), `p.logit_bias` and `p.allowed` host arrays as well; one token id per row into `out`.
void sample_tokens_host(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                        const SamplerParams& p, int* out);

}  // namespace strata::kernels
