// include/strata/prefill/fp16_rows_rule.hpp - how many tokens the FP16 expert path's buffers (Xs: the gathered activations, Hh: the SwiGLU output) are sized for.
//
// In a fused layout (every layer covered by the fused kernels, a streamed chunk of stream_all_min() tokens or more: prefill.cpp fused_layout) the MoE buffers hold only the fused
// path's tables, and the layers' MMQ / FP16 products can run only on chunks BELOW the floor, so GU / H / Hq are already sized for stream_all_min() - 1 tokens (moe_bufs).
// Xs and Hh follow the same rule: a build without MMQ marks every layer as a fallback layer (no MMQ tile exists), which makes the two buffers exist at all, and at the full chunk
// they were 500 MiB of dead weight at 8,192 tokens that the prompt path then borrowed from the expert cache (224 more lent slots, refilled afterwards).
// Everything else keeps the full chunk: no fused layout (a card the fused kernels do not cover, the Tesla P100 whose FP16 path runs from the floor up, STRATA_PF_FUSED=0).
// A pure function; fp16_rows_rule_test checks it.  No GPU, no model.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::prefill {

/// Tokens the FP16 expert buffers (Xs, Hh) hold for a chunk of `chunk` tokens.  `fused_layout`: prefill.cpp's fused_layout(chunk); `stream_floor` = stream_all_min().
constexpr int64_t fp16_buf_tokens(int64_t chunk, bool fused_layout, int64_t stream_floor) {
    return fused_layout ? std::min<int64_t>(chunk, stream_floor - 1) : chunk;
}

}  // namespace strata::prefill
