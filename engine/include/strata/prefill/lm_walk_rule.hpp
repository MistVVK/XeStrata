// include/strata/prefill/lm_walk_rule.hpp - which experts the layer-major prompt walk (src/prefill/prefill.cpp lm_run) counts and streams.
//
// #5104, #5354, #5966, #6018, #6134: a pruned pack (GSQ-RCO Coder IQ1_M: 48 layers, 256 experts) on an engine whose kernels are compiled for 512 experts
// (the default geometry, flash_next).  The walk used the compiled count NE for its tables, its prefetch loop and the stride of the resident-slot table
// (host_res is n_layers x the MODEL's n_expert), so it asked the stager for experts 256..511 that the pack does not have ("the expert source could not
// copy experts 256.. of layer 0", abort; 0xC0000409 on Windows) and read another layer's slots for every layer from 1 on.  It is reached from two chunks on a
// small expert cache (the small-card rule) and on every layer-major read.  The count is the model's, never above the compiled one.
// Pure: no GPU, no environment read.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::prefill {

/// the experts per layer the walk loops over: the model's, capped at the compiled shape's (the tables are sized for that many at most)
inline int32_t lm_expert_count(int64_t model_n_expert, int64_t compiled_n_expert) {
    return (int32_t) std::max<int64_t>(0, std::min(model_n_expert, compiled_n_expert));
}

/// the experts of `layer` the prefetch streams, in id order: every id of the model's range that has no resident slot (`host_res`: n_layers x `n_expert` slots, -1
/// none, nullptr no table) and that the source does not hold pinned.  `pinned(layer, e)` is the source's answer.
template <class Pinned>
std::vector<int32_t> lm_streamed_experts(int64_t layer, int32_t n_expert, const int32_t* host_res, Pinned&& pinned) {
    std::vector<int32_t> out;
    for (int32_t e = 0; e < n_expert; ++e) {
        if (host_res != nullptr && host_res[(size_t) layer * (size_t) n_expert + (size_t) e] >= 0) continue;
        if (pinned(layer, e)) continue;
        out.push_back(e);
    }
    return out;
}

}  // namespace strata::prefill
