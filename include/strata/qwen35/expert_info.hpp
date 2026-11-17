#pragma once
// #5012: the qwen35moe engine's (src/qwen35/q36_engine.cu) expert placement in its INFO line, the figures the Monitor shows as
// "Expert cache: <expert_slots> / <experts>": the main layers' routed experts resident in VRAM, all of them, and their VRAM.  The
// MTP block's drafting experts are not counted, as in the main engine's INFO line (experts = n_layers x n_expert there too).
// Header-only (no CUDA), so tests/qwen35/expert_info_test.cpp pins it without a GPU or a model.

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::qwen35 {

struct ExpertInfo {
    long long slots = 0;       ///< routed experts resident in VRAM (every layer's n_slots)
    long long experts = 0;     ///< every routed (layer, expert) pair
    long long cache_mib = 0;   ///< the resident experts' VRAM
};

/// `layers`: the main layers, each with `n_slots` (its experts in VRAM: all of them when they fit, a hybrid layer's slot count) and
/// `stride` (one expert's bytes).
template <class Layers>
ExpertInfo expert_info(const Layers& layers, int64_t n_expert) {
    ExpertInfo r;
    long long bytes = 0;
    for (const auto& L : layers) {
        r.slots += (long long) L.n_slots;
        bytes += (long long) L.n_slots * (long long) L.stride;
    }
    r.experts = (long long) layers.size() * (long long) n_expert;
    r.cache_mib = bytes >> 20;
    return r;
}

/// The INFO line (the server keeps every key; no vram_elastic key: this engine has no VRAM command, which the server checks for).
inline std::string expert_info_line(const ExpertInfo& e) {
    char b[160];
    std::snprintf(b, sizeof b, "INFO expert_slots=%lld experts=%lld expert_cache_mib=%lld\n", e.slots, e.experts, e.cache_mib);
    return b;
}

}  // namespace strata::qwen35
