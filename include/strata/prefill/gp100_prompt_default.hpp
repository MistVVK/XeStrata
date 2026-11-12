// include/strata/prefill/gp100_prompt_default.hpp - Tesla P100 (GP100, compute capability 6.0): whether the prompt experts take the FP16
// path by default instead of MMQ (prefill.cpp, mmq_plan()), with the few-row experts on the quantized grouped kernels (fewrows.hpp).
// GP100 has no __dp4a, so MMQ runs on the bit-exact emulation there; with the few-row route the FP16 path is at 0.95 / 0.91 / 0.93 / 1.00
// of MMQ's prompt speed at 300 / 1,000 / 2K / 4K tokens and above it from 8K (lane P4-FALLBACK, 11-11).  The rule is kept here, free of
// CUDA, so a test can walk its cases.
#pragma once

#include <atomic>

namespace strata::prefill {

/// The whole default: false keeps MMQ on the P100 as before; true makes the rule below the P100's default.  One line.
constexpr bool kGp100Default = true;

struct Gp100PromptInputs {
    const char* env = nullptr;   // STRATA_PREFILL_MMQ as read (nullptr: unset; any value keeps its old meaning: 0 FP16 path, else MMQ)
    bool all_cc60 = false;       // every visible CUDA card is compute capability 6.0 (one process, one decision)
    bool mmq_ws = false;         // STRATA_MMQ_WS=1 is in force: its weight-stationary kernels are MMQ's products
    bool nibble = false;         // STRATA_EXPERT_NIBBLE=1: the FP16 path would read the converted slots as GGUF blocks
    bool peer_device = false;    // --peer-device: the prompt peer (Prefill::set_peer) computes on MMQ only
};

/// true: the FP16 path is this process's default for the prompt experts.  Only with the variable unset, so an explicit
/// STRATA_PREFILL_MMQ=1 or =0 means what it always meant.
inline bool gp100_fp16_rule(const Gp100PromptInputs& in) {
    return in.env == nullptr && in.all_cc60 && !in.mmq_ws && !in.nibble && !in.peer_device;
}

/// Set by generate.cpp from --peer-device before any prompt path exists (mmq_plan() reads it once, at its first use).
inline std::atomic<bool>& peer_device_requested() {
    static std::atomic<bool> v{false};
    return v;
}

}  // namespace strata::prefill
