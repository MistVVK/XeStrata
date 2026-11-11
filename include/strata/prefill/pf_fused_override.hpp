// include/strata/prefill/pf_fused_override.hpp - one process-wide switch that turns the fused prompt experts off after the environment was read.
//
// STRATA_PF_FUSED is read once by several static caches (moe_fused.cu, moe_fused_iq.cu, prefill.cpp's per-device fused_ring answer).  The small-card
// prompt path (strata::small_card_prompt_path in core/sm120_series.hpp) is decided in `serve` only after the expert cache is sized, because it keys on the
// prompt chunk the default path would get, so it cannot set the variable early.  It sets this instead; every one of those readers asks it first.
// -1: nothing was decided (the environment and the card's table rule), 0: the fused prompt experts are off for this process,
// kPfFusedKernelsOnly (2, the small-card prompt path since P4-SMALL): the fused kernels run wherever the card's rule takes them (the layer-major walk's groups,
// the streamed chunks), in MMQ's buffer layout: no fused layout and no fused ring (prefill.cpp fused_ring), so the walk is not declined.
#pragma once

#include <atomic>

namespace strata::prefill {

inline constexpr int kPfFusedKernelsOnly = 2;
inline std::atomic<int>& pf_fused_override() {
    static std::atomic<int> v{-1};
    return v;
}

}  // namespace strata::prefill
