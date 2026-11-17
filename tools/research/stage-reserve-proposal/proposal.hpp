// research/stage-reserve-proposal/proposal.hpp - PROPOSAL, not used by the engine (moved out of
// include/strata/program/stage_reserve.hpp in cleanup step 4.2; the text of both functions is unchanged).
// See README.md in this folder.
#pragma once

#include "strata/program/stage_reserve.hpp"

#include <algorithm>
#include <cstdint>

namespace strata::program::stage_reserve {

constexpr int64_t kGraphMib = 6;            ///< one window graph, captured on first use (log of #3702: 4, 6, 6)
constexpr int64_t kInstantiateBurstMib = 48; ///< the largest single step seen (42 MiB in #3702): room kept for one more

/// PROPOSAL, unused: the window term of a later stage when the windows of every size 1..max_window_t can still be
/// captured after loading: the first window's arena (kWindowMib already holds it), a graph for each further size, and
/// one burst.
inline int64_t proposed_window_mib(int max_window_t) {
    const int64_t sizes = std::max<int64_t>(max_window_t, 1);
    return kWindowMib + (sizes - 1) * kGraphMib + kInstantiateBurstMib;
}

/// PROPOSAL, unused: reserve_mib with the proposed window term on a later stage.
inline int64_t proposed_reserve_mib(const Inputs& in, int max_window_t) {
    Inputs p = in;
    int64_t r = reserve_mib(p);
    if (in.later) r += proposed_window_mib(max_window_t) - kWindowMib;
    return r;
}

}  // namespace strata::program::stage_reserve
