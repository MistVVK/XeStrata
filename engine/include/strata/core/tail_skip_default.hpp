// STRATA_ROUTE_TAIL_SKIP: the rank from which a missed expert that every token of a verify window routes that low is neither copied nor
// computed.  The value when the variable is NOT set is 0 (off) on every backend (owner decision 2026-11-19, engine 0.1.63; 0.1.42 to 0.1.62
// set rank 7 at the serve start-up on CUDA and on HIP gfx1200/gfx1201 where it applied).  STRATA_ROUTE_TAIL_SKIP=7 turns it on, 1 to 9 pick a
// rank (only 7 is measured).  Header-only so a test can pin the rule.
#pragma once

#include <cstdlib>

namespace strata::core {

/// The rank when STRATA_ROUTE_TAIL_SKIP is not set.
constexpr int kTailSkipUnsetRank = 0;

/// The rank an environment text means: not set -> kTailSkipUnsetRank; a whole number from 1 to 9 -> itself; anything else (0, empty, text,
/// a number outside 1..9) -> 0, as before.
inline int tail_skip_rank_from_text(const char* text) {
    if (text == nullptr) return kTailSkipUnsetRank;
    const int r = std::atoi(text);
    return r > 0 && r < 10 ? r : 0;
}

}  // namespace strata::core
