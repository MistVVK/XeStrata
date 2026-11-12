// include/strata/prefill/loan_rules.hpp - which chunk a prompt piece lends its buffers for (no GPU needed to check).
//
// A prompt piece of `tokens` tokens is read by one of two walks: below stream_all_min() the staged walk (only the
// experts its tokens route to are copied to the card, a small ring), from it up the streamed walk (every expert the
// card does not hold goes through a big ring, in layer order).  The walk is decided from the real token count, but
// the loan of expert-cache slots that holds its buffers is sized from the chunk the request rounds UP to the
// 256-token grid.  A 999-token piece under a 1,024 floor was lent the streamed layout (a 512-slot ring, the fused
// MoE layout) while it walked staged: about 478 more experts were streamed per prompt, 275 ms at 1,000 tokens
// (RTX A6000, UD-Q4_K_XL, 1,952 against 1,677 ms).  Every piece within 255 tokens under the floor was affected
// (769-1,023 at the default floor, 2,817-3,071 under the CPU share's 3,072).
//
// The loan is now sized from the walk the piece takes: a piece below the floor lends for floor - 1 tokens at most,
// so its layout is the staged one.  STRATA_PREFILL_LOAN_WALK=0 restores the rounded chunk.
#pragma once

#include <cstdint>

namespace strata::prefill::loan_rules {

/// `tokens`: the piece's real token count; `chunk`: the chunk the caller chose (rounded up to 256, at most the
/// configured chunk); `floor`: stream_all_min(), the token count from which the streamed walk is taken.  A piece
/// below the floor whose rounded chunk reaches it gets floor - 1; anything else keeps `chunk`.
constexpr int64_t walk_chunk(int64_t tokens, int64_t chunk, int64_t floor) {
    return (tokens > 0 && chunk > 0 && tokens < floor && chunk >= floor) ? floor - 1 : chunk;
}

}  // namespace strata::prefill::loan_rules
