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

// The tail of a lent fused layout.  A layout laid out for chunks at or above the floor (the fused MoE buffers) also
// holds MMQ's rows for a chunk BELOW the floor, which runs staged in the same buffers.  Sizing that for floor - 1
// tokens costs expert slots on every prompt, whatever its own chunks are (a floor of 3,584 against 1,024: about 289 MB,
// 92 slots of an RTX A6000).  A segment of `tokens` read in chunks of `chunk` has a chunk below the floor only at its
// end, so that is all the tail it needs.

/// The most MMQ rows (in tokens) a segment of `tokens` read in chunks of `chunk` puts into the fused layout's buffers:
/// the last chunk when it is below `floor`, else 0 (every other chunk is a full one, at or above the floor, and runs
/// the fused kernels).  A layout whose chunk is itself below the floor is not a fused layout: min(chunk, tokens).
constexpr int64_t tail_need(int64_t tokens, int64_t chunk, int64_t floor) {
    if (tokens <= 0 || chunk <= 0) return 0;
    if (chunk < floor) return tokens < chunk ? tokens : chunk;
    const int64_t last = tokens > chunk ? tokens - ((tokens - 1) / chunk) * chunk : tokens;
    return last < floor ? last : 0;
}

/// The tail a layout is carved with for that need: on the 256-token grid (so that two requests with nearly the same tail
/// keep one layout), at most floor - 1 (today's size), 0 for no need.
constexpr int64_t tail_layout(int64_t need, int64_t floor) {
    if (need <= 0 || floor <= 1) return 0;
    const int64_t grid = (need + 255) / 256 * 256;
    return grid < floor - 1 ? grid : floor - 1;
}

/// The tail a layout for a segment is carved with (what lend asks the layout for).
constexpr int64_t segment_tail(int64_t tokens, int64_t chunk, int64_t floor) {
    return tail_layout(tail_need(tokens, chunk, floor), floor);
}

/// The tail a lend carves a layout with, given the layout that is in place: GROW-ONLY while the layout's chunk stays.
/// `layout_chunk` / `layout_tail`: the layout now laid out (a tail of -1: carved with today's floor - 1, i.e. never sized
/// by a segment: the start-up layout, or the way-back switch); `want`: the chunk this segment's loan is for; `need`: its
/// segment_tail().  A layout of the same chunk that is tail-sized keeps its tail when it covers the need (no relayout
/// for a smaller follow-up) and otherwise goes straight to today's size: floor - 1, so a sequence of segments costs at
/// most ONE relayout more than today's sizing, then runs on today's bytes.  Any other case (another chunk: a relayout
/// today as well; a start-up layout) starts again from the need of that moment.
constexpr int64_t next_tail(int64_t layout_chunk, int64_t layout_tail, int64_t want, int64_t need, int64_t floor) {
    if (layout_tail >= 0 && layout_chunk == want) return layout_tail >= need ? layout_tail : (floor > 1 ? floor - 1 : 0);
    return need;
}

/// A loan carved for `lent_chunk` with room for `lent_tail` MMQ tokens serves a later segment of `tokens` (wanting
/// chunks of `want`) without a new layout only if its chunk covers `want` AND the segment, read in the loan's own
/// chunks (a layout always reads in its own chunk), needs no more tail than the loan holds.
constexpr bool reuse_ok(int64_t want, int64_t tokens, int64_t lent_chunk, int64_t lent_tail, int64_t floor) {
    return want <= lent_chunk && tail_need(tokens, lent_chunk, floor) <= lent_tail;
}

}  // namespace strata::prefill::loan_rules
