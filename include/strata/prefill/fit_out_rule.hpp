// include/strata/prefill/fit_out_rule.hpp - which tail the serve auto chunk scan counts in a build without MMQ (no GPU needed to check).
//
// The scan (generate.cpp, the serve prompt path) takes the largest chunk on the 256-token grid whose loan of expert-cache slots
// keeps the 128-slot floor and at most kAutoLendPct % of the cache, with the ring at its full byte budget.  It counts every layout
// with the MMQ / FP16 rows of a short last chunk under the stream floor (tail -1 = floor - 1 tokens: GU, H, Hq and, in a build
// without MMQ, the FP16 Xs / Hh).  With the equal cut on one GPU (equal_cut_rule.hpp) and the request-sized tail loan on, no
// layout the request path carves at a chunk above the scan's small_max has such a tail: every chunk of an equal cut is at least
// half the chunk (above the floor), and a one-chunk prompt there is above the floor too, so its tail is 0.  The scan then counts
// what those layouts really carve: tail 0.  On an RTX 5070 12 GB (Q2_0 plain pack) that moves the auto chunk from 9,216 to an
// expected 10,240 tokens, where the build with MMQ takes 9,984; at equal chunks the two builds read level (1.001-1.002).
//
// Today's count stays wherever a layout with a short tail can still be carved at the scan's chunk: a build with MMQ (this header
// is not used there), --prefill N, a layer split, a peer card, more than one slot, a head alias, STRATA_PREFILL_TAIL_LOAN=0,
// STRATA_PREFILL_EQUAL=0, a chunk that is not a fused layout, the small-card prompt path (decided on today's count, so no card
// changes path), and the host-aware plan (#2824), whose own chunk is planned on today's count.
#pragma once

#include <cstdint>

namespace strata::prefill::fit_out {

/// What the serve scan knows when it picks its chunk.
struct ScanCase {
    bool auto_chunk = false;      // --prefill auto (an explicit --prefill N only has to fit)
    bool one_gpu = false;         // no layer split, no peer card, one slot, no head alias
    bool tail_loan = false;       // STRATA_PREFILL_TAIL_LOAN not 0: a request's layout is carved with its own tail
    bool equal_cut = false;       // equal_cut::pick() gives the equal cut for one GPU (STRATA_PREFILL_EQUAL not 0)
    bool small_path = false;      // the small-card prompt path was taken (decided on today's count)
    bool fused_at_chunk = false;  // today's chunk is a fused layout (Prefill::tail_matters), so a tail is carved at all
};

/// The tail the scan counts: 0 when no layout above small_max can carry a short tail, -1 (floor - 1, today) otherwise.
constexpr int64_t scan_tail(const ScanCase& c) {
    return c.auto_chunk && c.one_gpu && c.tail_loan && c.equal_cut && !c.small_path && c.fused_at_chunk ? 0 : -1;
}

/// The chunk the scan takes: the tail-0 scan's chunk when it counted tail 0 and found a larger one, today's otherwise.
constexpr int64_t pick(int64_t today, int64_t tail0, int64_t tail) {
    return tail == 0 && tail0 > today ? tail0 : today;
}

/// The request path keeps today's cut when the equal cut's loan is larger (equal_cut::loan_ok); after a tail-0 scan it may do so
/// only for a chunk at or below small_max (a cut above it would carve the old short tail at the scan's chunk, which the scan did
/// not count).  Above small_max the equal loan is never the larger one: both use the byte-budget ring and the equal chunk is the
/// smaller chunk with no tail.
constexpr bool today_cut_ok(int64_t tail, int64_t today_chunk, int64_t small_max) {
    return tail < 0 || today_chunk <= small_max;
}

/// The scan's slot test (generate.cpp fits_one): `k` slots lent out of `slots`, `keep` kept, at most `pct` % when `cap`.
constexpr bool loan_fits(int64_t k, int64_t slots, int64_t keep, int64_t pct, bool cap) {
    return k > 0 && k + keep <= slots && !(cap && k * 100 > pct * slots);
}

/// The scan's bisection on the 256-token grid (generate.cpp biggest_chunk): the largest multiple of 256 up to `ceiling` that
/// `ok` accepts, for an `ok` monotone in the chunk.
template <class Ok>
constexpr int64_t biggest_chunk(int64_t ceiling, Ok&& ok) {
    if (ceiling < 256) return 0;
    int64_t lo = 1, hi = ceiling / 256, best = 0;
    while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (ok(mid * 256)) { best = mid * 256; lo = mid + 1; } else hi = mid - 1;
    }
    return best;
}

}  // namespace strata::prefill::fit_out
