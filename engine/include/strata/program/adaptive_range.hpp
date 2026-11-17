// include/strata/program/adaptive_range.hpp - which swaps of an adaptive-tier round have had their copy in (PR #4844, @uvlad23).
//
// A round of the adaptive tier moves a list of swaps (aswaps) into the GPU expert caches.  Step 2 copies them in, step 3
// marks the ones that landed as resident.  On the serial loop the copies go in batches (the arena may cap one), and step 3
// marks the batch that just landed, [lo, hi).  On the pipelined loop every swap is copied at its card's next gap and
// step 3 marks the whole list.  Which loop runs is decided per tick: the pipelined loop can drain after step 2 has been
// planned, and the next tick then takes step 3 on the serial rule.  lo/hi/bi then still named the last batch of an earlier
// round, so step 3 marked only that range and every other swap of the round was lost (its slot held the new expert, the
// table still said the old one).  The pipelined plan therefore sets the range to the whole round.  Plain arithmetic, so a
// host test can walk the rounds without a GPU.
#pragma once

#include <algorithm>
#include <cstddef>

namespace strata::program {

struct AdaptiveCopyRange {
    size_t lo = 0, hi = 0;   ///< the batch being copied in / just landed
    size_t bi = 0;           ///< where the next batch starts

    void clear() { lo = hi = bi = 0; }

    /// Serial loop, step 2: the next batch is [bi, bi + batch) of `total` swaps (all the rest when batch == 0, the arena's
    /// cap otherwise).  Never past `total`, and an overflowing sum is the end.
    void next_batch(size_t total, size_t batch) {
        lo = std::min(bi, total);
        hi = total;
        if (batch > 0 && batch < total - lo) hi = lo + batch;
        bi = hi;
    }

    /// Pipelined loop, step 2: the whole round is one batch, whatever an earlier round left here.
    void whole_round(size_t total) {
        lo = 0;
        hi = bi = total;
    }

    /// Step 3 on the serial rule: more batches of the round still to copy in.
    bool more(size_t total) const { return bi < total; }

    /// Step 3: the swaps that landed.  `pipelined` is this tick's rule; the range is the whole round on that one.
    size_t landed_begin(bool pipelined) const { return pipelined ? 0 : lo; }
    size_t landed_end(bool pipelined, size_t total) const { return pipelined ? total : hi; }
};

}  // namespace strata::program
