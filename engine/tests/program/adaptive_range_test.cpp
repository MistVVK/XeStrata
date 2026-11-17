// The adaptive tier's copy-in range (PR #4844): which swaps of a round step 3 marks resident.  No GPU: the rounds are
// walked as generate.cpp walks them, and each swap is a counter that says whether it was copied in and whether it was marked.
#include <cstdio>
#include <vector>

#include "strata/program/adaptive_range.hpp"

using strata::program::AdaptiveCopyRange;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

struct Round {
    size_t total;
    std::vector<int> copied, marked;
    explicit Round(size_t n) : total(n), copied(n, 0), marked(n, 0) {}
    static bool all(const std::vector<int>& v) { for (int x : v) if (x != 1) return false; return true; }
};

// the serial loop: plan (clear + next_batch), copy the batch, step 3 marks it, repeat while more() (the arena's cap)
static void serial_round(AdaptiveCopyRange& r, Round& rd, size_t batch) {
    r.clear();
    r.next_batch(rd.total, batch);
    for (;;) {
        for (size_t i = r.lo; i < r.hi; ++i) ++rd.copied[i];
        for (size_t i = r.landed_begin(false); i < r.landed_end(false, rd.total); ++i) ++rd.marked[i];
        if (!r.more(rd.total)) break;
        r.next_batch(rd.total, batch);
    }
}

int main() {
    // the serial loop covers every swap once, whatever the cap, including zero swaps and one swap
    for (size_t total : {(size_t) 0, (size_t) 1, (size_t) 2, (size_t) 5, (size_t) 8, (size_t) 64})
        for (size_t batch : {(size_t) 0, (size_t) 1, (size_t) 2, (size_t) 3, (size_t) 7, (size_t) 8, (size_t) 100}) {
            AdaptiveCopyRange r;
            Round rd(total);
            serial_round(r, rd, batch);
            CHECK(Round::all(rd.copied) && Round::all(rd.marked));
            CHECK(r.bi == total && r.hi == total);
        }

    // a batch never runs past the list, even for a cap near the size_t limit
    {
        AdaptiveCopyRange r;
        r.next_batch(5, (size_t) -1);
        CHECK(r.lo == 0 && r.hi == 5 && r.bi == 5);
        r.next_batch(5, 2);                     // asked again after the end: an empty batch at the end
        CHECK(r.lo == 5 && r.hi == 5 && !r.more(5));
    }

    // the lost experts of #4844: a round planned on the pipelined loop whose loop drained before step 3 is marked on the
    // serial rule.  An earlier serial round left [lo, hi) at its last batch.
    {
        AdaptiveCopyRange r;
        Round first(5);
        serial_round(r, first, 2);              // leaves lo = 4, hi = 5, bi = 5
        CHECK(r.lo == 4 && r.hi == 5);

        // the old plan (nothing set for a pipelined round): step 3 on the serial rule marks only the stale range
        {
            AdaptiveCopyRange stale = r;
            Round pl(7);
            for (size_t i = 0; i < pl.total; ++i) ++pl.copied[i];                       // a_in copied all of them at the gaps
            for (size_t i = stale.landed_begin(false); i < stale.landed_end(false, pl.total); ++i) ++pl.marked[i];
            CHECK(!Round::all(pl.marked));                                              // the bug this test pins
        }

        // the fix: the pipelined plan names the whole round
        Round pl(7);
        r.whole_round(pl.total);
        for (size_t i = 0; i < pl.total; ++i) ++pl.copied[i];
        for (size_t i = r.landed_begin(false); i < r.landed_end(false, pl.total); ++i) ++pl.marked[i];   // the drained tick: serial rule
        CHECK(Round::all(pl.marked));
        CHECK(!r.more(pl.total));                                                       // and no second batch is posted
        // the same round finished on the pipelined rule marks the same range
        CHECK(r.landed_begin(true) == r.landed_begin(false) && r.landed_end(true, pl.total) == r.landed_end(false, pl.total));
    }

    // one swap, and an empty round, on the pipelined plan
    {
        AdaptiveCopyRange r;
        r.next_batch(9, 4);
        r.whole_round(1);
        CHECK(r.landed_begin(false) == 0 && r.landed_end(false, 1) == 1 && !r.more(1));
        r.whole_round(0);
        CHECK(r.landed_begin(false) == 0 && r.landed_end(false, 0) == 0 && !r.more(0));
    }

    // a serial round that follows a pipelined one starts clean
    {
        AdaptiveCopyRange r;
        r.whole_round(6);
        Round rd(3);
        serial_round(r, rd, 2);
        CHECK(Round::all(rd.copied) && Round::all(rd.marked));
    }

    if (failed) { std::fprintf(stderr, "%d check(s) failed\n", failed); return 1; }
    std::puts("adaptive_range_test: ok");
    return 0;
}
