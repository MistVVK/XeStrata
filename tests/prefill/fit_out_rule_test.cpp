// Which tail the serve auto chunk scan counts in a build without MMQ (fit_out_rule.hpp): no GPU, no model.
//
// The scan is modelled with the loans measured on an RTX 5070 12 GB (Q2_0 plain pack, build without MMQ, 2026-11-14,
// RESULTS-CHUNK-BALANCE block 4k): 3,572 slots at 9,984 tokens with today's tail (the start-up borrow), 3,391 at 9,984 with
// tail 0 (a request's relayout), 3,417 at 9,216 with today's tail; 0.2018 slots per token between 9,216 and 9,984.  Cache
// slots from the auto sizing (1.3175 MiB a slot): 3,851 on the 12 GB card, about 6,960 for 16 GB.  The model is the scan's test
// (loan_fits, kAutoLendPct 90, keep 128) and its bisection (biggest_chunk) applied to a linear loan; the engine's own loan
// comes from Prefill::bytes_needed.
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include "strata/prefill/fit_out_rule.hpp"

namespace fo = strata::prefill::fit_out;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// slots a chunk of t tokens lends (tail -1: floor - 1 rows; tail 0), from the 5070's loans
static int64_t loan(int64_t t, int64_t tail) {
    const int64_t a = tail < 0 ? 1558 : 1377;   // 3,572 - 2,014 and 3,391 - 2,014 (2,014 = 9,984 x 2,018 / 10,000)
    return a + t * 2018 / 10000;
}
// the scan at one tail: the largest chunk up to `ceiling` whose loan passes the 90 % cap and the 128-slot floor
static int64_t scan(int64_t slots, int64_t tail, int64_t ceiling = 32768) {
    return fo::biggest_chunk(ceiling, [&](int64_t t) { return fo::loan_fits(loan(t, tail), slots, 128, 90, true); });
}
static fo::ScanCase one_gpu_case() {
    fo::ScanCase c;
    c.auto_chunk = c.one_gpu = c.tail_loan = c.equal_cut = c.fused_at_chunk = true;
    return c;
}

int main() {
    // the model reproduces the measured loans and today's chunk
    CHECK(loan(9984, -1) == 3572);
    CHECK(loan(9984, 0) == 3391);
    CHECK(loan(9216, -1) == 3416 || loan(9216, -1) == 3417);

    // which tail: 0 only in the case where no layout above small_max can carry a short tail
    const fo::ScanCase on = one_gpu_case();
    CHECK(fo::scan_tail(on) == 0);
    {
        fo::ScanCase c = on; c.auto_chunk = false; CHECK(fo::scan_tail(c) == -1);      // --prefill N
        c = on; c.one_gpu = false; CHECK(fo::scan_tail(c) == -1);                      // a split, a peer card, slots, head alias
        c = on; c.tail_loan = false; CHECK(fo::scan_tail(c) == -1);                    // STRATA_PREFILL_TAIL_LOAN=0
        c = on; c.equal_cut = false; CHECK(fo::scan_tail(c) == -1);                    // STRATA_PREFILL_EQUAL=0: today's count
        c = on; c.small_path = true; CHECK(fo::scan_tail(c) == -1);                    // the small-card prompt path
        c = on; c.fused_at_chunk = false; CHECK(fo::scan_tail(c) == -1);               // no fused layout: no tail is carved
    }
    // with the equal cut off the chunk is today's whatever the tail-0 scan would find
    CHECK(fo::pick(9216, 10240, -1) == 9216);
    CHECK(fo::pick(9216, 10240, 0) == 10240);
    CHECK(fo::pick(9216, 8960, 0) == 9216);   // never smaller than today's

    // 12 GB (3,851 slots): today 9,216 (measured), tail 0 10,240 (the build with MMQ takes 9,984 there)
    CHECK(scan(3851, -1) == 9216);
    CHECK(scan(3851, 0) == 10240);
    CHECK(fo::pick(scan(3851, -1), scan(3851, 0), fo::scan_tail(on)) == 10240);
    // the start-up layout of 10,240 with today's tail keeps the 128-slot floor (the engine checks it before taking the chunk)
    CHECK(fo::loan_fits(loan(10240, -1), 3851, 128, 90, false));
    // 16 GB (about 6,960 slots): 23,296 today, 24,064 with tail 0
    CHECK(scan(6960, -1) == 23296);
    CHECK(scan(6960, 0) == 24064);
    // 24 GB (about 13,178 slots): the 32,768 ceiling either way
    CHECK(scan(13178, -1) == 32768 && scan(13178, 0) == 32768);

    // the band where a tail-0 count would cross the small-card line (chunk 8,192): 3,409-3,566 slots.  The small-card decision is
    // taken on today's count (generate.cpp: the tail-0 scan runs after it, and only when the path was not taken), so a card there
    // keeps the path it takes today.
    for (int64_t slots : {3409, 3480, 3566}) {
        const int64_t today = scan(slots, -1);
        CHECK(today < 8192);              // today: the small-card path is chosen
        CHECK(scan(slots, 0) >= 8192);    // a tail-0 count would have left it
        fo::ScanCase c = on;
        c.small_path = today < 8192;      // decided on today's count
        CHECK(fo::scan_tail(c) == -1);
        CHECK(fo::pick(today, scan(slots, 0), fo::scan_tail(c)) == today);
    }
    // just above the band the small-card path is not chosen today and the tail-0 count applies
    CHECK(scan(3567, -1) >= 8192 || scan(3600, -1) >= 8192);

    // the request path's today's-cut fallback after a tail-0 scan: only at or below small_max (0.1.39's chunk, 8,192 here)
    CHECK(fo::today_cut_ok(-1, 10240, 8192));
    CHECK(fo::today_cut_ok(0, 8192, 8192));
    CHECK(!fo::today_cut_ok(0, 10240, 8192));

    if (failed) { std::fprintf(stderr, "fit_out_rule_test: %d failed\n", failed); return 1; }
    std::printf("fit_out_rule_test: all passed\n");
    return 0;
}
