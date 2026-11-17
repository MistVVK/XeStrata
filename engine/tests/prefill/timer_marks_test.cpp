// STRATA_PREFILL_TIMING mark coalescing (PR #4006): repeated marks in the running phase add nothing to the totals.
// No GPU: the timeline is simulated, one fake clock tick per launch.
#include <cstdio>
#include <vector>

#include "strata/prefill/timer_marks.hpp"

using strata::prefill::timer_mark_repeats;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

enum { kIdle, kDequant, kGemm, kWaitCopy, kPhases };

struct Mark { int phase; double t; };

// PfTimer::fold(): the gap between mark i and mark i+1 is charged to the phase of mark i.
static std::vector<double> fold_totals(const std::vector<Mark>& marks) {
    std::vector<double> ms(kPhases, 0.0);
    for (size_t i = 0; i + 1 < marks.size(); ++i) ms[marks[i].phase] += marks[i + 1].t - marks[i].t;
    return ms;
}

// PfTimer::mark(): record unless the newest recorded mark is in the same phase.
static std::vector<Mark> record(const std::vector<Mark>& calls, size_t* skipped) {
    std::vector<Mark> out;
    std::vector<int> ph;
    *skipped = 0;
    for (const Mark& m : calls) {
        if (timer_mark_repeats(ph.data(), ph.size(), m.phase)) { ++*skipped; continue; }
        out.push_back(m);
        ph.push_back(m.phase);
    }
    return out;
}

int main() {
    // the predicate
    const int ph[] = {kGemm, kDequant};
    CHECK(!timer_mark_repeats(ph, 0, kDequant));        // nothing recorded yet: always record
    CHECK(timer_mark_repeats(ph, 2, kDequant));         // the newest mark is already dequant
    CHECK(!timer_mark_repeats(ph, 2, kGemm));           // an older mark's phase does not count, only the newest
    CHECK(!timer_mark_repeats(ph, 1, kDequant));
    CHECK(timer_mark_repeats(ph, 1, kGemm));

    // a compute() like timeline: gemm, then one dequant mark per expert (16 of them), then a copy wait, then dequant again
    std::vector<Mark> calls;
    double t = 0;
    calls.push_back({kGemm, t});
    t += 5;
    for (int e = 0; e < 16; ++e) { calls.push_back({kDequant, t}); t += 0.5 + e * 0.01; }
    calls.push_back({kWaitCopy, t});
    t += 3;
    for (int e = 0; e < 4; ++e) { calls.push_back({kDequant, t}); t += 1.0; }
    calls.push_back({kIdle, t});                         // the run's closing mark

    size_t skipped = 0;
    const std::vector<Mark> kept = record(calls, &skipped);
    const std::vector<double> full = fold_totals(calls), coal = fold_totals(kept);
    CHECK(skipped == 15 + 3);                            // 18 records saved of 23
    CHECK(kept.size() == 5);                             // gemm, dequant, wait copy, dequant, idle
    for (int p = 0; p < kPhases; ++p) CHECK(full[p] - coal[p] < 1e-9 && coal[p] - full[p] < 1e-9);   // the totals are the same

    // alternating phases: nothing is skipped
    std::vector<Mark> alt;
    for (int i = 0; i < 10; ++i) alt.push_back({i % 2 ? kGemm : kDequant, (double) i});
    record(alt, &skipped);
    CHECK(skipped == 0);

    // the same phase for the whole run: one record plus the closing mark
    std::vector<Mark> same;
    for (int i = 0; i < 100; ++i) same.push_back({kDequant, (double) i});
    same.push_back({kIdle, 100.0});
    const std::vector<Mark> k2 = record(same, &skipped);
    CHECK(skipped == 99 && k2.size() == 2);
    CHECK(fold_totals(k2)[kDequant] == fold_totals(same)[kDequant]);

    // after a fold the timer keeps its last mark (used == 1): a repeat of that phase is still skipped
    const int kept_ph[] = {kWaitCopy};
    CHECK(timer_mark_repeats(kept_ph, 1, kWaitCopy));

    if (failed) { std::fprintf(stderr, "%d check(s) failed\n", failed); return 1; }
    std::puts("timer_marks_test: ok");
    return 0;
}
