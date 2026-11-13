// src/spec/draft_policy_test.cpp - DraftPolicy: when does a lookup window beat the MTP's?
//
// Simulated rounds with the costs measured on the RTX 5070 (window-cost: ~10 ms more per token) check that
//   1. with no lookup proposal the MTP window is kept;
//   2. lookup drafts that are mostly rejected stop being taken (their bucket's rate falls);
//   3. lookup drafts that are always accepted are taken, and the window grows with them;
//   4. match-length buckets learn separately (short matches failing does not stop long ones);
//   5. the policy never proposes a window beyond its cap.
#include "strata/spec/draft_policy.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using strata::spec::DraftPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
double cost(int t) { return 19.0 + 10.5 * (t - 1); }   // ms per round, measured shape
}  // namespace

int main() {
    std::printf("draft_policy_test\n");
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));   // MTP windows of 4: 3 tokens each
        const DraftPolicy::Pick k = p.choose(4, 0, 0);
        check(!k.lookup && k.t == 4, "no proposal: the MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));
        for (int t = 2; t <= 6; ++t) p.observe(false, t, 0, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 0, 4, cost(6));      // short matches, all rejected
        check(p.lookup_rate(4) < 0.15, "rejected short-match drafts: their rate falls below 0.15");
        check(!p.choose(4, 5, 4).lookup, "rejected short-match drafts: no longer taken");
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 5, 30, cost(6));     // long matches, all accepted
        check(p.lookup_rate(30) > 0.9, "accepted long-match drafts: their rate rises above 0.9");
        const DraftPolicy::Pick k = p.choose(4, 5, 30);
        check(k.lookup && k.t == 6, "accepted long matches: the full lookup window is taken");
        check(!p.choose(4, 5, 4).lookup, "buckets are separate: short matches still not taken");
        check(p.choose(4, 20, 30).t <= 6, "never beyond the window cap");
    }
    {
        DraftPolicy p(8);
        for (int i = 0; i < 50; ++i) p.observe(false, 3, 2, 0, cost(3));      // a very good MTP: 3 of 3 tokens
        for (int t = 2; t <= 8; ++t) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 4, 2, 8, cost(4));       // lookup at q ~ 0.67
        check(!p.choose(3, 7, 8).lookup, "a mediocre lookup does not replace a strong MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a near-perfect MTP, only size 4 seen
        const DraftPolicy::Pick k = p.choose(4, 5, 40);
        check(k.lookup && k.t == 6, "an unmeasured size is probed for a confident lookup");
        for (int i = 0; i < 3; ++i) p.observe(true, 6, 5, 40, 3.0 * cost(6));   // it turns out very expensive
        check(!p.choose(4, 5, 40).lookup, "after the probes, the measured cost decides");
    }
    {
        // #1252: a wide size priced high in one slow stretch is measured again after a while
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a strong MTP
        for (int t = 2; t <= 6; ++t) for (int i = 0; i < 4; ++i) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 6; ++i) p.observe(true, 6, 5, 40, 4.0 * cost(6));  // the wide size, in a slow stretch
        for (int i = 0; i < 30; ++i) p.observe(true, 4, 3, 40, cost(4));       // and the short window's rate is fine
        bool saw6 = false;
        for (int i = 0; i < 400 && !saw6; ++i) {
            const DraftPolicy::Pick k = p.choose(4, 5, 40);
            if (k.lookup && k.t == 6) { saw6 = true; p.observe(true, 6, 5, 40, cost(6)); }
            else p.observe(k.lookup, k.t, k.t - 1, 40, cost(k.t));
        }
        check(saw6, "a stale wide size is tried again within a few hundred rounds");
    }
    {
        // --lookup-chain: rows cheap (UMA-like) and the chain always accepted -> chained; rows dear or the chain
        // always rejected -> not
        auto cheap = [](int t) { return 40.0 + 2.0 * (t - 1); };
        DraftPolicy p(8);
        for (int t = 2; t <= 8; ++t)
            for (int i = 0; i < 5; ++i) p.observe(false, t, 1, 0, cheap(t));
        for (int i = 0; i < 30; ++i) p.observe(false, 4, 2, 0, cheap(4));
        for (int i = 0; i < 30; ++i) p.observe_chain(4, 3, 6, 12, cheap(7));    // every chained token accepted
        check(p.chain(4, 0.9, 3, 12) == 3, "chain: cheap rows, always accepted -> all 3 chained");
        DraftPolicy q(8);
        for (int t = 2; t <= 8; ++t)
            for (int i = 0; i < 5; ++i) q.observe(false, t, 1, 0, cost(t));
        for (int i = 0; i < 30; ++i) q.observe_chain(4, 3, 3, 4, cost(7));       // reached, never accepted
        check(q.chain(4, 0.9, 3, 4) == 0, "chain: dear rows, never accepted -> none");
        check(q.chain(4, 0.9, 0, 4) == 0 && q.chain(8, 1.0, 3, 40) == 0, "chain: nothing proposed / no room -> none");
    }
    {
        // The stale-size check must also run when a shorter chain already pays. Synthetic costs isolate the
        // selection policy: even a size that stays expensive is retried, but only once per stale interval.
        DraftPolicy p(6);
        for (int i = 0; i < 3; ++i) p.observe(false, 4, 3, 0, 40.0);
        for (int i = 0; i < 3; ++i) p.observe_chain(4, 1, 4, 40, 42.0);
        for (int i = 0; i < 3; ++i) p.observe_chain(4, 2, 5, 40, 1000.0);
        check(p.chain(4, 1.0, 2, 40) == 1, "chain: the profitable shorter chain initially wins");
        int probes = 0, last_probe = -1;
        bool spaced = true, confident = true;
        for (int i = 0; i < 1000; ++i) {
            const int k = p.chain(4, 1.0, 2, 40);
            if (k == 2) {
                ++probes;
                spaced &= last_probe < 0 ? i >= 300 : i - last_probe > 300;
                last_probe = i;
                confident &= p.chain(4, 0.5, 2, 40) < 2;
            }
            p.observe_chain(4, k, 3 + k, 40, k == 2 ? 1000.0 : 42.0);
        }
        check(probes > 0 && probes <= 3 && spaced,
              "chain: stale full size is re-probed despite a profitable shorter chain");
        check(probes > 0 && confident, "chain: stale re-probes retain the MTP confidence gate");
    }
    {
        // Gated proposals: within one generation the policy copy learns from committed outcomes only. A size
        // whose cost is still guessed is probed until kProbes rounds of it have run, timed or not, not every round.
        DraftPolicy live(6);
        for (int i = 0; i < 50; ++i) live.observe(false, 4, 3, 0, cost(4));      // a strong MTP
        live.observe(true, 6, 5, 8, 3.0 * cost(6));                             // size 6 timed once, dear
        DraftPolicy frozen = live, copy = live;
        int frozen_probes = 0, copy_probes = 0;
        for (int i = 0; i < 20; ++i) {
            const DraftPolicy::Pick f = frozen.choose(4, 5, 8);
            frozen_probes += f.lookup && f.t == 6;
            const DraftPolicy::Pick c = copy.choose(4, 5, 8);
            copy_probes += c.lookup && c.t == 6;
            copy.observe_committed(c.lookup, c.lookup ? c.t : 4, c.lookup ? 5 : 3, 8);
        }
        check(frozen_probes == 20, "gated: a frozen copy probes the guessed size every round");
        check(copy_probes == 2, "gated: committed outcomes end the probes after kProbes rounds of the size");
        check(copy.cost_ms(6) == live.cost_ms(6), "gated: committed outcomes leave the measured costs alone");
    }
    {
        // Gated proposals: a round's time reaches the generation's copy only once the committed position has passed
        // every counter it depends on. Cheap costs from earlier text stop steering long lookups once the dear
        // repetitive-text times are applied; a held time is not a new probe.
        DraftPolicy live(6);
        for (int t = 2; t <= 6; ++t) for (int i = 0; i < 5; ++i) live.observe(false, t, 1, 0, cost(t) * 0.2);   // cheap stretch
        DraftPolicy copy = live;
        check(copy.choose(4, 5, 40).lookup && copy.choose(4, 5, 40).t == 6, "delayed: cheap inherited costs take the full lookup");
        const double before = copy.cost_ms(6);
        copy.observe_committed(true, 6, 5, 40);
        copy.defer_cost(110, 6, 40.0 * cost(6));
        copy.apply_costs(110);
        check(copy.cost_ms(6) == before, "delayed: a time is held while its counters can still be drawn");
        copy.apply_costs(111);
        check(copy.cost_ms(6) > before, "delayed: applied once the committed position passes its last counter");
        for (int i = 0; i < 40; ++i) {
            copy.observe_committed(true, 6, 5, 40);
            copy.defer_cost(120 + i, 6, 40.0 * cost(6));
            copy.apply_costs(121 + i);
        }
        check(copy.choose(4, 5, 40).t < 6, "delayed: dear measured times shorten the lookup window");
        DraftPolicy fresh(6);
        fresh.observe_committed(true, 6, 5, 40);
        fresh.defer_cost(5, 6, 30.0);
        fresh.apply_costs(6);
        fresh.observe_committed(true, 6, 5, 40);
        check(fresh.choose(4, 5, 40).lookup && fresh.choose(4, 5, 40).t == 6,
              "delayed: a held time does not count as another probe");
    }
    {
        // --lookup-max-t: only the allowed sizes are taken; a proposal between two sizes takes the smaller; a size
        // disabled (its graph found no VRAM) is never proposed again; sizes past 8 work like the others
        DraftPolicy p(16);
        p.set_sizes((1u << 1) | (1u << 2) | (1u << 3) | (1u << 5) | (1u << 8) | (1u << 16));
        for (int i = 0; i < 50; ++i) p.observe(false, 3, 2, 0, cost(3));
        for (int i = 0; i < 40; ++i) p.observe(true, 16, 15, 40, 19.0 + 3.0 * 15);   // long windows: cheap, all accepted
        for (int i = 0; i < 40; ++i) p.observe(true, 8, 7, 40, 19.0 + 3.0 * 7);
        for (int i = 0; i < 40; ++i) p.observe(true, 5, 4, 40, 19.0 + 3.0 * 4);
        const DraftPolicy::Pick a = p.choose(3, 15, 40);
        check(a.lookup && a.t == 16, "sizes: a full 16-token lookup window when it pays");
        const DraftPolicy::Pick b = p.choose(3, 10, 40);
        check(b.lookup && b.t == 8, "sizes: a 10-token proposal takes the 8-token size");
        bool ok = true;
        for (int k = 1; k <= 15; ++k) {
            const DraftPolicy::Pick c = p.choose(3, k, 40);
            ok &= !c.lookup || p.allowed(c.t);
        }
        check(ok, "sizes: never a size outside the mask");
        p.disable(16);
        const DraftPolicy::Pick d = p.choose(3, 15, 40);
        check(d.lookup && d.t == 8, "sizes: a disabled size falls back to the next one");
        check(p.fit(7) == 5 && p.fit(16) == 8 && p.fit(1) == 1, "sizes: fit() finds the largest allowed size");
        DraftPolicy q(16);
        q.set_sizes((1u << 1) | (1u << 2) | (1u << 3) | (1u << 8));
        bool cok = true;
        for (int k = 0; k <= 13; ++k) {
            const int n = q.chain(3, 1.0, k, 40);
            cok &= n == 0 || q.allowed(3 + n);
        }
        check(cok, "sizes: a chained window's total is an allowed size");
    }
    {
        // Fixed costs: measured round times never reach the costs, so two runs that see the same tokens choose the same windows
        // whatever their timing; the same policy without a table follows the time.
        const double table[6] = {19.0, 29.5, 40.0, 50.5, 61.0, 71.5};
        DraftPolicy a(6), b(6), live1(6), live2(6);
        a.set_fixed_costs(table, 6);
        b.set_fixed_costs(table, 6);
        check(a.fixed() && !live1.fixed(), "fixed: only the policy given a table is fixed");
        bool same = true;
        for (int i = 0; i < 400; ++i) {
            const int t = 2 + i % 5;
            const bool lk = i % 3 == 0;
            const int acc = (i * 7) % t;
            const int match = lk ? 12 + (i % 30) : 0;
            a.observe(lk, t, acc, match, cost(t));
            b.observe(lk, t, acc, match, cost(t) * (i % 4 ? 0.3 : 5.0));   // another run's times, far off and jittering
            live1.observe(lk, t, acc, match, cost(t));
            live2.observe(lk, t, acc, match, cost(t) * (i % 4 ? 0.3 : 5.0));
            const DraftPolicy::Pick pa = a.choose(4, 5, 12 + i % 30), pb = b.choose(4, 5, 12 + i % 30);
            same &= pa.lookup == pb.lookup && pa.t == pb.t;
            same &= a.chain(4, 0.8, 3, 20) == b.chain(4, 0.8, 3, 20);
            DraftPolicy::ChainState sa, sb;
            same &= a.keep_draft(sa, 0, 0.7f, 6) == b.keep_draft(sb, 0, 0.7f, 6) && sa.kept == sb.kept;
        }
        check(same, "fixed: different round times, the same windows, chains and kept drafts");
        check(a.cost_ms(3) == table[2] && b.cost_ms(3) == table[2], "fixed: the table is never replaced by a measurement");
        check(live1.cost_ms(3) != live2.cost_ms(3), "no table: the measured costs follow the round times");
        DraftPolicy f(6);                                           // held times (gated proposals) are dropped too
        f.set_fixed_costs(table, 6);
        f.defer_cost(5, 4, 999.0);
        f.apply_costs(100);
        check(f.cost_ms(4) == table[3], "fixed: defer_cost / apply_costs leave the table alone");
        DraftPolicy g(6), h(6);                                     // only size 4 seen: the adaptive policy probes size 6, the fixed one does not
        for (int i = 0; i < 50; ++i) { g.observe(false, 4, 3, 0, cost(4)); h.observe(false, 4, 3, 0, cost(4)); }
        g.set_fixed_costs(nullptr, 0);                              // fixed where it stands (no table): what was measured stays
        check(h.choose(4, 5, 40).lookup && h.choose(4, 5, 40).t == 6, "no table: the unmeasured size is probed (as before)");
        check(!g.choose(4, 5, 40).lookup, "fixed: no first-use probe; the table decides");
        DraftPolicy m(6);                                           // a table set after measuring replaces the EMA
        for (int i = 0; i < 30; ++i) m.observe(false, 4, 3, 0, 80.0);
        m.set_fixed_costs(table, 6);
        check(m.cost_ms(4) == table[3], "fixed: set after measuring, the table replaces the measured cost");
    }
    {
        // CostLearner: plain trimmed means per size, interpolation, readiness and the disturbance checks.
        using strata::spec::CostLearner;
        CostLearner L;
        for (int i = 0; i < 200; ++i) {
            L.add(0, 1, 10.0 + (i % 3));                            // 10, 11, 12: mean 11
            L.add(0, 4, i == 7 ? 400.0 : 20.0);                     // one capture-sized outlier: trimmed away
            if (i < 40) L.add(0, 6, 30.0);
        }
        const std::vector<double> t = L.table(0, 20);
        check(t.size() == 6, "learner: the table runs to the largest size seen");
        check(t.size() == 6 && t[0] > 10.9 && t[0] < 11.1 && t[3] == 20.0 && t[5] == 30.0, "learner: trimmed means of the measured sizes");
        check(t.size() == 6 && t[4] == 25.0, "learner: an unseen size between two measured ones is linear in the size");
        check(t.size() == 6 && t[1] > 11.0 && t[1] < t[3] && t[2] > t[1] && t[2] < t[3], "learner: sizes between are ordered");
        check(L.table(1, 20).empty(), "learner: a bucket with no rounds has no table");
        check(!L.check(1000, 20).ready && L.check(400, 20).ready, "learner: ready only after the given number of rounds");
        check(!L.check(400, 20).disturbed, "learner: a clean stretch is not disturbed");
        CostLearner D;                                              // the second half twice as slow: a disturbed stretch
        for (int i = 0; i < 200; ++i) D.add(0, 4, i < 100 ? 20.0 : 40.0);
        const CostLearner::Verdict dv = D.check(100, 20);
        check(dv.disturbed && !dv.why.empty(), "learner: a drift between the halves is a disturbance");
        CostLearner S;                                              // 10 % of rounds stalled at 5x
        for (int i = 0; i < 200; ++i) S.add(0, 4, i % 10 == 3 ? 100.0 : 20.0);
        check(S.check(100, 20).disturbed, "learner: many spikes are a disturbance");
        CostLearner O;                                              // a wider window measured cheaper than a narrower one
        for (int i = 0; i < 50; ++i) { O.add(0, 2, 30.0); O.add(0, 4, 20.0); }
        check(O.check(50, 20).disturbed, "learner: a wider window cheaper than a narrower one is a disturbance");
        DraftPolicy p(6);                                           // the policy feeds the learner only while learning
        p.observe(false, 4, 3, 0, 20.0);
        check(p.learner().rounds() == 0, "learning off: nothing is collected");
        p.set_learning(true);
        p.observe(false, 4, 3, 0, 20.0);
        p.observe_chain(3, 1, 2, 20, 21.0);
        p.defer_cost(5, 2, 15.0);
        p.apply_costs(100);
        check(p.learner().rounds() == 3 && p.learner().samples(0, 4) == 2 && p.learner().samples(0, 2) == 1,
              "learning on: observe, observe_chain and held times are collected");
    }
    {
        // The fixture of the Mac lane's fresh learning run (M5 Max, Q2_0, 30 requests of 256 tokens): sample counts and mean round ms per window size.
        // 4 rows had 2 samples: its value comes from its neighbours, not from the two samples.
        using strata::spec::CostLearner;
        const auto fill = [](CostLearner& L, int t, int n, double ms, double age) {   // n samples around ms (+-1 %, mean ms)
            for (int i = 0; i < n; ++i) L.add(0, t, ms * (1.0 + 0.005 * ((i % 5) - 2)), age);
        };
        CostLearner M;
        fill(M, 1, 1193, 13.50, 2.0); fill(M, 2, 894, 15.74, 2.0); fill(M, 3, 1834, 18.40, 2.0); fill(M, 4, 2, 22.62, 2.0); fill(M, 5, 83, 24.95, 2.0);
        const std::vector<double> t = M.table(0, 20);
        check(t.size() == 5 && std::fabs(t[0] - 13.50) < 0.05 && std::fabs(t[2] - 18.40) < 0.05 && std::fabs(t[4] - 24.95) < 0.05, "fit: the well-sampled sizes are their trimmed means");
        check(t.size() == 5 && std::fabs(t[3] - (18.40 + 24.95) / 2.0) < 0.05, "fit: 4 rows (2 samples) lie between its neighbours, not at its own 22.62");
        check(M.rounds() == 4006 && M.late_rounds() == 0, "fit: all 4,006 rounds of the fixture were fresh");
        // a size above the measured ones (5 rows with 8 samples): the straight line through the two largest measured sizes
        CostLearner E;
        fill(E, 1, 600, 12.5, 2.0); fill(E, 2, 400, 14.2, 2.0); fill(E, 3, 700, 16.2, 2.0); fill(E, 5, 8, 99.0, 2.0);
        const std::vector<double> e = E.table(0, 20);
        check(e.size() == 5 && std::fabs(e[3] - (16.2 + 2.0)) < 0.05 && std::fabs(e[4] - (16.2 + 4.0)) < 0.05,
              "fit: 4 and 5 rows above the measured sizes continue the line through 2 and 3 rows (8 samples of 5 rows are not trusted)");
        bool mono = true;
        for (size_t i = 1; i < e.size(); ++i) mono &= e[i] >= e[i - 1];
        CostLearner Dn;                                                       // a size measured cheaper than the one before it (noise): the table does not fall
        fill(Dn, 1, 100, 20.0, 2.0); fill(Dn, 2, 100, 19.0, 2.0); fill(Dn, 3, 100, 25.0, 2.0);
        const std::vector<double> d = Dn.table(0, 20);
        mono &= d.size() == 3 && d[1] >= d[0] && d[2] >= d[1];
        check(mono, "fit: the table never falls from one size to the next");
        // busy age: a fast state for the first seconds of a run, a slow and steeper one later; only the first is kept
        CostLearner B;
        for (int i = 0; i < 1400; ++i) {
            B.add(0, 1, 12.5, 4.0); B.add(0, 2, 14.2, 4.0); B.add(0, 3, 16.2, 4.0);               // fresh (the first seconds of a run)
            B.add(0, 1, 14.5, 60.0); B.add(0, 2, 17.1, 60.0); B.add(0, 3, 20.2, 60.0);            // the same machine after a minute of load
            if (i % 25 == 0) { B.add(0, 5, 21.3, 4.0); B.add(0, 5, 27.7, 60.0); }
        }
        const std::vector<double> b = B.table(0, 20);
        check(B.rounds() == 4200 + 56 && B.late_rounds() == 4200 + 56, "busy age: late rounds are counted and not kept");
        check(b.size() == 5 && std::fabs(b[2] - 16.2) < 0.05 && std::fabs(b[4] / b[2] - 21.3 / 16.2) < 0.01,
              "busy age: the table is the fast state (5 rows against 3 rows 1.31, not 1.37 of the late rounds)");
        CostLearner U;                                                        // an age that is not known counts as fresh
        U.add(0, 3, 16.0);
        check(U.rounds() == 1 && U.late_rounds() == 0, "busy age: an unknown age (none given) is kept");
        // readiness: a stretch that never paused ends with its first seconds after 3 x the quota; one that is mid-way is not ready
        CostLearner Bur;
        for (int i = 0; i < 300; ++i) { Bur.add(0, 1, 12.5, 2.0); Bur.add(0, 2, 14.2, 2.0); Bur.add(0, 3, 16.2, 2.0); }    // 900 fresh rounds
        check(!Bur.check(1000, 20).ready, "ready: 900 fresh rounds and nothing else is not ready");
        for (int i = 0; i < 700; ++i) Bur.add(0, 3, 20.2, 90.0);
        check(!Bur.check(1000, 20).ready, "ready: late rounds below 3 x the quota do not end a short stretch");
        CostLearner Bur2;                                                     // one long burst: 330 fresh rounds, then 2,700 late ones
        for (int i = 0; i < 110; ++i) { Bur2.add(0, 1, 12.5, 2.0); Bur2.add(0, 2, 14.2, 2.0); Bur2.add(0, 3, 16.2, 2.0); }
        for (int i = 0; i < 2000; ++i) Bur2.add(0, 3, 20.2, 90.0);
        check(!Bur2.check(1000, 20).ready, "ready: 330 fresh + 2,000 late rounds (under 3 x the quota) is not ready");
        for (int i = 0; i < 700; ++i) Bur2.add(0, 3, 20.2, 90.0);
        const CostLearner::Verdict bv = Bur2.check(1000, 20);
        const std::vector<double> bt = Bur2.table(0, 20);
        check(bv.ready && !bv.disturbed && bt.size() == 3 && std::fabs(bt[2] - 16.2) < 0.05,
              "ready: one burst that never paused ends after 3 x the quota, with the table of its first seconds");
        CostLearner Few;                                                      // 150 fresh rounds only: not enough even after a very long burst
        for (int i = 0; i < 50; ++i) { Few.add(0, 1, 12.5, 2.0); Few.add(0, 2, 14.2, 2.0); Few.add(0, 3, 16.2, 2.0); }
        for (int i = 0; i < 5000; ++i) Few.add(0, 3, 20.2, 90.0);
        check(!Few.check(1000, 20).ready, "ready: under a fifth of the quota kept is never ready");
        // a machine without a state effect (the Radeon 780M: 75.5 ms for one row early, 76.0 late; windows of 1, 3 and 6 rows): every round counts, the rare sizes
        // get all their samples (3 rows: 8 fresh + 100 late)
        CostLearner R;
        fill(R, 1, 80, 75.5, 2.0); fill(R, 3, 8, 111.0, 2.0); fill(R, 6, 3, 178.0, 2.0);
        fill(R, 1, 1200, 76.0, 40.0); fill(R, 3, 100, 111.2, 40.0); fill(R, 6, 14, 178.1, 40.0);
        const std::vector<double> rt = R.table(0, 20);
        check(!R.state_effect(0, 20), "pooled: 0.7 % between early and late rounds is no state effect");
        check(rt.size() == 6 && std::fabs(rt[2] - 111.1) < 0.2 && std::fabs(rt[0] - 76.0) < 0.2, "pooled: the table uses every round (3 rows from 108 samples)");
        check(R.check(1000, 20).ready, "pooled: ready after the quota of rounds of any age");
        check(B.state_effect(0, 20), "state effect: 16 % between early and late rounds is one");
        // the policy hands the age to the learner, and a held (gated) time keeps the age it had
        DraftPolicy pa(6);
        pa.set_learning(true);
        pa.set_busy_age(30.0);
        pa.observe(false, 3, 1, 0, 20.0);
        pa.set_busy_age(2.0);
        pa.observe(false, 3, 1, 0, 16.0);
        check(pa.learner().rounds() == 1 && pa.learner().late_rounds() == 1 && pa.learner().samples(0, 3) == 1, "policy: a round is kept by the busy age it ran at");
        pa.set_busy_age(40.0);
        pa.defer_cost(5, 3, 21.0);
        pa.set_busy_age(1.0);
        pa.apply_costs(100);
        check(pa.learner().rounds() == 1 && pa.learner().late_rounds() == 2, "policy: a held time keeps the busy age of its round");
    }
    {
        // The table file: a round trip, another key, a cut-short file, a missing file, a folder that does not exist yet.
        namespace fs = std::filesystem;
        using strata::spec::CostFile;
        std::error_code ec;
        const fs::path dir = fs::temp_directory_path(ec) / ("strata-dpt-" + std::to_string((unsigned long long) (uintptr_t) &ec));
        const std::string path = (dir / "sub" / "t.costs").string();
        CostFile c;
        c.key = "card A|engine 1|model m\nspec 6";
        c.tables[0] = {15.18, 18.55, 20.69, 23.65};
        c.tables[2] = {16.0, 19.0};
        std::string why;
        check(strata::spec::write_cost_file(path, c, why), "file: written, its folder made");
        CostFile r;
        const bool ok = strata::spec::read_cost_file(path, c.key, r, why);
        check(ok && r.tables[0].size() == 4 && r.tables[0][3] == 23.65 && r.tables[1].empty() && r.tables[2].size() == 2,
              "file: read back with its key, every bucket as written");
        check(!strata::spec::read_cost_file(path, "card B|engine 1|model m spec 6", r, why) && !why.empty(), "file: another key is not loaded");
        check(!strata::spec::read_cost_file((dir / "none.costs").string(), c.key, r, why), "file: a missing file is not loaded");
        { std::ofstream f(path, std::ios::binary | std::ios::trunc); f << "strata-draft-costs 2\nkey x\nbucket 0 15.1,18"; }
        check(!strata::spec::read_cost_file(path, "x", r, why), "file: a cut-short file (no end line) is not loaded");
        { std::ofstream f(path, std::ios::binary | std::ios::trunc); f << "strata-draft-costs 2\nkey x\nbucket 0 15.1,abc\nend\n"; }
        check(!strata::spec::read_cost_file(path, "x", r, why), "file: a malformed table is not loaded");
        check(strata::spec::write_cost_file(path, c, why) && strata::spec::read_cost_file(path, c.key, r, why), "file: a bad file is replaced whole");
        check(!fs::exists(fs::path(path + ".tmp"), ec), "file: no temporary file is left");
        check(r.cpu_share < 0.0, "file: no share written, none read");
        c.cpu_share = 0.45;
        check(strata::spec::write_cost_file(path, c, why) && strata::spec::read_cost_file(path, c.key, r, why) && r.cpu_share == 0.45,
              "file: the CPU prompt share is stored and read back with the table");
        { std::ofstream f(path, std::ios::binary | std::ios::trunc); f << "strata-draft-costs 2\nkey x\nbucket 0 15.1,18\nshare 1.7\nend\n"; }
        check(!strata::spec::read_cost_file(path, "x", r, why), "file: a share outside 0..1 is not loaded");
        // the decode PCIe share PcieAuto settled on (repeatable output): stored and read back; a file written before it has none, and
        // still loads (the engine then refines the share as before and writes it)
        { std::ofstream f(path, std::ios::binary | std::ios::trunc); f << "strata-draft-costs 2\nkey x\nbucket 0 15.1,18\nshare 0.900\nend\n"; }
        check(strata::spec::read_cost_file(path, "x", r, why) && r.cpu_share == 0.9 && r.pcie_frac < 0.0,
              "file: a file from before the PCIe share loads, with no PCIe share");
        c.pcie_frac = 0.15;
        check(strata::spec::write_cost_file(path, c, why) && strata::spec::read_cost_file(path, c.key, r, why) && r.pcie_frac == 0.15 &&
                  r.cpu_share == 0.45 && r.tables[0].size() == 4,
              "file: the PCIe share is stored and read back with the table and the prompt share");
        { std::ofstream f(path, std::ios::binary | std::ios::trunc); f << "strata-draft-costs 2\nkey x\nbucket 0 15.1,18\npcie -0.2\nend\n"; }
        check(!strata::spec::read_cost_file(path, "x", r, why), "file: a PCIe share outside 0..1 is not loaded");
        { std::ofstream f(path, std::ios::binary | std::ios::trunc); f << "strata-draft-costs 2\nkey x\nbucket 0 15.1,18\npcie 0.15x\nend\n"; }
        check(!strata::spec::read_cost_file(path, "x", r, why), "file: a malformed PCIe share is not loaded");
        // version 1 (learned from every round, so from the slow state of a long burst): not loaded, learned again
        { std::ofstream f(path, std::ios::binary | std::ios::trunc); f << "strata-draft-costs 1\nkey x\nbucket 0 15.1,18\nend\n"; }
        check(!strata::spec::read_cost_file(path, "x", r, why) && why.find("format") != std::string::npos, "file: a version 1 table is not loaded (learned again)");
        // a start closed in the middle of learning leaves no table: nothing is written before the stretch is ready, and a half-written temporary file
        // (the process died while writing) is never read as the table
        fs::remove(fs::path(path), ec);
        { std::ofstream f(path + ".tmp", std::ios::binary | std::ios::trunc); f << "strata-draft-costs 2\nkey x\nbucket 0 15.1,18"; }
        check(!strata::spec::read_cost_file(path, "x", r, why), "file: a half-written temporary file beside a missing table is not a table");
        check(strata::spec::write_cost_file(path, c, why) && strata::spec::read_cost_file(path, c.key, r, why) && !fs::exists(fs::path(path + ".tmp"), ec),
              "file: the next write replaces the leftover temporary file and the table loads");
        fs::remove_all(dir, ec);
    }
    {
        // sizes with too few samples: a rough measurement counts when it is not below the trend, a stalled handful does not, and a fill above the
        // measured sizes never flattens the measured trend (Radeon 780M, 11-13: sizes 1 / 3 measured at 76.2 / 111.6 ms, size 4 with 6 rounds at 143)
        using strata::spec::CostLearner;
        const auto fill = [](CostLearner& L, int t, int n, double ms) {
            for (int i = 0; i < n; ++i) L.add(0, t, ms * (1.0 + 0.004 * ((i % 5) - 2)));
        };
        CostLearner R;
        fill(R, 1, 1279, 76.2); fill(R, 2, 2, 97.0); fill(R, 3, 126, 111.6); fill(R, 4, 6, 143.0);
        const std::vector<double> t = R.table(0, 20);
        check(t.size() == 4 && std::fabs(t[3] - 143.0) < 0.5, "sizes: 6 rounds of 4 rows (143 ms, above the line's 129) count by their median");
        check(t.size() == 4 && std::fabs(t[1] - (76.2 + 111.6) / 2.0) < 0.2, "sizes: 2 rounds of 2 rows are filled in linearly");
        const std::vector<double> u = R.table(0, 20, 6);
        bool mono = u.size() == 6;
        for (size_t i = 1; i < u.size(); ++i) mono &= u[i] >= u[i - 1];
        check(u.size() == 6 && mono && u[4] >= 143.0 + 0.9 * (143.0 - 111.6) && u[5] > u[4], "sizes: 5 and 6 rows continue the steeper measured slope (32 ms a row), not the first segment's 18");
        CostLearner S;
        fill(S, 1, 600, 12.5); fill(S, 2, 400, 14.2); fill(S, 3, 700, 16.2); fill(S, 4, 6, 99.0);
        const std::vector<double> s = S.table(0, 20);
        check(s.size() == 4 && std::fabs(s[3] - 18.2) < 0.05, "sizes: a handful of rounds 5x above the trend (a stall) is not a measurement");
        CostLearner Q;                                              // costs 10 / 20 / 35: the fill goes on at 15 ms a row, not at 10
        fill(Q, 1, 100, 10.0); fill(Q, 2, 100, 20.0); fill(Q, 3, 100, 35.0);
        const std::vector<double> q = Q.table(0, 20, 5);
        check(q.size() == 5 && q[3] >= 35.0 + 15.0 - 0.1 && q[4] >= 35.0 + 30.0 - 0.1, "sizes: a size above the measured ones never costs less than the last slope says");
        check(Q.table(0, 20, 3).size() == 3 && Q.table(1, 20, 5).empty(), "sizes: upto adds nothing below the largest size seen, and an empty bucket stays empty");
        // the learning stretch measures the sizes it has too few rounds of
        const auto feed = [](DraftPolicy& p, int n) {
            for (int i = 0; i < n; ++i) { p.observe(false, 1, 0, 0, 76.0); if (i % 4 == 0) p.observe(false, 3, 2, 0, 111.0); }
        };
        DraftPolicy y(8);
        y.set_learning(true);
        feed(y, 40);
        y.choose(1, 5, 3);
        check(y.forced_rounds() == 0, "explore: nothing is forced in the first rounds of a stretch");
        feed(y, 300);
        const DraftPolicy::Pick f1 = y.choose(1, 5, 3);
        check(f1.lookup && f1.t == 2 && y.forced_rounds() == 1, "explore: after the first rounds the smallest size with few samples is taken, if a proposal reaches it");
        const DraftPolicy::Pick f2 = y.choose(1, 1, 3);
        check(y.forced_rounds() == 2 && f2.lookup && f2.t == 2, "explore: a proposal of one draft reaches 2 rows only");
        DraftPolicy y16(16);                                        // the budget: 14 sizes that need 30 rounds each are more than the stretch's budget
        y16.set_learning(true);
        feed(y16, 400);
        for (int i = 0; i < 1200; ++i) {
            const DraftPolicy::Pick pk = y16.choose(1, 15, 3);
            y16.observe(pk.lookup, pk.t, pk.t > 1 ? pk.t - 1 : 0, 3, 76.0 + 17.0 * (pk.t - 1));
        }
        check(y16.forced_rounds() == DraftPolicy::kExploreBudget, "explore: the forced rounds stop at the stretch's budget");
        const int before = y16.forced_rounds();
        for (int i = 0; i < 50; ++i) y16.choose(1, 15, 3);
        check(y16.forced_rounds() == before, "explore: nothing is forced once the budget is spent");
        DraftPolicy z(8);                                           // the cap per size: with a budget left over, one size is still not forced more than kExplorePerSize times
        z.set_learning(true);
        feed(z, 400);
        for (int i = 0; i < 200; ++i) { const DraftPolicy::Pick pk = z.choose(1, 1, 3); z.observe(pk.lookup, pk.t, 0, 3, 0.0); }   // no time, so no sample ever arrives
        check(z.forced_rounds() == DraftPolicy::kExplorePerSize, "explore: a size is forced at most kExplorePerSize times");
        z.reset_learning();
        check(z.forced_rounds() == 0 && z.learner().rounds() == 0, "explore: a new stretch starts with a new budget");
        DraftPolicy n(8);                                           // not learning, or fixed: the policy never takes a window for its size's sake
        feed(n, 400);
        n.choose(1, 5, 3);
        DraftPolicy x(8);
        x.set_learning(true);
        feed(x, 400);
        const double ms[3] = {76.0, 95.0, 111.0};
        x.set_fixed_costs(ms, 3);
        for (int i = 0; i < 20; ++i) x.choose(1, 5, 3);
        check(n.forced_rounds() == 0 && x.forced_rounds() == 0, "explore: not while learning is off, and not once the costs are fixed");
        DraftPolicy w(8);                                           // a size no proposal reaches stays filled in
        w.set_learning(true);
        feed(w, 600);
        for (int i = 0; i < 100; ++i) { const DraftPolicy::Pick pk = w.choose(1, 1, 3); w.observe(pk.lookup, pk.t, 1, 3, 95.0); }
        const std::vector<double> wt = w.learner().table(0, 20, 6);
        check(w.learner().samples(0, 4) == 0 && wt.size() == 6 && wt[3] > wt[2] && wt[5] > wt[3], "explore: a size no proposal reaches is filled in above the measured ones");
        // the window of a round that has no proposal: any undersampled size up to the cap, but not the size it already is
        DraftPolicy m(8);
        m.set_learning(true);
        feed(m, 400);
        check(m.explore_window(3, 6) == 2 && m.forced_rounds() == 1, "window: the smallest unmeasured size is run, not the one the round already has");
        check(m.explore_window(2, 6) == 4, "window: a size with samples (3 rows) is skipped, the next unmeasured one is run");
        check(m.explore_window(1, 1) == 0, "window: a cap of one row leaves nothing to run");
        DraftPolicy m0(8);
        feed(m0, 400);
        check(m0.explore_window(3, 6) == 0, "window: nothing while learning is off");
        DraftPolicy lk(8);                                          // a round with no proposal at all: a lookup-type window of an unmeasured size
        lk.set_learning(true);
        feed(lk, 400);
        check(lk.explore_lookup(5) == 2 && lk.explore_lookup(1) == 0 && m0.explore_lookup(5) == 0, "window: a round without a proposal can time a lookup-type size, only while learning");
        // a late round counts only where the stretch has shown no state effect
        DraftPolicy e1(8);                                          // fresh 76 ms, late 80 ms: a state effect
        e1.set_learning(true);
        e1.set_busy_age(2.0);
        for (int i = 0; i < 60; ++i) e1.observe(false, 1, 0, 0, 76.0);
        e1.set_busy_age(100.0);
        for (int i = 0; i < 60; ++i) e1.observe(false, 1, 0, 0, 80.0);
        check(e1.explore_window(3, 6) == 0, "gate: a late round is not forced when the stretch has a state effect");
        e1.set_busy_age(2.0);
        check(e1.explore_window(3, 6) == 2, "gate: a fresh round is");
        DraftPolicy e2(8);                                          // fresh and late alike: no state effect
        e2.set_learning(true);
        e2.set_busy_age(2.0);
        for (int i = 0; i < 60; ++i) e2.observe(false, 1, 0, 0, 76.0);
        e2.set_busy_age(100.0);
        for (int i = 0; i < 60; ++i) e2.observe(false, 1, 0, 0, 76.2);
        check(e2.explore_window(3, 6) == 2, "gate: a late round is forced when the stretch shows no state effect");
        DraftPolicy e3(8);                                          // late rounds only so far: unknown, so not forced
        e3.set_learning(true);
        e3.set_busy_age(100.0);
        for (int i = 0; i < 150; ++i) e3.observe(false, 1, 0, 0, 76.0);
        check(e3.explore_window(3, 6) == 0, "gate: a late round is not forced while the stretch cannot tell yet");
        // a padded window: a short proposal, a size past it that has no samples, the caller allows filler rows
        DraftPolicy g(8);
        g.set_learning(true);
        feed(g, 400);
        for (int i = 0; i < 40; ++i) g.observe_forced(2, 95.0);     // size 2 measured (3 is fed already)
        const DraftPolicy::Pick np = g.choose(1, 1, 3);
        check(g.forced_rounds() == 0 || !np.lookup || np.pad == 0, "pad: without pad_cap a proposal of one draft never makes a longer window");
        const DraftPolicy::Pick gp = g.choose(1, 1, 3, 7);
        check(gp.lookup && gp.t == 4 && gp.pad == 2 && g.forced_rounds() == 1, "pad: with pad_cap the smallest unmeasured size is run, 2 filler rows past the proposal");
        const double q0 = g.lookup_rate(3);
        for (int i = 0; i < 30; ++i) g.observe_forced(4, 143.0);
        check(g.lookup_rate(3) == q0 && g.learner().samples(0, 4) == 30, "pad: a forced round feeds the learner and no lookup statistic");
        const DraftPolicy::Pick g5 = g.choose(1, 1, 3, 7);
        check(g5.lookup && g5.t == 5 && g5.pad == 3, "pad: the next unmeasured size follows once one has its samples");
        check(g.choose(1, 1, 3, 1).pad == 0, "pad: a pad_cap of one draft makes no filler");
    }
    {   // DRAFT-COSTS-5: a real price is the MEDIAN of its 16 rounds; the monotone guard repairs an inversion from the trusted side
        namespace fs = std::filesystem;
        using strata::spec::CalAcc;
        using strata::spec::CostFile;
        using strata::spec::CostLearner;
        using Cal = std::array<std::array<CalAcc, CostLearner::kMaxT + 1>, CostLearner::kCtxBuckets>;
        using Rc = std::array<std::array<double, CostLearner::kMaxT + 1>, CostLearner::kCtxBuckets>;
        using Real = std::array<unsigned, CostLearner::kCtxBuckets>;
        const double tab[5] = {12.68, 14.30, 16.41, 18.57, 20.90};
        const Cal cal0{};
        const Rc rc0{};
        {   // one cold round among 16 (the first use of a width): the real price is the median, a mean would be moved by it
            DraftPolicy m(8);
            m.set_fixed_costs(tab, 5, 0);
            m.start_calibration(false, Real{}, cal0, rc0);
            for (int i = 0; i < 15; ++i) m.observe(false, 2, 1, 0, 14.5);
            m.observe(false, 2, 1, 0, 40.0);
            CostFile cf;
            std::string rep;
            check(m.calibrate(cf, rep) && (cf.real[0] & 2u), "median: the 16th round marks the size real");
            check(std::fabs(cf.real_cost[0][2] - 14.5) < 1e-9, "median: one cold round among 16 does not move the real price");
        }
        {   // eight slow rounds in the first request, eight normal ones after: the median sits between (even count: mean of the two middle rounds)
            DraftPolicy m(8);
            m.set_fixed_costs(tab, 5, 0);
            m.start_calibration(false, Real{}, cal0, rc0);
            for (int i = 0; i < 8; ++i) m.observe(false, 2, 1, 0, 15.0);
            for (int i = 0; i < 8; ++i) m.observe(false, 2, 1, 0, 14.0);
            CostFile cf;
            std::string rep;
            check(m.calibrate(cf, rep) && std::fabs(cf.real_cost[0][2] - 14.5) < 1e-9, "median: sixteen rounds of two values: the mean of the two middle ones");
        }
        {   // the rounds survive the file (a start collects part of them, the next start the rest) and a file line without them still loads (mean as before)
            const fs::path dir = fs::temp_directory_path() / ("strata-dpt-med-" + std::to_string((unsigned long long) (uintptr_t) &tab));
            const std::string path = (dir / "m.costs").string();
            CostFile c;
            c.key = "k";
            c.tables[0] = {12.68, 14.30, 16.41};
            c.cal[0][2] = CalAcc{};
            c.cal[0][2].n = 3;
            c.cal[0][2].sum = 43.5;
            c.cal[0][2].nv = 3;
            c.cal[0][2].vals = {14.5, 14.0, 15.0};
            c.cal[0][3].n = 2;   // no rounds kept (a file of an older build): only the sum
            c.cal[0][3].sum = 33.0;
            std::string why;
            check(strata::spec::write_cost_file(path, c, why), "median file: written");
            CostFile r;
            check(strata::spec::read_cost_file(path, c.key, r, why), "median file: read back");
            check(r.cal[0][2].n == 3 && r.cal[0][2].nv == 3 && std::fabs((double) r.cal[0][2].vals[1] - 14.0) < 1e-6 && std::fabs(r.cal[0][2].sum - 43.5) < 1e-3,
                  "median file: the kept rounds come back with their count and sum");
            check(r.cal[0][3].n == 2 && r.cal[0][3].nv == 0 && std::fabs(r.cal[0][3].sum - 33.0) < 1e-3, "median file: a line without rounds keeps only n and sum (the mean decides)");
            std::error_code ec;
            fs::remove_all(dir, ec);
        }
        {   // fixture 1: the srv118 table as taken over (a real 2-row price of 20.08 ms, the sizes above it real or unreached): the outlier is lowered, 3-6 rows keep their prices
            const double t12[12] = {7.928, 20.085, 9.577, 10.524, 12.136, 13.270, 19.223, 21.293, 23.659, 26.024, 28.390, 30.756};
            Real real0{};
            real0[0] = 63u;
            Cal cal1{};
            Rc rc1{};
            const double rcv[7] = {0.0, 7.9276, 20.0847, 9.5771, 10.5245, 12.1355, 13.2695};
            for (int t = 1; t <= 6; ++t) rc1[0][(size_t) t] = rcv[t];
            DraftPolicy p(12);
            p.set_fixed_costs(t12, 12, 0);
            p.start_calibration(false, real0, cal1, rc1, 100);
            bool mono = true;
            for (int t = 2; t <= 12; ++t) mono = mono && p.cost_ms(t) >= p.cost_ms(t - 1) - 1e-12;
            check(mono, "guard: the used table never costs less for more rows");
            check(std::fabs(p.cost_ms(2) - 9.5771) < 1e-9 && std::fabs(p.cost_ms(3) - 9.5771) < 1e-9 && std::fabs(p.cost_ms(4) - 10.5245) < 1e-9 && std::fabs(p.cost_ms(6) - 13.2695) < 1e-9,
                  "guard: the 20 ms 2-row outlier is lowered to the next real price, not raised into every larger size");
            check(std::fabs(p.cost_ms(1) - 7.9276) < 1e-9, "guard: the 1-row price is untouched");
        }
        {   // fixture 2: the Q2_0 table at take-over: 4 rows an unreached live value above the real 5 rows
            const double q[5] = {12.966, 14.482, 17.042, 21.796, 21.361};
            Real real0{};
            real0[0] = 23u;   // sizes 1, 2, 3, 5
            Cal cal1{};
            Rc rc1{};
            rc1[0][1] = 12.9665;
            rc1[0][2] = 14.4821;
            rc1[0][3] = 17.0419;
            rc1[0][5] = 21.3610;
            DraftPolicy p(8);
            p.set_fixed_costs(q, 5, 0);
            p.start_calibration(false, real0, cal1, rc1, 100);
            check(std::fabs(p.cost_ms(4) - 21.3610) < 1e-9 && std::fabs(p.cost_ms(5) - 21.3610) < 1e-9 && std::fabs(p.cost_ms(3) - 17.0419) < 1e-9,
                  "guard: an unreached 4 rows above the real 5 rows is lowered to it");
        }
        {   // fixture 3: the IQ3_S table at take-over: 4 rows 28.31 (a live value) above the real 5 rows 25.00
            const double q[5] = {14.734, 16.967, 20.096, 28.312, 25.005};
            Real real0{};
            real0[0] = 23u;
            Cal cal1{};
            Rc rc1{};
            rc1[0][1] = 14.7341;
            rc1[0][2] = 16.9669;
            rc1[0][3] = 20.0959;
            rc1[0][5] = 25.0046;
            DraftPolicy p(8);
            p.set_fixed_costs(q, 5, 0);
            p.start_calibration(false, real0, cal1, rc1, 100);
            check(std::fabs(p.cost_ms(4) - 25.0046) < 1e-9 && std::fabs(p.cost_ms(5) - 25.0046) < 1e-9 && std::fabs(p.cost_ms(2) - 16.9669) < 1e-9,
                  "guard: IQ3_S 4 rows 28.31 is lowered to the real 5 rows 25.00");
        }
        {   // a real size above an unreached next size is not lowered by the guess: the guess is lifted to it
            const double q[5] = {12.0, 14.0, 20.0, 15.0, 16.0};
            Real real0{};
            real0[0] = 4u;   // size 3 real at 20.0
            Cal cal1{};
            Rc rc1{};
            rc1[0][3] = 20.0;
            DraftPolicy p(8);
            p.set_fixed_costs(q, 5, 0);
            p.start_calibration(false, real0, cal1, rc1, 100);
            check(std::fabs(p.cost_ms(3) - 20.0) < 1e-9 && p.cost_ms(4) >= 20.0 - 1e-9 && p.cost_ms(5) >= p.cost_ms(4) - 1e-12 && std::fabs(p.cost_ms(2) - 14.0) < 1e-9,
                  "guard: a real price is kept against a guessed larger size, which is lifted to it");
        }
    }
    {   // calibration against real rounds
        using strata::spec::CalAcc;
        using strata::spec::CostFile;
        using strata::spec::CostLearner;
        using strata::spec::read_cost_file;
        using strata::spec::write_cost_file;
        using Cal = std::array<std::array<CalAcc, CostLearner::kMaxT + 1>, CostLearner::kCtxBuckets>;
        using Rc = std::array<std::array<double, CostLearner::kMaxT + 1>, CostLearner::kCtxBuckets>;
        using Real = std::array<unsigned, CostLearner::kCtxBuckets>;
        const double tab[5] = {12.68, 14.30, 16.41, 18.57, 20.90};
        const Cal cal0{};
        const Rc rc0{};
        DraftPolicy c(8), live(8);   // `live` has no table: a held table must decide exactly like it
        c.set_fixed_costs(tab, 5, 0);
        c.start_calibration(false, Real{}, cal0, rc0);
        check(c.calibrating() && !c.calib_dirty() && c.table_held() && !c.fixed(), "calib: a table with uncalibrated sizes is held back: live costs, nothing collected yet");
        bool same = true;
        for (int i = 0; i < 15; ++i) {
            const DraftPolicy::Pick pa = c.choose(i % 3 + 1, 5, 2), pb = live.choose(i % 3 + 1, 5, 2);
            same = same && pa.lookup == pb.lookup && pa.t == pb.t && std::fabs(c.cost_ms(3) - live.cost_ms(3)) < 1e-12;
            c.observe(false, 3, 2, 0, 16.70);
            live.observe(false, 3, 2, 0, 16.70);
        }
        CostFile cf;
        std::string rep;
        check(same && std::fabs(c.cost_ms(3) - 16.70) < 1e-9 && std::fabs(c.cost_ms(4) - live.cost_ms(4)) < 1e-12,
              "calib: while held back, picks and costs equal those of a policy without a table (the live cost follows the rounds)");
        check(c.calibrate(cf, rep) && rep.empty() && cf.cal[0][3].n == 15, "calib: 15 real rounds are kept, nothing replaced");
        c.observe(false, 3, 2, 0, 16.70);
        check(c.calibrate(cf, rep) && !rep.empty() && (cf.real[0] & 4u) && std::fabs(cf.real_cost[0][3] - 16.70) < 1e-9 && std::fabs(cf.tables[0][2] - 16.41) < 1e-9,
              "calib: the 16th real round marks the size real; the file keeps the learned cost beside it");
        check(c.table_held() && !c.fixed() && c.calibrating(), "calib: one real size is not enough: the table stays held back");
        // the other four sizes get their 16 rounds: the table takes over at that request's end, at the real costs
        const double rt[5] = {12.90, 14.60, 16.70, 19.90, 21.30};
        for (int t = 1; t <= 5; ++t)
            if (t != 3)
                for (int i = 0; i < 16; ++i) c.observe(t == 4 || t == 5, t, 1, 0, rt[t - 1]);
        check(c.calibrate(cf, rep) && c.fixed() && !c.table_held() && !c.calibrating(), "calib: with the last size calibrated the table is used");
        check(std::fabs(c.cost_ms(1) - 12.90) < 1e-9 && std::fabs(c.cost_ms(3) - 16.70) < 1e-9 && std::fabs(c.cost_ms(4) - 19.90) < 1e-9 && std::fabs(c.cost_ms(5) - 21.30) < 1e-9,
              "calib: the real costs decide");
        for (int i = 0; i < 40; ++i) c.observe(false, 3, 2, 0, 15.0);
        check(!c.calib_dirty() && std::fabs(c.cost_ms(3) - 16.70) < 1e-9, "calib: a real size never changes again");
        // a lookup size reached once per request: its rounds come across starts through the file
        CostFile cd;
        cd.key = "k";
        Cal cal1 = cal0;
        Real real1{};
        Rc rc1 = rc0;
        for (int start = 0; start < 16; ++start) {
            DraftPolicy e(8);
            e.set_fixed_costs(tab, 5, 0);
            e.start_calibration(false, real1, cal1, rc1);
            e.observe(true, 4, 2, 0, 18.9);
            std::string r2;
            check(e.calibrate(cd, r2), "calib: a start with one real round of a size writes the file");
            cal1 = cd.cal;
            real1 = cd.real;
            rc1 = cd.real_cost;
        }
        check((cd.real[0] & 8u) != 0 && std::fabs(cd.real_cost[0][4] - 18.9) < 1e-9 && cd.cal[0][4].n == 0 && std::fabs(cd.tables[0][3] - 18.57) < 1e-9,
              "calib: 16 starts of one round each calibrate the size");
        DraftPolicy again(8);   // a start from that file: size 4 real, the others not: the table is held back
        again.set_fixed_costs(tab, 5, 0);
        again.start_calibration(false, real1, cal1, rc1);
        check(again.table_held() && !again.fixed() && again.calibrating(), "calib: a restart with sizes still uncalibrated keeps the table held back");
        {   // a file with every size real: the table is used from the first round, at the real costs
            Real all{};
            all[0] = 31u;
            Rc rcall = rc0;
            for (int t = 1; t <= 5; ++t) rcall[0][(size_t) t] = tab[t - 1] * 1.03;
            DraftPolicy u(8);
            u.set_fixed_costs(tab, 5, 0);
            u.start_calibration(false, all, cal0, rcall);
            check(u.fixed() && !u.table_held() && !u.calibrating() && std::fabs(u.cost_ms(4) - 18.57 * 1.03) < 1e-9 && std::fabs(u.cost_ms(1) - 12.68 * 1.03) < 1e-9,
                  "calib: a start from a fully calibrated file uses the table at the real costs");
        }
        {   // sizes that real requests never reach: after kCalibReqs requests the table is used anyway, those sizes at a scaled learned cost
            DraftPolicy bd(8);
            bd.set_fixed_costs(tab, 5, 0);
            bd.start_calibration(false, Real{}, cal0, rc0, 0);
            CostFile cb;
            std::string rb;
            for (int i = 0; i < 16; ++i) bd.observe(false, 2, 1, 0, 14.60);
            bd.calibrate(cb, rb);
            for (int rq = 2; rq < DraftPolicy::kCalibReqs; ++rq) {
                bd.observe(false, 3, 2, 0, 16.70);
                bd.calibrate(cb, rb);
            }
            check(bd.table_held() && !bd.fixed() && cb.reqs == DraftPolicy::kCalibReqs - 1, "budget: held back before the last request of the budget");
            bd.observe(false, 3, 2, 0, 16.70);
            const bool wr = bd.calibrate(cb, rb);
            const double rr = (14.60 / 14.30 + 16.70 / 16.41) / 2.0;
            check(wr && bd.fixed() && !bd.table_held() && cb.reqs == DraftPolicy::kCalibReqs && bd.calibrating(), "budget: after kCalibReqs requests the table is used, calibration goes on");
            check(std::fabs(bd.cost_ms(2) - 14.60) < 1e-9 && std::fabs(bd.cost_ms(3) - 16.70) < 1e-9 && std::fabs(bd.cost_ms(1) - 12.68) < 1e-9 &&
                      std::fabs(bd.cost_ms(4) - 18.57 * rr) < 1e-6 && std::fabs(bd.cost_ms(5) - 20.90 * rr) < 1e-6,
                  "budget: real sizes at their real cost, sizes not reached at learned x the mean ratio, size 1 at its learned cost");
            DraftPolicy bs(8);   // a start from that file: used at once
            bs.set_fixed_costs(tab, 5, 0);
            bs.start_calibration(false, cb.real, cb.cal, cb.real_cost, cb.reqs);
            check(bs.fixed() && !bs.table_held() && std::fabs(bs.cost_ms(4) - 18.57 * rr) < 1e-6, "budget: a start after the budget uses the table at once, same prices");
            for (int i = 0; i < 16; ++i) bd.observe(true, 4, 2, 0, 19.90);   // a size the policy now chooses collects its own rounds while the table is in use
            check(bd.calibrate(cb, rb) && bd.fixed() && (cb.real[0] & 8u) && std::fabs(bd.cost_ms(4) - 19.90) < 1e-9 && cb.reqs == DraftPolicy::kCalibReqs,
                  "budget: in use, a size with its own kCalibN rounds becomes real");
        }
        {   // a machine with no table: built from the real rounds, no learning stretch, live costs decide until the take-over
            CostFile c0;
            c0.key = "k";
            c0.fresh = true;
            DraftPolicy p(8), lv(8);
            p.start_passive(c0, 5);
            check(p.passive() && p.table_held() && !p.fixed() && p.calibrating() && !p.learning(), "passive: held from the first request, no learning");
            bool same = true;
            for (int i = 0; i < 40; ++i) {
                const DraftPolicy::Pick pa = p.choose(i % 3 + 1, 5, 2), pb = lv.choose(i % 3 + 1, 5, 2);
                same = same && pa.lookup == pb.lookup && pa.t == pb.t && pa.pad == 0 && std::fabs(p.cost_ms(3) - lv.cost_ms(3)) < 1e-12;
                p.set_busy_age(2.0);
                p.observe(i % 3 == 0, i % 3 + 1, 1, 0, 12.7 + 2.0 * (i % 3));
                lv.observe(i % 3 == 0, i % 3 + 1, 1, 0, 12.7 + 2.0 * (i % 3));
            }
            check(same && p.forced_rounds() == 0, "passive: picks and costs equal a policy without a table, no forced window");
            CostFile cp;
            std::string rp;
            check(p.calibrate(cp, rp) && cp.fresh && cp.tables[0].empty() && cp.reqs == 1, "passive: the file has no table yet, one request counted");
            // only the first seconds of a busy run are counted
            DraftPolicy q(8);
            q.start_passive(c0, 5);
            q.set_busy_age(2.0);
            for (int i = 0; i < 10; ++i) q.observe(false, 2, 1, 0, 14.6);
            q.set_busy_age(50.0);
            for (int i = 0; i < 10; ++i) q.observe(false, 2, 1, 0, 14.6);
            CostFile cq;
            q.calibrate(cq, rp);
            check(cq.cal[0][2].n == 9 && cq.fresh, "passive: rounds late in a busy run are not counted; the first round of a size has no live cost to clamp to");
            // the sums survive a restart: 9 + 7 more = 16 own rounds make the size real
            cq.key = "k";
            std::string wpath = "dp_passive_test.costs", wwhy;
            check(write_cost_file(wpath, cq, wwhy), "passive: file written");
            CostFile cr;
            check(read_cost_file(wpath, "k", cr, wwhy) && cr.fresh && cr.tables[0].empty() && cr.cal[0][2].n == 9, "passive: a file with no table reads back");
            std::remove(wpath.c_str());
            DraftPolicy r(8);
            r.start_passive(cr, 5);
            r.set_busy_age(1.0);
            for (int i = 0; i < 8; ++i) r.observe(false, 2, 1, 0, 14.6);
            CostFile cs;
            check(r.calibrate(cs, rp) && (cs.real[0] & 2u) && std::fabs(cs.real_cost[0][2] - 14.6) < 1e-9 && r.passive(), "passive: the size is real after 16 own rounds across a restart");
            // take-over by every size real: the real costs, a table in the file
            DraftPolicy a(8);
            a.start_passive(c0, 5);
            a.set_busy_age(1.0);
            const double rt[5] = {12.9, 14.6, 16.7, 19.9, 21.3};
            for (int t = 1; t <= 5; ++t)
                for (int i = 0; i < 17; ++i) a.observe(t >= 4, t, 1, 0, rt[t - 1]);
            CostFile ca;
            check(a.calibrate(ca, rp) && a.fixed() && !a.passive() && !a.table_held() && !a.calibrating() && !ca.fresh && ca.tables[0].size() == 5, "passive: every size real: the table takes over, the file has a table");
            check(std::fabs(a.cost_ms(1) - 12.9) < 1e-9 && std::fabs(a.cost_ms(4) - 19.9) < 1e-9 && std::fabs(ca.tables[0][2] - 16.7) < 1e-9, "passive: the real costs decide, the file keeps them as the table");
            // take-over after kCalibReqs requests: a size never reached takes the live cost, a never measured one the policy's prior
            DraftPolicy b(8), lb(8);
            b.start_passive(c0, 5);
            b.set_busy_age(1.0);
            for (int i = 0; i < 17; ++i) {
                b.observe(false, 2, 1, 0, 14.6);
                lb.observe(false, 2, 1, 0, 14.6);
            }
            for (int i = 0; i < 6; ++i) {
                b.observe(true, 4, 2, 0, 19.9);
                lb.observe(true, 4, 2, 0, 19.9);
            }
            CostFile cb2;
            for (int rq = 0; rq < DraftPolicy::kCalibReqs; ++rq) {
                b.observe(false, 2, 1, 0, 14.6);
                b.calibrate(cb2, rp);
            }
            check(b.fixed() && !b.passive() && cb2.reqs == DraftPolicy::kCalibReqs && cb2.tables[0].size() == 5, "passive: after kCalibReqs requests the table takes over");
            check(std::fabs(cb2.tables[0][1] - 14.6) < 1e-9 && std::fabs(cb2.tables[0][3] - 19.9) < 1e-9 && std::fabs(cb2.tables[0][2] - lb.cost_ms(3)) < 0.1 && std::fabs(b.cost_ms(4) - 19.9) < 1e-9,
                  "passive: real size 2 at its real cost, size 4 at its live cost, size 3 (never measured) at the policy's prior");
            for (int i = 0; i < 17; ++i) b.observe(true, 4, 2, 0, 20.1);   // the unreached size keeps collecting
            CostFile cb3 = cb2;
            check(b.calibrate(cb3, rp) && (cb3.real[0] & 8u) && std::fabs(b.cost_ms(4) - 20.1) < 1e-9, "passive: after the take-over a size still collects and becomes real at 16 own rounds (its price the median: 5 rounds of 19.9, 11 of 20.1)");
        }
        // the file round trip: state, real, real cost, cal
        CostFile w;
        w.key = "k2";
        w.tables[0].assign(tab, tab + 5);
        w.state_effect = true;
        w.reqs = 7;
        w.real[0] = 6u;
        w.real_cost[0][2] = 14.6;
        w.real_cost[0][3] = 16.7;
        w.cal[0][4] = CalAcc{7, 131.0};
        const std::string path = "dp_calib_test.costs";
        std::string why;
        check(write_cost_file(path, w, why), "calib file: written");
        CostFile r;
        check(read_cost_file(path, "k2", r, why) && r.state_effect && r.reqs == 7 && r.real[0] == 6u && r.cal[0][4].n == 7 && std::fabs(r.cal[0][4].sum - 131.0) < 1e-3 && r.cal[0][2].n == 0 &&
                  std::fabs(r.real_cost[0][2] - 14.6) < 1e-9 && std::fabs(r.real_cost[0][3] - 16.7) < 1e-9,
              "calib file: state, real marks, real costs and collected rounds come back");
        std::remove(path.c_str());
        // a machine with a load-time effect counts only the first kFreshS seconds of a busy run
        DraftPolicy f(8);
        f.set_fixed_costs(tab, 5, 0);
        f.start_calibration(true, Real{}, cal0, rc0);
        f.set_busy_age(2.0);
        for (int i = 0; i < 10; ++i) f.observe(false, 2, 1, 0, 14.6);
        f.set_busy_age(50.0);
        for (int i = 0; i < 10; ++i) f.observe(false, 2, 1, 0, 17.0);
        CostFile cfz;
        std::string rz;
        f.calibrate(cfz, rz);
        check(cfz.cal[0][2].n == 10, "calib: with a state effect only fresh rounds are counted");
        // a disturbed stretch (a stalled machine) is dropped, not taken
        DraftPolicy g(8);
        g.set_fixed_costs(tab, 5, 0);
        g.start_calibration(false, Real{}, cal0, rc0);
        for (int i = 0; i < 16; ++i) g.observe(false, 2, 0, 0, 40.0);
        CostFile cg;
        std::string rg;
        g.calibrate(cg, rg);
        check(rg.empty() && g.table_held() && cg.cal[0][2].n == 0 && (cg.real[0] & 2u) == 0, "calib: rounds far above the learned cost are dropped");
        // a size that real requests never reach, and a machine whose real sizes are all 1 row: nothing is scaled
        DraftPolicy h(8);
        h.set_fixed_costs(tab, 5, 0);
        h.start_calibration(false, Real{}, cal0, rc0);
        for (int i = 0; i < 16; ++i) h.observe(false, 1, 0, 0, 13.0);
        CostFile ch;
        std::string rh;
        h.calibrate(ch, rh);
        check((ch.real[0] & 1u) != 0 && h.table_held() && !h.fixed(), "calib: one real size does not bring the table into use");
    }
    {
        // STRATA_DRAFT_COSTS_LEARN=1 keeps its one-start promise (f-dc-learn-fix): (a) a request of the learning stretch starts a new busy run.
        using strata::spec::CostLearner;
        static_assert(CostLearner::starts_busy_run(false, 0.0, false) && !CostLearner::starts_busy_run(true, 0.01, false) && CostLearner::starts_busy_run(true, 0.5, false),
                      "no learning stretch: the first request and a pause of kIdleS start a busy run, a request a few ms later does not");
        check(CostLearner::starts_busy_run(true, 0.0, true) && CostLearner::starts_busy_run(true, 0.01, true), "learn (a): a request of the learning stretch starts a new busy run, however close");
        check(!CostLearner::starts_busy_run(true, 0.2, false) && CostLearner::starts_busy_run(true, CostLearner::kIdleS, false), "learn (a): without the learning stretch the rule is the 0.5 s pause as before");
        // 40 requests back to back of 200 rounds each (3 window sizes, 40 / 44 / 48 ms, 5 % slower after 6 s of load): one long busy run keeps 150 fresh rounds, a busy run per request 6,000
        const auto sim = [](bool per_request_run) {
            CostLearner L;
            double clock = 0.0;
            for (int q = 0; q < 40; ++q) {
                double req_start = per_request_run ? clock : 0.0;
                for (int i = 0; i < 200; ++i) {
                    const int t = 1 + i % 3;
                    const double age = clock - req_start;
                    const double ms = (40.0 + 4.0 * (t - 1)) * (age > CostLearner::kFreshS ? 1.05 : 1.0) * (1.0 + 0.004 * ((i % 5) - 2));
                    L.add(0, t, ms, age);
                    clock += ms / 1000.0;
                }
            }
            return L.check(1000, 20);
        };
        check(!sim(false).ready, "learn (a): back to back, one busy run: fresh rounds stay at the first seconds, the stretch never finishes (the batch 15 behaviour)");
        check(sim(true).ready && !sim(true).disturbed, "learn (a): back to back, a busy run per request: the stretch finishes");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
