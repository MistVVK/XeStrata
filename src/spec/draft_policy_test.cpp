// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/spec/draft_policy_test.cpp - DraftPolicy: when does a lookup window beat the MTP's?
//
// Simulated rounds with the costs measured on the RTX 5070 (window-cost: ~10 ms more per token) check that
//   1. with no lookup proposal the MTP window is kept;
//   2. lookup drafts that are mostly rejected stop being taken (their bucket's rate falls);
//   3. lookup drafts that are always accepted are taken, and the window grows with them;
//   4. match-length buckets learn separately (short matches failing does not stop long ones);
//   5. the policy never proposes a window beyond its cap;
//   6. XESTRATA_SPEC_COST's auto runs the plain window below the second context bucket and the cost-aware one
//      from it;
//   7. --draft-costs: a fixed table is not moved by round times; the table file reads back what was written and
//      refuses another key or a cut file; a table built from real rounds takes over once every size has kCalibN
//      of them, priced at their median, and a start that reads it uses it at once; a table loaded without real
//      sizes is held back (live costs) until kCalibReqs requests have run;
//   8. at the take-over a size without real rounds is priced between the nearest real sizes, monotone up to the ends,
//      and a context bucket with no table of its own copies a monotone one.
#include "strata/spec/draft_policy.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

using strata::spec::CostFile;
using strata::spec::DraftPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
double cost(int t) { return 19.0 + 10.5 * (t - 1); }   // ms per round, measured shape
// XESTRATA_SPEC_COST for the policies below ("" = its auto, the default)
void set_cost_env(const char* value) {
#if defined(_WIN32)
    _putenv_s("XESTRATA_SPEC_COST", value);
#else
    if (*value) setenv("XESTRATA_SPEC_COST", value, 1);
    else unsetenv("XESTRATA_SPEC_COST");
#endif
}
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
        // XESTRATA_SPEC_COST: auto (the default) runs the plain window below kCostBucket and the cost-aware one from
        // it, each with its own cost table; 1 runs the cost-aware window everywhere, 0 the plain one everywhere
        set_cost_env("");
        DraftPolicy p(6);
        p.set_context(1000);
        for (int i = 0; i < 20; ++i) p.observe(false, 4, 2, 0, 40.0);
        check(p.cost_ms(4) < 45.0, "auto at 1K: a round of the shallow table");
        p.set_context(6143);
        check(!p.cost_aware(), "auto below the second bucket: the plain window");
        p.set_context(6144);
        check(p.cost_aware(), "auto at the second bucket: the cost-aware window");
        p.set_context(32768);
        for (int i = 0; i < 20; ++i) p.observe(false, 4, 2, 0, 120.0);
        check(p.cost_ms(4) > 100.0, "auto at 32K: a table of its own");
        p.set_context(1000);
        check(p.cost_ms(4) < 45.0, "auto back at 1K: the shallow table again");
        // auto at 32K looks one draft ahead (the costs are not convex: 10, 30, 14 ms); at 1K it does not
        DraftPolicy s(6);
        s.set_context(1000);
        for (int t = 1; t <= 3; ++t)
            for (int i = 0; i < 5; ++i) s.observe(false, t, t - 1, 0, t == 1 ? 10.0 : t == 2 ? 30.0 : 14.0);
        float probs[2] = {0.5f, 0.5f};
        check(s.mtp_window(probs, 6) == 1, "auto at 1K: no draft kept on a look-ahead");
        s.set_context(32768);   // the deep table starts from the shallow one's costs
        for (int t = 1; t <= 3; ++t)
            for (int i = 0; i < 3; ++i) s.observe(false, t, t - 1, 0, t == 1 ? 10.0 : t == 2 ? 30.0 : 14.0);
        check(s.mtp_window(probs, 6) > 1, "auto at 32K: the look-ahead keeps the draft");
        // 1: the cost-aware window everywhere, its outlier rounds a tenth of a sample
        set_cost_env("1");
        DraftPolicy q(6);
        q.set_context(1000);
        for (int i = 0; i < 20; ++i) q.observe(false, 4, 2, 0, 40.0);
        q.observe(false, 4, 2, 0, 400.0);
        check(q.cost_ms(4) < 50.0, "on at 1K: a far round counts as a tenth");
        // 0: the plain window everywhere, every round weighed alike and one table for every depth
        set_cost_env("0");
        DraftPolicy r(6);
        r.set_context(1000);
        for (int i = 0; i < 20; ++i) r.observe(false, 4, 2, 0, 40.0);
        r.observe(false, 4, 2, 0, 400.0);
        check(r.cost_ms(4) > 60.0, "off: the far round weighs as much as any");
        r.set_context(32768);
        check(r.cost_ms(4) > 60.0, "off: one table at every depth");
        set_cost_env("");
    }
    {
        DraftPolicy p(4);
        const double tab[4] = {20.0, 25.0, 30.0, 35.0};
        p.set_fixed_costs(tab, 4, 0);
        for (int i = 0; i < 30; ++i) p.observe(false, 3, 2, 0, 90.0);
        check(p.fixed() && p.cost_ms(3) == 30.0, "fixed costs: round times move no cost");
        check(p.cost_ms(4) == 35.0 && p.cost_ms(1) == 20.0, "fixed costs: every size of the table");
    }
    const std::string path = (std::filesystem::temp_directory_path() / "xestrata-draft-policy-test" / "t.costs").string();
    {
        CostFile c;
        c.key = "devices B70|args x";
        c.n[0] = 3;
        c.table[0][1] = 20.0; c.table[0][2] = 25.5; c.table[0][3] = 31.25;
        c.reqs = 7;
        c.real[0] = 1u;
        c.real_cost[0][1] = 19.5;
        c.cal[0][2].n = c.cal[0][2].nv = 2;
        c.cal[0][2].vals[0] = 24.0; c.cal[0][2].vals[1] = 26.0; c.cal[0][2].sum = 50.0;
        std::string why;
        check(strata::spec::write_cost_file(path, c, why), "table file: written");
        CostFile r;
        const bool ok = strata::spec::read_cost_file(path, c.key, r, why);
        check(ok && r.n[0] == 3 && r.table[0][3] == 31.25 && r.reqs == 7 && r.real[0] == 1u && r.real_cost[0][1] == 19.5 &&
                  r.cal[0][2].n == 2 && r.cal[0][2].vals[1] == 26.0 && !r.fresh, "table file: read back as written");
        check(!strata::spec::read_cost_file(path, "devices RTX|args x", r, why), "table file: another key is refused");
        std::ifstream in(path);
        std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::ofstream(path, std::ios::trunc) << all.substr(0, all.rfind("end"));
        check(!strata::spec::read_cost_file(path, c.key, r, why), "table file: a cut file is refused");
    }
    {
        DraftPolicy p(4);
        CostFile cf;
        cf.key = "k";
        cf.fresh = true;
        p.start_passive(cf, p.top_size());
        p.request_start();
        for (int t = 1; t <= 4; ++t) p.observe(false, t, t - 1, 0, cost(t));   // the live policy's first round of a size
        CostFile out = cf;
        std::string rep;
        p.calibrate(out, rep);
        check(p.table_held() && !p.fixed() && out.fresh, "built from real rounds: held while sizes lack rounds");
        p.request_start();
        for (int i = 0; i < DraftPolicy::kCalibN; ++i)
            for (int t = 1; t <= 4; ++t) p.observe(false, t, t - 1, 0, cost(t) + (i == 3 ? 4.0 : 0.0) - (i == 5 ? 1.0 : 0.0));
        p.request_end();
        const bool wrote = p.calibrate(out, rep);
        check(wrote && p.fixed() && !p.table_held() && !out.fresh && out.n[0] == 4, "built from real rounds: the table takes over");
        check(std::fabs(p.cost_ms(3) - cost(3)) < 1e-9, "built from real rounds: a size's price is its rounds' median");
        p.observe(false, 3, 2, 0, 500.0);
        check(std::fabs(p.cost_ms(3) - cost(3)) < 1e-9, "built from real rounds: fixed after the take-over");
        std::string why;
        check(strata::spec::write_cost_file(path, out, why), "built from real rounds: written");
        CostFile r;
        DraftPolicy q(4);
        if (strata::spec::read_cost_file(path, "k", r, why))
            for (int k = 0; k < CostFile::kCtxBuckets; ++k)
                if (r.n[(size_t) k] > 0) q.set_fixed_costs(r.table[(size_t) k].data() + 1, r.n[(size_t) k], k);
        q.start_calibration(r);
        bool same = q.fixed() && !q.calibrating();
        for (int t = 1; t <= 4; ++t) same = same && std::fabs(q.cost_ms(t) - p.cost_ms(t)) < 1e-3;
        check(same, "built from real rounds: the next start uses the same table at once");
    }
    {
        DraftPolicy p(4);
        const double tab[4] = {20.0, 25.0, 30.0, 35.0};
        p.set_fixed_costs(tab, 4, 0);
        CostFile cf;
        cf.key = "k";
        p.start_calibration(cf);
        check(p.table_held() && !p.fixed(), "a table without real sizes: held back, live costs decide");
        std::string rep;
        for (int i = 0; i < DraftPolicy::kCalibReqs && p.table_held(); ++i) {
            p.request_start();
            p.calibrate(cf, rep);
        }
        check(p.fixed() && p.cost_ms(2) == 25.0 && cf.reqs == DraftPolicy::kCalibReqs, "a held table: used after kCalibReqs requests");
    }
    {
        // A live stretch priced size 3 above the measured size 4 (B70 33.66 / 33.50 ms, RTX 66.88 / 60.28 ms):
        // a take-over must not put that guess in the table.  Sizes 2 and 4 have real rounds; 1, 3, 5 and 6 do not.
        set_cost_env("");
        DraftPolicy p(6);
        CostFile cf;
        cf.key = "k";
        cf.fresh = true;
        p.start_passive(cf, p.top_size());
        p.request_start();
        for (int i = 0; i < 20; ++i) {
            p.observe(false, 2, 1, 0, 30.0);
            p.observe(false, 4, 3, 0, 50.0);
        }
        for (int i = 0; i < 5; ++i) p.observe(false, 3, 2, 0, 120.0);   // live, not real
        for (int i = 0; i < 5; ++i) p.observe(false, 5, 4, 0, 200.0);
        for (int i = 0; i < 5; ++i) p.observe(false, 6, 5, 0, 250.0);
        std::string rep;
        for (int i = 0; i < DraftPolicy::kCalibReqs && p.table_held(); ++i) {
            p.request_start();
            p.calibrate(cf, rep);
        }
        check(p.fixed() && !p.table_held() && !cf.fresh && cf.n[0] == 6, "take-over: the table is used and saved");
        bool mono = true, live_mono = true;
        for (int t = 1; t < 6; ++t) {
            mono &= cf.table[0][(size_t) t + 1] >= cf.table[0][(size_t) t];
            live_mono &= p.cost_ms(t + 1) >= p.cost_ms(t);
        }
        check(mono, "take-over: costs do not fall as the window grows");
        check(live_mono, "take-over: the policy's fixed costs do not fall as the window grows");
        check(std::fabs(cf.table[0][3] - 40.0) < 1e-6, "take-over: an unmeasured size interpolates its real neighbours");
        check(std::fabs(cf.table[0][1] - 30.0) < 1e-6, "take-over: below the first real size takes its price");
        check(std::fabs(cf.table[0][5] - 50.0) < 1e-6 && std::fabs(cf.table[0][6] - 50.0) < 1e-6,
              "take-over: past the last real size takes its price");
        p.set_context(70000);   // auto: this bucket has no table of its own and copies the nearest one
        bool filled_mono = true;
        for (int t = 1; t < 6; ++t) filled_mono &= p.cost_ms(t + 1) >= p.cost_ms(t);
        check(filled_mono, "a filled bucket keeps the table monotone");
    }
    {
        // No size reached 16 real rounds: every price is a live guess.  The guesses fell with the window; the table
        // still starts at a price and never falls.
        DraftPolicy p(6);
        CostFile cf;
        cf.key = "k";
        cf.fresh = true;
        p.start_passive(cf, p.top_size());
        p.request_start();
        for (int t = 1; t <= 3; ++t)
            for (int i = 0; i < 3; ++i) p.observe(false, t, t - 1, 0, 40.0 - 10.0 * (t - 1));
        std::string rep;
        for (int i = 0; i < DraftPolicy::kCalibReqs && p.table_held(); ++i) {
            p.request_start();
            p.calibrate(cf, rep);
        }
        bool mono = cf.table[0][1] > 0.0;
        for (int t = 1; t < 6; ++t) mono &= cf.table[0][(size_t) t + 1] >= cf.table[0][(size_t) t];
        check(mono, "take-over with no real sizes: a priced first size and no fall");
        check(std::fabs(cf.table[0][1] - 40.0) < 1e-6 && std::fabs(cf.table[0][6] - 40.0) < 1e-6,
              "take-over with no real sizes: guesses are clamped to the highest one");
    }
    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
