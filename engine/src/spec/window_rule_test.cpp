// src/spec/window_rule_test.cpp - which rule picks the verify window, and whose q lists judge the candidates (spec/window_rule.hpp).
//
// DFlash2 runs with the MTP layer loaded beside it as the fallback (use_mtp stays true).  Before this rule existed, the cost-aware
// window (STRATA_SPEC_COST=1, a GB10 default) read `chain_rows`, which only the MTP chain writes, so DFlash2 candidates were
// verified one row at a time; and the MTP's q lists (STRATA_SPEC_PROB=1) were attached to DFlash2's windows.  The tests:
//   1. the table of the fixed cases (DFlash2 + cost rule with and without --draft-conf / --spec-min-p);
//   2. the DEFAULT PATH IS UNCHANGED: for every input the old inline code handled correctly (MTP as the source, any flags;
//      DFlash2 with --draft-conf > 0 or with the cost rule off) a copy of the old code returns the same window;
//   3. the q rule.
#include "strata/spec/window_rule.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace strata::spec;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// the inline code of f-0291-df2-mem 96f0307aa (the min-p rule on `draft_on`, the cost rule on use_mtp; CLI loop; the serve loop differs only in the start value and in first_window, which both callers
// overwrite afterwards)
int old_rule(const WindowRuleIn& in) {
    int T = in.start;
    if (in.use_mtp && in.spec_cost && !in.first_window) {
        T = std::clamp(in.chain_rows, 1, in.cap);
    } else if (in.draft_on && in.spec_min_p > 0.0) {
        T = 1;
        while (T < in.cap && in.dprob[T - 1] >= (float) in.spec_min_p) ++T;
    }
    if (in.df2_active && in.draft_conf > 0.0) {
        T = 1;
        double chain = 1.0;
        while (T < in.cap && (chain *= (double) in.dprob[T - 1]) >= in.draft_conf) ++T;
    }
    return T;
}
}  // namespace

int main() {
    std::printf("window_rule_test\n");
    const float p[8] = {0.9f, 0.8f, 0.3f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
    {   // 1. the table
        WindowRuleIn in;
        in.use_mtp = true; in.draft_on = true; in.cap = 6; in.start = 6; in.dprob = p; in.chain_rows = 1;
        in.df2_active = true; in.spec_cost = true;
        check(choose_window(in) == 6, "DFlash2 + cost rule, no min-p, no conf: the full cap (was 1)");
        in.spec_min_p = 0.5;
        check(choose_window(in) == 3, "DFlash2 + cost rule + --spec-min-p 0.5: stops at the weak draft (was 1)");
        in.spec_min_p = 0.0; in.draft_conf = 0.35;
        check(choose_window(in) == 3, "DFlash2 + cost rule + --draft-conf 0.35: the chain product rule (0.9*0.8*0.3=0.216)");
        in.draft_conf = 0.0; in.first_window = true;
        check(choose_window(in) == 6, "first window: left to the caller (it sets 1 / merge_T after)");
        in = WindowRuleIn{};
        in.use_mtp = true; in.draft_on = true; in.cap = 4; in.start = 4; in.dprob = p; in.spec_cost = true; in.chain_rows = 2;
        check(choose_window(in) == 2, "MTP + cost rule: the chain's rows");
        in.chain_rows = 9;
        check(choose_window(in) == 4, "MTP + cost rule: clamped to the cap");
        in.spec_cost = false; in.spec_min_p = 0.5;
        check(choose_window(in) == 3, "MTP + min-p 0.5: the first weak draft ends the window");
        in.spec_min_p = 0.0;
        check(choose_window(in) == 4, "MTP, no rule: the full cap");
        in.use_mtp = false; in.draft_on = false; in.start = 1;
        check(choose_window(in) == 1, "no MTP (serve): one row unless a lookup fires");
    }
    {   // 2. exhaustive over the small input space; random probabilities
        std::srand(7);
        long n = 0, same = 0, differ_df2_cost = 0, differ_other = 0;
        for (int use_mtp = 0; use_mtp < 2; ++use_mtp)
            for (int df2 = 0; df2 < 2; ++df2)
                for (int cost = 0; cost < 2; ++cost)
                    for (int first = 0; first < 2; ++first)
                        for (int cap = 1; cap <= 8; ++cap)
                            for (int rows = 1; rows <= 8; ++rows)
                                for (double mp : {0.0, 0.25, 0.5})
                                    for (double dc : {0.0, 0.15, 0.35}) {
                                        float pr[8];
                                        for (float& x : pr) x = (float) (std::rand() % 101) / 100.0f;
                                        WindowRuleIn in;
                                        in.use_mtp = use_mtp; in.draft_on = use_mtp || df2; in.df2_active = df2; in.spec_cost = cost; in.first_window = first;
                                        in.chain_rows = rows; in.cap = cap; in.start = use_mtp ? cap : 1; in.spec_min_p = mp;
                                        in.draft_conf = dc; in.dprob = pr;
                                        ++n;
                                        const int a = old_rule(in), b = choose_window(in);
                                        if (a == b) { ++same; continue; }
                                        // a difference is allowed only where the old rule used the MTP chain's rows for DFlash2's candidates
                                        if (use_mtp && df2 && cost && !first && dc == 0.0) ++differ_df2_cost; else ++differ_other;
                                    }
        std::printf("  %ld inputs: %ld identical, %ld differ (DFlash2 + cost rule, no --draft-conf), %ld differ elsewhere\n", n, same, differ_df2_cost, differ_other);
        check(differ_other == 0, "default path unchanged: no input outside the buggy case changes its window");
        check(differ_df2_cost > 0, "the buggy case does change (the test sees it)");
    }
    {   // 3. the q rule
        check(mtp_q_applies(true, false, true, false, false, 4), "MTP drew the window, sampled, probabilistic: q attached");
        check(!mtp_q_applies(true, true, true, false, false, 4), "DFlash2 drew the window: no MTP q (exact-match path)");
        check(!mtp_q_applies(true, false, true, true, false, 4), "a lookup window: point masses, no q");
        check(!mtp_q_applies(true, false, true, false, true, 4), "first window: no q");
        check(!mtp_q_applies(true, false, true, false, false, 1), "one row: no q");
        check(!mtp_q_applies(true, false, false, false, false, 4), "greedy / mode off: no q");
        check(!mtp_q_applies(false, false, true, false, false, 4), "no MTP: no q");
    }
    std::printf("%s\n", g_fail == 0 ? "window_rule_test: ok" : "window_rule_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
