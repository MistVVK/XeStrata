// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/spec/window_rule_test.cpp - the verify window's width by the active drafter (window_rule.hpp).
#include "strata/spec/window_rule.hpp"

#include <cstdio>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("window_rule_test\n");
    using namespace strata::spec;
    {
        const float probs[] = {0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
        check(df2_window(probs, 8, 0.0) == 8, "conf 0: the whole cap");
        check(df2_window(probs, 8, 0.35) == 8, "0.9^7 = 0.48 >= 0.35: all 7 drafts");
        check(df2_window(probs, 8, 0.5) == 7, "0.9^6 = 0.53 >= 0.5 > 0.9^7 = 0.48: 6 drafts");
        const float low[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
        check(df2_window(low, 8, 0.35) == 2, "0.5^1 = 0.5 >= 0.35 > 0.5^2 = 0.25: 1 draft");
        check(df2_window(low, 8, 0.9) == 1, "a first draft below conf: the anchor alone");
        check(df2_window(probs, 4, 0.0) == 4, "the cap bounds the window");
        check(df2_window(nullptr, 6, 0.5) == 6, "no probabilities: the cap");
    }
    {
        check(mtp_cost_applies(true, true, false, false), "the MTP cost window: MTP candidates, cost on");
        check(!mtp_cost_applies(true, true, true, false), "not DFlash2's candidates");
        check(!mtp_cost_applies(true, true, false, true), "not the first single-row window");
        check(!mtp_cost_applies(true, false, false, false), "not without the cost rule");
        check(!mtp_cost_applies(false, true, false, false), "not without the MTP");
    }
    {
        check(skip_draft_for_lookup(true, 8, 10, 8), "a long match defers the DFlash2 draft");
        check(!skip_draft_for_lookup(true, 8, 10, 7), "a short match does not");
        check(!skip_draft_for_lookup(true, 0, 10, 100), "min_match 0: off");
        check(!skip_draft_for_lookup(false, 8, 10, 100), "not for the MTP");
        check(!skip_draft_for_lookup(true, 8, 0, 100), "no lookup proposal: no deferral");
    }
    std::printf("window_rule_test: %s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}