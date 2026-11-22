// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/split_role_test.cpp - the second GPU's role in --layer-split auto (header-only, CPU): the helper cache's
// window prediction and the choice among the split, the helper and the first GPU alone, with the margins and the
// forced roles.  The one comparison of the shared part both engines call (generate.cpp and plain_model.cpp).
#include "strata/core/split_role.hpp"

#include <cmath>
#include <cstdio>

namespace core = strata::core;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-86s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

core::SplitRoleInputs base() {
    core::SplitRoleInputs in;
    in.alone_ms = 10;
    in.split_ms = 9;
    in.helper_ms = 8;
    in.helper_held = 100;
    in.gain = 0.05;
    in.helper_gain = 0.05;
    in.helper_eligible = true;
    return in;
}
}  // namespace

int main() {
    std::printf("split_role_test\n");
    // the choice
    {
        auto in = base();   // helper 8 < split 9 by >5%, and < alone 10 by >5%: the helper
        auto c = core::choose_split_role(in);
        check(c.role == core::SplitRole::Helper && c.helper_chosen && near(c.ms, 8), "helper clearly fastest: the helper");
    }
    {
        auto in = base();   // the helper beats the split, but alone 7 is faster still: alone
        in.alone_ms = 7;
        auto c = core::choose_split_role(in);
        check(c.role == core::SplitRole::Alone && c.helper_chosen, "helper beats split, alone beats both: alone");
    }
    {
        auto in = base();   // helper 8.8 is not >5% under split 9: the split stands
        in.helper_ms = 8.8;
        auto c = core::choose_split_role(in);
        check(c.role == core::SplitRole::Split && !c.helper_chosen, "helper within the margin: the split");
    }
    {
        auto in = base();   // split 9 not >5% under alone 9.2: alone
        in.helper_eligible = false;
        in.alone_ms = 9.2;
        auto c = core::choose_split_role(in);
        check(c.role == core::SplitRole::Alone, "split within the margin of alone: alone");
    }
    {
        auto in = base();   // --split-skip-if-fits with every pair on the first GPU
        in.skip_if_fits = true;
        in.alone_holds_all = true;
        auto c = core::choose_split_role(in);
        check(c.role == core::SplitRole::SkipIfFits && near(c.ms, -1), "skip-if-fits, alone holds all: one GPU");
    }
    {
        auto in = base();   // skip-if-fits asked but the first GPU does not hold all: the choice stands
        in.skip_if_fits = true;
        in.alone_holds_all = false;
        auto c = core::choose_split_role(in);
        check(c.role == core::SplitRole::Helper, "skip-if-fits, alone does not hold all: the helper");
    }
    {
        auto in = base();   // XESTRATA_SPLIT_ROLE=helper forces it past the margin and past alone
        in.role = "helper";
        in.helper_ms = 9.5;
        in.alone_ms = 7;
        auto c = core::choose_split_role(in);
        check(c.role == core::SplitRole::Helper && c.helper_chosen, "SPLIT_ROLE=helper: the helper whatever the prediction");
    }
    {
        auto in = base();   // XESTRATA_SPLIT_ROLE=split keeps the split
        in.role = "split";
        auto c = core::choose_split_role(in);
        check(c.role == core::SplitRole::Split && !c.helper_chosen, "SPLIT_ROLE=split: the split");
    }
    {
        auto in = base();   // --split-roles helper: the split is no choice, alone still can win
        in.roles_helper = true;
        in.alone_ms = 7;
        auto c = core::choose_split_role(in);
        check(c.role == core::SplitRole::Alone, "--split-roles helper, alone faster: alone");
        in.alone_ms = 10;
        c = core::choose_split_role(in);
        check(c.role == core::SplitRole::Helper, "--split-roles helper, helper faster: the helper");
    }
    {
        auto in = base();   // no expert would go to the helper: it is no choice
        in.helper_held = 0;
        auto c = core::choose_split_role(in);
        check(c.role == core::SplitRole::Split && !c.helper_chosen, "helper holds nothing: the split");
    }
    {
        auto in = base();   // not eligible (batch / pipeline / --no-pool): the split
        in.helper_eligible = false;
        in.alone_ms = 100;
        auto c = core::choose_split_role(in);
        check(c.role == core::SplitRole::Split, "helper not eligible: the split");
    }
    // the helper cache's window: one layer, two pairs, the first GPU holds one and the helper the other
    {
        core::HelperCacheInputs hi;
        hi.layers = 1;
        hi.profile = {{0, 0}, {0, 1}};
        hi.mass = {1.0, 1.0};
        hi.cost = [](int64_t, int64_t) { return 10; };
        hi.blob_bytes = [](int64_t) { return 10.0; };
        hi.cap0 = 10;
        hi.caph = 10;
        hi.bw0 = 10;
        hi.bwh = 10;
        hi.cpu_bw = 10;
        hi.rows = 1;
        hi.experts_per_row = 1;
        hi.rt = 1;
        const auto p = core::predict_helper_cache(hi);
        // hit0 = hith = 0.5: gpu 0.5 ms, cpu 0, help 0.5 + 1 = 1.5 ms; the layer takes the largest, 1.5 ms
        check(p.held == 1 && near(p.ms, 1.5) && near(p.kept_mass, 1.0), "one layer, one pair each: 1.5 ms, both cached");
    }
    {
        core::HelperCacheInputs hi;   // the helper's room is too small for a pair: it holds none, the CPU pays
        hi.layers = 1;
        hi.profile = {{0, 0}, {0, 1}};
        hi.cost = [](int64_t, int64_t) { return 10; };
        hi.blob_bytes = [](int64_t) { return 10.0; };
        hi.cap0 = 10;
        hi.caph = 5;
        hi.bw0 = 10;
        hi.bwh = 10;
        hi.cpu_bw = 10;
        hi.rows = 1;
        hi.experts_per_row = 1;
        const auto p = core::predict_helper_cache(hi);
        // the first GPU holds pair 0; pair 1 misses to the CPU: 10 * 0.5 / 10 = 0.5 ms
        check(p.held == 0 && near(p.ms, 0.5) && near(p.kept_mass, 0.5), "helper too small: the CPU reads the miss");
    }
    std::printf("%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
