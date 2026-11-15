// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/spec/drafter_policy_test.cpp - DrafterPolicy: which drafter a request uses, by kind and context depth.
//
//   1. without DFlash2 the MTP is the only arm;
//   2. a new kind explores the MTP twice, then DFlash2 twice, then exploits the faster;
//   3. the faster arm wins (DFlash2 unless the MTP is clearly ahead);
//   4. the loser is still explored now and then;
//   5. depth buckets learn separately (a drafter fast at 1K may lose at 64K);
//   6. kinds learn separately;
//   7. short requests teach nothing.
#include "strata/spec/drafter_policy.hpp"

#include <cstdio>
#include <vector>

using strata::spec::DrafterPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
std::vector<int64_t> prompt(int32_t seed) {
    std::vector<int64_t> ids(40);
    for (int i = 0; i < 40; ++i) ids[(size_t) i] = seed + i * 7;
    return ids;
}
// teach a cell: n requests of an arm at a depth, each committing `tok` tokens in `ms`
void teach(DrafterPolicy& p, uint64_t kind, int64_t pos, DrafterPolicy::Arm arm, int n, double tok, double ms) {
    for (int i = 0; i < n; ++i) p.observe(kind, pos, arm, (int64_t) tok, ms);
}
}  // namespace

int main() {
    std::printf("drafter_policy_test\n");
    const auto a = prompt(100), b = prompt(9000);
    const uint64_t ka = DrafterPolicy::kind_of(a.data(), (int64_t) a.size());
    const uint64_t kb = DrafterPolicy::kind_of(b.data(), (int64_t) b.size());
    check(ka != kb, "different prompts are different kinds");
    check(DrafterPolicy::kind_of(a.data(), (int64_t) a.size()) == ka, "the kind is stable");
    {
        DrafterPolicy p;
        check(p.choose(ka, 100, false) == DrafterPolicy::Mtp, "no DFlash2: the MTP");
        check(p.choose(ka, 100, true) == DrafterPolicy::Mtp, "a new kind starts with the MTP (the prior)");
        p.observe(ka, 100, DrafterPolicy::Mtp, 100, 1000);
        check(p.choose(ka, 100, true) == DrafterPolicy::Mtp, "the MTP is explored twice");
        p.observe(ka, 100, DrafterPolicy::Mtp, 100, 1000);
        check(p.choose(ka, 100, true) == DrafterPolicy::Df2, "then DFlash2 is explored");
        p.observe(ka, 100, DrafterPolicy::Df2, 100, 500);   // DFlash2 twice as fast
        p.observe(ka, 100, DrafterPolicy::Df2, 100, 500);
        check(p.choose(ka, 100, true) == DrafterPolicy::Df2, "the faster arm (DFlash2) wins");
    }
    {
        DrafterPolicy p;
        teach(p, ka, 100, DrafterPolicy::Mtp, 2, 100, 500);    // MTP faster
        teach(p, ka, 100, DrafterPolicy::Df2, 2, 100, 1000);
        check(p.choose(ka, 100, true) == DrafterPolicy::Mtp, "the MTP wins when it is clearly ahead");
        int df2 = 0;
        for (int i = 0; i < 24; ++i) if (p.choose(ka, 100, true) == DrafterPolicy::Df2) ++df2;
        check(df2 >= 1 && df2 <= 6, "the loser (DFlash2) is still explored now and then");
    }
    {
        DrafterPolicy p;
        // DFlash2 fast at 1K, slow at 64K (the MTP the opposite): the buckets decide separately
        teach(p, ka, 1000, DrafterPolicy::Mtp, 2, 100, 1000);
        teach(p, ka, 1000, DrafterPolicy::Df2, 2, 100, 500);
        teach(p, ka, 64000, DrafterPolicy::Mtp, 2, 100, 500);
        teach(p, ka, 64000, DrafterPolicy::Df2, 2, 100, 1000);
        check(p.choose(ka, 1000, true) == DrafterPolicy::Df2, "at 1K DFlash2 wins");
        check(p.choose(ka, 64000, true) == DrafterPolicy::Mtp, "at 64K the MTP wins");
    }
    {
        DrafterPolicy p;
        teach(p, ka, 100, DrafterPolicy::Mtp, 2, 100, 1000);
        teach(p, ka, 100, DrafterPolicy::Df2, 2, 100, 500);
        check(p.choose(ka, 100, true) == DrafterPolicy::Df2, "kind A learned DFlash2");
        check(p.choose(kb, 100, true) == DrafterPolicy::Mtp, "kind B is independent (starts with the MTP)");
    }
    {
        DrafterPolicy p;
        p.observe(ka, 100, DrafterPolicy::Df2, 10, 100);   // too short to teach
        check(p.tps(ka, 100, DrafterPolicy::Df2) == 0.0, "a short request teaches nothing");
        p.observe(ka, 100, DrafterPolicy::Df2, 100, 1000);
        check(p.tps(ka, 100, DrafterPolicy::Df2) > 0.0, "a long request teaches");
    }
    {
        check(DrafterPolicy::depth_bucket(0) == 0 && DrafterPolicy::depth_bucket(2047) == 0, "depth bucket 0: < 2K");
        check(DrafterPolicy::depth_bucket(2048) == 1 && DrafterPolicy::depth_bucket(8191) == 1, "depth bucket 1: 2K..8K");
        check(DrafterPolicy::depth_bucket(8192) == 2 && DrafterPolicy::depth_bucket(32767) == 2, "depth bucket 2: 8K..32K");
        check(DrafterPolicy::depth_bucket(32768) == 3 && DrafterPolicy::depth_bucket(1000000) == 3, "depth bucket 3: >= 32K");
    }
    {
        DrafterPolicy p(1.01, 1);
        teach(p, ka, 0, DrafterPolicy::Mtp, 2, 4, 10);
        teach(p, ka, 0, DrafterPolicy::Df2, 2, 4, 10);
        check(p.choose(ka, 0, true) == DrafterPolicy::Mtp, "per-window learner: short block wins a tie");
        teach(p, ka, 0, DrafterPolicy::Df2, 4, 8, 10);
        check(p.choose(ka, 0, true) == DrafterPolicy::Df2, "per-window learner: faster long block wins");
    }
    std::printf("drafter_policy_test: %s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}