// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// The auto layer split's stage-size rule (#1616): no GPU, no model.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/program/split_rules.hpp"

using namespace strata::program::split_rules;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the reporter's 4-GPU pick on a 48-layer model: layers 0-1, 2, 3-19, 20-47 -> cuts 2,3,20
    CHECK(smallest_stage({2, 3, 20}, 48) == 1);
    CHECK(has_tiny_stage({2, 3, 20}, 48));
    // the manual split that fixed it: 4,8,24 is not a cut list of 4 stages; as cuts 4,8,24 -> 4,4,16,24
    CHECK(smallest_stage({4, 8, 24}, 48) == 4);
    CHECK(!has_tiny_stage({4, 8, 24}, 48));
    // identical cards: the balanced split has no tiny stage, so the rule never fires and the pick stays
    CHECK(!has_tiny_stage({12, 24, 36}, 48));
    CHECK(!has_tiny_stage({24}, 48));
    CHECK(has_tiny_stage({47}, 48));
    CHECK(has_tiny_stage({2}, 48));
    CHECK(!has_tiny_stage({3}, 48));
    CHECK(smallest_stage({}, 48) == 48);
    // the restriction needs the layers for it
    CHECK(restriction_possible(48, 4));
    CHECK(restriction_possible(12, 4));
    CHECK(!restriction_possible(11, 4));
    CHECK(!restriction_possible(48, 1));
    // #2644: RTX 4060 Ti 16 GB + RTX 3060 12 GB, 48 layers.  The reporter's numbers: CUDA1 5.50 GiB free of 11.63 (the
    // 3060 side), CUDA0 the 16 GB card with ~14.2 GiB room.  The #1616 retry alone took K=3 (3 layers on the second card)
    // although the first pick K=2 was the same problem; the share of the room gives it 13 layers of 48.
    {
        const int64_t MiB = 1 << 20, L = 48;
        const std::vector<int64_t> room{14500 * MiB, 5632 * MiB};
        const std::vector<int64_t> f = share_floors(room, L);
        CHECK(f.size() == 2 && f[0] == 34 && f[1] == 13);
        CHECK(floors_exceed_minimum(f));
        // the cut K = the second stage's first layer.  With the big card first, the old rule's edge is K=45 (the second card
        // with 3 layers): below its floor.
        CHECK(below_floors({45}, L, f));
        CHECK(below_floors({47}, L, f));
        CHECK(!below_floors({35}, L, f));   // second card: 13 layers
        CHECK(!below_floors({34}, L, f));   // 14 layers
        CHECK(below_floors({36}, L, f));    // 12 layers: under its share
        CHECK(below_floors({33}, L, f));    // the big card under its share
        // the same pair the other way round: the small card first
        const std::vector<int64_t> r = share_floors({room[1], room[0]}, L);
        CHECK(r[0] == 13 && r[1] == 34);
        CHECK(below_floors({3}, L, r));     // the old retry's K=3 starves the small card
        CHECK(!below_floors({13}, L, r));
        CHECK(!below_floors({14}, L, r));
        // a predicted-time pick under the floors sits at the edge of the floor, not at the three-layer edge
        // (layer ms 0.82 on the 4060 Ti, 1.45 on the 3060 as the engine prices them: fewer layers on the 3060 predict faster)
        auto pick = [&](const std::vector<int64_t>& fl) {
            int64_t best_k = -1;
            double best = 1e30;
            for (int64_t k = 2; k < L; ++k) {
                if (!fl.empty() && below_floors({k}, L, fl)) continue;
                if (smallest_stage({k}, L) < (fl.empty() ? kMinStageLayers : 0)) continue;
                const double ms = (double) k * 1.45 + (double) (L - k) * 0.82;   // card 0 = the 3060 here
                if (ms < best) { best = ms; best_k = k; }
            }
            return best_k;
        };
        CHECK(pick({}) == 3);              // today: the 3060 gets 3 layers
        CHECK(pick(r) == 13);              // with the floors: its share
        std::printf("split_rules_test: #2644 floors %lld,%lld; 3060 first: old retry K=%lld, share retry K=%lld\n",
                    (long long) f[0], (long long) f[1], (long long) pick({}), (long long) pick(r));
        // identical cards: the even split, and the floors never exceed the layers
        const std::vector<int64_t> eq = share_floors({10000 * MiB, 10000 * MiB}, L);
        CHECK(eq[0] == 24 && eq[1] == 24);
        const std::vector<int64_t> three = share_floors({1 * MiB, 1 * MiB, 30000 * MiB}, 12);
        CHECK(three[0] == 3 && three[1] == 3 && three[2] == 6);   // lifted to 3, the big one trimmed to fit 12
        CHECK(share_floors({0, 0}, L)[0] == kMinStageLayers && share_floors({-5, 0}, L)[1] == kMinStageLayers);
        CHECK(!floors_exceed_minimum(share_floors({0, 0}, L)));
        CHECK(!below_floors({}, 48, {}));
    }
    {   // upstream becf4a3d9a: PCIe 4.0 x2 (3.6 GB/s) beside x8 (14.1 GB/s) is a gap of 3.92: unscaled by default,
        // scaled from XESTRATA_SPLIT_LINK_MIN=3
        const std::vector<double> x2x8 = link_miss_scale({3.6, 14.1});
        CHECK(x2x8[0] == 1.0 && x2x8[1] == 1.0);
        const std::vector<double> x2x8s = link_miss_scale({14.1, 3.6}, 3.0);
        CHECK(x2x8s[0] == 1.0 && x2x8s[1] > 3.9 && x2x8s[1] < 3.93);
        // the variable's text: unset / empty / not a number / below 1 keep the default
        CHECK(link_scale_min_resolve(nullptr) == kLinkScaleMin && link_scale_min_resolve("") == kLinkScaleMin);
        CHECK(link_scale_min_resolve("banana") == kLinkScaleMin && link_scale_min_resolve("0.5") == kLinkScaleMin);
        CHECK(link_scale_min_resolve("3") == 3.0 && link_scale_min_resolve("2.5") == 2.5);
    }
    if (failed == 0) std::printf("split_rules_test: ok\n");
    return failed ? 1 : 0;
}
