// src/core/device_hits_plan_test.cpp - CPU unit tests for partial-VRAM device hit planning and tail-skip weight renormalization.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "strata/core/device_hits_plan.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using namespace strata::core;

    CHECK(!device_hits_env_on(nullptr));
    CHECK(!device_hits_env_on(""));
    CHECK(!device_hits_env_on("0"));
    CHECK(device_hits_env_on("1"));

    CHECK(!route_tail_renorm_env_on(nullptr));
    CHECK(!route_tail_renorm_env_on("0"));
    CHECK(route_tail_renorm_env_on("1"));

    constexpr int K = 10;
    constexpr int NE = 512;
    constexpr long long capx = 80;
    constexpr long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    constexpr size_t plan_words = (size_t) (ptr_off + 4 * capx + capx + 16);
    constexpr uintptr_t cache_base = 0x50000000u;
    constexpr long long blob = 4096;

    std::mt19937 rng(20261120u);
    std::vector<unsigned long long> slot_off(1024);
    for (size_t s = 0; s < slot_off.size(); ++s) slot_off[s] = (unsigned long long) s * 8192u + (s % 5u) * 64u;

    // Sweep residency fractions (0%, 35%, 80%, 100%) and entry counts 1..80.
    const int hit_pcts[4] = {0, 35, 80, 100};
    for (int pct : hit_pcts) {
        std::vector<int32_t> res(NE, -1);
        for (int e = 0; e < NE; ++e) {
            if ((int) (rng() % 100) < pct) res[(size_t) e] = (e * 13 + 7) % 900;
        }
        if (pct == 0) std::fill(res.begin(), res.end(), -1);
        if (pct == 100) {
            for (int e = 0; e < NE; ++e) res[(size_t) e] = e;
        }

        for (int n : {1, 10, 23, 60, 80}) {
            std::vector<int32_t> ids((size_t) n);
            for (int i = 0; i < n; ++i) ids[(size_t) i] = (int32_t) (rng() % 24) * 7;
            if (n >= 10) ids[n - 1] = -1;   // out-of-range sentinel safety

            for (int use_off = 0; use_off < 2; ++use_off) {
                std::vector<int32_t> plan(plan_words, -99);
                build_device_hits_plan(ids.data(), n, K, res.data(), NE, cache_base,
                                       use_off ? slot_off.data() : nullptr, blob, plan.data(), capx);

                const int groups = plan[0];
                const int entries = plan[1];
                CHECK(plan[2] == 0);
                CHECK(plan[4 + groups] == entries);
                CHECK(plan[(size_t) (ptr_off + 4 * capx)] == entries);
                if (pct == 0) CHECK(groups == 0 && entries == 0);

                // Every hit entry in [0, n) must appear exactly once in dst[0..entries), and no miss entry may appear.
                const int32_t* dst = plan.data() + 4 + capx + 1;
                const int32_t* tok = dst + capx;
                std::vector<int> seen_entry((size_t) n, 0);
                for (int p = 0; p < entries; ++p) {
                    const int d = dst[p];
                    CHECK(d >= 0 && d < n);
                    CHECK(tok[p] == d / K);
                    const int32_t e = ids[(size_t) d];
                    CHECK(e >= 0 && e < NE && res[(size_t) e] >= 0);
                    ++seen_entry[(size_t) d];
                }
                for (int i = 0; i < n; ++i) {
                    const int32_t e = ids[(size_t) i];
                    const bool is_hit = (e >= 0 && e < NE && res[(size_t) e] >= 0);
                    CHECK(seen_entry[(size_t) i] == (is_hit ? 1 : 0));
                }
            }
        }
    }

    // Tail-skip renormalization: window of 2 tokens x 10 experts, tail_rank = 7.
    {
        std::vector<int32_t> res(32, -1);
        res[0] = 0; res[1] = 1; res[8] = 8;   // expert 8 is resident -> must NOT be skipped even at rank 8!
        // Token 0 routes expert 7 at rank 7 (miss), expert 8 at rank 8 (resident), expert 9 at rank 9 (miss, but token 1 routes 9 at rank 2!).
        const int32_t ids[20] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            0, 1, 9, 3, 4, 5, 6, 11, 12, 13
        };
        float w[20];
        for (int i = 0; i < 20; ++i) w[i] = 0.10f;
        renormalize_tail_skipped_weights(ids, w, res.data(), 2, 10, 32, 7);

        // Token 0: only rank 7 (expert 7) is skipped; rank 8 (expert 8) is resident and rank 9 (expert 9) has min_rank=2 on token 1!
        CHECK(w[7] == 0.0f);
        CHECK(w[8] > 0.10f && w[9] > 0.10f);
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int r = 0; r < 10; ++r) { sum0 += w[r]; sum1 += w[10 + r]; }
        CHECK(std::fabs(sum0 - 1.0f) < 1e-6f);
        // Token 1: ranks 7, 8, 9 (experts 11, 12, 13) are all non-resident with min_rank >= 7 -> zeroed, sum preserved at 1.0f!
        CHECK(w[17] == 0.0f && w[18] == 0.0f && w[19] == 0.0f);
        CHECK(std::fabs(sum1 - 1.0f) < 1e-6f);
    }

    if (fails == 0) std::printf("device_hits_plan_test: ok\n");
    return fails == 0 ? 0 : 1;
}
