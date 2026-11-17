// src/core/uma_cache_test.cpp - the RAM reserve of an integrated GPU's auto expert cache (no GPU needed).
#include "strata/core/uma_cache.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using namespace strata::core;
    constexpr uint64_t GiB = 1ull << 30;
    // the larger of 8 GiB and 8% of the RAM
    CHECK(uma_desktop_reserve_bytes(32 * GiB, nullptr) == 8 * GiB);
    CHECK(uma_desktop_reserve_bytes(64 * GiB, nullptr) == 8 * GiB);          // 8% = 5.1 GiB
    CHECK(uma_desktop_reserve_bytes(128 * GiB, nullptr) > 10 * GiB);         // 8% = 10.2 GiB
    CHECK(uma_desktop_reserve_bytes(128 * GiB, nullptr) < 11 * GiB);
    CHECK(uma_desktop_reserve_bytes(0, nullptr) == 8 * GiB);                 // RAM unknown: the floor
    // the person's own figure, 0 included
    CHECK(uma_desktop_reserve_bytes(128 * GiB, "0") == 0);
    CHECK(uma_desktop_reserve_bytes(128 * GiB, "4") == 4 * GiB);
    CHECK(uma_desktop_reserve_bytes(128 * GiB, "2.5") == 2 * GiB + GiB / 2);
    CHECK(uma_desktop_reserve_bytes(128 * GiB, "") > 10 * GiB);              // empty: not set
    CHECK(uma_desktop_reserve_bytes(64 * GiB, "x") == 8 * GiB);              // not a number: ignored
    CHECK(uma_desktop_reserve_bytes(64 * GiB, "-3") == 8 * GiB);
    // the room
    CHECK(uma_cache_room_bytes(100 * GiB, 10 * GiB) == 90 * GiB);
    CHECK(uma_cache_room_bytes(10 * GiB, 10 * GiB) == 0);
    CHECK(uma_cache_room_bytes(5 * GiB, 10 * GiB) == 0);
    CHECK(uma_cache_room_bytes(5 * GiB, 0) == 5 * GiB);
    // G16: the Windows HIP floor of the auto cache's reserve is for a discrete card of 15 GiB or more only
    CHECK(wddm_auto_reserve_mib(false, 20 * GiB, 700) == 2560);              // a 20 GB discrete card (#1376): the floor
    CHECK(wddm_auto_reserve_mib(false, 15 * GiB, 700) == 2560);              // 15 GiB: the floor (#1709)
    CHECK(wddm_auto_reserve_mib(false, 16 * GiB - 1, 700) == 2560);          // a 16 GB card reports just under 16
    CHECK(wddm_auto_reserve_mib(false, 8 * GiB, 700) == 700);                // a smaller card keeps its own sizing
    CHECK(wddm_auto_reserve_mib(false, 15 * GiB - 1, 700) == 700);
    CHECK(wddm_auto_reserve_mib(true, 94027ull << 20, 700) == 700);          // a Strix Halo (94,027 MiB pool): no floor
    CHECK(wddm_auto_reserve_mib(true, 20 * GiB, 700) == 700);                // integrated wins whatever the size
    CHECK(wddm_auto_reserve_mib(false, 20 * GiB, 3000) == 3000);             // a larger reserve is never lowered
    CHECK(wddm_auto_reserve_mib(true, 94027ull << 20, 3000) == 3000);
    // the pinned complement cap with batch slots (UD-Q4_K_XL, 131,072 tokens: 1.55 GiB of pinned K/V a session, the main one and each slot)
    {
        const uint64_t kv = GiB * 155 / 100;
        const uint64_t budget38 = 38 * GiB;
        auto capped = [&](int sessions) { const uint64_t c = uma_pinned_complement_cap_bytes(kv * (uint64_t) sessions); return c < budget38 ? c : budget38; };
        // the starts that worked keep their complement: 1 session (solo) 4.62 GiB, 3 (2 slots) 12.47, 4 (3 slots) 16.40
        CHECK(capped(1) > 4 * GiB + GiB * 62 / 100);
        CHECK(capped(3) >= 12 * GiB + GiB * 47 / 100);
        CHECK(capped(4) >= 16 * GiB + GiB * 40 / 100);
        // 4 slots (5 sessions): 22.75 - 7.75 = 15.0 GiB, the largest budget that started (job strix-20261112-1937-7dbd)
        CHECK(capped(5) >= 14 * GiB + GiB * 99 / 100 && capped(5) <= 15 * GiB + GiB / 100);
        CHECK(capped(5) < 18 * GiB);                                          // 18 failed
        // many slots: the K/V alone is over the level (15 sessions x 1.55 GiB = 23.25 > 22.75): the cap is 0 = no RAM copy, and the budget handed to the pin is 1 byte
        // (0 would mean "no limit"): a start with every expert outside the GPU cache read from the SSD, never an error, never a wrapped (negative) size
        CHECK(uma_pinned_complement_cap_bytes(30 * GiB) == 0);
        CHECK(uma_pinned_complement_cap_bytes(kv * 15) == 0);
        CHECK(uma_pinned_complement_budget_bytes(30 * GiB) == 1);
        CHECK(uma_pinned_complement_budget_bytes(kv * 15) == 1);
        CHECK(uma_pinned_complement_budget_bytes(~0ull) == 1);                // the largest input
        CHECK(uma_pinned_complement_cap_bytes(0) == GiB * 2275 / 100);        // no session at all: the whole level
        CHECK(uma_pinned_complement_cap_bytes(kv * 9) > 8 * GiB + GiB * 50 / 100);   // 8 slots + the main session (9 x 1.55 = 13.95): 8.8 GiB, still a copy
        CHECK(uma_pinned_complement_cap_bytes(kv * 14) > 0);                  // 14 sessions: 1.05 GiB left
        CHECK(uma_pinned_complement_budget_bytes(kv * 5) == uma_pinned_complement_cap_bytes(kv * 5));   // above zero the budget is the cap itself
    }
    if (fails == 0) std::printf("uma_cache_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
