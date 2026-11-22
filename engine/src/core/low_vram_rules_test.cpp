// src/core/low_vram_rules_test.cpp - pure unit tests for Track 6A low-VRAM (<= 13 GiB) rules (no GPU needed).
#include "strata/core/low_vram_rules.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using namespace strata::core;
    constexpr uint64_t GiB = 1ull << 30;

    // 1. Automatic non-vision VRAM reserve on discrete <= 12.5 GiB GPUs
    CHECK(low_vram_auto_reserve_mib(true, false, 700, 8 * GiB) == kLowVram8ReserveMib);
    CHECK(low_vram_auto_reserve_mib(true, false, 700, 8 * GiB - (128ull << 20)) == kLowVram8ReserveMib);
    CHECK(low_vram_auto_reserve_mib(true, false, 700, 10 * GiB) == kLowVram12ReserveMib);
    CHECK(low_vram_auto_reserve_mib(true, false, 700, 12 * GiB) == kLowVram12ReserveMib);
    CHECK(low_vram_auto_reserve_mib(true, false, 700, 16 * GiB) == 700);
    CHECK(low_vram_auto_reserve_mib(true, false, 700, 24 * GiB) == 700);
    CHECK(low_vram_auto_reserve_mib(true, true, 700, 8 * GiB) == 700);      // explicit reserve kept
    CHECK(low_vram_auto_reserve_mib(true, false, 2100, 8 * GiB) == 2100);   // vision reserve kept
    CHECK(low_vram_auto_reserve_mib(false, false, 700, 8 * GiB) == 700);    // iGPU untouched
    CHECK(low_vram_auto_reserve_mib(true, false, 700, 0) == 700);           // unknown VRAM untouched

    // 2. Tiny-cache (< 25%) adapt_swaps dampening
    constexpr int64_t kTotal = 48 * 512;  // 24,576
    CHECK(low_vram_adapt_swaps(96, false, 0, kTotal) == 0);        // 0-slot 4 GB mode
    CHECK(low_vram_adapt_swaps(96, false, 1400, kTotal) == 16);    // 5.7% (8 GB RTX 3070)
    CHECK(low_vram_adapt_swaps(96, false, 2560, kTotal) == 16);    // 10.4% (12 GB RTX 4070 Ti / 5070)
    CHECK(low_vram_adapt_swaps(96, false, 4500, kTotal) == 32);    // 18.3% (16 GB)
    CHECK(low_vram_adapt_swaps(96, false, 7000, kTotal) == 96);    // 28.5% (>= 25% untouched)
    CHECK(low_vram_adapt_swaps(96, true, 1400, kTotal) == 96);     // explicit --adapt-swaps kept
    CHECK(low_vram_adapt_swaps(12, false, 1400, kTotal) == 12);    // never raises smaller value
    CHECK(low_vram_adapt_swaps(0, false, 1400, kTotal) == 0);

    // 3. Zipf-head profile pinning on tiny caches (< 25% resident)
    CHECK(low_vram_pin_slots(0, kTotal, nullptr) == 0);
    CHECK(low_vram_pin_slots(1400, kTotal, nullptr) == 350);       // 25% of 1400
    CHECK(low_vram_pin_slots(2560, kTotal, nullptr) == 640);       // 25% of 2560
    CHECK(low_vram_pin_slots(7000, kTotal, nullptr) == 0);         // >= 25% resident: off by default
    CHECK(low_vram_pin_slots(1400, kTotal, "0") == 0);             // explicit off
    CHECK(low_vram_pin_slots(1400, kTotal, "0.5") == 700);         // explicit 50%
    CHECK(low_vram_pin_slots(1400, kTotal, "0.99") == 1050);       // clamped to 75%

    // 4. Auto STRATA_L3_PF=1 (<= 13 GiB, > 50% host experts, >= 8 workers)
    CHECK(l3_pf_auto_default(nullptr, 12 * GiB, 2560, kTotal, 9));       // 12 GB + 9 workers: ON
    CHECK(l3_pf_auto_default(nullptr, 8 * GiB, 1400, kTotal, 8));        // 8 GB + 8 workers: ON
    CHECK(!l3_pf_auto_default(nullptr, 8 * GiB, 1400, kTotal, 5));       // 5 workers: OFF (static slice hurts)
    CHECK(!l3_pf_auto_default(nullptr, 12 * GiB, 2039, kTotal, 5));      // 5 workers (owner): OFF
    CHECK(!l3_pf_auto_default(nullptr, 16 * GiB, 4000, kTotal, 16));     // > 13 GiB: OFF
    CHECK(!l3_pf_auto_default(nullptr, 12 * GiB, 13000, kTotal, 16));    // >= 50% VRAM resident: OFF
    CHECK(l3_pf_auto_default("1", 24 * GiB, 24000, kTotal, 4));          // explicit =1 wins
    CHECK(!l3_pf_auto_default("0", 12 * GiB, 2560, kTotal, 16));         // explicit =0 wins

    if (fails == 0) std::printf("low_vram_rules_test: ok\n");
    return fails ? 1 : 0;
}
