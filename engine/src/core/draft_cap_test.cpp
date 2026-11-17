// src/core/draft_cap_test.cpp - the per-card window cap of --draft dflash2 (CPU only).
#include "strata/core/draft_cap.hpp"

#include <cstdio>

using namespace strata::core;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static constexpr size_t GiB = (size_t) 1 << 30;

int main() {
    CHECK(dflash2_auto_cap(true, 0, 0, 0) == 8);               // Strix Halo (gfx1151)
    CHECK(dflash2_auto_cap(false, 12, 0, 72 * GiB) == 8);      // Blackwell cc 12.0, RTX PRO 5000 72 GB
    CHECK(dflash2_auto_cap(false, 12, 0, 96 * GiB) == 8);      // Blackwell cc 12.0, 96 GB
    CHECK(dflash2_auto_cap(false, 12, 0, 48000000000ull) == 8);// exactly 48 GB
    CHECK(dflash2_auto_cap(false, 12, 0, 48000000000ull - 1) == 6);
    CHECK(dflash2_auto_cap(false, 12, 0, 51033931776ull) == 8);// a 48 GB card reports about 47.5 GiB
    CHECK(dflash2_auto_cap(false, 12, 0, 32 * GiB) == 6);      // RTX 5090: cc 12.0 under 48 GB keeps 6
    CHECK(dflash2_auto_cap(false, 12, 0, 12 * GiB) == 6);      // RTX 5070
    CHECK(dflash2_auto_cap(false, 12, 0, 0) == 6);             // memory unknown keeps 6
    CHECK(dflash2_auto_cap(false, 12, 1, 128 * GiB) == 6);     // DGX Spark (cc 12.1): unchanged whatever the memory it reports
    CHECK(dflash2_auto_cap(false, 12, 1, 24 * GiB) == 6);
    CHECK(dflash2_auto_cap(false, 8, 0, 80 * GiB) == 0);       // Ampere / Hopper: --spec as given
    CHECK(dflash2_auto_cap(false, 8, 9, 48 * GiB) == 0);       // Ada
    CHECK(dflash2_auto_cap(false, 0, 0, 0) == 0);              // another AMD card
    CHECK(dflash2_big_blackwell(12, 0, 72 * GiB));
    CHECK(!dflash2_big_blackwell(12, 1, 72 * GiB));
    CHECK(!dflash2_big_blackwell(9, 0, 72 * GiB));
    CHECK(dflash2_small_nvidia(true, (size_t) 12 << 30));
    CHECK(!dflash2_small_nvidia(true, (size_t) 16 << 30));
    CHECK(!dflash2_small_nvidia(false, (size_t) 8 << 30));
    if (fails == 0) std::printf("draft_cap_test: ok\n");
    return fails == 0 ? 0 : 1;
}
