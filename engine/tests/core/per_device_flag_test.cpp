// per_device_flag (include/strata/core/per_device_flag.hpp, #4431): a per-card answer is asked once per card, and the first card's answer does not become the
// second card's.  `--legacy` is the old form, `static const bool` made by the first caller: with a gfx11 card first, the gfx12 card gets "yes" and must FAIL
// (registered WILL_FAIL).  Host only: the probe is a fake architecture table.
//   per_device_flag_test [--legacy]
#include <cstdio>
#include <cstring>

#include "strata/core/per_device_flag.hpp"

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

// two cards of a split: device 0 is a gfx1100 (gfx11 matrix cores), device 1 a gfx1201 (gfx12)
static bool is_gfx11(int dev) { return dev == 0; }

int main(int argc, char** argv) {
    const bool legacy = argc > 1 && std::strcmp(argv[1], "--legacy") == 0;
    if (legacy) {
        auto old_form = [](int dev) { static const bool gfx11 = is_gfx11(dev); return gfx11; };   // the answer of whichever card asks first
        check(old_form(0) == true, "legacy: the gfx1100 card asks first and gets yes");
        check(old_form(1) == false, "legacy: the gfx1201 card must get no");   // fails: it gets the first card's yes
        std::printf("%s\n", g_fail ? "FAILED" : "all ok");
        return g_fail ? 1 : 0;
    }
    strata::core::PerDeviceFlag<> f;
    int calls = 0;
    auto probe = [&](int dev) { ++calls; return is_gfx11(dev); };
    check(f.get(0, probe) == true, "device 0 (gfx1100) is asked and answers yes");
    check(f.get(1, probe) == false, "device 1 (gfx1201) is asked on its own and answers no");
    check(calls == 2, "each device was asked once");
    check(f.get(0, probe) == true && f.get(1, probe) == false && calls == 2, "later calls read the stored answers, no new probe");
    // the reverse order: the gfx1201 card first
    strata::core::PerDeviceFlag<> g;
    check(g.get(1, probe) == false && g.get(0, probe) == true, "the gfx1201 card first: the gfx1100 card still gets its own yes");
    // an ordinal outside the table is never stored
    strata::core::PerDeviceFlag<4> h;
    int n = 0;
    auto p2 = [&](int) { ++n; return true; };
    check(h.get(9, p2) && h.get(9, p2) && n == 2, "an ordinal past the table asks every time");
    check(h.get(-1, p2) && n == 3, "a negative ordinal asks every time");
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
