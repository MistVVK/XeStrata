// the HIP rule of the fused trellis (Gyro) prompt kernels (include/strata/prefill/fused_tq_hip.hpp): gfx12 with STRATA_PF_FUSED_TQ=1 as before; gfx1151 by STRATA_PF_TQ_G1151
// (1 on, other values off, unset = the Windows default unless STRATA_GFX1151_DEFAULTS=0); gfx1100-1102 by STRATA_PF_TQ_G11 (g11_on, #6067); every other device off.
// Every input combination is checked.  No GPU, no model.
#include <cstdio>

#include "strata/prefill/fused_tq_hip.hpp"

using strata::prefill::fused_tq_hip::device_on;
using strata::prefill::fused_tq_hip::kG1151DefaultOn;
using strata::prefill::fused_tq_hip::pair_on;
using strata::prefill::fused_tq_hip::parse_switch;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// the expected answer, written out apart from the header's form
static bool want_device(bool w12, bool g1151, bool win, bool tq, int sw, bool doff, bool def) {
    if (!tq) return false;                          // no STRATA_PF_FUSED_TQ=1 (a build with MMQ, nothing set): never
    if (w12) return true;                           // gfx12: unchanged, the switch and the gfx1151 rules do not apply
    if (!g1151) return false;                       // gfx1100-1103, gfx1150, gfx9: never
    if (sw == 1) return true;                       // gfx1151, the switch at 1: on (Linux too)
    if (sw == 0) return false;                      // gfx1151, the switch at another value: off
    return def && win && !doff;                     // unset: the Windows default
}

int main() {
    // the variable's three readings
    CHECK(parse_switch(nullptr) == -1 && parse_switch("") == -1);
    CHECK(parse_switch("1") == 1 && parse_switch("1x") == 1);
    CHECK(parse_switch("0") == 0 && parse_switch("2") == 0 && parse_switch("off") == 0);

    // the shipped default for Windows gfx1151
    CHECK(kG1151DefaultOn == true);
    // with the shipped default: Windows gfx1151 on when unset; Linux, STRATA_GFX1151_DEFAULTS=0, the switch at 0 and a build without STRATA_PF_FUSED_TQ off
    CHECK(strata::prefill::fused_tq_hip::device_on(false, true, true, true, -1, false, kG1151DefaultOn));
    CHECK(!strata::prefill::fused_tq_hip::device_on(false, true, false, true, -1, false, kG1151DefaultOn));
    CHECK(!strata::prefill::fused_tq_hip::device_on(false, true, true, true, -1, true, kG1151DefaultOn));
    CHECK(!strata::prefill::fused_tq_hip::device_on(false, true, true, true, 0, false, kG1151DefaultOn));
    CHECK(!strata::prefill::fused_tq_hip::device_on(false, true, true, false, -1, false, kG1151DefaultOn));

    const int sws[] = {-1, 0, 1};
    for (int m = 0; m < 64; ++m) {
        const bool w12 = m & 1, g1151 = m & 2, win = m & 4, tq = m & 8, doff = m & 16, def = m & 32;
        if (w12 && g1151) continue;   // (one device is never both)
        for (int sw : sws) {
            const bool dv = device_on(w12, g1151, win, tq, sw, doff, def);
            CHECK(dv == want_device(w12, g1151, win, tq, sw, doff, def));
            // Gyro-S's pair follows the device rule
            CHECK(pair_on(w12, g1151, win, tq, sw, doff, def, true, false) == dv);
            // Gyro-M's pair: gfx12 as before; gfx1151 only with the switch at 1
            CHECK(pair_on(w12, g1151, win, tq, sw, doff, def, false, true) == (dv && (w12 || sw == 1)));
            // anything that is neither pair: never
            CHECK(!pair_on(w12, g1151, win, tq, sw, doff, def, false, false));
        }
    }

    // spelled out: gfx12 under a build without MMQ keeps both pairs whatever the gfx1151 switch says
    CHECK(device_on(true, false, true, true, 0, true, false) && pair_on(true, false, false, true, -1, false, false, false, true));
    // a build with MMQ (STRATA_PF_FUSED_TQ unset): gfx1151 stays off even with the switch at 1
    CHECK(!device_on(false, true, true, false, 1, false, true));
    // gfx1100 (a 7900 XTX) and gfx1150: off with every switch
    CHECK(!device_on(false, false, true, true, 1, false, true) && !device_on(false, false, false, true, -1, false, true));
    // Linux gfx1151 with the default on: off unless the switch is 1
    CHECK(!device_on(false, true, false, true, -1, false, true) && device_on(false, true, false, true, 1, false, true));
    // Windows gfx1151 with the default on: STRATA_GFX1151_DEFAULTS=0 or the switch at 0 turns it off
    CHECK(device_on(false, true, true, true, -1, false, true) && !device_on(false, true, true, true, -1, true, true) && !device_on(false, true, true, true, 0, false, true));

    // #6067: the RDNA3 cards (gfx1100 / 1101 / 1102) by g11_on: the shipped default is on; the switch at 0 off, at 1 on; never without STRATA_PF_FUSED_TQ
    // (a build with MMQ) and never for another device class (g11 false: gfx1103, gfx1150, gfx1151, gfx12 keep their own rules above)
    using strata::prefill::fused_tq_hip::g11_on;
    using strata::prefill::fused_tq_hip::kG11DefaultOn;
    CHECK(kG11DefaultOn == true);
    for (int m = 0; m < 8; ++m) {
        const bool g11 = m & 1, tq = m & 2, def = m & 4;
        for (int sw : sws) {
            const bool want = g11 && tq && (sw >= 0 ? sw == 1 : def);
            CHECK(g11_on(g11, tq, sw, def) == want);
        }
    }
    CHECK(g11_on(true, true, -1, kG11DefaultOn));
    CHECK(!g11_on(true, true, 0, kG11DefaultOn) && g11_on(true, true, 1, false));
    CHECK(!g11_on(true, false, 1, kG11DefaultOn) && !g11_on(false, true, 1, kG11DefaultOn));
    // the Radeon 780M: only with the switch at 1 and STRATA_PF_FUSED_TQ (never unset, never at 0)
    using strata::prefill::fused_tq_hip::g1103_on;
    CHECK(g1103_on(true, true, 1));
    CHECK(!g1103_on(true, true, -1) && !g1103_on(true, true, 0) && !g1103_on(true, false, 1) && !g1103_on(false, true, 1));

    if (failed) { std::fprintf(stderr, "fused_tq_hip_test: %d failed\n", failed); return 1; }
    std::printf("fused_tq_hip_test: ok\n");
    return 0;
}
