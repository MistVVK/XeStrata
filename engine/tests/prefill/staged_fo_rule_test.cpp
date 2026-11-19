// staged_fo_switch / staged_fo_off / staged_fo_decide / staged_entry_layer (include/strata/prefill/staged_fo_rule.hpp): the staged group entry takes a fused-only layer
// with STRATA_PF_STAGED_FO=1, or by default on the measured class (STRATA_PF_STAGED_FO=0 turns it off).  No GPU, no model.
#include <cstdio>

#include "strata/prefill/staged_fo_rule.hpp"

using strata::prefill::staged_entry_layer;
using strata::prefill::staged_fo_decide;
using strata::prefill::staged_fo_off;
using strata::prefill::short_read_default;
using strata::prefill::staged_fo_switch;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the switch: only "1" is on, only "0" is off
    CHECK(!staged_fo_switch(nullptr));
    CHECK(!staged_fo_switch(""));
    CHECK(!staged_fo_switch("0"));
    CHECK(!staged_fo_switch("11"));
    CHECK(!staged_fo_switch("true"));
    CHECK(staged_fo_switch("1"));
    CHECK(!staged_fo_off(nullptr));
    CHECK(!staged_fo_off(""));
    CHECK(staged_fo_off("0"));
    CHECK(!staged_fo_off("00"));
    CHECK(!staged_fo_off("1"));
    // the decision: unset (or anything else) takes the class default, "1" on anywhere, "0" off everywhere
    CHECK(staged_fo_decide(nullptr, true));
    CHECK(!staged_fo_decide(nullptr, false));
    CHECK(staged_fo_decide("", true));
    CHECK(!staged_fo_decide("yes", false));
    CHECK(staged_fo_decide("1", false));
    CHECK(staged_fo_decide("1", true));
    CHECK(!staged_fo_decide("0", true));    // the way back: the measured class with the entry off
    CHECK(!staged_fo_decide("0", false));
    // the class default: a build without MMQ takes the entry on every class (when the prerequisites hold); a build with MMQ only on the measured class with every layer fused-only
    static_assert(strata::prefill::staged_fo_class_default(true, false, true), "no MMQ, any class: on");
    static_assert(strata::prefill::staged_fo_class_default(true, true, true), "no MMQ, the measured class: on");
    static_assert(!strata::prefill::staged_fo_class_default(true, false, false), "no MMQ, prerequisites missing (no gather or no group entry): off");
    static_assert(strata::prefill::staged_fo_class_default(false, true, true), "MMQ build, the measured class with every layer fused-only: on (as before)");
    static_assert(!strata::prefill::staged_fo_class_default(false, false, true), "MMQ build, another class: off (as before)");
    static_assert(!strata::prefill::staged_fo_class_default(false, true, false), "MMQ build, prerequisites missing: off");
    CHECK(staged_fo_decide(nullptr, strata::prefill::staged_fo_class_default(true, false, true)));
    CHECK(!staged_fo_decide("0", strata::prefill::staged_fo_class_default(true, false, true)));    // the way back stays
    CHECK(!staged_fo_decide(nullptr, strata::prefill::staged_fo_class_default(false, false, true)));
    // --short-read: the class default only where the entry is on for a measured class with every layer fused-only; elsewhere the engine's own 64
    static_assert(strata::prefill::kStagedEntryShortRead > 0 && strata::prefill::kStagedEntryShortRead < 64, "the entry's crossing is below the windows' 64");
    static_assert(strata::prefill::short_read_default(64, true, true, true) == strata::prefill::kStagedEntryShortRead, "the class default");
    static_assert(strata::prefill::short_read_default(64, true, true, false) == 64, "entry off (STRATA_PF_STAGED_FO=0): the windows' 64");
    static_assert(strata::prefill::short_read_default(64, false, true, true) == 64, "another class: unchanged");
    static_assert(strata::prefill::short_read_default(64, true, false, true) == 64, "a layer on MMQ or FP16: unchanged");
    // Windows gfx1151, an all-trellis set on the fused kernels with a Hadamard-folded layer (Gyro-S, STRATA_PF_TQ_G1151 on): the windows' 64
    static_assert(strata::prefill::short_read_default(64, true, true, true, true) == 64, "folded trellis set: the windows' 64");
    static_assert(strata::prefill::short_read_default(64, true, true, false, true) == 64, "folded trellis set, entry off: 64");
    static_assert(strata::prefill::short_read_default(64, true, true, true, false) == strata::prefill::kStagedEntryShortRead, "no folded trellis set: the class default as before");
    static_assert(strata::prefill::short_read_default(64, false, true, true, false) == 64, "another class: unchanged");
    CHECK(short_read_default(64, true, true, true, true) == 64);
    CHECK(short_read_default(64, true, true, true) == strata::prefill::kStagedEntryShortRead);
    // BW-12: a compute capability 12.0 CUDA card with every layer fused-only and the entry on reads 17 to 64 token pieces batched (16 and below through the windows)
    static_assert(strata::prefill::kSm120ShortRead == 16, "the sweep's crossing (RTX 5090, UD-Q4_K_XL)");
    static_assert(strata::prefill::short_read_default(64, false, false, true, false, true) == strata::prefill::kSm120ShortRead, "sm_120, every layer fused-only, entry on");
    static_assert(strata::prefill::short_read_default(64, false, false, false, false, true) == 64, "sm_120, entry off (STRATA_PF_STAGED_FO=0): the windows' 64");
    static_assert(strata::prefill::short_read_default(64, false, false, true, false, false) == 64, "another card (or a layer not fused-only): unchanged");
    static_assert(strata::prefill::short_read_default(64, true, true, true, false, true) == strata::prefill::kStagedEntryShortRead, "the Strix Halo class keeps its 24");
    CHECK(short_read_default(64, false, false, true, false, true) == 16);
    CHECK(short_read_default(64, false, false, true) == 64);
    // the layer rule: a layer with an MMQ tile is the entry's earlier rule, whatever the decision says
    static_assert(staged_entry_layer(true, false, false), "an MMQ layer keeps the entry");
    static_assert(staged_entry_layer(true, false, true), "an MMQ layer keeps the entry");
    static_assert(staged_entry_layer(true, true, false), "an MMQ layer keeps the entry");
    // a fused-only layer only when the entry is on
    static_assert(!staged_entry_layer(false, true, false), "entry off: a fused-only layer keeps the FP16 per-expert path");
    static_assert(staged_entry_layer(false, true, true), "entry on: a fused-only layer takes the entry");
    // a layer on the FP16 path (neither tile nor fused kernels) never takes it
    static_assert(!staged_entry_layer(false, false, false), "FP16-path layer");
    static_assert(!staged_entry_layer(false, false, true), "FP16-path layer, entry on: still the FP16 path");
    if (failed) {
        std::fprintf(stderr, "staged_fo_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("staged_fo_rule_test: OK\n");
    return 0;
}
