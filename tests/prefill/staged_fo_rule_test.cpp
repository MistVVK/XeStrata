// staged_fo_switch / staged_entry_layer (include/strata/prefill/staged_fo_rule.hpp): the staged group entry takes a fused-only layer only with STRATA_PF_STAGED_FO=1.
// No GPU, no model.
#include <cstdio>

#include "strata/prefill/staged_fo_rule.hpp"

using strata::prefill::staged_entry_layer;
using strata::prefill::staged_fo_switch;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the switch: only "1" is on
    CHECK(!staged_fo_switch(nullptr));
    CHECK(!staged_fo_switch(""));
    CHECK(!staged_fo_switch("0"));
    CHECK(!staged_fo_switch("11"));
    CHECK(!staged_fo_switch("true"));
    CHECK(staged_fo_switch("1"));
    // the layer rule: a layer with an MMQ tile is the entry's earlier rule, whatever the switch says
    static_assert(staged_entry_layer(true, false, false), "an MMQ layer keeps the entry");
    static_assert(staged_entry_layer(true, false, true), "an MMQ layer keeps the entry");
    static_assert(staged_entry_layer(true, true, false), "an MMQ layer keeps the entry");
    // a fused-only layer only behind the switch
    static_assert(!staged_entry_layer(false, true, false), "default off: a fused-only layer keeps the FP16 per-expert path");
    static_assert(staged_entry_layer(false, true, true), "switch on: a fused-only layer takes the entry");
    // a layer on the FP16 path (neither tile nor fused kernels) never takes it
    static_assert(!staged_entry_layer(false, false, false), "FP16-path layer");
    static_assert(!staged_entry_layer(false, false, true), "FP16-path layer, switch on: still the FP16 path");
    if (failed) {
        std::fprintf(stderr, "staged_fo_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("staged_fo_rule_test: OK\n");
    return 0;
}
