// share_layer_keeps_own_kernels (include/strata/prefill/share_own_rule.hpp): a layer whose CPU share took experts keeps the staged fused entry and the int8 kernels for the GPU's own
// experts in a build without MMQ, and leaves them (as before) in a build with MMQ.  No GPU, no model.
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include "strata/prefill/share_own_rule.hpp"

using strata::prefill::share_layer_keeps_own_kernels;
using strata::prefill::share_own_off;
using strata::prefill::share_own_switch;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the switch: only "1" is on, only "0" is off
    CHECK(!share_own_switch(nullptr));
    CHECK(!share_own_switch(""));
    CHECK(!share_own_switch("0"));
    CHECK(!share_own_switch("11"));
    CHECK(share_own_switch("1"));
    CHECK(!share_own_off(nullptr));
    CHECK(!share_own_off(""));
    CHECK(!share_own_off("00"));
    CHECK(!share_own_off("1"));
    CHECK(share_own_off("0"));
    // no CPU rows: the rule never blocks, in any build, whatever the switch says
    for (const bool own : {false, true})
        for (const char* env : {(const char*) nullptr, "", "0", "1", "x"})
            CHECK(share_layer_keeps_own_kernels(own, env, 0));
    // CPU rows, switch unset: a build without MMQ keeps its kernels, a build with MMQ leaves them (the old rule, MMQ takes the layer)
    CHECK(share_layer_keeps_own_kernels(true, nullptr, 1));
    CHECK(share_layer_keeps_own_kernels(true, nullptr, 4096));
    CHECK(share_layer_keeps_own_kernels(true, "", 7));
    CHECK(!share_layer_keeps_own_kernels(false, nullptr, 1));
    CHECK(!share_layer_keeps_own_kernels(false, nullptr, 4096));
    CHECK(!share_layer_keeps_own_kernels(false, "", 7));
    // the way back: "0" is the old rule in every build
    CHECK(!share_layer_keeps_own_kernels(true, "0", 1));
    CHECK(!share_layer_keeps_own_kernels(false, "0", 1));
    // the research A/B: "1" is the new rule in every build
    CHECK(share_layer_keeps_own_kernels(true, "1", 1));
    CHECK(share_layer_keeps_own_kernels(false, "1", 1));
    // any other text is not a switch: the build decides
    CHECK(share_layer_keeps_own_kernels(true, "yes", 1));
    CHECK(!share_layer_keeps_own_kernels(false, "yes", 1));
    if (failed == 0) std::printf("share_own_rule_test: ok\n");
    return failed == 0 ? 0 : 1;
}
