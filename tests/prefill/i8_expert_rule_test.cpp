// i8_expert_rule (include/strata/kernels/i8_expert_rule.hpp): when a prompt layer's expert groups take our own int8 kernels (native_expert_i8).
// Nothing set: the HIP cards gfx1030 / gfx1031 / gfx1032 take them (groups averaging 256 entries or more, the layout and the device allowing it); every
// other card keeps MMQ.  STRATA_EXPERT_I8=1 asks for them on any card, =0 goes back to the old kernels on the class.  No GPU, no model.
#include <cstdio>

#include "strata/kernels/i8_expert_rule.hpp"

using namespace strata::kernels;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the state of the variable
    CHECK(i8_switch_parse(nullptr) == -1);
    CHECK(i8_switch_parse("") == -1);
    CHECK(i8_switch_parse("0") == 0);
    CHECK(i8_switch_parse("1") == 1);
    CHECK(i8_switch_parse("yes") == 1);
    CHECK(i8_switch_parse("0x") == 0);
    // the class: exactly the three RDNA2 targets, with or without the feature suffix hipGetDeviceProperties adds
    CHECK(gfx_arch_is_gfx103x_i8("gfx1030"));
    CHECK(gfx_arch_is_gfx103x_i8("gfx1031"));
    CHECK(gfx_arch_is_gfx103x_i8("gfx1032"));
    CHECK(gfx_arch_is_gfx103x_i8("gfx1031:sramecc-:xnack-"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1034"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1035"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1036"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1100"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1151"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1201"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx906"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx10310"));
    CHECK(!gfx_arch_is_gfx103x_i8(""));
    CHECK(!gfx_arch_is_gfx103x_i8(nullptr));   // a CUDA card
    // nothing set: the class takes them from 256 entries per group, smaller groups and every other card keep today's path
    CHECK(i8_expert_takes(-1, "gfx1031", 256, true));
    CHECK(i8_expert_takes(-1, "gfx1030", 4096, true));
    CHECK(i8_expert_takes(-1, "gfx1032:sramecc-:xnack-", 300, true));
    CHECK(!i8_expert_takes(-1, "gfx1031", 255, true));
    CHECK(!i8_expert_takes(-1, "gfx1031", 0, true));
    CHECK(!i8_expert_takes(-1, "gfx1031", 4096, false));   // a format or a size the kernel does not cover, or a card whose shared memory is too small
    CHECK(!i8_expert_takes(-1, "gfx1201", 4096, true));    // RDNA4: MMQ / the fused kernels
    CHECK(!i8_expert_takes(-1, "gfx1100", 4096, true));
    CHECK(!i8_expert_takes(-1, "gfx1034", 4096, true));
    CHECK(!i8_expert_takes(-1, nullptr, 4096, true));      // CUDA
    CHECK(!i8_expert_takes(-1, "", 4096, true));
    // =0: the way back, on the class and everywhere
    CHECK(!i8_expert_takes(0, "gfx1031", 4096, true));
    CHECK(!i8_expert_takes(0, "gfx1030", 100000, true));
    CHECK(!i8_expert_takes(0, nullptr, 4096, true));
    // =1: any card that the kernel covers, same size floor
    CHECK(i8_expert_takes(1, "gfx1201", 256, true));
    CHECK(i8_expert_takes(1, nullptr, 4096, true));
    CHECK(i8_expert_takes(1, "gfx1031", 256, true));
    CHECK(!i8_expert_takes(1, "gfx1201", 255, true));
    CHECK(!i8_expert_takes(1, "gfx1201", 4096, false));
    // which of the two reasons: the class default only when the variable is unset
    CHECK(i8_expert_class_default(-1, "gfx1031"));
    CHECK(!i8_expert_class_default(1, "gfx1031"));
    CHECK(!i8_expert_class_default(0, "gfx1031"));
    CHECK(!i8_expert_class_default(-1, "gfx1201"));
    CHECK(!i8_expert_class_default(-1, nullptr));
    // the few-rows route: the limit (STRATA_I8_FEWROWS: not set = the class default, 0 = off, N = N rows per expert)
    CHECK(i8_fewrows_limit(nullptr, 10) == 10);
    CHECK(i8_fewrows_limit("", 8) == 8);
    CHECK(i8_fewrows_limit("0", 10) == 0);
    CHECK(i8_fewrows_limit("0x", 10) == 0);
    CHECK(i8_fewrows_limit("12", 10) == 12);
    CHECK(i8_fewrows_limit("6", 8) == 6);
    CHECK(i8_fewrows_limit("yes", 10) == 10);
    CHECK(i8_fewrows_limit("-3", 8) == 8);
    CHECK(i8_fewrows_limit("99999999999", 10) > 100000);   // a huge number does not wrap to a small one
    // the class default: the P100 (6.0) 6, every other card 7 (6.1, 7.0, 7.5, 8.6, 12.0, a HIP card with no capability at all)
    CHECK(i8_fewrows_class_default(6, 0) == 6);
    CHECK(i8_fewrows_class_default(6, 1) == 7);
    CHECK(i8_fewrows_class_default(7, 0) == 7);
    CHECK(i8_fewrows_class_default(8, 6) == 7);
    CHECK(i8_fewrows_class_default(12, 0) == 7);
    CHECK(i8_fewrows_class_default(0, 0) == 7);
    // the decision: rows strictly under limit x experts go to the grouped kernels (512 experts: 5,119 rows is under 10 per expert, 5,120 is 10)
    CHECK(i8_fewrows_takes(10, 5119, 512));
    CHECK(!i8_fewrows_takes(10, 5120, 512));
    CHECK(!i8_fewrows_takes(10, 20480, 512));
    CHECK(i8_fewrows_takes(8, 4095, 512));
    CHECK(!i8_fewrows_takes(8, 4096, 512));
    CHECK(i8_fewrows_takes(10, 2560, 400));      // 256 tokens x 10 over the 400 experts that got rows: 6.4 per expert
    CHECK(!i8_fewrows_takes(0, 100, 512));       // the route is off
    CHECK(!i8_fewrows_takes(10, 100, 0));        // no experts: nothing to decide
    CHECK(!i8_fewrows_takes(-1, 100, 512));
    // the 8-entry slices: not set (-1) = tile 2 only, 1 = tiles 2-4, 0 = none; tile 8 (the 128-entry kernel) never has them
    CHECK(i8_n8_takes(-1, 2));
    CHECK(!i8_n8_takes(-1, 3));
    CHECK(!i8_n8_takes(-1, 4));
    CHECK(!i8_n8_takes(-1, 8));
    CHECK(i8_n8_takes(1, 2));
    CHECK(i8_n8_takes(1, 3));
    CHECK(i8_n8_takes(1, 4));
    CHECK(!i8_n8_takes(1, 8));
    CHECK(!i8_n8_takes(0, 2));
    CHECK(!i8_n8_takes(0, 4));
    CHECK(i8_n8_takes(i8_switch_parse(nullptr), 2));
    CHECK(i8_n8_takes(i8_switch_parse(""), 2));
    CHECK(!i8_n8_takes(i8_switch_parse("0"), 2));
    CHECK(i8_n8_takes(i8_switch_parse("1"), 4));
    if (failed) {
        std::fprintf(stderr, "i8_expert_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("i8_expert_rule_test passed\n");
    return 0;
}
