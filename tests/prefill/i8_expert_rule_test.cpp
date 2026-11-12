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
    if (failed) {
        std::fprintf(stderr, "i8_expert_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("i8_expert_rule_test passed\n");
    return 0;
}
