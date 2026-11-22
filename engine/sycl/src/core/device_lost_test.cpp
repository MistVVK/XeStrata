// sycl/src/core/device_lost_test.cpp - #6445: host-only unit test for Level Zero DEVICE_LOST detection & diagnostics.
#include "strata/core/device_lost.hpp"

#include <cstdio>
#include <cstring>
#include <string>

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-80s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

int main() {
    check(strata::is_device_lost("level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)"),
          "matches UR_RESULT_ERROR_DEVICE_LOST");
    check(strata::is_device_lost("ZE_RESULT_ERROR_DEVICE_LOST"),
          "matches ZE_RESULT_ERROR_DEVICE_LOST");
    check(strata::is_device_lost("level_zero backend failed with error: 20"),
          "matches level_zero error: 20");
    check(!strata::is_device_lost("level_zero backend failed with error: 10 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)"),
          "does not match OUT_OF_DEVICE_MEMORY");
    check(!strata::is_device_lost("verify: timed out at layer 12"),
          "does not match generic verify error");

    const std::string hint = strata::device_lost_hint();
    check(hint.find("preempt_timeout_us") != std::string::npos, "hint names preempt_timeout_us");
    check(hint.find("xpu-smi") != std::string::npos, "hint names xpu-smi");
    check(hint.find("docs/INTEL.md") != std::string::npos, "hint points to docs/INTEL.md");

    check(!strata::device_is_lost(), "device_is_lost initially false");
    strata::mark_device_lost();
    check(strata::device_is_lost(), "device_is_lost true after mark_device_lost()");

    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
