// sycl/src/core/verify_env_test.cpp - #6250 / #4698: STRATA_VERIFY_NO_HOST=1 alone gets the device-built plan (strata::no_host_wants_device_plan).  Host only:
// no queue and no GPU.
//   verify_env_test
#include "strata/core/verify_env.hpp"

#include <cstdio>

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

int main() {
    using strata::no_host_wants_device_plan;
    check(no_host_wants_device_plan("1", nullptr), "NO_HOST=1, DEVICE_PLAN not set: the engine sets the device plan (#6250)");
    check(no_host_wants_device_plan("true", nullptr), "NO_HOST=true (any value but 0), DEVICE_PLAN not set: the device plan");
    check(!no_host_wants_device_plan("1", "1"), "both set: nothing to do");
    check(!no_host_wants_device_plan("1", "0"), "DEVICE_PLAN=0 set by the caller: kept");
    check(!no_host_wants_device_plan("0", nullptr), "NO_HOST=0 (an i915 card): nothing");
    check(!no_host_wants_device_plan("", nullptr), "NO_HOST empty: nothing");
    check(!no_host_wants_device_plan(nullptr, nullptr), "neither set: nothing");
    check(!no_host_wants_device_plan(nullptr, "1"), "DEVICE_PLAN alone: nothing");
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
