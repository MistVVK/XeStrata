// sycl/src/core/spin_bound_test.cpp - the choice of the device-spin bound (strata::spin_bound_for, sycl_doorbell.hpp): the override, the
// host-waits flag on a Linux xe card, i915 and unknown drivers, a B-series card with no driver name, and the --spec drafts flag.  A pure
// function: no queue and no GPU.
//   spin_bound_test
#include <sycl/sycl.hpp>
#include "strata/sycl_doorbell.hpp"

#include <cstdio>
#include <string>

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-110s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

int main() {
    using strata::spin_bound_for;
#ifdef STRATA_SYCL_SPIN_MAX
    check(spin_bound_for(0, "xe", false, true, false) == (uint32_t) STRATA_SYCL_SPIN_MAX, "build-fixed bound (-DSTRATA_SYCL_SPIN_MAX) wins over the table");
    check(spin_bound_for(777, "xe", false, true, false) == 777u, "STRATA_SPIN_MAX wins over a build-fixed bound");
#else
    // xe (Linux B-series): short while the host is never asked, long as soon as a stage waits for it
    check(spin_bound_for(0, "xe", false, false, false) == 20000u, "xe, host never asked: 20,000 reads");
    check(spin_bound_for(0, "xe", false, true, false) == 1000000u, "xe, host waits expected: 1,000,000 reads");
    check(spin_bound_for(0, "xe", false, true, true) == 1000000u, "xe: the drafts flag does not change the host-waits answer");
    check(spin_bound_for(0, "xe", true, false, true) == 20000u, "xe: the drafts flag alone leaves the short bound");
    // i915 (Arc A-series) and any other named driver: the CPU computes experts and the GPU waits at every layer
    check(spin_bound_for(0, "i915", false, false, false) == 2000000u, "i915: 2,000,000 reads");
    check(spin_bound_for(0, "i915", false, true, false) == 2000000u, "i915: the host flag does not matter");
    check(spin_bound_for(0, "nvidia", true, false, false) == 2000000u, "another named driver: 2,000,000 reads, even for a B-series name");
    // no sysfs driver name (Windows, OpenCL)
    check(spin_bound_for(0, "", true, false, false) == 20000u, "no driver, B-series: 20,000 reads");
    check(spin_bound_for(0, "", true, true, false) == 20000u, "no driver, B-series: the host flag is not what raises it");
    check(spin_bound_for(0, "", true, false, true) == 2000000u, "no driver, B-series, --spec without MTP: 2,000,000 reads");
    check(spin_bound_for(0, "", false, false, false) == 2000000u, "no driver, not a B-series card: 2,000,000 reads");
#endif
    // the override is a positive number of reads, in every case
    check(spin_bound_for(1234, "xe", false, false, false) == 1234u, "STRATA_SPIN_MAX on xe, host never asked");
    check(spin_bound_for(1234, "xe", false, true, false) == 1234u, "STRATA_SPIN_MAX on xe, host waits");
    check(spin_bound_for(1234, "i915", false, false, false) == 1234u, "STRATA_SPIN_MAX on i915");
    check(spin_bound_for(1234, "", true, false, true) == 1234u, "STRATA_SPIN_MAX with no driver");
    std::printf(g_fail ? "FAILED: %d\n" : "all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
