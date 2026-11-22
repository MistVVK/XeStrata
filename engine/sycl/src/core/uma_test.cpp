// sycl/src/core/uma_test.cpp - #6448: the room an integrated GPU's cache is sized from (strata/core/uma.hpp): MemAvailable
// plus the GPU driver's pooled pages (GPUReclaim) less the headroom, capped at the device's memory, and STRATA_UMA's
// values.  Host only: no queue and no GPU.
//   uma_test
#include "strata/core/uma.hpp"

#include <cstdio>

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

int main() {
    using namespace strata::core;
    constexpr uint64_t GiB = 1ull << 30;

    uint64_t avail = 0, pooled = 0;
    const char* meminfo[] = {"MemTotal:       65218560 kB\n", "MemFree:         1048576 kB\n",
                             "MemAvailable:   20971520 kB\n", "Cached:          8388608 kB\n",
                             "GPUActive:      41943040 kB\n", "GPUReclaim:     25165824 kB\n"};
    bool got = false;
    for (const char* line : meminfo) got = uma_meminfo_line(line, avail, pooled) || got;
    check(got && avail == 20 * GiB && pooled == 24 * GiB, "MemAvailable and GPUReclaim are read; MemFree, Cached and GPUActive are not");

    uint64_t a2 = 0, p2 = 0;
    got = false;
    const char* old_kernel[] = {"MemFree:  1048576 kB\n", "MemAvailable: 20971520 kB\n"};
    for (const char* line : old_kernel) got = uma_meminfo_line(line, a2, p2) || got;
    check(got && a2 == 20 * GiB && p2 == 0, "a kernel without GPUReclaim: nothing pooled");
    check(uma_meminfo_line("MemAvailable: 0 kB\n", a2, p2) && a2 == 0, "MemAvailable 0 is still read");

    check(uma_room(20 * GiB, 24 * GiB, 6 * GiB, 0) == 38 * GiB, "room = MemAvailable + GPUReclaim - headroom (no cap)");
    check(uma_room(20 * GiB, 0, 6 * GiB, 0) == 14 * GiB, "GPUReclaim 0 (kernel 7.0, #2412): MemAvailable - headroom");
    check(uma_room(20 * GiB, 24 * GiB, 6 * GiB, 30 * GiB) == 30 * GiB, "capped at the device's global memory");
    check(uma_room(4 * GiB, 1 * GiB, 6 * GiB, 30 * GiB) == 0, "a headroom larger than the RAM leaves nothing, not a wrap-around");
    check(uma_room(6 * GiB, 0, 6 * GiB, 0) == 0, "exactly the headroom: nothing");

    check(uma_override(nullptr) == -1, "STRATA_UMA unset: the device decides");
    check(uma_override("") == -1, "STRATA_UMA empty: the device decides");
    check(uma_override("1") == 1, "STRATA_UMA=1: integrated");
    check(uma_override("0") == 0, "STRATA_UMA=0: a card");

    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
