// win_hip_risky (include/strata/core/win_hip_risky.hpp): --adapt-async 1 and STRATA_ADAPT_OVERLAP=1 are refused on Windows HIP unless STRATA_WIN_HIP_RISKY=1.
// Both platforms, with and without the override, and the two lines. Host only: no GPU, no OS call.
#include <cstdio>
#include <string>

#include "strata/core/win_hip_risky.hpp"

using namespace strata::core;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // Windows + HIP: refused, unless the override is set
    CHECK(win_hip_risky_decide(true, true, true, nullptr) == WinHipRisky::Refuse);
    CHECK(win_hip_risky_decide(true, true, true, "") == WinHipRisky::Refuse);
    CHECK(win_hip_risky_decide(true, true, true, "0") == WinHipRisky::Refuse);
    CHECK(win_hip_risky_decide(true, true, true, "1") == WinHipRisky::AllowWithWarning);
    CHECK(win_hip_risky_decide(true, true, true, "yes") == WinHipRisky::AllowWithWarning);
    // not asked: nothing, with or without the override
    CHECK(win_hip_risky_decide(false, true, true, nullptr) == WinHipRisky::Allow);
    CHECK(win_hip_risky_decide(false, true, true, "1") == WinHipRisky::Allow);
    // Linux HIP, Windows CUDA / SYCL, Linux CUDA: no change, with or without the override
    CHECK(win_hip_risky_decide(true, false, true, nullptr) == WinHipRisky::Allow);
    CHECK(win_hip_risky_decide(true, false, true, "1") == WinHipRisky::Allow);
    CHECK(win_hip_risky_decide(true, true, false, nullptr) == WinHipRisky::Allow);
    CHECK(win_hip_risky_decide(true, true, false, "1") == WinHipRisky::Allow);
    CHECK(win_hip_risky_decide(true, false, false, nullptr) == WinHipRisky::Allow);
    // the lines
    CHECK(win_hip_refused_line("--adapt-async 1") ==
          "strata: --adapt-async 1 is off on Windows with an AMD card: it reset the display driver there during long answers. STRATA_WIN_HIP_RISKY=1 turns it on anyway.");
    CHECK(win_hip_refused_line("STRATA_ADAPT_OVERLAP=1") ==
          "strata: STRATA_ADAPT_OVERLAP=1 is off on Windows with an AMD card: it ended the first request with a GPU fault there, and it was only checked on NVIDIA cards. "
          "STRATA_WIN_HIP_RISKY=1 turns it on anyway.");
    CHECK(win_hip_warning_line("--adapt-async 1") ==
          "strata: STRATA_WIN_HIP_RISKY=1: --adapt-async 1 is on as asked, on Windows with an AMD card; it has reset the display driver or ended requests with a GPU fault there.");
    if (failed) { std::fprintf(stderr, "%d failed\n", failed); return 1; }
    std::printf("win_hip_risky_test: all ok\n");
    return 0;
}
