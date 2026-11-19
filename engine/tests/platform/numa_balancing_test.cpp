// The NUMA-balancing opt-out's decision (include/strata/platform/numa_balancing.hpp): no GPU, no NUMA box needed.
#include <cstdio>

#include "strata/platform/numa_balancing.hpp"

using namespace strata::platform::numa_balancing;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // balancing on (1, or 2 = memory tiering mode) and the default policy: set MPOL_LOCAL
    CHECK(want_local(1, kMpolDefault, false));
    CHECK(want_local(2, kMpolDefault, false));
    // balancing off, or unreadable (no /proc, not Linux): nothing to do
    CHECK(!want_local(0, kMpolDefault, false));
    CHECK(!want_local(-1, kMpolDefault, false));
    // a policy the user chose (numactl: 1 = preferred, 2 = bind, 3 = interleave; or MPOL_LOCAL already) is kept
    CHECK(!want_local(1, 1, false));
    CHECK(!want_local(1, 2, false));
    CHECK(!want_local(1, 3, false));
    CHECK(!want_local(1, kMpolLocal, false));
    CHECK(!want_local(1, -1, false));       // the policy could not be read
    // STRATA_NUMA_BALANCING=keep
    CHECK(!want_local(1, kMpolDefault, true));
    if (failed) return 1;
    std::printf("numa_balancing_test: ok\n");
    return 0;
}
