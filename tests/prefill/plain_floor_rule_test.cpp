// share_limit_raises_floor (include/strata/prefill/plain_floor_rule.hpp): the CPU share's limit raises the chunk floor everywhere except for a non-native pack on the fused kernels in a build
// without MMQ, where the share has nothing to take.  No GPU, no model.
#include <cstdio>
#include <initializer_list>

#include "strata/prefill/plain_floor_rule.hpp"

using strata::prefill::plain_floor_off;
using strata::prefill::share_limit_raises_floor;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the way back: only "0"
    CHECK(!plain_floor_off(nullptr));
    CHECK(!plain_floor_off(""));
    CHECK(!plain_floor_off("00"));
    CHECK(!plain_floor_off("1"));
    CHECK(!plain_floor_off("off"));
    CHECK(plain_floor_off("0"));
    // every other pack / build keeps the share's limit, whatever the switch says
    for (const char* env : {(const char*) nullptr, "", "0", "1", "x"})
        CHECK(share_limit_raises_floor(false, env));
    // the plain pack in a build without MMQ: the share's limit does not raise the floor ...
    CHECK(!share_limit_raises_floor(true, nullptr));
    CHECK(!share_limit_raises_floor(true, ""));
    CHECK(!share_limit_raises_floor(true, "1"));
    CHECK(!share_limit_raises_floor(true, "yes"));
    // ... unless the way back is asked for
    CHECK(share_limit_raises_floor(true, "0"));
    if (failed == 0) std::printf("plain_floor_rule_test: ok\n");
    return failed == 0 ? 0 : 1;
}
