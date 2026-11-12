// stream_floor_default (include/strata/prefill/stream_floor_rule.hpp): the chunk size from which a prompt chunk takes the long-prompt path.
// Only the measured class (Windows gfx1151) with a layer set that is fused-only on every layer leaves the 1,024 default.  No GPU, no model.
#include <cstdio>

#include "strata/prefill/stream_floor_rule.hpp"

using strata::prefill::kFusedOnlyFloor;
using strata::prefill::kStreamFloorDefault;
using strata::prefill::stream_floor_default;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    static_assert(kStreamFloorDefault == 1024, "the default floor is the discrete cards' measured 1,024");
    static_assert(kFusedOnlyFloor >= 1 && kFusedOnlyFloor < kStreamFloorDefault, "the fused-only floor is below the default");
    CHECK(stream_floor_default(true, true) == kFusedOnlyFloor);        // Windows gfx1151, every layer fused-only
    CHECK(stream_floor_default(true, false) == kStreamFloorDefault);   // a layer on MMQ or the FP16 path: unchanged
    CHECK(stream_floor_default(false, true) == kStreamFloorDefault);   // another card class (not measured): unchanged
    CHECK(stream_floor_default(false, false) == kStreamFloorDefault);
    if (failed) {
        std::fprintf(stderr, "stream_floor_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("stream_floor_rule_test: OK\n");
    return 0;
}
