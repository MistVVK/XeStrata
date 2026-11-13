// stream_floor_default (include/strata/prefill/stream_floor_rule.hpp): the chunk size from which a prompt chunk takes the long-prompt path.
// Only the measured class (Windows gfx1151) with a layer set that is fused-only on every layer AND without the staged fused entry leaves the 1,024 default.  No GPU, no model.
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
    // the staged fused entry off (STRATA_PF_STAGED_FO=0, or its prerequisites missing): step one's rule
    CHECK(stream_floor_default(true, true, false) == kFusedOnlyFloor);        // Windows gfx1151, every layer fused-only
    CHECK(stream_floor_default(true, false, false) == kStreamFloorDefault);   // a layer on MMQ or the FP16 path: unchanged
    CHECK(stream_floor_default(false, true, false) == kStreamFloorDefault);   // another card class (not measured): unchanged
    CHECK(stream_floor_default(false, false, false) == kStreamFloorDefault);
    // the entry on: pieces below 1,024 tokens take it, the long path from 1,024 up, on every class
    CHECK(stream_floor_default(true, true, true) == kStreamFloorDefault);
    CHECK(stream_floor_default(true, false, true) == kStreamFloorDefault);
    CHECK(stream_floor_default(false, true, true) == kStreamFloorDefault);    // STRATA_PF_STAGED_FO=1 on another class: the floor stays
    CHECK(stream_floor_default(false, false, true) == kStreamFloorDefault);
    // a build without MMQ with a non-native pack on the fused kernels (the Q2_0 pack): no staged entry or int8 path reads its layout, so the floor is 64 on every class, whatever the other flags
    CHECK(stream_floor_default(false, false, false, true) == kFusedOnlyFloor);
    CHECK(stream_floor_default(true, true, true, true) == kFusedOnlyFloor);
    CHECK(stream_floor_default(false, true, true, true) == kFusedOnlyFloor);
    // a build with MMQ, or a pack that is not plain-and-fused, passes false: the rows above are unchanged
    CHECK(stream_floor_default(false, false, false, false) == kStreamFloorDefault);
    CHECK(stream_floor_default(true, true, true, false) == kStreamFloorDefault);
    if (failed) {
        std::fprintf(stderr, "stream_floor_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("stream_floor_rule_test: OK\n");
    return 0;
}
