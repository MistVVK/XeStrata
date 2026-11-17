// stream_floor_default (include/strata/prefill/stream_floor_rule.hpp): the chunk size from which a prompt chunk takes the long-prompt path.
// Only the measured class (Windows gfx1151) with a layer set that is fused-only on every layer AND without the staged fused entry leaves the 1,024 default.  No GPU, no model.
#include <cstdio>

#include "strata/prefill/stream_floor_rule.hpp"

using strata::prefill::kFusedOnlyFloor;
using strata::prefill::kStreamFloorDefault;
using strata::prefill::kTrellisCudaFloor;
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
    // a build without MMQ on a gfx12 card with a native pack of trellis pairs on the fused kernels (Gyro-S on the R9700): the floor is 64 whatever the staged entry and the class say
    CHECK(stream_floor_default(false, false, true, false, true) == kFusedOnlyFloor);
    CHECK(stream_floor_default(false, false, false, false, true) == kFusedOnlyFloor);
    CHECK(stream_floor_default(true, true, true, false, true) == kFusedOnlyFloor);
    // a build without MMQ on a CUDA card on which the fused trellis kernels run (Gyro-S on an RTX A6000, sm_86 with 84 SMs): the floor is kTrellisCudaFloor whatever the class and the staged entry say
    CHECK(stream_floor_default(false, false, false, false, false, true) == kTrellisCudaFloor);
    CHECK(stream_floor_default(false, false, true, false, false, true) == kTrellisCudaFloor);
    CHECK(stream_floor_default(true, true, true, false, false, true) == kTrellisCudaFloor);
    CHECK(kTrellisCudaFloor >= 1 && kTrellisCudaFloor <= kStreamFloorDefault);
    // the gfx12 rows stay as gated: the CUDA flag adds nothing to them (the two flags are never both true in one build)
    CHECK(stream_floor_default(false, false, false, false, true, false) == kFusedOnlyFloor);
    CHECK(stream_floor_default(false, false, false, false, true, true) == kFusedOnlyFloor);
    // no CUDA flag (a build with MMQ, a card whose trellis kernels are off, another format set): every row above stands
    CHECK(stream_floor_default(false, false, true, false, false, false) == kStreamFloorDefault);
    CHECK(stream_floor_default(true, false, false, false, false, false) == kStreamFloorDefault);
    // without the flag (a build with MMQ, another card, another format set) the rows above stand
    CHECK(stream_floor_default(false, false, true, false, false) == kStreamFloorDefault);
    CHECK(stream_floor_default(false, true, true, false, false) == kStreamFloorDefault);
    // Windows gfx1151, an all-trellis set on the fused kernels (STRATA_PF_TQ_G1151 on, a build without MMQ): the gfx12 floor, with the staged entry on or off
    CHECK(stream_floor_default(true, true, true, false, false, false, true) == kFusedOnlyFloor);
    CHECK(stream_floor_default(true, true, false, false, false, false, true) == kFusedOnlyFloor);
    // the same class and set with the flag false (switch off, Linux, a build with MMQ, another format set): the rows above stand
    CHECK(stream_floor_default(true, true, true, false, false, false, false) == kStreamFloorDefault);
    CHECK(stream_floor_default(true, false, false, false, false, false, false) == kStreamFloorDefault);
    CHECK(stream_floor_default(false, false, false, false, false, false, false) == kStreamFloorDefault);
    // the gfx12 and CUDA rows are not changed by the new flag's default
    CHECK(stream_floor_default(false, false, false, false, true, false, false) == kFusedOnlyFloor);
    CHECK(stream_floor_default(false, false, false, false, false, true, false) == kTrellisCudaFloor);
    if (failed) {
        std::fprintf(stderr, "stream_floor_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("stream_floor_rule_test: OK\n");
    return 0;
}
