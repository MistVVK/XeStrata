// plain_floor_rule.hpp: the long-prompt floor of the plain pack by card class and share state.  No GPU, no model.
// Classes: plain pack (build without MMQ, non-native pack) or not; discrete CUDA card (answered, not integrated) or integrated / not asked / HIP; share armed (limit 3,072 or 1,024) or not; switch.
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include "strata/prefill/plain_floor_rule.hpp"

using strata::prefill::floor_with_share;
using strata::prefill::kPlainDiscreteFloor;
using strata::prefill::plain_floor_off;
using strata::prefill::plain_takes_discrete_floor;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    constexpr int64_t kFused = 64;      // stream_floor_rule.hpp kFusedOnlyFloor, the plain pack's default floor
    constexpr int64_t kDefault = 1024;  // kStreamFloorDefault
    // the way back: only "0"
    CHECK(!plain_floor_off(nullptr));
    CHECK(!plain_floor_off(""));
    CHECK(!plain_floor_off("00"));
    CHECK(!plain_floor_off("1"));
    CHECK(!plain_floor_off("off"));
    CHECK(plain_floor_off("0"));
    // class: only (plain pack) x (discrete CUDA card) x (switch not "0") takes the discrete floor
    for (const char* env : {(const char*) nullptr, "", "1", "x"}) {
        CHECK(plain_takes_discrete_floor(true, true, env));
        CHECK(!plain_takes_discrete_floor(true, false, env));    // integrated / not asked yet / HIP: unchanged
        CHECK(!plain_takes_discrete_floor(false, true, env));    // a native pack or a build with MMQ: unchanged
        CHECK(!plain_takes_discrete_floor(false, false, env));
    }
    for (const bool plain : {false, true})
        for (const bool disc : {false, true}) CHECK(!plain_takes_discrete_floor(plain, disc, "0"));   // the way back, every class
    // the answer: the plain pack on a discrete card, share armed (3,072 / 1,024) or not, takes the discrete floor
    CHECK(floor_with_share(true, 3072, kFused) == kPlainDiscreteFloor);
    CHECK(floor_with_share(true, 1024, kFused) == kPlainDiscreteFloor);
    CHECK(floor_with_share(true, 0, kFused) == kPlainDiscreteFloor);
    CHECK(kPlainDiscreteFloor == 1536);   // the measured constant (RTX 5070); above the default floor and the plain pack's 64, below the share's limit
    CHECK(kPlainDiscreteFloor > kDefault && kPlainDiscreteFloor > kFused && kPlainDiscreteFloor < 3072);
    // a default floor above the constant is never lowered
    CHECK(floor_with_share(true, 0, 4096) == 4096);
    CHECK(floor_with_share(true, 3072, 4096) == 4096);
    // every other class keeps the old answer: the share's limit when armed, else the default floor
    CHECK(floor_with_share(false, 3072, kFused) == 3072);      // integrated GPU / unknown card / HIP, plain pack, share armed
    CHECK(floor_with_share(false, 1024, kFused) == 1024);
    CHECK(floor_with_share(false, 0, kFused) == kFused);       // integrated GPU, plain pack, no share: 64 stays
    CHECK(floor_with_share(false, 3072, kDefault) == 3072);    // native pack, share armed
    CHECK(floor_with_share(false, 0, kDefault) == kDefault);   // native pack, no share
    CHECK(floor_with_share(false, -1, kDefault) == kDefault);  // no share (negative = not armed)
    if (failed == 0) std::printf("plain_floor_rule_test: ok\n");
    return failed == 0 ? 0 : 1;
}
