// The elastic K/V once the expert cache is known (strata/core/kv_grow_rules.hpp): pools made elastic always grow, from
// the cache's slots only when the cache can give them, else from new VRAM; a grow covers the request, never past the context.
#include "strata/core/kv_grow_rules.hpp"

#include <cstdio>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
}  // namespace

int main() {
    // pools not elastic (no --kv-grow, or it was turned off at init): nothing to grow
    CHECK(!kv_grow_start(false, true, true, true, 9000, 128).on);
    CHECK(!kv_grow_start(false, true, true, true, 9000, 128).from_cache);
    // every condition met: grows from the cache's slots
    CHECK(kv_grow_start(true, true, true, true, 9000, 128).on);
    CHECK(kv_grow_start(true, true, true, true, 9000, 128).from_cache);
    // the refused states: the pools are elastic, so the K/V still grows (new VRAM), the cache gives nothing
    const KvGrowStart floor_hit = kv_grow_start(true, true, true, true, 128, 128);   // --expert-cache 100 -> 128 slots
    CHECK(floor_hit.on && !floor_hit.from_cache);
    CHECK(kv_grow_start(true, true, true, true, 0, 128).on);   // a loan that starts at slot 0
    CHECK(!kv_grow_start(true, true, true, true, 0, 128).from_cache);
    CHECK(kv_grow_start(true, false, true, true, 9000, 128).on);   // no VMM range for the cache
    CHECK(!kv_grow_start(true, false, true, true, 9000, 128).from_cache);
    CHECK(kv_grow_start(true, true, false, true, 9000, 128).on);   // no residency table
    CHECK(!kv_grow_start(true, true, false, true, 9000, 128).from_cache);
    CHECK(kv_grow_start(true, true, true, false, 9000, 128).on);   // no expert source
    CHECK(!kv_grow_start(true, true, true, false, 9000, 128).from_cache);
    CHECK(kv_grow_start(true, true, true, true, 129, 128).from_cache);   // one slot above the floor
    // the grow target: whole steps, covering the request, at most the context (never a smaller context)
    CHECK(kv_grow_target(20256, 8192, 131072) == 24576);
    CHECK(kv_grow_target(16384, 8192, 131072) == 16384);
    CHECK(kv_grow_target(16385, 8192, 131072) == 24576);
    CHECK(kv_grow_target(1, 8192, 131072) == 8192);
    CHECK(kv_grow_target(130000, 8192, 131072) == 131072);
    CHECK(kv_grow_target(131072 + 256, 8192, 131072) == 131072);
    for (int64_t c = 1; c <= 131072; c += 997) {
        const int64_t t = kv_grow_target(c, 8192, 131072);
        CHECK(t >= c && t <= 131072 && t % 8192 == 0);
    }
    for (int64_t c = 1; c <= 100000; c += 991) CHECK(kv_grow_target(c, 8192, 100000) >= c);
    if (failed) {
        std::fprintf(stderr, "kv_grow_rules_test: %d failed\n", failed);
        return 1;
    }
    std::printf("kv_grow_rules_test: ok\n");
    return 0;
}
