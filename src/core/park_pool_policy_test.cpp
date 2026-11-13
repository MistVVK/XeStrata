// The park pools' sizes from the conversation cache's budget (park_pool_policy.hpp), with the cache sizes small machines get.
#include "strata/core/conversation_memory.hpp"
#include "strata/core/park_pool_policy.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
constexpr uint64_t MiB = uint64_t(1) << 20, GiB = uint64_t(1) << 30;
constexpr uint64_t kCkpt = 112 * MiB;   // one ready checkpoint on Qwen3.8-Flash-Next (the running state, ~118 MB)
}

int main() {
    // no cache, no pool
    check(park_pool_plan(0, kCkpt).segpool_mib == 0 && park_pool_plan(0, kCkpt).ckpt_ready == 0, "a cache that is off gets no pool");
    // the sizes at the cache sizes the automatic rule gives (free RAM after the model is loaded, floor 2560 MiB)
    struct Row { uint64_t free_gib; int64_t cache; size_t seg, ready; };
    const Row rows[] = {
        {4, 0, 0, 0},         // under ~5 GiB free: the cache is off, so no pool
        {8, 2048, 512, 1},    // a 16 GB machine with ~8 GiB free
        {12, 3840, 960, 2},
        {20, 7168, 1792, 4},  // a 32 GB machine with ~20 GiB free
        {40, 8192, 2048, 4},  // a 64 GB machine with 40 GiB or more free
        {200, 8192, 2048, 4}, // a big server: the cache cap is 8 GiB, so the pools stay at 2 GiB and 4
    };
    for (const Row& r : rows) {
        const int64_t cache = auto_conversation_cache_mib(r.free_gib * GiB, 2560);
        check(cache == r.cache, "the automatic cache size for this free RAM");
        const ParkPoolPlan p = park_pool_plan((uint64_t) cache, kCkpt);
        check(p.segpool_mib == r.seg && p.ckpt_ready == r.ready, "the pools for this cache");
        // what is held idle at most: the arena plus the ready checkpoints, never more than a third of the cache (34% with rounding)
        check(cache == 0 || (p.segpool_mib * MiB + p.ckpt_ready * kCkpt) * 100 <= (uint64_t) cache * MiB * 34, "held idle: at most 34% of the cache");
    }
    // small explicit budgets
    check(park_pool_plan(512, kCkpt).segpool_mib == 128 && park_pool_plan(512, kCkpt).ckpt_ready == 0, "a 512 MiB cache: a 128 MiB arena (a quarter), no ready checkpoint (a twelfth is 43 MiB)");
    check(park_pool_plan(448, kCkpt).segpool_mib == 0, "a 448 MiB cache: a quarter is 112 MiB, under the minimum: no arena");
    check(park_pool_plan(1024, kCkpt).segpool_mib == 256 && park_pool_plan(1024, kCkpt).ckpt_ready == 0, "a 1 GiB cache: a 256 MiB arena, no ready checkpoint");
    check(park_pool_plan(1536, kCkpt).ckpt_ready == 1 && park_pool_plan(1536, kCkpt).segpool_mib == 384, "1.5 GiB: one ready checkpoint, a 384 MiB arena");
    check(park_pool_plan(4096, kCkpt).ckpt_ready == 3 && park_pool_plan(4096, kCkpt).segpool_mib == 1024, "4 GiB: 3 ready checkpoints, a 1 GiB arena");
    check(park_pool_plan(1 << 20, kCkpt).segpool_mib == 2048 && park_pool_plan(1 << 20, kCkpt).ckpt_ready == 4, "a huge cache: the caps, 2 GiB and 4");
    check(park_pool_plan(8192, 0).ckpt_ready == 0 && park_pool_plan(8192, 0).segpool_mib == 2048, "an unknown checkpoint size: none ready, the arena as usual");
    // a bigger checkpoint (another model) gets fewer ready ones, never more than 4
    check(park_pool_plan(8192, 400 * MiB).ckpt_ready == 1 && park_pool_plan(8192, 30 * MiB).ckpt_ready == 4, "ready checkpoints scale with their size");
    // the segment pool is a multiple of 64 MiB (granules and 2 MiB alignment hold)
    for (uint64_t c = 1; c < 20000; c += 37) {
        const size_t s = park_pool_plan(c, kCkpt).segpool_mib;
        check(s == 0 || (s % 64 == 0 && s >= kSegpoolMinMib && s <= kSegpoolMaxMib && s <= c / 4), "an arena is whole 64 MiB steps, between 128 MiB and 2 GiB, within a quarter of the cache");
    }
    // upkeep only while the engine is idle
    check(park_pool_permit(true, 64 * MiB) == 0 && park_pool_permit(true, 1) == 0, "a request in flight: the upkeep may make nothing");
    check(park_pool_permit(false, 64 * MiB) == 64 * MiB && park_pool_permit(false, 0) == 0, "idle: the upkeep may make what it wants");
    std::printf("park_pool_policy_test: %d checks passed\n", checks);
    return 0;
}
