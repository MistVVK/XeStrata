#pragma once
// The size of the park pools (the segment pool of conversation_segment_pool.hpp and the ready checkpoints of the checkpoint pool,
// STRATA_CKPT_POOL_READY), from the conversation cache's own budget.  Both are memory held idle between parks so a park does not
// pay the first touch of fresh pages; both count against --conversation-cache-mib and both are given back before a parked
// conversation is evicted and before the RAM floor (--conversation-cache-min-free-mib) is reached (generate.cpp park_source,
// ConversationCache::make_room).  A cache that is off (budget 0) gets no pool.
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::core {

struct ParkPoolPlan {
    size_t segpool_mib = 0;   ///< the arena for the K/V buffers of parked conversations (0: none)
    size_t ckpt_ready = 0;    ///< checkpoints kept ready for the next save (0: none ahead of time)
};

constexpr size_t kSegpoolMaxMib = 2048;          ///< the arena never exceeds 2 GiB (l620, 11-12: a 55K-token park 283 ms with 1 GiB, 117 ms with 2 GiB)
constexpr size_t kSegpoolMinMib = 128;           ///< below this an arena does not hold even an 8K image: none
constexpr size_t kCkptReadyMax = 4;              ///< a park needs the live state plus one copy per checkpoint (up to 3 seen at 55K): 4

/// `cache_mib`: the conversation cache's budget (--conversation-cache-mib, or the automatic size);
/// `ckpt_bytes`: one ready checkpoint (the running state of the model: ~112 MB on Qwen3.8-Flash-Next).
/// segpool = min(2 GiB, a quarter of the cache) in whole 64 MiB steps, none under 128 MiB;
/// ckpt_ready = as many checkpoints as fit in a twelfth of the cache, at most 4.
/// Together they hold at most a third of the cache idle (30% at the usual 7-8 GiB caches); idle memory is given back first.
inline ParkPoolPlan park_pool_plan(uint64_t cache_mib, uint64_t ckpt_bytes) {
    ParkPoolPlan p;
    if (cache_mib == 0) return p;
    const size_t seg = std::min<uint64_t>(kSegpoolMaxMib, cache_mib / 4) / 64 * 64;
    p.segpool_mib = seg >= kSegpoolMinMib ? seg : 0;
    if (ckpt_bytes > 0) p.ckpt_ready = (size_t) std::min<uint64_t>(kCkptReadyMax, (cache_mib << 20) / 12 / ckpt_bytes);
    return p;
}

/// Pool upkeep (the segment pool's background fill, the checkpoint pool's background refill) runs only while the engine is idle:
/// page faults and madvise calls of a background thread must not overlap a request's GPU work (prompt read, decode).  A request
/// that finds a pool short falls back to fresh memory; it never waits for the upkeep.  `engine_busy` is the serve loop's
/// "a request is in flight" flag; the answer is how many of `wanted` bytes (or vectors) the upkeep may make now.
inline size_t park_pool_permit(bool engine_busy, size_t wanted) { return engine_busy ? 0 : wanted; }

}  // namespace strata::core
