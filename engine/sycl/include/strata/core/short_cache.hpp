// sycl/include/strata/core/short_cache.hpp - sizing a start that is short of VRAM (owner rule: a long context and requests at once never shrink the context or
// stop a start that can run; they run slower).  Pure arithmetic, no device.
//
//  * size_short_cache: the expert cache's sizing left no working cache.  The speed knobs go in this order, and each only as far as needed: the prompt path's own
//    chunk (its buffers are 160 + 0.66 MiB a token: 1182 MiB at 2048), then a reserve that was asked for (down to the small-card reserve).  The person's own chunk
//    (--prefill N) is left alone.  The result is the largest chunk (then reserve) that leaves `work_slots`, else the smallest chunk and reserve (whatever slots that leaves).
//  * pick_kv_resident: the largest KV window (65536, 32768, 20480 cells a layer) whose sessions fit; 20480 (the engine's smallest) when none does.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::core {

struct ShortCache {
    int64_t chunk = 0;
    int64_t reserve_mib = 0;
    int64_t slots = 0;
};

inline int64_t owned_prompt_mib(int64_t chunk) { return 160 + (chunk * 680) / 1024; }

/// Slots the cache gets when the prompt path allocates its own `chunk`-token buffers.
inline int64_t slots_with_own_chunk(int64_t free_b, int64_t reserve_mib, int64_t chunk, int64_t draft_head_b, int64_t pipe_first_b, int64_t blob_b, int64_t profile_cap) {
    int64_t n = (free_b - (((reserve_mib + owned_prompt_mib(chunk)) << 20) + draft_head_b + pipe_first_b)) / blob_b;
    if (profile_cap > 0) n = std::min(n, profile_cap);
    return std::max<int64_t>(n, 0);
}

inline ShortCache size_short_cache(int64_t free_b, int64_t reserve_mib, bool reserve_given, int64_t small_reserve_mib, int64_t chunk, bool chunk_given, int64_t min_chunk,
                                   int64_t work_slots, int64_t draft_head_b, int64_t pipe_first_b, int64_t blob_b, int64_t profile_cap) {
    auto slots_at = [&](int64_t c, int64_t r) { return slots_with_own_chunk(free_b, r, c, draft_head_b, pipe_first_b, blob_b, profile_cap); };
    while (!chunk_given && chunk > min_chunk && slots_at(chunk, reserve_mib) < work_slots) chunk /= 2;
    if (slots_at(chunk, reserve_mib) < work_slots && reserve_given && reserve_mib > small_reserve_mib) {
        int64_t r = reserve_mib;
        while (r > small_reserve_mib && slots_at(chunk, r) < work_slots) r = std::max<int64_t>(small_reserve_mib, r - 50);
        reserve_mib = r;
    }
    return ShortCache{chunk, reserve_mib, slots_at(chunk, reserve_mib)};
}

/// `need_at(window)` is the bytes the sessions and everything after them need with that KV window (0: the whole context in VRAM); `free_b` what the card has.
/// Returns 0 when the whole context fits (no streaming), else the largest window that fits, else the smallest the engine takes (20480).
template <class NeedAt>
inline int64_t pick_kv_resident(int64_t ctx, int64_t free_b, NeedAt need_at) {
    if (need_at((int64_t) 0) <= free_b) return 0;
    for (const int64_t w : {(int64_t) 65536, (int64_t) 32768, (int64_t) 20480})
        if (w < ctx && need_at(w) <= free_b) return w;
    return 20480;
}

}  // namespace strata::core
