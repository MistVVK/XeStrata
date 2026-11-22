// hy4_metal_slots.h - resident conversation slots of the Hy4-Flash Metal serve front end (pure parts: KV reuse point, slot choice).
// Included by hy4_metal_serve.h; depends only on the standard library so a probe can include it alone.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

// KV reuse point: longest common prefix of the resident sequence tok[0,fwd) and the new prompt, capped so the last prompt token is recomputed (logits exist).
// Rolling back from fwd to `start` needs the SWA window [start-512, start) intact in the ring: the speculative tail (64) written past fwd must stay within `slack` (= ring - 512) of start.
static inline int hy4_reuse_start(const std::vector<int32_t>& tok, int fwd, const std::vector<int32_t>& ids, int T0, int slack) {
    const int lim = std::min({fwd, T0, (int) tok.size(), (int) ids.size()});
    int L = 0;
    while (L < lim && tok[L] == ids[L]) ++L;
    if (L >= T0) L = T0 - 1;
    if (L < 1) return 0;
    if (fwd + 64 - L > slack) return 0;
    return L;
}

// A conversation slot: the KV buffer set (parked here while another slot is active; the Runner holds the active set) plus its token list and cursors.
template <class Buf>
struct HySlotT {
    std::vector<Buf> kcache, ivg, pooled;
    std::vector<int32_t> tok;
    int fwd = 0, hw = 0;
    uint64_t last_used = 0;
};

struct HyPickIn {
    const std::vector<int32_t>* tok;
    int fwd, hw;
    uint64_t last_used;
};

// Choose the slot for a request. Returns an index in [0, s.size()] (== s.size(): allocate a new slot) and the resume position in `start`.
//   ring_slack = ring - 512; max_slots = effective slot count; min_reuse = floor below which the request takes a free or LRU slot with start 0, also for the active slot
//   (a prompt shorter than 2*min_reuse uses half its length as the floor).
//   one slot only: the plain LCP resume, no floor. no_reuse (picture request): never resume, prefer a free slot, else a new one, else the LRU.
static inline int hy4_pick_slot(const std::vector<HyPickIn>& s, int active, const std::vector<int32_t>& ids, int T0, int ring_slack, int max_slots, int min_reuse, bool no_reuse, int& start) {
    start = 0;
    const int n = (int) s.size();
    if (max_slots <= 1) {
        if (!no_reuse && n > 0 && s[0].fwd > 0) start = hy4_reuse_start(*s[0].tok, s[0].fwd, ids, T0, ring_slack - std::max(0, s[0].hw - s[0].fwd));
        return 0;
    }
    if (!no_reuse) {
        int best = -1, bs = 0;
        for (int i = 0; i < n; ++i) {
            if (s[i].fwd <= 0) continue;
            int st = hy4_reuse_start(*s[i].tok, s[i].fwd, ids, T0, ring_slack - std::max(0, s[i].hw - s[i].fwd));
            if (st > bs) {
                bs = st;
                best = i;
            }
        }
        if (best >= 0 && bs > 0 && bs >= std::min(min_reuse, T0 / 2)) {
            start = bs;
            return best;
        }
    }
    for (int i = 0; i < n; ++i)
        if (s[i].fwd <= 0) return i;
    if (n < max_slots) return n;
    int lru = 0;
    for (int i = 1; i < n; ++i)
        if (s[i].last_used < s[lru].last_used) lru = i;
    return lru;
}
