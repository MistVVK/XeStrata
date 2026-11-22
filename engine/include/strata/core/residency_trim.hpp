#pragma once
// After the segmented expert cache gave its tail back (ExpertCache::shrink, --vram-elastic), the residency table may still
// point experts at slots that are no longer mapped.  That holds after a full shrink and also after one the driver refused
// part way: the segments released before the refusal are gone either way.  This brings the table in line with what is
// still mapped.  A pure function, no GPU: the engine's VRAM command calls it after every shrink, and tests call it directly.

#include "strata/core/expert_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::core {

/// The same over a layer range: only entries whose expert sits on a layer in `[lo_layer, hi_layer)` (a card's own layers
/// under a layer split) are checked against `live` (that cache's slots still mapped).  Slot numbers are per-card-local and
/// overlap across cards, so with a split each cache must trim only its own experts; with one GPU the range covers every
/// layer.  Returns how many entries it took, appended to `evicted` in table order (a later grow puts those back).
inline int64_t evict_slots_past(std::vector<int32_t>& res, int64_t live, int64_t lo_layer, int64_t hi_layer,
                                int64_t n_expert, std::vector<std::pair<int32_t, int32_t>>& evicted) {
    const size_t begin = (size_t) (lo_layer * n_expert);
    const int64_t total = (int64_t) res.size();   // one sign for the comparisons below: a residency table never nears INT64_MAX.
    const size_t cap = (size_t) ((hi_layer == INT64_MAX || hi_layer > total) ? total : hi_layer * n_expert);
    int64_t n = 0;
    for (size_t i = begin; i < res.size() && i < cap; ++i)
        if (res[i] >= live) {
            evicted.emplace_back((int32_t) i, res[i]);
            res[i] = kNotResident;
            ++n;
        }
    return n;
}

/// Every entry of `res` (residency index -> slot) whose slot is at or past `live` (the slots still mapped) becomes
/// kNotResident, and (index, slot) is appended to `evicted` in table order (a later grow puts those experts back).
/// Returns how many entries it took.  One GPU: the whole table belongs to that one cache.
inline int64_t evict_slots_past(std::vector<int32_t>& res, int64_t live, std::vector<std::pair<int32_t, int32_t>>& evicted) {
    return evict_slots_past(res, live, 0, INT64_MAX, 1, evicted);
}

}  // namespace strata::core
