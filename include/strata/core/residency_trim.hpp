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

/// Every entry of `res` (residency index -> slot) whose slot is at or past `live` (the slots still mapped) becomes
/// kNotResident, and (index, slot) is appended to `evicted` in table order (a later grow puts those experts back).
/// Returns how many entries it took.
inline int64_t evict_slots_past(std::vector<int32_t>& res, int64_t live, std::vector<std::pair<int32_t, int32_t>>& evicted) {
    int64_t n = 0;
    for (size_t i = 0; i < res.size(); ++i)
        if (res[i] >= live) {
            evicted.emplace_back((int32_t) i, res[i]);
            res[i] = kNotResident;
            ++n;
        }
    return n;
}

}  // namespace strata::core
