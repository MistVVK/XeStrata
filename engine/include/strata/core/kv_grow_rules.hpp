// include/strata/core/kv_grow_rules.hpp - the elastic K/V (--kv-grow) once the expert cache is known.
//
// When --kv-grow is accepted at session init, every QSA pool lives in its own VMM range with physical memory for its
// first cells only (16,384 by default; layer.cpp). From then on a request may only run once its cells are mapped, so the
// K/V must be able to grow for the whole run. It grows in two ways: the expert cache gives it the chunks of slots just
// below the prompt path's loan (the experts there are given up), and when the cache has nothing to give, new VRAM.
// The cache can give slots only with a VMM range of its own, a residency table, an expert source to compute the given-up
// experts from, and a top above its floor; without one of those the K/V still grows, from new VRAM alone. (Before, the
// whole elastic K/V was switched off in that case while its pools stayed at their first cells, and a longer prompt wrote
// past them: an illegal memory access.)
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::core {

struct KvGrowStart {
    bool on = false;          ///< the K/V grows before a request that needs more cells (the pools are elastic)
    bool from_cache = false;  ///< ... and the expert cache gives it slots; false: new VRAM only, the cache stays whole
};

/// `pools_elastic`: the session's pools were made elastic at init (qsa_kv_elastic()). `cache_vmm`: the expert cache
/// is one VMM range; `residency`: a residency table on the device and the host; `source`: an expert source; `top`: the
/// slot the K/V may take room below (the prompt path's loan starts there); `floor`: the slots the cache always keeps.
inline KvGrowStart kv_grow_start(bool pools_elastic, bool cache_vmm, bool residency, bool source, int64_t top,
                                 int64_t floor) {
    KvGrowStart s;
    s.on = pools_elastic;
    s.from_cache = pools_elastic && cache_vmm && residency && source && top > floor;
    return s;
}

/// The cells one grow maps for a request that needs `cells`: whole steps of `step` cells, at most the context.
inline int64_t kv_grow_target(int64_t cells, int64_t step, int64_t max_context) {
    return std::min<int64_t>(max_context, (cells + step - 1) / step * step);
}

}  // namespace strata::core
