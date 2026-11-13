// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/kv_grow_hold.hpp - how long the elastic K/V keeps a parked conversation's cells
#pragma once
// The elastic K/V (--kv-grow, vmm.hpp) holds physical memory only for the cells the requests reach; a request that
// needs fewer cells gives the rest back to the expert cache and the next long one grows again.  Upstream 56ae98dd
// (#1128) added the choice of keeping the longest parked conversation's cells while it waits, so a conversation that
// comes back does not grow (and shuffle the cache's experts) after every short request in between.  Holding keeps those
// cells out of the expert cache, and what that costs depends on the device: where its own pages are a chunk already
// (2 MiB or more, the same report --kv-grow's default reads) the mapped cache runs as ordinary allocations and the grow
// the hold saves is the larger wait (the RTX 4070), while where the pages are smaller (the B70's 64 KiB) the cache in
// mapped memory is itself slower and holding measured slower (2026-10-09, upstream 0.1.40).

#include "strata/core/vmm.hpp"
#include "strata/platform/env.hpp"

#include <cstdlib>

namespace strata::core {

/// XESTRATA_KV_GROW_HOLD: 0 = never keep the cells, 1 = always, unset / "auto" (the default) = where the device maps
/// pages a chunk large (vmm_large_pages).  Read per request: no file, and a device that reports otherwise the next time
/// is followed.
inline bool kv_grow_hold() {
    const char* v = strata::env_get("KV_GROW_HOLD");
    if (v != nullptr && v[0] != '\0' && v[0] != 'a' && v[0] != 'A') return std::strtol(v, nullptr, 10) != 0;
    return vmm_large_pages();
}

}  // namespace strata::core
