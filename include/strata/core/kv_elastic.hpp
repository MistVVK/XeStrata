// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/core/gpu.hpp"
#include "strata/core/vmm.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {

inline int64_t kv_elastic_target(int64_t cells, int64_t step) {
    return std::max<int64_t>(step, (cells + step - 1) / step * step);
}

inline int64_t kv_elastic_chunks(uint64_t bytes) {
    const uint64_t g = vmm_granularity();
    return (int64_t) ((bytes + g - 1) / g);
}

inline int64_t kv_elastic_need(const VmmRange& range, int64_t lo, int64_t hi) {
    int64_t n = 0;
    for (int64_t c = lo; c < std::min(hi, range.chunks()); ++c) n += !range.mapped(c);
    return n;
}

template<class Take>
bool kv_elastic_map(VmmRange& range, int64_t lo, int64_t hi, const Take& take) {
    const uint64_t g = vmm_granularity();
    hi = std::min(hi, range.chunks());
    std::vector<int64_t> fresh;
    for (int64_t c = lo; c < hi; ++c) if (!range.mapped(c)) fresh.push_back(c);
    if (fresh.empty()) return true;
    if (!range.map_range(lo, hi, take)) return false;
    for (int64_t c : fresh)
        if (!strata::gpu::memset(range.base() + (uint64_t) c * g, 0, (size_t) g)) return false;
    return true;
}

template<class Give>
int64_t kv_elastic_unmap(VmmRange& range, int64_t lo, int64_t hi, const Give& give) {
    int64_t n = 0;
    for (int64_t c = lo; c < std::min(hi, range.chunks()); ++c)
        if (const VmmChunk h = range.unmap(c)) { give(h); ++n; }
    return n;
}

}  // namespace strata::core
