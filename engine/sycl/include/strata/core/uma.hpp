// sycl/include/strata/core/uma.hpp - an integrated Intel GPU's memory (a GPU built into the processor, such as Panther
// Lake's Arc B390): one pool with the system's RAM, no card memory of its own (#6448).  The definitions are in
// sycl/src/core/expert_cache.cpp, beside device_free_bytes(), which answers from here for such a GPU.  The inline
// functions at the end are their arithmetic, without a device (uma_test.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace strata::core {

/// Is the current device an integrated GPU (dpct's integrated flag, Level Zero's ZE_DEVICE_PROPERTY_FLAG_INTEGRATED)?
/// STRATA_UMA=1 / =0 overrides the answer for every device.  Linux only: false elsewhere.
bool device_unified_memory();

/// The headroom kept free for the rest of the system on such a GPU: STRATA_UMA_HEADROOM_GIB, default 6 GiB.
uint64_t uma_headroom_bytes();

/// MemAvailable, and the freed pages the GPU driver keeps pooled for its next allocation (GPUReclaim in /proc/meminfo;
/// 0 when the kernel has no such line or reports 0 there, as kernel 7.0 did in #2412).  False when /proc/meminfo cannot
/// be read.
bool uma_memory(uint64_t& available, uint64_t& pooled);

/// What the model may still take: available + pooled - headroom, at most `total_b` (0 = no cap).
size_t uma_free_bytes(size_t total_b);

/// STRATA_UMA's value: 1 plans the GPU as an integrated one, 0 as a card, -1 (unset or empty) leaves it to the device.
inline int uma_override(const char* v) { return v != nullptr && v[0] != '\0' ? (v[0] != '0' ? 1 : 0) : -1; }

/// One line of /proc/meminfo: MemAvailable goes to `available`, GPUReclaim to `pooled` (bytes); any other line changes
/// neither.  True when the line was MemAvailable.
inline bool uma_meminfo_line(const char* line, uint64_t& available, uint64_t& pooled) {
    unsigned long long kb = 0;
    if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { available = (uint64_t) kb << 10; return true; }
    if (std::sscanf(line, "GPUReclaim: %llu kB", &kb) == 1) pooled = (uint64_t) kb << 10;
    return false;
}

/// available + pooled less the headroom (0 when the headroom is larger), at most `total_b` (0 = no cap).
inline uint64_t uma_room(uint64_t available, uint64_t pooled, uint64_t headroom, uint64_t total_b) {
    const uint64_t room = available + pooled > headroom ? available + pooled - headroom : 0;
    return total_b > 0 && room > total_b ? total_b : room;
}

}  // namespace strata::core
