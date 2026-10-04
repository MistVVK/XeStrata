// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/device.hpp - P2.S1: the device arena and the runtime's device facts.
//
// `DeviceArena` is ONE device allocation per planner region with bump sub-allocation below it and no frees.  That
// is not a simplification for the first version: the memory plan is fixed at startup, so the set of regions and
// their sizes is known before anything is allocated, and an allocator that can free would be solving a problem
// the engine does not have while adding fragmentation and failure modes it does.
//
// VRAM is the binding constraint of the whole design (the expert cache takes what the fixed regions leave), so a
// runtime that discovers at token 4000 that it has overcommitted has already lost.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

struct DeviceInfo {
    int ordinal = -1;
    std::string name, driver_version, platform_version;
    std::string backend;            ///< "Level Zero", "CUDA" or "HIP"
    uint32_t vendor_id = 0, device_id = 0;   ///< device_id: Intel GPUs only (0 elsewhere)
    uint64_t total_bytes = 0, free_bytes = 0;
    bool free_bytes_known = false;
    uint32_t compute_units = 0;
    bool integrated = false;        ///< the processor's own graphics: its memory is the system RAM (Level Zero)
    std::vector<size_t> subgroup_sizes;
    bool fp64 = false, host_usm = false, device_usm = false;
};

// Ordinal 0 is the GPU the runtime selected (src/core/device.cpp, select_device).
DeviceInfo device_info(int ordinal = 0);

/// A smaller device, for checking the engine's choices on the B70 (AGENTS.md, the first rule):
/// STRATA_VRAM_LIMIT_MIB caps the memory the engine sees (its total, and its free part as total less what is used),
/// STRATA_MAX_ALLOC_MIB the largest allocation it assumes.  0 when unset.
uint64_t vram_limit_bytes();
uint64_t max_alloc_limit_bytes();
/// `free` and `total` as the engine sees them under STRATA_VRAM_LIMIT_MIB.
void apply_vram_limit(uint64_t& free, uint64_t& total);
/// The device's largest allocation, under STRATA_MAX_ALLOC_MIB.
uint64_t max_alloc_bytes();

class DeviceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// One device USM allocation per planner region. Destruction drains the runtime
// before freeing it; callers may reuse a suballocation only after its last reader.
class DeviceArena {
public:
    explicit DeviceArena(uint64_t bytes, int ordinal = 0, bool poison = false);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // Alignment applies to the absolute address, including alignments above 4096.
    void* alloc(uint64_t bytes, uint64_t align = 256);
    uint64_t capacity() const { return capacity_; }
    uint64_t used() const { return used_; }
    uint64_t peak() const { return used_; }
    int ordinal() const { return ordinal_; }
    void* base() const { return base_; }

private:
    void* base_ = nullptr;
    uint64_t capacity_ = 0, used_ = 0;
    int ordinal_ = 0;
};

}  // namespace strata::core
