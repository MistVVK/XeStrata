// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
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

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

struct DeviceInfo {
    int ordinal = -1;
    std::string name, driver_version, platform_version;
    std::string backend;            ///< "Level Zero", "CUDA" or "HIP"
    std::string pci;                ///< its PCI address (domain:bus:device.function, lower case), "?" if not reported
    std::string arch;               ///< an AMD GPU's ISA as its HIP runtime names it (gfx1200), empty elsewhere
    uint32_t vendor_id = 0, device_id = 0;   ///< device_id: Intel GPUs only (0 elsewhere)
    uint64_t total_bytes = 0, free_bytes = 0;
    bool free_bytes_known = false;
    uint32_t compute_units = 0;
    bool integrated = false;        ///< the processor's own graphics: its memory is the system RAM (Level Zero)
    std::vector<size_t> subgroup_sizes;
    bool fp64 = false, host_usm = false, device_usm = false;
};

// Ordinal 0 is the GPU the runtime selected, 1.. the other usable GPUs (src/core/device.cpp, find_devices).
DeviceInfo device_info(int ordinal = 0);

/// A smaller device, for checking the engine's choices on the B70 (AGENTS.md, the first rule):
/// STRATA_VRAM_LIMIT_MIB caps the memory the engine sees (its total, and its free part as total less what is used),
/// STRATA_MAX_ALLOC_MIB the largest allocation it assumes.  0 when unset.
uint64_t vram_limit_bytes();
uint64_t max_alloc_limit_bytes();
/// `free` and `total` as the engine sees them under STRATA_VRAM_LIMIT_MIB.  `held`: the bytes this process holds on the
/// device, when the device's report may not show them yet.  `short_b` (when given): how far what is used is past the
/// limit, which `free` (0 then) cannot say.
void apply_vram_limit(uint64_t& free, uint64_t& total, uint64_t held = 0, uint64_t* short_b = nullptr);
/// The free VRAM a cache is sized from: `free` now, less what was free at the start (`free_at_start`) beyond
/// `vram_free_mib` (setup's --vram-free-mib: a start with a few MiB more free gets the same cache; -1: all of it).
inline uint64_t counted_free(uint64_t free, uint64_t free_at_start, int64_t vram_free_mib) {
    if (vram_free_mib < 0) return free;
    const uint64_t vf = (uint64_t) vram_free_mib << 20;
    return free_at_start > vf ? free - std::min(free, free_at_start - vf) : free;
}
/// An integrated GPU's free memory: at most what the OS can give back, less STRATA_UMA_HEADROOM_GIB (default 6).
/// Not where the device's total is larger than the OS's own RAM (`total`): that is a BIOS carve-out, a pool of its
/// own, and the system's RAM is not what it hands out (a Strix Halo: 64 GiB of carve-out, 62 GiB seen by the OS).
void apply_uma_limit(uint64_t& free, uint64_t total);
/// A discrete GPU's free memory on Windows: at most what the OS's budget for this process leaves it
/// (platform::gpu_local_memory_budget), for the GPU at the PCI address `pci`.  Elsewhere `free` stays as it is.
void apply_os_budget(const std::string& pci, uint64_t& free);
/// The room on a device no plan counts: what the driver and the runtime keep for themselves, the copies' staging
/// and the buffers allocated after the plan is made.  One rule for every engine and every stage (AGENTS.md, the
/// first rule: from the device's report, never its name): a 32nd of the memory the engine sees for the device
/// (`total_bytes`: gpu::mem_info's, STRATA_VRAM_LIMIT_MIB applied), kept between `least_b` and `most_b`, of which
/// the caller keeps one `parts`th.
///
/// The bounds are each caller's own measure.  The defaults, 512 MiB up to 768, are the room the image pipelines
/// and SeedVR2 keep for the driver's own and the copies' room on every card they were run on; the share reaches
/// that on a 16 GiB card (16304 / 32 = 509 MiB) and passes 512 MiB past 16 GiB of memory, up to 768.  The plain
/// engine passes its measured 192 MiB (src/core/plain_model.cpp, 2026-10-12: under STRATA_VRAM_LIMIT_MIB=4096 the
/// peak stayed below the card with 128 MiB, where 768 MiB of before left 700 MiB of it unused) and counts the
/// libraries and graphs it allocates after the tier beside it; Qwen3.8 passes 700 MiB, the room its graphs, hit
/// scratch and head measured.  A stage that runs beside a plan which already keeps the whole room asks for a part
/// of it: `parts` 2, 4 and 8 are SeedVR2's VAE, MOSS-Image's decode and its encoders, whose 256, 128 and 64 MiB
/// on the 16 GiB card are a half, a quarter and an eighth of that room.
uint64_t vram_room(uint64_t total_bytes, uint64_t parts = 1, uint64_t least_b = 512ull << 20,
                   uint64_t most_b = 768ull << 20);
/// The VRAM to leave unused for the runtime's and the driver's own allocations, where the driver's free memory shows
/// no use at all: i915 gives a process without CAP_PERFMON the whole VRAM as free (an A380's 5783 MiB however much is
/// allocated), so the engine counts what it holds itself, and this is what that count lacks.  0 for a GPU whose report
/// shows the use, and on Windows.  For the GPU at the PCI address `pci`, of `total` bytes.
uint64_t unreported_vram_margin(const std::string& pci, uint64_t total);
/// The device's largest allocation, under STRATA_MAX_ALLOC_MIB.
uint64_t max_alloc_bytes();

class Runtime;

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
    void* raw_ = nullptr;    // the allocation; base_ is it rounded up to 4 KB
    void* base_ = nullptr;
    uint64_t capacity_ = 0, used_ = 0;
    int ordinal_ = 0;
    Runtime* runtime_ = nullptr;   // the runtime of `ordinal_`, which frees `raw_`
};

}  // namespace strata::core
