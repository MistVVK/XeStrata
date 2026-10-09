// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/pinned.hpp - P2.S1: the host arena the expert weights live in.
//
// The expert weights do not fit in VRAM, so they stay in host memory: the CPU pool computes the missed experts
// from it and the copy engine streams the rest.  That makes this arena the engine's real working set, and it
// must say which backing it got, because the options differ in TLB reach and in copy bandwidth:
//
//   * LARGE PAGES (2 MB) - transparent huge pages via `madvise(MADV_HUGEPAGE)`.  34 GB at 4 KB pages is 8.3 million
//     TLB entries, which does not fit in any TLB.  THP needs no hugetlbfs pool to be reserved by the user.
//   * NORMAL PAGES - correct, and it must be REPORTED rather than silently accepted: an unregistered 4 KB-page
//     source copied at 4.3 GB/s against 14.4 GB/s on the B70's x8 link (bench/results/2026-09-30-b70).
//
// The arena is ONE mapping registered for device copies, not host USM: the Level Zero driver refuses a single USM
// allocation above the device's max_mem_alloc_size (the VRAM size, 32.5 GB on the B70), which every model but the
// Coder exceeds, while registering an ordinary 51 GB mapping works and copies at the same rate as host USM.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

enum class PageBacking { LargePages, NormalPages };

struct PinnedArena {
    void* base = nullptr;
    uint64_t capacity = 0;
    PageBacking backing = PageBacking::NormalPages;
    std::string note;              // why the backing is what it is, for the startup print
    uint64_t locked_bytes = 0;     // explicitly mlocked bytes; registration does not use mlock
    uint64_t registered_bytes = 0; // bytes registered for device copies (the whole arena, or the constructor throws)
    uint64_t slice_bytes = 0;      // 0: one registration covers the arena
    int registered_slices = 0;

    PinnedArena() = default;
    // Throws DeviceError when the mapping or its registration fails.
    explicit PinnedArena(uint64_t bytes, uint64_t slice = 0);
    /// The same, its registrations for device copies ending only at `cuts` (offsets) where it must be split.
    PinnedArena(uint64_t bytes, uint64_t slice, const std::vector<uint64_t>& cuts);
    // The native pack loader's layer boundaries (`bounds`: each layer's start, then the end).  The mapping is
    // registered in pieces that end at `cuts` (the layer starts when empty) when it is larger than the device's
    // largest allocation, or, with `host_usm`, every layer is its own host USM block of its bytes plus `pad`: pinned
    // and readable by GPU kernels at the same address (a registered mapping is for copies only, and one host USM
    // block may not exceed max_mem_alloc_size).  The layers are then not adjacent: address them with `at`.
    PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, uint64_t pad = 0, bool host_usm = false,
                const std::vector<uint64_t>& cuts = {});
    PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, uint64_t max_pinned_bytes,
                const std::string& shared_file, uint64_t pack_hash, const std::vector<uint64_t>& cuts = {});
    /// Holds the shared file's population lock until publish_shared() or destruction. False: already complete.
    bool begin_shared_population();
    void publish_shared();
    std::vector<uint64_t> slice_starts;
    ~PinnedArena();
    PinnedArena(const PinnedArena&) = delete;
    PinnedArena& operator=(const PinnedArena&) = delete;

    /// Registers the arena for device copies on engine device `device` too (a layer split's later GPU): registered
    /// with one GPU's context only, it is pageable memory to the others, and a CUDA GPU copied from it at a fraction
    /// of the rate, blocking the host.  Released by the destructor.  False (and `err`) when the device refuses; the
    /// copies are then correct, only slower.
    bool register_on(int device, std::string& err);

    /// The one registered mapping (a private one with layer bounds) moved into host USM blocks, one per layer of its
    /// bytes plus `pad`, as the `host_usm` constructor makes them: GPU kernels can then read it (a registration for
    /// copies does not give that on every runtime).  A layer at a time, each layer's pages given back once copied, so
    /// the arena is never held twice.  False (and `err`) with the mapping and its registrations as they were.
    bool to_host_usm(uint64_t pad, std::string& err);
    bool valid() const { return base != nullptr; }
    uint8_t* data() const { return (uint8_t*) base; }   // one mapping only: null for host USM blocks
    /// The byte at arena offset `off` (one mapping, or the host USM block of the layer that holds it).
    uint8_t* at(uint64_t off) const;
    std::vector<uint8_t*> layer_base;   // host USM blocks, one per layer (empty: one mapping)

private:
    int shared_fd_ = -1;
    bool shared_locked_ = false;
    std::vector<uint64_t> bounds_;
    void* map_ = nullptr;          // the mapping `base` is aligned inside
    size_t map_bytes_ = 0;
    std::vector<void*> reg_;       // the starts of the mapping's registrations for device copies
    std::vector<std::pair<void*, size_t>> pieces_;   // what is registered (or host USM) on the first device
    std::vector<std::pair<int, std::vector<void*>>> other_reg_;   // register_on: a device, its registrations
};

struct LoadStats {
    double seconds = 0.0;
    // Loader fix: the two halves of `seconds`, both SUMMED OVER THE READER THREADS rather than wall clock, so
    // either can exceed `seconds` when the threads overlap.  They exist to separate the read from the copy and
    // the hash: an 8 MiB chunk that reaches the disk as ~2048 4 KiB reads spends its time in the first one.
    double read_seconds = 0.0;                  // inside the read call only: no seek, no copy, no hash
    double copy_seconds = 0.0;                  // memcpy + FNV-1a
    // A short read is not a slow load, it is a WRONG one: the caller must refuse the pack rather than run
    // on an arena whose tail was never written.  `seconds` alone cannot say that (it is -1.0 on failure, but
    // the caller's own check is on the REQUESTED byte count, which a refused read does not change).
    bool ok = true;                             // false: the load failed, see `error`
    std::string error;                          // why it failed, for the caller's message
    uint64_t bytes = 0;
    uint64_t layers = 0;
    std::vector<uint64_t> layer_checksums;      // one FNV-1a per layer
    double gib_per_second() const { return seconds > 0 ? (double) bytes / (1024.0 * 1024 * 1024) / seconds : 0.0; }
    // Aggregate rate of the readers WHILE THEY WERE INSIDE THE READ CALL: `bytes` over the mean per-thread
    // read time.  `gib_per_second()` above divides the same bytes by the wall clock of the whole loop, which
    // also contains the copy and the hash, so the two differ by how much of the loop was not I/O.
    double read_gib_per_second(int threads) const {
        if (read_seconds <= 0.0 || threads < 1) return 0.0;
        return (double) bytes / (1024.0 * 1024 * 1024) / (read_seconds / (double) threads);
    }
};

// Load `layers` layers of the expert arena into `dst` with `threads` readers, `chunk` bytes at a time.
// Each thread opens its own FILE* and seeks, avoiding a lock around a shared file position.
// Reads use fread() in chunks, then copy and hash each layer.
LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk);
/// Plan v0.3 P6: the same with one byte range per layer (`layer_off[L]`, `layer_bytes[L]` of the file), each
/// written to `layer_dst[L]`.
LoadStats load_experts_ranges(const std::string& path, const std::vector<uint8_t*>& layer_dst,
                              const std::vector<uint64_t>& layer_off, const std::vector<uint64_t>& layer_bytes,
                              int threads, uint64_t chunk);

// FNV-1a 64.  Per layer, so a corrupt or short read names WHICH layer rather than just failing a whole-file
// comparison - the same reason the Phase 1 tools report the first differing element.
uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed = 1469598103934665603ull);

// ---- the expert stream -----------------------------------------------------
//
// THIS IS THE NUMBER THE OBJECTIVE DEPENDS ON.  With one expert per stream, the design's own arithmetic is:
//
//     per token: E = 663.6 MB of expert weights, of which only the MISSES cross the bus
//     miss fraction (1 - h) x 663.6 MB / measured bandwidth = milliseconds per token
//
// So `h` (the VRAM hit rate, ~0.167 by design) is only worth anything if a per-expert transfer can actually
// reach the bus's bandwidth.  A 1,382,400-byte copy is SMALL, and if small copies run at a fraction of large
// ones then the architecture's central constant is wrong and no kernel optimisation can fix it.  That is what
// this measures: the same total bytes at several granularities, so the difference is visible.
struct StreamStats {
    double seconds = 0.0;
    uint64_t bytes = 0;
    uint64_t chunk = 0;
    double gib_per_second() const { return seconds > 0 ? (double) bytes / (1024.0 * 1024 * 1024) / seconds : 0.0; }
};

// Copy `bytes` from pinned host memory at `src` to a device buffer of `chunk` bytes, in `chunk`-sized
// transfers, `iters` times.  The destination is REUSED, so this measures the bus and not allocation.
StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters);

}  // namespace strata::core
