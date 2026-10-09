// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/pinned.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/device.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

#include <sys/mman.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>

#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)

namespace strata::core {

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    for (uint64_t i = 0; i < n; ++i) { seed ^= p[i]; seed *= 1099511628211ull; }
    return seed;
}

namespace {
constexpr uint64_t kHugePage = 2ull << 20;
}

// The mapping registered for device copies in pieces no larger than the device's largest allocation: past it the
// UHD 770 (4 GiB) registered without an error and copied wrong bytes, the B70 (31 GB) did not.  `cuts` are the
// offsets a piece may end at (a layer's start, so no expert crosses two registrations), or empty: every `limit`.
namespace {
void register_pieces(uint8_t* base, uint64_t bytes, const std::vector<uint64_t>& cuts, std::vector<void*>& reg,
                     std::vector<std::pair<void*, size_t>>& pieces) {
    const uint64_t limit = max_alloc_bytes();
    // pieces end at the last cut that keeps them within the limit (or at the limit when no cut does)
    std::vector<uint64_t> ends, c;
    for (uint64_t x : cuts)
        if (x > 0 && x < bytes) c.push_back(x);
    c.push_back(bytes);
    for (uint64_t start = 0; start < bytes;) {
        size_t i = 0;
        while (i < c.size() && c[i] <= start) ++i;
        uint64_t prev = start;
        while (i < c.size() && c[i] - start <= limit) prev = c[i++];
        const uint64_t end = prev > start ? prev : std::min(bytes, start + limit);
        ends.push_back(end);
        start = end;
    }
    uint64_t start = 0;
    for (uint64_t end : ends) {
        sycl::ext::oneapi::experimental::prepare_for_device_copy(base + start, (size_t) (end - start),
                                                                 Runtime::get().context());
        reg.push_back(base + start);
        pieces.emplace_back(base + start, (size_t) (end - start));
        start = end;
    }
}

// Upstream #771: with transparent huge pages "always" the kernel backs a mapping with huge pages where it can; an
// extra MADV_HUGEPAGE only adds direct reclaim and compaction on every fault (defrag=madvise), which on a fragmented
// machine stretched a 25 s start to 432 s.  So it is not asked for there; STRATA_NO_ARENA_THP=1 skips it anywhere.
// Returns why it was skipped, or an empty string when it should be asked for.
std::string thp_request_skipped() {
    if (std::getenv("STRATA_NO_ARENA_THP") != nullptr) return "transparent huge pages skipped (STRATA_NO_ARENA_THP)";
    std::string mode;
    if (std::FILE* f = std::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
        char line[128] = {0};
        if (std::fgets(line, sizeof line, f) != nullptr) {
            const char* open = std::strchr(line, '[');
            const char* close = open ? std::strchr(open, ']') : nullptr;
            if (open && close) mode.assign(open + 1, close);
        }
        std::fclose(f);
    }
    return mode == "always" ? "transparent huge pages are on for every mapping (THP always): MADV_HUGEPAGE not requested"
                            : std::string();
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, slice, std::vector<uint64_t>{}) {}

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice, const std::vector<uint64_t>& cuts) : capacity(bytes) {
    (void) slice;   // the registrations cover any slice geometry
    if (!bytes || bytes > std::numeric_limits<size_t>::max() - kHugePage)
        throw DeviceError("the expert arena requires a nonzero size");
    // Over-map by one huge page so the arena starts on a 2 MB boundary and THP can back it from the first byte.
    map_bytes_ = (size_t) bytes + kHugePage;
    map_ = mmap(nullptr, map_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        throw DeviceError("mmap of the expert arena failed for " + std::to_string(bytes) + " bytes");
    }
    base = (void*) (((uintptr_t) map_ + kHugePage - 1) & ~(uintptr_t) (kHugePage - 1));
    const std::string thp_skip = thp_request_skipped();
    const bool huge = thp_skip.empty() && madvise(base, (size_t) bytes, MADV_HUGEPAGE) == 0;
    // Registered before the loader touches a page: the B70 run showed later writes reach the device intact.
    try {
        register_pieces((uint8_t*) base, bytes, cuts, reg_, pieces_);
    } catch (const std::exception& e) {
        for (void* r : reg_) sycl::ext::oneapi::experimental::release_from_device_copy(r, Runtime::get().context());
        reg_.clear();
        pieces_.clear();
        munmap(map_, map_bytes_);
        map_ = base = nullptr;
        throw DeviceError(std::string("registering the expert arena for device copies failed: ") + e.what());
    }
    backing = huge ? PageBacking::LargePages : PageBacking::NormalPages;
    registered_bytes = bytes;
    registered_slices = 1;
    slice_starts = {0};
    note = std::string("one mapping registered for device copies") +
           (reg_.size() > 1 ? " in " + std::to_string(reg_.size()) + " pieces (the GPU's largest allocation)" : "") +
           "; " +
           (!thp_skip.empty() ? thp_skip
            : huge             ? "transparent huge pages requested (madvise)"
                               : "madvise(MADV_HUGEPAGE) refused: 4 KB pages") +
           "; no mlock";
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, uint64_t pad, bool host_usm,
                         const std::vector<uint64_t>& cuts)
    : capacity(bytes), bounds_(bounds) {
    if (!host_usm) {
        // one mapping, registered in pieces that end at layer starts: the other constructor's work, then keep the
        // bounds for `at`
        PinnedArena one(bytes, 0, cuts.empty() ? std::vector<uint64_t>(bounds.begin() + 1, bounds.end()) : cuts);
        std::swap(base, one.base); std::swap(map_, one.map_); std::swap(map_bytes_, one.map_bytes_);
        std::swap(reg_, one.reg_);
        std::swap(pieces_, one.pieces_);
        backing = one.backing; note = one.note; registered_bytes = one.registered_bytes;
        registered_slices = one.registered_slices; slice_starts = one.slice_starts;
        return;
    }
    if (bounds.size() < 2 || bounds.back() > bytes) throw DeviceError("the expert arena's layer bounds are invalid");
    auto& runtime = Runtime::get();
    for (size_t l = 0; l + 1 < bounds.size(); ++l) {
        const uint64_t n = bounds[l + 1] - bounds[l] + pad;
        void* p = sycl::malloc_host((size_t) n, runtime.context());
        if (!p) {
            for (uint8_t* q : layer_base) runtime.free(q);
            layer_base.clear();
            throw DeviceError("host USM for the expert arena refused at layer " + std::to_string(l) + " (" +
                              std::to_string(n) + " B)");
        }
        layer_base.push_back((uint8_t*) p);
        pieces_.emplace_back(p, (size_t) n);
        slice_starts.push_back(bounds[l]);
    }
    base = layer_base[0];
    backing = PageBacking::NormalPages;   // the driver's pages; THP cannot be requested for them
    registered_bytes = bytes;
    slice_bytes = 1;                      // nonzero: one device alias per layer
    registered_slices = (int) layer_base.size();
    note = "host USM, one block per layer (" + std::to_string(layer_base.size()) +
           "): pinned and readable by GPU kernels";
}

namespace {
constexpr uint64_t kArenaHeaderBytes = 4096;
struct SharedArenaHeader {
    char magic[16] = "STRATA-ARENA-V1";
    uint32_t version = 1, header_bytes = kArenaHeaderBytes;
    uint64_t arena_bytes = 0, pack_hash = 0;
    uint64_t reserved[4] = {};   // reserved[0]: durable population completion
};
void arena_lock(int fd) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno != EWOULDBLOCK && errno != EINTR) throw DeviceError("shared arena: file lock failed");
        if (std::chrono::steady_clock::now() >= end) throw DeviceError("shared arena: population lock timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, uint64_t max_pinned_bytes,
                         const std::string& file, uint64_t pack_hash, const std::vector<uint64_t>& cuts)
    : capacity(bytes), bounds_(bounds) {
    (void) max_pinned_bytes;   // Xe registers every slice, bounded by the device's largest allocation.
    if (!bytes || bytes > (uint64_t) std::numeric_limits<off_t>::max() - kArenaHeaderBytes - kHugePage)
        throw DeviceError("shared arena: invalid size");
    shared_fd_ = open(file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (shared_fd_ < 0) throw DeviceError("shared arena: cannot open backing file");
    try {
        arena_lock(shared_fd_);
        shared_locked_ = true;
        struct stat st{};
        if (fstat(shared_fd_, &st) != 0) throw DeviceError("shared arena: cannot size backing file");
        SharedArenaHeader hdr;
        if (st.st_size == 0) {
            hdr.arena_bytes = bytes; hdr.pack_hash = pack_hash;
            if (ftruncate(shared_fd_, (off_t) (bytes + kArenaHeaderBytes)) != 0 ||
                pwrite(shared_fd_, &hdr, sizeof hdr, 0) != (ssize_t) sizeof hdr || fdatasync(shared_fd_) != 0)
                throw DeviceError("shared arena: cannot initialize header");
        } else {
            if ((uint64_t) st.st_size != bytes + kArenaHeaderBytes) {
                note = "shared arena: file size differs from expected arena size";
            } else if (pread(shared_fd_, &hdr, sizeof hdr, 0) != (ssize_t) sizeof hdr ||
                       std::memcmp(hdr.magic, "STRATA-ARENA-V1", 16) != 0 || hdr.version != 1 ||
                       hdr.header_bytes != kArenaHeaderBytes || hdr.arena_bytes != bytes) {
                note = "shared arena: incompatible header";
            } else if (hdr.pack_hash != pack_hash) {
                note = "shared arena: pack hash mismatch";
            }
            if (!note.empty()) {
                flock(shared_fd_, LOCK_UN); shared_locked_ = false;
                ::close(shared_fd_); shared_fd_ = -1;
                return;
            }
        }
        // Reserve the address range first: file offset 4096 can still start at a 2 MiB-aligned virtual address.
        map_bytes_ = (size_t) bytes + kHugePage;
        map_ = mmap(nullptr, map_bytes_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (map_ == MAP_FAILED) { map_ = nullptr; throw DeviceError("shared arena: address reservation failed"); }
        base = (void*) (((uintptr_t) map_ + kHugePage - 1) & ~(uintptr_t) (kHugePage - 1));
        if (mmap(base, (size_t) bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, shared_fd_,
                 (off_t) kArenaHeaderBytes) == MAP_FAILED) {
            base = nullptr; throw DeviceError("shared arena: mapping failed");
        }
        const bool huge = thp_request_skipped().empty() && madvise(base, (size_t) bytes, MADV_HUGEPAGE) == 0;
        register_pieces((uint8_t*) base, bytes, cuts.empty() ? bounds : cuts, reg_, pieces_);
        registered_bytes = bytes; registered_slices = 1; slice_starts = {0};
        backing = huge ? PageBacking::LargePages : PageBacking::NormalPages;
        note = "shared file mapping registered in device-sized slices; no mlock";
        flock(shared_fd_, LOCK_UN); shared_locked_ = false;
    } catch (...) {
        for (void* r : reg_) sycl::ext::oneapi::experimental::release_from_device_copy(r, Runtime::get().context());
        reg_.clear();
        if (map_) munmap(map_, map_bytes_);
        map_ = base = nullptr;
        ::close(shared_fd_); shared_fd_ = -1;
        throw;
    }
}

bool PinnedArena::begin_shared_population() {
    if (shared_fd_ < 0) return true;
    arena_lock(shared_fd_); shared_locked_ = true;
    SharedArenaHeader hdr;
    if (pread(shared_fd_, &hdr, sizeof hdr, 0) != (ssize_t) sizeof hdr)
        throw DeviceError("shared arena: cannot read population state");
    if (hdr.reserved[0] == 1) {
        flock(shared_fd_, LOCK_UN); shared_locked_ = false;
        return false;
    }
    return true;
}

void PinnedArena::publish_shared() {
    if (shared_fd_ < 0) return;
    if (!shared_locked_) throw DeviceError("shared arena: population lock is not held");
    // The complete body must reach storage before another process can observe its completion record.
    if (msync(base, (size_t) capacity, MS_SYNC) != 0 || fdatasync(shared_fd_) != 0)
        throw DeviceError("shared arena: cannot persist populated weights");
    const uint64_t complete = 1;
    if (pwrite(shared_fd_, &complete, sizeof complete, offsetof(SharedArenaHeader, reserved)) !=
            (ssize_t) sizeof complete || fdatasync(shared_fd_) != 0)
        throw DeviceError("shared arena: cannot publish completed weights");
    flock(shared_fd_, LOCK_UN); shared_locked_ = false;
}

uint8_t* PinnedArena::at(uint64_t off) const {
    if (layer_base.empty()) return (uint8_t*) base + off;
    size_t l = (size_t) (std::upper_bound(bounds_.begin(), bounds_.end() - 1, off) - bounds_.begin());
    l = l == 0 ? 0 : l - 1;
    return layer_base[l] + (off - bounds_[l]);
}

bool PinnedArena::register_on(int device, std::string& err) {
    for (const auto& o : other_reg_)
        if (o.first == device) return true;
    std::vector<void*> regs;
    try {
        const sycl::context ctx = Runtime::at(device).context();
        for (const auto& p : pieces_) {
            sycl::ext::oneapi::experimental::prepare_for_device_copy(p.first, p.second, ctx);
            regs.push_back(p.first);
        }
    } catch (const std::exception& e) {
        for (void* r : regs)
            sycl::ext::oneapi::experimental::release_from_device_copy(r, Runtime::at(device).context());
        err = std::string("registering the expert arena on GPU ") + std::to_string(device) + ": " + e.what();
        return false;
    }
    other_reg_.emplace_back(device, std::move(regs));
    return true;
}

PinnedArena::~PinnedArena() {
    if (shared_fd_ >= 0) ::close(shared_fd_);
    for (auto& o : other_reg_) {
        try {
            auto& r = Runtime::at(o.first);
            r.finish();   // no copy may still read the mapping
            for (void* p : o.second) sycl::ext::oneapi::experimental::release_from_device_copy(p, r.context());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Xe runtime: releasing the expert arena on GPU %d failed: %s; terminating\n", o.first,
                         e.what());
            std::fflush(stderr);
            std::_Exit(1);
        }
    }
    if (!layer_base.empty()) {
        for (uint8_t* p : layer_base) Runtime::get().free(p);
        return;
    }
    if (!base) return;
    auto& runtime = Runtime::get();
    try {
        runtime.finish();   // no copy may still read the mapping
        for (void* r : reg_) sycl::ext::oneapi::experimental::release_from_device_copy(r, runtime.context());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Xe runtime: releasing the expert arena failed: %s; terminating\n", e.what());
        std::fflush(stderr);
        std::_Exit(1);
    }
    munmap(map_, map_bytes_);
}

LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk) {
    std::vector<uint64_t> off((size_t) layers), n((size_t) layers, blobs_per_layer * blob_bytes);
    std::vector<uint8_t*> to((size_t) layers);
    for (uint64_t L = 0; L < layers; ++L) {
        off[(size_t) L] = L * blobs_per_layer * blob_bytes;
        to[(size_t) L] = dst == nullptr ? nullptr : dst + off[(size_t) L];
    }
    return load_experts_ranges(path, to, off, n, threads, chunk);
}

LoadStats load_experts_ranges(const std::string& path, const std::vector<uint8_t*>& layer_dst,
                              const std::vector<uint64_t>& layer_off, const std::vector<uint64_t>& layer_bytes,
                              int threads, uint64_t chunk) {
    LoadStats st;
    if (!chunk || layer_off.size() != layer_bytes.size() || layer_dst.size() != layer_off.size() ||
        std::find(layer_dst.begin(), layer_dst.end(), nullptr) != layer_dst.end()) {
        st.ok = false;
        st.error = "invalid destination, chunk size, or layer ranges";
        st.seconds = -1.0;
        return st;
    }
    const uint64_t layers = (uint64_t) layer_off.size();
    st.layers = layers;
    st.bytes = 0;
    for (uint64_t b : layer_bytes) st.bytes += b;
    if (threads < 1) threads = 1;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> layer_hash((size_t) layers, 1469598103934665603ull);
    std::atomic<uint64_t> next_layer{0};
    std::atomic<uint64_t> read_ns_sum{0};   // summed over the threads: see LoadStats::read_seconds
    std::atomic<uint64_t> copy_ns_sum{0};
    std::mutex err_mu;
    std::string err;

    auto worker = [&]() {
        std::vector<uint8_t> buf((size_t) chunk);
        // One handle per thread, seeked once per layer: a shared handle would need a lock around the seek and
        // would serialise the very thing the threads are here to parallelise.  `fread` on a `FILE*` rather than
        // `std::ifstream`: fread uses the requested chunk size and keeps the OS cache available.
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::lock_guard<std::mutex> g(err_mu);
            err = "cannot open " + path;
            return;
        }
        // A `FILE*` has no destructor that closes it, and this function has early returns below (open, seek and
        // short-read failures), so the guard is what keeps the closing correct on every path.
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        uint64_t read_ns = 0, copy_ns = 0;
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);
            if (L >= layers) break;
            const uint64_t off = layer_off[(size_t) L];
            uint64_t remaining = layer_bytes[(size_t) L];
            uint64_t pos = 0;
            uint64_t h = 1469598103934665603ull;
            // 64-bit seek: the pack is 42.9 GB, so the 32-bit `fseek` would wrap past 4 GiB
            if (STRATA_FSEEK64(f, off) != 0) {
                std::lock_guard<std::mutex> g(err_mu);
                err = "seek to " + std::to_string(off) + " B failed in layer " + std::to_string(L);
                return;
            }
            while (remaining > 0) {
                const uint64_t n = remaining < chunk ? remaining : chunk;
                const auto t_read = std::chrono::steady_clock::now();
                const size_t got = std::fread(buf.data(), 1, (size_t) n, f);
                read_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_read).count();
                // A short read is EOF or an I/O error, never a silent zero fill: say WHERE and HOW SHORT, and
                // keep going no further - the caller turns this into a refused load, not a wrong answer.
                if (got != (size_t) n) {
                    std::lock_guard<std::mutex> g(err_mu);
                    err = "short read in layer " + std::to_string(L) + ": got " + std::to_string(got) + " of "
                          + std::to_string(n) + " B at offset " + std::to_string(off + pos)
                          + (std::ferror(f) != 0 ? " (ferror set)" : "");
                    return;
                }
                const auto t_copy = std::chrono::steady_clock::now();
                std::memcpy(layer_dst[(size_t) L] + pos, buf.data(), (size_t) n);
                h = fnv1a64(buf.data(), n, h);
                copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_copy).count();
                pos += n;
                remaining -= n;
            }
            layer_hash[(size_t) L] = h;
        }
        read_ns_sum.fetch_add(read_ns);
        copy_ns_sum.fetch_add(copy_ns);
    };

    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();

    if (!err.empty()) {
        std::fprintf(stderr, "load_experts: %s\n", err.c_str());
        st.seconds = -1.0;
        st.ok = false;
        st.error = err;
        return st;
    }
    st.read_seconds = (double) read_ns_sum.load() / 1e9;
    st.copy_seconds = (double) copy_ns_sum.load() / 1e9;
    st.layer_checksums = std::move(layer_hash);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    if (!src || !bytes || !chunk || iters <= 0 ||
        bytes > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(iters))
        throw DeviceError("invalid stream bandwidth arguments");
    StreamStats st;
    st.bytes = bytes * static_cast<uint64_t>(iters);
    st.chunk = chunk;
    auto& runtime = Runtime::get();
    std::unique_ptr<DeviceArena> held;
    try {
        held = std::make_unique<DeviceArena>(std::min(bytes, chunk));
    } catch (const DeviceError& e) {
        // A granularity that does not fit in VRAM (the whole arena of a large model) is skipped, not fatal.
        std::fprintf(stderr, "stream_bandwidth: %s\n", e.what());
        st.seconds = -1.0;
        return st;
    }
    DeviceArena& destination = *held;
    auto pass = [&] {
        for (uint64_t off = 0; off < bytes;) {
            const uint64_t n = std::min(chunk, bytes - off);
            runtime.copy_async(destination.base(), src + off, n);
            off += n;
        }
        runtime.finish(runtime.transfer());
    };
    pass();  // Untimed page-fault and transfer setup warm-up.
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) pass();
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

}  // namespace strata::core
