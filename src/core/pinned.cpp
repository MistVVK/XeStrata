// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/core/pinned.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/device.hpp"
#include "strata/platform/memory.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <sys/stat.h>
#include <fcntl.h>
#include <cerrno>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <io.h>

#define STRATA_FSEEK64(f, o) _fseeki64((f), (long long) (o), SEEK_SET)
#else
#include <sys/mman.h>
#include <sys/file.h>
#include <unistd.h>

#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)
#endif

namespace strata::core {

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    for (uint64_t i = 0; i < n; ++i) { seed ^= p[i]; seed *= 1099511628211ull; }
    return seed;
}

namespace {
constexpr uint64_t kHugePage = 2ull << 20;
}

// The mapping registered for device copies in pieces no larger than the device's largest allocation: past it the
// UHD 770 (4 GiB) registered without an error and copied wrong bytes, the B70 (31 GB) did not.  With a layer split
// the later GPUs' too: the UHD 770 behind a B70 copied wrong bytes from the B70's pieces.  `cuts` are the offsets a
// piece may end at (a layer's start, so no expert crosses two registrations), or empty: every `limit`.
namespace {
std::vector<int> g_copy_devices;
std::atomic<uint64_t> g_registered{0};   // what the arenas hold registered on their first GPU (pin_room's fallback)

// The arena's registration on its first GPU: all of it, or STRATA_ARENA_PIN_GIB GiB where that is set (0: all).  Not
// pin_room: one GPU registered a 33 GiB arena past its shared-memory budget on a 64 GB Windows PC and ran steadily
// (#43, an Arc Pro B50); the budget is kept for the later GPUs (register_on).  `why` names the cap when there is one.
uint64_t arena_pin_cap(uint64_t bytes, std::string& why) {
    why.clear();
    if (const char* e = std::getenv("STRATA_ARENA_PIN_GIB"); e != nullptr && *e != '\0') {
        const uint64_t gib = std::strtoull(e, nullptr, 10);
        if (gib == 0) return bytes;
        why = "STRATA_ARENA_PIN_GIB=" + std::to_string(gib);
        return std::min(bytes, gib << 30);
    }
    return bytes;
}

// Registers [base, base + bytes) for the current device's copies in pieces, those that end within `cap` bytes of
// `base`; returns where the registered prefix ends (bytes when nothing capped it).
uint64_t register_pieces(uint8_t* base, uint64_t bytes, const std::vector<uint64_t>& cuts, std::vector<void*>& reg,
                         std::vector<std::pair<void*, size_t>>& pieces, uint64_t cap) {
    uint64_t limit = max_alloc_bytes();
    for (int d : g_copy_devices) {
        const uint64_t m = Runtime::at(d).device().get_info<sycl::info::device::max_mem_alloc_size>();
        limit = std::min(limit, m);
    }
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
        if (end > cap) break;
        sycl::ext::oneapi::experimental::prepare_for_device_copy(base + start, (size_t) (end - start),
                                                                 Runtime::get().context());
        reg.push_back(base + start);
        pieces.emplace_back(base + start, (size_t) (end - start));
        g_registered += end - start;
        start = end;
    }
    return start;
}

uint64_t piece_bytes(const std::vector<std::pair<void*, size_t>>& pieces) {
    uint64_t n = 0;
    for (const auto& p : pieces) n += p.second;
    return n;
}

std::string gib_text(uint64_t b) {
    char t[32];
    std::snprintf(t, sizeof t, "%.1f GiB", (double) b / 1073741824.0);
    return t;
}

// Upstream #771: with transparent huge pages "always" the kernel backs a mapping with huge pages where it can; an
// extra MADV_HUGEPAGE only adds direct reclaim and compaction on every fault (defrag=madvise), which on a fragmented
// machine stretched a 25 s start to 432 s.  So it is not asked for there; STRATA_NO_ARENA_THP=1 skips it anywhere.
// Returns why it was skipped, or an empty string when it should be asked for.  Linux only: Windows has no THP.
#if !defined(_WIN32)
// bytes in free blocks of 2 MiB or more (/proc/buddyinfo, order >= 9): what MADV_HUGEPAGE can get without compaction
uint64_t free_huge_bytes() {
    uint64_t total = 0;
    if (std::FILE* f = std::fopen("/proc/buddyinfo", "r")) {
        char line[512];
        while (std::fgets(line, sizeof line, f) != nullptr) {
            const char* z = std::strstr(line, "zone");
            if (z == nullptr) continue;
            const char* q = z + 4;
            while (*q == ' ') ++q;
            while (*q != '\0' && *q != ' ') ++q;   // the zone's name
            unsigned long long c[11] = {0};
            int n = 0;
            for (; n < 11; ++n) {
                char* e = nullptr;
                const unsigned long long v = std::strtoull(q, &e, 10);
                if (e == q) break;
                c[n] = v;
                q = e;
            }
            for (int o = 9; o < n; ++o) total += c[o] * (4096ull << o);
        }
        std::fclose(f);
    }
    return total;
}

std::string thp_request_skipped(uint64_t bytes) {
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
    if (mode == "always")
        return "transparent huge pages are on for every mapping (THP always): MADV_HUGEPAGE not requested";
    // upstream 6b395578 (#2126): with THP and defrag "madvise", every fault of a MADV_HUGEPAGE mapping reclaims and
    // compacts until it finds a 2 MiB block.  When the free 2 MiB blocks cover less than half of the arena (a 62 GB
    // machine holding 47 GB of anonymous memory: the load never ended, at 100% system time), that is minutes of kernel
    // work per GiB for a TLB gain.  STRATA_ARENA_THP=1 asks for them anyway.
    if (std::getenv("STRATA_ARENA_THP") == nullptr) {
        const uint64_t free_huge = free_huge_bytes();
        if (free_huge < bytes / 2) {
            std::string why = "transparent huge pages NOT requested: only " + std::to_string(free_huge >> 30) +
                              " GiB of free 2 MiB blocks for a " + std::to_string(bytes >> 30) +
                              " GiB arena, compaction would stall the load (STRATA_ARENA_THP=1 requests them anyway)";
            std::fprintf(stderr, "strata: expert arena: %s\n", why.c_str());   // now: the load that follows can be long
            return why;
        }
    }
    return std::string();
}
#endif
}  // namespace

void arena_copy_devices(const std::vector<int>& devices) { g_copy_devices = devices; }

uint64_t pin_room(int device, std::string& why) {
#if defined(_WIN32)
    constexpr uint64_t kMargin = 4ull << 30;   // upstream #243's margin below the budget
    std::string pci, dx_why = "the GPU reports no PCI address";
    try {
        const sycl::device d = (device < 0 ? Runtime::get() : Runtime::at(device)).device();
        if (d.has(sycl::aspect::ext_intel_pci_address)) {
            pci = d.get_info<sycl::ext::intel::info::device::pci_address>();
            for (auto& c : pci) c = (char) std::tolower((unsigned char) c);
        }
    } catch (const std::exception& e) {
        dx_why = e.what();
    }
    uint64_t budget = 0, usage = 0;
    if (!pci.empty() && platform::gpu_nonlocal_memory_budget(pci, budget, usage, dx_why)) {
        why = "Windows: the GPU's shared-memory budget " + gib_text(budget) + ", " + gib_text(usage) +
              " of it in use, less 4 GiB";
        return budget > usage + kMargin ? budget - usage - kMargin : 0;
    }
    // upstream's fallback where DXGI does not say: half the RAM less 8 GiB, less what the arenas hold already
    const uint64_t ram = platform::total_physical_memory(), half = ram / 2 > (8ull << 30) ? ram / 2 - (8ull << 30) : 0;
    const uint64_t held = g_registered.load();
    why = "Windows: half the RAM less 8 GiB (" + dx_why + ")";
    return half > held ? half - held : 0;
#else
    (void) device;
    why.clear();
    return UINT64_MAX;
#endif
}

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, slice, std::vector<uint64_t>{}) {}

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice, const std::vector<uint64_t>& cuts) : capacity(bytes) {
    (void) slice;   // the registrations cover any slice geometry
    if (!bytes || bytes > std::numeric_limits<size_t>::max() - kHugePage)
        throw DeviceError("the expert arena requires a nonzero size");
#if defined(_WIN32)
    // Windows has no transparent huge pages (its large pages need a privilege an ordinary account lacks): 4 KB pages,
    // committed against the page file up front, as Windows has nothing like MAP_NORESERVE.
    map_bytes_ = (size_t) bytes;
    map_ = VirtualAlloc(nullptr, map_bytes_, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (map_ == nullptr)
        throw DeviceError("VirtualAlloc of the expert arena failed for " + std::to_string(bytes) + " bytes (error " +
                          std::to_string(GetLastError()) + ")");
    base = map_;
    const bool huge = false;
#else
    // Over-map by one huge page so the arena starts on a 2 MB boundary and THP can back it from the first byte.
    map_bytes_ = (size_t) bytes + kHugePage;
    map_ = mmap(nullptr, map_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        throw DeviceError("mmap of the expert arena failed for " + std::to_string(bytes) + " bytes");
    }
    base = (void*) (((uintptr_t) map_ + kHugePage - 1) & ~(uintptr_t) (kHugePage - 1));
    const std::string thp_skip = thp_request_skipped(bytes);
    const bool huge = thp_skip.empty() && madvise(base, (size_t) bytes, MADV_HUGEPAGE) == 0;
#endif
    // Registered before the loader touches a page: the B70 run showed later writes reach the device intact.
    std::string capped;
    try {
        registered_bytes = register_pieces((uint8_t*) base, bytes, cuts, reg_, pieces_, arena_pin_cap(bytes, capped));
    } catch (const std::exception& e) {
        for (void* r : reg_) sycl::ext::oneapi::experimental::release_from_device_copy(r, Runtime::get().context());
        reg_.clear();
        pieces_.clear();
        unmap(false);
        throw DeviceError(std::string("registering the expert arena for device copies failed: ") + e.what());
    }
    backing = huge ? PageBacking::LargePages : PageBacking::NormalPages;
    registered_slices = 1;
    slice_starts = {0};
    note = std::string("one mapping registered for device copies") +
           (reg_.size() > 1 ? " in " + std::to_string(reg_.size()) + " pieces (the GPU's largest allocation)" : "") +
           (registered_bytes < bytes ? ", its first " + gib_text(registered_bytes) + " of " + gib_text(bytes) +
                                           " (" + capped + "; the rest is copied unregistered, slower)"
                                     : "") +
           "; " +
#if defined(_WIN32)
           "4 KB pages; no VirtualLock";
#else
           (!thp_skip.empty() ? thp_skip
            : huge             ? "transparent huge pages requested (madvise)"
                               : "madvise(MADV_HUGEPAGE) refused: 4 KB pages") +
           "; no mlock";
#endif
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
#if defined(_WIN32)
// The backing file is a CRT descriptor, as on Linux; its handle does the locking, the mapping and the positioned I/O.
HANDLE os_handle(int fd) { return (HANDLE) _get_osfhandle(fd); }
bool try_lock(int fd) {
    OVERLAPPED ov{};
    return LockFileEx(os_handle(fd), LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, MAXDWORD, MAXDWORD, &ov);
}
bool lock_busy() { return GetLastError() == ERROR_LOCK_VIOLATION; }
void unlock(int fd) {
    OVERLAPPED ov{};
    UnlockFileEx(os_handle(fd), 0, MAXDWORD, MAXDWORD, &ov);
}
bool pread_all(int fd, void* p, uint64_t n, uint64_t off) {
    OVERLAPPED ov{};
    ov.Offset = (DWORD) off;
    ov.OffsetHigh = (DWORD) (off >> 32);
    DWORD got = 0;
    return ReadFile(os_handle(fd), p, (DWORD) n, &got, &ov) && got == n;
}
bool pwrite_all(int fd, const void* p, uint64_t n, uint64_t off) {
    OVERLAPPED ov{};
    ov.Offset = (DWORD) off;
    ov.OffsetHigh = (DWORD) (off >> 32);
    DWORD put = 0;
    return WriteFile(os_handle(fd), p, (DWORD) n, &put, &ov) && put == n;
}
bool sync_file(int fd) { return FlushFileBuffers(os_handle(fd)) != 0; }
bool file_bytes(int fd, uint64_t& size) {
    struct _stat64 st{};
    if (_fstat64(fd, &st) != 0) return false;
    size = (uint64_t) st.st_size;
    return true;
}
bool resize_file(int fd, uint64_t size) { return _chsize_s(fd, (long long) size) == 0; }
void close_file(int fd) { _close(fd); }
#else
bool try_lock(int fd) { return flock(fd, LOCK_EX | LOCK_NB) == 0; }
bool lock_busy() { return errno == EWOULDBLOCK || errno == EINTR; }
void unlock(int fd) { flock(fd, LOCK_UN); }
bool pread_all(int fd, void* p, uint64_t n, uint64_t off) { return pread(fd, p, n, (off_t) off) == (ssize_t) n; }
bool pwrite_all(int fd, const void* p, uint64_t n, uint64_t off) { return pwrite(fd, p, n, (off_t) off) == (ssize_t) n; }
bool sync_file(int fd) { return fdatasync(fd) == 0; }
bool file_bytes(int fd, uint64_t& size) {
    struct stat st{};
    if (fstat(fd, &st) != 0) return false;
    size = (uint64_t) st.st_size;
    return true;
}
bool resize_file(int fd, uint64_t size) { return ftruncate(fd, (off_t) size) == 0; }
void close_file(int fd) { ::close(fd); }
#endif

void arena_lock(int fd) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (!try_lock(fd)) {
        if (!lock_busy()) throw DeviceError("shared arena: file lock failed");
        if (std::chrono::steady_clock::now() >= end) throw DeviceError("shared arena: population lock timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, uint64_t max_pinned_bytes,
                         const std::string& file, uint64_t pack_hash, const std::vector<uint64_t>& cuts)
    : capacity(bytes), bounds_(bounds) {
    (void) max_pinned_bytes;   // Xe registers every slice, bounded by the device's largest allocation.
    if (!bytes || bytes > (uint64_t) std::numeric_limits<int64_t>::max() - kArenaHeaderBytes - kHugePage)
        throw DeviceError("shared arena: invalid size");
#if defined(_WIN32)
    shared_fd_ = _open(file.c_str(), _O_RDWR | _O_CREAT | _O_BINARY | _O_NOINHERIT, _S_IREAD | _S_IWRITE);
#else
    shared_fd_ = open(file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
#endif
    if (shared_fd_ < 0) throw DeviceError("shared arena: cannot open backing file");
    try {
        arena_lock(shared_fd_);
        shared_locked_ = true;
        uint64_t file_size = 0;
        if (!file_bytes(shared_fd_, file_size)) throw DeviceError("shared arena: cannot size backing file");
        SharedArenaHeader hdr;
        if (file_size == 0) {
            hdr.arena_bytes = bytes; hdr.pack_hash = pack_hash;
            if (!resize_file(shared_fd_, bytes + kArenaHeaderBytes) || !pwrite_all(shared_fd_, &hdr, sizeof hdr, 0) ||
                !sync_file(shared_fd_))
                throw DeviceError("shared arena: cannot initialize header");
        } else {
            if (file_size != bytes + kArenaHeaderBytes) {
                note = "shared arena: file size differs from expected arena size";
            } else if (!pread_all(shared_fd_, &hdr, sizeof hdr, 0) ||
                       std::memcmp(hdr.magic, "STRATA-ARENA-V1", 16) != 0 || hdr.version != 1 ||
                       hdr.header_bytes != kArenaHeaderBytes || hdr.arena_bytes != bytes) {
                note = "shared arena: incompatible header";
            } else if (hdr.pack_hash != pack_hash) {
                note = "shared arena: pack hash mismatch";
            }
            if (!note.empty()) {
                unlock(shared_fd_); shared_locked_ = false;
                close_file(shared_fd_); shared_fd_ = -1;
                return;
            }
        }
#if defined(_WIN32)
        // A view starts at a multiple of the allocation granularity (64 KiB), so the header is mapped with the arena
        // and the arena starts 4 KiB in.  The mapping object closes at once: the view keeps it.
        HANDLE mapping = CreateFileMappingW(os_handle(shared_fd_), nullptr, PAGE_READWRITE, 0, 0, nullptr);
        if (mapping == nullptr) throw DeviceError("shared arena: mapping failed");
        map_bytes_ = (size_t) (bytes + kArenaHeaderBytes);
        map_ = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, map_bytes_);
        CloseHandle(mapping);
        if (map_ == nullptr) throw DeviceError("shared arena: mapping failed");
        base = (uint8_t*) map_ + kArenaHeaderBytes;
        const bool huge = false;
#else
        // Reserve the address range first: file offset 4096 can still start at a 2 MiB-aligned virtual address.
        map_bytes_ = (size_t) bytes + kHugePage;
        map_ = mmap(nullptr, map_bytes_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (map_ == MAP_FAILED) { map_ = nullptr; throw DeviceError("shared arena: address reservation failed"); }
        base = (void*) (((uintptr_t) map_ + kHugePage - 1) & ~(uintptr_t) (kHugePage - 1));
        if (mmap(base, (size_t) bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, shared_fd_,
                 (off_t) kArenaHeaderBytes) == MAP_FAILED) {
            base = nullptr; throw DeviceError("shared arena: mapping failed");
        }
        const bool huge = thp_request_skipped(bytes).empty() && madvise(base, (size_t) bytes, MADV_HUGEPAGE) == 0;
#endif
        std::string capped;
        registered_bytes = register_pieces((uint8_t*) base, bytes, cuts.empty() ? bounds : cuts, reg_, pieces_,
                                           arena_pin_cap(bytes, capped));
        registered_slices = 1; slice_starts = {0};
        backing = huge ? PageBacking::LargePages : PageBacking::NormalPages;
        note = "shared file mapping registered in device-sized slices" +
               (registered_bytes < bytes ? ", its first " + gib_text(registered_bytes) + " of " + gib_text(bytes) +
                                               " (" + capped + ")"
                                         : std::string()) + "; no mlock";
        unlock(shared_fd_); shared_locked_ = false;
    } catch (...) {
        for (void* r : reg_) sycl::ext::oneapi::experimental::release_from_device_copy(r, Runtime::get().context());
        reg_.clear();
        unmap(true);
        close_file(shared_fd_); shared_fd_ = -1;
        throw;
    }
}

bool PinnedArena::begin_shared_population() {
    if (shared_fd_ < 0) return true;
    arena_lock(shared_fd_); shared_locked_ = true;
    SharedArenaHeader hdr;
    if (!pread_all(shared_fd_, &hdr, sizeof hdr, 0)) throw DeviceError("shared arena: cannot read population state");
    if (hdr.reserved[0] == 1) {
        unlock(shared_fd_); shared_locked_ = false;
        return false;
    }
    return true;
}

void PinnedArena::publish_shared() {
    if (shared_fd_ < 0) return;
    if (!shared_locked_) throw DeviceError("shared arena: population lock is not held");
    // The complete body must reach storage before another process can observe its completion record.
#if defined(_WIN32)
    const bool flushed = FlushViewOfFile(base, (size_t) capacity) != 0;
#else
    const bool flushed = msync(base, (size_t) capacity, MS_SYNC) == 0;
#endif
    if (!flushed || !sync_file(shared_fd_)) throw DeviceError("shared arena: cannot persist populated weights");
    const uint64_t complete = 1;
    if (!pwrite_all(shared_fd_, &complete, sizeof complete, offsetof(SharedArenaHeader, reserved)) ||
        !sync_file(shared_fd_))
        throw DeviceError("shared arena: cannot publish completed weights");
    unlock(shared_fd_); shared_locked_ = false;
}

bool PinnedArena::to_host_usm(uint64_t pad, std::string& err, const std::function<uint64_t()>& ram_available) {
    if (!layer_base.empty()) return true;
    if (base == nullptr || map_ == nullptr || shared_fd_ >= 0 || bounds_.size() < 2) {
        err = "only a private mapping with layer bounds moves into host USM";
        return false;
    }
    auto& runtime = Runtime::at(owner_);
    namespace sx = sycl::ext::oneapi::experimental;
    // registered on a layer split's later GPU of another runtime as well (hold_registration; a helper GPU's goes, as
    // before): this runtime's host USM is its GPU driver's memory, which the other refuses to register (the B70's on an RTX 4070: CUDA_ERROR_INVALID_VALUE, 2026-11-01), so the move would
    // leave that GPU's copies pageable; the mapping stays, registered for copies on every GPU
    for (const auto& o : other_reg_)
        if (std::find(held_.begin(), held_.end(), o.first) != held_.end() &&
            Runtime::at(o.first).device().get_backend() != runtime.device().get_backend()) {
            err = "the arena is registered on " + Runtime::at(o.first).device().get_info<sycl::info::device::name>() +
                  " as well, whose runtime cannot register this one's host USM";
            return false;
        }
    try {   // no copy may read the mapping while it moves, and its registrations go
        runtime.finish();
        for (auto& o : other_reg_) {
            auto& r = Runtime::at(o.first);
            r.finish();
            for (void* q : o.second) sx::release_from_device_copy(q, r.context());
        }
        other_reg_.clear();
        for (void* r : reg_) sx::release_from_device_copy(r, runtime.context());
        reg_.clear();
        g_registered -= std::min(g_registered.load(), piece_bytes(pieces_));
    } catch (const std::exception& e) {
        err = std::string("releasing the arena's registrations: ") + e.what();
        return false;
    }
    std::vector<uint8_t*> blocks;
    const size_t layers = bounds_.size() - 1;
    for (size_t l = 0; l < layers; ++l) {
        const uint64_t n = bounds_[l + 1] - bounds_[l];
        // Issue #10: host USM is GPU driver memory, which the OS cannot take back when RAM runs short (the GPU VM's
        // rebind swapped it in and the OOM killer emptied the host).  Each layer moves only while the RAM has room
        // for it and a reserve (STRATA_HOST_USM_RESERVE_MIB, default 1024); otherwise the arena stays as it was.
        static const uint64_t reserve = [] {
            const char* v = std::getenv("STRATA_HOST_USM_RESERVE_MIB");
            return (uint64_t) (v != nullptr ? std::strtoull(v, nullptr, 10) : 1024) << 20;
        }();
        const uint64_t avail = ram_available ? ram_available() : UINT64_MAX;   // the caller's reading of the RAM
        const bool room = avail >= n + pad + reserve;
        void* q = room ? sycl::malloc_host((size_t) (n + pad), runtime.context()) : nullptr;
        if (q == nullptr) {   // back as it was: the moved layers into the mapping, then its registrations
            for (size_t k = 0; k < blocks.size(); ++k) {
                std::memcpy((uint8_t*) base + bounds_[k], blocks[k], (size_t) (bounds_[k + 1] - bounds_[k]));
                runtime.free(blocks[k]);
            }
            try {
                for (const auto& pc : pieces_) {
                    sx::prepare_for_device_copy(pc.first, pc.second, runtime.context());
                    reg_.push_back(pc.first);
                    g_registered += pc.second;
                }
            } catch (const std::exception& e) {
                err = std::string("host USM refused at layer ") + std::to_string(l) + ", and the mapping's registration " +
                      "again failed (" + e.what() + "): its copies are slower";
                return false;
            }
            err = room ? "host USM refused at layer " + std::to_string(l) + " (" + std::to_string(n + pad) + " B)"
                   : "the RAM has no room for layer " + std::to_string(l) + " in host USM (" +
                         std::to_string(avail >> 20) + " MiB available)";
            return false;
        }
        std::memcpy(q, (uint8_t*) base + bounds_[l], (size_t) n);
        if (pad > 0) std::memset((uint8_t*) q + n, 0, (size_t) pad);
        blocks.push_back((uint8_t*) q);
        // the moved layer's whole pages back to the system (MADV_DONTNEED on a private mapping drops them)
        const uintptr_t a = ((uintptr_t) base + bounds_[l] + 4095) & ~(uintptr_t) 4095;
        const uintptr_t b = ((uintptr_t) base + bounds_[l + 1]) & ~(uintptr_t) 4095;
#if defined(_WIN32)
        if (b > a && shared_fd_ < 0) VirtualFree((void*) a, (size_t) (b - a), MEM_DECOMMIT);
#else
        if (b > a) madvise((void*) a, (size_t) (b - a), MADV_DONTNEED);
#endif
    }
    unmap(shared_fd_ >= 0);
    map_bytes_ = 0;
    layer_base = std::move(blocks);
    pieces_.clear();
    slice_starts.clear();
    for (size_t l = 0; l < layers; ++l) {
        pieces_.emplace_back(layer_base[l], (size_t) (bounds_[l + 1] - bounds_[l] + pad));
        slice_starts.push_back(bounds_[l]);
    }
    base = layer_base[0];
    backing = PageBacking::NormalPages;
    registered_bytes = capacity;   // all of it, also where the registration was a prefix (arena_pin_cap)
    slice_bytes = 1;
    registered_slices = (int) layers;
    note = "host USM, one block per layer (" + std::to_string(layers) +
           "), moved from the registered mapping: readable by GPU kernels for the PCIe share";
    return true;
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
    {
        // Windows: that GPU's shared-memory budget is its own; past it the registration is left out (copies are
        // correct without it, only slower), as when the driver refuses it
        std::string why;
        const uint64_t room = pin_room(device, why), want = piece_bytes(pieces_);
        if (room != UINT64_MAX && want > room) {
            err = "the registration (" + gib_text(want) + ") does not fit in " + gib_text(room) + " (" + why + ")";
            return false;
        }
    }
    std::vector<void*> regs;
    try {
        const sycl::context ctx = Runtime::at(device).context();
        for (const auto& p : pieces_) {
            sycl::ext::oneapi::experimental::prepare_for_device_copy(p.first, p.second, ctx);
            regs.push_back(p.first);
        }
        // Copies through the registration are read back once, both ends of every piece: on the UHD 770 beside a B70
        // (Level Zero, 2026-10-15) a registration on the second GPU was taken without an error and that GPU's copies
        // from the arena then came back wrong, where the same copies without it were right.  Such a GPU copies
        // from the arena unregistered.
        auto& rt = Runtime::at(device);
        constexpr size_t kProbe = 64 << 10;
        auto* d = static_cast<uint8_t*>(sycl::malloc_device(kProbe, rt.device(), rt.context()));
        if (d == nullptr) throw std::runtime_error("no device memory to check the registration");
        std::vector<uint8_t> back(kProbe);
        std::string wrong;
        for (const auto& p : pieces_) {
            const size_t n = std::min(kProbe, p.second);
            for (const size_t off : {(size_t) 0, p.second - n}) {
                const uint8_t* src = static_cast<const uint8_t*>(p.first) + off;
                rt.compute().memcpy(d, src, n).wait();
                rt.compute().memcpy(back.data(), d, n).wait();
                if (wrong.empty() && std::memcmp(back.data(), src, n) != 0)
                    wrong = "copies through it came back wrong (byte " + std::to_string(off) + " of a " +
                            std::to_string(p.second) + "-byte piece)";
            }
        }
        rt.free(d);
        if (!wrong.empty()) throw std::runtime_error(wrong);
    } catch (const std::exception& e) {
        for (void* r : regs)
            sycl::ext::oneapi::experimental::release_from_device_copy(r, Runtime::at(device).context());
        err = std::string("registering the expert arena on GPU ") + std::to_string(device) + ": " + e.what();
        return false;
    }
    other_reg_.emplace_back(device, std::move(regs));
    return true;
}

void PinnedArena::unmap(bool file_view) {
    if (map_ != nullptr) {
#if defined(_WIN32)
        if (file_view) UnmapViewOfFile(map_);
        else VirtualFree(map_, 0, MEM_RELEASE);
#else
        (void) file_view;
        munmap(map_, map_bytes_);
#endif
    }
    map_ = base = nullptr;
}

PinnedArena::~PinnedArena() {
    const bool file_view = shared_fd_ >= 0;
    if (shared_fd_ >= 0) close_file(shared_fd_);
    for (auto& o : other_reg_) {
        try {
            auto& r = Runtime::at(o.first);
            r.finish();   // no copy may still read the mapping
            for (void* p : o.second) sycl::ext::oneapi::experimental::release_from_device_copy(p, r.context());
        } catch (const std::exception& e) {
            // The driver can refuse the release itself when it is in the state that refused the allocations
            // (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, the B70's xe driver, 2026-10-22).  The exit goes on: the
            // process's end reclaims the mapping with the driver, and a stop here would lose that message.
            std::fprintf(stderr, "Xe runtime: releasing the expert arena on GPU %d failed: %s; the exit continues "
                                 "(the driver reclaims it with the process)\n", o.first, e.what());
            std::fflush(stderr);
        }
    }
    if (!layer_base.empty()) {
        for (uint8_t* p : layer_base) Runtime::at(owner_).free(p);
        return;
    }
    if (!base) return;
    auto& runtime = Runtime::at(owner_);
    try {
        runtime.finish();   // no copy may still read the mapping
        for (void* r : reg_) sycl::ext::oneapi::experimental::release_from_device_copy(r, runtime.context());
        if (!reg_.empty()) g_registered -= std::min(g_registered.load(), piece_bytes(pieces_));
    } catch (const std::exception& e) {
        // as above: the driver can refuse the release in the state that refused the allocations - report it and
        // let the exit finish, the process's end reclaims the mapping
        std::fprintf(stderr, "Xe runtime: releasing the expert arena failed: %s; the exit continues (the driver "
                             "reclaims it with the process)\n", e.what());
        std::fflush(stderr);
    }
    unmap(file_view);
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
