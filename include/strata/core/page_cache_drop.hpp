// give the page cache of the model files back while and after the expert arena is read, on an integrated AMD GPU under Linux.
//
// Radeon 780M (gfx1103), Linux.  The expert arena (31.6 GiB of the Q2_0 pack) is filled with buffered reads of experts.bin and registered with the
// driver (a KFD userptr); the GPU's own buffers take about 12 GiB of the same RAM, and the file cache of the model files the start read fills most
// of what is left: MemFree 1.6-2.4 GiB, Cached 6.9-7.6 GiB at READY.  With free memory that small and that broken up, the kernel's proactive
// compaction (kcompactd, fragmentation score above its high mark) scans and migrates the node's pages in passes: one right after the fill and,
// when a pass made no progress, the next 32 s later (500 ms << COMPACT_MAX_DEFER_SHIFT).  A pass that moves pages of the registered arena makes
// KFD evict the process's queues, and the gc 11 MES firmware never answers the eviction while a dispatch sits in a barrier (
// passive starts under a g++ load, late passes in 3 of 8; with the file cache given back by the harness at this point, 0 of 8, MemFree at READY
// 8.4-9.3 GiB, order 9 / 10 free blocks 0.8-3.3 GiB against 0.01-0.25 GiB).
//
// So the engine gives the cache back: posix_fadvise(POSIX_FADV_DONTNEED) behind the read position of each chunk of experts.bin as it is copied
// into the arena, and once per model file after the load (experts.bin, the pack's dense.bin and embd.bin, the GGUF shards of --native except the
// PLE table's, whose rows are read directly).  It is advice for clean pages: a file still mapped keeps its mapped pages, a failure is ignored, no
// file is written or deleted, and what a later reader needs is read from the drive again.  Automatic on the device class (arch_defaults.hpp,
// igpu_cache_drop_default); STRATA_NO_ARENA_CACHE_DROP=1 turns it off (research / A-B only).
//
// Pure pieces, so they can be tested without a GPU: drop_range (one range of an open file), drop_file (a whole file by path), enabled, the
// meminfo reader, the draft line.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strata::core {

/// Set by the engine on the device class this is for (arch_defaults.hpp, igpu_cache_drop_default); enabled() is false while it is false.
inline std::atomic<bool>& arena_cache_drop_flag() {
    static std::atomic<bool> f{false};
    return f;
}

namespace page_cache_drop {

/// STRATA_NO_ARENA_CACHE_DROP: anything but empty or 0 turns the give-back off.
inline bool off_by_env() {
    const char* e = std::getenv("STRATA_NO_ARENA_CACHE_DROP");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}

/// Whether the engine gives the cache back: Linux, the device class set the flag, the switch is not set.
inline bool enabled() {
#if defined(__linux__)
    return arena_cache_drop_flag().load(std::memory_order_relaxed) && !off_by_env();
#else
    return false;
#endif
}

/// Bytes the engine has asked the kernel to drop behind its reads so far (drop_range's successful calls).
inline std::atomic<uint64_t>& advised_bytes() {
    static std::atomic<uint64_t> n{0};
    return n;
}

/// Advice for one range of an open file: its cached clean pages may go (pages that are mapped, dirty or in use stay).  False when the range
/// is empty, the descriptor is not valid or the kernel refused; never an error for the caller.
inline bool drop_range(int fd, uint64_t off, uint64_t len) {
#if defined(__linux__)
    if (fd < 0 || len == 0) return false;
    if (::posix_fadvise(fd, (off_t) off, (off_t) len, POSIX_FADV_DONTNEED) != 0) return false;
    advised_bytes().fetch_add(len, std::memory_order_relaxed);
    return true;
#else
    (void) fd; (void) off; (void) len;
    return false;
#endif
}

/// The same for a whole file by path (opened read-only for this call only).  False when it cannot be opened or the kernel refused.
inline bool drop_file(const std::string& path) {
#if defined(__linux__)
    if (path.empty()) return false;
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0;   // length 0 = to the end of the file
    ::close(fd);
    return ok;
#else
    (void) path;
    return false;
#endif
}

/// One value of /proc/meminfo in bytes ("Cached", "MemFree" ...); 0 where it cannot be read.
inline uint64_t meminfo_bytes(const char* key) {
#if defined(__linux__)
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) return 0;
    char line[256];
    const size_t n = std::strlen(key);
    uint64_t v = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        if (std::strncmp(line, key, n) == 0 && line[n] == ':') {
            v = (uint64_t) std::strtoull(line + n + 1, nullptr, 10) * 1024ull;
            break;
        }
    }
    std::fclose(f);
    return v;
#else
    (void) key;
    return 0;
#endif
}

struct Outcome {
    size_t files = 0;              ///< the files the final call went to (empty and repeated paths left out)
    size_t failed = 0;             ///< of them, the calls that did not work (ignored)
    uint64_t advised_behind = 0;   ///< bytes asked back behind the reads before the final call
    uint64_t cached_before = 0;    ///< /proc/meminfo Cached before the final call, bytes
    uint64_t cached_after = 0;     ///< and after it
    double ms = 0.0;               ///< the final call's time
};

/// The final call: every file once.  Empty and repeated paths are skipped, a failure is counted and ignored.
inline Outcome give_back(const std::vector<std::string>& files) {
    Outcome o;
    o.advised_behind = advised_bytes().load(std::memory_order_relaxed);
    o.cached_before = meminfo_bytes("Cached");
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::string> seen;
    for (const std::string& p : files) {
        if (p.empty()) continue;
        bool dup = false;
        for (const std::string& s : seen) dup = dup || s == p;
        if (dup) continue;
        seen.push_back(p);
        ++o.files;
        if (!drop_file(p)) ++o.failed;
    }
    o.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    o.cached_after = meminfo_bytes("Cached");
    return o;
}

/// The one line the engine prints: empty when nothing was given back.
inline std::string draft_line(const Outcome& o) {
    const double gib = 1073741824.0;
    const uint64_t after_call = o.cached_before > o.cached_after ? o.cached_before - o.cached_after : 0;
    if (o.advised_behind == 0 && after_call == 0) return std::string();
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "the model files' cached pages were given back after the expert load: %.1f GiB asked back during the read, %.1f GiB "
                  "by the call after it (%.0f ms); the file cache is %.1f GiB now (STRATA_NO_ARENA_CACHE_DROP=1 keeps it)",
                  (double) o.advised_behind / gib, (double) after_call / gib, o.ms, (double) o.cached_after / gib);
    return std::string(buf);
}

}  // namespace page_cache_drop
}  // namespace strata::core
