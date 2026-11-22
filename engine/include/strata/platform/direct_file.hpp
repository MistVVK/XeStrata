// include/strata/platform/direct_file.hpp - plan v0.3 P1/P2: unbuffered asynchronous file reads.
//
// The n-gram table (26.8 GiB) stays on the SSD and must never occupy RAM, including the OS file cache. A memory
// map cannot promise that; an unbuffered read can. On Windows this is FILE_FLAG_NO_BUFFERING | OVERLAPPED with
// an I/O completion port; on Linux, O_DIRECT preads.  Both issue the reads from a small thread pool (submit only
// queues), so many reads are in flight: STRATA_IO_THREADS sets its size (default 4 on Windows, 16 on Linux).
//
// Contract of every read: offset, length and buffer address are multiples of `alignment()` (4096 here).
// Reads past end of file return the bytes that exist; `Completion::bytes` says how many.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

struct Completion {
    uint64_t tag = 0;      ///< the caller's tag from `submit`
    uint32_t bytes = 0;    ///< bytes transferred (short only at end of file)
    bool ok = false;
};

/// Pure helper: computes a power-of-two unbuffered I/O alignment >= 4096 from Windows FileStorageInfo / Linux STATX_DIOALIGN.
constexpr uint32_t round_up_sector_alignment(uint32_t logical_sector, uint32_t atomic_sector = 0) noexcept {
    uint32_t need = 4096u;
    if (logical_sector > need) need = logical_sector;
    if (atomic_sector > need) need = atomic_sector;
    if (need > (1u << 20)) need = (1u << 20);
    uint32_t p2 = 4096u;
    while (p2 < need && p2 < (1u << 20)) p2 <<= 1;
    return p2;
}

class DirectFile {
public:
    DirectFile();
    ~DirectFile();
    DirectFile(const DirectFile&) = delete;
    DirectFile& operator=(const DirectFile&) = delete;

    bool open(const std::string& path, std::string& err);
    void close();
    bool is_open() const;
    uint64_t size() const;
    static constexpr uint32_t alignment() { return 4096; }
    /// Dynamic hardware sector alignment of the open file (>= 4096; handles 8K/16K Dev Drive / ReFS / RAID volumes).
    uint32_t sector_alignment() const;

    /// Queue one read. Returns false (and sets `err`) if the request could not be queued; a queued request
    /// always produces exactly one completion.
    bool submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err);

    /// Wait for up to `max` completions; returns how many were written to `out`. `timeout_ms` < 0 waits
    /// forever; 0 polls. A `wake()` shows up as one completion with tag `WAKE_TAG`.
    int wait(Completion* out, int max, int timeout_ms);

    /// Make a thread blocked in `wait` return (thread-safe). Used to hand it newly queued work.
    static constexpr uint64_t WAKE_TAG = ~0ull;
    void wake();

    /// Allocation helpers for aligned read buffers (page-aligned, never touched by the cache manager).
    static void* alloc_aligned(size_t bytes);
    static void free_aligned(void* p);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

/// Monotonic time in microseconds, for latency accounting.
double now_us();

}  // namespace strata::platform
