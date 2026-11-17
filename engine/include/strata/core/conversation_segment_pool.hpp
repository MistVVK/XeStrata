#pragma once
// STRATA_PARK_SEGPOOL_MIB (research, default off): a pre-touched arena for the host buffers of parked conversations.
//
// A park writes a conversation's K/V into freshly allocated host segments (ConversationBuffer); the first touch of fresh
// pages costs ~0.4 ms per MB (l620: 43 ms for an 8K park's 100 MB, 213-407 ms for a 64K park's 829 MB on a DGX Spark).
// The pool reserves ONE region once, makes it resident in the background, and hands out sub-ranges of it to any
// ConversationBuffer segment of 512 KiB or more; a freed range goes back to the pool with its pages still resident.
// A request the pool cannot serve (too small, no free run, pool off) is a plain ::operator new, as before.
//
// Everything OS-specific is behind SegmentPoolOs (reserve / unreserve / commit / populate / release), so the allocator
// logic is the same on every system and a test can count the calls.  The Linux implementation is below; a Windows one is
// VirtualAlloc(MEM_RESERVE) for reserve, VirtualAlloc(MEM_COMMIT) for commit, a touch of every page (or
// PrefetchVirtualMemory) for populate and DiscardVirtualMemory / VirtualFree(MEM_DECOMMIT) for release (notes in
// par\RESULTS-PARK-POOL.md); until it exists reserve() returns nullptr and the pool is inert.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <sys/mman.h>
#endif

namespace strata::core {

class SegmentPoolOs {
public:
    virtual ~SegmentPoolOs() = default;
    /// Address space of `bytes` (a multiple of 2 MiB), 2 MiB aligned, nothing resident.  nullptr: not possible here.
    virtual void* reserve(size_t bytes) = 0;
    virtual void unreserve(void* base, size_t bytes) = 0;
    /// Makes [p, p + n) usable (a no-op where reserved pages fault in on touch).  false: not possible.
    virtual bool commit(void* p, size_t n) = 0;
    /// Makes [p, p + n) resident without changing a byte a user of the range may already hold.  false: not possible.
    virtual bool populate(void* p, size_t n) = 0;
    /// Gives the pages back to the system; the range stays reserved and usable (its content is gone).
    virtual void release(void* p, size_t n) = 0;
};

#if defined(__linux__)
class LinuxSegmentPoolOs final : public SegmentPoolOs {
public:
    void* reserve(size_t bytes) override {
        constexpr size_t huge = size_t(2) << 20;
        if (bytes == 0 || bytes % huge != 0) return nullptr;
        void* raw = ::mmap(nullptr, bytes + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (raw == MAP_FAILED) return nullptr;
        const uintptr_t a = (reinterpret_cast<uintptr_t>(raw) + huge - 1) & ~(uintptr_t(huge) - 1);
        const size_t head = a - reinterpret_cast<uintptr_t>(raw);
        if (head) ::munmap(raw, head);
        const size_t tail = huge - head;
        if (tail) ::munmap(reinterpret_cast<void*>(a + bytes), tail);
        return reinterpret_cast<void*>(a);
    }
    void unreserve(void* base, size_t bytes) override { ::munmap(base, bytes); }
    bool commit(void*, size_t) override { return true; }
    bool populate(void* p, size_t n) override {
#ifndef MADV_POPULATE_WRITE
        constexpr int populate_write = 23;   // Linux 5.14
#else
        constexpr int populate_write = MADV_POPULATE_WRITE;
#endif
        if (::madvise(p, n, populate_write) == 0) return true;
        if (errno != EINVAL) return false;   // no memory (or a refusal): never fall back to touching pages then
        // a kernel without MADV_POPULATE_WRITE: one atomic write (an OR with 0) per page makes it resident and cannot undo a store the
        // host thread makes to the same byte meanwhile
        uint8_t* b = static_cast<uint8_t*>(p);
        for (size_t o = 0; o < n; o += 4096) __atomic_fetch_or(b + o, uint8_t(0), __ATOMIC_RELAXED);
        return true;
    }
    void release(void* p, size_t n) override { ::madvise(p, n, MADV_DONTNEED); }
};
inline SegmentPoolOs& segment_pool_os() { static LinuxSegmentPoolOs os; return os; }
#else
class NoSegmentPoolOs final : public SegmentPoolOs {
public:
    void* reserve(size_t) override { return nullptr; }
    void unreserve(void*, size_t) override {}
    bool commit(void*, size_t) override { return false; }
    bool populate(void*, size_t) override { return false; }
    void release(void*, size_t) override {}
};
inline SegmentPoolOs& segment_pool_os() { static NoSegmentPoolOs os; return os; }
#endif

/// What the conversation cache needs from the pool to keep it inside its own budget (see ConversationCache::set_pool).
class SegmentPoolBudget {
public:
    virtual ~SegmentPoolBudget() = default;
    /// Resident bytes of the pool that no live segment uses (what the pool adds to the process beyond the images).
    virtual size_t idle_bytes() const = 0;
    /// Gives idle resident bytes back to the system (at least `want` when that many are idle); returns what it gave.
    virtual size_t shrink(size_t want) = 0;
};

/// One reserved region cut into 256 KiB granules.  A segment of 512 KiB or more takes the best-fitting free run of
/// granules (ties: the lowest address); freeing it makes the run free and warm (resident) again.
/// Thread-safe.  Destroy it only when no segment from it is alive (the process-wide pool is never destroyed).
class SegmentPool final : public SegmentPoolBudget {
public:
    static constexpr size_t granule = size_t(256) << 10;
    static constexpr size_t min_alloc = size_t(512) << 10;
    static constexpr size_t populate_step_bytes = size_t(32) << 20;

    SegmentPool(SegmentPoolOs& os, size_t bytes) : os_(os) {
        const size_t huge = size_t(2) << 20;
        const size_t usable = bytes / huge * huge;
        if (usable < huge) return;
        base_ = static_cast<uint8_t*>(os_.reserve(usable));
        if (base_ == nullptr) return;
        n_ = usable / granule;
        state_.assign(n_, kFreeCold);
    }
    ~SegmentPool() override { if (base_) os_.unreserve(base_, n_ * granule); }
    SegmentPool(const SegmentPool&) = delete;
    SegmentPool& operator=(const SegmentPool&) = delete;

    bool active() const { return base_ != nullptr; }
    size_t capacity() const { return n_ * granule; }
    bool contains(const void* p) const {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        return base_ != nullptr && b >= base_ && b < base_ + n_ * granule;
    }

    /// A range of at least `bytes` (nullptr when this request is not served: too small, no free run, pool inert)
    void* allocate(size_t bytes) {
        if (base_ == nullptr || bytes < min_alloc) return nullptr;
        const size_t need = granules_for(bytes);
        std::lock_guard<std::mutex> lk(mu_);
        size_t best = n_, best_len = 0;
        for (size_t i = 0; i < n_;) {
            if (!is_free(state_[i])) { ++i; continue; }
            size_t j = i;
            while (j < n_ && is_free(state_[j])) ++j;
            const size_t len = j - i;
            if (len >= need && (best_len == 0 || len < best_len)) { best = i; best_len = len; }
            i = j;
        }
        if (best_len == 0) { ++misses_; return nullptr; }
        // cold granules of the chosen range become usable first (nothing to do on Linux); on a refusal nothing changed
        for (size_t i = best; i < best + need;) {
            if (state_[i] != kFreeCold) { ++i; continue; }
            size_t j = i;
            while (j < best + need && state_[j] == kFreeCold) ++j;
            if (!os_.commit(base_ + i * granule, (j - i) * granule)) { ++misses_; return nullptr; }
            i = j;
        }
        for (size_t i = best; i < best + need; ++i) {
            if (state_[i] == kFreeCold) ++resident_;
            state_[i] = kUsed;
        }
        live_requested_ += bytes;
        ++hits_;
        return base_ + best * granule;
    }

    /// true when `p` is this pool's (then the range is free again, its pages still resident); false: not ours
    bool deallocate(void* p, size_t bytes) {
        if (!contains(p)) return false;
        const size_t first = (static_cast<uint8_t*>(p) - base_) / granule;
        const size_t need = granules_for(bytes);
        std::lock_guard<std::mutex> lk(mu_);
        if (first + need > n_ || live_requested_ < bytes) fatal("a freed range does not fit the pool");
        for (size_t i = first; i < first + need; ++i)
            if (state_[i] != kUsed) fatal("a freed range is not in use");
        for (size_t i = first; i < first + need; ++i) state_[i] = kFreeWarm;
        live_requested_ -= bytes;
        return true;
    }

    /// Makes up to `max_bytes` (at most populate_step_bytes at a time) of cold free granules resident, off the lock for
    /// the OS call; returns the bytes made resident (0: nothing cold and free, or the OS refused).
    size_t populate_step(size_t max_bytes) {
        if (base_ == nullptr) return 0;
        const size_t want = std::min(std::max<size_t>(max_bytes / granule, 0), populate_step_bytes / granule);
        if (want == 0) return 0;
        size_t first = n_, count = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (size_t i = 0; i < n_ && count == 0; ++i) {
                if (state_[i] != kFreeCold) continue;
                first = i;
                while (i < n_ && state_[i] == kFreeCold && count < want) { state_[i] = kFilling; ++count; ++i; }
            }
        }
        if (count == 0) return 0;
        const bool ok = os_.populate(base_ + first * granule, count * granule);
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = first; i < first + count; ++i) state_[i] = ok ? kFreeWarm : kFreeCold;
        if (ok) resident_ += count; else populate_failed_ = true;
        return ok ? count * granule : 0;
    }

    size_t shrink(size_t want) override {
        if (base_ == nullptr || want == 0) return 0;
        std::lock_guard<std::mutex> lk(mu_);
        size_t given = 0;
        for (size_t i = n_; i-- > 0 && given < want;) {   // from the top: the low granules stay warm for the next best fit
            if (state_[i] != kFreeWarm) continue;
            size_t lo = i;
            while (lo > 0 && state_[lo - 1] == kFreeWarm && (i + 1 - lo) * granule < want - given) --lo;
            os_.release(base_ + lo * granule, (i + 1 - lo) * granule);
            for (size_t k = lo; k <= i; ++k) state_[k] = kFreeCold;
            resident_ -= i + 1 - lo;
            given += (i + 1 - lo) * granule;
            i = lo;
        }
        return given;
    }

    // ---- figures
    size_t resident_bytes() const { std::lock_guard<std::mutex> lk(mu_); return resident_ * granule; }
    size_t live_requested() const { std::lock_guard<std::mutex> lk(mu_); return live_requested_; }
    size_t idle_bytes() const override {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t r = resident_ * granule;
        return r > live_requested_ ? r - live_requested_ : 0;
    }
    size_t cold_free_bytes() const {
        std::lock_guard<std::mutex> lk(mu_);
        size_t c = 0;
        for (uint8_t s : state_) c += s == kFreeCold;
        return c * granule;
    }
    uint64_t hits() const { std::lock_guard<std::mutex> lk(mu_); return hits_; }
    uint64_t misses() const { std::lock_guard<std::mutex> lk(mu_); return misses_; }
    bool populate_failed() const { std::lock_guard<std::mutex> lk(mu_); return populate_failed_; }

private:
    enum : uint8_t { kFreeCold = 0, kFreeWarm = 1, kUsed = 2, kFilling = 3 };
    static bool is_free(uint8_t s) { return s == kFreeCold || s == kFreeWarm; }
    static size_t granules_for(size_t bytes) { return (bytes + granule - 1) / granule; }
    [[noreturn]] static void fatal(const char* what) {
        std::fprintf(stderr, "strata: conversation segment pool: %s\n", what);
        std::abort();
    }
    SegmentPoolOs& os_;
    uint8_t* base_ = nullptr;
    size_t n_ = 0;
    mutable std::mutex mu_;
    std::vector<uint8_t> state_;
    size_t resident_ = 0;        // granules: free warm + used (a used one counts as resident)
    size_t live_requested_ = 0;  // bytes, the sum of the sizes asked for by live segments
    uint64_t hits_ = 0, misses_ = 0;
    bool populate_failed_ = false;
};

/// The background top-up: a thread that makes the pool's cold granules resident in steps while `permit` allows it.
/// `permit(step)` returns how many bytes it may populate now (0: none: the cache's budget or the RAM floor says no),
/// `on_start` runs once on the thread (placement off the host's core).  kick() asks for a pass; stops on destruction.
class SegmentPoolFiller {
public:
    SegmentPoolFiller(SegmentPool& pool, std::function<size_t(size_t)> permit, std::function<void()> on_start = {})
        : pool_(pool), permit_(std::move(permit)), on_start_(std::move(on_start)), th_([this] { loop(); }) {}
    ~SegmentPoolFiller() {
        { std::lock_guard<std::mutex> lk(mu_); quit_ = true; }
        cv_.notify_all();
        th_.join();
    }
    SegmentPoolFiller(const SegmentPoolFiller&) = delete;
    SegmentPoolFiller& operator=(const SegmentPoolFiller&) = delete;
    void kick() { { std::lock_guard<std::mutex> lk(mu_); want_ = true; } cv_.notify_all(); }
    /// true while a pass is running or asked for (a test waits on it)
    bool busy() const { std::lock_guard<std::mutex> lk(mu_); return want_ || running_; }
private:
    void loop() {
        if (on_start_) on_start_();
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            cv_.wait(lk, [&] { return quit_ || want_; });
            if (quit_) return;
            want_ = false;
            running_ = true;
            lk.unlock();
            for (;;) {
                const size_t allowed = permit_(SegmentPool::populate_step_bytes);
                if (allowed < SegmentPool::granule) break;
                if (pool_.populate_step(allowed) == 0) break;
                { std::lock_guard<std::mutex> g(mu_); if (quit_) break; }
            }
            lk.lock();
            running_ = false;
        }
    }
    SegmentPool& pool_;
    std::function<size_t(size_t)> permit_;
    std::function<void()> on_start_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool want_ = false, quit_ = false, running_ = false;
    std::thread th_;   // last: starts once the members above exist
};

namespace segment_pool_detail {
inline std::atomic<SegmentPool*>& global() { static std::atomic<SegmentPool*> p{nullptr}; return p; }
}

/// The process-wide pool the ConversationBuffer segments ask first (nullptr: none, every segment is a plain allocation)
inline SegmentPool* segment_pool() { return segment_pool_detail::global().load(std::memory_order_acquire); }
/// Installs the process-wide pool once (it must outlive every segment: never destroy it); false when one is installed
inline bool segment_pool_install(SegmentPool* p) {
    SegmentPool* none = nullptr;
    return segment_pool_detail::global().compare_exchange_strong(none, p, std::memory_order_acq_rel);
}
/// Tests only: forget the pool, when no segment from it is alive
inline void segment_pool_uninstall_for_test() { segment_pool_detail::global().store(nullptr, std::memory_order_release); }

}  // namespace strata::core
