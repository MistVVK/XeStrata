// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Harin Mani Karri
// Copyright (c) 2026 Strata Contributors
//
// engine/src/inkling2/ink_cold_store.hpp - 3-Tier NVMe Streaming ColdStore for `strata-inkling`
// Unlocks first-class `GPU + SSD` Only (bounded 1-2 GiB pinned DMA ring, 0 OS swap) and
// `CPU + SSD` Only (bounded RAM LRU + O_DIRECT/FILE_FLAG_NO_BUFFERING + proj_ready[3] overlap).

#pragma once

#include "strata/core/moe_wave_plan.hpp"
#include "strata/core/ssd_stream_scheduler.hpp"
#include "strata/platform/direct_file.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ink2 {

struct InkColdExpertDesc {
    uint64_t file_off   = 0;   ///< absolute file offset in .strata (sections[4].file_offset + body_offset)
    uint32_t body_bytes = 0;
    uint32_t dn_off     = 0;
    uint8_t  gu_code    = 0;
    uint8_t  dn_code    = 0;
};

/// Bounded LRU + 2-Priority Late-Binding Chunked O_DIRECT / FILE_FLAG_NO_BUFFERING ColdStore
/// for `strata-inkling` CPU, CUDA, and HIP backends.
class InkColdStore {
public:
    using AllocFn = std::function<uint8_t*(size_t bytes)>;
    using FreeFn  = std::function<void(uint8_t* ptr)>;

    InkColdStore() = default;
    ~InkColdStore() { close(); }

    InkColdStore(const InkColdStore&) = delete;
    InkColdStore& operator=(const InkColdStore&) = delete;

    /// Open `strata_path` for unbuffered NVMe streaming into a bounded LRU cache of `budget_bytes`.
    /// Pass custom `alloc_fn`/`free_fn` (e.g., `cudaHostAlloc` / `cudaFreeHost`) for pinned GPU DMA rings.
    bool open(const std::string& strata_path,
              std::vector<InkColdExpertDesc> descs,
              size_t max_expert_bytes,
              size_t budget_bytes,
              int n_readers,
              std::string& err,
              AllocFn alloc_fn = nullptr,
              FreeFn free_fn = nullptr) {
        close();
        if (max_expert_bytes == 0) {
            err = "InkColdStore: max_expert_bytes is 0";
            return false;
        }
        descs_ = std::move(descs);
        alloc_fn_ = std::move(alloc_fn);
        free_fn_  = std::move(free_fn);

#ifdef _WIN32
        DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_NO_BUFFERING;
        h_ = CreateFileA(strata_path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                         OPEN_EXISTING, flags, nullptr);
        if (h_ != INVALID_HANDLE_VALUE) {
            direct_ = true;
            FILE_STORAGE_INFO fsi{};
            if (GetFileInformationByHandleEx(h_, FileStorageInfo, &fsi, sizeof(fsi))) {
                const size_t phys = fsi.PhysicalBytesPerSectorForPerformance
                    ? static_cast<size_t>(fsi.PhysicalBytesPerSectorForPerformance)
                    : static_cast<size_t>(fsi.LogicalBytesPerSector);
                sector_align_ = strata::platform::round_up_sector_alignment(phys);
            }
        } else {
            direct_ = false;
            flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS;
            h_ = CreateFileA(strata_path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                             OPEN_EXISTING, flags, nullptr);
            if (h_ == INVALID_HANDLE_VALUE) {
                err = "InkColdStore: cannot open " + strata_path;
                return false;
            }
        }
#else
#if defined(O_DIRECT)
        fd_ = ::open(strata_path.c_str(), O_RDONLY | O_DIRECT);
#else
        fd_ = -1;
#endif
        if (fd_ >= 0) {
            direct_ = true;
            sector_align_ = 4096;
        } else {
            direct_ = false;
            fd_ = ::open(strata_path.c_str(), O_RDONLY);
            if (fd_ < 0) {
                err = "InkColdStore: cannot open " + strata_path;
                return false;
            }
#if defined(__APPLE__) && defined(F_NOCACHE)
            (void)::fcntl(fd_, F_NOCACHE, 1);
            sector_align_ = 4096;
#endif
        }
#endif
        // Each slot stores one expert aligned up to `sector_align_` plus one trailing sector for unaligned file offsets!
        const size_t align = std::max<size_t>(4096u, sector_align_);
        slot_stride_ = ((max_expert_bytes + align * 2u - 1u) / align) * align;
        // Minimum 16 slots so Top-8 + prefetch + double-buffering never thrash.
        const size_t min_slots = 16u;
        const size_t want_slots = std::max(min_slots, budget_bytes / slot_stride_);
        n_slots_ = static_cast<int>(std::min<size_t>(want_slots, 4096u));
        pool_bytes_ = static_cast<size_t>(n_slots_) * slot_stride_;

        if (alloc_fn_) {
            pool_ = alloc_fn_(pool_bytes_);
            custom_alloc_ = true;
        } else {
#ifdef _WIN32
            pool_ = static_cast<uint8_t*>(VirtualAlloc(nullptr, pool_bytes_, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
            void* p = nullptr;
            if (posix_memalign(&p, align, pool_bytes_) == 0) pool_ = static_cast<uint8_t*>(p);
#endif
            custom_alloc_ = false;
        }
        if (!pool_) {
            close();
            err = "InkColdStore: failed to allocate " + std::to_string(pool_bytes_ >> 20) + " MiB cold ring";
            return false;
        }
        slots_.assign(static_cast<size_t>(n_slots_), Slot{});

        const int nr = std::max(1, std::min(8, n_readers > 0 ? n_readers : 4));
        sched_.start(
            nr, n_slots_,
            [this](int /*file_idx*/, uint64_t off, uint8_t* dst, size_t len) -> bool {
                return read_raw(off, dst, len);
            },
            [this](const strata::core::StreamExpertReq& req, int& out_slot, uint8_t*& out_dst) -> bool {
                return bind_prefetch_slot(req.key, out_slot, out_dst);
            },
            [this](const strata::core::StreamExpertReq& req, bool ok, bool /*was_pf*/) {
                if (req.slot >= 0 && req.slot < n_slots_) {
                    std::lock_guard<std::mutex> lk(mu_);
                    auto& sl = slots_[static_cast<size_t>(req.slot)];
                    if (sl.io_active > 0) sl.io_active--;
                    if (!ok && sl.key == req.key) {
                        where_.erase(sl.key);
                        sl.key = -1;
                    }
                }
            });
        open_ = true;
        return true;
    }

    void close() {
        idle_.stop();
        sched_.stop();
        {
            std::lock_guard<std::mutex> lk(mu_);
            where_.clear();
            slots_.clear();
            n_slots_ = 0;
        }
        if (pool_) {
            if (custom_alloc_ && free_fn_) {
                free_fn_(pool_);
            } else {
#ifdef _WIN32
                VirtualFree(pool_, 0, MEM_RELEASE);
#else
                std::free(pool_);
#endif
            }
            pool_ = nullptr;
            pool_bytes_ = 0;
        }
#ifdef _WIN32
        if (h_ != INVALID_HANDLE_VALUE) { CloseHandle(h_); h_ = INVALID_HANDLE_VALUE; }
#else
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
        open_ = false;
    }

    bool is_open() const noexcept { return open_; }
    bool is_direct() const noexcept { return direct_; }
    int num_slots() const noexcept { return n_slots_; }
    size_t pool_bytes() const noexcept { return pool_bytes_; }
    size_t sector_align() const noexcept { return sector_align_; }

    /// Return current pinned pointer if `key` is already resident in the cold ring (and bump LRU tick), else nullptr.
    /// Caller MUST call `unpin_slot(slot)` when finished using the returned pointer.
    const uint8_t* try_pin_resident(int key, int& out_slot) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = where_.find(key);
        if (it == where_.end()) return nullptr;
        const int s = it->second;
        auto& sl = slots_[static_cast<size_t>(s)];
        sl.pin++;
        sl.tick = ++tick_;
        out_slot = s;
        hits_.fetch_add(1, std::memory_order_relaxed);
        return pool_ + static_cast<size_t>(s) * slot_stride_ + sl.head_pad;
    }

    /// Acquire/start a demand read for `key` (returns bound slot + pointer immediately; caller waits on
    /// `slot_state(slot)->wait_gate_up()` / `wait_down()` or `wait_all()`!).
    /// If already resident, marks all projections ready immediately with zero I/O.
    /// If already queued/in-flight as prefetch, promotes it to demand without duplicate I/O!
    const uint8_t* demand_begin(int key, int layer, uint64_t gen, int& out_slot) {
        idle_.notify_active();
        if (key < 0 || static_cast<size_t>(key) >= descs_.size()) {
            out_slot = -1;
            return nullptr;
        }
        const auto& d = descs_[static_cast<size_t>(key)];
        const size_t align = std::max<size_t>(4096u, sector_align_);
        const uint64_t aligned_off = d.file_off & ~(static_cast<uint64_t>(align) - 1u);
        const size_t head_pad = static_cast<size_t>(d.file_off - aligned_off);
        const size_t read_len = ((head_pad + d.body_bytes + align - 1u) / align) * align;

        int slot = -1;
        bool need_io = false;
        for (int attempt = 0; attempt < 200; ++attempt) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                auto it = where_.find(key);
                if (it != where_.end()) {
                    slot = it->second;
                    auto& sl = slots_[static_cast<size_t>(slot)];
                    sl.pin++;
                    sl.tick = ++tick_;
                    // If it was an in-flight prefetch, promote priority!
                    sched_.promote_to_demand(key, slot, pool_ + static_cast<size_t>(slot) * slot_stride_);
                    hits_.fetch_add(1, std::memory_order_relaxed);
                    out_slot = slot;
                    return pool_ + static_cast<size_t>(slot) * slot_stride_ + sl.head_pad;
                }
                slot = evict_lru_locked();
                if (slot >= 0) {
                    auto& sl = slots_[static_cast<size_t>(slot)];
                    if (sl.key >= 0) where_.erase(sl.key);
                    sl.key = key;
                    sl.tick = ++tick_;
                    sl.pin = 1;
                    sl.io_active++;
                    sl.head_pad = head_pad;
                    if (auto* st = sched_.slot_state(slot)) st->reset_for_load(key);
                    where_[key] = slot;
                    need_io = true;
                    misses_.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
            }
            // All unpinned slots currently have active I/O: wait briefly for an in-flight read to finish!
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        if (slot < 0) {
            out_slot = -1;
            return nullptr;
        }
        out_slot = slot;
        if (need_io) {
            uint8_t* base = pool_ + static_cast<size_t>(slot) * slot_stride_;
            const size_t dn_split = (d.dn_off > 0 && d.dn_off < d.body_bytes) ? (head_pad + d.dn_off) : 0;
            auto req = strata::core::StreamExpertReq::make_contiguous_split(
                key, layer, gen, slot, base, 0, aligned_off, read_len, dn_split, align);
            sched_.enqueue_demand(std::move(req));
        }
        return pool_ + static_cast<size_t>(slot) * slot_stride_ + head_pad;
    }

    /// Synchronous convenience: demand-load `key`, wait for all projections, return pinned pointer.
    /// Caller MUST call `unpin_slot(out_slot)` after using the bytes!
    const uint8_t* demand_wait(int key, int layer, uint64_t gen, int& out_slot) {
        const uint8_t* p = demand_begin(key, layer, gen, out_slot);
        if (!p || out_slot < 0) return nullptr;
        if (auto* st = sched_.slot_state(out_slot)) {
            if (!st->wait_all()) {
                unpin_slot(out_slot);
                out_slot = -1;
                return nullptr;
            }
        }
        return p;
    }

    /// Wait only for gate+up projections so CPU compute can start while `down` finishes NVMe read!
    bool wait_gate_up(int slot) const noexcept {
        if (const auto* st = sched_.slot_state(slot)) return st->wait_gate_up();
        return false;
    }

    /// Wait for `down` projection after gate+up compute completes!
    bool wait_down(int slot) const noexcept {
        if (const auto* st = sched_.slot_state(slot)) return st->wait_down();
        return false;
    }

    void unpin_slot(int slot) {
        if (slot < 0 || slot >= n_slots_) return;
        std::lock_guard<std::mutex> lk(mu_);
        auto& sl = slots_[static_cast<size_t>(slot)];
        if (sl.pin > 0) sl.pin--;
    }

    /// Enqueue a late-binding 512 KiB chunked speculative prefetch for future layer `layer`.
    void prefetch_hint(int key, int layer, uint64_t gen) {
        if (!open_ || key < 0 || static_cast<size_t>(key) >= descs_.size()) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (where_.find(key) != where_.end()) return;
        }
        const auto& d = descs_[static_cast<size_t>(key)];
        const size_t align = std::max<size_t>(4096u, sector_align_);
        const uint64_t aligned_off = d.file_off & ~(static_cast<uint64_t>(align) - 1u);
        const size_t head_pad = static_cast<size_t>(d.file_off - aligned_off);
        const size_t read_len = ((head_pad + d.body_bytes + align - 1u) / align) * align;
        const size_t dn_split = (d.dn_off > 0 && d.dn_off < d.body_bytes) ? (head_pad + d.dn_off) : 0;
        auto req = strata::core::StreamExpertReq::make_contiguous_split(
            key, layer, gen, -1, nullptr, 0, aligned_off, read_len, dn_split, align);
        sched_.enqueue_prefetch(std::move(req));
    }

    /// Insert an expert copied back from VRAM (full-duplex D2H copyback) directly into an unpinned slot!
    bool insert_from_copyback(int key, const uint8_t* src, size_t bytes) {
        if (!open_ || !src || bytes == 0 || key < 0 || static_cast<size_t>(key) >= descs_.size()) return false;
        std::lock_guard<std::mutex> lk(mu_);
        if (where_.find(key) != where_.end()) return true;
        const int slot = evict_lru_locked();
        if (slot < 0) return false;
        auto& sl = slots_[static_cast<size_t>(slot)];
        if (sl.key >= 0) where_.erase(sl.key);
        sl.key = key;
        sl.tick = ++tick_;
        sl.pin = 0;
        sl.head_pad = 0;
        where_[key] = slot;
        std::memcpy(pool_ + static_cast<size_t>(slot) * slot_stride_, src, std::min(bytes, slot_stride_));
        if (auto* st = sched_.slot_state(slot)) st->mark_all_ready(key);
        return true;
    }

    void drop_stale(int layer, uint64_t gen) {
        if (open_) sched_.drop_stale(layer, gen);
    }

    /// Start background 500 ms quiet-time pre-population in priority order (`preload_keys`).
    void start_idle_preloader(std::vector<int> preload_keys, int idle_ms = 500) {
        if (!open_ || preload_keys.empty()) return;
        idle_keys_ = std::move(preload_keys);
        idle_cursor_.store(0, std::memory_order_relaxed);
        idle_.start([this](const std::atomic<bool>& should_yield) -> bool {
            if (should_yield.load(std::memory_order_acquire)) return false;
            const size_t idx = idle_cursor_.fetch_add(1, std::memory_order_relaxed);
            if (idx >= idle_keys_.size()) return false;
            // Only preload while free slots remain so we never evict active decode experts
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (static_cast<int>(where_.size()) >= std::max(1, n_slots_ - 4)) return false;
            }
            const int key = idle_keys_[idx];
            prefetch_hint(key, 0, UINT64_MAX);  // high gen so not dropped by drop_stale
            return true;
        }, idle_ms);
    }

    void notify_active() noexcept { idle_.notify_active(); }
    void notify_idle()   noexcept { idle_.notify_idle(); }

    uint64_t hits()   const noexcept { return hits_.load(std::memory_order_relaxed); }
    uint64_t misses() const noexcept { return misses_.load(std::memory_order_relaxed); }
    strata::core::SsdSchedulerStats scheduler_stats() const { return sched_.stats(); }

private:
    struct Slot {
        int      key       = -1;
        uint64_t tick      = 0;
        int      pin       = 0;
        int      io_active = 0;
        size_t   head_pad  = 0;
    };

    int evict_lru_locked() {
        int best = -1;
        uint64_t best_tick = ~0ull;
        for (int s = 0; s < n_slots_; ++s) {
            if (slots_[static_cast<size_t>(s)].pin > 0 || slots_[static_cast<size_t>(s)].io_active > 0) continue;
            if (slots_[static_cast<size_t>(s)].key < 0) return s;  // free slot immediately!
            if (slots_[static_cast<size_t>(s)].tick < best_tick) {
                best_tick = slots_[static_cast<size_t>(s)].tick;
                best = s;
            }
        }
        return best;
    }

    bool bind_prefetch_slot(int64_t key64, int& out_slot, uint8_t*& out_dst) {
        const int key = static_cast<int>(key64);
        if (key < 0 || static_cast<size_t>(key) >= descs_.size()) return false;
        const auto& d = descs_[static_cast<size_t>(key)];
        const size_t align = std::max<size_t>(4096u, sector_align_);
        const uint64_t aligned_off = d.file_off & ~(static_cast<uint64_t>(align) - 1u);
        const size_t head_pad = static_cast<size_t>(d.file_off - aligned_off);

        std::lock_guard<std::mutex> lk(mu_);
        if (where_.find(key) != where_.end()) return false;  // already present
        // Cap concurrent speculative in-flight slots to <= n_slots_/4 so demand always has free headroom!
        int io_busy = 0;
        for (int s = 0; s < n_slots_; ++s) {
            if (slots_[static_cast<size_t>(s)].io_active > 0) ++io_busy;
        }
        if (io_busy >= std::max(4, n_slots_ / 4)) return false;
        const int slot = evict_lru_locked();
        if (slot < 0) return false;
        auto& sl = slots_[static_cast<size_t>(slot)];
        if (sl.key >= 0) where_.erase(sl.key);
        sl.key = key;
        sl.tick = ++tick_;
        sl.pin = 0;  // speculative prefetches stay unpinned; io_active prevents eviction while reading!
        sl.io_active++;
        sl.head_pad = head_pad;
        if (auto* st = sched_.slot_state(slot)) st->reset_for_load(key);
        where_[key] = slot;
        out_slot = slot;
        out_dst  = pool_ + static_cast<size_t>(slot) * slot_stride_;
        return true;
    }

    bool read_raw(uint64_t off, uint8_t* dst, size_t len) const {
        size_t got = 0;
#ifdef _WIN32
        HANDLE ev = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        bool ok = true;
        while (got < len) {
            OVERLAPPED ov{};
            const uint64_t cur = off + got;
            ov.Offset = static_cast<DWORD>(cur & 0xFFFFFFFFu);
            ov.OffsetHigh = static_cast<DWORD>(cur >> 32);
            ov.hEvent = ev;
            if (ev) ResetEvent(ev);
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(len - got, 1u << 30));
            DWORD n = 0;
            if (!ReadFile(h_, dst + got, chunk, &n, &ov)) {
                const DWORD le = GetLastError();
                if (le == ERROR_IO_PENDING) {
                    if (!GetOverlappedResult(h_, &ov, &n, TRUE)) {
                        if (GetLastError() == ERROR_HANDLE_EOF) break;
                        ok = false;
                        break;
                    }
                } else if (le == ERROR_HANDLE_EOF) {
                    break;
                } else {
                    ok = false;
                    break;
                }
            }
            if (n == 0) break;
            got += static_cast<size_t>(n);
        }
        if (ev) CloseHandle(ev);
        if (ok && got < len) std::memset(dst + got, 0, len - got);
        return ok;
#else
        while (got < len) {
            const ssize_t n = ::pread(fd_, dst + got, len - got, static_cast<off_t>(off + got));
            if (n < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (n == 0) break;
            got += static_cast<size_t>(n);
        }
        if (got < len) std::memset(dst + got, 0, len - got);
        return true;
#endif
    }

    bool open_ = false;
    bool direct_ = false;
    size_t sector_align_ = 4096;
    size_t slot_stride_ = 0;
    size_t pool_bytes_ = 0;
    int n_slots_ = 0;
    uint8_t* pool_ = nullptr;
    bool custom_alloc_ = false;
    AllocFn alloc_fn_;
    FreeFn  free_fn_;

#ifdef _WIN32
    HANDLE h_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif

    std::vector<InkColdExpertDesc> descs_;
    std::mutex mu_;
    std::vector<Slot> slots_;
    std::unordered_map<int, int> where_;
    uint64_t tick_ = 0;
    strata::core::TwoPriorityExpertScheduler sched_;
    strata::core::IdlePreloader idle_;
    std::vector<int> idle_keys_;
    std::atomic<size_t> idle_cursor_{0};
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
};

}  // namespace ink2
