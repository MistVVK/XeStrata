// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Harin Mani Karri
// Copyright (c) 2026 Strata Contributors
//
// include/strata/core/ssd_stream_scheduler.hpp - 3-Tier NVMe Streaming Primitives:
//   1. TwoPriorityExpertScheduler:
//      - 2-priority queue (q_demand whole-expert preempts 512 KiB chunked q_prefetch)
//      - Late slot binding (speculative prefetches do not pin/evict slots until dequeued)
//      - drop_stale(current_layer, gen) and promote_to_demand(key) without duplicate disk I/O
//      - Per-projection atomic readiness proj_ready[3] (gate=0, up=1, down=2), ALWAYS reset to 0 on
//        enqueue/bind BEFORE pushing so compute threads overlap gate+up with down NVMe I/O safely
//   2. VramCopybackQueue:
//      - Bounded full-duplex reverse-PCIe D2H copyback of SSD-backed experts evicted from VRAM into RAM
//   3. IdlePreloader:
//      - 500 ms quiet-time background NVMe pre-population yielding in < 100 us on request arrival

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace strata::core {

/// 512 KiB chunk size for speculative prefetch reads so a demand read arriving mid-prefetch
/// preempts within ~50-100 us on PCIe Gen4/Gen5 NVMe.
inline constexpr size_t kPrefetchChunkBytes = 512u * 1024u;

/// Per-projection readiness states (gate=0, up=1, down=2).
inline constexpr uint8_t kProjPending = 0;
inline constexpr uint8_t kProjReady   = 1;
inline constexpr uint8_t kProjError   = 2;

struct ExpertProjSpan {
    int      file_idx = 0;
    uint64_t file_off = 0;
    size_t   dst_off  = 0;
    size_t   bytes    = 0;
};

struct StreamExpertReq {
    int64_t key        = -1;   ///< unique expert key, e.g. layer * n_expert + expert
    int64_t layer      = 0;
    uint64_t gen       = 0;    ///< monotonic token/window generation for drop_stale
    int     slot       = -1;   ///< >=0 if pre-bound (demand), -1 for late-binding prefetch
    uint8_t* dst       = nullptr;
    size_t  total_bytes = 0;
    /// Up to 3 projection spans: [0]=gate, [1]=up, [2]=down.
    /// For contiguous blobs where gate+up precede down (e.g., hy4/inkling2), span[0] = [0, dn_off),
    /// span[1] = empty (marked ready with span[0]), span[2] = [dn_off, total_bytes).
    std::array<ExpertProjSpan, 3> spans{};
    int n_spans = 0;

    /// Build a 2-phase (gate+up, then down) split for a contiguous expert body on disk.
    static StreamExpertReq make_contiguous_split(
        int64_t key_, int64_t layer_, uint64_t gen_, int slot_, uint8_t* dst_,
        int file_idx, uint64_t file_off, size_t total_bytes_, size_t dn_off, size_t align = 4096) {
        StreamExpertReq r{};
        r.key = key_;
        r.layer = layer_;
        r.gen = gen_;
        r.slot = slot_;
        r.dst = dst_;
        r.total_bytes = total_bytes_;
        // Only split when both halves are meaningful and alignment permits zero overlap corruption.
        const size_t split = (dn_off > 0 && dn_off < total_bytes_)
            ? ((dn_off + align - 1u) & ~(align - 1u))
            : total_bytes_;
        if (split >= align && split < total_bytes_) {
            r.spans[0] = {file_idx, file_off, 0, split};
            r.spans[1] = {file_idx, file_off, 0, 0};  // ready immediately when span[0] completes
            r.spans[2] = {file_idx, file_off + split, split, total_bytes_ - split};
            r.n_spans = 3;
        } else {
            r.spans[0] = {file_idx, file_off, 0, total_bytes_};
            r.n_spans = 1;
        }
        return r;
    }

    /// Build a 3-projection request (gate=0, up=1, down=2), e.g. GGUF shards in place.
    static StreamExpertReq make_three_proj(
        int64_t key_, int64_t layer_, uint64_t gen_, int slot_, uint8_t* dst_,
        const std::array<ExpertProjSpan, 3>& spans_, size_t total_bytes_) {
        StreamExpertReq r{};
        r.key = key_;
        r.layer = layer_;
        r.gen = gen_;
        r.slot = slot_;
        r.dst = dst_;
        r.total_bytes = total_bytes_;
        r.spans = spans_;
        r.n_spans = 3;
        return r;
    }
};

/// Per-slot atomic projection readiness flags (cache-line isolated to prevent false sharing).
struct alignas(64) SlotProjState {
    std::atomic<uint8_t> proj_ready[3]{{kProjReady}, {kProjReady}, {kProjReady}};
    std::atomic<bool>    cancelled{false};
    std::atomic<int64_t> active_key{-1};

    void reset_for_load(int64_t key) noexcept {
        cancelled.store(false, std::memory_order_relaxed);
        active_key.store(key, std::memory_order_relaxed);
        proj_ready[0].store(kProjPending, std::memory_order_relaxed);
        proj_ready[1].store(kProjPending, std::memory_order_relaxed);
        proj_ready[2].store(kProjPending, std::memory_order_release);
    }

    void mark_all_ready(int64_t key) noexcept {
        cancelled.store(false, std::memory_order_relaxed);
        active_key.store(key, std::memory_order_relaxed);
        proj_ready[0].store(kProjReady, std::memory_order_relaxed);
        proj_ready[1].store(kProjReady, std::memory_order_relaxed);
        proj_ready[2].store(kProjReady, std::memory_order_release);
    }

    bool wait_proj(int p) const noexcept {
        uint32_t spins = 0;
        for (;;) {
            const uint8_t st = proj_ready[p].load(std::memory_order_acquire);
            if (st != kProjPending) return st == kProjReady;
            if (++spins < 256u) {
                // brief spin before yielding
            } else if (spins < 4096u) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(15));
            }
        }
    }

    bool wait_gate_up() const noexcept {
        return wait_proj(0) && wait_proj(1);
    }

    bool wait_down() const noexcept {
        return wait_proj(2);
    }

    bool wait_all() const noexcept {
        const bool ok0 = wait_proj(0);
        const bool ok1 = wait_proj(1);
        const bool ok2 = wait_proj(2);
        return ok0 && ok1 && ok2;
    }
};

struct SsdSchedulerStats {
    uint64_t demand_submitted    = 0;
    uint64_t demand_completed    = 0;
    uint64_t prefetch_submitted  = 0;
    uint64_t prefetch_completed  = 0;
    uint64_t prefetch_promoted   = 0;
    uint64_t prefetch_dropped    = 0;
    uint64_t prefetch_preempted  = 0;
    uint64_t prefetch_chunks     = 0;
    uint64_t bytes_read          = 0;
    uint64_t io_errors           = 0;
};

/// 2-Priority Late-Binding Chunked I/O Scheduler.
///   - `ReadChunkFn(int file_idx, uint64_t file_off, uint8_t* dst, size_t len) -> bool`
///   - `BindSlotFn(const StreamExpertReq& req, int& out_slot, uint8_t*& out_dst) -> bool` (late binding for prefetch)
///   - `CompleteFn(const StreamExpertReq& req, bool ok, bool was_prefetch)` (optional completion hook)
class TwoPriorityExpertScheduler {
public:
    using ReadChunkFn = std::function<bool(int file_idx, uint64_t file_off, uint8_t* dst, size_t len)>;
    using BindSlotFn  = std::function<bool(const StreamExpertReq& req, int& out_slot, uint8_t*& out_dst)>;
    using CompleteFn  = std::function<void(const StreamExpertReq& req, bool ok, bool was_prefetch)>;

    TwoPriorityExpertScheduler() = default;
    ~TwoPriorityExpertScheduler() { stop(); }

    TwoPriorityExpertScheduler(const TwoPriorityExpertScheduler&) = delete;
    TwoPriorityExpertScheduler& operator=(const TwoPriorityExpertScheduler&) = delete;

    void start(int n_threads, int n_slots, ReadChunkFn read_fn,
               BindSlotFn bind_fn = nullptr, CompleteFn complete_fn = nullptr,
               size_t prefetch_chunk_bytes = kPrefetchChunkBytes) {
        stop();
        read_fn_ = std::move(read_fn);
        bind_fn_ = std::move(bind_fn);
        complete_fn_ = std::move(complete_fn);
        chunk_bytes_ = std::max<size_t>(4096u, prefetch_chunk_bytes);
        slots_.reset(n_slots > 0 ? new SlotProjState[static_cast<size_t>(n_slots)] : nullptr);
        n_slots_ = std::max(0, n_slots);
        {
            std::unique_lock<std::mutex> lk(mu_);
            stopping_ = false;
            q_demand_.clear();
            q_prefetch_.clear();
            inflight_.clear();
        }
        const int nt = std::max(1, n_threads);
        workers_.reserve(static_cast<size_t>(nt));
        for (int i = 0; i < nt; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    void stop() {
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (workers_.empty()) return;
            stopping_ = true;
            for (auto& kv : inflight_) {
                if (kv.second.slot >= 0 && kv.second.slot < n_slots_) {
                    slots_[kv.second.slot].cancelled.store(true, std::memory_order_release);
                }
            }
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
        {
            std::unique_lock<std::mutex> lk(mu_);
            q_demand_.clear();
            q_prefetch_.clear();
            inflight_.clear();
            demand_waiting_.store(0, std::memory_order_relaxed);
            pending_demand_.store(0, std::memory_order_relaxed);
            pending_prefetch_.store(0, std::memory_order_relaxed);
        }
    }

    SlotProjState* slot_state(int slot) noexcept {
        return (slots_ && slot >= 0 && slot < n_slots_) ? &slots_[slot] : nullptr;
    }
    const SlotProjState* slot_state(int slot) const noexcept {
        return (slots_ && slot >= 0 && slot < n_slots_) ? &slots_[slot] : nullptr;
    }

    /// Submit a high-priority whole-expert demand read.
    /// Resets `proj_ready[0..2] = 0` BEFORE enqueueing so compute never sees stale readiness.
    /// If `key` was queued in `q_prefetch`, promotes it to demand without duplicating I/O.
    /// Returns true if newly enqueued or promoted; false if already in-flight as demand/promoted.
    bool enqueue_demand(StreamExpertReq req) {
        if (req.slot >= 0 && req.slot < n_slots_) {
            slots_[req.slot].reset_for_load(req.key);
        }
        bool notify = false;
        {
            std::unique_lock<std::mutex> lk(mu_);
            // 1. Check if already in-flight (e.g., as a prefetch): promote priority so it stops yielding!
            auto it_inf = inflight_.find(req.key);
            if (it_inf != inflight_.end()) {
                if (it_inf->second.is_prefetch) {
                    it_inf->second.promoted_to_demand = true;
                    if (it_inf->second.slot == req.slot && req.slot >= 0 && req.slot < n_slots_) {
                        // Already reading into the exact same slot! Do not duplicate I/O.
                        stats_.prefetch_promoted++;
                        return false;
                    }
                    // Different slot requested: cancel the prefetch so demand reads cleanly into req.slot.
                    if (it_inf->second.slot >= 0 && it_inf->second.slot < n_slots_) {
                        slots_[it_inf->second.slot].cancelled.store(true, std::memory_order_release);
                    }
                }
            }
            // 2. Remove any unstarted duplicate from q_prefetch (promotion!)
            std::vector<StreamExpertReq> dropped_bound;
            for (auto it = q_prefetch_.begin(); it != q_prefetch_.end(); ) {
                if (it->req.key == req.key) {
                    if (it->req.slot >= 0 && it->req.slot != req.slot) {
                        dropped_bound.push_back(std::move(it->req));
                    }
                    it = q_prefetch_.erase(it);
                    pending_prefetch_.fetch_sub(1, std::memory_order_relaxed);
                    stats_.prefetch_promoted++;
                } else {
                    ++it;
                }
            }
            q_demand_.push_back(std::move(req));
            demand_waiting_.fetch_add(1, std::memory_order_release);
            pending_demand_.fetch_add(1, std::memory_order_release);
            stats_.demand_submitted++;
            notify = true;
            lk.unlock();
            for (const auto& dr : dropped_bound) {
                publish_span_ready(dr.slot, 0, 1, false);
                if (complete_fn_) complete_fn_(dr, false, true);
            }
        }
        if (notify) cv_.notify_one();
        return true;
    }

    /// Submit a speculative prefetch read (late-bound if `req.slot < 0`).
    /// Deduplicated against queued and in-flight keys.
    bool enqueue_prefetch(StreamExpertReq req) {
        bool notify = false;
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (inflight_.find(req.key) != inflight_.end()) return false;
            for (const auto& d : q_demand_) {
                if (d.key == req.key) return false;
            }
            for (const auto& p : q_prefetch_) {
                if (p.req.key == req.key) return false;
            }
            if (req.slot >= 0 && req.slot < n_slots_) {
                slots_[req.slot].reset_for_load(req.key);
            }
            PrefetchItem item{};
            item.req = std::move(req);
            q_prefetch_.push_back(std::move(item));
            pending_prefetch_.fetch_add(1, std::memory_order_release);
            stats_.prefetch_submitted++;
            notify = true;
        }
        if (notify) cv_.notify_one();
        return true;
    }

    /// Promote a queued or in-flight prefetch for `key` into `q_demand`.
    /// Returns true if `key` was found in `q_prefetch` or `inflight_`.
    bool promote_to_demand(int64_t key, int bound_slot = -1, uint8_t* bound_dst = nullptr) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it_inf = inflight_.find(key);
        if (it_inf != inflight_.end()) {
            if (it_inf->second.is_prefetch) {
                it_inf->second.promoted_to_demand = true;
                stats_.prefetch_promoted++;
            }
            return true;
        }
        for (auto it = q_prefetch_.begin(); it != q_prefetch_.end(); ++it) {
            if (it->req.key == key) {
                StreamExpertReq req = std::move(it->req);
                q_prefetch_.erase(it);
                pending_prefetch_.fetch_sub(1, std::memory_order_relaxed);
                if (req.slot < 0 && bound_slot >= 0) {
                    req.slot = bound_slot;
                    req.dst  = bound_dst;
                }
                if (req.slot >= 0 && req.slot < n_slots_) {
                    slots_[req.slot].reset_for_load(req.key);
                }
                q_demand_.push_back(std::move(req));
                demand_waiting_.fetch_add(1, std::memory_order_release);
                pending_demand_.fetch_add(1, std::memory_order_release);
                stats_.prefetch_promoted++;
                stats_.demand_submitted++;
                cv_.notify_one();
                return true;
            }
        }
        return false;
    }

    /// Drop unstarted prefetches whose `(gen < current_gen)` or `(gen == current_gen && layer < current_layer)`,
    /// and cancel in-flight stale prefetches so they abort at the next 512 KiB chunk boundary.
    size_t drop_stale(int64_t current_layer, uint64_t current_gen) {
        size_t dropped = 0;
        std::vector<StreamExpertReq> dropped_bound;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto is_stale = [&](int64_t layer, uint64_t gen) {
                return (gen < current_gen) || (gen == current_gen && layer < current_layer);
            };
            for (auto it = q_prefetch_.begin(); it != q_prefetch_.end(); ) {
                if (is_stale(it->req.layer, it->req.gen)) {
                    if (it->req.slot >= 0) dropped_bound.push_back(std::move(it->req));
                    it = q_prefetch_.erase(it);
                    pending_prefetch_.fetch_sub(1, std::memory_order_relaxed);
                    stats_.prefetch_dropped++;
                    ++dropped;
                } else {
                    ++it;
                }
            }
            for (auto& kv : inflight_) {
                if (is_stale(kv.second.layer, kv.second.gen)) {
                    if (kv.second.is_prefetch && !kv.second.promoted_to_demand) {
                        kv.second.cancelled = true;
                        cancel_epoch_.fetch_add(1, std::memory_order_release);
                        if (kv.second.slot >= 0 && kv.second.slot < n_slots_) {
                            slots_[kv.second.slot].cancelled.store(true, std::memory_order_release);
                        }
                    }
                }
            }
        }
        for (const auto& dr : dropped_bound) {
            publish_span_ready(dr.slot, 0, 1, false);
            if (complete_fn_) complete_fn_(dr, false, true);
        }
        return dropped;
    }

    /// Cancel any in-flight or queued prefetch targeting `slot` and wait until `slot` is quiescent.
    void cancel_slot(int slot) {
        if (slot < 0 || slot >= n_slots_) return;
        std::unique_lock<std::mutex> lk(mu_);
        slots_[slot].cancelled.store(true, std::memory_order_release);
        for (auto it = q_prefetch_.begin(); it != q_prefetch_.end(); ) {
            if (it->req.slot == slot) {
                it = q_prefetch_.erase(it);
                pending_prefetch_.fetch_sub(1, std::memory_order_relaxed);
                stats_.prefetch_dropped++;
            } else {
                ++it;
            }
        }
        for (auto& kv : inflight_) {
            if (kv.second.slot == slot) kv.second.cancelled = true;
        }
        done_cv_.wait(lk, [&]() {
            for (const auto& kv : inflight_) {
                if (kv.second.slot == slot) return false;
            }
            return true;
        });
    }

    void wait_all_demand() {
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&]() { return pending_demand_.load(std::memory_order_acquire) == 0; });
    }

    void wait_all() {
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&]() {
            return pending_demand_.load(std::memory_order_acquire) == 0 &&
                   pending_prefetch_.load(std::memory_order_acquire) == 0;
        });
    }

    int pending_demand() const noexcept { return pending_demand_.load(std::memory_order_acquire); }
    int pending_prefetch() const noexcept { return pending_prefetch_.load(std::memory_order_acquire); }

    SsdSchedulerStats stats() const {
        std::lock_guard<std::mutex> lk(mu_);
        return stats_;
    }

private:
    struct PrefetchItem {
        StreamExpertReq req{};
        int    span_idx = 0;
        size_t span_off = 0;
        bool   bound    = false;
    };

    struct InflightMeta {
        int      slot               = -1;
        int64_t  layer              = 0;
        uint64_t gen                = 0;
        bool     is_prefetch        = false;
        bool     promoted_to_demand = false;
        bool     cancelled          = false;
    };

    void publish_span_ready(int slot, int span_idx, int n_spans, bool ok) noexcept {
        if (!slots_ || slot < 0 || slot >= n_slots_) return;
        const uint8_t st = ok ? kProjReady : kProjError;
        if (n_spans <= 1) {
            slots_[slot].proj_ready[0].store(st, std::memory_order_relaxed);
            slots_[slot].proj_ready[1].store(st, std::memory_order_relaxed);
            slots_[slot].proj_ready[2].store(st, std::memory_order_release);
        } else {
            if (span_idx >= 0 && span_idx < 3) {
                slots_[slot].proj_ready[span_idx].store(st, std::memory_order_release);
            }
        }
    }

    void worker_loop() {
        for (;;) {
            bool is_demand = false;
            StreamExpertReq dreq{};
            PrefetchItem pitem{};

            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]() {
                    return stopping_ || !q_demand_.empty() || !q_prefetch_.empty();
                });
                if (stopping_ && q_demand_.empty() && q_prefetch_.empty()) return;

                // STRICT PRIORITY: q_demand ALWAYS preempts q_prefetch!
                if (!q_demand_.empty()) {
                    is_demand = true;
                    dreq = std::move(q_demand_.front());
                    q_demand_.pop_front();
                    demand_waiting_.fetch_sub(1, std::memory_order_relaxed);
                    inflight_[dreq.key] = {dreq.slot, dreq.layer, dreq.gen, false, true, false};
                } else {
                    is_demand = false;
                    pitem = std::move(q_prefetch_.front());
                    q_prefetch_.pop_front();
                    inflight_[pitem.req.key] = {pitem.req.slot, pitem.req.layer, pitem.req.gen, true, false, false};
                }
            }

            if (is_demand) {
                run_demand(dreq);
            } else {
                run_prefetch(pitem);
            }
        }
    }

    void run_demand(const StreamExpertReq& req) {
        bool all_ok = true;
        uint64_t bytes = 0;
        const int ns = std::max(1, req.n_spans);
        for (int s = 0; s < ns; ++s) {
            const auto& sp = req.spans[s];
            bool sp_ok = true;
            if (sp.bytes > 0) {
                sp_ok = read_fn_(sp.file_idx, sp.file_off, req.dst + sp.dst_off, sp.bytes);
                if (sp_ok) bytes += sp.bytes;
                else all_ok = false;
            }
            publish_span_ready(req.slot, s, ns, sp_ok);
        }
        if (complete_fn_) complete_fn_(req, all_ok, false);

        {
            std::unique_lock<std::mutex> lk(mu_);
            inflight_.erase(req.key);
            stats_.demand_completed++;
            stats_.bytes_read += bytes;
            if (!all_ok) stats_.io_errors++;
            pending_demand_.fetch_sub(1, std::memory_order_release);
        }
        done_cv_.notify_all();
    }

    void run_prefetch(PrefetchItem item) {
        // Late slot binding: acquire physical slot ONLY when actually starting I/O!
        if (!item.bound && item.req.slot < 0) {
            if (!bind_fn_) {
                finish_prefetch(item.req, false, 0, true);
                return;
            }
            int bound_slot = -1;
            uint8_t* bound_dst = nullptr;
            if (!bind_fn_(item.req, bound_slot, bound_dst) || bound_slot < 0 || !bound_dst) {
                finish_prefetch(item.req, false, 0, true);
                return;
            }
            item.req.slot = bound_slot;
            item.req.dst  = bound_dst;
            item.bound    = true;
            if (bound_slot < n_slots_ &&
                slots_[bound_slot].active_key.load(std::memory_order_acquire) != item.req.key) {
                slots_[bound_slot].reset_for_load(item.req.key);
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                auto it = inflight_.find(item.req.key);
                if (it != inflight_.end()) it->second.slot = bound_slot;
            }
        } else {
            item.bound = true;
        }

        bool all_ok = true;
        uint64_t bytes = 0;
        uint64_t chunks = 0;
        const int ns = std::max(1, item.req.n_spans);

        for (int s = item.span_idx; s < ns; ++s) {
            const auto& sp = item.req.spans[s];
            size_t off = (s == item.span_idx) ? item.span_off : 0;
            bool sp_ok = true;
            uint64_t seen_cancel = cancel_epoch_.load(std::memory_order_acquire);

            while (off < sp.bytes) {
                // Check cancellation / demand preemption at EVERY 512 KiB chunk boundary!
                bool cancelled = false;
                bool preempt   = false;
                if (item.req.slot >= 0 && item.req.slot < n_slots_ &&
                    slots_[item.req.slot].cancelled.load(std::memory_order_acquire)) {
                    cancelled = true;
                }
                const uint64_t cur_cancel = cancel_epoch_.load(std::memory_order_acquire);
                if (!cancelled && (cur_cancel != seen_cancel || demand_waiting_.load(std::memory_order_acquire) > 0)) {
                    seen_cancel = cur_cancel;
                    std::lock_guard<std::mutex> lk(mu_);
                    auto it = inflight_.find(item.req.key);
                    if (it != inflight_.end()) {
                        cancelled = it->second.cancelled;
                        preempt   = !cancelled && !it->second.promoted_to_demand && !q_demand_.empty();
                    }
                }
                if (cancelled) {
                    publish_span_ready(item.req.slot, s, ns, false);
                    for (int rem = s + 1; rem < ns; ++rem) publish_span_ready(item.req.slot, rem, ns, false);
                    finish_prefetch(item.req, false, bytes, true, chunks);
                    return;
                }
                if (preempt) {
                    // Yield immediately: re-queue remaining chunks at the FRONT of q_prefetch so already-bound
                    // slots always finish before binding new slots (bounds in-flight bound prefetches <= n_readers)!
                    item.span_idx = s;
                    item.span_off = off;
                    {
                        std::lock_guard<std::mutex> lk(mu_);
                        inflight_.erase(item.req.key);
                        stats_.prefetch_preempted++;
                        stats_.prefetch_chunks += chunks;
                        stats_.bytes_read += bytes;
                        q_prefetch_.push_front(std::move(item));
                    }
                    cv_.notify_one();
                    return;
                }

                const size_t step = std::min(chunk_bytes_, sp.bytes - off);
                if (!read_fn_(sp.file_idx, sp.file_off + off, item.req.dst + sp.dst_off + off, step)) {
                    sp_ok = false;
                    all_ok = false;
                    break;
                }
                off += step;
                bytes += step;
                chunks++;
            }
            publish_span_ready(item.req.slot, s, ns, sp_ok);
        }

        finish_prefetch(item.req, all_ok, bytes, false, chunks);
    }

    void finish_prefetch(const StreamExpertReq& req, bool ok, uint64_t bytes, bool dropped, uint64_t chunks = 0) {
        if (complete_fn_ && (!dropped || req.slot >= 0)) complete_fn_(req, ok && !dropped, true);
        {
            std::unique_lock<std::mutex> lk(mu_);
            inflight_.erase(req.key);
            if (dropped) stats_.prefetch_dropped++;
            else stats_.prefetch_completed++;
            stats_.prefetch_chunks += chunks;
            stats_.bytes_read += bytes;
            if (!ok && !dropped) stats_.io_errors++;
            pending_prefetch_.fetch_sub(1, std::memory_order_release);
        }
        done_cv_.notify_all();
    }

    ReadChunkFn read_fn_;
    BindSlotFn  bind_fn_;
    CompleteFn  complete_fn_;
    size_t chunk_bytes_ = kPrefetchChunkBytes;
    std::unique_ptr<SlotProjState[]> slots_;
    int n_slots_ = 0;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    bool stopping_ = false;
    std::deque<StreamExpertReq> q_demand_;
    std::deque<PrefetchItem>    q_prefetch_;
    std::unordered_map<int64_t, InflightMeta> inflight_;
    std::atomic<int> demand_waiting_{0};
    std::atomic<int> pending_demand_{0};
    std::atomic<int> pending_prefetch_{0};
    std::atomic<uint64_t> cancel_epoch_{0};
    SsdSchedulerStats stats_{};
    std::vector<std::thread> workers_;
};

/// Full-duplex VRAM -> RAM copyback queue: when an SSD-backed expert is evicted from VRAM cache,
/// captures its bytes over the idle reverse PCIe D2H lane into RAM cache so future misses hit RAM
/// instead of NVMe.
class VramCopybackQueue {
public:
    struct Job {
        int64_t key = -1;
        int64_t layer = 0;
        int64_t expert = 0;
        const void* dev_src = nullptr;
        uint8_t* host_dst = nullptr;
        size_t bytes = 0;
    };

    using CopyFn = std::function<bool(const Job&)>;

    explicit VramCopybackQueue(size_t max_queued = 32) : max_queued_(max_queued) {}
    ~VramCopybackQueue() { stop(); }

    void start(CopyFn copy_fn) {
        stop();
        copy_fn_ = std::move(copy_fn);
        {
            std::unique_lock<std::mutex> lk(mu_);
            stopping_ = false;
            q_.clear();
        }
        worker_ = std::thread([this]() { run(); });
    }

    void stop() {
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (!worker_.joinable()) return;
            stopping_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        std::lock_guard<std::mutex> lk(mu_);
        q_.clear();
    }

    bool try_enqueue(const Job& j) {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopping_ || q_.size() >= max_queued_) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        q_.push_back(j);
        cv_.notify_one();
        return true;
    }

    void drain() {
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&]() { return q_.empty() && !busy_; });
    }

    uint64_t completed() const noexcept { return completed_.load(std::memory_order_relaxed); }
    uint64_t dropped()   const noexcept { return dropped_.load(std::memory_order_relaxed); }

private:
    void run() {
        for (;;) {
            Job j{};
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]() { return stopping_ || !q_.empty(); });
                if (stopping_ && q_.empty()) return;
                j = q_.front();
                q_.pop_front();
                busy_ = true;
            }
            if (copy_fn_ && copy_fn_(j)) {
                completed_.fetch_add(1, std::memory_order_relaxed);
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                busy_ = false;
            }
            done_cv_.notify_all();
        }
    }

    size_t max_queued_ = 32;
    CopyFn copy_fn_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    bool stopping_ = false;
    bool busy_ = false;
    std::deque<Job> q_;
    std::thread worker_;
    std::atomic<uint64_t> completed_{0};
    std::atomic<uint64_t> dropped_{0};
};

/// Idle pre-loader: after `idle_threshold_ms` (default 500 ms) of request quiet time, streams cold
/// experts from NVMe into spare RAM/VRAM cache slots one step at a time, yielding in < 100 us the
/// instant `notify_active()` is called by any incoming prompt or decode token.
class IdlePreloader {
public:
    /// Returns true if one unit of idle preload work was done (and more may remain), false if done/full.
    using StepFn = std::function<bool(const std::atomic<bool>& should_yield)>;

    IdlePreloader() = default;
    ~IdlePreloader() { stop(); }

    void start(StepFn step_fn, int idle_threshold_ms = 500) {
        stop();
        step_fn_ = std::move(step_fn);
        threshold_ns_ = static_cast<int64_t>(std::max(1, idle_threshold_ms)) * 1000000LL;
        last_active_ns_.store(now_ns(), std::memory_order_release);
        active_flag_.store(false, std::memory_order_release);
        stopping_.store(false, std::memory_order_release);
        worker_ = std::thread([this]() { run(); });
    }

    void stop() {
        if (!worker_.joinable()) return;
        stopping_.store(true, std::memory_order_release);
        active_flag_.store(true, std::memory_order_release);
        {
            std::unique_lock<std::mutex> lk(mu_);
        }
        cv_.notify_all();
        worker_.join();
    }

    /// Call on prompt/decode entry; wakes/yields background preloader in < 100 us.
    void notify_active() noexcept {
        active_flag_.store(true, std::memory_order_release);
        last_active_ns_.store(now_ns(), std::memory_order_release);
    }

    /// Call when request finishes and engine becomes idle.
    void notify_idle() noexcept {
        last_active_ns_.store(now_ns(), std::memory_order_release);
        active_flag_.store(false, std::memory_order_release);
        cv_.notify_one();
    }

    uint64_t steps_done() const noexcept { return steps_done_.load(std::memory_order_relaxed); }

private:
    static int64_t now_ns() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void run() {
        while (!stopping_.load(std::memory_order_acquire)) {
            if (active_flag_.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait_for(lk, std::chrono::milliseconds(5), [&]() {
                    return stopping_.load(std::memory_order_acquire) || !active_flag_.load(std::memory_order_acquire);
                });
                continue;
            }
            const int64_t idle_for = now_ns() - last_active_ns_.load(std::memory_order_acquire);
            if (idle_for < threshold_ns_) {
                const int64_t rem_us = std::max<int64_t>(50, (threshold_ns_ - idle_for) / 1000LL);
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait_for(lk, std::chrono::microseconds(std::min<int64_t>(rem_us, 10000)), [&]() {
                    return stopping_.load(std::memory_order_acquire);
                });
                continue;
            }
            if (!step_fn_ || !step_fn_(active_flag_)) {
                // Nothing left to preload right now: sleep lightly until next idle window
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait_for(lk, std::chrono::milliseconds(50), [&]() {
                    return stopping_.load(std::memory_order_acquire);
                });
                continue;
            }
            steps_done_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    StepFn step_fn_;
    int64_t threshold_ns_ = 500000000LL;
    std::atomic<int64_t> last_active_ns_{0};
    std::atomic<bool> active_flag_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<uint64_t> steps_done_{0};
    std::mutex mu_;
    std::condition_variable cv_;
    std::thread worker_;
};

}  // namespace strata::core
