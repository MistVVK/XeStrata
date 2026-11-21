// engine/src/inkling2/hip/ink_hip_hybrid.hpp
// Strata Predictive CPU/RAM Hybrid Engine for AMD HIP (8-16 GB discrete RDNA2/RDNA3/RDNA4 + DDR4/DDR5):
//   1. Coherent Host Doorbell + Spin-Wait AVX-512/AVX2 CPU Expert Worker Pool (computes RAM experts in parallel
//      with GPU VRAM experts and returns only 9 KB/tok instead of streaming 4.5 MB expert weights over PCIe!).
//   2. Early-Router + 3x-Loop Predictive Async DMA Prefetcher (hipMemcpyAsync on dedicated dma_stream_ into an
//      elastic VRAM cache slot ring with decayed-frequency promotion & per-slot hipEvent_t readiness).
//   3. Chunked Double-Buffered GPU Matrix-Core Expert Streaming during Prefill (overlaps PCIe H2D into
//      d_stage[2] with GPU unpack + hipBLASLt/WMMA compute on stream_!).
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_common.hip.h"
#include "strata/inkling2/inkling2_loop.hpp"
#include "strata/quant/sk_codecs.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::inkling2::hip {

// Persistent spin-wait CPU worker pool executing RAM-resident experts via strata::inkling2::eval_inkling2_expert_batch
// (AVX-512 VNNI / AVX2 / FMA) in parallel with GPU VRAM expert execution!
class HybridCpuExpertPool {
public:
    struct CpuJob {
        int dir_idx = -1;
        const uint8_t* body = nullptr;
        uint8_t gu_code = 0;
        uint8_t dn_code = 0;
        uint32_t dn_offset = 0;
        // Up to B tokens routed to this expert
        std::vector<int32_t> tok_idx;
        std::vector<float> tok_w;
    };

    ~HybridCpuExpertPool() { shutdown(); }

    void init(int num_threads = 0) {
        shutdown();
        const int hw = static_cast<int>(std::thread::hardware_concurrency());
        int nt = (num_threads > 0) ? num_threads : std::max(2, std::min(24, hw > 4 ? (hw / 2) : hw));
        n_threads_ = std::max(1, nt);
        stop_.store(false, std::memory_order_release);
        gen_.store(0, std::memory_order_release);
        done_count_.store(0, std::memory_order_release);
        workers_.reserve(n_threads_);
        for (int t = 0; t < n_threads_; ++t) {
            workers_.emplace_back([this, t]() { worker_loop(t); });
        }
    }

    void shutdown() {
        if (workers_.empty()) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_.store(true, std::memory_order_release);
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
        n_threads_ = 0;
    }

    // Non-blocking launch: dispatches all CPU jobs to the spin-wait pool while the GPU executes VRAM experts!
    void launch_async(const std::vector<CpuJob>* jobs, const float* x_in, float* y_accum,
                      int B, int D, int inter) {
        if (!jobs || jobs->empty() || workers_.empty()) {
            in_flight_ = false;
            return;
        }
        jobs_ = jobs;
        x_in_ = x_in;
        y_accum_ = y_accum;
        B_ = B;
        D_ = D;
        inter_ = inter;
        if (static_cast<int>(thread_bufs_.size()) < n_threads_) {
            thread_bufs_.resize(n_threads_);
        }
        const size_t bd = static_cast<size_t>(B) * D;
        for (int t = 0; t < n_threads_; ++t) {
            if (thread_bufs_[t].size() < bd) thread_bufs_[t].resize(bd);
        }
        next_job_.store(0, std::memory_order_relaxed);
        done_count_.store(0, std::memory_order_relaxed);
        in_flight_ = true;

        const uint64_t next_g = gen_.load(std::memory_order_relaxed) + 1;
        gen_.store(next_g, std::memory_order_release);
        if (sleeping_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
    }

    // Waits for all CPU jobs to finish and reduces per-thread accumulators deterministically into y_accum[B, D]!
    void wait_and_reduce() {
        if (!in_flight_) return;
        while (done_count_.load(std::memory_order_acquire) < n_threads_) {
            std::this_thread::yield();
        }
        in_flight_ = false;
        const size_t bd = static_cast<size_t>(B_) * D_;
        std::memset(y_accum_, 0, bd * sizeof(float));
        for (int t = 0; t < n_threads_; ++t) {
            if (!thread_touched_[t]) continue;
            const float* src = thread_bufs_[t].data();
            for (size_t i = 0; i < bd; ++i) y_accum_[i] += src[i];
        }
    }

    int num_threads() const noexcept { return n_threads_; }

private:
    void worker_loop(int tid) {
        uint64_t seen = 0;
        std::vector<float> x_gather;
        std::vector<float> y_local;
        while (true) {
            // Brief spin-wait (~200us) for ultra-low decode doorbell latency, then sleep on CV if idle
            uint64_t cur_g = gen_.load(std::memory_order_acquire);
            if (cur_g == seen) {
                for (int sp = 0; sp < 4000; ++sp) {
                    cur_g = gen_.load(std::memory_order_acquire);
                    if (cur_g != seen || stop_.load(std::memory_order_acquire)) break;
#if defined(__x86_64__) || defined(_M_X64)
                    __builtin_ia32_pause();
#endif
                }
            }
            if (cur_g == seen && !stop_.load(std::memory_order_acquire)) {
                sleeping_.fetch_add(1, std::memory_order_acq_rel);
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]() {
                    return gen_.load(std::memory_order_acquire) != seen || stop_.load(std::memory_order_acquire);
                });
                sleeping_.fetch_sub(1, std::memory_order_acq_rel);
                cur_g = gen_.load(std::memory_order_acquire);
            }
            if (stop_.load(std::memory_order_acquire)) return;
            seen = cur_g;

            const size_t bd = static_cast<size_t>(B_) * D_;
            float* tbuf = thread_bufs_[tid].data();
            bool touched = false;
            const int n_jobs = static_cast<int>(jobs_->size());
            while (true) {
                const int ji = next_job_.fetch_add(1, std::memory_order_relaxed);
                if (ji >= n_jobs) break;
                if (!touched) {
                    std::memset(tbuf, 0, bd * sizeof(float));
                    touched = true;
                }
                const CpuJob& job = (*jobs_)[ji];
                const int m = static_cast<int>(job.tok_idx.size());
                if (m <= 0 || !job.body) continue;

                const auto gu = static_cast<strata::quant::sk::SkCode>(job.gu_code);
                const auto dn = static_cast<strata::quant::sk::SkCode>(job.dn_code);
                if (m == 1) {
                    if (y_local.size() < static_cast<size_t>(D_)) y_local.resize(D_);
                    const int b = job.tok_idx[0];
                    const float w = job.tok_w[0];
                    strata::inkling2::eval_inkling2_expert(
                        job.body, gu, dn, job.dn_offset,
                        x_in_ + static_cast<size_t>(b) * D_, D_, inter_, y_local.data());
                    float* dst = tbuf + static_cast<size_t>(b) * D_;
                    for (int d = 0; d < D_; ++d) dst[d] += w * y_local[d];
                } else {
                    const size_t md = static_cast<size_t>(m) * D_;
                    if (x_gather.size() < md) x_gather.resize(md);
                    if (y_local.size() < md) y_local.resize(md);
                    for (int i = 0; i < m; ++i) {
                        std::memcpy(x_gather.data() + static_cast<size_t>(i) * D_,
                                    x_in_ + static_cast<size_t>(job.tok_idx[i]) * D_,
                                    static_cast<size_t>(D_) * sizeof(float));
                    }
                    strata::inkling2::eval_inkling2_expert_batch(
                        job.body, gu, dn, job.dn_offset,
                        x_gather.data(), m, D_, inter_, y_local.data());
                    for (int i = 0; i < m; ++i) {
                        const int b = job.tok_idx[i];
                        const float w = job.tok_w[i];
                        const float* src = y_local.data() + static_cast<size_t>(i) * D_;
                        float* dst = tbuf + static_cast<size_t>(b) * D_;
                        for (int d = 0; d < D_; ++d) dst[d] += w * src[d];
                    }
                }
            }
            thread_touched_[tid] = touched ? 1 : 0;
            done_count_.fetch_add(1, std::memory_order_release);
        }
    }

    int n_threads_ = 0;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> next_job_{0};
    std::atomic<int> done_count_{0};
    std::atomic<int> sleeping_{0};
    std::mutex mu_;
    std::condition_variable cv_;

    bool in_flight_ = false;
    const std::vector<CpuJob>* jobs_ = nullptr;
    const float* x_in_ = nullptr;
    float* y_accum_ = nullptr;
    int B_ = 0, D_ = 0, inter_ = 0;
    std::vector<std::vector<float>> thread_bufs_;
    uint8_t thread_touched_[64]{};
};

// Elastic VRAM Cache Ring + 3x-Loop Predictive Async DMA Prefetcher + Double-Buffered Prefill Streamer!
class HipHybridController {
public:
    struct CacheSlot {
        int dir_idx = -1;
        uint64_t last_use = 0;
        hipEvent_t ready_ev = nullptr;
        bool dma_in_flight = false;
    };

    ~HipHybridController() { release(); }

    bool init(size_t n_dir, size_t max_exp_bytes, size_t cache_budget_bytes, bool is_uma,
              int cpu_threads, std::string& err) {
        release();
        if (max_exp_bytes == 0) return true;
        is_uma_ = is_uma;
        slot_stride_ = (max_exp_bytes + 63u) & ~size_t(63u);

        if (hipStreamCreateWithFlags(&dma_stream_, hipStreamNonBlocking) != hipSuccess) {
            err = "hybrid: failed to create dma_stream";
            return false;
        }
        // Double-buffered GPU staging for chunked prefill Matrix-Core expert streaming!
        for (int i = 0; i < 2; ++i) {
            if (strata::hip_uma::alloc_gpu_buffer(d_stage_[i], slot_stride_, is_uma_) != hipSuccess ||
                hipEventCreateWithFlags(&stage_ev_[i], hipEventDisableTiming) != hipSuccess ||
                hipEventCreateWithFlags(&compute_ev_[i], hipEventDisableTiming) != hipSuccess) {
                err = "hybrid: failed to allocate double-buffered prefill staging";
                release();
                return false;
            }
        }

        // Elastic VRAM cache slot ring (carved out of vram_budget so --vram-experts-gb is strictly respected!)
        const int n_slots = static_cast<int>(cache_budget_bytes / slot_stride_);
        if (n_slots > 0) {
            const size_t ring_bytes = static_cast<size_t>(n_slots) * slot_stride_;
            if (strata::hip_uma::alloc_gpu_buffer(d_cache_ring_, ring_bytes, is_uma_) == hipSuccess) {
                slots_.resize(n_slots);
                for (int i = 0; i < n_slots; ++i) {
                    (void)hipEventCreateWithFlags(&slots_[i].ready_ev, hipEventDisableTiming);
                }
            }
        }
        dir_to_slot_.assign(n_dir, -1);
        decayed_freq_.assign(n_dir, 0.0f);

        // Coherent host doorbell & activation/output buffers (zero-copy mapped pinned memory!)
        if (!ensure_doorbell_cap(16, 2304, 8)) {
            err = "hybrid: failed to allocate coherent host doorbell buffers";
            release();
            return false;
        }
        cpu_pool_.init(cpu_threads);
        active_ = true;
        return true;
    }

    bool ensure_doorbell_cap(int B, int D, int top_k) {
        if (B <= db_cap_B_ && D <= db_cap_D_) return true;
        free_doorbell();
        const int nb = std::max(B, 16);
        const int nd = std::max(D, 2304);
        const unsigned flags = hipHostMallocCoherent | hipHostMallocMapped;
        if (hipHostMalloc(&h_x_,     static_cast<size_t>(nb) * nd * sizeof(float), flags) != hipSuccess ||
            hipHostMalloc(&h_y_,     static_cast<size_t>(nb) * nd * sizeof(float), flags) != hipSuccess ||
            hipHostMalloc(&h_ids_,   static_cast<size_t>(nb) * top_k * sizeof(int32_t), flags) != hipSuccess ||
            hipHostMalloc(&h_ws_,    static_cast<size_t>(nb) * top_k * sizeof(float), flags) != hipSuccess ||
            strata::hip_uma::alloc_gpu_buffer(d_step_exps_, 128u * sizeof(GpuExpertDesc), is_uma_) != hipSuccess ||
            strata::hip_uma::alloc_gpu_buffer(d_cpu_y_, static_cast<size_t>(nb) * nd * sizeof(float), is_uma_) != hipSuccess) {
            free_doorbell();
            return false;
        }
        db_cap_B_ = nb;
        db_cap_D_ = nd;
        return true;
    }

    // Check if dir_idx is ready right now in the elastic VRAM cache ring (zero stall!).
    // Returns device pointer if resident & DMA complete, else nullptr.
    const uint8_t* probe_ready(int dir_idx) noexcept {
        if (!active_ || dir_idx < 0 || dir_idx >= static_cast<int>(dir_to_slot_.size())) return nullptr;
        const int s = dir_to_slot_[dir_idx];
        if (s < 0) return nullptr;
        auto& sl = slots_[s];
        if (sl.dma_in_flight) {
            if (hipEventQuery(sl.ready_ev) != hipSuccess) return nullptr;
            sl.dma_in_flight = false;
        }
        sl.last_use = ++clock_;
        ++cache_hits_;
        return d_cache_ring_.dev_ptr + static_cast<size_t>(s) * slot_stride_;
    }

    // Predictive async DMA prefetch into LRU elastic VRAM slot (triggered for 3x-loop core experts & hot RAM experts!)
    void prefetch_async(int dir_idx, const uint8_t* host_body, size_t body_bytes, bool is_core_loop_repeat) noexcept {
        if (!active_ || slots_.empty() || !host_body || body_bytes == 0 || body_bytes > slot_stride_) return;
        if (dir_idx < 0 || dir_idx >= static_cast<int>(dir_to_slot_.size())) return;

        // Decayed frequency tracking
        decayed_freq_[dir_idx] = decayed_freq_[dir_idx] * 0.96f + 1.0f;
        if (dir_to_slot_[dir_idx] >= 0) {
            slots_[dir_to_slot_[dir_idx]].last_use = ++clock_;
            return;
        }
        // Promote immediately on 3x-loop core passes (pass 0/1 -> will repeat in pass 1/2!) OR when hot (freq >= 1.8)
        if (!is_core_loop_repeat && decayed_freq_[dir_idx] < 1.8f) return;

        // Pick an idle LRU slot (skip slots whose DMA is still in flight)
        int victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
            auto& sl = slots_[i];
            if (sl.dma_in_flight) {
                if (hipEventQuery(sl.ready_ev) != hipSuccess) continue;
                sl.dma_in_flight = false;
            }
            if (sl.dir_idx < 0) { victim = i; break; }
            if (sl.last_use < oldest) { oldest = sl.last_use; victim = i; }
        }
        if (victim < 0) return;

        auto& vsl = slots_[victim];
        if (vsl.dir_idx >= 0) dir_to_slot_[vsl.dir_idx] = -1;
        vsl.dir_idx = dir_idx;
        vsl.last_use = ++clock_;
        vsl.dma_in_flight = true;
        dir_to_slot_[dir_idx] = victim;

        uint8_t* dst = d_cache_ring_.dev_ptr + static_cast<size_t>(victim) * slot_stride_;
        if (hipMemcpyAsync(dst, host_body, body_bytes, hipMemcpyHostToDevice, dma_stream_) == hipSuccess) {
            (void)hipEventRecord(vsl.ready_ev, dma_stream_);
            ++prefetches_issued_;
        } else {
            dir_to_slot_[dir_idx] = -1;
            vsl.dir_idx = -1;
            vsl.dma_in_flight = false;
        }
    }

    // Async stage expert into double-buffered prefill slot (buf_idx in {0, 1}) on dma_stream_!
    // Waits for prior compute on slot (buf_idx & 1) before overwriting it -> 100% race-free double buffering!
    bool stage_prefill_async(int buf_idx, const uint8_t* host_body, size_t body_bytes, bool wait_prior_compute) noexcept {
        const int bi = buf_idx & 1;
        if (wait_prior_compute) {
            (void)hipStreamWaitEvent(dma_stream_, compute_ev_[bi], 0);
        }
        if (hipMemcpyAsync(d_stage_[bi].dev_ptr, host_body, body_bytes, hipMemcpyHostToDevice, dma_stream_) != hipSuccess) {
            return false;
        }
        return hipEventRecord(stage_ev_[bi], dma_stream_) == hipSuccess;
    }

    const uint8_t* wait_prefill_stage(int buf_idx, hipStream_t compute_stream) noexcept {
        const int bi = buf_idx & 1;
        (void)hipStreamWaitEvent(compute_stream, stage_ev_[bi], 0);
        return d_stage_[bi].dev_ptr;
    }

    void mark_prefill_compute_done(int buf_idx, hipStream_t compute_stream) noexcept {
        (void)hipEventRecord(compute_ev_[buf_idx & 1], compute_stream);
    }

    bool active() const noexcept { return active_; }
    size_t cache_slots() const noexcept { return slots_.size(); }
    size_t allocated_bytes() const noexcept {
        return d_cache_ring_.bytes + d_stage_[0].bytes + d_stage_[1].bytes + d_step_exps_.bytes + d_cpu_y_.bytes;
    }
    uint64_t cache_hits() const noexcept { return cache_hits_; }
    uint64_t prefetches_issued() const noexcept { return prefetches_issued_; }

    float* h_x() noexcept { return static_cast<float*>(h_x_); }
    float* h_y() noexcept { return static_cast<float*>(h_y_); }
    int32_t* h_ids() noexcept { return static_cast<int32_t*>(h_ids_); }
    float* h_ws() noexcept { return static_cast<float*>(h_ws_); }
    float* d_cpu_y() noexcept { return reinterpret_cast<float*>(d_cpu_y_.dev_ptr); }
    GpuExpertDesc* d_step_exps() noexcept { return reinterpret_cast<GpuExpertDesc*>(d_step_exps_.dev_ptr); }
    hipStream_t dma_stream() noexcept { return dma_stream_; }
    HybridCpuExpertPool& pool() noexcept { return cpu_pool_; }

    void release() {
        cpu_pool_.shutdown();
        if (dma_stream_) {
            (void)hipStreamSynchronize(dma_stream_);
            (void)hipStreamDestroy(dma_stream_);
            dma_stream_ = nullptr;
        }
        for (int i = 0; i < 2; ++i) {
            if (stage_ev_[i]) { (void)hipEventDestroy(stage_ev_[i]); stage_ev_[i] = nullptr; }
            if (compute_ev_[i]) { (void)hipEventDestroy(compute_ev_[i]); compute_ev_[i] = nullptr; }
            strata::hip_uma::free_gpu_buffer(d_stage_[i]);
        }
        for (auto& sl : slots_) {
            if (sl.ready_ev) (void)hipEventDestroy(sl.ready_ev);
        }
        slots_.clear();
        strata::hip_uma::free_gpu_buffer(d_cache_ring_);
        dir_to_slot_.clear();
        decayed_freq_.clear();
        free_doorbell();
        active_ = false;
    }

private:
    void free_doorbell() {
        if (h_x_)   { (void)hipHostFree(h_x_);   h_x_ = nullptr; }
        if (h_y_)   { (void)hipHostFree(h_y_);   h_y_ = nullptr; }
        if (h_ids_) { (void)hipHostFree(h_ids_); h_ids_ = nullptr; }
        if (h_ws_)  { (void)hipHostFree(h_ws_);  h_ws_ = nullptr; }
        strata::hip_uma::free_gpu_buffer(d_step_exps_);
        strata::hip_uma::free_gpu_buffer(d_cpu_y_);
        db_cap_B_ = db_cap_D_ = 0;
    }

    bool active_ = false;
    bool is_uma_ = false;
    size_t slot_stride_ = 0;
    hipStream_t dma_stream_ = nullptr;
    strata::hip_uma::GpuBuffer d_stage_[2];
    hipEvent_t stage_ev_[2]{};
    hipEvent_t compute_ev_[2]{};
    strata::hip_uma::GpuBuffer d_cache_ring_;
    std::vector<CacheSlot> slots_;
    std::vector<int> dir_to_slot_;
    std::vector<float> decayed_freq_;
    uint64_t clock_ = 0;
    uint64_t cache_hits_ = 0;
    uint64_t prefetches_issued_ = 0;

    int db_cap_B_ = 0, db_cap_D_ = 0;
    void* h_x_ = nullptr;
    void* h_y_ = nullptr;
    void* h_ids_ = nullptr;
    void* h_ws_ = nullptr;
    strata::hip_uma::GpuBuffer d_step_exps_;
    strata::hip_uma::GpuBuffer d_cpu_y_;
    HybridCpuExpertPool cpu_pool_;
};

}  // namespace strata::inkling2::hip
