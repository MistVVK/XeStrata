// tools/vision/own/cpu_team.h - a small fork-join team of persistent threads for the CPU tower.
//
// run(n, f) calls f(task, thread) for task = 0 .. n-1 on the team (the caller is thread 0) and returns when all are done.  Workers spin for a short time
// between runs (a tower makes hundreds of runs per picture, a few microseconds apart) and then sleep on a condition variable, so an idle reader costs nothing.
// Claiming a task is a compare-and-swap on a counter tagged with the run's generation (the completion protocol of the engine's expert pool): a worker that
// wakes late can never take a task of the next run, and no task runs twice.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#if defined(_MSC_VER)
#include <intrin.h>
#define STRATA_VIT_PAUSE() _mm_pause()
#else
#define STRATA_VIT_PAUSE() __builtin_ia32_pause()
#endif
#else
#define STRATA_VIT_PAUSE() ((void) 0)
#endif

namespace strata_vit {

class Team {
public:
    explicit Team(int n) : n_(n < 1 ? 1 : n) {
        for (int t = 1; t < n_; ++t) workers_.emplace_back([this, t] { loop(t); });
    }
    ~Team() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_.store(true);
            gen_.fetch_add(1);
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
    Team(const Team&) = delete;
    Team& operator=(const Team&) = delete;
    int size() const { return n_; }

    void run(int ntasks, const std::function<void(int task, int thread)>& f) {
        if (ntasks <= 0) return;
        if (n_ == 1 || ntasks == 1) {
            for (int i = 0; i < ntasks; ++i) f(i, 0);
            return;
        }
        fn_ = &f;
        remaining_.store(ntasks, std::memory_order_relaxed);
        const uint64_t g = gen_.load(std::memory_order_relaxed) + 1;
        limit_.store((g << 32) | (uint32_t) ntasks, std::memory_order_relaxed);     // the task count carries the run's tag, as the claim counter does
        next_.store(g << 32, std::memory_order_release);
        gen_.store(g, std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
        work(g, 0);
        while (remaining_.load(std::memory_order_acquire) > 0) STRATA_VIT_PAUSE();
    }

private:
    // claim and run tasks of generation g
    void work(uint64_t g, int thread) {
        for (;;) {
            uint64_t cur = next_.load(std::memory_order_acquire);
            if ((cur >> 32) != g) return;                    // a later run is on: this one is over
            const uint32_t idx = (uint32_t) cur;
            // The task count must be read WITH its tag: a late worker (a run behind) that read a plain count already rewritten for the next run could claim an
            // index past its own run's end while the counter still shows the old run, and run a task of the new run's function twice (found by the unit test, 31
            // failures in 60 runs).  A count of another run ends this worker's pass.
            const uint64_t lim = limit_.load(std::memory_order_relaxed);
            if ((lim >> 32) != g || idx >= (uint32_t) lim) return;
            if (!next_.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel)) continue;
            (*fn_)((int) idx, thread);
            remaining_.fetch_sub(1, std::memory_order_release);
        }
    }

    void loop(int thread) {
        uint64_t seen = 0;
        for (;;) {
            uint64_t g = gen_.load(std::memory_order_seq_cst);
            if (g == seen) {
                const auto t0 = std::chrono::steady_clock::now();
                int spins = 0;
                while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                    STRATA_VIT_PAUSE();
                    if ((++spins & 255) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(150)) {
                        std::unique_lock<std::mutex> lk(mu_);
                        sleepers_.fetch_add(1, std::memory_order_seq_cst);
                        cv_.wait(lk, [&] { return gen_.load(std::memory_order_seq_cst) != seen; });
                        sleepers_.fetch_sub(1, std::memory_order_seq_cst);
                        g = gen_.load(std::memory_order_seq_cst);
                        break;
                    }
                }
            }
            if (stop_.load()) return;
            seen = g;
            work(g, thread);
        }
    }

    const int n_;
    std::vector<std::thread> workers_;
    const std::function<void(int, int)>* fn_ = nullptr;
    std::atomic<int> remaining_{0};
    std::atomic<uint64_t> next_{0}, gen_{0}, limit_{0};      // next_: (run tag << 32) | next task; limit_: (run tag << 32) | task count
    std::atomic<int> sleepers_{0};
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::condition_variable cv_;
};

}  // namespace strata_vit
