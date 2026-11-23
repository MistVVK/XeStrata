// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/cpu/pool.hpp - P2.S3: the CPU expert pool.
//
// A layer runs TEN experts against ONE activation, and the expert kernel is DRAM-bound (Memory/LEDGER.md L9:
// 42.55 GB/s on 6 cores, tracking core count almost exactly).  So the pool's job is not to be clever - it is
// to keep every physical core reading expert bytes for the whole layer, and to be cheap enough that ten
// dispatches per layer cost less than one expert.
//
// WHY A FLAT BATCH AND NOT A RING.  P2.S3 describes a "lock-free SPMC ring", which is what you need when
// jobs arrive while others are still being computed.  Here they cannot: the host must SUM all ten outputs
// before the next layer starts, so a layer is a barrier by construction and the queue never holds more than
// one batch.  A ring would add a wrap-around to get wrong and buy nothing.  What is kept from the phase is
// the part that matters: `head`/`done` are single fetch_add counters, one claim per worker, no lock.
//
// THE COMPLETION PROTOCOL, because this is where a pool usually goes wrong.  `run()` waits for `done == n`
// AND for every worker to PARK.  Waiting only for `done` is not enough: a worker can still be inside the
// drain loop after its last `done` increment, and the host resetting `head` underneath it would let that
// worker claim a job from the NEXT batch before the next batch has been published.
//
// **AND `parked` IS NOT ENOUGH EITHER (issue #29).**  A worker that went to sleep (after `kSpinBeforeSleep`) is
// still counted as parked when it wakes, so for a moment after it has seen a new epoch the host believes it is
// idle.  On a card with most experts in VRAM the workers sleep in the middle of a request, tiny batches finish
// before a sleeper is awake, and that moment comes round constantly: a late worker could claim from the NEXT
// batch while the host was still writing it, run a job twice, or add to `done` after the host had reset it -
// and `done != n` then never ended, with the GPU waiting on the pool forever.  So every claim carries its
// batch: `head` is one 64-bit word `epoch | njobs | index`, a claim is a CAS that only succeeds for the epoch
// the worker woke for, and a late worker's claim simply fails.  A claim that succeeds belongs to the current
// batch, whose description the host cannot change until that job's `done` has landed.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool_cores.hpp"

#include <atomic>
#include <cstdio>
#include <functional>
#include <memory>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::kernels::cpu {

/// One expert evaluation.  `act` is SHARED and read-only across the whole batch - that sharing is the point
/// of `s2_expert_vnni_q` and it is what saves 480 redundant activation conversions per token.
struct ExpertJob {
    const uint8_t* blob = nullptr;   ///< one 1,382,400-byte expert
    const ActQ* act = nullptr;       ///< the layer's quantized activation, shared
    float* out = nullptr;            ///< H floats, written by exactly one worker
    float weight = 1.0f;             ///< the router weight; applied by the HOST, not here
    int slot = -1;                   ///< the job's index, for diagnostics
};

/// Plan v0.3 P6: one expert for the `nt` tokens of a verify window that were routed to it.  Every token's output
/// is bitwise the single-token job's.
struct ExpertJobMulti {
    const uint8_t* blob = nullptr;
    int nt = 0;
    const ActQ* act[MAXT] = {};
    float* out[MAXT] = {};
    /// Plan v0.3 P6: a native pack's activations (native_quant_act's, in a NativeAct), one per token.
    const void* nact[MAXT] = {};
    /// Upstream 149 (zero-copy): this job's own layout (null: the layer's) - the GGUF read in place, where gate, up and
    /// down are three tensors: `blob` is the gate rows and the offsets reach the expert's up and down rows in the mapping.
    const NativeFmt* fmt = nullptr;
};

/// One logical processor per PHYSICAL core, so a worker is never scheduled onto an SMT sibling of another
/// worker.  On the 6-core/12-thread machine this project measures on, `hardware_concurrency()/2` workers on
/// logical processors 0..5 would put every worker on a sibling pair and halve the useful bandwidth - which is
/// exactly the kind of error that shows up as "the CPU path is slower than the model says" with no clue why.
///
/// `skip_first` drops the first core, which P2.S3 reserves for the host loop.
std::vector<int> physical_cores(bool skip_first);

/// `--pool-cores LIST` / STRATA_POOL_CORES / the config's "pool_cores" (experimental, off by default; upstream
/// 7a5599581): pin the pool's workers to exactly these logical processors, in this order, instead of one per physical
/// core.  An empty list (the default) leaves the placement as it has always been.  Set before the pool is built.
/// Processors the process may not run on are left out with a warning; SMT siblings of one core and the host's processor
/// are kept with a warning.  The worker count is the list's length unless --pool-workers says otherwise.
void set_pool_cores(const std::vector<int>& cpus);
const std::vector<int>& pool_cores_setting();

/// Every logical processor this process may run on and the physical core it belongs to (pool_cores.hpp's
/// LogicalCpuInfo).  Empty when the system does not say.
std::vector<LogicalCpuInfo> logical_cpu_layout();

/// **THE RESERVATION IS A FICTION UNLESS THE HOST IS ACTUALLY PUT THERE.**
///
/// `physical_cores(true)` keeps the workers off the first physical core so that the host loop can spin on
/// `cudaEventQuery` without stealing a worker's cycles.  Nothing in the pool can enforce the other half of
/// that, so this is it: the host loop calls this on entry and restores on exit.
///
/// MEASURED, and this is why it exists: the pool runs at **36.32 GB/s on 5 workers with nothing else running**
/// - exactly 5/6 of L9's 44.14 on 6 - and at **26.9 GB/s inside the host loop**, where the unpinned spinning
/// host is free to land on a worker's core or its SMT sibling.  That 1.35x is not the kernel.
///
/// Returns the PREVIOUS affinity (`valid` false if the platform refused); pass it to `restore_thread_affinity` on the
/// same thread.  Windows selects a CPU Set, which can be undone, rather than a hard group affinity: a caller's implicit
/// all-group eligibility (Windows 11) cannot be put back from a group affinity.
struct ThreadAffinity {
#if defined(_WIN32)
    std::vector<unsigned long> cpu_sets;   ///< the thread's selected CPU Sets; empty: the process defaults
#else
    std::vector<int> cpus;                 ///< every CPU in the calling thread's previous mask
#endif
    bool valid = false;
};
ThreadAffinity pin_current_thread(int core);
void restore_thread_affinity(const ThreadAffinity& previous);

/// Hold the host on its reserved core for a forward, restoring its former placement on return.
class ScopedThreadPin {
public:
    explicit ScopedThreadPin(int cpu) { if (cpu >= 0) previous_ = pin_current_thread(cpu); }
    ~ScopedThreadPin() { restore_thread_affinity(previous_); }
    ScopedThreadPin(const ScopedThreadPin&) = delete;
    ScopedThreadPin& operator=(const ScopedThreadPin&) = delete;
private:
    ThreadAffinity previous_;
};

class ExpertPool {
public:
    /// `n_workers <= 0` means "every physical core except the first".  Workers are pinned to physical cores
    /// (minus core 0 by default) and each owns one `ExpertScratch`, so nothing in the token path allocates.
    ///
    /// **`host_works` PUTS THE HOST THREAD INTO THE DRAIN (R2.2's FIRST HALF).**
    ///
    /// The pool reserves core 0 for the host loop so the doorbell spin cannot steal a worker's cycles - but
    /// during `run()` the host does not spin, it waits, so core 0 is idle for the whole drain. Measured on the
    /// 6-core machine this project targets: the engine's pool drains at **33.7 GB/s** (663.6 MB of expert
    /// blobs in 19.71 ms/token) where the same kernel on 5 workers should reach 5/6 x 44.14 = 36.8 and the
    /// machine measures 44.14 GB/s on all six. So the sixth core is being paid for and not used.
    ///
    /// With `host_works`, `run()` claims jobs itself instead of spinning on `done_`, and the pool is six
    /// threads on six cores. `false` is the A/B arm and exists so the change is measurable rather than
    /// asserted - the counter it moves is `pool phases ... drain`, which is host-side and needs no profiler.
    /// `first_core`: the workers take the physical cores from that one on (a second pool beside a first, which took
    /// the cores below it: a layer split's stages each with workers of their own).
    /// `tasks`: the row tasks of each batched expert phase (--pool-tasks), 0 = three per participating thread; a
    /// count is capped by the phase's rows.
    /// `generic` makes a pool for independent pieces; n_workers then includes the calling thread (0: auto).
    explicit ExpertPool(int n_workers = 0, bool pin = true, bool host_works = true, int first_core = 0, int tasks = 0,
                        bool generic = false);
    static constexpr int kMaxTasks = 4096;
    /// The watchdog's view of the pool (issue #31): the batch, the counters, every thread's state.
    void diag(std::FILE* f) const;
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    int workers() const { return n_; }
    int size() const { return n_ + (host_works_ ? 1 : 0); }
    /// Run independent pieces on the same pool, the caller included.
    void run(int n, const std::function<void(int)>& fn);
    /// `n` independent items, `fn(i)` each: in pieces of `chunk` items on the pool (run) when there are at least
    /// `min_items` of them and the pool has workers, else in line on the caller.  Each item writes only its own
    /// output, so the bits are the same either way (upstream 431a26db1's run_range: the CPU prompt share's activations
    /// quantized by the pool, host grouping 81 -> 7 ms on an RTX 5090).  XESTRATA_POOL_ITEMS=0: always in line.
    void run_items(int n, int chunk, int min_items, const std::function<void(int)>& fn);
    /// Whether the host thread also drains.  Reported at startup, because "the engine adapts to the machine it
    /// is on" is only true if the engine says which adaptation it took.
    bool host_works() const { return host_works_; }
    /// The logical CPUs the workers are pinned to (--aux-cpus keeps every other thread off them).
    const std::vector<int>& worker_cpus() const { return worker_cpus_; }
    /// True when the workers sit where `--pool-cores` said (the list, less what could not be used).
    bool explicit_cores() const { return explicit_cores_; }
    /// The start-up line about where the workers and the host run ("CPU workers on logical processors 2,4,...,30 (15
    /// workers, one per physical core; host on 0)"); "" without workers.
    std::string placement_line() const;

    /// Publish `n` jobs, then block until every one has been claimed AND every worker has parked.
    /// `jobs` must outlive the call (it does, and the workers never touch it afterwards).
    void run(ExpertJob* jobs, int n);

    /// Plan v0.3 P4: the same outputs as `run`, bitwise, with every expert split by rows across all threads
    /// (gate/up rows, then the intermediate's quantization, then down rows).  With fewer experts than threads -
    /// the case once the VRAM tier takes half of them - `run` leaves cores idle and each expert streams at one
    /// core's bandwidth; this streams every expert at all of them.  At most `kMaxSplit` experts.
    void run_split(ExpertJob* jobs, int n);
    static constexpr int kMaxSplit = 16;
    /// Plan v0.3 P6: `run_split` for multi-token jobs (at most `kMaxSplitMulti`); the rows of each expert are
    /// read once for all of its tokens.
    void run_split_multi(ExpertJobMulti* jobs, int n);
    /// Plan v0.3 P6: the same for a native pack's layer (ggml-cpu arithmetic, `nact` activations).
    void run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n);
    static constexpr int kMaxSplitMulti = 96;
    /// run_split_multi's phases, accumulated ms: gate/up rows, the intermediate quantization, down rows.
    double ms_multi_gu = 0, ms_multi_q = 0, ms_multi_down = 0;
    int64_t multi_bytes = 0;

    /// Total `_mm_pause` iterations spent waiting, over all workers, is no longer counted - see the note on the
    /// atomics below.  It was a LOCKED read-modify-write in the park loop, so measuring the contention added to
    /// it.
    long long pauses() const { return 0; }

    /// **WHERE `run()` SPENDS ITS TIME, in milliseconds accumulated over its lifetime.**  Three phases per
    /// layer - wait for every worker to be parked, wait for the drain, wait for them to re-park - and until now
    /// all three were reported as one number.  Without the split there is no way to tell a pool that is slow at
    /// the WORK from one that is slow at the SYNCHRONISATION, and those need opposite fixes: the first is a
    /// kernel problem and the second is a barrier problem.
    ///
    /// Only the host thread touches these, in `run()`, so they need no atomics.
    void phase_ms(double& wait_park, double& drain, double& repark) const {
        wait_park = ms_wait_park_;
        drain = ms_drain_;
        repark = ms_repark_;
    }

    /// **A PARKED WORKER SPINS FOR THIS LONG, THEN SLEEPS.**  The park is a `_mm_pause` spin because a layer's
    /// batches are microseconds apart and a wake-up from the OS costs more than that.  But a spin that never ends
    /// keeps every worker's core at 100% while the engine waits for a request - issue #4, "CPU 50% even when
    /// doing nothing" (7 of the 5700X's 16 threads).  Between requests the workers block on `sleep_cv_`.  NOT only
    /// between requests: with most experts in VRAM (a 24 GB card) many layers have no CPU work, so the workers
    /// also sleep mid-request - which the claim protocol above must survive (issue #29).
    /// `STRATA_POOL_SPIN_US` overrides it (a test knob: a short spin makes the workers sleep constantly).
    static constexpr std::chrono::milliseconds kSpinBeforeSleep{20};
    /// A pool wait that sees no completion for this long is a bug; the engine stops with a message instead of
    /// spinning forever, and the server starts it again (issue #29).
    static constexpr std::chrono::seconds kStall{60};

private:
    int phase_tasks(int64_t rows) const;
    int tasks_ = 0;
    void worker(int id);
    void drain(int id, ExpertScratch& scratch, uint32_t epoch);
    void run_phase(int mode, int n_tasks);
    /// Claim the next job of batch `epoch`, or -1 (that batch is exhausted, or it is not the current one).
    int claim(uint32_t epoch);
    /// Publish the batch whose description the caller has just written: reset `done`, then `head`, then the epoch.
    uint32_t begin_batch(int n);
    /// The host's waits, bounded by `kStall`.
    void wait_parked(const char* what);
    void wait_done(int n);
    /// Bump `epoch_`, and wake the workers that went to sleep.  Every publish goes through here.
    void publish();

    int n_ = 0;
    bool host_works_ = true;
    ExpertJob* jobs_ = nullptr;
    int njobs_ = 0;
    /// The host's own scratch when `host_works_`.  A separate object rather than a share of `scratch_[i]`,
    /// because a worker may own any index and the two must not be able to collide.
    ExpertScratch host_scratch_;
    // `run()`'s three phases, accumulated.  Host-thread only; see `phase_ms`.
    double ms_wait_park_ = 0.0;
    double ms_drain_ = 0.0;
    double ms_repark_ = 0.0;
    // ---- EACH ATOMIC GETS ITS OWN CACHE LINE, AND THE SPIN COUNTER IS GONE.  (Review finding C3.)
    //
    // These were six adjacent atomics, which put `head_`, `done_`, `parked_` and `epoch_` on ONE cache line -
    // the four that workers and the host actually contend on, invalidating each other on every access.
    //
    // Worse, the parked spin did `pauses_.fetch_add(1)` on EVERY iteration: a LOCKED read-modify-write, five
    // workers against one line, at roughly one iteration per `_mm_pause`.  So the line the host must WRITE to
    // publish work (`epoch_`) and READ to confirm the workers are parked (`parked_`) was being hammered by the
    // very threads waiting for it.  That is contention the pool imposes on itself.
    //
    // The counter was diagnostic only - `pauses()` was read in one place, to print a number nothing branched on
    // - so it is deleted rather than amortised.  `alignas(64)` then stops the remaining four sharing.
    // issue #31 diagnostics: each worker's state (kParked, kSleeping, kBetween, or the job it runs) and the host's
    // (kIdle, kWaitParked, kWaitDone, or its job), printed by the serve watchdog through `diag`
    static constexpr int32_t kParked = -1, kSleeping = -2, kBetween = -3, kIdle = -10, kWaitParked = -11,
                             kWaitDone = -12;
    std::unique_ptr<std::atomic<int32_t>[]> wstate_;
    std::atomic<int32_t> hstate_{kIdle};
    std::atomic<int64_t> hstate_ms_{0};
    alignas(64) std::atomic<uint64_t> head_{0};   // epoch << 32 | njobs << 16 | next index (issue #29)
    alignas(64) std::atomic<uint32_t> done_{0};
    alignas(64) std::atomic<uint32_t> parked_{0};
    alignas(64) std::atomic<uint32_t> epoch_{0};
    alignas(64) std::atomic<bool> stop_{false};
    // The sleep after `kSpinBeforeSleep`.  `sleepers_` is how `publish` knows whether anyone needs waking, so the
    // token path pays one uncontended load per publish and never takes the mutex while the workers spin.
    alignas(64) std::atomic<uint32_t> sleepers_{0};
    std::mutex sleep_mu_;
    std::condition_variable sleep_cv_;
    std::chrono::microseconds spin_before_sleep_{kSpinBeforeSleep};
    std::vector<std::thread> threads_;
    std::vector<int> worker_cpus_;
    bool explicit_cores_ = false;
    bool pinned_ = true;
    std::vector<ExpertScratch> scratch_;   // one per worker: no allocation, no false sharing of the hot data
    // run_split state: mode 0 = whole experts, 1 = gate/up row parts, 2 = down row parts (7 = run(n, fn), 8 = the
    // one-batch native layer call of XESTRATA_POOL_DEP)
    int mode_ = 0;
    const std::function<void(int)>* fn_ = nullptr;
    int fn_offset_ = 0;
    int parts_a_ = 1, parts_b_ = 1;
    struct SplitBuf {
        alignas(64) float ff[FF];
        ActQ a2;
    };
    std::vector<SplitBuf> split_;
    // run_split_multi state: mode 3 = gate/up row parts, 4 = down row parts
    ExpertJobMulti* mjobs_ = nullptr;
    int64_t mrows_ = 0;     // rows of the current multi phase across all its experts (n * FF, then n * H;
                            // a native layer's n_ff and n_embd)
    int mtasks_ = 1;        // equal row ranges the phase is cut into
    struct SplitBufMulti {
        alignas(64) float ff[MAXT][FF > kNativeMaxFF ? FF : kNativeMaxFF];
        ActQ a2[MAXT];
        alignas(64) uint8_t hq[MAXT][kNativeHBytes];   // plan v0.3 P6: native down activations
    };
    const NativeFmt* nfmt_ = nullptr;
    std::vector<SplitBufMulti> split_multi_;
    // XESTRATA_POOL_DEP (on unless "0"; upstream edef565b, 902ea232): a native layer call as ONE batch (mode 8) instead
    // of two phases with a barrier between them.  The batch's tasks are the experts' gate/up row ranges (expert-major),
    // then the experts' down row ranges.  A down task waits only for its own expert's gate/up tasks (counted in
    // gu_left), and the first thread that gets there quantizes that expert's intermediate (once: quant 0 -> 1 -> 2), so
    // the threads that finish the gate/up rows early start on the experts that are ready instead of waiting for the
    // slowest task of the phase, and the host does not quantize between the phases.  Every row is computed by the same
    // kernel from the same bytes: the outputs are the two-phase batch's bits.  Every gate/up task has a lower index than
    // every down task, a claim hands the indices out in order, and a gate/up task waits for nothing, so a waiting down
    // task always waits for a task that is running: no deadlock.
    struct alignas(64) DepExpert {
        std::atomic<int32_t> gu_left{0};
        std::atomic<int32_t> quant{0};
    };
    std::unique_ptr<DepExpert[]> dep_;   // kMaxSplitMulti
    bool dep_on_ = true;                 // XESTRATA_POOL_DEP, read once in the constructor
    int dep_nb_ = 0, dep_k_gu_ = 1, dep_k_dn_ = 1;   // the current batch: experts, gate/up and down tasks per expert
    void dep_task(int i);
};

}  // namespace strata::kernels::cpu
