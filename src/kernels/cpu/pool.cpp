// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/cpu/pool.cpp - P2.S3: the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/platform/thread_name.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/cpu/canon_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>   // __cpuidex (STRATA_POOL_UMWAIT)
#else
#include <cpuid.h>    // __cpuid_count (STRATA_POOL_UMWAIT)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::kernels::cpu {

namespace {
// STRATA_POOL_UMWAIT=<TSC cycles> (opt-in, upstream caf1856b, 54b61dbb): a parked worker waits with UMONITOR/UMWAIT on
// the epoch's cache line instead of spinning on _mm_pause.  It wakes when the host publishes (a write to that line) or
// after the cycles, in the light C0.1 state; on a laptop whose CPU and GPU share a power budget the spin's power goes
// to the GPU.  0, unset, or a CPU without WAITPKG (CPUID.7.0:ECX bit 5): the spin, unchanged.
// STRATA_POOL_UMWAIT_C02=1: the deeper C0.2 state; STRATA_POOL_UMWAIT_ESC=<cycles>: C0.1 first, C0.2 after.
// The instructions are compiled for this one function (clang and GCC, on Linux and Windows), not for the file.
bool cpu_waitpkg() {
    unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    int x[4];
    __cpuidex(x, 0, 0);
    if (x[0] < 7) return false;
    __cpuidex(x, 7, 0);
    r[2] = (unsigned) x[2];
#else
    __cpuid_count(0, 0, r[0], r[1], r[2], r[3]);
    if (r[0] < 7) return false;
    __cpuid_count(7, 0, r[0], r[1], r[2], r[3]);
#endif
    return ((r[2] >> 5) & 1u) != 0;
}
uint64_t umwait_cycles() {
    static const uint64_t c = [] {
        const char* e = strata::env_get("POOL_UMWAIT");
        if (e == nullptr || std::strtoll(e, nullptr, 10) <= 0) return (uint64_t) 0;
        if (!cpu_waitpkg()) {
            std::fprintf(stderr, "xestrata: XESTRATA_POOL_UMWAIT: this CPU has no WAITPKG, the workers spin\n");
            return (uint64_t) 0;
        }
        std::fprintf(stderr, "xestrata: XESTRATA_POOL_UMWAIT: parked workers wait with UMWAIT (%lld cycles)\n", std::strtoll(e, nullptr, 10));
        return (uint64_t) std::strtoll(e, nullptr, 10);
    }();
    return c;
}
unsigned umwait_state() {   // 1 = C0.1 (light), 0 = C0.2
    static const unsigned s = [] { const char* e = strata::env_get("POOL_UMWAIT_C02"); return e && *e == '1' ? 0u : 1u; }();
    return s;
}
uint64_t umwait_escalate() {   // a park that has lasted this long is a GPU phase, where a slower wake costs little
    static const uint64_t c = [] {
        const char* e = strata::env_get("POOL_UMWAIT_ESC");
        return e ? (uint64_t) (std::max)(0LL, std::strtoll(e, nullptr, 10)) : (uint64_t) 0;
    }();
    return c;
}
// false when `epoch` moved before the wait (the monitor is armed first, so a publish after the check wakes it)
#if defined(__clang__) || defined(__GNUC__)
__attribute__((target("waitpkg")))
#endif
bool umwait_unless_moved(const std::atomic<uint32_t>& epoch, uint32_t seen, unsigned state, uint64_t until) {
    _umonitor((void*) &epoch);
    if (epoch.load(std::memory_order_acquire) != seen) return false;
    _umwait(state, until);
    return true;
}

constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}
}  // namespace

std::vector<int> physical_cores(bool skip_first) {
    std::vector<int> cores;
#if defined(_WIN32)
    // One logical processor per core from the OS, numbered group * 64 + the processor's bit in its group.  Without
    // the core list (an old Windows) every active processor counts, group by group.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
    if (len != 0 && GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) {
        for (const char* p = buf.data(); p < buf.data() + len;) {
            const auto* e = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
            if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount > 0) {
                const GROUP_AFFINITY& g = e->Processor.GroupMask[0];
                for (int bit = 0; bit < 64; ++bit)
                    if (g.Mask & (KAFFINITY(1) << bit)) {
                        cores.push_back((int) g.Group * 64 + bit);
                        break;
                    }
            }
            p += e->Size;
        }
    } else {
        for (WORD group = 0; group < GetActiveProcessorGroupCount(); ++group) {
            const DWORD count = GetActiveProcessorCount(group);
            for (DWORD i = 0; i < count && i < 64; ++i) cores.push_back((int) group * 64 + (int) i);
        }
    }
#else
    // The logical CPUs this process may run on, ONE PER PHYSICAL CORE (issue #40): SMT siblings share a core's
    // load/store bandwidth, so a worker on each would put two workers on one core. sysfs names each CPU's
    // (package, core); the first allowed CPU of each pair is kept, so a taskset
    // that leaves out the first sibling still gets its core.  Without sysfs every allowed CPU counts, as before.
    auto topo = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    std::vector<int> allowed;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) allowed.push_back(i);
    } else {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) allowed.push_back((int) i);
    }
    std::vector<std::pair<long, long>> seen;
    for (int cpu : allowed) {
        const long pkg = topo(cpu, "physical_package_id"), core = topo(cpu, "core_id");
        if (pkg >= 0 && core >= 0) {
            const std::pair<long, long> key{pkg, core};
            if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;   // an SMT sibling
            seen.push_back(key);
        }
        cores.push_back(cpu);
    }
#endif
    if (skip_first && !cores.empty()) cores.erase(cores.begin());
    return cores;
}

namespace {
std::vector<int> g_pool_cores;   // --pool-cores: set once at start-up, before the pool exists
}  // namespace
void set_pool_cores(const std::vector<int>& cpus) { g_pool_cores = cpus; }
const std::vector<int>& pool_cores_setting() { return g_pool_cores; }

std::vector<LogicalCpuInfo> logical_cpu_layout() {
    std::vector<LogicalCpuInfo> out;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return out;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(),
                                          &len))
        return out;
    // a `start /affinity` or a job limit on a PC with one processor group: only those processors are the process's
    DWORD_PTR proc_mask = 0, sys_mask = 0;
    const bool limited = GetActiveProcessorGroupCount() == 1 &&
                         GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask);
    long core = 0;
    for (const char* p = buf.data(); p < buf.data() + len;
         p += ((const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p)->Size) {
        const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
        if (e->Relationship != RelationProcessorCore) continue;
        for (WORD group = 0; group < e->Processor.GroupCount; ++group) {
            const GROUP_AFFINITY& g = e->Processor.GroupMask[group];
            for (int bit = 0; bit < 64; ++bit)
                if ((g.Mask & (KAFFINITY(1) << bit)) && (!limited || (proc_mask & (DWORD_PTR(1) << bit))))
                    out.push_back({(int) (g.Group * 64 + bit), core});
        }
        ++core;
    }
#else
    auto topo = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return out;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) continue;
        const long pkg = topo(cpu, "physical_package_id"), core = topo(cpu, "core_id");
        out.push_back({cpu, pkg >= 0 && core >= 0 ? pkg * 1000000 + core : -1});
    }
#endif
    return out;
}

namespace {

#if defined(_WIN32)
// The CPU Set of a processor numbered group * 64 + index: CPU Set IDs are opaque, so the entry is found by its group and
// group-relative index.
bool cpu_set_for_core(int core, ULONG& id) {
    ULONG bytes = 0;
    if (!GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
    std::vector<unsigned char> buffer(bytes);
    if (bytes && !GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()), bytes,
                                            &bytes, GetCurrentProcess(), 0)) return false;
    constexpr size_t header_size = offsetof(SYSTEM_CPU_SET_INFORMATION, CpuSet);
    for (size_t offset = 0; offset + header_size <= bytes;) {
        const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
        if (info->Size < header_size || info->Size > bytes - offset) break;
        if (info->Type == CpuSetInformation && info->Size >= sizeof(SYSTEM_CPU_SET_INFORMATION) &&
            info->CpuSet.Group == core / 64 && info->CpuSet.LogicalProcessorIndex == (core & 63) &&
            (!info->CpuSet.Allocated || info->CpuSet.AllocatedToTargetProcess)) {
            id = info->CpuSet.Id;
            return true;
        }
        offset += info->Size;
    }
    SetLastError(ERROR_NOT_FOUND);
    return false;
}
#endif

// Hard affinity: a pool worker keeps it until it ends; on Linux pin_current_thread saves the mask before it.
void pin_this_thread(int core) {
    if (core < 0) return;
#if defined(_WIN32)
    GROUP_AFFINITY target{};
    target.Group = (WORD) (core / 64);
    target.Mask = KAFFINITY(1) << (core & 63);
    if (!SetThreadGroupAffinity(GetCurrentThread(), &target, nullptr))
        std::fprintf(stderr, "xestrata cpu pool: affinity for processor %d failed: %lu; previous affinity kept\n", core,
                     (unsigned long) GetLastError());
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

}  // namespace

ThreadAffinity pin_current_thread(int core) {
    if (core < 0) return {};
    ThreadAffinity previous;
#if defined(_WIN32)
    ULONG count = 0, target = 0;
    if (!GetThreadSelectedCpuSets(GetCurrentThread(), nullptr, 0, &count) && GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return {};
    previous.cpu_sets.resize(count);
    if ((count != 0 && !GetThreadSelectedCpuSets(GetCurrentThread(), previous.cpu_sets.data(), count, &count)) ||
        !cpu_set_for_core(core, target) || !SetThreadSelectedCpuSets(GetCurrentThread(), &target, 1)) {
        std::fprintf(stderr, "xestrata cpu pool: host CPU Set selection for processor %d failed: %lu; previous placement "
                             "kept\n", core, (unsigned long) GetLastError());
        return {};
    }
    previous.cpu_sets.resize(count);
#else
    cpu_set_t prev;
    CPU_ZERO(&prev);
    if (pthread_getaffinity_np(pthread_self(), sizeof prev, &prev) != 0) return {};
    for (int i = 0; i < CPU_SETSIZE; ++i)
        if (CPU_ISSET(i, &prev)) previous.cpus.push_back(i);
    pin_this_thread(core);
#endif
    previous.valid = true;
    return previous;
}

void restore_thread_affinity(const ThreadAffinity& previous) {
    if (!previous.valid) return;
#if defined(_WIN32)
    // an empty selection puts the thread back on the process defaults
    if (!SetThreadSelectedCpuSets(GetCurrentThread(), previous.cpu_sets.empty() ? nullptr : previous.cpu_sets.data(),
                                  (ULONG) previous.cpu_sets.size()))
        std::fprintf(stderr, "xestrata cpu pool: host CPU Set restoration failed: %lu\n", (unsigned long) GetLastError());
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : previous.cpus) CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

namespace {
std::atomic<const ExpertPool*> g_diag_pool{nullptr};
void diag_active_pool(std::FILE* f) {
    if (const ExpertPool* p = g_diag_pool.load()) p->diag(f);
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

void ExpertPool::diag(std::FILE* f) const {
    const uint64_t h = head_.load();
    std::fprintf(f, "  expert pool: epoch %u, batch epoch %u: %u of %u jobs claimed, %u done; %u of %d workers parked, "
                    "%u sleeping; mode %d\n", epoch_.load(), (uint32_t) (h >> 32), (uint32_t) h & 0xffffu,
                 (uint32_t) (h >> 16) & 0xffffu, done_.load(), parked_.load(), n_, sleepers_.load(), mode_);
    std::fprintf(f, "  expert pool threads:");
    for (int i = 0; i < n_; ++i) {
        const int32_t s = wstate_[(size_t) i].load();
        if (s == kParked) std::fprintf(f, " w%d=parked", i);
        else if (s == kSleeping) std::fprintf(f, " w%d=sleeping", i);
        else if (s == kBetween) std::fprintf(f, " w%d=draining", i);
        else std::fprintf(f, " w%d=job%d", i, s);
    }
    const int32_t hs = hstate_.load();
    const char* hn = hs == kIdle ? "idle" : hs == kWaitParked ? "waiting for the workers to park"
                   : hs == kWaitDone ? "waiting for the jobs to finish" : "running a job";
    std::fprintf(f, "; host %s", hn);
    if (hs >= 0) std::fprintf(f, " %d", hs);
    std::fprintf(f, " for %lld ms\n", (long long) (now_ms() - hstate_ms_.load()));
}

// The default workers at most: 32 (upstream 036c8f6ec, a 32-core EPYC: 24 lost 2-5% of the decode, 32 and 48 were
// equal, 63 oversubscribed lost 35%), and no more than the container's CPU quota (cgroup cpu.max / cfs_quota_us).
// Upstream's default for a hybrid CPU with more E-cores than P-cores (the P-cores and half the E-cores, bec4c838a) is
// not taken: on an i7-14700 it cost XeStrata 3% of the decode and 13% of the prompt (13 workers against 19, B70,
// GLM-5.3-Flash and llm-jp-4.1, 2026-10-20).
static int default_worker_cap() {
    int cap = 32;
#if defined(__linux__)
    auto first_line = [](const char* path) {
        std::string line;
        if (std::FILE* f = std::fopen(path, "r")) {
            char buf[64] = {0};
            if (std::fgets(buf, sizeof buf, f) != nullptr) line = buf;
            std::fclose(f);
        }
        return line;
    };
    double quota = 0;
    const std::string v2 = first_line("/sys/fs/cgroup/cpu.max");   // "max 100000" or "<quota> <period>"
    if (!v2.empty()) {
        char* end = nullptr;
        const double q = std::strtod(v2.c_str(), &end);
        const double period = end != v2.c_str() ? std::strtod(end, nullptr) : 0;
        if (end != v2.c_str() && q > 0 && period > 0) quota = q / period;
    } else {   // v1: cpu.cfs_quota_us (-1: none) / cpu.cfs_period_us
        const double q = std::strtod(first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").c_str(), nullptr);
        const double period = std::strtod(first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us").c_str(), nullptr);
        if (q > 0 && period > 0) quota = q / period;
    }
    // the host loop takes a core of the quota too
    if (quota > 0) cap = std::min(cap, std::max(1, (int) std::floor(quota) - 1));
#endif
    return cap;
}

ExpertPool::ExpertPool(int n_workers, bool pin, bool host_works, int first_core, int tasks, bool generic)
    : tasks_(tasks), host_works_(host_works) {
    if (tasks < 0 || tasks > kMaxTasks) throw std::invalid_argument("expert pool tasks must be in 0..4096");
    if (const char* e = strata::env_get("POOL_SPIN_US"))   // a test knob; see kSpinBeforeSleep
        spin_before_sleep_ = std::chrono::microseconds((std::max)(0, std::atoi(e)));
    pinned_ = pin;
    std::vector<int> cores = physical_cores(true);
    // --pool-cores (experimental, off by default): the person's own list of logical processors for the workers, in its
    // order; only where the workers are pinned (and, by default, their count) comes from it
    if (!pool_cores_setting().empty()) {
        const std::vector<int> all = physical_cores(false);   // the host takes the first one (SessionLoopScratch)
        const PoolCoresCheck chk = check_pool_cores(pool_cores_setting(), logical_cpu_layout(), all.empty() ? -1 : all[0]);
        static std::atomic<bool> told{false};   // (a second pool, --batch-cpu-split's, checks the same list again)
        const bool say = !told.exchange(true);
        if (say)
            for (const std::string& w : chk.warnings) std::fprintf(stderr, "xestrata cpu pool: --pool-cores: %s\n", w.c_str());
        if (chk.cpus.empty()) {
            if (say)
                std::fprintf(stderr, "xestrata cpu pool: --pool-cores: no usable logical processor in the list; the pool "
                                     "uses its default placement\n");
        } else {
            cores = chk.cpus;
            explicit_cores_ = true;
        }
    }
    n_ = n_workers > 0 ? n_workers
                       : explicit_cores_ ? (int) cores.size()
                                         : (int) std::min<size_t>(cores.size(), (size_t) default_worker_cap());
    if (generic && n_workers > 0) n_ = n_workers - 1;
    if (!generic && n_ < 1) n_ = 1;
    scratch_.resize((size_t) n_);
    wstate_.reset(new std::atomic<int32_t>[(size_t) n_]);
    for (int i = 0; i < n_; ++i) wstate_[(size_t) i].store(kParked);
    hstate_ms_.store(now_ms());
    if (!generic) {
        g_diag_pool.store(this);
        strata::core::diag_pool_fn().store(&diag_active_pool);
        split_.resize((size_t) kMaxSplit);
        split_multi_.resize((size_t) kMaxSplitMulti);
    }
    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) {
        const int c = first_core + i;
        const int core = pin ? (c < (int) cores.size() ? cores[(size_t) c] : -1) : -1;
        if (core >= 0) worker_cpus_.push_back(core);
        threads_.emplace_back([this, i, core] {
            strata::platform::name_this_thread("xs-pool", i);
            strata::aux_cpus::note_owned_thread();   // --aux-cpus never moves a worker
            pin_this_thread(core);
            worker(i);
        });
    }
}

std::string ExpertPool::placement_line() const {
    if (n_ <= 0) return "";
    if (!pinned_)
        return "CPU workers: " + std::to_string(n_) + " workers, not pinned to logical processors (the OS places them)";
    const std::vector<int> all = physical_cores(false);   // the host takes the first one (SessionLoopScratch)
    std::string s = describe_worker_placement(worker_cpus_, all.empty() ? -1 : all[0], logical_cpu_layout(),
                                              explicit_cores_);
    // a worker beyond the cores listed runs unpinned (more --pool-workers than logical processors)
    if ((int) worker_cpus_.size() < n_)
        s += "; " + std::to_string(n_ - (int) worker_cpus_.size()) + " more not pinned";
    return s;
}

ExpertPool::~ExpertPool() {
    const ExpertPool* self = this;
    g_diag_pool.compare_exchange_strong(self, nullptr);
    stop_.store(true, std::memory_order_release);
    // Bump the epoch so a PARKED worker notices the stop flag rather than sleeping through it.
    publish();
    for (auto& t : threads_) t.join();
}

void ExpertPool::publish() {
    // Both sides are seq_cst, and that is the whole lost-wakeup argument: a worker going to sleep does
    // `sleepers_++` and then reads `epoch_`, the host does `epoch_++` and then reads `sleepers_`.  In one total
    // order at least one of them sees the other's write - the worker sees the new epoch and does not sleep, or
    // the host sees the sleeper and notifies under the mutex the worker holds until it is inside `wait`.
    // On x86 the fetch_add is a locked xadd either way, so this costs the token path nothing.
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
}

void ExpertPool::worker(int id) {
    uint32_t seen = 0;
    // ARRIVE at the park before the first wait, so `parked_ == n_` is true from construction.  Counting only
    // on the RETURN from a drain leaves `parked_` at 0 until each worker has finished one batch, and the first
    // `run()` - which waits for `parked_ == n_` before publishing - then deadlocks.  It deadlocks on the very
    // first call, which is the good case; a version that deadlocked on the second would be far worse.
    parked_.fetch_add(1, std::memory_order_acq_rel);
    for (;;) {
        // Park: wait for work.  `_mm_pause` rather than a bare spin because it yields the pipeline to the
        // sibling hyperthread; `epoch_` is bumped once per LAYER, not once per expert, so most of these
        // iterations are spent here with nothing to do.
        //
        // **AND NOTHING ELSE HAPPENS IN HERE.**  This loop used to do `pauses_.fetch_add(1)` on every iteration
        // - a locked read-modify-write, five workers against one cache line - so the workers spent their wait
        // invalidating each other's caches and the very line the host writes to publish work.  The counter was
        // diagnostic and nothing branched on it.  See the note on the atomics in pool.hpp.
        //
        // After `kSpinBeforeSleep` with no work the worker sleeps instead (issue #4).  The clock is read once
        // every 1024 pauses, so the spin itself is unchanged.
        const auto parked_at = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        const uint64_t um = umwait_cycles(), um_esc = umwait_escalate();
        const uint64_t um_t0 = um != 0 && um_esc != 0 ? __rdtsc() : 0;
        while (epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
            if (um != 0) {   // STRATA_POOL_UMWAIT: wake on the publish's write to epoch_, or after `um` cycles
                const uint64_t now = __rdtsc();
                const unsigned st = um_esc != 0 ? (now - um_t0 < um_esc ? 1u : 0u) : umwait_state();
                if (!umwait_unless_moved(epoch_, seen, st, now + um)) break;
                if ((++spins & 15u) != 0) continue;   // each wait is ~um cycles: look at the clock more often
            } else {
                _mm_pause();
                if ((++spins & 1023u) != 0) continue;
            }
            if (std::chrono::steady_clock::now() - parked_at < spin_before_sleep_) continue;
            std::unique_lock<std::mutex> lk(sleep_mu_);
            wstate_[(size_t) id].store(kSleeping, std::memory_order_relaxed);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lk, [&] {
                return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_relaxed);
            });
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
            wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        // acquire: the batch this epoch published (`head`, and the description before it) is visible from here
        seen = epoch_.load(std::memory_order_acquire);
        parked_.fetch_sub(1, std::memory_order_acq_rel);   // leaving the park

        // Drain: one claim per iteration, so a slow worker takes fewer experts and a fast one takes more.
        // Every job is the same size (all experts are 1,382,400 bytes), so there is nothing to schedule.  Only
        // this epoch's jobs: if the host has already moved on, the claims fail and the worker parks again.
        wstate_[(size_t) id].store(kBetween, std::memory_order_relaxed);
        drain(id, scratch_[(size_t) id], seen);
        wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        parked_.fetch_add(1, std::memory_order_acq_rel);   // back at the park
    }
}

int ExpertPool::claim(uint32_t epoch) {
    uint64_t h = head_.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;               // not the batch this thread woke for
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                       // exhausted
        if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return (int) i;
    }
}

uint32_t ExpertPool::begin_batch(int n) {
    if (n < 0 || n > 0xffff) {
        std::fprintf(stderr, "xestrata: expert pool batch of %d jobs is out of range\n", n);
        std::abort();
    }
    // Every job of the previous batch has completed (`wait_done`), and a claim of it can no longer succeed, so
    // nothing adds to `done` until this batch's first claim - which the release below orders after the reset.
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;   // only the host bumps the epoch
    head_.store(pack_head(e, (uint32_t) n, 0), std::memory_order_release);
    publish();
    return e;
}

void ExpertPool::wait_parked(const char* what) {
    hstate_.store(kWaitParked, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0;
    std::chrono::steady_clock::time_point t0{};
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) {
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u) t0 = now;
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "xestrata: the CPU expert pool stalled %s (%u of %d workers parked) - stopping the engine "
                                 "so the server can start it again (issue #29)\n",
                         what, parked_.load(), n_);
            strata::core::release_gpu_waits(stderr);   // #267: the GPU may be spinning on this layer's flag
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::wait_done(int n) {
    hstate_.store(kWaitDone, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0, seen = 0;
    std::chrono::steady_clock::time_point t0{};
    for (;;) {
        const uint32_t d = done_.load(std::memory_order_acquire);
        if (d >= (uint32_t) n) return;                                // `>=`: never a wait that an overshoot outlives
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u || d != seen) { t0 = now; seen = d; }     // progress restarts the clock
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "xestrata: the CPU expert pool stalled: %u of %d jobs done, %u of %d workers parked - "
                                 "stopping the engine so the server can start it again (issue #29)\n",
                         d, n, parked_.load(), n_);
            strata::core::release_gpu_waits(stderr);   // #267
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::drain(int id, ExpertScratch& scratch, uint32_t epoch) {
    (void) id;
    for (;;) {
        const int ci = claim(epoch);
        if (ci < 0) break;
        const uint32_t i = (uint32_t) ci;
        if (id >= 0) wstate_[(size_t) id].store(ci, std::memory_order_relaxed);
        else { hstate_.store(ci, std::memory_order_relaxed); hstate_ms_.store(now_ms(), std::memory_order_relaxed); }
        if (mode_ == 7) {
            (*fn_)(fn_offset_ + (int) i);
        } else if (mode_ == 0) {
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni_q(j.blob, *j.act, j.out, scratch);
        } else if (mode_ == 1) {
            const int e = (int) i / parts_a_, part = (int) i % parts_a_;
            const int r0 = FF * part / parts_a_, r1 = FF * (part + 1) / parts_a_;
            s2_expert_gu_rows(jobs_[e].blob, *jobs_[e].act, split_[(size_t) e].ff, r0, r1);
        } else if (mode_ == 2) {
            const int e = (int) i / parts_b_, part = (int) i % parts_b_;
            const int r0 = H * part / parts_b_, r1 = H * (part + 1) / parts_b_;
            s2_expert_down_rows(jobs_[e].blob, split_[(size_t) e].a2, jobs_[e].out, r0, r1);
        } else if (mode_ >= 5) {
            // plan v0.3 P6: native layers, 5 = gate/up rows, 6 = down rows
            const int per = (int) (mode_ == 5 ? nfmt_->n_ff : nfmt_->n_embd);
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                const NativeFmt* const nfmt = mjobs_[e].fmt != nullptr ? mjobs_[e].fmt : nfmt_;   // zero-copy: same types and rows
                if (mode_ == 5) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    native_gu_rows(*nfmt, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                } else if (nfmt->d_type == 42 && !canon_layer(*nfmt)) {
                    // Q2_0 down (most IQ layers): the AVX-512 kernel, ggml-cpu has only a scalar one on x86; a layer of
                    // the canonical rows (canon_expert.hpp) takes native_down_rows, the GPU's arithmetic
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    q2_rows_any(mjobs_[e].blob + nfmt->down_off, nfmt->d_row, (int) (nfmt->n_ff / 64), a2,
                                mjobs_[e].nt, mjobs_[e].out, r0, r1);
                } else {
                    const void* hq[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) hq[t] = sb.hq[t];
                    native_down_rows(*nfmt, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        } else {
            // plan v0.3 P6: an equal range of the phase's rows across ALL its experts (a range may span two)
            const int per = mode_ == 3 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 3) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    s2_expert_gu_rows_multi(mjobs_[e].blob, mjobs_[e].act, mjobs_[e].nt, ff, r0, r1);
                } else {
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    s2_expert_down_rows_multi(mjobs_[e].blob, a2, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        }
        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::run_phase(int mode, int n_tasks) {
    wait_parked("before a phase");
    mode_ = mode;
    njobs_ = n_tasks;
    const uint32_t e = begin_batch(n_tasks);
    if (host_works_) drain(-1, host_scratch_, e);
    wait_done(n_tasks);
    wait_parked("after a phase");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
}

void ExpertPool::run(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    if (n_ == 0 || n == 1) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    fn_ = &fn;
    for (int offset = 0; offset < n;) {
        fn_offset_ = offset;
        const int count = std::min(n - offset, 0xffff);
        run_phase(7, count);
        offset += count;
    }
    fn_ = nullptr;
}

void ExpertPool::run_split(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplit || n_ == 1 || expert_oracle_q8_0_enabled()) { run(jobs, n); return; }
    const auto t0 = std::chrono::steady_clock::now();
    jobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    // about three tasks per thread in each phase, so the tail is short
    parts_a_ = (std::max)(1, (3 * threads + n - 1) / n);
    parts_b_ = parts_a_;
    run_phase(1, n * parts_a_);
    for (int e = 0; e < n; ++e) act_quant_q8_1(split_[(size_t) e].ff, FF, split_[(size_t) e].a2);
    run_phase(2, n * parts_b_);
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int ExpertPool::phase_tasks(int64_t rows) const {
    return tasks_ == 0 ? 3 * (n_ + (host_works_ ? 1 : 0)) : (int) (std::min)(rows, (int64_t) tasks_);
}

void ExpertPool::run_split_multi(ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplitMulti || expert_oracle_q8_0_enabled()) {
        // one token at a time through the single-token path (the oracle contract has no multi kernel)
        std::vector<ExpertJob> single;
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[e].nt; ++t) {
                ExpertJob j;
                j.blob = jobs[e].blob;
                j.act = jobs[e].act[t];
                j.out = jobs[e].out[t];
                single.push_back(j);
            }
        for (size_t i = 0; i < single.size(); i += kMaxSplit)
            run_split(single.data() + i, (int) (std::min)((size_t) kMaxSplit, single.size() - i));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    mjobs_ = jobs;
    mrows_ = (int64_t) n * FF;
    mtasks_ = phase_tasks(mrows_);
    run_phase(3, mtasks_);
    const auto t1 = std::chrono::steady_clock::now();
    for (int e = 0; e < n; ++e)
        for (int t = 0; t < jobs[e].nt; ++t)
            act_quant_q8_1(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
    const auto t2 = std::chrono::steady_clock::now();
    mrows_ = (int64_t) n * H;
    mtasks_ = phase_tasks(mrows_);
    run_phase(4, mtasks_);
    const auto t3 = std::chrono::steady_clock::now();
    ms_multi_gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms_multi_q += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ms_multi_down += std::chrono::duration<double, std::milli>(t3 - t2).count();
    multi_bytes += (int64_t) n * (int64_t) BLOB;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    // more distinct experts than buffers: run them in batches
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = (std::min)(kMaxSplitMulti, n - b0);
        mjobs_ = jobs + b0;
        nfmt_ = &f;
        mrows_ = (int64_t) nb * f.n_ff;
        mtasks_ = phase_tasks(mrows_);
        const auto a = std::chrono::steady_clock::now();
        run_phase(5, mtasks_);
        const auto b = std::chrono::steady_clock::now();
        const bool q2 = f.d_type == 42 && !canon_layer(f);
        for (int e = 0; e < nb; ++e)
            for (int t = 0; t < mjobs_[e].nt; ++t)
                if (q2) act_quant_any(split_multi_[(size_t) e].ff[t], (int) f.n_ff, split_multi_[(size_t) e].a2[t]);
                else native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
        const auto c = std::chrono::steady_clock::now();
        mrows_ = (int64_t) nb * f.n_embd;
        mtasks_ = phase_tasks(mrows_);
        run_phase(6, mtasks_);
        const auto d = std::chrono::steady_clock::now();
        ms_multi_gu += std::chrono::duration<double, std::milli>(b - a).count();
        ms_multi_q += std::chrono::duration<double, std::milli>(c - b).count();
        ms_multi_down += std::chrono::duration<double, std::milli>(d - c).count();
    }
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n_ == 1) {   // no workers: run inline, so a single-core machine still produces a token
        for (int i = 0; i < n; ++i) s2_expert_vnni_q(jobs[i].blob, *jobs[i].act, jobs[i].out, scratch_[0]);
        return;
    }
    // Wait for every worker to be parked BEFORE touching the batch, so the publish below is the only thing
    // that can move a worker into the drain loop.
    //
    // THE THREE PHASES ARE TIMED SEPARATELY.  They were one number, which cannot distinguish a pool that is
    // slow at the WORK from one that is slow at the SYNCHRONISATION - and those need opposite fixes.
    const auto t_a = std::chrono::steady_clock::now();
    wait_parked("before a batch");
    const auto t_b = std::chrono::steady_clock::now();
    jobs_ = jobs;
    njobs_ = n;
    mode_ = 0;
    const uint32_t e = begin_batch(n);   // done, then head (release), then the epoch: the batch is described first

    // ---- **THE HOST DRAINS TOO (R2.2), INSTEAD OF SPINNING ON `done_`.**
    //
    // The loop below used to be `while (done_ != n) _mm_pause();`.  The host is pinned to core 0 - the core
    // `physical_cores(true)` deliberately keeps the five workers off - so for the whole drain that core was
    // idle while five cores did six cores' worth of work.  Measured before the change: 33.7 GB/s against
    // 5/6 x 44.14 = 36.8 for five workers and 44.14 for six.
    //
    // The host claims through the SAME `head_` counter, so this is not a second scheduler and nothing about
    // the ordering changes: `head_` is a single `fetch_add`, every job is the same size, and a thread that
    // arrives late simply claims nothing.  `done_` is still the completion signal and the host still waits for
    // it - what changed is only that the host arrives at that wait having done a share of the work.
    //
    // The host's `done_.fetch_add` is a release for the same reason a worker's is: `j.out` is read by the
    // device after `run()` returns, so the write must be published, not merely performed.
    if (host_works_) {
        for (;;) {
            const int ci = claim(e);
            if (ci < 0) break;
            hstate_.store(ci, std::memory_order_relaxed);
            hstate_ms_.store(now_ms(), std::memory_order_relaxed);
            const ExpertJob& j = jobs_[ci];
            s2_expert_vnni_q(j.blob, *j.act, j.out, host_scratch_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    wait_done(n);
    // And park again, so the next `run` starts from a known state.  See the header for why `done` alone is
    // not enough.
    const auto t_c = std::chrono::steady_clock::now();
    wait_parked("after a batch");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    const auto t_d = std::chrono::steady_clock::now();

    ms_wait_park_ += std::chrono::duration<double, std::milli>(t_b - t_a).count();
    ms_drain_ += std::chrono::duration<double, std::milli>(t_c - t_b).count();
    ms_repark_ += std::chrono::duration<double, std::milli>(t_d - t_c).count();
}

}  // namespace strata::kernels::cpu

#if defined(_WIN32)
// strata/platform/thread_name.hpp: the Windows half lives here, where <windows.h> is already included (upstream
// 8c41d3611).  SetThreadDescription (Windows 10 1607 and later) is looked up, not linked: older SDKs and toolchains lack
// the import.
namespace strata::platform {
void name_this_thread(const char* base, int index) {
    char name[64];
    if (index >= 0) std::snprintf(name, sizeof name, "%s-%d", base, index);
    else std::snprintf(name, sizeof name, "%s", base);
    using SetDesc = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    static const SetDesc fn = (SetDesc) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription");
    if (fn == nullptr) return;
    wchar_t wide[64];
    int i = 0;
    for (; name[i] != '\0' && i < 63; ++i) wide[i] = (wchar_t) (unsigned char) name[i];
    wide[i] = L'\0';
    (void) fn(GetCurrentThread(), wide);
}
}  // namespace strata::platform
#endif
