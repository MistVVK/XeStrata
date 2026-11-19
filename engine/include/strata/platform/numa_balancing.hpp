// include/strata/platform/numa_balancing.hpp - keep the kernel's automatic NUMA balancing off the engine's memory.
//
// Linux, AMD (HIP): the expert arena and the other host ranges the GPU reads are registered with hipHostRegister,
// which the amdgpu driver keeps as a KFD "userptr" range.  With kernel.numa_balancing=1 the kernel's NUMA scanner
// periodically marks the PTEs of every migratable range PROT_NONE to sample which node touches it; each such change is
// an MMU-notifier invalidation, and KFD answers an invalidation of a userptr range by evicting ALL of the process's
// GPU queues until the range is validated again.  Work already submitted stops, nothing errors, and a prompt chunk or a
// decode window waits for seconds to minutes: the "no progress for 60 s" stops.  Measured on a Radeon PRO W7900D
// (gfx1100) in a two-socket host, Qwen3.8-Flash-Next IQ3_S, 20 fresh 1K prompts per arm: 7 watchdog stops and 77-236 s
// of queue eviction per engine start (the driver's own /sys/class/kfd/kfd/proc/<pid>/stats_<gpu>/evicted_ms) with the
// default policy; 0 stops and under 1 s of eviction with this change, decode unchanged.
//
// The scanner skips memory whose policy is an explicit one without MPOL_F_NUMA_BALANCING (vma_policy_mof() is false).
// MPOL_LOCAL is that: pages still go to the node of the CPU that first touches them (what the default policy does, and
// what the NUMA pool's first-touch placement relies on), but the kernel stops scanning, and so stops invalidating, the
// engine's memory.  It is set once, in main() before any thread is started (the policy is per thread: see
// thread_count) and before anything is allocated; a policy the user set
// (numactl --membind / --interleave, ...) is kept.  STRATA_NUMA_BALANCING=keep leaves the default policy in place.
#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__linux__)
#include <dirent.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::platform::numa_balancing {

constexpr int kMpolDefault = 0;
constexpr int kMpolLocal = 4;

/// Whether to set MPOL_LOCAL.  balancing: /proc/sys/kernel/numa_balancing (-1: unreadable); current: the process's
/// policy mode (get_mempolicy; -1: unknown); keep: STRATA_NUMA_BALANCING=keep.
constexpr bool want_local(long balancing, int current, bool keep) {
    return !keep && balancing > 0 && current == kMpolDefault;
}

/// Threads of this process now (/proc/self/task entries; 0 when it cannot be read).  set_mempolicy sets the policy of
/// the calling thread only, and the NUMA scanner runs as whichever thread of the process it interrupts, judging every
/// VMA by THAT thread's policy: a thread that existed before the call keeps the default policy and keeps the scanner,
/// and the evictions, going.  So the call has to come before the first thread is made.
inline int thread_count() {
#if defined(__linux__)
    int n = 0;
    if (DIR* d = opendir("/proc/self/task")) {
        while (const dirent* e = readdir(d))
            if (e->d_name[0] != '.') ++n;
        closedir(d);
    }
    return n;
#else
    return 0;
#endif
}

inline long read_balancing() {
    long v = -1;
    if (std::FILE* f = std::fopen("/proc/sys/kernel/numa_balancing", "r")) {
        if (std::fscanf(f, "%ld", &v) != 1) v = -1;
        std::fclose(f);
    }
    return v;
}

/// Called first thing in main(): sets MPOL_LOCAL when the kernel balances NUMA and nobody chose a policy.  Returns the
/// line to log, empty when nothing was done.
inline std::string keep_scanner_off() {
#if defined(__linux__) && defined(SYS_set_mempolicy) && defined(SYS_get_mempolicy)
    const char* e = std::getenv("STRATA_NUMA_BALANCING");
    const bool keep = e != nullptr && std::strcmp(e, "keep") == 0;
    const long balancing = read_balancing();
    int mode = -1;
    if (syscall(SYS_get_mempolicy, &mode, nullptr, 0ul, nullptr, 0ul) != 0) mode = -1;
    if (!want_local(balancing, mode, keep)) return "";
    if (syscall(SYS_set_mempolicy, kMpolLocal, nullptr, 0ul) != 0)
        return std::string("strata: numa: kernel.numa_balancing is on and MPOL_LOCAL could not be set (") +
               std::strerror(errno) + "): GPU queues can stall while the kernel scans host memory";
    std::string line = "strata: numa: kernel.numa_balancing is on: this process uses MPOL_LOCAL (the same placement, but "
                       "the kernel no longer scans the memory the GPU reads, which stalled the GPU queues; "
                       "STRATA_NUMA_BALANCING=keep: off)";
    if (const int n = thread_count(); n > 1)
        line += "\nstrata: numa: WARNING: " + std::to_string(n - 1) +
                " thread(s) already ran before the policy was set and keep the default one: the kernel can still scan "
                "through them (start the engine's threads after this call)";
    return line;
#else
    return "";
#endif
}

}  // namespace strata::platform::numa_balancing
