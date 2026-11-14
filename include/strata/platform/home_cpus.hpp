// strata/platform/home_cpus.hpp - the widest CPU set a thread of the engine may take.
//
// A few threads first widen their own CPU mask before they work: the background copy threads (STRATA_CKPT_ASYNC,
// STRATA_PARK_ASYNC) and the checkpoint helpers start on the host's single CPU, which they inherited, and must not take
// turns with the decode there.  They used to widen to every CPU number; the kernel then keeps the CPUs the cgroup allows,
// so a thread left a CPU set the person gave the process (taskset, numactl) or the one the engine gave itself on the
// card's NUMA node (strata/kernels/cpu/card_node.hpp).  widen_to_home() widens to the home CPUs, which generate.cpp sets
// once at start-up, before these threads exist, to the main thread's CPU set after the card-node step (the card's node, or
// the set it started with on one node or when the rule declined); not set (STRATA_NUMA_CARD=0, unreadable set): every CPU.
//
// Linux only (the other systems get an empty function).  Header-only: the callers sit in libraries that do not link each other.
#pragma once

#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace strata::platform {

#if defined(__linux__)
namespace home_cpus_detail {
struct Home {
    bool set = false;
    cpu_set_t mask;
};
inline Home& home() {
    static Home h;
    return h;
}
}  // namespace home_cpus_detail

/// The CPUs widen_to_home() gives a thread from now on.  Call once at start-up, before the threads that widen exist.
/// An empty list (or none of them below CPU_SETSIZE) changes nothing.
inline void set_home_cpus(const std::vector<int>& cpus) {
    cpu_set_t m;
    CPU_ZERO(&m);
    int n = 0;
    for (int c : cpus)
        if (c >= 0 && c < CPU_SETSIZE) { CPU_SET(c, &m); ++n; }
    if (n == 0) return;
    home_cpus_detail::home().mask = m;
    home_cpus_detail::home().set = true;
}

/// The mask widen_to_home() uses: the home CPUs, or every CPU number when none were set.
inline cpu_set_t home_mask() {
    if (home_cpus_detail::home().set) return home_cpus_detail::home().mask;
    cpu_set_t all;
    CPU_ZERO(&all);
    for (int c = 0; c < CPU_SETSIZE; ++c) CPU_SET(c, &all);
    return all;
}

/// Widens the calling thread to the home CPUs (see the top of this file).  The kernel keeps the CPUs the cgroup allows.
inline void widen_to_home() {
    const cpu_set_t m = home_mask();
    (void) sched_setaffinity(0, sizeof m, &m);
}
#else
inline void set_home_cpus(const std::vector<int>&) {}
inline void widen_to_home() {}
#endif

}  // namespace strata::platform
