// include/strata/kernels/cpu/card_node.hpp - run the engine on the graphics card's NUMA node (Linux, CUDA and HIP).
//
// On a machine with two memory nodes the card sits on one of them.  Left alone, the scheduler put a start's threads on
// either node, and the CPU expert pool spread over both.  On a two-socket RTX 5090 host, decode with CPU experts read
// 1.11-1.18 of the two-node default over three rounds with a 32-worker pool on one node (UD 4-bit set).  So at start-up,
// before any thread exists and before the first large allocation, the engine restricts itself to the card's node when
// that is safe, which is what `taskset -c <the node's CPUs>` did by hand.
//
// Pure decision first (decide_card_node, no system calls, so a test can feed it a made-up machine), then the readers of
// sysfs and /proc, each with a root directory a test can point at a fake tree.  The caller (generate.cpp) gathers the
// inputs, asks, and applies the answer with restrict_process_cpus().  STRATA_NUMA_CARD=0 (or off / false / no) is the
// way back: the engine then never restricts itself.  In every case the decision does not take, nothing changes.
#pragma once

#include "strata/kernels/cpu/pool_plan.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::kernels::cpu {

/// One logical CPU the process may run on: its NUMA node, CPU package and core (-1: not known).
struct CardNodeCpu {
    int cpu = -1;
    int node = -1;
    long pkg = -1;
    long core = -1;
};

/// Pinned host memory the engine allocates once the threads sit on the card's node, as far as it is known before the
/// model is opened: the host copy of the K/V cache (KV streaming) and the staging buffers.  The K/V copy of this model
/// geometry is about 12,700 bytes per context cell (measured: 1.55 GiB at 131,072 cells on the RTX 4090 pod), counted
/// at the full --max-context even when part of it stays in VRAM; the staging buffers and the other pinned scratch are
/// small (well under 1 GiB) and get a fixed 2 GiB.  The expert arena is NOT counted: this rule does not place it (it
/// lands where first touch and the kernel's fallback put it; on the pod it stayed on node 0 with every thread on node 1).
inline constexpr uint64_t kCardNodeKvBytesPerCell = 12700;
inline constexpr uint64_t kCardNodeStagingBytes = 2ull << 30;
inline uint64_t card_node_pinned_need(int64_t max_context) {
    return (max_context > 0 ? (uint64_t) max_context : 0) * kCardNodeKvBytesPerCell + kCardNodeStagingBytes;
}

struct CardNodeInput {
    bool switch_off = false;          ///< STRATA_NUMA_CARD=0 / off / false / no
    std::string user_placed;          ///< non-empty: the person placed the threads (--pool-cores, STRATA_NUMA_SPLIT, an --aux-cpus list): why
    int nodes_with_cpus = 1;          ///< NUMA nodes of the machine that have CPUs
    std::vector<CardNodeCpu> mask;    ///< the CPUs the process may run on now
    bool mems_known = false;          ///< mems_allowed was read (/proc/self/status Mems_allowed_list)
    std::vector<int> mems_allowed;    ///< the memory nodes the process may allocate from
    std::vector<int> card_nodes;      ///< the node of each card the run uses (-1: not known)
    int64_t card_node_free_bytes = -1;   ///< MemFree of the card's node (-1: not known)
    uint64_t pinned_need_bytes = 0;   ///< card_node_pinned_need(--max-context)
    double cpu_quota = 0;             ///< CPUs the container's quota allows (0: none)
    bool hybrid = false;              ///< a CPU with performance and efficiency cores
    int today_workers = 0;            ///< the pool's worker count without this rule (the default, or --pool-workers)
    int user_workers = 0;             ///< --pool-workers (0: the default)
};

struct CardNodeDecision {
    bool on = false;                  ///< restrict the process to `cpus`
    int node = -1;                    ///< the card's node (when known)
    std::vector<int> cpus;            ///< the process's CPUs on that node, ascending
    int node_cores = 0;               ///< physical cores among `cpus`
    int node_workers = 0;             ///< the pool's default worker count on `cpus` alone
    std::string why;                  ///< why the process is left as it is ("" when restricted)
};

/// The process's CPUs on one node (ascending), and the pool's plan input for them.
inline PoolPlanInput card_node_plan_input(const std::vector<CardNodeCpu>& cpus, double quota) {
    PoolPlanInput in;
    in.logical = (int) cpus.size();
    in.cpu_quota = quota;
    std::map<long, std::set<long>> cores;   // package -> its cores
    bool known = !cpus.empty();
    for (const CardNodeCpu& c : cpus) {
        if (c.pkg < 0 || c.core < 0) { known = false; break; }
        cores[c.pkg].insert(c.core);
    }
    if (known) {
        // the host thread's package first, as pool_plan_input orders it: the package of the lowest CPU number
        const auto lowest = std::min_element(cpus.begin(), cpus.end(),
                                             [](const CardNodeCpu& a, const CardNodeCpu& b) { return a.cpu < b.cpu; });
        const long first = lowest->pkg;
        in.cores_per_package.push_back((int) cores[first].size());
        for (const auto& kv : cores)
            if (kv.first != first) in.cores_per_package.push_back((int) kv.second.size());
    }
    return in;
}

/// Restrict the process to the card's node?  Yes only when all of these hold; otherwise `why` names the first that does not:
///   the switch is not off; the machine has two or more nodes with CPUs; the process's CPUs span more than one node (a
///   taskset / numactl / cpuset restriction to one node is left as it is); every CPU's node is known; the run uses at least
///   one card and every card reports the same, non-negative node; the process may allocate memory on that node; the node
///   has the pinned memory the engine allocates next (card_node_pinned_need); the person did not place the threads
///   themselves; the CPU is not hybrid; the node has CPUs of the process; and the pool keeps its size there: the pool's own
///   plan on the node's CPUs alone gives at least the worker count it would have without the rule (with --pool-workers N:
///   the node has N + 1 physical cores, one per worker and the host's).  The worker count must not drop: measured on an
///   RTX 5090 host, 32 workers on one node (either node, within 1-3 %) beat 32 spread over two nodes by 14-22 %, but 16
///   on one node only matched 32 spread and 8 on one node lost (0.84-0.93).  A container quota bounds both counts alike, so it never stops
///   the rule and the pool keeps the quota's count.
inline CardNodeDecision decide_card_node(const CardNodeInput& in) {
    CardNodeDecision d;
    if (in.switch_off) { d.why = "STRATA_NUMA_CARD=0"; return d; }
    if (in.nodes_with_cpus < 2) { d.why = "one NUMA node"; return d; }
    std::set<int> mask_nodes;
    for (const CardNodeCpu& c : in.mask) {
        if (c.node < 0) { d.why = "CPU " + std::to_string(c.cpu) + " has no known NUMA node"; return d; }
        mask_nodes.insert(c.node);
    }
    if (mask_nodes.size() < 2) {
        d.why = in.mask.empty() ? "the process's CPUs are not known" : "the process already runs on one NUMA node";
        return d;
    }
    if (in.card_nodes.empty()) { d.why = "no card"; return d; }
    const int node = in.card_nodes.front();
    for (int n : in.card_nodes) {
        if (n < 0) { d.why = "a card's NUMA node is not known"; return d; }
        if (n != node) { d.why = "the cards sit on different NUMA nodes"; return d; }
    }
    d.node = node;
    if (in.mems_known && std::find(in.mems_allowed.begin(), in.mems_allowed.end(), node) == in.mems_allowed.end()) {
        d.why = "the process may not allocate memory on node " + std::to_string(node);
        return d;
    }
    if (in.card_node_free_bytes < 0) { d.why = "node " + std::to_string(node) + "'s free memory is not known"; return d; }
    if ((uint64_t) in.card_node_free_bytes < in.pinned_need_bytes) {
        d.why = "node " + std::to_string(node) + " has " + std::to_string(in.card_node_free_bytes >> 20) + " MiB free, under the " +
                std::to_string(in.pinned_need_bytes >> 20) + " MiB of pinned memory the engine allocates next";
        return d;
    }
    if (!in.user_placed.empty()) { d.why = in.user_placed; return d; }
    if (in.hybrid) { d.why = "a hybrid CPU"; return d; }
    std::vector<CardNodeCpu> on;
    for (const CardNodeCpu& c : in.mask)
        if (c.node == node) on.push_back(c);
    std::sort(on.begin(), on.end(), [](const CardNodeCpu& a, const CardNodeCpu& b) { return a.cpu < b.cpu; });
    if (on.empty()) { d.why = "the process has no CPU on node " + std::to_string(node); return d; }
    const PoolPlan plan = plan_pool_workers(card_node_plan_input(on, in.cpu_quota));
    d.node_cores = plan.cores;
    d.node_workers = plan.workers;
    for (const CardNodeCpu& c : on) d.cpus.push_back(c.cpu);
    if (in.user_workers > 0) {
        if (in.user_workers + 1 > plan.cores) {
            d.why = "node " + std::to_string(node) + " has " + std::to_string(plan.cores) + " physical cores, too few for " +
                    std::to_string(in.user_workers) + " workers and the host";
            d.cpus.clear();
            return d;
        }
    } else if (plan.workers < in.today_workers) {
        d.why = "node " + std::to_string(node) + " fits " + std::to_string(plan.workers) + " pool workers, under the " +
                std::to_string(in.today_workers) + " the pool uses now";
        d.cpus.clear();
        return d;
    }
    d.on = true;
    return d;
}

// ---- readers (Linux sysfs and /proc; a root directory so a test can use a fake tree) --------------------------------------

/// "0-31,64-95\n" -> {0..31, 64..95}.  Whitespace around the text is ignored; false (and empty) for anything else.
inline bool card_node_parse_cpulist(std::string text, std::vector<int>& out) {
    out.clear();
    while (!text.empty() && std::isspace((unsigned char) text.back())) text.pop_back();
    size_t i = 0;
    while (i < text.size() && std::isspace((unsigned char) text[i])) ++i;
    if (i == text.size()) return true;   // an empty list (a node without CPUs)
    auto number = [&](long& v) {
        if (i >= text.size() || !std::isdigit((unsigned char) text[i])) return false;
        v = 0;
        while (i < text.size() && std::isdigit((unsigned char) text[i])) {
            v = v * 10 + (text[i] - '0');
            if (v > 1000000) return false;
            ++i;
        }
        return true;
    };
    for (;;) {
        long lo = 0, hi = 0;
        if (!number(lo)) { out.clear(); return false; }
        hi = lo;
        if (i < text.size() && text[i] == '-') {
            ++i;
            if (!number(hi) || hi < lo) { out.clear(); return false; }
        }
        for (long c = lo; c <= hi; ++c) out.push_back((int) c);
        if (i == text.size()) break;
        if (text[i] != ',') { out.clear(); return false; }
        ++i;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return true;
}

inline bool card_node_read_file(const std::string& path, std::string& text) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream s;
    s << f.rdbuf();
    text = s.str();
    return true;
}

/// node -> its CPUs, from <sys>/devices/system/node/node<N>/cpulist (nodes without CPUs are left out).
inline std::map<int, std::vector<int>> card_node_node_cpus(const std::string& sys_root) {
    std::map<int, std::vector<int>> nodes;
#if defined(__linux__)
    const std::string dir = sys_root + "/devices/system/node";
    if (DIR* dd = opendir(dir.c_str())) {
        while (const dirent* e = readdir(dd)) {
            const std::string name = e->d_name;
            if (name.size() < 5 || name.compare(0, 4, "node") != 0) continue;
            if (!std::all_of(name.begin() + 4, name.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) continue;
            std::string text;
            std::vector<int> cpus;
            if (!card_node_read_file(dir + "/" + name + "/cpulist", text) || !card_node_parse_cpulist(text, cpus) || cpus.empty())
                continue;
            nodes[std::atoi(name.c_str() + 4)] = cpus;
        }
        closedir(dd);
    }
#else
    (void) sys_root;
#endif
    return nodes;
}

/// The node, package and core of each CPU in `cpus` (node from `nodes`, the others from <sys>/devices/system/cpu/cpu<N>/topology).
inline std::vector<CardNodeCpu> card_node_describe(const std::string& sys_root, const std::vector<int>& cpus,
                                                   const std::map<int, std::vector<int>>& nodes) {
    auto read_id = [&](int cpu, const char* what) -> long {
        std::string text;
        if (!card_node_read_file(sys_root + "/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/" + what, text)) return -1;
        char* end = nullptr;
        const long v = std::strtol(text.c_str(), &end, 10);
        return end != text.c_str() && v >= 0 ? v : -1;
    };
    std::vector<CardNodeCpu> out;
    for (int c : cpus) {
        CardNodeCpu x;
        x.cpu = c;
        for (const auto& kv : nodes)
            if (std::binary_search(kv.second.begin(), kv.second.end(), c)) { x.node = kv.first; break; }
        x.pkg = read_id(c, "physical_package_id");
        x.core = read_id(c, "core_id");
        out.push_back(x);
    }
    return out;
}

/// A card's PCI address in sysfs's spelling ("0000:e1:00.0") from the runtime's domain / bus / device numbers.
inline std::string card_node_bdf(unsigned domain, unsigned bus, unsigned device, unsigned function = 0) {
    char s[32];
    std::snprintf(s, sizeof s, "%04x:%02x:%02x.%x", domain & 0xffffu, bus & 0xffu, device & 0x1fu, function & 7u);
    return s;
}

/// <sys>/bus/pci/devices/<bdf>/numa_node: the card's node, -1 when the file is missing, unreadable or says -1.
inline int card_node_pci_node(const std::string& sys_root, const std::string& bdf) {
    std::string text;
    if (!card_node_read_file(sys_root + "/bus/pci/devices/" + bdf + "/numa_node", text)) return -1;
    char* end = nullptr;
    const long v = std::strtol(text.c_str(), &end, 10);
    return end != text.c_str() && v >= 0 && v < 4096 ? (int) v : -1;
}

/// "Node 1 MemFree:  112233 kB" of <sys>/devices/system/node/node<N>/meminfo, in bytes; -1 when not readable.
inline int64_t card_node_free_bytes(const std::string& sys_root, int node) {
    std::string text;
    if (node < 0 || !card_node_read_file(sys_root + "/devices/system/node/node" + std::to_string(node) + "/meminfo", text)) return -1;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        const size_t at = line.find("MemFree:");
        if (at == std::string::npos) continue;
        char* end = nullptr;
        const char* p = line.c_str() + at + 8;
        const long long kb = std::strtoll(p, &end, 10);
        if (end == p || kb < 0) return -1;
        return (int64_t) kb * 1024;
    }
    return -1;
}

/// The Mems_allowed_list line of /proc/self/status text ("Mems_allowed_list:\t0-1").  False when it is not there.
inline bool card_node_mems_allowed(const std::string& status_text, std::vector<int>& mems) {
    std::istringstream in(status_text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, 18, "Mems_allowed_list:") != 0) continue;
        return card_node_parse_cpulist(line.substr(18), mems);
    }
    mems.clear();
    return false;
}

/// The switch: STRATA_NUMA_CARD set to 0 / off / false / no (any case).
inline bool card_node_switch_off(const char* v) {
    if (v == nullptr) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s == "0" || s == "off" || s == "false" || s == "no";
}

#if defined(__linux__)
/// Restricts every thread of the process that exists now to its own CPUs among `cpus` (the calling thread first; a
/// thread with none of them keeps its mask), as `taskset` before the start would have: threads created later inherit
/// the restriction from the thread that creates them.  Returns how many threads were changed; `failed` counts refusals.
inline int restrict_process_cpus(const std::vector<int>& cpus, int& failed) {
    failed = 0;
    auto apply = [&](int tid) -> int {
        cpu_set_t now, to;
        CPU_ZERO(&now);
        CPU_ZERO(&to);
        if (sched_getaffinity(tid, sizeof now, &now) != 0) { ++failed; return 0; }
        int n = 0;
        for (int c : cpus)
            if (c >= 0 && c < CPU_SETSIZE && CPU_ISSET(c, &now)) { CPU_SET(c, &to); ++n; }
        if (n == 0 || CPU_EQUAL(&now, &to)) return 0;
        if (sched_setaffinity(tid, sizeof to, &to) != 0) { ++failed; return 0; }
        return 1;
    };
    const int self = (int) syscall(SYS_gettid);
    int changed = apply(self);
    if (DIR* d = opendir("/proc/self/task")) {
        while (const dirent* e = readdir(d)) {
            if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
            const int tid = std::atoi(e->d_name);
            if (tid != self) changed += apply(tid);
        }
        closedir(d);
    }
    return changed;
}

/// The CPUs the calling thread may run on now (ascending).
inline std::vector<int> card_node_thread_cpus() {
    std::vector<int> out;
    cpu_set_t now;
    CPU_ZERO(&now);
    if (sched_getaffinity(0, sizeof now, &now) == 0)
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &now)) out.push_back(c);
    return out;
}
#endif

}  // namespace strata::kernels::cpu
