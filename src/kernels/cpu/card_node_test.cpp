// card_node_test: when the engine restricts itself to the graphics card's NUMA node (include/strata/kernels/cpu/card_node.hpp).
// The decision on made-up machines, the sysfs / proc readers on a fake tree in a temp directory, and (Linux) the restriction
// of a two-thread process to a subset of its own CPUs.  No GPU, no model.
#include "strata/kernels/cpu/card_node.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace fs = std::filesystem;

static int g_fail = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

// A two-socket machine, one node per socket, 2-way SMT: node k has cores [0, cores_k), CPU numbers as Linux gives them
// (node 0's first threads, node 1's first threads, then node 0's siblings, node 1's siblings).
static std::vector<cpu::CardNodeCpu> machine(int cores0, int cores1) {
    std::vector<cpu::CardNodeCpu> m;
    int n = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (int c = 0; c < cores0; ++c) m.push_back({n++, 0, 0, c});
        for (int c = 0; c < cores1; ++c) m.push_back({n++, 1, 1, c});
    }
    return m;
}

static std::vector<int> cpus_of(const std::vector<cpu::CardNodeCpu>& m, int node) {
    std::vector<int> out;
    for (const auto& c : m)
        if (c.node == node) out.push_back(c.cpu);
    return out;
}

// The RTX 4090 pod's shape: 2 x 32 cores, 128 logical, a 13.6-CPU quota (12 workers), card on node 1, plenty of free memory.
static cpu::CardNodeInput pod() {
    cpu::CardNodeInput in;
    in.nodes_with_cpus = 2;
    in.mask = machine(32, 32);
    in.mems_known = true;
    in.mems_allowed = {0, 1};
    in.card_nodes = {1};
    in.card_node_free_bytes = 100ll << 30;
    in.pinned_need_bytes = cpu::card_node_pinned_need(131072);
    in.cpu_quota = 13.6;
    in.today_workers = 12;
    return in;
}

static void decisions() {
    // two nodes, the card on node 1: restricted to node 1's 64 logical CPUs, the pool keeps the quota's 12 workers
    {
        const auto d = cpu::decide_card_node(pod());
        expect(d.on && d.why.empty(), "pod: restricted (" + d.why + ")");
        expect(d.node == 1, "pod: node 1");
        expect(d.cpus == cpus_of(machine(32, 32), 1) && d.cpus.size() == 64, "pod: node 1's 64 CPUs");
        expect(d.node_cores == 32 && d.node_workers == 12, "pod: 32 cores, 12 workers on the node (quota)");
    }
    // the card on node 0
    {
        auto in = pod();
        in.card_nodes = {0};
        const auto d = cpu::decide_card_node(in);
        expect(d.on && d.node == 0 && d.cpus == cpus_of(in.mask, 0), "card on node 0: node 0's CPUs");
    }
    // one node: nothing
    {
        auto in = pod();
        in.nodes_with_cpus = 1;
        const auto d = cpu::decide_card_node(in);
        expect(!d.on && d.why == "one NUMA node" && d.cpus.empty(), "one node: left as is");
    }
    // the switch off
    {
        auto in = pod();
        in.switch_off = true;
        const auto d = cpu::decide_card_node(in);
        expect(!d.on && d.why == "STRATA_NUMA_CARD=0", "switch off: left as is");
    }
    // the card's node not known (sysfs -1 or unreadable)
    {
        auto in = pod();
        in.card_nodes = {-1};
        expect(!cpu::decide_card_node(in).on, "card node -1: left as is");
        in.card_nodes = {};
        expect(!cpu::decide_card_node(in).on, "no card: left as is");
    }
    // two cards on different nodes; two on the same node
    {
        auto in = pod();
        in.card_nodes = {1, 0};
        const auto d = cpu::decide_card_node(in);
        expect(!d.on && d.why == "the cards sit on different NUMA nodes", "cards on nodes 1 and 0: left as is");
        in.card_nodes = {1, 1};
        expect(cpu::decide_card_node(in).on, "two cards on node 1: restricted");
        in.card_nodes = {1, -1};
        expect(!cpu::decide_card_node(in).on, "one card's node unknown: left as is");
    }
    // a process already restricted to one node (taskset / numactl / cpuset): left as is, also when it is the other node
    {
        auto in = pod();
        std::vector<cpu::CardNodeCpu> one;
        for (const auto& c : in.mask)
            if (c.node == 0) one.push_back(c);
        in.mask = one;
        const auto d = cpu::decide_card_node(in);
        expect(!d.on && d.why == "the process already runs on one NUMA node", "pre-restricted to node 0: left as is");
        std::vector<cpu::CardNodeCpu> part;   // a subset that still spans both nodes (a cpuset over both): the rule applies
        for (const auto& c : pod().mask)
            if (c.cpu % 4 == 0) part.push_back(c);
        in.mask = part;
        in.today_workers = 7;   // that mask's own pool: 16 cores in all, the host's socket's 8 -> 7 workers
        const auto e = cpu::decide_card_node(in);
        expect(e.on && e.cpus.size() == 16 && e.node_workers == 7, "a mask over both nodes, 16 of node 1's CPUs: restricted to those 16");
        in.today_workers = 12;
        expect(!cpu::decide_card_node(in).on, "the same mask with 12 workers today: 7 on the node, left as is");
    }
    // a CPU whose node is not known
    {
        auto in = pod();
        in.mask[5].node = -1;
        expect(!cpu::decide_card_node(in).on, "a CPU without a node: left as is");
    }
    // a node with too few CPUs for the pool: 4 + 60 cores, no quota, today's pool spans both sockets: 32 workers
    {
        cpu::CardNodeInput in;
        in.nodes_with_cpus = 2;
        in.mask = machine(4, 60);
        in.card_nodes = {0};
        in.card_node_free_bytes = 100ll << 30;
        in.today_workers = 32;
        const auto d = cpu::decide_card_node(in);
        expect(!d.on && d.node_workers == 3 && d.cpus.empty(), "card on a 4-core node, 32 workers today: left as is (" + d.why + ")");
        in.card_nodes = {1};
        const auto e = cpu::decide_card_node(in);
        expect(e.on && e.node_workers == 32, "card on the 60-core node: restricted, 32 workers");
        // a 16-core node against 32 spread: 16 pinned only matched 32 spread in the measurement, so the count must not drop
        in.mask = machine(16, 16);
        in.card_nodes = {1};
        in.today_workers = 30;
        expect(!cpu::decide_card_node(in).on, "2 x 16 cores, 30 workers today, 15 on a node: left as is");
    }
    // --pool-workers N: the node needs N + 1 physical cores
    {
        auto in = pod();
        in.user_workers = in.today_workers = 31;
        expect(cpu::decide_card_node(in).on, "--pool-workers 31 on a 32-core node: restricted");
        in.user_workers = in.today_workers = 32;
        expect(!cpu::decide_card_node(in).on, "--pool-workers 32 on a 32-core node: left as is");
        in.user_workers = in.today_workers = 4;
        expect(cpu::decide_card_node(in).on, "--pool-workers 4: restricted");
    }
    // a container quota smaller than the node: the pool is bounded by the quota either way and the rule does not move it
    {
        cpu::CardNodeInput in;
        in.nodes_with_cpus = 2;
        in.mask = machine(96, 96);   // the second RTX 5090 host: 2 x EPYC 9654, card on node 0, a 46-CPU quota
        in.card_nodes = {0};
        in.card_node_free_bytes = 100ll << 30;
        in.cpu_quota = 46;
        in.today_workers = 32;
        const auto d = cpu::decide_card_node(in);
        expect(d.on && d.node_workers == 32 && d.cpus.size() == 192, "quota 46 on 2 x 96 cores: restricted, 32 workers");
        in.cpu_quota = 17.55;   // the first RTX 5090 host's quota: 16 workers either way
        in.today_workers = 16;
        const auto e = cpu::decide_card_node(in);
        expect(e.on && e.node_workers == 16, "quota 17.55: restricted, 16 workers");
        // a quota larger than one node (2 x 8 cores, 30 CPUs): today 15 workers, a node 7: left as is
        in.mask = machine(8, 8);
        in.cpu_quota = 30;
        in.today_workers = 15;
        expect(!cpu::decide_card_node(in).on, "quota above one node's CPUs: left as is");
    }
    // memory: the process may not allocate on the card's node, or the node is short of free memory
    {
        auto in = pod();
        in.mems_allowed = {0};
        const auto d = cpu::decide_card_node(in);
        expect(!d.on && d.why == "the process may not allocate memory on node 1", "mems_allowed without the card's node: left as is");
        in.mems_known = false;   // not readable: not a reason to stop
        expect(cpu::decide_card_node(in).on, "mems_allowed not known: restricted");
        in = pod();
        in.card_node_free_bytes = (int64_t) (2.6 * 1073741824.0);   // the 2.6 GB free seen on one node of the first 5090 host
        const auto e = cpu::decide_card_node(in);
        expect(!e.on && e.why.find("MiB free, under the") != std::string::npos, "2.6 GiB free on the card's node: left as is (" + e.why + ")");
        in.card_node_free_bytes = -1;
        expect(!cpu::decide_card_node(in).on, "free memory not known: left as is");
        in.card_node_free_bytes = (int64_t) in.pinned_need_bytes;
        expect(cpu::decide_card_node(in).on, "free memory exactly the need: restricted");
        expect(cpu::card_node_pinned_need(131072) == 131072ull * 12700 + (2ull << 30), "pinned need at 131,072 cells");
        expect(cpu::card_node_pinned_need(0) == (2ull << 30) && cpu::card_node_pinned_need(-5) == (2ull << 30), "pinned need without a context");
    }
    // the person placed the threads; a hybrid CPU; the process has no CPU on the card's node
    {
        auto in = pod();
        in.user_placed = "--pool-cores is set";
        const auto d = cpu::decide_card_node(in);
        expect(!d.on && d.why == "--pool-cores is set", "--pool-cores: left as is");
        in = pod();
        in.hybrid = true;
        expect(!cpu::decide_card_node(in).on, "hybrid: left as is");
        in = pod();
        in.nodes_with_cpus = 3;
        in.card_nodes = {2};
        in.mems_known = false;
        const auto e = cpu::decide_card_node(in);
        expect(!e.on && e.why == "the process has no CPU on node 2", "card on a node the process has no CPU on: left as is");
    }
}

static void write(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

static void readers() {
    const fs::path root = fs::temp_directory_path() / ("card_node_test_" + std::to_string((long long) std::time(nullptr)));
    fs::remove_all(root);
    const fs::path node = root / "devices/system/node";
    write(node / "node0/cpulist", "0-1,4-5\n");
    write(node / "node1/cpulist", "2-3,6-7\n");
    write(node / "node2/cpulist", "\n");   // a memory-only node (CXL): not counted
    write(node / "online", "0-2\n");
    fs::create_directories(node / "nodeX");
    write(node / "node1/meminfo", "Node 1 MemTotal:       264000000 kB\nNode 1 MemFree:        112233 kB\nNode 1 MemUsed: 1 kB\n");
    write(node / "node0/meminfo", "Node 0 MemTotal: 5 kB\n");   // no MemFree line
    for (int c = 0; c < 8; ++c) {
        const fs::path t = root / ("devices/system/cpu/cpu" + std::to_string(c)) / "topology";
        write(t / "physical_package_id", std::to_string((c / 2) % 2) + "\n");
        write(t / "core_id", std::to_string(c % 2) + "\n");
    }
#if !defined(_WIN32)
    // (a PCI address has colons, which no Windows file name may have: create_directories throws there.  The engine reads sysfs on Linux only.)
    write(root / "bus/pci/devices/0000:e1:00.0/numa_node", "1\n");
    write(root / "bus/pci/devices/0000:21:00.0/numa_node", "-1\n");
    write(root / "bus/pci/devices/0000:22:00.0/numa_node", "junk\n");
#endif

#if defined(__linux__)
    const auto nodes = cpu::card_node_node_cpus(root.string());
    expect(nodes.size() == 2, "fake sysfs: 2 nodes with CPUs (memory-only node2 and nodeX left out)");
    expect(nodes.count(0) && nodes.at(0) == std::vector<int>({0, 1, 4, 5}), "fake sysfs: node 0's CPUs");
    expect(nodes.count(1) && nodes.at(1) == std::vector<int>({2, 3, 6, 7}), "fake sysfs: node 1's CPUs");
    const auto desc = cpu::card_node_describe(root.string(), {0, 3, 6, 9}, nodes);
    expect(desc.size() == 4 && desc[0].node == 0 && desc[1].node == 1 && desc[2].node == 1 && desc[3].node == -1, "describe: nodes");
    expect(desc[1].pkg == 1 && desc[1].core == 1 && desc[3].pkg == -1 && desc[3].core == -1, "describe: package and core, missing CPU -1");
    expect(cpu::card_node_node_cpus((root / "nothing").string()).empty(), "no node folder: no nodes");
#endif
#if !defined(_WIN32)
    expect(cpu::card_node_pci_node(root.string(), "0000:e1:00.0") == 1, "pci numa_node 1");
    expect(cpu::card_node_pci_node(root.string(), "0000:21:00.0") == -1, "pci numa_node -1");
    expect(cpu::card_node_pci_node(root.string(), "0000:22:00.0") == -1, "pci numa_node junk: -1");
    expect(cpu::card_node_pci_node(root.string(), "0000:99:00.0") == -1, "pci device missing: -1");
#endif
    expect(cpu::card_node_free_bytes(root.string(), 1) == 112233ll * 1024, "node 1 MemFree");
    expect(cpu::card_node_free_bytes(root.string(), 0) == -1, "node 0 without MemFree: -1");
    expect(cpu::card_node_free_bytes(root.string(), 5) == -1, "node 5 missing: -1");
    expect(cpu::card_node_free_bytes(root.string(), -1) == -1, "node -1: -1");
    expect(cpu::card_node_bdf(0, 0xE1, 0) == "0000:e1:00.0", "bdf 0000:e1:00.0");
    expect(cpu::card_node_bdf(0x10, 0x3b, 0x1f, 1) == "0010:3b:1f.1", "bdf 0010:3b:1f.1");
    fs::remove_all(root);

    std::vector<int> v;
    expect(cpu::card_node_parse_cpulist("0-31,64-95\n", v) && v.size() == 64 && v.front() == 0 && v.back() == 95, "cpulist 0-31,64-95");
    expect(cpu::card_node_parse_cpulist(" 3 ", v) && v == std::vector<int>({3}), "cpulist 3");
    expect(cpu::card_node_parse_cpulist("\n", v) && v.empty(), "cpulist empty");
    expect(!cpu::card_node_parse_cpulist("3-1", v) && v.empty(), "cpulist reversed: refused");
    expect(!cpu::card_node_parse_cpulist("1,,2", v), "cpulist empty item: refused");
    expect(!cpu::card_node_parse_cpulist("a", v), "cpulist letters: refused");
    std::vector<int> mems;
    expect(cpu::card_node_mems_allowed("Name:\tstrata\nMems_allowed:\t00000003\nMems_allowed_list:\t0-1\nVoluntary: 1\n", mems) &&
               mems == std::vector<int>({0, 1}), "Mems_allowed_list 0-1");
    expect(cpu::card_node_mems_allowed("Mems_allowed_list:\t1\n", mems) && mems == std::vector<int>({1}), "Mems_allowed_list 1");
    expect(!cpu::card_node_mems_allowed("Name:\tx\n", mems) && mems.empty(), "no Mems_allowed_list line");
    for (const char* off : {"0", "off", "OFF", "false", "no"}) expect(cpu::card_node_switch_off(off), std::string("switch off: ") + off);
    for (const char* on : {"1", "", "on", "yes"}) expect(!cpu::card_node_switch_off(on), std::string("switch not off: '") + on + "'");
    expect(!cpu::card_node_switch_off(nullptr), "switch unset: not off");
}

#if defined(__linux__)
// restrict_process_cpus on this process: a second thread exists before the call and must be restricted too
static void restriction() {
    const std::vector<int> mine = cpu::card_node_thread_cpus();
    if (mine.size() < 2) {
        std::printf("card_node_test: restriction skipped (this process may run on %zu CPU)\n", mine.size());
        return;
    }
    std::atomic<bool> stop{false};
    std::atomic<int> other_tid{0};
    std::vector<int> other_after;
    std::thread t([&] {
        other_tid = (int) syscall(SYS_gettid);
        while (!stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        other_after = cpu::card_node_thread_cpus();
    });
    while (other_tid.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const std::vector<int> half(mine.begin(), mine.begin() + (long) (mine.size() / 2));
    std::vector<int> want = half;
    want.push_back(100000);   // a CPU the process does not have is ignored
    int refused = -1;
    const int changed = cpu::restrict_process_cpus(want, refused);
    expect(changed >= 2 && refused == 0, "restriction: both threads changed (" + std::to_string(changed) + ", refused " +
                                             std::to_string(refused) + ")");
    expect(cpu::card_node_thread_cpus() == half, "restriction: the calling thread has the subset");
    std::vector<int> created_after;
    std::thread n([&] { created_after = cpu::card_node_thread_cpus(); });
    n.join();
    expect(created_after == half, "restriction: a thread created afterwards inherits the subset");
    stop = true;
    t.join();
    expect(other_after == half, "restriction: the thread that existed before has the subset");
    int refused2 = -1;
    expect(cpu::restrict_process_cpus(half, refused2) == 0 && refused2 == 0, "restriction: a second call changes nothing");
    std::vector<int> none = {100000};
    expect(cpu::restrict_process_cpus(none, refused2) == 0 && cpu::card_node_thread_cpus() == half,
           "restriction: no common CPU keeps the mask");
}
#endif

int main() {
    decisions();
    readers();
#if defined(__linux__)
    restriction();
#endif
    if (g_fail != 0) {
        std::fprintf(stderr, "card_node_test: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("card_node_test: all checks passed\n");
    return 0;
}
