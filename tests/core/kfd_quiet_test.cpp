// keeping the KFD quiet on an integrated gfx11 GPU under Linux (include/strata/core/kfd_quiet.hpp).  No GPU: the switch rule and the
// load order's rule.  (The device rule itself is igpu_kfd_settle_default, checked in src/core/arch_defaults_test.cpp.)
#include "strata/core/kfd_quiet.hpp"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#endif

using namespace strata::core::kfd_quiet;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

void switches() {
    CHECK(switch_on(nullptr, nullptr));
    CHECK(switch_on("1", nullptr));
    CHECK(switch_on("", ""));            // empty = unset
    CHECK(!switch_on("0", nullptr));     // the kill switch turns everything off
    CHECK(!switch_on("0", "1"));         // ... also when the fix's own switch says 1
    CHECK(!switch_on(nullptr, "0"));     // one fix off
    CHECK(!switch_on("1", "0"));
    CHECK(switch_on("yes", "on"));
    CHECK(std::string(fix_env(Fix::Load)) == "STRATA_KFD_QUIET_LOAD");
    CHECK(std::string(fix_env(Fix::Exit)) == "STRATA_KFD_QUIET_EXIT");
    CHECK(std::string(fix_env(Fix::Xfer)) == "STRATA_KFD_QUIET_XFER");
}

void load_order() {
    // not the class: none of the three moves, whatever the switches say (every other card keeps the old order)
    for (const char* all : {(const char*) nullptr, "1", "0"})
        for (const char* one : {(const char*) nullptr, "1", "0"}) {
            const LoadMoves m = load_moves(false, all, one);
            CHECK(!m.hc_check_early && !m.settle_after_cache && !m.first_prompt_buffers);
        }
    // the class, nothing set: all three
    {
        const LoadMoves m = load_moves(true, nullptr, nullptr);
        CHECK(m.hc_check_early && m.settle_after_cache && m.first_prompt_buffers);
    }
    // the class, STRATA_KFD_QUIET=0 or STRATA_KFD_QUIET_LOAD=0: the old order (the A/B arm of the same binary)
    using P = std::pair<const char*, const char*>;
    for (const P& p : std::vector<P>{P{"0", nullptr}, P{nullptr, "0"}, P{"0", "0"}, P{"0", "1"}}) {
        const LoadMoves m = load_moves(true, p.first, p.second);
        CHECK(!m.hc_check_early && !m.settle_after_cache && !m.first_prompt_buffers);
    }
    // the flag as the engine sets it: off until the card is named
    CHECK(!class_flag().load());
    CHECK(!on(Fix::Load) && !on(Fix::Exit));
}
// the class from the topology text, as /sys/class/kfd/kfd/topology/nodes/N/properties reads (shortened)
const char* kCpuNode = "cpu_cores_count 16\nsimd_count 0\nmem_banks_count 1\ngfx_target_version 0\nvendor_id 0\n";
const char* k780M = "cpu_cores_count 0\nsimd_count 24\nmem_banks_count 1\ngfx_target_version 110003\nvendor_id 4098\ndevice_id 5567\n";
const char* k890M = "simd_count 32\ngfx_target_version 110500\n";
const char* k8060S = "simd_count 80\r\ngfx_target_version 110501\r\n";   // \r tolerated: the number ends at the first non-digit
const char* k7900 = "simd_count 192\ngfx_target_version 110000\n";        // a discrete RX 7900 XTX
const char* kR9700 = "simd_count 128\ngfx_target_version 120001\n";

void topology() {
    {
        const Node n = parse_node_properties(k780M);
        CHECK(n.gfx == 110003 && n.simds == 24);
        const Node c = parse_node_properties(kCpuNode);
        CHECK(c.gfx == 0 && c.simds == 0);
        CHECK(parse_node_properties(k8060S).gfx == 110501);
        CHECK(parse_node_properties("").gfx == 0);
        CHECK(parse_node_properties("gfx_target_version\nsimd_count x\n").simds == 0);   // no number: not read
        CHECK(parse_node_properties("xgfx_target_version 110003\n").gfx == 0);            // another key
    }
    auto cls = [](std::vector<const char*> nodes) {
        std::vector<Node> v;
        for (const char* t : nodes) v.push_back(parse_node_properties(t));
        return nodes_in_class(v);
    };
    CHECK(cls({kCpuNode, k780M}));
    CHECK(cls({kCpuNode, k890M}));
    CHECK(cls({kCpuNode, k8060S}));
    CHECK(!cls({}));                        // no topology (no KFD, another OS, a container without /sys/class/kfd)
    CHECK(!cls({kCpuNode}));                // no GPU node
    CHECK(!cls({kCpuNode, k7900}));         // a discrete gfx11
    CHECK(!cls({kCpuNode, kR9700}));
    CHECK(!cls({kCpuNode, k780M, k7900}));  // a 780M beside a discrete card: not the class (the card keeps its path)
    CHECK(!cls({kCpuNode, k890M, kR9700}));
    // the value: only in the class, only with the switches on, never over the caller's own value
    CHECK(xfer_value(true, nullptr, nullptr, nullptr) != nullptr && std::string(xfer_value(true, nullptr, nullptr, nullptr)) == "1000000");
    CHECK(xfer_value(false, nullptr, nullptr, nullptr) == nullptr);
    CHECK(xfer_value(true, "0", nullptr, nullptr) == nullptr);
    CHECK(xfer_value(true, nullptr, "0", nullptr) == nullptr);
    CHECK(xfer_value(true, nullptr, nullptr, "4") == nullptr);
    CHECK(xfer_value(true, nullptr, nullptr, "") == nullptr);   // set, even empty: the caller's
#if defined(__linux__)
    // the folder reader on a temporary tree: two nodes, a stray file, a node without properties
    char tmpl[] = "/tmp/kfdqXXXXXX";
    if (mkdtemp(tmpl) != nullptr) {
        const std::string root = tmpl;
        auto put = [](const std::string& path, const char* text) {
            if (std::FILE* f = std::fopen(path.c_str(), "w")) { std::fputs(text, f); std::fclose(f); }
        };
        mkdir((root + "/0").c_str(), 0700);
        mkdir((root + "/1").c_str(), 0700);
        mkdir((root + "/2").c_str(), 0700);
        put(root + "/0/properties", kCpuNode);
        put(root + "/1/properties", k780M);
        put(root + "/README", "not a node");
        const std::vector<Node> v = read_topology(root);
        CHECK(v.size() == 2);
        CHECK(nodes_in_class(v));
        put(root + "/2/properties", k7900);
        CHECK(!nodes_in_class(read_topology(root)));
        CHECK(read_topology(root + "/none").empty());
        for (const char* p : {"/0/properties", "/1/properties", "/2/properties", "/README"}) std::remove((root + p).c_str());
        for (const char* p : {"/0", "/1", "/2", ""}) rmdir((root + p).c_str());
    } else {
        std::fprintf(stderr, "FAIL %s:%d: mkdtemp failed\n", __FILE__, __LINE__);
        ++failed;
    }
#endif
}

// a fake GPU layer for the stop: a clock in seconds, a GPU that finishes after `busy_s` once released (never when `hung`)
struct FakeGpu {
    double t = 0.0;
    double busy_s = 0.5;
    bool hung = false;
    bool stopped = false, unregistered = false;
    std::string order;
    TeardownResult run(double bound) {
        auto stop = [&] { stopped = true; order += "S"; };
        auto wait_for = [&](int ms) {   // waits until done or ms passed; true when done
            const double lim = ms / 1000.0;
            if (!hung && busy_s <= lim) { t += busy_s; busy_s = 0.0; return true; }
            t += lim;
            if (!hung) busy_s -= lim;
            return false;
        };
        auto release = [&](int ms) { order += "R"; return wait_for(ms); };
        auto drain = [&](int ms) { order += "D"; return wait_for(ms); };
        auto unreg = [&] { unregistered = true; order += "U"; };
        return teardown(stop, release, drain, unreg, [&] { return t; }, bound);
    }
};

void stop_order() {
    // a GPU that finishes in 0.5 s: stop, release, drain, unregister; inside the bound
    {
        FakeGpu g;
        const TeardownResult r = g.run(kExitBoundS);
        CHECK(g.order == "SRDU");
        CHECK(r.released && r.drained && r.unregistered && g.unregistered);
        CHECK(g.t <= kExitBoundS + 1e-9);
    }
    // a GPU that takes 2.5 s: the release uses most of the bound, the drain the rest; still drained
    {
        FakeGpu g;
        g.busy_s = 2.5;
        const TeardownResult r = g.run(kExitBoundS);
        CHECK(r.released && r.drained && r.unregistered);
        CHECK(g.t <= kExitBoundS + 1e-9);
    }
    // a GPU that takes 5 s: not drained within 3 s: the registration stays (unregistering under a running copy is worse), and
    // the whole stop took no more than the bound
    {
        FakeGpu g;
        g.busy_s = 5.0;
        const TeardownResult r = g.run(kExitBoundS);
        CHECK(!r.released && !r.drained && !r.unregistered && !g.unregistered);
        CHECK(g.order == "SRD");
        CHECK(g.t <= kExitBoundS + 0.002);
    }
    // a hung GPU: the same, bounded
    {
        FakeGpu g;
        g.hung = true;
        const TeardownResult r = g.run(kExitBoundS);
        CHECK(!r.drained && !g.unregistered && g.stopped);
        CHECK(g.t <= kExitBoundS + 0.002);
    }
    // the bound stays inside the server's 5 s between SIGTERM and SIGKILL (serve/server.py ENGINE_TERM_WAIT_S)
    CHECK(kExitBoundS > 0.0 && kExitBoundS < 5.0);
    // the first signal starts the stop, a second one takes the default action at once
    CHECK(on_signal(0) == OnSignal::StartTeardown);
    CHECK(on_signal(1) == OnSignal::DefaultAction);
    CHECK(on_signal(2) == OnSignal::DefaultAction);
}

void noreg() {
    CHECK(std::string(fix_env(Fix::NoReg)) == "STRATA_KFD_QUIET_NOREG");
    // the class default: unregistered, unless a switch says 0 or STRATA_UMA_REGISTER=1 asks for the registration
    CHECK(noreg_default(true, nullptr, nullptr, nullptr));
    CHECK(noreg_default(true, "1", "1", nullptr));
    CHECK(noreg_default(true, nullptr, nullptr, "0"));
    CHECK(!noreg_default(false, nullptr, nullptr, nullptr));   // every other card keeps its registered arena
    CHECK(!noreg_default(false, "1", "1", nullptr));
    CHECK(!noreg_default(true, "0", nullptr, nullptr));        // STRATA_KFD_QUIET=0
    CHECK(!noreg_default(true, nullptr, "0", nullptr));        // STRATA_KFD_QUIET_NOREG=0
    CHECK(!noreg_default(true, nullptr, nullptr, "1"));        // STRATA_UMA_REGISTER=1
    // the arena's mode: the user's STRATA_ARENA_PIN_GIB wins both ways; unset or empty, the engine's flag
    CHECK(arena_unregistered(nullptr, true));
    CHECK(arena_unregistered("", true));
    CHECK(!arena_unregistered(nullptr, false));
    CHECK(arena_unregistered("none", false));
    CHECK(arena_unregistered("none", true));
    CHECK(!arena_unregistered("0", true));      // the user registers the whole arena
    CHECK(!arena_unregistered("8", true));      // ... or a capped part of it
    CHECK(!arena_unregistered("auto", true));
    // the lock: never tried in the class default; a STRATA_ARENA_LOCK the user set decides; outside NOREG as before (lock unless "0")
    CHECK(!arena_lock_wanted(true, nullptr));
    CHECK(arena_lock_wanted(true, "1"));
    CHECK(!arena_lock_wanted(true, "0"));
    CHECK(arena_lock_wanted(false, nullptr));
    CHECK(arena_lock_wanted(false, "1"));
    CHECK(arena_lock_wanted(false, ""));
    CHECK(!arena_lock_wanted(false, "0"));
}
}  // namespace

int main() {
    switches();
    noreg();
    load_order();
    topology();
    stop_order();
    if (failed) {
        std::fprintf(stderr, "kfd_quiet_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("kfd_quiet_test: ok\n");
    return 0;
}
