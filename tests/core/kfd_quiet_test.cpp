// keeping the KFD quiet on an integrated gfx11 GPU under Linux (include/strata/core/kfd_quiet.hpp).  No GPU: the switch rule and the
// load order's rule.  (The device rule itself is igpu_kfd_settle_default, checked in src/core/arch_defaults_test.cpp.)
#include "strata/core/kfd_quiet.hpp"

#include <cstdio>
#include <string>
#include <utility>

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
    for (const auto& p : {std::pair<const char*, const char*>{"0", nullptr}, {nullptr, "0"}, {"0", "0"}, {"0", "1"}}) {
        const LoadMoves m = load_moves(true, p.first, p.second);
        CHECK(!m.hc_check_early && !m.settle_after_cache && !m.first_prompt_buffers);
    }
    // the flag as the engine sets it: off until the card is named
    CHECK(!class_flag().load());
    CHECK(!on(Fix::Load) && !on(Fix::Exit));
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
}  // namespace

int main() {
    switches();
    load_order();
    stop_order();
    if (failed) {
        std::fprintf(stderr, "kfd_quiet_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("kfd_quiet_test: ok\n");
    return 0;
}
