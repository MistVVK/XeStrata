// Keeping the KFD quiet while our GPU queues are busy, on an integrated gfx11 GPU under Linux (gfx1103, gfx1150, gfx1151).
//
// Radeon 780M (gfx1103), Linux, p2006: the GPU resets when the kernel evicts the process's queues while a dispatch is in flight.
// The eviction comes from page moves inside memory the GPU reaches as a userptr (the registered expert arena; pageable copies the
// HIP runtime pins in place) and from process exit with the arena still registered.  What the engine controls is WHEN it provokes
// the kernel (its own allocations, frees and its exit) and that its queues are idle then.  Three changes, one switch each:
//   LOAD  the load order: the hyper-connection check before the expert arena exists, one more KFD settle between the expert cache
//         and the profile fill, the first prompt's lazy device buffers made in Prefill::init;
//   EXIT  SIGTERM / SIGINT / SIGHUP: stop the request, release the GPU waits, drain the GPU for at most 3 s, unregister the arena
//         only when drained, then end by the same signal as before;
//   XFER  GPU_PINNED_MIN_XFER_SIZE set before HIP starts (pageable copies go through the runtime's staging buffers instead of
//         pinning our pages in place); a value the caller set wins.
// None changes a kernel or what is computed.  STRATA_KFD_QUIET=0 turns all of them off at run time (A/B with the same binary);
// STRATA_KFD_QUIET_LOAD / _EXIT / _XFER = 0 turn one off.  STRATA_TRACE prints one line per change that is active.
//
// Pure pieces, testable without a GPU (tests/core/kfd_quiet_test.cpp): the switch rule, the topology rule (the device class
// from the KFD topology files, readable before HIP starts), and the teardown order with the GPU layer passed in.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <cstdio>
#endif

namespace strata::core::kfd_quiet {

enum class Fix { Load, Exit, Xfer };

/// The device class, set by the engine once the card is named (the rule of igpu_kfd_settle_default: integrated gfx1103 / gfx1150 /
/// gfx1151, Linux, every visible card).  Load and Exit do nothing while it is false.
inline std::atomic<bool>& class_flag() {
    static std::atomic<bool> f{false};
    return f;
}

/// The switch rule: `all` is STRATA_KFD_QUIET, `one` the fix's own variable (nullptr = unset).  A value starting with '0' turns off;
/// unset, empty or anything else leaves it on.
inline bool switch_on(const char* all, const char* one) {
    auto off = [](const char* v) { return v != nullptr && v[0] == '0'; };
    return !off(all) && !off(one);
}

inline const char* fix_env(Fix f) {
    switch (f) {
        case Fix::Load: return "STRATA_KFD_QUIET_LOAD";
        case Fix::Exit: return "STRATA_KFD_QUIET_EXIT";
        case Fix::Xfer: return "STRATA_KFD_QUIET_XFER";
    }
    return "STRATA_KFD_QUIET_LOAD";
}

inline bool env_on(Fix f) { return switch_on(std::getenv("STRATA_KFD_QUIET"), std::getenv(fix_env(f))); }

/// Load and Exit: the device class and the switches.
inline bool on(Fix f) { return class_flag().load(std::memory_order_relaxed) && env_on(f); }

/// LOAD, as one rule for the three call sites (generate.cpp: the check after the card is named, the settle after the expert cache;
/// prefill.cpp: the first prompt's buffers in Prefill::init).  `cls`: class_flag(); `all` / `one`: the switches.
struct LoadMoves {
    bool hc_check_early = false;
    bool settle_after_cache = false;
    bool first_prompt_buffers = false;
};
inline LoadMoves load_moves(bool cls, const char* all, const char* one) {
    LoadMoves m;
    m.hc_check_early = m.settle_after_cache = m.first_prompt_buffers = cls && switch_on(all, one);
    return m;
}
inline LoadMoves load_moves() {
    return load_moves(class_flag().load(std::memory_order_relaxed), std::getenv("STRATA_KFD_QUIET"), std::getenv(fix_env(Fix::Load)));
}

// ---- EXIT: the teardown order ---------------------------------------------------------------------------------------------------

/// The bound of the stop, seconds.  The server stops its engine with STOP + QUIT and waits 20 s, then sends SIGTERM and waits 5 s
/// before SIGKILL (serve/server.py ENGINE_QUIT_WAIT_S / ENGINE_TERM_WAIT_S); the macOS lifeline waits 10 s.  3 s stays inside 5 s.
constexpr double kExitBoundS = 3.0;

struct TeardownResult {
    bool released = false;      // the verify windows' GPU waits were released and their streams finished
    bool drained = false;       // the whole device finished within the bound
    bool unregistered = false;  // the arena's registration was dropped (only when drained)
};

/// The order of the stop: `stop()` (the request ends at its next step), `release(ms)` -> bool (raise the spin flags; wait at most ms
/// for the verify streams), `drain(ms)` -> bool (wait at most ms for the whole device), and only when drained `unregister()`.
/// `now()` is seconds; the release and the drain share one bound of `bound_s` from the call.
template <class Stop, class Release, class Drain, class Unregister, class Now>
TeardownResult teardown(Stop stop, Release release, Drain drain, Unregister unregister, Now now, double bound_s) {
    TeardownResult r;
    const double t0 = now();
    auto left_ms = [&] {
        const double l = bound_s - (now() - t0);
        return l > 0.0 ? (int) (l * 1000.0) : 0;
    };
    stop();
    r.released = release(left_ms());
    r.drained = drain(left_ms());
    if (r.drained) {
        unregister();
        r.unregistered = true;
    }
    return r;
}

/// What a signal does once the handler is in: the first SIGTERM / SIGINT / SIGHUP starts the teardown; a second one while it runs
/// ends the process at once by its default action (a user pressing Ctrl+C twice).
enum class OnSignal { StartTeardown, DefaultAction };
inline OnSignal on_signal(int signals_before) { return signals_before == 0 ? OnSignal::StartTeardown : OnSignal::DefaultAction; }

}  // namespace strata::core::kfd_quiet
