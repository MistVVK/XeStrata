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
//   NOREG the expert arena is not registered with the driver (the unregistered "UMA" path of pinned.cu), and not mlock'ed; a value
//         the user set for STRATA_ARENA_PIN_GIB (or STRATA_UMA_REGISTER=1) wins.
// None changes a kernel or what is computed.  STRATA_KFD_QUIET=0 turns all of them off at run time (A/B with the same binary);
// STRATA_KFD_QUIET_LOAD / _EXIT / _XFER / _NOREG = 0 turn one off.  STRATA_TRACE prints one line per change that is active.
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

enum class Fix { Load, Exit, Xfer, NoReg, Lend };

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
        case Fix::NoReg: return "STRATA_KFD_QUIET_NOREG";
        case Fix::Lend: return "STRATA_KFD_QUIET_LEND";
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

// ---- NOREG: the expert arena is not registered with the GPU driver on the class ---------------------------------------------------

/// Set by the engine (with arena_uma_no_register_flag, pinned.hpp) when NOREG chose the unregistered arena: the arena is then not
/// locked with mlock either (it cannot succeed under the default memlock limit of this class and only said "mlock failed").
inline std::atomic<bool>& noreg_flag() {
    static std::atomic<bool> f{false};
    return f;
}

/// NOREG: the class, the switches, and STRATA_UMA_REGISTER=1 (the existing "register it as before") not set.
inline bool noreg_default(bool cls, const char* all, const char* one, const char* uma_register) {
    if (uma_register != nullptr && uma_register[0] == '1') return false;
    return cls && switch_on(all, one);
}

/// The arena's mode (pinned.cu arena_no_register): a value the user set for STRATA_ARENA_PIN_GIB wins ("none" = not registered,
/// anything else = registered as that value says); unset or empty, the engine's flag decides.
inline bool arena_unregistered(const char* pin_gib, bool engine_flag) {
    if (pin_gib != nullptr && *pin_gib != '\0') return std::string(pin_gib) == "none";
    return engine_flag;
}

/// Whether the unregistered arena is locked with mlock: a STRATA_ARENA_LOCK the user set decides as before ("0" = no lock, anything
/// else = lock); unset, it is locked unless NOREG chose the mode (`noreg`).
inline bool arena_lock_wanted(bool noreg, const char* lock_env) {
    if (lock_env != nullptr) return std::string(lock_env) != "0";
    return !noreg;
}

// ---- XFER: the device class from the KFD topology (before HIP starts) ------------------------------------------------------------

/// One node's properties text: `gfx_target_version` and `simd_count` (0 when absent).  A CPU node has simd_count 0.
struct Node {
    int64_t gfx = 0;
    int64_t simds = 0;
};

inline Node parse_node_properties(const std::string& text) {
    Node n;
    size_t i = 0;
    while (i < text.size()) {
        size_t e = text.find('\n', i);
        if (e == std::string::npos) e = text.size();
        const std::string line = text.substr(i, e - i);
        i = e + 1;
        const size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        const std::string key = line.substr(0, sp);
        int64_t* dst = key == "gfx_target_version" ? &n.gfx : key == "simd_count" ? &n.simds : nullptr;
        if (dst == nullptr) continue;
        int64_t v = 0;
        size_t j = sp;
        while (j < line.size() && line[j] == ' ') ++j;
        bool any = false;
        for (; j < line.size() && line[j] >= '0' && line[j] <= '9' && v < (int64_t) 1000000000000000LL; ++j) {
            v = v * 10 + (line[j] - '0');
            any = true;
        }
        if (any) *dst = v;
    }
    return n;
}

/// gfx1103 = 110003, gfx1150 = 110500, gfx1151 = 110501 (major * 10000 + minor * 100 + stepping).  All three exist only as
/// integrated GPUs.
inline bool gfx_in_class(int64_t gfx) { return gfx == 110003 || gfx == 110500 || gfx == 110501; }

/// The class: at least one GPU node, and every GPU node one of the three.  A machine with any other GPU beside them (a discrete
/// card) is not the class, so its path stays the old one.
inline bool nodes_in_class(const std::vector<Node>& nodes) {
    int gpus = 0;
    for (const Node& n : nodes) {
        if (n.simds <= 0) continue;
        ++gpus;
        if (!gfx_in_class(n.gfx)) return false;
    }
    return gpus > 0;
}

#if defined(__linux__)
/// The nodes under `topo` (normally /sys/class/kfd/kfd/topology/nodes); empty when the folder is not there.
inline std::vector<Node> read_topology(const std::string& topo) {
    std::vector<Node> out;
    DIR* d = opendir(topo.c_str());
    if (d == nullptr) return out;
    while (const dirent* e = readdir(d)) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        const std::string path = topo + "/" + e->d_name + "/properties";
        std::FILE* f = std::fopen(path.c_str(), "r");
        if (f == nullptr) continue;
        std::string text;
        char buf[4096];
        size_t got = 0;
        while ((got = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, got);
        std::fclose(f);
        out.push_back(parse_node_properties(text));
    }
    closedir(d);
    return out;
}
#endif

/// XFER: the value the engine sets, or nullptr when nothing is to be set.  `in_class`: the topology rule; `already`:
/// GPU_PINNED_MIN_XFER_SIZE as the caller set it (a value the caller set wins).  1000000 is what STRATA_ARENA_MMAP=1 sets already
/// (generate.cpp): the runtime reads it in MiB, so no pageable copy of ours reaches the pin-in-place path.
inline const char* xfer_value(bool in_class, const char* all, const char* one, const char* already) {
    if (!in_class || !switch_on(all, one) || already != nullptr) return nullptr;
    return "1000000";
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
