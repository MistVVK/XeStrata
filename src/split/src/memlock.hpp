// Locking the split engine's host-read tables in RAM, as far as this machine lets it (header only, Linux; the CUDA and the HIP build use the same code).
//
// The host gathers rows of token_embd.weight and per_layer_token_embd.weight every window; the tables are file mappings, and a row that is not in memory
// costs a read from the disk. Locking them (mlock) keeps the pages in memory when the system is short. It is a speed measure, never a need: the rows are
// plain reads through the mapping (nothing here is page-locked for the cards), so the engine runs the same with the tables unlocked.
//
// So the engine works out what it can lock and locks that, and never fails because of it:
//   1. the soft limit of RLIMIT_MEMLOCK is raised to the hard limit (no privilege needed), as far as the tables need;
//   2. the budget is what that limit still allows, and what the free RAM allows with a reserve left for the rest of the system;
//   3. the parts are locked in the order the caller gives them (the caller puts the table whose rows the host reads most scattered first: the PLE table), each as much as the budget allows;
//   4. what could not be locked is read into memory anyway (16 threads side by side), as before: it stays in the page cache, unlocked.
// A failing mlock call (another process used the allowance, the kernel said no) only leaves its range unlocked.
#pragma once

#include <sys/mman.h>
#include <sys/resource.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace fnx {

struct LockPart {
    const char* name = "";
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    size_t locked = 0;   // bytes at the start of the part that ended up locked (filled by lock_parts)
};

// what the machine allows right now (a test fills it by hand)
struct LockEnv {
    uint64_t limit = UINT64_MAX;     // bytes this process may still lock (UINT64_MAX: no limit)
    uint64_t avail = UINT64_MAX;     // MemAvailable
    uint64_t reserve = 0;            // RAM left to the rest of the system
    uint64_t limit_hard = UINT64_MAX; // hard limit as read (information for the message)
    bool raised = false;             // the soft limit was raised by us
};

struct LockOutcome {
    size_t want = 0;       // bytes of all parts
    size_t locked = 0;     // bytes locked
    int err = 0;           // first errno of an mlock that failed although the plan allowed it (0: none)
    bool limited_by_ram = false;   // the budget was cut by the free RAM, not by the memory-lock limit
    LockEnv env;
};

using MlockFn = int (*)(const void*, size_t);

constexpr uintptr_t kLockPage = 4096;

// RAM left to the rest of the system when the tables are locked: the larger of a fixed amount and a share of the machine's RAM. These two numbers are a CHOICE, not a
// measurement: nothing was measured to say that 4 GiB or 5% is the right amount to leave. They only keep the lock from taking the last free memory of a small PC.
constexpr uint64_t kLockReserveMinBytes = 4ull << 30;   // 4 GiB
constexpr uint64_t kLockReserveShareDiv = 20;            // 1/20 = 5% of MemTotal

inline uint64_t proc_kib(const char* file, const char* key) {   // "Key:   123 kB" of /proc/<file>; 0 when absent
    std::FILE* f = std::fopen(file, "r");
    if (!f) return 0;
    char line[256];
    uint64_t v = 0;
    const size_t kl = std::strlen(key);
    while (std::fgets(line, sizeof line, f))
        if (!std::strncmp(line, key, kl) && line[kl] == ':') {
            v = std::strtoull(line + kl + 1, nullptr, 10);
            break;
        }
    std::fclose(f);
    return v;
}

inline uint64_t lock_page_ceil(uint64_t x) { return (x + kLockPage - 1) / kLockPage * kLockPage; }

// Step 1 and the environment: the soft limit raised towards the hard limit for `need` bytes (plus 1 GiB of room for the other locks), then what is left.
inline LockEnv read_lock_env(uint64_t need) {
    LockEnv e;
    struct rlimit rl {};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        const uint64_t room = need + (1ull << 30);
        if (rl.rlim_cur != RLIM_INFINITY && (uint64_t) rl.rlim_cur < need && rl.rlim_max > rl.rlim_cur) {
            struct rlimit up = rl;
            up.rlim_cur = rl.rlim_max == RLIM_INFINITY ? (rlim_t) room : std::min<rlim_t>(rl.rlim_max, (rlim_t) room);
            if (setrlimit(RLIMIT_MEMLOCK, &up) == 0) {
                rl = up;
                e.raised = true;
            }
        }
        e.limit_hard = rl.rlim_max == RLIM_INFINITY ? UINT64_MAX : (uint64_t) rl.rlim_max;
        if (rl.rlim_cur == RLIM_INFINITY) e.limit = UINT64_MAX;
        else {
            const uint64_t held = proc_kib("/proc/self/status", "VmLck") * 1024;
            e.limit = (uint64_t) rl.rlim_cur > held ? (uint64_t) rl.rlim_cur - held : 0;
        }
    }
    const uint64_t avail = proc_kib("/proc/meminfo", "MemAvailable") * 1024, total = proc_kib("/proc/meminfo", "MemTotal") * 1024;
    e.avail = avail ? avail : UINT64_MAX;   // no /proc/meminfo: no RAM cut
    e.reserve = std::max<uint64_t>(kLockReserveMinBytes, total / kLockReserveShareDiv);
    return e;
}

// Steps 2 and 3: how many bytes of each part are to be locked (the budget handed out in order, page-sized pieces); returns the total.
inline uint64_t plan_lock(const std::vector<LockPart>& parts, const LockEnv& env, std::vector<uint64_t>& target, bool* ram_cut = nullptr) {
    const uint64_t ram_budget = env.avail == UINT64_MAX ? UINT64_MAX : (env.avail > env.reserve ? env.avail - env.reserve : 0);
    uint64_t budget = std::min(env.limit, ram_budget);
    if (ram_cut) *ram_cut = ram_budget < env.limit;
    target.assign(parts.size(), 0);
    uint64_t total = 0;
    for (size_t i = 0; i < parts.size() && budget > 0; ++i) {
        // the pages the part touches: from its page start to its end
        const uintptr_t a = (uintptr_t) parts[i].data / kLockPage * kLockPage, e = (uintptr_t) parts[i].data + parts[i].nbytes;
        const uint64_t span = e - a;
        const uint64_t t = budget == UINT64_MAX ? span : std::min<uint64_t>(span, budget / kLockPage * kLockPage);
        target[i] = t;
        total += t;
        if (budget != UINT64_MAX) budget -= t;
    }
    return total;
}

// Step 4 and the locking: LOCKERS threads, each locks its piece of the planned range and then reads in its piece of the rest (a cold file is read 2x faster
// by 16 threads side by side than by one; tools/stage_test.cpp). A part's `locked` is the length of the planned prefix whose pieces all succeeded in a row.
inline LockOutcome lock_parts(std::vector<LockPart>& parts, LockEnv env, MlockFn fn = ::mlock) {
    constexpr int LOCKERS = 16;
    LockOutcome out;
    out.env = env;
    for (const auto& p : parts) out.want += p.nbytes;
    std::vector<uint64_t> target;
    bool ram_cut = false;
    plan_lock(parts, env, target, &ram_cut);
    out.limited_by_ram = ram_cut;
    for (size_t i = 0; i < parts.size(); ++i) {
        LockPart& p = parts[i];
        p.locked = 0;
        if (!p.nbytes) continue;
        const uintptr_t a = (uintptr_t) p.data / kLockPage * kLockPage, e = (uintptr_t) p.data + p.nbytes;
        const uintptr_t lock_end = std::min<uintptr_t>(e, a + target[i]);
        const uintptr_t part = ((e - a) / LOCKERS + kLockPage - 1) / kLockPage * kLockPage;
        std::vector<int> err(LOCKERS, 0);
        std::vector<uint8_t> ok(LOCKERS, 0);
        std::vector<std::thread> th;
        for (int k = 0; k < LOCKERS; ++k)
            th.emplace_back([&, k] {
                const uintptr_t lo = std::min(e, a + (uintptr_t) k * part), hi = std::min(e, lo + part);
                const uintptr_t mid = std::min(hi, std::max(lo, lock_end));   // [lo, mid) is to be locked, [mid, hi) only read in
                if (mid > lo) {
                    if (fn((const void*) lo, mid - lo) == 0) ok[(size_t) k] = 1;
                    else err[(size_t) k] = errno;
                }
                volatile unsigned char sink = 0;
                for (uintptr_t q = ok[(size_t) k] ? mid : lo; q < hi; q += kLockPage) sink = sink + *(const volatile unsigned char*) q;
                (void) sink;
            });
        for (auto& t : th) t.join();
        // every piece that locked counts (a piece after a failed one still holds its pages locked)
        uint64_t locked = 0;
        for (int k = 0; k < LOCKERS; ++k) {
            const uintptr_t lo = std::min(e, a + (uintptr_t) k * part), hi = std::min(e, lo + part);
            if (ok[(size_t) k]) locked += std::min(hi, lock_end) - lo;
            if (!out.err && err[(size_t) k]) out.err = err[(size_t) k];
        }
        p.locked = (size_t) std::min<uint64_t>(locked, p.nbytes);
        out.locked += p.locked;
    }
    return out;
}

// A size for the line: MiB under 1 GiB, else GiB with one decimal.
inline std::string size_text(double bytes) {
    char b[48];
    if (bytes < 1024.0 * 1024.0 * 1024.0) std::snprintf(b, sizeof b, "%.0f MiB", bytes / (1024.0 * 1024.0));
    else std::snprintf(b, sizeof b, "%.1f GiB", bytes / (1024.0 * 1024.0 * 1024.0));
    return b;
}

// The one line for the log when not everything is locked (empty when everything is); the owner's wording of 11-11.
inline std::string lock_line(const std::vector<LockPart>& parts, const LockOutcome& o) {
    (void) parts;
    if (o.locked >= o.want) return "";
    std::string why;
    if (o.limited_by_ram) why = "locking more would leave less than " + size_text((double) o.env.reserve) + " of RAM free";
    else if (o.env.limit != UINT64_MAX) why = "this account may lock at most " + size_text((double) o.env.limit);
    else why = std::string("the system refused: ") + std::strerror(o.err ? o.err : ENOMEM);
    // a sliver (under 0.05 GiB of a table of GiB: it would print as "0.0 of its 27.5 GiB") counts as nothing locked
    const bool big = (double) o.want >= 1024.0 * 1024.0 * 1024.0;
    if (o.locked == 0 || (big && (double) o.locked < 1024.0 * 1024.0 * 1024.0 / 20.0))
        return "Split engine: its word tables (" + size_text((double) o.want) + ") are not locked in RAM (" + why + "). It runs as usual. The tables are read into memory at the start; "
               "only if the PC runs short of memory can long prompts slow down.";
    // "14.0 of its 27.5 GiB of word tables": one unit after the total when both are in GiB
    const bool gib = big;
    char got[32];
    std::snprintf(got, sizeof got, "%.1f", (double) o.locked / (1024.0 * 1024.0 * 1024.0));
    const std::string locked_text = gib ? std::string(got) : size_text((double) o.locked);
    return "Split engine: " + locked_text + " of its " + size_text((double) o.want) + " of word tables are locked in RAM (" + why + "). It runs as usual. The rest is read into memory at the start; "
           "only if the PC runs short of memory can long prompts slow down.";
}

}  // namespace fnx
