// src/memlock.hpp: the split engine locks its host-read tables as far as the machine lets it and never fails because of it. No GPU. Cases:
//   1 plan     budget handed out in order: the first part first, page-sized pieces, cut by the memory-lock limit and by the free RAM
//   2 fake     an mlock that always succeeds / always fails / fails past a total: no exception, the numbers add up, the tables are untouched, the line is right
//   3 real     the real mlock under a lowered RLIMIT_MEMLOCK in a child process (the lock may fail or succeed; either way the lock call returns and the bytes are intact)
//   4 raise    a soft limit below the need and a hard limit above it is raised (child process)
#include "../src/memlock.hpp"

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

using namespace fnx;

namespace {

int fails = 0;
#define CHECK(c)                                                                                  \
    do {                                                                                          \
        if (!(c)) {                                                                               \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);                                      \
            ++fails;                                                                              \
        }                                                                                         \
    } while (0)

constexpr size_t MiB = 1u << 20;

uint8_t* map_pages(size_t n, uint8_t seed) {
    void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) std::abort();
    auto* b = (uint8_t*) p;
    for (size_t i = 0; i < n; ++i) b[i] = (uint8_t) (seed + i * 7);
    return b;
}
uint64_t sum(const uint8_t* b, size_t n) {
    uint64_t s = 0;
    for (size_t i = 0; i < n; ++i) s = s * 31 + b[i];
    return s;
}

std::atomic<uint64_t> g_locked{0}, g_calls{0}, g_cap{UINT64_MAX};
int fake_ok(const void*, size_t) { ++g_calls; return 0; }
int fake_fail(const void*, size_t) { ++g_calls; errno = ENOMEM; return -1; }
int fake_cap(const void*, size_t n) {   // like the kernel: a call that would pass the allowance fails whole
    ++g_calls;
    uint64_t cur = g_locked.load();
    for (;;) {
        if (cur + n > g_cap.load()) { errno = ENOMEM; return -1; }
        if (g_locked.compare_exchange_weak(cur, cur + n)) return 0;
    }
}

std::vector<LockPart> two_parts(uint8_t* a, size_t na, uint8_t* b, size_t nb) {
    LockPart x, y;
    x.name = "token_embd.weight"; x.data = a; x.nbytes = na;
    y.name = "per_layer_token_embd.weight"; y.data = b; y.nbytes = nb;
    return {x, y};
}

void test_plan() {
    uint8_t* a = map_pages(2 * MiB, 1);
    uint8_t* b = map_pages(8 * MiB, 2);
    auto parts = two_parts(a, 2 * MiB, b, 8 * MiB);
    std::vector<uint64_t> t;
    LockEnv e;   // nothing limits
    CHECK(plan_lock(parts, e, t) == 10 * MiB && t[0] == 2 * MiB && t[1] == 8 * MiB);
    e.limit = 1 * MiB;   // the limit cuts inside the first part: the second gets nothing
    CHECK(plan_lock(parts, e, t) == 1 * MiB && t[0] == 1 * MiB && t[1] == 0);
    e.limit = 5 * MiB;   // the first whole, the second its start
    CHECK(plan_lock(parts, e, t) == 5 * MiB && t[0] == 2 * MiB && t[1] == 3 * MiB);
    e.limit = 0;
    CHECK(plan_lock(parts, e, t) == 0);
    e = LockEnv{};
    e.avail = 12 * MiB; e.reserve = 9 * MiB;   // the free RAM leaves 3 MiB
    bool ram = false;
    CHECK(plan_lock(parts, e, t, &ram) == 3 * MiB && ram);
    e.avail = 4 * MiB;   // below the reserve: nothing
    CHECK(plan_lock(parts, e, t, &ram) == 0 && ram);
    munmap(a, 2 * MiB); munmap(b, 8 * MiB);
}

void test_fake() {
    uint8_t* a = map_pages(2 * MiB + 100, 3);   // not a whole number of pages: the piece edges must still work
    uint8_t* b = map_pages(8 * MiB + 17, 4);
    const uint64_t sa = sum(a, 2 * MiB + 100), sb = sum(b, 8 * MiB + 17);
    // everything locks: no line
    {
        auto parts = two_parts(a, 2 * MiB + 100, b, 8 * MiB + 17);
        g_calls = 0;
        LockOutcome o = lock_parts(parts, LockEnv{}, fake_ok);
        CHECK(o.locked == o.want && o.want == 10 * MiB + 117);
        CHECK(parts[0].locked == parts[0].nbytes && parts[1].locked == parts[1].nbytes);
        CHECK(lock_line(parts, o).empty());
        CHECK(g_calls > 0);
    }
    // the limit lets the first part and a bit of the second: partial, the line says "only partly" and names the limit
    {
        auto parts = two_parts(a, 2 * MiB + 100, b, 8 * MiB + 17);
        LockEnv e;
        e.limit = 4 * MiB;
        LockOutcome o = lock_parts(parts, e, fake_ok);
        CHECK(parts[0].locked == parts[0].nbytes);
        CHECK(parts[1].locked > 0 && parts[1].locked < parts[1].nbytes);
        CHECK(o.locked <= 4 * MiB + 4096);
        const std::string l = lock_line(parts, o);
        CHECK(l.find(" of its ") != std::string::npos && l.find("locked in RAM") != std::string::npos && l.find("at most 4 MiB") != std::string::npos);
        CHECK(l.find("runs as usual") != std::string::npos);
    }
    // nothing may be locked: mlock is not even called, the tables are read in, the line says "not locked"
    {
        auto parts = two_parts(a, 2 * MiB + 100, b, 8 * MiB + 17);
        LockEnv e;
        e.limit = 0;
        g_calls = 0;
        LockOutcome o = lock_parts(parts, e, fake_ok);
        CHECK(o.locked == 0 && g_calls == 0);
        const std::string l = lock_line(parts, o);
        CHECK(l.find("not locked in RAM") != std::string::npos);
    }
    // the system refuses although the plan allowed it (another process took the allowance): no throw, nothing counted as locked, the errno in the line
    {
        auto parts = two_parts(a, 2 * MiB + 100, b, 8 * MiB + 17);
        LockEnv e;   // no limit known
        LockOutcome o = lock_parts(parts, e, fake_fail);
        CHECK(o.locked == 0 && o.err == ENOMEM);
        CHECK(lock_line(parts, o).find("the system refused") != std::string::npos);
    }
    // the allowance runs out in the middle (a kernel that fails a call whole): a partial state, no throw, the count equals what the fake locked
    {
        auto parts = two_parts(a, 2 * MiB + 100, b, 8 * MiB + 17);
        g_locked = 0; g_cap = 6 * MiB;
        LockOutcome o = lock_parts(parts, LockEnv{}, fake_cap);
        CHECK(o.locked > 0 && o.locked < o.want);
        CHECK(o.locked == g_locked.load());
        CHECK(!lock_line(parts, o).empty());
        g_cap = UINT64_MAX;
    }
    // RAM short: the line names the reserve
    {
        auto parts = two_parts(a, 2 * MiB + 100, b, 8 * MiB + 17);
        LockEnv e;
        e.avail = 6 * MiB; e.reserve = 5 * MiB;
        LockOutcome o = lock_parts(parts, e, fake_ok);
        CHECK(o.limited_by_ram && o.locked <= 1 * MiB + 4096);
        CHECK(lock_line(parts, o).find("less than") != std::string::npos);
    }
    CHECK(sum(a, 2 * MiB + 100) == sa && sum(b, 8 * MiB + 17) == sb);   // the tables are exactly as they were
    munmap(a, 2 * MiB + 100); munmap(b, 8 * MiB + 17);
}

// runs `f` in a child (limits set there stay there); returns the child's exit code
template <class F> int in_child(F f) {
    const pid_t pid = fork();
    if (pid == 0) _exit(f());
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 99;
}

// the sentences in the owner's wording of 11-11, at the real sizes
void test_words() {
    const uint64_t G = 1ull << 30;
    std::vector<LockPart> none;
    LockOutcome o;
    o.want = (size_t) (27.5 * G);
    o.env.limit = 8 * (1ull << 20);
    o.locked = 0;
    CHECK(lock_line(none, o) == "Split engine: its word tables (27.5 GiB) are not locked in RAM (this account may lock at most 8 MiB). It runs as usual. The tables are read into memory at the start; "
                                "only if the PC runs short of memory can long prompts slow down.");
    o.locked = 8u << 20;   // a sliver of a table of GiB counts as nothing locked
    CHECK(lock_line(none, o).find("are not locked in RAM") != std::string::npos);
    o.locked = (size_t) (14.0 * G);
    o.env.limit = 14 * G;
    CHECK(lock_line(none, o) == "Split engine: 14.0 of its 27.5 GiB of word tables are locked in RAM (this account may lock at most 14.0 GiB). It runs as usual. The rest is read into memory at the start; "
                                "only if the PC runs short of memory can long prompts slow down.");
    o.env.limit = UINT64_MAX;
    o.limited_by_ram = true;
    o.env.reserve = (uint64_t) (6.3 * G);
    CHECK(lock_line(none, o).find("(locking more would leave less than 6.3 GiB of RAM free)") != std::string::npos);
    o.locked = o.want;
    CHECK(lock_line(none, o).empty());
}

void test_real() {
    // the soft AND hard limit are lowered to 256 KiB: the soft limit cannot be raised, the real mlock refuses most of 10 MiB (a process with CAP_IPC_LOCK is not limited: then all locks)
    const int rc = in_child([] {
        struct rlimit lo { 256 * 1024, 256 * 1024 };
        if (setrlimit(RLIMIT_MEMLOCK, &lo) != 0) return 90;
        uint8_t* a = map_pages(2 * MiB, 5);
        uint8_t* b = map_pages(8 * MiB, 6);
        const uint64_t sa = sum(a, 2 * MiB), sb = sum(b, 8 * MiB);
        auto parts = two_parts(a, 2 * MiB, b, 8 * MiB);
        const LockEnv e = read_lock_env(10 * MiB);
        LockOutcome o = lock_parts(parts, e);   // must return, whatever the kernel says
        const bool privileged = o.locked == o.want;
        if (!privileged && o.locked > 256 * 1024 + 4096) return 91;
        if (!privileged && lock_line(parts, o).empty()) return 92;
        if (sum(a, 2 * MiB) != sa || sum(b, 8 * MiB) != sb) return 93;
        return 0;
    });
    CHECK(rc == 0);
}

void test_raise() {
    // soft 128 KiB, hard 64 MiB, need 10 MiB: the soft limit is raised (privileged or not) and the whole lock succeeds
    const int rc = in_child([] {
        struct rlimit lo { 128 * 1024, 64 * MiB };
        if (setrlimit(RLIMIT_MEMLOCK, &lo) != 0) return 77;   // the hard limit of this process is already below 64 MiB (a test run under ulimit -l 8192): it cannot be raised here, the case does not apply
        const LockEnv e = read_lock_env(10 * MiB);
        if (!e.raised) return 91;
        struct rlimit now {};
        getrlimit(RLIMIT_MEMLOCK, &now);
        if (now.rlim_cur != 64 * MiB) return 92;   // need + 1 GiB, capped at the hard limit
        uint8_t* a = map_pages(2 * MiB, 7);
        uint8_t* b = map_pages(8 * MiB, 8);
        auto parts = two_parts(a, 2 * MiB, b, 8 * MiB);
        LockOutcome o = lock_parts(parts, e);
        if (o.locked != o.want && !(e.avail != UINT64_MAX && e.avail < o.want + e.reserve)) return 93;   // all locked, unless this PC is short of RAM
        return 0;
    });
    CHECK(rc == 0 || rc == 77);
}

}  // namespace

int main() {
    test_plan();
    test_fake();
    test_words();
    test_real();
    test_raise();
    std::printf("%s\n", fails ? "memlock_test: FAILED" : "memlock_test: all passed");
    return fails ? 1 : 0;
}
