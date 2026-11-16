// sycl/src/core/window_wait_test.cpp - a GPU wait on the host that runs out fails its window (strata/core/window_wait.hpp), no GPU, no model.
//
// A thread plays the GPU of one verify window: per step it writes the step's payload (expert ids) into "mapped" memory, rings,
// waits for the host's plan (flag A) and the CPU's rows (flag M) with the SAME bounded_wait the device kernels run, copies the
// rows and goes on.  The test thread plays the host: it waits for the ring, takes one copy of the ids, computes the rows,
// checks the record, raises the flags, and at the end accepts the window's rows only when the record is clean.
//
//   window_wait_test            the protocol of this build: every case must pass
//   window_wait_test --legacy   the protocol of 756242554 (a wait that runs out just returns, the host never checks):
//                               cases 2, 5, 6 and 7 accept stale rows, so this run must FAIL (CTest: WILL_FAIL)
//
// Each wait spins `kSpin` times with a 50 us pause, so it gives up after at least 0.2 s; a delayed worker sleeps 1.5 s.
#include "strata/core/window_wait.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;
using Clock = std::chrono::steady_clock;

static int g_fail = 0;
static bool g_legacy = false;
static void check(bool ok, const char* what) { std::printf("  %-112s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; std::fflush(stdout); }

constexpr uint32_t kSpin = 4000;
constexpr int kK = 4;                       // entries per step
constexpr auto kLate = std::chrono::milliseconds(1500);

struct Load { uint32_t operator()(const volatile uint32_t* p) const { return __atomic_load_n(p, __ATOMIC_ACQUIRE); } };
struct Store { void operator()(volatile uint32_t* p, uint32_t v) const { __atomic_store_n(p, v, __ATOMIC_RELEASE); } };
struct Pause { void operator()() const { std::this_thread::sleep_for(std::chrono::microseconds(50)); } };

static int32_t router(int win, int step, int i) { return (int32_t) ((win * 131 + step * 17 + i * 5) % 512); }   // the "layer's" experts
static float expert_row(int32_t e) { return 1.0f + (float) e; }                                              // the CPU's answer

/// One verifier's mapped memory: seq, flags, the payload, the CPU rows and the wait record.
struct Mapped {
    volatile uint32_t seq = 0, flagA = 0, flagM = 0;
    volatile int32_t ids[kK] = {};
    volatile float rows[kK] = {};
    volatile uint32_t rec[kWaitWords] = {};
    uint32_t epoch = 0;
    void stage() {   // stage_inputs(): flags to zero, the record armed (the previous window has completed)
        Store{}(&seq, 0); Store{}(&flagA, 0); Store{}(&flagM, 0);
        if (!g_legacy) wait_record_arm(rec, ++epoch);
    }
};

/// The GPU of one window of `steps` steps.  out[s] = the rows it used at step s (what the window's tokens are made of).
struct Gpu {
    Mapped& m;
    int win, steps;
    std::vector<float> out;
    std::atomic<bool> done{false};
    std::thread th;
    Gpu(Mapped& mm, int w, int s) : m(mm), win(w), steps(s), out((size_t) s * kK, -1.0f) {}
    void start() {
        th = std::thread([this] {
            volatile uint32_t* rec = g_legacy ? nullptr : m.rec;
            for (int s = 0; s < steps; ++s) {
                const uint32_t ring = (uint32_t) s + 1;
                for (int i = 0; i < kK; ++i) m.ids[i] = router(win, s, i);   // the payload of step s (the next step overwrites it)
                Store{}(&m.seq, ring);
                bounded_wait(&m.flagA, kSpin, rec, kWaitPlan, ring, Load{}, Store{}, Pause{}, [ring](uint32_t v) { return v >= ring; });
                bounded_wait(&m.flagM, kSpin, rec, kWaitRows, ring, Load{}, Store{}, Pause{}, [ring](uint32_t v) { return v >= ring; });
                for (int i = 0; i < kK; ++i) out[(size_t) s * kK + i] = m.rows[i];
            }
            done = true;
        });
    }
    bool join_within(std::chrono::milliseconds ms) {
        const auto t0 = Clock::now();
        while (!done.load() && Clock::now() - t0 < ms) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!done.load()) return false;
        th.join();
        return true;
    }
};

enum class Outcome { Accepted, Failed, HostTimeout, Cancelled };
struct Result {
    Outcome outcome = Outcome::Accepted;
    WaitExpiry x;
    bool rows_right = false;   // out == the reference for every step
    bool drained = true;
    double ms = 0;
};

/// The host's plan for one window: how late each step is, and where.
struct HostPlan {
    int late_before_plan = -1;     // sleep kLate before raising flag A at this step (inside the "pool")
    int late_before_rows = -1;     // raise flag A on time, sleep kLate before flag M
    int late_after_check = -1;     // check the record (clean), THEN sleep kLate, then raise A and M (the late expiry)
    bool late_every_step = false;  // every step is late
    int delay_ms_every = 0;        // every step this many ms late (within the bound)
    bool dead = false;             // never serves
    int cancel_after = -1;         // stop serving after this step
};

static Result run_window(Mapped& m, int win, int steps, const HostPlan& hp) {
    Result r;
    const auto t0 = Clock::now();
    m.stage();
    Gpu gpu(m, win, steps);
    gpu.start();
    bool failed = false, cancelled = false;
    auto drain = [&] {   // drain_failed_window(): every flag past every ring, then the "queue", bounded
        Store{}(&m.flagA, UINT32_MAX);
        Store{}(&m.flagM, UINT32_MAX);
        r.drained = gpu.join_within(std::chrono::milliseconds(5000));
    };
    for (int s = 0; s < steps && !failed && !cancelled; ++s) {
        const uint32_t want = (uint32_t) s + 1;
        const auto a = Clock::now();
        if (hp.dead) {   // the host never answers: its own ring timeout (here 4 s instead of 20) ends the wait
            while (Clock::now() - a < std::chrono::seconds(4)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
            r.outcome = Outcome::HostTimeout;
            drain();
            r.x = wait_record_read(m.rec);
            r.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            return r;
        }
        while (Load{}(&m.seq) < want) {
            if (Clock::now() - a > std::chrono::seconds(20)) { r.outcome = Outcome::HostTimeout; drain(); return r; }
            std::this_thread::yield();
        }
        int32_t ids[kK];
        for (int i = 0; i < kK; ++i) ids[i] = m.ids[i];   // ONE copy of the payload, read once (#6016)
        float rows[kK];
        for (int i = 0; i < kK; ++i) rows[i] = expert_row(ids[i]);
        if (hp.late_every_step || s == hp.late_before_plan) std::this_thread::sleep_for(kLate);
        if (hp.delay_ms_every > 0) std::this_thread::sleep_for(std::chrono::milliseconds(hp.delay_ms_every));
        if (!g_legacy && wait_record_read(m.rec).expired()) { failed = true; drain(); break; }   // after the CPU step
        if (s == hp.late_after_check) std::this_thread::sleep_for(kLate);
        for (int i = 0; i < kK; ++i) m.rows[i] = rows[i];
        Store{}(&m.flagA, want);
        if (s == hp.late_before_rows) {
            std::this_thread::sleep_for(kLate);
            if (!g_legacy && wait_record_read(m.rec).expired()) { failed = true; drain(); break; }
        }
        Store{}(&m.flagM, want);
        if (s == hp.cancel_after) { cancelled = true; drain(); }
    }
    if (!failed && !cancelled) r.drained = gpu.join_within(std::chrono::milliseconds(20000));   // the window's sync
    if (!g_legacy && wait_record_read(m.rec).expired()) failed = true;                          // after the sync
    r.x = wait_record_read(m.rec);
    r.outcome = cancelled ? Outcome::Cancelled : failed ? Outcome::Failed : Outcome::Accepted;
    r.rows_right = true;
    for (int s = 0; s < steps; ++s)
        for (int i = 0; i < kK; ++i)
            if (gpu.out[(size_t) s * kK + i] != expert_row(router(win, s, i))) r.rows_right = false;
    r.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return r;
}

/// The fail-closed rule every case must keep: a window is accepted only with every row right.
static bool fail_closed(const Result& r) { return r.outcome != Outcome::Accepted || r.rows_right; }

int main(int argc, char** argv) {
    g_legacy = argc > 1 && std::strcmp(argv[1], "--legacy") == 0;
    std::printf("window_wait_test%s\n", g_legacy ? " --legacy (the 756242554 protocol: must fail)" : "");
    // 1. on time
    {
        Mapped m;
        const Result r = run_window(m, 1, 6, {});
        check(r.outcome == Outcome::Accepted && r.rows_right && !r.x.expired(), "1. host on time: window accepted, every row right, record clean");
    }
    // 2. a delayed worker past the bound at step 2 (before the plan): explicit failure, the stale rows are not accepted
    {
        Mapped m;
        HostPlan hp; hp.late_before_plan = 2;
        const Result r = run_window(m, 2, 6, hp);
        check(fail_closed(r), "2a. worker 1.5 s late at step 2: the window is not accepted with wrong rows");
        check(r.outcome == Outcome::Failed && r.x.site == kWaitPlan && r.x.step == 3, "2b. ... it fails with its own site (plan) and step (3 = ring of step 2)");
        check(r.drained && r.ms < 8000, "2c. ... and the GPU finishes once released (bounded)");
    }
    // 3. a delayed worker within the bound
    {
        Mapped m;
        HostPlan hp; hp.delay_ms_every = 40;
        const Result r = run_window(m, 3, 4, hp);
        check(r.outcome == Outcome::Accepted && r.rows_right, "3. worker 40 ms late at every step (the bound is 0.2 s or more): accepted, rows right");
    }
    // 4. cancel mid-window, then the same record serves the next window (slot reuse)
    {
        Mapped m;
        HostPlan hp; hp.cancel_after = 1;
        const Result r = run_window(m, 4, 6, hp);
        check(r.outcome == Outcome::Cancelled && r.drained, "4a. cancelled after step 1: the GPU drains within the bound");
        const Result r2 = run_window(m, 5, 6, {});
        check(r2.outcome == Outcome::Accepted && r2.rows_right && !r2.x.expired() && (g_legacy || r2.x.epoch == m.epoch),
              "4b. the next window on the same record: clean, its own epoch, rows right");
    }
    // 5. late expiry: the host checked a clean record, THEN was late; the check after the sync fails the window; next one clean
    {
        Mapped m;
        HostPlan hp; hp.late_after_check = 3;
        const Result r = run_window(m, 6, 5, hp);
        check(fail_closed(r), "5a. late after the host's check: the window is not accepted with wrong rows");
        check(r.outcome == Outcome::Failed && r.x.step == 4, "5b. ... it fails at a later check, naming step 4");
        const Result r2 = run_window(m, 7, 5, {});
        check(r2.outcome == Outcome::Accepted && r2.rows_right && !r2.x.expired(), "5c. the next window starts with a clean record and is right");
    }
    // 6. slow host at every step, and a dead host: bounded, explicit, no hang
    {
        Mapped m;
        HostPlan hp; hp.late_every_step = true;
        const Result r = run_window(m, 8, 4, hp);
        check(fail_closed(r) && r.outcome != Outcome::Accepted && r.ms < 12000, "6a. slow host at every step: explicit failure within a bounded time");
        Mapped d;
        HostPlan dead; dead.dead = true;
        const Result rd = run_window(d, 9, 4, dead);
        check(rd.outcome == Outcome::HostTimeout && rd.drained && (g_legacy || (rd.x.site == kWaitPlan && rd.x.step == 1)),
              "6b. dead host: the host's timeout ends it, the GPU finishes, the record names step 1 (plan)");
    }
    // 7. simultaneous queues: three windows at once on three records; two fail at different steps and sites, one is clean
    {
        Mapped a, b, c;
        Result ra, rb, rc;
        HostPlan pa; pa.late_before_plan = 1;
        HostPlan pb; pb.late_before_rows = 4;
        std::thread ta([&] { ra = run_window(a, 10, 6, pa); });
        std::thread tb([&] { rb = run_window(b, 11, 6, pb); });
        std::thread tc([&] { rc = run_window(c, 12, 6, {}); });
        ta.join(); tb.join(); tc.join();
        check(fail_closed(ra) && fail_closed(rb) && fail_closed(rc), "7a. three windows at once: none is accepted with wrong rows");
        check(ra.outcome == Outcome::Failed && ra.x.site == kWaitPlan && ra.x.step == 2, "7b. window A fails with its own site (plan) and step 2");
        check(rb.outcome == Outcome::Failed && rb.x.site == kWaitRows && rb.x.step == 5, "7c. window B fails with its own site (rows) and step 5");
        check(rc.outcome == Outcome::Accepted && rc.rows_right && !rc.x.expired(), "7d. window C beside them is clean and right");
    }
    // 8. the record helpers: arm clears, the first expiry keeps its place, the count goes on
    {
        volatile uint32_t rec[kWaitWords] = {7, 7, 7, 7};
        wait_record_arm(rec, 42);
        const WaitExpiry z = wait_record_read(rec);
        volatile uint32_t never = 0;
        bounded_wait(&never, 3, rec, kWaitRows, 9, Load{}, Store{}, [] {}, [](uint32_t v) { return v >= 1; });
        bounded_wait(&never, 3, rec, kWaitPlan, 11, Load{}, Store{}, [] {}, [](uint32_t v) { return v >= 1; });
        const WaitExpiry x = wait_record_read(rec);
        volatile uint32_t ready = 5;
        const bool ok_ready = bounded_wait(&ready, 0, rec, kWaitPcie, 12, Load{}, Store{}, [] {}, [](uint32_t v) { return v >= 5; });
        check(!z.expired() && z.epoch == 42 && z.count == 0 && x.site == kWaitRows && x.step == 9 && x.count == 2 && ok_ready &&
                  wait_record_read(rec).count == 2,
              "8. arm clears with the epoch; first expiry keeps site/step; count 2; a flag already up passes without a write");
    }
    std::printf(g_fail ? "window_wait_test: %d FAILED\n" : "window_wait_test: all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
