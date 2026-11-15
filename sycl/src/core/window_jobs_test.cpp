// sycl/src/core/window_jobs_test.cpp - the CPU jobs of a verify window (strata/core/window_jobs.hpp), no GPU, no model.
//
// The bug (a segfault in the CPU expert pool at --max-context 262144 with --layer-split): `job_of` is read at the start of
// every window and must be all -1. The old code cleared it by walking `ids` AFTER the pool ran, so a call that left early,
// or whose `ids` (host memory the GPU writes) changed meanwhile, left an entry set; the next window then added tokens to
// that stale job, `nt` passed MAXT and `nact[MAXT]` (which is `fmt`) was overwritten by an activation pointer.
//
// This file runs the engine's own entry loop (build_window_jobs) and WindowJobs; the snapshot of `ids` and the final
// release are repeated in Window::run, so the wiring inside expert_pool_dispatch_multi is not covered here.
#include "strata/core/window_jobs.hpp"

#include <cstdio>
#include <stdexcept>

using namespace strata::core;
using strata::kernels::cpu::ExpertJobMulti;
using strata::kernels::cpu::MAXT;

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-118s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

constexpr int kExperts = 512;
static const uint8_t kBlob[4] = {0, 1, 2, 3};
static const uint8_t* blob_of_expert(int64_t e) { return kBlob + (e % 4); }

static bool all_free(const std::vector<int16_t>& job_of) {
    for (int16_t v : job_of) if (v != -1) return false;
    return true;
}

// What one window does: copy ids (as expert_pool_dispatch_multi does), build the jobs, "run the pool" (the hook: the GPU
// rewrites the live ids meanwhile), then clean up the way the engine does. `stop_at_expert` makes the source refuse a blob.
struct Window {
    std::vector<int16_t>& job_of;
    std::vector<ExpertJobMulti>& jobs;
    int njobs = 0;
    WindowFailure failure;
    bool nt_overflow = false;
    Window(std::vector<int16_t>& j, std::vector<ExpertJobMulti>& js) : job_of(j), jobs(js) {}

    void run(int32_t* live_ids, int64_t n_tok, int64_t k, int64_t stop_at_expert, bool rewrite_ids_meanwhile, bool throw_after) {
        int32_t ids[128];
        for (int64_t i = 0; i < n_tok * k; ++i) ids[i] = live_ids[i];   // the snapshot
        int32_t kind[128];
        for (int64_t i = 0; i < n_tok * k; ++i) kind[i] = -1;           // every entry on the CPU
        njobs = 0;
        WindowJobs w(job_of, jobs, njobs);
        failure = build_window_jobs(
            w, ids, n_tok, k, kExperts, kind, [](int64_t) {}, [](int64_t) {},
            [&](int64_t e) -> const uint8_t* { return e == stop_at_expert ? nullptr : blob_of_expert(e); },
            [&](ExpertJobMulti& jb, int64_t, int64_t) {
                if (jb.nt >= MAXT) nt_overflow = true;      // the old rule would write act/out/nact[MAXT] (= fmt) here: do not
                else { jb.out[jb.nt] = nullptr; jb.nact[jb.nt] = nullptr; }
                ++jb.nt;
            });
        if (rewrite_ids_meanwhile)                          // the GPU runs ahead while the pool works
            for (int64_t i = 0; i < n_tok * k; ++i) live_ids[i] = (int32_t) ((live_ids[i] + 1 + i) % kExperts);
        if (throw_after) throw std::runtime_error("the call leaves early");
        w.release();
    }
};

int main() {
    // 1. ids rewritten while the pool works: the next window must start clean
    {
        std::vector<int16_t> job_of(kExperts, -1);
        std::vector<ExpertJobMulti> jobs(128);
        int32_t ids[20] = {50, 279, 296, 87, 405, 223, 8, 488, 299, 203, 279, 488, 361, 50, 296, 405, 60, 299, 119, 473};
        Window a(job_of, jobs);
        a.run(ids, 2, 10, -1, /*rewrite*/ true, false);
        check(all_free(job_of), "ids rewritten during the call: job_of is all -1 afterwards");
        int32_t ids2[20] = {50, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        Window b(job_of, jobs);
        b.run(ids2, 2, 10, -1, false, false);
        bool fresh = b.failure.status == WindowStatus::Ok && jobs[0].blob == blob_of_expert(50) && jobs[0].nt == 1;
        check(fresh, "the next window: expert 50 gets a NEW job with its own blob and nt = 1 (not the previous window's)");
    }
    // 2. one expert ten times in a window: never more than MAXT tokens in a job, fmt null even on a dirty jobs vector
    {
        std::vector<int16_t> job_of(kExperts, -1);
        std::vector<ExpertJobMulti> jobs(128);
        for (auto& j : jobs) { j.nt = 5; j.fmt = (const strata::kernels::cpu::NativeFmt*) 0x1; }   // a vector the last window left dirty
        int32_t ids[20];
        for (int i = 0; i < 20; ++i) ids[i] = (i < 10) ? 246 : 100 + i;
        Window a(job_of, jobs);
        a.run(ids, 2, 10, -1, false, false);
        bool within = true, fmt_null = true;
        for (int j = 0; j < a.njobs; ++j) { within = within && jobs[(size_t) j].nt <= MAXT; fmt_null = fmt_null && jobs[(size_t) j].fmt == nullptr; }
        check(!a.nt_overflow && within, "no job holds more than MAXT tokens (nact[MAXT] is fmt: it is never written)");
        int total = 0;
        for (int j = 0; j < a.njobs; ++j) total += jobs[(size_t) j].nt;
        check(total == 20 && jobs[0].nt == MAXT && jobs[1].nt == 2 && jobs[0].blob == jobs[1].blob, "all 20 entries land: expert 246's ten split 8 + 2 on one blob");
        check(fmt_null, "every job opened has fmt == nullptr and starts at nt == 0 (a dirty jobs vector is not inherited)");
    }
    // 2b. a stale entry the window did not open (job_of[e] points past the jobs opened so far): it is not trusted
    {
        std::vector<int16_t> job_of(kExperts, -1);
        std::vector<ExpertJobMulti> jobs(128);
        job_of[50] = 5;
        int32_t ids[2] = {50, 51};
        Window a(job_of, jobs);
        a.run(ids, 1, 2, -1, false, false);
        check(a.failure.status == WindowStatus::Ok && a.njobs == 2 && jobs[0].blob == blob_of_expert(50) && jobs[0].nt == 1,
              "a stale job_of entry past the opened jobs is ignored: expert 50 gets a new job, nt = 1");
    }
    // 3. a real early return (the source has no blob mid-window) and an exception: job_of is clean either way
    {
        std::vector<int16_t> job_of(kExperts, -1);
        std::vector<ExpertJobMulti> jobs(128);
        int32_t ids[20] = {50, 279, 296, 87, 405, 223, 8, 488, 299, 203, 279, 488, 361, 50, 296, 405, 60, 299, 119, 473};
        Window a(job_of, jobs);
        a.run(ids, 2, 10, /*no blob for*/ 8, false, false);
        check(a.failure.status == WindowStatus::NoBlob && a.failure.expert == 8, "no blob for expert 8: the loop stops there");
        // the engine's early return skips the end-of-call walk entirely, so only the scope exit can clean it up
        std::vector<int16_t> job_of2(kExperts, -1);
        std::vector<ExpertJobMulti> jobs2(128);
        {
            int n = 0;
            WindowJobs w(job_of2, jobs2, n);
            int32_t kind[20]; for (int i = 0; i < 20; ++i) kind[i] = -1;
            const WindowFailure f = build_window_jobs(w, ids, 2, 10, kExperts, kind, [](int64_t) {}, [](int64_t) {},
                [&](int64_t e) -> const uint8_t* { return e == 8 ? nullptr : blob_of_expert(e); },
                [](ExpertJobMulti& jb, int64_t, int64_t) { ++jb.nt; });
            (void) f;
        }   // leaving the scope on the failure path is the engine's `return`
        check(all_free(job_of2), "after the failed call (a `return`) job_of is all -1");
        std::vector<int16_t> jo(kExperts, -1);
        std::vector<ExpertJobMulti> js(128);
        Window b(jo, js);
        bool thrown = false;
        try { b.run(ids, 2, 10, -1, false, true); } catch (const std::exception&) { thrown = true; }
        check(thrown && all_free(jo), "an exception out of the call: job_of is all -1");
    }
    std::printf(g_fail ? "window_jobs_test: %d FAILED\n" : "window_jobs_test: all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
