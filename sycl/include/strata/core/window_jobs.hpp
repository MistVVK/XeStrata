// sycl/include/strata/core/window_jobs.hpp - the CPU jobs of one verify window (expert_pool_dispatch_multi).
//
// One job per expert the CPU runs: its blob plus up to MAXT tokens' activations and output rows. `job_of[e]` is the job
// an expert already has in this window, -1 when it has none, and it MUST be all -1 whenever a call starts: the next
// window reads it before it writes it. So the entries a call opens are remembered here and cleared on EVERY way out of
// the call (a return, an exception, a window whose `ids` the GPU rewrote meanwhile), not by walking `ids` again at the
// end. A job that already holds MAXT tokens is closed and a new one opens on the same blob, so `act`/`out`/`nact`
// (MAXT entries each, `fmt` right after them) are never written past.
#pragma once

#include "strata/kernels/cpu/pool.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

namespace strata::core {

/// Room for the experts one window can open a job for (its entries; expert_source.cpp's kMaxWindowEntries is the same number).
constexpr int kWindowJobsMax = 128;

class WindowJobs {
public:
    /// `job_of` all -1 on entry; `jobs` with room for one job per entry of the window; `njobs` counts the jobs opened.
    WindowJobs(std::vector<int16_t>& job_of, std::vector<strata::kernels::cpu::ExpertJobMulti>& jobs, int& njobs)
        : job_of_(job_of), jobs_(jobs), njobs_(njobs) {}
    WindowJobs(const WindowJobs&) = delete;
    WindowJobs& operator=(const WindowJobs&) = delete;
    ~WindowJobs() { release(); }

    /// The job expert `e` adds an entry to: its open one, or a new one on the blob `blob_of()` returns (asked only when a
    /// new job is needed). nullptr when the source has no blob for it. The entry is yours to fill at `nt`, then `++nt`.
    template <class BlobFn>
    strata::kernels::cpu::ExpertJobMulti* slot(int64_t e, BlobFn&& blob_of) {
        using strata::kernels::cpu::ExpertJobMulti;
        assert(e >= 0 && (size_t) e < job_of_.size());
        int16_t& jo = job_of_[(size_t) e];
        if (jo >= njobs_) jo = -1;   // an entry this window did not open (the invariant broke): never trust it
        if (jo < 0) {
            const uint8_t* b = blob_of();
            if (b == nullptr) return nullptr;
            assert(nopened_ < kWindowJobsMax);
            opened_[(size_t) nopened_++] = (int32_t) e;
            jo = open(b);
        } else if (jobs_[(size_t) jo].nt >= strata::kernels::cpu::MAXT) {
            jo = open(jobs_[(size_t) jo].blob);   // full: the next entries go to a fresh job on the same blob
        }
        return &jobs_[(size_t) jo];
    }

    /// Clear what this call opened. Idempotent; the destructor calls it.
    void release() {
        for (int i = 0; i < nopened_; ++i) job_of_[(size_t) opened_[(size_t) i]] = -1;
        nopened_ = 0;
    }

private:
    int16_t open(const uint8_t* blob) {
        assert((size_t) njobs_ < jobs_.size());
        const int16_t j = (int16_t) njobs_++;
        strata::kernels::cpu::ExpertJobMulti& nj = jobs_[(size_t) j];
        nj.blob = blob;
        nj.nt = 0;
        nj.fmt = nullptr;
        return j;
    }
    std::vector<int16_t>& job_of_;
    std::vector<strata::kernels::cpu::ExpertJobMulti>& jobs_;
    int& njobs_;
    std::array<int32_t, kWindowJobsMax> opened_{};   // no allocation on the token path
    int nopened_ = 0;
};

enum class WindowStatus { Ok, IdOutOfRange, NoBlob };
struct WindowFailure { WindowStatus status = WindowStatus::Ok; int64_t expert = -1; };

/// The entry loop of a verify window, for the engine and for its test. `ids` are the window's n_tok * k experts and
/// `kind[i] >= 0` marks an entry something else serves (a GPU tier, a peer, a remote: `on_other(i)`). The others go to a
/// CPU job: `on_refused(i)` counts them, `blob_of(e)` is asked for a new job's blob, `fill(job, t, i)` writes the entry at
/// `job.nt` and bumps it. Stops at the first id out of range or expert with no blob. Precondition: n_tok * k <= kWindowJobsMax.
template <class OtherFn, class RefusedFn, class BlobFn, class FillFn>
WindowFailure build_window_jobs(WindowJobs& w, const int32_t* ids, int64_t n_tok, int64_t k, int64_t n_expert, const int32_t* kind,
                                OtherFn&& on_other, RefusedFn&& on_refused, BlobFn&& blob_of, FillFn&& fill) {
    assert(n_tok * k <= kWindowJobsMax);
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int64_t e = ids[i];
            if (e < 0 || e >= n_expert) return {WindowStatus::IdOutOfRange, e};
            if (kind[i] >= 0) { on_other(i); continue; }
            on_refused(i);
            strata::kernels::cpu::ExpertJobMulti* const jp = w.slot(e, [&] { return blob_of(e); });
            if (jp == nullptr) return {WindowStatus::NoBlob, e};
            fill(*jp, t, i);
        }
    return {};
}

}  // namespace strata::core
