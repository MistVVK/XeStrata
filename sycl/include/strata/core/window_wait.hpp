// sycl/include/strata/core/window_wait.hpp - a GPU wait on a host flag that runs out is an error of its window (#5160, #5992, #6016).
//
// THE CONTRACT.  In a verify window (and the single-token session loop) the GPU rings a step, writes that step's payload (activations,
// expert ids, routing weights) into mapped host memory and then waits for the host: the plan (flag A), the PCIe share (flag B) and the
// CPU's rows (flag M).  The payload belongs to the host from the ring until it raises flag M; the GPU writes that region again only after
// it passed flag M.  Every device wait is bounded (sycl_doorbell.hpp, spin_max) because an unbounded spin wedges an xe card.  Before this
// header a wait that ran out simply returned: the GPU used the previous layer's plan and rows (a wrong token, "!!!!" from then on,
// #5160) and rang the next step over a payload the CPU pool was still reading (#6016's crash at 262144).
//
// So a wait that runs out writes a RECORD, and the host treats a written record as the failure of the window:
//   - one record per verifier (and per session), never per device: two windows in flight on one card (pipelined parity verifiers,
//     --split-device 0, a batch window beside a single one) each read only their own;
//   - only that verifier's wait kernels write it, one at a time on its in-order queue, so plain uncached loads and stores are enough;
//     the first expiry keeps its site and step, every expiry bumps the count;
//   - the host arms it (all zero, a new epoch) where it resets the flags for a new window, i.e. after the previous window completed;
//   - the host reads it after every CPU step (before it raises flag M) and after the window's final sync; a written record fails the
//     window: its tokens are never returned or committed.
//
// The wait itself is a template on its loads and stores so the device kernels (uncached system loads / stores) and the CPU fixture
// (window_wait_test.cpp, std::atomic) run the same code.
#pragma once

#include <cstdint>

namespace strata::core {

/// Which wait ran out (WaitRecord::site).  0 = none.
enum WaitSite : uint32_t {
    kWaitNone = 0,
    kWaitPlan = 1,         ///< flag A: the host's plan for the GPU's share of this step
    kWaitPcie = 2,         ///< flag B: the PCIe share staged
    kWaitRows = 3,         ///< flag M: the CPU's rows in the mapped buffer
    kWaitDevicePlan = 4,   ///< a device-plan wait (wait_flag_ge_or) on any of the three
    kWaitFirst = 5,        ///< flag 1 of an all-resident window (the PLE rows)
    kWaitDoorbell = 6,     ///< the single-token session loop's doorbell
};

/// Words of the record, in mapped host memory (4 x uint32).  The device writes STEP before SITE, so a host that sees a site sees its step.
enum WaitWord : int { kWordSite = 0, kWordStep = 1, kWordEpoch = 2, kWordCount = 3, kWaitWords = 4 };

inline const char* wait_site_name(uint32_t site) {
    switch (site) {
        case kWaitPlan: return "the plan for its GPU share (flag A)";
        case kWaitPcie: return "the PCIe share (flag B)";
        case kWaitRows: return "the CPU experts' rows (flag M)";
        case kWaitDevicePlan: return "a device-plan wait";
        case kWaitFirst: return "the window's first flag (PLE rows)";
        case kWaitDoorbell: return "the CPU experts (doorbell)";
        default: return "an unknown wait";
    }
}

/// THE WAIT, device and fixture.  `ready(v)` says whether the flag value v lets the GPU go on (>= value, or == want for the doorbell).
/// Spins at most `spin_max` reads.  Returns true when the flag arrived; otherwise writes the record (if any) and returns false.  The
/// caller goes on either way (a bounded wait never hangs the card): a written record is what makes the host discard the window.
template <class Load, class Store, class Pause, class Ready>
inline bool bounded_wait(const volatile uint32_t* flag, uint32_t spin_max, volatile uint32_t* rec, uint32_t site, uint32_t step,
                         Load&& load, Store&& store, Pause&& pause, Ready&& ready) {
    for (uint32_t spin = 0; spin < spin_max; ++spin) {
        if (ready(load(flag))) return true;
        pause();
    }
    if (ready(load(flag))) return true;   // the last read after the bound: arrived just in time
    if (rec != nullptr) {
        if (load(rec + kWordSite) == 0u) {   // the first expiry of this window keeps its place
            store(rec + kWordStep, step);
            store(rec + kWordSite, site);
        }
        store(rec + kWordCount, load(rec + kWordCount) + 1u);
    }
    return false;
}

/// Host side: what a record says.
struct WaitExpiry {
    uint32_t site = 0, step = 0, epoch = 0, count = 0;
    bool expired() const { return site != 0; }
};

/// Host side: arm the record for a new window (all zero, the window's epoch).  Only when no kernel of the previous window can still
/// write it: the caller has synchronised that window or drained it.
inline void wait_record_arm(volatile uint32_t* rec, uint32_t epoch) {
    if (rec == nullptr) return;
    rec[kWordSite] = 0u;
    rec[kWordStep] = 0u;
    rec[kWordCount] = 0u;
    rec[kWordEpoch] = epoch;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

/// Host side: read the record (site first; the device wrote it last).
inline WaitExpiry wait_record_read(const volatile uint32_t* rec) {
    WaitExpiry x;
    if (rec == nullptr) return x;
    x.site = rec[kWordSite];
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    x.step = rec[kWordStep];
    x.epoch = rec[kWordEpoch];
    x.count = rec[kWordCount];
    return x;
}

}  // namespace strata::core

namespace strata::kernels {
// The launchers with a record (sycl/src/kernels/cuda/*.dp.cpp).  The three-argument forms of the shared headers stay and pass none.
void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream, uint32_t* rec, uint32_t site);
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream, uint32_t* rec, uint32_t site);
void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream, uint32_t* rec);
}  // namespace strata::kernels
