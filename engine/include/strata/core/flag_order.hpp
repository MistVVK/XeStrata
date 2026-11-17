// include/strata/core/flag_order.hpp - flag B is raised behind the copies queued before it (PR #5096, @AlanJager).
//
// Flag B tells the graph that a layer's PCIe experts have landed in staging.  It is one counter that only rises, and the
// graph waits for "flag B >= layer + 1".  A non-empty plan raises it from a host function on the copy stream, so it
// rises after that plan's copies.  An empty plan used to raise it on the spot, even when the copy stream still held an
// earlier layer's (or the other token group's) uploads: the counter jumped past a layer whose bytes had not landed, and
// the graph read a half-written staging slot.  raise_flag_b_ordered() keeps the counter behind every copy queued on the
// stream, and raises it directly only when nothing was queued in this window.  The plain raise is shared with Verifier.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#if defined(_WIN32)
#include <intrin.h>
#endif

namespace strata::core {

/// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
inline void flag_raise_max(uint32_t* flag, uint32_t value) {
#if defined(_WIN32)
    volatile long* f = (volatile long*) flag;
    long cur = *f;
    while ((uint32_t) cur < value) {
        const long prev = _InterlockedCompareExchange(f, (long) value, cur);
        if (prev == cur) break;
        cur = prev;
    }
#else
    uint32_t cur = __atomic_load_n(flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n(flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
#endif
}

/// The argument of the host function (it must outlive the call: it lives in the Verifier's table).
struct FlagRaise {
    uint32_t* flag;
    uint32_t value;
};

/// Raise `flag` to `value` once every copy queued so far on `copy` has landed.  `copies_queued` is false when nothing was
/// queued on the stream since it was last drained: then there is nothing to wait for and the flag rises at once.
/// Returns the launch status (cudaSuccess when the flag was raised directly).  On a failure the flag has NOT been raised:
/// the caller must not release the consumers (Verifier::fetch_dma stops the engine).
inline cudaError_t raise_flag_b_ordered(bool copies_queued, cudaStream_t copy, FlagRaise* slot, uint32_t* flag, uint32_t value) {
    if (!copies_queued || copy == nullptr) {
        flag_raise_max(flag, value);
        return cudaSuccess;
    }
    slot->flag = flag;
    slot->value = value;
    return cudaLaunchHostFunc(copy, [](void* p) {
        FlagRaise* s = (FlagRaise*) p;
        flag_raise_max(s->flag, s->value);
    }, slot);
}

}  // namespace strata::core
