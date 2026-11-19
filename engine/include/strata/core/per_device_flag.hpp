// include/strata/core/per_device_flag.hpp - a yes / no answer that depends on the card, remembered per device.
//
// A layer split can run its stages on cards of two architectures (an RX 7900 XTX gfx1100 beside an RX 9070 XT gfx1201, #4431).  A choice of kernel that was
// made with `static const bool x = [] { ...the current device's arch... }();` is made once, by whichever stage reaches it first, and every other card gets
// that answer: a gfx11 matrix-core kernel launched on the gfx12 card, which has no code for it ("no kernel image is available for execution on the device",
// reported late by the next error check: gr_mix_r).  This asks the probe once per device ordinal instead.  Header only, no GPU: the probe is the caller's.
#pragma once

#include <atomic>

namespace strata::core {

template <int N = 64>
class PerDeviceFlag {
public:
    /// The answer for device `dev`: `probe(dev)` is called the first time (a few threads asking at once may all call it; the answer is the same), later calls
    /// read the stored one.  A device ordinal outside 0..N-1 is not remembered: the probe runs every time.
    template <class Probe>
    bool get(int dev, Probe&& probe) {
        if (dev < 0 || dev >= N) return probe(dev);
        signed char v = v_[dev].load(std::memory_order_relaxed);
        if (v == 0) {
            v = probe(dev) ? 2 : 1;
            v_[dev].store(v, std::memory_order_relaxed);
        }
        return v == 2;
    }

private:
    std::atomic<signed char> v_[N] = {};   // 0 not asked yet, 1 no, 2 yes
};

}  // namespace strata::core
