// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/i8_means.hpp - the int8 attention's K means kept from a step to the next, the part the DiTs
// that run an image over several steps share (kernels::FlashAttnI8Stats).  Any mean leaves the softmax as it is (a
// row's scores all move by q . mean), so one made at an earlier step serves: only the keys' int8 rounding asks it be
// near, which is why a new image resets it.  A slot is a (block, context); the buffer [slots * C] is the caller's
// (zeroed at allocation).  A slot past the kept ones gives no mean (the kernel makes its own each call).
#pragma once

#include "strata/kernels/flash_attn.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::image {

struct I8Means {
    float* buf = nullptr;          ///< [slots * C], the caller's device buffer (zero until a step makes a mean)
    int64_t slots = 0, C = 0;
    std::vector<char> ready;       ///< a slot's mean was made this image (reset() on a new one)

    void init(float* b, int64_t n, int64_t c) {
        buf = b;
        slots = n;
        C = c;
        ready.assign((size_t) n, 0);
    }
    void clear() {
        buf = nullptr;
        slots = C = 0;
        ready.clear();
    }
    void reset() { std::fill(ready.begin(), ready.end(), 0); }

    /// The statistics of a slot: its V scales `sv` (null: the kernel reads V for them) and its kept K mean.  A slot
    /// past the kept ones (a third context) gives no mean: the kernel makes its own and keeps none.
    kernels::FlashAttnI8Stats stats(int64_t slot, const float* sv = nullptr) const {
        if (slot >= slots) return {sv, nullptr, false};
        return {sv, buf + slot * C, ready[(size_t) slot] != 0};
    }
    /// A step ran the slot's attention: its mean is made (a slot past the kept ones keeps none).
    void mark(int64_t slot) {
        if (slot < slots) ready[(size_t) slot] = 1;
    }
};

}  // namespace strata::image