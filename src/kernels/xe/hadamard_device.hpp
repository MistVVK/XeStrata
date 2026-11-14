// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/hadamard_device.hpp - the in-place butterfly of one block-Hadamard transform in local memory, the
// one device part the rotated paths share: the APR experts' hadamard_rows (iq_kernels.cpp) and the KV trellis's
// rotation (kv_q4.cpp), the same natural (Sylvester) order and pairs as the CPU's hadamard_rotate (cpu/hadamard.hpp),
// so the rotated values are the same floats on every path.  The block is a power of two (256, 512, ...).
#pragma once

#include <sycl/sycl.hpp>

namespace strata::kernels::xe {

/// `buf[0..block)` transformed in place: `n` work-items of the group `g`, work-item `lid`.  The caller loads buf
/// before (the first stage's barrier covers the load) and reads it after a barrier following this.  No scale here:
/// the caller multiplies by 1 / sqrt(block), the float the CPU's hadamard_rotate computes (hadamard_rows does too).
template<class Group>
inline void hadamard_block(float* buf, int block, int lid, int n, Group g) {
    for (int len = 1; len < block; len <<= 1) {
        sycl::group_barrier(g);
        for (int p = lid; p < block / 2; p += n) {
            const int i = (p / len) * 2 * len + p % len;
            const float a = buf[i], c = buf[i + len];
            buf[i] = a + c;
            buf[i + len] = a - c;
        }
    }
}

}  // namespace strata::kernels::xe