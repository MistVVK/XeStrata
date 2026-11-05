// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/kv_q4_parts.hpp - a Q4_0 K/V block's quantization and store, out of kv_q4.cpp (the Xe port of
// Strata's src/kernels/cuda/kv_q4.cu) for the cache's one append (kv_q8.cpp's kv_append_cells).
#pragma once

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels::kvq4 {

// One 32-value group: the signed value of largest magnitude (ties to the larger value) sets d = max / -8, and lanes
// 0..15 pack their code with lane + 16's into one byte.
inline uint16_t q4_group(const sycl::sub_group& sg, float x, uint8_t& byte) {
    float amax = sycl::fabs(x), mval = x;
    for (int o = 16; o > 0; o >>= 1) {
        const float a = sycl::permute_group_by_xor(sg, amax, o);
        const float v = sycl::permute_group_by_xor(sg, mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const int q = (int) (x * id + 8.5f);                   // __float2int_rz: truncation toward zero
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    const uint8_t qhi = sycl::permute_group_by_xor(sg, qc, 16);   // lanes 0..15 read lane + 16, as __shfl_down(16)
    byte = (uint8_t) (qc | (qhi << 4));
    return f16_from_f32(d);
}

inline void q4_store(uint8_t* pool, long long row, int b, int lane, uint16_t d, uint8_t byte) {
    block_q4_0* blk = reinterpret_cast<block_q4_0*>(pool + row * (long long) sizeof(block_q4_0) * 8) + b;
    if (lane == 0) blk->d = d;
    if (lane < 16) blk->qs[lane] = byte;
}

}  // namespace strata::kernels::kvq4
