// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/iq_bits.hpp - the i-quant decoders' word handling, shared by the dot products (iq_kernels.cpp) and
// the int8 matrix products (iq_mmq.cpp): unaligned words of a block, the grids' value pairs, the sign bytes, IQ4's
// 16-entry tables and Q2_0's codes.  Transcribed from llama.cpp (ggml/src/ggml-cuda/vecdotq.cuh; MIT license,
// third_party/main/ggml/LICENSE, quoted in iq_kernels.cpp) through Strata's CUDA version.
#pragma once

#include "cuda_intrinsics.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace strata::kernels::iq_bits {

using xe::byte_perm;

inline int get_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    return (int) ((uint32_t) x16[2 * i32 + 0] | ((uint32_t) x16[2 * i32 + 1] << 16));
}
inline int get_int_b4(const void* x, int i32) { return ((const int*) x)[i32]; }
// Two consecutive ints at a two-byte aligned address (get_int_b2 at i32 and i32 + 1): the aligned words around them,
// joined by shifts, instead of four 16-bit loads each with its own address.
struct U2 { uint32_t x, y; };
inline U2 get_int2_b2(const void* x, int i32) {
    const uintptr_t a = (uintptr_t) ((const uint8_t*) x + (size_t) 4 * i32);
    const uint32_t* w = (const uint32_t*) (a & ~(uintptr_t) 3);
    const uint32_t w0 = w[0], w1 = w[1];
    if ((a & 2) == 0) return {w0, w1};
    const uint32_t w2 = w[2];
    return {(w0 >> 16) | (w1 << 16), (w1 >> 16) | (w2 << 16)};
}
inline uint8_t byte_of(uint32_t v, int i) { return (uint8_t) (v >> (8 * i)); }
inline uint16_t half_of(uint32_t v, int i) { return (uint16_t) (v >> (16 * i)); }
// The 7-bit signs with their parity as bit 7 (CUDA's unpack_ksigns, one byte instead of four copies).
inline uint32_t ksigns_byte(uint8_t v) { return (uint32_t) (v ^ (sycl::popcount((uint32_t) v) & 1) << 7) & 0xFF; }
struct Int2 { int x, y; };
inline Int2 get_int_from_table_16(int q4, const int8_t* table) {
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(table);
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210 | (((uint32_t) q4 & 0x88888888) >> 1);
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = byte_perm(table32[0], table32[1], (uint32_t) q4 >> shift);
        const uint32_t high = byte_perm(table32[2], table32[3], (uint32_t) q4 >> shift);
        tmp[i] = byte_perm(low, high, sel >> shift);
    }
    return {(int) byte_perm(tmp[0], tmp[1], 0x6420), (int) byte_perm(tmp[0], tmp[1], 0x7531)};
}
inline Int2 grid_pair(const uint64_t* grid, int index) {
    const uint64_t g = grid[index];
    return {(int) (uint32_t) g, (int) (uint32_t) (g >> 32)};
}
// four sign bits as a 0xFF byte mask each
inline uint32_t sign_mask(uint32_t nib) { return (((nib & 0xF) * 0x00204081u) & 0x01010101u) * 0xFFu; }

// Q2_0's four 2-bit codes of a byte (the lowest first) as four signed bytes code - 1: the bytes CUDA's __byte_perm
// table 0x020100FF gives.  The codes are spread to the low bits of the four bytes with two shift-and-mask steps, and
// each byte has 1 taken off without a borrow crossing bytes (its high bit set first, flipped back after).  On Xe the
// __byte_perm translation was about 30 integer instructions a call, and the expert down projections ran out of
// integer issue (VTune: ALU1 72% busy at 95 GB/s).
inline uint32_t q2_0_bytes(uint32_t b) {
    uint32_t t = (b | (b << 12)) & 0x000F000Fu;
    t = (t | (t << 6)) & 0x03030303u;
    return ((t | 0x80808080u) - 0x01010101u) ^ 0x80808080u;
}

}  // namespace strata::kernels::iq_bits
