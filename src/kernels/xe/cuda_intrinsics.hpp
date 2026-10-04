// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/cuda_intrinsics.hpp - CUDA's integer intrinsics as the Xe kernels ported from llama.cpp use them.
//
// __dp4a is SPIR-V's non-saturating 4x8-bit dot product (SPV_KHR_integer_dot_product, enabled for the device
// compile in CMakeLists.txt), which the B70 runs 4.2 times faster than the byte loop it replaces; CUDA's __dp4a
// does not saturate either, so the sum is the same integer.  The others are written out with their PTX meaning.
// Compiled for NVIDIA GPUs (__NVPTX__), __dp4a and __byte_perm are the PTX instructions themselves.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#if defined(__SYCL_DEVICE_ONLY__) && !defined(__NVPTX__)
extern SYCL_EXTERNAL int __spirv_SDotKHR(int a, int b, int packed_format);   // 0: PackedVectorFormat4x8Bit
#endif

namespace strata::kernels::xe {

inline int dp4a(int a, int b, int c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    int r;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
#elif defined(__SYCL_DEVICE_ONLY__)
    return c + __spirv_SDotKHR(a, b, 0);
#else
    for (int k = 0; k < 4; ++k) c += (int) (int8_t) (a >> (8 * k)) * (int) (int8_t) (b >> (8 * k));
    return c;
#endif
}

// __byte_perm(x, y, s): result byte i is byte ((s >> 4i) & 7) of the eight bytes y:x.  In 32-bit operations: the
// selector's bit 2 picks x or y, its low bits the byte.  A variable shift of the 64-bit y:x, as first written, costs
// several instructions on Xe, and the i-quant decoders call this per weight word (the B70's expert kernels were
// bound by these integer instructions, not by memory: bench/results/2026-10-02-xe-decode-gpu).
inline uint32_t byte_perm(uint32_t x, uint32_t y, uint32_t s) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    return __nvvm_prmt(x, y, s & 0x7777u);   // prmt's default mode; bit 3 of each selector (sign replication) unused
#else
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t sel = (s >> (4 * i)) & 7;
        const uint32_t src = (sel & 4) ? y : x;
        r |= ((src >> (8 * (sel & 3))) & 0xFF) << (8 * i);
    }
    return r;
#endif
}

// __vcmpne4: 0xFF in every byte where a and b differ.
// All four bytes at once: a byte of a ^ b is nonzero when its high bit is set or its low seven bits carry into it.
inline int vcmpne4(uint32_t a, uint32_t b) {
    const uint32_t x = a ^ b;
    const uint32_t t = (((x & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | x) & 0x80808080u;
    return (int) ((t << 1) - (t >> 7));                 // (t >> 7) * 0xFF without the multiply
}

// __vsub4: bytewise a - b, modulo 256.
// All four bytes at once (Hacker's Delight 2-18): the high bits are set aside so no borrow crosses a byte.
inline int vsub4(uint32_t a, uint32_t b) {
    return (int) (((a | 0x80808080u) - (b & 0x7F7F7F7Fu)) ^ ((a ^ ~b) & 0x80808080u));
}

// __vsubss4: bytewise signed subtraction, saturated to [-128, 127].  All four bytes at once: the wrapped difference
// (as vsub4), then the bytes that overflowed (a and b of different signs, and the difference's sign not a's) take
// -128 or 127 by a's sign.
inline int vsubss4(int ai, int bi) {
    const uint32_t a = (uint32_t) ai, b = (uint32_t) bi;
    const uint32_t d = ((a | 0x80808080u) - (b & 0x7F7F7F7Fu)) ^ ((a ^ ~b) & 0x80808080u);
    const uint32_t o = (a ^ b) & (a ^ d) & 0x80808080u;
    const uint32_t m = (o << 1) - (o >> 7);             // 0xFF in the bytes that overflowed
    const uint32_t sat = 0x7F7F7F7Fu + ((a & 0x80808080u) >> 7);
    return (int) ((d & ~m) | (sat & m));
}

}  // namespace strata::kernels::xe
