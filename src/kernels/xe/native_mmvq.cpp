// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/native_mmvq.cpp - see include/strata/kernels/native_mmvq.hpp; the Xe port of
// Strata's src/kernels/cuda/native_mmvq.cu.
//
// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h, through the CUDA version.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// The CUDA file keeps separate ncols = 1 kernels and a multi-column kernel built to be bitwise equal to them; here
// the multi-column kernel with NCOLS = 1 and the ncols = 1 layout (4 warps, 1 row, or 4 rows for small K) is the
// single-column path, so both share one set of dot-product traits.  A warp is a sub-group of 32.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_act.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "cuda_intrinsics.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_native_mmvq
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeindex>

namespace strata::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int MAX_NCOLS = 8;

using half = sycl::half;
using half2 = sycl::half2;

struct Q5KBlock { half2 dm; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct Q81Block { half2 ds; int8_t qs[32]; };
struct Q20Block { half d; uint8_t qs[16]; };
struct Q3KBlock { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; half d; };
struct Q2KBlock { uint8_t scales[16]; uint8_t qs[64]; half2 dm; };
struct IQ4XSBlock { half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };
struct Q4KBlock { half2 dm; uint8_t scales[12]; uint8_t qs[128]; };
struct Q6KBlock { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; };
struct Q40Block { half d; uint8_t qs[16]; };
struct Q50Block { half d; uint8_t qh[4]; uint8_t qs[16]; };
struct Q80Block { half d; int8_t qs[32]; };
struct IQ4NLBlock { half d; uint8_t qs[16]; };
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(Q2KBlock) == 84 && alignof(Q2KBlock) == 4 && offsetof(Q2KBlock, qs) == 16 &&
              offsetof(Q2KBlock, dm) == 80);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

inline int dp4a(int a, int b, int c) { return xe::dp4a(a, b, c); }
inline int byte_perm(uint32_t x, uint32_t y, uint32_t s) { return (int) xe::byte_perm(x, y, s); }
using xe::vsubss4;
inline float low(const half2& v) { return (float) v[0]; }
inline float high(const half2& v) { return (float) v[1]; }
inline int load_int(const void* p, int i) { return static_cast<const int*>(p)[i]; }

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
inline int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    return (int) ((uint32_t) x[(size_t) 2 * i32] | ((uint32_t) x[2 * i32 + 1] << 16));
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup.
alignas(4) const int8_t iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};
struct Int2 { int x, y; };
inline Int2 iq4_table_lookup(int q4) {
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | (((uint32_t) q4 & 0x88888888) >> 1);
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t lo = (uint32_t) byte_perm(table32[0], table32[1], (uint32_t) q4 >> shift);
        const uint32_t hi = (uint32_t) byte_perm(table32[2], table32[3], (uint32_t) q4 >> shift);
        tmp[i] = (uint32_t) byte_perm(lo, hi, low_high_selection_indices >> shift);
    }
    return {byte_perm(tmp[0], tmp[1], 0x6420), byte_perm(tmp[0], tmp[1], 0x7531)};
}

// ---------------------------------------------------------------- the pinned dot expressions
// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
inline float q5_q8_dot_impl(const int* vl, const int* vh, const int* u, const uint8_t* sc, const uint8_t* m,
                            const half2& dm5, const float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = dp4a(v0i, u[(size_t) 2 * i], dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = dp4a(0x01010101, u[(size_t) 2 * i], dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (float) (dot1 * sc[i]);
        sumf_m += d8[i] * (float) (dot2 * m[i]);
    }
    return low(dm5) * sumf_d - high(dm5) * sumf_m;
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
inline float q4_q8_dot_impl(const int* v, const int* u, const uint8_t* sc, const uint8_t* m, const half2& dm4,
                            const float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = dp4a(v1i, u[2 * i + 1], dp4a(v0i, u[(size_t) 2 * i], 0));
        const int dot2 = dp4a(0x01010101, u[2 * i + 1], dp4a(0x01010101, u[(size_t) 2 * i], 0));
        sumf_d += d8[i] * (float) (dot1 * sc[i]);
        sumf_m += d8[i] * (float) (dot2 * m[i]);
    }
    return low(dm4) * sumf_d - high(dm4) * sumf_m;
}

// vec_dot_q2_K_q8_1_impl_mmvq: a 4-bit scale and a 4-bit min a 16 values, the min against the Q8_1 values' sum by
// DP4A with the min in every byte
inline float q2_q8_dot_impl(int v, const int* u, const uint8_t* scales, const half2& dm2, const float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int sc = scales[2 * (ptrdiff_t) i];
        const int vi = (v >> (2 * i)) & 0x03030303;
        sumf_d += d8[i] * (float) (dp4a(vi, u[i], 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += d8[i] * (float) dp4a(m, u[i], 0);
    }
    return low(dm2) * sumf_d - high(dm2) * sumf_m;
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
inline float q6_q8_dot_impl(int vl, int vh, const int* u, const int8_t* scales, float d, const float* d8) {
    float sumf = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[(size_t) 4 * i];  // NOLINT(bugprone-signed-char-misuse): Q6_K scales are signed
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (float) (dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
inline float small_q8_dot(const Q40Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = dp4a(vi0, load_int(x->qs, iqs + i), sumi);
        sumi = dp4a(vi1, load_int(x->qs, iqs + i + 4), sumi);
    }
    const float d = (float) w->d;
    return d * ((float) sumi * low(x->ds) - 4 * high(x->ds));
}
inline float small_q8_dot(const Q50Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = dp4a(vi0, load_int(x->qs, iqs + i), sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = dp4a(vi1, load_int(x->qs, iqs + i + 4), sumi);
    }
    const float d = (float) w->d;
    return d * ((float) sumi * low(x->ds) - 8 * high(x->ds));
}
inline float small_q8_dot(const Q80Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
    for (int i = 0; i < 2; ++i) sumi = dp4a(load_int_b2(w->qs, iqs + i), load_int(x->qs, iqs + i), sumi);
    const float d0 = (float) w->d;
    const float d1 = low(x->ds);
    return d0 * d1 * float(sumi);
}
inline float small_q8_dot(const IQ4NLBlock* w, const Q81Block* x, int iqs) {
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        const Int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        sumi = dp4a(v.x, load_int(x->qs, iqs + i), sumi);
        sumi = dp4a(v.y, load_int(x->qs, iqs + i + 4), sumi);
    }
    return (float) w->d * low(x->ds) * (float) sumi;
}

// ---------------------------------------------------------------- per-format iteration traits
// `load` is everything that depends only on the weight block; `apply` is the activation loads and the pinned
// expression, once per column.  Thread-to-block mapping, blocks per iteration and the small-K rule are the
// CUDA ncols = 1 kernels'.
inline void unpack_k_scales(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(scales8);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}

struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int vl[2], vh[2]; uint8_t sc[4]; half2 dm; int bq8_offset; };
    static W load(const Block* bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const uint8_t* ql = bq5->qs + (size_t) 16 * r.bq8_offset + (size_t) 4 * ((iqs / 2) % 4);
        const uint8_t* qh = bq5->qh + (size_t) 4 * ((iqs / 2) % 4);
        r.vl[0] = load_int(ql, 0);
        r.vl[1] = load_int(ql, 4);
        r.vh[0] = load_int(qh, 0) >> r.bq8_offset;
        r.vh[1] = load_int(qh, 4) >> r.bq8_offset;
        uint16_t aux[2];
        unpack_k_scales(bq5->scales, r.bq8_offset, aux);
        for (int k = 0; k < 4; ++k) r.sc[k] = (uint8_t) (aux[k / 2] >> (8 * (k % 2)));
        r.dm = bq5->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* bq8, int iqs) {
        int u[4];
        float d8[2];
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = low(bq8i->ds);
            u[(size_t) 2 * i] = load_int(bq8i->qs, (iqs / 2) % 4);
            u[2 * i + 1] = load_int(bq8i->qs, (iqs / 2) % 4 + 4);
        }
        return q5_q8_dot_impl(r.vl, r.vh, u, r.sc, r.sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint8_t sc[4]; half2 dm; int bq8_offset; };
    static W load(const Block* bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const uint8_t* ql = bq4->qs + (size_t) 16 * r.bq8_offset + (size_t) 4 * ((iqs / 2) % 4);
        r.v[0] = load_int(ql, 0);
        r.v[1] = load_int(ql, 4);
        uint16_t aux[2];
        unpack_k_scales(bq4->scales, r.bq8_offset, aux);
        for (int k = 0; k < 4; ++k) r.sc[k] = (uint8_t) (aux[k / 2] >> (8 * (k % 2)));
        r.dm = bq4->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* bq8, int iqs) {
        int u[4];
        float d8[2];
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = low(bq8i->ds);
            u[(size_t) 2 * i] = load_int(bq8i->qs, (iqs / 2) % 4);
            u[2 * i + 1] = load_int(bq8i->qs, (iqs / 2) % 4 + 4);
        }
        return q4_q8_dot_impl(r.v, u, r.sc, r.sc + 2, r.dm, d8);
    }
};
// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.  The weight block is only 2-byte
// aligned, so qs is loaded as int16_t, unlike the naturally 4-byte aligned activation codes.
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    static W load(const Block* w, int iqs) {
        W r;
        r.d2 = (float) w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + (size_t) iqs * 4;
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const uint32_t qe = (uint32_t) byte_perm(0x020100ff, 0x020100ff, (uint32_t) (q >> 0));
            const uint32_t qo = (uint32_t) byte_perm(0x020100ff, 0x020100ff, (uint32_t) (q >> 2));
            r.qx[j] = byte_perm(qe, qo, 0x5140);
            r.qy[j] = byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    static float apply(const W& r, const Q81Block* x, int iqs) {
        const Q81Block* chunk = x + iqs;
        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            sumi = dp4a(load_int(chunk->qs, j * 2), r.qx[j], sumi);
            sumi = dp4a(load_int(chunk->qs, j * 2 + 1), r.qy[j], sumi);
        }
        return r.d2 * low(chunk->ds) * (float) sumi;
    }
};
// The signed values and the scales of a lane's Q3_K chunk are worked out once in load() and reused for every column
// (upstream 195d3791: the multi-column Q3_K kernels were 14% of an RTX A4000's GPU time on the Q2_0 pack); apply() is
// the arithmetic of pinned vec_dot_q3_K_q8_1 in its order, so the output is the same bits.
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    static int kqs(int tid) { return tid % 16; }
    struct W { int vi[4], sc[4]; float d; int bq8_offset; };
    static W load(const Block* w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = (float) w->d;
        const int vl = load_int_b2(w->qs, iqs);
        const int vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        for (int i = 0; i < 4; ++i) {
            const int isc = scale_offset + 2 * i;
            const int sc_low = (w->scales[isc % 8] >> (4 * (isc / 8))) & 0xf;
            const int sc_high = ((w->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
            r.sc[i] = (sc_low | sc_high) - 32;
            r.vi[i] = vsubss4((vl >> (2 * i)) & 0x03030303, ((vh >> i) << 2) & 0x04040404);
        }
        return r;
    }
    static float apply(const W& r, const Q81Block* x, int iqs) {
        float sumf = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const int u = load_int(x[r.bq8_offset + i].qs, iqs % 8);
            sumf += low(x[r.bq8_offset + i].ds) * (float) (dp4a(r.vi[i], u, 0) * r.sc[i]);
        }
        return r.d * sumf;
    }
};
// vec_dot_q2_K_q8_1: QI2_K = 16 lanes a block, each one 32-bit word of qs against four Q8_1 blocks
struct Q2KTraits {
    using Block = Q2KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    static int kqs(int tid) { return tid % 16; }
    struct W { int v; half2 dm; const uint8_t* scales; int bq8_offset; };
    static W load(const Block* w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scales = w->scales + (iqs - iqs % 8 + (iqs % 8) / 4);
        r.v = load_int(w->qs, iqs);
        r.dm = w->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* x, int iqs) {
        int u[4];
        float d8[4];
        for (int i = 0; i < 4; ++i) {
            u[i] = load_int(x[r.bq8_offset + i].qs, iqs % 8);
            d8[i] = low(x[r.bq8_offset + i].ds);
        }
        return q2_q8_dot_impl(r.v, u, r.scales, r.dm, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    static W load(const Block* w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = (float) w->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* x, int iqs) {
        int u[2];
        float d8[2];
        for (int i = 0; i < 2; ++i) {
            u[i] = load_int(x[r.bq8_offset + 2 * i].qs, iqs % 8);
            d8[i] = low(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
// kNativeQ6KRows: Q6KTraits' loads from the row's split arrays, every 32-bit load aligned; the same values reach
// the same dot, so the results are Q6KTraits' bit for bit.
struct Q6KRowTraits : Q6KTraits {
    static W load_row(const uint8_t* weights, int row, int kbx, int bpr, int iqs) {
        const uint8_t* base = weights + std::size_t(row) * bpr * sizeof(Q6KBlock);
        const uint8_t* ql = base + std::size_t(kbx) * 128;
        const uint8_t* qh = base + std::size_t(bpr) * 128 + std::size_t(kbx) * 64;
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int(ql, iqs);
        r.vh = load_int(qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = reinterpret_cast<const int8_t*>(base + std::size_t(bpr) * 192 + std::size_t(kbx) * 16) + scale_offset;
        r.d = (float) *reinterpret_cast<const half*>(base + std::size_t(bpr) * 208 + std::size_t(kbx) * 2);
        return r;
    }
};
// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock, computes integer dot products, applies
// the signed scale in the integer domain, then multiplies the two half scales and integer sum in the original order.
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { Int2 v[4]; int ls; float dw; };
    static W load(const Block* w, int iqs) {
        W r;
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(load_int(w->qs, iqs + j));
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = (float) w->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* x, int iqs) {
        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            sumi = dp4a(r.v[j].x, load_int(x[iqs / 4].qs, j), sumi);
            sumi = dp4a(r.v[j].y, load_int(x[iqs / 4].qs, j + 4), sumi);
        }
        sumi *= r.ls - 32;
        return r.dw * low(x[iqs / 4].ds) * (float) sumi;
    }
};
// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves the pinned 64/32-block iteration and
// 2048/1024-element small-K thresholds.
template<typename Weight, int Qi>
struct SmallTraits {
    using Block = Weight;
    static constexpr int DIV = 32, T = Qi / 2, KBY = 1, BPI = 2 * WARPS * WARP / Qi;
    static int kqs(int tid) { return 2 * (tid % (Qi / 2)); }
    struct W { const Weight* w; };
    static W load(const Block* w, int) { return W{w}; }
    static float apply(const W& r, const Q81Block* x, int k) { return small_q8_dot(r.w, x, k); }
};

// ---------------------------------------------------------------- the kernel
inline float warp_sum(const sycl::sub_group& sg, float x) {
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += sycl::permute_group_by_xor(sg, x, offset);
    return x;
}

// NW warps per work-group and ROWS rows per work-group.  The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is
// the ncols = 1 layout and keeps every column bitwise equal to a single-column call.  The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block).
bool g_multi_exact = true;

template<typename F, int NCOLS, int NW, int ROWS>
sycl::event launch_kernel(sycl::queue& q, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out,
                          size_t groups) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    constexpr int PW = NW - 1 > 0 ? NW - 1 : 1;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>(PW * NCOLS * ROWS * WARP), h);
        h.parallel_for(sycl::nd_range<1>(groups * NW * WARP, (size_t) NW * WARP),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            constexpr int BPI = F::BPI * NW / WARPS;           // blocks per iteration scale with the warp count
            const sycl::sub_group sg = it.get_sub_group();
            const int ty = (int) sg.get_group_linear_id(), tx = (int) sg.get_local_linear_id();
            const int tid = WARP * ty + tx;
            const int row0 = ROWS * (int) it.get_group(0);
            const int blocks_per_row = n_in / F::DIV;
            const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
            float tmp[NCOLS][ROWS] = {};
            for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
                const int kby = kbx * F::KBY;
                const int kqs = F::kqs(tid);
                #pragma unroll
                for (int i = 0; i < ROWS; ++i) {
                    // The source assumes allocator padding for partial row groups. This
                    // guard preserves every valid row's math without an out-of-bounds read.
                    if (row0 + i < n_out) {
                        typename F::W wv;
                        if constexpr (requires { F::load_row; })
                            wv = F::load_row(static_cast<const uint8_t*>(weights), row0 + i, kbx, blocks_per_row, kqs);
                        else
                            wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
                        #pragma unroll
                        for (int j = 0; j < NCOLS; ++j)
                            tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
                    }
                }
            }
            auto at = [&](int l, int j, int i, int lane) -> float& {
                return partial[((l * NCOLS + j) * ROWS + i) * WARP + lane];
            };
            if (ty > 0) {
                #pragma unroll
                for (int j = 0; j < NCOLS; ++j)
                    #pragma unroll
                    for (int i = 0; i < ROWS; ++i) at(ty - 1, j, i, tx) = tmp[j][i];
            }
            sycl::group_barrier(it.get_group());
            if (ty > 0) return;
            #pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                #pragma unroll
                for (int i = 0; i < ROWS; ++i) {
                    for (int l = 0; l < NW - 1; ++l) tmp[j][i] += at(l, j, i, tx);
                    tmp[j][i] = warp_sum(sg, tmp[j][i]);
                    if (tx == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
                }
            }
        });
    });
}

// The Xe layout: one sub-group a row, RPG rows a work-group.  The sub-group walks the row's blocks (WARP / T at a
// time, T lanes a block, as the traits split one), applies each loaded block to every column, and reduces by xor
// within the sub-group: no local memory and no barrier.  The CUDA layout above gives a row a whole work-group of 128
// lanes, most of them idle on the model's 2560- and 6144-wide rows, then waits on a barrier for the partial sums.
// A column's arithmetic does not depend on NCOLS, so every column is bitwise its single-column call.
// With `weights2` (a second matrix of the same type and shape, upstream b08cf3e1's pair), rows n_out .. 2 n_out - 1
// are its rows, written to y2: two products of the same activations in one launch, each row's arithmetic unchanged.
template<typename F, int NCOLS, int RPG>
sycl::event launch_row(sycl::queue& q, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out,
                       const void* weights2 = nullptr, float* y2 = nullptr) {
    static_assert(WARP % F::T == 0, "a block's lanes must divide the sub-group");
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const int rows_all = weights2 ? 2 * n_out : n_out;
    const std::size_t groups = (std::size_t(rows_all) + RPG - 1) / RPG;
    return q.parallel_for(sycl::nd_range<1>(groups * RPG * WARP, std::size_t(RPG) * WARP),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        int row = (int) it.get_group(0) * RPG + (int) sg.get_group_linear_id();
        if (row >= rows_all) return;                        // whole sub-groups only, and no barrier follows
        const void* wsrc = weights;
        float* ydst = y;
        if (row >= n_out) {   // the second matrix
            row -= n_out;
            wsrc = weights2;
            ydst = y2;
        }
        const auto* w = static_cast<const typename F::Block*>(wsrc);
        const int blocks_per_row = n_in / F::DIV;
        const int x_stride = n_in / Q8K;
        const int kqs = F::kqs(lane);
        float tmp[NCOLS] = {};
        for (int kbx = lane / F::T; kbx < blocks_per_row; kbx += WARP / F::T) {
            typename F::W wv;
            if constexpr (requires { F::load_row; })
                wv = F::load_row(static_cast<const uint8_t*>(wsrc), row, kbx, blocks_per_row, kqs);
            else
                wv = F::load(w + std::size_t(row) * blocks_per_row + kbx, kqs);
            for (int j = 0; j < NCOLS; ++j)
                tmp[j] += F::apply(wv, x + std::size_t(j) * x_stride + std::size_t(kbx) * F::KBY, kqs);
        }
        for (int j = 0; j < NCOLS; ++j) {
            const float s = warp_sum(sg, tmp[j]);
            if (lane == 0) ydst[std::size_t(j) * n_out + row] = s;
        }
    });
}

// the work-group layout: four sub-groups a row (or a row each, for rows of fewer blocks than an iteration takes)
template<typename F, int NCOLS>
sycl::event launch_group(sycl::queue& q, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out) {
    if (n_in / F::DIV < F::BPI)
        return launch_kernel<F, NCOLS, WARPS, WARPS>(q, weights, x_q8_1, y, n_in, n_out,
                                                     (std::size_t(n_out) + WARPS - 1) / WARPS);
    return launch_kernel<F, NCOLS, WARPS, 1>(q, weights, x_q8_1, y, n_in, n_out, std::size_t(n_out));
}

// Which layout a shape's rows take on a GPU.  A sub-group alone walks a row in ceil(blocks / (WARP / T)) dependent
// steps; the work-group's four sub-groups split it and add their sums through local memory.  On a B70 the sub-group
// is the faster up to 10 steps (the 2560-wide rows), the work-group beyond (the 6144-wide rows, 12-24 steps; the
// 2026-10-02-xe-decode-gpu record); on an RX 9060 XT the work-group from 8 (Qwen3.5-9B's 4096-wide Q4_K and Q5_K rows:
// its decode 47.1 -> 49.4 tok/s, 2026-11-01).  So a shape's layout is timed on each GPU on its first product
// (native_mmvq_tune, or the first call outside a graph's capture) and kept for the process: either layout gives every
// column the bits of its one-column call, and a shape keeping its layout keeps a token's rows the same in every
// window.  The steps rule stands where the shape is first met in a capture, and under STRATA_MMVQ_TUNE=0.
bool steps_rule(int blocks, int per_step) { return (blocks + per_step - 1) / per_step <= 10; }

// Timed by the host over back-to-back launches, as the decode runs them (HIP's profiling events added some 35 us to
// each kernel on an RX 9060 XT and turned its choices around).  Each timing lasts about kTuneUs, the two layouts in
// turn kTuneRounds times.  The launches read random weights at places through a span of VRAM several times the GPU's
// last cache (kTuneSpanCaches of it, at least kTuneSpanMin): the decode reads every other weight before a matrix
// comes again, so its rows come from VRAM, and the same rows read again and again stayed in an RX 9060 XT's 32 MiB
// Infinity Cache, where the sub-group's layout came out ahead and its decode the slower (4096-wide Q4_K: 54.8 us
// against the work-group's 66.2 in the cache, the decode 47.1 tok/s against 49.4).  The span is not the cache the
// device reports alone: HIP reports the L2 only.  Random, not zero: a GPU that compresses its memory may read a
// zeroed buffer faster than weights.  The other layout is taken only where it beats the rule's in all the rounds but
// one, by kTuneMargin: on an RTX 4070 every shape of Qwen3.5-9B was within 5% either way and the choices of those
// ties changed from run to run, the text with them, for no speed; at 0.95 an RX 9060 XT's 4096 x 12288 Q4_K (0.944)
// did too.  A shape that near the margin can still change its layout from one run to the next.  A product shorter than kTuneFloorUs keeps the
// rule: what it could save is small and its times are the least steady.
constexpr double kTuneUs = 3000.0;
constexpr int kTuneRounds = 7;
constexpr double kTuneMargin = 0.93;
constexpr double kTuneFloorUs = 10.0;
constexpr uint64_t kTuneSpanCaches = 4;
constexpr uint64_t kTuneSpanMin = 256ull << 20;
using LayoutKey = std::tuple<std::type_index, int, int>;   // the format, n_in, n_out
std::mutex layouts_mu;
std::map<LayoutKey, std::unique_ptr<core::PerDevice<char>>> layouts;

core::PerDevice<char>& layouts_for(const LayoutKey& key) {
    std::lock_guard<std::mutex> lock(layouts_mu);
    auto& p = layouts[key];
    if (!p) p = std::make_unique<core::PerDevice<char>>();
    return *p;
}

const char* format_name(const std::type_index& t);

// the faster layout of the shape on the queue's GPU (y is written; the caller's weights only where no span fits);
// true: the sub-group's
struct Regions {
    int n = 0;
    const void* const* ptrs = nullptr;
    const uint64_t* bytes = nullptr;
};

template<typename F>
bool time_layout(sycl::queue& q, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out,
                 const Regions& regions = {}) {
    const bool rule = steps_rule(n_in / F::DIV, WARP / F::T);
    uint8_t* span = nullptr;
    try {
        // the span: as many places for the matrix as fit, each on a 256-byte boundary
        const sycl::device dev = q.get_device();
        const uint64_t bytes = (uint64_t) n_out * (uint64_t) (n_in / F::DIV) * sizeof(typename F::Block);
        const uint64_t stride = (bytes + 255) / 256 * 256;
        uint64_t want = std::max<uint64_t>(kTuneSpanCaches * dev.get_info<sycl::info::device::global_mem_cache_size>(),
                                           kTuneSpanMin);
        want = std::min<uint64_t>(std::max(want, 2 * stride), dev.get_info<sycl::info::device::max_mem_alloc_size>());
        // the places: in the regions given (each a matrix's bytes at a 256-byte boundary within one), until they
        // cover the span; else a span allocated for them
        std::vector<const uint8_t*> at;
        uint64_t covered = 0;
        for (int r = 0; r < regions.n && covered < want; ++r) {
            const auto* base = static_cast<const uint8_t*>(regions.ptrs[r]);
            const uint64_t lead = (256 - reinterpret_cast<uintptr_t>(base) % 256) % 256;
            for (uint64_t o = lead; o + bytes <= regions.bytes[r] && covered < want; o += stride) {
                at.push_back(base + o);
                covered += stride;
            }
        }
        if (covered < want) at.clear();
        uint64_t got = 0;
        for (uint64_t w = want; at.empty() && w >= 2 * stride && span == nullptr; w /= 2) {
            span = sycl::malloc_device<uint8_t>(w, q);
            got = w;
        }
        const uint64_t places = !at.empty() ? at.size() : span != nullptr ? got / stride : 1;
        if (span != nullptr) {
            auto* w32 = reinterpret_cast<uint32_t*>(span);
            q.parallel_for(sycl::range<1>((size_t) (places * stride / 4)), [=](sycl::id<1> i) {
                uint32_t x = (uint32_t) i[0] * 2654435761u;
                x ^= x >> 13;
                x *= 0x5bd1e995u;
                w32[i] = x ^ (x >> 15);
            }).wait_and_throw();
        }
        uint64_t next = 0;
        auto run = [&](int l) {
            const void* w = !at.empty()       ? (const void*) at[next++ % places]
                            : span != nullptr ? (const void*) (span + (next++ % places) * stride)
                                              : weights;
            if (l == 1) launch_row<F, 1, 8>(q, w, x_q8_1, y, n_in, n_out);
            else launch_group<F, 1>(q, w, x_q8_1, y, n_in, n_out);
        };
        auto timed = [&](int l, int reps) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) run(l);
            q.wait_and_throw();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        run(0);
        run(1);
        q.wait_and_throw();
        const int reps = (int) std::clamp(kTuneUs / std::max(timed(rule ? 1 : 0, 4), 1.0), 4.0, 2000.0);
        timed(0, reps);   // the GPU's clocks up before the rounds
        double us[2] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        double ratio[kTuneRounds];   // the other layout's time over the rule's, a round
        for (int round = 0; round < kTuneRounds; ++round) {
            const double g = timed(0, reps), r = timed(1, reps);
            us[0] = std::min(us[0], g);
            us[1] = std::min(us[1], r);
            ratio[round] = rule ? g / r : r / g;
        }
        std::sort(ratio, ratio + kTuneRounds);
        const double worse = ratio[kTuneRounds - 2];   // the other's second least favourable round
        const bool row = us[rule ? 1 : 0] >= kTuneFloorUs && worse < kTuneMargin ? !rule : rule;
        static const bool log_all = strata::env_get("MMVQ_TUNE_LOG") != nullptr;
        if (row != rule || log_all)
            std::fprintf(stderr, "xestrata: the decode products of %s %d -> %d on %s: %s a row (timed: %.1f us a "
                                 "work-group, %.1f a sub-group; the other's rounds at %.3f of the rule's or less "
                                 "but one)\n", format_name(std::type_index(typeid(F))), n_in, n_out,
                         dev.get_info<sycl::info::device::name>().c_str(), row ? "a sub-group" : "a work-group",
                         us[0], us[1], worse);
        if (span != nullptr) sycl::free(span, q);
        return row;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata: timing the decode products of %s %d -> %d: %s\n",
                     format_name(std::type_index(typeid(F))), n_in, n_out, e.what());
        if (span != nullptr) {
            q.wait();
            sycl::free(span, q);
        }
        return rule;
    }
}

bool env_off(const char* name) {
    const char* v = strata::env_get(name);
    return v != nullptr && std::strtol(v, nullptr, 10) == 0;
}

// the shape's layout on the queue's GPU (true: a sub-group a row), timed over the call's buffers the first time
template<typename F>
bool row_layout(sycl::queue& q, void* stream, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out) {
    static const bool tune = !env_off("MMVQ_TUNE");
    if (!tune) return steps_rule(n_in / F::DIV, WARP / F::T);
    return layouts_for(LayoutKey{std::type_index(typeid(F)), n_in, n_out}).get(q.get_device(), [&]() -> char {
        if (gpu::is_capturing(stream)) return steps_rule(n_in / F::DIV, WARP / F::T);
        return time_layout<F>(q, weights, x_q8_1, y, n_in, n_out);
    }) != 0;
}

template<typename F, int NCOLS>
void launch_n(sycl::queue& q, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, bool row) {
    if (NCOLS > 1 && !g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        launch_kernel<F, NCOLS, NW, 2>(q, weights, x_q8_1, y, n_in, n_out, (std::size_t(n_out) + 1) / 2);
        return;
    }
    if (row) launch_row<F, NCOLS, 8>(q, weights, x_q8_1, y, n_in, n_out);
    else launch_group<F, NCOLS>(q, weights, x_q8_1, y, n_in, n_out);
}

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
sycl::queue& validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit stream");
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

template<typename F>
void mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, F::DIV);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    auto& q = validate_stream(stream);
    const bool row = row_layout<F>(q, stream, weights, x_q8_1, y, n_in, n_out);
    switch (ncols) {
        case 1: launch_n<F, 1>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 2: launch_n<F, 2>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 3: launch_n<F, 3>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 4: launch_n<F, 4>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 5: launch_n<F, 5>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 6: launch_n<F, 6>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 7: launch_n<F, 7>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 8: launch_n<F, 8>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        default: break;   // validate_shape took only 1 to 8 columns
    }
}

// native_mmvq_tune: the layout timed over a zero activation and an output of the shape's own
template<typename F>
void mmvq_tune(const void* weights, int n_in, int n_out, void* stream, const Regions& regions) {
    validate_shape(n_in, 1, F::DIV);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    auto& q = validate_stream(stream);
    if (env_off("MMVQ_TUNE") || gpu::is_capturing(stream)) return;
    layouts_for(LayoutKey{std::type_index(typeid(F)), n_in, n_out}).get(q.get_device(), [&]() -> char {
        void* x = sycl::malloc_device(native_q8_1_bytes(n_in, 1), q);
        float* y = sycl::malloc_device<float>((size_t) n_out, q);
        if (x == nullptr || y == nullptr) {   // no VRAM for it: the rule, as in a capture
            if (x != nullptr) sycl::free(x, q);
            if (y != nullptr) sycl::free(y, q);
            return steps_rule(n_in / F::DIV, WARP / F::T);
        }
        q.memset(x, 0, native_q8_1_bytes(n_in, 1)).wait_and_throw();
        const bool row = time_layout<F>(q, weights, x, y, n_in, n_out, regions);
        sycl::free(x, q);
        sycl::free(y, q);
        return row;
    });
}

// the pair in one launch where launch_n would take launch_row (a row of at most 10 steps; exact multi-column
// layout); false, launching nothing, otherwise
template<typename F>
bool mmvq_pair(const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2, int n_in, int n_out, int ncols,
               void* stream) {
    validate_shape(n_in, ncols, F::DIV);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(w1);
    validate_pointer(w2);
    validate_pointer(x_q8_1);
    validate_pointer(y1);
    validate_pointer(y2);
    auto& q = validate_stream(stream);
    if ((ncols > 1 && !g_multi_exact) || !row_layout<F>(q, stream, w1, x_q8_1, y1, n_in, n_out)) return false;
    switch (ncols) {
        case 1: launch_row<F, 1, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 2: launch_row<F, 2, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 3: launch_row<F, 3, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 4: launch_row<F, 4, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 5: launch_row<F, 5, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 6: launch_row<F, 6, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 7: launch_row<F, 7, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 8: launch_row<F, 8, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        default: return false;
    }
    return true;
}

template<typename F>
void mmvq_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
              void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, F::DIV);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    mmvq<F>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

using Q40Traits = SmallTraits<Q40Block, 4>;
using Q50Traits = SmallTraits<Q50Block, 4>;
using Q80Traits = SmallTraits<Q80Block, 8>;
// kNativeQ8Rows: Q80Traits' dot with the qs from the row's split arrays, every 32-bit load aligned; the same values
// reach the same dot in the same order, so the results are Q80Traits' bit for bit.
struct Q80RowTraits {
    using Block = Q80Block;
    static constexpr int DIV = Q80Traits::DIV, T = Q80Traits::T, KBY = Q80Traits::KBY, BPI = Q80Traits::BPI;
    static int kqs(int tid) { return Q80Traits::kqs(tid); }
    struct W { const uint8_t* qs; float d; };
    static W load_row(const uint8_t* weights, int row, int kbx, int bpr, int) {
        const uint8_t* base = weights + std::size_t(row) * bpr * sizeof(Q80Block);
        return W{base + std::size_t(kbx) * 32,
                 (float) *reinterpret_cast<const half*>(base + std::size_t(bpr) * 32 + std::size_t(kbx) * 2)};
    }
    static float apply(const W& r, const Q81Block* x, int iqs) {
        int sumi = 0;
        for (int i = 0; i < 2; ++i) sumi = dp4a(load_int(r.qs, iqs + i), load_int(x->qs, iqs + i), sumi);
        const float d1 = low(x->ds);
        return r.d * d1 * float(sumi);
    }
};
using IQ4NLTraits = SmallTraits<IQ4NLBlock, 4>;

const char* format_name(const std::type_index& t) {
    const std::pair<std::type_index, const char*> names[] = {
        {typeid(Q5KTraits), "Q5_K"}, {typeid(Q4KTraits), "Q4_K"}, {typeid(Q20Traits), "Q2_0"},
        {typeid(Q3KTraits), "Q3_K"}, {typeid(Q2KTraits), "Q2_K"}, {typeid(Q6KTraits), "Q6_K"},
        {typeid(Q6KRowTraits), "Q6_K"}, {typeid(IQ4XSTraits), "IQ4_XS"}, {typeid(Q40Traits), "Q4_0"},
        {typeid(Q50Traits), "Q5_0"}, {typeid(Q80Traits), "Q8_0"}, {typeid(Q80RowTraits), "Q8_0"},
        {typeid(IQ4NLTraits), "IQ4_NL"}};
    for (const auto& [ti, name] : names)
        if (ti == t) return name;
    return "a format";
}

} // namespace

// native_swiglu_q8_1's values, in a namespace of their own (SYCL names a kernel by its type)
namespace native_q8 {
struct Swiglu {
    static constexpr bool kRows = false;
    const float *gate, *up;
    float operator()(int64_t, int64_t c, int j) const {
        // contraction off: the SwiGLU product must not fuse into the sub-group sum's first add (the stored and
        // reloaded value of the two launches is rounded on its own)
#pragma clang fp contract(off)
        const int64_t i = c * Q8K + j;
        const float g = gate[i];
        return g / (1.0f + sycl::exp(-g)) * up[i];   // native_swiglu's expression
    }
};
}  // namespace native_q8

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    native_quantize_q8_1_rows(x, x_q8_1, n_in, ncols, stream);
}

std::size_t native_q8_1_rows_bytes(int n_in, int64_t rows) {
    if (n_in <= 0 || n_in % Q8K != 0 || rows < 0)
        throw std::invalid_argument("native Q8_1 rows require n_in > 0 and divisible by 32, and rows >= 0");
    return std::size_t(rows) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1_rows(const float* x, void* x_q8_1, int n_in, int64_t rows, void* stream) {
    if (n_in <= 0 || n_in % Q8K != 0 || rows < 1 || (int64_t) n_in * rows > INT32_MAX)
        throw std::invalid_argument("native Q8_1 rows require n_in > 0 and divisible by 32, and 1 <= rows");
    validate_pointer(x);
    validate_pointer(x_q8_1);
    auto& q = validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    q8act::quantize<1, q8act::Ggml>(q, (int64_t) n_in * rows / Q8K, 0, q8act::Flat{x},
                                    q8act::Q8_1{static_cast<uint8_t*>(x_q8_1)});
}

void native_swiglu_q8_1(const float* gate, const float* up, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(gate);
    validate_pointer(up);
    validate_pointer(x_q8_1);
    auto& q = validate_stream(stream);
    q8act::quantize<1, q8act::Ggml>(q, (int64_t) n_in * ncols / Q8K, 0, native_q8::Swiglu{gate, up},
                                    q8act::Q8_1{static_cast<uint8_t*>(x_q8_1)});
}

void native_swiglu_f32(const float* gate, const float* up, float* out, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(gate);
    validate_pointer(up);
    validate_pointer(out);
    auto& q = validate_stream(stream);
    q.parallel_for(sycl::range<1>((std::size_t) n_in * (std::size_t) ncols), [=](sycl::id<1> id) {
#pragma clang fp contract(off)
        const std::size_t i = id[0];
        const float g = gate[i];
        out[i] = g / (1.0f + sycl::exp(-g)) * up[i];
    });
}

void native_q5_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q5KTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_q5_k_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q5KTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q2_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q20Traits>(w, x, y, n_in, n_out, ncols, s); }
void native_q2_0_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q20Traits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q2_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q2KTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_q2_k_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q2KTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q3_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q3KTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_q3_k_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q3KTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_iq4_xs_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<IQ4XSTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_iq4_xs_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<IQ4XSTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q4_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q4KTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_q4_k_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q4KTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q6_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q6KTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_q6_k_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q6KTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q4_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q40Traits>(w, x, y, n_in, n_out, ncols, s); }
void native_q4_0_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q40Traits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q5_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q50Traits>(w, x, y, n_in, n_out, ncols, s); }
void native_q5_0_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q50Traits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q8_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q80Traits>(w, x, y, n_in, n_out, ncols, s); }
void native_q8_0_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q80Traits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_iq4_nl_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<IQ4NLTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_iq4_nl_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<IQ4NLTraits>(w, x, xq, y, n_in, n_out, ncols, s); }

bool native_q6_k_rows_ok(int n_in) noexcept { return n_in > 0 && n_in % 512 == 0; }
bool native_q8_0_rows_ok(int n_in) noexcept { return n_in > 0 && n_in % 64 == 0; }

void native_q8_0_to_rows(void* weights, int n_in, int n_out, void* stream) {
    if (!native_q8_0_rows_ok(n_in)) throw std::invalid_argument("Q8_0 rows need an even number of blocks per row");
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    auto& q = validate_stream(stream);
    const int bpr = n_in / 32;
    const std::size_t row_bytes = std::size_t(bpr) * sizeof(Q80Block);
    auto* w = static_cast<uint8_t*>(weights);
    constexpr int WG = 256;
    // as native_q6_k_to_rows: one work-group a row, through local memory, in place
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> row(sycl::range<1>(row_bytes), h);
        h.parallel_for(sycl::nd_range<1>(std::size_t(n_out) * WG, WG), [=](sycl::nd_item<1> it) {
            uint8_t* r = w + it.get_group(0) * row_bytes;
            const int t = (int) it.get_local_id(0);
            for (std::size_t i = t; i < row_bytes; i += WG) row[i] = r[i];
            sycl::group_barrier(it.get_group());
            for (std::size_t i = t; i < row_bytes; i += WG) {
                const std::size_t b = i / sizeof(Q80Block), o = i % sizeof(Q80Block);
                r[o < 2 ? std::size_t(bpr) * 32 + b * 2 + o : b * 32 + (o - 2)] = row[i];
            }
        });
    });
}

void native_q6_k_to_rows(void* weights, int n_in, int n_out, void* stream) {
    if (!native_q6_k_rows_ok(n_in)) throw std::invalid_argument("Q6_K rows need an even number of blocks per row");
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    auto& q = validate_stream(stream);
    const int bpr = n_in / QK;
    const std::size_t row_bytes = std::size_t(bpr) * sizeof(Q6KBlock);
    auto* w = static_cast<uint8_t*>(weights);
    constexpr int WG = 256;
    // One work-group a row: the row goes to local memory, then each byte is written to its place in the split
    // arrays, so the rewrite is in place and needs no second buffer.
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> row(sycl::range<1>(row_bytes), h);
        h.parallel_for(sycl::nd_range<1>(std::size_t(n_out) * WG, WG), [=](sycl::nd_item<1> it) {
            uint8_t* r = w + it.get_group(0) * row_bytes;
            const int t = (int) it.get_local_id(0);
            for (std::size_t i = t; i < row_bytes; i += WG) row[i] = r[i];
            sycl::group_barrier(it.get_group());
            for (std::size_t i = t; i < row_bytes; i += WG) {
                const std::size_t b = i / sizeof(Q6KBlock), o = i % sizeof(Q6KBlock);
                std::size_t to;
                if (o < 128) to = b * 128 + o;
                else if (o < 192) to = std::size_t(bpr) * 128 + b * 64 + (o - 128);
                else if (o < 208) to = std::size_t(bpr) * 192 + b * 16 + (o - 192);
                else to = std::size_t(bpr) * 208 + b * 2 + (o - 208);
                r[to] = row[i];
            }
        });
    });
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == kNativeQ6KRows || ggml_type == kNativeQ8Rows || ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 ||
           ggml_type == 10 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29 || ggml_type == 19;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;   // Q5_1 (OrcaRouter Q4_K_S, upstream d652cd6)
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 10: block_elems = 256; block_bytes = 84; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case kNativeQ6KRows: block_elems = 512; block_bytes = 420; break;   // whole rows of an even block count
    case kNativeQ8Rows: block_elems = 64; block_bytes = 68; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 19: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: return mmvq_pair<Q40Traits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case 6: return mmvq_pair<Q50Traits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case 8: return mmvq_pair<Q80Traits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case 20: return mmvq_pair<IQ4NLTraits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case 14: return mmvq_pair<Q6KTraits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case kNativeQ6KRows:
        return native_q6_k_rows_ok(n_in) && mmvq_pair<Q6KRowTraits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case kNativeQ8Rows:
        return native_q8_0_rows_ok(n_in) && mmvq_pair<Q80RowTraits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case 23: return mmvq_pair<IQ4XSTraits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    default: return false;   // the K-quants' and i-quants' own kernels: two launches
    }
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 10: native_q2_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case kNativeQ6KRows:
        if (!native_q6_k_rows_ok(n_in)) throw std::invalid_argument("Q6_K rows need an even number of blocks per row");
        mmvq<Q6KRowTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        break;
    case kNativeQ8Rows:
        if (!native_q8_0_rows_ok(n_in)) throw std::invalid_argument("Q8_0 rows need an even number of blocks per row");
        mmvq<Q80RowTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 7: case 16: case 17: case 18: case 19: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

void native_mmvq_tune(int ggml_type, const void* weights, int n_in, int n_out, void* stream, int regions,
                      const void* const* region_ptrs, const uint64_t* region_bytes) {
    const Regions r{regions, region_ptrs, region_bytes};
    switch (ggml_type) {
    case 2: mmvq_tune<Q40Traits>(weights, n_in, n_out, stream, r); break;
    case 6: mmvq_tune<Q50Traits>(weights, n_in, n_out, stream, r); break;
    case 8: mmvq_tune<Q80Traits>(weights, n_in, n_out, stream, r); break;
    case 20: mmvq_tune<IQ4NLTraits>(weights, n_in, n_out, stream, r); break;
    case 10: mmvq_tune<Q2KTraits>(weights, n_in, n_out, stream, r); break;
    case 11: mmvq_tune<Q3KTraits>(weights, n_in, n_out, stream, r); break;
    case 12: mmvq_tune<Q4KTraits>(weights, n_in, n_out, stream, r); break;
    case 13: mmvq_tune<Q5KTraits>(weights, n_in, n_out, stream, r); break;
    case 14: mmvq_tune<Q6KTraits>(weights, n_in, n_out, stream, r); break;
    case kNativeQ6KRows:
        if (native_q6_k_rows_ok(n_in)) mmvq_tune<Q6KRowTraits>(weights, n_in, n_out, stream, r);
        break;
    case kNativeQ8Rows:
        if (native_q8_0_rows_ok(n_in)) mmvq_tune<Q80RowTraits>(weights, n_in, n_out, stream, r);
        break;
    case 23: mmvq_tune<IQ4XSTraits>(weights, n_in, n_out, stream, r); break;
    case 42: mmvq_tune<Q20Traits>(weights, n_in, n_out, stream, r); break;
    default: break;   // the i-quants' own kernels (iq_mmvq), BF16: one layout
    }
}

} // namespace strata::kernels
