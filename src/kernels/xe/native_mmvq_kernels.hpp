// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/native_mmvq_kernels.hpp - the native MMVQ's kernels and its formats' dot-product traits, shared
// by native_mmvq.cpp (1..8 columns) and native_mmvq_wide.cpp (9..16), the Xe port of Strata's native_mmvq.cu.
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
// Two sources, so a window of up to 8 rows loads the device code it always loaded: the 9..16-column kernels are a
// device image of their own (native_mmvq_wide.cpp), loaded only when a wide window first runs (in one source they
// took 10 to 14 MiB of VRAM at the start from the expert cache, on the RTX 4070 and the B70).
#pragma once

#include "strata/kernels/verify_kernels.hpp"
#include "cuda_intrinsics.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace strata::kernels::mmvq_xe {
/// Columns 9..16 of a wide window for the format numbered `fmt` (kWideFmt), as native_mmvq.cpp's launch_n takes 1..8
/// (`row`: a sub-group a row; `exact`: the exact layouts, native_mmvq_set_multi_exact): native_mmvq_wide.cpp.
void mmvq_wide(int fmt, sycl::queue& q, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out,
               int ncols, bool row, bool exact);
}  // namespace strata::kernels::mmvq_xe

// Internal to each source that includes this (as they were in native_mmvq.cpp alone): the device compiler keeps a
// source's kernels and their helpers its own, and each source instantiates its own column counts.
namespace strata::kernels {
namespace {   // NOLINT(google-build-namespaces,cert-dcl59-cpp,misc-anonymous-namespace-in-header)

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
// Columns one launch takes: a window has one column a row, so the window's most (kVerifyMaxT, 16).  Up to 8 columns
// run the instances they always ran; 9..16 have their own, reached only by a window wider than 8 rows (they read every
// weight once for all their columns: two 8-column passes read the dense weights twice, upstream 9d3c8c366).
constexpr int MAX_NCOLS = kVerifyMaxT;

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

// the formats' numbers for mmvq_wide
template <typename F> inline constexpr int kWideFmt = -1;
template <> inline constexpr int kWideFmt<Q5KTraits> = 0;
template <> inline constexpr int kWideFmt<Q4KTraits> = 1;
template <> inline constexpr int kWideFmt<Q20Traits> = 2;
template <> inline constexpr int kWideFmt<Q3KTraits> = 3;
template <> inline constexpr int kWideFmt<Q2KTraits> = 4;
template <> inline constexpr int kWideFmt<Q6KTraits> = 5;
template <> inline constexpr int kWideFmt<Q6KRowTraits> = 6;
template <> inline constexpr int kWideFmt<IQ4XSTraits> = 7;
template <> inline constexpr int kWideFmt<Q40Traits> = 8;
template <> inline constexpr int kWideFmt<Q50Traits> = 9;
template <> inline constexpr int kWideFmt<Q80Traits> = 10;
template <> inline constexpr int kWideFmt<Q80RowTraits> = 11;
template <> inline constexpr int kWideFmt<IQ4NLTraits> = 12;

}  // namespace
}  // namespace strata::kernels
