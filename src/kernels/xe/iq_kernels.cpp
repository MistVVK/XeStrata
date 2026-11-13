// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/iq_kernels.cpp - see include/strata/kernels/iq_kernels.hpp; the Xe port of Strata's src/kernels/cuda/iq_kernels.cu.
//
// The dot products (vec_dot_*_q8_1), the dequantizers and the q8_1 quantizer are transcribed from llama.cpp
// (ggml/src/ggml-cuda/vecdotq.cuh, dequantize.cuh, quantize.cu at the commit in third_party/main/ggml/VERSION.txt;
// MIT license, third_party/main/ggml/LICENSE) through the CUDA version.  The block structs and codebook grids come from
// its ggml-common.h, included unchanged.  CUDA's integer intrinsics come from cuda_intrinsics.hpp, and a warp is a
// sub-group of 32 whose butterfly sums run in the CUDA order.
// MIT License
//
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
#include "strata/kernels/canon_math.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/swiglu_clamp.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/timed_choice.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/gpu.hpp"
#include "cuda_intrinsics.hpp"
#include "device_target.hpp"
#include "iq_bits.hpp"
#include "strata/kernels/q8_act.hpp"
#include "sm70_table.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_iq_kernels
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include "ggml-common-tq.h"
#include "tq_lut_i8.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>

namespace strata::kernels {
namespace {

constexpr int kWarp = 32;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
void sync_if_needed(void* stream, const sycl::event& event, const char* what) {
    if (!stream) core::Runtime::get().wait(event, what);
}

using xe::byte_perm;
using xe::dp4a;
using namespace iq_bits;

inline float low_half(const ggml_half2& ds) { return (float) ds[0]; }

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
// Each dot is split into `load` (everything that depends only on the weights: the grid words, the sign masks, the
// integer scales, the block scale as a float) and `apply` (the activation loads, the integer chain and the same
// float expression), as upstream's 8ec94aa does for the CUDA kernels.  A row read for several columns (the verify
// window's tokens, a group's entries) loads each block once and applies it to every column; `dot` is
// apply(load(...)), so one column's sum is bitwise the same either way.  The integer sums are exact, so the order of
// their terms does not matter; the float expressions are those of the single dot.
//
// The sign handling: CUDA builds a 0xFF byte mask with __vcmpne4 and negates the grid bytes with __vsub4 (two thirds
// of the decoders' integer instructions on Xe); here the negated bytes are taken out twice from the plain dot instead,
// with the mask spread from the four sign bits by one multiply (sign_mask).  Exact integers, so the same sum (grid bytes are below
// 128).
// The grid bytes (1-62 in every i-quant grid) negated where m is 0xFF, then one dp4a: (g ^ 0xFF) + 1 is -g in that
// byte, and with g never 0 the + 1 cannot carry into the next byte.  The same integer as CUDA's
// dp4a(g, u) - 2 * dp4a(g, u & m), in one dp4a instead of two (the Xe decode is integer-issue bound).
inline int dot_masked(int g, int u, uint32_t m, int acc) {
    return dp4a((int) (((uint32_t) g ^ m) + (m & 0x01010101u)), u, acc);
}

template<int TY> struct Split;

template<> struct Split<42> {   // Q2_0
    struct W { int qx[4], qy[4]; float d; };
    static W load(const void* vbq, int kbx, int iqs) {
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        const uint16_t* qs = (const uint16_t*) bq2_0->qs + (ptrdiff_t) iqs * 4;
        W r{};
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = qs[j];   // eight codes, the lowest bits first
            r.qx[j] = (int) q2_0_bytes(q & 0xFFu);
            r.qy[j] = (int) q2_0_bytes(q >> 8);
        }
        r.d = (float) bq2_0->d;
        return r;
    }
    static void parts(const W& r, const block_q8_1* bq8_1, int iqs, float& c, int& s) {
        const block_q8_1* chunk = bq8_1 + iqs;
        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            sumi = dp4a(get_int_b4(chunk->qs, j * 2 + 0), r.qx[j], sumi);
            sumi = dp4a(get_int_b4(chunk->qs, j * 2 + 1), r.qy[j], sumi);
        }
        s = sumi;
        c = r.d * low_half(chunk->ds);
    }
    // the products' order an expert's row takes on every device and on the CPU (canon_math.hpp, seg_dot_multi)
    static constexpr bool kParts = true;
    // the calls in a row that share one activation scale (their integer sums add before the scale)
    static constexpr int kGroup = 1;
    // the fixed-point sums (seg_dot_canon_fx): K 1, the block's d, the activation scale a call reads
    static constexpr int kFxLog2 = 0;
    static float block_d(const void* blk) { return (float) static_cast<const block_q2_0*>(blk)->d; }
    static float act_d(const block_q8_1* bq8_1, int iqs) { return low_half(bq8_1[iqs].ds); }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        float c;
        int s;
        parts(r, bq8_1, iqs, c, s);
        return c * (float) s;
    }
};

// IQ2_XXS, IQ3_XXS, IQ3_S: eight signed grid words, one 32-value sub-block
struct SignedGrid { int g[8]; uint32_t m[8]; int ls; float d; };
inline int signed_sum(const SignedGrid& r, const block_q8_1* q8) {
    int sumi = 0;
    for (int j = 0; j < 8; ++j) sumi = dot_masked(r.g[j], get_int_b4(q8->qs, j), r.m[j], sumi);
    return sumi;
}

template<> struct Split<16> {   // IQ2_XXS
    using W = SignedGrid;
    static W load(const void* vbq, int kbx, int iqs) { return load(vbq, kbx, iqs, iq2xxs_grid); }
    // grid: the codebook, the global table or a work-group's copy of it (GridOf below)
    static W load(const void* vbq, int kbx, int iqs, const uint64_t* grid) {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
        const uint32_t q2 = (uint32_t) get_int_b2(bq2->qs, iqs);
        const uint32_t aux32 = (uint32_t) get_int_b2(bq2->qs, iqs + 1);
        W r{};
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const Int2 grid_pos = grid_pair(grid, byte_of(q2, k0 / 2));
            const uint32_t s8 = ksigns_byte((uint8_t) (aux32 >> (7 * k0 / 2)));
            r.g[k0] = grid_pos.x; r.m[k0] = sign_mask(s8);
            r.g[k0 + 1] = grid_pos.y; r.m[k0 + 1] = sign_mask(s8 >> 4);
        }
        r.ls = (int) (aux32 >> 27 | 1);
        r.d = (float) bq2->d;
        return r;
    }
    static void parts(const W& r, const block_q8_1* bq8_1, int iqs, float& c, int& s) {
        s = signed_sum(r, &bq8_1[iqs / 2]) * r.ls;   // r.ls = 2 ls + 1: (2 ls + 1) / 8 a value
        c = r.d * low_half(bq8_1[iqs / 2].ds) * 0.125f;
    }
    // the products' order an expert's row takes on every device and on the CPU (canon_math.hpp, seg_dot_multi)
    static constexpr bool kParts = true;
    // the calls in a row that share one activation scale (their integer sums add before the scale)
    static constexpr int kGroup = 8;
    // the fixed-point sums (Split<18>'s): the 1/8, the block's d, the activation scale a call reads
    static constexpr int kFxLog2 = -3;
    static float block_d(const void* blk) { return (float) static_cast<const block_iq2_xxs*>(blk)->d; }
    static float act_d(const block_q8_1* bq8_1, int iqs) { return low_half(bq8_1[iqs / 2].ds); }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        float c;
        int s;
        parts(r, bq8_1, iqs, c, s);
        return c * (float) s;
    }
};

template<> struct Split<18> {   // IQ3_XXS
    using W = SignedGrid;
    static W load(const void* vbq, int kbx, int iqs) { return load(vbq, kbx, iqs, iq3xxs_grid); }
    static W load(const void* vbq, int kbx, int iqs, const uint32_t* grid) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const U2 pk = get_int2_b2(bq3->qs, iqs);
        const uint32_t packed[2] = {pk.x, pk.y};
        const uint32_t aux32 = (uint32_t) get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
        W r{};
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t s8 = ksigns_byte((uint8_t) (aux32 >> (7 * l0 / 2)));
            r.g[l0] = (int) grid[byte_of(packed[l0 / 4], (l0 + 0) % 4)]; r.m[l0] = sign_mask(s8);
            r.g[l0 + 1] = (int) grid[byte_of(packed[l0 / 4], (l0 + 1) % 4)]; r.m[l0 + 1] = sign_mask(s8 >> 4);
        }
        r.ls = (int) (aux32 >> 28);
        r.d = (float) bq3->d;
        return r;
    }
    static void parts(const W& r, const block_q8_1* bq8_1, int iqs, float& c, int& s) {
        s = signed_sum(r, &bq8_1[iqs / 2]) * (2 * r.ls + 1);   // d (0.5 + ls) / 2 a value: (2 ls + 1) / 4
        c = r.d * low_half(bq8_1[iqs / 2].ds) * 0.25f;
    }
    // the products' order an expert's row takes on every device and on the CPU (canon_math.hpp, seg_dot_multi)
    static constexpr bool kParts = true;
    // the calls in a row that share one activation scale (their integer sums add before the scale)
    static constexpr int kGroup = 8;
    // the fixed-point sums (canon_math.hpp's fx_*, seg_dot_canon_fx): log2 of the (2 ls + 1) / 4's 1/4, the block's d
    // and the activation scale a call's parts read
    static constexpr int kFxLog2 = -2;
    static float block_d(const void* blk) { return (float) static_cast<const block_iq3_xxs*>(blk)->d; }
    static float act_d(const block_q8_1* bq8_1, int iqs) { return low_half(bq8_1[iqs / 2].ds); }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        float c;
        int s;
        parts(r, bq8_1, iqs, c, s);
        return c * (float) s;
    }
};

template<> struct Split<21> {   // IQ3_S
    using W = SignedGrid;
    static W load(const void* vbq, int kbx, int iqs) { return load(vbq, kbx, iqs, iq3s_grid); }
    static W load(const void* vbq, int kbx, int iqs, const uint32_t* grid) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const U2 pk = get_int2_b2(bq3->qs, iqs);
        const uint32_t packed[2] = {pk.x, pk.y};
        const int qh = bq3->qh[iqs / 2];
        const uint32_t signs_packed = (uint32_t) get_int_b2(bq3->signs, iqs / 2);
        W r{};
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t s8 = byte_of(signs_packed, l0 / 2);
            r.g[l0] = (int) grid[byte_of(packed[l0 / 4], (l0 + 0) % 4) | ((qh << (8 - l0)) & 0x100)];
            r.g[l0 + 1] = (int) grid[byte_of(packed[l0 / 4], (l0 + 1) % 4) | ((qh << (7 - l0)) & 0x100)];
            r.m[l0] = sign_mask(s8);
            r.m[l0 + 1] = sign_mask(s8 >> 4);
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.d = (float) bq3->d;
        return r;
    }
    static void parts(const W& r, const block_q8_1* bq8_1, int iqs, float& c, int& s) {
        const int sumi = signed_sum(r, &bq8_1[iqs / 2]) * r.ls;
        s = sumi;
        c = r.d * low_half(bq8_1[iqs / 2].ds);
    }
    // the products' order an expert's row takes on every device and on the CPU (canon_math.hpp, seg_dot_multi)
    static constexpr bool kParts = true;
    // the calls in a row that share one activation scale (their integer sums add before the scale)
    static constexpr int kGroup = 8;
    // the fixed-point sums (Split<18>'s): K 1, the block's d, the activation scale a call reads
    static constexpr int kFxLog2 = 0;
    static float block_d(const void* blk) { return (float) static_cast<const block_iq3_s*>(blk)->d; }
    static float act_d(const block_q8_1* bq8_1, int iqs) { return low_half(bq8_1[iqs / 2].ds); }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        float c;
        int s;
        parts(r, bq8_1, iqs, c, s);
        return c * (float) s;
    }
};

// IQ2_XS and IQ2_S: two half sums, each with its 4-bit scale
struct SignedGrid2 { int g[8]; uint32_t m[8]; int ls0, ls1; float d; };
inline void parts_ls2(const SignedGrid2& r, const block_q8_1* bq8_1, int iqs, float& c, int& s) {
    const block_q8_1* q8 = &bq8_1[iqs / 2];
    int sumi0 = 0, sumi1 = 0;
    for (int j = 0; j < 4; ++j) sumi0 = dot_masked(r.g[j], get_int_b4(q8->qs, j), r.m[j], sumi0);
    for (int j = 4; j < 8; ++j) sumi1 = dot_masked(r.g[j], get_int_b4(q8->qs, j), r.m[j], sumi1);
    s = sumi0 * (2 * r.ls0 + 1) + sumi1 * (2 * r.ls1 + 1);   // d (0.5 + ls) / 4 a value: (2 ls + 1) / 8
    c = r.d * low_half(q8->ds) * 0.125f;
}

template<> struct Split<17> {   // IQ2_XS
    using W = SignedGrid2;
    static W load(const void* vbq, int kbx, int iqs) {
        const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
        const U2 pk = get_int2_b2(bq2->qs, iqs);
        const uint32_t packed[2] = {pk.x, pk.y};
        W r{};
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint16_t q2 = half_of(packed[l0 / 4], (l0 / 2) % 2);
            const Int2 grid_pos = grid_pair(iq2xs_grid, q2 & 0x1FF);
            const uint32_t s8 = ksigns_byte((uint8_t) (q2 >> 9));
            r.g[l0] = grid_pos.x; r.m[l0] = sign_mask(s8);
            r.g[l0 + 1] = grid_pos.y; r.m[l0 + 1] = sign_mask(s8 >> 4);
        }
        r.d = (float) bq2->d;
        return r;
    }
    static void parts(const W& r, const block_q8_1* bq8_1, int iqs, float& c, int& s) { parts_ls2(r, bq8_1, iqs, c, s); }
    // the fixed-point sums (seg_dot_canon_fx): the 1/8, the block's d, the activation scale a call reads
    static constexpr int kFxLog2 = -3;
    static float block_d(const void* blk) { return (float) static_cast<const block_iq2_xs*>(blk)->d; }
    static float act_d(const block_q8_1* bq8_1, int iqs) { return low_half(bq8_1[iqs / 2].ds); }
    // the products' order an expert's row takes on every device and on the CPU (canon_math.hpp, seg_dot_multi)
    static constexpr bool kParts = true;
    // the calls in a row that share one activation scale (their integer sums add before the scale)
    static constexpr int kGroup = 8;
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        float c;
        int s;
        parts(r, bq8_1, iqs, c, s);
        return c * (float) s;
    }
};

template<> struct Split<22> {   // IQ2_S
    using W = SignedGrid2;
    static W load(const void* vbq, int kbx, int iqs) { return load(vbq, kbx, iqs, iq2s_grid); }
    static W load(const void* vbq, int kbx, int iqs, const uint64_t* grid) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const uint32_t qs = (uint32_t) get_int_b2(bq2->qs, iqs / 2);
        const int qh = bq2->qh[iqs / 2];
        const uint32_t signs_packed = (uint32_t) get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        W r{};
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const Int2 grid_pos = grid_pair(grid, byte_of(qs, l0 / 2) | ((qh << (8 - l0)) & 0x300));
            const uint32_t s8 = byte_of(signs_packed, l0 / 2);
            r.g[l0] = grid_pos.x; r.m[l0] = sign_mask(s8);
            r.g[l0 + 1] = grid_pos.y; r.m[l0 + 1] = sign_mask(s8 >> 4);
        }
        r.d = (float) bq2->d;
        return r;
    }
    static void parts(const W& r, const block_q8_1* bq8_1, int iqs, float& c, int& s) { parts_ls2(r, bq8_1, iqs, c, s); }
    // the fixed-point sums (seg_dot_canon_fx): the 1/8, the block's d, the activation scale a call reads
    static constexpr int kFxLog2 = -3;
    static float block_d(const void* blk) { return (float) static_cast<const block_iq2_s*>(blk)->d; }
    static float act_d(const block_q8_1* bq8_1, int iqs) { return low_half(bq8_1[iqs / 2].ds); }
    // the products' order an expert's row takes on every device and on the CPU (canon_math.hpp, seg_dot_multi)
    static constexpr bool kParts = true;
    // the calls in a row that share one activation scale (their integer sums add before the scale)
    static constexpr int kGroup = 8;
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        float c;
        int s;
        parts(r, bq8_1, iqs, c, s);
        return c * (float) s;
    }
};

template<> struct Split<19> {   // IQ1_S (ggml-cuda's vec_dot_iq1_s_q8_1): one 32-value sub-block, a scale and a delta
    struct W { int g[8]; float d1q, delta; };
    static W load(const void* vbq, int kbx, int iqs) {
        const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
        const uint32_t qs = (uint32_t) get_int_b2(bq1->qs, iqs);
        const int qh = bq1->qh[iqs];
        W r{};
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int grid = (int) iq1s_grid_gpu[byte_of(qs, l0 / 2) | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
            r.g[l0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
        }
        r.d1q = (float) bq1->d * (float) (((qh >> 11) & 0x0E) + 1);
        r.delta = -1.0f + IQ1S_DELTA - (float) (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        int sumi = 0;
        for (int l0 = 0; l0 < 8; l0 += 2) {
            sumi = dp4a(r.g[l0], get_int_b4(bq8_1[iqs].qs, l0 + 0), sumi);
            sumi = dp4a(r.g[l0 + 1], get_int_b4(bq8_1[iqs].qs, l0 + 1), sumi);
        }
        return r.d1q * (low_half(bq8_1[iqs].ds) * (float) sumi + (float) bq8_1[iqs].ds[1] * r.delta);
    }
};

template<> struct Split<29> {   // IQ1_M
    struct W { int g[8]; float delta[4]; int sc0, sc1; float d; };
    static W load(const void* vbq, int kbx, int iqs) {
        const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
        const uint32_t qs = (uint32_t) get_int_b4(bq1->qs, iqs);
        W r{};
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = (int) iq1s_grid_gpu[byte_of(qs, l0 / 2) | ((qhl & 0x07) << 8)];
            r.g[l0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
            r.delta[l0 / 2] = -1.0f + IQ1M_DELTA - (float) (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        }
        const uint16_t* sc = (const uint16_t*) bq1->scales;
        r.d = iq1m_scale(sc);
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        r.sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        r.sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
            const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
            sumi[l0 / 4] = dp4a(r.g[l0], u0, sumi[l0 / 4]);
            sumi[l0 / 4] = dp4a(r.g[l0 + 1], u1, sumi[l0 / 4]);
            int sumy = 0;
            sumy = dp4a(u0, 0x01010101, sumy);
            sumy = dp4a(u1, 0x01010101, sumy);
            sumf[l0 / 4] += r.delta[l0 / 2] * (float) sumy;
        }
        const float d = r.d * low_half(bq8_1[iqs].ds);
        return d * (((float) sumi[0] + sumf[0]) * (float) r.sc0 + ((float) sumi[1] + sumf[1]) * (float) r.sc1);
    }
};

template<> struct Split<20> {   // IQ4_NL
    struct W { Int2 v[2]; float d; };
    static W load(const void* vbq, int kbx, int iqs) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
        W r{};
        for (int l = 0; l < 2; ++l) r.v[l] = get_int_from_table_16(get_int_b2(bq4->qs, iqs + l), kvalues_iq4nl);
        r.d = (float) bq4->d;
        return r;
    }
    static void parts(const W& r, const block_q8_1* bq8_1, int iqs, float& c, int& s) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        int sumi = 0;
        for (int l = 0; l < 2; ++l) {
            sumi = dp4a(r.v[l].x, q8[l + 0], sumi);
            sumi = dp4a(r.v[l].y, q8[l + 4], sumi);
        }
        s = sumi;
        c = r.d * low_half(bq8_1->ds);
    }
    // the products' order an expert's row takes on every device and on the CPU (canon_math.hpp, seg_dot_multi)
    static constexpr bool kParts = true;
    // the calls in a row that share one activation scale (their integer sums add before the scale): one, though a
    // block's two calls have the same scale, so a call is a lane's own fma and no lanes exchange sums (the CPU's rows
    // add the 8 lanes as cheaply)
    static constexpr int kGroup = 1;
    // the fixed-point sums (seg_dot_canon_fx; a down row: a scale for each 32): K 1, the block's d, its activation's
    static constexpr int kFxLog2 = 0;
    static float block_d(const void* blk) { return (float) static_cast<const block_iq4_nl*>(blk)->d; }
    static float act_d(const block_q8_1* bq8_1, int) { return low_half(bq8_1->ds); }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        float c;
        int s;
        parts(r, bq8_1, iqs, c, s);
        return c * (float) s;
    }
};

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block),
// and `bq8_1` is the super-block's first q8_1 block, so the call's activation is bq8_1[iqs / 4].
template<> struct Split<23> {   // IQ4_XS
    struct W { Int2 v[4]; int ls; float d; };
    static W load(const void* vbq, int kbx, int iqs) {
        const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
        W r{};
        for (int j = 0; j < 4; ++j) r.v[j] = get_int_from_table_16(get_int_b4(bq4->qs, iqs + j), kvalues_iq4nl);
        r.ls = (((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4)) - 32;
        r.d = (float) bq4->d;
        return r;
    }
    static void parts(const W& r, const block_q8_1* bq8_1, int iqs, float& c, int& s) {
        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            sumi = dp4a(r.v[j].x, get_int_b4(bq8_1[iqs / 4].qs, j + 0), sumi);
            sumi = dp4a(r.v[j].y, get_int_b4(bq8_1[iqs / 4].qs, j + 4), sumi);
        }
        s = sumi * r.ls;
        c = r.d * low_half(bq8_1[iqs / 4].ds);
    }
    // the products' order an expert's gate/up row takes on every device and on the CPU (canon_math.hpp,
    // seg_dot_multi); a down row's h has a scale for each 32, so there its calls stay apart
    static constexpr bool kParts = true;
    // the calls in a row that share one activation scale (their integer sums add before the scale)
    static constexpr int kGroup = 8;
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        float c;
        int s;
        parts(r, bq8_1, iqs, c, s);
        return c * (float) s;
    }
};

// ---------------------------------------------------------------- Unsloth's UD-Q4_K_XL experts (upstream efeffd8)
// Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down: llama.cpp's vec_dot_*_q8_1 (vecdotq.cuh, VDR 2 each), transcribed through
// upstream's CUDA version; the Q5_1 min term is its one departure (below).  Their decode-once splits follow the dots.
constexpr int VDR_Q4_K = 2, VDR_Q5_K = 2, VDR_Q5_1 = 2, VDR_Q8_0 = 2;

inline float vec_dot_q4_K_q8_1_impl_vmmq(const int* v, const int* u, const uint8_t* sc, const uint8_t* m,
                                         const ggml_half2& dm4, const float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = dp4a(v1i, u[2 * i + 1], dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = dp4a(0x01010101, u[2 * i + 1], dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (float) (dot1 * sc[i]);
        sumf_m += d8[i] * (float) (dot2 * m[i]);   // the min times the sum of the QUANTIZED activations
    }
    return (float) dm4[0] * sumf_d - (float) dm4[1] * sumf_m;
}
inline float vec_dot_q5_K_q8_1_impl_vmmq(const int* vl, const int* vh, const int* u, const uint8_t* sc,
                                         const uint8_t* m, const ggml_half2& dm5, const float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = dp4a(v0i, u[2 * i + 0], dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = dp4a(0x01010101, u[2 * i + 0], dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (float) (dot1 * sc[i]);
        sumf_m += d8[i] * (float) (dot2 * m[i]);
    }
    return (float) dm5[0] * sumf_d - (float) dm5[1] * sumf_m;
}
// the 6-bit scales and mins of the 32-value group pair bq8_offset / 2, branchless (llama.cpp; shared by Q4_K, Q5_K)
inline void k_scale_min(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const auto* scales = reinterpret_cast<const uint16_t*>(scales8);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}
// Q5_0 (OrcaRouter Q4_K_S's down projections, upstream d652cd6): llama.cpp's vec_dot_q5_0_q8_1 integer chain, with its
// -16 offset taken from the sum of the QUANTIZED activations as for Q5_1 (ggml-cpu's exact sum of (q - 16) * u), not
// from the q8_1 block's `ds.y`.
constexpr int VDR_Q5_0 = 2;
inline float vec_dot_q5_0_q8_1(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs) {
    const block_q5_0* bq5_0 = (const block_q5_0*) vbq + kbx;
    int sumi = 0, sumu = 0;
    for (int i = 0; i < VDR_Q5_0; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = dp4a(vi1, u1, sumi);
        sumu = dp4a(0x01010101, u1, dp4a(0x01010101, u0, sumu));
    }
    return (float) bq5_0->d * low_half(bq8_1->ds) * (float) (sumi - 16 * sumu);
}
// Q4_0 (plain llama-quantize Q4_0 files: gate/up and most down projections; upstream 16bdc707): the codes are centred
// (q - 8 as signed bytes) before the dot, the exact integer form ggml-cpu's q4_0 x q8_0 dot uses, rather than llama.cpp's
// CUDA chain (uncentred codes minus 8 x the q8_1 block's float sum `ds.y`: 1.4e-2 against the float product on a real
// Q4_0 expert, the centred form 5.3e-3).  Unsigned: (b & 0x08080808) * 0x1E reaches 0xF0F0F0F0, which overflows a
// signed int (upstream 44b0ed4a: the sm_60 build optimised it into a wrong value).
constexpr int VDR_Q4_0 = 2, VDR_Q4_1 = 2;
inline int q4_0_centred(int v) {
    const uint32_t b = (uint32_t) v ^ 0x08080808u;    // per byte: 0..15 -> (q - 8) as a 4-bit two's complement value
    return (int) (b | ((b & 0x08080808u) * 0x1Eu));   // sign-extend each nibble into its byte (0x08 * 0x1E = 0xF0)
}
inline float vec_dot_q4_0_q8_1(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs) {
    const block_q4_0* bq4 = (const block_q4_0*) vbq + kbx;
    int sumi = 0;
    for (int i = 0; i < VDR_Q4_0; ++i) {
        const int v = get_int_b2(bq4->qs, iqs + i);
        sumi = dp4a(q4_0_centred((v >> 0) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i), sumi);
        sumi = dp4a(q4_0_centred((v >> 4) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i + QI4_0), sumi);
    }
    return (float) bq4->d * low_half(bq8_1->ds) * (float) sumi;
}
// Q4_1 (llama-quantize puts it on a few down projections of its Q4_0 files): the Q5_1 chain (Split<7> below) without
// the fifth bit, the same min-term choice (the activations' own int8 codes)
inline float vec_dot_q4_1_q8_1(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs) {
    const block_q4_1* bq4_1 = (const block_q4_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
    for (int i = 0; i < VDR_Q4_1; ++i) {
        const int v = get_int_b4(bq4_1->qs, iqs + i);
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
        sumi = dp4a((v >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = dp4a((v >> 4) & 0x0F0F0F0F, u1, sumi);
        sumu = dp4a(0x01010101, u1, dp4a(0x01010101, u0, sumu));
    }
    const float d8 = low_half(bq8_1->ds);
    return (float) sumi * ((float) bq4_1->dm[0] * d8) + (float) sumu * ((float) bq4_1->dm[1] * d8);
}
// Q6_K gate/up (Unsloth's UD-Q6_K_XL experts; upstream 87659162): llama.cpp's integer chain verbatim
// (vecdotq.cuh vec_dot_q6_K_q8_1_impl_mmvq / vec_dot_q6_K_q8_1) - signed per-16-element scales, the -32 offset folded
// into the per-byte value, the q8_1 block's ds.low for the scales (Q6_K has no learned min); its split is below
constexpr int VDR_Q6_K = 1;
// Q3_K experts (Ornith-1.5's GSQ-RCO files, GLM-5.3-Flash's): llama.cpp's vecdotq.cuh vec_dot_q3_K_q8_1 with its
// _impl_mmvq, VDR 1 - QI 16 calls a block, each one 32-bit word of the quants against four q8_1 blocks (Q2_K's: below)
constexpr int VDR_Q3_K = 1;
inline float vec_dot_q3_K_q8_1(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs) {
    const block_q3_K* bq3_K = (const block_q3_K*) vbq + kbx;
    const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
    const int vl = get_int_b2(bq3_K->qs, iqs);
    const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;   // a 0 bit subtracts 4
    float sumf = 0.0f;
    for (int i = 0; i < QR3_K; ++i) {
        const block_q8_1* b8 = bq8_1 + bq8_offset + i;
        const int isc = scale_offset + 2 * i;
        const int sc_low = (bq3_K->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        const int sc_high = ((bq3_K->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = xe::vsubss4(vil, vih);
        sumf += low_half(b8->ds) * (float) (dp4a(vi, get_int_b4(b8->qs, iqs % QI8_1), 0) * sc);
    }
    return (float) bq3_K->d * sumf;
}

template<float (*DOT)(const void*, const block_q8_1*, int, int)>
struct WholeDot {
    struct W { const void* v; int kbx; };
    static W load(const void* vbq, int kbx, int iqs) { (void) iqs; return {vbq, kbx}; }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) { return DOT(r.v, bq8_1, r.kbx, iqs); }
};
// Unsloth's expert formats decode each weight word once for a group's entries and a window's columns (upstream
// e65858f8, its KqSplit): `load` takes the weight words, the unpacked scales and the block scale, `apply` the
// activations and the same integer chain and float expression as the whole dot above, so every entry's sum is the
// whole dot's bit for bit.  (B70 upstream, 2560 x 640 experts, 2- / 4-token windows: Q4_K/Q5_1 362 -> 402 / 369 -> 425
// GB/s of expert bytes, Q6_K/Q8_0 259 -> 307 / 256 -> 318; one token within 1-2%.)
template<> struct Split<12> {   // Q4_K
    struct W { int v[2]; uint16_t aux[2]; ggml_half2 dm; int bq8_offset; };
    static W load(const void* vbq, int kbx, int iqs) {
        const block_q4_K* b = (const block_q4_K*) vbq + kbx;
        W r{};
        r.bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        const auto* q4 = reinterpret_cast<const int*>(b->qs + (size_t) (16 * r.bq8_offset + 4 * ((iqs / 2) % 4)));
        r.v[0] = q4[0];
        r.v[1] = q4[4];
        k_scale_min(b->scales, r.bq8_offset, r.aux);
        r.dm = b->dm;
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        int u[2 * QR4_K] = {};
        float d8[QR4_K] = {};
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + r.bq8_offset + i;
            d8[i] = low_half(bq8i->ds);
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const auto* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return vec_dot_q4_K_q8_1_impl_vmmq(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
template<> struct Split<13> {   // Q5_K
    struct W { int vl[2], vh[2]; uint16_t aux[2]; ggml_half2 dm; int bq8_offset; };
    static W load(const void* vbq, int kbx, int iqs) {
        const block_q5_K* b = (const block_q5_K*) vbq + kbx;
        W r{};
        r.bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        const auto* ql = reinterpret_cast<const int*>(b->qs + (size_t) (16 * r.bq8_offset + 4 * ((iqs / 2) % 4)));
        const auto* qh = reinterpret_cast<const int*>(b->qh + (size_t) (4 * ((iqs / 2) % 4)));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        k_scale_min(b->scales, r.bq8_offset, r.aux);
        r.dm = b->dm;
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        int u[2 * QR5_K] = {};
        float d8[QR5_K] = {};
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + r.bq8_offset + i;
            d8[i] = low_half(bq8i->ds);
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const auto* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return vec_dot_q5_K_q8_1_impl_vmmq(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
// Q5_1: llama.cpp's integer chain, but the min term multiplies the sum of the QUANTIZED activations (dp4a with
// 0x01010101, times d8) instead of the q8_1 block's `ds.y`, which the quantizer (like llama.cpp's) fills with the sum
// of the ORIGINAL activations: ggml-cpu's convention, and the one the K-quant mins use.
template<> struct Split<7> {   // Q5_1
    struct W { int vi0[VDR_Q5_1], vi1[VDR_Q5_1]; float d, m; };
    static W load(const void* vbq, int kbx, int iqs) {
        const block_q5_1* b = (const block_q5_1*) vbq + kbx;
        W r{};
        for (int i = 0; i < VDR_Q5_1; ++i) {
            const int vl = get_int_b4(b->qs, iqs + i);
            const int vh = get_int_b4(b->qh, 0) >> (4 * (iqs + i));
            int vi0 = (vl >> 0) & 0x0F0F0F0F;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            int vi1 = (vl >> 4) & 0x0F0F0F0F;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            r.vi0[i] = vi0;
            r.vi1[i] = vi1;
        }
        r.d = (float) b->dm[0];
        r.m = (float) b->dm[1];
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        int sumi = 0, sumu = 0;
        for (int i = 0; i < VDR_Q5_1; ++i) {
            const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
            sumi = dp4a(r.vi0[i], u0, sumi);
            sumi = dp4a(r.vi1[i], u1, sumi);
            sumu = dp4a(0x01010101, u1, dp4a(0x01010101, u0, sumu));
        }
        const float d8 = low_half(bq8_1->ds);
        return (float) sumi * (r.d * d8) + (float) sumu * (r.m * d8);
    }
};
template<> struct Split<8> {   // Q8_0
    struct W { int v[VDR_Q8_0]; float d; };
    static W load(const void* vbq, int kbx, int iqs) {
        const block_q8_0* b = (const block_q8_0*) vbq + kbx;
        W r{};
        for (int i = 0; i < VDR_Q8_0; ++i) r.v[i] = get_int_b2(b->qs, iqs + i);
        r.d = (float) b->d;
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        int sumi = 0;
        for (int i = 0; i < VDR_Q8_0; ++i) sumi = dp4a(r.v[i], get_int_b4(bq8_1->qs, iqs + i), sumi);
        return r.d * low_half(bq8_1->ds) * (float) sumi;
    }
};
template<> struct Split<6> : WholeDot<vec_dot_q5_0_q8_1> {};    // Q5_0
template<> struct Split<2> : WholeDot<vec_dot_q4_0_q8_1> {};    // Q4_0
template<> struct Split<3> : WholeDot<vec_dot_q4_1_q8_1> {};    // Q4_1
template<> struct Split<14> {   // Q6_K: the whole dot's chain, the weight words and scales loaded once
    struct W { int vl, vh; int8_t sc[QR6_K]; float d; int bq8_offset; };
    static W load(const void* vbq, int kbx, int iqs) {
        const block_q6_K* b = (const block_q6_K*) vbq + kbx;
        W r{};
        r.bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
        r.vl = get_int_b2(b->ql, iqs);
        r.vh = get_int_b2(b->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
        for (int i = 0; i < QR6_K; ++i) r.sc[i] = b->scales[scale_offset + 4 * i];
        r.d = (float) b->d;
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        float sumf = 0.0f;
        for (int i = 0; i < QR6_K; ++i) {
            const block_q8_1* b8 = bq8_1 + r.bq8_offset + (ptrdiff_t) 2 * i;
            const int sc = r.sc[i];  // NOLINT(bugprone-signed-char-misuse): Q6_K scales are signed
            const int vil = (r.vl >> (4 * i)) & 0x0F0F0F0F;
            const int vih = ((r.vh >> (4 * i)) << 4) & 0x30303030;
            const int vi = xe::vsubss4(vil | vih, 0x20202020);   // (vil | vih) - 32
            sumf += low_half(b8->ds) * (float) (dp4a(vi, get_int_b4(b8->qs, iqs % QI8_1), 0) * sc);
        }
        return r.d * sumf;
    }
};
// Q2_K two words a call: the two share their q8_1 blocks and their scales (and the block's d and dmin), so the scales
// are unpacked, the integer sums made and the floats multiplied once for both - a decode token's experts (one entry
// each) were bound by that per-word work (B70: 131 GB/s against Q4_K's 343).  The sums add in another order than
// llama.cpp's vec_dot_q2_K_q8_1 (one word a call).
template<> struct Split<10> {
    struct W { int v0, v1, bq8_offset; uint32_t sc; float d, dmin; };
    static W load(const void* vbq, int kbx, int iqs) {
        const block_q2_K* b = (const block_q2_K*) vbq + kbx;
        W r{};
        r.bq8_offset = QR2_K * (iqs / QI8_1);
        const int so = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
        r.v0 = get_int_b4(b->qs, iqs);
        r.v1 = get_int_b4(b->qs, iqs + 1);
        r.sc = 0;
        for (int i = 0; i < QR2_K; ++i) r.sc |= (uint32_t) b->scales[so + 2 * i] << (8 * i);
        r.d = (float) b->dm[0];
        r.dmin = (float) b->dm[1];
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        float sumf_d = 0.0f, sumf_m = 0.0f;
        for (int i = 0; i < QR2_K; ++i) {
            const block_q8_1* b8 = bq8_1 + r.bq8_offset + i;
            const int u0 = get_int_b4(b8->qs, iqs % QI8_1), u1 = get_int_b4(b8->qs, iqs % QI8_1 + 1);
            const float d8 = low_half(b8->ds);
            const int sc = (int) ((r.sc >> (8 * i)) & 0xFF);
            const int sumi = dp4a((r.v1 >> (2 * i)) & 0x03030303, u1, dp4a((r.v0 >> (2 * i)) & 0x03030303, u0, 0));
            const int m = (sc >> 4) * 0x01010101;   // the min in every byte, against the q8_1 values' sum
            sumf_d += d8 * (float) (sumi * (sc & 0xF));
            sumf_m += d8 * (float) dp4a(m, u1, dp4a(m, u0, 0));
        }
        return r.d * sumf_d - r.dmin * sumf_m;
    }
};
template<> struct Split<11> : WholeDot<vec_dot_q3_K_q8_1> {};   // Q3_K

// ---------------------------------------------------------------- the formats
// bsz = bytes per block, qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
// ---------------------------------------------------------------- APR trellis types (ggml 144 TQ2_T, 145 TQK6,
// 146 TQK7, 147 / 148 / 149 TQP5 / TQP6 / TQP7)
// Upstream's (src/kernels/cuda/iq_kernels.cu, transcribed from the agention llama.cpp fork's ggml-cuda/tq.cuh; format
// in ggml-common-tq.h).  A 128-value block is 8 lanes of 4 trellis steps (16 values); lane l reads q8_1 block l / 2,
// ints 4 (l & 1) .. 4 (l & 1) + 3.  The lane's 16 codebook values come from the int8 codebook tq_lut_i8 (one global
// scale, relative RMS error 0.85% of the codebook's), two 2-byte entries a step = one int of packed int8, so a lane is
// 4 dp4a; the prompt path's FP16 dequantizer (dq_tq) reads the fp16 codebook, as ggml's to_float does, so the decode
// and the prompt agree within tolerance, not bitwise (upstream's Numerics).
// TQP (pair trellis, agentionai/llama.cpp 9aa51ce): 64 steps of K bits a block, each step two values (the state's
// first codebook point only), so a lane is 8 steps and the same 4 dp4a.
template<int TY> struct TqInfo;
template<> struct TqInfo<144> { using block_t = block_tq2_t; static constexpr int NS = 32; };   // byte-aligned, no K
template<> struct TqInfo<145> { using block_t = block_tqk6; static constexpr int K = 6, NS = 32; };
template<> struct TqInfo<146> { using block_t = block_tqk7; static constexpr int K = 7, NS = 32; };
template<> struct TqInfo<147> { using block_t = block_tqp5; static constexpr int K = 5, NS = 64; };
template<> struct TqInfo<148> { using block_t = block_tqp6; static constexpr int K = 6, NS = 64; };
template<> struct TqInfo<149> { using block_t = block_tqp7; static constexpr int K = 7, NS = 64; };
constexpr bool is_tq(int t) { return t >= 144 && t <= 149; }
inline float tq_h2f(uint32_t bits) { return (float) sycl::bit_cast<sycl::half>((uint16_t) bits); }
// The u16 words the four states of tq_states4 span at most: the window's offset into its first word (0..15, set by
// t0 and K) plus 3 K + 16 bits.  K = 5 and 6 need three words wherever t0 falls; IGC proved it for K = 5 only and
// read a fourth for K = 6, and TQP6's grouped products ran 35% slower than TQP5's and TQP7's (B70, 2026-10-25).
template<int TY>
constexpr int tq_window_words() {
    int bits = 0;
    for (int t0 = 0; t0 + 4 <= TqInfo<TY>::NS; t0 += 4) {
        const int b = (((TqInfo<TY>::NS - 4 - t0) * TqInfo<TY>::K) & 15) + 3 * TqInfo<TY>::K + 16;
        bits = b > bits ? b : bits;
    }
    return (bits + 15) / 16;
}
// the states of steps t0 .. t0 + 3 of one block, t0 a multiple of 4 (qs is 2-byte aligned: every trellis block has an
// even size)
template<int TY>
inline void tq_states4(const uint8_t* qs, int t0, uint32_t s[4]) {
    const uint16_t* qs16 = (const uint16_t*) qs;
    if constexpr (TY == 144) {   // s_t = qs[(t + 31) % 32] << 8 | qs[t]
        const int l = t0 / 4;
        const uint32_t w01 = qs16[2 * l + 0], w23 = qs16[2 * l + 1];
        const uint32_t prev = qs[(4 * l + 31) & 31];
        const uint32_t b0 = w01 & 0xFF, b1 = w01 >> 8, b2 = w23 & 0xFF, b3 = w23 >> 8;
        s[0] = (prev << 8) | b0;
        s[1] = (b0 << 8) | b1;
        s[2] = (b1 << 8) | b2;
        s[3] = (b2 << 8) | b3;
    } else {
        // step t's state: the 16 stream bits from bit (NS - 1 - t) K, circular over NS K / 16 u16 words; steps
        // t0 .. t0 + 3 all lie in the tq_window_words words from (NS - 4 - t0) K >> 4
        constexpr int K = TqInfo<TY>::K, nwords = TqInfo<TY>::NS * K / 16;
        const int off3 = (TqInfo<TY>::NS - 4 - t0) * K, wb = off3 >> 4, r = off3 & 15;
        const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
        const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
        uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) | ((uint64_t) qs16[wi2] << 32);  // NOLINT(clang-analyzer-security.ArrayBound): wb + 1 >= nwords implies wb + 2 >= nwords, which the analyzer takes apart
        if constexpr (tq_window_words<TY>() > 3) {
            const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
            v |= (uint64_t) qs16[wi3] << 48;
        }
        for (int i = 0; i < 4; ++i) s[i] = (uint32_t) (v >> (r + (3 - i) * K)) & 0xFFFFu;
    }
}
template<int TY> struct SplitTq {
    struct W { int q[4]; float d; };
    static W load(const void* vbq, int kbx, int iqs) {
        const auto* b = (const typename TqInfo<TY>::block_t*) vbq + kbx;
        W r{};
        if constexpr (TqInfo<TY>::NS == 64) {   // lane l = steps 8l .. 8l + 7, int i = steps 8l + 2i, 8l + 2i + 1
            uint32_t s[8];
            tq_states4<TY>(b->qs, 2 * iqs, s);
            tq_states4<TY>(b->qs, 2 * iqs + 4, s + 4);
            for (int i = 0, j = 0; i < 4; ++i, j += 2) {
                const uint32_t x0 = (s[j] & TQ_STATE_MASK) * 0x9e3779b1u;
                const uint32_t x1 = (s[j + 1] & TQ_STATE_MASK) * 0x9e3779b1u;
                r.q[i] = (int) ((uint32_t) tq_lut_i8[x0 >> 21] | ((uint32_t) tq_lut_i8[x1 >> 21] << 16));
            }
        } else {
            uint32_t s[4];
            tq_states4<TY>(b->qs, iqs, s);
            for (int i = 0; i < 4; ++i) {
                const uint32_t x = (s[i] & TQ_STATE_MASK) * 0x9e3779b1u;
                r.q[i] = (int) ((uint32_t) tq_lut_i8[x >> 21] | ((uint32_t) tq_lut_i8[(x >> 10) & 2047u] << 16));
            }
        }
        r.d = tq_h2f(b->d);
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        const int l = iqs / 4;
        const block_q8_1* b8 = bq8_1 + l / 2;
        const int* q8 = (const int*) b8->qs + (size_t) 4 * (l & 1);
        int sumi = 0;
        for (int i = 0; i < 4; ++i) sumi = dp4a(r.q[i], q8[i], sumi);
        return r.d * (low_half(b8->ds) * TQ_LUT_I8_SCALE) * (float) sumi;
    }
};
template<> struct Split<144> : SplitTq<144> {};
template<> struct Split<145> : SplitTq<145> {};
template<> struct Split<146> : SplitTq<146> {};
template<> struct Split<147> : SplitTq<147> {};
template<> struct Split<148> : SplitTq<148> {};
template<> struct Split<149> : SplitTq<149> {};

template<int TY, int BSZ, int QK, int IPB, int STEP> struct FmtOf {
    static constexpr int bsz = BSZ, qk = QK, ipb = IPB, step = STEP;
    using S = Split<TY>;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) {
        return S::apply(S::load(v, kbx, iqs), y, iqs);
    }
};
template<int TY> struct Fmt;
template<> struct Fmt<16> : FmtOf<16, (int) sizeof(block_iq2_xxs), 256, 8, 2> {};
template<> struct Fmt<17> : FmtOf<17, (int) sizeof(block_iq2_xs), 256, 8, 2> {};
template<> struct Fmt<18> : FmtOf<18, (int) sizeof(block_iq3_xxs), 256, 8, 2> {};
template<> struct Fmt<20> : FmtOf<20, (int) sizeof(block_iq4_nl), 32, 2, 2> {};
template<> struct Fmt<21> : FmtOf<21, (int) sizeof(block_iq3_s), 256, 8, 2> {};
template<> struct Fmt<22> : FmtOf<22, (int) sizeof(block_iq2_s), 256, 8, 2> {};
template<> struct Fmt<23> : FmtOf<23, (int) sizeof(block_iq4_xs), 256, 8, 4> {};
template<> struct Fmt<19> : FmtOf<19, (int) sizeof(block_iq1_s), 256, 8, 1> {};
template<> struct Fmt<29> : FmtOf<29, (int) sizeof(block_iq1_m), 256, 8, 1> {};
template<> struct Fmt<42> : FmtOf<42, (int) sizeof(block_q2_0), 64, 2, 1> {};
template<> struct Fmt<12> : FmtOf<12, (int) sizeof(block_q4_K), 256, QI4_K / VDR_Q4_K, VDR_Q4_K> {};
template<> struct Fmt<13> : FmtOf<13, (int) sizeof(block_q5_K), 256, QI5_K / VDR_Q5_K, VDR_Q5_K> {};
template<> struct Fmt<7> : FmtOf<7, (int) sizeof(block_q5_1), 32, QI5_1 / VDR_Q5_1, VDR_Q5_1> {};
template<> struct Fmt<8> : FmtOf<8, (int) sizeof(block_q8_0), 32, QI8_0 / VDR_Q8_0, VDR_Q8_0> {};
template<> struct Fmt<6> : FmtOf<6, (int) sizeof(block_q5_0), 32, QI5_0 / VDR_Q5_0, VDR_Q5_0> {};
template<> struct Fmt<2> : FmtOf<2, (int) sizeof(block_q4_0), 32, QI4_0 / VDR_Q4_0, VDR_Q4_0> {};
template<> struct Fmt<3> : FmtOf<3, (int) sizeof(block_q4_1), 32, QI4_1 / VDR_Q4_1, VDR_Q4_1> {};
template<> struct Fmt<14> : FmtOf<14, (int) sizeof(block_q6_K), 256, QI6_K / VDR_Q6_K, VDR_Q6_K> {};
// the trellis types: 8 lanes of 4 steps a 128-value block, iqs = 4 * lane
template<> struct Fmt<144> : FmtOf<144, (int) sizeof(block_tq2_t), 128, 8, 4> {};
template<> struct Fmt<145> : FmtOf<145, (int) sizeof(block_tqk6), 128, 8, 4> {};
template<> struct Fmt<146> : FmtOf<146, (int) sizeof(block_tqk7), 128, 8, 4> {};
template<> struct Fmt<147> : FmtOf<147, (int) sizeof(block_tqp5), 128, 8, 4> {};
template<> struct Fmt<148> : FmtOf<148, (int) sizeof(block_tqp6), 128, 8, 4> {};
template<> struct Fmt<149> : FmtOf<149, (int) sizeof(block_tqp7), 128, 8, 4> {};
template<> struct Fmt<10> : FmtOf<10, (int) sizeof(block_q2_K), 256, QI2_K / 2, 2> {};   // two words a call
template<> struct Fmt<11> : FmtOf<11, (int) sizeof(block_q3_K), 256, QI3_K / VDR_Q3_K, VDR_Q3_K> {};

// A grid format's codebook, for a work-group's copy in local memory (the Volta experts below): its word, its length
// and the global table.  N = 0: no codebook (IQ4_XS's 16 values stay where they are).
template<int TY> struct GridOf { using T = uint32_t; static constexpr int N = 0; };
template<> struct GridOf<16> { using T = uint64_t; static constexpr int N = 256; static const T* src() { return iq2xxs_grid; } };
template<> struct GridOf<18> { using T = uint32_t; static constexpr int N = 256; static const T* src() { return iq3xxs_grid; } };
template<> struct GridOf<21> { using T = uint32_t; static constexpr int N = 512; static const T* src() { return iq3s_grid; } };
template<> struct GridOf<22> { using T = uint64_t; static constexpr int N = 1024; static const T* src() { return iq2s_grid; } };

// CUDA's __shfl_xor butterfly, in the same order, so a row's sum does not depend on the backend's reduction tree.
inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}

// A row's block kbx, its offset in 32 bits: an int index into the block array made the compiler multiply and add the
// addresses in 64 bits (two SIMD16 halves each on Xe).  A row is far below 4 GiB.
template<class F>
inline const void* block_at(const uint8_t* row, int kbx) { return row + (size_t) ((uint32_t) kbx * (uint32_t) F::bsz); }

// One row against up to NE activations (x[e], the first `n` in use) on LPR lanes (a segment of the sub-group, LPR a
// power of two up to 32): the sub-group carries WARP / LPR rows at once, so a short row (the experts' 640-wide down
// projection: 180-360 bytes) keeps its lanes busy and reduces in log2(LPR) steps.  Each block is loaded once for all
// the entries (upstream 8ec94aa); entry e's sum is what one entry alone gives, bit for bit.  `grid`
// (GridOf<TY>::T): the codebook the blocks are decoded with, when not the global table.
template<class S, class = void> struct HasParts : std::false_type {};
template<class S> struct HasParts<S, std::void_t<decltype(S::kParts)>> : std::true_type {};
// a format with parts: the calls in a row that share one activation scale (0 for a format without)
template<class S> constexpr int group_of() {
    if constexpr (HasParts<S>::value) return S::kGroup;
    else return 0;
}

// A format with parts (canon_math.hpp; the CPU's rows take the same steps): the calls of a row that share one
// activation scale (kGroup neighbouring calls, on kGroup neighbouring lanes, a block of an i-quant) add their exact
// integer sums first, a butterfly of integers, and each lane then adds the group's scale times that sum in one
// written-out fma; the lane sums are reduced over the lanes kGroup apart and more (the lanes of a group hold the same
// sum).  With groups, every lane runs nb_all blocks' trips, so the integer butterflies see the whole sub-group, and a
// lane whose row is out of range (nb 0) adds nothing; a call with a scale of its own (kGroup 1) is its lane's own
// fma, in the plain loop, which was 2.5% faster on an RTX 4070 (the Q2_0 down rows).  Two trips a round of exchanges
// (half the exchanges) were 5-13% slower on the B70 and the RTX 4070.
template<int TY, int LPR, int NE, class G>
inline void seg_dot_canon(const sycl::sub_group& sg, const uint8_t* row, const block_q8_1* const (&x)[NE], int n,
                          int nb, int nb_all, int seg_lane, float (&out)[NE], const G* grid) {
    STRATA_CANON_NO_CONTRACT
    using F = Fmt<TY>;
    constexpr int KG = F::S::kGroup;
    static_assert(LPR % KG == 0 && (KG & (KG - 1)) == 0, "a group's calls on neighbouring lanes of one segment");
    float s[NE];
    for (int e = 0; e < NE; ++e) s[e] = 0.0f;
    const int total = nb_all * F::ipb, mine = nb * F::ipb;
    if constexpr (KG == 1) {   // no integer butterfly: each lane its own calls, as seg_dot_multi's plain loop
        (void) total;
        for (int k = seg_lane; k < mine; k += LPR) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const auto wd = [&] {
                if constexpr (std::is_void_v<G>) return F::S::load(block_at<F>(row, kbx), 0, iqs);
                else return F::S::load(block_at<F>(row, kbx), 0, iqs, grid);
            }();
            const uint32_t xo32 = (uint32_t) kbx * (uint32_t) (F::qk / 32);   // in 32 bits (block_at)
            const size_t xo = xo32;
            for (int e = 0; e < NE; ++e)
                if (e < n) {
                    float c;
                    int t;
                    F::S::parts(wd, x[e] + xo, iqs, c, t);
                    s[e] = canon::fma(c, (float) t, s[e]);
                }
        }
    } else
    for (int k0 = 0; k0 < total; k0 += LPR) {
        const int k = k0 + seg_lane;
        const bool on = k < mine;
        const int kk = on ? k : 0, kbx = kk / F::ipb, iqs = F::step * (kk % F::ipb);
        const auto wd = [&] {
            if constexpr (std::is_void_v<G>) return F::S::load(block_at<F>(row, kbx), 0, iqs);
            else return F::S::load(block_at<F>(row, kbx), 0, iqs, grid);
        }();
        const uint32_t xo32 = (uint32_t) kbx * (uint32_t) (F::qk / 32);   // in 32 bits (block_at)
        const size_t xo = xo32;
        for (int e = 0; e < NE; ++e)
            if (e < n) {   // n is the whole sub-group's: the butterflies stay uniform
                float c = 0.0f;
                int t = 0;
                if (on) F::S::parts(wd, x[e] + xo, iqs, c, t);
                for (int o = KG / 2; o > 0; o >>= 1) t += sycl::permute_group_by_xor(sg, t, o);
                if (on) s[e] = canon::fma(c, (float) t, s[e]);
            }
    }
    for (int e = 0; e < NE; ++e) {
        for (int o = LPR / 2; o >= KG; o >>= 1) s[e] += sycl::permute_group_by_xor(sg, s[e], o);
        out[e] = s[e];
    }
}

// A format with kFxLog2 (the gate/up's i-quants, IQ4_NL's down): seg_dot_canon's calls and integer butterflies, each
// group's sum times its block's integer scale (canon_math.hpp's fx_scale) added in int64 and the lanes' sums by an
// int64 butterfly - exact, so in any order: the CPU's rows (iq_avx2_rows.inl's row_dot_canon_fx) and the prompt's
// products (iq_mmq.cpp's FX bodies) give the same bits.  The row's top: the largest exponent of its blocks' d's, read
// by the segment's lanes, plus each activation's (the scale of each 256, in its first q8_1 block).
template<class S, class = void> struct HasFx : std::false_type {};
template<class S> struct HasFx<S, std::void_t<decltype(S::kFxLog2)>> : std::true_type {};

// The row's weights' part of its top, canon_math.hpp's fx_row_top, read whole: every block's d by the segment's lanes
// (the rare row with a d past its first block's span, fx_first_top's redo)
template<int TY, int LPR>
inline int fx_row_top(const sycl::sub_group& sg, const uint8_t* row, int nb, int seg_lane) {
    using F = Fmt<TY>;
    int top = canon::kFxNone;
    for (int b = seg_lane; b < nb; b += LPR) top = sycl::max(top, canon::fx_exp(F::S::block_d(block_at<F>(row, b))));
    for (int o = LPR / 2; o > 0; o >>= 1) top = sycl::max(top, sycl::permute_group_by_xor(sg, top, o));
    return nb > 0 ? canon::fx_row_top(canon::fx_exp(F::S::block_d(block_at<F>(row, 0))), top) : top;
}
// The same where no block's d is past its first block's span (fx_row_top's usual case): the first block's d alone.
// Its sums then flag a block past it (its exponent above the top), and the row is done again with fx_row_top.
template<int TY>
inline int fx_first_top(const uint8_t* row, int nb) {
    using F = Fmt<TY>;
    return nb > 0 ? canon::fx_exp(F::S::block_d(block_at<F>(row, 0))) + canon::kFxRowSpan : canon::kFxNone;
}
// an activation of nb_all of TY's blocks; S256: its scale is one for each 256 values (each 8th q8_1 block read), else
// one for each 32
template<int TY, int LPR, bool S256>
inline int fx_act_top(const sycl::sub_group& sg, const block_q8_1* x, int nb_all, int seg_lane) {
    constexpr int step = S256 ? 8 : 1;
    const int n32 = nb_all * (Fmt<TY>::qk / 32);
    int top = canon::kFxNone;
    for (int b = seg_lane * step; b < n32; b += LPR * step)
        top = sycl::max(top, canon::fx_exp(low_half(x[b].ds)));
    for (int o = LPR / 2; o > 0; o >>= 1) top = sycl::max(top, sycl::permute_group_by_xor(sg, top, o));
    return canon::fx_row_top(canon::fx_exp(low_half(x[0].ds)), top);   // the activation's part, as the weights'
}
// The activation's part where no scale is past its first block's span (fx_first_top's, for the activation)
inline int fx_act_first_top(const block_q8_1* x) { return canon::fx_exp(low_half(x[0].ds)) + canon::kFxRowSpan; }

template<int TY, int LPR, int NE, class G, bool S256>
inline void seg_dot_canon_fx(const sycl::sub_group& sg, const uint8_t* row, const block_q8_1* const (&x)[NE], int n,
                             int nb, int nb_all, int seg_lane, float (&out)[NE], const G* grid) {
    using F = Fmt<TY>;
    constexpr int KG = F::S::kGroup;
    static_assert(LPR % KG == 0 && (KG & (KG - 1)) == 0, "a group's calls on neighbouring lanes of one segment");
    int ta[NE];
    for (int e = 0; e < NE; ++e) ta[e] = e < n ? fx_act_first_top(x[e]) : canon::kFxNone;
    int top[NE];
    int64_t s[NE];
    // the sums at the weights' top topw and the activations' ta; `over`: a block's d or scale past them (the first
    // blocks' spans: a redo with the whole tops)
    const auto sums = [&](int topw, bool& over) {
        for (int e = 0; e < NE; ++e) {
            s[e] = 0;
            top[e] = ta[e] + topw;
        }
        const int total = nb_all * F::ipb, mine = nb * F::ipb;
        for (int k0 = 0; k0 < total; k0 += LPR) {
            const int k = k0 + seg_lane;
            const bool on = k < mine;
            const int kk = on ? k : 0, kbx = kk / F::ipb, iqs = F::step * (kk % F::ipb);
            const auto wd = [&] {
                if constexpr (std::is_void_v<G>) return F::S::load(block_at<F>(row, kbx), 0, iqs);
                else return F::S::load(block_at<F>(row, kbx), 0, iqs, grid);
            }();
            const canon::FxPart wp = canon::fx_part(wd.d);
            over = over || (on && wp.e > topw);
            const uint32_t xo32 = (uint32_t) kbx * (uint32_t) (F::qk / 32);   // in 32 bits (block_at)
            const size_t xo = xo32;
            for (int e = 0; e < NE; ++e)
                if (e < n) {   // n is the whole sub-group's: the butterflies stay uniform
                    float c = 0.0f;
                    int t = 0;
                    if (on) F::S::parts(wd, x[e] + xo, iqs, c, t);
                    for (int o = KG / 2; o > 0; o >>= 1) t += sycl::permute_group_by_xor(sg, t, o);
                    const canon::FxPart ap = canon::fx_part(F::S::act_d(x[e] + xo, iqs));
                    over = over || (on && ap.e > ta[e]);
                    if (on) s[e] += (int64_t) canon::fx_scale(ap, wp, top[e]) * t;
                }
        }
    };
    bool over = false;
    sums(fx_first_top<TY>(row, nb), over);
    if (sycl::any_of_group(sg, over)) {   // the whole sub-group again, each row at its whole tops
        for (int e = 0; e < NE; ++e)
            if (e < n) ta[e] = fx_act_top<TY, LPR, S256>(sg, x[e], nb_all, seg_lane);
        bool again = false;
        sums(fx_row_top<TY, LPR>(sg, row, nb, seg_lane), again);
    }
    for (int e = 0; e < NE; ++e) {
        for (int o = LPR / 2; o >= KG; o >>= 1) s[e] += sycl::permute_group_by_xor(sg, s[e], o);
        out[e] = canon::fx_value(s[e], top[e], F::S::kFxLog2);
    }
}

// S256: the activation's scale is one for each 256 values (the gate/up rows' x, quantize_q8_1_rows); a down row's h
// has one for each 32, where a format's calls of a block (kGroup > 1) do not share it: the plain loop
template<int TY, int LPR, int NE, class G = void, bool S256 = true>
inline void seg_dot_multi(const sycl::sub_group& sg, const uint8_t* row, const block_q8_1* const (&x)[NE], int n,
                          int nb, int nb_all, int seg_lane, float (&out)[NE], const G* grid = nullptr) {
    using F = Fmt<TY>;
    if constexpr (HasFx<typename F::S>::value && (S256 || group_of<typename F::S>() == 1)) {
        seg_dot_canon_fx<TY, LPR, NE, G, S256>(sg, row, x, n, nb, nb_all, seg_lane, out, grid);
        return;
    } else if constexpr (HasParts<typename F::S>::value && (S256 || group_of<typename F::S>() == 1)) {
        seg_dot_canon<TY, LPR, NE, G>(sg, row, x, n, nb, nb_all, seg_lane, out, grid);
        return;
    } else {
    (void) nb_all;
    float s[NE];
    for (int e = 0; e < NE; ++e) s[e] = 0.0f;
    for (int k = seg_lane; k < nb * F::ipb; k += LPR) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const auto wd = [&] {
            if constexpr (std::is_void_v<G>) return F::S::load(block_at<F>(row, kbx), 0, iqs);
            else return F::S::load(block_at<F>(row, kbx), 0, iqs, grid);
        }();
        const size_t xo = (size_t) ((uint32_t) kbx * (uint32_t) (F::qk / 32));  // NOLINT(bugprone-misplaced-widening-cast): a 32-bit product on purpose, kbx < nb
        for (int e = 0; e < NE; ++e)
            if (e < n) s[e] += F::S::apply(wd, x[e] + xo, iqs);
    }
    for (int e = 0; e < NE; ++e) {
        for (int o = LPR / 2; o > 0; o >>= 1) s[e] += sycl::permute_group_by_xor(sg, s[e], o);
        out[e] = s[e];
    }
    }
}


// One row against NC q8_1 activations at once (x + c * xstride), the whole sub-group: call k = (block, part) is
// lane-strided, each block loaded once and applied to every column (upstream 8ec94aa: the verify window's tokens
// re-read and re-decoded every row once each).  Column c's sum is what one column alone gives, bit for bit.
template<int TY, int NC>
inline void row_dot_multi(const sycl::sub_group& sg, const uint8_t* row, const block_q8_1* x, size_t xstride, int nb,
                          int lane, float (&out)[NC]) {
    using F = Fmt<TY>;
    float s[NC];
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += kWarp) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const auto wd = F::S::load(block_at<F>(row, kbx), 0, iqs);
        const size_t xo = (size_t) ((uint32_t) kbx * (uint32_t) (F::qk / 32));  // NOLINT(bugprone-misplaced-widening-cast): a 32-bit product on purpose, kbx < nb
        for (int c = 0; c < NC; ++c) s[c] += F::S::apply(wd, x + (size_t) c * xstride + xo, iqs);
    }
    for (int c = 0; c < NC; ++c) out[c] = warp_sum(sg, s[c]);
}

template<int TY, int NC>
sycl::event launch_mmvq_n(sycl::queue& q, const uint8_t* w, size_t row_bytes, const block_q8_1* x, float* y, int n_in,
                          int n_out) {
    const size_t groups = (size_t) ((n_out + 3) / 4);
    return q.parallel_for(sycl::nd_range<1>(groups * 4 * kWarp, (size_t) 4 * kWarp),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int row = (int) it.get_group(0) * 4 + (int) sg.get_group_linear_id();
        if (row >= n_out) return;
        const int lane = (int) sg.get_local_linear_id();
        const int nb = n_in / Fmt<TY>::qk;
        float s[NC];
        row_dot_multi<TY, NC>(sg, w + (size_t) row * row_bytes, x, (size_t) (n_in / 32), nb, lane, s);
        if (lane == 0)
            for (int c = 0; c < NC; ++c) y[(size_t) c * n_out + row] = s[c];
    });
}

// up to 8 columns in one pass, the count exact (upstream 2715a752: 3, 5, 6 and 7 took two or three passes); 8 at a
// time past 8 (the verify window takes up to 8 tokens)
template<int TY>
sycl::event launch_mmvq(sycl::queue& q, const uint8_t* w, size_t row_bytes, const block_q8_1* x, float* y, int n_in,
                        int n_out, int ncols) {
    sycl::event e;
    for (int c0 = 0; c0 < ncols;) {
        const int n = ncols - c0;
        const block_q8_1* xc = x + (size_t) c0 * (n_in / 32);
        float* yc = y + (size_t) c0 * n_out;
        const int take = n >= 8 ? 8 : n;
        switch (take) {
            case 8: e = launch_mmvq_n<TY, 8>(q, w, row_bytes, xc, yc, n_in, n_out); break;
            case 7: e = launch_mmvq_n<TY, 7>(q, w, row_bytes, xc, yc, n_in, n_out); break;
            case 6: e = launch_mmvq_n<TY, 6>(q, w, row_bytes, xc, yc, n_in, n_out); break;
            case 5: e = launch_mmvq_n<TY, 5>(q, w, row_bytes, xc, yc, n_in, n_out); break;
            case 4: e = launch_mmvq_n<TY, 4>(q, w, row_bytes, xc, yc, n_in, n_out); break;
            case 3: e = launch_mmvq_n<TY, 3>(q, w, row_bytes, xc, yc, n_in, n_out); break;
            case 2: e = launch_mmvq_n<TY, 2>(q, w, row_bytes, xc, yc, n_in, n_out); break;
            default: e = launch_mmvq_n<TY, 1>(q, w, row_bytes, xc, yc, n_in, n_out);
        }
        c0 += take;
    }
    return e;
}

// ---------------------------------------------------------------- grouped native experts
// Lanes a row in the experts' projections, measured on the B70 (the 2026-10-02-xe-decode-gpu record): 16 for the
// 2560-wide gate/up rows, 8 for the 640-wide down rows (32 lanes left most of a 180-360 byte row's lanes idle).  A
// down row of under 12 bytes per 32 values (Q2_0's 180) takes 4: a verify window's grouped experts 3-6% faster on the
// B70 and on an RTX 4070 (scratch lpr_probe), where IQ4_NL's 360-byte rows were 11% slower with 4 (8 and 16 for gate
// and up were slower on one or both).
constexpr int GU_LPR = 16;
constexpr int DOWN_LPR = 8;
template<int TG> constexpr int down_lpr() { return Fmt<TG>::bsz * 32 / Fmt<TG>::qk < 12 ? 4 : DOWN_LPR; }
constexpr int GU_ROWS = 8;     // rows per work-group (one sub-group each)
// entries (a window's tokens routed to one expert) a pass over the row; the verify window of --spec 4 fits in one
constexpr int kEntriesPass = 4;

// The experts' projections: LPR lanes a row (seg_dot), GU_ROWS sub-groups a work-group, so a work-group covers
// GU_ROWS * kWarp / LPR rows.  A row in range for some lanes of a sub-group and not others is masked per segment,
// with no early return, so the segment shuffles see the whole sub-group.
// `part`: -1 the gate and up rows, 0 the gate rows only, 1 the up rows only (a layer whose gate and up differ in type
// takes one launch a type)
template<int TG, int LPR>
sycl::event launch_native_gu(sycl::queue& q, size_t cap_groups, const unsigned long long* grp_ptr,
                             const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok,
                             const block_q8_1* xq, NativeExpertLayout L, float* gate, float* up, int part = -1) {
    constexpr int RPW = kWarp / LPR;
    const int row_lo = part == 1 ? (int) L.n_ff : 0, row_hi = part == 0 ? (int) L.n_ff : (int) (2 * L.n_ff);
    const size_t xgroups = (size_t) ((row_hi - row_lo + (int64_t) GU_ROWS * RPW - 1) / ((int64_t) GU_ROWS * RPW));
    return q.parallel_for(sycl::nd_range<2>({cap_groups, xgroups * GU_ROWS * kWarp}, {1, (size_t) GU_ROWS * kWarp}),
                          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kWarp)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int warp = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int row0 = row_lo + ((int) it.get_group(1) * GU_ROWS + warp) * RPW;
        if (row0 >= row_hi) return;                                          // the whole sub-group
        const int row = row0 + lane / LPR, seg_lane = lane % LPR;
        const bool valid = row < row_hi;
        const bool is_up = row >= L.n_ff;
        const int r = valid ? (is_up ? row - (int) L.n_ff : row) : 0;
        const int nb = valid ? (int) (L.n_embd / Fmt<TG>::qk) : 0, xb = (int) (L.n_embd / 32);
        // the groups g, g + grid rows, ...: a group's rows are the same code in the same order whatever the grid
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const uint8_t* wr = blob + (is_up && valid ? L.up_off + (size_t) r * L.up_row : (size_t) r * L.gu_row);
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; e += kEntriesPass) {   // the row's blocks decoded once for kEntriesPass entries
                const int n = sycl::min(kEntriesPass, e1 - e);
                const block_q8_1* xs[kEntriesPass];
                for (int j = 0; j < kEntriesPass; ++j) xs[j] = xq + (size_t) ent_tok[j < n ? e + j : e] * xb;
                float s[kEntriesPass];
                seg_dot_multi<TG, LPR, kEntriesPass>(sg, wr, xs, n, nb, (int) (L.n_embd / Fmt<TG>::qk), seg_lane, s);
                if (valid && seg_lane == 0)
                    for (int j = 0; j < n; ++j) (is_up ? up : gate)[(size_t) (e + j) * L.n_ff + r] = s[j];
            }
        }
    });
}

// Two gate/up rows share each loaded Q8 activation block.  Each row keeps the same integer butterflies and
// accumulation order as seg_dot_canon; only formats whose parts read exactly one Q8 block can take this form.
template<int TG>
sycl::event launch_native_gu_pair(sycl::queue& q, size_t cap_groups, const unsigned long long* grp_ptr,
                                  const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok,
                                  const block_q8_1* xq, NativeExpertLayout L, float* gate, float* up, int part) {
    static_assert(TG == 18 || TG == 21 || TG == 22);
    using F = Fmt<TG>;
    constexpr int LPR = GU_LPR, RPW = kWarp / LPR, KG = F::S::kGroup;
    const int row_lo = part == 1 ? (int) L.n_ff : 0, row_hi = part == 0 ? (int) L.n_ff : (int) (2 * L.n_ff);
    const size_t xgroups = (size_t) ((row_hi - row_lo + 2 * GU_ROWS * RPW - 1) / (2 * GU_ROWS * RPW));
    return q.parallel_for(sycl::nd_range<2>({cap_groups, xgroups * GU_ROWS * kWarp}, {1, (size_t) GU_ROWS * kWarp}),
                          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kWarp)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int warp = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int row0 = row_lo + ((int) it.get_group(1) * GU_ROWS + warp) * 2 * RPW + lane / LPR;
        const int rows[2] = {row0, row0 + RPW};
        if (row0 - lane / LPR >= row_hi) return;
        const int seg_lane = lane % LPR, nb_all = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32);
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const uint8_t* wr[2];
            for (int r = 0; r < 2; ++r) {
                const bool valid = rows[r] < row_hi, is_up = rows[r] >= L.n_ff;
                const int idx = valid ? (is_up ? rows[r] - (int) L.n_ff : rows[r]) : 0;
                wr[r] = blob + (is_up && valid ? L.up_off + (size_t) idx * L.up_row : (size_t) idx * L.gu_row);
            }
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            // seg_dot_canon_fx's sums, each row's own top: its first block's (fx_first_top) until a block past it shows
            // up in the first pass over the entries, then the whole (fx_row_top) from that pass on
            [[maybe_unused]] int topw[2] = {canon::kFxNone, canon::kFxNone};
            [[maybe_unused]] bool whole = false;
            if constexpr (HasFx<typename F::S>::value)
                for (int r = 0; r < 2; ++r) topw[r] = fx_first_top<TG>(wr[r], rows[r] < row_hi ? nb_all : 0);
            for (int e = e0; e < e1; e += kEntriesPass) {
                const int n = sycl::min(kEntriesPass, e1 - e);
                const block_q8_1* xs[kEntriesPass];
                for (int j = 0; j < kEntriesPass; ++j) xs[j] = xq + (size_t) ent_tok[j < n ? e + j : e] * xb;
                if constexpr (HasFx<typename F::S>::value) {
                    int ta[kEntriesPass], top[2][kEntriesPass];
                    int64_t fs[2][kEntriesPass];
                    for (int j = 0; j < kEntriesPass; ++j) ta[j] = j < n ? fx_act_first_top(xs[j]) : canon::kFxNone;
                    const auto sums = [&](bool& over) {
                        for (int j = 0; j < kEntriesPass; ++j)
                            for (int r = 0; r < 2; ++r) {
                                top[r][j] = ta[j] + topw[r];
                                fs[r][j] = 0;
                            }
                        for (int k0 = 0; k0 < nb_all * F::ipb; k0 += LPR) {
                            const int k = k0 + seg_lane;
                            const bool active = k < nb_all * F::ipb;
                            const int kk = active ? k : 0, kbx = kk / F::ipb, iqs = F::step * (kk % F::ipb);
                            const auto w0 = F::S::load(block_at<F>(wr[0], kbx), 0, iqs);
                            const auto w1 = F::S::load(block_at<F>(wr[1], kbx), 0, iqs);
                            const canon::FxPart p0 = canon::fx_part(w0.d), p1 = canon::fx_part(w1.d);
                            over = over || (active && rows[0] < row_hi && p0.e > topw[0]) ||
                                   (active && rows[1] < row_hi && p1.e > topw[1]);
                            const size_t xo =
                                (size_t) ((uint32_t) kbx * (uint32_t) (F::qk / 32) + (uint32_t) (iqs / 2));
                            for (int j = 0; j < n; ++j) {
                                const block_q8_1 act = xs[j][xo];
                                const canon::FxPart pa = canon::fx_part(F::S::act_d(&act, 0));
                                over = over || (active && pa.e > ta[j]);
                                float c0 = 0.0f, c1 = 0.0f;
                                int t0 = 0, t1 = 0;
                                if (active && rows[0] < row_hi) F::S::parts(w0, &act, 0, c0, t0);
                                if (active && rows[1] < row_hi) F::S::parts(w1, &act, 0, c1, t1);
                                for (int o = KG / 2; o > 0; o >>= 1) {
                                    t0 += sycl::permute_group_by_xor(sg, t0, o);
                                    t1 += sycl::permute_group_by_xor(sg, t1, o);
                                }
                                if (active && rows[0] < row_hi)
                                    fs[0][j] += (int64_t) canon::fx_scale(pa, p0, top[0][j]) * t0;
                                if (active && rows[1] < row_hi)
                                    fs[1][j] += (int64_t) canon::fx_scale(pa, p1, top[1][j]) * t1;
                            }
                        }
                    };
                    bool over = false;
                    sums(over);
                    if (sycl::any_of_group(sg, over)) {   // the whole tops (the rows' from here on)
                        if (!whole)
                            for (int r = 0; r < 2; ++r)
                                topw[r] = fx_row_top<TG, LPR>(sg, wr[r], rows[r] < row_hi ? nb_all : 0, seg_lane);
                        whole = true;
                        for (int j = 0; j < kEntriesPass; ++j)
                            if (j < n) ta[j] = fx_act_top<TG, LPR, true>(sg, xs[j], nb_all, seg_lane);
                        bool again = false;
                        sums(again);
                    }
                    for (int r = 0; r < 2; ++r) {
                        for (int j = 0; j < n; ++j)
                            for (int o = LPR / 2; o >= KG; o >>= 1)
                                fs[r][j] += sycl::permute_group_by_xor(sg, fs[r][j], o);
                        if (rows[r] < row_hi && seg_lane == 0) {
                            const bool is_up = rows[r] >= L.n_ff;
                            const int idx = is_up ? rows[r] - (int) L.n_ff : rows[r];
                            for (int j = 0; j < n; ++j)
                                (is_up ? up : gate)[(size_t) (e + j) * L.n_ff + idx] =
                                    canon::fx_value(fs[r][j], top[r][j], F::S::kFxLog2);
                        }
                    }
                    continue;
                }
                float sum[2][kEntriesPass] = {};
                for (int k0 = 0; k0 < nb_all * F::ipb; k0 += LPR) {
                    const int k = k0 + seg_lane;
                    const bool active = k < nb_all * F::ipb;
                    const int kk = active ? k : 0, kbx = kk / F::ipb, iqs = F::step * (kk % F::ipb);
                    const auto w0 = F::S::load(block_at<F>(wr[0], kbx), 0, iqs);
                    const auto w1 = F::S::load(block_at<F>(wr[1], kbx), 0, iqs);
                    const size_t xo = (size_t) ((uint32_t) kbx * (uint32_t) (F::qk / 32) + (uint32_t) (iqs / 2));
                    for (int j = 0; j < n; ++j) {
                        const block_q8_1 act = xs[j][xo];
                        float c0 = 0.0f, c1 = 0.0f;
                        int t0 = 0, t1 = 0;
                        if (active && rows[0] < row_hi) F::S::parts(w0, &act, 0, c0, t0);
                        if (active && rows[1] < row_hi) F::S::parts(w1, &act, 0, c1, t1);
                        for (int o = KG / 2; o > 0; o >>= 1) {
                            t0 += sycl::permute_group_by_xor(sg, t0, o);
                            t1 += sycl::permute_group_by_xor(sg, t1, o);
                        }
                        if (active && rows[0] < row_hi) sum[0][j] = canon::fma(c0, (float) t0, sum[0][j]);
                        if (active && rows[1] < row_hi) sum[1][j] = canon::fma(c1, (float) t1, sum[1][j]);
                    }
                }
                for (int r = 0; r < 2; ++r) {
                    for (int j = 0; j < n; ++j)
                        for (int o = LPR / 2; o >= KG; o >>= 1) sum[r][j] += sycl::permute_group_by_xor(sg, sum[r][j], o);
                    if (rows[r] < row_hi && seg_lane == 0) {
                        const bool is_up = rows[r] >= L.n_ff;
                        const int idx = is_up ? rows[r] - (int) L.n_ff : rows[r];
                        for (int j = 0; j < n; ++j) (is_up ? up : gate)[(size_t) (e + j) * L.n_ff + idx] = sum[r][j];
                    }
                }
            }
        }
    });
}

using GuPairKey = std::tuple<int, int64_t, int64_t, size_t, size_t, size_t, int>;
core::TimedCache<sycl::device, GuPairKey, bool> g_gu_pair_choices;
core::TimedCache<sycl::device, int, bool> g_gu_pair_warmed;
GuPairKey gu_pair_key(const NativeExpertLayout& L, int type, int part) {
    return {type, L.n_embd, L.n_ff, L.gu_row, L.up_row, L.up_off, part};
}
bool gu_pair_type(int type) { return type == 18 || type == 21 || type == 22; }
int gu_pair_mode() {
    const char* v = strata::env_get("IQ_GU_PAIR");
    if (!v || !*v || std::strcmp(v, "auto") == 0) return -1;
    if (std::strcmp(v, "0") == 0) return 0;
    if (std::strcmp(v, "1") == 0) return 1;
    throw core::DeviceError("XESTRATA_IQ_GU_PAIR must be auto, 0 or 1");
}
bool gu_pair_selected(sycl::queue& q, const NativeExpertLayout& L, int type, int part) {
    if (!gu_pair_type(type)) return false;
    const int mode = gu_pair_mode();
    if (mode >= 0) return mode == 1;
    return g_gu_pair_choices.get(q.get_device(), gu_pair_key(L, type, part)).value_or(false);
}

template<int TD, int LPR>
sycl::event launch_native_down(sycl::queue& q, size_t cap_groups, const unsigned long long* grp_ptr,
                               const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                               const block_q8_1* hq, NativeExpertLayout L, float* out) {
    constexpr int RPW = kWarp / LPR;
    const size_t xgroups = (size_t) ((L.n_embd + (int64_t) 8 * RPW - 1) / ((int64_t) 8 * RPW));
    return q.parallel_for(sycl::nd_range<2>({cap_groups, xgroups * 8 * kWarp}, {1, (size_t) 8 * kWarp}),
                          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kWarp)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int warp = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int r0 = ((int) it.get_group(1) * 8 + warp) * RPW;
        if (r0 >= L.n_embd) return;                                          // the whole sub-group
        const int r = r0 + lane / LPR, seg_lane = lane % LPR;
        const bool valid = r < L.n_embd;
        const int nb = valid ? (int) (L.n_ff / Fmt<TD>::qk) : 0, hb = (int) (L.n_ff / 32);
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {   // as gate/up
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const uint8_t* wr = blob + L.down_off + (size_t) (valid ? r : 0) * L.d_row;
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; e += kEntriesPass) {
                const int n = sycl::min(kEntriesPass, e1 - e);
                const block_q8_1* xs[kEntriesPass];
                for (int j = 0; j < kEntriesPass; ++j) xs[j] = hq + (size_t) (j < n ? e + j : e) * hb;
                float s[kEntriesPass];
                seg_dot_multi<TD, LPR, kEntriesPass, void, false>(sg, wr, xs, n, nb, (int) (L.n_ff / Fmt<TD>::qk), seg_lane, s);
                if (valid && seg_lane == 0)
                    for (int j = 0; j < n; ++j) out[(size_t) ent_dst[e + j] * L.n_embd + r] = s[j];
            }
        }
    });
}

// ---------------------------------------------------------------- the Volta experts (upstream f6330507)
// Upstream's expert mode 8, gfx906's layout, which upstream runs on sm_70 too (V100-SXM2, a verify window's VRAM call:
// 227 -> 159 us).  A work-group holds a group's kEntriesPass entries' q8_1 activations and, for a grid format, its
// codebook in local memory.  gate/up: a work-group takes h rows [r0, r0 + 32) of one group (gate rows r0.. and up
// rows r0.., 64 weight rows) and ends with SwiGLU and the q8_1 quantization, a sub-group an entry and one q8_1 block
// of h each, as the one-pass kernel in native_expert_grouped.  down (mode 7's, IQ4_NL and Q2_0): 64 rows a
// work-group against the entries' h.  A row's sum is seg_dot_multi's with the default kernels' LPR (upstream's are
// its CUDA default's 32 lanes a row), so the output is bitwise the default kernels'.
constexpr int kM8Rows = 64;   // weight rows a work-group

template<int TG, int LPR>
sycl::event launch_native_gu_fused(sycl::queue& q, size_t cap_groups, const unsigned long long* grp_ptr,
                                   const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok,
                                   const block_q8_1* xq, NativeExpertLayout L, block_q8_1* hq) {
    using GT = typename GridOf<TG>::T;
    constexpr int NG = GridOf<TG>::N, RPW = kWarp / LPR, WG = GU_ROWS * kWarp;
    static_assert(kM8Rows % (GU_ROWS * RPW) == 0, "whole passes");
    const int xb = (int) (L.n_embd / 32);
    const float lim = expert_swiglu_limit();
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<GT, 1> sgrid(sycl::range<1>(NG > 0 ? NG : 1), h);
        sycl::local_accessor<float, 1> res(sycl::range<1>((size_t) kEntriesPass * kM8Rows), h);
        sycl::local_accessor<int, 1> sx(sycl::range<1>((size_t) kEntriesPass * xb * 9), h);   // 36-byte blocks
        h.parallel_for(sycl::nd_range<2>({cap_groups, (size_t) (L.n_ff / 32) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kWarp)]] {
            const int g = (int) it.get_group(0);
            if (g >= *n_groups) return;   // the whole work-group
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(1);
            const int warp = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int seg_lane = lane % LPR;
            [[maybe_unused]] const GT* grid = nullptr;   // IQ4_XS: none
            if constexpr (NG > 0) {
                const GT* src = GridOf<TG>::src();
                for (int i = tid; i < NG; i += WG) sgrid[i] = src[i];
                grid = sgrid.template get_multi_ptr<sycl::access::decorated::no>().get();
            }
            const auto* sxb = reinterpret_cast<const block_q8_1*>(sx.template get_multi_ptr<sycl::access::decorated::no>().get());
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int nb = (int) (L.n_embd / Fmt<TG>::qk), hb = (int) (L.n_ff / 32);
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            const int r0 = (int) it.get_group(1) * 32;
            for (int c0 = e0; c0 < e1; c0 += kEntriesPass) {
                const int cn = sycl::min(kEntriesPass, e1 - c0);
                sycl::group_barrier(it.get_group());
                for (int j = 0; j < cn; ++j) {
                    const int* src = (const int*) (xq + (size_t) ent_tok[c0 + j] * xb);
                    for (int i = tid; i < xb * 9; i += WG) sx[j * xb * 9 + i] = src[i];
                }
                sycl::group_barrier(it.get_group());
                const block_q8_1* xs[kEntriesPass];
                for (int j = 0; j < kEntriesPass; ++j) xs[j] = sxb + (size_t) (j < cn ? j : 0) * xb;
                for (int p = 0; p < kM8Rows / (GU_ROWS * RPW); ++p) {
                    const int i = (p * GU_ROWS + warp) * RPW + lane / LPR;   // gate row r0 + i, or up row r0 + i - 32
                    const uint8_t* wr = blob + (i < 32 ? (size_t) 0 : L.up_off) + (size_t) (r0 + (i & 31)) * L.gu_row;
                    float s[kEntriesPass];
                    if constexpr (NG > 0) seg_dot_multi<TG, LPR, kEntriesPass>(sg, wr, xs, cn, nb, nb, seg_lane, s, grid);
                    else seg_dot_multi<TG, LPR, kEntriesPass>(sg, wr, xs, cn, nb, nb, seg_lane, s);
                    if (seg_lane == 0)
                        for (int j = 0; j < cn; ++j) res[j * kM8Rows + i] = s[j];
                }
                sycl::group_barrier(it.get_group());
                if (warp < cn) {   // native_expert_grouped's one-pass SwiGLU + q8_1 on entry c0 + warp's block
                    // the SwiGLU (after GLM-5.3-Flash's clamp; none elsewhere) and the int8 scale of canon_math.hpp,
                    // as native_expert_grouped's pass
                    float gv = res[warp * kM8Rows + lane], uv = res[warp * kM8Rows + 32 + lane];
                    swiglu_clamp(gv, uv, lim);
                    auto* y = reinterpret_cast<uint8_t*>(hq + (size_t) (c0 + warp) * hb + it.get_group(1));
                    q8act::q8_1_block<q8act::Canon<32>>(sg, canon::silu_mul(gv, uv), y, lane);
                }
            }
        });
    });
}

// WS (STRATA_DOWN_WS, upstream 6a6d58ad0: down_ws_on): the work-group's kM8Rows weight rows are first copied into
// local memory, 16 bytes a load where the rows' start allows, and the row dots read that copy.  The copy keeps the
// rows' start address modulo 16, so every load inside a row sees the alignment it has in device memory, and the sums
// are the same code on the same bytes: the output is bitwise the kernel's without WS.
template<int TD, int LPR, bool WS = false>
sycl::event launch_native_down_lds(sycl::queue& q, size_t cap_groups, const unsigned long long* grp_ptr,
                                   const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                                   const block_q8_1* hq, NativeExpertLayout L, float* out) {
    constexpr int RPW = kWarp / LPR, WG = 8 * kWarp;
    static_assert(kM8Rows % (8 * RPW) == 0, "whole passes");
    const int hb = (int) (L.n_ff / 32);
    const size_t wvecs = WS ? ((size_t) kM8Rows * L.d_row + 15) / 16 + 1 : 1;   // + 1: room to keep the alignment
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> sh(sycl::range<1>((size_t) kEntriesPass * hb * 9), h);
        sycl::local_accessor<sycl::vec<uint32_t, 4>, 1> sw(sycl::range<1>(wvecs), h);
        h.parallel_for(sycl::nd_range<2>({cap_groups, (size_t) ((L.n_embd + kM8Rows - 1) / kM8Rows) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kWarp)]] {
            const int g = (int) it.get_group(0);
            if (g >= *n_groups) return;
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(1);
            const int warp = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int seg_lane = lane % LPR;
            const auto* shb = reinterpret_cast<const block_q8_1*>(sh.template get_multi_ptr<sycl::access::decorated::no>().get());
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            const int rb = (int) it.get_group(1) * kM8Rows;   // the work-group's first row
            [[maybe_unused]] const uint8_t* lw = nullptr;     // its rows in local memory (WS)
            if constexpr (WS) {
                const uint8_t* wbase = blob + L.down_off + (size_t) rb * L.d_row;
                const size_t nbytes = (size_t) sycl::min(kM8Rows, (int) L.n_embd - rb) * L.d_row;
                const size_t off = (size_t) ((uintptr_t) wbase & 15);
                uint8_t* lb = reinterpret_cast<uint8_t*>(sw.template get_multi_ptr<sycl::access::decorated::no>().get());
                if (off == 0 && nbytes % 16 == 0) {
                    const auto* src = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(wbase);
                    for (size_t i = (size_t) tid; i < nbytes / 16; i += WG) sw[i] = src[i];
                } else {
                    for (size_t i = (size_t) tid; i < nbytes; i += WG) lb[off + i] = wbase[i];
                }
                lw = lb + off;   // made visible by the barrier before the entries' activations are read
            }
            for (int c0 = e0; c0 < e1; c0 += kEntriesPass) {
                const int cn = sycl::min(kEntriesPass, e1 - c0);
                sycl::group_barrier(it.get_group());
                const int* src = (const int*) (hq + (size_t) c0 * hb);   // a group's entries are contiguous
                for (int i = tid; i < cn * hb * 9; i += WG) sh[i] = src[i];
                sycl::group_barrier(it.get_group());
                const block_q8_1* xs[kEntriesPass];
                for (int j = 0; j < kEntriesPass; ++j) xs[j] = shb + (size_t) (j < cn ? j : 0) * hb;
                for (int p = 0; p < kM8Rows / (8 * RPW); ++p) {
                    const int r0 = (int) it.get_group(1) * kM8Rows + (p * 8 + warp) * RPW;
                    if (r0 >= L.n_embd) break;                                       // the whole sub-group
                    const int r = r0 + lane / LPR;
                    const bool valid = r < L.n_embd;
                    const int nb = valid ? (int) (L.n_ff / Fmt<TD>::qk) : 0;
                    float s[kEntriesPass];
                    const uint8_t* wr = WS ? lw + (size_t) (valid ? r - rb : 0) * L.d_row
                                           : blob + L.down_off + (size_t) (valid ? r : 0) * L.d_row;
                    seg_dot_multi<TD, LPR, kEntriesPass, void, false>(sg, wr, xs, cn, nb, (int) (L.n_ff / Fmt<TD>::qk),
                                                                      seg_lane, s);
                    if (valid && seg_lane == 0)
                        for (int j = 0; j < cn; ++j) out[(size_t) ent_dst[c0 + j] * L.n_embd + r] = s[j];
                }
            }
        });
    });
}

// Mode 8 on the queue's device: the default, where its local memory fits (the callers check it; the same bits as the
// default kernels).  The window's entries' q8_1 rows are read once into local memory instead of by every weight row
// (upstream f7eaf2de's STRATA_EXPDEC idea): B70 decode +0.8..+1.1% at 1K and 32K (two A/B runs), RTX 4070 equal.
// STRATA_EXP_MODE=<another value> (0): the default kernels; upstream's modes 1-7 are gfx906's and not here.
bool exp_mode8(sycl::queue& q) {
    static const int env = [] {
        const char* v = strata::env_get("EXP_MODE");
        return v != nullptr ? (int) std::strtol(v, nullptr, 10) : -1;
    }();
    (void) q;
    return env < 0 || env == 8;
}

size_t local_mem_bytes(sycl::queue& q) {
    static core::PerDevice<size_t> per_device;
    const sycl::device d = q.get_device();
    return per_device.get(d, [&d] { return (size_t) d.get_info<sycl::info::device::local_mem_size>(); });
}

// STRATA_DOWN_WS=1 (upstream 6a6d58ad0, opt-in): the mode-8 down rows of IQ4_NL and Q2_0 staged in local memory
// (launch_native_down_lds' WS) on a device that reports compute capability 6.0 (Pascal GP100: no dp4a, its row dots
// wait on 2-byte loads from device memory).  Upstream measured it on one GP100 die only (a one-token window's down
// 72.6 / 53.5 us -> 42.3 / 38.6 us for IQ4_NL / Q2_0); there is no such GPU here.  Elsewhere the switch does nothing.
bool down_ws_on(sycl::queue& q) {
    static const bool env = [] {
        const char* v = strata::env_get("DOWN_WS");
        return v != nullptr && v[0] == '1';
    }();
    if (!env) return false;
    static core::PerDevice<bool> gp100;
    const sycl::device d = q.get_device();
    return gp100.get(d, [&d] {
        namespace syclex = sycl::ext::oneapi::experimental;
        return d.get_backend() == sycl::backend::ext_oneapi_cuda &&
               d.get_info<syclex::info::device::architecture>() == syclex::architecture::nvidia_gpu_sm_60;
    });
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
// n is a multiple of 32, so every sub-group is either wholly in range or wholly out of it.  canon_math.hpp's scale (the
// experts' activations: the CPU's rows of an expert quantize alike).
sycl::event launch_quantize_q8_1_canon(sycl::queue& q, const float* x, block_q8_1* y, long long n) {
    return q8act::quantize<1, q8act::Canon<32>>(q, n / 32, 0, q8act::Flat{x},
                                                q8act::Q8_1{reinterpret_cast<uint8_t*>(y)});
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
// The dequantizers hand their values to an output, runs of 4 or 8 at an offset in the 256-value piece they decode:
// ToGlobal stores them to memory, iq_gemm's ToLocal to a work-group's tile in local memory.
template<typename dst_t> inline dst_t cvt(float v);
template<> inline float cvt<float>(float v) { return v; }
template<> inline sycl::half cvt<sycl::half>(float v) { return sycl::half(v); }
// N consecutive values in one vector store instead of N scalar ones (y is aligned to N values in every caller): the
// dequant's fp16 writes were 128-163 GB/s with one 2-byte store per value (the 2026-10-02-xe-decode-gpu record).
template<int N, typename dst_t>
inline void store(dst_t* y, const float* v) {
    sycl::vec<dst_t, N> o;
    for (int j = 0; j < N; ++j) o[j] = cvt<dst_t>(v[j]);  // NOLINT(clang-analyzer-core.CallAndMessage): every caller fills v[0 .. N)
    *reinterpret_cast<sycl::vec<dst_t, N>*>(y) = o;
}

template<typename dst_t>
struct ToGlobal {
    dst_t* y;
    template<int N> void put(int off, const float* v) const { store<N>(y + off, v); }
};

template<typename Out>
void dq_iq2_xxs(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid % 4, ib = tid / 4;          // neighbours in one sub-block: contiguous writes
    const int y = (int) (32 * ib + 8 * il);
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t aux8 = (uint8_t) (q2[il / 2] >> (8 * (il % 2)));
    const uint64_t grid = iq2xxs_grid[aux8];   // whole, and the signs computed: as dq_iq3_xxs
    const uint32_t aux32 = q2[2] | ((uint32_t) q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (float) (aux32 >> 28)) * 0.25f;
    const uint32_t signs = ksigns_byte((uint8_t) ((aux32 >> 7 * il) & 127));
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = d * (float) (uint8_t) (grid >> (8 * j)) * (signs & (1u << j) ? -1.f : 1.f);
    o.template put<8>(y, v);
}
template<typename Out>
void dq_iq2_xs(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid % 4, ib = tid / 4;          // neighbours in one sub-block: contiguous writes
    const int y = (int) (32 * ib + 8 * il);
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint64_t grid = iq2xs_grid[q2[il] & 511];   // whole, and the signs computed: as dq_iq3_xxs
    const float d = (float) x[ibs].d * (0.5f + (float) ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint32_t signs = ksigns_byte((uint8_t) (q2[il] >> 9));
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = d * (float) (uint8_t) (grid >> (8 * j)) * (signs & (1u << j) ? -1.f : 1.f);
    o.template put<8>(y, v);
}
template<typename Out>
void dq_iq2_s(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid % 4, ib = tid / 4;          // neighbours in one sub-block: contiguous writes
    const int y = (int) (32 * ib + 8 * il);
    const uint64_t grid = iq2s_grid[x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)];   // whole
    const float d = (float) x[ibs].d * (0.5f + (float) ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint32_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = d * (float) (uint8_t) (grid >> (8 * j)) * (signs & (1u << j) ? -1.f : 1.f);
    o.template put<8>(y, v);
}
template<typename Out>
void dq_iq3_xxs(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid % 4, ib = tid / 4;          // neighbours in one sub-block: contiguous writes
    const int y = (int) (32 * ib + 8 * il);
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    // the grid words whole and the signs computed (ksigns_byte is ksigns_iq2xs): a load a byte of the grid and the
    // tables' lookups made the dequant wait on loads (Arc A380: IQ3_XXS 16 experts 2.69 ms)
    const uint32_t grid1 = iq3xxs_grid[q3[2 * il + 0]];
    const uint32_t grid2 = iq3xxs_grid[q3[2 * il + 1]];
    const uint32_t aux32 = gas[0] | ((uint32_t) gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (float) (aux32 >> 28)) * 0.5f;
    const uint32_t signs = ksigns_byte((uint8_t) ((aux32 >> 7 * il) & 127));
    float v[8];
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * (float) byte_of(grid1, j) * (signs & (1u << j) ? -1.f : 1.f);
        v[j + 4] = d * (float) byte_of(grid2, j) * (signs & (1u << (j + 4)) ? -1.f : 1.f);
    }
    o.template put<8>(y, v);
}
template<typename Out>
void dq_iq3_s(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid % 4, ib = tid / 4;          // neighbours in one sub-block: contiguous writes
    const int y = (int) (32 * ib + 8 * il);
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)];   // whole words
    const uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)];
    const float d = (float) x[ibs].d * (float) (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint32_t signs = x[ibs].signs[4 * ib + il];
    float v[8];
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * (float) byte_of(grid1, j) * (signs & (1u << j) ? -1.f : 1.f);
        v[j + 4] = d * (float) byte_of(grid2, j) * (signs & (1u << (j + 4)) ? -1.f : 1.f);
    }
    o.template put<8>(y, v);
}
template<typename Out>
void dq_iq1_s(const void* vx, int64_t ibs, const Out& o, int tid) {   // ggml-cuda's dequantize_block_iq1_s
    const block_iq1_s* x = (const block_iq1_s*) vx;
    const int64_t il = tid % 4, ib = tid / 4;          // neighbours in one sub-block: contiguous writes
    const int y = (int) (32 * ib + 8 * il);
    const int qh = x[ibs].qh[ib];
    const float d = (float) x[ibs].d * (float) (2 * ((qh >> 12) & 7) + 1);
    const float delta = qh & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const uint32_t g = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((qh >> 3 * il) & 7) << 8)];
    const uint32_t grid32[2] = {g & 0x0f0f0f0f, (g >> 4) & 0x0f0f0f0f};
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = d * ((float) (int8_t) byte_of(grid32[j / 4], j % 4) + delta);
    o.template put<8>(y, v);
}
template<typename Out>
void dq_iq1_m(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid % 4, ib = tid / 4;          // neighbours in one sub-block: contiguous writes
    const int y = (int) (32 * ib + 8 * il);
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = iq1m_scale(sc) * (float) (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    const uint32_t g = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    const uint32_t grid32[2] = {g & 0x0f0f0f0f, (g >> 4) & 0x0f0f0f0f};
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = d * ((float) (int8_t) byte_of(grid32[j / 4], j % 4) + delta);
    o.template put<8>(y, v);
}
template<typename Out>
void dq_iq4_nl(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    // four neighbouring work-items share a block, so a sub-group writes whole runs (one block each was 64 bytes
    // apart: IQ4_NL's down matrix dequantized at 156 GB/s against Q2_0's 1 TB/s)
    const int64_t il = tid % 4, ib = tid / 4;
    const int y = (int) (32 * ib + 4 * il);
    // the codebook for all eight nibbles at once, as the dot product reads it, instead of eight table loads
    const Int2 v = get_int_from_table_16(get_int_b2(x[ib].qs, (int) il), kvalues_iq4nl);
    const float d = (float) x[ib].d;
    float lo[4], hi[4];
    for (int j = 0; j < 4; ++j) {
        lo[j] = d * (float) (int8_t) byte_of((uint32_t) v.x, j);
        hi[j] = d * (float) (int8_t) byte_of((uint32_t) v.y, j);
    }
    o.template put<4>(y, lo);
    o.template put<4>(y + 16, hi);
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename Out>
void dq_q3_k(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (float) (us - 32);
        const uint8_t* q = x->qs + (size_t) 32 * n;
        const uint8_t* hm = x->hmask;
        float v[4];
        for (int l = l0; l < l0 + 4; ++l)
            v[l - l0] = dl * (float) ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
        o.template put<4>(128 * n + 32 * j + l0, v);
    }
}
// Q2_K (Ornith-1.5's GSQ-RCO experts): llama.cpp's dequantize_block_q2_K, 8 consecutive values a work-item - value
// 128 n + 32 k + l is plane k (bits 2k) of qs[32 n + l], under scale byte 8 n + 2 k + l / 16
template<typename Out>
void dq_q2_k(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_q2_K* x = (const block_q2_K*) vx + ibs;
    const int n = tid / 16, k = (tid % 16) / 4, l0 = 8 * (tid % 4);
    const uint8_t sc = x->scales[8 * n + 2 * k + l0 / 16];
    const float dl = (float) x->dm[0] * (float) (sc & 0xF), ml = (float) x->dm[1] * (float) (sc >> 4);
    const uint8_t* q = x->qs + 32 * (ptrdiff_t) n;
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = dl * (float) ((int8_t) ((q[l0 + j] >> (2 * k)) & 3)) - ml;
    o.template put<8>(128 * n + 32 * k + l0, v);
}
template<typename Out>
void dq_iq4_xs(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid % 4, ib = tid / 4;              // neighbours in one sub-block, as dq_iq4_nl
    const int y = (int) (32 * ib + 4 * il);
    const uint8_t* q4 = x->qs + (size_t) 16 * ib + (size_t) 4 * il;
    const float d = (float) x->d * (float) ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    float lo[4], hi[4];
    for (int j = 0; j < 4; ++j) {
        lo[j] = d * (float) kvalues_iq4nl[q4[j] & 0xf];
        hi[j] = d * (float) kvalues_iq4nl[q4[j] >> 4];
    }
    o.template put<4>(y, lo);
    o.template put<4>(y + 16, hi);
}
template<typename Out>
void dq_q2_0(const void* vx, int64_t ibs, const Out& o, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    const uint32_t codes = ((const uint16_t*) x[b].qs)[part];   // the 8 values' 2-bit codes, the lowest first
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = d * (float) ((int) ((codes >> (2 * j)) & 3) - 1);
    o.template put<8>(b * 64 + part * 8, v);
}

// UD-Q4_K_XL's formats (upstream efeffd8): llama.cpp's dequantize_block_q4_K / q5_K and dequantize_q5_1 / q8_0, one
// 256-value piece a 32-work-item call like the others (Q5_1 and Q8_0: eight 32-value blocks).
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (uint8_t) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (uint8_t) ((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}
template<typename Out>
void dq_q4_k(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_q4_K* x = (const block_q4_K*) vx + ibs;
    const int il = tid / 8, ir = tid % 8, is = 2 * il;
    const float dall = (float) x->dm[0], dmin = (float) x->dm[1];
    const uint8_t* q = x->qs + (size_t) (32 * il + 4 * ir);
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x->scales, sc, m);
    const float d1 = dall * (float) sc, m1 = dmin * (float) m;
    get_scale_min_k4(is + 1, x->scales, sc, m);
    const float d2 = dall * (float) sc, m2 = dmin * (float) m;
    float lo[4], hi[4];
    for (int l = 0; l < 4; ++l) {
        lo[l] = d1 * (float) (q[l] & 0xF) - m1;
        hi[l] = d2 * (float) (q[l] >> 4) - m2;
    }
    o.template put<4>(64 * il + 4 * ir, lo);
    o.template put<4>(64 * il + 4 * ir + 32, hi);
}
template<typename Out>
void dq_q5_k(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_q5_K* x = (const block_q5_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        const float dall = (float) x->dm[0], dmin = (float) x->dm[1];
        const uint8_t* ql = x->qs + (size_t) (32 * il + 2 * ir);
        const uint8_t* qh = x->qh + (size_t) (2 * ir);
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x->scales, sc, m);
        const float d1 = dall * (float) sc, m1 = dmin * (float) m;
        get_scale_min_k4(is + 1, x->scales, sc, m);
        const float d2 = dall * (float) sc, m2 = dmin * (float) m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        float lo[2], hi[2];
        lo[0] = d1 * (float) ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1;
        lo[1] = d1 * (float) ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1;
        hm = (uint8_t) (hm << 1);
        hi[0] = d2 * (float) ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2;
        hi[1] = d2 * (float) ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2;
        o.template put<2>(64 * il + 2 * ir, lo);
        o.template put<2>(64 * il + 2 * ir + 32, hi);
    }
}
template<typename Out>
void dq_q5_1(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_q5_1* x = (const block_q5_1*) vx + ibs * (QK_K / QK5_1);
    const int ib = tid % 8, il = tid / 8;
    const float d = (float) x[ib].dm[0], m = (float) x[ib].dm[1];
    uint32_t qh;
    std::memcpy(&qh, x[ib].qh, sizeof(qh));
    float lo[4], hi[4];
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q5_1 for value pairs iqs, iqs + 16
        const int xh_0 = (int) ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = (int) (qh >> (iqs + 12)) & 0x10;
        lo[j] = (float) ((x[ib].qs[iqs] & 0xf) | xh_0) * d + m;
        hi[j] = (float) ((x[ib].qs[iqs] >> 4) | xh_1) * d + m;
    }
    o.template put<4>(32 * ib + 4 * il, lo);
    o.template put<4>(32 * ib + 4 * il + 16, hi);
}
template<typename Out>
void dq_q5_0(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_q5_0* x = (const block_q5_0*) vx + ibs * (QK_K / QK5_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = (float) x[ib].d;
    uint32_t qh;
    std::memcpy(&qh, x[ib].qh, sizeof(qh));
    float lo[4], hi[4];
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q5_0 for value pairs iqs, iqs + 16
        const int xh_0 = (int) ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = (int) (qh >> (iqs + 12)) & 0x10;
        lo[j] = (float) (((x[ib].qs[iqs] & 0xf) | xh_0) - 16) * d;
        hi[j] = (float) (((x[ib].qs[iqs] >> 4) | xh_1) - 16) * d;
    }
    o.template put<4>(32 * ib + 4 * il, lo);
    o.template put<4>(32 * ib + 4 * il + 16, hi);
}
template<typename Out>
void dq_q4_0(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_q4_0* x = (const block_q4_0*) vx + ibs * (QK_K / QK4_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = (float) x[ib].d;
    float lo[4], hi[4];
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q4_0 for value pairs iqs, iqs + 16
        lo[j] = ((float) (x[ib].qs[iqs] & 0xf) - 8.0f) * d;
        hi[j] = ((float) (x[ib].qs[iqs] >> 4) - 8.0f) * d;
    }
    o.template put<4>(32 * ib + 4 * il, lo);
    o.template put<4>(32 * ib + 4 * il + 16, hi);
}
template<typename Out>
void dq_q4_1(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_q4_1* x = (const block_q4_1*) vx + ibs * (QK_K / QK4_1);
    const int ib = tid % 8, il = tid / 8;
    const float d = (float) x[ib].dm[0], m = (float) x[ib].dm[1];
    float lo[4], hi[4];
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q4_1 for value pairs iqs, iqs + 16
        lo[j] = (float) (x[ib].qs[iqs] & 0xf) * d + m;
        hi[j] = (float) (x[ib].qs[iqs] >> 4) * d + m;
    }
    o.template put<4>(32 * ib + 4 * il, lo);
    o.template put<4>(32 * ib + 4 * il + 16, hi);
}
// llama.cpp's dequantize_q6_K (dequantize.cuh, written for 64 threads) folded onto 32 work-items: each of the 64 roles
// runs at tid and tid + 32, its four values 32 apart
template<typename Out>
void dq_q6_k(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_q6_K* x = (const block_q6_K*) vx + ibs;
    const float d = (float) x->d;
    for (int tt = tid; tt < 64; tt += 32) {
        const int ip = tt / 32, il = tt - 32 * ip, is = 8 * ip + il / 16;
        const uint8_t* ql = x->ql + (ptrdiff_t) 64 * ip + il;
        const uint8_t qh = x->qh[32 * ip + il];
        const int8_t* sc = x->scales + is;
        const float v[4] = {d * (float) sc[0] * (float) ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32),
                            d * (float) sc[2] * (float) ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32),
                            d * (float) sc[4] * (float) ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32),
                            d * (float) sc[6] * (float) ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32)};
        for (int k = 0; k < 4; ++k) o.template put<1>(128 * ip + il + 32 * k, v + k);
    }
}
template<typename Out>
void dq_q8_0(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = (float) x[ib].d;
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = (float) x[ib].qs[8 * il + j] * d;
    o.template put<8>(32 * ib + 8 * il, v);
}

// BF16 (the token embedding as the checkpoint ships it, tools/embd_bf16_pack.py; upstream c627d51): 8 values a
// work-item, widened exactly
template<typename Out>
inline void dq_bf16(const void* vx, int64_t ibs, const Out& o, int tid) {
    const uint16_t* x = (const uint16_t*) vx + ibs * 256 + (int64_t) tid * 8;
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = sycl::bit_cast<float>((uint32_t) x[j] << 16);
    o.template put<8>(tid * 8, v);
}

// the trellis types: one 256-value call is two 128-value blocks; work-item tid decodes block tid / 16, lane
// (tid / 2) % 8, steps 2 (tid % 2) .. 2 (tid % 2) + 1 (8 values; TQP: steps 8l + 4 (tid % 2) .. + 3), through the fp16
// codebook (ggml's to_float)
template<int TY, typename Out>
inline void dq_tq(const void* vx, int64_t ibs, const Out& o, int tid) {
    const int blk = tid >> 4, l = (tid >> 1) & 7, half = tid & 1;
    const auto* b = (const typename TqInfo<TY>::block_t*) vx + 2 * ibs + blk;
    const float d = tq_h2f(b->d);
    const uint32_t* lut = (const uint32_t*) tq_lut_f16;
    float v[8];
    if constexpr (TqInfo<TY>::NS == 64) {
        uint32_t s[4];
        tq_states4<TY>(b->qs, 8 * l + 4 * half, s);
        for (int i = 0; i < 4; ++i) {
            const uint32_t pa = lut[((s[i] & TQ_STATE_MASK) * 0x9e3779b1u) >> 21];
            v[2 * i + 0] = d * tq_h2f(pa & 0xFFFFu);
            v[2 * i + 1] = d * tq_h2f(pa >> 16);
        }
        o.template put<8>(blk * 128 + 16 * l + 8 * half, v);
        return;
    }
    uint32_t s[4];
    tq_states4<TY>(b->qs, 4 * l, s);
    for (int i = 0; i < 2; ++i) {
        const uint32_t x = (s[2 * half + i] & TQ_STATE_MASK) * 0x9e3779b1u;
        const uint32_t pa = lut[x >> 21], pb = lut[(x >> 10) & 2047u];
        v[4 * i + 0] = d * tq_h2f(pa & 0xFFFFu);
        v[4 * i + 1] = d * tq_h2f(pa >> 16);
        v[4 * i + 2] = d * tq_h2f(pb & 0xFFFFu);
        v[4 * i + 3] = d * tq_h2f(pb >> 16);
    }
    o.template put<8>(blk * 128 + 16 * l + 8 * half, v);
}

template<typename Out>
inline void dq_dispatch(int ty, const void* vx, int64_t ibs, const Out& o, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, o, tid); break;
        case 17: dq_iq2_xs(vx, ibs, o, tid); break;
        case 18: dq_iq3_xxs(vx, ibs, o, tid); break;
        case 20: dq_iq4_nl(vx, ibs, o, tid); break;
        case 21: dq_iq3_s(vx, ibs, o, tid); break;
        case 22: dq_iq2_s(vx, ibs, o, tid); break;
        case 29: dq_iq1_m(vx, ibs, o, tid); break;
        case 19: dq_iq1_s(vx, ibs, o, tid); break;
        case 23: dq_iq4_xs(vx, ibs, o, tid); break;
        case 11: dq_q3_k(vx, ibs, o, tid); break;
        case 10: dq_q2_k(vx, ibs, o, tid); break;
        case 42: dq_q2_0(vx, ibs, o, tid); break;
        case 30: dq_bf16(vx, ibs, o, tid); break;   // the token embedding only (embed_type_supported)
        case 12: dq_q4_k(vx, ibs, o, tid); break;
        case 13: dq_q5_k(vx, ibs, o, tid); break;
        case 7: dq_q5_1(vx, ibs, o, tid); break;
        case 8: dq_q8_0(vx, ibs, o, tid); break;
        case 6: dq_q5_0(vx, ibs, o, tid); break;
        case 2: dq_q4_0(vx, ibs, o, tid); break;
        case 3: dq_q4_1(vx, ibs, o, tid); break;
        case 14: dq_q6_k(vx, ibs, o, tid); break;
        case 144: dq_tq<144>(vx, ibs, o, tid); break;
        case 145: dq_tq<145>(vx, ibs, o, tid); break;
        case 146: dq_tq<146>(vx, ibs, o, tid); break;
        case 147: dq_tq<147>(vx, ibs, o, tid); break;
        case 148: dq_tq<148>(vx, ibs, o, tid); break;
        case 149: dq_tq<149>(vx, ibs, o, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template<typename dst_t>
sycl::event launch_dequant_flat(sycl::queue& q, int ty, const void* vx, int64_t n, dst_t* y) {
    // eight superblocks a work-group (one each was 32 work-items, too few to keep the GPU busy)
    const int64_t nsb = n / QK_K;
    return q.parallel_for(sycl::nd_range<1>((size_t) (nsb + 7) / 8 * 256, 256), [=](sycl::nd_item<1> it) {
        const int64_t i = (int64_t) (it.get_global_linear_id() / 32);
        if (i >= nsb) return;
        dq_dispatch(ty, vx, i, ToGlobal<dst_t>{y + i * QK_K}, (int) (it.get_local_id(0) % 32));
    });
}

// ---------------------------------------------------------------- iq_gemm_grouped_f16
// The grouped product of xmx_gemm_grouped with the weights still in GGUF blocks.  A work-group takes WM rows of one
// expert and WN of its weight rows; for each KC values of K its work-items dequantize the WN x KC weights into local
// memory (FP16, the dequantizers above: the values iq_dequant_f16 writes), and its sub-groups multiply them, each a
// 32 x 64 tile of Y as 4 x 4 XMX accumulators, A read from X in global memory, B row-major from local memory.
// The dequantization and the products do not overlap (bench/results/2026-10-02-iq-gemm).
namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace ix = sycl::ext::intel::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;

// [[maybe_unused]]: the device compiles for other GPUs leave out the code that uses GT_K (device_target.hpp)
[[maybe_unused]] constexpr int GT_M = 8, GT_N = 16, GT_K = 16, G_SGM = 4, G_SGN = 4;
constexpr int G_MS = GT_M * G_SGM, G_NS = GT_N * G_SGN;

// The 8 values a dequantizer hands one work-item, kept in registers: one run of 8, or two runs of 4 that are 16 apart
// (the IQ4 formats), with their offsets in the 256-value piece.
struct Collect {
    float* v;   // [8]
    int* off;   // [2]: where v[0..3] and v[4..7] go
    template<int N> void put(int o, const float* x) const {
        if constexpr (N == 8) {
            for (int j = 0; j < 8; ++j) v[j] = x[j];
            off[0] = o;
            off[1] = o + 4;
        } else {
            const int h = (o / 16) & 1;
            for (int j = 0; j < 4; ++j) v[4 * h + j] = x[j];
            off[h] = o;
        }
    }
};

template<int TYPE, typename Out>
inline void dq_one(const void* vx, int64_t ibs, const Out& o, int tid) {
    if constexpr (TYPE == 16) dq_iq2_xxs(vx, ibs, o, tid);
    else if constexpr (TYPE == 17) dq_iq2_xs(vx, ibs, o, tid);
    else if constexpr (TYPE == 18) dq_iq3_xxs(vx, ibs, o, tid);
    else if constexpr (TYPE == 20) dq_iq4_nl(vx, ibs, o, tid);
    else if constexpr (TYPE == 21) dq_iq3_s(vx, ibs, o, tid);
    else if constexpr (TYPE == 22) dq_iq2_s(vx, ibs, o, tid);
    else if constexpr (TYPE == 23) dq_iq4_xs(vx, ibs, o, tid);
    else if constexpr (TYPE == 29) dq_iq1_m(vx, ibs, o, tid);
    else if constexpr (TYPE == 19) dq_iq1_s(vx, ibs, o, tid);
    else if constexpr (TYPE == 42) dq_q2_0(vx, ibs, o, tid);
    else if constexpr (is_tq(TYPE)) dq_tq<TYPE>(vx, ibs, o, tid);
    else static_assert(TYPE == 16, "iq_gemm: type");
}

struct IqGemmPtrs { const uint8_t* p[kIqGemmGroup]; };

// 256 x 128 work-group tiles, K 128 at a time: one tile of rows takes an expert's 80-240 rows, so its weights are
// dequantized once (128 x 256 tiles dequantized most experts' twice and were 10% slower)
template<int TYPE>
struct IqGemmKernel {
    static constexpr int WM = kIqGemmTileRows, WN = 128, KC = 128;
    static constexpr int LDN = WN + 8;          // the tile's row stride in values (+8: rows start in other banks)
    static constexpr int NSG_N = WN / G_NS, NSG = (WM / G_MS) * NSG_N, WG = NSG * 16;
    const sycl::half* X;
    IqGemmPtrs W;
    float* Y;
    const int32_t* bounds;
    size_t row_bytes;
    int64_t inter, ldy, tiles_m, nN, N, K;
    sycl::local_accessor<sycl::half, 1> wt;     // [KC][LDN]: K by weight rows (B row-major)

    // Block k0 of K into the tile: a work-item dequantizes the same 8 values of 4 neighbouring weight rows and writes
    // them as 8 bytes a row of the tile, neighbouring work-items the next 4 rows.  B column-major took vector stores
    // of one weight row but loaded for the products at half the speed; row-major with 2-byte stores, one weight row
    // a work-item, dequantized at half the speed.
    void dequant(int lid, const uint8_t* w, int64_t n0, int64_t k0) const {
        const int piece = (int) (k0 % 256);
        for (int t = lid; t < (WN / 4) * (KC / 8); t += WG) {
            const int r4 = t % (WN / 4) * 4, tt = t / (WN / 4);
            float v[4][8] = {};
            int off[2] = {};
            for (int q = 0; q < 4; ++q) {
                // with `inter` the weights' rows are gate rows then up rows and Y's columns alternate between them
                const int64_t n = n0 + r4 + q, row = inter ? (n & 1) * inter + (n >> 1) : n;
                dq_one<TYPE>(w + (size_t) row * row_bytes, k0 / 256, Collect{v[q], off}, piece / 8 + tt);
            }
            for (int j = 0; j < 8; ++j) {
                const int k = off[j / 4] + j % 4 - piece;
                *reinterpret_cast<sycl::vec<sycl::half, 4>*>(&wt[(size_t) k * LDN + (size_t) r4]) =
                    sycl::vec<sycl::half, 4>(sycl::half(v[0][j]), sycl::half(v[1][j]), sycl::half(v[2][j]),
                                             sycl::half(v[3][j]));
            }
        }
    }

#if !STRATA_DEVICE_NOT_INTEL
    // One block of K for a sub-group's tile with all its 32 rows (WHOLE) or not.
    template<bool WHOLE>
    void block(sycl::sub_group sg, const sycl::half* x, int64_t rows, int64_t m0, int nl, int64_t k0,
               mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, GT_M, GT_N> (&c)[G_SGM][G_SGN]) const {
        const auto gx = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(const_cast<sycl::half*>(x));
        // no prefetch of the next block of A (joint_matrix_prefetch): as xmx_gemm.cpp says
        for (int kk = 0; kk < KC; kk += GT_K) {
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, GT_M, GT_K, mx::layout::row_major> a[G_SGM];
            for (int i = 0; i < G_SGM; ++i) {
                if constexpr (WHOLE) mx::joint_matrix_load(sg, a[i], gx + (m0 + (int64_t) i * GT_M) * K + k0 + kk, K);
                else ix::joint_matrix_load_checked(sg, a[i], gx, K, rows, K, m0 + (int64_t) i * GT_M, k0 + kk);
            }
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, GT_K, GT_N, mx::layout::row_major> b[G_SGN];
            for (int j = 0; j < G_SGN; ++j) {
                const auto off = (std::ptrdiff_t) ((size_t) kk * LDN + (size_t) nl + (size_t) j * GT_N);
                mx::joint_matrix_load(sg, b[j], wt.template get_multi_ptr<sycl::access::decorated::no>() + off, LDN);
            }
            for (int i = 0; i < G_SGM; ++i)
                for (int j = 0; j < G_SGN; ++j) mx::joint_matrix_mad(sg, c[i][j], a[i], b[j], c[i][j]);
        }
    }

    // The K loop, one copy for a work-group tile of all rows and one for the rest, chosen for the whole work-group so
    // that every work-item meets the same barriers.  (Choosing per sub-group put barriers on different paths, which
    // let some work-items dequantize the next block while others still read the last; choosing the loads per tile
    // inside one loop ran several times slower.)
    template<bool WHOLE>
    void loop(sycl::nd_item<1> it, const sycl::half* x, int64_t rows, int64_t m0, int nl, const uint8_t* w,
              int64_t wn0,
              mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, GT_M, GT_N> (&c)[G_SGM][G_SGN]) const {
        const auto sg = it.get_sub_group();
        const int lid = (int) it.get_local_linear_id();
        for (int64_t k0 = 0; k0 < K; k0 += KC) {
            dequant(lid, w, wn0, k0);
            sycl::group_barrier(it.get_group());
            block<WHOLE>(sg, x, rows, m0, nl, k0, c);
            sycl::group_barrier(it.get_group());
        }
    }
#endif

    void operator()(sycl::nd_item<1> it) const {
#if STRATA_DEVICE_NOT_INTEL
        (void) it;
#else
        const auto sg = it.get_sub_group();
        const int64_t g = (int64_t) it.get_group(0);
        const int64_t ex = g / (tiles_m * nN), r = g % (tiles_m * nN), tm = r / nN, tn = r % nN;
        const int64_t row0 = bounds[ex], rows = bounds[ex + 1] - row0;
        if (tm * WM >= rows) return;   // the whole work-group: the expert has fewer rows than the launch allows for
        const int sgid = (int) sg.get_group_linear_id();
        const int64_t m0 = tm * WM + (int64_t) (sgid / NSG_N) * G_MS, wn0 = tn * WN;
        const int nl = (sgid % NSG_N) * G_NS;   // this sub-group's first column in the tile
        // filled by index: a range-for over the 2-D array left all but the first column unfilled (they kept what an
        // earlier work-group on the same hardware thread had left)
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, GT_M, GT_N> c[G_SGM][G_SGN];
        for (int i = 0; i < G_SGM; ++i)
            for (int j = 0; j < G_SGN; ++j) mx::joint_matrix_fill(sg, c[i][j], 0.0f);
        const sycl::half* x = X + row0 * K;
        if (tm * WM + WM <= rows) loop<true>(it, x, rows, m0, nl, W.p[ex], wn0, c);
        else loop<false>(it, x, rows, m0, nl, W.p[ex], wn0, c);
        if (m0 >= rows) return;
        const bool whole = m0 + G_MS <= rows;
        auto gy = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
            Y + row0 * ldy);
        for (int i = 0; i < G_SGM; ++i)
            for (int j = 0; j < G_SGN; ++j) {
                const int64_t m = m0 + (int64_t) i * GT_M, n = wn0 + nl + (int64_t) j * GT_N;
                if (whole) mx::joint_matrix_store(sg, c[i][j], gy + m * ldy + n, ldy, mx::layout::row_major);
                else ix::joint_matrix_store_checked(sg, c[i][j], gy, ldy, mx::layout::row_major, rows, N, m, n);
            }
#endif
    }
    auto get(syclex::properties_tag) const {
        return syclex::properties{sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<16>};
    }
};

template<int TYPE>
sycl::event launch_iq_gemm(sycl::queue& q, const sycl::half* X, const IqGemmPtrs& W, int64_t inter, float* Y,
                           int64_t ldy, const int32_t* bounds, int G, int64_t max_rows, int64_t N, int64_t K) {
    using Kn = IqGemmKernel<TYPE>;
    const int64_t tiles_m = (max_rows + Kn::WM - 1) / Kn::WM, nN = N / Kn::WN;
    const size_t row_bytes = iq_row_bytes(TYPE, K);
    return q.submit([&](sycl::handler& h) {
        Kn k{X, W, Y, bounds, row_bytes, inter, ldy, tiles_m, nN, N, K,
             sycl::local_accessor<sycl::half, 1>(sycl::range<1>((size_t) Kn::KC * Kn::LDN), h)};
        h.parallel_for(sycl::nd_range<1>((size_t) (G * tiles_m * nN) * Kn::WG, Kn::WG), k);
    });
}

// the formats iq_gemm_grouped_f16 dequantizes into its tile (dq_one); the K-quants and Q5_1 / Q8_0 take the fp16
// dequant and xmx_gemm_grouped instead
bool in_iq_gemm(int t) {
    return t == 16 || t == 17 || t == 18 || t == 19 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 ||
           is_tq(t);
}
bool is_iq(int t) {
    return in_iq_gemm(t) || t == 10 || t == 11 || t == 12 || t == 13 || t == 14 || t == 6 || t == 7 || t == 8 || t == 2 ||
           t == 3;
}

void require(bool ok, const char* what) {
    if (!ok) throw core::DeviceError(std::string(what) + ": bad arguments");
}

}  // namespace

bool iq_supported(int t) noexcept { return is_iq(t); }
bool embed_type_supported(int t) noexcept { return is_iq(t) || t == 30; }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 19: return (size_t) (n / 256) * sizeof(block_iq1_s);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 10: return (size_t) (n / 256) * sizeof(block_q2_K);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 2: return (size_t) (n / 32) * sizeof(block_q4_0);
        case 3: return (size_t) (n / 32) * sizeof(block_q4_1);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 30: return (size_t) n * 2;   // BF16: the token embedding (embed_rows, iq_dequant_f32), BF16 experts
        case 144: return (size_t) (n / 128) * sizeof(block_tq2_t);
        case 145: return (size_t) (n / 128) * sizeof(block_tqk6);
        case 146: return (size_t) (n / 128) * sizeof(block_tqk7);
        case 147: return (size_t) (n / 128) * sizeof(block_tqp5);
        case 148: return (size_t) (n / 128) * sizeof(block_tqp6);
        case 149: return (size_t) (n / 128) * sizeof(block_tqp7);
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    require(n_cols % 32 == 0, "quantize_q8_1_rows");
    // the experts' activations with canon_math.hpp's scale, one scale for each 256 values (stored in each of their
    // eight q8_1 blocks) where the rows are whole 256s: the i-quants' calls of a 256-value block then share it, and a
    // row's products add their integer sums before the scale (seg_dot_canon; the CPU's canon_expert.cpp quantizes
    // its rows the same way).  A row of another width: a scale for each 32.  The sum of each 32 (the K-quants' min
    // term) as before.
    if (n_cols % 256 == 0) {   // 8 values a lane, the 256 a sub-group's
        sync_if_needed(stream, q8act::quantize<8, q8act::Canon<256>>(queue_for(stream), n / 32, 0, q8act::Flat{x},
                                                                     q8act::Q8_1{(uint8_t*) y}),
                       "quantize_q8_1_rows");
        return;
    }
    sync_if_needed(stream, launch_quantize_q8_1_canon(queue_for(stream), x, (block_q8_1*) y, n), "quantize_q8_1_rows");
}

namespace {
// APR (prism.hadamard): y = H_block (signs * x) per block of each row, a work-group a block.  Each butterfly is the
// CPU's (hadamard.cpp: a + c, a - c in the same order, then the scale), so the rotated values are the same floats; a
// work-group smaller than the block's pairs (the device's limit) takes several pairs a work-item.
template<typename T>
sycl::event launch_hadamard_rows(sycl::queue& q, const T* x, T* y, int64_t rows, int64_t width, int block,
                                 const float* signs, const char* what) {
    if (block < 2 || (block & (block - 1)) != 0 || width % block != 0)
        throw core::DeviceError(std::string(what) + ": Hadamard block " + std::to_string(block) +
                                " is not a power of two dividing the row width " + std::to_string(width) +
                                " (hadamard.txt's block and the expert widths disagree)");
    const size_t max_wg = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    const size_t wg = std::min<size_t>((size_t) block / 2, max_wg);
    const size_t nb = (size_t) (width / block);
    const float scale = 1.0f / std::sqrt((float) block);
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>((size_t) block), h);
        // the rows on the fast dimension: a prompt chunk's T * K rows pass the 65535 a CUDA or HIP grid allows in y
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(nb, (size_t) rows * wg), sycl::range<2>(1, wg)),
                       [=](sycl::nd_item<2> it) {
            const int lid = (int) it.get_local_id(1), n = (int) wg;
            const int64_t col0 = (int64_t) it.get_group(0) * block;
            const int64_t base = (int64_t) it.get_group(1) * width + col0;
            for (int i = lid; i < block; i += n) {
                float v = (float) x[base + i];
                if (signs) v *= signs[col0 + i];
                buf[i] = v;
            }
            for (int len = 1; len < block; len <<= 1) {
                sycl::group_barrier(it.get_group());
                for (int p = lid; p < block / 2; p += n) {
                    const int i = (p / len) * 2 * len + p % len;
                    const float a = buf[i], c = buf[i + len];
                    buf[i] = a + c;
                    buf[i + len] = a - c;
                }
            }
            sycl::group_barrier(it.get_group());
            for (int i = lid; i < block; i += n) y[base + i] = (T) (buf[i] * scale);
        });
    });
}
}  // namespace

void hadamard_rows(const float* x, float* y, int64_t rows, int64_t width, int block, const float* signs, void* stream) {
    if (rows <= 0) return;
    sync_if_needed(stream, launch_hadamard_rows<float>(queue_for(stream), x, y, rows, width, block, signs, "hadamard_rows"),
                   "hadamard_rows");
}

void hadamard_rows_f16(uint16_t* x, int64_t rows, int64_t width, int block, const float* signs, void* stream) {
    if (rows <= 0) return;
    auto* h = reinterpret_cast<sycl::half*>(x);
    sync_if_needed(stream, launch_hadamard_rows<sycl::half>(queue_for(stream), h, h, rows, width, block, signs,
                                                            "hadamard_rows_f16"),
                   "hadamard_rows_f16");
}

const float* hadamard_device_signs(const float* host, int width) {
    if (host == nullptr || width <= 0) return nullptr;
    static std::mutex mu;
    static std::map<std::tuple<int, const float*, int>, float*> table;   // (engine device, host vector, width)
    auto& rt = core::Runtime::get();
    std::lock_guard<std::mutex> lock(mu);
    float*& d = table[std::make_tuple(core::current_device(), host, width)];
    if (d == nullptr) {
        d = sycl::malloc_device<float>((size_t) width, rt.device(), rt.context());
        if (d == nullptr) throw core::DeviceError("hadamard_device_signs: device allocation failed");
        auto& q = rt.compute();
        q.memcpy(d, host, (size_t) width * sizeof(float));
        rt.finish(q);
    }
    return d;
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    auto& q = queue_for(stream);
    const size_t rb = iq_row_bytes(t, n_in);
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    sycl::event e;
    switch (t) {
        case 16: e = launch_mmvq<16>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 17: e = launch_mmvq<17>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 18: e = launch_mmvq<18>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 20: e = launch_mmvq<20>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 21: e = launch_mmvq<21>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 22: e = launch_mmvq<22>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 23: e = launch_mmvq<23>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 29: e = launch_mmvq<29>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 19: e = launch_mmvq<19>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 42: e = launch_mmvq<42>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 12: e = launch_mmvq<12>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 13: e = launch_mmvq<13>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 7: e = launch_mmvq<7>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 8: e = launch_mmvq<8>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 6: e = launch_mmvq<6>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 2: e = launch_mmvq<2>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 3: e = launch_mmvq<3>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 144: e = launch_mmvq<144>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 145: e = launch_mmvq<145>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 146: e = launch_mmvq<146>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 147: e = launch_mmvq<147>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 148: e = launch_mmvq<148>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 149: e = launch_mmvq<149>(q, W, rb, X, y, n_in, n_out, ncols); break;
        default: throw core::DeviceError("iq_mmvq: type " + std::to_string(t) + " is not supported");
    }
    sync_if_needed(stream, e, "iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    require(n % 256 == 0 && is_iq(t), "iq_dequant_f16");
    sync_if_needed(stream, launch_dequant_flat<sycl::half>(queue_for(stream), t, src, n, reinterpret_cast<sycl::half*>(dst)),
                   "iq_dequant_f16");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    require(n % 256 == 0 && embed_type_supported(t), "iq_dequant_f32");
    sync_if_needed(stream, launch_dequant_flat<float>(queue_for(stream), t, src, n, dst), "iq_dequant_f32");
}

void embed_rows(const EmbedTable& tb, const int32_t* ids, int64_t token, int64_t n_tok, float* out, void* stream) {
    if (n_tok <= 0) return;
    const int64_t n = tb.n_embd;
    const auto* tab = (const uint8_t*) tb.rows;
    const size_t rb = tb.row_bytes;
    sycl::event e;
    if (tb.type == EmbedTable::kPacked) {
        const int bits = tb.code_bits, bias = tb.code_bias, ge = tb.group_elems;
        require((bits == 2 || bits == 4 || bits == 8) && ge > 0 && n % ge == 0 && n % (8 / bits) == 0, "embed_rows");
        const float *scales = tb.scales, *offsets = tb.offsets;
        const uint64_t row_groups = (uint64_t) (n / ge);
        e = queue_for(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n), [=](sycl::id<2> id) {
            const int64_t t = (int64_t) id[0], i = (int64_t) id[1];
            const uint64_t row = (uint64_t) (ids ? (int64_t) ids[t] : token);
            const uint8_t* c = tab + row * rb;
            const float* sc = scales + row * row_groups;
            const float* of = offsets ? offsets + row * row_groups : nullptr;
            const int per_byte = 8 / bits;
            const unsigned mask = (1u << bits) - 1u;
            const int code = (int) ((c[i / per_byte] >> ((i % per_byte) * bits)) & mask);
            const int64_t group = i / ge;
            // the product and the sum round separately: this library is built without contraction
            const float product = (float) (code + bias) * sc[group];
            out[(size_t) t * n + i] = product + (of ? of[group] : 0.0f);
        });
    } else {
        require(n % 256 == 0 && embed_type_supported(tb.type), "embed_rows");
        const int ty = tb.type;
        e = queue_for(stream).parallel_for(
            sycl::nd_range<2>({(size_t) n_tok, (size_t) (n / 256) * 32}, {1, 32}), [=](sycl::nd_item<2> it) {
                const int64_t t = (int64_t) it.get_group(0), b = (int64_t) it.get_group(1);
                const uint8_t* row = tab + (size_t) (ids ? (int64_t) ids[t] : token) * rb;
                dq_dispatch(ty, row, b, ToGlobal<float>{out + (size_t) t * n + b * QK_K}, (int) it.get_local_id(1));
            });
    }
    sync_if_needed(stream, e, "embed_rows");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    require(n_embd % 256 == 0 && is_iq(t), "iq_dequant_gu_f16");
    const int64_t per_row = n_embd / 256;
    auto* y = reinterpret_cast<sycl::half*>(dst);
    const int64_t nsb = n_ff * per_row;   // superblocks per matrix, eight a work-group
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<2>({2, (size_t) (nsb + 7) / 8 * 256}, {1, 256}), [=](sycl::nd_item<2> it) {
            const int64_t i = (int64_t) (it.get_global_id(1) / 32);
            if (i >= nsb) return;
            const int parity = (int) it.get_group(0);
            const int64_t r = i / per_row, c = i % per_row;
            dq_dispatch(t, parity ? up : gate, i, ToGlobal<sycl::half>{y + ((2 * r + parity) * per_row + c) * QK_K},
                                    (int) (it.get_local_id(1) % 32));
        });
    sync_if_needed(stream, e, "iq_dequant_gu_f16");
}

void iq_dequant_expert_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* gu_dst,
                           int dt, const void* down, uint16_t* d_dst, void* stream) {
    require(n_embd % 256 == 0 && is_iq(t) && is_iq(dt), "iq_dequant_expert_f16");
    const int64_t per_row = n_embd / 256;
    auto* y = reinterpret_cast<sycl::half*>(gu_dst);
    auto* yd = reinterpret_cast<sycl::half*>(d_dst);
    const int64_t nsb = n_ff * per_row;   // superblocks per matrix (the down matrix has as many), eight a work-group
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<2>({3, (size_t) (nsb + 7) / 8 * 256}, {1, 256}), [=](sycl::nd_item<2> it) {
            const int64_t i = (int64_t) (it.get_global_id(1) / 32);
            if (i >= nsb) return;
            const int part = (int) it.get_group(0);   // 0 gate, 1 up, 2 down
            const int lane = (int) (it.get_local_id(1) % 32);
            if (part == 2) {
                dq_dispatch(dt, down, i, ToGlobal<sycl::half>{yd + i * QK_K}, lane);
                return;
            }
            const int64_t r = i / per_row, c = i % per_row;
            dq_dispatch(t, part ? up : gate, i, ToGlobal<sycl::half>{y + ((2 * r + part) * per_row + c) * QK_K}, lane);
        });
    sync_if_needed(stream, e, "iq_dequant_expert_f16");
}

bool iq_gemm_grouped_ok(int t, int64_t N, int64_t K) noexcept {
    return in_iq_gemm(t) && N > 0 && N % 128 == 0 && K > 0 && K % 128 == 0;
}

void iq_gemm_grouped_f16(int t, const uint16_t* X, const void* const* W, int64_t interleave, float* Y, int64_t ldy,
                         const int32_t* bounds, int G, int64_t max_rows, int64_t N, int64_t K, void* stream) {
    if (G <= 0 || max_rows <= 0) return;
    require(iq_gemm_grouped_ok(t, N, K) && ldy >= N && G <= kIqGemmGroup && (interleave == 0 || 2 * interleave == N),
            "iq_gemm_grouped_f16");
    auto& q = queue_for(stream);
    const auto* x = reinterpret_cast<const sycl::half*>(X);
    IqGemmPtrs w{};
    for (int e = 0; e < G; ++e) w.p[e] = static_cast<const uint8_t*>(W[e]);
    sycl::event e;
    switch (t) {
        case 16: e = launch_iq_gemm<16>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 17: e = launch_iq_gemm<17>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 18: e = launch_iq_gemm<18>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 20: e = launch_iq_gemm<20>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 21: e = launch_iq_gemm<21>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 22: e = launch_iq_gemm<22>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 23: e = launch_iq_gemm<23>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 29: e = launch_iq_gemm<29>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 19: e = launch_iq_gemm<19>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 144: e = launch_iq_gemm<144>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 145: e = launch_iq_gemm<145>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 146: e = launch_iq_gemm<146>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 147: e = launch_iq_gemm<147>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 148: e = launch_iq_gemm<148>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        case 149: e = launch_iq_gemm<149>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
        default: e = launch_iq_gemm<42>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
    }
    sync_if_needed(stream, e, "iq_gemm_grouped_f16");
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, int up_type) noexcept {
    if (up_type >= 0 && up_type != gu_type)   // a mixed layer: each of gate and up as a gate/up type of its own
        return up_type != 30 && gu_type != 30 && native_expert_supported(gu_type, d_type, n_embd, n_ff) &&
               native_expert_supported(up_type, d_type, n_embd, n_ff);
    // the cases of native_expert_grouped's two switches, and the prompt path's dequantizer (iq_dequant_gu_f16 and the
    // down rows) for both
    if (gu_type == 30 || d_type == 30)   // BF16 (native_expert_grouped_bf16): both sides, rows of whole value pairs
        return gu_type == 30 && d_type == 30 && n_embd > 0 && n_ff > 0 && n_embd % 32 == 0 && n_ff % 32 == 0;
    const bool gu = in_iq_gemm(gu_type) ? gu_type != 20
                                        : gu_type == 12 || gu_type == 13 || gu_type == 14 || gu_type == 8 ||
                                          gu_type == 6 || gu_type == 2 || gu_type == 3 || gu_type == 10 || gu_type == 11;
    const bool d = d_type == 20 || d_type == 23 || d_type == 42 || d_type == 6 || d_type == 7 || d_type == 8 ||
                   d_type == 2 || d_type == 3 || d_type == 14 || d_type == 10 || d_type == 11 || d_type == 12 ||
                   d_type == 13 || d_type == 16 || d_type == 17 || d_type == 18 || d_type == 21 || d_type == 22 ||
                   is_tq(d_type);
    const auto qk = [](int t) {
        return t == 20 || t == 6 || t == 7 || t == 8 || t == 2 || t == 3 ? 32 : t == 42 ? 64 : is_tq(t) ? 128 : 256;
    };
    return gu && d && n_embd > 0 && n_ff > 0 && n_embd % qk(gu_type) == 0 && n_ff % qk(d_type) == 0 &&
           n_embd % 256 == 0;   // the dequantizers write whole 256-value pieces of a gate/up row
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, int up_type) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.up_type = up_type >= 0 ? up_type : gu_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.up_row = iq_row_bytes(L.up_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = L.up_off + (size_t) n_ff * L.up_row;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

void native_expert_gu_pair_tune(const NativeExpertLayout& L, int type, int part, void* stream) {
    if (!gu_pair_type(type) || gu_pair_mode() != -1) return;
    auto& q = queue_for(stream);
    const GuPairKey key = gu_pair_key(L, type, part);
    if (g_gu_pair_choices.get(q.get_device(), key)) return;
    std::vector<void*> allocated;
    auto alloc = [&](size_t bytes) {
        void* p = sycl::malloc_device(bytes, q);
        if (!p) throw core::DeviceError("IQ gate/up timing allocation failed");
        allocated.push_back(p);
        return p;
    };
    auto release = [&] { q.wait_and_throw(); for (void* p : allocated) sycl::free(p, q); };
    try {
        auto* blob = static_cast<uint8_t*>(alloc(L.bytes));
        auto* ptr = static_cast<unsigned long long*>(alloc(4 * sizeof(unsigned long long)));
        auto* starts = static_cast<int32_t*>(alloc(5 * sizeof(int32_t)));
        auto* count = static_cast<int32_t*>(alloc(sizeof(int32_t)));
        auto* toks = static_cast<int32_t*>(alloc(16 * sizeof(int32_t)));
        auto* x = static_cast<block_q8_1*>(alloc((size_t) 4 * (L.n_embd / 32) * sizeof(block_q8_1)));
        auto* gate = static_cast<float*>(alloc((size_t) 16 * L.n_ff * sizeof(float)));
        auto* up = static_cast<float*>(alloc((size_t) 16 * L.n_ff * sizeof(float)));
        const unsigned long long hp[4] = {(unsigned long long) (uintptr_t) blob, (unsigned long long) (uintptr_t) blob,
                                          (unsigned long long) (uintptr_t) blob, (unsigned long long) (uintptr_t) blob};
        const int32_t hc = 4, ht[16] = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
        q.memset(blob, 0, L.bytes);
        q.memset(x, 0, (size_t) 4 * (L.n_embd / 32) * sizeof(block_q8_1));
        q.memcpy(ptr, hp, sizeof hp); q.memcpy(count, &hc, sizeof hc); q.memcpy(toks, ht, sizeof ht);
        auto run = [&](bool pair) {
            switch (type) {
            case 18: if (pair) launch_native_gu_pair<18>(q, 4, ptr, starts, count, toks, x, L, gate, up, part);
                     else launch_native_gu<18, GU_LPR>(q, 4, ptr, starts, count, toks, x, L, gate, up, part); break;
            case 21: if (pair) launch_native_gu_pair<21>(q, 4, ptr, starts, count, toks, x, L, gate, up, part);
                     else launch_native_gu<21, GU_LPR>(q, 4, ptr, starts, count, toks, x, L, gate, up, part); break;
            default: if (pair) launch_native_gu_pair<22>(q, 4, ptr, starts, count, toks, x, L, gate, up, part);
                     else launch_native_gu<22, GU_LPR>(q, 4, ptr, starts, count, toks, x, L, gate, up, part); break;
            }
            return true;
        };
        const bool first = g_gu_pair_warmed.put_first(q.get_device(), 0, true, [](int) { return true; }).second;
        bool faster = true;
        for (int entries : {1, 4}) {
            const int32_t hs[5] = {0, entries, 2 * entries, 3 * entries, 4 * entries};
            q.memcpy(starts, hs, sizeof hs).wait_and_throw();
            const auto m = core::timed_choice({{[&] { for (int i = 0; i < 4; ++i) run(false); return true; }},
                                               {[&] { for (int i = 0; i < 4; ++i) run(true); return true; }}},
                                              [&] { q.wait_and_throw(); }, 5, 1.10, first && entries == 1 ? 0.5 : 0.0);
            faster = faster && m.pick == 1;
            std::fprintf(stderr, "xestrata: IQ gate/up %d entries, type %d: %.1f / %.1f us\n", entries, type,
                         m.seconds[0] * 1e6 / 4, m.seconds[1] * 1e6 / 4);
        }
        g_gu_pair_choices.put(q.get_device(), key, faster);
        release();
    } catch (...) {
        for (void* p : allocated) sycl::free(p, q);
        throw;
    }
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    auto& q = queue_for(stream);   // in order: the launches below follow one another
    const float lim = expert_swiglu_limit();
    // upstream 0bc1594: the gate/up and down grids take grid_groups rows of groups (each striding over the groups) and
    // SwiGLU + q8_1 run as one pass over the call's own entries
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = reinterpret_cast<float*>((uint8_t*) scratch + fa);
    float* h = reinterpret_cast<float*>((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = reinterpret_cast<block_q8_1*>((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const size_t cg = (size_t) (grid_groups <= 0 ? cap_groups : std::min(grid_groups, cap_groups));
    // upstream's expert mode 8 (launch_native_gu_fused): its gate/up for the grid formats and IQ4_XS, its down for
    // IQ4_NL and Q2_0, each where its local memory fits; never for a call whose grid strides over the groups (a verify
    // window's PCIe call), which upstream keeps on its default layout
    // APR: Hadamard-folded down weights take the rotated h, so h exists as floats first (the separate passes, not
    // the fused gate/up that writes q8_1 itself)
    const bool had = L.had_block > 0;
    const bool m8 = !(grid_groups > 0 && grid_groups < cap_groups) && !had && exp_mode8(q);
    const size_t xbytes = (size_t) kEntriesPass * (size_t) (L.n_embd / 32) * sizeof(block_q8_1);
    const size_t grid_bytes = L.gu_type == 22 ? sizeof(iq2s_grid) : L.gu_type == 16 ? sizeof(iq2xxs_grid)
                              : L.gu_type == 18 ? sizeof(iq3xxs_grid) : L.gu_type == 21 ? sizeof(iq3s_grid) : 0;
    const bool fused_gu = m8 && (L.up_type < 0 || L.up_type == L.gu_type) && (L.gu_type == 16 || L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 ||
                                 L.gu_type == 23) &&
                          grid_bytes + xbytes + (size_t) kEntriesPass * kM8Rows * sizeof(float) <= local_mem_bytes(q);
    const size_t hbytes = (size_t) kEntriesPass * (size_t) (L.n_ff / 32) * sizeof(block_q8_1);
    const bool lds_down = m8 && (L.d_type == 20 || L.d_type == 42) && hbytes <= local_mem_bytes(q);
    const bool ws_down = lds_down && down_ws_on(q) && hbytes + (size_t) kM8Rows * L.d_row + 32 <= local_mem_bytes(q);
    if (fused_gu) {
        const size_t cgs = (size_t) cap_groups;
        switch (L.gu_type) {
            case 16: launch_native_gu_fused<16, GU_LPR>(q, cgs, grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 18: launch_native_gu_fused<18, GU_LPR>(q, cgs, grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 21: launch_native_gu_fused<21, GU_LPR>(q, cgs, grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 22: launch_native_gu_fused<22, GU_LPR>(q, cgs, grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            default: launch_native_gu_fused<23, GU_LPR>(q, cgs, grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
        }
    } else {
    // the gate and up rows in one launch, or one a type where they differ (native_mixed_gu's layers)
    auto gu_rows = [&](int ty, int part) {
        if (gu_pair_selected(q, L, ty, part)) {
            switch (ty) {
                case 18: launch_native_gu_pair<18>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); return;
                case 21: launch_native_gu_pair<21>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); return;
                case 22: launch_native_gu_pair<22>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); return;
            }
        }
        switch (ty) {
            case 16: launch_native_gu<16, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 17: launch_native_gu<17, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 18: launch_native_gu<18, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 21: launch_native_gu<21, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 22: launch_native_gu<22, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 23: launch_native_gu<23, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 29: launch_native_gu<29, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 19: launch_native_gu<19, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 42: launch_native_gu<42, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 12: launch_native_gu<12, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 13: launch_native_gu<13, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 8: launch_native_gu<8, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 6: launch_native_gu<6, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 2: launch_native_gu<2, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 3: launch_native_gu<3, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 14: launch_native_gu<14, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 10: launch_native_gu<10, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 11: launch_native_gu<11, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 144: launch_native_gu<144, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 145: launch_native_gu<145, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 146: launch_native_gu<146, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 147: launch_native_gu<147, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 148: launch_native_gu<148, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            case 149: launch_native_gu<149, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, part); break;
            default: throw core::DeviceError("native_expert_grouped: gate/up type " + std::to_string(ty));
        }
    };
    if (L.up_type >= 0 && L.up_type != L.gu_type) {
        gu_rows(L.gu_type, 0);
        gu_rows(L.up_type, 1);
    } else {
        gu_rows(L.gu_type, -1);
    }
    const long long nh = (long long) cap_entries * L.n_ff;
    if (had) {
        q.parallel_for(sycl::range<1>((size_t) nh), [=](sycl::id<1> i) {
            float g = gate[i], u = up[i];
            swiglu_clamp(g, u, lim);
            h[i] = canon::silu_mul(g, u);
        });
        if (had) launch_hadamard_rows<float>(q, h, h, cap_entries, L.n_ff, L.had_block, L.d_signs, "native_expert_grouped");
        launch_quantize_q8_1_canon(q, h, hq, nh);
    } else {
        // SwiGLU into q8_1 in one pass over [grp_start[0], grp_start[n]): the product rounded to FP32 on its own, as
        // when it went through memory, then launch_quantize_q8_1_canon's body (a sub-group is one block of one entry)
        const int64_t n_ff = L.n_ff;
        q.parallel_for(sycl::nd_range<1>((size_t) ((nh + 255) / 256) * 256, 256),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
            const long long i = (long long) it.get_global_linear_id();
            if (i >= nh) return;
            const long long e = i / n_ff;
            if (e < grp_start[0] || e >= grp_start[*n_groups]) return;
            const sycl::sub_group sg = it.get_sub_group();
            // the SwiGLU (after GLM-5.3-Flash's clamp; none elsewhere) and the int8 scale of canon_math.hpp: the CPU's
            // rows of the same expert are the same bits
            float g = gate[i], u = up[i];
            swiglu_clamp(g, u, lim);
            q8act::q8_1_block<q8act::Canon<32>>(sg, canon::silu_mul(g, u), reinterpret_cast<uint8_t*>(hq + i / 32),
                                                (int) (i % 32));
        });
    }
    }
    sycl::event e;
    if (lds_down) {
        const size_t cgs = (size_t) cap_groups;
        if (ws_down) {
            if (L.d_type == 20) e = launch_native_down_lds<20, down_lpr<20>(), true>(q, cgs, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            else e = launch_native_down_lds<42, down_lpr<42>(), true>(q, cgs, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        } else if (L.d_type == 20) e = launch_native_down_lds<20, down_lpr<20>()>(q, cgs, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        else e = launch_native_down_lds<42, down_lpr<42>()>(q, cgs, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    } else
    switch (L.d_type) {
        case 20: e = launch_native_down<20, down_lpr<20>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 23: e = launch_native_down<23, down_lpr<23>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 42: e = launch_native_down<42, down_lpr<42>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 7: e = launch_native_down<7, down_lpr<7>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 8: e = launch_native_down<8, down_lpr<8>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 6: e = launch_native_down<6, down_lpr<6>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 2: e = launch_native_down<2, down_lpr<2>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 3: e = launch_native_down<3, down_lpr<3>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 14: e = launch_native_down<14, down_lpr<14>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 10: e = launch_native_down<10, down_lpr<10>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 11: e = launch_native_down<11, down_lpr<11>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 12: e = launch_native_down<12, down_lpr<12>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 13: e = launch_native_down<13, down_lpr<13>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        // the grid formats as down rows (GLM-5.3-Flash's unsloth files: IQ3_S, IQ3_XXS), seg_dot_multi as gate/up's
        case 16: e = launch_native_down<16, down_lpr<16>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 17: e = launch_native_down<17, down_lpr<17>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 18: e = launch_native_down<18, down_lpr<18>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 21: e = launch_native_down<21, down_lpr<21>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 22: e = launch_native_down<22, down_lpr<22>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        // the trellis types (Gyro)
        case 144: e = launch_native_down<144, down_lpr<144>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 145: e = launch_native_down<145, down_lpr<145>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 146: e = launch_native_down<146, down_lpr<146>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 147: e = launch_native_down<147, down_lpr<147>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 148: e = launch_native_down<148, down_lpr<148>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 149: e = launch_native_down<149, down_lpr<149>()>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        default: throw core::DeviceError("native_expert_grouped: down type " + std::to_string(L.d_type));
    }
    sync_if_needed(stream, e, "native_expert_grouped");
}

namespace {
// sum_i w[i] * x[i] over a BF16 row of `len` values for `n` (<= kEntriesPass) FP32 rows xs[j]: the lanes over the row's
// value pairs, then a sub-group sum
inline void bf16_dot_multi(const sycl::sub_group& sg, const uint32_t* w, const float* const* xs, int n, int64_t len,
                           int lane, float* s) {
    for (int j = 0; j < kEntriesPass; ++j) s[j] = 0.0f;
    for (int64_t i = lane; i < len / 2; i += kWarp) {
        const uint32_t p = w[i];
        const float w0 = sycl::bit_cast<float>(p << 16), w1 = sycl::bit_cast<float>(p & 0xffff0000u);
        for (int j = 0; j < kEntriesPass; ++j)
            if (j < n) s[j] += w0 * xs[j][2 * i] + w1 * xs[j][2 * i + 1];
    }
    for (int j = 0; j < kEntriesPass; ++j) s[j] = sycl::reduce_over_group(sg, s[j], sycl::plus<float>());
}
}  // namespace

void native_expert_grouped_bf16(const NativeExpertLayout& L, const unsigned long long* grp_ptr,
                                const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                                const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const float* x,
                                void* scratch, float* out, void* stream, int64_t grid_groups) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (L.gu_type != 30 || L.d_type != 30) throw core::DeviceError("native_expert_grouped_bf16: not a BF16 expert");
    auto& q = queue_for(stream);
    const float lim = expert_swiglu_limit();
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = reinterpret_cast<float*>(static_cast<uint8_t*>(scratch) + fa);
    float* h = reinterpret_cast<float*>(static_cast<uint8_t*>(scratch) + 2 * fa);
    const size_t cg = (size_t) (grid_groups <= 0 ? cap_groups : std::min(grid_groups, cap_groups));
    const int64_t N = L.n_embd, FF = L.n_ff;
    constexpr int kRows = 8;   // rows (sub-groups) a work-group
    // gate/up: row r of the gate, then of the up, against each entry's token row
    q.parallel_for(sycl::nd_range<2>({cg, (size_t) ((2 * FF + kRows - 1) / kRows) * kRows * kWarp}, {1, (size_t) kRows * kWarp}),
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kWarp)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int64_t row = (int64_t) it.get_group(1) * kRows + (int64_t) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        if (row >= 2 * FF) return;   // the whole sub-group
        const bool is_up = row >= FF;
        const int64_t r = is_up ? row - FF : row;
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const auto* wr = (const uint32_t*) ((const uint8_t*) grp_ptr[g] + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row);
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; e += kEntriesPass) {
                const int n = sycl::min(kEntriesPass, e1 - e);
                const float* xs[kEntriesPass];
                for (int j = 0; j < kEntriesPass; ++j) xs[j] = x + (size_t) ent_tok[j < n ? e + j : e] * N;
                float s[kEntriesPass];
                bf16_dot_multi(sg, wr, xs, n, N, lane, s);
                if (lane == 0)
                    for (int j = 0; j < n; ++j) (is_up ? up : gate)[(size_t) (e + j) * FF + r] = s[j];
            }
        }
    });
    // SwiGLU over the call's entries, in FP32 (the down rows read floats)
    const int64_t nh = cap_entries * FF;
    q.parallel_for(sycl::range<1>((size_t) nh), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], e = i / FF;
        if (e < grp_start[0] || e >= grp_start[*n_groups]) return;
        float g = gate[i], u = up[i];
        swiglu_clamp(g, u, lim);
        h[i] = (g / (1.0f + sycl::exp(-g))) * u;
    });
    // down: row r of the output for each entry, written to row ent_dst[e]
    const sycl::event e = q.parallel_for(
        sycl::nd_range<2>({cg, (size_t) ((N + kRows - 1) / kRows) * kRows * kWarp}, {1, (size_t) kRows * kWarp}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kWarp)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int64_t r = (int64_t) it.get_group(1) * kRows + (int64_t) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        if (r >= N) return;
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const auto* wr = (const uint32_t*) ((const uint8_t*) grp_ptr[g] + L.down_off + (size_t) r * L.d_row);
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int en = e0; en < e1; en += kEntriesPass) {
                const int n = sycl::min(kEntriesPass, e1 - en);
                const float* xs[kEntriesPass];
                for (int j = 0; j < kEntriesPass; ++j) xs[j] = h + (size_t) (j < n ? en + j : en) * FF;
                float s[kEntriesPass];
                bf16_dot_multi(sg, wr, xs, n, FF, lane, s);
                if (lane == 0)
                    for (int j = 0; j < n; ++j) out[(size_t) ent_dst[en + j] * N + r] = s[j];
            }
        }
    });
    sync_if_needed(stream, e, "native_expert_grouped_bf16");
}

}  // namespace strata::kernels
