// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
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
#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/runtime.hpp"
#include "cuda_intrinsics.hpp"
#include "device_target.hpp"
#include "iq_bits.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

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
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        const block_q8_1* chunk = bq8_1 + iqs;
        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            sumi = dp4a(get_int_b4(chunk->qs, j * 2 + 0), r.qx[j], sumi);
            sumi = dp4a(get_int_b4(chunk->qs, j * 2 + 1), r.qy[j], sumi);
        }
        return r.d * low_half(chunk->ds) * (float) sumi;
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
    static W load(const void* vbq, int kbx, int iqs) {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
        const uint32_t q2 = (uint32_t) get_int_b2(bq2->qs, iqs);
        const uint32_t aux32 = (uint32_t) get_int_b2(bq2->qs, iqs + 1);
        W r{};
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const Int2 grid_pos = grid_pair(iq2xxs_grid, byte_of(q2, k0 / 2));
            const uint32_t s8 = ksigns_byte((uint8_t) (aux32 >> (7 * k0 / 2)));
            r.g[k0] = grid_pos.x; r.m[k0] = sign_mask(s8);
            r.g[k0 + 1] = grid_pos.y; r.m[k0 + 1] = sign_mask(s8 >> 4);
        }
        r.ls = (int) (aux32 >> 27 | 1);
        r.d = (float) bq2->d;
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        const int sumi = signed_sum(r, &bq8_1[iqs / 2]) * r.ls / 8;
        return r.d * low_half(bq8_1[iqs / 2].ds) * (float) sumi;
    }
};

template<> struct Split<18> {   // IQ3_XXS
    using W = SignedGrid;
    static W load(const void* vbq, int kbx, int iqs) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const U2 pk = get_int2_b2(bq3->qs, iqs);
        const uint32_t packed[2] = {pk.x, pk.y};
        const uint32_t aux32 = (uint32_t) get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
        W r{};
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t s8 = ksigns_byte((uint8_t) (aux32 >> (7 * l0 / 2)));
            r.g[l0] = (int) iq3xxs_grid[byte_of(packed[l0 / 4], (l0 + 0) % 4)]; r.m[l0] = sign_mask(s8);
            r.g[l0 + 1] = (int) iq3xxs_grid[byte_of(packed[l0 / 4], (l0 + 1) % 4)]; r.m[l0 + 1] = sign_mask(s8 >> 4);
        }
        r.ls = (int) (aux32 >> 28);
        r.d = (float) bq3->d;
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        int sumi = signed_sum(r, &bq8_1[iqs / 2]);
        sumi = (r.ls * sumi + sumi / 2) / 2;
        return r.d * low_half(bq8_1[iqs / 2].ds) * (float) sumi;
    }
};

template<> struct Split<21> {   // IQ3_S
    using W = SignedGrid;
    static W load(const void* vbq, int kbx, int iqs) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const U2 pk = get_int2_b2(bq3->qs, iqs);
        const uint32_t packed[2] = {pk.x, pk.y};
        const int qh = bq3->qh[iqs / 2];
        const uint32_t signs_packed = (uint32_t) get_int_b2(bq3->signs, iqs / 2);
        W r{};
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t s8 = byte_of(signs_packed, l0 / 2);
            r.g[l0] = (int) iq3s_grid[byte_of(packed[l0 / 4], (l0 + 0) % 4) | ((qh << (8 - l0)) & 0x100)];
            r.g[l0 + 1] = (int) iq3s_grid[byte_of(packed[l0 / 4], (l0 + 1) % 4) | ((qh << (7 - l0)) & 0x100)];
            r.m[l0] = sign_mask(s8);
            r.m[l0 + 1] = sign_mask(s8 >> 4);
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.d = (float) bq3->d;
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        const int sumi = signed_sum(r, &bq8_1[iqs / 2]) * r.ls;
        return r.d * low_half(bq8_1[iqs / 2].ds) * (float) sumi;
    }
};

// IQ2_XS and IQ2_S: two half sums, each with its 4-bit scale
struct SignedGrid2 { int g[8]; uint32_t m[8]; int ls0, ls1; float d; };
inline float apply_ls2(const SignedGrid2& r, const block_q8_1* bq8_1, int iqs) {
    const block_q8_1* q8 = &bq8_1[iqs / 2];
    int sumi0 = 0, sumi1 = 0;
    for (int j = 0; j < 4; ++j) sumi0 = dot_masked(r.g[j], get_int_b4(q8->qs, j), r.m[j], sumi0);
    for (int j = 4; j < 8; ++j) sumi1 = dot_masked(r.g[j], get_int_b4(q8->qs, j), r.m[j], sumi1);
    const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
    return r.d * low_half(q8->ds) * (float) sumi;
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
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) { return apply_ls2(r, bq8_1, iqs); }
};

template<> struct Split<22> {   // IQ2_S
    using W = SignedGrid2;
    static W load(const void* vbq, int kbx, int iqs) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const uint32_t qs = (uint32_t) get_int_b2(bq2->qs, iqs / 2);
        const int qh = bq2->qh[iqs / 2];
        const uint32_t signs_packed = (uint32_t) get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        W r{};
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const Int2 grid_pos = grid_pair(iq2s_grid, byte_of(qs, l0 / 2) | ((qh << (8 - l0)) & 0x300));
            const uint32_t s8 = byte_of(signs_packed, l0 / 2);
            r.g[l0] = grid_pos.x; r.m[l0] = sign_mask(s8);
            r.g[l0 + 1] = grid_pos.y; r.m[l0 + 1] = sign_mask(s8 >> 4);
        }
        r.d = (float) bq2->d;
        return r;
    }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) { return apply_ls2(r, bq8_1, iqs); }
};

inline float iq1m_scale(const uint16_t* sc) {
    const uint16_t u16 = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    return (float) sycl::bit_cast<sycl::half>(u16);
}

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
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        int sumi = 0;
        for (int l = 0; l < 2; ++l) {
            sumi = dp4a(r.v[l].x, q8[l + 0], sumi);
            sumi = dp4a(r.v[l].y, q8[l + 4], sumi);
        }
        return r.d * low_half(bq8_1->ds) * (float) sumi;
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
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            sumi = dp4a(r.v[j].x, get_int_b4(bq8_1[iqs / 4].qs, j + 0), sumi);
            sumi = dp4a(r.v[j].y, get_int_b4(bq8_1[iqs / 4].qs, j + 4), sumi);
        }
        sumi *= r.ls;
        return r.d * low_half(bq8_1[iqs / 4].ds) * (float) sumi;
    }
};

// ---------------------------------------------------------------- Unsloth's UD-Q4_K_XL experts (upstream efeffd8)
// Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down: llama.cpp's vec_dot_*_q8_1 (vecdotq.cuh, VDR 2 each), transcribed through
// upstream's CUDA version; the Q5_1 min term is its one departure (below).  These have no decode-once split: `load`
// keeps the block's address and `apply` is the whole dot, so a row read for several columns decodes it per column.
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
inline float vec_dot_q4_K_q8_1(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs) {
    const block_q4_K* bq4_K = (const block_q4_K*) vbq + kbx;
    int v[2];
    int u[2 * QR4_K] = {};
    float d8[QR4_K] = {};
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const auto* q4 = reinterpret_cast<const int*>(bq4_K->qs + (size_t) (16 * bq8_offset + 4 * ((iqs / 2) % 4)));
    v[0] = q4[0];
    v[1] = q4[4];
    uint16_t aux[2];
    k_scale_min(bq4_K->scales, bq8_offset, aux);
    const auto* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = low_half(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}
inline float vec_dot_q5_K_q8_1(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs) {
    const block_q5_K* bq5_K = (const block_q5_K*) vbq + kbx;
    int vl[2];
    int vh[2];
    int u[2 * QR5_K] = {};
    float d8[QR5_K] = {};
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const auto* ql = reinterpret_cast<const int*>(bq5_K->qs + (size_t) (16 * bq8_offset + 4 * ((iqs / 2) % 4)));
    const auto* qh = reinterpret_cast<const int*>(bq5_K->qh + (size_t) (4 * ((iqs / 2) % 4)));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint16_t aux[2];
    k_scale_min(bq5_K->scales, bq8_offset, aux);
    const auto* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = low_half(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}
// Q5_1: llama.cpp's integer chain, but the min term multiplies the sum of the QUANTIZED activations (dp4a with
// 0x01010101, times d8) instead of the q8_1 block's `ds.y`, which the quantizer (like llama.cpp's) fills with the sum
// of the ORIGINAL activations: ggml-cpu's convention, and the one the K-quant mins above use.
inline float vec_dot_q5_1_q8_1(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs) {
    const block_q5_1* bq5_1 = (const block_q5_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
    for (int i = 0; i < VDR_Q5_1; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
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
    const float d8 = low_half(bq8_1->ds);
    return (float) sumi * ((float) bq5_1->dm[0] * d8) + (float) sumu * ((float) bq5_1->dm[1] * d8);
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
inline float vec_dot_q8_0_q8_1(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs) {
    const block_q8_0* bq8_0 = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
    for (int i = 0; i < VDR_Q8_0; ++i) sumi = dp4a(get_int_b2(bq8_0->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    return (float) bq8_0->d * low_half(bq8_1->ds) * (float) sumi;
}

template<float (*DOT)(const void*, const block_q8_1*, int, int)>
struct WholeDot {
    struct W { const void* v; int kbx; };
    static W load(const void* vbq, int kbx, int iqs) { (void) iqs; return {vbq, kbx}; }
    static float apply(const W& r, const block_q8_1* bq8_1, int iqs) { return DOT(r.v, bq8_1, r.kbx, iqs); }
};
template<> struct Split<12> : WholeDot<vec_dot_q4_K_q8_1> {};   // Q4_K
template<> struct Split<13> : WholeDot<vec_dot_q5_K_q8_1> {};   // Q5_K
template<> struct Split<7> : WholeDot<vec_dot_q5_1_q8_1> {};    // Q5_1
template<> struct Split<8> : WholeDot<vec_dot_q8_0_q8_1> {};    // Q8_0
template<> struct Split<6> : WholeDot<vec_dot_q5_0_q8_1> {};    // Q5_0

// ---------------------------------------------------------------- the formats
// bsz = bytes per block, qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
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
template<> struct Fmt<29> : FmtOf<29, (int) sizeof(block_iq1_m), 256, 8, 1> {};
template<> struct Fmt<42> : FmtOf<42, (int) sizeof(block_q2_0), 64, 2, 1> {};
template<> struct Fmt<12> : FmtOf<12, (int) sizeof(block_q4_K), 256, QI4_K / VDR_Q4_K, VDR_Q4_K> {};
template<> struct Fmt<13> : FmtOf<13, (int) sizeof(block_q5_K), 256, QI5_K / VDR_Q5_K, VDR_Q5_K> {};
template<> struct Fmt<7> : FmtOf<7, (int) sizeof(block_q5_1), 32, QI5_1 / VDR_Q5_1, VDR_Q5_1> {};
template<> struct Fmt<8> : FmtOf<8, (int) sizeof(block_q8_0), 32, QI8_0 / VDR_Q8_0, VDR_Q8_0> {};
template<> struct Fmt<6> : FmtOf<6, (int) sizeof(block_q5_0), 32, QI5_0 / VDR_Q5_0, VDR_Q5_0> {};

// CUDA's __shfl_xor butterfly, in the same order, so a row's sum does not depend on the backend's reduction tree.
inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline float warp_max(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, o));
    return v;
}

// A row's block kbx, its offset in 32 bits: an int index into the block array made the compiler multiply and add the
// addresses in 64 bits (two SIMD16 halves each on Xe).  A row is far below 4 GiB.
template<class F>
inline const void* block_at(const uint8_t* row, int kbx) { return row + (size_t) ((uint32_t) kbx * (uint32_t) F::bsz); }

// One row against up to NE activations (x[e], the first `n` in use) on LPR lanes (a segment of the sub-group, LPR a
// power of two up to 32): the sub-group carries WARP / LPR rows at once, so a short row (the experts' 640-wide down
// projection: 180-360 bytes) keeps its lanes busy and reduces in log2(LPR) steps.  Each block is loaded once for all
// the entries (upstream 8ec94aa); entry e's sum is what one entry alone gives, bit for bit.
template<int TY, int LPR, int NE>
inline void seg_dot_multi(const sycl::sub_group& sg, const uint8_t* row, const block_q8_1* const (&x)[NE], int n,
                          int nb, int seg_lane, float (&out)[NE]) {
    using F = Fmt<TY>;
    float s[NE];
    for (int e = 0; e < NE; ++e) s[e] = 0.0f;
    for (int k = seg_lane; k < nb * F::ipb; k += LPR) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const auto wd = F::S::load(block_at<F>(row, kbx), 0, iqs);
        const size_t xo = (size_t) ((uint32_t) kbx * (uint32_t) (F::qk / 32));
        for (int e = 0; e < NE; ++e)
            if (e < n) s[e] += F::S::apply(wd, x[e] + xo, iqs);
    }
    for (int e = 0; e < NE; ++e) {
        for (int o = LPR / 2; o > 0; o >>= 1) s[e] += sycl::permute_group_by_xor(sg, s[e], o);
        out[e] = s[e];
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
        const size_t xo = (size_t) ((uint32_t) kbx * (uint32_t) (F::qk / 32));
        for (int c = 0; c < NC; ++c) s[c] += F::S::apply(wd, x + (size_t) c * xstride + xo, iqs);
    }
    for (int c = 0; c < NC; ++c) out[c] = warp_sum(sg, s[c]);
}

template<int TY, int NC>
sycl::event launch_mmvq_n(sycl::queue& q, const uint8_t* w, size_t row_bytes, const block_q8_1* x, float* y, int n_in,
                          int n_out) {
    const size_t groups = (size_t) ((n_out + 3) / 4);
    return q.parallel_for(sycl::nd_range<1>(groups * 4 * kWarp, 4 * kWarp),
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

// 1, 2, 4 or 8 columns a pass, 8 at a time past 8 (the verify window takes up to 8 tokens)
template<int TY>
sycl::event launch_mmvq(sycl::queue& q, const uint8_t* w, size_t row_bytes, const block_q8_1* x, float* y, int n_in,
                        int n_out, int ncols) {
    sycl::event e;
    for (int c0 = 0; c0 < ncols;) {
        const int n = ncols - c0;
        const block_q8_1* xc = x + (size_t) c0 * (n_in / 32);
        float* yc = y + (size_t) c0 * n_out;
        const int take = n >= 8 ? 8 : n >= 4 ? 4 : n >= 2 ? 2 : 1;
        if (take == 8) e = launch_mmvq_n<TY, 8>(q, w, row_bytes, xc, yc, n_in, n_out);
        else if (take == 4) e = launch_mmvq_n<TY, 4>(q, w, row_bytes, xc, yc, n_in, n_out);
        else if (take == 2) e = launch_mmvq_n<TY, 2>(q, w, row_bytes, xc, yc, n_in, n_out);
        else e = launch_mmvq_n<TY, 1>(q, w, row_bytes, xc, yc, n_in, n_out);
        c0 += take;
    }
    return e;
}

// ---------------------------------------------------------------- grouped native experts
// Lanes a row in the experts' projections, measured on the B70 (bench/results/2026-10-02-xe-decode-gpu): 16 for the
// 2560-wide gate/up rows, 8 for the 640-wide down rows (32 lanes left most of a 180-360 byte row's lanes idle).
constexpr int GU_LPR = 16;
constexpr int DOWN_LPR = 8;
constexpr int GU_ROWS = 8;     // rows per work-group (one sub-group each)
// entries (a window's tokens routed to one expert) a pass over the row; the verify window of --spec 4 fits in one
constexpr int kEntriesPass = 4;

// The experts' projections: LPR lanes a row (seg_dot), GU_ROWS sub-groups a work-group, so a work-group covers
// GU_ROWS * kWarp / LPR rows.  A row in range for some lanes of a sub-group and not others is masked per segment,
// with no early return, so the segment shuffles see the whole sub-group.
template<int TG, int LPR>
sycl::event launch_native_gu(sycl::queue& q, size_t cap_groups, const unsigned long long* grp_ptr,
                             const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok,
                             const block_q8_1* xq, NativeExpertLayout L, float* gate, float* up) {
    constexpr int RPW = kWarp / LPR;
    const size_t xgroups = (size_t) ((2 * L.n_ff + (int64_t) GU_ROWS * RPW - 1) / ((int64_t) GU_ROWS * RPW));
    return q.parallel_for(sycl::nd_range<2>({cap_groups, xgroups * GU_ROWS * kWarp}, {1, GU_ROWS * kWarp}),
                          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kWarp)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int warp = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int row0 = ((int) it.get_group(1) * GU_ROWS + warp) * RPW;
        if (row0 >= 2 * L.n_ff) return;                                      // the whole sub-group
        const int row = row0 + lane / LPR, seg_lane = lane % LPR;
        const bool valid = row < 2 * L.n_ff;
        const bool is_up = row >= L.n_ff;
        const int r = valid ? (is_up ? row - (int) L.n_ff : row) : 0;
        const int nb = valid ? (int) (L.n_embd / Fmt<TG>::qk) : 0, xb = (int) (L.n_embd / 32);
        // the groups g, g + grid rows, ...: a group's rows are the same code in the same order whatever the grid
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const uint8_t* wr = blob + (is_up && valid ? L.up_off : 0) + (size_t) r * L.gu_row;
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; e += kEntriesPass) {   // the row's blocks decoded once for kEntriesPass entries
                const int n = sycl::min(kEntriesPass, e1 - e);
                const block_q8_1* xs[kEntriesPass];
                for (int j = 0; j < kEntriesPass; ++j) xs[j] = xq + (size_t) ent_tok[j < n ? e + j : e] * xb;
                float s[kEntriesPass];
                seg_dot_multi<TG, LPR, kEntriesPass>(sg, wr, xs, n, nb, seg_lane, s);
                if (valid && seg_lane == 0)
                    for (int j = 0; j < n; ++j) (is_up ? up : gate)[(size_t) (e + j) * L.n_ff + r] = s[j];
            }
        }
    });
}

template<int TD, int LPR>
sycl::event launch_native_down(sycl::queue& q, size_t cap_groups, const unsigned long long* grp_ptr,
                               const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                               const block_q8_1* hq, NativeExpertLayout L, float* out) {
    constexpr int RPW = kWarp / LPR;
    const size_t xgroups = (size_t) ((L.n_embd + (int64_t) 8 * RPW - 1) / ((int64_t) 8 * RPW));
    return q.parallel_for(sycl::nd_range<2>({cap_groups, xgroups * 8 * kWarp}, {1, 8 * kWarp}),
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
                seg_dot_multi<TD, LPR, kEntriesPass>(sg, wr, xs, n, nb, seg_lane, s);
                if (valid && seg_lane == 0)
                    for (int j = 0; j < n; ++j) out[(size_t) ent_dst[e + j] * L.n_embd + r] = s[j];
            }
        }
    });
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
// n is a multiple of 32, so every sub-group is either wholly in range or wholly out of it.
sycl::event launch_quantize_q8_1(sycl::queue& q, const float* x, block_q8_1* y, long long n,
                                 const std::vector<sycl::event>& deps = {}) {
    const size_t total = (size_t) ((n + 255) / 256) * 256;
    return q.parallel_for(sycl::nd_range<1>(total, 256), deps,
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
        const long long i = (long long) it.get_global_linear_id();
        if (i >= n) return;
        const sycl::sub_group sg = it.get_sub_group();
        const float xi = x[i];
        const float amax = warp_max(sg, sycl::fabs(xi));
        const float sum = warp_sum(sg, xi);
        const float d = amax / 127.0f;
        const int8_t qv = amax == 0.0f ? 0 : (int8_t) sycl::round(xi / d);
        const long long ib = i / 32, iqs = i % 32;
        y[ib].qs[iqs] = qv;
        if (iqs == 0) y[ib].ds = ggml_half2(sycl::half(d), sycl::half(sum));
    });
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
// The dequantizers hand their values to an output, runs of 4 or 8 at an offset in the 256-value piece they decode:
// ToGlobal stores them to memory, iq_gemm's ToLocal to a work-group's tile in local memory.
template<typename dst_t> inline dst_t cvt(float v);
template<> inline float cvt<float>(float v) { return v; }
template<> inline sycl::half cvt<sycl::half>(float v) { return sycl::half(v); }
// N consecutive values in one vector store instead of N scalar ones (y is aligned to N values in every caller): the
// dequant's fp16 writes were 128-163 GB/s with one 2-byte store per value (bench/results/2026-10-02-xe-decode-gpu).
template<int N, typename dst_t>
inline void store(dst_t* y, const float* v) {
    sycl::vec<dst_t, N> o;
    for (int j = 0; j < N; ++j) o[j] = cvt<dst_t>(v[j]);
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
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint32_t signs = ksigns_byte((uint8_t) ((aux32 >> 7 * il) & 127));
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = d * (int) (uint8_t) (grid >> (8 * j)) * (signs & (1u << j) ? -1.f : 1.f);
    o.template put<8>(y, v);
}
template<typename Out>
void dq_iq2_xs(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid % 4, ib = tid / 4;          // neighbours in one sub-block: contiguous writes
    const int y = (int) (32 * ib + 8 * il);
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint64_t grid = iq2xs_grid[q2[il] & 511];   // whole, and the signs computed: as dq_iq3_xxs
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint32_t signs = ksigns_byte((uint8_t) (q2[il] >> 9));
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = d * (int) (uint8_t) (grid >> (8 * j)) * (signs & (1u << j) ? -1.f : 1.f);
    o.template put<8>(y, v);
}
template<typename Out>
void dq_iq2_s(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid % 4, ib = tid / 4;          // neighbours in one sub-block: contiguous writes
    const int y = (int) (32 * ib + 8 * il);
    const uint64_t grid = iq2s_grid[x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)];   // whole
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint32_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = d * (int) (uint8_t) (grid >> (8 * j)) * (signs & (1u << j) ? -1.f : 1.f);
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
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint32_t signs = ksigns_byte((uint8_t) ((aux32 >> 7 * il) & 127));
    float v[8];
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * (int) byte_of(grid1, j) * (signs & (1u << j) ? -1.f : 1.f);
        v[j + 4] = d * (int) byte_of(grid2, j) * (signs & (1u << (j + 4)) ? -1.f : 1.f);
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
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint32_t signs = x[ibs].signs[4 * ib + il];
    float v[8];
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * (int) byte_of(grid1, j) * (signs & (1u << j) ? -1.f : 1.f);
        v[j + 4] = d * (int) byte_of(grid2, j) * (signs & (1u << (j + 4)) ? -1.f : 1.f);
    }
    o.template put<8>(y, v);
}
template<typename Out>
void dq_iq1_m(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid % 4, ib = tid / 4;          // neighbours in one sub-block: contiguous writes
    const int y = (int) (32 * ib + 8 * il);
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = iq1m_scale(sc) * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    const uint32_t g = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    const uint32_t grid32[2] = {g & 0x0f0f0f0f, (g >> 4) & 0x0f0f0f0f};
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = d * ((int8_t) byte_of(grid32[j / 4], j % 4) + delta);
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
        lo[j] = d * (int8_t) byte_of((uint32_t) v.x, j);
        hi[j] = d * (int8_t) byte_of((uint32_t) v.y, j);
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
        const float dl = (float) x->d * (us - 32);
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        float v[4];
        for (int l = l0; l < l0 + 4; ++l)
            v[l - l0] = dl * (float) ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
        o.template put<4>(128 * n + 32 * j + l0, v);
    }
}
template<typename Out>
void dq_iq4_xs(const void* vx, int64_t ibs, const Out& o, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid % 4, ib = tid / 4;              // neighbours in one sub-block, as dq_iq4_nl
    const int y = (int) (32 * ib + 4 * il);
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    float lo[4], hi[4];
    for (int j = 0; j < 4; ++j) {
        lo[j] = d * kvalues_iq4nl[q4[j] & 0xf];
        hi[j] = d * kvalues_iq4nl[q4[j] >> 4];
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
        case 23: dq_iq4_xs(vx, ibs, o, tid); break;
        case 11: dq_q3_k(vx, ibs, o, tid); break;
        case 42: dq_q2_0(vx, ibs, o, tid); break;
        case 30: dq_bf16(vx, ibs, o, tid); break;   // the token embedding only (embed_type_supported)
        case 12: dq_q4_k(vx, ibs, o, tid); break;
        case 13: dq_q5_k(vx, ibs, o, tid); break;
        case 7: dq_q5_1(vx, ibs, o, tid); break;
        case 8: dq_q8_0(vx, ibs, o, tid); break;
        case 6: dq_q5_0(vx, ibs, o, tid); break;
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
    else if constexpr (TYPE == 42) dq_q2_0(vx, ibs, o, tid);
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
        if constexpr (WHOLE)
            if (k0 + KC < K)   // the next block of A, 32 values (64 bytes, the most a prefetch takes) at a time
                for (int kp = 0; kp < KC; kp += 32)
                    mx::joint_matrix_prefetch<G_MS, 32>(sg, const_cast<sycl::half*>(x) + m0 * K + k0 + KC + kp, K,
                                                        mx::layout::row_major,
                                                        syclex::properties{syclex::prefetch_hint_L1});
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
bool in_iq_gemm(int t) { return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42; }
bool is_iq(int t) { return in_iq_gemm(t) || t == 11 || t == 12 || t == 13 || t == 6 || t == 7 || t == 8; }

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
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 30: return (size_t) n * 2;   // BF16: the token embedding only (iq_embed_rows, iq_dequant_f32)
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    require(n_cols % 32 == 0, "quantize_q8_1_rows");
    sync_if_needed(stream, launch_quantize_q8_1(queue_for(stream), x, (block_q8_1*) y, n), "quantize_q8_1_rows");
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
        case 42: e = launch_mmvq<42>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 12: e = launch_mmvq<12>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 13: e = launch_mmvq<13>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 7: e = launch_mmvq<7>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 8: e = launch_mmvq<8>(q, W, rb, X, y, n_in, n_out, ncols); break;
        case 6: e = launch_mmvq<6>(q, W, rb, X, y, n_in, n_out, ncols); break;
        default: throw core::DeviceError("iq_mmvq: type " + std::to_string(t) + " is not supported");
    }
    sync_if_needed(stream, e, "iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    require(n % 256 == 0 && is_iq(t), "iq_dequant_f16");
    sync_if_needed(stream, launch_dequant_flat<sycl::half>(queue_for(stream), t, src, n, (sycl::half*) dst),
                   "iq_dequant_f16");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    require(n % 256 == 0 && embed_type_supported(t), "iq_dequant_f32");
    sync_if_needed(stream, launch_dequant_flat<float>(queue_for(stream), t, src, n, dst), "iq_dequant_f32");
}

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    require(n_embd % 256 == 0 && embed_type_supported(t), "iq_embed_rows");
    const auto* tab = (const uint8_t*) table;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<2>({(size_t) n_tok, (size_t) (n_embd / 256) * 32}, {1, 32}), [=](sycl::nd_item<2> it) {
            const int64_t tok = (int64_t) it.get_group(0), b = (int64_t) it.get_group(1);
            const uint8_t* row = tab + (size_t) tokens[tok] * row_bytes;
            dq_dispatch(t, row, b, ToGlobal<float>{out + (size_t) tok * n_embd + b * QK_K}, (int) it.get_local_id(1));
        });
    sync_if_needed(stream, e, "iq_embed_rows");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    require(n_embd % 256 == 0 && is_iq(t), "iq_dequant_gu_f16");
    const int64_t per_row = n_embd / 256;
    auto* y = (sycl::half*) dst;
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
        default: e = launch_iq_gemm<42>(q, x, w, interleave, Y, ldy, bounds, G, max_rows, N, K); break;
    }
    sync_if_needed(stream, e, "iq_gemm_grouped_f16");
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    // the cases of native_expert_grouped's two switches, and the prompt path's dequantizer (iq_dequant_gu_f16 and the
    // down rows) for both
    const bool gu = in_iq_gemm(gu_type) ? gu_type != 20 : gu_type == 12 || gu_type == 13 || gu_type == 8;
    const bool d = d_type == 20 || d_type == 23 || d_type == 42 || d_type == 6 || d_type == 7 || d_type == 8;
    const auto qk = [](int t) { return t == 20 || t == 6 || t == 7 || t == 8 ? 32 : t == 42 ? 64 : 256; };
    return gu && d && n_embd > 0 && n_ff > 0 && n_embd % qk(gu_type) == 0 && n_ff % qk(d_type) == 0 &&
           n_embd % 256 == 0;   // the dequantizers write whole 256-value pieces of a gate/up row
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
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
    // upstream 0bc1594: the gate/up and down grids take grid_groups rows of groups (each striding over the groups) and
    // SwiGLU + q8_1 run as one pass over the call's own entries; STRATA_GROUPED_V1=1: a row per possible group and the
    // two passes over every entry (A/B; the same bits)
    static const bool v1 = [] {
        const char* v = std::getenv("STRATA_GROUPED_V1");
        return v != nullptr && v[0] == '1';
    }();
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const size_t cg = (size_t) (v1 || grid_groups <= 0 ? cap_groups : std::min(grid_groups, cap_groups));
    switch (L.gu_type) {
        case 16: launch_native_gu<16, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 17: launch_native_gu<17, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 18: launch_native_gu<18, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 21: launch_native_gu<21, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 22: launch_native_gu<22, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 23: launch_native_gu<23, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 29: launch_native_gu<29, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 42: launch_native_gu<42, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 12: launch_native_gu<12, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 13: launch_native_gu<13, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 8: launch_native_gu<8, GU_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        default: throw core::DeviceError("native_expert_grouped: gate/up type " + std::to_string(L.gu_type));
    }
    const long long nh = (long long) cap_entries * L.n_ff;
    if (v1) {
        q.parallel_for(sycl::range<1>((size_t) nh), [=](sycl::id<1> i) {
            const float g = gate[i];
            // CUDA used __expf; the precise exponential keeps SwiGLU within the reference's float rounding.
            h[i] = (g / (1.0f + sycl::exp(-g))) * up[i];
        });
        launch_quantize_q8_1(q, h, hq, nh);
    } else {
        // SwiGLU into q8_1 in one pass over [grp_start[0], grp_start[n]): the product rounded to FP32 on its own, as
        // when it went through memory, then launch_quantize_q8_1's body (a sub-group is one block of one entry)
        const int64_t n_ff = L.n_ff;
        q.parallel_for(sycl::nd_range<1>((size_t) ((nh + 255) / 256) * 256, 256),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
            const long long i = (long long) it.get_global_linear_id();
            if (i >= nh) return;
            const long long e = i / n_ff;
            if (e < grp_start[0] || e >= grp_start[*n_groups]) return;
            const sycl::sub_group sg = it.get_sub_group();
            const float g = gate[i];
            const float xi = (g / (1.0f + sycl::exp(-g))) * up[i];
            const float amax = warp_max(sg, sycl::fabs(xi));
            const float sum = warp_sum(sg, xi);
            const float d = amax / 127.0f;
            const int8_t qv = amax == 0.0f ? 0 : (int8_t) sycl::round(xi / d);
            const long long ib = i / 32, iqs = i % 32;
            hq[ib].qs[iqs] = qv;
            if (iqs == 0) hq[ib].ds = ggml_half2(sycl::half(d), sycl::half(sum));
        });
    }
    sycl::event e;
    switch (L.d_type) {
        case 20: e = launch_native_down<20, DOWN_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 23: e = launch_native_down<23, DOWN_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 42: e = launch_native_down<42, DOWN_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 7: e = launch_native_down<7, DOWN_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 8: e = launch_native_down<8, DOWN_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 6: e = launch_native_down<6, DOWN_LPR>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        default: throw core::DeviceError("native_expert_grouped: down type " + std::to_string(L.d_type));
    }
    sync_if_needed(stream, e, "native_expert_grouped");
}

}  // namespace strata::kernels
