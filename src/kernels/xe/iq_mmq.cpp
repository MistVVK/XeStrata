// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/iq_mmq.cpp - see include/strata/kernels/iq_mmq.hpp.
//
// The weights' values and scales follow llama.cpp's dequantizers (ggml/src/ggml-quants.c, dequantize_row_*; MIT
// license, third_party/main/ggml/LICENSE, quoted in iq_kernels.cpp): a block of 32 values (16 for IQ2_XS and IQ2_S) is
// its grid or table bytes, signed, times d and the block's scale.  The activations are rounded as its q8_1 (amax / 127
// a block of 32).
//
// A work-group takes a tile of one expert's rows (16, 32 or 64, by the group's largest) and 128 of its outputs, K 128
// values a step: the step's activation rows copied to local memory, the weight rows decoded there (a work-item a block
// of 32), then each sub-group's tiles multiplied a block at a time into int32 accumulators, which are scaled by the
// block's two scales into FP32 ones.  joint_matrix_apply gives a lane its elements but not where they sit in the tile,
// so the scales come as tiles too: loaded with stride 0, a column of one row's scale and a row of one output's, and
// multiplied element by element with the int32 tile (iq_mmq_usable checks that an int32 tile and an FP32 one hold
// their elements in the same order).
//
// In a source file of its own, so its kernels are in a device image apart from the Intel-only ones.
#include "strata/kernels/iq_mmq.hpp"
#include "device_target.hpp"
#include "iq_bits.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using namespace iq_bits;
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));   // one 16-byte access (sycl::uint4 is split on NVPTX)

// The tile shape (NVIDIA's int8 one on a warp; the only one written for so far) and the K step.
constexpr int TM = 16, TN = 16, TK = 16, SG = 32;
constexpr int NE = TM * TN / SG;      // accumulator elements a lane
constexpr int KC = 128;               // K a step: four blocks of 32
constexpr int NB = KC / 32;
constexpr int LDA = KC + 16;          // local row strides in bytes (16-byte multiples, rows apart in the banks)
constexpr int LDB = KC + 16;

template<int S>
constexpr bool built() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
    return false;
#else
    return STRATA_NV_ARCH == 0 || STRATA_NV_ARCH >= 720;   // int8 on NVIDIA's tensor cores from sm_72 on
#endif
}

// ---------------------------------------------------------------- the weights' blocks of 32 values
// v[j]: values 4j .. 4j + 3 as signed bytes; s0, s1: the scales of values 0-15 and 16-31.
struct Sub {
    uint32_t v[8];
    float s0, s1;
};
// the grid bytes (below 128) negated where m is 0xFF
inline uint32_t signed_bytes(uint32_t g, uint32_t m) { return (g ^ m) + (m & 0x01010101u); }

template<int TY> struct Dec;
template<> struct Dec<16> {   // IQ2_XXS: d (0.5 + ls) / 4
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq2_xxs);
    static Sub get(const uint8_t* row, int sb) {
        const block_iq2_xxs* b = reinterpret_cast<const block_iq2_xxs*>(row) + sb / 8;
        const int ib = sb % 8;
        const uint32_t q2 = (uint32_t) get_int_b2(b->qs, 2 * ib), aux = (uint32_t) get_int_b2(b->qs, 2 * ib + 1);
        Sub r;
        for (int k = 0; k < 8; k += 2) {
            const Int2 gp = grid_pair(iq2xxs_grid, byte_of(q2, k / 2));
            const uint32_t s8 = ksigns_byte((uint8_t) (aux >> (7 * k / 2)));
            r.v[k] = signed_bytes((uint32_t) gp.x, sign_mask(s8));
            r.v[k + 1] = signed_bytes((uint32_t) gp.y, sign_mask(s8 >> 4));
        }
        r.s0 = r.s1 = (float) b->d * (float) (2 * (aux >> 28) + 1) * 0.125f;
        return r;
    }
};
template<> struct Dec<18> {   // IQ3_XXS: d (0.5 + ls) / 2
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq3_xxs);
    static Sub get(const uint8_t* row, int sb) {
        const block_iq3_xxs* b = reinterpret_cast<const block_iq3_xxs*>(row) + sb / 8;
        const int ib = sb % 8;
        const U2 pk = get_int2_b2(b->qs, 2 * ib);
        const uint32_t packed[2] = {pk.x, pk.y};
        const uint32_t aux = (uint32_t) get_int_b2(b->qs, QK_K / 16 + ib);
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const uint32_t s8 = ksigns_byte((uint8_t) (aux >> (7 * l / 2)));
            r.v[l] = signed_bytes(iq3xxs_grid[byte_of(packed[l / 4], l % 4)], sign_mask(s8));
            r.v[l + 1] = signed_bytes(iq3xxs_grid[byte_of(packed[l / 4], (l + 1) % 4)], sign_mask(s8 >> 4));
        }
        r.s0 = r.s1 = (float) b->d * (float) (2 * (aux >> 28) + 1) * 0.25f;
        return r;
    }
};
template<> struct Dec<21> {   // IQ3_S: d (1 + 2 ls)
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq3_s);
    static Sub get(const uint8_t* row, int sb) {
        const block_iq3_s* b = reinterpret_cast<const block_iq3_s*>(row) + sb / 8;
        const int ib = sb % 8;
        const U2 pk = get_int2_b2(b->qs, 2 * ib);
        const uint32_t packed[2] = {pk.x, pk.y};
        const int qh = b->qh[ib];
        const uint32_t signs = (uint32_t) get_int_b2(b->signs, ib);
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const uint32_t s8 = byte_of(signs, l / 2);
            r.v[l] = signed_bytes(iq3s_grid[byte_of(packed[l / 4], l % 4) | ((qh << (8 - l)) & 0x100)], sign_mask(s8));
            r.v[l + 1] = signed_bytes(iq3s_grid[byte_of(packed[l / 4], (l + 1) % 4) | ((qh << (7 - l)) & 0x100)],
                                      sign_mask(s8 >> 4));
        }
        r.s0 = r.s1 = (float) b->d * (float) (1 + 2 * ((b->scales[ib / 2] >> (4 * (ib & 1))) & 0x0F));
        return r;
    }
};
template<> struct Dec<17> {   // IQ2_XS: d (0.5 + ls) / 4, a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq2_xs);
    static Sub get(const uint8_t* row, int sb) {
        const block_iq2_xs* b = reinterpret_cast<const block_iq2_xs*>(row) + sb / 8;
        const int ib = sb % 8;
        const U2 pk = get_int2_b2(b->qs, 2 * ib);
        const uint32_t packed[2] = {pk.x, pk.y};
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const uint16_t q2 = half_of(packed[l / 4], (l / 2) % 2);
            const Int2 gp = grid_pair(iq2xs_grid, q2 & 0x1FF);
            const uint32_t s8 = ksigns_byte((uint8_t) (q2 >> 9));
            r.v[l] = signed_bytes((uint32_t) gp.x, sign_mask(s8));
            r.v[l + 1] = signed_bytes((uint32_t) gp.y, sign_mask(s8 >> 4));
        }
        const float d = (float) b->d;
        r.s0 = d * (float) (2 * (b->scales[ib] & 0x0F) + 1) * 0.125f;
        r.s1 = d * (float) (2 * (b->scales[ib] >> 4) + 1) * 0.125f;
        return r;
    }
};
template<> struct Dec<22> {   // IQ2_S: d (0.5 + ls) / 4, a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq2_s);
    static Sub get(const uint8_t* row, int sb) {
        const block_iq2_s* b = reinterpret_cast<const block_iq2_s*>(row) + sb / 8;
        const int ib = sb % 8;
        const uint32_t qs = (uint32_t) get_int_b2(b->qs, ib);
        const int qh = b->qh[ib];
        const uint32_t signs = (uint32_t) get_int_b2(b->qs, QK_K / 32 + ib);
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const Int2 gp = grid_pair(iq2s_grid, byte_of(qs, l / 2) | ((qh << (8 - l)) & 0x300));
            const uint32_t s8 = byte_of(signs, l / 2);
            r.v[l] = signed_bytes((uint32_t) gp.x, sign_mask(s8));
            r.v[l + 1] = signed_bytes((uint32_t) gp.y, sign_mask(s8 >> 4));
        }
        const float d = (float) b->d;
        r.s0 = d * (float) (2 * (b->scales[ib] & 0x0F) + 1) * 0.125f;
        r.s1 = d * (float) (2 * (b->scales[ib] >> 4) + 1) * 0.125f;
        return r;
    }
};
template<> struct Dec<20> {   // IQ4_NL: d
    static constexpr int QK = 32, BSZ = (int) sizeof(block_iq4_nl);
    static Sub get(const uint8_t* row, int sb) {
        const block_iq4_nl* b = reinterpret_cast<const block_iq4_nl*>(row) + sb;
        Sub r;
        for (int j = 0; j < 4; ++j) {
            const Int2 t = get_int_from_table_16(get_int_b2(b->qs, j), kvalues_iq4nl);
            r.v[j] = (uint32_t) t.x;
            r.v[4 + j] = (uint32_t) t.y;
        }
        r.s0 = r.s1 = (float) b->d;
        return r;
    }
};
template<> struct Dec<23> {   // IQ4_XS: d (ls - 32)
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq4_xs);
    static Sub get(const uint8_t* row, int sb) {
        const block_iq4_xs* b = reinterpret_cast<const block_iq4_xs*>(row) + sb / 8;
        const int ib = sb % 8;
        Sub r;
        for (int j = 0; j < 4; ++j) {
            const Int2 t = get_int_from_table_16(get_int_b4(b->qs, 4 * ib + j), kvalues_iq4nl);
            r.v[j] = (uint32_t) t.x;
            r.v[4 + j] = (uint32_t) t.y;
        }
        const int ls = (((b->scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F) | (((b->scales_h >> (2 * ib)) & 0x03) << 4)) - 32;
        r.s0 = r.s1 = (float) b->d * (float) ls;
        return r;
    }
};
template<> struct Dec<42> {   // Q2_0: d (code - 1), 64 values a block
    static constexpr int QK = 64, BSZ = (int) sizeof(block_q2_0);
    static Sub get(const uint8_t* row, int sb) {
        const block_q2_0* b = reinterpret_cast<const block_q2_0*>(row) + sb / 2;
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(b->qs) + (ptrdiff_t) (sb % 2) * 4;
        Sub r;
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = qs[j];
            const int k = 2 * j;
            r.v[k] = q2_0_bytes(q & 0xFFu);
            r.v[k + 1] = q2_0_bytes(q >> 8);
        }
        r.s0 = r.s1 = (float) b->d;
        return r;
    }
};

template<int TY>
constexpr bool half_scales() { return TY == 17 || TY == 22; }

// ---------------------------------------------------------------- the products
template<typename T>
inline auto local_as(sycl::multi_ptr<uint8_t, sycl::access::address_space::local_space, sycl::access::decorated::no> p) {
    using V = sycl::multi_ptr<void, sycl::access::address_space::local_space, sycl::access::decorated::no>;
    return static_cast<sycl::multi_ptr<T, sycl::access::address_space::local_space, sycl::access::decorated::no>>(V(p));
}
constexpr int up128(int b) { return (b + 127) / 128 * 128; }

// An int32 sum below 2^22 in magnitude (a block's: 32 products of int8 values) as a float, exactly: its bits added to
// those of 1.5 * 2^23, less that.  Two full-rate instructions instead of a conversion (a quarter of the rate on the
// RTX 4070, where it was a top stall).
[[maybe_unused]] inline float exact_float(int32_t c) { return sycl::bit_cast<float>(c + 0x4B400000) - 12582912.0f; }

// SGM x 2 tiles a sub-group, NSGM x 4 sub-groups a work-group: WM rows by 128 outputs.  Local memory, one buffer
// carved at 128-byte boundaries (the matrix loads take 32-byte aligned tiles): the step's activations (WM x KC int8,
// row stride LDA) and weights (WN x KC, row stride LDB: B column-major), the activations' scales as [block][row] and
// the weights' as [block][half][output] (so that a tile of either loads with stride 0: the same scale down a column or
// along a row), and a tile a sub-group on its way out.
template<int TY, int SGM, int NSGM>
struct Kernel {
    static constexpr int SGN = 2, NSGN = 4, NSG = NSGM * NSGN;
    static constexpr int WM = NSGM * SGM * TM, WN = NSGN * SGN * TN;
    static constexpr bool HALF = half_scales<TY>();
    static constexpr int A_OFF = 0, B_OFF = up128(A_OFF + WM * LDA), XS_OFF = up128(B_OFF + WN * LDB),
                         WS_OFF = up128(XS_OFF + NB * WM * 4), ST_OFF = up128(WS_OFF + 2 * NB * WN * 4),
                         LOCAL = up128(ST_OFF + NSG * TM * TN * 4);
    const uint8_t* w;
    size_t expert_bytes, row_bytes;
    const int8_t* xq;
    const float* xs;
    int64_t K;
    const int32_t* bounds;
    const int32_t* ids;
    float* dst;
    int64_t ld_dst, tiles_m, tiles_n;
    sycl::local_accessor<uint8_t, 1> lm;

    void operator()(sycl::nd_item<1> it) const {
        if constexpr (!built<TY>()) {
            (void) it;
        } else {
            const int64_t g = (int64_t) it.get_group(0), per = tiles_m * tiles_n, e = g / per, r = g % per;
            const int64_t m0 = r % tiles_m * WM, n0 = r / tiles_m * WN;
            const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
            if (m0 >= rows) return;   // the whole work-group: this expert has fewer rows than the launch
            const auto sg = it.get_sub_group();
            const int sgid = (int) sg.get_group_linear_id(), lid = (int) it.get_local_id(0);
            const int lane = (int) sg.get_local_linear_id();
            constexpr int lanes = NSG * SG;
            const int ms = sgid / NSGN * SGM * TM, ns = sgid % NSGN * SGN * TN;
            const uint8_t* wx = w + (size_t) e * expert_bytes;
            const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
            const auto ap = local_as<int8_t>(base + A_OFF);
            const auto bp = local_as<int8_t>(base + B_OFF);
            const auto xsp = local_as<float>(base + XS_OFF);
            const auto wsp = local_as<float>(base + WS_OFF);
            const auto stp = local_as<float>(base + ST_OFF) + static_cast<int>(sgid * TM * TN);
            int8_t* al = ap.get();
            int8_t* bl = bp.get();
            float* xsl = xsp.get();
            float* wsl = wsp.get();

            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[SGM][SGN];
            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, TM, TN> ci[SGM][SGN];
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) mx::joint_matrix_fill(sg, acc[i][j], 0.0f);

            const int64_t kb = K / 32;   // activation scales a row
            for (int64_t k0 = 0; k0 < K; k0 += KC) {
                sycl::group_barrier(it.get_group());   // the last step's tiles are read
                for (int v = lid; v < WM * (KC / 16); v += lanes) {
                    const int rr = v / (KC / 16), c = v % (KC / 16) * 16;
                    const int64_t row = m0 + rr;
                    u32x4 val = {0u, 0u, 0u, 0u};
                    if (row < rows) val = *reinterpret_cast<const u32x4*>(xq + (row0 + row) * K + k0 + c);
                    *reinterpret_cast<u32x4*>(al + static_cast<int>(rr * LDA + c)) = val;
                }
                for (int v = lid; v < WM * NB; v += lanes) {
                    const int b = v / WM, rr = v % WM;
                    const int64_t row = m0 + rr;
                    xsl[v] = row < rows ? xs[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                }
                for (int v = lid; v < WN * NB; v += lanes) {
                    const int nn = v / NB, b = v % NB;
                    const Sub d = Dec<TY>::get(wx + (size_t) (n0 + nn) * row_bytes, (int) (k0 / 32) + b);
                    u32x4* o = reinterpret_cast<u32x4*>(bl + static_cast<int>(nn * LDB + b * 32));
                    o[0] = u32x4{d.v[0], d.v[1], d.v[2], d.v[3]};
                    o[1] = u32x4{d.v[4], d.v[5], d.v[6], d.v[7]};
                    wsl[(2 * b) * WN + nn] = d.s0;
                    wsl[(2 * b + 1) * WN + nn] = d.s1;
                }
                sycl::group_barrier(it.get_group());
                for (int b = 0; b < NB; ++b) {
                    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> xsf[SGM];
                    for (int i = 0; i < SGM; ++i)   // element (r, c): the scale of row r
                        mx::joint_matrix_load(sg, xsf[i], xsp + static_cast<int>(b * WM + ms + i * TM), 0, mx::layout::col_major);
                    for (int h = 0; h < (HALF ? 2 : 1); ++h) {
                        for (int i = 0; i < SGM; ++i)
                            for (int j = 0; j < SGN; ++j) mx::joint_matrix_fill(sg, ci[i][j], 0);
                        for (int kk = HALF ? h * TK : 0; kk < (HALF ? h * TK + TK : 32); kk += TK) {
                            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, TM, TK, mx::layout::row_major> a[SGM];
                            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, TK, TN, mx::layout::col_major> bm[SGN];
                            for (int i = 0; i < SGM; ++i)
                                mx::joint_matrix_load(sg, a[i], ap + static_cast<int>((ms + i * TM) * LDA + b * 32 + kk), LDA);
                            for (int j = 0; j < SGN; ++j)
                                mx::joint_matrix_load(sg, bm[j], bp + static_cast<int>((ns + j * TN) * LDB + b * 32 + kk), LDB);
                            for (int i = 0; i < SGM; ++i)
                                for (int j = 0; j < SGN; ++j) mx::joint_matrix_mad(sg, ci[i][j], a[i], bm[j], ci[i][j]);
                        }
                        for (int j = 0; j < SGN; ++j)
                            for (int i = 0; i < SGM; ++i) {
                                // the pair of scales (r, c): the weights' of column c along each row, times the row's
                                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> sc;
                                mx::joint_matrix_load(sg, sc, wsp + static_cast<int>((2 * b + h) * WN + ns + j * TN), 0,
                                                      mx::layout::row_major);
                                mx::joint_matrix_apply(sg, sc, xsf[i], [](float& s, float& x) { s *= x; });
                                mx::joint_matrix_apply(sg, sc, ci[i][j], [](float& s, int32_t& c) { s *= exact_float(c); });
                                mx::joint_matrix_apply(sg, acc[i][j], sc, [](float& a, float& s) { a += s; });
                            }
                    }
                }
            }
            // out through local memory, the rows past the expert's last not written
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) {
                    mx::joint_matrix_store(sg, acc[i][j], stp, TN, mx::layout::row_major);
                    sycl::group_barrier(sg);
                    const int64_t rb = m0 + ms + (int64_t) i * TM, cb = n0 + ns + (int64_t) j * TN;
                    for (int q = lane; q < TM * TN; q += SG) {
                        const int64_t row = rb + q / TN;
                        if (row < rows) dst[(int64_t) ids[row0 + row] * ld_dst + cb + q % TN] = stp[q];
                    }
                    sycl::group_barrier(sg);
                }
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>}; }
};

template<int TY, int SGM, int NSGM>
sycl::event launch(sycl::queue& q, const void* w, size_t expert_bytes, int64_t w_rows, int64_t w_cols, int n,
                   const void* xq, int64_t total_rows, const int32_t* bounds, int64_t max_rows, const int32_t* ids,
                   float* dst, int64_t ld_dst) {
    using KT = Kernel<TY, SGM, NSGM>;
    const int64_t tiles_m = (max_rows + KT::WM - 1) / KT::WM, tiles_n = w_rows / KT::WN;
    const size_t row_bytes = (size_t) (w_cols / Dec<TY>::QK) * (size_t) Dec<TY>::BSZ;
    const auto* x8 = static_cast<const int8_t*>(xq);
    const auto* xs = reinterpret_cast<const float*>(x8 + total_rows * w_cols);
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>((size_t) KT::LOCAL), h);
        const size_t wg = (size_t) KT::NSG * SG;
        h.parallel_for(sycl::nd_range<1>((size_t) n * (size_t) (tiles_m * tiles_n) * wg, wg),
                       KT{static_cast<const uint8_t*>(w), expert_bytes, row_bytes, x8, xs, w_cols, bounds, ids, dst,
                          ld_dst, tiles_m, tiles_n, lm});
    });
}

template<int TY>
sycl::event by_rows(sycl::queue& q, const void* w, size_t expert_bytes, int64_t w_rows, int64_t w_cols, int n,
                    const void* xq, int64_t total_rows, const int32_t* bounds, int64_t max_rows, const int32_t* ids,
                    float* dst, int64_t ld_dst) {
    if (max_rows <= 16)
        return launch<TY, 1, 1>(q, w, expert_bytes, w_rows, w_cols, n, xq, total_rows, bounds, max_rows, ids, dst, ld_dst);
    if (max_rows <= 32)
        return launch<TY, 2, 1>(q, w, expert_bytes, w_rows, w_cols, n, xq, total_rows, bounds, max_rows, ids, dst, ld_dst);
    return launch<TY, 2, 2>(q, w, expert_bytes, w_rows, w_cols, n, xq, total_rows, bounds, max_rows, ids, dst, ld_dst);
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strtol(v, nullptr, 10) != 0;
}

bool shape_reported(const sycl::device& d) {
    const auto sgs = d.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sgs.begin(), sgs.end(), (size_t) SG) == sgs.end()) return false;
    for (const auto& c : d.get_info<syclex::info::device::matrix_combinations>()) {
        const bool m = c.msize == (size_t) TM || (c.msize == 0 && c.max_msize >= (size_t) TM);
        const bool n = c.nsize == (size_t) TN || (c.nsize == 0 && c.max_nsize >= (size_t) TN);
        const bool k = c.ksize == (size_t) TK || (c.ksize == 0 && c.max_ksize >= (size_t) TK);
        if (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 &&
            c.ctype == mx::matrix_type::sint32 && c.dtype == mx::matrix_type::sint32 && m && n && k)
            return true;
    }
    return false;
}

// The positions a lane's int32 accumulator elements take against those of its FP32 one, walked together as the
// products walk them: 1 when every pair agrees.
struct LayoutCheck {
    sycl::local_accessor<int32_t, 1> pi;
    sycl::local_accessor<float, 1> pf;
    int* out;
    void operator()(sycl::nd_item<1> it) const {
        if constexpr (!built<0>()) {
            (void) it;
        } else {
            const auto sg = it.get_sub_group();
            const int lid = (int) it.get_local_id(0);
            for (int i = lid; i < TM * TN; i += SG) { pi[i] = i; pf[i] = (float) i; }
            sycl::group_barrier(it.get_group());
            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, TM, TN> a;
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> b;
            mx::joint_matrix_load(sg, a, pi.template get_multi_ptr<sycl::access::decorated::no>(), TN, mx::layout::row_major);
            mx::joint_matrix_load(sg, b, pf.template get_multi_ptr<sycl::access::decorated::no>(), TN, mx::layout::row_major);
            int bad = 0, n = 0;
            mx::joint_matrix_apply(sg, b, a, [&](float& f, int32_t& v) { bad |= (int) f != v; ++n; });
            bad |= n != NE;
            bad = sycl::reduce_over_group(sg, bad, sycl::bit_or<int>());
            if (lid == 0) *out = bad ? 0 : 1;
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>}; }
};

}  // namespace

bool iq_mmq_usable(sycl::queue& q) {
    static const bool ok = [&q] {
        if (env_on("STRATA_NO_XMX")) return false;
        const sycl::device d = q.get_device();
        if (!shape_reported(d)) return false;
        using Big = Kernel<16, 2, 2>;
        if (d.get_info<sycl::info::device::local_mem_size>() < (size_t) Big::LOCAL ||
            d.get_info<sycl::info::device::max_work_group_size>() < (size_t) Big::NSG * SG)
            return false;
        int* out = sycl::malloc_device<int>(1, q);
        if (out == nullptr) return false;
        int res = 0;
        try {
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<int32_t, 1> pi(sycl::range<1>((size_t) TM * TN), h);
                sycl::local_accessor<float, 1> pf(sycl::range<1>((size_t) TM * TN), h);
                h.parallel_for(sycl::nd_range<1>(SG, SG), LayoutCheck{pi, pf, out});
            });
            q.memcpy(&res, out, sizeof(int)).wait();
        } catch (const sycl::exception& e) {
            std::fprintf(stderr, "strata: the int8 expert products' layout check did not run (%s)\n", e.what());
            res = 0;
        }
        sycl::free(out, q);
        if (!res) std::fprintf(stderr, "strata: the GPU's int8 and FP32 accumulators differ in layout: the experts' "
                               "prompt products stay in FP16\n");
        return res == 1;
    }();
    return ok;
}

bool iq_mmq_type_ok(int t) {
    switch (t) {
        case 16: case 17: case 18: case 20: case 21: case 22: case 23: case 42: return true;
        default: return false;
    }
}

size_t iq_mmq_row_bytes(int t, int64_t cols) {
    switch (t) {
        case 16: return (size_t) (cols / Dec<16>::QK) * Dec<16>::BSZ;
        case 17: return (size_t) (cols / Dec<17>::QK) * Dec<17>::BSZ;
        case 18: return (size_t) (cols / Dec<18>::QK) * Dec<18>::BSZ;
        case 20: return (size_t) (cols / Dec<20>::QK) * Dec<20>::BSZ;
        case 21: return (size_t) (cols / Dec<21>::QK) * Dec<21>::BSZ;
        case 22: return (size_t) (cols / Dec<22>::QK) * Dec<22>::BSZ;
        case 23: return (size_t) (cols / Dec<23>::QK) * Dec<23>::BSZ;
        case 42: return (size_t) (cols / Dec<42>::QK) * Dec<42>::BSZ;
        default: return 0;
    }
}

size_t iq_mmq_act_bytes(int64_t rows, int64_t cols) { return (size_t) (rows * cols + rows * (cols / 32) * 4); }

sycl::event iq_mmq_quantize(sycl::queue& q, const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld,
                            int64_t rows) {
    auto* x8 = static_cast<int8_t*>(xq);
    auto* xs = reinterpret_cast<float*>(x8 + rows * cols);
    const int64_t nb = cols / 32;
    return q.parallel_for(sycl::range<1>((size_t) (rows * nb)), [=](sycl::id<1> i) {
        const int64_t r = (int64_t) i[0] / nb, b = (int64_t) i[0] % nb;
        const float* xr = x + (int64_t) (ids ? ids[r] : r) * ld + b * 32;
        float v[32], amax = 0.0f;
        for (int j = 0; j < 32; ++j) { v[j] = xr[j]; amax = sycl::fmax(amax, sycl::fabs(v[j])); }
        const float d = amax / 127.0f, id = amax > 0.0f ? 1.0f / d : 0.0f;
        uint32_t p[8];
        for (int j = 0; j < 8; ++j) {
            uint32_t u = 0;
            for (int k = 0; k < 4; ++k) u |= (uint32_t) (uint8_t) (int8_t) sycl::round(v[4 * j + k] * id) << (8 * k);
            p[j] = u;
        }
        u32x4* o = reinterpret_cast<u32x4*>(x8 + r * cols + b * 32);
        o[0] = u32x4{p[0], p[1], p[2], p[3]};
        o[1] = u32x4{p[4], p[5], p[6], p[7]};
        xs[r * nb + b] = d;
    });
}

sycl::event iq_mmq_grouped(sycl::queue& q, int t, const void* w, size_t expert_bytes, int64_t w_rows, int64_t w_cols,
                           int n, const void* xq, int64_t total_rows, const int32_t* bounds, int64_t max_rows,
                           const int32_t* ids, float* dst, int64_t ld_dst) {
#define STRATA_MMQ_CASE(TY) \
    case TY: return by_rows<TY>(q, w, expert_bytes, w_rows, w_cols, n, xq, total_rows, bounds, max_rows, ids, dst, ld_dst)
    switch (t) {
        STRATA_MMQ_CASE(16);
        STRATA_MMQ_CASE(17);
        STRATA_MMQ_CASE(18);
        STRATA_MMQ_CASE(20);
        STRATA_MMQ_CASE(21);
        STRATA_MMQ_CASE(22);
        STRATA_MMQ_CASE(23);
        STRATA_MMQ_CASE(42);
        default: return {};
    }
#undef STRATA_MMQ_CASE
}

}  // namespace strata::kernels
