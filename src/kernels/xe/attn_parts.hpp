// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/attn_parts.hpp - the parts the attention kernels share (flash_attn.cpp, src/image/qi_attn.cpp): the
// FP16 matrix engines' tiles in one form (joint_matrix's, RDNA's WMMA instructions), the form a device takes, and the
// K/V cache's formats read as FP16.
#pragma once

#include "amd_wmma.hpp"
#include "device_caps.hpp"
#include "device_target.hpp"
#include "kv_tcq_parts.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdlib>
#include <type_traits>

namespace strata::kernels::xe::attn {

namespace mx = sycl::ext::oneapi::experimental::matrix;

// a local accessor's element `off` (row-major) as the multi_ptr joint_matrix loads from and stores to
template<typename A>
inline auto at(const A& a, size_t off) {
    return a.template get_multi_ptr<sycl::access::decorated::no>() + off;
}

// joint_matrix's FP16 tiles of TM x TN x TK on SG lanes, FP32 accumulators
template<int TM_, int TN_, int TK_, int SG_>
struct Jm {
    static constexpr int TM = TM_, TN = TN_, TK = TK_, SG = SG_;
    using A = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major>;
    using Bc = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::col_major>;
    using Br = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major>;
    using C = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN>;
    template<typename M, typename P> static void load(sycl::sub_group sg, M& m, P p, size_t ld) {
        if constexpr (std::is_pointer_v<P>)   // FP16 bits in global memory
            mx::joint_matrix_load(sg, m,
                                  sycl::address_space_cast<sycl::access::address_space::global_space,
                                                           sycl::access::decorated::no>(
                                      const_cast<sycl::half*>(reinterpret_cast<const sycl::half*>(p))),
                                  ld);
        else
            mx::joint_matrix_load(sg, m, p, ld);
    }
    static void zero(sycl::sub_group sg, C& c) { mx::joint_matrix_fill(sg, c, 0.0f); }
    template<typename B> static void mad(sycl::sub_group sg, C& c, A& a, B& b) { mx::joint_matrix_mad(sg, c, a, b, c); }
    template<typename P> static void store(sycl::sub_group sg, C& c, P p, size_t ld) {
        mx::joint_matrix_store(sg, c, p, ld, mx::layout::row_major);
    }
    template<typename P> static void load_acc(sycl::sub_group sg, C& c, P p, size_t ld) {
        mx::joint_matrix_load(sg, c, p, ld, mx::layout::row_major);
    }
};

#if STRATA_AMD_WMMA
// RDNA3's and RDNA4's WMMA instructions in the same form (amd_wmma.hpp)
struct Wm {
    static constexpr int TM = 16, TN = 16, TK = 16, SG = 32;
    using A = wmma::A<sycl::half>;
    using Bc = wmma::Bc<sycl::half>;
    using Br = wmma::Br<sycl::half>;
    using C = wmma::C;
    template<typename M, typename P> static void load(sycl::sub_group sg, M& m, P p, size_t ld) {
        wmma::load<sycl::half>(sg, m, p, ld);
    }
    static void zero(sycl::sub_group sg, C& c) { wmma::fill(sg, c, 0.0f); }
    template<typename B> static void mad(sycl::sub_group sg, C& c, A& a, B& b) { wmma::mad(sg, c, a, b, c); }
    template<typename P> static void store(sycl::sub_group sg, C& c, P p, size_t ld) { wmma::store(sg, c, p, ld); }
    template<typename P> static void load_acc(sycl::sub_group sg, C& c, P p, size_t ld) {
        const int lo = (int) (sg.get_local_linear_id() & 15), hi = (int) (sg.get_local_linear_id() >> 4);
        for (int e = 0; e < 8; ++e) c.v[e] = p[(size_t) wmma::row_of(hi, e) * ld + lo];
    }
};
#else
using Wm = Jm<16, 16, 16, 32>;   // the host's and the other devices' view: the shape the kernel is launched with
#endif

// The forms, as path() numbers them: 0 Xe2's 8 x 16 x 16 on 16 lanes, 1 Alchemist's 8 x 8 x 16 on 8, 2 NVIDIA's
// 16 x 16 x 16 on 32, 3 RDNA's WMMA
template<int K> struct Path;
template<> struct Path<0> { using Ops = Jm<8, 16, 16, 16>; };
template<> struct Path<1> { using Ops = Jm<8, 8, 16, 8>; };
template<> struct Path<2> { using Ops = Jm<16, 16, 16, 32>; };
template<> struct Path<3> { using Ops = Wm; };

// the forms a device compile carries: SPIR-V Intel's, NVPTX NVIDIA's (the FP16 tensor cores from sm_70 on: Pascal's
// code has none, and its GPUs report no shape), AMDGCN RDNA's WMMA
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
[[maybe_unused]] constexpr bool built(int p) { return p == 2 && (STRATA_NV_ARCH == 0 || STRATA_NV_ARCH >= 700); }
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
[[maybe_unused]] constexpr bool built(int p) { return p == 3 && STRATA_AMD_WMMA; }
#else
[[maybe_unused]] constexpr bool built(int p) { return p == 0 || p == 1; }
#endif

// The form the device takes, from its matrix combinations and sub-group sizes; -1: none (no FP16 matrix engines, or
// STRATA_NO_XMX=1)
inline int path(const sycl::device& d) {
    if (no_xmx()) return -1;
    if (has_form(d, MatrixForm::wmma)) return 3;
    struct S { int m, n, k, sg; };
    const S shapes[] = {{8, 16, 16, 16}, {8, 8, 16, 8}, {16, 16, 16, 32}};
    for (int i = 0; i < 3; ++i) {
        const S& s = shapes[i];
        if (!has_sub_group(d, (size_t) s.sg)) continue;
        for (const auto& c : matrix_combinations(d)) {
            const bool m = c.msize == (size_t) s.m || (c.msize == 0 && c.max_msize >= (size_t) s.m);
            const bool n = c.nsize == (size_t) s.n || (c.nsize == 0 && c.max_nsize >= (size_t) s.n);
            const bool k = c.ksize == (size_t) s.k || (c.ksize == 0 && c.max_ksize >= (size_t) s.k);
            if (c.atype == mx::matrix_type::fp16 && c.btype == mx::matrix_type::fp16 &&
                c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 && m && n && k)
                return i;
        }
    }
    return -1;
}

// ---- the K/V cache's formats (qsa_decode_attn.hpp's QsaAttnPools) read as FP16

// K8V4: INT8 K, Q4_0 V; TCQ: trellis-coded K and V (kv_tcq.hpp), which only the gathers below read (the matrix
// engines' and the int8 kernels take the other formats)
enum PoolFormat : int { kFmtF16 = 0, kFmtI8 = 1, kFmtQ4 = 2, kFmtK8V4 = 3, kFmtTcq = 4 };
inline constexpr const char* kFmtNames[5] = {"FP16", "INT8", "Q4_0", "K8V4", "TCQ"};

inline int pool_format(const QsaAttnPools& p) {
    if (p.k_tc != nullptr) return p.v_tc != nullptr && (p.tc_bits == 4 || p.tc_bits == 5) ? kFmtTcq : -1;
    if (p.k_q4 != nullptr) return p.v_q4 != nullptr ? kFmtQ4 : -1;
    if (p.v_q4 != nullptr) return p.k_q != nullptr && p.k_scale != nullptr ? kFmtK8V4 : -1;
    if (p.k_q != nullptr) return p.v_q != nullptr && p.k_scale != nullptr && p.v_scale != nullptr ? kFmtI8 : -1;
    return p.k_pool != nullptr && p.v_pool != nullptr ? kFmtF16 : -1;
}

using h8 = sycl::vec<uint16_t, 8>;
using u32x2 = uint32_t __attribute__((ext_vector_type(2)));   // one 8- or 16-byte access (sycl::vec is split on NVPTX)
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

// 8 int8 codes (two words) as 8 FP16 halves (four words of two), two at a time: code + 128 in the low mantissa bits
// of 1024 (0x6400) is exactly 1152 + code, and 1152 comes off in one two-half subtraction.  [[maybe_unused]]: only
// the device code calls it.
[[maybe_unused]] inline u32x4 i8x8_to_h8(u32x2 w) {
    u32x4 r = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        const uint32_t x = w[k / 2] >> (16 * (k % 2));
        const uint32_t b = (((x & 0xFFu) | ((x & 0xFF00u) << 8)) ^ 0x00800080u) | 0x64006400u;
        const sycl::vec<sycl::half, 2> h =
            sycl::bit_cast<sycl::vec<sycl::half, 2>>(b) - sycl::vec<sycl::half, 2>(sycl::half(1152.0f));
        r[k] = sycl::bit_cast<uint32_t>(h);
    }
    return r;
}

// 8 values of an INT8 pool row (codes with an FP16 scale per 64 values) from element dd on, as FP16 bits: the codes
// are exact in FP16 and so is their product with the scale before its one rounding.  Intel's GPUs convert them in one
// instruction; elsewhere i8x8_to_h8's mantissa trick (the convert was 10% slower on Xe, flash_attn.cpp's selection form)
template<int HD>
inline h8 i8_row8(const int8_t* codes, const uint16_t* scales, int64_t row, int dd) {
    const sycl::half sc = sycl::bit_cast<sycl::half>(scales[row * (HD / 64) + dd / 64]);
#if STRATA_DEVICE_NOT_INTEL
    const u32x4 w = i8x8_to_h8(*reinterpret_cast<const u32x2*>(codes + row * HD + dd));
    const sycl::vec<sycl::half, 8> c =
        sycl::bit_cast<sycl::vec<sycl::half, 8>>(sycl::vec<uint32_t, 4>(w[0], w[1], w[2], w[3]));
#else
    const sycl::vec<sycl::half, 8> c =
        (*reinterpret_cast<const sycl::vec<int8_t, 8>*>(codes + row * HD + dd)).template convert<sycl::half>();
#endif
    const sycl::vec<sycl::half, 8> v = c * sycl::vec<sycl::half, 8>(sc);
    return v.template as<h8>();
}

// 8 values of a Q4_0 pool row (kv_q4.hpp) from element dd on, as FP16 bits: element j of a block in the low nibble of
// byte j, j + 16 in the high one, (code - 8) x d.  A block and its codes start at even bytes: four 2-byte words hold
// the eight bytes, two codes in each
template<int HD>
inline h8 q4_row8(const uint8_t* q4, int64_t row, int dd) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(q4 + row * (int64_t) (HD / QK4_0 * sizeof(block_q4_0)) +
                                                          (dd / QK4_0) * sizeof(block_q4_0));
    const sycl::half d = sycl::bit_cast<sycl::half>(w[0]);
    const int j0 = dd % QK4_0;
    const uint16_t* c = w + 1 + (j0 & 15) / 2;
    sycl::vec<uint16_t, 4> u(c[0], c[1], c[2], c[3]);
    u = (j0 >= 16 ? u >> 4 : u) & (uint16_t) 0x0F0F;   // each byte its code
    const sycl::vec<uint8_t, 8> codes = u.template as<sycl::vec<uint8_t, 8>>();
    const sycl::vec<sycl::half, 8> v = (codes.template convert<sycl::half>() - (sycl::half) 8.0f) * d;
    return v.template as<h8>();
}

// 8 values of a trellis-coded pool row (kv_tcq.hpp) from element dd on, as FP16 bits
template<int HD>
inline h8 tcq_row8(const uint8_t* pool, int bits, int64_t row, int dd) {
    float v[8];
    tcq::values<8>(pool + row * (int64_t) tcq::row_bytes(HD, bits), HD, bits, dd, v);
    h8 r;
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = sycl::bit_cast<uint16_t>(sycl::half(v[e]));
    return r;
}

// ... a whole block of 32 of a Q4_0 row into dst (4 x 8 values): one scale and 16 bytes read for 32 values
template<int HD>
inline void q4_block32(const uint8_t* q4, int64_t row, int b, sycl::half* dst) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(q4 + row * (int64_t) (HD / QK4_0 * sizeof(block_q4_0)) +
                                                          b * sizeof(block_q4_0));
    const sycl::half d = sycl::bit_cast<sycl::half>(w[0]);
    sycl::vec<uint16_t, 8> u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) u[i] = w[1 + i];
    const sycl::vec<uint16_t, 4> lo0(u[0], u[1], u[2], u[3]), lo1(u[4], u[5], u[6], u[7]);
    const sycl::vec<uint16_t, 4> parts[4] = {lo0 & (uint16_t) 0x0F0F, lo1 & (uint16_t) 0x0F0F,
                                             (lo0 >> 4) & (uint16_t) 0x0F0F, (lo1 >> 4) & (uint16_t) 0x0F0F};
    #pragma unroll
    for (int k = 0; k < 4; ++k) {   // elements 0-7, 8-15 (low nibbles), 16-23, 24-31 (high ones)
        const sycl::vec<uint8_t, 8> codes = parts[k].template as<sycl::vec<uint8_t, 8>>();
        const sycl::vec<sycl::half, 8> v = (codes.template convert<sycl::half>() - (sycl::half) 8.0f) * d;
        *reinterpret_cast<h8*>(dst + (size_t) k * 8) = v.template as<h8>();
    }
}

}  // namespace strata::kernels::xe::attn
