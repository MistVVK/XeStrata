// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/amd_wmma.hpp - RDNA3's and RDNA4's matrix instructions (GFX11, GFX12: WMMA) for the kernels written
// on joint_matrix's 16 x 16 x 16 tiles on 32 lanes.  AMD's device compile has joint_matrix on CDNA's matrix cores only
// and the HIP runtime reports no matrix combination for RDNA, so these kernels take the instructions themselves,
// through the same calls (wmma::load, mad, fill, store) in place of joint_matrix's.
//
// The layouts (iq_mmq.cpp's int8 products take the same): lane l holds row l % 16 of A and column l % 16 of B and of
// the accumulator.  GFX12: a lane 8 of the 16 K values (l / 16 picks which half), the accumulator's rows 8 (l / 16) + e.
// GFX11: a lane all 16 K values (lanes 16-31 repeat 0-15), the accumulator's rows 2 e + l / 16.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && (defined(__GFX11__) || defined(__GFX12__))
#define STRATA_AMD_WMMA 1
#else
#define STRATA_AMD_WMMA 0
#endif

namespace strata::kernels::xe {

/// Whether the device is an RDNA3 or RDNA4 GPU (GFX11, GFX12), whose WMMA these kernels take: the architecture as SYCL
/// reports it (its values carry the version: 0x0200000000110000 is gfx1100).
inline bool amd_wmma(const sycl::device& d) {
    namespace syclex = sycl::ext::oneapi::experimental;
    if (d.get_backend() != sycl::backend::ext_oneapi_hip) return false;
    const auto a = static_cast<uint64_t>(d.get_info<syclex::info::device::architecture>());
    return a >= static_cast<uint64_t>(syclex::architecture::amd_gpu_gfx1100) && a < 0x0200000000130000ull;
}

/// Whether the device is an RDNA4 GPU (GFX12), whose WMMA takes its operands in another layout than RDNA3's (and whose
/// products take tile layouts of their own, mma_gemm.cpp).
inline bool amd_wmma_gfx12(const sycl::device& d) {
    namespace syclex = sycl::ext::oneapi::experimental;
    if (!amd_wmma(d)) return false;
    return static_cast<uint64_t>(d.get_info<syclex::info::device::architecture>()) >= 0x0200000000120000ull;
}

#if STRATA_AMD_WMMA
namespace wmma {

using fv8 = float __attribute__((ext_vector_type(8)));
#if defined(__GFX12__)
constexpr int KN = 8;      // K values a lane holds
constexpr int KOFF = 8;    // past its half's start
inline int row_of(int hi, int e) { return 8 * hi + e; }
#else
constexpr int KN = 16;
constexpr int KOFF = 0;
inline int row_of(int hi, int e) { return 2 * e + hi; }
#endif
using h16 = _Float16 __attribute__((ext_vector_type(KN)));
using s16 = short __attribute__((ext_vector_type(KN)));

// a 16-bit element type's fragment: FP16 (sycl::half) or BF16 (its bits, as the instructions take them)
template<typename E>
struct Frag {
    using V = std::conditional_t<std::is_same_v<E, sycl::half>, h16, s16>;
    V v;
};
// A (row-major) and B (column-major: column n's K values contiguous, or row-major: strided) as fragments
template<typename E> struct A : Frag<E> {};
template<typename E> struct Bc : Frag<E> {};
template<typename E> struct Br : Frag<E> {};
struct C {
    fv8 v;
};

inline int lane16(sycl::sub_group sg) { return (int) (sg.get_local_linear_id() & 15); }
inline int half_of(sycl::sub_group sg) { return (int) (sg.get_local_linear_id() >> 4); }

template<typename P> inline auto* raw(P p) {
    if constexpr (std::is_pointer_v<P>) return p;
    else return p.get();
}
template<typename E, typename T> inline auto bits(T x) {
    using U = std::conditional_t<std::is_same_v<E, sycl::half>, _Float16, short>;
    return sycl::bit_cast<U>(x);
}

// m[i][k] at p + i * ld + k (A, row-major); B column-major is the same with its columns as the rows
template<typename E, typename F, typename P>
inline void load_rows(sycl::sub_group sg, F& f, P p, size_t ld) {
    const auto* s = raw(p) + (size_t) lane16(sg) * ld + (size_t) (half_of(sg) * KOFF);
    for (int e = 0; e < KN; ++e) f.v[e] = bits<E>(s[e]);
}
template<typename E, typename P> inline void load(sycl::sub_group sg, A<E>& f, P p, size_t ld) { load_rows<E>(sg, f, p, ld); }
template<typename E, typename P> inline void load(sycl::sub_group sg, Bc<E>& f, P p, size_t ld) { load_rows<E>(sg, f, p, ld); }
// B row-major: m[k][n] at p + k * ld + n
template<typename E, typename P>
inline void load(sycl::sub_group sg, Br<E>& f, P p, size_t ld) {
    const auto* s = raw(p) + (size_t) (half_of(sg) * KOFF) * ld + (size_t) lane16(sg);
    for (int e = 0; e < KN; ++e) f.v[e] = bits<E>(s[(size_t) e * ld]);
}

inline void fill(sycl::sub_group, C& c, float x) { c.v = fv8{x, x, x, x, x, x, x, x}; }

template<typename E, typename B>
inline void mad(sycl::sub_group, C& d, const A<E>& a, const B& b, const C& c) {
    if constexpr (std::is_same_v<E, sycl::half>) {
#if defined(__GFX12__)
        d.v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a.v, b.v, c.v);
#else
        d.v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a.v, b.v, c.v);
#endif
    } else {
#if defined(__GFX12__)
        d.v = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a.v, b.v, c.v);
#else
        d.v = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a.v, b.v, c.v);
#endif
    }
}

// int8 (iq_mmq.cpp's products, the image engine's int8 attention): a lane's KN of A's row or B's column as bytes in
// 32-bit words, the accumulator's eight int32 elements in the FP32 one's places
using s8f = int __attribute__((ext_vector_type(KN / 4)));
using iv8 = int __attribute__((ext_vector_type(8)));
/// A . B + c, both operands signed int8
inline iv8 mad_s8(const s8f& a, const s8f& b, const iv8& c) {
#if defined(__GFX12__)
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#endif
}

// row-major: the accumulator's element (i, j) to p + i * ld + j
template<typename P>
inline void store(sycl::sub_group sg, const C& c, P p, size_t ld) {
    auto* d = raw(p);
    const int lo = lane16(sg), hi = half_of(sg);
    for (int e = 0; e < 8; ++e) d[(size_t) row_of(hi, e) * ld + (size_t) lo] = c.v[e];
}

}  // namespace wmma
#endif

}  // namespace strata::kernels::xe
