// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/prefill/gdn_tc.cpp - the prompt path's DeltaNet recurrence as a chunked scan on the matrix engines
// (STRATA_GDN_TC=1, gdn_tc.hpp).  The scheme of upstream's src/prefill/gdn_tc.cuh (8c2a84694, PR #3779), written on
// joint_matrix and RDNA's WMMA instead of sm_80's mma.sync.
//
// A chunk of GCH tokens of one value head, with S0 its state at the chunk's start, gamma_i the running sum of the
// gates (log decay) and e_i = exp(gamma_i) (the WY form of the delta rule; derivation as gdn_chunk_scan_kernel's in
// upstream's kernels.cu):
//   A[t][i] = beta_t exp(gamma_t - gamma_i) k_t.k_i (i < t),  T = (I + A)^-1,  P[t][i] = exp(gamma_t - gamma_i) q_t.k_i
//   (i <= t), Y_i = beta_i (v_i - e_i S0^T k_i), D = T Y, o_t = e_t S0^T q_t + sum_i P[t][i] D_i,
//   S = e_L S0 + sum_i exp(gamma_L - gamma_i) k_i D_i^T.
// Two kernels a super-block of chunks: the prep (FP32, no matrix engines; a work-group a chunk and key head) works out
// A, T and P and writes what the scan reads as FP16 images; the scan (a sub-group a value head's TM value columns,
// which nothing couples to the other sub-groups) walks the chunks in order with its slice of S^T in the accumulators,
// every product on the matrix engines, transposed so that the state is the A operand and the accumulator:
//   X^T = S0^T K^T, QS^T = S0^T Q^T;  D^T = Y^T T^T;  o^T = e_t QS^T + D^T P^T;  S^T = e_L S0^T + (f_i D^T) K.
// The images and the per-token scalars are read from global memory (the caches): the four products of a sub-group
// share them with the other sub-groups of the head, and local memory holds only what each sub-group splits itself.
//
// Numerics (upstream's): an FP16 operand is split x = hi + lo, hi = f16(x), lo = f16(x - hi), and a product takes
// lo.hi + hi.lo + hi.hi with FP32 accumulation, dropping lo.lo: within 3 x 2^-22 |a||b| of the exact product while
// both parts are normal.  So each operand is scaled by a power of two that puts its largest entry in [2^14, 2^15): k,
// q and P (|entries| <= 1) by 2^8, the state, Y and D by the sub-group's largest entry each chunk, T by its own
// (the prep's); the powers come out of the accumulators exactly.  Not the bits of kernels.cpp's recurrence kernels.
//
// The matrix calls are joint_matrix's where the device reports the shape (Intel's XMX: 8 x 16 x 16 on 16 lanes, or
// 8 x 8 x 16 on 8; NVIDIA's tensor cores: 16 x 16 x 16 on 32), or RDNA3 and RDNA4's WMMA instructions (amd_wmma.hpp).
// A GPU with none of them, or STRATA_NO_XMX=1, keeps kernels.cpp's kernels (gdn_rec_tc returns false).
#include "gdn_tc.hpp"

#include "strata/core/per_device.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "../kernels/xe/amd_wmma.hpp"
#include "../kernels/xe/attn_parts.hpp"
#include "../kernels/xe/device_caps.hpp"
#include "../kernels/xe/device_target.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU prefill_gdn_tc
#include "strata/kernels/device_arch_tag.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace strata::prefill {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;

constexpr int S = 128;            // a head's width
constexpr int GCH = 32;           // tokens a chunk (the prep's solve is a lane a token)
constexpr int SHLD = S + 8;       // the split state's rows in local memory (halves; 16-byte multiples)
constexpr int YLD = GCH + 8;      // the split Y / D rows (halves)
constexpr int SLD = S + 4;        // the state's FP32 rows (floats)
constexpr int XLD = GCH + 4;      // X^T, QS^T, D^T's FP32 rows (floats)

template<typename A>
inline auto at(const A& a, size_t off) {
    return a.template get_multi_ptr<sycl::access::decorated::no>() + off;
}

// 2^e for |e| < 127, and the power of two that puts m = max |x| >= 0 in [2^14, 2^15): FP16's hi never overflows and
// the lo of every entry within 2^-17 of the largest stays normal; clamped so that the compensating powers stay far
// from FP32's range (upstream's gtc_pow2, gtc_scale_exp)
[[maybe_unused]] inline float pow2(int e) { return sycl::bit_cast<float>((e + 127) << 23); }
[[maybe_unused]] inline int scale_exp(float m) {
    if (m == 0.0f) return 0;
    const int e = 14 - ((sycl::bit_cast<int>(m) >> 23) - 127);
    return e < -40 ? -40 : (e > 40 ? 40 : e);
}
// x as FP16 hi + lo.  hi goes back to FP32 through integer operations (as flash_attn.cpp's selection form's split): written as
// x - (float) half(x), the compiler may take the round trip for exact and make lo zero
inline void split(float x, sycl::half& hi, sycl::half& lo) {
    hi = sycl::half(x);
    lo = sycl::half(x - kernels::f32_from_f16(sycl::bit_cast<uint16_t>(hi)));
}
[[maybe_unused]] inline void split16(float x, uint16_t& hi, uint16_t& lo) {
    sycl::half h, l;
    split(x, h, l);
    hi = sycl::bit_cast<uint16_t>(h);
    lo = sycl::bit_cast<uint16_t>(l);
}

// joint_matrix's tiles of TM x TN x TK on SG lanes (as qi_attn.cpp's)
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
    template<typename P> static void load_acc(sycl::sub_group sg, C& c, P p, size_t ld) {
        mx::joint_matrix_load(sg, c, p, ld, mx::layout::row_major);
    }
    static void zero(sycl::sub_group sg, C& c) { mx::joint_matrix_fill(sg, c, 0.0f); }
    static void scale(sycl::sub_group sg, C& c, float f) { mx::joint_matrix_apply(sg, c, [=](float& x) { x *= f; }); }
    template<typename B> static void mad(sycl::sub_group sg, C& c, A& a, B& b) { mx::joint_matrix_mad(sg, c, a, b, c); }
    template<typename P> static void store(sycl::sub_group sg, C& c, P p, size_t ld) {
        mx::joint_matrix_store(sg, c, p, ld, mx::layout::row_major);
    }
};

#if STRATA_AMD_WMMA
namespace w = strata::kernels::xe::wmma;
struct Wm {
    static constexpr int TM = 16, TN = 16, TK = 16, SG = 32;
    using A = w::A<sycl::half>;
    using Bc = w::Bc<sycl::half>;
    using Br = w::Br<sycl::half>;
    using C = w::C;
    template<typename M, typename P> static void load(sycl::sub_group sg, M& m, P p, size_t ld) {
        w::load<sycl::half>(sg, m, p, ld);
    }
    template<typename P> static void load_acc(sycl::sub_group sg, C& c, P p, size_t ld) {
        const auto* s = w::raw(p);
        const int lo = w::lane16(sg), hi = w::half_of(sg);
        for (int e = 0; e < 8; ++e) c.v[e] = s[(size_t) w::row_of(hi, e) * ld + (size_t) lo];
    }
    static void zero(sycl::sub_group sg, C& c) { w::fill(sg, c, 0.0f); }
    static void scale(sycl::sub_group, C& c, float f) { c.v *= f; }
    template<typename B> static void mad(sycl::sub_group sg, C& c, A& a, B& b) { w::mad(sg, c, a, b, c); }
    template<typename P> static void store(sycl::sub_group sg, C& c, P p, size_t ld) { w::store(sg, c, p, ld); }
};
#else
using Wm = Jm<16, 16, 16, 32>;   // the host's and the other devices' view: the shape the kernel is launched with
#endif

// the instances a device compile carries: SPIR-V Intel's shapes, NVPTX NVIDIA's (the FP16 tensor cores from sm_70 on),
// AMDGCN RDNA's WMMA
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
[[maybe_unused]] constexpr bool built(int p) { return p == 2 && (STRATA_NV_ARCH == 0 || STRATA_NV_ARCH >= 700); }
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
[[maybe_unused]] constexpr bool built(int p) { return p == 3 && STRATA_AMD_WMMA; }
#else
[[maybe_unused]] constexpr bool built(int p) { return p == 0 || p == 1; }
#endif

template<int K> struct Path;   // the tiles of a scan instance: 0 Xe2 (8x16x16/16), 1 Alchemist, 2 NVIDIA, 3 RDNA
template<> struct Path<0> { using Ops = Jm<8, 16, 16, 16>; };
template<> struct Path<1> { using Ops = Jm<8, 8, 16, 8>; };
template<> struct Path<2> { using Ops = Jm<16, 16, 16, 32>; };
template<> struct Path<3> { using Ops = Wm; };

// The scratch of a super-block of NB chunks: the k and q images x 2^8 [chunk][key head][k hi, k lo, q hi, q lo][GCH][S]
// (FP16 bits), T's images x 2^et and P's x 2^8 [chunk][value head][hi, lo][GCH][GCH], the per-token scalars (beta,
// exp(gamma), exp(gamma_L - gamma), gamma) [chunk][value head][GCH], and T's scales 2^-et [chunk][value head]; every
// part a multiple of 32 bytes, in this order
template<int HK, int HV>
struct Scratch {
    uint16_t* kq;
    uint16_t* ti;
    uint16_t* pi;
    sycl::float4* sc;
    float* te;
    static size_t kq_bytes(int nb) { return (size_t) nb * HK * 4 * GCH * S * 2; }
    static size_t tp_bytes(int nb) { return (size_t) nb * HV * 2 * GCH * GCH * 2; }
    static size_t sc_bytes(int nb) { return (size_t) nb * HV * GCH * 16; }
    static size_t bytes(int nb) { return kq_bytes(nb) + 2 * tp_bytes(nb) + sc_bytes(nb) + (size_t) nb * HV * 4; }
    static Scratch carve(uint8_t* p, int nb) {
        Scratch s;
        s.kq = reinterpret_cast<uint16_t*>(p);
        s.ti = reinterpret_cast<uint16_t*>(p + kq_bytes(nb));
        s.pi = reinterpret_cast<uint16_t*>(p + kq_bytes(nb) + tp_bytes(nb));
        s.sc = reinterpret_cast<sycl::float4*>(p + kq_bytes(nb) + 2 * tp_bytes(nb));
        s.te = reinterpret_cast<float*>(p + kq_bytes(nb) + 2 * tp_bytes(nb) + sc_bytes(nb));
        return s;
    }
};

// The prep, a work-group of 8 sub-groups of 32 a (chunk, key head): the k and q rows of the chunk (zeros past its end)
// into local memory and their images to the scratch, the Gram products k_t.k_i and q_t.k_i (i <= t) in FP32, then for
// each of the key head's VPK value heads one sub-group A, T = (I + A)^-1 by forward substitution (lane i keeps column
// i of T in registers) and T's images, another P's images and the token scalars.  Past the chunk's end gate and beta
// are 0: those tokens change nothing.
template<int HK, int HV>
struct GdnTcPrep {
    static constexpr int C = 2 * HK * S + HV * S, VPK = HV / HK, NT = 256;
    static constexpr int KF = S + 1, GP = GCH + 1;   // the FP32 rows in local memory: lanes' rows in distinct banks
    static constexpr int QO = GCH * KF, KKO = 2 * GCH * KF, QKO = KKO + GCH * GP;
    static constexpr size_t LOCAL = (size_t) 2 * GCH * KF + (size_t) 2 * GCH * GP;   // floats
    static_assert(2 * VPK <= NT / 32 && VPK * GCH * GP <= GCH * KF, "the prep's sub-groups and A over the k rows");
    const float* h;
    const float* gate;
    const float* beta;
    Scratch<HK, HV> sc;
    int64_t t0;
    int64_t T;
    sycl::local_accessor<float, 1> lm;

    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<32>}; }

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int grp = (int) it.get_group(0), ch = grp / HK, qh = grp % HK, tid = (int) it.get_local_id(0);
        const int lane = (int) sg.get_local_linear_id(), wg = (int) sg.get_group_linear_id();
        const int64_t tb = t0 + (int64_t) ch * GCH;
        const int n = (int) sycl::min((int64_t) GCH, T - tb);
        // the images: each one pointer stepped from the last (AGENTS.md, IGC's GEP simplification)
        uint16_t* ikh = sc.kq + ((size_t) ch * HK + qh) * 4 * GCH * S;
        uint16_t* ikl = ikh + (size_t) GCH * S;
        uint16_t* iqh = ikl + (size_t) GCH * S;
        uint16_t* iql = iqh + (size_t) GCH * S;
        for (int e = tid; e < GCH * S; e += NT) {
            const int t = e / S, r = e % S;
            const float* ht = h + (tb + t) * C + (int64_t) (qh * S) + r;
            const float kv = t < n ? ht[(size_t) HK * S] : 0.0f;
            const float qv = t < n ? ht[0] : 0.0f;
            lm[t * KF + r] = kv;
            lm[QO + t * KF + r] = qv;
            split16(kv * 256.0f, ikh[e], ikl[e]);
            split16(qv * 256.0f, iqh[e], iql[e]);
        }
        sycl::group_barrier(it.get_group());
        for (int p = tid; p < GCH * GCH; p += NT) {
            const int t = p / GCH, i = p % GCH;
            float a[4] = {0.0f, 0.0f, 0.0f, 0.0f}, b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (i <= t)
                for (int r = 0; r < S; r += 4)
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const float ki = lm[i * KF + r + u];
                        a[u] = sycl::fma(lm[t * KF + r + u], ki, a[u]);
                        b[u] = sycl::fma(lm[QO + t * KF + r + u], ki, b[u]);
                    }
            lm[KKO + t * GP + i] = (a[0] + a[1]) + (a[2] + a[3]);
            lm[QKO + t * GP + i] = (b[0] + b[1]) + (b[2] + b[3]);
        }
        sycl::group_barrier(it.get_group());
        if (wg >= 2 * VPK) return;   // no work-group barrier below
        const int j = wg % VPK, vh = qh + j * HK, i = lane;
        const bool in = i < n;
        const float bi = in ? beta[(tb + i) * HV + vh] : 0.0f;
        const float gam = sycl::inclusive_scan_over_group(sg, in ? gate[(tb + i) * HV + vh] : 0.0f, sycl::plus<float>());
        const size_t cv = (size_t) ch * HV + vh;
        if (wg >= VPK) {   // P x 2^8 and the token scalars
            uint16_t* ph = sc.pi + cv * 2 * GCH * GCH;
            uint16_t* pl = ph + (size_t) GCH * GCH;
            for (int t = 0; t < GCH; ++t) {
                const float gt = sycl::select_from_group(sg, gam, t);
                const float p = i <= t ? sycl::exp(gt - gam) * lm[QKO + t * GP + i] * 256.0f : 0.0f;
                split16(p, ph[t * GCH + i], pl[t * GCH + i]);
            }
            const float gL = sycl::select_from_group(sg, gam, GCH - 1);
            sc.sc[cv * GCH + i] = sycl::float4(bi, sycl::exp(gam), sycl::exp(gL - gam), gam);
            return;
        }
        // A over the k rows (read by now): lane i writes column i
        const int ao = j * GCH * GP;
        for (int t = 0; t < GCH; ++t) {
            const float bt = sycl::select_from_group(sg, bi, t), gt = sycl::select_from_group(sg, gam, t);
            lm[ao + t * GP + i] = i < t ? bt * sycl::exp(gt - gam) * lm[KKO + t * GP + i] : 0.0f;
        }
        sycl::group_barrier(sg);
        // T[t][i] = -sum_{k < t} A[t][k] T[k][i] below the diagonal, 1 on it, 0 above
        float tv[GCH];
        tv[0] = i == 0 ? 1.0f : 0.0f;
#pragma unroll
        for (int t = 1; t < GCH; ++t) {
            float v = 0.0f;
#pragma unroll
            for (int k = 0; k < t; ++k) v = sycl::fma(-lm[ao + t * GP + k], tv[k], v);
            tv[t] = i < t ? v : (i == t ? 1.0f : 0.0f);
        }
        float m = 0.0f;
#pragma unroll
        for (int t = 0; t < GCH; ++t) m = sycl::fmax(m, sycl::fabs(tv[t]));
        const int et = scale_exp(sycl::reduce_over_group(sg, m, sycl::maximum<float>()));
        const float sT = pow2(et);
        uint16_t* th = sc.ti + cv * 2 * GCH * GCH;
        uint16_t* tl = th + (size_t) GCH * GCH;
#pragma unroll
        for (int t = 0; t < GCH; ++t) split16(tv[t] * sT, th[t * GCH + i], tl[t * GCH + i]);
        if (i == 0) sc.te[cv] = pow2(-et);
    }
};

// The scan: sub-group sgi of the launch takes value head sgi / CPH, its value columns c0 .. c0 + TM, and walks the
// super-block's chunks in order with that slice of S^T in its accumulators (rows c, columns the state rows r).  Each
// chunk: (a) the state through local memory into its split images x 2^es, X^T and QS^T (x 2^(8 + es)); (b) Y^T; (c)
// D^T = Y^T T^T (T's tiles above its diagonal skipped); (d) o^T from QS^T and D^T P^T (likewise), the output written;
// (e) the state x e_L plus (f_i D^T) K.  The sub-groups share nothing: a work-group is only the launch's unit.
template<int HK, int HV, int P>
struct GdnTcScan {
    using Ops = typename Path<P>::Ops;
    static constexpr int TM = Ops::TM, TN = Ops::TN, TK = Ops::TK, SG = Ops::SG;
    static constexpr int C = 2 * HK * S + HV * S, CPH = S / TM, NC = GCH / TN, NS = S / TN;
    static constexpr size_t RS = (size_t) HV * S;
    // local memory a sub-group: the state's images, two pairs of Y / D images (halves); the state, X^T (then Y^T,
    // D^T), QS^T (then o^T) (floats)
    static constexpr int H_SUB = TM * (2 * SHLD + 4 * YLD), F_SUB = TM * (SLD + 2 * XLD);
    static constexpr size_t SG_BYTES = (size_t) H_SUB * 2 + (size_t) F_SUB * 4;
    static_assert(S % TM == 0 && GCH % TN == 0 && GCH % TK == 0 && S % TN == 0 && S % TK == 0, "the tiles");
    float* state;
    const float* h;
    Scratch<HK, HV> sc;
    float* y;
    int64_t t0;
    int nch;
    int64_t T;
    sycl::local_accessor<sycl::half, 1> lh;
    sycl::local_accessor<float, 1> lf;

    // Intel's instances take the large register file: the state's S / TN accumulator tiles stay in registers
    auto get(syclex::properties_tag) const {
        if constexpr (P == 0 || P == 1)
            return syclex::properties{sycl::ext::intel::experimental::grf_size<256>,
                                      syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
        else
            return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
    }

    template<typename B>
    static void mad3(sycl::sub_group sg, typename Ops::C& c, typename Ops::A& ah, typename Ops::A& al, B& bh, B& bl) {
        Ops::mad(sg, c, al, bh);   // the small terms first
        Ops::mad(sg, c, ah, bl);
        Ops::mad(sg, c, ah, bh);
    }

    void operator()(sycl::nd_item<1> it) const {
#if defined(__SYCL_DEVICE_ONLY__)
        if constexpr (!built(P)) {
            return;   // another GPU's instance: never launched on this target
        } else {
        sycl::sub_group sg = it.get_sub_group();
        const int g = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int nsg = (int) (it.get_local_range(0) / SG);
        const int sgi = (int) it.get_group(0) * nsg + g;
        const int vh = sgi / CPH, c0 = (sgi % CPH) * TM, qhd = vh % HK;
        const size_t o_sh = (size_t) g * H_SUB, o_sl = o_sh + (size_t) TM * SHLD, o_uh = o_sl + (size_t) TM * SHLD,
                     o_ul = o_uh + (size_t) TM * YLD, o_wh = o_ul + (size_t) TM * YLD, o_wl = o_wh + (size_t) TM * YLD;
        const size_t o_st = (size_t) g * F_SUB, o_xs = o_st + (size_t) TM * SLD, o_qs = o_xs + (size_t) TM * XLD;
        float* col0 = state + (size_t) vh * S + c0;
        for (int e = lane; e < TM * S; e += SG)
            lf[o_st + (size_t) ((e % TM) * SLD) + e / TM] = col0[(size_t) (e / TM) * RS + e % TM];
        sycl::group_barrier(sg);
        typename Ops::C sacc[NS];
        for (int j = 0; j < NS; ++j) Ops::load_acc(sg, sacc[j], at(lf, o_st + (size_t) j * TN), SLD);
        const float out_scale = sycl::rsqrt((float) S);
        for (int ch = 0; ch < nch; ++ch) {
            const int64_t tb = t0 + (int64_t) ch * GCH;
            const int n = (int) sycl::min((int64_t) GCH, T - tb);
            const uint16_t* kh = sc.kq + ((size_t) ch * HK + qhd) * 4 * GCH * S;
            const uint16_t* kl = kh + (size_t) GCH * S;
            const uint16_t* qh = kl + (size_t) GCH * S;
            const uint16_t* ql = qh + (size_t) GCH * S;
            const size_t cv = (size_t) ch * HV + vh;
            const uint16_t* th = sc.ti + cv * 2 * GCH * GCH;
            const uint16_t* tl = th + (size_t) GCH * GCH;
            const uint16_t* ph = sc.pi + cv * 2 * GCH * GCH;
            const uint16_t* pl = ph + (size_t) GCH * GCH;
            const sycl::float4* s4 = sc.sc + cv * GCH;
            // (a) the state x 2^es as FP16 hi / lo; X^T = S0^T K^T and QS^T = S0^T Q^T
            for (int j = 0; j < NS; ++j) Ops::store(sg, sacc[j], at(lf, o_st + (size_t) j * TN), SLD);
            sycl::group_barrier(sg);
            float m = 0.0f;
            for (int e = lane; e < TM * S; e += SG)
                m = sycl::fmax(m, sycl::fabs(lf[o_st + (size_t) ((e / S) * SLD) + e % S]));
            const int es = scale_exp(sycl::reduce_over_group(sg, m, sycl::maximum<float>()));
            const float sA = pow2(es);
            for (int e = lane; e < TM * S; e += SG) {
                const int r = e / S, k = e % S;
                split(lf[o_st + (size_t) (r * SLD) + k] * sA, lh[o_sh + (size_t) (r * SHLD) + k],
                      lh[o_sl + (size_t) (r * SHLD) + k]);
            }
            sycl::group_barrier(sg);
            typename Ops::C xt[NC], qt[NC];
            for (int nt = 0; nt < NC; ++nt) {
                Ops::zero(sg, xt[nt]);
                Ops::zero(sg, qt[nt]);
            }
            for (int kk = 0; kk < S / TK; ++kk) {
                typename Ops::A ah, al;
                Ops::load(sg, ah, at(lh, o_sh + (size_t) kk * TK), SHLD);
                Ops::load(sg, al, at(lh, o_sl + (size_t) kk * TK), SHLD);
                for (int nt = 0; nt < NC; ++nt) {
                    const size_t o = (size_t) nt * TN * S + (size_t) kk * TK;
                    typename Ops::Bc bh, bl;
                    Ops::load(sg, bh, kh + o, S);
                    Ops::load(sg, bl, kl + o, S);
                    mad3(sg, xt[nt], ah, al, bh, bl);
                    Ops::load(sg, bh, qh + o, S);
                    Ops::load(sg, bl, ql + o, S);
                    mad3(sg, qt[nt], ah, al, bh, bl);
                }
            }
            for (int nt = 0; nt < NC; ++nt) {
                Ops::store(sg, xt[nt], at(lf, o_xs + (size_t) nt * TN), XLD);
                Ops::store(sg, qt[nt], at(lf, o_qs + (size_t) nt * TN), XLD);
            }
            sycl::group_barrier(sg);
            // (b) Y^T = beta_i (V^T - e_i X^T), X^T out of its 2^(8 + es); its images x 2^ey
            const float eS = pow2(-(8 + es));
            const float* vc = h + (size_t) 2 * HK * S + (size_t) vh * S + c0;
            float my = 0.0f;
            for (int e = lane; e < TM * GCH; e += SG) {
                const int c = e % TM, i = e / TM;
                float yv = 0.0f;
                if (i < n) {
                    const sycl::float4 si = s4[i];
                    yv = si[0] * sycl::fma(-(si[1] * eS), lf[o_xs + (size_t) (c * XLD) + i], vc[(tb + i) * C + c]);
                }
                lf[o_xs + (size_t) (c * XLD) + i] = yv;
                my = sycl::fmax(my, sycl::fabs(yv));
            }
            const int ey = scale_exp(sycl::reduce_over_group(sg, my, sycl::maximum<float>()));
            const float sY = pow2(ey);
            for (int e = lane; e < TM * GCH; e += SG) {
                const int c = e % TM, i = e / TM;
                split(lf[o_xs + (size_t) (c * XLD) + i] * sY, lh[o_uh + (size_t) (c * YLD) + i],
                      lh[o_ul + (size_t) (c * YLD) + i]);
            }
            sycl::group_barrier(sg);
            // (c) D^T = Y^T T^T: T[i][i'] is zero for i' > i
            typename Ops::C dt[NC];
            for (int nt = 0; nt < NC; ++nt) Ops::zero(sg, dt[nt]);
            for (int ks = 0; ks < GCH / TK; ++ks) {
                typename Ops::A ah, al;
                Ops::load(sg, ah, at(lh, o_uh + (size_t) ks * TK), YLD);
                Ops::load(sg, al, at(lh, o_ul + (size_t) ks * TK), YLD);
                for (int nt = 0; nt < NC; ++nt) {
                    if (ks * TK >= (nt + 1) * TN) continue;
                    const size_t o = (size_t) nt * TN * GCH + (size_t) ks * TK;
                    typename Ops::Bc bh, bl;
                    Ops::load(sg, bh, th + o, GCH);
                    Ops::load(sg, bl, tl + o, GCH);
                    mad3(sg, dt[nt], ah, al, bh, bl);
                }
            }
            for (int nt = 0; nt < NC; ++nt) Ops::store(sg, dt[nt], at(lf, o_xs + (size_t) nt * TN), XLD);
            sycl::group_barrier(sg);
            // D^T out of its 2^(ey + et); its images x 2^ed for (d), f_i D^T's for (e); QS^T to 2^(ed + 8) e_t QS^T
            const float sYT = pow2(-ey) * sc.te[cv];
            float md = 0.0f;
            for (int e = lane; e < TM * GCH; e += SG) {
                const size_t a = o_xs + (size_t) ((e / GCH) * XLD) + e % GCH;
                const float d = lf[a] * sYT;
                lf[a] = d;
                md = sycl::fmax(md, sycl::fabs(d));
            }
            const int ed = scale_exp(sycl::reduce_over_group(sg, md, sycl::maximum<float>()));
            const float sD = pow2(ed), sOi = pow2(-(ed + 8)), eO = pow2(ed - es);
            for (int e = lane; e < TM * GCH; e += SG) {
                const int c = e / GCH, i = e % GCH;
                const sycl::float4 si = s4[i];
                const float d = lf[o_xs + (size_t) (c * XLD) + i] * sD;
                split(d, lh[o_wh + (size_t) (c * YLD) + i], lh[o_wl + (size_t) (c * YLD) + i]);
                split(d * si[2], lh[o_uh + (size_t) (c * YLD) + i], lh[o_ul + (size_t) (c * YLD) + i]);
                lf[o_qs + (size_t) (c * XLD) + i] *= si[1] * eO;
            }
            sycl::group_barrier(sg);
            // (d) o^T = e_t QS^T + D^T P^T: P[t][i] is zero for i > t; out of its 2^(ed + 8), then rsqrt(S)
            typename Ops::C oacc[NC];
            for (int nt = 0; nt < NC; ++nt) Ops::load_acc(sg, oacc[nt], at(lf, o_qs + (size_t) nt * TN), XLD);
            for (int ks = 0; ks < GCH / TK; ++ks) {
                typename Ops::A ah, al;
                Ops::load(sg, ah, at(lh, o_wh + (size_t) ks * TK), YLD);
                Ops::load(sg, al, at(lh, o_wl + (size_t) ks * TK), YLD);
                for (int nt = 0; nt < NC; ++nt) {
                    if (ks * TK >= (nt + 1) * TN) continue;
                    const size_t o = (size_t) nt * TN * GCH + (size_t) ks * TK;
                    typename Ops::Bc bh, bl;
                    Ops::load(sg, bh, ph + o, GCH);
                    Ops::load(sg, bl, pl + o, GCH);
                    mad3(sg, oacc[nt], ah, al, bh, bl);
                }
            }
            for (int nt = 0; nt < NC; ++nt) Ops::store(sg, oacc[nt], at(lf, o_qs + (size_t) nt * TN), XLD);
            sycl::group_barrier(sg);
            float* yo = y + (size_t) vh * S + c0;
            for (int e = lane; e < TM * GCH; e += SG) {
                const int c = e % TM, t = e / TM;
                if (t < n) yo[(size_t) (tb + t) * RS + c] = (lf[o_qs + (size_t) (c * XLD) + t] * sOi) * out_scale;
            }
            // (e) S^T = e_L S0^T + (f_i D^T) K: the accumulators x e_L 2^(ed + 8) before (one rounding), f_i D^T x
            // 2^ed in, K x 2^8, the accumulators x 2^-(ed + 8) after
            const float sPre = s4[GCH - 1][1] * pow2(ed + 8);
            for (int j = 0; j < NS; ++j) Ops::scale(sg, sacc[j], sPre);
            for (int ks = 0; ks < GCH / TK; ++ks) {
                typename Ops::A ah, al;
                Ops::load(sg, ah, at(lh, o_uh + (size_t) ks * TK), YLD);
                Ops::load(sg, al, at(lh, o_ul + (size_t) ks * TK), YLD);
                for (int nt = 0; nt < NS; ++nt) {
                    const size_t o = (size_t) ks * TK * S + (size_t) nt * TN;
                    typename Ops::Br bh, bl;
                    Ops::load(sg, bh, kh + o, S);
                    Ops::load(sg, bl, kl + o, S);
                    mad3(sg, sacc[nt], ah, al, bh, bl);
                }
            }
            for (int j = 0; j < NS; ++j) Ops::scale(sg, sacc[j], sOi);
        }
        for (int j = 0; j < NS; ++j) Ops::store(sg, sacc[j], at(lf, o_st + (size_t) j * TN), SLD);
        sycl::group_barrier(sg);
        for (int e = lane; e < TM * S; e += SG)
            col0[(size_t) (e / TM) * RS + e % TM] = lf[o_st + (size_t) ((e % TM) * SLD) + e / TM];
        }
#else
        (void) it;
#endif
    }
};

// the scan's path for the device: 0-3 as the attention's Path<> (attn_parts.hpp), -1 none
int pick_path(const sycl::device& d) { return kernels::xe::attn::path(d); }

// what a device runs: the path, the scan's sub-groups a work-group, the chunks a super-block and its scratch
struct TcDev {
    int path = -1;
    int nsg = 0;
    int nb = 0;
    uint8_t* buf = nullptr;
};

template<int HK, int HV, int P>
void run(sycl::queue& q, const TcDev& dv, float* state, const float* h, const float* gate, const float* beta, float* y,
         int64_t T) {
    using Pr = GdnTcPrep<HK, HV>;
    using K = GdnTcScan<HK, HV, P>;
    const Scratch<HK, HV> sc = Scratch<HK, HV>::carve(dv.buf, dv.nb);
    const int64_t span = (int64_t) dv.nb * GCH;
    const int nsg = dv.nsg;
    for (int64_t t0 = 0; t0 < T; t0 += span) {   // the queue is in order: a super-block's prep waits for the last scan
        const int nch = (int) ((std::min<int64_t>(span, T - t0) + GCH - 1) / GCH);
        q.submit([&](sycl::handler& hd) {
            hd.parallel_for(sycl::nd_range<1>((size_t) nch * HK * Pr::NT, Pr::NT),
                            Pr{h, gate, beta, sc, t0, T, sycl::local_accessor<float, 1>(sycl::range<1>(Pr::LOCAL), hd)});
        });
        q.submit([&](sycl::handler& hd) {
            const size_t groups = (size_t) HV * K::CPH / (size_t) nsg;
            hd.parallel_for(sycl::nd_range<1>(groups * (size_t) nsg * K::SG, (size_t) nsg * K::SG),
                            K{state, h, sc, y, t0, nch, T,
                              sycl::local_accessor<sycl::half, 1>(sycl::range<1>((size_t) nsg * K::H_SUB), hd),
                              sycl::local_accessor<float, 1>(sycl::range<1>((size_t) nsg * K::F_SUB), hd)});
        });
    }
}

// whether the scan's instance for path P and the prep build on the device, and the most sub-groups a work-group (up
// to 4) its local memory and work-group size take; 0 when it cannot run there
template<int HK, int HV, int P>
int scan_fit(sycl::queue& q) {
    using K = GdnTcScan<HK, HV, P>;
    using Pr = GdnTcPrep<HK, HV>;
    const sycl::device d = q.get_device();
    const size_t local = d.get_info<sycl::info::device::local_mem_size>();
    const size_t wg = d.get_info<sycl::info::device::max_work_group_size>();
    if (Pr::LOCAL * 4 > local || (size_t) Pr::NT > wg || !kernels::xe::has_sub_group(d, 32))
        return 0;
    int nsg = 4;
    while (nsg > 1 && ((size_t) nsg * K::SG_BYTES > local || (size_t) nsg * K::SG > wg)) nsg /= 2;
    if (K::SG_BYTES > local || (size_t) K::SG > wg) return 0;
    try {   // the large register file and the shapes are compiled when first asked: a device that refuses says so here
        (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            q.get_context(), {d}, {sycl::get_kernel_id<K>(), sycl::get_kernel_id<Pr>()});
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xestrata: XESTRATA_GDN_TC: the GPU does not build the chunked DeltaNet scan (%s)\n", e.what());
        return 0;
    }
    return nsg;
}

template<int HK, int HV>
bool tc(sycl::queue& q, float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T) {
    static core::PerDevice<TcDev> per_device;
    const TcDev& dv = per_device.get(q.get_device(), [&] {
        TcDev r;
        const sycl::device d = q.get_device();
        const int path = pick_path(d);
        const char* why = "no FP16 matrix engines this scan is written for, or XESTRATA_NO_XMX=1";
        int nsg = 0;
        if (path >= 0) {
            nsg = path == 0 ? scan_fit<HK, HV, 0>(q) : path == 1 ? scan_fit<HK, HV, 1>(q)
                : path == 2 ? scan_fit<HK, HV, 2>(q) : scan_fit<HK, HV, 3>(q);
            if (nsg == 0) why = "too little local memory or work-group size, or the kernels did not build";
        }
        if (path >= 0 && nsg > 0) {
            // the scratch for 64 chunks (2048 tokens) a super-block, halved down to 8 while the device has no room
            const size_t most = d.get_info<sycl::info::device::max_mem_alloc_size>();
            for (int nb = 64; nb >= 8 && r.buf == nullptr; nb /= 2) {
                const size_t bytes = Scratch<HK, HV>::bytes(nb);
                if (bytes > most) continue;
                try {
                    r.buf = sycl::malloc_device<uint8_t>(bytes, q);
                } catch (const sycl::exception&) {
                    r.buf = nullptr;
                }
                if (r.buf != nullptr) r.nb = nb;
            }
            if (r.buf == nullptr) why = "no room for its scratch";
        }
        if (r.buf != nullptr) {
            r.path = path;
            r.nsg = nsg;
            const int tm = path == 1 || path == 0 ? 8 : 16, tn = path == 1 ? 8 : 16;
            std::fprintf(stderr, "xestrata: XESTRATA_GDN_TC: the DeltaNet recurrence of prompt chunks as a chunked scan on "
                         "the matrix engines (%dx%dx16 tiles, %d sub-groups a work-group, %d-token super-blocks, "
                         "%.1f MiB of scratch)\n", tm, tn, nsg, r.nb * GCH,
                         (double) Scratch<HK, HV>::bytes(r.nb) / (1024.0 * 1024.0));
        } else {
            std::fprintf(stderr, "xestrata: XESTRATA_GDN_TC: not on this GPU (%s): the recurrence kernels as before\n", why);
        }
        return r;
    });
    if (dv.path < 0) return false;
    switch (dv.path) {
    case 0: run<HK, HV, 0>(q, dv, state, h, gate, beta, y, T); break;
    case 1: run<HK, HV, 1>(q, dv, state, h, gate, beta, y, T); break;
    case 2: run<HK, HV, 2>(q, dv, state, h, gate, beta, y, T); break;
    default: run<HK, HV, 3>(q, dv, state, h, gate, beta, y, T); break;
    }
    return true;
}

}  // namespace

bool gdn_rec_tc(sycl::queue& q, int64_t h_k, int64_t h_v, float* state, const float* h, const float* gate,
                const float* beta, float* y, int64_t T) {
    if (T <= 0) return false;
    if (h_k == 16 && h_v == 48) return tc<16, 48>(q, state, h, gate, beta, y, T);
    if (h_k == 16 && h_v == 32) return tc<16, 32>(q, state, h, gate, beta, y, T);
    if (h_k == 16 && h_v == 16) return tc<16, 16>(q, state, h, gate, beta, y, T);
    return false;
}

}  // namespace strata::prefill
