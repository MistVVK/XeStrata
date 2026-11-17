// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/flash_attn_bwd.cpp - see include/strata/kernels/flash_attn_bwd.hpp.
//
// FlashAttention-2's backward on the forward's tiles (attn_parts.hpp's Path: Intel's XMX on 16 or 8 lanes, NVIDIA's
// tensor cores on 32, RDNA3's and RDNA4's WMMA on 32).  Three kernels, none with an atomic, one gradient each:
//   * dV: a sub-group takes TM keys of one head, K's tiles in registers, and walks the queries CH at a time:
//     S^T = K . Q^T on the matrix engines into its local memory, P^T = exp(S^T scale - lse) there as FP16, then
//     dV += P^T . dO in the accumulators.
//   * dK: the same with V's tiles too: dP^T = V . dO^T beside S^T, dS^T = P^T (dP^T - D) scale, dK += dS^T . Q.
//   * dQ: a sub-group takes TM queries of one head, Q's and dO's tiles in registers, and walks the keys: S and dP the
//     same way, dS as FP16, dQ += dS . K.
// A row below n_causal sees the keys up to itself, the rest every key; keys and queries past `rows` are left out.
// dO, D and so dP and dS are in the head's units of g (the caller's choice keeps |dS| within FP16's range); the
// outputs are multiplied by g on the way out.

#include "strata/kernels/flash_attn_bwd.hpp"

#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "attn_parts.hpp"
#include "device_caps.hpp"
#include "device_target.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_flash_attn_bwd
#include "strata/kernels/device_arch_tag.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace strata::kernels {

namespace {

namespace syclex = sycl::ext::oneapi::experimental;
namespace mx = sycl::ext::oneapi::experimental::matrix;
using namespace xe::attn;

enum Mode : int { kDq = 0, kDk = 1, kDv = 2 };

template<int P, int D, int M>   // P: the tiles (attn_parts.hpp's path); D: head_dim; M: the gradient made (Mode)
struct Bwd {
    using Ops = typename Path<P>::Ops;
    static constexpr bool DQ = M == kDq, DP = M != kDv;   // DP: dP (and dS) made, not P alone
    // a chunk: queries (dK, dV) or keys (dQ).  32, not 64: half the local memory a sub-group, so more of them share a
    // core (the B70's 2,304 rows of 32 heads: dQ 5.4 -> 3.1 ms, dK 5.3 -> 3.4, dV 3.4 -> 2.4 with 16 sub-groups a
    // work-group, 2026-11-09)
    static constexpr int CH = 32;
    static constexpr int TM = Ops::TM, TN = Ops::TN, TK = Ops::TK, SG = Ops::SG;
    static constexpr int NT = D / TN;              // the output's column tiles
    static constexpr int CW = CH / SG;             // a chunk's columns a lane takes
    static_assert(CH % SG == 0 && CH % TN == 0 && CH % TK == 0, "a chunk splits into the tiles and the lanes");
    static constexpr int SLD = CH + 4, PLD = CH + 8;   // local rows apart in banks
    static constexpr int SREG = std::max(TM * SLD, TM * TN);   // S, and the output a tile at a time on its way out
    // local memory a sub-group takes: S (and dP) [TM][SLD] FP32, P or dS [TM][PLD] FP16
    static constexpr size_t SG_FLOATS = (size_t) SREG + (DP ? (size_t) TM * SLD : 0);
    static constexpr size_t SG_BYTES = SG_FLOATS * 4 + (size_t) TM * PLD * 2;

    FlashAttnBwdArgs a;
    sycl::local_accessor<float, 1> st;      // per sub-group S [SREG], then dP [TM][SLD]
    sycl::local_accessor<sycl::half, 1> ph; // per sub-group P or dS [TM][PLD]

    auto get(syclex::properties_tag) const {
        if constexpr (P == 0)
            return syclex::properties{sycl::ext::intel::experimental::grf_size<256>,
                                      syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
        else
            return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
    }

    void operator()(sycl::nd_item<2> it) const {
#if defined(__SYCL_DEVICE_ONLY__)
        if constexpr (!built(P)) {
            return;   // another GPU's instance: never launched on this target
        } else {
        sycl::sub_group sg = it.get_sub_group();
        const int g = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int nsg = (int) (it.get_local_range(1) / SG);
        const int64_t h = (int64_t) it.get_group(0), H = a.n_head, R = a.rows, ld = a.ld;
        const int64_t QR = a.q_rows > 0 && a.q_rows < R ? a.q_rows : R;   // the query rows taken
        const int64_t my0 = ((int64_t) it.get_group(1) * nsg + g) * TM;   // the sub-group's keys (dK, dV) or queries
        if (my0 >= (DQ ? QR : R)) return;
        const size_t soff = (size_t) g * SG_FLOATS, doff = soff + SREG, poff = (size_t) g * TM * PLD;
        const int64_t col = h * D;
        // the sub-group's own rows' tiles in registers: Q and dO (dQ), K and V (dK), K (dV).  (Two row tiles a
        // sub-group, each chunk's tiles read once for both, ran dQ and dK 5 times slower on the B70: the registers
        // ran out)
        const uint16_t* xa = DQ ? a.q : a.k;
        const uint16_t* xb = DQ ? a.dO : a.v;
        typename Ops::A ta[D / TK], tb[DP ? D / TK : 1];
        for (int kk = 0; kk < D / TK; ++kk) {
            Ops::load(sg, ta[kk], xa + my0 * ld + col + int64_t(kk) * TK, ld);
            if constexpr (DP) Ops::load(sg, tb[kk], xb + my0 * ld + col + int64_t(kk) * TK, ld);
        }
        // the gradient's tiles: one set a kernel (dK's and dV's in one gave wrong tiles past the second on the B70:
        // IGC with 16 accumulators beside the operands, 2026-11-09)
        typename Ops::C acc[NT];
        for (int nt = 0; nt < NT; ++nt) Ops::zero(sg, acc[nt]);
        const float scale = a.scale, gh = a.g[h];
        // the chunks walked: dQ's causal rows see the keys up to their last row; the rest walk every row
        const int64_t last = std::min(R, my0 + TM) - 1;
        const int64_t end = DQ && last < a.n_causal ? last + 1 : DQ ? R : QR;
        // the other side's operands a chunk at a time: dQ reads K and V, dK Q and dO, dV Q (then dO)
        const uint16_t* ob1 = DQ ? a.k : a.q;    // with ta: S (dK and dV: S^T)
        const uint16_t* ob2 = DQ ? a.v : a.dO;   // with tb: dP (dK: dP^T)
        // the chunk's operand of the gradient's product: K (dQ), Q (dK), dO (dV)
        const uint16_t* ob3 = M == kDq ? a.k : M == kDk ? a.q : a.dO;
        // dQ's rows' lse and D, the same for every chunk
        float rl[DQ ? TM : 1], rd[DQ ? TM : 1];
        if constexpr (DQ)
            for (int r = 0; r < TM; ++r) {
                const int64_t qq = my0 + r;
                rl[r] = qq < R ? a.lse[qq * H + h] : 0.f;
                rd[r] = qq < R ? a.D[h * R + qq] : 0.f;
            }
        for (int64_t j0 = 0; j0 < end; j0 += CH) {
            if constexpr (P == 0) {
                // Intel's: the next chunk's two operands into L1 while this one is worked, in blocks of CH rows x 32
                // columns shared out over the work-group's sub-groups (they walk the same chunks), as the forward
                // does (not on the 8-wide path: its GPUs' compiler takes no joint_matrix prefetch; 2.0 -> 1.6 times
                // the forward's time for dV on the B70)
                const int64_t jn = j0 + CH;
                if (jn + CH <= end) {
                    constexpr int PR = CH, PC = 32, NB = D / PC;
                    const uint16_t* second = DP ? ob2 : ob3;
                    for (int b = g; b < 2 * NB; b += nsg) {
                        const uint16_t* base = (b < NB ? ob1 : second) + jn * ld + col + int64_t(b % NB) * PC;
                        mx::joint_matrix_prefetch<PR, PC>(
                            sg, const_cast<sycl::half*>(reinterpret_cast<const sycl::half*>(base)), (size_t) ld,
                            mx::layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
                    }
                }
            }
            for (int nt = 0; nt < CH / TN; ++nt) {
                typename Ops::C c1, c2;
                Ops::zero(sg, c1);
                if constexpr (DP) Ops::zero(sg, c2);
                for (int kk = 0; kk < D / TK; ++kk) {
                    typename Ops::Bc b1;
                    Ops::load(sg, b1, ob1 + (j0 + int64_t(nt) * TN) * ld + col + int64_t(kk) * TK, ld);
                    Ops::mad(sg, c1, ta[kk], b1);
                    if constexpr (DP) {
                        typename Ops::Bc b2;
                        Ops::load(sg, b2, ob2 + (j0 + int64_t(nt) * TN) * ld + col + int64_t(kk) * TK, ld);
                        Ops::mad(sg, c2, tb[kk], b2);
                    }
                }
                Ops::store(sg, c1, at(st, soff + (size_t) nt * TN), SLD);
                if constexpr (DP) Ops::store(sg, c2, at(st, doff + (size_t) nt * TN), SLD);
            }
            sycl::group_barrier(sg);
            // P, or dS: the element (r, c) is (key my0 + r, query j0 + c) for dK and dV, (query my0 + r, key j0 + c)
            // for dQ; a lane takes the columns lane + SG i, so the chunk's queries' lse and D (dK, dV) are read once
            // into registers, not once an element (lse's rows are H apart: a cache line each)
            float cl[CW], cd[CW];
            if constexpr (!DQ)
                for (int i = 0; i < CW; ++i) {
                    const int64_t qq = j0 + lane + int64_t(i) * SG;
                    cl[i] = qq < QR ? a.lse[qq * H + h] : 0.f;
                    cd[i] = DP && qq < QR ? a.D[h * R + qq] : 0.f;
                }
            for (int r = 0; r < TM; ++r) {
                for (int i = 0; i < CW; ++i) {
                    const int c = lane + SG * i;
                    const int64_t qq = DQ ? my0 + r : j0 + c, kq = DQ ? j0 + c : my0 + r;
                    const float l = DQ ? rl[r] : cl[i], d = DQ ? rd[r] : cd[i];
                    float x = 0.f;
                    if (qq < QR && kq < R && !(qq < a.n_causal && kq > qq)) {
                        // native::exp: P and dS go to FP16 (the forward's choice)
                        x = sycl::native::exp(st[soff + (size_t) r * SLD + c] * scale - l);
                        if constexpr (DP) x = x * (st[doff + (size_t) r * SLD + c] - d) * scale;
                    }
                    ph[poff + (size_t) r * PLD + c] = sycl::half(x);
                }
            }
            sycl::group_barrier(sg);
            // dQ += dS . K, dK += dS^T . Q, dV += P^T . dO
            for (int kc = 0; kc < CH / TK; ++kc) {
                typename Ops::A pa;
                Ops::load(sg, pa, at(ph, poff + (size_t) kc * TK), PLD);
                for (int nt = 0; nt < NT; ++nt) {
                    typename Ops::Br b;
                    Ops::load(sg, b, ob3 + (j0 + int64_t(kc) * TK) * ld + col + int64_t(nt) * TN, ld);
                    Ops::mad(sg, acc[nt], pa, b);
                }
            }
            sycl::group_barrier(sg);
        }
        // the output a tile at a time through local memory, times g: dQ written; dK and dV written or added into
        float* out = M == kDq ? a.dq : M == kDk ? a.dk : a.dv;
        const bool add = !DQ && a.accumulate;
        for (int nt = 0; nt < NT; ++nt) {
            Ops::store(sg, acc[nt], at(st, soff), TN);
            sycl::group_barrier(sg);
            for (int e = lane; e < TM * TN; e += SG) {
                const int64_t row = my0 + e / TN;
                if (row >= (DQ ? QR : R)) continue;
                float* o = out + row * (H * D) + col + int64_t(nt) * TN + e % TN;
                const float x = st[soff + e] * gh;
                *o = add ? *o + x : x;
            }
            sycl::group_barrier(sg);
        }
        }
#else
        (void) it;
#endif
    }
};

template<int P, int D, int M>
int nsg_most(const sycl::queue& q) {
    // as many sub-groups a work-group as the device takes, up to 32: they walk the same chunks, read once into the
    // core's cache for all of them (the B70, 2,304 rows: 8 sub-groups 3.7 ms for dQ, 16 3.4, 32 3.2; 2026-11-09).
    // The kernel's own largest work-group bounds it too: on CUDA its registers do (the RTX 4070's 172 a work-item
    // passed a block's 65,536 at 16 sub-groups of 32)
    using K = Bwd<P, D, M>;
    const sycl::device d = q.get_device();
    const size_t local = d.get_info<sycl::info::device::local_mem_size>();
    const size_t wg = std::min(d.get_info<sycl::info::device::max_work_group_size>(), xe::kernel_work_group<K>(q));
    int nsg = 32;
    while (nsg > 1 && ((size_t) nsg * K::SG_BYTES > local || (size_t) nsg * K::SG > wg)) nsg /= 2;
    return (size_t) nsg * K::SG_BYTES <= local && (size_t) nsg * K::SG <= wg ? nsg : 0;
}

template<int P, int D, int M>
bool launch(sycl::queue& q, const FlashAttnBwdArgs& a) {
    using K = Bwd<P, D, M>;
    static core::PerDevice<int> most;
    const int nsg = most.get(q.get_device(), [&q] { return nsg_most<P, D, M>(q); });
    if (nsg == 0) return false;
    const int64_t per = (int64_t) nsg * K::TM;
    const size_t groups = (size_t) ((a.rows + per - 1) / per);
    q.submit([&](sycl::handler& h) {
        K k{a, sycl::local_accessor<float, 1>(sycl::range<1>((size_t) nsg * K::SG_FLOATS), h),
            sycl::local_accessor<sycl::half, 1>(sycl::range<1>((size_t) nsg * K::TM * K::PLD), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) a.n_head, groups * nsg * K::SG}, {1, (size_t) nsg * K::SG}), k);
    });
    return true;
}

template<int P>
bool run(sycl::queue& q, const FlashAttnBwdArgs& a) {
    // every kernel's room first: a device that takes one but not another runs none
    if (nsg_most<P, 128, kDq>(q) == 0 || nsg_most<P, 128, kDk>(q) == 0 || nsg_most<P, 128, kDv>(q) == 0) return false;
    return launch<P, 128, kDv>(q, a) && launch<P, 128, kDk>(q, a) && launch<P, 128, kDq>(q, a);
}

}  // namespace

bool flash_attn_bwd(const FlashAttnBwdArgs& a, void* stream) {
    if (a.head_dim != 128 || a.rows <= 0) return false;
    sycl::queue& q = core::Runtime::get().stream(stream);
    switch (path(q.get_device())) {
    case 0: return run<0>(q, a);
    case 1: return run<1>(q, a);
    case 2: return run<2>(q, a);
    case 3: return run<3>(q, a);
    default: return false;   // no FP16 matrix engines: the caller's products
    }
}

}  // namespace strata::kernels
