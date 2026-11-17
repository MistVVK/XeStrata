// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/prefill/gdn_chunk.cpp - the prompt path's DeltaNet recurrence as a chunked scan on the matrix engines
// (gdn_chunk.hpp): upstream's scheme for RDNA's WMMA (src/prefill/gdn_wmma12.inc: chunks of 16 tokens, FP16 operands,
// the state kept in the accumulators, a work-group a value head, the output norm inside the scan), written on
// joint_matrix and RDNA's WMMA (attn_parts.hpp) for every GPU that reports FP16 matrix engines.
//
// A chunk of CK tokens of one value head, S0 its state (rows r: the key dimension, columns c: the value dimension) at
// the chunk's start, gamma_t the running sum of the gates in the chunk (log decay), e_t = exp(gamma_t),
// f_t = exp(gamma_L - gamma_t) (the WY form of the gated delta rule):
//   A[t][i] = beta_t exp(gamma_t - gamma_i) k_t.k_i (i < t),  T = (I + A)^-1,  P[t][i] = exp(gamma_t - gamma_i) q_t.k_i
//   (i <= t);  X = K S0,  Y_t = beta_t (v_t - e_t X_t),  D = T Y,  O_t = e_t (Q S0)_t + (P D)_t,
//   S = e_L S0 + K^T (f D),  y_t = O_t / sqrt(S)
// - the token-by-token recurrence (kernels.cpp) to rounding.  Past the end of the call gate and beta are 0 and k, q, v
// are 0: those rows of A, Y and so D are 0 and change nothing.
//
// Two kernels a super-block of chunks.  The prep (a work-group a chunk and key head, no matrix engines) works out the
// Gram products in FP32, then for each of the key head's value heads A, T by forward substitution (a work-item a
// column), P, and the tokens' e, f and beta, and writes T as an FP16 hi/lo pair and P in FP16 and the scalars in
// FP32.  The scan (a work-group a value head's 128 columns, or half or a quarter of them: the form) walks the
// super-block's chunks in order, each sub-group holding its TN columns of the state in its accumulators throughout.
// Per chunk the work-group puts k, q, k^T (hi and lo), T (hi and lo), P, v and the scalars in local memory once for
// all its sub-groups (read into registers while the chunk before computes), then each sub-group: its state as FP16
// hi/lo B tiles 16 rows at a time for X (the hi alone for Q S0); Y and f D as FP16 hi/lo B tiles, D as an FP16 B tile;
// O = e Q S0 + P D; S = e_L S + K^T (f D).  Form 0 then normalizes the outputs (one sub-group a token, all 128 columns)
// and writes y16; the others write y and kernels.cpp's norm kernel follows.
//
// The element-wise steps (Y, the row scales, the FP16 tiles) need where each of a lane's accumulator elements sits in
// the tile: a small kernel asks the device once (an accumulator loaded from row * TN + col, its elements written out
// in their order) and the scan reads its lane's row of that table.  The coordinate form of joint_matrix_apply asks
// each element and cost more than the products (flash_attn.cpp); a device whose table is not a permutation of the
// tile does not take the scan.
//
// Numerics: FP32 sums, the state and the running sums FP32 throughout.  The products that only feed the outputs
// (Q S0, O = e QS + P D) take single FP16 operands (q, the state's hi, P, D).  The products that reach the state take
// FP16 hi/lo pairs: X = K S0 with the state split (two products, K single: its error feeds the state through
// Y = beta (v - e X)), and three products each (hi.hi + hi.lo + lo.hi; lo.lo ~ 2^-22 dropped) for D = T Y with T and Y
// split and S = e_L S + K^T (f D) with K and f D split.  Single FP16 everywhere left the state ~ 5e-4 from the FP64
// recurrence, over GDN_CONTRACT.md's 1e-4 floor (CTest on the B70 and the RX 9060 XT); the pairs leave k's own
// rounding in X, of order 1e-5.  Not the bits of kernels.cpp's recurrence kernels.
#include "gdn_chunk.hpp"

#include "strata/core/per_device.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "../kernels/xe/attn_parts.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU prefill_gdn_chunk
#include "strata/kernels/device_arch_tag.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::prefill {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
namespace attn = kernels::xe::attn;
using attn::at;
using sycl::half;

constexpr int S = 128;          // a head's width
constexpr int CK = 16;          // tokens a chunk: T and P are one 16 x 16 tile each
constexpr int KP = S + 8;       // the k and q rows in local memory (halves)
constexpr int KTP = CK + 8;     // k^T's, T's and P's rows (halves)
constexpr int NB_MOST = 256;    // chunks a super-block (4096 tokens), halved while there is no room for its scratch

// a lane's elements of an accumulator tile, in their order, with their index in it (the coordinate table's column)
template<typename G, typename T, mx::use U, size_t M, size_t N, mx::layout L, typename F>
inline void each(G sg, mx::joint_matrix<G, T, U, M, N, L>& c, F&& f) {
    int i = 0;
    mx::joint_matrix_apply(sg, c, [&](T& x) {
        f(x, i);
        ++i;
    });
}
#if STRATA_AMD_WMMA
template<typename F>
inline void each(sycl::sub_group, kernels::xe::wmma::C& c, F&& f) {
    for (int e = 0; e < 8; ++e) {
        float x = c.v[e];
        f(x, e);
        c.v[e] = x;
    }
}
#endif

[[maybe_unused]] inline float sigm(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }   // the device pass only

// The coordinate probe: one sub-group loads an accumulator tile whose element (row, col) is row * TN + col and writes
// each lane's elements in their order as row << 8 | col
template<int P>
struct GdnChunkProbe {
    using Ops = typename attn::Path<P>::Ops;
    static constexpr int TM = Ops::TM, TN = Ops::TN, SG = Ops::SG, NE = TM * TN / SG;
    uint16_t* out;
    sycl::local_accessor<float, 1> lf;

    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>}; }

    void operator()(sycl::nd_item<1> it) const {
#if defined(__SYCL_DEVICE_ONLY__)
        if constexpr (!attn::built(P)) {
            return;   // another GPU's instance: never launched on this target
        } else {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id();
            for (int e = lane; e < TM * TN; e += SG) lf[e] = (float) e;
            sycl::group_barrier(sg);
            typename Ops::C c;
            Ops::load_acc(sg, c, at(lf, 0), TN);
            each(sg, c, [&](float& x, int i) {
                const int e = (int) x;
                out[lane * NE + i] = (uint16_t) ((e / TN) << 8 | (e % TN));
            });
        }
#else
        (void) it;
#endif
    }
};

// The prep, a work-group of NT work-items a (chunk, key head): the chunk's k and q rows (zeros past the call's end) in
// local memory, the Gram products k_t.k_i and q_t.k_i (i <= t) in FP32, then for each of the key head's VPK value
// heads the gates' running sum, P and A, T = (I + A)^-1 (a work-item a column), written as tp[chunk][value head]
// [T hi, T lo, P][CK][CK] (FP16; T lo = T - T hi, the state update's pair) and sc[chunk][value head][e, f, beta][CK]
// (FP32)
template<int HK, int HV>
struct GdnChunkPrep {
    static constexpr int C = 2 * HK * S + HV * S, VPK = HV / HK, NT = 256;
    static constexpr int KF = S + 1, GP = CK + 1;   // the FP32 rows in local memory
    static constexpr size_t O_K = 0, O_Q = (size_t) CK * KF, O_KK = 2 * (size_t) CK * KF, O_QK = O_KK + (size_t) CK * GP,
                            O_A = O_QK + (size_t) CK * GP, O_G = O_A + (size_t) VPK * CK * GP, O_B = O_G + (size_t) VPK * CK,
                            LOCAL = O_B + (size_t) VPK * CK;   // floats
    static_assert(CK * CK == NT && VPK * CK <= NT, "a work-item a Gram product; a work-item a column of T");
    const float* h;
    const float* gate;
    const float* beta;
    half* tp;
    float* sc;
    int64_t t0;
    int64_t T;
    sycl::local_accessor<float, 1> lm;

    void operator()(sycl::nd_item<1> it) const {
        const int ch = (int) it.get_group(0) / HK, qh = (int) it.get_group(0) % HK, tid = (int) it.get_local_id(0);
        const int64_t tb = t0 + (int64_t) ch * CK;
        for (int e = tid; e < CK * S; e += NT) {
            const int t = e / S, r = e % S;
            const int64_t tok = tb + t;
            const bool ok = tok < T;
            const float* ht = h + (ok ? tok : 0) * C + (int64_t) (qh * S + r);
            lm[O_K + (size_t) (t * KF + r)] = ok ? ht[(size_t) HK * S] : 0.0f;
            lm[O_Q + (size_t) (t * KF + r)] = ok ? ht[0] : 0.0f;
        }
        if (tid < VPK) {   // past the end: gate 0, beta 0
            const int vh = qh + tid * HK;
            float gm = 0.0f;
            for (int t = 0; t < CK; ++t) {
                const int64_t tok = tb + t;
                const bool ok = tok < T;
                gm += ok ? gate[tok * HV + vh] : 0.0f;
                lm[O_G + (size_t) (tid * CK + t)] = gm;
                lm[O_B + (size_t) (tid * CK + t)] = ok ? beta[tok * HV + vh] : 0.0f;
            }
        }
        sycl::group_barrier(it.get_group());
        {
            const int t = tid / CK, i = tid % CK;
            float a[4] = {0.0f, 0.0f, 0.0f, 0.0f}, b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (i <= t)
                for (int r = 0; r < S; r += 4)
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const float ki = lm[O_K + (size_t) (i * KF + r + u)];
                        a[u] = sycl::fma(lm[O_K + (size_t) (t * KF + r + u)], ki, a[u]);
                        b[u] = sycl::fma(lm[O_Q + (size_t) (t * KF + r + u)], ki, b[u]);
                    }
            lm[O_KK + (size_t) (t * GP + i)] = (a[0] + a[1]) + (a[2] + a[3]);
            lm[O_QK + (size_t) (t * GP + i)] = (b[0] + b[1]) + (b[2] + b[3]);
        }
        sycl::group_barrier(it.get_group());
        for (int e = tid; e < VPK * CK * CK; e += NT) {
            const int j = e / (CK * CK), t = (e / CK) % CK, i = e % CK, vh = qh + j * HK;
            const float gt = lm[O_G + (size_t) (j * CK + t)], gi = lm[O_G + (size_t) (j * CK + i)];
            lm[O_A + (size_t) ((j * CK + t) * GP + i)] =
                i < t ? lm[O_B + (size_t) (j * CK + t)] * sycl::exp(gt - gi) * lm[O_KK + (size_t) (t * GP + i)] : 0.0f;
            half* pv = tp + ((size_t) ch * HV + (size_t) vh) * (size_t{3} * CK * CK) + size_t{2} * CK * CK;
            pv[t * CK + i] = half(i <= t ? sycl::exp(gt - gi) * lm[O_QK + (size_t) (t * GP + i)] : 0.0f);
        }
        if (tid < VPK * CK) {
            const int j = tid / CK, t = tid % CK, vh = qh + j * HK;
            const float gt = lm[O_G + (size_t) (j * CK + t)], gl = lm[O_G + (size_t) (j * CK + CK - 1)];
            float* s = sc + ((size_t) ch * HV + (size_t) vh) * (size_t{3} * CK);
            s[t] = sycl::exp(gt);
            s[CK + t] = sycl::exp(gl - gt);
            s[2 * CK + t] = lm[O_B + (size_t) (j * CK + t)];
        }
        sycl::group_barrier(it.get_group());
        if (tid < VPK * CK) {   // column c of T: T[t][c] = [t == c] - sum_{m < t} A[t][m] T[m][c]
            const int j = tid / CK, c = tid % CK, vh = qh + j * HK;
            half* tv = tp + ((size_t) ch * HV + (size_t) vh) * (size_t{3} * CK * CK);
            float tc[CK];
#pragma unroll
            for (int t = 0; t < CK; ++t) {
                float v = t == c ? 1.0f : 0.0f;
#pragma unroll
                for (int m = 0; m < t; ++m) v = sycl::fma(-lm[O_A + (size_t) ((j * CK + t) * GP + m)], tc[m], v);
                tc[t] = v;
                const half hi = half(v);
                tv[t * CK + c] = hi;
                tv[CK * CK + t * CK + c] = half(v - (float) hi);   // the lo of the state update's pair
            }
        }
    }
};

// The scan, form NWI: a work-group of NW sub-groups takes CPW = NW TN of value head vh's columns, sub-group g columns
// c0 = cg CPW + g TN onwards, and walks the super-block's chunks.  NORM (form 0 only): the outputs normalized into y16.
template<int HK, int HV, int P, int NWI, bool NORM>
struct GdnChunkScan {
    using Ops = typename attn::Path<P>::Ops;
    static constexpr int TM = Ops::TM, TN = Ops::TN, TK = Ops::TK, SG = Ops::SG;
    static constexpr int NW = (S / TN) >> NWI, NT = NW * SG, CPW = NW * TN, NG = S / CPW;
    static constexpr int C = 2 * HK * S + HV * S, NE = TM * TN / SG, MT = CK / TM, RT = S / TM, KT = S / TK;
    static constexpr int SP = TN + 8, VP = CPW + 4, OP = S + 4;
    static constexpr size_t RS = (size_t) HV * S;
    // local memory, halves: k and q [CK][KP], k^T hi and lo [S][KTP], T hi, T lo and P [CK][KTP] (the work-group's);
    // then a sub-group's state hi and lo slices [TK][SP] and two FP16 tiles [CK][SP] (Y hi and lo, then D hi;
    // f D hi and lo)
    static constexpr size_t O_SK = 0, O_SQ = O_SK + (size_t) CK * KP, O_SKT = O_SQ + (size_t) CK * KP,
                            O_SKTL = O_SKT + (size_t) S * KTP, O_ST = O_SKTL + (size_t) S * KTP,
                            O_STL = O_ST + (size_t) CK * KTP, O_SP = O_STL + (size_t) CK * KTP,
                            H_WG = O_SP + (size_t) CK * KTP, H_SG = 2 * (size_t) TK * SP + 2 * (size_t) CK * SP,
                            H_ALL = H_WG + NW * H_SG;
    // floats: v [CK][VP], e, f, beta [CK]; NORM: the outputs [CK][OP]
    static constexpr size_t O_SV = 0, O_SE = (size_t) CK * VP, O_SF = O_SE + CK, O_SB = O_SF + CK, O_SO = O_SB + CK,
                            F_ALL = O_SO + (NORM ? (size_t) CK * OP : 0);
    static constexpr size_t BYTES = H_ALL * sizeof(half) + F_ALL * sizeof(float);
    // a work-item's share of a chunk's global reads: k and q (float4 each, k its lo too), v, T hi, T lo and P
    // (four halves each)
    static constexpr int QK4 = 2 * CK * S / 4, V4 = CK * CPW / 4, TP4 = 3 * CK * CK / 4;
    static constexpr int NKQ = (QK4 + NT - 1) / NT, NV = (V4 + NT - 1) / NT, NTP = (TP4 + NT - 1) / NT;
    static_assert(S % TN == 0 && S % TM == 0 && CK % TM == 0 && CK == TK && TK % TM == 0 && NW >= 1, "the tiles");
    static_assert(!NORM || NWI == 0, "the norm over all columns");

    float* state;
    const float* h;
    const half* tp;
    const float* sc;
    float* y;
    uint16_t* y16;
    const float* z;
    const float* gamma;
    float eps;
    bool silu;
    const uint16_t* coord;
    int64_t t0;
    int nch;
    int64_t T;
    sycl::local_accessor<half, 1> lh;
    sycl::local_accessor<float, 1> lf;

    // Intel's instances take the large register file: the state's S / TM accumulator tiles stay in registers
    auto get(syclex::properties_tag) const {
        if constexpr (P == 0 || P == 1)
            return syclex::properties{sycl::ext::intel::experimental::grf_size<256>,
                                      syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
        else
            return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
    }

    struct Pre {
        sycl::vec<half, 4> kq[NKQ];
        sycl::vec<half, 4> klo[NKQ];   // k - k hi (the state update's pair; q's entries unused)
        sycl::float4 v[NV];
        sycl::vec<half, 4> tpv[NTP];
        float s[(3 * CK + NT - 1) / NT];   // e, f and beta: a work-item one or more (NT can be under 3 CK)
    };

    // chunk `ch` of the super-block (its first token tb) into registers
    void fetch(Pre& pre, int tid, int vh, int cg, int64_t tb, int ch) const {
        const int qh = vh % HK;
#pragma unroll
        for (int n = 0; n < NKQ; ++n) {
            const int p = tid + n * NT;
            if (p >= QK4) break;
            const int isk = p / (CK * S / 4), w = p % (CK * S / 4), t = w / (S / 4), r = (w % (S / 4)) * 4;
            const int64_t tok = tb + t;
            sycl::float4 x(0.0f);
            if (tok < T)
                x = *reinterpret_cast<const sycl::float4*>(h + tok * C + (int64_t) ((isk ? HK * S : 0) + qh * S + r));
            pre.kq[n] = x.template convert<half>();
            if (isk) pre.klo[n] = (x - pre.kq[n].template convert<float>()).template convert<half>();
        }
#pragma unroll
        for (int n = 0; n < NV; ++n) {
            const int p = tid + n * NT;
            if (p >= V4) break;
            const int t = p / (CPW / 4), c = (p % (CPW / 4)) * 4;
            const int64_t tok = tb + t;
            sycl::float4 x(0.0f);
            if (tok < T)
                x = *reinterpret_cast<const sycl::float4*>(h + tok * C +
                                                            (int64_t) (2 * HK * S + vh * S + cg * CPW + c));
            pre.v[n] = x;
        }
        const half* tpc = tp + ((size_t) ch * HV + (size_t) vh) * (size_t{3} * CK * CK);
#pragma unroll
        for (int n = 0; n < NTP; ++n) {
            const int p = tid + n * NT;
            if (p >= TP4) break;
            pre.tpv[n] = *reinterpret_cast<const sycl::vec<half, 4>*>(tpc + (size_t) p * 4);
        }
        for (int p = tid; p < 3 * CK; p += NT)
            pre.s[p / NT] = sc[((size_t) ch * HV + (size_t) vh) * (size_t{3} * CK) + (size_t) p];
    }

    // the registers into local memory
    void stage(const Pre& pre, int tid) const {
#pragma unroll
        for (int n = 0; n < NKQ; ++n) {
            const int p = tid + n * NT;
            if (p >= QK4) break;
            const int isk = p / (CK * S / 4), w = p % (CK * S / 4), t = w / (S / 4), r = (w % (S / 4)) * 4;
            for (int e = 0; e < 4; ++e) {
                const half v = pre.kq[n][e];
                if (isk) {
                    lh[O_SK + (size_t) (t * KP + r + e)] = v;
                    lh[O_SKT + (size_t) ((r + e) * KTP + t)] = v;
                    lh[O_SKTL + (size_t) ((r + e) * KTP + t)] = pre.klo[n][e];
                } else {
                    lh[O_SQ + (size_t) (t * KP + r + e)] = v;
                }
            }
        }
#pragma unroll
        for (int n = 0; n < NV; ++n) {
            const int p = tid + n * NT;
            if (p >= V4) break;
            const int t = p / (CPW / 4), c = (p % (CPW / 4)) * 4;
            for (int e = 0; e < 4; ++e) lf[O_SV + (size_t) (t * VP + c + e)] = pre.v[n][e];
        }
#pragma unroll
        for (int n = 0; n < NTP; ++n) {
            const int p = tid + n * NT;
            if (p >= TP4) break;
            const int m = p / (CK * CK / 4), w = p % (CK * CK / 4), t = w / (CK / 4), i = (w % (CK / 4)) * 4;
            for (int e = 0; e < 4; ++e)
                lh[(m == 0 ? O_ST : m == 1 ? O_STL : O_SP) + (size_t) (t * KTP + i + e)] = pre.tpv[n][e];
        }
        for (int p = tid; p < 3 * CK; p += NT)   // e, f and beta lie one after the other
            lf[O_SE + (size_t) p] = pre.s[p / NT];
    }

    void operator()(sycl::nd_item<1> it) const {
#if defined(__SYCL_DEVICE_ONLY__)
        if constexpr (!attn::built(P)) {
            return;   // another GPU's instance: never launched on this target
        } else {
        const sycl::sub_group sg = it.get_sub_group();
        const int g = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int tid = (int) it.get_local_id(0);
        const int vh = (int) it.get_group(0) / NG, cg = (int) it.get_group(0) % NG;
        const int c0 = cg * CPW + g * TN;   // the sub-group's first column of the head
        int cr[NE], cc[NE];                 // the lane's elements' rows and columns in a tile
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            const uint16_t w = coord[lane * NE + i];
            cr[i] = w >> 8;
            cc[i] = w & 0xff;
        }
        float* scol = state + (size_t) vh * S + (size_t) c0;   // state[r][vh][c0 + c] at scol[r * RS + c]
        typename Ops::C sacc[RT];
#pragma unroll
        for (int j = 0; j < RT; ++j) {
            Ops::zero(sg, sacc[j]);
            each(sg, sacc[j], [&](float& x, int i) { x = scol[(size_t) (j * TM + cr[i]) * RS + (size_t) cc[i]]; });
        }
        const size_t o_sg = H_WG + (size_t) g * H_SG, o_y0 = o_sg + 2 * (size_t) TK * SP;
        const size_t o_y1 = o_y0 + (size_t) CK * SP;
        const float osc = sycl::rsqrt((float) S);
        Pre pre;
        fetch(pre, tid, vh, cg, t0, 0);
        for (int ch = 0; ch < nch; ++ch) {
            const int64_t tb = t0 + (int64_t) ch * CK;
            const int n = (int) sycl::min((int64_t) CK, T - tb);
            stage(pre, tid);
            sycl::group_barrier(it.get_group());   // (1) the chunk staged
            if (ch + 1 < nch) fetch(pre, tid, vh, cg, tb + CK, ch + 1);
            // (a) X = K S0 and Q S0, the state as FP16 hi/lo B tiles TK rows at a time (X takes the pair: its error
            // feeds the state through Y; Q S0 only the outputs, the hi alone).  One slice a step: the load is done
            // before the next step writes it.
            typename Ops::C x[MT], qs[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                Ops::zero(sg, x[m]);
                Ops::zero(sg, qs[m]);
            }
#pragma unroll
            for (int kk = 0; kk < KT; ++kk) {
                const size_t o_s = o_sg;
#pragma unroll
                for (int jj = 0; jj < TK / TM; ++jj)
                    each(sg, sacc[kk * (TK / TM) + jj], [&](float& v, int i) {
                        const half hi = half(v);
                        lh[o_s + (size_t) ((jj * TM + cr[i]) * SP + cc[i])] = hi;
                        lh[o_s + (size_t) (TK * SP) + (size_t) ((jj * TM + cr[i]) * SP + cc[i])] = half(v - (float) hi);
                    });
                sycl::group_barrier(sg);
                typename Ops::Br bhi, blo;
                Ops::load(sg, bhi, at(lh, o_s), SP);
                Ops::load(sg, blo, at(lh, o_s + (size_t) (TK * SP)), SP);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    typename Ops::A a;
                    Ops::load(sg, a, at(lh, O_SK + (size_t) (m * TM * KP + kk * TK)), KP);
                    Ops::mad(sg, x[m], a, bhi);
                    Ops::mad(sg, x[m], a, blo);
                    Ops::load(sg, a, at(lh, O_SQ + (size_t) (m * TM * KP + kk * TK)), KP);
                    Ops::mad(sg, qs[m], a, bhi);
                }
            }
            // (b) Y = beta (V - e X) as an FP16 hi/lo pair (D feeds the state)
#pragma unroll
            for (int m = 0; m < MT; ++m)
                each(sg, x[m], [&](float& v, int i) {
                    const int t = m * TM + cr[i];
                    const float yv = lf[O_SB + (size_t) t] *
                                     (lf[O_SV + (size_t) (t * VP + g * TN + cc[i])] - lf[O_SE + (size_t) t] * v);
                    const half hi = half(yv);
                    lh[o_y0 + (size_t) (t * SP + cc[i])] = hi;
                    lh[o_y1 + (size_t) (t * SP + cc[i])] = half(yv - (float) hi);
                });
            sycl::group_barrier(sg);
            // (c) D = T Y = T hi Y hi + T hi Y lo + T lo Y hi (lo.lo ~ 2^-22 dropped)
            typename Ops::C d[MT];
            {
                typename Ops::Br bhi, blo;
                Ops::load(sg, bhi, at(lh, o_y0), SP);
                Ops::load(sg, blo, at(lh, o_y1), SP);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    Ops::zero(sg, d[m]);
                    typename Ops::A ahi, a_lo;
                    Ops::load(sg, ahi, at(lh, O_ST + (size_t) (m * TM * KTP)), KTP);
                    Ops::load(sg, a_lo, at(lh, O_STL + (size_t) (m * TM * KTP)), KTP);
                    Ops::mad(sg, d[m], ahi, bhi);
                    Ops::mad(sg, d[m], ahi, blo);
                    Ops::mad(sg, d[m], a_lo, bhi);
                }
            }
            // (d) D as FP16 (the outputs' P D), e Q S0
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                each(sg, d[m], [&](float& v, int i) { lh[o_y0 + (size_t) ((m * TM + cr[i]) * SP + cc[i])] = half(v); });
                each(sg, qs[m], [&](float& v, int i) { v *= lf[O_SE + (size_t) (m * TM + cr[i])]; });
            }
            sycl::group_barrier(sg);   // D hi written; Y's loads done
            // (e) O = e Q S0 + P D
            {
                typename Ops::Br b;
                Ops::load(sg, b, at(lh, o_y0), SP);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    typename Ops::A a;
                    Ops::load(sg, a, at(lh, O_SP + (size_t) (m * TM * KTP)), KTP);
                    Ops::mad(sg, qs[m], a, b);
                }
            }
            // (f) f D as an FP16 hi/lo pair (the state update), from the FP32 D
#pragma unroll
            for (int m = 0; m < MT; ++m)
                each(sg, d[m], [&](float& v, int i) {
                    const int t = m * TM + cr[i];
                    const float fd = v * lf[O_SF + (size_t) t];
                    const half hi = half(fd);
                    lh[o_y0 + (size_t) (t * SP + cc[i])] = hi;
                    lh[o_y1 + (size_t) (t * SP + cc[i])] = half(fd - (float) hi);
                });
            sycl::group_barrier(sg);   // f D written; D hi's loads done
            // (g) S = e_L S + K^T (f D) = e_L S + K hi^T f D hi + K hi^T f D lo + K lo^T f D hi
            {
                const float el = lf[O_SE + CK - 1];
                typename Ops::Br bhi, blo;
                Ops::load(sg, bhi, at(lh, o_y0), SP);
                Ops::load(sg, blo, at(lh, o_y1), SP);
#pragma unroll
                for (int j = 0; j < RT; ++j) {
                    each(sg, sacc[j], [&](float& v, int) { v *= el; });
                    typename Ops::A ahi, a_lo;
                    Ops::load(sg, ahi, at(lh, O_SKT + (size_t) (j * TM * KTP)), KTP);
                    Ops::load(sg, a_lo, at(lh, O_SKTL + (size_t) (j * TM * KTP)), KTP);
                    Ops::mad(sg, sacc[j], ahi, bhi);
                    Ops::mad(sg, sacc[j], ahi, blo);
                    Ops::mad(sg, sacc[j], a_lo, bhi);
                }
            }
            // (h) the outputs
#pragma unroll
            for (int m = 0; m < MT; ++m)
                each(sg, qs[m], [&](float& v, int i) {
                    const int t = m * TM + cr[i];
                    if constexpr (NORM)
                        lf[O_SO + (size_t) (t * OP + g * TN + cc[i])] = v * osc;
                    else if (t < n)
                        y[(size_t) (tb + t) * RS + (size_t) vh * S + (size_t) (c0 + cc[i])] = v * osc;
                });
            sycl::group_barrier(it.get_group());   // (2) the chunk's local memory read; the outputs written
            if constexpr (NORM) {
                // a sub-group a token: rmsnorm over its 128 outputs, gamma, the gate of z, FP16
                for (int t = g; t < n; t += NW) {
                    float o[S / SG], ss = 0.0f;
#pragma unroll
                    for (int k = 0; k < S / SG; ++k) {
                        o[k] = lf[O_SO + (size_t) (t * OP + lane + k * SG)];
                        ss = sycl::fma(o[k], o[k], ss);
                    }
                    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                    const float rs = sycl::rsqrt(ss / (float) S + eps);
                    const size_t row = (size_t) (tb + t) * RS + (size_t) vh * S;
#pragma unroll
                    for (int k = 0; k < S / SG; ++k) {
                        const int col = lane + k * SG;
                        const float zv = z[row + (size_t) col];
                        const float zg = silu ? zv * sigm(zv) : sigm(zv);
                        y16[row + (size_t) col] = kernels::f16_gpu::f16_from_f32(o[k] * rs * gamma[col] * zg);
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < RT; ++j)
            each(sg, sacc[j], [&](float& x, int i) { scol[(size_t) (j * TM + cr[i]) * RS + (size_t) cc[i]] = x; });
        }
#else
        (void) it;
#endif
    }
};

// what a device takes for one shape: the path (attn::path's numbering), its coordinate table, the forms that fit and
// build, and the scratch of a super-block
struct ChunkDev {
    int path = -1;
    uint16_t* coord = nullptr;
    std::vector<int> forms;
    bool norm = false;   // form 0 with the output norm fits as well
    half* tp = nullptr;
    float* sc = nullptr;
    int nb = 0;
};

template<int HK, int HV, int P, int NWI, bool NORM>
bool form_fits(sycl::queue& q) {
    using K = GdnChunkScan<HK, HV, P, NWI, NORM>;
    const sycl::device d = q.get_device();
    if (K::BYTES > d.get_info<sycl::info::device::local_mem_size>() ||
        (size_t) K::NT > d.get_info<sycl::info::device::max_work_group_size>())
        return false;
    try {   // the large register file and the shapes are compiled when first asked: a device that refuses says so here
        (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {d},
                                                                       {sycl::get_kernel_id<K>()});
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xestrata: the chunked DeltaNet scan (form %d) does not build on this GPU (%s)\n", NWI,
                     e.what());
        return false;
    }
    return true;
}

// the coordinate table of path P's accumulator tiles, or null when the probe does not give each element once
template<int P>
uint16_t* probe(sycl::queue& q) {
    using K = GdnChunkProbe<P>;
    constexpr int n = K::SG * K::NE;
    uint16_t* dev = sycl::malloc_device<uint16_t>(n, q);
    if (dev == nullptr) return nullptr;
    std::vector<uint16_t> host(n, 0xffff);
    try {
        q.submit([&](sycl::handler& hd) {
            hd.parallel_for(sycl::nd_range<1>(K::SG, K::SG),
                            K{dev, sycl::local_accessor<float, 1>(sycl::range<1>((size_t) K::TM * K::TN), hd)});
        });
        q.copy(dev, host.data(), n).wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xestrata: the chunked DeltaNet scan's tile probe failed (%s)\n", e.what());
        sycl::free(dev, q);
        return nullptr;
    }
    std::vector<int> seen((size_t) K::TM * K::TN, 0);
    bool ok = true;
    for (uint16_t w : host) {
        const int r = w >> 8, c = w & 0xff;
        if (r >= K::TM || c >= K::TN || seen[(size_t) r * K::TN + (size_t) c]++ != 0) ok = false;
    }
    if (!ok) {
        sycl::free(dev, q);
        return nullptr;
    }
    return dev;
}

template<int HK, int HV, int P>
void fit_forms(sycl::queue& q, ChunkDev& r) {
    if (!form_fits<HK, HV, P, 0, false>(q)) return;
    r.forms.push_back(0);
    r.norm = form_fits<HK, HV, P, 0, true>(q);
    if (form_fits<HK, HV, P, 1, false>(q)) r.forms.push_back(1);
    if (form_fits<HK, HV, P, 2, false>(q)) r.forms.push_back(2);
}

template<int HK, int HV>
const ChunkDev& chunk_dev(sycl::queue& q) {
    static core::PerDevice<ChunkDev> per_device;
    return per_device.get(q.get_device(), [&] {
        ChunkDev r;
        const sycl::device d = q.get_device();
        const int path = attn::path(d);
        std::string why = "no FP16 matrix engines this scan is written for, or XESTRATA_NO_XMX=1";
        using Pr = GdnChunkPrep<HK, HV>;
        if (path >= 0) {
            why = "too little local memory or work-group size for the prep";
            if (Pr::LOCAL * sizeof(float) <= d.get_info<sycl::info::device::local_mem_size>() &&
                (size_t) Pr::NT <= d.get_info<sycl::info::device::max_work_group_size>()) {
                r.coord = path == 0 ? probe<0>(q) : path == 1 ? probe<1>(q) : path == 2 ? probe<2>(q) : probe<3>(q);
                why = "the matrix engines' accumulator tiles did not map one to one in the probe";
            }
        }
        if (r.coord != nullptr) {
            switch (path) {
            case 0: fit_forms<HK, HV, 0>(q, r); break;
            case 1: fit_forms<HK, HV, 1>(q, r); break;
            case 2: fit_forms<HK, HV, 2>(q, r); break;
            default: fit_forms<HK, HV, 3>(q, r); break;
            }
            if (r.forms.empty()) why = "too little local memory or work-group size, or the kernels did not build";
        }
        if (!r.forms.empty()) {
            // the scratch of NB_MOST chunks a super-block, halved down to 16 while the device has no room
            const size_t most = d.get_info<sycl::info::device::max_mem_alloc_size>();
            for (int nb = NB_MOST; nb >= 16 && r.tp == nullptr; nb /= 2) {
                const size_t ntp = (size_t) nb * HV * 3 * CK * CK, nsc = (size_t) nb * HV * 3 * CK;
                if (ntp * sizeof(half) > most) continue;
                try {
                    r.tp = sycl::malloc_device<half>(ntp, q);
                    r.sc = r.tp != nullptr ? sycl::malloc_device<float>(nsc, q) : nullptr;
                } catch (const sycl::exception&) {
                    r.sc = nullptr;
                }
                if (r.sc == nullptr && r.tp != nullptr) {
                    sycl::free(r.tp, q);
                    r.tp = nullptr;
                }
                if (r.tp != nullptr) r.nb = nb;
            }
            if (r.tp == nullptr) {
                r.forms.clear();
                why = "no room for its scratch";
            }
        }
        if (!r.forms.empty()) {
            r.path = path;
            std::fprintf(stderr, "xestrata: the chunked DeltaNet scan can run here (%d forms, %d-token super-blocks)\n",
                         (int) r.forms.size(), r.nb * CK);
        } else {
            std::fprintf(stderr, "xestrata: the chunked DeltaNet scan is not on this GPU (%s)\n", why.c_str());
        }
        return r;
    });
}

template<int HK, int HV, int P, int NWI, bool NORM>
void launch_scan(sycl::queue& q, const ChunkDev& dv, float* state, const float* h, float* y, const GdnChunkNorm* nm,
                 int64_t t0, int nch, int64_t T) {
    using K = GdnChunkScan<HK, HV, P, NWI, NORM>;
    q.submit([&](sycl::handler& hd) {
        hd.parallel_for(sycl::nd_range<1>((size_t) HV * K::NG * K::NT, (size_t) K::NT),
                        K{state, h, dv.tp, dv.sc, y, NORM ? nm->y16 : nullptr, NORM ? nm->z : nullptr,
                          NORM ? nm->gamma : nullptr, NORM ? nm->eps : 0.0f, NORM && nm->silu, dv.coord, t0, nch, T,
                          sycl::local_accessor<half, 1>(sycl::range<1>(K::H_ALL), hd),
                          sycl::local_accessor<float, 1>(sycl::range<1>(K::F_ALL), hd)});
    });
}

template<int HK, int HV, int P>
bool run(sycl::queue& q, const ChunkDev& dv, int form, float* state, const float* h, const float* gate,
         const float* beta, float* y, const GdnChunkNorm* nm, int64_t T) {
    using Pr = GdnChunkPrep<HK, HV>;
    const bool fused = nm != nullptr && form == 0 && dv.norm;
    const int64_t span = (int64_t) dv.nb * CK;
    for (int64_t t0 = 0; t0 < T; t0 += span) {   // the queue is in order: a super-block's prep waits for the last scan
        const int nch = (int) ((std::min<int64_t>(span, T - t0) + CK - 1) / CK);
        q.submit([&](sycl::handler& hd) {
            hd.parallel_for(sycl::nd_range<1>((size_t) nch * HK * Pr::NT, Pr::NT),
                            Pr{h, gate, beta, dv.tp, dv.sc, t0, T,
                               sycl::local_accessor<float, 1>(sycl::range<1>(Pr::LOCAL), hd)});
        });
        if (fused) launch_scan<HK, HV, P, 0, true>(q, dv, state, h, y, nm, t0, nch, T);
        else if (form == 0) launch_scan<HK, HV, P, 0, false>(q, dv, state, h, y, nm, t0, nch, T);
        else if (form == 1) launch_scan<HK, HV, P, 1, false>(q, dv, state, h, y, nm, t0, nch, T);
        else launch_scan<HK, HV, P, 2, false>(q, dv, state, h, y, nm, t0, nch, T);
    }
    return fused;
}

template<int HK, int HV>
bool run_shape(sycl::queue& q, int form, float* state, const float* h, const float* gate, const float* beta, float* y,
               const GdnChunkNorm* nm, int64_t T) {
    const ChunkDev& dv = chunk_dev<HK, HV>(q);
    switch (dv.path) {
    case 0: return run<HK, HV, 0>(q, dv, form, state, h, gate, beta, y, nm, T);
    case 1: return run<HK, HV, 1>(q, dv, form, state, h, gate, beta, y, nm, T);
    case 2: return run<HK, HV, 2>(q, dv, form, state, h, gate, beta, y, nm, T);
    default: return run<HK, HV, 3>(q, dv, form, state, h, gate, beta, y, nm, T);
    }
}

const std::vector<int> kNone;

}  // namespace

const std::vector<int>& gdn_chunk_forms(sycl::queue& q, int64_t h_k, int64_t h_v) {
    if (h_k == 16 && h_v == 48) return chunk_dev<16, 48>(q).forms;
    if (h_k == 16 && h_v == 32) return chunk_dev<16, 32>(q).forms;
    if (h_k == 16 && h_v == 16) return chunk_dev<16, 16>(q).forms;
    return kNone;
}

bool gdn_chunk_run(sycl::queue& q, int form, int64_t h_k, int64_t h_v, float* state, const float* h,
                   const float* gate, const float* beta, float* y, const GdnChunkNorm* norm, int64_t T) {
    if (T <= 0) return false;
    if (h_k == 16 && h_v == 48) return run_shape<16, 48>(q, form, state, h, gate, beta, y, norm, T);
    if (h_k == 16 && h_v == 32) return run_shape<16, 32>(q, form, state, h, gate, beta, y, norm, T);
    return run_shape<16, 16>(q, form, state, h, gate, beta, y, norm, T);
}

}  // namespace strata::prefill
