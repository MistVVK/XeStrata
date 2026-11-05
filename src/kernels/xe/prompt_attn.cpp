// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/prompt_attn.cpp - see include/strata/kernels/prompt_attn.hpp.
#include "strata/kernels/prompt_attn.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/core/runtime.hpp"
#include "device_target.hpp"
#include "amd_wmma.hpp"
#include "attn_parts.hpp"
#include "strata/core/per_device.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int NT = 256;   // work-items a work-group

// the device's own conversion (f16_bits.hpp's exact one branches and loops: in the inner loops it took most of the
// time); the pools hold no subnormals worth the difference
[[maybe_unused]] inline float h2f(uint16_t bits) { return (float) sycl::bit_cast<sycl::half>(bits); }

// BQ queries of one KV head: R = BQ x G rows (query, head), Q in local memory as FP32 (scaled), the step's BK cells of
// K and V as FP16, the step's scores (then probabilities) per row.  Local memory is read 4 values at a time (Q and
// the probabilities as float4, K as 4 halves), the rows padded to keep those reads aligned and K's rows off one bank.
// Scores: work-item t takes cell t % BK of rows t / BK, + NT / BK, ...: neighbouring work-items read neighbouring K
// rows and the same Q row.  Output: work-item t owns column d = t % HD of rows t / HD, + NT / HD, ... in registers.
template <int HD, int G, int BQ, int BK>
struct Kernel {
    static constexpr int R = BQ * G, RS = NT / HD, OPT = R / RS, QS = HD + 4, KS = HD + 4;
    static constexpr int SG = NT / BK, SR = (R + SG - 1) / SG;   // the scores: row groups, rows a work-item
    static_assert(NT % HD == 0 && R % RS == 0 && HD % 8 == 0 && BK % 4 == 0, "the work splits evenly");
    using f4 = sycl::vec<float, 4>;
    using h4 = sycl::vec<uint16_t, 4>;
    using h8 = sycl::vec<uint16_t, 8>;
    const float* q;
    const uint16_t* kp;
    const uint16_t* vp;
    const int32_t* table;
    float* attn;
    int64_t pos0, T;
    int n_head, n_kv, page_size;
    float scale;
    sycl::local_accessor<float, 1> qs, sc, mrow, lrow, arow;
    sycl::local_accessor<uint16_t, 1> ks, vs;

    void operator()(sycl::nd_item<2> it) const {
        constexpr float NEG = -std::numeric_limits<float>::infinity();
        const int t = (int) it.get_local_id(1), h = (int) it.get_group(0);
        const int64_t qbase = (int64_t) it.get_group(1) * BQ;
        const int d = t % HD, r0 = t / HD;
        float* Q = qs.get_multi_ptr<sycl::access::decorated::no>().get();
        float* S = sc.get_multi_ptr<sycl::access::decorated::no>().get();
        float* M = mrow.get_multi_ptr<sycl::access::decorated::no>().get();
        float* L = lrow.get_multi_ptr<sycl::access::decorated::no>().get();
        float* A = arow.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* K = ks.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* V = vs.get_multi_ptr<sycl::access::decorated::no>().get();
        for (int i = t; i < R * HD; i += NT) {
            const int r = i / HD, dd = i - r * HD;
            const int64_t tq = qbase + r / G;
            Q[r * QS + dd] = tq < T ? q[(tq * n_head + (int64_t) h * G + r % G) * HD + dd] * scale : 0.0f;
        }
        if (t < R) {
            M[t] = NEG;
            L[t] = 0.0f;
        }
        float o[OPT];
        for (int k = 0; k < OPT; ++k) o[k] = 0.0f;
        const int64_t n_cells = pos0 + sycl::min<int64_t>(qbase + BQ, T);   // through the block's last query
        const int j = t % BK, rg = t / BK;
        for (int64_t c0 = 0; c0 < n_cells; c0 += BK) {
            sycl::group_barrier(it.get_group());   // the last step's K, V and probabilities are read
            for (int i = t; i < BK * HD / 8; i += NT) {   // 8 halves a work-item
                const int jj = i / (HD / 8), dd = (i - jj * (HD / 8)) * 8;
                const int64_t c = c0 + jj;
                h8 kv(0), vv(0);
                if (c < n_cells) {
                    const int64_t page = table[c / page_size];
                    const int64_t row = ((page * n_kv + h) * page_size + c % page_size) * HD + dd;
                    kv = *reinterpret_cast<const h8*>(kp + row);
                    vv = *reinterpret_cast<const h8*>(vp + row);
                }
                *reinterpret_cast<h4*>(K + (ptrdiff_t) jj * KS + dd) = kv.template swizzle<0, 1, 2, 3>();
                *reinterpret_cast<h4*>(K + (ptrdiff_t) jj * KS + dd + 4) = kv.template swizzle<4, 5, 6, 7>();
                *reinterpret_cast<h8*>(V + (ptrdiff_t) jj * HD + dd) = vv;
            }
            sycl::group_barrier(it.get_group());
            {   // scores, masked past each row's own position
                const int64_t c = c0 + j;
                float acc[SR];
                for (int k = 0; k < SR; ++k) acc[k] = 0.0f;
                for (int dd = 0; dd < HD; dd += 4) {
                    const h4 kh = *reinterpret_cast<const h4*>(K + (ptrdiff_t) j * KS + dd);
                    const float k0 = h2f(kh[0]), k1 = h2f(kh[1]), k2 = h2f(kh[2]), k3 = h2f(kh[3]);
                    for (int k = 0; k < SR; ++k) {
                        const int r = rg + k * SG;
                        if (SR * SG > R && r >= R) break;
                        const f4 qv = *reinterpret_cast<const f4*>(Q + (ptrdiff_t) r * QS + dd);
                        acc[k] = sycl::fma(qv[0], k0, sycl::fma(qv[1], k1, sycl::fma(qv[2], k2, sycl::fma(qv[3], k3, acc[k]))));
                    }
                }
                for (int k = 0; k < SR; ++k) {
                    const int r = rg + k * SG;
                    if (r >= R) break;
                    S[r * BK + j] = c <= pos0 + qbase + r / G && c < n_cells ? acc[k] : NEG;
                }
            }
            sycl::group_barrier(it.get_group());
            if (t < R) {   // the row's online softmax: its new maximum, the old sum's factor, the step's probabilities
                float mx = M[t];
                for (int jj = 0; jj < BK; ++jj) mx = sycl::fmax(mx, S[t * BK + jj]);
                const float a = M[t] == NEG ? 0.0f : sycl::exp(M[t] - mx);
                float sum = 0.0f;
                for (int jj = 0; jj < BK; ++jj) {
                    const float p = S[t * BK + jj] == NEG ? 0.0f : sycl::exp(S[t * BK + jj] - mx);
                    S[t * BK + jj] = p;
                    sum += p;
                }
                L[t] = L[t] * a + sum;
                M[t] = mx;
                A[t] = a;
            }
            sycl::group_barrier(it.get_group());
            for (int k = 0; k < OPT; ++k) o[k] *= A[r0 + k * RS];
            for (int jj = 0; jj < BK; jj += 4) {
                const float v0 = h2f(V[jj * HD + d]), v1 = h2f(V[(jj + 1) * HD + d]), v2 = h2f(V[(jj + 2) * HD + d]),
                            v3 = h2f(V[(jj + 3) * HD + d]);
                for (int k = 0; k < OPT; ++k) {
                    const f4 p = *reinterpret_cast<const f4*>(S + (ptrdiff_t) (r0 + k * RS) * BK + jj);
                    o[k] = sycl::fma(p[0], v0, sycl::fma(p[1], v1, sycl::fma(p[2], v2, sycl::fma(p[3], v3, o[k]))));
                }
            }
        }
        for (int k = 0; k < OPT; ++k) {
            const int r = r0 + k * RS;
            const int64_t tq = qbase + r / G;
            if (tq < T) attn[(tq * n_head + (int64_t) h * G + r % G) * HD + d] = o[k] / L[r];
        }
    }
};

// RDNA3 and RDNA4 (GFX11, GFX12): Q.K^T and P.V through the FP16 WMMA instructions (16 x 16 x 16, FP32
// accumulators; the layouts as iq_mmq.cpp's int8 ones: lane l holds row l % 16 of A and column l % 16 of B and of the
// accumulator; GFX12 8 of the 16 K values a lane (l / 16 picks which) and accumulator rows 8 (l / 16) + e, GFX11 all
// 16 and rows 2 e + l / 16).  As flash attention 2: each wave owns 16 rows (query, head) of the BQ queries' R = BQ x G,
// with their Q in registers (head_dim 128: FP16 high and low parts, two products, for FP32's precision), and keeps
// the scores, the online softmax (the rows' maxima and sums across the 16 lanes by shuffles) and the output in its
// accumulators; the waves share only the step's 32 cells of K and V (transposed) in local memory.  The probabilities
// go through a wave's own local rows to become the A operand of P.V.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && (defined(__GFX11__) || defined(__GFX12__))
#define STRATA_PA_WMMA 1
#else
#define STRATA_PA_WMMA 0
#endif

// the pools' formats (prompt_attn_tune's pool_format) and their readers, the attention kernels' (attn_parts.hpp)
using xe::attn::kFmtF16;
using xe::attn::kFmtI8;
using xe::attn::kFmtQ4;
using xe::attn::kFmtK8V4;
using xe::attn::kFmtNames;
using xe::attn::pool_format;
template <int HD>
inline sycl::vec<uint16_t, 8> q4_row8(const uint8_t* q4, int64_t row, int dd) {
    return xe::attn::q4_row8<HD>(q4, row, dd);
}

template <int HD, int G, int BQ, int F = kFmtF16>   // F: the pools' format, as JmKernel takes them
struct WmmaKernel {
    static constexpr int BK = 32, R = BQ * G, W = R / 16, DT = HD / 16, CT = BK / 16;
    static constexpr int KS = HD + 8, VS = BK + 8, PS = BK + 8;   // local row strides (halves), 16-byte multiples
    static constexpr bool LO = HD <= 128;                         // Q's low part (registers permitting)
    static_assert(R % 16 == 0 && HD % 16 == 0, "whole tiles");
    using h8 = sycl::vec<uint16_t, 8>;
    const float* q;
    const uint16_t* kp;
    const uint16_t* vp;
    const int32_t* table;
    float* attn;
    int64_t pos0, T;
    int n_head, n_kv, page_size;
    float scale;
    sycl::local_accessor<uint16_t, 1> ks, vs, ps;
    const int8_t* kq = nullptr;     // INT8: the codes and their FP16 scales per 64 values (kp, vp unused)
    const int8_t* vq = nullptr;
    const uint16_t* ksc = nullptr;
    const uint16_t* vsc = nullptr;
    const uint8_t* k4 = nullptr;    // Q4_0's K and V (K8V4: V)
    const uint8_t* v4 = nullptr;

    h8 krow(int64_t row, int dd) const {
        if constexpr (F == kFmtQ4) return q4_row8<HD>(k4, row, dd);
        else return row8(F == kFmtF16, kp, kq, ksc, row, dd);
    }
    h8 vrow(int64_t row, int dd) const {
        if constexpr (F == kFmtQ4 || F == kFmtK8V4) return q4_row8<HD>(v4, row, dd);
        else return row8(F == kFmtF16, vp, vq, vsc, row, dd);
    }
    // 8 values of a pool row from element dd on, as FP16 bits (JmKernel::row8)
    static h8 row8(bool f16_pool, const uint16_t* f16, const int8_t* codes, const uint16_t* scales, int64_t row, int dd) {
        if (f16_pool) {
            return *reinterpret_cast<const h8*>(f16 + row * HD + dd);
        } else {
            const auto c = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(codes + row * HD + dd);
            const float sc = (float) sycl::bit_cast<sycl::half>(scales[row * (HD / 64) + dd / 64]);
            h8 out;
            for (int e = 0; e < 8; ++e) out[e] = sycl::bit_cast<uint16_t>((sycl::half) ((float) c[e] * sc));
            return out;
        }
    }

    void operator()(sycl::nd_item<2> it) const {
#if STRATA_PA_WMMA
        using fv8 = float __attribute__((ext_vector_type(8)));
#if defined(__GFX12__)
        using frag = _Float16 __attribute__((ext_vector_type(8)));
        constexpr int KOFF = 8, KN = 8;   // a lane's K values: 8 of the 16, the half l / 16 picks
        auto row_of = [](int hi, int e) { return 8 * hi + e; };
#else
        using frag = _Float16 __attribute__((ext_vector_type(16)));
        constexpr int KOFF = 0, KN = 16;
        auto row_of = [](int hi, int e) { return 2 * e + hi; };
#endif
        auto mma = [](frag a, frag b, fv8 c) {
#if defined(__GFX12__)
            return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
            return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
        };
        constexpr float NEG = -std::numeric_limits<float>::infinity();
        const auto sg = it.get_sub_group();
        const int t = (int) it.get_local_id(1), h = (int) it.get_group(0);
        const int wave = t / 32, lane = t % 32, lo = lane & 15, hi = lane >> 4;
        const int64_t qbase = (int64_t) it.get_group(1) * BQ;
        uint16_t* K = ks.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* Vt = vs.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* P = ps.get_multi_ptr<sycl::access::decorated::no>().get() + wave * 16 * PS;
        auto frag_at = [](const uint16_t* p) { return *reinterpret_cast<const frag*>(p); };
        // the wave's Q rows as A fragments: row lo of its tile, K values k0 + hi KOFF .. + KN
        frag qh[DT], ql[LO ? DT : 1];
        {
            const int r = wave * 16 + lo;
            const int64_t tq = qbase + r / G;
            const float* src = q + (tq * n_head + (int64_t) h * G + r % G) * HD;
            for (int k = 0; k < DT; ++k)
                for (int e = 0; e < KN; ++e) {
                    const float v = tq < T ? src[k * 16 + hi * KOFF + e] * scale : 0.0f;
                    const _Float16 vh = (_Float16) v;
                    qh[k][e] = vh;
                    if constexpr (LO) ql[k][e] = (_Float16) (v - (float) vh);
                }
        }
        float m[8], l[8];
        fv8 o[DT];
        for (int e = 0; e < 8; ++e) {
            m[e] = NEG;
            l[e] = 0.0f;
        }
        for (int k = 0; k < DT; ++k) o[k] = fv8{0, 0, 0, 0, 0, 0, 0, 0};
        const int64_t n_cells = pos0 + sycl::min<int64_t>(qbase + BQ, T);   // through the block's last query
        const int nt = W * 32;
        for (int64_t c0 = 0; c0 < n_cells; c0 += BK) {
            sycl::group_barrier(it.get_group());   // the last step's K and V are read
            for (int i = t; i < BK * HD / 8; i += nt) {   // K as [cell][dim], V transposed as [dim][cell]
                const int j = i / (HD / 8), dd = (i - j * (HD / 8)) * 8;
                const int64_t c = c0 + j;
                h8 kv(0), vv(0);
                if (c < n_cells) {
                    const int64_t row = ((int64_t) table[c / page_size] * n_kv + h) * page_size + c % page_size;
                    kv = krow(row, dd);
                    vv = vrow(row, dd);
                }
                *reinterpret_cast<h8*>(K + j * KS + dd) = kv;
                for (int e = 0; e < 8; ++e) Vt[(dd + e) * VS + j] = vv[e];
            }
            sycl::group_barrier(it.get_group());
            fv8 sc[CT];
            for (int ct = 0; ct < CT; ++ct) {   // S = Q . K^T for the wave's rows
                fv8 acc = fv8{0, 0, 0, 0, 0, 0, 0, 0};
                for (int k = 0; k < DT; ++k) {
                    const frag b = frag_at(K + (ct * 16 + lo) * KS + k * 16 + hi * KOFF);
                    acc = mma(qh[k], b, acc);
                    if constexpr (LO) acc = mma(ql[k], b, acc);
                }
                sc[ct] = acc;
            }
            float alpha[8];
            for (int e = 0; e < 8; ++e) {   // row 16 wave + row_of(hi, e): its 16 lanes hold its BK columns
                const int64_t qpos = pos0 + qbase + (wave * 16 + row_of(hi, e)) / G;
                float mx = m[e];
                for (int ct = 0; ct < CT; ++ct) {
                    const int64_t c = c0 + ct * 16 + lo;
                    if (c > qpos || c >= n_cells) sc[ct][e] = NEG;
                    mx = sycl::fmax(mx, sc[ct][e]);
                }
                for (int x = 1; x < 16; x <<= 1) mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, x));
                alpha[e] = m[e] == NEG ? 0.0f : sycl::exp(m[e] - mx);
                float sum = 0.0f;
                for (int ct = 0; ct < CT; ++ct) {
                    const _Float16 p = (_Float16) (sc[ct][e] == NEG ? 0.0f : sycl::exp(sc[ct][e] - mx));
                    P[row_of(hi, e) * PS + ct * 16 + lo] = sycl::bit_cast<uint16_t>(p);
                    sum += (float) p;
                }
                for (int x = 1; x < 16; x <<= 1) sum += sycl::permute_group_by_xor(sg, sum, x);
                l[e] = l[e] * alpha[e] + sum;
                m[e] = mx;
            }
            sycl::group_barrier(sg);   // the wave's probabilities are written
            for (int k = 0; k < DT; ++k) {   // O += P . V
                for (int e = 0; e < 8; ++e) o[k][e] *= alpha[e];
                for (int k0 = 0; k0 < BK; k0 += 16)
                    o[k] = mma(frag_at(P + lo * PS + k0 + hi * KOFF), frag_at(Vt + (k * 16 + lo) * VS + k0 + hi * KOFF), o[k]);
            }
        }
        for (int k = 0; k < DT; ++k)
            for (int e = 0; e < 8; ++e) {
                const int r = wave * 16 + row_of(hi, e);
                const int64_t tq = qbase + r / G;
                if (tq < T) attn[(tq * n_head + (int64_t) h * G + r % G) * HD + k * 16 + lo] = o[k][e] / l[e];
            }
#else
        (void) it;
#endif
    }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::sub_group_size<STRATA_SUB_GROUP(32)>};
    }
};

template <int HD, int G, int BQ, int F = kFmtF16>
void launch_wmma(sycl::queue& q, const float* qv, const uint16_t* kp, const uint16_t* vp, const int32_t* table,
                 int64_t pos0, int64_t T, const QsaShapes& s, float* attn, const QsaAttnPools* i8 = nullptr) {
    using KT = WmmaKernel<HD, G, BQ, F>;
    const size_t blocks = (size_t) ((T + BQ - 1) / BQ), nt = (size_t) KT::W * 32;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint16_t, 1> ks(sycl::range<1>((size_t) KT::BK * KT::KS), h),
            vs(sycl::range<1>((size_t) HD * KT::VS), h), ps(sycl::range<1>((size_t) KT::W * 16 * KT::PS), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) s.n_head_kv, blocks * nt), sycl::range<2>(1, nt)),
                       KT{qv, kp, vp, table, attn, pos0, T, (int) s.n_head, (int) s.n_head_kv, (int) s.page_size,
                          1.0f / std::sqrt((float) HD), ks, vs, ps, i8 ? i8->k_q : nullptr, i8 ? i8->v_q : nullptr,
                          i8 ? i8->k_scale : nullptr, i8 ? i8->v_scale : nullptr, i8 ? i8->k_q4 : nullptr,
                          i8 ? i8->v_q4 : nullptr});
    });
}

// The work-group of WmmaKernel<HD, G, BQ> fits the device: its waves within the work-group size, its K, V and
// probabilities within the local memory
template <int HD, int G, int BQ>
bool wmma_fits(const sycl::device& d) {
    using KT = WmmaKernel<HD, G, BQ>;   // (its INT8 form takes the same work-group and local memory)
    const size_t bytes = (size_t) (KT::BK * KT::KS + HD * KT::VS + KT::W * 16 * KT::PS) * sizeof(uint16_t);
    return (size_t) KT::W * 32 <= d.get_info<sycl::info::device::max_work_group_size>() &&
           bytes <= d.get_info<sycl::info::device::local_mem_size>();
}

template <int HD, int G, int BQ, int BK>
void launch(sycl::queue& q, const float* qv, const uint16_t* kp, const uint16_t* vp, const int32_t* table, int64_t pos0,
            int64_t T, const QsaShapes& s, float* attn) {
    using KT = Kernel<HD, G, BQ, BK>;
    const size_t blocks = (size_t) ((T + BQ - 1) / BQ);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>((size_t) KT::R * KT::QS), h), sc(sycl::range<1>((size_t) KT::R * BK), h),
            mrow(sycl::range<1>(KT::R), h), lrow(sycl::range<1>(KT::R), h), arow(sycl::range<1>(KT::R), h);
        sycl::local_accessor<uint16_t, 1> ks(sycl::range<1>((size_t) BK * KT::KS), h), vs(sycl::range<1>((size_t) BK * HD), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) s.n_head_kv, blocks * NT), sycl::range<2>(1, NT)),
                       KT{qv, kp, vp, table, attn, pos0, T, (int) s.n_head, (int) s.n_head_kv, (int) s.page_size,
                          1.0f / std::sqrt((float) HD), qs, sc, mrow, lrow, arow, ks, vs});
    });
}


// RDNA3's and RDNA4's WmmaKernel over INT8, Q4_0 or K8V4 pools, for the shapes prompt_attn takes there (the Q4_0 forms
// heads of 256); false for another shape
template <int F>
bool wmma_form(sycl::queue& q, const float* qv, const QsaAttnPools& pools, int64_t pos0, int64_t T, const QsaShapes& s,
               float* attn) {
    const int64_t g = s.n_head / s.n_head_kv;
    const int32_t* t = pools.page_table;
    if constexpr (F == kFmtI8)
        if (s.head_dim == 128 && g == 10) {
            launch_wmma<128, 10, 8, F>(q, qv, nullptr, nullptr, t, pos0, T, s, attn, &pools);
            return true;
        }
    if (s.head_dim == 256 && g == 8) launch_wmma<256, 8, 8, F>(q, qv, nullptr, nullptr, t, pos0, T, s, attn, &pools);
    else if (s.head_dim == 256 && g == 12) launch_wmma<256, 12, 4, F>(q, qv, nullptr, nullptr, t, pos0, T, s, attn, &pools);
    else if (s.head_dim == 256 && g == 4) {
        const sycl::device d = q.get_device();
        if (wmma_fits<256, 4, 64>(d)) launch_wmma<256, 4, 64, F>(q, qv, nullptr, nullptr, t, pos0, T, s, attn, &pools);
        else if (wmma_fits<256, 4, 32>(d)) launch_wmma<256, 4, 32, F>(q, qv, nullptr, nullptr, t, pos0, T, s, attn, &pools);
        else launch_wmma<256, 4, 16, F>(q, qv, nullptr, nullptr, t, pos0, T, s, attn, &pools);
    } else {
        return false;
    }
    return true;
}

bool wmma_pools(sycl::queue& q, const float* qv, const QsaAttnPools& pools, int fmt, int64_t pos0, int64_t T,
                const QsaShapes& s, float* attn) {
    switch (fmt) {
        case kFmtI8: return wmma_form<kFmtI8>(q, qv, pools, pos0, T, s, attn);
        case kFmtQ4: return wmma_form<kFmtQ4>(q, qv, pools, pos0, T, s, attn);
        case kFmtK8V4: return wmma_form<kFmtK8V4>(q, qv, pools, pos0, T, s, attn);
        default: return false;
    }
}

}  // namespace

bool prompt_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                 int64_t pos0, int64_t T, const QsaShapes& s, float* attn, void* stream) {
    if (T <= 0) return true;
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv != 0 || s.page_size <= 0) return false;
    const int64_t g = s.n_head / s.n_head_kv;
    sycl::queue& queue = core::Runtime::get().stream(stream);
    static const bool no_mma = [] {   // STRATA_PROMPT_ATTN_MMA=0: the FMA kernel on every GPU
        const char* v = std::getenv("STRATA_PROMPT_ATTN_MMA");
        return v != nullptr && std::strtol(v, nullptr, 10) == 0;
    }();
    bool mma = false;
    if (!no_mma && xe::amd_wmma(queue.get_device())) {
        // a wave per 16 rows: 5 for llm-jp's 8 queries of 10 heads, 4 for 8 queries of 8 (local memory 25 to 43 KiB)
        mma = true;
        if (s.head_dim == 128 && g == 10) launch_wmma<128, 10, 8>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else if (s.head_dim == 256 && g == 8) launch_wmma<256, 8, 8>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else if (s.head_dim == 256 && g == 12) launch_wmma<256, 12, 4>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        // Qwen3.5-9B's 4 query heads a KV head: the most queries a work-group the device takes, every wave's 16 rows
        // reading the step's K and V from local memory, so that fewer work-groups read them from the pool.  The RX
        // 9060 XT takes 64 (16 waves, 57 KiB): its 32K prompt's attention 5.1 s against 11.3 s at 16 (2026-10-27)
        else if (s.head_dim == 256 && g == 4) {
            const sycl::device d = queue.get_device();
            if (wmma_fits<256, 4, 64>(d)) launch_wmma<256, 4, 64>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
            else if (wmma_fits<256, 4, 32>(d))
                launch_wmma<256, 4, 32>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
            else launch_wmma<256, 4, 16>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        }
        else mma = false;
    }
    // local memory: BQ x G rows of FP32 Q and BK cells of K and V, 37 to 42 KiB: within the 48 KiB every GPU gives
    if (!mma) {
        if (s.head_dim == 128 && g == 10) launch<128, 10, 4, 32>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else if (s.head_dim == 256 && g == 8) launch<256, 8, 2, 16>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else if (s.head_dim == 256 && g == 12) launch<256, 12, 2, 16>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        // (4 queries of 4 heads: the 16 rows and the local memory of 2 queries of 8)
        else if (s.head_dim == 256 && g == 4) launch<256, 4, 4, 16>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else return false;
    }
    if (stream == nullptr) queue.wait_and_throw();
    return true;
}

namespace {

// The kernel a prompt's attention takes, for a GPU, a shape and a pool format (prompt_attn_tune): the matrix
// engines' (JmKernel, or WmmaKernel's INT8 form on RDNA3 and RDNA4), the FP16 pools' FMA or WMMA kernel, or none: the
// caller's decode kernel (qsa_dense_attn_batch).  Their order changes with the GPU, the pools and the context: over a
// 2,048-query chunk after 28K cells of Qwen3.6-35B-A3B's 16/2 heads of 256 they ran 222, 256 and 216 ms on a B70 with
// FP16 pools (237, -, 272 with INT8 ones), 217, 166 and 397 ms on an RTX 4070 (222, -, 391); llm-jp-4.1's 40/4 heads
// of 128 187 and 1,078 ms on the B70 (226 and 194 on the RTX 4070).  The decode kernel reads every cell again for
// each query: Qwen3.5-9B's 16/4 heads of 256 took it at 16K cells on the B70 and its 64K prompt then ran 86.1 s against
// the FP16 kernel's 83.5.  So the choice is timed at a short and a long context and taken by the request's length.
enum Pick : int { kPickUntimed = -1, kPickNone = 0, kPickMma = 1, kPickFp16 = 2 };
const char* const kPickNames[3] = {"the decode kernel", "the matrix engines'", "the FP16 kernel"};
constexpr double kDecodeMargin = 0.9;   // the decode kernel's dummy time must be under this share of the others' (below)
struct Picks {
    int short_pick = kPickUntimed, long_pick = kPickUntimed;
    int64_t split = 0;   // a chunk reaching this many cells takes long_pick
};
using PickKey = std::tuple<int64_t, int64_t, int64_t, int>;   // head_dim, query heads, KV heads, the pools' format
std::mutex picks_mu;
std::map<PickKey, std::unique_ptr<core::PerDevice<Picks>>> picks;

core::PerDevice<Picks>& picks_for(const QsaShapes& s, int fmt) {
    std::lock_guard<std::mutex> lock(picks_mu);
    auto& p = picks[PickKey{s.head_dim, s.n_head, s.n_head_kv, fmt}];
    if (!p) p = std::make_unique<core::PerDevice<Picks>>();
    return *p;
}

bool env_off(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strtol(v, nullptr, 10) == 0;
}

// The attention the image engine shares (flash_attn.hpp): FP16 pools in pages of 64 cells (or a multiple) read where
// they are, the other formats and page sizes gathered into local memory as FP16 a chunk at a time; the output kept in
// the matrix engines' accumulators.  STRATA_PROMPT_ATTN_FLASH=0: the kernels above (A/B)
bool flash_pools(sycl::queue& queue, const float* q, const QsaAttnPools& pools, int fmt, int64_t pos0, int64_t T,
                 const QsaShapes& s, float* attn) {
    static const bool off = env_off("STRATA_PROMPT_ATTN_FLASH");
    // RDNA3's and RDNA4's WMMA keep the kernels above: the RX 9060 XT ran the shared one 3 to 5 times slower (16/4
    // heads of 256, 2,048 queries after 28K cells: 185 ms against 59 with FP16 pools, 321 against 60 with INT8 ones)
    if (off || xe::amd_wmma(queue.get_device())) return false;
    FlashAttnArgs a;
    a.q32 = q;
    a.q_ld = s.n_head * s.head_dim;
    if (fmt == kFmtF16 && s.page_size % 64 == 0) {
        a.k = pools.k_pool;
        a.v = pools.v_pool;
        a.page_table = pools.page_table;
    } else {
        a.pools = pools;
        a.pools_format = fmt;
    }
    a.page_size = s.page_size;
    a.n_head = s.n_head;
    a.n_kv = s.n_head_kv;
    a.head_dim = s.head_dim;
    a.rows = a.n_causal = T;
    a.keys = pos0 + T;
    a.pos0 = pos0;
    a.first = 0;
    a.row_end = T;
    a.out = attn;
    a.out_ld = s.n_head * s.head_dim;
    a.out_f32 = true;
    a.scale = 1.0f / std::sqrt((float) s.head_dim);
    return flash_attn(a, &queue);
}

// the matrix engines' kernel for the pools, launched (false: none takes them on this GPU)
bool run_mma(sycl::queue& queue, const float* q, const QsaAttnPools& pools, int fmt, int64_t pos0, int64_t T,
             const QsaShapes& s, float* attn) {
    if (flash_pools(queue, q, pools, fmt, pos0, T, s, attn)) return true;
    if (xe::amd_wmma(queue.get_device())) return wmma_pools(queue, q, pools, fmt, pos0, T, s, attn);
    return false;
}

// The fastest of the kernels over a dummy chunk: up to 2,048 queries after the rest of `cells` cells, zero pools
// behind an identity table; one line to stderr.  kPickUntimed when the dummy could not be made (its VRAM).  As many
// queries as a prompt's chunk keeps the GPU as busy as the prompt does: with 512 the FMA kernel's blocks of a few
// queries left it idle and the decode kernel (which also splits the cells) came out 29% ahead at 64K cells on a B70,
// where the prompt ran it 3.5% behind
int time_pick(sycl::queue& queue, const QsaShapes& s, int fmt, int64_t cells, void* stream) {
    const int64_t PS = s.page_size, nkv = s.n_head_kv, nh = s.n_head, hd = s.head_dim;
    const int64_t T = std::min<int64_t>(2048, std::max<int64_t>(1, cells / 2));
    const int64_t pos0 = cells - T, pages = (cells + PS - 1) / PS, rows = pages * nkv * PS;
    std::vector<void*> owned;
    auto dev = [&](size_t bytes) {
        void* p = sycl::malloc_device(std::max<size_t>(bytes, 64), queue);
        if (p == nullptr) throw std::bad_alloc();
        queue.memset(p, 0, std::max<size_t>(bytes, 64));
        owned.push_back(p);
        return p;
    };
    int best = kPickNone;
    double ms[3] = {-1.0, -1.0, -1.0};   // the decode kernel, the matrix engines', the FP16 kernel
    try {
        QsaAttnPools pools;
        std::vector<int32_t> table((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;
        auto* tbl = static_cast<int32_t*>(dev(table.size() * 4));
        queue.memcpy(tbl, table.data(), table.size() * 4);
        pools.page_table = tbl;
        const size_t q4 = (size_t) rows * (size_t) kv_q4_bytes_per_head((int) hd);
        if (fmt == kFmtI8 || fmt == kFmtK8V4) {
            pools.k_q = static_cast<int8_t*>(dev((size_t) (rows * hd)));
            pools.k_scale = static_cast<uint16_t*>(dev((size_t) (rows * hd / 64) * 2));
        }
        if (fmt == kFmtI8) {
            pools.v_q = static_cast<int8_t*>(dev((size_t) (rows * hd)));
            pools.v_scale = static_cast<uint16_t*>(dev((size_t) (rows * hd / 64) * 2));
        }
        if (fmt == kFmtQ4) pools.k_q4 = static_cast<uint8_t*>(dev(q4));
        if (fmt == kFmtQ4 || fmt == kFmtK8V4) pools.v_q4 = static_cast<uint8_t*>(dev(q4));
        if (fmt == kFmtF16) {
            pools.k_pool = static_cast<uint16_t*>(dev((size_t) (rows * hd) * 2));
            pools.v_pool = static_cast<uint16_t*>(dev((size_t) (rows * hd) * 2));
        }
        auto* qv = static_cast<float*>(dev((size_t) (T * nh * hd) * 4));
        auto* out = static_cast<float*>(dev((size_t) (T * nh * hd) * 4));
        queue.wait_and_throw();
        // a warm run of a few queries (the kernel's compile), then the timed chunk; -1: not taken
        auto timed = [&](auto&& run) -> double {
            if (!run(std::min<int64_t>(T, 16))) return -1.0;
            queue.wait_and_throw();
            const auto t0 = std::chrono::steady_clock::now();
            run(T);
            queue.wait_and_throw();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        ms[kPickMma] = timed([&](int64_t n) { return run_mma(queue, qv, pools, fmt, pos0 + T - n, n, s, out); });
        if (fmt == kFmtF16)
            ms[kPickFp16] = timed([&](int64_t n) {
                return prompt_attn(qv, pools.k_pool, pools.v_pool, pools.page_table, pos0 + T - n, n, s, out, stream);
            });
        // the decode kernel over a slice of the queries its scratch fits in (it takes each query alike)
        const uint64_t per_q = qsa_decode_attn_scratch_floats(cells, s) * 4;
        const int64_t nq = std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) ((256ull << 20) / std::max<uint64_t>(per_q, 1))));
        std::vector<int32_t> hs((size_t) (nq * kStepCount));
        for (int64_t t = 0; t < nq; ++t) qsa_step_fill(hs.data() + t * kStepCount, pos0 + T - nq + t, s);
        auto* steps = static_cast<int32_t*>(dev(hs.size() * 4));
        queue.memcpy(steps, hs.data(), hs.size() * 4);
        auto* scratch = static_cast<float*>(dev((size_t) (per_q * (uint64_t) nq)));
        queue.wait_and_throw();
        try {
            const double d = timed([&](int64_t n) {
                if (n == T) qsa_dense_attn_batch(qv, pools, steps, cells, s, scratch, out, nq, stream);
                else qsa_dense_attn_batch(qv, pools, steps, cells, s, scratch, out, std::min<int64_t>(nq, n), stream);
                return true;
            });
            ms[kPickNone] = d * (double) T / (double) nq;
        } catch (const std::exception&) {   // a shape the decode kernel does not take
            ms[kPickNone] = -1.0;
        }
        // the kernels that read a cell once for many queries first; the decode kernel only when it is clearly ahead:
        // timed alone it came out 1.3% ahead of the matrix engines' at 64K cells of Qwen3.5-9B's 16/4 heads of 256 on
        // a B70, and in the prompt (its per-query scratch beside the other layers' work) it ran 15% behind them
        // (attention 75.1 s against 65.2 of a 63K prompt)
        double t_best = -1.0;
        for (int k : {(int) kPickMma, (int) kPickFp16})
            if (ms[k] >= 0.0 && (t_best < 0.0 || ms[k] < t_best)) {
                t_best = ms[k];
                best = k;
            }
        if (ms[kPickNone] >= 0.0 && (t_best < 0.0 || ms[kPickNone] < kDecodeMargin * t_best)) best = kPickNone;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata: timing the prompt attention at %lld cells: %s\n", (long long) cells, e.what());
        best = kPickUntimed;
    }
    for (void* p : owned) sycl::free(p, queue);
    if (best >= 0)
        std::fprintf(stderr, "strata: prompt attention (%lld/%lld heads of %lld, %s pools, %lld queries after %lld cells): "
                             "the decode kernel %.1f ms, the matrix engines' %.1f, the FP16 kernel %.1f: %s\n",
                     (long long) s.n_head, (long long) s.n_head_kv, (long long) s.head_dim, kFmtNames[fmt],
                     (long long) T, (long long) pos0, ms[kPickNone], ms[kPickMma], ms[kPickFp16], kPickNames[best]);
    return best;
}

// STRATA_PROMPT_ATTN_PICK=decode|mma|fp16: that kernel whatever the timing would choose (A/B); kPickUntimed: none
int forced_pick() {
    const char* v = std::getenv("STRATA_PROMPT_ATTN_PICK");
    if (v == nullptr) return kPickUntimed;
    const std::string w = v;
    return w == "decode" ? kPickNone : w == "mma" ? kPickMma : w == "fp16" ? kPickFp16 : kPickUntimed;
}

}  // namespace

void prompt_attn_tune(const QsaShapes& s, int fmt, int64_t max_cells, void* stream) {
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv != 0 || s.page_size <= 0 || max_cells <= 0 || fmt < kFmtF16 ||
        fmt > kFmtK8V4)
        return;
    sycl::queue& queue = core::Runtime::get().stream(stream);
    picks_for(s, fmt).get(queue.get_device(), [&] {
        Picks p;
        if (env_off("STRATA_PROMPT_ATTN_MMA")) return p;   // the fixed choice (prompt_attn)
        if (const int f = forced_pick(); f != kPickUntimed) {   // prompt_attn takes it untimed
            std::fprintf(stderr, "strata: prompt attention: %s (STRATA_PROMPT_ATTN_PICK)\n", kPickNames[f]);
            return p;
        }
        // 8K and up to 64K cells (1 to 2 s on a B70); a cache shorter than the long point takes the short one's choice
        const int64_t a = std::min<int64_t>(max_cells, 8192), b = std::min<int64_t>(max_cells, 65536);
        p.short_pick = time_pick(queue, s, fmt, a, stream);
        p.long_pick = b > a ? time_pick(queue, s, fmt, b, stream) : p.short_pick;
        p.split = (a + b) / 2;
        return p;
    });
}

bool prompt_attn(const float* q, const QsaAttnPools& pools, int64_t pos0, int64_t T, const QsaShapes& s, float* attn,
                 void* stream) {
    if (T <= 0) return true;
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv != 0 || s.page_size <= 0 || pools.page_table == nullptr) return false;
    const int fmt = pool_format(pools);
    if (fmt < 0) return false;
    sycl::queue& queue = core::Runtime::get().stream(stream);
    // STRATA_PROMPT_ATTN_MMA=0: the FP16 kernel for FP16 pools and the decode kernel for INT8 ones, as before; untimed
    // (no prompt_attn_tune for this shape): the matrix engines' kernel where there is one
    static const bool no_mma = env_off("STRATA_PROMPT_ATTN_MMA");
    static const int forced = forced_pick();
    int pick = kPickUntimed;
    if (no_mma) {
        pick = fmt != kFmtF16 ? kPickNone : kPickFp16;
    } else if (forced != kPickUntimed) {
        pick = forced;
    } else {
        const Picks& p = picks_for(s, fmt).get(queue.get_device(), [] { return Picks{}; });
        pick = pos0 + T >= p.split ? p.long_pick : p.short_pick;
    }
    if (pick == kPickNone) return false;
    if (pick != kPickFp16 && run_mma(queue, q, pools, fmt, pos0, T, s, attn)) {
        if (stream == nullptr) queue.wait_and_throw();
        return true;
    }
    if (fmt != kFmtF16) return false;   // the FP32 and WMMA kernels read FP16 pools: the caller keeps the decode kernel
    return prompt_attn(q, pools.k_pool, pools.v_pool, pools.page_table, pos0, T, s, attn, stream);
}

}  // namespace strata::kernels
