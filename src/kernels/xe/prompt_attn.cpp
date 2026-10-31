// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/prompt_attn.cpp - see include/strata/kernels/prompt_attn.hpp.
#include "strata/kernels/prompt_attn.hpp"
#include "strata/core/runtime.hpp"
#include "device_target.hpp"
#include "amd_wmma.hpp"
#include "mma_gemm.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/per_device.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

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

// The pools' formats (prompt_attn_tune's pool_format): FP16, INT8 with FP16 scales, Q4_0, INT8 K with Q4_0 V
enum PoolFormat : int { kFmtF16 = 0, kFmtI8 = 1, kFmtQ4 = 2, kFmtK8V4 = 3 };
const char* const kFmtNames[4] = {"FP16", "INT8", "Q4_0", "K8V4"};

int pool_format(const QsaAttnPools& p) {
    if (p.k_q4 != nullptr) return p.v_q4 != nullptr ? kFmtQ4 : -1;
    if (p.v_q4 != nullptr) return p.k_q != nullptr && p.k_scale != nullptr ? kFmtK8V4 : -1;
    if (p.k_q != nullptr) return p.v_q != nullptr && p.k_scale != nullptr && p.v_scale != nullptr ? kFmtI8 : -1;
    return p.k_pool != nullptr && p.v_pool != nullptr ? kFmtF16 : -1;
}

// 8 values of a Q4_0 pool row (kv_q4.hpp) from element dd on, as FP16 bits: element j of a block in the low nibble of
// byte j, j + 16 in the high one, (code - 8) x d.  A block and its codes start at even bytes: four 2-byte words hold
// the eight bytes, two codes in each
template <int HD>
inline sycl::vec<uint16_t, 8> q4_row8(const uint8_t* q4, int64_t row, int dd) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(q4 + row * (int64_t) (HD / QK4_0 * sizeof(block_q4_0)) +
                                                          (dd / QK4_0) * sizeof(block_q4_0));
    const sycl::half d = sycl::bit_cast<sycl::half>(w[0]);
    const int j0 = dd % QK4_0;
    const uint16_t* c = w + 1 + (j0 & 15) / 2;
    sycl::vec<uint16_t, 4> u(c[0], c[1], c[2], c[3]);
    u = (j0 >= 16 ? u >> 4 : u) & (uint16_t) 0x0F0F;   // each byte its code
    const sycl::vec<uint8_t, 8> codes = u.template as<sycl::vec<uint8_t, 8>>();
    const sycl::vec<sycl::half, 8> v = (codes.template convert<sycl::half>() - (sycl::half) 8.0f) * d;
    return v.template as<sycl::vec<uint16_t, 8>>();
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


// The matrix engines through joint_matrix: Intel's XMX (FP16 8 x 16 x 16 on 16 lanes, the large register file) and
// any GPU whose joint_matrix runs FP16 16 x 16 x 16 on 32 lanes (NVIDIA's tensor cores from sm_70 on; RDNA3 and RDNA4
// keep WmmaKernel).  The FMA kernel above ran the prompt's attention at about 3 TFLOP/s on a B70: 63 to 96% of a 32K
// prompt of the plain models (llm-jp-4.1: 274 of 286 s), the share growing with the length.  The scheme is
// WmmaKernel's with the accumulators' layout unknown, as qsa_prompt_attn's portable kernel takes it: each sub-group
// owns TM rows (query, head) of the work-group's R = NSG x TM, the BQ = R / G queries' K and V go through local memory
// BK cells at a step and are read once for all of them; Q . K^T lands in the sub-group's local rows, where its lanes
// keep the rows' online softmax in FP32 and write the probabilities in FP16; P . V starts from zero each step and goes
// through local memory into FP32 registers, scaled by the softmax's correction there (the output is never accumulated
// in the matrix engines, which align and cut the addends).  Q and the probabilities enter in FP16, as WmmaKernel's at
// head_dim 256.  INT8 pools (--kv int8: codes with an FP16 scale per 64 values) are turned into FP16 values as K and
// V come in, so they take this kernel as FP16 ones do; the FMA kernel never read them (the decode kernel did, every
// cell again for each query).  So are Q4_0 ones (--kv q4_0, and k8v4's V), a block of 32 values a work-item: 8 values
// at a time, each with its block's scale read again, ran 715 ms against FP16's 486 (B70, Qwen3.5-9B's 16/4 heads of
// 256, a 2,048-query chunk after 62K cells), a block at a time 493.
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__AMDGCN__) || (defined(__NVPTX__) && STRATA_NV_ARCH < 700))
#define STRATA_PA_JM_PORTABLE 0
#else
#define STRATA_PA_JM_PORTABLE 1
#endif
// The body exists only in a compile with joint_matrix: AMD's device compile has it on CDNA alone, and B's FP16 16 x 16
// tiles do not depend on the kernel's template, so even an uncalled body would instantiate them there
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
#define STRATA_PA_JM_BODY 0
#else
#define STRATA_PA_JM_BODY 1
#endif

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;

template <typename A>
inline auto lptr(const A& a, size_t off) {   // a local accessor's element `off`, as joint_matrix loads and stores
    return a.template get_multi_ptr<sycl::access::decorated::no>() + off;
}

template <int HD, int TM, int SG, int NSG, int BK, int F>
struct JmKernel {
    static constexpr int R = NSG * TM, DT = HD / 16, CT = BK / 16, EPL = TM * 16 / SG, WG = NSG * SG;
    static constexpr int QS = HD + 8, KS = HD + 8, SS = BK + 4, PS = BK + 8;   // local row strides
    static constexpr float TAU = 8.0f;   // the lazy rescaling's margin (log2 units)
    static_assert(HD % 64 == 0 && BK % 16 == 0 && SG % 16 == 0 && (TM * 16) % SG == 0, "whole tiles");
    using half = sycl::half;
    using h8 = sycl::vec<uint16_t, 8>;
    const float* q;
    QsaAttnPools pools;
    float* attn;
    int64_t pos0, T;
    int G, BQ, n_head, n_kv, page_size;
    float scale;
    sycl::local_accessor<half, 1> qs, ks, vs, ps;   // Q [R][QS]; K, V [BK][KS]; P [NSG][TM][PS]
    sycl::local_accessor<float, 1> ss, cs, as;      // S [NSG][TM][SS]; P.V's tile [NSG][TM][16]; correction [NSG][TM]

    // 8 values of a pool row from element dd on, as FP16 bits: FP16 pools as they are, INT8 ones as code x scale
    static h8 row8(const uint16_t* f16, const int8_t* codes, const uint16_t* scales, int64_t row, int dd) {
        if (f16 != nullptr) {
            return *reinterpret_cast<const h8*>(f16 + row * HD + dd);
        } else {
            // the codes are exact in FP16 and so is their product with the FP16 scale before its one rounding: as
            // (half) (code x scale in FP32), eight at a time
            const auto c = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(codes + row * HD + dd);
            const half sc = sycl::bit_cast<half>(scales[row * (HD / 64) + dd / 64]);
            const sycl::vec<half, 8> v = c.template convert<half>() * sycl::vec<half, 8>(sc);
            return v.template as<h8>();
        }
    }
    static h8 q4row8(const uint8_t* q4, int64_t row, int dd) { return q4_row8<HD>(q4, row, dd); }
    // ... a whole block of 32 into dst (local memory, 4 x 8 values): one scale and 16 bytes read for 32 values
    static void q4block(const uint8_t* q4, int64_t row, int b, half* dst) {
        const uint16_t* w = reinterpret_cast<const uint16_t*>(q4 + row * (int64_t) (HD / QK4_0 * sizeof(block_q4_0)) +
                                                              b * sizeof(block_q4_0));
        const half d = sycl::bit_cast<half>(w[0]);
        sycl::vec<uint16_t, 8> u;
        #pragma unroll
        for (int i = 0; i < 8; ++i) u[i] = w[1 + i];
        const sycl::vec<uint16_t, 4> lo0(u[0], u[1], u[2], u[3]), lo1(u[4], u[5], u[6], u[7]);
        const sycl::vec<uint16_t, 4> parts[4] = {lo0 & (uint16_t) 0x0F0F, lo1 & (uint16_t) 0x0F0F,
                                                 (lo0 >> 4) & (uint16_t) 0x0F0F, (lo1 >> 4) & (uint16_t) 0x0F0F};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {   // elements 0-7, 8-15 (low nibbles), 16-23, 24-31 (high ones)
            const sycl::vec<uint8_t, 8> codes = parts[k].template as<sycl::vec<uint8_t, 8>>();
            const sycl::vec<half, 8> v = (codes.template convert<half>() - (half) 8.0f) * d;
            *reinterpret_cast<h8*>(dst + (size_t) k * 8) = v.template as<h8>();
        }
    }
    static h8 k8(const QsaAttnPools& p, int64_t row, int dd) {
        if constexpr (F == kFmtQ4) return q4row8(p.k_q4, row, dd);
        else return row8(F == kFmtF16 ? p.k_pool : nullptr, p.k_q, p.k_scale, row, dd);
    }
    static h8 v8(const QsaAttnPools& p, int64_t row, int dd) {
        if constexpr (F == kFmtQ4 || F == kFmtK8V4) return q4row8(p.v_q4, row, dd);
        else return row8(F == kFmtF16 ? p.v_pool : nullptr, p.v_q, p.v_scale, row, dd);
    }

    // an empty body in a device compile without this form (Intel's 8-row tiles outside Intel's compiles; the 16-row
    // ones on AMD and on NVIDIA before sm_70): the run-time checks (prompt_attn) never launch it there
    void operator()(sycl::nd_item<2> it) const {
#if STRATA_PA_JM_BODY
        if constexpr (TM == 8) {
#if STRATA_DEVICE_NOT_INTEL
            (void) it;
#else
            body(it);
#endif
        } else {
#if STRATA_PA_JM_PORTABLE
            body(it);
#else
            (void) it;
#endif
        }
#else
        (void) it;
#endif
    }

#if STRATA_PA_JM_BODY

    void body(sycl::nd_item<2> it) const {
        using MA = mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, 16, mx::layout::row_major>;
        using MBc = mx::joint_matrix<sycl::sub_group, half, mx::use::b, 16, 16, mx::layout::col_major>;
        using MBr = mx::joint_matrix<sycl::sub_group, half, mx::use::b, 16, 16, mx::layout::row_major>;
        using MC = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, 16>;
        constexpr float NEG = -std::numeric_limits<float>::infinity();
        // Intel's 8 x 16 accumulator gives each lane one column, its element i in row i (Xe's DPAS layout, as
        // qsa_prompt_attn takes it; prompt_attn_parity checks it): the output stays in the accumulators there and its
        // rows are scaled by element order.  Elsewhere the layout is unknown and each step's P . V goes through local
        // memory into registers
        constexpr bool IN_ACC = TM == 8;
        // and the scores stay in the accumulators too where the registers hold them beside the output: head_dim 128
        // (at 256 the output's 16 tiles and the scores' spilled: 261 ms against 218 through local memory, B70, a
        // 2,048-query chunk after 28K cells; at 128 they ran 187 ms against 315)
        constexpr bool S_IN_ACC = IN_ACC && HD <= 128;
        const auto grp = it.get_group();
        const auto sg = it.get_sub_group();
        const int t = (int) it.get_local_id(1), h = (int) it.get_group(0);
        const int w = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int64_t qbase = (int64_t) it.get_group(1) * BQ;
        half* Q = qs.template get_multi_ptr<sycl::access::decorated::no>().get();
        half* K = ks.template get_multi_ptr<sycl::access::decorated::no>().get();
        half* V = vs.template get_multi_ptr<sycl::access::decorated::no>().get();
        half* P = ps.template get_multi_ptr<sycl::access::decorated::no>().get() + (size_t) w * TM * PS;
        float* S = ss.template get_multi_ptr<sycl::access::decorated::no>().get() + (size_t) w * TM * SS;
        float* C = cs.template get_multi_ptr<sycl::access::decorated::no>().get() + (size_t) w * TM * 16;
        const int rows = BQ * G;   // rows past them (and queries past T) are zero and written nowhere
        // the scores in log2 units: exp2 in the softmax
        const float qscale = scale * 1.4426950408889634f;
        for (int i = t; i < R * HD; i += WG) {
            const int r = i / HD, dd = i - r * HD;
            const int64_t tq = qbase + r / G;
            const float v = r < rows && tq < T ? q[(tq * n_head + (int64_t) h * G + r % G) * HD + dd] * qscale : 0.0f;
            Q[r * QS + dd] = (half) v;
        }
        float m[TM], l[TM], f[TM];
        for (int r = 0; r < TM; ++r) {
            m[r] = NEG;
            l[r] = 0.0f;
        }
        MC oa[IN_ACC ? DT : 1];
        float o[IN_ACC ? 1 : DT][IN_ACC ? 1 : EPL];
        if constexpr (IN_ACC) {
            for (int d = 0; d < DT; ++d) mx::joint_matrix_fill(sg, oa[d], 0.0f);
        } else {
            for (int d = 0; d < DT; ++d)
                for (int e = 0; e < EPL; ++e) o[d][e] = 0.0f;
        }
        const int64_t n_cells = pos0 + sycl::min<int64_t>(qbase + BQ, T);   // through the block's last query
        const int r0 = w * TM;   // the sub-group's first row
        for (int64_t c0 = 0; c0 < n_cells; c0 += BK) {
            sycl::group_barrier(grp);   // the last step's K and V are read
            auto cell_row = [&](int64_t c) {
                return ((int64_t) pools.page_table[c / page_size] * n_kv + h) * page_size + c % page_size;
            };
            if constexpr (F == kFmtF16 || F == kFmtI8) {
                for (int i = t; i < BK * HD / 8; i += WG) {   // K and V as [cell][dim]
                    const int j = i / (HD / 8), dd = (i - j * (HD / 8)) * 8;
                    const int64_t c = c0 + j;
                    h8 kv(0), vv(0);
                    if (c < n_cells) {
                        const int64_t row = cell_row(c);
                        kv = k8(pools, row, dd);
                        vv = v8(pools, row, dd);
                    }
                    *reinterpret_cast<h8*>(K + (size_t) j * KS + dd) = kv;
                    *reinterpret_cast<h8*>(V + (size_t) j * KS + dd) = vv;
                }
            } else {   // Q4_0's planes a block of 32 values a work-item; K8V4's INT8 K 8 at a time
                constexpr int NB = HD / QK4_0;
                for (int i = t; i < BK * NB; i += WG) {
                    const int j = i / NB, b = i - j * NB;
                    const int64_t c = c0 + j;
                    half* dv = V + (size_t) j * KS + (size_t) b * QK4_0;
                    half* dk = K + (size_t) j * KS + (size_t) b * QK4_0;
                    if (c < n_cells) {
                        const int64_t row = cell_row(c);
                        q4block(pools.v_q4, row, b, dv);
                        if constexpr (F == kFmtQ4) {
                            q4block(pools.k_q4, row, b, dk);
                        } else {
                            #pragma unroll
                            for (int k = 0; k < 4; ++k)
                                *reinterpret_cast<h8*>(dk + (size_t) k * 8) = row8(nullptr, pools.k_q, pools.k_scale,
                                                                                   row, b * QK4_0 + k * 8);
                        }
                    } else {
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            *reinterpret_cast<h8*>(dv + (size_t) k * 8) = h8(0);
                            *reinterpret_cast<h8*>(dk + (size_t) k * 8) = h8(0);
                        }
                    }
                }
            }
            sycl::group_barrier(grp);
            bool same = true;
            if constexpr (S_IN_ACC) {
                // S = Q . K^T kept in the accumulators: lane `lane` holds column ct 16 + lane of tile ct, its element i
                // row i; masked, the rows' maxima, the probabilities written to local memory as FP16 and summed, all
                // without S itself going through local memory
                MC sacc[CT];
                #pragma unroll
                for (int ct = 0; ct < CT; ++ct) {
                    mx::joint_matrix_fill(sg, sacc[ct], 0.0f);
                    for (int d = 0; d < DT; ++d) {
                        MA a;
                        MBc b;
                        mx::joint_matrix_load(sg, a, lptr(qs, (size_t) r0 * QS + (size_t) d * 16), QS);
                        mx::joint_matrix_load(sg, b, lptr(ks, (size_t) ct * 16 * KS + (size_t) d * 16), KS);
                        mx::joint_matrix_mad(sg, sacc[ct], a, b, sacc[ct]);
                    }
                }
                int64_t qpos[TM];
                float mxr[TM], sum[TM];
                for (int r = 0; r < TM; ++r) {
                    qpos[r] = pos0 + qbase + (r0 + r) / G;
                    mxr[r] = NEG;
                    sum[r] = 0.0f;
                }
                #pragma unroll
                for (int ct = 0; ct < CT; ++ct) {
                    const int64_t c = c0 + (int64_t) ct * 16 + lane;
                    int i = 0;
                    mx::joint_matrix_apply(sg, sacc[ct], [&](float& x) {
                        const int r = i++;
                        if (c > qpos[r] || c >= n_cells) x = NEG;
                        mxr[r] = sycl::fmax(mxr[r], x);
                    });
                }
                for (int r = 0; r < TM; ++r) {   // the lazy reference (below)
                    const float mxv = sycl::reduce_over_group(sg, mxr[r], sycl::maximum<float>());
                    mxr[r] = m[r] == NEG || mxv > m[r] + TAU ? mxv : m[r];
                }
                #pragma unroll
                for (int ct = 0; ct < CT; ++ct) {
                    int i = 0;
                    mx::joint_matrix_apply(sg, sacc[ct], [&](float& x) {
                        const int r = i++;
                        const half p = (half) (x == NEG ? 0.0f : sycl::exp2(x - mxr[r]));
                        P[r * PS + ct * 16 + lane] = p;
                        sum[r] += (float) p;
                    });
                }
                for (int r = 0; r < TM; ++r) {
                    const float total = sycl::reduce_over_group(sg, sum[r], sycl::plus<float>());
                    const float a = mxr[r] == m[r] ? 1.0f : m[r] == NEG ? 0.0f : sycl::exp2(m[r] - mxr[r]);
                    l[r] = l[r] * a + total;
                    m[r] = mxr[r];
                    f[r] = a;
                    same = same && a == 1.0f;
                }
            } else {
            for (int ct = 0; ct < CT; ++ct) {   // S = Q . K^T for the sub-group's rows
                MC acc;
                mx::joint_matrix_fill(sg, acc, 0.0f);
                for (int d = 0; d < DT; ++d) {
                    MA a;
                    MBc b;
                    mx::joint_matrix_load(sg, a, lptr(qs, (size_t) r0 * QS + (size_t) d * 16), QS);
                    mx::joint_matrix_load(sg, b, lptr(ks, (size_t) ct * 16 * KS + (size_t) d * 16), KS);
                    mx::joint_matrix_mad(sg, acc, a, b, acc);
                }
                mx::joint_matrix_store(sg, acc, lptr(ss, (size_t) w * TM * SS + (size_t) ct * 16), SS, mx::layout::row_major);
            }
            sycl::group_barrier(sg);
            // each row's online softmax, its BK columns across the lanes.  Lazily rescaled (as FlashAttention-3 and
            // qsa_prompt_attn): a row's reference moves only when a score passes it by more than 2^TAU, so p <= 2^TAU
            // stays within FP16, and most steps leave the output as it is
            for (int r = 0; r < TM; ++r) {
                const int64_t qpos = pos0 + qbase + (r0 + r) / G;
                float mxv = NEG;
                for (int cj = lane; cj < BK; cj += SG) {
                    const int64_t c = c0 + cj;
                    if (c > qpos || c >= n_cells) S[r * SS + cj] = NEG;
                    mxv = sycl::fmax(mxv, S[r * SS + cj]);
                }
                mxv = sycl::reduce_over_group(sg, mxv, sycl::maximum<float>());
                const float m_new = m[r] == NEG || mxv > m[r] + TAU ? mxv : m[r];
                float sum = 0.0f;
                for (int cj = lane; cj < BK; cj += SG) {
                    const float x = S[r * SS + cj];
                    const half p = (half) (x == NEG ? 0.0f : sycl::exp2(x - m_new));
                    P[r * PS + cj] = p;
                    sum += (float) p;
                }
                sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
                const float a = m_new == m[r] ? 1.0f : m[r] == NEG ? 0.0f : sycl::exp2(m[r] - m_new);
                l[r] = l[r] * a + sum;
                m[r] = m_new;
                f[r] = a;
                same = same && a == 1.0f;
            }
            }
            sycl::group_barrier(sg);   // the probabilities are written
            if constexpr (IN_ACC) {   // O = O x correction (only when a reference moved) + P . V, in the accumulators
                if (!same)
                    #pragma unroll
                    for (int d = 0; d < DT; ++d) {
                        int i = 0;
                        mx::joint_matrix_apply(sg, oa[d], [&](float& x) { x *= f[i++]; });
                    }
                #pragma unroll
                for (int kk = 0; kk < CT; ++kk) {
                    MA a;
                    mx::joint_matrix_load(sg, a, lptr(ps, (size_t) w * TM * PS + (size_t) kk * 16), PS);
                    #pragma unroll
                    for (int d = 0; d < DT; ++d) {
                        MBr b;
                        mx::joint_matrix_load(sg, b, lptr(vs, (size_t) kk * 16 * KS + (size_t) d * 16), KS);
                        mx::joint_matrix_mad(sg, oa[d], a, b, oa[d]);
                    }
                }
            } else {   // P . V from zero, through local memory into the registers: O = O x correction + it
                for (int d = 0; d < DT; ++d) {
                    MC acc;
                    mx::joint_matrix_fill(sg, acc, 0.0f);
                    for (int kk = 0; kk < CT; ++kk) {
                        MA a;
                        MBr b;
                        mx::joint_matrix_load(sg, a, lptr(ps, (size_t) w * TM * PS + (size_t) kk * 16), PS);
                        mx::joint_matrix_load(sg, b, lptr(vs, (size_t) kk * 16 * KS + (size_t) d * 16), KS);
                        mx::joint_matrix_mad(sg, acc, a, b, acc);
                    }
                    mx::joint_matrix_store(sg, acc, lptr(cs, (size_t) w * TM * 16), 16, mx::layout::row_major);
                    sycl::group_barrier(sg);
                    for (int e = 0; e < EPL; ++e) {   // element e * SG + lane of the tile: row (e SG + lane) / 16
                        const int r = (e * SG + lane) / 16;
                        o[d][e] = (same ? o[d][e] : o[d][e] * f[r]) + C[r * 16 + lane % 16];
                    }
                    sycl::group_barrier(sg);   // read before the next tile overwrites it
                }
            }
        }
        for (int d = 0; d < DT; ++d) {
            if constexpr (IN_ACC) {   // the accumulators through local memory, once
                mx::joint_matrix_store(sg, oa[d], lptr(cs, (size_t) w * TM * 16), 16, mx::layout::row_major);
                sycl::group_barrier(sg);
            }
            for (int e = 0; e < EPL; ++e) {
                const int r = (e * SG + lane) / 16, row = r0 + r;
                const int64_t tq = qbase + row / G;
                const float x = IN_ACC ? C[r * 16 + lane % 16] : o[IN_ACC ? 0 : d][IN_ACC ? 0 : e];
                if (row < rows && tq < T) attn[(tq * n_head + (int64_t) h * G + row % G) * HD + (int64_t) d * 16 + lane % 16] = x / l[r];
            }
            if constexpr (IN_ACC) sycl::group_barrier(sg);
        }
    }

#endif

    auto get(syclex::properties_tag) const {
        if constexpr (TM == 8)
            return syclex::properties{sycl::ext::intel::experimental::grf_size<256>,
                                      syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
        else
            return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
    }

    static size_t local_bytes() {
        return (size_t) (R * QS + 2 * BK * KS + NSG * TM * PS) * sizeof(half) +
               (size_t) (NSG * TM * SS + NSG * TM * 16 + NSG * TM) * sizeof(float);
    }
};

template <int HD, int TM, int SG, int NSG, int BK, int F>
void launch_jm(sycl::queue& q, const float* qv, const QsaAttnPools& pools, int64_t pos0, int64_t T, const QsaShapes& s,
               float* attn) {
    using KT = JmKernel<HD, TM, SG, NSG, BK, F>;
    const int G = (int) (s.n_head / s.n_head_kv), BQ = std::max(1, KT::R / G);
    const size_t blocks = (size_t) ((T + BQ - 1) / BQ);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> qs(sycl::range<1>((size_t) KT::R * KT::QS), h),
            ks(sycl::range<1>((size_t) BK * KT::KS), h), vs(sycl::range<1>((size_t) BK * KT::KS), h),
            ps(sycl::range<1>((size_t) NSG * TM * KT::PS), h);
        sycl::local_accessor<float, 1> ss(sycl::range<1>((size_t) NSG * TM * KT::SS), h),
            cs(sycl::range<1>((size_t) NSG * TM * 16), h), as(sycl::range<1>((size_t) NSG * TM), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) s.n_head_kv, blocks * KT::WG), sycl::range<2>(1, KT::WG)),
                       KT{qv, pools, attn, pos0, T, G, BQ, (int) s.n_head, (int) s.n_head_kv, (int) s.page_size,
                          1.0f / std::sqrt((float) HD), qs, ks, vs, ps, ss, cs, as});
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

// JmKernel<HD, TM, SG, NSG, BK, F> launched when its work-group fits the device (work-items and local memory)
template <int HD, int TM, int SG, int NSG, int BK, int F>
bool jm_try(sycl::queue& q, const float* qv, const QsaAttnPools& pools, int64_t pos0, int64_t T, const QsaShapes& s,
            float* attn) {
    using KT = JmKernel<HD, TM, SG, NSG, BK, F>;
    const sycl::device d = q.get_device();
    if ((size_t) KT::WG > d.get_info<sycl::info::device::max_work_group_size>() ||
        KT::local_bytes() > d.get_info<sycl::info::device::local_mem_size>())
        return false;
    launch_jm<HD, TM, SG, NSG, BK, F>(q, qv, pools, pos0, T, s, attn);
    return true;
}

// the largest work-group of the form that fits, more rows (queries reading the same K and V) first
template <int HD, int TM, int SG, int F>
bool jm_form(sycl::queue& q, const float* qv, const QsaAttnPools& pools, int64_t pos0, int64_t T, const QsaShapes& s,
             float* attn) {
    // 64 rows either way, then half as many; steps of at least 32 cells (Intel's compiler took no single-tile array of
    // scores through joint_matrix_apply)
    constexpr int WIDE = TM == 8 ? 8 : 4;
    return jm_try<HD, TM, SG, WIDE, 64, F>(q, qv, pools, pos0, T, s, attn) ||
           jm_try<HD, TM, SG, WIDE, 32, F>(q, qv, pools, pos0, T, s, attn) ||
           jm_try<HD, TM, SG, WIDE / 2, 32, F>(q, qv, pools, pos0, T, s, attn);
}

template <int V>
using IntC = std::integral_constant<int, V>;

// The joint_matrix kernel for the device: Intel's XMX form where xmx_available says so (the XMX products' check: its
// matrix combinations, 16-wide sub-groups and the large register file), else the portable one where the device's
// joint_matrix takes FP16 16 x 16 x 16 on 32 lanes; false when there is neither, for a head_dim other than 128 or 256,
// or when no work-group of it fits

bool jm_attn(sycl::queue& q, const float* qv, const QsaAttnPools& pools, int fmt, int64_t pos0, int64_t T,
             const QsaShapes& s, float* attn) {
    static core::PerDevice<int> per_device;   // 8: Intel's form, 16: the portable one, 0: none
    const int form = per_device.get(q.get_device(), [&q] {
        if (xmx_available(XmxType::f16)) return 8;
        return xe::mma_shape_is(q, 16, 16, 16, 32) ? 16 : 0;
    });
    if (form == 0 || (s.head_dim != 128 && s.head_dim != 256)) return false;
    // the Q4_0 forms for heads of 256 only (the Hadamard rotation's width)
    if ((fmt == kFmtQ4 || fmt == kFmtK8V4) && s.head_dim != 256) return false;
    auto run = [&](auto hd, auto tm, auto sg) {
        constexpr int HD = decltype(hd)::value, TM = decltype(tm)::value, SG = decltype(sg)::value;
        switch (fmt) {
            case kFmtF16: return jm_form<HD, TM, SG, kFmtF16>(q, qv, pools, pos0, T, s, attn);
            case kFmtI8: return jm_form<HD, TM, SG, kFmtI8>(q, qv, pools, pos0, T, s, attn);
            case kFmtQ4:
                if constexpr (HD == 256) return jm_form<HD, TM, SG, kFmtQ4>(q, qv, pools, pos0, T, s, attn);
                return false;
            case kFmtK8V4:
                if constexpr (HD == 256) return jm_form<HD, TM, SG, kFmtK8V4>(q, qv, pools, pos0, T, s, attn);
                return false;
            default: return false;
        }
    };
    if (form == 8)
        return s.head_dim == 128 ? run(IntC<128>{}, IntC<8>{}, IntC<16>{}) : run(IntC<256>{}, IntC<8>{}, IntC<16>{});
    return s.head_dim == 128 ? run(IntC<128>{}, IntC<16>{}, IntC<32>{}) : run(IntC<256>{}, IntC<16>{}, IntC<32>{});
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

// the matrix engines' kernel for the pools, launched (false: none takes them on this GPU)
bool run_mma(sycl::queue& queue, const float* q, const QsaAttnPools& pools, int fmt, int64_t pos0, int64_t T,
             const QsaShapes& s, float* attn) {
    if (xe::amd_wmma(queue.get_device())) return wmma_pools(queue, q, pools, fmt, pos0, T, s, attn);
    return jm_attn(queue, q, pools, fmt, pos0, T, s, attn);
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
