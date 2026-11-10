// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/anima_train_block.cpp - see include/strata/image/anima_train.hpp's anima_block.  One Anima block forwards
// and backwards, out of the shared row kernels and products; the cross-attention a head at a time through the shared
// products, as the inference's attn_q runs it.
#include "strata/image/anima_train.hpp"
#include "strata/image/gelu.hpp"
#include "strata/image/qi_kernels.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

constexpr float kEps = 1e-6f;   // the LayerNorms' and the head norms' eps, as anima_dit.cpp takes them

MfOut rows_of(const float* p, int64_t C) {
    MfOut m;
    m.f = p;
    m.C = C;
    return m;
}

// GELU (erf form) of a product's sums, rows16's Op (anima_dit.cpp's GeluErf)
struct GeluErf {
    MfOut x;
    template<class R> dit_rows::Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const dit_rows::Stats&) const { return gelu_erf(x.at(t, c)); }
    float bound(int64_t t, int64_t c) const { return sycl::fmax(x.at(t, c), 0.17f); }
};

// x [T][heads hd] -> out [heads][T][hd] FP16, each head RMS-normed with w (null: copied as it is)
void heads_fwd(const float* x, int64_t T, int64_t heads, int64_t hd, const float* w, uint16_t* out, void* s) {
    constexpr int64_t WG = 128;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * heads * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t g = static_cast<int64_t>(it.get_group_linear_id()), t = g / heads, h = g - t * heads;
            const float* r = x + t * heads * hd + h * hd;
            float ss = 0.f;
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += WG) ss += r[d] * r[d];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float inv = w != nullptr ? 1.f / sycl::sqrt(ss / (float) hd + kEps) : 1.f;
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += WG)
                out[(h * T + t) * hd + d] =
                    kernels::f16_from_f32(sycl::clamp(r[d] * inv * (w != nullptr ? w[d] : 1.f), -65504.f, 65504.f));
        });
}

// heads_fwd's backward (w given): dy [T][heads hd] the gradient of the normed rows, dx [T][heads hd] written
void heads_bwd(const float* x, const float* w, const float* dy, int64_t T, int64_t heads, int64_t hd, float* dx,
               void* s) {
    constexpr int64_t WG = 128;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * heads * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t g = static_cast<int64_t>(it.get_group_linear_id()), t = g / heads, h = g - t * heads;
            const int64_t o = t * heads * hd + h * hd;
            float ss = 0.f, sm = 0.f;
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += WG) {
                ss += x[o + d] * x[o + d];
                sm += dy[o + d] * w[d] * x[o + d];
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            sm = sycl::reduce_over_group(it.get_group(), sm, sycl::plus<float>());
            const float inv = 1.f / sycl::sqrt(ss / (float) hd + kEps);
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += WG)
                dx[o + d] = inv * (dy[o + d] * w[d] - x[o + d] * inv * inv * sm / (float) hd);
        });
}

// one head's softmax of the scores sc [T][L] times scale: P [T][L] FP32 and FP16
void softmax_rows(const float* sc, int64_t T, int64_t L, float scale, float* P, uint16_t* P16, void* s) {
    constexpr int64_t WG = 256;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t t = static_cast<int64_t>(it.get_group_linear_id());
            const float* r = sc + t * L;
            float mx = -INFINITY;
            for (auto j = static_cast<int64_t>(it.get_local_linear_id()); j < L; j += WG) mx = sycl::fmax(mx, r[j] * scale);
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            float sum = 0.f;
            for (auto j = static_cast<int64_t>(it.get_local_linear_id()); j < L; j += WG) sum += sycl::exp(r[j] * scale - mx);
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            for (auto j = static_cast<int64_t>(it.get_local_linear_id()); j < L; j += WG) {
                const float p = sycl::exp(r[j] * scale - mx) / sum;
                P[t * L + j] = p;
                P16[t * L + j] = kernels::f16_from_f32(p);
            }
        });
}

// The cross-attention's gradients in FP16 at a power of two a head (dit_rows::pow2_full): dS sits far below FP16's
// smallest normal value (its largest 7e-6 in block 13 of a real step, its median 1e-11), where rounding it lost 22% of
// dQ and dK (against FP64, 2026-11-10); dO is held the same way.

// each head's largest |x| of rows [T][C] (heads of hd columns) as bits into bits[h] (zeroed)
void heads_amax(const float* x, int64_t T, int64_t heads, int64_t hd, uint32_t* bits, void* s) {
    constexpr int64_t WG = 128;
    const int64_t C = heads * hd;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * heads * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t g = static_cast<int64_t>(it.get_group_linear_id()), t = g / heads, h = g - t * heads;
            float m = 0.f;
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += WG)
                m = sycl::fmax(m, sycl::fabs(x[t * C + h * hd + d]));
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            if (it.get_local_linear_id() == 0)   // non-negative floats order as their bits
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(bits[h]).fetch_max(
                    sycl::bit_cast<uint32_t>(m));
        });
}

// dS = P (dP G_o - D) scale in FP32 over dP (dP from dO / G_o, D = rowsum(dO o O) over the head's values), its
// largest |dS| as bits into *sbits (zeroed)
void softmax_bwd(const float* P, float* dP, const float* dO, const float* O, int64_t T, int64_t L, int64_t hd,
                 int64_t C, float scale, const uint32_t* obits, uint32_t* sbits, void* s) {
    constexpr int64_t WG = 256;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t t = static_cast<int64_t>(it.get_group_linear_id());
            const float Go = dit_rows::pow2_full(sycl::bit_cast<float>(*obits));
            float D = 0.f;
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += WG)
                D += dO[t * C + d] * O[t * C + d];
            D = sycl::reduce_over_group(it.get_group(), D, sycl::plus<float>());
            float m = 0.f;
            for (auto j = static_cast<int64_t>(it.get_local_linear_id()); j < L; j += WG) {
                const float v = P[t * L + j] * (dP[t * L + j] * Go - D) * scale;
                dP[t * L + j] = v;
                m = sycl::fmax(m, sycl::fabs(v));
            }
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            if (it.get_local_linear_id() == 0)
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*sbits).fetch_max(
                    sycl::bit_cast<uint32_t>(m));
        });
}

// x [n] FP32 / G (G from bits) -> FP16
void f16_at(const float* x, int64_t n, const uint32_t* bits, uint16_t* out, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        out[id[0]] = kernels::f16_from_f32(x[id[0]] / dit_rows::pow2_full(sycl::bit_cast<float>(*bits)));
    });
}

// a head's columns of [T][C] FP32 rows / G (G from bits) -> [T][hd] FP16
void head_cols16(const float* x, int64_t T, int64_t C, int64_t hd, const uint32_t* bits, uint16_t* out, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (T * hd)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / hd, d = i - t * hd;
        const float G = dit_rows::pow2_full(sycl::bit_cast<float>(*bits));
        out[i] = kernels::f16_from_f32(sycl::clamp(x[t * C + d] / G, -65504.f, 65504.f));
    });
}

// rows [T][heads hd]: head h's columns times its G (from bits[h])
void heads_times(float* x, int64_t T, int64_t heads, int64_t hd, const uint32_t* bits, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (T * heads * hd)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), h = (i % (heads * hd)) / hd;
        x[i] *= dit_rows::pow2_full(sycl::bit_cast<float>(bits[h]));
    });
}

}  // namespace

void anima_block(const AnimaBlockRun& run, prefill::Gemm& gemm, void* stream, train::Arena* arena) {
    const int64_t C = run.C, F = run.F, H = run.heads, hd = run.hd, R = run.rows, L = run.L, Cc = run.Cc;
    if (C != H * hd) throw std::invalid_argument("anima_block: heads * head_dim must be the width");
    // the call's buffers from the arena (kept from call to call); a buffer written whole before it is read is not
    // zeroed (rf, ru), as moss_block's
    train::Pool pool(stream, arena);
    const int64_t nmax = std::max({C, F, Cc});
    const float* msa = run.mod_sa;
    const float* mca = run.mod_ca;
    const float* mml = run.mod_ml;

    // ---- the adapters' kept rows and shared scratch
    train::AdKeep keep[kAnProducts];
    int64_t mx_T = 0, mx_N = 0, mx_rank = 0, mx_scratch = 0, mx_tmp = 0;
    for (int i = 0; i < kAnProducts; ++i) {
        const train::Adapter* az = run.ad[i];
        if (az == nullptr) continue;
        const int64_t T = (i == kAnCk || i == kAnCv) ? L : R, N = run.p[i].N, K = run.p[i].K, rk = az->rank;
        mx_T = std::max(mx_T, T);
        mx_N = std::max(mx_N, N);
        mx_rank = std::max(mx_rank, rk);
        mx_scratch = std::max(mx_scratch, T * rk + T * N + N * rk + rk * K);
        mx_tmp = std::max(mx_tmp, 2 * T * N + T * rk);
        keep[i] = train::AdKeep{pool.ru(T * rk), pool.rf(T), pool.ru(T * rk)};
    }
    train::AdScratch adsc;
    if (mx_N > 0) {
        adsc.hf = pool.rf(mx_T * mx_rank);
        adsc.y1 = pool.rf(mx_T * mx_N);
        bool plain = true;   // a plain LoRA's delta takes y1 alone; DoRA's and LoKr's maps take y2 beside it
        for (const train::Adapter* az : run.ad)
            plain = plain && (az == nullptr || az->method == train::AdapterMethod::lora);
        adsc.y2 = plain ? nullptr : pool.rf(mx_T * mx_N);
        adsc.fa = pool.rf(mx_N);
        adsc.fb = pool.rf(mx_N);
        adsc.scratch = pool.rf(mx_scratch);
        adsc.tmp16 = pool.ru(mx_tmp);
        adsc.rsT = pool.rf(mx_T);
    }
    // product i of T rows: the base's sums, then its row scales, no bias, and its adapter's delta
    auto fwd = [&](int i, const uint16_t* x16, const float* rs, float* y, int64_t T) {
        const train::Product& P = run.p[i];
        gemm.f16(x16, P.w, y, T, P.N, P.K);
        train::ad_product_fwd(run.ad[i], x16, rs, y, nullptr, T, P.N, P.K, keep[i], adsc, gemm, stream);
    };

    // ---- forward: the self-attention
    uint16_t *a16 = pool.ru(R * C), *a_t = pool.ru(R * C);
    float *a_s = pool.rf(R), *mean1 = pool.rf(R), *inv1 = pool.rf(R);
    dit_rows::rows16(dit_rows::LnMod{run.x, msa, msa + C, C}, R, C, a16, a_s, stream, mean1, inv1);
    train::transpose_f16(a16, R, C, a_t, stream);
    float *yq = pool.rf(R * C), *yk = pool.rf(R * C), *yv = pool.rf(R * C);
    fwd(kAnSq, a16, a_s, yq, R);
    fwd(kAnSk, a16, a_s, yk, R);
    fwd(kAnSv, a16, a_s, yv, R);
    const int64_t Rp = qi_attn_rows(R);
    uint16_t *jq = pool.u(Rp * C), *jk = pool.u(Rp * C), *jv = pool.u(Rp * C), *jo = pool.ru(R * C);
    float* jof = pool.rf(R * C);
    uint32_t* vamax = reinterpret_cast<uint32_t*>(pool.f(1));
    float* vs = pool.rf(R + 1);
    float* lse = pool.rf(R * H);
    float* hsv = pool.rf(H);
    train::v_scale(yv, R * C, nullptr, 0, vamax, vs, R, stream);
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) H), [=](sycl::id<1> id) { hsv[id[0]] = vs[0]; });
    dit_rows::qk_rope16(rows_of(yq, C), rows_of(yk, C), rows_of(yv, C), run.sqn, run.skn, run.cos, run.sin, hsv, R,
                        H, hd, true, jq, jk, jv, stream, kEps);
    qi_attention_train(jq, jk, jv, R, H, hd, jof, lse, stream);
    train::f16_copy(jof, R * C, jo, stream);
    float* yo = pool.rf(R * C);
    fwd(kAnSo, jo, vs + 1, yo, R);
    float* h1 = pool.rf(R * C);
    dit_rows::gate_add_to(h1, run.x, rows_of(yo, C), msa + 2 * C, R, C, stream);

    // ---- the cross-attention
    uint16_t *b16 = pool.ru(R * C), *b_t = pool.ru(R * C);
    float *b_s = pool.rf(R), *mean2 = pool.rf(R), *inv2 = pool.rf(R);
    dit_rows::rows16(dit_rows::LnMod{h1, mca, mca + C, C}, R, C, b16, b_s, stream, mean2, inv2);
    train::transpose_f16(b16, R, C, b_t, stream);
    float *ycq = pool.rf(R * C), *yck = pool.rf(L * C), *ycv = pool.rf(L * C);
    fwd(kAnCq, b16, b_s, ycq, R);
    fwd(kAnCk, run.ctx16, run.ctx_rs, yck, L);
    fwd(kAnCv, run.ctx16, run.ctx_rs, ycv, L);
    uint16_t *qh = pool.ru(H * R * hd), *kh = pool.ru(H * L * hd), *vh = pool.ru(H * L * hd);
    uint16_t *kht = pool.ru(H * hd * L), *vht = pool.ru(H * hd * L);
    heads_fwd(ycq, R, H, hd, run.cqn, qh, stream);
    heads_fwd(yck, L, H, hd, run.ckn, kh, stream);
    heads_fwd(ycv, L, H, hd, nullptr, vh, stream);
    for (int64_t h = 0; h < H; ++h) {
        train::transpose_f16(kh + h * L * hd, L, hd, kht + h * hd * L, stream);
        train::transpose_f16(vh + h * L * hd, L, hd, vht + h * hd * L, stream);
    }
    const float scale = 1.f / std::sqrt((float) hd);
    float* Pf = pool.rf(H * R * L);
    uint16_t* P16 = pool.ru(H * R * L);
    float* sc = pool.rf(R * L);
    float* o32 = pool.rf(R * C);
    for (int64_t h = 0; h < H; ++h) {
        gemm.f16(qh + h * R * hd, kh + h * L * hd, sc, R, L, hd);
        softmax_rows(sc, R, L, scale, Pf + h * R * L, P16 + h * R * L, stream);
        gemm.f16(P16 + h * R * L, vht + h * hd * L, o32 + h * hd, R, hd, L, C);
    }
    uint16_t *o16 = pool.ru(R * C), *o_t = pool.ru(R * C);
    float* o_s = pool.rf(R);
    dit_rows::rows16(dit_rows::Plain{o32, C}, R, C, o16, o_s, stream);
    train::transpose_f16(o16, R, C, o_t, stream);
    float* yco = pool.rf(R * C);
    fwd(kAnCo, o16, o_s, yco, R);
    float* h2 = pool.rf(R * C);
    dit_rows::gate_add_to(h2, h1, rows_of(yco, C), mca + 2 * C, R, C, stream);

    // ---- the MLP
    uint16_t *c16 = pool.ru(R * C), *c_t = pool.ru(R * C);
    float *c_s = pool.rf(R), *mean3 = pool.rf(R), *inv3 = pool.rf(R);
    dit_rows::rows16(dit_rows::LnMod{h2, mml, mml + C, C}, R, C, c16, c_s, stream, mean3, inv3);
    train::transpose_f16(c16, R, C, c_t, stream);
    float* yl1 = pool.rf(R * F);
    fwd(kAnL1, c16, c_s, yl1, R);
    uint16_t *g16 = pool.ru(R * F), *g_t = pool.ru(R * F);
    float* g_s = pool.rf(R);
    dit_rows::rows16(GeluErf{rows_of(yl1, F)}, R, F, g16, g_s, stream);
    train::transpose_f16(g16, R, F, g_t, stream);
    float* yl2 = pool.rf(R * C);
    fwd(kAnL2, g16, g_s, yl2, R);
    dit_rows::gate_add_to(run.out, h2, rows_of(yl2, C), mml + 2 * C, R, C, stream);
    if (run.dout == nullptr) return;

    // ---- backward
    uint16_t *t16 = pool.ru(std::max(R, L) * nmax), *t16t = pool.ru(std::max(R, L) * nmax);
    float* trs = pool.rf(std::max(R, L));
    // dgrad's sums (T K); a frozen weight takes no gradient, so no N K for a wgrad's
    float* sums = pool.rf(std::max(R, L) * nmax);
    auto bwd = [&](int i, const uint16_t* x16, const float* rs, const uint16_t* xt, float* dy, float* dxv, int64_t T,
                   bool acc) {
        train::ad_product_bwd(run.p[i], run.ad[i], x16, rs, xt, dy, dxv, acc, keep[i], adsc, t16, trs, t16t, sums, T,
                              gemm, stream);
    };
    float *dh2 = pool.rf(R * C), *dy = pool.rf(R * nmax), *dz = pool.rf(R * nmax), *dtmp = pool.rf(R * C);
    // the MLP's gated residual
    train::copy_rows(run.dout, R * C, dh2, stream);
    if (run.dmod_ml != nullptr)
        dit_rows::gate_add_bwd(yl2, nullptr, run.dout, R, C, nullptr, run.dmod_ml + 2 * C, stream);
    train::mul_cols(run.dout, mml + 2 * C, R * C, C, dy, stream);
    bwd(kAnL2, g16, g_s, g_t, dy, dz, R, false);   // dz: the GELU's output's gradient [R][F]
    {
        const float* v = yl1;
        float* d = dz;
        dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) (R * F)),
                                            [=](sycl::id<1> id) { d[id[0]] *= gelu_erf_grad(v[id[0]]); });
    }
    bwd(kAnL1, c16, c_s, c_t, dz, dy, R, false);   // dy: the norm's output's gradient [R][C]
    dit_rows::ln_mod_bwd(h2, mean3, inv3, mml + C, dy, R, C, dtmp, run.dmod_ml,
                         run.dmod_ml != nullptr ? run.dmod_ml + C : nullptr, stream);
    train::add_into(dh2, dtmp, R * C, stream);
    // the cross-attention's gated residual
    float* dh1 = pool.rf(R * C);
    train::copy_rows(dh2, R * C, dh1, stream);
    if (run.dmod_ca != nullptr) dit_rows::gate_add_bwd(yco, nullptr, dh2, R, C, nullptr, run.dmod_ca + 2 * C, stream);
    train::mul_cols(dh2, mca + 2 * C, R * C, C, dy, stream);
    float* do32 = pool.rf(R * C);
    bwd(kAnCo, o16, o_s, o_t, dy, do32, R, false);
    float *dq32 = pool.rf(R * C), *dk32 = pool.rf(L * C), *dv32 = pool.rf(L * C), *dP = pool.rf(R * L);
    uint16_t *dO16 = pool.ru(R * hd), *dOt = pool.ru(hd * R), *dS16 = pool.ru(R * L), *dSt = pool.ru(L * R);
    uint16_t *Pt = pool.ru(L * R), *qht = pool.ru(hd * R);
    uint32_t* obits = reinterpret_cast<uint32_t*>(pool.f(2 * H));   // each head's dO's and dS's largest, zeroed
    uint32_t* sbits = obits + H;
    heads_amax(do32, R, H, hd, obits, stream);
    for (int64_t h = 0; h < H; ++h) {
        head_cols16(do32 + h * hd, R, C, hd, obits + h, dO16, stream);
        gemm.f16(dO16, vh + h * L * hd, dP, R, L, hd);   // dP = dO V^T (over G_o)
        softmax_bwd(Pf + h * R * L, dP, do32 + h * hd, o32 + h * hd, R, L, hd, C, scale, obits + h, sbits + h, stream);
        f16_at(dP, R * L, sbits + h, dS16, stream);
        gemm.f16(dS16, kht + h * hd * L, dq32 + h * hd, R, hd, L, C);   // dQ = dS K
        train::transpose_f16(dS16, R, L, dSt, stream);
        train::transpose_f16(qh + h * R * hd, R, hd, qht, stream);
        gemm.f16(dSt, qht, dk32 + h * hd, L, hd, R, C);   // dK = dS^T Q
        train::transpose_f16(P16 + h * R * L, R, L, Pt, stream);
        train::transpose_f16(dO16, R, hd, dOt, stream);
        gemm.f16(Pt, dOt, dv32 + h * hd, L, hd, R, C);   // dV = P^T dO
    }
    // each head's columns back from their powers of two
    heads_times(dq32, R, H, hd, sbits, stream);
    heads_times(dk32, L, H, hd, sbits, stream);
    heads_times(dv32, L, H, hd, obits, stream);
    float *dycq = pool.rf(R * C), *dyck = pool.rf(L * C), *dctx = pool.rf(L * Cc);
    heads_bwd(ycq, run.cqn, dq32, R, H, hd, dycq, stream);
    heads_bwd(yck, run.ckn, dk32, L, H, hd, dyck, stream);
    bwd(kAnCk, run.ctx16, run.ctx_rs, run.ctx16t, dyck, dctx, L, false);
    bwd(kAnCv, run.ctx16, run.ctx_rs, run.ctx16t, dv32, dctx, L, false);
    bwd(kAnCq, b16, b_s, b_t, dycq, dy, R, false);
    dit_rows::ln_mod_bwd(h1, mean2, inv2, mca + C, dy, R, C, dtmp, run.dmod_ca,
                         run.dmod_ca != nullptr ? run.dmod_ca + C : nullptr, stream);
    train::add_into(dh1, dtmp, R * C, stream);
    // the self-attention's gated residual
    if (run.dmod_sa != nullptr) dit_rows::gate_add_bwd(yo, nullptr, dh1, R, C, nullptr, run.dmod_sa + 2 * C, stream);
    if (run.dx != nullptr) train::add_into(run.dx, dh1, R * C, stream);
    train::mul_cols(dh1, msa + 2 * C, R * C, C, dy, stream);
    uint16_t* jo_t = pool.ru(R * C);
    train::transpose_f16(jo, R, C, jo_t, stream);
    float* jdo = pool.rf(R * C);
    bwd(kAnSo, jo, vs + 1, jo_t, dy, jdo, R, false);
    train::scale_by(jdo, R * C, vs, false, stream);
    float *jdq = pool.rf(R * C), *jdk = pool.f(R * C), *jdv = pool.f(R * C);
    qi_attention_bwd(jq, jk, jv, jof, lse, jdo, R, H, hd, jdq, jdk, jdv, true, gemm, stream);
    train::scale_by(jdv, R * C, vs, true, stream);
    float *dyq = pool.rf(R * C), *dyk = pool.rf(R * C);
    dit_rows::qk_rope_bwd(yq, yk, run.sqn, run.skn, run.cos, run.sin, true, jdq, jdk, R, H, hd, dyq, dyk, nullptr,
                          nullptr, stream, true);
    bwd(kAnSq, a16, a_s, a_t, dyq, dy, R, false);
    bwd(kAnSk, a16, a_s, a_t, dyk, dy, R, true);
    bwd(kAnSv, a16, a_s, a_t, jdv, dy, R, true);
    dit_rows::ln_mod_bwd(run.x, mean1, inv1, msa + C, dy, R, C, dtmp, run.dmod_sa,
                         run.dmod_sa != nullptr ? run.dmod_sa + C : nullptr, stream);
    if (run.dx != nullptr) train::add_into(run.dx, dtmp, R * C, stream);
}

}  // namespace strata::image
