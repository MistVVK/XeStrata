// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/train_adapter.cpp - see include/strata/image/train_ops.hpp's ad_product_fwd and ad_product_bwd: one
// product's adapter (a plain LoRA, a DoRA, a LoKr) forwards beside the base product's rows and backwards into its
// factors' gradients and the input's, the part every DiT's training block shares.
#include "strata/image/train_ops.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>

#include "dit_rows.hpp"

namespace strata::image::train {

namespace {

using kernels::f16_from_f32;

// dX[t][k] += sum_r dh[t][r] A[r][k] (dh [R][rank] FP32, A [rank][K] FP16): an item a column and kTR rows, A's value
// read once for them
void lora_dx(const float* dh, const uint16_t* A, int64_t R, int64_t rank, int64_t K, float* dX, void* s) {
    constexpr int64_t kTR = 16;
    const int64_t groups = (R + kTR - 1) / kTR;
    dit_rows::q_of(s).parallel_for(sycl::range<2>((size_t) groups, (size_t) K), [=](sycl::item<2> it) {
        const int64_t t0 = (int64_t) it.get_id(0) * kTR, k = (int64_t) it.get_id(1);
        const int64_t n = sycl::min(kTR, R - t0);
        float acc[kTR] = {};
        for (int64_t r = 0; r < rank; ++r) {
            const float a = kernels::f32_from_f16(A[r * K + k]);
            for (int64_t i = 0; i < kTR; ++i)
                if (i < n) acc[i] += dh[(t0 + i) * rank + r] * a;
        }
        for (int64_t i = 0; i < kTR; ++i)
            if (i < n) dX[(t0 + i) * K + k] += acc[i];
    });
}

// ---- adapters: the forward keeps h16/hr/h16t (AdKeep), the backward reads them back ----

// y[t][n] = fa[n] * (y[t][n] * rsx[t]) + fb[n] * (delta[t][n] * hr[t]) + bias[n] (DoRA), or the plain LoRA's
// y[t][n] = y[t][n] * rsx[t] + scale * (delta[t][n] * hr[t]) + bias[n] (fa/fb null).  y holds the base GEMM's rows
// (still divided by rsx), delta the adapter's GEMM rows (still divided by hr): both scales come back here.
void ad_rows_back(float* y, const float* rsx, const float* bias, const float* delta, const float* hr,
                  const float* fa, const float* fb, float scale, int64_t R, int64_t N, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) R * N), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / N, n = i - t * N;
        const float yb = y[i] * (rsx != nullptr ? rsx[t] : 1.f);
        const float dd = delta[i] * hr[t];
        y[i] = (fa != nullptr ? fa[n] * yb + fb[n] * dd : yb + scale * dd) + (bias != nullptr ? bias[n] : 0.f);
    });
}

// ---- LoKr (kronecker-structured delta): Y += s kron(w1, w2a w2b) X without materializing the product ----

// dst rows (r,b) cols a from src rows (r,a) cols b.
void lokr_transpose(const float* src, int64_t R, int64_t A, int64_t B, float* dst, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<2>((size_t) R * B, (size_t) A), [=](sycl::id<2> id) {
        const int64_t rb = static_cast<int64_t>(id[0]), a = static_cast<int64_t>(id[1]);
        const int64_t r = rb / B, b = rb - r * B;
        dst[rb * A + a] = src[(r * A + a) * B + b];
    });
}

// y = (y + s D) rsx + b, D read through the transposed product's rows: D[r][l*ok+k] = o[(r*ok+k)][l].  o was read
// from the same divided rows as y, so it takes rsx too.
void lokr_rows_back(float* y, const float* rsx, const float* bias, const float* o, float scale, int64_t R,
                    int64_t N, int64_t ol, int64_t ok, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) R * N), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / N, j = i - t * N;
        const int64_t l = j / ok, k = j - l * ok;
        y[i] = (y[i] + scale * o[(t * ok + k) * ol + l]) * (rsx != nullptr ? rsx[t] : 1.f) +
               (bias != nullptr ? bias[j] : 0.f);
    });
}

// dxv[t*K + m*inn+n] += s * dyrs[t] * dx[((t*inn+n)*im)+m]: the delta path's input gradient joins the base's.
void lokr_dgrad_add(float* dxv, const float* dx, const float* dyrs, float scale, int64_t R, int64_t K,
                    int64_t im, int64_t inn, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) R * K), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / K, j = i - t * K;
        const int64_t m = j / inn, n = j - m * inn;
        dxv[i] += scale * dyrs[t] * dx[(t * inn + n) * im + m];
    });
}

// one LoKr product's forward rows: Y += s kron(w1, w2a w2b) X through two small products, w2 never formed;
// t1/t2 are R*ad_maxN scratch, tt is u16 scratch (two f16 row blocks).  x16 is [R][im*inn] rows, (m,n) with n fastest.
void lokr_product_fwd(const Adapter* ad, const uint16_t* x16, const float* rsx, float* y, const float* bias,
                      int64_t R, int64_t N, float* t1, float* t2, uint16_t* tt,
                      prefill::Gemm& gemm, void* stream) {
    const int64_t ol = ad->ol, ok = ad->ok, im = ad->im, inn = ad->inn, rk = ad->rank;
    const float s = ad->scale;
    // U[(R*im), rk] = X[(R*im), inn] w2b ; T[(R*im), ok] = U w2a
    gemm.f16(x16, ad->w2b, t1, R * im, rk, inn);
    uint16_t* u16 = tt;
    f16_copy(t1, R * im * rk, u16, stream);
    gemm.f16(u16, ad->w2a, t2, R * im, ok, rk);
    // T2[(R*ok), im] = transpose(T); O[(R*ok), ol] = T2 w1
    lokr_transpose(t2, R, im, ok, t1, stream);
    uint16_t* t2_16 = tt + (size_t) R * im * rk;
    f16_copy(t1, R * ok * im, t2_16, stream);
    gemm.f16(t2_16, ad->w1, t2, R * ok, ol, im);
    lokr_rows_back(y, rsx, bias, t2, s, R, N, ol, ok, stream);
}

// one LoKr product's backward: the delta path's input gradient (added into dxv) and factor gradients (added into
// dW1/dW2a/dW2b).  dYv is the gradient reaching the product's rows (unscaled: the base path above read it as is).
// t1/t2 are R*ad_maxN scratch, tt/lrs R*N u16/R f32 scratch, t16t the [N][R] transpose map, sums 2*N*K scratch.
void lokr_product_bwd(const Adapter* ad, const uint16_t* xt, const float* rsx, float* dYv, float* dxv, float* t1,
                      float* t2, uint16_t* tt, float* trs, uint16_t* t16t, float* sums, int64_t R, int64_t N,
                      int64_t K, prefill::Gemm& gemm, void* stream) {
    const int64_t ol = ad->ol, ok = ad->ok, im = ad->im, inn = ad->inn, rk = ad->rank;
    const float s = ad->scale;
    // dY in FP16 rows with their scales, then the transposed apply: P1 = dY w2a, P2 = P1 w2b, dX = P2T w1t
    uint16_t* dy16 = tt;
    float* dyrs = trs;
    dit_rows::rows16(dit_rows::Grad{{dYv, N}}, R, N, dy16, dyrs, stream);
    gemm.f16(dy16, ad->w2a, t1, R * ol, rk, ok);
    uint16_t* p1_16 = tt + (size_t) R * N;
    f16_copy(t1, R * ol * rk, p1_16, stream);
    gemm.f16(p1_16, ad->w2b, t2, R * ol, inn, rk);
    lokr_transpose(t2, R, ol, inn, t1, stream);
    uint16_t* p2_16 = p1_16 + (size_t) R * ol * rk;
    f16_copy(t1, R * inn * ol, p2_16, stream);
    gemm.f16(p2_16, ad->w1t, t2, R * inn, im, ol);
    lokr_dgrad_add(dxv, t2, dyrs, s, R, K, im, inn, stream);
    // the whole delta's gradient, then each factor through its partial trace (w2 splits as a product does)
    uint32_t* g = scale_word(0, stream);
    scaled_transpose_f16(dYv, rsx, R, N, t16t, g, stream);
    float* dDelta = sums;
    wgrad(t16t, xt, R, N, K, g, s, dDelta, sums + (size_t) N * K, false, gemm, stream);
    float* w2f = t1;
    gemm.f16(ad->w2a, ad->w2b, w2f, ok, inn, rk);
    uint16_t* w2_16 = p2_16 + (size_t) R * inn * ol;
    float* w2rs = dyrs;
    dit_rows::rows16(dit_rows::Plain{w2f, inn}, ok, inn, w2_16, w2rs, stream);
    float* dw1 = t2;
    float* dw2 = t2 + (size_t) ol * im;
    lokr_partial_trace(dDelta, ad->w1, w2_16, ol, ok, im, inn, dw1, dw2, stream);
    add_into(ad->dW1, dw1, ol * im, stream);
    float* dw2a = dw2 + (size_t) ok * inn;
    float* dw2b = dw2a + (size_t) ok * rk;
    lokr_factor_bwd(dw2, ad->w2a, ad->w2b, ok, inn, rk, dw2a, dw2b, stream);
    add_into(ad->dW2a, dw2a, ok * rk, stream);
    add_into(ad->dW2b, dw2b, rk * inn, stream);
}

// a DoRA's row factors (the same formula dora_bwd's da/db read: c = m / (|W| + eps), fa = 1 + s (c - 1),
// fb = s c, with kLoraDoraEps as image_lora.cpp takes it)
void ad_factors(const float* m, const float* norms, int64_t N, float scale, float* fa, float* fb, void* s) {
    constexpr float kDoraEps = 1.1920929e-07f;
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) N), [=](sycl::id<1> id) {
        const int64_t n = static_cast<int64_t>(id[0]);
        const float c = m[n] / (norms[n] + kDoraEps);
        fa[n] = 1.f + scale * (c - 1.f);
        fb[n] = scale * c;
    });
}

}  // namespace

void ad_product_fwd(const Adapter* ad, const uint16_t* x16, const float* rsx, float* y, const float* bias, int64_t R,
                    int64_t N, int64_t K, const AdKeep& keep, const AdScratch& sc, prefill::Gemm& gemm,
                    void* stream) {
    float* hf = sc.hf;
    uint16_t* h16 = keep.h16;
    float* hr = keep.hr;
    uint16_t* h16t = keep.h16t;
    float *delta = sc.y1, *delta2 = sc.y2, *fa = sc.fa, *fb = sc.fb;
    uint16_t* tmp16 = sc.tmp16;
    if (ad == nullptr) {
        dit_rows::rows_back(y, rsx, bias, R, N, stream);
        return;
    }
    if (ad->method == AdapterMethod::lokr) {
        lokr_product_fwd(ad, x16, rsx, y, bias, R, N, delta, delta2, tmp16, gemm, stream);
        return;
    }
    gemm.f16(x16, ad->a, hf, R, ad->rank, K);
    dit_rows::rows16(dit_rows::Plain{hf, ad->rank}, R, ad->rank, h16, hr, stream);
    // hf read x16, the input divided by rsx: hr takes rsx too, so h16 * hr is x A^T itself for the delta below,
    // dB (lora_bwd) and a DoRA's dm (without it the delta trained 1 / rsx of what inference applies)
    if (rsx != nullptr)
        dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) R),
                                            [=](sycl::id<1> id) { hr[id[0]] *= rsx[id[0]]; });
    if (!gemm.f16_batch_ex_ok()) transpose_f16(h16, R, ad->rank, h16t, stream);   // lora_bwd's dB reads it
    gemm.f16(h16, ad->b, delta, R, N, ad->rank);
    const float* fa_use = nullptr;
    const float* fb_use = nullptr;
    if (ad->method == AdapterMethod::dora) {
        ad_factors(ad->m, ad->norms, N, ad->scale, fa, fb, stream);
        fa_use = fa;
        fb_use = fb;
    }
    ad_rows_back(y, rsx, bias, delta, hr, fa_use, fb_use, ad->scale, R, N, stream);
}

bool ad_needs_xt(const Product& P, const Adapter* ad, prefill::Gemm& gemm) {
    return P.dw != nullptr || (ad != nullptr && (ad->method == AdapterMethod::lokr || !gemm.f16_batch_ex_ok()));
}

namespace {
StageTimes& ad_times() {
    static StageTimes t("ad_product_bwd");
    return t;
}
}  // namespace

void ad_profile_report() { ad_times().report(); }

void ad_product_bwd(const Product& P, const Adapter* ad, const uint16_t* x16, const float* rsx, const uint16_t* xt,
                    float* dYv, float* dxv, bool accumulate, const AdKeep& keep, const AdScratch& sc, uint16_t* t16,
                    float* trs, uint16_t* t16t, float* sums, int64_t R, prefill::Gemm& gemm, void* stream) {
    const int64_t N = P.N, K = P.K;
    uint16_t* h16 = keep.h16;
    float* hr = keep.hr;
    uint16_t* h16t = keep.h16t;
    float *yb = sc.y1, *yd = sc.y2, *fa = sc.fa, *fb = sc.fb, *lscratch = sc.scratch, *lrsT = sc.rsT;
    uint16_t* ltmp16 = sc.tmp16;
    const float* dYa = dYv;
    const float* dYb = dYv;
    if (ad != nullptr && ad->method == AdapterMethod::dora) {
        const float* pb = nullptr;
        const float* db = nullptr;
        if (ad->dm != nullptr) {   // only a DoRA that trains its magnitude reads the rows back
            if (P.w == nullptr) throw std::invalid_argument("image engine: a DoRA's magnitude on an int8 base");
            gemm.f16(x16, P.w, yb, R, N, K);
            dit_rows::rows_back(yb, rsx, nullptr, R, N, stream);
            gemm.f16(h16, ad->b, yd, R, N, ad->rank);
            dit_rows::rows_back(yd, hr, nullptr, R, N, stream);
            pb = yb;
            db = yd;
        }
        dora_bwd(dYv, pb, db, ad->m, ad->norms, R, N, ad->scale, ad->dm, fa, fb, stream);
        copy_rows(dYv, R * N, yb, stream);   // yb is dead after dm: dYb = dY o fb lives here
        scale_cols(yb, fb, R, N, stream);
        copy_rows(dYv, R * N, yd, stream);   // yd is dead after dm, too: dYa = dY o fa lives here.  dYv itself
        scale_cols(yd, fa, R, N, stream);   // stays pristine for the bias below: b is added after the scaling,
        dYa = yd;                             // so dL/db = sum dY, not sum dYa
        dYb = yb;
    }
    StageTimes& tm = ad_times();
    const char* const tag = N > K ? "N>K" : N < K ? "N<K" : "N=K";
    tm.begin(stream);
    dit_rows::rows16(dit_rows::Grad{{dYa, N}}, R, N, t16, trs, stream);
    tm.mark("grad rows16", stream);
    if (P.wt == nullptr && P.wr != nullptr) dgrad_rows(t16, trs, P.wr, R, N, K, dxv, sums, accumulate, gemm, stream);
    else dgrad(t16, trs, P.wt, R, N, K, dxv, sums, accumulate, gemm, stream);
    tm.mark(tag[1] == '<' ? "base dgrad (N<K)" : tag[1] == '>' ? "base dgrad (N>K)" : "base dgrad (N=K)", stream);
    if (P.dw != nullptr) {
        uint32_t* gi = scale_word(0, stream);
        scaled_transpose_f16(dYa, rsx, R, N, t16t, gi, stream);
        wgrad(t16t, xt, R, N, K, gi, 1.f, P.dw, sums, true, gemm, stream);
    }
    if (P.db != nullptr) bias_bwd(dYv, R, N, P.db, stream);   // the bias is added after
        // the row scaling, so its gradient reads the unscaled rows (dYv, not dYa; the same buffer for a plain LoRA)
    if (ad != nullptr) {
        if (ad->method == AdapterMethod::lokr) {
            lokr_product_bwd(ad, xt, rsx, dYv, dxv, yb, yd, ltmp16, lrsT, t16t, sums, R, N, K, gemm, stream);
        } else {
        // a plain LoRA's dY carries no scale, so lora_bwd takes scale; a DoRA's dYb already carries its row
        // factor fb = scale * c, so it takes 1 (its dh is dL/dh either way: dh = scale * dYb B)
        const float ad_scale = ad->method == AdapterMethod::dora ? 1.0f : ad->scale;
        lora_bwd(hr, rsx, h16t, xt, ad->bt, dYb, R, N, K, ad->rank, ad_scale, ad->dB, ad->dA, lscratch,
                    ltmp16, lrsT, gemm, stream, dYb == dYa ? t16 : nullptr, dYb == dYa ? trs : nullptr, h16, x16,
                    ad->b);
        tm.mark(tag[1] == '<' ? "lora_bwd (N<K)" : tag[1] == '>' ? "lora_bwd (N>K)" : "lora_bwd (N=K)", stream);
        // dX += dh A in FP32 (dh lives in lscratch): the input gradient's dX is judged at 3e-5, which an FP16
        // rounding of dh alone costs; two FP16 products of its split took a sixth of the backward's products on the
        // B70 and the RTX 4070 at rank 32 (2026-11-10, MOSS-Image), this one pass reads A once a tile of rows
        lora_dx(lscratch, ad->a, R, ad->rank, K, dxv, stream);
        tm.mark("lora dX", stream);
        }
    }
}

}  // namespace strata::image::train
