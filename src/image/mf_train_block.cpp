// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_train_block.cpp - see include/strata/image/mf_train.hpp's mf_block.  One Mage-Flow block forwards with a
// tape and backwards through it, made of the per-op gradients of mf_train.cpp, so a torch reference can check the whole
// block and not only its parts.
//
// Every buffer has one role and one owner: the tape holds what the backward reads, and the gradients live beside it.
// Nothing is aliased between a product's output and another one's gradient - the block is small and the GPU has room.
#include "strata/image/mf_train.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

constexpr float kEps = 1e-6f;    // the LayerNorm's eps, as mf_dit.cpp takes it
constexpr int64_t kWG = 256;
using kernels::f16_from_f32;
using kernels::f32_from_f16;

// out = x + gate * h
void resid_gate(const float* x, const float* h, const float* gate, int64_t n, int64_t C, float* out, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]);
        out[i] = x[i] + gate[i % C] * h[i];
    });
}

// out[i] = a[i] * col[i % C]
void scale_cols(const float* a, const float* col, int64_t n, int64_t C, float* out, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]);
        out[i] = a[i] * (col != nullptr ? col[i % C] : 1.f);
    });
}

// x = hi + lo to 2^-22, for an FP32 map the products read twice (the input gradient's dX is judged at 3e-5,
// and an FP16 rounding of dh alone costs that budget)
void ad_split16(const float* x, int64_t n, uint16_t* hi, uint16_t* lo, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = x[id[0]];
        const uint16_t h = kernels::f16_from_f32(v);
        hi[id[0]] = h;
        lo[id[0]] = kernels::f16_from_f32(v - kernels::f32_from_f16(h));
    });
}

void copy_rows(const float* a, int64_t n, float* out, void* s) {    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { out[id[0]] = a[id[0]]; });
}

void add_into(float* a, const float* b, int64_t n, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { a[id[0]] += b[id[0]]; });
}

// ---- adapters (MfBlockW::ad): the forward keeps h16/hr/h16t, the backward reads them back ----

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

// one product's forward rows with an adapter (ad null: the product as always): the base GEMM's output is in y
// (still divided by rsx, no bias yet).  hf/delta/fa/fb are shared transients; h16/hr/h16t are kept for the backward.
void ad_factors(const float* m, const float* norms, int64_t N, float scale, float* fa, float* fb, void* s);
void ad_rows_back(float* y, const float* rsx, const float* bias, const float* delta, const float* hr,
                  const float* fa, const float* fb, float scale, int64_t R, int64_t N, void* s);
void ad_product_fwd(const MfAdapter* ad, const uint16_t* x16, const float* rsx, float* y, const float* bias,
                    int64_t R, int64_t N, int64_t K, float* hf, uint16_t* h16, float* hr, uint16_t* h16t,
                    float* delta, float* fa, float* fb, prefill::Gemm& gemm, void* stream) {
    if (ad == nullptr) {
        dit_rows::rows_back(y, rsx, bias, R, N, stream);
        return;
    }
    gemm.f16(x16, ad->a, hf, R, ad->rank, K);
    mf_rows16(hf, R, ad->rank, h16, hr, stream);
    mf_transpose_f16(h16, R, ad->rank, h16t, stream);
    gemm.f16(h16, ad->b, delta, R, N, ad->rank);
    const float* fa_use = nullptr;
    const float* fb_use = nullptr;
    if (ad->method == MfAdapterMethod::dora) {
        ad_factors(ad->m, ad->norms, N, ad->scale, fa, fb, stream);
        fa_use = fa;
        fb_use = fb;
    }
    ad_rows_back(y, rsx, bias, delta, hr, fa_use, fb_use, ad->scale, R, N, stream);
}

// one product's backward: dYv is the gradient reaching the product's rows (consumed, scaled in place for a DoRA),
// dxv the input's gradient (overwritten when first, added into after).  t16/trs/t16t/sums are the block's own temps;
// h16/hr/h16t the forward's kept rows (h16 is dead after a dm read and takes dh's rows16); yb/yd/fa/fb and
// lscratch/ltmp16/lrsT shared adapter transients, sized for the largest adapted product.
void ad_product_bwd(const MfBlockW& W, int pi, const MfAdapter* ad, const uint16_t* x16, const float* rsx,
                    const uint16_t* xt, float* dYv, float* dxv, bool first, uint16_t* h16, float* hr,
                    uint16_t* h16t, float* yb, float* yd, float* fa, float* fb, float* lscratch,
                    uint16_t* ltmp16, float* lrsT, uint16_t* t16, float* trs, uint16_t* t16t, float* sums,
                    int64_t R, prefill::Gemm& gemm, void* stream) {
    const int64_t N = W.N[pi], K = W.K[pi];
    const float* dYa = dYv;
    const float* dYb = dYv;
    if (ad != nullptr && ad->method == MfAdapterMethod::dora) {
        const float* pb = nullptr;
        const float* db = nullptr;
        if (ad->dm != nullptr) {   // only a DoRA that trains its magnitude reads the rows back
            gemm.f16(x16, W.w[pi], yb, R, N, K);
            dit_rows::rows_back(yb, rsx, nullptr, R, N, stream);
            gemm.f16(h16, ad->b, yd, R, N, ad->rank);
            dit_rows::rows_back(yd, hr, nullptr, R, N, stream);
            pb = yb;
            db = yd;
        }
        mf_dora_bwd(dYv, pb, db, ad->m, ad->norms, R, N, ad->scale, ad->dm, fa, fb, stream);
        copy_rows(dYv, R * N, yb, stream);   // yb is dead after dm: dYb = dY o fb lives here
        mf_scale_rows(yb, fb, R, N, stream);
        copy_rows(dYv, R * N, yd, stream);   // yd is dead after dm, too: dYa = dY o fa lives here.  dYv itself
        mf_scale_rows(yd, fa, R, N, stream);   // stays pristine for the bias below: b is added after the scaling,
        dYa = yd;                             // so dL/db = sum dY, not sum dYa
        dYb = yb;
    }
    mf_rows16(dYa, R, N, t16, trs, stream);
    mf_dgrad(t16, trs, W.wt[pi], R, N, K, dxv, sums, first, gemm, stream);
    if (W.dw[pi] != nullptr) {
        const float gi = mf_scaled_transpose_f16(dYa, rsx, R, N, t16t, stream);
        mf_wgrad(t16t, xt, R, N, K, gi, W.dw[pi], sums, true, gemm, stream);
    }
    if (W.db[pi] != nullptr) mf_bias_bwd(dYv, R, N, W.db[pi], stream);   // the bias is added after
        // the row scaling, so its gradient reads the unscaled rows (dYv, not dYa; the same buffer for a plain LoRA)
    if (ad != nullptr) {
        // a plain LoRA's dY carries no scale, so mf_lora_bwd takes scale; a DoRA's dYb already carries its row
        // factor fb = scale * c, so it takes 1 (its dh is dL/dh either way: dh = scale * dYb B)
        const float ad_scale = ad->method == MfAdapterMethod::dora ? 1.0f : ad->scale;
        mf_lora_bwd(hr, rsx, h16t, xt, ad->bt, dYb, R, N, K, ad->rank, ad_scale, ad->dB, ad->dA, lscratch,
                    ltmp16, lrsT, gemm, stream);
        // dh lives in lscratch; h16 and h16t are dead and take its split (the input gradient's dX is judged
        // at 3e-5, and an FP16 rounding of dh alone costs that budget)
        ad_split16(lscratch, R * ad->rank, h16, reinterpret_cast<uint16_t*>(h16t), stream);
        mf_dgrad(h16, nullptr, ad->at, R, ad->rank, K, dxv, sums, true, gemm, stream);
        mf_dgrad(reinterpret_cast<uint16_t*>(h16t), nullptr, ad->at, R, ad->rank, K, dxv, sums, true,
                 gemm, stream);
    }
}
// a DoRA's row factors (the same formula mf_dora_bwd's da/db read: c = m / (|W| + eps), fa = 1 + s (c - 1),
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

// one stage of the forward out to the parity harness's tape: null (no tape) pays nothing
void tape_put(float* dst, const float* src, int64_t n, void* s) {
    if (dst == nullptr) return;
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n),
                                   [=](sycl::id<1> id) { dst[id[0]] = src[id[0]]; });
}

void tape_put16(float* dst, const uint16_t* src, int64_t n, void* s) {
    if (dst == nullptr) return;
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n),
                                   [=](sycl::id<1> id) { dst[id[0]] = f32_from_f16(src[id[0]]); });
}

// LayerNorm(no affine, eps 1e-6) * (1 + scale) + shift, as a product's FP16 input rows, with the statistics
void ln_mod16(const float* x, const float* shift, const float* scale, int64_t rows, int64_t C, uint16_t* y16,
              float* rs, float* mean, float* inv, void* s) {
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t t = static_cast<int64_t>(it.get_group_linear_id());
            const float* xr = x + t * C;
            float sum = 0.f, va = 0.f, m = 0.f;
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < C; c += kWG) sum += xr[c];
            const float mn = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>()) / (float) C;
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < C; c += kWG)
                va += (xr[c] - mn) * (xr[c] - mn);
            const float iv =
                1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), va, sycl::plus<float>()) / (float) C + kEps);
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < C; c += kWG)
                m = sycl::fmax(m, sycl::fabs((xr[c] - mn) * iv * (1.f + scale[c]) + shift[c]));
            const float sc = dit_rows::pow2_scale(sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>()));
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < C; c += kWG)
                y16[t * C + c] = f16_from_f32(((xr[c] - mn) * iv * (1.f + scale[c]) + shift[c]) / sc);
            if (it.get_local_linear_id() == 0) {
                mean[t] = mn;
                inv[t] = iv;
                rs[t] = sc;
            }
        });
}

// each head's q and k RMS-normed with its weight and rotated pairwise; v carried as it is (qk_rope_v16's forward)
void qk_norm_rope16(const float* q, const float* k, const float* v, const float* wq, const float* wk,
                    const float* cos, const float* sin, bool rope, int64_t rows, int64_t heads, int64_t hd,
                    uint16_t* q16, uint16_t* k16, uint16_t* v16, void* s) {
    const int64_t half = hd / 2;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * heads * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t g = static_cast<int64_t>(it.get_group_linear_id()), t = g / heads;
            for (int which = 0; which < 3; ++which) {
                const float* src = (which == 0 ? q : which == 1 ? k : v) + g * hd;
                uint16_t* dst = (which == 0 ? q16 : which == 1 ? k16 : v16) + g * hd;
                if (which == 2) {
                    for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += kWG)
                        dst[d] = f16_from_f32(src[d]);
                    continue;
                }
                const float* w = which == 0 ? wq : wk;
                float ss = 0.f;
                for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += kWG)
                    ss += src[d] * src[d];
                const float r = 1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) /
                                                      (float) hd + kEps);
                for (auto j = static_cast<int64_t>(it.get_local_linear_id()); j < half; j += kWG) {
                    const float a = src[2 * j] * r * w[2 * j];
                    const float b = src[2 * j + 1] * r * w[2 * j + 1];
                    const float c = rope ? cos[t * half + j] : 1.f, sn = rope ? sin[t * half + j] : 0.f;
                    dst[2 * j] = f16_from_f32(a * c - b * sn);
                    dst[2 * j + 1] = f16_from_f32(a * sn + b * c);
                }
            }
        });
}

void gelu_fwd(const float* x, int64_t n, float* y, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float z = x[id[0]];
        y[id[0]] = 0.5f * z * (1.f + sycl::tanh(0.7978845608028654f * (z + 0.044715f * z * z * z)));
    });
}

/// device memory for one block's tape and gradients, all of it freed with the call
class Pool {
public:
    explicit Pool(void* s) : s_(s) {}
    // the zero goes on the stream the block's kernels run on: the compute queue's own memset is not ordered against
    // them, so a buffer allocated here could be wiped AFTER a kernel had already written it (the late zero is what a
    // gradient that reads as exactly zero, and a forward a few percent off, looked like)
    float* f(int64_t n) { return static_cast<float*>(take(n, 4)); }
    uint16_t* u(int64_t n) { return static_cast<uint16_t*>(take(n, 2)); }
    ~Pool() {
        strata::gpu::stream_sync(s_);
        for (void* p : owned_) gpu::free(p);
    }

private:
    void* take(int64_t n, int64_t bytes) {
        void* p = nullptr;
        if (!gpu::alloc_device(&p, (size_t) n * (size_t) bytes)) throw std::runtime_error("mf_block: GPU memory");
        if (!gpu::memset_async(p, 0, (size_t) n * (size_t) bytes, s_)) throw std::runtime_error("mf_block: zero");
        owned_.push_back(p);
        return p;
    }

    void* s_;
    std::vector<void*> owned_;
};

/// The tape's layout, in MfTapeSlice's order: one stream's stages first (the image's, then the text's), then the
/// joint rows and the log-sum-exp under them.  Nine of a stream's eleven slices are C wide; m1 and the GELU's rows
/// (indices 7 and 8) are F wide.
int64_t tape_stream_elems(int64_t rows, int64_t C, int64_t F) { return rows * (17 * C + 4 * F); }

/// an F-wide slice is mlp1's own or the GELU rows', forward or back
bool tape_is_f(int which) { return which == 7 || which == 8 || which == 18 || which == 19; }

int64_t tape_slice_elems(int which, int64_t rows, int64_t C, int64_t F) { return rows * (tape_is_f(which) ? F : C); }

/// one stream's slice
int64_t tape_off(int st, int which, int64_t Ri, int64_t Rt, int64_t C, int64_t F) {
    const int64_t rows = st == 0 ? Ri : Rt;
    int64_t o = st == 0 ? 0 : tape_stream_elems(Ri, C, F);
    for (int i = 0; i < which; ++i) o += tape_slice_elems(i, rows, C, F);
    return o;
}

}  // namespace

int64_t mf_block_tape_elems(const MfBlockRun& run) {
    const int64_t Rj = run.rows_img + run.rows_txt;
    return tape_stream_elems(run.rows_img, run.C, run.F) + tape_stream_elems(run.rows_txt, run.C, run.F) +
           kTapeLse * Rj * run.C + Rj * run.heads          // the eight row sets and the statistics
           + 3 * Rj;                                      // and the six scale sets: three pairs
}


void mf_block(const MfBlockRun& run, prefill::Gemm& gemm, void* stream) {
    const int64_t C = run.C, F = run.F, heads = run.heads, hd = run.hd, Ri = run.rows_img, Rt = run.rows_txt;
    const int64_t Rj = Ri + Rt, R[2] = {Ri, Rt};
    const float* in[2] = {run.img, run.txt};
    const MfBlockW* w[2] = {&run.img_w, &run.txt_w};
    float* outp[2] = {run.out_img, run.out_txt};
    const float* dout[2] = {run.d_out_img, run.d_out_txt};
    float* dIn[2] = {run.dimg, run.dtxt};
    if (C % heads != 0 || C / heads != hd) throw std::invalid_argument("mf_block: heads * head_dim must be the width");
    Pool pool(stream);

    const int64_t nmax = std::max(C, F);   // the wider of the block's two widths: a buffer reused between roles takes it

    // ---- the tape: one buffer per value the backward reads
    uint16_t* ln1_16[2];      // the first norm's FP16 rows (q, k and v read them)
    float* ln1_s[2];
    uint16_t* ln1_t[2];      // the same rows transposed ([C][rows])
    float* mean1[2];
    float* inv1[2];
    float* yq[2];
    float* yk[2];
    float* yv[2];
    float* yo[2];
    float* h1[2];            // the first residual's output, which the second norm reads
    uint16_t* ln2_16[2];
    float* ln2_s[2];
    uint16_t* ln2_t[2];
    float* mean2[2];
    float* inv2[2];
    float* ym1[2];           // mlp1's output, before the GELU
    uint16_t* gelu_16[2];    // the GELU's output as mlp2's input rows
    float* gelu_s[2];
    uint16_t* gelu_t[2];
    float* ym2[2];
    uint16_t *jq = pool.u(Rj * C), *jk = pool.u(Rj * C), *jv = pool.u(Rj * C), *jo = pool.u(Rj * C);
    float* jof = pool.f(Rj * C);   // the same output before it was rounded: the backward's D reads this
    float* lse = pool.f(Rj * heads);
    // ---- the gradients
    float* dym2[2];
    float* dgelu[2];
    float* dym1[2];
    float* dln2[2];
    float* dh1[2];
    float* dyo[2];
    float* dattn[2];
    uint16_t* attn_t[2];   // the attention's FP16 rows transposed ([C][rows])
    float* jdq = pool.f(Rj * C), *jdk = pool.f(Rj * C), *jdv = pool.f(Rj * C), *jdo = pool.f(Rj * C);
    float* dyq[2];
    float* dyk[2];
    float* dyv[2];
    float* dln1[2];
    float* dxn1[2];
    for (int st = 0; st < 2; ++st) {
        ln1_16[st] = pool.u(R[st] * C);
        ln1_s[st] = pool.f(R[st]);
        ln1_t[st] = pool.u(R[st] * C);
        mean1[st] = pool.f(R[st]);
        inv1[st] = pool.f(R[st]);
        yq[st] = pool.f(R[st] * C);
        yk[st] = pool.f(R[st] * C);
        yv[st] = pool.f(R[st] * C);
        yo[st] = pool.f(R[st] * C);
        h1[st] = pool.f(R[st] * C);
        ln2_16[st] = pool.u(R[st] * C);
        ln2_s[st] = pool.f(R[st]);
        ln2_t[st] = pool.u(R[st] * C);
        mean2[st] = pool.f(R[st]);
        inv2[st] = pool.f(R[st]);
        ym1[st] = pool.f(R[st] * F);
        gelu_16[st] = pool.u(R[st] * F);
        gelu_s[st] = pool.f(R[st]);
        gelu_t[st] = pool.u(R[st] * F);
        ym2[st] = pool.f(R[st] * C);
        dym2[st] = pool.f(R[st] * C);
        dgelu[st] = pool.f(R[st] * nmax);   // the GELU's output, then a dgrad's dx: the wider of the two
        dym1[st] = pool.f(R[st] * F);
        dln2[st] = pool.f(R[st] * C);
        dh1[st] = pool.f(R[st] * C);
        dyo[st] = pool.f(R[st] * C);
        dattn[st] = pool.f(R[st] * C);
        attn_t[st] = pool.u(R[st] * C);
        dyq[st] = pool.f(R[st] * C);
        dyk[st] = pool.f(R[st] * C);
        dyv[st] = pool.f(R[st] * C);
        dln1[st] = pool.f(R[st] * C);
        dxn1[st] = pool.f(R[st] * C);
    }
    uint16_t* t16 = pool.u(Rj * nmax), *t16t = pool.u(Rj * nmax);
    float* trs = pool.f(Rj);
    // a wgrad's sums are N*K wide (at most F*C, never the square of the wider one) and a dgrad's T*K (at most rows*C
    // or rows*F), so this is the scratch the block's eight products can actually need
    float* sums = pool.f(C * F + Rj * nmax);

    // ---- adapters: the forward keeps h16/hr/h16t per adapted product, the backward reads them back
    uint16_t* adh16[2][6] = {};
    float* adhr[2][6] = {};
    uint16_t* adh16t[2][6] = {};
    int64_t ad_maxR = 0, ad_maxN = 0, ad_maxRank = 0, ad_maxScratch = 0, ad_maxTmp = 0;
    for (int st = 0; st < 2; ++st)
        for (int i = 0; i < 6; ++i) {
            const MfAdapter* az = w[st]->ad[i];
            if (az == nullptr) continue;
            const int64_t T = R[st], N = w[st]->N[i], K = w[st]->K[i], rk = az->rank;
            ad_maxR = std::max(ad_maxR, T);
            ad_maxN = std::max(ad_maxN, N);
            ad_maxRank = std::max(ad_maxRank, rk);
            // mf_lora_bwd's scratch (dh plus the largest of its three sums) and its three transposed FP16 maps
            ad_maxScratch = std::max(ad_maxScratch, T * rk + T * N + N * rk + rk * K);
            ad_maxTmp = std::max(ad_maxTmp, 2 * T * N + T * rk);
            adh16[st][i] = pool.u(T * rk);
            adhr[st][i] = pool.f(T);
            adh16t[st][i] = pool.u(T * rk);
        }
    // shared transients, sized for the largest adapted product (products run one at a time): hf is h = x A^T, y1 the
    // delta then yb/yd/dYb, y2 the second map a DoRA that trains its magnitude reads beside the first
    const bool ad_any = ad_maxN > 0;
    float* ad_hf = ad_any ? pool.f(ad_maxR * ad_maxRank) : nullptr;
    float* ad_y1 = ad_any ? pool.f(ad_maxR * ad_maxN) : nullptr;
    float* ad_y2 = ad_any ? pool.f(ad_maxR * ad_maxN) : nullptr;
    float* ad_fa = ad_any ? pool.f(ad_maxN) : nullptr;
    float* ad_fb = ad_any ? pool.f(ad_maxN) : nullptr;
    float* ad_scratch = ad_any ? pool.f(ad_maxScratch) : nullptr;
    uint16_t* ad_tmp16 = ad_any ? pool.u(ad_maxTmp) : nullptr;
    float* ad_rsT = ad_any ? pool.f(ad_maxR) : nullptr;

    // ---- forward, every stage handed to the tape when the harness asked for one (null: nothing copied)
    float* tp = run.tape;
    auto slice = [&](int st, int which) -> float* {
        return tp == nullptr ? nullptr : tp + tape_off(st, which, Ri, Rt, C, F);
    };
    const int64_t jbase = tape_stream_elems(Ri, C, F) + tape_stream_elems(Rt, C, F);
    auto jslice = [&](int which) -> float* {   // one slot a row set, the log-sum-exp last (`heads` wide, still one C)
        return tp == nullptr ? nullptr : tp + jbase + (int64_t)which * Rj * C;
    };
    for (int st = 0; st < 2; ++st) {
        const float* mod = w[st]->mod;
        const int64_t off = st == 0 ? Rt : 0;   // the joint rows: the text's first, then the image's
        ln_mod16(in[st], mod, mod + C, R[st], C, ln1_16[st], ln1_s[st], mean1[st], inv1[st], stream);
        tape_put16(slice(st, kTapeLn1), ln1_16[st], R[st] * C, stream);
        mf_transpose_f16(ln1_16[st], R[st], C, ln1_t[st], stream);
        for (int i = 0; i < 3; ++i) {
            float* dst = i == 0 ? yq[st] : i == 1 ? yk[st] : yv[st];
            gemm.f16(ln1_16[st], w[st]->w[i], dst, R[st], C, C);
            ad_product_fwd(w[st]->ad[i], ln1_16[st], ln1_s[st], dst, w[st]->b[i], R[st], C, C, ad_hf,
                           adh16[st][i], adhr[st][i], adh16t[st][i], ad_y1, ad_fa, ad_fb, gemm, stream);
            tape_put(slice(st, kTapeQ + i), dst, R[st] * C, stream);
        }
        qk_norm_rope16(yq[st], yk[st], yv[st], w[st]->nq, w[st]->nk, st == 0 ? run.cos : nullptr,
                       st == 0 ? run.sin : nullptr, st == 0 && run.cos != nullptr, R[st], heads, hd,
                       jq + off * C, jk + off * C, jv + off * C, stream);
    }
    mf_attn_fwd(jq, jk, jv, Rj, heads, hd, jo, jof, lse, gemm, stream);
    tape_put16(jslice(kTapeJQ), jq, Rj * C, stream);
    tape_put16(jslice(kTapeJK), jk, Rj * C, stream);
    tape_put16(jslice(kTapeJV), jv, Rj * C, stream);
    tape_put16(jslice(kTapeJO), jo, Rj * C, stream);
    tape_put(jslice(kTapeLse), lse, Rj * heads, stream);
    for (int st = 0; st < 2; ++st) {
        const float* mod = w[st]->mod;
        const int64_t off = st == 0 ? Rt : 0;
        gemm.f16(jo + off * C, w[st]->w[3], yo[st], R[st], C, C);
        ad_product_fwd(w[st]->ad[3], jo + off * C, nullptr, yo[st], w[st]->b[3], R[st], C, C, ad_hf,
                       adh16[st][3], adhr[st][3], adh16t[st][3], ad_y1, ad_fa, ad_fb, gemm, stream);
        tape_put(slice(st, kTapeO), yo[st], R[st] * C, stream);
        resid_gate(in[st], yo[st], mod + 2 * C, R[st] * C, C, h1[st], stream);
        tape_put(slice(st, kTapeH1), h1[st], R[st] * C, stream);
        ln_mod16(h1[st], mod + 3 * C, mod + 4 * C, R[st], C, ln2_16[st], ln2_s[st], mean2[st], inv2[st], stream);
        tape_put16(slice(st, kTapeLn2), ln2_16[st], R[st] * C, stream);
        mf_transpose_f16(ln2_16[st], R[st], C, ln2_t[st], stream);
        gemm.f16(ln2_16[st], w[st]->w[4], ym1[st], R[st], F, C);
        ad_product_fwd(w[st]->ad[4], ln2_16[st], ln2_s[st], ym1[st], w[st]->b[4], R[st], F, C, ad_hf,
                       adh16[st][4], adhr[st][4], adh16t[st][4], ad_y1, ad_fa, ad_fb, gemm, stream);
        tape_put(slice(st, kTapeM1), ym1[st], R[st] * F, stream);
        gelu_fwd(ym1[st], R[st] * F, dgelu[st], stream);           // dgelu still free here: it holds the GELU's output
        mf_rows16(dgelu[st], R[st], F, gelu_16[st], gelu_s[st], stream);
        tape_put16(slice(st, kTapeGelu), gelu_16[st], R[st] * F, stream);
        mf_transpose_f16(gelu_16[st], R[st], F, gelu_t[st], stream);
        gemm.f16(gelu_16[st], w[st]->w[5], ym2[st], R[st], C, F);
        ad_product_fwd(w[st]->ad[5], gelu_16[st], gelu_s[st], ym2[st], w[st]->b[5], R[st], C, F, ad_hf,
                       adh16[st][5], adhr[st][5], adh16t[st][5], ad_y1, ad_fa, ad_fb, gemm, stream);
        tape_put(slice(st, kTapeM2), ym2[st], R[st] * C, stream);
        resid_gate(h1[st], ym2[st], mod + 5 * C, R[st] * C, C, outp[st], stream);
        tape_put(slice(st, kTapeOut), outp[st], R[st] * C, stream);
    }
    if (tp != nullptr) {   // the rows' power-of-two scales, the products' other input: these exist from here on, and
                           // a product that reads as its own bias is a zero weight or a zero scale - this tells them apart
        const int64_t so = jbase + kTapeLse * Rj * C + Rj * heads;
        const int64_t slot[kTapeScaleN] = {so, so + Ri, so + Rj, so + Rj + Ri, so + 2 * Rj, so + 2 * Rj + Ri};
        const int64_t wide[2] = {Ri, Rt};
        tape_put(tp + slot[kTapeScLn1I], ln1_s[0], wide[0], stream);
        tape_put(tp + slot[kTapeScLn1T], ln1_s[1], wide[1], stream);
        tape_put(tp + slot[kTapeScLn2I], ln2_s[0], wide[0], stream);
        tape_put(tp + slot[kTapeScLn2T], ln2_s[1], wide[1], stream);
        tape_put(tp + slot[kTapeScGeluI], gelu_s[0], wide[0], stream);
        tape_put(tp + slot[kTapeScGeluT], gelu_s[1], wide[1], stream);
    }

    // ---- backward, from the outputs' gradients
    for (int st = 0; st < 2; ++st) {
        const MfBlockW& W = *w[st];
        const float* mod = W.mod;
        // the MLP's gated residual: dh1 is the output's gradient as it reaches the first residual's sum
        copy_rows(dout[st], R[st] * C, dh1[st], stream);
        scale_cols(dout[st], mod + 5 * C, R[st] * C, C, dym2[st], stream);
        tape_put(slice(st, kTapeDM2), dym2[st], R[st] * C, stream);
        ad_product_bwd(W, 5, W.ad[5], gelu_16[st], gelu_s[st], gelu_t[st], dym2[st], dgelu[st], false,
                       adh16[st][5], adhr[st][5], adh16t[st][5], ad_y1, ad_y2, ad_fa, ad_fb, ad_scratch,
                       ad_tmp16, ad_rsT, t16, trs, t16t, sums, R[st], gemm, stream);
        tape_put(slice(st, kTapeDGelu), dgelu[st], R[st] * F, stream);
        mf_gelu_bwd(ym1[st], dgelu[st], R[st] * F, dym1[st], stream);
        tape_put(slice(st, kTapeDM1), dym1[st], R[st] * F, stream);
        ad_product_bwd(W, 4, W.ad[4], ln2_16[st], ln2_s[st], ln2_t[st], dym1[st], dln2[st], false,
                       adh16[st][4], adhr[st][4], adh16t[st][4], ad_y1, ad_y2, ad_fa, ad_fb, ad_scratch,
                       ad_tmp16, ad_rsT, t16, trs, t16t, sums, R[st], gemm, stream);
        tape_put(slice(st, kTapeDLn2), dln2[st], R[st] * C, stream);
        // the second norm: its input's gradient joins the residual's own
        mf_ln_mod_bwd(h1[st], mean2[st], inv2[st], mod + 4 * C, dln2[st], R[st], C, dgelu[st], nullptr, nullptr,
                      stream);
        add_into(dh1[st], dgelu[st], R[st] * C, stream);
        tape_put(slice(st, kTapeDH1), dh1[st], R[st] * C, stream);
        // the attention's gated residual
        scale_cols(dh1[st], mod + 2 * C, R[st] * C, C, dyo[st], stream);
        tape_put(slice(st, kTapeDYo), dyo[st], R[st] * C, stream);
        if (dIn[st] != nullptr) add_into(dIn[st], dh1[st], R[st] * C, stream);   // the residual's own path
        const int64_t off = st == 0 ? Rt : 0;
        mf_transpose_f16(jo + off * C, R[st], C, attn_t[st], stream);
        ad_product_bwd(W, 3, W.ad[3], jo + off * C, nullptr, attn_t[st], dyo[st], dattn[st], false,
                       adh16[st][3], adhr[st][3], adh16t[st][3], ad_y1, ad_y2, ad_fa, ad_fb, ad_scratch,
                       ad_tmp16, ad_rsT, t16, trs, t16t, sums, R[st], gemm, stream);
        copy_rows(dattn[st], R[st] * C, jdo + off * C, stream);
    }
    mf_attn_bwd(jq, jk, jv, jof, jdo, Rj, heads, hd, jdq, jdk, jdv, true, gemm, stream);
    tape_put(jslice(kTapeJDO), jdo, Rj * C, stream);
    tape_put(jslice(kTapeJDQ), jdq, Rj * C, stream);
    tape_put(jslice(kTapeJDK), jdk, Rj * C, stream);
    tape_put(jslice(kTapeJDV), jdv, Rj * C, stream);
    for (int st = 0; st < 2; ++st) {
        const MfBlockW& W = *w[st];
        const int64_t off = st == 0 ? Rt : 0;
        copy_rows(jdv + off * C, R[st] * C, dyv[st], stream);
        mf_qk_norm_rope_bwd(yq[st], yk[st], W.nq, W.nk, st == 0 ? run.cos : nullptr, st == 0 ? run.sin : nullptr,
                            st == 0 && run.cos != nullptr, jdq + off * C, jdk + off * C, R[st], heads, hd, dyq[st],
                            dyk[st], W.dnq, W.dnk, stream);
        for (int i = 0; i < 3; ++i) {
            float* dy = i == 0 ? dyq[st] : i == 1 ? dyk[st] : dyv[st];
            tape_put(slice(st, kTapeDQ + i), dy, R[st] * C, stream);
            float* dxv = i == 0 ? dln1[st] : dgelu[st];
            ad_product_bwd(W, i, W.ad[i], ln1_16[st], ln1_s[st], ln1_t[st], dy, dxv, false,
                           adh16[st][i], adhr[st][i], adh16t[st][i], ad_y1, ad_y2, ad_fa, ad_fb,
                           ad_scratch, ad_tmp16, ad_rsT, t16, trs, t16t, sums, R[st], gemm, stream);
            if (i > 0) add_into(dln1[st], dgelu[st], R[st] * C, stream);
        }
        tape_put(slice(st, kTapeDLn1), dln1[st], R[st] * C, stream);
        mf_ln_mod_bwd(in[st], mean1[st], inv1[st], w[st]->mod + C, dln1[st], R[st], C, dxn1[st], nullptr, nullptr,
                      stream);
        add_into(dIn[st], dxn1[st], R[st] * C, stream);
    }
}

}  // namespace strata::image
