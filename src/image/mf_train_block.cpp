// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_train_block.cpp - see include/strata/image/mf_train.hpp's mf_block.  One Mage-Flow block forwards with a
// tape and backwards through it, made of the per-op gradients of mf_train.cpp, so a torch reference can check the whole
// block and not only its parts.
//
// Every buffer has one role and one owner: the tape holds what the backward reads, and the gradients live beside it.
// Nothing is aliased between a product's output and another one's gradient - the block is small and the GPU has room.
#include "strata/image/mf_train.hpp"
#include "strata/image/qi_kernels.hpp"

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

using kernels::f32_from_f16;

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

void mf_block(const MfBlockRun& run, prefill::Gemm& gemm, void* stream, train::Arena* arena) {
    const int64_t C = run.C, F = run.F, heads = run.heads, hd = run.hd, Ri = run.rows_img, Rt = run.rows_txt;
    const int64_t Rj = Ri + Rt, R[2] = {Ri, Rt};
    const float* in[2] = {run.img, run.txt};
    const MfBlockW* w[2] = {&run.img_w, &run.txt_w};
    float* outp[2] = {run.out_img, run.out_txt};
    const float* dout[2] = {run.d_out_img, run.d_out_txt};
    float* dIn[2] = {run.dimg, run.dtxt};
    if (C % heads != 0 || C / heads != hd) throw std::invalid_argument("mf_block: heads * head_dim must be the width");
    // the call's buffers from the arena (kept from call to call); a buffer written whole before it is read is not
    // zeroed (rf, ru), as moss_block's
    train::Pool pool(stream, arena);

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
    // q, k and v rounded up to the shared attention's rows (qi_attn_rows), zero past Rj as the pool leaves them
    const int64_t Rp = qi_attn_rows(Rj);
    uint16_t *jq = pool.u(Rp * C), *jk = pool.u(Rp * C), *jv = pool.u(Rp * C), *jo = pool.ru(Rj * C);
    float* jof = pool.rf(Rj * C);   // the same output before it was rounded: the backward's D reads this
    // v's power of two (v_scale): jv, jo and jof are the true rows divided by vs[0], and vs[1 ..] is the output
    // projection's row scale that puts it back
    uint32_t* vamax = reinterpret_cast<uint32_t*>(pool.f(1));
    float* vs = pool.rf(Rj + 1);
    float* lse = pool.rf(Rj * heads);
    // ---- the gradients
    float* dym2[2];
    float* dgelu[2];
    float* dym1[2];
    float* dln2[2];
    float* dh1[2];
    float* dyo[2];
    float* dattn[2];
    uint16_t* attn_t[2];   // the attention's FP16 rows transposed ([C][rows])
    float* jdq = pool.rf(Rj * C), *jdk = pool.f(Rj * C), *jdv = pool.f(Rj * C), *jdo = pool.rf(Rj * C);
    float* dyq[2];
    float* dyk[2];
    float* dyv[2];
    float* dln1[2];
    float* dxn1[2];
    for (int st = 0; st < 2; ++st) {
        ln1_16[st] = pool.ru(R[st] * C);
        ln1_s[st] = pool.rf(R[st]);
        ln1_t[st] = pool.ru(R[st] * C);
        mean1[st] = pool.rf(R[st]);
        inv1[st] = pool.rf(R[st]);
        yq[st] = pool.rf(R[st] * C);
        yk[st] = pool.rf(R[st] * C);
        yv[st] = pool.rf(R[st] * C);
        yo[st] = pool.rf(R[st] * C);
        h1[st] = pool.rf(R[st] * C);
        ln2_16[st] = pool.ru(R[st] * C);
        ln2_s[st] = pool.rf(R[st]);
        ln2_t[st] = pool.ru(R[st] * C);
        mean2[st] = pool.rf(R[st]);
        inv2[st] = pool.rf(R[st]);
        ym1[st] = pool.rf(R[st] * F);
        gelu_16[st] = pool.ru(R[st] * F);
        gelu_s[st] = pool.rf(R[st]);
        gelu_t[st] = pool.ru(R[st] * F);
        ym2[st] = pool.rf(R[st] * C);
        dym2[st] = pool.rf(R[st] * C);
        dgelu[st] = pool.rf(R[st] * nmax);   // the GELU's output, then a dgrad's dx: the wider of the two
        dym1[st] = pool.rf(R[st] * F);
        dln2[st] = pool.rf(R[st] * C);
        dh1[st] = pool.rf(R[st] * C);
        dyo[st] = pool.rf(R[st] * C);
        dattn[st] = pool.rf(R[st] * C);
        attn_t[st] = pool.ru(R[st] * C);
        dyq[st] = pool.rf(R[st] * C);
        dyk[st] = pool.rf(R[st] * C);
        dyv[st] = pool.rf(R[st] * C);
        dln1[st] = pool.rf(R[st] * C);
        dxn1[st] = pool.rf(R[st] * C);
    }
    uint16_t* t16 = pool.ru(Rj * nmax), *t16t = pool.ru(Rj * nmax);
    float* trs = pool.rf(Rj);
    // a wgrad's sums are N*K wide (at most F*C, never the square of the wider one) and a dgrad's T*K (at most rows*C
    // or rows*F), so this is the scratch the block's eight products can actually need
    float* sums = pool.rf(C * F + Rj * nmax);

    // ---- adapters: the forward keeps h16/hr/h16t per adapted product, the backward reads them back
    uint16_t* adh16[2][6] = {};
    float* adhr[2][6] = {};
    uint16_t* adh16t[2][6] = {};
    auto prod = [&](const MfBlockW& W, int i) {
        return train::Product{W.N[i], W.K[i], W.w[i], W.wt[i], W.dw[i], W.db[i], W.w8[i], W.sw[i]};
    };
    train::Product ps[12];
    const train::Adapter* ads[12] = {};
    const train::Product* prods[12];
    for (int st = 0; st < 2; ++st)
        for (int i = 0; i < 6; ++i) {
            const train::Adapter* az = w[st]->ad[i];
            const int64_t T = R[st], rk = az != nullptr ? az->rank : 0;
            if (az != nullptr) {
                adh16[st][i] = pool.ru(T * rk);
                adhr[st][i] = pool.rf(T);
                adh16t[st][i] = pool.ru(T * rk);
            }
            ps[st * 6 + i] = prod(*w[st], i);
            ads[st * 6 + i] = az;
            prods[st * 6 + i] = &ps[st * 6 + i];
        }
    // shared transients, sized for the largest adapted product (products run one at a time): hf is h = x A^T, y1 the
    // delta then yb/yd/dYb, y2 the second map a DoRA that trains its magnitude reads beside the first
    train::AdScratch adsc =
        train::take_ad_scratch(pool, train::ad_sizes(prods, ads, 12, [&](int i) { return R[i / 6]; }), true);
    auto keep = [&](int st, int i) { return train::AdKeep{adh16[st][i], adhr[st][i], adh16t[st][i]}; };
    // the base products: FP16, or a QLoRA's int8 base (W8A8, ai-toolkit's convrot8) with the input's int8 rows as
    // scratch, and FP16 ones where the GPU takes no int8 products
    const bool w8 = w[0]->w8[0] != nullptr;
    int8_t* x8 = nullptr;
    float* sx = nullptr;
    uint16_t *x16f = nullptr, *w16f = nullptr;
    if (w8) {
        const int64_t maxIn = std::max(Ri, Rt) * std::max(C, F);
        x8 = reinterpret_cast<int8_t*>(pool.ru((maxIn + 1) / 2));
        sx = pool.rf(std::max(Ri, Rt));
        if (!train::w8_int8(stream)) {
            x16f = pool.ru(maxIn);
            w16f = pool.ru(F * C);
        }
    }
    auto base = [&](int st, int i, const uint16_t* x16, float* y) {
        const MfBlockW& W = *w[st];
        if (W.w8[i] != nullptr) train::w8_product(x16, R[st], prod(W, i), y, x8, sx, x16f, w16f, gemm, stream);
        else gemm.f16(x16, W.w[i], y, R[st], W.N[i], W.K[i]);
    };

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
        dit_rows::rows16(dit_rows::LnMod{in[st], mod, mod + C, C}, R[st], C, ln1_16[st], ln1_s[st], stream,
                         mean1[st], inv1[st]);
        tape_put16(slice(st, kTapeLn1), ln1_16[st], R[st] * C, stream);
        train::transpose_f16(ln1_16[st], R[st], C, ln1_t[st], stream);
        for (int i = 0; i < 3; ++i) {
            float* dst = i == 0 ? yq[st] : i == 1 ? yk[st] : yv[st];
            base(st, i, ln1_16[st], dst);
            train::ad_product_fwd(w[st]->ad[i], ln1_16[st], ln1_s[st], dst, w[st]->b[i], R[st], C, C, keep(st, i), adsc,
                                  gemm, stream);
            tape_put(slice(st, kTapeQ + i), dst, R[st] * C, stream);
        }
    }
    train::v_scale(yv[0], R[0] * C, yv[1], R[1] * C, vamax, vs, Rj, stream);
    float* hsv = pool.rf(heads);   // v's one power of two for every head, as the shared kernel reads a head's
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) heads), [=](sycl::id<1> id) { hsv[id[0]] = vs[0]; });
    for (int st = 0; st < 2; ++st) {
        const int64_t off = st == 0 ? Rt : 0;   // the joint rows: the text's first, then the image's
        // the inference's own kernel: each head of q and k normed with its weight, then rotated (pairs 2j, 2j + 1)
        dit_rows::qk_rope16(rows_of(yq[st], C), rows_of(yk[st], C), rows_of(yv[st], C), w[st]->nq, w[st]->nk,
                            st == 0 ? run.cos : nullptr, st == 0 ? run.sin : nullptr, hsv, R[st], heads, hd, false,
                            jq + off * C, jk + off * C, jv + off * C, stream, kEps);
    }
    qi_attention_train(jq, jk, jv, Rj, heads, hd, jof, lse, stream);   // the shared attention, inference's kernel
    train::f16_copy(jof, Rj * C, jo, stream);
    tape_put16(jslice(kTapeJQ), jq, Rj * C, stream);
    tape_put16(jslice(kTapeJK), jk, Rj * C, stream);
    tape_put16(jslice(kTapeJV), jv, Rj * C, stream);
    tape_put16(jslice(kTapeJO), jo, Rj * C, stream);
    tape_put(jslice(kTapeLse), lse, Rj * heads, stream);
    for (int st = 0; st < 2; ++st) {
        const float* mod = w[st]->mod;
        const int64_t off = st == 0 ? Rt : 0;
        base(st, 3, jo + off * C, yo[st]);
        train::ad_product_fwd(w[st]->ad[3], jo + off * C, vs + 1, yo[st], w[st]->b[3], R[st], C, C, keep(st, 3), adsc,
                              gemm, stream);
        tape_put(slice(st, kTapeO), yo[st], R[st] * C, stream);
        dit_rows::gate_add_to(h1[st], in[st], rows_of(yo[st], C), mod + 2 * C, R[st], C, stream);
        tape_put(slice(st, kTapeH1), h1[st], R[st] * C, stream);
        dit_rows::rows16(dit_rows::LnMod{h1[st], mod + 3 * C, mod + 4 * C, C}, R[st], C, ln2_16[st], ln2_s[st],
                         stream, mean2[st], inv2[st]);
        tape_put16(slice(st, kTapeLn2), ln2_16[st], R[st] * C, stream);
        train::transpose_f16(ln2_16[st], R[st], C, ln2_t[st], stream);
        base(st, 4, ln2_16[st], ym1[st]);
        train::ad_product_fwd(w[st]->ad[4], ln2_16[st], ln2_s[st], ym1[st], w[st]->b[4], R[st], F, C, keep(st, 4), adsc,
                              gemm, stream);
        tape_put(slice(st, kTapeM1), ym1[st], R[st] * F, stream);
        dit_rows::rows16(dit_rows::GeluTanh{rows_of(ym1[st], F)}, R[st], F, gelu_16[st], gelu_s[st], stream);
        tape_put16(slice(st, kTapeGelu), gelu_16[st], R[st] * F, stream);
        train::transpose_f16(gelu_16[st], R[st], F, gelu_t[st], stream);
        base(st, 5, gelu_16[st], ym2[st]);
        train::ad_product_fwd(w[st]->ad[5], gelu_16[st], gelu_s[st], ym2[st], w[st]->b[5], R[st], C, F, keep(st, 5),
                              adsc, gemm, stream);
        tape_put(slice(st, kTapeM2), ym2[st], R[st] * C, stream);
        dit_rows::gate_add_to(outp[st], h1[st], rows_of(ym2[st], C), mod + 5 * C, R[st], C, stream);
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

    // no output gradients: a forward-only read (the trainer's eval and kept forwards)
    if (dout[0] != nullptr || dout[1] != nullptr) {
    // ---- backward, from the outputs' gradients
        for (int st = 0; st < 2; ++st) {
            const MfBlockW& W = *w[st];
            const float* mod = W.mod;
            // the MLP's gated residual: dh1 is the output's gradient as it reaches the first residual's sum
            train::copy_rows(dout[st], R[st] * C, dh1[st], stream);
            // the modulation's own gradient (its LoRA's, ai-toolkit trains img_mod.1 and txt_mod.1): each gate's
            // the gated output times the gradient over the rows, each norm's shift and scale from its backward
            if (W.dmod != nullptr)
                dit_rows::gate_add_bwd(ym2[st], nullptr, dout[st], R[st], C, nullptr, W.dmod + 5 * C, stream);
            train::mul_cols(dout[st], mod + 5 * C, R[st] * C, C, dym2[st], stream);
            tape_put(slice(st, kTapeDM2), dym2[st], R[st] * C, stream);
            train::ad_product_bwd(prod(W, 5), W.ad[5], gelu_16[st], gelu_s[st], gelu_t[st], dym2[st], dgelu[st], false,
                                  keep(st, 5), adsc, t16, trs, t16t, sums, R[st], gemm, stream);
            tape_put(slice(st, kTapeDGelu), dgelu[st], R[st] * F, stream);
            dit_rows::gelu_tanh_bwd(ym1[st], dgelu[st], R[st] * F, dym1[st], stream);
            tape_put(slice(st, kTapeDM1), dym1[st], R[st] * F, stream);
            train::ad_product_bwd(prod(W, 4), W.ad[4], ln2_16[st], ln2_s[st], ln2_t[st], dym1[st], dln2[st], false,
                                  keep(st, 4), adsc, t16, trs, t16t, sums, R[st], gemm, stream);
            tape_put(slice(st, kTapeDLn2), dln2[st], R[st] * C, stream);
            // the second norm: its input's gradient joins the residual's own
            dit_rows::ln_mod_bwd(h1[st], mean2[st], inv2[st], mod + 4 * C, dln2[st], R[st], C, dgelu[st],
                                 W.dmod != nullptr ? W.dmod + 3 * C : nullptr,
                                 W.dmod != nullptr ? W.dmod + 4 * C : nullptr, stream);
            train::add_into(dh1[st], dgelu[st], R[st] * C, stream);
            if (W.dmod != nullptr)
                dit_rows::gate_add_bwd(yo[st], nullptr, dh1[st], R[st], C, nullptr, W.dmod + 2 * C, stream);
            tape_put(slice(st, kTapeDH1), dh1[st], R[st] * C, stream);
            // the attention's gated residual
            train::mul_cols(dh1[st], mod + 2 * C, R[st] * C, C, dyo[st], stream);
            tape_put(slice(st, kTapeDYo), dyo[st], R[st] * C, stream);
            if (dIn[st] != nullptr) train::add_into(dIn[st], dh1[st], R[st] * C, stream);   // the residual's own path
            const int64_t off = st == 0 ? Rt : 0;
            train::transpose_f16(jo + off * C, R[st], C, attn_t[st], stream);
            train::ad_product_bwd(prod(W, 3), W.ad[3], jo + off * C, vs + 1, attn_t[st], dyo[st], dattn[st], false,
                                  keep(st, 3), adsc, t16, trs, t16t, sums, R[st], gemm, stream);
            train::copy_rows(dattn[st], R[st] * C, jdo + off * C, stream);
        }
        // the attention ran on v / vs[0]: its output's gradient comes in times vs[0] and dV leaves divided by it
        train::scale_by(jdo, Rj * C, vs, false, stream);
        qi_attention_bwd(jq, jk, jv, jof, lse, jdo, Rj, heads, hd, jdq, jdk, jdv, true, gemm, stream);
        tape_put(jslice(kTapeJDO), jdo, Rj * C, stream);
        tape_put(jslice(kTapeJDQ), jdq, Rj * C, stream);
        tape_put(jslice(kTapeJDK), jdk, Rj * C, stream);
        tape_put(jslice(kTapeJDV), jdv, Rj * C, stream);   // the joint rows' own gradients, as jv's
        train::scale_by(jdv, Rj * C, vs, true, stream);
        for (int st = 0; st < 2; ++st) {
            const MfBlockW& W = *w[st];
            const int64_t off = st == 0 ? Rt : 0;
            train::copy_rows(jdv + off * C, R[st] * C, dyv[st], stream);
            dit_rows::qk_rope_bwd(yq[st], yk[st], W.nq, W.nk, st == 0 ? run.cos : nullptr, st == 0 ? run.sin : nullptr,
                                st == 0 && run.cos != nullptr, jdq + off * C, jdk + off * C, R[st], heads, hd, dyq[st],
                                dyk[st], W.dnq, W.dnk, stream, false);
            for (int i = 0; i < 3; ++i) {
                float* dy = i == 0 ? dyq[st] : i == 1 ? dyk[st] : dyv[st];
                tape_put(slice(st, kTapeDQ + i), dy, R[st] * C, stream);
                float* dxv = i == 0 ? dln1[st] : dgelu[st];
                train::ad_product_bwd(prod(W, i), W.ad[i], ln1_16[st], ln1_s[st], ln1_t[st], dy, dxv, false, keep(st,
                                      i), adsc, t16, trs, t16t, sums, R[st], gemm, stream);
                if (i > 0) train::add_into(dln1[st], dgelu[st], R[st] * C, stream);
            }
            tape_put(slice(st, kTapeDLn1), dln1[st], R[st] * C, stream);
            dit_rows::ln_mod_bwd(in[st], mean1[st], inv1[st], w[st]->mod + C, dln1[st], R[st], C, dxn1[st],
                                 W.dmod, W.dmod != nullptr ? W.dmod + C : nullptr, stream);
            if (dIn[st] != nullptr) train::add_into(dIn[st], dxn1[st], R[st] * C, stream);
        }
    }
}

}  // namespace strata::image
