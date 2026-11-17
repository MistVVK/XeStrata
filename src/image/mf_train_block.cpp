// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_train_block.cpp - see include/strata/image/mf_train.hpp's mf_block.  One Mage-Flow block forwards with a
// tape and backwards through it, made of the per-op gradients of mf_train.cpp, so a torch reference can check the whole
// block and not only its parts.
//
// The tape holds what the backward reads. Scratch buffers whose uses do not overlap share memory on the ordered stream.
#include "strata/image/mf_train.hpp"
#include "strata/image/qi_kernels.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <optional>
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

namespace {

// What a block's backward reads of its forward (its tape), each stream's [2] and the joint attention's, taken in this
// order from a pool or a Sizer
struct Tape {
    uint16_t *ln1_16[2], *ln1_t[2], *ln2_16[2], *ln2_t[2], *gelu_16[2], *gelu_t[2];
    float *ln1_s[2], *mean1[2], *inv1[2], *yq[2], *yk[2], *yo[2], *h1[2];
    float *ln2_s[2], *mean2[2], *inv2[2], *ym1[2], *gelu_s[2], *ym2[2];
    uint16_t *jq, *jk, *jv, *jo;
    float *jof, *vs, *lse;
    train::AdKeep keep[2][6];
};

const MfBlockW& weights_of(const MfBlockRun& run, int st) { return st == 0 ? run.img_w : run.txt_w; }

train::Product product_of(const MfBlockW& W, int i) {
    return train::Product{W.N[i], W.K[i], W.w[i], W.wt[i], W.dw[i], W.db[i], W.w8[i], W.sw[i], W.wr[i]};
}

// whether the products' backwards read their inputs transposed (train::ad_needs_xt: a weight's own gradient, a LoKr,
// a GPU without transposed products): the transposed copies are made and kept only then
bool needs_xt(const MfBlockRun& run, prefill::Gemm& gemm) {
    for (int st = 0; st < 2; ++st)
        for (int i = 0; i < 6; ++i)
            if (train::ad_needs_xt(product_of(weights_of(run, st), i), weights_of(run, st).ad[i], gemm)) return true;
    return false;
}

template <class Take> Tape take_tape(Take& t, const MfBlockRun& run, bool xt) {
    const int64_t C = run.C, F = run.F, R[2] = {run.rows_img, run.rows_txt}, Rj = R[0] + R[1];
    const int64_t Rp = qi_attn_rows(Rj);
    Tape p{};
    // q, k and v rounded up to the shared attention's rows (qi_attn_rows), zero past Rj as the pool leaves them
    p.jq = t.u(Rp * C);
    p.jk = t.u(Rp * C);
    p.jv = t.u(Rp * C);
    p.jo = t.ru(Rj * C);
    p.jof = t.rf(Rj * C);   // the same output before it was rounded: the backward's D reads this
    // v's power of two (v_scale): jv, jo and jof are the true rows divided by vs[0], and vs[1 ..] is the output
    // projection's row scale that puts it back
    p.vs = t.rf(Rj + 1);
    p.lse = t.rf(Rj * run.heads);
    for (int st = 0; st < 2; ++st) {
        p.ln1_16[st] = t.ru(R[st] * C);   // the first norm's FP16 rows (q, k and v read them)
        p.ln1_s[st] = t.rf(R[st]);
        p.ln1_t[st] = xt ? t.ru(R[st] * C) : nullptr;   // the same rows transposed ([C][rows])
        p.mean1[st] = t.rf(R[st]);
        p.inv1[st] = t.rf(R[st]);
        p.yq[st] = t.rf(R[st] * C);
        p.yk[st] = t.rf(R[st] * C);
        p.yo[st] = t.rf(R[st] * C);
        p.h1[st] = t.rf(R[st] * C);   // the first residual's output, which the second norm reads
        p.ln2_16[st] = t.ru(R[st] * C);
        p.ln2_s[st] = t.rf(R[st]);
        p.ln2_t[st] = xt ? t.ru(R[st] * C) : nullptr;
        p.mean2[st] = t.rf(R[st]);
        p.inv2[st] = t.rf(R[st]);
        p.ym1[st] = t.rf(R[st] * F);   // mlp1's output, before the GELU
        p.gelu_16[st] = t.ru(R[st] * F);   // the GELU's output as mlp2's input rows
        p.gelu_s[st] = t.rf(R[st]);
        p.gelu_t[st] = xt ? t.ru(R[st] * F) : nullptr;
        p.ym2[st] = t.rf(R[st] * C);
    }
    // adapters: the forward keeps h16, hr and h16t per adapted product, the backward reads them back
    for (int st = 0; st < 2; ++st)
        for (int i = 0; i < 6; ++i)
            if (const train::Adapter* az = weights_of(run, st).ad[i]; az != nullptr)
                p.keep[st][i] = train::AdKeep{t.ru(R[st] * az->rank), t.rf(R[st]), t.ru(R[st] * az->rank)};
    return p;
}

}  // namespace

size_t mf_block_tape_bytes(const MfBlockRun& run, prefill::Gemm& gemm) {
    train::Sizer z;
    take_tape(z, run, needs_xt(run, gemm));
    return z.bytes;
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
    const bool backward = dout[0] != nullptr || dout[1] != nullptr;
    // the call's buffers from the arena (kept from call to call); a buffer written whole before it is read is not
    // zeroed (rf, ru), as moss_block's
    train::Pool pool(stream, arena);
    // the forward's values the backward reads: the block's own kept tape (a kept forward's, read back by a replay),
    // else the call's scratch
    const bool replay = run.kept != nullptr && run.kept_replay;
    if (replay && !backward) throw std::invalid_argument("mf_block: a kept tape is read by the backward");
    if (replay && run.tape != nullptr) throw std::invalid_argument("mf_block: the harness's tape takes a forward");
    std::optional<train::Pool> tpool;
    if (run.kept != nullptr) tpool.emplace(stream, run.kept, replay);
    const bool xt = needs_xt(run, gemm);
    const Tape T = run.kept != nullptr ? take_tape(*tpool, run, xt) : take_tape(pool, run, xt);
    uint16_t *jq = T.jq, *jk = T.jk, *jv = T.jv, *jo = T.jo;
    float *jof = T.jof, *vs = T.vs, *lse = T.lse;
    uint16_t* const* ln1_16 = T.ln1_16;
    uint16_t* const* ln1_t = T.ln1_t;
    uint16_t* const* ln2_16 = T.ln2_16;
    uint16_t* const* ln2_t = T.ln2_t;
    uint16_t* const* gelu_16 = T.gelu_16;
    uint16_t* const* gelu_t = T.gelu_t;
    float* const* ln1_s = T.ln1_s;
    float* const* mean1 = T.mean1;
    float* const* inv1 = T.inv1;
    float* const* yq = T.yq;
    float* const* yk = T.yk;
    float* const* yo = T.yo;
    float* const* h1 = T.h1;
    float* const* ln2_s = T.ln2_s;
    float* const* mean2 = T.mean2;
    float* const* inv2 = T.inv2;
    float* const* ym1 = T.ym1;
    float* const* gelu_s = T.gelu_s;
    float* const* ym2 = T.ym2;
    // Unscaled V is dead before the MLP writes ym2. In the backward ym2 is read for the modulation's
    // gradient before its storage is reused for the MLP, attention-output and V gradients.
    float* yv[2] = {ym2[0], ym2[1]};
    uint32_t* vamax = reinterpret_cast<uint32_t*>(pool.f(1));

    train::Product ps[12];
    const train::Adapter* ads[12] = {};
    const train::Product* prods[12];
    for (int st = 0; st < 2; ++st)
        for (int i = 0; i < 6; ++i) {
            ps[st * 6 + i] = product_of(*w[st], i);
            ads[st * 6 + i] = w[st]->ad[i];
            prods[st * 6 + i] = &ps[st * 6 + i];
        }
    auto prod = [&](const MfBlockW& W, int i) { return product_of(W, i); };
    // shared transients, sized for the largest adapted product (products run one at a time): hf is h = x A^T, y1 the
    // delta then yb/yd/dYb, y2 the second map a DoRA that trains its magnitude reads beside the first
    train::AdSizes sizes = train::ad_sizes(prods, ads, 12, [&](int i) { return R[i / 6]; });
    if (backward && sizes.plain && sizes.n > 0) {
        sizes.rows = std::max(Ri, Rt);
        sizes.n = std::max(sizes.n, std::max(C, F));
        if (xt) sizes.tmp = std::max(sizes.tmp, Rj * std::max(C, F));
    }
    train::AdScratch adsc = train::take_ad_scratch(pool, sizes);
    auto keep = [&](int st, int i) { return T.keep[st][i]; };
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
    if (!replay) {   // a kept tape's backward has its forward already
    for (int st = 0; st < 2; ++st) {
        const float* mod = w[st]->mod;
        dit_rows::rows16(dit_rows::LnMod{in[st], mod, mod + C, C}, R[st], C, ln1_16[st], ln1_s[st], stream,
                         mean1[st], inv1[st]);
        tape_put16(slice(st, kTapeLn1), ln1_16[st], R[st] * C, stream);
        if (xt) train::transpose_f16(ln1_16[st], R[st], C, ln1_t[st], stream);
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
        if (xt) train::transpose_f16(ln2_16[st], R[st], C, ln2_t[st], stream);
        base(st, 4, ln2_16[st], ym1[st]);
        train::ad_product_fwd(w[st]->ad[4], ln2_16[st], ln2_s[st], ym1[st], w[st]->b[4], R[st], F, C, keep(st, 4), adsc,
                              gemm, stream);
        tape_put(slice(st, kTapeM1), ym1[st], R[st] * F, stream);
        dit_rows::rows16(dit_rows::GeluTanh{rows_of(ym1[st], F)}, R[st], F, gelu_16[st], gelu_s[st], stream);
        tape_put16(slice(st, kTapeGelu), gelu_16[st], R[st] * F, stream);
        if (xt) train::transpose_f16(gelu_16[st], R[st], F, gelu_t[st], stream);
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
    }

    // no output gradients: a forward-only read (the trainer's eval and kept forwards)
    if (backward) {
    // ---- backward, from the outputs' gradients
        const int64_t nmax = std::max(C, F);   // the wider of the block's two widths: a buffer reused between roles
        float *dym2[2], *dgelu[2], *dym1[2], *dln2[2], *dh1[2], *dyo[2], *dattn[2], *dyq[2], *dyk[2], *dyv[2];
        float *dln1[2], *dxn1[2];
        uint16_t* attn_t[2];   // the attention's FP16 rows transposed ([C][rows]) where they are read
        float *jdq = pool.rf(Rj * C), *jdk = pool.f(Rj * C), *jdv = pool.f(Rj * C), *jdo = pool.rf(Rj * C);
        for (int st = 0; st < 2; ++st) {
            dym2[st] = yv[st];   // the forward's unscaled V is dead
            // Plain LoRA's y1 is only used in the forward. The two streams' backwards run in order.
            dgelu[st] = sizes.plain && sizes.n > 0 ? adsc.y1 : pool.rf(R[st] * nmax);
            dym1[st] = dgelu[st];   // GELU backward is elementwise, so its input gradient can replace dgelu
            dln2[st] = pool.rf(R[st] * C);
            dh1[st] = pool.rf(R[st] * C);
            dyo[st] = dym2[st];   // the MLP output gradient is dead before the attention output's gradient
            dattn[st] = dln2[st];  // the second norm's gradient is dead before the attention's input gradient
            attn_t[st] = xt ? gelu_t[st] : nullptr;   // the GELU transpose is dead after the MLP backward
            // The QK backward reads each pair before writing it. Its second pass needs the original rows only
            // when the norm weights themselves take gradients; otherwise reuse the forward Q and K.
            const bool qk_weights = w[st]->dnq != nullptr || w[st]->dnk != nullptr;
            dyq[st] = qk_weights ? pool.rf(R[st] * C) : yq[st];
            dyk[st] = qk_weights ? pool.rf(R[st] * C) : yk[st];
            dyv[st] = yv[st];   // dym2 and dyo are dead before the attention backward
            dln1[st] = dh1[st];
            dxn1[st] = dattn[st];
        }
        uint16_t *t16 = pool.ru(Rj * nmax);
        uint16_t* t16t = xt ? (sizes.plain && sizes.n > 0 ? adsc.tmp16 : pool.ru(Rj * nmax)) : nullptr;
        // One product's dgrad weight lives until its queued product finishes. The same ordered stream can then
        // overwrite it for the next product; the parity harness may still supply the twelve pre-made views.
        int64_t weight_elems = 0;
        bool has_w8 = false;
        for (const train::Product& p : ps) {
            weight_elems = std::max(weight_elems, p.N * p.K);
            has_w8 = has_w8 || p.w8 != nullptr;
        }
        const bool row_dgrad = gemm.f16_batch_ex_ok();
        uint16_t* dweight = run.lazy_dgrad && (!row_dgrad || has_w8) ? pool.ru(weight_elems) : nullptr;
        uint16_t* dweight_rows = run.lazy_dgrad && !row_dgrad && has_w8 ? pool.ru(weight_elems) : nullptr;
        auto bwd_prod = [&](const MfBlockW& W, int i) {
            train::Product p = prod(W, i);
            if (!run.lazy_dgrad) return p;
            if (row_dgrad) {
                if (p.w8 != nullptr) train::w8_dequant_rows(p.w8, p.sw, p.N, p.K, dweight, stream);
                p.wr = p.w8 != nullptr ? dweight : p.w;
            } else {
                if (p.w8 != nullptr) train::w8_dequant_t(p.w8, p.sw, p.N, p.K, dweight_rows, dweight, stream);
                else train::transpose_f16(p.w, p.N, p.K, dweight, stream);
                p.wt = dweight;
            }
            return p;
        };
        float* trs = pool.rf(Rj);
        // A dgrad that overwrites its output uses that output for its sums. Only a full wgrad or LoKr's two
        // simultaneous N*K matrices need separate sums; the usual frozen-base LoRA takes none.
        int64_t sum_elems = 0;
        for (int i = 0; i < 12; ++i) {
            const int64_t nk = ps[i].N * ps[i].K;
            if (ps[i].dw != nullptr) sum_elems = std::max(sum_elems, nk);
            if (ads[i] != nullptr && ads[i]->method == train::AdapterMethod::lokr)
                sum_elems = std::max(sum_elems, 2 * nk);
        }
        float* sums = sum_elems > 0 ? pool.rf(sum_elems) : nullptr;
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
            train::ad_product_bwd(bwd_prod(W, 5), W.ad[5], gelu_16[st], gelu_s[st], gelu_t[st], dym2[st], dgelu[st], false,
                                  keep(st, 5), adsc, t16, trs, t16t, sums, R[st], gemm, stream);
            tape_put(slice(st, kTapeDGelu), dgelu[st], R[st] * F, stream);
            dit_rows::gelu_tanh_bwd(ym1[st], dgelu[st], R[st] * F, dym1[st], stream);
            tape_put(slice(st, kTapeDM1), dym1[st], R[st] * F, stream);
            train::ad_product_bwd(bwd_prod(W, 4), W.ad[4], ln2_16[st], ln2_s[st], ln2_t[st], dym1[st], dln2[st], false,
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
            if (xt) train::transpose_f16(jo + off * C, R[st], C, attn_t[st], stream);
            train::ad_product_bwd(bwd_prod(W, 3), W.ad[3], jo + off * C, vs + 1, attn_t[st], dyo[st], dattn[st], false,
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
                train::ad_product_bwd(bwd_prod(W, i), W.ad[i], ln1_16[st], ln1_s[st], ln1_t[st], dy, dxv, false, keep(st,
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
