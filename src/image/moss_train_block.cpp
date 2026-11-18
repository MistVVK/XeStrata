// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_train_block.cpp - see include/strata/image/moss_train.hpp's moss_block.  One MOSS-Image block forwards
// and backwards, out of the shared row kernels and products, in the order mf_train_block.cpp runs Mage-Flow's (one
// stream here: the image's and the text's rows take the same weights).
#include "strata/image/moss_train.hpp"
#include "strata/image/qi_kernels.hpp"

#include "strata/core/gpu.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

constexpr float kEps = 1e-6f;   // the RMSNorms' eps, as moss_dit.cpp takes it
// The MLP's rows a pass: its two products and their backwards a band of this many rows at a time, so its widest
// scratch (2F = 16384 values a row) is this many rows, not the image's (Qwen-Image-2.1's block takes the same 1024)
constexpr int64_t kMlpRows = 1024;

// [R][3C] (q, k, v side by side) <-> three [R][C]
void split3(const float* qkv, int64_t R, int64_t C, float* q, float* k, float* v, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (R * C)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / C, c = i - t * C;
        q[i] = qkv[t * 3 * C + c];
        k[i] = qkv[(t * 3 + 1) * C + c];
        v[i] = qkv[(t * 3 + 2) * C + c];
    });
}

void join3(const float* q, const float* k, const float* v, int64_t R, int64_t C, float* qkv, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (R * C)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / C, c = i - t * C;
        qkv[t * 3 * C + c] = q[i];
        qkv[(t * 3 + 1) * C + c] = k[i];
        qkv[(t * 3 + 2) * C + c] = v[i];
    });
}

// What a block's backward reads of its forward (its tape), taken in this order from a pool or a Sizer
struct Tape {
    uint16_t *ln1_16, *ln1_t, *ln2_16, *ln2_t, *act16, *act_t, *jq, *jk, *jv, *jo;
    float *ln1_s, *ln2_s, *inv1, *inv2, *yq, *yk, *h1, *yug, *act_s, *jof, *vs, *lse;
    train::AdKeep keep[kMossProducts];
};

// whether the products' backwards read their inputs transposed (train::ad_needs_xt: a GPU without transposed products,
// a LoKr): the transposed copies are made and kept only then
bool needs_xt(const MossBlockRun& run, prefill::Gemm& gemm) {
    for (int i = 0; i < kMossProducts; ++i)
        if (train::ad_needs_xt(run.p[i], run.ad[i], gemm)) return true;
    return false;
}

template <class Take> Tape take_tape(Take& t, const MossBlockRun& run, bool xt) {
    const int64_t C = run.C, F = run.F, R = run.rows, Rp = qi_attn_rows(R);
    Tape p{};
    p.ln1_16 = t.ru(R * C);
    p.ln1_t = xt ? t.ru(R * C) : nullptr;
    p.ln2_16 = t.ru(R * C);
    p.ln2_t = xt ? t.ru(R * C) : nullptr;
    p.ln1_s = t.rf(R);
    p.ln2_s = t.rf(R);
    p.inv1 = t.rf(R);
    p.inv2 = t.rf(R);
    p.yq = t.rf(R * C);
    p.yk = t.rf(R * C);
    p.h1 = t.rf(R * C);
    p.yug = t.rf(R * 2 * F);   // the MLP's hidden values and gates
    p.act16 = t.ru(R * F);
    p.act_t = xt ? t.ru(R * F) : nullptr;
    p.act_s = t.rf(R);
    p.jq = t.u(Rp * C);   // q, k and v rounded up to the shared attention's rows, zero past R
    p.jk = t.u(Rp * C);
    p.jv = t.u(Rp * C);
    p.jo = t.ru(R * C);
    p.jof = t.rf(R * C);
    p.vs = t.rf(R + 1);
    p.lse = t.rf(R * run.heads);
    for (int i = 0; i < kMossProducts; ++i)   // what each adapted product's forward keeps
        if (run.ad[i] != nullptr) {
            const int64_t rk = run.ad[i]->rank;
            p.keep[i] = train::AdKeep{t.ru(R * rk), t.rf(R), t.ru(R * rk)};
        }
    return p;
}

train::StageTimes& times() {
    static train::StageTimes t("moss_block");
    return t;
}

}  // namespace

void moss_block_profile_report() { times().report(); }

size_t moss_block_tape_bytes(const MossBlockRun& run, prefill::Gemm& gemm) {
    train::Sizer z;
    take_tape(z, run, needs_xt(run, gemm));
    return z.bytes;
}

void moss_block(const MossBlockRun& run, prefill::Gemm& gemm, void* stream, train::Arena* arena) {
    const int64_t C = run.C, F = run.F, heads = run.heads, hd = run.hd, R = run.rows;
    if (C % heads != 0 || C / heads != hd) throw std::invalid_argument("moss_block: heads * head_dim must be the width");
    const float* mod = run.mod;
    // the call's buffers from the arena (kept from call to call); a buffer written whole before it is read is not
    // zeroed (rf, ru): zeroing a block's half a gigabyte of them each call cost more than some of its products
    train::Pool pool(stream, arena);
    // the forward's values the backward reads: the block's own tape (a kept forward's, read back by a replay), else
    // the call's scratch
    const bool replay = run.tape != nullptr && run.tape_replay;
    if (replay && run.dout == nullptr) throw std::invalid_argument("moss_block: a kept tape is read by the backward");
    std::optional<train::Pool> tpool;
    if (run.tape != nullptr) tpool.emplace(stream, run.tape, replay);
    const bool xt = needs_xt(run, gemm);
    const Tape tp = run.tape != nullptr ? take_tape(*tpool, run, xt) : take_tape(pool, run, xt);
    uint16_t *ln1_16 = tp.ln1_16, *ln1_t = tp.ln1_t, *ln2_16 = tp.ln2_16, *ln2_t = tp.ln2_t, *act16 = tp.act16,
             *act_t = tp.act_t, *jq = tp.jq, *jk = tp.jk, *jv = tp.jv, *jo = tp.jo;
    float *ln1_s = tp.ln1_s, *ln2_s = tp.ln2_s, *inv1 = tp.inv1, *inv2 = tp.inv2, *yq = tp.yq, *yk = tp.yk,
          *h1 = tp.h1, *yug = tp.yug, *act_s = tp.act_s, *jof = tp.jof, *vs = tp.vs, *lse = tp.lse;
    const train::AdKeep* keep = tp.keep;
    // the forward's transients
    float* mean = pool.rf(R);
    float* yqkv = pool.rf(R * 3 * C);
    float *yv = pool.rf(R * C), *yo = pool.rf(R * C), *ym2 = pool.rf(R * C);
    uint32_t* vamax = reinterpret_cast<uint32_t*>(pool.f(1));
    float* hsv = pool.rf(heads);

    // the MLP's band of rows: every row at once where the backward reads the inputs transposed (a band of them is not
    // a slice of the transposed rows)
    const int64_t Bm = xt ? R : std::min(R, kMlpRows);
    auto rows_of_product = [&](int i) { return i == kMossUpGate || i == kMossDown ? Bm : R; };

    // ---- the adapters' shared scratch, sized for the largest product's rows
    const train::Product* prods[kMossProducts];
    for (int i = 0; i < kMossProducts; ++i) prods[i] = &run.p[i];
    train::AdScratch adsc =
        train::take_ad_scratch(pool, train::ad_sizes(prods, run.ad, kMossProducts, rows_of_product));
    // a QLoRA's int8 base: the input's int8 rows, and FP16 ones where the GPU takes no int8 products
    const bool w8 = run.p[0].w8 != nullptr;
    int8_t* x8 = nullptr;
    float* sx = nullptr;
    uint16_t *x16f = nullptr, *w16f = nullptr;
    const int64_t mx_in = std::max(R * C, Bm * F);   // a product's input values: q, k, v's and o's, or a band's
    if (w8) {
        x8 = reinterpret_cast<int8_t*>(pool.ru((mx_in + 1) / 2));
        sx = pool.rf(R);
        if (!train::w8_int8(stream)) {
            x16f = pool.ru(mx_in);
            w16f = pool.ru(2 * F * C);
        }
    }
    auto base = [&](int i, const uint16_t* x16, float* y, int64_t n) {
        const train::Product& P = run.p[i];
        if (i == kMossQkv && run.w_plain_qkv != nullptr) gemm.f16(x16, run.w_plain_qkv, y, n, P.N, P.K);
        else if (P.w8 != nullptr) train::w8_product(x16, n, P, y, x8, sx, x16f, w16f, gemm, stream);
        else gemm.f16(x16, P.w, y, n, P.N, P.K);
    };
    // an adapter's kept rows from row r0 on (a band of the MLP's)
    auto keep_at = [&](int i, int64_t r0) {
        train::AdKeep k = keep[i];
        if (run.ad[i] != nullptr && r0 > 0) {
            const int64_t rk = run.ad[i]->rank;
            k.h16 += r0 * rk;
            k.hr += r0;
            if (k.h16t != nullptr) k.h16t += r0 * rk;   // (not read where the MLP is banded: no transposed rows)
        }
        return k;
    };

    // ---- forward
    train::StageTimes& tm = times();
    tm.begin(stream);
    if (!replay) {   // a kept tape's backward has its forward already
        dit_rows::rows16(dit_rows::RmsAffine{run.x, run.n1, mod + C, mod, C, kEps, true}, R, C, ln1_16, ln1_s, stream,
                         mean, inv1);
        if (xt) train::transpose_f16(ln1_16, R, C, ln1_t, stream);
        base(kMossQkv, ln1_16, yqkv, R);
        train::ad_product_fwd(run.ad[kMossQkv], ln1_16, ln1_s, yqkv, nullptr, R, 3 * C, C, keep[kMossQkv], adsc, gemm,
                              stream);
        tm.mark("fwd norm + qkv", stream);
        split3(yqkv, R, C, yq, yk, yv, stream);
        train::v_scale(yv, R * C, nullptr, 0, vamax, vs, R, stream);
        dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) heads), [=](sycl::id<1> id) { hsv[id[0]] = vs[0]; });
        dit_rows::qk_rope16(rows_of(yq, C), rows_of(yk, C), rows_of(yv, C), run.nq, run.nk, run.cos, run.sin, hsv, R,
                            heads, hd, true, jq, jk, jv, stream, kEps);
        tm.mark("fwd rope", stream);
        qi_attention_train(jq, jk, jv, R, heads, hd, jof, lse, stream);
        tm.mark("fwd attention", stream);
        train::f16_copy(jof, R * C, jo, stream);
        base(kMossO, jo, yo, R);
        train::ad_product_fwd(run.ad[kMossO], jo, vs + 1, yo, nullptr, R, C, C, keep[kMossO], adsc, gemm, stream);
        dit_rows::gate_add_to(h1, run.x, rows_of(yo, C), mod + 2 * C, R, C, stream);
        tm.mark("fwd o", stream);
        dit_rows::rows16(dit_rows::RmsAffine{h1, run.n2, mod + 4 * C, mod + 3 * C, C, kEps, true}, R, C, ln2_16, ln2_s,
                         stream, mean, inv2);
        if (xt) train::transpose_f16(ln2_16, R, C, ln2_t, stream);
        for (int64_t r0 = 0; r0 < R; r0 += Bm) {
            const int64_t n = std::min(Bm, R - r0);
            base(kMossUpGate, ln2_16 + r0 * C, yug + r0 * 2 * F, n);
            train::ad_product_fwd(run.ad[kMossUpGate], ln2_16 + r0 * C, ln2_s + r0, yug + r0 * 2 * F, nullptr, n, 2 * F,
                                  C, keep_at(kMossUpGate, r0), adsc, gemm, stream);
        }
        tm.mark("fwd norm + up_gate", stream);
        dit_rows::rows16(dit_rows::Situ{rows_of(yug, 2 * F), F}, R, F, act16, act_s, stream);
        if (xt) train::transpose_f16(act16, R, F, act_t, stream);
        tm.mark("fwd situ", stream);
        for (int64_t r0 = 0; r0 < R; r0 += Bm) {
            const int64_t n = std::min(Bm, R - r0);
            base(kMossDown, act16 + r0 * F, ym2 + r0 * C, n);
            train::ad_product_fwd(run.ad[kMossDown], act16 + r0 * F, act_s + r0, ym2 + r0 * C, nullptr, n, C, F,
                                  keep_at(kMossDown, r0), adsc, gemm, stream);
        }
        dit_rows::gate_add_to(run.out, h1, rows_of(ym2, C), mod + 5 * C, R, C, stream);
        tm.mark("fwd down", stream);
    }
    if (run.dout == nullptr) return;

    // ---- backward
    float *dh1 = pool.rf(R * C), *dym2 = pool.rf(R * C), *dact = pool.rf(Bm * F), *dug = pool.rf(Bm * 2 * F);
    float *dln = pool.rf(R * C), *dtmp = pool.rf(R * C), *dyo = pool.rf(R * C), *dattn = pool.rf(R * C);
    float *jdq = pool.rf(R * C), *jdk = pool.f(R * C), *jdv = pool.f(R * C), *jdo = pool.rf(R * C);
    float *dyq = pool.rf(R * C), *dyk = pool.rf(R * C), *dqkv = pool.rf(R * 3 * C);
    uint16_t* attn_t = xt ? pool.ru(R * C) : nullptr;
    // a product's dY rows (T N) and dgrad's sums (T K): q, k, v's and o's over every row, the MLP's a band; a frozen
    // weight takes no gradient, so no N K for a wgrad's
    const int64_t mx_TNb = std::max(R * 3 * C, Bm * 2 * F), mx_TK = std::max(R * C, Bm * F);
    uint16_t *t16 = pool.ru(mx_TNb), *t16t = xt ? pool.ru(mx_TNb) : nullptr;
    float* trs = pool.rf(R);
    float* sums = pool.rf(mx_TK);
    auto bwd = [&](int i, const uint16_t* x16, const float* rsx, const uint16_t* xti, float* dy, float* dxv,
                   int64_t n = -1, int64_t r0 = 0) {
        train::ad_product_bwd(run.p[i], run.ad[i], x16, rsx, xti, dy, dxv, false, keep_at(i, r0), adsc, t16, trs,
                              t16t, sums, n < 0 ? R : n, gemm, stream);
    };
    // the MLP's gated residual
    train::copy_rows(run.dout, R * C, dh1, stream);
    train::mul_cols(run.dout, mod + 5 * C, R * C, C, dym2, stream);
    for (int64_t r0 = 0; r0 < R; r0 += Bm) {   // down, SiTU and up_gate a band at a time
        const int64_t n = std::min(Bm, R - r0);
        bwd(kMossDown, act16 + r0 * F, act_s + r0, act_t, dym2 + r0 * C, dact, n, r0);
        dit_rows::situ_bwd(yug + r0 * 2 * F, dact, n, F, dug, stream);
        bwd(kMossUpGate, ln2_16 + r0 * C, ln2_s + r0, ln2_t, dug, dln + r0 * C, n, r0);
    }
    tm.mark("bwd mlp", stream);
    dit_rows::rms_affine_bwd(h1, inv2, run.n2, mod + 4 * C, dln, R, C, dtmp, stream);
    train::add_into(dh1, dtmp, R * C, stream);
    // the attention's gated residual
    tm.mark("bwd norm2", stream);
    train::mul_cols(dh1, mod + 2 * C, R * C, C, dyo, stream);
    if (run.dx != nullptr) train::add_into(run.dx, dh1, R * C, stream);
    if (xt) train::transpose_f16(jo, R, C, attn_t, stream);
    bwd(kMossO, jo, vs + 1, attn_t, dyo, dattn);
    tm.mark("bwd o", stream);
    train::copy_rows(dattn, R * C, jdo, stream);
    // the attention ran on v / vs[0]: its output's gradient comes in times vs[0] and dV leaves divided by it
    train::scale_by(jdo, R * C, vs, false, stream);
    qi_attention_bwd(jq, jk, jv, jof, lse, jdo, R, heads, hd, jdq, jdk, jdv, true, gemm, stream);
    train::scale_by(jdv, R * C, vs, true, stream);
    tm.mark("bwd attention", stream);
    dit_rows::qk_rope_bwd(yq, yk, run.nq, run.nk, run.cos, run.sin, true, jdq, jdk, R, heads, hd, dyq, dyk, nullptr,
                          nullptr, stream, true);
    tm.mark("bwd rope", stream);
    join3(dyq, dyk, jdv, R, C, dqkv, stream);
    bwd(kMossQkv, ln1_16, ln1_s, ln1_t, dqkv, dln);
    tm.mark("bwd qkv", stream);
    dit_rows::rms_affine_bwd(run.x, inv1, run.n1, mod + C, dln, R, C, dtmp, stream);
    if (run.dx != nullptr) train::add_into(run.dx, dtmp, R * C, stream);
    tm.mark("bwd norm1", stream);
}

}  // namespace strata::image
