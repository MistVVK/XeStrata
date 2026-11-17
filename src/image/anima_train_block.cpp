// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/anima_train_block.cpp - see include/strata/image/anima_train.hpp's anima_block.  One Anima block forwards
// and backwards, out of the shared row kernels and products; both attentions through the shared attention
// (qi_attention_train and qi_attention_bwd), the cross-attention's keys the context's rows.
#include "strata/image/anima_train.hpp"
#include "strata/image/gelu.hpp"
#include "strata/image/qi_kernels.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

constexpr float kEps = 1e-6f;   // the LayerNorms' and the head norms' eps, as anima_dit.cpp takes them

// GELU (erf form) of a product's sums, rows16's Op (anima_dit.cpp's GeluErf)
struct GeluErf {
    DitOut x;
    template<class R> dit_rows::Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const dit_rows::Stats&) const { return gelu_erf(x.at(t, c)); }
    float bound(int64_t t, int64_t c) const { return sycl::fmax(x.at(t, c), 0.17f); }
};

// The MLP's rows a pass: its two products and their backwards a band of this many rows at a time, so its widest
// scratch (F a row) is this many rows, not the image's (Qwen-Image-2.1's and MOSS-Image's blocks take the same 1024)
constexpr int64_t kMlpRows = 1024;

// whether the products' backwards read their inputs transposed (train::ad_needs_xt: a GPU without transposed products,
// a LoKr): the transposed copies are made and kept only then
bool needs_xt(const AnimaBlockRun& run, prefill::Gemm& gemm) {
    for (int i = 0; i < kAnProducts; ++i)
        if (train::ad_needs_xt(run.p[i], run.ad[i], gemm)) return true;
    return false;
}

// What a block's backward reads of its forward (its tape), taken in this order from a pool or a Sizer
struct Tape {
    uint16_t *a16, *a_t, *jq, *jk, *jv, *jo, *b16, *b_t, *cq16, *ck16, *cv16, *o16, *o_t, *c16, *c_t, *g16, *g_t;
    float *a_s, *mean1, *inv1, *yq, *yk, *jof, *vs, *lse, *yo, *h1;
    float *b_s, *mean2, *inv2, *ycq, *yck, *o32, *lse_c, *o_s, *yco, *h2;
    float *c_s, *mean3, *inv3, *yl1, *g_s, *yl2;
    train::AdKeep keep[kAnProducts];
};

template <class Take> Tape take_tape(Take& t, const AnimaBlockRun& run, bool xt) {
    const int64_t C = run.C, F = run.F, H = run.heads, R = run.rows, L = run.L;
    const int64_t Rp = qi_attn_rows(R), Lp = qi_attn_rows(L);
    Tape p{};
    // the self-attention
    p.a16 = t.ru(R * C);
    p.a_t = xt ? t.ru(R * C) : nullptr;
    p.a_s = t.rf(R);
    p.mean1 = t.rf(R);
    p.inv1 = t.rf(R);
    p.yq = t.rf(R * C);
    p.yk = t.rf(R * C);
    p.jq = t.u(Rp * C);   // q, k and v rounded up to the shared attention's rows, zero past R
    p.jk = t.u(Rp * C);
    p.jv = t.u(Rp * C);
    p.jo = t.ru(R * C);
    p.jof = t.rf(R * C);
    p.vs = t.rf(R + 1);
    p.lse = t.rf(R * H);
    p.yo = t.rf(R * C);
    p.h1 = t.rf(R * C);
    // the cross-attention: q [Rp][C], k and v [Lp][C], the rows past R and L zero
    p.b16 = t.ru(R * C);
    p.b_t = xt ? t.ru(R * C) : nullptr;
    p.b_s = t.rf(R);
    p.mean2 = t.rf(R);
    p.inv2 = t.rf(R);
    p.ycq = t.rf(R * C);
    p.yck = t.rf(L * C);
    p.cq16 = t.u(Rp * C);
    p.ck16 = t.u(Lp * C);
    p.cv16 = t.u(Lp * C);
    p.o32 = t.rf(R * C);
    p.lse_c = t.rf(R * H);
    p.o16 = t.ru(R * C);
    p.o_t = xt ? t.ru(R * C) : nullptr;
    p.o_s = t.rf(R);
    p.yco = t.rf(R * C);
    p.h2 = t.rf(R * C);
    // the MLP
    p.c16 = t.ru(R * C);
    p.c_t = xt ? t.ru(R * C) : nullptr;
    p.c_s = t.rf(R);
    p.mean3 = t.rf(R);
    p.inv3 = t.rf(R);
    p.yl1 = t.rf(R * F);
    p.g16 = t.ru(R * F);
    p.g_t = xt ? t.ru(R * F) : nullptr;
    p.g_s = t.rf(R);
    p.yl2 = t.rf(R * C);
    for (int i = 0; i < kAnProducts; ++i)   // what each adapted product's forward keeps
        if (run.ad[i] != nullptr) {
            const int64_t T = i == kAnCk || i == kAnCv ? L : R, rk = run.ad[i]->rank;
            p.keep[i] = train::AdKeep{t.ru(T * rk), t.rf(T), t.ru(T * rk)};
        }
    return p;
}

}  // namespace

size_t anima_block_tape_bytes(const AnimaBlockRun& run, prefill::Gemm& gemm) {
    train::Sizer z;
    take_tape(z, run, needs_xt(run, gemm));
    return z.bytes;
}

void anima_block(const AnimaBlockRun& run, prefill::Gemm& gemm, void* stream, train::Arena* arena) {
    const int64_t C = run.C, F = run.F, H = run.heads, hd = run.hd, R = run.rows, L = run.L, Cc = run.Cc;
    if (C != H * hd) throw std::invalid_argument("anima_block: heads * head_dim must be the width");
    // the call's buffers from the arena (kept from call to call); a buffer written whole before it is read is not
    // zeroed (rf, ru), as moss_block's
    train::Pool pool(stream, arena);
    // the forward's values the backward reads: the block's own tape (a kept forward's, read back by a replay), else
    // the call's scratch
    const bool replay = run.tape != nullptr && run.tape_replay;
    if (replay && run.dout == nullptr) throw std::invalid_argument("anima_block: a kept tape is read by the backward");
    std::optional<train::Pool> tpool;
    if (run.tape != nullptr) tpool.emplace(stream, run.tape, replay);
    const bool xt = needs_xt(run, gemm);
    const Tape tp = run.tape != nullptr ? take_tape(*tpool, run, xt) : take_tape(pool, run, xt);
    const float* msa = run.mod_sa;
    const float* mca = run.mod_ca;
    const float* mml = run.mod_ml;
    const train::AdKeep* keep = tp.keep;

    // the MLP's band of rows: every row at once where the backward reads the inputs transposed (a band of them is not
    // a slice of the transposed rows)
    const int64_t Bm = xt ? R : std::min(R, kMlpRows);
    auto rows_of_product = [&](int i) { return i == kAnCk || i == kAnCv ? L : i == kAnL1 || i == kAnL2 ? Bm : R; };
    // ---- the adapters' shared scratch, sized for the largest product's rows
    const train::Product* prods[kAnProducts];
    for (int i = 0; i < kAnProducts; ++i) prods[i] = &run.p[i];
    train::AdScratch adsc =
        train::take_ad_scratch(pool, train::ad_sizes(prods, run.ad, kAnProducts, rows_of_product));
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
    // product i of T rows from row r0: the base's sums, then its row scales, no bias, and its adapter's delta
    auto fwd = [&](int i, const uint16_t* x16, const float* rs, float* y, int64_t T, int64_t r0 = 0) {
        const train::Product& P = run.p[i];
        gemm.f16(x16, P.w, y, T, P.N, P.K);
        train::ad_product_fwd(run.ad[i], x16, rs, y, nullptr, T, P.N, P.K, keep_at(i, r0), adsc, gemm, stream);
    };

    if (!replay) {   // a kept tape's backward has its forward already
        // ---- forward: the self-attention
        dit_rows::rows16(dit_rows::LnMod{run.x, msa, msa + C, C}, R, C, tp.a16, tp.a_s, stream, tp.mean1, tp.inv1);
        if (xt) train::transpose_f16(tp.a16, R, C, tp.a_t, stream);
        float* yv = pool.rf(R * C);
        fwd(kAnSq, tp.a16, tp.a_s, tp.yq, R);
        fwd(kAnSk, tp.a16, tp.a_s, tp.yk, R);
        fwd(kAnSv, tp.a16, tp.a_s, yv, R);
        auto* vamax = reinterpret_cast<uint32_t*>(pool.f(1));
        float* hsv = pool.rf(H);
        float* vs = tp.vs;
        train::v_scale(yv, R * C, nullptr, 0, vamax, vs, R, stream);
        dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) H), [=](sycl::id<1> id) { hsv[id[0]] = vs[0]; });
        dit_rows::qk_rope16(rows_of(tp.yq, C), rows_of(tp.yk, C), rows_of(yv, C), run.sqn, run.skn, run.cos, run.sin,
                            hsv, R, H, hd, true, tp.jq, tp.jk, tp.jv, stream, kEps);
        qi_attention_train(tp.jq, tp.jk, tp.jv, R, H, hd, tp.jof, tp.lse, stream);
        train::f16_copy(tp.jof, R * C, tp.jo, stream);
        fwd(kAnSo, tp.jo, vs + 1, tp.yo, R);
        dit_rows::gate_add_to(tp.h1, run.x, rows_of(tp.yo, C), msa + 2 * C, R, C, stream);

        // ---- the cross-attention
        dit_rows::rows16(dit_rows::LnMod{tp.h1, mca, mca + C, C}, R, C, tp.b16, tp.b_s, stream, tp.mean2, tp.inv2);
        if (xt) train::transpose_f16(tp.b16, R, C, tp.b_t, stream);
        float* ycv = pool.rf(L * C);
        fwd(kAnCq, tp.b16, tp.b_s, tp.ycq, R);
        fwd(kAnCk, run.ctx16, run.ctx_rs, tp.yck, L);
        fwd(kAnCv, run.ctx16, run.ctx_rs, ycv, L);
        dit_rows::qk_norm16(rows_of(tp.ycq, C), run.cqn, R, H, hd, tp.cq16, stream, kEps);
        dit_rows::qk_norm16(rows_of(tp.yck, C), run.ckn, L, H, hd, tp.ck16, stream, kEps);
        dit_rows::qk_norm16(rows_of(ycv, C), nullptr, L, H, hd, tp.cv16, stream, kEps);
        qi_attention_train(tp.cq16, tp.ck16, tp.cv16, R, H, hd, tp.o32, tp.lse_c, stream, 0, L);
        dit_rows::rows16(dit_rows::Plain{tp.o32, C}, R, C, tp.o16, tp.o_s, stream);
        if (xt) train::transpose_f16(tp.o16, R, C, tp.o_t, stream);
        fwd(kAnCo, tp.o16, tp.o_s, tp.yco, R);
        dit_rows::gate_add_to(tp.h2, tp.h1, rows_of(tp.yco, C), mca + 2 * C, R, C, stream);

        // ---- the MLP, a band of rows at a time
        dit_rows::rows16(dit_rows::LnMod{tp.h2, mml, mml + C, C}, R, C, tp.c16, tp.c_s, stream, tp.mean3, tp.inv3);
        if (xt) train::transpose_f16(tp.c16, R, C, tp.c_t, stream);
        for (int64_t r0 = 0; r0 < R; r0 += Bm) {
            const int64_t n = std::min(Bm, R - r0);
            fwd(kAnL1, tp.c16 + r0 * C, tp.c_s + r0, tp.yl1 + r0 * F, n, r0);
        }
        dit_rows::rows16(GeluErf{rows_of(tp.yl1, F)}, R, F, tp.g16, tp.g_s, stream);
        if (xt) train::transpose_f16(tp.g16, R, F, tp.g_t, stream);
        for (int64_t r0 = 0; r0 < R; r0 += Bm) {
            const int64_t n = std::min(Bm, R - r0);
            fwd(kAnL2, tp.g16 + r0 * F, tp.g_s + r0, tp.yl2 + r0 * C, n, r0);
        }
        dit_rows::gate_add_to(run.out, tp.h2, rows_of(tp.yl2, C), mml + 2 * C, R, C, stream);
    }
    if (run.dout == nullptr) return;

    // ---- backward
    // a product's dY rows (T N) and dgrad's sums (T K): the attentions' over every row (the context's L), the MLP's a
    // band; a frozen weight takes no gradient, so no N K for a wgrad's
    const int64_t RL = std::max(R, L), mx_TN = std::max(RL * std::max(C, Cc), Bm * F), mx_TK = mx_TN;
    uint16_t *t16 = pool.ru(mx_TN), *t16t = xt ? pool.ru(mx_TN) : nullptr;
    float* trs = pool.rf(RL);
    float* sums = pool.rf(mx_TK);
    auto bwd = [&](int i, const uint16_t* x16, const float* rs, const uint16_t* xti, float* dy, float* dxv, int64_t T,
                   bool acc, int64_t r0 = 0) {
        train::ad_product_bwd(run.p[i], run.ad[i], x16, rs, xti, dy, dxv, acc, keep_at(i, r0), adsc, t16, trs, t16t,
                              sums, T, gemm, stream);
    };
    float *dh2 = pool.rf(R * C), *dy = pool.rf(R * C), *dtmp = pool.rf(R * C), *dln = pool.rf(R * C);
    float* dz = pool.rf(Bm * F);
    // the MLP's gated residual
    train::copy_rows(run.dout, R * C, dh2, stream);
    if (run.dmod_ml != nullptr)
        dit_rows::gate_add_bwd(tp.yl2, nullptr, run.dout, R, C, nullptr, run.dmod_ml + 2 * C, stream);
    train::mul_cols(run.dout, mml + 2 * C, R * C, C, dy, stream);
    for (int64_t r0 = 0; r0 < R; r0 += Bm) {   // layer2, the GELU and layer1 a band at a time
        const int64_t n = std::min(Bm, R - r0);
        bwd(kAnL2, tp.g16 + r0 * F, tp.g_s + r0, tp.g_t, dy + r0 * C, dz, n, false, r0);   // dz: the GELU's output's
        {
            const float* v = tp.yl1 + r0 * F;
            float* d = dz;
            dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) (n * F)),
                                                [=](sycl::id<1> id) { d[id[0]] *= gelu_erf_grad(v[id[0]]); });
        }
        bwd(kAnL1, tp.c16 + r0 * C, tp.c_s + r0, tp.c_t, dz, dln + r0 * C, n, false, r0);   // the norm's output's
    }
    dit_rows::ln_mod_bwd(tp.h2, tp.mean3, tp.inv3, mml + C, dln, R, C, dtmp, run.dmod_ml,
                         run.dmod_ml != nullptr ? run.dmod_ml + C : nullptr, stream);
    train::add_into(dh2, dtmp, R * C, stream);
    // the cross-attention's gated residual
    float* dh1 = pool.rf(R * C);
    train::copy_rows(dh2, R * C, dh1, stream);
    if (run.dmod_ca != nullptr)
        dit_rows::gate_add_bwd(tp.yco, nullptr, dh2, R, C, nullptr, run.dmod_ca + 2 * C, stream);
    train::mul_cols(dh2, mca + 2 * C, R * C, C, dy, stream);
    float* do32 = pool.rf(R * C);
    bwd(kAnCo, tp.o16, tp.o_s, tp.o_t, dy, do32, R, false);
    float *dq32 = pool.rf(R * C), *dk32 = pool.rf(L * C), *dv32 = pool.rf(L * C);
    qi_attention_bwd(tp.cq16, tp.ck16, tp.cv16, tp.o32, tp.lse_c, do32, R, H, hd, dq32, dk32, dv32, false, gemm, stream,
                     0, L);
    float *dycq = pool.rf(R * C), *dyck = pool.rf(L * C), *dctx = pool.rf(L * Cc);
    // the norms' backward (the shared kernel, one tensor at a time: q's rows and k's are not the same count)
    dit_rows::qk_rope_bwd(tp.ycq, nullptr, run.cqn, nullptr, nullptr, nullptr, false, dq32, nullptr, R, H, hd, dycq,
                          nullptr, nullptr, nullptr, stream, true, kEps);
    dit_rows::qk_rope_bwd(tp.yck, nullptr, run.ckn, nullptr, nullptr, nullptr, false, dk32, nullptr, L, H, hd, dyck,
                          nullptr, nullptr, nullptr, stream, true, kEps);
    bwd(kAnCk, run.ctx16, run.ctx_rs, run.ctx16t, dyck, dctx, L, false);
    bwd(kAnCv, run.ctx16, run.ctx_rs, run.ctx16t, dv32, dctx, L, false);
    bwd(kAnCq, tp.b16, tp.b_s, tp.b_t, dycq, dy, R, false);
    dit_rows::ln_mod_bwd(tp.h1, tp.mean2, tp.inv2, mca + C, dy, R, C, dtmp, run.dmod_ca,
                         run.dmod_ca != nullptr ? run.dmod_ca + C : nullptr, stream);
    train::add_into(dh1, dtmp, R * C, stream);
    // the self-attention's gated residual
    if (run.dmod_sa != nullptr)
        dit_rows::gate_add_bwd(tp.yo, nullptr, dh1, R, C, nullptr, run.dmod_sa + 2 * C, stream);
    if (run.dx != nullptr) train::add_into(run.dx, dh1, R * C, stream);
    train::mul_cols(dh1, msa + 2 * C, R * C, C, dy, stream);
    uint16_t* jo_t = xt ? pool.ru(R * C) : nullptr;
    if (xt) train::transpose_f16(tp.jo, R, C, jo_t, stream);
    float* jdo = pool.rf(R * C);
    bwd(kAnSo, tp.jo, tp.vs + 1, jo_t, dy, jdo, R, false);
    train::scale_by(jdo, R * C, tp.vs, false, stream);
    float *jdq = pool.rf(R * C), *jdk = pool.f(R * C), *jdv = pool.f(R * C);
    qi_attention_bwd(tp.jq, tp.jk, tp.jv, tp.jof, tp.lse, jdo, R, H, hd, jdq, jdk, jdv, true, gemm, stream);
    train::scale_by(jdv, R * C, tp.vs, true, stream);
    float *dyq = pool.rf(R * C), *dyk = pool.rf(R * C);
    dit_rows::qk_rope_bwd(tp.yq, tp.yk, run.sqn, run.skn, run.cos, run.sin, true, jdq, jdk, R, H, hd, dyq, dyk, nullptr,
                          nullptr, stream, true);
    bwd(kAnSq, tp.a16, tp.a_s, tp.a_t, dyq, dy, R, false);
    bwd(kAnSk, tp.a16, tp.a_s, tp.a_t, dyk, dy, R, true);
    bwd(kAnSv, tp.a16, tp.a_s, tp.a_t, jdv, dy, R, true);
    dit_rows::ln_mod_bwd(run.x, tp.mean1, tp.inv1, msa + C, dy, R, C, dtmp, run.dmod_sa,
                         run.dmod_sa != nullptr ? run.dmod_sa + C : nullptr, stream);
    if (run.dx != nullptr) train::add_into(run.dx, dtmp, R * C, stream);
}

}  // namespace strata::image
