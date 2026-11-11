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
#include <cstdint>
#include <stdexcept>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

constexpr float kEps = 1e-6f;   // the LayerNorms' and the head norms' eps, as anima_dit.cpp takes them

// GELU (erf form) of a product's sums, rows16's Op (anima_dit.cpp's GeluErf)
struct GeluErf {
    MfOut x;
    template<class R> dit_rows::Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const dit_rows::Stats&) const { return gelu_erf(x.at(t, c)); }
    float bound(int64_t t, int64_t c) const { return sycl::fmax(x.at(t, c), 0.17f); }
};

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
    auto rows_of_product = [&](int i) { return i == kAnCk || i == kAnCv ? L : R; };
    const train::Product* prods[kAnProducts];
    for (int i = 0; i < kAnProducts; ++i) prods[i] = &run.p[i];
    for (int i = 0; i < kAnProducts; ++i) {
        const train::Adapter* az = run.ad[i];
        if (az == nullptr) continue;
        const int64_t T = rows_of_product(i), rk = az->rank;
        keep[i] = train::AdKeep{pool.ru(T * rk), pool.rf(T), pool.ru(T * rk)};
    }
    train::AdScratch adsc =
        train::take_ad_scratch(pool, train::ad_sizes(prods, run.ad, kAnProducts, rows_of_product), false);
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
    // the shared attention's rows (qi_attention_train): q [Rp][C], k and v [Lp][C], the rows past R and L zero
    const int64_t Lp = qi_attn_rows(L);
    uint16_t *cq16 = pool.u(Rp * C), *ck16 = pool.u(Lp * C), *cv16 = pool.u(Lp * C);
    dit_rows::qk_norm16(rows_of(ycq, C), run.cqn, R, H, hd, cq16, stream, kEps);
    dit_rows::qk_norm16(rows_of(yck, C), run.ckn, L, H, hd, ck16, stream, kEps);
    dit_rows::qk_norm16(rows_of(ycv, C), nullptr, L, H, hd, cv16, stream, kEps);
    float* o32 = pool.rf(R * C);
    float* lse_c = pool.rf(R * H);
    qi_attention_train(cq16, ck16, cv16, R, H, hd, o32, lse_c, stream, 0, L);
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
    float *dq32 = pool.rf(R * C), *dk32 = pool.rf(L * C), *dv32 = pool.rf(L * C);
    qi_attention_bwd(cq16, ck16, cv16, o32, lse_c, do32, R, H, hd, dq32, dk32, dv32, false, gemm, stream, 0, L);
    float *dycq = pool.rf(R * C), *dyck = pool.rf(L * C), *dctx = pool.rf(L * Cc);
    // the norms' backward (the shared kernel, one tensor at a time: q's rows and k's are not the same count)
    dit_rows::qk_rope_bwd(ycq, nullptr, run.cqn, nullptr, nullptr, nullptr, false, dq32, nullptr, R, H, hd, dycq,
                          nullptr, nullptr, nullptr, stream, true, kEps);
    dit_rows::qk_rope_bwd(yck, nullptr, run.ckn, nullptr, nullptr, nullptr, false, dk32, nullptr, L, H, hd, dyck,
                          nullptr, nullptr, nullptr, stream, true, kEps);
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
