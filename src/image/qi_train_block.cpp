// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_train_block.cpp - see include/strata/image/qi_train.hpp's qi_block.  One Qwen-Image-2.1 block
// forwards with a tape and backwards through it.  Every buffer has one role and one owner (train::Pool): the tape
// holds what the backward reads, and the gradients live beside it.
#include "strata/image/qi_train.hpp"
#include "strata/image/qi_kernels.hpp"

#include "strata/core/gpu.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

constexpr float kEps = 1e-6f;   // the q and k norms' eps, as the inference takes it

// FP32 rows [T][C] as the shared row kernels read a product's output
MfOut rows_of(const float* p, int64_t C) {
    MfOut m;
    m.f = p;
    m.C = C;
    return m;
}

// The two row ranges a modulation tells apart: the text's rows [0, nt) take row 1 of a [2][C] modulation (t = 0), the
// image's [nt, R) row 0 (the step's t).  f(r0, n, row) runs one range.
template<class F> void by_range(int64_t nt, int64_t R, F f) {
    if (nt > 0) f((int64_t) 0, nt, 1);
    if (R > nt) f(nt, R - nt, 0);
}

}  // namespace

void qi_block(const QiBlockRun& run, prefill::Gemm& gemm, void* s) {
    const int64_t C = run.C, F = run.F, H = run.heads, hd = run.hd, R = run.rows, nt = run.n_text;
    if (C != H * hd) throw std::invalid_argument("qi_block: heads * head_dim must be the width");
    const train::Product* P = run.p;
    auto mod = [&](int k, int row) { return run.mod + ((int64_t) k * 2 + row) * C; };
    train::Pool pool(s);
    const int64_t Rp = qi_attn_rows(R);
    float* zero = pool.f(C);   // LnMod's shift: Qwen-Image-2.1's norms scale only

    // ---- the tape
    uint16_t* ln1_16 = pool.u(R * C);
    float *ln1_s = pool.f(R), *mean1 = pool.f(R), *inv1 = pool.f(R);
    float *yq = pool.f(R * C), *yk = pool.f(R * C), *yv = pool.f(R * C), *yo = pool.f(R * C), *h1 = pool.f(R * C);
    // q, k and v rounded up to the shared attention's rows, zero past R as the pool leaves them
    uint16_t *jq = pool.u(Rp * C), *jk = pool.u(Rp * C), *jv = pool.u(Rp * C), *jo = pool.u(R * C);
    float* jof = pool.f(R * C);   // the attention's FP32 output: the backward's D reads it
    auto* vamax = reinterpret_cast<uint32_t*>(pool.f(1));
    float *vs = pool.f(R + 1), *hsv = pool.f(H), *lse = pool.f(R * H);
    uint16_t* ln2_16 = pool.u(R * C);
    float *ln2_s = pool.f(R), *mean2 = pool.f(R), *inv2 = pool.f(R);
    float* gu = pool.f(R * 2 * F);   // gate_up's output, the SwiGLU's input
    uint16_t* sw16 = pool.u(R * F);
    float* sw_s = pool.f(R);
    float* ydn = pool.f(R * C);

    // ---- the adapters: what each adapted product's forward keeps, and the transients they share
    train::AdKeep keep[kQiProducts];
    int64_t maxN = 0, maxRank = 0, maxScratch = 0, maxTmp = 0;
    bool lokr = false;
    for (int i = 0; i < kQiProducts; ++i) {
        const train::Adapter* a = run.ad[i];
        if (a == nullptr) continue;
        const int64_t N = P[i].N, K = P[i].K, rk = a->rank;
        maxN = std::max(maxN, N);
        maxRank = std::max(maxRank, rk);
        maxScratch = std::max(maxScratch, R * rk + R * N + N * rk + rk * K);
        maxTmp = std::max(maxTmp, 2 * R * N + R * rk);
        lokr = lokr || a->method == train::AdapterMethod::lokr;
        keep[i] = {pool.u(R * rk), pool.f(R), pool.u(R * rk)};
    }
    train::AdScratch sc;
    if (maxN > 0) {
        sc.hf = pool.f(R * maxRank);
        sc.y1 = pool.f(R * maxN);
        sc.y2 = pool.f(R * maxN);
        sc.fa = pool.f(maxN);
        sc.fb = pool.f(maxN);
        sc.scratch = pool.f(maxScratch);
        sc.tmp16 = pool.u(maxTmp);
        sc.rsT = pool.f(R);
    }
    auto product = [&](int i, const uint16_t* x16, const float* rs, float* y) {
        gemm.f16(x16, P[i].w, y, R, P[i].N, P[i].K);
        train::ad_product_fwd(run.ad[i], x16, rs, y, nullptr, R, P[i].N, P[i].K, keep[i], sc, gemm, s);
    };

    // ---- forward
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        dit_rows::rows16(dit_rows::LnMod{run.x + r0 * C, zero, mod(0, row), C}, n, C, ln1_16 + r0 * C, ln1_s + r0, s,
                         mean1 + r0, inv1 + r0);
    });
    product(kQiQ, ln1_16, ln1_s, yq);
    product(kQiK, ln1_16, ln1_s, yk);
    product(kQiV, ln1_16, ln1_s, yv);
    // v's one power of two (its rows can pass FP16's reach): jv, jo and jof are the true rows divided by vs[0], and
    // vs[1 ..] is the output projection's row scale that puts it back
    train::v_scale(yv, R * C, nullptr, 0, vamax, vs, R, s);
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) H), [=](sycl::id<1> id) { hsv[id[0]] = vs[0]; });
    // the inference's own kernel: each head of q and k normed with its weight, then rotated in pairs (2j, 2j + 1)
    dit_rows::qk_rope16(rows_of(yq, C), rows_of(yk, C), rows_of(yv, C), run.nq, run.nk, run.cos, run.sin, hsv, R, H,
                        hd, false, jq, jk, jv, s, kEps);
    qi_attention_train(jq, jk, jv, R, H, hd, jof, lse, s, nt);
    train::f16_copy(jof, R * C, jo, s);
    product(kQiO, jo, vs + 1, yo);
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        dit_rows::gate_add_to(h1 + r0 * C, run.x + r0 * C, rows_of(yo + r0 * C, C), mod(1, row), n, C, s);
        dit_rows::rows16(dit_rows::LnMod{h1 + r0 * C, zero, mod(2, row), C}, n, C, ln2_16 + r0 * C, ln2_s + r0, s,
                         mean2 + r0, inv2 + r0);
    });
    product(kQiGateUp, ln2_16, ln2_s, gu);
    dit_rows::rows16(dit_rows::SwiGlu{rows_of(gu, 2 * F), F}, R, F, sw16, sw_s, s);
    product(kQiDown, sw16, sw_s, ydn);
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        dit_rows::gate_add_to(run.out + r0 * C, h1 + r0 * C, rows_of(ydn + r0 * C, C), mod(3, row), n, C, s);
    });
    if (run.dout == nullptr) return;

    // ---- backward, from the output's gradient
    const int64_t W = std::max(2 * F, C);   // the widest product's rows
    uint16_t *t16 = pool.u(R * W), *t16t = pool.u(R * W), *xt = pool.u(R * F);
    float* trs = pool.f(R);
    float* sums = pool.f(R * F + (lokr ? 4 * F * C : 0));   // a dgrad's T K (a LoKr's N K twice more)
    float *dh1 = pool.f(R * C), *dy = pool.f(R * C), *dsw = pool.f(R * F), *dgu = pool.f(R * 2 * F);
    float *dln = pool.f(R * C), *dtmp = pool.f(R * C);
    auto backward = [&](int i, const uint16_t* x16, const float* rs, float* dY, float* dX, bool acc) {
        if (run.ad[i] != nullptr || P[i].dw != nullptr) train::transpose_f16(x16, R, P[i].K, xt, s);   // dA's, dW's
        train::ad_product_bwd(P[i], run.ad[i], x16, rs, xt, dY, dX, acc, keep[i], sc, t16, trs, t16t, sums, R, gemm,
                              s);
    };
    // the MLP's gated residual: dh1 is the output's gradient as it reaches the first residual's sum
    train::copy_rows(run.dout, R * C, dh1, s);
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        train::mul_cols(run.dout + r0 * C, mod(3, row), n * C, C, dy + r0 * C, s);
    });
    backward(kQiDown, sw16, sw_s, dy, dsw, false);
    dit_rows::swiglu_bwd(gu, dsw, R, F, dgu, s);
    backward(kQiGateUp, ln2_16, ln2_s, dgu, dln, false);
    // the second norm: its input's gradient joins the residual's own
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        dit_rows::ln_mod_bwd(h1 + r0 * C, mean2 + r0, inv2 + r0, mod(2, row), dln + r0 * C, n, C, dtmp + r0 * C,
                             nullptr, nullptr, s);
    });
    train::add_into(dh1, dtmp, R * C, s);
    // the attention's gated residual and its output projection
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        train::mul_cols(dh1 + r0 * C, mod(1, row), n * C, C, dy + r0 * C, s);
    });
    float *jdo = pool.f(R * C), *jdq = pool.f(R * C), *jdk = pool.f(R * C), *jdv = pool.f(R * C);
    backward(kQiO, jo, vs + 1, dy, jdo, false);
    // the attention ran on v / vs[0]: its output's gradient comes in times vs[0] and dV leaves divided by it
    train::scale_by(jdo, R * C, vs, false, s);
    qi_attention_bwd(jq, jk, jv, jof, lse, jdo, R, H, hd, jdq, jdk, jdv, false, gemm, s, nt);
    train::scale_by(jdv, R * C, vs, true, s);
    float *dyq = pool.f(R * C), *dyk = pool.f(R * C);
    dit_rows::qk_rope_bwd(yq, yk, run.nq, run.nk, run.cos, run.sin, true, jdq, jdk, R, H, hd, dyq, dyk, nullptr,
                          nullptr, s, false, kEps);
    backward(kQiQ, ln1_16, ln1_s, dyq, dln, false);
    backward(kQiK, ln1_16, ln1_s, dyk, dln, true);
    backward(kQiV, ln1_16, ln1_s, jdv, dln, true);
    if (run.dx == nullptr) return;
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        dit_rows::ln_mod_bwd(run.x + r0 * C, mean1 + r0, inv1 + r0, mod(0, row), dln + r0 * C, n, C, dtmp + r0 * C,
                             nullptr, nullptr, s);
    });
    train::copy_rows(dh1, R * C, run.dx, s);
    train::add_into(run.dx, dtmp, R * C, s);
}

}  // namespace strata::image
