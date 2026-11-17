// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_train_block.cpp - see include/strata/image/qi_train.hpp's qi_block.  One Qwen-Image-2.1 block
// forwards with a tape and backwards through it. The tape holds what the backward reads; scratch whose uses do not
// overlap shares memory on the ordered stream.
#include "strata/image/qi_train.hpp"
#include "strata/image/qi_kernels.hpp"

#include "strata/core/gpu.hpp"
#include "strata/platform/env.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

int64_t setting_integer(const char* text, const char* name) {
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end != text) while (std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (end == text || *end || errno == ERANGE)
        throw std::invalid_argument(std::string("XESTRATA_") + name + ": expected an integer");
    return static_cast<int64_t>(value);
}

constexpr float kEps = 1e-6f;   // the q and k norms' eps, as the inference takes it

// The two row ranges a modulation tells apart: the text's rows [0, nt) take row 1 of a [2][C] modulation (t = 0), the
// image's [nt, R) row 0 (the step's t).  f(r0, n, row) runs one range.
template<class F> void by_range(int64_t nt, int64_t R, F f) {
    if (nt > 0) f((int64_t) 0, nt, 1);
    if (R > nt) f(nt, R - nt, 0);
}

// XESTRATA_TRAIN_PROFILE=1: the stages' times (train::StageTimes), summed over the calls and printed by
// qi_block_profile_report (each stage waits for the GPU, so the sums are the stages' own and the whole runs slower)
const char* const kStage[20] = {"fwd gate", "ln2 bwd", "o bwd", "attention bwd", "qk bwd", "qkv bwd", "norm bwd",
                                 "fwd ln1", "fwd v scale + qk rope", "fwd attention", "fwd ln2", "fwd swiglu", "down bwd",
                                 "swiglu bwd", "gate_up bwd", "mlp gate bwd", "fwd base products", "fwd lora", "",
                                 ""};

train::StageTimes& times() {
    static train::StageTimes t("qi_block");
    return t;
}

// What a block's backward reads of its forward, taken in this order from a pool (or a Sizer)
struct Tape {
    uint16_t *ln1_16, *jq, *jk, *jv, *jo, *ln2_16, *sw16;
    float *ln1_s, *mean1, *inv1, *yq, *yk, *h1, *jof, *vs, *lse, *ln2_s, *mean2, *inv2, *gu, *sw_s;
    train::AdKeep keep[kQiProducts];
};

template<class Take> Tape take_tape(Take& t, const QiBlockRun& run) {
    const int64_t C = run.C, F = run.F, H = run.heads, R = run.rows, Rp = qi_attn_rows(R);
    Tape p{};
    p.ln1_16 = t.ru(R * C);
    p.ln1_s = t.rf(R);
    p.mean1 = t.rf(R);
    p.inv1 = t.rf(R);
    p.yq = t.rf(R * C);
    p.yk = t.rf(R * C);
    p.h1 = t.rf(R * C);
    // q, k and v rounded up to the shared attention's rows, zero past R as the pool leaves them
    p.jq = t.u(Rp * C);
    p.jk = t.u(Rp * C);
    p.jv = t.u(Rp * C);
    p.jo = t.ru(R * C);
    p.jof = t.rf(R * C);   // the attention's FP32 output: the backward's D reads it
    p.vs = t.rf(R + 1);
    p.lse = t.rf(R * H);
    p.ln2_16 = t.ru(R * C);
    p.ln2_s = t.rf(R);
    p.mean2 = t.rf(R);
    p.inv2 = t.rf(R);
    p.gu = t.rf(R * 2 * F);   // gate_up's output, the SwiGLU's input
    p.sw16 = t.ru(R * F);
    p.sw_s = t.rf(R);
    for (int i = 0; i < kQiProducts; ++i)   // what each adapted product's forward keeps
        if (run.ad[i] != nullptr) {
            const int64_t rk = run.ad[i]->rank;
            p.keep[i] = {t.ru(R * rk), t.rf(R), t.ru(R * rk)};
        }
    return p;
}

}  // namespace

size_t qi_block_tape_bytes(const QiBlockRun& run) {
    train::Sizer z;
    take_tape(z, run);
    return z.bytes;
}

void qi_block_profile_report() { times().report(); }

void qi_block(const QiBlockRun& run, prefill::Gemm& gemm, void* s, train::Arena* arena) {
    const int64_t C = run.C, F = run.F, H = run.heads, hd = run.hd, R = run.rows, nt = run.n_text;
    if (C != H * hd) throw std::invalid_argument("qi_block: heads * head_dim must be the width");
    const train::Product* P = run.p;
    auto mod = [&](int k, int row) { return run.mod + ((int64_t) k * 2 + row) * C; };
    train::StageTimes& tm = times();
    tm.begin(s);
    auto mark = [&](int stage) { tm.mark(kStage[stage], s); };
    train::Pool pool(s, arena);
    float* zero = pool.f(C);   // LnMod's shift: Qwen-Image-2.1's norms scale only

    // ---- the tape (Tape, take_tape): from the call's own pool, or from run.tape, kept for the backward
    std::optional<train::Pool> own_tape;
    if (run.tape != nullptr) own_tape.emplace(s, run.tape, run.tape_replay);
    const Tape tp = own_tape ? take_tape(*own_tape, run) : take_tape(pool, run);
    uint16_t *ln1_16 = tp.ln1_16, *jq = tp.jq, *jk = tp.jk, *jv = tp.jv, *jo = tp.jo, *ln2_16 = tp.ln2_16;
    uint16_t* sw16 = tp.sw16;
    float *ln1_s = tp.ln1_s, *mean1 = tp.mean1, *inv1 = tp.inv1, *yq = tp.yq, *yk = tp.yk, *h1 = tp.h1;
    float *jof = tp.jof, *vs = tp.vs, *lse = tp.lse, *ln2_s = tp.ln2_s, *mean2 = tp.mean2, *inv2 = tp.inv2;
    float *gu = tp.gu, *sw_s = tp.sw_s;
    train::AdKeep keep[kQiProducts];
    for (int i = 0; i < kQiProducts; ++i) keep[i] = tp.keep[i];
    // V is dead before the output projection, and that projection is dead before the down projection.
    float* yv = pool.rf(R * C);
    float *yo = yv, *ydn = yv;
    auto* vamax = reinterpret_cast<uint32_t*>(pool.f(1));
    float* hsv = pool.rf(H);

    // The MLP's rows a pass (QiBlockRun::mlp_rows): its products, SwiGLU and their backwards a block of RC rows at a
    // time, so that its widest scratch (2F a row) is RC rows, not R.  Only where its adapters are plain LoRAs (or none)
    // and the GPU takes the transposed products: the transposed views of the rows are not cut into blocks.
    bool chunkable = gemm.f16_batch_ex_ok();
    for (int i = kQiGateUp; i <= kQiDown; ++i)
        chunkable = chunkable && P[i].dw == nullptr &&
                    (run.ad[i] == nullptr || run.ad[i]->method == train::AdapterMethod::lora);
    int64_t mlp_rows = run.mlp_rows;
    if (const char* e = strata::env_get("TRAIN_MLP_ROWS")) mlp_rows = setting_integer(e, "TRAIN_MLP_ROWS");   // fixed, for the checks
    const int64_t RC = chunkable && mlp_rows > 0 ? std::min(R, mlp_rows) : R;
    auto rows_of_product = [&](int i) { return i >= kQiGateUp ? RC : R; };

    // ---- the adapters' transients
    const train::Product* prods[kQiProducts];
    for (int i = 0; i < kQiProducts; ++i) prods[i] = &P[i];
    train::AdScratch sc =
        train::take_ad_scratch(pool, train::ad_sizes(prods, run.ad, kQiProducts, rows_of_product));
    bool lokr = false;
    for (const train::Adapter* a : run.ad)
        lokr = lokr || (a != nullptr && a->method == train::AdapterMethod::lokr);
    // a QLoRA's base (W8A8, ai-toolkit's convrot8): the input's int8 rows, and the FP16 ones where the GPU takes no
    // int8 products
    const bool w8 = P[0].w8 != nullptr;
    int8_t* x8 = nullptr;
    float* sx = nullptr;
    uint16_t *x16f = nullptr, *w16f = nullptr;
    const int64_t maxIn = std::max(R * C, RC * F);   // a product's input values: the attention's or the MLP's
    if (w8) {
        x8 = reinterpret_cast<int8_t*>(pool.ru((maxIn + 1) / 2));
        sx = pool.rf(R);
        if (!train::w8_int8(s)) {
            x16f = pool.ru(maxIn);
            w16f = pool.ru(2 * F * C);
        }
    }
    // an adapter's kept rows from row r0 on (a block of the MLP's)
    auto keep_at = [&](int i, int64_t r0) {
        train::AdKeep k = keep[i];
        if (run.ad[i] != nullptr && r0 > 0) {
            const int64_t rk = run.ad[i]->rank;
            k.h16 += r0 * rk;
            k.hr += r0;
            k.h16t += r0 * rk;   // (scratch alone where the blocks are taken: no transposed view of rows is read)
        }
        return k;
    };
    // product i of n rows from row r0 (x16, rs and y already at it)
    auto product = [&](int i, const uint16_t* x16, const float* rs, float* y, int64_t n = -1, int64_t r0 = 0) {
        if (n < 0) n = R;
        if (i < 3 && run.w_plain[i] != nullptr) gemm.f16(x16, run.w_plain[i], y, n, P[i].N, P[i].K);
        else if (P[i].w8 != nullptr) train::w8_product(x16, n, P[i], y, x8, sx, x16f, w16f, gemm, s);
        else gemm.f16(x16, P[i].w, y, n, P[i].N, P[i].K);
        mark(16);
        train::ad_product_fwd(run.ad[i], x16, rs, y, nullptr, n, P[i].N, P[i].K, keep_at(i, r0), sc, gemm, s);
        mark(17);
    };

    // ---- forward (a kept tape's backward has it already)
    if (run.tape_replay && run.dout == nullptr)
        throw std::invalid_argument("qi_block: a kept tape is read back by the backward only");
    if (!run.tape_replay) {
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        dit_rows::rows16(dit_rows::LnMod{run.x + r0 * C, zero, mod(0, row), C}, n, C, ln1_16 + r0 * C, ln1_s + r0, s,
                         mean1 + r0, inv1 + r0);
    });
    mark(7);
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
    mark(8);
    qi_attention_train(jq, jk, jv, R, H, hd, jof, lse, s, nt);
    train::f16_copy(jof, R * C, jo, s);
    mark(9);
    product(kQiO, jo, vs + 1, yo);
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        dit_rows::gate_add_to(h1 + r0 * C, run.x + r0 * C, rows_of(yo + r0 * C, C), mod(1, row), n, C, s);
        dit_rows::rows16(dit_rows::LnMod{h1 + r0 * C, zero, mod(2, row), C}, n, C, ln2_16 + r0 * C, ln2_s + r0, s,
                         mean2 + r0, inv2 + r0);
    });
    mark(10);
    for (int64_t r0 = 0; r0 < R; r0 += RC) {   // the MLP a block of RC rows at a time
        const int64_t n = std::min(RC, R - r0);
        product(kQiGateUp, ln2_16 + r0 * C, ln2_s + r0, gu + r0 * 2 * F, n, r0);
        dit_rows::rows16(dit_rows::SwiGlu{rows_of(gu + r0 * 2 * F, 2 * F), F}, n, F, sw16 + r0 * F, sw_s + r0, s);
        product(kQiDown, sw16 + r0 * F, sw_s + r0, ydn + r0 * C, n, r0);
    }
    mark(11);
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        dit_rows::gate_add_to(run.out + r0 * C, h1 + r0 * C, rows_of(ydn + r0 * C, C), mod(3, row), n, C, s);
    });
    mark(0);
    }
    if (run.dout == nullptr) return;

    // ---- backward, from the output's gradient
    const int64_t W = std::max(2 * F, C);   // the widest product's rows
    const int64_t RW = std::max(R * C, RC * W);   // a product's rows: the attention's R of C, the MLP's RC of 2F
    uint16_t *t16 = pool.ru(RW), *t16t = pool.ru(RW), *xt = pool.ru(maxIn);
    float* trs = pool.rf(R);
    float* sums = pool.rf(maxIn + (lokr ? 4 * F * C : 0));   // a dgrad's T K (a LoKr's N K twice more)
    float *dh1 = pool.rf(R * C), *dy = yv, *dsw = pool.rf(RC * F), *dgu = pool.rf(RC * 2 * F);
    float *dln = pool.rf(R * C), *dtmp = dy;   // each norm's temporary is dead before dy's next use
    auto backward = [&](int i, const uint16_t* x16, const float* rs, float* dY, float* dX, bool acc, int64_t n = -1,
                        int64_t r0 = 0) {
        if (n < 0) n = R;
        if (train::ad_needs_xt(P[i], run.ad[i], gemm)) train::transpose_f16(x16, n, P[i].K, xt, s);
        train::ad_product_bwd(P[i], run.ad[i], x16, rs, xt, dY, dX, acc, keep_at(i, r0), sc, t16, trs, t16t, sums, n,
                              gemm, s);
    };
    // the MLP's gated residual: dh1 is the output's gradient as it reaches the first residual's sum
    train::copy_rows(run.dout, R * C, dh1, s);
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        train::mul_cols(run.dout + r0 * C, mod(3, row), n * C, C, dy + r0 * C, s);
    });
    mark(15);
    for (int64_t r0 = 0; r0 < R; r0 += RC) {   // the MLP a block of RC rows at a time, as its forward
        const int64_t n = std::min(RC, R - r0);
        backward(kQiDown, sw16 + r0 * F, sw_s + r0, dy + r0 * C, dsw, false, n, r0);
        mark(12);
        dit_rows::swiglu_bwd(gu + r0 * 2 * F, dsw, n, F, dgu, s);
        mark(13);
        backward(kQiGateUp, ln2_16 + r0 * C, ln2_s + r0, dgu, dln + r0 * C, false, n, r0);
        mark(14);
    }
    // the second norm: its input's gradient joins the residual's own
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        dit_rows::ln_mod_bwd(h1 + r0 * C, mean2 + r0, inv2 + r0, mod(2, row), dln + r0 * C, n, C, dtmp + r0 * C,
                             nullptr, nullptr, s);
    });
    train::add_into(dh1, dtmp, R * C, s);
    mark(1);
    // the attention's gated residual and its output projection
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        train::mul_cols(dh1 + r0 * C, mod(1, row), n * C, C, dy + r0 * C, s);
    });
    float *jdo = pool.rf(R * C), *jdq = pool.rf(R * C), *jdk = pool.rf(R * C), *jdv = pool.rf(R * C);
    backward(kQiO, jo, vs + 1, dy, jdo, false);
    mark(2);
    // the attention ran on v / vs[0]: its output's gradient comes in times vs[0] and dV leaves divided by it
    train::scale_by(jdo, R * C, vs, false, s);
    qi_attention_bwd(jq, jk, jv, jof, lse, jdo, R, H, hd, jdq, jdk, jdv, false, gemm, s, nt);
    train::scale_by(jdv, R * C, vs, true, s);
    mark(3);
    // No QK norm weight gradients are taken here: each input pair may be replaced by its gradient.
    float *dyq = yq, *dyk = yk;
    dit_rows::qk_rope_bwd(yq, yk, run.nq, run.nk, run.cos, run.sin, true, jdq, jdk, R, H, hd, dyq, dyk, nullptr,
                          nullptr, s, false, kEps);
    mark(4);
    backward(kQiQ, ln1_16, ln1_s, dyq, dln, false);
    backward(kQiK, ln1_16, ln1_s, dyk, dln, true);
    backward(kQiV, ln1_16, ln1_s, jdv, dln, true);
    mark(5);
    if (run.dx == nullptr) return;
    by_range(nt, R, [&](int64_t r0, int64_t n, int row) {
        dit_rows::ln_mod_bwd(run.x + r0 * C, mean1 + r0, inv1 + r0, mod(0, row), dln + r0 * C, n, C, dtmp + r0 * C,
                             nullptr, nullptr, s);
    });
    train::copy_rows(dh1, R * C, run.dx, s);
    train::add_into(run.dx, dtmp, R * C, s);
    mark(6);
}

}  // namespace strata::image
