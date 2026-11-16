// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/glm5_parity.cpp - GLM-5.3-Flash's kernels (glm5.hpp) against host references written from llama.cpp
// f9a95ae's graph and ggml CPU operators (MIT License, Copyright (c) 2023-2026 The ggml authors, see
// third_party/main/ggml/LICENSE): the mHC read, write and mean, the sigmoid router, the KDA gate and step, the clamped
// SwiGLU.  The references are in double where ggml sums in float, so the tolerances are those of float rounding.
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/window_limits.hpp"

#include "parity_device.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int HC = 4, MIX = 24;
std::mt19937& rng() {
    static std::mt19937 r(5353);   // NOLINT(bugprone-random-generator-seed): fixed, so a failure repeats
    return r;
}

std::vector<float> randn(size_t n, float s = 1.0f) {
    std::normal_distribution<float> d(0.0f, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng());
    return v;
}

double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }

int compare(const char* what, const std::vector<float>& got, const std::vector<double>& want, double tol) {
    double worst = 0;
    long long bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double err = std::fabs((double) got[i] - want[i]) / std::max(1.0, std::fabs(want[i]));
        worst = std::max(worst, err);
        if (!(err <= tol)) ++bad;
    }
    std::printf("  %-34s worst %.2e  %s\n", what, worst, bad ? "*** FAIL ***" : "ok");
    return bad ? 1 : 0;
}

// ---- mHC
void ref_hc_pre(const std::vector<float>& x, const std::vector<float>& mixes, const float* scale, const float* base,
                double rms_eps, double hc_eps, int iters, int64_t N, int64_t T, std::vector<double>& y,
                std::vector<double>& post, std::vector<double>& comb) {
    y.assign((size_t) T * N, 0);
    post.assign((size_t) T * HC, 0);
    comb.assign((size_t) T * HC * HC, 0);
    for (int64_t t = 0; t < T; ++t) {
        double ss = 0;
        for (int64_t i = 0; i < HC * N; ++i) ss += (double) x[(size_t) t * HC * N + i] * x[(size_t) t * HC * N + i];
        const double inv = 1.0 / std::sqrt(ss / (double) (HC * N) + rms_eps);
        double m[MIX];
        for (int i = 0; i < MIX; ++i) m[i] = mixes[(size_t) t * MIX + i] * inv;
        double pre[HC];
        for (int s = 0; s < HC; ++s) {
            pre[s] = sigm(m[s] * scale[0] + base[s]) + hc_eps;
            post[(size_t) t * HC + s] = 2.0 * sigm(m[HC + s] * scale[1] + base[HC + s]);
        }
        double c[HC * HC];
        for (int src = 0; src < HC; ++src) {
            double mx = -HUGE_VAL, sum = 0;
            for (int dst = 0; dst < HC; ++dst) mx = std::max(mx, m[2 * HC + dst + HC * src] * scale[2] + base[2 * HC + dst + HC * src]);
            for (int dst = 0; dst < HC; ++dst) {
                c[dst + HC * src] = std::exp(m[2 * HC + dst + HC * src] * scale[2] + base[2 * HC + dst + HC * src] - mx);
                sum += c[dst + HC * src];
            }
            for (int dst = 0; dst < HC; ++dst) c[dst + HC * src] = c[dst + HC * src] / sum + hc_eps;
        }
        auto cols = [&] {
            for (int dst = 0; dst < HC; ++dst) {
                double sum = hc_eps;
                for (int src = 0; src < HC; ++src) sum += c[dst + HC * src];
                for (int src = 0; src < HC; ++src) c[dst + HC * src] /= sum;
            }
        };
        auto rows = [&] {
            for (int src = 0; src < HC; ++src) {
                double sum = hc_eps;
                for (int dst = 0; dst < HC; ++dst) sum += c[dst + HC * src];
                for (int dst = 0; dst < HC; ++dst) c[dst + HC * src] /= sum;
            }
        };
        cols();
        for (int i = 1; i < iters; ++i) { rows(); cols(); }
        for (int i = 0; i < HC * HC; ++i) comb[(size_t) t * HC * HC + i] = c[i];
        for (int64_t n = 0; n < N; ++n) {
            double v = 0;
            for (int s = 0; s < HC; ++s) v += x[((size_t) t * HC + s) * N + n] * pre[s];
            y[(size_t) t * N + n] = v;
        }
    }
}

int test_hc() {
    const int64_t N = 4096, T = 3;
    const float scale[3] = {0.7f, 1.3f, 2.1f};
    std::vector<float> base = randn(MIX, 0.5f), x = randn((size_t) T * HC * N), mixes = randn((size_t) T * MIX, 30.0f);
    const float rms_eps = 1e-5f, hc_eps = 1e-6f;
    std::vector<double> ry, rpost, rcomb;
    ref_hc_pre(x, mixes, scale, base.data(), rms_eps, hc_eps, 20, N, T, ry, rpost, rcomb);

    strata::parity::Device dev;
    std::vector<float> sc(scale, scale + 3);
    float *dx = dev.upload(x), *dm = dev.upload(mixes), *ds = dev.upload(sc), *db = dev.upload(base);
    float *dy = dev.alloc<float>((size_t) T * N), *dp = dev.alloc<float>((size_t) T * HC),
          *dc = dev.alloc<float>((size_t) T * HC * HC);
    strata::kernels::glm5_hc_pre(dx, dm, ds, db, rms_eps, hc_eps, 20, dy, dp, dc, N, T, nullptr);
    std::vector<float> y((size_t) T * N), post((size_t) T * HC), comb((size_t) T * HC * HC);
    strata::parity::copy_out(y.data(), dy, y.size());
    strata::parity::copy_out(post.data(), dp, post.size());
    strata::parity::copy_out(comb.data(), dc, comb.size());
    int bad = compare("hc_pre y", y, ry, 1e-4) + compare("hc_pre post", post, rpost, 1e-5) +
              compare("hc_pre comb (Sinkhorn 20)", comb, rcomb, 1e-4);
    // Sinkhorn: the columns sum to ~1 after the last round
    double worst_col = 0;
    for (int64_t t = 0; t < T; ++t)
        for (int dst = 0; dst < HC; ++dst) {
            double s = 0;
            for (int src = 0; src < HC; ++src) s += comb[((size_t) t * HC + src) * HC + dst];
            worst_col = std::max(worst_col, std::fabs(s - 1.0));
        }
    std::printf("  %-34s %.2e  %s\n", "hc_pre comb column sums - 1", worst_col, worst_col < 1e-4 ? "ok" : "*** FAIL ***");
    bad += worst_col < 1e-4 ? 0 : 1;

    // the write, out of place and in place on the residual
    std::vector<float> f = randn((size_t) T * N), res = randn((size_t) T * HC * N);
    std::vector<double> rout((size_t) T * HC * N);
    for (int64_t t = 0; t < T; ++t)
        for (int d = 0; d < HC; ++d)
            for (int64_t n = 0; n < N; ++n) {
                double v = (double) f[(size_t) t * N + n] * post[(size_t) t * HC + d];
                for (int s = 0; s < HC; ++s)
                    v += (double) res[((size_t) t * HC + s) * N + n] * comb[((size_t) t * HC + s) * HC + d];
                rout[((size_t) t * HC + d) * N + n] = v;
            }
    float *df = dev.upload(f), *dr = dev.upload(res), *dout = dev.alloc<float>(res.size());
    strata::kernels::glm5_hc_post(df, dr, dp, dc, dout, N, T, nullptr);
    std::vector<float> out(res.size());
    strata::parity::copy_out(out.data(), dout, out.size());
    bad += compare("hc_post", out, rout, 1e-5);
    strata::kernels::glm5_hc_post(df, dr, dp, dc, dr, N, T, nullptr);
    strata::parity::copy_out(out.data(), dr, out.size());
    bad += compare("hc_post in place", out, rout, 1e-5);

    // the streams' start and the axis swap
    {
        std::vector<float> e = randn((size_t) T * N);
        std::vector<double> rinit((size_t) T * HC * N);
        for (int64_t t = 0; t < T; ++t)
            for (int s2 = 0; s2 < HC; ++s2)
                for (int64_t n = 0; n < N; ++n) rinit[((size_t) t * HC + s2) * N + n] = e[(size_t) t * N + n];
        float *de = dev.upload(e), *dhs = dev.alloc<float>(rinit.size());
        strata::kernels::glm5_hc_init(de, dhs, N, T, nullptr);
        std::vector<float> hs(rinit.size());
        strata::parity::copy_out(hs.data(), dhs, hs.size());
        bad += compare("hc_init", hs, rinit, 0.0);
        const int64_t A = 3, B = 5, D = 7;
        std::vector<float> src = randn((size_t) A * B * D);
        std::vector<double> rt((size_t) A * B * D);
        for (int64_t i = 0; i < A; ++i)
            for (int64_t j = 0; j < B; ++j)
                for (int64_t k = 0; k < D; ++k) rt[((size_t) j * A + i) * D + k] = src[((size_t) i * B + j) * D + k];
        float *dsrc = dev.upload(src), *ddst = dev.alloc<float>(src.size());
        strata::kernels::glm5_transpose_abd(dsrc, ddst, A, B, D, nullptr);
        std::vector<float> tr(src.size());
        strata::parity::copy_out(tr.data(), ddst, tr.size());
        bad += compare("transpose_abd", tr, rt, 0.0);
    }

    std::vector<double> rmean((size_t) T * N);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < N; ++n) {
            double v = 0;
            for (int s = 0; s < HC; ++s) v += x[((size_t) t * HC + s) * N + n];
            rmean[(size_t) t * N + n] = v / HC;
        }
    float* dmean = dev.alloc<float>((size_t) T * N);
    strata::kernels::glm5_hc_mean(dx, dmean, N, T, nullptr);
    std::vector<float> mean((size_t) T * N);
    strata::parity::copy_out(mean.data(), dmean, mean.size());
    bad += compare("hc_mean", mean, rmean, 1e-6);
    return bad;
}

// ---- the sigmoid router
int test_router() {
    const int E = 288, K = 8, T = 40;
    std::vector<float> logits = randn((size_t) T * E, 2.0f), bias = randn(E, 0.05f);
    for (int e = 0; e < 12; ++e) logits[(size_t) 3 * E + e] = 1.25f;   // token 3: a 12-way tie (with bias 0 below)
    for (int e = 0; e < 12; ++e) bias[(size_t) e] = 0.0f;
    std::vector<int> rid((size_t) T * K);
    std::vector<double> rw((size_t) T * K);
    for (int t = 0; t < T; ++t) {
        std::vector<int> idx(E);
        std::iota(idx.begin(), idx.end(), 0);
        // the kernel's order in float: the larger biased score, then the lower index
        std::vector<float> sel(E);
        for (int e = 0; e < E; ++e)
            sel[(size_t) e] = 1.0f / (1.0f + std::exp(-logits[(size_t) t * E + e])) + bias[(size_t) e];
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return sel[(size_t) a] > sel[(size_t) b]; });
        double sum = 0;
        for (int i = 0; i < K; ++i) {
            rid[(size_t) t * K + i] = idx[(size_t) i];
            rw[(size_t) t * K + i] = sigm(logits[(size_t) t * E + idx[(size_t) i]]);
            sum += rw[(size_t) t * K + i];
        }
        sum = std::max(sum, 6.103515625e-05);
        for (int i = 0; i < K; ++i) rw[(size_t) t * K + i] = rw[(size_t) t * K + i] / sum * 2.5;
    }
    strata::parity::Device dev;
    float *dl = dev.upload(logits), *db = dev.upload(bias), *dw = dev.alloc<float>((size_t) T * K);
    int32_t* di = dev.alloc<int32_t>((size_t) T * K);
    strata::kernels::glm5_router_sigmoid(dl, db, T, E, K, true, 2.5f, di, dw, nullptr);
    std::vector<int32_t> id((size_t) T * K);
    std::vector<float> w((size_t) T * K);
    strata::parity::copy_out(id.data(), di, id.size());
    strata::parity::copy_out(w.data(), dw, w.size());
    long long id_bad = 0;
    for (size_t i = 0; i < id.size(); ++i) id_bad += id[i] != rid[i];
    std::printf("  %-34s %lld wrong  %s\n", "router ids (288 -> 8, a 12-way tie)", id_bad, id_bad ? "*** FAIL ***" : "ok");
    return (id_bad ? 1 : 0) + compare("router weights (norm, x2.5)", w, rw, 1e-5);
}

// ---- KDA
int test_kda() {
    const int H = 4, S = 128, T = 3;
    const float lb = -5.0f;
    std::vector<float> f = randn((size_t) T * H * S), dt = randn((size_t) H * S, 0.5f), a(H);
    for (int h = 0; h < H; ++h) a[(size_t) h] = -std::exp(0.3f * (float) h - 0.5f);
    std::vector<double> rg((size_t) T * H * S);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h)
            for (int s = 0; s < S; ++s) {
                const size_t i = ((size_t) t * H + h) * S + s;
                rg[i] = lb * sigm(-((double) f[i] + dt[(size_t) h * S + s]) * a[(size_t) h]);
            }
    strata::parity::Device dev;
    float *dfa = dev.upload(f), *ddt = dev.upload(dt), *da = dev.upload(a), *dg = dev.alloc<float>(f.size());
    strata::kernels::glm5_kda_gate(dfa, ddt, da, lb, dg, H, S, T, nullptr);
    std::vector<float> g(f.size());
    strata::parity::copy_out(g.data(), dg, g.size());
    int bad = compare("kda_gate", g, rg, 1e-5);

    // the short convolution over 3 tokens, the state carried
    {
        const int64_t C = 300;
        std::vector<float> x = randn((size_t) T * C), st = randn((size_t) C * 3), w = randn((size_t) C * 4, 0.5f);
        std::vector<double> ry((size_t) T * C), rst((size_t) C * 3);
        for (int64_t c = 0; c < C; ++c) {
            std::vector<double> seq = {st[(size_t) c * 3], st[(size_t) c * 3 + 1], st[(size_t) c * 3 + 2]};
            for (int t = 0; t < T; ++t) seq.push_back(x[(size_t) t * C + c]);
            for (int t = 0; t < T; ++t) {
                double s = 0;
                for (int j = 0; j < 4; ++j) s += seq[(size_t) t + j] * w[(size_t) c * 4 + j];
                ry[(size_t) t * C + c] = s / (1.0 + std::exp(-s));
            }
            for (int j = 0; j < 3; ++j) rst[(size_t) c * 3 + j] = seq[(size_t) T + j];
        }
        float *dx = dev.upload(x), *dst = dev.upload(st), *dw = dev.upload(w), *dy = dev.alloc<float>(x.size());
        strata::kernels::glm5_kda_conv(dx, dst, dw, dy, C, T, true, nullptr);
        std::vector<float> y(x.size()), st2(st.size());
        strata::parity::copy_out(y.data(), dy, y.size());
        strata::parity::copy_out(st2.data(), dst, st2.size());
        bad += compare("kda_conv (3 tokens, SiLU)", y, ry, 1e-5) + compare("kda_conv state", st2, rst, 0.0);

        // the head L2 norm and the gated output norm
        std::vector<float> o = randn((size_t) T * H * S), g2 = randn((size_t) T * H * S), wn = randn(S, 0.2f);
        std::vector<double> rl2((size_t) T * H * S), rout((size_t) T * H * S);
        for (int th = 0; th < T * H; ++th) {
            double ss = 0;
            for (int i = 0; i < S; ++i) ss += (double) o[(size_t) th * S + i] * o[(size_t) th * S + i];
            for (int i = 0; i < S; ++i) {
                rl2[(size_t) th * S + i] = o[(size_t) th * S + i] / std::max(std::sqrt(ss), 1e-6);
                rout[(size_t) th * S + i] = o[(size_t) th * S + i] / std::sqrt(ss / S + 1e-5) * wn[(size_t) i] *
                                              sigm(g2[(size_t) th * S + i]);
            }
        }
        float *dov = dev.upload(o), *dg2 = dev.upload(g2), *dwn = dev.upload(wn), *dyo = dev.alloc<float>(o.size());
        strata::kernels::glm5_kda_out(dov, dg2, dwn, 1e-5f, dyo, H, T, nullptr);
        std::vector<float> yo(o.size());
        strata::parity::copy_out(yo.data(), dyo, yo.size());
        bad += compare("kda_out (RMSNorm x sigmoid)", yo, rout, 1e-5);
        strata::kernels::glm5_l2_heads(dov, H, 1e-6f, T, nullptr);
        strata::parity::copy_out(yo.data(), dov, yo.size());
        bad += compare("l2_heads", yo, rl2, 1e-5);
    }

    // T steps from a random state; q, k L2-normalised per head
    std::vector<float> state = randn((size_t) S * H * S, 0.1f);
    std::vector<double> rs(state.begin(), state.end());
    float* dstate = dev.upload(state);
    std::vector<double> rout_all;
    std::vector<float> out_all;
    for (int t = 0; t < T; ++t) {
        std::vector<float> q = randn((size_t) H * S), k = randn((size_t) H * S), v = randn((size_t) H * S), beta(H);
        for (auto* p : {&q, &k})
            for (int h = 0; h < H; ++h) {
                double n = 0;
                for (int i = 0; i < S; ++i) n += (double) (*p)[(size_t) h * S + i] * (*p)[(size_t) h * S + i];
                for (int i = 0; i < S; ++i) (*p)[(size_t) h * S + i] = (float) ((*p)[(size_t) h * S + i] / std::sqrt(n + 1e-6));
            }
        for (int h = 0; h < H; ++h) beta[(size_t) h] = (float) sigm(0.3 * h - 0.2);
        const float* gt = g.data() + (size_t) t * H * S;
        for (int h = 0; h < H; ++h) {
            auto at = [&](int i, int j) -> double& { return rs[((size_t) i * H + h) * S + j]; };
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < S; ++j) at(i, j) *= std::exp((double) gt[h * S + i]);
            for (int j = 0; j < S; ++j) {
                double kv = 0;
                for (int i = 0; i < S; ++i) kv += at(i, j) * k[(size_t) h * S + i];
                const double delta = ((double) v[(size_t) h * S + j] - kv) * beta[(size_t) h];
                for (int i = 0; i < S; ++i) at(i, j) += (double) k[(size_t) h * S + i] * delta;
            }
            for (int j = 0; j < S; ++j) {
                double o = 0;
                for (int i = 0; i < S; ++i) o += at(i, j) * q[(size_t) h * S + i];
                rout_all.push_back(o / std::sqrt((double) S));
            }
        }
        float *dq = dev.upload(q), *dk = dev.upload(k), *dv = dev.upload(v), *dbeta = dev.upload(beta),
              *dout = dev.alloc<float>((size_t) H * S);
        strata::kernels::glm5_kda_step(dstate, dq, dk, dv, dg + (size_t) t * H * S, dbeta, dout, H, nullptr);
        strata::parity::sync();
        std::vector<float> out((size_t) H * S);
        strata::parity::copy_out(out.data(), dout, out.size());
        out_all.insert(out_all.end(), out.begin(), out.end());
    }
    strata::parity::copy_out(state.data(), dstate, state.size());
    bad += compare("kda_step out (3 steps)", out_all, rout_all, 1e-4);
    bad += compare("kda_step state", state, rs, 1e-4);

    // glm5_kda_scan over a wide window (16 tokens, 15 state snapshots: the snapshots' wide argument) against
    // glm5_kda_step token by token: every output, snapshot and the final state bit for bit
    {
        const int TW = strata::kernels::kWindowMaxT;
        const size_t TS = (size_t) H * S;
        std::vector<float> s0 = randn((size_t) S * H * S, 0.1f), qw = randn(TW * TS), kw = randn(TW * TS),
                           vw = randn(TW * TS), gw = randn(TW * TS, 0.1f), bw((size_t) TW * H);
        for (auto& x : gw) x = -std::fabs(x);
        for (int i = 0; i < TW * H; ++i) bw[(size_t) i] = (float) sigm(0.1 * i - 0.5);
        for (auto* p : {&qw, &kw})
            for (size_t r = 0; r < (size_t) TW * H; ++r) {
                double n = 0;
                for (int i = 0; i < S; ++i) n += (double) (*p)[r * S + i] * (*p)[r * S + i];
                for (int i = 0; i < S; ++i) (*p)[r * S + i] = (float) ((*p)[r * S + i] / std::sqrt(n + 1e-6));
            }
        float *dsa = dev.upload(s0), *dsb = dev.upload(s0), *dqw = dev.upload(qw), *dkw = dev.upload(kw),
              *dvw = dev.upload(vw), *dgw = dev.upload(gw), *dbw = dev.upload(bw);
        float *doa = dev.alloc<float>(TW * TS), *dob = dev.alloc<float>(TW * TS);
        std::vector<float*> snaps((size_t) TW - 1);
        for (auto& p : snaps) p = dev.alloc<float>(s0.size());
        strata::kernels::glm5_kda_scan(dsa, dqw, dkw, dvw, dgw, dbw, doa, H, TW, nullptr, snaps.data(), TW - 1);
        strata::parity::sync();
        int wbad = 0;
        std::vector<float> a(s0.size()), b(s0.size());
        for (int t = 0; t < TW; ++t) {
            strata::kernels::glm5_kda_step(dsb, dqw + t * TS, dkw + t * TS, dvw + t * TS, dgw + t * TS, dbw + (ptrdiff_t) t * H,
                                           dob + t * TS, H, nullptr);
            strata::parity::sync();
            if (t == TW - 1) break;
            strata::parity::copy_out(a.data(), snaps[(size_t) t], a.size());
            strata::parity::copy_out(b.data(), dsb, b.size());
            wbad += a != b;
        }
        std::vector<float> oa(TW * TS), ob(TW * TS);
        strata::parity::copy_out(oa.data(), doa, oa.size());
        strata::parity::copy_out(ob.data(), dob, ob.size());
        strata::parity::copy_out(a.data(), dsa, a.size());
        strata::parity::copy_out(b.data(), dsb, b.size());
        wbad += (oa != ob) + (a != b);
        std::printf("  %-40s %s\n", "kda_scan 16 tokens, 15 snapshots",
                    wbad ? "*** DIFFER from kda_step ***" : "bitwise");
        bad += wbad;
    }
    return bad;
}

// ---- the DSA layer: the k-pool indexer (append, pools, scores), the selection (qsa_block_topk), the MLA attention
int test_dsa() {
    using namespace strata::kernels;
    const int D = 128, KP = 4, IH = 32, P = 53, TOPK = 24;   // 53 cells: 13 pools and one unfinished
    const int L = 512, H = 4;
    const float ln_eps = 1e-6f;
    std::vector<float> kraw = randn((size_t) P * D), graw = randn((size_t) P * D), lw = randn(D, 0.3f), lb = randn(D, 0.1f),
                       ape = randn((size_t) KP * D, 0.5f);
    for (auto& x : lw) x += 1.0f;
    std::vector<int32_t> steps_in((size_t) P * kStepCount, 0);
    for (int t = 0; t < P; ++t) steps_in[(size_t) t * kStepCount + kStepPos] = t;

    strata::parity::Device dev;
    int32_t *dsi = dev.upload(steps_in), *dso = dev.alloc<int32_t>(steps_in.size());
    glm5_idx_steps(dsi, P, KP, TOPK, dso, nullptr);
    float *dk = dev.upload(kraw), *dg = dev.upload(graw), *dlw = dev.upload(lw), *dlb = dev.upload(lb),
          *dape = dev.upload(ape), *dcache = dev.alloc<float>((size_t) P * 2 * D), *dpool = dev.alloc<float>((size_t) (P / KP + 1) * D);
    glm5_idx_append(dk, dg, dlw, dlb, ln_eps, dso, P, dcache, nullptr);
    glm5_idx_pool(dcache, dape, KP, dso, P, dpool, nullptr);

    // the host: keys, pools
    std::vector<double> key((size_t) P * D);
    for (int t = 0; t < P; ++t) {
        double mean = 0, var = 0;
        for (int c = 0; c < D; ++c) mean += kraw[(size_t) t * D + c];
        mean /= D;
        for (int c = 0; c < D; ++c) var += (kraw[(size_t) t * D + c] - mean) * (kraw[(size_t) t * D + c] - mean);
        var /= D;
        for (int c = 0; c < D; ++c)
            key[(size_t) t * D + c] = (kraw[(size_t) t * D + c] - mean) / std::sqrt(var + ln_eps) * lw[(size_t) c] + lb[(size_t) c];
    }
    const int NP = P / KP;
    std::vector<double> pooled((size_t) NP * D);
    for (int m = 0; m < NP; ++m)
        for (int c = 0; c < D; ++c) {
            double l[KP], mx = -HUGE_VAL, sum = 0, acc = 0;
            for (int j = 0; j < KP; ++j) {
                l[j] = graw[((size_t) m * KP + j) * D + c] + ape[(size_t) j * D + c];
                mx = std::max(mx, l[j]);
            }
            for (int j = 0; j < KP; ++j) sum += (l[j] = std::exp(l[j] - mx));
            for (int j = 0; j < KP; ++j) acc += l[j] / sum * key[((size_t) m * KP + j) * D + c];
            pooled[(size_t) m * D + c] = acc;
        }
    std::vector<float> cache((size_t) P * 2 * D), pool((size_t) NP * D);
    strata::parity::copy_out(cache.data(), dcache, cache.size());
    strata::parity::copy_out(pool.data(), dpool, pool.size());
    std::vector<float> keys_got((size_t) P * D);
    for (int t = 0; t < P; ++t)
        for (int c = 0; c < D; ++c) keys_got[(size_t) t * D + c] = cache[(size_t) t * 2 * D + c];
    int bad = compare("idx_append keys (LayerNorm)", keys_got, key, 1e-5) + compare("idx_pool (softmax pooling)", pool, pooled, 1e-5);

    // scores and the selection for the last T tokens
    const int T = 6, MB = P / KP + 1;
    std::vector<float> q = randn((size_t) T * IH * D), w = randn((size_t) T * IH, 0.05f);
    float *dq = dev.upload(q), *dw = dev.upload(w), *dsc = dev.alloc<float>((size_t) T * MB);
    const int32_t* dso_tail = dso + (size_t) (P - T) * kStepCount;
    glm5_idx_scores(dpool, dq, dw, IH, dso_tail, T, MB, dsc, nullptr);
    std::vector<float> sc((size_t) T * MB);
    strata::parity::copy_out(sc.data(), dsc, sc.size());
    std::vector<float> scg;
    std::vector<double> scw;
    for (int t = 0; t < T; ++t) {
        const int pos = P - T + t, nb = (pos + 1) / KP;
        for (int b = 0; b < nb; ++b) {
            double s = 0;
            for (int h = 0; h < IH; ++h) {
                double d = 0;
                for (int c = 0; c < D; ++c) d += (double) q[((size_t) t * IH + h) * D + c] * pool[(size_t) b * D + c];
                s += std::max(d, 0.0) * w[(size_t) t * IH + h];
            }
            scg.push_back(sc[(size_t) t * MB + b]);
            scw.push_back(s);
        }
    }
    bad += compare("idx_scores (32 heads, weighted)", scg, scw, 1e-4);

    QsaShapes qs;
    qs.idx_block = KP;
    qs.idx_top_k = 2048;
    const int64_t cap = qsa_selection_width(kTopkMaxCells, qs);
    int32_t* dids = dev.alloc<int32_t>((size_t) T * cap);
    qsa_block_topk(dsc, dso_tail, T, MB, cap, qs, dids, nullptr);
    std::vector<int32_t> ids((size_t) T * cap);
    strata::parity::copy_out(ids.data(), dids, ids.size());
    long long sel_bad = 0;
    std::vector<std::vector<int>> rsel(T);
    for (int t = 0; t < T; ++t) {
        const int pos = P - T + t, nb = (pos + 1) / KP, n_top = std::min(nb, TOPK / KP);
        std::vector<int> order(nb);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a2, int b2) { return sc[(size_t) t * MB + a2] > sc[(size_t) t * MB + b2]; });
        for (int i = 0; i < n_top; ++i)
            for (int j = 0; j < KP; ++j) rsel[(size_t) t].push_back(order[(size_t) i] * KP + j);
        std::sort(rsel[(size_t) t].begin(), rsel[(size_t) t].end());
        for (size_t i = 0; i < rsel[(size_t) t].size(); ++i) sel_bad += ids[(size_t) t * cap + i] != rsel[(size_t) t][i];
    }
    std::printf("  %-34s %lld wrong  %s\n", "selection (top 6 of 13 pools)", sel_bad, sel_bad ? "*** FAIL ***" : "ok");
    bad += sel_bad ? 1 : 0;

    // the MLA attention over the selection
    std::vector<float> kv = randn((size_t) P * L, 0.5f), qa = randn((size_t) T * H * L, 0.1f);
    float *dkv = dev.upload(kv), *dqa = dev.upload(qa), *dout = dev.alloc<float>((size_t) T * H * L);
    uint16_t* dlat = dev.alloc<uint16_t>((size_t) P * L);
    glm5_latent_store(dkv, dso, P, L, dlat, nullptr);
    const float scale = 1.0f / std::sqrt(256.0f);
    glm5_mla_attn(dqa, dlat, dids, cap, dso_tail, H, scale, T, dout, nullptr);
    std::vector<float> out((size_t) T * H * L);
    strata::parity::copy_out(out.data(), dout, out.size());
    std::vector<double> rout((size_t) T * H * L, 0);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            const auto& cs = rsel[(size_t) t];
            if (cs.empty()) continue;
            std::vector<double> sj(cs.size());
            double mx = -HUGE_VAL, sum = 0;
            for (size_t j = 0; j < cs.size(); ++j) {
                double d = 0;
                for (int l = 0; l < L; ++l)
                    d += (double) qa[((size_t) t * H + h) * L + l] *
                         f32_from_f16(f16_from_f32(kv[(size_t) cs[j] * L + l]));
                sj[j] = d * scale;
                mx = std::max(mx, sj[j]);
            }
            for (auto& v : sj) sum += (v = std::exp(v - mx));
            for (int l = 0; l < L; ++l) {
                double acc = 0;
                for (size_t j = 0; j < cs.size(); ++j) acc += sj[j] / sum * f32_from_f16(f16_from_f32(kv[(size_t) cs[j] * L + l]));
                rout[((size_t) t * H + h) * L + l] = acc;
            }
        }
    bad += compare("mla_attn over the selection", out, rout, 1e-4);

    // the first tokens: no complete pool, nothing selected, zeros
    float* dz = dev.alloc<float>((size_t) H * L);
    glm5_mla_attn(dqa, dlat, dids, cap, dso, H, scale, 1, dz, nullptr);
    std::vector<float> z((size_t) H * L);
    strata::parity::copy_out(z.data(), dz, z.size());
    bad += compare("mla_attn with no pool yet (zeros)", z, std::vector<double>(z.size(), 0.0), 0.0);
    return bad;
}

// ---- IQ1_S (UD-IQ1_S's experts): random blocks, the GPU's dequantization and dot against ggml's dequantize_row_iq1_s
int test_iq1_s() {
    const int rows = 16, cols = 2048, nb = cols / QK_K;
    std::uniform_int_distribution<int> byte(0, 255), word(0, 65535);
    std::vector<block_iq1_s> blocks((size_t) rows * nb);
    for (auto& b : blocks) {
        b.d = strata::kernels::f16_from_f32(std::uniform_real_distribution<float>(0.001f, 0.02f)(rng()));
        for (auto& q : b.qs) q = (uint8_t) byte(rng());
        for (auto& h : b.qh) h = (uint16_t) word(rng());
    }
    std::vector<double> ref((size_t) rows * cols);
    for (size_t i = 0; i < blocks.size(); ++i) {   // ggml-quants.c's dequantize_row_iq1_s
        const block_iq1_s& b = blocks[i];
        const double d = strata::kernels::f32_from_f16(b.d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const double dl = d * (2 * ((b.qh[ib] >> 12) & 7) + 1);
            const double delta = b.qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const int8_t* grid = (const int8_t*) (iq1s_grid + (b.qs[ib * 4 + l] | (((b.qh[ib] >> 3 * l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) ref[i * QK_K + (size_t) ib * 32 + (size_t) l * 8 + j] = dl * (grid[j] + delta);
            }
        }
    }
    strata::parity::Device dev;
    uint8_t* dw = dev.alloc<uint8_t>(blocks.size() * sizeof(block_iq1_s));
    strata::parity::copy_in(dw, (const uint8_t*) blocks.data(), blocks.size() * sizeof(block_iq1_s));
    float* dq = dev.alloc<float>(ref.size());
    strata::kernels::iq_dequant_f32(19, dw, (int64_t) ref.size(), dq, strata::parity::stream());
    std::vector<float> got(ref.size());
    strata::parity::copy_out(got.data(), dq, got.size());
    int bad = compare("IQ1_S dequantization", got, ref, 1e-6);
    // the dot of two columns, as iq_parity: to the q8_1 rounding of the activations
    std::vector<float> x = randn((size_t) 2 * cols);
    float* dx = dev.upload(x);
    uint8_t* xq = dev.alloc<uint8_t>(strata::kernels::native_q8_1_bytes(cols, 2));
    float* dy = dev.alloc<float>((size_t) 2 * rows);
    strata::kernels::quantize_q8_1_rows(dx, 2, cols, xq, strata::parity::stream());
    strata::kernels::native_mmvq(19, dw, xq, dy, cols, rows, 2, strata::parity::stream());
    std::vector<float> y((size_t) 2 * rows);
    strata::parity::copy_out(y.data(), dy, y.size());
    double num = 0, den = 0;
    for (int c = 0; c < 2; ++c)
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int k = 0; k < cols; ++k) acc += ref[(size_t) r * cols + k] * x[(size_t) c * cols + k];
            num += std::fabs(y[(size_t) c * rows + r] - acc);
            den += std::fabs(acc);
        }
    const double rel = num / (den + 1e-30);
    std::printf("  %-34s rel %.2e  %s\n", "IQ1_S mmvq (q8_1 activations)", rel, rel < 2e-2 ? "ok" : "*** FAIL ***");
    return bad + (rel < 2e-2 ? 0 : 1);
}

// ---- the clamped SwiGLU
int test_swiglu_clamp() {
    const size_t n = 4096;
    std::vector<float> g = randn(n, 8.0f), u = randn(n, 8.0f);
    std::vector<double> r(n);
    for (size_t i = 0; i < n; ++i) {
        const double gv = std::min((double) g[i], 10.0), uv = std::clamp((double) u[i], -10.0, 10.0);
        r[i] = gv / (1.0 + std::exp(-gv)) * uv;
    }
    strata::parity::Device dev;
    float *dg = dev.upload(g), *du = dev.upload(u), *dout = dev.alloc<float>(n);
    strata::kernels::glm5_swiglu_clamp(dg, du, dout, (int64_t) n, 10.0f, nullptr);
    std::vector<float> out(n);
    strata::parity::copy_out(out.data(), dout, n);
    int bad = compare("swiglu_clamp (limit 10)", out, r, 1e-5);
    strata::kernels::glm5_swiglu_clamp(dg, du, dg, (int64_t) n, 10.0f, nullptr);
    strata::parity::copy_out(out.data(), dg, n);
    return bad + compare("swiglu_clamp in place", out, r, 1e-5);
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: glm5_parity [--selftest]\n");
            return 2;
        }
    std::printf("glm5_parity: GLM-5.3-Flash's kernels against host references\n");
    const int bad = test_hc() + test_router() + test_kda() + test_dsa() + test_iq1_s() + test_swiglu_clamp();
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
