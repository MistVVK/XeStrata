// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/qsa_dense_attn_parity.cpp - the decode attention over every cell (`qsa_dense_attn_step`) for the shapes it
// is built for, against an FP64 host reference (GPU, synthetic, no model):
//
//   Qwen3.8's 24/2 heads of 256, Ornith-1.5's 16/2 of 256, Qwen3.5-9B's 16/4 of 256 and llm-jp-4.1's 40/4 of 128, on
//   FP16 pools behind a shuffled page table, with a context that ends inside a chunk;
//   1. the output agrees with FP64 to 1e-4 of the output's scale (the short context's smaller chunks, the default);
//   2. with whole chunks (STRATA_QSA_MIN_CHUNK=64) it is bit for bit `qsa_decode_attn_step` given the selection 0, 1,
//      ..., n-1 (the same kernels, cell c in slot c);
// each with the fp16 score loop's every form (STRATA_QSA_NC 2, 4 and 8: the cells a step, qsa_dense_attn_tune's choice).
//
//   qsa_dense_attn_parity --bench CELLS   Qwen3.5-9B's decode attention over CELLS cells, each form's time a call (no
//                                         graph: a profiler such as rocprofv3 can wrap it)
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "parity_device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
namespace par = strata::parity;

// the fp16 score loop's form for the calls that follow (qsa_decode_attn.cpp's STRATA_QSA_NC); "" back to its choice
void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    if (*value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}
void set_nc(const char* nc) { set_env("XESTRATA_QSA_NC", nc); }

template <typename T> T* up(par::Device& dev, const std::vector<T>& h) {
    T* d = dev.alloc<T>(h.size() + 64 / sizeof(T));
    par::copy_in(d, h.data(), h.size());
    return d;
}

int check(const char* name, int64_t nh, int64_t nkv, int64_t hd) {
    par::Device dev;
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = nh; s.n_head_kv = nkv; s.head_dim = hd;
    const int64_t PS = s.page_size, max_cells = 512, n = 300;   // 300: four whole chunks of 64 and a partial one
    const int64_t pages = max_cells / PS, rows = pages * nkv * PS, G = nh / nkv;
    std::mt19937 rng(77 + (unsigned) (nh + hd));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> kh((size_t) (rows * hd)), vh((size_t) (rows * hd));
    for (auto& x : kh) x = k::f16_from_f32(nd(rng) * 1.5f);
    for (auto& x : vh) x = k::f16_from_f32(nd(rng));
    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<float> q((size_t) (nh * hd));
    for (auto& x : q) x = nd(rng) * 2.0f;
    std::vector<int32_t> step((size_t) k::kStepCount, 0), ids((size_t) max_cells, 0);
    step[k::kStepNKv] = (int32_t) n;
    step[k::kStepWidth] = (int32_t) n;
    for (int64_t c = 0; c < n; ++c) ids[(size_t) c] = (int32_t) c;

    // FP64 reference: query head h reads KV head h / G (QsaShapes' map)
    std::vector<double> ref((size_t) (nh * hd), 0.0);
    double scale_out = 0.0;
    for (int64_t h = 0; h < nh; ++h) {
        const int64_t kvh = h / G;
        std::vector<double> sc((size_t) n);
        double m = -1e300;
        for (int64_t c = 0; c < n; ++c) {
            const int64_t row = ((int64_t) table[(size_t) (c / PS)] * nkv + kvh) * PS + c % PS;
            double d = 0.0;
            for (int64_t i = 0; i < hd; ++i) d += (double) q[(size_t) (h * hd + i)] * k::f32_from_f16(kh[(size_t) (row * hd + i)]);
            sc[(size_t) c] = d / std::sqrt((double) hd);
            m = std::max(m, sc[(size_t) c]);
        }
        double l = 0.0;
        for (auto& v : sc) { v = std::exp(v - m); l += v; }
        for (int64_t c = 0; c < n; ++c) {
            const int64_t row = ((int64_t) table[(size_t) (c / PS)] * nkv + kvh) * PS + c % PS;
            for (int64_t i = 0; i < hd; ++i)
                ref[(size_t) (h * hd + i)] += sc[(size_t) c] / l * k::f32_from_f16(vh[(size_t) (row * hd + i)]);
        }
    }
    for (double v : ref) scale_out = std::max(scale_out, std::fabs(v));

    k::QsaAttnPools pl;
    pl.k_pool = up(dev, kh);
    pl.v_pool = up(dev, vh);
    pl.page_table = up(dev, table);
    const float* d_q = up(dev, q);
    const int32_t* d_step = up(dev, step);
    const int32_t* d_ids = up(dev, ids);
    float* scratch = dev.alloc<float>((size_t) k::qsa_decode_attn_scratch_floats(max_cells, s));
    float* d_dense = dev.alloc<float>((size_t) (nh * hd));
    float* d_sel = dev.alloc<float>((size_t) (nh * hd));
    int bad = 0;
    for (const char* nc : {"2", "4", "8"}) {
        set_nc(nc);
        std::vector<float> dense((size_t) (nh * hd)), whole((size_t) (nh * hd)), sel((size_t) (nh * hd));
        k::qsa_dense_attn_step(d_q, pl, d_step, max_cells, s, scratch, d_dense, nullptr);
        par::copy_out(dense.data(), d_dense, dense.size());
        set_env("XESTRATA_QSA_MIN_CHUNK", "64");
        k::qsa_dense_attn_step(d_q, pl, d_step, max_cells, s, scratch, d_dense, nullptr);
        par::copy_out(whole.data(), d_dense, whole.size());
        set_env("XESTRATA_QSA_MIN_CHUNK", "");
        k::qsa_decode_attn_step(d_q, pl, d_ids, d_step, max_cells, s, scratch, d_sel, nullptr);
        par::copy_out(sel.data(), d_sel, sel.size());

        double err = 0.0;
        for (size_t i = 0; i < dense.size(); ++i) err = std::max(err, std::fabs((double) dense[i] - ref[i]));
        const bool same = std::memcmp(whole.data(), sel.data(), whole.size() * sizeof(float)) == 0;
        const double rel = err / scale_out;
        std::printf("  %-34s %s cells a step: vs FP64 %.3e of the scale %s; the identity selection's bits %s\n", name,
                    nc, rel, rel <= 1e-4 ? "ok" : "FAIL", same ? "ok" : "FAIL");
        bad += (rel <= 1e-4 ? 0 : 1) + (same ? 0 : 1);
    }
    set_nc("");
    return bad;
}

// --bench: Qwen3.5-9B's 16/4 heads of 256 over `cells` cells in order, each form 50 calls after 5, waited one by one
int bench(int64_t cells) {
    par::Device dev;
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = 16; s.n_head_kv = 4; s.head_dim = 256;
    const int64_t PS = s.page_size, pages = (cells + PS - 1) / PS, rows = pages * s.n_head_kv * PS;
    std::vector<uint16_t> kv((size_t) (rows * s.head_dim), k::f16_from_f32(0.0625f));
    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;
    std::vector<float> q((size_t) (s.n_head * s.head_dim), 0.01f);
    std::vector<int32_t> step((size_t) k::kStepCount, 0);
    k::qsa_step_fill(step.data(), cells - 1, s);
    k::QsaAttnPools pl;
    pl.k_pool = up(dev, kv);
    pl.v_pool = up(dev, kv);
    pl.page_table = up(dev, table);
    const float* d_q = up(dev, q);
    const int32_t* d_step = up(dev, step);
    float* scratch = dev.alloc<float>((size_t) k::qsa_decode_attn_scratch_floats(pages * PS, s));
    float* out = dev.alloc<float>((size_t) (s.n_head * s.head_dim));
    for (const char* nc : {"2", "4", "8"}) {
        set_nc(nc);
        for (int i = 0; i < 5; ++i) k::qsa_dense_attn_step(d_q, pl, d_step, pages * PS, s, scratch, out, nullptr);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 50; ++i) k::qsa_dense_attn_step(d_q, pl, d_step, pages * PS, s, scratch, out, nullptr);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 50;
        std::printf("qsa_dense_attn --bench: Qwen3.5-9B, %lld cells, %s cells a step: %.1f us a call\n",
                    (long long) cells, nc, us);
    }
    set_nc("");
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--bench") == 0) {
        const long long cells = std::strtoll(argv[2], nullptr, 10);
        if (cells < 64) {
            std::fprintf(stderr, "qsa_dense_attn --bench: 64 cells or more\n");
            return 2;
        }
        return bench(cells);
    }
    int bad = 0;
    bad += check("Qwen3.8 QSA (24/2 x 256)", 24, 2, 256);
    bad += check("Ornith-1.5 (16/2 x 256)", 16, 2, 256);
    bad += check("Qwen3.5-9B (16/4 x 256)", 16, 4, 256);
    bad += check("llm-jp-4.1 (40/4 x 128)", 40, 4, 128);
    std::printf("qsa_dense_attn: %d failures\n", bad);
    return bad ? 1 : 0;
}
