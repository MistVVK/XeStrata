// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// tests/kernels/bf16_gemm_parity.cpp - the CPU's BF16 x FP32 matrix product (kernels/cpu/bf16_gemm.hpp): the AVX2
// kernel against the scalar reference bit for bit, the packing against a split of the same matrix, the answer
// against the number of threads and the token group, and their speed.  CPU only: no model, no GPU.
//
//   bf16_gemm_parity                       every check below on the EmbeddingGemma 2 shapes and on odd ones
//   bf16_gemm_parity --bench [--reps N]    GFLOP/s per shape, one thread and six, on the ISA this CPU runs
//   bf16_gemm_parity --numpy DIR           the independent check against a float64 reference in DIR (the record's
//                                          `s2/numpy/`; made by `make_numpy_ref.py`).  Without DIR or without the
//                                          files the test exits 77 (a CTest skip), because the bit-exact checks
//                                          above share their shape of the arithmetic with the reference: a mistake
//                                          in THAT would be invisible to them, and only a foreign sum catches it.
//
// WHY "BIT FOR BIT" IS THE CONTRACT, and what each of these five checks protects:
//   (1) AVX2 == scalar, every shape, every token group 1..11 - the vector kernel is the one that runs, the scalar
//       one is the definition.  They differ by a mis-ordered lane or a missing pad and nothing else.
//   (2) a band of panels == the whole matrix - the forward splits a matrix over the CPU workers by panel band, so
//       the embedding must not depend on where the bands fall.
//   (3) a token inside a group == the same token alone - the batch is built by the server, not by the model, and a
//       request's vector cannot move when another request joins it (the reference's own padded-vs-single batch
//       differs by 8.8e-08 for the same reason: only the sums' order, and here not even that).
//   (4) the packed bytes of panels [p0,p1) == the same panels of the whole pack - the packing is what the load does,
//       and it is done per tensor, in whatever order the loader walks them.
//   (5) bias == the same product plus the bias added afterwards - the checkpoint's only biased linear (the audio
//       output projection) must not shift the rest.
//   (6) threads: the same binary on one, two and six threads gives one answer (the same check as (2) at the thread
//       level, through std::thread, so it also proves no accumulator is shared between bands).
#include "strata/kernels/cpu/bf16_gemm.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::kernels::cpu;

namespace {

int g_failures = 0;

void fail(const char* what, const char* detail) {
    ++g_failures;
    std::printf("  FAIL %s: %s\n", what, detail);
}

/// The EmbeddingGemma 2 forward's own linear layers (s0 FINDINGS 4・5・6): n_in x n_out.  The odd shapes after them
/// are the padding and the remainder paths of a card we do not have.
struct Shape {
    const char* name;
    int n_in;
    int n_out;
};

const Shape kShapes[] = {
    {"text q_proj sliding", 512, 1024},
    {"text q_proj full", 512, 2048},
    {"text k_proj", 512, 512},
    {"text o_proj sliding", 1024, 512},
    {"text o_proj full", 2048, 512},
    {"text gate/up", 512, 2048},
    {"text down", 2048, 512},
    {"text ple projection", 512, 12288},
    {"text ple gate", 512, 512},
    {"text embedding projection", 512, 768},
    {"vision patch embed", 768, 768},
    {"vision mlp", 768, 3072},
    {"vision out proj", 768, 512},
    {"audio sscp conv out", 1024, 1024},
    {"audio ffn", 1024, 4096},
    {"audio output proj", 1024, 1536},
    {"audio embed", 1536, 512},
    {"odd in 8", 517, 1024},
    {"odd out 8", 512, 1001},
    {"odd both", 613, 907},
    {"tiny", 8, 8},
    {"one panel less", 512, 7},
};

/// Random BF16 bits of a plausible size for a weight (the reference's own magnitudes, not a wide random field).
void fill_weights(std::mt19937& rng, std::vector<uint16_t>& w) {
    std::normal_distribution<float> g(0.0f, 0.03f);
    for (auto& v : w) v = strata::kernels::bf16_from_f32(g(rng));
}

void fill_activations(std::mt19937& rng, std::vector<float>& x) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (auto& v : x) v = g(rng);
}

/// One shape: pack, run both ISAs, check (1)(2)(3)(5).
void check_shape(const Shape& sh, int n_tok, bool with_bias) {
    const int panels = bf16_gemm_panels(sh.n_out);
    std::mt19937 rng(0x5eedu + (unsigned) sh.n_in * 7919u + (unsigned) sh.n_out + (unsigned) n_tok * 31u +
                     (with_bias ? 1u : 0u));   // NOLINT(bugprone-random-generator-seed): a shape's rows are fixed.
    std::vector<uint16_t> w((size_t) sh.n_out * sh.n_in);
    fill_weights(rng, w);
    std::vector<uint16_t> panel(bf16_gemm_packed_bytes(sh.n_out, sh.n_in));
    bf16_gemm_pack(w.data(), (size_t) sh.n_in, sh.n_in, sh.n_out, panel.data(), 0, panels);

    // (4) the pack split by panel band is the same bytes
    std::vector<uint16_t> split(panel.size());
    const int cut = panels / 2;
    bf16_gemm_pack(w.data(), (size_t) sh.n_in, sh.n_in, sh.n_out, split.data(), 0, cut);
    bf16_gemm_pack(w.data(), (size_t) sh.n_in, sh.n_in, sh.n_out, split.data(), cut, panels);
    if (split != panel) fail(sh.name, "pack split over panels differs");

    std::vector<float> x((size_t) n_tok * sh.n_in), bias;
    fill_activations(rng, x);
    if (with_bias) {
        bias.resize((size_t) sh.n_out);
        std::normal_distribution<float> g(0.0f, 0.03f);
        for (auto& v : bias) v = g(rng);
    }

    std::vector<float> yv((size_t) n_tok * sh.n_out, 0.0f), ys((size_t) n_tok * sh.n_out, 0.0f);
    // The scalar kernel is the definition, so it runs first; the vector one either agrees bit for bit or the shape
    // is compared against itself (a CPU without AVX2: the checks below still hold, and only the vector-vs-scalar one
    // has nothing to say).
    bf16_gemm_rows(Bf16GemmIsa::scalar, panel.data(), sh.n_in, sh.n_out, x.data(), (size_t) sh.n_in, ys.data(),
                   (size_t) sh.n_out, n_tok, 0, panels, bias.empty() ? nullptr : bias.data());
    if (bf16_gemm_isa() != Bf16GemmIsa::scalar)
        bf16_gemm_rows(bf16_gemm_isa(), panel.data(), sh.n_in, sh.n_out, x.data(), (size_t) sh.n_in, yv.data(),
                       (size_t) sh.n_out, n_tok, 0, panels, bias.empty() ? nullptr : bias.data());
    else
        yv = ys;

    // (1) bit for bit
    if (yv != ys) {
        size_t first = 0;
        while (first < ys.size() && yv[first] == ys[first]) ++first;
        if (first < ys.size())
            std::printf("  FAIL %s: %s != scalar at %zu (%g vs %g), n_tok %d, bias %d\n", sh.name,
                        bf16_gemm_isa_name(bf16_gemm_isa()), first,
                        (double) yv[first], (double) ys[first], n_tok, (int) with_bias);
        else
            std::printf("  FAIL %s: vector != scalar (different sizes?), n_tok %d\n", sh.name, n_tok);
        ++g_failures;
        return;
    }

    // (2) a band of panels is the same answer
    if (panels > 2) {
        std::vector<float> band(ys.size(), 0.0f);
        const int a = 1, b = panels - 1;
        const auto isa = bf16_gemm_isa();
        bf16_gemm_rows(isa, panel.data(), sh.n_in, sh.n_out, x.data(), (size_t) sh.n_in, band.data(),
                       (size_t) sh.n_out, n_tok, 0, a, bias.empty() ? nullptr : bias.data());
        bf16_gemm_rows(isa, panel.data(), sh.n_in, sh.n_out, x.data(), (size_t) sh.n_in, band.data(),
                       (size_t) sh.n_out, n_tok, a, b, bias.empty() ? nullptr : bias.data());
        bf16_gemm_rows(isa, panel.data(), sh.n_in, sh.n_out, x.data(), (size_t) sh.n_in, band.data(),
                       (size_t) sh.n_out, n_tok, b, panels, bias.empty() ? nullptr : bias.data());
        if (band != ys) {
            std::printf("  FAIL %s: panel bands differ, n_tok %d\n", sh.name, n_tok);
            ++g_failures;
        }
    }

    // (3) every token alone is the same row as inside the group
    for (int t = 0; t < n_tok; ++t) {
        std::vector<float> one((size_t) sh.n_out, 0.0f);
        const auto isa = bf16_gemm_isa();
        bf16_gemm_rows(isa, panel.data(), sh.n_in, sh.n_out, x.data() + (size_t) t * sh.n_in, (size_t) sh.n_in,
                       one.data(), (size_t) sh.n_out, 1, 0, panels, bias.empty() ? nullptr : bias.data());
        if (std::memcmp(one.data(), ys.data() + (size_t) t * sh.n_out, (size_t) sh.n_out * sizeof(float)) != 0) {
            std::printf("  FAIL %s: token %d of %d differs from the same token alone\n", sh.name, t, n_tok);
            ++g_failures;
            break;
        }
    }

    // (5) with a bias: the same product plus the bias, added after the reduction
    if (!with_bias && sh.n_out > 16) {
        std::vector<float> b2((size_t) sh.n_out);
        std::mt19937 rb(7);   // NOLINT(bugprone-random-generator-seed): a reproducible bias.
        std::normal_distribution<float> g(0.0f, 0.03f);
        for (auto& v : b2) v = g(rb);
        std::vector<float> yb(ys.size(), 0.0f);
        const auto isa = bf16_gemm_isa();
        bf16_gemm_rows(isa, panel.data(), sh.n_in, sh.n_out, x.data(), (size_t) sh.n_in, yb.data(), (size_t) sh.n_out,
                       n_tok, 0, panels, b2.data());
        size_t bad = 0;
        for (size_t i = 0; i < ys.size(); ++i)
            if (yb[i] != ys[i] + b2[i % (size_t) sh.n_out]) ++bad;
        if (bad != 0) {
            std::printf("  FAIL %s: bias is not the sum plus the bias (%zu values)\n", sh.name, bad);
            ++g_failures;
        }
    }
}

/// (6) the same matrix on 1, 2 and 6 threads, through the band split a worker would take.
void check_threads() {
    const Shape sh{"threads down_proj", 2048, 512};
    const int n_tok = 37;
    const int panels = bf16_gemm_panels(sh.n_out);
    std::mt19937 rng(99);   // NOLINT(bugprone-random-generator-seed): reproducible bands.
    std::vector<uint16_t> w((size_t) sh.n_out * sh.n_in);
    fill_weights(rng, w);
    std::vector<uint16_t> panel(bf16_gemm_packed_bytes(sh.n_out, sh.n_in));
    bf16_gemm_pack(w.data(), (size_t) sh.n_in, sh.n_in, sh.n_out, panel.data(), 0, panels);
    std::vector<float> x((size_t) n_tok * sh.n_in);
    fill_activations(rng, x);
    std::vector<float> once((size_t) n_tok * sh.n_out, 0.0f);
    const auto isa = cpu_avx2_ok() ? Bf16GemmIsa::avx2 : Bf16GemmIsa::scalar;
    bf16_gemm_rows(isa, panel.data(), sh.n_in, sh.n_out, x.data(), (size_t) sh.n_in, once.data(), (size_t) sh.n_out,
                   n_tok, 0, panels, nullptr);

    for (int nt : {1, 2, 6}) {
        std::vector<float> y((size_t) n_tok * sh.n_out, 0.0f);
        std::vector<std::thread> th;
        std::atomic<int> next{0};
        // Panels per worker, in whole panels: the same split the forward uses, and the same determinism it needs.
        const int per = (panels + nt - 1) / nt;
        for (int i = 0; i < nt; ++i) {
            const int p0 = i * per, p1 = std::min(panels, p0 + per);
            if (p0 >= p1) break;
            th.emplace_back([&, p0, p1] {
                bf16_gemm_rows(isa, panel.data(), sh.n_in, sh.n_out, x.data(), (size_t) sh.n_in, y.data(),
                               (size_t) sh.n_out, n_tok, p0, p1, nullptr);
                next.fetch_add(1);
            });
        }
        for (auto& t : th) t.join();
        if (y != once) {
            std::printf("  FAIL threads %d: differs from one thread\n", nt);
            ++g_failures;
        }
    }
}

/// A raw little-endian float32/uint16 array, the record's `make_numpy_ref.py` writes them.
std::vector<char> read_file(const std::string& path, size_t expect_bytes, bool& ok) {
    std::ifstream f(path, std::ios::binary);
    ok = false;
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const auto n = f.tellg();
    if ((size_t) n != expect_bytes) {
        std::printf("  numpy: %s is %lld bytes, expected %zu - skipping the file\n", path.c_str(), (long long) n,
                    expect_bytes);
        return {};
    }
    std::vector<char> buf((size_t) n);
    f.seekg(0);
    f.read(buf.data(), (std::streamsize) n);
    ok = (bool) f;
    return buf;
}

/// (7) against a foreign float64 sum: cosine and the largest relative difference.  Made by the record's
/// `make_numpy_ref.py` for real EmbeddingGemma 2 tensors and real reference activations.
int check_numpy(const std::string& dir) {
    std::string line;
    std::ifstream list(dir + "/shapes.txt");
    if (!list) {
        std::printf("bf16_gemm_parity: no %s - skipped (exit 77)\n", (dir + "/shapes.txt").c_str());
        return 77;
    }
    int cases = 0, bad = 0;
    while (list >> line) {
        const char* name = line.c_str();
        int n_in = 0, n_out = 0, n_tok = 0;
        if (!(list >> n_in >> n_out >> n_tok)) break;
        bool ok = false;
        std::string base = std::string(dir) + "/" + name;
        auto w = read_file(base + ".w_u16.bin", (size_t) n_out * n_in * 2, ok);
        if (!ok) {
            std::printf("  %s: missing weight file\n", name);
            ++bad;
            continue;
        }
        auto xa = read_file(base + ".x_f32.bin", (size_t) n_tok * n_in * 4, ok);
        auto yref = read_file(base + ".y_f64.bin", (size_t) n_tok * n_out * 8, ok);
        if (!ok) {
            std::printf("  %s: missing reference file\n", name);
            ++bad;
            continue;
        }
        const int panels = bf16_gemm_panels(n_out);
        std::vector<uint16_t> panel(bf16_gemm_packed_bytes(n_out, n_in));
        bf16_gemm_pack(reinterpret_cast<const uint16_t*>(w.data()), (size_t) n_in, n_in, n_out, panel.data(), 0,
                       panels);
        std::vector<float> y((size_t) n_tok * n_out, 0.0f);
        bf16_gemm_rows(bf16_gemm_isa(), panel.data(), n_in, n_out,
                       reinterpret_cast<const float*>(xa.data()), (size_t) n_in, y.data(), (size_t) n_out, n_tok, 0,
                       panels, nullptr);
        const double* ref = reinterpret_cast<const double*>(yref.data());
        double dot = 0, na = 0, nb = 0, maxrel = 0, maxabs = 0, absmax = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            dot += (double) y[i] * ref[i];
            na += (double) y[i] * y[i];
            nb += ref[i] * ref[i];
            const double d = std::fabs((double) y[i] - ref[i]);
            if (d > maxabs) maxabs = d;
            if (std::fabs(ref[i]) > 1e-30) maxrel = std::max(maxrel, d / std::fabs(ref[i]));
            absmax = std::max(absmax, std::fabs((double) y[i]));
        }
        const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-300);
        std::printf("  %-28s %6dx%-6d x%-4d cos %.12f  max rel %.3e  max abs %.3e (absmax %.1f)\n", name, n_in,
                    n_out, n_tok, cos, maxrel, maxabs, absmax);
        ++cases;
        if (cos < 0.9999999) {
            std::printf("  FAIL %s: cosine %.12f below 0.9999999 against the float64 sum\n", name, cos);
            ++bad;
        }
    }
    list.close();
    std::printf("bf16_gemm_parity: numpy reference %d cases, %d problems\n", cases, bad);
    return bad == 0 ? 0 : 1;
}

// ---- bench ----

double gflops(int n_in, int n_out, int n_tok, double ms) {
    return ms <= 0 ? 0.0 : 2.0 * (double) n_in * (double) n_out * (double) n_tok / (ms * 1e6);
}

/// One thread on one matrix, `reps` times, without the pack.
double bench_one(int n_in, int n_out, int n_tok, int reps, std::vector<uint16_t>& panel, std::vector<float>& x,
                 std::vector<float>& y, Bf16GemmIsa isa) {
    const int panels = bf16_gemm_panels(n_out);
    bf16_gemm_rows(isa, panel.data(), n_in, n_out, x.data(), (size_t) n_in, y.data(), (size_t) n_out, n_tok, 0,
                   panels, nullptr);   // warm
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
        bf16_gemm_rows(isa, panel.data(), n_in, n_out, x.data(), (size_t) n_in, y.data(), (size_t) n_out, n_tok, 0,
                       panels, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

/// `threads` workers over panel bands of the same matrix.
double bench_mt(int n_in, int n_out, int n_tok, int reps, int threads, const std::vector<uint16_t>& panel,
                const std::vector<float>& x, std::vector<float>& y, Bf16GemmIsa isa) {
    const int panels = bf16_gemm_panels(n_out);
    const int per = (panels + threads - 1) / threads;
    auto run = [&] {
        std::vector<std::thread> th;
        for (int i = 0; i < threads; ++i) {
            const int p0 = i * per, p1 = std::min(panels, p0 + per);
            if (p0 >= p1) break;
            th.emplace_back([&, p0, p1] {
                bf16_gemm_rows(isa, panel.data(), n_in, n_out, x.data(), (size_t) n_in, y.data(), (size_t) n_out,
                               n_tok, p0, p1, nullptr);
            });
        }
        for (auto& t : th) t.join();
    };
    run();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) run();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int bench(int reps) {
    if (!cpu_avx2_ok()) {
        std::printf("bf16_gemm_parity --bench: no AVX2 here (the scalar kernel is the reference, not a speed)\n");
        return 0;
    }
    std::printf("bf16_gemm_parity --bench on %s: AVX2, token block %d at n_in 512, %d reps\n",
                cpu_name().c_str(), bf16_gemm_token_block(512), reps);
    std::printf("  %-28s %8s %8s %10s %10s %10s\n", "matrix", "n_in x n_out", "n_tok", "1 thread", "6 threads",
                "6 th GFLOP/s");
    struct Cell { const char* name; int n_in, n_out, n_tok; };
    const Cell cells[] = {
        {"gate/up (24 layers)", 512, 2048, 8},
        {"gate/up (24 layers)", 512, 2048, 64},
        {"gate/up (24 layers)", 512, 2048, 375},
        {"gate/up (24 layers)", 512, 2048, 1104},
        {"down (24 layers)", 2048, 512, 64},
        {"down (24 layers)", 2048, 512, 1104},
        {"ple projection", 512, 12288, 64},
        {"o_proj full", 2048, 512, 64},
        {"embedding projection", 512, 768, 1104},
        {"audio ffn", 1024, 4096, 80},
        {"vision mlp", 768, 3072, 264},
        {"one token gate/up", 512, 2048, 1},
    };
    for (const auto& c : cells) {
        std::mt19937 rng(4242);   // NOLINT(bugprone-random-generator-seed): a reproducible bench.
        std::vector<uint16_t> w((size_t) c.n_out * c.n_in);
        fill_weights(rng, w);
        std::vector<uint16_t> panel(bf16_gemm_packed_bytes(c.n_out, c.n_in));
        bf16_gemm_pack(w.data(), (size_t) c.n_in, c.n_in, c.n_out, panel.data(), 0, bf16_gemm_panels(c.n_out));
        std::vector<float> x((size_t) c.n_tok * c.n_in);
        fill_activations(rng, x);
        std::vector<float> y((size_t) c.n_tok * c.n_out);
        const double s1 = bench_one(c.n_in, c.n_out, c.n_tok, reps, panel, x, y, Bf16GemmIsa::avx2);
        const double s2 = bench_one(c.n_in, c.n_out, c.n_tok, reps, panel, x, y, Bf16GemmIsa::scalar);
        std::vector<float> ym((size_t) c.n_tok * c.n_out);
        const double m6 = bench_mt(c.n_in, c.n_out, c.n_tok, std::max(1, reps / 4), 6, panel, x, ym,
                                   Bf16GemmIsa::avx2);
        std::printf("  %-28s %4d x %-5d %7d %7.2f ms %7.2f ms %9.1f  (scalar %6.2f ms)\n", c.name, c.n_in, c.n_out,
                    c.n_tok, s1, m6, gflops(c.n_in, c.n_out, c.n_tok, m6), s2);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string a1 = argc > 1 ? argv[1] : "";
    if (a1 == "--numpy") {
        if (argc >= 3) return check_numpy(argv[2]);
        const char* env = strata::env_get("EG2_S2");   // the record's s2/, whose numpy/ holds the float64 sums
        return check_numpy(env != nullptr ? std::string(env) + "/numpy" : std::string());
    }
    std::printf("bf16_gemm_parity: %s (avx2 %d, panels of %d, token block %d at n_in 512)\n", cpu_name().c_str(),
                (int) cpu_avx2_ok(), kBf16Panel, bf16_gemm_token_block(512));
    if (a1 == "--bench") {
        int reps = 20;
        if (argc > 3 && !std::strcmp(argv[2], "--reps")) reps = std::max(1, (int) std::strtol(argv[3], nullptr, 10));
        return bench(reps);
    }

    for (const auto& sh : kShapes)
        for (int n_tok : {1, 2, 3, 4, 5, 7, 8, 9, 11}) check_shape(sh, n_tok, false);
    for (const auto& sh : kShapes)
        if (sh.n_out > 64) check_shape(sh, 6, true);
    check_threads();
    std::printf("bf16_gemm_parity: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
