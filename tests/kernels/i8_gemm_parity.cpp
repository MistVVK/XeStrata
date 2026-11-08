// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// tests/kernels/i8_gemm_parity.cpp - the CPU's int8 x int8 matrix product (kernels/cpu/i8_gemm.hpp): the AVX-VNNI
// and the AVX2 kernels against the scalar reference bit for bit, the quantizer's two copies bit for bit, the packing
// against a split of the same matrix and its FP32 form against its BF16 one, the answer against the number of threads
// and the token group, how far the int8 answer is from the exact one, and their speed; with a scale per row
// (EmbeddingGemma 2) and per 32 k values (the plain dense model), the latter also in the seven-bit weight form the AVX2
// kernel takes.  CPU only: no model, no GPU.
//
//   cpu_i8_gemm_parity                   every check below on the EmbeddingGemma 2 and Qwen3.5 shapes and on odd ones
//   cpu_i8_gemm_parity --bench [--reps N]  GOP/s per shape, one thread and six; the Qwen3.5 shapes on each copy and
//                                          weight form
//   cpu_i8_gemm_parity --one N_IN N_OUT N_TOK  one product between SDE's region marks (instruction counts)
//   cpu_i8_gemm_parity --numpy DIR       the independent check against a float64 reference in DIR (the record's
//                                        `s2b/numpy/`; made by `make_numpy_int8_ref.py`).  Without DIR or without
//                                        the files the test exits 77, a CTest skip: the bit-exact checks below
//                                        share their scheme of the quantization with the reference, and only a
//                                        foreign implementation catches a mistake in THAT.
//
// WHAT EACH CHECK PROTECTS.
//   (1) AVX-512 VNNI == AVX-VNNI-INT8 == AVX-VNNI == AVX2 == scalar, every shape, every token group - the vector
//       kernels are the ones that run, the scalar one is the definition.  Integer sums, so they have no reason to
//       differ; a lane put in the wrong place or a missing quad of padding gives a difference here and nowhere else.
//   (2) the two quantizers' bytes: the AVX2 lanes and the scalar loop must write the same levels, or every check
//       below would be comparing two different inputs rather than two ways of multiplying one.
//   (3) a band of panels == the whole matrix, and a token inside a group == the same token alone - the forward
//       splits a matrix over the CPU workers by panel band and the server builds the batch, and neither may move
//       an answer.
//   (4) the packed bytes, scales and corrections of panels [p0,p1) == the same panels of the whole pack.
//   (5) bias == the same product with the bias added at the end, in the fused multiply-add.
//   (6) the offset: an activation row of all +128 bytes (which is a row of zeros) must answer the bias, whatever
//       the weights - the one check that says corr is the weight's row sum and not something else.
//   (7) threads: one, two and six threads give one answer.
//   (8) how far int8 is from exact: the same product summed in double, with the checkpoint's own BF16 weights.
//       This is not a pass condition on a random field - it says the quantization is doing what the scheme's
//       bound says it does, and it is the number the record quotes next to the forward's cosine.
#include "strata/kernels/cpu/i8_gemm.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/cpu/bf16_gemm.hpp"
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

/// The EmbeddingGemma 2 forward's own linear layers (s0 FINDINGS 4・5・6), the same table as the BF16 kernel's
/// test, plus odd shapes for the padding paths and one wide shape for kI8MaxIn's edge.
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
    {"one value", 1, 8},
};

/// The plain dense model's (Qwen3.5) matrices on the CPU, packed with a scale per 32 k values: the 0.8B's and the
/// 2B's attention, GDN and FFN shapes, and odd ones for a short last block.
const Shape kBlockShapes[] = {
    {"q35 0.8B qkv", 1024, 6144},
    {"q35 0.8B ffn gate/up", 1024, 3584},
    {"q35 0.8B ffn down", 3584, 1024},
    {"q35 2B ssm_out", 2048, 2048},
    {"q35 2B ffn down", 6144, 2048},
    {"q35 alpha", 1024, 16},
    {"block short last", 517, 1024},
    {"block and a quad", 36, 40},
    {"one block", 32, 8},
};

void fill_weights(std::mt19937& rng, std::vector<uint16_t>& w) {
    std::normal_distribution<float> g(0.0f, 0.03f);
    for (auto& v : w) v = strata::kernels::bf16_from_f32(g(rng));
}

void fill_activations(std::mt19937& rng, std::vector<float>& x) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (auto& v : x) v = g(rng);
}

/// One shape: pack, quantize, run every kernel and both quantizers, check (1)(2)(3)(5)(8).  `block` is the scale
/// block (0: one per row, EmbeddingGemma 2's; 32: the plain model's), `levels` the weight form.
void check_shape(const Shape& sh, int n_tok, bool with_bias, int block, int levels = kI8Levels) {
    const int panels = i8_gemm_panels(sh.n_out);
    const int row_bytes = i8_gemm_act_row_bytes(sh.n_in);
    const int nb = i8_gemm_blocks(sh.n_in, block);
    std::mt19937 rng(0x18u + (unsigned) sh.n_in * 7919u + (unsigned) sh.n_out + (unsigned) n_tok * 31u +
                     (with_bias ? 1u : 0u) + (unsigned) block);   // NOLINT(bugprone-random-generator-seed): a shape's rows are fixed.
    std::vector<uint16_t> w((size_t) sh.n_out * sh.n_in);
    fill_weights(rng, w);
    std::vector<float> bias;
    if (with_bias) {
        bias.resize((size_t) sh.n_out);
        std::normal_distribution<float> g(0.0f, 0.03f);
        for (auto& v : bias) v = g(rng);
    }
    std::vector<float> x((size_t) n_tok * sh.n_in);
    fill_activations(rng, x);

    std::vector<uint8_t> panel(i8_gemm_packed_bytes(sh.n_out, sh.n_in));
    std::vector<float> wscale(i8_gemm_scale_count(sh.n_out, sh.n_in, block));
    std::vector<int32_t> wcorr(wscale.size());
    if (!i8_gemm_pack(w.data(), (size_t) sh.n_in, sh.n_in, sh.n_out, panel.data(), wscale.data(), wcorr.data(), 0,
                      panels, block, levels)) {
        fail(sh.name, "the packer refused a shape it should take");
        return;
    }

    // (4) the pack split by panel band is the same bytes, scales and corrections
    {
        std::vector<uint8_t> split(panel.size());
        std::vector<float> ss(wscale.size());
        std::vector<int32_t> cs(wcorr.size());
        const int cut = panels / 2;
        i8_gemm_pack(w.data(), (size_t) sh.n_in, sh.n_in, sh.n_out, split.data(), ss.data(), cs.data(), 0, cut,
                     block, levels);
        i8_gemm_pack(w.data(), (size_t) sh.n_in, sh.n_in, sh.n_out, split.data(), ss.data(), cs.data(), cut, panels,
                     block, levels);
        if (split != panel || ss != wscale || cs != wcorr) fail(sh.name, "pack split over panels differs");
        // the FP32 rows give the same pack as the BF16 ones they widen to
        std::vector<float> wf(w.size());
        for (size_t i = 0; i < w.size(); ++i) wf[i] = strata::kernels::f32_from_bf16(w[i]);
        std::fill(split.begin(), split.end(), (uint8_t) 0);
        i8_gemm_pack_f32(wf.data(), (size_t) sh.n_in, sh.n_in, sh.n_out, split.data(), ss.data(), cs.data(), 0,
                         panels, block, levels);
        if (split != panel || ss != wscale || cs != wcorr) fail(sh.name, "the FP32 pack differs from the BF16 one");
    }

    // (2) the two quantizers write the same bytes and scales
    std::vector<uint8_t> xq((size_t) n_tok * row_bytes), xqs(xq.size());
    std::vector<float> xs((size_t) n_tok * nb), xss(xs);
    const I8GemmIsa isa = i8_gemm_isa();
    i8_gemm_quantize(I8GemmIsa::scalar, x.data(), (size_t) sh.n_in, n_tok, sh.n_in, xqs.data(), (size_t) row_bytes,
                     xss.data(), block);
    i8_gemm_quantize(isa, x.data(), (size_t) sh.n_in, n_tok, sh.n_in, xq.data(), (size_t) row_bytes, xs.data(), block);
    // the VNNI-INT8 copy's bytes are the signed levels: the scalar bytes without their +128 (the padding stays 0)
    auto as_isa = [&](std::vector<uint8_t> b, I8GemmIsa i) {
        if (i == I8GemmIsa::vnni_int8)
            for (int t = 0; t < n_tok; ++t)
                for (int k = 0; k < sh.n_in; ++k) b[(size_t) t * row_bytes + k] ^= 0x80;
        return b;
    };
    if (xq != as_isa(xqs, isa)) {
        size_t first = 0;
        while (first < xq.size() && xq[first] == xqs[first]) ++first;
        char buf[128];
        const size_t at = first < xq.size() ? first : 0;      // the print never reads past the buffer
        std::snprintf(buf, sizeof buf, "quantizer bytes differ at %zu (%u vs %u), n_tok %d", first,
                      (unsigned) xq[at], (unsigned) xqs[at], n_tok);
        fail(sh.name, buf);
    }
    if (xs != xss) fail(sh.name, "quantizer scales differ");

    // (1) the kernels: the scalar one is the definition, every vector copy this CPU runs is held to it
    std::vector<float> ys((size_t) n_tok * sh.n_out, 0.0f);
    i8_gemm_rows(I8GemmIsa::scalar, panel.data(), wscale.data(), wcorr.data(), sh.n_in, sh.n_out, xqs.data(),
                 (size_t) row_bytes, xs.data(), ys.data(), (size_t) sh.n_out, n_tok, 0, panels,
                 bias.empty() ? nullptr : bias.data(), block, levels);
    // the copies' order of preference: a run capped at one (XESTRATA_I8_GEMM_ISA) checks it and the ones below it
    auto rank = [](I8GemmIsa i) {
        return i == I8GemmIsa::avx512 ? 4 : i == I8GemmIsa::vnni_int8 ? 3 : i == I8GemmIsa::vnni ? 2
               : i == I8GemmIsa::avx2 ? 1 : 0;
    };
    for (const I8GemmIsa vi : {I8GemmIsa::avx512, I8GemmIsa::vnni_int8, I8GemmIsa::vnni, I8GemmIsa::avx2}) {
        if (isa == I8GemmIsa::scalar || !i8_gemm_isa_ok(vi) || rank(vi) > rank(isa)) continue;
        const std::vector<uint8_t> xv = as_isa(xqs, vi);   // the bytes this copy's quantizer writes
        std::vector<float> yv((size_t) n_tok * sh.n_out, 0.0f);
        i8_gemm_rows(vi, panel.data(), wscale.data(), wcorr.data(), sh.n_in, sh.n_out, xv.data(), (size_t) row_bytes,
                     xs.data(), yv.data(), (size_t) sh.n_out, n_tok, 0, panels, bias.empty() ? nullptr : bias.data(),
                     block, levels);
        if (yv != ys) {
            size_t first = 0;
            while (first < ys.size() && yv[first] == ys[first]) ++first;
            char buf[160];
            const size_t at = first < ys.size() ? first : 0;   // the print never reads past the buffer
            std::snprintf(buf, sizeof buf, "%s != scalar at %zu (%g vs %g), n_tok %d, bias %d, block %d, levels %d",
                          i8_gemm_isa_name(vi), first, (double) yv[at], (double) ys[at], n_tok, (int) with_bias,
                          block, levels);
            fail(sh.name, buf);
            return;
        }
    }

    // (3) a band of panels is the same answer
    if (panels > 2) {
        std::vector<float> band(ys.size(), 0.0f);
        const int a = 1, b = panels - 1;
        const int lo[3] = {0, a, b}, hi[3] = {a, b, panels};
        for (int i = 0; i < 3; ++i)
            i8_gemm_rows(isa, panel.data(), wscale.data(), wcorr.data(), sh.n_in, sh.n_out, xq.data(),
                         (size_t) row_bytes, xs.data(), band.data(), (size_t) sh.n_out, n_tok, lo[i], hi[i],
                         bias.empty() ? nullptr : bias.data(), block, levels);
        if (band != ys) fail(sh.name, "panel bands differ");
    }

    // (3) every token alone is the same row as inside the group
    for (int t = 0; t < n_tok; ++t) {
        std::vector<float> one((size_t) sh.n_out, 0.0f);
        i8_gemm_rows(isa, panel.data(), wscale.data(), wcorr.data(), sh.n_in, sh.n_out, xq.data() + (size_t) t * row_bytes,
                     (size_t) row_bytes, xs.data() + (size_t) t * nb, one.data(), (size_t) sh.n_out, 1, 0, panels,
                     bias.empty() ? nullptr : bias.data(), block, levels);
        if (std::memcmp(one.data(), ys.data() + (size_t) t * sh.n_out, (size_t) sh.n_out * sizeof(float)) != 0) {
            fail(sh.name, "a token alone differs from the same token in the group");
            break;
        }
    }

    // (6) a zero activation row answers the bias, not the offset
    if (n_tok >= 1) {
        std::vector<uint8_t> z((size_t) row_bytes, isa == I8GemmIsa::vnni_int8 ? 0 : 128);   // the level 0
        std::vector<float> zs((size_t) nb, 0.0f);
        std::vector<float> zy((size_t) sh.n_out, 0.0f);
        i8_gemm_rows(isa, panel.data(), wscale.data(), wcorr.data(), sh.n_in, sh.n_out, z.data(), (size_t) row_bytes,
                     zs.data(), zy.data(), (size_t) sh.n_out, 1, 0, panels, nullptr, block, levels);
        double worst = 0;
        for (int o = 0; o < sh.n_out; ++o) worst = std::max(worst, std::fabs((double) zy[o]));
        if (worst > 0.0) {
            char buf[128];
            std::snprintf(buf, sizeof buf, "a zero row answers %g, not 0 (corr wrong?)", worst);
            fail(sh.name, buf);
        }
    }

    // (5) with a bias: the same product with the bias folded in at the end
    if (!with_bias && sh.n_out > 16) {
        std::vector<float> b2((size_t) sh.n_out);
        std::mt19937 rb(7);   // NOLINT(bugprone-random-generator-seed): a reproducible bias.
        std::normal_distribution<float> g(0.0f, 0.03f);
        for (auto& v : b2) v = g(rb);
        std::vector<float> yb(ys.size(), 0.0f);
        i8_gemm_rows(isa, panel.data(), wscale.data(), wcorr.data(), sh.n_in, sh.n_out, xq.data(),
                     (size_t) row_bytes, xs.data(), yb.data(), (size_t) sh.n_out, n_tok, 0, panels, b2.data(), block,
                     levels);
        size_t bad = 0;
        for (size_t i = 0; i < ys.size(); ++i) {
            const double want = (double) ys[i] + b2[i % (size_t) sh.n_out];
            if (std::fabs((double) yb[i] - want) > 1e-4 * (1.0 + std::fabs(want))) ++bad;
        }
        if (bad != 0) {
            char buf[128];
            std::snprintf(buf, sizeof buf, "bias is not the product plus the bias (%zu values)", bad);
            fail(sh.name, buf);
        }
    }

    // (8) how far the int8 answer is from the exact one on the same weights and activations
    if (sh.n_out >= 512) {
        const int cols = std::min(sh.n_out, 64);
        double dot = 0, na = 0, nb = 0, maxrel = 0;
        for (int t = 0; t < n_tok; ++t) {
            for (int o = 0; o < cols; ++o) {
                double exact = 0;
                for (int k = 0; k < sh.n_in; ++k)
                    exact += (double) x[(size_t) t * sh.n_in + k] *
                             strata::kernels::f32_from_bf16(w[(size_t) o * sh.n_in + k]);
                const double got = ys[(size_t) t * sh.n_out + o];
                dot += got * exact;
                na += got * got;
                nb += exact * exact;
                if (std::fabs(exact) > 1e-3)
                    maxrel = std::max(maxrel, std::fabs(got - exact) / std::fabs(exact));
            }
        }
        const double c = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-300);
        if (c < 0.98) {
            char buf[128];
            std::snprintf(buf, sizeof buf, "int8 cosine against the exact sum is %.6f", c);
            fail(sh.name, buf);
        }
    }
}

/// (7) the same matrix on 1, 2 and 6 threads, through the band split a worker would take.
void check_threads(int block, int levels = kI8Levels) {
    const Shape sh{"threads down_proj", 2048, 512};
    const int n_tok = 37;
    const int panels = i8_gemm_panels(sh.n_out);
    const int row_bytes = i8_gemm_act_row_bytes(sh.n_in);
    std::mt19937 rng(99);   // NOLINT(bugprone-random-generator-seed): reproducible bands.
    std::vector<uint16_t> w((size_t) sh.n_out * sh.n_in);
    fill_weights(rng, w);
    std::vector<uint8_t> panel(i8_gemm_packed_bytes(sh.n_out, sh.n_in));
    std::vector<float> wscale(i8_gemm_scale_count(sh.n_out, sh.n_in, block));
    std::vector<int32_t> wcorr(wscale.size());
    i8_gemm_pack(w.data(), (size_t) sh.n_in, sh.n_in, sh.n_out, panel.data(), wscale.data(), wcorr.data(), 0, panels,
                 block, levels);
    std::vector<float> x((size_t) n_tok * sh.n_in);
    fill_activations(rng, x);
    std::vector<uint8_t> xq((size_t) n_tok * row_bytes);
    std::vector<float> xs((size_t) n_tok * i8_gemm_blocks(sh.n_in, block));
    const I8GemmIsa isa = i8_gemm_isa();
    i8_gemm_quantize(isa, x.data(), (size_t) sh.n_in, n_tok, sh.n_in, xq.data(), (size_t) row_bytes, xs.data(), block);

    std::vector<float> once((size_t) n_tok * sh.n_out, 0.0f);
    i8_gemm_rows(isa, panel.data(), wscale.data(), wcorr.data(), sh.n_in, sh.n_out, xq.data(), (size_t) row_bytes,
                 xs.data(), once.data(), (size_t) sh.n_out, n_tok, 0, panels, nullptr, block, levels);

    for (int nt : {1, 2, 6}) {
        std::vector<float> y((size_t) n_tok * sh.n_out, 0.0f);
        std::vector<std::thread> th;
        const int per = (panels + nt - 1) / nt;
        for (int i = 0; i < nt; ++i) {
            const int p0 = i * per, p1 = std::min(panels, p0 + per);
            if (p0 >= p1) break;
            th.emplace_back([&, p0, p1] {
                i8_gemm_rows(isa, panel.data(), wscale.data(), wcorr.data(), sh.n_in, sh.n_out, xq.data(),
                             (size_t) row_bytes, xs.data(), y.data(), (size_t) sh.n_out, n_tok, p0, p1, nullptr,
                             block, levels);
            });
        }
        for (auto& t : th) t.join();
        if (y != once) {
            std::printf("  FAIL threads %d (block %d, levels %d): differs from one thread\n", nt, block, levels);
            ++g_failures;
        }
    }
}

/// A raw little-endian array, the record's `make_numpy_int8_ref.py` writes it.
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

/// (9) against a foreign implementation of the same quantization and a float64 sum of it: real EmbeddingGemma 2
/// tensors and real fp32 reference activations.  This is the check that the packing, the row sum and the scales
/// mean what the header says they mean, because the Python writes them from the formulas and not from the C.
int check_numpy(const std::string& dir) {
    std::string line;
    std::ifstream list(dir + "/shapes.txt");
    if (!list) {
        std::printf("i8_gemm_parity: no %s - skipped (exit 77)\n", (dir + "/shapes.txt").c_str());
        return 77;
    }
    int cases = 0, bad = 0;
    while (list >> line) {
        int n_in = 0, n_out = 0, n_tok = 0;
        if (!(list >> n_in >> n_out >> n_tok)) break;
        bool ok = false;
        std::string base = dir + "/";
        base += line;
        auto w = read_file(base + ".w_u16.bin", (size_t) n_out * n_in * 2, ok);
        if (!ok) {
            std::printf("  %s: missing weight file\n", line.c_str());
            ++bad;
            continue;
        }
        auto xa = read_file(base + ".x_f32.bin", (size_t) n_tok * n_in * 4, ok);
        auto yref = read_file(base + ".y8_f64.bin", (size_t) n_tok * n_out * 8, ok);
        if (!ok) {
            std::printf("  %s: missing reference file\n", line.c_str());
            ++bad;
            continue;
        }
        const int panels = i8_gemm_panels(n_out);
        const int row_bytes = i8_gemm_act_row_bytes(n_in);
        std::vector<uint8_t> panel(i8_gemm_packed_bytes(n_out, n_in));
        std::vector<float> wscale((size_t) panels * kI8Panel);
        std::vector<int32_t> wcorr((size_t) panels * kI8Panel);
        i8_gemm_pack(reinterpret_cast<const uint16_t*>(w.data()), (size_t) n_in, n_in, n_out, panel.data(),
                     wscale.data(), wcorr.data(), 0, panels);
        std::vector<uint8_t> xq((size_t) n_tok * row_bytes);
        std::vector<float> xs((size_t) n_tok);
        i8_gemm_quantize(i8_gemm_isa(), reinterpret_cast<const float*>(xa.data()), (size_t) n_in, n_tok, n_in,
                         xq.data(), (size_t) row_bytes, xs.data());
        std::vector<float> y((size_t) n_tok * n_out, 0.0f);
        i8_gemm_rows(i8_gemm_isa(), panel.data(), wscale.data(), wcorr.data(), n_in, n_out, xq.data(),
                     (size_t) row_bytes, xs.data(), y.data(), (size_t) n_out, n_tok, 0, panels, nullptr);
        const double* ref = reinterpret_cast<const double*>(yref.data());
        double dot = 0, na = 0, nb = 0, maxabs = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            dot += (double) y[i] * ref[i];
            na += (double) y[i] * y[i];
            nb += ref[i] * ref[i];
            maxabs = std::max(maxabs, std::fabs((double) y[i] - ref[i]));
        }
        const double c = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-300);
        std::printf("  %-28s %6dx%-6d x%-4d cos %.12f  max abs %.3e\n", line.c_str(), n_in, n_out, n_tok, c,
                    maxabs);
        ++cases;
        if (c < 0.9999999) {
            std::printf("  FAIL %s: cosine %.12f below 0.9999999 against the foreign float64 sum\n", line.c_str(),
                        c);
            ++bad;
        }
    }
    std::printf("i8_gemm_parity: numpy int8 reference %d cases, %d problems\n", cases, bad);
    return bad == 0 ? 0 : 1;
}

// ---- bench ----

/// One int8 product costs n_in*n_out*n_tok byte-multiplies; the table quotes them as GOP/s so the BF16 kernel's
/// GFLOP/s column (2 ops per multiply-add) can be read beside it.
double gops(int n_in, int n_out, int n_tok, double ms) {
    return ms <= 0 ? 0.0 : (double) n_in * (double) n_out * (double) n_tok / (ms * 1e6);
}

struct Pack {
    std::vector<uint8_t> panel;
    std::vector<float> wscale;
    std::vector<int32_t> wcorr;
    std::vector<uint8_t> xq;
    std::vector<float> xs;
    std::vector<float> y;
    int row_bytes = 0;
    int block = 0;
    int levels = kI8Levels;
};

void make(int n_in, int n_out, int n_tok, Pack& p, int block = 0, I8GemmIsa isa = i8_gemm_isa(),
          int levels = kI8Levels) {
    std::mt19937 rng(4242);   // NOLINT(bugprone-random-generator-seed): a reproducible bench.
    std::vector<uint16_t> w((size_t) n_out * n_in);
    fill_weights(rng, w);
    const int panels = i8_gemm_panels(n_out);
    p.panel.resize(i8_gemm_packed_bytes(n_out, n_in));
    p.wscale.assign(i8_gemm_scale_count(n_out, n_in, block), 0.0f);
    p.wcorr.assign(p.wscale.size(), 0);
    i8_gemm_pack(w.data(), (size_t) n_in, n_in, n_out, p.panel.data(), p.wscale.data(), p.wcorr.data(), 0, panels,
                 block, levels);
    std::vector<float> x((size_t) n_tok * n_in);
    fill_activations(rng, x);
    p.row_bytes = i8_gemm_act_row_bytes(n_in);
    p.xq.resize((size_t) n_tok * p.row_bytes);
    p.xs.resize((size_t) n_tok * i8_gemm_blocks(n_in, block));
    i8_gemm_quantize(isa, x.data(), (size_t) n_in, n_tok, n_in, p.xq.data(), (size_t) p.row_bytes, p.xs.data(),
                     block);
    p.block = block;
    p.levels = levels;
    p.y.assign((size_t) n_tok * n_out, 0.0f);
}

double bench_one(int n_in, int n_out, int n_tok, int reps, Pack& p, I8GemmIsa isa) {
    const int panels = i8_gemm_panels(n_out);
    auto run = [&] {
        i8_gemm_rows(isa, p.panel.data(), p.wscale.data(), p.wcorr.data(), n_in, n_out, p.xq.data(),
                     (size_t) p.row_bytes, p.xs.data(), p.y.data(), (size_t) n_out, n_tok, 0, panels, nullptr,
                     p.block, p.levels);
    };
    run();
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) run();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

double bench_mt(int n_in, int n_out, int n_tok, int reps, int threads, Pack& p, I8GemmIsa isa) {
    const int panels = i8_gemm_panels(n_out);
    const int per = (panels + threads - 1) / threads;
    auto run = [&] {
        std::vector<std::thread> th;
        for (int i = 0; i < threads; ++i) {
            const int p0 = i * per, p1 = std::min(panels, p0 + per);
            if (p0 >= p1) break;
            th.emplace_back([&, p0, p1] {
                i8_gemm_rows(isa, p.panel.data(), p.wscale.data(), p.wcorr.data(), n_in, n_out, p.xq.data(),
                             (size_t) p.row_bytes, p.xs.data(), p.y.data(), (size_t) n_out, n_tok, p0, p1, nullptr,
                             p.block, p.levels);
            });
        }
        for (auto& t : th) t.join();
    };
    run();
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) run();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

// SDE's region marks (`sde64 -start_ssc_mark 111 -stop_ssc_mark 222 -mix`): only the product between them is counted
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define SSC_MARK(tag) __asm__ __volatile__("movl %0, %%ebx; .byte 0x64, 0x67, 0x90" ::"i"(tag) : "%ebx", "memory")
#else
#define SSC_MARK(tag)
#endif

/// One product of [n_out, n_in] over n_tok rows on the engine's own choice of copy (XESTRATA_I8_GEMM_ISA caps it), a
/// scale per 32 k values, between SDE's marks: the dynamic instruction counts of one copy, per multiply-accumulate.
int one(int n_in, int n_out, int n_tok) {
    Pack p;
    make(n_in, n_out, n_tok, p, 32, i8_gemm_isa(), i8_gemm_weight_levels(i8_gemm_isa()));
    std::printf("cpu_i8_gemm_parity --one %d x %d x %d on %s, %d-bit weights: %.0f multiply-accumulates\n", n_in,
                n_out, n_tok, i8_gemm_isa_name(i8_gemm_isa()), p.levels == kI8Levels7 ? 7 : 8,
                (double) n_in * n_out * n_tok);
    SSC_MARK(0x111);
    i8_gemm_rows(i8_gemm_isa(), p.panel.data(), p.wscale.data(), p.wcorr.data(), n_in, n_out, p.xq.data(),
                 (size_t) p.row_bytes, p.xs.data(), p.y.data(), (size_t) n_out, n_tok, 0, i8_gemm_panels(n_out), nullptr,
                 p.block, p.levels);
    SSC_MARK(0x222);
    return 0;
}

int bench(int reps) {
    std::printf("i8_gemm_parity --bench on %s: int8 isa=%s, token block %d at n_in 512, %d reps\n",
                cpu_name().c_str(), i8_gemm_isa_name(i8_gemm_isa()), bf16_gemm_token_block(512), reps);
    std::printf("  %-28s %10s %10s %10s %10s %10s\n", "matrix", "n_in x n_out", "n_tok", "int8 1th", "int8 6th",
                "6th GOP/s");
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
        Pack p;
        make(c.n_in, c.n_out, c.n_tok, p);
        const double s1 = bench_one(c.n_in, c.n_out, c.n_tok, reps, p, i8_gemm_isa());
        const double m6 = bench_mt(c.n_in, c.n_out, c.n_tok, std::max(1, reps / 4), 6, p, i8_gemm_isa());
        std::printf("  %-28s %4d x %-5d %7d %8.2f ms %8.2f ms %9.1f\n", c.name, c.n_in, c.n_out, c.n_tok, s1, m6,
                    gops(c.n_in, c.n_out, c.n_tok, m6));
    }
    // the plain dense model's matrices, a scale per 32 k values, on each vector copy this CPU runs (the AVX2 one in
    // both weight forms)
    const Cell q35[] = {
        {"q35 2B ffn gate/up", 2048, 6144, 1692},
        {"q35 2B ffn down", 6144, 2048, 1692},
        {"q35 2B qkv", 2048, 6144, 64},
        {"q35 2B ffn gate/up 1 token", 2048, 6144, 1},
    };
    for (const I8GemmIsa vi : {I8GemmIsa::avx512, I8GemmIsa::vnni_int8, I8GemmIsa::vnni, I8GemmIsa::avx2}) {
        if (!i8_gemm_isa_ok(vi)) continue;
        for (const int levels : {kI8Levels, kI8Levels7}) {
            if (levels == kI8Levels7 && vi != I8GemmIsa::avx2) continue;
            for (const auto& c : q35) {
                Pack p;
                make(c.n_in, c.n_out, c.n_tok, p, 32, vi, levels);
                const int r = c.n_tok > 256 ? std::max(1, reps / 10) : reps;
                const double s1 = bench_one(c.n_in, c.n_out, c.n_tok, r, p, vi);
                const double m6 = bench_mt(c.n_in, c.n_out, c.n_tok, std::max(1, r / 4), 6, p, vi);
                std::printf("  %-22s %-11s w%d %4d x %-5d %7d %8.2f ms %8.2f ms %9.1f\n", c.name,
                            i8_gemm_isa_name(vi), levels == kI8Levels7 ? 7 : 8, c.n_in, c.n_out, c.n_tok, s1, m6,
                            gops(c.n_in, c.n_out, c.n_tok, m6));
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string a1 = argc > 1 ? argv[1] : "";
    if (a1 == "--numpy") {
        if (argc >= 3) return check_numpy(argv[2]);
        const char* env = strata::env_get("EG2_S2B");   // the record's s2b/, whose numpy/ holds the float64 sums
        return check_numpy(env != nullptr ? std::string(env) + "/numpy" : std::string());
    }
    std::printf("i8_gemm_parity: %s (avx2 %d, avxvnni %d, int8 isa %s, panels of %d, quads of %d)\n",
                cpu_name().c_str(), (int) cpu_avx2_ok(), (int) cpu_avxvnni_ok(),
                i8_gemm_isa_name(i8_gemm_isa()), kI8Panel, kI8Quad);
    if (a1 == "--one" && argc >= 5)
        return one((int) std::strtol(argv[2], nullptr, 10), (int) std::strtol(argv[3], nullptr, 10),
                   (int) std::strtol(argv[4], nullptr, 10));
    if (a1 == "--bench") {
        int reps = 20;
        if (argc >= 4 && !std::strcmp(argv[2], "--reps")) reps = std::max(1, (int) std::strtol(argv[3], nullptr, 10));
        return bench(reps);
    }
    if (!cpu_avxvnni_ok() && strata::env_get("I8_GEMM_ISA") == nullptr)
        std::printf("  note: this CPU does not run AVX-VNNI, so the AVX2 copy (or the scalar one) is under test; the\n"
                    "        VNNI copy is checked on a CPU that has it and the copies agree bit for bit there.\n");

    for (const auto& sh : kShapes)
        for (int n_tok : {1, 2, 3, 4, 5, 7, 8, 9, 11}) check_shape(sh, n_tok, false, 0);
    for (const auto& sh : kShapes)
        if (sh.n_out > 64) check_shape(sh, 6, true, 0);
    for (const auto& sh : kBlockShapes)
        for (int n_tok : {1, 3, 4, 5, 8, 9, 13}) check_shape(sh, n_tok, false, 32);
    for (const auto& sh : kBlockShapes)
        if (sh.n_out > 64) check_shape(sh, 6, true, 32);
    // the seven-bit weight form the AVX2 kernel packs for, on the plain model's shapes: every copy against the scalar
    for (const auto& sh : kBlockShapes)
        for (int n_tok : {1, 3, 4, 5, 8, 9, 13}) check_shape(sh, n_tok, false, 32, kI8Levels7);
    for (const auto& sh : kBlockShapes)
        if (sh.n_out > 64) check_shape(sh, 6, true, 32, kI8Levels7);
    check_threads(0);
    check_threads(32);
    check_threads(32, kI8Levels7);
    std::printf("i8_gemm_parity: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
