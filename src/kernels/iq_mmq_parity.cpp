// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/iq_mmq_parity.cpp - the prompt's int8 expert products (iq_mmq.hpp) in the form this GPU takes (Intel's
// XMX, NVIDIA's tensor cores, AMD's WMMA, or dp4a) against the host: the activations as the device quantized them,
// the weights dequantized by ggml (to_float), multiplied in double.
//
//     build/iq_mmq_parity --selftest
//     build/iq_mmq_parity --bench      the int8 and the FP16 form of a layer's experts, timed (not in CTest)
//
// A type for each way the products read scales: a scale per 32 (Q8_0, at a K with a tail past the step; IQ3_XXS), a
// scale per 16 (IQ2_XS) and a min term (Q4_K); each over three experts of 5, 40 and 17 rows and one of 70 (the tile
// shapes of 16, 32 and 64 rows).  The weights are random block bytes with a sane d: every bit pattern of the grid
// indices and signs is a valid block.  Exit 77 (skipped) where the GPU runs neither form.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "parity_device.hpp"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

namespace P = strata::parity;

// A fixed sequence (a std engine seeded with a constant draws a lint warning).
struct XorShift32 {
    using result_type = uint32_t;
    uint32_t state;
    static constexpr result_type min() { return 1; }
    static constexpr result_type max() { return 0xFFFFFFFFu; }
    result_type operator()() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
};
XorShift32& rng() {
    static XorShift32 g{20261013u};
    return g;
}

// Rows of `cols` values of type t: ggml's quantizer where it needs no importance matrix, else random block bytes whose
// fp16 scales (the block's first two bytes, and Q4_K's next two) are set to small positive values.
std::vector<uint8_t> weights(ggml_type t, int64_t rows, int64_t cols) {
    const size_t rb = ggml_row_size(t, cols);
    std::vector<uint8_t> w(rb * (size_t) rows);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    if (!ggml_quantize_requires_imatrix(t) && t != GGML_TYPE_IQ3_XXS) {
        std::vector<float> f((size_t) (rows * cols));
        for (float& v : f) v = u(rng());
        ggml_quantize_chunk(t, f.data(), w.data(), 0, rows, cols, nullptr);
        return w;
    }
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint8_t& b : w) b = (uint8_t) byte(rng());
    const size_t bs = ggml_type_size(t);
    std::uniform_real_distribution<float> d(0.002f, 0.02f);
    for (size_t at = 0; at < w.size(); at += bs) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(d(rng()));
        std::memcpy(&w[at], &h, 2);
    }
    return w;
}

int check(ggml_type t, int64_t N, int64_t K, const std::vector<int32_t>& counts) {
    sycl::queue& q = strata::core::Runtime::get().compute();
    const int n = (int) counts.size();
    std::vector<int32_t> bounds(1, 0);
    int64_t max_rows = 0;
    for (int32_t c : counts) {
        bounds.push_back(bounds.back() + c);
        max_rows = std::max<int64_t>(max_rows, c);
    }
    const int64_t R = bounds.back();
    std::vector<std::vector<uint8_t>> w;
    std::vector<uint8_t> all;
    for (int e = 0; e < n; ++e) {
        w.push_back(weights(t, N, K));
        all.insert(all.end(), w.back().begin(), w.back().end());
    }
    std::vector<float> x((size_t) (R * K));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x) v = nd(rng());
    std::vector<int32_t> ids((size_t) R);
    for (int64_t i = 0; i < R; ++i) ids[(size_t) i] = (int32_t) i;

    P::Device dev;
    const size_t xq_bytes = strata::kernels::iq_mmq_act_bytes(R, K);
    float* dx = dev.upload(x);
    uint8_t* dw = dev.upload(all);
    int32_t* db = dev.upload(bounds);
    int32_t* di = dev.upload(ids);
    uint8_t* dxq = dev.alloc<uint8_t>(xq_bytes);
    float* dy = dev.alloc<float>((size_t) (R * N));
    strata::kernels::iq_mmq_quantize(q, dx, nullptr, dxq, K, K, R, strata::kernels::iq_mmq_type_sums((int) t)).wait();
    strata::kernels::iq_mmq_grouped(q, (int) t, dw, w[0].size(), N, K, n, dxq, R, db, max_rows, di, dy, N).wait();
    std::vector<uint8_t> xq(xq_bytes);
    std::vector<float> y((size_t) (R * N));
    P::copy_out(xq.data(), dxq, xq_bytes);
    P::copy_out(y.data(), dy, y.size());

    // the reference: the device's int8 activations times their scales, the weights through ggml's dequantizer
    const auto* x8 = reinterpret_cast<const int8_t*>(xq.data());
    const auto* xs = reinterpret_cast<const float*>(xq.data() + R * K);
    const ggml_type_traits* tr = ggml_get_type_traits(t);
    std::vector<float> wf((size_t) (N * K));
    double worst = 0.0, scale = 0.0;
    for (int e = 0; e < n; ++e) {
        for (int64_t o = 0; o < N; ++o) tr->to_float(w[(size_t) e].data() + (size_t) o * ggml_row_size(t, K), &wf[(size_t) (o * K)], K);
        for (int64_t r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r)
            for (int64_t o = 0; o < N; ++o) {
                double s = 0.0;
                for (int64_t k = 0; k < K; ++k)
                    s += (double) x8[r * K + k] * xs[r * (K / 32) + k / 32] * wf[(size_t) (o * K + k)];
                worst = std::max(worst, std::fabs(s - y[(size_t) (r * N + o)]));
                scale = std::max(scale, std::fabs(s));
            }
    }
    const double rel = worst / std::max(scale, 1e-30);
    const bool ok = rel < 1e-4 && std::isfinite(rel);
    std::printf("  %-8s K %4lld, %d experts of up to %3lld rows: rel %.2e  %s\n", ggml_type_name(t), (long long) K, n,
                (long long) max_rows, rel, ok ? "ok" : "*** FAIL ***");
    return ok ? 0 : 1;
}

// --bench: a layer's experts as PlainModel's prompt takes them, 16 experts of r rows each, in the two forms the
// prompt chooses between: int8 (gate and up products, down product: iq_mmq_grouped) and FP16 (the weights
// dequantized, then grouped products: dequant_f16, xmx_gemm_grouped).  The steps both forms share (the routed rows'
// input, the SwiGLU, the down product's input) are left out.  Synthetic weights (any bits time alike), the shapes and
// types of Ornith-1.5 (Q4_K_M, GSQ-RCO 3bit), llm-jp-4.1 Q4_K_M and Qwen3.8's IQ3_XXS pack.
struct Combo {
    const char* name;
    ggml_type gate, up, down;
    int64_t N, FF;
};

double time_us(sycl::queue& q, const std::function<void()>& f) {
    f();
    q.wait();
    f();
    q.wait();
    constexpr int reps = 10, trials = 5;   // the best of 5 means of 10: the GPU's clock and the host's moves
    double best = 1e30;
    for (int t = 0; t < trials; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) f();
        q.wait();
        best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps);
    }
    return best;
}

int bench() {
    namespace k = strata::kernels;
    sycl::queue& q = strata::core::Runtime::get().compute();
    void* s = P::stream();
    const bool matrix = k::iq_mmq_usable(q);
    std::printf("iq_mmq_parity --bench: int8 %s, FP16 products %s\n", matrix ? "on the matrix engines" : "by dp4a",
                k::gemm_path(k::XmxType::f16));
    const Combo combos[] = {{"ornith-q4_k_m", GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 2048, 512},
                            {"ornith-3bit", GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 2048, 512},
                            {"llmjp-q4_k_m", GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, 2560, 960},
                            {"qwen-iq2_xs", GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS, 2560, 640}};
    constexpr int G = 16;
    for (const Combo& c : combos) {
        const int64_t N = c.N, FF = c.FF;
        const bool one = c.gate == c.up;   // gate and up as one matrix of 2 FF rows (one product)
        const size_t gb = ggml_row_size(c.gate, N) * (size_t) FF, ub = ggml_row_size(c.up, N) * (size_t) FF;
        const size_t db = ggml_row_size(c.down, FF) * (size_t) N;
        P::Device dev;
        uint8_t* wg = dev.alloc<uint8_t>((gb + ub) * G);   // per expert: gate's rows, then up's
        uint8_t* wd = dev.alloc<uint8_t>(db * G);
        q.memset(wg, 0x11, (gb + ub) * G).wait();
        q.memset(wd, 0x11, db * G).wait();
        uint16_t* dq_gu = dev.alloc<uint16_t>((size_t) G * 2 * (size_t) (FF * N));
        uint16_t* dq_d = dev.alloc<uint16_t>((size_t) (G * N * FF));
        if (one) {   // the forms the engine's warming chooses (iq_mmq_choose_forms)
            k::iq_mmq_choose_forms(q, (int) c.gate, 2 * FF, N, G);
        } else {
            k::iq_mmq_choose_forms(q, (int) c.gate, FF, N, G);
            k::iq_mmq_choose_forms(q, (int) c.up, FF, N, G);
        }
        k::iq_mmq_choose_forms(q, (int) c.down, N, FF, G);
        for (int64_t r : {8, 16, 32, 64, 128, 256}) {
            const int64_t R = G * r;
            std::vector<int32_t> bounds(G + 1), ids((size_t) R);
            for (int e = 0; e <= G; ++e) bounds[(size_t) e] = (int32_t) (e * r);
            for (int64_t i = 0; i < R; ++i) ids[(size_t) i] = (int32_t) i;
            P::Device io;
            int32_t* db_ = io.upload(bounds);
            int32_t* di = io.upload(ids);
            auto xq_of = [&](ggml_type t, int64_t cols) {   // quantized rows of zeros for type t's products
                const bool dup = k::iq_mmq_type_dup((int) t);
                uint8_t* p = io.alloc<uint8_t>(k::iq_mmq_act_bytes(R, dup ? 2 * cols : cols));
                float* z = io.alloc<float>((size_t) (R * cols));
                q.memset(z, 0, (size_t) (R * cols) * 4).wait();
                k::iq_mmq_quantize(q, z, nullptr, p, cols, cols, R, k::iq_mmq_type_sums((int) t), dup).wait();
                return p;
            };
            const uint8_t *xg = xq_of(c.gate, N), *xu = one ? xg : xq_of(c.up, N), *xh = xq_of(c.down, FF);
            float* gu = io.alloc<float>((size_t) (R * 2 * FF));
            float* dn = io.alloc<float>((size_t) (R * N));
            uint16_t* x16 = io.alloc<uint16_t>((size_t) (R * N));
            uint16_t* h16 = io.alloc<uint16_t>((size_t) (R * FF));
            q.memset(x16, 0, (size_t) (R * N) * 2).wait();
            q.memset(h16, 0, (size_t) (R * FF) * 2).wait();
            auto cols = [&](ggml_type t, int64_t n) { return k::iq_mmq_type_dup((int) t) ? 2 * n : n; };
            const double us8 = time_us(q, [&] {
                if (one) {
                    k::iq_mmq_grouped(q, (int) c.gate, wg, gb + ub, 2 * FF, cols(c.gate, N), G, xg, R, db_, r, di, gu,
                                      2 * FF);
                } else {
                    k::iq_mmq_grouped(q, (int) c.gate, wg, gb + ub, FF, cols(c.gate, N), G, xg, R, db_, r, di, gu, 2 * FF);
                    k::iq_mmq_grouped(q, (int) c.up, wg + gb, gb + ub, FF, cols(c.up, N), G, xu, R, db_, r, di, gu + FF,
                                      2 * FF);
                }
                k::iq_mmq_grouped(q, (int) c.down, wd, db, N, cols(c.down, FF), G, xh, R, db_, r, di, dn, N);
            });
            const double us16 = time_us(q, [&] {
                for (int e = 0; e < G; ++e) {
                    const uint8_t* b = wg + (size_t) e * (gb + ub);
                    uint16_t* o = dq_gu + (size_t) e * (size_t) (2 * FF * N);
                    k::dequant_f16((int) c.gate, b, 0, FF, N, o, s);
                    k::dequant_f16((int) c.up, b + gb, 0, FF, N, o + FF * N, s);
                    k::dequant_f16((int) c.down, wd + (size_t) e * db, 0, N, FF, dq_d + (size_t) e * (size_t) (N * FF), s);
                }
                k::xmx_gemm_grouped(x16, dq_gu, 2 * FF * N, gu, db_, G, r, 2 * FF, N, s);
                k::xmx_gemm_grouped(h16, dq_d, N * FF, dn, db_, G, r, N, FF, s);
            });
            std::printf("  %-14s %3lld rows an expert: int8 %8.1f us  FP16 %8.1f us  %s\n", c.name, (long long) r, us8,
                        us16, us8 < us16 ? "int8" : "FP16");
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc > 1 && std::string(argv[1]) == "--bench") return bench();
    sycl::queue& q = strata::core::Runtime::get().compute();
    const bool matrix = strata::kernels::iq_mmq_usable(q), dp4a = strata::kernels::iq_mmq_dp4a_usable(q);
    std::printf("iq_mmq_parity: the int8 expert products %s\n",
                matrix ? "on the matrix engines" : dp4a ? "by dp4a" : "(neither form runs here: skipped)");
    if (!matrix && !dp4a) return 77;
    const std::vector<int32_t> three = {5, 40, 17}, one = {70};
    int bad = 0;
    for (const auto& c : {three, one}) {
        bad += check(GGML_TYPE_Q8_0, 128, 160, c);
        bad += check(GGML_TYPE_IQ3_XXS, 128, 512, c);
        bad += check(GGML_TYPE_IQ2_XS, 128, 512, c);
        bad += check(GGML_TYPE_Q4_K, 128, 512, c);
    }
    std::printf(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "iq_mmq_parity: %s\n", e.what());
    return 1;
}
