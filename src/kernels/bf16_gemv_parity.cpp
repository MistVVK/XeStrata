// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/bf16_gemv_parity.cpp - P2.S5's test for the BF16 GEMV (bf16_gemv_fp32_mmvf: BF16 weights, FP32
// activations, the pinned CUDA single-token MMVF).
//
// TWO THINGS ARE CHECKED, and the second is the one the engine's correctness rests on:
//
//   1. the kernel against a host reference in double over the SAME bf16 weight bits and FP32 activations - so the
//      tolerance is set by f32 products and summation order and nothing else;
//   2. **THE ACTIVATION CONTRACT IS OBSERVABLE.**  The kernel reads the activation in FP32; a BF16-rounded
//      activation is a 2^-9 = 1.95e-03 relative change, the rival reading (the engine's former contract), computed
//      and required to differ, because it produces perfectly plausible output.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"

#include "parity_device.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

/// The host reference, in DOUBLE, over the same bf16 weight bits and the given activations.
void reference(const std::vector<float>& x, const std::vector<uint16_t>& w, long long n_in, long long n_out,
               std::vector<float>& y) {
    y.assign((size_t) n_out, 0.0f);
    for (long long o = 0; o < n_out; ++o) {
        double acc = 0;
        for (long long i = 0; i < n_in; ++i)
            acc += (double) x[(size_t) i] * (double) f32_from_bf16(w[(size_t) (o * n_in + i)]);
        y[(size_t) o] = (float) acc;
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: bf16_gemv_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;

    // THE REAL SHAPES, because a kernel that only works on a square toy is not a kernel this engine can use.
    // `ssm_alpha` is [2560, 48] - a wide reduction and 48 outputs - and `indexer.k_proj` is [2560, 128].
    struct Shape { long long n_in, n_out; const char* what; };
    const Shape shapes[] = {
        {2560, 48, "ssm_alpha.weight  [2560, 48]"},
        {2560, 128, "indexer.k_proj   [2560, 128]"},
        {2560, 512, "indexer.q_proj   [2560, 512]"},
        {2560, 2560, "ple_value        [2560, 2560]"},
    };

    for (const Shape& s : shapes) {
        std::mt19937 rng(4242);  // NOLINT(bugprone-random-generator-seed): a fixed seed keeps the parity inputs reproducible
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> fx((size_t) s.n_in), fw((size_t) (s.n_in * s.n_out));
        for (auto& v : fx) v = g(rng);
        // a realistic weight scale: these are projections, not embeddings
        for (auto& v : fw) v = g(rng) * 0.05f;

        // THE WEIGHT is already bf16 in the pack - the loaded bytes ARE the high 16 bits of an f32; the activation
        // stays f32.
        std::vector<uint16_t> w((size_t) (s.n_in * s.n_out));
        for (size_t i = 0; i < fw.size(); ++i) w[i] = bf16_from_f32(fw[i]);

        std::vector<float> want;
        reference(fx, w, s.n_in, s.n_out, want);

        strata::parity::Device dev;
        float* d_x = dev.upload(fx);
        uint16_t* d_w = dev.upload(w);
        float* d_y = dev.alloc<float>((size_t) s.n_out);

        std::vector<float> got((size_t) s.n_out);
        strata::kernels::bf16_gemv_fp32_mmvf(d_x, d_w, d_y, s.n_in, s.n_out, nullptr);
        strata::parity::copy_out(got.data(), d_y, got.size());

        const double r = rel_l1(want, got);
        std::printf("  %-34s mmvf %.2e\n", s.what, r);
        // f32 products of an f32 activation and a bf16 weight, summed in f32 over n_in terms: a tight bound
        if (!(r <= 1e-5)) { std::printf("    *** over 1e-5 ***\n"); ++bad; }

        // ---- 2. THE ACTIVATION CONTRACT.  The rival: the activation rounded to bf16 first.  It must be observable,
        // or check 1 above would pass against either and the contract would be untested.
        {
            std::vector<float> xb((size_t) s.n_in);
            for (size_t i = 0; i < fx.size(); ++i) xb[i] = f32_from_bf16(bf16_from_f32(fx[i]));
            std::vector<float> rival;
            reference(xb, w, s.n_in, s.n_out, rival);
            const double rv = rel_l1(rival, got);
            const bool visible = rv > 1e-4;
            std::printf("      %-30s %s (%.4f%% apart)\n", "fp32 vs bf16 activation",
                        visible ? "yes" : "*** NO ***", rv * 100);
            if (!visible) ++bad;
        }
    }

    std::printf("\nbf16_gemv: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("bf16_gemv_parity OK\n");
    return 0;
}
