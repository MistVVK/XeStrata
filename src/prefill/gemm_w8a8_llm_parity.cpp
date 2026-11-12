// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/gemm_w8a8_llm_parity.cpp - Gemm::native_llm's W8A8 form (the LLMs' dense prompt products) against the
// same parts put together by hand (kernels::rows_i8 / rows_i8_blocks, Gemm::w8a8's int32 sums, kernels::i8_scale):
// the same bits, and the FP16 product (native) within the int8 rows' rounding of it.  The shapes the form does not
// take (a narrow product, few rows, K not in whole rotation groups) stay native, bit for bit, with
// XESTRATA_LLM_W8A8=1 fixing the form on (a MoE model's dense=false too).  GPU, synthetic, no model; skipped (77)
// where the GPU takes no int8 product.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/rows_i8.hpp"
#include "strata/prefill/gemm.hpp"
#include "../kernels/parity_device.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// A Q8_0 / Q4_K weight's blocks and an FP16 activation's rows, deterministic
struct Fixture {
    int type;
    int64_t T, N, K;
    std::vector<uint8_t> w;
    std::vector<uint16_t> x;
    Fixture(int type_, int64_t T_, int64_t N_, int64_t K_, std::mt19937& rng)
        : type(type_), T(T_), N(N_), K(K_) {
        const size_t blk = type == 8 ? 34 : 144, per = type == 8 ? 32 : 256;
        w.resize((size_t) (N * K) / per * blk);
        std::uniform_int_distribution<int> byte(0, 255);
        for (auto& b : w) b = (uint8_t) byte(rng);
        x.resize((size_t) (T * K));
        std::uniform_int_distribution<int> v(-1000, 1000);
        for (auto& h : x) h = strata::kernels::f16_from_f32((float) v(rng) / 256.f);
    }
};

int check(const Fixture& f, strata::prefill::Gemm& g, bool take) {
    using namespace strata;
    void* s = parity::stream();
    parity::Device dev;
    const auto* dw = dev.upload(f.w);
    const auto* dx = dev.upload(f.x);
    auto* dy = dev.alloc<float>((size_t) (f.T * f.N));
    auto* dr = dev.alloc<float>((size_t) (f.T * f.N));
    auto* dn = dev.alloc<float>((size_t) (f.T * f.N));
    g.native_llm(dx, f.type, dw, dy, f.T, f.N, f.K, /*dense=*/true);
    g.native(dx, f.type, dw, dn, f.T, f.N, f.K);
    if (take) {
        // the form by hand: the weight's int8 rows, the activations', the int32 sums, the scales back
        auto* w8 = dev.alloc<int8_t>((size_t) (f.N * f.K));
        auto* sw = dev.alloc<float>((size_t) f.N);
        auto* x8 = dev.alloc<int8_t>((size_t) (f.T * f.K));
        auto* sx = dev.alloc<float>((size_t) f.T);
        if (!kernels::rows_i8_blocks(f.type, dw, f.N, f.K, w8, sw, s)) {
            auto* w16 = dev.alloc<uint16_t>((size_t) (f.N * f.K));
            kernels::dequant_f16(f.type, dw, 0, f.N, f.K, w16, s);
            kernels::rows_i8(w16, false, f.N, f.K, w8, sw, nullptr, s);
        }
        kernels::rows_i8(dx, false, f.T, f.K, x8, sx, nullptr, s);
        prefill::Gemm::w8a8(x8, sx, w8, sw, nullptr, dr, f.T, f.N, f.K, f.N, false, s);
        kernels::i8_scale(dr, f.T, f.N, f.N, sx, sw, nullptr, nullptr, s);
    }
    parity::sync();
    std::vector<float> y((size_t) (f.T * f.N)), r((size_t) (f.T * f.N)), n((size_t) (f.T * f.N));
    parity::copy_out(y.data(), dy, y.size());
    parity::copy_out(n.data(), dn, n.size());
    if (take) parity::copy_out(r.data(), dr, r.size());
    int bad = 0;
    double worst = 0, scale = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        // the bits: a random block's scale may be a NaN, and NaN != NaN
        if (take && std::bit_cast<uint32_t>(y[i]) != std::bit_cast<uint32_t>(r[i])) ++bad;
        if (!take && std::bit_cast<uint32_t>(y[i]) != std::bit_cast<uint32_t>(n[i])) ++bad;
        worst = std::max(worst, (double) std::fabs(y[i] - n[i]));
        scale = std::max(scale, (double) std::fabs(n[i]));
    }
    std::printf("  type %d T %lld N %lld K %lld: %s, %d differing, worst |W8A8 - native| %.3g (largest |native| %.3g)\n",
                f.type, (long long) f.T, (long long) f.N, (long long) f.K,
                take ? "the parts by hand" : "native kept", bad, worst, scale);
    if (bad) return 1;
    if (take && scale > 0 && worst / scale > 0.1) {
        std::printf("  the int8 rows' rounding too large\n");
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
#if defined(_WIN32)
    _putenv_s("XESTRATA_LLM_W8A8", "1");
#else
    setenv("XESTRATA_LLM_W8A8", "1", 1);
#endif
    void* s = strata::parity::stream();
    if (!strata::prefill::Gemm::i8_usable(s)) {
        std::printf("no int8 product on this GPU: skipped\n");
        return 77;
    }
    // XESTRATA_LLM_W8A8=1 takes the form for a MoE model too (dense=false), without the timing
    if (!strata::prefill::Gemm::llm_w8a8(s, /*dense=*/false)) {
        std::printf("XESTRATA_LLM_W8A8=1 did not take the form for a MoE model\n");
        return 1;
    }
    strata::prefill::Gemm g;
    std::string err;
    if (!g.init(s, 32ll << 20, err)) {
        std::printf("gemm init: %s\n", err.c_str());
        return 1;
    }
    std::mt19937 rng(7);  // NOLINT(bugprone-random-generator-seed): reproducible inputs.
    int rc = 0;
    for (const int type : {8, 12}) {
        rc |= check(Fixture(type, 2048, 2048, 1024, rng), g, true);    // the form taken
        rc |= check(Fixture(type, 2048, 512, 1024, rng), g, false);    // a narrow product: native
        rc |= check(Fixture(type, 512, 2048, 1024, rng), g, false);    // few rows: native
    }
    rc |= check(Fixture(8, 2048, 2048, 1152, rng), g, false);          // K not in whole rotation groups: native
    std::printf(rc ? "FAILED\n" : "OK\n");
    return rc;
}
