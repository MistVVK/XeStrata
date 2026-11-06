// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/i8_gemm_parity.cpp - kernels::i8_gemm_bf16 (CUTLASS's kernels on NVIDIA GPUs, comfy-kitchen's on AMD
// GPUs with WMMA) against the library's int8 product (prefill::Gemm::i8, int32 sums) scaled on the host (GPU,
// synthetic, no model).
//
//   The products the image models make (MOSS-Image's 3072 and 16384, Mage-Flow's 12288, Anima's 2048 and 8192,
//   SeedVR2's 2560 and 10240) at rows that take each kernel and its edges (1024, 2047, 2048, a tile's part over:
//   4115), and one of two contexts' rows (8230); with a bias and without.  Every value is checked: within BF16's
//   rounding of the scaled sum (2^-8 of it, and 1%), so a tile, a row's or a column's scale out of place is far past
//   it.
//   Skipped (77) where the GPU takes no such kernel or the library has no int8 product.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/i8_gemm.hpp"
#include "strata/prefill/gemm.hpp"
#include "../parity_device.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct Case {
    int64_t T, N, K;
    bool bias;
};

// The values out of BF16's rounding of the reference (the worst ratio of error to bound kept in `worst`).
int64_t check(const Case& c, std::mt19937& rng, double& worst) {
    using namespace strata;
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> sc(1e-4f, 2e-2f), bv(-4.f, 4.f);
    std::vector<int8_t> x((size_t) (c.T * c.K)), w((size_t) (c.N * c.K));
    for (auto& v : x) v = (int8_t) q(rng);
    for (auto& v : w) v = (int8_t) q(rng);
    std::vector<float> sx((size_t) c.T), sw((size_t) c.N), b((size_t) c.N);
    for (auto& v : sx) v = sc(rng);
    for (auto& v : sw) v = sc(rng);
    for (auto& v : b) v = bv(rng);
    parity::Device dev;
    const int8_t* dx = dev.upload(x);
    const int8_t* dw = dev.upload(w);
    const float* dsx = dev.upload(sx);
    const float* dsw = dev.upload(sw);
    const float* db = c.bias ? dev.upload(b) : nullptr;
    auto* dy = dev.alloc<uint16_t>((size_t) (c.T * c.N));
    auto* di = dev.alloc<int32_t>((size_t) (c.T * c.N));
    void* s = parity::stream();
    if (!kernels::i8_gemm_bf16(dx, dsx, dw, dsw, db, dy, c.T, c.N, c.K, 0, s)) {
        std::printf("  %lld x %lld x %lld: no kernel took it\n", (long long) c.T, (long long) c.N, (long long) c.K);
        return 1;
    }
    prefill::Gemm::i8(dx, dw, di, c.T, c.N, c.K, c.N, s);
    parity::sync();
    std::vector<uint16_t> y((size_t) (c.T * c.N));
    std::vector<int32_t> sums((size_t) (c.T * c.N));
    parity::copy_out(y.data(), dy, y.size());
    parity::copy_out(sums.data(), di, sums.size());
    int64_t bad = 0;
    for (int64_t t = 0; t < c.T; ++t)
        for (int64_t n = 0; n < c.N; ++n) {
            const size_t i = (size_t) (t * c.N + n);
            const double want = (double) sums[i] * sx[(size_t) t] * sw[(size_t) n] + (c.bias ? b[(size_t) n] : 0.0);
            // half BF16's step at the bottom of the value's binade is 2^-8 of it, and the rounding reaches it (the
            // worst error is the whole bound); 1% over it for FP32's order of the epilogue's products
            const double bound = std::fabs(want) * std::ldexp(1.0, -8) * 1.01 + 1e-6;
            const double err = std::fabs((double) kernels::f32_from_bf16(y[i]) - want);
            worst = std::max(worst, err / bound);
            if (err > bound) ++bad;
        }
    return bad;
}

}  // namespace

int main() {
    using namespace strata;
    void* s = parity::stream();
    if (!kernels::i8_gemm_ok(s) || !prefill::Gemm::i8_usable(s)) {
        std::printf("i8_gemm_parity: no int8 product with scaled outputs on this GPU (skipped)\n");
        return 77;
    }
    const std::vector<Case> cases = {
        {4115, 3072, 3072, true},   {4115, 16384, 3072, false}, {4115, 3072, 16384, true},  {8230, 3072, 16384, false},
        {1024, 3072, 3072, false},  {2047, 3072, 3072, true},   {2048, 9216, 3072, false},  {4352, 12288, 3072, true},
        {4352, 3072, 12288, false}, {4115, 2048, 2048, true},   {4115, 8192, 2048, false},  {2047, 2048, 8192, true},
        {4115, 2560, 2560, false},  {4115, 10240, 2560, true},  {1024, 2560, 10240, false},
    };
    std::mt19937 rng(7);   // NOLINT(bugprone-random-generator-seed): a fixed seed keeps the inputs reproducible
    int failed = 0;
    for (const Case& c : cases) {
        double worst = 0;
        const int64_t bad = check(c, rng, worst);
        std::printf("  %lld x %lld x %lld%s: %lld out of bound, worst error %.2f of the bound\n", (long long) c.T,
                    (long long) c.N, (long long) c.K, c.bias ? " (bias)" : "", (long long) bad, worst);
        failed += bad != 0;
    }
    std::printf("i8_gemm_parity: %s\n", failed ? "FAILED" : "passed");
    return failed ? 1 : 0;
}
