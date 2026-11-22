// SPDX-License-Identifier: MIT
// tests/core/dot_x4_test.cpp - bit-exact parity tests for 4-row shared-activation VPDPBUSD dot kernels
// (dot_q4_0_q8_0_x4, dot_mxfp4_q8_0_x4, dot_q4_K_q8_K_x4, dot_iq4_xs_q8_K_x4) against 1-row Arith::Avx2 kernels.

#include "strata/quant/dot.hpp"
#include "strata/quant/types.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::quant;

void verify_x4_type(Type wt, Type at, int n) {
    const TypeInfo* wi = strata::quant::type_info(wt);
    const TypeInfo* ai = strata::quant::type_info(at);
    assert(wi != nullptr && ai != nullptr);

    const DotKernel* k1 = dot_kernel(wt, Arith::Avx2);
    const DotX4Kernel* k4 = dot_x4_kernel(wt, Arith::Avx2);
    if (k4 == nullptr || k4->fn == nullptr) {
        std::printf("dot_x4_test: %s x4 skipped (CPU lacks AVX-512 VNNI)\n", wi->name);
        return;
    }
    assert(k1 != nullptr && k1->fn != nullptr);

    const size_t w_row = row_bytes((int) wt, n);
    const size_t a_row = row_bytes((int) at, n);
    std::vector<uint8_t> w(4 * w_row);
    std::vector<uint8_t> a(a_row);

    std::mt19937 rng(0x57A7A000u ^ static_cast<uint32_t>(wt));
    for (int trial = 0; trial < 16; ++trial) {
        for (auto& b : w) b = static_cast<uint8_t>(rng() & 0xFFu);
        for (auto& b : a) b = static_cast<uint8_t>(rng() & 0xFFu);

        auto make_small_half = [&](uint8_t* p) {
            // Valid normal fp16 in +/- [0.0078, 2.0) (exponent 8..15 -> bits 8..15 << 10)
            const uint16_t sign = static_cast<uint16_t>((rng() & 1u) << 15);
            const uint16_t exp  = static_cast<uint16_t>((8u + (rng() % 8u)) << 10);
            const uint16_t man  = static_cast<uint16_t>(rng() & 0x03FFu);
            const uint16_t h = static_cast<uint16_t>(sign | exp | man);
            std::memcpy(p, &h, 2);
        };
        const int nb_a = n / ai->blck;
        const size_t bs_a = ai->bytes;
        for (int i = 0; i < nb_a; ++i) {
            uint8_t* bp = a.data() + i * bs_a;
            if (at == Type::Q8_0) {
                make_small_half(bp);
                // Real quantize_q8_0 outputs symmetric int8 in [-127, 127] (never -128).
                int8_t* qs = reinterpret_cast<int8_t*>(bp + 2);
                for (int k = 0; k < 32; ++k) qs[k] = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
            } else if (at == Type::Q8_K) {
                float d = static_cast<float>(static_cast<int>(rng() % 200) - 100) * 0.01f;
                std::memcpy(bp, &d, 4);
                int8_t* qs = reinterpret_cast<int8_t*>(bp + 4);
                int16_t* bsums = reinterpret_cast<int16_t*>(bp + 4 + 256);
                for (int k = 0; k < 256; ++k) qs[k] = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
                for (int g = 0; g < 16; ++g) {
                    int s = 0;
                    for (int k = 0; k < 16; ++k) s += qs[g * 16 + k];
                    bsums[g] = static_cast<int16_t>(s);
                }
            }
        }
        const int nb_w = n / wi->blck;
        const size_t bs_w = wi->bytes;
        for (int r = 0; r < 4; ++r) {
            uint8_t* rowp = w.data() + r * w_row;
            for (int i = 0; i < nb_w; ++i) {
                uint8_t* bp = rowp + i * bs_w;
                if (wt == Type::Q4_0 || wt == Type::IQ4_XS) {
                    make_small_half(bp);
                } else if (wt == Type::Q4_K) {
                    make_small_half(bp);
                    make_small_half(bp + 2);
                } else if (wt == Type::MXFP4) {
                    bp[0] = static_cast<uint8_t>(120 + (bp[0] % 16));
                }
            }
        }

        float ref[4] = {};
        for (int r = 0; r < 4; ++r) {
            k1->fn(n, &ref[r], w.data() + r * w_row, a.data());
        }
        float got[4] = {};
        k4->fn(n, got, w.data(), w_row, a.data());

        for (int r = 0; r < 4; ++r) {
            const float diff = std::fabs(got[r] - ref[r]);
            const float tol = std::max(1e-5f, 1e-5f * std::fabs(ref[r]));
            if (!(diff <= tol) || std::isnan(got[r])) {
                std::fprintf(stderr, "dot_x4_test FAIL: %s (%s vs %s) trial=%d row=%d ref=%.8g got=%.8g diff=%.8g\n",
                             wi->name, k4->name, k1->name, trial, r, ref[r], got[r], diff);
                std::abort();
            }
        }
    }
    std::printf("dot_x4_test: %s (%s) vs %s PASS (n=%d, 16x4 rows)\n", wi->name, k4->name, k1->name, n);
}

}  // namespace

int main() {
    verify_x4_type(Type::Q4_0,   Type::Q8_0, 2048);
    verify_x4_type(Type::MXFP4,  Type::Q8_0, 2048);
    verify_x4_type(Type::Q4_K,   Type::Q8_K, 2048);
    verify_x4_type(Type::IQ4_XS, Type::Q8_K, 2048);
    std::printf("dot_x4_test: ALL SUITES PASSED\n");
    return 0;
}
