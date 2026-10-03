// Scratch check: bf16_gemv_fp32_mmvf against FP64, and the multi-row call against single-row calls (not committed).
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "parity_device.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
int main() {
    using namespace strata;
    std::mt19937 rng(9); std::normal_distribution<float> nd(0, 1);
    int fails = 0;
    for (auto [n_in, n_out] : {std::pair{2560, 48}, {2560, 512}, {10240, 320}, {320, 10240}, {62, 7}}) {
        std::vector<uint16_t> w((size_t) n_in * n_out); for (auto& v : w) v = kernels::bf16_from_f32(nd(rng) * 0.05f);
        std::vector<float> x((size_t) 8 * n_in); for (auto& v : x) v = nd(rng);
        parity::Device dev; auto* dw = dev.upload(w); auto* dx = dev.upload(x); auto* dy = dev.alloc<float>((size_t) 8 * n_out);
        std::vector<float> single((size_t) 8 * n_out);
        for (int k = 0; k < 8; ++k) {
            kernels::bf16_gemv_fp32_mmvf(dx + (size_t) k * n_in, dw, dy, n_in, n_out, parity::stream());
            parity::copy_out(&single[(size_t) k * n_out], dy, n_out);
        }
        double num = 0, den = 0;
        for (int k = 0; k < 8; ++k) for (int o = 0; o < n_out; ++o) {
            double a = 0; for (int i = 0; i < n_in; ++i) a += (double) kernels::f32_from_bf16(w[(size_t) o * n_in + i]) * x[(size_t) k * n_in + i];
            num += std::fabs(single[(size_t) k * n_out + o] - a); den += std::fabs(a);
        }
        size_t diff = 0;
        for (int nt = 2; nt <= 8; ++nt) {
            kernels::bf16_gemv_fp32_mmvf_multi(dx, n_in, dw, dy, n_out, n_in, n_out, nt, parity::stream());
            std::vector<float> y((size_t) nt * n_out); parity::copy_out(y.data(), dy, y.size());
            for (size_t i = 0; i < y.size(); ++i) diff += std::memcmp(&y[i], &single[i], 4) != 0;
        }
        const bool ok = num / den < 1e-5 && diff == 0;
        std::printf("mmvf %5d x %5d  vs FP64 rel %.2e   multi(2..8) bit diffs vs single %zu  %s\n", n_in, n_out, num / den, diff, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    return fails;
}
