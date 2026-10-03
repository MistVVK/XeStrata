// Scratch check: GPU dequant/MMVQ vs the repository's scalar dequantizers on random blocks (not committed).
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/runtime.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace strata;
int main() {
    auto& rt = core::Runtime::get();
    std::mt19937 rng(11);
    struct T { const char* name; int type, qk, bytes; void (*dq)(const uint8_t*, float*); };
    const T types[] = {{"Q2_0", 42, 64, 18, dequantize_q2_0}, {"IQ4_NL", 20, 32, 18, dequantize_iq4_nl},
                       {"IQ4_XS", 23, 256, 136, dequantize_iq4_xs}, {"Q3_K", 11, 256, 110, dequantize_q3_K}};
    int fails = 0;
    for (const T& t : types) {
        const int rows = 64, cols = 2048, nb = cols / t.qk;
        std::vector<uint8_t> w((size_t) rows * nb * t.bytes);
        for (auto& b : w) b = (uint8_t) rng();
        // a finite fp16 scale in every block's d (first 2 bytes; Q3_K keeps d last)
        std::uniform_real_distribution<float> ud(0.001f, 0.05f);
        for (size_t blk = 0; blk < (size_t) rows * nb; ++blk) {
            const uint16_t h = sycl::bit_cast<uint16_t>(sycl::half(ud(rng)));
            std::memcpy(&w[blk * t.bytes + (t.type == 11 ? t.bytes - 2 : 0)], &h, 2);
        }
        std::vector<float> ref((size_t) rows * cols);
        for (size_t blk = 0; blk < (size_t) rows * nb; ++blk) t.dq(&w[blk * t.bytes], &ref[blk * t.qk]);
        auto& q = rt.compute();
        uint8_t* dw = sycl::malloc_device<uint8_t>(w.size(), q);
        float* dq = sycl::malloc_device<float>(ref.size(), q);
        q.memcpy(dw, w.data(), w.size()).wait();
        std::vector<float> got(ref.size());
        kernels::iq_dequant_f32(t.type, dw, (int64_t) ref.size(), dq, nullptr);
        q.memcpy(got.data(), dq, got.size() * 4).wait();
        size_t diff = 0;
        for (size_t i = 0; i < ref.size(); ++i) diff += std::memcmp(&got[i], &ref[i], 4) != 0;
        double mm = -1;
        if (t.type != 11) {
            std::normal_distribution<float> nd(0, 1);
            std::vector<float> x(2 * cols);
            for (auto& v : x) v = nd(rng);
            float* dx = sycl::malloc_device<float>(x.size(), q);
            void* xq = sycl::malloc_device(2 * cols / 32 * 36, q);
            float* dy = sycl::malloc_device<float>(2 * rows, q);
            q.memcpy(dx, x.data(), x.size() * 4).wait();
            kernels::quantize_q8_1_rows(dx, 2, cols, xq, nullptr);
            kernels::iq_mmvq(t.type, dw, xq, dy, cols, rows, 2, nullptr);
            std::vector<float> y(2 * rows);
            q.memcpy(y.data(), dy, y.size() * 4).wait();
            double num = 0, den = 0;
            for (int c = 0; c < 2; ++c) for (int r = 0; r < rows; ++r) {
                double acc = 0;
                for (int k = 0; k < cols; ++k) acc += (double) ref[(size_t) r * cols + k] * x[(size_t) c * cols + k];
                num += std::fabs(y[(size_t) c * rows + r] - acc); den += std::fabs(acc);
            }
            mm = num / (den + 1e-30);
        }
        const bool ok = diff == 0 && (mm < 0 || mm < 2e-2);
        std::printf("%-7s dequant bit mismatches %zu / %zu   mmvq rel %.2e  %s\n", t.name, diff, ref.size(), mm, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    return fails;
}
