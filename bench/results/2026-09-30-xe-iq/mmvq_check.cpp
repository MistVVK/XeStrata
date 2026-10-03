// Scratch check: native MMVQ for the formats iq_parity does not list, and the multi-column contract (not committed).
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/core/runtime.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace strata;
int main() {
    auto& q = core::Runtime::get().compute();
    void* s = &q;
    std::mt19937 rng(5);
    struct T { const char* name; int type, qk, bytes; void (*dq)(const uint8_t*, float*); std::vector<int> half_offsets; };
    const T types[] = {{"Q6_K", 14, 256, 210, dequantize_q6_K, {208}}, {"Q8_0", 8, 32, 34, dequantize_q8_0, {0}},
                       {"Q4_K", 12, 256, 144, dequantize_q4_K, {0, 2}}, {"Q5_K", 13, 256, 176, dequantize_q5_K, {0, 2}},
                       {"Q4_0", 2, 32, 18, dequantize_q4_0, {0}}, {"Q5_0", 6, 32, 22, dequantize_q5_0, {0}},
                       {"Q2_0", 42, 64, 18, dequantize_q2_0, {0}}, {"IQ4_XS", 23, 256, 136, dequantize_iq4_xs, {0}}};
    int fails = 0;
    for (const T& t : types) for (int cols : {512, 4096}) {
        const int rows = 48, nb = cols / t.qk, ncmax = 8;
        std::vector<uint8_t> w((size_t) rows * nb * t.bytes);
        for (auto& b : w) b = (uint8_t) rng();
        std::uniform_real_distribution<float> ud(0.001f, 0.05f);
        for (size_t blk = 0; blk < (size_t) rows * nb; ++blk)
            for (int off : t.half_offsets) { uint16_t h = sycl::bit_cast<uint16_t>(sycl::half(ud(rng))); std::memcpy(&w[blk * t.bytes + off], &h, 2); }
        std::vector<float> ref((size_t) rows * cols);
        for (size_t blk = 0; blk < (size_t) rows * nb; ++blk) t.dq(&w[blk * t.bytes], &ref[blk * t.qk]);
        std::normal_distribution<float> nd(0, 1);
        std::vector<float> x((size_t) ncmax * cols); for (auto& v : x) v = nd(rng);
        auto* dw = sycl::malloc_device<uint8_t>(w.size(), q); auto* dx = sycl::malloc_device<float>(x.size(), q);
        void* xq = sycl::malloc_device(kernels::native_q8_1_bytes(cols, ncmax), q);
        auto* dy = sycl::malloc_device<float>((size_t) ncmax * rows, q);
        q.memcpy(dw, w.data(), w.size()).wait(); q.memcpy(dx, x.data(), x.size() * 4).wait();
        kernels::native_quantize_q8_1(dx, xq, cols, ncmax, s);
        // single-column calls, one per column
        std::vector<float> single((size_t) ncmax * rows);
        for (int c = 0; c < ncmax; ++c) {
            kernels::native_mmvq(t.type, dw, (const uint8_t*) xq + kernels::native_q8_1_bytes(cols, 1) * c, dy, cols, rows, 1, s);
            q.wait(); q.memcpy(&single[(size_t) c * rows], dy, rows * 4).wait();
        }
        double num = 0, den = 0;
        for (int c = 0; c < ncmax; ++c) for (int r = 0; r < rows; ++r) {
            double acc = 0; for (int k = 0; k < cols; ++k) acc += (double) ref[(size_t) r * cols + k] * x[(size_t) c * cols + k];
            num += std::fabs(single[(size_t) c * rows + r] - acc); den += std::fabs(acc);
        }
        size_t exact_diff = 0; double up_num = 0, up_den = 0;
        for (int nc = 2; nc <= ncmax; ++nc) for (bool exact : {true, false}) {
            kernels::native_mmvq_set_multi_exact(exact);
            kernels::native_mmvq(t.type, dw, xq, dy, cols, rows, nc, s); q.wait();
            std::vector<float> y((size_t) nc * rows); q.memcpy(y.data(), dy, y.size() * 4).wait();
            for (size_t i = 0; i < y.size(); ++i) {
                if (exact) exact_diff += std::memcmp(&y[i], &single[i], 4) != 0;
                else { up_num += std::fabs(y[i] - single[i]); up_den += std::fabs(single[i]); }
            }
        }
        kernels::native_mmvq_set_multi_exact(true);
        const double rel = num / (den + 1e-30), up_rel = up_num / (up_den + 1e-30);
        const bool ok = rel < 2e-2 && exact_diff == 0 && up_rel < 1e-4;
        std::printf("%-7s K=%4d  vs FP64 rel %.2e   ncols 2-8 exact-layout bit diffs %zu   upstream-layout rel L1 %.1e  %s\n",
                    t.name, cols, rel, exact_diff, up_rel, ok ? "ok" : "FAIL");
        fails += !ok;
        sycl::free(dw, q); sycl::free(dx, q); sycl::free(xq, q); sycl::free(dy, q);
    }
    return fails;
}
