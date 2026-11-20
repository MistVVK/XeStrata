// sy_gemm_test.cpp - XMX prefill GEMM vs the reference linear_h kernel (random S_K weights), plus timing at M = 512.
// build: b_sycl.sh <tag> sy_gemm_test sycl/sy_gemm_test.cpp
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include "sy_gemm_xmx.h"
#include "sy_linear.h"

using namespace hy4s;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    printf("device %s\n", Q().get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(7);
    auto rnd = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
    for (int fmt : {FMT_S2, FMT_S3, FMT_S4, FMT_S5, FMT_S6, FMT_S8}) {
        const int N = 640, K = 1536, M = 141;
        const int Mr = (M + 7) & ~7;
        const int sb = sk_sb_bytes(fmt);
        std::vector<uint8_t> w((size_t)N * (K / 256) * sb);
        for (auto& b : w) b = (uint8_t)rng();
        for (size_t i = 0; i < w.size(); i += sb) {
            const uint16_t d = sycl::bit_cast<uint16_t>((half)rnd(0.001f, 0.05f));
            w[i] = d & 255;
            w[i + 1] = d >> 8;
            for (int g = 0; g < 4; ++g) w[i + 2 + g] = (uint8_t)(100 + rng() % 150);
        }
        std::vector<half> x((size_t)Mr * K);
        for (auto& v : x) v = (half)rnd(-1.f, 1.f);
        uint8_t* dw = dalloc<uint8_t>(w.size());
        half* dx = dalloc<half>(x.size());
        float* y1 = dalloc<float>((size_t)Mr * N);
        float* y2 = dalloc<float>((size_t)Mr * N);
        h2d(dw, w.data(), w.size());
        h2d(dx, x.data(), x.size() * 2);
        linear_h(fmt, dw, N, K, dx, y1, M);
        linear_auto(fmt, dw, N, K, dx, y2, M); const bool ok = true;
        std::vector<float> a((size_t)M * N), b((size_t)M * N);
        d2h(a.data(), y1, a.size() * 4);
        d2h(b.data(), y2, b.size() * 4);
        wait();
        double num = 0, den = 0, mx = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = (double)a[i] - b[i];
            num += d * d;
            den += (double)a[i] * a[i];
            mx = std::max(mx, std::fabs(d));
        }
        printf("gemm_xmx S%d M=%d: supported %d rel L2 %.3e max abs %.3e (weights rounded to fp16: expect ~1e-3)\n", fmt, M, (int)ok, std::sqrt(num / den), mx);
    }
    // timing
    struct Case { int fmt, N, K; const char* what; };
    for (Case cs : {Case{FMT_S4, 16384, 3072, "[16384,3072] S4"}, Case{FMT_S4, 3072, 8192, "[3072,8192] S4"}, Case{FMT_S4, 13824, 3072, "[13824,3072] S4"}}) {
        const size_t wb = weight_bytes(cs.fmt, cs.N, cs.K);
        uint8_t* w = dalloc<uint8_t>(wb);
        std::vector<uint8_t> hw(1 << 20);
        for (auto& b : hw) b = (uint8_t)rng();
        for (size_t off = 0; off < wb; off += hw.size()) h2d(w + off, hw.data(), std::min(hw.size(), wb - off));
        const int M = 512;
        half* x = dalloc<half>((size_t)M * cs.K);
        float* y = dalloc<float>((size_t)M * cs.N);
        dzero(x, (size_t)M * cs.K * 2);
        wait();
        for (int mode = 0; mode < 2; ++mode) {
            auto run = [&]() { if (mode == 0) linear_h(cs.fmt, w, cs.N, cs.K, x, y, M); else gemm_xmx(cs.fmt, w, cs.N, cs.K, x, y, M); };
            run();
            wait();
            const double t0 = now();
            for (int r = 0; r < 3; ++r) run();
            wait();
            const double dt = (now() - t0) / 3;
            printf("%-20s M=512 %s: %8.2f ms  %7.1f GFLOP/s\n", cs.what, mode ? "xmx " : "simple", dt * 1e3, 2.0 * M * cs.N * cs.K / dt / 1e9);
        }
        dfree(w); dfree(x); dfree(y);
    }
    return 0;
}
