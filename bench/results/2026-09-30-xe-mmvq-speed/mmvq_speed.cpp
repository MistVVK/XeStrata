// MMVQ speed on the B70 at the IQ2_XS model's shapes.  Host-timed loops of back-to-back launches on the in-order
// compute queue (launch cost included, as decode pays it), five repeats, median reported.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace strata;
using clk = std::chrono::steady_clock;

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main() {
    auto& q = core::Runtime::get().compute();
    void* s = &q;
    // The bandwidth reference is bandwidth_dp4a.cpp: it reads incompressible data, which a constant fill is not.
    struct Shape { const char* what; int type; int n_in, n_out; };
    const Shape shapes[] = {
        {"output head        IQ4_XS", 23, 2560, 248320},
        {"attn_qkv           IQ4_XS", 23, 2560, 10240},
        {"ssm_out            IQ4_XS", 23, 6144, 2560},
        {"attn_gate          IQ3_S ", 21, 2560, 6144},
        {"shexp gate         IQ3_S ", 21, 2560, 640},
        {"shexp up           IQ4_XS", 23, 2560, 640},
        {"shexp down         IQ4_NL", 20, 640, 2560},
        {"expert gate        IQ2_S ", 22, 2560, 640},
        {"expert gate        IQ2_XXS", 16, 2560, 640},
        {"expert gate        IQ1_M ", 29, 2560, 640},
        {"expert down        Q2_0  ", 42, 640, 2560},
    };
    std::mt19937 rng(3);
    std::printf("%-27s %5s %6s %10s %10s %9s\n", "shape", "ncols", "MB", "us/call", "GB/s", "loops");
    for (const Shape& sh : shapes) {
        const size_t wb = kernels::native_mmvq_weight_bytes(sh.type, sh.n_in, sh.n_out);
        std::vector<uint8_t> w(wb);
        for (auto& b : w) b = (uint8_t) rng();
        // random weight bytes: incompressible, so the GPU's memory compression cannot inflate the rate
        auto* dw = sycl::malloc_device<uint8_t>(wb, q);
        q.memcpy(dw, w.data(), wb).wait();
        std::vector<float> x(8 * (size_t) sh.n_in, 0.5f);
        auto* dx = sycl::malloc_device<float>(x.size(), q);
        q.memcpy(dx, x.data(), x.size() * 4).wait();
        void* xq = sycl::malloc_device(kernels::native_q8_1_bytes(sh.n_in, 8), q);
        kernels::native_quantize_q8_1(dx, xq, sh.n_in, 8, s);
        auto* dy = sycl::malloc_device<float>(8 * (size_t) sh.n_out, q);
        for (int nc : {1, 4}) {
            const int loops = wb > (64u << 20) ? 20 : 400;
            for (int i = 0; i < 10; ++i) kernels::native_mmvq(sh.type, dw, xq, dy, sh.n_in, sh.n_out, nc, s);
            q.wait();
            std::vector<double> us;
            for (int rep = 0; rep < 5; ++rep) {
                const auto t0 = clk::now();
                for (int i = 0; i < loops; ++i) kernels::native_mmvq(sh.type, dw, xq, dy, sh.n_in, sh.n_out, nc, s);
                q.wait();
                us.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count() / loops);
            }
            const double m = median(us);
            std::printf("%-27s %5d %6.2f %10.1f %10.1f %9d\n", sh.what, nc, wb / 1e6, m, wb / m / 1e3, loops);
        }
        sycl::free(dw, q); sycl::free(dx, q); sycl::free(xq, q); sycl::free(dy, q);
    }
    return 0;
}
