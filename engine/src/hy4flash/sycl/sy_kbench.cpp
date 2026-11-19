// sy_kbench.cpp - timing of linear_h (the trunk GEMV/GEMM) on the Arc A750: GB/s of weight bytes per call at M = 1, 8, 64, 512.
// build: b_sycl.sh <tag> sy_kbench sycl/sy_kbench.cpp
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "sy_linear.h"

using namespace hy4s;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    printf("device %s\n", Q().get_device().get_info<sycl::info::device::name>().c_str());
    struct Case { int fmt, N, K; const char* what; };
    std::vector<Case> cases = {{FMT_S4, 16384, 3072, "q_b / linear_gate-like [16384,3072]"}, {FMT_S4, 3072, 8192, "o_proj-like [3072,8192]"}, {FMT_S4, 13824, 3072, "dense gate_proj [13824,3072]"},
                               {FMT_S8, 120832, 3072, "lm_head S8 [120832,3072]"}, {FMT_S5, 3072, 3072, "[3072,3072] S5"}};
    std::mt19937 rng(1);
    for (auto& cs : cases) {
        const size_t wb = weight_bytes(cs.fmt, cs.N, cs.K);
        uint8_t* w = dalloc<uint8_t>(wb);
        std::vector<uint8_t> hw(1 << 20);
        for (auto& b : hw) b = (uint8_t)rng();
        for (size_t off = 0; off < wb; off += hw.size()) h2d(w + off, hw.data(), std::min(hw.size(), wb - off));
        wait();
        for (int M : {1, 8, 64, 512}) {
            half* x = dalloc<half>((size_t)M * cs.K);
            float* y = dalloc<float>((size_t)M * cs.N);
            dzero(x, (size_t)M * cs.K * 2);
            linear_h(cs.fmt, w, cs.N, cs.K, x, y, M);
            wait();
            const int reps = M >= 64 ? 3 : 20;
            const double t0 = now();
            for (int r = 0; r < reps; ++r) linear_h(cs.fmt, w, cs.N, cs.K, x, y, M);
            wait();
            const double dt = (now() - t0) / reps;
            printf("%-38s M=%-4d %8.3f ms  %7.1f GB/s weights  %7.1f GFLOP/s\n", cs.what, M, dt * 1e3, wb / dt / 1e9, 2.0 * M * cs.N * cs.K / dt / 1e9);
            dfree(x);
            dfree(y);
        }
        dfree(w);
    }
    return 0;
}
