// sy_kbench.cpp - timing of the trunk GEMV/GEMM on the Arc A750: raw layout (linear_h) and RP layout (linear_auto), GB/s of weight bytes per call at M = 1, 8, 64, 512.
// build: b_sycl.sh <tag> sy_kbench sycl/sy_kbench.cpp
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "sy_gemm_xmx.h"
#include "sy_linear.h"
#include "sy_rp.h"

using namespace hy4s;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    printf("device %s\n", Q().get_device().get_info<sycl::info::device::name>().c_str());
    struct Case { int fmt, N, K; const char* what; };
    std::vector<Case> cases = {{FMT_S4, 16384, 3072, "q_b / linear_gate-like [16384,3072]"}, {FMT_S4, 3072, 8192, "o_proj-like [3072,8192]"}, {FMT_S4, 13824, 3072, "dense gate_proj [13824,3072]"},
                               {FMT_S8, 120832, 3072, "lm_head S8 [120832,3072]"}, {FMT_S4, 3072, 3072, "[3072,3072] S4"}, {FMT_S4, 1024, 3072, "q_a-like [1024,3072]"}};
    std::mt19937 rng(1);
    for (auto& cs : cases) {
        const size_t wb = weight_bytes(cs.fmt, cs.N, cs.K);
        std::vector<uint8_t> raw(wb);
        for (auto& b : raw) b = (uint8_t)rng();
        std::vector<uint8_t> rp(rp_bytes(cs.fmt, cs.N, cs.K));
        rp_repack_host(cs.fmt, raw.data(), rp.data(), cs.N, cs.K);
        uint8_t* w = dalloc<uint8_t>(wb);
        uint8_t* wr = dalloc<uint8_t>(rp.size());
        h2d(w, raw.data(), wb);
        h2d(wr, rp.data(), rp.size());
        wait();
        for (int M : {1, 8, 64, 512}) {
            half* x = dalloc<half>((size_t)M * cs.K);
            float* y = dalloc<float>((size_t)M * cs.N);
            dzero(x, (size_t)M * cs.K * 2);
            for (int mode = 0; mode < 2; ++mode) {
                if (mode == 0 && M > 8) continue;  // the simple kernel at big M is known to be slow
                auto run = [&]() { if (mode == 0) linear_h(cs.fmt, w, cs.N, cs.K, x, y, M); else linear_auto(cs.fmt, wr, cs.N, cs.K, x, y, M, true); };
                run();
                wait();
                const int reps = M >= 64 ? 5 : 30;
                const double t0 = now();
                for (int r = 0; r < reps; ++r) run();
                wait();
                const double dt = (now() - t0) / reps;
                printf("%-36s M=%-4d %s %8.3f ms  %7.1f GB/s  %8.1f GFLOP/s\n", cs.what, M, mode ? "RP    " : "raw   ", dt * 1e3, wb / dt / 1e9, 2.0 * M * cs.N * cs.K / dt / 1e9);
            }
            dfree(x);
            dfree(y);
        }
        dfree(w);
        dfree(wr);
    }
    return 0;
}
