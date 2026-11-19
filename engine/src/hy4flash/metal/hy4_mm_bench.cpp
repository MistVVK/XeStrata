// hy4_mm_bench - throughput of MPP matmul2d vs simdgroup_matrix for the Hy4-Flash prefill GEMM shapes (fp16 x fp16 -> fp32). usage: hy4_mm_bench hy4_mpp.metal
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "mtl_rt.h"

struct MmArgs {
    int M, N, K;
};

int main(int argc, char** argv) {
    mtl::Context cx;
    std::string why;
    int lib = cx.compile_try({argc > 1 ? argv[1] : "hy4_mpp.metal"}, "#define NO_GGML_COMMON 1\n", false, &why);
    if (lib < 0) { printf("compile failed: %s\n", why.c_str()); return 1; }
    struct Shape { int M, N, K; const char* what; };
    const Shape shapes[] = {{512, 16384, 1024, "q_b swa (512 tok)"}, {512, 16384, 3072, "gate swa"}, {512, 3072, 16384, "o_proj swa"}, {512, 3072, 1536, "shared down"}, {128, 16384, 3072, "gate swa (128 tok)"}, {2048, 16384, 3072, "gate swa (2048 tok)"}};
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const Shape& s : shapes) {
        const size_t na = (size_t) s.M * s.K, nb = (size_t) s.N * s.K, nc = (size_t) s.M * s.N;
        mtl::Buffer A = cx.alloc(na * 2), B = cx.alloc(nb * 2), C = cx.alloc(nc * 4);
        auto f2h = [](float f) { __fp16 h = (__fp16) f; uint16_t u; std::memcpy(&u, &h, 2); return u; };
        for (size_t i = 0; i < na; ++i) A.as<uint16_t>()[i] = f2h(nd(rng) * 0.1f);
        for (size_t i = 0; i < nb; ++i) B.as<uint16_t>()[i] = f2h(nd(rng) * 0.1f);
        printf("%-22s M %5d N %5d K %5d\n", s.what, s.M, s.N, s.K);
        struct Cfg { std::string name; int tm, tn, tgs; };
        std::vector<Cfg> cfgs = {{"sg_mm", 64, 64, 128}, {"mpp_mm_64_64_4", 64, 64, 128}, {"mpp_mm_128_64_4", 128, 64, 128}, {"mpp_mm_64_128_4", 64, 128, 128}, {"mpp_mm_128_128_4", 128, 128, 128},
                                 {"mpp_mm_64_32_4", 64, 32, 128}, {"mpp_mm_32_64_4", 32, 64, 128}, {"mpp_mm_32_32_2", 32, 32, 64}};
        std::vector<float> ref;
        for (auto& c : cfgs) {
            mtl::Pipeline p = cx.pipeline(lib, c.name);
            mtl::Dispatch d;
            d.p = &p;
            MmArgs a{s.M, s.N, s.K};
            d.args = {mtl::Arg::B(A), mtl::Arg::B(B), mtl::Arg::B(C), mtl::Arg::V<MmArgs>(a)};
            d.grid[0] = (s.N + c.tn - 1) / c.tn;
            d.grid[1] = (s.M + c.tm - 1) / c.tm;
            d.tg[0] = c.tgs;
            std::memset(C.ptr, 0, nc * 4);
            cx.run({d});
            // check against the first config
            if (ref.empty()) ref.assign((float*) C.ptr, (float*) C.ptr + std::min<size_t>(nc, 1 << 20));
            double maxd = 0;
            for (size_t i = 0; i < ref.size(); ++i) maxd = std::max(maxd, (double) std::fabs(((float*) C.ptr)[i] - ref[i]));
            std::vector<double> ts;
            for (int r = 0; r < 7; ++r) ts.push_back(cx.run({d}, 3) / 3);
            std::sort(ts.begin(), ts.end());
            const double t = ts[ts.size() / 2];
            printf("   %-18s %8.3f ms  %6.2f TFLOP/s  (max diff vs sg_mm %.2e)\n", c.name.c_str(), t * 1e3, 2.0 * s.M * s.N * s.K / t / 1e12, maxd);
        }
        cx.free(A); cx.free(B); cx.free(C);
    }
    return 0;
}
