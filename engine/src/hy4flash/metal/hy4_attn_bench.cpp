// hy4_attn_bench - time of the prefill attention kernels (mla_attend simple, mla_flash and variants) on random data: M queries, NH heads, keys per query by mode.
// usage: hy4_attn_bench dir_with_metal_files [kernel names...]
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "mtl_rt.h"

struct MlaArgs {
    int32_t M, NH, p0, mode, window, ring, kmax;
    float scale;
};

int main(int argc, char** argv) {
    mtl::Context cx;
    const std::string dir = argc > 1 ? argv[1] : ".";
    std::vector<std::string> files = {dir + "/hy4.metal", dir + "/hy4_flash.metal"};
    std::string why;
    int lib = cx.compile_try(files, "#define NO_GGML_COMMON 1\n", false, &why);
    if (lib < 0) { printf("compile failed: %s\n", why.c_str()); return 1; }
    std::vector<std::string> kernels;
    for (int i = 2; i < argc; ++i) kernels.push_back(argv[i]);
    if (kernels.empty()) kernels = {"mla_attend", "mla_flash", "mla_flash_q"};
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto f2h = [](float f) { __fp16 h = (__fp16) f; uint16_t u; std::memcpy(&u, &h, 2); return u; };
    struct Case { const char* name; int mode, M, NH, keys, ring; };
    const Case cases[] = {{"list 2048 keys, 32 heads", 2, 512, 32, 2048, 0}, {"window 512, 64 heads", 1, 512, 64, 512, 1024}, {"causal 1024, 32 heads", 0, 512, 32, 1024, 0}};
    for (const Case& cs : cases) {
        const int slots = std::max(cs.keys + cs.M, 4096);
        std::vector<uint16_t> qa((size_t) cs.M * cs.NH * 576), cache((size_t) slots * 576);
        for (auto& v : qa) v = f2h(nd(rng) * 0.3f);
        for (auto& v : cache) v = f2h(nd(rng) * 0.3f);
        mtl::Buffer bq = cx.alloc(qa.size() * 2), bc = cx.alloc(cache.size() * 2), bs = cx.alloc(cs.NH * 4), bo = cx.alloc((size_t) cs.M * cs.NH * 512 * 2);
        mtl::Buffer bk = cx.alloc((size_t) cs.M * 2048 * 4), bn = cx.alloc(cs.M * 4);
        std::memcpy(bq.ptr, qa.data(), qa.size() * 2);
        std::memcpy(bc.ptr, cache.data(), cache.size() * 2);
        for (int h = 0; h < cs.NH; ++h) ((float*) bs.ptr)[h] = nd(rng);
        for (int t = 0; t < cs.M; ++t) {
            ((int32_t*) bn.ptr)[t] = cs.mode == 2 ? 2048 : 0;
            for (int j = 0; j < 2048; ++j) ((int32_t*) bk.ptr)[(size_t) t * 2048 + j] = (int) (rng() % slots);
        }
        // p0 so that causal/window keys exist: positions p0..p0+M-1
        const int p0 = cs.mode == 0 ? cs.keys - cs.M : (cs.mode == 1 ? 3000 : 0);
        MlaArgs a{cs.M, cs.NH, std::max(p0, 0), cs.mode, 512, cs.ring, 2048, 0.0625f};
        printf("%s (M %d):\n", cs.name, cs.M);
        std::vector<float> ref;
        for (const auto& kn : kernels) {
            mtl::Pipeline p = cx.pipeline(lib, kn);
            mtl::Dispatch d;
            d.p = &p;
            d.args = {mtl::Arg::B(bq), mtl::Arg::B(bc), mtl::Arg::B(bs), mtl::Arg::B(bo), mtl::Arg::V<MlaArgs>(a), mtl::Arg::B(bk), mtl::Arg::B(bn)};
            const bool flash = kn.find("flash") != std::string::npos;
            d.grid[0] = flash ? cs.NH / 32 : cs.NH;
            d.grid[1] = cs.M;
            d.tg[0] = 256;
            cx.run({d});
            std::vector<double> ts;
            for (int r = 0; r < 5; ++r) ts.push_back(cx.run({d}, 1));
            std::sort(ts.begin(), ts.end());
            std::vector<float> o(1 << 16);
            for (size_t i = 0; i < o.size(); ++i) { uint16_t u = ((uint16_t*) bo.ptr)[i]; __fp16 h; std::memcpy(&h, &u, 2); o[i] = (float) h; }
            if (ref.empty()) ref = o;
            double md = 0;
            for (size_t i = 0; i < o.size(); ++i) md = std::max(md, (double) std::fabs(o[i] - ref[i]));
            const double keys = cs.mode == 2 ? 2048 : (cs.mode == 1 ? 512 : cs.keys - cs.M / 2);
            const double flop = 2.0 * cs.M * cs.NH * keys * (576 + 512);
            printf("   %-16s %9.3f ms  %6.2f TFLOP/s-equiv  (max diff vs first %.2e)\n", kn.c_str(), ts[2] * 1e3, flop / ts[2] / 1e12, md);
        }
        cx.free(bq); cx.free(bc); cx.free(bs); cx.free(bo); cx.free(bk); cx.free(bn);
    }
    return 0;
}
