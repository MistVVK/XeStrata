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
    bool has_mpp = false;
    int lib = cx.compile_try({dir + "/hy4.metal", dir + "/hy4_flash.metal", dir + "/hy4_mpp.metal", dir + "/hy4_mppflash.metal"}, "#define NO_GGML_COMMON 1\n", false, &why);
    if (lib >= 0) {
        has_mpp = true;
    } else {
        fprintf(stderr, "[attb] mppflash compile skipped (%s), using simdgroup_matrix only\n", why.substr(0, 200).c_str());
        lib = cx.compile_try(files, "#define NO_GGML_COMMON 1\n", false, &why);
        if (lib < 0) { printf("compile failed: %s\n", why.c_str()); return 1; }
    }
    std::vector<std::string> kernels;
    for (int i = 2; i < argc; ++i) kernels.push_back(argv[i]);
    if (kernels.empty()) {
        kernels = {"mla_attend", "mla_flash", "mla_flash_sg16"};
        if (has_mpp) {
            kernels.push_back("mla_flash_mpp");
            kernels.push_back("mla_blk_mpp");
        }
    }
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
            if (kn == "mla_blk_mpp" && cs.mode == 2) continue;
            std::vector<mtl::Dispatch> ds;
            mtl::Pipeline p, p_kv, p_mm, p_sm, p_f2h;
            mtl::Buffer bK, bVt, bS, bP, bOf;
            if (kn == "mla_blk_mpp") {
                p_kv = cx.pipeline(lib, "mla_blk_kv");
                p_mm = cx.pipeline(lib, "mpp_mm_128_64_4");
                p_sm = cx.pipeline(lib, "mla_blk_softmax");
                p_f2h = cx.pipeline(lib, "f2h");
                const int BQ = 64;
                const int max_nk = ((cs.mode == 1 ? (512 + BQ) : (a.p0 + cs.M)) + 63) / 64 * 64;
                bK = cx.alloc((size_t) max_nk * 576 * 2);
                bVt = cx.alloc((size_t) 512 * max_nk * 2);
                bS = cx.alloc((size_t) BQ * cs.NH * max_nk * 4);
                bP = cx.alloc((size_t) BQ * cs.NH * max_nk * 2);
                bOf = cx.alloc((size_t) BQ * cs.NH * 512 * 4);
                struct MmArgs { int M, N, K; };
                for (int t0 = 0; t0 < cs.M; t0 += BQ) {
                    const int nbq = std::min(BQ, cs.M - t0);
                    const int p0_blk = a.p0 + t0;
                    const int k_lo = cs.mode == 1 ? std::max(0, p0_blk - 512 + 1) : 0;
                    const int k_hi = p0_blk + nbq - 1;
                    const int nk_b = k_hi - k_lo + 1;
                    const int nk_pad = ((nk_b + 63) / 64) * 64;
                    const int rows = nbq * cs.NH;
                    mtl::Dispatch d0; d0.p = &p_kv; d0.grid[0] = nk_pad; d0.tg[0] = 256;
                    d0.args = {mtl::Arg::B(bc), mtl::Arg::B(bK), mtl::Arg::B(bVt), mtl::Arg::V<int32_t>(k_lo), mtl::Arg::V<int32_t>(nk_b), mtl::Arg::V<int32_t>(nk_pad), mtl::Arg::V<int32_t>(a.ring)};
                    ds.push_back(std::move(d0));
                    MmArgs m1{rows, nk_pad, 576};
                    mtl::Dispatch d1; d1.p = &p_mm; d1.grid[0] = (nk_pad + 63) / 64; d1.grid[1] = (rows + 127) / 128; d1.tg[0] = 128;
                    d1.args = {mtl::Arg::B(bq, (size_t) t0 * cs.NH * 576 * 2), mtl::Arg::B(bK), mtl::Arg::B(bS), mtl::Arg::V<MmArgs>(m1)};
                    ds.push_back(std::move(d1));
                    mtl::Dispatch d2; d2.p = &p_sm; d2.grid[0] = rows; d2.tg[0] = 256;
                    d2.args = {mtl::Arg::B(bS), mtl::Arg::B(bs), mtl::Arg::B(bP), mtl::Arg::V<int32_t>(cs.NH), mtl::Arg::V<int32_t>(p0_blk), mtl::Arg::V<int32_t>(k_lo),
                               mtl::Arg::V<int32_t>(nk_pad), mtl::Arg::V<int32_t>(512), mtl::Arg::V<int32_t>(cs.mode), mtl::Arg::V<float>(0.0625f)};
                    ds.push_back(std::move(d2));
                    MmArgs m2{rows, 512, nk_pad};
                    mtl::Dispatch d3; d3.p = &p_mm; d3.grid[0] = 512 / 64; d3.grid[1] = (rows + 127) / 128; d3.tg[0] = 128;
                    d3.args = {mtl::Arg::B(bP), mtl::Arg::B(bVt), mtl::Arg::B(bOf), mtl::Arg::V<MmArgs>(m2)};
                    ds.push_back(std::move(d3));
                    uint32_t no = (uint32_t) rows * 512;
                    mtl::Dispatch d4; d4.p = &p_f2h; d4.grid[0] = (no + 255) / 256; d4.tg[0] = 256;
                    d4.args = {mtl::Arg::B(bOf), mtl::Arg::B(bo, (size_t) t0 * cs.NH * 512 * 2), mtl::Arg::V<uint32_t>(no)};
                    ds.push_back(std::move(d4));
                }
            } else {
                p = cx.pipeline(lib, kn);
                mtl::Dispatch d;
                d.p = &p;
                d.args = {mtl::Arg::B(bq), mtl::Arg::B(bc), mtl::Arg::B(bs), mtl::Arg::B(bo), mtl::Arg::V<MlaArgs>(a), mtl::Arg::B(bk), mtl::Arg::B(bn)};
                const bool mpp = kn == "mla_flash_mpp";
                const bool sg16 = kn == "mla_flash_sg16";
                const bool flash = kn.find("flash") != std::string::npos;
                d.grid[0] = (mpp || sg16) ? cs.NH / 16 : (flash ? cs.NH / 32 : cs.NH);
                d.grid[1] = cs.M;
                d.grid[2] = 1;
                d.tg[0] = mpp ? 128 : 256;
                ds.push_back(std::move(d));
            }
            cx.run(ds, 1, false, true);
            std::vector<double> ts;
            for (int r = 0; r < 5; ++r) ts.push_back(cx.run(ds, 1, false, true));
            std::sort(ts.begin(), ts.end());
            if (bK.ptr) { cx.free(bK); cx.free(bVt); cx.free(bS); cx.free(bP); cx.free(bOf); }
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
