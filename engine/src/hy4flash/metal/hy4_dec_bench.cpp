// hy4_dec_bench - decode-time bandwidth of the GEMV kernels (M=1) and of the MoE pair path on real weights. usage: hy4_dec_bench hy4.metal model.strata
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>

#include "hy4_manifest.h"
#include "mtl_rt.h"
#include "strata/artifact/strata_reader.hpp"

using namespace hy4;
struct ExpertDev {
    uint64_t body_off;
    uint32_t dn_off;
    uint8_t gu_code, dn_code;
    uint16_t pad;
};

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    mtl::Context cx;
    int lib = cx.compile({argv[1]}, "#define NO_GGML_COMMON 1\n", false);
    strata::artifact::StrataFileReader rd;
    std::string err;
    if (!rd.open(argv[2], err)) { printf("open: %s\n", err.c_str()); return 1; }
    Manifest man;
    parse_manifest(rd.manifest_json(), man);
    int fd = open(argv[2], O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    uint8_t* base = (uint8_t*) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    mtl::Buffer wb = cx.wrap(base, ((size_t) st.st_size + 16383) / 16384 * 16384);
    const uint64_t s1 = rd.header().sections[1].file_offset, s4 = rd.header().sections[4].file_offset;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto f2h = [](float f) { __fp16 h = (__fp16) f; uint16_t u; std::memcpy(&u, &h, 2); return u; };
    const char* names[] = {"model.layers.3.self_attn.o_proj.weight", "model.layers.3.self_attn.linear_gate.weight", "model.layers.3.self_attn.q_b_proj.weight", "model.layers.5.mlp.shared_experts.down_proj.weight",
                           "model.layers.5.mlp.routed_expert_up_proj.weight", "lm_head.weight"};
    for (const char* nm : names) {
        const TInfo& t = man.get(nm);
        const int N = (int) t.rows(), K = (int) t.cols();
        mtl::Buffer X = cx.alloc((size_t) K * 2), Y = cx.alloc((size_t) N * 4);
        for (int i = 0; i < K; ++i) ((uint16_t*) X.ptr)[i] = f2h(nd(rng));
        std::vector<std::string> kn = {"linear_sk_" + std::to_string(t.fmt) + "_1"};
        for (const auto& k : kn) {
            mtl::Pipeline p = cx.pipeline(lib, k);
            mtl::Dispatch d;
            d.p = &p;
            d.args = {mtl::Arg::B(wb, s1 + t.off), mtl::Arg::B(X), mtl::Arg::B(Y), mtl::Arg::V<int32_t>(N), mtl::Arg::V<int32_t>(K), mtl::Arg::V<int32_t>(1)};
            d.grid[0] = (N + 7) / 8;
            d.tg[0] = 256;
            cx.run({d});
            std::vector<double> ts;
            for (int r = 0; r < 9; ++r) ts.push_back(cx.run({d}, 1));
            std::sort(ts.begin(), ts.end());
            printf("%-48s %-18s N %6d K %5d  %8.1f us  %6.1f GB/s (%.1f MB)\n", nm, k.c_str(), N, K, ts[4] * 1e6, t.bytes / ts[4] / 1e9, t.bytes / 1e6);
        }
    }
    // MoE pair path, T = 1: 16 random experts of layer 1 per iteration
    {
        const auto& dir = rd.expert_directory();
        std::vector<ExpertDev> ex(512);
        for (int i = 0; i < 512; ++i) ex[dir[i].expert_id] = ExpertDev{dir[i].body_offset, dir[i].dn_offset, dir[i].gu_code_id, dir[i].dn_code_id, 0};
        mtl::Buffer bex = cx.alloc(512 * sizeof(ExpertDev));
        std::memcpy(bex.ptr, ex.data(), 512 * sizeof(ExpertDev));
        mtl::Buffer bz = cx.alloc(1536 * 2), bi = cx.alloc(64), bw = cx.alloc(64), bh = cx.alloc(16 * 1536 * 2), by = cx.alloc(16 * 1536 * 4), bo = cx.alloc(1536 * 4);
        for (int i = 0; i < 1536; ++i) ((uint16_t*) bz.ptr)[i] = f2h(nd(rng));
        for (int i = 0; i < 16; ++i) ((float*) bw.ptr)[i] = 0.06f;
        mtl::Pipeline pa = cx.pipeline(lib, "moe_gu_pair"), pb = cx.pipeline(lib, "moe_dn_pair"), pc = cx.pipeline(lib, "moe_reduce");
        double tot = 0, bytes = 0;
        const int iters = 40;
        for (int it = 0; it < iters; ++it) {
            std::vector<int> pick;
            while (pick.size() < 16) { int e = (int) (rng() % 512); if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e); }
            for (int s = 0; s < 16; ++s) ((int32_t*) bi.ptr)[s] = pick[s];
            bytes = 0;
            for (int e : pick) bytes += dir[0].body_bytes ? 0 : 0;
            mtl::Dispatch a, b, c;
            a.p = &pa; a.args = {mtl::Arg::B(wb, s4), mtl::Arg::B(bex), mtl::Arg::B(bz), mtl::Arg::B(bi), mtl::Arg::B(bh)}; a.grid[0] = 192; a.grid[1] = 16; a.tg[0] = 256;
            b.p = &pb; b.args = {mtl::Arg::B(wb, s4), mtl::Arg::B(bex), mtl::Arg::B(bw), mtl::Arg::B(bi), mtl::Arg::B(bh), mtl::Arg::B(by)}; b.grid[0] = 192; b.grid[1] = 16; b.tg[0] = 256;
            c.p = &pc; c.args = {mtl::Arg::B(by), mtl::Arg::B(bo)}; c.grid[0] = 1; c.tg[0] = 256;
            tot += cx.run({a, b, c}, 1, false, true);
        }
        // bytes of one expert set: sum of the picked experts' bodies (approximate with the directory's mean)
        double mean = 0;
        for (int i = 0; i < 512; ++i) mean += dir[i].body_bytes;
        mean /= 512;
        printf("moe pair path (16 random experts, layer 1): %.1f us per layer, mean body %.2f MB -> %.1f GB/s\n", tot / iters * 1e6, mean / 1e6, 16 * mean / (tot / iters) / 1e9);
    }
    return 0;
}
