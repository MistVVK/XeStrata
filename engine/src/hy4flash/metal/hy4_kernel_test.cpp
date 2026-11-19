// hy4_kernel_test - checks the linear kernels of hy4.metal (GEMV M=1/8 and the simdgroup_matrix GEMM) against a CPU double-precision dequant+dot reference on real trunk tensors.
// usage: hy4_kernel_test hy4.metal model.strata   (needs model.strata.manifest.json; trunk only is read)
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "hy4_manifest.h"
#include "mtl_rt.h"
#include "strata/artifact/strata_reader.hpp"

using namespace hy4;

static float h2f(uint16_t b) {
    uint32_t s = (b >> 15) & 1, e = (b >> 10) & 31, m = b & 1023;
    float v;
    if (e == 0) v = std::ldexp((float) m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float) (m + 1024), (int) e - 25);
    return s ? -v : v;
}
static uint16_t f2h(float f) {  // round to nearest even
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000;
    int e = (int) ((x >> 23) & 255) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) return (uint16_t) s;
    if (e >= 31) return (uint16_t) (s | 0x7c00);
    uint32_t r = (m + 0xfff + ((m >> 13) & 1)) >> 13;
    return (uint16_t) (s | ((e << 10) + r));
}
static int sbb(int f) { return f == 2 ? 70 : f == 3 ? 102 : f == 4 ? 134 : f == 5 ? 166 : f == 6 ? 198 : 262; }
static float hh(int f) { return f == 2 ? 1.5f : f == 3 ? 3.5f : f == 4 ? 7.5f : f == 5 ? 15.5f : f == 6 ? 31.5f : 0.f; }
static int qv(int f, const uint8_t* pl, int i) {
    switch (f) {
        case 2: return (pl[i >> 2] >> ((i & 3) * 2)) & 3;
        case 3: return ((pl[i >> 2] >> ((i & 3) * 2)) & 3) | (((pl[64 + (i >> 3)] >> (i & 7)) & 1) << 2);
        case 4: return (pl[i >> 1] >> ((i & 1) * 4)) & 15;
        case 5: return ((pl[i >> 1] >> ((i & 1) * 4)) & 15) | (((pl[128 + (i >> 3)] >> (i & 7)) & 1) << 4);
        case 6: return ((pl[i >> 1] >> ((i & 1) * 4)) & 15) | (((pl[128 + (i >> 2)] >> ((i & 3) * 2)) & 3) << 4);
        default: return (int8_t) pl[i];
    }
}
static double wval(const TInfo& t, const uint8_t* W, int row, int k) {
    const int N = (int) t.rows(), K = (int) t.cols();
    if (t.fmt == F_Q8ROW) return (double) ((const float*) W)[row] * (double) (int8_t) (W + (size_t) N * 4)[(size_t) row * K + k];
    const int SB = sbb(t.fmt);
    const uint8_t* blk = W + ((size_t) row * (K / 256) + k / 256) * SB;
    float D = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
    int i = k & 255;
    float eff = D * ((float) blk[2 + (i >> 6)] / 255.0f);
    return ((double) qv(t.fmt, blk + 6, i) - hh(t.fmt)) * eff;
}

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
    const uint64_t s1 = rd.header().sections[1].file_offset, s1n = rd.header().sections[1].byte_length;
    const size_t pg = 16384;
    // map the trunk only: file offset must be page aligned -> map from the floor
    uint64_t mo = s1 / pg * pg;
    size_t mlen = (s1 + s1n - mo + pg - 1) / pg * pg;
    uint8_t* base = (uint8_t*) mmap(nullptr, mlen, PROT_READ, MAP_PRIVATE, fd, (off_t) mo);
    mtl::Buffer wb = cx.wrap(base, mlen);
    const char* names[] = {"model.layers.0.self_attn.q_a_proj.weight", "model.layers.3.self_attn.q_b_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.5.mlp.shared_experts.down_proj.weight",
                           "model.layers.1.self_attn.compressed_indexer.key_value_projection.weight", "model.layers.0.self_attn.compressed_indexer.head_weight_projection.weight",
                           "model.layers.4.self_attn.kv_a_proj_with_mqa.weight", "model.layers.5.mlp.routed_expert_down_proj.weight"};
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    int bad = 0;
    for (const char* name : names) {
        auto it = man.T.find(name);
        if (it == man.T.end()) { printf("skip %s\n", name); continue; }
        const TInfo& t = it->second;
        const int N = (int) t.rows(), K = (int) t.cols();
        const uint8_t* W = base + (s1 - mo) + t.off;
        for (int M : {1, 8, 40}) {
            std::vector<uint16_t> xh((size_t) M * K);
            for (auto& v : xh) v = f2h(nd(rng));
            mtl::Buffer X = cx.alloc(xh.size() * 2), Y = cx.alloc((size_t) M * N * 4);
            std::memcpy(X.ptr, xh.data(), xh.size() * 2);
            std::string k = t.fmt == F_Q8ROW ? (M >= 16 ? "linear_mm_100" : M == 1 ? "linear_q8row_1" : "linear_q8row_8")
                                              : (M >= 16 ? "linear_mm_" + std::to_string(t.fmt) : "linear_sk_" + std::to_string(t.fmt) + (M == 1 ? "_1" : "_8"));
            mtl::Pipeline p = cx.pipeline(lib, k);
            mtl::Dispatch d;
            d.p = &p;
            d.args = {mtl::Arg::B(wb, (s1 - mo) + t.off), mtl::Arg::B(X), mtl::Arg::B(Y), mtl::Arg::V<int32_t>(N), mtl::Arg::V<int32_t>(K), mtl::Arg::V<int32_t>(M)};
            if (M >= 16) { d.grid[0] = (N + 63) / 64; d.grid[1] = (M + 31) / 32; d.tg[0] = 128; }
            else { d.grid[0] = (N + 7) / 8; d.tg[0] = 256; }
            cx.run({d});
            // CPU reference on up to 24 rows x all M
            double maxerr = 0, maxref = 0;
            int nrows = std::min(N, 24);
            for (int ri = 0; ri < nrows; ++ri) {
                int row = (int) ((long) ri * N / nrows);
                for (int m = 0; m < M; ++m) {
                    double acc = 0;
                    for (int kk = 0; kk < K; ++kk) acc += wval(t, W, row, kk) * (double) h2f(xh[(size_t) m * K + kk]);
                    double g = ((const float*) Y.ptr)[(size_t) m * N + row];
                    maxerr = std::max(maxerr, std::fabs(g - acc));
                    maxref = std::max(maxref, std::fabs(acc));
                }
            }
            // GEMM uses fp16 weights: tolerance relative to the dot magnitude
            double rel = maxerr / (maxref + 1e-9);
            bool ok = rel < (M >= 16 ? 2e-3 : 1e-5);
            if (!ok) ++bad;
            printf("%-78s fmt %3d N %5d K %5d M %2d  kernel %-16s maxerr %.3e rel %.3e %s\n", name, t.fmt, N, K, M, k.c_str(), maxerr, rel, ok ? "ok" : "BAD");
            cx.free(X);
            cx.free(Y);
        }
    }
    printf(bad ? "FAILED %d\n" : "ALL OK\n", bad);
    return bad ? 1 : 0;
}
