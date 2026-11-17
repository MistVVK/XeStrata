// tests/quant/quant_bench.cpp - one-thread speed of every dot kernel of the layer (all contracts and ISAs this CPU can run), no ggml.
//
// usage: quant_bench [--type NAME] [--mb N] [--reps N] [--n N] [--arith avx2|avx|generic]
// For each weight type and each runnable kernel: GB/s of weight bytes with the rows streamed from DRAM (a --mb MB set) and with them resident in L2
// (256 KB).  Interleaved A/B discipline: the kernels of one type run round-robin, `reps` rounds, the median of each is printed.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/isa.hpp"
#include "strata/quant/types.hpp"

namespace sq = strata::quant;

namespace {
uint64_t g_s = 88172645463325252ull;
uint32_t rnd() {
    g_s ^= g_s << 13;
    g_s ^= g_s >> 7;
    g_s ^= g_s << 17;
    return (uint32_t) (g_s >> 16);
}

double run(const sq::DotKernel* k, const std::vector<uint8_t>& w, size_t rb, int n, size_t rows, const void* act, float* sink) {
    using C = std::chrono::steady_clock;
    const auto t0 = C::now();
    float acc = 0.f;
    for (size_t r = 0; r < rows; ++r) {
        float s;
        k->fn(n, &s, w.data() + r * rb, act);
        acc += s;
    }
    *sink += acc;
    return std::chrono::duration<double>(C::now() - t0).count();
}
}  // namespace

int main(int argc, char** argv) {
    const char* only = nullptr;
    int mb = 256, reps = 7, n_el = 4096;
    sq::Arith want = sq::default_arith();
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--type") && i + 1 < argc) only = argv[++i];
        else if (!std::strcmp(argv[i], "--mb") && i + 1 < argc) mb = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n_el = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--arith") && i + 1 < argc) {
            const std::string a = argv[++i];
            want = a == "avx" ? sq::Arith::Avx : a == "generic" ? sq::Arith::Generic : sq::Arith::Avx2;
        }
    }
    std::printf("quant_bench: contract %s, best isa %s, DRAM set %d MB, L2 set 256 KB, n=%d, median of %d\n", sq::arith_name(want), sq::isa_name(sq::best_isa()),
                mb, n_el, reps);
    float sink = 0.f;
    for (int t = 0; t < 150; ++t) {
        const sq::TypeInfo* ti = sq::type_info(t);
        if (ti == nullptr || !ti->has_dot || (only && std::strcmp(only, ti->name))) continue;
        const sq::DotKernel* ks[16];
        const int nk = sq::dot_kernels(ti->type, want, ks, 16);
        std::vector<const sq::DotKernel*> run_k;
        for (int i = 0; i < nk; ++i)
            if (sq::isa_ok(ks[i]->isa)) run_k.push_back(ks[i]);
        if (run_k.empty()) continue;
        const sq::TypeInfo* ai = sq::type_info(ti->act);
        const int unit = std::max(ti->blck, ai->blck);
        const int n = n_el / unit * unit;
        const size_t rb = sq::row_bytes(t, n);
        const size_t rows_dram = (size_t) mb * 1048576 / rb, rows_l2 = std::max<size_t>(1, 256 * 1024 / rb);
        std::vector<uint8_t> w(rows_dram * rb);
        for (auto& b : w) b = (uint8_t) rnd();
        // sane scales: first two bytes of each block as fp16 ~ 0.01
        for (size_t o = 0; o + ti->bytes <= w.size(); o += ti->bytes) {
            if (ti->type == sq::Type::IQ1_M || ti->type == sq::Type::NVFP4 || ti->type == sq::Type::MXFP4 || ti->type == sq::Type::Q2_K ||
                ti->type == sq::Type::Q3_K || ti->type == sq::Type::Q6_K) continue;   // scale not at the front: random bytes still give finite-enough sums
            w[o] = 0x66, w[o + 1] = 0x24;
        }
        std::vector<float> x((size_t) n);
        for (auto& v : x) v = (float) ((int) (rnd() % 2001) - 1000) * 1e-3f;
        std::vector<uint8_t> act((size_t) sq::row_bytes((int) ti->act, n));
        const sq::ActKernel* ak = sq::act_kernel(ti->act, want);
        if (ak == nullptr) continue;
        ak->fn(x.data(), act.data(), n);
        std::vector<std::vector<double>> td(run_k.size()), tl(run_k.size());
        for (int rep = 0; rep < reps; ++rep)
            for (size_t i = 0; i < run_k.size(); ++i) {
                td[i].push_back(run(run_k[i], w, rb, n, rows_dram, act.data(), &sink));
                double best = 1e9;
                for (int r2 = 0; r2 < 8; ++r2) best = std::min(best, run(run_k[i], w, rb, n, rows_l2, act.data(), &sink));
                tl[i].push_back(best);
            }
        for (size_t i = 0; i < run_k.size(); ++i) {
            std::sort(td[i].begin(), td[i].end());
            std::sort(tl[i].begin(), tl[i].end());
            const double gd = (double) rows_dram * rb / td[i][td[i].size() / 2] / 1e9, gl = (double) rows_l2 * rb / tl[i][tl[i].size() / 2] / 1e9;
            std::printf("  %-8s %-26s %-6s  DRAM %6.2f GB/s   L2 %6.2f GB/s\n", ti->name, run_k[i]->name, sq::isa_name(run_k[i]->isa), gd, gl);
        }
    }
    if (sink == 12345.f) std::printf("\n");
    return 0;
}
