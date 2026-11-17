// tests/quant/quant_bench_ggml.cpp - the layer's dot kernels and activation quantizers against ggml-cpu's, speed (links ggml-cpu: test only).
//
// For every dot type: R rows of n elements (64 rows of 8192: an L2-resident chunk; --rows 4096 streams from DRAM), one activation row; ours and ggml's alternate in interleaved rounds (ABAB...), each round
// runs every row `reps` times; the report is the median of the per-round ratio ggml_time / our_time (> 1: ours is faster) and the number of rounds ours won.
// Bit-exactness is the oracle's job (quant_parity); this only answers "is the row faster or slower than ggml's".
//
// usage: quant_bench_ggml [--rounds N] [--n ELEMENTS] [--rows R] [--type NAME]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/isa.hpp"
#include "strata/quant/types.hpp"

namespace sq = strata::quant;

namespace {

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Rng {
    uint64_t s = 88172645463325252ull;
    float gauss() {
        float a = 0;
        for (int i = 0; i < 4; ++i) {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            a += (float) ((s * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f - 0.5f;
        }
        return a;
    }
};

sq::Arith oracle_arith() {
#if defined(__aarch64__)
    return sq::Arith::Neon;
#else
    if (ggml_cpu_has_avx2() && ggml_cpu_has_fma()) return sq::Arith::Avx2;
    if (ggml_cpu_has_avx()) return sq::Arith::Avx;
    return sq::Arith::Generic;
#endif
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    int rounds = 11, n = 8192, R = 64;
    const char* only = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) R = std::atoi(argv[++i]);   // 4096+ rows stream from DRAM like the pool's experts
        else if (!std::strcmp(argv[i], "--type") && i + 1 < argc) only = argv[++i];
    }
    ggml_cpu_init();
    const sq::Arith ar = oracle_arith();
    std::printf("quant_bench_ggml: contract %s, n=%d, %d rounds (ratio = ggml time / ours; > 1 means ours is faster)\n", sq::arith_name(ar), n, rounds);
    static const sq::Type kTypes[] = {sq::Type::Q2_0, sq::Type::Q4_0, sq::Type::Q4_1, sq::Type::Q5_0, sq::Type::Q5_1, sq::Type::Q8_0, sq::Type::IQ4_NL,
                                      sq::Type::MXFP4, sq::Type::NVFP4, sq::Type::Q2_K, sq::Type::Q3_K, sq::Type::Q4_K, sq::Type::Q5_K, sq::Type::Q6_K,
                                      sq::Type::IQ1_S, sq::Type::IQ1_M, sq::Type::IQ2_XXS, sq::Type::IQ2_XS, sq::Type::IQ2_S, sq::Type::IQ3_XXS,
                                      sq::Type::IQ3_S, sq::Type::IQ4_XS};
    Rng rng;
    for (sq::Type st : kTypes) {
        const sq::TypeInfo* ti = sq::type_info(st);
        if (only != nullptr && std::strcmp(only, ti->name) != 0) continue;
        const ggml_type gt = (ggml_type) (int) st;
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu(gt);
        const sq::DotKernel* k = sq::dot_kernel(st, ar);
        if (tr == nullptr || tr->vec_dot == nullptr || k == nullptr) continue;
        const ggml_type_traits_cpu* ta = ggml_get_type_traits_cpu(tr->vec_dot_type);
        const int unit = std::max<int>(ti->blck, (int) ggml_blck_size(tr->vec_dot_type));
        const int nn = n / unit * unit;
        std::vector<float> f((size_t) nn * R);
        for (float& v : f) v = rng.gauss();
        std::vector<uint8_t> w(ggml_row_size(gt, nn) * R), act(ggml_row_size(tr->vec_dot_type, nn));
        ggml_quantize_init(gt);
        std::vector<float> im((size_t) nn, 1.f);
        ggml_quantize_chunk(gt, f.data(), w.data(), 0, R, nn, ggml_quantize_requires_imatrix(gt) ? im.data() : nullptr);
        ta->from_float(f.data(), act.data(), nn);
        const size_t rb = ggml_row_size(gt, nn);
        // enough repetitions for ~20 ms per arm
        int reps = 1;
        float sink = 0.f;
        for (;; reps *= 2) {
            const double t0 = now();
            for (int r = 0; r < reps; ++r)
                for (int row = 0; row < R; ++row) {
                    float s;
                    tr->vec_dot(nn, &s, 0, w.data() + rb * row, 0, act.data(), 0, 1);
                    sink += s;
                }
            if (now() - t0 > 0.02 || reps >= 4096) break;
        }
        std::vector<double> ratios;
        double tg_all = 0, to_all = 0;
        int won = 0;
        for (int rd = 0; rd < rounds; ++rd) {
            double tg, to;
            auto run_ggml = [&] {
                const double t0 = now();
                for (int r = 0; r < reps; ++r)
                    for (int row = 0; row < R; ++row) {
                        float s;
                        tr->vec_dot(nn, &s, 0, w.data() + rb * row, 0, act.data(), 0, 1);
                        sink += s;
                    }
                return now() - t0;
            };
            auto run_ours = [&] {
                const double t0 = now();
                for (int r = 0; r < reps; ++r)
                    for (int row = 0; row < R; ++row) {
                        float s;
                        k->fn(nn, &s, w.data() + rb * row, act.data());
                        sink += s;
                    }
                return now() - t0;
            };
            if (rd & 1) { to = run_ours(); tg = run_ggml(); } else { tg = run_ggml(); to = run_ours(); }
            ratios.push_back(tg / to);
            tg_all += tg;
            to_all += to;
            if (to < tg) ++won;
        }
        const double bytes = (double) rb * R * reps * rounds;
        std::printf("  %-8s %-18s median ratio %.3f  ours faster in %2d/%d rounds   ours %.1f GB/s  ggml %.1f GB/s%s\n", ti->name, k->name, median(ratios), won, rounds,
                    bytes / to_all / 1e9, bytes / tg_all / 1e9, sink == 12345.f ? " " : "");
    }
    // activation quantizers
    for (sq::Type at : {sq::Type::Q8_0, sq::Type::Q8_1, sq::Type::Q8_K}) {
        const sq::TypeInfo* ti = sq::type_info(at);
        const ggml_type_traits_cpu* ta = ggml_get_type_traits_cpu((ggml_type) (int) at);
        const sq::ActKernel* ak = sq::act_kernel(at, ar);
        if (ta == nullptr || ta->from_float == nullptr || ak == nullptr) continue;
        if (only != nullptr && std::strcmp(only, ti->name) != 0) continue;
        const int nn = 4096;
        std::vector<float> f((size_t) nn);
        for (float& v : f) v = rng.gauss();
        std::vector<uint8_t> o((size_t) nn / ti->blck * ti->bytes + 64);
        const int reps = 2000;
        std::vector<double> ratios;
        int won = 0;
        for (int rd = 0; rd < rounds; ++rd) {
            double tg, to;
            auto g = [&] { const double t0 = now(); for (int r = 0; r < reps; ++r) ta->from_float(f.data(), o.data(), nn); return now() - t0; };
            auto u = [&] { const double t0 = now(); for (int r = 0; r < reps; ++r) ak->fn(f.data(), o.data(), nn); return now() - t0; };
            if (rd & 1) { to = u(); tg = g(); } else { tg = g(); to = u(); }
            ratios.push_back(tg / to);
            if (to < tg) ++won;
        }
        std::printf("  act %-5s %-18s median ratio %.3f  ours faster in %2d/%d rounds\n", ti->name, ak->name, median(ratios), won, rounds);
    }
    return 0;
}
