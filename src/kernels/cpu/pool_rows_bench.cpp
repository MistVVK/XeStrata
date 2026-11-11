// src/kernels/cpu/pool_rows_bench.cpp - the CPU expert pool's own row functions (native_gu_rows / native_down_rows), no model, no GPU:
// (1) bits: every token's gate/up and down rows against ggml-cpu's own one-row dot and the same SwiGLU, at 1..16 tokens per expert;
// (2) speed: one thread, E experts streamed from memory, ns per expert for gate/up and down at each token count.
// With STRATA_OWN_CPU_QUANT the rows run on the Strata quant layer and (1) checks them against ggml-cpu; without it both sides are ggml-cpu.
// Comparing the two builds' (2) is the pool-level A/B of the layer against ggml-cpu (OWN-2N2: the arm64 gap sat in this loop, not in a dot).
//
// usage: pool_rows_bench [--check-only] [--rounds N] [--experts E] [--nt 1,2,4,8,16] [--pair gu/down ...]   (pairs by ggml type name, e.g. iq3_s/iq4_nl)
// exit status 1 on any differing row.
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    float gauss() {
        float a = 0;
        for (int i = 0; i < 4; ++i) {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            a += (float) ((s * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f - 0.5f;
        }
        return a;
    }
};

int type_by_name(const std::string& n) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* tn = ggml_type_name((ggml_type) t);
        if (tn != nullptr && n == tn) return t;
    }
    return -1;
}

void quantize_rows(int type, const float* src, uint8_t* dst, int64_t rows, int64_t n) {
    ggml_quantize_init((ggml_type) type);
    std::vector<float> im((size_t) n, 1.f);
    ggml_quantize_chunk((ggml_type) type, src, dst, 0, rows, n, ggml_quantize_requires_imatrix((ggml_type) type) ? im.data() : nullptr);
}

float ggml_dot(int type, int n, const void* w, const void* a) {
    float s = 0.f;
    ggml_get_type_traits_cpu((ggml_type) type)->vec_dot(n, &s, 0, w, 0, a, 0, 1);
    return s;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    int rounds = 9, E = 48;
    bool check_only = false;
    std::vector<int> nts = {1, 2, 3, 4, 8, 16};
    std::vector<std::string> pairs;
    int64_t n_embd = 2048, n_ff = 768;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--check-only") check_only = true;
        else if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (a == "--experts" && i + 1 < argc) E = std::max(1, std::atoi(argv[++i]));
        else if (a == "--embd" && i + 1 < argc) n_embd = std::atoll(argv[++i]);
        else if (a == "--ff" && i + 1 < argc) n_ff = std::atoll(argv[++i]);
        else if (a == "--nt" && i + 1 < argc) {
            nts.clear();
            for (const char* p = argv[++i]; *p;) { nts.push_back(std::max(1, std::min(16, std::atoi(p)))); while (*p && *p != ',') ++p; if (*p) ++p; }
        } else if (a == "--pair" && i + 1 < argc) pairs.push_back(argv[++i]);
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (pairs.empty()) pairs = {"iq3_s/iq4_nl", "iq2_s/iq4_nl", "iq3_xxs/iq4_nl", "iq2_xxs/q5_0", "iq4_xs/q5_0", "q4_K/q5_0"};
    ggml_cpu_init();
#if defined(STRATA_OWN_CPU_QUANT)
    const char* lib = "the quant layer";
#else
    const char* lib = "ggml-cpu";
#endif
    std::printf("pool_rows_bench: rows on %s, n_embd %lld, n_ff %lld, %d experts, %d rounds, one thread\n", lib, (long long) n_embd, (long long) n_ff, E,
                rounds);
    int failures = 0;
    Rng rng;
    for (const std::string& pr : pairs) {
        const size_t sl = pr.find('/');
        const int gt = type_by_name(pr.substr(0, sl)), dt = sl == std::string::npos ? -1 : type_by_name(pr.substr(sl + 1));
        cpu::NativeFmt f;
        std::string err;
        if (gt < 0 || dt < 0 || !cpu::native_fmt(gt, dt, n_embd, n_ff, f, err)) {
            std::printf("  %-16s skipped: %s\n", pr.c_str(), err.empty() ? "unknown type" : err.c_str());
            continue;
        }
        // one expert's weights from gaussian floats, copied into E blobs (distinct memory: the timing streams like the pool's experts)
        std::vector<uint8_t> one(f.bytes, 0);
        {
            std::vector<float> w((size_t) (n_ff * n_embd));
            for (auto& v : w) v = rng.gauss();
            quantize_rows(gt, w.data(), one.data(), n_ff, n_embd);
            for (auto& v : w) v = rng.gauss();
            quantize_rows(gt, w.data(), one.data() + f.up_off, n_ff, n_embd);
            std::vector<float> d((size_t) (n_embd * n_ff));
            for (auto& v : d) v = rng.gauss();
            quantize_rows(dt, d.data(), one.data() + f.down_off, n_embd, n_ff);
        }
        const size_t stride = (f.bytes + 4095) / 4096 * 4096;
        std::vector<uint8_t> blobs(stride * (size_t) E);
        for (int e = 0; e < E; ++e) std::memcpy(blobs.data() + stride * (size_t) e, one.data(), f.bytes);
        // 16 tokens' activations and hiddens
        constexpr int MT = 16;
        std::vector<std::vector<uint8_t>> act(MT, std::vector<uint8_t>(f.act_bytes)), hq(MT, std::vector<uint8_t>(f.h_bytes));
        std::vector<std::vector<float>> ff(MT, std::vector<float>((size_t) n_ff)), out(MT, std::vector<float>((size_t) n_embd));
        for (int t = 0; t < MT; ++t) {
            std::vector<float> x((size_t) n_embd), h((size_t) n_ff);
            for (auto& v : x) v = rng.gauss();
            for (auto& v : h) v = 0.25f * rng.gauss();
            cpu::native_quant_act(f, x.data(), act[t].data());
            cpu::native_quant_h(f, h.data(), hq[t].data());
        }
        const void* ap[MT];
        const void* hp[MT];
        float* fp[MT];
        float* op[MT];
        for (int t = 0; t < MT; ++t) { ap[t] = act[t].data(); hp[t] = hq[t].data(); fp[t] = ff[t].data(); op[t] = out[t].data(); }
        // (1) bits at every token count (arm64: every row goes through the one-row dot loop; on x86 the multi-token kernels sum differently and
        // have their own parity programs, so only the speed part runs there)
        int bad = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
        for (const int nt : nts) {
            cpu::native_gu_rows(f, one.data(), ap, nt, fp, 0, (int) n_ff);
            cpu::native_down_rows(f, one.data(), hp, nt, op, 0, (int) n_embd);
            for (int t = 0; t < nt; ++t) {
                for (int64_t r = 0; r < n_ff; ++r) {
                    const float g = ggml_dot(gt, (int) n_embd, one.data() + (size_t) r * f.gu_row, ap[t]);
                    const float u = ggml_dot(gt, (int) n_embd, one.data() + f.up_off + (size_t) r * f.gu_row, ap[t]);
                    const float want = (g / (1.f + std::exp(-g))) * u;
                    if (std::memcmp(&want, &ff[t][(size_t) r], sizeof want) != 0) ++bad;
                }
                for (int64_t r = 0; r < n_embd; ++r) {
                    const float want = ggml_dot(dt, (int) n_ff, one.data() + f.down_off + (size_t) r * f.d_row, hp[t]);
                    if (std::memcmp(&want, &out[t][(size_t) r], sizeof want) != 0) ++bad;
                }
            }
        }
        std::printf("  %-16s bits vs ggml-cpu's one-row dot at %zu token counts: %s (%d rows differ)\n", pr.c_str(), nts.size(), bad ? "DIFFER" : "identical",
                    bad);
#else
        std::printf("  %-16s bits: not checked on this ISA (see native_arithmetic_parity)\n", pr.c_str());
#endif
        failures += bad != 0;
        if (check_only) continue;
        // (2) speed
        for (const int nt : nts) {
            std::vector<double> tg, td;
            for (int rd = 0; rd < rounds; ++rd) {
                const double t0 = now();
                for (int e = 0; e < E; ++e) cpu::native_gu_rows(f, blobs.data() + stride * (size_t) e, ap, nt, fp, 0, (int) n_ff);
                const double t1 = now();
                for (int e = 0; e < E; ++e) cpu::native_down_rows(f, blobs.data() + stride * (size_t) e, hp, nt, op, 0, (int) n_embd);
                const double t2 = now();
                tg.push_back((t1 - t0) / E * 1e9);
                td.push_back((t2 - t1) / E * 1e9);
            }
            std::printf("    nt %2d  gate/up %10.0f ns/expert  down %10.0f ns/expert  (median of %d)\n", nt, median(tg), median(td), rounds);
        }
    }
    std::printf("pool_rows_bench: %s\n", failures ? "FAILED" : "all rows identical");
    return failures ? 1 : 0;
}
