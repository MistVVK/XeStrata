// hy4s_cpuexp_test.cpp - tests and benchmarks of the CPU expert engine (lane HY4-SYCL, p1006).
//   hy4s_cpuexp_test acc                         kernel accuracy vs sk_codecs dot_sk_q8_1_row (random S1.25/S2/S3 data)
//   hy4s_cpuexp_test bench [threads...]          kernel speed (engine kernels vs the AVX2 ones of this module), streaming + cached
//   hy4s_cpuexp_test real <model.strata> [nthreads] [warm]   accuracy vs an independent oracle-style implementation + decode / prefill timing
#include <immintrin.h>
#include <omp.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "hy4s_cpuexp.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/quant/sk_codecs.hpp"

using namespace hy4s;
using namespace strata::quant::sk;

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static uint16_t f2h(float f) { return _cvtss_sh(f, 0); }

// random but valid S_K rows: nrows x 6 super-blocks of `code`, +64 bytes padding
static std::vector<uint8_t> make_weights(int code, int nrows, uint32_t seed) {
    const size_t sb = superblock_bytes((SkCode)code), rb = 6 * sb;
    std::vector<uint8_t> w(rb * nrows + 64);
    std::mt19937 rng(seed);
    for (auto& b : w) b = (uint8_t)rng();
    for (int r = 0; r < nrows; ++r)
        for (int s = 0; s < 6; ++s) {
            uint8_t* blk = &w[r * rb + s * sb];
            float D = 0.002f + 0.02f * (float)(rng() % 1000) / 1000.0f;
            uint16_t h = f2h(D);
            std::memcpy(blk, &h, 2);
            for (int g = 0; g < 4; ++g) blk[2 + g] = (uint8_t)(1 + rng() % 255);
        }
    return w;
}
static void make_act(std::mt19937& rng, float* x, int n) {
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < n; ++i) {
        x[i] = nd(rng);
        if (rng() % 97 == 0) x[i] *= 8.f;
    }
}

static int cmd_acc() {
    const int codes[4] = {0, 2, 3, 4};
    bool all_ok = true;
    for (int code : codes) {
        const int NR = 192;
        auto W = make_weights(code, NR, 1234 + code);
        const size_t rb = 6 * superblock_bytes((SkCode)code);
        std::mt19937 rng(77);
        for (int M : {1, 3, 16}) {
            std::vector<float> x((size_t)M * CX_K);
            std::vector<ActPlan> plans(M);
            std::vector<BlockQ8_1_G64> blk((size_t)M * CX_NG);
            for (int t = 0; t < M; ++t) {
                make_act(rng, &x[(size_t)t * CX_K], CX_K);
                quantize_q8_1_g64(&x[(size_t)t * CX_K], &blk[(size_t)t * CX_NG], CX_K);
                plan_from_blocks(&blk[(size_t)t * CX_NG], plans[t]);
            }
            std::vector<const ActPlan*> ap(M);
            for (int t = 0; t < M; ++t) ap[t] = &plans[t];
            std::vector<float> mine((size_t)M * NR);
            if (!cx_rows_q8(code, W.data(), NR, ap.data(), M, mine.data(), NR)) { printf("code %d unsupported\n", code); return 1; }
            long exact = 0, tot = 0;
            double maxabs = 0, maxrel = 0, maxref = 0;
            for (int t = 0; t < M; ++t)
                for (int r = 0; r < NR; ++r) {
                    float ref = dot_sk_q8_1_row(W.data() + (size_t)r * rb, (SkCode)code, &blk[(size_t)t * CX_NG], CX_K);
                    float m = mine[(size_t)t * NR + r];
                    tot++;
                    if (std::memcmp(&ref, &m, 4) == 0) exact++;
                    double ad = std::fabs((double)ref - m);
                    maxabs = std::max(maxabs, ad);
                    maxrel = std::max(maxrel, ad / std::max(1e-12, (double)std::fabs(ref)));
                    maxref = std::max(maxref, (double)std::fabs(ref));
                }
            // with FMA contraction the float tail differs from the engine's compiled kernel in the last bits; -ffp-contract=off gives bit identity
            printf("acc code %d M=%2d: %ld/%ld bit-identical, max abs diff %.3g (%.3g of max|ref| %.3g), max rel diff %.3g\n", code, M, exact, tot, maxabs, maxabs / maxref, maxref, maxrel);
            if (maxabs / maxref > 1e-6) all_ok = false;
        }
    }
    printf(all_ok ? "ACC OK\n" : "ACC FAIL\n");
    return all_ok ? 0 : 1;
}

struct Pool {
    int code;
    int nm;
    std::vector<uint8_t> w;  // nm matrices of 3072 rows
    size_t mat_bytes;
};

static Pool make_pool(int code, size_t total_bytes) {
    Pool p;
    p.code = code;
    const size_t rb = 6 * superblock_bytes((SkCode)code);
    p.mat_bytes = rb * 3072;
    p.nm = (int)std::max<size_t>(1, total_bytes / p.mat_bytes);
    p.w.resize(p.mat_bytes * p.nm + 64);
    // random payload fast: fill one matrix validly, then copy it with a rolling perturbation
    auto one = make_weights(code, 3072, 4242 + code);
    for (int m = 0; m < p.nm; ++m) {
        std::memcpy(&p.w[(size_t)m * p.mat_bytes], one.data(), p.mat_bytes);
        uint8_t* base = &p.w[(size_t)m * p.mat_bytes];
        for (size_t i = 0; i < p.mat_bytes; i += 997) base[i] ^= (uint8_t)(m * 31 + 1) & 0x0F ? 0 : 0;  // keep headers valid
    }
    return p;
}

static void cmd_bench(const std::vector<int>& threads, bool nocache, int only_code) {
    printf("ISA %s, hw threads %d\n", active_sk_isa(), omp_get_max_threads());
    const int codes[4] = {0, 2, 3, 4};
    for (int mode = 0; mode < (nocache ? 1 : 2); ++mode) {  // 0 = streaming (256 MB pool, DRAM bound), 1 = cached (1 MB per thread group, compute bound)
        printf("\n== %s ==\n", mode == 0 ? "streaming (pool >> L3)" : "cached (one matrix, L2/L3 resident)");
        printf("%-6s %-4s %-3s %-8s %8s %10s %12s\n", "code", "thr", "M", "impl", "GB/s", "ms/matrix", "Mrow-tok/s");
        for (int code : codes) {
            if (only_code >= 0 && code != only_code) continue;
            Pool pool = make_pool(code, mode == 0 ? (size_t)256 << 20 : 1);
            const size_t rb = 6 * superblock_bytes((SkCode)code);
            std::mt19937 rng(5);
            std::vector<float> x(16 * CX_K);
            std::vector<ActPlan> plans(16);
            std::vector<BlockQ8_1_G64> blk(16 * CX_NG);
            for (int t = 0; t < 16; ++t) {
                make_act(rng, &x[(size_t)t * CX_K], CX_K);
                quantize_q8_1_g64(&x[(size_t)t * CX_K], &blk[(size_t)t * CX_NG], CX_K);
                plan_from_blocks(&blk[(size_t)t * CX_NG], plans[t]);
            }
            for (int nthr : threads)
                for (int M : {1, 16})
                    for (int impl = 0; impl < 2; ++impl) {
                        if (impl == 1 && code == 4) continue;
                        omp_set_num_threads(nthr);
                        const int NB = pool.nm * 48;  // blocks of 64 rows
                        const int reps = mode == 0 ? 3 : 400;
                        double best = 1e9;
                        std::vector<float> sink((size_t)nthr * 16 * 64 + 64);
                        for (int rep = 0; rep < (mode == 0 ? 4 : 3); ++rep) {
                            double t0 = now_s();
                            for (int rr = 0; rr < (mode == 0 ? 1 : reps); ++rr) {
#pragma omp parallel
                                {
                                    float* o = &sink[(size_t)omp_get_thread_num() * 16 * 64];
                                    const ActPlan* ap[16];
                                    for (int t = 0; t < M; ++t) ap[t] = &plans[t];
#pragma omp for schedule(dynamic, 2)
                                    for (int b = 0; b < NB; ++b) {
                                        const uint8_t* wp = &pool.w[(size_t)(b / 48) * pool.mat_bytes + (size_t)(b % 48) * 64 * rb];
                                        if (impl == 0) gemm_sk_q8_1(wp, (SkCode)code, blk.data(), o, 64, CX_K, M);
                                        else cx_rows_q8(code, wp, 64, ap, M, o, 64);
                                    }
                                }
                            }
                            double dt = (now_s() - t0) / (mode == 0 ? 1 : reps);
                            if (rep > 0) best = std::min(best, dt);
                        }
                        double gbs = (double)pool.nm * pool.mat_bytes / best / 1e9;
                        double rows = (double)pool.nm * 3072 * M / best / 1e6;
                        printf("%-6d %-4d %-3d %-8s %8.2f %10.3f %12.2f\n", code, nthr, M, impl == 0 ? "engine" : "cx-avx2", gbs, best * 1e3 / pool.nm, rows);
                        fflush(stdout);
                    }
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------------------- real
struct Dir {
    strata::artifact::StrataFileReader rd;
    int fd = -1;
    uint64_t sec4 = 0;
    std::vector<int> rank_of, id_of_rank;  // [41*512]
    bool open(const std::string& path) {
        std::string err;
        if (!rd.open(path, err)) { fprintf(stderr, "%s\n", err.c_str()); return false; }
        sec4 = rd.header().sections[4].file_offset;
        const auto& d = rd.expert_directory();
        rank_of.assign(41 * 512, -1);
        id_of_rank.assign(41 * 512, -1);
        for (size_t i = 0; i < d.size(); ++i) { rank_of[(i / 512) * 512 + d[i].expert_id] = (int)(i % 512); id_of_rank[i] = d[i].expert_id; }
        fd = ::open(path.c_str(), O_RDONLY);
        return fd >= 0;
    }
    CpuExperts* synth = nullptr;  // synthetic pool (no file)
    void open_synth(CpuExperts* ce) {
        synth = ce;
        id_of_rank.resize(41 * 512);
        for (int i = 0; i < 41 * 512; ++i) id_of_rank[i] = i % 512;
    }
    void read_body(int L, int e, std::vector<uint8_t>& b, int& gu, int& dn, uint32_t& dn_off) {
        if (synth) {
            const uint8_t* p;
            uint32_t bytes;
            synth->expert_info(L, e, &p, &gu, &dn, &dn_off, &bytes);
            b.assign(p, p + bytes);
            b.resize(bytes + 64, 0);
            return;
        }
        const auto& ent = rd.expert_directory()[(size_t)L * 512 + rank_of[L * 512 + e]];
        b.resize(ent.body_bytes + 64);
        size_t got = 0;
        while (got < ent.body_bytes) {
            ssize_t r = pread(fd, b.data() + got, ent.body_bytes - got, (off_t)(sec4 + ent.body_offset + got));
            if (r <= 0) { fprintf(stderr, "pread failed\n"); exit(1); }
            got += (size_t)r;
        }
        gu = ent.gu_code_id; dn = ent.dn_code_id; dn_off = ent.dn_offset;
    }
};

static float siluf(float x) { return x / (1.0f + std::exp(-x)); }
static float h2f(uint16_t h) { return _cvtsh_ss(h); }

// oracle-style reference of one layer. mode 0: float activations (dequant_sk_row + sequential dot); mode 1: quantized activations through the engine's gemm_sk_q8_1
static void ref_layer(Dir& D, int L, int T, const float* z, const int32_t* idx, const float* w, float* zout, int mode) {
    const int Lw = 1536;
    std::fill(zout, zout + (size_t)T * Lw, 0.0f);
    std::vector<std::pair<int, std::vector<std::pair<int, float>>>> by;
    for (int e = 0; e < 512; ++e) {
        std::vector<std::pair<int, float>> lst;
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < 16; ++k)
                if (idx[t * 16 + k] == e) lst.push_back({t, w[t * 16 + k]});
        if (!lst.empty()) by.push_back({e, lst});
    }
    std::vector<uint8_t> body;
    auto lin = [&](int code, const uint8_t* W, int N, const float* x, float* y, int n) {
        const size_t rb = 6 * superblock_bytes((SkCode)code);
        if (mode == 0) {
            std::vector<float> buf(Lw);
            for (int r = 0; r < N; ++r) {
                dequant_sk_row(W + r * rb, (SkCode)code, buf.data(), Lw);
                for (int t = 0; t < n; ++t) {
                    float s = 0;
                    for (int k = 0; k < Lw; ++k) s += buf[k] * x[(size_t)t * Lw + k];
                    y[(size_t)t * N + r] = s;
                }
            }
        } else {
            std::vector<BlockQ8_1_G64> xq((size_t)n * CX_NG);
            for (int t = 0; t < n; ++t) quantize_q8_1_g64(x + (size_t)t * Lw, &xq[(size_t)t * CX_NG], Lw);
            for (int t0 = 0; t0 < n; t0 += 16) {
                int mm = std::min(16, n - t0);
                std::vector<float> tmp((size_t)16 * N);
                gemm_sk_q8_1(W, (SkCode)code, &xq[(size_t)t0 * CX_NG], tmp.data(), N, Lw, mm);
                for (int t = 0; t < mm; ++t)
                    for (int r = 0; r < N; ++r) y[(size_t)(t0 + t) * N + r] = tmp[(size_t)t * N + r];
            }
        }
    };
    for (auto& pr : by) {
        int e = pr.first;
        auto& lst = pr.second;
        int n = (int)lst.size(), gu, dn;
        uint32_t dno;
        D.read_body(L, e, body, gu, dn, dno);
        std::vector<float> sq(Lw);
        for (int i = 0; i < Lw; ++i) sq[i] = h2f(((const uint16_t*)body.data())[i]);
        std::vector<float> ze((size_t)n * Lw), g((size_t)n * 2 * Lw), h((size_t)n * Lw), ye((size_t)n * Lw);
        for (int i = 0; i < n; ++i) std::memcpy(&ze[(size_t)i * Lw], z + (size_t)lst[i].first * Lw, Lw * 4);
        lin(gu, body.data() + 3072, 2 * Lw, ze.data(), g.data(), n);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < Lw; ++d) {
                float gg = g[(size_t)i * 2 * Lw + d], uu = g[(size_t)i * 2 * Lw + Lw + d];
                float s = std::min(siluf(gg), 10.0f), uc = std::max(-10.0f, std::min(uu, 10.0f));
                h[(size_t)i * Lw + d] = (s * uc) / sq[d];
            }
        lin(dn, body.data() + dno, Lw, h.data(), ye.data(), n);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < Lw; ++d) zout[(size_t)lst[i].first * Lw + d] += ye[(size_t)i * Lw + d] * lst[i].second;
    }
}

static void rel_stats(const float* a, const float* b, size_t n, double& l2rel, double& maxabs, double& maxrelref) {
    double num = 0, den = 0, mx = 0, mr = 0, scale = 0;
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, (double)std::fabs(b[i]));
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - b[i];
        num += d * d;
        den += (double)b[i] * b[i];
        mx = std::max(mx, std::fabs(d));
    }
    l2rel = std::sqrt(num / std::max(den, 1e-300));
    maxabs = mx;
    mr = mx / std::max(scale, 1e-30);
    maxrelref = mr;
}

static void gen_route(std::mt19937& rng, Dir& D, int L, int T, bool skew, std::vector<int32_t>& idx, std::vector<float>& w) {
    idx.assign((size_t)T * 16, 0);
    w.assign((size_t)T * 16, 0);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    for (int t = 0; t < T; ++t) {
        int chosen[16];
        for (int k = 0; k < 16; ++k) {
            for (;;) {
                int r = skew ? (int)(512.0 * std::pow((double)U(rng), 2.6)) : (int)(rng() % 512);
                if (r > 511) r = 511;
                int e = D.id_of_rank[L * 512 + r];
                bool dup = false;
                for (int j = 0; j < k; ++j) dup |= chosen[j] == e;
                if (!dup) { chosen[k] = e; break; }
            }
            idx[t * 16 + k] = chosen[k];
            w[t * 16 + k] = 0.3f + 0.4f * U(rng);
        }
    }
}

static int cmd_real(const std::string& model, int nthr, bool do_warm, bool synth = false, bool quick = false) {
    Dir D;
    CpuExperts ce;
    std::string err;
    if (synth) {
        if (!ce.open_synthetic(nthr, err, 512, 99)) { fprintf(stderr, "open_synthetic: %s\n", err.c_str()); return 1; }
        D.open_synth(&ce);
        printf("synthetic SQ-32-like pool (ranks 0-31 gu S2/dn S3, 32-175 S2/S2, 176-511 S1.25/S1.25), %d threads, ISA %s\n", nthr, active_sk_isa());
    } else {
        if (!D.open(model)) return 1;
        if (!ce.open(model, nthr, err, 512)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        printf("opened %s: section 4 %.2f GB, %d threads, ISA %s\n", model.c_str(), ce.section4_bytes() / 1e9, nthr, active_sk_isa());
    }
    // code histogram by rank
    if (!synth) {
        int h[9][9] = {};
        for (int L = 0; L < 41; ++L)
            for (int r = 0; r < 512; ++r) { const auto& e = D.rd.expert_directory()[L * 512 + r]; h[e.gu_code_id][e.dn_code_id]++; }
        printf("directory (gu,dn) code counts over 41 layers:");
        for (int a = 0; a < 9; ++a) for (int b = 0; b < 9; ++b) if (h[a][b]) printf(" (%d,%d)=%d", a, b, h[a][b]);
        printf("\n");
    }
    if (do_warm) ce.warm(nthr);
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.f, 1.f);
    // ---- accuracy
    bool ok = true;
    for (int L : (quick ? std::vector<int>{} : std::vector<int>{0, 7, 40})) {
        for (int T : {1, 5, 24}) {
            std::vector<float> z((size_t)T * 1536), zo((size_t)T * 1536), zr0((size_t)T * 1536), zr1((size_t)T * 1536);
            for (auto& v : z) v = nd(rng) * 0.7f;
            std::vector<int32_t> idx;
            std::vector<float> w;
            gen_route(rng, D, L, T, (T % 2) == 1, idx, w);
            ref_layer(D, L, T, z.data(), idx.data(), w.data(), zr0.data(), 0);
            ref_layer(D, L, T, z.data(), idx.data(), w.data(), zr1.data(), 1);
            double l2, ma, mr;
            ce.run(L, T, z.data(), idx.data(), w.data(), zo.data(), 0);
            rel_stats(zo.data(), zr0.data(), zo.size(), l2, ma, mr);
            printf("layer %2d T=%2d mode0 vs oracle-float: rel L2 %.3g max abs %.3g (max abs / max|ref| %.3g)\n", L, T, l2, ma, mr);
            if (mr > 1e-5) ok = false;
            ce.run(L, T, z.data(), idx.data(), w.data(), zo.data(), 1);
            rel_stats(zo.data(), zr1.data(), zo.size(), l2, ma, mr);
            printf("layer %2d T=%2d mode1 vs engine-q8 ref : rel L2 %.3g max abs %.3g (max abs / max|ref| %.3g)\n", L, T, l2, ma, mr);
            // 0 with -ffp-contract=off (identical to the engine kernels); with FMA contraction a flipped int8 rounding of h gives ~1e-4
            if (mr > 1e-3) ok = false;
            rel_stats(zo.data(), zr0.data(), zo.size(), l2, ma, mr);
            printf("layer %2d T=%2d mode1 vs oracle-float  : rel L2 %.3g (activation quantization error)\n", L, T, l2);
            if (l2 > 0.05) ok = false;
        }
    }
    if (synth && !quick) {  // T larger than max_tokens is split into passes: same numbers
        CpuExperts c2;
        c2.open_synthetic(nthr, err, 8, 99);
        const int T = 24, L = 7;
        std::vector<float> z((size_t)T * 1536), a((size_t)T * 1536), b((size_t)T * 1536);
        for (auto& v : z) v = nd(rng) * 0.7f;
        std::vector<int32_t> idx;
        std::vector<float> w;
        gen_route(rng, D, L, T, false, idx, w);
        for (int mode : {0, 1}) {
            ce.run(L, T, z.data(), idx.data(), w.data(), a.data(), mode);
            c2.run(L, T, z.data(), idx.data(), w.data(), b.data(), mode);
            printf("split pass test (max_tokens 8 vs 512), mode %d: %s\n", mode, std::memcmp(a.data(), b.data(), a.size() * 4) == 0 ? "identical" : "DIFFERENT");
            if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) ok = false;
        }
    }
    printf(ok ? "REAL ACC OK\n" : "REAL ACC FAIL\n");
    // ---- decode-like: 41 layers x T=1 (random z, random routing: uniform over all experts / skewed to the hot ranks)
    auto decode_tokens = [&](int skew, int ntok, std::vector<double>& per_tok) {
        ce.reset_stats();
        per_tok.clear();
        for (int tk = 0; tk < ntok; ++tk) {
            double t0 = now_s();
            for (int L = 0; L < 41; ++L) {
                std::vector<float> z(1536), zo(1536);
                for (auto& v : z) v = nd(rng) * 0.7f;
                std::vector<int32_t> idx;
                std::vector<float> w;
                gen_route(rng, D, L, 1, skew == 1, idx, w);
                ce.run(L, 1, z.data(), idx.data(), w.data(), zo.data(), 1);
            }
            per_tok.push_back(now_s() - t0);
        }
    };
    auto decode_report = [&](const char* what, std::vector<double> per_tok) {
        const int n = (int)per_tok.size();
        std::sort(per_tok.begin(), per_tok.end());
        const auto& s = ce.stats();
        double med = per_tok[n / 2];
        printf("decode-like %s: %d token(s), per token median %.1f ms (min %.1f), %.2f ms/layer, %.2f GB/s expert bytes (%.2f MB/layer), %lu pairs/token\n", what, n, med * 1e3, per_tok[0] * 1e3,
               med * 1e3 / 41, s.bytes / 1e9 / s.seconds, s.bytes / 1e6 / s.calls, (unsigned long)(s.pairs / n));
        printf("   phases (ms/token): prep %.2f gate_up %.2f down %.2f reduce %.2f; thread-sec/token by code:", s.t_prep * 1e3 / n, s.t_gu * 1e3 / n, s.t_dn * 1e3 / n, s.t_red * 1e3 / n);
        for (int c = 0; c < 9; ++c) if (s.code_sec[c] > 0) printf(" code%d %.1f ms", c, s.code_sec[c] * 1e3 / n);
        printf("\n");
        fflush(stdout);
    };
    {
        std::vector<double> pt;
        decode_tokens(0, 8, pt);
        decode_report("uniform routing, warm", pt);
        decode_tokens(1, 8, pt);
        decode_report("skewed routing (hot ranks), warm", pt);
    }
    // ---- prefill-like: T=512 per layer
    for (int mode : {1}) {
        const int T = 512;
        std::vector<float> z((size_t)T * 1536), zo((size_t)T * 1536);
        for (auto& v : z) v = nd(rng) * 0.7f;
        ce.reset_stats();
        double t0 = now_s();
        for (int L = 0; L < 41; ++L) {
            std::vector<int32_t> idx;
            std::vector<float> w;
            gen_route(rng, D, L, T, true, idx, w);
            ce.run(L, T, z.data(), idx.data(), w.data(), zo.data(), mode);
        }
        double dt = now_s() - t0;
        const auto& s = ce.stats();
        printf("prefill-like T=512 skewed routing mode %d: %.2f s per chunk (%.1f ms/layer) = %.0f tok/s through the experts; weights %.2f GB (%.2f GB/s)\n", mode, dt, dt * 1e3 / 41, T / dt, s.bytes / 1e9, s.bytes / 1e9 / dt);
        printf("   phases (s): prep %.2f gate_up %.2f down %.2f reduce %.2f\n", s.t_prep, s.t_gu, s.t_dn, s.t_red);
        fflush(stdout);
    }
    if (!synth && !quick) {  // cold: Section 4 dropped from the page cache, first touches come from the SSD (run() issues WILLNEED for the routed experts)
        ce.drop_cache();
        std::vector<double> pt;
        for (int i = 0; i < 4; ++i) {
            std::vector<double> one;
            decode_tokens(0, 1, one);
            printf("cold decode token %d: %.1f ms (%.2f GB/s of expert bytes)\n", i, one[0] * 1e3, ce.stats().bytes / 1e9 / ce.stats().seconds);
            fflush(stdout);
        }
        ce.warm(nthr);
        std::vector<double> w2;
        decode_tokens(0, 6, w2);
        decode_report("uniform routing, after re-warm", w2);
    }
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: acc | bench [threads...] | real <model> [nthreads] [warm]\n"); return 2; }
    std::string c = argv[1];
    if (c == "acc") return cmd_acc();
    if (c == "bench") {
        std::vector<int> th;
        bool nocache = false;
        int only = -1;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "nocache") nocache = true;
            else if (a.rfind("code=", 0) == 0) only = atoi(a.c_str() + 5);
            else th.push_back(atoi(argv[i]));
        }
        if (th.empty()) th = {1, 6, 12};
        cmd_bench(th, nocache, only);
        return 0;
    }
    if (c == "synth") return cmd_real("", argc > 2 ? atoi(argv[2]) : 6, false, true);
    if (c == "real") {
        if (argc < 3) return 2;
        int nthr = argc > 3 ? atoi(argv[3]) : 6;
        bool warm = argc > 4 && std::string(argv[4]) == "warm";
        return cmd_real(argv[2], nthr, warm, false, argc > 5 && std::string(argv[5]) == "quick");
    }
    return 2;
}
