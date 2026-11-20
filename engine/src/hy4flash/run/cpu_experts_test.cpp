// cpu_experts_test.cpp - test / benchmark of hy4_cpu_experts (CPU routed-expert evaluator)
//   cpu_experts_test <model.strata> <mode> [nlayers=10] [threads=6]
//   modes: num     numeric check vs an fp32/double dequant reference (+ determinism over thread counts, T>512 chunking)
//          bench   T = 1,4,8,64 timing, 16 experts/token, nlayers fully resident; DRAM read ceiling
//          stress  ~10% random residency over all layers, unresident / invalid pairs must be skipped
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include "hy4_io.h"
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "hy4_cpu_experts.h"
#include "strata/quant/sk_codecs.hpp"

using namespace hy4;
using strata::quant::sk::SkCode;

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    double gauss() { double u = uni() + 1e-12, v = uni(); return std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v); }
    int below(int n) { return (int)(next() % (uint64_t)n); }
};

static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, ex = (h >> 10) & 0x1F, mn = h & 0x3FF, f;
    if (ex == 0) {
        if (mn == 0) f = sign;
        else { uint32_t e2 = 127 - 14, m2 = mn; while (!(m2 & 0x400)) { m2 <<= 1; --e2; } m2 &= 0x3FF; f = sign | (e2 << 23) | (m2 << 13); }
    } else if (ex == 31) f = sign | 0x7F800000u | (mn << 13);
    else f = sign | ((ex + 112) << 23) | (mn << 13);
    float o;
    std::memcpy(&o, &f, 4);
    return o;
}

static long vm_kb(const char* key) {
    long v = -1;
    if (FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        size_t kl = std::strlen(key);
        while (std::fgets(line, sizeof(line), f))
            if (!std::strncmp(line, key, kl)) { std::sscanf(line + kl, " %ld", &v); break; }
        std::fclose(f);
    }
    return v;
}
static long smaps_kb(const char* key) {  // from /proc/self/smaps_rollup
    long v = -1;
    if (FILE* f = std::fopen("/proc/self/smaps_rollup", "r")) {
        char line[256];
        size_t kl = std::strlen(key);
        while (std::fgets(line, sizeof(line), f))
            if (!std::strncmp(line, key, kl)) { std::sscanf(line + kl, " %ld", &v); break; }
        std::fclose(f);
    }
    return v;
}
static void ram_line(const char* tag) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        std::printf("[ram] %s: WorkingSet %.2f GB, PeakWorkingSet %.2f GB\n", tag, pmc.WorkingSetSize / 1073741824.0, pmc.PeakWorkingSetSize / 1073741824.0);
#else
    std::printf("[ram] %s: VmRSS %.2f GB, VmHWM(peak) %.2f GB, AnonHugePages %.2f GB\n", tag, vm_kb("VmRSS:") / 1048576.0, vm_kb("VmHWM:") / 1048576.0,
                smaps_kb("AnonHugePages:") / 1048576.0);
#endif
}

// quantize-dequantize 1536 floats with the same Q8_1_G64 rule the evaluator uses (for the tight "same activations" check)
static void qdq(float* x, int mode = 1) {  // mode 1: one Q8_1 term, mode 2: two terms (x = da*qa + db*qb, qb = Q8_1 of the residual)
    strata::quant::sk::BlockQ8_1_G64 blk[24], blk2[24];
    strata::quant::sk::quantize_q8_1_g64(x, blk, 1536);
    float res[1536];
    for (int b = 0; b < 24; ++b)
        for (int i = 0; i < 64; ++i) {
            const float dq = blk[b].d * (float)blk[b].qs[i];
            res[b * 64 + i] = x[b * 64 + i] - dq;
            x[b * 64 + i] = dq;
        }
    if (mode == 2) {
        strata::quant::sk::quantize_q8_1_g64(res, blk2, 1536);
        for (int b = 0; b < 24; ++b)
            for (int i = 0; i < 64; ++i) x[b * 64 + i] += blk2[b].d * (float)blk2[b].qs[i];
    }
}

// straightforward reference: fp32 dequant of the real S_K rows, double accumulation, exact z (no activation quantization)
static void reference(const CpuExperts& cx, int layer, int T, const float* z, const int32_t* idx, const float* w, const uint8_t* use, std::vector<double>& out, int qact = 0) {
    out.assign((size_t)T * 1536, 0.0);
    std::vector<float> row(1536);
    std::vector<double> gu(3072), h(1536), sq(1536);
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < 16; ++s) {
            const size_t q = (size_t)t * 16 + s;
            if (use && !use[q]) continue;
            if (idx[q] < 0 || idx[q] >= 512) continue;
            CpuExpertView v = cx.view(layer, idx[q]);
            if (!v.body) continue;
            const size_t gb = strata::quant::sk::superblock_bytes((SkCode)v.gu_code) * 6, db = strata::quant::sk::superblock_bytes((SkCode)v.dn_code) * 6;
            for (int d = 0; d < 1536; ++d) sq[d] = h2f(((const uint16_t*)v.body)[d]);
            std::vector<float> zt(z + (size_t)t * 1536, z + (size_t)(t + 1) * 1536);
            if (qact) qdq(zt.data(), qact);
            for (int r = 0; r < 3072; ++r) {
                strata::quant::sk::dequant_sk_row(v.body + 3072 + (size_t)r * gb, (SkCode)v.gu_code, row.data(), 1536);
                double a = 0;
                for (int k = 0; k < 1536; ++k) a += (double)row[k] * zt[k];
                gu[r] = a;
            }
            for (int d = 0; d < 1536; ++d) {
                double g = gu[d], u = gu[1536 + d];
                double si = g / (1.0 + std::exp(-g));
                h[d] = std::min(si, 10.0) * std::max(-10.0, std::min(u, 10.0)) / sq[d];
            }
            std::vector<float> hf(h.begin(), h.end());
            if (qact) qdq(hf.data(), qact);
            for (int r = 0; r < 1536; ++r) {
                strata::quant::sk::dequant_sk_row(v.body + v.dn_off + (size_t)r * db, (SkCode)v.dn_code, row.data(), 1536);
                double a = 0;
                for (int k = 0; k < 1536; ++k) a += (double)row[k] * hf[k];
                out[(size_t)t * 1536 + r] += (double)w[q] * a;
            }
        }
}

struct Err { double max_rel = 0, mean_rel = 0, min_cos = 1, mean_cos = 0, max_abs_rel = 0; int n = 0; };
static Err compare(const float* got, const std::vector<double>& ref, int T, const int* tok_list = nullptr, int nlist = 0) {
    Err e;
    for (int i = 0; i < (tok_list ? nlist : T); ++i) {
        const int t = tok_list ? tok_list[i] : i;
        double dd = 0, rr = 0, gg = 0, gr = 0, rmax = 0, dmax = 0;
        for (int c = 0; c < 1536; ++c) {
            double g = got[(size_t)t * 1536 + c], r = ref[(size_t)t * 1536 + c], d = g - r;
            dd += d * d; rr += r * r; gg += g * g; gr += g * r;
            rmax = std::max(rmax, std::fabs(r)); dmax = std::max(dmax, std::fabs(d));
        }
        if (rr == 0) continue;
        const double rel = std::sqrt(dd / rr), cs = gr / (std::sqrt(gg) * std::sqrt(rr) + 1e-300);
        e.max_rel = std::max(e.max_rel, rel); e.mean_rel += rel; e.min_cos = std::min(e.min_cos, cs); e.mean_cos += cs;
        e.max_abs_rel = std::max(e.max_abs_rel, dmax / rmax);
        ++e.n;
    }
    if (e.n) { e.mean_rel /= e.n; e.mean_cos /= e.n; }
    return e;
}
static void print_err(const char* tag, const Err& e) {
    std::printf("  %-34s tokens %3d  rel-L2 max %.3e mean %.3e | cos min %.8f mean %.8f | max|d|/max|ref| %.3e\n", tag, e.n, e.max_rel, e.mean_rel, e.min_cos, e.mean_cos, e.max_abs_rel);
}

struct Batch {
    int T;
    std::vector<float> z, w;
    std::vector<int32_t> idx;
};
static Batch make_batch(Rng& r, int T, const std::vector<int>& pool /*expert ids to draw from; empty = 0..511*/) {
    Batch b;
    b.T = T;
    b.z.resize((size_t)T * 1536);
    b.w.resize((size_t)T * 16);
    b.idx.resize((size_t)T * 16);
    for (auto& x : b.z) x = (float)r.gauss();
    for (int t = 0; t < T; ++t) {
        std::set<int> used;
        for (int s = 0; s < 16; ++s) {
            int e;
            do { e = pool.empty() ? r.below(512) : pool[r.below((int)pool.size())]; } while (used.count(e));
            used.insert(e);
            b.idx[(size_t)t * 16 + s] = e;
            b.w[(size_t)t * 16 + s] = (float)(0.1 + 0.6 * r.uni());
        }
    }
    return b;
}

static int test_num(const std::string& path, int nth) {
    Rng rng(7);
    std::vector<uint16_t> eor, codes;
    std::string err0;
    if (!CpuExperts::read_rank_table(path, eor, codes, err0)) { std::printf("read_rank_table failed: %s\n", err0.c_str()); return 1; }
    // layer with the most distinct (gu,dn) code combinations; pool = up to 48 experts round-robin over the combos
    int L = 5, best = -1;
    for (int l = 0; l < 41; ++l) {
        std::set<int> cs;
        for (int e = 0; e < 512; ++e) cs.insert(codes[(size_t)l * 512 + e]);
        if ((int)cs.size() > best) { best = (int)cs.size(); L = l; }
    }
    std::vector<int> pool;
    std::vector<std::vector<bool>> mask(41, std::vector<bool>(512, false));
    {
        std::vector<std::vector<int>> by;
        std::vector<uint16_t> keys;
        for (int e = 0; e < 512; ++e) {
            const uint16_t c = codes[(size_t)L * 512 + e];
            size_t k = 0;
            while (k < keys.size() && keys[k] != c) ++k;
            if (k == keys.size()) { keys.push_back(c); by.emplace_back(); }
            by[k].push_back(e);
        }
        for (size_t k = 0; k < by.size(); ++k)
            std::printf("  layer %d combo gu=S%d dn=S%d: %zu experts\n", L, keys[k] >> 8, keys[k] & 255, by[k].size());
        std::set<int> s;
        for (int round = 0; (int)s.size() < 48 && round < 64; ++round)
            for (size_t k = 0; k < by.size() && (int)s.size() < 48; ++k)
                if (round < (int)by[k].size()) s.insert(by[k][(round * 7 + (int)k) % by[k].size()]);
        pool.assign(s.begin(), s.end());
        for (int e : pool) mask[L][e] = true;
    }
    CpuExperts cx;
    std::string err;
    if (!cx.init(path, nth, &mask, err)) { std::printf("init failed: %s\n", err.c_str()); return 1; }
    {
        int hist[9] = {};
        for (int e : pool) { CpuExpertView v = cx.view(L, e); hist[v.gu_code]++; }
        std::printf("layer %d: %zu resident experts, kernel=%s, gu code histogram S2..S6,S8: ", L, pool.size(), cx.kernel_name());
        for (int c = 0; c <= 8; ++c) if (hist[c]) std::printf("code%d=%d ", c, hist[c]);
        std::printf("\n");
    }
    int bad = 0;
    for (int T : {1, 4, 8, 64}) {
        Batch b = make_batch(rng, T, pool);
        std::vector<float> out((size_t)T * 1536, -777.f);
        cx.submit(L, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, out.data());
        cx.wait();
        std::vector<double> ref;
        reference(cx, L, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, ref);
        Err e = compare(out.data(), ref, T);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "T=%d (all 16 experts CPU)", T);
        print_err(tag, e);
        if (!(e.max_rel < 3e-2 && e.min_cos > 0.9995)) { std::printf("  FAIL: error larger than int8-activation level\n"); ++bad; }
        if (T <= 8 || T == 64) {
            // tight check: same Q8_1 activations in the reference (only h-quantization differs where rounding flips) -> kernel correctness
            std::vector<double> refq;
            reference(cx, L, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, refq, true);
            Err eq = compare(out.data(), refq, T);
            print_err("  same-activations reference", eq);
            if (!(eq.max_rel < 2e-3)) { std::printf("  FAIL: kernel differs from same-activation reference\n"); ++bad; }
        }
        // accumulate mode
        std::vector<float> acc((size_t)T * 1536, 1.0f);
        cx.submit(L, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, acc.data(), true);
        cx.wait();
        double md = 0;
        for (size_t i = 0; i < acc.size(); ++i) md = std::max(md, (double)std::fabs(acc[i] - 1.0f - out[i]));
        std::printf("  accumulate mode: max |(1+partial)-1 - partial| = %.3e (float rounding only)\n", md);
        if (md > 1e-5) ++bad;
        // determinism: second run bit-exact
        std::vector<float> out2((size_t)T * 1536, 5.f);
        cx.submit(L, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, out2.data());
        cx.wait();
        std::printf("  rerun bit-exact: %s\n", std::memcmp(out.data(), out2.data(), out.size() * 4) == 0 ? "yes" : "NO");
        if (std::memcmp(out.data(), out2.data(), out.size() * 4) != 0) ++bad;
        {   // high-precision (two-term) activations
            cx.set_hi_activations(true);
            std::vector<float> oh((size_t)T * 1536, -3.f), oh2((size_t)T * 1536, 4.f);
            cx.submit(L, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, oh.data());
            cx.wait();
            cx.submit(L, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, oh2.data());
            cx.wait();
            print_err("  [hi] vs exact fp32 z", compare(oh.data(), ref, T));
            const Err e1 = compare(oh.data(), ref, T);
            if (!(e1.max_rel < 2e-3)) { std::printf("  FAIL: hi mode error above 2e-3\n"); ++bad; }
            if (T <= 8 || T == 64) {
                std::vector<double> refh;
                reference(cx, L, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, refh, 2);
                const Err e2 = compare(oh.data(), refh, T);
                print_err("  [hi] vs same-act 2-term ref", e2);
                if (!(e2.max_rel < 2e-3)) { std::printf("  FAIL: hi kernel differs from 2-term reference\n"); ++bad; }
            }
            const bool same = std::memcmp(oh.data(), oh2.data(), oh.size() * 4) == 0;
            std::printf("  [hi] rerun bit-exact: %s\n", same ? "yes" : "NO");
            if (!same) ++bad;
            if (T == 8) {
                CpuExperts c1;
                if (!c1.init(path, 1, &mask, err)) { std::printf("init(1) failed: %s\n", err.c_str()); return 1; }
                c1.set_hi_activations(true);
                std::vector<float> o1((size_t)T * 1536, 9.f);
                c1.submit(L, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, o1.data());
                c1.wait();
                const bool s1 = std::memcmp(oh.data(), o1.data(), oh.size() * 4) == 0;
                std::printf("  [hi] 1 thread vs %d threads bit-exact: %s\n", nth, s1 ? "yes" : "NO");
                if (!s1) ++bad;
            }
            cx.set_hi_activations(false);
        }
        if (T == 8) {
            // thread-count independence
            CpuExperts c1;
            if (!c1.init(path, 1, &mask, err)) { std::printf("init(1) failed: %s\n", err.c_str()); return 1; }
            std::vector<float> o1((size_t)T * 1536, 9.f);
            c1.submit(L, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, o1.data());
            c1.wait();
            const bool same = std::memcmp(out.data(), o1.data(), out.size() * 4) == 0;
            std::printf("  1 thread vs %d threads bit-exact: %s\n", nth, same ? "yes" : "NO");
            if (!same) ++bad;
        }
    }
    // chunking T > 512: tokens around the chunk boundary vs single-token jobs and vs reference
    {
        const int T = 600;
        Batch b = make_batch(rng, T, pool);
        std::vector<float> out((size_t)T * 1536, -1.f);
        cx.submit(L, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, out.data());
        cx.wait();
        const int probe[] = {0, 1, 255, 510, 511, 512, 513, 599};
        double md = 0;
        std::vector<float> one(1536);
        for (int t : probe) {
            cx.submit(L, 1, b.z.data() + (size_t)t * 1536, b.idx.data() + (size_t)t * 16, b.w.data() + (size_t)t * 16, nullptr, one.data());
            cx.wait();
            for (int c = 0; c < 1536; ++c) md = std::max(md, (double)std::fabs(one[c] - out[(size_t)t * 1536 + c]));
        }
        std::printf("  T=600 (2 chunks) vs single-token jobs on 8 probe tokens: max |diff| = %.3e\n", md);
        if (md > 1e-3) ++bad;
        // reference on 2 probe tokens
        std::vector<int32_t> idx2(32);
        std::vector<float> w2(32), z2(2 * 1536);
        for (int k = 0; k < 2; ++k) {
            int t = k ? 512 : 510;
            std::memcpy(&idx2[k * 16], &b.idx[(size_t)t * 16], 64);
            std::memcpy(&w2[k * 16], &b.w[(size_t)t * 16], 64);
            std::memcpy(&z2[(size_t)k * 1536], &b.z[(size_t)t * 1536], 6144);
        }
        std::vector<double> ref;
        reference(cx, L, 2, z2.data(), idx2.data(), w2.data(), nullptr, ref);
        std::vector<float> g2(2 * 1536);
        std::memcpy(g2.data(), &out[(size_t)510 * 1536], 6144);
        std::memcpy(g2.data() + 1536, &out[(size_t)512 * 1536], 6144);
        print_err("T=600 tokens 510,512 vs reference", compare(g2.data(), ref, 2));
    }
    ram_line("num end");
    std::printf("NUM TEST: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

static double sum_arena(const CpuExperts& cx, int layers, int nth, double* gbps_out) {
    // multi-threaded read-only sum over all resident bodies = DRAM read ceiling for this thread count
    std::vector<std::pair<const uint8_t*, size_t>> segs;
    for (int l = 0; l < layers; ++l)
        for (int e = 0; e < 512; ++e) {
            CpuExpertView v = cx.view(l, e);
            if (v.body) segs.push_back({v.body, v.body_bytes});
        }
    size_t total = 0;
    for (auto& s : segs) total += s.second;
    double best = 0;
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<std::thread> th;
        std::vector<uint64_t> part(nth, 0);
        double t0 = now_s();
        for (int i = 0; i < nth; ++i)
            th.emplace_back([&, i]() {
                uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                for (size_t k = i; k < segs.size(); k += nth) {
                    const uint64_t* p = (const uint64_t*)segs[k].first;
                    size_t n = segs[k].second / 8 / 4 * 4;
                    for (size_t j = 0; j < n; j += 4) { a0 += p[j]; a1 += p[j + 1]; a2 += p[j + 2]; a3 += p[j + 3]; }
                }
                part[i] = a0 + a1 + a2 + a3;
            });
        for (auto& t : th) t.join();
        double dt = now_s() - t0;
        best = std::max(best, total / dt / 1e9);
        volatile uint64_t sink = 0;
        for (auto v : part) sink += v;
    }
    *gbps_out = best;
    return (double)total;
}

static int test_bench(const std::string& path, int nlayers, int nth) {
    std::vector<std::vector<bool>> mask(41, std::vector<bool>(512, false));
    for (int l = 0; l < nlayers; ++l) std::fill(mask[l].begin(), mask[l].end(), true);
    CpuExperts cx;
    std::string err;
    if (!cx.init(path, nth, &mask, err)) { std::printf("init failed: %s\n", err.c_str()); return 1; }
    ram_line("after load");
    double bw;
    sum_arena(cx, nlayers, nth, &bw);
    std::printf("DRAM read ceiling (plain sum loop, %d threads over the arena): %.1f GB/s\n", nth, bw);
    Rng rng(11);
    std::printf("kernel: %s\n", cx.kernel_name());
    std::printf("%-14s %-4s %-6s %-10s %-10s %-10s %-12s %-10s\n", "scenario", "T", "reps", "ms/layer", "p50 ms", "min ms", "MB/layer", "GB/s(mean)");
    struct Sc { const char* name; int T; int reps; bool same16; };
    const Sc scs[] = {{"spread", 1, 200, false}, {"spread", 4, 200, false}, {"spread", 8, 200, false}, {"spread", 64, 12, false},
                      {"same16", 16, 12, true}, {"same16", 64, 8, true}, {"same16", 512, 4, true},
                      {"spread(all512)", 512, 4, false}};
    const bool no512 = std::getenv("STRATA_BENCH_NO512") != nullptr;  // skip the T=512 scenarios (slow on the library kernel)
    const int maxt = std::getenv("STRATA_BENCH_MAXT") ? std::atoi(std::getenv("STRATA_BENCH_MAXT")) : 1 << 30;  // skip scenarios with larger T
    const int gap_us = std::getenv("STRATA_BENCH_GAP_US") ? std::atoi(std::getenv("STRATA_BENCH_GAP_US")) : 0;  // idle gap between jobs (emulates GPU work: pool threads go to sleep)
    for (const Sc& sc : scs) {
        const int T = sc.T, reps = sc.reps;
        if (no512 && T >= 512) continue;
        if (T > maxt || (sc.same16 && maxt < 16)) continue;
        std::vector<int> pool16;
        if (sc.same16) {
            std::set<int> s;
            while ((int)s.size() < 16) s.insert(rng.below(512));
            pool16.assign(s.begin(), s.end());
        }
        std::vector<double> ms;
        std::vector<float> out((size_t)T * 1536);
        double bytes_total = 0, wall_total = 0, ph[5] = {0, 0, 0, 0, 0};
        // STRATA_BENCH_K = number of experts per token actually evaluated on the CPU (slots 0..K-1; the rest are skipped), default all 16
        const int Kx = std::getenv("STRATA_BENCH_K") ? std::atoi(std::getenv("STRATA_BENCH_K")) : 16;
        std::vector<uint8_t> use;
        if (Kx < 16) {
            use.assign((size_t)T * 16, 0);
            for (int t = 0; t < T; ++t)
                for (int s = 0; s < Kx; ++s) use[(size_t)t * 16 + s] = 1;
        }
        const uint8_t* usep = use.empty() ? nullptr : use.data();
        // warm-up (one job)
        { Batch b = make_batch(rng, T, pool16); cx.submit(0, T, b.z.data(), b.idx.data(), b.w.data(), usep, out.data()); cx.wait(); }
        for (int r = 0; r < reps; ++r) {
            Batch b = make_batch(rng, T, pool16);
            const int layer = r % nlayers;
            cx.reset_stats();
            double t0 = now_s();
            cx.submit(layer, T, b.z.data(), b.idx.data(), b.w.data(), usep, out.data());
            cx.wait();
            double dt = now_s() - t0;
            CpuExpertStats s = cx.stats();
            ms.push_back(dt * 1e3);
            bytes_total += (double)s.bytes;
            wall_total += dt;
            for (int k = 0; k < 5; ++k) ph[k] += s.phase_s[k];
            if (gap_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
        }
        std::sort(ms.begin(), ms.end());
        double mean = 0;
        for (double v : ms) mean += v;
        mean /= ms.size();
        std::printf("%-14s %-4d %-6d %-10.3f %-10.3f %-10.3f %-12.1f %-10.2f\n", sc.name, T, reps, mean, ms[ms.size() / 2], ms[0], bytes_total / reps / 1e6, bytes_total / wall_total / 1e9);
        std::printf("   phases ms/layer: plan %.3f quantize %.3f gate/up+h %.3f down %.3f combine %.3f\n", ph[0] / reps * 1e3, ph[1] / reps * 1e3, ph[2] / reps * 1e3, ph[3] / reps * 1e3, ph[4] / reps * 1e3);
    }
    // T=1 with a fixed layer and the same 16 experts every time (cache-warm upper bound, 16 experts ~ 45 MB > L3)
    {
        Batch b = make_batch(rng, 1, {});
        std::vector<float> out(1536);
        std::vector<double> ms;
        for (int r = 0; r < 40; ++r) {
            double t0 = now_s();
            cx.submit(1, 1, b.z.data(), b.idx.data(), b.w.data(), nullptr, out.data());
            cx.wait();
            ms.push_back((now_s() - t0) * 1e3);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("T=1 same 16 experts repeated: p50 %.3f ms min %.3f ms\n", ms[ms.size() / 2], ms[0]);
    }
    ram_line("bench end");
    return 0;
}

static int test_stress(const std::string& path, int nth) {
    Rng rng(99);
    std::vector<std::vector<bool>> mask(41, std::vector<bool>(512, false));
    int nres = 0;
    for (int l = 0; l < 41; ++l)
        for (int e = 0; e < 512; ++e)
            if (rng.below(10) == 0) { mask[l][e] = true; ++nres; }
    CpuExperts cx;
    std::string err;
    if (!cx.init(path, nth, &mask, err)) { std::printf("init failed: %s\n", err.c_str()); return 1; }
    ram_line("after load");
    std::printf("mask %d experts, resident_count %llu, arena %.2f GB\n", nres, (unsigned long long)cx.resident_count(), cx.arena_bytes() / 1e9);
    int bad = 0;
    if ((uint64_t)nres != cx.resident_count()) { std::printf("FAIL resident_count\n"); ++bad; }
    for (int l = 0; l < 41; ++l)
        for (int e = 0; e < 512; ++e)
            if (cx.resident(l, e) != (bool)mask[l][e]) { std::printf("FAIL resident() mismatch l=%d e=%d\n", l, e); ++bad; l = 99; break; }
    uint64_t exp_skipped = 0, exp_pairs = 0;
    double worst_rel = 0, worst_cos = 1, worst_q = 0;
    cx.reset_stats();
    for (int l = 0; l < 41; ++l) {
        const int T = 1 + (l % 5);
        Batch b = make_batch(rng, T, {});
        std::vector<uint8_t> use((size_t)T * 16);
        for (auto& u : use) u = rng.below(10) != 0;
        if (l % 7 == 3) { b.idx[0] = -1; use[0] = 1; b.idx[1] = 600; use[1] = 1; exp_skipped += 2; }  // invalid ids must be skipped
        for (size_t q = 0; q < use.size(); ++q) {
            const int e = b.idx[q];
            if (e < 0 || e >= 512) continue;
            if (!use[q]) continue;
            if (mask[l][e]) ++exp_pairs; else ++exp_skipped;
        }
        std::vector<float> out((size_t)T * 1536, std::nanf(""));
        cx.submit(l, T, b.z.data(), b.idx.data(), b.w.data(), use.data(), out.data());
        cx.wait();
        std::vector<double> ref;
        reference(cx, l, T, b.z.data(), b.idx.data(), b.w.data(), use.data(), ref);
        bool nan = false;
        for (float v : out) if (std::isnan(v)) nan = true;
        if (nan) { std::printf("FAIL layer %d: NaN left in zout (overwrite mode must write every token)\n", l); ++bad; }
        Err e = compare(out.data(), ref, T);
        worst_rel = std::max(worst_rel, e.max_rel);
        worst_cos = std::min(worst_cos, e.min_cos);
        {
            std::vector<double> refq;
            reference(cx, l, T, b.z.data(), b.idx.data(), b.w.data(), use.data(), refq, true);
            Err eq = compare(out.data(), refq, T);
            worst_q = std::max(worst_q, eq.max_rel);
        }
        // tokens with no CPU pair must be exactly zero
        for (int t = 0; t < T; ++t) {
            bool any = false;
            for (int s = 0; s < 16; ++s) {
                size_t q = (size_t)t * 16 + s;
                if (use[q] && b.idx[q] >= 0 && b.idx[q] < 512 && mask[l][b.idx[q]]) any = true;
            }
            if (!any) for (int c = 0; c < 1536; ++c) if (out[(size_t)t * 1536 + c] != 0.0f) { std::printf("FAIL layer %d token %d should be zero\n", l, t); ++bad; break; }
        }
    }
    CpuExpertStats s = cx.stats();
    std::printf("stats: jobs %llu pairs %llu (expected %llu) skipped %llu (expected %llu)\n", (unsigned long long)s.jobs, (unsigned long long)s.pairs,
                (unsigned long long)exp_pairs, (unsigned long long)s.skipped_unresident, (unsigned long long)exp_skipped);
    if (s.pairs != exp_pairs || s.skipped_unresident != exp_skipped) { std::printf("FAIL counters\n"); ++bad; }
    std::printf("over 41 layers: worst rel-L2 %.3e, worst cos %.8f\n", worst_rel, worst_cos);
    if (worst_rel > 3e-2) ++bad;
    std::printf("same-activations reference (kernel=%s), worst rel-L2 over 41 layers: %.3e\n", cx.kernel_name(), worst_q);
    if (worst_q > 2e-3) { std::printf("FAIL kernel differs from same-activation reference\n"); ++bad; }
    // rank-range masks
    {
        std::vector<std::vector<bool>> rm;
        std::string e2;
        std::vector<uint16_t> eor, codes;
        if (!CpuExperts::read_rank_table(path, eor, codes, e2) || !CpuExperts::make_mask_rank_range(path, 100, 110, rm, e2)) { std::printf("FAIL rank mask: %s\n", e2.c_str()); ++bad; }
        else {
            int cnt = 0, wrong = 0;
            for (int l = 0; l < (int)rm.size(); ++l)
                for (int e = 0; e < 512; ++e) if (rm[l][e]) ++cnt;
            for (int l = 0; l < (int)rm.size(); ++l)
                for (int r = 0; r < 512; ++r) if (rm[l][eor[(size_t)l * 512 + r]] != (r >= 100 && r < 110)) ++wrong;
            std::printf("make_mask_rank_range(100,110): %zu layers, %d resident (expected %zu), %d wrong\n", rm.size(), cnt, rm.size() * 10, wrong);
            if (cnt != (int)rm.size() * 10 || wrong) ++bad;
            std::vector<int> fr(41, 0), lr(41, 0);
            for (int l = 0; l < 41; ++l) { fr[l] = l; lr[l] = l + 3; }
            std::vector<std::vector<bool>> rm2;
            if (!CpuExperts::make_mask_rank_range(path, fr, lr, rm2, e2)) ++bad;
            else {
                int c2 = 0;
                for (auto& v : rm2) for (bool b : v) c2 += b;
                std::printf("per-layer ranges [l, l+3): %d resident (expected %d)\n", c2, 41 * 3);
                if (c2 != 41 * 3) ++bad;
            }
        }
    }
    ram_line("stress end");
    std::printf("STRESS TEST: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

// ==================================================================================================== cold experts
static void print_cold_stats(const char* tag, const CpuExpertStats& s) {
    std::printf("  [%s] cold: %llu reads, %.1f MB read, %llu cache hits, %llu errors, wait %.1f ms, io %.1f ms, SSD %.2f GB/s\n", tag, (unsigned long long)s.cold_reads,
                s.cold_bytes / 1e6, (unsigned long long)s.cold_hits, (unsigned long long)s.cold_errors, s.cold_wait_s * 1e3, s.cold_io_s * 1e3, s.cold_gbps());
}

// cold vs all-resident must be bit-exact (bodies from the arena, from the slot cache and from the file give identical results)
static int test_cold(const std::string& path, int nth) {
#ifdef _WIN32
    _putenv_s("STRATA_CPUEXP_COLD_CACHE_MB", "16");  // 4 slots of ~4 MB: forces many waves and evictions
#else
    setenv("STRATA_CPUEXP_COLD_CACHE_MB", "16", 1);  // 4 slots of ~4 MB: forces many waves and evictions
#endif
    Rng rng(5);
    const int layers[2] = {3, 7};
    std::vector<std::vector<bool>> all(41, std::vector<bool>(512, false)), half(41, std::vector<bool>(512, false)), cold_half(41, std::vector<bool>(512, false)),
        none(41, std::vector<bool>(512, false));
    std::vector<int> pool[2];
    for (int k = 0; k < 2; ++k) {
        std::set<int> s;
        while ((int)s.size() < 48) s.insert(rng.below(512));
        pool[k].assign(s.begin(), s.end());
        for (size_t i = 0; i < pool[k].size(); ++i) {
            all[layers[k]][pool[k][i]] = true;
            (i % 2 == 0 ? half : cold_half)[layers[k]][pool[k][i]] = true;
        }
    }
    CpuExperts A, B, C;
    std::string err;
    if (!A.init(path, nth, &all, err) || !B.init(path, nth, &half, err) || !C.init(path, nth, &none, err)) { std::printf("init failed: %s\n", err.c_str()); return 1; }
    if (!B.set_cold_experts(cold_half, err)) { std::printf("set_cold B failed: %s\n", err.c_str()); return 1; }
    if (!C.set_cold_experts(all, err)) { std::printf("set_cold C failed: %s\n", err.c_str()); return 1; }
    int bad = 0;
    // overlap must be rejected
    { CpuExperts X; if (!X.init(path, 1, &half, err)) return 1; std::string e2; if (X.set_cold_experts(half, e2)) { std::printf("FAIL: cold/resident overlap accepted\n"); ++bad; } }
    for (int k = 0; k < 2; ++k)
        for (int T : {1, 8, 64}) {
            for (int rep = 0; rep < 2; ++rep) {  // second pass: cache hits
                Batch b = make_batch(rng, T, pool[k]);
                std::vector<float> oA((size_t)T * 1536, 1.f), oB((size_t)T * 1536, 2.f), oC((size_t)T * 1536, 3.f);
                A.submit(layers[k], T, b.z.data(), b.idx.data(), b.w.data(), nullptr, oA.data()); A.wait();
                B.submit(layers[k], T, b.z.data(), b.idx.data(), b.w.data(), nullptr, oB.data()); B.wait();
                C.submit(layers[k], T, b.z.data(), b.idx.data(), b.w.data(), nullptr, oC.data()); C.wait();
                C.submit(layers[k], T, b.z.data(), b.idx.data(), b.w.data(), nullptr, oC.data()); C.wait();  // same job again: cache hits
                const bool sb = std::memcmp(oA.data(), oB.data(), oA.size() * 4) == 0, sc = std::memcmp(oA.data(), oC.data(), oA.size() * 4) == 0;
                std::printf("  layer %d T=%-3d pass %d: half-cold vs resident %s, all-cold vs resident %s\n", layers[k], T, rep, sb ? "bit-exact" : "DIFFERENT", sc ? "bit-exact" : "DIFFERENT");
                if (!sb || !sc) ++bad;
            }
        }
    {   // partial use_pair mask + T > 512 (two chunks) on the all-cold instance
        const int T = 600;
        Batch b = make_batch(rng, T, pool[0]);
        std::vector<uint8_t> use((size_t)T * 16);
        for (auto& u : use) u = rng.below(4) != 0;
        std::vector<float> oA((size_t)T * 1536), oC((size_t)T * 1536);
        A.submit(layers[0], T, b.z.data(), b.idx.data(), b.w.data(), use.data(), oA.data()); A.wait();
        C.submit(layers[0], T, b.z.data(), b.idx.data(), b.w.data(), use.data(), oC.data()); C.wait();
        const bool same = std::memcmp(oA.data(), oC.data(), oA.size() * 4) == 0;
        std::printf("  T=600 (2 chunks) with a use_pair mask, all-cold vs resident: %s\n", same ? "bit-exact" : "DIFFERENT");
        if (!same) ++bad;
    }
    {   // experts in neither mask are skipped, cold ones are computed
        std::vector<int32_t> idx(16);
        std::vector<float> w(16, 0.1f), z(1536, 0.5f), o(1536, 7.f);
        std::vector<int> others;
        for (int e = 0; e < 512 && others.size() < 8; ++e) if (!all[layers[0]][e]) others.push_back(e);
        for (int s = 0; s < 16; ++s) idx[s] = s < 8 ? pool[0][s] : others[s - 8];
        C.reset_stats();
        C.submit(layers[0], 1, z.data(), idx.data(), w.data(), nullptr, o.data()); C.wait();
        const CpuExpertStats s = C.stats();
        std::printf("  GPU-owned experts skipped: pairs %llu (expected 8), skipped %llu (expected 8)\n", (unsigned long long)s.pairs, (unsigned long long)s.skipped_unresident);
        if (s.pairs != 8 || s.skipped_unresident != 8) ++bad;
    }
    print_cold_stats("B half-cold", B.stats());
    print_cold_stats("C all-cold ", C.stats());
    if (B.stats().cold_errors || C.stats().cold_errors) ++bad;
    std::printf("ram_budget_bytes() = %.2f GB\n", CpuExperts::ram_budget_bytes() / 1e9);
    ram_line("cold end");
    std::printf("COLD TEST: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

// T=1..8 jobs with a given fraction of the used experts cold: ms/layer and the SSD rate
//   coldbench <model> <nlayers> <threads> <resident_ranks R> <pc: fraction of the 16 picks that are cold> <reps> [T]
static int test_coldbench(const std::string& path, int nlayers, int nth, int R, double pc, int reps, int T) {
    std::vector<uint16_t> eor, codes;
    std::string err;
    if (!CpuExperts::read_rank_table(path, eor, codes, err)) { std::printf("%s\n", err.c_str()); return 1; }
    std::vector<std::vector<bool>> res(41, std::vector<bool>(512, false)), cold(41, std::vector<bool>(512, false));
    std::vector<std::vector<int>> res_ids(nlayers), cold_ids(nlayers);
    for (int l = 0; l < nlayers; ++l)
        for (int r = 0; r < 512; ++r) {
            const int e = eor[(size_t)l * 512 + r];
            if (r < R) { res[l][e] = true; res_ids[l].push_back(e); }
            else { cold[l][e] = true; cold_ids[l].push_back(e); }
        }
    CpuExperts cx;
    const double t0 = now_s();
    if (!cx.init(path, nth, &res, err)) { std::printf("init failed: %s\n", err.c_str()); return 1; }
    if (!cx.set_cold_experts(cold, err)) { std::printf("set_cold failed: %s\n", err.c_str()); return 1; }
    std::printf("loaded %llu resident experts (%.2f GB) in %.1f s; %d layers, R=%d, cold fraction of picks %.2f, T=%d\n", (unsigned long long)cx.resident_count(),
                cx.arena_bytes() / 1e9, now_s() - t0, nlayers, R, pc, T);
    ram_line("after load");
    Rng rng(21);
    std::vector<double> ms;
    std::vector<float> out((size_t)T * 1536);
    double ncold_picks = 0;
    cx.reset_stats();
    for (int r = -3; r < reps; ++r) {
        const int layer = (r < 0 ? 0 : r) % nlayers;
        Batch b;
        b.T = T;
        b.z.resize((size_t)T * 1536);
        b.w.assign((size_t)T * 16, 0.1f);
        b.idx.resize((size_t)T * 16);
        for (auto& x : b.z) x = (float)rng.gauss();
        for (int t = 0; t < T; ++t) {
            std::set<int> used;
            for (int s = 0; s < 16; ++s) {
                const bool c = rng.uni() < pc && !cold_ids[layer].empty();
                int e;
                do { e = c ? cold_ids[layer][rng.below((int)cold_ids[layer].size())] : res_ids[layer][rng.below((int)res_ids[layer].size())]; } while (used.count(e));
                used.insert(e);
                b.idx[(size_t)t * 16 + s] = e;
                if (r >= 0 && c) ncold_picks += 1;
            }
        }
        if (r == 0) cx.reset_stats();
        const double a = now_s();
        cx.submit(layer, T, b.z.data(), b.idx.data(), b.w.data(), nullptr, out.data());
        cx.wait();
        if (r >= 0) ms.push_back((now_s() - a) * 1e3);
    }
    const CpuExpertStats s = cx.stats();
    std::sort(ms.begin(), ms.end());
    double mean = 0;
    for (double v : ms) mean += v;
    mean /= ms.size();
    std::printf("ms/layer: mean %.3f p50 %.3f min %.3f max %.3f over %d jobs (%.2f cold picks/job)\n", mean, ms[ms.size() / 2], ms[0], ms.back(), reps, ncold_picks / reps);
    std::printf("per job: %.2f cold reads, %.2f MB read, %.2f cache hits, wait %.3f ms, io %.3f ms; SSD while reading %.2f GB/s; avg read rate over wall time %.2f GB/s\n",
                (double)s.cold_reads / reps, s.cold_bytes / 1e6 / reps, (double)s.cold_hits / reps, s.cold_wait_s * 1e3 / reps, s.cold_io_s * 1e3 / reps, s.cold_gbps(),
                s.cold_bytes / (mean * reps * 1e-3) / 1e9);
    print_cold_stats("total", s);
    ram_line("bench end");
    return s.cold_errors ? 1 : 0;
}

// raw pread throughput of the model file: coldio <file> <block MB> <qd> <seq|rand> <direct|buffered> <seconds>
static int test_rawio(const std::string& path, double block_mb, int qd, bool rnd, bool direct, double secs) {
#ifdef _WIN32
    hy4::AlignedFile af;
    if (!af.open(path, direct)) { std::printf("open failed (%s)\n", direct ? "NO_BUFFERING" : "buffered"); return 1; }
    const uint64_t fsize = af.size();
    const size_t blk = ((size_t)(block_mb * 1048576.0) + 4095) & ~(size_t)4095;
    std::atomic<uint64_t> bytes{0}, ops{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    const double t0 = now_s();
    for (int i = 0; i < qd; ++i)
        th.emplace_back([&, i]() {
            void* buf = VirtualAlloc(nullptr, blk, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (!buf) return;
            hy4::AlignedFile tf;
            if (!tf.open(path, direct)) { VirtualFree(buf, 0, MEM_RELEASE); return; }
            Rng r(1000 + i);
            uint64_t pos = ((fsize / 2 / qd) * i) & ~4095ull;
            while (!stop.load()) {
                uint64_t off;
                if (rnd) off = (r.next() % (fsize - blk)) & ~4095ull;
                else { off = pos; pos += blk; if (pos + blk > fsize) pos = 0; }
                const size_t n = tf.read_at(off, buf, blk);
                if (n == 0) break;
                bytes.fetch_add((uint64_t)n);
                ops.fetch_add(1);
            }
            VirtualFree(buf, 0, MEM_RELEASE);
        });
    std::this_thread::sleep_for(std::chrono::milliseconds((int)(secs * 1000)));
    stop.store(true);
    for (auto& t : th) t.join();
    const double dt = now_s() - t0;
    std::printf("rawio %s %s block %.1f MB qd %d: %.2f GB/s, %.0f reads/s (%.1f s)\n", rnd ? "random" : "sequential", direct ? "NO_BUFFERING" : "buffered", block_mb, qd,
                bytes.load() / dt / 1e9, ops.load() / dt, dt);
    return 0;
#else
    const int fd = ::open(path.c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
    if (fd < 0) { std::printf("open failed (%s)\n", direct ? "O_DIRECT" : "buffered"); return 1; }
    const off_t fsize = lseek(fd, 0, SEEK_END);
    const size_t blk = ((size_t)(block_mb * 1048576.0) + 4095) & ~(size_t)4095;
    std::atomic<uint64_t> bytes{0}, ops{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    const double t0 = now_s();
    for (int i = 0; i < qd; ++i)
        th.emplace_back([&, i]() {
            void* buf = nullptr;
            if (posix_memalign(&buf, 4096, blk) != 0) return;
            Rng r(1000 + i);
            off_t pos = (off_t)(((uint64_t)fsize / 2 / qd) * i) & ~(off_t)4095;  // sequential: qd interleaved streams in the first half
            while (!stop.load()) {
                off_t off;
                if (rnd) off = (off_t)((r.next() % (uint64_t)(fsize - (off_t)blk)) & ~(uint64_t)4095);
                else { off = pos; pos += (off_t)blk; if (pos + (off_t)blk > fsize) pos = 0; }
                const ssize_t n = ::pread(fd, buf, blk, off);
                if (n <= 0) break;
                bytes.fetch_add((uint64_t)n);
                ops.fetch_add(1);
            }
            free(buf);
        });
    std::this_thread::sleep_for(std::chrono::milliseconds((int)(secs * 1000)));
    stop.store(true);
    for (auto& t : th) t.join();
    const double dt = now_s() - t0;
    std::printf("rawio %s %s block %.1f MB qd %d: %.2f GB/s, %.0f reads/s (%.1f s)\n", rnd ? "random" : "sequential", direct ? "O_DIRECT" : "buffered", block_mb, qd,
                bytes.load() / dt / 1e9, ops.load() / dt, dt);
    ::close(fd);
    return 0;
#endif
}

// decode-token SSD read rate for a resident rank prefix: p(rank r) ~ 1/(r+8), 16 distinct experts per layer, LRU slot cache
//   coldsim <model> <resident GB> <cache MB> <tokens>
static int test_coldsim(const std::string& path, double res_gb, double cache_mb, int tokens) {
    std::vector<uint32_t> sz;
    std::vector<uint16_t> eor, codes;
    std::string err;
    if (!CpuExperts::read_body_bytes(path, sz, err) || !CpuExperts::read_rank_table(path, eor, codes, err)) { std::printf("%s\n", err.c_str()); return 1; }
    const int nl = (int)(sz.size() / 512);
    // uniform rank cut R over all layers: largest R with resident bytes <= budget
    int R = 0;
    for (int r = 1; r <= 512; ++r) {
        double b = 0;
        for (int l = 0; l < nl; ++l) for (int k = 0; k < r; ++k) b += sz[(size_t)l * 512 + k];
        if (b > res_gb * 1e9) break;
        R = r;
    }
    double res_b = 0, tot_b = 0;
    for (int l = 0; l < nl; ++l) for (int k = 0; k < 512; ++k) { tot_b += sz[(size_t)l * 512 + k]; if (k < R) res_b += sz[(size_t)l * 512 + k]; }
    std::printf("resident budget %.1f GB -> rank cut R=%d of 512 per layer (%.1f GB resident of %.1f GB experts), cold cache %.0f MB\n", res_gb, R, res_b / 1e9, tot_b / 1e9, cache_mb);
    Rng rng(77);
    std::vector<double> wgt(512);
    for (int r = 0; r < 512; ++r) wgt[r] = 1.0 / (r + 8);
    // LRU of cold bodies by (layer, rank), capacity in bytes
    std::map<int, uint64_t> lastuse;  // key -> tick
    std::map<int, uint32_t> size_of;
    uint64_t tick = 0;
    double cap = cache_mb * 1e6, used = 0, cold_total_experts = 0, cold_total_bytes = 0, read_bytes = 0, reads = 0;
    const int warm = tokens / 4;
    for (int tkn = 0; tkn < tokens; ++tkn) {
        double tok_reads = 0, tok_bytes = 0, tok_cold = 0;
        for (int l = 0; l < nl; ++l) {
            // 16 distinct ranks without replacement ~ weights (exponential-key trick)
            std::vector<std::pair<double, int>> keys(512);
            for (int r = 0; r < 512; ++r) keys[r] = {-std::log(rng.uni() + 1e-300) / wgt[r], r};
            std::partial_sort(keys.begin(), keys.begin() + 16, keys.end());
            for (int j = 0; j < 16; ++j) {
                const int r = keys[j].second;
                if (r < R) continue;
                ++tok_cold;
                const int key = l * 512 + r;
                ++tick;
                auto it = lastuse.find(key);
                if (it != lastuse.end()) { it->second = tick; continue; }
                const uint32_t b = sz[(size_t)l * 512 + r];
                while (used + b > cap && !lastuse.empty()) {  // evict LRU
                    auto old = lastuse.begin();
                    for (auto k = lastuse.begin(); k != lastuse.end(); ++k) if (k->second < old->second) old = k;
                    used -= size_of[old->first];
                    lastuse.erase(old);
                }
                if (cap >= b) { lastuse[key] = tick; size_of[key] = b; used += b; }
                tok_reads += 1;
                tok_bytes += b;
            }
        }
        if (tkn >= warm) { cold_total_experts += tok_cold; cold_total_bytes += tok_bytes; reads += tok_reads; read_bytes += tok_bytes; }
    }
    const int n = tokens - warm;
    std::printf("per decode token (%d tokens after warm-up): %.1f cold expert uses, %.1f SSD reads (%.1f%% of cold uses hit the cache), %.0f MB read from SSD\n", n,
                cold_total_experts / n, reads / n, cold_total_experts > 0 ? 100.0 * (1.0 - reads / cold_total_experts) : 0.0, read_bytes / n / 1e6);
    for (double gbps : {1.5, 3.0, 6.0})
        std::printf("  at %.1f GB/s SSD: %.1f ms of reads per token -> at most %.1f tokens/s from the SSD alone\n", gbps, read_bytes / n / (gbps * 1e9) * 1e3, 1.0 / (read_bytes / n / (gbps * 1e9)));
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: %s <model.strata> <num|bench|stress|cold|coldbench|coldio|coldsim> [nlayers] [threads]\n", argv[0]); return 2; }
    const std::string path = argv[1], mode = argv[2];
    const int nl = argc > 3 ? std::atoi(argv[3]) : 10, nth = argc > 4 ? std::atoi(argv[4]) : 6;
    std::printf("isa %s, threads %d\n", strata::quant::sk::active_sk_isa(), nth);
    if (mode == "kernels") {
        std::string rep;
        const bool ok = CpuExperts::selftest_kernels(rep);
        std::printf("%sKERNEL SELFTEST: %s\n", rep.c_str(), ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (mode == "codes") {
        std::vector<uint16_t> eor, codes;
        std::string e;
        if (!CpuExperts::read_rank_table(path, eor, codes, e)) { std::printf("%s\n", e.c_str()); return 1; }
        std::map<uint16_t, int> hist;
        for (auto c : codes) ++hist[c];
        for (auto& kv : hist) std::printf("gu=S%d dn=S%d: %d experts\n", kv.first >> 8, kv.first & 255, kv.second);
        return 0;
    }
    if (mode == "cold") return test_cold(path, nth);
    if (mode == "coldbench") {
        if (argc < 8) { std::printf("usage: coldbench <model> <nlayers> <threads> <R> <pc> <reps> [T]\n"); return 2; }
        return test_coldbench(path, std::min(nl, 41), nth, std::atoi(argv[5]), std::atof(argv[6]), std::atoi(argv[7]), argc > 8 ? std::atoi(argv[8]) : 1);
    }
    if (mode == "coldio") {
        if (argc < 8) { std::printf("usage: coldio <file> <block MB> <qd> <seq|rand> <direct|buffered> <seconds>\n"); return 2; }
        return test_rawio(path, std::atof(argv[3]), std::atoi(argv[4]), std::string(argv[5]) == "rand", std::string(argv[6]) == "direct", std::atof(argv[7]));
    }
    if (mode == "coldsim") {
        if (argc < 6) { std::printf("usage: coldsim <model> <resident GB> <cache MB> <tokens>\n"); return 2; }
        return test_coldsim(path, std::atof(argv[3]), std::atof(argv[4]), std::atoi(argv[5]));
    }
    if (mode == "num") return test_num(path, nth);
    if (mode == "bench") return test_bench(path, std::min(nl, 41), nth);
    if (mode == "stress") return test_stress(path, nth);
    return 2;
}
