// src/kernels/cpu/bwg_pool_bench.cpp - the CPU pool's decode-window work on UD-Q4_K_XL shaped experts, without a GPU or ggml.
//
// Q4_K gate/up (2560 -> 640) and Q5_1 / Q8_0 down (640 -> 2560) experts in one big arena, a layer call picks ~7-8 distinct experts of 1-3 tokens at random and
// runs them through the pool's own run_split_multi_native (the engine's entry point), so every expert is a cold read like the engine's.  Modes:
//   bwg_pool_bench [--workers 22] [--tasks 0] [--host 1] [--gu 12] [--down 7] [--arena-mb 6000] [--calls 3000] [--warm 300] [--distinct 7.7] [--thp 0|1]
//                  [--nt1 0.78] [--seed 1] [--pin 1] [--sum 1]
//   bwg_pool_bench --direct [--arena-mb N] [--calls N]      one thread, native_gu_rows + native_down_rows over whole experts (hot when the arena is small)
// Prints ms per layer call (median, mean) and, with --sum, an FNV hash of the outputs of the first calls so two builds / switch settings can be compared bit for bit.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <malloc.h>
#endif
#include <atomic>
#include <immintrin.h>
#include <memory>
#include <thread>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <sys/mman.h>
#endif

namespace cpu = strata::kernels::cpu;

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 12345) {}
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    double uni() { return (double) (next() >> 11) / 9007199254740992.0; }
};

size_t block_bytes(int type) {
    switch (type) {
        case 12: return 144;   // Q4_K
        case 13: return 176;   // Q5_K
        case 14: return 210;   // Q6_K
        case 7: return 24;     // Q5_1
        case 8: return 34;     // Q8_0
        case 6: return 22;     // Q5_0
        default: return 0;
    }
}
int block_values(int type) { return type == 12 || type == 13 || type == 14 ? 256 : 32; }

// scales / mins of every block get small finite f16 values (0x2e66 = ~0.1); the rest stays random
void fix_scales(uint8_t* base, size_t rows, size_t row_bytes, int type, int n) {
    const size_t bb = block_bytes(type);
    const size_t nb = (size_t) n / block_values(type);
    for (size_t r = 0; r < rows; ++r)
        for (size_t k = 0; k < nb; ++k) {
            uint8_t* d = base + r * row_bytes + k * bb;
            d[0] = 0x8e; d[1] = 0x06;
            if (type == 12 || type == 13 || type == 7) { d[2] = 0x8e; d[3] = 0x06; }
        }
}

#if defined(__GNUC__) && !defined(_MSC_VER)
#define BWG_AVX2 __attribute__((target("avx2")))
#else
#define BWG_AVX2
#endif
BWG_AVX2 uint32_t read_expert(const uint8_t* blob, size_t gu, size_t dn, size_t up_off, size_t down_off, int n_ff, int n_embd) {
    __m256i acc = _mm256_setzero_si256();
    for (int row = 0; row < n_ff; ++row) {
        const uint8_t* g = blob + (size_t) row * gu;
        const uint8_t* u = blob + up_off + (size_t) row * gu;
        for (size_t o = 0; o + 32 <= gu; o += 32) {
            acc = _mm256_add_epi8(acc, _mm256_loadu_si256((const __m256i*) (g + o)));
            acc = _mm256_add_epi8(acc, _mm256_loadu_si256((const __m256i*) (u + o)));
        }
    }
    for (int row = 0; row < n_embd; ++row) {
        const uint8_t* d = blob + down_off + (size_t) row * dn;
        for (size_t o = 0; o + 32 <= dn; o += 32) acc = _mm256_add_epi8(acc, _mm256_loadu_si256((const __m256i*) (d + o)));
    }
    return (uint32_t) _mm256_extract_epi32(acc, 0);
}

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

uint64_t fnv(const void* p, size_t n, uint64_t h) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    int readonly = 0, mt = 0, workers = 22, tasks = 0, host = 1, gu = 12, down = 7, calls = 3000, warm = 300, thp = 0, pin = 1, direct = 0, sum = 0;
    double arena_mb = 6000, distinct = 7.7, nt1 = 0.78, spread = 6.0;
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : "0"; };
        if (a == "--workers") workers = std::atoi(val());
        else if (a == "--tasks") tasks = std::atoi(val());
        else if (a == "--host") host = std::atoi(val());
        else if (a == "--gu") gu = std::atoi(val());
        else if (a == "--down") down = std::atoi(val());
        else if (a == "--arena-mb") arena_mb = std::atof(val());
        else if (a == "--calls") calls = std::atoi(val());
        else if (a == "--warm") warm = std::atoi(val());
        else if (a == "--distinct") distinct = std::atof(val());
        else if (a == "--thp") thp = std::atoi(val());
        else if (a == "--nt1") nt1 = std::atof(val());
        else if (a == "--spread") spread = std::atof(val());
        else if (a == "--seed") seed = (uint64_t) std::atoll(val());
        else if (a == "--pin") pin = std::atoi(val());
        else if (a == "--sum") sum = std::atoi(val());
        else if (a == "--direct") direct = 1;
        else if (a == "--mt") mt = std::atoi(val());
        else if (a == "--readonly") readonly = std::atoi(val());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (!cpu::cpu_avx2_ok()) { std::printf("bwg_pool_bench: no AVX2, skipped\n"); return 0; }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu, down, cpu::H, cpu::FF, f, err) || block_bytes(gu) == 0 || block_bytes(down) == 0) {
        std::fprintf(stderr, "format %d/%d: %s\n", gu, down, err.c_str());
        return 1;
    }
    const size_t ebytes = (f.bytes + 4095) & ~(size_t) 4095;
    size_t E = (size_t) (arena_mb * 1024.0 * 1024.0 / (double) ebytes);
    if (E < 16 && !direct && mt == 0) E = 16;
    if (E < 1) E = 1;
    std::fprintf(stderr, "format %d/%d: expert %zu bytes (gate/up row %zu, down row %zu), arena %zu experts = %.0f MiB, workers %d host %d tasks %d\n", gu, down,
                 f.bytes, f.gu_row, f.d_row, E, (double) (E * ebytes) / 1048576.0, workers, host, tasks);
    const size_t arena_bytes = ((E * ebytes) + (2u << 20) - 1) & ~(size_t) ((2u << 20) - 1);
#if defined(_WIN32)
    uint8_t* arena = (uint8_t*) _aligned_malloc(arena_bytes, 2 * 1024 * 1024);
#else
    uint8_t* arena = (uint8_t*) std::aligned_alloc(2 * 1024 * 1024, arena_bytes);
#endif
    if (arena == nullptr) { std::fprintf(stderr, "no memory\n"); return 1; }
#if defined(__linux__)
    if (thp) madvise(arena, E * ebytes, MADV_HUGEPAGE);
#else
    (void) thp;
#endif
    {
        Rng r(seed);
        uint64_t* w = (uint64_t*) arena;
        for (size_t i = 0; i < E * ebytes / 8; ++i) w[i] = r.next();
        for (size_t e = 0; e < E; ++e) {
            uint8_t* b = arena + e * ebytes;
            fix_scales(b, (size_t) cpu::FF, f.gu_row, gu, (int) f.n_embd);
            fix_scales(b + f.up_off, (size_t) cpu::FF, f.gu_row, gu, (int) f.n_embd);
            fix_scales(b + f.down_off, (size_t) cpu::H, f.d_row, down, (int) f.n_ff);
        }
    }
    std::vector<std::vector<uint8_t>> act((size_t) cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<cpu::ActQ> qact((size_t) cpu::MAXT);
    std::vector<float> x((size_t) cpu::H);
    for (int t = 0; t < cpu::MAXT; ++t) {
        for (int i = 0; i < cpu::H; ++i) x[(size_t) i] = std::sin((float) (i + 13 * t) * 0.031f);
        cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
        cpu::act_quant_any(x.data(), cpu::H, qact[(size_t) t]);
    }

    Rng rng(seed + 99);
    if (direct) {
        std::vector<std::vector<float>> ffb((size_t) cpu::MAXT, std::vector<float>((size_t) cpu::FF));
        std::vector<std::vector<float>> outb((size_t) cpu::MAXT, std::vector<float>((size_t) cpu::H));
        std::vector<double> tg, td;
        uint64_t h = 1469598103934665603ull;
        for (int c = 0; c < calls + warm; ++c) {
            const uint8_t* blob = arena + (size_t) (rng.next() % E) * ebytes;
            const void* ap[1] = {act[0].data()};
            float* fp[1] = {ffb[0].data()};
            const double a = now_ms();
            cpu::native_gu_rows(f, blob, ap, 1, fp, 0, (int) f.n_ff);
            const double b = now_ms();
            cpu::native_quant_h(f, ffb[0].data(), act[1].data());
            const double b2 = now_ms();
            const void* hp[1] = {act[1].data()};
            float* op[1] = {outb[0].data()};
            cpu::native_down_rows(f, blob, hp, 1, op, 0, (int) f.n_embd);
            const double d = now_ms();
            if (c >= warm) { tg.push_back(b - a); td.push_back(d - b2); }
            if (sum && c < 64) { h = fnv(ffb[0].data(), sizeof(float) * (size_t) cpu::FF, h); h = fnv(outb[0].data(), sizeof(float) * (size_t) cpu::H, h); }
        }
        std::sort(tg.begin(), tg.end());
        std::sort(td.begin(), td.end());
        std::printf("direct 1 thread, 1 token: gate/up %.1f us (median), down %.1f us (median), expert %.1f us = %.2f GB/s%s\n", tg[tg.size() / 2] * 1000.0,
                    td[td.size() / 2] * 1000.0, (tg[tg.size() / 2] + td[td.size() / 2]) * 1000.0,
                    (double) f.bytes / ((tg[tg.size() / 2] + td[td.size() / 2]) * 1e-3) / 1e9, sum ? "" : "");
        if (sum) std::printf("hash %016llx\n", (unsigned long long) h);
        return 0;
    }

    if (mt > 0) {
        // `mt` independent threads (no pool, no phases): each runs whole experts (gate/up, quantize, down) over random cold experts; the aggregate expert bytes per second
        std::atomic<int> ready{0}, go{0};
        std::vector<std::thread> th;
        std::vector<double> secs((size_t) mt, 0.0);
        const int per = calls;
        for (int i = 0; i < mt; ++i)
            th.emplace_back([&, i] {
#if defined(__linux__)
                if (pin) {
                    cpu_set_t s;
                    CPU_ZERO(&s);
                    CPU_SET(1 + i, &s);
                    sched_setaffinity(0, sizeof s, &s);
                }
#endif
                std::vector<uint8_t> a0(cpu::kNativeActBytes), a1(cpu::kNativeActBytes);
                std::memcpy(a0.data(), act[0].data(), a0.size());
                std::vector<float> ffv((size_t) cpu::FF), ov((size_t) cpu::H);
                Rng r(seed + 1000 + (uint64_t) i);
                ready.fetch_add(1);
                while (go.load() == 0) {}
                volatile uint32_t sink = 0;
                const double t0 = now_ms();
                for (int c = 0; c < per; ++c) {
                    const uint8_t* blob = arena + (size_t) (r.next() % E) * ebytes;
                    if (readonly) {   // the same bytes in the same order (gate row, up row, ..., down rows), only summed: what the memory system alone costs
                        sink ^= read_expert(blob, f.gu_row, f.d_row, f.up_off, f.down_off, (int) f.n_ff, (int) f.n_embd);
                        continue;
                    }
                    const void* ap[1] = {a0.data()};
                    float* fp[1] = {ffv.data()};
                    cpu::native_gu_rows(f, blob, ap, 1, fp, 0, (int) f.n_ff);
                    cpu::native_quant_h(f, ffv.data(), a1.data());
                    const void* hp[1] = {a1.data()};
                    float* op[1] = {ov.data()};
                    cpu::native_down_rows(f, blob, hp, 1, op, 0, (int) f.n_embd);
                }
                secs[(size_t) i] = (now_ms() - t0) / 1000.0;
            });
        while (ready.load() < mt) {}
        go.store(1);
        for (auto& t : th) t.join();
        double mx = 0;
        for (double s : secs) mx = std::max(mx, s);
        std::printf("mt %d threads x %d experts: %.3f s, %.1f us per expert per thread, aggregate %.1f GB/s\n", mt, per, mx, mx * 1e6 / per,
                    (double) mt * per * (double) f.bytes / mx / 1e9);
        return 0;
    }

    cpu::ExpertPool pool(workers, pin != 0, host != 0, cpu::PoolAffinity::All, tasks);
    std::fprintf(stderr, "%s\n", pool.placement_line().c_str());
    const int maxn = cpu::ExpertPool::kMaxSplitMulti;
    std::vector<cpu::ExpertJobMulti> jobs((size_t) maxn);
    std::vector<float> out((size_t) maxn * cpu::MAXT * cpu::H);
    std::vector<double> ms;
    double tot = 0;
    uint64_t h = 1469598103934665603ull;
    for (int c = 0; c < calls + warm; ++c) {
        int n = (int) std::lround(distinct + (rng.uni() - 0.5) * spread);
        n = std::max(1, std::min(n, maxn));
        for (int e = 0; e < n; ++e) {
            cpu::ExpertJobMulti& j = jobs[(size_t) e];
            j = cpu::ExpertJobMulti();
            j.blob = arena + (size_t) (rng.next() % E) * ebytes;
            const double u = rng.uni();
            j.nt = u < nt1 ? 1 : u < nt1 + (1.0 - nt1) * 0.6 ? 2 : 3;
            for (int t = 0; t < j.nt; ++t) {
                j.nact[t] = act[(size_t) t].data();
                j.act[t] = &qact[(size_t) t];
                j.out[t] = out.data() + ((size_t) e * cpu::MAXT + (size_t) t) * cpu::H;
            }
        }
        const double a = now_ms();
        pool.run_split_multi_native(f, jobs.data(), n);
        const double b = now_ms();
        if (c >= warm) { ms.push_back(b - a); tot += b - a; }
        if (sum && c < 64)
            for (int e = 0; e < n; ++e)
                for (int t = 0; t < jobs[(size_t) e].nt; ++t) h = fnv(jobs[(size_t) e].out[t], sizeof(float) * (size_t) cpu::H, h);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("pool %d workers host %d tasks %d: %d calls, ms per layer call median %.4f mean %.4f p10 %.4f p90 %.4f\n", workers, host, tasks, (int) ms.size(),
                ms[ms.size() / 2], tot / (double) ms.size(), ms[ms.size() / 10], ms[ms.size() * 9 / 10]);
    if (sum) {
        size_t fin = 0, tot = 0;
        double mean = 0;
        for (int e = 0; e < 4; ++e)
            for (int i = 0; i < cpu::H; ++i) {
                const float v = out[((size_t) e * cpu::MAXT) * cpu::H + (size_t) i];
                ++tot;
                if (std::isfinite(v)) { ++fin; mean += std::fabs((double) v); }
            }
        std::printf("last call, experts 0-3 token 0: %zu of %zu outputs finite, mean |out| %.4g\n", fin, tot, fin ? mean / (double) fin : 0.0);
        std::printf("hash %016llx\n", (unsigned long long) h);
    }
    return 0;
}
