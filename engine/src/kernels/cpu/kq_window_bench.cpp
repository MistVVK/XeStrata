// src/kernels/cpu/kq_window_bench.cpp - round 0216: the CPU pool's share of a decode window, replayed without a GPU.
//
// kq_cpu_bench times one format pair; a decode window is 48 layers back to back, each layer a different format pair, ~12 missed experts a layer
// of 1-4 token rows each, split over the pool's threads with a join between gate/up and down.  This runs that shape on the engine's own entry points
// (native_gu_rows + native_down_rows) over DRAM-sized expert memory: for every layer the pair is that layer's (native_experts.txt of the pack), the
// number of distinct missed experts and each one's token rows are drawn to match what the engine logs (STRATA_DECODE_TIMING: "CPU experts X (Y entries)"),
// and the experts are picked at random from a per-pair arena of several GB, so every expert is a cold read.  Output: ms per window and per layer.
//
//   kq_window_bench --layers PACK/native_experts.txt [--windows 60] [--threads 8] [--distinct 11.6] [--rows 1.7] [--arena-mb 2048] [--reps 3]
//                   [--min-experts 64] [--idle-us G --static --pf --pf-recall R --pf-extra X] [--pin] [--check-split]
//
// --idle-us/--pf/--static: the STRATA_L3_PF shape (a GPU share of G us before each layer, during which each pinned thread
// reads its static slice of the predicted experts).  --check-split: each pair's rows whole and in 15 slices, compared bitwise.
//
// It is a replay with synthetic routing calibrated on observed means, not an engine run: it says what the kernels do to the pool's time, not what a
// request's tokens/s do.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int64_t kH = 2560, kFF = 640;
constexpr int kMaxT = 8;

struct Pair {
    cpu::NativeFmt f;
    std::unique_ptr<uint8_t[]> mem;
    uint8_t* blobs = nullptr;
    size_t E = 0;
};

struct Barrier {   // spinning sense-reversing barrier: the pool's workers spin briefly between phases too
    explicit Barrier(int n) : n_(n) {}
    void wait() {
        const int s = sense_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
            count_.store(0, std::memory_order_relaxed);
            sense_.fetch_add(1, std::memory_order_release);
        } else {
            while (sense_.load(std::memory_order_acquire) == s) { }
        }
    }
    int n_;
    std::atomic<int> count_{0}, sense_{0};
};

struct Task { int pair; size_t blob; int nt; };

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string layers_file, dump_trace;
    int windows = 60, threads = 8, reps = 3;
    double distinct = 11.6, rows_mean = 1.7, arena_mb = 2048;
    unsigned seed = 7;
    // the L3 prefetch prototype: --idle-us G spins G us before each layer (the GPU's share of a layer); --pf reads the predicted
    // experts' rows during it, each thread the rows it computes (--static: thread t computes slice t of every expert);
    // --pf-recall R: each routed expert is predicted with probability R, --pf-extra X: Poisson(X) wrong experts per layer
    double idle_us = 0, pf_recall = 0, pf_extra = 0;
    bool pf = false, fixed = false, pin = false, check_split = false;
    size_t min_experts = 64;   // per pair; a smaller arena (with --min-experts) is cache-resident
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--layers" && more) layers_file = argv[++i];
        else if (a == "--windows" && more) windows = std::atoi(argv[++i]);
        else if (a == "--threads" && more) threads = std::atoi(argv[++i]);
        else if (a == "--distinct" && more) distinct = std::atof(argv[++i]);
        else if (a == "--rows" && more) rows_mean = std::atof(argv[++i]);
        else if (a == "--arena-mb" && more) arena_mb = std::atof(argv[++i]);
        else if (a == "--reps" && more) reps = std::atoi(argv[++i]);
        else if (a == "--dump-trace" && more) dump_trace = argv[++i];
        else if (a == "--seed" && more) seed = (unsigned) std::atoi(argv[++i]);
        else if (a == "--idle-us" && more) idle_us = std::atof(argv[++i]);
        else if (a == "--pf-recall" && more) pf_recall = std::atof(argv[++i]);
        else if (a == "--pf-extra" && more) pf_extra = std::atof(argv[++i]);
        else if (a == "--pf") pf = true;
        else if (a == "--static") fixed = true;
        else if (a == "--pin") pin = true;
        else if (a == "--check-split") check_split = true;
        else if (a == "--min-experts" && more) min_experts = (size_t) std::max(1, std::atoi(argv[++i]));
        else { std::fprintf(stderr, "usage: kq_window_bench --layers native_experts.txt [--windows 60] [--threads 8] [--distinct 11.6] [--rows 1.7] [--arena-mb 2048] [--reps 3]\n"); return 2; }
    }
    if (layers_file.empty()) { std::fprintf(stderr, "kq_window_bench: --layers is required\n"); return 2; }
    ggml_cpu_init();
    // layer -> pair
    std::vector<int> layer_pair;
    std::vector<std::pair<int, int>> pair_types;
    {
        std::FILE* fp = std::fopen(layers_file.c_str(), "r");
        if (!fp) { std::fprintf(stderr, "cannot open %s\n", layers_file.c_str()); return 2; }
        char line[512];
        while (std::fgets(line, sizeof line, fp)) {
            int l, g, d;
            if (std::sscanf(line, "%d %d %d", &l, &g, &d) == 3) {
                auto it = std::find(pair_types.begin(), pair_types.end(), std::make_pair(g, d));
                int idx = (int) (it - pair_types.begin());
                if (it == pair_types.end()) pair_types.push_back({g, d});
                if ((int) layer_pair.size() <= l) layer_pair.resize((size_t) l + 1, 0);
                layer_pair[(size_t) l] = idx;
            }
        }
        std::fclose(fp);
    }
    const int L = (int) layer_pair.size();
    std::vector<Pair> pairs(pair_types.size());
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> row((size_t) std::max(kH, kFF));
    const std::vector<float> imatrix((size_t) std::max(kH, kFF), 1.0f);
    for (size_t p = 0; p < pairs.size(); ++p) {
        std::string err;
        Pair& P = pairs[p];
        if (!cpu::native_fmt(pair_types[p].first, pair_types[p].second, kH, kFF, P.f, err)) { std::fprintf(stderr, "pair %zu: %s\n", p, err.c_str()); return 2; }
        ggml_quantize_init((ggml_type) P.f.gu_type);
        ggml_quantize_init((ggml_type) P.f.d_type);
        P.E = std::max<size_t>(min_experts, (size_t) (arena_mb * 1048576.0) / P.f.bytes);
        P.mem.reset(new uint8_t[P.E * P.f.bytes + 4096]);
        P.blobs = (uint8_t*) (((uintptr_t) P.mem.get() + 4095) & ~(uintptr_t) 4095);
        const size_t U = std::min<size_t>(16, P.E);
        auto quant = [&](int type, uint8_t* dst, int64_t n) {
            ggml_quantize_chunk((ggml_type) type, row.data(), dst, 0, 1, n, ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
        };
        for (size_t e = 0; e < U; ++e) {
            uint8_t* b = P.blobs + e * P.f.bytes;
            for (int64_t r = 0; r < 2 * kFF; ++r) { for (int64_t i = 0; i < kH; ++i) row[(size_t) i] = nd(rng); quant(P.f.gu_type, b + (size_t) r * P.f.gu_row, kH); }
            for (int64_t r = 0; r < kH; ++r) { for (int64_t i = 0; i < kFF; ++i) row[(size_t) i] = nd(rng); quant(P.f.d_type, b + P.f.down_off + (size_t) r * P.f.d_row, kFF); }
        }
        for (size_t e = U; e < P.E; ++e) std::memcpy(P.blobs + e * P.f.bytes, P.blobs + (e % U) * P.f.bytes, P.f.bytes);
    }
    // activations per pair (kMaxT tokens), quantized the way the pool does
    struct Act { std::vector<std::vector<uint8_t>> x, h; };
    std::vector<Act> acts(pairs.size());
    std::normal_distribution<float> xd(0.f, 1.f);
    for (size_t p = 0; p < pairs.size(); ++p) {
        acts[p].x.assign(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes));
        acts[p].h.assign(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes));
        for (int t = 0; t < kMaxT; ++t) {
            std::vector<float> x(kH), h(kFF);
            for (auto& v : x) v = xd(rng);
            for (auto& v : h) v = 0.3f * xd(rng);
            cpu::native_quant_act(pairs[p].f, x.data(), acts[p].x[(size_t) t].data());
            cpu::native_quant_h(pairs[p].f, h.data(), acts[p].h[(size_t) t].data());
        }
    }
    if (check_split) {   // the rows of one expert computed whole and in 15 slices must be the same bits
        int bad = 0;
        for (size_t p = 0; p < pairs.size(); ++p)
            for (int nt = 1; nt <= 4; ++nt) {
                Pair& P = pairs[p];
                std::vector<float> fa((size_t) kMaxT * kFF), fb((size_t) kMaxT * kFF), oa((size_t) kMaxT * kH), ob((size_t) kMaxT * kH);
                float* ffa[kMaxT]; float* ffb[kMaxT]; float* ooa[kMaxT]; float* oob[kMaxT];
                const void* ap[kMaxT]; const void* hp[kMaxT];
                for (int t = 0; t < kMaxT; ++t) { ffa[t] = &fa[(size_t) t * kFF]; ffb[t] = &fb[(size_t) t * kFF]; ooa[t] = &oa[(size_t) t * kH]; oob[t] = &ob[(size_t) t * kH];
                                                  ap[t] = acts[p].x[(size_t) t].data(); hp[t] = acts[p].h[(size_t) t].data(); }
                cpu::native_gu_rows(P.f, P.blobs, ap, nt, ffa, 0, (int) kFF);
                cpu::native_down_rows(P.f, P.blobs, hp, nt, ooa, 0, (int) kH);
                for (int j = 0; j < 15; ++j) {
                    cpu::native_gu_rows(P.f, P.blobs, ap, nt, ffb, (int) (kFF * j / 15), (int) (kFF * (j + 1) / 15));
                    cpu::native_down_rows(P.f, P.blobs, hp, nt, oob, (int) (kH * j / 15), (int) (kH * (j + 1) / 15));
                }
                bool same_rows = true;
                for (int t = 0; t < nt; ++t)
                    same_rows = same_rows && std::memcmp(ffa[t], ffb[t], sizeof(float) * kFF) == 0 && std::memcmp(ooa[t], oob[t], sizeof(float) * kH) == 0;
                std::printf("split check pair %s/%s, %d token(s): %s\n", ggml_type_name((ggml_type) P.f.gu_type),
                            ggml_type_name((ggml_type) P.f.d_type), nt, same_rows ? "same bits" : "DIFFERENT");
                bad += !same_rows;
            }
        return bad ? 1 : 0;
    }
    // the windows: per layer a list of (pair, blob, rows)
    std::mt19937 wr(seed);
    std::vector<std::vector<std::vector<Task>>> win((size_t) windows, std::vector<std::vector<Task>>((size_t) L));
    const double q = 1.0 - 1.0 / rows_mean;   // rows - 1 ~ geometric: mean rows = 1 / (1 - q)
    long long n_exp = 0, n_rows = 0;
    for (int w = 0; w < windows; ++w)
        for (int l = 0; l < L; ++l) {
            std::poisson_distribution<int> pd(distinct);
            const int ne = std::max(1, std::min(24, pd(wr)));
            for (int e = 0; e < ne; ++e) {
                int nt = 1;
                std::uniform_real_distribution<double> u(0.0, 1.0);
                while (nt < 4 && u(wr) < q) ++nt;
                const int p = layer_pair[(size_t) l];
                win[(size_t) w][(size_t) l].push_back({p, (size_t) (wr() % pairs[(size_t) p].E), nt});
                ++n_exp;
                n_rows += nt;
            }
        }
    // per window and layer: the blobs the prefetch reads (a share of the routed ones, plus wrong ones)
    std::vector<std::vector<std::vector<size_t>>> pred((size_t) windows, std::vector<std::vector<size_t>>((size_t) L));
    long long n_pred = 0, n_hit = 0;
    {
        std::mt19937 pr(seed + 99);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::poisson_distribution<int> px(pf_extra > 0 ? pf_extra : 1.0);
        for (int w = 0; w < windows; ++w)
            for (int l = 0; l < L; ++l) {
                auto& pv = pred[(size_t) w][(size_t) l];
                for (const Task& t : win[(size_t) w][(size_t) l])
                    if (u(pr) < pf_recall) { pv.push_back(t.blob); ++n_hit; }
                const int nx = pf_extra > 0 ? px(pr) : 0;
                for (int x = 0; x < nx; ++x) pv.push_back((size_t) (pr() % pairs[(size_t) layer_pair[(size_t) l]].E));
                n_pred += (long long) pv.size();
            }
    }
    if (!dump_trace.empty()) {   // the synthetic windows in the STRATA_ROUTE_DUMP text form (every expert a miss), for replay_place.py
        std::FILE* tf = std::fopen(dump_trace.c_str(), "w");
        if (tf) {
            for (int w = 0; w < windows; ++w)
                for (int l = 0; l < L; ++l) {
                    const auto& tk = win[(size_t) w][(size_t) l];
                    int rows = 0;
                    for (const Task& t : tk) rows += t.nt;
                    std::fprintf(tf, "%d %d %d %d |", l, 3, (int) tk.size(), (int) tk.size());
                    for (size_t i = 0; i < tk.size(); ++i) std::fprintf(tf, " %d:%d:0", (int) i, tk[i].nt);
                    std::fputc(10, tf);
                }
            std::fclose(tf);
        }
    }
    std::printf("kq_window_bench: %s, %d layers, %zu format pairs, %d windows, %d threads, %.0f MB arena per pair; %.2f experts/layer, %.2f rows/expert\n",
                cpu::cpu_name().c_str(), L, pairs.size(), windows, threads, arena_mb, (double) n_exp / ((double) windows * L), (double) n_rows / (double) n_exp);
    for (size_t p = 0; p < pairs.size(); ++p) {
        int nl = 0;
        for (int l = 0; l < L; ++l) nl += layer_pair[(size_t) l] == (int) p;
        std::printf("  pair %s/%s: %d layers\n", ggml_type_name((ggml_type) pairs[p].f.gu_type), ggml_type_name((ggml_type) pairs[p].f.d_type), nl);
    }
    // the run: every thread takes tasks of a layer phase from a shared counter (or, --static, slice `tid` of every expert),
    // a barrier joins the phases
    Barrier bar(threads);
    std::atomic<int> next{0};
    std::atomic<int> cur_w{0};
    std::vector<double> ms;
    std::atomic<long long> pf_lines{0}, pf_cut{0};
    constexpr int kGuChunk = 160, kDnChunk = 640;   // rows per task: gate/up 640 rows in 4, down 2560 rows in 4
    if (pf || idle_us > 0)
        std::printf("  idle %.0f us/layer, prefetch %s, %s rows, predicted %.2f/layer (%.2f routed)\n", idle_us, pf ? "on" : "off",
                    fixed ? "static" : "claimed", (double) n_pred / ((double) windows * L), (double) n_hit / ((double) windows * L));
    for (int rep = 0; rep < reps; ++rep) {
        double wall = 0, idle_wall = 0;
        auto work = [&](int tid) {
            if (pin || fixed) {
#if defined(__linux__)
                cpu_set_t cs;
                CPU_ZERO(&cs);
                CPU_SET(tid, &cs);
                pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
#endif
            }
            std::vector<float> ffb((size_t) kMaxT * kFF), outb((size_t) kMaxT * kH);
            float* ff[kMaxT];
            float* out[kMaxT];
            for (int t = 0; t < kMaxT; ++t) { ff[t] = &ffb[(size_t) t * kFF]; out[t] = &outb[(size_t) t * kH]; }
            uint64_t acc = 0;
            double idle_ms = 0;
            bar.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int w = 0; w < windows; ++w) {
                for (int l = 0; l < L; ++l) {
                    const auto& tasks = win[(size_t) w][(size_t) l];
                    const int nT = (int) tasks.size();
                    if (idle_us > 0) {
                        bar.wait();
                        const auto i0 = std::chrono::steady_clock::now();
                        const auto dl = i0 + std::chrono::nanoseconds((int64_t) (idle_us * 1000.0));
                        if (pf) {
                            Pair& P = pairs[(size_t) layer_pair[(size_t) l]];
                            // this thread's slices: gate+up rows [g0,g1) of FF (and the up half at up_off), down rows [d0,d1) of H
                            const int64_t g0 = kFF * tid / threads, g1 = kFF * (tid + 1) / threads;
                            const int64_t d0 = kH * tid / threads, d1 = kH * (tid + 1) / threads;
                            bool cut = false;
                            for (size_t b : pred[(size_t) w][(size_t) l]) {
                                if (std::chrono::steady_clock::now() >= dl) { cut = true; break; }
                                const uint8_t* blob = P.blobs + b * P.f.bytes;
                                auto touch = [&](const uint8_t* a, size_t n) {
                                    for (size_t i = 0; i < n; i += 64) acc += *(const volatile uint8_t*) (a + i);
                                    pf_lines.fetch_add((long long) (n / 64), std::memory_order_relaxed);
                                };
                                touch(blob + (size_t) g0 * P.f.gu_row, (size_t) (g1 - g0) * P.f.gu_row);
                                touch(blob + P.f.up_off + (size_t) g0 * P.f.gu_row, (size_t) (g1 - g0) * P.f.gu_row);
                                touch(blob + P.f.down_off + (size_t) d0 * P.f.d_row, (size_t) (d1 - d0) * P.f.d_row);
                            }
                            if (cut) pf_cut.fetch_add(1, std::memory_order_relaxed);
                        }
                        while (std::chrono::steady_clock::now() < dl) { }
                        idle_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - i0).count();
                    }
                    for (int phase = 0; phase < 2; ++phase) {
                        if (tid == 0) next.store(0, std::memory_order_relaxed);
                        bar.wait();
                        const int rows = phase == 0 ? (int) kFF : (int) kH;
                        auto run_rows = [&](const Task& tk, int r0, int r1) {
                            Pair& P = pairs[(size_t) tk.pair];
                            const uint8_t* blob = P.blobs + tk.blob * P.f.bytes;
                            const void* ap[kMaxT];
                            const void* hp[kMaxT];
                            for (int t = 0; t < tk.nt; ++t) { ap[t] = acts[(size_t) tk.pair].x[(size_t) t].data(); hp[t] = acts[(size_t) tk.pair].h[(size_t) t].data(); }
                            if (phase == 0) cpu::native_gu_rows(P.f, blob, ap, tk.nt, ff, r0, r1);
                            else cpu::native_down_rows(P.f, blob, hp, tk.nt, out, r0, r1);
                        };
                        if (fixed) {
                            const int r0 = (int) ((int64_t) rows * tid / threads), r1 = (int) ((int64_t) rows * (tid + 1) / threads);
                            if (r1 > r0)
                                for (const Task& tk : tasks) run_rows(tk, r0, r1);
                        } else {
                            const int chunk = phase == 0 ? kGuChunk : kDnChunk;
                            const int per = (rows + chunk - 1) / chunk;
                            for (;;) {
                                const int k = next.fetch_add(1, std::memory_order_relaxed);
                                if (k >= nT * per) break;
                                const Task& tk = tasks[(size_t) (k / per)];
                                const int r0 = (k % per) * chunk, r1 = std::min(rows, r0 + chunk);
                                run_rows(tk, r0, r1);
                            }
                        }
                        bar.wait();
                    }
                }
            }
            if (tid == 0) {
                wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                idle_wall = idle_ms;
            }
            volatile float sink = outb[0] + ffb[0] + (float) (acc & 1);
            (void) sink;
        };
        std::vector<std::thread> th;
        for (int t = 1; t < threads; ++t) th.emplace_back(work, t);
        work(0);
        for (auto& x : th) x.join();
        const double comp = (wall - idle_wall) / windows;
        ms.push_back(comp);
        std::printf("  rep %d: %.2f ms per window computing (%.3f ms per layer), %.2f ms idle\n", rep, comp, comp / L, idle_wall / windows);
    }
    if (pf) std::printf("  prefetch cut short by the idle end in %.1f%% of thread-layers\n",
                        100.0 * (double) pf_cut.load() / ((double) reps * windows * L * threads));
    std::sort(ms.begin(), ms.end());
    std::printf("median %.2f ms per window\n", ms[ms.size() / 2]);
    (void) cur_w;
    return 0;
}
