// hy4_main.hip.cpp - strata-hy4: Hy4-Flash runner for gfx1151 (Windows HIP)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "hy4_graph.hip.h"

using namespace hy4;

static void top5_line(const float* lg, int V, int pos, std::string& out) {
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    char buf[256];
    snprintf(buf, sizeof buf, "pos %d top5: %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f)\n", pos, idx[0], lg[idx[0]], idx[1], lg[idx[1]], idx[2], lg[idx[2]], idx[3], lg[idx[3]],
             idx[4], lg[idx[4]]);
    out += buf;
}

int main(int argc, char** argv) {
    std::string model, engram, tok_path, out_dir = ".";
    bool load_only = false, dump = false, all_logits = false, no_engram = false, no_index = false, prof = false, prof2 = false;
    int gen = 0, max_ctx = 0, chunk = 512, index_topk = 2048;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model = nxt();
        else if (a == "--engram") engram = nxt();
        else if (a == "--tokens") tok_path = nxt();
        else if (a == "--out") out_dir = nxt();
        else if (a == "--gen") gen = atoi(nxt().c_str());
        else if (a == "--ctx") max_ctx = atoi(nxt().c_str());
        else if (a == "--chunk") chunk = atoi(nxt().c_str());
        else if (a == "--load-only") load_only = true;
        else if (a == "--dump") dump = true;
        else if (a == "--all-logits") all_logits = true;
        else if (a == "--no-engram") no_engram = true;
        else if (a == "--index-topk") index_topk = atoi(nxt().c_str());
        else if (a == "--no-index") no_index = true;
        else if (a == "--prof") prof = true;
        else if (a == "--prof2") prof2 = true;
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty()) {
        fprintf(stderr, "usage: strata-hy4 --model SQ64.strata --engram Engram-Q8.strata --tokens ids.txt [--gen N] [--chunk N] [--ctx N] [--dump] [--all-logits] [--out dir]\n");
        return 2;
    }
    hipDeviceProp_t p;
    HCK(hipGetDeviceProperties(&p, 0));
    size_t fr = 0, tot = 0;
    HCK(hipMemGetInfo(&fr, &tot));
    fprintf(stderr, "[gpu] %s %s free %.1f / %.1f GiB\n", p.name, p.gcnArchName, fr / 1073741824.0, tot / 1073741824.0);
    GpuModel g;
    double t_load0 = now_s();
    if (!load_gpu_model(g, model, true)) return 1;
    fprintf(stderr, "[load] total %.1f s\n", now_s() - t_load0);
    if (load_only) return 0;
    std::vector<int32_t> tok;
    {
        std::ifstream f(tok_path);
        long v;
        while (f >> v) tok.push_back((int32_t)v);
    }
    int T0 = (int)tok.size();
    if (T0 == 0) {
        fprintf(stderr, "no tokens\n");
        return 2;
    }
    if (max_ctx <= 0) max_ctx = T0 + gen + 16;
    Runner r(g, max_ctx, std::max(chunk, 1));
    r.dump = dump;
    r.dump_dir = out_dir;
    r.index_topk = index_topk;
    r.index_on = no_index ? 0 : 1;
    r.prof = prof;
    r.prof2 = prof2;
    std::string err;
    if (!r.init(no_engram ? std::string() : engram, err)) {
        fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    tok.reserve(max_ctx + 8);
    if (prof && r.engram_ready) {
        // random 4 KiB read probe: pages/s with 16 threads x qd 8
        const size_t NP = 20000;
        std::vector<uint64_t> pg(NP);
        uint64_t s = 88172645463325252ull;
        for (auto& v : pg) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            v = r.eio.sec4_offset() / 4096 + (s % r.eio.total_pages());
        }
        uint8_t* buf = (uint8_t*)engram_aligned_alloc(NP * 4096);
        double tp = now_s();
        std::string e2;
        bool okp = r.eio.file().read_pages(pg.data(), NP, buf, 16, 8, e2);
        double dtp = now_s() - tp;
        fprintf(stderr, "[engram-probe] %zu random pages in %.3f s = %.0f pages/s (%.0f MB/s) ok=%d %s\n", NP, dtp, NP / dtp, NP * 4096 / dtp / 1e6, (int)okp, e2.c_str());
        engram_aligned_free(buf);
    }
    std::string top_out;
    // prefill in chunks
    double t0 = now_s();
    int done = 0;
    int V = r.c.vocab;
    r.prefetch_engram(tok.data(), 0, std::min(r.Tc, T0));
    std::vector<float> hl;
    while (done < T0) {
        int T = std::min(r.Tc, T0 - done);
        if (!r.forward_chunk(tok.data(), done, T, err)) {
            fprintf(stderr, "forward: %s\n", err.c_str());
            return 1;
        }
        done += T;
        if (done < T0) r.prefetch_engram(tok.data(), done, std::min(r.Tc, T0 - done));
        bool last = done == T0;
        if (last) {
            int first = all_logits ? 0 : T - 1;
            r.head(T, first);
            HCK(hipStreamSynchronize(r.st)); r.pflush();
            int n = T - first;
            hl.resize((size_t)n * V);
            HCK(hipMemcpy(hl.data(), r.logits, hl.size() * 4, hipMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) top5_line(&hl[(size_t)i * V], V, done - n + i, top_out);
            if (dump) save_npy_f32(out_dir + "/logits.npy", hl.data(), {n, V});
        } else {
            HCK(hipStreamSynchronize(r.st)); r.pflush();
        }
        fprintf(stderr, "[prefill] %d/%d tokens, %.2f s, %.1f tok/s\n", done, T0, now_s() - t0, done / (now_s() - t0));
    }
    double t_pf = now_s() - t0;
    r.prof_report("prefill");
    r.pacc_report("prefill", (T0 + r.Tc - 1) / r.Tc);
    printf("%s", top_out.c_str());
    fprintf(stderr, "[prefill] total %.2f s = %.1f ppts (engram cpu %.2f s)\n", t_pf, T0 / t_pf, r.t_engram_cpu);
    // greedy decode
    if (gen > 0) {
        int next = 0;
        {
            const float* lg = &hl[(size_t)(hl.size() / V - 1) * V];
            next = (int)(std::max_element(lg, lg + V) - lg);
        }
        std::string gen_ids;
        double t1 = now_s();
        std::vector<double> times;
        for (int s = 0; s < gen; ++s) {
            tok.push_back(next);
            gen_ids += std::to_string(next) + " ";
            if (next == r.c.eos) break;
            double ts = now_s();
            int pos = (int)tok.size() - 1;
            if (!r.forward_chunk(tok.data(), pos, 1, err)) {
                fprintf(stderr, "forward: %s\n", err.c_str());
                return 1;
            }
            r.head(1, 0);
            HCK(hipStreamSynchronize(r.st)); r.pflush();
            float* d = r.logits;
            hipLaunchKernelGGL(k_argmax, dim3(1), dim3(1024), 0, r.st, d, V, r.d_argmax);
            int32_t nx = 0;
            HCK(hipMemcpy(&nx, r.d_argmax, 4, hipMemcpyDeviceToHost));
            next = nx;
            times.push_back(now_s() - ts);
        }
        double td = now_s() - t1;
        printf("generated ids: %s\n", gen_ids.c_str());
        r.prof_report("decode");
        r.pacc_report("decode", (int)times.size());
        std::sort(times.begin(), times.end());
        fprintf(stderr, "[decode] %zu tokens in %.2f s = %.2f tok/s (median step %.1f ms)\n", times.size(), td, times.size() / td, times.empty() ? 0.0 : times[times.size() / 2] * 1000);
        std::ofstream f(out_dir + "/gen_ids.txt");
        f << gen_ids << "\n";
    }
    {
        std::ofstream f(out_dir + "/top5.txt");
        f << top_out;
    }
    {
        size_t f2 = 0, t2 = 0;
        HCK(hipMemGetInfo(&f2, &t2));
        fprintf(stderr, "[gpu] end: used %.2f GiB of %.2f GiB (free %.2f)\n", (t2 - f2) / 1073741824.0, t2 / 1073741824.0, f2 / 1073741824.0);
    }
    auto s = r.eio.stats();
    fprintf(stderr, "[engram] lookups %llu seed %llu lru %llu page reads %llu (%.1f MB) io %.2f s\n", (unsigned long long)s.lookups, (unsigned long long)s.seed_hits,
            (unsigned long long)s.lru_hits, (unsigned long long)s.page_reads, s.bytes_read / 1e6, s.io_seconds);
    return 0;
}
