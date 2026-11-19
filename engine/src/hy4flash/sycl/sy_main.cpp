// sy_main.cpp - strata-hy4-sycl: Hy4-Flash runner for Intel Arc (SYCL trunk + CPU routed experts), Linux.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "sy_serve.h"

using namespace hy4s;

#ifndef HY4_VERSION
#define HY4_VERSION "0.2.18-dev-sycl"
#endif

// `strata-hy4 --export-tokenizer FILE.strata DIR`: Section 0's tokenizer_json and chat_template to DIR/tokenizer.hf.json and DIR/chat_template.jinja
static int export_tokenizer(const std::string& file, const std::string& dir) {
    strata::artifact::StrataFileReader rd;
    std::string err;
    if (!rd.open(file, err)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    hy4::Manifest m;
    hy4::parse_manifest(rd.manifest_json(), m, true);
    auto put = [&](const std::string& name, const std::string& text) {
        const std::string path = dir + "/" + name;
        FILE* f = fopen(path.c_str(), "wb");
        if (!f || fwrite(text.data(), 1, text.size(), f) != text.size()) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            if (f) fclose(f);
            return false;
        }
        fclose(f);
        return true;
    };
    if (!put("tokenizer.hf.json", m.tokenizer_json) || !put("chat_template.jinja", m.chat_template)) return 1;
    printf("tokenizer: %zu bytes, chat template: %zu bytes\n", m.tokenizer_json.size(), m.chat_template.size());
    return 0;
}

// `strata-hy4 --info FILE.strata`: one line of key=value facts, no GPU needed. drafter= and images= are what THIS engine can do with this file (images: not ported to SYCL yet).
static int print_info(const std::string& file) {
    strata::artifact::StrataFileReader rd;
    std::string err;
    if (!rd.open(file, err)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    hy4::Manifest m;
    hy4::parse_manifest(rd.manifest_json(), m, false);
    printf("arch=hy4flash layers=%d hidden=%d vocab=%d experts=%d eos=%d drafter=%s images=%d backend=sycl\n", m.c.n_layers, m.c.hidden, m.c.vocab, m.c.n_experts, m.c.eos,
           manifest_has_drafter(m) ? "dfly2" : "none", 0);
    return 0;
}

static void top5_line(const float* lg, int V, int pos, std::string& out) {
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    char buf[256];
    snprintf(buf, sizeof buf, "pos %d top5: %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f)\n", pos, idx[0], lg[idx[0]], idx[1], lg[idx[1]], idx[2], lg[idx[2]], idx[3], lg[idx[3]], idx[4],
             lg[idx[4]]);
    out += buf;
}

int main(int argc, char** argv) {
    // Backend: the OpenCL runtime of the Intel compute stack launches small kernels ~4x cheaper than Level Zero on the A750 (4.7 vs 19 us per dependent tiny kernel;
    // a decode step has ~1300 of them: 46 ms vs 87 ms without the routed experts). HY4S_BACKEND=level_zero (or an own ONEAPI_DEVICE_SELECTOR) overrides.
    const bool user_sel = std::getenv("ONEAPI_DEVICE_SELECTOR") != nullptr;
    if (!user_sel) {
        const char* be = std::getenv("HY4S_BACKEND");
        setenv("ONEAPI_DEVICE_SELECTOR", (be && std::string(be) == "level_zero") ? "level_zero:gpu" : "opencl:gpu", 1);
    }
    std::string model, engram, tok_path, out_dir = ".", force_route, draft;  // draft: "" = default (serve: auto, CLI runs: off)
    SpecCfg scfg;
    bool load_only = false, dbg = false, dump = false, all_logits = false, no_engram = false, no_index = false, prof = false, prof2 = false, warm = false;
    bool serve = false, session_reuse = false, vision_requested = false;
    int kv_host_above = 24000, gx_mb = 384, gx_min_t = 32, gen = 0, max_ctx = 0, chunk = 1024, index_topk = 2048, threads = (int)std::max(2u, std::thread::hardware_concurrency() / 2), expert_act = 1;  // CPU expert threads: half the hardware threads (6 on the 5600X)
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--serve") serve = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() {
            if (i + 1 >= argc) {
                printf("ERR %s needs a value\n", a.c_str());
                fflush(stdout);
                exit(2);
            }
            return std::string(argv[++i]);
        };
        if (a == "--version") {
            printf("strata-hy4 %s\n", HY4_VERSION);
            return 0;
        }
        if (a == "--info") return print_info(nxt());
        if (a == "--export-tokenizer") {
            const std::string f = nxt();
            return export_tokenizer(f, nxt());
        }
        if (a == "--serve") continue;
        else if (a == "--native" || a == "--model") model = nxt();
        else if (a == "--max-context") max_ctx = atoi(nxt().c_str());
        else if (a == "--session-reuse") session_reuse = atoi(nxt().c_str()) != 0;
        else if (a == "--gpu") {  // device index of the OpenCL runtime (an own ONEAPI_DEVICE_SELECTOR still wins)
            const std::string n = nxt();
            if (!user_sel) setenv("ONEAPI_DEVICE_SELECTOR", ("opencl:gpu:" + n).c_str(), 1);
        } else if (a == "--vision") {
            nxt();
            vision_requested = true;
        } else if (a == "--vram-experts-gb" || a == "--ram-experts-gb" || a == "--engram-cache-mib") fprintf(stderr, "[args] %s %s: not used by this runner (experts: page cache + GPU ring)\n", a.c_str(), nxt().c_str());
        else if (a == "--kv") {
            const std::string kv = nxt();
            if (kv != "fp16") {
                printf("ERR --kv %s: only fp16\n", kv.c_str());
                fflush(stdout);
                return 2;
            }
        }
        else if (a == "--engram-path") engram = nxt();
        else if (a == "--engram") engram = nxt();
        else if (a == "--tokens") tok_path = nxt();
        else if (a == "--out") out_dir = nxt();
        else if (a == "--gen") gen = atoi(nxt().c_str());
        else if (a == "--ctx") max_ctx = atoi(nxt().c_str());
        else if (a == "--chunk") chunk = atoi(nxt().c_str());
        else if (a == "--threads") threads = atoi(nxt().c_str());
        else if (a == "--expert-act") expert_act = atoi(nxt().c_str());
        else if (a == "--warm") warm = true;
        else if (a == "--gpu-experts") gx_mb = atoi(nxt().c_str());
        else if (a == "--gx-min-t") gx_min_t = atoi(nxt().c_str());
        else if (a == "--kv-host-above") kv_host_above = atoi(nxt().c_str());
        else if (a == "--load-only") load_only = true;
        else if (a == "--dump") dump = true;
        else if (a == "--dbg-layers") dbg = true;
        else if (a == "--force-route") force_route = nxt();
        else if (a == "--all-logits") all_logits = true;
        else if (a == "--no-engram") no_engram = true;
        else if (a == "--index-topk") index_topk = atoi(nxt().c_str());
        else if (a == "--no-index") no_index = true;
        else if (a == "--prof") prof = true;
        else if (a == "--prof2") prof2 = true;
        else if (a == "--draft") draft = nxt();  // off (CLI default) | dfly2
        else if (a == "--spec-k") scfg.K = atoi(nxt().c_str());
        else if (a == "--spec-fixed") scfg.adaptive = false;
        else if (a == "--spec-cut") scfg.cut = (float)atof(nxt().c_str());
        else if (a == "--spec-debug") scfg.debug = true;
        else if (a == "--spec-rv") scfg.rv = atoi(nxt().c_str());
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            if (serve) {
                printf("ERR unknown flag %s\n", a.c_str());
                fflush(stdout);
            }
            return 2;
        }
    }
    if (serve) {
        if (model.empty()) {
            printf("ERR --native FILE.strata is required\n");
            fflush(stdout);
            return 2;
        }
        if (!draft.empty() && draft != "off" && draft != "dfly2" && draft != "auto") {
            printf("ERR --draft %s: expected off, dfly2 or auto\n", draft.c_str());
            fflush(stdout);
            return 2;
        }
        ServeOpts so;
        so.model = model;
        so.engram = engram;
        so.max_ctx = max_ctx > 0 ? max_ctx : 131072;
        so.chunk = std::max(chunk, 1);
        so.index_topk = index_topk;
        so.threads = threads;
        so.gx_mb = gx_mb;
        so.kv_host_above = kv_host_above;
        so.no_index = no_index;
        so.no_engram = no_engram;
        so.session_reuse = session_reuse;
        so.draft = draft.empty() ? "auto" : draft;
        so.vision_requested = vision_requested;
        so.spec = scfg;
        return hy4s_serve_main(so);
    }
    if (draft.empty()) draft = "off";
    if (model.empty()) {
        fprintf(stderr, "usage: strata-hy4-sycl --model SQ32.strata --engram Engram-Q8.strata --tokens ids.txt [--gen N] [--chunk N] [--ctx N] [--threads N] [--expert-act 0|1] [--warm] [--dump] [--all-logits] [--out dir]\n       strata-hy4-sycl --serve --native SQ.strata --engram ENG.strata [--max-context N] [--draft off|dfly2|auto]   (engine line protocol)\n       strata-hy4-sycl --info FILE.strata | --version | --export-tokenizer FILE.strata DIR\n");
        return 2;
    }
    try {
        auto dev = Q().get_device();
        fprintf(stderr, "[gpu] %s, %.2f GiB\n", dev.get_info<sycl::info::device::name>().c_str(), dev.get_info<sycl::info::device::global_mem_size>() / 1073741824.0);
        SyModel g;
        std::string err;
        if (!g.load(model, err)) {
            fprintf(stderr, "load: %s\n", err.c_str());
            return 1;
        }
        CpuExperts cx;
        if (!cx.open(model, threads, err, std::max(chunk, 1))) {
            fprintf(stderr, "experts: %s\n", err.c_str());
            return 1;
        }
        if (warm) { double ws_ = 0; double gbs = cx.warm(threads, &ws_); fprintf(stderr, "[experts] warm %.1f s (%.2f GB/s)\n", ws_, gbs); }
        if (load_only) return 0;
        std::vector<int32_t> tok;
        {
            std::ifstream f(tok_path);
            long v;
            while (f >> v) tok.push_back((int32_t)v);
        }
        const int T0 = (int)tok.size();
        if (T0 == 0) {
            fprintf(stderr, "no tokens\n");
            return 2;
        }
        if (max_ctx <= 0) max_ctx = std::max(T0 + gen + 16, 131072);  // owner default for Hy4-Flash: 128K context on every card (the sparse-layer KV then lives in pinned host memory, see kv_host_above)
        const int Tc = std::max(chunk, 1);
        const bool spec_mode = draft == "dfly2" && gen > 0;
        Runner r(g, cx, max_ctx, Tc, spec_mode ? 8 : (all_logits ? std::min(Tc, T0) : 1));
        r.dump = dump;
        r.dbg_layers = dbg;
        r.force_route_dir = force_route;
        r.dump_dir = out_dir;
        r.index_topk = index_topk;
        r.index_on = no_index ? 0 : 1;
        r.prof = prof;
        r.prof2 = prof2;
        r.expert_act = expert_act;
        r.kv_host_above = kv_host_above;
        if (!r.init(no_engram ? std::string() : engram, err)) {
            fprintf(stderr, "init: %s\n", err.c_str());
            return 1;
        }
        DFly2 df(r);
        if (draft == "dfly2") {
            if (!df.init(err)) {
                fprintf(stderr, "dfly2: %s\n", err.c_str());
                return 1;
            }
            fprintf(stderr, "[dfly2] drafter ready (K=%d%s, rv=%d), device allocated %.2f GiB\n", scfg.K, scfg.adaptive ? " adaptive" : " fixed", scfg.rv, dev_bytes_allocated() / 1073741824.0);
        }
        GpuExperts gxe;
        if (gx_mb > 0 && T0 >= gx_min_t) {
            std::string gerr;
            try {
                if (gxe.init(cx, (size_t)gx_mb << 20, Tc, gerr, threads)) {
                    r.gx = &gxe;
                    r.gx_min_T = gx_min_t;
                    fprintf(stderr, "[gpu-experts] on for chunks >= %d tokens: %.1f MiB device, device allocated %.2f GiB total\n", gx_min_t, gxe.device_bytes() / 1048576.0, (dev_bytes_allocated() + gxe.device_bytes()) / 1073741824.0);
                } else {
                    fprintf(stderr, "[gpu-experts] disabled: %s\n", gerr.c_str());
                }
            } catch (const sycl::exception& e) {
                fprintf(stderr, "[gpu-experts] disabled: %s\n", e.what());
            }
        }
        tok.reserve(max_ctx + 8);
        std::string top_out;
        const double t0 = now_s();
        int done = 0;
        const int V = r.c.vocab;
        r.prefetch_engram(tok.data(), 0, std::min(r.Tc, T0));
        std::vector<float> hl;
        while (done < T0) {
            const int T = std::min(r.Tc, T0 - done);
            if (!r.forward_chunk(tok.data(), done, T, err)) {
                fprintf(stderr, "forward: %s\n", err.c_str());
                return 1;
            }
            done += T;
            if (done < T0) r.prefetch_engram(tok.data(), done, std::min(r.Tc, T0 - done));
            const bool last = done == T0;
            if (last) {
                const int first = (all_logits && !spec_mode) ? std::max(0, T - r.logit_rows) : T - 1;
                r.head(T, first);
                if (df.ok) df.snap_h(T - first - 1);
                const int n = T - first;
                hl.resize((size_t)n * V);
                d2h(hl.data(), r.logits, hl.size() * 4);
                wait();
                for (int i = 0; i < n; ++i) top5_line(&hl[(size_t)i * V], V, done - n + i, top_out);
                if (dump) save_npy_f32(out_dir + "/logits.npy", hl.data(), {n, V});
            } else {
                wait();
            }
            fprintf(stderr, "[prefill] %d/%d tokens, %.2f s, %.1f tok/s\n", done, T0, now_s() - t0, done / (now_s() - t0));
        }
        const double t_pf = now_s() - t0;
        r.prof_report("prefill");
        r.pacc_report("prefill", (T0 + r.Tc - 1) / r.Tc);
        printf("%s", top_out.c_str());
        fprintf(stderr, "[prefill] total %.2f s = %.1f ppts (engram cpu %.2f s, cpu experts %.2f s)\n", t_pf, T0 / t_pf, r.t_engram_cpu, r.t_cpu_exp);
        if (spec_mode) {
            Spec sp(r, df, scfg);
            std::string gen_ids, finish, e2;
            int made = 0;
            r.t_cpu_exp = 0;
            const double t1 = now_s();
            sp.run(tok, gen, max_ctx, r.c.eos, -1, [&](int t) { gen_ids += std::to_string(t) + " "; return true; }, finish, e2, made);
            const double td = now_s() - t1;
            if (finish == "error") {
                fprintf(stderr, "spec: %s\n", e2.c_str());
                return 1;
            }
            printf("generated ids: %s\n", gen_ids.c_str());
            fprintf(stderr, "[decode] %d tokens in %.2f s = %.2f tok/s (spec dfly2, finish %s, cpu experts %.1f ms/token)\n", made, td, made / td, finish.c_str(), made ? r.t_cpu_exp / made * 1000 : 0.0);
            sp.report(stderr, td * 1000.0);
            std::ofstream f(out_dir + "/gen_ids.txt");
            f << gen_ids << "\n";
        } else if (gen > 0) {
            int next = 0;
            {
                const float* lg = &hl[(size_t)(hl.size() / V - 1) * V];
                next = (int)(std::max_element(lg, lg + V) - lg);
            }
            std::string gen_ids;
            const double t1 = now_s();
            std::vector<double> times;
            std::vector<float> margins;
            r.t_cpu_exp = 0;
            for (int s = 0; s < gen; ++s) {
                tok.push_back(next);
                gen_ids += std::to_string(next) + " ";
                if (next == r.c.eos) break;
                const double ts = now_s();
                const int pos = (int)tok.size() - 1;
                if (!r.forward_chunk(tok.data(), pos, 1, err)) {
                    fprintf(stderr, "forward: %s\n", err.c_str());
                    return 1;
                }
                r.head(1, 0);
                argmax_row(r.logits, V, r.d_argmax);
                int32_t nx = 0;
                d2h(&nx, r.d_argmax, 4);
                wait();
                if (std::getenv("HY4S_MARGINS")) {  // analysis: top1 - top2 logit gap of every generated token (near-tie flips between plain and speculative decode)
                    std::vector<float> row(V);
                    d2h(row.data(), r.logits, (size_t)V * 4);
                    wait();
                    float b1 = -3e38f, b2 = -3e38f;
                    for (int i = 0; i < V; ++i) {
                        if (row[i] > b1) { b2 = b1; b1 = row[i]; }
                        else if (row[i] > b2) b2 = row[i];
                    }
                    margins.push_back(b1 - b2);
                }
                next = nx;
                times.push_back(now_s() - ts);
            }
            const double td = now_s() - t1;
            printf("generated ids: %s\n", gen_ids.c_str());
            r.prof_report("decode");
            r.pacc_report("decode", (int)times.size());
            std::sort(times.begin(), times.end());
            fprintf(stderr, "[decode] %zu tokens in %.2f s = %.2f tok/s (median step %.1f ms, cpu experts %.1f ms/token)\n", times.size(), td, times.size() / td,
                    times.empty() ? 0.0 : times[times.size() / 2] * 1000, times.empty() ? 0.0 : r.t_cpu_exp / times.size() * 1000);
            std::ofstream f(out_dir + "/gen_ids.txt");
            f << gen_ids << "\n";
            if (!margins.empty()) {
                std::ofstream fm(out_dir + "/margins.txt");
                for (float m : margins) fm << m << "\n";
            }
        }
        {
            std::ofstream f(out_dir + "/top5.txt");
            f << top_out;
        }
        if (r.engram_ready) {
            auto s = r.eio.stats();
            fprintf(stderr, "[engram] lookups %llu seed %llu lru %llu page reads %llu (%.1f MB) io %.2f s\n", (unsigned long long)s.lookups, (unsigned long long)s.seed_hits,
                    (unsigned long long)s.lru_hits, (unsigned long long)s.page_reads, s.bytes_read / 1e6, s.io_seconds);
        }
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 4;
    }
    return 0;
}
