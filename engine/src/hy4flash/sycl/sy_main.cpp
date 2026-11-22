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
#define HY4_VERSION "0.1.66"
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

// `strata-hy4 --info [FILE.strata]`: one line of key=value facts, no GPU needed.
static int print_info(const std::string& file) {
    if (file.empty()) {
        printf("arch=hy4flash engine=hy4-serve-sycl-0 backend=sycl drafter=dfly2 spec_tree=1 spec_prob=1 lcp_resume=1 images=1 version=%s\n", HY4_VERSION);
        return 0;
    }
    std::string verr;
    hy4::VisionShardInfo vinfo;
    if (hy4::vision_probe(file, vinfo, verr)) {
        printf("arch=hy4flash_vision backend=sycl images=1 tensors=%zu\n", vinfo.T.size());
        return 0;
    }
    strata::artifact::StrataFileReader rd;
    std::string err;
    if (!rd.open(file, err)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    hy4::Manifest m;
    hy4::parse_manifest(rd.manifest_json(), m, false);
    printf("arch=hy4flash layers=%d hidden=%d vocab=%d experts=%d eos=%d drafter=%s images=%d backend=sycl\n", m.c.n_layers, m.c.hidden, m.c.vocab, m.c.n_experts, m.c.eos,
           manifest_has_drafter(m) ? "dfly2" : "none", 1);
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
    // Work around oneMKL + Level Zero V2 adapter segfault (exit 139) on Battlemage Xe2 (Arc Pro B70 bmg_g31):
    // default UR_LOADER_USE_LEVEL_ZERO_V2=0 when unset in user env BEFORE any SYCL device/queue init.
    if (!std::getenv("UR_LOADER_USE_LEVEL_ZERO_V2")) {
        hy4s_setenv("UR_LOADER_USE_LEVEL_ZERO_V2", "0", 0);
    }
    // Backend: Level Zero with SYCL command_graph replay (~2.15 us/kernel) beats OpenCL eager (~7.35 us/kernel) on both prefill & decode!
    // HY4S_BACKEND=opencl (or an explicit ONEAPI_DEVICE_SELECTOR) overrides.
    const bool user_sel = std::getenv("ONEAPI_DEVICE_SELECTOR") != nullptr;
    const char* be_env = std::getenv("HY4S_BACKEND");
    const bool want_ocl = (be_env && std::string(be_env) == "opencl");
    if (!user_sel) {
        hy4s_setenv("ONEAPI_DEVICE_SELECTOR", want_ocl ? "opencl:gpu" : "level_zero:gpu", 1);
    }
    std::string model, engram, tok_path, out_dir = ".", force_route, draft;
    std::string gpus_spec = std::getenv("HY4_GPUS") ? std::getenv("HY4_GPUS") : "";
    int gpu_arg = -1;
    SpecCfg scfg;
    int wincheck = 0, wincheck_skip = 0, detcheck = 0;
    bool detnotap = false;
    bool load_only = false, dbg = false, dump = false, all_logits = false, no_engram = false, no_index = false, prof = false, prof2 = false, warm = false, hi_act = false;
    bool serve = false, session_reuse = false, vision_requested = false;
    std::string vision_path;
    int64_t image_max_pixels = 1048576;
    double vram_experts_gb = -1.0, ram_experts_gb = -1.0;
    int kv_host_above = -1, gx_mb = 256, gx_min_t = 32, gen = 0, max_ctx = 0, chunk = 0, index_topk = 2048, threads = (int)std::max(2u, std::thread::hardware_concurrency() / 2), expert_act = 1;
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
        if (a == "--info") {
            const std::string f = (i + 1 < argc && argv[i + 1][0] != '-') ? std::string(argv[++i]) : std::string();
            return print_info(f);
        }
        if (a == "--export-tokenizer") {
            const std::string f = nxt();
            return export_tokenizer(f, nxt());
        }
        if (a == "--serve") continue;
        else if (a == "--native" || a == "--model") model = nxt();
        else if (a == "--max-context") max_ctx = atoi(nxt().c_str());
        else if (a == "--session-reuse") session_reuse = atoi(nxt().c_str()) != 0;
        else if (a == "--gpu") {
            gpu_arg = atoi(nxt().c_str());
        } else if (a == "--gpus" || a == "--devices") {
            gpus_spec = nxt();
        } else if (a == "--vision") {
            vision_path = nxt();
            vision_requested = true;
        } else if (a == "--vram-experts-gb") vram_experts_gb = atof(nxt().c_str());
        else if (a == "--ram-experts-gb") ram_experts_gb = atof(nxt().c_str());
        else if (a == "--engram-cache-mib") (void)nxt();
        else if (a == "--cpu-hi") hi_act = true;
        else if (a == "--kv") {
            const std::string kv = nxt();
            if (kv != "fp16") {
                printf("ERR --kv %s: only fp16\n", kv.c_str());
                fflush(stdout);
                return 2;
            }
        }
        else if (a == "--image-max-pixels") image_max_pixels = atoll(nxt().c_str());
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
        else if (a == "--wincheck") wincheck = atoi(nxt().c_str());
        else if (a == "--wincheck-skip") wincheck_skip = atoi(nxt().c_str());
        else if (a == "--detcheck") detcheck = atoi(nxt().c_str());
        else if (a == "--detnotap") detnotap = true;
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
        else if (a == "--spec-check") scfg.check = true;
        else if (a == "--spec-plain-only") scfg.plain_only = true;
        else if (a == "--spec-rv") scfg.rv = atoi(nxt().c_str());
        else if (a == "--spec-tree") scfg.tree = true;
        else if (a == "--no-spec-tree") scfg.tree = false;
        else if (a == "--spec-early-cut") scfg.early_cut = (float)atof(nxt().c_str());
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            if (serve) {
                printf("ERR unknown flag %s\n", a.c_str());
                fflush(stdout);
            }
            return 2;
        }
    }
    // Resolve single-GPU (--gpu N) or multi-GPU (--gpus 0,1 / --gpus 2 / --devices 0,1 / HY4_GPUS) BEFORE touching Q()/dev_caps().
    int primary_gpu = (gpu_arg >= 0) ? gpu_arg : 0;
    std::vector<int> peer_devs;
    if (!gpus_spec.empty()) {
        const auto avail_devs = enum_gpu_devices();
        parse_gpu_spec(gpus_spec, primary_gpu, (int)avail_devs.size(), primary_gpu, peer_devs);
        set_primary_gpu_index(primary_gpu);
    } else if (gpu_arg >= 0) {
        if (!user_sel) {
            const std::string sel = std::string(want_ocl ? "opencl:gpu:" : "level_zero:gpu:") + std::to_string(gpu_arg);
            hy4s_setenv("ONEAPI_DEVICE_SELECTOR", sel.c_str(), 1);
            set_primary_gpu_index(0);
        } else {
            set_primary_gpu_index(gpu_arg);
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
        so.chunk = chunk > 0 ? chunk : (dev_caps().global_mem >= (11ull << 30) ? 4096 : 2048);
        so.index_topk = index_topk;
        so.threads = threads;
        so.gx_mb = gx_mb;
        so.kv_host_above = kv_host_above >= 0 ? kv_host_above : (dev_caps().global_mem >= (24ull << 30) ? 131072 : 24000);
        so.vram_experts_gb = vram_experts_gb;
        so.ram_experts_gb = ram_experts_gb;
        so.hi_act = hi_act || (expert_act == 0);
        so.no_index = no_index;
        so.no_engram = no_engram;
        so.session_reuse = session_reuse;
        so.draft = draft.empty() ? "auto" : draft;
        so.vision_requested = vision_requested;
        so.vision = vision_path;
        so.image_max_pixels = image_max_pixels;
        so.primary_gpu = primary_gpu;
        so.peer_devs = peer_devs;
        so.spec = scfg;
        return hy4s_serve_main(so);
    }
    if (draft.empty()) draft = "off";
    if (model.empty()) {
        fprintf(stderr, "usage: strata-hy4-sycl --model SQ32.strata --engram Engram-Q8.strata --tokens ids.txt [--gen N] [--chunk N] [--ctx N] [--threads N] [--expert-act 0|1] [--warm] [--dump] [--all-logits] [--out dir]\n       strata-hy4-sycl --serve --native SQ.strata --engram ENG.strata [--max-context N] [--draft off|dfly2|auto]   (engine line protocol)\n       strata-hy4-sycl --info FILE.strata | --version | --export-tokenizer FILE.strata DIR\n");
        return 2;
    }
    try {
        const auto& caps = dev_caps();
        fprintf(stderr, "[gpu] %s, %.2f GiB (runtime: %s, sg8=%d sg16=%d xmx8=%d xmx16=%d graph=%d onemkl=%d)\n",
                caps.name.c_str(), caps.global_mem / 1073741824.0, caps.platform.c_str(),
                (int)caps.has_sg8, (int)caps.has_sg16, (int)caps.has_xmx_sg8, (int)caps.has_xmx_sg16, (int)caps.has_graph, (int)onemkl_enabled());
        SyModel g;
        std::string err;
        if (!g.load(model, err)) {
            fprintf(stderr, "load: %s\n", err.c_str());
            return 1;
        }
        std::vector<int32_t> tok;
        if (!load_only) {
            std::ifstream f(tok_path);
            long v;
            while (f >> v) tok.push_back((int32_t)v);
            if (tok.empty()) {
                fprintf(stderr, "no tokens\n");
                return 2;
            }
        }
        const int T0 = (int)tok.size();
        if (max_ctx <= 0) max_ctx = std::max(T0 + gen + 16, 131072);
        const int Tc = chunk > 0 ? chunk : (caps.global_mem >= (11ull << 30) ? 4096 : 2048);
        const bool spec_mode = draft == "dfly2" && gen > 0;
        hy4::CpuExperts ucx;
        SyVramExpertCache vcache;
        SyPeerExperts peers;
        Runner r(g, ucx, max_ctx, Tc, std::max(8, all_logits ? std::min(Tc, std::max(T0, 1)) : 8));
        r.dump = dump;
        r.dbg_layers = dbg;
        r.force_route_dir = force_route;
        r.dump_dir = out_dir;
        r.index_topk = index_topk;
        r.index_on = no_index ? 0 : 1;
        r.prof = prof;
        r.prof2 = prof2;
        r.expert_act = expert_act;
        r.kv_host_above = kv_host_above >= 0 ? kv_host_above : (caps.global_mem >= (24ull << 30) ? 131072 : 24000);
        if (!r.init(no_engram ? std::string() : engram, err)) {
            fprintf(stderr, "init: %s\n", err.c_str());
            return 1;
        }
        // Size VRAM expert cache + unified CPU expert pool BEFORE DFly2 (reserving 16 MiB for DFly2 in BOTH plain and DFly2)
        // so plain and DFly2 ALWAYS receive the exact same VRAM slot count -> 100% bitwise plain == DFly2 even with adaptive=1!
        if (scfg.check || wincheck > 0 || detcheck > 0) hy4s_setenv("HY4S_VRAM_STATIC", "1", 1);
        const bool want_gx = (gx_mb > 0 && (load_only || T0 >= gx_min_t));
        const size_t gx_reserve = (want_gx ? ((size_t)gx_mb * 2 / 3 + 220) * (1ull << 20) : 0) + (16ull << 20);
        const size_t host_oh = r.kv_host_bytes + g.host_bytes_total + ((size_t)std::max(gx_mb, 0) << 20) + (640ull << 20);
        if (!setup_experts_unified(model, threads, vram_experts_gb, ram_experts_gb, gx_reserve, host_oh, hi_act || (expert_act == 0), vcache, ucx, err, peer_devs, &peers)) {
            fprintf(stderr, "experts: %s\n", err.c_str());
            return 1;
        }
        r.vcache = vcache.active() ? &vcache : nullptr;
        r.peers = peers.active() ? &peers : nullptr;
        DFly2 df(r);
        if (draft == "dfly2") {
            if (!df.init(err)) {
                fprintf(stderr, "dfly2: %s\n", err.c_str());
                return 1;
            }
            fprintf(stderr, "[dfly2] drafter ready (K=%d%s, rv=%d), device allocated %.2f GiB\n", scfg.K, scfg.adaptive ? " adaptive" : " fixed", scfg.rv, dev_bytes_allocated() / 1073741824.0);
        }
        (void)warm;
        if (load_only) return 0;
        // Start Chunk 0 Engram NVMe fetch upfront so it overlaps with GpuExperts setup AND Layer 0!
        tok.reserve(max_ctx + 8);
        r.prefetch_engram(tok.data(), 0, std::min(r.Tc, T0));
        GpuExperts gxe;
        if (gx_mb > 0 && T0 >= gx_min_t) {
            std::string gerr;
            try {
                if (gxe.init_unified(model, &ucx, r.vcache, (size_t)gx_mb << 20, Tc, gerr, threads)) {
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
        std::string top_out;
        const double io0 = io_read_gb();
        const double t0 = now_s();
        int done = 0;
        const int V = r.c.vocab;
        std::vector<float> hl;
        while (done < T0) {
            const int T = std::min(r.Tc, T0 - done);
            // Prefetch NEXT chunk BEFORE running forward_chunk so NVMe Engram reads overlap 100% with the 42-layer GPU forward!
            if (done + T < T0) r.prefetch_engram(tok.data(), done + T, std::min(r.Tc, T0 - (done + T)));
            if (!r.forward_chunk(tok.data(), done, T, err)) {
                fprintf(stderr, "forward: %s\n", err.c_str());
                return 1;
            }
            done += T;
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
        const double io1 = io_read_gb();
        const auto ucs_pf = ucx.stats();
        fprintf(stderr, "[io] prefill: SSD read %.2f GB in %.2f s = %.3f GB/s (cold %.2f GB, gx vram_experts %llu)\n",
                io1 - io0, t_pf, (io1 - io0) / std::max(t_pf, 1e-9), ucs_pf.cold_bytes / 1e9,
                (unsigned long long)(r.gx ? r.gx->stats().vram_experts : 0));
        if (spec_mode) {
            spec_set_batch_invariant();
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
            fprintf(stderr, "[decode] %d tokens in %.2f s = %.2f tok/s (spec dfly2, finish %s, cpu experts %.1f ms/token, graph replays %llu captures %llu)\n",
                    made, td, made / td, finish.c_str(), made ? r.t_cpu_exp / made * 1000 : 0.0,
                    (unsigned long long)r.graph_replays, (unsigned long long)r.graph_captures);
            if (vcache.active()) {
                const uint64_t ht = vcache.hits(), ms = vcache.misses();
                const uint64_t uq = vcache.unique_expert_launches(), tp_l = vcache.total_pair_launches();
                fprintf(stderr, "[vram-cache] hits %llu / %llu (%.1f%%), promotions %llu, dedup %llu/%llu (%.2fx)\n",
                        (unsigned long long)ht, (unsigned long long)(ht + ms),
                        (ht + ms) ? 100.0 * (double)ht / (double)(ht + ms) : 0.0, (unsigned long long)vcache.promotions(),
                        (unsigned long long)uq, (unsigned long long)tp_l, uq ? (double)tp_l / (double)uq : 1.0);
            }
            fprintf(stderr, "[io] decode: SSD read %.2f GB in %.2f s = %.3f GB/s (cpu pool %.1f GB/s)\n",
                    io_read_gb() - io1, td, (io_read_gb() - io1) / std::max(td, 1e-9), ucx.stats().gbps());
            sp.report(stderr, td * 1000.0);
            std::ofstream f(out_dir + "/gen_ids.txt");
            f << gen_ids << "\n";
        } else if (detcheck > 0) {
            // is a 1-token step repeatable? Greedy-decode wincheck_skip tokens, then run the SAME step (same token, same position, same KV prefix) detcheck times, tapping every layer, and compare the repeats bitwise.
            int next = 0;
            {
                const float* lg = &hl[(size_t)(hl.size() / V - 1) * V];
                next = (int)(std::max_element(lg, lg + V) - lg);
            }
            for (int s = 0; s < wincheck_skip; ++s) {
                tok.push_back(next);
                const int pos = (int)tok.size() - 1;
                if (!r.forward_chunk(tok.data(), pos, 1, err)) { fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
                r.head(1, 0);
                argmax_row(r.logits, V, r.d_argmax);
                int32_t nx = 0;
                d2h(&nx, r.d_argmax, 4);
                wait();
                next = nx;
            }
            tok.push_back(next);
            const int pos = (int)tok.size() - 1;
            std::vector<std::vector<std::pair<std::string, std::vector<float>>>> rep(detcheck);
            std::vector<std::vector<float>> rlog(detcheck);
            for (int k = 0; k < detcheck; ++k) {
                r.tap = detnotap ? nullptr : &rep[k];
                if (!r.forward_chunk(tok.data(), pos, 1, err)) { fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
                r.tap = nullptr;
                r.head(1, 0);
                rlog[k].resize(V);
                d2h(rlog[k].data(), r.logits, (size_t)V * 4);
                wait();
            }
            printf("detcheck pos=%d repeats=%d\n", pos, detcheck);
            int bad = 0;
            for (int k = 1; k < detcheck; ++k) {
                std::string first;
                size_t nbad = 0;
                for (size_t t = 0; t < rep[0].size() && t < rep[k].size(); ++t) {
                    size_t nb = 0;
                    const auto& a0 = rep[0][t].second;
                    const auto& a1 = rep[k][t].second;
                    for (size_t e = 0; e < a0.size() && e < a1.size(); ++e)
                        if (std::memcmp(&a0[e], &a1[e], 4) != 0) nb++;
                    if (nb) { nbad++; if (first.empty()) first = rep[0][t].first + " (" + std::to_string(nb) + " of " + std::to_string(a0.size()) + ")"; }
                }
                size_t lb = 0;
                for (int e = 0; e < V; ++e) if (std::memcmp(&rlog[0][e], &rlog[k][e], 4) != 0) lb++;
                if (!detnotap || lb) printf("repeat %d vs 0: differing taps %zu of %zu, first: %s, logits differing %zu\n", k, nbad, rep[0].size(), first.empty() ? "none" : first.c_str(), lb);
                if (nbad || lb) bad++;
            }
            printf("DETCHECK %s\n", bad ? "NONDETERMINISTIC" : "REPEATABLE");
        } else if (wincheck > 0) {
            // does a window of M rows give bit-identical rows to M single-token steps? (the verify of speculative decoding must). Greedy-decode wincheck_skip tokens, then
            // run M single steps (tapping every layer), then the same tokens as ONE window, and compare the tapped rows bitwise.
            const int M = std::min(wincheck, 8);
            if (std::getenv("HY4S_BATCH_INV") == nullptr || std::getenv("HY4S_BATCH_INV")[0] != '0') spec_set_batch_invariant();
            int next = 0;
            {
                const float* lg = &hl[(size_t)(hl.size() / V - 1) * V];
                next = (int)(std::max_element(lg, lg + V) - lg);
            }
            for (int s = 0; s < wincheck_skip; ++s) {
                tok.push_back(next);
                const int pos = (int)tok.size() - 1;
                if (!r.forward_chunk(tok.data(), pos, 1, err)) { fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
                r.head(1, 0);
                argmax_row(r.logits, V, r.d_argmax);
                int32_t nx = 0;
                d2h(&nx, r.d_argmax, 4);
                wait();
                next = nx;
            }
            const int p0 = (int)tok.size();
            std::vector<std::vector<std::pair<std::string, std::vector<float>>>> singles(M);
            std::vector<std::vector<float>> slog(M);
            for (int i = 0; i < M; ++i) {
                tok.push_back(next);
                r.tap = &singles[i];
                if (!r.forward_chunk(tok.data(), p0 + i, 1, err)) { fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
                r.tap = nullptr;
                r.head(1, 0);
                slog[i].resize(V);
                d2h(slog[i].data(), r.logits, (size_t)V * 4);
                wait();
                next = (int)(std::max_element(slog[i].begin(), slog[i].end()) - slog[i].begin());
            }
            std::vector<std::pair<std::string, std::vector<float>>> win;
            r.tap = &win;
            if (!r.forward_chunk(tok.data(), p0, M, err)) { fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
            r.tap = nullptr;
            r.head(M, 0);
            std::vector<float> wlog((size_t)M * V);
            d2h(wlog.data(), r.logits, wlog.size() * 4);
            wait();
            printf("wincheck p0=%d M=%d batch_invariant=%d\n", p0, M, g_mla_batch_invariant);
            int bad_rows = 0;
            for (int i = 0; i < M; ++i) {
                std::string first;
                size_t nbad = 0, ntap = 0;
                double maxd = 0;
                for (size_t k = 0; k < singles[i].size() && k < win.size(); ++k) {
                    const auto& sv = singles[i][k].second;
                    const auto& wv = win[k].second;
                    const size_t per = sv.size();
                    if (singles[i][k].first != win[k].first || wv.size() < (size_t)(i + 1) * per) { if (first.empty()) first = singles[i][k].first + " (shape)"; continue; }
                    ntap++;
                    size_t nb = 0;
                    for (size_t e = 0; e < per; ++e) {
                        const float a = sv[e], b = wv[(size_t)i * per + e];
                        if (std::memcmp(&a, &b, 4) != 0) { nb++; maxd = std::max(maxd, (double)std::fabs(a - b)); }
                    }
                    if (nb) { nbad++; if (first.empty()) first = singles[i][k].first + " (" + std::to_string(nb) + " of " + std::to_string(per) + " differ)"; }
                }
                size_t lb = 0;
                for (int e = 0; e < V; ++e) if (std::memcmp(&slog[i][e], &wlog[(size_t)i * V + e], 4) != 0) lb++;
                printf("row %d: taps compared %zu, differing taps %zu, first: %s, max abs diff %.3g, logits differing %zu of %d\n", i, ntap, nbad, first.empty() ? "none" : first.c_str(), maxd, lb, V);
                if (nbad || lb) bad_rows++;
            }
            printf("WINCHECK %s\n", bad_rows ? "DIFFERS" : "IDENTICAL");
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
                if (std::getenv("HY4S_LOGSUM")) {  // analysis: FNV-1a hash of the exact bits of every decode step's logits (run-to-run determinism: compare two runs step by step)
                    std::vector<float> lrow(V);
                    d2h(lrow.data(), r.logits, (size_t)V * 4);
                    wait();
                    uint64_t hsh = 1469598103934665603ull;
                    const uint8_t* bp = (const uint8_t*)lrow.data();
                    for (size_t i = 0; i < (size_t)V * 4; ++i) hsh = (hsh ^ bp[i]) * 1099511628211ull;
                    printf("H %d %016llx\n", s, (unsigned long long)hsh);
                }
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
                r.commit_vram_adapt(1, pos + 1);
                times.push_back(now_s() - ts);
            }
            const double td = now_s() - t1;
            printf("generated ids: %s\n", gen_ids.c_str());
            r.prof_report("decode");
            r.pacc_report("decode", (int)times.size());
            std::sort(times.begin(), times.end());
            fprintf(stderr, "[decode] %zu tokens in %.2f s = %.2f tok/s (median step %.1f ms, cpu experts %.1f ms/token, graph replays %llu captures %llu)\n",
                    times.size(), td, times.size() / td,
                    times.empty() ? 0.0 : times[times.size() / 2] * 1000, times.empty() ? 0.0 : r.t_cpu_exp / times.size() * 1000,
                    (unsigned long long)r.graph_replays, (unsigned long long)r.graph_captures);
            if (vcache.active()) {
                const uint64_t ht = vcache.hits(), ms = vcache.misses();
                const uint64_t uq = vcache.unique_expert_launches(), tp_l = vcache.total_pair_launches();
                fprintf(stderr, "[vram-cache] hits %llu / %llu (%.1f%%), promotions %llu, dedup %llu/%llu (%.2fx)\n",
                        (unsigned long long)ht, (unsigned long long)(ht + ms),
                        (ht + ms) ? 100.0 * (double)ht / (double)(ht + ms) : 0.0, (unsigned long long)vcache.promotions(),
                        (unsigned long long)uq, (unsigned long long)tp_l, uq ? (double)tp_l / (double)uq : 1.0);
            }
            fprintf(stderr, "[io] decode: SSD read %.2f GB in %.2f s = %.3f GB/s (cpu pool %.1f GB/s)\n",
                    io_read_gb() - io1, td, (io_read_gb() - io1) / std::max(td, 1e-9), ucx.stats().gbps());
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
