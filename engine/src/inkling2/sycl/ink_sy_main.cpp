// ink_sy_main.cpp - Standalone Intel Arc SYCL SG16 CLI for Inkling-2-Mini (.strata v1.1)
// Supports Alchemist (A750/A770 dg2) & Battlemage (B580 bmg_g21, Arc Pro B70 bmg_g31) on Linux & Windows!
// CRITICAL BATTLEMAGE B70 RULE: Zero reqd_sub_group_size(8); strictly SG16!
// 100% MIT clean-room implementation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define INK_DEFINE_STBI 1
#include "ink_sy_runner.h"

using namespace ink_sycl;

static std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return out;
}

static bool run_greedy_decode(InkSyRunner& R, const std::vector<int>& prompt, int n_gen, bool use_spec,
                              std::vector<int>& out_tokens, double& prefill_s, double& decode_s,
                              int& total_drafts, int& total_accepted) {
    R.reset_slot(0);
    const int P = (int)prompt.size();
    std::vector<int> sids(P, 0), pos(P);
    for (int i = 0; i < P; ++i) pos[i] = i;

    const double t0 = now_s();
    int next_tok = 0;
    if (!R.forward_batch(prompt.data(), sids.data(), pos.data(), P, nullptr, nullptr, 1, nullptr, &next_tok, use_spec)) {
        return false;
    }
    prefill_s = now_s() - t0;

    out_tokens.clear();
    out_tokens.push_back(next_tok);
    int cur_pos = P;
    total_drafts = 0;
    total_accepted = 0;

    const double t1 = now_s();
    while ((int)out_tokens.size() < n_gen) {
        if (!use_spec || !R.use_dspark || cur_pos <= 0 || cur_pos + 9 >= R.max_ctx) {
            int sid = 0, p = cur_pos, nt = 0;
            if (!R.forward_batch(&next_tok, &sid, &p, 1, nullptr, nullptr, 1, nullptr, &nt, use_spec)) return false;
            out_tokens.push_back(nt);
            next_tok = nt;
            cur_pos++;
            continue;
        }

        int drafts[7]{};
        const int nd = R.dspark_draft_block8(0, next_tok, cur_pos, drafts);
        if (nd <= 0) {
            int sid = 0, p = cur_pos, nt = 0;
            if (!R.forward_batch(&next_tok, &sid, &p, 1, nullptr, nullptr, 1, nullptr, &nt, true)) return false;
            out_tokens.push_back(nt);
            next_tok = nt;
            cur_pos++;
            continue;
        }

        R.snapshot_slot(0);
        int vtoks[8], vsids[8], vpos[8], varg[8]{};
        vtoks[0] = next_tok; vsids[0] = 0; vpos[0] = cur_pos;
        for (int i = 0; i < nd; ++i) { vtoks[1 + i] = drafts[i]; vsids[1 + i] = 0; vpos[1 + i] = cur_pos + 1 + i; }
        const int M = 1 + nd;
        if (!R.forward_batch(vtoks, vsids, vpos, M, nullptr, nullptr, 2, nullptr, varg, true)) return false;

        int acc = 0;
        while (acc < nd && varg[acc] == drafts[acc]) acc++;
        total_drafts += nd;
        total_accepted += acc;

        if (acc < nd) {
            // O(1) ZERO-REPLAY rollback via per-step ShortConv verify ring!
            R.rollback_slot(0, cur_pos + 1 + acc);
        }

        for (int i = 0; i <= acc && (int)out_tokens.size() < n_gen; ++i) {
            out_tokens.push_back(varg[i]);
        }
        next_tok = varg[acc];
        cur_pos += (acc + 1);
    }
    decode_s = now_s() - t1;
    return true;
}

int main(int argc, char** argv) {
    ensure_l0_v1_default();
    std::string model_path, vision_path, image_path, prompt_csv = "1,2,3,4,5,6,7,8";
    int n_gen = 32, max_ctx = 4096, parallel = 1;
    int gpu_index = 0, req_gpus = 1;
    std::vector<int> req_gpu_ids;
    double ep_split = 0.5;
    double vram_gb = -1.0;
    bool show_info = false, do_selfcheck = false, do_spec_check = false, do_bench = false, do_par_sweep = false;
    bool no_dspark = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--version") { printf("strata-inkling 0.1.66\n"); return 0; }
        else if (a == "--model" || a == "-m" || a == "--native") model_path = need();
        else if (a == "--vision") vision_path = need();
        else if (a == "--image") image_path = need();
        else if (a == "--info") { show_info = true; if (i + 1 < argc && argv[i + 1][0] != '-') model_path = argv[++i]; }
        else if (a == "--selfcheck") do_selfcheck = true;
        else if (a == "--spec-check") do_spec_check = true;
        else if (a == "--bench") do_bench = true;
        else if (a == "--par-sweep") do_par_sweep = true;
        else if (a == "--no-dspark") no_dspark = true;
        else if (a == "--gen" || a == "-n") n_gen = std::atoi(need().c_str());
        else if (a == "--ctx" || a == "--max-context") max_ctx = std::atoi(need().c_str());
        else if (a == "--parallel" || a == "-p") parallel = std::atoi(need().c_str());
        else if (a == "--gpu") gpu_index = std::atoi(need().c_str());
        else if (a == "--gpus" || a == "--devices") {
            std::string spec = need();
            if (spec.find(',') != std::string::npos) {
                req_gpu_ids = parse_csv_ints(spec);
                if (!req_gpu_ids.empty()) {
                    gpu_index = req_gpu_ids[0];
                    req_gpus = (int)req_gpu_ids.size();
                }
            } else {
                req_gpu_ids.clear();
                req_gpus = std::max(1, std::atoi(spec.c_str()));
            }
        }
        else if (a == "--ep-split") ep_split = std::atof(need().c_str());
        else if (a == "--vram-experts-gb") vram_gb = std::atof(need().c_str());
        else if (a == "--prompt-tokens" || a == "--prompt-ids") prompt_csv = need();
    }
    if (const char* ep_env = std::getenv("IK2_EP_SPLIT"); ep_env && *ep_env) {
        double v = std::atof(ep_env);
        if (v > 0.0 && v < 1.0) ep_split = v;
    }
    if (req_gpus == 1 && req_gpu_ids.empty()) {
        if (const char* g_env = std::getenv("IK2_GPUS"); g_env && *g_env) {
            std::string spec(g_env);
            if (spec.find(',') != std::string::npos) {
                req_gpu_ids = parse_csv_ints(spec);
                if (!req_gpu_ids.empty()) {
                    gpu_index = req_gpu_ids[0];
                    req_gpus = (int)req_gpu_ids.size();
                }
            } else {
                req_gpus = std::max(1, std::atoi(g_env));
            }
        }
    }
    set_primary_gpu_index(gpu_index);

    if (model_path.empty()) {
        fprintf(stderr, "Usage: strata-inkling-sycl --model <path.strata> [--info] [--selfcheck] [--spec-check] [--bench] [--par-sweep] [--parallel N] [--vram-experts-gb G] [--vision <vis.strata> --image <img.png>]\n");
        return 1;
    }

    InkSyModel mod;
    std::string err;
    if (show_info) {
        if (!mod.probe_only(model_path, err)) { fprintf(stderr, "[ERROR] %s\n", err.c_str()); return 1; }
        printf("strata-inkling 0.1.66\n");
        printf("arch=inkling2_loop layers=%d visits=%d hidden=%d vocab=%d experts=%d eos=%d drafter=%s images=%d image_pad=%d parallel=64 backend=sycl build=%s\n",
               mod.man.c.n_phys, mod.man.c.n_visits, mod.man.c.hidden, mod.man.c.vocab_unpadded,
               mod.man.c.n_experts, mod.man.c.eos_id, mod.man.c.has_dspark ? "dspark" : "none",
               vision_path.empty() ? 0 : 1, mod.man.c.image_pad_id, mod.man.build.c_str());
        return 0;
    }

    if (!mod.load(model_path, vram_gb, err, 0, req_gpus, req_gpu_ids, ep_split)) { fprintf(stderr, "[ERROR] %s\n", err.c_str()); return 1; }

    // Optional Vision test
    if (!vision_path.empty() && !image_path.empty()) {
        InkSyVision vis;
        if (!vis.load(vision_path, err)) { fprintf(stderr, "[ERROR] %s\n", err.c_str()); return 1; }
        std::vector<float> vemb; int n_vis_tok = 0;
        if (!vis.encode_image_file(image_path, vemb, n_vis_tok, err)) { fprintf(stderr, "[ERROR] %s\n", err.c_str()); return 1; }
        double sum = 0.0, l2 = 0.0;
        for (float v : vemb) { sum += v; l2 += (double)v * v; }
        printf("[VISION] image=%s tokens=%d mean=%.6f l2=%.4f first4=[%.4f, %.4f, %.4f, %.4f]\n",
               image_path.c_str(), n_vis_tok, sum / vemb.size(), std::sqrt(l2),
               vemb[0], vemb[1], vemb[2], vemb[3]);
    }

    const int init_slots = do_par_sweep ? 64 : std::max(1, parallel);
    const int init_batch = std::max(64, init_slots);
    InkSyRunner runner;
    if (!runner.init(mod, init_slots, max_ctx, init_batch, !no_dspark, err)) {
        fprintf(stderr, "[ERROR] runner init: %s\n", err.c_str());
        return 1;
    }

    std::vector<int> prompt = parse_csv_ints(prompt_csv);
    if (prompt.empty()) prompt = {1, 2, 3, 4, 5, 6, 7, 8};

    if (do_selfcheck) {
        runner.reset_slot(0);
        const int P = (int)prompt.size();
        std::vector<int> sids(P, 0), pos(P);
        for (int i = 0; i < P; ++i) pos[i] = i;
        std::vector<float> logits(mod.man.c.vocab_unpadded);
        int argm = 0;
        const double t0 = now_s();
        runner.forward_batch(prompt.data(), sids.data(), pos.data(), P, nullptr, nullptr, 1, logits.data(), &argm, false);
        const double dt = now_s() - t0;
        int nans = 0;
        std::vector<std::pair<float, int>> top;
        top.reserve(logits.size());
        for (int i = 0; i < (int)logits.size(); ++i) {
            if (!std::isfinite(logits[i])) nans++;
            top.push_back({logits[i], i});
        }
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        printf("[SELFCHECK] dev=\"%s\" build=%s nans=%d argmax=%d dt=%.1fms top5=[%d:%.4f, %d:%.4f, %d:%.4f, %d:%.4f, %d:%.4f]\n",
               dev_caps().name.c_str(), mod.man.build.c_str(), nans, argm, dt * 1e3,
               top[0].second, top[0].first, top[1].second, top[1].first,
               top[2].second, top[2].first, top[3].second, top[3].first,
               top[4].second, top[4].first);
        if (nans > 0) return 2;
    }

    if (do_spec_check) {
        std::vector<int> t_plain, t_spec;
        double pf1 = 0, dc1 = 0, pf2 = 0, dc2 = 0;
        int d1 = 0, a1 = 0, d2 = 0, a2 = 0;
        run_greedy_decode(runner, prompt, n_gen, false, t_plain, pf1, dc1, d1, a1);
        run_greedy_decode(runner, prompt, n_gen, true, t_spec, pf2, dc2, d2, a2);
        int matches = 0;
        for (int i = 0; i < n_gen; ++i) if (t_plain[i] == t_spec[i]) matches++;
        printf("[SPEC-CHECK] parity=%d/%d (%s) | plain=%.2f tok/s | dspark=%.2f tok/s (acc=%d/%d = %.1f%%, mean_step=%.2f)\n",
               matches, n_gen, (matches == n_gen ? "BITWISE-EXACT" : "MISMATCH"),
               (n_gen - 1) / std::max(dc1, 1e-6),
               (n_gen - 1) / std::max(dc2, 1e-6),
               a2, d2, d2 > 0 ? (100.0 * a2 / d2) : 0.0,
               d2 > 0 ? (1.0 + 7.0 * a2 / d2) : 1.0);
        if (matches != n_gen) {
            printf("  plain:"); for (int x : t_plain) printf(" %d", x); printf("\n");
            printf("  spec :"); for (int x : t_spec) printf(" %d", x); printf("\n");
            return 3;
        }
    }

    if (do_bench) {
        std::vector<int> toks;
        double pf = 0, dc = 0; int td = 0, ta = 0;
        run_greedy_decode(runner, prompt, n_gen, !no_dspark, toks, pf, dc, td, ta);
        printf("[BENCH] build=%s mode=%s prompt=%zu ppts=%.1f decode_toks=%d decode_tps=%.2f acc=%.1f%% vram_exp=%d/3072\n",
               mod.man.build.c_str(), no_dspark ? "plain" : "dspark", prompt.size(),
               prompt.size() / std::max(pf, 1e-6), n_gen - 1, (n_gen - 1) / std::max(dc, 1e-6),
               td > 0 ? (100.0 * ta / td) : 0.0, mod.n_vram_experts);
    }

    if (do_par_sweep) {
        const int cs[] = {1, 4, 8, 32, 64};
        const int steps = 16;
        for (int c : cs) {
            for (int s = 0; s < c; ++s) runner.reset_slot(s);
            std::vector<int> toks(c), sids(c), pos(c, 0), argm(c, 0);
            for (int s = 0; s < c; ++s) { toks[s] = prompt[s % prompt.size()]; sids[s] = s; }
            // Warmup step 0
            runner.forward_batch(toks.data(), sids.data(), pos.data(), c, nullptr, nullptr, 2, nullptr, argm.data(), false);
            const double t0 = now_s();
            for (int st = 1; st <= steps; ++st) {
                for (int s = 0; s < c; ++s) { toks[s] = argm[s]; pos[s] = st; }
                runner.forward_batch(toks.data(), sids.data(), pos.data(), c, nullptr, nullptr, 2, nullptr, argm.data(), false);
            }
            const double dt = now_s() - t0;
            const double agg = (double)(c * steps) / std::max(dt, 1e-6);
            printf("[PAR-SWEEP] c=%2d steps=%d dt=%.3fs agg_tps=%.2f per_req_tps=%.2f\n",
                   c, steps, dt, agg, agg / c);
        }
    }

    return 0;
}
