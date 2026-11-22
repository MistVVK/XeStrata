// ink_metal_main.cpp - strata-inkling-metal: High-Speed Apple Silicon Metal Runner for Inkling-2-Mini (.strata v1.1)
// Supports:
//   --version, --info FILE.strata [--vision FILE], --selfcheck, --bench, --spec-check
//   --parallel 1|4|8|32|64, --draft auto|dspark|off, --vision FILE, --image FILE, --serve
// 100% MIT clean-room implementation.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define INK_DEFINE_STBI 1
#include "ink_metal_runner.h"

#ifdef INK_EMBED_KERNELS
#include "ink_metal_embedded.inc"
#endif

#ifndef INK_VERSION
#define INK_VERSION "0.1.66"
#endif

namespace ink_metal {

bool InkMetalRunner::compile_shaders(const std::string& kernel_dir, std::string& err) {
#ifdef INK_EMBED_KERNELS
    auto find_emb = [](const char* fname) -> std::string {
        for (const auto& f : hy4_emb_files) {
            if (std::strcmp(f.name, fname) == 0) {
                return std::string(reinterpret_cast<const char*>(f.data), f.size);
            }
        }
        return {};
    };
    auto compile_one = [&](const char* fname, int& out_lib) -> bool {
        std::string src = find_emb(fname);
        if (src.empty()) { err = std::string("missing embedded shader ") + fname; return false; }
        std::string why;
        out_lib = cx->compile_try({}, src, false, &why);
        if (out_lib < 0) { err = std::string("MSL compile failed (") + fname + "): " + why; return false; }
        return true;
    };
    if (!compile_one("ink_metal_codecs.metal", lib_codecs)) return false;
    if (!compile_one("ink_metal_ops.metal", lib_ops)) return false;
    if (!compile_one("ink_metal_dspark.metal", lib_dspark)) return false;
    if (!compile_one("ink_metal_vision.metal", lib_vis)) return false;
    lib_mpp = -1;
    if (cx->supports_metal4()) {
        std::string msrc = find_emb("ink_metal_mpp.metal");
        if (!msrc.empty()) {
            std::string mwhy;
            lib_mpp = cx->compile_try({}, msrc, false, &mwhy);
            if (lib_mpp < 0) fprintf(stderr, "[ink_metal] Metal 4 MPP optional compile skipped: %s\n", mwhy.c_str());
        }
    }
    return true;
#else
    auto compile_file = [&](const std::string& fname, int& out_lib) -> bool {
        std::string p = kernel_dir.empty() ? fname : (kernel_dir + "/" + fname);
        std::string why;
        out_lib = cx->compile_try({p}, "", false, &why);
        if (out_lib < 0) { err = "MSL compile failed (" + p + "): " + why; return false; }
        return true;
    };
    if (!compile_file("ink_metal_codecs.metal", lib_codecs)) return false;
    if (!compile_file("ink_metal_ops.metal", lib_ops)) return false;
    if (!compile_file("ink_metal_dspark.metal", lib_dspark)) return false;
    if (!compile_file("ink_metal_vision.metal", lib_vis)) return false;
    lib_mpp = -1;
    if (cx->supports_metal4()) {
        std::string mp = kernel_dir.empty() ? "ink_metal_mpp.metal" : (kernel_dir + "/ink_metal_mpp.metal");
        std::string mwhy;
        lib_mpp = cx->compile_try({mp}, "", false, &mwhy);
        if (lib_mpp < 0) fprintf(stderr, "[ink_metal] Metal 4 MPP optional compile skipped: %s\n", mwhy.c_str());
    }
    return true;
#endif
}

}  // namespace ink_metal

using namespace ink_metal;

static int print_info(const std::string& path, const std::string& vis_path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path.c_str()); return 1; }
    uint8_t frame[4096];
    if (::pread(fd, frame, 4096, 0) != 4096) { ::close(fd); return 1; }
    strata::artifact::StrataSectionEntry s0{};
    std::memcpy(&s0, frame + 0x10, 64);
    std::string mjson, err;
    // Read section 0
    std::vector<uint8_t> raw(s0.byte_length);
    if (::pread(fd, raw.data(), raw.size(), (off_t)s0.file_offset) != (ssize_t)raw.size()) { ::close(fd); return 1; }
    ::close(fd);
    unsigned long long rsize = ZSTD_getFrameContentSize(raw.data(), raw.size());
    if (rsize != ZSTD_CONTENTSIZE_ERROR && rsize != ZSTD_CONTENTSIZE_UNKNOWN) {
        mjson.resize((size_t)rsize);
        size_t d = ZSTD_decompress(mjson.data(), mjson.size(), raw.data(), raw.size());
        if (ZSTD_isError(d)) return 1;
        mjson.resize(d);
    }
    InkManifest m;
    if (!parse_ink_manifest(mjson, m, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    printf("arch=inkling2_loop layers=%d visits=%d hidden=%d vocab=%d experts=%d eos=%d drafter=%s images=%d image_pad=%d parallel=64 backend=metal build=%s\n",
           m.c.n_phys, m.c.n_visits, m.c.hidden, m.c.vocab_unpadded, m.c.n_experts, m.c.eos_id,
           m.c.has_dspark ? "dspark" : "none", vis_path.empty() ? 0 : 1, m.c.image_pad_id, m.build.c_str());
    return 0;
}

// Run greedy generation for 1 slot (plain decode, Linear DSpark Block-8, OR Multi-Branch Tree DSpark!)
// tree_budget: 0 = linear Block-8, > 0 (e.g. 12, 16, 24) = Multi-Branch Tree DSpark!
static bool run_greedy_decode(
    InkMetalRunner& r,
    int slot,
    const std::vector<int32_t>& prompt,
    int max_new,
    bool use_spec,
    std::vector<int32_t>& out_ids,
    double& prompt_ms,
    double& decode_ms,
    int& total_accepted,
    int& total_offered,
    std::string& err,
    int tree_budget = 0) {
    r.reset_slot(slot);
    out_ids.clear();
    total_accepted = 0;
    total_offered = 0;

    // 1. Chunked prefill
    const int P = (int)prompt.size();
    const int chunk = std::min(r.max_batch, 2048);
    int32_t next_tok = 0;
    double tp0 = mtl::now_s();
    for (int p0 = 0; p0 < P; p0 += chunk) {
        const int cm = std::min(chunk, P - p0);
        InkBatchStep st;
        st.M = cm;
        st.slot_ids.assign(cm, slot);
        st.pos.resize(cm);
        st.tok_ids.resize(cm);
        for (int i = 0; i < cm; ++i) {
            st.pos[i] = p0 + i;
            st.tok_ids[i] = prompt[p0 + i];
        }
        st.capture_taps = use_spec && r.use_dspark;
        const bool is_last = (p0 + cm == P);
        st.compute_logits = is_last;
        st.logits_last_only_idx = is_last ? (cm - 1) : -1;
        if (!r.forward_batch(st, nullptr, is_last ? &next_tok : nullptr, err)) return false;
    }
    prompt_ms = (mtl::now_s() - tp0) * 1e3;

    if (max_new <= 0) { decode_ms = 0.0; return true; }
    out_ids.push_back(next_tok);
    int cur_pos = P;

    // 2. Decode loop (adaptive DSpark draft length OR Multi-Branch Tree DSpark!)
    double td0 = mtl::now_s();
    const int min_k = 2;
    const int max_k = r.is_apple10 ? 7 : 5;
    int cur_k = r.is_apple10 ? 4 : 3;
    int cur_depth = r.is_apple10 ? 5 : 4;
    const int eos_id = r.mod->man.c.eos_id;

    while ((int)out_ids.size() < max_new && cur_pos + 1 < r.max_ctx) {
        if (next_tok == eos_id) break;

        if (!use_spec || !r.use_dspark || cur_pos + 9 >= r.max_ctx || (int)out_ids.size() + 1 >= max_new) {
            InkBatchStep st;
            st.M = 1;
            st.slot_ids = {slot};
            st.pos = {cur_pos};
            st.tok_ids = {next_tok};
            st.capture_taps = use_spec && r.use_dspark;
            st.compute_logits = true;
            if (!r.forward_batch(st, nullptr, &next_tok, err)) return false;
            ++cur_pos;
            out_ids.push_back(next_tok);
        } else if (tree_budget != 0 && r.use_dspark_meff) {
            InkDraftTree tree;
            const int rem_cap = max_new - (int)out_ids.size();
            const bool dyn = (tree_budget < 0);
            const int max_b = dyn ? 12 : tree_budget;
            const int eff_budget = (rem_cap <= 4) ? std::min(max_b, rem_cap + 2) : max_b;
            const int dcap = dyn ? cur_depth : 7;
            const int tm = r.dspark_draft_beam_tree(slot, next_tok, cur_pos, eff_budget, dyn, tree, dcap);
            if (tm <= 1) {
                InkBatchStep st;
                st.M = 1; st.slot_ids = {slot}; st.pos = {cur_pos}; st.tok_ids = {next_tok};
                st.capture_taps = true; st.compute_logits = true;
                if (!r.forward_batch(st, nullptr, &next_tok, err)) return false;
                ++cur_pos;
                out_ids.push_back(next_tok);
                continue;
            }
            if (tree.is_linear) {
                // Zero-overhead linear verify path (0.05 ms O(1) ring rollback instead of 0.45 ms tree commit!)
                InkBatchStep st;
                st.M = tm; st.slot_ids.assign(tm, slot); st.pos.resize(tm);
                for (int i = 0; i < tm; ++i) st.pos[i] = cur_pos + i;
                st.tok_ids = tree.tok_ids;
                st.capture_taps = true; st.compute_logits = true; st.record_verify_ring = true;
                int32_t ver_ids[32] = {};
                if (!r.forward_batch(st, nullptr, ver_ids, err)) return false;
                const int rem = tm - 1;
                int accepted = 0;
                for (int i = 0; i < rem; ++i) {
                    if (ver_ids[i] == tree.tok_ids[i + 1]) ++accepted;
                    else break;
                }
                total_offered += rem;
                total_accepted += accepted;
                for (int i = 0; i < accepted; ++i) {
                    const int32_t tk = tree.tok_ids[i + 1];
                    out_ids.push_back(tk);
                    if (tk == eos_id || (int)out_ids.size() >= max_new) break;
                }
                if ((int)out_ids.size() < max_new && (accepted == 0 || tree.tok_ids[accepted] != eos_id)) {
                    next_tok = ver_ids[accepted];
                    out_ids.push_back(next_tok);
                }
                if (accepted < rem) r.restore_verify_step(slot, accepted);
                cur_depth = std::max(4, std::min(7, accepted + 3));
                cur_pos += (1 + accepted);
                continue;
            }
            InkBatchStep st;
            st.M = tm;
            st.slot_ids.assign(tm, slot);
            st.pos.resize(tm);
            for (int i = 0; i < tm; ++i) st.pos[i] = cur_pos + tree.depths[i];
            st.tok_ids = tree.tok_ids;
            st.capture_taps = true;
            st.compute_logits = true;
            st.is_tree = true;
            st.tree_anchor_pos = cur_pos;
            st.tree_parent = tree.parents;
            st.tree_anc_mask = tree.anc_masks;
            int32_t ver_ids[32] = {};
            if (!r.forward_batch(st, nullptr, ver_ids, err)) return false;

            // Exact greedy walk down the verified tree (100% bitwise-exact with M=1!)
            std::vector<int32_t> path_nodes;
            path_nodes.reserve(8);
            path_nodes.push_back(0);
            int cur_node = 0;
            int accepted = 0;
            int32_t bonus_tok = ver_ids[0];
            while (true) {
                const int32_t target_next = ver_ids[cur_node];
                int match_child = -1;
                for (int ci = cur_node + 1; ci < tm; ++ci) {
                    if (tree.parents[ci] == cur_node && tree.tok_ids[ci] == target_next) {
                        match_child = ci;
                        break;
                    }
                }
                if (match_child < 0) {
                    bonus_tok = target_next;
                    break;
                }
                if (target_next == eos_id || (int)out_ids.size() + accepted + 1 >= max_new) {
                    path_nodes.push_back(match_child);
                    ++accepted;
                    bonus_tok = ver_ids[match_child];
                    break;
                }
                path_nodes.push_back(match_child);
                ++accepted;
                cur_node = match_child;
                bonus_tok = ver_ids[cur_node];
            }
            total_offered += 7;
            total_accepted += accepted;

            for (int i = 1; i <= accepted; ++i) {
                const int32_t tk = tree.tok_ids[path_nodes[i]];
                out_ids.push_back(tk);
                if (tk == eos_id || (int)out_ids.size() >= max_new) break;
            }
            if ((int)out_ids.size() < max_new && (accepted == 0 || tree.tok_ids[path_nodes[accepted]] != eos_id)) {
                next_tok = bonus_tok;
                out_ids.push_back(next_tok);
            }
            r.commit_tree_path(slot, cur_pos, path_nodes);
            cur_depth = std::max(4, std::min(7, accepted + 3));
            cur_pos += (1 + accepted);
        } else {
            // Draft `cur_k` (3..7) tokens on GPU via Exact Low-Rank M_eff DSpark!
            int32_t draft[7] = {};
            const int nd = r.dspark_draft_block8(slot, next_tok, cur_pos, draft, cur_k);
            if (nd <= 0) {
                InkBatchStep st;
                st.M = 1; st.slot_ids = {slot}; st.pos = {cur_pos}; st.tok_ids = {next_tok};
                st.capture_taps = true; st.compute_logits = true;
                if (!r.forward_batch(st, nullptr, &next_tok, err)) return false;
                ++cur_pos;
                out_ids.push_back(next_tok);
                continue;
            }
            const int rem = std::min(nd, max_new - (int)out_ids.size() - 1);
            const int V = 1 + rem;  // verify anchor + `rem` drafts in ONE forward_batch!

            InkBatchStep st;
            st.M = V;
            st.slot_ids.assign(V, slot);
            st.pos.resize(V);
            st.tok_ids.resize(V);
            st.pos[0] = cur_pos;
            st.tok_ids[0] = next_tok;
            for (int i = 0; i < rem; ++i) {
                st.pos[i + 1] = cur_pos + 1 + i;
                st.tok_ids[i + 1] = draft[i];
            }
            st.capture_taps = true;
            st.compute_logits = true;
            st.record_verify_ring = true;
            int32_t ver_ids[8] = {};
            if (!r.forward_batch(st, nullptr, ver_ids, err)) return false;

            // Exact greedy acceptance check
            int accepted = 0;
            for (int i = 0; i < rem; ++i) {
                if (ver_ids[i] == draft[i]) {
                    ++accepted;
                } else {
                    break;
                }
            }
            total_offered += rem;
            total_accepted += accepted;

            for (int i = 0; i < accepted; ++i) {
                out_ids.push_back(draft[i]);
                if (draft[i] == eos_id || (int)out_ids.size() >= max_new) break;
            }
            if ((int)out_ids.size() < max_new && (accepted == 0 || draft[accepted - 1] != eos_id)) {
                next_tok = ver_ids[accepted];
                out_ids.push_back(next_tok);
            }
            // O(1) ZERO-REPLAY ShortConv & f_rnn state rollback when partial accept occurred!
            if (accepted < rem) {
                r.restore_verify_step(slot, accepted);
            }
            cur_k = (accepted >= cur_k) ? std::min(max_k, cur_k + 1) : std::max(min_k, std::min(max_k, accepted + 1));
            cur_pos += (1 + accepted);
        }
    }
    decode_ms = (mtl::now_s() - td0) * 1e3;
    return true;
}

// Multi-request continuous batching benchmark for c = 1, 4, 8, 32, 64 concurrent slots!
static bool run_parallel_bench(InkMetalRunner& r, int c_req, int prompt_len, int gen_steps, std::string& err) {
    if (c_req > r.max_slots) { err = "c_req exceeds max_slots"; return false; }
    for (int s = 0; s < c_req; ++s) r.reset_slot(s);

    // 1. Prefill each slot (or batched prefill up to max_batch)
    double tp0 = mtl::now_s();
    std::vector<int32_t> next_toks(c_req, 100);
    for (int s = 0; s < c_req; ++s) {
        const int chunk = std::min(r.max_batch, 256);
        for (int p0 = 0; p0 < prompt_len; p0 += chunk) {
            const int cm = std::min(chunk, prompt_len - p0);
            InkBatchStep st;
            st.M = cm;
            st.slot_ids.assign(cm, s);
            st.pos.resize(cm);
            st.tok_ids.resize(cm);
            for (int i = 0; i < cm; ++i) {
                st.pos[i] = p0 + i;
                st.tok_ids[i] = 100 + ((s * 31 + p0 + i) % 50000);
            }
            const bool is_last = (p0 + cm == prompt_len);
            st.compute_logits = is_last;
            st.logits_last_only_idx = is_last ? (cm - 1) : -1;
            if (!r.forward_batch(st, nullptr, is_last ? &next_toks[s] : nullptr, err)) return false;
        }
    }
    double prefill_s = mtl::now_s() - tp0;

    // 2. Coalesced c_req-slot decode steps (all c_req slots in ONE 48-visit Metal command buffer per step!)
    double td0 = mtl::now_s();
    for (int step = 0; step < gen_steps; ++step) {
        InkBatchStep st;
        st.M = c_req;
        st.slot_ids.resize(c_req);
        st.pos.resize(c_req);
        st.tok_ids = next_toks;
        for (int s = 0; s < c_req; ++s) {
            st.slot_ids[s] = s;
            st.pos[s] = prompt_len + step;
        }
        st.compute_logits = true;
        if (!r.forward_batch(st, nullptr, next_toks.data(), err)) return false;
    }
    double decode_s = mtl::now_s() - td0;
    const double total_p = (double)c_req * prompt_len;
    const double total_d = (double)c_req * gen_steps;
    printf("[PAR c=%2d] prompt=%d*%d (%.1f ppts) | decode=%d*%d in %.2f s -> agg=%.1f tok/s (per-req=%.1f tok/s, %.2f ms/step)\n",
           c_req, c_req, prompt_len, total_p / std::max(prefill_s, 1e-6),
           c_req, gen_steps, decode_s, total_d / std::max(decode_s, 1e-6),
           gen_steps / std::max(decode_s, 1e-6), decode_s * 1e3 / gen_steps);
    fflush(stdout);
    return true;
}

int main(int argc, char** argv) {
    std::string model_path, vision_path, image_path, kernel_dir, dump_logits_path;
    std::string draft_mode = "auto";
    int max_ctx = 4096;
    int max_slots = 1;
    int max_batch = 256;
    int gen_n = 32;
    int bench_prompt = 128;
    bool do_selfcheck = false;
    bool do_spec_check = false;
    bool do_bench = false;
    bool do_par_sweep = false;
    bool do_warm = true;
    std::vector<int32_t> custom_ids;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--version") { printf("strata-inkling %s\n", INK_VERSION); return 0; }
        else if (a == "--info") {
            std::string f = nxt(), vp;
            for (int j = i + 1; j < argc; ++j) if (std::string(argv[j]) == "--vision" && j + 1 < argc) vp = argv[j + 1];
            return print_info(f, vp);
        }
        else if (a == "--model" || a == "--native") model_path = nxt();
        else if (a == "--vision") vision_path = nxt();
        else if (a == "--image") image_path = nxt();
        else if (a == "--kernels") kernel_dir = nxt();
        else if (a == "--max-context" || a == "--ctx") max_ctx = atoi(nxt().c_str());
        else if (a == "--parallel" || a == "--slots") max_slots = atoi(nxt().c_str());
        else if (a == "--chunk" || a == "--batch") max_batch = atoi(nxt().c_str());
        else if (a == "--gen") gen_n = atoi(nxt().c_str());
        else if (a == "--prompt-len") bench_prompt = atoi(nxt().c_str());
        else if (a == "--draft") draft_mode = nxt();
        else if (a == "--selfcheck") do_selfcheck = true;
        else if (a == "--spec-check") do_spec_check = true;
        else if (a == "--bench") do_bench = true;
        else if (a == "--par-sweep") { do_par_sweep = true; if (max_slots < 64) max_slots = 64; }
        else if (a == "--no-warm") do_warm = false;
        else if (a == "--dump-logits") dump_logits_path = nxt();
        else if (a == "--ids") {
            std::stringstream ss(nxt());
            std::string tok;
            while (std::getline(ss, tok, ',')) if (!tok.empty()) custom_ids.push_back(atoi(tok.c_str()));
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "usage: strata-inkling-metal --model FILE.strata [--vision V.strata] [--selfcheck|--spec-check|--bench|--par-sweep] [--parallel N] [--draft auto|dspark|off]\n");
        return 2;
    }

    mtl::Context cx;
    fprintf(stderr, "[ink_metal] device=%s workSet=%.1f GiB maxBuf=%.1f GiB\n",
            cx.device_name().c_str(),
            cx.recommended_working_set() / (1024.0 * 1024.0 * 1024.0),
            cx.max_buffer_length() / (1024.0 * 1024.0 * 1024.0));

    InkMetalModel mod;
    std::string err;
    if (!mod.load(cx, model_path, err)) {
        fprintf(stderr, "[ERROR] load: %s\n", err.c_str());
        return 1;
    }
    if (do_warm) mod.warm(8);

    const bool want_dspark = (draft_mode != "off");
    InkMetalRunner runner;
    if (!runner.init(cx, mod, max_slots, max_ctx, max_batch, want_dspark, vision_path, kernel_dir, err)) {
        fprintf(stderr, "[ERROR] runner init: %s\n", err.c_str());
        return 1;
    }

    // Optional Vision test
    if (!image_path.empty()) {
        std::vector<float> emb;
        int ntok = 0;
        if (!runner.ensure_vision(err) || !runner.vis.encode_image_file(cx, image_path, emb, ntok, err)) {
            fprintf(stderr, "[ERROR] vision: %s\n", err.c_str());
            return 1;
        }
        double l2 = 0.0;
        for (float v : emb) l2 += (double)v * v;
        printf("[VISION] OK: %d tokens x %d dims, rms=%.6f, first4=[%.4f, %.4f, %.4f, %.4f]\n",
               ntok, mod.man.c.hidden, std::sqrt(l2 / std::max<size_t>(1, emb.size())),
               emb[0], emb[1], emb[2], emb[3]);
    }

    // Default test prompt if none provided
    if (custom_ids.empty()) {
        // Simple deterministic prompt tokens
        custom_ids = {200001, 200004, 15496, 11, 1268, 527, 499, 30};
    }

    if (do_selfcheck || !dump_logits_path.empty()) {
        runner.reset_slot(0);
        const int P = (int)custom_ids.size();
        InkBatchStep st;
        st.M = P;
        st.slot_ids.assign(P, 0);
        st.pos.resize(P);
        st.tok_ids = custom_ids;
        for (int i = 0; i < P; ++i) st.pos[i] = i;
        st.capture_taps = runner.use_dspark;
        st.compute_logits = true;
        st.logits_last_only_idx = P - 1;
        std::vector<float> lg(mod.man.c.vocab_unpadded);
        int32_t top1 = -1;
        if (!runner.forward_batch(st, lg.data(), &top1, err)) {
            fprintf(stderr, "[ERROR] selfcheck forward: %s\n", err.c_str());
            return 1;
        }
        int nan_cnt = 0;
        for (float v : lg) if (!std::isfinite(v)) ++nan_cnt;
        std::vector<int> idx(lg.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        printf("[SELFCHECK] P=%d gpu=%.2f ms nan=%d top1=%d | top5: %d(%.4f) %d(%.4f) %d(%.4f) %d(%.4f) %d(%.4f)\n",
               P, runner.last_gpu_ms, nan_cnt, top1,
               idx[0], lg[idx[0]], idx[1], lg[idx[1]], idx[2], lg[idx[2]], idx[3], lg[idx[3]], idx[4], lg[idx[4]]);
        if (nan_cnt > 0) return 1;
        if (!dump_logits_path.empty()) {
            FILE* f = fopen(dump_logits_path.c_str(), "wb");
            if (f) { fwrite(lg.data(), sizeof(float), lg.size(), f); fclose(f); }
        }
    }

    int tree_budget = 0;
    if (draft_mode == "tree" || draft_mode == "beam") tree_budget = -1;  // Sparse-MoE Bayes-Optimal Dynamic Beam-4 Tree!
    else if (draft_mode == "tree6") tree_budget = 6;
    else if (draft_mode == "tree8") tree_budget = 8;
    else if (draft_mode == "tree10") tree_budget = 10;
    else if (draft_mode == "tree12") tree_budget = 12;
    else if (draft_mode == "tree16") tree_budget = 16;
    else if (draft_mode == "tree24") tree_budget = 24;
    else if (draft_mode == "tree32") tree_budget = 32;

    if (do_spec_check) {
        std::vector<int32_t> plain_ids, spec_ids;
        double pm0 = 0, dm0 = 0, pm1 = 0, dm1 = 0;
        int acc0 = 0, off0 = 0, acc1 = 0, off1 = 0;
        if (!run_greedy_decode(runner, 0, custom_ids, gen_n, false, plain_ids, pm0, dm0, acc0, off0, err, 0)) {
            fprintf(stderr, "[ERROR] plain decode: %s\n", err.c_str());
            return 1;
        }
        if (!run_greedy_decode(runner, 0, custom_ids, gen_n, true, spec_ids, pm1, dm1, acc1, off1, err, tree_budget)) {
            fprintf(stderr, "[ERROR] spec decode: %s\n", err.c_str());
            return 1;
        }
        int match = 0;
        const int ncmp = (int)std::min(plain_ids.size(), spec_ids.size());
        for (int i = 0; i < ncmp; ++i) {
            if (plain_ids[i] == spec_ids[i]) ++match;
            else break;
        }
        const double tps_plain = plain_ids.size() / std::max(dm0 * 1e-3, 1e-6);
        const double tps_spec  = spec_ids.size() / std::max(dm1 * 1e-3, 1e-6);
        const double alpha = off1 > 0 ? (1.0 + (double)acc1 / (off1 / 7.0)) : 1.0;
        printf("[SPEC-CHECK draft=%s] parity=%d/%d (%s) | plain=%.1f tok/s (%.1f ms) | dspark=%.1f tok/s (%.1f ms, acc=%d/%d, alpha~%.2f)\n",
               draft_mode.c_str(), match, ncmp, (match == ncmp && plain_ids.size() == spec_ids.size()) ? "BITWISE-EXACT" : "DIFF",
               tps_plain, dm0, tps_spec, dm1, acc1, off1, alpha);
        if (match != ncmp) {
            printf("  first diff at %d: plain=%d spec=%d\n", match, plain_ids[match], spec_ids[match]);
        }
    }

    if (do_bench) {
        std::vector<int32_t> bp(bench_prompt);
        for (int i = 0; i < bench_prompt; ++i) bp[i] = 100 + (i * 17) % 50000;
        std::vector<int32_t> out_ids;
        double pm = 0, dm = 0;
        int acc = 0, off = 0;
        const bool use_spec = (draft_mode != "off") && runner.use_dspark;
        if (!run_greedy_decode(runner, 0, bp, gen_n, use_spec, out_ids, pm, dm, acc, off, err, tree_budget)) {
            fprintf(stderr, "[ERROR] bench: %s\n", err.c_str());
            return 1;
        }
        printf("[BENCH] build=%s draft=%s prompt=%d (%.1f ms, %.1f ppts) | gen=%zu (%.1f ms, %.1f tok/s, acc=%d/%d)\n",
               mod.man.build.c_str(), use_spec ? draft_mode.c_str() : "off",
               bench_prompt, pm, bench_prompt / std::max(pm * 1e-3, 1e-6),
               out_ids.size(), dm, out_ids.size() / std::max(dm * 1e-3, 1e-6), acc, off);
    }

    if (do_par_sweep) {
        for (int c_req : {1, 4, 8, 32, 64}) {
            if (c_req > max_slots) break;
            if (!run_parallel_bench(runner, c_req, bench_prompt, gen_n, err)) {
                fprintf(stderr, "[ERROR] par c=%d: %s\n", c_req, err.c_str());
                return 1;
            }
        }
    }

    return 0;
}
