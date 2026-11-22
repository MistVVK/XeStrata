// engine/src/inkling2/cuda/ink_cuda_main.cu - Standalone Inkling-2-Mini CUDA Benchmark, Parity & Multi-Slot Runner
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Usage:
//   strata-inkling-cuda --model Inkling-2-Mini-SQ-20.strata [--gen 128] [--prompt-len 512] [--parallel 1|4|8|32|64]
//                       [--draft on|off] [--parity] [--kv-fp8] [--no-graph] [--dump-logits path.bin]
#include "ink_cuda.cuh"
#include "ink_dspark.cuh"
#include "ink_runner.cuh"
#include "ink_spec.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::ink;

static std::vector<int32_t> parse_ids_arg(const std::string& s) {
    std::vector<int32_t> ids;
    const char* p = s.c_str();
    while (*p) {
        while (*p == ',' || *p == ' ') ++p;
        if (!*p) break;
        char* e = nullptr;
        long v = std::strtol(p, &e, 10);
        if (e == p) break;
        ids.push_back((int32_t)v);
        p = e;
    }
    return ids;
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string tokens_arg;
    int gen_n = 128;
    int prompt_len = 512;
    int parallel = 1;
    int max_ctx = 4096;
    int chunk = 512;
    int gpu_id = 0;
    bool use_dspark = true;
    bool check_parity = false;
    bool kv_fp8 = false;
    bool use_graph = true;
    bool use_screen = true;
    bool prof = false;
    bool probe_early = false;
    int beam_w = 4;
    std::string dump_logits_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next_s = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : ""; };
        if (a == "--model" || a == "--native") model_path = next_s();
        else if (a == "--tokens") tokens_arg = next_s();
        else if (a == "--gen") gen_n = std::atoi(next_s().c_str());
        else if (a == "--prompt-len") prompt_len = std::atoi(next_s().c_str());
        else if (a == "--parallel" || a == "--slots") parallel = std::atoi(next_s().c_str());
        else if (a == "--ctx" || a == "--max-context") max_ctx = std::atoi(next_s().c_str());
        else if (a == "--chunk") chunk = std::atoi(next_s().c_str());
        else if (a == "--gpu") gpu_id = std::atoi(next_s().c_str());
        else if (a == "--draft") use_dspark = (next_s() != "off");
        else if (a == "--parity") check_parity = true;
        else if (a == "--kv-fp8") kv_fp8 = true;
        else if (a == "--no-graph") use_graph = false;
        else if (a == "--no-screen") use_screen = false;
        else if (a == "--prof") { prof = true; use_graph = false; }
        else if (a == "--probe-early-exit") probe_early = true;
        else if (a == "--beam-w") beam_w = std::atoi(next_s().c_str());
        else if (a == "--dump-logits") dump_logits_path = next_s();
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "Usage: %s --model FILE.strata [--gen 128] [--prompt-len 512] [--parallel 1|4|8|32|64] [--draft on|off] [--parity] [--kv-fp8]\n", argv[0]);
        return 1;
    }

    cudaDeviceProp prop{};
    INK_CK(cudaGetDeviceProperties(&prop, gpu_id));
    std::printf("GPU: %s (sm_%d%d, %d SMs, %.1f GiB VRAM, sm120_tc=%d)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), (int)sm120::available());

    // Auto-select FP8 KV for high concurrency (16/32/64 slots) on < 40 GB VRAM cards BEFORE pinning experts!
    InkKvDtype kvd = kv_fp8 ? InkKvDtype::FP8 : InkKvDtype::FP16;
    if (!kv_fp8 && parallel >= 16 && prop.totalGlobalMem < (40ull << 30)) {
        kvd = InkKvDtype::FP8;
        std::printf("Auto-enabling FP8 KV cache for parallel=%d on %.1f GiB VRAM\n", parallel, (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    size_t extra_reserve = 0;
    if (parallel >= 8) {
        const size_t swa_r = (parallel >= 16) ? (size_t)(kInkSwaWin + 32) : (size_t)(kInkSwaWin + std::max(chunk, 64));
        const size_t bpe = (kvd == InkKvDtype::FP8) ? 2ull : 4ull;
        const size_t sc_per_pos = (kvd == InkKvDtype::FP8) ? (size_t)(2 * kInkKVHeads * 4) : 0ull;
        extra_reserve = (size_t)40 * parallel * swa_r * ((size_t)kInkKVDim * bpe + sc_per_pos)
                      + (size_t)kInkVisits * parallel * 3 * kInkSconvCh * 4
                      + (size_t)8 * parallel * 256 * ((size_t)kInkKVDim * bpe + sc_per_pos)
                      + (1536ull << 20);
    }

    InkCudaModel model;
    std::string err;
    if (!model.load(model_path, gpu_id, use_screen, err, extra_reserve)) {
        std::fprintf(stderr, "Load failed: %s\n", err.c_str());
        return 1;
    }

    InkCudaRunner runner(model);
    runner.use_cuda_graph = use_graph;
    runner.init(parallel, max_ctx, chunk, kvd);

    InkDSpark dspark(runner);
    if (use_dspark && model.man.has_dspark) dspark.init(runner.max_ctx, 512);

    // Build prompt token IDs (realistic default prompt if --tokens not passed)
    std::vector<int32_t> prompt;
    if (!tokens_arg.empty()) {
        prompt = parse_ids_arg(tokens_arg);
    } else {
        // Synthetic non-trivial prompt tokens inside valid vocab [100..50000]
        prompt.resize((size_t)prompt_len);
        for (int i = 0; i < prompt_len; ++i) prompt[i] = 1000 + ((i * 7919) % 40000);
    }
    const int P = (int)prompt.size();

    // ---- 1. Prompt Prefill Benchmark (ppts) ----
    runner.reset_slot(0);
    runner.prof_enabled = prof;
    runner.prof_reset();
    INK_CK(cudaDeviceSynchronize());
    double t_pp0 = now_s();
    double dt_ingest_s = 0.0;
    for (int off = 0; off < P; off += chunk) {
        const int n = std::min(chunk, P - off);
        runner.forward_seq(prompt.data() + off, off, n, 0, dspark.ready);
        if (dspark.ready) {
            if (prof) INK_CK(cudaStreamSynchronize(runner.st));
            double ti0 = now_s();
            dspark.ingest_taps(off, n);
            if (prof) { INK_CK(cudaStreamSynchronize(runner.st)); dt_ingest_s += now_s() - ti0; }
        }
    }
    INK_CK(cudaStreamSynchronize(runner.st));
    double dt_pp = now_s() - t_pp0;
    std::printf("[PREFILL] %d tokens in %.2f ms -> %.1f ppts", P, dt_pp * 1000.0, (double)P / dt_pp);
    if (prof && dspark.ready) std::printf(" (dspark_ingest=%.2f ms)", dt_ingest_s * 1000.0);
    std::printf("\n");
    if (prof) runner.prof_print("PREFILL", 1);

    // Emit first token from last prompt position!
    const int last_chunk = ((P - 1) % chunk) + 1;
    if (!dump_logits_path.empty()) {
        runner.head_logits_full(last_chunk);
        const int V = model.man.c.vocab_unpadded;
        std::vector<float> h_log((size_t)last_chunk * V);
        INK_CK(cudaMemcpy(h_log.data(), runner.d_logits, h_log.size() * 4, cudaMemcpyDeviceToHost));
        FILE* fp = std::fopen(dump_logits_path.c_str(), "wb");
        if (fp) {
            std::fwrite(h_log.data(), 4, h_log.size(), fp);
            std::fclose(fp);
            std::printf("[DUMP] Wrote %d x %d float32 logits to %s\n", last_chunk, V, dump_logits_path.c_str());
        }
    }
    std::vector<int32_t> arg_buf(last_chunk);
    runner.head_argmax_exact(last_chunk, arg_buf.data());
    int32_t first_tok = arg_buf[last_chunk - 1];
    std::printf("First greedy token after prompt: %d\n", first_tok);

    // ---- 2. Single-Stream Plain Decode Benchmark ----
    std::vector<int32_t> plain_toks;
    plain_toks.reserve(gen_n);
    plain_toks.push_back(first_tok);
    {
        // Warmup 2 steps for CUDA Graph capture
        int32_t cur = first_tok;
        int pos = P;
        double t_dec0 = 0.0;
        for (int step = 1; step < gen_n; ++step) {
            if (step == 3) {
                INK_CK(cudaStreamSynchronize(runner.st));
                runner.prof_reset();
                t_dec0 = now_s();
            }
            runner.forward_seq(&cur, pos, 1, 0, false);
            int32_t nxt = 0;
            runner.head_argmax_exact(1, &nxt);
            plain_toks.push_back(nxt);
            cur = nxt;
            ++pos;
        }
        INK_CK(cudaStreamSynchronize(runner.st));
        if (gen_n > 3) {
            double dt_dec = now_s() - t_dec0;
            int measured = std::max(1, gen_n - 3);
            std::printf("[PLAIN DECODE c=1] %d tokens in %.2f ms -> %.1f tps (%.2f ms/tok)\n",
                        measured, dt_dec * 1000.0, (double)measured / dt_dec, (dt_dec * 1000.0) / measured);
            if (prof) runner.prof_print("DECODE-M1", measured);
        }
        std::printf("  first 16 plain tokens:");
        for (int i = 0; i < std::min(16, (int)plain_toks.size()); ++i) std::printf(" %d", plain_toks[i]);
        std::printf("\n");
    }

    // ---- 2B. Lossless DSpark Block-8 Speculative Decode Benchmark + Bitwise Parity Check! ----
    if (use_dspark && dspark.ready && gen_n > 8) {
        for (int n_draft : {4, 7}) {
            runner.reset_slot(0);
            for (int off = 0; off < P; off += chunk) {
                const int n = std::min(chunk, P - off);
                runner.forward_seq(prompt.data() + off, off, n, 0, true);
                dspark.ingest_taps(off, n);
            }
            INK_CK(cudaStreamSynchronize(runner.st));
            // Warmup 1 spec step (capture verify CUDA Graph!)
            std::vector<int32_t> spec_toks;
            spec_toks.reserve(gen_n + 16);
            spec_toks.push_back(first_tok);
            int32_t cur = first_tok;
            int pos = P;
            InkSpecStats warm_st{}, st{};
            int32_t emit[16];
            int n0 = ink_spec_step_block8(runner, dspark, cur, pos, n_draft, emit, warm_st);
            for (int i = 0; i < n0 && (int)spec_toks.size() < gen_n; ++i) spec_toks.push_back(emit[i]);

            INK_CK(cudaStreamSynchronize(runner.st));
            double t_sp0 = now_s();
            while ((int)spec_toks.size() < gen_n) {
                int ne = ink_spec_step_block8(runner, dspark, cur, pos, n_draft, emit, st);
                for (int i = 0; i < ne && (int)spec_toks.size() < gen_n; ++i) spec_toks.push_back(emit[i]);
            }
            INK_CK(cudaStreamSynchronize(runner.st));
            double dt_sp = now_s() - t_sp0;
            const int meas_toks = (int)st.tokens;
            const double mean_acc = st.iters > 0 ? (double)st.tokens / (double)st.iters : 0.0;
            std::printf("[DSPARK BLOCK-%d c=1] %d tokens (%ld iters, %.2f tok/iter) in %.2f ms -> %.1f tps (draft=%.2fms verify=%.2fms commit=%.2fms /iter)\n",
                        n_draft + 1, meas_toks, st.iters, mean_acc, dt_sp * 1000.0,
                        (double)meas_toks / dt_sp,
                        st.t_draft_ms / std::max(1L, st.iters),
                        st.t_verify_ms / std::max(1L, st.iters),
                        st.t_commit_ms / std::max(1L, st.iters));
            if (check_parity) {
                int match = 0, cmp_n = std::min((int)plain_toks.size(), (int)spec_toks.size());
                int first_diff = -1;
                for (int i = 0; i < cmp_n; ++i) {
                    if (plain_toks[i] == spec_toks[i]) ++match;
                    else if (first_diff < 0) first_diff = i;
                }
                std::printf("  [PARITY vs PLAIN] %d/%d EXACT (first_diff=%d)\n", match, cmp_n, first_diff);
                if (first_diff >= 0) {
                    std::printf("    plain[%d]=%d vs spec[%d]=%d\n", first_diff, plain_toks[first_diff], first_diff, spec_toks[first_diff]);
                }
            }
        }
        // Also benchmark Lossless Tree-12 2-branch Markov Beam!
        {
            runner.reset_slot(0);
            for (int off = 0; off < P; off += chunk) {
                const int n = std::min(chunk, P - off);
                runner.forward_seq(prompt.data() + off, off, n, 0, true);
                dspark.ingest_taps(off, n);
            }
            INK_CK(cudaStreamSynchronize(runner.st));
            std::vector<int32_t> spec_toks;
            spec_toks.reserve(gen_n + 16);
            spec_toks.push_back(first_tok);
            int32_t cur = first_tok;
            int pos = P;
            InkSpecStats warm_st{}, st{};
            int32_t emit[16];
            int n0 = ink_spec_step_tree(runner, dspark, cur, pos, 7, 12, emit, warm_st);
            for (int i = 0; i < n0 && (int)spec_toks.size() < gen_n; ++i) spec_toks.push_back(emit[i]);
            INK_CK(cudaStreamSynchronize(runner.st));
            double t_sp0 = now_s();
            while ((int)spec_toks.size() < gen_n) {
                int ne = ink_spec_step_tree(runner, dspark, cur, pos, 7, 12, emit, st);
                for (int i = 0; i < ne && (int)spec_toks.size() < gen_n; ++i) spec_toks.push_back(emit[i]);
            }
            INK_CK(cudaStreamSynchronize(runner.st));
            double dt_sp = now_s() - t_sp0;
            const int meas_toks = (int)st.tokens;
            const double mean_acc = st.iters > 0 ? (double)st.tokens / (double)st.iters : 0.0;
            std::printf("[DSPARK TREE-12 c=1] %d tokens (%ld iters, %.2f tok/iter) in %.2f ms -> %.1f tps (draft=%.2fms verify=%.2fms commit=%.2fms /iter)\n",
                        meas_toks, st.iters, mean_acc, dt_sp * 1000.0,
                        (double)meas_toks / dt_sp,
                        st.t_draft_ms / std::max(1L, st.iters),
                        st.t_verify_ms / std::max(1L, st.iters),
                        st.t_commit_ms / std::max(1L, st.iters));
            if (check_parity) {
                int match = 0, cmp_n = std::min((int)plain_toks.size(), (int)spec_toks.size());
                int first_diff = -1;
                for (int i = 0; i < cmp_n; ++i) {
                    if (plain_toks[i] == spec_toks[i]) ++match;
                    else if (first_diff < 0) first_diff = i;
                }
                std::printf("  [PARITY vs PLAIN] %d/%d EXACT (first_diff=%d)\n", match, cmp_n, first_diff);
            }
        }
        // Benchmark True W=4 Multi-Branch Bayes-Optimal Markov Beam Trees (fixed M = 6, 8, 12, 16 + Sparse-MoE Adaptive!)
        struct BeamCfg { const char* name; int max_nodes; float moe_ratio; };
        const BeamCfg cfgs[] = {
            {"BEAM-W4 TREE-6",         6, 0.0f},
            {"BEAM-W4 TREE-8",         8, 0.0f},
            {"BEAM-W4 ADAPT-3..12 r35",12, 3.5f},
            {"BEAM-W4 ADAPT-3..12 r42",12, 4.2f},
        };
        for (const auto& cfg : cfgs) {
            if (cfg.max_nodes > runner.tree_cap) continue;
            runner.reset_slot(0);
            for (int off = 0; off < P; off += chunk) {
                const int n = std::min(chunk, P - off);
                runner.forward_seq(prompt.data() + off, off, n, 0, true);
                dspark.ingest_taps(off, n);
            }
            INK_CK(cudaStreamSynchronize(runner.st));
            std::vector<int32_t> spec_toks;
            spec_toks.reserve(gen_n + 32);
            spec_toks.push_back(first_tok);
            int32_t cur = first_tok;
            int pos = P;
            InkSpecStats warm_st{}, st{};
            int32_t emit[kInkMaxTree];
            int n0 = ink_spec_step_beam_tree(runner, dspark, cur, pos, beam_w, cfg.max_nodes, emit, warm_st, cfg.moe_ratio);
            for (int i = 0; i < n0 && (int)spec_toks.size() < gen_n; ++i) spec_toks.push_back(emit[i]);
            INK_CK(cudaStreamSynchronize(runner.st));
            double t_sp0 = now_s();
            while ((int)spec_toks.size() < gen_n) {
                int ne = ink_spec_step_beam_tree(runner, dspark, cur, pos, beam_w, cfg.max_nodes, emit, st, cfg.moe_ratio);
                for (int i = 0; i < ne && (int)spec_toks.size() < gen_n; ++i) spec_toks.push_back(emit[i]);
            }
            INK_CK(cudaStreamSynchronize(runner.st));
            double dt_sp = now_s() - t_sp0;
            const int meas_toks = (int)st.tokens;
            const double mean_acc = st.iters > 0 ? (double)st.tokens / (double)st.iters : 0.0;
            const double mean_M = st.iters > 0 ? (double)(st.drafted + st.iters) / (double)st.iters : 0.0;
            std::printf("[DSPARK %s c=1] %d tokens (%ld iters, %.2f tok/iter, avgM=%.1f) in %.2f ms -> %.1f tps (draft=%.2fms verify=%.2fms commit=%.2fms /iter)\n",
                        cfg.name, meas_toks, st.iters, mean_acc, mean_M, dt_sp * 1000.0,
                        (double)meas_toks / dt_sp,
                        st.t_draft_ms / std::max(1L, st.iters),
                        st.t_verify_ms / std::max(1L, st.iters),
                        st.t_commit_ms / std::max(1L, st.iters));
            if (check_parity) {
                int match = 0, cmp_n = std::min((int)plain_toks.size(), (int)spec_toks.size());
                int first_diff = -1;
                for (int i = 0; i < cmp_n; ++i) {
                    if (plain_toks[i] == spec_toks[i]) ++match;
                    else if (first_diff < 0) first_diff = i;
                }
                std::printf("  [PARITY vs PLAIN] %d/%d EXACT (first_diff=%d)\n", match, cmp_n, first_diff);
                if (first_diff >= 0) {
                    std::printf("    plain[%d]=%d vs spec[%d]=%d\n", first_diff, plain_toks[first_diff], first_diff, spec_toks[first_diff]);
                }
            }
        }
    }

    // ---- 2C. Empirical Early-Pass Exit Probe (1-Pass+Coda = 24 visits, 2-Pass+Coda = 36 visits vs 3-Pass = 48 visits!) ----
    if (probe_early && (int)plain_toks.size() >= 16) {
        const int probe_n = std::min((int)plain_toks.size() - 1, 64);
        for (int np : {1, 2}) {
            runner.max_passes_override = np;
            runner.reset_slot(0);
            for (int off = 0; off < P; off += chunk) {
                const int n = std::min(chunk, P - off);
                runner.forward_seq(prompt.data() + off, off, n, 0, false);
            }
            int agree = 0;
            double t_ep0 = now_s();
            for (int k = 0; k < probe_n; ++k) {
                // Feed exact teacher-forced plain_toks[k] and compare Early-Pass prediction vs plain_toks[k+1]!
                int32_t tok = plain_toks[k];
                runner.forward_seq(&tok, P + k, 1, 0, false);
                int32_t pred = 0;
                runner.head_argmax_exact(1, &pred);
                if (pred == plain_toks[k + 1]) ++agree;
            }
            double dt_ep = now_s() - t_ep0;
            std::printf("[EARLY-EXIT PROBE %d-PASS+CODA (%d visits)] Top-1 agree vs 3-Pass = %d/%d (%.1f%%), %.2f ms/tok\n",
                        np, 6 + np * 12 + 6, agree, probe_n, 100.0 * agree / std::max(1, probe_n),
                        (dt_ep * 1000.0) / std::max(1, probe_n));
        }
        runner.max_passes_override = 0;
    }

    // ---- 3. Multi-Request Parallel Continuous Batching Benchmark (c = parallel) ----
    if (parallel > 1) {
        for (int s = 0; s < parallel; ++s) runner.reset_slot(s);
        // Seed each slot with 16 tokens
        const int seed_len = 16;
        for (int s = 0; s < parallel; ++s) {
            runner.forward_seq(prompt.data(), 0, seed_len, s, false);
        }
        INK_CK(cudaStreamSynchronize(runner.st));

        std::vector<int32_t> b_toks(parallel, first_tok);
        std::vector<int32_t> b_slots(parallel);
        std::vector<int32_t> b_pos(parallel, seed_len);
        std::vector<int32_t> b_out(parallel);
        for (int s = 0; s < parallel; ++s) b_slots[s] = s;

        // Warmup 2 batched steps for CUDA Graph capture
        for (int w = 0; w < 2; ++w) {
            runner.forward_batched(b_toks.data(), b_slots.data(), b_pos.data(), parallel);
            runner.head_argmax_exact(parallel, b_out.data());
            for (int s = 0; s < parallel; ++s) { b_toks[s] = b_out[s]; ++b_pos[s]; }
        }
        const int steps = 32;
        INK_CK(cudaStreamSynchronize(runner.st));
        double t_b0 = now_s();
        for (int stp = 0; stp < steps; ++stp) {
            runner.forward_batched(b_toks.data(), b_slots.data(), b_pos.data(), parallel);
            runner.head_argmax_exact(parallel, b_out.data());
            for (int s = 0; s < parallel; ++s) { b_toks[s] = b_out[s]; ++b_pos[s]; }
        }
        INK_CK(cudaStreamSynchronize(runner.st));
        double dt_b = now_s() - t_b0;
        const int total_toks = steps * parallel;
        std::printf("[BATCHED DECODE c=%d] %d steps (%d tok) in %.2f ms -> %.1f agg tps (%.1f tps/req, %.2f ms/step)\n",
                    parallel, steps, total_toks, dt_b * 1000.0,
                    (double)total_toks / dt_b, (double)steps / dt_b, (dt_b * 1000.0) / steps);
    }

    (void)check_parity;
    dspark.free_all();
    runner.free_all();
    model.free_all();
    return 0;
}
