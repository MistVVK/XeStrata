// engine/src/inkling2/ink_main.cpp - Standalone `strata-inkling` Shape-Engine Binary for Inkling-2-Mini (.strata v1.1)
// Supports: --version, --info, --export-tokenizer, --selfcheck, --bench (--parallel 1|4|8|32|64), --kld-eval, --serve, and direct CLI generation.
// MIT License - Copyright (c) 2026 Strata contributors
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "ink_backend.h"
#include "ink_batch.h"
#include "ink_manifest.h"
#include "ink_serve.h"
#include "ink_vision.h"

using namespace ink2;

#ifndef INKLING_VERSION
#define INKLING_VERSION "0.1.65"
#endif

static int cmd_export_tokenizer(const std::string& file, const std::string& dir) {
    InkModelFile mf;
    std::string err;
    if (!mf.open(file, err, /*keep_tokenizer=*/true)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    std::filesystem::create_directories(dir);
    auto write_str = [&](const std::string& name, const std::string& content) -> bool {
        if (content.empty()) return true;
        std::string path = dir + "/" + name;
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f || std::fwrite(content.data(), 1, content.size(), f) != content.size()) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            if (f) std::fclose(f);
            return false;
        }
        std::fclose(f);
        return true;
    };
    if (!write_str("chat_template.jinja", mf.m.chat_template)) return 1;
    if (!write_str("tokenizer.hf.json", mf.m.tokenizer_json)) return 1;
    printf("format=%s chat_template: %zu bytes, tokenizer_json: %zu bytes\n",
           source_kind_name(mf.kind), mf.m.chat_template.size(), mf.m.tokenizer_json.size());
    return 0;
}

static int cmd_info(const std::string& file, const std::string& vision, const std::string& dspark_path) {
    std::string err;
    // Check VisionShard (.strata) first
    if (!SafetensorsAdapter::is_safetensors_path(file) && !GgufAdapter::is_gguf_path(file)) {
        strata::artifact::StrataFileReader rd;
        if (!rd.open(file, err)) {
            fprintf(stderr, "open: %s\n", err.c_str());
            return 1;
        }
        if (rd.header().profile == strata::artifact::StrataProfile::VisionShard) {
            rd.close();
            VisionShardInfo vi;
            if (!vision_probe(file, vi, err)) {
                fprintf(stderr, "vision shard: %s\n", err.c_str());
                return 1;
            }
            printf("arch=inkling2_vision layers=%d hidden=%d out_hidden=%d patch=%d merge=%d tensors=%zu image_pad=%d\n",
                   vi.c.layers, vi.c.hidden, vi.c.out_hidden, vi.c.patch, vi.c.merge, vi.T.size(), vi.c.image_token);
            return 0;
        }
    }
    InkModelFile mf;
    if (!mf.open(file, err, /*keep_tokenizer=*/false, dspark_path)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    const InkManifest& m = mf.m;
    int images = 1;
    if (!vision.empty()) {
        VisionShardInfo vi;
        std::string verr;
        images = vision_probe(vision, vi, verr) ? 1 : 0;
        if (!images) fprintf(stderr, "vision shard: %s\n", verr.c_str());
    }
    printf("arch=inkling2_loop format=%s build=%s layers=%d visits=%d hidden=%d vocab=%d experts=%d eos=%d drafter=%s images=%d image_pad=%d parallel=%d\n",
           source_kind_name(mf.kind),
           m.build.empty() ? "unknown" : m.build.c_str(),
           m.c.n_phys, m.c.n_visits, m.c.hidden, m.c.vocab_unpadded, m.c.n_experts, m.c.eos_token_id,
           m.dspark.present ? "dspark" : "none", images, m.c.image_token_id, kMaxParallelSlots);
    return 0;
}

static int cmd_selfcheck(const std::string& model_path, bool force_cpu, bool verify_hash, const BackendOpts& bopts) {
    auto backend = make_default_backend(force_cpu, &bopts);
    std::string err;
    InkModelFile mf;
    if (!model_path.empty()) {
        if (!mf.open(model_path, err, false, bopts.dspark_path)) {
            fprintf(stderr, "[selfcheck] FAIL open: %s\n", err.c_str());
            return 1;
        }
        if (verify_hash) {
            printf("[selfcheck] Verifying SHA-256 of Sections 0..5...\n");
            if (!mf.verify_sections(err, true)) {
                fprintf(stderr, "[selfcheck] FAIL hash: %s\n", err.c_str());
                return 1;
            }
        }
        if (!backend->init(mf, bopts, err)) {
            fprintf(stderr, "[selfcheck] FAIL init: %s\n", err.c_str());
            return 1;
        }
    }
    if (!backend->selfcheck(err)) {
        fprintf(stderr, "[selfcheck] FAIL: %s\n", err.c_str());
        return 1;
    }
    printf("[selfcheck] PASS backend=%s (%s)\n", backend->backend_name(), backend->device_info().c_str());
    return 0;
}

static int cmd_bench(const std::string& model_path, const BackendOpts& bopts, int prompt_len, int gen_len, bool force_cpu, bool use_dspark) {
    InkModelFile mf;
    std::string err;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    BackendOpts bo = bopts;
    bo.enable_dspark = use_dspark;
    auto backend = make_default_backend(force_cpu, &bo);
    if (!backend->init(mf, bo, err)) {
        fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    const int P = std::max(1, std::min(kMaxParallelSlots, bo.parallel));
    const int V = mf.m.c.vocab_unpadded;
    for (int s = 0; s < P; ++s) {
        if (!backend->alloc_slot(s, bo.max_ctx, bo.kv_dtype, err)) {
            fprintf(stderr, "alloc_slot %d: %s\n", s, err.c_str());
            return 1;
        }
    }
    // Synthetic prompt per slot
    std::vector<int32_t> next_tok(P, 100);
    std::vector<float> logits(static_cast<size_t>(P) * V);
    StepBatch batch;
    double t_pf0 = now_s();
    for (int p0 = 0; p0 < prompt_len; p0 += bo.chunk) {
        int len = std::min(bo.chunk, prompt_len - p0);
        batch.clear();
        batch.capture_taps = use_dspark;
        for (int s = 0; s < P; ++s) {
            for (int k = 0; k < len; ++k) {
                bool is_last = (p0 + k == prompt_len - 1);
                batch.push(s, p0 + k, 100 + ((s * 31 + p0 + k) % 1000), is_last);
            }
        }
        if (!backend->forward_batch(batch, (p0 + len == prompt_len) ? logits.data() : nullptr, nullptr, err)) {
            fprintf(stderr, "bench prefill err: %s\n", err.c_str());
            return 1;
        }
    }
    double pf_s = now_s() - t_pf0;
    for (int s = 0; s < P; ++s) {
        const float* lg = logits.data() + static_cast<size_t>(s) * V;
        int32_t best = 0;
        for (int i = 1; i < V; ++i) if (lg[i] > lg[best]) best = i;
        next_tok[s] = best;
    }

    double t_dec0 = now_s();
    int total_gen = 0;
    if (use_dspark && P == 1) {
        int cur_pos = prompt_len;
        while (total_gen < gen_len) {
            const int rem = gen_len - total_gen;
            if (rem >= 2 && cur_pos + 12 < bo.max_ctx) {
                int32_t emitted[16]{};
                int drafted = 0;
                const int n_emit = backend->dspark_spec_step_greedy(
                    0, next_tok[0], cur_pos, rem, emitted, &drafted, err);
                if (n_emit < 0) { fprintf(stderr, "bench spec err: %s\n", err.c_str()); return 1; }
                if (n_emit > 0) {
                    next_tok[0] = emitted[n_emit - 1];
                    cur_pos += n_emit;
                    total_gen += n_emit;
                    continue;
                }
            }
            batch.clear();
            batch.capture_taps = true;
            batch.push(0, cur_pos++, next_tok[0], true);
            if (!backend->forward_batch(batch, logits.data(), nullptr, err)) {
                fprintf(stderr, "bench decode err: %s\n", err.c_str());
                return 1;
            }
            int32_t best = 0;
            for (int i = 1; i < V; ++i) if (logits[i] > logits[best]) best = i;
            next_tok[0] = best;
            ++total_gen;
        }
    } else {
        for (int step = 0; step < gen_len; ++step) {
            batch.clear();
            batch.capture_taps = use_dspark;
            for (int s = 0; s < P; ++s) batch.push(s, prompt_len + step, next_tok[s], true);
            if (!backend->forward_batch(batch, logits.data(), nullptr, err)) {
                fprintf(stderr, "bench decode err: %s\n", err.c_str());
                return 1;
            }
            for (int s = 0; s < P; ++s) {
                const float* lg = logits.data() + static_cast<size_t>(s) * V;
                int32_t best = 0;
                for (int i = 1; i < V; ++i) if (lg[i] > lg[best]) best = i;
                next_tok[s] = best;
                ++total_gen;
            }
        }
    }
    double dec_s = now_s() - t_dec0;
    double ppts = (P * prompt_len) / std::max(pf_s, 1e-9);
    double tps_agg = total_gen / std::max(dec_s, 1e-9);
    printf("[bench] backend=%s format=%s parallel=%d gpus=%d kv=%s prompt=%d gen=%d | prefill=%.1f ppts (%.2fs) | decode=%.1f tps agg (%.1f tps/slot, %.2fs)\n",
           backend->backend_name(), source_kind_name(mf.kind), P, bo.gpus, kv_dtype_name(bo.kv_dtype),
           prompt_len, gen_len, ppts, pf_s, tps_agg, tps_agg / P, dec_s);
    return 0;
}

// Teacher-forced KLD/PPL/Top-1 evaluation against a binary reference file:
// Header: int32 n_tokens, int32 vocab_unpadded, int32 tokens[n_tokens], float ref_logits[n_tokens * vocab_unpadded]
static int cmd_kld_eval(const std::string& model_path, const std::string& ref_path, const BackendOpts& bopts, bool force_cpu) {
    FILE* f = std::fopen(ref_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "kld-eval: cannot open %s\n", ref_path.c_str()); return 1; }
    int32_t n_tok = 0, V_ref = 0;
    if (std::fread(&n_tok, 4, 1, f) != 1 || std::fread(&V_ref, 4, 1, f) != 1 || n_tok <= 1) {
        fprintf(stderr, "kld-eval: invalid reference header\n");
        std::fclose(f);
        return 1;
    }
    std::vector<int32_t> ids(n_tok);
    std::vector<float> ref_lg(static_cast<size_t>(n_tok) * V_ref);
    if (std::fread(ids.data(), 4, n_tok, f) != static_cast<size_t>(n_tok) ||
        std::fread(ref_lg.data(), 4, ref_lg.size(), f) != ref_lg.size()) {
        fprintf(stderr, "kld-eval: truncated reference file\n");
        std::fclose(f);
        return 1;
    }
    std::fclose(f);

    InkModelFile mf;
    std::string err;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    auto backend = make_default_backend(force_cpu, &bopts);
    if (!backend->init(mf, bopts, err) || !backend->alloc_slot(0, bopts.max_ctx, bopts.kv_dtype, err)) {
        fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    const int V = mf.m.c.vocab_unpadded;
    if (V != V_ref) { fprintf(stderr, "vocab mismatch %d vs %d\n", V, V_ref); return 1; }

    std::vector<float> got_lg(static_cast<size_t>(n_tok) * V);
    StepBatch batch;
    for (int i = 0; i < n_tok; ++i) batch.push(0, i, ids[i], true);
    if (!backend->forward_batch(batch, got_lg.data(), nullptr, err)) {
        fprintf(stderr, "forward: %s\n", err.c_str());
        return 1;
    }

    double kld_sum = 0.0, nll_sum = 0.0;
    int top1_ok = 0;
    for (int t = 0; t < n_tok - 1; ++t) {
        const float* p_lg = ref_lg.data() + static_cast<size_t>(t) * V;
        const float* q_lg = got_lg.data() + static_cast<size_t>(t) * V;
        float mp = p_lg[0], mq = q_lg[0];
        int bp = 0, bq = 0;
        for (int v = 1; v < V; ++v) {
            if (p_lg[v] > mp) { mp = p_lg[v]; bp = v; }
            if (q_lg[v] > mq) { mq = q_lg[v]; bq = v; }
        }
        if (bp == bq) ++top1_ok;
        double zp = 0.0, zq = 0.0;
        for (int v = 0; v < V; ++v) { zp += std::exp(p_lg[v] - mp); zq += std::exp(q_lg[v] - mq); }
        double lzp = mp + std::log(zp), lzq = mq + std::log(zq);
        double kld = 0.0;
        for (int v = 0; v < V; ++v) {
            double lp = p_lg[v] - lzp;
            double lq = q_lg[v] - lzq;
            double pv = std::exp(lp);
            kld += pv * (lp - lq);
        }
        kld_sum += kld;
        nll_sum += -(q_lg[ids[t + 1]] - lzq);
    }
    const int n_eval = n_tok - 1;
    printf("[kld-eval] n=%d | Mean KLD=%.6f | Top-1=%.2f%% | PPL=%.3f\n",
           n_eval, kld_sum / n_eval, 100.0 * top1_ok / n_eval, std::exp(nll_sum / n_eval));
    return 0;
}

// Dump teacher-forced FP32 reference logits for --kld-eval parity checks across formats & backends
static int cmd_dump_ref(const std::string& model_path, const std::string& out_path,
                        const std::string& prompt_ids_str, const BackendOpts& bopts, bool force_cpu) {
    InkModelFile mf;
    std::string err;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    auto backend = make_default_backend(force_cpu, &bopts);
    if (!backend->init(mf, bopts, err) || !backend->alloc_slot(0, bopts.max_ctx, bopts.kv_dtype, err)) {
        fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> ids;
    if (!prompt_ids_str.empty()) {
        if (!parse_id_list(prompt_ids_str, ids, err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
    } else {
        ids = {1, 2, 3, 4, 5, 6, 7, 8};
    }
    const int32_t n_tok = static_cast<int32_t>(ids.size());
    const int32_t V = mf.m.c.vocab_unpadded;
    std::vector<float> logits(static_cast<size_t>(n_tok) * V);
    StepBatch batch;
    for (int i = 0; i < n_tok; ++i) batch.push(0, i, ids[i], true);
    if (!backend->forward_batch(batch, logits.data(), nullptr, err)) {
        fprintf(stderr, "forward: %s\n", err.c_str());
        return 1;
    }
    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) { fprintf(stderr, "dump-ref: cannot write %s\n", out_path.c_str()); return 1; }
    std::fwrite(&n_tok, 4, 1, f);
    std::fwrite(&V, 4, 1, f);
    std::fwrite(ids.data(), 4, n_tok, f);
    std::fwrite(logits.data(), 4, logits.size(), f);
    std::fclose(f);
    printf("[dump-ref] wrote %s (n_tok=%d, V=%d, backend=%s)\n", out_path.c_str(), n_tok, V, backend->backend_name());
    return 0;
}

int main(int argc, char** argv) {
    ServeConfig cfg;
    cfg.version = INKLING_VERSION;
    bool mode_serve = false, mode_bench = false, mode_selfcheck = false, mode_info = false, mode_export = false;
    bool verify_hash = false;
    std::string info_file, export_file, export_dir, kld_ref, dump_ref, prompt_ids_str;
    int bench_prompt = 64, bench_gen = 32;
    float cli_temp = 0.0f;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value after %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--version") {
            printf("strata-inkling %s\n", INKLING_VERSION);
            return 0;
        } else if (a == "--info") {
            mode_info = true; info_file = next();
        } else if (a == "--export-tokenizer") {
            mode_export = true; export_file = next(); export_dir = next();
        } else if (a == "--serve") {
            mode_serve = true;
        } else if (a == "--bench") {
            mode_bench = true;
        } else if (a == "--selfcheck") {
            mode_selfcheck = true;
        } else if (a == "--verify-hash") {
            verify_hash = true;
        } else if (a == "--kld-eval") {
            kld_ref = next();
        } else if (a == "--dump-ref") {
            dump_ref = next();
        } else if (a == "--native" || a == "--model") {
            cfg.model_path = next();
        } else if (a == "--vision") {
            cfg.vision_path = next();
            cfg.bopts.vision_path = cfg.vision_path;
        } else if (a == "--max-context" || a == "--ctx") {
            cfg.bopts.max_ctx = std::atoi(next().c_str());
        } else if (a == "--parallel" || a == "--batch" || a == "--slots") {
            cfg.bopts.parallel = std::max(1, std::min(kMaxParallelSlots, std::atoi(next().c_str())));
        } else if (a == "--chunk") {
            cfg.bopts.chunk = std::max(1, std::atoi(next().c_str()));
        } else if (a == "--kv-dtype" || a == "--kv") {
            std::string kv = next();
            if (kv == "fp8" || kv == "fp8_e4m3") { cfg.bopts.kv_dtype = KvDtype::FP8_E4M3; cfg.bopts.kv_dtype_auto = false; }
            else if (kv == "fp16" || kv == "f16") { cfg.bopts.kv_dtype = KvDtype::FP16; cfg.bopts.kv_dtype_auto = false; }
            else if (kv == "k8v8" || kv == "i8" || kv == "q8") { cfg.bopts.kv_dtype = KvDtype::K8V8; cfg.bopts.kv_dtype_auto = false; }
            else if (kv == "k8v4") { cfg.bopts.kv_dtype = KvDtype::K8V4; cfg.bopts.kv_dtype_auto = false; }
            else { cfg.bopts.kv_dtype_auto = true; }
        } else if (a == "--draft") {
            cfg.draft_mode = next();
            cfg.bopts.enable_dspark = (cfg.draft_mode != "off");
        } else if (a == "--dspark") {
            cfg.bopts.dspark_path = next();
        } else if (a == "--vram-experts-gb") {
            cfg.bopts.vram_experts_gb = std::atof(next().c_str());
        } else if (a == "--ram-experts-gb") {
            cfg.bopts.ram_experts_gb = std::atof(next().c_str());
        } else if (a == "--vram-reserve-mib") {
            cfg.bopts.vram_reserve_mib = std::atoi(next().c_str());
        } else if (a == "--cpu-threads") {
            cfg.bopts.cpu_threads = std::atoi(next().c_str());
        } else if (a == "--gpu") {
            cfg.bopts.gpu_index = std::atoi(next().c_str());
        } else if (a == "--gpus") {
            cfg.bopts.gpus = std::max(1, std::atoi(next().c_str()));
        } else if (a == "--ep-split") {
            cfg.bopts.ep_split = std::atof(next().c_str());
        } else if (a == "--threads") {
            cfg.bopts.threads = std::atoi(next().c_str());
        } else if (a == "--cpu") {
            cfg.force_cpu = true;
        } else if (a == "--prompt-len") {
            bench_prompt = std::atoi(next().c_str());
        } else if (a == "--gen") {
            bench_gen = std::atoi(next().c_str());
        } else if (a == "--prompt-ids") {
            prompt_ids_str = next();
        } else if (a == "--temp") {
            cli_temp = static_cast<float>(std::atof(next().c_str()));
        } else if (a == "--no-anti-loop") {
            cfg.anti_loop = false;
        } else if (a == "--help" || a == "-h") {
            printf("strata-inkling %s - Unified Inkling-2-Mini (.strata v1.1 / HF safetensors / .gguf) Shape Engine\n"
                   "  --version\n"
                   "  --info PATH [--vision VISION] [--dspark PATH]\n"
                   "  --export-tokenizer PATH OUT_DIR\n"
                   "  --selfcheck [--native PATH] [--verify-hash] [--cpu] [--gpus 1|2] [--kv fp16|fp8|k8v8|k8v4]\n"
                   "  --bench --native PATH [--parallel 1..64] [--gpus 1|2] [--ep-split F] [--vram-reserve-mib N] [--cpu-threads N] [--kv auto|fp16|fp8|k8v8|k8v4] [--prompt-len N] [--gen N]\n"
                   "  --kld-eval REF.bin --native PATH\n"
                   "  --serve --native PATH [--vision VISION] [--dspark PATH] [--max-context N] [--parallel 1..64] [--gpus 1|2] [--draft auto|dspark|off]\n",
                   INKLING_VERSION);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 2;
        }
    }

    // Read optional IK2_EP_SPLIT env override (#6506)
    if (const char* ep_env = std::getenv("IK2_EP_SPLIT"); ep_env && *ep_env) {
        double v = std::atof(ep_env);
        if (v > 0.0 && v < 1.0) cfg.bopts.ep_split = v;
    }

    // Auto-select FP8_E4M3 KV when parallel >= 16 unless user explicitly forced a format
    if (cfg.bopts.kv_dtype_auto && cfg.bopts.parallel >= 16) {
        cfg.bopts.kv_dtype = KvDtype::FP8_E4M3;
    }

    if (mode_info) return cmd_info(info_file, cfg.vision_path, cfg.bopts.dspark_path);
    if (mode_export) return cmd_export_tokenizer(export_file, export_dir);
    if (mode_selfcheck) return cmd_selfcheck(cfg.model_path, cfg.force_cpu, verify_hash, cfg.bopts);
    if (!kld_ref.empty()) return cmd_kld_eval(cfg.model_path, kld_ref, cfg.bopts, cfg.force_cpu);
    if (!dump_ref.empty()) return cmd_dump_ref(cfg.model_path, dump_ref, prompt_ids_str, cfg.bopts, cfg.force_cpu);
    if (mode_bench) return cmd_bench(cfg.model_path, cfg.bopts, bench_prompt, bench_gen, cfg.force_cpu, cfg.bopts.enable_dspark);

    if (cfg.model_path.empty()) {
        fprintf(stderr, "error: --native <model.strata|hf_dir|model.gguf> is required\n");
        return 2;
    }

    InkModelFile mf;
    std::string err;
    if (!mf.open(cfg.model_path, err, false, cfg.bopts.dspark_path)) {
        if (mode_serve) proto("ERR open: %s", err.c_str());
        else fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }

    auto backend = make_default_backend(cfg.force_cpu, &cfg.bopts);
    if (!backend->init(mf, cfg.bopts, err)) {
        if (mode_serve) proto("ERR init: %s", err.c_str());
        else fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }

    if (mode_serve) {
        return run_serve_loop(cfg, mf, *backend);
    }

    // Direct CLI generation mode
    if (!backend->alloc_slot(0, cfg.bopts.max_ctx, cfg.bopts.kv_dtype, err)) {
        fprintf(stderr, "alloc_slot: %s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> ids;
    if (!prompt_ids_str.empty()) {
        if (!parse_id_list(prompt_ids_str, ids, err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
    } else {
        ids = {200002, 200004, 100, 200010, 200000, 200004, 101, 200010, 200001, 200004};
    }
    const int V = mf.m.c.vocab_unpadded;
    std::vector<float> logits(V);
    StepBatch batch;
    batch.capture_taps = cfg.bopts.enable_dspark;
    for (size_t i = 0; i < ids.size(); ++i) batch.push(0, static_cast<int>(i), ids[i], i + 1 == ids.size());
    if (!backend->forward_batch(batch, logits.data(), nullptr, err)) {
        fprintf(stderr, "prefill: %s\n", err.c_str());
        return 1;
    }
    SampleOpts so;
    so.temperature = cli_temp;
    std::mt19937_64 rng(42);
    std::vector<int32_t> hist = ids;
    int pos = static_cast<int>(ids.size());
    for (int g = 0; g < bench_gen; ++g) {
        int32_t tok = sample_logits(logits.data(), V, so, hist, rng);
        printf("%d%s", tok, (g + 1 == bench_gen) ? "\n" : ",");
        fflush(stdout);
        hist.push_back(tok);
        if (tok == mf.m.c.eos_token_id || tok == mf.m.c.eot_token_id) break;
        batch.clear();
        batch.push(0, pos++, tok, true);
        if (!backend->forward_batch(batch, logits.data(), nullptr, err)) {
            fprintf(stderr, "decode: %s\n", err.c_str());
            return 1;
        }
    }
    return 0;
}
