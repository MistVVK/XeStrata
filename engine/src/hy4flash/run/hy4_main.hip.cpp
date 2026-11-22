// hy4_main.hip.cpp - strata-hy4: Hy4-Flash runner for gfx1151 (Windows HIP)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "hy4_graph.hip.h"
#include "hy4_spec.hip.h"
#include "hy4_serve.hip.h"
#include "strata/core/volta_forever.hpp"
#if defined(__linux__) && !HY4_CUDA
#include "strata/platform/numa_balancing.hpp"
#endif

using namespace hy4;

#ifndef HY4_VERSION
#define HY4_VERSION "0.1.66"  // the engine numbering of the 0.2.20 release (engines: 0.1.66; app and Terminal: 0.2.20)
#endif

// `strata-hy4 --export-tokenizer FILE.strata DIR`: Section 0's tokenizer_json and chat_template to DIR/tokenizer.hf.json and DIR/chat_template.jinja (Python builds serve's tokenizer
// directory from them: terminal/tools/hy4_tokenizer.py convert_files), so Python needs no zstd.
static int export_tokenizer(const std::string& file, const std::string& dir) {
    strata::artifact::StrataFileReader rd;
    std::string err;
    if (!rd.open(file, err)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    Manifest m;
    parse_manifest(rd.manifest_json(), m, true);
    auto put = [&](const std::string& name, const std::string& text) {
        std::string path = dir + "/" + name;
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

// `strata-hy4 --info FILE.strata`: one line of key=value facts, no GPU needed
static int print_info(const std::string& file, const std::string& vision) {
    strata::artifact::StrataFileReader rd;
    std::string err;
    if (!rd.open(file, err)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    if (rd.header().profile == strata::artifact::StrataProfile::VisionShard) {  // HY4-VISION: the Vision shard itself
        rd.close();
        VisionShardInfo vi;
        if (!vision_probe(file, vi, err)) {
            fprintf(stderr, "vision shard: %s\n", err.c_str());
            return 1;
        }
        printf("arch=hy4flash-vision layers=%d hidden=%d out_hidden=%d patch=%d merge=%d tensors=%zu image_pad=%d\n", vi.c.layers, vi.c.hidden, vi.c.out_hidden, vi.c.patch, vi.c.merge,
               vi.T.size(), vi.c.image_token);
        return 0;
    }
    Manifest m;
    parse_manifest(rd.manifest_json(), m, false);
    // what this program CAN do (serve starts without what it cannot): drafter=none|dfly2 (the 5-layer DFly2 drafter in Section 1), images=0|1 (the Vision shard's SigLIP2 reader)
    const bool has_dfly = m.T.count("model.dfly_layers.0.self_attn.q_a_proj.weight") && m.T.count("model.dfly_layers.hidden_correction.gate_proj.weight");
    // images=1: this engine build reads pictures (image= keys on GEN, started with --vision Hy4-Flash-Vision.strata). With `--vision FILE` given the key says whether THAT shard is usable
    // (profile 3, the 441-tensor manifest parses): 0 when it is missing or damaged. Without --vision it is the program's capability (what serve asks before it knows the shard).
    int images = 1;
    if (!vision.empty()) {
        VisionShardInfo vi;
        std::string verr;
        images = vision_probe(vision, vi, verr) ? 1 : 0;
        if (!images) fprintf(stderr, "vision shard: %s\n", verr.c_str());
    }
    printf("arch=hy4flash layers=%d hidden=%d vocab=%d experts=%d eos=%d drafter=%s images=%d image_pad=%d image_max_pixels=%d\n", m.c.n_layers, m.c.hidden, m.c.vocab, m.c.n_experts, m.c.eos,
           has_dfly ? "dfly2" : "none", images, VitCfg().image_token, (int)VitCfg().max_pixels);
    return 0;
}

static void top5_line(const float* lg, int V, int pos, std::string& out) {
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    char buf[256];
    snprintf(buf, sizeof buf, "pos %d top5: %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f)\n", pos, idx[0], lg[idx[0]], idx[1], lg[idx[1]], idx[2], lg[idx[2]], idx[3], lg[idx[3]],
             idx[4], lg[idx[4]]);
    out += buf;
}

// ---- Fast GPU top-64 exact log-prob + tail_p extractor for --dump-dfly-batch ----
static inline float fp16_bits_host(uint16_t h) {
    uint32_t s = ((uint32_t)h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t f = 0;
    if (e == 0) {
        if (m != 0) {
            e = 113;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            f = s | (e << 23) | (m << 13);
        } else f = s;
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112) << 23) | (m << 13);
    }
    float out = 0.f;
    std::memcpy(&out, &f, 4);
    return out;
}

__global__ void __launch_bounds__(1024) k_dfly_logp_top64(const float* __restrict__ logits, int V, int32_t* __restrict__ top_idx, __half* __restrict__ top_logp, __half* __restrict__ tail_p) {
    __shared__ float sm[32];
    __shared__ float slg[64];
    __shared__ int32_t sid[64];
    const float* lg = logits + (size_t)blockIdx.x * V;
    const int tid = threadIdx.x;
    float mx = -3.0e38f;
    for (int i = tid; i < V; i += blockDim.x) mx = fmaxf(mx, lg[i]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o));
    if ((tid & 31) == 0) sm[tid >> 5] = mx;
    __syncthreads();
    if (tid == 0) {
        float b = sm[0];
        for (int w = 1; w < 32; ++w) b = fmaxf(b, sm[w]);
        sm[0] = b;
    }
    __syncthreads();
    const float row_mx = sm[0];
    float se = 0.0f;
    for (int i = tid; i < V; i += blockDim.x) se += __expf(lg[i] - row_mx);
    for (int o = 16; o > 0; o >>= 1) se += __shfl_xor(se, o);
    if ((tid & 31) == 0) sm[tid >> 5] = se;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < 32; ++w) s += sm[w];
        sm[0] = row_mx + logf(fmaxf(s, 1e-30f));
    }
    __syncthreads();
    const float logZ = sm[0];
    if (tid < 64) {
        const int32_t id = top_idx[(size_t)blockIdx.x * 64 + tid];
        sid[tid] = id;
        slg[tid] = lg[id];
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 1; i < 64; ++i) {
            float kv = slg[i];
            int32_t ki = sid[i];
            int j = i - 1;
            while (j >= 0 && (slg[j] < kv || (slg[j] == kv && sid[j] > ki))) {
                slg[j + 1] = slg[j];
                sid[j + 1] = sid[j];
                --j;
            }
            slg[j + 1] = kv;
            sid[j + 1] = ki;
        }
        float psum = 0.0f;
        for (int i = 0; i < 64; ++i) {
            const float lp = slg[i] - logZ;
            top_idx[(size_t)blockIdx.x * 64 + i] = sid[i];
            top_logp[(size_t)blockIdx.x * 64 + i] = __float2half(lp);
            psum += expf(lp);
        }
        const float tp = fmaxf(1e-7f, 1.0f - fminf(psum, 1.0f - 1e-7f));
        tail_p[blockIdx.x] = __float2half(tp);
    }
}

static int run_dump_dfly_batch(Runner& r, const std::string& manifest_path, const std::string& out_dir, int gen_n, float dump_temp, uint64_t seed) {
    std::ifstream mf(manifest_path);
    if (!mf) {
        fprintf(stderr, "dump-dfly-batch: cannot open %s\n", manifest_path.c_str());
        return 1;
    }
    std::filesystem::create_directories(out_dir);
    std::vector<std::vector<int32_t>> prompts;
    std::string line;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<int32_t> ids;
        const char* p = line.c_str();
        while (*p) {
            while (*p == ' ' || *p == '\t' || *p == ',' || *p == '\r') ++p;
            if (!*p) break;
            char* e = nullptr;
            long v = strtol(p, &e, 10);
            if (e == p) break;
            ids.push_back((int32_t)v);
            p = e;
        }
        if (ids.size() >= 8) prompts.push_back(std::move(ids));
    }
    fprintf(stderr, "[dump-dfly] loaded %zu prompts from %s, gen=%d, temp=%.2f -> %s\n", prompts.size(), manifest_path.c_str(), gen_n, dump_temp, out_dir.c_str());
    const int V = r.c.vocab;
    const int max_rows = r.Tc;
    int32_t* d_top_idx = r.dalloc<int32_t>((size_t)max_rows * 64);
    __half* d_top_lp = r.dalloc<__half>((size_t)max_rows * 64);
    __half* d_tail_p = r.dalloc<__half>((size_t)max_rows);
    std::mt19937_64 rng(seed ? seed : 20261120ULL);
    std::string err;
    long total_toks = 0;
    const double t_all = now_s();
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        std::vector<int32_t> tok = prompts[pi];
        if ((int)tok.size() > r.Tc) tok.resize((size_t)r.Tc);
        const int T0 = (int)tok.size();
        const int max_T = std::min(r.max_ctx - 8, T0 + gen_n);
        if (max_T <= T0) continue;
        tok.reserve((size_t)max_T + 8);
        std::vector<uint16_t> hfin_all;
        std::vector<int32_t> tidx_all;
        std::vector<uint16_t> tlp_all;
        std::vector<uint16_t> tp_all;
        hfin_all.reserve((size_t)max_T * HID);
        tidx_all.reserve((size_t)max_T * 64);
        tlp_all.reserve((size_t)max_T * 64);
        tp_all.reserve((size_t)max_T);
        const double t0 = now_s();
        if (!r.forward_chunk(tok.data(), 0, T0, err)) {
            fprintf(stderr, "[dump-dfly] prefill err seq %zu: %s\n", pi, err.c_str());
            return 1;
        }
        r.head(T0, 0);
        hipLaunchKernelGGL(k_dfly_topn, dim3(T0), dim3(1024), 0, r.st, (const float*)r.logits, V, 64, d_top_idx);
        hipLaunchKernelGGL(k_dfly_logp_top64, dim3(T0), dim3(1024), 0, r.st, (const float*)r.logits, V, d_top_idx, d_top_lp, d_tail_p);
        hfin_all.resize((size_t)T0 * HID);
        tidx_all.resize((size_t)T0 * 64);
        tlp_all.resize((size_t)T0 * 64);
        tp_all.resize((size_t)T0);
        HCK(hipMemcpyAsync(hfin_all.data(), r.hfin, (size_t)T0 * HID * 2, hipMemcpyDeviceToHost, r.st));
        HCK(hipMemcpyAsync(tidx_all.data(), d_top_idx, (size_t)T0 * 64 * 4, hipMemcpyDeviceToHost, r.st));
        HCK(hipMemcpyAsync(tlp_all.data(), d_top_lp, (size_t)T0 * 64 * 2, hipMemcpyDeviceToHost, r.st));
        HCK(hipMemcpyAsync(tp_all.data(), d_tail_p, (size_t)T0 * 2, hipMemcpyDeviceToHost, r.st));
        HCK(hipStreamSynchronize(r.st));
        const float seq_temp = dump_temp < 0.f ? ((pi % 3 == 2) ? -dump_temp : 0.f) : dump_temp;
        auto pick_next = [&](int row_idx) -> int32_t {
            const int32_t* ids = tidx_all.data() + (size_t)row_idx * 64;
            if (seq_temp <= 0.f) return ids[0];
            const uint16_t* lps = tlp_all.data() + (size_t)row_idx * 64;
            double w[64], z = 0.0;
            const double lp0 = (double)fp16_bits_host(lps[0]);
            for (int k = 0; k < 64; ++k) {
                const double lp = (double)fp16_bits_host(lps[k]);
                w[k] = std::exp((lp - lp0) / (double)seq_temp);
                z += w[k];
            }
            double c = 0.0;
            int keep = 0;
            while (keep < 64) {
                c += w[keep++] / z;
                if (c >= 0.95) break;
            }
            double z2 = 0.0;
            for (int k = 0; k < keep; ++k) z2 += w[k];
            std::uniform_real_distribution<double> u(0.0, z2);
            double rv = u(rng), acc = 0.0;
            for (int k = 0; k < keep; ++k) {
                acc += w[k];
                if (rv <= acc) return ids[k];
            }
            return ids[keep - 1];
        };
        int cur_row = T0 - 1;
        while ((int)tok.size() < max_T) {
            const int32_t nx = pick_next(cur_row);
            tok.push_back(nx);
            if (nx == r.c.eos || nx == 120008) break;
            const int pos = (int)tok.size() - 1;
            if (!r.forward_chunk(tok.data(), pos, 1, err)) {
                fprintf(stderr, "[dump-dfly] decode err seq %zu pos %d: %s\n", pi, pos, err.c_str());
                return 1;
            }
            r.head(1, 0);
            hipLaunchKernelGGL(k_dfly_topn, dim3(1), dim3(1024), 0, r.st, (const float*)r.logits, V, 64, d_top_idx);
            hipLaunchKernelGGL(k_dfly_logp_top64, dim3(1), dim3(1024), 0, r.st, (const float*)r.logits, V, d_top_idx, d_top_lp, d_tail_p);
            const size_t ridx = (size_t)pos;
            hfin_all.resize((ridx + 1) * HID);
            tidx_all.resize((ridx + 1) * 64);
            tlp_all.resize((ridx + 1) * 64);
            tp_all.resize(ridx + 1);
            HCK(hipMemcpyAsync(hfin_all.data() + ridx * HID, r.hfin, (size_t)HID * 2, hipMemcpyDeviceToHost, r.st));
            HCK(hipMemcpyAsync(tidx_all.data() + ridx * 64, d_top_idx, 64 * 4, hipMemcpyDeviceToHost, r.st));
            HCK(hipMemcpyAsync(tlp_all.data() + ridx * 64, d_top_lp, 64 * 2, hipMemcpyDeviceToHost, r.st));
            HCK(hipMemcpyAsync(tp_all.data() + ridx, d_tail_p, 2, hipMemcpyDeviceToHost, r.st));
            HCK(hipStreamSynchronize(r.st));
            cur_row = pos;
        }
        const int T = (int)tp_all.size();
        tok.resize((size_t)T);
        std::vector<uint16_t> kv_all((size_t)5 * T * MLA_ROW);
        for (int d = 0; d < 5; ++d) {
            const int L = DFLY_LAYERS[d];
            HCK(hipMemcpy(kv_all.data() + (size_t)d * T * MLA_ROW, r.kcache[L], (size_t)T * MLA_ROW * 2, hipMemcpyDeviceToHost));
        }
        char fn[512];
        snprintf(fn, sizeof fn, "%s/seq_%04zu.bin", out_dir.c_str(), pi);
        FILE* fo = fopen(fn, "wb");
        if (!fo) {
            fprintf(stderr, "[dump-dfly] cannot write %s\n", fn);
            return 1;
        }
        int32_t hdr[8] = {0x44464C59, 1, T, T0, 64, 5, HID, MLA_ROW};
        fwrite(hdr, 4, 8, fo);
        fwrite(tok.data(), 4, (size_t)T, fo);
        fwrite(hfin_all.data(), 2, (size_t)T * HID, fo);
        fwrite(kv_all.data(), 2, (size_t)5 * T * MLA_ROW, fo);
        fwrite(tidx_all.data(), 4, (size_t)T * 64, fo);
        fwrite(tlp_all.data(), 2, (size_t)T * 64, fo);
        fwrite(tp_all.data(), 2, (size_t)T, fo);
        fclose(fo);
        total_toks += T;
        const double dt = now_s() - t0;
        fprintf(stderr, "[dump-dfly] %zu/%zu -> %s (T=%d, prompt=%d, gen=%d, temp=%.2f, %.1f tok/s, cum=%ld tok)\n",
                pi + 1, prompts.size(), fn, T, T0, T - T0, seq_temp, (T - T0) / std::max(dt, 1e-6), total_toks);
    }
    fprintf(stderr, "[dump-dfly] DONE: %zu sequences, %ld total tokens in %.1f s\n", prompts.size(), total_toks, now_s() - t_all);
    return 0;
}

int main(int argc, char** argv) {
#if defined(__linux__) && !HY4_CUDA
    // Must run before any worker thread is created: hipHostRegister on Linux HIP creates KFD userptr mappings that
    // kernel.numa_balancing=1 otherwise invalidates, stalling AMD GPU queues for 77-236 s (numa_balancing.hpp).
    if (const std::string nb = strata::platform::numa_balancing::keep_scanner_off(); !nb.empty()) {
        std::fprintf(stderr, "%s\n", nb.c_str());
    }
#endif
#if HY4_CUDA
    if (strata::core::apply_default_cuda_module_loading()) {
        std::fprintf(stderr, "[gpu] volta-forever detected: setting CUDA_MODULE_LOADING=LAZY\n");
    }
#endif
    std::string model, engram, tok_path, out_dir = ".", vision_path, dfly_override;
    std::vector<std::string> image_files;
    std::string info_file;
    int64_t image_max_pixels = 4194304;
    bool load_only = false, dump = false, save_logits = false, all_logits = false, no_engram = false, no_index = false, prof = false, prof2 = false;
    int gen = 0, max_ctx = 0, chunk = 0 /* 0: auto from the card */, index_topk = 2048;
    bool serve = false, session_reuse = false;
#ifdef STRATA_HY4_DISTRIBUTED
    bool mpi_experts = false;
#endif
    std::string cc_dir;
    double cc_gib = 40.0;
    int cc_min = 1024;
    int gpu = -1;
    std::string gpus_csv = getenv("HY4_GPUS") ? getenv("HY4_GPUS") : "";
    std::string layer_split = getenv("HY4_LAYER_SPLIT") ? getenv("HY4_LAYER_SPLIT") : "";
    std::string draft = "auto";  // auto: dfly2 under --serve (when the file carries the drafter), off for the CLI runs (gates / benches); "off" and "dfly2" force
    SpecCfg scfg;
    SampleOpts cli_samp;
    int tail_skip_arg = 0;  // opt-in only (0 = OFF by default)
    double vram_experts_gb = -1.0, ram_experts_gb = -1.0;
    std::string dump_dfly_batch, dump_dfly_out;
    int dump_dfly_gen = 384;
    float dump_dfly_temp = -0.6f;  // <0: mixed 2/3 greedy + 1/3 sampled at |temp|; 0: pure greedy; >0: sampled
    std::string parity_list;
    bool det_flag = true, want_margins = false;
    int ab_rounds = 0, trace_m = 0, trace_off = 0, trace_mut = -1, dump_layer_arg = -1;
    bool ab_sab = false, alloc_only = false, knobs16 = false, dfly_warm = false;
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
        if (a == "--info") {  // answered after the whole command line is read, so `--info MODEL --vision SHARD` can check the shard
            info_file = nxt();
            continue;
        }
        if (a == "--export-tokenizer") {
            std::string f = nxt();
            return export_tokenizer(f, nxt());
        }
        if (a == "--serve") serve = true;
#ifdef STRATA_HY4_DISTRIBUTED
        else if (a == "--mpi-experts") mpi_experts = true;
#endif
        else if (a == "--model" || a == "--native") model = nxt();
        else if (a == "--max-context") max_ctx = atoi(nxt().c_str());
        else if (a == "--gpu") gpu = atoi(nxt().c_str());
        else if (a == "--gpus" || a == "--devices") gpus_csv = nxt();
        else if (a == "--layer-split") layer_split = nxt();
        else if (a == "--draft") draft = nxt();
        else if (a == "--dfly-override") dfly_override = nxt();
        else if (a == "--spec-k") scfg.K = atoi(nxt().c_str());
        else if (a == "--spec-fixed") scfg.adaptive = false;
        else if (a == "--spec-cut") scfg.cut = (float)atof(nxt().c_str());
        else if (a == "--spec-debug") scfg.debug = true;
        else if (a == "--spec-rv") scfg.rv = atoi(nxt().c_str());
        else if (a == "--spec-tree") scfg.tree = atoi(nxt().c_str()) != 0;
        else if (a == "--spec-early") scfg.early_cut = (float)atof(nxt().c_str());
        else if (a == "--tail-skip") tail_skip_arg = atoi(nxt().c_str());
        else if (a == "--temperature") cli_samp.temperature = (float)atof(nxt().c_str());
        else if (a == "--top-p") cli_samp.top_p = (float)atof(nxt().c_str());
        else if (a == "--top-k") cli_samp.top_k = atoi(nxt().c_str());
        else if (a == "--min-p") cli_samp.min_p = (float)atof(nxt().c_str());
        else if (a == "--seed") cli_samp.seed = (uint64_t)strtoull(nxt().c_str(), nullptr, 10);
        else if (a == "--dump-dfly-batch") dump_dfly_batch = nxt();
        else if (a == "--dump-dfly-out") dump_dfly_out = nxt();
        else if (a == "--dump-dfly-gen") dump_dfly_gen = atoi(nxt().c_str());
        else if (a == "--dump-dfly-temp") dump_dfly_temp = (float)atof(nxt().c_str());
        else if (a == "--spec-sabotage") scfg.sabotage = true;
        else if (a == "--spec-inv") scfg.inv = atoi(nxt().c_str()) != 0;
        else if (a == "--spec-check") scfg.check = true;
        else if (a == "--spec-nodraft") { scfg.sabotage = true; scfg.nodraft = true; }
        else if (a == "--det") det_flag = atoi(nxt().c_str()) != 0;
        else if (a == "--margins") want_margins = true;
        else if (a == "--ab") ab_rounds = atoi(nxt().c_str());
        else if (a == "--ab-sab") ab_sab = true;
        else if (a == "--dfly-alloc-only") alloc_only = true;
        else if (a == "--knobs16") knobs16 = true;
        else if (a == "--dfly-warm") dfly_warm = true;
        else if (a == "--trace-m") trace_m = atoi(nxt().c_str());
        else if (a == "--trace-off") trace_off = atoi(nxt().c_str());
        else if (a == "--trace-mut") trace_mut = atoi(nxt().c_str());
        else if (a == "--dump-layer") dump_layer_arg = atoi(nxt().c_str());
        else if (a == "--mla-blocks") g_mla_target_blocks = atoi(nxt().c_str());
        else if (a == "--mla-inv") g_mla_batch_invariant = atoi(nxt().c_str());
        else if (a == "--spec-exact") scfg.exact = atoi(nxt().c_str()) != 0;
        else if (a == "--parity-m") parity_list = nxt();
        else if (a == "--session-reuse") session_reuse = atoi(nxt().c_str()) != 0;
        else if (a == "--conversation-cache-dir") cc_dir = nxt();
        else if (a == "--conversation-cache-gib") cc_gib = atof(nxt().c_str());
        else if (a == "--conversation-cache-min-tokens") cc_min = atoi(nxt().c_str());
        else if (a == "--vision") vision_path = nxt();           // HY4-VISION: the Vision shard; loaded on the first request that carries a picture
        else if (a == "--image") image_files.push_back(nxt());  // HY4-VISION (offline runs): a picture file per image-pad token of --tokens, in order (the engine expands each pad)
        else if (a == "--image-max-pixels") image_max_pixels = atoll(nxt().c_str());
        else if (a == "--vram-experts-gb") {
            const std::string v = nxt();
            vram_experts_gb = atof(v.c_str());
        }
        else if (a == "--ram-experts-gb") {
            const std::string v = nxt();
            ram_experts_gb = atof(v.c_str());
            if (ram_experts_gb >= 0.0) hy4_setenv("STRATA_HY4_RAM_GB", v.c_str(), 1);
        }
        else if (a == "--ssd-experts") {
            // First-class GPU+SSD Only mode: 0 GiB full host arena, bounded O_DIRECT/FILE_FLAG_NO_BUFFERING ColdStore!
            ram_experts_gb = 0.0;
            hy4_setenv("HY4_RAM_BUDGET_GB", "0", 1);
            hy4_setenv("STRATA_HY4_RAM_GB", "0", 1);
            hy4_setenv("STRATA_CPUEXP_COLD_IO", "direct", 1);
        }
        else if (a == "--engram-cache-mib") {
            const std::string v = nxt();
            if (atof(v.c_str()) >= 0.0) hy4_setenv("HY4_ENGRAM_CACHE_MIB", v.c_str(), 1);
        }
        else if (a == "--kv") {
            std::string kv = nxt();
            if (kv == "fp16" || kv == "f16") {
                hy4_setenv("HY4_KV_Q8", "0", 1);
            } else if (kv == "q8" || kv == "int8" || kv == "i8" || kv == "k8v8") {
                hy4_setenv("HY4_KV_Q8", "1", 1);
            } else if (kv != "auto") {
                printf("ERR --kv %s: expected fp16, q8 or auto\n", kv.c_str());
                fflush(stdout);
                return 2;
            }
        }
        else if (a == "--engram") engram = nxt();
        else if (a == "--tokens") tok_path = nxt();
        else if (a == "--out") out_dir = nxt();
        else if (a == "--gen") gen = atoi(nxt().c_str());
        else if (a == "--ctx") max_ctx = atoi(nxt().c_str());
        else if (a == "--chunk") chunk = atoi(nxt().c_str());
        else if (a == "--load-only") load_only = true;
        else if (a == "--dump") dump = true;
        else if (a == "--all-logits") all_logits = true;
        else if (a == "--save-logits") save_logits = true;
        else if (a == "--no-engram") no_engram = true;
        else if (a == "--index-topk") index_topk = atoi(nxt().c_str());
        else if (a == "--no-index") no_index = true;
        else if (a == "--prof") prof = true;
        else if (a == "--prof2") prof2 = true;
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            if (serve) {
                printf("ERR unknown flag %s\n", a.c_str());
                fflush(stdout);
            }
            return 2;
        }
    }
    if (!info_file.empty()) return print_info(info_file, vision_path);
#ifdef STRATA_HY4_DISTRIBUTED
    if (mpi_experts && (!serve || !gpus_csv.empty() || gpu > 0)) {
        fprintf(stderr, "--mpi-experts requires --serve and a single visible CUDA device per host\n");
        return 2;
    }
#endif
    std::vector<int> peer_devs;
    int primary_gpu = gpu >= 0 ? gpu : 0;
    if (!gpus_csv.empty()) {
        std::vector<int> ids;
        if (gpus_csv.find(',') == std::string::npos) {
            int val = atoi(gpus_csv.c_str());
            int ndev = 0;
            if (val >= 2 && hipGetDeviceCount(&ndev) == hipSuccess && ndev >= 2) {
                int use_n = std::min(val, ndev);
                for (int k = 0; k < use_n; ++k) ids.push_back(k);
            } else {
                ids.push_back(val);
            }
        } else {
            for (size_t pos = 0; pos < gpus_csv.size();) {
                ids.push_back(atoi(gpus_csv.c_str() + pos));
                size_t q = gpus_csv.find(',', pos);
                if (q == std::string::npos) break;
                pos = q + 1;
            }
        }
        if (!ids.empty()) {
            primary_gpu = ids[0];
            for (size_t k = 1; k < ids.size(); ++k)
                if (ids[k] != primary_gpu && std::find(peer_devs.begin(), peer_devs.end(), ids[k]) == peer_devs.end())
                    peer_devs.push_back(ids[k]);
            gpu = primary_gpu;
        }
    }
    if (gpu >= 0) HCK(hipSetDevice(gpu));
    if (serve) {
        if (model.empty()) {
            printf("ERR --native FILE.strata is required\n");
            fflush(stdout);
            return 2;
        }
        if (draft != "off" && draft != "dfly2" && draft != "auto") {
            printf("ERR --draft %s: expected off, dfly2 or auto\n", draft.c_str());
            fflush(stdout);
            return 2;
        }
        ServeOpts so;
        so.spec = scfg;
        so.model = model;
        so.engram = engram;
        so.max_ctx = max_ctx > 0 ? max_ctx : 32768;
        so.chunk = chunk;  // 0 = auto: plan_memory picks it from the card (max(chunk, 1) made the serve chunk 1 after the CUDA lane's chunk = 0 default)
        so.index_topk = index_topk;
        so.no_index = no_index;
        so.no_engram = no_engram;
        so.session_reuse = session_reuse;
        so.cc_dir = cc_dir;
        so.cc_gib = cc_gib;
        so.cc_min = cc_min;
        so.tail_skip = tail_skip_arg;
        so.draft = draft;
        so.dfly_override = dfly_override;
        so.vision = vision_path;  // HY4-VISION
        so.image_max_pixels = image_max_pixels;
        so.primary_gpu = primary_gpu;
        so.peer_devs = peer_devs;
        so.vram_experts_gb = vram_experts_gb;
        so.ram_experts_gb = ram_experts_gb;
        so.layer_split = layer_split;
#ifdef STRATA_HY4_DISTRIBUTED
        if (mpi_experts) {
            try {
                strata::tp::MpiRuntime mpi(argc, argv, std::chrono::minutes(10));
                if (mpi.size() != 2 || mpi.minimum(mpi.local_rank() == 0 ? 1 : 0) != 1)
                    mpi.fail("Hy4 expert parallelism requires two hosts with one MPI rank each");
                try {
                    int devices = 0;
                    HCK(cudaGetDeviceCount(&devices));
                    if (devices != 1) mpi.fail("Hy4 each host must expose exactly one CUDA device");
                    HCK(cudaSetDevice(0));
                    mpi.agree(expert_model_identity(model));
                    {
                        RemoteExperts remote(mpi);
                        if (mpi.rank() == 0) {
                            so.remote_experts = &remote;
                            if (hy4_serve_main(so) != 0) mpi.fail("Hy4 coordinator failed to serve");
                            remote.finish();
                        } else {
                            remote.work(model);
                        }
                    }
                    mpi.finish();
                    return 0;
                } catch (const std::exception& failure) {
                    mpi.fail(failure.what());
                }
            } catch (const std::exception& failure) {
                fprintf(stderr, "Hy4 MPI initialization failed: %s\n", failure.what());
                return 1;
            }
        }
#endif
        return hy4_serve_main(so);
    }
    if (model.empty()) {
        fprintf(stderr, "usage: strata-hy4 --model SQ64.strata --engram Engram-Q8.strata --tokens ids.txt [--gen N] [--chunk N] [--ctx N] [--gpus 0,1] [--dump] [--all-logits] [--out dir]\n");
        return 2;
    }
    hipDeviceProp_t p;
    HCK(hipGetDeviceProperties(&p, primary_gpu));
    size_t fr = 0, tot = 0;
    gpu_mem_avail(fr, tot);
    fprintf(stderr, "[gpu] %s %s free %.1f / %.1f GiB%s\n", p.name, gpu_arch(p), fr / 1073741824.0, tot / 1073741824.0, p.integrated ? " (UMA)" : "");
    if (getenv("HY4_VRAM_CAP_GB")) {  // test knob: pretend the card is smaller (ballast allocation), e.g. 7.6 = an 8 GB card
        const double cap = atof(getenv("HY4_VRAM_CAP_GB")) * 1073741824.0;
        if (cap < (double)fr) {
            void* ballast = nullptr;
            HCK(hipMalloc(&ballast, fr - (size_t)cap));
            fr = (size_t)cap;
            tot = (size_t)cap;
            fprintf(stderr, "[gpu] VRAM capped to %.1f GiB (test)\n", cap / 1073741824.0);
        }
    }
    std::vector<int32_t> tok;
    const bool ctx_auto = max_ctx <= 0;
    if (!tok_path.empty()) {
        std::ifstream f(tok_path);
        long v;
        while (f >> v) tok.push_back((int32_t)v);
        if (max_ctx <= 0 && !tok.empty()) max_ctx = (int)tok.size() + gen + 16;
    }
    size_t min_peer_tot = 0;
    for (int pd : peer_devs) {
        HCK(hipSetDevice(pd));
        size_t pfr = 0, ptot = 0;
        HCK(hipMemGetInfo(&pfr, &ptot));
        min_peer_tot = (min_peer_tot == 0) ? ptot : std::min(min_peer_tot, ptot);
    }
    if (!peer_devs.empty()) HCK(hipSetDevice(primary_gpu));
    chunk = plan_memory(max_ctx > 0 ? max_ctx : 32768, chunk, tot, model, !peer_devs.empty(), !vision_path.empty(), (int)peer_devs.size(), min_peer_tot, fr, draft != "off");
    GpuModel g;
    g.primary_gpu = primary_gpu;
    g.planned_chunk = chunk;
    g.has_vision = !vision_path.empty();
    g.draft_enabled = (draft != "off");
    g.vram_experts_gb_cap = vram_experts_gb;
    g.ram_experts_gb_cap = ram_experts_gb;
    g.layer_split_spec = layer_split;
    g.peer_devs = peer_devs;
    double t_load0 = now_s();
    if (!load_gpu_model(g, model, true)) return 1;
    if (!dfly_override.empty()) {
        std::string derr;
        if (!load_dfly_override(g, dfly_override, derr)) { fprintf(stderr, "dfly-override: %s\n", derr.c_str()); return 1; }
    }
    fprintf(stderr, "[load] total %.1f s\n", now_s() - t_load0);
    if (load_only) return 0;
    // HY4-VISION: decode + resize the pictures and expand each image-pad token of the prompt to the picture's run of pads (CPU only; the ViT runs after the runner is up)
    std::vector<ImageRec> recs;
    if (!image_files.empty()) {
        std::string ierr;
        std::vector<int32_t> ex;
        for (const std::string& f : image_files) {
            ImageRec rc;
            if (!prepare_picture(f, VitCfg().min_pixels, image_max_pixels, rc, ierr)) {
                fprintf(stderr, "image: %s\n", ierr.c_str());
                return 1;
            }
            recs.push_back(std::move(rc));
        }
        if (vision_path.empty() || !expand_image_pads(tok, VitCfg().image_token, recs, ex, ierr)) {
            fprintf(stderr, "image: %s\n", vision_path.empty() ? "--vision FILE is required with --image" : ierr.c_str());
            return 1;
        }
        tok.swap(ex);
        if (ctx_auto) max_ctx = (int)tok.size() + gen + 16;  // the pads grew the prompt (the card plan above used the unexpanded length)
    }
    int T0 = (int)tok.size();
    if (T0 == 0 && dump_dfly_batch.empty()) {
        fprintf(stderr, "no tokens\n");
        return 2;
    }
    if (max_ctx <= 0) max_ctx = dump_dfly_batch.empty() ? (T0 + gen + 16) : (dump_dfly_gen + 512);
    Runner r(g, max_ctx, std::max(chunk, 1));
    r.dump = dump;
    r.dump_dir = out_dir;
    r.index_topk = index_topk;
    r.index_on = no_index ? 0 : 1;
    r.prof = prof;
    r.prof2 = prof2;
    r.tail_skip = tail_skip_arg;
    r.det_mix = det_flag || draft == "dfly2";
    const bool spec_mode = draft == "dfly2" && gen > 0 && !alloc_only;
    std::string err;
    if (!r.init(no_engram ? std::string() : engram, err)) {
        fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    if (!dump_dfly_batch.empty()) {
        return run_dump_dfly_batch(r, dump_dfly_batch, dump_dfly_out.empty() ? out_dir : dump_dfly_out, dump_dfly_gen, dump_dfly_temp, cli_samp.seed);
    }
    tok.reserve(max_ctx + 8);
    DFly2 df(r);
    if (draft == "dfly2") {
        if (!df.init(err)) {
            fprintf(stderr, "dfly2: %s\n", err.c_str());
            return 1;
        }
    }
    // HY4-VISION: pictures of an offline run (--images): encode them now, the runner hooks do the rest
    VisionEncoder venc;
    VisionCtx vctx;
    if (!recs.empty()) {
        r.vcache_release_vram(1280ull << 20);  // reclaim ~1.25 GiB from tail vcache slabs for vision weights + encode scratch
        if (!venc.load(vision_path, err) || !vctx.setup(venc, r.st, tok, recs, r.Tc, err)) {
            fprintf(stderr, "images: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "[vision] %zu picture(s), %d image rows, encoded in %.2f s\n", recs.size(), vctx.n_rows, vctx.t_encode);
        venc.unload();
        r.vis = &vctx;
    }
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
    r.pf_eng_tok = tok.data();
    r.pf_eng_total = T0;
    r.prefetch_engram(tok.data(), 0, std::min(r.Tc, T0));
    std::vector<float> hl;
    int last_p0 = 0, last_T = 0;
    while (done < T0) {
        int T = std::min(r.Tc, T0 - done);
        if (!r.forward_chunk(tok.data(), done, T, err)) {
            fprintf(stderr, "forward: %s\n", err.c_str());
            return 1;
        }
        done += T;
        bool last = done == T0;
        if (last) {
            last_p0 = done - T;
            last_T = T;
            int first = (all_logits && !spec_mode) ? 0 : T - 1;
            r.head(T, first);
            if (df.ok) df.snap_h(T - first - 1);
            HCK(hipStreamSynchronize(r.st)); r.pflush();
            int n = T - first;
            hl.resize((size_t)n * V);
            HCK(hipMemcpy(hl.data(), r.logits, hl.size() * 4, hipMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) top5_line(&hl[(size_t)i * V], V, done - n + i, top_out);
            if (dump || save_logits) save_npy_f32(out_dir + "/logits.npy", hl.data(), {n, V});
        } else {
            HCK(hipStreamSynchronize(r.st)); r.pflush();
        }
        fprintf(stderr, "[prefill] %d/%d tokens, %.2f s, %.1f tok/s\n", done, T0, now_s() - t0, done / (now_s() - t0));
    }
    r.pf_eng_tok = nullptr;
    r.pf_eng_total = 0;
    double t_pf = now_s() - t0;
    r.prof_report("prefill");
    r.pacc_report("prefill", (T0 + r.Tc - 1) / r.Tc);
    printf("%s", top_out.c_str());
    fprintf(stderr, "[prefill] total %.2f s = %.1f ppts (engram cpu %.2f s)\n", t_pf, T0 / t_pf, r.t_engram_cpu);
    // greedy decode
    const double cpu_wait0 = r.t_cpu_wait;
    if (knobs16) { skd::lin_gemv_max_m() = 16; skd::moe_dec_max_m() = 16; }
    if (trace_m > 0) {
        // layer-by-layer bit comparison of an M-token window against the same positions one at a time: dumps x after every attention / MLP of every layer into out/seq<i> and out/win
        // (compare with tools: first file whose bytes differ names the layer and the op that is not batch invariant)
        const int M = trace_m;
        const int p0 = T0 - M - trace_off;
        if (trace_mut >= 0) tok[T0 - 1 - trace_off] = trace_mut;  // causality test: change the LAST token of the window
        skd::lin_gemv_max_m() = 16;
        skd::moe_dec_max_m() = 16;
        r.dump = true;
        r.dump_layer = dump_layer_arg;
        for (int i = 0; i < M; ++i) {
            const std::string d = out_dir + "/seq" + std::to_string(i);
            std::filesystem::create_directories(d);
            r.dump_dir = d;
            if (!r.forward_chunk(tok.data(), p0 + i, 1, err)) return 1;
        }
        {
            const std::string d = out_dir + "/win";
            std::filesystem::create_directories(d);
            r.dump_dir = d;
            if (!r.forward_chunk(tok.data(), p0, M, err)) return 1;
        }
        HCK(hipStreamSynchronize(r.st));
        printf("[trace] M=%d window p0=%d dumped\n", M, p0);
        return 0;
    }
    const hy4::CpuExpertStats cold0 = r.hyb ? r.cpu.stats() : hy4::CpuExpertStats();
    const EngramStats es0 = r.engram_ready ? r.eio.stats() : EngramStats();
    if (!parity_list.empty()) {
        // verify-window parity: rows of an M-token chunk vs the same positions run one token at a time (M = 1 decode), same prefix
        r.vcache_flush();
        r.vcache_frozen = true;
        std::vector<int> ws;
        for (size_t p = 0; p < parity_list.size();) {
            ws.push_back(atoi(parity_list.c_str() + p));
            size_t q = parity_list.find(',', p);
            if (q == std::string::npos) break;
            p = q + 1;
        }
        int Wmax = *std::max_element(ws.begin(), ws.end());
        if (Wmax > 16 || T0 < Wmax + 16 || r.Tc < Wmax) {
            fprintf(stderr, "parity: need W <= 16, chunk >= W and a longer prompt\n");
            return 2;
        }
        const int base = T0 - Wmax;
        std::vector<float> seq((size_t)Wmax * V), win((size_t)Wmax * V);
        skd::lin_gemv_max_m() = 4;
        skd::moe_dec_max_m() = 4;
        {   // run-to-run determinism of the same M=1 step: three repetitions of the last position, no state change in between (the row rewrites itself)
            std::vector<float> a(V), b(V);
            int nd = 0;
            for (int rep = 0; rep < 4; ++rep) {
                if (!r.forward_chunk(tok.data(), T0 - 1, 1, err)) return 1;
                r.head(1, 0);
                HCK(hipStreamSynchronize(r.st));
                HCK(hipMemcpy(rep == 0 ? a.data() : b.data(), r.logits, (size_t)V * 4, hipMemcpyDeviceToHost));
                if (rep > 0) nd += !std::equal(a.begin(), a.end(), b.begin());
            }
            printf("[determinism] M=1 last position repeated: %d of 3 repeats differ from the first\n", nd);
        }
        for (int i = 0; i < Wmax; ++i) {
            if (!r.forward_chunk(tok.data(), base + i, 1, err)) return 1;
            r.head(1, 0);
            HCK(hipStreamSynchronize(r.st));
            HCK(hipMemcpy(&seq[(size_t)i * V], r.logits, (size_t)V * 4, hipMemcpyDeviceToHost));
        }
        for (int mode = 0; mode < 2; ++mode) {
            skd::lin_gemv_max_m() = mode ? 16 : 4;
            skd::moe_dec_max_m() = mode ? 16 : 4;
            for (int W : ws) {
                const int p0 = T0 - W;
                for (int i = 0; base + i < p0; ++i) {  // restore the sequential prefix rows of the window range (earlier windows rewrote them)
                    if (!r.forward_chunk(tok.data(), base + i, 1, err)) return 1;
                }
                HCK(hipStreamSynchronize(r.st));
                if (!r.forward_chunk(tok.data(), p0, W, err)) return 1;
                r.head(W, 0);
                HCK(hipStreamSynchronize(r.st));
                HCK(hipMemcpy(win.data(), r.logits, (size_t)W * V * 4, hipMemcpyDeviceToHost));
                {   // timing of the window forward + head (5 reps)
                    r.pacc.clear();
                    const double tb0 = now_s();
                    for (int rep = 0; rep < 5; ++rep) {
                        r.forward_chunk(tok.data(), p0, W, err);
                        r.head(W, 0);
                    }
                    HCK(hipStreamSynchronize(r.st));
                    const double tb1 = now_s();
                    r.pflush();
                    fprintf(stderr, "[bench] M=%d %s: forward+head %.1f ms\n", W, mode ? "exact" : "default", (tb1 - tb0) * 200.0);
                    r.pacc_report("window", 5);
                }
                int top1 = 0, bit = 0;
                double maxabs = 0, kl_sum = 0, kl_max = 0;
                for (int i = 0; i < W; ++i) {
                    const float* s = &seq[(size_t)(p0 - base + i) * V];
                    const float* w = &win[(size_t)i * V];
                    int as = (int)(std::max_element(s, s + V) - s), aw = (int)(std::max_element(w, w + V) - w);
                    top1 += as == aw;
                    bit += std::equal(s, s + V, w);
                    double ms = *std::max_element(s, s + V), mw = *std::max_element(w, w + V), zs = 0, zw = 0;
                    for (int v = 0; v < V; ++v) {
                        zs += std::exp((double)s[v] - ms);
                        zw += std::exp((double)w[v] - mw);
                        maxabs = std::max(maxabs, (double)std::fabs(s[v] - w[v]));
                    }
                    double kl = 0;
                    for (int v = 0; v < V; ++v) {
                        double ps = std::exp((double)s[v] - ms) / zs;
                        if (ps > 1e-12) kl += ps * ((s[v] - ms - std::log(zs)) - (w[v] - mw - std::log(zw)));
                    }
                    kl_sum += kl;
                    kl_max = std::max(kl_max, kl);
                }
                printf("[parity] M=%d %s: top1 %d/%d, bit-identical rows %d/%d, max|dlogit| %.4g, mean KL %.3g, max KL %.3g\n", W, mode ? "gemv16+moe_dec16" : "default(gemv4+moe_dec4)", top1, W, bit, W,
                       maxabs, kl_sum / W, kl_max);
                fflush(stdout);
            }
        }
        return 0;
    }
    if (ab_rounds > 0 && spec_mode) {
        r.vcache_flush();
        r.vcache_frozen = true;  // both plain and dfly2 rounds share the identical prompt-adapted VRAM expert cache for exact 256/256 comparison
        // DFly2 off vs on from ONE prefill: each mode starts from the resident prompt state (the last prefill chunk is re-run, which rebuilds the logits, h_final and the key-list capture;
        // decode rows beyond the prompt are overwritten). Only the LAST 16 prompt positions are re-run: a longer re-run would need SWA ring slots of the early part of the last chunk,
        // which decode has overwritten (ring = 512 + chunk). Both modes start from this same re-run state. Rounds alternate plain / spec, same prompt, same flags.
        const int redo_T = std::min(16, T0), redo_p0 = T0 - redo_T;
        (void)last_p0;
        (void)last_T;
        if (scfg.exact) {
            skd::moe_dec_max_m() = 16;
            skd::lin_gemv_max_m() = 16;
        }
        r.det_mix = true;
        g_mla_batch_invariant = scfg.inv ? 1 : 0;
        auto restore = [&]() {
            tok.resize(T0);
            if (!r.forward_chunk(tok.data(), redo_p0, redo_T, err)) { fprintf(stderr, "ab restore: %s\n", err.c_str()); exit(1); }
            r.head(redo_T, redo_T - 1);
            df.snap_h(0);
            HCK(hipStreamSynchronize(r.st));
            HCK(hipMemcpy(hl.data(), r.logits, (size_t)V * 4, hipMemcpyDeviceToHost));
        };
        for (int round = 1; round <= ab_rounds; ++round) {
            // plain
            restore();
            std::vector<int> pids;
            double tp0 = now_s();
            {
                int next = (int)(std::max_element(hl.begin(), hl.begin() + V) - hl.begin());
                for (int s = 0; s < gen; ++s) {
                    tok.push_back(next);
                    pids.push_back(next);
                    if (next == r.c.eos || s + 1 == gen) break;
                    if (!r.forward_chunk(tok.data(), (int)tok.size() - 1, 1, err)) { fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
                    r.head(1, 0);
                    hipLaunchKernelGGL(k_argmax, dim3(1), dim3(1024), 0, r.st, r.logits, V, r.d_argmax);
                    int32_t nx = 0;
                    HCK(hipMemcpy(&nx, r.d_argmax, 4, hipMemcpyDeviceToHost));
                    next = nx;
                }
            }
            const double tp = now_s() - tp0;
            // spec
            restore();
            Spec sp(r, df, scfg);
            std::vector<int> sids;
            std::string finish, e2;
            int made = 0;
            const double ts0 = now_s();
            sp.run(tok, gen, max_ctx, r.c.eos, -1, [&](int t) { sids.push_back(t); return true; }, finish, e2, made);
            const double tsd = now_s() - ts0;
            if (finish == "error") { fprintf(stderr, "spec: %s\n", e2.c_str()); return 1; }
            size_t same = 0;
            while (same < pids.size() && same < sids.size() && pids[same] == sids[same]) ++same;
            const long steps = sp.S.spec_iters + sp.S.plain_iters;
            printf("[ab] ctx %d round %d: plain %zu tok %.2f s = %.2f tok/s | dfly2 %zu tok %.2f s = %.2f tok/s | speedup %.3f | tokens/iteration %.2f (accepted/verify-step %.2f, iterations %ld) | identical prefix %zu/%zu tokens\n",
                   T0, round, pids.size(), tp, pids.size() / tp, sids.size(), tsd, sids.size() / tsd, (sids.size() / tsd) / (pids.size() / tp), steps ? (double)sp.S.tokens / steps : 0.0,
                   sp.S.spec_iters ? (double)sp.S.accepted / sp.S.spec_iters : 0.0, steps, same, pids.size());
            fflush(stdout);
            sp.report(stderr, tsd * 1000.0);
            if (ab_sab) {
                // worst case: the same controller, every draft replaced by a random id (acceptance ~0)
                restore();
                SpecCfg sc2 = scfg;
                sc2.sabotage = true;
                Spec sb(r, df, sc2);
                std::vector<int> bids;
                std::string fin2, e3;
                int made2 = 0;
                const double tb0 = now_s();
                sb.run(tok, gen, max_ctx, r.c.eos, -1, [&](int t) { bids.push_back(t); return true; }, fin2, e3, made2);
                const double tbd = now_s() - tb0;
                printf("[ab-sab] ctx %d round %d: plain %.2f tok/s | dfly2 with random drafts (worst case) %zu tok %.2f s = %.2f tok/s | speedup %.3f | backoffs %ld, probes %ld, plain-mode iterations %ld, plain after draft %ld\n", T0, round,
                       pids.size() / tp, bids.size(), tbd, bids.size() / tbd, (bids.size() / tbd) / (pids.size() / tp), sb.S.backoffs, sb.S.probes, sb.S.plain_mode_iters, sb.S.plain_after_draft);
                fflush(stdout);
            }
        }
    } else if (spec_mode) {
        Spec sp(r, df, scfg);
        sp.want_margin = want_margins;
        std::string gen_ids, finish, e2;
        int made = 0;
        std::mt19937_64 cli_rng(cli_samp.seed ? cli_samp.seed : 20261120ULL);
        const SampleOpts* sptr = (!cli_samp.greedy() || cli_samp.penalties()) ? &cli_samp : nullptr;
        const double t1 = now_s();
        sp.run(tok, gen, max_ctx, r.c.eos, -1, [&](int t) { gen_ids += std::to_string(t) + " "; return true; }, finish, e2, made, sptr, sptr ? &cli_rng : nullptr);
        const double td = now_s() - t1;
        if (finish == "error") {
            fprintf(stderr, "spec: %s\n", e2.c_str());
            return 1;
        }
        printf("generated ids: %s\n", gen_ids.c_str());
        fprintf(stderr, "[decode] %d tokens in %.2f s = %.2f tok/s (spec dfly2, finish %s)\n", made, td, made / td, finish.c_str());
        sp.report(stderr, td * 1000.0);
        r.pacc_report("spec total (all steps)", 1);
        std::ofstream f(out_dir + "/gen_ids.txt");
        f << gen_ids << "\n";
        if (want_margins) {
            std::ofstream fm(out_dir + "/margins.txt");
            for (float m : sp.marg) fm << m << '\n';
            std::ofstream fh(out_dir + "/hashes.txt");
            for (uint64_t hv : sp.hsh) fh << std::hex << hv << '\n';
        }
    } else if (gen > 0) {
        int next = 0;
        {
            const float* lg = &hl[(size_t)(hl.size() / V - 1) * V];
            next = (int)(std::max_element(lg, lg + V) - lg);
        }
        if (dfly_warm && df.ok) {
            // test: run the drafter once (all stages) before the plain decode, to compare plain decode with / without the drafter's kernels having run
            int yy[9] = {};
            float cc[9] = {};
            df.rv = 256;
            if (!df.draft(T0, next, 7, yy, cc)) fprintf(stderr, "dfly warm-up draft refused\n");
        }
        std::string gen_ids;
        double t1 = now_s();
        std::vector<double> times, enq;
        std::vector<float> margs;
        std::vector<uint64_t> hashes;  // FNV-1a of the bits of every logits row the token choice came from
        auto gap2 = [&](const float* lg) {
            float b1 = -3e38f, b2 = -3e38f;
            uint64_t hh = 1469598103934665603ull;
            for (int i = 0; i < V; ++i) {
                if (lg[i] > b1) { b2 = b1; b1 = lg[i]; }
                else if (lg[i] > b2) b2 = lg[i];
                uint32_t bits;
                memcpy(&bits, &lg[i], 4);
                hh = (hh ^ bits) * 1099511628211ull;
            }
            hashes.push_back(hh);
            return b1 - b2;
        };
        if (want_margins) margs.push_back(gap2(&hl[(size_t)(hl.size() / V - 1) * V]));
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
            enq.push_back(now_s() - ts);
            HCK(hipStreamSynchronize(r.st)); r.pflush();
            float* d = r.logits;
            if (want_margins) {
                std::vector<float> hrow(V);
                HCK(hipMemcpy(hrow.data(), d, (size_t)V * 4, hipMemcpyDeviceToHost));
                margs.push_back(gap2(hrow.data()));
            }
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
        if (!enq.empty()) { std::sort(enq.begin(), enq.end()); fprintf(stderr, "[decode] median host enqueue time per step %.1f ms (step %.1f ms)\n", enq[enq.size() / 2] * 1000, times.empty() ? 0.0 : times[times.size() / 2] * 1000); }
        std::sort(times.begin(), times.end());
        if (r.n_eng_dec && r.engram_ready) { const EngramStats e1 = r.eio.stats(); const double n = r.n_eng_dec; fprintf(stderr, "[decode] engram io per step: %.1f pages, page-read wall %.3f ms, whole fetch %.3f ms, lookups %.0f (seed+lru hits %.0f)\n", (e1.page_reads - es0.page_reads) / n, (e1.io_seconds - es0.io_seconds) * 1000 / n, (e1.fetch_seconds - es0.fetch_seconds) * 1000 / n, (e1.lookups - es0.lookups) / n, (e1.seed_hits + e1.lru_hits - es0.seed_hits - es0.lru_hits) / n); }
        if (r.n_eng_dec) fprintf(stderr, "[decode] engram per step: prefetch issue %.3f ms, host wait+upload at layer 1 %.3f ms\n", r.t_eng_issue * 1000 / r.n_eng_dec, r.t_eng_prep * 1000 / r.n_eng_dec);
        if (r.hyb) fprintf(stderr, "[hybrid] decode: host waited %.2f ms per step for the CPU experts after queueing the GPU work\n", (r.t_cpu_wait - cpu_wait0) * 1000 / std::max<size_t>(times.size(), 1));
        if (r.vcache_nslots > 1 && (r.vc_hits + r.vc_misses) > 0) {
            const uint64_t tot_p = r.vc_hits + r.vc_misses;
            fprintf(stderr, "[vcache] decode: %.1f%% VRAM expert hit rate (%llu / %llu pairs), %llu promotions\n", 100.0 * (double)r.vc_hits / (double)tot_p,
                    (unsigned long long)r.vc_hits, (unsigned long long)tot_p, (unsigned long long)r.vc_promotes);
        }
        if (r.hyb && r.has_cold) {
            const hy4::CpuExpertStats c1 = r.cpu.stats();
            const double nt = (double)std::max<size_t>(times.size(), 1);
            fprintf(stderr, "[cold] decode: %.1f SSD reads per token (%.1f MB), %.1f cache hits per token, SSD %.2f GB/s while reading, pool waited %.2f ms per token for reads, %llu read errors\n", (c1.cold_reads - cold0.cold_reads) / nt,
                    (c1.cold_bytes - cold0.cold_bytes) / 1e6 / nt, (c1.cold_hits - cold0.cold_hits) / nt, (c1.cold_io_s - cold0.cold_io_s) > 0 ? (c1.cold_bytes - cold0.cold_bytes) / 1e9 / (c1.cold_io_s - cold0.cold_io_s) : 0.0,
                    (c1.cold_wait_s - cold0.cold_wait_s) * 1000 / nt, (unsigned long long)c1.cold_errors);
        }
        fprintf(stderr, "[decode] %zu tokens in %.2f s = %.2f tok/s (median step %.1f ms)\n", times.size(), td, times.size() / td, times.empty() ? 0.0 : times[times.size() / 2] * 1000);
        std::ofstream f(out_dir + "/gen_ids.txt");
        f << gen_ids << "\n";
        if (want_margins) {
            std::ofstream fm(out_dir + "/margins.txt");
            for (float m : margs) fm << m << '\n';
            std::ofstream fh(out_dir + "/hashes.txt");
            for (uint64_t hv : hashes) fh << std::hex << hv << '\n';
        }
    }
    {
        std::ofstream f(out_dir + "/top5.txt");
        f << top_out;
    }
    if (r.vcache_nslots > 1 && (r.vc_hits + r.vc_misses) > 0) {
        const uint64_t tot_p = r.vc_hits + r.vc_misses;
        fprintf(stderr, "[vcache] session: %.1f%% VRAM expert hit rate (%llu / %llu pairs), %llu promotions\n", 100.0 * (double)r.vc_hits / (double)tot_p,
                (unsigned long long)r.vc_hits, (unsigned long long)tot_p, (unsigned long long)r.vc_promotes);
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
