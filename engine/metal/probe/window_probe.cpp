// metal/probe/window_probe.cpp - the Metal decode window on its own: Window::init and a few windows of one token, without the engine around it.
//
//   window_probe <model-metal.gguf> [tokens N] [--ple <model.gguf>]        (built with the same -DSTRATA_GEOMETRY_FILE as the model)
//   --requests F --logits-out G   the gate-2 requests without the engine: every line of F is a token list (comma separated); each is read as a prompt
//             (the prompt chain from 32 tokens, windows of 8 below, as the engine reads it), its last token runs as a head window, and the logits of
//             that position (the vocabulary's floats) are appended to G.  The judge (g3kl_mac.py --from-dump G) compares them with llama.cpp.
//   --session N   carve the engine's session state for N cells of context (session_init, as generate.cpp does) before the window comes up
//   --ple F  open F's PLE table as the engine does (direct reads, the I/O thread) and run the PLE block: the engine's first window
//
// A session with no PLE and no QSA state is enough for the window (the PLE block is skipped, the K/V live in the window's own buffers), so this is
// the engine's process state - the model mapped and wrapped, every library compiled, every buffer allocated, the first pipelines created - for
// a Metal fault that has to be looked at without an engine start.  Not a correctness test: the tokens are whatever the model makes of them.
#include "strata/core/session.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/ngram.hpp"
#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::metal_engine {
void set_model_shard(const char* shard1);
void* window_create();
bool window_init(void* w, const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int max_t, int64_t max_cells, int64_t* n_vocab, std::string& err);
bool window_run(void* w, int T, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, std::string& err);
bool window_commit(void* w, int n_keep, std::string& err);
bool window_copy_logits(void* w, int t, float* host);
void window_reset(const strata::core::SessionState* ss);
bool prompt_run(const int64_t* tokens, int64_t n, int64_t pos0, const float* const* embd_rows,
                const std::function<bool(const float*, int64_t, int64_t, std::string&)>& on_chunk, std::string& err);
}  // namespace strata::metal_engine

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: window_probe <model-metal.gguf> [tokens N]\n"); return 2; }
    int n = argc > 2 && argv[2][0] >= '0' && argv[2][0] <= '9' ? std::atoi(argv[2]) : 4;
    setenv("STRATA_METAL_STREAM", "0", 1);
    strata::metal_engine::set_model_shard(argv[1]);
    strata::core::SessionState ss;
    std::string err;
    strata::kernels::PleTable table;
    static float dummy_f[16] = {};
    static uint16_t dummy_h[16] = {};
    static int32_t dummy_i[4] = {-1, -1, -1, -1};
    for (int i = 2; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--ple") {
            strata::kernels::PleIoOptions pio;
            if (!table.open(argv[i + 1], err, pio)) { std::printf("PLE table: %s\n", err.c_str()); return 5; }
            ss.ple.table = &table;   // (the window reads its weights from the model file; the rest of ready() is pointers it does not use)
            ss.ple.consts = strata::kernels::ple_artifact_consts();
            ss.ple.hist = dummy_f; ss.ple.emb_dev = dummy_f; ss.ple.emb_host = dummy_f; ss.ple.scratch = dummy_f;
            ss.ple.token = dummy_i; ss.ple.prev = dummy_i;
            ss.ple.w.key_bf16 = dummy_h; ss.ple.w.value_bf16 = dummy_h; ss.ple.w.norm_key = dummy_f; ss.ple.w.norm_query = dummy_f;
            ss.ple.w.norm_conv = dummy_f; ss.ple.w.conv1d_f16 = dummy_h;
            std::printf("window_probe: PLE table %s open, %llu rows\n", argv[i + 1], (unsigned long long) table.rows());
        }
    for (int i = 2; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--session") {
            const int64_t cells = std::atoll(argv[i + 1]);
            const int64_t K = strata::core::kCompiledGeometry.n_expert_used;
            void* sbuf = nullptr;
            const uint64_t bytes = strata::core::session_bytes(strata::core::kCompiledGeometry, cells, K, 0, -1);
            if (cudaMalloc(&sbuf, bytes) != cudaSuccess) { std::printf("session allocation of %.1f GB failed\n", bytes / 1e9); return 9; }
            if (strata::core::session_init(strata::core::kCompiledGeometry, cells, K, sbuf, ss, 0, -1) == 0) { std::printf("session_init failed\n"); return 9; }
            std::printf("window_probe: session state for %lld cells carved (%.2f GB)\n", (long long) cells, bytes / 1e9);
        }
    int64_t nv = 0;
    void* w = strata::metal_engine::window_create();
    const auto t0 = std::chrono::steady_clock::now();
    if (!strata::metal_engine::window_init(w, strata::core::kCompiledGeometry, ss, 8, 4096, &nv, err)) { std::printf("init failed: %s\n", err.c_str()); return 3; }
    std::printf("window_probe: init ok, vocabulary %lld, %.0f ms\n", (long long) nv, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    std::vector<int32_t> toks = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t out[8] = {};
    for (int i = 0; i < n; ++i) {
        const auto t1 = std::chrono::steady_clock::now();
        if (!strata::metal_engine::window_run(w, 1, &toks[(size_t) i % 8], i, true, out, err) || !strata::metal_engine::window_commit(w, 1, err)) {
            std::printf("window %d failed: %s\n", i, err.c_str());
            return 4;
        }
        std::printf("window %d: token %d, %.1f ms\n", i, out[0], std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
    }
    std::string reqf, outf;
    for (int i = 2; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--requests") reqf = argv[i + 1];
        if (std::string(argv[i]) == "--logits-out") outf = argv[i + 1];
    }
    if (!reqf.empty() && !outf.empty()) {
        std::ifstream in(reqf);
        std::ofstream outl(outf, std::ios::binary | std::ios::trunc);
        std::vector<float> lg((size_t) nv);
        std::string line;
        int nreq = 0;
        const auto t2 = std::chrono::steady_clock::now();
        while (std::getline(in, line)) {
            std::vector<int64_t> ids;
            std::stringstream ss2(line);
            std::string tok;
            while (std::getline(ss2, tok, ',')) if (!tok.empty()) ids.push_back(std::atoll(tok.c_str()));
            if (ids.size() < 2) continue;
            strata::metal_engine::window_reset(&ss);
            ss.ple_prev[0] = ss.ple_prev[1] = -1;
            const int64_t npre = (int64_t) ids.size() - 1;
            if (!strata::metal_engine::prompt_run(ids.data(), npre, 0, nullptr, nullptr, err)) { std::printf("request %d: prompt failed: %s\n", nreq, err.c_str()); return 6; }
            int32_t last = (int32_t) ids.back(), o = 0;
            if (!strata::metal_engine::window_run(w, 1, &last, npre, true, &o, err)) { std::printf("request %d: window failed: %s\n", nreq, err.c_str()); return 7; }
            if (!strata::metal_engine::window_copy_logits(w, 0, lg.data())) { std::printf("request %d: no logits\n", nreq); return 8; }
            outl.write((const char*) lg.data(), (std::streamsize) lg.size() * 4);
            ++nreq;
        }
        std::printf("window_probe: %d requests, logits in %s, %.1f s\n", nreq, outf.c_str(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count());
        return 0;
    }
    std::printf("window_probe: ok\n");
    return 0;
}
