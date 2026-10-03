// The model-level reference: llama.cpp at the pinned commit 3cf03257 on the CPU evaluates a prompt given as token
// ids, then continues greedily.  It writes, per evaluated position (the last prompt position and each generated
// token's), one line of all logits, the same format as `strata generate --dump-logits`: space-separated floats.
//
//     ref_logits MODEL_SHARD1.gguf "760,6511,..." N_NEW OUT_LOGITS.txt
//
// Built against libllama from a checkout of that commit (see README.md).  No sampling, no chat template: the
// engine and this program see the same ids.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: ref_logits MODEL \"ids\" N_NEW OUT\n"); return 2; }
    std::vector<llama_token> ids;
    for (const char* p = argv[2]; *p;) {
        char* end = nullptr;
        ids.push_back((llama_token) std::strtol(p, &end, 10));
        p = *end ? end + 1 : end;
    }
    const int n_new = std::atoi(argv[3]);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_threads = cp.n_threads_batch = (int) std::thread::hardware_concurrency() / 2;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create a context\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::FILE* out = std::fopen(argv[4], "w");
    auto dump = [&](const float* l) {
        for (int i = 0; i < n_vocab; ++i) std::fprintf(out, i ? " %.6g" : "%.6g", l[i]);
        std::fprintf(out, "\n");
    };
    if (llama_decode(ctx, llama_batch_get_one(ids.data(), (int32_t) ids.size())) != 0) {
        std::fprintf(stderr, "decode failed\n");
        return 1;
    }
    std::printf("output :");
    for (int k = 0; k < n_new; ++k) {
        const float* l = llama_get_logits_ith(ctx, -1);
        dump(l);
        llama_token best = 0;
        for (int i = 1; i < n_vocab; ++i) if (l[i] > l[best]) best = i;
        std::printf(" %d", best);
        std::fflush(stdout);
        if (llama_decode(ctx, llama_batch_get_one(&best, 1)) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
    }
    std::printf("\n");
    std::fclose(out);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
