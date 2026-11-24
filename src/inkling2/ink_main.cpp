// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/ink_main.cpp - Inkling-2-Mini's engine entry inside xestrata-engine (upstream's strata-inkling front
// end, v0.2.19, without its own binary): a .strata v1.1 model given with --native runs here, on the GPU XeStrata
// selected.  The options are the engine's own names (--native, --serve, --max-context, --slots, --tokens, --max-new,
// --temperature, --seed, --dump-logits, --draft, --vision, --vram-reserve-mib); the ones only Inkling has follow.
#include "strata/inkling2/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define STB_IMAGE_STATIC   // a static copy here and in sycl/ink_sycl_backend.cpp (the image engine has its own)
#define INKLING_DEFINE_STBI
#include "ink_backend.h"
#include "ink_batch.h"
#include "ink_manifest.h"
#include "ink_serve.h"
#include "ink_vision.h"

namespace strata::inkling2 {
namespace {

using namespace ink2;

const char* kWho = "xestrata (inkling2_loop)";

bool is_strata_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    unsigned char m[8] = {};
    const bool ok = std::fread(m, 1, 8, f) == 8 && std::memcmp(m, "STRATA\0\x01", 8) == 0;
    std::fclose(f);
    return ok;
}

int cmd_info(const InkModelFile& mf, const std::string& vision) {
    const InkManifest& m = mf.m;
    int images = 0;
    if (!vision.empty()) {
        VisionShardInfo vi;
        std::string verr;
        images = vision_probe(vision, vi, verr) ? 1 : 0;
        if (!images) std::fprintf(stderr, "%s: vision shard: %s\n", kWho, verr.c_str());
    }
    std::printf("arch=inkling2_loop format=strata build=%s layers=%d visits=%d hidden=%d vocab=%d experts=%d eos=%d "
                "drafter=%s images=%d image_pad=%d parallel=%d\n",
                m.build.empty() ? "unknown" : m.build.c_str(), m.c.n_phys, m.c.n_visits, m.c.hidden,
                m.c.vocab_unpadded, m.c.n_experts, m.c.eos_token_id, m.dspark.present ? "dspark" : "none", images,
                m.c.image_token_id, kMaxParallelSlots);
    return 0;
}

int argmax(const float* lg, int V) {
    int best = 0;
    for (int i = 1; i < V; ++i)
        if (lg[i] > lg[best]) best = i;
    return best;
}

// The command line's generation: the prompt, then `max_new` tokens (greedy unless --temperature), DSpark's drafts
// verified in one forward and rolled back exactly where one is rejected (greedy only, as serve's).  The logits of
// every produced token go to `dump` (one row of the vocabulary each, FP32), as the plain engine's --dump-logits.
int cmd_generate(const InkModelFile& mf, InkBackend& backend, const ServeConfig& cfg, const std::vector<int32_t>& ids,
                 int max_new, const SampleOpts& so, bool stop_eos, std::FILE* dump, double load_s) {
    std::string err;
    if (!backend.alloc_slot(0, cfg.bopts.max_ctx, cfg.bopts.kv_dtype, err)) {
        std::fprintf(stderr, "%s: %s\n", kWho, err.c_str());
        return 1;
    }
    if ((int) ids.size() + max_new + 8 > cfg.bopts.max_ctx) {
        std::fprintf(stderr, "%s: the prompt and --max-new pass --max-context %d\n", kWho, cfg.bopts.max_ctx);
        return 2;
    }
    const int V = mf.m.c.vocab_unpadded;
    const bool dspark = cfg.bopts.enable_dspark && mf.m.dspark.present && so.greedy();
    std::vector<float> logits((size_t) 9 * V);
    StepBatch batch;
    std::vector<int32_t> produced, hist = ids;
    std::mt19937_64 rng(so.seed ? so.seed : 20261121ULL);
    auto emit = [&](const float* lg, int32_t tok) -> bool {
        produced.push_back(tok);
        hist.push_back(tok);
        if (dump != nullptr && std::fwrite(lg, sizeof(float), (size_t) V, dump) != (size_t) V) {
            std::fprintf(stderr, "%s: cannot write the logits\n", kWho);
            return false;
        }
        return true;
    };
    auto is_stop = [&](int32_t t) { return stop_eos && (t == mf.m.c.eos_token_id || t == mf.m.c.eot_token_id); };

    const double t1 = now_s();
    for (size_t p0 = 0; p0 < ids.size(); p0 += (size_t) cfg.bopts.chunk) {
        const size_t n = std::min(ids.size() - p0, (size_t) cfg.bopts.chunk);
        batch.clear();
        batch.capture_taps = dspark;
        for (size_t i = 0; i < n; ++i) batch.push(0, (int32_t) (p0 + i), ids[p0 + i], p0 + i + 1 == ids.size());
        if (!backend.forward_batch(batch, logits.data(), nullptr, err)) {
            std::fprintf(stderr, "%s: prompt: %s\n", kWho, err.c_str());
            return 1;
        }
    }
    SampleOpts s = so;
    int32_t tok = sample_logits(logits.data(), V, s, hist, rng);
    if (!emit(logits.data(), tok)) return 1;
    const double t2 = now_s();
    int pos = (int) ids.size();
    long long blocks = 0, proposed = 0, accepted_total = 0;
    while ((int) produced.size() < max_new && !is_stop(tok)) {
        int32_t draft[8] = {};
        const int left = max_new - (int) produced.size();
        const int nd = dspark && left > 1 ? backend.dspark_draft(0, tok, pos, std::min(7, left - 1), draft, err) : 0;
        batch.clear();
        batch.capture_taps = dspark;
        batch.push(0, pos, tok, true);
        for (int i = 0; i < nd; ++i) batch.push(0, pos + 1 + i, draft[i], true);
        if (nd > 0) backend.snapshot_slot(0);
        if (!backend.forward_batch(batch, logits.data(), nullptr, err)) {
            std::fprintf(stderr, "%s: decode: %s\n", kWho, err.c_str());
            return 1;
        }
        int kept = 0;
        for (int i = 0; i <= nd; ++i) {
            const float* lg = logits.data() + (size_t) i * V;
            const int32_t t = nd > 0 ? argmax(lg, V) : sample_logits(logits.data(), V, s, hist, rng);
            if (!emit(lg, t)) return 1;
            ++kept;
            tok = t;
            if (i == nd || t != draft[i] || is_stop(t) || (int) produced.size() >= max_new) break;
        }
        if (nd > 0) {
            ++blocks;
            proposed += nd;
            accepted_total += kept - 1;
            if (kept <= nd) backend.rollback_slot(0, pos + kept);   // the cells and ShortConv of the rejected drafts
        }
        pos += kept;
    }
    const double t3 = now_s();
    std::printf("tokens: ");
    for (size_t i = 0; i < produced.size(); ++i) std::printf(i ? ",%d" : "%d", produced[i]);
    std::printf("\n");
    const double pp = t2 - t1, tg = t3 - t2;
    std::fprintf(stderr, "%s: load %.1f s, prompt %zu tokens %.2f s (%.1f tok/s), %zu generated %.2f s (%.2f tok/s)\n",
                 kWho, load_s, ids.size(), pp, (double) ids.size() / (pp > 0 ? pp : 1), produced.size() - 1, tg,
                 (double) (produced.size() - 1) / (tg > 0 ? tg : 1));
    if (blocks > 0)
        std::fprintf(stderr, "%s: DSpark %lld blocks, %lld proposals, %lld accepted (%.1f%%)\n", kWho, blocks, proposed,
                     accepted_total, proposed > 0 ? 100.0 * (double) accepted_total / (double) proposed : 0.0);
    return 0;
}

int usage(int rc) {
    std::fprintf(rc ? stderr : stdout,
                 "%s: Inkling-2-Mini from its .strata v1.1 or .gguf file\n"
                 "  --native FILE.strata|gguf  the model (SQ-20/15/10 or UD-Q4_K_XL)\n"
                 "  --serve                    the engine's line protocol on stdin/stdout (serve/server.py)\n"
                 "  --tokens IDS --max-new N   the command line's prompt and the tokens to generate (16)\n"
                 "  --temperature T --seed S   sampling (greedy without --temperature)\n"
                 "  --stop-eos                 stop at the model's end tokens\n"
                 "  --dump-logits FILE         every produced token's logits (FP32 rows)\n"
                 "  --max-context N            the context of a slot (32768)\n"
                 "  --slots N                  concurrent conversations, 1..64 (1)\n"
                 "  --chunk N                  prompt tokens a forward (512)\n"
                 "  --draft auto|dspark|off    DSpark's drafts (auto: on where the model has its drafter)\n"
                 "  --vision FILE.strata       Inkling-2-Mini-Vision's shard (pictures)\n"
                 "  --vram-reserve-mib N       VRAM to leave free beside the experts (the engine's room by default)\n"
                 "  --vram-experts-gb G        the experts' VRAM outright\n"
                 "  --qsa-indexer FILE         the QSA indexer (QSAW v1): long prompts attend to the blocks it picks\n"
                 "  --qsa on|off               whether a request without its own qsa= key uses it (off)\n"
                 "  --qsa-budget N --qsa-local N --qsa-sink N   the tokens it keeps (2048, 256, 16)\n"
                 "  --info                     the model's shape, nothing loaded\n"
                 "  --selfcheck [--verify-hash] one token's logits finite (and the file's SHA-256)\n",
                 kWho);
    return rc;
}

}  // namespace

int inkling_main(int argc, char** argv) {
    std::string native;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--native") == 0) native = argv[i + 1];
    if (native.empty() || (!is_strata_file(native) && !GgufAdapter::is_inkling_gguf(native))) return -1;

    ServeConfig cfg;
    cfg.model_path = native;
    cfg.version = STRATA_VERSION;
    cfg.bopts.max_ctx = 32768;
    bool serve = false, info = false, selfcheck = false, verify_hash = false, stop_eos = false;
    std::string tokens, dump_path;
    int max_new = 16;
    SampleOpts so;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: %s needs a value\n", kWho, a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        auto num = [&](long lo, long hi) -> long {
            const char* v = next();
            char* e = nullptr;
            const long x = std::strtol(v, &e, 10);
            if (e == v || *e != '\0' || x < lo || x > hi) {
                std::fprintf(stderr, "%s: %s takes %ld..%ld\n", kWho, a.c_str(), lo, hi);
                std::exit(2);
            }
            return x;
        };
        if (a == "--native" || a == "--gpu") next();   // --gpu: generate.cpp's main took it (XESTRATA_GPU_PCI)
        else if (a == "--serve") serve = true;
        else if (a == "--info") info = true;
        else if (a == "--selfcheck") selfcheck = true;
        else if (a == "--verify-hash") verify_hash = true;
        else if (a == "--stop-eos") stop_eos = true;
        else if (a == "--tokens") tokens = next();
        else if (a == "--max-new") max_new = (int) num(1, 1 << 24);
        else if (a == "--temperature") so.temperature = (float) std::strtod(next(), nullptr);
        else if (a == "--seed") so.seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--dump-logits") dump_path = next();
        else if (a == "--max-context") cfg.bopts.max_ctx = (int) num(256, 1 << 24);
        else if (a == "--slots") cfg.bopts.parallel = (int) num(1, kMaxParallelSlots);
        else if (a == "--chunk") cfg.bopts.chunk = (int) num(1, 1 << 16);
        else if (a == "--vision") cfg.vision_path = cfg.bopts.vision_path = next();
        else if (a == "--vram-reserve-mib") cfg.bopts.vram_reserve_mib = (int) num(0, 1 << 30);
        else if (a == "--vram-experts-gb") cfg.bopts.vram_experts_gb = std::strtod(next(), nullptr);
        else if (a == "--draft") {
            cfg.draft_mode = next();
            if (cfg.draft_mode != "auto" && cfg.draft_mode != "dspark" && cfg.draft_mode != "off") {
                std::fprintf(stderr, "%s: --draft takes auto, dspark or off\n", kWho);
                return 2;
            }
            cfg.bopts.enable_dspark = cfg.draft_mode != "off";
        }
        else if (a == "--qsa-indexer") cfg.bopts.qsa_path = next();
        else if (a == "--qsa") {
            const std::string v = next();
            if (v != "on" && v != "off") {
                std::fprintf(stderr, "%s: --qsa takes on or off\n", kWho);
                return 2;
            }
            cfg.bopts.qsa_default = v == "on";
        }
        else if (a == "--qsa-budget") cfg.bopts.qsa_budget = (int) num(64, 1 << 20);
        else if (a == "--qsa-local") cfg.bopts.qsa_local = (int) num(0, 1 << 20);
        else if (a == "--qsa-sink") cfg.bopts.qsa_sink = (int) num(0, 1 << 20);
        else if (a == "--help" || a == "-h") return usage(0);
        else {
            std::fprintf(stderr, "%s: unknown option %s (Inkling-2-Mini's engine takes these:)\n", kWho, a.c_str());
            return usage(2);
        }
    }
    if (cfg.bopts.qsa_default && cfg.bopts.qsa_path.empty()) {
        std::fprintf(stderr, "%s: --qsa on needs --qsa-indexer\n", kWho);
        return 2;
    }
    std::string err;
    if (!qsa_geometry_ok(cfg.bopts, err)) {
        std::fprintf(stderr, "%s: %s\n", kWho, err.c_str());
        return 2;
    }

    const double t0 = now_s();
    InkModelFile mf;
    if (!mf.open(native, err, /*keep_tokenizer=*/false)) {
        if (serve) proto("ERR open: %s", err.c_str());
        std::fprintf(stderr, "%s: %s: %s\n", kWho, native.c_str(), err.c_str());
        return 1;
    }
    if (mf.m.arch != "inkling2_loop" && !mf.m.arch.empty()) {
        std::fprintf(stderr, "%s: %s is a .strata file of %s, not Inkling-2-Mini\n", kWho, native.c_str(),
                     mf.m.arch.c_str());
        return 2;
    }
    if (info) return cmd_info(mf, cfg.vision_path);
    if (verify_hash && !mf.verify_sections(err, true)) {
        std::fprintf(stderr, "%s: %s\n", kWho, err.c_str());
        return 1;
    }

    std::unique_ptr<InkBackend> backend = make_sycl_backend();
    if (!backend->init(mf, cfg.bopts, err)) {
        if (serve) proto("ERR init: %s", err.c_str());
        std::fprintf(stderr, "%s: %s\n", kWho, err.c_str());
        return 1;
    }
    std::fprintf(stderr, "%s: %s on %s\n", kWho, native.c_str(), backend->device_info().c_str());
    if (selfcheck) {
        if (!backend->selfcheck(err)) {
            std::fprintf(stderr, "%s: selfcheck: %s\n", kWho, err.c_str());
            return 1;
        }
        std::printf("selfcheck: ok\n");
        return 0;
    }
    if (serve) return run_serve_loop(cfg, mf, *backend);

    std::vector<int32_t> ids;
    if (tokens.empty() || !parse_id_list(tokens, ids, err)) {
        std::fprintf(stderr, "%s: --tokens takes the prompt's token ids (comma separated)\n", kWho);
        return 2;
    }
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> dump(nullptr, &std::fclose);
    if (!dump_path.empty()) {
        dump.reset(std::fopen(dump_path.c_str(), "wb"));
        if (!dump) {
            std::fprintf(stderr, "%s: cannot write %s\n", kWho, dump_path.c_str());
            return 1;
        }
    }
    backend->set_slot_qsa(0, cfg.bopts.qsa_default);
    return cmd_generate(mf, *backend, cfg, ids, max_new, so, stop_eos, dump.get(), now_s() - t0);
}

}  // namespace strata::inkling2
