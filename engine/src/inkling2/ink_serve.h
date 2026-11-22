// engine/src/inkling2/ink_serve.h - Continuous-Batching (1..64 slots) `--serve` Protocol Engine for strata-inkling
// Speaks Strata's native line protocol (INFO / READY / GEN / BGEN / BCO / STOP / BSTOP / BYIELD / QUIT -> PP / T / DONE / BADM / BT / BDONE / BCOK / IMG / ERR)
// PLUS extended protocol commands (HELLO / GENERATE <req_id> / CANCEL <req_id> / STATS) on the same non-blocking stdin loop.
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ink_backend.h"
#include "ink_batch.h"
#include "ink_manifest.h"
#include "ink_vision.h"

namespace ink2 {

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#ifndef _MSC_VER
inline void proto(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#endif
inline void proto(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fputs(buf, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

struct SampleOpts {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 64;
    float min_p = 0.0f;
    float rep = 1.0f;
    float freq = 0.0f;
    float pres = 0.0f;
    int last_n = 64;
    uint64_t seed = 0;
    int hold_budget = 0;       // hold=N:ID reasoning budget
    int32_t hold_token_id = -1;
    int pen_from = 0;

    bool greedy() const noexcept { return temperature <= 0.0f; }
    bool penalties() const noexcept { return rep != 1.0f || freq != 0.0f || pres != 0.0f; }
};

inline int32_t sample_logits(float* lg, int V, const SampleOpts& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.penalties() && s.last_n > 0) {
        size_t start = static_cast<size_t>(std::max(0, s.pen_from));
        if (hist.size() > start) {
            size_t from = (hist.size() - start > static_cast<size_t>(s.last_n)) ? (hist.size() - static_cast<size_t>(s.last_n)) : start;
            std::unordered_map<int32_t, int> cnt;
            for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
            for (const auto& kv : cnt) {
                if (kv.first < 0 || kv.first >= V) continue;
                float& x = lg[kv.first];
                if (s.rep != 1.0f) x = (x > 0.0f) ? (x / s.rep) : (x * s.rep);
                x -= static_cast<float>(kv.second) * s.freq + (s.pres != 0.0f ? s.pres : 0.0f);
            }
        }
    }
    if (s.greedy()) {
        int32_t best = 0;
        float mx = lg[0];
        for (int i = 1; i < V; ++i) {
            if (lg[i] > mx) { mx = lg[i]; best = i; }
        }
        return best;
    }
    const int K = std::min(std::max(1, s.top_k), std::min(V, 128));
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    std::vector<double> p(K);
    double mx = lg[idx[0]], sum = 0.0;
    for (int i = 0; i < K; ++i) {
        p[i] = std::exp((lg[idx[i]] - mx) / static_cast<double>(s.temperature));
        sum += p[i];
    }
    for (double& x : p) x /= sum;
    int keep = K;
    if (s.min_p > 0.0f) {
        const double thr = static_cast<double>(s.min_p) * p[0];
        keep = 1;
        while (keep < K && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.0f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= static_cast<double>(s.top_p)) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    for (int i = 0; i < keep; ++i) z += p[i];
    std::uniform_real_distribution<double> u(0.0, z);
    const double r = u(rng);
    double c = 0.0;
    for (int i = 0; i < keep; ++i) {
        c += p[i];
        if (r <= c) return idx[i];
    }
    return idx[keep - 1];
}

inline bool finite_logits(const float* lg, int V) noexcept {
    for (int i = 0; i < V; i += 67) if (!std::isfinite(lg[i])) return false;
    return true;
}

// Exact lossless speculative verification sampler for DSpark (Leviathan et al., 2023):
// Given target logits `lg[0..V-1]` and a deterministic draft token `draft_tok`:
//   - If greedy: picks argmax and accepts iff argmax == draft_tok.
//   - If non-greedy (temperature > 0 / top_k / min_p / top_p / penalties):
//     Computes exact target distribution p(x), accepts `draft_tok` with probability p(draft_tok),
//     else samples from exact residual p(x) / (1 - p(draft_tok)) over x != draft_tok.
// Marginal distribution of emitted tokens is 100% mathematically identical to sample_logits()!
inline int32_t verify_draft_logits(
    float* lg, int V, const SampleOpts& s, const std::vector<int32_t>& hist,
    std::mt19937_64& rng, int32_t draft_tok, bool& accepted) {
    accepted = false;
    if (s.greedy()) {
        const int32_t tok = sample_logits(lg, V, s, hist, rng);
        accepted = (tok == draft_tok);
        return tok;
    }
    if (s.penalties() && s.last_n > 0) {
        size_t start = static_cast<size_t>(std::max(0, s.pen_from));
        if (hist.size() > start) {
            size_t from = (hist.size() - start > static_cast<size_t>(s.last_n)) ? (hist.size() - static_cast<size_t>(s.last_n)) : start;
            std::unordered_map<int32_t, int> cnt;
            for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
            for (const auto& kv : cnt) {
                if (kv.first < 0 || kv.first >= V) continue;
                float& x = lg[kv.first];
                if (s.rep != 1.0f) x = (x > 0.0f) ? (x / s.rep) : (x * s.rep);
                x -= static_cast<float>(kv.second) * s.freq + (s.pres != 0.0f ? s.pres : 0.0f);
            }
        }
    }
    const int K = std::min(std::max(1, s.top_k), std::min(V, 128));
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    std::vector<double> p(K);
    double mx = lg[idx[0]], sum = 0.0;
    for (int i = 0; i < K; ++i) {
        p[i] = std::exp((lg[idx[i]] - mx) / static_cast<double>(s.temperature));
        sum += p[i];
    }
    for (double& x : p) x /= sum;
    int keep = K;
    if (s.min_p > 0.0f) {
        const double thr = static_cast<double>(s.min_p) * p[0];
        keep = 1;
        while (keep < K && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.0f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= static_cast<double>(s.top_p)) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    int draft_pos = -1;
    for (int i = 0; i < keep; ++i) {
        z += p[i];
        if (idx[i] == draft_tok) draft_pos = i;
    }
    if (z <= 0.0) return idx[0];
    const double p_draft = (draft_pos >= 0) ? (p[draft_pos] / z) : 0.0;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (draft_pos >= 0 && (keep == 1 || u01(rng) < p_draft)) {
        accepted = true;
        return draft_tok;
    }
    // Rejected: sample from residual p(x) over i != draft_pos
    const double z_rem = z - (draft_pos >= 0 ? p[draft_pos] : 0.0);
    if (z_rem <= 0.0) {
        accepted = (draft_pos >= 0);
        return (draft_pos >= 0) ? draft_tok : idx[0];
    }
    const double r = u01(rng) * z_rem;
    double c = 0.0;
    int last_valid = idx[0];
    for (int i = 0; i < keep; ++i) {
        if (i == draft_pos) continue;
        last_valid = idx[i];
        c += p[i];
        if (r <= c) return idx[i];
    }
    return last_valid;
}

// Anti-loop repetition detector: detects pathological n-gram stalling while avoiding false positives on structured YAML/JSON (#6620)
inline bool detect_repetition_loop(const std::vector<int32_t>& gen_ids) noexcept {
    const size_t n = gen_ids.size();
    if (n < 96) return false;
    for (size_t period = 1; period <= 32; ++period) {
        const size_t need_reps = (period <= 4) ? 24 : 10;
        if (n < period * need_reps) continue;
        bool match = true;
        for (size_t k = 0; k < period * (need_reps - 1); ++k) {
            if (gen_ids[n - 1 - k] != gen_ids[n - 1 - k - period]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

struct ParsedGen {
    std::string cmd;      // GEN, BGEN, GENERATE
    int slot = 0;         // for BGEN <slot> ...
    std::string req_id;   // for GENERATE <req_id> ...
    int max_new = 128;
    SampleOpts s;
    std::vector<int32_t> ids;
    std::vector<std::string> images;
    bool prefill_only = false;
    bool slot_read = false;
};

inline bool parse_id_list(const std::string& s, std::vector<int32_t>& out, std::string& err) {
    out.clear();
    size_t p = 0;
    while (p < s.size()) {
        char* ee = nullptr;
        long v = std::strtol(s.c_str() + p, &ee, 10);
        if (ee == s.c_str() + p) {
            err = "bad token id list";
            return false;
        }
        out.push_back(static_cast<int32_t>(v));
        p = static_cast<size_t>(ee - s.c_str());
        if (p < s.size() && s[p] == ',') ++p;
    }
    return !out.empty();
}

inline bool parse_gen_cmd(const std::string& line, ParsedGen& r, std::string& err) {
    std::vector<std::string> w;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && line[i] == ' ') ++i;
        size_t j = i;
        while (j < line.size() && line[j] != ' ') ++j;
        if (j > i) w.push_back(line.substr(i, j - i));
        i = j;
    }
    if (w.empty()) return false;
    r.cmd = w[0];
    size_t arg_idx = 1;
    if (r.cmd == "BGEN") {
        if (w.size() < 4) { err = "expected: BGEN <slot> <max_new> [k=v ...] <ids>"; return false; }
        r.slot = std::atoi(w[1].c_str());
        arg_idx = 2;
    } else if (r.cmd == "GENERATE") {
        if (w.size() < 4) { err = "expected: GENERATE <req_id> <max_new> [k=v ...] <ids>"; return false; }
        r.req_id = w[1];
        arg_idx = 2;
    } else if (r.cmd == "GEN") {
        if (w.size() < 3) { err = "expected: GEN <max_new> [k=v ...] <ids>"; return false; }
        r.slot = 0;
        arg_idx = 1;
    } else {
        err = "unknown command " + r.cmd;
        return false;
    }
    char* e = nullptr;
    long mn = std::strtol(w[arg_idx].c_str(), &e, 10);
    if (*e || mn < 0) { err = "bad max_new"; return false; }
    r.max_new = static_cast<int>(mn);

    for (size_t k = arg_idx + 1; k + 1 < w.size(); ++k) {
        size_t eq = w[k].find('=');
        if (eq == std::string::npos) { err = "bad key " + w[k]; return false; }
        std::string key = w[k].substr(0, eq), val = w[k].substr(eq + 1);
        double d = std::atof(val.c_str());
        if (key == "temperature") r.s.temperature = static_cast<float>(d);
        else if (key == "top_p") r.s.top_p = static_cast<float>(d);
        else if (key == "top_k") r.s.top_k = std::max(1, std::min(128, static_cast<int>(d)));
        else if (key == "min_p") r.s.min_p = static_cast<float>(d);
        else if (key == "penalty_repeat") r.s.rep = static_cast<float>(d);
        else if (key == "penalty_freq") r.s.freq = static_cast<float>(d);
        else if (key == "penalty_present") r.s.pres = static_cast<float>(d);
        else if (key == "penalty_last_n") r.s.last_n = static_cast<int>(d);
        else if (key == "pen_from") r.s.pen_from = static_cast<int>(d);
        else if (key == "seed") r.s.seed = std::strtoull(val.c_str(), nullptr, 10);
        else if (key == "prefill") r.prefill_only = (d != 0.0);
        else if (key == "slot_read") r.slot_read = (d != 0.0);
        else if (key == "image") r.images.push_back(pct_decode(val));
        else if (key == "hold") {
            size_t col = val.find(':');
            if (col != std::string::npos) {
                r.s.hold_budget = std::atoi(val.substr(0, col).c_str());
                r.s.hold_token_id = std::atoi(val.substr(col + 1).c_str());
            }
        } else if (key == "pcie_frac" || key == "spec_min_p" || key == "prefill_pipe") {
            continue;
        } else {
            err = "unsupported key " + key;
            return false;
        }
    }
    return parse_id_list(w.back(), r.ids, err);
}

// ---- Non-blocking stdin reader thread ----
class ServeInput {
public:
    std::atomic<bool> solo_stop{false};
    void start() {
        th_ = std::thread([this] {
            std::string line;
            while (std::getline(std::cin, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                if (line == "STOP") {
                    solo_stop.store(true);
                    std::lock_guard<std::mutex> lk(m_);
                    q_.push_back(line);
                    cv_.notify_one();
                    continue;
                }
                std::lock_guard<std::mutex> lk(m_);
                q_.push_back(std::move(line));
                cv_.notify_one();
            }
            std::lock_guard<std::mutex> lk(m_);
            eof_ = true;
            cv_.notify_all();
        });
        th_.detach();
    }

    bool poll(std::string& out) {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    bool wait_next(std::string& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return !q_.empty() || eof_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    bool is_eof() {
        std::lock_guard<std::mutex> lk(m_);
        return eof_ && q_.empty();
    }

private:
    std::thread th_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    bool eof_ = false;
};

struct ActiveReq {
    bool active = false;
    bool is_solo = false;       // GEN vs BGEN / GENERATE
    bool is_ext_gen = false;    // GENERATE <req_id>
    std::string req_id;
    int slot = 0;
    int max_new = 0;
    int generated = 0;
    int prompt_len = 0;
    int prefill_pos = 0;        // next prompt token index to feed
    int cur_pos = 0;            // next decode position in slot
    int32_t next_tok = -1;      // token ready to feed in decode step
    SampleOpts s;
    std::mt19937_64 rng;
    std::vector<int32_t> prompt_ids;
    std::vector<int32_t> prompt_img_idx;
    std::vector<float> img_embeds;
    std::vector<int32_t> hist;
    std::vector<int32_t> gen_ids;
    std::vector<int32_t> cached_prefix;       // tokens currently resident in slot KV
    std::vector<int32_t> prompt_ckpt_prefix;  // prompt-end checkpoint prefix saved before generation (#6627)
    int reused_tokens = 0;                    // prefix tokens reused for this request
    bool ckpt_pending = false;                // true while prefilling up to prompt_len - 1 to save checkpoint
    double t_start = 0.0;
    double prompt_ms = 0.0;
    double decode_t0 = 0.0;
    int drafts_offered = 0;
    int drafts_accepted = 0;
    bool cancelled = false;
    bool admitted = false;
    bool slot_read = false;
};

struct ServeConfig {
    std::string model_path;
    std::string vision_path;
    BackendOpts bopts;
    std::string draft_mode = "auto";  // auto | dspark | off
    bool force_cpu = false;
    bool anti_loop = true;
    std::string version = "0.1.66";
};

inline int run_serve_loop(const ServeConfig& cfg, InkModelFile& mf, InkBackend& backend) {
    const InkCfg& c = mf.m.c;
    const int V = c.vocab_unpadded;
    const int P = std::max(1, std::min(kMaxParallelSlots, cfg.bopts.parallel));
    const int chunk_sz = std::max(1, cfg.bopts.chunk);
    const bool use_dspark = (cfg.draft_mode != "off") && mf.m.dspark.present && backend.dspark_ready();

    // Probe Vision shard lazily (zero VRAM/RAM until first image request)
    bool images_ok = false;
    VisionShardInfo vprobe;
    CpuVisionEncoder cpu_vit;
    if (!cfg.vision_path.empty()) {
        std::string verr;
        if (vision_probe(cfg.vision_path, vprobe, verr)) images_ok = true;
        else fprintf(stderr, "[inkling2-vision] disabled: %s\n", verr.c_str());
    }

    // Allocate all P slots (lazy paged KV uses near-zero memory until populated!)
    for (int s = 0; s < P; ++s) {
        std::string aerr;
        if (!backend.alloc_slot(s, cfg.bopts.max_ctx, cfg.bopts.kv_dtype, aerr)) {
            proto("ERR alloc_slot %d: %s", s, aerr.c_str());
            return 1;
        }
    }

    std::string base = cfg.model_path;
    size_t sl = base.find_last_of("/\\");
    if (sl != std::string::npos) base = base.substr(sl + 1);

    const int eff_max_ctx = (backend.effective_max_ctx() > 0) ? backend.effective_max_ctx() : cfg.bopts.max_ctx;
    const auto kv_str = [&]() -> const char* {
        const KvDtype kd = backend.stats().kv_dtype;
        if (kd == KvDtype::FP8_E4M3) return "fp8";
        if (kd == KvDtype::K8V8)     return "k8v8";
        if (kd == KvDtype::K8V4)     return "k8v4";
        return "fp16";
    }();
    proto("INFO engine=strata-inkling-%s arch=inkling2_loop backend=%s model=%s layers=%d visits=%d vocab=%d chunk=%d batch_slots=%d slot_read=1 kv_dtype=%s drafter=%s drafting=%s images=%d image_pad=%d parallel=%d",
          cfg.version.c_str(), backend.backend_name(), base.c_str(), c.n_phys, c.n_visits, V, chunk_sz,
          P, kv_str,
          mf.m.dspark.present ? "dspark" : "none",
          use_dspark ? "dspark" : "off",
          images_ok ? 1 : 0, c.image_token_id, kMaxParallelSlots);
    proto("READY %d stop hold holdb", eff_max_ctx);

    std::vector<ActiveReq> reqs(P);
    std::unordered_map<std::string, int> req_id_to_slot;
    ServeInput in;
    in.start();

    auto is_stop_tok = [&](int32_t tok) -> bool {
        return tok == c.eos_token_id || tok == c.eot_token_id;
    };

    auto finish_req = [&](ActiveReq& rq, const char* reason) {
        const double dec_ms = (rq.decode_t0 > 0.0) ? (now_s() - rq.decode_t0) * 1000.0 : 0.0;
        if (rq.is_solo) {
            // Emit full 9-field DONE line so server.py _parse_done() records drafts_accepted, drafts_offered, reused (#6633, #6627)!
            proto("DONE %d %d %.1f %.1f %s %d %d %d",
                  rq.generated, rq.prompt_len, rq.prompt_ms, dec_ms, reason,
                  rq.drafts_accepted, rq.drafts_offered, rq.reused_tokens);
        } else if (rq.is_ext_gen) {
            proto("DONE %s %d %d %.1f %.1f %s %d %d %d",
                  rq.req_id.c_str(), rq.generated, rq.prompt_len, rq.prompt_ms, dec_ms, reason,
                  rq.drafts_accepted, rq.drafts_offered, rq.reused_tokens);
            req_id_to_slot.erase(rq.req_id);
        } else {
            if (!rq.admitted) {
                proto("DONE %d %d %.1f 0.0 %s %d %d %d",
                      rq.generated, rq.prompt_len, rq.prompt_ms, reason,
                      rq.drafts_accepted, rq.drafts_offered, rq.reused_tokens);
                proto("BADM %d 0", rq.slot);
            } else {
                char dstr[64] = "";
                if (rq.drafts_offered > 0) snprintf(dstr, sizeof(dstr), " drafts=%d:%d", rq.drafts_offered, rq.drafts_accepted);
                proto("BDONE %d %d %s %.1f read=%.1f%s", rq.slot, rq.generated, reason, dec_ms, rq.prompt_ms, dstr);
            }
        }
        rq.active = false;
    };

    auto prepare_req = [&](const ParsedGen& pg, ActiveReq& rq, std::string& err) -> bool {
        rq.active = true;
        rq.is_solo = (pg.cmd == "GEN");
        rq.is_ext_gen = (pg.cmd == "GENERATE");
        rq.req_id = pg.req_id;
        rq.slot = pg.slot;
        rq.max_new = pg.max_new;
        rq.generated = 0;
        rq.s = pg.s;
        rq.rng.seed(pg.s.seed ? pg.s.seed : 20261121ULL);
        rq.cancelled = false;
        rq.admitted = false;
        rq.slot_read = pg.slot_read;
        rq.drafts_offered = 0;
        rq.drafts_accepted = 0;
        rq.t_start = now_s();
        rq.prompt_ms = 0.0;
        rq.decode_t0 = 0.0;
        rq.gen_ids.clear();
        rq.img_embeds.clear();
        rq.prompt_img_idx.clear();

        if (!pg.images.empty()) {
            if (!images_ok) {
                err = "image: engine started without --vision Inkling-2-Mini-Vision.strata";
                rq.active = false;
                return false;
            }
            std::vector<ImageRec> recs;
            for (const auto& f : pg.images) {
                ImageRec rc;
                if (!prepare_picture(f, vprobe.c, rc, err)) { rq.active = false; return false; }
                recs.push_back(std::move(rc));
            }
            if (!expand_image_pads(pg.ids, c.image_token_id, recs, rq.prompt_ids, rq.prompt_img_idx, err)) {
                rq.active = false;
                return false;
            }
            // Encode images (try GPU backend first, fallback to CPU ViT)
            for (size_t k = 0; k < recs.size(); ++k) {
                std::vector<float> emb;
                std::string verr;
                if (!backend.encode_image(recs[k], emb, verr)) {
                    if (!cpu_vit.loaded && !cpu_vit.load(cfg.vision_path, err)) { rq.active = false; return false; }
                    if (!cpu_vit.encode(recs[k], emb, err)) { rq.active = false; return false; }
                }
                rq.img_embeds.insert(rq.img_embeds.end(), emb.begin(), emb.end());
                proto("IMG %zu %d %d %d", k, recs[k].grid_h, recs[k].grid_w, recs[k].num_lm_tokens());
            }
        } else {
            rq.prompt_ids = pg.ids;
            rq.prompt_img_idx.assign(rq.prompt_ids.size(), -1);
        }

        rq.prompt_len = static_cast<int>(rq.prompt_ids.size());
        if (rq.prompt_len <= 0 || rq.prompt_len >= eff_max_ctx) {
            err = "prompt length out of range (1.." + std::to_string(eff_max_ctx - 1) + ")";
            rq.active = false;
            return false;
        }
        // Prefix reuse check (#6627): prefer full cached_prefix match, else restore prompt-end checkpoint!
        // Because prompt_ckpt_prefix is saved at prompt_len - 1 before generation, BOTH multi-turn follow-ups
        // (even if hidden thinking or stop tokens differed) AND exact repeats reuse the full prior prompt!
        int reuse = 0;
        bool used_ckpt = false;
        if (pg.images.empty()) {
            int full_match = 0;
            if (!rq.cached_prefix.empty() && static_cast<int>(rq.cached_prefix.size()) < rq.prompt_len) {
                const int clen = static_cast<int>(rq.cached_prefix.size());
                int m = 0;
                while (m < clen && rq.cached_prefix[m] == rq.prompt_ids[m]) ++m;
                if (m == clen) full_match = clen;
            }
            int ckpt_match = 0;
            if (!rq.prompt_ckpt_prefix.empty() && static_cast<int>(rq.prompt_ckpt_prefix.size()) < rq.prompt_len) {
                const int plen = static_cast<int>(rq.prompt_ckpt_prefix.size());
                int m = 0;
                while (m < plen && rq.prompt_ckpt_prefix[m] == rq.prompt_ids[m]) ++m;
                if (m == plen) ckpt_match = plen;
            }
            if (full_match > 0 && full_match >= ckpt_match) {
                reuse = full_match;
            } else if (ckpt_match > 0 && backend.restore_prompt_checkpoint(rq.slot, ckpt_match)) {
                reuse = ckpt_match;
                rq.cached_prefix = rq.prompt_ckpt_prefix;
                used_ckpt = true;
            }
        }
        (void)used_ckpt;
        if (reuse == 0) {
            backend.reset_slot(rq.slot);
            rq.cached_prefix.clear();
            rq.prompt_ckpt_prefix.clear();
        } else {
            proto("RESUME %d", reuse);
        }
        rq.reused_tokens = reuse;
        rq.ckpt_pending = (pg.images.empty() && rq.prompt_len >= 16 && reuse < rq.prompt_len - 1);
        rq.prefill_pos = reuse;
        rq.cur_pos = reuse;
        rq.hist = rq.prompt_ids;
        rq.next_tok = -1;
        return true;
    };

    bool quit_requested = false;
    auto handle_line = [&](const std::string& line) -> bool {
        if (line.empty()) return true;
        if (line == "QUIT") { quit_requested = true; return false; }
        if (line == "HELLO") {
            proto("HELLO ok engine=strata-inkling-%s arch=inkling2_loop backend=%s max_ctx=%d parallel=%d",
                  cfg.version.c_str(), backend.backend_name(), eff_max_ctx, P);
            return true;
        }
        if (line == "STATS") {
            auto st = backend.stats();
            proto("STATS active_slots=%d max_slots=%d vram_experts=%zu ram_experts=%zu total_experts=%zu kv=%s",
                  st.active_slots, st.max_slots, st.vram_experts_loaded, st.ram_experts_loaded, st.total_experts,
                  st.kv_dtype == KvDtype::FP8_E4M3 ? "fp8" : "fp16");
            return true;
        }
        if (line == "STOP") {
            for (auto& rq : reqs) if (rq.active && rq.is_solo) rq.cancelled = true;
            return true;
        }
        if (line.rfind("BSTOP ", 0) == 0) {
            int s = std::atoi(line.c_str() + 6);
            if (s >= 0 && s < P && reqs[s].active) reqs[s].cancelled = true;
            return true;
        }
        if (line.rfind("CANCEL ", 0) == 0) {
            std::string rid = line.substr(7);
            auto it = req_id_to_slot.find(rid);
            if (it != req_id_to_slot.end() && reqs[it->second].active) reqs[it->second].cancelled = true;
            return true;
        }
        if (line.rfind("BYIELD ", 0) == 0) {
            // Interleaved continuous batching already prevents long-prompt head-of-line blocking!
            return true;
        }
        if (line.rfind("BCO ", 0) == 0) {
            // BCO <slot> <ids>: pre-register coalesced prompt read into free slot
            size_t sp = line.find(' ', 4);
            if (sp != std::string::npos) {
                int s = std::atoi(line.substr(4, sp - 4).c_str());
                std::vector<int32_t> cids;
                std::string cerr;
                if (s >= 0 && s < P && !reqs[s].active && parse_id_list(line.substr(sp + 1), cids, cerr)) {
                    backend.reset_slot(s);
                    reqs[s].cached_prefix.clear();
                    proto("BCOK %d %zu", s, cids.size());
                }
            }
            return true;
        }
        if (line.rfind("GEN ", 0) == 0 || line.rfind("BGEN ", 0) == 0 || line.rfind("GENERATE ", 0) == 0) {
            ParsedGen pg;
            std::string perr;
            if (!parse_gen_cmd(line, pg, perr)) {
                proto("ERR %s", perr.c_str());
                return true;
            }
            if (pg.cmd == "GENERATE") {
                int free_s = -1;
                for (int s = 0; s < P; ++s) if (!reqs[s].active) { free_s = s; break; }
                if (free_s < 0) {
                    proto("ERR busy: all %d parallel slots active", P);
                    return true;
                }
                pg.slot = free_s;
            }
            if (pg.slot < 0 || pg.slot >= P) {
                proto("ERR BGEN: slot %d out of range 0..%d", pg.slot, P - 1);
                return true;
            }
            if (reqs[pg.slot].active) {
                proto("ERR BGEN: slot %d is busy", pg.slot);
                return true;
            }
            if (!prepare_req(pg, reqs[pg.slot], perr)) {
                proto("ERR %s", perr.c_str());
                return true;
            }
            if (pg.cmd == "GENERATE") {
                req_id_to_slot[pg.req_id] = pg.slot;
            } else if (pg.cmd == "BGEN" && pg.slot_read && pg.max_new > 0) {
                // Zero-HoL admission: immediately release server.py control lines so concurrent
                // HTTP requests 1..64 all prefill and decode together in the coalesced StepBatch!
                proto("DONE 0 %d 0.0 0.0 admit 0 0 %d", reqs[pg.slot].prompt_len, reqs[pg.slot].reused_tokens);
                proto("BADM %d 1", pg.slot);
                reqs[pg.slot].admitted = true;
            }
            return true;
        }
        proto("ERR unknown command");
        return true;
    };

    // Main continuous batching event loop!
    StepBatch batch;
    std::vector<float> merged_img_embeds;
    std::vector<int> logit_slots;
    std::vector<float> logits_buf;

    while (true) {
        // 1. Drain all queued stdin commands without blocking while any slot is active
        bool any_active = false;
        for (const auto& rq : reqs) if (rq.active) { any_active = true; break; }

        std::string line;
        if (!any_active) {
            if (quit_requested) break;
            if (!in.wait_next(line)) break;
            handle_line(line);
            while (!quit_requested && in.poll(line)) {
                handle_line(line);
            }
        } else {
            while (!quit_requested && in.poll(line)) {
                handle_line(line);
            }
        }

        // Handle cancellations immediately
        for (auto& rq : reqs) {
            if (rq.active && rq.cancelled) finish_req(rq, "cancel");
        }

        // Count active slots
        int n_active = 0;
        int sole_slot = -1;
        for (int s = 0; s < P; ++s) {
            if (reqs[s].active) { ++n_active; sole_slot = s; }
        }
        if (n_active == 0) {
            if (quit_requested || in.is_eof()) break;
            continue;
        }

        // 2. Fast-path: Single-stream DSpark Adaptive Beam Tree (greedy) / Block-8 Rejection Sampling (temperature >= 0) (#6633)!
        if (use_dspark && n_active == 1 && reqs[sole_slot].prefill_pos >= reqs[sole_slot].prompt_len &&
            reqs[sole_slot].generated + 2 <= reqs[sole_slot].max_new &&
            reqs[sole_slot].cur_pos + 8 < eff_max_ctx) {
            auto& rq = reqs[sole_slot];
            // 2a. Zero-D2H fused greedy Beam Tree speculative step (when greedy, no penalties, and headroom >= 12)
            if (rq.s.greedy() && !rq.s.penalties() && rq.cur_pos + 12 < eff_max_ctx) {
                int32_t emitted[16]{};
                int drafted = 0;
                std::string serr;
                const int max_emit = rq.max_new - rq.generated;
                const int n_emit = backend.dspark_spec_step_greedy(
                    rq.slot, rq.next_tok, rq.cur_pos, max_emit, emitted, &drafted, serr);
                if (n_emit < 0) {
                    proto("ERR fatal GPU error: %s", serr.c_str());
                    return 1;
                }
                if (n_emit > 0) {
                    rq.drafts_offered += drafted;
                    int kept = 0;
                    bool done = false;
                    const char* fin = nullptr;
                    for (int i = 0; i < n_emit; ++i) {
                        const int32_t tok = emitted[i];
                        rq.hist.push_back(tok);
                        rq.gen_ids.push_back(tok);
                        rq.cached_prefix.push_back(i == 0 ? rq.next_tok : emitted[i - 1]);
                        ++rq.generated;
                        ++kept;
                        if (i + 1 < n_emit) ++rq.drafts_accepted;
                        if (rq.is_solo) proto("T %d", tok);
                        else if (rq.is_ext_gen) proto("T %s %d", rq.req_id.c_str(), tok);
                        else proto("BT %d %d", rq.slot, tok);

                        if (is_stop_tok(tok)) { fin = "stop"; done = true; break; }
                        if (rq.generated >= rq.max_new) { fin = "length"; done = true; break; }
                        if (cfg.anti_loop && detect_repetition_loop(rq.gen_ids)) { fin = "stop"; done = true; break; }
                    }
                    if (!done) {
                        rq.next_tok = emitted[n_emit - 1];
                        rq.cur_pos += n_emit;
                    } else {
                        if (kept < n_emit) {
                            backend.rollback_slot(rq.slot, rq.cur_pos + kept);
                        }
                        rq.cur_pos += kept;
                        if (fin) finish_req(rq, fin);
                    }
                    continue;
                }
            }
            int32_t draft_ids[8]{};
            std::string derr;
            const int max_d = std::min(7, rq.max_new - rq.generated - 1);
            int nd = backend.dspark_draft(rq.slot, rq.next_tok, rq.cur_pos, max_d, draft_ids, derr);
            if (nd > 0) {
                backend.snapshot_slot(rq.slot);
                batch.clear();
                batch.capture_taps = true;
                batch.push(rq.slot, rq.cur_pos, rq.next_tok, true);
                for (int i = 0; i < nd; ++i) {
                    batch.push(rq.slot, rq.cur_pos + 1 + i, draft_ids[i], true);
                }
                logits_buf.resize(static_cast<size_t>(nd + 1) * V);
                std::string ferr;
                if (!backend.forward_batch(batch, logits_buf.data(), nullptr, ferr)) {
                    proto("ERR fatal GPU error: %s", ferr.c_str());
                    return 1;
                }
                rq.drafts_offered += nd;
                int accepted = 0;
                bool done = false;
                const char* fin = nullptr;
                for (int i = 0; i <= nd; ++i) {
                    float* lg = logits_buf.data() + static_cast<size_t>(i) * V;
                    if (!finite_logits(lg, V)) {
                        // Never leave poisoned KV/ShortConv prefix cached across retries (#879)!
                        rq.cached_prefix.clear();
                        rq.prompt_ckpt_prefix.clear();
                        backend.reset_slot(rq.slot);
                        finish_req(rq, "nonfinite");
                        done = true;
                        break;
                    }
                    bool draft_ok = false;
                    int32_t tok = (i < nd)
                        ? verify_draft_logits(lg, V, rq.s, rq.hist, rq.rng, draft_ids[i], draft_ok)
                        : sample_logits(lg, V, rq.s, rq.hist, rq.rng);
                    rq.hist.push_back(tok);
                    rq.gen_ids.push_back(tok);
                    rq.cached_prefix.push_back(i == 0 ? rq.next_tok : draft_ids[i - 1]);
                    ++rq.generated;
                    if (rq.is_solo) proto("T %d", tok);
                    else if (rq.is_ext_gen) proto("T %s %d", rq.req_id.c_str(), tok);
                    else proto("BT %d %d", rq.slot, tok);

                    if (is_stop_tok(tok)) { fin = "stop"; done = true; ++accepted; break; }
                    if (rq.generated >= rq.max_new) { fin = "length"; done = true; ++accepted; break; }
                    if (cfg.anti_loop && detect_repetition_loop(rq.gen_ids)) { fin = "stop"; done = true; ++accepted; break; }
                    if (i < nd && draft_ok) {
                        ++accepted;
                        ++rq.drafts_accepted;
                    } else {
                        rq.next_tok = tok;
                        ++accepted;
                        break;
                    }
                }
                if (!done) {
                    if (accepted <= nd) {
                        // Rolled back to anchor + accepted tokens
                        backend.rollback_slot(rq.slot, rq.cur_pos + accepted);
                    }
                    rq.cur_pos += accepted;
                } else if (fin) {
                    finish_req(rq, fin);
                }
                continue;
            }
        }

        // 3. Continuous Batched 48-Visit Step across ALL active slots (B = 1..64)!
        // Coalesces chunked-prefill tokens + decode tokens from every active request into ONE StepBatch!
        batch.clear();
        batch.capture_taps = use_dspark;
        merged_img_embeds.clear();
        logit_slots.clear();
        int token_budget = std::max(chunk_sz, n_active);
        std::vector<int> prefill_fed(P, 0);
        double t_step0 = now_s();

        // First schedule 1 decode token for every decoding slot (zero head-of-line blocking!)
        for (int s = 0; s < P; ++s) {
            auto& rq = reqs[s];
            if (!rq.active || rq.prefill_pos < rq.prompt_len) continue;
            if (rq.cur_pos >= eff_max_ctx) {
                finish_req(rq, "length");
                continue;
            }
            batch.push(s, rq.cur_pos, rq.next_tok, true);
            logit_slots.push_back(s);
            --token_budget;
        }

        // Then fill remaining budget with chunked-prefill tokens across prefilling slots (round-robin!)
        if (token_budget > 0) {
            std::vector<int> pf_slots;
            for (int s = 0; s < P; ++s) if (reqs[s].active && reqs[s].prefill_pos < reqs[s].prompt_len) pf_slots.push_back(s);
            if (!pf_slots.empty()) {
                const int per_slot = std::max(1, token_budget / static_cast<int>(pf_slots.size()));
                for (int s : pf_slots) {
                    auto& rq = reqs[s];
                    const int rem = rq.prompt_len - rq.prefill_pos;
                    int take = std::min(rem, per_slot);
                    // #6627: Pause 1 token before prompt_len so we save the exact prompt-end checkpoint at prompt_len - 1!
                    if (rq.ckpt_pending && rq.prefill_pos < rq.prompt_len - 1 && rq.prefill_pos + take >= rq.prompt_len) {
                        take = (rq.prompt_len - 1) - rq.prefill_pos;
                    }
                    const int img_base = static_cast<int>(merged_img_embeds.size()) / c.hidden;
                    if (!rq.img_embeds.empty()) {
                        merged_img_embeds.insert(merged_img_embeds.end(), rq.img_embeds.begin(), rq.img_embeds.end());
                    }
                    for (int k = 0; k < take; ++k) {
                        const int p_i = rq.prefill_pos + k;
                        const bool is_last = (p_i == rq.prompt_len - 1);
                        int img_idx = (rq.prompt_img_idx[p_i] >= 0) ? (img_base + rq.prompt_img_idx[p_i]) : -1;
                        batch.push(s, p_i, rq.prompt_ids[p_i], is_last, img_idx);
                        if (is_last) logit_slots.push_back(s);
                    }
                    prefill_fed[s] = take;
                }
            }
        }
        if (!merged_img_embeds.empty()) batch.img_embeds = merged_img_embeds.data();

        const int n_want = batch.count_logits();
        logits_buf.resize(static_cast<size_t>(std::max(1, n_want)) * V);
        std::string ferr;
        if (!backend.forward_batch(batch, n_want > 0 ? logits_buf.data() : nullptr, nullptr, ferr)) {
            proto("ERR fatal GPU error: %s", ferr.c_str());
            return 1;
        }
        const double step_ms = (now_s() - t_step0) * 1000.0;

        // Update prefill progress & emit PP lines (+ save prompt-end checkpoint at prompt_len - 1 for #6627!)
        for (int s = 0; s < P; ++s) {
            if (prefill_fed[s] > 0) {
                auto& rq = reqs[s];
                for (int k = 0; k < prefill_fed[s]; ++k) rq.cached_prefix.push_back(rq.prompt_ids[rq.prefill_pos + k]);
                rq.prefill_pos += prefill_fed[s];
                rq.cur_pos = rq.prefill_pos;
                rq.prompt_ms += step_ms;
                if (rq.ckpt_pending && rq.prefill_pos == rq.prompt_len - 1) {
                    rq.ckpt_pending = false;
                    if (backend.save_prompt_checkpoint(rq.slot, rq.prefill_pos)) {
                        rq.prompt_ckpt_prefix.assign(rq.prompt_ids.begin(), rq.prompt_ids.begin() + rq.prefill_pos);
                    }
                }
                if (rq.is_solo || !rq.admitted) {
                    const double ppts = step_ms > 0.0 ? (prefill_fed[s] * 1000.0 / step_ms) : 0.0;
                    proto("PP %d %d %.1f %.1f", rq.prefill_pos, rq.prompt_len, rq.prompt_ms, ppts);
                }
            }
        }

        // Sample next token for every slot that produced logits in this batch!
        for (int wi = 0; wi < n_want; ++wi) {
            const int s = logit_slots[wi];
            auto& rq = reqs[s];
            float* lg = logits_buf.data() + static_cast<size_t>(wi) * V;
            if (!finite_logits(lg, V)) {
                // Purge poisoned KV/ShortConv state so subsequent retries start clean (#879)!
                rq.cached_prefix.clear();
                backend.reset_slot(rq.slot);
                finish_req(rq, "nonfinite");
                continue;
            }
            const bool just_finished_prefill = (prefill_fed[s] > 0);
            if (!just_finished_prefill) {
                rq.cached_prefix.push_back(rq.next_tok);
                ++rq.cur_pos;
            }
            if (rq.max_new == 0) {
                finish_req(rq, "length");
                continue;
            }
            int32_t tok = sample_logits(lg, V, rq.s, rq.hist, rq.rng);
            rq.hist.push_back(tok);
            rq.gen_ids.push_back(tok);
            rq.next_tok = tok;
            ++rq.generated;

            const bool stop = is_stop_tok(tok);
            const bool hit_len = (rq.generated >= rq.max_new) || (rq.cur_pos + 1 >= eff_max_ctx);
            const bool hit_hold = (rq.s.hold_budget > 0 && rq.generated >= rq.s.hold_budget && tok != rq.s.hold_token_id);
            const bool hit_loop = cfg.anti_loop && detect_repetition_loop(rq.gen_ids);

            if (rq.is_solo) {
                if (just_finished_prefill) rq.decode_t0 = now_s();
                proto("T %d", tok);
                if (stop) finish_req(rq, "stop");
                else if (hit_hold) finish_req(rq, "hold");
                else if (hit_len) finish_req(rq, "length");
                else if (hit_loop) finish_req(rq, "stop");
            } else if (rq.is_ext_gen) {
                if (just_finished_prefill) rq.decode_t0 = now_s();
                proto("T %s %d", rq.req_id.c_str(), tok);
                if (stop || hit_loop) finish_req(rq, "stop");
                else if (hit_len) finish_req(rq, "length");
            } else {
                // Native Strata BGEN slot protocol: first token goes on control lines (T + DONE + BADM <slot> <cont>),
                // subsequent tokens stream asynchronously on slot lines (BT <slot> <id> ... BDONE <slot> ...)
                if (!rq.admitted) {
                    rq.decode_t0 = now_s();
                    const bool cont = !(stop || hit_len || hit_hold || hit_loop);
                    if (!rq.slot_read) proto("T %d", tok);
                    const char* fin = stop ? "stop" : hit_hold ? "hold" : hit_len ? "length" : hit_loop ? "stop" : "admit";
                    proto("DONE %d %d %.1f 0.0 %s 0 0 %d", rq.slot_read ? 0 : 1, rq.prompt_len, rq.prompt_ms, fin, rq.reused_tokens);
                    proto("BADM %d %d", rq.slot, cont ? 1 : 0);
                    if (rq.slot_read) proto("BT %d %d", rq.slot, tok);
                    rq.admitted = true;
                    if (!cont) {
                        if (rq.slot_read) proto("BDONE %d %d %s 0.0 read=%.1f", rq.slot, rq.generated, fin, rq.prompt_ms);
                        rq.active = false;
                    }
                } else {
                    if (just_finished_prefill) rq.decode_t0 = now_s();
                    proto("BT %d %d", rq.slot, tok);
                    if (stop) finish_req(rq, "stop");
                    else if (hit_hold) finish_req(rq, "hold");
                    else if (hit_len) finish_req(rq, "length");
                    else if (hit_loop) finish_req(rq, "stop");
                }
            }
        }
    }
    return 0;
}

}  // namespace ink2
