// sy_serve.h - the `strata-hy4 --serve` front end of the Intel Arc (SYCL) runner: Strata's engine line protocol over Runner.
// SYCL port of the HY4-STRIX hy4_serve.hip.h (f-hy4-dfly2 a70c0e8fc); protocol: phaseA-tests/hy4/HY4-INTEGRATION-DESIGN.md section 1.
//
//   stdout, at start:        INFO engine=... arch=hy4flash ... drafter=dfly2|off draft_window=N   then exactly one   READY <max_context> stop
//   stdin, per request:      GEN <max_new> [temperature=F top_p=F top_k=N min_p=F penalty_repeat=F penalty_freq=F penalty_present=F penalty_last_n=N seed=N] <id>,<id>,...
//   stdout, per request:     [RESUME n]  PP <read> <total> <ms> <tok/s> (per prefill chunk)  T <id> (per token)  DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <acc> <off> <reused>
//   stdin, any time:         STOP (end the running request at the next token / chunk boundary: DONE ... cancel)   QUIT (or EOF: free the GPU and exit)
// Only protocol lines go to stdout; everything else goes to stderr. Each GEN reads its whole prompt (no session kept) unless --session-reuse 1: then a prompt that extends the tokens whose
// KV is resident exactly reads only the rest and says RESUME n. Images are not read by this engine yet (`image=` keys are answered with ERR image: ...; --info says images=0).
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>

#include "sy_spec.h"

namespace hy4s {

struct ServeOpts {
    std::string model, engram;
    int max_ctx = 131072;
    int chunk = 1024;
    int index_topk = 2048;
    int threads = 6;
    int gx_mb = 384;
    int kv_host_above = 24000;
    double vram_experts_gb = -1.0;  // < 0: auto from free VRAM
    double ram_experts_gb = -1.0;   // < 0: auto from RAM budget
    bool hi_act = false;
    bool no_index = false, no_engram = false;
    bool session_reuse = false;
    std::string draft = "auto";
    std::string vision;
    int64_t image_max_pixels = 1048576;
    bool vision_requested = false;
    int primary_gpu = 0;
    std::vector<int> peer_devs;
    SpecCfg spec;
};

#ifndef _MSC_VER
inline void proto(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#endif
inline void proto(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fputs(buf, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

// ---- stdin on its own thread: STOP is seen while a request runs
class LineInput {
public:
    std::atomic<bool> stop{false};
    void start() {
        th_ = std::thread([this] {
            std::string line;
            while (std::getline(std::cin, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                if (line == "STOP") {
                    stop.store(true);
                    continue;
                }
                std::lock_guard<std::mutex> lk(m_);
                q_.push_back(line);
                cv_.notify_one();
            }
            std::lock_guard<std::mutex> lk(m_);
            eof_ = true;
            cv_.notify_all();
        });
        th_.detach();
    }
    bool next(std::string& out) {  // false at EOF with nothing left
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return !q_.empty() || eof_; });
        if (q_.empty()) return false;
        out = q_.front();
        q_.pop_front();
        return true;
    }

private:
    std::thread th_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    bool eof_ = false;
};

struct GenReq {
    int max_new = 0;
    SampleOpts s;
    std::vector<int32_t> ids;
    std::vector<std::string> images;
    bool prefill_only = false;
};

inline bool parse_gen(const std::string& line, GenReq& r, std::string& err) {
    std::vector<std::string> w;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && line[i] == ' ') ++i;
        size_t j = i;
        while (j < line.size() && line[j] != ' ') ++j;
        if (j > i) w.push_back(line.substr(i, j - i));
        i = j;
    }
    if (w.size() < 3 || w[0] != "GEN") {
        err = "expected: GEN <max_new> [key=value ...] <ids>";
        return false;
    }
    char* e = nullptr;
    long mn = strtol(w[1].c_str(), &e, 10);
    if (*e || mn < 0) {
        err = "bad max_new";
        return false;
    }
    r.max_new = (int)mn;
    for (size_t k = 2; k + 1 < w.size(); ++k) {
        size_t eq = w[k].find('=');
        if (eq == std::string::npos) {
            err = "bad key " + w[k];
            return false;
        }
        std::string key = w[k].substr(0, eq), val = w[k].substr(eq + 1);
        double d = atof(val.c_str());
        if (key == "temperature") r.s.temperature = (float)d;
        else if (key == "top_p") r.s.top_p = (float)d;
        else if (key == "top_k") r.s.top_k = std::max(1, std::min(64, (int)d));
        else if (key == "min_p") r.s.min_p = (float)d;
        else if (key == "penalty_repeat") r.s.rep = (float)d;
        else if (key == "penalty_freq") r.s.freq = (float)d;
        else if (key == "penalty_present") r.s.pres = (float)d;
        else if (key == "penalty_last_n") r.s.last_n = (int)d;
        else if (key == "seed") r.s.seed = (uint64_t)strtoull(val.c_str(), nullptr, 10);
        else if (key == "prefill") r.prefill_only = d != 0;
        else if (key == "image") r.images.push_back(hy4::pct_decode(val));
        else if (key == "pcie_frac" || key == "spec_min_p" || key == "prefill_pipe") continue;  // tuning keys of the Strata engine: not ours
        else {
            err = "unsupported key " + key;
            return false;
        }
    }
    const std::string& ids = w.back();
    size_t p = 0;
    while (p < ids.size()) {
        char* ee = nullptr;
        long v = strtol(ids.c_str() + p, &ee, 10);
        if (ee == ids.c_str() + p) {
            err = "bad token id list";
            return false;
        }
        r.ids.push_back((int32_t)v);
        p = (size_t)(ee - ids.c_str());
        if (p < ids.size() && ids[p] == ',') ++p;
    }
    return true;
}

// sampling on the host from the fp32 logits of the last position: penalties, top-k candidates (<= 64), temperature softmax, min_p, top_p, draw
inline int sample_host(float* lg, int V, const SampleOpts& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.penalties() && s.last_n > 0) {
        size_t from = hist.size() > (size_t)s.last_n ? hist.size() - (size_t)s.last_n : 0;
        std::unordered_map<int32_t, int> count;
        for (size_t i = from; i < hist.size(); ++i) ++count[hist[i]];
        for (auto& kv : count) {
            if (kv.first < 0 || kv.first >= V) continue;
            float& x = lg[kv.first];
            if (s.rep != 1.f) x = x > 0 ? x / s.rep : x * s.rep;
            x -= (float)kv.second * s.freq + (s.pres != 0.f ? s.pres : 0.f);
        }
    }
    int K = std::min(std::max(1, s.top_k), 64);
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    if (s.greedy()) return idx[0];
    std::vector<double> p(K);
    double mx = lg[idx[0]], sum = 0;
    for (int i = 0; i < K; ++i) {
        p[i] = std::exp((lg[idx[i]] - mx) / s.temperature);
        sum += p[i];
    }
    for (auto& x : p) x /= sum;
    int keep = K;
    if (s.min_p > 0.f) {
        double thr = s.min_p * p[0];
        keep = 1;
        while (keep < K && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.f) {
        double c = 0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= s.top_p) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0;
    for (int i = 0; i < keep; ++i) z += p[i];
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng), c = 0;
    for (int i = 0; i < keep; ++i) {
        c += p[i];
        if (r <= c) return idx[i];
    }
    return idx[keep - 1];
}

inline bool finite_row(const float* lg, int V) {
    for (int i = 0; i < V; i += 97)
        if (!std::isfinite(lg[i])) return false;
    return true;
}

inline bool manifest_has_drafter(const hy4::Manifest& m) {
    return m.T.count("model.dfly_layers.0.self_attn.q_a_proj.weight") && m.T.count("model.dfly_layers.hidden_correction.gate_proj.weight");
}

// ---- the loop
inline int hy4s_serve_main(const ServeOpts& o) {
    try {
        auto dev = Q().get_device();
        fprintf(stderr, "[gpu] %s, %.2f GiB (runtime: %s)\n", dev.get_info<sycl::info::device::name>().c_str(), dev.get_info<sycl::info::device::global_mem_size>() / 1073741824.0,
                dev.get_platform().get_info<sycl::info::platform::name>().c_str());
        SyModel g;
        std::string err;
        const double t_load0 = now_s();
        if (!g.load(o.model, err)) {
            proto("ERR cannot load %s: %s", o.model.c_str(), err.c_str());
            return 1;
        }
        const bool has_drafter = manifest_has_drafter(g.man);
        if (o.draft == "dfly2" && !has_drafter) {
            proto("ERR --draft dfly2: this .strata has no DFly2 drafter tensors");
            return 2;
        }
        const bool want_spec = o.draft != "off" && has_drafter && o.chunk >= 8;
        if (o.draft == "dfly2" && o.chunk < 8) {
            proto("ERR --draft dfly2 needs --chunk >= 8");
            return 2;
        }
        const int max_ctx = o.max_ctx;
        hy4::CpuExperts ucx;
        SyVramExpertCache vcache;
        SyPeerExperts peers;
        Runner r(g, ucx, max_ctx, std::max(o.chunk, 1), want_spec ? 8 : 1);
        r.index_topk = o.index_topk;
        r.index_on = o.no_index ? 0 : 1;
        r.kv_host_above = o.kv_host_above;
        r.expert_act = 1;
        if (!r.init(o.no_engram ? std::string() : o.engram, err)) {
            proto("ERR init: %s", err.c_str());
            return 1;
        }
        DFly2 df(r);
        bool use_spec = false;
        if (want_spec) {
            if (df.init(err)) use_spec = true;
            else if (o.draft == "dfly2") {
                proto("ERR --draft dfly2: %s", err.c_str());
                return 2;
            }
        }
        // Size VRAM expert cache + unified CPU expert pool AFTER trunk/KV/DFly2 are allocated!
        const size_t gx_reserve = o.gx_mb > 0 ? ((size_t)o.gx_mb * 2 / 3 + 220) * (1ull << 20) : 0;
        const size_t host_oh = r.kv_host_bytes + g.host_bytes_total + ((size_t)std::max(o.gx_mb, 0) << 20) + (640ull << 20);
        if (!setup_experts_unified(o.model, o.threads, o.vram_experts_gb, o.ram_experts_gb, gx_reserve, host_oh, o.hi_act, vcache, ucx, err, o.peer_devs, &peers)) {
            proto("ERR experts: %s", err.c_str());
            return 1;
        }
        r.vcache = vcache.active() ? &vcache : nullptr;
        r.peers = peers.active() ? &peers : nullptr;
        GpuExperts gxe;
        if (o.gx_mb > 0) {
            std::string gerr;
            try {
                if (gxe.init_unified(o.model, &ucx, r.vcache, (size_t)o.gx_mb << 20, r.Tc, gerr, o.threads)) {
                    r.gx = &gxe;
                    r.gx_min_T = 32;
                } else {
                    fprintf(stderr, "[gpu-experts] disabled: %s\n", gerr.c_str());
                }
            } catch (const sycl::exception& e) {
                fprintf(stderr, "[gpu-experts] disabled: %s\n", e.what());
            }
        }
        std::unique_ptr<Spec> spec;
        if (use_spec) {
            SpecCfg sc = o.spec;
            spec.reset(new Spec(r, df, sc));
            spec_set_batch_invariant();
        }
        fprintf(stderr, "[load] total %.1f s, GPU%d allocated %.2f GiB of %.2f GiB%s%s\n", now_s() - t_load0,
                o.primary_gpu,
                (dev_bytes_allocated() + (r.gx ? gxe.device_bytes() : 0)) / 1073741824.0, dev_caps().global_mem / 1073741824.0,
                peers.active() ? ", peers: " : "", peers.active() ? peers.describe().c_str() : "");
        // pictures need the Vision shard: probe it now (cheap, no GPU) so INFO tells serve the truth; it is read into pinned host memory on the first picture request, not at start
        bool images_ok = false;
        VisionEncoder venc;
        VisionCtx vctx;
        if (!o.vision.empty()) {
            std::string verr;
            hy4::VisionShardInfo vprobe;
            if (hy4::vision_probe(o.vision, vprobe, verr)) images_ok = true;
            else fprintf(stderr, "[vision] disabled: %s\n", verr.c_str());
        }
        const int V = r.c.vocab, EOS = r.c.eos, EOT = 120008;
        std::vector<int32_t> tok;
        tok.reserve((size_t)max_ctx + 16);  // a stable host buffer: the Engram prefetch thread reads it
        int fwd = 0;                        // positions whose KV is resident (tok[0, fwd) went through forward_chunk)
        std::vector<float> hl((size_t)V);
        std::string base = o.model;
        size_t sl = base.find_last_of("/\\");
        if (sl != std::string::npos) base = base.substr(sl + 1);
        proto("INFO engine=hy4-serve-sycl-0 arch=hy4flash backend=sycl model=%s layers=%d vocab=%d chunk=%d session_reuse=%d drafter=%s draft_window=%d images=%d", base.c_str(), r.c.n_layers, V,
              r.Tc, o.session_reuse ? 1 : 0, use_spec ? "dfly2" : "off", use_spec ? o.spec.K : 0, images_ok ? 1 : 0);
        proto("READY %d stop", max_ctx);

        LineInput in;
        in.start();
        std::string line;
        while (in.next(line)) {
            if (line == "QUIT") break;
            if (line.empty()) continue;
            GenReq q;
            std::string perr;
            if (!parse_gen(line, q, perr)) {
                proto("ERR %s", perr.c_str());
                continue;
            }
            in.stop.store(false);
            // pictures (protocol 1.6b): image=<path> keys, one per image-pad token of the ids. Decode + resize the pictures (CPU), expand each pad to its run, tell serve with IMG lines.
            r.vis = nullptr;
            vctx.release();
            const bool has_images = !q.images.empty();
            const double t_req = now_s();
            std::vector<hy4::ImageRec> recs;
            if (has_images) {
                std::string ierr;
                if (!images_ok) {
                    proto("ERR image: this engine was started without the Vision shard (--vision Hy4-Flash-Vision.strata)");
                    continue;
                }
                bool bad = false;
                for (const std::string& f : q.images) {
                    hy4::ImageRec rc;
                    if (!hy4::prepare_picture(f, hy4::VitCfg().min_pixels, o.image_max_pixels, rc, ierr)) {
                        bad = true;
                        break;
                    }
                    recs.push_back(std::move(rc));
                }
                std::vector<int32_t> ex;
                if (bad || !hy4::expand_image_pads(q.ids, hy4::VitCfg().image_token, recs, ex, ierr)) {
                    proto("ERR image: %s", ierr.c_str());
                    continue;
                }
                q.ids.swap(ex);
                for (size_t i = 0; i < recs.size(); ++i) proto("IMG %zu %d", i + 1, hy4::image_token_count(recs[i].h, recs[i].w));
            } else if (std::find(q.ids.begin(), q.ids.end(), (int32_t)hy4::VitCfg().image_token) != q.ids.end()) {
                proto("ERR image: 0 files for %d image placeholders", (int)std::count(q.ids.begin(), q.ids.end(), (int32_t)hy4::VitCfg().image_token));
                continue;
            }
            const int T0 = (int)q.ids.size();
            if (T0 == 0) {
                proto("ERR empty prompt");
                continue;
            }
            if (T0 >= max_ctx) {
                proto("ERR prompt exceeds context (%d of %d)", T0, max_ctx);
                continue;
            }
            if (q.prefill_only) {
                proto("ERR prefill=1 is not supported");
                continue;
            }
            // encode the pictures now (reads the shard on first use); the runner hooks fill the prompt's image rows. No KV reuse across picture requests.
            if (has_images) {
                std::string ierr;
                venc.cancel = &in.stop;
                if (!venc.load(o.vision, ierr) || !vctx.setup(venc, q.ids, recs, r.Tc, ierr)) {
                    if (in.stop.load()) {
                        proto("DONE 0 %d %.1f 0.0 cancel", T0, (now_s() - t_req) * 1000.0);
                        continue;
                    }
                    proto("ERR image: %s", ierr.c_str());
                    continue;
                }
                fprintf(stderr, "[vision] %zu picture(s), %d image rows, encoded in %.2f s\n", recs.size(), vctx.n_rows, vctx.t_encode);
                r.vis = &vctx;
            }
            // KV reuse: pure extension OR LCP prefix rollback (4-token IndexPool-aligned, within SWA ring safety bound)
            int start = 0;
            if (!has_images && o.session_reuse && fwd > 0 && (int)tok.size() >= fwd) {
                int lcp = 0;
                const int lim = std::min(fwd, T0);
                while (lcp < lim && tok[lcp] == q.ids[lcp]) ++lcp;
                if (lcp == fwd && fwd < T0) {
                    start = fwd;  // exact pure extension: zero wasted tokens
                } else if (lcp >= 4) {
                    const int s = std::min(lcp, T0 - 1) & ~3;  // align to 4-token IndexPool group so ivg rebuilds cleanly
                    if (s > 0 && std::max(0, s - r.c.window + 1) >= std::max(0, fwd - r.ring)) start = s;
                }
            }
            tok.assign(q.ids.begin(), q.ids.end());
            fwd = 0;
            if (start > 0) proto("RESUME %d", start);
            std::mt19937_64 rng(q.s.seed ? q.s.seed : std::random_device{}());
            const double t0 = has_images ? t_req : now_s();  // the prompt time of a picture request includes the picture encoding
            int done = start;
            bool cancelled = false, failed = false;
            r.prefetch_engram(tok.data(), done, std::min(r.Tc, T0 - done));
            while (done < T0) {
                const int T = std::min(r.Tc, T0 - done);
                if (done + T < T0) r.prefetch_engram(tok.data(), done + T, std::min(r.Tc, T0 - (done + T)));
                if (!r.forward_chunk(tok.data(), done, T, err)) {
                    failed = true;
                    break;
                }
                done += T;
                const bool last = done == T0;
                if (last) {
                    r.head(T, T - 1);
                    if (use_spec) df.snap_h(0);
                }
                wait();
                const double ms = (now_s() - t0) * 1000.0;
                proto("PP %d %d %.0f %.1f", done, T0, ms, (done - start) / std::max(1e-9, ms / 1000.0));
                fwd = done;
                if (!last && in.stop.load()) {
                    cancelled = true;
                    break;
                }
            }
            if (failed) {
                proto("ERR forward: %s", err.c_str());
                return 1;
            }
            const double prompt_ms = (now_s() - t0) * 1000.0;
            int made = 0;
            const char* finish = "length";
            const double t1 = now_s();
            if (cancelled) {
                proto("DONE 0 %d %.1f 0.0 cancel", T0, prompt_ms);
                fwd = 0;
                continue;
            }
            const int room = max_ctx - T0 - 1;
            const int limit = std::min(q.max_new, std::max(0, room));
            bool nonfinite = false;
            const bool greedy_fast = q.s.greedy() && !q.s.penalties();
            const bool spec_path = use_spec && limit > 0 && !has_images;  // lossless for both greedy AND probabilistic (temperature/penalties)!
            long acc0 = spec ? spec->S.accepted : 0, off0 = spec ? spec->S.drafted : 0;
            if (spec_path) {
                std::string fin, e2;
                fwd = spec->run(tok, limit, max_ctx, EOS, EOT, [&](int t) { proto("T %d", t); return !in.stop.load(); }, fin, e2, made, &q.s, &rng);
                if (fin == "error") {
                    proto("ERR spec: %s", e2.c_str());
                    return 1;
                }
                finish = fin == "stop" ? "stop" : fin == "cancel" ? "cancel" : "length";
            }
            while (!spec_path && made < limit) {
                int next = 0;
                if (greedy_fast) {
                    argmax_row(r.logits, V, r.d_argmax);
                    int32_t nx = 0;
                    d2h(&nx, r.d_argmax, 4);
                    wait();
                    next = nx;
                } else {
                    d2h(hl.data(), r.logits, (size_t)V * 4);
                    wait();
                    if (!finite_row(hl.data(), V)) {
                        nonfinite = true;
                        break;
                    }
                    next = sample_host(hl.data(), V, q.s, tok, rng);
                }
                tok.push_back(next);
                ++made;
                proto("T %d", next);
                if (next == EOS || next == EOT) {
                    finish = "stop";
                    break;
                }
                if (in.stop.load()) {
                    finish = "cancel";
                    break;
                }
                if (made >= limit) break;
                const int pos = (int)tok.size() - 1;
                if (!r.forward_chunk(tok.data(), pos, 1, err)) {
                    failed = true;
                    break;
                }
                fwd = pos + 1;
                r.head(1, 0);
                wait();
                r.step_adapt();
            }
            if (failed) {
                proto("ERR forward: %s", err.c_str());
                return 1;
            }
            if (nonfinite) finish = "nonfinite";
            if (!o.session_reuse || has_images) fwd = 0;
            r.vis = nullptr;
            const double decode_ms = (now_s() - t1) * 1000.0;
            proto("DONE %d %d %.1f %.1f %s %ld %ld %d", made, T0, prompt_ms, decode_ms, finish, spec ? spec->S.accepted - acc0 : 0L, spec ? spec->S.drafted - off0 : 0L, start);
        }
        wait();
        // QUIT / EOF: leave without running the destructors. After some request mixes (cancel, errors) the teardown of the async Engram tasks / runtime threads did not return in
        // tests on p1006 (main thread in a futex wait for >60 s); everything the process holds (device memory, pinned memory, mmaps) is released by the OS.
        hy4s_exit_fast(0);
    } catch (const sycl::exception& e) {
        proto("ERR sycl: %s", e.what());
        return 1;
    }
}

}  // namespace hy4s
