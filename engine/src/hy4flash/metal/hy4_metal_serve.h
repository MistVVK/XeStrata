// hy4_metal_serve.h - the `strata-hy4-metal --serve` front end: Strata's engine line protocol over the Metal Runner (phaseA-tests/hy4/HY4-INTEGRATION-DESIGN.md section 1).
// Port of engine/src/hy4flash/run/hy4_serve.hip.h. Included by hy4_metal_main.cpp after the Runner and the DFly2 / Spec classes.
//
//   stdout, at start:   INFO engine=... arch=hy4flash backend=metal ... drafter=dfly2|none images=0|1    then exactly one   READY <max_context> stop
//   stdin, per request: GEN <max_new> [temperature=F top_p=F top_k=N min_p=F penalty_repeat=F penalty_freq=F penalty_present=F penalty_last_n=N seed=N] [image=<path> ...] id,id,...
//   stdout per request: [RESUME n] (n = longest common prefix of the chosen resident conversation slot's KV and the prompt, capped at prompt-1 and by the SWA ring slack; the slot is the one with the longest prefix, else a free or least-recently-used one)  [IMG i n]  PP <read> <total> <ms> <tok/s>  T <id>  DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <drafts_accepted> <drafts_offered> <reused>
//   SSD tier (--session-reuse 1): a slot evicted with >= --conversation-min-tokens tokens is written to <spill root>/hy4-<fp>[.p<pid>] by a background thread; a later request whose prefix matches an entry better than any resident slot reloads it (RESUME n as usual). INFO: spill=0|1 persist=0|1. With --conversation-spill-persist the folder outlives the process (entries adopted at start, resident slots written at QUIT/EOF).
//   stdin, any time:    STOP (cancel at the next token / chunk boundary)   QUIT or EOF (exit)
// Only protocol lines go to stdout; diagnostics go to stderr.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <mutex>
#include <random>
#include <iostream>
#include "hy4_metal_slots.h"
#include "hy4_metal_spill.h"

#ifndef HY4_METAL_IMAGES
#define HY4_METAL_IMAGES 1  // the Metal ViT (hy4_metal_vision.h, hy4_vision.metal) is linked into this build
#endif

struct ServeOpts {
    std::string model, engram, vision;
    int64_t image_max_pixels = 4194304;  // HY4-VISION
    int max_ctx = 131072;
    int chunk = 2048;
    int index_topk = 2048;
    bool no_index = false, no_engram = false, session_reuse = false;
    int conv_slots = 1, conv_mib = 16384, reuse_min = 1024;  // --conversation-slots / --conversation-cache-mib / --conversation-min-tokens
    std::string spill_root;  // resolved --conversation-spill-dir / default root; "" = SSD tier off
    int spill_mib = 32768;   // --conversation-spill-mib (0 = off)
    bool spill_persist = false;
    int ring_slack = 0;  // extra SWA ring slots (--ring-slack): how far a resume may roll the KV back below the highest position written
    std::string draft = "auto";
    SpecCfg spec;
};

static void proto(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void proto(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fputs(buf, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

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
    bool next(std::string& out) {
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
    bool prefill_only = false;
    std::vector<std::string> images;  // image=<path> keys (percent-decoded), one per picture in prompt order (protocol 1.6b)
};

// pct_decode: hy4::pct_decode of hy4_vision.h (HY4-VISION)
static bool parse_gen(const std::string& line, GenReq& r, std::string& err) {
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
        err = "expected: GEN <max_new> [key=value ...] [image=<path> ...] <ids>";
        return false;
    }
    char* e = nullptr;
    long mn = strtol(w[1].c_str(), &e, 10);
    if (*e || mn < 0) {
        err = "bad max_new";
        return false;
    }
    r.max_new = (int) mn;
    for (size_t k = 2; k + 1 < w.size(); ++k) {
        size_t eq = w[k].find('=');
        if (eq == std::string::npos) {
            err = "bad key " + w[k];
            return false;
        }
        std::string key = w[k].substr(0, eq), val = w[k].substr(eq + 1);
        double d = atof(val.c_str());
        if (key == "temperature") r.s.temperature = (float) d;
        else if (key == "top_p") r.s.top_p = (float) d;
        else if (key == "top_k") r.s.top_k = std::max(1, std::min(64, (int) d));
        else if (key == "min_p") r.s.min_p = (float) d;
        else if (key == "penalty_repeat") r.s.rep = (float) d;
        else if (key == "penalty_freq") r.s.freq = (float) d;
        else if (key == "penalty_present") r.s.pres = (float) d;
        else if (key == "penalty_last_n") r.s.last_n = (int) d;
        else if (key == "seed") r.s.seed = (uint64_t) strtoull(val.c_str(), nullptr, 10);
        else if (key == "prefill") r.prefill_only = d != 0;
        else if (key == "image") r.images.push_back(pct_decode(val));
        else if (key == "pcie_frac" || key == "spec_min_p" || key == "prefill_pipe") continue;
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
        r.ids.push_back((int32_t) v);
        p = (size_t) (ee - ids.c_str());
        if (p < ids.size() && ids[p] == ',') ++p;
    }
    return true;
}

// sampling on the host from the fp32 logits of the last position: penalties, top-k (<= 64), temperature softmax, min_p, top_p, draw
static int sample_host(float* lg, int V, const SampleOpts& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.penalties() && s.last_n > 0) {
        size_t from = hist.size() > (size_t) s.last_n ? hist.size() - (size_t) s.last_n : 0;
        std::unordered_map<int32_t, int> count;
        for (size_t i = from; i < hist.size(); ++i) ++count[hist[i]];
        for (auto& kv : count) {
            if (kv.first < 0 || kv.first >= V) continue;
            float& x = lg[kv.first];
            if (s.rep != 1.f) x = x > 0 ? x / s.rep : x * s.rep;
            x -= (float) kv.second * s.freq + (s.pres != 0.f ? s.pres : 0.f);
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

static bool finite_row(const float* lg, int V) {
    for (int i = 0; i < V; i += 97)
        if (!std::isfinite(lg[i])) return false;
    return true;
}

static int hy4_serve_main(Model& g, mtl::Context& cx, const ServeOpts& o, const std::string& kernel_path) {
    const int max_ctx = o.max_ctx;
    const double t_load0 = now_s();
    Runner r(g, cx, max_ctx, std::max(o.chunk, 1));
    r.kernel_path = kernel_path;
    r.index_topk = o.index_topk;
    r.index_on = o.no_index ? 0 : 1;
    r.ring_slack = o.ring_slack;
    if (o.session_reuse) {
        // Reserve working-set room in cold-expert LRU cache for additional conversation-slot KV sets + SSD spill spare
        r.plan_slots = o.conv_slots;
        r.plan_conv_mib = (size_t) std::max(0, o.conv_mib);
        r.plan_spill = !o.spill_root.empty() && o.spill_mib > 0;
    }
    const bool has_df = g.man.T.count("model.dfly_layers.0.self_attn.q_a_proj.weight") && g.man.T.count("model.dfly_layers.hidden_correction.gate_proj.weight");
    const bool want_spec = o.draft != "off" && has_df && r.Tc >= 16;
    if (o.draft == "dfly2" && !want_spec) {
        proto("ERR --draft dfly2: no drafter in the file or --chunk < 16");
        return 2;
    }
    std::string err;
    if (!r.init(o.no_engram ? std::string() : o.engram, want_spec ? 8 : 1, err)) {
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
    std::unique_ptr<Spec> spec;
    if (use_spec) spec.reset(new Spec(r, df, o.spec));
    fprintf(stderr, "[load] total %.1f s\n", now_s() - t_load0);
    const int V = r.c.vocab, EOS = r.c.eos, EOT = 120008;
    std::vector<int32_t> tok;
    tok.reserve((size_t) max_ctx + 16);  // a stable host buffer: the Engram prefetch thread reads it
    int fwd = 0;                         // positions whose KV is resident (tok[0, fwd) went through forward_chunk); all of tok/fwd/hw below belong to the ACTIVE slot
    // conversation slots: the Runner holds the active slot's KV buffers; switching parks them in the slot and takes the target's (O(1) vector swaps)
    typedef HySlotT<mtl::Buffer> Slot;
    const size_t set_bytes = r.kv_set_bytes;
    int eff_slots = 1;
    if (o.session_reuse) eff_slots = (int) std::max<int64_t>(1, std::min<int64_t>(o.conv_slots, (int64_t) o.conv_mib * 1048576 / (int64_t) std::max<size_t>(set_bytes, 1)));
    std::vector<Slot> slots;
    slots.reserve((size_t) eff_slots);
    slots.emplace_back();
    slots[0].tok.reserve((size_t) max_ctx + 16);
    int act = 0;
    uint64_t use_clock = 0;
    // SSD tier for evicted slots (HySpill): spare KV sets circulate between the slots, the writer thread and a free pool
    HySpill<mtl::Buffer> sp;
    if (o.session_reuse && !o.spill_root.empty() && o.spill_mib > 0) {
        std::vector<char> sw((size_t) r.c.n_layers);
        for (int il = 0; il < r.c.n_layers; ++il) sw[(size_t) il] = r.c.is_swa[il] ? 1 : 0;
        sp.init(o.spill_root, hy4_spill_fp(o.model, o.no_engram ? std::string() : o.engram, max_ctx, r.ring, r.slots_sparse, r.c.n_layers), o.spill_persist, o.spill_mib, sw, (size_t) r.ring * 1152);
    }
    int nsets = 1;  // KV sets alive: the slots' + the free pool's + those held by the writer
    const int max_sets = eff_slots + (sp.on ? 1 : 0);
    auto acquire = [&](HyKvSet<mtl::Buffer>& st) {
        if (sp.on && sp.take_free(st)) return;
        if (nsets < max_sets || !sp.on || !sp.wait_free(st)) {
            r.alloc_kv_set(st.kc, st.iv, st.pl);
            ++nsets;
            fprintf(stderr, "[slots] allocated KV set %d (%.2f GB, cache.allocated %.2f GB)\n", nsets, set_bytes / 1e9, cx.allocated() / 1e9);
        }
    };
    std::vector<float> hl((size_t) V);
    std::string base = o.model;
    size_t sl = base.find_last_of("/\\");
    if (sl != std::string::npos) base = base.substr(sl + 1);
    // HY4-VISION: pictures need the Vision shard; probe it now (cheap, no GPU) so INFO tells serve the truth, and load it on the first picture request (not at start)
    bool images_ok = false;
    hy4::VitMetal venc;
    hy4::VisionCtxM vctx;
    if (HY4_METAL_IMAGES && !o.vision.empty()) {
        std::string verr;
        hy4::VisionShardInfo vprobe;
        if (hy4::vision_probe(o.vision, vprobe, verr)) images_ok = true;
        else fprintf(stderr, "[vision] disabled: %s\n", verr.c_str());
    }
    proto("INFO engine=hy4-metal-serve-0 arch=hy4flash backend=metal model=%s layers=%d vocab=%d chunk=%d session_reuse=%d drafter=%s images=%d ring_slack=%d slots=%d slot_mib=%.0f spill=%d persist=%d", base.c_str(), r.c.n_layers, V, r.Tc,
          o.session_reuse ? 1 : 0, use_spec ? "dfly2" : "none", images_ok ? 1 : 0, r.ring_slack, eff_slots, set_bytes / 1048576.0, sp.on ? 1 : 0, sp.on && sp.persist ? 1 : 0);
    proto("READY %d stop", max_ctx);

    int hw = 0;
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
        // HY4-VISION (protocol 1.6b): image=<path> keys, one per image-pad token of the ids. Decode + resize the pictures (CPU), expand each pad to its run, tell serve with IMG lines.
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
        } else if (std::find(q.ids.begin(), q.ids.end(), (int32_t) hy4::VitCfg().image_token) != q.ids.end()) {
            proto("ERR image: 0 files for %d image placeholders", (int) std::count(q.ids.begin(), q.ids.end(), (int32_t) hy4::VitCfg().image_token));
            continue;
        }
        const int T0 = (int) q.ids.size();
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
        // HY4-VISION: encode the pictures now (loads the shard on first use); the runner hooks fill the prompt's image rows. No KV reuse across picture requests.
        if (has_images) {
            std::string ierr;
            if (!venc.load(cx, o.vision, kernel_path.substr(0, kernel_path.rfind('/') + 1), ierr) || !vctx.setup(venc, cx, q.ids, recs, ierr)) {
                proto("ERR image: %s", ierr.c_str());
                continue;
            }
            fprintf(stderr, "[vision] %zu picture(s), %d image rows, encoded in %.2f s\n", recs.size(), vctx.n_rows, vctx.t_encode);
            r.vis = &vctx;
        }
        // pick the conversation slot (resume point, free, new or least recently used), then make it the active one
        slots[act].fwd = fwd;
        slots[act].hw = hw;
        std::vector<HyPickIn> pin;
        for (size_t i = 0; i < slots.size(); ++i) pin.push_back({(int) i == act ? &tok : &slots[i].tok, slots[i].fwd, slots[i].hw, slots[i].last_used});
        int start = 0;
        const bool no_reuse = has_images || !o.session_reuse;
        int pk = hy4_pick_slot(pin, act, q.ids, T0, r.ring - 512, eff_slots, o.reuse_min, no_reuse, start);
        // SSD tier: an entry that shares a longer prefix than any resident slot is reloaded into the LRU / free slot
        HySpill<mtl::Buffer>::Cand sc;
        bool from_ssd = false;
        if (sp.on && !no_reuse && sp.best(q.ids, T0, r.ring - 512, sc) && sc.start >= o.reuse_min && sc.start > start) {
            int dummy = 0;
            pk = hy4_pick_slot(pin, act, q.ids, T0, r.ring - 512, eff_slots, o.reuse_min, true, dummy);
            from_ssd = true;
            start = 0;
        }
        // no prefetch of the old tok may still be in flight or pending: drop them before tok (or the KV set) changes
        for (int s = 0; s < 2; ++s) {
            if (r.ep[s].fut.valid()) r.ep[s].fut.wait();
            r.ep[s].pending = false;
        }
        if (pk >= (int) slots.size()) {
            slots.emplace_back();
            slots.back().tok.reserve((size_t) max_ctx + 16);
            HyKvSet<mtl::Buffer> ns;
            acquire(ns);
            slots.back().kcache = std::move(ns.kc);
            slots.back().ivg = std::move(ns.iv);
            slots.back().pooled = std::move(ns.pl);
        }
        if (pk != act) {
            std::swap(r.kcache, slots[act].kcache);
            std::swap(r.ivg, slots[act].ivg);
            std::swap(r.pooled, slots[act].pooled);
            std::swap(tok, slots[act].tok);
            std::swap(r.kcache, slots[pk].kcache);
            std::swap(r.ivg, slots[pk].ivg);
            std::swap(r.pooled, slots[pk].pooled);
            std::swap(tok, slots[pk].tok);
            act = pk;
            fwd = slots[act].fwd;
            hw = slots[act].hw;
        }
        // the active slot is now the victim (or the resume target): a victim with a conversation worth keeping goes to the writer, the Runner gets another set
        if (sp.on && start == 0 && fwd >= o.reuse_min && fwd <= (int) tok.size()) {
            HyKvSet<mtl::Buffer> vs;
            vs.kc = std::move(r.kcache);
            vs.iv = std::move(r.ivg);
            vs.pl = std::move(r.pooled);
            if (sp.submit(vs, tok, fwd, hw)) {
                HyKvSet<mtl::Buffer> ns;
                acquire(ns);
                r.kcache = std::move(ns.kc);
                r.ivg = std::move(ns.iv);
                r.pooled = std::move(ns.pl);
            } else {
                r.kcache = std::move(vs.kc);
                r.ivg = std::move(vs.iv);
                r.pooled = std::move(vs.pl);
            }
            fwd = 0;
            hw = 0;
        }
        if (from_ssd) {
            HyKvSet<mtl::Buffer> ds;
            ds.kc = std::move(r.kcache);
            ds.iv = std::move(r.ivg);
            ds.pl = std::move(r.pooled);
            int lf = 0, lh = 0;
            std::vector<int32_t> lt;
            const bool ok = sp.load(sc.seq, ds, lt, lf, lh);
            r.kcache = std::move(ds.kc);
            r.ivg = std::move(ds.iv);
            r.pooled = std::move(ds.pl);
            if (ok) {
                // adopt the entry as the slot's resident conversation; its own hy4_reuse_start (same inputs as the match) gives the resume point
                tok.assign(lt.begin(), lt.end());
                fwd = lf;
                hw = lh;
                start = hy4_reuse_start(tok, fwd, q.ids, T0, r.ring - 512 - std::max(0, hw - fwd));
            } else {
                start = 0;
                fwd = 0;
                hw = 0;
            }
        }
        slots[act].last_used = ++use_clock;
        if (start == 0) hw = 0;
        tok.assign(q.ids.begin(), q.ids.end());
        fwd = 0;
        if (start > 0) proto("RESUME %d", start);
        std::mt19937_64 rng(q.s.seed ? q.s.seed : std::random_device{}());
        const double t0 = t_req;  // the prompt time of a picture request includes the picture encoding
        int done = start;
        bool cancelled = false, failed = false;
        r.prefetch_engram(tok.data(), done, std::min(r.Tc, T0 - done));
        while (done < T0) {
            int T = std::min(r.Tc, T0 - done);
            int np0 = done + T < T0 ? done + T : -1;
            int nT = np0 >= 0 ? std::min(r.Tc, T0 - np0) : 0;
            if (!r.forward_chunk(tok.data(), done, T, err, np0, nT)) {
                failed = true;
                break;
            }
            done += T;
            const bool last = done == T0;
            if (last) r.head(T, T - 1);
            r.run_all();
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
        long acc0 = 0, off0 = 0;
        const bool greedy_fast = q.s.greedy() && !q.s.penalties();
        const bool spec_path = use_spec && limit > 0 && !has_images;  // Tree-8 + lossless SpecInfer rejection sampling for both greedy and temp>0/top_p/top_k/penalties!
        if (spec_path) {
            std::string fin, e2;
            const long a_before = spec->S.accepted, o_before = spec->S.verified_slots;
            fwd = spec->run(tok, limit, max_ctx, EOS, EOT, [&](int t) { proto("T %d", t); return !in.stop.load(); }, fin, e2, made, greedy_fast ? nullptr : &q.s, greedy_fast ? nullptr : &rng);
            if (fin == "error") {
                proto("ERR spec: %s", e2.c_str());
                return 1;
            }
            finish = fin == "stop" ? "stop" : fin == "cancel" ? "cancel" : "length";
            acc0 = spec->S.accepted - a_before;
            off0 = spec->S.verified_slots - o_before;
            spec->report(stderr, (now_s() - t1) * 1000.0);
        }
        while (!spec_path && made < limit) {
            int next = 0;
            if (greedy_fast) {
                r.argmax(V);
                r.run_all();
                next = *(const int32_t*) r.d_argmax.ptr;
            } else {
                std::memcpy(hl.data(), r.logits.ptr, (size_t) V * 4);
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
            int pos = (int) tok.size() - 1;
            if (!r.forward_chunk(tok.data(), pos, 1, err)) {
                failed = true;
                break;
            }
            fwd = pos + 1;
            r.head(1, 0);
            r.run_all();
        }
        if (failed) {
            proto("ERR forward: %s", err.c_str());
            return 1;
        }
        if (nonfinite) finish = "nonfinite";
        if (!o.session_reuse || has_images) fwd = 0;
        hw = std::max(hw, std::max(fwd, (int) tok.size()));
        r.vis = nullptr;
        const double decode_ms = (now_s() - t1) * 1000.0;
        proto("DONE %d %d %.1f %.1f %s %ld %ld %d", made, T0, prompt_ms, decode_ms, finish, acc0, off0, start);
    }
    // QUIT / EOF: with --conversation-spill-persist every resident conversation worth keeping that is not on the SSD yet is written now
    if (sp.on && sp.persist) {
        slots[act].fwd = fwd;
        slots[act].hw = hw;
        sp.drain();
        for (size_t i = 0; i < slots.size(); ++i) {
            const std::vector<int32_t>& t = (int) i == act ? tok : slots[i].tok;
            const int f = std::min(slots[i].fwd, (int) t.size());
            if (f < o.reuse_min) continue;
            std::vector<mtl::Buffer>& kc = (int) i == act ? r.kcache : slots[i].kcache;
            std::vector<mtl::Buffer>& iv = (int) i == act ? r.ivg : slots[i].ivg;
            std::vector<mtl::Buffer>& pl = (int) i == act ? r.pooled : slots[i].pooled;
            HyKvSet<mtl::Buffer> ss;
            ss.kc = std::move(kc);
            ss.iv = std::move(iv);
            ss.pl = std::move(pl);
            sp.write_now(ss, t, f, slots[i].hw);
            kc = std::move(ss.kc);
            iv = std::move(ss.iv);
            pl = std::move(ss.pl);
        }
    }
    sp.shutdown();
    return 0;
}
