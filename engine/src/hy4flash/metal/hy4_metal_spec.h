// hy4_metal_spec.h - DFly2 drafter + lossless greedy speculative loop on the Metal Runner (hy4_metal_main.cpp). Included by it after the Runner class.
// Port of hy4_dfly2.hip.h / hy4_spec.hip.h (branch f-hy4-dfly2); kernels in hy4_dfly2.metal.
//
// One draft step (target processed positions 0..a-1, anchor x_a emitted but not processed, h_{a-1} = row `hrow` of the Runner's hfin):
//   block = [x_a, 120009 x K] at positions a..a+K ; slot 0: H[0] += gate * eh_proj([enorm(E(x_a)) ; hnorm(h_{a-1})])
//   5 layers (draft layer d reads the cached latents of target layer 25+4d: rows < a, or its top-2048 picks at a-1, plus the block's own latents, both through the draft layer's own kv_b_proj)
//   final_norm, then j = 1..K: hc_in = [hidden_norm(Hf[j]) ; embed_norm(E(prev))], h_corr = Hf[j] + down(silu(gate(hc_in)) * up(hc_in)), y_j = argmax lm_head(h_corr), conf_j = sigmoid(conf_head(h_corr)).
// The own block rows are written into the TARGET layer's cache slots a..a+K: the verify chunk rewrites a..a+K' before it reads them.
#pragma once
#include <deque>
#include <functional>

constexpr int DFLY_MASK_ID = 120009;
constexpr int DFLY_KMAX = 7;
constexpr int DFLY_BMAX = DFLY_KMAX + 1;
constexpr int DFLY_LAYERS[5] = {25, 29, 33, 37, 41};

struct DFly2 {
    Runner& r;
    const Cfg& c;
    bool ok = false;
    Buffer H, Hf, F1, F2, F3, F4, Ec, Dn, G, U, d_conf, hprev_f, Hf_h, hn, qr_h, qa_h, ctx_h, o_h, mlp_h, hcin, act, hc_h, d_bids, d_chain, dlist, dnk, wkd, wvd;
    size_t wk_stride = 0, wv_stride = 0;
    int lstride = 0;
    double t_draft = 0;
    long n_draft = 0;

    explicit DFly2(Runner& rr) : r(rr), c(rr.c) {}
    Arg wf(const std::string& n) const { return r.g.tw(n); }

    bool init(std::string& err) {
        if (!r.g.man.T.count("model.dfly_layers.0.self_attn.q_a_proj.weight")) {
            err = "this .strata has no DFly2 drafter tensors";
            return false;
        }
        r.dfly_capture_alloc();
        auto A = [&](size_t b) { return r.cx.alloc(b); };
        const size_t B = DFLY_BMAX;
        H = A(B * 3072 * 4);
        Hf = A(B * 3072 * 4);
        F1 = A(B * 16384 * 4);
        F2 = A(B * 16384 * 4);
        F3 = A(B * 16384 * 4);
        F4 = A(B * 16384 * 4);
        Ec = A(3072 * 4);
        Dn = A(3072 * 4);
        G = A(3072 * 4);
        U = A(3072 * 4);
        d_conf = A(64);
        hprev_f = A(3072 * 4);
        Hf_h = A(B * 3072 * 2);
        hn = A(B * 3072 * 2);
        qr_h = A(B * 1024 * 2);
        qa_h = A(B * 32 * 576 * 2);
        ctx_h = A(B * 32 * 512 * 2);
        o_h = A(B * 8192 * 2);
        mlp_h = A(B * 9216 * 2);
        hcin = A(2 * 3072 * 2);
        act = A(3072 * 2);
        hc_h = A(3072 * 2);
        d_bids = A(64);
        d_chain = A(64);
        lstride = r.index_topk + 8;
        dlist = A((size_t) B * lstride * 4);
        dnk = A(64);
        // absorbed kv_b_proj of the 5 draft layers (fp16 wk [32][192][512], wv [32][256][512])
        wk_stride = (size_t) 32 * 192 * 512 * 2;
        wv_stride = (size_t) 32 * 256 * 512 * 2;
        wkd = A(5 * wk_stride);
        wvd = A(5 * wv_stride);
        for (int d = 0; d < 5; ++d) {
            const std::string nm = "model.dfly_layers." + std::to_string(d) + ".self_attn.kv_b_proj.weight";
            const TInfo& t = r.g.ti(nm);
            if (t.fmt != F_S4 && t.fmt != F_S5 && t.fmt != F_S6 && t.fmt != F_S8) { err = "dfly kv_b_proj format not supported"; return false; }
            r.add("split_kvb_" + std::to_string(t.fmt), 32 * 448, 1, 128, {r.g.tw(nm), Arg::B(wkd, d * wk_stride), Arg::B(wvd, d * wv_stride)});
        }
        r.run_all();
        ok = true;
        return true;
    }

    bool can_draft(int a, int K) const {
        if (!ok || K < 1 || K > DFLY_KMAX) return false;
        if (a + K + 1 > r.slots_sparse) return false;
        if (r.cap_p0 < 0) return false;
        const int row = a - 1 - r.cap_p0;
        return row >= 0 && row < r.cap_rows;
    }

    // Draft K tokens from h_{a-1} = r.hfin row hrow. y[1..K], conf[1..K] on return.
    bool draft(int a, int xa, int K, int hrow, int* y, float* conf) {
        if (!can_draft(a, K)) return false;
        const double t0 = now_s();
        const int B = K + 1;
        const int row = a - 1 - r.cap_p0;
        int* hb = (int*) d_bids.ptr;
        hb[0] = xa;
        for (int i = 1; i <= K; ++i) hb[i] = DFLY_MASK_ID;
        ((int*) d_chain.ptr)[0] = xa;
        const std::string pf = "model.dfly_layers.";
        const float eps = c.rms_eps;
        auto& I = Runner::I;
        auto& Fl = Runner::Fl;
        auto Bf = [](const Buffer& b, size_t off = 0) { return Arg::B(b, off); };
        const Arg embw = r.g.tw("model.embed_tokens.weight");
        r.add("dfly_embed", B, 1, 256, {embw, Bf(d_bids), Bf(H)});
        r.add("dfly_h2f", (3072 + 255) / 256, 1, 256, {Bf(r.hfin, (size_t) hrow * 3072 * 2), Bf(hprev_f), I(3072)});
        r.add("dfly_cat2", 1, 1, 256, {Bf(H), wf(pf + "mtp_fusion.enorm.weight"), Bf(hprev_f), wf(pf + "mtp_fusion.hnorm.weight"), Bf(hcin), Fl(eps)});
        r.lin(pf + "mtp_fusion.eh_proj.weight", hcin, F1, 1);
        r.add("dfly_fuse", 1, 1, 256, {Bf(H), wf(pf + "mtp_fusion.gate"), Bf(F1)});
        for (int d = 0; d < 5; ++d) {
            const int L = DFLY_LAYERS[d];
            const std::string pd = pf + std::to_string(d);
            const std::string pa = pd + ".self_attn", pm = pd + ".mlp";
            const bool use_list = r.cap_list[d];
            const double th = c.theta_sparse;
            r.add("rms_rows", B, 1, 256, {Bf(H), wf(pd + ".input_layernorm.weight"), Bf(F1), Bf(hn), I(3072), Fl(eps), I(1 | 4)});
            r.lin(pa + ".q_a_proj.weight", hn, F1, B);
            r.add("rms_rows", B, 1, 256, {Bf(F1), wf(pa + ".q_a_layernorm.weight"), Bf(F1), Bf(qr_h), I(1024), Fl(eps), I(1 | 4)});
            r.lin(pa + ".q_b_proj.weight", qr_h, F2, B);
            r.lin(pa + ".kv_a_proj_with_mqa.weight", hn, F3, B);
            r.add("mla_write_kv", B, 1, 256, {Bf(F3), wf(pa + ".kv_a_layernorm.weight"), Bf(r.kcache[L]), I(a), I(0), Fl((float) th), Fl(eps)});
            r.add("mla_prep_q_1", 32, B, 256, {Bf(F2), Bf(wkd, d * wk_stride), Bf(qa_h), I(B), I(32), I(a), Fl((float) th)});
            r.lin(pa + ".linear_gate.weight", hn, F4, B);
            MlaArgs ma{};
            ma.M = B;
            ma.NH = 32;
            ma.p0 = a;
            ma.window = c.window;
            ma.ring = 0;
            ma.scale = 0.0625f;
            if (use_list) {
                r.add("dfly_lists", B, 1, 256, {Bf(r.dkeys[d], (size_t) row * r.index_topk * 4), Bf(r.dnkc[d], (size_t) row * 4), I(a), Bf(dlist), Bf(dnk), I(lstride)});
                ma.mode = MODE_LIST;
                ma.kmax = lstride;
            } else {
                ma.mode = MODE_CAUSAL;
                ma.kmax = 0;
            }
            r.add("mla_attend", 32, B, 256, {Bf(qa_h), Bf(r.kcache[L]), wf(pa + ".learnable_sink_param"), Bf(ctx_h), Arg::V<MlaArgs>(ma), Bf(dlist), Bf(dnk)});
            r.add("mla_vup_1", 32, B, 256, {Bf(ctx_h), Bf(wvd, d * wv_stride), Bf(F4), Bf(o_h), I(B), I(32)});
            r.lin(pa + ".o_proj.weight", o_h, F1, B);
            r.add("dfly_rms_resid", B, 1, 256, {Bf(F1), wf(pa + ".post_proj_norm.weight"), Bf(H), Fl(eps)});
            r.add("rms_rows", B, 1, 256, {Bf(H), wf(pd + ".post_attention_layernorm.weight"), Bf(F1), Bf(hn), I(3072), Fl(eps), I(1 | 4)});
            r.lin(pm + ".gate_proj.weight", hn, F1, B);
            r.lin(pm + ".up_proj.weight", hn, F2, B);
            const uint32_t n9 = (uint32_t) B * 9216;
            r.add("swiglu_h", (n9 + 255) / 256, 1, 256, {Bf(F1), Bf(F2), Bf(mlp_h), Runner::U(n9)});
            r.lin(pm + ".down_proj.weight", mlp_h, F3, B);
            const uint32_t nh = (uint32_t) B * 3072;
            r.add("add_inplace", (nh + 255) / 256, 1, 256, {Bf(H), Bf(F3), Runner::U(nh)});
        }
        r.add("rms_rows", B, 1, 256, {Bf(H), wf(pf + "final_norm.weight"), Bf(Hf), Bf(Hf_h), I(3072), Fl(eps), I(1 | 2 | 4)});
        for (int j = 1; j <= K; ++j) {
            r.add("dfly_embed", 1, 1, 256, {embw, Bf(d_chain, (size_t) (j - 1) * 4), Bf(Ec)});
            r.add("dfly_cat2", 1, 1, 256, {Bf(Hf, (size_t) j * 3072 * 4), wf(pf + "hidden_correction.hidden_norm.weight"), Bf(Ec), wf(pf + "hidden_correction.embed_norm.weight"), Bf(hcin), Fl(eps)});
            r.lin(pf + "hidden_correction.gate_proj.weight", hcin, G, 1);
            r.lin(pf + "hidden_correction.up_proj.weight", hcin, U, 1);
            r.add("swiglu_h", (3072 + 255) / 256, 1, 256, {Bf(G), Bf(U), Bf(act), Runner::U(3072)});
            r.lin(pf + "hidden_correction.down_proj.weight", act, Dn, 1);
            r.add("dfly_corr", 1, 1, 256, {Bf(Hf, (size_t) j * 3072 * 4), Bf(Dn), wf(pf + "confidence_head.proj.weight"), wf(pf + "confidence_head.proj.bias"), Bf(hc_h), Bf(d_conf, (size_t) j * 4)});
            r.lin("lm_head.weight", hc_h, r.logits, 1);
            r.add("argmax_row", 1, 1, 1024, {Bf(r.logits), I(c.vocab), Bf(d_chain, (size_t) j * 4)});
        }
        r.run_all();
        for (int j = 1; j <= K; ++j) {
            y[j] = ((int*) d_chain.ptr)[j];
            conf[j] = ((float*) d_conf.ptr)[j];
        }
        t_draft += now_s() - t0;
        ++n_draft;
        return true;
    }
};

struct SpecCfg {
    int K = 5;             // drafted tokens per block (1..7)
    bool adaptive = true;  // false: always verify K (fixed window, no back-off); for sweeps
    float cut = -1.f;      // >= 0: verify the leading slots with conf > cut instead of the cost model
    bool debug = false;
    bool sabotage = false; // test only: random draft ids with conf 0.9 (acceptance ~0)
};

struct SpecStats {
    long spec_iters = 0, plain_iters = 0, plain_after_draft = 0, tokens = 0, accepted = 0, verified_slots = 0, drafted = 0, probes = 0, backoffs = 0;
    long hist[9] = {};
    double t_draft = 0, t_verify = 0, t_plain = 0, t_spec_total = 0;
    int P_final = 8;
    long plain_mode_iters = 0;
};

class Spec {
public:
    Runner& r;
    DFly2& d;
    SpecCfg cfg;
    SpecStats S;
    int V;
    double tp = 0;
    double tv[9] = {};
    static constexpr double DM[8] = {16.0, 26.5, 35.6, 43.9, 51.4, 58.2, 64.8, 70.8};
    static constexpr float EDGE[5] = {0.2f, 0.4f, 0.6f, 0.8f, 0.9f};
    static constexpr float MID[6] = {0.1f, 0.3f, 0.5f, 0.7f, 0.85f, 0.95f};
    double cn[6], cy[6];
    struct Iter { double ms; int tok; };
    std::deque<Iter> win;
    int plain_left = 0, P = 8;
    bool probing = false;
    int y[DFLY_BMAX + 1] = {};
    float cf[DFLY_BMAX + 1] = {};
    int hrow = 0;  // row of r.hfin that holds h_{a-1} of the current anchor

    Spec(Runner& rr, DFly2& dd, const SpecCfg& c) : r(rr), d(dd), cfg(c), V(rr.c.vocab) {
        for (int b = 0; b < 6; ++b) { cn[b] = 2.0; cy[b] = 2.0 * MID[b]; }
        if (cfg.K > DFLY_KMAX) cfg.K = DFLY_KMAX;
        if (cfg.K < 1) cfg.K = 1;
        plain_left = 2;  // calibrate the plain step time first
    }

    static int bucket(float c) {
        int b = 0;
        while (b < 5 && c > EDGE[b]) ++b;
        return b;
    }
    double pcal(float c) const {
        const int b = bucket(c);
        return cy[b] / cn[b];
    }
    double tver(int m) const {
        if (tv[m] > 0) return tv[m];
        const double base = tp > 0 ? tp : 25.0;
        return base * (5.1 + 0.127 * DM[m - 1]) / 7.13;
    }
    int choose_k(int K) const {
        if (!cfg.adaptive) return K;
        if (cfg.cut >= 0) {
            int k = 0;
            while (k < K && cf[k + 1] > cfg.cut) ++k;
            return k;
        }
        const double tpl = tp > 0 ? tp : 25.0;
        int best = 0;
        double bt = 1.0 / tpl, E = 0, Pr = 1;
        for (int k = 1; k <= K; ++k) {
            Pr *= pcal(cf[k]);
            E += Pr;
            const double thr = (1.0 + E) / tver(k + 1);
            if (thr > bt) { bt = thr; best = k; }
        }
        return best;
    }

    int argmax_host_row0() {
        r.add("argmax_row", 1, 1, 1024, {Runner::B(r.logits), Runner::I(V), Runner::B(r.d_argmax)});
        r.run_all();
        return ((int*) r.d_argmax.ptr)[0];
    }

    bool plain_step(std::vector<int32_t>& tok, int a, int& nx, std::string& err) {
        if (!r.forward_chunk(tok.data(), a, 1, err)) return false;
        r.head(1, 0);
        hrow = 0;
        nx = argmax_host_row0();
        return true;
    }

    void note_iter(double ms, int ntok, bool is_spec) {
        if (!cfg.adaptive || cfg.cut >= 0 || !is_spec) return;
        win.push_back({ms, ntok});
        while (win.size() > 8) win.pop_front();
        const int n = (int) win.size();
        if (n < 3 || tp <= 0) return;
        double sm = 0, st = 0;
        for (auto& w : win) { sm += w.ms; st += w.tok; }
        const double ratio = sm / st;
        const double thr = tp * (n < 5 ? 1.25 : 1.0);
        if (ratio > thr) {
            ++S.backoffs;
            plain_left = P;
            S.P_final = P;
            P = std::min(P * 2, 256);
            win.clear();
            probing = false;
        } else if (n >= 8 && ratio < 0.9 * tp) {
            P = 8;
        }
    }

    // Generates greedily. Precondition: tok = prompt (all processed), r.logits row 0 = logits of the last prompt position, hfin row 0 = its h_final, draft capture on.
    // emit(token) returns false to stop. Returns the number of processed positions.
    int run(std::vector<int32_t>& tok, int limit, int max_ctx, int eos, const std::function<bool(int)>& emit, std::string& finish, std::string& err, int& made_out) {
        int made = 0;
        finish = "length";
        err.clear();
        int fwd = (int) tok.size();
        hrow = 0;
        auto push_emit = [&](int t) -> bool {
            tok.push_back(t);
            ++made;
            ++S.tokens;
            if (!emit(t)) { finish = "cancel"; return false; }
            if (t == eos) { finish = "stop"; return false; }
            if (made >= limit) { finish = "length"; return false; }
            return true;
        };
        int x = argmax_host_row0();
        if (limit <= 0 || !push_emit(x)) { made_out = made; return fwd; }
        for (;;) {
            const int a = (int) tok.size() - 1;
            fwd = a;
            if (a + 2 >= max_ctx) { finish = "length"; break; }
            const double ti0 = now_s();
            int Kd = cfg.K;
            if (a + Kd + 2 >= max_ctx) Kd = max_ctx - a - 3;
            const bool use_plain = plain_left > 0 || Kd < 1 || !d.can_draft(a, Kd);
            if (use_plain) {
                if (plain_left > 0) { --plain_left; ++S.plain_mode_iters; if (plain_left == 0) { probing = true; win.clear(); ++S.probes; } }
                int nx = 0;
                if (!plain_step(tok, a, nx, err)) { finish = "error"; break; }
                const double dt = (now_s() - ti0) * 1000.0;
                tp = tp > 0 ? 0.8 * tp + 0.2 * dt : dt;
                S.t_plain += dt;
                ++S.plain_iters;
                fwd = a + 1;
                if (!push_emit(nx)) break;
                continue;
            }
            const double td0 = now_s();
            if (!d.draft(a, tok[a], Kd, hrow, y, cf)) { finish = "error"; err = "draft failed"; break; }
            const double tdraft = (now_s() - td0) * 1000.0;
            if (cfg.sabotage) {
                static uint64_t xs = 88172645463325252ull;
                for (int j = 1; j <= Kd; ++j) { xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; y[j] = (int) (xs % 100000); cf[j] = 0.9f; }
            }
            S.t_draft += tdraft;
            S.drafted += Kd;
            int Kp = choose_k(Kd);
            if (Kp > Kd) Kp = Kd;
            if (Kp == 0) {
                int nx = 0;
                const double tq0 = now_s();
                if (!plain_step(tok, a, nx, err)) { finish = "error"; break; }
                const double dtp = (now_s() - tq0) * 1000.0;
                tp = tp > 0 ? 0.8 * tp + 0.2 * dtp : dtp;
                S.t_plain += dtp;
                ++S.plain_after_draft;
                fwd = a + 1;
                ++S.hist[0];
                note_iter((now_s() - ti0) * 1000.0, 1, true);
                S.t_spec_total += (now_s() - ti0) * 1000.0;
                ++S.spec_iters;
                if (!push_emit(nx)) break;
                continue;
            }
            const double tv0 = now_s();
            for (int i = 1; i <= Kp; ++i) tok.push_back(y[i]);
            const int T = Kp + 1;
            if (!r.forward_chunk(tok.data(), a, T, err)) { finish = "error"; break; }
            r.head(T, 0);
            r.add("argmax_rows", T, 1, 1024, {Runner::B(r.logits), Runner::I(V), Runner::B(r.d_argmax)});
            r.run_all();
            const int* h_arg = (const int*) r.d_argmax.ptr;
            const double tver_ms = (now_s() - tv0) * 1000.0;
            int n = 0;
            while (n < Kp && y[n + 1] == h_arg[n]) ++n;
            hrow = n;
            for (int j = 1; j <= std::min(Kp, n + 1); ++j) {
                const int b = bucket(cf[j]);
                cn[b] += 1.0;
                cy[b] += (j <= n) ? 1.0 : 0.0;
                if (cn[b] > 200) { cn[b] *= 0.5; cy[b] *= 0.5; }
            }
            tv[T] = tv[T] > 0 ? 0.7 * tv[T] + 0.3 * tver_ms : tver_ms;
            S.t_verify += tver_ms;
            S.verified_slots += Kp;
            S.accepted += n;
            ++S.hist[std::min(n, 8)];
            ++S.spec_iters;
            const int nxt = h_arg[n];
            if (cfg.debug) fprintf(stderr, "[spec] a=%d K'=%d n=%d draft %.1f ms verify %.1f ms conf %.2f %.2f %.2f\n", a, Kp, n, tdraft, tver_ms, cf[1], Kp > 1 ? cf[2] : 0.f, Kp > 2 ? cf[3] : 0.f);
            tok.resize((size_t) a + 1);
            fwd = a + 1 + n;
            bool alive = true;
            for (int i = 1; i <= n && alive; ++i) alive = push_emit(y[i]);
            if (alive) alive = push_emit(nxt);
            fwd = std::min(fwd, (int) tok.size());
            const double itms = (now_s() - ti0) * 1000.0;
            S.t_spec_total += itms;
            note_iter(itms, n + 1, true);
            if (!alive) break;
        }
        made_out = made;
        return std::min(fwd, (int) tok.size());
    }

    void report(FILE* f, double total_ms) const {
        const long steps = S.spec_iters + S.plain_iters;
        fprintf(f, "[spec] K=%d %s: %ld tokens in %ld iterations (%ld spec + %ld plain), %.2f tokens/iteration, total %.0f ms\n", cfg.K, cfg.adaptive ? "adaptive" : "fixed", S.tokens, steps,
                S.spec_iters, S.plain_iters, steps ? (double) S.tokens / steps : 0.0, total_ms);
        fprintf(f, "[spec] verify steps %ld: accepted drafts per step %.3f (verified slots %.2f/step), draft %.1f ms/step, verify %.1f ms/step, plain %.1f ms/step (EMA %.1f), spec iteration %.1f ms\n",
                S.spec_iters, S.spec_iters ? (double) S.accepted / S.spec_iters : 0.0, S.spec_iters ? (double) S.verified_slots / S.spec_iters : 0.0, S.spec_iters ? S.t_draft / S.spec_iters : 0.0,
                S.spec_iters ? S.t_verify / S.spec_iters : 0.0, S.plain_iters ? S.t_plain / (S.plain_iters + S.plain_after_draft) : 0.0, tp, S.spec_iters ? S.t_spec_total / S.spec_iters : 0.0);
        fprintf(f, "[spec] accepted histogram (0..8):");
        for (int i = 0; i < 9; ++i) fprintf(f, " %ld", S.hist[i]);
        fprintf(f, "\n[spec] backoff state: backoffs %ld, probes %ld, plain-mode iterations %ld, plain after draft (K'=0) %ld, P now %d (last %d)\n", S.backoffs, S.probes, S.plain_mode_iters,
                S.plain_after_draft, P, S.P_final);
        fprintf(f, "[spec] verify ms by m:");
        for (int m = 2; m <= 8; ++m) fprintf(f, " m%d=%.1f", m, tv[m]);
        fprintf(f, "\n");
    }
};
