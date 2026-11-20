// sy_spec.h - lossless greedy speculative decoding with the DFly2 drafter (sy_dfly2.h) and an adaptive controller that falls back to plain decode.
// SYCL port of the HY4-STRIX hy4_spec.hip.h (branch f-hy4-dfly2 a70c0e8fc); same controller (cost model + outer back-off), see that file for the design notes.
//   anchor x_a (emitted, not processed), target processed 0..a-1.  draft K tokens -> choose K' <= K -> verify [x_a, y_1..y_K'] as ONE chunk of the Runner (M = K'+1 <= 8)
//   -> accept y_j while it equals the target argmax of the previous row -> emit y_1..y_n and the target token t_n; new anchor t_n, h_{a'-1} = h_final row n.
//   KV rollback is positional: sparse rows beyond the accepted position are rewritten by the next chunk, the SWA ring has 512+Tc slots, IndexPool pooled keys of a group are rebuilt
//   when a chunk completes the group again.
#pragma once
#include <deque>
#include <functional>

#include "sy_dfly2.h"

namespace hy4s {

// Speculative decoding verifies windows of 2..8 rows: make their attention split the keys like the 1-token step so that a row has the same fp32 summation order in a window and in plain decode
// (otherwise spec output differs from plain greedy at near-ties). HY4S_BATCH_INV=0 turns it off.
inline void spec_set_batch_invariant() {
    const char* e = std::getenv("HY4S_BATCH_INV");
    g_mla_batch_invariant = (e && e[0] == '0') ? 0 : 1;
}

struct SpecCfg {
    int K = 5;             // drafted tokens per block (1..7)
    bool adaptive = true;  // false: always verify K (fixed window, no back-off); for sweeps
    float cut = -1.f;      // >= 0: verify the leading slots with conf > cut instead of the cost model
    bool debug = false;
    bool plain_only = false;  // --spec-plain-only: drafter initialised and capturing, but every step is a plain step (isolates the capture / drafter from the verify windows)
    bool check = false;    // --spec-check: after every verify, recompute the accepted window rows one token at a time and compare bitwise (diagnostic; slow)
    int rv = 256;          // reduced-vocabulary draft chain: candidates per slot (0 = full lm_head per drafted token)
};

struct SpecStats {
    long spec_iters = 0, plain_iters = 0, plain_after_draft = 0, tokens = 0, accepted = 0, verified_slots = 0, drafted = 0, probes = 0, backoffs = 0;
    long hist[9] = {};  // accepted drafts per verify step
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
    double tp = 0;      // plain step ms (EMA)
    double tv[9] = {};  // verify ms by m (EMA), 0 = not measured
    static constexpr double DM[8] = {16.0, 26.5, 35.6, 43.9, 51.4, 58.2, 64.8, 70.8};  // distinct experts per layer in a window of m tokens (HY4-STRIX measurement)
    static constexpr float EDGE[5] = {0.2f, 0.4f, 0.6f, 0.8f, 0.9f};
    static constexpr float MID[6] = {0.1f, 0.3f, 0.5f, 0.7f, 0.85f, 0.95f};
    double cn[6], cy[6];
    struct Iter { double ms; int tok; };
    std::deque<Iter> win;
    int plain_left = 0, P = 8;
    bool probing = false;
    int32_t* h_arg = nullptr;
    int y[DFLY_BMAX + 1] = {};
    float cf[DFLY_BMAX + 1] = {};

    Spec(Runner& rr, DFly2& dd, const SpecCfg& c) : r(rr), d(dd), cfg(c), V(rr.c.vocab) {
        for (int b = 0; b < 6; ++b) { cn[b] = 2.0; cy[b] = 2.0 * MID[b]; }
        h_arg = halloc<int32_t>(16);
        if (cfg.K > DFLY_KMAX) cfg.K = DFLY_KMAX;
        if (cfg.K < 1) cfg.K = 1;
        d.rv = cfg.rv;
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
    double tver(int m) const {  // verify ms for m tokens
        if (tv[m] > 0) return tv[m];
        const double base = tp > 0 ? tp : 46.0;
        return base * (5.1 + 0.127 * DM[m - 1]) / 7.13;
    }
    int choose_k(int K) const {
        if (!cfg.adaptive) return K;
        if (cfg.cut >= 0) {
            int k = 0;
            while (k < K && cf[k + 1] > cfg.cut) ++k;
            return k;
        }
        const double tpl = tp > 0 ? tp : 46.0;
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
        argmax_rows(r.logits, V, r.d_argmax, 1);
        d2h(h_arg, r.d_argmax, 4);
        wait();
        return h_arg[0];
    }

    // One plain greedy step for the anchor at index a (tok[a] pushed): returns the next token.
    bool plain_step(std::vector<int32_t>& tok, int a, int& nx, std::string& err) {
        if (!r.forward_chunk(tok.data(), a, 1, err)) return false;
        r.head(1, 0);
        d.snap_h(0);
        nx = argmax_host_row0();
        return true;
    }

    void note_iter(double ms, int ntok, bool is_spec) {
        if (!cfg.adaptive || cfg.cut >= 0 || !is_spec) return;
        win.push_back({ms, ntok});
        while (win.size() > 8) win.pop_front();
        const int n = (int)win.size();
        if (n < 3 || tp <= 0) return;
        double sm = 0, st = 0;
        for (auto& w : win) { sm += w.ms; st += w.tok; }
        const double ratio = sm / st;  // ms per token
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

    // Generates greedily. Precondition: tok = prompt (all processed), r.logits row 0 = logits of the last prompt position, d.snap_h() done for it, drafter capture on.
    // emit(token) returns false to stop (the token is already counted). Returns the number of processed positions (resident KV prefix of tok).
    int run(std::vector<int32_t>& tok, int limit, int max_ctx, int eos, int eot, const std::function<bool(int)>& emit, std::string& finish, std::string& err, int& made_out) {
        int made = 0;
        finish = "length";
        err.clear();
        int fwd = (int)tok.size();
        auto push_emit = [&](int t) -> bool {
            tok.push_back(t);
            ++made;
            ++S.tokens;
            if (!emit(t)) { finish = "cancel"; return false; }
            if (t == eos || t == eot) { finish = "stop"; return false; }
            if (made >= limit) { finish = "length"; return false; }
            return true;
        };
        int x = argmax_host_row0();
        if (limit <= 0 || !push_emit(x)) { made_out = made; return fwd; }
        for (;;) {
            const int a = (int)tok.size() - 1;
            fwd = a;
            if (a + 2 >= max_ctx) { finish = "length"; break; }
            const double ti0 = now_s();
            int Kd = cfg.K;
            if (a + Kd + 2 >= max_ctx) Kd = max_ctx - a - 3;
            // Clamp speculative window so it never crosses a 32-token adaptive VRAM promotion boundary (plain == DFly2 bitwise!)
            if (!cfg.check && r.vcache && r.vcache->adaptive()) {
                const int next_b = ((a >> 5) + 1) << 5;
                if (a + 1 + Kd > next_b) Kd = std::max(0, next_b - a - 1);
            }
            const bool use_plain = cfg.plain_only || plain_left > 0 || Kd < 1 || !d.can_draft(a, Kd);
            if (use_plain) {
                if (plain_left > 0) { --plain_left; ++S.plain_mode_iters; if (plain_left == 0) { probing = true; win.clear(); ++S.probes; } }
                int nx = 0;
                if (!plain_step(tok, a, nx, err)) { finish = "error"; break; }
                const double dt = (now_s() - ti0) * 1000.0;
                tp = tp > 0 ? 0.8 * tp + 0.2 * dt : dt;
                S.t_plain += dt;
                ++S.plain_iters;
                fwd = a + 1;
                if (!cfg.check) r.commit_vram_adapt(1, a + 1);
                if (!push_emit(nx)) break;
                continue;
            }
            // draft
            const double td0 = now_s();
            if (!d.draft(a, tok[a], Kd, y, cf)) { finish = "error"; err = "draft failed"; break; }
            const double tdraft = (now_s() - td0) * 1000.0;
            S.t_draft += tdraft;
            S.drafted += Kd;
            int Kp = choose_k(Kd);
            if (Kp > Kd) Kp = Kd;
            if (Kp == 0) {
                // nothing worth verifying: a plain step (the draft time is the price of this iteration)
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
                if (!cfg.check) r.commit_vram_adapt(1, a + 1);
                if (!push_emit(nx)) break;
                continue;
            }
            // verify [x_a, y_1..y_Kp]
            const double tv0 = now_s();
            for (int i = 1; i <= Kp; ++i) tok.push_back(y[i]);
            const int T = Kp + 1;
            std::vector<std::pair<std::string, std::vector<float>>> wtap;
            const bool dfly_cap_saved = r.dfly_cap;
            if (cfg.check) r.tap = &wtap;
            if (!r.forward_chunk(tok.data(), a, T, err)) { finish = "error"; break; }
            r.tap = nullptr;
            r.head(T, 0);
            std::vector<float> wlog;
            if (cfg.check) {
                wlog.resize((size_t)T * V);
                d2h(wlog.data(), r.logits, wlog.size() * 4);
                wait();
            }
            argmax_rows(r.logits, V, r.d_argmax, T);
            d2h(h_arg, r.d_argmax, T * 4);
            wait();
            const double tver_ms = (now_s() - tv0) * 1000.0;
            int n = 0;
            while (n < Kp && y[n + 1] == h_arg[n]) ++n;
            d.snap_h(n);
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
            if (cfg.check) {
                // rows 0..n are the accepted ones: recompute each as a 1-token step (same KV prefix) and compare the logits bitwise; on a difference say which tap differs first
                r.dfly_cap = false;
                static long nchecks = 0, nbad = 0;
                for (int i = 0; i <= n; ++i) {
                    std::vector<std::pair<std::string, std::vector<float>>> stap;
                    r.tap = &stap;
                    if (!r.forward_chunk(tok.data(), a + i, 1, err)) { finish = "error"; break; }
                    r.tap = nullptr;
                    r.head(1, 0);
                    std::vector<float> sl((size_t)V);
                    d2h(sl.data(), r.logits, (size_t)V * 4);
                    wait();
                    size_t lb = 0;
                    for (int e = 0; e < V; ++e) if (std::memcmp(&sl[e], &wlog[(size_t)i * V + e], 4) != 0) ++lb;
                    ++nchecks;
                    if (lb) {
                        ++nbad;
                        std::string first = "none";
                        for (size_t k = 0; k < stap.size() && k < wtap.size(); ++k) {
                            const size_t per = stap[k].second.size();
                            if (stap[k].first != wtap[k].first || wtap[k].second.size() < (size_t)(i + 1) * per) continue;
                            size_t nb = 0;
                            for (size_t e = 0; e < per; ++e) {
                                const float x = stap[k].second[e], y2 = wtap[k].second[(size_t)i * per + e];
                                if (std::memcmp(&x, &y2, 4) != 0) ++nb;
                            }
                            if (nb) { first = stap[k].first + " (" + std::to_string(nb) + " of " + std::to_string(per) + ")"; break; }
                        }
                        fprintf(stderr, "[spec-check] MISMATCH a=%d row %d of window T=%d (accepted n=%d, p0 mod 4 = %d): %zu of %d logits differ; first differing tap: %s\n", a, i, T, n, a & 3, lb, V, first.c_str());
                    }
                }
                r.dfly_cap = dfly_cap_saved;
                if (nchecks % 20 == 0) fprintf(stderr, "[spec-check] %ld rows checked, %ld mismatches so far\n", nchecks, nbad);
            }
            if (cfg.debug) fprintf(stderr, "[spec] a=%d K'=%d n=%d draft %.1f ms verify %.1f ms conf %.2f %.2f %.2f\n", a, Kp, n, tdraft, tver_ms, cf[1], Kp > 1 ? cf[2] : 0.f, Kp > 2 ? cf[3] : 0.f);
            tok.resize((size_t)a + 1);
            fwd = a + 1 + n;
            bool alive = true;
            for (int i = 1; i <= n && alive; ++i) alive = push_emit(y[i]);
            if (alive) alive = push_emit(nxt);
            fwd = std::min(fwd, (int)tok.size());
            const double itms = (now_s() - ti0) * 1000.0;
            S.t_spec_total += itms;
            note_iter(itms, n + 1, true);
            if (!cfg.check) r.commit_vram_adapt(n + 1, a + n + 1);
            if (!alive) break;
        }
        made_out = made;
        return std::min(fwd, (int)tok.size());
    }

    void report(FILE* f, double total_ms) const {
        if (cfg.check) fprintf(f, "[spec-check] done (mismatch lines above, if any)\n");
        const long steps = S.spec_iters + S.plain_iters;
        fprintf(f, "[spec] K=%d %s: %ld tokens in %ld iterations (%ld spec + %ld plain), %.2f tokens/iteration, total %.0f ms\n", cfg.K, cfg.adaptive ? "adaptive" : "fixed", S.tokens, steps,
                S.spec_iters, S.plain_iters, steps ? (double)S.tokens / steps : 0.0, total_ms);
        fprintf(f, "[spec] verify steps %ld: accepted drafts per step %.3f (verified slots %.2f/step), draft %.1f ms/step, verify %.1f ms/step, plain %.1f ms/step (EMA %.1f), spec iteration %.1f ms\n",
                S.spec_iters, S.spec_iters ? (double)S.accepted / S.spec_iters : 0.0, S.spec_iters ? (double)S.verified_slots / S.spec_iters : 0.0, S.spec_iters ? S.t_draft / S.spec_iters : 0.0,
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

}  // namespace hy4s
