// hy4_spec.hip.h - lossless greedy speculative decoding with the DFly2 drafter (hy4_dfly2.hip.h) and an adaptive controller that falls back to plain decode.
//
// Loop (greedy only: the output is the target's greedy continuation, token for token, up to the numerics of the M-token verify kernels, see the parity note):
//   anchor x_a (emitted, not processed), target processed 0..a-1.  draft K tokens -> choose K' <= K (confidence model) -> verify [x_a, y_1..y_K'] as ONE chunk of the Runner
//   (M = K'+1 <= 8 decode path) -> accept y_j while it equals the target argmax of the previous row -> emit y_1..y_n and the target token t_n; new anchor t_n, h_{a'-1} = h_final row n.
//   KV rollback is positional (nothing to restore): sparse rows beyond the accepted position are rewritten by the next chunk, the SWA ring has 512+Tc >= 520 slots (a speculative row
//   only overwrites a slot older than every window that can still be asked), IndexPool pooled keys of a group are rebuilt by the chunk that completes the group again.
// Controller: (1) per step K' = argmax_k (1 + E_k) / t_verify(k+1) against plain decode, E_k from calibrated confidences; (2) outer back-off on the measured ms/token of the last
// iterations against plain decode: the loop falls back to plain steps for P steps (P doubles up to 256 on repeated failures) and then probes again.
#pragma once
#include <deque>
#include <functional>

#include "hy4_dfly2.hip.h"

namespace hy4 {

struct SpecCfg {
    int K = 7;             // drafted tokens per block (1..7)
    bool adaptive = true;  // false: always verify K (fixed window, no back-off); for sweeps
    float cut = -1.f;      // >= 0: verify the leading slots with conf > cut instead of the cost model
    bool exact = true;     // verify windows on the M-independent GEMV / decode expert kernels
    bool debug = false;
    bool check = false;    // diagnostic: after every verify recompute the accepted rows one token at a time and compare bitwise
    bool inv = true;       // batch-invariant MLA split-K for verify windows (costs ~5 ms per verify step, makes window rows sum in the one-token order)
    bool nodraft = false;  // test only (with sabotage): skip the drafter entirely, verify windows of fixed junk tokens
    bool sabotage = false; // test only: replace every draft token by a random id with conf 0.9 (acceptance ~0): the worst case for the controller
    int rv = 256;          // reduced-vocabulary draft chain: candidates per slot (0 = full lm_head per drafted token)
};

struct SpecStats {
    long chk_rows = 0, chk_bad = 0;
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
    // cost model
    double tp = 0;      // plain step ms (EMA)
    double tv[9] = {};  // verify ms by m (EMA), 0 = not measured
    static constexpr double DM[8] = {16.0, 26.5, 35.6, 43.9, 51.4, 58.2, 64.8, 70.8};
    // confidence calibration, buckets by conf
    static constexpr float EDGE[5] = {0.2f, 0.4f, 0.6f, 0.8f, 0.9f};
    static constexpr float MID[6] = {0.1f, 0.3f, 0.5f, 0.7f, 0.85f, 0.95f};
    double cn[6], cy[6];
    // controller
    struct Iter { double ms; int tok; };
    std::deque<Iter> win;
    int plain_left = 0, P = 8;
    bool probing = false;
    int32_t* h_arg = nullptr;
    int y[DFLY_BMAX + 1] = {};
    float cf[DFLY_BMAX + 1] = {};

    Spec(Runner& rr, DFly2& dd, const SpecCfg& c) : r(rr), d(dd), cfg(c), V(rr.c.vocab) {
        for (int b = 0; b < 6; ++b) { cn[b] = 2.0; cy[b] = 2.0 * MID[b]; }
        HCK(hipHostMalloc((void**)&h_arg, 64, hipHostMallocDefault));
        if (cfg.exact) {
            skd::moe_dec_max_m() = 16;
            skd::lin_gemv_max_m() = 16;
        }
        if (cfg.K > DFLY_KMAX) cfg.K = DFLY_KMAX;
        if (cfg.K < 1) cfg.K = 1;
        d.rv = cfg.rv;
        r.det_mix = true;  // schedule-independent iHC mix: the output must not depend on atomics
        g_mla_batch_invariant = cfg.inv ? 1 : 0;  // verify windows split the keys like a one-token step
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

    // optional: top1 - top2 logit gap of every emitted token's row (analysis of divergences)
    std::vector<float> marg;
    std::vector<uint64_t> hsh;
    bool want_margin = false;
    void note_margin(int row) {
        if (!want_margin) return;
        std::vector<float> h(V);
        HCK(hipMemcpy(h.data(), r.logits + (size_t)row * V, (size_t)V * 4, hipMemcpyDeviceToHost));
        float b1 = -3e38f, b2 = -3e38f;
        for (int i = 0; i < V; ++i) {
            if (h[i] > b1) { b2 = b1; b1 = h[i]; }
            else if (h[i] > b2) b2 = h[i];
        }
        marg.push_back(b1 - b2);
        {
            uint64_t hh = 1469598103934665603ull;
            for (int i = 0; i < V; ++i) {
                uint32_t bits;
                memcpy(&bits, &h[i], 4);
                hh = (hh ^ bits) * 1099511628211ull;
            }
            hsh.push_back(hh);
        }
    }

    int argmax_host_row0() {
        note_margin(0);
        hipLaunchKernelGGL(k_argmax, dim3(1), dim3(1024), 0, r.st, (const float*)r.logits, V, r.d_argmax);
        HCK(hipMemcpyAsync(h_arg, r.d_argmax, 4, hipMemcpyDeviceToHost, r.st));
        HCK(hipStreamSynchronize(r.st));
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
            // draft
            const double td0 = now_s();
            if (cfg.nodraft) {  // test only: no drafter work at all (HY4_DUMMY_MB: a device-to-device copy of that many MB instead, to test timing / scheduling sensitivity of the target kernels)
                for (int j = 1; j <= Kd; ++j) { y[j] = 1000 + j; cf[j] = 0.9f; }
                static float* dummy = nullptr;
                static size_t dummy_bytes = getenv("HY4_DUMMY_MB") ? (size_t)atoi(getenv("HY4_DUMMY_MB")) << 20 : 0;
                if (dummy_bytes) {
                    if (!dummy) HCK(hipMalloc((void**)&dummy, 2 * dummy_bytes));
                    HCK(hipMemcpyAsync((char*)dummy + dummy_bytes, dummy, dummy_bytes, hipMemcpyDeviceToDevice, r.st));
                    HCK(hipStreamSynchronize(r.st));
                }
            }
            else if (!d.draft(a, tok[a], Kd, y, cf)) { finish = "error"; err = "draft failed"; break; }
            const double tdraft = (now_s() - td0) * 1000.0;
            if (cfg.sabotage) {
                static uint64_t xs = 88172645463325252ull;
                for (int j = 1; j <= Kd; ++j) { xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; y[j] = (int)(xs % 100000); cf[j] = 0.9f; }
            }
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
                if (!push_emit(nx)) break;
                continue;
            }
            // verify [x_a, y_1..y_Kp]
            const double tv0 = now_s();
            for (int i = 1; i <= Kp; ++i) tok.push_back(y[i]);
            const int T = Kp + 1;
            if (!r.forward_chunk(tok.data(), a, T, err)) { finish = "error"; break; }
            r.head(T, 0);
            hipLaunchKernelGGL(k_argmax_rows, dim3(T), dim3(1024), 0, r.st, (const float*)r.logits, V, r.d_argmax);
            HCK(hipMemcpyAsync(h_arg, r.d_argmax, T * 4, hipMemcpyDeviceToHost, r.st));
            HCK(hipStreamSynchronize(r.st));
            const double tver_ms = (now_s() - tv0) * 1000.0;
            int n = 0;
            while (n < Kp && y[n + 1] == h_arg[n]) ++n;
            d.snap_h(n);
            if (cfg.check) {
                // diagnostic: recompute rows 0..n one token at a time and compare them with the window's rows (bitwise); the rewritten KV rows are the same when the kernels are batch invariant
                std::vector<float> wl((size_t)T * V), sl(V);
                HCK(hipMemcpy(wl.data(), r.logits, (size_t)T * V * 4, hipMemcpyDeviceToHost));
                for (int i = 0; i <= n; ++i) {
                    if (!r.forward_chunk(tok.data(), a + i, 1, err)) { finish = "error"; break; }
                    r.head(1, 0);
                    HCK(hipStreamSynchronize(r.st));
                    HCK(hipMemcpy(sl.data(), r.logits, (size_t)V * 4, hipMemcpyDeviceToHost));
                    ++S.chk_rows;
                    if (!std::equal(sl.begin(), sl.end(), wl.begin() + (size_t)i * V)) {
                        double md = 0;
                        for (int v = 0; v < V; ++v) md = std::max(md, (double)std::fabs(sl[v] - wl[(size_t)i * V + v]));
                        ++S.chk_bad;
                        if (S.chk_bad <= 12) fprintf(stderr, "[check] a=%d (a mod 4 = %d) row %d of window T=%d (n=%d): window row differs from the one-token row, max|dlogit| %.4g, argmax %d vs %d\n", a, a & 3, i, T, n, md,
                                                     (int)(std::max_element(sl.begin(), sl.end()) - sl.begin()), (int)(std::max_element(wl.begin() + (size_t)i * V, wl.begin() + (size_t)(i + 1) * V) - wl.begin() - (size_t)i * V));
                    }
                }
                if (finish == "error") break;
            }
            if (want_margin)
                for (int i = 0; i <= n; ++i) note_margin(i);
            // calibration + cost model
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
            tok.resize((size_t)a + 1);
            fwd = a + 1 + n;
            bool alive = true;
            for (int i = 1; i <= n && alive; ++i) alive = push_emit(y[i]);
            if (alive) alive = push_emit(nxt);
            fwd = std::min(fwd, (int)tok.size());
            const double itms = (now_s() - ti0) * 1000.0;
            S.t_spec_total += itms;
            note_iter(itms, n + 1, true);
            if (!alive) break;
        }
        made_out = made;
        return std::min(fwd, (int)tok.size());
    }

    void report(FILE* f, double total_ms) const {
        const long steps = S.spec_iters + S.plain_iters;
        fprintf(f, "[spec] K=%d %s: %ld tokens in %ld iterations (%ld spec + %ld plain), %.2f tokens/iteration, total %.0f ms\n", cfg.K, cfg.adaptive ? "adaptive" : "fixed", S.tokens, steps,
                S.spec_iters, S.plain_iters, steps ? (double)S.tokens / steps : 0.0, total_ms);
        fprintf(f, "[spec] verify steps %ld: accepted drafts per step %.3f (verified slots %.2f/step), draft %.1f ms/step, verify %.1f ms/step, plain %.1f ms/step (EMA %.1f), "
                   "spec iteration %.1f ms\n",
                S.spec_iters, S.spec_iters ? (double)S.accepted / S.spec_iters : 0.0, S.spec_iters ? (double)S.verified_slots / S.spec_iters : 0.0,
                S.spec_iters ? S.t_draft / S.spec_iters : 0.0, S.spec_iters ? S.t_verify / S.spec_iters : 0.0, S.plain_iters ? S.t_plain / (S.plain_iters + S.plain_after_draft) : 0.0, tp,
                S.spec_iters ? S.t_spec_total / S.spec_iters : 0.0);
        fprintf(f, "[spec] accepted histogram (0..8):");
        for (int i = 0; i < 9; ++i) fprintf(f, " %ld", S.hist[i]);
        fprintf(f, "\n[spec] backoff state: backoffs %ld, probes %ld, plain-mode iterations %ld, plain after draft (K'=0) %ld, P now %d (last %d)\n", S.backoffs, S.probes, S.plain_mode_iters,
                S.plain_after_draft, P, S.P_final);
        fprintf(f, "[spec] verify ms by m:");
        for (int m = 2; m <= 8; ++m) fprintf(f, " m%d=%.1f", m, tv[m]);
        fprintf(f, "\n");
        if (cfg.check) fprintf(f, "[spec] check: %ld window rows recomputed one token at a time, %ld differ bitwise\n", S.chk_rows, S.chk_bad);
    }
};

}  // namespace hy4
