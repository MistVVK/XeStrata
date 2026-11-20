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
    int K = 7;               // drafted tokens per block (1..7)
    bool adaptive = true;    // false: always verify K (fixed window, no back-off); for sweeps
    float cut = -1.f;        // >= 0: verify the leading slots with conf > cut instead of the cost model
    bool exact = true;       // verify windows on the M-independent GEMV / decode expert kernels
    bool debug = false;
    bool check = false;      // diagnostic: after every verify recompute the accepted rows one token at a time and compare bitwise
    bool inv = true;         // batch-invariant MLA split-K for verify windows (costs ~5 ms per verify step, makes window rows sum in the one-token order)
    bool nodraft = false;    // test only (with sabotage): skip the drafter entirely, verify windows of fixed junk tokens
    bool sabotage = false;   // test only: replace every draft token by a random id with conf 0.9 (acceptance ~0): the worst case for the controller
    int rv = 256;            // reduced-vocabulary draft chain: candidates per slot (0 = full lm_head per drafted token)
    bool tree = true;        // Tree-8 speculative verification (top-2 branching at uncertain slots within M <= 8 verify tile)
    float early_cut = 0.08f; // early-exit draft chain when best remaining frontier reach prob < early_cut (0 = disabled)
};

struct SpecStats {
    long chk_rows = 0, chk_bad = 0;
    long spec_iters = 0, plain_iters = 0, plain_after_draft = 0, tokens = 0, accepted = 0, verified_slots = 0, drafted = 0, probes = 0, backoffs = 0;
    long tree_steps = 0, tree_branch_hits = 0, prob_iters = 0;
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
    float node_conf[DFLY_BMAX] = {};
    float node_reach[DFLY_BMAX] = {};
    std::vector<float> h_vlogits;
    std::vector<SparseDist> node_qdist;

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

    int choose_m(const TreeSpec& tr) const {
        if (!cfg.adaptive) return tr.M;
        if (cfg.cut >= 0) {
            int m = 1;
            while (m < tr.M && node_conf[m] > cfg.cut) ++m;
            return m;
        }
        const double tpl = tp > 0 ? tp : 46.0;
        int best_m = 1;
        double bt = 1.0 / tpl, E = 0.0;
        for (int m = 2; m <= tr.M; ++m) {
            E += (double)node_reach[m - 1];
            const double thr = (1.0 + E) / tver(m);
            if (thr > bt) { bt = thr; best_m = m; }
        }
        return best_m;
    }

    int sample_host_row0(const SampleOpts* samp, const std::vector<int32_t>& hist, std::mt19937_64* rng) {
        if (!samp || (samp->greedy() && !samp->penalties())) return argmax_host_row0();
        if ((int)h_vlogits.size() < V) h_vlogits.resize((size_t)V);
        HCK(hipMemcpy(h_vlogits.data(), r.logits, (size_t)V * 4, hipMemcpyDeviceToHost));
        SparseDist dist;
        compute_dist_full(h_vlogits.data(), V, *samp, hist, dist);
        return sample_from_dist(dist, *rng);
    }

    // One plain step for the anchor at index a (tok[a] pushed): returns the next token.
    bool plain_step(std::vector<int32_t>& tok, int a, int& nx, std::string& err, const SampleOpts* samp = nullptr, std::mt19937_64* rng = nullptr) {
        if (!r.forward_chunk(tok.data(), a, 1, err)) return false;
        r.head(1, 0);
        d.snap_h(0);
        nx = sample_host_row0(samp, tok, rng);
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

    // Generates greedily (samp == nullptr or greedy & no penalties) or with lossless probabilistic rejection sampling (temperature > 0 / top_p / top_k / min_p / penalties).
    // Precondition: tok = prompt (all processed), r.logits row 0 = logits of the last prompt position, d.snap_h() done for it, drafter capture on.
    // emit(token) returns false to stop (the token is already counted). Returns the number of processed positions (resident KV prefix of tok).
    int run(std::vector<int32_t>& tok, int limit, int max_ctx, int eos, int eot, const std::function<bool(int)>& emit, std::string& finish, std::string& err, int& made_out,
            const SampleOpts* samp = nullptr, std::mt19937_64* rng = nullptr) {
        int made = 0;
        finish = "length";
        err.clear();
        const bool prob_mode = samp != nullptr && (!samp->greedy() || samp->penalties());
        std::mt19937_64 local_rng(samp && samp->seed ? samp->seed : 0x57A7A123456789ABull);
        if (prob_mode && !rng) rng = &local_rng;
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
        int x = sample_host_row0(samp, tok, rng);
        if (limit <= 0 || !push_emit(x)) { made_out = made; return fwd; }
        auto pcal_fn = [this](float c) { return pcal(c); };
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
                if (!plain_step(tok, a, nx, err, samp, rng)) { finish = "error"; break; }
                const double dt = (now_s() - ti0) * 1000.0;
                tp = tp > 0 ? 0.8 * tp + 0.2 * dt : dt;
                S.t_plain += dt;
                ++S.plain_iters;
                fwd = a + 1;
                if (!push_emit(nx)) break;
                continue;
            }
            // draft (Tree-8 + Early-Exit + Probabilistic-aware)
            const double td0 = now_s();
            TreeSpec tr;
            const float ecut = (cfg.adaptive && cfg.cut < 0 && !cfg.sabotage && !cfg.nodraft) ? cfg.early_cut : 0.f;
            if (cfg.nodraft) {
                tr.M = Kd + 1;
                tr.tok[0] = tok[a];
                tr.parent[0] = -1;
                tr.depth[0] = 0;
                node_conf[0] = 1.0f;
                node_reach[0] = 1.0f;
                for (int j = 1; j <= Kd; ++j) {
                    tr.tok[j] = 1000 + j;
                    tr.parent[j] = (int8_t)(j - 1);
                    tr.depth[j] = (uint8_t)j;
                    node_conf[j] = 0.9f;
                    node_reach[j] = (float)std::pow(0.9, j);
                }
                tr.build_anc();
                static float* dummy = nullptr;
                static size_t dummy_bytes = getenv("HY4_DUMMY_MB") ? (size_t)atoi(getenv("HY4_DUMMY_MB")) << 20 : 0;
                if (dummy_bytes) {
                    if (!dummy) HCK(hipMalloc((void**)&dummy, 2 * dummy_bytes));
                    HCK(hipMemcpyAsync((char*)dummy + dummy_bytes, dummy, dummy_bytes, hipMemcpyDeviceToDevice, r.st));
                    HCK(hipStreamSynchronize(r.st));
                }
            } else if (!d.draft_tree(a, tok[a], Kd, tr, node_conf, node_reach, samp, &tok, rng, prob_mode ? &node_qdist : nullptr, ecut, cfg.tree && !cfg.sabotage, pcal_fn)) {
                finish = "error";
                err = "draft failed";
                break;
            }
            const double tdraft = (now_s() - td0) * 1000.0;
            if (cfg.sabotage) {
                static uint64_t xs = 88172645463325252ull;
                tr.M = Kd + 1;
                tr.tok[0] = tok[a];
                tr.parent[0] = -1;
                tr.depth[0] = 0;
                node_conf[0] = 1.0f;
                node_reach[0] = 1.0f;
                for (int j = 1; j <= Kd; ++j) {
                    xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17;
                    tr.tok[j] = (int)(xs % 100000);
                    tr.parent[j] = (int8_t)(j - 1);
                    tr.depth[j] = (uint8_t)j;
                    node_conf[j] = 0.9f;
                    node_reach[j] = (float)std::pow(0.9, j);
                }
                tr.build_anc();
            }
            S.t_draft += tdraft;
            S.drafted += (tr.M - 1);
            int T = choose_m(tr);
            if (T > tr.M) T = tr.M;
            tr.M = T;
            const int Kp = T - 1;
            if (Kp <= 0) {
                int nx = 0;
                const double tq0 = now_s();
                if (!plain_step(tok, a, nx, err, samp, rng)) { finish = "error"; break; }
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
            bool is_branched = false;
            for (int i = 1; i < T; ++i)
                if (tr.parent[i] != i - 1) { is_branched = true; break; }
            if (is_branched) ++S.tree_steps;
            // verify tree (or linear fast path when !is_branched)
            const bool chk_fr_was = r.vcache_frozen;
            if (cfg.check && !prob_mode) {
                r.vcache_flush();
                r.vcache_frozen = true;
            }
            const double tv0 = now_s();
            if (!is_branched) {
                for (int i = 1; i < T; ++i) tok.push_back(tr.tok[i]);
                if (!r.forward_chunk(tok.data(), a, T, err, nullptr)) { r.vcache_frozen = chk_fr_was; finish = "error"; break; }
            } else {
                if (!r.forward_chunk(tok.data(), a, T, err, &tr)) { r.vcache_frozen = chk_fr_was; finish = "error"; break; }
            }
            r.head(T, 0);
            int path[9] = {};
            int n = 0;
            int u = 0;
            int nxt = 0;
            double tver_ms = 0.0;
            if (!prob_mode) {
                hipLaunchKernelGGL(k_argmax_rows, dim3(T), dim3(1024), 0, r.st, (const float*)r.logits, V, r.d_argmax);
                HCK(hipMemcpyAsync(h_arg, r.d_argmax, T * 4, hipMemcpyDeviceToHost, r.st));
                HCK(hipStreamSynchronize(r.st));
                tver_ms = (now_s() - tv0) * 1000.0;
                path[0] = 0;
                for (;;) {
                    const int target_tok = h_arg[u];
                    int hit = -1;
                    for (int c = u + 1; c < T; ++c) {
                        if (tr.parent[c] == u && tr.tok[c] == target_tok) { hit = c; break; }
                    }
                    if (hit < 0) break;
                    ++n;
                    path[n] = hit;
                    if (hit != n) ++S.tree_branch_hits;
                    u = hit;
                }
                nxt = h_arg[u];
            } else {
                ++S.prob_iters;
                if ((int)h_vlogits.size() < T * V) h_vlogits.resize((size_t)T * V);
                HCK(hipMemcpyAsync(h_vlogits.data(), r.logits, (size_t)T * V * 4, hipMemcpyDeviceToHost, r.st));
                HCK(hipStreamSynchronize(r.st));
                tver_ms = (now_s() - tv0) * 1000.0;
                std::vector<int32_t> cur_hist(tok.begin(), tok.begin() + (size_t)a + 1);
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                path[0] = 0;
                for (;;) {
                    SparseDist p_res;
                    compute_dist_full(h_vlogits.data() + (size_t)u * V, V, *samp, cur_hist, p_res);
                    int hit = -1;
                    for (int c = u + 1; c < T; ++c) {
                        if (tr.parent[c] != u) continue;
                        const int32_t y_tok = tr.tok[c];
                        const SparseDist& q_c = node_qdist[(size_t)c];
                        const double py = dist_prob_of(p_res, y_tok);
                        const double qy = dist_prob_of(q_c, y_tok);
                        if (qy > 0.0 && u01(*rng) * qy < py) {
                            hit = c;
                            break;
                        }
                        double z = 0.0;
                        for (auto& kv : p_res) {
                            const double diff = kv.second - dist_prob_of(q_c, kv.first);
                            kv.second = diff > 0.0 ? diff : 0.0;
                            z += kv.second;
                        }
                        if (z > 1e-15) {
                            for (auto& kv : p_res) kv.second /= z;
                        } else {
                            compute_dist_full(h_vlogits.data() + (size_t)u * V, V, *samp, cur_hist, p_res);
                        }
                    }
                    if (hit < 0) {
                        nxt = sample_from_dist(p_res, *rng);
                        break;
                    }
                    ++n;
                    path[n] = hit;
                    if (hit != n) ++S.tree_branch_hits;
                    cur_hist.push_back(tr.tok[hit]);
                    u = hit;
                }
            }
            d.snap_h(u);
            if (is_branched) r.commit_tree_path(a, path, n);
            if (cfg.check && !prob_mode) {
                // diagnostic: recompute rows 0..n one token at a time and compare them with the window's rows (bitwise); vcache already flushed & frozen before verify
                std::vector<float> wl((size_t)T * V), sl(V);
                HCK(hipMemcpy(wl.data(), r.logits, (size_t)T * V * 4, hipMemcpyDeviceToHost));
                std::vector<int32_t> chk_tok(tok.begin(), tok.begin() + (size_t)a + 1);
                for (int i = 1; i <= n; ++i) chk_tok.push_back(tr.tok[path[i]]);
                for (int i = 0; i <= n; ++i) {
                    if (!r.forward_chunk(chk_tok.data(), a + i, 1, err)) { finish = "error"; break; }
                    r.head(1, 0);
                    HCK(hipStreamSynchronize(r.st));
                    HCK(hipMemcpy(sl.data(), r.logits, (size_t)V * 4, hipMemcpyDeviceToHost));
                    ++S.chk_rows;
                    const int row_idx = path[i];
                    if (!std::equal(sl.begin(), sl.end(), wl.begin() + (size_t)row_idx * V)) {
                        double md = 0;
                        for (int v = 0; v < V; ++v) md = std::max(md, (double)std::fabs(sl[v] - wl[(size_t)row_idx * V + v]));
                        ++S.chk_bad;
                        if (S.chk_bad <= 12) fprintf(stderr, "[check] a=%d (a mod 4 = %d) depth %d node %d of tree T=%d (n=%d, branched=%d): row differs from 1-token row, max|dlogit| %.4g, argmax %d vs %d\n",
                                                     a, a & 3, i, row_idx, T, n, (int)is_branched, md,
                                                     (int)(std::max_element(sl.begin(), sl.end()) - sl.begin()),
                                                     (int)(std::max_element(wl.begin() + (size_t)row_idx * V, wl.begin() + (size_t)(row_idx + 1) * V) - wl.begin() - (size_t)row_idx * V));
                    }
                }
                r.vcache_frozen = chk_fr_was;
                if (finish == "error") break;
            }
            if (want_margin)
                for (int i = 0; i <= n; ++i) note_margin(path[i]);
            // calibration + cost model (calibrate primary children along the traversed path)
            for (int k = 0; k <= n; ++k) {
                const int par = path[k];
                int first_child = -1;
                for (int c = par + 1; c < T; ++c)
                    if (tr.parent[c] == par) { first_child = c; break; }
                if (first_child < 0) continue;
                const bool accepted_primary = (k < n && path[k + 1] == first_child);
                const int b = bucket(node_conf[first_child]);
                cn[b] += 1.0;
                cy[b] += accepted_primary ? 1.0 : 0.0;
                if (cn[b] > 200) { cn[b] *= 0.5; cy[b] *= 0.5; }
            }
            tv[T] = tv[T] > 0 ? 0.7 * tv[T] + 0.3 * tver_ms : tver_ms;
            S.t_verify += tver_ms;
            S.verified_slots += Kp;
            S.accepted += n;
            ++S.hist[std::min(n, 8)];
            ++S.spec_iters;
            if (cfg.debug) fprintf(stderr, "[spec] a=%d T=%d branched=%d n=%d draft %.1f ms verify %.1f ms conf %.2f %.2f %.2f\n", a, T, (int)is_branched, n, tdraft, tver_ms,
                                   T > 1 ? node_conf[1] : 0.f, T > 2 ? node_conf[2] : 0.f, T > 3 ? node_conf[3] : 0.f);
            tok.resize((size_t)a + 1);
            fwd = a + 1 + n;
            bool alive = true;
            for (int i = 1; i <= n && alive; ++i) alive = push_emit(tr.tok[path[i]]);
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
        fprintf(f, "[spec] K=%d %s tree=%d early=%.2f: %ld tokens in %ld iterations (%ld spec + %ld plain), %.2f tokens/iteration, total %.0f ms\n", cfg.K, cfg.adaptive ? "adaptive" : "fixed",
                (int)cfg.tree, cfg.early_cut, S.tokens, steps, S.spec_iters, S.plain_iters, steps ? (double)S.tokens / steps : 0.0, total_ms);
        fprintf(f, "[spec] verify steps %ld: accepted drafts per step %.3f (verified slots %.2f/step, tree_steps %ld, branch_hits %ld, prob_iters %ld, chain_evals %ld, early_exits %ld), draft %.1f ms/step, verify %.1f ms/step, plain %.1f ms/step (EMA %.1f), "
                   "spec iteration %.1f ms\n",
                S.spec_iters, S.spec_iters ? (double)S.accepted / S.spec_iters : 0.0, S.spec_iters ? (double)S.verified_slots / S.spec_iters : 0.0,
                S.tree_steps, S.tree_branch_hits, S.prob_iters, d.n_chain_evals, d.n_early_exits,
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
