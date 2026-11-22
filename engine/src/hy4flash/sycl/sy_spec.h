// sy_spec.h - lossless greedy & probabilistic (SpecInfer) Tree-8 speculative decoding with DFly2 on Intel Arc (SYCL).
// Supports:
//   * Tree-8 speculative verification (top-2 branching at uncertain slots within M <= 8 verify tile)
//   * Early-exit draft chain when best remaining frontier reach probability < early_cut
//   * Lossless probabilistic SpecInfer rejection sampling (temperature > 0 / top_p / top_k / min_p / penalties)
//   * Adaptive cost-model window selection + outer back-off + bitwise batch-invariant verification
#pragma once
#include <deque>
#include <functional>

#include "sy_dfly2.h"

namespace hy4s {

// Speculative decoding verifies windows of 2..8 rows: make their attention split the keys like the 1-token step so that a row has the same fp32 summation order in a window and in plain decode.
inline void spec_set_batch_invariant() {
    const char* e = std::getenv("HY4S_BATCH_INV");
    g_mla_batch_invariant = (e && e[0] == '0') ? 0 : 1;
}

struct SpecCfg {
    int K = 5;               // drafted tokens per block (1..7; K=5 matches hybrid Arc sweet spot, override via --spec-k)
    bool adaptive = true;    // false: always verify K (fixed window, no back-off)
    float cut = -1.f;        // >= 0: verify leading nodes with conf > cut instead of the cost model
    bool debug = false;
    bool plain_only = false; // --spec-plain-only
    bool check = false;      // --spec-check: recompute accepted rows 1-by-1 and compare bitwise
    int rv = 256;            // reduced-vocabulary draft chain: candidates per slot (0 = full lm_head)
    bool tree = true;        // Tree-8 speculative verification (top-2 branching within M <= 8 tile)
    float early_cut = 0.08f; // early-exit draft chain when best frontier reach < early_cut (0 = off)
};

struct SpecStats {
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
    double tp = 0;      // plain step ms (EMA)
    double tv[9] = {};  // verify ms by m (EMA), 0 = not measured
    static constexpr double DM[8] = {16.0, 26.5, 35.6, 43.9, 51.4, 58.2, 64.8, 70.8};
    static constexpr float EDGE[5] = {0.2f, 0.4f, 0.6f, 0.8f, 0.9f};
    static constexpr float MID[6] = {0.1f, 0.3f, 0.5f, 0.7f, 0.85f, 0.95f};
    double cn[6], cy[6];
    struct Iter { double ms; int tok; };
    std::deque<Iter> win;
    int plain_left = 0, P = 8;
    bool probing = false;
    int32_t* h_arg = nullptr;
    float node_conf[DFLY_BMAX] = {};
    float node_reach[DFLY_BMAX] = {};
    std::vector<float> h_vlogits;
    std::vector<SparseDist> node_qdist;

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
    double tver(int m) const {
        if (tv[m] > 0) return tv[m];
        const double base = tp > 0 ? tp : 46.0;
        return base * (5.1 + 0.127 * DM[m - 1]) / 7.13;
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

    int argmax_host_row0() {
        argmax_rows(r.logits, V, r.d_argmax, 1);
        d2h(h_arg, r.d_argmax, 4);
        wait();
        return h_arg[0];
    }

    int sample_host_row0(const SampleOpts* samp, const std::vector<int32_t>& hist, std::mt19937_64* rng) {
        if (!samp || (samp->greedy() && !samp->penalties())) return argmax_host_row0();
        if ((int)h_vlogits.size() < V) h_vlogits.resize((size_t)V);
        d2h(h_vlogits.data(), r.logits, (size_t)V * 4);
        wait();
        SparseDist dist;
        compute_dist_full(h_vlogits.data(), V, *samp, hist, dist);
        return sample_from_dist(dist, *rng);
    }

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

    // Generates greedily (samp == nullptr or greedy & no penalties) or with lossless probabilistic SpecInfer rejection sampling.
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
        const bool clamp32 = std::getenv("HY4S_CLAMP32") && std::getenv("HY4S_CLAMP32")[0] == '1';
        for (;;) {
            const int a = (int)tok.size() - 1;
            fwd = a;
            if (a + 2 >= max_ctx) { finish = "length"; break; }
            const double ti0 = now_s();
            int Kd = cfg.K;
            if (a + Kd + 2 >= max_ctx) Kd = max_ctx - a - 3;
            if (clamp32 && !cfg.check && r.vcache && r.vcache->adaptive()) {
                const int next_b = ((a >> 5) + 1) << 5;
                if (a + 1 + Kd > next_b) Kd = std::max(0, next_b - a - 1);
            }
            const bool use_plain = cfg.plain_only || plain_left > 0 || Kd < 1 || !d.can_draft(a, Kd);
            if (use_plain) {
                if (plain_left > 0) { --plain_left; ++S.plain_mode_iters; if (plain_left == 0) { probing = true; win.clear(); ++S.probes; } }
                int nx = 0;
                if (!plain_step(tok, a, nx, err, samp, rng)) { finish = "error"; break; }
                const double dt = (now_s() - ti0) * 1000.0;
                tp = tp > 0 ? 0.8 * tp + 0.2 * dt : dt;
                S.t_plain += dt;
                ++S.plain_iters;
                fwd = a + 1;
                if (!cfg.check) r.commit_vram_adapt(1, a + 1);
                if (!push_emit(nx)) break;
                continue;
            }
            // Draft (Tree-8 + Early-Exit + Probabilistic-aware)
            const double td0 = now_s();
            TreeSpec tr;
            const float ecut = (cfg.adaptive && cfg.cut < 0) ? cfg.early_cut : 0.f;
            if (!d.draft_tree(a, tok[a], Kd, tr, node_conf, node_reach, samp, &tok, rng, prob_mode ? &node_qdist : nullptr, ecut, cfg.tree, pcal_fn)) {
                finish = "error";
                err = "draft failed";
                break;
            }
            const double tdraft = (now_s() - td0) * 1000.0;
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
                if (!cfg.check) r.commit_vram_adapt(1, a + 1);
                if (!push_emit(nx)) break;
                continue;
            }
            bool is_branched = false;
            for (int i = 1; i < T; ++i)
                if (tr.parent[i] != i - 1) { is_branched = true; break; }
            if (is_branched) ++S.tree_steps;

            // Verify tree (or linear fast path when !is_branched)
            const double tv0 = now_s();
            std::vector<std::pair<std::string, std::vector<float>>> wtap;
            const bool dfly_cap_saved = r.dfly_cap;
            if (cfg.check && !prob_mode) r.tap = &wtap;
            if (!is_branched) {
                for (int i = 1; i < T; ++i) tok.push_back(tr.tok[i]);
                if (!r.forward_chunk(tok.data(), a, T, err, nullptr)) { finish = "error"; break; }
            } else {
                if (!r.forward_chunk(tok.data(), a, T, err, &tr)) { finish = "error"; break; }
            }
            r.tap = nullptr;
            r.head(T, 0);

            int path[9] = {};
            int n = 0;
            int u = 0;
            int nxt = 0;
            double tver_ms = 0.0;
            std::vector<float> wlog;
            if (!prob_mode) {
                if (cfg.check) {
                    wlog.resize((size_t)T * V);
                    d2h(wlog.data(), r.logits, wlog.size() * 4);
                    wait();
                }
                argmax_rows(r.logits, V, r.d_argmax, T);
                d2h(h_arg, r.d_argmax, (size_t)T * 4);
                wait();
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
                d2h(h_vlogits.data(), r.logits, (size_t)T * V * 4);
                wait();
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

            // Calibrate confidence buckets on primary accepted/rejected children along the traversed path
            for (int depth = 1; depth <= n; ++depth) {
                const int b = bucket(node_conf[path[depth]]);
                cn[b] += 1.0;
                cy[b] += 1.0;
                if (cn[b] > 200) { cn[b] *= 0.5; cy[b] *= 0.5; }
            }
            for (int c = u + 1; c < T; ++c) {
                if (tr.parent[c] == u) {
                    const int b = bucket(node_conf[c]);
                    cn[b] += 1.0;
                    if (cn[b] > 200) { cn[b] *= 0.5; cy[b] *= 0.5; }
                    break;
                }
            }

            tv[T] = tv[T] > 0 ? 0.7 * tv[T] + 0.3 * tver_ms : tver_ms;
            S.t_verify += tver_ms;
            S.verified_slots += Kp;
            S.accepted += n;
            ++S.hist[std::min(n, 8)];
            ++S.spec_iters;

            if (cfg.check && !prob_mode) {
                r.dfly_cap = false;
                static long nchecks = 0, nbad = 0;
                std::vector<int32_t> chk_tok(tok.begin(), tok.begin() + (size_t)a + 1);
                for (int i = 1; i <= n; ++i) chk_tok.push_back(tr.tok[path[i]]);
                for (int i = 0; i <= n; ++i) {
                    const int node = path[i];
                    std::vector<std::pair<std::string, std::vector<float>>> stap;
                    r.tap = &stap;
                    if (!r.forward_chunk(chk_tok.data(), a + i, 1, err)) { finish = "error"; break; }
                    r.tap = nullptr;
                    r.head(1, 0);
                    std::vector<float> sl((size_t)V);
                    d2h(sl.data(), r.logits, (size_t)V * 4);
                    wait();
                    size_t lb = 0;
                    for (int e = 0; e < V; ++e) if (std::memcmp(&sl[e], &wlog[(size_t)node * V + e], 4) != 0) ++lb;
                    ++nchecks;
                    if (lb) {
                        ++nbad;
                        std::string first = "none";
                        for (size_t k = 0; k < stap.size() && k < wtap.size(); ++k) {
                            const size_t per = stap[k].second.size();
                            if (stap[k].first != wtap[k].first || wtap[k].second.size() < (size_t)(node + 1) * per) continue;
                            size_t nb = 0;
                            for (size_t e = 0; e < per; ++e) {
                                const float vx = stap[k].second[e], vy = wtap[k].second[(size_t)node * per + e];
                                if (std::memcmp(&vx, &vy, 4) != 0) ++nb;
                            }
                            if (nb) { first = stap[k].first + " (" + std::to_string(nb) + " of " + std::to_string(per) + ")"; break; }
                        }
                        fprintf(stderr, "[spec-check] MISMATCH a=%d step=%d node=%d of T=%d (branch=%d, n=%d): %zu/%d logits differ; first: %s\n",
                                a, i, node, T, (int)is_branched, n, lb, V, first.c_str());
                    }
                }
                r.dfly_cap = dfly_cap_saved;
                if (nchecks % 20 == 0) fprintf(stderr, "[spec-check] %ld rows checked, %ld mismatches so far\n", nchecks, nbad);
            }

            if (cfg.debug)
                fprintf(stderr, "[spec] a=%d T=%d branch=%d n=%d draft %.1f ms verify %.1f ms conf1 %.2f\n",
                        a, T, (int)is_branched, n, tdraft, tver_ms, T > 1 ? node_conf[1] : 0.f);

            tok.resize((size_t)a + 1);
            fwd = a + 1 + n;
            bool alive = true;
            for (int i = 1; i <= n && alive; ++i) alive = push_emit(tr.tok[path[i]]);
            if (alive) alive = push_emit(nxt);
            fwd = std::min(fwd, (int)tok.size());
            const double itms = (now_s() - ti0) * 1000.0;
            S.t_spec_total += itms;
            note_iter(itms, n + 1, true);
            if (!cfg.check) r.commit_vram_adapt_path(path, n + 1, a + n + 1);
            if (!alive) break;
        }
        made_out = made;
        return std::min(fwd, (int)tok.size());
    }

    void report(FILE* f, double total_ms) const {
        if (cfg.check) fprintf(f, "[spec-check] done (mismatch lines above, if any)\n");
        const long steps = S.spec_iters + S.plain_iters;
        fprintf(f, "[spec] K=%d %s tree=%d early=%.2f: %ld tokens in %ld iterations (%ld spec + %ld plain), %.2f tokens/iteration, total %.0f ms\n",
                cfg.K, cfg.adaptive ? "adaptive" : "fixed", (int)cfg.tree, cfg.early_cut, S.tokens, steps, S.spec_iters, S.plain_iters,
                steps ? (double)S.tokens / steps : 0.0, total_ms);
        fprintf(f, "[spec] verify steps %ld (tree %ld, branch hits %ld, prob %ld, early exits %ld): accepted %.3f/step (verified %.2f/step), draft %.1f ms, verify %.1f ms, plain %.1f ms (EMA %.1f), iter %.1f ms\n",
                S.spec_iters, S.tree_steps, S.tree_branch_hits, S.prob_iters, d.n_early_exits,
                S.spec_iters ? (double)S.accepted / S.spec_iters : 0.0, S.spec_iters ? (double)S.verified_slots / S.spec_iters : 0.0,
                S.spec_iters ? S.t_draft / S.spec_iters : 0.0, S.spec_iters ? S.t_verify / S.spec_iters : 0.0,
                (S.plain_iters + S.plain_after_draft) ? S.t_plain / (S.plain_iters + S.plain_after_draft) : 0.0, tp,
                S.spec_iters ? S.t_spec_total / S.spec_iters : 0.0);
        fprintf(f, "[spec] accepted histogram (0..8):");
        for (int i = 0; i < 9; ++i) fprintf(f, " %ld", S.hist[i]);
        fprintf(f, "\n[spec] backoff state: backoffs %ld, probes %ld, plain-mode %ld, plain-after-draft %ld, P now %d (last %d)\n",
                S.backoffs, S.probes, S.plain_mode_iters, S.plain_after_draft, P, S.P_final);
        fprintf(f, "[spec] verify ms by m:");
        for (int m = 2; m <= 8; ++m) fprintf(f, " m%d=%.1f", m, tv[m]);
        fprintf(f, "\n");
    }
};

}  // namespace hy4s
