// hy4_metal_spec.h - DFly2 drafter (Tree-8 + Early-Exit + Reduced-Vocab) + lossless greedy & probabilistic SpecInfer speculative loop on Metal.
// Port of engine/src/hy4flash/run/hy4_dfly2.hip.h & hy4_spec.hip.h (branch f-hy4sp-dfly); kernels in hy4_dfly2.metal & hy4_dfly2b.metal.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

struct SampleOpts {
    float temperature = 0.f;  // 0 = greedy
    float top_p = 1.f;
    int top_k = 64;
    float min_p = 0.f;
    float rep = 1.f, freq = 0.f, pres = 0.f;
    int last_n = 64;
    uint64_t seed = 0;
    bool greedy() const { return temperature <= 0.f; }
    bool penalties() const { return rep != 1.f || freq != 0.f || pres != 0.f; }
};

using SparseDist = std::vector<std::pair<int32_t, double>>;

// Exact normalized probability distribution from full logits row [V] under SampleOpts + history penalties.
// Matches hy4_metal_serve.h sample_host bit-for-bit: penalties -> top-k (<=64) -> temp softmax -> min_p -> top_p -> renorm.
inline void compute_dist_full(const float* lg_in, int V, const SampleOpts& s, const std::vector<int32_t>& hist, SparseDist& out) {
    out.clear();
    std::unordered_map<int32_t, int> count;
    if (s.penalties() && s.last_n > 0) {
        size_t from = hist.size() > (size_t) s.last_n ? hist.size() - (size_t) s.last_n : 0;
        for (size_t i = from; i < hist.size(); ++i) ++count[hist[i]];
    }
    auto adj = [&](int id) -> float {
        float x = lg_in[id];
        if (!count.empty()) {
            auto it = count.find(id);
            if (it != count.end()) {
                if (s.rep != 1.f) x = x > 0 ? x / s.rep : x * s.rep;
                x -= (float) it->second * s.freq + (s.pres != 0.f ? s.pres : 0.f);
            }
        }
        return x;
    };
    const int K = std::min(std::max(1, s.top_k), 64);
    std::pair<float, int> top[65];
    int ntop = 0;
    for (int i = 0; i < V; ++i) {
        float v = adj(i);
        if (ntop < K || v > top[ntop - 1].first) {
            int pos = ntop < K ? ntop++ : K - 1;
            while (pos > 0 && (v > top[pos - 1].first || (v == top[pos - 1].first && i < top[pos - 1].second))) {
                top[pos] = top[pos - 1];
                --pos;
            }
            top[pos] = {v, i};
        }
    }
    if (ntop == 0) return;
    if (s.greedy()) {
        out.push_back({top[0].second, 1.0});
        return;
    }
    double p[64];
    const double mx = top[0].first;
    double sum = 0.0;
    for (int i = 0; i < ntop; ++i) {
        p[i] = std::exp(((double) top[i].first - mx) / (double) s.temperature);
        sum += p[i];
    }
    for (int i = 0; i < ntop; ++i) p[i] /= sum;
    int keep = ntop;
    if (s.min_p > 0.f) {
        const double thr = (double) s.min_p * p[0];
        keep = 1;
        while (keep < ntop && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= (double) s.top_p) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    for (int i = 0; i < keep; ++i) z += p[i];
    out.reserve(keep);
    for (int i = 0; i < keep; ++i) out.push_back({top[i].second, p[i] / z});
}

inline void compute_dist_cand(const int32_t* ids, const float* scores, int N, const SampleOpts& s, const std::vector<int32_t>& hist, SparseDist& out) {
    out.clear();
    if (N <= 0) return;
    std::unordered_map<int32_t, int> count;
    if (s.penalties() && s.last_n > 0) {
        size_t from = hist.size() > (size_t) s.last_n ? hist.size() - (size_t) s.last_n : 0;
        for (size_t i = from; i < hist.size(); ++i) ++count[hist[i]];
    }
    const int K = std::min(std::min(std::max(1, s.top_k), 64), N);
    std::vector<std::pair<float, int32_t>> cand((size_t) N);
    for (int i = 0; i < N; ++i) {
        float x = scores[i];
        int32_t id = ids[i];
        if (!count.empty()) {
            auto it = count.find(id);
            if (it != count.end()) {
                if (s.rep != 1.f) x = x > 0 ? x / s.rep : x * s.rep;
                x -= (float) it->second * s.freq + (s.pres != 0.f ? s.pres : 0.f);
            }
        }
        cand[i] = {x, id};
    }
    std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    if (s.greedy()) {
        out.push_back({cand[0].second, 1.0});
        return;
    }
    double p[64];
    const double mx = cand[0].first;
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        p[i] = std::exp(((double) cand[i].first - mx) / (double) s.temperature);
        sum += p[i];
    }
    for (int i = 0; i < K; ++i) p[i] /= sum;
    int keep = K;
    if (s.min_p > 0.f) {
        const double thr = (double) s.min_p * p[0];
        keep = 1;
        while (keep < K && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= (double) s.top_p) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    for (int i = 0; i < keep; ++i) z += p[i];
    out.reserve(keep);
    for (int i = 0; i < keep; ++i) out.push_back({cand[i].second, p[i] / z});
}

inline int32_t sample_from_dist(const SparseDist& dist, std::mt19937_64& rng) {
    if (dist.empty()) return 0;
    double z = 0.0;
    for (const auto& kv : dist) z += kv.second;
    if (z <= 0.0) return dist[0].first;
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng), c = 0.0;
    for (const auto& kv : dist) {
        c += kv.second;
        if (r <= c) return kv.first;
    }
    return dist.back().first;
}

inline double dist_prob_of(const SparseDist& dist, int32_t tok) {
    for (const auto& kv : dist)
        if (kv.first == tok) return kv.second;
    return 0.0;
}

constexpr int DFLY_MASK_ID = 120009;
constexpr int DFLY_KMAX = 7;
constexpr int DFLY_BMAX = DFLY_KMAX + 1;
constexpr int DFLY_LAYERS[5] = {25, 29, 33, 37, 41};

struct DFly2 {
    Runner& r;
    const Cfg& c;
    bool ok = false;
    Buffer H, Hf, F1, F2, F3, F4, Ec, Dn, G, U, d_conf, hprev_f, hprev_h, Hf_h, hn, qr_h, qa_h, ctx_h, o_h, mlp_h, hcin, act, hc_h, d_bids, d_chain, d_prev_one, dlist, dnk;
    Buffer d_cand, d_cscore;
    int rv = 256;  // reduced-vocabulary chain: candidates per slot (0 = full lm_head per drafted token)
    int lstride = 0;
    double t_draft = 0;
    long n_draft = 0;
    long n_chain_evals = 0;
    long n_early_exits = 0;

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
        hprev_h = A(3072 * 2);
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
        d_prev_one = A(16);
        lstride = r.index_topk + 8;
        dlist = A((size_t) B * lstride * 4);
        dnk = A(64);
        d_cand = A((size_t) DFLY_BMAX * 1024 * 4);
        d_cscore = A(1024 * 4);
        for (int d = 0; d < 5; ++d) {
            const std::string nm = "model.dfly_layers." + std::to_string(d) + ".self_attn.kv_b_proj.weight";
            const TInfo& t = r.g.ti(nm);
            if (t.fmt != F_S4 && t.fmt != F_S5 && t.fmt != F_S6 && t.fmt != F_S8) { err = "dfly kv_b_proj format not supported"; return false; }
        }
        ok = true;
        return true;
    }

    // Snapshot h_final row `row` (fp16 [3072]) from unified-memory r.hfin into hprev_h
    void snap_h(int row) {
        std::memcpy(hprev_h.ptr, (const uint16_t*) r.hfin.ptr + (size_t) row * 3072, 3072 * 2);
    }

    bool can_draft(int a, int K) const {
        if (!ok || K < 1 || K > DFLY_KMAX) return false;
        if (a + K + 1 > r.slots_sparse) return false;
        if (r.cap_p0 < 0) return false;
        const int row = a - 1 - r.cap_p0;
        return row >= 0 && row < r.cap_rows;
    }

    // Encode the 5 DFly2 backbone layers + batched base-logits top-N candidates into r.ds (caller runs r.run_all() or chains linear loop)
    bool encode_backbone(int a, int xa, int K, int& N_out) {
        if (!can_draft(a, K)) return false;
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
        r.allow_nb = r.relax_ok;
        r.add("dfly_embed", B, 1, 256, {embw, Bf(d_bids), Bf(H)});
        r.add("dfly_h2f", (3072 + 255) / 256, 1, 256, {Bf(hprev_h), Bf(hprev_f), I(3072)});
        r.add("dfly_cat2", 1, 1, 256, {Bf(H), wf(pf + "mtp_fusion.enorm.weight"), Bf(hprev_f), wf(pf + "mtp_fusion.hnorm.weight"), Bf(hcin), Fl(eps)});
        r.lin(pf + "mtp_fusion.eh_proj.weight", hcin, F1, 1);
        r.add("dfly_fuse", 1, 1, 256, {Bf(H), wf(pf + "mtp_fusion.gate"), Bf(F1)});
        for (int d = 0; d < 5; ++d) {
            const int L = DFLY_LAYERS[d];
            const std::string pd = pf + std::to_string(d);
            const std::string pa = pd + ".self_attn", pm = pd + ".mlp";
            const bool use_list = r.cap_list[d];
            const double th = c.theta_sparse;
            const std::string kvb_nm = pa + ".kv_b_proj.weight";
            const int kvb_fmt = r.g.ti(kvb_nm).fmt;
            r.add("rms_rows", B, 1, 256, {Bf(H), wf(pd + ".input_layernorm.weight"), Bf(F1), Bf(hn), I(3072), Fl(eps), I(1 | 4)});
            r.lin(pa + ".q_a_proj.weight", hn, F1, B);
            r.lin(pa + ".kv_a_proj_with_mqa.weight", hn, F3, B);
            r.relax();
            r.lin(pa + ".linear_gate.weight", hn, F4, B);
            r.relax();
            r.add("rms_rows", B, 1, 256, {Bf(F1), wf(pa + ".q_a_layernorm.weight"), Bf(F1), Bf(qr_h), I(1024), Fl(eps), I(1 | 4)});
            r.add("mla_write_kv", B, 1, 256, {Bf(F3), wf(pa + ".kv_a_layernorm.weight"), Bf(r.kcache[L]), I(a), I(0), Fl((float) th), Fl(eps)});
            r.relax();
            r.lin(pa + ".q_b_proj.weight", qr_h, F2, B);
            r.add("mla_prep_q_dec_" + std::to_string(kvb_fmt) + "_" + std::to_string(B), 32, 2, 256,
                  {wf(kvb_nm), Bf(F2), Bf(qa_h), I(B), I(32), I(a), Fl((float) th)});
            MlaArgs ma{};
            ma.M = B;
            ma.NH = 32;
            ma.p0 = a;
            ma.window = c.window;
            ma.ring = 0;
            ma.scale = 0.0625f;
            if (use_list) {
                r.add("dfly_lists", B, 1, 256, {Bf(r.dkeys[d], (size_t) row * r.index_topk * 4), Bf(r.dnkc[d], (size_t) row * 4), I(a), Bf(dlist), Bf(dnk), I(lstride)});
                r.relax();
                ma.mode = MODE_LIST;
                ma.kmax = lstride;
            } else {
                ma.mode = MODE_CAUSAL;
                ma.kmax = 0;
            }
            if (r.split_max > 0) {
                const int S = std::min(r.split_max, 8);
                Dispatch dd;
                dd.p = &r.P("mla_attend_part");
                dd.args = {Bf(qa_h), Bf(r.kcache[L]), wf(pa + ".learnable_sink_param"), Bf(r.attn_ws), Arg::V<MlaArgs>(ma), Bf(dlist), Bf(dnk), I(S)};
                dd.grid[0] = 32; dd.grid[1] = S; dd.grid[2] = B; dd.tg[0] = 256;
                r.ds.push_back(std::move(dd));
                r.add("mla_attend_merge", 32, B, 256, {Bf(r.attn_ws), Bf(ctx_h), Arg::V<MlaArgs>(ma), I(S)});
            } else {
                r.add("mla_attend", 32, B, 256, {Bf(qa_h), Bf(r.kcache[L]), wf(pa + ".learnable_sink_param"), Bf(ctx_h), Arg::V<MlaArgs>(ma), Bf(dlist), Bf(dnk)});
            }
            r.add("mla_vup_dec_" + std::to_string(kvb_fmt) + "_" + std::to_string(B), 32, 32, 256,
                  {wf(kvb_nm), Bf(ctx_h), Bf(F4), Bf(o_h), I(B), I(32)});
            r.lin(pa + ".o_proj.weight", o_h, F1, B);
            r.add("dfly_rms_resid_norm", B, 1, 256, {Bf(F1), wf(pa + ".post_proj_norm.weight"), Bf(H), wf(pd + ".post_attention_layernorm.weight"), Bf(hn), Fl(eps)});
            r.lin(pm + ".gate_proj.weight", hn, F1, B);
            r.lin(pm + ".up_proj.weight", hn, F2, B);
            r.relax();
            const uint32_t n9 = (uint32_t) B * 9216;
            r.add("swiglu_h", (n9 + 255) / 256, 1, 256, {Bf(F1), Bf(F2), Bf(mlp_h), Runner::U(n9)});
            r.lin(pm + ".down_proj.weight", mlp_h, F3, B);
            const uint32_t nh = (uint32_t) B * 3072;
            r.add("add_inplace", (nh + 255) / 256, 1, 256, {Bf(H), Bf(F3), Runner::U(nh)});
        }
        r.add("rms_rows", B, 1, 256, {Bf(H), wf(pf + "final_norm.weight"), Bf(Hf), Bf(Hf_h), I(3072), Fl(eps), I(1 | 2 | 4)});
        const int NC = std::min(rv, 1024);
        N_out = NC;
        if (NC > 0) {
            r.lin("lm_head.weight", Hf_h, r.logits, K, (size_t) 3072 * 2, 0);
            r.add("dfly_topn", K, 1, 1024, {Bf(r.logits), I(c.vocab), I(NC), Bf(d_cand)});
        }
        return true;
    }

    // Fused 1-command-buffer linear draft (used when --no-tree && !prob_mode && early_cut <= 0)
    bool draft_linear(int a, int xa, int K, int* y, float* conf) {
        const double t0 = now_s();
        int NC = 0;
        if (!encode_backbone(a, xa, K, NC)) return false;
        const std::string pf = "model.dfly_layers.";
        const float eps = c.rms_eps;
        auto& I = Runner::I;
        auto& Fl = Runner::Fl;
        auto Bf = [](const Buffer& b, size_t off = 0) { return Arg::B(b, off); };
        const Arg embw = r.g.tw("model.embed_tokens.weight");
        const Arg lmw = r.g.tw("lm_head.weight");
        for (int j = 1; j <= K; ++j) {
            r.add("dfly_embed", 1, 1, 256, {embw, Bf(d_chain, (size_t) (j - 1) * 4), Bf(Ec)});
            r.add("dfly_cat2", 1, 1, 256, {Bf(Hf, (size_t) j * 3072 * 4), wf(pf + "hidden_correction.hidden_norm.weight"), Bf(Ec), wf(pf + "hidden_correction.embed_norm.weight"), Bf(hcin), Fl(eps)});
            r.lin(pf + "hidden_correction.gate_proj.weight", hcin, G, 1);
            r.lin(pf + "hidden_correction.up_proj.weight", hcin, U, 1);
            r.relax();
            r.add("swiglu_h", (3072 + 255) / 256, 1, 256, {Bf(G), Bf(U), Bf(act), Runner::U(3072)});
            r.lin(pf + "hidden_correction.down_proj.weight", act, Dn, 1);
            r.add("dfly_corr", 1, 1, 256, {Bf(Hf, (size_t) j * 3072 * 4), Bf(Dn), wf(pf + "confidence_head.proj.weight"), wf(pf + "confidence_head.proj.bias"), Bf(hc_h), Bf(d_conf, (size_t) j * 4)});
            if (NC > 0) {
                r.add("dfly_cand_score", NC, 1, 32, {lmw, Bf(d_cand, (size_t) (j - 1) * NC * 4), Bf(hc_h), Bf(d_cscore)});
                r.add("dfly_cand_argmax", 1, 1, 1024, {Bf(d_cscore), Bf(d_cand, (size_t) (j - 1) * NC * 4), I(NC), Bf(d_chain, (size_t) j * 4)});
            } else {
                r.lin("lm_head.weight", hc_h, r.logits, 1);
                r.add("argmax_row", 1, 1, 1024, {Bf(r.logits), I(c.vocab), Bf(d_chain, (size_t) j * 4)});
            }
            ++n_chain_evals;
        }
        r.run_all();
        for (int j = 1; j <= K; ++j) {
            y[j] = ((const int*) d_chain.ptr)[j];
            conf[j] = ((const float*) d_conf.ptr)[j];
        }
        t_draft += now_s() - t0;
        ++n_draft;
        return true;
    }

    // Evaluate ONE hidden_correction + cand_score slot j (1..K) given prev_tok. Zero-copy unified-memory read on d_cscore / r.logits / d_conf!
    float eval_slot_host(int j, int32_t prev_tok, int NC) {
        const std::string pf = "model.dfly_layers.";
        const float eps = c.rms_eps;
        auto& Fl = Runner::Fl;
        auto Bf = [](const Buffer& b, size_t off = 0) { return Arg::B(b, off); };
        const Arg embw = r.g.tw("model.embed_tokens.weight");
        const Arg lmw = r.g.tw("lm_head.weight");
        ((int*) d_prev_one.ptr)[0] = prev_tok;
        r.allow_nb = r.relax_ok;
        r.add("dfly_embed", 1, 1, 256, {embw, Bf(d_prev_one), Bf(Ec)});
        r.add("dfly_cat2", 1, 1, 256, {Bf(Hf, (size_t) j * 3072 * 4), wf(pf + "hidden_correction.hidden_norm.weight"), Bf(Ec), wf(pf + "hidden_correction.embed_norm.weight"), Bf(hcin), Fl(eps)});
        r.lin(pf + "hidden_correction.gate_proj.weight", hcin, G, 1);
        r.lin(pf + "hidden_correction.up_proj.weight", hcin, U, 1);
        r.relax();
        r.add("swiglu_h", (3072 + 255) / 256, 1, 256, {Bf(G), Bf(U), Bf(act), Runner::U(3072)});
        r.lin(pf + "hidden_correction.down_proj.weight", act, Dn, 1);
        r.add("dfly_corr", 1, 1, 256, {Bf(Hf, (size_t) j * 3072 * 4), Bf(Dn), wf(pf + "confidence_head.proj.weight"), wf(pf + "confidence_head.proj.bias"), Bf(hc_h), Bf(d_conf)});
        if (NC > 0) {
            r.add("dfly_cand_score", NC, 1, 32, {lmw, Bf(d_cand, (size_t) (j - 1) * NC * 4), Bf(hc_h), Bf(d_cscore)});
        } else {
            r.lin("lm_head.weight", hc_h, r.logits, 1);
        }
        r.run_all();
        ++n_chain_evals;
        return ((const float*) d_conf.ptr)[0];
    }

    // Tree-8 + Early-Exit + Probabilistic-aware drafter (identical algorithm to hy4_dfly2.hip.h).
    bool draft_tree(int a, int xa, int K, TreeSpec& tree, float* node_conf, float* node_reach, const SampleOpts* samp, const std::vector<int32_t>* tok_hist, std::mt19937_64* rng,
                    std::vector<SparseDist>* node_qdist, float early_cut, bool enable_tree, const std::function<double(float)>& pcal_fn) {
        const bool prob_mode = samp != nullptr && (!samp->greedy() || samp->penalties());
        if (!enable_tree && !prob_mode && early_cut <= 0.f) {
            int yl[DFLY_BMAX + 1] = {};
            float cfl[DFLY_BMAX + 1] = {};
            if (!draft_linear(a, xa, K, yl, cfl)) return false;
            tree = TreeSpec{};
            tree.M = K + 1;
            tree.tok[0] = xa;
            tree.parent[0] = -1;
            tree.depth[0] = 0;
            node_conf[0] = 1.0f;
            node_reach[0] = 1.0f;
            double pr = 1.0;
            for (int j = 1; j <= K; ++j) {
                pr *= std::clamp(pcal_fn(cfl[j]), 0.02, 0.995);
                tree.tok[j] = yl[j];
                tree.parent[j] = (int8_t) (j - 1);
                tree.depth[j] = (uint8_t) j;
                node_conf[j] = cfl[j];
                node_reach[j] = (float) pr;
            }
            tree.build_anc();
            return true;
        }
        const double t0 = now_s();
        int NC = 0;
        if (!encode_backbone(a, xa, K, NC)) return false;
        r.run_all();  // d_cand.ptr is now readable in unified memory (zero copy!)
        tree = TreeSpec{};
        tree.M = 1;
        tree.tok[0] = xa;
        tree.parent[0] = -1;
        tree.depth[0] = 0;
        node_conf[0] = 1.0f;
        node_reach[0] = 1.0f;
        if (node_qdist) {
            node_qdist->clear();
            node_qdist->resize(1);
        }
        struct Cand {
            int parent;
            int depth;
            int32_t tok;
            float conf;
            float reach;
            SparseDist qdist;
        };
        std::vector<Cand> frontier;
        std::vector<int32_t> branch_hist;
        const int32_t* h_cand = (const int32_t*) d_cand.ptr;
        const float* h_cscore = (const float*) d_cscore.ptr;
        const float* full_lg = (const float*) r.logits.ptr;

        auto expand = [&](int u) {
            const int d_next = (int) tree.depth[u] + 1;
            if (d_next > K) return;
            const float cf = eval_slot_host(d_next, tree.tok[u], NC);
            const double p_acc = std::clamp(pcal_fn(cf), 0.02, 0.995);
            const float r_u = node_reach[u];
            if (!prob_mode) {
                int32_t id1 = 0, id2 = -1;
                float s1 = -3e38f, s2 = -3e38f;
                if (NC > 0) {
                    const int32_t* cids = h_cand + (size_t) (d_next - 1) * NC;
                    for (int i = 0; i < NC; ++i) {
                        const float sc = h_cscore[i];
                        const int32_t id = cids[i];
                        if (sc > s1 || (sc == s1 && id < id1)) { s2 = s1; id2 = id1; s1 = sc; id1 = id; }
                        else if (id != id1 && (sc > s2 || (sc == s2 && id < id2))) { s2 = sc; id2 = id; }
                    }
                } else {
                    for (int i = 0; i < c.vocab; ++i) {
                        const float sc = full_lg[i];
                        if (sc > s1) { s2 = s1; id2 = id1; s1 = sc; id1 = i; }
                        else if (sc > s2) { s2 = sc; id2 = i; }
                    }
                }
                const float r1 = (float) (r_u * p_acc);
                frontier.push_back({u, d_next, id1, cf, r1, {}});
                if (enable_tree && id2 >= 0 && id2 != id1) {
                    const double gap = std::max(0.0, (double) (s1 - s2));
                    const double ratio2 = std::exp(-gap);
                    const double p2 = std::min(1.0 - p_acc, p_acc * ratio2 * 0.85);
                    if (p2 > 0.05) {
                        frontier.push_back({u, d_next, id2, (float) std::clamp(p2, 0.01, 0.95), (float) (r_u * p2), {}});
                    }
                }
            } else {
                const std::vector<int32_t>* hptr = tok_hist;
                if (samp->penalties() && tok_hist) {
                    branch_hist = *tok_hist;
                    int chain_nodes[8], nc = 0;
                    for (int cur = u; cur > 0; cur = tree.parent[cur]) chain_nodes[nc++] = cur;
                    for (int k = nc - 1; k >= 0; --k) branch_hist.push_back(tree.tok[chain_nodes[k]]);
                    hptr = &branch_hist;
                }
                static const std::vector<int32_t> empty_h;
                SparseDist q1;
                if (NC > 0) compute_dist_cand(h_cand + (size_t) (d_next - 1) * NC, h_cscore, NC, *samp, hptr ? *hptr : empty_h, q1);
                else compute_dist_full(full_lg, c.vocab, *samp, hptr ? *hptr : empty_h, q1);
                if (q1.empty()) return;
                const int32_t y1 = sample_from_dist(q1, *rng);
                const float r1 = (float) (r_u * p_acc);
                frontier.push_back({u, d_next, y1, cf, r1, q1});
                if (enable_tree && q1.size() >= 2) {
                    SparseDist q2;
                    double z2 = 0.0;
                    for (const auto& kv : q1)
                        if (kv.first != y1) z2 += kv.second;
                    if (z2 > 0.08) {
                        q2.reserve(q1.size() - 1);
                        for (const auto& kv : q1)
                            if (kv.first != y1) q2.push_back({kv.first, kv.second / z2});
                        const int32_t y2 = sample_from_dist(q2, *rng);
                        const double p2 = std::min(1.0 - p_acc, z2 * 0.75);
                        if (p2 > 0.05) {
                            frontier.push_back({u, d_next, y2, (float) std::clamp(p2, 0.01, 0.95), (float) (r_u * p2), std::move(q2)});
                        }
                    }
                }
            }
        };

        expand(0);
        const int max_nodes = std::min(K + 1, DFLY_BMAX);
        while (tree.M < max_nodes && !frontier.empty()) {
            int best_i = 0;
            for (int i = 1; i < (int) frontier.size(); ++i)
                if (frontier[i].reach > frontier[best_i].reach) best_i = i;
            if (early_cut > 0.f && tree.M > 1 && frontier[best_i].reach < early_cut) {
                ++n_early_exits;
                break;
            }
            Cand chosen = std::move(frontier[best_i]);
            frontier.erase(frontier.begin() + best_i);
            const int idx = tree.M++;
            tree.tok[idx] = chosen.tok;
            tree.parent[idx] = (int8_t) chosen.parent;
            tree.depth[idx] = (uint8_t) chosen.depth;
            node_conf[idx] = chosen.conf;
            node_reach[idx] = chosen.reach;
            if (node_qdist) node_qdist->push_back(std::move(chosen.qdist));
            if (tree.M < max_nodes && chosen.depth < K && (early_cut <= 0.f || chosen.reach >= early_cut)) {
                expand(idx);
            }
        }
        tree.build_anc();
        t_draft += now_s() - t0;
        ++n_draft;
        return true;
    }
};

struct SpecCfg {
    int K = 5;               // drafted tokens per block (1..7, fits M<=8 Tree-8 tile; K=5 best speed on M5 Max)
    bool adaptive = true;    // false: always verify full drafted tree (fixed window, no back-off)
    float cut = -1.f;        // >= 0: verify leading nodes with conf > cut instead of cost model
    bool debug = false;
    bool check = false;      // diagnostic: recompute accepted rows 1 token at a time and assert bitwise equality
    bool sabotage = false;   // test only: random draft ids with conf 0.9
    int rv = 256;            // reduced-vocabulary draft chain: candidates per slot (0 = full lm_head)
    bool tree = true;        // Tree-8 speculative verification (top-2 branching within M<=8 verify tile)
    float early_cut = 0.08f; // early-exit draft chain when max frontier reach < early_cut (0 = disabled)
};

struct SpecStats {
    long chk_rows = 0, chk_bad = 0;
    long spec_iters = 0, plain_iters = 0, plain_after_draft = 0, tokens = 0, accepted = 0, verified_slots = 0, drafted = 0, probes = 0, backoffs = 0;
    long tree_steps = 0, tree_branch_hits = 0, prob_iters = 0;
    long hist[9] = {};
    double t_draft = 0, t_verify = 0, t_plain = 0, t_spec_total = 0, g_plain = 0, b_plain = 0, g_verify = 0, e_plain = 0, e_verify = 0;
    long n_plainstep = 0;
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
    float node_conf[DFLY_BMAX] = {};
    float node_reach[DFLY_BMAX] = {};
    std::vector<SparseDist> node_qdist;

    Spec(Runner& rr, DFly2& dd, const SpecCfg& c) : r(rr), d(dd), cfg(c), V(rr.c.vocab) {
        for (int b = 0; b < 6; ++b) { cn[b] = 2.0; cy[b] = 2.0 * MID[b]; }
        if (cfg.K > DFLY_KMAX) cfg.K = DFLY_KMAX;
        if (cfg.K < 1) cfg.K = 1;
        plain_left = 2;  // calibrate plain step time first
        d.rv = cfg.rv;
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
    double tver(int m, bool det = false) const {
        if (!det && tv[m] > 0) return tv[m];
        const double base = (!det && tp > 0) ? tp : 24.0;
        return base * (5.1 + 0.127 * DM[m - 1]) / 7.13;
    }
    int choose_m(const TreeSpec& tr, bool det = false) const {
        if (!cfg.adaptive) return tr.M;
        if (cfg.cut >= 0) {
            int m = 1;
            while (m < tr.M && node_conf[m] > cfg.cut) ++m;
            return m;
        }
        const double tpl = (!det && tp > 0) ? tp : 24.0;
        int best_m = 1;
        double bt = 1.0 / tpl, E = 0.0;
        for (int m = 2; m <= tr.M; ++m) {
            E += (double) node_reach[m - 1];
            const double thr = (1.0 + E) / tver(m, det);
            if (thr > bt) { bt = thr; best_m = m; }
        }
        return best_m;
    }

    int argmax_host_row0() {
        r.add("argmax_row", 1, 1, 1024, {Runner::B(r.logits), Runner::I(V), Runner::B(r.d_argmax)});
        r.run_all();
        return ((const int*) r.d_argmax.ptr)[0];
    }

    int sample_host_row0(const SampleOpts* samp, const std::vector<int32_t>& hist, std::mt19937_64* rng) {
        if (!samp || (samp->greedy() && !samp->penalties())) return argmax_host_row0();
        SparseDist dist;
        compute_dist_full((const float*) r.logits.ptr, V, *samp, hist, dist);
        return sample_from_dist(dist, *rng);
    }

    bool plain_step(std::vector<int32_t>& tok, int a, int& nx, std::string& err, const SampleOpts* samp = nullptr, std::mt19937_64* rng = nullptr) {
        const double g0 = r.gpu_total, e0 = r.t_engram_wait, tb0 = now_s();
        if (!r.forward_chunk(tok.data(), a, 1, err)) return false;
        r.head(1, 0);
        const bool gfast = !samp || (samp->greedy() && !samp->penalties());
        if (gfast) r.add("argmax_row", 1, 1, 1024, {Runner::B(r.logits), Runner::I(V), Runner::B(r.d_argmax)});
        S.b_plain += now_s() - tb0;
        r.run_all();
        d.snap_h(0);
        if (gfast) nx = ((const int*) r.d_argmax.ptr)[0];
        else {
            SparseDist dist;
            compute_dist_full((const float*) r.logits.ptr, V, *samp, tok, dist);
            nx = sample_from_dist(dist, *rng);
        }
        S.g_plain += r.gpu_total - g0;
        S.e_plain += r.t_engram_wait - e0;
        ++S.n_plainstep;
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

    // Generates greedily (samp == nullptr or greedy & no penalties) or with exact SpecInfer probabilistic rejection sampling.
    int run(std::vector<int32_t>& tok, int limit, int max_ctx, int eos, int eot, const std::function<bool(int)>& emit, std::string& finish, std::string& err, int& made_out,
            const SampleOpts* samp = nullptr, std::mt19937_64* rng = nullptr) {
        int made = 0;
        finish = "length";
        err.clear();
        const bool prob_mode = samp != nullptr && (!samp->greedy() || samp->penalties());
        const bool det_seed = prob_mode && samp->seed != 0;
        if (det_seed) {
            // Wall-clock-independent repeatability for seeded SpecInfer requests (same seed -> bit-identical token stream every time)
            for (int b = 0; b < 6; ++b) { cn[b] = 2.0; cy[b] = 2.0 * MID[b]; }
            plain_left = 0;
            win.clear();
            P = 8;
        }
        std::mt19937_64 local_rng(samp && samp->seed ? samp->seed : 0x57A7A123456789ABull);
        if (prob_mode && !rng) rng = &local_rng;
        int fwd = (int) tok.size();
        d.snap_h(0);
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
                if (!plain_step(tok, a, nx, err, samp, rng)) { finish = "error"; break; }
                const double dt = (now_s() - ti0) * 1000.0;
                tp = tp > 0 ? 0.8 * tp + 0.2 * dt : dt;
                S.t_plain += dt;
                ++S.plain_iters;
                fwd = a + 1;
                if (!push_emit(nx)) break;
                continue;
            }
            const double td0 = now_s();
            TreeSpec tr;
            const float ecut = (cfg.adaptive && cfg.cut < 0 && !cfg.sabotage) ? cfg.early_cut : 0.f;
            if (!d.draft_tree(a, tok[a], Kd, tr, node_conf, node_reach, samp, &tok, rng, prob_mode ? &node_qdist : nullptr, ecut, cfg.tree && !cfg.sabotage, pcal_fn)) {
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
                    tr.tok[j] = (int) (xs % 100000);
                    tr.parent[j] = (int8_t) (j - 1);
                    tr.depth[j] = (uint8_t) j;
                    node_conf[j] = 0.9f;
                    node_reach[j] = (float) std::pow(0.9, j);
                }
                tr.build_anc();
            }
            S.t_draft += tdraft;
            S.drafted += (tr.M - 1);
            int T = choose_m(tr, det_seed);
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
            const double tv0 = now_s(), g0 = r.gpu_total, e0 = r.t_engram_wait;
            if (!is_branched) {
                for (int i = 1; i < T; ++i) tok.push_back(tr.tok[i]);
                if (!r.forward_chunk(tok.data(), a, T, err, -1, 0, nullptr)) { finish = "error"; break; }
            } else {
                if (!r.forward_chunk(tok.data(), a, T, err, -1, 0, &tr)) { finish = "error"; break; }
            }
            r.head(T, 0);
            int path[9] = {};
            int n = 0, u = 0, nxt = 0;
            double tver_ms = 0.0;
            if (!prob_mode) {
                r.add("argmax_rows", T, 1, 1024, {Runner::B(r.logits), Runner::I(V), Runner::B(r.d_argmax)});
                { static int pc = 0; if (cfg.debug && T == 4 && ++pc == 5) r.kernel_profile(10); }
                r.run_all();
                tver_ms = (now_s() - tv0) * 1000.0;
                const int* h_arg = (const int*) r.d_argmax.ptr;
                path[0] = 0;
                for (;;) {
                    const int target_tok = h_arg[u];
                    int hit = -1;
                    for (int c = u + 1; c < T; ++c)
                        if (tr.parent[c] == u && tr.tok[c] == target_tok) { hit = c; break; }
                    if (hit < 0) break;
                    ++n;
                    path[n] = hit;
                    if (hit != n) ++S.tree_branch_hits;
                    u = hit;
                }
                nxt = h_arg[u];
            } else {
                ++S.prob_iters;
                r.run_all();
                tver_ms = (now_s() - tv0) * 1000.0;
                const float* vlogits = (const float*) r.logits.ptr;
                std::vector<int32_t> cur_hist(tok.begin(), tok.begin() + (size_t) a + 1);
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                path[0] = 0;
                for (;;) {
                    SparseDist p_res;
                    compute_dist_full(vlogits + (size_t) u * V, V, *samp, cur_hist, p_res);
                    int hit = -1;
                    for (int c = u + 1; c < T; ++c) {
                        if (tr.parent[c] != u) continue;
                        const int32_t y_tok = tr.tok[c];
                        const SparseDist& q_c = node_qdist[(size_t) c];
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
                            compute_dist_full(vlogits + (size_t) u * V, V, *samp, cur_hist, p_res);
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
                std::vector<float> wl((const float*) r.logits.ptr, (const float*) r.logits.ptr + (size_t) T * V);
                std::vector<int32_t> chk_tok(tok.begin(), tok.begin() + (size_t) a + 1);
                for (int i = 1; i <= n; ++i) chk_tok.push_back(tr.tok[path[i]]);
                for (int i = 0; i <= n; ++i) {
                    if (!r.forward_chunk(chk_tok.data(), a + i, 1, err)) { finish = "error"; break; }
                    r.head(1, 0);
                    r.run_all();
                    const float* sl = (const float*) r.logits.ptr;
                    ++S.chk_rows;
                    const int row_idx = path[i];
                    if (std::memcmp(sl, &wl[(size_t) row_idx * V], (size_t) V * 4) != 0) {
                        double md = 0;
                        for (int v = 0; v < V; ++v) md = std::max(md, (double) std::fabs(sl[v] - wl[(size_t) row_idx * V + v]));
                        ++S.chk_bad;
                        if (S.chk_bad <= 12) fprintf(stderr, "[check] a=%d (a mod 4 = %d) depth %d node %d of tree T=%d (n=%d, branched=%d): row differs from 1-token row, max|dlogit| %.4g\n",
                                                     a, a & 3, i, row_idx, T, n, (int) is_branched, md);
                    }
                }
                if (finish == "error") break;
            }
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
            S.g_verify += r.gpu_total - g0;
            S.e_verify += r.t_engram_wait - e0;
            S.verified_slots += Kp;
            S.accepted += n;
            ++S.hist[std::min(n, 8)];
            ++S.spec_iters;
            if (cfg.debug) fprintf(stderr, "[spec] a=%d T=%d branched=%d n=%d draft %.1f ms verify %.1f ms (gpu %.1f engram %.1f) conf %.2f %.2f\n",
                                   a, T, (int) is_branched, n, tdraft, tver_ms, (r.gpu_total - g0) * 1e3, (r.t_engram_wait - e0) * 1e3,
                                   T > 1 ? node_conf[1] : 0.f, T > 2 ? node_conf[2] : 0.f);
            tok.resize((size_t) a + 1);
            fwd = a + 1 + n;
            bool alive = true;
            for (int i = 1; i <= n && alive; ++i) alive = push_emit(tr.tok[path[i]]);
            if (alive) alive = push_emit(nxt);
            fwd = std::min(fwd, (int) tok.size());
            const double itms = (now_s() - ti0) * 1000.0;
            S.t_spec_total += itms;
            if (!det_seed) note_iter(itms, n + 1, true);
            if (!alive) break;
        }
        made_out = made;
        return std::min(fwd, (int) tok.size());
    }

    void report(FILE* f, double total_ms) const {
        const long steps = S.spec_iters + S.plain_iters;
        fprintf(f, "[spec] K=%d %s tree=%d early=%.2f: %ld tokens in %ld iterations (%ld spec + %ld plain), %.2f tokens/iteration, total %.0f ms\n",
                cfg.K, cfg.adaptive ? "adaptive" : "fixed", (int) cfg.tree, cfg.early_cut, S.tokens, steps,
                S.spec_iters, S.plain_iters, steps ? (double) S.tokens / steps : 0.0, total_ms);
        fprintf(f, "[spec] verify steps %ld: accepted drafts per step %.3f (verified slots %.2f/step, tree_steps %ld, branch_hits %ld, prob_iters %ld, chain_evals %ld, early_exits %ld), draft %.1f ms/step, verify %.1f ms/step, plain %.1f ms/step (EMA %.1f), spec iteration %.1f ms\n",
                S.spec_iters, S.spec_iters ? (double) S.accepted / S.spec_iters : 0.0, S.spec_iters ? (double) S.verified_slots / S.spec_iters : 0.0,
                S.tree_steps, S.tree_branch_hits, S.prob_iters, d.n_chain_evals, d.n_early_exits,
                S.spec_iters ? S.t_draft / S.spec_iters : 0.0, S.spec_iters ? S.t_verify / S.spec_iters : 0.0,
                S.plain_iters ? S.t_plain / (S.plain_iters + S.plain_after_draft) : 0.0, tp, S.spec_iters ? S.t_spec_total / S.spec_iters : 0.0);
        fprintf(f, "[spec] GPU time per plain step %.1f ms (wall %.1f, host build %.1f ms, engram wait %.1f ms), per verify step %.1f ms (engram wait %.1f ms)\n",
                S.n_plainstep ? S.g_plain / S.n_plainstep * 1e3 : 0.0, S.n_plainstep ? S.t_plain / S.n_plainstep : 0.0,
                S.n_plainstep ? S.b_plain / S.n_plainstep * 1e3 : 0.0, S.n_plainstep ? S.e_plain / S.n_plainstep * 1e3 : 0.0,
                S.spec_iters ? S.g_verify / S.spec_iters * 1e3 : 0.0, S.spec_iters ? S.e_verify / S.spec_iters * 1e3 : 0.0);
        fprintf(f, "[spec] accepted histogram (0..8):");
        for (int i = 0; i < 9; ++i) fprintf(f, " %ld", S.hist[i]);
        fprintf(f, "\n[spec] backoff state: backoffs %ld, probes %ld, plain-mode iterations %ld, plain after draft (K'=0) %ld, P now %d (last %d)\n",
                S.backoffs, S.probes, S.plain_mode_iters, S.plain_after_draft, P, S.P_final);
        fprintf(f, "[spec] verify ms by m:");
        for (int m = 2; m <= 8; ++m) fprintf(f, " m%d=%.1f", m, tv[m]);
        fprintf(f, "\n");
        if (cfg.check) fprintf(f, "[spec] check: %ld window rows recomputed one token at a time, %ld differ bitwise\n", S.chk_rows, S.chk_bad);
    }
};
