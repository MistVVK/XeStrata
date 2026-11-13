// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/spec/draft_policy.cpp - see include/strata/spec/draft_policy.hpp.
#include "strata/spec/draft_policy.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <cstdlib>

namespace strata::spec {
namespace {

// The shape of a round's cost by window size, relative to one token, used only for sizes not measured yet (the
// measured round times replace it). Between the RTX 5070's measured curves: +10 ms per token with every missed expert
// on the CPU (bench/results/2026-09-27-spec/window-cost), flatter with the default CPU/DMA split.
constexpr double kShape[DraftPolicy::kMaxT + 1] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6};
constexpr double kCostAlpha = 0.1;    // EMA weight of a new round time
constexpr double kTokAlpha = 0.05;    // EMA weight of a new MTP window outcome
constexpr double kDecay = 0.97;       // lookup counts: older windows fade
// Before a bucket has data: the longer the match, the likelier its continuation (llama.cpp's lookup decoding gates
// on the same thing); worth 4 observations, so a few real windows override it
constexpr double kPriorQ[DraftPolicy::kBuckets] = {0.75, 0.88, 0.93, 0.96};
constexpr double kPriorN = 4.0;
constexpr int kProbes = 3;       // a lookup window size is tried this often before its guessed cost can veto it
// A size not measured for this many rounds is tried again when a confident lookup could use it: its cost was set in
// one stretch (a prompt's cache refill, say) and, priced high, it would never be measured again (#1252)
constexpr double kStaleRounds = 300.0;
// The MTP's window: a draft's acceptance before any is observed (the same 0.7 as prior_tokens), and the chance the
// drafts must have of holding for a size whose cost is only guessed to be tried (as chain()'s 0.6 for a lookup)
constexpr double kDraftPrior = 0.7;
constexpr double kProbeHeld = 0.5;

}  // namespace

DraftPolicy::DraftPolicy(int max_t, double margin)
    : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin) {
    const char* v = strata::env_get("SPEC_COST");
    if (v == nullptr || v[0] == '\0' || v[0] == 'a' || v[0] == 'A')
        mode_ = Mode::kAuto;
    else
        mode_ = std::strtol(v, nullptr, 10) != 0 ? Mode::kOn : Mode::kOff;
}

int DraftPolicy::ctx_bucket(int64_t pos) { return pos < 6144 ? 0 : pos < 24576 ? 1 : 2; }

void DraftPolicy::set_context(int64_t pos) {
    pos_ = pos;
    const int b = cost_aware() ? ctx_bucket(pos) : 0;
    if (b == bk_) return;
    bk_ = b;
    bool any = false;
    for (int t = 1; t <= kMaxT && !any; ++t) any = cost_n_[b][t] > 0;
    if (any) return;
    for (int d = 1; d < kCtxBuckets; ++d)   // a table not measured yet starts from the nearest one that is
        for (int o : {b - d, b + d}) {
            if (o < 0 || o >= kCtxBuckets) continue;
            bool has = false;
            for (int t = 1; t <= kMaxT && !has; ++t) has = cost_n_[o][t] > 0;
            if (!has) continue;
            for (int t = 1; t <= kMaxT; ++t) {
                cost_[b][t] = cost_[o][t];
                cost_n_[b][t] = std::min(cost_n_[o][t], 1.0);   // a guess the bucket's own rounds soon replace
                last_[b][t] = last_[o][t];
            }
            return;
        }
}

void DraftPolicy::add_cost(int t, double round_ms) {
    double w = 1.0;
    // a round far above what is known for its width (a graph capture, a stall) counts as a tenth of a sample
    if (cost_aware() && cost_n_[bk_][t] >= kProbes && round_ms > 3.0 * cost_[bk_][t]) w = 0.1;
    const double alpha = cost_aware() ? std::max(kCostAlpha * w, w / (cost_n_[bk_][t] + w)) : kCostAlpha;
    cost_[bk_][t] = cost_n_[bk_][t] > 0 ? (1.0 - alpha) * cost_[bk_][t] + alpha * round_ms : round_ms;
    cost_n_[bk_][t] += w;
    last_[bk_][t] = rounds_;
}

int DraftPolicy::bucket(int match) {
    return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
}

double DraftPolicy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

double DraftPolicy::cost_ms(int t) const {
    t = std::clamp(t, 1, kMaxT);
    if (cost_n_[bk_][t] > 0) return cost_[bk_][t];
    // scale from the nearest measured size below: a round's fixed part flattens the measured curve at its small end,
    // and scaled from there the wider sizes came out too dear to be tried (Gyro-M on the RX 9060 XT: 46 ms at one
    // token, 57 at four, the six guessed at 98 against its 74)
    for (int u = t - 1; u >= 1 && !scale_all_; --u)
        if (cost_n_[bk_][u] > 0) return cost_[bk_][u] * kShape[t] / kShape[u];
    // none below: scale from the measured sizes, weighting each by how often it was seen
    double num = 0.0, den = 0.0;
    for (int u = 1; u <= kMaxT; ++u)
        if (cost_n_[bk_][u] > 0) {
            const double w = std::min(cost_n_[bk_][u], 20.0);
            num += w * cost_[bk_][u] * kShape[t] / kShape[u];
            den += w;
        }
    return den > 0 ? num / den : kShape[t];
}

bool DraftPolicy::stale(int t) const { return cost_n_[bk_][t] > 0 && rounds_ - last_[bk_][t] > kStaleRounds; }

double DraftPolicy::mtp_tokens(int t) const {
    if (mtp_n_[t] > 0) return mtp_tok_[t];
    return prior_tokens(t);
}

DraftPolicy::Pick DraftPolicy::choose(int t_mtp, int lookup_k, int match) const {
    Pick p;
    p.t = std::clamp(t_mtp, 1, max_t_);
    if (lookup_k <= 0) return p;
    const double base = mtp_tokens(p.t) / cost_ms(p.t);
    const double q = lookup_rate(match);
    double e = 1.0, qi = 1.0, best = 0.0;
    int best_t = 0;
    for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
        qi *= q;
        e += qi;
        const double r = e / cost_ms(k + 1);
        if (r > best) { best = r; best_t = k + 1; }
    }
    const int t_full = std::min(lookup_k, max_t_ - 1) + 1;
    if (best_t > 0 && best > base * (1.0 + margin_)) {
        p.lookup = true;
        p.t = best_t;
        // a wider size whose cost has gone stale is re-measured, so one slow stretch cannot price it out for good
        if (best_t < t_full && q >= 0.85 && stale(t_full)) p.t = t_full;
        return p;
    }
    // a guessed cost can keep the policy from ever measuring a size: the first few times a confident lookup would
    // need a size not measured yet, it is tried (verification keeps the output; only the one round's speed is at stake)
    if (t_full > p.t && (cost_n_[bk_][t_full] < kProbes || stale(t_full)) && q >= 0.85) {
        p.lookup = true;
        p.t = t_full;
    }
    return p;
}

void DraftPolicy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    if (round_ms > 0) add_cost(t, round_ms);
    rounds_ += 1.0;
    if (lookup) {
        const int b = bucket(match);
        ok_[b] = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}

// The draft layer's probability is not the model's: the calibration is the accepted drafts over the probabilities
// of the drafts tested (a draft is tested only while those before it held, so each ratio is conditional on them)
double DraftPolicy::draft_q(float p) const {
    const double calib = (dq_ok_ + kPriorN * kDraftPrior) / (dq_p_ + kPriorN * kDraftPrior);
    return std::clamp((double) p * calib, 0.0, 1.0);
}

double DraftPolicy::draft_q_mean() const { return (dq_ok_ + kPriorN * kDraftPrior) / (dq_n_ + kPriorN); }

// A window of t tokens, expecting `expected` committed tokens, whose drafts hold with probability `held`: does one
// more draft of acceptance q raise its tokens per ms?  Yes when the marginal tokens over the marginal ms beat the
// window's rate (the greedy step of a ratio).  A size whose cost is only a guess is measured instead, when likely.
bool DraftPolicy::mtp_step_pays(int t, double held, double q, double expected) const {
    if (t + 1 > max_t_) return false;
    const double gain = held * q;
    if (cost_n_[bk_][t + 1] < kProbes || stale(t + 1)) return gain >= kProbeHeld;
    const double dc = cost_ms(t + 1) - cost_ms(t);
    if (dc <= 0.0) return gain > 0.0;   // measured no dearer: the tokens come free
    if (gain / dc > expected / cost_ms(t)) return true;
    // the cost-aware window: the cost is not convex (with experts in RAM a window of 3 can cost about one of 2), so a
    // draft that does not pay alone is kept when it and one more as likely to hold would
    if (!cost_aware() || t + 2 > max_t_ || cost_n_[bk_][t + 2] < kProbes || stale(t + 2)) return false;
    const double dc2 = cost_ms(t + 2) - cost_ms(t);
    return dc2 > 0.0 && (gain + gain * q) / dc2 > expected / cost_ms(t);
}

int DraftPolicy::mtp_window(const float* probs, int max_t) const {
    max_t = std::clamp(max_t, 1, max_t_);
    int t = 1;
    double held = 1.0, expected = 1.0;
    while (t < max_t) {
        const double q = draft_q(probs[t - 1]);
        if (q <= 0.0 || !mtp_step_pays(t, held, q, expected)) break;
        expected += held * q;
        held *= q;
        ++t;
    }
    return t;
}

bool DraftPolicy::mtp_extend(const float* probs, int n, int max_t) const {
    max_t = std::clamp(max_t, 1, max_t_);
    if (n < 0 || n + 1 >= max_t) return false;
    double held = 1.0, expected = 1.0;
    for (int t = 1; t <= n; ++t) {   // the drafts made must pay first, each at its own probability
        const double q = draft_q(probs[t - 1]);
        if (q <= 0.0 || !mtp_step_pays(t, held, q, expected)) return false;
        expected += held * q;
        held *= q;
    }
    return mtp_step_pays(n + 1, held, draft_q_mean(), expected);
}

void DraftPolicy::observe_mtp_drafts(const float* probs, int t, int accepted) {
    dq_ok_ *= kDecay;
    dq_p_ *= kDecay;
    dq_n_ *= kDecay;
    for (int d = 0; d < t - 1 && d <= accepted; ++d) {   // tested: reached while the drafts before it held
        dq_n_ += 1.0;
        dq_p_ += std::clamp((double) probs[d], 0.0, 1.0);
        dq_ok_ += d < accepted ? 1.0 : 0.0;
    }
}

double DraftPolicy::chain_rate(int match) const {
    // prior: a chained draft is a lookup continuing the MTP's drafts - weaker evidence than a plain match of that length
    constexpr double kChainPrior[kBuckets] = {0.45, 0.65, 0.8, 0.9};
    const int b = bucket(match);
    return (cok_[b] + kPriorN * kChainPrior[b]) / (cok_[b] + cbad_[b] + kPriorN);
}

int DraftPolicy::chain(int t_mtp, double p_mtp, int k_avail, int match) const {
    t_mtp = std::clamp(t_mtp, 1, max_t_);
    const int kmax = std::min(k_avail, max_t_ - t_mtp);
    if (kmax <= 0) return 0;
    const double e0 = mtp_tokens(t_mtp), base = e0 / cost_ms(t_mtp), c = chain_rate(match);
    p_mtp = std::clamp(p_mtp, 0.0, 1.0);
    double gain = 0.0, ci = 1.0, best = 0.0;
    int best_k = 0;
    for (int k = 1; k <= kmax; ++k) {
        ci *= c;
        gain += ci;
        const double r = (e0 + p_mtp * gain) / cost_ms(t_mtp + k);
        if (r > best) { best = r; best_k = k; }
    }
    const int t_full = t_mtp + kmax;
    if (best_k > 0 && best > base * (1.0 + margin_)) {
        // As choose(): a profitable shorter chain must not keep a stale full size from being measured again.
        if (best_k < kmax && p_mtp * c >= 0.6 && stale(t_full)) return kmax;
        return best_k;
    }
    // as choose(): a size whose cost is only guessed is tried a few times when the continuation looks likely
    if ((cost_n_[bk_][t_full] < kProbes || stale(t_full)) && p_mtp * c >= 0.6) return kmax;
    return 0;
}

void DraftPolicy::observe_chain(int t_mtp, int k, int accepted, int match, double round_ms) {
    const int t = std::clamp(t_mtp + k, 1, kMaxT);
    if (round_ms > 0) add_cost(t, round_ms);
    rounds_ += 1.0;
    // the MTP part is an ordinary MTP window as far as its own drafts go
    const int mtp_acc = std::min(accepted, t_mtp - 1);
    const double got = mtp_acc + 1.0;
    mtp_tok_[t_mtp] = mtp_n_[t_mtp] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t_mtp] + kTokAlpha * got : got;
    mtp_n_[t_mtp] += 1.0;
    if (accepted < t_mtp - 1) return;   // the chain was never reached: it says nothing about the lookup
    const int b = bucket(match), ok = accepted - (t_mtp - 1);
    cok_[b] = kDecay * cok_[b] + ok;
    cbad_[b] = kDecay * cbad_[b] + (ok < k ? 1.0 : 0.0);
}

}  // namespace strata::spec
