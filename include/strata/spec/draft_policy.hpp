// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/spec/draft_policy.hpp - per verify round: the MTP's window, or a lookup (suffix) window?
//
// The suffix drafter (prompt lookup) proposes the tokens that followed an earlier repeat of the context. Taken
// whenever it proposes more than the MTP, it lost 2-8% on ordinary text: a long lookup window costs much more to
// verify than the MTP's usual 3-4 tokens, and it is only worth that when enough of it is accepted. llama.cpp's
// lookup decoding answers the same problem with confidence thresholds on its n-gram statistics; here the policy
// learns both sides online and compares expected committed tokens per millisecond:
//
//   MTP     E = the mean tokens a window of that size has committed (EMA), at the measured cost of that size
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup drafts whose
//           match was about as long (4 buckets of match length, decayed counts), at the measured cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by `margin`. Costs are the measured round
// times per window size (EMA; sizes not seen yet are scaled from seen ones by a prior shape), so the policy adapts
// to the machine and the context length. Its sizes are part of what it verifies: the same row of a 3-token window and
// of a 4-token window got different argmaxes (XeStrata's own products and dense kernels, RX 9060 XT, 2026-11-14), so
// the sizes moving with the measured times move the answer's last bits. A run whose answer must be the same every time
// fixes them (--spec-window min-p --spec-min-p 0) and turns the suffix drafts off (--suffix-draft 0).
#pragma once

#include <array>
#include <cstdint>

namespace strata::spec {

class DraftPolicy {
public:
    static constexpr int kMaxT = 8;
    static constexpr int kBuckets = 4;
    static constexpr int kCtxBuckets = 3;   // XESTRATA_SPEC_COST: round costs per context (< 6K, < 24K, longer)
    // XESTRATA_SPEC_COST=auto (the default): the cost-aware window from this bucket of context length on, below it
    // the plain window.  Measured again on 2026-11-13 (Qwen3.8 IQ3_XXS and Qwen3.6-35B-A3B through generate, the text
    // followed, the B70 and the RTX 4070): at 1K it chose narrower windows than the plain rule and decoded 2.7%
    // slower on the B70; from 4K up it chose the same windows as the plain rule on both GPUs, so the shallow bucket is
    // where it costs.  (0.2.6 measured the same 1K loss and a 1.3-3.2% gain at 32K.)
    static constexpr int kCostBucket = 1;

    explicit DraftPolicy(int max_t, double margin = 0.03);

    struct Pick {
        bool lookup = false;
        int t = 1;                      // window size (1 + drafts)
    };
    /// `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none); `match`: its match length.
    Pick choose(int t_mtp, int lookup_k, int match) const;
    /// After the round: the window it used, the drafts accepted, and the round's time (verify + commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    /// --lookup-chain: how many of `k_avail` lookup tokens to append after an MTP window of `t_mtp` tokens whose
    /// drafts all hold with probability `p_mtp` (estimated from the draft layer's own probabilities): the k with the
    /// best expected tokens per ms, E(t_mtp) + p_mtp (c + c^2 + .. + c^k) at the cost of t_mtp + k, if it beats the MTP
    /// window alone by `margin`; c = the acceptance of chained drafts by match length (their own counts). 0 = none.
    int chain(int t_mtp, double p_mtp, int k_avail, int match) const;
    /// After a chained round: the MTP window's size, the chained tokens, the window's accepted drafts, its time.
    void observe_chain(int t_mtp, int k, int accepted, int match, double round_ms);
    double chain_rate(int match) const;
    double lookup_rate(int match) const;   // current q for a match length
    double cost_ms(int t) const;           // measured or scaled round time of a window of t tokens
    /// Under the min-p window rule (`true`): a size not measured is scaled from every measured size, as upstream does,
    /// instead of from the nearest one below.  The rule sizes the MTP's window by the drafts' probabilities alone, so
    /// the costs only weigh a lookup window against it, and in the low-RAM file tier, whose small windows' times swing
    /// with the SSD, the scaling from below priced the short lookups cheap: on the RX 9060 XT in a 32 GiB cgroup
    /// (IQ3_XXS, 32K, the text followed, 2026-10-24) it took five short lookup windows where upstream took two of six
    /// tokens, and decoded 16.6-17.4 tok/s against 17.2-17.7 with the scaling from every size.
    void set_scale_all(bool on) { scale_all_ = on; }
    /// The tokens a window of t committed before any is observed: a typical acceptance of a draft
    static double prior_tokens(int t) { return 1.0 + 0.7 * (t - 1); }
    /// The MTP's own window, from this round's drafts (the idea of DeepInfer's confidence-scheduled verification,
    /// written anew): a draft is kept while the tokens it is expected to add, at its calibrated acceptance, raise the
    /// window's expected tokens per ms at the measured window costs; a size whose cost is only guessed is tried when
    /// its drafts are likely to hold.  `probs[j]`: draft j's probability under the draft layer (0 past the drafts
    /// made).  Returns the window size, 1..max_t.
    int mtp_window(const float* probs, int max_t) const;
    /// While drafting: with `n` drafts made (probs[0..n-1]), whether one more could still pay, at the mean
    /// acceptance of a draft; false as well when the drafts made already stop paying.
    bool mtp_extend(const float* probs, int n, int max_t) const;
    /// After an MTP window of `t` tokens: its drafts' probabilities and the accepted count, which calibrate the
    /// drafts' probabilities against what the model accepts (decayed, as the lookup counts).
    void observe_mtp_drafts(const float* probs, int t, int accepted);
    /// XESTRATA_SPEC_COST (upstream 687810556 .. b1ebc3d7c): the cost-aware window: the context position of the next
    /// round selects its own table of round costs (a row of attention costs more at 16K than at 1K; a table with no
    /// measurement yet starts from the nearest one that has); a round far above what is known for its width (the first
    /// use of a width captures its graph) counts as a tenth of a sample, the first samples as a running mean; and a
    /// draft that does not pay by itself is kept when it and the next one together would.  auto (the default): from
    /// kCostBucket on; below, the plain window, where it was measured slower.  1: always (the opt-in as it was); 0:
    /// never.  Call before each round.
    void set_context(int64_t pos);
    /// The window rule the next round runs: the cost-aware one now, or the plain one.
    bool cost_aware() const {
        return mode_ == Mode::kOn || (mode_ == Mode::kAuto && ctx_bucket(pos_) >= kCostBucket);
    }

private:
    enum class Mode { kOff, kOn, kAuto };
    static int ctx_bucket(int64_t pos);
    static int bucket(int match);
    double mtp_tokens(int t) const;
    bool stale(int t) const;               // a size not measured for a long stretch of rounds
    double draft_q(float p) const;         // a draft's acceptance, from its probability and the calibration
    double draft_q_mean() const;           // the mean acceptance of a tested draft
    bool mtp_step_pays(int t, double held, double q, double expected) const;
    void add_cost(int t, double round_ms);

    int max_t_;
    double margin_;
    bool scale_all_ = false;
    Mode mode_ = Mode::kAuto;                              // XESTRATA_SPEC_COST
    int64_t pos_ = 0;                                      // the context position of the next round
    int bk_ = 0;                                           // the cost table of the next round (0 below kCostBucket)
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> cost_{}, cost_n_{};   // round ms by context and window size
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{};    // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
    double rounds_ = 0;                                    // rounds observed so far
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> last_{};   // the round each size was last measured
    std::array<double, kBuckets> cok_{}, cbad_{};          // chained lookup drafts, the same (reached rounds only)
    double dq_ok_ = 0, dq_p_ = 0, dq_n_ = 0;               // MTP drafts tested: accepted, their probabilities, count
};

}  // namespace strata::spec
