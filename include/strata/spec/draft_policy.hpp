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
// to the machine and the context length. It only chooses which drafts to verify: the output is unchanged.
//
// Window sizes: every size from 1 to max_t is allowed unless set_sizes() narrows them. A size the verifier cannot
// hold (its window graph found no VRAM) is dropped with disable(); a lookup or chain then takes the largest allowed
// size that fits its proposal (--lookup-max-t: a few long lookup sizes, each its own window graph).
//
// Fixed costs (repeatable output): a different window is a different batch for the verifier, so two runs whose measured
// round times differ can choose different windows and print different tokens.  set_fixed_costs() replaces the measured
// costs by a table that never changes (no EMA update, no first-use probe, no stale-size retry); the token statistics keep
// learning, so the same tokens give the same windows on every run.  The table comes from CostLearner: plain trimmed means
// of the round times of one learning stretch on this machine (the EMA follows the last ~10-20 rounds and moved 5-20 %
// between requests of one run; whole-run means of three runs agreed within 0.3-1.8 %, phaseA-tests/par/RESULTS-REPEAT.md).
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::spec {

/// The round times of a learning stretch, by context bucket and window size, for a fixed cost table.  Keeps up to kKeep
/// samples per (bucket, size) in arrival order.
class CostLearner {
public:
    static constexpr int kMaxT = 16;
    static constexpr int kCtxBuckets = 3;
    static constexpr int kKeep = 4096;
    void add(int bucket, int t, double ms);
    void reset();
    int64_t rounds() const { return rounds_; }
    int samples(int bucket, int t) const;
    /// The table of `bucket` for windows 1..n (n = the largest size seen in that bucket; empty when the bucket has none):
    /// the trimmed mean (the lowest and highest kTrim share dropped) of each size with at least `min_n` samples; the other
    /// sizes are filled between measured neighbours (linear in the size) or, outside them, by the policy's prior shape.
    std::vector<double> table(int bucket, int min_n) const;
    struct Verdict {
        bool ready = false;          ///< enough rounds, and every bucket with data has a size with min_n samples
        bool disturbed = false;      ///< the stretch looks disturbed: do not keep a table from it (start learning again)
        std::string why;             ///< what was seen (for the log)
    };
    /// Ready after `min_rounds` timed rounds.  Disturbed (each check per size with at least 2 x min_n samples):
    /// (1) its trimmed means of the first and the second half of its samples differ by more than kDriftMax;
    /// (2) more than kSpikeMax of its samples are above twice its median;
    /// (3) a measured size is cheaper than a measured smaller one by more than kOrderSlack (a round of more rows cannot be cheaper).
    Verdict check(int64_t min_rounds, int min_n) const;
    static constexpr double kTrim = 0.1;
    static constexpr double kDriftMax = 0.15;
    static constexpr double kSpikeMax = 0.05;
    static constexpr double kOrderSlack = 0.10;

private:
    std::array<std::array<std::vector<float>, kMaxT + 1>, kCtxBuckets> s_{};
    int64_t rounds_ = 0;
};

/// The per-machine table file of --draft-costs: text, one table per context bucket, valid only for the `key` it was learned under
/// (card, driver, engine, model, drafter and the settings that move a round's cost; the engine builds it).
///   strata-draft-costs 1
///   key <one line>
///   bucket <0..2> <ms of T1>,<ms of T2>,...
///   share <0..1>              (optional: the CPU prompt share learned on this machine, src/prefill/prefill.cpp)
///   end
struct CostFile {
    std::string key;
    std::array<std::vector<double>, CostLearner::kCtxBuckets> tables{};
    double cpu_share = -1.0;   ///< < 0: none stored
};
/// false (with `why` for the log) when the file is missing, unreadable, malformed or truncated, or was learned under another key: the caller
/// then learns a new table.  Never throws.
bool read_cost_file(const std::string& path, const std::string& key, CostFile& out, std::string& why);
/// Writes the file whole or not at all (a temporary file beside it, then renamed over it); makes the folder.  Never throws.
bool write_cost_file(const std::string& path, const CostFile& f, std::string& why);

class DraftPolicy {
public:
    static constexpr int kMaxT = 16;
    static constexpr int kBuckets = 4;
    static constexpr int kCtxBuckets = 3;   // STRATA_SPEC_COST: row costs per context length (< 6K, < 24K, longer)

    explicit DraftPolicy(int max_t, double margin = 0.03);

    struct Pick {
        bool lookup = false;
        int t = 1;                      // window size (1 + drafts)
    };
    /// `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none); `match`: its match length.
    Pick choose(int t_mtp, int lookup_k, int match) const;
    /// After the round: the window it used, the drafts accepted, and the round's time (verify + commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);
    /// Gated proposals, inside one generation: a round's committed outcome only, never its time. `t` is the
    /// lookup window chosen before the draws, or the configured MTP window; both are fixed by the history. A
    /// lookup window counts as a probe of its size, so a size whose cost is still guessed is tried kProbes times,
    /// not every round. Round times include the next window's drafts and the rows past the accepted ones, whose
    /// Philox counters the next rounds draw again, so they wait (defer_cost) or go to the live policy (observe).
    void observe_committed(bool lookup, int t, int accepted, int match);
    /// Gated proposals, inside one generation: a round's time, held until the committed position passes `end`,
    /// the last Philox counter of any draw that time depends on (the window's drafts and rows, the next window's
    /// drafts when the time includes them). Until then those counters can be drawn again by the next rounds and the
    /// time (rejected rows' routing, the next draft's gated length) would steer a window by its own draws.
    void defer_cost(int64_t end, int t, double round_ms);
    /// Apply, in order, the held times whose `end` is below `committed` (the next window's first counter), exactly
    /// as observe() prices a round, in the context bucket the round ran in.
    void apply_costs(int64_t committed);

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
    /// A per-card row-cost table (round ms of the windows of 1..n rows, the draft included) counted as `weight` earlier rounds each:
    /// the start of a process that already knows the card (STRATA_SPEC_COST_TABLE; tuning time), replaced by what it then measures.
    void seed_costs(const double* ms, int n, double weight, int bucket = 0);
    /// STRATA_SPEC_COST: the context position of the next round; selects the row-cost table of its length (a row of attention costs
    /// more at 16K than at 1K).  A bucket with no measurement yet starts from the nearest one that has.  A no-op without set_robust(true).
    void set_context(long long pos);
    /// The measured round cost of every window size seen: "T1:39.8(n=120) T2:..." for the log / the next process's table.
    std::string cost_table_text() const;
    /// STRATA_SPEC_COST: a round time far above what is known for that width (the first use of a width captures its graph; a stalled
    /// round) counts as a tenth of a sample, so one capture cannot price a width out of use.  Off: the cost EMA as it always was.
    void set_robust(bool on) { robust_ = on; }

    /// Fixed costs for `bucket` (round ms of the windows of 1..n rows): they replace whatever was measured, and from now on no round time
    /// changes any cost, no size is probed for its cost and no size goes stale (see the file comment).  Call once per bucket that has a table;
    /// a bucket without one takes the nearest bucket's table at set_context (as before), and that copy is fixed too.  ms == nullptr or n <= 0:
    /// fixed at the built-in shape.  STRATA_SPEC_FIXED=1 (+ STRATA_SPEC_FIXED_TABLE="T1,T2,...", every bucket) applies it in the constructor.
    void set_fixed_costs(const double* ms, int n, int bucket = 0);
    bool fixed() const { return fixed_; }
    /// Learning a fixed table: every timed round also goes to learner() (raw round time, in its context bucket).  Off by default.
    void set_learning(bool on) { learning_ = on; }
    bool learning() const { return learning_; }
    CostLearner& learner() { return learner_; }
    const CostLearner& learner() const { return learner_; }

    /// The window sizes a lookup or chained window may use, as a bit mask (bit t = size t; bit 1 is always kept).
    void set_sizes(unsigned mask);
    void disable(int t);                   // drop one size (its window graph did not fit)
    bool allowed(int t) const { return t >= 1 && t <= max_t_ && ((sizes_ >> t) & 1u); }
    /// The largest allowed size <= t (1 if none).
    int fit(int t) const;
    int max_t() const { return max_t_; }
    unsigned sizes() const { return sizes_; }
    /// COST-AWARE MTP WINDOW (STRATA_SPEC_COST=1): the MTP chain is drafted one step at a time; after draft j (probability p_j)
    ///   E_j = 1 + q_0 + .. + q_j,  q_i = prod_{k<=i} a(p_k)   a() = the acceptance measured for MTP drafts of that probability
    /// is the expected tokens of the window of j + 2 rows, worth E_j / cost(j + 2) tokens per ms (cost() = the measured round time of
    /// that window size, which includes the draft).  The window is the best of the widths seen; the chain goes on only while one or two
    /// more drafts could still beat it (the cost is not convex: on a card with experts in RAM a window of 3 costs about a window of 2).
    /// It only chooses which rows are verified; the output is unchanged.
    struct ChainState {
        double q = 1.0, e = 1.0;
        double best_v = 0.0;
        int best_rows = 1;                ///< the window to verify: 1 + the drafts kept
        int kept = 0;                     ///< best_rows - 1
    };
    /// Draft `idx` (0-based, probability p) was just computed: update `st` and say whether to compute draft idx + 1.
    /// idx == 0 starts a new chain.  `cap` = the widest window (rows).
    bool keep_draft(ChainState& st, int idx, float p, int cap) const;
    /// After a round of the MTP window T whose drafts were p[0..T-1): `accepted` of them held.  Teaches draft_rate().
    void observe_probs(const float* p, int n_window_drafts, int accepted);
    double draft_rate(double p) const;     // acceptance of an MTP draft that came with probability p

private:
    static int bucket(int match);
    double mtp_tokens(int t) const;
    bool stale(int t) const;               // a size not measured for a long stretch of rounds

    int max_t_;
    double margin_;
    unsigned sizes_ = ~0u;                                 // allowed window sizes (bit t)
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> cost_{}, cost_n_{};   // round ms by context bucket and window size
    int bk_ = 0;                                           // the current context bucket
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{};    // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
    std::array<double, kBuckets> cok_{}, cbad_{};          // chained lookup drafts, the same (reached rounds only)
    double rounds_ = 0;                                    // rounds observed so far
    bool robust_ = false;
    bool fixed_ = false;                                   // set_fixed_costs: no round time changes a cost
    bool learning_ = false;                                // set_learning: round times also go to learner_
    CostLearner learner_;
    double clamp_sample(int t, double ms) const;
    static constexpr int kPBuckets = 8;
    std::array<double, kPBuckets> pa_ok_{}, pa_n_{};       // MTP drafts reached / accepted by probability bucket (decayed)
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> last_{};   // the round at which each size was last measured
    // Gated proposals (observe_committed / defer_cost / apply_costs); all zero or empty on any other path.
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> tried_{};      // lookup windows of that size run with no time yet
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> delayed_n_{};  // of cost_n_: times applied by apply_costs (not new probes)
    struct HeldCost { int64_t end; int bucket; int t; double ms; };
    std::vector<HeldCost> held_;                                          // defer_cost, oldest first
};

}  // namespace strata::spec
