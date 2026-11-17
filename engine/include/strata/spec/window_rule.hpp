// include/strata/spec/window_rule.hpp - which rule picks a round's verify window, by the round's ACTIVE draft source.
//
// `use_mtp` means "the MTP layer is loaded"; with DFlash2 it stays true because the MTP is the fallback.  The decisions below
// read what the MTP chain just produced (the cost-aware rule's `chain_rows`, the draft probabilities), so they apply only when
// the MTP chain is the source of this round's candidates.  With DFlash2 as the source, `chain_rows` is never written (the chain
// callback does not run) and would pin the window at one row.  Header only, no device types: the CLI loop and the serve loop of
// every backend call it.
#pragma once

#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace strata::spec {

struct WindowRuleIn {
    bool use_mtp = false;        ///< the MTP layer is loaded (a fallback beside DFlash2 counts)
    bool draft_on = false;       ///< some drafter proposes (the MTP, or DFlash2 alone with no MTP loaded)
    bool df2_active = false;     ///< DFlash2 drew this round's candidates (CLI: use_df2; serve: this request's drafter)
    bool spec_cost = false;      ///< STRATA_SPEC_COST=1: the cost-aware MTP window
    bool first_window = false;   ///< the prompt's last token alone: the caller sets its own width afterwards
    int chain_rows = 1;          ///< the window the cost rule kept in the MTP chain just drafted (MTP only)
    int cap = 1;                 ///< S_mtp: the widest window of this source
    int start = 1;               ///< the width before any rule (CLI: cap; serve: cap with the MTP, 1 without)
    double spec_min_p = 0.0;     ///< --spec-min-p: stop at the first draft whose probability is below it (0: off)
    double draft_conf = 0.0;     ///< --draft-conf: DFlash2's window ends where the product of the probabilities falls below it (0: off)
    const float* dprob = nullptr;   ///< the drafts' probabilities, dprob[j] for draft j (the row j + 1 of the window)
};

/// The cost-aware MTP window applies only to the MTP chain's own candidates.
inline bool cost_window_applies(bool use_mtp, bool spec_cost, bool df2_active, bool first_window) {
    return use_mtp && spec_cost && !df2_active && !first_window;
}

/// The MTP's q lists (STRATA_SPEC_PROB) are the proposal law only for candidates the MTP drew.
inline bool mtp_q_applies(bool use_mtp, bool df2_active, bool mtp_prob, bool from_lookup, bool first_window, int mtp_rows) {
    return use_mtp && !df2_active && mtp_prob && !from_lookup && !first_window && mtp_rows > 1;
}

/// DFlash2 with a LONG suffix match: the lookup window is taken without a DFlash2 draft round.  The choice "lookup or DFlash2" needs the draft only to
/// test that its first token agrees; when the match is long the lookup is the likelier winner and the draft (2.4 ms on Metal, 1.2 ms on the RTX PRO
/// 5000) would be thrown away.  The round's draft is then DEFERRED to the start of the next iteration, where the policy decides (lookup: the committed
/// rows only update the drafter's ring; otherwise the draft is made then, from the same taps).  min_match 0 = off.
inline bool skip_draft_for_lookup(bool df2_active, int min_match, int lookup_k, int match) {
    return df2_active && min_match > 0 && lookup_k > 0 && match >= min_match;
}

/// STRATA_DF2_LOOKUP_SKIP=N (match length in tokens; 0 = off).  `build_default`: what a build without the variable uses (Metal 8, others 0 until measured).
inline int df2_lookup_skip_min(int build_default) {
    static const int v = [build_default] {
        const char* e = std::getenv("STRATA_DF2_LOOKUP_SKIP");
        return e != nullptr && e[0] != '\0' ? std::atoi(e) : build_default;
    }();
    return v;
}

inline int choose_window(const WindowRuleIn& in) {
    int T = in.start;
    if (cost_window_applies(in.use_mtp, in.spec_cost, in.df2_active, in.first_window)) {
        T = std::clamp(in.chain_rows, 1, in.cap);
    } else if (in.draft_on && in.spec_min_p > 0.0 && in.dprob != nullptr) {
        T = 1;
        while (T < in.cap && in.dprob[T - 1] >= (float) in.spec_min_p) ++T;
    }
    if (in.df2_active && in.draft_conf > 0.0 && in.dprob != nullptr) {   // DFlash2's own confidence: every draft so far holds
        T = 1;
        double chain = 1.0;
        while (T < in.cap && (chain *= (double) in.dprob[T - 1]) >= in.draft_conf) ++T;
    }
    return T;
}

// (An opt-in calibrated DFlash2 window gate, STRATA_DF2_GATE / STRATA_DF2_ROW_MS, was removed in 0.2.13: it measured 2.7-4.9% slower.)

}  // namespace strata::spec
