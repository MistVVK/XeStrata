// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/spec/window_rule.hpp - the verify window's width, by the round's ACTIVE drafter.
//
// The MTP's cost-aware window (DraftPolicy::mtp_window) and its probability lists describe the MTP chain's own
// candidates; they do not govern DFlash2's (a different drafter, a different acceptance).  DFlash2's window ends
// where the product of its selector probabilities falls below --draft-conf.  Header only: both engines' loops call it.
#pragma once

namespace strata::spec {

/// DFlash2's window: 1 + the drafts whose selector-probability product stays at or above `conf` (up to `cap`).
/// conf <= 0: the whole cap (every draft made).  probs[j] is draft j's probability (the window's row j + 1).
inline int df2_window(const float* probs, int cap, double conf) {
    if (conf <= 0.0 || probs == nullptr) return cap;
    int t = 1;
    double chain = 1.0;
    while (t < cap && (chain *= (double) probs[t - 1]) >= conf) ++t;
    return t;
}

/// The MTP's cost-aware window applies only to candidates the MTP chain drew (not DFlash2's, not a lookup's, not
/// the prompt's first single-row window).
inline bool mtp_cost_applies(bool use_mtp, bool spec_cost, bool df2_active, bool first_window) {
    return use_mtp && spec_cost && !df2_active && !first_window;
}

/// DFlash2 with a long suffix match: the lookup window is taken without a DFlash2 draft round (the draft would be
/// thrown away).  min_match 0 = off.  The round's draft is deferred to the next iteration.
inline bool skip_draft_for_lookup(bool df2_active, int min_match, int lookup_k, int match) {
    return df2_active && min_match > 0 && lookup_k > 0 && match >= min_match;
}

}  // namespace strata::spec