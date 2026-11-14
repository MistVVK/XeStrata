// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/split_role.hpp - the second GPU's role in --layer-split auto: a stage of the split, a helper
// expert cache (RemoteExperts, as --expert-cache-device1) beside the first GPU alone, or no role at all.  The window
// prediction and the choice are one part both engines call (generate.cpp's Qwen3.8-Flash-Next and plain_model.cpp's
// other models), so the two cannot come to choose differently from the same measurements.  The prediction prices a
// decode window only, not the prompt: the choice keeps the margin (XESTRATA_SPLIT_GAIN_PCT) for that error.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// A helper GPU's fixed cost a layer (RemoteExperts::begin and finish): a small upload and three small commands on one
/// of its streams, then polled until done, the median of 16 after one warm-up.  `submit_ms`: the host's part, the
/// submissions, which come before the CPU pool's work; the return: launch to done.  -1 where it could not run.  Runs
/// on the current device (the caller sets it with OnDevice).
double probe_helper_layer_ms(double& submit_ms);

/// The helper expert cache's predicted decode window: the first GPU's cache takes the profile's hottest pairs as it
/// does alone, the helper the next ones that fit its free memory; in a layer the GPU's, the CPU's and the helper's
/// shares overlap (the largest counts), and the helper's pays a round trip over its link.  `mass` empty: every pair
/// counts the same.  Only the prediction, no choice and no side effect.
struct HelperCacheInputs {
    int64_t layers = 0;
    std::vector<std::pair<int32_t, int32_t>> profile;   ///< (layer, expert), hottest first
    std::vector<double> mass;                         ///< each pair's routed weight (empty: equal)
    std::function<int64_t(int64_t)> cost;             ///< a pair's bytes in the cache, by its layer
    std::function<double(int64_t)> blob_bytes;        ///< one expert's bytes, by its layer
    int64_t cap0 = 0;                                 ///< the first GPU's room for experts alone (bytes)
    int64_t caph = 0;                                 ///< the helper's room for experts (bytes)
    double bw0 = 0, bwh = 0;                          ///< the two GPUs' read bandwidths (bytes/ms)
    double cpu_bw = 0;                                ///< the CPU experts' RAM read bandwidth (bytes/ms)
    double link_scale0 = 1;                           ///< the first GPU's miss cost scale (its link)
    double rows = 0;                                  ///< the window's rows (the average verify window)
    double experts_per_row = 0;                       ///< the experts a row routes to
    double dense_layer = 0;                           ///< a layer's dense bytes
    double tail_keep = 1;                             ///< the route tail skip's share of the misses kept
    double head_bytes = 0;
    double rt = 0;                                    ///< the helper's round trip a layer it computes in (ms)
    double submit_ms = 0;                             ///< the host's submissions of that round trip (ms)
    bool drafts = false;
};
struct HelperCachePrediction {
    double ms = 0;            ///< the predicted decode window
    int64_t held = 0;         ///< the experts the helper takes
    double kept_mass = 0;     ///< the share of the routed mass on the two GPUs (0..1)
};
HelperCachePrediction predict_helper_cache(const HelperCacheInputs& in);

/// The role --layer-split auto gives the second GPU, from the three predicted windows and the margins.  `role` is
/// XESTRATA_SPLIT_ROLE (split / helper, empty: the prediction decides); `roles_helper` is --split-roles helper (the
/// split is no choice, but the first GPU alone still can be); `skip_if_fits` is --split-skip-if-fits.  The caller
/// applies the side effects (the split's points, the helper's count and device) and says the choice in its own words.
enum class SplitRole { Split, Helper, Alone, SkipIfFits };
struct SplitRoleInputs {
    double alone_ms = 0;      ///< the first GPU alone's predicted window
    double split_ms = 0;      ///< the split's predicted window
    double helper_ms = 0;     ///< the helper cache's predicted window
    bool alone_holds_all = false;   ///< the first GPU alone holds every profiled pair
    int64_t helper_held = 0;        ///< the experts the helper would take
    double gain = 0.05;             ///< the margin a role must beat the next choice by (XESTRATA_SPLIT_GAIN_PCT)
    double helper_gain = 0.05;      ///< the helper's own margin against the split (XESTRATA_SPLIT_HELPER_GAIN_PCT)
    bool roles_helper = false;      ///< --split-roles helper
    bool skip_if_fits = false;      ///< --split-skip-if-fits
    bool helper_eligible = false;   ///< two GPUs, the expert pool, no pipeline windows, no batch slots
    std::string role;               ///< XESTRATA_SPLIT_ROLE
};
struct SplitRoleChoice {
    SplitRole role = SplitRole::Split;
    double ms = 0;   ///< the chosen role's predicted window (-1: SkipIfFits, decided)
    bool helper_chosen = false;   ///< the helper beat the split (before the first-GPU-alone check), for the message
};
SplitRoleChoice choose_split_role(const SplitRoleInputs& in);

}  // namespace strata::core
