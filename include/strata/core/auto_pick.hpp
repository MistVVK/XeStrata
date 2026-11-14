// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/auto_pick.hpp - a speed switch's three settings, and the choice its first windows make.
#pragma once

#include "strata/platform/env.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

/// A speed switch's settings as XESTRATA_<name> gives them: unset or "auto" lets the engine choose at run time,
/// "0"/"off" turns it off, "1"/"on" on.  A value that is none of these reads as auto, with one line.
enum class Pick { Auto, Off, On };

inline Pick pick_from_env(const char* name) {
    const char* v = strata::env_get(name);
    if (v == nullptr || v[0] == 0 || std::strcmp(v, "auto") == 0) return Pick::Auto;
    if (v[0] == '0' || std::strcmp(v, "off") == 0 || std::strcmp(v, "no") == 0) return Pick::Off;
    if (v[0] == '1' || std::strcmp(v, "on") == 0 || std::strcmp(v, "yes") == 0) return Pick::On;
    std::fprintf(stderr, "xestrata: XESTRATA_%s=%s is not auto, off or on: auto\n", name, v);
    return Pick::Auto;
}

/// A switch that shows only inside the engine's windows, chosen by the first ones: the windows where it can change
/// anything run in blocks of `block`, the switch's other form and its default in turn, `rounds` rounds of two blocks
/// (two rounds: default, other, other, default, so that a drift over the run falls on both alike), and the faster time a
/// token stays.  The other form has to be `gain` faster AND clearer than the windows' own spread (twice the standard
/// error of the two means: a handful of windows at a long context differ by 10% from one to the next), so a run the two
/// cannot be told apart in keeps the default.  Blocks run on the windows where the switch can show, only; one that
/// never sees such a window keeps the default and says so.  Nothing is kept between runs; the decision is one line.
/// While the serve engines' start measures the drafters (strata/spec/draft_probe.hpp) the switches below are not
/// measured on its windows: they decide on the first real requests', as a start without the measurement does.
inline bool& window_choices_paused() {
    static bool paused = false;
    return paused;
}

class WindowChoice {
public:
    WindowChoice() = default;
    /// `name` is the switch as a user reads it (XESTRATA_<name>), `what` the decision's subject in its line.
    WindowChoice(const char* who, const char* name, const char* what, int block = 8, int rounds = 2,
                 double gain = 1.03, const char* unit = "a window")
        : who_(who), name_(name), what_(what), unit_(unit), block_(block < 1 ? 1 : block),
          rounds_(rounds < 1 ? 1 : rounds), gain_(gain) {}

    /// The form for the window about to run: false (the default) until it decides, then the winner.
    bool on() const { return on_; }
    /// Still measuring.
    bool choosing() const { return block_index_ < blocks(); }
    int blocks() const { return 2 * rounds_; }

    /// The window that just ran under on(): its milliseconds and tokens, and whether it could show the difference.
    void took(double ms, int64_t tokens, bool informative = true) {
        if (!choosing() || !informative || tokens <= 0 || window_choices_paused()) return;
        const int f = on_ ? 1 : 0;
        const double per = ms / (double) tokens;
        ++n_[f];
        sum_[f] += per;
        sumsq_[f] += per * per;
        if (++windows_in_block_ < block_) return;
        windows_in_block_ = 0;
        ++block_index_;
        if (block_index_ < blocks()) on_ = other(block_index_);
        else decide();
    }

private:
    /// ABBA: the default, the other, the other, the default, ...  A round is two blocks.
    static bool other(int block) { return (block % 2) != ((block / 2) % 2); }

    void decide() {
        if (n_[0] < 2 || n_[1] < 2) {
            on_ = false;
            std::fprintf(stderr, "%s: XESTRATA_%s: %s: not seen in the first windows: off\n", who_, name_, what_);
            return;
        }
        const double m0 = sum_[0] / (double) n_[0], m1 = sum_[1] / (double) n_[1];   // ms a token, default / other
        const double v0 = std::max(0.0, (sumsq_[0] - sum_[0] * sum_[0] / (double) n_[0]) / (double) (n_[0] - 1));
        const double v1 = std::max(0.0, (sumsq_[1] - sum_[1] * sum_[1] / (double) n_[1]) / (double) (n_[1] - 1));
        const double se = std::sqrt(v0 / (double) n_[0] + v1 / (double) n_[1]);
        on_ = m0 - m1 > std::max((gain_ - 1.0) * m0, 2.0 * se);
        std::fprintf(stderr, "%s: XESTRATA_%s: %s: %.2f ms %s with it, %.2f without (+-%.2f over %d and %d windows): "
                             "%s (XESTRATA_%s=0/1 picks one)\n", who_, name_, what_, m1, unit_, m0, se, n_[1], n_[0],
                     on_ ? "on" : "off", name_);
    }

    const char* who_ = "xestrata";
    const char* name_ = "";
    const char* what_ = "";
    const char* unit_ = "a window";
    int block_ = 8, rounds_ = 2;
    double gain_ = 1.03;
    int block_index_ = 0, windows_in_block_ = 0;
    bool on_ = false;
    int n_[2] = {0, 0};
    double sum_[2] = {0.0, 0.0}, sumsq_[2] = {0.0, 0.0};
};

/// Whether every MoE layer of the pack rounds the CPU's and the GPU's experts to the same bits (a canon layer, or a
/// trellis one the CPU supports): the adaptive tier's swaps then move an expert between them without changing a
/// token, and the switches that only change when a swap is applied can be measured on the running text.  The same
/// rule that decides whether a window is cut at a swap (ADAPT_CUT).
bool adapt_layers_round_alike();

/// ADAPT_LAG=auto: the value the first rounds chose (1: the next window waits for the copies, 0: not yet, the
/// engine's own default stays).  XESTRATA_ADAPT_LAG=<n> fixes it.
int adapt_lag_forced();
int adapt_lag_chosen();
/// The form the trial is in for the next window (1: the next window waits; 0: not measuring).
int adapt_lag_forming();
/// What a round's lag cost it: the wait the window took for the copies (lag 1), or the swapped experts' CPU work over
/// the windows a swap waited (lag > 1).  One a round, at the window that applies the swap.
void adapt_lag_window(double cost_ms);

/// DISJOINT_ADAPT=auto: whether a helper GPU's experts stay out of the primary's promotion candidates, chosen by the
/// first windows of a run whose helpers hold experts.  XESTRATA_DISJOINT_ADAPT=0/1 fixes it.
bool disjoint_adapt_on();
/// A decode window of a run with helpers: its time, rows, and whether a helper GPU held an expert.
void disjoint_adapt_window(double ms, int64_t tokens, bool helpers);

}  // namespace strata::core
