// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/spec/guess_gate.hpp - --pipeline-windows 2: is a guessed next window worth starting?
//
// Over a layer split, the first stage can compute a guess of the next window while the last checks this one. A guess
// that holds saves the next window the first stage's part (F); every guess costs the time it adds to the window it runs
// beside (C: the first stage's work against the last's for the CPU's expert pool and the bus), and drafting on to it
// costs the drafter's longer chain (E; an engine whose chain always runs that far never asks extend). With p the
// drafter's estimate that the guess holds and s the share of that estimate the guesses have held (decayed counts; at
// temperature the drafts hold less often than their probabilities say), the chain goes on where s p F > C + E and a
// guess starts where s p F > C. Until the costs it uses are measured it is the fixed rule p >= theta; a guess the
// costs refuse is started once in kExplore, so they stay current.
// It only chooses which windows run beside which: the text is unchanged.
#pragma once

#include <cstdint>

namespace strata::spec {

class GuessGate {
public:
    explicit GuessGate(double theta) : theta_(theta) {}
    void set_theta(double theta) { theta_ = theta; }

    /// whether to draft on past this window to a guess the drafter estimates at p
    bool extend(float p) const;
    /// whether to start that guess (counts the refusals for the exploration)
    bool start(float p);
    /// the first stage's part of a window that was not a guess; the drafter's chain past a window, ms
    void observe_front(double ms);
    void observe_extend(double ms);
    /// the last stage's part of a window, until the guess beside it (if any) was done too, ms
    void observe_back(double ms, bool beside_guess);
    /// a guess started at estimate p, and whether it held
    void observe_guess(float p, bool held);
    /// s p F > C (+ E) as a threshold on p, or theta's before the costs are measured (for the log)
    double threshold(bool with_extend) const;
    /// the guesses started and held, for the report
    int64_t started() const { return started_; }
    int64_t held() const { return held_n_; }

    static constexpr int kExplore = 16;

private:
    bool measured(bool with_extend) const {
        return n_front_ >= kMin && n_back_[0] >= kMin && n_back_[1] >= kMin && (!with_extend || n_ext_ >= kMin);
    }
    static constexpr int kMin = 3;
    double theta_;
    double front_ = 0.0, ext_ = 0.0, back_[2] = {0.0, 0.0};
    int n_front_ = 0, n_ext_ = 0, n_back_[2] = {0, 0};
    double held_ = 0.0, est_ = 0.0;   // decayed: the guesses that held, and their estimates' sum
    int refused_ = 0;
    int64_t started_ = 0, held_n_ = 0;
};

}  // namespace strata::spec
