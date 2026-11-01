// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/spec/guess_gate.cpp - see include/strata/spec/guess_gate.hpp.
#include "strata/spec/guess_gate.hpp"

#include <algorithm>

namespace strata::spec {
namespace {

constexpr double kAlpha = 0.1;    // EMA weight of a new time
constexpr double kDecay = 0.97;   // the guesses' outcomes: older ones fade
constexpr double kPriorN = 4.0;   // before any guess, the estimates are taken as they are, worth 4 guesses

void ema(double& v, int& n, double x) {
    v = n == 0 ? x : v + kAlpha * (x - v);
    n = std::min(n + 1, 1 << 20);
}

}  // namespace

double GuessGate::threshold(bool with_extend) const {
    if (!measured(with_extend)) return theta_;
    const double s = (held_ + kPriorN) / (est_ + kPriorN);
    const double cost = std::max(0.0, back_[1] - back_[0]) + (with_extend ? ext_ : 0.0);
    return front_ > 0.0 ? cost / (s * front_) : 1e30;
}

bool GuessGate::extend(float p) const {
    return p >= threshold(true) || (p >= theta_ && refused_ >= kExplore - 1);
}

bool GuessGate::start(float p) {
    if (p >= threshold(false)) return true;
    if (p < theta_) return false;
    if (++refused_ < kExplore) return false;
    refused_ = 0;
    return true;
}

void GuessGate::observe_front(double ms) { ema(front_, n_front_, ms); }
void GuessGate::observe_extend(double ms) { ema(ext_, n_ext_, ms); }
void GuessGate::observe_back(double ms, bool beside_guess) {
    const int b = beside_guess ? 1 : 0;
    ema(back_[b], n_back_[b], ms);
}

void GuessGate::observe_guess(float p, bool held) {
    ++started_;
    held_n_ += held ? 1 : 0;
    held_ = held_ * kDecay + (held ? 1.0 : 0.0);
    est_ = est_ * kDecay + (double) p;
}

}  // namespace strata::spec
