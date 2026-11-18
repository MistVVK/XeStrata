// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/sigmas.hpp - the sampling schedules' one owner (the engine): the time shift every model's sigmas take
// and the two tables the server used to compute and send (Qwen-Image-2.1's base schedule, Mage-Flow's static
// shift).  The server now sends the model's kind and its settings (SCHED:, STEPS:) and the engine makes the table;
// a SIGMAS: list on the request still overrides it.  MOSS-Image's and Anima's tables stay in their pipelines (the
// reference's grids), their shift the same one formula.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace strata::image {

/// the flow-matching time shift: shift * t / (1 + (shift - 1) t).  T is float or double — each caller keeps the
/// precision its table was made in (the values are a contract with the pictures already made).
template<class T>
inline T sigma_shift(T t, T shift) {
    return shift * t / (T(1) + (shift - T(1)) * t);
}

/// Mage-Flow's sigmas for n steps and the terminal 0: t from 1 down to 1/n, the static shift (its
/// FlowMatchEulerDiscreteScheduler).  Double throughout, as the table was made before.
inline std::vector<float> mage_sigmas(int n, float shift = 6.0f) {
    std::vector<float> s;
    for (int i = 0; i < n; ++i) {
        const double t = n > 1 ? 1.0 - (1.0 - 1.0 / n) * i / (n - 1) : 1.0;
        s.push_back((float) sigma_shift(t, (double) shift));
    }
    s.push_back(0.0f);
    return s;
}

/// Qwen-Image-2.1's base schedule for `steps` steps of a width x height image and the terminal 0: diffusers'
/// QwenImage21Pipeline with the FlowMatchEulerDiscreteScheduler its scheduler_config sets up (linspace(1, 1/steps),
/// the exponential time shift by mu, a line in the image's tokens from base_shift 0.5 at 256 to max_shift 0.9 at
/// 8192, then stretched to end at shift_terminal 0.02).  An image token is a 16 x 16 pixel square.
inline std::vector<float> qi_base_sigmas(int steps, int64_t width, int64_t height) {
    if (steps == 1) return {1.0f, 0.0f};
    const int64_t seq = (width / 16) * (height / 16);
    const double mu = 0.5 + (0.9 - 0.5) / (8192 - 256) * (double) (seq - 256);
    const double e = std::exp(mu);
    std::vector<double> shifted;
    for (int i = 0; i < steps; ++i) {
        const double t = steps > 1 ? 1.0 - (double) i * (1.0 - 1.0 / steps) / (steps - 1) : 1.0;
        shifted.push_back(e / (e + (1.0 / t - 1.0)));
    }
    const double scale = (1.0 - shifted.back()) / (1.0 - 0.02);
    std::vector<float> s;
    s.reserve(shifted.size() + 1);
    for (const double x : shifted) s.push_back((float) (1.0 - (1.0 - x) / scale));
    s.push_back(0.0f);
    return s;
}

}  // namespace strata::image
