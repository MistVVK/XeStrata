// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::core {

// Process-local choices, separated by device and shape. Failed or graph-recording trials are not stored.
template<class Device, class Shape, class Choice> class TimedCache {
public:
    std::optional<Choice> get(const Device& device, const Shape& shape) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto d = choices_.find(device);
        if (d == choices_.end()) return {};
        const auto s = d->second.find(shape);
        return s == d->second.end() ? std::optional<Choice>{} : s->second;
    }
    Choice put(const Device& device, const Shape& shape, Choice choice) {
        std::lock_guard<std::mutex> lock(mu_);
        return choices_[device].try_emplace(shape, choice).first->second;
    }
    template<class Predicate> std::pair<Choice, bool> put_first(const Device& device, const Shape& shape, Choice choice, Predicate matches) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& by_shape = choices_[device];
        const bool first = std::none_of(by_shape.begin(), by_shape.end(), [&](const auto& item) { return matches(item.first); });
        return {by_shape.try_emplace(shape, choice).first->second, first};
    }
private:
    std::mutex mu_;
    std::unordered_map<Device, std::map<Shape, Choice>> choices_;
};

// Fourfold classes keep successive prompt chunks from retuning at every key count.
inline int64_t timed_key_class(int64_t keys) {
    int64_t n = 1024;
    while (n < keys && n <= std::numeric_limits<int64_t>::max() / 4) n *= 4;
    return n;
}

// Nested candidate launches use already measured choices, but defer new tuning until the actual call.
inline thread_local int timing_depth = 0;

// Candidates are in preference order. A later one must be at least `margin` faster than the current choice (5%).
// A false return means that candidate cannot take this shape. `scale` accounts for a timed slice of the work.
// `warm_s`: after each candidate's first run, the candidates run in turn for this long before the timings, so that a
// GPU that idled clocks up first (0: none, the choices of short products, whose callers count every millisecond).
struct TimedCandidate {
    std::function<bool()> run;
    double scale = 1.0;
};
struct TimedChoice {
    int pick = -1;
    std::vector<double> seconds;
};

inline TimedChoice timed_choice(const std::vector<TimedCandidate>& candidates, const std::function<void()>& wait,
                                int samples = 5, double margin = 1.05, double warm_s = 0.0) {
    struct Guard { Guard() { ++timing_depth; } ~Guard() { --timing_depth; } } guard;
    TimedChoice result;
    result.seconds.assign(candidates.size(), -1.0);
    wait();
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].run()) {
            wait();
            result.seconds[i] = std::numeric_limits<double>::infinity();
        }
    }
    if (warm_s > 0) {
        const auto start = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < warm_s) {
            bool any = false;
            for (size_t i = 0; i < candidates.size(); ++i)
                if (result.seconds[i] >= 0 && candidates[i].run()) any = true;
            wait();
            if (!any) break;
        }
    }
    for (int sample = 0; sample < samples; ++sample) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (result.seconds[i] < 0) continue;
            const auto start = std::chrono::steady_clock::now();
            if (!candidates[i].run()) { result.seconds[i] = -1.0; continue; }
            wait();
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            result.seconds[i] = std::min(result.seconds[i], seconds * candidates[i].scale);
        }
        // As in dense products: after two rounds, only a wide gap is conclusive. Ties take five rounds.
        if (sample == 1) {
            double first = std::numeric_limits<double>::infinity(), second = first;
            for (double t : result.seconds) if (t >= 0) {
                if (t < first) { second = first; first = t; }
                else if (t < second) second = t;
            }
            if (candidates.size() == 2 && result.seconds[0] >= 0 && result.seconds[1] >= 0) {
                if (result.seconds[0] * 1.25 < result.seconds[1] ||
                    result.seconds[1] * 1.20 < result.seconds[0]) break;
            } else if (first * 1.25 < second) break;
        }
    }
    for (size_t i = 0; i < result.seconds.size(); ++i) {
        const double t = result.seconds[i];
        if (t >= 0 && (result.pick < 0 || t * margin < result.seconds[(size_t) result.pick])) result.pick = (int) i;
    }
    return result;
}

}  // namespace strata::core
