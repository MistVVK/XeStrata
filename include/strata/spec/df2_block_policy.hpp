// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// The DFlash2 block-width rule after YASAKANI's specx.
#pragma once

#include "strata/kernels/window_limits.hpp"

#include <algorithm>

namespace strata::spec {

class Df2BlockPolicy {
public:
    enum class Mode { Short, Long, Adaptive };
    explicit Df2BlockPolicy(Mode mode = Mode::Adaptive) : mode_(mode) {}

    int choose(int room) const {
        if (room <= kernels::kVerifyNarrowT || mode_ == Mode::Short) return kernels::kVerifyNarrowT;
        if (mode_ == Mode::Long || extra_ >= std::max(0.6, 0.22 * short_) || since_ >= 256 ||
            (full_ >= 0.45 && since_ >= 4)) return kernels::kWindowMaxT;
        return kernels::kVerifyNarrowT;
    }

    // accepted excludes the anchor. Do not teach from a truncated or stopped verification.
    void observe(int width, int accepted) {
        if (mode_ != Mode::Adaptive || (width != kernels::kVerifyNarrowT && width != kernels::kWindowMaxT)) return;
        accepted = std::clamp(accepted, 0, width - 1);
        short_ = 0.9 * short_ + 0.1 * (std::min(accepted, kernels::kVerifyNarrowT - 1) + 1);
        if (width == kernels::kWindowMaxT) {
            const double extra = std::max(0, accepted - (kernels::kVerifyNarrowT - 1));
            extra_ = seen_long_ ? 0.9 * extra_ + 0.1 * extra : extra;
            seen_long_ = true;
            since_ = 0;
        } else {
            ++since_;
            full_ = 0.75 * full_ + 0.25 * (accepted >= kernels::kVerifyNarrowT - 1 ? 1.0 : 0.0);
            extra_ *= 0.97;
        }
    }

private:
    Mode mode_;
    double extra_ = 0.0, short_ = 4.0, full_ = 0.0;
    int since_ = 0;
    bool seen_long_ = false;
};

}  // namespace strata::spec
