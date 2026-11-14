// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/spec/drafter_policy.cpp - see include/strata/spec/drafter_policy.hpp.
#include "strata/spec/drafter_policy.hpp"

#include <algorithm>
#include <cmath>

namespace strata::spec {

uint64_t DrafterPolicy::kind_of(const int64_t* ids, int64_t n) {
    uint64_t key = 1469598103934665603ull;
    const int64_t lim = std::min<int64_t>(n, 33);
    for (int64_t i = 1; i < lim; ++i) key = (key ^ (uint64_t) ids[i]) * 1099511628211ull;
    return key;
}

int DrafterPolicy::depth_bucket(int64_t pos) {
    if (pos < 2048) return 0;
    if (pos < 8192) return 1;
    if (pos < 32768) return 2;
    return 3;
}

DrafterPolicy::Arm DrafterPolicy::choose(uint64_t kind, int64_t pos, bool second_avail) {
    if (!second_avail) return First;
    Cell& c = cells_[kind];
    Bucket& b = c.b[depth_bucket(pos)];
    // explore each arm twice before exploiting (the MTP first: the conservative startup prior)
    if (b.n[Mtp] < 2) return Mtp;
    if (b.n[Df2] < 2) return Df2;
    const double t0 = b.tps[Mtp], t1 = b.tps[Df2];
    const bool df_better = t1 >= second_margin_ * t0;   // the second arm unless the first is clearly ahead
    const int every = (t0 > 0 && std::fabs(t1 / t0 - 1.0) < 0.04) ? 6 : 12;
    const int64_t r = b.reqs++;
    const bool explore = r % every == every - 1;
    if (explore) return df_better ? Mtp : Df2;
    return df_better ? Df2 : Mtp;
}

void DrafterPolicy::observe(uint64_t kind, int64_t pos, Arm arm, int64_t committed, double decode_ms) {
    if (committed < min_tokens_ || decode_ms <= 0.0 || (int) arm < 0 || (int) arm >= kArms) return;
    const double speed = (double) committed / (decode_ms / 1000.0);
    Cell& c = cells_[kind];
    ++c.reqs;
    Bucket& b = c.b[depth_bucket(pos)];
    double& cur = b.tps[arm];
    cur = b.n[arm] == 0 ? speed : 0.7 * cur + 0.3 * speed;   // EMA: adapts to the machine and the text
    ++b.n[arm];
}

double DrafterPolicy::tps(uint64_t kind, int64_t pos, Arm arm) const {
    auto it = cells_.find(kind);
    if (it == cells_.end()) return 0.0;
    return it->second.b[depth_bucket(pos)].tps[arm];
}

int64_t DrafterPolicy::requests(uint64_t kind) const {
    auto it = cells_.find(kind);
    return it == cells_.end() ? 0 : it->second.reqs;
}

}  // namespace strata::spec