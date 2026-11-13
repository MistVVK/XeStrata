// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/spec/drafter_policy.hpp - which drafter a request uses: the MTP layer or DFlash2.
//
// The two drafters propose different candidates at different costs, and which is faster depends on the kind of
// request (its system prompt) and how deep the context is (a DFlash2 ring row costs the same at any depth; the MTP
// layer's attention and the verify window cost more as the context grows).  This learns, per (kind, depth) cell,
// each drafter's measured decode speed and picks the faster, exploring the other now and then.  It only chooses
// which drafter proposes: the verify window decides every emitted token, so the output is unchanged.
//
// Nothing is written to disk: a new process measures again from its first requests (the user's decision).  The
// startup prior is the MTP (conservative: DFlash2 tied or lost on the Arc A-series in upstream's measurement, and
// the RDNA4 prompt stall is open); the first requests of each kind measure both arms and the faster one wins.
#pragma once

#include <cstdint>
#include <map>

namespace strata::spec {

class DrafterPolicy {
public:
    enum Arm { Mtp = 0, Df2 = 1 };
    static constexpr int kArms = 2;
    static constexpr int kDepthBuckets = 4;   // [0,2K) [2K,8K) [8K,32K) [32K,...)
    static constexpr int kMinTokens = 48;     // a request shorter than this teaches nothing (its speed is noise)

    DrafterPolicy() = default;

    /// The request's kind: FNV-1a of its first prompt tokens (the system prompt decides the kind; the first token,
    /// a BOS, is skipped).  Upstream's bandit key.
    static uint64_t kind_of(const int64_t* ids, int64_t n);
    /// The context-depth bucket of a position.
    static int depth_bucket(int64_t pos);

    /// Pick the drafter for a request of `kind` starting at context position `pos`.  `df2_avail`: DFlash2 is loaded.
    /// Each arm is tried twice in a cell before the faster is exploited; the loser is retried every 6th request
    /// when the two are within 4%, every 12th otherwise.  Without DFlash2 the MTP is the only arm.
    Arm choose(uint64_t kind, int64_t pos, bool df2_avail);
    /// After a request: the arm it used, the tokens it committed, its decode time (ms).  Only >= kMinTokens teaches.
    void observe(uint64_t kind, int64_t pos, Arm arm, int64_t committed, double decode_ms);

    /// The measured decode tok/s of an arm in a cell (0: not measured yet).
    double tps(uint64_t kind, int64_t pos, Arm arm) const;
    /// How many requests of a kind have been seen (any depth).
    int64_t requests(uint64_t kind) const;

private:
    struct Bucket {
        double tps[kArms] = {0, 0};
        int n[kArms] = {0, 0};
        int64_t reqs = 0;
    };
    struct Cell {
        Bucket b[kDepthBuckets];
        int64_t reqs = 0;
    };
    std::map<uint64_t, Cell> cells_;
};

}  // namespace strata::spec