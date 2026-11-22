// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Harin Mani Karri
// Copyright (c) 2026 Strata Contributors
//
// include/strata/core/moe_wave_plan.hpp - Layer-Major NVMe Prefill Wave Planner (`pe128` + Order A).
//
// Given the routed experts across a prompt chunk [0, T), computes prompt-end recency weights:
//   w_pe(e) = sum_{t in T(e)} 2^{-(T - 1 - t) / 128}
// and builds an Order A wave schedule over a bounded cache of S slots:
//   - Wave 0: all currently resident active experts R (0 disk reads!)
//   - Pin budget P = min(|R|, max(0, floor((W * S - |U|) / (W - 1)))) keeps the hottest w_pe residents
//     pinned across waves so they survive prefill into decode token 1 without re-reading.
//   - Misses M = U \ R are sorted ASCENDING by w_pe(e), so cold misses stream first and the hottest
//     prompt-end misses land in the FINAL wave and remain resident when decode begins!
//   - Every distinct miss in U is read AT MOST ONCE (<= 1.00 reads/expert; ~0.98-1.00 on warm cache).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace strata::core {

struct MoeWavePlan {
    /// Experts already resident in cache (Wave 0, zero NVMe reads).
    std::vector<int32_t> wave0_residents;
    /// Subset of `wave0_residents` (highest w_pe) to pin across cold waves so they survive into decode.
    std::vector<int32_t> pinned_residents;
    /// Cold misses partitioned into sequential waves; misses are ordered ASCENDING by w_pe so the
    /// hottest prompt-end experts land in the final wave and stay resident for decode token 1.
    std::vector<std::vector<int32_t>> cold_waves;
    /// Total distinct misses scheduled across all cold waves (= sum of cold_waves[w].size()).
    size_t total_misses = 0;
    /// Total distinct active experts (= wave0_residents.size() + total_misses).
    size_t total_active = 0;
    /// Expected reads per active expert (= total_misses / total_active, <= 1.0).
    double reads_per_expert() const noexcept {
        return total_active > 0 ? static_cast<double>(total_misses) / static_cast<double>(total_active) : 0.0;
    }
};

/// Compute `pe128` prompt-end recency score for each expert e in [0, n_experts):
///   w_pe[e] = sum_{t, k : ids[t * top_k + k] == e} 2^{-(T - 1 - t) / half_life}
inline std::vector<float> compute_pe128_scores(
    const int32_t* token_expert_ids, int64_t n_tokens, int64_t top_k, int64_t n_experts,
    double half_life = 128.0) {
    std::vector<float> w(static_cast<size_t>(std::max<int64_t>(0, n_experts)), 0.0f);
    if (!token_expert_ids || n_tokens <= 0 || top_k <= 0 || n_experts <= 0) return w;
    const double inv_hl = 1.0 / std::max(1.0, half_life);
    const int64_t t_end = n_tokens - 1;
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float decay = static_cast<float>(std::exp2(-static_cast<double>(t_end - t) * inv_hl));
        const int32_t* row = token_expert_ids + t * top_k;
        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t e = row[k];
            if (e >= 0 && e < n_experts) w[static_cast<size_t>(e)] += decay;
        }
    }
    return w;
}

/// Build an Order A wave plan for `active_experts` given:
///   - `is_resident[e]`: non-zero iff expert `e` is already in cache
///   - `w_pe[e]`: prompt-end recency weight (from `compute_pe128_scores`)
///   - `cache_slots`: total cache capacity S (>= 1)
///   - `max_wave_slots`: optional cap per cold wave (e.g., S/2 for double-buffered I/O overlap); 0 = auto
inline MoeWavePlan plan_moe_waves(
    const std::vector<int32_t>& active_experts,
    const std::vector<uint8_t>& is_resident,
    const std::vector<float>& w_pe,
    int cache_slots,
    int max_wave_slots = 0) {
    MoeWavePlan plan{};
    const int S = std::max(1, cache_slots);

    // 1. Deduplicate and split U into residents R and misses M.
    std::vector<int32_t> uniq = active_experts;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    std::vector<int32_t> R, M;
    R.reserve(uniq.size());
    M.reserve(uniq.size());
    for (int32_t e : uniq) {
        if (e < 0) continue;
        const bool res = (static_cast<size_t>(e) < is_resident.size() && is_resident[static_cast<size_t>(e)] != 0);
        if (res) R.push_back(e);
        else     M.push_back(e);
    }

    auto score_of = [&](int32_t e) -> float {
        return (static_cast<size_t>(e) < w_pe.size()) ? w_pe[static_cast<size_t>(e)] : 0.0f;
    };

    // Sort residents DESCENDING by w_pe so hottest prompt-end residents are pinned first.
    std::sort(R.begin(), R.end(), [&](int32_t a, int32_t b) {
        const float sa = score_of(a), sb = score_of(b);
        return (sa != sb) ? (sa > sb) : (a < b);
    });
    // Sort misses ASCENDING by w_pe (Order A): coldest misses stream in early waves,
    // hottest prompt-end misses stream in the LAST wave so they stay resident for decode!
    std::sort(M.begin(), M.end(), [&](int32_t a, int32_t b) {
        const float sa = score_of(a), sb = score_of(b);
        return (sa != sb) ? (sa < sb) : (a < b);
    });

    plan.wave0_residents = R;
    plan.total_misses = M.size();
    plan.total_active = R.size() + M.size();
    if (M.empty()) {
        plan.pinned_residents = R;
        return plan;
    }

    const int U = static_cast<int>(plan.total_active);
    const int nR = static_cast<int>(R.size());
    const int nM = static_cast<int>(M.size());

    // Compute minimum wave count W >= 2 when U > S, else W = 1 (all fit in S simultaneously!).
    int W = 1;
    int P = 0;
    if (U <= S) {
        // Every active expert (R + M) fits simultaneously in S! Pin all residents R.
        P = nR;
        W = 1;
    } else {
        // Choose W >= 2 waves; pin budget P = floor((W * S - U) / (W - 1)) clamped to [0, min(nR, S - 1)].
        // Ensure at least 1 unpinned slot exists for streaming misses.
        const int wave_cap_hint = (max_wave_slots > 0) ? std::min(S, max_wave_slots) : S;
        const int min_waves_for_m = std::max(1, (nM + wave_cap_hint - 1) / wave_cap_hint);
        W = std::max(2, min_waves_for_m + (nR > 0 ? 1 : 0));
        // Pinning formula: across W total waves (1 resident wave + (W-1) cold waves),
        // keeping P residents pinned leaves (S - P) slots per cold wave, holding (W-1)*(S - P) >= nM ->
        // P <= floor(((W-1)*S - nM) / (W - 1)) = S - ceil(nM / (W-1)), equivalently when W includes wave0:
        // P = floor((W * S - U) / (W - 1)) when nR <= S.
        const int64_t num = static_cast<int64_t>(W) * S - U;
        const int p_formula = (num > 0) ? static_cast<int>(num / (W - 1)) : 0;
        const int max_p_for_m = std::max(0, S - std::min(S, nM));
        P = std::min({nR, S - 1, std::max(p_formula, max_p_for_m)});
        if (P < 0) P = 0;
    }

    plan.pinned_residents.assign(R.begin(), R.begin() + P);

    const int avail_slots = std::max(1, S - P);
    const int wave_sz = (max_wave_slots > 0) ? std::max(1, std::min(avail_slots, max_wave_slots)) : avail_slots;

    for (size_t off = 0; off < M.size(); off += static_cast<size_t>(wave_sz)) {
        const size_t end = std::min(M.size(), off + static_cast<size_t>(wave_sz));
        plan.cold_waves.emplace_back(M.begin() + static_cast<ptrdiff_t>(off),
                                     M.begin() + static_cast<ptrdiff_t>(end));
    }
    return plan;
}

}  // namespace strata::core
