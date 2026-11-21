// include/strata/core/device_hits_plan.hpp - partial-VRAM on-device hit planning and tail-skip weight renormalization.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// In hybrid VRAM+RAM decode with --pcie-frac 0 (zero PCIe expert staging), every GPU-computed expert of a verify window
// comes from the on-device VRAM cache (`res_layer[e] >= 0`), while the CPU worker pool computes the missed experts
// (`res_layer[e] < 0`) into mapped `m_ymiss_`.  Building the VRAM hit plan on the device (`resident_hits_plan`, matching
// `expert_pool_dispatch_multi`'s exact layout when `fetches == 0`) eliminates the per-layer host-spin barriers on
// `m_flagA_` and `m_flagB_`, the mapped `m_plan_` copy, and the three empty PCIe staging kernel launches (`fetch_blobs`,
// `rebase_ptrs`, `grouped`), letting GPU VRAM hits overlap 100% with CPU RAM misses while keeping bitwise-identical
// expert assignments to `--pcie-frac 0`.
//
// Also provides the reference contract for `STRATA_ROUTE_TAIL_RENORM=1`, which renormalizes each token's remaining
// routing weights to preserve its router weight sum when `STRATA_ROUTE_TAIL_SKIP > 0` drops low-rank missed experts.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace strata::core {

inline bool device_hits_env_on(const char* env) {
    return env != nullptr && std::atoi(env) != 0;
}

inline bool route_tail_renorm_env_on(const char* env) {
    return env != nullptr && std::atoi(env) != 0;
}

/// Host reference of `kernels::resident_hits_plan` (and `expert_pool_dispatch_multi` when `fetches == 0`):
/// groups distinct resident experts (`0 <= e < n_expert && res_layer[e] >= 0`) in first-occurrence routing order,
/// emits their ascending entry indices into `dst[0..hit_entries)` and token indices into `tok[0..hit_entries)`,
/// writes `counts = {hit_groups, hit_entries, 0}`, `start[hit_groups] = hit_entries`, and `start2[0] = hit_entries`.
inline void build_device_hits_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                                   uintptr_t cache_base, const unsigned long long* slot_off, long long blob,
                                   int32_t* plan, long long capx) {
    int32_t* counts = plan;
    int32_t* start = plan + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = reinterpret_cast<unsigned long long*>(plan + ptr_off);
    int32_t* start2 = plan + ptr_off + 4 * capx;

    int groups = 0;
    int entries = 0;
    for (int i = 0; i < n_entries; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert) continue;
        const int32_t slot = res_layer[e];
        if (slot < 0) continue;
        bool seen = false;
        for (int j = 0; j < i; ++j) {
            if (ids[j] == e) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        ptr[groups] = (unsigned long long) cache_base +
                      (slot_off != nullptr ? slot_off[slot] : (unsigned long long) slot * (unsigned long long) blob);
        start[groups] = entries;
        for (int m = i; m < n_entries; ++m) {
            if (ids[m] == e) {
                dst[entries] = m;
                tok[entries] = m / k;
                ++entries;
            }
        }
        ++groups;
    }
    start[groups] = entries;
    start2[0] = entries;
    counts[0] = groups;
    counts[1] = entries;
    counts[2] = 0;
}

/// Host reference of `kernels::route_tail_renorm`: for a window of `n_tok` rows (`k` routed experts per row), any
/// non-resident expert (`0 <= e < n_expert && res_layer[e] < 0`) that EVERY token in the window routes only at
/// rank `>= tail_rank` has its routing weight zeroed, and each token's remaining weights are rescaled so their sum
/// equals the token's original router weight sum.
inline void renormalize_tail_skipped_weights(const int32_t* ids, float* weights, const int32_t* res_layer,
                                             int n_tok, int k, int n_expert, int tail_rank) {
    if (tail_rank <= 0 || n_tok <= 0 || k <= 0) return;
    const int n = n_tok * k;
    auto is_skipped = [&](int idx) -> bool {
        const int32_t e = ids[idx];
        if (e < 0 || e >= n_expert || res_layer[e] >= 0) return false;
        int min_r = k;
        for (int j = 0; j < n; ++j) {
            if (ids[j] == e) min_r = std::min(min_r, j % k);
        }
        return min_r >= tail_rank;
    };
    for (int t = 0; t < n_tok; ++t) {
        float orig_sum = 0.0f;
        float kept_sum = 0.0f;
        bool skipped_mask[32] = {};
        for (int r = 0; r < k; ++r) {
            const float w = weights[t * k + r];
            orig_sum += w;
            const bool sk = r < 32 ? is_skipped(t * k + r) : false;
            if (r < 32) skipped_mask[r] = sk;
            if (!sk) kept_sum += w;
        }
        if (kept_sum > 0.0f && kept_sum < orig_sum) {
            const float scale = orig_sum / kept_sum;
            for (int r = 0; r < k; ++r) {
                weights[t * k + r] = (r < 32 && skipped_mask[r]) ? 0.0f : (weights[t * k + r] * scale);
            }
        }
    }
}

}  // namespace strata::core
