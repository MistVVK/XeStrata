// Host-only opt-in grouping and readiness-tier permutation. No HIP calls, device math, or precision changes.
//
// Readiness-tier work-queue partitioning (Tier 0 VRAM resident -> Tier 1 pinned host RAM -> Tier 2 unpinned/file
// stager) combined with within-tier row-count (length-aware) MMQ group formation is extrapolated from:
//   - Multi-V-VM/GPUOS: Yiwei Yang, Xiangyu Gao, Yuan Zhou, Yuhang Gan, Yusheng Zheng, and Andi Quinn,
//     "GPUOS: An Operating-System Architecture for Shared GPU Execution" (https://github.com/Multi-V-VM/GPUOS,
//     manuscript §3 and §5.3).
//   - Reza Nourai and Behnam Bastani, "The GPU Needs an Operating System", arXiv:2611.08720
//     (https://arxiv.org/abs/2611.08720).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::prefill::detail {

inline bool mmq_resident_sort_flag(const char* value) {
    return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool mmq_resident_sort_requested() {
    static const bool on = mmq_resident_sort_flag(std::getenv("STRATA_MMQ_RESIDENT_SORT_NE"));
    return on;
}

// STRATA_MMQ_GROUP_SORT=1: the same row-count order for the full 16-expert groups only; the last, partial group keeps
// its experts in ID order. A full group's launch has 160 * ntx (gate/up) or 320 * ntx (down) tiles, at or above the
// 90 % tile efficiency on the cards' SM counts, so it runs one block per tile with no stream-k K split whatever its tile
// width: each expert's rows get the same bits in any full group. The partial group can take the stream-k split, whose
// K boundaries depend on the group's tile count - leaving it unchanged keeps the prompt's bits.
inline bool mmq_group_sort_requested() {
    static const bool on = mmq_resident_sort_flag(std::getenv("STRATA_MMQ_GROUP_SORT"));
    return on;
}

// STRATA_MMQ_TIER_SORT=1: GPUOS-inspired readiness-tier + row-count expert ordering across staged (!stream_all) and
// layer-major (lm_run) MMQ prefill walks on any GPU architecture (from Volta sm_70 V100 through Ada sm_89 and newer).
inline bool mmq_tier_sort_requested() {
    static const bool on = mmq_resident_sort_flag(std::getenv("STRATA_MMQ_TIER_SORT"));
    return on;
}

// Operand readiness tier for an expert `e`:
//   0: VRAM-resident in the expert cache (`layer_resident[e] >= 0`)
//   1: Page-locked (pinned) host RAM (`layer_pinned[e] != 0`)
//   2: Unpinned host / file tier (staged via background Stager threads)
inline int mmq_expert_tier(int32_t e, const int32_t* layer_resident, const char* layer_pinned = nullptr) {
    if (layer_resident != nullptr && layer_resident[(size_t) e] >= 0) return 0;
    if (layer_pinned != nullptr && layer_pinned[(size_t) e] != 0) return 1;
    return 2;
}

// How many of `n` ordered experts the sort may move: all of them (STRATA_MMQ_RESIDENT_SORT_NE / STRATA_MMQ_TIER_SORT),
// or the full groups of `group` experts (STRATA_MMQ_GROUP_SORT).
inline size_t mmq_resident_sort_span(size_t n, size_t group, bool keep_tail) {
    return keep_tail && group > 0 ? n - n % group : n;
}

inline bool mmq_resident_sort_eligible(bool requested, bool use_mmq, bool native_layout, bool has_cache,
                                       const int32_t* layer_resident, int64_t n_expert,
                                       bool stream_all, bool layer_stream_empty,
                                       bool allow_staged_tiers = false) {
    // native-layout packs too: slot/src/off are built the same way for both layouts, and nothing reads off[e + 1] as a
    // count (4x GP100, sm_60: 1.2K prompt 7.27 -> 5.32 s, 35K 63.3 -> 54.1 s, the same bits)
    (void) native_layout;
    if (!requested || !use_mmq || n_expert <= 0) return false;
    // In the staged walk (!stream_all) with STRATA_MMQ_TIER_SORT=1, bounds_host, Stager jobs (job_of), and stage_one
    // are all built after `order` is finalized and index by `order[j]` directly, so partially cached and uncached
    // layers are eligible too.
    if (allow_staged_tiers && !stream_all) return true;
    // In the streamed walk (stream_all) or resident-only sort, non-resident experts stream in fixed ascending ID order
    // (`seq`), so only layers whose experts are all VRAM-resident may reorder `order`.
    if (!has_cache || !layer_resident || !layer_stream_empty) return false;
    for (int64_t e = 0; e < n_expert; ++e) if (layer_resident[e] < 0) return false;
    return true;
}

// Preconditions are established by the existing caller: IDs were range-checked and counted;
// order contains each positive-count GPU-local expert once in ID order; off has count.size()+1 entries;
// routes fits its original int32 offsets, and slot/src each have routes writable entries.
// After sorting, off[e] is an expert-indexed START, not an ID-prefix end at off[e+1].
// Only order[0, span) is sorted (span >= order.size(): all of it); the entries after it keep their positions.
// When `layer_resident` or `layer_pinned` is supplied, experts in order[0, span) are partitioned by readiness tier
// (Tier 0 VRAM resident, Tier 1 pinned host RAM, Tier 2 unpinned/file stager) and sorted by ascending row count
// within each tier. On a fully resident layer every expert has tier 0, reducing identically to row-count sort.
// Experts outside `order` (e.g. CPU-shared `on_cpu` or peer `on_peer` blocks) keep their existing `off[e]` starts.
inline void mmq_resident_sort_rows(const int32_t* ids, int64_t routes, int32_t k,
                                   const std::vector<int32_t>& count, std::vector<int32_t>& off,
                                   std::vector<int32_t>& order, int32_t* slot, int32_t* src,
                                   size_t span = SIZE_MAX,
                                   const int32_t* layer_resident = nullptr,
                                   const char* layer_pinned = nullptr) {
    const bool use_tiers = layer_resident != nullptr || layer_pinned != nullptr;
    std::stable_sort(order.begin(), order.begin() + (ptrdiff_t) std::min(span, order.size()), [&](int32_t a, int32_t b) {
        if (use_tiers) {
            const int ta = mmq_expert_tier(a, layer_resident, layer_pinned);
            const int tb = mmq_expert_tier(b, layer_resident, layer_pinned);
            if (ta != tb) return ta < tb;
        }
        return count[(size_t) a] != count[(size_t) b]
            ? count[(size_t) a] < count[(size_t) b] : a < b;
    });
    for (size_t e = 0; e + 1 < off.size(); ++e)
        if (count[e] == 0) off[e] = 0;
    int32_t cursor = 0;
    for (int32_t e : order) {
        off[(size_t) e] = cursor;
        cursor += count[(size_t) e];
    }
    if (!off.empty() && (int64_t) cursor == routes) off.back() = cursor;
    std::vector<int32_t> fill(off.begin(), off.end() - 1);
    // Preserve token-major/rank-major traversal, hence the original within-expert row order.
    for (int64_t i = 0; i < routes; ++i) {
        const int32_t e = ids[(size_t) i];
        const int32_t p = fill[(size_t) e]++;
        slot[(size_t) i] = p;
        src[(size_t) p] = (int32_t) (i / k);
    }
}

// Layer-major prefill (`lm_experts`): orders `uniq` by readiness tier (Tier 0 VRAM resident, Tier 1 pinned host RAM,
// Tier 2 unpinned/file stager) and by ascending row count `count[e]` within each tier. When `keep_tier2_id_order` is
// true (`STRATA_LM_PREFETCH=1`), Tier 2 experts keep ascending ID order so their prefetched `Stager` job indices stay
// monotonic for `Stager::pass` and `Stager::issued_one`. Updates `off[e]` as each expert's contiguous row start in
// `uniq` order.
inline void mmq_lm_sort_uniq(const std::vector<int32_t>& count, std::vector<int32_t>& off,
                             std::vector<int32_t>& uniq, const int32_t* layer_resident,
                             const char* layer_pinned, bool keep_tier2_id_order) {
    std::stable_sort(uniq.begin(), uniq.end(), [&](int32_t a, int32_t b) {
        const int ta = mmq_expert_tier(a, layer_resident, layer_pinned);
        const int tb = mmq_expert_tier(b, layer_resident, layer_pinned);
        if (ta != tb) return ta < tb;
        if (ta == 2 && keep_tier2_id_order) return a < b;
        return count[(size_t) a] != count[(size_t) b]
            ? count[(size_t) a] < count[(size_t) b] : a < b;
    });
    for (size_t e = 0; e + 1 < off.size(); ++e)
        if (count[e] == 0) off[e] = 0;
    int32_t cursor = 0;
    for (int32_t e : uniq) {
        off[(size_t) e] = cursor;
        cursor += count[(size_t) e];
    }
    if (!off.empty()) off.back() = cursor;
}

}  // namespace strata::prefill::detail
