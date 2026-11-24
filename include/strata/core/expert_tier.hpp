// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/expert_tier.hpp - the rules of the expert tiers that Qwen3.8-Flash-Next's engine
// (generate.cpp) and the other models' (PlainModel) share: which experts an adaptive round swaps into the VRAM tier,
// which ones STRATA_RAM_ADAPT moves into the RAM copy, the copies back into the RAM copy's exchange buffers, and the
// bound of a round's copies (--adapt-mib-per-token).  When a round runs and how its copies are issued stay with each
// engine: those were measured apart (generate.cpp's adapt_lag and --adapt-async).
#pragma once

#include "strata/core/auto_pick.hpp"
#include "strata/core/device.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace strata::core {

/// One swap of a round: `in` (not in VRAM) takes the slot of `out` (in VRAM), or a free slot when `out` < 0.
struct TierSwap {
    float gain;
    int32_t layer, in, out;
};

/// The adaptive tier's rule.  In each layer the missing experts routed at least twice since the last round (their
/// decayed count; with free slots, any routed one for each free slot first) pair with the least routed experts in
/// VRAM, the most routed with the least, while the one coming in leads by `min_gain` (--adapt-min-gain); then every
/// layer's pairs by gain, the largest first.  `usage` and `res` are layer-major (`res` the slot of an expert in VRAM,
/// -1 when it is not); `free_slots(layer)` the layer's free slots; `skip(layer, expert)` leaves an expert out of the
/// candidates (STRATA_DISJOINT_ADAPT: one a helper GPU holds).
template <class FreeSlots, class Skip>
std::vector<TierSwap> choose_swaps(const float* usage, const int32_t* res, int64_t layers, int64_t n_expert,
                                   float min_gain, FreeSlots free_slots, Skip skip) {
    constexpr float kRouted = 2.0f;
    std::vector<TierSwap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < layers; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * n_expert;
        const int32_t* r = res + l * n_expert;
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
            if (r[e] < 0) { if (u[e] > 0.0f && !skip(l, e)) cand.emplace_back(u[e], e); }
            else vict.emplace_back(u[e], e);
        }
        // a free slot takes any expert routed since (an empty slot only costs a copy); one held, a missing expert
        // routed at least twice
        const size_t n_free = (size_t) free_slots(l);
        for (size_t i = 0; i < n_free; ++i) vict.emplace_back(-1.0f, -1);
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        size_t keep = 0;
        while (keep < cand.size() && (keep < n_free || cand[keep].first >= kRouted)) ++keep;
        cand.resize(keep);
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + min_gain) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](const TierSwap& a, const TierSwap& b) { return a.gain > b.gain; });
    return swaps;
}

/// choose_swaps for a tier without free slots and with every missing expert a candidate.
inline std::vector<TierSwap> choose_swaps(const float* usage, const int32_t* res, int64_t layers, int64_t n_expert,
                                          float min_gain) {
    return choose_swaps(usage, res, layers, n_expert, min_gain, [](int64_t) { return 0; },
                        [](int64_t, int32_t) { return false; });
}

/// --adapt-mib-per-token: the swaps a round may take, `cap` (--adapt-swaps) at most and, with `mib_per_token` > 0, at
/// most that many MiB of copies, at the largest expert's size, for each of the `tokens` read since the last round (at
/// least one swap).  The copies share the RAM's bandwidth with the missed experts the CPU and PCIe read.
inline int64_t adapt_swap_budget(int64_t cap, int64_t mib_per_token, int64_t tokens) {
    const uint64_t blob = kernels::cpu::expert_layout().max_blob;
    if (mib_per_token <= 0 || blob == 0) return cap;
    const uint64_t n = ((uint64_t) mib_per_token << 20) * (uint64_t) std::max<int64_t>(tokens, 0) / blob;
    return std::min<int64_t>(cap, std::max<int64_t>((int64_t) n, 1));
}

/// STRATA_RAM_ADAPT's rule (upstream d7298883, 2b13a4493): the RAM copy is chosen once, at start, so an expert it does
/// not hold is read from the file every time the conversation routes to it.  In each layer up to 4 of the experts the
/// windows route to most (a decayed count of 2 or more) of those neither in VRAM nor in the RAM copy take the RAM places
/// of the copy's least routed ones (below 0.5), paired in that order; all the layers' pairs, the largest difference
/// first.  `res` as choose_swaps'.
inline std::vector<TierSwap> choose_ram_swaps(const float* usage, const int32_t* res, int64_t layers, int64_t n_expert,
                                              const FileExpertSource& src) {
    constexpr float kRouted = 2.0f, kUnused = 0.5f;
    constexpr size_t kPerLayer = 4;
    std::vector<TierSwap> swaps;
    std::vector<int32_t> ins, outs;
    for (int64_t l = 0; l < layers; ++l) {
        const float* u = usage + l * n_expert;
        const int32_t* r = res + l * n_expert;
        ins.clear();
        outs.clear();
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
            if (r[e] >= 0) continue;   // in VRAM: the RAM copy does not hold it
            if (src.has_resident(l, e)) outs.push_back(e);
            else if (u[e] >= kRouted) ins.push_back(e);
        }
        const size_t ni = std::min(ins.size(), kPerLayer), no = std::min(outs.size(), kPerLayer);
        std::partial_sort(ins.begin(), ins.begin() + (std::ptrdiff_t) ni, ins.end(),
                          [&](int32_t a, int32_t b) { return u[a] > u[b]; });
        std::partial_sort(outs.begin(), outs.begin() + (std::ptrdiff_t) no, outs.end(),
                          [&](int32_t a, int32_t b) { return u[a] < u[b]; });
        for (size_t i = 0; i < ni && i < no && u[outs[i]] < kUnused; ++i)
            swaps.push_back({u[ins[i]] - u[outs[i]], (int32_t) l, ins[i], outs[i]});
    }
    std::sort(swaps.begin(), swaps.end(), [](const TierSwap& a, const TierSwap& b) { return a.gain > b.gain; });
    return swaps;
}

/// STRATA_RAM_ADAPT=<n>: one round of choose_ram_swaps' pairs.  The incoming expert was usually read for the window
/// before: a pair whose `in` still sits in that read's stage buffer moves with a copy out of it (no file read), and `n`
/// caps only the pairs that need the file read.  The caller's thread, between windows, as replace_resident.
inline void ram_adapt_round(const float* usage, const int32_t* res, int64_t layers, int64_t n_expert,
                            FileExpertSource& src, int n) {
    int file_left = n;
    for (const TierSwap& x : choose_ram_swaps(usage, res, layers, n_expert, src)) {
        if (src.replace_resident_staged(x.layer, x.in, x.out)) continue;
        if (file_left > 0) {
            --file_left;
            src.replace_resident(x.layer, x.in, x.out);
        }
    }
}

/// The plain engine's floor for the room no plan counts (core::vram_room): under STRATA_VRAM_LIMIT_MIB=4096 on
/// the RX 9060 XT the peak stayed below the card with 128 MiB kept free - llm-jp Q4_K_M, Ornith Q4_K_M and
/// GSQ-RCO 3bit, 2026-10-12 - where the 768 MiB of before left 700 MiB of the card unused.
inline constexpr uint64_t kTierRoomLeast = 192ull << 20;

/// What a verify window takes beside a VRAM tier, counted apart from it: its rows (`max_t` tokens of `k` routed
/// experts, three FP32 rows of `n_embd` each), sixteen experts in staging and 64 MiB.
inline uint64_t tier_window_bytes(int64_t max_t, int64_t k, int64_t n_embd, uint64_t max_blob) {
    return (uint64_t) max_t * (uint64_t) k * (uint64_t) n_embd * 4 * 3 + 16 * max_blob + (64ull << 20);
}

/// A VRAM tier's room, for every engine that fills one from the free memory (PlainModel's rule).  Of `free_b` (counted_free's) on a device of `total_b`: less `window_b` and less the reserve -
/// what is allocated after the tier (the products' libraries, the graphs) and the driver's margin, --vram-reserve-mib
/// (`reserve_mib` >= 0) or the room of the device's memory (core::vram_room, its floor kTierRoomLeast, its 768 MiB
/// ceiling past 24 GiB).  `per_layer_b`: one expert of every layer the tier holds; `max_slots`: --expert-cache N over
/// the `layers` (< 0: what fits).
struct TierRoom {
    uint64_t reserve = 0;   ///< what stays free beside the tier and the window
    uint64_t room = 0;      ///< the tier's bytes at most
    int64_t q = 0;          ///< the experts a layer takes when the layers take alike (`n_expert` at most)
    uint64_t bytes = 0;     ///< per_layer_b * q: the budget plan_profile_fill shares out
};
inline TierRoom plan_tier_room(uint64_t free_b, uint64_t total_b, uint64_t window_b, int64_t reserve_mib,
                               int64_t max_slots, uint64_t per_layer_b, int64_t layers, int64_t n_expert) {
    TierRoom t;
    t.reserve = reserve_mib >= 0 ? (uint64_t) reserve_mib << 20 : vram_room(total_b, /*parts*/ 1, kTierRoomLeast);
    t.room = free_b > window_b + t.reserve ? free_b - window_b - t.reserve : 0;
    t.q = per_layer_b > 0 ? (int64_t) (t.room / per_layer_b) : 0;
    t.q = std::min<int64_t>(t.q, n_expert);
    if (max_slots >= 0) t.q = std::min<int64_t>(t.q, max_slots / std::max<int64_t>(layers, 1));
    t.bytes = per_layer_b * (uint64_t) t.q;
    return t;
}

/// How many slots of a tier of `budget` bytes each layer takes, from its profile (`prior`, (layer, expert) pairs in
/// rank order): Qwen3.8-Flash-Next's fill (generate.cpp), the pairs in their order over all the layers while the bytes
/// last (`bytes(l)` a layer's expert, `local(l)` the layers this tier holds), then the rest of the bytes the layers in
/// turn; `alike`: every layer the same count instead (PlainModel's for Ornith-1.5, below).  `fill` the pairs that get
/// a slot, in rank order.
template <class Pairs>
struct ProfileFill {
    std::vector<int64_t> quota;   ///< per layer
    Pairs fill;                   ///< the profile's pairs within the quotas
};
template <class Pairs, class Bytes, class Local>
ProfileFill<Pairs> plan_profile_fill(const Pairs& prior, int64_t layers, int64_t n_expert, uint64_t budget, Bytes bytes,
                                     Local local, bool alike) {
    ProfileFill<Pairs> f;
    f.quota.assign((size_t) layers, 0);
    uint64_t left = budget;
    auto take = [&](int64_t l) {
        if (l < 0 || l >= layers || !local(l) || f.quota[(size_t) l] >= n_expert || bytes(l) > left) return false;
        ++f.quota[(size_t) l];
        left -= bytes(l);
        return true;
    };
    if (!alike)
        for (const auto& p : prior) take((int64_t) p.first);
    for (bool more = true; more;) {   // the rest (or, alike, all of it) the layers in turn
        more = false;
        for (int64_t l = 0; l < layers; ++l) more |= take(l);
    }
    std::vector<int64_t> used((size_t) layers, 0);
    for (const auto& p : prior)
        if ((int64_t) p.first >= 0 && (int64_t) p.first < layers && used[(size_t) p.first] < f.quota[(size_t) p.first]) {
            ++used[(size_t) p.first];
            f.fill.push_back(p);
        }
    return f;
}

/// The slots each layer of a tier of `room` takes, and the experts that take them at the start.  With a profile
/// (`prior` not null): plan_profile_fill's, over room.bytes.  Without one: room.q in each layer the tier holds, and
/// each of those layers' first experts, the layers in turn.
template <class Pairs, class Bytes, class Local>
ProfileFill<Pairs> plan_tier_fill(const Pairs* prior, int64_t layers, int64_t n_expert, const TierRoom& room,
                                  Bytes bytes, Local local, bool alike) {
    if (prior != nullptr) return plan_profile_fill(*prior, layers, n_expert, room.bytes, bytes, local, alike);
    ProfileFill<Pairs> f;
    f.quota.assign((size_t) layers, 0);
    for (int64_t l = 0; l < layers; ++l)
        if (local(l)) f.quota[(size_t) l] = room.q;
    for (int64_t e = 0; e < room.q; ++e)
        for (int64_t l = 0; l < layers; ++l)
            if (local(l)) f.fill.emplace_back((int32_t) l, (int32_t) e);
    return f;
}

/// How many of a tier's slots, in the order ExpertCache::open_sized lays them (each from a 256-byte boundary), fit
/// `limit` bytes: the free memory is read again just before the tier is opened, and what was allocated since the plan
/// comes off the last slots.
inline size_t tier_slots_within(const std::vector<int64_t>& slot_bytes, uint64_t limit) {
    uint64_t need = 0;
    size_t n = 0;
    for (; n < slot_bytes.size(); ++n) {
        need += ((uint64_t) slot_bytes[n] + 255) / 256 * 256;
        if (need > limit) break;
    }
    return n;
}

/// #463 without the stall (upstream d69424c9, #764): a verify window takes the adaptive tier's previous swaps only once
/// they are this many windows old, and then waits for their copies (which have landed by then).  The window that first
/// computes a swapped-in expert on the GPU depends only on the window count, as with #463's wait, so the answers are as
/// reproducible.  Default 2 (upstream 2725c23f7 on sm_86 and sm_60; on XeStrata's GPUs the RTX 4070's decode 59.1 ->
/// 64.2 tok/s, IQ2_XS, and the B70's equal, IQ3_XXS 1K and 32K within 0.7%, 2026-10-20); STRATA_ADAPT_LAG=1 is #463's
/// (the very next window waits for the copies).  Both engines' (generate.cpp, PlainModel); `windows` the engine's
/// default, STRATA_ADAPT_LAG over it.  `auto` (the default) lets the first windows where a swap is waiting choose
/// between the engine's default and 1 (auto_pick.hpp), on a pack whose layers round alike on the CPU and the GPU.
inline int adapt_lag(int windows = 2) {
    const int forced = adapt_lag_forced();
    if (forced > 0) return forced;
    const int chosen = adapt_lag_chosen();
    if (chosen > 0) return chosen;
    return adapt_lag_forming() == 1 ? 1 : windows;   // the trial: the next window waits, and its cost is measured
}

/// The adaptive tier's period when --adapt-every is not given, for both engines (generate.cpp, PlainModel): chosen once,
/// after kAdaptDecideAt tokens of output, from `share`, the CPU pool's share of the verify windows' time.  A round
/// every 4 windows (16 tokens) while the pool takes a small share, every window (4 tokens) once it takes
/// kAdaptPoolShare or more.  A round more often moves the experts the text has turned to into VRAM sooner, which pays
/// where the CPU computes many missed experts; where it computes few the rounds cost more than they save.  Measured
/// 2026-10-23 on one 512-token text that every setting followed (decode, 3 runs each): the pool's share and every
/// window against every 4 - B70 0.10-0.13, IQ3_XXS 1K equal, Gyro-M 1K 6% slower; RX 9060 XT 0.49-0.52, IQ3_XXS
/// +1.7%, Gyro-M +8%; RTX 4070 0.61-0.64, IQ3_XXS +5.5%, Gyro-M +6%.  The share depends on the card's VRAM against the
/// model, the CPU and the link, not on a device's name.  Chosen once and at a position of the text, so the swaps stay
/// at fixed positions (window-invariant output, #152); the 4-token positions include the 16-token ones before them.
/// `from_files`: the experts the GPU does not hold are read from the model files (--mmap-experts, no RAM copy), and a
/// swap's blob is a file read too: there the period stays.  Measured 2026-10-24 on the RX 9060 XT in a 32 GiB cgroup
/// (IQ3_XXS, the page cache emptied before each run, decode): every 4 tokens 23.6-24.2 tok/s at 1K and 14.7 at 32K,
/// every 16 25.5 and 16.9, every 32 25.3-26.5 and 16.4, every 64 25.8 and 15.9 - the pool's share was 0.85.
inline constexpr int64_t kAdaptDecideAt = 32;
inline constexpr double kAdaptPoolShare = 0.3;

inline int64_t auto_adapt_period(double share, int64_t period, bool from_files, const char* who) {
    const int64_t next = share >= kAdaptPoolShare && !from_files ? std::min<int64_t>(period, 4) : period;
    std::fprintf(stderr, "%s: adaptive tier every %lld tokens: the CPU pool took %.0f%% of the verify windows "
                         "(a round every window from %.0f%%%s; --adapt-every sets it)\n",
                 who, (long long) next, 100.0 * share, 100.0 * kAdaptPoolShare,
                 from_files ? ", not while the experts are read from the files" : "");
    return next;
}

/// After a prompt, the adaptive tier's counts start from the prompt's routing instead of from nothing.  A count is the
/// rows routed to an expert, decayed by `decay` once a period of `period` tokens, so an expert the prompt routed with
/// probability p holds p * period / (1 - decay) rows at the tier's steady state: that is added for the routings
/// `routed` holds beyond `seen` (a request's own prompt in serve, where the counts run on from request to request).
/// `routed` [layers][n_expert], `routed_k` the experts a row takes.  The tier's rounds then swap by the counts as before.
/// STRATA_USAGE_SEED=0: the counts start from nothing.
inline void seed_usage(std::vector<float>& usage, const std::vector<int64_t>& routed, int64_t routed_k,
                       std::vector<int64_t>& seen, int64_t n_expert, int64_t period, double decay) {
    static const bool on = [] { const char* v = strata::env_get("USAGE_SEED"); return v == nullptr || *v != '0'; }();
    if (!on || usage.empty() || routed.empty() || routed_k <= 0 || usage.size() != routed.size() || n_expert <= 0 ||
        period <= 0)
        return;
    if (seen.size() != routed.size()) seen.assign(routed.size(), 0);
    const double rows = (double) period / std::max(1e-3, 1.0 - decay);
    for (size_t l0 = 0; l0 + (size_t) n_expert <= routed.size(); l0 += (size_t) n_expert) {
        int64_t total = 0;
        for (size_t i = l0; i < l0 + (size_t) n_expert; ++i) total += routed[i] - seen[i];
        if (total <= 0) continue;   // not a MoE layer, or nothing new
        for (size_t i = l0; i < l0 + (size_t) n_expert; ++i)
            usage[i] += (float) ((double) (routed[i] - seen[i]) * (double) routed_k / (double) total * rows);
    }
    seen = routed;
}

/// The decode's counts from the rows a window kept (the drafts the model accepted and the token after them), not from
/// every row it computed: `rec` [layers][max_t][k] the window's routing (ExpertDispatch::route_rec), its first `rows`
/// rows counted for the layers in `layers` (Qwen3.8's Verifier::add_routing does the same from its own record).
template <class Layers>
void add_route_rows(std::vector<float>& usage, const std::vector<int32_t>& rec, const Layers& layers, int64_t n_expert,
                    int64_t max_t, int64_t k, int rows) {
    rows = (int) std::min<int64_t>(rows, max_t);
    for (int64_t l : layers) {
        if ((size_t) ((l + 1) * max_t * k) > rec.size()) continue;
        for (int t = 0; t < rows; ++t)
            for (int64_t j = 0; j < k; ++j) {
                const int32_t e = rec[(size_t) ((l * max_t + t) * k + j)];
                if (e >= 0 && e < n_expert) usage[(size_t) (l * n_expert + e)] += 1.0f;
            }
    }
}

// auto tunes the staged walk; 0 keeps streaming and 1 forces the CPU jobs.
inline int cpu_share_staged_3k_mode() {
    static const int mode = [] {
        const char* v = strata::env_get("PREFILL_STAGED_3K");
        if (!v || std::strcmp(v, "auto") == 0) return -1;
        return *v == '1' ? 1 : 0;
    }();
    return mode;
}
inline bool cpu_share_batched(int64_t rows, bool qualified = false) {
    const int mode = cpu_share_staged_3k_mode();
    return (mode == 1 || (mode < 0 && qualified)) && rows >= 2048 && rows <= 3072;
}

// Compare adjacent layers after warm-up. Require two ratios to beat the unshared layer by 10%, so a
// near-tie keeps the earlier path. One instance belongs to one model/device.
struct PromptShareChoice {
    int attempts = 0, samples = 0, ratios = 0, last_arm = -1;
    int64_t last_layer = -1;
    double last_cost = 0, values[2] = {};
    bool selected = false;
    bool decided() const { return ratios >= 2; }
    bool next() { return decided() ? selected : (attempts++ & 1) != 0; }
    void add(bool arm, int64_t layer, int64_t missed, double ms, const char* who) {
        if (decided() || missed <= 0 || ms <= 0) return;
        const double cost = ms / (double) missed;
        if (++samples > 1 && last_arm >= 0 && last_arm != (int) arm && layer > last_layer) {
            values[ratios++] = arm ? cost / last_cost : last_cost / cost;
            if (decided()) {
                const double slower = std::max(values[0], values[1]);
                selected = slower < 0.90;
                std::fprintf(stderr, "xestrata: CPU prompt share auto (%s): slower shared/unshared %.3f, %s\n",
                             who, slower, selected ? "share" : "do not share");
            }
        }
        last_arm = (int) arm; last_layer = layer; last_cost = cost;
    }
};

/// A prompt chunk's layer: missed experts and those the CPU can read, fewest rows first. max_rows bounds an
/// expert's rows; the default is one multi-token job. The blob must stay at its address during the jobs.
struct CpuShareCandidates {
    int64_t missed = 0;
    std::vector<std::pair<int32_t, int32_t>> cand;
};
template <class InVram, class Skip>
CpuShareCandidates cpu_share_candidates(const int32_t* counts, int64_t n_expert, int64_t layer, const ExpertSource& src,
                                        InVram in_vram, Skip skip, int64_t max_rows = kernels::cpu::MAXT) {
    CpuShareCandidates r;
    for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
        const int32_t c = counts[e];
        if (c == 0 || in_vram(e) || skip(e)) continue;
        ++r.missed;
        if (c <= max_rows && (src.pinned(layer, e) || !src.transient(layer, e))) r.cand.emplace_back(c, e);
    }
    std::sort(r.cand.begin(), r.cand.end());
    return r;
}

/// Of `cand` (cpu_share_candidates, fewest rows first), the first `want` the CPU can take: `blob(e)` non-null (the
/// expert's bytes where the CPU reads them), their rows `row_cap` at most together; `take(e, rows, blob)` for each.
/// The experts taken.
template <class Blob, class Take>
int64_t cpu_share_take(const std::vector<std::pair<int32_t, int32_t>>& cand, int64_t want, int64_t row_cap, Blob blob,
                       Take take) {
    int64_t n = 0, rows = 0;
    for (const auto& [c, e] : cand) {
        if (n >= want || rows + c > row_cap) break;
        const uint8_t* b = blob(e);
        if (b == nullptr) continue;   // an expert without a blob stays on the GPU
        take(e, c, b);
        rows += c;
        ++n;
    }
    return n;
}

// The resident RAM mode and the adaptive tier.  A swap copies `in` (held in RAM) into the slot of `out` (held only
// by that slot).  Before the slot is overwritten, `out`'s bytes are copied back from it into an exchange buffer, so
// the CPU computes `out` from RAM while the swap is in flight; when the swap has landed, `commit_exchanges` moves
// them into `in`'s place in RAM.  The RAM copy then again holds exactly the experts no core slot does, and no swap
// reads the file.  Swaps that need no exchange (`out` in the lend region is held in RAM already; or `in` is not; or
// a free slot) go on as before; ones beyond the buffers' room wait for a later round.  Runs on the adaptive tier's
// thread while the GPU commits and drafts: the copies back are on its stream, and waited for before the refills are
// queued.
//
// With a layer split, `host_res` holds slot numbers of WHICHEVER cache owns the layer, so the copy back reads the
// owning card's cache on that card's stream: `locate(layer)` names them (upstream fe9c10ca: reading a later stage's
// slot number out of the first card's cache put another expert's bytes in RAM, and the answers turned to garbage).
struct SwapHome {
    ExpertCache* cache;
    gpu::Stream stream;
    int dev;   ///< -1: the first card
};

template <class Swap, class Locate>
bool resident_stage_swaps(FileExpertSource& src, const std::vector<int32_t>& host_res, int64_t n_expert,
                          std::vector<Swap>& swaps, Locate locate) {
    const auto& layout = kernels::cpu::expert_layout();
    if (layout.mixed()) {
        swaps.erase(std::remove_if(swaps.begin(), swaps.end(), [&](const Swap& s) {
            if (s.out < 0) return false;
            const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
            if (slot < 0) return true;
            const auto* cache = locate(s.layer).cache;
            return layout.blob_bytes(s.layer, s.in) > cache->slot_capacity(slot);
        }), swaps.end());
    }
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Staged { int32_t layer, in, out; int64_t q; };
    std::vector<Staged> staged;
    std::vector<SwapHome> used;   // the streams the copies back went on
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (s.out < 0 || !src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) {
            kept.push_back(s);
            continue;
        }
        const int64_t q = (int64_t) staged.size();
        if (q >= src.exchange_capacity()) continue;
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        const SwapHome home = locate(s.layer);
        const OnDevice on(home.dev);
        if (gpu::copy_async(src.exchange_buffer(q, s.layer), home.cache->device_slot(slot),
                            (size_t) layout.blob_bytes(s.layer, s.out), home.stream) != true)
            return false;
        if (std::find_if(used.begin(), used.end(), [&](const SwapHome& h) { return h.stream == home.stream; }) == used.end())
            used.push_back(home);
        staged.push_back({s.layer, s.in, s.out, q});
        kept.push_back(s);
    }
    if (!staged.empty()) {
        for (const SwapHome& h : used) {
            const OnDevice on(h.dev);
            if (gpu::stream_sync(h.stream) != true) return false;
        }
        for (const Staged& x : staged) {
            if (src.stage_exchange(x.layer, x.in, x.out, x.q)) continue;
            if (!layout.mixed()) return false;
            kept.erase(std::remove_if(kept.begin(), kept.end(), [&](const Swap& s) {
                return s.layer == x.layer && s.in == x.in && s.out == x.out;
            }), kept.end());
        }
    }
    swaps.swap(kept);
    return true;
}

}  // namespace strata::core
