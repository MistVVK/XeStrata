// include/strata/prefill/switch_point_rule.hpp - the chunk size below which a prompt chunk takes the STAGED walk and from which it takes the STREAMED walk
// (src/prefill/prefill.cpp stream_all_min()), chosen by estimated cost from the card's share of non-resident experts instead of the fixed 1,024.
// Research switch STRATA_SWITCH_POINT=1, default off.  Pure: no GPU, no environment read here.
//
// The two walks (prefill.cpp):
//   streamed: every expert the card does not hold is copied through a ring in layer order, whatever the chunk routes; the copy runs through the attention
//             halves, so the chunk costs about its copy time: P(T) = c_p * X + a_p * T (X = non-resident experts, T = chunk tokens).
//   staged:   the routing is read back per layer and only the routed non-resident experts are staged; the copy cannot overlap the layer's own compute:
//             S(T) = s0 + c_s * Xs(T) + a_s * T, with Xs(T) = X * (1 - u(T)) the routed ones.  u(T) is the share of the non-resident experts NOT routed by T tokens.
// Fitted on an RTX A6000 (48 GB, 12 GB/s link) with Qwen3.8-Flash-Next UD-Q4_K_XL (111 GB set, 55% of the expert slots resident), K-quant fused kernels, CPU share off
// (par/RESULTS-SWITCH-POINT.md, par/switchpt/fit1.py): c_p = 0.2612 ms, a_p = 0.0566 ms/token (at X = 12,097), c_s = 0.2278, a_s = 0.2294, s0 = 202 ms,
// u(T) = (T / 391)^-0.702 (measured routed shares 51 / 70 / 80 / 88 % at 1,100 / 2,049 / 4,097 / 8,193 tokens; the uniform-routing formula says 100% at all of them).
// Staged wins while  advantage(T) = P(T) - S(T) = (c_p - c_s) * X - s0 + c_s * X * u(T) - (a_s - a_p) * T  is positive.
//
// The floor is ONE scalar for the card, decided before the first lend (the lent cache slots, ring_slots(T), fused_layout(T) and moe_bufs() all read stream_all_min()), so it
// depends on the card's counts and never on the chunk in hand.  X = 0 (every expert resident) gives exactly the default 1,024.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::prefill::switch_point {

inline constexpr int64_t kFloorDefault = 1024;   // = kStreamFloorDefault (stream_floor_rule.hpp): the floor of every card the rule does not move
inline constexpr int64_t kGrid = 256;            // the prompt path's chunk grid
inline constexpr int64_t kFloorMax = 8192;       // never past the auto chunk's usual top: above it the streamed walk's one pass is cheaper on every card measured

struct Constants {
    double c_p = 0.2612;       // ms per non-resident expert, streamed walk (its copy-bound floor)
    double c_s = 0.2278;       // ms per streamed expert, staged walk
    double a_p = 0.0566;       // ms per token, streamed walk
    double a_s = 0.2294;       // ms per token, staged walk
    double s0 = 201.75;        // ms, staged walk's fixed part
    double t0 = 391.0;         // u(T) = (T / t0)^-gamma, 1 below t0
    double gamma = 0.702;
    double x_inflate = 1.0;    // the streamed walk's lend: X as streamed / X as counted from the cache before the lend
};

/// Share of the non-resident experts no token of a chunk of T tokens routes to.
inline double uncovered(int64_t T, const Constants& k) {
    if (T <= 0) return 1.0;
    return std::min(1.0, std::pow((double) T / k.t0, -k.gamma));
}

/// Milliseconds the staged walk saves over the streamed walk for a chunk of T tokens on a card with X non-resident experts (negative: it costs).
inline double advantage_ms(int64_t T, double X, const Constants& k) {
    return (k.c_p - k.c_s) * X - k.s0 + k.c_s * X * uncovered(T, k) - (k.a_s - k.a_p) * (double) T;
}

/// The floor for a card holding `resident_slots` of `total_pairs` (layers x experts): staged below it, streamed from it.
/// total_pairs <= 0 (unknown) or nothing non-resident: the default.  A card that gains nothing at 1,024 keeps 1,024 (never lower).
inline int64_t floor_for(int64_t resident_slots, int64_t total_pairs, const Constants& k = Constants{}) {
    if (total_pairs <= 0 || resident_slots < 0 || resident_slots >= total_pairs) return kFloorDefault;
    const double X = (double) (total_pairs - resident_slots) * k.x_inflate;
    int64_t T = kFloorDefault;
    while (T < kFloorMax && advantage_ms(T, X, k) > 0.0) T += kGrid;   // advantage falls with T: the first T where it is not positive is the floor
    return T;
}

// ---- per-card scaling (SWITCH-POINT-2) ----
// The A6000 fit above does not transfer as it is (the A4000's crossing is about 1,930 tokens, the fit's 5,632), but its structure does: the copy terms scale with the cost of moving one expert
// (mean blob bytes / the link's GB/s) and the compute terms with 1 / the SM count (the A4000 streams at 0.63 ms a token against the A6000's 0.36: 84 / 48).  s0 and u(T) stay as fitted.
// Measured crossings of the staged walk against the streamed one with the CPU share off (RTX A6000 3.8K, RTX A4000 1.9K) are matched to within one 256-token step by the real crossing
// rounded DOWN to the grid.  The R9700 (HIP, 28.3 GB/s probe but 0.14 ms a streamed expert) is not matched (model 1.1K, measured 1.5K): CUDA builds only.
struct Card {
    double blob_mb = 3.19;     // mean expert blob of the cache (cache bytes / slots), MB: the A6000 fit's value
    double link_gbps = 12.2;   // the start-up host-to-device probe, GB/s
    int sm = 84;               // multiprocessor count
};
inline bool card_known(const Card& c) { return c.blob_mb > 0.0 && c.link_gbps > 0.0 && c.sm > 0; }
inline Constants constants_for(const Card& c) {
    Constants k;
    const double copy = (c.blob_mb / c.link_gbps) / (3.19 / 12.2);
    const double comp = 84.0 / (double) c.sm;
    k.c_p *= copy; k.c_s *= copy;
    k.a_p *= comp; k.a_s *= comp;
    return k;
}
/// The floor for this card with the CPU share off: the largest grid size at which the staged walk still wins (the first size where it does not, one step back), never below 1,024.
inline int64_t floor_for_card(int64_t resident_slots, int64_t total_pairs, const Card& c) {
    if (!card_known(c)) return kFloorDefault;
    const int64_t first = floor_for(resident_slots, total_pairs, constants_for(c));
    return std::max<int64_t>(kFloorDefault, first - kGrid);
}
/// With the share on, the staged walk's limit is the floor times this ratio. Measured share-on break-even / share-off crossing: RTX A4000 2.9K / 1.93K = 1.50, RTX A6000 6.19K / 4.18K = 1.48,
/// R9700 2.5K / 1.5K = 1.67 (not covered by this rule). Against this rule's own floors as the engine printed them (A6000 3,584, A4000 2,048) the same break-evens are 1.73 and 1.42, so no single ratio
/// fits both: 1.6 gives limits 5,632 (A6000, 558 under its break-even: up to 6% left on the table at 5,633) and 3,072 (A4000, 172 over: up to 3% lost at 2,900-3,072); both are far better than the
/// fixed 3,072 on the A6000 (par/RESULTS-SWITCH-POINT.md, 06:12 block).
inline constexpr double kShareRatio = 1.6;
/// The share's limit for a card whose floor moved: floor x kShareRatio on the grid; 0 (the share's own default stays) when the floor is the default.
inline int64_t share_limit_for(int64_t floor) {
    if (floor <= kFloorDefault) return 0;
    return std::min<int64_t>(2 * kFloorMax, ((int64_t) ((double) floor * kShareRatio) / kGrid) * kGrid);
}

}  // namespace strata::prefill::switch_point
