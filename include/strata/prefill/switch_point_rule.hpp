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

}  // namespace strata::prefill::switch_point
