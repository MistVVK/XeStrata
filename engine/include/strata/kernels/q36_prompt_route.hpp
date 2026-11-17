// include/strata/kernels/q36_prompt_route.hpp - which kernels carry the prompt experts of one layer of the Qwen3.6 engine (strata-q36, src/qwen35/q36_engine.cu).
//
// The order, first that applies:
//   1. the fused kernels (`fused_takes`: the layer's pair is covered on this card by the engine's own fused rule),
//   2. our int8 expert kernels (native_expert_i8) when STRATA_EXPERT_I8 is set to 1 and they cover the layer (a set switch wins over MMQ, as in the Qwen3.8 engine),
//   3. llama.cpp's MMQ (a build that has it, a layer it has a tile for),
//   4. our int8 expert kernels when the switch is not 0 and they cover the layer (this is where a build without MMQ lands, and a layer MMQ has no tile for),
//   5. the native grouped kernels (the way back: STRATA_EXPERT_I8=0 in a build without MMQ, or a layer no faster kernel covers).
// An unset STRATA_EXPERT_I8 reads as 1 in a build without MMQ (i8_switch_effective), so rule 2 is the default there; in a build with MMQ an unset switch leaves rule 3 first.
// Header-only, no GPU: tests/prefill/q36_prompt_route_test.cpp.
#pragma once

#include <cstdint>

#include "strata/kernels/i8_expert_rule.hpp"

namespace strata::kernels {

enum class Q36Route { Fused, Int8, Mmq, Grouped };

/// `raw_i8_state` = i8_switch_parse(getenv("STRATA_EXPERT_I8")); `mmq_free` = the build has no MMQ; `mmq_takes` = the build has MMQ, it has a tile for the layer's pair and
/// STRATA_Q36_NO_MMQ is not set; `i8_covers` = native_expert_i8_supported(the layer's layout) (formats, sizes, the device's shared memory).
inline Q36Route q36_layer_route(bool fused_takes, int raw_i8_state, bool mmq_free, bool mmq_takes, bool i8_covers) {
    if (fused_takes) return Q36Route::Fused;
    const int st = i8_switch_effective(raw_i8_state, mmq_free);
    if (st == 1 && i8_covers) return Q36Route::Int8;
    if (mmq_takes) return Q36Route::Mmq;
    if (st != 0 && i8_covers) return Q36Route::Int8;
    return Q36Route::Grouped;
}

/// Can the whole prompt path (the streamed layers of the hybrid layout, the big chunks) run: every layer has a route faster than the grouped kernels.
/// One layer on the grouped kernels sends every prompt chunk to them (the engine has one prompt path per model, not per layer).
inline bool q36_prompt_path_ok(int n_layers, int n_grouped) { return n_layers > 0 && n_grouped == 0; }

/// A prompt chunk of `rows` tokens (T * top_k routed entries over `experts` experts) on an int8 layer: the few-rows route (i8_fewrows_takes) hands a chunk whose experts average
/// fewer rows than the crossing to the grouped kernels, which do not decode a tile for so few rows.  `limit` = native_expert_i8_few_rows_limit() (0: no route).
inline bool q36_i8_chunk_takes(int64_t entries, int64_t experts, int limit) { return !i8_fewrows_takes(limit, entries, experts); }

}  // namespace strata::kernels
