// include/strata/prefill/staged_fo_rule.hpp - whether the staged walk's batched group entry (prefill.cpp fz_staged: fused::group_bounds + experts_native on the routed
// experts, each where it is) may also take a layer that has NO MMQ tile for its formats but that the fused kernels cover (a "fused-only" layer: UD-Q4_K_XL's Q4_K / Q5_K
// gate/up with a Q5_1 / Q8_0 down without STRATA_MMQ_KQ, and every layer of a build without llama.cpp's MMQ kernels, STRATA_MMQ=OFF).
//
// Without it such a layer runs each routed expert on the dequantize + FP16 GEMM path in every prompt chunk below the long-prompt floor (stream_all_min()): par/RESULTS-STRIX-SHORT.md,
// 3.6 to 4.4 times slower than the streamed walk on Strix Halo, and the cost every card meets once MMQ is compiled out.
//
// RESEARCH SWITCH STRATA_PF_STAGED_FO=1, off unless set (register status research).  Pure function: no GPU, no environment read here (the caller passes getenv's answer).
#pragma once

namespace strata::prefill {

/// STRATA_PF_STAGED_FO: on only for "1".
inline bool staged_fo_switch(const char* env) { return env != nullptr && env[0] == '1' && env[1] == '\0'; }

/// A layer the staged group entry may take on the layer-side rule: it has an MMQ tile (the entry's earlier rule), or it is fused-only and the research switch is on.
/// The rest of the entry's conditions (native pack, no CPU share, no peer, no Hadamard fold, ...) stay where they are in prefill.cpp.
constexpr bool staged_entry_layer(bool use_mmq, bool fused_only, bool fo_switch) { return use_mmq || (fused_only && fo_switch); }

}  // namespace strata::prefill
