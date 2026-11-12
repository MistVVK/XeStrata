#pragma once
// Few-row experts on the FP16 prompt path (lane P4-FALLBACK, 11-11).
//
// The FP16 path (no MMQ) dequantizes every routed expert of a layer to FP16 before its two GEMMs.  The dequantize costs
// about the same for an expert with 6 rows as for one with 600, so a short piece (a few hundred tokens routes only a handful
// of rows to most experts) pays it for nothing.  An expert with 1..kFewRows rows instead runs on the quantized grouped
// kernels the verify window already uses (strata::kernels::native_expert_grouped, reading the GGUF blocks and q8_1
// activations), with no dequantize.  Measured on a Tesla P100 (IQ3_XXS, plan phaseA-tests/par/RESULTS-P4-FALLBACK.md):
// 0.846 -> 0.954 of MMQ's prompt speed at 300 tokens, 0.868 -> 0.911 at 1,000, 0.909 -> 0.931 at 2K, 0.954 -> 0.997 at 4K,
// and above plain FP16 at 8K and 20K (1.044 / 1.035 of MMQ).
//
// The switch STRATA_PF_FEWROWS is a research switch: unset = `dflt` (off in this build's code), "0" off, anything else on.
#include <cstddef>
#include <cstdint>

namespace strata::prefill {

/// An expert with 1..kFewRows rows takes the quantized kernels.  Screened at 6, 12, 16, 24 and 32 on the P100: 12 was best.
constexpr int kFewRows = 12;
/// One-expert plan record (the grouped kernels' arguments, written by a one-block kernel so no host buffer is involved):
/// {u64 blob, i32 start[2] = {0, ne}, i32 n_groups = 1, i32 pad, i32 dst[ne] = o0 + i}.
constexpr size_t kFewRowsRecBytes = 24 + 4 * (size_t) kFewRows;

/// The switch's value: `v` is the variable's text (nullptr when unset); `dflt` is the answer when it is unset or empty.
inline bool fewrows_switch(const char* v, bool dflt) {
    if (v == nullptr || v[0] == '\0') return dflt;
    return v[0] != '0';
}

/// Whether the expert with `rows` rows (its rows of this layer's chunk) runs on the quantized kernels instead of the
/// dequantize + FP16 GEMMs.  `on`: the switch; `fp16_path`: the layer takes the FP16 path (not MMQ, not the fused kernels);
/// `native_pack`: the layer's experts are GGUF blocks in a native pack (a Strata pack's blob has another layout and no
/// grouped-kernel layout here); `apr`: the pack carries APR (Hadamard-folded) gate/up or down weights, which the FP16 path
/// rotates itself (hadamard_rows_f16) and this route does not.
inline bool fewrows_takes(bool on, bool fp16_path, bool native_pack, bool apr, int64_t rows) {
    return on && fp16_path && native_pack && !apr && rows >= 1 && rows <= (int64_t) kFewRows;
}

}  // namespace strata::prefill
