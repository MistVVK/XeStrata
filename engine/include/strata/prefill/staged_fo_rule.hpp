// include/strata/prefill/staged_fo_rule.hpp - whether the staged walk's batched group entry (prefill.cpp fz_staged: fused::group_bounds + experts_native on the routed
// experts, each where it is) may also take a layer that has NO MMQ tile for its formats but that the fused kernels cover (a "fused-only" layer: UD-Q4_K_XL's Q4_K / Q5_K
// gate/up with a Q5_1 / Q8_0 down without STRATA_MMQ_KQ, and every layer of a build without llama.cpp's MMQ kernels, STRATA_MMQ=OFF).
//
// Without it such a layer runs each routed expert on the dequantize + FP16 GEMM path in every prompt chunk below the long-prompt floor (stream_all_min()): par/RESULTS-STRIX-SHORT.md,
// 3.6 to 4.4 times slower than the streamed walk on Strix Halo, and the cost every card meets once MMQ is compiled out.
//
// STRATA_PF_STAGED_FO: "1" asks for it on any card (research), "0" turns it off, unset takes the class default: on where it was measured (Windows gfx1151, a layer set with no MMQ tile
// that the fused kernels take: stream_floor_rule.hpp's class), off elsewhere.  Measured on Strix Halo (UD-Q4_K_XL, 96 to 1,000 token pieces): fastest of the old FP16 path, the long path and
// this entry at every size, bit-identical to the long path run with FP32 down rows (STRATA_PF_DM16=0).  Pure functions: no GPU, no environment read here (the caller passes getenv's answer).
#pragma once

namespace strata::prefill {

/// STRATA_PF_STAGED_FO asked for on: only "1".
inline bool staged_fo_switch(const char* env) { return env != nullptr && env[0] == '1' && env[1] == '\0'; }
/// STRATA_PF_STAGED_FO asked for off: only "0".
inline bool staged_fo_off(const char* env) { return env != nullptr && env[0] == '0' && env[1] == '\0'; }

/// The class default of the entry for fused-only layers.  `prereqs`: the entry's own prerequisites (the staged walk's resident-group gather and its group entry).  A build without
/// llama.cpp's MMQ takes the path a build with MMQ takes wherever that path does not need MMQ: the fused kernels' staged entry for every layer they cover, on every card class
/// (`own_only_build`).  A build with MMQ keeps the measured class only (`measured_class_all_fused`: Windows gfx1151 with every layer fused-only).
constexpr bool staged_fo_class_default(bool own_only_build, bool measured_class_all_fused, bool prereqs) { return prereqs && (own_only_build || measured_class_all_fused); }

/// The entry for fused-only layers on or off for this run: "1" on, "0" off, else the class default (`class_default`: the measured class AND the entry's own prerequisites hold).
inline bool staged_fo_decide(const char* env, bool class_default) {
    if (staged_fo_switch(env)) return true;
    if (staged_fo_off(env)) return false;
    return class_default;
}

/// `--short-read` (serve: pieces of at most N fresh tokens, compared without the last one, are read through the verify windows, longer ones batched), 64 by default.  With the entry on
/// the batched read of a short piece is cheaper than the windows from about 28 tokens (Strix Halo, UD-Q4_K_XL, par/RESULTS-STRIX-SHORT.md job s8: 24 tokens windows 231 ms against 261 ms
/// batched; 32 tokens 303 against 265-270; 48 tokens 453 against 293-299; 64 tokens 598 against 314-316), so on that class the default is the largest size where the windows won.
inline constexpr long long kStagedEntryShortRead = 24;

/// `base`: the engine's own default (64).  `class_measured` / `all_layers_fused_only`: stream_floor_rule.hpp's class; `entry_on`: staged_fo_decide().
/// `folded_trellis`: that class with an all-trellis set on the fused kernels (prefill.cpp trellis_g1151_all_fused(), Gyro-S with STRATA_PF_TQ_G1151 on) in which a layer is Hadamard-folded.  The
/// staged entry does not take a folded layer, so a batched piece below the floor ran it on the FP16 path (Strix Halo, Gyro-S, 26 / 31 / 33 / 49 tokens 705 / 766 / 778 / 993 ms against 293 / 391 /
/// 351 / 486 ms through the windows); such a set keeps `base`.
///
/// `sm120_cuda`: a CUDA card of compute capability 12.0 (RTX 5090 / 5080 / 5070 / PRO Blackwell; sm120_series.hpp, not a gfx12 HIP card, and STRATA_SM120_PERF=0 turns it off) AND every layer
/// fused-only (the caller's test, not the Strix class's).  With the entry on, a piece of 17 to 64 tokens reads in about 125-175 ms batched whatever its length, while the windows cost about 6.5 ms a token (RTX 5090, UD-Q4_K_XL, DFlash2, prompt
/// ms median of 3 rounds, `--short-read` 64 / 16 / 24 / 32 / 48: 24 tokens 172 / 130 / 175 / 174 / 172, 32 tokens 201 / 143 / 139 / 191 / 188, 48 tokens 284 / 166 / 170 / 159 / 268, 64 tokens 403 / 175 / 183 / 175 / 173;
/// 16 tokens level at 140 / 127 / 123 / 137 / 129; 84 and 512 tokens are batched in every arm and level), so the windows only win up to about 16 tokens: the default there is 16
/// (phaseA-tests/bw/RESULTS-BW-A.md BW-12).
inline constexpr long long kSm120ShortRead = 16;

constexpr long long short_read_default(long long base, bool class_measured, bool all_layers_fused_only, bool entry_on, bool folded_trellis = false, bool sm120_cuda = false) {
    return class_measured && all_layers_fused_only && entry_on && !folded_trellis ? kStagedEntryShortRead
         : sm120_cuda && entry_on ? kSm120ShortRead
         : base;
}

/// A layer the staged group entry may take on the layer-side rule: it has an MMQ tile (the entry's earlier rule), or it is fused-only and the entry is on (staged_fo_decide).
/// The rest of the entry's conditions (native pack, no CPU share, no peer, no Hadamard fold, ...) stay where they are in prefill.cpp.
constexpr bool staged_entry_layer(bool use_mmq, bool fused_only, bool fo_on) { return use_mmq || (fused_only && fo_on); }

}  // namespace strata::prefill
