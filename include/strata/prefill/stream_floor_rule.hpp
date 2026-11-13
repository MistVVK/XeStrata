// include/strata/prefill/stream_floor_rule.hpp - from which chunk size a prompt chunk takes the long-prompt path (src/prefill/prefill.cpp stream_all_min()).
//
// A chunk of the floor's size or more is read by the streamed walk: every non-resident expert of every layer comes through the ring in layer order and the
// layers' experts run on the fused kernels where those cover the layer.  A shorter chunk takes the staged walk: only the routed experts are staged, and a
// layer with no MMQ tile for its formats (UD-Q4_K_XL's Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down, without STRATA_MMQ_KQ) runs each routed expert on the dequantize-to-FP16
// path: a dequantize and two FP16 GEMMs per expert, a fixed cost of about 1.6 s a prompt on Windows gfx1151 whatever the length (128 tokens 1,754 ms, 1,000 tokens 3,629 ms,
// against 1,110 ms for 1,100 tokens on the streamed walk: par/RESULTS-STRIX-SHORT.md).  The streamed walk reads 128 to 1,000 tokens 3.6 to 4.4 times sooner there.
//
// The rule: the default floor is 1,024 (kStreamAllMinDefault in prefill.cpp, measured on a discrete card).  For a layer set in which EVERY layer has no MMQ tile and takes the
// fused kernels, on a class of device where that was measured (today: Windows gfx1151 only), the staged walk has a fused entry for such layers (staged_fo_rule.hpp, on by default on
// that class: pieces below 1,024 tokens 5x faster than the FP16 path, 1.03 to 1.23x faster than the long path from 96 to 1,000 tokens) and the floor stays 1,024; with the entry off
// (STRATA_PF_STAGED_FO=0) the floor is kFusedOnlyFloor, which sends pieces of 64 tokens and more to the streamed walk.  STRATA_PREFILL_STREAM_MIN, set to any value,
// replaces the floor (1024 restores the old rule); the CPU share's own limit (g_stream_min_share) keeps precedence where it applies.  Pure: no GPU, no environment read here.
#pragma once

#include <cstdint>

namespace strata::prefill {

inline constexpr int64_t kStreamFloorDefault = 1024;
inline constexpr int64_t kFusedOnlyFloor = 64;   // the streamed walk is faster than the FP16 path at every size measured from 96 up (par/RESULTS-STRIX-SHORT.md); the server reads pieces of up to --short-read tokens through the verify windows and longer ones batched

/// `class_measured`: the device class this default was measured on (Windows gfx1151).  `all_layers_fused_only`: the layer set has no MMQ tile and the fused kernels take every layer.
/// `staged_entry`: the staged walk takes such layers on its fused group entry (staged_fo_rule.hpp), so the old FP16 path below the floor is gone and the floor needs no lowering.
///
/// `plain_pack_fused`: a build without llama.cpp's MMQ (prefill.cpp STRATA_PF_OWN_ONLY) loading a pack that is not a native one (the AVX-512 Q2_0 pack: no native_experts.txt) whose layers the plan marks
/// fused (fused::enabled()).  The staged entry and the int8 kernels read the native layout only, so below the floor such a layer would run each routed expert on the dequantize + FP16 path
/// (Radeon 780M, Q2_0 pack: 367-token prompts 7,435 ms against 3,222 ms with MMQ); the floor is kFusedOnlyFloor there, on every card class.  A build with MMQ passes false.
constexpr int64_t stream_floor_default(bool class_measured, bool all_layers_fused_only, bool staged_entry, bool plain_pack_fused = false) {
    return (class_measured && all_layers_fused_only && !staged_entry) || plain_pack_fused ? kFusedOnlyFloor : kStreamFloorDefault;
}

}  // namespace strata::prefill
