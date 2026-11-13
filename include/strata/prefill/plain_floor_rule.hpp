// include/strata/prefill/plain_floor_rule.hpp - whether the CPU share's chunk limit may raise the long-prompt floor (prefill.cpp stream_all_min()).
//
// With the share armed (up to 3,072 tokens by default on a CUDA card) stream_all_min() is the share's limit instead of the default floor, because the share applies to staged chunks
// only.  The share takes experts of a NATIVE pack only (prefill.cpp cpu_maybe needs lay.native); a build without llama.cpp's MMQ that loads a pack which is not a native one (the plain
// Q2_0 pack setup builds on a CPU with AVX-512) has nothing for the share to take, and its default floor is the lowest one (stream_floor_rule.hpp, kFusedOnlyFloor) because the staged
// walk has neither the staged fused entry nor the int8 kernels for that layout and runs each routed expert on the dequantize + FP16 path.  The share's limit raised the floor above that
// anyway: a chunk of 64 to 3,072 tokens (the tail of a long prompt, 1,567 tokens at 20,000) took the FP16 path for nothing.  There the floor stays the default floor.
//
// STRATA_PF_PLAIN_FLOOR=0 is the way back (the share's limit raises the floor as before); anything else, or unset: the rule.  A build with MMQ passes plain_pack_fused == false and is
// unchanged.  Pure function: no GPU, no environment read here (the caller passes getenv's answer).
#pragma once

namespace strata::prefill {

/// STRATA_PF_PLAIN_FLOOR asked for off: only "0".
inline bool plain_floor_off(const char* env) { return env != nullptr && env[0] == '0' && env[1] == '\0'; }

/// Does the CPU share's limit raise the chunk floor?  `plain_pack_fused`: a build without MMQ and a non-native pack on the fused kernels (prefill.cpp plain_pack_all_fused()).
inline bool share_limit_raises_floor(bool plain_pack_fused, const char* env) { return !plain_pack_fused || plain_floor_off(env); }

}  // namespace strata::prefill
