// include/strata/prefill/plain_floor_rule.hpp - the long-prompt floor of the plain pack by card class (prefill.cpp stream_all_min()).
//
// A chunk of the floor's size or more is read by the streamed walk: every non-resident expert of every layer comes through the ring, whatever the chunk's size, and the layers run on the fused
// kernels.  A shorter chunk takes the staged walk: only the routed experts are staged.  A build without llama.cpp's MMQ that loads a pack which is not a native one (the plain Q2_0 pack setup
// builds on a CPU with AVX-512) has neither the staged fused entry nor the int8 kernels for that layout, so the staged walk runs each routed expert there on the dequantize + FP16 path.
//
// Two things set the floor of that pack today.  stream_floor_rule.hpp gives it kFusedOnlyFloor (64) on every card, a value measured on an integrated GPU (Radeon 780M), where streaming costs
// nothing.  And the CPU share, when armed (on a CUDA card by default), replaces the floor by its own limit (3,072 tokens) although it takes no experts of a non-native pack (prefill.cpp cpu_maybe
// needs a native pack): that raised floor is what sent the 1,567-token tail of a 20,000-token prompt through the FP16 path, and it also hid what 64 does on a discrete card.
//
// Measured on an RTX 5070 12 GB on PCIe 4.0 x16 (15.6 % of the experts resident), plain pack, build without MMQ, against the build with MMQ: floor 64 streams every non-resident expert for
// every chunk, a flat cost of about 1.27 s, so a 256-token prompt takes 1,224 ms against 590 ms (2.07x) and a 1,000-token prompt 1,252 against 1,050 ms; the staged FP16 path grows by about
// 0.41 ms a token (1,182 ms at 1,200 tokens, 1,305 at 1,500, 1,452 at 2,048), so the two cross near 1,420 tokens and the streamed walk is faster from there (1,800 tokens: 0.93x of the build with MMQ,
// 3,000 tokens: 0.78x).  A floor of 1,024 loses 8 % at 1,200 tokens, 1,536 is level there and gains from 1,800 on.  On a DISCRETE CUDA card the plain pack therefore takes kPlainDiscreteFloor
// (1,536, a round number above the crossover, which stays closer to the old behaviour on a card with a slower link), with or without the share; an integrated card (unified memory),
// any card whose class is not known yet, and every other backend keep the old answer (the share's limit when armed, else the default floor).  Only that one discrete card was measured: every other
// discrete class is unmeasured.  STRATA_PF_PLAIN_FLOOR=0 is the way back (the old answer on every class).  A build with MMQ and the native packs pass plain_pack_fused == false and are unchanged.
// Pure functions: no GPU, no environment read here (the caller passes getenv's answer).
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::prefill {

/// The floor of the plain pack on a discrete CUDA card (tokens).  Measured on one card only (RTX 5070 on PCIe 4.0 x16); other discrete classes are unmeasured.
inline constexpr int64_t kPlainDiscreteFloor = 1536;

/// STRATA_PF_PLAIN_FLOOR asked for off: only "0".
inline bool plain_floor_off(const char* env) { return env != nullptr && env[0] == '0' && env[1] == '\0'; }

/// Does the plain pack take kPlainDiscreteFloor here?  `plain_pack_fused`: a build without MMQ and a non-native pack on the fused kernels (prefill.cpp plain_pack_all_fused()).
/// `discrete_cuda`: a CUDA card that answered as not integrated (a card not asked yet, an integrated one and any other backend pass false).
inline bool plain_takes_discrete_floor(bool plain_pack_fused, bool discrete_cuda, const char* env) {
    return plain_pack_fused && discrete_cuda && !plain_floor_off(env);
}

/// The floor stream_all_min() answers (no STRATA_PREFILL_STREAM_MIN): `share_limit` is the CPU share's limit when it is armed, else 0; `default_floor` the floor without it
/// (prefill.cpp default_stream_floor()).  The old answer: the share's limit when armed, else the default floor.
inline int64_t floor_with_share(bool takes_discrete_floor, int64_t share_limit, int64_t default_floor) {
    if (takes_discrete_floor) return std::max(default_floor, kPlainDiscreteFloor);
    return share_limit > 0 ? share_limit : default_floor;
}

}  // namespace strata::prefill
