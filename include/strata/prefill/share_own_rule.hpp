// include/strata/prefill/share_own_rule.hpp - whether a prompt layer on which the CPU share took experts (rows_cpu > 0) may still run the GPU's own experts on Strata's own kernels:
// the staged fused entry (prefill.cpp fz_staged) and the int8 kernels (i8_l).
//
// Both once required rows_cpu == 0, because they were written for a layer whose rows are all the GPU's.  They do not need it: the share carves the CPU experts' rows out as the LAST
// block of the layer's rows (m.off, `rows_local` already net of them, `order` without them), and the entry's and the kernels' row counts, bounds, scratch and destinations all come from
// `order` / `off` / `cnt`, never from T * K (T * K only bounds the tile count of the staged entry).  The CPU's rows are written into Dm after the walk and never by the GPU.
//
// In a build WITH llama.cpp's MMQ the layer runs MMQ's groups, which take the share as they always did, so that build keeps the old rule (a shared layer leaves both).  In a build WITHOUT MMQ
// (-DSTRATA_MMQ=OFF, or HIP without STRATA_PREFILL_MMQ; prefill.cpp's STRATA_PF_OWN_ONLY) a shared layer left both and ran every GPU expert of it on the FP16 path: 12 - 18 % slower than the
// build with MMQ at 2,049 tokens on an RTX A4000 (the share's default limit is 3,072 tokens).  There they stay on.
//
// STRATA_PF_SHARE_OWN: "0" keeps the old rule in every build (the way back), "1" applies the new rule in a build with MMQ too (research A/B); unset: by build.  Pure functions: no GPU, no
// environment read here (the caller passes getenv's answer).
#pragma once

#include <cstdint>

namespace strata::prefill {

/// STRATA_PF_SHARE_OWN asked for on: only "1".
inline bool share_own_switch(const char* env) { return env != nullptr && env[0] == '1' && env[1] == '\0'; }
/// STRATA_PF_SHARE_OWN asked for off: only "0".
inline bool share_own_off(const char* env) { return env != nullptr && env[0] == '0' && env[1] == '\0'; }

/// May a layer with `rows_cpu` rows on the CPU still use the staged fused entry / the int8 kernels?  Always when the CPU took none.
inline bool share_layer_keeps_own_kernels(bool own_only_build, const char* env, int64_t rows_cpu) {
    if (rows_cpu == 0) return true;
    if (share_own_off(env)) return false;
    if (share_own_switch(env)) return true;
    return own_only_build;
}

}  // namespace strata::prefill
