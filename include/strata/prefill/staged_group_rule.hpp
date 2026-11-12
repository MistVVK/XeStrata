// include/strata/prefill/staged_group_rule.hpp - whether the staged walk (prompt chunks below stream_all_min(), 1,024 tokens) runs a layer's MMQ groups through
// the batched group entry (fused::group_bounds + experts_native, prefill.cpp fz_staged) in place of MMQ's per-row quantize / gather / gate-up / SwiGLU / down.
//
// Owner, 2026-11-12 ("8 yes for nvidia"): on NVIDIA the entry is the default wherever the card's own fused rule takes the expert formats (fused::native_supported:
// the 80-SM sm_120 / sm_86 rule, STRATA_PF_FUSED=1 on any sm_80+ card); on AMD (a HIP build) it is not yet: a HIP build keeps MMQ's staged walk unless STRATA_PF_FUSED=1
// is named.  STRATA_PF_FUSED=0 turns it off in both builds (native_supported answers false then as well).  STRATA_PF_FUSED set to anything else non-empty reads as
// off, as moe_fused_iq.cu's native_supported reads it.  Pure function: no GPU, no environment read here (the caller passes getenv's answer).
#pragma once

namespace strata::prefill {

inline bool staged_group_entry_default(bool hip_build, const char* pf_fused) {
    if (pf_fused != nullptr && pf_fused[0] != '\0') return pf_fused[0] == '1';
    return !hip_build;
}

}  // namespace strata::prefill
