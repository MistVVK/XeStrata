// include/strata/prefill/nvfp4_prefill_mode.hpp - resolution of STRATA_PREFILL_NVFP4 across Strata's own prompt kernels.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Since 0.2.16 llama.cpp's MMQ prompt kernels are absent from the tree (mmq::built() == false).  Earlier, setting
// STRATA_PREFILL_NVFP4=w4a4 (including sm_121's former default in nv_defaults.cpp) made moe_fused_iq.cu decline the
// NVFP4 / NVFP4 pair so that MMQ's w4a4 tile could take it; without MMQ, declining the pair silently dropped the prompt
// onto FP16 dequantization + cuBLAS.  This helper resolves STRATA_PREFILL_NVFP4 cleanly:
//   - unset, empty, or "w4a8" -> FusedW4A8 (int8 tensor cores, Q8_1 activations);
//   - "w4a4" -> FusedW4A4 when Strata's native Blackwell mxf4nvf4 kernels (fp4_tc) are available on the active card,
//               LegacyMmqOrF16 if legacy MMQ is built, or FusedW4A8 (with *degraded_to_w4a8 = true) when neither W4A4
//               engine exists so the prompt stays on fused tensor cores instead of falling back to FP16 cuBLAS;
//   - "fp16" (or any other explicit value) -> LegacyMmqOrF16 (the exact FP16 reference path).
#pragma once

#include <cstring>

namespace strata::prefill {

enum class Nvfp4PrefillPath {
    FusedW4A8,
    FusedW4A4,
    LegacyMmqOrF16,
};

inline Nvfp4PrefillPath resolve_nvfp4_prefill_path(const char* env, bool fp4_tc_available, bool mmq_built,
                                                   bool* degraded_to_w4a8 = nullptr) {
    if (degraded_to_w4a8 != nullptr) *degraded_to_w4a8 = false;
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "w4a8") == 0) {
        return Nvfp4PrefillPath::FusedW4A8;
    }
    if (std::strcmp(env, "w4a4") == 0) {
        if (fp4_tc_available) return Nvfp4PrefillPath::FusedW4A4;
        if (mmq_built) return Nvfp4PrefillPath::LegacyMmqOrF16;
        if (degraded_to_w4a8 != nullptr) *degraded_to_w4a8 = true;
        return Nvfp4PrefillPath::FusedW4A8;
    }
    return Nvfp4PrefillPath::LegacyMmqOrF16;
}

inline bool nvfp4_fused_covers(const char* env, bool fp4_tc_available, bool mmq_built,
                               bool* degraded_to_w4a8 = nullptr) {
    const Nvfp4PrefillPath p = resolve_nvfp4_prefill_path(env, fp4_tc_available, mmq_built, degraded_to_w4a8);
    return p == Nvfp4PrefillPath::FusedW4A8 || p == Nvfp4PrefillPath::FusedW4A4;
}

}  // namespace strata::prefill
