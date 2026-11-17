// The default of --lookup-max-t (the owner's #4040 decision): 12 where the pr-nv2 runs measured it, 0 everywhere else.
// Header-only so a test can pin every condition.  Metal holds 8 rows in its windows (metal/src/engine/window.cpp MAXT) and has
// no 12-row verify window: "metal window: window size out of range" ended the first Qwen serve request there, so a Metal
// build keeps the option off.  (SYCL's generate.cpp has no such default at all.)
#pragma once

namespace strata::core {

struct LookupDefaultIn {
    bool env_off = false;         ///< STRATA_LOOKUP_MAX_T=0
    bool narrow_window = false;   ///< kVerifyMaxT == kVerifyNarrowT (the 8-row build)
    bool metal_build = false;     ///< STRATA_GPU_METAL
    bool serve = false;
    bool multi_gpu = false;
    bool layer_split = false;
    bool batch = false;           ///< --batch above 1
    bool dflash2 = false;
    bool lookup_chain = false;
    bool suffix_draft = false;    ///< --suffix-draft above 0
    bool spec_at_least_2 = false;
};

inline int default_lookup_max_t(const LookupDefaultIn& in) {
    return !in.env_off && in.narrow_window && !in.metal_build && in.serve && !in.multi_gpu && !in.layer_split && !in.batch &&
                   !in.dflash2 && !in.lookup_chain && in.suffix_draft && in.spec_at_least_2
               ? 12
               : 0;
}

}  // namespace strata::core
