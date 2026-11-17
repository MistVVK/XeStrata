// metal/src/engine/stubs_df2.cpp - the DFlash2 tap kernels of the CUDA / HIP verifier and prefill (src/kernels/dflash2.hpp).
// Verifier::record_window and Prefill::run_impl call them, but on Metal neither runs: the decode window and the prompt chunks are
// metal/src/engine/window.cpp, which writes the taps itself (dflash2.inc).  They exist so the shared host files link.

#include "strata/kernels/dflash2.hpp"

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels {
void df2_tap_gr(const float*, int64_t, const float*, int64_t, const float*, int64_t, float*, int64_t, int, int64_t, int, void*) {
    strata::metal_engine::missing("strata::kernels::df2_tap_gr");
}
void df2_tap_rows(const float*, float*, int64_t, int64_t, int, int, void*) {
    strata::metal_engine::missing("strata::kernels::df2_tap_rows");
}
}  // namespace strata::kernels
