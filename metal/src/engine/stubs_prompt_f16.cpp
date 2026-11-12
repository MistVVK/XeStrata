// metal/src/engine/stubs_prompt_f16.cpp - two entry points of the CUDA / HIP prompt path that the Metal engine does not implement:
// the F16 request of the fused expert kernels and the few-row expert plan record.  Prefill::run_impl (src/prefill/prefill.cpp) calls
// them but does not run on Metal (the prompt chunks are metal/src/engine/window.cpp, see stubs_df2.cpp); they exist so the shared host
// files link.
//   fused::want_dm_f16   asks the fused kernels to write the down rows as FP16 (gfx11): this build has no fused kernels, nothing to ask.
//   fewrows_record       STRATA_PF_FEWROWS (research switch, off): the plan record of a few-row expert, written by a CUDA kernel;
//                        stops the engine with its name, like stubs.cpp, if it is ever reached.

#include <cstdint>

#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_fused_iq.hpp"

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::prefill {
void fewrows_record(void*, const void*, int, int, void*) {
    strata::metal_engine::missing("strata::prefill::fewrows_record");
}
namespace fused {
void want_dm_f16(bool) {}
}  // namespace fused
}  // namespace strata::prefill
