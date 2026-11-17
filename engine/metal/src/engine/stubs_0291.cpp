// metal/src/engine/stubs_0291.cpp - the 0.2.9 entry points the Metal engine does not implement: the QSA prompt-layer fusions of
// f-042-strix-pf (STRATA_QSA_FUSE_NR, STRATA_QSA_FUSE_GATE).  The gated attention answers "not taken" (its documented contract:
// nothing is launched on false, the caller runs the two-step path); the fused norm + rope stops the engine with its name like the
// other QSA prompt kernels (stubs.cpp) - no Metal model has a QSA prompt layer.

#include <cstdint>

#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/prefill/kernels.hpp"

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels {
bool qsa_prompt_attn_batch_gated(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&,
                                 const float*, uint16_t*, int64_t, int64_t, void*) {
    return false;
}
}  // namespace strata::kernels

namespace strata::prefill {
void rms_rope_rows(const float*, int64_t, int64_t, float*, int64_t, const float*, int64_t, int64_t, int64_t, float, int64_t,
                   const strata::kernels::RopeScaling&, void*) {
    strata::metal_engine::missing("strata::prefill::rms_rope_rows");
}
}  // namespace strata::prefill
