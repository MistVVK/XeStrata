// metal/src/engine/stubs_0219.cpp - 0.2.19 CUDA/HIP-only kernel entry points so shared host files link on Metal.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors
//
// `NativeHead::load` / `NativeHead::unload` (`native_head.cpp`), `Verifier` (`verify.cpp`), and
// `warm_gemm_libraries` (`prefill.cpp`) call these helpers; on Metal the decode window and prompt chunks run
// through `metal/src/engine/window.cpp`. Query/lifecycle helpers return false/nullptr/0 safely, and GPU-launch
// stubs trap via `missing(...)` like `stubs.cpp`.
#include <cstddef>
#include <cstdint>
#include <string>

#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/megakernel.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/vhead_bitplane.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_fp4_tc.hpp"

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels {

bool vhead_bitplane_supported() { return false; }
bool vhead_bitplane_enabled() { return false; }
BitplaneHeadHandle* vhead_bitplane_create(int, const void*, int, int) { return nullptr; }
void vhead_bitplane_destroy(BitplaneHeadHandle*) {}
size_t vhead_bitplane_bytes(const BitplaneHeadHandle*) { return 0; }
bool vhead_bitplane_argmax(BitplaneHeadHandle*, const void*, int, int32_t*, void*, uint32_t*, int) { return false; }

unsigned* mk_counters_alloc() { return nullptr; }
int mk_gdn_grid(int) { return 0; }
void mk_gdn_dense(const MkGdnArgs&, int, void*) { strata::metal_engine::missing("strata::kernels::mk_gdn_dense"); }
void q80_mmvq_mapped(const void*, const MkMat&, const void*, float*, int, int, int, void*) {
    strata::metal_engine::missing("strata::kernels::q80_mmvq_mapped");
}

void df2_iota_pos(int32_t*, int32_t, int, void*) {
    strata::metal_engine::missing("strata::kernels::df2_iota_pos");
}

void gather_rows(const uint8_t*, int64_t, const int32_t*, int64_t, uint8_t*, void*) {
    strata::metal_engine::missing("strata::kernels::gather_rows");
}

void kv_stream_resolve_groups(const KvResolveGroups&, const int32_t*, const int32_t*, int64_t,
                              const QsaShapes&, void*) {
    strata::metal_engine::missing("strata::kernels::kv_stream_resolve_groups");
}

bool qsa_block_scores_slot_groups(const QsaScoreGroups&, const float*, const int32_t*, int64_t,
                                  const QsaShapes&, float*, void*) {
    return false;
}

}  // namespace strata::kernels

namespace strata::prefill {
bool Gemm::init(void*, int64_t, std::string&) { return false; }
namespace fp4_tc {
void set_model_has_input_scale(bool) {}
bool model_has_input_scale() { return true; }
}  // namespace fp4_tc
}  // namespace strata::prefill
