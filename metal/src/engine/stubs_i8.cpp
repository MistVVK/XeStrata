// metal/src/engine/stubs_i8.cpp - the entry points of the CUDA / HIP int8 prompt-expert kernel (src/kernels/cuda/i8_expert.cuh) that the shared
// prompt path (Prefill::run_impl, src/prefill/prefill.cpp) calls.  Prefill::run_impl does not run on Metal (the prompt chunks are
// metal/src/engine/window.cpp, see stubs_df2.cpp); these exist so the shared host files link.
//   native_expert_i8_supported / native_expert_i8_prompt   false: this build has no such kernel
//   native_expert_i8_scratch_bytes                         the size formula of iq_kernels.cu (keep in step)
//   native_expert_i8                                       stops the engine with its name, like stubs.cpp, if it is ever reached

#include <cstddef>
#include <cstdint>

#include "strata/kernels/iq_kernels.hpp"

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels {
bool native_expert_i8_supported(const NativeExpertLayout&) noexcept { return false; }
bool native_expert_i8_prompt(const NativeExpertLayout&, int64_t) noexcept { return false; }
size_t native_expert_i8_scratch_bytes(int64_t cap_entries, int64_t cap_groups, int64_t n_ff) {
    const size_t hqb = ((size_t) cap_entries * (size_t) (n_ff / 32) * 36 + 255) & ~(size_t) 255;   // 36 bytes: sizeof(block_q8_1)
    return hqb + (size_t) (cap_groups + 1) * sizeof(int32_t);
}
void native_expert_i8(const NativeExpertLayout&, const unsigned long long*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int64_t, int64_t,
                      const void*, void*, float*, void*) {
    strata::metal_engine::missing("strata::kernels::native_expert_i8");
}
}  // namespace strata::kernels
