#pragma once
#include "strata/kernels/iq_kernels.hpp"

namespace strata::kernels {
// Graph-compatible composition of the existing native expert kernels and blob copier.
// The caller provides an exclusively owned device workspace, ordered on stream.
// Intermediate batch metadata uses the otherwise unused v2 hidden-activation scratch.
// Unsupported layouts/modes retain native_expert_grouped.
void native_expert_grouped_staged(
    const NativeExpertLayout& layout, const unsigned long long* pointers,
    const int32_t* starts, const int32_t* groups, const int32_t* destinations,
    const int32_t* tokens, int64_t cap_groups, int64_t cap_entries,
    const void* x_q8_1, void* scratch, float* output, void* stream,
    void* workspace, size_t workspace_bytes, int max_batch_groups = 8,
    int copy_mode = 0);
}
