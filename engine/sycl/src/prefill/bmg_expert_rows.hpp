#pragma once
#include "strata/sycl_queue.hpp"
#include "../bmg_prefill_options.hpp"
#include <cstdint>
#include <cstdlib>
#include <sycl/sycl.hpp>

namespace strata::prefill::bmg_rows {

// Offsetting each sorted row by its expert's prefix leaves the upper half
// of every expert interval available for FP32 gate/up scratch.
template<bool InDown>
inline void copy_expert_rows(sycl::queue& queue, const uint16_t* input, const int32_t* slot, uint16_t* output,
                             int64_t tokens, const int32_t* ids, const int32_t* starts) {
    using Block = sycl::vec<uint32_t, 8>;
    constexpr int blocks_per_row = 2560 / 16;
    constexpr int lanes = 64;
    const auto* source = reinterpret_cast<const Block*>(input);
    auto* destination = reinterpret_cast<Block*>(output);
    queue.parallel_for(sycl::nd_range<1>{size_t(tokens) * lanes, lanes},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            const int token = item.get_group(0);
            int slots[10];
#pragma unroll
            for (int i = 0; i < 10; ++i) {
                const int j = token * 10 + i;
                slots[i] = slot[j];
                if constexpr (InDown) slots[i] += starts[ids[j]];
            }
#pragma unroll
            for (int column = item.get_local_id(0); column < blocks_per_row; column += lanes) {
                const Block value = source[token * blocks_per_row + column];
#pragma unroll
                for (int i = 0; i < 10; ++i) destination[slots[i] * blocks_per_row + column] = value;
            }
        });
}

// slot[token * 10 + expert] names a distinct row of the routed input matrix.
// Load each segment of a token once and copy the same bits to its ten slots.
inline bool scatter_expert_rows(sycl::queue& queue, const uint16_t* input, const int32_t* slot,
                                uint16_t* output, int64_t tokens, int64_t experts, int64_t width) {
    const bool enabled = strata::bmg::experts_enabled();
    if (!enabled || tokens < 1 || tokens > 8192 || experts != 10 || width != 2560 ||
        ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) & 31u) ||
        queue.get_backend() != sycl::backend::ext_oneapi_level_zero) return false;
    const auto device = queue.get_device();
    if (strata::intel_gpu_gen(device) != strata::IntelGpuGen::Battlemage) return false;

    copy_expert_rows<false>(queue, input, slot, output, tokens, nullptr, nullptr);
    return true;
}
}  // namespace strata::prefill::bmg_rows
