#pragma once

#include "strata/sycl_queue.hpp"
#include <cstdint>

namespace strata::prefill {
inline bool use_bmg_gdn_tiles(const sycl::queue& queue, int64_t rows) {
    return rows >= 512 && queue.get_backend() == sycl::backend::ext_oneapi_level_zero &&
           strata::intel_gpu_gen(queue.get_device()) == strata::IntelGpuGen::Battlemage;
}

namespace strata_bf16_det {
void gemm(sycl::queue&, const uint16_t*, int64_t, const uint16_t*, float*, int64_t, int64_t, int64_t, int64_t, float);
void bmg_gdn_projection(sycl::queue&, const uint16_t*, int64_t, const uint16_t*, float*, int64_t, int64_t);
} // namespace strata_bf16_det
enum class GdnConvPath { Serial, SeparateL2, FusedL2, Battlemage };
GdnConvPath select_gdn_conv_path(const sycl::queue& queue, int64_t rows);
void gdn_conv_impl(float* history, const float* qkv, const float* weights, float* output, int64_t rows, float eps,
                   void* stream, GdnConvPath path);
} // namespace strata::prefill
