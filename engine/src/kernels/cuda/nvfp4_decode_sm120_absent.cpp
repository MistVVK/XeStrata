// src/kernels/cuda/nvfp4_decode_sm120_absent.cpp - Fallback stub when sm_120a/sm_121a is not in CUDA_ARCHITECTURES.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "strata/kernels/nvfp4_decode_sm120.hpp"

namespace strata::kernels {

void nvfp4_decode_sm120_set_override(int) noexcept {}

bool nvfp4_decode_sm120_available() noexcept { return false; }

bool nvfp4_decode_sm120_grouped(const NativeExpertLayout&, const unsigned long long*, const int32_t*,
                                const int32_t*, const int32_t*, const int32_t*, int64_t, int64_t,
                                const void*, void*, float*, void*) {
    return false;
}

}  // namespace strata::kernels
