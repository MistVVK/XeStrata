// include/strata/prefill/dense_fp4_tc.hpp - Clean-room MIT Blackwell (sm_120a / sm_121a) mxf4nvf4 dense GEMM.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include <cstdint>

namespace strata::prefill::dense_fp4_tc {

/// True when compiled with sm_120a / sm_121a mxf4nvf4 support.
bool built();

/// True when built() and active CUDA device is sm_120 / sm_121 with runnable mxf4nvf4 kernels.
bool available();

/// True when enabled (default on for sm_120 / sm_121; STRATA_PF_DENSE_FP4=0 disables) and available().
bool enabled();

/// Dense GEMM: Y[T, N] = beta * Y + X[T, K] @ W[N, K]^T where X is FP16 row-major and W is ggml block_nvfp4 (type 40).
/// Returns true if executed on hardware mxf4nvf4 tensor cores, false to fall back to FP16 dequant + cuBLAS.
bool gemm_f16_nvfp4(const uint16_t* X_f16, const void* W_nvfp4, float* Y, int64_t T, int64_t N, int64_t K,
                    int64_t ldy, float beta, void* stream);

}  // namespace strata::prefill::dense_fp4_tc
