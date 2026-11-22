// include/strata/kernels/dense_i8_tc.hpp - Clean-room MIT INT8 Tensor Core dense GEMM (sm_80+: RTX 3090 & RTX 5090).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Direct quantized INT8 Tensor Core GEMM (mma.sync.aligned.m16n8k32 / m16n8k16 .s32.s8.s8.s32) for GGUF dense
// projections (Q8_0=8, Q4_K=12, Q5_K=13, Q6_K=14, IQ4_XS=23):
//   1. Decode / speculative verify (`dense_i8_tc_mmvq`): consumes `block_q8_1` activations directly for T = 1..16
//      tokens in a single Tensor Core pass (no 4-col slicing, no scalar __dp4a loops), matching native_mmvq math.
//   2. Prompt prefill (`dense_i8_tc_prefill`): quantizes FP16 activations into scratch `block_q8_1` and streams
//      quantized weights directly into INT8 Tensor Cores without global FP16 weight dequantization round-trips.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// True when the current CUDA device supports INT8 Tensor Core dense GEMM (sm_80+, non-HIP).
bool dense_i8_tc_supported();

/// True when `ggml_type` is supported by dense_i8_tc (Q8_0=8, Q4_K=12, Q5_K=13, Q6_K=14, IQ4_XS=23).
bool dense_i8_tc_type_ok(int ggml_type) noexcept;

/// True when STRATA_DENSE_I8_TC is enabled for decode verify (default ON on sm_80+ for T >= 4; =0 disables).
bool dense_i8_tc_decode_enabled();

/// True when STRATA_PREFILL_DENSE_I8_TC is enabled for prompt prefill (default ON on sm_80+; =0 disables).
bool dense_i8_tc_prefill_enabled();

/// Decode / verify multi-token INT8 TC GEMM: Y[T, n_out] = X_q8_1[T, n_in/32] * W[n_out, n_in]^T.
/// `T` in [1..16], `n_in` multiple of 32 (Q8_0) or 256 (Q4_K/Q5_K/Q6_K/IQ4_XS).
/// Returns true if launched; false if caller should fall back to `native_mmvq`.
bool dense_i8_tc_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int T, void* stream);

/// Required scratch bytes for `dense_i8_tc_prefill` (holds quantized `block_q8_1` rows for `T` tokens of width `K`).
size_t dense_i8_tc_prefill_scratch_bytes(int64_t T, int64_t K);

/// Prompt prefill INT8 TC GEMM: Y[T, N] (stride `ldy`) = beta * Y + X_f16[T, K] * W_blocks[N, K]^T.
/// Uses `scratch` (at least `dense_i8_tc_prefill_scratch_bytes(T, K)` bytes, e.g. Gemm's existing FP16 scratch).
/// Returns true if launched; false if caller should fall back to `dequant_f16 + cuBLAS`.
bool dense_i8_tc_prefill(const uint16_t* X_f16, int ggml_type, const void* W_blocks, float* Y,
                         int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                         void* scratch, size_t scratch_bytes, void* stream);

}  // namespace strata::kernels
