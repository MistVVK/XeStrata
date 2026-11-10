// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/i8_gemm.hpp - the int8 products whose sums leave scaled (src/kernels/xe/i8_gemm.cpp, on NVIDIA
// GPUs CUTLASS's kernel in the contrib-llvm mode: src/kernels/cuda/i8_gemm_cutlass.cu): Y =
// (X . W^T) times X's row scales and W's (a W8A8 product's), plus a bias, written as BF16 - what the products' readers
// take, where a library's int8 product writes int32 sums for them to scale (prefill::Gemm::i8): half the bytes read
// after it.  BF16, not FP16: some models' products leave values past FP16's range (Mage-Flow's to 4e8), and the
// reference implementations keep these activations in BF16.  The GPU makers' libraries take no such product
// (cuBLASLt and hipBLASLt write int8 products' sums as int32 or int8 only, 2026-11-06).
// RTX 4070, 2026-11-10: Qwen-Image-2.1-Turbo sampling took 3.43 s with W8A8 and 6.85 s with the removed Q8_0
// row kernel (1024 x 1024, 8 steps), so the dense int8 path uses W8A8 instead.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Whether `stream`'s GPU has a kernel for it (NVIDIA's from sm_80 on, where the build carries CUTLASS's).
bool i8_gemm_ok(void* stream);

/// Y [T][N] (BF16, row stride ldy, 0: N) = (X [T][K] . W [N][K]^T) * sx[t] * sw[n] + bias[n] (null: none), X and W
/// int8 row-major; false (nothing written) where no kernel takes the product (i8_gemm_ok, or its shape: CUTLASS's
/// takes at least 1024 rows, K a multiple of 16 and N and ldy of 8), for the caller's library product.
bool i8_gemm_bf16(const int8_t* X, const float* sx, const int8_t* W, const float* sw, const float* bias, uint16_t* Y,
                 int64_t T, int64_t N, int64_t K, int64_t ldy, void* stream);

/// Y [T][N] (int32 sums, row stride ldy, 0: N) = X [T][K] . W [N][K]^T through oneDNN, where its GPU engine takes
/// `stream`'s GPU (Intel's): the int8 product prefill::Gemm::i8 takes where oneMath has none (the free mode); false
/// (nothing written) where oneDNN does not take it.
bool i8_gemm_s32(const int8_t* X, const int8_t* W, int32_t* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                 void* stream);
/// Whether i8_gemm_s32 takes `stream`'s GPU.
bool i8_gemm_s32_ok(void* stream);

}  // namespace strata::kernels
