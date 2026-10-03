// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/prefill/gemm.hpp - plan v0.3 P5: the batched projections of prompt processing.
//
// Every projection of a chunk of T tokens is Y[T, N] = X[T, K] . W[N, K]^T with W row-major (the GGUF / pack layout)
// and FP32 outputs.  Weights are either BF16 on the device already (the pack's BF16 tensors) or dequantized from their
// native GGUF blocks to FP16 into a reusable scratch right before the product; the activations are in the same type.
// The products run on the XMX engines (strata/kernels/xmx_gemm.hpp).
#pragma once

#include <cstdint>
#include <string>

namespace strata::prefill {

class Gemm {
public:
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    /// `scratch_elems`: 16-bit elements of the dequantization scratch (the largest weight dequantized at once).
    bool init(void* stream, int64_t scratch_elems, std::string& err);
    /// The same with a caller-owned device buffer (the prompt path borrowing expert-cache slots).
    void init_external(void* stream, uint16_t* scratch, int64_t scratch_elems);

    /// Y[T, N] (fp32, row stride ldy) = X[T, K] (bf16, row-major) . W[N, K]^T (bf16, row-major); `accumulate`: Y +=.
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              bool accumulate = false);

    /// Y = X . W^T with both in FP16 (bits).
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0);

    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16.
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0);

    /// Caller-owned buffer only: the scratch moved (the prompt path laid its buffers out again).
    void rebind(uint16_t* scratch, int64_t scratch_elems);

    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    void* stream() const { return stream_; }

private:
    void* stream_ = nullptr;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    bool external_ = false;
};

}  // namespace strata::prefill
