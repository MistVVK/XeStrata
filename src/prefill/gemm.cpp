// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/gemm.cpp - see include/strata/prefill/gemm.hpp.  The CUDA build ran these products through
// cublasGemmEx (FP32 compute), the first Xe port through oneMKL; both are replaced by XeStrata's own XMX kernels, so the
// engine needs no non-free library.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/gpu.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {

using strata::kernels::XmxType;

Gemm::~Gemm() {
    if (!external_ && scratch_) strata::gpu::free(scratch_);
}

void Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems) {
    stream_ = stream;
    external_ = true;
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    stream_ = stream;
    if (scratch_elems > 0 && !strata::gpu::alloc_device(&scratch_, (size_t) scratch_elems * 2)) {
        err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB";
        return false;
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                bool accumulate) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (strata::kernels::xmx_gemm_ok(X, W, Y, N, K, ldy))
        strata::kernels::xmx_gemm(XmxType::bf16, X, W, Y, T, N, K, ldy, stream_, accumulate);
    else
        strata::kernels::gemm_rows(XmxType::bf16, X, W, Y, T, N, K, ldy, stream_, accumulate);
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (strata::kernels::xmx_gemm_ok(X, W, Y, N, K, ldy))
        strata::kernels::xmx_gemm(XmxType::f16, X, W, Y, T, N, K, ldy, stream_);
    else
        strata::kernels::gemm_rows(XmxType::f16, X, W, Y, T, N, K, ldy, stream_);
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices (of 16 rows, so every slice's Y stays 64-byte aligned).
        const int64_t rows = scratch_elems_ / K / 16 * 16;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min(rows, N - r0);
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy);
}

}  // namespace strata::prefill
