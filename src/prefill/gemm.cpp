// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/gemm.cpp - see include/strata/prefill/gemm.hpp.  The CUDA build ran these products through
// cublasGemmEx (FP32 compute).  The contrib modes run them through oneMath (STRATA_ONEMATH: oneMKL on Intel GPUs,
// cuBLAS on NVIDIA ones), the free mode through XeStrata's own kernels, which also take a GPU oneMath has no backend
// for.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/gpu.hpp"

#if STRATA_ONEMATH
#include "strata/core/runtime.hpp"
#include <oneapi/math/blas.hpp>
#include <future>
#include <string>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {

using strata::kernels::XmxType;

namespace {
#if STRATA_ONEMATH
bool g_blas = true;   // false once oneMath had no backend for the GPU
// Gemm::prepare's product: empty when oneMath took it, else why not
std::future<std::string> g_ready;

template <class E>
void product(sycl::queue& q, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K,
             int64_t ldy, bool accumulate) {
    namespace om = oneapi::math;
    om::blas::column_major::gemm(q, om::transpose::trans, om::transpose::nontrans, N, T, K, 1.0f,
                                 reinterpret_cast<const E*>(W), K, reinterpret_cast<const E*>(X), K,
                                 accumulate ? 1.0f : 0.0f, Y, ldy);
}

// Y[T, N] row-major is Y^T[N, T] column-major (ld ldy) = W^T . X^T, with row-major W[N, K] read as the transpose of a
// column-major K x N and X[T, K] as a column-major K x T.  beta 0 leaves Y unread.
template <class E>
bool blas(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
          bool accumulate, void* stream) {
    if (g_ready.valid()) {
        const std::string why = g_ready.get();
        if (!why.empty()) {
            std::fprintf(stderr, "strata: oneMath: %s; the prompt's matrix products take XeStrata's own kernels\n",
                         why.c_str());
            g_blas = false;
        }
    }
    if (!g_blas) return false;
    try {
        product<E>(core::Runtime::get().stream(stream), X, W, Y, T, N, K, ldy, accumulate);
        return true;
    } catch (const oneapi::math::exception& e) {
        std::fprintf(stderr, "strata: oneMath: %s; the prompt's matrix products take XeStrata's own kernels\n",
                     e.what());
        g_blas = false;
        return false;
    }
}
#endif
}  // namespace

void Gemm::prepare() {
#if STRATA_ONEMATH
    // A backend's first product loads its library and its kernels (cuBLAS: about 0.6 s on an RTX 4070, inside the
    // first prompt): one small product of each type on a queue of its own, beside the model's loading.
    sycl::queue& c = core::Runtime::get().compute();
    g_ready = std::async(std::launch::async, [ctx = c.get_context(), dev = c.get_device()]() -> std::string {
        constexpr int64_t n = 16;
        constexpr size_t tile = (size_t) n * n;
        sycl::queue q(ctx, dev, sycl::property::queue::in_order{});
        auto* x = sycl::malloc_device<uint16_t>(2 * tile, q);
        auto* y = sycl::malloc_device<float>(tile, q);
        std::string why;
        try {
            q.fill<uint16_t>(x, 0, 2 * tile);
            product<sycl::half>(q, x, x + tile, y, n, n, n, n, false);
            product<oneapi::math::bfloat16>(q, x, x + tile, y, n, n, n, n, false);
            q.wait_and_throw();
        } catch (const oneapi::math::exception& e) {
            why = e.what();
        }
        q.wait();
        sycl::free(x, q);
        sycl::free(y, q);
        return why;
    });
#endif
}

const char* Gemm::path() {
#if STRATA_ONEMATH
    if (g_blas) return "oneMath";
#endif
    return strata::kernels::gemm_path(XmxType::f16);
}

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
#if STRATA_ONEMATH
    if (blas<oneapi::math::bfloat16>(X, W, Y, T, N, K, ldy, accumulate, stream_)) return;
#endif
    if (strata::kernels::xmx_gemm_ok(X, W, Y, N, K, ldy))
        strata::kernels::xmx_gemm(XmxType::bf16, X, W, Y, T, N, K, ldy, stream_, accumulate);
    else
        strata::kernels::gemm_rows(XmxType::bf16, X, W, Y, T, N, K, ldy, stream_, accumulate);
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
#if STRATA_ONEMATH
    if (blas<sycl::half>(X, W, Y, T, N, K, ldy, false, stream_)) return;
#endif
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
