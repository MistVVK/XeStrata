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
#include "strata/kernels/matrix_report.hpp"
#include <oneapi/math/blas.hpp>
#include <future>
#include <memory>
#include <string>
#include <type_traits>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {

using strata::kernels::XmxType;

namespace {
#if STRATA_ONEMATH
bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strtol(v, nullptr, 10) != 0;
}

// Whether oneMath takes the products: not once it had no backend for the GPU, nor with STRATA_NO_BLAS=1 (XeStrata's
// own kernels, to compare).  BF16 also not when the trial product failed (a GPU whose library has no BF16 product), or
// with STRATA_NO_BF16_MMA=1, which imitates a GPU without BF16 matrix engines.
bool& blas_on() {
    static bool v = !env_on("STRATA_NO_BLAS");
    return v;
}
bool& bf16_on() {
    static bool v = !strata::kernels::no_bf16_mma();
    return v;
}

struct Trial {
    std::string why;   // empty when oneMath took the products
    bool bf16 = true;
};
std::future<Trial> g_ready;   // Gemm::prepare's trial

void settle() {
    if (!g_ready.valid()) return;
    const Trial t = g_ready.get();
    if (!t.why.empty()) {
        std::fprintf(stderr, "strata: oneMath: %s; the prompt's matrix products take XeStrata's own kernels\n",
                     t.why.c_str());
        blas_on() = false;
    } else if (!t.bf16 && bf16_on()) {
        std::fprintf(stderr, "strata: oneMath has no BF16 product on this GPU: those take XeStrata's own kernels\n");
        bf16_on() = false;
    }
}

// Y[T, N] row-major is Y^T[N, T] column-major (ld ldy) = W^T . X^T, with row-major W[N, K] read as the transpose of a
// column-major K x N and X[T, K] as a column-major K x T.  beta 0 leaves Y unread.
template <class E>
void product(sycl::queue& q, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K,
             int64_t ldy, bool accumulate) {
    namespace om = oneapi::math;
    om::blas::column_major::gemm(q, om::transpose::trans, om::transpose::nontrans, N, T, K, 1.0f,
                                 reinterpret_cast<const E*>(W), K, reinterpret_cast<const E*>(X), K,
                                 accumulate ? 1.0f : 0.0f, Y, ldy);
}

template <class E>
bool blas(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
          bool accumulate, void* stream) {
    settle();
    if (!blas_on() || (std::is_same_v<E, oneapi::math::bfloat16> && !bf16_on())) return false;
    try {
        product<E>(core::Runtime::get().stream(stream), X, W, Y, T, N, K, ldy, accumulate);
        return true;
    } catch (const oneapi::math::exception& e) {
        std::fprintf(stderr, "strata: oneMath: %s; the prompt's matrix products take XeStrata's own kernels\n",
                     e.what());
        blas_on() = false;
        return false;
    }
}
#endif
}  // namespace

void Gemm::prepare() {
#if STRATA_ONEMATH
    if (!blas_on()) return;
    // A backend's first product loads its library and its kernels (cuBLAS: about 0.6 s on an RTX 4070, inside the
    // first prompt): one small product of each type on a queue of its own, beside the model's loading.  An error of the
    // BF16 one (thrown, or reported by the queue afterwards) leaves the BF16 products to XeStrata's own kernels.
    sycl::queue& c = core::Runtime::get().compute();
    g_ready = std::async(std::launch::async, [ctx = c.get_context(), dev = c.get_device()]() -> Trial {
        constexpr int64_t n = 16;
        constexpr size_t tile = (size_t) n * n;
        auto failed = std::make_shared<bool>(false);
        sycl::queue q(ctx, dev, [failed](const sycl::exception_list& l) { *failed = *failed || l.size() > 0; },
                      sycl::property::queue::in_order{});
        auto* x = sycl::malloc_device<uint16_t>(2 * tile, q);
        auto* y = sycl::malloc_device<float>(tile, q);
        Trial t;
        try {
            q.fill<uint16_t>(x, 0, 2 * tile);
            product<sycl::half>(q, x, x + tile, y, n, n, n, n, false);
            q.wait_and_throw();
            try {
                product<oneapi::math::bfloat16>(q, x, x + tile, y, n, n, n, n, false);
                q.wait_and_throw();
            } catch (const std::exception&) {
                t.bf16 = false;
            }
            t.bf16 = t.bf16 && !*failed;
        } catch (const oneapi::math::exception& e) {
            t.why = e.what();
        }
        q.wait();
        sycl::free(x, q);
        sycl::free(y, q);
        return t;
    });
#endif
}

const char* Gemm::path() {
#if STRATA_ONEMATH
    if (blas_on()) return "oneMath";
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
