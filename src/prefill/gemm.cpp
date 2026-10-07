// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/gemm.cpp - see include/strata/prefill/gemm.hpp.  The CUDA build ran these products through
// cublasGemmEx (FP32 compute).  The contrib modes run them through oneMath (STRATA_ONEMATH: oneMKL on Intel GPUs,
// cuBLAS on NVIDIA ones), the free mode through XeStrata's own kernels, which also take a GPU oneMath has no backend
// for.
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/gpu.hpp"

#if STRATA_ONEMATH
#include "strata/core/runtime.hpp"
#include "strata/kernels/matrix_report.hpp"
#include <oneapi/math/blas.hpp>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill {

using strata::kernels::XmxType;

namespace {
#if STRATA_ONEMATH
bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strtol(v, nullptr, 10) != 0;
}

struct Trial {
    std::string why;   // empty when oneMath took the products
    bool bf16 = true;
};
// Per GPU (a layer split drives GPUs of different makers, so oneMKL on one and cuBLAS on the next): whether oneMath
// takes the products - not once it had no backend for the GPU, nor with STRATA_NO_BLAS=1 (XeStrata's own kernels, to
// compare) - and BF16 ones too - not when the trial product failed (a GPU whose library has no BF16 product), nor with
// STRATA_NO_BF16_MMA=1, which imitates a GPU without BF16 matrix engines.
struct Backend {
    bool blas = !env_on("STRATA_NO_BLAS");
    bool bf16 = !strata::kernels::no_bf16_mma();
    std::shared_future<Trial> ready;   // Gemm::prepare's trial, until it is settled
};
std::mutex g_mutex;
std::unordered_map<sycl::device, Backend> g_backends;

// The backend of `dev`, its trial's outcome taken in (waited for) the first time
Backend backend(const sycl::device& dev) {
    std::shared_future<Trial> ready;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        Backend& b = g_backends[dev];
        if (!b.ready.valid()) return b;
        ready = b.ready;
    }
    const Trial t = ready.get();
    std::lock_guard<std::mutex> lock(g_mutex);
    Backend& b = g_backends[dev];
    if (b.ready.valid()) {
        b.ready = {};
        if (!t.why.empty()) {
            std::fprintf(stderr, "strata: oneMath: %s; the prompt's matrix products take XeStrata's own kernels\n",
                         t.why.c_str());
            b.blas = false;
        } else if (!t.bf16 && b.bf16) {
            std::fprintf(stderr, "strata: oneMath has no BF16 product on this GPU: those take XeStrata's own "
                                 "kernels\n");
            b.bf16 = false;
        }
    }
    return b;
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
    sycl::queue& q = core::Runtime::get().stream(stream);
    const Backend b = backend(q.get_device());
    if (!b.blas || (std::is_same_v<E, oneapi::math::bfloat16> && !b.bf16)) return false;
    try {
        product<E>(q, X, W, Y, T, N, K, ldy, accumulate);
        return true;
    } catch (const oneapi::math::exception& e) {
        std::fprintf(stderr, "strata: oneMath: %s; the prompt's matrix products take XeStrata's own kernels\n",
                     e.what());
        std::lock_guard<std::mutex> lock(g_mutex);
        g_backends[q.get_device()].blas = false;
        return false;
    }
}
#endif
}  // namespace

void Gemm::prepare() {
#if STRATA_ONEMATH
    // A backend's first product loads its library and its kernels (cuBLAS: about 0.6 s on an RTX 4070, inside the
    // first prompt): one small product of each type on a queue of its own, beside the model's loading.  An error of the
    // BF16 one (thrown, or reported by the queue afterwards) leaves the BF16 products to XeStrata's own kernels.
    sycl::queue& c = core::Runtime::get().compute();
    std::lock_guard<std::mutex> lock(g_mutex);
    Backend& b = g_backends[c.get_device()];
    if (!b.blas || b.ready.valid()) return;
    b.ready = std::async(std::launch::async, [ctx = c.get_context(), dev = c.get_device()]() -> Trial {
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
    }).share();
#endif
}

void Gemm::settle() {
#if STRATA_ONEMATH
    (void) backend(core::Runtime::get().compute().get_device());
#endif
}

const char* Gemm::path() {
#if STRATA_ONEMATH
    if (backend(core::Runtime::get().device()).blas) return "oneMath";
#endif
    return strata::kernels::gemm_path(XmxType::f16);
}

Gemm::~Gemm() {
    if (!external_ && scratch_) strata::gpu::free(scratch_);
    if (tc_w_) strata::gpu::free(tc_w_);
    if (tc_x_) strata::gpu::free(tc_x_);
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

// The BF16 products on a GPU whose matrix engines take FP16 and not BF16 (NVIDIA's before sm_80, where cuBLAS runs BF16
// inputs on an FP32 kernel without the tensor cores: a fifth of a 4K prompt on an RTX 2080 Ti, upstream f2fb7c1,
// ff6f9f1): W, then X a slice at a time, converted to FP16 in this instance's own buffers and multiplied as FP16 (FP32
// accumulation).  The conversion is exact for every value in FP16's normal range, where the weights and the normalized
// activations are; the finite ones past it saturate.  An accumulating product (STRATA_PREFILL_BF16X2's remainder, deep
// in FP16's subnormal range) and a single output row keep the BF16 path, as does a failed allocation.
// STRATA_BF16_TC=0 turns it off.
bool Gemm::bf16_through_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K,
                            int64_t ldy) {
    static const bool off = [] {
        const char* v = std::getenv("STRATA_BF16_TC");
        return v != nullptr && v[0] == '0';
    }();
    if (off || N <= 1 || K <= 0 || !strata::kernels::f16_only_matrix_engines()) return false;
    constexpr int64_t kXSliceElems = 16ll << 20;   // 32 MiB of FP16 activations a slice (upstream's)
    const int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
    const auto grow = [this](uint16_t*& p, int64_t& have, int64_t want) {
        if (have >= want) return true;
        if (p) {
            strata::gpu::stream_sync(stream_);   // the products queued on the old buffer
            strata::gpu::free(p);
        }
        p = static_cast<uint16_t*>(strata::gpu::alloc_device((size_t) want * 2));
        have = p ? want : 0;
        return p != nullptr;
    };
    if (!grow(tc_w_, tc_w_elems_, N * K) || !grow(tc_x_, tc_x_elems_, x_rows * K)) return false;
    bf16_to_f16(W, tc_w_, N * K, stream_);
    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
        const int64_t nt = std::min(x_rows, T - t0);
        bf16_to_f16(X + t0 * K, tc_x_, nt * K, stream_);
        f16(tc_x_, tc_w_, Y + t0 * ldy, nt, N, K, ldy);
    }
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                bool accumulate) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (!accumulate && bf16_through_f16(X, W, Y, T, N, K, ldy)) return;
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

bool Gemm::f16_groups(const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y, const int32_t* bounds, int G,
                      int64_t N, int64_t K, void* stream) {
#if STRATA_ONEMATH
    // Only where the grouped kernel would run on DP4a: with matrix engines it beats a product an expert (on the B70 56
    // and 55 TFLOP/s against oneMKL's 45 and 44).  Without them, a product an expert (dozens to hundreds of rows): a
    // 4K prompt's experts took 1.7 s through hipBLASLt on the RX 9060 XT, against 8.5 s in the grouped DP4a kernel.  A
    // failed product leaves the whole call to the grouped kernel, which writes every row again.
    if (std::strcmp(strata::kernels::gemm_path(XmxType::f16), "DP4a") != 0) return false;
    for (int e = 0; e < G; ++e) {
        const int64_t r0 = bounds[e], rows = (int64_t) bounds[e + 1] - r0;
        if (rows > 0 && !blas<sycl::half>(X + r0 * K, W + e * w_stride, Y + r0 * N, rows, N, K, N, false, stream))
            return false;
    }
    return true;
#else
    (void) X; (void) W; (void) w_stride; (void) Y; (void) bounds; (void) G; (void) N; (void) K; (void) stream;
    return false;
#endif
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
