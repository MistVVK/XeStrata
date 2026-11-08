// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/prefill/gemm.cpp - see include/strata/prefill/gemm.hpp.  The CUDA build ran these products through
// cublasGemmEx (FP32 compute).  The contrib modes run them through oneMath (STRATA_ONEMATH: oneMKL on Intel GPUs,
// cuBLAS on NVIDIA ones), the free mode through XeStrata's own kernels, which also take a GPU oneMath has no backend
// for.
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/i8_gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/rows_cols.hpp"
#include "strata/platform/env.hpp"

#if STRATA_ONEMATH
#include "strata/kernels/matrix_report.hpp"
#include <oneapi/math/blas.hpp>
#include <filesystem>
#include <future>
#include <type_traits>
#include <vector>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

namespace strata::prefill {

using strata::kernels::XmxType;

namespace {
#if STRATA_ONEMATH
bool env_on(const char* name) {
    const char* v = strata::env_get(name);
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
    bool blas = !env_on("NO_BLAS");
    bool bf16 = !strata::kernels::no_bf16_mma();
    int i8 = -1;   // whether the backend takes int8 products (Gemm::i8_usable's trial; -1 before it)
    std::shared_future<Trial> ready;   // Gemm::prepare's trial, until it is settled
};
std::mutex g_mutex;
std::unordered_map<sycl::device, Backend> g_backends;

// Windows: the GPU makers' libraries the 7z does not carry, where their installers put them, for oneMath's backends to
// load: before the PATH the user set, oneMKL of the series the engine was built with (the Library\bin pip's oneMKL makes
// in the Python the 7z carries, <7z>\python beside <7z>\app\engine, and oneAPI's mkl folder of the series, under
// ONEAPI_ROOT, which the installer sets, else its default folder); after it oneAPI's tbb folder and MKLROOT's for
// oneMKL; the CUDA toolkit's (CUDA_PATH) and the folders pip's nvidia-cublas-cu12 and nvidia-cuda-runtime-cu12 make in
// that Python for cuBLAS; the folder AMD's rocm-sdk-libraries-<family> makes in that Python for rocBLAS and
// hipBLASLt (free software, but built for each GPU family; the import libraries name its DLLs,
// tools/package/win_rocm.sh).  Without them the backend does not load and the products take XeStrata's own kernels.
void add_library_dirs() {
#if defined(_WIN32)
    static std::once_flag once;
    std::call_once(once, [] {
        std::vector<std::string> roots;
        if (const char* r = std::getenv("ONEAPI_ROOT")) roots.emplace_back(r);
        roots.emplace_back("C:\\Program Files (x86)\\Intel\\oneAPI");
        // oneMKL of the series the engine was built with first, before the PATH's (another series's SYCL library may
        // not load on the engine's SYCL runtime: setup.py's ONEMKL_SERIES): pip's in the Python the 7z carries and
        // oneAPI's folder of the series
        std::vector<std::string> first;
        std::vector<std::string> dirs;
        for (const std::string& r : roots) {
#if defined(STRATA_ONEMKL_SERIES)
            first.push_back(r + "\\mkl\\" + std::string(STRATA_ONEMKL_SERIES) + "\\bin");
#endif
            dirs.push_back(r + "\\tbb\\latest\\bin");
        }
        if (const char* m = std::getenv("MKLROOT")) dirs.push_back(std::string(m) + "\\bin");
        if (const char* c = std::getenv("CUDA_PATH")) dirs.push_back(std::string(c) + "\\bin");
        char exe[MAX_PATH] = {};
        if (GetModuleFileNameA(nullptr, exe, MAX_PATH) > 0) {
            const std::filesystem::path site =
                std::filesystem::path(exe).parent_path().parent_path().parent_path() / "python" / "Lib" / "site-packages";
            for (const char* sub : {"cublas", "cuda_runtime"}) dirs.push_back((site / "nvidia" / sub / "bin").string());
            // pip's oneMKL (onemkl-sycl-blas, mkl, tbb): their DLLs in the Python's Library\bin
            first.insert(first.begin(), (site.parent_path().parent_path() / "Library" / "bin").string());
            // AMD's rocm-sdk-libraries-<family> (rocBLAS and hipBLASLt): one folder for the GPU family it was made for
            std::error_code ec;
            for (const auto& e : std::filesystem::directory_iterator(site, ec))
                if (e.path().filename().string().rfind("_rocm_sdk_libraries_", 0) == 0)
                    dirs.push_back((e.path() / "bin").string());
        }
        const char* old = std::getenv("PATH");
        std::string path;
        std::error_code ec;
        for (const std::string& d : first)
            if (std::filesystem::is_directory(d, ec)) path += d + ";";
        path += old != nullptr ? old : "";
        for (const std::string& d : dirs)
            if (std::filesystem::is_directory(d, ec)) path += ";" + d;
        _putenv_s("PATH", path.c_str());
    });
#endif
}

// The backend of `dev`, its trial's outcome taken in (waited for) the first time
Backend backend(const sycl::device& dev) {
    add_library_dirs();
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
            std::fprintf(stderr, "xestrata: oneMath: %s; the prompt's matrix products take XeStrata's own kernels\n",
                         t.why.c_str());
            b.blas = false;
        } else if (!t.bf16 && b.bf16) {
            std::fprintf(stderr, "xestrata: oneMath has no BF16 product on this GPU: those take XeStrata's own "
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
        std::fprintf(stderr, "xestrata: oneMath: %s; the prompt's matrix products take XeStrata's own kernels\n",
                     e.what());
        std::lock_guard<std::mutex> lock(g_mutex);
        g_backends[q.get_device()].blas = false;
        return false;
    }
}
// STRATA_DENSE_F32=1 (opt-in, upstream 40ba0cf9): the FP16 products as FP32 ones on FP32 copies of both operands.
// For FP16 operands with FP32 sums cuBLAS 12.4 has two Maxwell kernels on a Tesla P40 and picks the slow one for most
// chunk sizes (qkv 10240 x 2560 at 7.6 TFLOPS up to T 7040, 4.3 from T 7041); its SGEMM runs at 8.4-9.8 from T 512,
// and for the engine's shapes upstream measured the same bits as the FP16 call.  That is the cuBLAS build's property,
// so opt-in.  The copies are kept per queue and grown to the largest product (a few hundred MiB for the big ones).
// The product with FP16 sums into FP16 rows Y16 (row stride ldy): oneMath's half GEMM (cuBLAS's HGEMM on NVIDIA GPUs,
// where GeForce cards run it at twice the FP32 sums' rate); false where oneMath has none for the GPU, or it failed.
bool half_out(const uint16_t* X, const uint16_t* W, uint16_t* Y16, int64_t T, int64_t N, int64_t K, int64_t ldy,
              void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    if (!backend(q.get_device()).blas) return false;
    try {
        namespace om = oneapi::math;
        om::blas::column_major::gemm(q, om::transpose::trans, om::transpose::nontrans, N, T, K, sycl::half(1.0f),
                                     reinterpret_cast<const sycl::half*>(W), K, reinterpret_cast<const sycl::half*>(X),
                                     K, sycl::half(0.0f), reinterpret_cast<sycl::half*>(Y16), ldy);
    } catch (const oneapi::math::exception& e) {
        std::fprintf(stderr, "xestrata: oneMath (FP16 sums): %s; the FP32 sums instead\n", e.what());
        return false;
    }
    return true;
}

// f16's half_sums: half_out into a half buffer kept per queue and grown to the largest, then widened into Y.
bool half_sums(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    const Backend b = backend(q.get_device());
    if (!b.blas) return false;
    struct Buf { sycl::half* p = nullptr; size_t n = 0; };
    static std::mutex mu;
    static std::unordered_map<const sycl::queue*, Buf> bufs;
    const size_t need = (size_t) T * (size_t) N;
    sycl::half* yh = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu);
        Buf& c = bufs[&q];
        if (c.n < need) {
            q.wait();
            if (c.p != nullptr) {
                sycl::free(c.p, q);
                core::graph_epoch_bump();   // a recorded step may hold the old one
            }
            c.p = sycl::malloc_device<sycl::half>(need, q);
            c.n = c.p != nullptr ? need : 0;
        }
        yh = c.p;
    }
    if (yh == nullptr || !half_out(X, W, reinterpret_cast<uint16_t*>(yh), T, N, K, N, stream)) return false;
    strata::kernels::for_rows_cols(q, T, N, [=](int64_t t, int64_t n) { Y[t * ldy + n] = (float) yh[t * N + n]; });
    return true;
}

bool dense_f32(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               void* stream) {
    static const bool on = env_on("DENSE_F32");
    if (!on) return false;
    sycl::queue& q = core::Runtime::get().stream(stream);
    const Backend b = backend(q.get_device());
    if (!b.blas) return false;
    struct Copies { float* p = nullptr; size_t n = 0; };
    static std::mutex mu;
    static std::unordered_map<const sycl::queue*, Copies> copies;
    const size_t need = (size_t) (T + N) * (size_t) K;
    float* buf = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu);
        Copies& c = copies[&q];
        if (c.n < need) {
            q.wait();
            if (c.p != nullptr) {
                sycl::free(c.p, q);
                core::graph_epoch_bump();
            }
            c.p = sycl::malloc_device<float>(need, q);
            c.n = c.p != nullptr ? need : 0;
        }
        buf = c.p;
    }
    if (buf == nullptr) return false;
    float* xf = buf;
    float* wf = buf + (size_t) T * (size_t) K;
    const auto* xh = reinterpret_cast<const sycl::half*>(X);
    const auto* wh = reinterpret_cast<const sycl::half*>(W);
    q.parallel_for(sycl::range<1>((size_t) T * (size_t) K), [=](sycl::id<1> i) { xf[i] = (float) xh[i]; });
    q.parallel_for(sycl::range<1>((size_t) N * (size_t) K), [=](sycl::id<1> i) { wf[i] = (float) wh[i]; });
    try {
        namespace om = oneapi::math;
        om::blas::column_major::gemm(q, om::transpose::trans, om::transpose::nontrans, N, T, K, 1.0f, wf, K, xf, K, 0.0f,
                                     Y, ldy);
        return true;
    } catch (const oneapi::math::exception& e) {
        std::fprintf(stderr, "xestrata: oneMath (XESTRATA_DENSE_F32): %s; the FP16 products instead\n", e.what());
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
    add_library_dirs();
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
    // what runs the products on the current GPU, said as the report finds it: oneMath with the library its backend
    // for the device's SYCL backend uses, or XeStrata's own kernels in the form they take; the own kernels also run
    // the products oneMath does not (the experts' grouped ones), so that form is said beside oneMath
    static thread_local std::string text;
    const std::string own = std::string("XeStrata's own kernels (") + strata::kernels::gemm_path(XmxType::f16) + ")";
    text = own;
#if STRATA_ONEMATH
    const sycl::device dev = core::Runtime::get().device();
    if (backend(dev).blas) {
        const sycl::backend be = dev.get_backend();
        const char* lib = be == sycl::backend::ext_oneapi_hip ? "rocBLAS"
                          : be == sycl::backend::ext_oneapi_cuda ? "cuBLAS"
                                                                 : "oneMKL";
        text = std::string("oneMath (") + lib + "), " + own + " for the rest";
    }
#endif
    return text.c_str();
}

bool gemm_dequant_side() {
    static const bool on = [] { const char* v = strata::env_get("DEQUANT_SIDE"); return v && v[0] == '1'; }();
    return on;
}

Gemm::~Gemm() {
    for (strata::gpu::Event* e : side_ready_)
        if (e) strata::gpu::event_destroy(e);
    for (strata::gpu::Event* e : side_free_)
        if (e) strata::gpu::event_destroy(e);
    if (side_stream_) strata::gpu::stream_destroy(side_stream_);
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

bool Gemm::set_dequant_side(uint16_t* scratch2, std::string& err) {
    side_scratch_ = scratch2;
    side_cur_ = 0;
    if (scratch2 == nullptr) return true;
    if (side_stream_ == nullptr) {   // once: the stream and the four events outlive every rebind
        side_stream_ = strata::gpu::stream_create();
        if (side_stream_ == nullptr) {
            err = std::string("prefill gemm: dequant side stream: ") + strata::gpu::last_error();
            side_scratch_ = nullptr;
            return false;
        }
        for (int i = 0; i < 2; ++i) {
            // kept even on a failure: the destructor frees what was made
            if (!strata::gpu::event_create(&side_ready_[i]) || !strata::gpu::event_create(&side_free_[i])) {
                err = "prefill gemm: dequant side events";
                side_scratch_ = nullptr;
                return false;
            }
        }
    }
    // both scratches are free once what the compute stream holds now is done (a rebind follows a stream sync, so
    // this is belt and braces)
    for (strata::gpu::Event* e : side_free_) strata::gpu::event_record(e, stream_);
    return true;
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
        const char* v = strata::env_get("BF16_TC");
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

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               bool half, bool accumulate) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
#if STRATA_ONEMATH
    // the FP32 copies and the half-sum path leave Y unread (or write it whole): an accumulating product keeps the
    // half GEMM's FP32 sums, as the engine's own kernels do
    if (!accumulate && dense_f32(X, W, Y, T, N, K, ldy, stream_)) return;
    if (half && !accumulate && half_sums(X, W, Y, T, N, K, ldy, stream_)) return;
    if (blas<sycl::half>(X, W, Y, T, N, K, ldy, accumulate, stream_)) return;
#else
    (void) half;
#endif
    if (strata::kernels::xmx_gemm_ok(X, W, Y, N, K, ldy))
        strata::kernels::xmx_gemm(XmxType::f16, X, W, Y, T, N, K, ldy, stream_, accumulate);
    else
        strata::kernels::gemm_rows(XmxType::f16, X, W, Y, T, N, K, ldy, stream_, accumulate);
}

void Gemm::f16_batch(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t batch,
                     int64_t sx, int64_t sw, int64_t sy, int64_t ldy) {
    if (T <= 0 || N <= 0 || batch <= 0) return;
    if (ldy <= 0) ldy = N;
#if STRATA_ONEMATH
    sycl::queue& q = core::Runtime::get().stream(stream_);
    if (backend(q.get_device()).blas) {
        try {
            namespace om = oneapi::math;
            om::blas::column_major::gemm_batch(q, om::transpose::trans, om::transpose::nontrans, N, T, K, 1.0f,
                                               reinterpret_cast<const sycl::half*>(W), K, sw,
                                               reinterpret_cast<const sycl::half*>(X), K, sx, 0.0f, Y, ldy, sy, batch);
            return;
        } catch (const oneapi::math::exception& e) {
            std::fprintf(stderr, "xestrata: oneMath (a batch of FP16 products): %s; one product each\n", e.what());
        }
    }
#endif
    for (int64_t b = 0; b < batch; ++b) f16(X + b * sx, W + b * sw, Y + b * sy, T, N, K, ldy);
}

// Whether oneMath's backend for `stream`'s GPU has int8 products (asked of it once, by a trial).
static bool onemath_i8(void* stream) {
#if STRATA_ONEMATH
    sycl::queue& q = core::Runtime::get().stream(stream);
    const Backend b = backend(q.get_device());
    if (!b.blas) return false;
    if (b.i8 >= 0) return b.i8 != 0;
    // a small product on a queue of its own, so that an error the backend reports afterwards is caught here: a backend
    // without int8 products (XeStrata's oneMath before xestrata-5: rocBLAS's and cuBLAS's) throws at the call
    constexpr int64_t n = 64;
    auto failed = std::make_shared<bool>(false);
    sycl::queue t(q.get_context(), q.get_device(),
                  [failed](const sycl::exception_list& l) { *failed = *failed || l.size() > 0; },
                  sycl::property::queue::in_order{});
    auto* x = sycl::malloc_device<int8_t>(2 * n * n, t);
    auto* y = sycl::malloc_device<int32_t>(n * n, t);
    bool ok = x != nullptr && y != nullptr;
    if (ok) {
        try {
            t.fill<int8_t>(x, 1, 2 * n * n);
            namespace om = oneapi::math;
            om::blas::column_major::gemm_batch(t, om::transpose::trans, om::transpose::nontrans, n, n, n, 1.0f, x, n,
                                               n * n, x + n * n, n, n * n, 0.0f, y, n, n * n, 1);
            t.wait_and_throw();
            int32_t v = 0;
            t.memcpy(&v, y, sizeof v).wait();
            ok = !*failed && v == n;
        } catch (const std::exception&) {
            ok = false;
        }
    }
    t.wait();
    if (x != nullptr) sycl::free(x, t);
    if (y != nullptr) sycl::free(y, t);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_backends[q.get_device()].i8 = ok ? 1 : 0;
    return ok;
#else
    (void) stream;
    return false;
#endif
}

bool Gemm::i8_usable(void* stream) { return onemath_i8(stream) || strata::kernels::i8_gemm_s32_ok(stream); }

void Gemm::i8(const int8_t* X, const int8_t* W, int32_t* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
              void* stream) {
    if (T <= 0 || N <= 0) return;
    // oneDNN's where it takes the GPU (Intel's), in both modes: as fast as oneMKL's through oneMath or faster, the same
    // sums (the B70, the image DiTs' shapes from 256 x 3072 x 3072 to 8230 x 16384 x 3072: 0.80 to 1.00 of its time,
    // 2026-11-06), and the int8 products' one library there (its BF16 ones, kernels::i8_gemm_bf16, are oneDNN's)
    if (strata::kernels::i8_gemm_s32_ok(stream) || !onemath_i8(stream)) {
        if (!strata::kernels::i8_gemm_s32(X, W, Y, T, N, K, ldy, stream))
            throw std::runtime_error("an int8 product on a GPU no library takes it on");
        return;
    }
#if STRATA_ONEMATH
    namespace om = oneapi::math;
    if (ldy <= 0) ldy = N;
    // one product of a batch (gemm_batch: oneMath's int8 product), the strides those of a whole matrix (oneMKL checks
    // them even for one)
    om::blas::column_major::gemm_batch(core::Runtime::get().stream(stream), om::transpose::trans,
                                       om::transpose::nontrans, N, T, K, 1.0f, W, K, N * K, X, K, T * K, 0.0f, Y, ldy,
                                       ldy * T, 1);
#endif
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

namespace {
// the shapes warm on each GPU in the process (Gemm::native_warm)
struct WarmShapes {
    std::mutex mu;
    std::set<std::tuple<int, int64_t, int64_t, int64_t>> of;
};
WarmShapes& warm_shapes(void* stream) {
    static core::PerDevice<std::shared_ptr<WarmShapes>> per_device;
    return *per_device.get(core::Runtime::get().stream(stream).get_device(),
                           [] { return std::make_shared<WarmShapes>(); });
}
}  // namespace

bool Gemm::native_cold(int ggml_type, int64_t N, int64_t K, int64_t ldy) const {
    WarmShapes& w = warm_shapes(stream_);
    std::lock_guard<std::mutex> lock(w.mu);
    return ggml_type != 0 && w.of.count({ggml_type, N, K, ldy}) == 0;
}

bool Gemm::native_warm(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N,
                       int64_t K, int64_t ldy) {
    if (W_blocks == nullptr || ggml_type == 0) return false;
    {
        WarmShapes& w = warm_shapes(stream_);
        std::lock_guard<std::mutex> lock(w.mu);
        if (!w.of.insert({ggml_type, N, K, ldy}).second) return false;
    }
    native(X, ggml_type, W_blocks, Y, T, N, K, ldy);
    return true;
}

// W's FP16 rows for prod(w16, r0, n) (rows r0 .. r0 + n - 1 of W): as they are for F16 rows, else dequantized into the
// scratch, at once or in row slices, on the side stream with STRATA_DEQUANT_SIDE
template<class F> void Gemm::with_f16(int ggml_type, const void* W_blocks, int64_t N, int64_t K, F prod) {
    if (ggml_type == 1) {   // F16 rows (an F16 GGUF's, Anima's Qwen3-0.6B-Base): the product takes them as they are
        prod(static_cast<const uint16_t*>(W_blocks), 0, N);
        return;
    }
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices (of 16 rows, so every slice's Y stays 64-byte aligned).
        const int64_t rows = scratch_elems_ / K / 16 * 16;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min(rows, N - r0);
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            prod(scratch_, r0, n);
        }
        // the side stream's next dequant into scratch_ waits for these slices' reads and writes of it
        if (side_scratch_) strata::gpu::event_record(side_free_[0], stream_);
        return;
    }
    if (side_scratch_) {
        // STRATA_DEQUANT_SIDE (upstream dc390c386): the dequant on the side stream, into the scratch the compute
        // stream read last but one.  free[cur] -> dequant -> ready[cur] -> product -> free[cur] orders the dequant
        // after the product that last read this scratch and the product after the dequant that filled it, so every
        // read and write of a scratch is where the serial form had it - the same kernels, the same bits - and the
        // dequant runs while the kernels queued ahead of this product do.
        uint16_t* const s = side_cur_ == 0 ? scratch_ : side_scratch_;
        strata::gpu::stream_wait_event(side_stream_, side_free_[side_cur_]);
        strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, s, side_stream_);
        strata::gpu::event_record(side_ready_[side_cur_], side_stream_);
        strata::gpu::stream_wait_event(stream_, side_ready_[side_cur_]);
        prod(s, 0, N);
        strata::gpu::event_record(side_free_[side_cur_], stream_);
        side_cur_ ^= 1;
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    prod(scratch_, 0, N);
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, bool half_sums) {
    if (ldy <= 0) ldy = N;
    with_f16(ggml_type, W_blocks, N, K, [&](const uint16_t* w, int64_t r0, int64_t n) {
        f16(X, w, Y + r0, T, n, K, ldy, half_sums);
    });
}

bool Gemm::half_ok(void* stream) {
#if STRATA_ONEMATH
    return backend(core::Runtime::get().stream(stream).get_device()).blas;
#else
    (void) stream;
    return false;
#endif
}

void Gemm::native_half(const uint16_t* X, int ggml_type, const void* W_blocks, uint16_t* Y16, int64_t T, int64_t N,
                       int64_t K, int64_t ldy) {
    if (ldy <= 0) ldy = N;
    with_f16(ggml_type, W_blocks, N, K, [&](const uint16_t* w, int64_t r0, int64_t n) {
#if STRATA_ONEMATH
        if (half_out(X, w, Y16 + r0, T, n, K, ldy, stream_)) return;
#endif
        // no half product for the GPU: the FP32 product into a buffer kept per queue (grown to the largest), narrowed
        sycl::queue& q = core::Runtime::get().stream(stream_);
        struct Buf { float* p = nullptr; size_t n = 0; };
        static std::mutex mu;
        static std::unordered_map<const sycl::queue*, Buf> bufs;
        const size_t need = (size_t) T * (size_t) n;
        float* y = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu);
            Buf& c = bufs[&q];
            if (c.n < need) {
                q.wait();
                if (c.p != nullptr) {
                    sycl::free(c.p, q);
                    core::graph_epoch_bump();
                }
                c.p = sycl::malloc_device<float>(need, q);
                c.n = c.p != nullptr ? need : 0;
            }
            y = c.p;
        }
        if (y == nullptr) throw std::runtime_error("prefill gemm: no GPU memory for an FP16 product's FP32 sums");
        f16(X, w, y, T, n, K, n);
        uint16_t* out = Y16 + r0;
        const int64_t nn = n, ld = ldy;
        strata::kernels::for_rows_cols(q, T, n, [=](int64_t t, int64_t c) {
            out[t * ld + c] = strata::kernels::f16_from_f32(y[t * nn + c]);
        });
    });
}

}  // namespace strata::prefill
