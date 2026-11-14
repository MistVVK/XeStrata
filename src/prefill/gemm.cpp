// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/prefill/gemm.cpp - see include/strata/prefill/gemm.hpp.  The CUDA build ran these products through
// cublasGemmEx (FP32 compute).  The contrib modes run them through oneMath (STRATA_ONEMATH: oneMKL on Intel GPUs,
// cuBLAS on NVIDIA ones), the free mode through XeStrata's own kernels, which also take a GPU oneMath has no backend
// for.
#include "strata/prefill/gemm.hpp"
#include "strata/platform/downgrade.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/i8_gemm.hpp"
#include "strata/kernels/rows_i8.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/timed_choice.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/rows_cols.hpp"
#include "strata/platform/env.hpp"

#if STRATA_ONEMATH
#include "strata/kernels/matrix_report.hpp"
#include <oneapi/math/blas.hpp>
#include <filesystem>
#include <future>
#include <map>
#include <functional>
#include <type_traits>
#include <vector>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <new>
#include <type_traits>
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
            platform::downgrade("gemm.onemath", "oneMath: " + t.why + "; the prompt's matrix products take XeStrata's own "
                                                "kernels");
            b.blas = false;
        } else if (!t.bf16 && b.bf16) {
            platform::downgrade("gemm.onemath_bf16", "oneMath has no BF16 product on this GPU: those take XeStrata's "
                                                     "own kernels");
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
        platform::downgrade("gemm.onemath", std::string("oneMath: ") + e.what() + "; the prompt's matrix products take "
                                            "XeStrata's own kernels");
        std::lock_guard<std::mutex> lock(g_mutex);
        g_backends[q.get_device()].blas = false;
        return false;
    }
}
// The dense products XeStrata's own kernels can take in place of oneMath's library: X . W^T in FP16 and in BF16, and
// X . W with W as it is (FP16; a training step's dgrad).
enum class Own { f16 = 0, bf16 = 1, kn = 2 };

// own_first's choices: by GPU, then by the product's kind, its rows' power-of-two class, N and K (in the process-local timed cache)
core::TimedCache<sycl::device, std::tuple<int, int, int64_t, int64_t>, bool> g_own;

// Whether a product of kind `k` and shape T x N x K on `stream`'s GPU takes XeStrata's own kernels (mma_gemm) before
// oneMath's library: the faster of the two, `own` and `lib` (each runs the product itself, into its own output), timed
// the first time a product of that kind, N, K and rows' power-of-two class comes, and kept for the GPU (the own
// kernels 5% faster or more; the RTX 4070 took them for its small-N products: 1016 x 512 x 2560 at 44.9 TFLOPS against
// cuBLAS's 35.2, the large ones staying with cuBLAS at 55-63 against 47-51).  One shape for
// them all was not enough: on an RX 7900 XTX (ROCm 10.1, 2026-11-09) mma_gemm ran 1024 x 4096 x 4096 in FP16 at 26-28
// TFLOPS and rocBLAS at 9-12, but with that choice for every product Qwen3.8-Flash-Next's decode (verify windows of a
// few rows) ran at 51 tok/s against rocBLAS's 70, and its 1K prefill at 226 against 305.  `measure` false (an
// accumulating product, whose output a trial would change; a queue recording a graph, which cannot wait) takes a
// choice already made for the class and otherwise the library.  XESTRATA_BLAS=1 keeps the library; without one (the
// trial failed, or XESTRATA_NO_BLAS=1) the products take the own kernels anyway.
//
// The choice is made per start, and the library's speed moves between starts: on the RX 9060 XT the same BF16 product
// (4096 x 320 x 10240) measured 13.1 to 25.5 TFLOPS through rocBLAS while XeStrata's kernels held at 16.8-21.3, and a
// class near the two paths' speed changed sides from start to start, taking the answer's last bits with it (2026-11-14:
// the same binary and settings split a 64K greedy answer at token 18).  The library's own fast and slow states gave the
// same bits.  A run whose answer must be the same every time takes the own kernels throughout: XESTRATA_NO_BLAS=1.
bool own_first(void* stream, Own k, int64_t T, int64_t N, int64_t K, bool measure, const std::function<void()>& own,
               const std::function<void()>& lib) {
    static const bool library = env_on("BLAS");
    const XmxType t = k == Own::bf16 ? XmxType::bf16 : XmxType::f16;
    if (library || !strata::kernels::mma_products(t, k == Own::kn)) return false;
    sycl::queue& q = core::Runtime::get().stream(stream);
    const sycl::device dev = q.get_device();
    const Backend b = backend(dev);
    if (!b.blas || (k == Own::bf16 && !b.bf16)) return true;
    int rows = 0;
    while (((int64_t) 1 << rows) < T) ++rows;
    const auto key = std::make_tuple((int) k, rows, N, K);
    if (const auto found = g_own.get(dev, key)) return *found;
    namespace sx = sycl::ext::oneapi::experimental;
    if (!measure || q.ext_oneapi_get_state() == sx::queue_state::recording) return false;
    // each once first (its kernels' build, the library's load), then the two in turn, the fastest of each kept: the
    // mean of three runs, one after the other, moved a shape from run to run, and the image with it (the RTX 4070,
    // Mage-Flow's 1024 x 384 x 384: the library 13.9 or 20.0 TFLOPS from start to start, 2026-11-11)
    double own_s = 0, lib_s = 0;
    try {
        const auto measured = core::timed_choice({{[&] { own(); return true; }}, {[&] { lib(); return true; }}},
                                                 [&] { q.wait_and_throw(); });
        own_s = measured.seconds[0];
        lib_s = measured.seconds[1];
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata: measuring a dense product: %s\n", e.what());
        own_s = lib_s = 0;
    }
    // the own kernels only where they are 5% faster or more: a near tie stays with the library, so that the noise of a
    // trial does not flip a shape from run to run (as the image engine's W8A8 and FP16 forms are chosen)
    const auto [pick, first] = g_own.put_first(dev, key, own_s > 0 && lib_s > 0 && own_s * 1.05 < lib_s,
                                              [&](const auto& shape) { return std::get<0>(shape) == (int) k; });
    if (own_s > 0 && lib_s > 0 && (first || strata::env_get("TRACE") != nullptr)) {
        const sycl::backend be = dev.get_backend();
        const char* name = be == sycl::backend::ext_oneapi_hip    ? "rocBLAS"
                           : be == sycl::backend::ext_oneapi_cuda ? "cuBLAS"
                                                                  : "oneMKL";
        const double flop = 2.0 * (double) T * (double) N * (double) K;
        static const char* const kName[3] = {"FP16", "BF16", "FP16 (W as it is)"};
        std::fprintf(stderr, "xestrata: dense %s products, measured a shape at a time (%lld x %lld x %lld: XeStrata's "
                     "own kernels %.1f TFLOPS, %s %.1f)\n", kName[(int) k], (long long) T, (long long) N,
                     (long long) K, flop / own_s / 1e12, name, flop / lib_s / 1e12);
    }
    return pick;
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
        platform::downgrade("gemm.onemath_f16_sums", std::string("oneMath (FP16 sums): ") + e.what() +
                                                     "; the FP32 sums instead");
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

void Gemm::finish() {
#if STRATA_ONEMATH
    std::vector<std::shared_future<Trial>> running;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (const auto& [dev, b] : g_backends)
            if (b.ready.valid()) running.push_back(b.ready);
    }
    for (const auto& f : running) f.wait();
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
    if (backend(dev).blas && strata::kernels::mma_products(XmxType::f16)) {
        const char* lib = dev.get_backend() == sycl::backend::ext_oneapi_hip ? "rocBLAS" : "the library";
        text = own + " or oneMath (" + lib + "), the faster a product's shape";
    } else if (backend(dev).blas) {
        const sycl::backend be = dev.get_backend();
        const char* lib = be == sycl::backend::ext_oneapi_hip ? "rocBLAS"
                          : be == sycl::backend::ext_oneapi_cuda ? "cuBLAS"
                                                                 : "oneMKL";
        text = std::string("oneMath (") + lib + "), " + own + " for the rest";
    }
#endif
    return text.c_str();
}

Gemm::~Gemm() {
    if (!external_ && scratch_) strata::gpu::free(scratch_);
    if (tc_w_) strata::gpu::free(tc_w_);
    if (tc_x_) strata::gpu::free(tc_x_);
    for (void* p : {(void*) w8_x_, (void*) w8_sx_, (void*) w8_w_, (void*) w8_sw_})
        if (p) strata::gpu::free(p);
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
    const bool own = strata::kernels::xmx_gemm_ok(X, W, Y, N, K, ldy) &&
                     own_first(stream_, Own::bf16, T, N, K, !accumulate,
                               [&] { strata::kernels::xmx_gemm(XmxType::bf16, X, W, Y, T, N, K, ldy, stream_); },
                               [&] {
                                   product<oneapi::math::bfloat16>(core::Runtime::get().stream(stream_), X, W, Y, T,
                                                                   N, K, ldy, false);
                               });
    if (!own && blas<oneapi::math::bfloat16>(X, W, Y, T, N, K, ldy, accumulate, stream_)) return;
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
    const bool own = strata::kernels::xmx_gemm_ok(X, W, Y, N, K, ldy) &&
                     own_first(stream_, Own::f16, T, N, K, !accumulate,
                               [&] { strata::kernels::xmx_gemm(XmxType::f16, X, W, Y, T, N, K, ldy, stream_); },
                               [&] {
                                   product<sycl::half>(core::Runtime::get().stream(stream_), X, W, Y, T, N, K, ldy,
                                                       false);
                               });
    // the FP32 copies and the half-sum path leave Y unread (or write it whole): an accumulating product keeps the
    // half GEMM's FP32 sums, as the engine's own kernels do
    if (!own && !accumulate && dense_f32(X, W, Y, T, N, K, ldy, stream_)) return;
    if (!own && half && !accumulate && half_sums(X, W, Y, T, N, K, ldy, stream_)) return;
    if (!own && blas<sycl::half>(X, W, Y, T, N, K, ldy, accumulate, stream_)) return;
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
    // a choice own_first made for single products of this shape (none measured here: the batch's own form is a product
    // each)
    if (backend(q.get_device()).blas && !own_first(stream_, Own::f16, T, N, K, false, nullptr, nullptr)) {
        try {
            namespace om = oneapi::math;
            om::blas::column_major::gemm_batch(q, om::transpose::trans, om::transpose::nontrans, N, T, K, 1.0f,
                                               reinterpret_cast<const sycl::half*>(W), K, sw,
                                               reinterpret_cast<const sycl::half*>(X), K, sx, 0.0f, Y, ldy, sy, batch);
            return;
        } catch (const oneapi::math::exception& e) {
            platform::downgrade("gemm.onemath_batch", std::string("oneMath (a batch of FP16 products): ") + e.what() +
                                                      "; one product each");
        }
    }
#endif
    for (int64_t b = 0; b < batch; ++b) f16(X + b * sx, W + b * sw, Y + b * sy, T, N, K, ldy);
}

bool Gemm::f16_batch_ex_ok() const {
#if STRATA_ONEMATH
    return backend(core::Runtime::get().stream(stream_).get_device()).blas;
#else
    return false;
#endif
}

bool Gemm::f16_batch_ex(bool x_t, bool w_nk, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N,
                        int64_t K, int64_t batch, int64_t sx, int64_t sw, int64_t sy, int64_t ldx, int64_t ldw,
                        int64_t ldy, float beta) {
    if (T <= 0 || N <= 0 || batch <= 0) return true;
#if STRATA_ONEMATH
    // X [T][K] and W [K][N] as they are, one product (a training step's dgrad, dY . W): XeStrata's own kernels where
    // they are the faster (own_first)
    if (!x_t && !w_nk && batch == 1 && beta == 0.0f && ldx == K &&
        own_first(stream_, Own::kn, T, N, K, true,
                  [&] {
                      if (!strata::kernels::gemm_kn(X, W, ldw, Y, T, N, K, ldy, stream_))
                          throw std::runtime_error("gemm_kn does not take this shape");
                  },
                  [&] {
                      namespace om = oneapi::math;
                      om::blas::column_major::gemm(core::Runtime::get().stream(stream_), om::transpose::nontrans,
                                                   om::transpose::nontrans, N, T, K, 1.0f,
                                                   reinterpret_cast<const sycl::half*>(W), ldw,
                                                   reinterpret_cast<const sycl::half*>(X), K, 0.0f, Y, ldy);
                  }) &&
        strata::kernels::gemm_kn(X, W, ldw, Y, T, N, K, ldy, stream_))
        return true;
    sycl::queue& q = core::Runtime::get().stream(stream_);
    if (backend(q.get_device()).blas) {
        try {
            namespace om = oneapi::math;
            // column-major: Y^T [N][T] = op(W)^T-side [N][K] . op(X)^T-side [K][T].  W [N][K] row-major is K x N
            // column-major (transposed to N x K); W [K][N] row-major is N x K as it is.  X [T][K] row-major is K x T
            // as it is; X [K][T] row-major is T x K (transposed to K x T)
            om::blas::column_major::gemm_batch(q, w_nk ? om::transpose::trans : om::transpose::nontrans,
                                               x_t ? om::transpose::trans : om::transpose::nontrans, N, T, K, 1.0f,
                                               reinterpret_cast<const sycl::half*>(W), ldw, sw,
                                               reinterpret_cast<const sycl::half*>(X), ldx, sx, beta, Y, ldy, sy,
                                               batch);
            return true;
        } catch (const oneapi::math::exception& e) {
            std::fprintf(stderr, "xestrata: oneMath (a batch of transposed FP16 products): %s\n", e.what());
        }
    }
#else
    (void) x_t;
    (void) w_nk;
    (void) X;
    (void) W;
    (void) Y;
    (void) K;
    (void) sx;
    (void) sw;
    (void) sy;
    (void) ldx;
    (void) ldw;
    (void) ldy;
    (void) beta;
#endif
    return false;
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

namespace {
// Whether the W8A8 products whose readers take BF16 go through the GPU maker's fused kernel (kernels::i8_gemm_bf16: the
// scales and bias in its epilogue, BF16 out; oneDNN's on Intel, CUTLASS's on NVIDIA, comfy-kitchen's on RDNA3 and
// RDNA4) or the library's int32 sums (Gemm::i8, scaled by their reader): timed once a GPU over an image DiT's three
// shapes at 4096 rows (3072 x 3072, 12288 x 3072, 3072 x 12288) through core::timed_choice (each form run once before
// its timings, the two in turn), the fused kernel kept unless the library is 5% faster.  The image DiTs chose so in
// their own code before (dit_rows::w8_fused, the least of three after a warm-up): the B70 took the library in the first
// start after a build and the fused kernel after.  Timed a shape at a time as the products came instead (2026-11-12),
// the choice moved from start to start with the small shapes' noise (the B70's Mage-Flow at 512 x 512, its 54-row
// products, at a margin of 15%), and the product's own time favours the library: the fused kernel's BF16 values save
// the reader half the bytes (the RTX 4070's Qwen-Image-2.1, the library 7 to 10.5% faster alone at four of its shapes:
// taken, the sampling ran 1.997 -> 2.025 s).  A queue recording a graph (which cannot wait), or a timing around this
// product (core::timing_depth), takes the library until the GPU is timed.  XESTRATA_W8_FUSED=0 or 1 fixes it.
bool w8_fused(void* stream) {
    static const int fixed = [] {
        const char* e = strata::env_get("W8_FUSED");
        return e == nullptr ? -1 : std::strtol(e, nullptr, 10) != 0 ? 1 : 0;
    }();
    if (fixed >= 0) return fixed == 1;
    if (!strata::kernels::i8_gemm_ok(stream)) return false;
    if (!Gemm::i8_usable(stream)) return true;
    sycl::queue& q = core::Runtime::get().stream(stream);
    struct Kept {
        std::mutex m;
        int of = -1;
    };
    static core::PerDevice<std::shared_ptr<Kept>> per_device;
    Kept& kept = *per_device.get(q.get_device(), [] { return std::make_shared<Kept>(); });
    std::lock_guard<std::mutex> lock(kept.m);
    if (kept.of >= 0) return kept.of == 1;
    namespace sx = sycl::ext::oneapi::experimental;
    if (core::timing_depth > 0 || q.ext_oneapi_get_state() == sx::queue_state::recording) return false;
    constexpr int64_t T = 4096;
    const int64_t shapes[3][2] = {{3072, 3072}, {12288, 3072}, {3072, 12288}};
    std::vector<void*> held;
    auto get = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!strata::gpu::alloc_device(&p, bytes)) return nullptr;
        held.push_back(p);
        strata::gpu::memset_async(p, 1, bytes, stream);
        return p;
    };
    double fused = -1, lib = -1;
    try {
        auto* x8 = static_cast<int8_t*>(get((size_t) (T * 12288)));
        auto* w8 = static_cast<int8_t*>(get((size_t) (12288 * 3072)));
        void* y = get((size_t) (T * 12288) * 4);
        auto* sx = static_cast<float*>(get((size_t) T * 4));
        auto* sw = static_cast<float*>(get((size_t) 12288 * 4));
        if (x8 == nullptr || w8 == nullptr || y == nullptr || sx == nullptr || sw == nullptr) throw std::bad_alloc();
        const auto measured = core::timed_choice(
            {{[&] {
                 for (const auto& sh : shapes)
                     if (!strata::kernels::i8_gemm_bf16(x8, sx, w8, sw, nullptr, static_cast<uint16_t*>(y), T, sh[0],
                                                        sh[1], 0, stream))
                         return false;
                 return true;
             }},
             {[&] {
                 for (const auto& sh : shapes) Gemm::i8(x8, w8, static_cast<int32_t*>(y), T, sh[0], sh[1], sh[0], stream);
                 return true;
             }}},
            [&] { q.wait_and_throw(); }, 3);   // three timings each, as before: five took the B70's Anima 11 ms more
        fused = measured.seconds[0];
        lib = measured.seconds[1];
        kept.of = measured.pick == 0 ? 1 : 0;
    } catch (const std::exception& e) {
        // as before: the fused kernel (a product it does not take goes to the library at the call)
        std::fprintf(stderr, "xestrata: timing the int8 products: %s\n", e.what());
        kept.of = 1;
    }
    strata::gpu::stream_sync(stream);
    for (void* p : held) strata::gpu::free(p);
    std::fprintf(stderr, "xestrata: int8 products through %s (an image DiT's three at 4096 rows: the fused kernel %.2f ms, "
                         "the library %.2f)\n", kept.of == 1 ? "the fused kernel" : "the library",
                 fused > 0 ? 1e3 * fused : -1.0, lib > 0 ? 1e3 * lib : -1.0);
    return kept.of == 1;
}
}  // namespace

// Whether the LLMs' dense prompt products take the W8A8 form (Gemm::native_llm): for a dense model (no experts),
// timed once a GPU over three of their representative shapes at 4096 rows (Qwen3.8-Flash-Next's attn_qkv 2560 ->
// 10240 IQ4_XS and attn_output 6144 -> 2560 Q6_K, Qwen3.6-35B-A3B's attn_qkv 2048 -> 8192 Q8_0), the whole form
// against native (the weight's int8 rows made at each call, the activations' rows, the int32 product, the scales
// back into FP32), through core::timed_choice, native the preferred side with a wide margin: W8A8 taken only when
// 10% faster.  The margin sits between the GPUs' edges, where no start takes the other side: at timed_choice's own 5%
// the choice moved from run to run on the B70 (2026-11-12: the three shapes native 3.77 to 3.81 ms, W8A8 3.57 to
// 3.62, a 4 to 6% edge astride the margin), at 15% on the RX 9060 XT (2026-11-12: native 8.91 to 9.06 ms, W8A8 7.74
// to 7.78, a 14.5 to 16.5% edge, native in 2 starts of 5; its prompts take W8A8 for the speed, Qwen3.5-9B's 32K 2009
// -> 2497 tok/s); the RTX 4070's 8.1 to 8.8 against 4.7 to 4.8, some 45%, is far past either.  Five timings each
// after 50 ms of both in turn (the GPU clocked up): three right after each form's first run took W8A8 in 6 of 7
// starts and native in one on the RX 9060 XT (2026-11-12).
// Not a shape at a time either: the narrow margins of the small shapes moved the choice from start to start (as
// recorded; the image DiTs' w8_fused above).  A queue recording a graph,
// or a timing around this product (core::timing_depth), takes native until the GPU is timed.
// A MoE model keeps native by default: its dense products (the GDN's and QSA's projections, not its experts) timed
// W8A8 faster on every GPU tried, and the form's KL against the same binary's native runs passed the bar (a mean of
// 0.022, the top token kept) nowhere it was taken.  The RX 9060 XT (2026-11-14, 64 tokens after a 32K prompt, the
// native runs' own floor 0 to 0.006): Qwen3.8-Flash-Next 0.029 / 98.4%, Qwen3.6-35B-A3B 0.013 / 96.9%, Ornith-1.5
// 0.051 / 95.3%, for a prompt 4% faster (Qwen3.8-Flash-Next 1218 -> 1267 tok/s) or the same (Ornith-1.5 1924 ->
// 1922); the RTX 4070 (2026-11-12, 32K): Qwen3.8-Flash-Next 0.00955 / 100%, Qwen3.6-35B-A3B 0.04622 / 89.06%.  The
// timing's own buffers and the int8 library's load also shift the expert cache's placement, and the answers with it
// (the B70, 2026-11-12; the RX 9060 XT's Qwen3.8-Flash-Next 1K, where the form is not taken, 0.040 against a floor of
// 0).  Strix Halo's gfx1151 (2026-11-14) is where the form pays: 2.1x on the three shapes, the 8K prompt 552 -> 732
// tok/s and the 32K 628 -> 811 with the same greedy answers; its KL is not measured, so XESTRATA_LLM_W8A8=1 takes it
// there.  A dense model's default is the timed choice (the user's decision 2026-11-12: the RTX 4070's Qwen3.5-9B 32K
// prompt 3245 -> 5462 tok/s; its KL 0.00411 / 96.88% against the BLAS floor 0.00107 / 100% accepted).
// XESTRATA_LLM_W8A8=0 fixes the form off (a dense model too), =1 on (a MoE model too, without the timing).
bool Gemm::llm_w8a8(void* stream, bool dense) {
    static const int fixed = [] {
        const char* e = strata::env_get("LLM_W8A8");
        return e == nullptr ? -1 : std::strtol(e, nullptr, 10) != 0 ? 1 : 0;
    }();
    if (fixed >= 0) return fixed == 1 && Gemm::i8_usable(stream);
    if (!dense) return false;   // a MoE model: no timing, no int8 library (the comment above)
    if (!Gemm::i8_usable(stream)) return false;
    sycl::queue& q = core::Runtime::get().stream(stream);
    struct Kept {
        std::mutex m;
        int of = -1;
    };
    static core::PerDevice<std::shared_ptr<Kept>> per_device;
    Kept& kept = *per_device.get(q.get_device(), [] { return std::make_shared<Kept>(); });
    std::lock_guard<std::mutex> lock(kept.m);
    if (kept.of >= 0) return kept.of == 1;
    namespace sx = sycl::ext::oneapi::experimental;
    if (core::timing_depth > 0 || q.ext_oneapi_get_state() == sx::queue_state::recording) return false;
    constexpr int64_t T = 4096;
    struct Shape { int type; int64_t N, K; size_t blk; };   // blk: a block's bytes (256 values, Q8_0's 32)
    const Shape shapes[] = {{23, 10240, 2560, 136}, {14, 2560, 6144, 210}, {8, 8192, 2048, 272}};
    std::vector<void*> held;
    auto get = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!strata::gpu::alloc_device(&p, bytes)) return nullptr;
        held.push_back(p);
        strata::gpu::memset_async(p, 0x11, bytes, stream);
        return p;
    };
    double nat = -1, w8 = -1;
    int64_t timed_rows = T;
    try {
        auto* x16 = static_cast<uint16_t*>(get((size_t) (T * 6144) * 2));
        auto* y = static_cast<float*>(get((size_t) (T * 10240) * 4));
        const void* wb[3];
        int64_t max_elems = 0;
        for (int i = 0; i < 3; ++i) {
            const size_t bytes = (size_t) (shapes[i].N * shapes[i].K) / 256 * shapes[i].blk;
            wb[i] = get(bytes);
            max_elems = std::max(max_elems, shapes[i].N * shapes[i].K);
        }
        if (x16 == nullptr || y == nullptr || wb[0] == nullptr || wb[1] == nullptr || wb[2] == nullptr)
            throw std::bad_alloc();
        Gemm tg;
        std::string err;
        if (!tg.init(stream, max_elems, err)) throw std::runtime_error(err);
        const auto measured = core::timed_choice(
            {{[&] {
                 for (int i = 0; i < 3; ++i) tg.native(x16, shapes[i].type, wb[i], y, T, shapes[i].N, shapes[i].K);
                 return true;
             }},
             {[&] {
                 for (int i = 0; i < 3; ++i)
                     tg.native_w8a8(x16, shapes[i].type, wb[i], y, T, shapes[i].N, shapes[i].K, 0);
                 return true;
             }}},
            [&] { q.wait_and_throw(); }, 5, 1.10, 0.05);
        nat = measured.seconds[0];
        w8 = measured.seconds[1];
        kept.of = measured.pick == 1 ? 1 : 0;
    } catch (const std::exception& e) {
        // native: the form the products took before (a GPU whose memory the timing's buffers do not fit)
        std::fprintf(stderr, "xestrata: timing the LLMs' dense products: %s\n", e.what());
        kept.of = 0;
    }
    strata::gpu::stream_sync(stream);
    for (void* p : held) strata::gpu::free(p);
    std::fprintf(stderr, "xestrata: the LLMs' dense prompt products through %s (three shapes at %lld rows: native "
                         "%.2f ms, W8A8 %.2f)\n", kept.of == 1 ? "W8A8" : "native", (long long) timed_rows,
                 nat > 0 ? 1e3 * nat : -1.0, w8 > 0 ? 1e3 * w8 : -1.0);
    return kept.of == 1;
}

// The LLMs' dense prompt products take the W8A8 form through native_llm (their callers only: a DiT's FP16 products
// stay FP16).  Timed on synthetic data (2026-11-11, a probe: the rows to int8 with the rotation, the weight made
// int8 at each call, FP32 out as the LLMs' readers take it), Qwen3.8-Flash-Next's attn_qkv (2560 -> 10240, IQ4_XS)
// at 8192 rows: the RTX 4070 6.93 ms native against about 3.9 W8A8 (2.36 with BF16 out, 3.50 with the weight's int8
// rows resident), at 1024 rows 1.07 against 0.81.  The B70: 2.76 native against about 2.8 (1.84 with BF16 out, 1.47
// resident), at 1024 rows 0.72 against 0.73.  The whole form timed on the three representative shapes (2026-11-12)
// put the B70's W8A8 4 to 6% ahead of native (3.77 to 3.81 ms against 3.57 to 3.62), an edge that moved the choice
// from run to run at a 5% margin: the 10% margin keeps native there.  Making it faster on the B70 needs a BF16
// reader (every kernel downstream of the products) or the weights' int8 rows resident (several GiB of VRAM for a
// whole model's dense weights, in the budget): not taken here.  Narrow
// products (the shared experts, N 512 and 640) were slower in W8A8 from 4096 rows on the B70 (0.21 -> 0.28 ms at
// 8192 rows), so native_llm takes only N >= 1024; below 1024 rows the weight's conversion outweighs the product
// (the B70's attn_qkv at 512 rows: 0.74 native, 0.73 W8A8 with BF16 out, more with FP32's).
bool Gemm::w8a8(const int8_t* X, const float* sx, const int8_t* W, const float* sw, const float* bias, void* Y,
                int64_t T, int64_t N, int64_t K, int64_t ldy, bool bf16_out, void* stream) {
    if (T <= 0 || N <= 0) return false;
    auto* y16 = static_cast<uint16_t*>(Y);
    auto* y32 = static_cast<int32_t*>(Y);
    if (bf16_out && w8_fused(stream) && strata::kernels::i8_gemm_bf16(X, sx, W, sw, bias, y16, T, N, K, ldy, stream))
        return true;
    i8(X, W, y32, T, N, K, ldy, stream);
    return false;
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
// scratch, at once or in row slices
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

void Gemm::native_w8a8(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N,
                       int64_t K, int64_t ldy) {
    if (ldy <= 0) ldy = N;
    const auto grow = [this](auto*& p, int64_t& have, int64_t want) {
        if (have >= want) return true;
        if (p) {
            strata::gpu::stream_sync(stream_);   // the products queued on the old buffer
            strata::gpu::free(p);
        }
        p = static_cast<std::remove_reference_t<decltype(p)>>(strata::gpu::alloc_device((size_t) want * sizeof(*p)));
        have = p ? want : 0;
        return p != nullptr;
    };
    if (!grow(w8_x_, w8_x_elems_, T * K) || !grow(w8_sx_, w8_sx_elems_, T) || !grow(w8_w_, w8_w_elems_, N * K) ||
        !grow(w8_sw_, w8_sw_elems_, N))
        throw std::runtime_error("prefill gemm: no GPU memory for the W8A8 prompt products' scratch");
    // the weight's int8 rows: straight from the blocks where the quantizer takes their type, else through the FP16
    // scratch (in row slices where the weight is larger than it, as with_f16)
    if (!strata::kernels::rows_i8_blocks(ggml_type, W_blocks, N, K, w8_w_, w8_sw_, stream_)) {
        if (ggml_type == 1) {   // F16 rows: the quantizer takes them as they are
            strata::kernels::rows_i8(static_cast<const uint16_t*>(W_blocks), false, N, K, w8_w_, w8_sw_, nullptr,
                                     stream_);
        } else {
            const int64_t rows = std::min(N, scratch_elems_ / K / 16 * 16);
            if (rows <= 0) throw std::runtime_error("prefill gemm: scratch too small for a W8A8 weight's rows");
            for (int64_t r0 = 0; r0 < N; r0 += rows) {
                const int64_t n = std::min(rows, N - r0);
                strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
                strata::kernels::rows_i8(scratch_, false, n, K, w8_w_ + r0 * K, w8_sw_ + r0, nullptr, stream_);
            }
        }
    }
    strata::kernels::rows_i8(X, false, T, K, w8_x_, w8_sx_, nullptr, stream_);
    w8a8(w8_x_, w8_sx_, w8_w_, w8_sw_, nullptr, Y, T, N, K, ldy, false, stream_);
    strata::kernels::i8_scale(Y, T, N, ldy, w8_sx_, w8_sw_, nullptr, nullptr, stream_);
}

void Gemm::native_llm(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N,
                      int64_t K, bool dense, int64_t ldy, bool half_sums) {
    // the shapes the W8A8 form was measured faster for (the comment at w8a8): wide products of many rows, K in whole
    // rotation groups.  The rest stay native: the narrow shared experts (N 512, 640) slower from 4096 rows on the
    // B70, the weight's conversion outweighing the product below 1024 rows
    if (T >= kLlmW8a8Rows && N >= 1024 && K % strata::kernels::kI8Rotate == 0 && ggml_type != 0 && W_blocks != nullptr &&
        llm_w8a8(stream_, dense)) {
        native_w8a8(X, ggml_type, W_blocks, Y, T, N, K, ldy);
        return;
    }
    native(X, ggml_type, W_blocks, Y, T, N, K, ldy, half_sums);
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
