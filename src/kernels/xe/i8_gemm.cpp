// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/i8_gemm.cpp - see i8_gemm.hpp.
//
// RTX 4070, 2026-11-06: the removed own int8 kernel took 0.595 ms for 4115 x 3072 x 3072;
// cuBLAS took 0.407 ms. Use the library or the fused library kernels below instead.
#include "strata/kernels/i8_gemm.hpp"

#include "strata/core/per_device.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/platform/downgrade.hpp"
#include "device_target.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>

#if STRATA_CUTLASS
#include "i8_gemm_cutlass.hpp"
#endif
#if STRATA_CUTLASS || STRATA_HIP_KITCHEN
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif
#if STRATA_ONEDNN_I8
#include "strata/kernels/onednn.hpp"
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#endif
#if STRATA_HIP_KITCHEN
#include "hip_kitchen.hpp"

#include <cstdio>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#endif

namespace strata::kernels {

namespace {

#if STRATA_CUTLASS
// CUTLASS's kernel (i8_gemm_cutlass.cu) through NVIDIA's driver, which the SYCL runtime's CUDA backend has loaded:
// the driver's C entry points by name (their ABI is fixed; cuda.h is not needed), the module loaded in the GPU's
// primary context, which is the SYCL runtime's.
using CUfunction = struct CUfunc_st*;
struct Driver {
    int (*retain)(CUcontext*, CUdevice) = nullptr;
    int (*push)(CUcontext) = nullptr;
    int (*pop)(CUcontext*) = nullptr;
    int (*attr)(int*, int, CUdevice) = nullptr;
    int (*load)(CUmodule*, const void*) = nullptr;
    int (*function)(CUfunction*, CUmodule, const char*) = nullptr;
    int (*fn_attr)(CUfunction, int, int) = nullptr;
    int (*to_host)(void*, CUdeviceptr, size_t) = nullptr;
    int (*launch)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**,
                  void**) = nullptr;
};

const Driver* driver() {
    static const Driver d = [] {
        Driver r;
#if defined(_WIN32)
        HMODULE h = LoadLibraryA("nvcuda.dll");
        const auto sym = [h](const char* n) { return h ? (void*) GetProcAddress(h, n) : nullptr; };
#else
        void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        const auto sym = [h](const char* n) { return h ? dlsym(h, n) : nullptr; };
#endif
        r.retain = (decltype(r.retain)) sym("cuDevicePrimaryCtxRetain");
        r.push = (decltype(r.push)) sym("cuCtxPushCurrent_v2");
        r.pop = (decltype(r.pop)) sym("cuCtxPopCurrent_v2");
        r.attr = (decltype(r.attr)) sym("cuDeviceGetAttribute");
        r.load = (decltype(r.load)) sym("cuModuleLoadData");
        r.function = (decltype(r.function)) sym("cuModuleGetFunction");
        r.fn_attr = (decltype(r.fn_attr)) sym("cuFuncSetAttribute");
        r.launch = (decltype(r.launch)) sym("cuLaunchKernel");
        r.to_host = (decltype(r.to_host)) sym("cuMemcpyDtoH_v2");
        return r;
    }();
    const bool ok = d.retain && d.push && d.pop && d.attr && d.load && d.function && d.fn_attr && d.launch && d.to_host;
    return ok ? &d : nullptr;
}

struct Cutlass {
    CUcontext ctx = nullptr;
    CUfunction fn[cutlass_i8::kKernels] = {};   // null: the GPU takes none of the code built, or not that kernel
    int smem[cutlass_i8::kKernels] = {};        // the dynamic local memory each function was allowed
    bool any() const {
        for (CUfunction f : fn)
            if (f != nullptr) return true;
        return false;
    }
};

// The kernels' code for `dev`: the cubin of the highest architecture of its major version it runs (an sm_80 cubin runs
// on an sm_86 or sm_89 GPU), else the PTX where the GPU is as new (its driver compiles it), loaded once a GPU.  A kernel
// is taken where the device compile's sizes of its local memory and parameters are the host compile's (the host's
// parameters are then its argument) and the GPU gives a work-group that local memory.
const Cutlass& cutlass_of(sycl::queue& q) {
    static core::PerDevice<Cutlass> of;
    return of.get(q.get_device(), [&q] {
        Cutlass c;
        const Driver* d = driver();
        if (d == nullptr || q.get_device().get_backend() != sycl::backend::ext_oneapi_cuda) return c;
        const CUdevice dev = sycl::get_native<sycl::backend::ext_oneapi_cuda>(q.get_device());
        int major = 0, minor = 0, optin = 0;
        // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _MINOR, MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        if (d->attr(&major, 75, dev) || d->attr(&minor, 76, dev) || d->attr(&optin, 97, dev)) return c;
        const int cc = major * 10 + minor;
        int pick = -1;
        for (int i = 0; i < cutlass_i8::count; ++i) {
            const int sm = cutlass_i8::sm[i];
            const bool runs = cutlass_i8::ptx[i] ? sm <= cc : sm / 10 == major && sm <= cc;
            // a cubin before the PTX; a newer architecture's before an older one's
            if (runs && (pick < 0 || (cutlass_i8::ptx[pick] && !cutlass_i8::ptx[i]) ||
                         (cutlass_i8::ptx[pick] == cutlass_i8::ptx[i] && sm > cutlass_i8::sm[pick])))
                pick = i;
        }
        if (pick < 0 || d->retain(&c.ctx, dev) || d->push(c.ctx)) return Cutlass{};
        constexpr size_t K = cutlass_i8::kKernels;
        unsigned host[2 * K], sizes[2 * K] = {};
        cutlass_i8::sizes(host);
        CUmodule m = nullptr;
        CUfunction fs = nullptr;
        auto* out = sycl::malloc_device<unsigned>(2 * K, q);
        void* sargs[] = {static_cast<void*>(&out)};
        if (out != nullptr && !d->load(&m, cutlass_i8::image[pick]) &&
            !d->function(&fs, m, "strata_i8_gemm_sizes") &&
            !d->launch(fs, 1, 1, 1, 1, 1, 1, 0, nullptr, sargs, nullptr) &&
            !d->to_host(sizes, reinterpret_cast<CUdeviceptr>(out), sizeof sizes)) {
            for (size_t k = 0; k < K; ++k) {
                const std::string name = "strata_i8_gemm_bf16_" + std::to_string(k);
                const unsigned smem = sizes[2 * k], bytes = sizes[2 * k + 1];
                CUfunction f = nullptr;
                // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES for the local memory
                if (smem == host[2 * k] && bytes == host[2 * k + 1] && (int) smem <= optin &&
                    !d->function(&f, m, name.c_str()) && !d->fn_attr(f, 8, (int) smem)) {
                    c.fn[k] = f;
                    c.smem[k] = (int) smem;
                }
            }
        }
        CUcontext was = nullptr;
        d->pop(&was);
        if (out != nullptr) sycl::free(out, q);
        return c;
    });
}

// Y through CUTLASS's kernel on q's CUDA stream; false (nothing launched) where it takes no such product.
bool cutlass_gemm(sycl::queue& q, const cutlass_i8::Problem& p) {
    const Cutlass& c = cutlass_of(q);
    cutlass_i8::Launch l;
    if (!cutlass_i8::params(p, l) || c.fn[l.kernel] == nullptr) return false;
    const Driver* d = driver();
    if (d->push(c.ctx)) return false;
    void* args[] = {l.params};
    const CUstream s = sycl::get_native<sycl::backend::ext_oneapi_cuda>(q);
    const int rc = d->launch(c.fn[l.kernel], l.grid[0], l.grid[1], l.grid[2], l.threads, 1, 1,
                             (unsigned) c.smem[l.kernel], s, args, nullptr);
    CUcontext was = nullptr;
    d->pop(&was);
    return rc == 0;
}
#endif

#if STRATA_HIP_KITCHEN
// comfy-kitchen's int8 product on AMD GPUs with WMMA, in its module (src/kernels/hip/kitchen_module.cpp), opened at an
// AMD GPU's first product from the engine's folder or its lib/; null where it does not open (no HIP runtime here),
// said once.
using KitchenFn = int (*)(const int8_t*, const int8_t*, uint16_t*, const float*, const float*, const float*, int64_t,
                          int64_t, int64_t, int64_t, void*);

KitchenFn kitchen_tiled = nullptr;   // the entry taking X in its tiles (strata_kitchen_i8_gemm_tiled)
using KitchenAttnFn = int (*)(const uint16_t*, const uint16_t*, const uint16_t*, int64_t, int64_t, int64_t, int64_t,
                              int64_t, uint16_t*, void*);
KitchenAttnFn kitchen_attn = nullptr;   // its int8 attention (strata_kitchen_attn)

KitchenFn kitchen_module() {
    static KitchenFn fn = nullptr;
    KitchenFn& tiled = kitchen_tiled;
    static std::once_flag once;
    std::call_once(once, [] {
        namespace fs = std::filesystem;
        std::error_code ec;
        const fs::path dir = fs::read_symlink("/proc/self/exe", ec).parent_path();
        const char* name = "libxestrata_hip_kitchen.so";
        std::string why = "not found";
        for (const fs::path& p : {dir / name, dir / "lib" / name}) {
            if (!fs::exists(p, ec)) continue;
            void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (h == nullptr) {
                const char* e = dlerror();
                why = e != nullptr ? e : p.string();
                continue;
            }
            fn = reinterpret_cast<KitchenFn>(dlsym(h, "strata_kitchen_i8_gemm"));
            tiled = reinterpret_cast<KitchenFn>(dlsym(h, "strata_kitchen_i8_gemm_tiled"));
            kitchen_attn = reinterpret_cast<KitchenAttnFn>(dlsym(h, "strata_kitchen_attn"));
            if (fn != nullptr && tiled != nullptr) return;
            fn = nullptr;
            why = p.string() + ": no strata_kitchen_i8_gemm";
        }
        platform::downgrade("i8_gemm.kitchen", "comfy-kitchen's int8 product module did not open (" + why +
                                               "): the library's product");
    });
    return fn;
}

// Whether q's GPU takes comfy-kitchen's product: a small one (ones by ones, 128 a sum) made and read back once a GPU,
// which fails where the module has no code for the GPU's architecture.
bool kitchen_ok(sycl::queue& q) {
    static core::PerDevice<int> of;
    return of.get(q.get_device(), [&q] {
        const KitchenFn fn = q.get_device().get_backend() == sycl::backend::ext_oneapi_hip ? kitchen_module() : nullptr;
        if (fn == nullptr) return 0;
        constexpr int64_t n = 64, k = 128;
        auto* x = sycl::malloc_device<int8_t>(n * k, q);
        auto* sc = sycl::malloc_device<float>(n, q);
        auto* y = sycl::malloc_device<uint16_t>(n * n, q);
        int ok = 0;
        if (x != nullptr && sc != nullptr && y != nullptr) {
            q.fill<int8_t>(x, 1, n * k);
            q.fill(sc, 1.f, n).wait();
            void* s = sycl::get_native<sycl::backend::ext_oneapi_hip>(q);
            if (fn(x, x, y, sc, sc, nullptr, n, n, k, n, s) == 0) {
                uint16_t v = 0;
                q.wait();
                q.memcpy(&v, y + n * n - 1, sizeof v).wait();
                ok = f32_from_bf16(v) == (float) k;
            }
        }
        q.wait();
        for (void* p : {(void*) x, (void*) sc, (void*) y})
            if (p != nullptr) sycl::free(p, q);
        return ok;
    }) == 1;
}

// X [T][K] into comfy-kitchen's tiles ([T / 128 rounded up][K / 128][128][128], the rows past T zero) in a buffer of
// the GPU's own, grown as needed: its tiled product's schedule, tuned for RDNA4, takes X so.  On the RX 9060 XT
// (2026-11-06) 4115 x 3072 x 16384 took 4.17 ms that way against 7.06 with X's rows as they are (hipBLASLt's int8
// product 7.89), 4115 x 16384 x 3072 3.75 against 4.29 (4.02), 4115 x 3072 x 3072 0.75 against 0.80 (0.66); the copy
// is a pass over X.
const int8_t* kitchen_tiles(sycl::queue& q, const int8_t* X, int64_t T, int64_t K) {
    struct Buf {
        int8_t* p = nullptr;
        size_t bytes = 0;
    };
    static std::mutex mu;
    static std::unordered_map<sycl::device, Buf> bufs;
    const int64_t Tp = (T + 127) / 128 * 128;
    const size_t need = (size_t) (Tp * K);
    int8_t* d = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu);
        Buf& b = bufs[q.get_device()];
        if (b.bytes < need) {
            q.wait();   // the products that read the old buffer
            if (b.p != nullptr) {
                core::graph_epoch_bump();
                sycl::free(b.p, q);
            }
            b.p = sycl::malloc_device<int8_t>(need, q);
            b.bytes = b.p != nullptr ? need : 0;
        }
        d = b.p;
    }
    if (d == nullptr) return nullptr;
    const int64_t kt = K / 128, chunks = Tp * K / 16;   // 16 bytes a work-item
    q.parallel_for(sycl::range<1>((size_t) chunks), [=](sycl::id<1> i) {
        const int64_t c = (int64_t) i[0], tile = c / 1024, in = c % 1024;   // 1024 chunks a tile
        const int64_t rt = tile / kt, ct = tile % kt, r = rt * 128 + in / 8, col = ct * 128 + in % 8 * 16;
        sycl::uint4 v{0, 0, 0, 0};
        if (r < T) v = *reinterpret_cast<const sycl::uint4*>(X + r * K + col);
        *reinterpret_cast<sycl::uint4*>(d + c * 16) = v;
    });
    return d;
}
#endif

#if STRATA_ONEDNN_I8
// oneDNN's int8 matmul (the image engine's oneDNN), where its GPU engine takes the queue's device (Intel's GPUs): X's
// row scales (one a row over the whole of K: oneDNN takes them as groups of K), W's column scales and the bias in its
// epilogue, BF16 out.  On the B70 (2026-11-06), against oneMKL's int8 product (int32 sums out): 4115 x 16384 x 3072
// 1.46 ms (1.59), 4115 x 3072 x 16384 1.22 (1.24), 8230 x 3072 x 16384 2.48 (2.51), 4115 x 3072 x 3072 0.33 (0.29),
// 1024 rows 0.081 (0.082).  A primitive a shape, kept; none on a device whose engine oneDNN refuses.
struct Dnnl {
    std::optional<dnnl::engine> eng;
    bool failed = false;
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, bool>, dnnl::matmul> prims;
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, dnnl::matmul> prims_s32;   // int32 out, no scales
};

// oneDNN's stream on a queue: the queue's own belonging (Runtime::queue_item), gone when its stream is destroyed -
// keyed by the bare queue address it would outlive the queue and meet the next one there
struct DnnlStream {
    explicit DnnlStream(sycl::queue&) {}
    std::optional<dnnl::stream> s;
};

std::mutex g_dnnl_mu;
std::unordered_map<sycl::device, Dnnl> g_dnnl;

// Whether oneDNN's GPU engine takes q's device (made once a device).
bool onednn_ok(sycl::queue& q) {
    onednn_quiet();
    std::lock_guard<std::mutex> lock(g_dnnl_mu);
    Dnnl& d = g_dnnl[q.get_device()];
    if (!d.failed && !d.eng) {
        try {
            d.eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        } catch (const std::exception&) {
            d.failed = true;
        }
    }
    return !d.failed;
}

bool onednn_gemm(sycl::queue& q, const int8_t* X, const float* sx, const int8_t* W, const float* sw,
                 const float* bias, uint16_t* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
    using namespace dnnl;
    if (!onednn_ok(q)) return false;
    std::lock_guard<std::mutex> lock(g_dnnl_mu);
    Dnnl& d = g_dnnl[q.get_device()];
    try {
        const memory::desc a_md({T, K}, memory::data_type::s8, memory::format_tag::ab);
        const memory::desc w_md({K, N}, memory::data_type::s8, memory::format_tag::ba);   // W [N][K] row-major
        const memory::desc b_md({1, N}, memory::data_type::f32, memory::format_tag::ab);
        const memory::desc y_md({T, N}, memory::data_type::bf16, memory::dims{ldy, 1});
        const auto key = std::make_tuple(T, N, K, ldy, bias != nullptr);
        auto it = d.prims.find(key);
        if (it == d.prims.end()) {
            primitive_attr attr;
            attr.set_scales(DNNL_ARG_SRC, 3, {1, K}, memory::data_type::f32);
            attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);
            const auto pd = bias != nullptr ? matmul::primitive_desc(*d.eng, a_md, w_md, b_md, y_md, attr)
                                            : matmul::primitive_desc(*d.eng, a_md, w_md, y_md, attr);
            it = d.prims.emplace(key, matmul(pd)).first;
        }
        DnnlStream& ds = core::Runtime::of_queue(q).queue_item<DnnlStream>(q);
        if (!ds.s) ds.s = sycl_interop::make_stream(*d.eng, q);
        const auto usm = sycl_interop::memory_kind::usm;
        auto mem = [&](const memory::desc& md, const void* p) {
            return sycl_interop::make_memory(md, *d.eng, usm, const_cast<void*>(p));
        };
        std::unordered_map<int, memory> args = {
            {DNNL_ARG_SRC, mem(a_md, X)}, {DNNL_ARG_WEIGHTS, mem(w_md, W)}, {DNNL_ARG_DST, mem(y_md, Y)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
             mem(memory::desc({T, 1}, memory::data_type::f32, memory::format_tag::ab), sx)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
             mem(memory::desc({N}, memory::data_type::f32, memory::format_tag::a), sw)}};
        if (bias != nullptr) args.emplace(DNNL_ARG_BIAS, mem(b_md, bias));
        it->second.execute(*ds.s, args);
        return true;
    } catch (const std::exception&) {
        d.failed = true;   // the product refused: the library's from here on
        return false;
    }
}

// The same with the int32 sums out, no scales nor bias (prefill::Gemm::i8's product where oneMath has none).
bool onednn_gemm_s32(sycl::queue& q, const int8_t* X, const int8_t* W, int32_t* Y, int64_t T, int64_t N, int64_t K,
                     int64_t ldy) {
    using namespace dnnl;
    if (!onednn_ok(q)) return false;
    std::lock_guard<std::mutex> lock(g_dnnl_mu);
    Dnnl& d = g_dnnl[q.get_device()];
    try {
        const memory::desc a_md({T, K}, memory::data_type::s8, memory::format_tag::ab);
        const memory::desc w_md({K, N}, memory::data_type::s8, memory::format_tag::ba);
        const memory::desc y_md({T, N}, memory::data_type::s32, memory::dims{ldy, 1});
        const auto key = std::make_tuple(T, N, K, ldy);
        auto it = d.prims_s32.find(key);
        if (it == d.prims_s32.end())
            it = d.prims_s32.emplace(key, matmul(matmul::primitive_desc(*d.eng, a_md, w_md, y_md))).first;
        DnnlStream& ds = core::Runtime::of_queue(q).queue_item<DnnlStream>(q);
        if (!ds.s) ds.s = sycl_interop::make_stream(*d.eng, q);
        const auto usm = sycl_interop::memory_kind::usm;
        auto mem = [&](const memory::desc& md, const void* p) {
            return sycl_interop::make_memory(md, *d.eng, usm, const_cast<void*>(p));
        };
        it->second.execute(*ds.s, {{DNNL_ARG_SRC, mem(a_md, X)}, {DNNL_ARG_WEIGHTS, mem(w_md, W)},
                                   {DNNL_ARG_DST, mem(y_md, Y)}});
        return true;
    } catch (const std::exception&) {
        d.failed = true;
        return false;
    }
}
#endif

}  // namespace

#if STRATA_HIP_KITCHEN
bool hip_kitchen::attn(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T,
                       int64_t heads, int64_t hd, int64_t r0, int64_t first, uint16_t* out16) {
    if (!kitchen_ok(q) || kitchen_attn == nullptr) return false;
    void* s = sycl::get_native<sycl::backend::ext_oneapi_hip>(q);
    return kitchen_attn(q16, k16, v16, T, heads, hd, r0, first, out16, s) == 0;
}
#endif

bool i8_gemm_ok(void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
#if STRATA_CUTLASS
    if (cutlass_of(q).any()) return true;
#endif
#if STRATA_ONEDNN_I8
    if (onednn_ok(q)) return true;
#endif
#if STRATA_HIP_KITCHEN
    if (kitchen_ok(q)) return true;
#endif
    return false;
}

bool i8_gemm_bf16(const int8_t* X, const float* sx, const int8_t* W, const float* sw, const float* bias, uint16_t* Y,
                 int64_t T, int64_t N, int64_t K, int64_t ldy, void* stream) {
    if (T <= 0 || N <= 0) return true;
    sycl::queue& q = core::Runtime::get().stream(stream);
    if (ldy <= 0) ldy = N;
#if STRATA_CUTLASS
    if (cutlass_gemm(q, cutlass_i8::Problem{X, W, sx, sw, bias, Y, T, N, K, ldy})) {
        if (!stream) q.wait();
        return true;
    }
#endif
#if STRATA_ONEDNN_I8
    if (onednn_gemm(q, X, sx, W, sw, bias, Y, T, N, K, ldy)) {
        if (!stream) q.wait();
        return true;
    }
#endif
#if STRATA_HIP_KITCHEN
    // its kernels read rows 16 bytes at a time; X in its tiles where K takes them
    if (K % 16 == 0 && kitchen_ok(q)) {
        void* s = sycl::get_native<sycl::backend::ext_oneapi_hip>(q);
        const int8_t* xt = K % 128 == 0 ? kitchen_tiles(q, X, T, K) : nullptr;
        if (xt != nullptr ? kitchen_tiled(xt, W, Y, sx, sw, bias, T, N, K, ldy, s) == 0
                          : kitchen_module()(X, W, Y, sx, sw, bias, T, N, K, ldy, s) == 0) {
            if (!stream) q.wait();
            return true;
        }
    }
#endif
    return false;
}

bool i8_gemm_s32_ok([[maybe_unused]] void* stream) {
#if STRATA_ONEDNN_I8
    return onednn_ok(core::Runtime::get().stream(stream));
#else
    return false;
#endif
}

bool i8_gemm_s32([[maybe_unused]] const int8_t* X, [[maybe_unused]] const int8_t* W, [[maybe_unused]] int32_t* Y,
                 int64_t T, int64_t N, [[maybe_unused]] int64_t K, int64_t ldy, void* stream) {
    if (T <= 0 || N <= 0) return true;
#if STRATA_ONEDNN_I8
    sycl::queue& q = core::Runtime::get().stream(stream);
    if (!onednn_gemm_s32(q, X, W, Y, T, N, K, ldy > 0 ? ldy : N)) return false;
    if (!stream) q.wait();
    return true;
#else
    (void) ldy;
    (void) stream;
    return false;
#endif
}

}  // namespace strata::kernels
