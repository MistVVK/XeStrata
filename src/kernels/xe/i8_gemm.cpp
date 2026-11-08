// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/i8_gemm.cpp - see i8_gemm.hpp.
//
// CUTLASS's kernel where the contrib-llvm build carries it (STRATA_CUTLASS: NVIDIA GPUs from sm_80 on), launched
// through NVIDIA's driver; else the engine's own, which no product takes yet (own_ok):
// NVIDIA's (mma.sync m16n8k32 s8 . s8 into s32, sm_80 on): a work-group of 8 warps a tile of 128 x 128 of Y, a warp
// 64 x 32 of it (4 x 4 tiles of 16 x 8), K 64 at a time.  The step's rows of X and W go into local memory with cp.async
// (two buffers: the next step's copies run while this one is multiplied), rows padded 16 bytes so that ldmatrix's eight
// rows fall in distinct banks; the operands come out with ldmatrix (X's as A: four 8 x 16-byte tiles, W's rows as B's
// columns).  The sums leave scaled, as BF16, two columns a store.
#include "strata/kernels/i8_gemm.hpp"

#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "device_target.hpp"
#include "nv_mma.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>

#if STRATA_CUTLASS
#include "i8_gemm_cutlass.hpp"
#endif
#if STRATA_CUTLASS || STRATA_HIP_KITCHEN
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif
#if STRATA_ONEDNN_I8
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

// On the RTX 4070 (2026-11-06) the best of the shapes tried: 4115 x 3072 x 3072 in 0.595 ms against cuBLAS's int8
// product's 0.407 (int32 sums out), 4115 x 16384 x 3072 in 3.09 against 1.99 - 3.4 instructions an mma, the tensor
// pipe busy 55% of the time; 128 x 128 tiles, three or four buffers and K 128 a step were no faster.  So no product
// takes it yet (i8_gemm_ok): it waits until it is as fast as the library's
constexpr int BM = 256, BN = 128, BK = 64, NW = 8, LD = BK + 16, NS = 2;
[[maybe_unused]] constexpr int WM = 64, WN = 64;   // a warp's tile (the NVIDIA kernel's)
constexpr int NT = NW * 32;

struct NvI8Gemm {
    const int8_t *x, *w;
    const float *sx, *sw, *bias;
    uint16_t* y;
    int64_t T, N, K, ldy;
    sycl::local_accessor<int8_t, 1> sm;   // NS steps' X [BM][LD] and W [BN][LD]

    [[sycl::reqd_work_group_size(1, NT), sycl::reqd_sub_group_size(STRATA_SUB_GROUP(32))]]
    void operator()(sycl::nd_item<2> it) const {
#if STRATA_NV_MMA
        namespace nv = xe::nv;
        const int lid = (int) it.get_local_linear_id(), lane = lid & 31, wave = lid >> 5;
        const int wm = wave / (BN / WN), wn = wave % (BN / WN);   // the warp's tile: 2 x 4 of them
        // the work-groups in turn take X's tiles first: one tile of W read by the neighbours that run together (W,
        // 16384 x 3072 for MOSS-Image's MLP, is past the L2 cache; X is not)
        const int64_t n0 = (int64_t) it.get_group(0) * BN, m0 = (int64_t) it.get_group(1) * BM;
        int8_t* S = &sm[0];
        const uint32_t s0 = nv::smem_addr(S);
        constexpr uint32_t XS = BM * LD, STEP = (BM + BN) * LD;   // W's start in a step, a step's bytes
        // a step's copies: 16 bytes a piece, (BM + BN) * BK / 16 of them, NP a work-item - their sources (at k 0) and
        // places in a buffer worked out once, a step adding its k and buffer to them
        constexpr int NP = (BM + BN) * BK / 16 / NT;
        static_assert(NP * NT == (BM + BN) * BK / 16, "whole pieces a work-item");
        const int8_t* src[NP];
        uint32_t dst[NP];
        uint32_t ok = 0;   // the pieces whose rows are there (the others zero)
#pragma unroll
        for (int p = 0; p < NP; ++p) {
            const int c = p * NT + lid, row = c / (BK / 16), col = c % (BK / 16) * 16;
            const bool isx = row < BM;
            const int r = isx ? row : row - BM;
            const int64_t g = (isx ? m0 : n0) + r;
            const bool in = g < (isx ? T : N);
            src[p] = (isx ? x : w) + (in ? g : 0) * K + col;
            dst[p] = s0 + (uint32_t) ((isx ? 0 : XS) + r * LD + col);
            ok |= (uint32_t) in << p;
        }
        auto issue = [&](int64_t k0, int bf) {
            const uint32_t boff = (uint32_t) bf * STEP;
#pragma unroll
            for (int p = 0; p < NP; ++p)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst[p] + boff), "l"(src[p] + k0),
                             "r"((ok >> p & 1u) ? 16 : 0));
            nv::cp_async_commit();
        };
        // ldmatrix's row addresses in a buffer: A's rows lane % 16, bytes (lane / 16) * 16; B's two tiles of 8
        // columns a load, matrix lane / 8 its (n half, k half)
        uint32_t aoff[WM / 16], boffs[WN / 16];
#pragma unroll
        for (int i = 0; i < WM / 16; ++i) aoff[i] = (uint32_t) ((wm * WM + i * 16 + (lane & 15)) * LD + (lane >> 4) * 16);
#pragma unroll
        for (int j = 0; j < WN / 16; ++j)
            boffs[j] = XS + (uint32_t) ((wn * WN + j * 16 + ((lane >> 3) >> 1) * 8 + (lane & 7)) * LD + ((lane >> 3) & 1) * 16);
        int acc[WM / 16][WN / 8][4];
#pragma unroll
        for (int i = 0; i < WM / 16; ++i)
#pragma unroll
            for (int j = 0; j < WN / 8; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;
        const int steps = (int) (K / BK);
        // NS buffers in a ring: NS - 1 steps' copies out ahead (a group committed each step, empty past the last)
#pragma unroll
        for (int p = 0; p < NS - 1; ++p) {
            if (p < steps) issue((int64_t) p * BK, p);
            else nv::cp_async_commit();
        }
        int bf = 0, nb = NS - 1;   // this step's buffer, the next copy's
        for (int s = 0; s < steps; ++s) {
            nv::cp_async_wait<NS - 2>();   // this step's copies landed
            sycl::group_barrier(it.get_group());   // and every warp left the step before (its buffer is next's)
            if (s + NS - 1 < steps) issue((int64_t) (s + NS - 1) * BK, nb);
            else nv::cp_async_commit();
            nb = nb + 1 == NS ? 0 : nb + 1;
            const uint32_t base = s0 + (uint32_t) bf * STEP;
            bf = bf + 1 == NS ? 0 : bf + 1;
            // the k32 slices' operands a slice ahead of their products (the loads are volatile: written in this order,
            // they overlap the products only so)
            uint32_t a[2][WM / 16][4], b[2][WN / 8][2];
            auto load = [&](int ks, int u) {
#pragma unroll
                for (int i = 0; i < WM / 16; ++i) nv::ldsm4(a[u][i], base + aoff[i] + ks * 32);
#pragma unroll
                for (int j = 0; j < WN / 16; ++j) {
                    uint32_t r[4];
                    nv::ldsm4(r, base + boffs[j] + ks * 32);
                    const int j2 = 2 * j;
                    b[u][j2][0] = r[0];
                    b[u][j2][1] = r[1];
                    b[u][j2 + 1][0] = r[2];
                    b[u][j2 + 1][1] = r[3];
                }
            };
            load(0, 0);
#pragma unroll
            for (int ks = 0; ks < BK / 32; ++ks) {
                if (ks + 1 < BK / 32) load(ks + 1, (ks + 1) & 1);
#pragma unroll
                for (int i = 0; i < WM / 16; ++i)
#pragma unroll
                    for (int j = 0; j < WN / 8; ++j) nv::mma_s8(acc[i][j], a[ks & 1][i], b[ks & 1][j][0], b[ks & 1][j][1]);
            }
        }
        // out: rows g and g + 8 of a tile, columns 2 (lane % 4) and the next
        const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
        for (int i = 0; i < WM / 16; ++i)
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int64_t row = m0 + (int64_t) (wm * WM + i * 16 + g + hr * 8);
                if (row >= T) continue;
                const float rs = sx[row];
                const int e = 2 * hr;   // the accumulators' pair for the row
#pragma unroll
                for (int j = 0; j < WN / 8; ++j) {
                    const int64_t col = n0 + (int64_t) (wn * WN + j * 8 + t4 * 2);
                    if (col >= N) continue;
                    const float v0 = (float) acc[i][j][e] * rs * sw[col] + (bias ? bias[col] : 0.f);
                    if (col + 1 < N) {
                        const float v1 = (float) acc[i][j][e + 1] * rs * sw[col + 1] + (bias ? bias[col + 1] : 0.f);
                        const uint32_t pk = (uint32_t) bf16_from_f32(v0) | (uint32_t) bf16_from_f32(v1) << 16;
                        *reinterpret_cast<uint32_t*>(y + row * ldy + col) = pk;
                    } else {
                        y[row * ldy + col] = bf16_from_f32(v0);
                    }
                }
            }
#else
        (void) it;
#endif
    }
};

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
        std::fprintf(stderr, "xestrata: comfy-kitchen's int8 product module did not open (%s): the library's product\n",
                     why.c_str());
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
            if (b.p != nullptr) sycl::free(b.p, q);
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
    std::map<const sycl::queue*, dnnl::stream> streams;
};

std::mutex g_dnnl_mu;
std::unordered_map<sycl::device, Dnnl> g_dnnl;

// Whether oneDNN's GPU engine takes q's device (made once a device).
bool onednn_ok(sycl::queue& q) {
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
        auto st = d.streams.find(&q);
        if (st == d.streams.end()) st = d.streams.emplace(&q, sycl_interop::make_stream(*d.eng, q)).first;
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
        it->second.execute(st->second, args);
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
        auto st = d.streams.find(&q);
        if (st == d.streams.end()) st = d.streams.emplace(&q, sycl_interop::make_stream(*d.eng, q)).first;
        const auto usm = sycl_interop::memory_kind::usm;
        auto mem = [&](const memory::desc& md, const void* p) {
            return sycl_interop::make_memory(md, *d.eng, usm, const_cast<void*>(p));
        };
        it->second.execute(st->second, {{DNNL_ARG_SRC, mem(a_md, X)}, {DNNL_ARG_WEIGHTS, mem(w_md, W)},
                                        {DNNL_ARG_DST, mem(y_md, Y)}});
        return true;
    } catch (const std::exception&) {
        d.failed = true;
        return false;
    }
}
#endif

// Whether q's GPU takes the engine's own kernel (NvI8Gemm).
bool own_ok(sycl::queue& q) {
    if (true) return false;   // not taken yet: see the shapes' note above
    static core::PerDevice<int> of;
    return of.get(q.get_device(), [&q] {
        const sycl::device d = q.get_device();
        if (d.get_backend() != sycl::backend::ext_oneapi_cuda) return 0;
        int* f = sycl::malloc_device<int>(1, q);   // the binary the runtime picked for the GPU carries the kernel
        if (f == nullptr) return 0;
        int v = 0;
        q.single_task([=] { *f = STRATA_NV_MMA; }).wait();
        q.memcpy(&v, f, sizeof v).wait();
        sycl::free(f, q);
        return v;
    }) == 1;
}

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
    return own_ok(q);
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
    if (K % BK != 0 || !own_ok(q)) return false;
    q.submit([&](sycl::handler& h) {
        NvI8Gemm k{X, W, sx, sw, bias, Y, T, N, K, ldy,
                   sycl::local_accessor<int8_t, 1>(sycl::range<1>((size_t) (NS * (BM + BN) * LD)), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) ((N + BN - 1) / BN), (size_t) ((T + BM - 1) / BM) * NT},
                                         {1, (size_t) NT}),
                       k);
    });
    if (!stream) q.wait();
    return true;
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
