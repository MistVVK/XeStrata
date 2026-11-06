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
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
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

bool i8_gemm_ok(void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
#if STRATA_CUTLASS
    if (cutlass_of(q).any()) return true;
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

}  // namespace strata::kernels
