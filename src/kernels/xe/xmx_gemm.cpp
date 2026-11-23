// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/xmx_gemm.cpp - see include/strata/kernels/xmx_gemm.hpp.
//
// The scheme of intel/llvm's joint_matrix performance tests (joint_matrix_bf16_fill_k_cache): a sub-group computes a
// 32 x 64 tile of Y as 4 x 4 accumulators of 8 x 16, K advances 32 at a time.  The work-groups are numbered in strips
// of GM row tiles, so the ones running at once share both their A and their B tiles in the cache: without the strips
// the wide products (N 10240) ran at 105 TFLOP/s, with them at 140-150, oneMKL at 158.  W is the B operand read
// column-major straight from its rows, as fast as the VNNI-packed layout the XMX engines take natively (143-153
// TFLOP/s against 142-147 for the wide products), while the dequantizers write row-major twice as fast as packed (the
// 2026-10-02-xmx-gemm record).
// The tests' prefetch of the next A and B tiles (joint_matrix_prefetch) is left out: it gained nothing (B70, Gyro-M
// 32K prefill 2,137 tok/s with it and without, IGC 2.41.9, 2026-10-22), and since intel/llvm v7.2.0 IGC 2.28 does
// not build a kernel with it ("Failed to resolve matrix prefetch pointer type").
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/per_device.hpp"
#include "dp4a_gemm.hpp"
#include "mma_gemm.hpp"
#include "amd_wmma.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "device_target.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_xmx_gemm
#include "strata/kernels/device_arch_tag.hpp"
#include "device_caps.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace ix = sycl::ext::intel::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using bf16_t = sycl::ext::oneapi::bfloat16;

constexpr int TM = 8, TN = 16, TK = 16;     // one XMX tile (FP16 and BF16)
constexpr int SGM = 4, SGN = 4;             // accumulators a sub-group
constexpr int MS = TM * SGM, NS = TN * SGN; // a sub-group's tile of Y: 32 x 64
[[maybe_unused]] constexpr int KC = 32;     // K a step (unused where the device compile leaves out tile())

template<typename E>
inline auto global(const E* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
        const_cast<E*>(p));
}

#if !STRATA_DEVICE_NOT_INTEL
// One sub-group's tile at (m0, n0) of a WM x WN work-group tile at (wm0, wn0).  CM / CN: the tile may cross the last
// row / column, so its loads and stores are the bounds-checked ones (the others read past nothing).  ACC: Y += the
// product (a template argument: as a run-time flag the default products ran 6 times slower).
template<typename E, int WM, int WN, bool CM, bool CN, bool ACC>
inline void tile(sycl::sub_group sg, int sgid, const E* X, const E* W, float* Y, int64_t T, int64_t N, int64_t K,
                 int64_t ldy, int64_t wm0, int64_t wn0) {
    constexpr int NSG_N = WN / NS;
    const int64_t m0 = wm0 + (int64_t) (sgid / NSG_N) * MS, n0 = wn0 + (int64_t) (sgid % NSG_N) * NS;
    // A column-major B tile past the last row of W is read from the last 16 rows instead (its columns of Y are not
    // stored): the bounds-checked column-major load lost the device.
    int64_t bn[SGN];
    for (int j = 0; j < SGN; ++j) bn[j] = CN ? sycl::min(n0 + (int64_t) j * TN, N - TN) : n0 + (int64_t) j * TN;
    const auto gx = global(X), gw = global(W);
    const auto gy = global(Y);
    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c[SGM][SGN];
    for (int i = 0; i < SGM; ++i)
        for (int j = 0; j < SGN; ++j) {
            if constexpr (!ACC) mx::joint_matrix_fill(sg, c[i][j], 0.0f);
            else if constexpr (CM || CN)
                ix::joint_matrix_load_checked(sg, c[i][j], gy, ldy, mx::layout::row_major, T, N, m0 + (int64_t) i * TM,
                                              n0 + (int64_t) j * TN);
            else
                mx::joint_matrix_load(sg, c[i][j], gy + (m0 + (int64_t) i * TM) * ldy + n0 + (int64_t) j * TN, ldy,
                                      mx::layout::row_major);
        }
    for (int64_t k0 = 0; k0 < K; k0 += KC) {
        for (int kk = 0; kk < KC; kk += TK) {
            const int64_t k = k0 + kk;
            mx::joint_matrix<sycl::sub_group, E, mx::use::a, TM, TK, mx::layout::row_major> a[SGM];
            mx::joint_matrix<sycl::sub_group, E, mx::use::b, TK, TN, mx::layout::col_major> b[SGN];
            for (int i = 0; i < SGM; ++i) {
                if constexpr (CM) ix::joint_matrix_load_checked(sg, a[i], gx, K, T, K, m0 + (int64_t) i * TM, k);
                else mx::joint_matrix_load(sg, a[i], gx + (m0 + (int64_t) i * TM) * K + k, K);
            }
            for (int j = 0; j < SGN; ++j) mx::joint_matrix_load(sg, b[j], gw + bn[j] * K + k, K);
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) mx::joint_matrix_mad(sg, c[i][j], a[i], b[j], c[i][j]);
        }
    }
    for (int i = 0; i < SGM; ++i)
        for (int j = 0; j < SGN; ++j) {
            if constexpr (CM || CN)
                ix::joint_matrix_store_checked(sg, c[i][j], gy, ldy, mx::layout::row_major, T, N, m0 + (int64_t) i * TM,
                                               n0 + (int64_t) j * TN);
            else
                mx::joint_matrix_store(sg, c[i][j], gy + (m0 + (int64_t) i * TM) * ldy + n0 + (int64_t) j * TN, ldy,
                                       mx::layout::row_major);
        }
}

template<typename E, int WM, int WN, bool ACC = false>
inline void tile_any(sycl::sub_group sg, int sgid, const E* X, const E* W, float* Y, int64_t T, int64_t N, int64_t K,
                     int64_t ldy, int64_t wm0, int64_t wn0) {
    if (wm0 + WM <= T && wn0 + WN <= N) tile<E, WM, WN, false, false, ACC>(sg, sgid, X, W, Y, T, N, K, ldy, wm0, wn0);
    else tile<E, WM, WN, true, true, ACC>(sg, sgid, X, W, Y, T, N, K, ldy, wm0, wn0);
}
#endif

[[maybe_unused]] constexpr int GM = 2;   // row tiles a strip (unused in the device compiles for other GPUs)

// The kernels are functors whose properties ask for the large register file (the 16 accumulators and their operands
// spill with 128 registers) and 16-wide sub-groups.  Their arguments are their own members: a functor wrapping the
// kernel lambda passed the captures as one by-value struct, and the same code ran 8-30% slower.
#define STRATA_XMX_PROPERTIES                                                                                          \
    auto get(syclex::properties_tag) const {                                                                           \
        return syclex::properties{sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<16>};         \
    }

template<typename E, int WM, int WN, bool ACC = false>
struct GemmKernel {
    const E* X;
    const E* W;
    float* Y;
    int64_t T, N, K, ldy, nM, nN;
    void operator()(sycl::nd_item<1> it) const {
#if STRATA_DEVICE_NOT_INTEL
        (void) it;
#else
        const auto sg = it.get_sub_group();
        const int64_t g = (int64_t) it.get_group(0);
        const int64_t strip = g / (GM * nN), r = g % (GM * nN);
        const int64_t gm = std::min<int64_t>(GM, nM - strip * GM);
        const int64_t tm = strip * GM + r % gm, tn = r / gm;
        tile_any<E, WM, WN, ACC>(sg, (int) sg.get_group_linear_id(), X, W, Y, T, N, K, ldy, tm * WM, tn * WN);
#endif
    }
    STRATA_XMX_PROPERTIES
};

template<int WM, int WN>
struct GroupedKernel {
    const sycl::half* X;
    const sycl::half* W;
    float* Y;
    const int32_t* bounds;
    int64_t w_stride, tiles_m, nN, N, K;
    void operator()(sycl::nd_item<1> it) const {
#if STRATA_DEVICE_NOT_INTEL
        (void) it;
#else
        const auto sg = it.get_sub_group();
        const int64_t g = (int64_t) it.get_group(0);
        const int64_t ex = g / (tiles_m * nN), r = g % (tiles_m * nN), tm = r / nN, tn = r % nN;
        const int64_t row0 = bounds[ex], rows = bounds[ex + 1] - row0;
        if (tm * WM >= rows) return;   // the expert has fewer rows than the launch allows for
        tile_any<sycl::half, WM, WN>(sg, (int) sg.get_group_linear_id(), X + row0 * K, W + ex * w_stride,
                                     Y + row0 * N, rows, N, K, N, tm * WM, tn * WN);
#endif
    }
    STRATA_XMX_PROPERTIES
};
#undef STRATA_XMX_PROPERTIES

template<typename E, int WM, int WN, bool ACC = false>
sycl::event launch(sycl::queue& q, const E* X, const E* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
    constexpr int NSG = (WM / MS) * (WN / NS);
    const int64_t nM = (T + WM - 1) / WM, nN = (N + WN - 1) / WN;
    return q.parallel_for(sycl::nd_range<1>((size_t) (nM * nN * NSG * 16), (size_t) NSG * 16),
                          GemmKernel<E, WM, WN, ACC>{X, W, Y, T, N, K, ldy, nM, nN});
}

// The work-group tile, from the timings of the prompt path's shapes at 512, 1716 and 8192 rows on the B70 (256
// compute units): 32 sub-groups of 32 x 64 fill the GPU only with enough tiles; a narrow product takes a narrow tile
// (or, with a long K, one that covers all its columns, so X is read once).  The row thresholds are how many tiles
// fill the B70, so they scale with the device's compute units.
template<typename E>
sycl::event dispatch(sycl::queue& q, const E* X, const E* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
    const int64_t many = xe::fill_rows(q, 4096), few = xe::fill_rows(q, 1024);
    if (N <= 64) return launch<E, 256, 64>(q, X, W, Y, T, N, K, ldy);
    if (T >= many) {
        if (N <= 128) return launch<E, 256, 128>(q, X, W, Y, T, N, K, ldy);
        if (N <= 384 && K >= 4096) return launch<E, 128, 512>(q, X, W, Y, T, N, K, ldy);
        return launch<E, 256, 256>(q, X, W, Y, T, N, K, ldy);
    }
    if (N <= 512 || (T < few && N < 2048)) return launch<E, 128, 256>(q, X, W, Y, T, N, K, ldy);
    return launch<E, 256, 256>(q, X, W, Y, T, N, K, ldy);
}

// Whether the device runs these kernels: a joint_matrix combination of 8 x 16 x 16 for the input type with FP32
// accumulators, and 16-wide sub-groups, as the device (and the SYCL runtime) reports them.  Otherwise the products go
// to dp4a_gemm.  STRATA_NO_XMX=1 forces that path, to compare the two.
bool device_has_xmx(const sycl::device& d, mx::matrix_type in) {
    if (no_xmx()) return false;
    if (!xe::has_sub_group(d, 16)) return false;
    for (const auto& c : matrix_combinations(d)) {
        const bool m = c.msize == TM || (c.msize == 0 && c.max_msize >= TM);
        const bool n = c.nsize == TN || (c.nsize == 0 && c.max_nsize >= TN);
        const bool k = c.ksize == TK || (c.ksize == 0 && c.max_ksize >= TK);
        if (c.atype == in && c.btype == in && c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 &&
            m && n && k)
            return true;
    }
    return false;
}

// Whether the device builds the kernels' large register file (grf_size<256>): XMX devices that list the matrix
// combinations may still lack the mode, and a kernel that asks for it then fails to build at its first launch.  One
// kernel is built here, once, before any is launched.
bool large_grf_ok(const sycl::device& d) {
    try {
        const auto ctx = core::Runtime::get().compute().get_context();
        (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            ctx, {d}, {sycl::get_kernel_id<GemmKernel<sycl::half, 128, 256>>()});
        return true;
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xestrata: the GPU does not build kernels with the large register file (%s): XeStrata's own "
                     "matrix-product kernels take their DP4a form\n", e.what());
        return false;
    }
}

// the current device's XMX, worked out and reported once for each GPU
struct XmxUse {
    bool f16, bf16;
};
bool use_xmx(XmxType t) {
    static core::PerDevice<XmxUse> per_device;
    auto& q = core::Runtime::get().compute();
    const XmxUse& u = per_device.get(q.get_device(), [&q] {
        const sycl::device dev = q.get_device();
        const bool grf = (device_has_xmx(dev, mx::matrix_type::fp16) || device_has_xmx(dev, mx::matrix_type::bf16))
                         && large_grf_ok(dev);
        // (the path the products take is reported once, from the prompt path's Gemm::path: oneMath may run them)
        return XmxUse{grf && device_has_xmx(dev, mx::matrix_type::fp16), grf && device_has_xmx(dev, mx::matrix_type::bf16)};
    });
    return t == XmxType::f16 ? u.f16 : u.bf16;
}

void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

}  // namespace

bool xmx_available(XmxType t) { return use_xmx(t); }

bool f16_matrix_engines() {
    return use_xmx(XmxType::f16) || xe::mma_usable(core::Runtime::get().compute(), false);
}

bool f16_only_matrix_engines() {
    static core::PerDevice<bool> per_device;
    auto& q = core::Runtime::get().compute();
    return per_device.get(q.get_device(), [&q] {
        return (use_xmx(XmxType::f16) || xe::mma_usable(q, false)) && !use_xmx(XmxType::bf16) && !xe::mma_usable(q, true);
    });
}

bool mma_products(XmxType t, bool kn) {
    auto& q = core::Runtime::get().compute();
    const bool bf16 = t == XmxType::bf16;
    if (kn && (bf16 || !xe::mma_gemm_kn_usable(q))) return false;
    return xe::mma_usable(q, bf16) && (xe::mma_forced() || !use_xmx(t));
}

bool gemm_kn(const uint16_t* X, const uint16_t* W, int64_t ldw, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
             void* stream) {
    if (T <= 0 || N <= 0) return true;
    const auto al = [](const void* p) { return ((uintptr_t) p & 63) == 0; };
    if (N % 16 != 0 || K % 32 != 0 || ldw < N || ldw % 16 != 0 || ldy < N || ldy % 16 != 0 || !al(X) || !al(W) ||
        !al(Y))
        return false;
    auto& q = core::Runtime::get().stream(stream);
    if (!xe::mma_gemm_kn_usable(q)) return false;
    finish(stream, xe::mma_gemm_kn(q, X, W, ldw, Y, T, N, K, ldy), "gemm_kn");
    return true;
}

const char* gemm_path(XmxType t) {
    sycl::queue& q = core::Runtime::get().compute();
    if (xe::mma_usable(q, t == XmxType::bf16) && (xe::mma_forced() || !use_xmx(t)))
        return xe::amd_wmma(q.get_device()) ? "WMMA" : "joint_matrix";   // RDNA3 and RDNA4: the instructions, written out
    return use_xmx(t) ? "XMX" : "DP4a";
}

bool xmx_gemm_ok(const void* X, const void* W, const float* Y, int64_t N, int64_t K, int64_t ldy) noexcept {
    const auto al = [](const void* p) { return ((uintptr_t) p & 63) == 0; };
    return N >= 16 && N % 16 == 0 && K >= 32 && K % 32 == 0 && ldy >= N && ldy % 16 == 0 && al(X) && al(W) && al(Y);
}

void xmx_gemm(XmxType t, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
              void* stream, bool accumulate) {
    if (T <= 0 || N <= 0) return;
    if (!xmx_gemm_ok(X, W, Y, N, K, ldy))
        throw core::DeviceError("xmx_gemm: shape " + std::to_string(N) + " x " + std::to_string(K) + " not supported");
    auto& q = core::Runtime::get().stream(stream);
    if (const bool bf16 = t == XmxType::bf16; xe::mma_usable(q, bf16) && (xe::mma_forced() || !use_xmx(t))) {
        finish(stream, xe::mma_gemm(q, bf16, X, W, Y, T, N, K, ldy, accumulate), "xmx_gemm");
        return;
    }
    if (!use_xmx(t)) {
        // the remainder products go to gemm_rows: an accumulating DP4a kernel beside the others moved the last bits
        // of the default products (icpx's fast floating-point model)
        if (accumulate) gemm_rows(t, X, W, Y, T, N, K, ldy, stream, true);
        else finish(stream, xe::dp4a_gemm(q, t == XmxType::bf16, X, W, Y, T, N, K, ldy), "xmx_gemm");
        return;
    }
    if (accumulate) {
        // the opt-in remainder products (STRATA_PREFILL_BF16X2) and the image engine's adapters (a LoRA's delta into
        // a product's output): one tile shape; the accumulator is FP32 for both input types, so only the operands'
        // type differs
        if (t == XmxType::bf16)
            finish(stream, launch<bf16_t, 128, 256, true>(q, reinterpret_cast<const bf16_t*>(X),
                                                          reinterpret_cast<const bf16_t*>(W), Y, T, N, K, ldy),
                   "xmx_gemm");
        else
            finish(stream, launch<sycl::half, 128, 256, true>(q, reinterpret_cast<const sycl::half*>(X),
                                                              reinterpret_cast<const sycl::half*>(W), Y, T, N, K,
                                                              ldy),
                   "xmx_gemm");
        return;
    }
    const sycl::event e = t == XmxType::f16
        ? dispatch(q, reinterpret_cast<const sycl::half*>(X), reinterpret_cast<const sycl::half*>(W), Y, T, N, K, ldy)
        : dispatch(q, reinterpret_cast<const bf16_t*>(X), reinterpret_cast<const bf16_t*>(W), Y, T, N, K, ldy);
    finish(stream, e, "xmx_gemm");
}

void xmx_gemm_grouped(const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y, const int32_t* bounds, int G,
                      int64_t max_rows, int64_t N, int64_t K, void* stream) {
    if (G <= 0 || max_rows <= 0) return;
    if (!xmx_gemm_ok(X, W, Y, N, K, N) || (w_stride * 2) % 64 != 0)
        throw core::DeviceError("xmx_gemm_grouped: shape " + std::to_string(N) + " x " + std::to_string(K));
    // 128-row work-group tiles: the best of 32, 64 and 128 for 16 experts of 20-420 rows (56 and 55 TFLOP/s for the
    // gate/up and down shapes against oneMKL's 45 and 44, one call per expert)
    if (auto& q = core::Runtime::get().stream(stream); xe::mma_usable(q, false) &&
                                                        (xe::mma_forced() || !use_xmx(XmxType::f16))) {
        finish(stream, xe::mma_gemm_grouped(q, X, W, w_stride, Y, bounds, G, max_rows, N, K), "xmx_gemm_grouped");
        return;
    }
    if (!use_xmx(XmxType::f16)) {
        finish(stream, xe::dp4a_gemm_grouped(core::Runtime::get().stream(stream), X, W, w_stride, Y, bounds, G,
                                             max_rows, N, K), "xmx_gemm_grouped");
        return;
    }
    constexpr int WM = 128, WN = 256, NSG = (WM / MS) * (WN / NS);
    const int64_t tiles_m = (max_rows + WM - 1) / WM, nN = (N + WN - 1) / WN;
    const sycl::event e = core::Runtime::get().stream(stream).parallel_for(
        sycl::nd_range<1>((size_t) (G * tiles_m * nN * NSG * 16), (size_t) NSG * 16),
        GroupedKernel<WM, WN>{reinterpret_cast<const sycl::half*>(X), reinterpret_cast<const sycl::half*>(W), Y,
                              bounds, w_stride, tiles_m, nN, N, K});
    finish(stream, e, "xmx_gemm_grouped");
}

namespace {
// gemm_rows on sub-groups of SG lanes: a sub-group a row of X and up to 16 outputs, so X is read once (one output a
// sub-group read it N times: the hyper-connection's inject, N 4 K 10240, took 1.7 ms a chunk of 8192 tokens); 8 values
// a load when K allows
template<int SG>
sycl::event rows_kernel(sycl::queue& q, bool half, const uint16_t* X, const uint16_t* W, float* Y, int64_t T,
                        int64_t N, int64_t K, int64_t ldy, bool accumulate) {
    constexpr int NB = 16;
    const bool vec = K % 8 == 0 && ((uintptr_t) X | (uintptr_t) W) % 16 == 0;
    const int64_t nb = (N + NB - 1) / NB;
    // the rows in the last dimension: a CUDA grid's second dimension is capped at 65535 blocks, which the rows of a
    // large product (a VAE convolution's 262144 pixels) pass; the output blocks (N / 16) stay far below it
    return q.parallel_for(
        sycl::nd_range<2>({(size_t) nb, (size_t) T * SG}, {1, SG}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]] {
            const auto sg = it.get_sub_group();
            const int64_t r = (int64_t) it.get_group(1), n0 = (int64_t) it.get_group(0) * NB;
            const int lane = (int) sg.get_local_linear_id();
            const auto val = [half](uint16_t b) {
                return half ? (float) sycl::bit_cast<sycl::half>(b) : sycl::bit_cast<float>((uint32_t) b << 16);
            };
            float acc[NB] = {};
            const uint16_t* x = X + r * K;
            if (vec) {
                using V = sycl::vec<uint16_t, 8>;
                for (int64_t k = (int64_t) lane * 8; k < K; k += SG * 8ll) {
                    const V xv = *reinterpret_cast<const V*>(x + k);
                    for (int n = 0; n < NB; ++n) {
                        if (n0 + n >= N) break;
                        const V wv = *reinterpret_cast<const V*>(W + (n0 + n) * K + k);
                        for (int j = 0; j < 8; ++j) acc[n] = sycl::fma(val(xv[j]), val(wv[j]), acc[n]);
                    }
                }
            } else {
                for (int64_t k = lane; k < K; k += SG)
                    for (int n = 0; n < NB; ++n) {
                        if (n0 + n >= N) break;
                        acc[n] = sycl::fma(val(x[k]), val(W[(n0 + n) * K + k]), acc[n]);
                    }
            }
            for (int n = 0; n < NB; ++n) {
                if (n0 + n >= N) break;
                const float s = sycl::reduce_over_group(sg, acc[n], sycl::plus<float>());
                if (lane == 0) Y[r * ldy + n0 + n] = accumulate ? Y[r * ldy + n0 + n] + s : s;
            }
        });
}
}  // namespace

void gemm_rows(XmxType t, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               void* stream, bool accumulate) {
    if (T <= 0 || N <= 0) return;
    auto& q = core::Runtime::get().stream(stream);
    const bool half = t == XmxType::f16;
    const sycl::event e = xe::narrow_sub_group(q) == 16 ? rows_kernel<16>(q, half, X, W, Y, T, N, K, ldy, accumulate)
                                                        : rows_kernel<32>(q, half, X, W, Y, T, N, K, ldy, accumulate);
    finish(stream, e, "gemm_rows");
}

}  // namespace strata::kernels
