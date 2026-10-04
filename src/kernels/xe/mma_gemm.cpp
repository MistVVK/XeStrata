// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/mma_gemm.cpp - see mma_gemm.hpp.
//
// A work-group of sub-groups computes a tile of Y, each sub-group 2 x 2 matrix tiles of it.  K advances 32 values at a
// time: the work-group copies the step's rows of X and of W into local memory, 16 bytes a work-item, zero past the
// last row (so any row count works, the experts' groups too), into one of two buffers while the sub-groups multiply
// out of the other.  W goes in as B in the layout the matrix engines take: Intel's VNNI-packed one (pairs of K values
// together) on Intel GPUs, column-major on NVIDIA's.  The accumulators go out through local memory, row by row, so
// the rows past the last one are not written.
//
// On the A380 (Xe-HPG; bench/results/2026-10-04-dg2-dp4a) these steps took a 4096 x 2560 x 2560 FP16 product from
// 116 ms (W read by every sub-group, column-major, from local memory) to 52 (from global memory, as before), 29 (VNNI-
// packed in local memory), 21 (its neighbouring lanes storing side by side) and 20 (two buffers); DP4a takes 20.
#include "mma_gemm.hpp"
#include "device_target.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iterator>

namespace strata::kernels::xe {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using bf16_t = sycl::ext::oneapi::bfloat16;

// A joint_matrix tile shape and the sub-group size it is built for: Intel Xe2's 8 x 16 x 16 on 16 lanes (to check this
// code there; xmx_gemm.cpp's kernels are the fast ones on Xe2), Intel Xe-HPG's 8 x 8 x 16 on 8 (the Arc A series, whose
// runtime refuses xmx_gemm's prefetches and checked loads) and NVIDIA's 16 x 16 x 16 on 32 (a warp).  The work-group:
// wgm x wgn sub-groups for a product, wgm_g x wgn for the experts' groups (a group of 160 rows wastes less of a short
// tile); packed: B in Intel's VNNI layout.  Xe-HPG's from the A380's timings (8 x 8: 20 ms for the product above,
// 4 x 8: 20, 2 x 8: 23; for 16 experts of 160 rows, 1280 x 2560: 9.2, 7.5 and 6.9 ms), the others untuned.
struct Shape {
    int m, n, k, sg, wgm, wgm_g, wgn;
    bool packed;
};
constexpr Shape kShapes[] = {{8, 16, 16, 16, 2, 2, 2, true}, {8, 8, 16, 8, 8, 2, 8, true},
                             {16, 16, 16, 32, 2, 2, 2, false}};

constexpr int SGM = 2, SGN = 2;   // tiles a sub-group (4 x 4 spilled on the A380: 86 ms for the product above)
constexpr int KC = 32;            // K a step (64 was no faster on the A380)
constexpr int PAD = 8;            // column-major B's row stride is KC + PAD (16-byte aligned rows)
constexpr int VEC = 8;            // values a work-item copies at once (16 bytes)

// Whether this device compile carries shape S's kernels: Intel's shapes only Intel's, NVIDIA's only the others (an
// AMD GPU has joint_matrix on CDNA's matrix cores only, which these shapes are not).
template<int S>
constexpr bool built() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
    return false;
#else
    return kShapes[S].sg < 32 ? !STRATA_DEVICE_NOT_INTEL : true;
#endif
}

// Rows row0 .. row0 + R of A (row stride K, `n` rows in all) from column k0, KC values each, into local memory at
// `dst` + off, zero past the last row; the work-group's lanes `lid` of `lanes`, 16 bytes at a time.  X and W rows
// start 64-byte aligned (xmx_gemm_ok) and K is a multiple of 32, so every load is aligned.  PACKED: as Intel's
// VNNI-packed B (row k/2 holds the pairs of values k, k+1 of every row, R pairs), and neighbouring lanes take
// neighbouring rows so that their stores land side by side; else row by row (row stride LD).
template<typename E, int R, int LD, bool PACKED>
inline void stage(const E* A, int64_t row0, int64_t n, int64_t K, int64_t k0, const sycl::local_accessor<E, 1>& dst,
                  int off, int lid, int lanes) {
    using V = sycl::vec<uint16_t, VEC>;
    for (int v = lid; v < R * (KC / VEC); v += lanes) {
        const int r = PACKED ? v % R : v / (KC / VEC);
        const int c = (PACKED ? v / R : v % (KC / VEC)) * VEC;
        const int64_t row = row0 + r;
        const V val = row < n ? *reinterpret_cast<const V*>(A + row * K + k0 + c) : V(0);
        for (int j = 0; j < VEC; ++j) {
            const int k = c + j;
            if constexpr (PACKED) dst[off + (k / 2) * (2 * R) + r * 2 + (k & 1)] = sycl::bit_cast<E>(val[j]);
            else dst[off + r * LD + k] = sycl::bit_cast<E>(val[j]);
        }
    }
}

// G groups of rows of X and Y (bounds[e] .. bounds[e + 1], or all T rows when bounds is null), each by its W.  A
// functor: its members are the same in the host compile and in a device compile that leaves its body out (a lambda's
// captures would not be).
template<int S, typename E, bool ACC, bool GROUPED>
struct Kernel {
    static constexpr int TM = kShapes[S].m, TN = kShapes[S].n, TK = kShapes[S].k, SG = kShapes[S].sg;
    static constexpr bool PACKED = kShapes[S].packed;
    static constexpr int NSGM = GROUPED ? kShapes[S].wgm_g : kShapes[S].wgm, NSGN = kShapes[S].wgn;
    static constexpr int NSG = NSGM * NSGN;
    static constexpr int WM = NSGM * SGM * TM, WN = NSGN * SGN * TN;
    static constexpr int LDA = KC, LDB = PACKED ? KC : KC + PAD;   // A's row stride; B's (column-major) row stride
    static constexpr int ASZ = WM * LDA, BSZ = WN * LDB;           // a buffer's elements
    const E* X;
    const E* W;
    int64_t w_stride;
    float* Y;
    int64_t ldy;
    const int32_t* bounds;
    int64_t T, N, K, tiles_m, tiles_n;
    sycl::local_accessor<E, 1> as;       // two buffers of the step's rows of X: WM x KC
    sycl::local_accessor<E, 1> bs;       // two buffers of the step's rows of W, as B: WN x KC
    sycl::local_accessor<float, 1> cs;   // a TM x TN tile of Y a sub-group, on its way out

    void operator()(sycl::nd_item<1> it) const {
        if constexpr (!built<S>()) {
            (void) it;
        } else {
            const int64_t g = (int64_t) it.get_group(0), ex = g / (tiles_m * tiles_n), r = g % (tiles_m * tiles_n);
            const int64_t wm0 = r / tiles_n * WM, wn0 = r % tiles_n * WN;
            const int64_t row0 = bounds ? bounds[ex] : 0, rows = bounds ? bounds[ex + 1] - row0 : T;
            if (wm0 >= rows) return;   // the whole work-group: this group of rows is shorter than the launch
            const E* x = X + row0 * K;
            const E* w = W + ex * w_stride;
            float* y = Y + row0 * ldy;
            const auto sg = it.get_sub_group();
            const int sgid = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int lid = (int) it.get_local_id(0);
            const int ms = sgid / NSGN * SGM * TM, ns = sgid % NSGN * SGN * TN;   // the sub-group's tile
            const auto ap = as.template get_multi_ptr<sycl::access::decorated::no>();
            const auto bp = bs.template get_multi_ptr<sycl::access::decorated::no>();
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c[SGM][SGN];
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) mx::joint_matrix_fill(sg, c[i][j], 0.0f);
            stage<E, WM, LDA, false>(x, wm0, rows, K, 0, as, 0, lid, NSG * SG);
            stage<E, WN, LDB, PACKED>(w, wn0, N, K, 0, bs, 0, lid, NSG * SG);
            sycl::group_barrier(it.get_group());
            int buf = 0;
            for (int64_t k0 = 0; k0 < K; k0 += KC) {
                if (k0 + KC < K) {
                    stage<E, WM, LDA, false>(x, wm0, rows, K, k0 + KC, as, (buf ^ 1) * ASZ, lid, NSG * SG);
                    stage<E, WN, LDB, PACKED>(w, wn0, N, K, k0 + KC, bs, (buf ^ 1) * BSZ, lid, NSG * SG);
                }
                const auto apb = ap + buf * ASZ;
                const auto bpb = bp + buf * BSZ;
                for (int kk = 0; kk < KC; kk += TK) {
                    constexpr auto BLAYOUT = PACKED ? mx::layout::ext_intel_packed : mx::layout::col_major;
                    mx::joint_matrix<sycl::sub_group, E, mx::use::a, TM, TK, mx::layout::row_major> a[SGM];
                    mx::joint_matrix<sycl::sub_group, E, mx::use::b, TK, TN, BLAYOUT> b[SGN];
                    for (int i = 0; i < SGM; ++i) mx::joint_matrix_load(sg, a[i], apb + (ms + i * TM) * LDA + kk, LDA);
                    for (int j = 0; j < SGN; ++j) {
                        if constexpr (PACKED)
                            mx::joint_matrix_load(sg, b[j], bpb + (kk / 2) * (2 * WN) + (ns + j * TN) * 2, 2 * WN);
                        else mx::joint_matrix_load(sg, b[j], bpb + (ns + j * TN) * LDB + kk, LDB);
                    }
                    for (int i = 0; i < SGM; ++i)
                        for (int j = 0; j < SGN; ++j) mx::joint_matrix_mad(sg, c[i][j], a[i], b[j], c[i][j]);
                }
                sycl::group_barrier(it.get_group());
                buf ^= 1;
            }
            const auto cp = cs.template get_multi_ptr<sycl::access::decorated::no>() + (std::ptrdiff_t) sgid * TM * TN;
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) {
                    const int64_t n = wn0 + ns + (int64_t) j * TN;
                    mx::joint_matrix_store(sg, c[i][j], cp, TN, mx::layout::row_major);
                    sycl::group_barrier(sg);
                    for (int e = lane; e < TM * TN; e += SG) {
                        const int64_t row = wm0 + ms + (int64_t) i * TM + e / TN;
                        if (row < rows && n < N) {
                            float* o = y + row * ldy + n + e % TN;
                            *o = ACC ? *o + cp[e] : cp[e];
                        }
                    }
                    sycl::group_barrier(sg);
                }
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>}; }
};

template<int S, typename E, bool ACC, bool GROUPED>
sycl::event launch(sycl::queue& q, const E* X, const E* W, int64_t w_stride, float* Y, int64_t ldy,
                   const int32_t* bounds, int G, int64_t T, int64_t N, int64_t K) {
    using KT = Kernel<S, E, ACC, GROUPED>;
    const int64_t tiles_m = (T + KT::WM - 1) / KT::WM, tiles_n = (N + KT::WN - 1) / KT::WN;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<E, 1> as(sycl::range<1>(2 * KT::ASZ), h), bs(sycl::range<1>(2 * KT::BSZ), h);
        sycl::local_accessor<float, 1> cs(sycl::range<1>(KT::NSG * KT::TM * KT::TN), h);
        const size_t wg = (size_t) KT::NSG * KT::SG;
        h.parallel_for(sycl::nd_range<1>((size_t) (G * tiles_m * tiles_n) * wg, wg),
                       KT{X, W, w_stride, Y, ldy, bounds, T, N, K, tiles_m, tiles_n, as, bs, cs});
    });
}

bool reported(const sycl::device& d, mx::matrix_type in, const Shape& s) {
    const auto sg = d.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sg.begin(), sg.end(), (size_t) s.sg) == sg.end()) return false;
    for (const auto& c : d.get_info<syclex::info::device::matrix_combinations>()) {
        const bool m = c.msize == (size_t) s.m || (c.msize == 0 && c.max_msize >= (size_t) s.m);
        const bool n = c.nsize == (size_t) s.n || (c.nsize == 0 && c.max_nsize >= (size_t) s.n);
        const bool k = c.ksize == (size_t) s.k || (c.ksize == 0 && c.max_ksize >= (size_t) s.k);
        if (c.atype == in && c.btype == in && c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 &&
            m && n && k)
            return true;
    }
    return false;
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strtol(v, nullptr, 10) != 0;
}

// The first shape the device reports for the input type, or -1.  The engine drives one device, so the first call's
// answer holds.
int shape(const sycl::queue& q, bool bf16) {
    static const auto pick = [](const sycl::device& d, mx::matrix_type in) {
        for (int s = 0; s < (int) std::size(kShapes); ++s)
            if (reported(d, in, kShapes[s])) return s;
        return -1;
    };
    static const int f16 = pick(q.get_device(), mx::matrix_type::fp16);
    static const int b16 = pick(q.get_device(), mx::matrix_type::bf16);
    return bf16 ? b16 : f16;
}

template<int S, typename E>
sycl::event run(sycl::queue& q, const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y, int64_t ldy,
                const int32_t* bounds, int G, int64_t T, int64_t N, int64_t K, bool accumulate) {
    const auto* x = reinterpret_cast<const E*>(X);
    const auto* w = reinterpret_cast<const E*>(W);
    if (bounds) return launch<S, E, false, true>(q, x, w, w_stride, Y, ldy, bounds, G, T, N, K);
    if (accumulate) return launch<S, E, true, false>(q, x, w, w_stride, Y, ldy, bounds, G, T, N, K);
    return launch<S, E, false, false>(q, x, w, w_stride, Y, ldy, bounds, G, T, N, K);
}

template<typename E>
sycl::event by_shape(int s, sycl::queue& q, const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y,
                     int64_t ldy, const int32_t* bounds, int G, int64_t T, int64_t N, int64_t K, bool accumulate) {
    if (s == 0) return run<0, E>(q, X, W, w_stride, Y, ldy, bounds, G, T, N, K, accumulate);
    if (s == 1) return run<1, E>(q, X, W, w_stride, Y, ldy, bounds, G, T, N, K, accumulate);
    return run<2, E>(q, X, W, w_stride, Y, ldy, bounds, G, T, N, K, accumulate);
}

}  // namespace

bool mma_forced() {
    static const bool v = env_on("STRATA_MMA");
    return v;
}

bool mma_usable(const sycl::queue& q, bool bf16) {
    static const bool off = env_on("STRATA_NO_XMX") && !mma_forced();
    return !off && shape(q, bf16) >= 0;
}

sycl::event mma_gemm(sycl::queue& q, bool bf16, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N,
                     int64_t K, int64_t ldy, bool accumulate) {
    const int s = shape(q, bf16);
    if (bf16) return by_shape<bf16_t>(s, q, X, W, 0, Y, ldy, nullptr, 1, T, N, K, accumulate);
    return by_shape<sycl::half>(s, q, X, W, 0, Y, ldy, nullptr, 1, T, N, K, accumulate);
}

sycl::event mma_gemm_grouped(sycl::queue& q, const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y,
                             const int32_t* bounds, int G, int64_t max_rows, int64_t N, int64_t K) {
    return by_shape<sycl::half>(shape(q, false), q, X, W, w_stride, Y, N, bounds, G, max_rows, N, K, false);
}

}  // namespace strata::kernels::xe
