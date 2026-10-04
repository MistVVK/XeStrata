// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/mma_gemm.cpp - see mma_gemm.hpp.
//
// A work-group of sub-groups computes a tile of Y, each sub-group some matrix tiles of it (Layout below).  K advances
// 32 values at a time: the work-group copies the step's rows of X and of W into local memory, 16 bytes a work-item,
// zero past the last row (so any row count works, the experts' groups too), into one of two buffers while the
// sub-groups multiply out of the other.  W goes in as B in the layout the matrix engines take: Intel's VNNI-packed one
// (pairs of K values together) on Intel GPUs, column-major on NVIDIA's.  The accumulators go out through local memory,
// row by row, so the rows past the last one are not written.
//
// On the A380 (Xe-HPG; bench/results/2026-10-04-dg2-dp4a) a 4096 x 2560 x 2560 FP16 product took 52 ms with W read
// from global memory by every sub-group; with W in local memory 116 (column-major), 33 (transposed), 23 (VNNI-packed),
// 21 (neighbouring lanes storing side by side), 20 (two buffers), 16 (4 x 4 tiles a sub-group in the large register
// file; in the default 128 registers they spilled), 14.2 (a fragment's rows contiguous) and 13.6 (16-byte stores).
// DP4a takes 20.  Loading the next step into registers while multiplying, to store it after, spilled (91-103 ms).
#include "mma_gemm.hpp"
#include "device_target.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iterator>

namespace strata::kernels::xe {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using bf16_t = sycl::ext::oneapi::bfloat16;

// A joint_matrix tile shape and the sub-group size it is built for: Intel Xe2's 8 x 16 x 16 on 16 lanes (to check this
// code there; xmx_gemm.cpp's kernels are the fast ones on Xe2), Intel Xe-HPG's 8 x 8 x 16 on 8 (the Arc A series, whose
// runtime refuses xmx_gemm's prefetches and checked loads) and NVIDIA's 16 x 16 x 16 on 32 (a warp); the work-group's
// layout for a product (dense) and for the experts' groups (grouped); packed: B in Intel's VNNI layout.  Xe-HPG's
// layouts from the A380's timings (bench/results/2026-10-04-dg2-dp4a): a product 4 x 8 sub-groups of 4 x 4 tiles in the
// large register file (13.6 ms for the product above; 8 x 4: 14.4, 4 x 4: 15.4; 2 x 2 tiles in 128 registers, 8 x 8
// sub-groups: 19 before the contiguous fragments), the groups of 160 rows 2 x 8 of 2 x 2 (6.8 ms for 16 experts of
// 1280 x 2560; none of the large-register ones was faster).  The others untuned.
// A work-group's layout: wgm x wgn sub-groups, each sgm x sgn tiles, with the large register file (grf) or not.
struct Layout {
    int wgm, wgn, sgm, sgn;
    bool grf;
};
struct Shape {
    int m, n, k, sg;
    Layout dense, grouped;
    bool packed;
};
constexpr Shape kShapes[] = {{8, 16, 16, 16, {2, 2, 2, 2, false}, {2, 2, 2, 2, false}, true},
                             {8, 8, 16, 8, {4, 8, 4, 4, true}, {2, 8, 2, 2, false}, true},
                             {16, 16, 16, 32, {2, 2, 2, 2, false}, {2, 2, 2, 2, false}, false}};
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
template<typename T, typename U>
inline auto local_as(sycl::multi_ptr<U, sycl::access::address_space::local_space, sycl::access::decorated::no> p) {
    using V = sycl::multi_ptr<void, sycl::access::address_space::local_space, sycl::access::decorated::no>;
    return static_cast<sycl::multi_ptr<T, sycl::access::address_space::local_space, sycl::access::decorated::no>>(V(p));
}

// TT > 0: in fragments, the K step's two halves of 16 one after the other: A as [k / 16][row][k % 16] (a fragment's
// rows contiguous), B packed as [k / 16][row / TT][k % 16 / 2][row % TT][k & 1] (a fragment of TT columns contiguous);
// on the A380 the matrix loads from local memory took the 32-byte rows of a fragment 64 bytes apart one at a time,
// and contiguous fragments made a product 12% faster.  WIDE: those stores 16 bytes (A) and 4 bytes (a pair of B) at a
// time.
template<typename E, int R, int LD, bool PACKED, int TT = 0, bool WIDE = false>
inline void stage(const E* A, int64_t row0, int64_t n, int64_t K, int64_t k0, const sycl::local_accessor<E, 1>& dst,
                  int off, int lid, int lanes) {
    using V = sycl::vec<uint16_t, VEC>;
    for (int v = lid; v < R * (KC / VEC); v += lanes) {
        const int r = PACKED ? v % R : v / (KC / VEC);
        const int c = (PACKED ? v / R : v % (KC / VEC)) * VEC;
        const int64_t row = row0 + r;
        const V val = row < n ? *reinterpret_cast<const V*>(A + row * K + k0 + c) : V(0);
        if constexpr (TT > 0 && WIDE) {
            const auto base = dst.template get_multi_ptr<sycl::access::decorated::no>() + off + (c / 16) * (R * 16);
            if constexpr (PACKED) {
                const auto d = local_as<uint32_t>(base + (r / TT) * (16 * TT) + (r % TT) * 2);
                for (int j = 0; j < VEC; j += 2)
                    d[(c % 16 + j) / 2 * TT] = (uint32_t) val[j] | (uint32_t) val[j + 1] << 16;
            } else {
                *local_as<V>(base + r * 16 + c % 16) = val;
            }
            continue;
        }
        for (int j = 0; j < VEC; ++j) {
            const int k = c + j;
            if constexpr (TT > 0 && PACKED)   // B in fragments: [k / 16][r / TT][k % 16 / 2][r % TT][k & 1]
                dst[off + (k / 16) * (R * 16) + (r / TT) * (16 * TT) + (k % 16 / 2) * (2 * TT) + (r % TT) * 2 +
                    (k & 1)] = sycl::bit_cast<E>(val[j]);
            else if constexpr (TT > 0)        // A in fragments: [k / 16][r][k % 16]
                dst[off + (k / 16) * (R * 16) + r * 16 + k % 16] = sycl::bit_cast<E>(val[j]);
            else if constexpr (PACKED) dst[off + (k / 2) * (2 * R) + r * 2 + (k & 1)] = sycl::bit_cast<E>(val[j]);
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
    // A and B in local memory a fragment after another (Intel's shapes), stored 16 and 4 bytes at a time in a product's
    // layout (in the groups' layout the 2-byte stores were faster on the A380: 6.8 ms against 7.1)
    static constexpr bool FR = PACKED && TK == 16, WIDE = FR && !GROUPED;
    static constexpr Layout L = GROUPED ? kShapes[S].grouped : kShapes[S].dense;
    static constexpr int NSGM = L.wgm, NSGN = L.wgn, NSG = NSGM * NSGN;
    static constexpr int SGM = L.sgm, SGN = L.sgn;   // tiles a sub-group
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
            stage<E, WM, LDA, false, FR, WIDE>(x, wm0, rows, K, 0, as, 0, lid, NSG * SG);
            stage<E, WN, LDB, PACKED, FR ? TN : 0, WIDE>(w, wn0, N, K, 0, bs, 0, lid, NSG * SG);
            sycl::group_barrier(it.get_group());
            int buf = 0;
            for (int64_t k0 = 0; k0 < K; k0 += KC) {
                if (k0 + KC < K) {
                    const int nb = buf ^ 1;
                    stage<E, WM, LDA, false, FR, WIDE>(x, wm0, rows, K, k0 + KC, as, nb * ASZ, lid, NSG * SG);
                    stage<E, WN, LDB, PACKED, FR ? TN : 0, WIDE>(w, wn0, N, K, k0 + KC, bs, nb * BSZ, lid, NSG * SG);
                }
                const auto apb = ap + buf * ASZ;
                const auto bpb = bp + buf * BSZ;
                for (int kk = 0; kk < KC; kk += TK) {
                    constexpr auto BLAYOUT = PACKED ? mx::layout::ext_intel_packed : mx::layout::col_major;
                    mx::joint_matrix<sycl::sub_group, E, mx::use::a, TM, TK, mx::layout::row_major> a[SGM];
                    mx::joint_matrix<sycl::sub_group, E, mx::use::b, TK, TN, BLAYOUT> b[SGN];
                    if constexpr (FR) {
                        for (int i = 0; i < SGM; ++i)
                            mx::joint_matrix_load(sg, a[i], apb + (kk / 16) * (WM * 16) + (ms + i * TM) * 16, 16);
                    } else {
                        for (int i = 0; i < SGM; ++i)
                            mx::joint_matrix_load(sg, a[i], apb + (ms + i * TM) * LDA + kk, LDA);
                    }
                    for (int j = 0; j < SGN; ++j) {
                        if constexpr (FR)
                            mx::joint_matrix_load(sg, b[j],
                                                  bpb + (kk / 16) * (WN * 16) + (ns + j * TN) / TN * (16 * TN), 2 * TN);
                        else if constexpr (PACKED)
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
    auto get(syclex::properties_tag) const {
        if constexpr (L.grf)
            return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>,
                                      sycl::ext::intel::experimental::grf_size<256>};
        else return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
    }
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

// Whether the device builds shape S's dense layout: one with the large register file fails to build where the device
// lacks the mode, so one such kernel is built here, once, before any is launched; without it a product takes the
// groups' layout, which needs no large register file.
template<int S>
bool dense_layout_ok(const sycl::queue& q) {
    if constexpr (!kShapes[S].dense.grf) {
        return true;
    } else {
        static const bool ok = [&q] {
            try {
                (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(
                    q.get_context(), {q.get_device()}, {sycl::get_kernel_id<Kernel<S, sycl::half, false, false>>()});
                return true;
            } catch (const sycl::exception& e) {
                std::fprintf(stderr, "strata: the GPU does not build mma_gemm's kernel with the large register file "
                             "(%s): its products take the smaller layout\n", e.what());
                return false;
            }
        }();
        return ok;
    }
}

template<int S, typename E>
sycl::event run(sycl::queue& q, const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y, int64_t ldy,
                const int32_t* bounds, int G, int64_t T, int64_t N, int64_t K, bool accumulate) {
    const auto* x = reinterpret_cast<const E*>(X);
    const auto* w = reinterpret_cast<const E*>(W);
    if (bounds) return launch<S, E, false, true>(q, x, w, w_stride, Y, ldy, bounds, G, T, N, K);
    const bool dense = dense_layout_ok<S>(q);
    if (accumulate)
        return dense ? launch<S, E, true, false>(q, x, w, w_stride, Y, ldy, bounds, G, T, N, K)
                     : launch<S, E, true, true>(q, x, w, w_stride, Y, ldy, bounds, G, T, N, K);
    return dense ? launch<S, E, false, false>(q, x, w, w_stride, Y, ldy, bounds, G, T, N, K)
                 : launch<S, E, false, true>(q, x, w, w_stride, Y, ldy, bounds, G, T, N, K);
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
