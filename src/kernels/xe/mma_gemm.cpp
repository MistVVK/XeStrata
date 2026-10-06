// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/mma_gemm.cpp - see mma_gemm.hpp.
//
// A work-group of sub-groups computes a tile of Y, each sub-group some matrix tiles of it (Layout below).  K advances
// 32 values at a time: the work-group copies the step's rows of X and of W into local memory, 16 bytes a work-item,
// zero past the last row (so any row count works, the experts' groups too), into one of two buffers while the
// sub-groups multiply out of the other.  W goes in as B in the layout the matrix engines take: Intel's VNNI-packed one
// (pairs of K values together) on Intel GPUs, column-major on NVIDIA's.  A tile of accumulators within Y goes straight
// there; one that crosses the last row or column goes out through local memory, row by row, so the rows and columns
// past the last are not written (straight stores made the A380's products 0-4% faster than all through local memory).
//
// On the A380 (Xe-HPG; bench/results/2026-10-04-dg2-dp4a) a 4096 x 2560 x 2560 FP16 product took 52 ms with W read
// from global memory by every sub-group; with W in local memory 116 (column-major), 33 (transposed), 23 (VNNI-packed),
// 21 (neighbouring lanes storing side by side), 20 (two buffers), 16 (4 x 4 tiles a sub-group in the large register
// file; in the default 128 registers they spilled), 14.2 (a fragment's rows contiguous) and 13.6 (16-byte stores).
// DP4a takes 20.  Loading the next step into registers while multiplying, to store it after, spilled (91-103 ms).
#include "mma_gemm.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "device_target.hpp"
#include "strata/core/per_device.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <type_traits>
#include <unordered_map>

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

// NVIDIA's shape from sm_80 on: the tiles read from local memory with ldmatrix and multiplied with mma.sync
// m16n8k16, in PTX.  joint_matrix's CUDA loads read a 16 x 16 tile as 32-bit generic loads (four a lane, four times
// the instructions of ldmatrix's), and the products of a 512 x 10240 x 2560 FP16 product ran at 46 TFLOP/s on the
// RTX 4070 against cuBLAS's 60.  The sums are the same HMMA instructions in the same order.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && defined(__SYCL_CUDA_ARCH__) && __SYCL_CUDA_ARCH__ >= 800
#define STRATA_MMA_PTX 1
inline uint32_t smem_addr(const void* p) {
    uint32_t a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
}
inline void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
template<bool BF>
inline void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    if constexpr (BF)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%0, %1, %2, %3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%0, %1, %2, %3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#else
#define STRATA_MMA_PTX 0
#endif
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
        if constexpr (TT == 0 && !PACKED && LD * sizeof(E) % 16 == 0) {   // row by row, 16 bytes at a time
            *local_as<V>(dst.template get_multi_ptr<sycl::access::decorated::no>() + off + r * LD + c) = val;
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
    static constexpr int LDA = PACKED ? KC : KC + PAD, LDB = PACKED ? KC : KC + PAD;   // A's and B's row strides
    static constexpr int ASZ = WM * LDA, BSZ = WN * LDB;           // a buffer's elements
    // NVIDIA's shape reads ahead into registers (PREFETCH below) and needs one buffer of each; Intel's packed shapes two
    static constexpr int NBUF = PACKED ? 2 : 1;
    static constexpr size_t LOCAL = (NBUF * ASZ + NBUF * BSZ) * sizeof(uint16_t) + (size_t) NSG * TM * TN * sizeof(float);
    const E* X;
    const E* W;
    int64_t w_stride;
    float* Y;
    int64_t ldy;
    const int32_t* bounds;
    int64_t T, N, K, tiles_m, tiles_n;
    int64_t kslice;   // a product's K split: the values of K one slice takes (K: none), slice s writing Y + s * T * ldy
    sycl::local_accessor<E, 1> as;       // two buffers of the step's rows of X: WM x KC
    sycl::local_accessor<E, 1> bs;       // two buffers of the step's rows of W, as B: WN x KC
    sycl::local_accessor<float, 1> cs;   // a TM x TN tile of Y a sub-group, on its way out

    // NVIDIA's shape (B column-major): the next step's rows are read into registers before this step's products and
    // stored to local memory after them.  stage() stores what it has just read, so each step waited on its loads
    // before multiplying (RTX 4070: long-scoreboard stalls, 33% occupancy by local memory).  Intel's packed shapes keep
    // stage(): on the A380 the registers spilled (see the file's note).
    static constexpr bool PREFETCH = !PACKED;
    static constexpr bool PTX = STRATA_MMA_PTX && !PACKED && TM == 16 && TN == 16 && TK == 16;
    static constexpr int LANES = NSG * SG, NVA = (WM * (KC / VEC) + LANES - 1) / LANES,
                         NVB = (WN * (KC / VEC) + LANES - 1) / LANES;
    using V8 = sycl::vec<uint16_t, VEC>;
    template<int R, int NV>
    static void fetch(const E* A, int64_t row0, int64_t n, int64_t K, int64_t k0, int lid, V8 (&v)[NV]) {
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int q = lid + i * LANES, r = q / (KC / VEC), c = q % (KC / VEC) * VEC;
            const int64_t row = row0 + r;
            v[i] = q < R * (KC / VEC) && row < n ? *reinterpret_cast<const V8*>(A + row * K + k0 + c) : V8(0);
        }
    }
    template<int R, int LD, int NV>
    static void put(const sycl::local_accessor<E, 1>& dst, int off, int lid, const V8 (&v)[NV]) {
        const auto base = dst.template get_multi_ptr<sycl::access::decorated::no>() + off;
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int q = lid + i * LANES, r = q / (KC / VEC), c = q % (KC / VEC) * VEC;
            if (q < R * (KC / VEC)) *local_as<V8>(base + r * LD + c) = v[i];
        }
    }

    void operator()(sycl::nd_item<1> it) const {
        // BF16 on NVIDIA's tensor cores from sm_80 on, FP16 from sm_70 on (device_target.hpp)
        constexpr bool arch_ok = STRATA_NV_ARCH == 0 || STRATA_NV_ARCH >= (std::is_same_v<E, bf16_t> ? 800 : 700);
        if constexpr (!built<S>() || !arch_ok) {
            (void) it;
        } else {
            const int64_t g = (int64_t) it.get_group(0), ex = g / (tiles_m * tiles_n), r = g % (tiles_m * tiles_n);
            // M first: the work-groups that read a tile of W run side by side and find it in L2 (N first, a 52 MB W
            // was read from memory once a tile of 64 rows: RTX 4070, 512 x 10240 x 2560, 1735 -> 1015 us)
            const int64_t wm0 = r % tiles_m * WM, wn0 = r / tiles_m * WN;
            const int64_t row0 = bounds ? bounds[ex] : 0, rows = bounds ? bounds[ex + 1] - row0 : T;
            if (wm0 >= rows) return;   // the whole work-group: this group of rows is shorter than the launch
            // without bounds, ex is the slice of K (run: a product split along K)
            const int64_t kb = bounds ? 0 : ex * kslice, ke = bounds ? K : sycl::min(K, kb + kslice);
            const E* x = X + row0 * K;
            const E* w = W + (bounds ? ex * w_stride : 0);
            float* y = Y + (bounds ? row0 * ldy : ex * T * ldy);
            const auto sg = it.get_sub_group();
            const int sgid = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int lid = (int) it.get_local_id(0);
            const int ms = sgid / NSGN * SGM * TM, ns = sgid % NSGN * SGN * TN;   // the sub-group's tile
            const auto ap = as.template get_multi_ptr<sycl::access::decorated::no>();
            const auto bp = bs.template get_multi_ptr<sycl::access::decorated::no>();
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c[SGM][SGN];
            float pacc[PTX ? SGM : 1][PTX ? 2 * SGN : 1][4] = {};   // PTX: (16 rows, 8 columns) tiles, mma.sync's
            if constexpr (!PTX)
                for (int i = 0; i < SGM; ++i)
                    for (int j = 0; j < SGN; ++j) mx::joint_matrix_fill(sg, c[i][j], 0.0f);
            stage<E, WM, LDA, false, FR, WIDE>(x, wm0, rows, K, kb, as, 0, lid, NSG * SG);
            stage<E, WN, LDB, PACKED, FR ? TN : 0, WIDE>(w, wn0, N, K, kb, bs, 0, lid, NSG * SG);
            sycl::group_barrier(it.get_group());
            int buf = 0;
            V8 va[PREFETCH ? NVA : 1], vb[PREFETCH ? NVB : 1];
            for (int64_t k0 = kb; k0 < ke; k0 += KC) {
                const bool more = k0 + KC < ke;
                if constexpr (PREFETCH) {
                    if (more) {
                        fetch<WM, NVA>(x, wm0, rows, K, k0 + KC, lid, va);
                        fetch<WN, NVB>(w, wn0, N, K, k0 + KC, lid, vb);
                    }
                } else if (more) {
                    const int nb = buf ^ 1;
                    stage<E, WM, LDA, false, FR, WIDE>(x, wm0, rows, K, k0 + KC, as, nb * ASZ, lid, NSG * SG);
                    stage<E, WN, LDB, PACKED, FR ? TN : 0, WIDE>(w, wn0, N, K, k0 + KC, bs, nb * BSZ, lid, NSG * SG);
                }
                const auto apb = ap + buf * ASZ;
                const auto bpb = bp + buf * BSZ;
#if STRATA_MMA_PTX
                if constexpr (PTX) {
                    constexpr bool BF = std::is_same_v<E, bf16_t>;
                    const E* al = apb.get();
                    const E* bl = bpb.get();
#pragma unroll
                    for (int kk = 0; kk < KC; kk += 16) {
                        uint32_t af[SGM][4], bf[SGN][4];
#pragma unroll
                        for (int i = 0; i < SGM; ++i)   // rows lane % 16, columns kk + 8 (lane / 16)
                            ldsm4(af[i], smem_addr(al + (ms + i * 16 + lane % 16) * LDA + kk + lane / 16 * 8));
#pragma unroll
                        for (int j = 0; j < SGN; ++j)   // outputs 8 (lane / 16) + lane % 8, columns kk + 8 (lane / 8 % 2)
                            ldsm4(bf[j], smem_addr(bl + (ns + j * 16 + lane / 16 * 8 + lane % 8) * LDB + kk +
                                                   lane / 8 % 2 * 8));
#pragma unroll
                        for (int i = 0; i < SGM; ++i)
#pragma unroll
                            for (int j = 0; j < SGN; ++j) {
                                mma16816<BF>(pacc[i][2 * j], af[i], bf[j][0], bf[j][1]);
                                mma16816<BF>(pacc[i][2 * j + 1], af[i], bf[j][2], bf[j][3]);
                            }
                    }
                }
#endif
                for (int kk = 0; kk < (PTX ? 0 : KC); kk += TK) {
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
                if constexpr (PREFETCH) {
                    if (more) {   // into the one buffer, once every sub-group's products have read it
                        sycl::group_barrier(it.get_group());
                        put<WM, LDA, NVA>(as, 0, lid, va);
                        put<WN, LDB, NVB>(bs, 0, lid, vb);
                    }
                }
                sycl::group_barrier(it.get_group());
                buf = (buf + 1) % NBUF;
            }
            if constexpr (PTX) {   // mma.sync's tile: d0, d1 at row lane / 4, columns 2 (lane % 4) and +1; d2, d3 8 rows on
                const int gr = lane / 4, gc = lane % 4 * 2;
#pragma unroll
                for (int i = 0; i < SGM; ++i)
#pragma unroll
                    for (int jn = 0; jn < 2 * SGN; ++jn) {
                        const int64_t n = wn0 + ns + (int64_t) jn * 8 + gc;
                        if (n >= N) continue;
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int64_t row = wm0 + ms + (int64_t) i * 16 + gr + (int64_t) h * 8;
                            if (row >= rows) continue;
                            float* o = y + row * ldy + n;
                            o[0] = ACC ? o[0] + pacc[i][jn][2 * h] : pacc[i][jn][2 * h];
                            o[1] = ACC ? o[1] + pacc[i][jn][2 * h + 1] : pacc[i][jn][2 * h + 1];
                        }
                    }
                return;
            }
            const auto cp = cs.template get_multi_ptr<sycl::access::decorated::no>() + (std::ptrdiff_t) sgid * TM * TN;
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) {
                    const int64_t n = wn0 + ns + (int64_t) j * TN, m = wm0 + ms + (int64_t) i * TM;
                    if (!ACC && m + TM <= rows && n + TN <= N) {   // a tile within Y: stored straight there
                        const auto gy = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                                  sycl::access::decorated::no>(y + m * ldy + n);
                        mx::joint_matrix_store(sg, c[i][j], gy, ldy, mx::layout::row_major);
                        continue;
                    }
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
                   const int32_t* bounds, int G, int64_t T, int64_t N, int64_t K, int64_t kslice) {
    using KT = Kernel<S, E, ACC, GROUPED>;
    const int64_t tiles_m = (T + KT::WM - 1) / KT::WM, tiles_n = (N + KT::WN - 1) / KT::WN;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<E, 1> as(sycl::range<1>(KT::NBUF * KT::ASZ), h), bs(sycl::range<1>(KT::NBUF * KT::BSZ), h);
        sycl::local_accessor<float, 1> cs(sycl::range<1>(KT::NSG * KT::TM * KT::TN), h);
        const size_t wg = (size_t) KT::NSG * KT::SG;
        h.parallel_for(sycl::nd_range<1>((size_t) (G * tiles_m * tiles_n) * wg, wg),
                       KT{X, W, w_stride, Y, ldy, bounds, T, N, K, tiles_m, tiles_n, kslice, as, bs, cs});
    });
}

bool reported(const sycl::device& d, mx::matrix_type in, const Shape& s) {
    const auto sg = d.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sg.begin(), sg.end(), (size_t) s.sg) == sg.end()) return false;
    for (const auto& c : matrix_combinations(d)) {
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

bool groups_layout_fits(int s, const sycl::queue& q);

// The first shape the device reports for the input type whose layout for the experts' groups (the smaller one) it
// runs, or -1; worked out once for each GPU.
int shape(const sycl::queue& q, bool bf16) {
    struct Shapes {
        int f16, b16;
    };
    static core::PerDevice<Shapes> per_device;
    const Shapes& s = per_device.get(q.get_device(), [&q] {
        const auto pick = [&q](mx::matrix_type in) {
            for (int i = 0; i < (int) std::size(kShapes); ++i)
                if (reported(q.get_device(), in, kShapes[i])) return groups_layout_fits(i, q) ? i : -1;
            return -1;
        };
        return Shapes{pick(mx::matrix_type::fp16), pick(mx::matrix_type::bf16)};
    });
    return bf16 ? s.b16 : s.f16;
}

// Whether shape S's layout for a product (GROUPED false) or for the experts' groups fits the device: its local memory
// and its work-group size, as the device reports them.  A layout in the large register file takes at most a quarter
// of the device's largest work-group: on the A380 (1,024) 256 lanes ran, and 512 lost the device although the kernel
// reported taking them (kernel_device_specific::work_group_size), so the kernel's own report does not guard it.  The
// kernels are not built to check them: built beforehand (in the engine's context or one of its own), the products ran
// 3% slower on the A380.  A device without the large register file fails the build at the first launch (run).
template<int S, bool GROUPED>
bool layout_fits(const sycl::queue& q) {
    using KT = Kernel<S, sycl::half, false, GROUPED>;
    static core::PerDevice<bool> per_device;
    return per_device.get(q.get_device(), [&q] {
        const sycl::device d = q.get_device();
        const char* what = GROUPED ? "the experts' groups" : "a product";
        const size_t local = d.get_info<sycl::info::device::local_mem_size>();
        const size_t wg = d.get_info<sycl::info::device::max_work_group_size>() / (KT::L.grf ? 4 : 1);
        if (local >= KT::LOCAL && wg >= (size_t) KT::NSG * KT::SG) return true;
        std::fprintf(stderr, "strata: mma_gemm's layout for %s needs %zu bytes of local memory and work-groups of %d, "
                     "the GPU has %zu and %zu%s\n", what, KT::LOCAL, KT::NSG * KT::SG, local, wg,
                     KT::L.grf ? " (a quarter of its largest, with the large register file)" : "");
        return false;
    });
}

bool groups_layout_fits(int s, const sycl::queue& q) {
    if (s == 0) return layout_fits<0, true>(q);
    if (s == 1) return layout_fits<1, true>(q);
    return layout_fits<2, true>(q);
}

// A product's slices along K: when its tiles fill less than twice the device's compute units (its report) and it has
// the rows of a prompt, so many slices that they do, each at least 4 steps of K.  The slices' sums go to a scratch
// buffer and one kernel adds them in slice order: the result has other last bits than one slice's, the same on every
// run.  RTX 4070: 512 x 320 x 10240 (BF16, the hyper-connection read: 40 tiles) 305 -> 159 us (four times the compute
// units: no faster).  Not in a decode window's few rows (a captured graph would keep the scratch buffer's address).
int64_t k_slices(const sycl::queue& q, int64_t T, int64_t tiles, int64_t K) {
    const int64_t cu = (int64_t) q.get_device().get_info<sycl::info::device::max_compute_units>();
    if (T < 64 || tiles >= 2 * cu) return 1;
    return std::max<int64_t>(1, std::min((2 * cu + tiles - 1) / tiles, K / ((int64_t) 4 * KC)));
}

// one buffer for each GPU, on its device
float* k_scratch(sycl::queue& q, size_t floats) {
    struct Scratch {
        float* buf = nullptr;
        size_t cap = 0;
    };
    static std::mutex mu;
    static std::unordered_map<sycl::device, Scratch> per_device;
    std::lock_guard<std::mutex> lock(mu);
    Scratch& s = per_device[q.get_device()];
    if (floats > s.cap) {
        q.wait();   // the products queued before may still read the old one
        if (s.buf != nullptr) sycl::free(s.buf, q);
        s.buf = sycl::malloc_device<float>(floats, q);
        s.cap = s.buf != nullptr ? floats : 0;
    }
    return s.buf;
}

template<int S, typename E, bool GROUPED>
sycl::event launch_split(sycl::queue& q, const E* x, const E* w, float* Y, int64_t ldy, int64_t T, int64_t N,
                         int64_t K, int64_t* slices) {
    using KT = Kernel<S, E, false, GROUPED>;
    const int64_t tiles = ((T + KT::WM - 1) / KT::WM) * ((N + KT::WN - 1) / KT::WN);
    int64_t ns = k_slices(q, T, tiles, K);
    const int64_t kslice = (K / ns + KC - 1) / KC * KC;
    ns = (K + kslice - 1) / kslice;
    *slices = ns;
    if (ns <= 1) return launch<S, E, false, GROUPED>(q, x, w, 0, Y, ldy, nullptr, 1, T, N, K, K);
    float* part = k_scratch(q, (size_t) (ns * T * N));
    if (part == nullptr) { *slices = 1; return launch<S, E, false, GROUPED>(q, x, w, 0, Y, ldy, nullptr, 1, T, N, K, K); }
    launch<S, E, false, GROUPED>(q, x, w, 0, part, N, nullptr, (int) ns, T, N, K, kslice);
    return q.parallel_for(sycl::range<2>((size_t) T, (size_t) N), [=](sycl::id<2> i) {
        const int64_t t = (int64_t) i[0], n = (int64_t) i[1];
        float v = part[t * N + n];
        for (int64_t k = 1; k < ns; ++k) v += part[(k * T + t) * N + n];
        Y[t * ldy + n] = v;
    });
}

template<int S, typename E>
sycl::event run(sycl::queue& q, const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y, int64_t ldy,
                const int32_t* bounds, int G, int64_t T, int64_t N, int64_t K, bool accumulate) {
    const auto* x = reinterpret_cast<const E*>(X);
    const auto* w = reinterpret_cast<const E*>(W);
    if (bounds) return launch<S, E, false, true>(q, x, w, w_stride, Y, ldy, bounds, G, T, N, K, K);
    int64_t slices = 1;
    // a product's layout until it fails to launch (the large register file not built: the launch throws before anything
    // is queued), then the groups' layout
    static std::atomic<bool> dense = layout_fits<S, false>(q);
    if (dense.load(std::memory_order_relaxed)) {
        try {
            return accumulate ? launch<S, E, true, false>(q, x, w, w_stride, Y, ldy, bounds, G, T, N, K, K)
                              : launch_split<S, E, false>(q, x, w, Y, ldy, T, N, K, &slices);
        } catch (const sycl::exception& e) {
            if (dense.exchange(false))
                std::fprintf(stderr, "strata: mma_gemm's layout for a product did not launch (%s): its products take "
                             "the experts' layout\n", e.what());
        }
    }
    return accumulate ? launch<S, E, true, true>(q, x, w, w_stride, Y, ldy, bounds, G, T, N, K, K)
                      : launch_split<S, E, true>(q, x, w, Y, ldy, T, N, K, &slices);
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

bool mma_shape_is(const sycl::queue& q, int m, int n, int k, int sg) {
    if (!mma_usable(q, false)) return false;
    const Shape& s = kShapes[shape(q, false)];
    return s.m == m && s.n == n && s.k == k && s.sg == sg;
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
