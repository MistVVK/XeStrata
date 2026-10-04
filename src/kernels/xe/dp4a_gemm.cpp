// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/dp4a_gemm.cpp - see dp4a_gemm.hpp.
//
// A work-group of 16 x 16 work-items (in sub-groups of 16, or of 32 on a GPU without 16: see narrow_sub_group) computes a BM x BN tile of Y, PM x PN outputs each (rows and columns 16 apart, so
// neighbouring work-items read neighbouring words of local memory).  K advances one 32-value block at a time: the
// work-items load the block's BM rows of X and BN rows of W, each row shared by a few neighbouring lanes that find its
// largest magnitude together, and write the int8 values (4 a word) and the scale to local memory.  Quantizing while
// loading needs no int8 copies in memory; with them made beforehand the products alone were 15% faster on the
// largest shapes and no faster on the experts', before counting the quantization (bench/results/2026-10-02-dp4a).
#include "dp4a_gemm.hpp"
#include "cuda_intrinsics.hpp"
#include "device_target.hpp"

#include <algorithm>
#include <cstdlib>

namespace strata::kernels::xe {
namespace {

constexpr int TS = 16;   // the work-group's side

template<bool BF16>
inline float value(uint16_t b) {
    if constexpr (BF16) return sycl::bit_cast<float>((uint32_t) b << 16);
    else return (float) sycl::bit_cast<sycl::half>(b);
}

// Block k0 .. k0 + 32 of ROWS rows from row0 of A (row stride K) into q[8][ROWS] (int8 x 4) and d[ROWS] in local
// memory; lanes `lid`.  Rows past `n` are zero.  The local memory goes in as accessors: through generic pointers every
// access paid an address-space check, and the kernel ran at a quarter of its speed.
template<bool BF16, int ROWS>
inline void load_block(sycl::sub_group sg, int lid, const uint16_t* A, int64_t row0, int64_t n, int64_t K,
                       int64_t k0, const sycl::local_accessor<int, 2>& q, const sycl::local_accessor<float, 1>& d) {
    constexpr int V = ROWS * 32 / (TS * TS), P = 32 / V;   // values a work-item, work-items a row
    static_assert(V >= 4 && V % 4 == 0 && P >= 1 && P <= 16, "tile");
    const int r = lid / P, part = lid % P;
    const bool ok = row0 + r < n;
    float v[V];
    float m = 0.0f;
    for (int j = 0; j < V; ++j) {
        v[j] = ok ? value<BF16>(A[(row0 + r) * K + k0 + (int64_t) part * V + j]) : 0.0f;
        m = sycl::fmax(m, sycl::fabs(v[j]));
    }
    for (int o = 1; o < P; o <<= 1) m = sycl::fmax(m, sycl::permute_group_by_xor(sg, m, o));
    const float inv = m == 0.0f ? 0.0f : 127.0f / m;
    for (int j = 0; j < V; j += 4) {
        uint32_t p = 0;
        for (int t = 0; t < 4; ++t) p |= ((uint32_t) (int) sycl::round(v[j + t] * inv) & 0xFFu) << (8 * t);
        q[(part * V + j) / 4][r] = (int) p;
    }
    if (part == 0) d[r] = m / 127.0f;
}

// G groups of rows (bounds[e] .. bounds[e + 1], or all `rows_max` rows when bounds is null), each by its W.  A row's
// P lanes are neighbours in a P-aligned block, so its reduction stays within a sub-group of 16 or 32.
template<bool BF16, int PM, int PN, int SG>
sycl::event launch(sycl::queue& q, const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y,
                   int64_t ldy, const int32_t* bounds, int G, int64_t rows_max, int64_t N, int64_t K) {
    constexpr int BM = TS * PM, BN = TS * PN;
    const int64_t tiles_m = (rows_max + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 2> xs(sycl::range<2>(8, BM), h), ws(sycl::range<2>(8, BN), h);
        sycl::local_accessor<float, 1> xds(sycl::range<1>(BM), h), wds(sycl::range<1>(BN), h);
        h.parallel_for(sycl::nd_range<2>({(size_t) (G * tiles_m * TS), (size_t) (tiles_n * TS)}, {TS, TS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]] {
            const int64_t g = (int64_t) it.get_group(0), ex = g / tiles_m, m0 = g % tiles_m * BM;
            const int64_t n0 = (int64_t) it.get_group(1) * BN;
            const int64_t r0 = bounds ? bounds[ex] : 0, rows = bounds ? bounds[ex + 1] - r0 : rows_max;
            if (m0 >= rows) return;   // the whole work-group: this group of rows is shorter than the launch
            const uint16_t* x = X + r0 * K;
            const uint16_t* w = W + ex * w_stride;
            const auto sg = it.get_sub_group();
            const int ty = (int) it.get_local_id(0), tx = (int) it.get_local_id(1), lid = ty * TS + tx;
            float acc[PM][PN] = {};
            for (int64_t k0 = 0; k0 < K; k0 += 32) {
                load_block<BF16, BM>(sg, lid, x, m0, rows, K, k0, xs, xds);
                load_block<BF16, BN>(sg, lid, w, n0, N, K, k0, ws, wds);
                sycl::group_barrier(it.get_group());
                int iacc[PM][PN] = {};
                for (int c = 0; c < 8; ++c) {
                    int a[PM], b[PN];
                    for (int i = 0; i < PM; ++i) a[i] = xs[c][ty + TS * i];
                    for (int j = 0; j < PN; ++j) b[j] = ws[c][tx + TS * j];
                    for (int i = 0; i < PM; ++i)
                        for (int j = 0; j < PN; ++j) iacc[i][j] = dp4a(a[i], b[j], iacc[i][j]);
                }
                for (int i = 0; i < PM; ++i)
                    for (int j = 0; j < PN; ++j)
                        acc[i][j] = sycl::fma((float) iacc[i][j], xds[ty + TS * i] * wds[tx + TS * j], acc[i][j]);
                sycl::group_barrier(it.get_group());
            }
            float* y = Y + r0 * ldy;
            for (int i = 0; i < PM; ++i) {
                const int64_t r = m0 + ty + (int64_t) TS * i;
                if (r >= rows) break;
                for (int j = 0; j < PN; ++j) {
                    const int64_t c = n0 + tx + (int64_t) TS * j;
                    if (c < N) y[r * ldy + c] = acc[i][j];
                }
            }
        });
    });
}

template<bool BF16, int PM, int PN>
sycl::event launch(sycl::queue& q, const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y,
                   int64_t ldy, const int32_t* bounds, int G, int64_t rows_max, int64_t N, int64_t K) {
    if (narrow_sub_group(q) == 16) return launch<BF16, PM, PN, 16>(q, X, W, w_stride, Y, ldy, bounds, G, rows_max, N, K);
    return launch<BF16, PM, PN, 32>(q, X, W, w_stride, Y, ldy, bounds, G, rows_max, N, K);
}

}  // namespace

int narrow_sub_group(const sycl::queue& q) {
    static const int sg = [&q] {
        // STRATA_SUB_GROUP_32=1 takes 32 where 16 is listed too, to compare the two on one GPU
        if (const char* v = std::getenv("STRATA_SUB_GROUP_32"); v != nullptr && std::strtol(v, nullptr, 10) != 0)
            return 32;
        const auto s = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
        return std::find(s.begin(), s.end(), size_t(16)) != s.end() ? 16 : 32;
    }();
    return sg;
}

int64_t fill_rows(const sycl::queue& q, int64_t b70_rows) {
    const int64_t cu = (int64_t) q.get_device().get_info<sycl::info::device::max_compute_units>();
    return std::max<int64_t>(1, b70_rows * cu / 256);
}

sycl::event dp4a_gemm(sycl::queue& q, bool bf16, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N,
                      int64_t K, int64_t ldy) {
    // 128 x 64 tiles where there are rows enough to fill the GPU with them (20 TOPS against 17 for 64 x 64 on the
    // wide products, on the B70), 64 x 64 for short prompts
    const int64_t many = fill_rows(q, 1024);
    if (bf16) {
        if (T >= many) return launch<true, 8, 4>(q, X, W, 0, Y, ldy, nullptr, 1, T, N, K);
        return launch<true, 4, 4>(q, X, W, 0, Y, ldy, nullptr, 1, T, N, K);
    }
    if (T >= many) return launch<false, 8, 4>(q, X, W, 0, Y, ldy, nullptr, 1, T, N, K);
    return launch<false, 4, 4>(q, X, W, 0, Y, ldy, nullptr, 1, T, N, K);
}

sycl::event dp4a_gemm_grouped(sycl::queue& q, const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y,
                              const int32_t* bounds, int G, int64_t max_rows, int64_t N, int64_t K) {
    // the experts' 80-240 rows: 64 x 64 tiles (10 TOPS against 5 for 128 x 64)
    return launch<false, 4, 4>(q, X, W, w_stride, Y, N, bounds, G, max_rows, N, K);
}

}  // namespace strata::kernels::xe
