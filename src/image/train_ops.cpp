// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/train_ops.cpp - see include/strata/image/train_ops.hpp.  FP32 where a gradient lives, FP16 only where a
// product reads one; nothing here is on the inference path, and these kernels take their own buffers.
#include "strata/image/train_ops.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "dit_rows.hpp"

namespace strata::image::train {

namespace {

constexpr int64_t kWG = 256;   // a work-group: one over a row's columns, or over the rows of one column

/// out[c] += sum over the rows t of f(t, c): a work-group a column, its items striding the rows
template <class F>
inline void col_sum(int64_t rows, int64_t cols, float* out, F f, void* s) {
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) cols * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t c = static_cast<int64_t>(it.get_group_linear_id());
            float v = 0.f;
            for (auto t = static_cast<int64_t>(it.get_local_linear_id()); t < rows; t += kWG) v += f(t, c);
            v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
            if (it.get_local_linear_id() == 0) out[c] += v;
        });
}

/// one reduction over the whole tensor: a work-group a chunk, the chunks' parts read back
/// COMBINE is plus or maximum over the device type
template <class F>
inline float tensor_reduce(int64_t n, F f, bool maximum, void* s) {
    const int64_t nparts = (n + kWG - 1) / kWG;
    float* part = nullptr;
    if (!gpu::alloc_device(&part, (size_t) nparts * 4))
        throw std::runtime_error("image engine (training): reduction");
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) nparts * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            float v = 0.f;
            for (auto i = static_cast<int64_t>(it.get_global_linear_id()); i < n;
                 i += static_cast<int64_t>(it.get_global_range(0)))
                v = maximum ? sycl::fmax(v, f(i)) : v + f(i);
            v = maximum ? sycl::reduce_over_group(it.get_group(), v, sycl::maximum<float>())
                        : sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
            if (it.get_local_linear_id() == 0) part[it.get_group_linear_id()] = v;
        });
    gpu::stream_sync(s);   // the copy below is the compute queue's; the kernels above ran on s
    std::vector<float> host((size_t) nparts);
    if (!gpu::copy(host.data(), part, (size_t) nparts * 4))
        throw std::runtime_error("image engine (training): reduction read");
    gpu::free(part);
    float all = 0.f;
    for (float v : host) all = maximum ? std::max(all, v) : all + v;
    return all;
}

}  // namespace

void lora_bwd(const float* hr, const float* xr, const uint16_t* h16t, const uint16_t* x16t,
                 const uint16_t* bt16, const float* dY, int64_t T, int64_t rows, int64_t cols, int64_t rank,
                 float scale, float* dB, float* dA, float* scratch, uint16_t* tmp16, float* rsT, prefill::Gemm& gemm,
                 void* stream) {
    float* dh = scratch;                  // [T][rank]
    float* sums = scratch + (int64_t) T * rank;
    uint16_t* dy16 = tmp16;               // [T][rows]
    uint16_t* dyt16 = dy16 + (int64_t) T * rows;    // [rows][T], dY's rows scaled by hr and transposed
    uint16_t* dht16 = dyt16 + (int64_t) rows * T;   // [rank][T], dh's rows scaled by xr and transposed
    float* rsY = rsT;                    // [T] dY's own row scales
    dit_rows::rows16(dit_rows::Plain{dY, rows}, T, rows, dy16, rsY, stream);
    // dh = scale (dY B): the same product the forward read, with B's transposed view as the weight
    // dh is written whole here (every one of its T * rank values), so it is overwritten, not added into: the
    // caller passes scratch, which the probe leaves uninitialized (fresh device pages read as zero, which is the
    // only reason adding into it ever passed parity)
    dgrad(dy16, rsY, bt16, T, rows, rank, dh, sums, false, gemm, stream);
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) T * rank), [=](sycl::id<1> id) {
        dh[id[0]] *= scale;
    });
    // dB = scale dY^T h, dA = scale dh^T x: both through one global scale on the gradient's rows
    const float g2 = scaled_transpose_f16(dh, xr, T, rank, dht16, stream);
    const float g1 = scaled_transpose_f16(dY, hr, T, rows, dyt16, stream);
    wgrad(dyt16, h16t, T, rows, rank, g1 * scale, dB, sums, true, gemm, stream);
    // dA reads dh, which already carries scale (dL/dh = scale * dY B): no second scale here
    wgrad(dht16, x16t, T, rank, cols, g2, dA, sums + (int64_t) rows * rank, true, gemm, stream);
}

void lokr_partial_trace(const float* dDelta, const uint16_t* w1, const uint16_t* w2, int64_t out_l, int64_t out_k,
                           int64_t in_m, int64_t in_n, float* dw1, float* dw2, void* stream) {
    // dw1[l][m] over the (k,n) of its block, and dw2[k][n] over the (l,m) of its block: each reads every dDelta value
    // once, so the two together are two passes over out x in.
    dit_rows::q_of(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) out_l * in_m * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t lm = static_cast<int64_t>(it.get_group_linear_id()), l = lm / in_m, m = lm - l * in_m;
            float v = 0.f;
            for (int64_t k = 0; k < out_k; ++k)
                for (auto n = static_cast<int64_t>(it.get_local_linear_id()); n < in_n; n += kWG)
                    v += dDelta[(l * out_k + k) * (in_m * in_n) + m * in_n + n] *
                         kernels::f32_from_f16(w2[k * in_n + n]);
            v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
            if (it.get_local_linear_id() == 0) dw1[l * in_m + m] = v;
        });
    dit_rows::q_of(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) out_k * in_n * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t kn = static_cast<int64_t>(it.get_group_linear_id()), k = kn / in_n, n = kn - k * in_n;
            float v = 0.f;
            for (int64_t l = 0; l < out_l; ++l)
                for (auto m = static_cast<int64_t>(it.get_local_linear_id()); m < in_m; m += kWG)
                    v += dDelta[(l * out_k + k) * (in_m * in_n) + m * in_n + n] *
                         kernels::f32_from_f16(w1[l * in_m + m]);
            v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
            if (it.get_local_linear_id() == 0) dw2[k * in_n + n] = v;
        });
}

void lokr_factor_bwd(const float* dw2, const uint16_t* w2a, const uint16_t* w2b, int64_t out_k, int64_t in_n,
                        int64_t rank, float* dw2a, float* dw2b, void* stream) {
    dit_rows::q_of(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) out_k * rank * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t g = static_cast<int64_t>(it.get_group_linear_id()), o = g / rank, r = g - o * rank;
            float v = 0.f;
            for (auto n = static_cast<int64_t>(it.get_local_linear_id()); n < in_n; n += kWG)
                v += dw2[o * in_n + n] * kernels::f32_from_f16(w2b[r * in_n + n]);
            v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
            if (it.get_local_linear_id() == 0) dw2a[o * rank + r] = v;
        });
    dit_rows::q_of(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rank * in_n * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t g = static_cast<int64_t>(it.get_group_linear_id()), r = g / in_n, n = g - r * in_n;
            float v = 0.f;
            for (auto o = static_cast<int64_t>(it.get_local_linear_id()); o < out_k; o += kWG)
                v += kernels::f32_from_f16(w2a[o * rank + r]) * dw2[o * in_n + n];
            v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
            if (it.get_local_linear_id() == 0) dw2b[r * in_n + n] = v;
        });
}

void dora_bwd(const float* dY, const float* p, const float* d, const float* m, const float* norms, int64_t T,
                 int64_t rows, float scale, float* dm, float* da, float* db, void* stream) {
    constexpr float kDoraEps = 1.1920929e-07f;   // kLoraDoraEps, as image_lora.cpp takes it
    if (dm != nullptr)
        col_sum(T, rows, dm,
                [=](int64_t t, int64_t r) {
                    return dY[t * rows + r] * (p[t * rows + r] + d[t * rows + r]) * scale / (norms[r] + kDoraEps);
                },
                stream);
    if (da == nullptr && db == nullptr) return;
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) rows), [=](sycl::id<1> id) {
        const int64_t r = static_cast<int64_t>(id[0]);
        const float cr = m[r] / (norms[r] + kDoraEps);
        if (da != nullptr) da[r] = 1.f + scale * (cr - 1.f);
        if (db != nullptr) db[r] = scale * cr;
    });
}

void scale_cols(float* y, const float* r, int64_t rows, int64_t cols, void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) rows * cols), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]);
        y[i] *= r[i % cols];   // r is per column: y[t][c] *= r[c]
    });
}

float flow_loss(const float* v_pred, const float* x0, const float* eps, int64_t n, float* dv_pred, float* dx0,
                   float* deps, void* stream) {
    const float inv_n = 1.f / (float) n;
    if (dv_pred != nullptr || dx0 != nullptr || deps != nullptr)
        dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const float g = 2.f * (v_pred[id[0]] - (eps[id[0]] - x0[id[0]])) * inv_n;
            if (dv_pred != nullptr) dv_pred[id[0]] = g;
            if (dx0 != nullptr) dx0[id[0]] = -g;   // the target eps - x0 reads x0 with a minus
            if (deps != nullptr) deps[id[0]] = g;
        });
    const float sq = tensor_reduce(n, [=](int64_t i) { const float d = v_pred[i] - (eps[i] - x0[i]); return d * d; },
                                        false, stream);
    return sq / (float) n;
}

float scaled_transpose_f16(const float* g, const float* rsX, int64_t rows, int64_t cols, uint16_t* out,
                              void* stream) {
    const int64_t n = rows * cols;
    const float G = dit_rows::pow2_scale(
        tensor_reduce(n,
                  [=](int64_t i) { return sycl::fabs(g[i] * (rsX != nullptr ? rsX[i / cols] : 1.f)); }, true,
                  stream));
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / cols, c = i - t * cols;
        out[c * rows + t] =
            kernels::f16_from_f32(g[i] * (rsX != nullptr ? rsX[t] : 1.f) / G);
    });
    return G;
}

void dgrad(const uint16_t* dy16, const float* rsY, const uint16_t* wt, int64_t T, int64_t N, int64_t K, float* dX,
              float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream) {
    gemm.f16(dy16, wt, scratch, T, K, N);   // wt[K,N] holds W[N,K]'s rows: the product is dY . W, in FP32 sums
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) T * K), [=](sycl::id<1> id) {
        const int64_t t = static_cast<int64_t>(id[0]) / K;
        const float v = scratch[id[0]] * (rsY != nullptr ? rsY[t] : 1.f);
        dX[id[0]] = accumulate ? dX[id[0]] + v : v;
    });
}

void wgrad(const uint16_t* dy16t, const uint16_t* x16t, int64_t T, int64_t N, int64_t K, float gscale, float* dW,
              float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream) {
    gemm.f16(dy16t, x16t, scratch, N, K, T);   // sum_t dY[t,n] X[t,k], the rows' scales folded in as scaled_transpose_f16 did
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) N * K), [=](sycl::id<1> id) {
        const float v = scratch[id[0]] * gscale;
        dW[id[0]] = accumulate ? dW[id[0]] + v : v;
    });
}

void bias_bwd(const float* dy, int64_t T, int64_t N, float* db, void* stream) {
    col_sum(T, N, db, [=](int64_t t, int64_t c) { return dy[t * N + c]; }, stream);
}

void transpose_f16(const uint16_t* in, int64_t rows, int64_t cols, uint16_t* out, void* stream) {
    constexpr int kT = 32;
    const size_t gr = (size_t) ((rows + kT - 1) / kT), gc = (size_t) ((cols + kT - 1) / kT);
    dit_rows::q_of(stream).submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint16_t, 2> t(sycl::range<2>(kT, kT + 1), cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(gr * kT, gc * kT), sycl::range<2>(kT, kT)),
                         [=](sycl::nd_item<2> it) {
                             const int64_t a = (int64_t) it.get_group(0) * kT, b = (int64_t) it.get_group(1) * kT;
                             const int i = (int) it.get_local_id(0), j = (int) it.get_local_id(1);
                             if (a + i < rows && b + j < cols) t[i][j] = in[(a + i) * cols + b + j];
                             sycl::group_barrier(it.get_group());
                             if (b + i < cols && a + j < rows) out[(b + i) * rows + a + j] = t[j][i];
                         });
    });
}

void mul_cols(const float* a, const float* col, int64_t n, int64_t C, float* out, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]);
        out[i] = a[i] * (col != nullptr ? col[i % C] : 1.f);
    });
}

void scale_by(float* a, int64_t n, const float* k, bool divide, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        a[id[0]] = divide ? a[id[0]] / k[0] : a[id[0]] * k[0];
    });
}

void v_scale(const float* v0, int64_t n0, const float* v1, int64_t n1, uint32_t* amax, float* vs, int64_t rows,
             void* s) {
    const int64_t n = n0 + n1, groups = std::min<int64_t>((n + kWG - 1) / kWG, 1024);
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (groups * kWG)), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            float m = 0.f;
            for (int64_t i = static_cast<int64_t>(it.get_global_linear_id()); i < n; i += groups * kWG)
                m = sycl::fmax(m, sycl::fabs(i < n0 ? v0[i] : v1[i - n0]));
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            if (it.get_local_linear_id() == 0)
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*amax).fetch_max(
                    sycl::bit_cast<uint32_t>(m));
        });
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (rows + 1)), [=](sycl::id<1> id) {
        vs[id[0]] = dit_rows::pow2_scale(sycl::bit_cast<float>(*amax));
    });
}

void copy_rows(const float* a, int64_t n, float* out, void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { out[id[0]] = a[id[0]]; });
}

void add_into(float* a, const float* b, int64_t n, void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { a[id[0]] += b[id[0]]; });
}

void f16_copy(const float* x, int64_t n, uint16_t* y, void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n),
                                        [=](sycl::id<1> id) { y[id[0]] = kernels::f16_from_f32(x[id[0]]); });
}

Pool::~Pool() {
    gpu::stream_sync(s_);
    for (void* p : owned_) gpu::free(p);
}

// the zero goes on the stream the block's kernels run on: the compute queue's own memset is not ordered against
// them, so a buffer allocated here could be wiped after a kernel had already written it (the late zero is what a
// gradient that reads as exactly zero, and a forward a few percent off, looked like)
void* Pool::take(int64_t n, int64_t bytes) {
    void* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) n * (size_t) bytes))
        throw std::runtime_error("image engine: GPU memory for a training block (" +
                                 std::to_string(((size_t) n * (size_t) bytes) >> 20) + " MiB)");
    owned_.push_back(p);
    if (!gpu::memset_async(p, 0, (size_t) n * (size_t) bytes, s_))
        throw std::runtime_error("image engine: zeroing a training block's buffer");
    return p;
}

}  // namespace strata::image::train
