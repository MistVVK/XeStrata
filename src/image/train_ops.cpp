// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/train_ops.cpp - see include/strata/image/train_ops.hpp.  FP32 where a gradient lives, FP16 only where a
// product reads one; nothing here is on the inference path, and these kernels take their own buffers.
#include "strata/image/train_ops.hpp"

#include "strata/core/gpu.hpp"
#include "strata/platform/env.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>
#include <map>

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

/// out [rows][cols] FP16 = f(i, t) / G for every value i (row t), G the power of two that puts the largest |f| within
/// FP16's range (pow2_full, into gbits): the rows' own scales folded into a product's operand as a whole
template <class F>
inline void scaled_rows16(F f, int64_t rows, int64_t cols, uint16_t* out, uint32_t* gbits, void* stream) {
    const int64_t n = rows * cols, groups = std::min<int64_t>((n + kWG - 1) / kWG, 1024);
    gpu::memset_async(gbits, 0, 4, stream);
    dit_rows::q_of(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (groups * kWG)), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            float m = 0.f;
            for (int64_t i = (int64_t) it.get_global_linear_id(); i < n; i += groups * kWG)
                m = sycl::fmax(m, sycl::fabs(f(i, i / cols)));
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            if (it.get_local_linear_id() == 0)   // non-negative floats order as their bits
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*gbits).fetch_max(
                    sycl::bit_cast<uint32_t>(m));
        });
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]);
        out[i] = kernels::f16_from_f32(f(i, i / cols) / dit_rows::pow2_full(sycl::bit_cast<float>(*gbits)));
    });
}

/// dW (+)= src * mul * G, G the power of two in gbits (1 without)
inline void add_gscaled(const float* src, int64_t n, const uint32_t* gbits, float mul, float* dW, bool accumulate,
                        void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float gscale = mul * (gbits != nullptr ? dit_rows::pow2_full(sycl::bit_cast<float>(*gbits)) : 1.f);
        const float v = src[id[0]] * gscale;
        dW[id[0]] = accumulate ? dW[id[0]] + v : v;
    });
}

}  // namespace

void lora_bwd(const float* hr, const float* xr, const uint16_t* h16t, const uint16_t* x16t,
                 const uint16_t* bt16, const float* dY, int64_t T, int64_t rows, int64_t cols, int64_t rank,
                 float scale, float* dB, float* dA, float* scratch, uint16_t* tmp16, float* rsT, prefill::Gemm& gemm,
                 void* stream, const uint16_t* dy16_done, const float* rsY_done, const uint16_t* h16,
                 const uint16_t* x16, const uint16_t* b16) {
    float* dh = scratch;                  // [T][rank]
    float* sums = scratch + (int64_t) T * rank;
    uint16_t* dy16 = tmp16;               // [T][rows]
    // The caller's dgrad already made dY's rows for a plain LoRA: reuse them and omit their slot in tmp16.
    uint16_t* dyt16 = tmp16 + (dy16_done != nullptr ? 0 : (int64_t) T * rows);   // [rows][T]
    uint16_t* dht16 = dyt16 + (int64_t) rows * T;   // [rank][T], dh's rows scaled by xr and transposed
    float* rsY = rsT;                    // [T] dY's own row scales
    if (dy16_done != nullptr) {   // the caller's rows of the same dY (its own dgrad read them)
        dy16 = const_cast<uint16_t*>(dy16_done);
        rsY = const_cast<float*>(rsY_done);
    } else {
        dit_rows::rows16(dit_rows::Grad{{dY, rows}}, T, rows, dy16, rsY, stream);
    }
    // dh = scale (dY B): the same product the forward read, with B's transposed view as the weight
    // dh is written whole here (every one of its T * rank values), so it is overwritten, not added into: the
    // caller passes scratch, which the probe leaves uninitialized (fresh device pages read as zero, which is the
    // only reason adding into it ever passed parity)
    if (bt16 != nullptr) dgrad(dy16, rsY, bt16, T, rows, rank, dh, sums, false, gemm, stream);
    else dgrad_rows(dy16, rsY, b16, T, rows, rank, dh, sums, false, gemm, stream);
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) T * rank), [=](sycl::id<1> id) {
        dh[id[0]] *= scale;
    });
    uint32_t *g2 = scale_word(0, stream), *g1 = scale_word(1, stream);
    if (h16 != nullptr && x16 != nullptr && gemm.f16_batch_ex_ok()) {
        // dB = scale dY^T h from dY's own FP16 rows read transposed, dY's row scales folded into h's rows (and h's
        // own, hr) under one power of two; dA = dh^T x the same, x's row scales folded into dh's.  Nothing of
        // rows x T is written again: transposing dY and its product of 32 columns took 4.5 of the B70's 10.6 ms of
        // gate_up's backward at 2,400 rows (2026-11-09)
        uint16_t* hs = dyt16;    // [T][rank]
        uint16_t* dhs = dht16;   // [T][rank]
        const float* rY = rsY;
        scaled_rows16([=](int64_t i, int64_t t) { return kernels::f32_from_f16(h16[i]) * hr[t] * rY[t]; }, T, rank,
                      hs, g1, stream);
        scaled_rows16([=](int64_t i, int64_t t) { return dh[i] * (xr != nullptr ? xr[t] : 1.f); }, T, rank, dhs, g2,
                      stream);
        float* sA = sums + (int64_t) rows * rank;
        // (a batch of one: the strides are the operands' sizes, oneMKL refuses 0)
        if (!gemm.f16_batch_ex(true, false, dy16, hs, sums, rows, rank, T, 1, T * rows, T * rank, rows * rank, rows,
                               rank, rank, 0.f) ||
            !gemm.f16_batch_ex(true, false, dhs, x16, sA, rank, cols, T, 1, T * rank, T * cols, rank * cols, rank,
                               cols, cols, 0.f))
            // the callers left h16t and x16t unmade (ad_needs_xt): no way back to the transposed products
            throw std::runtime_error("image engine (training): a LoRA's transposed products failed");
        add_gscaled(sums, rows * rank, g1, scale, dB, true, stream);
        // dA reads dh, which already carries scale (dL/dh = scale * dY B): no second scale here
        add_gscaled(sA, rank * cols, g2, 1.f, dA, true, stream);
        return;
    }
    // dB = scale dY^T h, dA = scale dh^T x: both through one global scale on the gradient's rows
    scaled_transpose_f16(dh, xr, T, rank, dht16, g2, stream);
    scaled_transpose_f16(dY, hr, T, rows, dyt16, g1, stream);
    wgrad(dyt16, h16t, T, rows, rank, g1, scale, dB, sums, true, gemm, stream);
    // dA reads dh, which already carries scale (dL/dh = scale * dY B): no second scale here
    wgrad(dht16, x16t, T, rank, cols, g2, 1.f, dA, sums + (int64_t) rows * rank, true, gemm, stream);
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
    // PEFT's DoRA at strength 1: y_r = c_r p_r + scale c_r d_r with c_r = m_r / (n_r + eps) and n the composed norm
    // detached, so dm reads (p + scale d) / (n + eps) and the factors carry no norm gradient
    if (dm != nullptr)
        col_sum(T, rows, dm,
                [=](int64_t t, int64_t r) {
                    return dY[t * rows + r] * (p[t * rows + r] + scale * d[t * rows + r]) / (norms[r] + kDoraEps);
                },
                stream);
    if (da == nullptr && db == nullptr) return;
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) rows), [=](sycl::id<1> id) {
        const int64_t r = static_cast<int64_t>(id[0]);
        const float cr = m[r] / (norms[r] + kDoraEps);
        if (da != nullptr) da[r] = cr;
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

void scaled_transpose_f16(const float* g, const float* rsX, int64_t rows, int64_t cols, uint16_t* out, uint32_t* gbits,
                          void* stream) {
    const int64_t n = rows * cols, groups = std::min<int64_t>((n + kWG - 1) / kWG, 1024);
    gpu::memset_async(gbits, 0, 4, stream);
    dit_rows::q_of(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (groups * kWG)), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            float m = 0.f;
            for (int64_t i = (int64_t) it.get_global_linear_id(); i < n; i += groups * kWG)
                m = sycl::fmax(m, sycl::fabs(g[i] * (rsX != nullptr ? rsX[i / cols] : 1.f)));
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            if (it.get_local_linear_id() == 0)   // non-negative floats order as their bits
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*gbits).fetch_max(
                    sycl::bit_cast<uint32_t>(m));
        });
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / cols, c = i - t * cols;
        const float G = dit_rows::pow2_full(sycl::bit_cast<float>(*gbits));
        out[c * rows + t] = kernels::f16_from_f32(g[i] * (rsX != nullptr ? rsX[t] : 1.f) / G);
    });
}

float scale_of(const uint32_t* gbits, void* stream) {
    uint32_t b = 0;
    gpu::stream_sync(stream);
    if (!gpu::copy(&b, gbits, 4)) throw std::runtime_error("image engine (training): a scale read back");
    return dit_rows::pow2_full(sycl::bit_cast<float>(b));
}

uint32_t* scale_word(int k, void* stream) {
    static std::mutex mu;
    static std::map<void*, uint32_t*> words;
    std::lock_guard<std::mutex> lock(mu);
    uint32_t*& w = words[stream];
    if (w == nullptr && !gpu::alloc_device(&w, 2 * sizeof(uint32_t)))
        throw std::runtime_error("image engine (training): the scales' words");
    return w + k;
}

namespace {

// dX (+)= the rows' sums times their scales
void dgrad_rows_back(const float* scratch, const float* rsY, int64_t T, int64_t K, float* dX, bool accumulate,
                     void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) T * K), [=](sycl::id<1> id) {
        const int64_t t = static_cast<int64_t>(id[0]) / K;
        const float v = scratch[id[0]] * (rsY != nullptr ? rsY[t] : 1.f);
        dX[id[0]] = accumulate ? dX[id[0]] + v : v;
    });
}

}  // namespace

void dgrad(const uint16_t* dy16, const float* rsY, const uint16_t* wt, int64_t T, int64_t N, int64_t K, float* dX,
              float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream) {
    float* sums = accumulate ? scratch : dX;   // when overwriting, the elementwise scaling can run in place
    gemm.f16(dy16, wt, sums, T, K, N);   // wt[K,N] holds W[N,K]'s rows: the product is dY . W, in FP32 sums
    dgrad_rows_back(sums, rsY, T, K, dX, accumulate, stream);
}

void dgrad_rows(const uint16_t* dy16, const float* rsY, const uint16_t* w, int64_t T, int64_t N, int64_t K, float* dX,
                float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream) {
    // dY [T][N] . W [N][K], W as it is (a batch of one: the strides are the operands' sizes, oneMKL refuses 0)
    float* sums = accumulate ? scratch : dX;
    if (!gemm.f16_batch_ex(false, false, dy16, w, sums, T, K, N, 1, T * N, N * K, T * K, N, K, K, 0.f))
        throw std::runtime_error("image engine (training): a dgrad's product on the weight's rows failed");
    dgrad_rows_back(sums, rsY, T, K, dX, accumulate, stream);
}

void wgrad(const uint16_t* dy16t, const uint16_t* x16t, int64_t T, int64_t N, int64_t K, const uint32_t* gbits,
           float mul, float* dW, float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream) {
    // sum_t dY[t,n] X[t,k], the rows' scales folded in as scaled_transpose_f16 did
    gemm.f16(dy16t, x16t, scratch, N, K, T);
    add_gscaled(scratch, N * K, gbits, mul, dW, accumulate, stream);
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

void f16_copy_full(const float* x, int64_t n, uint16_t* y, uint32_t* gbits, void* stream) {
    const int64_t groups = std::min<int64_t>((n + kWG - 1) / kWG, 1024);
    gpu::memset_async(gbits, 0, 4, stream);
    dit_rows::q_of(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (groups * kWG)), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            float m = 0.f;
            for (int64_t i = (int64_t) it.get_global_linear_id(); i < n; i += groups * kWG)
                m = sycl::fmax(m, sycl::fabs(x[i]));
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            if (it.get_local_linear_id() == 0)   // non-negative floats order as their bits
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*gbits).fetch_max(
                    sycl::bit_cast<uint32_t>(m));
        });
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float G = dit_rows::pow2_full(sycl::bit_cast<float>(*gbits));
        y[id[0]] = kernels::f16_from_f32(x[id[0]] / G);
    });
}

void bf16_keep(float* x, int64_t n, uint16_t* y, void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const uint16_t b = dit_rows::bf16_bits(x[id[0]]);
        y[id[0]] = b;
        x[id[0]] = dit_rows::resid(b);
    });
}

void bf16_load(const uint16_t* y, int64_t n, float* x, void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n),
                                        [=](sycl::id<1> id) { x[id[0]] = dit_rows::resid(y[id[0]]); });
}

void w8_quantize(const uint16_t* w16, int64_t N, int64_t K, int8_t* q, float* s, void* stream) {
    if (K % kQiRotate != 0) throw std::invalid_argument("image engine: a QLoRA base's rows are not 256-wide groups");
    dit_rows::rows_i8(dit_rows::HalfI8<true>{w16, K}, N, K, q, s, stream);
}

bool w8_int8(void* stream) { return prefill::Gemm::i8_usable(stream); }

void w8_dequant_rows(const int8_t* q, const float* s, int64_t N, int64_t K, uint16_t* tmp, void* stream) {
    // a work-group: 4 rows of one group of 256, each rotated back (the regular Hadamard is its own inverse)
    constexpr int kG = 256, kRowsWG = 4;
    const size_t gn = (size_t) ((N + kRowsWG - 1) / kRowsWG), gk = (size_t) (K / kG);
    dit_rows::q_of(stream).submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> v(sycl::range<1>((size_t) kRowsWG * kG), cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(gn * kRowsWG, gk * kG), sycl::range<2>(kRowsWG, kG)),
                         [=](sycl::nd_item<2> it) {
                             const int r = (int) it.get_local_id(0), j = (int) it.get_local_id(1);
                             const int64_t n = (int64_t) it.get_global_id(0), k = (int64_t) it.get_global_id(1);
                             float* vr = v.get_multi_ptr<sycl::access::decorated::no>().get() + (ptrdiff_t) r * kG;
                             vr[j] = n < N ? (float) q[n * K + k] * s[n] : 0.f;
                             const float y = dit_rows::regular256(vr, j, it.get_group());
                             if (n < N) tmp[n * K + k] = dit_rows::f16_from_f32(y);
                         });
    });
}

void w8_dequant_t(const int8_t* q, const float* s, int64_t N, int64_t K, uint16_t* tmp, uint16_t* wt, void* stream) {
    w8_dequant_rows(q, s, N, K, tmp, stream);
    transpose_f16(tmp, N, K, wt, stream);
}

void w8_product(const uint16_t* x16, int64_t T, const Product& P, float* y, int8_t* x8, float* sx, uint16_t* x16f,
                uint16_t* w16f, prefill::Gemm& gemm, void* stream) {
    const int64_t N = P.N, K = P.K;
    dit_rows::rows_i8(dit_rows::HalfI8<true>{x16, K}, T, K, x8, sx, stream);
    if (w8_int8(stream)) {
        prefill::Gemm::w8a8(x8, sx, P.w8, P.sw, nullptr, y, T, N, K, N, false, stream);
        kernels::i8_scale(y, T, N, N, sx, P.sw, nullptr, nullptr, stream);
        return;
    }
    // no int8 products: the same int8 values (the input's and the weight's, both in the rotated basis) as FP16
    const int8_t* w8 = P.w8;
    const float* sw = P.sw;
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) (T * K)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        x16f[i] = kernels::f16_from_f32((float) x8[i] * sx[i / K]);
    });
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) (N * K)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        w16f[i] = kernels::f16_from_f32((float) w8[i] * sw[i / K]);
    });
    gemm.f16(x16f, w16f, y, T, N, K);
}

Arena::~Arena() {
    for (void* p : spill_) gpu::free(p);
    if (base_ != nullptr && owned_) gpu::free(base_);
}

void Arena::attach(void* base, size_t cap) {
    if (base_ != nullptr && owned_) gpu::free(base_);
    base_ = static_cast<uint8_t*>(base);
    cap_ = need_ = cap;
    used_ = 0;
    owned_ = false;
    taken_.clear();
    replay_ = false;
}

void Arena::reset(void* stream) {
    taken_.clear();
    replay_ = false;
    need_ = std::max(need_, used_);
    if (spill_.empty() && need_ <= cap_) {
        used_ = 0;
        return;
    }
    gpu::stream_sync(stream);   // the last call's kernels still read the buffers freed here
    for (void* p : spill_) gpu::free(p);
    spill_.clear();
    if (!owned_ && need_ <= cap_) {   // the caller's memory: what a call took past it went to spills alone
        used_ = 0;
        return;
    }
    if (base_ != nullptr && owned_) gpu::free(base_);
    owned_ = true;
    base_ = nullptr;
    cap_ = 0;
    if (!gpu::alloc_device(reinterpret_cast<void**>(&base_), need_))
        throw std::runtime_error("image engine: GPU memory for a training block (" + std::to_string(need_ >> 20) +
                                 " MiB)");
    cap_ = need_;
    used_ = 0;
}

void Arena::rewind() {
    replay_ = true;
    next_ = 0;
}

void Arena::release(void* stream) {
    gpu::stream_sync(stream);
    for (void* p : spill_) gpu::free(p);
    spill_.clear();
    if (base_ != nullptr && owned_) gpu::free(base_);
    base_ = nullptr;
    owned_ = true;
    cap_ = used_ = need_ = 0;
    taken_.clear();
    replay_ = false;
    next_ = 0;
}

void* Arena::take(size_t bytes) {
    bytes = (bytes + 255) / 256 * 256;
    if (replay_) {
        if (next_ >= taken_.size() || taken_[next_].second != bytes)
            throw std::logic_error("image engine: a training block's tape read back in another order");
        return taken_[next_++].first;
    }
    void* p = nullptr;
    if (used_ + bytes <= cap_) {
        p = base_ + used_;
    } else {   // past the buffer: a buffer of its own until the next reset makes the arena large enough
        if (!gpu::alloc_device(&p, bytes))
            throw std::runtime_error("image engine: GPU memory for a training block (" + std::to_string(bytes >> 20) +
                                     " MiB)");
        spill_.push_back(p);
    }
    used_ += bytes;
    taken_.emplace_back(p, bytes);
    return p;
}

Pool::~Pool() {
    if (arena_ != nullptr) return;   // the arena keeps them for the next call
    gpu::stream_sync(s_);
    for (void* p : owned_) gpu::free(p);
}

// the zero goes on the stream the block's kernels run on: the compute queue's own memset is not ordered against
// them, so a buffer allocated here could be wiped after a kernel had already written it (the late zero is what a
// gradient that reads as exactly zero, and a forward a few percent off, looked like)
void* Pool::take(int64_t n, int64_t bytes, bool zero) {
    void* p = nullptr;
    if (arena_ != nullptr) {
        p = arena_->take((size_t) n * (size_t) bytes);
    } else {
        if (!gpu::alloc_device(&p, (size_t) n * (size_t) bytes))
            throw std::runtime_error("image engine: GPU memory for a training block (" +
                                     std::to_string(((size_t) n * (size_t) bytes) >> 20) + " MiB)");
        owned_.push_back(p);
    }
    static const bool all_zero = strata::env_get("POOL_ZERO") != nullptr;   // a check: every buffer zeroed
    if (arena_ != nullptr && arena_->replaying()) return p;   // the last call's values, kept
    if ((zero || all_zero) && !gpu::memset_async(p, 0, (size_t) n * (size_t) bytes, s_))
        throw std::runtime_error("image engine: zeroing a training block's buffer");
    return p;
}

// h = A x for a one-row input (A [rank][K]): a work-group a rank
void row_down(const float* A, const float* x, int64_t rank, int64_t K, float* h, void* s) {
    constexpr int64_t WG = 256;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rank * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group_linear_id());
            float acc = 0.f;
            for (auto k = static_cast<int64_t>(it.get_local_linear_id()); k < K; k += WG) acc += A[r * K + k] * x[k];
            acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
            if (it.get_local_linear_id() == 0) h[r] = acc;
        });
}

// a LoRA on a one-row product (a modulation linear's): y += scale B (A x), h the rank's scratch
void row_lora_fwd(const float* A, const float* B, const float* x, float scale, int64_t rank, int64_t N, int64_t K,
                  float* y, float* h, void* s) {
    row_down(A, x, rank, K, h, s);
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) N), [=](sycl::id<1> id) {
        const int64_t n = static_cast<int64_t>(id[0]);
        float acc = 0.f;
        for (int64_t r = 0; r < rank; ++r) acc += B[n * rank + r] * h[r];
        y[n] += scale * acc;
    });
}

// its gradients from the modulation's dy: dB += scale dy h^T, dA += (scale B^T dy) x^T (h and g the rank's scratch)
void row_lora_bwd(const float* A, const float* B, const float* x, const float* dy, float scale, int64_t rank,
                  int64_t N, int64_t K, float* dA, float* dB, float* h, float* g, void* s) {
    constexpr int64_t WG = 256;
    row_down(A, x, rank, K, h, s);
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (N * rank)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), n = i / rank, r = i - n * rank;
        dB[i] += scale * dy[n] * h[r];
    });
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rank * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group_linear_id());
            float acc = 0.f;
            for (auto n = static_cast<int64_t>(it.get_local_linear_id()); n < N; n += WG) acc += B[n * rank + r] * dy[n];
            acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
            if (it.get_local_linear_id() == 0) g[r] = scale * acc;
        });
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (rank * K)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), r = i / K, k = i - r * K;
        dA[i] += g[r] * x[k];
    });
}

StageTimes::StageTimes(const char* name) : name_(name), on_(strata::env_get("TRAIN_PROFILE") != nullptr) {}

AdScratch take_ad_scratch(Pool& pool, const AdSizes& s) {
    AdScratch sc;
    if (s.n == 0) return sc;
    sc.hf = pool.rf(s.rows * s.rank);
    sc.y1 = pool.rf(s.map);
    sc.y2 = !s.plain ? pool.rf(s.map) : nullptr;
    sc.fa = pool.rf(s.n);
    sc.fb = pool.rf(s.n);
    sc.scratch = pool.rf(s.scratch);
    sc.tmp16 = pool.ru(s.tmp);
    sc.rsT = pool.rf(s.rows);
    return sc;
}


namespace {
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

void StageTimes::begin(void* stream) {
    if (!on_) return;
    gpu::stream_sync(stream);
    last_ = now_s();
    ++calls_;
}

void StageTimes::mark(const char* stage, void* stream) {
    if (!on_) return;
    gpu::stream_sync(stream);
    const double t = now_s();
    auto it = std::find_if(t_.begin(), t_.end(), [&](const auto& e) { return std::strcmp(e.first, stage) == 0; });
    if (it == t_.end()) t_.emplace_back(stage, t - last_);
    else it->second += t - last_;
    last_ = t;
}

void StageTimes::report() {
    if (!on_ || calls_ == 0) return;
    double sum = 0.0;
    for (const auto& e : t_) sum += e.second;
    std::fprintf(stderr, "%s over %lld calls, %.2f s:", name_, (long long) calls_, sum);
    for (const auto& e : t_) std::fprintf(stderr, " %s %.2f s (%.0f%%),", e.first, e.second, 100.0 * e.second / sum);
    std::fprintf(stderr, "\n");
    t_.clear();
    calls_ = 0;
}

}  // namespace strata::image::train
