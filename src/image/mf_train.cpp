// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_train.cpp - see include/strata/image/mf_train.hpp.  The gradients of Mage-Flow's DiT, read from the
// same tapes the inference's block keeps: FP32 where a gradient lives, FP16 only where a product reads one.  Nothing
// here is on the inference path: mf_dit.cpp is not touched, and these kernels take their own buffers.
#include "strata/image/mf_train.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

constexpr int64_t kWG = 256;   // a work-group: one over a row's columns, or over the rows of one column
constexpr float kEps = 1e-6f;  // the LayerNorm's and the q/k norms' eps, as mf_dit.cpp and dit_rows take it

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
        throw std::runtime_error("image engine (Mage-Flow training): reduction");
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
        throw std::runtime_error("image engine (Mage-Flow training): reduction read");
    gpu::free(part);
    float all = 0.f;
    for (float v : host) all = maximum ? std::max(all, v) : all + v;
    return all;
}

}  // namespace

void mf_lora_bwd(const float* hr, const float* xr, const uint16_t* h16t, const uint16_t* x16t,
                 const uint16_t* bt16, const float* dY, int64_t T, int64_t rows, int64_t cols, int64_t rank,
                 float scale, float* dB, float* dA, float* scratch, uint16_t* tmp16, float* rsT, prefill::Gemm& gemm,
                 void* stream) {
    float* dh = scratch;                  // [T][rank]
    float* sums = scratch + (int64_t) T * rank;
    uint16_t* dy16 = tmp16;               // [T][rows]
    uint16_t* dyt16 = dy16 + (int64_t) T * rows;    // [rows][T], dY's rows scaled by hr and transposed
    uint16_t* dht16 = dyt16 + (int64_t) rows * T;   // [rank][T], dh's rows scaled by xr and transposed
    float* rsY = rsT;                    // [T] dY's own row scales
    mf_rows16(dY, T, rows, dy16, rsY, stream);
    // dh = scale (dY B): the same product the forward read, with B's transposed view as the weight
    // dh is written whole here (every one of its T * rank values), so it is overwritten, not added into: the
    // caller passes scratch, which the probe leaves uninitialized (fresh device pages read as zero, which is the
    // only reason adding into it ever passed parity)
    mf_dgrad(dy16, rsY, bt16, T, rows, rank, dh, sums, false, gemm, stream);
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) T * rank), [=](sycl::id<1> id) {
        dh[id[0]] *= scale;
    });
    // dB = scale dY^T h, dA = scale dh^T x: both through one global scale on the gradient's rows
    const float g2 = mf_scaled_transpose_f16(dh, xr, T, rank, dht16, stream);
    const float g1 = mf_scaled_transpose_f16(dY, hr, T, rows, dyt16, stream);
    mf_wgrad(dyt16, h16t, T, rows, rank, g1 * scale, dB, sums, true, gemm, stream);
    // dA reads dh, which already carries scale (dL/dh = scale * dY B): no second scale here
    mf_wgrad(dht16, x16t, T, rank, cols, g2, dA, sums + (int64_t) rows * rank, true, gemm, stream);
}

void mf_lokr_partial_trace(const float* dDelta, const uint16_t* w1, const uint16_t* w2, int64_t out_l, int64_t out_k,
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

void mf_lokr_factor_bwd(const float* dw2, const uint16_t* w2a, const uint16_t* w2b, int64_t out_k, int64_t in_n,
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

void mf_dora_bwd(const float* dY, const float* p, const float* d, const float* m, const float* norms, int64_t T,
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

void mf_qk_norm_rope_bwd(const float* q, const float* k, const float* wq, const float* wk, const float* cos,
                         const float* sin, bool rope, const float* dgq, const float* dgk, int64_t rows, int64_t heads,
                         int64_t head_dim, float* dq, float* dk, float* wqn, float* wkn, void* stream) {
    const int64_t half = head_dim / 2, n = rows * heads;
    // each head-row's RMS inverse, q's and k's interleaved: rs[2 g + which], so TWO floats a head-row.  The
    // record's 6.6 (a GPU fault at some row counts) was this buffer sized as one: the write of k's inverse at
    // rs[2 g + 1] ran past the allocation, and whether the page behind it was mapped decided the hit or miss.
    float* rs = nullptr;
    const bool need_w = wqn != nullptr || wkn != nullptr;
    if (need_w && !gpu::alloc_device(&rs, (size_t) n * 2 * 4))
        throw std::runtime_error("image engine (Mage-Flow training): norm scales");
    dit_rows::q_of(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t g = static_cast<int64_t>(it.get_group_linear_id());
            const int64_t t = g / heads;
            for (int which = 0; which < 2; ++which) {
                const float* src = (which == 0 ? q : k) + g * head_dim;
                const float* dg = (which == 0 ? dgq : dgk) + g * head_dim;
                float* dst = (which == 0 ? dq : dk) + g * head_dim;
                const float* w = which == 0 ? wq : wk;
                // 1. the rotation undone: the pair (2j, 2j+1) turns back by its own angle
                float sa = 0.f;   // the head's sum of the weighted gradient times the source value
                for (int64_t j = static_cast<int64_t>(it.get_local_linear_id()); j < half; j += kWG) {
                    const float c = rope ? cos[t * half + j] : 1.f, s = rope ? sin[t * half + j] : 0.f;
                    const float ga = dg[2 * j] * c + dg[2 * j + 1] * s;
                    const float gb = -dg[2 * j] * s + dg[2 * j + 1] * c;
                    dst[2 * j] = ga * w[2 * j];
                    dst[2 * j + 1] = gb * w[2 * j + 1];
                    sa += ga * src[2 * j] * w[2 * j] + gb * src[2 * j + 1] * w[2 * j + 1];
                }
                sa = sycl::reduce_over_group(it.get_group(), sa, sycl::plus<float>());
                // 2. the RMS norm's own backward: y[c] = x[c] r w[c], r = 1 / sqrt(mean(x^2) + eps)
                float ss = 0.f;
                for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < head_dim; d += kWG)
                    ss += src[d] * src[d];
                const float r = 1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) /
                                                     (float) head_dim + kEps);
                for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < head_dim; d += kWG)
                    dst[d] = dst[d] * r - src[d] * (r * r * r / (float) head_dim) * sa;
                if (need_w && it.get_local_linear_id() == 0)
                    rs[2 * g + which] = r;   // the weights' sums read it back per (row, head, which)
            }
        });
    if (need_w) {
        // The weight multiplies the normed value before the rotation, so its gradient reads the de-rotated one.  rs
        // holds the two streams' RMS inverses interleaved per head-row (rs[2 g + which]).
        auto gsum = [&](const float* src, const float* dg, int which, float* out) {
            col_sum(n, head_dim, out,
                    [=](int64_t i, int64_t d) {
                        const int64_t t = i / heads, j = d / 2;
                        const float c = rope ? cos[t * half + j] : 1.f, s = rope ? sin[t * half + j] : 0.f;
                        const float self_ = dg[i * head_dim + d];
                        const float pair = dg[i * head_dim + (d % 2 == 0 ? d + 1 : d - 1)];
                        const float gv = d % 2 == 0 ? self_ * c + pair * s : self_ * c - pair * s;
                        return gv * src[i * head_dim + d] * rs[2 * i + which];
                    },
                    stream);
        };
        if (wqn != nullptr) gsum(q, dgq, 0, wqn);
        if (wkn != nullptr) gsum(k, dgk, 1, wkn);
        gpu::free(rs);
    }
}

// ---- the joint attention, tiled over the heads and read through the products (card 5) ----
//
// A work-group a (row, head) pair re-read every key from global memory, which was 88% of a block at real bucket
// sizes.  Here one head's rows are gathered contiguous and the scores are a product instead: S = Q K^T through
// prefill::Gemm (FP32 sums), the softmax in place, O = P V.  The backward recomputes P from Q and K with the same
// kernel (the same bits as the forward's) instead of keeping rows x rows per head.  Heads run one after the other
// over one workspace, so the peak is a few rows x rows maps.
//
// The products only take FP16, but the parity harness (tools/image/mf_train_ref.py, tol 2e-5 on o32) reads FP32:
// an FP32 map split as x = hi + lo (hi = f16(x), lo = f16(x - hi)) is exact to 2^-22, so a product with an FP32
// side runs as two FP16 products added in FP32.  The maps that are FP16 already (Q, K, V and their transposes) run
// as one.

namespace {

// one head's rows out of the joint [rows][heads * head_dim] rows into contiguous [rows][head_dim] (and back)
void attn_gather16(const uint16_t* src, int64_t rows, int64_t heads, int64_t hd, int64_t h, uint16_t* dst,
                   void* s) {
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group_linear_id());
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += kWG)
                dst[r * hd + d] = src[(r * heads + h) * hd + d];
        });
}

void attn_gather32(const float* src, int64_t rows, int64_t heads, int64_t hd, int64_t h, float* dst, void* s) {
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group_linear_id());
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += kWG)
                dst[r * hd + d] = src[(r * heads + h) * hd + d];
        });
}

// the head's output rows back into the joint rows, over the FP32 dump and the FP16 rows the next product reads
void attn_scatter_out(const float* src, int64_t rows, int64_t heads, int64_t hd, int64_t h, uint16_t* d16,
                       float* d32, void* s) {
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group_linear_id());
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += kWG) {
                const float v = src[r * hd + d];
                d16[(r * heads + h) * hd + d] = kernels::f16_from_f32(v);
                d32[(r * heads + h) * hd + d] = v;
            }
        });
}

// a gradient's head back into the joint rows (null: the stream takes no gradient there)
void attn_scatter_grad(const float* src, int64_t rows, int64_t heads, int64_t hd, int64_t h, float* dst, bool acc,
                       void* s) {
    if (dst == nullptr) return;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group_linear_id());
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += kWG) {
                float* o = dst + (r * heads + h) * hd + d;
                *o = acc ? *o + src[r * hd + d] : src[r * hd + d];
            }
        });
}

// the transposed gradient's head back ([head_dim][rows] into the joint rows)
void attn_scatter_grad_t(const float* src, int64_t rows, int64_t heads, int64_t hd, int64_t h, float* dst,
                         bool acc, void* s) {
    if (dst == nullptr) return;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group_linear_id());
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += kWG) {
                float* o = dst + (r * heads + h) * hd + d;
                *o = acc ? *o + src[d * rows + r] : src[d * rows + r];
            }
        });
}

// the scores scaled and softmaxed in place, with each row's log-sum-exp (null: not taken)
void attn_softmax(float* sp, int64_t rows, float inv_sqrt, float* lse, int64_t heads, int64_t h, void* s) {
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group_linear_id());
            float m = -1e30f;
            for (auto j = static_cast<int64_t>(it.get_local_linear_id()); j < rows; j += kWG)
                m = sycl::fmax(m, sp[r * rows + j] * inv_sqrt);
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            float l = 0.f;
            for (auto j = static_cast<int64_t>(it.get_local_linear_id()); j < rows; j += kWG) {
                const float e = sycl::exp(sp[r * rows + j] * inv_sqrt - m);
                sp[r * rows + j] = e;
                l += e;
            }
            l = sycl::reduce_over_group(it.get_group(), l, sycl::plus<float>());
            const float inv_l = 1.f / l;
            for (auto j = static_cast<int64_t>(it.get_local_linear_id()); j < rows; j += kWG) sp[r * rows + j] *= inv_l;
            if (it.get_local_linear_id() == 0 && lse != nullptr) lse[r * heads + h] = m + sycl::log(l);
        });
}

// x = hi + lo to 2^-22: hi the FP16 image, lo the FP16 image of what is left
void attn_split(const float* src, int64_t n, uint16_t* hi, uint16_t* lo, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = src[id[0]];
        const uint16_t h = kernels::f16_from_f32(v);
        hi[id[0]] = h;
        lo[id[0]] = kernels::f16_from_f32(v - kernels::f32_from_f16(h));
    });
}

// dst[j * rows + r] = src[r * rows + j]: the transposed map a gradient's product reads
void attn_transpose16(const uint16_t* src, int64_t rows, int64_t cols, uint16_t* dst, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) rows * cols), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), r = i / cols, c = i - r * cols;
        dst[c * rows + r] = src[i];
    });
}

// x = hi + lo * 2^-10 to 2^-21: the split above, but the difference is scaled up first so small values do not
// fall into FP16's subnormals (a softmax row at real sizes is ~1e-3, whose difference is ~5e-7: three mantissa bits).
// The products take lo10 and the caller adds 2^-10 back (attn_add_scaled).
void attn_split_scaled(const float* src, int64_t n, uint16_t* hi, uint16_t* lo10, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = src[id[0]];
        const uint16_t h = kernels::f16_from_f32(v);
        hi[id[0]] = h;
        lo10[id[0]] = kernels::f16_from_f32((v - kernels::f32_from_f16(h)) * 1024.f);
    });
}

// dst[i] += src[i] * s
void attn_add_scaled(float* dst, const float* src, int64_t n, float s, void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n),
                                        [=](sycl::id<1> id) { dst[id[0]] += src[id[0]] * s; });
}

// d[r] = sum_c a[r][c] b[r][c]
void attn_rowdot(const float* a, const float* b, int64_t rows, int64_t cols, float* d, void* s) {
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group_linear_id());
            float v = 0.f;
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < cols; c += kWG)
                v += a[r * cols + c] * b[r * cols + c];
            v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
            if (it.get_local_linear_id() == 0) d[r] = v;
        });
}

// dS = P (A - D) / sqrt(head_dim): A[r][j] = dO[r] V[j] is the products' (split) sum, D[r] = dO[r] O[r]
// the row's own output dot.  The 1 / sqrt(head_dim) is the score's own scale coming back: S = Q K^T / sqrt(hd).
void attn_mk_ds(const float* p, const float* a, const float* d, int64_t rows, float inv_sqrt, float* ds,
                void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) rows * rows), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]);
        ds[i] = p[i] * (a[i] - d[i / rows]) * inv_sqrt;
    });
}

}  // namespace

void mf_attn_fwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t heads,
                 int64_t head_dim, uint16_t* o16, float* o32, float* lse, prefill::Gemm& gemm, void* stream) {
    const float inv_sqrt = 1.f / std::sqrt((float) head_dim);
    const int64_t rr = rows * rows, rd = rows * head_dim, dh = head_dim * rows;
    // S/P (FP32), P's split (FP16 x2), one head's Q/K/V ([rows][head_dim] FP16), Vt ([head_dim][rows]) and O
    const int64_t n_sp = rr, n_o = rd, n_tmp = rd;
    const int64_t n_phi = rr, n_plo = rr, n_q = rd, n_k = rd, n_v = rd, n_vt = dh;
    const int64_t total = (n_sp + n_o + n_tmp) * 4 + (n_phi + n_plo + n_q + n_k + n_v + n_vt) * 2;
    uint8_t* ws = nullptr;
    if (!gpu::alloc_device(reinterpret_cast<void**>(&ws), (size_t) total))
        throw std::runtime_error("image engine (Mage-Flow training): attention workspace");
    float* sp = reinterpret_cast<float*>(ws);
    float* oh = reinterpret_cast<float*>(ws + n_sp * 4);
    float* tmp = reinterpret_cast<float*>(ws + (n_sp + n_o) * 4);
    uint16_t* phi = reinterpret_cast<uint16_t*>(ws + (n_sp + n_o + n_tmp) * 4);
    uint16_t* plo = phi + n_phi;
    uint16_t* qh = plo + n_plo;
    uint16_t* kh = qh + n_q;
    uint16_t* vh = kh + n_k;
    uint16_t* vt = vh + n_v;
    for (int64_t h = 0; h < heads; ++h) {
        attn_gather16(q16, rows, heads, head_dim, h, qh, stream);
        attn_gather16(k16, rows, heads, head_dim, h, kh, stream);
        attn_gather16(v16, rows, heads, head_dim, h, vh, stream);
        gemm.f16(qh, kh, sp, rows, rows, head_dim);
        attn_softmax(sp, rows, inv_sqrt, lse, heads, h, stream);
        attn_split_scaled(sp, rr, phi, plo, stream);
        attn_transpose16(vh, rows, head_dim, vt, stream);
        gemm.f16(phi, vt, oh, rows, head_dim, rows);
        gemm.f16(plo, vt, tmp, rows, head_dim, rows);
        attn_add_scaled(oh, tmp, rd, 1.f / 1024.f, stream);
        attn_scatter_out(oh, rows, heads, head_dim, h, o16, o32, stream);
    }
    gpu::free(ws);
}

void mf_attn_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const float* o32,
                 const float* dO, int64_t rows, int64_t heads, int64_t head_dim, float* dQ, float* dK, float* dV,
                 bool accumulate, prefill::Gemm& gemm, void* stream) {
    const float inv_sqrt = 1.f / std::sqrt((float) head_dim);
    const int64_t rr = rows * rows, rd = rows * head_dim, dh = head_dim * rows;
    // S/P and A/dS (FP32), the splits of P, dO and dS (FP16 x2 each), one head's Q/K/V/O/dO, the transposes
    // (Q/V/dO are [rows][head_dim], P/dS [rows][rows]) and one head's dQ/dKt/dVt
    const int64_t n_sp = rr, n_a = rr, n_o = rd, n_do = rd, n_dq = rd, n_dkt = dh, n_dvt = dh, n_d = rows;
    const int64_t n_phi = rr, n_plo = rr, n_dshi = rr, n_dslow = rr;
    const int64_t n_q = rd, n_k = rd, n_v = rd, n_dohi = rd, n_dolo = rd;
    const int64_t n_qt = dh, n_kt = dh, n_pthi = rr, n_ptlo = rr, n_dothi = dh, n_dotlo = dh;
    const int64_t total = (n_sp + n_a + n_o + n_do + n_dq + n_dkt + n_dvt + n_d) * 4 +
                          (n_phi + n_plo + n_dshi + n_dslow + n_q + n_k + n_v + n_dohi + n_dolo + n_qt + n_kt +
                           n_pthi + n_ptlo + n_dothi + n_dotlo) * 2;
    uint8_t* ws = nullptr;
    if (!gpu::alloc_device(reinterpret_cast<void**>(&ws), (size_t) total))
        throw std::runtime_error("image engine (Mage-Flow training): attention workspace");
    float* sp = reinterpret_cast<float*>(ws);
    float* ab = sp + n_sp;
    float* oh = ab + n_a;
    float* doh = oh + n_o;
    float* dqh = doh + n_do;
    float* dkt = dqh + n_dq;
    float* dvt = dkt + n_dkt;
    float* dd = dvt + n_dvt;
    uint16_t* phi = reinterpret_cast<uint16_t*>(ws + (n_sp + n_a + n_o + n_do + n_dq + n_dkt + n_dvt + n_d) * 4);
    uint16_t* plo = phi + n_phi;
    uint16_t* dshi = plo + n_plo;
    uint16_t* dslo = dshi + n_dshi;
    uint16_t* qh = dslo + n_dslow;
    uint16_t* kh = qh + n_q;
    uint16_t* vh = kh + n_k;
    uint16_t* dohi = vh + n_v;
    uint16_t* dolo = dohi + n_dohi;
    uint16_t* qt = dolo + n_dolo;
    uint16_t* kt = qt + n_qt;
    uint16_t* pthi = kt + n_kt;
    uint16_t* ptlo = pthi + n_pthi;
    uint16_t* dothi = ptlo + n_ptlo;
    uint16_t* dotlo = dothi + n_dothi;
    for (int64_t h = 0; h < heads; ++h) {
        attn_gather16(q16, rows, heads, head_dim, h, qh, stream);
        attn_gather16(k16, rows, heads, head_dim, h, kh, stream);
        attn_gather16(v16, rows, heads, head_dim, h, vh, stream);
        attn_gather32(o32, rows, heads, head_dim, h, oh, stream);
        attn_gather32(dO, rows, heads, head_dim, h, doh, stream);
        gemm.f16(qh, kh, sp, rows, rows, head_dim);          // S, recomputed: the same kernel, the same bits
        attn_softmax(sp, rows, inv_sqrt, nullptr, heads, h, stream);   // P over S
        attn_split(sp, rr, phi, plo, stream);
        attn_split(doh, rd, dohi, dolo, stream);
        gemm.f16(dohi, vh, ab, rows, rows, head_dim);       // A = dO V^T, split over dO
        gemm.f16(dolo, vh, ab, rows, rows, head_dim, 0, false, true);
        attn_rowdot(doh, oh, rows, head_dim, dd, stream);    // D[r] = dO[r] O[r], the FP32 dump's own dot
        attn_mk_ds(sp, ab, dd, rows, inv_sqrt, ab, stream);   // dS over A (P stays in S for dV below)
        attn_split(ab, rr, dshi, dslo, stream);
        attn_transpose16(kh, rows, head_dim, kt, stream);
        gemm.f16(dshi, kt, dqh, rows, head_dim, rows);       // dQ = dS K, split over dS
        gemm.f16(dslo, kt, dqh, rows, head_dim, rows, 0, false, true);
        attn_scatter_grad(dqh, rows, heads, head_dim, h, dQ, false, stream);
        attn_transpose16(qh, rows, head_dim, qt, stream);
        attn_transpose16(dshi, rows, rows, pthi, stream);    // dS^T into the P^T maps (dV has not used them yet)
        attn_transpose16(dslo, rows, rows, ptlo, stream);
        gemm.f16(pthi, qt, dkt, rows, head_dim, rows);       // dK = dS^T Q, split over dS
        gemm.f16(ptlo, qt, dkt, rows, head_dim, rows, 0, false, true);
        attn_scatter_grad(dkt, rows, heads, head_dim, h, dK, accumulate, stream);
        attn_transpose16(phi, rows, rows, pthi, stream);     // P^T, exact: the splits transpose exactly
        attn_transpose16(plo, rows, rows, ptlo, stream);
        attn_transpose16(dohi, rows, head_dim, dothi, stream);
        attn_transpose16(dolo, rows, head_dim, dotlo, stream);
        gemm.f16(dothi, pthi, dvt, head_dim, rows, rows);    // dV^T = dO^T P, split over both
        gemm.f16(dothi, ptlo, dvt, head_dim, rows, rows, 0, false, true);
        gemm.f16(dotlo, pthi, dvt, head_dim, rows, rows, 0, false, true);
        gemm.f16(dotlo, ptlo, dvt, head_dim, rows, rows, 0, false, true);
        attn_scatter_grad_t(dvt, rows, heads, head_dim, h, dV, accumulate, stream);
    }
    gpu::free(ws);
}
void mf_scale_rows(float* y, const float* r, int64_t rows, int64_t cols, void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) rows * cols), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]);
        y[i] *= r[i % cols];   // r is per column: y[t][c] *= r[c]
    });
}

void mf_rows16(const float* y, int64_t rows, int64_t cols, uint16_t* y16, float* rs, void* stream) {
    dit_rows::q_of(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t t = static_cast<int64_t>(it.get_group_linear_id());
            float m = 0.f;
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < cols; c += kWG)
                m = sycl::fmax(m, sycl::fabs(y[t * cols + c]));
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            const float sc = dit_rows::pow2_scale(m), inv = 1.f / sc;
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < cols; c += kWG)
                y16[t * cols + c] = kernels::f16_from_f32(y[t * cols + c] * inv);
            if (it.get_local_linear_id() == 0 && rs != nullptr) rs[t] = sc;
        });
}

void mf_ln_mod_bwd(const float* x, const float* mean, const float* inv, const float* scale, const float* dy,
                   int64_t rows, int64_t cols, float* dx, float* dshift, float* dscale, void* stream) {
    const float inv_c = 1.f / (float) cols;
    // dx[t][c] = inv (m - (S1 + xhat S2) / C), with m = dy (1 + scale) and xhat the normed row
    dit_rows::q_of(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t t = static_cast<int64_t>(it.get_group_linear_id());
            const float iv = inv[t], mn = mean[t];
            float s1 = 0.f, s2 = 0.f;
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < cols; c += kWG) {
                const float m = dy[t * cols + c] * (1.f + scale[c]);
                s1 += m;
                s2 += m * (x[t * cols + c] - mn) * iv;
            }
            s1 = sycl::reduce_over_group(it.get_group(), s1, sycl::plus<float>());
            s2 = sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>());
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < cols; c += kWG)
                dx[t * cols + c] = iv * (dy[t * cols + c] * (1.f + scale[c]) -
                                         (s1 + ((x[t * cols + c] - mn) * iv) * s2) * inv_c);
        });
    if (dshift != nullptr)
        col_sum(rows, cols, dshift, [=](int64_t t, int64_t c) { return dy[t * cols + c]; }, stream);
    if (dscale != nullptr)
        col_sum(rows, cols, dscale,
                [=](int64_t t, int64_t c) { return dy[t * cols + c] * (x[t * cols + c] - mean[t]) * inv[t]; },
                stream);
}

void mf_gelu_bwd(const float* v, const float* dy, int64_t n, float* dv, void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float z = v[id[0]];
        const float th = sycl::tanh(0.7978845608028654f * (z + 0.044715f * z * z * z));
        dv[id[0]] = dy[id[0]] * (0.5f * (1.f + th) +
                                 0.5f * z * (1.f - th * th) * 0.7978845608028654f * (1.f + 0.134145f * z * z));
    });
}

void mf_gate_add_bwd(const float* h, const float* gate, const float* dx, int64_t rows, int64_t cols, float* dh,
                     float* dgate, void* stream) {
    if (dh != nullptr)
        dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) rows * cols), [=](sycl::id<1> id) {
            const int64_t c = static_cast<int64_t>(id[0]) % cols;
            dh[id[0]] = (gate != nullptr ? gate[c] : 1.f) * dx[id[0]];
        });
    if (dgate != nullptr)
        col_sum(rows, cols, dgate, [=](int64_t t, int64_t c) { return dx[t * cols + c] * h[t * cols + c]; }, stream);
}

void mf_silu_bwd(const float* y, const float* d, int64_t n, float* dy, void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = y[id[0]], s = 1.f / (1.f + sycl::exp(-v));
        dy[id[0]] = d[id[0]] * s * (1.f + v * (1.f - s));
    });
}

void mf_patch_bwd(const float* m, const float* mean, const float* sd, int64_t h, int64_t w, const float* dz, float* dm,
                  float* dmean, float* dsd, void* stream) {
    const int64_t W = 2 * w;
    if (dm != nullptr)   // f2_patch's gather, with the batch norm's division undone
        dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) (4 * h * w * 32)), [=](sycl::id<1> id) {
            const auto i = static_cast<int64_t>(id[0]);
            const int64_t p = i / 32, c = i - p * 32, Y = p / W, X = p - Y * W;
            const int64_t k = 4 * c + 2 * (Y % 2) + X % 2;
            dm[i] = dz[((Y / 2) * w + X / 2) * 128 + k] / sd[k];
        });
    if (dmean != nullptr)
        col_sum(h * w, 128, dmean, [=](int64_t t, int64_t k) { return -dz[t * 128 + k] / sd[k]; }, stream);
    if (dsd != nullptr)
        col_sum(h * w, 128, dsd,
                [=](int64_t t, int64_t k) {
                    const int64_t c = k / 4, ay = (k % 4) / 2, ax = k % 2;
                    const int64_t p = (2 * (t / w) + ay) * W + 2 * (t % w) + ax;
                    return -dz[t * 128 + k] * (m[p * 64 + c] - mean[k]) / (sd[k] * sd[k]);
                },
                stream);
}

float mf_flow_loss(const float* v_pred, const float* x0, const float* eps, int64_t n, float* dv_pred, float* dx0,
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

void mf_transpose_f16(const uint16_t* in, int64_t rows, int64_t cols, uint16_t* out, void* stream) {
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) rows * cols), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), r = i / cols, c = i - r * cols;
        out[c * rows + r] = in[i];
    });
}

float mf_scaled_transpose_f16(const float* g, const float* rsX, int64_t rows, int64_t cols, uint16_t* out,
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

void mf_dgrad(const uint16_t* dy16, const float* rsY, const uint16_t* wt, int64_t T, int64_t N, int64_t K, float* dX,
              float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream) {
    gemm.f16(dy16, wt, scratch, T, K, N);   // wt[K,N] holds W[N,K]'s rows: the product is dY . W, in FP32 sums
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) T * K), [=](sycl::id<1> id) {
        const int64_t t = static_cast<int64_t>(id[0]) / K;
        const float v = scratch[id[0]] * (rsY != nullptr ? rsY[t] : 1.f);
        dX[id[0]] = accumulate ? dX[id[0]] + v : v;
    });
}

void mf_wgrad(const uint16_t* dy16t, const uint16_t* x16t, int64_t T, int64_t N, int64_t K, float gscale, float* dW,
              float* scratch, bool accumulate, prefill::Gemm& gemm, void* stream) {
    gemm.f16(dy16t, x16t, scratch, N, K, T);   // sum_t dY[t,n] X[t,k], the rows' scales folded in as mf_scaled... did
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) N * K), [=](sycl::id<1> id) {
        const float v = scratch[id[0]] * gscale;
        dW[id[0]] = accumulate ? dW[id[0]] + v : v;
    });
}

void mf_bias_bwd(const float* dy, int64_t T, int64_t N, float* db, void* stream) {
    col_sum(T, N, db, [=](int64_t t, int64_t c) { return dy[t * N + c]; }, stream);
}

}  // namespace strata::image
