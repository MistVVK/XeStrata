// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/gdn.cpp - the Xe port of Strata's src/kernels/cuda/gdn.cu: the gated delta-net recurrence, its causal
// convolution, and the norms around it.  Work-group shapes are the CUDA kernels'; a warp is a sub-group of 32, and
// lane 0 of its xor butterfly adds in the order of CUDA's shuffle-down tree.
#include "strata/kernels/gdn.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"

#include <cmath>
#include <string>

namespace strata::kernels {
namespace {

constexpr int JTHREADS = 32;   ///< work-items along j, the state's fast axis
constexpr int MAX_H = 1;       ///< heads staged in local memory per work-group
constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

inline float sigmoid_f(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

template <typename D>
inline D warp_sum(const sycl::sub_group& sg, D v) {
    for (int off = 16; off > 0; off >>= 1) v += sycl::permute_group_by_xor(sg, v, off);
    return v;
}

}  // namespace

void gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
              const float* beta, float* o, const GdnShapes& s, void* stream) {
    if (s.S <= 0 || s.h_k <= 0 || s.h_v <= 0) return;
    if (s.S > 128) throw core::DeviceError("gdn_step: S = " + std::to_string(s.S) + " exceeds the staged 128");
    const int S = (int) s.S, h_k = (int) s.h_k, h_v = (int) s.h_v;
    const size_t gx = (size_t) ((S + JTHREADS - 1) / JTHREADS), gy = (size_t) ((h_v + MAX_H - 1) / MAX_H);
    const auto e = queue_for(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> ks(sycl::range<1>(MAX_H * 128), hd);
        sycl::local_accessor<float, 1> qs(sycl::range<1>(MAX_H * 128), hd);
        sycl::local_accessor<float, 1> dec_s(sycl::range<1>(MAX_H), hd);
        sycl::local_accessor<float, 1> beta_s(sycl::range<1>(MAX_H), hd);
        hd.parallel_for(sycl::nd_range<2>({gy, gx * JTHREADS}, {1, JTHREADS}), [=](sycl::nd_item<2> it) {
            const int tx = (int) it.get_local_id(1);
            const int h0 = (int) it.get_group(0) * MAX_H;
            const int nh = sycl::min(MAX_H, h_v - h0);
            const int j = (int) it.get_group(1) * JTHREADS + tx;
            for (int hi = 0; hi < nh; ++hi) {
                const int h = h0 + hi;
                const int src = h % h_k;                       // MODULO head pairing
                for (int i = tx; i < S; i += JTHREADS) {
                    ks[hi * 128 + i] = k[src * S + i];
                    qs[hi * 128 + i] = q[src * S + i];
                }
                if (tx == 0) {
                    dec_s[hi] = sycl::exp(gate[h]);
                    beta_s[hi] = beta[h];
                }
            }
            sycl::group_barrier(it.get_group());
            if (j >= S) return;
            for (int hi = 0; hi < nh; ++hi) {
                const int h = h0 + hi;
                const float dec = dec_s[hi], b = beta_s[hi];
                float* col = state + (size_t) h * S + j;        // (S, h_v, S) with j fastest
                const size_t stride = (size_t) h_v * S;
                float sk = 0.0f;
                for (int i = 0; i < S; ++i) {
                    const float sv = col[(size_t) i * stride] * dec;
                    col[(size_t) i * stride] = sv;
                    sk += sv * ks[hi * 128 + i];
                }
                const float d = (v[(size_t) h * S + j] - sk) * b;
                float dot = 0.0f;
                for (int i = 0; i < S; ++i) {
                    const float sv = col[(size_t) i * stride] + ks[hi * 128 + i] * d;
                    col[(size_t) i * stride] = sv;
                    dot += sv * qs[hi * 128 + i];
                }
                o[(size_t) h * S + j] = dot;
            }
        });
    });
    finish(stream, e, "gdn_step");
}

void gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int64_t channels,
                   int64_t d_conv, void* stream) {
    if (channels <= 0 || d_conv < 1) return;
    const int C = (int) channels, dc = (int) d_conv;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        float* st = conv_state + (size_t) c * (dc - 1);
        const float* w = kW + (size_t) c * dc;
        float acc = 0.0f;
        for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];   // kernel[0] reads the OLDEST state row
        acc += x[c] * w[dc - 1];                                // the new input lands in the LAST tap
        out[c] = acc;
        for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];     // slide: drop the oldest, append the newest
        st[dc - 2] = x[c];
    });
    finish(stream, e, "gdn_conv_step");
}

namespace {
// D: double where the device has FP64 (the reference's sums), float elsewhere (device_caps.hpp)
template <typename D>
sycl::event gdn_l2_norm_t(sycl::queue& q, float* x, int64_t rows, int ncols, float eps) {
    return q.parallel_for(sycl::nd_range<1>((size_t) rows * WARP, WARP),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        float* p = x + it.get_group(0) * (size_t) ncols;
        const int t = (int) it.get_local_id(0);
        D acc = 0;
        for (int i = t; i < ncols; i += WARP) acc += (D) p[i] * (D) p[i];
        const D ssum = sycl::group_broadcast(sg, warp_sum(sg, acc), 0);
        const float inv = (float) ((D) 1 / sycl::sqrt(ssum + (D) eps));
        for (int i = t; i < ncols; i += WARP) p[i] *= inv;
    });
}
template <typename D>
sycl::event gdn_out_norm_t(sycl::queue& q, const float* o, const float* z, const float* ssm_norm, float* y,
                           int64_t h_v, int Si, float eps) {
    return q.parallel_for(sycl::nd_range<1>((size_t) h_v * WARP, WARP),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const size_t h = it.get_group(0);
        const float* po = o + h * Si;
        const float* pz = z + h * Si;
        float* py = y + h * Si;
        const int t = (int) it.get_local_id(0);
        D acc = 0;
        for (int i = t; i < Si; i += WARP) acc += (D) po[i] * (D) po[i];
        const D ssum = sycl::group_broadcast(sg, warp_sum(sg, acc), 0);
        const float inv = (float) ((D) 1 / sycl::sqrt(ssum / (D) Si + (D) eps));
        for (int i = t; i < Si; i += WARP) py[i] = po[i] * inv * ssm_norm[i] * sigmoid_f(pz[i]);
    });
}
}  // namespace

void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols > 1024) throw core::DeviceError("gdn_l2_norm: cols = " + std::to_string(cols) + " exceeds 1024");
    const int ncols = (int) cols;
    auto& q = queue_for(stream);
    const auto e = xe::has_fp64(q) ? gdn_l2_norm_t<double>(q, x, rows, ncols, eps)
                                   : gdn_l2_norm_t<float>(q, x, rows, ncols, eps);
    finish(stream, e, "gdn_l2_norm");
}

void gdn_beta_gate(float* beta, int64_t h_v, void* stream) {
    if (beta == nullptr || h_v <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) h_v), [=](sycl::id<1> i) {
        beta[i] = sigmoid_f(beta[i]);
    });
    finish(stream, e, "gdn_beta_gate");
}

void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v, int64_t S,
                  float eps, void* stream) {
    if (h_v <= 0 || S <= 0) return;
    const int Si = (int) S;
    auto& q = queue_for(stream);
    const auto e = xe::has_fp64(q) ? gdn_out_norm_t<double>(q, o, z, ssm_norm, y, h_v, Si, eps)
                                   : gdn_out_norm_t<float>(q, o, z, ssm_norm, y, h_v, Si, eps);
    finish(stream, e, "gdn_out_norm");
}

}  // namespace strata::kernels
