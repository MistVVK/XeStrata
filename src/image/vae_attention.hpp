// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/core/runtime.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/product_store.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/row_stats.hpp"
#include "strata/prefill/gemm.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::image::vae_mid {

namespace syclex = sycl::ext::oneapi::experimental;

// rms16 a device-selected sub-group a pixel, PIX pixels a work-group
template<int SG> struct Rms16 {
    static constexpr int64_t PIX = 256 / SG, LANES = SG * PIX;
    const float *x, *g;
    int64_t HW, C;
    bool silu;
    uint16_t* out16;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t p = static_cast<int64_t>(it.get_group_linear_id()) * PIX + sg.get_group_linear_id();
        const int64_t lane = sg.get_local_linear_id();
        if (p >= HW) return;   // the whole sub-group
        const float* r = x + p * C;
        const auto st = kernels::row_stats::calculate<kernels::row_stats::Kind::rms>(
            lane, SG, C, 1e-12f, [=](int64_t c) { return r[c]; },
            [&](float v) { return sycl::reduce_over_group(sg, v, sycl::plus<float>()); });
        const float inv = st.inv;
        for (int64_t c = lane; c < C; c += SG) {
            float v = r[c] * inv * g[c];
            if (silu) v = v / (1.f + sycl::exp(-v));
            out16[p * C + c] = kernels::f16_gpu::f16_from_f32(v);
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<SG>}; }
};

// out16[p] = RMS-normed x[p] over the channels times g (SiLU after it when `silu`): one sub-group a pixel, the
// sum within it (a work-group a pixel and its sum through local memory and barriers took the RX 9060 XT 14.8 ms a
// call on average in a 1024 x 1024 decode)
inline void rms16(const float* x, const float* g, int64_t HW, int64_t C, bool silu, uint16_t* out16, void* s) {
    auto& q = core::Runtime::get().stream(s);
    auto launch = [&]<int SG>() {
        constexpr int64_t PIX = Rms16<SG>::PIX, LANES = Rms16<SG>::LANES;
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((HW + PIX - 1) / PIX * LANES)),
                                         sycl::range<1>((size_t) LANES)), Rms16<SG>{x, g, HW, C, silu, out16});
    };
    // B70, 512-square decode: 16 -> 32 gave Qwen-Image 0.068 -> 0.066 s and Anima 0.018 -> 0.017 s
    // (warm medians, 3 runs); 32 also restored Qwen-Image's bytes in the isolated Rms16 comparison.
    const auto widths = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(widths.begin(), widths.end(), 32) != widths.end()) launch.template operator()<32>();
    else launch.template operator()<16>();
}

enum class Norm { rms, group };

// QKV is [H*W][3*C], FP32 or FP16/BF16. V is transposed for Gemm.
// A window replicates edge pixels and scatters only positions inside the image.
// Scratch holds rows*T scores and probabilities and T*C each of Q, K and V.
struct Attention {
    int64_t h, w, c, rows, window = 0;
    Norm norm = Norm::group;
    const float *x = nullptr, *weight = nullptr, *bias = nullptr;
    const uint16_t* x16 = nullptr;
    uint16_t* normalized = nullptr;
    float* norm_workspace = nullptr;
    float eps = 1e-6f;
    const float* qkv = nullptr;
    const uint16_t* qkv16 = nullptr;
    bool bf = false, group_softmax = false;
    uint16_t *q = nullptr, *k = nullptr, *vt = nullptr, *p = nullptr;
    float *scores = nullptr, *out = nullptr, *window_out = nullptr;
};

// The kernels below are templates on the caller's projection (Tag), so that each VAE's source holds its own copy and
// its decode runs them from its own device image.  As plain inline functions one copy served every VAE from another
// VAE's image (HunyuanImage 3.0's, Qwen-Image's), which Mage-Flow's first decode then loaded: 317 KB more, its first
// 512-square decode on the RX 9060 XT 38 to 40 ms and its img2img 22 to 25 ms (2026-11-11).
// The 256-lane form preserves the reduction order of the 16-bit VAE path.
template<class Tag>
void softmax(const float* scores, int64_t rows, int64_t n, float scale, uint16_t* p, bool group, void* s) {
    if (!group) { qi_softmax16(scores, rows, n, scale, p, s); return; }
    constexpr int64_t local = 256;
    core::Runtime::get().stream(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (rows * local)), sycl::range<1>((size_t) local)),
        [=](sycl::nd_item<1> it) {
            const auto r = (int64_t) it.get_group_linear_id(), lane = (int64_t) it.get_local_linear_id();
            const float* row = scores + r * n;
            float m = -INFINITY;
            for (int64_t j = lane; j < n; j += local) m = sycl::fmax(m, row[j] * scale);
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            float sum = 0.f;
            for (int64_t j = lane; j < n; j += local) sum += sycl::exp(row[j] * scale - m);
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float inv = 1.f / sum;
            for (int64_t j = lane; j < n; j += local)
                p[r * n + j] = kernels::f16_gpu::f16_from_f32(sycl::exp(row[j] * scale - m) * inv);
        });
}

template<class Tag>
void calculate(const Attention& a, prefill::Gemm& gemm, void* s) {
    const int64_t H = a.h, W = a.w, C = a.c, win = a.window;
    const int64_t T = win ? win * win : H * W;
    const int64_t ny = win ? (H + win - 1) / win : 1, nx = win ? (W + win - 1) / win : 1;
    auto& queue = core::Runtime::get().stream(s);
    for (int64_t wy = 0; wy < ny; ++wy) for (int64_t wx = 0; wx < nx; ++wx) {
        const auto qkv = a.qkv; const auto qkv16 = a.qkv16;
        const bool bf = a.bf;
        auto q = a.q; auto k = a.k; auto vt = a.vt;
        queue.parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
            const int64_t i = (int64_t) id[0], t = i / C, c = i - t * C;
            const int64_t y = win ? sycl::min(wy * win + t / win, H - 1) : t / W;
            const int64_t x = win ? sycl::min(wx * win + t % win, W - 1) : t % W;
            const int64_t base = (y * W + x) * 3 * C + c;
            auto value = [=](int64_t offset) -> uint16_t {
                if (qkv) return kernels::f16_gpu::f16_from_f32(qkv[base + offset]);
                const uint16_t v = qkv16[base + offset];
                return bf ? kernels::f16_gpu::f16_from_f32(kernels::f32_from_bf16(v)) : v;
            };
            q[i] = value(0); k[i] = value(C); vt[c * T + t] = value(2 * C);
        });
        float* out = win ? a.window_out : a.out;
        for (int64_t r = 0; r < T; r += a.rows) {
            const int64_t n = std::min(a.rows, T - r);
            gemm.f16(a.q + r * C, a.k, a.scores, n, T, C);
            softmax<Tag>(a.scores, n, T, 1.f / std::sqrt((float) C), a.p, a.group_softmax, s);
            gemm.f16(a.p, a.vt, out + r * C, n, C, T);
        }
        if (win) {
            auto dest = a.out;
            queue.parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
                const int64_t i = (int64_t) id[0], t = i / C;
                const int64_t y = wy * win + t / win, x = wx * win + t % win;
                if (y < H && x < W) dest[(y * W + x) * C + i % C] = out[i];
            });
        }
    }
}

// The caller owns the projection weights and storage; its callbacks apply each 1x1 product
// and the output residual in the format of that VAE.
template<class ProjectQkv, class ProjectOut>
void run(const Attention& a, prefill::Gemm& gemm, void* s, ProjectQkv project_qkv, ProjectOut project_out) {
    const int64_t pixels = a.h * a.w;
    if (a.norm == Norm::rms) rms16(a.x, a.weight, pixels, a.c, false, a.normalized, s);
    else if (a.x16) qi_group_norm16(a.x16, a.bf, pixels, a.c, a.weight, a.bias, a.eps, false,
                                    a.normalized, a.norm_workspace, s);
    else qi_group_norm16(a.x, pixels, a.c, a.weight, a.bias, a.eps, false, a.normalized, a.norm_workspace, s);
    project_qkv();
    calculate<ProjectQkv>(a, gemm, s);
    ProductStore p; p.in = a.out; p.out16 = a.normalized; p.rows = pixels; p.cols = a.c; p.add_zero = false;
    product_store(p, s);
    project_out();
}

}  // namespace strata::image::vae_mid
