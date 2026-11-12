// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/vision_rows.cpp - the row kernels the vision encoder (src/core/vision.cpp) shares with the image
// engine, moved from src/image/qi_kernels.cpp as they were: the GELUs, the bias add, the axpy, the LayerNorm and the
// ViT's QKV with its 2-D RoPE and the heads' padding to 128 (include/strata/kernels/image_attn.hpp).

#include "strata/kernels/image_attn.hpp"

#include "strata/core/runtime.hpp"
#include "strata/image/gelu.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/qk_norm_rope.hpp"
#include "strata/kernels/row_stats.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::image {

namespace {

using strata::kernels::f16_gpu::f16_from_f32;   // the GPU's conversions (f16_gpu.hpp)

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

}  // namespace

void qi_gelu16(const float* x, int64_t n, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i];
        y16[i] = f16_from_f32(gelu_tanh(v));
    });
    finish(stream, e, "qi_gelu16");
}

void qi_axpy(float* x, const float* v, float a, int64_t n, void* stream) { kernels::add_inplace(x, v, n, stream, a); }

void qi_bias_add(float* y, const float* b, int64_t T, int64_t N, void* stream, const float* add, int64_t ldy) {
    if (T <= 0 || N <= 0 || b == nullptr) return;
    if (ldy == 0) ldy = N;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * N)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % N;
        y[ldy == N ? i : (i / N) * ldy + c] += add != nullptr ? b[c] + add[i] : b[c];
    });
    finish(stream, e, "qi_bias_add");
}

void qi_layer_norm16(const float* x, const float* w, const float* b, int64_t T, int64_t C, float eps, uint16_t* y16,
                     void* stream) {
    if (T <= 0) return;
    constexpr size_t L = 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) T * L), sycl::range<1>(L)), [=](sycl::nd_item<1> it) {
            const auto t = static_cast<int64_t>(it.get_group_linear_id());
            const auto l = static_cast<int64_t>(it.get_local_linear_id());
            const float* r = x + t * C;
            const auto st = kernels::row_stats::calculate<kernels::row_stats::Kind::moments>(
                l, (int64_t) L, C, eps, [=](int64_t c) { return r[c]; },
                [&](float v) { return sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()); });
            for (int64_t c = l; c < C; c += (int64_t) L)
                y16[t * C + c] = f16_from_f32((r[c] - st.mean) * st.inv * w[c] + b[c]);
        });
    finish(stream, e, "qi_layer_norm16");
}

void qi_gelu_erf16(const float* x, int64_t n, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i];
        y16[i] = f16_from_f32(gelu_erf(v));
    });
    finish(stream, e, "qi_gelu_erf16");
}

void qi_vit_qkv_rope16(const float* qkv, const int32_t* yx, int64_t T, int64_t heads, int64_t hd, float theta,
                       float q_scale, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream) {
    if (T <= 0) return;
    constexpr int64_t D = 128;
    const int64_t C = heads * hd;
    // the rows [T][q | k | v][heads][hd] in; out [T][heads][D], the columns past hd zero, q scaled
    struct Rows {
        const float* qkv;
        int64_t C, hd;
        auto head(int which, int64_t t, int64_t h) const {
            const float* p = qkv + t * 3 * C + which * C + h * hd;
            return [p](int64_t d) { return p[d]; };
        }
    };
    struct Padded {
        uint16_t *q16, *k16, *v16;
        int64_t heads, hd;
        float q_scale;
        auto out_head(int which, int64_t t, int64_t h) const {
            uint16_t* dst = (which == 0 ? q16 : which == 1 ? k16 : v16) + (t * heads + h) * D;
            const float sc = which == 0 ? q_scale : 1.f;
            const int64_t n = hd;
            return [=](int64_t d, float v) {
                dst[d] = f16_from_f32(v * sc);
                for (int64_t e = n + d; e < D; e += n) dst[e] = 0;
            };
        }
    };
    kernels::qk::Shape sh{T, heads, heads, hd};
    kernels::qk::Theta ang{theta, 0.5f * (float) hd};
    ang.yx = yx;
    ang.quarter = hd / 4;
    finish(stream, kernels::qk::qk_norm_rope(queue_for(stream), sh, Rows{qkv, C, hd}, ang,
                                             Padded{q16, k16, v16, heads, hd, q_scale}),
           "qi_vit_qkv_rope16");
}

void qi_unpad_heads16(const uint16_t* in16, int64_t T, int64_t heads, int64_t hd, uint16_t* out16, void* stream) {
    if (T <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * heads * hd)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / (heads * hd), r = i % (heads * hd), h = r / hd, d = r % hd;
        out16[i] = in16[(t * heads + h) * 128 + d];
    });
    finish(stream, e, "qi_unpad_heads16");
}

void qi_pad_heads16(const uint16_t* in16, int64_t T, int64_t heads, int64_t hd, int64_t ld, uint16_t* out16,
                    void* stream) {
    if (T <= 0) return;
    if (ld == 0) ld = heads * hd;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * heads * 128)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / (heads * 128), r = i % (heads * 128), h = r / 128, d = r % 128;
        out16[i] = d < hd ? in16[t * ld + h * hd + d] : (uint16_t) 0;
    });
    finish(stream, e, "qi_pad_heads16");
}

}  // namespace strata::image
