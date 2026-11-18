// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/dit_rows.hpp - the parts Mage-Flow's DiT (mf_dit.cpp) and MOSS-Image's (moss_dit.cpp) share: the blocks'
// bf16 weights quantized on the host (Q8_0, or W8A8's rotated int8 rows), and the row-wise kernels that write a
// product's FP16 input with each row's power-of-two scale, scale its output back and add it through a gate.
//
// The kernels' helpers are in a namespace of their own, not an anonymous one: SYCL names a lambda's kernel by its
// mangled type, and an anonymous namespace mangles the same in every file (_GLOBAL__N_1).
#pragma once

#include "strata/core/gpu.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/gelu.hpp"
#include "strata/image/gpu_buf.hpp"
#include "strata/image/mf_dit.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/w8_choice.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/platform/env.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_to_f16.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/q8_act.hpp"
#include "strata/kernels/qk_norm_rope.hpp"
#include "strata/kernels/rows_cols.hpp"
#include "strata/kernels/rows16.hpp"
#include "strata/kernels/rows_i8.hpp"
#include "strata/kernels/row_stats.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::image::dit_rows {

namespace syclex = sycl::ext::oneapi::experimental;
using kernels::f16_gpu::f16_from_f32;   // the GPU's conversions (f16_gpu.hpp)
using kernels::f16_gpu::f32_from_f16;

using strata::image::q_of;   // the shared one (gpu_buf.hpp), as every image file's own copy was

// the W8A8 rows' kernel and launcher, shared with the LLMs' prompt products (kernels/rows_i8.hpp)
using kernels::RowsI8Of;
using kernels::rows_i8_as;

inline float bf16_value(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// x rounded to the nearest bf16 (ties to even), as a float
inline float round_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// row of K bf16 values -> Q8_0: blocks of an FP16 scale and 32 int8 values.  The rounding is q8act::Mul's, the one
// rule the GPU's quantize (fused_gr_quantize_q8_0) also takes, so the host's and the device's bytes agree.
inline void quantize_row(const uint16_t* src, int64_t K, uint8_t* dst) {
    namespace q8 = kernels::q8act;
    for (int64_t b = 0; b < K / 32; ++b) {
        float v[32], amax = 0.f;
        for (int j = 0; j < 32; ++j) {
            v[j] = bf16_value(src[b * 32 + j]);
            amax = std::max(amax, std::fabs(v[j]));
        }
        const float d = q8::Mul::scale(amax), id = q8::Mul::inv(d);
        const uint16_t dh = f16_from_f32(d);
        int8_t* qs = reinterpret_cast<int8_t*>(dst + b * 34 + 2);
        for (int j = 0; j < 32; ++j) qs[j] = q8::Mul::value(v[j], d, id, amax);
        std::memcpy(dst + b * 34, &dh, 2);
    }
}

// row of K bf16 values -> int8 values and their scale as kernels::rows_i8 makes them: each group of kQiRotate times the
// normalized Hadamard matrix (Sylvester's, the same butterflies), then one scale for the row
inline void quantize_row_w8(const uint16_t* src, int64_t K, int8_t* dst, float* scale) {
    std::vector<float> v((size_t) K);
    float amax = 0.f;
    for (int64_t g = 0; g < K; g += kQiRotate) {
        float* x = v.data() + g;
        for (int64_t j = 0; j < kQiRotate; ++j) x[j] = bf16_value(src[g + j]);
        for (int64_t m = 1; m < kQiRotate; m <<= 1)
            for (int64_t i = 0; i < kQiRotate; i += 2 * m)
                for (int64_t j = i; j < i + m; ++j) {
                    const float a = x[j], b = x[j + m];
                    x[j] = a + b;
                    x[j + m] = a - b;
                }
        for (int64_t j = 0; j < kQiRotate; ++j) {
            x[j] *= 1.f / 16.f;
            amax = std::max(amax, std::fabs(x[j]));
        }
    }
    const float d = kernels::q8act::Mul::scale(amax), id = kernels::q8act::Mul::inv(d);
    for (int64_t j = 0; j < K; ++j) dst[j] = kernels::q8act::Mul::value(v[(size_t) j], d, id, amax);
    *scale = d;
}

// the rows of a [n][k] bf16 matrix quantized on the host's threads: Q8_0, or with w8 as quantize_row_w8 (the n rows'
// int8 values, then their n scales)
inline void quantize_matrix(const uint16_t* src, int64_t n, int64_t k, uint8_t* dst, bool w8) {
    const uint64_t row = w8 ? (uint64_t) k : (uint64_t) (k / 32 * 34);
    auto* scales = reinterpret_cast<float*>(dst + (uint64_t) n * row);
    const int nt = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    th.reserve((size_t) nt);
    for (int t = 0; t < nt; ++t)
        th.emplace_back([=] {
            for (int64_t r = t; r < n; r += nt) {
                if (w8) quantize_row_w8(src + r * k, k, reinterpret_cast<int8_t*>(dst + r * row), scales + r);
                else quantize_row(src + r * k, k, dst + r * row);
            }
        });
    for (auto& x : th) x.join();
}

// Mage-Flow's activations pass FP16's range (its model ran in bf16: the image stream's normed rows up to 1.9e5, the
// attention's v up to 6.1e5, a text MLP's output 5.7e6 at the first steps of a 1024 x 1024 Turbo run).  So a product's
// FP16 input holds each row divided by a power of two (rs, at least 1) that brings its largest value under kRowLim, and
// the product's output rows are multiplied back by it; the attention's v is divided by one power of two a head over a
// context's rows (its output is a weighted sum of them), and the output multiplied back.  Powers of two leave the
// rounding as it is; FP16 keeps 3 more bits than bf16.
// a BF16 tensor's values as FP16, in place, clamped to FP16's range (the DiTs' 16-bit weights)
inline void bf16_to_f16_inplace(uint16_t* p, int64_t n, sycl::queue& q) {
    kernels::bf16_to_f16<kernels::F16Range::clamp>(q, p, p, n);
}

// q, k and v [T][heads hd] -> FP16 [T][heads][hd]: q and k RMS-normed over the head (wq, wk; eps, 1e-6 by default)
// and rotated in pairs by cos[t][j], sin[t][j] (tables [T][hd / 2]; null: not rotated), (j, j + n_rot / 2) with `neox`
// (Qwen-Image's heirs: MOSS-Image, Anima) or (2j, 2j + 1) (Mage-Flow, SeedVR2), the first n_rot channels of a head (0:
// all of them; SeedVR2 7B's 60), the rest normed only; v divided by its head's scale hs (null: as it is)
// (qk_norm_rope.hpp)
inline void qk_rope16(const DitOut& q, const DitOut& k, const DitOut& v, const float* wq, const float* wk,
                      const float* cos, const float* sin, const float* hs, int64_t T, int64_t heads, int64_t hd,
                      bool neox, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* s, float eps = 1e-6f,
                      int64_t n_rot = 0) {
    if (T <= 0) return;
    struct Heads {
        DitOut q, k, v;
        const float* hs;
        int64_t hd;
        struct Head {
            DitOut::Row r;
            float inv;   // v's 1 / hs, else 1
            float operator()(int64_t d) const { return r(d) * inv; }
            void r2(int64_t d, float& a, float& b) const {
                r.r2(d, a, b);
                a *= inv;
                b *= inv;
            }
        };
        Head head(int which, int64_t t, int64_t h) const {
            return {(which == 0 ? q : which == 1 ? k : v).row(t, h * hd),
                    which == 2 && hs != nullptr ? 1.f / hs[h] : 1.f};
        }
    };
    kernels::qk::Shape sh{T, heads, heads, hd};
    sh.wq = wq;
    sh.wk = wk;
    sh.neox = neox;
    sh.eps = eps;
    sh.n_rot = n_rot;
    const Heads in{q, k, v, hs, hd};
    const kernels::qk::Rows16 out{q16, k16, v16, heads, heads, hd};
    if (cos != nullptr) kernels::qk::qk_norm_rope(q_of(s), sh, in, kernels::qk::Tables{cos, sin, hd / 2}, out);
    else kernels::qk::qk_norm_rope(q_of(s), sh, in, kernels::qk::NoRope{}, out);
}

// One tensor's heads RMS-normed over the head (w; null: the values copied as they are) into FP16 rows
// [T][heads * hd], not rotated: a cross-attention's q, k and v, each its own row count (qk_rope16 norms q, k and v of
// one count together).  The same kernel as qk_rope16's, its k and v heads left out.
inline void qk_norm16(const DitOut& x, const float* w, int64_t T, int64_t heads, int64_t hd, uint16_t* out, void* s,
                      float eps = 1e-6f) {
    if (T <= 0) return;
    struct In {
        DitOut x;
        int64_t hd;
        auto head(int, int64_t t, int64_t h) const { return x.row(t, h * hd); }
    };
    kernels::qk::Shape sh{T, heads, 0, hd};
    sh.wq = w;
    sh.values = false;
    sh.eps = eps;
    const kernels::qk::Rows16 o{out, nullptr, nullptr, heads, 0, hd};
    kernels::qk::qk_norm_rope(q_of(s), sh, In{x, hd}, kernels::qk::NoRope{}, o);
}

using kernels::rows16::kSG;
using kernels::rows16::kRows;
using kernels::rows16::pow2_scale;
using kernels::rows16::pow2_full;

template<class Op> inline void rows16(const Op& op, int64_t T, int64_t C, uint16_t* y16, float* rs, void* s,
                                      float* mean = nullptr, float* inv = nullptr) {
    if (T <= 0) return;
    kernels::rows16::rows16(op, T, C, y16, rs, q_of(s), mean, inv);
}

using Stats = kernels::row_stats::Stats;

// LayerNorm(x[t]) over the channels (no affine; eps 1e-6) * (1 + scale) + shift: the DiTs' modulated norm (rows16's
// Op)
struct LnMod {
    const float *x, *shift, *scale;
    int64_t C;
    template<class R> Stats stats(int64_t t, int64_t lane, int64_t step, R reduce) const {
        const float* r = x + t * C;
        return kernels::row_stats::calculate<kernels::row_stats::Kind::variance>(
            lane, step, C, 1e-6f, [=](int64_t c) { return r[c]; }, reduce);
    }
    float value(int64_t t, int64_t c, const Stats& st) const {
        return (x[t * C + c] - st.mean) * st.inv * (1.f + scale[c]) + shift[c];
    }
};

// A DiT's residual stream: FP32, or BF16 (uint16_t: MOSS-Image's), rounded to the nearest.  NVIDIA's
// GPUs take the conversion instruction (kernels::bf16_from_f32's integer rounding doubled MOSS-Image's RMS norm's
// instructions on the RTX 4070, 2026-11-06); elsewhere the integer rounding: a kernel with
// sycl::ext::oneapi::bfloat16's conversion crashes intel/llvm v7.2.0's SYCL runtime (libsycl 9) on Intel's GPUs where
// its on-disk cache is on, as the runtime sets it (getItemFromDisc, getSortedImages; the B70, the same day).  The #if
// goes once the intel/llvm the build takes runs the data folder's records/2026-11-06-libsycl-bf16-cache/repro with
// SYCL_CACHE_PERSISTENT=1
inline float resid(float v) { return v; }
inline float resid(uint16_t v) { return kernels::f32_from_bf16(v); }
inline uint16_t bf16_bits(float v) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    return sycl::bit_cast<uint16_t>(sycl::ext::oneapi::bfloat16(v));
#else
    return kernels::bf16_from_f32(v);
#endif
}
template<class X> inline X resid_of(float v) {
    if constexpr (std::is_same_v<X, uint16_t>) return bf16_bits(v);
    else return v;
}

// RMSNorm(x[t]) over the channels (eps), times w where it is given, then times scale and plus shift where given
// (SeedVR2's modulation: the time's and the layer's scale summed), or with one_plus times (1 + scale) and plus shift
// where scale is given (MOSS-Image's), rows16's Op (X the residual's type)
template<class X> struct RmsAffineOf {
    const X *x;
    const float *w, *scale, *shift;
    int64_t C;
    float eps;
    bool one_plus = false;
    AltRows alt{};   // rows whose scale and shift are another set's
    template<class R> Stats stats(int64_t t, int64_t lane, int64_t step, R reduce) const {
        const X* r = x + t * C;
        return kernels::row_stats::calculate<kernels::row_stats::Kind::rms>(
            lane, step, C, eps, [=](int64_t c) { return resid(r[c]); }, reduce);
    }
    float value(int64_t t, int64_t c, const Stats& st) const {
        const int64_t m = c + alt.at(t);
        if (one_plus) {   // (its weight given)
            const float y = resid(x[t * C + c]) * st.inv * w[c];
            return scale != nullptr ? y * (1.f + scale[m]) + shift[m] : y;
        }
        float y = resid(x[t * C + c]) * st.inv;
        if (w != nullptr) y *= w[c];
        if (scale != nullptr) y *= scale[m];
        return shift != nullptr ? y + shift[m] : y;
    }
};
using RmsAffine = RmsAffineOf<float>;

// RowsI8Of's ops over the same rows: RmsAffine's (no weight), GeluTanh's, FP16 rows and FP32 rows as they are; R the
// regular rotation (kRegular: int8_convrot's weights)
template<bool R> struct RmsAffineI8 {
    static constexpr bool kRms = true, kKeep = false, kRegular = R;
    const float *x, *scale, *shift;
    int64_t C;
    float eps;
    float raw(int64_t t, int64_t c) const { return x[t * C + c]; }
    void keep(int64_t, int64_t, float) const {}
    float inv(float ss) const { return 1.f / sycl::sqrt(ss / (float) C + eps); }
    float out(int64_t, int64_t c, float v, float inv) const { return v * inv * scale[c] + shift[c]; }
};
template<bool R> struct GeluTanhI8 {
    static constexpr bool kRms = false, kKeep = false, kRegular = R;
    DitOut x;
    float raw(int64_t t, int64_t c) const { return gelu_tanh(x.at(t, c)); }
    void keep(int64_t, int64_t, float) const {}
    float inv(float) const { return 1.f; }
};
template<bool R> struct HalfI8 {
    static constexpr bool kRms = false, kKeep = false, kRegular = R;
    const uint16_t* h;
    int64_t C;
    float raw(int64_t t, int64_t c) const { return f32_from_f16(h[t * C + c]); }
    void keep(int64_t, int64_t, float) const {}
    float inv(float) const { return 1.f; }
};
template<bool R> struct FloatI8 {
    static constexpr bool kRms = false, kKeep = false, kRegular = R;
    const float* x;
    int64_t C;
    float raw(int64_t t, int64_t c) const { return x[t * C + c]; }
    void keep(int64_t, int64_t, float) const {}
    float inv(float) const { return 1.f; }
};

template<bool R> struct SwiGluI8 {
    static constexpr bool kRms = false, kKeep = false, kRegular = R;
    DitOut g;
    int64_t F;
    float raw(int64_t t, int64_t c) const {
        const float a = g.at(t, c);
        return kernels::rows16::swiglu(a, g.at(t, c + F));
    }
    void keep(int64_t, int64_t, float) const {}
    float inv(float) const { return 1.f; }
};

// SwiGLU of a fused product's output g [T][2F]: SiLU of the first F (the gate) times the second F, rows16's Op
struct SwiGlu {
    DitOut g;
    int64_t F;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const {
        const float a = g.at(t, c);
        return kernels::rows16::swiglu(a, g.at(t, c + F));
    }
};

// FP32 rows [T][C] as they are, rows16's Op (a text encoder's states, a training step's gradients)
struct Plain {
    const float* x;
    int64_t C;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const { return x[t * C + c]; }
};

// A training step's gradient rows [T][C] as they are, each at the power of two that fills FP16 (pow2_full), rows16's Op.
// A zero row has scale zero, so it cannot dominate a later product's global scale.
struct Grad : Plain {
    static constexpr bool kFullRange = true;
};

// SiTU of the up_gate product's sums g [T][2F] (the hidden values, then the gates): (4 tanh(g / 4) sigmoid(g)) times
// (25 tanh(h / 25)), F values a row (MOSS-Image's MLP; its backward situ_bwd)
struct Situ {
    DitOut g;
    int64_t F;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const {
        const float h = g.at(t, c), gt = g.at(t, c + F);
        return 4.f * sycl::tanh(gt * 0.25f) / (1.f + sycl::exp(-gt)) * (25.f * sycl::tanh(h * 0.04f));
    }
    // below 4 x 25 in magnitude: FP16 takes the rows as they are (their scales 1)
    float bound(int64_t, int64_t) const { return 0.f; }
};

// GELU (tanh form) of a product's output, rows16's Op
struct GeluTanh {
    DitOut x;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const { return gelu_tanh(x.at(t, c)); }
    // |GELU(v)| <= v for v >= 0 and below 0.17 for v < 0: the row's scale without a tanh more a value
    float bound(int64_t t, int64_t c) const { return sycl::fmax(x.at(t, c), 0.17f); }
};

// rows_i8's timing passes, as a block's: x[t] += gate y[t] (a product's int32 sums, scaled), written back, then
// RMS-normed and modulated (gate_add into the MLP's input), over 4096 rows of 3072, twice; FP16 rows of 3072 (the
// attention's output); an activation of two int32 sums side by side F apart (SiTU's hidden value and gate), over 1024
// rows of F = 8192
struct TimedI8 {
    static constexpr bool kRms = true, kKeep = true;
    float* x;
    DitOut y;
    const float* w;   // the norm's weights, then the modulation's scale and shift
    int64_t C;
    float raw(int64_t t, int64_t c) const { return x[t * C + c] + w[c] * y.at(t, c); }
    void keep(int64_t t, int64_t c, float v) const { x[t * C + c] = v; }
    float inv(float ss) const { return 1.f / sycl::sqrt(ss / (float) C + 1e-6f); }
    float out(int64_t, int64_t c, float v, float inv) const { return v * inv * w[c] * (1.f + w[C + c]) + w[2 * C + c]; }
};
struct TimedHalfI8 {
    static constexpr bool kRms = false, kKeep = false;
    const uint16_t* h;
    int64_t C;
    float raw(int64_t t, int64_t c) const { return f32_from_f16(h[t * C + c]); }
    void keep(int64_t, int64_t, float) const {}
    float inv(float) const { return 1.f; }
    float out(int64_t, int64_t, float v, float) const { return v; }
};
struct TimedPairI8 {
    static constexpr bool kRms = false, kKeep = false;
    DitOut y;
    int64_t F;
    float raw(int64_t t, int64_t c) const { return y.at(t, c) * sycl::tanh(y.at(t, c + F)); }
    void keep(int64_t, int64_t, float) const {}
    float inv(float) const { return 1.f; }
    float out(int64_t, int64_t, float v, float) const { return v; }
};

// The values a lane holds side by side, the fastest of 8, 4 and 1 on the timing passes once a GPU.  MOSS-Image's step
// (2026-10-29): the B70 754 ms with 1 (4: 781, 8: 848), the RTX 4070 874 with 4 (1: 876, 8: 979), the RX 9060 XT
// 1474 with 4 (8: 1479, 1: 1497); the passes took 1, 4 and 8.  The int8 values and the scales are the same bits
// whichever it is (RowsI8Of's bit_at), so a near tie taken one way or the other from start to start changes no image.
inline int rows_i8_width(sycl::queue& qu) {
    static core::PerDevice<int> width;
    return width.get(qu.get_device(), [&qu] {
        constexpr int64_t T = 4096, C = 3072, TF = 1024, F = 8192;   // y: T x C or TF x 2F int32 sums
        float* x = sycl::malloc_device<float>((size_t) (T * C), qu);
        auto* y = sycl::malloc_device<int32_t>((size_t) (TF * 2 * F), qu);
        float* sxw = sycl::malloc_device<float>((size_t) (T + 2 * F + 3 * C), qu);
        int8_t* q = sycl::malloc_device<int8_t>((size_t) (T * C), qu);
        float* s = sycl::malloc_device<float>((size_t) T, qu);
        int best = 8;
        if (x != nullptr && y != nullptr && sxw != nullptr && q != nullptr && s != nullptr) {
            qu.fill(x, 1.f, (size_t) (T * C));
            qu.fill(y, 1, (size_t) (TF * 2 * F));
            qu.fill(sxw, 1e-3f, (size_t) (T + 2 * F + 3 * C)).wait();
            DitOut yc, yf;
            yc.i = yf.i = y;
            yc.sx = yf.sx = sxw;
            yc.sw = yf.sw = sxw + T;
            yc.C = C;
            yf.C = 2 * F;
            const TimedI8 gate{x, yc, sxw + T + 2 * F, C};
            const TimedHalfI8 half{reinterpret_cast<const uint16_t*>(y), C};
            const TimedPairI8 pair{yf, F};
            auto run = [&](auto w) {
                rows_i8_as<TimedI8, decltype(w)::value>(gate, T, C, q, s, qu);
                rows_i8_as<TimedI8, decltype(w)::value>(gate, T, C, q, s, qu);
                rows_i8_as<TimedHalfI8, decltype(w)::value>(half, T, C, q, s, qu);
                rows_i8_as<TimedPairI8, decltype(w)::value>(pair, TF, F, q, s, qu);
            };
            double best_ms = 1e30;
            std::string said;
            std::vector<std::pair<int, double>> took;
            for (const int w : {8, 4, 1}) {
                double ms = 1e30;
                for (int i = 0; i < 4; ++i) {   // each time 4 passes, the first time to build the kernels
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int k = 0; k < 4; ++k) {
                        if (w == 1) run(std::integral_constant<int, 1>{});
                        else if (w == 4) run(std::integral_constant<int, 4>{});
                        else run(std::integral_constant<int, 8>{});
                    }
                    qu.wait();
                    const auto t1 = std::chrono::steady_clock::now();
                    if (i > 0) ms = std::min(ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
                }
                said += (said.empty() ? "" : ", ") + std::to_string(w) + ": " + std::to_string(ms).substr(0, 5) + " ms";
                took.emplace_back(w, ms);
                best_ms = std::min(best_ms, ms);
            }
            // the first of 8, 4, 1 within 3% of the fastest: the RX 9060 XT's three within 2% took one or another
            // from start to start
            for (const auto& [w, ms] : took)
                if (ms <= best_ms * 1.03) {
                    best = w;
                    break;
                }
            std::fprintf(stderr, "image engine: the W8A8 inputs' rows %d values side by side a lane (%s)\n", best,
                         said.c_str());
        }
        for (void* p : {(void*) x, (void*) y, (void*) sxw, (void*) q, (void*) s})
            if (p != nullptr) sycl::free(p, qu);
        return best;
    });
}

// the fewest sub-groups a row (G 4): a work-group of 1024 items (SiTU's 8192 values a sub-group's group each) took
// MOSS-Image's SiTU 2.76 ms a block on the RX 9060 XT, a quarter of them with 4 groups each 1.83 ms (2026-10-29)
template<class Op> inline void rows_i8(const Op& op, int64_t T, int64_t K, int8_t* q, float* s, void* st) {
    if (T <= 0) return;
    sycl::queue& qu = q_of(st);
    switch (rows_i8_width(qu)) {
    case 1: rows_i8_as<Op, 1>(op, T, K, q, s, qu); break;
    case 4: rows_i8_as<Op, 4>(op, T, K, q, s, qu); break;
    default: rows_i8_as<Op, 8>(op, T, K, q, s, qu); break;
    }
}

// The regular Hadamard rotation (RowsI8Of's kRegular, kron(H4, H4, H4, H4) / 16: int8_convrot's basis) of a group of
// kQiRotate values in local memory t, by 256 work-items of a work-group (all of it, or a slice of it for each of its
// groups), the item `lid` writing value `lid`: four butterflies on the group's base-4 digits (strides 64, 16, 4, 1),
// each sum in the order of a single work-item's loop, and the scale 1 / 16 at the end.  Every item of the work-group
// `g` calls it.  Returns value lid.  For FP rows out: RowsI8Of keeps its values in registers and rotates them through
// two of Sylvester's instead, the same matrix with the sums in another order (FP rows' bits would differ).
template<class Group> inline float regular256(float* t, int lid, const Group& g) {
    static_assert(kQiRotate == 256);
    float sum = 0.f;
    for (int stride = 64; stride >= 1; stride /= 4) {
        const int i = (lid / stride) % 4, first = lid - i * stride;
        sycl::group_barrier(g);
        sum = 0.f;
        for (int j = 0; j < 4; ++j) sum += i + j == 3 ? -t[first + j * stride] : t[first + j * stride];
        sycl::group_barrier(g);
        t[lid] = sum;
    }
    return sum * (1.f / 16.f);
}

// FP16 rows [T][K] from rows [T][K] of FP16 (uint16_t) or FP32 values: every group of kQiRotate values times the
// regular Hadamard (regular256), rounded to FP16 once, after it.  A work-group a group.
template<class In> inline sycl::event regular_rotate16(const In* x, int64_t T, int64_t K, uint16_t* out16, void* s) {
    static_assert(std::is_same_v<In, uint16_t> || std::is_same_v<In, float>);
    if (K % kQiRotate != 0)
        throw std::runtime_error("image engine: the regular rotation takes rows of a multiple of 256 values, not " +
                                 std::to_string(K));
    if (T <= 0) return {};
    const int64_t gpr = K / kQiRotate;
    return q_of(s).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> t(sycl::range<1>((size_t) kQiRotate), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (T * K), (size_t) kQiRotate), [=](sycl::nd_item<1> it) {
            const int64_t g = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const int64_t at = (g / gpr) * K + (g % gpr) * kQiRotate + lid;
            float* tp = t.get_multi_ptr<sycl::access::decorated::no>().get();
            if constexpr (std::is_same_v<In, uint16_t>) tp[lid] = f32_from_f16(x[at]);
            else tp[lid] = x[at];
            out16[at] = f16_from_f32(regular256(tp, lid, it.get_group()));
        });
    });
}

// the attention's output o16 [T][C] times its heads' scales hs (head_dim channels a head)
struct Heads {
    const uint16_t* o16;
    const float* hs;
    int64_t C, hd;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const { return f32_from_f16(o16[t * C + c]) * hs[c / hd]; }
};

// The per-head kernels below: a sub-group of 32 a row's head (its values read side by side, kSG apart), kRows heads a
// work-group.

// part[r][h] = the largest |v| of row r's head h, rows of a ([na][C]) then b ([nb][C])
struct HeadMax {
    DitOut a, b;
    int64_t na, nb, heads, hd;
    float* part;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t i = static_cast<int64_t>(it.get_group_linear_id()) * kRows + sg.get_group_linear_id();
        if (i >= (na + nb) * heads) return;
        const int64_t r = i / heads, h = i - r * heads;
        float m = 0.f;
        for (int64_t d = sg.get_local_linear_id(); d < hd; d += kSG)
            m = sycl::fmax(m, sycl::fabs(r < na ? a.at(r, h * hd + d) : b.at(r - na, h * hd + d)));
        m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
        if (sg.get_local_linear_id() == 0) part[i] = m;
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<kSG>}; }
};

inline void head_max(const DitOut& a, int64_t na, const DitOut& b, int64_t nb, int64_t heads, int64_t hd, float* part,
              void* s) {
    const int64_t n = (na + nb) * heads;
    if (n <= 0) return;
    sycl::queue& q = q_of(s);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((n + kRows - 1) / kRows * kRows * kSG)),
                                     sycl::range<1>((size_t) (kRows * kSG))),
                   HeadMax{a, b, na, nb, heads, hd, part});
}

// The column form of head_max and head_scale: part[s][c] = the largest |x| of column c over slice s of the rows, a's
// na then b's nb (kColSlices slices; the rows read side by side, a column a work-item), then col_scale: hs[h] = head h's power of
// two as head_scale makes it (the same: a head's largest is its columns'), and sv[c] = column c's largest at it, over
// 127 (1 for a column of zeros) - the V scales the int8 attention takes (kernels::FlashAttnI8Stats), so that it need
// not read the attention's FP16 v again for them
// A work-item 4 columns side by side (where C is a multiple of 4) of a slice: MOSS-Image's v on the RTX 4070 (4115 rows
// of 3072, 2026-11-06) took 0.12 ms read a column a work-item over 32 slices, at 40% of the memory's bandwidth
constexpr int64_t kColSlices = 128;

inline void col_max(const DitOut& a, int64_t na, const DitOut& b, int64_t nb, int64_t C, float* part, void* s) {
    const int64_t n = na + nb;
    if (n <= 0) return;
    auto run = [&]<int W>() {
        q_of(s).parallel_for(sycl::range<2>((size_t) kColSlices, (size_t) (C / W)), [=](sycl::id<2> id) {
            const auto sl = static_cast<int64_t>(id[0]), c = static_cast<int64_t>(id[1]) * W;
            const int64_t t0 = n * sl / kColSlices, t1 = n * (sl + 1) / kColSlices;
            float m[W] = {};
            for (int64_t t = t0; t < t1; ++t) {
                float v[W];
                if constexpr (W == 1) v[0] = t < na ? a.at(t, c) : b.at(t - na, c);
                else if (t < na) a.at_w<W>(t, c, v);
                else b.at_w<W>(t - na, c, v);
                for (int p = 0; p < W; ++p) m[p] = sycl::fmax(m[p], sycl::fabs(v[p]));
            }
            for (int p = 0; p < W; ++p) part[sl * C + c + p] = m[p];
        });
    };
    if (C % 4 == 0) run.template operator()<4>();
    else run.template operator()<1>();
}

inline void col_scale(const float* part, int64_t C, int64_t heads, int64_t hd, float* hs, float* sv, void* s) {
    q_of(s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (heads * hd)), sycl::range<1>((size_t) hd)),
                         [=](sycl::nd_item<1> it) {
        const auto h = static_cast<int64_t>(it.get_group_linear_id());
        const int64_t c = h * hd + static_cast<int64_t>(it.get_local_linear_id());
        float cm = 0.f;
        for (int64_t sl = 0; sl < kColSlices; ++sl) cm = sycl::fmax(cm, part[sl * C + c]);
        const float m = sycl::reduce_over_group(it.get_group(), cm, sycl::maximum<float>());
        const float hsv = pow2_scale(m);
        if (it.get_local_linear_id() == 0) hs[h] = hsv;
        sv[c] = cm > 0.f ? cm / hsv / 127.f : 1.f;
    });
}

// hs[h] = the scale of head h over the rows' part: a work-group a head, its items striding the rows
inline void head_scale(const float* part, int64_t rows, int64_t heads, float* hs, void* s) {
    constexpr size_t kWG = 256;
    q_of(s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) heads * kWG), sycl::range<1>(kWG)),
                         [=](sycl::nd_item<1> it) {
        const auto h = static_cast<int64_t>(it.get_group_linear_id());
        float m = 0.f;
        for (auto r = static_cast<int64_t>(it.get_local_linear_id()); r < rows; r += (int64_t) kWG)
            m = sycl::fmax(m, part[r * heads + h]);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        if (it.get_local_linear_id() == 0) hs[h] = pow2_scale(m);
    });
}

// y[t] = y[t] * rs[t] * col + b (rs null: 1; b null: none; col null: 1, a DoRA's factor of the weight's row, the
// bias not scaled)
inline void rows_back(float* y, const float* rs, const float* b, int64_t T, int64_t N, void* s,
                      const float* col = nullptr) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * N)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / N, r = i - t * N;
        float v = y[i];
        if (rs != nullptr) v *= rs[t];
        if (col != nullptr) v *= col[r];
        if (b != nullptr) v += b[r];
        y[i] = v;
    });
}

// y[t] += b (the weight's own bias, after a product whose output an OFT turned: a rotation of the weight's rows is
// not one of its bias's)
inline void bias_add(float* y, const float* b, int64_t T, int64_t N, void* s) { qi_bias_add(y, b, T, N, s); }

// The DiTs' gated residual: x[t][c] += g(t, c) h[t][c]
template<class X, class G> sycl::event gate_add_by(X* x, const DitOut& h, G g, int64_t T, int64_t C, void* s) {
    if (T <= 0) return {};
    return kernels::for_rows_cols(q_of(s), T, C, [=](int64_t t, int64_t c) {
        x[t * C + c] = resid_of<X>(resid(x[t * C + c]) + g(t, c) * h.at(t, c));
    });
}

// x[t] += gate * h[t] (per channel; gate null: 1)
template<class X> inline void gate_add(X* x, const DitOut& h, const float* gate, int64_t T, int64_t C, void* s) {
    (void) gate_add_by(x, h, [=](int64_t, int64_t c) { return gate != nullptr ? gate[c] : 1.f; }, T, C, s);
}

// out[t] = x[t] + gate * h[t]: the same, x kept (a training step's backward reads it)
inline void gate_add_to(float* out, const float* x, const DitOut& h, const float* gate, int64_t T, int64_t C, void* s) {
    if (T <= 0) return;
    kernels::for_rows_cols(q_of(s), T, C, [=](int64_t t, int64_t c) {
        out[t * C + c] = x[t * C + c] + (gate != nullptr ? gate[c] : 1.f) * h.at(t, c);
    });
}

// ---- the backward of the row operations above (a training step's), FP32 rows

// out[c] += the sum over the rows of f(t, c): a work-group a column
template <class F> inline void col_sum(int64_t rows, int64_t cols, float* out, F f, void* s) {
    constexpr int64_t WG = 256;
    q_of(s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) cols * WG), sycl::range<1>(WG)),
                         [=](sycl::nd_item<1> it) {
                             const int64_t c = static_cast<int64_t>(it.get_group_linear_id());
                             float v = 0.f;
                             for (auto t = static_cast<int64_t>(it.get_local_linear_id()); t < rows; t += WG)
                                 v += f(t, c);
                             v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
                             if (it.get_local_linear_id() == 0) out[c] += v;
                         });
}

// qk_rope16's backward (kernels::qk::qk_norm_rope_bwd), FP32: dgq and dgk the gradients of the normed and rotated
// rows (what the attention reads), dq and dk those of the products' outputs q and k ([T][heads][hd]), each head normed
// with wq and wk and then rotated by the tables cos and sin [T][hd / 2] (rope false or cos null: normed only), pairs
// (j, j + hd / 2) with neox, else (2j, 2j + 1).  dwq and dwk (null: not taken) add the norms' weights' gradients.
inline void qk_rope_bwd(const float* q, const float* k, const float* wq, const float* wk, const float* cos,
                        const float* sin, bool rope, const float* dgq, const float* dgk, int64_t T, int64_t heads,
                        int64_t hd, float* dq, float* dk, float* dwq, float* dwk, void* s, bool neox,
                        float eps = 1e-6f) {
    kernels::qk::Shape sh{T, heads, heads, hd};
    sh.wq = wq;
    sh.wk = wk;
    sh.eps = eps;
    sh.neox = neox;
    float* rs = nullptr;
    if ((dwq != nullptr || dwk != nullptr) && !gpu::alloc_device(&rs, (size_t) (2 * T * heads) * 4))
        throw std::runtime_error("image engine: the QK norm's backward scales");
    if (rope && cos != nullptr)
        kernels::qk::qk_norm_rope_bwd(q_of(s), sh, q, k, dgq, dgk, kernels::qk::Tables{cos, sin, hd / 2}, dq, dk,
                                      dwq, dwk, rs);
    else
        kernels::qk::qk_norm_rope_bwd(q_of(s), sh, q, k, dgq, dgk, kernels::qk::NoRope{}, dq, dk, dwq, dwk, rs);
    if (rs != nullptr) {
        gpu::stream_sync(s);   // rs is freed below: the kernels above run async
        gpu::free(rs);
    }
}

// LnMod's: dx = inv (m - (S1 + x^ S2) / C), m = dy (1 + scale), x^ the normed row, S1 and S2 the sums over the row of m
// and m x^ (mean and inv: the forward's statistics); dshift += dy and dscale += dy x^ over the rows (null: not taken)
inline void ln_mod_bwd(const float* x, const float* mean, const float* inv, const float* scale, const float* dy,
                       int64_t rows, int64_t cols, float* dx, float* dshift, float* dscale, void* s) {
    constexpr int64_t WG = 256;
    const float inv_c = 1.f / (float) cols;
    q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t t = static_cast<int64_t>(it.get_group_linear_id());
            const float iv = inv[t], mn = mean[t];
            float s1 = 0.f, s2 = 0.f;
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < cols; c += WG) {
                const float m = dy[t * cols + c] * (1.f + scale[c]);
                s1 += m;
                s2 += m * (x[t * cols + c] - mn) * iv;
            }
            s1 = sycl::reduce_over_group(it.get_group(), s1, sycl::plus<float>());
            s2 = sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>());
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < cols; c += WG)
                dx[t * cols + c] = iv * (dy[t * cols + c] * (1.f + scale[c]) -
                                         (s1 + ((x[t * cols + c] - mn) * iv) * s2) * inv_c);
        });
    if (dshift != nullptr) col_sum(rows, cols, dshift, [=](int64_t t, int64_t c) { return dy[t * cols + c]; }, s);
    if (dscale != nullptr)
        col_sum(rows, cols, dscale,
                [=](int64_t t, int64_t c) { return dy[t * cols + c] * (x[t * cols + c] - mean[t]) * inv[t]; }, s);
}

// GeluTanh's: dv = dy gelu_tanh'(v), v the GELU's input
inline void gelu_tanh_bwd(const float* v, const float* dy, int64_t n, float* dv, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n),
                         [=](sycl::id<1> id) { dv[id[0]] = dy[id[0]] * gelu_tanh_grad(v[id[0]]); });
}

// SiLU's: dy = d silu'(y), y the SiLU's input
inline void silu_bwd(const float* y, const float* d, int64_t n, float* dy, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = y[id[0]], g = 1.f / (1.f + sycl::exp(-v));
        dy[id[0]] = d[id[0]] * g * (1.f + v * (1.f - g));
    });
}

// SwiGlu's: g [T][2F] the fused product's output (the gate's F, then the up's F), d [T][F] the gradient of
// silu(gate) * up; dg [T][2F] gets the gate's half d * up * silu'(gate) and the up's half d * silu(gate)
inline void swiglu_bwd(const float* g, const float* d, int64_t T, int64_t F, float* dg, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * F)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / F, c = i - t * F;
        const float a = g[t * 2 * F + c], u = g[t * 2 * F + F + c], sg = 1.f / (1.f + sycl::exp(-a));
        dg[t * 2 * F + c] = d[i] * u * sg * (1.f + a * (1.f - sg));
        dg[t * 2 * F + F + c] = d[i] * a * sg;
    });
}

// Situ's (MOSS-Image's MLP activation): g [T][2F] the fused product's output (the hidden values, then the gates), d
// [T][F] the gradient of (4 tanh(gt / 4) sigmoid(gt)) (25 tanh(h / 25)); dg [T][2F] gets the hidden half and the gate
// half
inline void situ_bwd(const float* g, const float* d, int64_t T, int64_t F, float* dg, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * F)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / F, c = i - t * F;
        const float h = g[t * 2 * F + c], gt = g[t * 2 * F + F + c];
        const float th = sycl::tanh(h * 0.04f), tg = sycl::tanh(gt * 0.25f), sg = 1.f / (1.f + sycl::exp(-gt));
        const float a = 4.f * tg * sg, b = 25.f * th;
        dg[t * 2 * F + c] = d[i] * a * (1.f - th * th);
        dg[t * 2 * F + F + c] = d[i] * b * ((1.f - tg * tg) * sg + 4.f * tg * sg * (1.f - sg));
    });
}

// RmsAffine's (one_plus, its weight w): y = x inv w (1 + scale) + shift, inv = 1 / sqrt(mean(x^2) + eps) from the
// forward's statistics; dx = inv (m - n S / C), m = dy w (1 + scale), n = x inv, S the sum over the row of m n
inline void rms_affine_bwd(const float* x, const float* inv, const float* w, const float* scale, const float* dy,
                           int64_t rows, int64_t cols, float* dx, void* s) {
    constexpr int64_t WG = 256;
    const float inv_c = 1.f / (float) cols;
    q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t t = static_cast<int64_t>(it.get_group_linear_id());
            const float iv = inv[t];
            float s1 = 0.f;
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < cols; c += WG)
                s1 += dy[t * cols + c] * w[c] * (scale != nullptr ? 1.f + scale[c] : 1.f) * x[t * cols + c] * iv;
            s1 = sycl::reduce_over_group(it.get_group(), s1, sycl::plus<float>());
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < cols; c += WG) {
                const float m = dy[t * cols + c] * w[c] * (scale != nullptr ? 1.f + scale[c] : 1.f);
                dx[t * cols + c] = iv * (m - x[t * cols + c] * iv * s1 * inv_c);
            }
        });
}

// gate_add's: dh = gate dx (gate null: 1) and dgate += dx h over the rows (null dh or dgate: not taken)
inline void gate_add_bwd(const float* h, const float* gate, const float* dx, int64_t rows, int64_t cols, float* dh,
                         float* dgate, void* s) {
    if (dh != nullptr)
        q_of(s).parallel_for(sycl::range<1>((size_t) (rows * cols)), [=](sycl::id<1> id) {
            const int64_t c = static_cast<int64_t>(id[0]) % cols;
            dh[id[0]] = (gate != nullptr ? gate[c] : 1.f) * dx[id[0]];
        });
    if (dgate != nullptr)
        col_sum(rows, cols, dgate, [=](int64_t t, int64_t c) { return dx[t * cols + c] * h[t * cols + c]; }, s);
}

// The time (seconds) of one product of the DiT's FFN, T rows x K -> N, in 16 bits (prefill::Gemm::f16) and in 8
// (the rows to int8, the product as the DiT's reader takes it - Gemm::w8a8: the fused kernel's BF16 where it is the
// faster on the GPU, else the library's sums and the scales back), the least of three after a warm-up: what a DiT
// chooses its form by (0: the form does not run here).  `what` names the DiT in the message of a form the library
// could not run.
inline std::pair<double, double> time_w8_f16(int64_t T, int64_t K, int64_t N, const char* what, void* s_) {
    if (T <= 0 || K <= 0 || N <= 0) return {0.0, 0.0};
    std::vector<void*> held;
    auto get = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!gpu::alloc_device(&p, bytes)) return nullptr;
        held.push_back(p);
        gpu::memset_async(p, 0, bytes, s_);
        return p;
    };
    auto best = [&](const std::function<void()>& run) {
        double t = 1e30;
        for (int i = 0; i < 4; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            run();
            if (!gpu::stream_sync(s_)) return 0.0;
            if (i > 0) t = std::min(t, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return t;
    };
    double f16 = 0.0, w8 = 0.0;
    try {
        auto* x16 = static_cast<uint16_t*>(get((size_t) (T * K) * 2));
        auto* w16 = static_cast<uint16_t*>(get((size_t) (N * K) * 2));
        auto* y = static_cast<float*>(get((size_t) (T * N) * 4));
        prefill::Gemm g;
        std::string err;
        if (x16 != nullptr && w16 != nullptr && y != nullptr && g.init(s_, 0, err))
            f16 = best([&] { g.f16(x16, w16, y, T, N, K); });
        if (x16 != nullptr && y != nullptr && prefill::Gemm::i8_usable(s_)) {
            auto* x8 = static_cast<int8_t*>(get((size_t) (T * K)));
            auto* w8p = static_cast<int8_t*>(get((size_t) (N * K)));
            auto* sx = static_cast<float*>(get((size_t) T * 4));
            auto* sw = static_cast<float*>(get((size_t) N * 4));
            if (x8 != nullptr && w8p != nullptr && sx != nullptr && sw != nullptr)
                w8 = best([&] {
                    kernels::rows_i8(x16, false, T, K, x8, sx, nullptr, s_);
                    if (!prefill::Gemm::w8a8(x8, sx, w8p, sw, nullptr, y, T, N, K, N, true, s_))
                        kernels::i8_scale(y, T, N, N, sx, sw, nullptr, nullptr, s_);
                });
        }
    } catch (const std::exception& e) {   // a form the library could not run: the 8-bit one is not chosen
        std::fprintf(stderr, "image engine: timing %s in 16 and 8 bits: %s (16 bits kept)\n", what, e.what());
        w8 = 0.0;
    }
    gpu::stream_sync(s_);
    for (void* p : held) gpu::free(p);
    return {f16, w8};
}

// The choice (W8Choice) of a DiT that lays its weights out when it opens, before a request's rows are known
// (Qwen-Image's, SeedVR2's), and the engine's word to the server (w8_here): its MLP's first product, K -> N, timed at
// a 1024 x 1024 image's 4096 rows, once a GPU and shape -> 1 W8A8, 0 FP16, -1 neither form could be timed (no room for
// the timing: not kept, timed again the next time)
inline int w8_at_open(int64_t K, int64_t N, const char* what, void* s_) {
    struct Kept {
        std::mutex m;
        std::map<std::pair<int64_t, int64_t>, int> of;
    };
    static core::PerDevice<std::shared_ptr<Kept>> per_device;
    Kept& kept = *per_device.get(core::Runtime::get().stream(s_).get_device(), [] { return std::make_shared<Kept>(); });
    std::lock_guard<std::mutex> lock(kept.m);
    if (const auto it = kept.of.find({K, N}); it != kept.of.end()) return it->second;
    const W8Choice c;
    const int w8 = c.take(4096, what, s_, [&](int64_t rows) { return time_w8_f16(rows, K, N, what, s_); }) ? 1 : 0;
    if (!c.decided()) return -1;
    kept.of.emplace(std::make_pair(K, N), w8);
    return w8;
}

}  // namespace strata::image::dit_rows
