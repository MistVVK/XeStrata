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
#include "strata/kernels/i8_gemm.hpp"
#include "strata/platform/env.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/q8_act.hpp"
#include "strata/kernels/qk_norm_rope.hpp"
#include "strata/kernels/rows_cols.hpp"

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

// row of K bf16 values -> int8 values and their scale as qi_rows_i8 makes them: each group of kQiRotate times the
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
constexpr float kRowLim = 16384.f;

inline float pow2_scale(float amax) {
    return amax > kRowLim ? sycl::exp2(sycl::ceil(sycl::log2(amax / kRowLim))) : 1.f;
}

// A training step's gradient rows, small or large: the power of two that brings the row's largest value into
// [2^13, 2^14), up as well as down.  A gradient is often far below FP16's smallest normal value (6.1e-5: the LoRA
// factors' gradients of Qwen-Image-2.1's real blocks sit near 1e-6), where FP16 keeps a few bits or none.
// The exponent is read off the value's bits (a subnormal's as -127): sycl::ilogb inside this row kernel gave
// 2^-144 for a row whose largest value was 2 (intel/llvm v7.2.0, the B70, 2026-11-09).
inline float pow2_full(float amax) {
    if (!(amax > 0.f)) return 1.f;
    const int e = (int) ((sycl::bit_cast<uint32_t>(amax) >> 23) & 0xffu) - 127;
    return sycl::bit_cast<float>((uint32_t) (sycl::clamp(e - 13, -126, 127) + 127) << 23);
}

// a BF16 tensor's values as FP16, in place, clamped to FP16's range (the DiTs' 16-bit weights)
inline void bf16_to_f16_inplace(uint16_t* p, int64_t n, sycl::queue& q) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = kernels::f32_from_bf16(p[id[0]]);
        p[id[0]] = f16_from_f32(sycl::clamp(v, -65504.f, 65504.f));
    });
}

// q, k and v [T][heads hd] -> FP16 [T][heads][hd]: q and k RMS-normed over the head (wq, wk; eps, 1e-6 by default)
// and rotated in pairs by cos[t][j], sin[t][j] (tables [T][hd / 2]; null: not rotated), (j, j + n_rot / 2) with `neox`
// (Qwen-Image's heirs: MOSS-Image, Anima) or (2j, 2j + 1) (Mage-Flow, SeedVR2), the first n_rot channels of a head (0:
// all of them; SeedVR2 7B's 60), the rest normed only; v divided by its head's scale hs (null: as it is)
// (qk_norm_rope.hpp)
inline void qk_rope16(const MfOut& q, const MfOut& k, const MfOut& v, const float* wq, const float* wk,
                      const float* cos, const float* sin, const float* hs, int64_t T, int64_t heads, int64_t hd,
                      bool neox, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* s, float eps = 1e-6f,
                      int64_t n_rot = 0) {
    if (T <= 0) return;
    struct Heads {
        MfOut q, k, v;
        const float* hs;
        int64_t hd;
        struct Head {
            MfOut::Row r;
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
inline void qk_norm16(const MfOut& x, const float* w, int64_t T, int64_t heads, int64_t hd, uint16_t* out, void* s,
                      float eps = 1e-6f) {
    if (T <= 0) return;
    struct In {
        MfOut x;
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

// A row-wise kernel writing FP16 with each row's scale: Op (LnMod, Gelu, Heads below) gives a row's statistics and its
// values; a sub-group of 32 a row, 8 rows a work-group (every engine GPU has them: core/device.cpp).
constexpr int64_t kSG = 32, kRows = 8;

// the row's scale is taken from a bound of |op.value| where the op has a cheaper one than the value itself
template<class Op, class St> float bound(const Op& op, int64_t t, int64_t c, const St& st) {
    if constexpr (requires { op.bound(t, c); }) return op.bound(t, c);
    else return sycl::fabs(op.value(t, c, st));
}

template<class Op> struct Rows16 {
    Op op;
    int64_t T, C;
    uint16_t* y16;
    float* rs;
    float *keep_mean = nullptr, *keep_inv = nullptr;   // the rows' statistics kept (a training step's backward)
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t t = static_cast<int64_t>(it.get_group_linear_id()) * kRows + sg.get_group_linear_id();
        const int64_t lane = sg.get_local_linear_id();
        if (t >= T) return;
        const auto sum = [&](float v) { return sycl::reduce_over_group(sg, v, sycl::plus<float>()); };
        const auto st = op.stats(t, lane, kSG, sum);
        float m = 0.f;
        for (int64_t c = lane; c < C; c += kSG) m = sycl::fmax(m, bound(op, t, c, st));
        const float rmax = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
        float sc;
        if constexpr (requires { Op::kFullRange; }) sc = pow2_full(rmax);
        else sc = pow2_scale(rmax);
        const float inv = 1.f / sc;
        for (int64_t c = lane; c < C; c += kSG) y16[t * C + c] = f16_from_f32(op.value(t, c, st) * inv);
        if (lane == 0) {
            if (rs != nullptr) rs[t] = sc;
            if (keep_mean != nullptr) keep_mean[t] = st.mean;
            if (keep_inv != nullptr) keep_inv[t] = st.inv;
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<kSG>}; }
};

// mean and inv (null: not kept): the rows' statistics (Stats) the Op gave, which a training step's backward reads
template<class Op> inline void rows16(const Op& op, int64_t T, int64_t C, uint16_t* y16, float* rs, void* s,
                                      float* mean = nullptr, float* inv = nullptr) {
    if (T <= 0) return;
    sycl::queue& q = q_of(s);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((T + kRows - 1) / kRows * kRows * kSG)),
                                     sycl::range<1>((size_t) (kRows * kSG))),
                   Rows16<Op>{op, T, C, y16, rs, mean, inv});
}

struct Stats {
    float mean = 0.f, inv = 1.f;
};

// LayerNorm(x[t]) over the channels (no affine; eps 1e-6) * (1 + scale) + shift: the DiTs' modulated norm (rows16's
// Op)
struct LnMod {
    const float *x, *shift, *scale;
    int64_t C;
    template<class R> Stats stats(int64_t t, int64_t lane, int64_t step, R reduce) const {
        const float* r = x + t * C;
        float s = 0.f;
        for (int64_t c = lane; c < C; c += step) s += r[c];
        const float mean = reduce(s) / (float) C;
        float v = 0.f;
        for (int64_t c = lane; c < C; c += step) v += (r[c] - mean) * (r[c] - mean);
        return {mean, 1.f / sycl::sqrt(reduce(v) / (float) C + 1e-6f)};
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
        float s = 0.f;
        for (int64_t c = lane; c < C; c += step) s += resid(r[c]) * resid(r[c]);
        return {0.f, 1.f / sycl::sqrt(reduce(s) / (float) C + eps)};
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
    MfOut x;
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
    MfOut g;
    int64_t F;
    float raw(int64_t t, int64_t c) const {
        const float a = g.at(t, c);
        return a / (1.f + sycl::exp(-a)) * g.at(t, c + F);
    }
    void keep(int64_t, int64_t, float) const {}
    float inv(float) const { return 1.f; }
};

// SwiGLU of a fused product's output g [T][2F]: SiLU of the first F (the gate) times the second F, rows16's Op
struct SwiGlu {
    MfOut g;
    int64_t F;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const {
        const float a = g.at(t, c);
        return a / (1.f + sycl::exp(-a)) * g.at(t, c + F);
    }
};

// FP32 rows [T][C] as they are, rows16's Op (a text encoder's states, a training step's gradients)
struct Plain {
    const float* x;
    int64_t C;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const { return x[t * C + c]; }
};

// A training step's gradient rows [T][C] as they are, each at the power of two that fills FP16 (pow2_full), rows16's Op
struct Grad : Plain {
    static constexpr bool kFullRange = true;
};

// SiTU of the up_gate product's sums g [T][2F] (the hidden values, then the gates): (4 tanh(g / 4) sigmoid(g)) times
// (25 tanh(h / 25)), F values a row (MOSS-Image's MLP; its backward situ_bwd)
struct Situ {
    MfOut g;
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
    MfOut x;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const { return gelu_tanh(x.at(t, c)); }
    // |GELU(v)| <= v for v >= 0 and below 0.17 for v < 0: the row's scale without a tanh more a value
    float bound(int64_t t, int64_t c) const { return sycl::fmax(x.at(t, c), 0.17f); }
};

// A W8A8 product's int8 input (qi_kernels.hpp: qi_rows_i8's rows) straight from a row-wise op: kQiRotate values at a
// time through Sylvester's Hadamard butterflies, times 1 / 16, then one scale a row (q8_act.hpp's Mul rule), without
// FP16 rows and a second pass between: on the RX 9060 XT those two passes were a fifth of MOSS-Image's step
// (2026-10-29).  A work-group a row, a sub-group of 32 a group of kQiRotate values (8 a lane, in registers), up to G
// groups each.  A lane holds its 8 values W side by side (value j at (j / W) 32 W + lane W + j % W of the group), so a
// bit of a value's position in the group is a bit of j or of lane by W.  The butterflies go over the position's bits
// in one order whatever W is (bit_at), within the lane for j's and across the lanes for lane's, and the row's sum of
// the squares is a tree over the same bits: the int8 values and the scales are the same bits for 1, 4 and 8.  Each
// sum in the lane's own order before (W 8 its bits 0 to 7, W 1 5, 6, 7, then 0 to 4; the squares a lane at a time)
// made the images depend on the width rows_i8_width timed (the B70, 2026-11-10: SeedVR2 7B's FP16 file, MOSS-Image
// and Mage-Flow each two or three images from start to start).  Op: raw(t, c), the value before the row's statistic (keep(t, c, v) writes it back
// where kKeep); where kRms, inv(sum of the squares) of the row; out(t, c, v, inv), the value given it (none: v).  Or
// W values at once: raw_w, keep_w, out_w; or with the sub-group, load_sg (a lane's 8 values of a GGUF block, W 8).
// row_scale(t), where the op has it: the row's own factor on its scale.  An op with kRegular rotates by the regular
// Hadamard matrix instead (kron(H4, H4, H4, H4) / 16, H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] / 2: the
// basis Comfy-Org's int8_convrot weights are saved in, hunyuanimage3_file.hpp), which is Sylvester's, a sign that flips
// for each of the position's base-4 digits that is 3, and Sylvester's again: H4 = S D S with D = diag(1, 1, 1, -1).
template<class Op> constexpr bool regular_rotation() {
    if constexpr (requires { Op::kRegular; }) return Op::kRegular;
    else return false;
}
template<class Op, int W, int G = 4> struct RowsI8Of {
    static constexpr int SG = 32, V = kQiRotate / SG, kMaxG = G;
    static_assert(W == 1 || W == 4 || W == V);
    static_assert(G == 1 || G == 2 || G == 4);
    static constexpr bool kSg = requires(const Op& o, const sycl::sub_group& sg, float* p) {
        o.template load_sg<W>(sg, int64_t{}, int64_t{}, p);
    };
    // an op's W values side by side in one access (raw_w, keep_w where it keeps them; out_w): RmsI8 on the RTX 4070
    // through scalar reads W values apart kept the L1 busy at 44% of the memory's bandwidth (2026-11-05)
    static constexpr bool kVec = W > 1 && requires(const Op& o, float* p) {
        o.template raw_w<W>(int64_t{}, int64_t{}, p);
    } && (!Op::kKeep || requires(const Op& o, const float* p) { o.template keep_w<W>(int64_t{}, int64_t{}, p); });
    static constexpr bool kVecOut = W > 1 && requires(const Op& o, float* p) {
        o.template out_w<W>(int64_t{}, int64_t{}, p, 1.f);
    };
    static constexpr bool kRegular = regular_rotation<Op>();
    // the position's bits in the order the sums take them (W 4's bits of j, then of lane, as before: W 4 kept its
    // rotation), and where each bit is: a mask of j's, else of lane's
    static constexpr int kLogW = W == 1 ? 0 : W == 4 ? 2 : 3;
    static constexpr int bit_at(int st) {
        constexpr int order[8] = {0, 1, 7, 2, 3, 4, 5, 6};
        return order[st];
    }
    static constexpr int j_mask(int b) { return b < kLogW ? 1 << b : b >= kLogW + 5 ? 1 << (b - 5) : 0; }
    static constexpr int lane_mask(int b) { return j_mask(b) != 0 ? 0 : 1 << (b - kLogW); }
    static_assert(V == 8 && SG == 32 && kQiRotate == 256);
    Op op;
    int64_t K, gpr, nsg;
    int8_t* q;
    float* s;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const auto r = static_cast<int64_t>(it.get_group_linear_id());
        const auto si = static_cast<int64_t>(sg.get_group_linear_id());
        const int lane = static_cast<int>(sg.get_local_linear_id());
        auto off = [](int j) { return (int64_t) (j / W) * SG * W + j % W; };   // value j past the lane's first
        // every loop unrolled, the groups a sub-group skips by a condition: rolled (the breaks they had), v went to
        // local memory on NVPTX (qi_kernels.cpp's RowsI8: the L2's writes 35 times its output, RTX 4070, 2026-11-05)
        float v[kMaxG][V];
        float ss = 0.f;
#pragma unroll
        for (int c = 0; c < kMaxG; ++c) {
            const int64_t g = si + c * nsg;
            if (g >= gpr) continue;   // the whole sub-group
            const int64_t k0 = g * kQiRotate + (int64_t) lane * W;
            if constexpr (kSg) {
                op.template load_sg<W>(sg, r, k0, v[c]);
            } else if constexpr (kVec) {   // W values a load: one access each, not W the lanes' accesses W values apart
#pragma unroll
                for (int h = 0; h < V / W; ++h) {
                    const int j = h * W;
                    op.template raw_w<W>(r, k0 + (int64_t) h * SG * W, &v[c][j]);
                    if constexpr (Op::kKeep) op.template keep_w<W>(r, k0 + (int64_t) h * SG * W, &v[c][j]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < V; ++j) {
                    v[c][j] = op.raw(r, k0 + off(j));
                    if constexpr (Op::kKeep) op.keep(r, k0 + off(j), v[c][j]);
                }
            }
            if constexpr (Op::kRms) {   // the group's sum of the squares, every lane the same tree (a + b == b + a)
                float sq[V];
#pragma unroll
                for (int j = 0; j < V; ++j) sq[j] = v[c][j] * v[c][j];
                int done = 0;   // j's bits summed: the sums left at the j that have none of them
#pragma unroll
                for (int st = 0; st < 8; ++st) {
                    const int b = bit_at(st), m = j_mask(b);
#pragma unroll
                    for (int j = 0; j < V; ++j) {
                        if ((j & done) != 0 || (j & m) != 0) continue;
                        sq[j] = m != 0 ? sq[j] + sq[j + m] : sq[j] + sycl::permute_group_by_xor(sg, sq[j], lane_mask(b));
                    }
                    done |= m;
                }
                ss += sq[0];
            }
        }
        float inv = 1.f;
        // the sub-groups' sums (each lane's the same) once each: the work-group's tree over them whatever W is
        if constexpr (Op::kRms)
            inv = op.inv(sycl::reduce_over_group(it.get_group(), lane == 0 ? ss : 0.f, sycl::plus<float>()));
        float amax = 0.f;
#pragma unroll
        for (int c = 0; c < kMaxG; ++c) {
            const int64_t g = si + c * nsg;
            if (g >= gpr) continue;
            const int64_t k0 = g * kQiRotate + (int64_t) lane * W;
            if constexpr (kVecOut) {
#pragma unroll
                for (int h = 0; h < V / W; ++h) {
                    const int j = h * W;
                    op.template out_w<W>(r, k0 + (int64_t) h * SG * W, &v[c][j], inv);
                }
            } else if constexpr (requires { op.out(r, k0, 1.f, inv); }) {
#pragma unroll
                for (int j = 0; j < V; ++j) v[c][j] = op.out(r, k0 + off(j), v[c][j], inv);
            }
            // Sylvester's butterflies over the position's bits in bit_at's order: within the lane for j's, across the
            // lanes for lane's
            auto sylvester = [&] {
#pragma unroll
                for (int st = 0; st < 8; ++st) {
                    const int b = bit_at(st), m = j_mask(b), l = lane_mask(b);
#pragma unroll
                    for (int j = 0; j < V; ++j) {
                        if (m != 0) {
                            if ((j & m) == 0) {
                                const float a = v[c][j], e = v[c][j + m];
                                v[c][j] = a + e;
                                v[c][j + m] = a - e;
                            }
                        } else {
                            const float o = sycl::permute_group_by_xor(sg, v[c][j], l);
                            v[c][j] = (lane & l) != 0 ? o - v[c][j] : v[c][j] + o;
                        }
                    }
                }
            };
            sylvester();
            float norm = 1.f / 16.f;
            if constexpr (kRegular) {   // D between two of Sylvester's: the value at position p of the group
#pragma unroll
                for (int j = 0; j < V; ++j) {
                    const int p = (int) off(j) + lane * W;
                    int threes = 0;
#pragma unroll
                    for (int d = 0; d < 8; d += 2) threes += ((p >> d) & 3) == 3;
                    if (threes & 1) v[c][j] = -v[c][j];
                }
                sylvester();
                norm = 1.f / 256.f;
            }
#pragma unroll
            for (int j = 0; j < V; ++j) {
                v[c][j] *= norm;
                amax = sycl::fmax(amax, sycl::fabs(v[c][j]));
            }
        }
        amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        const float d = kernels::q8act::Mul::scale(amax), id = kernels::q8act::Mul::inv(d);
#pragma unroll
        for (int c = 0; c < kMaxG; ++c) {
            const int64_t g = si + c * nsg;
            if (g >= gpr) continue;
            int8_t* qg = q + r * K + g * kQiRotate + (int64_t) lane * W;
#pragma unroll
            for (int h = 0; h < V / W; ++h) {
                auto q8 = [&](int p) { return (uint32_t) (uint8_t) kernels::q8act::Mul::value(v[c][h * W + p], d, id, amax); };
                if constexpr (W == 1) {
                    qg[(int64_t) h * SG] = (int8_t) q8(0);
                } else {   // packed into 32-bit words and stored as one: a sycl::vec<int8_t, W> store is split into
                           // its bytes on NVPTX
                    uint32_t o[W / 4] = {};
#pragma unroll
                    for (int p = 0; p < W; ++p) o[p / 4] |= q8(p) << (8 * (p % 4));
                    if constexpr (W == 4) {
                        *reinterpret_cast<uint32_t*>(qg + (int64_t) h * SG * W) = o[0];
                    } else {
                        using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
                        *reinterpret_cast<u32x2*>(qg + (int64_t) h * SG * W) = u32x2{o[0], o[1]};
                    }
                }
            }
        }
        if (si == 0 && lane == 0) {
            if constexpr (requires { op.row_scale(r); }) s[r] = d * op.row_scale(r);
            else s[r] = d;
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<SG>}; }
};

// G groups a sub-group: ceil(K / kQiRotate / G) sub-groups a row
template<class Op, int W, int G = 4>
inline sycl::event rows_i8_as(const Op& op, int64_t T, int64_t K, int8_t* q, float* s, sycl::queue& qu) {
    constexpr int64_t SG = RowsI8Of<Op, W, G>::SG;
    const int64_t gpr = K / kQiRotate;
    const auto max_sg =
        static_cast<int64_t>(qu.get_device().get_info<sycl::info::device::max_work_group_size>()) / SG;
    if (K % kQiRotate != 0 || gpr > max_sg * G)
        throw std::runtime_error("image engine: a W8A8 product takes rows of a multiple of 256 values, up to " +
                                 std::to_string(max_sg * G * kQiRotate));
    const int64_t nsg = (gpr + G - 1) / G;
    return qu.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * nsg * SG)), sycl::range<1>((size_t) (nsg * SG))),
        RowsI8Of<Op, W, G>{op, K, gpr, nsg, q, s});
}

// rows_i8's timing passes, as a block's: x[t] += gate y[t] (a product's int32 sums, scaled), written back, then
// RMS-normed and modulated (gate_add into the MLP's input), over 4096 rows of 3072, twice; FP16 rows of 3072 (the
// attention's output); an activation of two int32 sums side by side F apart (SiTU's hidden value and gate), over 1024
// rows of F = 8192
struct TimedI8 {
    static constexpr bool kRms = true, kKeep = true;
    float* x;
    MfOut y;
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
    MfOut y;
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
            MfOut yc, yf;
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
    MfOut a, b;
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

inline void head_max(const MfOut& a, int64_t na, const MfOut& b, int64_t nb, int64_t heads, int64_t hd, float* part,
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

inline void col_max(const MfOut& a, int64_t na, const MfOut& b, int64_t nb, int64_t C, float* part, void* s) {
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
template<class X, class G> sycl::event gate_add_by(X* x, const MfOut& h, G g, int64_t T, int64_t C, void* s) {
    if (T <= 0) return {};
    return kernels::for_rows_cols(q_of(s), T, C, [=](int64_t t, int64_t c) {
        x[t * C + c] = resid_of<X>(resid(x[t * C + c]) + g(t, c) * h.at(t, c));
    });
}

// x[t] += gate * h[t] (per channel; gate null: 1)
template<class X> inline void gate_add(X* x, const MfOut& h, const float* gate, int64_t T, int64_t C, void* s) {
    (void) gate_add_by(x, h, [=](int64_t, int64_t c) { return gate != nullptr ? gate[c] : 1.f; }, T, C, s);
}

// out[t] = x[t] + gate * h[t]: the same, x kept (a training step's backward reads it)
inline void gate_add_to(float* out, const float* x, const MfOut& h, const float* gate, int64_t T, int64_t C, void* s) {
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
// (the rows to int8, Gemm::i8, the scales back), the least of three after a warm-up: what a DiT chooses its form by
// (0: the form does not run here).  `what` names the DiT in the message of a form the library could not run.
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
                    qi_rows_i8(x16, false, T, K, x8, sx, nullptr, s_);
                    prefill::Gemm::i8(x8, w8p, reinterpret_cast<int32_t*>(y), T, N, K, N, s_);
                    qi_i8_scale(y, T, N, N, sx, sw, nullptr, nullptr, s_);
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

// Whether the DiTs' W8A8 products go through the GPU maker's fused kernel (kernels::i8_gemm_bf16: the scales and bias
// in its epilogue, BF16 out; oneDNN's on Intel, CUTLASS's on NVIDIA, comfy-kitchen's on RDNA3 and RDNA4) or the
// library's int8 product (prefill::Gemm::i8: int32 sums, scaled by their reader): the faster, timed once a GPU over an
// image DiT's three shapes at 4096 rows (3072 x 3072, 12288 x 3072, 3072 x 12288), the least of three after a warm-up
// of each.  XESTRATA_W8_FUSED=0 or 1 fixes it.
inline bool w8_fused(void* s_) {
    static core::PerDevice<int> of;
    return of.get(core::Runtime::get().stream(s_).get_device(), [s_] {
        if (const char* e = strata::env_get("W8_FUSED")) return std::atoi(e) != 0 ? 1 : 0;
        if (!kernels::i8_gemm_ok(s_)) return 0;
        if (!prefill::Gemm::i8_usable(s_)) return 1;
        constexpr int64_t T = 4096;
        const int64_t shapes[3][2] = {{3072, 3072}, {12288, 3072}, {3072, 12288}};
        std::vector<void*> held;
        auto get = [&](size_t bytes) -> void* {
            void* p = nullptr;
            if (!gpu::alloc_device(&p, bytes)) return nullptr;
            held.push_back(p);
            gpu::memset_async(p, 1, bytes, s_);
            return p;
        };
        auto best = [&](const std::function<bool()>& run) {
            double t = 1e30;
            for (int i = 0; i < 4; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                if (!run() || !gpu::stream_sync(s_)) return 1e30;
                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (i > 0) t = std::min(t, dt);
            }
            return t;
        };
        double fused = 0.0, lib = 0.0;
        try {
            auto* x8 = static_cast<int8_t*>(get((size_t) (T * 12288)));
            auto* w8 = static_cast<int8_t*>(get((size_t) (12288 * 3072)));
            auto* y = get((size_t) (T * 12288) * 4);
            auto* sx = static_cast<float*>(get((size_t) T * 4));
            auto* sw = static_cast<float*>(get((size_t) 12288 * 4));
            if (x8 == nullptr || w8 == nullptr || y == nullptr || sx == nullptr || sw == nullptr)
                throw std::bad_alloc();
            for (const auto& sh : shapes) {
                const int64_t N = sh[0], K = sh[1];
                fused += best([&] {
                    return kernels::i8_gemm_bf16(x8, sx, w8, sw, nullptr, static_cast<uint16_t*>(y), T, N, K, 0, s_);
                });
                lib += best([&] {
                    prefill::Gemm::i8(x8, w8, static_cast<int32_t*>(y), T, N, K, N, s_);
                    return true;
                });
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "image engine: timing the int8 products: %s\n", e.what());
        }
        gpu::stream_sync(s_);
        for (void* p : held) gpu::free(p);
        const int f = fused < 1e29 && (lib <= 0.0 || lib >= 1e29 || fused <= lib) ? 1 : 0;
        std::fprintf(stderr, "image engine: int8 products through %s (an image DiT's three at 4096 rows: the fused "
                             "kernel %.2f ms, the library %.2f)\n", f ? "the fused kernel" : "the library",
                     fused < 1e29 ? 1e3 * fused : -1.0, lib < 1e29 ? 1e3 * lib : -1.0);
        return f;
    }) == 1;
}

}  // namespace strata::image::dit_rows
