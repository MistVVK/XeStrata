// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_kernels.cpp - see include/strata/image/qi_kernels.hpp.

#include "strata/image/qi_kernels.hpp"
#include "strata/image/gelu.hpp"
#include "dit_rows.hpp"

#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/row_stats.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/qk_norm_rope.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strata::image {

namespace {

namespace syclex = sycl::ext::oneapi::experimental;

using strata::kernels::f32_from_bf16;
using kernels::f16_gpu::f16_from_f32;   // the GPU's conversions (f16_gpu.hpp)
using kernels::f16_gpu::f32_from_f16;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

constexpr float kEps = 1e-6f;

// a W8A8 product's scales as the kernel reads them (DitOut's sx and sw; bf: its values already scaled, as BF16)
struct I8Scales {
    const float *sx = nullptr, *sw = nullptr;
    bool bf = false;
};

// qi_qk_norm_rope16 a sub-group of 32 a head (t, h), HEADS heads a work-group
// q, k and v as FP32, or (i8) a W8A8 product's int32 sums with their scales (s[0], s[1], s[2]: q's, k's and v's,
// the activations' shared)
struct QkvIn {
    const float *q, *k, *v;
    bool i8;
    I8Scales s[3];
    float at(int which, int64_t t, int64_t idx, int64_t c) const {
        // the three by selects, not s[which]: a run-time index into the array puts it in private memory (Qwen-Image's
        // QK kernel took the B70 twice its time, 3.3 s of an image's 6.8)
        const float* p = which == 0 ? q : which == 1 ? k : v;
        if (!i8) return p[idx];
        if (which == 0 ? s[0].bf : which == 1 ? s[1].bf : s[2].bf)
            return strata::kernels::f32_from_bf16(reinterpret_cast<const uint16_t*>(p)[idx]);
        const float* sx = which == 0 ? s[0].sx : which == 1 ? s[1].sx : s[2].sx;
        const float* sw = which == 0 ? s[0].sw : which == 1 ? s[1].sw : s[2].sw;
        return (float) sycl::bit_cast<int32_t>(p[idx]) * sx[t] * sw[c];
    }
};

}  // namespace

// The SwiGLU's W8A8 input (dit_rows.hpp's RowsI8Of ops), in a namespace of their own: SYCL names a kernel by its
// type, and an anonymous namespace mangles the same in every file.  The rows' quantizer itself is kernels/rows_i8.hpp.
namespace qi_rows {

using strata::kernels::f32_from_bf16;
using kernels::f16_gpu::f16_from_f32;
using kernels::f16_gpu::f32_from_f16;

// qi_swiglu_i8's: SwiGLU of a W8A8 product's int32 sums [gate | up]
struct I8FromSwiglu {
    static constexpr bool kRms = false, kKeep = false;
    const int32_t* gu;
    const float *sx, *sw;
    int64_t inter;
    bool bf;   // gu's values already scaled, as BF16
    template<int V> void load_sg(const sycl::sub_group&, int64_t r, int64_t k, float* v) const {
        if (bf) {
            const auto* row = reinterpret_cast<const uint16_t*>(gu) + r * 2 * inter;
            for (int j = 0; j < V; ++j) {
                const float g = f32_from_bf16(row[k + j]), u = f32_from_bf16(row[inter + k + j]);
                v[j] = u * g / (1.f + sycl::exp(-g));
            }
            return;
        }
        const int32_t* row = gu + r * 2 * inter;
        for (int j = 0; j < V; ++j) {
            const float g = (float) row[k + j] * sx[r] * sw[k + j];
            const float u = (float) row[inter + k + j] * sx[r] * sw[inter + k + j];
            v[j] = u * g / (1.f + sycl::exp(-g));
        }
    }
};

}  // namespace qi_rows

void qi_ln_mod16(const float* x, const float* mod, int64_t n_text, int64_t T, int64_t C, uint16_t* y16, void* stream,
                 const float* shift) {
    if (T <= 0) return;
    // a work-group a row: 128 lanes up to 512 channels (the VAEs' pixels), 256 past them (the DiTs' 3,072)
    const size_t local = C <= 512 ? 128 : 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) T * local), sycl::range<1>(local)), [=](sycl::nd_item<1> it) {
            const int64_t t = static_cast<int64_t>(it.get_group_linear_id()), lane = static_cast<int64_t>(it.get_local_linear_id());
            const float* r = x + t * C;
            const auto st = kernels::row_stats::calculate<kernels::row_stats::Kind::variance>(
                lane, (int64_t) local, C, kEps, [=](int64_t c) { return r[c]; },
                [&](float v) { return sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()); });
            const float mean = st.mean, inv = st.inv;
            const float* m = mod + (t < n_text ? C : 0);
            for (int64_t c = lane; c < C; c += (int64_t) local) {
                // one expression each, as the two kernels this replaced wrote them (the FMA they compile to)
                y16[t * C + c] = f16_from_f32(shift != nullptr ? (r[c] - mean) * inv * (1.f + m[c]) + shift[c]
                                                               : (r[c] - mean) * inv * (1.f + m[c]));
            }
        });
    finish(stream, e, "qi_ln_mod16");
}

void qi_gate_add(float* x, const float* h, const float* mod, int64_t n_text, int64_t T, int64_t C, void* stream,
                 const DitOut* i8) {
    // the DiTs' gated residual (dit_rows::gate_add_by), the gate the text's or the image's row of mod, through tanh
    DitOut in;
    in.C = C;
    if (i8 != nullptr && i8->bf != nullptr) {
        in.bf = reinterpret_cast<const uint16_t*>(h);
    } else if (i8 != nullptr) {
        in.i = reinterpret_cast<const int32_t*>(h);
        in.sx = i8->sx;
        in.sw = i8->sw;
    } else {
        in.f = h;
    }
    const auto e = dit_rows::gate_add_by(
        x, in, [=](int64_t t, int64_t c) { return sycl::tanh(mod[(t < n_text ? C : 0) + c]); }, T, C, stream);
    finish(stream, e, "qi_gate_add");
}

void qi_qk_norm_rope16(const float* q, const float* k, const float* v, const float* wq, const float* wk,
                       const float* cos, const float* sin, int64_t T, int64_t heads, int64_t head_dim,
                       uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream, const DitOut* i8) {
    QkvIn in{q, k, v, i8 != nullptr, {}};
    if (i8 != nullptr)
        for (int j = 0; j < 3; ++j) in.s[j] = I8Scales{i8[j].sx, i8[j].sw, i8[j].bf != nullptr};
    if (T <= 0) return;
    if (head_dim % 2 != 0 || head_dim > 256) throw std::invalid_argument("qi_qk_norm_rope16: head_dim");
    // the rows [T][heads][head_dim], their pairs (2j, 2j + 1)
    struct Heads {
        QkvIn in;
        int64_t heads, hd;
        auto head(int which, int64_t t, int64_t h) const {
            // QkvIn::at's value, the row's pointers and scale taken once (the selects, not s[which]: see QkvIn)
            const int64_t c0 = h * hd;
            const float* p = (which == 0 ? in.q : which == 1 ? in.k : in.v) + t * heads * hd + c0;
            const bool bf = in.i8 && (which == 0 ? in.s[0].bf : which == 1 ? in.s[1].bf : in.s[2].bf);
            const bool i8 = in.i8 && !bf;
            const float sxt = i8 ? (which == 0 ? in.s[0].sx : which == 1 ? in.s[1].sx : in.s[2].sx)[t] : 0.f;
            const float* sw = i8 ? (which == 0 ? in.s[0].sw : which == 1 ? in.s[1].sw : in.s[2].sw) + c0 : nullptr;
            // BF16 values at the same element offsets as the FP32 ones
            const uint16_t* rb = bf ? reinterpret_cast<const uint16_t*>(
                                          which == 0 ? in.q : which == 1 ? in.k : in.v) + t * heads * hd + c0
                                    : nullptr;
            return [=](int64_t d) {
                return bf ? strata::kernels::f32_from_bf16(rb[d])
                          : i8 ? (float) sycl::bit_cast<int32_t>(p[d]) * sxt * sw[d] : p[d];
            };
        }
    };
    kernels::qk::Shape sh{T, heads, heads, head_dim};
    sh.wq = wq;
    sh.wk = wk;
    sh.eps = kEps;
    sh.neox = false;
    finish(stream, kernels::qk::qk_norm_rope(queue_for(stream), sh, Heads{in, heads, head_dim},
                                             kernels::qk::Tables{cos, sin, head_dim / 2},
                                             kernels::qk::Rows16{q16, k16, v16, heads, heads, head_dim}),
           "qi_qk_norm_rope16");
}

void qi_swiglu_i8(const int32_t* gu, const DitOut& g, int64_t T, int64_t inter, int8_t* q, float* s, void* stream) {
    kernels::rows_i8_of(qi_rows::I8FromSwiglu{gu, g.sx, g.sw, inter, g.bf != nullptr}, T, inter, q, s, stream);
}

void qi_rows_scale(float* y, int64_t T, int64_t N, int64_t ldy, const float* col, void* stream) {
    if (T <= 0 || N <= 0 || col == nullptr) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<2>((size_t) T, (size_t) N), [=](sycl::id<2> i) {
        y[(int64_t) i[0] * ldy + (int64_t) i[1]] *= col[(int64_t) i[1]];
    });
    finish(stream, e, "qi_rows_scale");
}

void qi_silu(float* x, int64_t n, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i] / (1.f + sycl::exp(-x[i]));
        x[i] = v;
        if (y16) y16[i] = f16_from_f32(v);
    });
    finish(stream, e, "qi_silu");
}

void qi_latent_tokens16(const float* latent, int64_t C, int64_t HW, uint16_t* tokens16, void* stream) {
    if (HW <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (C * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C;
        tokens16[i] = f16_from_f32(latent[c * HW + p]);
    });
    finish(stream, e, "qi_latent_tokens16");
}

void qi_tokens_latent(const float* tokens, int64_t C, int64_t HW, float* latent, void* stream) {
    if (HW <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (C * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / HW, p = i - c * HW;
        latent[i] = tokens[p * C + c];
    });
    finish(stream, e, "qi_tokens_latent");
}

void qi_inpaint_blend(float* x, const float* z0, const float* noise, const float* m, float s, int64_t C, int64_t HW,
                      void* stream, bool tokens) {
    if (C * HW <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (C * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const float k = m[tokens ? i / C : i % HW];
        x[i] = k * x[i] + (1.f - k) * ((1.f - s) * z0[i] + s * noise[i]);
    });
    finish(stream, e, "qi_inpaint_blend");
}

void qi_noise_mix(float* x, const float* z0, const float* noise, float s, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        x[i] = (1.f - s) * z0[i] + s * noise[i];
    });
    finish(stream, e, "qi_noise_mix");
}

namespace {
// KT and K known (the VAEs' 3 x 3: its divisions by constants), else 0 and kt, k read at run time.  CH is the
// channels one work item moves: 8 where they and the row stride are both a multiple of 8 (one read and one write of
// 8 channels), 1 where they are not (a 3-channel picture's patch embedding: SAM 3, BiRefNet, Florence-2).
template <int KT, int K, int CH>
sycl::event im2col_launch(sycl::queue& q, const uint16_t* in16, int64_t T, int64_t H, int64_t W, int64_t C, int kt_,
                          int k_, int s, int pad, int dil, int64_t OH, int64_t OW, int64_t p0, int64_t rows,
                          uint16_t* col16, int64_t ldx) {
    const int64_t kt = KT > 0 ? KT : kt_, k = K > 0 ? K : k_, kk2 = k * k, taps = kt * kk2, nc = C / CH;
    return q.parallel_for(sycl::range<1>((size_t) (rows * taps * nc)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t rk = i / nc, r = rk / taps, tap = rk - r * taps, c = (i - rk * nc) * CH;
        const int64_t pq = p0 + r, ox = pq % OW, oyt = pq / OW, oy = oyt % OH, t = oyt / OH;
        const int64_t kk = tap % kk2, tt = t + (kt == 3 ? (tap / kk2 - 1) * dil : 0);
        const int64_t y = oy * s + (kk / k) * dil - pad, x = ox * s + (kk % k) * dil - pad;
        const bool inside = tt >= 0 && tt < T && y >= 0 && y < H && x >= 0 && x < W;
        if constexpr (CH == 1) {
            col16[rk * C + c] = inside ? in16[((tt * H + y) * W + x) * ldx + c] : (uint16_t) 0;
        } else {
            using V = sycl::vec<uint16_t, 8>;
            V v(0);
            if (inside) v = *reinterpret_cast<const V*>(in16 + ((tt * H + y) * W + x) * ldx + c);
            *reinterpret_cast<V*>(col16 + rk * C + c) = v;
        }
    });
}
}  // namespace

void qi_im2col16(const uint16_t* in16, int64_t T, int64_t H, int64_t W, int64_t C, int kt, int k, int s, int pad,
                 int64_t OH, int64_t OW, int64_t p0, int64_t rows, uint16_t* col16, void* stream, int64_t ldx,
                 int dil) {
    if (rows <= 0 || C <= 0 || k <= 0) return;
    if (ldx == 0) ldx = C;
    if (ldx < C || (kt != 1 && kt != 3) || dil < 1) throw std::invalid_argument("qi_im2col16: ldx, kt or dil");
    sycl::queue& q = queue_for(stream);
    // Eight channels at a time wherever a pixel's channels and its row stride both take the aligned vectors: the
    // engine's own tensors and the aux nets' maps (their channels padded to 8), which is every unfold but a picture's
    // 3-channel patch embedding (SAM 3, BiRefNet, Florence-2), left one value at a time as it was
    const bool wide = C % 8 == 0 && ldx % 8 == 0;
    const sycl::event e =
        !wide   ? im2col_launch<0, 0, 1>(q, in16, T, H, W, C, kt, k, s, pad, dil, OH, OW, p0, rows, col16, ldx)
        : k != 3 ? im2col_launch<0, 0, 8>(q, in16, T, H, W, C, kt, k, s, pad, dil, OH, OW, p0, rows, col16, ldx)
        : kt == 1 ? im2col_launch<1, 3, 8>(q, in16, T, H, W, C, kt, k, s, pad, dil, OH, OW, p0, rows, col16, ldx)
                  : im2col_launch<3, 3, 8>(q, in16, T, H, W, C, kt, k, s, pad, dil, OH, OW, p0, rows, col16, ldx);
    finish(stream, e, "qi_im2col16");
}

void product_store(const ProductStore& p, void* stream) {
    if (p.rows <= 0 || p.cols <= 0) return;
    const int64_t is = p.in_stride ? p.in_stride : p.cols, os = p.out_stride ? p.out_stride : p.cols;
    const auto calculate = [=](float v, int64_t c, uint16_t residual) {
        if (p.bias) v += p.bias[c];
        else if (p.add_zero) v += 0.f;
        v = image_activate(p.act, v, p.slope);
        if (p.residual) v += p.bf ? kernels::f32_from_bf16(residual) : f32_from_f16(residual);
        return v;
    };
    // Vector loads need every row and offset aligned, including the optional second destination.
    const auto aligned = [](const void* a, uintptr_t n) { return !a || reinterpret_cast<uintptr_t>(a) % n == 0; };
    const bool vec = p.cols % 8 == 0 && is % 8 == 0 && os % 8 == 0 && p.offset % 8 == 0 &&
        (!p.out2 || p.from2 % 8 == 0) && aligned(p.in, 32) && aligned(p.in16, 16) &&
        aligned(p.residual, 16) && aligned(p.out16, 16) && aligned(p.out2, 16) && aligned(p.out, 32);
    // Contiguous FP16 rows need neither per-element conversion through integer bits nor generic stride handling.
    // Synchronized warm calls, before -> this path: 2304x4096 GELU 124 -> 53 us (RTX 4070),
    // 576x8192 GELU 43 -> 36 us (B70). Both are 8-value paths; the scalar path remains for unaligned rows.
    if (vec && p.in16 && p.out16 && !p.out && !p.out2 && !p.residual && !p.bf &&
        is == p.cols && os == p.cols && p.offset == 0) {
        using V = sycl::vec<sycl::half, 8>;
        const auto in = p.in16; const auto out = p.out16; const auto bias = p.bias;
        const auto cols = p.cols; const auto act = p.act; const auto slope = p.slope;
        const bool add_zero = p.add_zero;
        const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t)(p.rows * cols / 8)), [=](sycl::id<1> id) {
            const int64_t i = (int64_t)id[0], c = (i % (cols / 8)) * 8;
            const V x = reinterpret_cast<const V*>(in)[i];
            V y;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float v = (float)x[j];
                if (bias) v += bias[c + j]; else if (add_zero) v += 0.f;
                y[j] = sycl::half(image_activate(act, v, slope));
            }
            reinterpret_cast<V*>(out)[i] = y;
        });
        finish(stream, e, "product_store");
        return;
    }
    sycl::event e;
    if (vec) {
        e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (p.rows * p.cols / 8)), [=](sycl::id<1> id) {
            const int64_t k = (int64_t) id[0] * 8, r = k / p.cols, c = k % p.cols;
            sycl::vec<float, 8> x;
            if (p.in) x = *reinterpret_cast<const sycl::vec<float, 8>*>(p.in + r * is + c);
            else {
                auto h = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.in16 + r * is + c);
                for (int j = 0; j < 8; ++j) x[j] = f32_from_f16(h[j]);
            }
            sycl::vec<uint16_t, 8> res(0), out;
            if (p.residual) res = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.residual + k);
            for (int j = 0; j < 8; ++j) {
                const float v = calculate(x[j], c + j, res[j]);
                if (p.out) x[j] = v;
                if (p.out16 || p.out2) out[j] = p.bf ? kernels::bf16_from_f32(v) : f16_from_f32(v);
            }
            if (p.out) *reinterpret_cast<sycl::vec<float, 8>*>(p.out + r * os + p.offset + c) = x;
            if (p.out16) *reinterpret_cast<sycl::vec<uint16_t, 8>*>(p.out16 + r * os + p.offset + c) = out;
            if (p.out2 && c >= p.from2) *reinterpret_cast<sycl::vec<uint16_t, 8>*>(p.out2 + r * (p.cols - p.from2) + c - p.from2) = out;
        });
    } else {
        e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (p.rows * p.cols)), [=](sycl::id<1> id) {
            const int64_t k = (int64_t) id[0], r = k / p.cols, c = k % p.cols;
            float v = p.in ? p.in[r * is + c] : f32_from_f16(p.in16[r * is + c]);
            v = calculate(v, c, p.residual ? p.residual[k] : uint16_t{0});
            if (p.out) p.out[r * os + p.offset + c] = v;
            const uint16_t h = p.bf ? kernels::bf16_from_f32(v) : f16_from_f32(v);
            if (p.out16) p.out16[r * os + p.offset + c] = h;
            if (p.out2 && c >= p.from2) p.out2[r * (p.cols - p.from2) + c - p.from2] = h;
        });
    }
    finish(stream, e, "product_store");
}

void qi_conv_store16(const float* y, int64_t T, int64_t O, int64_t ldy, const float* bias, ImageAct act, float slope,
                     const uint16_t* res, uint16_t* dst, int64_t ld, int64_t off, void* stream, uint16_t* dst2,
                     int64_t from2) {
    ProductStore p;
    p.in = y; p.rows = T; p.cols = O; p.in_stride = ldy; p.bias = bias; p.act = act; p.slope = slope;
    p.residual = res; p.out16 = dst; p.out_stride = ld; p.offset = off; p.out2 = dst2; p.from2 = from2;
    product_store(p, stream);
}

namespace {
template <class T>
void up2x16(const T* x, int64_t H, int64_t W, int64_t C, uint16_t* out16, void* stream) {
    if (H * W * C <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (4 * H * W * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C, Y = p / (2 * W), X = p - Y * 2 * W;
        const T v = x[((Y / 2) * W + X / 2) * C + c];
        if constexpr (std::is_same_v<T, float>) out16[i] = f16_from_f32(v);
        else out16[i] = v;
    });
    finish(stream, e, "qi_up2x16");
}
}  // namespace

void qi_up2x16(const float* x, int64_t H, int64_t W, int64_t C, uint16_t* out16, void* stream) {
    up2x16(x, H, W, C, out16, stream);
}

void qi_up2x16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, uint16_t* out16, void* stream) {
    up2x16(x16, H, W, C, out16, stream);
}

namespace {
// GroupNorm statistics: a work-group a chunk and group, with at most kGnMaxChunk chunks per reducer.
constexpr int64_t kGnChunk = 256;      // positions a chunk
constexpr int64_t kGnMaxChunk = 256;   // chunks a reducer's work-group

// a GroupNorm's input: FP32, or 16-bit (FP16 or BF16 bits) activations
struct GnIn {
    const void* p;
    int kind;   // 0 FP32, 1 FP16, 2 BF16
    float operator[](int64_t i) const {
        if (kind == 0) return static_cast<const float*>(p)[i];
        const uint16_t h = static_cast<const uint16_t*>(p)[i];
        return kind == 1 ? f32_from_f16(h) : kernels::f32_from_bf16(h);
    }
};

void group_part(GnIn x, const float* mean, int64_t P, int64_t C, int64_t groups, bool sq, float* part, void* stream) {
    if (P <= 0) return;
    const int64_t cg = C / groups, B = (P + kGnChunk - 1) / kGnChunk;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (B * groups * 32)), sycl::range<1>(32)), [=](sycl::nd_item<1> it) {
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const auto blk = static_cast<int64_t>(it.get_group_linear_id());
            const int64_t b = blk / groups, g = blk - b * groups;
            const int64_t p1 = std::min<int64_t>(P, (b + 1) * kGnChunk);
            const float m = sq ? mean[g] : 0.f;
            float a = 0.f, comp = 0.f;
            for (int64_t p = b * kGnChunk + lane; p < p1; p += 32) {
                const int64_t r = p * C + g * cg;
                for (int64_t c = 0; c < cg; ++c) {
                    const float d = x[r + c] - m;
                    const float y = (sq ? d * d : d) - comp, t = a + y;
                    comp = (t - a) - y;
                    a = t;
                }
            }
            a = sycl::reduce_over_group(it.get_group(), a, sycl::plus<float>());
            if (lane == 0) part[blk] = a;
        });
    (void) e;
}

void group_stats(const float* part, int64_t B, int64_t P, int64_t CG, int64_t groups, float eps, bool sq,
                               float* stats, float* mid, void* stream) {
    const int64_t G = (B + kGnMaxChunk - 1) / kGnMaxChunk;
    if (G > 1) {
        const auto e = queue_for(stream).parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) (G * groups * 32)), sycl::range<1>(32)),
            [=](sycl::nd_item<1> it) {
                const auto lane = static_cast<int64_t>(it.get_local_linear_id());
                const auto blk = static_cast<int64_t>(it.get_group_linear_id());
                const int64_t gi = blk / groups, g = blk - gi * groups;
                const int64_t b1 = std::min<int64_t>(B, (gi + 1) * kGnMaxChunk);
                float a = 0.f, comp = 0.f;
                for (int64_t b = gi * kGnMaxChunk + lane; b < b1; b += 32) {
                    const float y = part[b * groups + g] - comp, t = a + y;
                    comp = (t - a) - y;
                    a = t;
                }
                a = sycl::reduce_over_group(it.get_group(), a, sycl::plus<float>());
                if (lane == 0) mid[gi * groups + g] = a;
            });
        (void) e;
        queue_for(stream).wait();   // the last level reads every group's entry
    }
    // one work-item a group, serial Kahan over the (at most kGnMaxChunk) entries of its slice, or over the B chunk
    // sums themselves when they fit one reducer
    const int64_t MG = G > 1 ? G : B;
    const float* src = G > 1 ? mid : part;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) groups), [=](sycl::id<1> id) {
        const auto g = static_cast<int64_t>(id[0]);
        float a = 0.f, comp = 0.f;
        for (int64_t i = 0; i < MG; ++i) {
            const float y = src[i * groups + g] - comp, t = a + y;
            comp = (t - a) - y;
            a = t;
        }
        const float n = (float) P * (float) CG;   // FP32 here too (no device FP64): the sums are Kahan-folded
        stats[g] = sq ? 1.0f / sycl::sqrt(a / n + eps) : a / n;   // and a 1e-7 share of a ratio is far under the
    });                                                          // FP16 floor the convolutions set
    (void) e;
}
// The group norm's statistics in one pass over x: chunks of kGnChunk pixels, a work-group a chunk whose lanes read
// whole rows of channels (a lane one channel, R rows at a time: C R lanes), each lane its values' mean and squared
// deviations, merged within each group (Chan's): part[(k groups + g) 3 + 0, 1, 2] = the count, the mean
// and the squared deviations of chunk k's group g.  (A work-group a chunk and group, reading its C / 32 channels of
// each pixel, and a second pass for the squares took MOSS-Image's VAE 64 ms of its 170 ms decode on the B70.)
int64_t gn_chunks(int64_t HW) { return (HW + kGnChunk - 1) / kGnChunk; }
int64_t gn_rows(int64_t C) { return sycl::max<int64_t>(1, 256 / C); }

void gn_partial(GnIn x, int64_t HW, int64_t C, int64_t groups, float* part, void* s) {
    const int64_t cg = C / groups, nk = gn_chunks(HW), R = gn_rows(C), local = C * R;
    queue_for(s).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ln(sycl::range<1>((size_t) local), h), lm(sycl::range<1>((size_t) local), h),
            lq(sycl::range<1>((size_t) local), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (nk * local)), sycl::range<1>((size_t) local)),
                       [=](sycl::nd_item<1> it) {
            const auto k = static_cast<int64_t>(it.get_group_linear_id());
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t r = lane / C, c = lane - r * C, p0 = k * kGnChunk, p1 = sycl::min(p0 + kGnChunk, HW);
            // sums about the lane's first value (no division a value; the shift keeps them accurate in FP32)
            const float k0 = p0 + r < p1 ? x[(p0 + r) * C + c] : 0.f;
            float n = 0.f, s1 = 0.f, s2 = 0.f;
            for (int64_t p = p0 + r; p < p1; p += R) {
                const float d = x[p * C + c] - k0;
                n += 1.f;
                s1 += d;
                s2 += d * d;
            }
            ln[lane] = n;
            lm[lane] = n > 0.f ? k0 + s1 / n : 0.f;
            lq[lane] = n > 0.f ? s2 - s1 * s1 / n : 0.f;
            sycl::group_barrier(it.get_group());
            if (lane < groups) {
                float N = 0.f, M = 0.f, Q = 0.f;
                for (int64_t rr = 0; rr < R; ++rr)
                    for (int64_t j = 0; j < cg; ++j) {
                        const int64_t e = rr * C + lane * cg + j;
                        const float nb = ln[e];
                        if (nb == 0.f) continue;
                        const float t = N + nb, d = lm[e] - M;
                        M += d * nb / t;
                        Q += lq[e] + d * d * N * nb / t;
                        N = t;
                    }
                float* o = part + (k * groups + lane) * 3;
                o[0] = N;
                o[1] = M;
                o[2] = Q;
            }
        });
    });
}

// stats[2 g] = group g's mean over every pixel, stats[2 g + 1] = 1 / its std, gn_partial's chunks merged (Chan's): a
// work-group a group, each lane a stride of the chunks, then the lanes in a tree
void gn_stats(const float* part, int64_t HW, int64_t C, int64_t groups, float eps, float* mean, float* inv, void* s) {
    (void) C;
    constexpr int64_t local = 256;
    const int64_t nk = gn_chunks(HW);
    queue_for(s).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ln(sycl::range<1>((size_t) local), h), lm(sycl::range<1>((size_t) local), h),
            lq(sycl::range<1>((size_t) local), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (groups * local)), sycl::range<1>((size_t) local)),
                       [=](sycl::nd_item<1> it) {
            const auto g = static_cast<int64_t>(it.get_group_linear_id());
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const auto merge = [](float& N, float& M, float& Q, float nb, float mb, float qb) {
                if (nb == 0.f) return;
                const float t = N + nb, d = mb - M;
                M += d * nb / t;
                Q += qb + d * d * N * nb / t;
                N = t;
            };
            float N = 0.f, M = 0.f, Q = 0.f;
            for (int64_t k = lane; k < nk; k += local) {
                const float* e = part + (k * groups + g) * 3;
                merge(N, M, Q, e[0], e[1], e[2]);
            }
            ln[lane] = N;
            lm[lane] = M;
            lq[lane] = Q;
            for (int64_t half = local / 2; half > 0; half /= 2) {
                sycl::group_barrier(it.get_group());
                if (lane < half) {
                    merge(N, M, Q, ln[lane + half], lm[lane + half], lq[lane + half]);
                    ln[lane] = N;
                    lm[lane] = M;
                    lq[lane] = Q;
                }
            }
            if (lane == 0) {
                mean[g] = M;
                inv[g] = 1.f / sycl::sqrt(Q / N + eps);
            }
        });
    });
}

}  // namespace

int64_t qi_group_norm_ws_floats(int64_t P, int64_t groups) {
    const int64_t B = (P + kGnChunk - 1) / kGnChunk, G = (B + kGnMaxChunk - 1) / kGnMaxChunk;
    return 2 * groups + std::max((B + G) * groups, B * groups * 3);   // the two passes' sums, or the one's
}

namespace {

void group_norm16(GnIn x, int64_t P, int64_t C, int64_t groups, const float* w, const float* b, float eps, ImageAct act, float* y, uint16_t* y16,
                  float* ws, void* stream, const float* scale, const float* shift, uint16_t* lo16) {
    if (P <= 0) return;
    if (groups <= 0 || C <= 0 || C % groups != 0 || (!y && !y16)) throw std::invalid_argument("qi_group_norm16: C");
    float* mean = ws, *inv = ws + groups, *part = ws + 2 * groups;
    const int64_t B = (P + kGnChunk - 1) / kGnChunk, CG = C / groups;
    // One pass where a pixel's channels fit a work-group's lanes (the VAEs' up to 1,024: the two passes took
    // MOSS-Image's VAE 64 ms of its 170 ms decode on the B70), two (a mean, then the squares) past them (HunyuanImage
    // 3.0's entry and exit, 4,096 channels)
    static core::PerDevice<size_t> max_wg;
    sycl::queue& q = queue_for(stream);
    const size_t wg = max_wg.get(q.get_device(),
                                 [&q] { return q.get_device().get_info<sycl::info::device::max_work_group_size>(); });
    if ((size_t) (C * gn_rows(C)) <= wg) {
        gn_partial(x, P, C, groups, part, stream);
        gn_stats(part, P, C, groups, eps, mean, inv, stream);
    } else {
        float* mid = part + B * groups;
        group_part(x, nullptr, P, C, groups, false, part, stream);
        group_stats(part, B, P, CG, groups, eps, false, mean, mid, stream);
        group_part(x, mean, P, C, groups, true, part, stream);
        group_stats(part, B, P, CG, groups, eps, true, inv, mid, stream);
    }
    if (C % 8 != 0) {
        const auto e = q.parallel_for(sycl::range<1>((size_t) (P * C)), [=](sycl::id<1> id) {
            const int64_t i = static_cast<int64_t>(id[0]), c = i % C, g = c / CG;
            float v = (x[i] - mean[g]) * inv[g];
            if (w != nullptr) v = v * w[c] + b[c];
            if (scale != nullptr) v = v * (1.f + scale[c]) + shift[c];
            v = image_activate(act, v);
            if (y != nullptr) y[i] = v;
            const uint16_t h = f16_from_f32(v);
            if (y16 != nullptr) y16[i] = h;
            if (lo16 != nullptr) lo16[i] = f16_from_f32(v - f32_from_f16(h));
        });
        finish(stream, e, "qi_group_norm16");
        return;
    }
    // 8 values a work-item, in one load and one store: a value each took 30 ms of MOSS-Image's decode on the B70
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (P * C / 8)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]) * 8, c0 = i % C;
        sycl::vec<float, 8> xv;
        if (x.kind == 0) {
            xv = *reinterpret_cast<const sycl::vec<float, 8>*>(static_cast<const float*>(x.p) + i);
        } else {
            const sycl::vec<uint16_t, 8> hv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(
                static_cast<const uint16_t*>(x.p) + i);
            for (int j = 0; j < 8; ++j) xv[j] = x.kind == 1 ? f32_from_f16(hv[j]) : kernels::f32_from_bf16(hv[j]);
        }
        sycl::vec<uint16_t, 8> o, lo;
        for (int j = 0; j < 8; ++j) {
            const int64_t c = c0 + j, g = c / CG;
            float v = (xv[j] - mean[g]) * inv[g];
            if (w != nullptr) v = v * w[c] + b[c];
            if (scale != nullptr) v = v * (1.f + scale[c]) + shift[c];
            v = image_activate(act, v);
            if (y != nullptr) y[i + j] = v;
            o[j] = f16_from_f32(v);
            lo[j] = f16_from_f32(v - f32_from_f16(o[j]));
        }
        if (y16 != nullptr) *reinterpret_cast<sycl::vec<uint16_t, 8>*>(y16 + i) = o;
        if (lo16 != nullptr) *reinterpret_cast<sycl::vec<uint16_t, 8>*>(lo16 + i) = lo;
    });
    finish(stream, e, "qi_group_norm16");
}

}  // namespace

void qi_group_norm16(const float* x, int64_t P, int64_t C, const float* w, const float* b, float eps, bool silu,
                     uint16_t* y16, float* ws, void* stream, const float* scale, const float* shift, uint16_t* lo16, int64_t groups) {
    group_norm16(GnIn{x, 0}, P, C, groups, w, b, eps, silu ? ImageAct::silu : ImageAct::none, nullptr, y16, ws, stream, scale, shift, lo16);
}

void qi_group_norm16(const uint16_t* x16, bool bf16, int64_t P, int64_t C, const float* w, const float* b, float eps,
                     bool silu, uint16_t* y16, float* ws, void* stream, int64_t groups) {
    group_norm16(GnIn{x16, bf16 ? 2 : 1}, P, C, groups, w, b, eps, silu ? ImageAct::silu : ImageAct::none, nullptr, y16, ws, stream, nullptr, nullptr, nullptr);
}
void image_group_norm(const float* x, const uint16_t* x16, int64_t P, int64_t C, int64_t groups, const float* w,
                      const float* b, float eps, ImageAct act, float* y, uint16_t* y16, float* ws, void* stream) {
    if ((x == nullptr) == (x16 == nullptr) || (w == nullptr) != (b == nullptr)) throw std::invalid_argument("image_group_norm: input or affine");
    group_norm16(GnIn{x ? static_cast<const void*>(x) : static_cast<const void*>(x16), x ? 0 : 1}, P, C, groups,
                 w, b, eps, act, y, y16, ws, stream, nullptr, nullptr, nullptr);
}


void qi_softmax16(const float* sc, int64_t rows, int64_t n, float scale, uint16_t* p16, void* stream, int64_t ld) {
    if (rows <= 0) return;
    if (ld == 0) ld = n;
    sycl::queue& q = queue_for(stream);
    // one pass's lanes over a row: the sub-group's own sums (no local memory, no barrier) where a row is short, a
    // work-group's where it is long enough to need more lanes
    auto row_softmax = [=](auto& grp, int64_t r, int64_t lane, int64_t lanes) {
        const float* row = sc + r * ld;
        float m = -INFINITY;
        for (int64_t j = lane; j < n; j += lanes) m = sycl::fmax(m, row[j] * scale);
        m = sycl::reduce_over_group(grp, m, sycl::maximum<float>());
        float sum = 0.f;
        for (int64_t j = lane; j < n; j += lanes) sum += sycl::exp(row[j] * scale - m);
        const float inv = 1.f / sycl::reduce_over_group(grp, sum, sycl::plus<float>());
        for (int64_t j = lane; j < ld; j += lanes)
            p16[r * ld + j] = j < n ? f16_from_f32(sycl::exp(row[j] * scale - m) * inv) : (uint16_t) 0;
    };
    sycl::event e;
    if (n <= 1024) {
        constexpr int64_t SG = 32, ROWS = 8;
        e = q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((rows + ROWS - 1) / ROWS * ROWS * SG)),
                                             sycl::range<1>((size_t) (ROWS * SG))),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            auto sg = it.get_sub_group();
            const int64_t r = (int64_t) it.get_group_linear_id() * ROWS + (int64_t) sg.get_group_linear_id();
            if (r >= rows) return;   // the whole sub-group
            row_softmax(sg, r, (int64_t) sg.get_local_linear_id(), SG);
        });
    } else {
        constexpr size_t local = 256;
        e = q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * local), sycl::range<1>(local)),
                           [=](sycl::nd_item<1> it) {
            auto g = it.get_group();
            row_softmax(g, (int64_t) it.get_group_linear_id(), (int64_t) it.get_local_linear_id(), (int64_t) local);
        });
    }
    finish(stream, e, "qi_softmax16");
}

}  // namespace strata::image
