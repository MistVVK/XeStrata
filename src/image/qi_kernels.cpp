// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_kernels.cpp - see include/strata/image/qi_kernels.hpp.

#include "strata/image/qi_kernels.hpp"
#include "strata/image/gelu.hpp"
#include "dit_rows.hpp"

#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/qk_norm_rope.hpp"
#include "strata/kernels/q8_act.hpp"

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

// qi_qk_norm_rope16 a sub-group of 32 a head (t, h), HEADS heads a work-group
// q, k and v as FP32, or (i8) a W8A8 product's int32 sums with their scales (s[0], s[1], s[2]: q's, k's and v's,
// the activations' shared)
struct QkvIn {
    const float *q, *k, *v;
    bool i8;
    QiI8 s[3];
    float at(int which, int64_t t, int64_t idx, int64_t c) const {
        // the three by selects, not s[which]: a run-time index into the array puts it in private memory (Qwen-Image's
        // QK kernel took the B70 twice its time, 3.3 s of an image's 6.8)
        const float* p = which == 0 ? q : which == 1 ? k : v;
        if (!i8) return p[idx];
        const float* sx = which == 0 ? s[0].sx : which == 1 ? s[1].sx : s[2].sx;
        const float* sw = which == 0 ? s[0].sw : which == 1 ? s[1].sw : s[2].sw;
        return (float) sycl::bit_cast<int32_t>(p[idx]) * sx[t] * sw[c];
    }
};

}  // namespace

// The W8A8 inputs' sources (dit_rows.hpp's RowsI8Of ops), in a namespace of their own: SYCL names a kernel by its
// type, and an anonymous namespace mangles the same in every file.
namespace qi_rows {

using strata::kernels::f32_from_bf16;
using kernels::f16_gpu::f16_from_f32;
using kernels::f16_gpu::f32_from_f16;

// qi_rows_i8's values: V of row r from column k, FP16 or BF16
struct I8From16 {
    static constexpr bool kRms = false, kKeep = false;
    const uint16_t* src;
    bool bf16;
    int64_t K;
    const float* rs;   // the rows' factors on their scales (null: 1)
    template<int V> void load_sg(const sycl::sub_group&, int64_t r, int64_t k, float* v) const {
        const auto raw = *reinterpret_cast<const sycl::vec<uint16_t, V>*>(src + r * K + k);
        for (int j = 0; j < V; ++j) v[j] = bf16 ? f32_from_bf16(raw[j]) : f32_from_f16(raw[j]);
    }
    float row_scale(int64_t r) const { return rs != nullptr ? rs[r] : 1.f; }
};

// qi_rows_i8_blocks's: a row of GGUF blocks, each value as kernels::dequant_f16 writes it (rounded to FP16), so the
// int8 rows are those of qi_rows_i8 of its output.  RowsI8Of gives a sub-group one block (256 values, 8 a lane): its
// lanes read the block's bytes 8 a lane in 16-bit loads side by side (a block is 2-byte aligned) and take the bytes
// each value needs from the lane that read them, not a byte at a time from all over the block.
template<int TYPE> struct I8FromBlocks {
    static constexpr bool kRms = false, kKeep = false;
    static constexpr int BYTES = TYPE == 12 ? 144 : TYPE == 13 ? 176 : 210;
    const uint8_t* b;
    int64_t row_bytes;
    // inlined: else its stores through v put the values on the stack (RTX 4070, 2026-11-05: 8 stores and 4 loads a
    // lane's 8 values)
    template<int V>
    __attribute__((always_inline)) void load_sg(const sycl::sub_group& sg, int64_t r, int64_t k, float* v) const {
        static_assert(V == 8, "a lane's 8 values: 32 lanes a block");
        const uint8_t* blk = b + r * row_bytes + (k / 256) * BYTES;
        const int lane = (int) sg.get_local_linear_id();
        uint32_t w0 = 0, w1 = 0;   // the block's bytes 8 lane .. 8 lane + 7
        const auto* p = reinterpret_cast<const uint16_t*>(blk + (ptrdiff_t) 8 * lane);
        if (8 * lane + 8 <= BYTES) {
            w0 = (uint32_t) p[0] | (uint32_t) p[1] << 16;
            w1 = (uint32_t) p[2] | (uint32_t) p[3] << 16;
        } else if (8 * lane < BYTES) {   // Q6_K's d, the last 2 bytes
            w0 = p[0];
        }
        auto byte = [](uint32_t lo, uint32_t hi, int i) { return (int) (((i < 4 ? lo : hi) >> (8 * (i & 3))) & 0xFF); };
        auto f16 = [](uint32_t bits) { return f32_from_f16((uint16_t) bits); };
        if constexpr (TYPE == 14) {   // Q6_K: ql[128] qh[64] scales[16] d, 256 values
            const int n = lane / 16, qu = lane % 16 / 4, l4 = lane % 4;
            const int lq = 8 * n + 4 * (qu & 1) + l4, lh = 16 + 4 * n + l4, ls = 24 + n;
            const uint32_t q0 = sycl::select_from_group(sg, w0, lq), q1 = sycl::select_from_group(sg, w1, lq);
            const uint32_t h0 = sycl::select_from_group(sg, w0, lh), h1 = sycl::select_from_group(sg, w1, lh);
            const uint32_t s0 = sycl::select_from_group(sg, w0, ls), s1 = sycl::select_from_group(sg, w1, ls);
            const float d = f16(sycl::select_from_group(sg, w0, 26));
            const float sc = d * (float) (int8_t) byte(s0, s1, l4 / 2 + 2 * qu);
#pragma unroll
            for (int j = 0; j < V; ++j) {
                const int ql = byte(q0, q1, j), lo = qu < 2 ? (ql & 0xF) : (ql >> 4);
                const int q = lo | ((byte(h0, h1, j) >> (2 * qu)) & 3) << 4;
                v[j] = f32_from_f16(f16_from_f32(sc * (float) (q - 32)));
            }
        } else {   // Q4_K (12): d dmin scales[12] qs[128]; Q5_K (13): d dmin scales[12] qh[32] qs[128]
            const int gi = lane / 4, j64 = gi / 2, hi = gi % 2, l4 = lane % 4;
            const uint32_t a0 = sycl::group_broadcast(sg, w0, 0), a1 = sycl::group_broadcast(sg, w1, 0);
            const uint32_t c0 = sycl::group_broadcast(sg, w0, 1), c1 = sycl::group_broadcast(sg, w1, 1);
            const int lq = (TYPE == 12 ? 2 : 6) + 4 * j64 + l4;
            const uint32_t q0 = sycl::select_from_group(sg, w0, lq), q1 = sycl::select_from_group(sg, w1, lq);
            uint32_t h0 = 0, h1 = 0;
            if constexpr (TYPE == 13) {
                h0 = sycl::select_from_group(sg, w0, 2 + l4);
                h1 = sycl::select_from_group(sg, w1, 2 + l4);
            }
            const float d = f16(a0 & 0xFFFF), dmin = f16(a0 >> 16);
            auto scb = [&](int i) { return i < 4 ? byte(a1, a1, i) : byte(c0, c1, i - 4); };   // scales[i]
            int sc, m;
            if (gi < 4) {
                sc = scb(gi) & 63;
                m = scb(gi + 4) & 63;
            } else {
                sc = (scb(gi + 4) & 0xF) | ((scb(gi - 4) >> 6) << 4);
                m = (scb(gi + 4) >> 4) | ((scb(gi) >> 6) << 4);
            }
            const float d1 = d * (float) sc, m1 = dmin * (float) m;
#pragma unroll
            for (int j = 0; j < V; ++j) {
                int nib = hi ? (byte(q0, q1, j) >> 4) : (byte(q0, q1, j) & 0xF);
                if constexpr (TYPE == 13) nib += (byte(h0, h1, j) & (1 << (2 * j64 + hi))) ? 16 : 0;
                v[j] = f32_from_f16(f16_from_f32(d1 * (float) nib - m1));
            }
        }
    }
};

// qi_swiglu_i8's: SwiGLU of a W8A8 product's int32 sums [gate | up]
struct I8FromSwiglu {
    static constexpr bool kRms = false, kKeep = false;
    const int32_t* gu;
    const float *sx, *sw;
    int64_t inter;
    template<int V> void load_sg(const sycl::sub_group&, int64_t r, int64_t k, float* v) const {
        const int32_t* row = gu + r * 2 * inter;
        for (int j = 0; j < V; ++j) {
            const float g = (float) row[k + j] * sx[r] * sw[k + j];
            const float u = (float) row[inter + k + j] * sx[r] * sw[inter + k + j];
            v[j] = u * g / (1.f + sycl::exp(-g));
        }
    }
};

// qi_rows_i8 through dit_rows.hpp's RowsI8Of, 8 values side by side a lane (a GGUF block's 256 values a sub-group)
// and as few groups a sub-group as the work-group allows: the registers of 4 held the RTX 4070's occupancy to a third
// where 1 was enough (2026-11-05)
template<class Src>
void rows_i8(const Src& src, int64_t rows, int64_t K, int8_t* q, float* s, void* stream) {
    if (rows <= 0) return;
    sycl::queue& qu = queue_for(stream);
    const int64_t gpr = K / kQiRotate;
    const auto max_sg = static_cast<int64_t>(qu.get_device().get_info<sycl::info::device::max_work_group_size>()) / 32;
    const int64_t per = (gpr + max_sg - 1) / max_sg;
    const auto e = per <= 1 ? dit_rows::rows_i8_as<Src, 8, 1>(src, rows, K, q, s, qu)
                 : per == 2 ? dit_rows::rows_i8_as<Src, 8, 2>(src, rows, K, q, s, qu)
                            : dit_rows::rows_i8_as<Src, 8, 4>(src, rows, K, q, s, qu);
    finish(stream, e, "qi_rows_i8");
}

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
            float s = 0.f;
            for (int64_t c = lane; c < C; c += (int64_t) local) s += r[c];
            const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
            float v = 0.f;
            for (int64_t c = lane; c < C; c += (int64_t) local) {
                const float d = r[c] - mean;
                v += d * d;
            }
            const float inv = 1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) /
                                               (float) C + kEps);
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
                 const QiI8* i8) {
    // the DiTs' gated residual (dit_rows::gate_add_by), the gate the text's or the image's row of mod, through tanh
    MfOut in;
    in.C = C;
    if (i8 != nullptr) {
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
                       uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream, const QiI8* i8) {
    QkvIn in{q, k, v, i8 != nullptr, {}};
    if (i8 != nullptr)
        for (int j = 0; j < 3; ++j) in.s[j] = i8[j];
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
            const bool i8 = in.i8;
            const float sxt = i8 ? (which == 0 ? in.s[0].sx : which == 1 ? in.s[1].sx : in.s[2].sx)[t] : 0.f;
            const float* sw = i8 ? (which == 0 ? in.s[0].sw : which == 1 ? in.s[1].sw : in.s[2].sw) + c0 : nullptr;
            return [=](int64_t d) { return i8 ? (float) sycl::bit_cast<int32_t>(p[d]) * sxt * sw[d] : p[d]; };
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

void qi_text_qk_rope16(const float* q, const float* k, const float* v, const float* wq, const float* wk, int64_t T,
                       int64_t q_heads, int64_t kv_heads, int64_t head_dim, float theta, uint16_t* q16, uint16_t* k16,
                       uint16_t* v16, void* stream, const int32_t* pos3, int sec_h, int sec_w) {
    if (T <= 0) return;
    if (head_dim % 2 != 0 || head_dim > 256) throw std::invalid_argument("qi_text_qk_rope16: head_dim");
    struct Heads {
        const float *q, *k, *v;
        int64_t qh, kvh, hd;
        auto head(int which, int64_t t, int64_t h) const {
            const float* p = which == 0 ? q + (t * qh + h) * hd : (which == 1 ? k : v) + (t * kvh + h) * hd;
            return [p](int64_t d) { return p[d]; };
        }
    };
    kernels::qk::Shape sh{T, q_heads, kv_heads, head_dim};
    sh.wq = wq;
    sh.wk = wk;
    sh.eps = kEps;
    kernels::qk::Theta ang{theta, (float) head_dim};
    ang.pos3 = pos3;
    ang.T = T;
    ang.sec_h = sec_h;
    ang.sec_w = sec_w;
    finish(stream, kernels::qk::qk_norm_rope(queue_for(stream), sh, Heads{q, k, v, q_heads, kv_heads, head_dim}, ang,
                                             kernels::qk::Rows16{q16, k16, v16, q_heads, kv_heads, head_dim}),
           "qi_text_qk_rope16");
}

void qi_quantize_q8(const uint16_t* x16, int64_t rows, int64_t cols, void* xq, void* stream) {
    if (rows <= 0) return;
    auto* x8 = static_cast<int8_t*>(xq);
    auto* xs = reinterpret_cast<float*>(x8 + rows * cols);
    const kernels::q8act::Rows<false> out{x8, xs, nullptr, cols, cols / 32};
    finish(stream, kernels::q8act::quantize<8, kernels::q8act::Mul>(queue_for(stream), rows * (cols / 32), cols / 32,
                                                                       kernels::q8act::HalfRows{x16, cols}, out),
           "qi_quantize_q8");
}

void qi_rows_i8(const uint16_t* src, bool bf16, int64_t rows, int64_t K, int8_t* q, float* s, const float* rs,
                void* stream) {
    qi_rows::rows_i8(qi_rows::I8From16{src, bf16, K, rs}, rows, K, q, s, stream);
}

bool qi_rows_i8_blocks(int type, const void* blocks, int64_t rows, int64_t K, int8_t* q, float* s, void* stream) {
    const auto* b = static_cast<const uint8_t*>(blocks);
    switch (type) {
    case 12: qi_rows::rows_i8(qi_rows::I8FromBlocks<12>{b, K / 256 * 144}, rows, K, q, s, stream); return true;
    case 13: qi_rows::rows_i8(qi_rows::I8FromBlocks<13>{b, K / 256 * 176}, rows, K, q, s, stream); return true;
    case 14: qi_rows::rows_i8(qi_rows::I8FromBlocks<14>{b, K / 256 * 210}, rows, K, q, s, stream); return true;
    default: return false;
    }
}

void qi_swiglu_i8(const int32_t* gu, const QiI8& g, int64_t T, int64_t inter, int8_t* q, float* s, void* stream) {
    qi_rows::rows_i8(qi_rows::I8FromSwiglu{gu, g.sx, g.sw, inter}, T, inter, q, s, stream);
}

void qi_i8_scale(float* y, int64_t T, int64_t N, int64_t ldy, const float* sx, const float* sw, const float* bias,
                 const float* col, void* stream) {
    if (T <= 0 || N <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<2>((size_t) T, (size_t) N), [=](sycl::id<2> i) {
        float& v = y[(int64_t) i[0] * ldy + (int64_t) i[1]];
        // without a DoRA's factors (col) the expression the product had before them, so its rounding stays the same
        const int64_t n = (int64_t) i[1];
        const float s = (float) sycl::bit_cast<int32_t>(v) * sx[i[0]];
        v = col != nullptr ? s * sw[n] * col[n] + (bias != nullptr ? bias[n] : 0.f)
                           : s * sw[n] + (bias != nullptr ? bias[n] : 0.f);
    });
    finish(stream, e, "qi_i8_scale");
}

void qi_rows_scale(float* y, int64_t T, int64_t N, int64_t ldy, const float* col, void* stream) {
    if (T <= 0 || N <= 0 || col == nullptr) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<2>((size_t) T, (size_t) N), [=](sycl::id<2> i) {
        y[(int64_t) i[0] * ldy + (int64_t) i[1]] *= col[(int64_t) i[1]];
    });
    finish(stream, e, "qi_rows_scale");
}

void qi_gelu16(const float* x, int64_t n, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i];
        y16[i] = f16_from_f32(gelu_tanh(v));
    });
    finish(stream, e, "qi_gelu16");
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

void qi_axpy(float* x, const float* v, float a, int64_t n, void* stream) { kernels::add_inplace(x, v, n, stream, a); }

void qi_repeat_kv16(const uint16_t* src, int64_t T, int64_t q_heads, int64_t kv_heads, int64_t head_dim, uint16_t* out,
                    void* stream) {
    if (T <= 0) return;
    const int64_t group = q_heads / kv_heads, row = q_heads * head_dim, kvrow = kv_heads * head_dim;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * row)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / row, c = i - t * row, h = c / head_dim, d = c - h * head_dim;
        out[i] = src[t * kvrow + (h / group) * head_dim + d];
    });
    finish(stream, e, "qi_repeat_kv16");
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
// KT and K known (the VAEs' 3 x 3: its divisions by constants), else 0 and kt, k read at run time
template <int KT, int K>
sycl::event im2col_launch(sycl::queue& q, const uint16_t* in16, int64_t T, int64_t H, int64_t W, int64_t C, int kt_,
                          int k_, int s, int pad, int64_t OH, int64_t OW, int64_t p0, int64_t rows, uint16_t* col16,
                          int64_t ldx) {
    const int64_t C8 = C / 8, kt = KT > 0 ? KT : kt_, k = K > 0 ? K : k_, kk2 = k * k, taps = kt * kk2;
    return q.parallel_for(sycl::range<1>((size_t) (rows * taps * C8)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t rk = i / C8, r = rk / taps, tap = rk - r * taps, c8 = (i - rk * C8) * 8;
        const int64_t pq = p0 + r, ox = pq % OW, oyt = pq / OW, oy = oyt % OH, t = oyt / OH;
        const int64_t kk = tap % kk2, tt = t + (kt == 3 ? tap / kk2 - 1 : 0);
        const int64_t y = oy * s + kk / k - pad, x = ox * s + kk % k - pad;
        using V = sycl::vec<uint16_t, 8>;
        V v(0);
        if (tt >= 0 && tt < T && y >= 0 && y < H && x >= 0 && x < W)
            v = *reinterpret_cast<const V*>(in16 + ((tt * H + y) * W + x) * ldx + c8);
        *reinterpret_cast<V*>(col16 + rk * C + c8) = v;
    });
}
}  // namespace

void qi_im2col16(const uint16_t* in16, int64_t T, int64_t H, int64_t W, int64_t C, int kt, int k, int s, int pad,
                 int64_t OH, int64_t OW, int64_t p0, int64_t rows, uint16_t* col16, void* stream, int64_t ldx) {
    if (rows <= 0) return;
    if (ldx == 0) ldx = C;
    if (C % 8 != 0 || ldx % 8 != 0 || ldx < C || (kt != 1 && kt != 3))
        throw std::invalid_argument("qi_im2col16: C, ldx or kt");
    sycl::queue& q = queue_for(stream);
    const sycl::event e =
        k != 3    ? im2col_launch<0, 0>(q, in16, T, H, W, C, kt, k, s, pad, OH, OW, p0, rows, col16, ldx)
        : kt == 1 ? im2col_launch<1, 3>(q, in16, T, H, W, C, kt, k, s, pad, OH, OW, p0, rows, col16, ldx)
                  : im2col_launch<3, 3>(q, in16, T, H, W, C, kt, k, s, pad, OH, OW, p0, rows, col16, ldx);
    finish(stream, e, "qi_im2col16");
}

void qi_conv_store16(const float* y, int64_t T, int64_t O, int64_t ldy, const float* bias, QiAct act, float slope,
                     const uint16_t* res, uint16_t* dst, int64_t ld, int64_t off, void* stream, uint16_t* dst2,
                     int64_t from2) {
    if (T <= 0 || O <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * O)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t q = i / O, o = i - q * O;
        float v = y[q * ldy + o] + (bias != nullptr ? bias[o] : 0.f);
        if (act == QiAct::silu) v = v / (1.f + sycl::exp(-v));
        else if (act == QiAct::leaky) v = v > 0.f ? v : v * slope;
        if (res != nullptr) v += f32_from_f16(res[q * O + o]);
        const uint16_t h = f16_from_f32(v);
        if (dst != nullptr) dst[q * ld + off + o] = h;
        if (dst2 != nullptr && o >= from2) dst2[q * (O - from2) + o - from2] = h;
    });
    finish(stream, e, "qi_conv_store16");
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
// The GroupNorm(32) statistics (group_part and group_stats): a work-group a chunk of kGnChunk positions and
// one of the 32 groups, the chunk sums reduced over at most kGnMaxChunk of them a work-group.  Every image VAE's
// GroupNorm and HunyuanImage 3.0's entry and exit are the same reduction; only eps differs.
constexpr int64_t kGnGroups = 32;      // every GroupNorm here: nn.GroupNorm(32, C)
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

void group_part(GnIn x, const float* mean, int64_t P, int64_t C, bool sq, float* part, void* stream) {
    if (P <= 0) return;
    const int64_t cg = C / kGnGroups, B = (P + kGnChunk - 1) / kGnChunk;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (B * kGnGroups * 32)), sycl::range<1>(32)), [=](sycl::nd_item<1> it) {
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const auto blk = static_cast<int64_t>(it.get_group_linear_id());
            const int64_t b = blk / kGnGroups, g = blk - b * kGnGroups;
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

void group_stats(const float* part, int64_t B, int64_t P, int64_t CG, float eps, bool sq,
                               float* stats, float* mid, void* stream) {
    const int64_t G = (B + kGnMaxChunk - 1) / kGnMaxChunk;
    if (G > 1) {
        const auto e = queue_for(stream).parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) (G * kGnGroups * 32)), sycl::range<1>(32)),
            [=](sycl::nd_item<1> it) {
                const auto lane = static_cast<int64_t>(it.get_local_linear_id());
                const auto blk = static_cast<int64_t>(it.get_group_linear_id());
                const int64_t gi = blk / kGnGroups, g = blk - gi * kGnGroups;
                const int64_t b1 = std::min<int64_t>(B, (gi + 1) * kGnMaxChunk);
                float a = 0.f, comp = 0.f;
                for (int64_t b = gi * kGnMaxChunk + lane; b < b1; b += 32) {
                    const float y = part[b * kGnGroups + g] - comp, t = a + y;
                    comp = (t - a) - y;
                    a = t;
                }
                a = sycl::reduce_over_group(it.get_group(), a, sycl::plus<float>());
                if (lane == 0) mid[gi * kGnGroups + g] = a;
            });
        (void) e;
        queue_for(stream).wait();   // the last level reads every group's entry
    }
    // one work-item a group, serial Kahan over the (at most kGnMaxChunk) entries of its slice, or over the B chunk
    // sums themselves when they fit one reducer
    const int64_t MG = G > 1 ? G : B;
    const float* src = G > 1 ? mid : part;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) kGnGroups), [=](sycl::id<1> id) {
        const auto g = static_cast<int64_t>(id[0]);
        float a = 0.f, comp = 0.f;
        for (int64_t i = 0; i < MG; ++i) {
            const float y = src[i * kGnGroups + g] - comp, t = a + y;
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
// deviations, merged within each group (Chan's): part[(k kGnGroups + g) 3 + 0, 1, 2] = the count, the mean
// and the squared deviations of chunk k's group g.  (A work-group a chunk and group, reading its C / 32 channels of
// each pixel, and a second pass for the squares took MOSS-Image's VAE 64 ms of its 170 ms decode on the B70.)
int64_t gn_chunks(int64_t HW) { return (HW + kGnChunk - 1) / kGnChunk; }
int64_t gn_rows(int64_t C) { return sycl::max<int64_t>(1, 256 / C); }

void gn_partial(GnIn x, int64_t HW, int64_t C, float* part, void* s) {
    const int64_t cg = C / kGnGroups, nk = gn_chunks(HW), R = gn_rows(C), local = C * R;
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
            if (lane < kGnGroups) {
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
                float* o = part + (k * kGnGroups + lane) * 3;
                o[0] = N;
                o[1] = M;
                o[2] = Q;
            }
        });
    });
}

// stats[2 g] = group g's mean over every pixel, stats[2 g + 1] = 1 / its std, gn_partial's chunks merged (Chan's): a
// work-group a group, each lane a stride of the chunks, then the lanes in a tree
void gn_stats(const float* part, int64_t HW, int64_t C, float eps, float* mean, float* inv, void* s) {
    (void) C;
    constexpr int64_t local = 256;
    const int64_t nk = gn_chunks(HW);
    queue_for(s).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ln(sycl::range<1>((size_t) local), h), lm(sycl::range<1>((size_t) local), h),
            lq(sycl::range<1>((size_t) local), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (kGnGroups * local)), sycl::range<1>((size_t) local)),
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
                const float* e = part + (k * kGnGroups + g) * 3;
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

int64_t qi_group_norm_ws_floats(int64_t P) {
    const int64_t B = (P + kGnChunk - 1) / kGnChunk, G = (B + kGnMaxChunk - 1) / kGnMaxChunk;
    return 2 * kGnGroups + std::max((B + G) * kGnGroups, B * kGnGroups * 3);   // the two passes' sums, or the one's
}

namespace {

void group_norm16(GnIn x, int64_t P, int64_t C, const float* w, const float* b, float eps, bool silu, uint16_t* y16,
                  float* ws, void* stream, const float* scale, const float* shift, uint16_t* lo16) {
    if (P <= 0) return;
    if (C % kGnGroups != 0 || C % 8 != 0) throw std::invalid_argument("qi_group_norm16: C");
    float* mean = ws, *inv = ws + kGnGroups, *part = ws + 2 * kGnGroups;
    const int64_t B = (P + kGnChunk - 1) / kGnChunk, CG = C / kGnGroups;
    // One pass where a pixel's channels fit a work-group's lanes (the VAEs' up to 1,024: the two passes took
    // MOSS-Image's VAE 64 ms of its 170 ms decode on the B70), two (a mean, then the squares) past them (HunyuanImage
    // 3.0's entry and exit, 4,096 channels)
    static core::PerDevice<size_t> max_wg;
    sycl::queue& q = queue_for(stream);
    const size_t wg = max_wg.get(q.get_device(),
                                 [&q] { return q.get_device().get_info<sycl::info::device::max_work_group_size>(); });
    if ((size_t) (C * gn_rows(C)) <= wg) {
        gn_partial(x, P, C, part, stream);
        gn_stats(part, P, C, eps, mean, inv, stream);
    } else {
        float* mid = part + B * kGnGroups;
        group_part(x, nullptr, P, C, false, part, stream);
        group_stats(part, B, P, CG, eps, false, mean, mid, stream);
        group_part(x, mean, P, C, true, part, stream);
        group_stats(part, B, P, CG, eps, true, inv, mid, stream);
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
            float v = (xv[j] - mean[g]) * inv[g] * w[c] + b[c];
            if (scale != nullptr) v = v * (1.f + scale[c]) + shift[c];
            if (silu) v = v / (1.f + sycl::exp(-v));
            o[j] = f16_from_f32(v);
            lo[j] = f16_from_f32(v - f32_from_f16(o[j]));
        }
        *reinterpret_cast<sycl::vec<uint16_t, 8>*>(y16 + i) = o;
        if (lo16 != nullptr) *reinterpret_cast<sycl::vec<uint16_t, 8>*>(lo16 + i) = lo;
    });
    finish(stream, e, "qi_group_norm16");
}

}  // namespace

void qi_group_norm16(const float* x, int64_t P, int64_t C, const float* w, const float* b, float eps, bool silu,
                     uint16_t* y16, float* ws, void* stream, const float* scale, const float* shift, uint16_t* lo16) {
    group_norm16(GnIn{x, 0}, P, C, w, b, eps, silu, y16, ws, stream, scale, shift, lo16);
}

void qi_group_norm16(const uint16_t* x16, bool bf16, int64_t P, int64_t C, const float* w, const float* b, float eps,
                     bool silu, uint16_t* y16, float* ws, void* stream) {
    group_norm16(GnIn{x16, bf16 ? 2 : 1}, P, C, w, b, eps, silu, y16, ws, stream, nullptr, nullptr, nullptr);
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
            float s = 0.f, s2 = 0.f;
            for (int64_t c = l; c < C; c += (int64_t) L) {
                s += r[c];
                s2 += r[c] * r[c];
            }
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            s2 = sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>());
            const float mean = s / (float) C;
            const float inv = 1.f / sycl::sqrt(sycl::fmax(s2 / (float) C - mean * mean, 0.f) + eps);
            for (int64_t c = l; c < C; c += (int64_t) L) y16[t * C + c] = f16_from_f32((r[c] - mean) * inv * w[c] + b[c]);
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

}  // namespace strata::image
