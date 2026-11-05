// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_kernels.cpp - see include/strata/image/qi_kernels.hpp.

#include "strata/image/qi_kernels.hpp"

#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/qk_norm_rope.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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

// qi_rows_i8's values: V of row r from column k, FP16 or BF16
struct I8From16 {
    const uint16_t* src;
    bool bf16;
    int64_t K;
    template<int V> void load(const sycl::sub_group&, int64_t r, int64_t k, float* v) const {
        const auto raw = *reinterpret_cast<const sycl::vec<uint16_t, V>*>(src + r * K + k);
        for (int j = 0; j < V; ++j) v[j] = bf16 ? f32_from_bf16(raw[j]) : f32_from_f16(raw[j]);
    }
};

// qi_rows_i8_blocks's: a row of GGUF blocks, each value as kernels::dequant_f16 writes it (rounded to FP16), so the
// int8 rows are those of qi_rows_i8 of its output.  RowsI8 gives a sub-group one block (256 values, 8 a lane): its
// lanes read the block's bytes 8 a lane in 16-bit loads side by side (a block is 2-byte aligned) and take the bytes
// each value needs from the lane that read them, not a byte at a time from all over the block.
template<int TYPE> struct I8FromBlocks {
    static constexpr int BYTES = TYPE == 12 ? 144 : TYPE == 13 ? 176 : 210;
    const uint8_t* b;
    int64_t row_bytes;
    // inlined: else its stores through v put the values on the stack (RTX 4070, 2026-11-05: 8 stores and 4 loads a
    // lane's 8 values)
    template<int V>
    __attribute__((always_inline)) void load(const sycl::sub_group& sg, int64_t r, int64_t k, float* v) const {
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
    const int32_t* gu;
    const float *sx, *sw;
    int64_t inter;
    template<int V> void load(const sycl::sub_group&, int64_t r, int64_t k, float* v) const {
        const int32_t* row = gu + r * 2 * inter;
        for (int j = 0; j < V; ++j) {
            const float g = (float) row[k + j] * sx[r] * sw[k + j];
            const float u = (float) row[inter + k + j] * sx[r] * sw[inter + k + j];
            v[j] = u * g / (1.f + sycl::exp(-g));
        }
    }
};

// qi_rows_i8 a work-group a row: a sub-group of 32 a group of 256 values (8 a lane, held in registers), up to MAXG
// groups each (a template argument: the registers of 4 held the RTX 4070's occupancy to a third where 1 was
// enough, 2026-11-05); the rotation's butterflies of strides 1 to 4 within a lane, 8 to 128 across lanes
template<class Src, int MAXG> struct RowsI8 {
    static constexpr int SG = 32, V = kQiRotate / SG;
    static_assert(V == 8, "a lane's values: two 32-bit words");
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    Src src;
    int64_t K, gpr, nsg;
    int8_t* q;
    float* s;
    const float* rs;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const auto r = static_cast<int64_t>(it.get_group_linear_id());
        const auto si = static_cast<int64_t>(sg.get_group_linear_id());
        const int lane = static_cast<int>(sg.get_local_linear_id());
        // every loop unrolled, the groups a sub-group skips by a condition: a loop left rolled (the break it had
        // here) puts v in local memory, whose traffic through the L2 was 35 times the kernel's output (RTX 4070,
        // Qwen-Image-2.1-Turbo Q6_K's weights, 2026-11-05: 96 M sectors written for 87 MB)
        float v[MAXG][V];
        float amax = 0.f;
#pragma unroll
        for (int c = 0; c < MAXG; ++c) {
            const int64_t g = si + c * nsg;
            if (g < gpr) {   // the whole sub-group
                src.template load<V>(sg, r, g * kQiRotate + (int64_t) lane * V, v[c]);
#pragma unroll
                for (int m = 1; m < V; m <<= 1)
#pragma unroll
                    for (int j = 0; j < V; ++j)
                        if ((j & m) == 0) {
                            const float a = v[c][j], b = v[c][j + m];
                            v[c][j] = a + b;
                            v[c][j + m] = a - b;
                        }
#pragma unroll
                for (int l = 1; l < SG; l <<= 1)
#pragma unroll
                    for (int j = 0; j < V; ++j) {
                        const float o = sycl::permute_group_by_xor(sg, v[c][j], l);
                        v[c][j] = (lane & l) != 0 ? o - v[c][j] : v[c][j] + o;
                    }
#pragma unroll
                for (int j = 0; j < V; ++j) {
                    v[c][j] *= 1.f / 16.f;
                    amax = sycl::fmax(amax, sycl::fabs(v[c][j]));
                }
            }
        }
        amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        const float d = amax / 127.f, id = amax > 0.f ? 1.f / d : 0.f;
#pragma unroll
        for (int c = 0; c < MAXG; ++c) {
            const int64_t g = si + c * nsg;
            if (g < gpr) {
                // packed into 32-bit words and stored as one: a sycl::vec<int8_t, 8> store is split into its bytes on
                // NVPTX
                u32x2 o = {0u, 0u};
#pragma unroll
                for (int j = 0; j < V; ++j)
                    o[j / 4] |= (uint32_t) (uint8_t) (int8_t) (int) sycl::round(v[c][j] * id) << (8 * (j % 4));
                *reinterpret_cast<u32x2*>(q + r * K + g * kQiRotate + (int64_t) lane * V) = o;
            }
        }
        if (si == 0 && lane == 0) s[r] = rs != nullptr ? d * rs[r] : d;
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<SG>}; }
};

template<class Src>
void rows_i8(const Src& src, int64_t rows, int64_t K, int8_t* q, float* s, const float* rs, void* stream) {
    if (rows <= 0) return;
    sycl::queue& qu = queue_for(stream);
    constexpr int64_t SG = RowsI8<Src, 1>::SG, MAXG = 4;
    const int64_t gpr = K / kQiRotate;
    const auto max_sg =
        static_cast<int64_t>(qu.get_device().get_info<sycl::info::device::max_work_group_size>()) / SG;
    const int64_t per = (gpr + max_sg - 1) / max_sg;
    if (K % kQiRotate != 0 || per > MAXG)
        throw std::runtime_error("image engine: a W8A8 product takes rows of a multiple of 256 values, up to " +
                                 std::to_string(max_sg * MAXG * kQiRotate));
    const int64_t nsg = (gpr + per - 1) / per;
    const sycl::nd_range<1> nd(sycl::range<1>((size_t) (rows * nsg * SG)), sycl::range<1>((size_t) (nsg * SG)));
    const auto e = per == 1 ? qu.parallel_for(nd, RowsI8<Src, 1>{src, K, gpr, nsg, q, s, rs})
                 : per == 2 ? qu.parallel_for(nd, RowsI8<Src, 2>{src, K, gpr, nsg, q, s, rs})
                            : qu.parallel_for(nd, RowsI8<Src, 4>{src, K, gpr, nsg, q, s, rs});
    finish(stream, e, "qi_rows_i8");
}

}  // namespace

void qi_ln_mod16(const float* x, const float* mod, int64_t n_text, int64_t T, int64_t C, uint16_t* y16, void* stream) {
    if (T <= 0) return;
    constexpr size_t local = 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) T * local), sycl::range<1>(local)), [=](sycl::nd_item<1> it) {
            const int64_t t = static_cast<int64_t>(it.get_group_linear_id()), lane = static_cast<int64_t>(it.get_local_linear_id());
            const float* r = x + t * C;
            float s = 0.f;
            for (int64_t c = lane; c < C; c += local) s += r[c];
            const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
            float v = 0.f;
            for (int64_t c = lane; c < C; c += local) {
                const float d = r[c] - mean;
                v += d * d;
            }
            const float inv = 1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) /
                                               (float) C + kEps);
            const float* m = mod + (t < n_text ? C : 0);
            for (int64_t c = lane; c < C; c += local) y16[t * C + c] = f16_from_f32((r[c] - mean) * inv * (1.f + m[c]));
        });
    finish(stream, e, "qi_ln_mod16");
}

void qi_gate_add(float* x, const float* h, const float* mod, int64_t n_text, int64_t T, int64_t C, void* stream,
                 const QiI8* i8) {
    if (T <= 0) return;
    const float* sx = i8 != nullptr ? i8->sx : nullptr;
    const float* sw = i8 != nullptr ? i8->sw : nullptr;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / C, c = i - t * C;
        const float v = sx != nullptr ? (float) sycl::bit_cast<int32_t>(h[i]) * sx[t] * sw[c] : h[i];
        x[i] += v * sycl::tanh(mod[(t < n_text ? C : 0) + c]);
    });
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
        float at(int which, int64_t t, int64_t h, int64_t d) const {
            const int64_t c = h * hd + d;
            return in.at(which, t, t * heads * hd + c, c);
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
        float at(int which, int64_t t, int64_t h, int64_t d) const {
            return which == 0 ? q[(t * qh + h) * hd + d] : (which == 1 ? k : v)[(t * kvh + h) * hd + d];
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
    const int64_t nb = cols / 32;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (rows * nb)), [=](sycl::id<1> i) {
        const int64_t r = (int64_t) i[0] / nb, b = (int64_t) i[0] % nb;
        const uint16_t* xr = x16 + r * cols + b * 32;
        float v[32], amax = 0.f;
        for (int j = 0; j < 32; ++j) { v[j] = f32_from_f16(xr[j]); amax = sycl::fmax(amax, sycl::fabs(v[j])); }
        const float d = amax / 127.f, id = amax > 0.f ? 1.f / d : 0.f;
        uint32_t p[8];
        for (int j = 0; j < 8; ++j) {
            uint32_t u = 0;
            for (int k = 0; k < 4; ++k)
                u |= (uint32_t) (uint8_t) (int8_t) (int) sycl::round(v[4 * j + k] * id) << (8 * k);
            p[j] = u;
        }
        auto* o = reinterpret_cast<sycl::vec<uint32_t, 4>*>(x8 + r * cols + b * 32);
        o[0] = sycl::vec<uint32_t, 4>{p[0], p[1], p[2], p[3]};
        o[1] = sycl::vec<uint32_t, 4>{p[4], p[5], p[6], p[7]};
        xs[r * nb + b] = d;
    });
    finish(stream, e, "qi_quantize_q8");
}

void qi_rows_i8(const uint16_t* src, bool bf16, int64_t rows, int64_t K, int8_t* q, float* s, const float* rs,
                void* stream) {
    rows_i8(I8From16{src, bf16, K}, rows, K, q, s, rs, stream);
}

bool qi_rows_i8_blocks(int type, const void* blocks, int64_t rows, int64_t K, int8_t* q, float* s, void* stream) {
    const auto* b = static_cast<const uint8_t*>(blocks);
    switch (type) {
    case 12: rows_i8(I8FromBlocks<12>{b, K / 256 * 144}, rows, K, q, s, nullptr, stream); return true;
    case 13: rows_i8(I8FromBlocks<13>{b, K / 256 * 176}, rows, K, q, s, nullptr, stream); return true;
    case 14: rows_i8(I8FromBlocks<14>{b, K / 256 * 210}, rows, K, q, s, nullptr, stream); return true;
    default: return false;
    }
}

void qi_swiglu_i8(const int32_t* gu, const QiI8& g, int64_t T, int64_t inter, int8_t* q, float* s, void* stream) {
    rows_i8(I8FromSwiglu{gu, g.sx, g.sw, inter}, T, inter, q, s, nullptr, stream);
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
        y16[i] = f16_from_f32(0.5f * v * (1.f + sycl::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v))));
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

void qi_axpy(float* x, const float* v, float a, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] += a * v[i]; });
    finish(stream, e, "qi_axpy");
}

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
                          int k_, int s, int pad, int64_t OH, int64_t OW, int64_t p0, int64_t rows, uint16_t* col16) {
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
            v = *reinterpret_cast<const V*>(in16 + ((tt * H + y) * W + x) * C + c8);
        *reinterpret_cast<V*>(col16 + rk * C + c8) = v;
    });
}
}  // namespace

void qi_im2col16(const uint16_t* in16, int64_t T, int64_t H, int64_t W, int64_t C, int kt, int k, int s, int pad,
                 int64_t OH, int64_t OW, int64_t p0, int64_t rows, uint16_t* col16, void* stream) {
    if (rows <= 0) return;
    if (C % 8 != 0 || (kt != 1 && kt != 3)) throw std::invalid_argument("qi_im2col16: C or kt");
    sycl::queue& q = queue_for(stream);
    const sycl::event e = k != 3 ? im2col_launch<0, 0>(q, in16, T, H, W, C, kt, k, s, pad, OH, OW, p0, rows, col16)
                          : kt == 1 ? im2col_launch<1, 3>(q, in16, T, H, W, C, kt, k, s, pad, OH, OW, p0, rows, col16)
                                    : im2col_launch<3, 3>(q, in16, T, H, W, C, kt, k, s, pad, OH, OW, p0, rows, col16);
    finish(stream, e, "qi_im2col16");
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
        y16[i] = f16_from_f32(0.5f * v * (1.f + sycl::erf(v * 0.70710678118654752f)));
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
        float at(int which, int64_t t, int64_t h, int64_t d) const { return qkv[t * 3 * C + which * C + h * hd + d]; }
    };
    struct Padded {
        uint16_t *q16, *k16, *v16;
        int64_t heads, hd;
        float q_scale;
        void put(int which, int64_t t, int64_t h, int64_t d, float v) const {
            uint16_t* dst = (which == 0 ? q16 : which == 1 ? k16 : v16) + (t * heads + h) * D;
            dst[d] = f16_from_f32(which == 0 ? v * q_scale : v);
            for (int64_t e = hd + d; e < D; e += hd) dst[e] = 0;
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
