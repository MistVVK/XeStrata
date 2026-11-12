// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/vision_cpu.cpp - see include/strata/kernels/cpu/vision_cpu.hpp.  Compiled for the baseline x86-64:
// the work goes out on the pool from here, the row loops are vision_cpu_avx2.cpp's, vision_cpu_avx512.cpp's or this
// file's own scalar copies, and the products bf16_gemm's.
#include "strata/kernels/cpu/vision_cpu.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cmath>

namespace strata::kernels::cpu {

// each instruction set's copy of the row loops (vision_cpu_avx2.cpp, vision_cpu_avx512.cpp, and the scalar copies
// below, made in this file)
#define STRATA_VISION_ROWS                                                                                            \
    void layer_norm_row(const float* x, int64_t C, const float* w, const float* b, float eps, float* y);              \
    void gelu_tanh_span(float* x, int64_t n);                                                                         \
    void add_span(float* x, const float* v, int64_t n);                                                               \
    float softmax_row(float* x, int64_t n);                                                                           \
    void qkv_rope_rows(const float* qkv, int64_t t0, int64_t t1, int64_t T, int64_t heads, int64_t hd,                \
                       const float* cos, const float* sin, float q_scale, float* q, float* k, float* vt);
namespace vision_detail::avx2 {
STRATA_VISION_ROWS
}
namespace vision_detail::avx512 {
STRATA_VISION_ROWS
}
namespace vision_detail::scalar {
STRATA_VISION_ROWS
}
#undef STRATA_VISION_ROWS

// The scalar copies of the row loops, written in this file because it is the one compiled for the baseline x86-64.
// They are vision_cpu_avx2.cpp's own loops for the elements its vectors leave over, so a CPU without AVX2 - and
// XESTRATA_BF16_GEMM_ISA=scalar, which stands for one - reads a picture the same way; only the order a whole row's
// sums round in differs.
namespace vision_detail::scalar {

namespace {
inline float gelu_tanh1(float v) {
    return 0.5f * v * (1.f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
}  // namespace

void layer_norm_row(const float* x, int64_t C, const float* w, const float* b, float eps, float* y) {
    float s = 0;
    for (int64_t c = 0; c < C; ++c) s += x[c];
    const float mean = s / (float) C;
    float v = 0;
    for (int64_t c = 0; c < C; ++c) v += (x[c] - mean) * (x[c] - mean);
    const float inv = 1.f / std::sqrt(v / (float) C + eps);
    for (int64_t c = 0; c < C; ++c) y[c] = (x[c] - mean) * inv * w[c] + b[c];
}

void gelu_tanh_span(float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) x[i] = gelu_tanh1(x[i]);
}

void add_span(float* x, const float* v, int64_t n) {
    for (int64_t i = 0; i < n; ++i) x[i] += v[i];
}

// x[i] = e^(x[i] - max), returns the sum
float softmax_row(float* x, int64_t n) {
    float m = -INFINITY;
    for (int64_t i = 0; i < n; ++i) m = std::fmax(m, x[i]);
    float s = 0;
    for (int64_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - m);
        s += x[i];
    }
    return s;
}

#include "vision_cpu_rows.inl"

}  // namespace vision_detail::scalar

namespace {

struct Rows {
    void (*layer_norm_row)(const float*, int64_t, const float*, const float*, float, float*);
    void (*gelu_tanh_span)(float*, int64_t);
    void (*add_span)(float*, const float*, int64_t);
    float (*softmax_row)(float*, int64_t);
    void (*qkv_rope_rows)(const float*, int64_t, int64_t, int64_t, int64_t, int64_t, const float*, const float*, float,
                          float*, float*, float*);
};

// The row loops as wide as the BF16 product's kernel: sixteen floats where it takes its AVX-512 one (so
// XESTRATA_BF16_GEMM_ISA caps both), eight where this CPU has AVX2, and the scalar copies above where it has neither
// or the product was capped to scalar - the rule resize_rows.cpp follows for the host resize's rows.
const Rows& rows() {
    static const Rows r = [] {
        namespace a2 = vision_detail::avx2;
        namespace a5 = vision_detail::avx512;
        namespace sc = vision_detail::scalar;
        const Bf16GemmIsa isa = bf16_gemm_isa();
        if (isa == Bf16GemmIsa::avx512)
            return Rows{a5::layer_norm_row, a5::gelu_tanh_span, a5::add_span, a5::softmax_row, a5::qkv_rope_rows};
        if (isa == Bf16GemmIsa::avx2 && cpu_avx2_ok())
            return Rows{a2::layer_norm_row, a2::gelu_tanh_span, a2::add_span, a2::softmax_row, a2::qkv_rope_rows};
        return Rows{sc::layer_norm_row, sc::gelu_tanh_span, sc::add_span, sc::softmax_row, sc::qkv_rope_rows};
    }();
    return r;
}

// [0, n) in pieces of `chunk`, fn(i0, i1) each, on the pool
template <class Fn> void spans(ExpertPool& pool, int64_t n, int64_t chunk, const Fn& fn) {
    if (n <= 0) return;
    const int64_t pieces = (n + chunk - 1) / chunk;
    pool.run((int) pieces, [&](int p) { fn(p * chunk, std::min(n, (p + 1) * chunk)); });
}

constexpr int64_t kSpan = 1 << 15;   // elements a piece of an elementwise pass: 128 KiB of FP32

}  // namespace

void vision_layer_norm(ExpertPool& pool, const float* x, int64_t n, int64_t C, const float* w, const float* b, float eps,
                       float* y) {
    const int64_t per = std::max<int64_t>(1, kSpan / std::max<int64_t>(1, C));
    spans(pool, n, per, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) rows().layer_norm_row(x + t * C, C, w, b, eps, y + t * C);
    });
}

void vision_gelu_tanh(ExpertPool& pool, float* x, int64_t n) {
    spans(pool, n, kSpan, [&](int64_t i0, int64_t i1) { rows().gelu_tanh_span(x + i0, i1 - i0); });
}

void vision_gelu_erf(ExpertPool& pool, float* x, int64_t n) {
    spans(pool, n, kSpan, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) x[i] = 0.5f * x[i] * (1.f + std::erf(x[i] * 0.70710678118654752f));
    });
}

void vision_add(ExpertPool& pool, float* x, const float* v, int64_t n) {
    spans(pool, n, kSpan, [&](int64_t i0, int64_t i1) { rows().add_span(x + i0, v + i0, i1 - i0); });
}

void vision_rope_table(const int32_t* yx, int64_t T, int64_t hd, float theta, std::vector<float>& cos,
                       std::vector<float>& sin) {
    const int64_t half = hd / 2, quarter = hd / 4;
    cos.resize((size_t) (T * half));
    sin.resize((size_t) (T * half));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t j = 0; j < half; ++j) {
            const bool row = j < quarter;
            const float pos = (float) yx[2 * t + (row ? 0 : 1)];
            const int64_t f = row ? j : j - quarter;
            const float a = pos * std::pow(theta, -(float) (2 * f) / (0.5f * (float) hd));
            cos[(size_t) (t * half + j)] = std::cos(a);
            sin[(size_t) (t * half + j)] = std::sin(a);
        }
}

void VisionAttention::run(ExpertPool& pool, const float* qkv, int64_t T, int64_t heads, int64_t hd,
                          const std::vector<float>& cos, const std::vector<float>& sin, float* out,
                          const I8GemmIsa* i8) {
    const int64_t C = heads * hd;
    const float q_scale = 1.f / std::sqrt((float) hd);
    const Bf16GemmIsa isa = bf16_gemm_isa();
    q_.resize((size_t) (heads * T * hd));
    k_.resize(q_.size());
    vt_.resize(q_.size());
    // the heads out of the rows, q and k turned: [heads][T][hd], the values transposed [heads][hd][T]
    spans(pool, T, 64, [&](int64_t t0, int64_t t1) {
        rows().qkv_rope_rows(qkv, t0, t1, T, heads, hd, cos.data(), sin.data(), q_scale, q_.data(), k_.data(),
                             vt_.data());
    });
    if (i8 != nullptr) {
        run_i8(pool, *i8, T, heads, hd, out);
        return;
    }
    // per head the keys ([T][hd]) and the transposed values ([hd][T]) in BF16 panels
    keys_.resize((size_t) heads);
    values_.resize((size_t) heads);
    auto pack = [](Bf16Matrix& m, const float* w, int n_in, int n_out) {
        m.n_in = n_in;
        m.n_out = n_out;
        std::vector<uint16_t> rows((size_t) n_out * (size_t) n_in);
        for (size_t e = 0; e < rows.size(); ++e) rows[e] = bf16_from_f32(w[e]);
        m.panel.resize(bf16_gemm_packed_bytes(n_out, n_in) / 2);
        bf16_gemm_pack(rows.data(), (size_t) n_in, n_in, n_out, m.panel.data(), 0, m.panels());
    };
    pool.run((int) (2 * heads), [&](int i) {
        const int64_t h = i / 2;
        if (i % 2 == 0) pack(keys_[(size_t) h], k_.data() + h * T * hd, (int) hd, (int) T);
        else pack(values_[(size_t) h], vt_.data() + h * hd * T, (int) T, (int) hd);
    });
    // a piece: one head's kRows queries - the scores, the softmax, the weighted values over the softmax's sum
    constexpr int64_t kRows = 64;
    const int64_t pieces_h = (T + kRows - 1) / kRows;
    pool.run((int) (heads * pieces_h), [&](int i) {
        thread_local std::vector<float> score, sum, acc;
        const int64_t h = i / pieces_h, t0 = (i % pieces_h) * kRows, t1 = std::min(T, t0 + kRows), n = t1 - t0;
        score.resize((size_t) (kRows * T));
        sum.resize((size_t) kRows);
        acc.resize((size_t) (kRows * hd));
        const Bf16Matrix& K = keys_[(size_t) h];
        const Bf16Matrix& V = values_[(size_t) h];
        bf16_gemm_rows(isa, K.panel.data(), K.n_in, K.n_out, q_.data() + (h * T + t0) * hd, (size_t) hd, score.data(),
                       (size_t) T, (int) n, 0, K.panels(), nullptr);
        for (int64_t r = 0; r < n; ++r) sum[(size_t) r] = rows().softmax_row(score.data() + r * T, T);
        bf16_gemm_rows(isa, V.panel.data(), V.n_in, V.n_out, score.data(), (size_t) T, acc.data(), (size_t) hd, (int) n,
                       0, V.panels(), nullptr);
        for (int64_t r = 0; r < n; ++r) {
            const float inv = 1.f / sum[(size_t) r];
            float* o = out + (t0 + r) * C + h * hd;
            for (int64_t d = 0; d < hd; ++d) o[d] = acc[(size_t) (r * hd + d)] * inv;
        }
    });
}

void VisionAttention::run_i8(ExpertPool& pool, I8GemmIsa isa, int64_t T, int64_t heads, int64_t hd, float* out) {
    const int64_t C = heads * hd;
    const int levels = i8_gemm_weight_levels(isa);
    // the keys, a scale a key; the values' rows in chunks of keys the product can sum, a scale per kVisionAttnBlock
    const int64_t chunks = (T + kI8MaxIn - 1) / kI8MaxIn;
    const int64_t kc = ((T + chunks - 1) / chunks + kVisionAttnBlock - 1) / kVisionAttnBlock * kVisionAttnBlock;
    keys8_.resize((size_t) heads);
    values8_.resize((size_t) (heads * chunks));
    auto pack = [&](I8Matrix& m, const float* w, size_t ld, int n_in, int n_out, int block) {
        m.n_in = n_in;
        m.n_out = n_out;
        m.block = block;
        m.levels = levels;
        m.panel.resize(i8_gemm_packed_bytes(n_out, n_in));
        m.scale.resize(i8_gemm_scale_count(n_out, n_in, block));
        m.corr.resize(m.scale.size());
        i8_gemm_pack_f32(w, ld, n_in, n_out, m.panel.data(), m.scale.data(), m.corr.data(), 0, m.panels(), block,
                         levels);
    };
    pool.run((int) (heads * (1 + chunks)), [&](int i) {
        const int64_t h = i / (1 + chunks), c = i % (1 + chunks);
        if (c == 0) {
            pack(keys8_[(size_t) h], k_.data() + h * T * hd, (size_t) hd, (int) hd, (int) T, 0);
        } else {
            const int64_t k0 = (c - 1) * kc, k1 = std::min(T, k0 + kc);
            pack(values8_[(size_t) (h * chunks + c - 1)], vt_.data() + h * hd * T + k0, (size_t) T, (int) (k1 - k0),
                 (int) hd, kVisionAttnBlock);
        }
    });
    constexpr int64_t kRows = 64;
    const int64_t pieces_h = (T + kRows - 1) / kRows;
    const size_t ldq_q = (size_t) i8_gemm_act_row_bytes((int) hd);
    const size_t ldq_p = (size_t) i8_gemm_act_row_bytes((int) kc);
    const int nb_p = i8_gemm_blocks((int) kc, kVisionAttnBlock);
    pool.run((int) (heads * pieces_h), [&](int i) {
        thread_local std::vector<float> score, sum, acc, part, xs;
        thread_local std::vector<uint8_t> xq;
        const int64_t h = i / pieces_h, t0 = (i % pieces_h) * kRows, t1 = std::min(T, t0 + kRows), n = t1 - t0;
        score.resize((size_t) (kRows * T));
        sum.resize((size_t) kRows);
        acc.assign((size_t) (kRows * hd), 0.f);
        part.resize((size_t) (kRows * hd));
        xq.resize(std::max(ldq_q, ldq_p) * (size_t) kRows);
        xs.resize((size_t) (kRows * std::max(nb_p, 1)));
        const I8Matrix& K = keys8_[(size_t) h];
        i8_gemm_quantize(isa, q_.data() + (h * T + t0) * hd, (size_t) hd, (int) n, (int) hd, xq.data(), ldq_q,
                         xs.data(), 0);
        i8_gemm_rows(isa, K.panel.data(), K.scale.data(), K.corr.data(), K.n_in, K.n_out, xq.data(), ldq_q, xs.data(),
                     score.data(), (size_t) T, (int) n, 0, K.panels(), nullptr, 0, K.levels);
        for (int64_t r = 0; r < n; ++r) sum[(size_t) r] = rows().softmax_row(score.data() + r * T, T);
        for (int64_t c = 0; c < chunks; ++c) {
            const I8Matrix& V = values8_[(size_t) (h * chunks + c)];
            float* dst = chunks == 1 ? acc.data() : part.data();
            i8_gemm_quantize(isa, score.data() + c * kc, (size_t) T, (int) n, V.n_in, xq.data(), ldq_p, xs.data(),
                             kVisionAttnBlock);
            i8_gemm_rows(isa, V.panel.data(), V.scale.data(), V.corr.data(), V.n_in, V.n_out, xq.data(), ldq_p,
                         xs.data(), dst, (size_t) hd, (int) n, 0, V.panels(), nullptr, kVisionAttnBlock, V.levels);
            if (chunks > 1)
                for (size_t e = 0; e < (size_t) (n * hd); ++e) acc[e] += part[e];
        }
        for (int64_t r = 0; r < n; ++r) {
            const float inv = 1.f / sum[(size_t) r];
            float* o = out + (t0 + r) * C + h * hd;
            for (int64_t d = 0; d < hd; ++d) o[d] = acc[(size_t) (r * hd + d)] * inv;
        }
    });
}

}  // namespace strata::kernels::cpu
