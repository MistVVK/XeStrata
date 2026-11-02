// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/anima_dit.cpp - see include/strata/image/anima_dit.hpp.

#include "strata/image/anima_dit.hpp"

#include "dit_rows.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mapped_file.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace anima_dit_detail {

using namespace dit_rows;

constexpr int kTypeF16 = 1, kTypeQ8_0 = 8, kTypeW8 = -8;
constexpr float kEps = 1e-6f;
constexpr int64_t kCtxRows = 512;   // the context's rows at least: the adapter's output padded with zero rows
constexpr int64_t kKeyPad = 32;     // the keys of a library product's attention padded to a multiple of this

template<class T> T* alloc(std::vector<void*>& owned, int64_t n, bool zero = false) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) std::max<int64_t>(n, 1) * sizeof(T)))
        throw std::runtime_error(std::string("image engine (Anima DiT): GPU memory: ") + gpu::last_error());
    if (zero && !gpu::memset(p, 0, (size_t) std::max<int64_t>(n, 1) * sizeof(T)))
        throw std::runtime_error(std::string("image engine (Anima DiT): memset: ") + gpu::last_error());
    owned.push_back(p);
    return p;
}

void release(std::vector<void*>& owned, void* p) {
    if (p == nullptr) return;
    owned.erase(std::remove(owned.begin(), owned.end(), p), owned.end());
    gpu::free(p);
}

int64_t pad_keys(int64_t n) { return (n + kKeyPad - 1) / kKeyPad * kKeyPad; }

bool is_text_encoder_key(const std::string& k) {
    return k.rfind("lora_te", 0) == 0 || k.rfind("text_encoder", 0) == 0 || k.rfind("te.", 0) == 0;
}

std::string name_list(const std::vector<std::string>& names) {
    std::string out;
    const size_t n = std::min<size_t>(names.size(), 6);
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) out += ", ";
        out += names[i];
    }
    if (names.size() > n) out += " (and " + std::to_string(names.size() - n) + " more)";
    return out;
}

// a name with its dots as underscores and back: sd-scripts' keys reach the table with every underscore a dot
std::string dotted(std::string s) {
    std::replace(s.begin(), s.end(), '_', '.');
    return s;
}

// LayerNorm(x[t]) (no affine, eps 1e-6) * (1 + scale) + shift
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
        return {mean, 1.f / sycl::sqrt(reduce(v) / (float) C + kEps)};
    }
    float value(int64_t t, int64_t c, const Stats& st) const {
        return (x[t * C + c] - st.mean) * st.inv * (1.f + scale[c]) + shift[c];
    }
};

// RMSNorm(x[t]) * w (eps 1e-6), times the row's factor where given (the T5 ids' weights)
struct RmsW {
    const float *x, *w, *row;
    int64_t C;
    template<class R> Stats stats(int64_t t, int64_t lane, int64_t step, R reduce) const {
        const float* r = x + t * C;
        float s = 0.f;
        for (int64_t c = lane; c < C; c += step) s += r[c] * r[c];
        return {0.f, 1.f / sycl::sqrt(reduce(s) / (float) C + kEps)};
    }
    float value(int64_t t, int64_t c, const Stats& st) const {
        const float y = x[t * C + c] * st.inv * w[c];
        return row != nullptr ? y * row[t] : y;
    }
};

struct Plain {
    const float* x;
    int64_t C;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const { return x[t * C + c]; }
};

// GELU (erf form, PyTorch's nn.GELU()) of a product's sums
struct GeluErf {
    MfOut x;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const {
        const float v = x.at(t, c);
        return 0.5f * v * (1.f + sycl::erf(v * 0.7071067811865476f));
    }
    float bound(int64_t t, int64_t c) const { return sycl::fmax(x.at(t, c), 0.17f); }
};

// For each row and head: q and k RMS-normed over the head (wq, wk; eps 1e-6) and rotated in pairs (j, j + hd / 2) by
// cos[t][j], sin[t][j], v divided by its head's scale; FP16 out (the self-attention's, MossDit's form)
struct QkRope {
    MfOut q, k, v;
    const float *wq, *wk, *cos, *sin, *hs;
    int64_t T, heads, hd;
    uint16_t *q16, *k16, *v16;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t i = static_cast<int64_t>(it.get_group_linear_id()) * kRows + sg.get_group_linear_id();
        if (i >= T * heads) return;
        const int64_t t = i / heads, h = i - t * heads, c0 = h * hd, o = t * heads * hd + c0, half = hd / 2;
        const auto lane = static_cast<int64_t>(sg.get_local_linear_id());
        for (int which = 0; which < 2; ++which) {
            const MfOut& src = which == 0 ? q : k;
            const float* w = which == 0 ? wq : wk;
            uint16_t* dst = (which == 0 ? q16 : k16) + o;
            float ss = 0.f;
            for (int64_t j = lane; j < half; j += kSG) {
                const float a = src.at(t, c0 + j), b = src.at(t, c0 + half + j);
                ss += a * a + b * b;
            }
            const float r = 1.f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / (float) hd + kEps);
            for (int64_t j = lane; j < half; j += kSG) {
                const float a = src.at(t, c0 + j) * r * w[j], b = src.at(t, c0 + half + j) * r * w[half + j];
                const float c = cos[t * half + j], sn = sin[t * half + j];
                dst[j] = f16_from_f32(a * c - b * sn);
                dst[half + j] = f16_from_f32(b * c + a * sn);
            }
        }
        const float inv = 1.f / hs[h];
        for (int64_t d = lane; d < hd; d += kSG) v16[o + d] = f16_from_f32(v.at(t, c0 + d) * inv);
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<kSG>}; }
};

void qk_rope16(const MfOut& q, const MfOut& k, const MfOut& v, const float* wq, const float* wk, const float* cos,
               const float* sin, const float* hs, int64_t T, int64_t heads, int64_t hd, uint16_t* q16, uint16_t* k16,
               uint16_t* v16, void* s) {
    if (T <= 0) return;
    sycl::queue& qu = q_of(s);
    if (has_sg32(qu)) {
        qu.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((T * heads + kRows - 1) / kRows * kRows * kSG)),
                                          sycl::range<1>((size_t) (kRows * kSG))),
                        QkRope{q, k, v, wq, wk, cos, sin, hs, T, heads, hd, q16, k16, v16});
        return;
    }
    const int64_t C = heads * hd, half = hd / 2;
    qu.parallel_for(sycl::range<1>((size_t) (T * heads)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / heads, h = i - t * heads, c0 = h * hd, o = t * C + c0;
        for (int which = 0; which < 2; ++which) {
            const MfOut& src = which == 0 ? q : k;
            const float* w = which == 0 ? wq : wk;
            uint16_t* dst = (which == 0 ? q16 : k16) + o;
            float ss = 0.f;
            for (int64_t d = 0; d < hd; ++d) ss += src.at(t, c0 + d) * src.at(t, c0 + d);
            const float r = 1.f / sycl::sqrt(ss / (float) hd + kEps);
            for (int64_t j = 0; j < half; ++j) {
                const float a = src.at(t, c0 + j) * r * w[j], b = src.at(t, c0 + half + j) * r * w[half + j];
                const float c = cos[t * half + j], sn = sin[t * half + j];
                dst[j] = f16_from_f32(a * c - b * sn);
                dst[half + j] = f16_from_f32(b * c + a * sn);
            }
        }
        const float inv = 1.f / hs[h];
        for (int64_t d = 0; d < hd; ++d) v16[o + d] = f16_from_f32(v.at(t, c0 + d) * inv);
    });
}

// A head-major copy for the library's products: row t's head h RMS-normed over the head (w; null: not normed) and
// rotated in pairs (j, j + hd / 2) by cos[t][j], sin[t][j] (null: not rotated), into out [heads][Tp][hd] (rows past T
// left as they are: zero), or with `rows` into out [T][heads hd] (the fused attention's layout).  A sub-group of 32 a row's head, its values read side by side (QkRope's form); a work-item a
// row's head without sub-groups of 32.
struct Heads16 {
    MfOut x;
    int64_t T, Tp, heads, hd;
    const float *w, *cos, *sin;
    uint16_t* out;
    bool rows;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t i = static_cast<int64_t>(it.get_group_linear_id()) * kRows + sg.get_group_linear_id();
        if (i >= T * heads) return;
        const int64_t t = i / heads, h = i - t * heads, c0 = h * hd, half = hd / 2;
        const auto lane = static_cast<int64_t>(sg.get_local_linear_id());
        uint16_t* dst = out + (rows ? t * heads + h : h * Tp + t) * hd;
        float r = 1.f;
        if (w != nullptr) {
            float ss = 0.f;
            for (int64_t j = lane; j < hd; j += kSG) ss += x.at(t, c0 + j) * x.at(t, c0 + j);
            r = 1.f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / (float) hd + kEps);
        }
        for (int64_t j = lane; j < half; j += kSG) {
            float a = x.at(t, c0 + j) * r, b = x.at(t, c0 + half + j) * r;
            if (w != nullptr) {
                a *= w[j];
                b *= w[half + j];
            }
            if (cos != nullptr) {
                const float c = cos[t * half + j], sn = sin[t * half + j], a2 = a * c - b * sn;
                b = b * c + a * sn;
                a = a2;
            }
            dst[j] = f16_from_f32(sycl::clamp(a, -65504.f, 65504.f));
            dst[half + j] = f16_from_f32(sycl::clamp(b, -65504.f, 65504.f));
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<kSG>}; }
};

void heads16(const MfOut& x, int64_t T, int64_t Tp, int64_t heads, int64_t hd, const float* w, const float* cos,
             const float* sin, uint16_t* out, void* s, bool rows = false) {
    if (T <= 0) return;
    sycl::queue& q = q_of(s);
    if (has_sg32(q)) {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((T * heads + kRows - 1) / kRows * kRows * kSG)),
                                         sycl::range<1>((size_t) (kRows * kSG))),
                       Heads16{x, T, Tp, heads, hd, w, cos, sin, out, rows});
        return;
    }
    q.parallel_for(sycl::range<1>((size_t) (T * heads)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / heads, h = i - t * heads, c0 = h * hd, half = hd / 2;
        uint16_t* dst = out + (rows ? t * heads + h : h * Tp + t) * hd;
        float r = 1.f;
        if (w != nullptr) {
            float ss = 0.f;
            for (int64_t d = 0; d < hd; ++d) ss += x.at(t, c0 + d) * x.at(t, c0 + d);
            r = 1.f / sycl::sqrt(ss / (float) hd + kEps);
        }
        for (int64_t j = 0; j < half; ++j) {
            float a = x.at(t, c0 + j) * r, b = x.at(t, c0 + half + j) * r;
            if (w != nullptr) {
                a *= w[j];
                b *= w[half + j];
            }
            if (cos != nullptr) {
                const float c = cos[t * half + j], sn = sin[t * half + j], a2 = a * c - b * sn;
                b = b * c + a * sn;
                a = a2;
            }
            dst[j] = f16_from_f32(sycl::clamp(a, -65504.f, 65504.f));
            dst[half + j] = f16_from_f32(sycl::clamp(b, -65504.f, 65504.f));
        }
    });
}

// out [T][C] = v's rows in FP16 (the fused attention's values)
void values16(const MfOut& v, int64_t T, int64_t C, uint16_t* out, void* s) {
    if (T <= 0) return;
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        out[i] = f16_from_f32(sycl::clamp(v.at(i / C, i % C), -65504.f, 65504.f));
    });
}

// out [heads][hd][Tp] = v's row t, head h, transposed (the weighted values' product reads V^T's rows)
void values_t16(const MfOut& v, int64_t T, int64_t Tp, int64_t heads, int64_t hd, uint16_t* out, void* s) {
    if (T <= 0) return;
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * heads * hd)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / (heads * hd), c = i - t * heads * hd, h = c / hd, d = c - h * hd;
        out[(h * hd + d) * Tp + t] = f16_from_f32(sycl::clamp(v.at(t, c), -65504.f, 65504.f));
    });
}

// p16[r] = softmax(scale * s[r]) over the first n of its ld columns, the rest zero: a sub-group of 32 a row (kRows
// rows a work-group), or without sub-groups of 32 a work-group of 256 a row
struct Softmax16 {
    const float* sc;
    int64_t rows, n, ld;
    float scale;
    uint16_t* p16;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t r = static_cast<int64_t>(it.get_group_linear_id()) * kRows + sg.get_group_linear_id();
        if (r >= rows) return;
        const auto lane = static_cast<int64_t>(sg.get_local_linear_id());
        const float* row = sc + r * ld;
        float m = -INFINITY;
        for (int64_t j = lane; j < n; j += kSG) m = sycl::fmax(m, row[j]);
        m = sycl::reduce_over_group(sg, m, sycl::maximum<float>()) * scale;
        float sum = 0.f;
        for (int64_t j = lane; j < n; j += kSG) sum += sycl::exp(row[j] * scale - m);
        const float inv = 1.f / sycl::reduce_over_group(sg, sum, sycl::plus<float>());
        uint16_t* out = p16 + r * ld;
        for (int64_t j = lane; j < ld; j += kSG)
            out[j] = j < n ? f16_from_f32(sycl::exp(row[j] * scale - m) * inv) : (uint16_t) 0;
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<kSG>}; }
};

void softmax16(const float* sc, int64_t rows, int64_t n, int64_t ld, float scale, uint16_t* p16, void* s) {
    sycl::queue& q = q_of(s);
    if (has_sg32(q)) {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((rows + kRows - 1) / kRows * kRows * kSG)),
                                         sycl::range<1>((size_t) (kRows * kSG))),
                       Softmax16{sc, rows, n, ld, scale, p16});
        return;
    }
    constexpr size_t local = 256;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * local), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> it) {
        const auto r = static_cast<int64_t>(it.get_group_linear_id());
        const auto lane = static_cast<int64_t>(it.get_local_linear_id());
        const float* row = sc + r * ld;
        float m = -INFINITY;
        for (int64_t j = lane; j < n; j += local) m = sycl::fmax(m, row[j] * scale);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float sum = 0.f;
        for (int64_t j = lane; j < n; j += local) sum += sycl::exp(row[j] * scale - m);
        const float inv = 1.f / sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        for (int64_t j = lane; j < ld; j += local)
            p16[r * ld + j] = j < n ? f16_from_f32(sycl::exp(row[j] * scale - m) * inv) : (uint16_t) 0;
    });
}

// The adapter's FP16 rows of a row-wise op without a row scale (its values, normed or a GELU's, stay well inside
// FP16; saturated at its largest), so that its products need no scale after them: a work-group of 256 a row, as its
// few rows (a prompt's ids) left Rows16's sub-group a row with too little to do (78 us for the GELU of 84 rows of 4096
// on the RX 9060 XT).
template<class Op> void rows16_plain(const Op& op, int64_t T, int64_t C, uint16_t* y16, void* s) {
    if (T <= 0) return;
    constexpr size_t kLocal = 256;
    q_of(s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * kLocal), sycl::range<1>(kLocal)),
                         [=](sycl::nd_item<1> it) {
        const auto t = static_cast<int64_t>(it.get_group_linear_id());
        const auto lane = static_cast<int64_t>(it.get_local_linear_id());
        const auto sum = [&](float v) { return sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()); };
        const auto st = op.stats(t, lane, (int64_t) kLocal, sum);
        for (int64_t c = lane; c < C; c += (int64_t) kLocal)
            y16[t * C + c] = f16_from_f32(sycl::clamp(op.value(t, c, st), -65504.f, 65504.f));
    });
}

// The adapter's attention of a few rows (the ids over the ids or the states) for one or two contexts in a row,
// every head in one kernel: a work-group a query row's head, its lanes over the keys for the scores (kept in local
// memory), then over the head's dimensions for the weighted values.  The rows before q0 are the first context's and
// see the keys before k0, the rest the keys from k0.  q [heads][Tq][hd] and k [heads][Tk][hd] as heads16 writes them,
// v the values' product [Tk][heads hd], out [Tq][heads hd] FP32.  (A work-item a row, the softmax online in
// registers, took 0.8 ms on the RX 9060 XT for 31 rows over 31 keys: its 2 HD values spilled.)
template<int HD> void small_attention(const uint16_t* q, const uint16_t* k, const MfOut v, int64_t Tq, int64_t q0,
                                      int64_t Tk, int64_t k0, int64_t heads, float scale, float* out, void* s) {
    constexpr int kLocal = 64;
    const int64_t most = std::max(k0, Tk - k0);
    q_of(s).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(HD), h), ps(sycl::range<1>((size_t) most), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (heads * Tq * kLocal)), sycl::range<1>(kLocal)),
                       [=](sycl::nd_item<1> it) {
            const auto g = static_cast<int64_t>(it.get_group_linear_id());
            const auto lane = static_cast<int>(it.get_local_linear_id());
            const int64_t hh = g / Tq, t = g - hh * Tq;
            const int64_t ks = t < q0 ? 0 : k0, n = t < q0 ? k0 : Tk - k0;
            for (int d = lane; d < HD; d += kLocal) qs[d] = kernels::f32_from_f16(q[(hh * Tq + t) * HD + d]) * scale;
            sycl::group_barrier(it.get_group());
            float m = -INFINITY;
            for (int64_t j = lane; j < n; j += kLocal) {
                // a key's row in 16-byte pieces
                const auto* kr = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(k + (hh * Tk + ks + j) * HD);
                float sc = 0.f;
                for (int c = 0; c < HD / 8; ++c) {
                    const sycl::vec<uint32_t, 4> p = kr[c];
                    for (int e = 0; e < 4; ++e) {
                        const uint32_t u = p[e];
                        sc += qs[c * 8 + 2 * e] * kernels::f32_from_f16((uint16_t) (u & 0xffffu)) +
                              qs[c * 8 + 2 * e + 1] * kernels::f32_from_f16((uint16_t) (u >> 16));
                    }
                }
                ps[j] = sc;
                m = sycl::fmax(m, sc);
            }
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            float sum = 0.f;
            for (int64_t j = lane; j < n; j += kLocal) {
                const float e = sycl::exp(ps[j] - m);
                ps[j] = e;
                sum += e;
            }
            const float inv = 1.f / sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            sycl::group_barrier(it.get_group());
            for (int d = lane; d < HD; d += kLocal) {
                float acc = 0.f;
                for (int64_t j = 0; j < n; ++j) acc += ps[j] * v.at(ks + j, hh * HD + d);
                out[t * heads * HD + hh * HD + d] = acc * inv;
            }
        });
    });
}

// the final layer's [hw][64] (pixel (p1, p2) of channel c at (p1 * 2 + p2) * L + c) in the tokens' order (c * 4 +
// p1 * 2 + p2)
void to_tokens(const float* in, int64_t HW, int64_t L, float* out, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (HW * 4 * L)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / (4 * L), j = i - t * 4 * L, p = j / L, c = j - p * L;
        out[t * 4 * L + c * 4 + p] = in[i];
    });
}

// x[t] += y[t] (the adapter's residuals)
void add_rows(float* x, const float* y, int64_t n, void* s) { qi_axpy(x, y, 1.0f, n, s); }

// p's n bf16 values rewritten in place as FP16, saturated at FP16's largest
void bf16_to_f16_inplace(uint16_t* p, int64_t n, sycl::queue& q) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = kernels::f32_from_bf16(p[id[0]]);
        p[id[0]] = f16_from_f32(sycl::clamp(v, -65504.f, 65504.f));
    });
}

uint16_t bf16_of(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);   // NaN stays NaN
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// an F8 E4M3 byte (ComfyUI's scaled fp8: no inf, 0x7f / 0xff NaN)
float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? (float) m / 8.f * std::ldexp(1.f, -6) : (1.f + (float) m / 8.f) * std::ldexp(1.f, e - 7);
    if (e == 15 && m == 7) v = NAN;
    return s ? -v : v;
}

// The RoPE angles' cosine and sine of each run's rows at positions 0 .. its rows - 1, half pairs each with frequency
// 10000^(-2j / (2 half)), rounded to bf16 as the adapter's RotaryEmbedding casts them to its activations' type
void rope_1d(const std::vector<int64_t>& rows, int64_t half, std::vector<float>& c, std::vector<float>& s) {
    int64_t all = 0;
    for (const int64_t r : rows) all += r;
    c.assign((size_t) (all * half), 0.f);
    s.assign(c.size(), 0.f);
    for (int64_t j = 0; j < half; ++j) {
        const float f = 1.f / std::pow(10000.f, (float) (2 * j) / (float) (2 * half));
        int64_t at = 0;
        for (const int64_t n : rows)
            for (int64_t r = 0; r < n; ++r, ++at) {
                const float ph = (float) r * f;
                c[(size_t) (at * half + j)] = round_bf16(std::cos(ph));
                s[(size_t) (at * half + j)] = round_bf16(std::sin(ph));
            }
    }
}

}  // namespace anima_dit_detail

using namespace anima_dit_detail;

// the DiT's file: its tensors by their names without the prefix, BF16 in place or converted
struct AnimaDit::File {
    SafeTensors st;
    MappedFile map;
    std::string pre;
    explicit File(const std::string& path) : st(path), map(path) {
        for (const char* p : {"", "net.", "model.diffusion_model.", "diffusion_model."})
            if (st.find(std::string(p) + "llm_adapter.embed.weight") != nullptr) {
                pre = p;
                return;
            }
        throw std::runtime_error("image engine (Anima DiT): " + path + " has no llm_adapter.embed.weight");
    }
    const SafeTensor* find(const std::string& name) const { return st.find(pre + name); }
    const SafeTensor& get(const std::string& name) const {
        const SafeTensor* t = find(name);
        if (t == nullptr) throw std::runtime_error("image engine (Anima DiT): no " + name);
        return *t;
    }
    const uint8_t* raw(const SafeTensor& t) const { return map.data() + st.data_base() + t.begin; }
    // a tensor's values as bf16: the mapping itself for a BF16 one, else converted into tmp
    const uint16_t* bf16(const std::string& name, std::vector<uint16_t>& tmp) {
        const SafeTensor& t = get(name);
        if (t.dtype == "BF16") return reinterpret_cast<const uint16_t*>(raw(t));
        const std::vector<float> v = f32(name);
        tmp.resize(v.size());
        for (size_t i = 0; i < v.size(); ++i) tmp[i] = bf16_of(v[i]);
        return tmp.data();
    }
    std::vector<float> f32(const std::string& name) {
        const SafeTensor& t = get(name);
        if (t.dtype != "F8_E4M3") return st.f32(pre + name);
        // ComfyUI's scaled fp8: the weight times its scale_weight (one value), where the file has one
        float scale = 1.f;
        const std::string base = name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0
                                     ? name.substr(0, name.size() - 7) : name;
        if (find(base + ".scale_weight") != nullptr) scale = st.f32(pre + base + ".scale_weight").at(0);
        const uint8_t* p = raw(t);
        std::vector<float> out((size_t) t.elements());
        for (size_t i = 0; i < out.size(); ++i) out[i] = e4m3(p[i]) * scale;
        return out;
    }
    void drop_cache(const std::string& name) const {
        if (const SafeTensor* t = find(name); t != nullptr) map.drop_cache(st.data_base() + t->begin, t->end - t->begin);
    }
};

bool anima_is_dit(const std::string& path) {
    if (path.size() < 12 || path.compare(path.size() - 12, 12, ".safetensors") != 0) return false;
    try {
        SafeTensors st(path);
        for (const char* p : {"", "net.", "model.diffusion_model.", "diffusion_model."})
            if (st.find(std::string(p) + "llm_adapter.embed.weight") != nullptr &&
                st.find(std::string(p) + "x_embedder.proj.1.weight") != nullptr)
                return true;
    } catch (const std::exception&) {
        return false;   // not a safetensors file this engine reads: not Anima's
    }
    return false;
}

AnimaDit::AnimaDit() = default;

AnimaDit::~AnimaDit() {
    blocks_stream_.drop();
    for (void* p : allocs_) gpu::free(p);
    for (void* p : text_allocs_) gpu::free(p);
}

bool AnimaDit::open(const std::string& path, void* stream, std::string& err, bool dry, bool q8,
                    std::vector<std::vector<uint8_t>>* reuse) {
    try {
        s_ = stream;
        dry_ = dry;
        norm_cache_.clear();
        file_ = std::make_unique<File>(path);
        File& f = *file_;
        // a file whose blocks are not bf16 has no bytes the GPU could rewrite in place: 8 bits
        q8_ = q8 || f.get("blocks.0.self_attn.q_proj.weight").dtype != "BF16";
        w8_ = q8_ && std::getenv("STRATA_NO_W8A8") == nullptr && prefill::Gemm::i8_usable(stream);
        // the patch embedder: its first 64 columns (the 17th channel, the padding mask, is zero for an image)
        lat_ = f.get("final_layer.linear.weight").shape[0] / 4;
        xemb_ = dense("x_embedder.proj.1", false, nullptr, allocs_, weight_bytes_, 4 * lat_);
        C_ = xemb_.n;
        t1_ = dense("t_embedder.1.linear_1", false, nullptr, allocs_, weight_bytes_);
        t2_ = dense("t_embedder.1.linear_2", false, nullptr, allocs_, weight_bytes_);
        flin_ = dense("final_layer.linear", false, nullptr, allocs_, weight_bytes_);
        fm1_ = dense("final_layer.adaln_modulation.1", false, nullptr, allocs_, weight_bytes_);
        fm2_ = dense("final_layer.adaln_modulation.2", false, nullptr, allocs_, weight_bytes_);
        tnorm_host_ = f.f32("t_embedding_norm.weight");
        hd_ = f.get("blocks.0.self_attn.q_norm.weight").shape[0];
        heads_ = C_ / hd_;
        mod_dim_ = fm1_.n;
        if (C_ % hd_ != 0 || hd_ % 8 != 0 || t2_.n != 3 * C_ || flin_.n != 4 * lat_ || fm2_.n != 2 * C_)
            throw std::runtime_error("image engine (Anima DiT): an unexpected shape");
        load_text();
        for (int i = 0;; ++i) {
            const std::string p = "blocks." + std::to_string(i) + ".";
            if (f.find(p + "self_attn.q_proj.weight") == nullptr) break;
            Block b;
            const std::vector<std::pair<W*, std::string>> items = {
                {&b.sq, "self_attn.q_proj"}, {&b.sk, "self_attn.k_proj"}, {&b.sv, "self_attn.v_proj"},
                {&b.so, "self_attn.output_proj"}, {&b.cq, "cross_attn.q_proj"}, {&b.ck, "cross_attn.k_proj"},
                {&b.cv, "cross_attn.v_proj"}, {&b.co, "cross_attn.output_proj"}, {&b.l1, "mlp.layer1"},
                {&b.l2, "mlp.layer2"}, {&b.msa1, "adaln_modulation_self_attn.1"},
                {&b.msa2, "adaln_modulation_self_attn.2"}, {&b.mca1, "adaln_modulation_cross_attn.1"},
                {&b.mca2, "adaln_modulation_cross_attn.2"}, {&b.mml1, "adaln_modulation_mlp.1"},
                {&b.mml2, "adaln_modulation_mlp.2"}};
            const std::vector<std::pair<uint64_t*, std::string>> norms = {
                {&b.sqn, "self_attn.q_norm.weight"}, {&b.skn, "self_attn.k_norm.weight"},
                {&b.cqn, "cross_attn.q_norm.weight"}, {&b.ckn, "cross_attn.k_norm.weight"}};
            uint64_t qbytes = 0, vbytes = 0;
            for (const auto& [w, name] : items) {
                const SafeTensor& t = f.get(p + name + ".weight");
                w->n = t.shape[0];
                w->k = t.shape[1];
                if (w->k % 32 != 0) throw std::runtime_error("image engine (Anima DiT): columns not 32s");
                const bool w8 = w8_ && w->k % kQiRotate == 0;
                w->type = w8 ? kTypeW8 : q8_ ? kTypeQ8_0 : kTypeF16;
                w->row_bytes = w8 ? (uint64_t) w->k : q8_ ? (uint64_t) (w->k / 32 * 34) : (uint64_t) w->k * 2;
                if (q8_) qbytes += w->row_bytes * (uint64_t) w->n + (w8 ? (uint64_t) w->n * 4 : 0);
                if (w->type == kTypeQ8_0) scratch_ = std::max(scratch_, w->n * w->k);
            }
            for (const auto& [off, name] : norms) vbytes += (uint64_t) f.get(p + name).elements() * 4;
            std::vector<LayerStream::Span> spans;
            std::vector<uint8_t> host;
            const bool reused = !dry_ && q8_ && reuse != nullptr && (size_t) i < reuse->size() &&
                                (*reuse)[(size_t) i].size() == qbytes + vbytes;
            if (reused) host = std::move((*reuse)[(size_t) i]);
            else if (!dry_) host.resize((size_t) (qbytes + vbytes));
            uint64_t at = 0;
            std::vector<size_t> wspan, nspan;
            for (const auto& [w, name] : items) {
                const uint64_t bytes = w->row_bytes * (uint64_t) w->n + (w->type == kTypeW8 ? (uint64_t) w->n * 4 : 0);
                if (q8_) {
                    if (!dry_ && !reused) {
                        std::vector<uint16_t> tmp;
                        quantize_matrix(f.bf16(p + name + ".weight", tmp), w->n, w->k, host.data() + at,
                                        w->type == kTypeW8);
                        f.drop_cache(p + name + ".weight");
                    }
                    spans.push_back({host.data() + at, bytes, 0});
                    at += bytes;
                } else {
                    spans.push_back({f.raw(f.get(p + name + ".weight")), bytes, 0});
                }
                wspan.push_back(spans.size() - 1);
            }
            for (const auto& [off, name] : norms) {
                const int64_t n = f.get(p + name).elements();
                if (!dry_ && !reused) {
                    const auto v = f.f32(p + name);
                    std::memcpy(host.data() + at, v.data(), v.size() * 4);
                }
                spans.push_back({host.data() + at, (uint64_t) n * 4, 0});
                at += (uint64_t) n * 4;
                nspan.push_back(spans.size() - 1);
            }
            const int layer = blocks_stream_.add(spans);
            for (size_t j = 0; j < items.size(); ++j)
                items[j].first->d = reinterpret_cast<const void*>(blocks_stream_.offset(layer, (int) wspan[j]));
            for (size_t j = 0; j < norms.size(); ++j) *norms[j].first = blocks_stream_.offset(layer, (int) nspan[j]);
            host_.push_back(std::move(host));
            blocks_.push_back(b);
        }
        if (blocks_.empty()) throw std::runtime_error("image engine (Anima DiT): no blocks");
        F_ = blocks_[0].l1.n;
        src_dim_ = blocks_[0].ck.k;
        build_lora_targets();
        install_prepare();
        for (Ctx& c : ctx_) {
            c.kv.assign(blocks_.size(), nullptr);
            c.kv_lp.assign(blocks_.size(), 0);
            c.kv_ok.assign(blocks_.size(), 0);
        }
        if (dry_) return true;
        if (!gemm_.init(s_, scratch_, err)) return false;
        blocks_stream_.configure(blocks_stream_.total(), s_);
        e16_ = alloc<uint16_t>(allocs_, C_ + 4 * lat_);
        m16_ = alloc<uint16_t>(allocs_, std::max(mod_dim_, C_));
        alora_ = alloc<float>(allocs_, 3 * C_);
        mh_ = alloc<float>(allocs_, std::max(mod_dim_, C_));
        mod_ = alloc<float>(allocs_, 3 * C_);
        fmod_ = alloc<float>(allocs_, 2 * C_);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// a part outside the blocks as FP16 (its first `cols` columns when given), its bias FP32
AnimaDit::W AnimaDit::dense(const std::string& name, bool biased, const float** bias, std::vector<void*>& owned,
                            uint64_t& bytes, int64_t cols) {
    File& f = *file_;
    const SafeTensor& t = f.get(name + ".weight");
    W w;
    w.type = kTypeF16;
    w.n = t.shape[0];
    const int64_t K = t.shape[1];
    w.k = cols > 0 ? cols : K;
    w.row_bytes = (uint64_t) w.k * 2;
    bytes += (uint64_t) (w.n * w.k) * 2 + (biased ? (uint64_t) w.n * 4 : 0);
    if (dry_) return w;
    std::vector<uint16_t> tmp;
    const uint16_t* src = f.bf16(name + ".weight", tmp);
    std::vector<uint16_t> h((size_t) (w.n * w.k));
    for (int64_t r = 0; r < w.n; ++r)
        for (int64_t c = 0; c < w.k; ++c) h[(size_t) (r * w.k + c)] = f16_from_f32(bf16_value(src[r * K + c]));
    uint16_t* d = alloc<uint16_t>(owned, (int64_t) h.size());
    gpu::copy(d, h.data(), h.size() * 2);
    w.d = d;
    if (biased) {
        const auto v = f.f32(name + ".bias");
        float* db = alloc<float>(owned, (int64_t) v.size());
        gpu::copy(db, v.data(), v.size() * 4);
        *bias = db;
    }
    return w;
}

const float* AnimaDit::vec(const std::string& name, std::vector<void*>& owned, uint64_t& bytes) {
    const int64_t n = file_->get(name).elements();
    bytes += (uint64_t) n * 4;
    if (dry_) return nullptr;
    const auto v = file_->f32(name);
    float* d = alloc<float>(owned, n);
    gpu::copy(d, v.data(), v.size() * 4);
    return d;
}

// the LLM adapter's weights, FP16 on the GPU (the embedding stays in the file: a prompt's rows are gathered on the host)
void AnimaDit::load_text() {
    text_bytes_ = 0;
    const auto keep = [](W& dst, const W& src) {
        LoraLayer* l = dst.lora;
        dst = src;
        dst.lora = l;
    };
    const SafeTensor& e = file_->get("llm_adapter.embed.weight");
    vocab_ = e.shape[0];
    ad_C_ = e.shape[1];
    for (int i = 0;; ++i) {
        const std::string p = "llm_adapter.blocks." + std::to_string(i) + ".";
        if (file_->find(p + "self_attn.q_proj.weight") == nullptr) break;
        if (ablocks_.size() <= (size_t) i) ablocks_.resize((size_t) i + 1);
        AdapterBlock& b = ablocks_[(size_t) i];
        keep(b.sq, dense(p + "self_attn.q_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.sk, dense(p + "self_attn.k_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.sv, dense(p + "self_attn.v_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.so, dense(p + "self_attn.o_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.cq, dense(p + "cross_attn.q_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.ck, dense(p + "cross_attn.k_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.cv, dense(p + "cross_attn.v_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.co, dense(p + "cross_attn.o_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.m0, dense(p + "mlp.0", true, &b.bm0, text_allocs_, text_bytes_));
        keep(b.m2, dense(p + "mlp.2", true, &b.bm2, text_allocs_, text_bytes_));
        b.n_sa = vec(p + "norm_self_attn.weight", text_allocs_, text_bytes_);
        b.n_ca = vec(p + "norm_cross_attn.weight", text_allocs_, text_bytes_);
        b.n_ml = vec(p + "norm_mlp.weight", text_allocs_, text_bytes_);
        b.sqn = vec(p + "self_attn.q_norm.weight", text_allocs_, text_bytes_);
        b.skn = vec(p + "self_attn.k_norm.weight", text_allocs_, text_bytes_);
        b.cqn = vec(p + "cross_attn.q_norm.weight", text_allocs_, text_bytes_);
        b.ckn = vec(p + "cross_attn.k_norm.weight", text_allocs_, text_bytes_);
    }
    if (ablocks_.empty()) throw std::runtime_error("image engine (Anima DiT): the LLM adapter has no blocks");
    ad_hd_ = file_->get("llm_adapter.blocks.0.self_attn.q_norm.weight").shape[0];
    ad_heads_ = ad_C_ / ad_hd_;
    ad_F_ = ablocks_[0].m0.n;
    keep(aout_, dense("llm_adapter.out_proj", true, &b_aout_, text_allocs_, text_bytes_));
    anorm_ = vec("llm_adapter.norm.weight", text_allocs_, text_bytes_);
    text_loaded_ = !dry_;
    build_lora_targets();
}

void AnimaDit::drop_text() {
    if (!text_loaded_) return;
    gpu::stream_sync(s_);
    for (void* p : text_allocs_) gpu::free(p);
    text_allocs_.clear();
    ad_scratch_ = nullptr;
    ad_bytes_ = 0;
    text_loaded_ = false;
}

std::pair<double, double> AnimaDit::time_forms(int64_t rows) const {
    const int64_t T = rows, K = C_, N = F_;
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
    } catch (const std::exception& e) {
        std::fprintf(stderr, "image engine: timing Anima's DiT in 16 and 8 bits: %s (16 bits kept)\n", e.what());
        w8 = 0.0;
    }
    gpu::stream_sync(s_);
    for (void* p : held) gpu::free(p);
    return {f16, w8};
}

MfOut AnimaDit::product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
                        const float* bias, bool same_input, bool defer) {
    MfOut out;
    out.f = y;
    out.C = w.n;
    if (T <= 0) return out;
    LoraLayer* lora = w.baked ? nullptr : w.lora;
    const float* col = lora != nullptr && lora->has_base() ? lora->base() : nullptr;
    const auto* d = base != nullptr ? base + reinterpret_cast<uintptr_t>(w.d) : static_cast<const uint8_t*>(w.d);
    const bool oft = lora != nullptr && lora->has_oft();
    if (w.type == kTypeW8) {
        float* scales = reinterpret_cast<float*>(xq8_ + xq8_rows_ * xq8_cols_);
        if (!same_input) xs_next_ ^= 1;
        float* sx = scales + xs_next_ * xq8_rows_;
        if (!same_input) qi_rows_i8(x16, false, T, w.k, xq8_, sx, rs, s_);
        prefill::Gemm::i8(xq8_, reinterpret_cast<const int8_t*>(d), reinterpret_cast<int32_t*>(y), T, w.n, w.k, w.n,
                          s_);
        const auto* sw = reinterpret_cast<const float*>(d + w.n * w.k);
        if (defer && lora == nullptr) {   // the reader scales the sums (sx carries the rows' scales)
            out.f = nullptr;
            out.i = reinterpret_cast<const int32_t*>(y);
            out.sx = sx;
            out.sw = sw;
            out.b = bias;
            return out;
        }
        qi_i8_scale(y, T, w.n, w.n, sx, sw, oft ? nullptr : bias, col, s_);
    } else {
        if (w.type == kTypeF16) gemm_.f16(x16, reinterpret_cast<const uint16_t*>(d), y, T, w.n, w.k);
        else gemm_.native(x16, w.type, d, y, T, w.n, w.k);
        if (rs != nullptr || bias != nullptr || col != nullptr) rows_back(y, rs, oft ? nullptr : bias, T, w.n, s_, col);
    }
    if (lora != nullptr) {
        lora_scratch(T);
        lora->add(x16, T, y, w.n, 0, w.n, rs, gemm_, lora_f32_, lora16_, s_);
        add_glora(lora, x16, rs, w, d, T, y);
        if (oft && bias != nullptr) dit_rows::bias_add(y, bias, T, w.n, s_);
    }
    return out;
}

void AnimaDit::add_glora(const LoraLayer* lora, const uint16_t* x16, const float* rs, const W& w, const uint8_t* d,
                         int64_t T, float* y) {
    if (lora == nullptr || !lora->has_glora()) return;
    lora->add_glora(x16, T, 0, w.n, rs, gemm_, lora_f32_, lora16_, s_,
                    [&](const uint16_t* v, int64_t t0, int64_t nt, int64_t r0, int64_t n2) {
                        glora_weight(v, w, d, r0, n2, nt, t0, y);
                    });
}

void AnimaDit::glora_weight(const uint16_t* v, const W& w, const uint8_t* d, int64_t r0, int64_t n, int64_t nt,
                            int64_t t0, float* y) {
    if (w.type == kTypeW8) {
        qi_rows_i8(v, false, nt, w.k, lora_glora_x8_, lora_glora_sx_, nullptr, s_);
        const auto* sw = reinterpret_cast<const float*>(d + w.n * w.k) + r0;
        prefill::Gemm::i8(lora_glora_x8_, reinterpret_cast<const int8_t*>(d) + r0 * w.k,
                          reinterpret_cast<int32_t*>(lora_glora_y_), nt, n, w.k, n, s_);
        qi_i8_scale(lora_glora_y_, nt, n, n, lora_glora_sx_, sw, nullptr, nullptr, s_);
    } else if (w.type == kTypeF16) {
        gemm_.f16(v, reinterpret_cast<const uint16_t*>(d) + r0 * w.k, lora_glora_y_, nt, n, w.k, n);
    } else {
        gemm_.native(v, w.type, d + (uint64_t) r0 * w.row_bytes, lora_glora_y_, nt, n, w.k, n);
    }
    lora_output_add(y + t0 * w.n, nt, w.n, r0, lora_glora_y_, n, s_);
}

uint64_t AnimaDit::activation_bytes(int64_t hw, int n, int64_t mlp_rows) const {
    const int64_t R = n * hw, L = pad_keys(kCtxRows), TP = qi_attn_rows(hw), Rk = std::max(R, L);
    const int64_t band = mlp_rows > 0 ? std::min(R, mlp_rows) : R;
    return (uint64_t) R * C_ * 4 * 3 +                          // x, q, the attention's FP32 output
           (uint64_t) Rk * C_ * 4 * 3 +                         // k, v, a
           (uint64_t) band * F_ * 4 +                           // the MLP's first sums
           (uint64_t) std::max(R * C_, band * F_) * 2 +         // the products' FP16 input
           (uint64_t) TP * C_ * 2 * 4 +                         // the self-attention's q, k, v, output
           (uint64_t) R * C_ * 2 * 2 +                          // the attention's FP16 output, the heads' q
           (uint64_t) hw * hd_ * 4 +                            // the rotary tables
           (uint64_t) hw * L * 6 +                              // a head's scores and their probabilities
           (uint64_t) TP * C_ * 4 +                             // the plain attention's scratch
           (uint64_t) n * blocks_.size() * L * C_ * 2 * 2 +     // the contexts' kept keys and values
           (w8_ ? (uint64_t) std::max(Rk, band) * std::max(C_, F_) + (uint64_t) std::max(Rk, band) * 8 : 0) +
           (lora_cols_ > 0 ? (uint64_t) std::max(Rk, lora_tmp_t_) * (uint64_t) lora_cols_ * 6 : 0);
}

void AnimaDit::ensure(int64_t hw, int n) {
    const int64_t cap_rows = cap_n_ * cap_hw_;
    if (hw <= cap_hw_ && n <= cap_n_ && mlp_rows_ == (mlp_cap_ > 0 ? std::min(cap_rows, mlp_cap_) : cap_rows)) return;
    hw = std::max(hw, cap_hw_);
    n = std::max<int>(n, (int) cap_n_);
    drop_activations();
    const int64_t R = n * hw, L = pad_keys(std::max(kCtxRows, cap_L_)), TP = qi_attn_rows(hw), Rk = std::max(R, L);
    mlp_rows_ = mlp_cap_ > 0 ? std::min(R, mlp_cap_) : R;
    x_ = alloc<float>(allocs_, R * C_);
    q_ = alloc<float>(allocs_, Rk * C_);
    k_ = alloc<float>(allocs_, Rk * C_);
    v_ = alloc<float>(allocs_, Rk * C_);
    a_ = alloc<float>(allocs_, Rk * C_);
    o32_ = alloc<float>(allocs_, R * C_);
    g_ = alloc<float>(allocs_, mlp_rows_ * F_);
    h16_ = alloc<uint16_t>(allocs_, std::max(Rk * C_, mlp_rows_ * F_));
    rs_ = alloc<float>(allocs_, std::max(Rk, mlp_rows_));
    rs2_ = alloc<float>(allocs_, std::max(Rk, mlp_rows_));
    rs_o_ = alloc<float>(allocs_, Rk);
    hs_ = alloc<float>(allocs_, heads_);
    vpart_ = alloc<float>(allocs_, hw * heads_);
    if (w8_) {
        xq8_rows_ = std::max(Rk, mlp_rows_);
        xq8_cols_ = std::max(C_, F_);
        xq8_ = alloc<int8_t>(allocs_, xq8_rows_ * xq8_cols_ + 2 * xq8_rows_ * 4);
    }
    jrows_ = TP;
    jq16_ = alloc<uint16_t>(allocs_, TP * C_, true);
    jk16_ = alloc<uint16_t>(allocs_, TP * C_, true);
    jv16_ = alloc<uint16_t>(allocs_, TP * C_, true);
    jo16_ = alloc<uint16_t>(allocs_, TP * C_);
    o16_ = alloc<uint16_t>(allocs_, Rk * C_);
    qh16_ = alloc<uint16_t>(allocs_, Rk * C_);
    sc_ = alloc<float>(allocs_, std::max(hw, L) * L);
    p16_ = alloc<uint16_t>(allocs_, std::max(hw, L) * L);
    cos_ = alloc<float>(allocs_, hw * (hd_ / 2));
    sin_ = alloc<float>(allocs_, hw * (hd_ / 2));
    rope_h_ = rope_w_ = -1;
    if (std::strcmp(qi_attention_path(), "plain (FP32)") == 0) attn_scratch_ = alloc<float>(allocs_, TP * C_);
    cap_hw_ = hw;
    cap_n_ = n;
    dirty_[0] = dirty_[1] = 0;
}

void AnimaDit::drop_activations() {
    gpu::stream_sync(s_);
    for (void* p : {(void*) x_, (void*) q_, (void*) k_, (void*) v_, (void*) a_, (void*) o32_, (void*) g_, (void*) h16_,
                    (void*) rs_, (void*) rs2_, (void*) rs_o_, (void*) hs_, (void*) vpart_, (void*) xq8_,
                    (void*) jq16_, (void*) jk16_, (void*) jv16_, (void*) jo16_, (void*) o16_, (void*) qh16_,
                    (void*) sc_, (void*) p16_, (void*) cos_, (void*) sin_, (void*) attn_scratch_, (void*) lora_f32_,
                    (void*) lora16_, (void*) lora_glora_y_, (void*) lora_glora_x8_, (void*) lora_glora_sx_})
        release(allocs_, p);
    x_ = q_ = k_ = v_ = a_ = o32_ = g_ = rs_ = rs2_ = rs_o_ = hs_ = vpart_ = sc_ = cos_ = sin_ = attn_scratch_ = nullptr;
    h16_ = jq16_ = jk16_ = jv16_ = jo16_ = o16_ = qh16_ = p16_ = nullptr;
    xq8_ = nullptr;
    lora_f32_ = nullptr;
    lora16_ = nullptr;
    lora_tmp_t_ = 0;
    lora_glora_y_ = nullptr;
    lora_glora_x8_ = nullptr;
    lora_glora_sx_ = nullptr;
    cap_hw_ = cap_n_ = 0;
    rope_h_ = rope_w_ = -1;
    drop_kv();
}

void AnimaDit::drop_kv() {
    gpu::stream_sync(s_);
    for (Ctx& c : ctx_)
        for (uint16_t*& p : c.kv) {
            release(allocs_, p);
            p = nullptr;
        }
}

// The image tokens' RoPE (VideoRopePosition3DEmb with one frame): pairs 0 .. 21 by the time (0: not turned), 22 .. 42
// by the row, 43 .. 63 by the column, each axis's frequencies 10000^(-2k / its width) (widths 44, 42, 42 of the 128)
void AnimaDit::rope_tables(int64_t h, int64_t w) {
    if (rope_h_ == h && rope_w_ == w) return;
    const int64_t half = hd_ / 2, dh = hd_ / 6 * 2, dt = hd_ - 2 * dh, nt = dt / 2, nh = dh / 2;
    std::vector<float> fr((size_t) half);
    for (int64_t j = 0; j < half; ++j) {
        const bool t = j < nt;
        const int64_t k = t ? j : (j - nt) % nh;
        const float range = (float) (2 * k) / (float) (t ? dt : dh);
        fr[(size_t) j] = 1.f / std::pow(10000.f, range);
    }
    std::vector<float> c((size_t) (h * w * half)), s(c.size());
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x)
            for (int64_t j = 0; j < half; ++j) {
                const float pos = j < nt ? 0.f : j < nt + nh ? (float) y : (float) x;
                const float ph = pos * fr[(size_t) j];
                c[(size_t) ((y * w + x) * half + j)] = std::cos(ph);
                s[(size_t) ((y * w + x) * half + j)] = std::sin(ph);
            }
    gpu::copy(cos_, c.data(), c.size() * 4);
    gpu::copy(sin_, s.data(), s.size() * 4);
    rope_h_ = h;
    rope_w_ = w;
}

// the timestep embedder: the sinusoid of sigma rounded to bf16, its two layers (the AdaLN-LoRA term) and its RMS norm
// (the blocks' embedding, whose SiLU every modulation reads)
void AnimaDit::time_embedding(float sigma) {
    const int64_t half = C_ / 2;
    std::vector<float> e((size_t) C_);
    for (int64_t i = 0; i < half; ++i) {
        const float ex = std::exp(-std::log(10000.f) * (float) i / (float) half);
        const float ph = sigma * ex;
        e[(size_t) i] = round_bf16(std::cos(ph));
        e[(size_t) (half + i)] = round_bf16(std::sin(ph));
    }
    double ss = 0.0;
    for (const float v : e) ss += (double) v * v;
    const float inv = 1.f / std::sqrt((float) (ss / (double) C_) + kEps);
    std::vector<uint16_t> h((size_t) (2 * C_));
    for (int64_t i = 0; i < C_; ++i) {
        h[(size_t) i] = f16_from_f32(e[(size_t) i]);
        const float y = round_bf16(e[(size_t) i] * inv * tnorm_host_[(size_t) i]);
        h[(size_t) (C_ + i)] = f16_from_f32(y / (1.f + std::exp(-y)));
    }
    uint16_t* s16 = m16_;   // the sinusoid, then the embedding's SiLU in e16_
    gpu::copy_async(s16, h.data(), (size_t) C_ * 2, s_);
    gpu::copy_async(e16_, h.data() + C_, (size_t) C_ * 2, s_);
    gpu::stream_sync(s_);
    product(s16, nullptr, t1_, nullptr, mh_, 1, nullptr);
    qi_silu(mh_, C_, m16_, s_);
    product(m16_, nullptr, t2_, nullptr, alora_, 1, nullptr);
}

void AnimaDit::modulation(const W& w1, const W& w2, const uint8_t* base, float* out) {
    product(e16_, nullptr, w1, base, mh_, 1, nullptr);
    qi_to_f16(mh_, w1.n, m16_, s_);
    product(m16_, nullptr, w2, base, out, 1, nullptr);
    add_rows(out, alora_, w2.n, s_);
}

void AnimaDit::attn_kv(const MfOut& k, const MfOut& v, int64_t Tk, int64_t heads, int64_t hd, const float* wk,
                       const float* cos, const float* sin, uint16_t* kv) {
    const int64_t Tp = pad_keys(Tk);
    gpu::memset_async(kv, 0, (size_t) (2 * heads * Tp * hd) * 2, s_);
    heads16(k, Tk, Tp, heads, hd, wk, cos, sin, kv, s_);
    values_t16(v, Tk, Tp, heads, hd, kv + heads * Tp * hd, s_);
}

void AnimaDit::attn_q(const MfOut& q, int64_t Tq, int64_t heads, int64_t hd, const float* wq, const float* cos,
                      const float* sin, const uint16_t* kv, int64_t Tk, float* out) {
    const int64_t Tp = pad_keys(Tk), C = heads * hd;
    heads16(q, Tq, Tq, heads, hd, wq, cos, sin, qh16_, s_);
    const float scale = 1.f / std::sqrt((float) hd);
    for (int64_t h = 0; h < heads; ++h) {
        gemm_.f16(qh16_ + h * Tq * hd, kv + h * Tp * hd, sc_, Tq, Tp, hd);
        softmax16(sc_, Tq, Tk, Tp, scale, p16_, s_);
        gemm_.f16(p16_, kv + heads * Tp * hd + h * hd * Tp, out + h * hd, Tq, hd, Tp, C);
    }
}

// The adapter (comfy/ldm/anima/model.py's LLMAdapter) of the T5 ids over the states, its rows times their weights,
// into each context's FP16 rows padded with zero rows to 512.  The contexts go through it as one run of rows (each
// attending to its own), so that a guided request's two cost the launches of one.
void AnimaDit::adapter(const std::vector<AdapterIn>& in) {
    const int64_t C = ad_C_;
    std::vector<int64_t> lt, lq;
    int64_t LT = 0, LQ = 0;
    for (const AdapterIn& a : in) {
        if (a.t5->empty() || a.rows <= 0) throw std::invalid_argument("image engine (Anima): an empty prompt");
        lt.push_back((int64_t) a.t5->size());
        lq.push_back(a.rows);
        LT += lt.back();
        LQ += lq.back();
        cap_L_ = std::max(cap_L_, std::max(kCtxRows, lt.back()));
    }
    // the second context's first rows (the first's when there is one)
    const int64_t t0 = in.size() > 1 ? lt[0] : LT, q0 = in.size() > 1 ? lq[0] : LQ;
    // the working buffers, carved out of one region kept for the next prompt (256-byte aligned pieces)
    const int64_t W = std::max(C, ad_F_), RT = std::max<int64_t>(64, std::max(LT, LQ)), half = ad_hd_ / 2;
    const uint64_t sizes[] = {(uint64_t) (RT * C) * 4,     (uint64_t) (RT * src_dim_) * 4, (uint64_t) (RT * src_dim_) * 2,
                              (uint64_t) (RT * W) * 2,     (uint64_t) (RT * C) * 4 * 3,    (uint64_t) (RT * W) * 4,
                              (uint64_t) (2 * RT * C) * 2, (uint64_t) RT * 4,              (uint64_t) (4 * RT * half) * 4};
    uint64_t need = 0;
    for (const uint64_t b : sizes) need += (b + 255) / 256 * 256;
    if (need > ad_bytes_) {
        gpu::stream_sync(s_);
        release(text_allocs_, ad_scratch_);
        ad_scratch_ = alloc<uint8_t>(text_allocs_, (int64_t) need);
        ad_bytes_ = need;
    }
    uint8_t* at = ad_scratch_;
    auto take = [&](uint64_t b) { uint8_t* p = at; at += (b + 255) / 256 * 256; return p; };
    float* xa = reinterpret_cast<float*>(take(sizes[0]));
    float* src = reinterpret_cast<float*>(take(sizes[1]));
    uint16_t* s16 = reinterpret_cast<uint16_t*>(take(sizes[2]));
    uint16_t* x16 = reinterpret_cast<uint16_t*>(take(sizes[3]));
    float* q = reinterpret_cast<float*>(take(sizes[4]));
    float *k = q + RT * C, *v = k + RT * C;
    float* o = reinterpret_cast<float*>(take(sizes[5]));
    uint16_t* qh = reinterpret_cast<uint16_t*>(take(sizes[6]));
    uint16_t* kh = qh + RT * C;
    float* wrow = reinterpret_cast<float*>(take(sizes[7]));
    float* ropes = reinterpret_cast<float*>(take(sizes[8]));
    // the ids' embedding rows from the file, their weights and the states, each context's after the other's
    {
        const SafeTensor& e = file_->get("llm_adapter.embed.weight");
        const uint8_t* base = file_->raw(e);
        const int kind = e.dtype == "BF16" ? 0 : e.dtype == "F16" ? 1 : e.dtype == "F32" ? 2 : -1;
        if (kind < 0) throw std::runtime_error("image engine (Anima DiT): the adapter's embedding is " + e.dtype);
        std::vector<float> rowsf((size_t) (LT * C)), w((size_t) LT), st((size_t) (LQ * src_dim_));
        int64_t t = 0, r = 0;
        for (const AdapterIn& a : in) {
            for (size_t i = 0; i < a.t5->size(); ++i, ++t) {
                const int64_t id = (*a.t5)[i];
                if (id < 0 || id >= vocab_)
                    throw std::invalid_argument("image engine (Anima): a T5 id outside the vocabulary");
                w[(size_t) t] = (*a.t5w)[i];
                float* dst = rowsf.data() + t * C;
                if (kind == 2) {
                    std::memcpy(dst, base + id * C * 4, (size_t) C * 4);
                    continue;
                }
                const uint8_t* row = base + id * C * 2;
                for (int64_t ch = 0; ch < C; ++ch) {
                    uint16_t bits;
                    std::memcpy(&bits, row + ch * 2, 2);
                    dst[ch] = kind == 0 ? bf16_value(bits) : kernels::f32_from_f16(bits);
                }
            }
            std::memcpy(st.data() + r * src_dim_, a.states, (size_t) (a.rows * src_dim_) * 4);
            r += a.rows;
        }
        std::vector<float> ct, sn, cc, sc;
        rope_1d(lt, half, ct, sn);
        rope_1d(lq, half, cc, sc);
        gpu::copy_async(xa, rowsf.data(), rowsf.size() * 4, s_);
        gpu::copy_async(wrow, w.data(), w.size() * 4, s_);
        gpu::copy_async(src, st.data(), st.size() * 4, s_);
        gpu::copy_async(ropes, ct.data(), ct.size() * 4, s_);
        gpu::copy_async(ropes + RT * half, sn.data(), sn.size() * 4, s_);
        gpu::copy_async(ropes + 2 * RT * half, cc.data(), cc.size() * 4, s_);
        gpu::copy_async(ropes + 3 * RT * half, sc.data(), sc.size() * 4, s_);
        gpu::stream_sync(s_);   // the host vectors go
    }
    const float *dct = ropes, *dst = ropes + RT * half, *dcc = ropes + 2 * RT * half, *dsc = ropes + 3 * RT * half;
    const float scale = 1.f / std::sqrt((float) ad_hd_);
    // the attention of the rows' q over k and v, each context's over its own
    const auto attend = [&](const MfOut& mq, const float* wq, const float* cq, const float* sq, const MfOut& mk,
                            const MfOut& mv, const float* wk, const float* ck, const float* sk, int64_t Tk, int64_t k0) {
        heads16(mq, LT, LT, ad_heads_, ad_hd_, wq, cq, sq, qh, s_);
        heads16(mk, Tk, Tk, ad_heads_, ad_hd_, wk, ck, sk, kh, s_);
        if (ad_hd_ == 64) small_attention<64>(qh, kh, mv, LT, t0, Tk, k0, ad_heads_, scale, o, s_);
        else small_attention<128>(qh, kh, mv, LT, t0, Tk, k0, ad_heads_, scale, o, s_);
    };
    rows16_plain(Plain{src, src_dim_}, LQ, src_dim_, s16, s_);
    for (const AdapterBlock& b : ablocks_) {
        // self-attention among the ids
        rows16_plain(RmsW{xa, b.n_sa, nullptr, C}, LT, C, x16, s_);
        const MfOut mq = product(x16, nullptr, b.sq, nullptr, q, LT, nullptr);
        const MfOut mk = product(x16, nullptr, b.sk, nullptr, k, LT, nullptr);
        const MfOut mv = product(x16, nullptr, b.sv, nullptr, v, LT, nullptr);
        attend(mq, b.sqn, dct, dst, mk, mv, b.skn, dct, dst, LT, t0);
        rows16_plain(Plain{o, C}, LT, C, x16, s_);
        product(x16, nullptr, b.so, nullptr, q, LT, nullptr);
        add_rows(xa, q, LT * C, s_);
        // cross-attention to the states, each side rotated by its own positions
        rows16_plain(RmsW{xa, b.n_ca, nullptr, C}, LT, C, x16, s_);
        const MfOut cq = product(x16, nullptr, b.cq, nullptr, q, LT, nullptr);
        const MfOut ck = product(s16, nullptr, b.ck, nullptr, k, LQ, nullptr);
        const MfOut cv = product(s16, nullptr, b.cv, nullptr, v, LQ, nullptr);
        attend(cq, b.cqn, dct, dst, ck, cv, b.ckn, dcc, dsc, LQ, q0);
        rows16_plain(Plain{o, C}, LT, C, x16, s_);
        product(x16, nullptr, b.co, nullptr, q, LT, nullptr);
        add_rows(xa, q, LT * C, s_);
        // the MLP
        rows16_plain(RmsW{xa, b.n_ml, nullptr, C}, LT, C, x16, s_);
        const MfOut h = product(x16, nullptr, b.m0, nullptr, o, LT, b.bm0);
        rows16_plain(GeluErf{h}, LT, ad_F_, x16, s_);
        product(x16, nullptr, b.m2, nullptr, q, LT, b.bm2);
        add_rows(xa, q, LT * C, s_);
    }
    rows16_plain(Plain{xa, C}, LT, C, x16, s_);
    product(x16, nullptr, aout_, nullptr, q, LT, b_aout_);
    // each context's rows: the norm times the ids' weights, then the zero rows
    gpu::stream_sync(s_);
    int64_t t = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        Ctx& c = *in[i].c;
        const int64_t L = std::max(kCtxRows, lt[i]);
        if (c.c16 == nullptr || c.rows != L) {
            release(allocs_, c.c16);
            release(allocs_, c.rs);
            c.c16 = alloc<uint16_t>(allocs_, L * C, true);
            c.rs = alloc<float>(allocs_, L);
            std::vector<float> ones((size_t) L, 1.f);
            gpu::copy(c.rs, ones.data(), ones.size() * 4);
        } else {   // the zero rows after these, at a scale of 1 as alloc left them
            gpu::memset_async(c.c16 + lt[i] * C, 0, (size_t) ((L - lt[i]) * C) * 2, s_);
            std::vector<float> ones((size_t) L, 1.f);
            gpu::copy(c.rs, ones.data(), ones.size() * 4);
        }
        rows16(RmsW{q + t * C, anorm_, wrow + t, C}, lt[i], C, c.c16, c.rs, s_);
        c.rows = L;
        t += lt[i];
    }
    gpu::stream_sync(s_);
}

void AnimaDit::set_context(int i, const float* states, int64_t rows, const std::vector<int32_t>& t5,
                           const std::vector<float>& t5_weights) {
    set_contexts({{i, states, rows, &t5, &t5_weights}});
}

void AnimaDit::set_contexts(const std::vector<ContextIn>& in) {
    gpu::stream_sync(s_);
    std::vector<AdapterIn> run;
    for (const ContextIn& e : in) {
        Ctx& c = ctx_[e.i];
        std::fill(c.kv_ok.begin(), c.kv_ok.end(), (char) 0);
        if (e.rows <= 0) {
            for (uint16_t*& p : c.kv) {
                release(allocs_, p);
                p = nullptr;
            }
            release(allocs_, c.c16);
            release(allocs_, c.rs);
            c.c16 = nullptr;
            c.rs = nullptr;
            c.rows = 0;
            continue;
        }
        if (e.t5->size() != e.t5w->size())
            throw std::invalid_argument("image engine (Anima): a T5 weight for each id");
        run.push_back({&c, e.states, e.rows, e.t5, e.t5w});
    }
    if (run.empty()) return;
    if (!text_loaded_) load_text();
    adapter(run);
    bool longer = false;
    for (const AdapterIn& a : run) longer = longer || pad_keys(a.c->rows) > pad_keys(kCtxRows);
    if (longer && cap_hw_ > 0) {   // a longer context: the scores' buffers again
        const int64_t hw = cap_hw_, n = cap_n_;
        drop_activations();
        ensure(hw, (int) n);
    }
}

// A block's weights as each copy of it reaches the GPU: FP16 from the file's bf16 (the FP16 blocks), and the adapters
// baked in (set_loras); none of it for 8-bit blocks without baked adapters (the stream then copies them as they are)
void AnimaDit::install_prepare() {
    if (q8_ && !baked_any_) {
        blocks_stream_.set_prepare(nullptr);
        return;
    }
    blocks_stream_.set_prepare([this](int i, uint8_t* d) {
        sycl::queue& q = q_of(s_);
        const Block& b = blocks_[(size_t) i];
        for (const W* w : {&b.sq, &b.sk, &b.sv, &b.so, &b.cq, &b.ck, &b.cv, &b.co, &b.l1, &b.l2, &b.msa1, &b.msa2,
                           &b.mca1, &b.mca2, &b.mml1, &b.mml2}) {
            uint8_t* wd = d + reinterpret_cast<uintptr_t>(w->d);
            if (!q8_) bf16_to_f16_inplace(reinterpret_cast<uint16_t*>(wd), w->n * w->k, q);
            if (w->baked) w->lora->bake(w->type, wd, w->n, w->k, gemm_, bake_tmp_, bake_at16_, s_);
        }
    });
}

// The cross-attention through the GPU maker's fused attention (qi_sdpa: oneDNN, cuDNN, aotriton) where the build and
// the GPU have one: a probe of a few rows.  Per head, the library's products and a softmax took the B70 51 ms of a
// 421 ms step (2026-11-02), an eighth of the self-attention's work in half its time.
bool AnimaDit::fused_attention_ok() {
    const int64_t rows = 16, C = heads_ * hd_;
    uint16_t* t = nullptr;
    if (!gpu::alloc_device(&t, (size_t) (4 * rows * C) * 2)) return false;
    gpu::memset_async(t, 0, (size_t) (4 * rows * C) * 2, s_);
    const bool ok = qi_sdpa(t, t + rows * C, t + 2 * rows * C, 0, rows, rows, heads_, hd_, t + 3 * rows * C, s_);
    gpu::stream_sync(s_);
    gpu::free(t);
    return ok;
}

void AnimaDit::forward(const float* x, float sigma, int64_t h, int64_t w, int n, float* out) {
    const int64_t HW = h * w, C = C_, R = n * HW, TV = 4 * lat_;
    for (int i = 0; i < n; ++i)
        if (ctx_[i].rows <= 0) throw std::runtime_error("image engine (Anima DiT): a context not set");
    ensure(HW, n);
    rope_tables(h, w);
    // the patches through the embedder (each context's image rows the same at the start)
    qi_to_f16(x, HW * TV, h16_, s_);
    product(h16_, nullptr, xemb_, nullptr, x_, HW, nullptr);
    for (int i = 1; i < n; ++i) gpu::copy_async(x_ + i * HW * C, x_, (size_t) (HW * C) * 4, s_);
    time_embedding(sigma);
    if (cross_sdpa_ < 0) cross_sdpa_ = fused_attention_ok() ? 1 : 0;
    const bool fused = cross_sdpa_ == 1;
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        const Block& b = blocks_[(size_t) bi];
        const uint8_t* base = blocks_stream_.begin(bi);
        auto f = [base](uint64_t o) { return reinterpret_cast<const float*>(base + o); };
        // self-attention, each context's image rows among themselves
        modulation(b.msa1, b.msa2, base, mod_);
        rows16(LnMod{x_, mod_, mod_ + C, C}, R, C, h16_, rs_, s_);
        const MfOut q = product(h16_, rs_, b.sq, base, q_, R, nullptr, false, true);
        const MfOut k = product(h16_, rs_, b.sk, base, k_, R, nullptr, b.sq.type == kTypeW8, true);
        const MfOut v = product(h16_, rs_, b.sv, base, v_, R, nullptr, b.sk.type == kTypeW8, true);
        for (int i = 0; i < n; ++i) {
            const int64_t r0 = i * HW;
            if (dirty_[0] > HW)
                for (uint16_t* jb : {jq16_, jk16_, jv16_})
                    gpu::memset_async(jb + HW * C, 0, (size_t) ((dirty_[0] - HW) * C) * 2, s_);
            dirty_[0] = HW;
            head_max(v.from(r0), HW, MfOut{}, 0, heads_, hd_, vpart_, s_);
            head_scale(vpart_, HW, heads_, hs_, s_);
            qk_rope16(q.from(r0), k.from(r0), v.from(r0), f(b.sqn), f(b.skn), cos_, sin_, hs_, HW, heads_, hd_, jq16_,
                      jk16_, jv16_, s_);
            qi_attention(jq16_, jk16_, jv16_, HW, heads_, hd_, 0, jo16_, attn_scratch_, s_, nullptr, 0, false);
            rows16(Heads{jo16_, hs_, C, hd_}, HW, C, o16_ + r0 * C, rs_o_ + r0, s_);
        }
        gate_add(x_, product(o16_, rs_o_, b.so, base, a_, R, nullptr, false, true), mod_ + 2 * C, R, C, s_);
        // cross-attention to each context's rows (their keys and values made at the first step and kept, before the
        // queries: their products' input scales would take the deferred queries' slot)
        for (int i = 0; i < n; ++i) {
            Ctx& c = ctx_[i];
            const int64_t L = c.rows, Lp = pad_keys(L);
            uint16_t*& kv = c.kv[(size_t) bi];
            if (kv != nullptr && c.kv_lp[(size_t) bi] != Lp) {   // a longer or shorter context than it was made for
                gpu::stream_sync(s_);
                release(allocs_, kv);
                kv = nullptr;
            }
            if (kv == nullptr) {
                kv = alloc<uint16_t>(allocs_, 2 * heads_ * Lp * hd_);
                c.kv_lp[(size_t) bi] = Lp;
                c.kv_ok[(size_t) bi] = 0;
            }
            if (!c.kv_ok[(size_t) bi]) {
                const MfOut ck = product(c.c16, c.rs, b.ck, base, k_, L, nullptr);
                const MfOut cv = product(c.c16, c.rs, b.cv, base, v_, L, nullptr, b.ck.type == kTypeW8);
                if (fused) {   // K [L][C] normed, V [L][C]
                    heads16(ck, L, L, heads_, hd_, f(b.ckn), nullptr, nullptr, kv, s_, true);
                    values16(cv, L, C, kv + L * C, s_);
                } else {
                    attn_kv(ck, cv, L, heads_, hd_, f(b.ckn), nullptr, nullptr, kv);
                }
                c.kv_ok[(size_t) bi] = 1;
            }
        }
        modulation(b.mca1, b.mca2, base, mod_);
        rows16(LnMod{x_, mod_, mod_ + C, C}, R, C, h16_, rs_, s_);
        const MfOut cq = product(h16_, rs_, b.cq, base, q_, R, nullptr, false, true);
        if (fused) {   // each context's rows through the GPU maker's fused attention, FP16 out
            for (int i = 0; i < n; ++i) {
                const Ctx& c = ctx_[i];
                heads16(cq.from(i * HW), HW, HW, heads_, hd_, f(b.cqn), nullptr, nullptr, jq16_, s_, true);
                const uint16_t* kv = c.kv[(size_t) bi];
                if (!qi_sdpa(jq16_, kv, kv + c.rows * C, 0, HW, c.rows, heads_, hd_, o16_ + i * HW * C, s_))
                    throw std::runtime_error("image engine (Anima DiT): the fused attention failed");
            }
            gate_add(x_, product(o16_, nullptr, b.co, base, a_, R, nullptr, false, true), mod_ + 2 * C, R, C, s_);
        } else {
            for (int i = 0; i < n; ++i)
                attn_q(cq.from(i * HW), HW, heads_, hd_, f(b.cqn), nullptr, nullptr, ctx_[i].kv[(size_t) bi],
                       ctx_[i].rows, o32_ + i * HW * C);
            rows16(Plain{o32_, C}, R, C, o16_, rs_o_, s_);
            gate_add(x_, product(o16_, rs_o_, b.co, base, a_, R, nullptr, false, true), mod_ + 2 * C, R, C, s_);
        }
        // the MLP, a band of rows at a time
        modulation(b.mml1, b.mml2, base, mod_);
        for (int64_t m0 = 0; m0 < R; m0 += mlp_rows_) {
            const int64_t mr = std::min(mlp_rows_, R - m0);
            rows16(LnMod{x_ + m0 * C, mod_, mod_ + C, C}, mr, C, h16_, rs_, s_);
            rows16(GeluErf{product(h16_, rs_, b.l1, base, g_, mr, nullptr, false, true)}, mr, F_, h16_, rs2_, s_);
            gate_add(x_ + m0 * C, product(h16_, rs2_, b.l2, base, a_, mr, nullptr, false, true), mod_ + 2 * C, mr, C,
                     s_);
        }
        blocks_stream_.end(bi);
    }
    // the final layer: each context's rows normed and modulated, projected to the patch's values
    modulation(fm1_, fm2_, nullptr, fmod_);
    for (int i = 0; i < n; ++i) {
        rows16(LnMod{x_ + i * HW * C, fmod_, fmod_ + C, C}, HW, C, h16_, rs_, s_);
        product(h16_, rs_, flin_, nullptr, a_, HW, nullptr);
        to_tokens(a_, HW, lat_, out + i * HW * TV, s_);
    }
}

std::vector<AnimaDit::LoraTarget> AnimaDit::lora_targets() const {
    std::vector<LoraTarget> out;
    out.reserve(lora_slots_.size());
    for (const LoraSlot& s : lora_slots_) out.push_back({s.name, s.row0, s.rows, s.w->k, s.src_row});
    return out;
}

void AnimaDit::build_lora_targets() {
    lora_slots_.clear();
    auto add = [this](std::string name, W& w, bool biased) {
        if (w.n <= 0 || w.k <= 0) return;
        lora_slots_.push_back({std::move(name), &w, 0, w.n, 0, biased});
    };
    add("x_embedder.proj.1", xemb_, false);
    add("t_embedder.1.linear_1", t1_, false);
    add("t_embedder.1.linear_2", t2_, false);
    add("final_layer.linear", flin_, false);
    add("final_layer.adaln_modulation.1", fm1_, false);
    add("final_layer.adaln_modulation.2", fm2_, false);
    for (int i = 0; i < (int) ablocks_.size(); ++i) {
        const std::string p = "llm_adapter.blocks." + std::to_string(i) + ".";
        AdapterBlock& b = ablocks_[(size_t) i];
        add(p + "self_attn.q_proj", b.sq, false);
        add(p + "self_attn.k_proj", b.sk, false);
        add(p + "self_attn.v_proj", b.sv, false);
        add(p + "self_attn.o_proj", b.so, false);
        add(p + "cross_attn.q_proj", b.cq, false);
        add(p + "cross_attn.k_proj", b.ck, false);
        add(p + "cross_attn.v_proj", b.cv, false);
        add(p + "cross_attn.o_proj", b.co, false);
        add(p + "mlp.0", b.m0, true);
        add(p + "mlp.2", b.m2, true);
    }
    if (aout_.n > 0) add("llm_adapter.out_proj", aout_, true);
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        const std::string p = "blocks." + std::to_string(bi) + ".";
        Block& b = blocks_[(size_t) bi];
        add(p + "self_attn.q_proj", b.sq, false);
        add(p + "self_attn.k_proj", b.sk, false);
        add(p + "self_attn.v_proj", b.sv, false);
        add(p + "self_attn.output_proj", b.so, false);
        add(p + "cross_attn.q_proj", b.cq, false);
        add(p + "cross_attn.k_proj", b.ck, false);
        add(p + "cross_attn.v_proj", b.cv, false);
        add(p + "cross_attn.output_proj", b.co, false);
        add(p + "mlp.layer1", b.l1, false);
        add(p + "mlp.layer2", b.l2, false);
        add(p + "adaln_modulation_self_attn.1", b.msa1, false);
        add(p + "adaln_modulation_self_attn.2", b.msa2, false);
        add(p + "adaln_modulation_cross_attn.1", b.mca1, false);
        add(p + "adaln_modulation_cross_attn.2", b.mca2, false);
        add(p + "adaln_modulation_mlp.1", b.mml1, false);
        add(p + "adaln_modulation_mlp.2", b.mml2, false);
    }
}

void AnimaDit::lora_scratch(int64_t T) {
    if (lora_cols_ <= 0 || T <= 0) return;
    if (lora_glora_cols_ > 0 && lora_glora_y_ == nullptr) lora_glora_scratch();
    if (lora_f32_ != nullptr && T <= lora_tmp_t_) return;
    gpu::stream_sync(s_);
    for (void** p : {reinterpret_cast<void**>(&lora_f32_), reinterpret_cast<void**>(&lora16_)})
        if (*p != nullptr) {
            release(allocs_, *p);
            *p = nullptr;
        }
    lora_f32_ = alloc<float>(allocs_, T * lora_cols_);
    lora16_ = alloc<uint16_t>(allocs_, T * lora_cols_);
    lora_tmp_t_ = T;
}

bool AnimaDit::set_loras(const std::vector<LoraUse>& loras, std::string& err) {
    std::string key;
    for (const LoraUse& u : loras) {
        if (!u.file) continue;
        char num[32];
        std::snprintf(num, sizeof num, "%.7g", (double) u.scale);
        key += u.file->path();
        key += ':';
        key += num;
        key += ';';
    }
    if (key == lora_key_) return true;
    drop_loras();
    drop_kv();   // the kept keys and values were the other adapters' weights'
    if (key.empty()) return true;
    std::vector<std::string> unknown, text_encoder;
    // a block's number in a key ("blocks.K." or sd-scripts' dotted "blocks.K."), -1 for none
    auto block_of = [](const std::string& k) -> int {
        if (k.rfind("blocks.", 0) != 0) return -1;
        const size_t dot = k.find('.', 7);
        if (dot == std::string::npos) return -1;
        try {
            return std::stoi(k.substr(7, dot - 7));
        } catch (const std::exception&) {
            return -1;
        }
    };
    for (const LoraUse& u : loras) {
        if (!u.file) continue;
        // the release's adapter on an expanded DiT: every block it names within the map's size
        bool remap = !block_map_.empty() && (int) block_map_.size() < (int) blocks_.size();
        for (const LoraEntry& entry : u.file->entries())
            if (block_of(entry.key) >= (int) block_map_.size()) remap = false;
        for (const LoraEntry& raw : u.file->entries()) {
            LoraEntry moved;
            const LoraEntry* ep = &raw;
            if (const int b = block_of(raw.key); remap && b >= 0) {
                moved = raw;
                moved.key = "blocks." + std::to_string(block_map_[(size_t) b]) + raw.key.substr(raw.key.find('.', 7));
                ep = &moved;
            }
            const LoraEntry& entry = *ep;
            if (is_text_encoder_key(entry.key)) {
                text_encoder.push_back(entry.key);
                continue;
            }
            // the table's name as it is, or with its underscores dotted (sd-scripts' keys reach here dotted)
            std::vector<const LoraSlot*> slots;
            for (const LoraSlot& s : lora_slots_)
                if (s.name == entry.key || dotted(s.name) == entry.key) slots.push_back(&s);
            if (slots.empty()) {
                unknown.push_back(entry.key);
                continue;
            }
            int64_t rows = 0;
            for (const LoraSlot* s : slots) rows = std::max(rows, s->src_row + s->rows);
            LoraEntry alt;
            const LoraEntry& e = *lora_glora_for_weight(entry, rows, slots[0]->w->k, alt);
            const bool oft = e.kind == LoraKind::oft;
            const bool glora = e.kind == LoraKind::glora;
            if (e.kind == LoraKind::bias) {
                if (e.rows != rows) {
                    err = "image engine: " + u.file->path() + ": " + e.key + ": a bias's delta (diff_b/b_norm) of " +
                          std::to_string(e.rows) + " values, not the weight's " + std::to_string(rows) + " rows";
                    drop_loras();
                    return false;
                }
            } else if (e.rows != rows || (oft ? e.cols != 0 : e.cols != slots[0]->w->k)) {
                err = "image engine: " + u.file->path() + ": " + e.key + " is " + std::to_string(e.cols) + " x " +
                      std::to_string(e.rows) + ", not the DiT's " + std::to_string(slots[0]->w->k) + " x " +
                      std::to_string(rows);
                drop_loras();
                return false;
            }
            if (!e.bias.empty())
                for (const LoraSlot* s : slots)
                    if (!s->biased) {
                        err = "image engine: " + u.file->path() + ": " + e.key +
                              ": a bias's delta (diff_b/b_norm), and the DiT's weight adds no bias";
                        drop_loras();
                        return false;
                    }
            const float scale = u.scale * e.scale();
            if (u.scale == 0.0f || (scale == 0.0f && e.bias.empty() && e.dora.empty())) continue;
            const bool dora = !e.dora.empty();
            std::vector<float> coef((size_t) e.rows, 1.0f);
            std::vector<std::vector<float>> bases(slots.size());
            for (size_t i = 0; i < slots.size(); ++i) {
                const LoraSlot* slot = slots[i];
                std::vector<float> norms;
                if (dora && !dora_norms(*slot, norms, err)) {
                    drop_loras();
                    return false;
                }
                if (dora) {
                    std::vector<float> cf;
                    std::string why;
                    if (!lora_dora_rows(e, slot->src_row, slot->row0, slot->rows, norms, u.scale, cf, bases[i], why)) {
                        err = "image engine: " + u.file->path() + ": " + e.key + ": " + why;
                        drop_loras();
                        return false;
                    }
                    for (int64_t j = 0; j < slot->rows; ++j) coef[(size_t) (slot->src_row + j)] = cf[(size_t) j];
                }
                if (e.replaces) bases[i].assign((size_t) slot->rows, 1.0f - u.scale);
                if (const LoraPart* clash = lora_dora_overlap(slot->w->lora, slot->row0, slot->rows, dora, oft, glora,
                                                              e.replaces)) {
                    err = "image engine: " + u.file->path() + ": " + e.key + ": the weight's rows " +
                          std::to_string(slot->row0) + " .. " + std::to_string(slot->row0 + slot->rows) +
                          " are another adapter's, and " + lora_patch_clash_why(*clash, oft, glora, e.replaces);
                    drop_loras();
                    return false;
                }
            }
            auto dev = std::make_shared<LoraDeviceEntry>();
            if (!dev->upload(e, s_, err, dora ? coef.data() : nullptr)) {
                drop_loras();
                return false;
            }
            for (size_t i = 0; i < slots.size(); ++i) {
                const LoraSlot* slot = slots[i];
                if (slot->w->lora == nullptr) {
                    lora_layers_.push_back(std::make_unique<LoraLayer>());
                    slot->w->lora = lora_layers_.back().get();
                }
                LoraPart part;
                part.entry = dev;
                part.row0 = slot->row0;
                part.n = slot->rows;
                part.src_row = slot->src_row;
                part.scale = scale;
                part.bias_scale = u.scale;
                part.dora = dora;
                part.oft = oft;
                part.glora = glora;
                part.replaces = e.replaces;
                part.base = std::move(bases[i]);
                slot->w->lora->parts.push_back(std::move(part));
            }
            lora_cols_ = std::max(lora_cols_, lora_scratch_cols(e));
            if (glora) {
                lora_glora_cols_ = std::max(lora_glora_cols_, e.cols);
                lora_glora_rows_ = std::max(lora_glora_rows_, e.rows);
            }
        }
    }
    if (!text_encoder.empty()) {
        err = "image engine: a text encoder's adapter: the image engine applies adapters to the DiT only (" +
              name_list(text_encoder) + ")";
        drop_loras();
        return false;
    }
    if (!unknown.empty()) {
        err = "image engine: the DiT takes no adapter for " + name_list(unknown);
        drop_loras();
        return false;
    }
    if (lora_layers_.empty()) {
        drop_loras();
        return true;
    }
    for (const LoraSlot& slot : lora_slots_)
        if (slot.w->lora != nullptr && !slot.w->lora->build_base(slot.w->n, s_, err)) {
            drop_loras();
            return false;
        }
    if (lora_glora_cols_ > 0) lora_glora_scratch();
    if (lora_bake_) {   // the blocks' plain LoRAs folded into their weights as the blocks come back to the GPU
        int64_t nk = 0, kr = 0;
        for (Block& b : blocks_)
            for (W* w : {&b.sq, &b.sk, &b.sv, &b.so, &b.cq, &b.ck, &b.cv, &b.co, &b.l1, &b.l2, &b.msa1, &b.msa2,
                         &b.mca1, &b.mca2, &b.mml1, &b.mml2})
                if (w->lora != nullptr && w->lora->bakeable()) {
                    w->baked = true;
                    nk = std::max(nk, w->n * w->k);
                    for (const LoraPart& p : w->lora->parts) kr = std::max(kr, w->k * p.entry->rank());
                }
        if (nk > 0) {
            bake_tmp_ = alloc<float>(allocs_, nk);
            bake_at16_ = alloc<uint16_t>(allocs_, kr);
            baked_any_ = true;
            blocks_stream_.reload();   // the queued work drained before the prepare changes
            install_prepare();
        }
    }
    for (const LoraUse& u : loras)
        if (u.file) lora_files_.push_back(u.file);
    lora_key_ = key;
    return true;
}

bool AnimaDit::tensor_norms(const std::string& name, const std::vector<float>*& out, std::string& err) {
    auto it = norm_cache_.find(name);
    if (it != norm_cache_.end()) {
        out = &it->second;
        return true;
    }
    try {
        const SafeTensor& t = file_->get(name);
        const int64_t rows = t.shape[0], k = t.shape[1];
        std::vector<uint16_t> tmp;
        const uint16_t* h = file_->bf16(name, tmp);
        std::vector<float> norms((size_t) rows);
        for (int64_t r = 0; r < rows; ++r) {
            double s = 0;
            for (int64_t c = 0; c < k; ++c) {
                const float v = kernels::f32_from_bf16(h[r * k + c]);
                s += (double) v * (double) v;
            }
            norms[(size_t) r] = (float) std::sqrt(s);
        }
        file_->drop_cache(name);
        out = &norm_cache_.emplace(name, std::move(norms)).first->second;
        return true;
    } catch (const std::exception& ex) {
        err = std::string("image engine (Anima DiT): a DoRA's row norms: ") + ex.what();
        return false;
    }
}

bool AnimaDit::dora_norms(const LoraSlot& slot, std::vector<float>& out, std::string& err) {
    const std::vector<float>* norms = nullptr;
    if (!tensor_norms(slot.name + ".weight", norms, err)) return false;
    if ((int64_t) norms->size() < slot.src_row + slot.w->n) {
        err = "image engine (Anima DiT): " + slot.name + ": the file's rows are fewer than the weight's";
        return false;
    }
    out.assign(norms->begin() + slot.src_row, norms->begin() + slot.src_row + slot.w->n);
    return true;
}

uint64_t AnimaDit::lora_bytes() const {
    uint64_t n = 0;
    for (const std::unique_ptr<LoraLayer>& l : lora_layers_) n += l->bytes();
    if (lora_glora_y_ != nullptr)
        n += (uint64_t) kLoraGloraRows * ((uint64_t) lora_glora_rows_ * 4 + (uint64_t) lora_glora_cols_ + 4);
    return n;
}

void AnimaDit::drop_loras() {
    for (LoraSlot& s : lora_slots_) {
        s.w->lora = nullptr;
        s.w->baked = false;
    }
    if (baked_any_) {   // the weights as the file has them again: the blocks copied anew, without the bake
        baked_any_ = false;
        blocks_stream_.reload();
        install_prepare();
        for (void** p : {reinterpret_cast<void**>(&bake_tmp_), reinterpret_cast<void**>(&bake_at16_)})
            if (*p != nullptr) {
                release(allocs_, *p);
                *p = nullptr;
            }
    }
    lora_layers_.clear();
    lora_files_.clear();
    lora_key_.clear();
    lora_cols_ = 0;
    lora_glora_cols_ = lora_glora_rows_ = 0;
    if (lora_f32_ != nullptr || lora16_ != nullptr) {
        gpu::stream_sync(s_);
        for (void** p : {reinterpret_cast<void**>(&lora_f32_), reinterpret_cast<void**>(&lora16_)})
            if (*p != nullptr) {
                release(allocs_, *p);
                *p = nullptr;
            }
        lora_tmp_t_ = 0;
    }
    free_glora_scratch();
}

void AnimaDit::lora_glora_scratch() {
    if (lora_glora_cols_ <= 0 || lora_glora_rows_ <= 0) return;
    const size_t band = (size_t) kLoraGloraRows;
    lora_glora_y_ = alloc<float>(allocs_, (int64_t) (band * (size_t) lora_glora_rows_));
    lora_glora_x8_ = alloc<int8_t>(allocs_, (int64_t) (band * (size_t) lora_glora_cols_));
    lora_glora_sx_ = alloc<float>(allocs_, (int64_t) band);
}

void AnimaDit::free_glora_scratch() {
    for (void* p : {(void*) lora_glora_y_, (void*) lora_glora_x8_, (void*) lora_glora_sx_})
        if (p != nullptr) {
            gpu::stream_sync(s_);
            release(allocs_, p);
        }
    lora_glora_y_ = nullptr;
    lora_glora_x8_ = nullptr;
    lora_glora_sx_ = nullptr;
}

}  // namespace strata::image
