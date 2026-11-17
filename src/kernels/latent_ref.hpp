// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// src/kernels/latent_ref.hpp - the latent MoE layer's references, shared by the CPU-only check
// (tests/kernels/latent_ref_test.cpp) and the GPU parity (src/kernels/latent_parity.cpp).  Host code only.
//
//   (a) the float reference: ggml's own dequantizers, the engine's q8_1 rule (canon_math.hpp, the 256-value scale
//       of quantize_q8_1_rows where the row is whole 256s), double accumulation;
//   (b) the CPU reference: the same pipeline, the D/U products as scalar integer block chains (exact 32-value
//       integer sums, (dx * dw) * sumi a block) - the arithmetic the U kernel uses.
//
// The fixture is one layer at a published shape: H 2560, ELL 1792 or 1280, I 640, G 8 groups, E experts, T tokens,
// K routes a token, Q8_0 or Q4_0 D/U, Q3_K gate/up and IQ4_NL down, positive column scales, random group labels.
// Written after edge0-strata's q3k_real_ref.cpp.
#pragma once

#include "strata/kernels/canon_math.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace strata::latent_ref {

namespace k = strata::kernels;

inline double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

// ---- the engine's q8_1 rule on the host (canon_math.hpp; quantize_q8_1_rows' 256-value scale where the row is
// whole 256s, else a scale a 32).  The sum word is the 32 values' own sum, as the kernel stores it.
inline void q8_1_quant(const float* x, int64_t rows, int64_t n, uint8_t* y) {
    const int64_t group = n % 256 == 0 ? 256 : 32;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t g0 = 0; g0 < n; g0 += group) {
            float amax = 0;
            for (int64_t j = g0; j < g0 + group; ++j) amax = std::fmax(amax, std::fabs(x[r * n + j]));
            const float d = k::canon::q8_scale(amax), id = k::canon::q8_inv(d);
            const uint16_t dh = k::f16_from_f32(d);
            for (int64_t b = g0; b < g0 + group; b += 32) {
                float sum = 0;
                for (int64_t j = b; j < b + 32; ++j) sum += x[r * n + j];
                uint8_t* out = y + (r * (n / 32) + b / 32) * 36;
                const uint16_t sh = k::f16_from_f32(sum);
                out[0] = (uint8_t) (dh & 0xFF); out[1] = (uint8_t) (dh >> 8);
                out[2] = (uint8_t) (sh & 0xFF); out[3] = (uint8_t) (sh >> 8);
                for (int64_t j = 0; j < 32; ++j) out[4 + j] = (uint8_t) k::canon::q8_value(x[r * n + b + j], id);
            }
        }
    }
}

inline std::vector<double> q8_1_dequant(const uint8_t* y, int64_t n) {
    std::vector<double> v((size_t) n);
    for (int64_t b = 0; b < n / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, y + b * 36, 2);
        const double d = k::f32_from_f16(dh);
        for (int j = 0; j < 32; ++j) v[(size_t) (b * 32 + j)] = d * (int8_t) y[b * 36 + 4 + j];
    }
    return v;
}

// ---- the scalar integer block chains (the U kernel's arithmetic, one block at a time, no SYCL)
inline float block_dot_q8_0(const uint8_t* wb, const uint8_t* xb) {
    uint16_t dw, dx;
    std::memcpy(&dw, wb, 2);
    std::memcpy(&dx, xb, 2);
    int sumi = 0;
    for (int j = 0; j < 32; ++j) sumi += (int) (int8_t) wb[2 + j] * (int) (int8_t) xb[4 + j];
    return (k::f32_from_f16(dx) * k::f32_from_f16(dw)) * (float) sumi;
}
inline float block_dot_q4_0(const uint8_t* wb, const uint8_t* xb) {
    uint16_t dw, dx;
    std::memcpy(&dw, wb, 2);
    std::memcpy(&dx, xb, 2);
    int sumi = 0;
    for (int j = 0; j < 16; ++j) {
        const int lo = (wb[2 + j] & 0xF) - 8, hi = (wb[2 + j] >> 4) - 8;
        sumi += lo * (int) (int8_t) xb[4 + j] + hi * (int) (int8_t) xb[4 + j + 16];
    }
    return (k::f32_from_f16(dx) * k::f32_from_f16(dw)) * (float) sumi;
}
inline float row_dot(const uint8_t* w, const uint8_t* xq, int64_t n, int fmt) {
    float acc = 0;
    for (int64_t b = 0; b < n / 32; ++b)
        acc += fmt == 4 ? block_dot_q4_0(w + b * 18, xq + b * 36) : block_dot_q8_0(w + b * 34, xq + b * 36);
    return acc;
}

// ---- the fixture: one layer's weights and routing at a published shape
struct Fixture {
    int64_t H = 2560, ELL, I = 640;
    int G = 8, E = 16, T = 2, K = 10, fmt;
    int64_t cap;
    size_t d_row, u_row, gu_row, dn_row, up_off, down_off, blob_bytes;
    std::vector<float> x, s, inv, w;
    std::vector<uint8_t> D, U;
    std::vector<float> Df, Uf;
    std::vector<std::vector<uint8_t>> blobs;
    std::vector<std::vector<float>> Af;   // per expert: gate rows, up rows, down rows, dequantized
    std::vector<int32_t> labels, ids;

    explicit Fixture(int64_t ell, int fmt_) : ELL(ell), fmt(fmt_), cap((int64_t) T * K) {
        d_row = (size_t) (H / 32) * (fmt == 4 ? 18 : 34);
        u_row = (size_t) (ELL / 32) * (fmt == 4 ? 18 : 34);
        gu_row = ggml_row_size(GGML_TYPE_Q3_K, ELL);
        dn_row = ggml_row_size(GGML_TYPE_IQ4_NL, I);
        up_off = (size_t) I * gu_row;
        down_off = up_off + (size_t) I * gu_row;
        blob_bytes = down_off + (size_t) ELL * dn_row;
        std::mt19937 rng(20261113 + (unsigned) ell + (unsigned) fmt);   // NOLINT(bugprone-random-generator-seed)
        std::normal_distribution<float> nd(0.f, 1.f);
        std::uniform_real_distribution<float> uni(0.5f, 2.f);
        x.resize((size_t) (T * H));
        for (float& v : x) v = nd(rng);
        s.resize((size_t) ELL);
        inv.resize((size_t) ELL);
        for (int64_t c = 0; c < ELL; ++c) { s[(size_t) c] = uni(rng); inv[(size_t) c] = 1.f / s[(size_t) c]; }
        labels.resize((size_t) E);
        for (auto& l : labels) l = (int32_t) (rng() % (unsigned) G);
        ids.resize((size_t) cap);
        w.resize((size_t) cap);
        for (int t = 0; t < T; ++t) {   // K distinct experts a token, as the router gives
            std::vector<int> perm((size_t) E);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int j = 0; j < K; ++j) {
                ids[(size_t) t * (size_t) K + (size_t) j] = perm[(size_t) j];
                w[(size_t) t * (size_t) K + (size_t) j] = 0.05f + 0.1f * uni(rng) / 2.f;
            }
        }
        const ggml_type dt = fmt == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
        const auto* td = ggml_get_type_traits_cpu(dt);
        const auto* tg = ggml_get_type_traits_cpu(GGML_TYPE_Q3_K);
        const auto* tn = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL);
        auto quant = [&](std::vector<uint8_t>& q, std::vector<float>& f, int64_t rows, int64_t cols,
                         const ggml_type_traits_cpu* tr, ggml_type t) {
            q.resize((size_t) (rows * ggml_row_size(t, cols)));
            f.resize((size_t) (rows * cols));
            std::vector<float> row((size_t) cols);
            for (int64_t r = 0; r < rows; ++r) {
                for (auto& v : row) v = nd(rng);
                tr->from_float(row.data(), q.data() + (size_t) r * ggml_row_size(t, cols), cols);
            }
            ggml_get_type_traits(t)->to_float(q.data(), f.data(), rows * cols);
        };
        quant(D, Df, (int64_t) G * ELL, H, td, dt);
        quant(U, Uf, (int64_t) G * H, ELL, td, dt);
        blobs.resize((size_t) E);
        Af.resize((size_t) E);
        for (int e = 0; e < E; ++e) {
            std::vector<uint8_t>& b = blobs[(size_t) e];
            b.resize(blob_bytes);
            std::vector<float> rowgu((size_t) ELL), rowdn((size_t) I);
            for (int64_t r = 0; r < 2 * I; ++r) {
                for (auto& v : rowgu) v = nd(rng);
                tg->from_float(rowgu.data(), b.data() + (size_t) r * gu_row, ELL);
            }
            for (int64_t r = 0; r < ELL; ++r) {
                for (auto& v : rowdn) v = nd(rng);
                tn->from_float(rowdn.data(), b.data() + down_off + (size_t) r * dn_row, I);
            }
            std::vector<float>& a = Af[(size_t) e];
            a.resize((size_t) (2 * I * ELL + ELL * I));
            ggml_get_type_traits(GGML_TYPE_Q3_K)->to_float(b.data(), a.data(), 2 * I * ELL);
            ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(b.data() + down_off, a.data() + (size_t) (2 * I * ELL),
                                                             ELL * I);
        }
    }
};

// ---- the layer: (a) the float reference (int_du false) and (b) the CPU reference (int_du true)
inline std::vector<float> layer_ref(const Fixture& f, bool int_du) {
    const int64_t H = f.H, ELL = f.ELL, I = f.I;
    const int G = f.G, T = f.T, K = f.K;
    const size_t xq_row = (size_t) (H / 32) * 36, lq_row = (size_t) (ELL / 32) * 36;
    std::vector<uint8_t> xq((size_t) (T * H / 32) * 36);
    q8_1_quant(f.x.data(), T, H, xq.data());
    std::vector<float> xl((size_t) T * G * ELL);
    for (int t = 0; t < T; ++t) {
        const auto xd = q8_1_dequant(xq.data() + (size_t) t * xq_row, H);
        for (int g = 0; g < G; ++g) {
            for (int64_t c = 0; c < ELL; ++c) {
                const int64_t row = (int64_t) g * ELL + c;
                double acc = 0;
                if (int_du) {
                    acc = row_dot(f.D.data() + (size_t) row * f.d_row, xq.data() + (size_t) t * xq_row, H, f.fmt);
                } else {
                    for (int64_t h = 0; h < H; ++h) acc += (double) f.Df[(size_t) (row * H + h)] * xd[(size_t) h];
                }
                xl[(size_t) ((t * G + g) * ELL + c)] = (float) (acc * f.inv[(size_t) c]);
            }
        }
    }
    std::vector<uint8_t> xlq(((size_t) T * G * ELL / 32) * 36);
    q8_1_quant(xl.data(), (int64_t) T * G, ELL, xlq.data());
    std::vector<float> yl((size_t) (f.cap * ELL));
    std::vector<uint8_t> hq((size_t) (I / 32) * 36);
    for (int64_t r = 0; r < f.cap; ++r) {
        const int e = f.ids[(size_t) r], g = f.labels[(size_t) e], t = (int) (r / K);
        const auto xld = q8_1_dequant(xlq.data() + ((size_t) (t * G + g)) * lq_row, ELL);
        const std::vector<float>& a = f.Af[(size_t) e];
        std::vector<float> h((size_t) I);
        for (int64_t j = 0; j < I; ++j) {
            double gate = 0, up = 0;
            for (int64_t c = 0; c < ELL; ++c) {
                gate += (double) a[(size_t) (j * ELL + c)] * xld[(size_t) c];
                up += (double) a[(size_t) ((I + j) * ELL + c)] * xld[(size_t) c];
            }
            h[(size_t) j] = (float) (gate / (1.0 + std::exp(-gate)) * up);
        }
        q8_1_quant(h.data(), 1, I, hq.data());
        const auto hd = q8_1_dequant(hq.data(), I);
        for (int64_t c = 0; c < ELL; ++c) {
            double acc = 0;
            for (int64_t j = 0; j < I; ++j) acc += (double) a[(size_t) (2 * I * ELL + c * I + j)] * hd[(size_t) j];
            yl[(size_t) (r * ELL + c)] = (float) acc;
        }
    }
    std::vector<uint8_t> ylq((size_t) (f.cap * ELL / 32) * 36);
    q8_1_quant(yl.data(), f.cap, ELL, ylq.data());
    std::vector<float> out((size_t) (T * H), 0.f);
    for (int64_t r = 0; r < f.cap; ++r) {
        const int e = f.ids[(size_t) r], g = f.labels[(size_t) e], t = (int) (r / K);
        const auto yld = int_du ? std::vector<double>() : q8_1_dequant(ylq.data() + (size_t) r * lq_row, ELL);
        for (int64_t h = 0; h < H; ++h) {
            const int64_t row = (int64_t) g * H + h;
            double acc;
            if (int_du) {
                acc = row_dot(f.U.data() + (size_t) row * f.u_row, ylq.data() + (size_t) r * lq_row, ELL, f.fmt);
            } else {
                acc = 0;
                for (int64_t c = 0; c < ELL; ++c) acc += (double) f.Uf[(size_t) (row * ELL + c)] * yld[(size_t) c];
            }
            out[(size_t) (t * H + h)] += (float) (f.w[(size_t) r] * acc);
        }
    }
    return out;
}

}  // namespace strata::latent_ref