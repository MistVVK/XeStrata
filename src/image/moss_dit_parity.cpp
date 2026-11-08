// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_dit_parity.cpp - MOSS-Image's DiT (MossDit) on a synthetic transformer (bf16 safetensors: 256 wide, two
// heads of 128, an MLP of 512, the text embedder's two blocks and one DiT block) against an FP64 host reference of the
// same model (GPU, no model):
//
//   1. the text embedder's rows of a context (its input projection, then its two blocks: RMSNorm, q and k normed a head
//      and rotated by the text's RoPE, attention among the text rows, SiTU's MLP);
//   2. the block's output over the context's rows, the image's then the text's (the time's shared modulation, the
//      four-axis RoPE, one attention over every row, the gates);
//   3. the velocity (the head over the image's rows), and a second context's that skips the block;
//   4. the velocity with the color and depth control's input projection (set_control: an adapter's video_embedder
//      over the latent's, a color map's and a depth map's channels side by side);
// each with the blocks in FP16 and in 8 bits (W8A8 where the GPU's library takes int8 products, else Q8_0).
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/moss_dit.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what, double err, double tol) {
    std::printf("  %-62s %.2e (%.0e) %s\n", what, err, tol, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

constexpr int64_t kC = 256, kHeads = 2, kHd = 128, kF = 512, kLat = 64, kTxt = 256, kFreq = 256;
constexpr int64_t kH = 4, kW = 6, kHW = kH * kW, kRows = 7, kNegRows = 3;
constexpr float kTime = 640.0f;

uint16_t bf16(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
double from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// the model's tensors: bf16 in the file, their values (as bf16 gives them) for the reference
struct Model {
    std::map<std::string, std::vector<int64_t>> shape;
    std::map<std::string, std::vector<double>> v;
    std::vector<std::string> order;
    std::mt19937 rng{20261028};  // NOLINT(bugprone-random-generator-seed): the same model each run.
    void add(const std::string& name, std::vector<int64_t> sh, float sd, float mean = 0.f) {
        int64_t n = 1;
        for (int64_t d : sh) n *= d;
        std::normal_distribution<float> nd(mean, sd);
        std::vector<double> x((size_t) n);
        for (double& e : x) e = from_bf16(bf16(nd(rng)));
        shape[name] = std::move(sh);
        v[name] = std::move(x);
        order.push_back(name);
    }
    // a linear layer [n][k] (and its bias)
    void linear(const std::string& p, int64_t n, int64_t k, bool bias, float scale = 1.f) {
        add(p + ".weight", {n, k}, scale / std::sqrt((float) k));
        if (bias) add(p + ".bias", {n}, 0.05f);
    }
    const std::vector<double>& operator[](const std::string& n) const { return v.at(n); }
    void write(const std::filesystem::path& path) const {
        std::string h = "{";
        uint64_t off = 0;
        for (const std::string& n : order) {
            const uint64_t bytes = v.at(n).size() * 2;
            h += (h.size() > 1 ? "," : "") + std::string("\"") + n + "\":{\"dtype\":\"BF16\",\"shape\":[";
            for (size_t i = 0; i < shape.at(n).size(); ++i) h += (i ? "," : "") + std::to_string(shape.at(n)[i]);
            h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
            off += bytes;
        }
        h += "}";
        h.resize((h.size() + 7) / 8 * 8, ' ');
        std::ofstream f(path, std::ios::binary);
        const uint64_t hn = h.size();
        f.write(reinterpret_cast<const char*>(&hn), 8);
        f.write(h.data(), (std::streamsize) h.size());
        for (const std::string& n : order)
            for (double x : v.at(n)) {
                const uint16_t b = bf16((float) x);
                f.write(reinterpret_cast<const char*>(&b), 2);
            }
    }
};

Model make_model() {
    Model m;
    m.linear("video_embedder", kC, kLat, true);
    m.linear("text_embedder.proj", kC, kTxt, true);
    for (int i = 0; i < 2; ++i) {
        const std::string p = "text_embedder.blocks." + std::to_string(i) + ".";
        m.linear(p + "qkv", 3 * kC, kC, true);
        m.linear(p + "proj", kC, kC, true, 0.5f);
        m.linear(p + "ffn_up", kF, kC, true);
        m.linear(p + "ffn_gate", kF, kC, true);
        m.linear(p + "ffn_down", kC, kF, true, 0.5f);
        for (const char* n : {"norm1", "norm2"}) m.add(p + n + ".weight", {kC}, 0.1f, 1.f);
        for (const char* n : {"q_norm", "k_norm"}) m.add(p + n + ".weight", {kHd}, 0.1f, 1.f);
    }
    m.linear("time_embedding.0", kC, kFreq, true);
    m.linear("time_embedding.2", kC, kC, true);
    m.linear("shared_block_modulation", 6 * kC, kC, true, 0.2f);
    m.linear("video_head_modulation", 2 * kC, kC, true, 0.2f);
    m.linear("final_linear_video", kLat, kC, false);
    m.add("final_norm_video.weight", {kC}, 0.1f, 1.f);
    const std::string p = "layers.0.";
    for (const char* n : {"proj.q", "proj.k", "proj.v"}) m.linear(p + n, kC, kC, false);
    m.linear(p + "proj.o", kC, kC, false, 0.5f);
    m.linear(p + "proj.ffn_up_gate", 2 * kF, kC, false);
    m.linear(p + "proj.ffn_down", kC, kF, false, 0.5f);
    for (const char* n : {"norm1", "norm2"}) m.add(p + n + ".weight", {kC}, 0.1f, 1.f);
    for (const char* n : {"q_norm", "k_norm"}) m.add(p + n + ".weight", {kHd}, 0.1f, 1.f);
    return m;
}

using Rows = std::vector<double>;   // [T][width]

// y [T][n] = x [T][k] W^T + b
Rows lin(const Model& m, const std::string& p, const Rows& x, int64_t T, bool bias = true) {
    const auto& w = m[p + ".weight"];
    const int64_t n = m.shape.at(p + ".weight")[0], k = m.shape.at(p + ".weight")[1];
    Rows y((size_t) (T * n));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t o = 0; o < n; ++o) {
            double s = bias ? m[p + ".bias"][(size_t) o] : 0.0;
            for (int64_t i = 0; i < k; ++i) s += x[(size_t) (t * k + i)] * w[(size_t) (o * k + i)];
            y[(size_t) (t * n + o)] = s;
        }
    return y;
}

// RMSNorm(x) w (eps 1e-6), then (1 + scale) and + shift when given
Rows rms(const Rows& x, int64_t T, int64_t C, const std::vector<double>& w, const double* shift = nullptr,
         const double* scale = nullptr) {
    Rows y(x.size());
    for (int64_t t = 0; t < T; ++t) {
        double s = 0;
        for (int64_t c = 0; c < C; ++c) s += x[(size_t) (t * C + c)] * x[(size_t) (t * C + c)];
        const double inv = 1.0 / std::sqrt(s / (double) C + 1e-6);
        for (int64_t c = 0; c < C; ++c) {
            double v = x[(size_t) (t * C + c)] * inv * w[(size_t) c];
            if (scale != nullptr && shift != nullptr) v = v * (1.0 + scale[c]) + shift[c];
            y[(size_t) (t * C + c)] = v;
        }
    }
    return y;
}

// a row's four RoPE coordinates
using Coord = std::vector<std::array<double, 4>>;

// the cosine and sine of a coordinate's phases, rounded to bf16 as the model's tables are
void rope(const std::array<double, 4>& p, std::vector<double>& c, std::vector<double>& s) {
    const int64_t bands = kHd / 8, half = kHd / 2;
    c.assign((size_t) half, 0);
    s.assign((size_t) half, 0);
    for (int64_t j = 0; j < half; ++j) {
        const float f = 1.f / (float) std::pow(10000.0, (double) ((float) (j % bands) / (float) bands));
        const float ph = (float) p[(size_t) (j / bands)] * f;
        c[(size_t) j] = from_bf16(bf16((float) std::cos((double) ph)));
        s[(size_t) j] = from_bf16(bf16((float) std::sin((double) ph)));
    }
}

// a block over T rows: its weights under prefix p (bias: the text embedder's), its modulation mod (6 C, or null)
void block(const Model& m, const std::string& p, Rows& x, int64_t T, const Coord& at, const double* mod,
           const char* q, const char* k, const char* v, const char* o, bool bias) {
    const double* sa = mod, *ca = mod ? mod + kC : nullptr, *ga = mod ? mod + 2 * kC : nullptr;
    const double* sf = mod ? mod + 3 * kC : nullptr, *cf = mod ? mod + 4 * kC : nullptr;
    const double* gf = mod ? mod + 5 * kC : nullptr;
    const Rows h = rms(x, T, kC, m[p + "norm1.weight"], sa, ca);
    Rows Q, K, V;
    if (q == nullptr) {   // the text blocks' one qkv
        const Rows qkv = lin(m, p + "qkv", h, T, bias);
        Q.resize((size_t) (T * kC));
        K = V = Q;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < kC; ++c) {
                Q[(size_t) (t * kC + c)] = qkv[(size_t) (t * 3 * kC + c)];
                K[(size_t) (t * kC + c)] = qkv[(size_t) (t * 3 * kC + kC + c)];
                V[(size_t) (t * kC + c)] = qkv[(size_t) (t * 3 * kC + 2 * kC + c)];
            }
    } else {
        Q = lin(m, p + q, h, T, bias);
        K = lin(m, p + k, h, T, bias);
        V = lin(m, p + v, h, T, bias);
    }
    // q and k normed a head, then rotated in pairs (j, j + hd / 2)
    for (Rows* r : {&Q, &K}) {
        const auto& w = m[p + (r == &Q ? "q_norm.weight" : "k_norm.weight")];
        for (int64_t t = 0; t < T; ++t) {
            std::vector<double> c, s;
            rope(at[(size_t) t], c, s);
            for (int64_t hh = 0; hh < kHeads; ++hh) {
                double* e = r->data() + t * kC + hh * kHd;
                double ss = 0;
                for (int64_t d = 0; d < kHd; ++d) ss += e[d] * e[d];
                const double inv = 1.0 / std::sqrt(ss / (double) kHd + 1e-6);
                for (int64_t d = 0; d < kHd; ++d) e[d] *= inv * w[(size_t) d];
                for (int64_t j = 0; j < kHd / 2; ++j) {
                    const double a = e[j], b = e[j + kHd / 2];
                    e[j] = a * c[(size_t) j] - b * s[(size_t) j];
                    e[j + kHd / 2] = b * c[(size_t) j] + a * s[(size_t) j];
                }
            }
        }
    }
    Rows A((size_t) (T * kC), 0.0);
    for (int64_t hh = 0; hh < kHeads; ++hh)
        for (int64_t t = 0; t < T; ++t) {
            std::vector<double> sc((size_t) T);
            double mx = -1e300, l = 0;
            for (int64_t u = 0; u < T; ++u) {
                double d = 0;
                for (int64_t e = 0; e < kHd; ++e) d += Q[(size_t) (t * kC + hh * kHd + e)] * K[(size_t) (u * kC + hh * kHd + e)];
                sc[(size_t) u] = d / std::sqrt((double) kHd);
                mx = std::max(mx, sc[(size_t) u]);
            }
            for (double& e : sc) l += (e = std::exp(e - mx));
            for (int64_t u = 0; u < T; ++u)
                for (int64_t e = 0; e < kHd; ++e)
                    A[(size_t) (t * kC + hh * kHd + e)] += sc[(size_t) u] / l * V[(size_t) (u * kC + hh * kHd + e)];
        }
    const Rows O = lin(m, p + o, A, T, bias);
    for (int64_t i = 0; i < T * kC; ++i) x[(size_t) i] += (ga ? ga[i % kC] : 1.0) * O[(size_t) i];
    // SiTU's MLP: (4 tanh(g / 4) sigmoid(g)) (25 tanh(h / 25)), h the up rows, g the gate's
    const Rows h2 = rms(x, T, kC, m[p + "norm2.weight"], sf, cf);
    Rows up, gate;
    if (bias) {
        up = lin(m, p + "ffn_up", h2, T);
        gate = lin(m, p + "ffn_gate", h2, T);
    } else {
        const Rows ug = lin(m, p + "proj.ffn_up_gate", h2, T, false);
        up.resize((size_t) (T * kF));
        gate = up;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < kF; ++c) {
                up[(size_t) (t * kF + c)] = ug[(size_t) (t * 2 * kF + c)];
                gate[(size_t) (t * kF + c)] = ug[(size_t) (t * 2 * kF + kF + c)];
            }
    }
    Rows a((size_t) (T * kF));
    for (size_t i = 0; i < a.size(); ++i) {
        const double g = gate[i];
        a[i] = 4.0 * std::tanh(g / 4.0) / (1.0 + std::exp(-g)) * (25.0 * std::tanh(up[i] / 25.0));
    }
    const Rows D = lin(m, p + (bias ? "ffn_down" : "proj.ffn_down"), a, T, bias);
    for (int64_t i = 0; i < T * kC; ++i) x[(size_t) i] += (gf ? gf[i % kC] : 1.0) * D[(size_t) i];
}

// the text embedder's rows of states [L][kTxt]
Rows text_rows(const Model& m, const Rows& states, int64_t L) {
    Rows x = lin(m, "text_embedder.proj", states, L);
    Coord at((size_t) L);
    for (int64_t i = 0; i < L; ++i) at[(size_t) i] = {0.0, (double) i, 0.0, 0.0};
    for (int i = 0; i < 2; ++i)
        block(m, "text_embedder.blocks." + std::to_string(i) + ".", x, L, at, nullptr, nullptr, nullptr, nullptr,
              "proj", true);
    return x;
}

// the time's embedding through its layers: the shared modulation (6 C) and the head's (2 C)
void time_mod(const Model& m, Rows& mod, Rows& head) {
    Rows e((size_t) kFreq);
    for (int64_t i = 0; i < kFreq / 2; ++i) {
        const double ph = (double) kTime * std::pow(10000.0, -2.0 * (double) i / (double) kFreq);
        e[(size_t) i] = from_bf16(bf16((float) std::cos(ph)));
        e[(size_t) (kFreq / 2 + i)] = from_bf16(bf16((float) std::sin(ph)));
    }
    Rows t = lin(m, "time_embedding.0", e, 1);
    for (double& v : t) v = v / (1.0 + std::exp(-v));
    t = lin(m, "time_embedding.2", t, 1);
    mod = lin(m, "shared_block_modulation", t, 1);
    head = lin(m, "video_head_modulation", t, 1);
}

// the velocity [kHW][kLat] of a context's rows x (the image's first)
Rows head_out(const Model& m, const Rows& x, const Rows& head) {
    const Rows img(x.begin(), x.begin() + kHW * kC);
    return lin(m, "final_linear_video", rms(img, kHW, kC, m["final_norm_video.weight"], head.data(), head.data() + kC),
               kHW, false);
}

// the largest difference over the largest reference value
double rel(const std::vector<float>& got, const Rows& want) {
    double d = 0, s = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        d = std::max(d, std::fabs((double) got[i] - want[i]));
        s = std::max(s, std::fabs(want[i]));
    }
    return d / s;
}

std::vector<float> fetch(const float* d, size_t n) {
    std::vector<float> h(n);
    strata::gpu::copy(h.data(), d, n * 4);
    return h;
}

// BF16 rows (the DiT's residual stream) as floats
std::vector<float> fetch(const uint16_t* d, size_t n) {
    std::vector<uint16_t> b(n);
    strata::gpu::copy(b.data(), d, n * 2);
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = strata::kernels::f32_from_bf16(b[i]);
    return h;
}

}  // namespace

int main() {
    const Model m = make_model();
    const auto path = std::filesystem::temp_directory_path() / "xestrata_moss_dit_parity.safetensors";
    m.write(path);
    std::mt19937 rng(7);  // NOLINT(bugprone-random-generator-seed): the same inputs each run.
    std::normal_distribution<float> nd;
    auto rnd = [&](int64_t n) {
        Rows r((size_t) n);
        for (double& e : r) e = nd(rng);
        return r;
    };
    const Rows states = rnd(kRows * kTxt), neg = rnd(kNegRows * kTxt), lat = rnd(kHW * kLat);
    // the reference: the text embedder's rows, the block over the context, the two velocities
    const Rows txt = text_rows(m, states, kRows), ntxt = text_rows(m, neg, kNegRows);
    Rows mod, head;
    time_mod(m, mod, head);
    auto context = [&](const Rows& t) {
        Rows x = lin(m, "video_embedder", lat, kHW);
        x.insert(x.end(), t.begin(), t.end());
        return x;
    };
    Rows x0 = context(txt);
    Coord at((size_t) (kHW + kRows));
    for (int64_t r = 0; r < kHW + kRows; ++r) {
        const int64_t y = r / kW, xc = r % kW;
        at[(size_t) r] = r < kHW ? std::array<double, 4>{0.0, (double) kRows, (double) y - (double) (kH - 1) / 2.0,
                                                        (double) xc - (double) (kW - 1) / 2.0}
                                 : std::array<double, 4>{0.0, (double) (r - kHW), 0.0, 0.0};
    }
    block(m, "layers.0.", x0, kHW + kRows, at, mod.data(), "proj.q", "proj.k", "proj.v", "proj.o", false);
    const Rows v0 = head_out(m, x0, head), v1 = head_out(m, context(ntxt), head);   // context 1 skips it
    // the control's adapter: the input projection over three latents' channels, the color's and the depth's after
    Model ctl;
    ctl.linear("video_embedder", kC, 3 * kLat, true);
    const auto ctl_path = std::filesystem::temp_directory_path() / "xestrata_moss_dit_parity_control.safetensors";
    ctl.write(ctl_path);
    const Rows col = rnd(kHW * kLat), dep = rnd(kHW * kLat);
    Rows wide((size_t) (kHW * 3 * kLat));
    for (int64_t t = 0; t < kHW; ++t)
        for (int64_t c = 0; c < kLat; ++c) {
            wide[(size_t) (t * 3 * kLat + c)] = lat[(size_t) (t * kLat + c)];
            wide[(size_t) (t * 3 * kLat + kLat + c)] = col[(size_t) (t * kLat + c)];
            wide[(size_t) (t * 3 * kLat + 2 * kLat + c)] = dep[(size_t) (t * kLat + c)];
        }
    Rows xc = lin(ctl, "video_embedder", wide, kHW);
    xc.insert(xc.end(), txt.begin(), txt.end());
    block(m, "layers.0.", xc, kHW + kRows, at, mod.data(), "proj.q", "proj.k", "proj.v", "proj.o", false);
    const Rows vc = head_out(m, xc, head);
    std::vector<float> fc(col.begin(), col.end()), fd(dep.begin(), dep.end());

    std::vector<float> fs(states.begin(), states.end()), fn(neg.begin(), neg.end()), fl(lat.begin(), lat.end());
    try {
        void* s = strata::core::Runtime::get().create_stream();
        for (const bool q8 : {false, true}) {
            strata::image::MossDit d;
            std::string err;
            if (!d.open(path.string(), s, err, false, q8)) throw std::runtime_error(err);
            const char* form = !q8 ? "FP16" : d.w8a8() ? "W8A8" : "Q8_0";
            std::printf(" the block in %s\n", form);
            d.keep_block0(true);
            d.set_context(0, fs.data(), kRows);
            d.set_context(1, fn.data(), kNegRows);
            d.set_skip(1, {0});
            float *x = nullptr, *out = nullptr;
            if (!strata::gpu::alloc_device(&x, fl.size() * 4) || !strata::gpu::alloc_device(&out, fl.size() * 8))
                throw std::runtime_error("GPU memory");
            strata::gpu::copy(x, fl.data(), fl.size() * 4);
            d.forward(x, kTime, kH, kW, 2, out);
            strata::gpu::stream_sync(s);
            // FP16 products: within 3e-3 of the largest value (about 3.5e-4 on the B70 and the RTX 4070); the 8-bit
            // block's 5e-3 (8.4e-4).  The residual stream is BF16: the rows read from it (1, 2)
            // take two of its roundings more, 2^-7 of the largest (6.7e-3 and 8.5e-3 on the RTX 4070, 2026-11-06),
            // and the velocity, the head over them, 5e-3 in either form (3.0e-3 in FP16)
            const double tol = q8 ? 5e-3 : 3e-3, bf = 0x1p-7;
            double e = rel(fetch(d.text_rows(0), (size_t) (kRows * kC)), txt);
            check(e <= 3e-3 + bf, "1. the text embedder's rows (FP16)", e, 3e-3 + bf);
            e = rel(fetch(d.block0(), (size_t) ((kHW + kRows) * kC)), x0);
            check(e <= tol + bf, "2. the block's output over the image's and the text's rows", e, tol + bf);
            const auto o = fetch(out, fl.size() * 2);
            e = rel(std::vector<float>(o.begin(), o.begin() + kHW * kLat), v0);
            check(e <= 5e-3, "3. the velocity", e, 5e-3);
            e = rel(std::vector<float>(o.begin() + kHW * kLat, o.end()), v1);
            check(e <= 3e-3, "3. a second context's, skipping the block", e, 3e-3);
            float *dc = nullptr, *dd = nullptr;
            if (!strata::gpu::alloc_device(&dc, fc.size() * 4) || !strata::gpu::alloc_device(&dd, fd.size() * 4))
                throw std::runtime_error("GPU memory");
            strata::gpu::copy(dc, fc.data(), fc.size() * 4);
            strata::gpu::copy(dd, fd.data(), fd.size() * 4);
            d.set_control(ctl_path.string());
            d.set_control_input(dc, dd, 1.f, 1.f);
            d.forward(x, kTime, kH, kW, 1, out);
            strata::gpu::stream_sync(s);
            e = rel(fetch(out, (size_t) (kHW * kLat)), vc);
            check(e <= 5e-3, "4. the velocity with the color and depth control's input projection", e, 5e-3);
            d.set_control_input(nullptr, nullptr, 1.f, 1.f);
            d.set_control("");
            strata::gpu::free(dc);
            strata::gpu::free(dd);
            strata::gpu::free(x);
            strata::gpu::free(out);
        }
    } catch (const std::exception& e) {
        std::printf("  %s\n", e.what());
        ++g_fail;
    }
    std::filesystem::remove(path);
    std::filesystem::remove(ctl_path);
    std::printf("moss_dit_parity: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
