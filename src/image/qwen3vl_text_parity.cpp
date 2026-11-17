// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qwen3vl_text_parity.cpp - the image models' text encoder (TextEncoder through PlainModel) on a synthetic
// Qwen3-VL (qwen3vl: two attention blocks 256 wide, heads of 128 rotated whole with the interleaved M-RoPE's sections
// 24/20/20, BF16 projections) against the host's FP64 of the same model, a picture among the prompt's tokens:
//
//   1. every row of the last block's output before the final norm (Qwen-Image's context), the picture's rows at their
//      M-RoPE positions (t = p, h = p + y, w = p + x; the text after at p + max(nx, ny)) and the DeepStack rows added
//      after the first blocks: each row's cosine to the host's at least 0.999 (FP16 products);
//   2. the check sees what it guards: the host's rows with the picture at text positions (no M-RoPE), or without the
//      DeepStack rows, are further from the engine's than 1.
//
// STRATA_PLAIN_DENSE_MMQ=0: every dense product through the FP16 path (no int8 rows chosen by their timings).
#include "strata/core/runtime.hpp"
#include "strata/image/text_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-90s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

constexpr int64_t kEmbd = 256, kFf = 512, kVocab = 512, kHead = 4, kKv = 2, kHd = 128, kLayers = 2;
constexpr int kSecH = 20, kSecW = 20;   // rope.dimension_sections 24/20/20/0: the time's 24 are the rest
constexpr double kBase = 5e6, kEps = 1e-6;
constexpr uint32_t kF32 = 0, kBF16 = 30;

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
uint16_t bf16(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;   // ne0 (the input) first
    uint32_t type = kF32;
    std::vector<uint8_t> data;
    std::vector<double> v;         // the values as the host reads them
};

Tensor make(const std::string& name, std::vector<uint64_t> shape, uint32_t type, std::mt19937& rng, float sd,
            float mean = 0.0f) {
    Tensor t{name, std::move(shape), type, {}, {}};
    uint64_t n = 1;
    for (uint64_t d : t.shape) n *= d;
    std::normal_distribution<float> nd(mean, sd);
    for (uint64_t i = 0; i < n; ++i) {
        const float x = nd(rng);
        if (type == kF32) {
            put<float>(t.data, x);
            t.v.push_back(x);
        } else {
            const uint16_t h = bf16(x);
            put<uint16_t>(t.data, h);
            t.v.push_back(from_bf16(h));
        }
    }
    return t;
}

struct Model {
    std::vector<Tensor> ts;
    const Tensor& get(const std::string& n) const {
        for (const Tensor& t : ts)
            if (t.name == n) return t;
        throw std::logic_error(n);
    }
};

std::filesystem::path write_model(Model& m) {
    std::mt19937 rng(20261110);  // NOLINT(bugprone-random-generator-seed): the same model each run.
    const float w_sd = 1.0f / std::sqrt((float) kEmbd);
    auto& ts = m.ts;
    ts.push_back(make("token_embd.weight", {kEmbd, kVocab}, kBF16, rng, 1.0f));
    ts.push_back(make("output_norm.weight", {kEmbd}, kF32, rng, 0.1f, 1.0f));
    for (int64_t l = 0; l < kLayers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        ts.push_back(make(p + "attn_norm.weight", {kEmbd}, kF32, rng, 0.1f, 1.0f));
        ts.push_back(make(p + "ffn_norm.weight", {kEmbd}, kF32, rng, 0.1f, 1.0f));
        ts.push_back(make(p + "attn_q.weight", {kEmbd, kHead * kHd}, kBF16, rng, w_sd));
        ts.push_back(make(p + "attn_k.weight", {kEmbd, kKv * kHd}, kBF16, rng, w_sd));
        ts.push_back(make(p + "attn_v.weight", {kEmbd, kKv * kHd}, kBF16, rng, w_sd));
        ts.push_back(make(p + "attn_output.weight", {kHead * kHd, kEmbd}, kBF16, rng, 1.0f / 32.0f));
        ts.push_back(make(p + "attn_q_norm.weight", {kHd}, kF32, rng, 0.1f, 1.0f));
        ts.push_back(make(p + "attn_k_norm.weight", {kHd}, kF32, rng, 0.1f, 1.0f));
        ts.push_back(make(p + "ffn_gate.weight", {kEmbd, kFf}, kBF16, rng, w_sd));
        ts.push_back(make(p + "ffn_up.weight", {kEmbd, kFf}, kBF16, rng, w_sd));
        ts.push_back(make(p + "ffn_down.weight", {kFf, kEmbd}, kBF16, rng, 1.0f / std::sqrt((float) kFf)));
    }
    const std::string a = "qwen3vl.";
    std::vector<uint8_t> kv;
    uint64_t n_kv = 0;
    auto u32 = [&](const std::string& k, uint32_t v) {
        put_str(kv, k); put<uint32_t>(kv, 4); put<uint32_t>(kv, v); ++n_kv;
    };
    auto f32 = [&](const std::string& k, float v) {
        put_str(kv, k); put<uint32_t>(kv, 6); put<float>(kv, v); ++n_kv;
    };
    put_str(kv, "general.architecture"); put<uint32_t>(kv, 8); put_str(kv, "qwen3vl"); ++n_kv;
    u32(a + "block_count", kLayers);
    u32(a + "context_length", 4096);
    u32(a + "embedding_length", kEmbd);
    u32(a + "feed_forward_length", kFf);
    u32(a + "attention.head_count", kHead);
    u32(a + "attention.head_count_kv", kKv);
    u32(a + "attention.key_length", kHd);
    u32(a + "attention.value_length", kHd);
    f32(a + "attention.layer_norm_rms_epsilon", (float) kEps);
    u32(a + "rope.dimension_count", kHd);
    put_str(kv, a + "rope.dimension_sections"); put<uint32_t>(kv, 9); put<uint32_t>(kv, 4); put<uint64_t>(kv, 4);
    for (uint32_t x : {24u, (uint32_t) kSecH, (uint32_t) kSecW, 0u}) put<uint32_t>(kv, x);
    ++n_kv;
    f32(a + "rope.freq_base", (float) kBase);

    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);   // "GGUF"
    put<uint32_t>(b, 3);
    put<uint64_t>(b, ts.size());
    put<uint64_t>(b, n_kv);
    b.insert(b.end(), kv.begin(), kv.end());
    uint64_t off = 0;
    for (const Tensor& t : ts) {
        put_str(b, t.name);
        put<uint32_t>(b, (uint32_t) t.shape.size());
        for (uint64_t d : t.shape) put<uint64_t>(b, d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, off);
        off += (t.data.size() + 31) / 32 * 32;
    }
    b.resize((b.size() + 31) / 32 * 32, 0);
    for (const Tensor& t : ts) {
        b.insert(b.end(), t.data.begin(), t.data.end());
        b.resize((b.size() + 31) / 32 * 32, 0);
    }
    const auto path = std::filesystem::temp_directory_path() / "xestrata_qwen3vl_text_parity.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize) b.size());
    return path;
}

using Rows = std::vector<std::vector<double>>;

// y[r] = W x[r] with W [n_out][n_in]
Rows matmul(const Rows& x, const std::vector<double>& w, int64_t n_out, int64_t n_in) {
    Rows y(x.size(), std::vector<double>((size_t) n_out));
    for (size_t r = 0; r < x.size(); ++r)
        for (int64_t o = 0; o < n_out; ++o) {
            double s = 0;
            for (int64_t i = 0; i < n_in; ++i) s += w[(size_t) (o * n_in + i)] * x[r][(size_t) i];
            y[r][(size_t) o] = s;
        }
    return y;
}
void rms(std::vector<double>& x, const double* w, int64_t off, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; ++i) s += x[(size_t) (off + i)] * x[(size_t) (off + i)];
    const double k = 1.0 / std::sqrt(s / (double) n + kEps);
    for (int64_t i = 0; i < n; ++i) x[(size_t) (off + i)] *= k * w[i];
}

// The host's last block output (before the final norm): `pos3` [T][3] the rows' time, height and width positions (all
// three the same: text positions), `deep` the DeepStack rows added to rows at..at+deep_rows after each first block.
Rows host(const Model& m, const Rows& emb, const std::vector<int64_t>& pos3, const std::vector<std::vector<float>>& deep,
          int64_t at, int64_t deep_rows) {
    const int64_t T = (int64_t) emb.size();
    Rows x = emb;
    for (int64_t l = 0; l < kLayers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        Rows xn = x;
        for (auto& r : xn) rms(r, m.get(p + "attn_norm.weight").v.data(), 0, kEmbd);
        Rows q = matmul(xn, m.get(p + "attn_q.weight").v, kHead * kHd, kEmbd);
        Rows k = matmul(xn, m.get(p + "attn_k.weight").v, kKv * kHd, kEmbd);
        const Rows v = matmul(xn, m.get(p + "attn_v.weight").v, kKv * kHd, kEmbd);
        auto rope = [&](std::vector<double>& r, int64_t heads, int64_t t) {   // NEOX pairs (j, j + 64)
            for (int64_t h = 0; h < heads; ++h)
                for (int64_t j = 0; j < kHd / 2; ++j) {
                    const int a = j % 3 == 1 && j < 3 * (int64_t) kSecH ? 1 : j % 3 == 2 && j < 3 * (int64_t) kSecW ? 2 : 0;
                    const double ang = (double) pos3[(size_t) (t * 3 + a)] * std::pow(kBase, -2.0 * (double) j / kHd);
                    double& x0 = r[(size_t) (h * kHd + j)];
                    double& x1 = r[(size_t) (h * kHd + j + kHd / 2)];
                    const double c = std::cos(ang), s = std::sin(ang), a0 = x0, a1 = x1;
                    x0 = a0 * c - a1 * s;
                    x1 = a1 * c + a0 * s;
                }
        };
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t h = 0; h < kHead; ++h) rms(q[(size_t) t], m.get(p + "attn_q_norm.weight").v.data(), h * kHd, kHd);
            for (int64_t h = 0; h < kKv; ++h) rms(k[(size_t) t], m.get(p + "attn_k_norm.weight").v.data(), h * kHd, kHd);
            rope(q[(size_t) t], kHead, t);
            rope(k[(size_t) t], kKv, t);
        }
        Rows att(x.size(), std::vector<double>((size_t) (kHead * kHd)));
        for (int64_t t = 0; t < T; ++t)
            for (int64_t h = 0; h < kHead; ++h) {
                const int64_t g = h / (kHead / kKv);
                std::vector<double> s((size_t) (t + 1));
                double mx = -1e300;
                for (int64_t u = 0; u <= t; ++u) {
                    double d = 0;
                    for (int64_t i = 0; i < kHd; ++i)
                        d += q[(size_t) t][(size_t) (h * kHd + i)] * k[(size_t) u][(size_t) (g * kHd + i)];
                    s[(size_t) u] = d / std::sqrt((double) kHd);
                    mx = std::max(mx, s[(size_t) u]);
                }
                double z = 0;
                for (double& e : s) z += (e = std::exp(e - mx));
                for (int64_t u = 0; u <= t; ++u)
                    for (int64_t i = 0; i < kHd; ++i)
                        att[(size_t) t][(size_t) (h * kHd + i)] += s[(size_t) u] / z * v[(size_t) u][(size_t) (g * kHd + i)];
            }
        const Rows o = matmul(att, m.get(p + "attn_output.weight").v, kEmbd, kHead * kHd);
        for (int64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < kEmbd; ++i) x[(size_t) t][(size_t) i] += o[(size_t) t][(size_t) i];
        xn = x;
        for (auto& r : xn) rms(r, m.get(p + "ffn_norm.weight").v.data(), 0, kEmbd);
        Rows gt = matmul(xn, m.get(p + "ffn_gate.weight").v, kFf, kEmbd);
        const Rows up = matmul(xn, m.get(p + "ffn_up.weight").v, kFf, kEmbd);
        for (size_t t = 0; t < gt.size(); ++t)
            for (int64_t i = 0; i < kFf; ++i) {
                const double gv = gt[t][(size_t) i];
                gt[t][(size_t) i] = gv / (1.0 + std::exp(-gv)) * up[t][(size_t) i];
            }
        const Rows d = matmul(gt, m.get(p + "ffn_down.weight").v, kEmbd, kFf);
        for (int64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < kEmbd; ++i) x[(size_t) t][(size_t) i] += d[(size_t) t][(size_t) i];
        if (l < (int64_t) deep.size())
            for (int64_t r = 0; r < deep_rows; ++r)
                for (int64_t i = 0; i < kEmbd; ++i)
                    x[(size_t) (at + r)][(size_t) i] += deep[(size_t) l][(size_t) (r * kEmbd + i)];
    }
    return x;
}

// the least cosine of a row of the engine's to the host's
double least_cos(const std::vector<float>& got, const Rows& want) {
    double worst = 1;
    for (size_t t = 0; t < want.size(); ++t) {
        double ab = 0, aa = 0, bb = 0;
        for (int64_t i = 0; i < kEmbd; ++i) {
            const double a = got[t * kEmbd + (size_t) i], b = want[t][(size_t) i];
            ab += a * b;
            aa += a * a;
            bb += b * b;
        }
        worst = std::min(worst, ab / std::sqrt(aa * bb));
    }
    return worst;
}

}  // namespace

int main() {
#if defined(_WIN32)
    _putenv_s("XESTRATA_PLAIN_DENSE_MMQ", "0");
#else
    setenv("XESTRATA_PLAIN_DENSE_MMQ", "0", 1);
#endif
    try {
        strata::core::Runtime::get();
        Model m;
        const auto path = write_model(m);
        // the prompt: 6 text tokens, a picture of 4 x 3 merged patches, 6 text tokens
        constexpr int64_t kAt = 6, kNx = 4, kNy = 3, kPic = kNx * kNy, kT = kAt + kPic + 6;
        std::mt19937 rng(7);  // NOLINT(bugprone-random-generator-seed): the same prompt each run.
        std::uniform_int_distribution<int32_t> tok(0, kVocab - 1);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<int32_t> ids((size_t) kT);
        for (auto& i : ids) i = tok(rng);
        std::vector<float> pic((size_t) (kPic * kEmbd));
        for (auto& v : pic) v = nd(rng);
        std::vector<std::vector<float>> deep(2, std::vector<float>((size_t) (kPic * kEmbd)));
        for (auto& d : deep)
            for (auto& v : d) v = 0.5f * nd(rng);

        // the host's embeddings (the picture's rows in place of its tokens') and positions (mtmd's)
        const Tensor& te = m.get("token_embd.weight");
        Rows emb((size_t) kT, std::vector<double>((size_t) kEmbd));
        for (int64_t t = 0; t < kT; ++t)
            for (int64_t i = 0; i < kEmbd; ++i)
                emb[(size_t) t][(size_t) i] = t >= kAt && t < kAt + kPic ? (double) pic[(size_t) ((t - kAt) * kEmbd + i)]
                                                                       : te.v[(size_t) (ids[(size_t) t] * kEmbd + i)];
        std::vector<int64_t> grid((size_t) (kT * 3)), flat((size_t) (kT * 3));
        for (int64_t t = 0, p = 0; t < kT;) {
            if (t == kAt) {
                for (int64_t j = 0; j < kPic; ++j) {
                    grid[(size_t) ((t + j) * 3)] = p;
                    grid[(size_t) ((t + j) * 3 + 1)] = p + j / kNx;
                    grid[(size_t) ((t + j) * 3 + 2)] = p + j % kNx;
                }
                t += kPic;
                p += std::max(kNx, kNy);
                continue;
            }
            grid[(size_t) (t * 3)] = grid[(size_t) (t * 3 + 1)] = grid[(size_t) (t * 3 + 2)] = p++;
            ++t;
        }
        for (int64_t t = 0; t < kT; ++t) flat[(size_t) (t * 3)] = flat[(size_t) (t * 3 + 1)] = flat[(size_t) (t * 3 + 2)] = t;
        const Rows want = host(m, emb, grid, deep, kAt, kPic);
        const Rows no_mrope = host(m, emb, flat, deep, kAt, kPic);
        const Rows no_deep = host(m, emb, grid, {}, kAt, kPic);

        strata::image::TextEncoder enc;
        std::string err;
        if (!enc.open(path.string(), 64, err)) throw std::runtime_error(err);
        enc.set_final_norm(false);   // Qwen-Image's context
        strata::image::TextEncoder::Picture p;
        p.rows = pic.data();
        p.nx = kNx;
        p.ny = kNy;
        p.index = kAt;
        for (const auto& d : deep) p.deepstack.push_back(d.data());
        const std::vector<float> got = enc.encode(ids, err, {p});
        if (got.size() != (size_t) (kT * kEmbd)) throw std::runtime_error("encode: " + err);
        enc.close();
        std::filesystem::remove(path);

        const double c = least_cos(got, want), c_flat = least_cos(got, no_mrope), c_deep = least_cos(got, no_deep);
        std::printf("least row cosines: the host's %.6f, without M-RoPE %.6f, without DeepStack %.6f\n", c, c_flat,
                    c_deep);
        check(c >= 0.999, "1. the picture's M-RoPE positions and DeepStack rows: every row the host's (cos 0.999)");
        check(c_flat < c && 1.0 - c_flat > 10.0 * (1.0 - c), "2. the host's rows at text positions are further");
        check(c_deep < c && 1.0 - c_deep > 10.0 * (1.0 - c), "2. the host's rows without DeepStack are further");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwen3vl_text_parity: %s\n", e.what());
        return 1;
    }
    std::printf("qwen3vl_text_parity: %d failures\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
