// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_host.cpp - HunyuanImage 3.0 on the host, in FP32, from a quantized file: one decoder layer,
// the pieces around it and a whole denoise step, the check the GPU engine is held to
// (records/2026-10-23-hunyuanimage3-plan/hunyuanimage3_ref.py is the same layer in NumPy, and that folder's
// hy3_torch_ref.py with hy3_conform.py hold both to Tencent's own modules).
//
// It is not the engine's path: it exists so that the layer's arithmetic (the per-KV-head qkv interleave, the qk-norm,
// the 2D RoPE, the block's mask, the routed experts' gate and SwiGLU order) can be compared with an independent
// implementation before any of it is written as a GPU kernel.  The positions are the caller's (text without them:
// y = x = the token's index, a causal mask).
#include "strata/image/hunyuanimage3_host.hpp"

#include "strata/kernels/cpu/native_expert.hpp"

#include <chrono>
#include <functional>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace strata::image {

namespace {

constexpr int64_t kHidden = 4096, kHeads = 32, kKvHeads = 8, kHeadDim = 128, kExperts = 64, kTopk = 8;
constexpr int64_t kRot = 256;
constexpr float kEps = 1e-5f;
// The model's special ids are the header's (`kHunyuanImage3*`), the same numbers the tokenizer's added_tokens give
// (records/ CALIB-2026-10-23.md).  `<timestep>`, `<guidance>` and `<timestep_r>` are the meta tokens whose rows a
// step replaces with its embedders' outputs (Instruct-Distil: cfg_distilled and use_meanflow, so all three; a Base
// model has only the timestep's).

// The save's rotation: kron(H4, H4, H4, H4) / sqrt(256), applied to every group of 256 values of a row.
const std::vector<float>& hadamard256() {
    static const std::vector<float> h = [] {
        static const float h4[16] = {1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1};
        std::vector<float> m = {1.0f};
        for (int64_t size = 1; size < kRot; size *= 4) {
            std::vector<float> next((size_t) (size * 4) * (size_t) (size * 4));
            for (int64_t r = 0; r < size * 4; ++r)
                for (int64_t c = 0; c < size * 4; ++c)
                    next[(size_t) r * (size_t) (size * 4) + (size_t) c] =
                        m[(size_t) (r / 4) * (size_t) size + (size_t) (c / 4)] * h4[(r % 4) * 4 + c % 4];
            m = std::move(next);
        }
        const float s = 1.0f / std::sqrt((float) kRot);
        for (float& v : m) v *= s;
        return m;
    }();
    return h;
}

void rotate_rows(float* x, int64_t rows, int64_t k) {
    const std::vector<float>& h = hadamard256();
    std::vector<float> out((size_t) kRot);
    for (int64_t r = 0; r < rows; ++r) {
        float* row = x + r * k;
        for (int64_t g = 0; g < k; g += kRot) {
            for (int64_t i = 0; i < kRot; ++i) {
                float sum = 0;
                const float* hrow = &h[(size_t) i * kRot];
                for (int64_t j = 0; j < kRot; ++j) sum += hrow[j] * row[g + j];
                out[(size_t) i] = sum;
            }
            std::memcpy(row + g, out.data(), (size_t) kRot * sizeof(float));
        }
    }
}

// [T][K] x [N][K]^T -> [T][N]
void gemm(const float* x, int64_t t, const float* w, int64_t n, int64_t k, float* y) {
    for (int64_t i = 0; i < t; ++i) {
        const float* xi = x + i * k;
        for (int64_t j = 0; j < n; ++j) {
            const float* wj = w + j * k;
            float sum = 0;
            for (int64_t c = 0; c < k; ++c) sum += xi[c] * wj[c];
            y[i * n + j] = sum;
        }
    }
}

void rmsnorm(const float* x, int64_t rows, int64_t k, const std::vector<float>& w, float* out) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * k;
        double v = 0;
        for (int64_t c = 0; c < k; ++c) v += (double) xr[c] * xr[c];
        const float inv = 1.0f / std::sqrt((float) (v / (double) k) + kEps);
        for (int64_t c = 0; c < k; ++c) out[r * k + c] = w[(size_t) c] * xr[c] * inv;
    }
}

float silu(float v) { return v / (1.0f + std::exp(-v)); }

// The fp32 image of a quantized matrix's expert `e`, or of the whole matrix when it has one.
bool dequant_rows(const HunyuanImage3File& f, const HunyuanImage3Matrix& m, int64_t e, int64_t r0, int64_t r1, std::vector<float>& out,
                  std::string& err) {
    out.assign((size_t) ((r1 - r0) * m.k), 0.0f);
    try {
        f.rows(m, e, r0, r1, out.data(), HunyuanImage3File::low_first());
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
    return true;
}

bool bf16_vector(const HunyuanImage3File& f, const std::string& name, std::vector<float>& out, int64_t want,
                 std::string& err) {
    const SafeTensor* t = f.tensors().find(name);
    if (t == nullptr) {
        err = name + ": no such tensor";
        return false;
    }
    if (t->dtype != "BF16" || t->elements() != want) {
        err = name + ": " + t->dtype + ", " + std::to_string(t->elements()) + " values (wanted " +
              std::to_string(want) + ")";
        return false;
    }
    // the bytes in the mapping: HunyuanImage3File::bf16 copies them through the same mapping its weights are read from
    out.assign((size_t) want, 0.0f);
    f.bf16(name, out.data());
    return true;
}

}  // namespace

bool hunyuanimage3_layer_host(const HunyuanImage3File& f, int64_t layer, const float* x, int64_t t, float* out,
                              const int32_t* pos_y, const int32_t* pos_x, int64_t n_img, std::string& err,
                              HunyuanImage3HostStages* stages) {
    if (t <= 0) {
        err = "hunyuanimage3_layer_host: no tokens";
        return false;
    }
    // Text-only when the caller gives no positions; the image block, when there is one, is the LAST n_img tokens
    // (the reference's sequence: its text first, then the <img> run, and nothing that the image must see after).
    const std::vector<int32_t> idx_y = [&] {
        std::vector<int32_t> v((size_t) t);
        for (int64_t i = 0; i < t; ++i) v[(size_t) i] = (int32_t) i;
        return v;
    }();
    const int32_t* py = pos_y != nullptr ? pos_y : idx_y.data();
    const int32_t* px = pos_x != nullptr ? pos_x : idx_y.data();
    if (n_img < 0 || n_img > t) {
        err = "hunyuanimage3_layer_host: the image block does not fit the sequence";
        return false;
    }
    const int64_t img0 = t - n_img;
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    auto matrix = [&](const std::string& suffix, HunyuanImage3Matrix& m) {
        return f.matrix(p + suffix + ".weight", m, err);
    };
    HunyuanImage3Matrix qkv, o, sgu, sdn, experts_gu, experts_dn;
    if (!matrix("self_attn.qkv_proj", qkv) || !matrix("self_attn.o_proj", o) ||
        !matrix("mlp.shared_mlp.gate_and_up_proj", sgu) || !matrix("mlp.shared_mlp.down_proj", sdn) ||
        !matrix("mlp.experts_gate_up_proj", experts_gu) || !matrix("mlp.experts_down_proj", experts_dn))
        return false;
    std::vector<float> norm_in, norm_post, qln, kln, gate_w;
    if (!bf16_vector(f, p + "input_layernorm.weight", norm_in, kHidden, err) ||
        !bf16_vector(f, p + "post_attention_layernorm.weight", norm_post, kHidden, err) ||
        !bf16_vector(f, p + "self_attn.query_layernorm.weight", qln, kHeadDim, err) ||
        !bf16_vector(f, p + "self_attn.key_layernorm.weight", kln, kHeadDim, err) ||
        !bf16_vector(f, p + "mlp.gate.wg.weight", gate_w, kExperts * kHidden, err))
        return false;

    const int64_t n = t * kHidden;
    std::vector<float> xn((size_t) n), qkv_out((size_t) (t * 6 * kKvHeads * kHeadDim)), attn((size_t) n);
    std::copy(x, x + n, out);
    // `probe stages` reads the same points of the walk; a snapshot before the in-place rotation overwrites it.
    auto take = [&](std::vector<float> HunyuanImage3HostStages::*field, const float* src, size_t count) {
        if (stages != nullptr) (stages->*field).assign(src, src + count);
    };

    // ---- attention
    rmsnorm(x, t, kHidden, norm_in, xn.data());
    take(&HunyuanImage3HostStages::xn32, xn.data(), (size_t) n);
    rotate_rows(xn.data(), t, kHidden);
    take(&HunyuanImage3HostStages::xr16, xn.data(), (size_t) n);
    {
        std::vector<float> w;
        if (!dequant_rows(f, qkv, 0, 0, qkv.n, w, err)) return false;
        gemm(xn.data(), t, w.data(), qkv.n, qkv.k, qkv_out.data());
        take(&HunyuanImage3HostStages::qkv32, qkv_out.data(), qkv_out.size());
    }
    // the per-KV-head order [q0..q3, k, v]; the q heads' RMSNorm and the 2D RoPE, text positions y = x = the index
    std::vector<float> q((size_t) (t * kHeads * kHeadDim)), k((size_t) (t * kKvHeads * kHeadDim)),
        v((size_t) (t * kKvHeads * kHeadDim));
    for (int64_t i = 0; i < t; ++i)
        for (int64_t h = 0; h < kKvHeads; ++h) {
            const float* base = qkv_out.data() + (i * 6 * kKvHeads + h * 6) * kHeadDim;
            for (int64_t g = 0; g < 4; ++g)
                std::memcpy(q.data() + ((i * kHeads + h * 4 + g) * kHeadDim), base + g * kHeadDim,
                            kHeadDim * sizeof(float));
            std::memcpy(k.data() + ((i * kKvHeads + h) * kHeadDim), base + 4 * kHeadDim, kHeadDim * sizeof(float));
            std::memcpy(v.data() + ((i * kKvHeads + h) * kHeadDim), base + 5 * kHeadDim, kHeadDim * sizeof(float));
        }
    // The reference's order: the 2D RoPE first, then the qk-norm.  Column j (0 .. head_dim/2 - 1) and its partner
    // j + head_dim/2 turn by the SAME angle (build_2d_rope computes head_dim/2 angles and `idx.repeat(1, 2)` TILES
    // them; it is not numpy's pairwise repeat, which is what this file got wrong until 2026-10-24 - and the
    // record's hy3_ref.py
    // with it, the two agreeing and neither matching Tencent's code).  The angle is pos[j % 2] * base^(-2 j / 128)
    // with pos[0] = y and pos[1] = x, from theta = base^(-arange(0, n_elem, 2) / n_elem) reshaped [n_elem/4][2].
    // Text's position is y = x = the token's index; an image block has the grid (see hunyuanimage3_positions).
    std::vector<float> cosp((size_t) t * kHeadDim), sinp((size_t) t * kHeadDim);
    for (int64_t i = 0; i < t; ++i) {
        for (int64_t j = 0; j < kHeadDim / 2; ++j) {
            const double pos = (j % 2) ? (double) px[i] : (double) py[i];
            const double ang = pos * std::pow(10000.0, -2.0 * (double) j / (double) kHeadDim);
            const float c = (float) std::cos(ang), s = (float) std::sin(ang);
            cosp[(size_t) (i * kHeadDim + j)] = c;
            sinp[(size_t) (i * kHeadDim + j)] = s;
            cosp[(size_t) (i * kHeadDim + j + kHeadDim / 2)] = c;
            sinp[(size_t) (i * kHeadDim + j + kHeadDim / 2)] = s;
        }
    }
    auto rope = [&](std::vector<float>& vec, int64_t heads) {
        for (int64_t i = 0; i < t; ++i)
            for (int64_t h = 0; h < heads; ++h) {
                float* d = vec.data() + (i * heads + h) * kHeadDim;
                const float* c = cosp.data() + i * kHeadDim;
                const float* s = sinp.data() + i * kHeadDim;
                for (int64_t j = 0; j < kHeadDim / 2; ++j) {
                    const float a = d[j], b = d[j + kHeadDim / 2];
                    d[j] = a * c[j] - b * s[j];
                    d[j + kHeadDim / 2] = b * c[j] + a * s[j];
                }
            }
    };
    rope(q, kHeads);
    rope(k, kKvHeads);
    std::vector<float> qn(q.size()), kn(k.size());
    rmsnorm(q.data(), t * kHeads, kHeadDim, qln, qn.data());
    rmsnorm(k.data(), t * kKvHeads, kHeadDim, kln, kn.data());
    if (stages != nullptr) {
        // the device's q, k and v hold each KV head repeated `groups` times, so the host's are expanded the same way
        take(&HunyuanImage3HostStages::q16, qn.data(), qn.size());
        stages->k16.assign((size_t) t * kHeads * kHeadDim, 0.0f);
        stages->v16.assign((size_t) t * kHeads * kHeadDim, 0.0f);
        for (int64_t i = 0; i < t; ++i)
            for (int64_t h = 0; h < kHeads; ++h) {
                std::copy(kn.data() + (i * kKvHeads + h / 4) * kHeadDim,
                          kn.data() + (i * kKvHeads + h / 4) * kHeadDim + kHeadDim,
                          stages->k16.begin() + (i * kHeads + h) * kHeadDim);
                std::copy(v.data() + (i * kKvHeads + h / 4) * kHeadDim,
                          v.data() + (i * kKvHeads + h / 4) * kHeadDim + kHeadDim,
                          stages->v16.begin() + (i * kHeads + h) * kHeadDim);
            }
    }
    // attention: 32 heads against the KV heads they repeat.  The mask is the reference's
    // `_prepare_attention_mask_for_generation`: causal (j <= i) for every row, with the image block filled True -
    // so a row of the block sees all of the block (and all the text before it), and a text row sees its prefix.
    std::vector<float> probs((size_t) t);
    auto visible = [&](int64_t i, int64_t j) { return j <= i || (i >= img0 && j >= img0); };
    for (int64_t h = 0; h < kHeads; ++h) {
        const int64_t kh = h / 4;
        for (int64_t i = 0; i < t; ++i) {
            const float* qi = qn.data() + (i * kHeads + h) * kHeadDim;
            float best = -1e30f;
            for (int64_t j = 0; j < t; ++j) {
                if (!visible(i, j)) continue;
                const float* kj = kn.data() + (j * kKvHeads + kh) * kHeadDim;
                float dot = 0;
                for (int64_t c = 0; c < kHeadDim; ++c) dot += qi[c] * kj[c];
                probs[(size_t) j] = dot / std::sqrt((float) kHeadDim);
                best = std::max(best, probs[(size_t) j]);
            }
            float den = 0;
            for (int64_t j = 0; j < t; ++j) {
                if (!visible(i, j)) continue;
                probs[(size_t) j] = std::exp(probs[(size_t) j] - best);
                den += probs[(size_t) j];
            }
            float* acc = attn.data() + (i * kHeads + h) * kHeadDim;
            std::fill(acc, acc + kHeadDim, 0.0f);
            for (int64_t j = 0; j < t; ++j) {
                if (!visible(i, j)) continue;
                const float pj = probs[(size_t) j] / den;
                const float* vj = v.data() + (j * kKvHeads + kh) * kHeadDim;
                for (int64_t c = 0; c < kHeadDim; ++c) acc[c] += pj * vj[c];
            }
        }
    }
    take(&HunyuanImage3HostStages::a16, attn.data(), (size_t) n);
    rotate_rows(attn.data(), t, kHidden);
    {
        std::vector<float> w, o_out((size_t) n);
        if (!dequant_rows(f, o, 0, 0, o.n, w, err)) return false;
        gemm(attn.data(), t, w.data(), o.n, o.k, o_out.data());
        take(&HunyuanImage3HostStages::o32, o_out.data(), (size_t) n);
        for (int64_t i = 0; i < n; ++i) out[i] += o_out[(size_t) i];
    }
    // ---- the shared expert and the routed ones
    rmsnorm(out, t, kHidden, norm_post, xn.data());
    std::vector<float> moe((size_t) n, 0.0f), xr((size_t) n);
    std::copy(xn.begin(), xn.end(), xr.begin());
    take(&HunyuanImage3HostStages::xn232, xn.data(), (size_t) n);
    rotate_rows(xr.data(), t, kHidden);
    take(&HunyuanImage3HostStages::xr216, xr.data(), (size_t) n);
    {
        std::vector<float> gu_w, gu((size_t) (t * sgu.n));
        if (!dequant_rows(f, sgu, 0, 0, sgu.n, gu_w, err)) return false;
        gemm(xr.data(), t, gu_w.data(), sgu.n, sgu.k, gu.data());
        std::vector<float> act((size_t) (t * (sgu.n / 2)));
        for (int64_t i = 0; i < t; ++i)
            for (int64_t c = 0; c < sgu.n / 2; ++c)
                act[(size_t) (i * (sgu.n / 2) + c)] = gu[(size_t) (i * sgu.n + c)] *
                                                       silu(gu[(size_t) (i * sgu.n + sgu.n / 2 + c)]);
        rotate_rows(act.data(), t, sgu.n / 2);
        std::vector<float> dn_w, dn((size_t) n);
        if (!dequant_rows(f, sdn, 0, 0, sdn.n, dn_w, err)) return false;
        gemm(act.data(), t, dn_w.data(), sdn.n, sdn.k, dn.data());
        for (int64_t i = 0; i < n; ++i) moe[(size_t) i] += dn[(size_t) i];
        take(&HunyuanImage3HostStages::act16, act.data(), act.size());
        take(&HunyuanImage3HostStages::shared32, dn.data(), dn.size());
    }
    // the router: fp32, softmax, the top 8, renormalized over them (norm_topk_prob)
    std::vector<float> logits((size_t) (t * kExperts));
    for (int64_t i = 0; i < t; ++i) {
        const float* xi = xn.data() + i * kHidden;
        float best = -1e30f;
        for (int64_t e = 0; e < kExperts; ++e) {
            const float* we = gate_w.data() + e * kHidden;
            float dot = 0;
            for (int64_t c = 0; c < kHidden; ++c) dot += xi[c] * we[c];
            logits[(size_t) (i * kExperts + e)] = dot;
            best = std::max(best, dot);
        }
    }
    take(&HunyuanImage3HostStages::logits, logits.data(), logits.size());
    for (int64_t i = 0; i < t; ++i) {
        const float* li = logits.data() + i * kExperts;
        float best = -1e30f;
        for (int64_t e = 0; e < kExperts; ++e) best = std::max(best, li[e]);
        float den = 0;
        for (int64_t e = 0; e < kExperts; ++e) {
            logits[(size_t) (i * kExperts + e)] = std::exp(logits[(size_t) (i * kExperts + e)] - best);
            den += logits[(size_t) (i * kExperts + e)];
        }
        for (int64_t e = 0; e < kExperts; ++e) logits[(size_t) (i * kExperts + e)] /= den;
    }
    std::vector<int32_t> idx((size_t) (t * kTopk));
    std::vector<float> weights((size_t) (t * kTopk));
    for (int64_t i = 0; i < t; ++i) {
        std::vector<int32_t> order(kExperts);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + kTopk, order.end(), [&](int32_t a, int32_t b) {
            return logits[(size_t) (i * kExperts + a)] > logits[(size_t) (i * kExperts + b)];
        });
        float sum = 0;
        for (int64_t j = 0; j < kTopk; ++j) {
            idx[(size_t) (i * kTopk + j)] = order[(size_t) j];
            weights[(size_t) (i * kTopk + j)] = logits[(size_t) (i * kExperts + order[(size_t) j])];
            sum += weights[(size_t) (i * kTopk + j)];
        }
        for (int64_t j = 0; j < kTopk; ++j) weights[(size_t) (i * kTopk + j)] /= sum;
    }
    for (int64_t e = 0; e < kExperts; ++e) {
        std::vector<int64_t> rows;
        for (int64_t i = 0; i < t; ++i)
            for (int64_t j = 0; j < kTopk; ++j)
                if (idx[(size_t) (i * kTopk + j)] == e) rows.push_back(i);
        if (rows.empty()) continue;
        std::vector<float> ex((size_t) (rows.size() * kHidden));
        for (size_t r = 0; r < rows.size(); ++r)
            std::copy(xn.begin() + rows[r] * kHidden, xn.begin() + (rows[r] + 1) * kHidden,
                      ex.begin() + (int64_t) r * kHidden);
        rotate_rows(ex.data(), (int64_t) rows.size(), kHidden);
        std::vector<float> gu_w, gu((size_t) (rows.size() * experts_gu.n));
        if (!dequant_rows(f, experts_gu, e, 0, experts_gu.n, gu_w, err)) return false;
        gemm(ex.data(), (int64_t) rows.size(), gu_w.data(), experts_gu.n, experts_gu.k, gu.data());
        std::vector<float> act((size_t) (rows.size() * (experts_gu.n / 2)));
        for (size_t r = 0; r < rows.size(); ++r)
            for (int64_t c = 0; c < experts_gu.n / 2; ++c)
                act[(size_t) (r * (experts_gu.n / 2) + c)] =
                    gu[(size_t) (r * experts_gu.n + c)] * silu(gu[(size_t) (r * experts_gu.n + experts_gu.n / 2 + c)]);
        rotate_rows(act.data(), (int64_t) rows.size(), experts_gu.n / 2);
        std::vector<float> dn_w, dn((size_t) (rows.size() * kHidden));
        if (!dequant_rows(f, experts_dn, e, 0, experts_dn.n, dn_w, err)) return false;
        gemm(act.data(), (int64_t) rows.size(), dn_w.data(), experts_dn.n, experts_dn.k, dn.data());
        for (size_t r = 0; r < rows.size(); ++r) {
            float w = 0;
            for (int64_t j = 0; j < kTopk; ++j)
                if (idx[(size_t) (rows[r] * kTopk + j)] == e) w += weights[(size_t) (rows[r] * kTopk + j)];
            for (int64_t c = 0; c < kHidden; ++c) moe[(size_t) (rows[r] * kHidden + c)] += w * dn[(size_t) (r * kHidden + c)];
        }
    }
    for (int64_t i = 0; i < n; ++i) out[i] += moe[(size_t) i];
    return true;
}


// The text-only form: positions y = x = the token's index, a causal mask (no image block).
bool hunyuanimage3_layer_host(const HunyuanImage3File& f, int64_t layer, const float* x, int64_t t, float* out,
                              std::string& err) {
    return hunyuanimage3_layer_host(f, layer, x, t, out, nullptr, nullptr, 0, err);
}

// The reference's sequence ends with `</answer>` after `<eoi>`; none of it can reach the image (a text row's mask
// is causal, and the block is before it), and the reference itself drops it from step 1 on.  So the engine's
// sequence is the ids up to and including the LAST `<img>`.  Returns that length, or -1 with `err` when the ids
// hold no image block; `<boi>` before the block and `<eoi>` after it are what the reference puts there.
int64_t hunyuanimage3_trim(const int32_t* ids, int64_t t, std::string& err) {
    int64_t last = -1, boi = 0;
    for (int64_t i = 0; i < t; ++i) {
        if (ids[i] == kHunyuanImage3Image) last = i;
        else if (ids[i] == kHunyuanImage3Boi) ++boi;
    }
    if (last < 0) {
        err = "hunyuanimage3_trim: no <img> token in the sequence";
        return -1;
    }
    if (boi == 0) {
        err = "hunyuanimage3_trim: the image block has no <boi> before it";
        return -1;
    }
    if (last + 1 < t && ids[last + 1] != kHunyuanImage3Eoi) {
        err = "hunyuanimage3_trim: id " + std::to_string(ids[last + 1]) + " follows the block, not <eoi>";
        return -1;
    }
    return last + 1;
}

bool hunyuanimage3_blocks(const int32_t* ids, int64_t t, std::vector<HunyuanImage3Block>& blocks, std::string& err,
                          bool generated) {
    blocks.clear();
    auto is_size = [](int32_t id) { return id >= kHunyuanImage3ImgSize256 && id < kHunyuanImage3ImgRatio0; };
    auto run = [&](int64_t from) {   // the `<img>` run from `from`
        int64_t n = 0;
        while (from + n < t && ids[from + n] == kHunyuanImage3Image) ++n;
        return n;
    };
    for (int64_t i = 0; i < t; ++i) {
        if (ids[i] != kHunyuanImage3Boi) {
            if (ids[i] == kHunyuanImage3Image) {
                err = "hunyuanimage3_blocks: an <img> token at " + std::to_string(i) + " outside a block";
                return false;
            }
            continue;
        }
        HunyuanImage3Block b;
        b.boi = i;
        int64_t k = i + 1;
        if (k + 1 < t && is_size(ids[k]) && hunyuanimage3_ratio_of(ids[k + 1]) >= 0) {
            b.base_size = hunyuanimage3_img_size_of(ids[k]);
            b.ratio_index = hunyuanimage3_ratio_of(ids[k + 1]);
            k += 2;
        } else {
            err = "hunyuanimage3_blocks: the block at " + std::to_string(i) + " has no <img_size_N> <img_ratio_i>";
            return false;
        }
        if (k < t && ids[k] == kHunyuanImage3Timestep) b.timestep = k++;
        else {
            err = "hunyuanimage3_blocks: the block at " + std::to_string(i) + " has no <timestep>";
            return false;
        }
        if (k + 1 < t && ids[k] == kHunyuanImage3Guidance && ids[k + 1] == kHunyuanImage3TimestepR) {
            // the generated image's block: its tokens end the sequence
            k += 2;
            b.vae_start = k;
            b.vae_n = run(k);
            if (b.vae_n == 0 || b.vae_start + b.vae_n != t) {
                err = "hunyuanimage3_blocks: the generated image's block at " + std::to_string(i) +
                      " does not end the sequence";
                return false;
            }
            b.end = t;
            blocks.push_back(b);
            return true;
        }
        b.reference = true;
        b.vae_start = k;
        b.vae_n = run(k);
        k += b.vae_n;
        if (b.vae_n == 0 || k >= t || ids[k] != kHunyuanImage3JointImgSep) {
            err = "hunyuanimage3_blocks: the reference at " + std::to_string(i) +
                  " has no <img> run followed by <joint_img_sep>";
            return false;
        }
        ++k;
        b.vit_start = k;
        b.vit_n = run(k);
        k += b.vit_n;
        if (b.vit_n == 0 || k >= t || ids[k] != kHunyuanImage3Eoi) {
            err = "hunyuanimage3_blocks: the reference at " + std::to_string(i) + " has no <eoi> after its ViT tokens";
            return false;
        }
        b.end = k + 1;
        blocks.push_back(b);
        i = k;
    }
    if (!generated) return true;
    err = "hunyuanimage3_blocks: no generated image's block at the end of the sequence";
    return false;
}

bool hunyuanimage3_positions_of(int64_t t, const std::vector<HunyuanImage3Block>& blocks, std::vector<int32_t>& y,
                                std::vector<int32_t>& x, std::vector<int32_t>& key_lim, std::string& err) {
    y.assign((size_t) t, 0);
    x.assign((size_t) t, 0);
    key_lim.assign((size_t) t, 0);
    for (int64_t i = 0; i < t; ++i) {
        y[(size_t) i] = x[(size_t) i] = (int32_t) i;
        key_lim[(size_t) i] = (int32_t) (i + 1);
    }
    // the grids, in sequence order: (start, h, w, slots) - a grid's rows past h * w (a ViT block's padding) keep
    // their index, as build_2d_rope's "previous text" range covers them
    struct Grid { int64_t L, h, w, slots; };
    std::vector<Grid> grids;
    for (const HunyuanImage3Block& b : blocks) {
        if (b.vae_h * b.vae_w != b.vae_n) {
            err = "hunyuanimage3_positions_of: a block's " + std::to_string(b.vae_n) + " VAE tokens are not its " +
                  std::to_string(b.vae_h) + " x " + std::to_string(b.vae_w) + " grid";
            return false;
        }
        grids.push_back({b.vae_start, b.vae_h, b.vae_w, b.vae_n});
        if (b.reference) {
            if (b.vit_h * b.vit_w > b.vit_n || b.vit_h <= 0 || b.vit_w <= 0) {
                err = "hunyuanimage3_positions_of: a reference's " + std::to_string(b.vit_h) + " x " +
                      std::to_string(b.vit_w) + " ViT grid does not fit its " + std::to_string(b.vit_n) + " slots";
                return false;
            }
            grids.push_back({b.vit_start, b.vit_h, b.vit_w, b.vit_n});
            for (int64_t i = b.vae_start; i < b.vit_start + b.vit_n; ++i)
                key_lim[(size_t) i] = (int32_t) (b.vit_start + b.vit_n);   // the joint span sees all of itself
        }
    }
    for (const Grid& g : grids) {
        const double beta_y = (double) g.L + ((double) (g.w * g.h) - (double) g.h) / 2.0;
        const double beta_x = (double) g.L + ((double) (g.w * g.h) - (double) g.w) / 2.0;
        for (int64_t r = 0; r < g.h; ++r)
            for (int64_t c = 0; c < g.w; ++c) {
                const int64_t i = g.L + r * g.w + c;
                y[(size_t) i] = (int32_t) std::floor(beta_y + (double) r);
                x[(size_t) i] = (int32_t) std::floor(beta_x + (double) c);
            }
    }
    return true;
}

// The sequence's 2D rope positions: text keeps y = x = its index; the image block at L, h x w tokens, gets
// beta_y = L + (w h - h)/2 and beta_x = L + (w h - w)/2 with the grid (r, c) at (beta_y + r, beta_x + c), each
// truncated to an integer as the reference's `.long()` does.  `ids` must hold the block as `token_h * token_w`
// `<img>` tokens; the block has to be the sequence's tail (post-image text cannot influence the image, so the
// engine drops it - and the reference drops it after step 0 anyway).
bool hunyuanimage3_positions(const int32_t* ids, int64_t t, int64_t token_h, int64_t token_w, std::vector<int32_t>& y,
                             std::vector<int32_t>& x, int64_t& img_start, std::string& err) {
    if (t <= 0 || token_h <= 0 || token_w <= 0) {
        err = "hunyuanimage3_positions: an empty sequence or size";
        return false;
    }
    const int64_t n_img = token_h * token_w;
    if (n_img > t) {
        err = "hunyuanimage3_positions: " + std::to_string(n_img) + " image tokens in a sequence of " +
              std::to_string(t);
        return false;
    }
    img_start = t - n_img;
    for (int64_t i = 0; i < t; ++i) {
        const bool img = ids[i] == kHunyuanImage3Image;
        if (img && i < img_start) {
            err = "hunyuanimage3_positions: an <img> token at " + std::to_string(i) + " before the block at " +
                  std::to_string(img_start) + " (the engine's sequence is text, then the image block)";
            return false;
        }
        if (!img && i >= img_start) {
            err = "hunyuanimage3_positions: the sequence's tail is not " + std::to_string(n_img) + " <img> tokens";
            return false;
        }
    }
    y.assign((size_t) t, 0);
    x.assign((size_t) t, 0);
    for (int64_t i = 0; i < img_start; ++i) {
        y[(size_t) i] = (int32_t) i;
        x[(size_t) i] = (int32_t) i;
    }
    const double beta_y = (double) img_start + ((double) n_img - (double) token_h) / 2.0;
    const double beta_x = (double) img_start + ((double) n_img - (double) token_w) / 2.0;
    for (int64_t r = 0; r < token_h; ++r)
        for (int64_t c = 0; c < token_w; ++c) {
            const int64_t i = img_start + r * token_w + c;
            y[(size_t) i] = (int32_t) std::floor(beta_y + (double) r);
            x[(size_t) i] = (int32_t) std::floor(beta_x + (double) c);
        }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// the pieces around the layers: the ResBlock (GroupNorm(32) + SiLU + conv 3x3, an embedding's scale and shift, the
// skip's 1x1 conv), the TimestepEmbedder, patch_embed and final_layer

namespace {

struct Ctx {
    const HunyuanImage3File& f;
    std::string& err;
    bool bf16(const std::string& name, std::vector<float>& out, int64_t want) {
        const SafeTensor* t = f.tensors().find(name);
        if (t == nullptr) {
            err = name + ": no such tensor";
            return false;
        }
        if (t->dtype != "BF16" || t->elements() != want) {
            err = name + ": " + t->dtype + ", " + std::to_string(t->elements()) + " values (wanted " +
                  std::to_string(want) + ")";
            return false;
        }
        out.assign((size_t) want, 0.0f);
        f.bf16(name, out.data());
        return true;
    }
};

// [oc][h][w] = conv3x3([ic][h][w]) + bias, padding 1
bool conv3x3(Ctx& c, const std::string& prefix, const std::vector<float>& x, int64_t ic, int64_t oc, int64_t h,
             int64_t w, std::vector<float>& y) {
    std::vector<float> wt, bias;
    if (!c.bf16(prefix + ".weight", wt, oc * ic * 9) || !c.bf16(prefix + ".bias", bias, oc)) return false;
    y.assign((size_t) (oc * h * w), 0.0f);
    for (int64_t o = 0; o < oc; ++o)
        for (int64_t yy = 0; yy < h; ++yy)
            for (int64_t xx = 0; xx < w; ++xx) {
                float sum = bias[(size_t) o];
                for (int64_t i = 0; i < ic; ++i)
                    for (int64_t dy = -1; dy <= 1; ++dy)
                        for (int64_t dx = -1; dx <= 1; ++dx) {
                            const int64_t y2 = yy + dy, x2 = xx + dx;
                            if (y2 < 0 || y2 >= h || x2 < 0 || x2 >= w) continue;
                            sum += wt[(size_t) (((o * ic + i) * 3 + (dy + 1)) * 3 + (dx + 1))] *
                                   x[(size_t) ((i * h + y2) * w + x2)];
                        }
                y[(size_t) ((o * h + yy) * w + xx)] = sum;
            }
    return true;
}

// [oc][h][w] = conv1x1([ic][h][w]) + bias
bool conv1x1(Ctx& c, const std::string& prefix, const std::vector<float>& x, int64_t ic, int64_t oc, int64_t h,
             int64_t w, std::vector<float>& y) {
    std::vector<float> wt, bias;
    if (!c.bf16(prefix + ".weight", wt, oc * ic) || !c.bf16(prefix + ".bias", bias, oc)) return false;
    y.assign((size_t) (oc * h * w), 0.0f);
    for (int64_t o = 0; o < oc; ++o)
        for (int64_t yy = 0; yy < h; ++yy)
            for (int64_t xx = 0; xx < w; ++xx) {
                float sum = bias[(size_t) o];
                for (int64_t i = 0; i < ic; ++i) sum += wt[(size_t) (o * ic + i)] * x[(size_t) ((i * h + yy) * w + xx)];
                y[(size_t) ((o * h + yy) * w + xx)] = sum;
            }
    return true;
}

// GroupNorm(32, c), eps 1e-5, affine, in place
bool group_norm(Ctx& c, const std::string& prefix, std::vector<float>& x, int64_t ch, int64_t h, int64_t w) {
    std::vector<float> g, b;
    if (!c.bf16(prefix + ".weight", g, ch) || !c.bf16(prefix + ".bias", b, ch)) return false;
    const int64_t n = h * w, groups = 32, per = ch / groups;
    for (int64_t grp = 0; grp < groups; ++grp) {
        double mean = 0, var = 0;
        for (int64_t i = grp * per; i < (grp + 1) * per; ++i)
            for (int64_t j = 0; j < n; ++j) mean += (double) x[(size_t) (i * n + j)];
        mean /= (double) (per * n);
        for (int64_t i = grp * per; i < (grp + 1) * per; ++i)
            for (int64_t j = 0; j < n; ++j) {
                const double d = (double) x[(size_t) (i * n + j)] - mean;
                var += d * d;
            }
        var /= (double) (per * n);
        const float inv = 1.0f / std::sqrt((float) var + 1e-5f);
        for (int64_t i = grp * per; i < (grp + 1) * per; ++i)
            for (int64_t j = 0; j < n; ++j) {
                float& v = x[(size_t) (i * n + j)];
                v = ((v - (float) mean) * inv) * g[(size_t) i] + b[(size_t) i];
            }
    }
    return true;
}

void silu_all(std::vector<float>& x) {
    for (float& v : x) v = silu(v);
}

// The ResBlock: h = conv3x3(SiLU(GN_in(x))); e = Linear(SiLU(emb)) -> scale, shift; h = conv3x3(SiLU(GN_out(h) *
// (1 + scale) + shift)); out = skip(x) + h, the skip a 1x1 conv unless the channels are equal.
bool res_block(Ctx& c, const std::string& prefix, std::vector<float>& x, int64_t ic, int64_t oc, int64_t h,
               int64_t w, const std::vector<float>& emb) {
    const int64_t n = h * w;
    std::vector<float> t = x;
    if (!group_norm(c, prefix + ".in_layers.0", t, ic, h, w)) return false;
    silu_all(t);
    std::vector<float> mid;
    if (!conv3x3(c, prefix + ".in_layers.2", t, ic, oc, h, w, mid)) return false;
    std::vector<float> eb = emb, ew, ebn;
    silu_all(eb);
    if (!c.bf16(prefix + ".emb_layers.1.weight", ew, 2 * oc * 4096) ||
        !c.bf16(prefix + ".emb_layers.1.bias", ebn, 2 * oc))
        return false;
    std::vector<float> es((size_t) (2 * oc), 0.0f);
    for (int64_t o = 0; o < 2 * oc; ++o) {
        float sum = ebn[(size_t) o];
        for (int64_t i = 0; i < 4096; ++i) sum += ew[(size_t) (o * 4096 + i)] * eb[(size_t) i];
        es[(size_t) o] = sum;
    }
    if (!group_norm(c, prefix + ".out_layers.0", mid, oc, h, w)) return false;
    for (int64_t ch = 0; ch < oc; ++ch) {
        const float scale = es[(size_t) ch], shift = es[(size_t) (oc + ch)];
        for (int64_t i = 0; i < n; ++i)
            mid[(size_t) (ch * n + i)] = mid[(size_t) (ch * n + i)] * (1.0f + scale) + shift;
    }
    silu_all(mid);
    std::vector<float> out;
    if (!conv3x3(c, prefix + ".out_layers.3", mid, oc, oc, h, w, out)) return false;
    std::vector<float> skip;
    if (ic == oc) {
        skip = x;
    } else if (!conv1x1(c, prefix + ".skip_connection", x, ic, oc, h, w, skip)) {
        return false;
    }
    for (size_t i = 0; i < out.size(); ++i) out[i] += skip[i];
    x = std::move(out);
    return true;
}

float gelu(float v) { return 0.5f * v * (1.0f + std::erf(v * 0.70710678f)); }

// The TimestepEmbedder: 256 sinusoidal values (half cos, half sin, max period 10000) through Linear(4096) -> GELU ->
// Linear(4096).
bool time_embed(Ctx& c, const std::string& which, float t, std::vector<float>& out) {
    std::vector<float> w0, b0, w2, b2;
    const std::string p = which + ".mlp.";
    if (!c.bf16(p + "0.weight", w0, (int64_t) 4096 * 256) || !c.bf16(p + "0.bias", b0, 4096) ||
        !c.bf16(p + "2.weight", w2, (int64_t) 4096 * 4096) || !c.bf16(p + "2.bias", b2, 4096))
        return false;
    std::vector<float> f((size_t) 256);
    for (int64_t i = 0; i < 128; ++i) {
        const float freq = std::exp(-std::log(10000.0f) * (float) i / 128.0f);
        f[(size_t) i] = std::cos(t * freq);
        f[(size_t) (128 + i)] = std::sin(t * freq);
    }
    std::vector<float> h1((size_t) 4096);
    for (int64_t o = 0; o < 4096; ++o) {
        float sum = b0[(size_t) o];
        for (int64_t i = 0; i < 256; ++i) sum += w0[(size_t) (o * 256 + i)] * f[(size_t) i];
        h1[(size_t) o] = gelu(sum);
    }
    out.assign((size_t) 4096, 0.0f);
    for (int64_t o = 0; o < 4096; ++o) {
        float sum = b2[(size_t) o];
        for (int64_t i = 0; i < 4096; ++i) sum += w2[(size_t) (o * 4096 + i)] * h1[(size_t) i];
        out[(size_t) o] = sum;
    }
    return true;
}

}  // namespace

bool hunyuanimage3_wte_host(const HunyuanImage3File& f, const int32_t* ids, int64_t t, float* out, std::string& err) {
    // the rows the sequence reads, gathered out of the file's BF16: a prompt is tens to thousands of ids and the
    // table is 133120 x 4096, so converting the whole thing to pick rows out of it costs more than everything the
    // rows feed (measured: the walk's host entry spent most of its seconds there).  bf16 widens exactly, by the one
    // host lookup the text models share (native_row_to_float).
    const SafeTensor* wte = f.tensors().find("model.wte.weight");
    if (wte == nullptr) {
        err = "wte: no tensor model.wte.weight";
        return false;
    }
    if (wte->dtype != "BF16" || wte->shape.size() != 2 || wte->shape[1] != kHidden) {
        err = "wte: " + wte->dtype + " of " + std::to_string(wte->shape.size()) + " dimensions";
        return false;
    }
    const uint8_t* bytes = nullptr;
    uint64_t taken = 0;
    if (!f.raw("model.wte.weight", bytes, taken, err)) return false;
    const int64_t vocab = wte->shape[0];
    for (int64_t i = 0; i < t; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || id >= vocab) {
            err = "wte: id " + std::to_string(id) + " outside the vocabulary of " + std::to_string(vocab);
            return false;
        }
        if (!strata::kernels::cpu::native_row_to_float(30, bytes + (uint64_t) id * kHidden * 2, out + i * kHidden,
                                                       kHidden)) {
            err = "wte: the BF16 row could not be read";
            return false;
        }
    }
    return true;
}

bool hunyuanimage3_time_host(const HunyuanImage3File& f, const std::string& which, float t, float* out,
                             std::string& err) {
    Ctx c{f, err};
    std::vector<float> v;
    if (!time_embed(c, which, t, v)) return false;
    std::copy(v.begin(), v.end(), out);
    return true;
}

bool hunyuanimage3_patch_embed_host(const HunyuanImage3File& f, const float* latent, int64_t h, int64_t w, float t,
                                    float* out, std::string& err) {
    Ctx c{f, err};
    const int64_t n = h * w;
    std::vector<float> x(latent, latent + 32 * n), hidden, emb;
    if (!conv3x3(c, "patch_embed.model.0", x, 32, 1024, h, w, hidden)) return false;
    if (!time_embed(c, "time_embed", t, emb)) return false;
    if (!res_block(c, "patch_embed.model.1", hidden, 1024, 4096, h, w, emb)) return false;
    for (int64_t i = 0; i < n; ++i)   // [4096][h][w] -> [h*w][4096]
        for (int64_t ch = 0; ch < 4096; ++ch) out[i * 4096 + ch] = hidden[(size_t) (ch * n + i)];
    return true;
}

bool hunyuanimage3_final_layer_host(const HunyuanImage3File& f, const float* x, int64_t h, int64_t w, float t,
                                    float* out, std::string& err) {
    Ctx c{f, err};
    const int64_t n = h * w;
    std::vector<float> hidden((size_t) (4096 * n)), emb;
    for (int64_t i = 0; i < n; ++i)   // [h*w][4096] -> [4096][h][w]
        for (int64_t ch = 0; ch < 4096; ++ch) hidden[(size_t) (ch * n + i)] = x[i * 4096 + ch];
    if (!time_embed(c, "time_embed_2", t, emb)) return false;
    if (!res_block(c, "final_layer.model.0", hidden, 4096, 1024, h, w, emb)) return false;
    if (!group_norm(c, "final_layer.model.1.0", hidden, 1024, h, w)) return false;
    silu_all(hidden);
    std::vector<float> vel;
    if (!conv3x3(c, "final_layer.model.1.2", hidden, 1024, 32, h, w, vel)) return false;
    std::copy(vel.begin(), vel.end(), out);
    return true;
}

// the rows of the three meta tokens: the reference's `instantiate_*_tokens` scatter, which changes every step
bool hunyuanimage3_meta_rows_host(const HunyuanImage3File& f, const int32_t* ids, int64_t t, float timestep,
                                  float timestep_r, float guidance, float* hidden, std::string& err) {
    std::vector<float> row((size_t) kHidden);
    for (int64_t i = 0; i < t; ++i) {
        const char* which = nullptr;
        float v = 0.0f;
        if (ids[i] == kHunyuanImage3Timestep) {
            which = "timestep_emb";
            v = timestep;
        } else if (ids[i] == kHunyuanImage3Guidance) {
            which = "guidance_emb";
            v = guidance;
        } else if (ids[i] == kHunyuanImage3TimestepR) {
            which = "timestep_r_emb";
            v = timestep_r;
        }
        if (which == nullptr) continue;
        if (!hunyuanimage3_time_host(f, which, v, row.data(), err)) return false;
        std::copy(row.begin(), row.end(), hidden + i * kHidden);
    }
    return true;
}

// The model's input for one denoise step, host FP32: the text rows through `wte`, the three meta rows replaced
// by their embedders' outputs, and the image block through `patch_embed`.  The three pieces are `wte_host`,
// `meta_rows_host` and `patch_embed_host`, which a walk that keeps the text rows between steps calls itself.
bool hunyuanimage3_hidden_in_host(const HunyuanImage3File& f, const int32_t* ids, int64_t t, int64_t token_h,
                                  int64_t token_w, const float* latent, float timestep, float timestep_r,
                                  float guidance, float* hidden, std::string& err) {
    std::vector<int32_t> y, x;
    int64_t start = 0;
    if (!hunyuanimage3_positions(ids, t, token_h, token_w, y, x, start, err)) return false;
    if (!hunyuanimage3_wte_host(f, ids, start, hidden, err)) return false;
    if (!hunyuanimage3_meta_rows_host(f, ids, start, timestep, timestep_r, guidance, hidden, err)) return false;
    return hunyuanimage3_patch_embed_host(f, latent, token_h, token_w, timestep,
                                          hidden + (size_t) start * kHidden, err);
}

// One denoise step on the host, FP32, from the quantized file: the whole model, the sequence built from `ids`.
//
//   text rows  <- wte, with the meta tokens' rows replaced by their embedders: <timestep> (128017) by
//                 timestep_emb(t), <guidance> (130082) by guidance_emb(g), <timestep_r> (130107) by
//                 timestep_r_emb(t_r) - the reference scatters those in `instantiate_continuous_tokens`,
//                 `instantiate_guidance_tokens` and `instantiate_timestep_r_tokens`.
//   image rows <- patch_embed(latent, time_embed(t))
//   `layers` decoder layers, over the block's positions and the block's mask
//   velocity   <- final_layer(the image rows, time_embed_2(t), h, w); no ln_f: the reference's image path does not
//                 take it (its `ln_f` is only for the text's logits).
//
// Slow by nature (every layer dequantizes its whole bank: minutes a step on this machine's CPU); it is the number
// the GPU step is compared with, not the shipping path.
bool hunyuanimage3_step_host(const HunyuanImage3File& f, const int32_t* ids, int64_t t, int64_t token_h,
                             int64_t token_w, const float* latent, float timestep, float timestep_r, float guidance,
                             int64_t layers, float* velocity, std::string& err,
                             const std::function<void(int64_t)>& layer_done) {
    std::vector<int32_t> y, x;
    int64_t start = 0;
    if (!hunyuanimage3_positions(ids, t, token_h, token_w, y, x, start, err)) return false;
    std::vector<float> hid((size_t) (t * kHidden));
    if (!hunyuanimage3_hidden_in_host(f, ids, t, token_h, token_w, latent, timestep, timestep_r, guidance,
                                      hid.data(), err))
        return false;
    for (int64_t l = 0; l < layers; ++l) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!hunyuanimage3_layer_host(f, l, hid.data(), t, hid.data(), y.data(), x.data(), token_h * token_w, err))
            return false;
        if (layer_done) {
            layer_done(l);
        } else {
            std::fprintf(stderr, "  layer %2lld  %.1f s\n", (long long) l,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            std::fflush(stderr);
        }
    }
    return hunyuanimage3_final_layer_host(f, hid.data() + (size_t) start * kHidden, token_h, token_w, timestep,
                                          velocity, err);
}

// The whole sampler on the host: Euler from the seed's Gaussian noise through `sigmas` (steps + 1, falling to 0),
// the latent out [32][token_h][token_w].  The noise is the engine's own only once the engine's Gaussian is shared;
// until then this is the finite-and-diverge check, and `seed` is the caller's.
bool hunyuanimage3_sample_host(const HunyuanImage3File& f, const int32_t* ids, int64_t t, int64_t token_h,
                               int64_t token_w,
                               const std::vector<float>& sigmas, float guidance_scale, int64_t layers, uint32_t seed,
                               float* latent_out, std::string& err) {
    const int64_t n = 32 * token_h * token_w;
    std::vector<float> x((size_t) n), v((size_t) n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    for (float& a : x) a = norm(rng);
    const int64_t steps = (int64_t) sigmas.size() - 1;
    for (int64_t s = 0; s < steps; ++s) {
        std::fprintf(stderr, "step %lld  sigma %.6f\n", (long long) s, (double) sigmas[(size_t) s]);
        std::fflush(stderr);
        if (!hunyuanimage3_step_host(f, ids, t, token_h, token_w, x.data(), sigmas[(size_t) s] * 1000.0f,
                                     sigmas[(size_t) (s + 1)] * 1000.0f, 1000.0f * guidance_scale, layers, v.data(),
                                     err))
            return false;
        double sum = 0, absmax = 0;
        for (float a : v) {
            sum += a;
            absmax = std::max(absmax, (double) std::fabs(a));
        }
        std::fprintf(stderr, "  v mean %+.6f absmax %.5f\n", sum / (double) v.size(), absmax);
        std::fflush(stderr);
        hunyuanimage3_euler_step(x.data(), v.data(), n, sigmas[(size_t) s], sigmas[(size_t) (s + 1)]);
    }
    std::copy(x.begin(), x.end(), latent_out);
    return true;
}

void hunyuanimage3_sigmas(int64_t steps, float shift, float* sigmas) {
    for (int64_t i = 0; i <= steps; ++i) {
        const float t = 1.0f - (float) i / (float) steps;          // linspace(1, 0, steps + 1)
        sigmas[i] = shift * t / (1.0f + (shift - 1.0f) * t);
    }
}

void hunyuanimage3_euler_step(float* x, const float* v, int64_t n, float sigma, float sigma_next) {
    const float dt = sigma_next - sigma;
    for (int64_t i = 0; i < n; ++i) x[i] += dt * v[i];   // x and v fp32; no cast back, as the reference
}

}  // namespace strata::image