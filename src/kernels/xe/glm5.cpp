// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/glm5.cpp - see include/strata/kernels/glm5.hpp.
//
// Transcribed from llama.cpp f9a95ae (src/models/glm5-next.cpp; ggml/src/ggml-cpu/ops.cpp's dsv4_hc_comb,
// gated_delta_net and swiglu_clamp); MIT License, Copyright (c) 2023-2026 The ggml authors, see
// third_party/main/ggml/LICENSE.  The KDA step keeps the layout and the sub-group shape of native_gdn.cpp.
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_glm5
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/kernels/rows_cols.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace strata::kernels {
namespace {

constexpr int HC = 4;                  // the mHC streams (model_spec refuses another count)
constexpr int MIX = (2 + HC) * HC;     // pre, post and the HC x HC comb
constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }
void sync_if_needed(void* stream, const sycl::event& event, const char* what) {
    if (!stream) core::Runtime::get().wait(event, what);
}

inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

}  // namespace

void glm5_hc_pre(const float* x, const float* mixes, const float* scale, const float* base, float rms_eps, float hc_eps,
                 int iters, float* y, float* post, float* comb, int64_t n_embd, int64_t n_tokens, void* stream,
                 int streams, float pre_scale) {
    if (n_tokens <= 0) return;
    if (streams != 2 && streams != 4) throw core::DeviceError("glm5_hc_pre: unsupported stream count");
    if (n_embd <= 0 || iters < 1) throw core::DeviceError("glm5_hc_pre: n_embd " + std::to_string(n_embd) + ", iters " +
                                                          std::to_string(iters));
    constexpr int WG = 256;
    const int64_t N = n_embd;
    if (streams == 2) {   // SenseNova-Loop: pre/post/bias and row-major log-domain Sinkhorn
        const auto event = queue_for(stream).submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> pre(sycl::range<1>(2), h);
            h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * WG, WG), [=](sycl::nd_item<1> it) {
                const int64_t t = (int64_t) it.get_group(0);
                const int lid = (int) it.get_local_id(0);
                const float* xt = x + t * 2 * N;
                float ss = 0.0f;
                for (int64_t i = lid; i < 2 * N; i += WG) ss += xt[i] * xt[i];
                ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                if (lid == 0) {
                    const float inv = sycl::rsqrt(ss / (float) (2 * N) + rms_eps);
                    float m[8];
                    for (int i = 0; i < 8; ++i) m[i] = mixes[t * 8 + i] * inv + base[i];
                    for (int i = 0; i < 2; ++i) {
                        pre[i] = pre_scale * sigmoid(m[i]);
                        post[t * 2 + i] = 2.0f * sigmoid(m[2 + i]);
                    }
                    auto normalize = [](float& a, float& b) {
                        const float delta = a - b;
                        const float z = sycl::log1p(sycl::exp(-sycl::fabs(delta)));
                        a = sycl::fmin(delta, 0.0f) - z;
                        b = sycl::fmin(-delta, 0.0f) - z;
                    };
                    for (int i = 0; i < iters; ++i) {
                        normalize(m[4], m[5]); normalize(m[6], m[7]);
                        normalize(m[4], m[6]); normalize(m[5], m[7]);
                    }
                    for (int i = 0; i < 4; ++i) comb[t * 4 + i] = sycl::exp(m[4 + i]);
                }
                sycl::group_barrier(it.get_group());
                for (int64_t n = lid; n < N; n += WG) y[t * N + n] = xt[n] * pre[0] + xt[N + n] * pre[1];
            });
        });
        sync_if_needed(stream, event, "glm5_hc_pre");
        return;
    }
    const auto event = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> pre_l(sycl::range<1>(HC), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t t = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const float* xt = x + t * HC * N;
            // the RMS of the token's 4N stream values
            float ss = 0.0f;
            for (int64_t i = lid; i < HC * N; i += WG) ss += xt[i] * xt[i];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            if (lid == 0) {
                const float inv = 1.0f / sycl::sqrt(ss / (float) (HC * N) + rms_eps);
                float m[MIX];
                for (int i = 0; i < MIX; ++i) m[i] = mixes[t * MIX + i] * inv;
                for (int s = 0; s < HC; ++s) {
                    pre_l[s] = sigmoid(m[s] * scale[0] + base[s]) + hc_eps;
                    post[t * HC + s] = 2.0f * sigmoid(m[HC + s] * scale[1] + base[HC + s]);
                }
                // ggml_dsv4_hc_comb: softmax over dst for each src, + eps, then the Sinkhorn rounds
                float c[HC * HC];
                for (int src = 0; src < HC; ++src) {
                    float mx = -INFINITY;
                    for (int dst = 0; dst < HC; ++dst) {
                        const int idx = dst + HC * src;
                        c[idx] = m[2 * HC + idx] * scale[2] + base[2 * HC + idx];
                        mx = sycl::fmax(mx, c[idx]);
                    }
                    float sum = 0.0f;
                    for (int dst = 0; dst < HC; ++dst) {
                        const int idx = dst + HC * src;
                        c[idx] = sycl::exp(c[idx] - mx);
                        sum += c[idx];
                    }
                    const float inv_sum = 1.0f / sum;
                    for (int dst = 0; dst < HC; ++dst) c[dst + HC * src] = c[dst + HC * src] * inv_sum + hc_eps;
                }
                auto norm_cols = [&]() {
                    for (int dst = 0; dst < HC; ++dst) {
                        float sum = hc_eps;
                        for (int src = 0; src < HC; ++src) sum += c[dst + HC * src];
                        const float inv_sum = 1.0f / sum;
                        for (int src = 0; src < HC; ++src) c[dst + HC * src] *= inv_sum;
                    }
                };
                auto norm_rows = [&]() {
                    for (int src = 0; src < HC; ++src) {
                        float sum = hc_eps;
                        for (int dst = 0; dst < HC; ++dst) sum += c[dst + HC * src];
                        const float inv_sum = 1.0f / sum;
                        for (int dst = 0; dst < HC; ++dst) c[dst + HC * src] *= inv_sum;
                    }
                };
                norm_cols();
                for (int i = 1; i < iters; ++i) {
                    norm_rows();
                    norm_cols();
                }
                for (int i = 0; i < HC * HC; ++i) comb[t * HC * HC + i] = c[i];
            }
            sycl::group_barrier(it.get_group());
            const float p0 = pre_l[0], p1 = pre_l[1], p2 = pre_l[2], p3 = pre_l[3];
            for (int64_t n = lid; n < N; n += WG)   // ggml: x0 p0, then + x1 p1, + x2 p2, + x3 p3
                y[t * N + n] = ((xt[n] * p0 + xt[N + n] * p1) + xt[2 * N + n] * p2) + xt[3 * N + n] * p3;
        });
    });
    sync_if_needed(stream, event, "glm5_hc_pre");
}

void glm5_hc_init(const float* x, float* hs, int64_t n_embd, int64_t n_tokens, void* stream, int streams) {
    if (streams != 2 && streams != 4) throw core::DeviceError("glm5_hc_init: unsupported stream count");
    if (n_tokens <= 0 || n_embd <= 0) return;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * streams * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / (streams * N), n = i % N;
        hs[i] = x[t * N + n];
    });
    sync_if_needed(stream, event, "glm5_hc_init");
}

void glm5_transpose_abd(const float* src, float* dst, int64_t a, int64_t b, int64_t d, void* stream) {
    if (a <= 0 || b <= 0 || d <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (a * b * d)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], ia = i / (b * d), ib = (i / d) % b, k = i % d;
        dst[(ib * a + ia) * d + k] = src[i];
    });
    sync_if_needed(stream, event, "glm5_transpose_abd");
}

void glm5_hc_post(const float* f, const float* residual, const float* post, const float* comb, float* out,
                  int64_t n_embd, int64_t n_tokens, void* stream, int streams) {
    if (streams != 2 && streams != 4) throw core::DeviceError("glm5_hc_post: unsupported stream count");
    if (n_tokens <= 0 || n_embd <= 0) return;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / N, n = i % N;
        if (streams == 2) {
            const float a = residual[t * 2 * N + n], b = residual[(t * 2 + 1) * N + n];
            out[t * 2 * N + n] = a * comb[t * 4] + b * comb[t * 4 + 1] + f[i] * post[t * 2];
            out[(t * 2 + 1) * N + n] = a * comb[t * 4 + 2] + b * comb[t * 4 + 3] + f[i] * post[t * 2 + 1];
            return;
        }
        const float fv = f[i];
        float r[HC];
        for (int s = 0; s < HC; ++s) r[s] = residual[(t * HC + s) * N + n];
        for (int d = 0; d < HC; ++d) {
            float v = fv * post[t * HC + d];
            for (int s = 0; s < HC; ++s) v += r[s] * comb[(t * HC + s) * HC + d];
            out[(t * HC + d) * N + n] = v;
        }
    });
    sync_if_needed(stream, event, "glm5_hc_post");
}

void glm5_hc_mean(const float* x, float* y, int64_t n_embd, int64_t n_tokens, void* stream, int streams) {
    if (streams != 2 && streams != 4) throw core::DeviceError("glm5_hc_mean: unsupported stream count");
    if (n_tokens <= 0 || n_embd <= 0) return;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / N, n = i % N;
        // each stream's pointer one step of N from the last: written as xt[N], xt[2 * N], xt[3 * N], IGC 2.27 to 2.30
        // read xt[N], xt[8 * N], xt[9 * N] on Xe2 GPUs, a fault past the buffer (AGENTS.md, "Intel Graphics Compiler";
        // the runtime turns the pass off, this form compiles right either way)
        const float* x0 = x + t * streams * N + n;
        const float* x1 = x0 + N;
        if (streams == 2) { y[i] = (*x0 + *x1) * 0.5f; return; }
        const float* x2 = x1 + N;
        const float* x3 = x2 + N;
        y[i] = (((*x0 + *x1) + *x2) + *x3) * (1.0f / HC);
    });
    sync_if_needed(stream, event, "glm5_hc_mean");
}

void glm5_router_sigmoid(const float* logits, const float* bias, int n_tokens, int n_expert, int k, bool norm,
                         float scale, int32_t* ids, float* weights, void* stream) {
    if (n_tokens <= 0) return;
    if (n_expert <= 0 || k <= 0 || k > WARP || k > n_expert)
        throw core::DeviceError("glm5_router_sigmoid: " + std::to_string(k) + " of " + std::to_string(n_expert) +
                                " experts (at most 32 picks)");
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) n_tokens * WARP, WARP), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = (int) it.get_group(0), lane = (int) sg.get_local_linear_id();
            const float* l = logits + (size_t) t * n_expert;
            // pass i: the best below pass i - 1's pick in the order (larger biased score, then lower index)
            float pv = INFINITY, mine = 0.0f;
            int pi = -1;
            for (int i = 0; i < k; ++i) {
                float bv = -INFINITY;
                int bi = n_expert;
                for (int e = lane; e < n_expert; e += WARP) {
                    const float v = sigmoid(l[e]) + bias[e];
                    if ((v < pv || (v == pv && e > pi)) && (v > bv || (v == bv && e < bi))) { bv = v; bi = e; }
                }
                for (int off = WARP / 2; off > 0; off >>= 1) {
                    const float ov = sycl::permute_group_by_xor(sg, bv, off);
                    const int oi = sycl::permute_group_by_xor(sg, bi, off);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                }
                if (lane == i) {
                    mine = sigmoid(l[bi]);
                    ids[(size_t) t * k + i] = bi;
                }
                pv = bv;
                pi = bi;
            }
            float div = 1.0f;
            if (norm) {
                const float sum = sycl::reduce_over_group(sg, lane < k ? mine : 0.0f, sycl::plus<float>());
                div = sycl::fmax(sum, 6.103515625e-05f);   // 2**-14, as llama.cpp
            }
            if (lane < k) weights[(size_t) t * k + lane] = mine / div * scale;
        });
    sync_if_needed(stream, event, "glm5_router_sigmoid");
}

void glm5_kda_gate(const float* f, const float* dt_bias, const float* ssm_a, float lb, float* g, int64_t n_heads,
                   int64_t head_dim, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    if (n_heads <= 0 || head_dim <= 0 || !(lb < 0.0f) || !std::isfinite(lb))
        throw core::DeviceError("glm5_kda_gate: a finite negative lower bound and positive head shapes");
    const int64_t C = n_heads * head_dim;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], c = i % C, h = c / head_dim;
        const float x = (f[i] + dt_bias[c]) * ssm_a[h];
        g[i] = lb * sigmoid(-x);
    });
    sync_if_needed(stream, event, "glm5_kda_gate");
}

void glm5_kda_conv(const float* x, float* state, const float* w, float* y, int64_t n_channels, int64_t n_tokens,
                   bool update, void* stream) {
    constexpr int K = 4;
    if (n_tokens <= 0 || n_channels <= 0) return;
    const int64_t C = n_channels, T = n_tokens;
    // a work-item a channel, the tokens in order (the window is short: one, or the drafts)
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int64_t c = (int64_t) id[0];
        float win[K];
        for (int j = 0; j < K - 1; ++j) win[j] = state[c * (K - 1) + j];
        const float* wc = w + c * K;
        for (int64_t t = 0; t < T; ++t) {
            win[K - 1] = x[t * C + c];
            float sum = 0.0f;
            for (int j = 0; j < K; ++j) sum += win[j] * wc[j];
            y[t * C + c] = sum / (1.0f + sycl::exp(-sum));
            for (int j = 0; j < K - 1; ++j) win[j] = win[j + 1];
        }
        if (update)
            for (int j = 0; j < K - 1; ++j) state[c * (K - 1) + j] = win[j];
    });
    sync_if_needed(stream, event, "glm5_kda_conv");
}

void glm5_l2_heads(float* x, int64_t n_heads, float eps, int64_t n_tokens, void* stream) {
    constexpr int S = 128;
    if (n_tokens <= 0 || n_heads <= 0) return;
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) (n_tokens * n_heads) * S, S), [=](sycl::nd_item<1> it) {
            float* v = x + (int64_t) it.get_group(0) * S;
            const int i = (int) it.get_local_id(0);
            const float xi = v[i];
            const float sum = sycl::reduce_over_group(it.get_group(), xi * xi, sycl::plus<float>());
            v[i] = xi * (1.0f / sycl::fmax(sycl::sqrt(sum), eps));
        });
    sync_if_needed(stream, event, "glm5_l2_heads");
}

void glm5_kda_out(const float* o, const float* g2, const float* w, float eps, float* y, int64_t n_heads,
                  int64_t n_tokens, void* stream) {
    constexpr int S = 128;
    if (n_tokens <= 0 || n_heads <= 0) return;
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) (n_tokens * n_heads) * S, S), [=](sycl::nd_item<1> it) {
            const int64_t base = (int64_t) it.get_group(0) * S;
            const int i = (int) it.get_local_id(0);
            const float oi = o[base + i];
            const float ss = sycl::reduce_over_group(it.get_group(), oi * oi, sycl::plus<float>());
            const float inv = 1.0f / sycl::sqrt(ss / (float) S + eps);
            y[base + i] = oi * inv * w[i] * sigmoid(g2[base + i]);
        });
    sync_if_needed(stream, event, "glm5_kda_out");
}

void glm5_kda_step(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
                   float* out, int64_t n_heads, void* stream) {
    constexpr int S = 128;
    constexpr int COLS = 4;   // columns (sub-groups) a work-group
    if (n_heads <= 0 || n_heads > 65535) throw core::DeviceError("glm5_kda_step: " + std::to_string(n_heads) + " heads");
    const int H = (int) n_heads;
    const float scale = 1.0f / sycl::sqrt((float) S);
    // one sub-group a (head, column): its four state rows a lane in registers, as native_gdn_step
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<2>({(size_t) H, (size_t) S * WARP}, {1, (size_t) COLS * WARP}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int head = (int) it.get_group(0);
            const int lane = (int) sg.get_local_linear_id();
            const int col = (int) it.get_group(1) * COLS + (int) sg.get_group_linear_id();
            float s[4], kr[4], qr[4], dr[4];
            float kv = 0.0f;
            for (int r = 0; r < 4; ++r) {
                const int i = r * WARP + lane;
                dr[r] = sycl::exp(g[head * S + i]);
                s[r] = dr[r] * state[((size_t) i * H + head) * S + col];   // the decay first, as ggml
                kr[r] = k[head * S + i];
                qr[r] = q[head * S + i];
                kv += s[r] * kr[r];
            }
            kv = sycl::reduce_over_group(sg, kv, sycl::plus<float>());
            const float delta = (v[head * S + col] - kv) * beta[head];
            float o = 0.0f;
            for (int r = 0; r < 4; ++r) {
                s[r] += kr[r] * delta;
                o += s[r] * qr[r];
            }
            o = sycl::reduce_over_group(sg, o, sycl::plus<float>());
            if (lane == 0) out[head * S + col] = o * scale;
            for (int r = 0; r < 4; ++r) state[((size_t) (r * WARP + lane) * H + head) * S + col] = s[r];
        });
    sync_if_needed(stream, event, "glm5_kda_step");
}

void glm5_kda_scan(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
                   float* out, int64_t n_heads, int64_t n_tokens, void* stream, float* const* snaps, int n_snaps) {
    constexpr int S = 128;
    constexpr int COLS = 4;   // columns (sub-groups) a work-group
    if (n_heads <= 0 || n_heads > 65535) throw core::DeviceError("glm5_kda_scan: " + std::to_string(n_heads) + " heads");
    if (n_snaps < 0 || n_snaps > kGlm5MaxSnaps) throw core::DeviceError("glm5_kda_scan: too many snapshots");
    if (n_tokens <= 0) return;
    const int NS = (int) std::min<int64_t>(n_snaps, n_tokens);
    const int H = (int) n_heads;
    const int64_t T = n_tokens, TS = (int64_t) H * S;
    const float scale = 1.0f / sycl::sqrt((float) S);
    // CAP: the snapshots' argument, 8 for a window of up to 8 rows (the kernel it always ran), else the wide window's
    const auto launch = [&]<int CAP>() {
    struct Snaps { float* p[CAP]; } sn{};
    for (int i = 0; i < n_snaps; ++i) sn.p[i] = snaps[i];
    // one sub-group a (head, column): its four state rows a lane in registers, as native_gdn_step
    return queue_for(stream).parallel_for(
        sycl::nd_range<2>({(size_t) H, (size_t) S * WARP}, {1, (size_t) COLS * WARP}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int head = (int) it.get_group(0);
            const int lane = (int) sg.get_local_linear_id();
            const int col = (int) it.get_group(1) * COLS + (int) sg.get_group_linear_id();
            float s[4];
            for (int r = 0; r < 4; ++r) s[r] = state[((size_t) (r * WARP + lane) * H + head) * S + col];
            const int64_t hs = (int64_t) head * S;
            for (int64_t t = 0; t < T; ++t) {   // glm5_kda_step's body a token, the column's state kept in registers
                const float* gt = g + t * TS + hs;
                const float* kt = k + t * TS + hs;
                const float* qt = q + t * TS + hs;
                float kr[4], qr[4];
                float kv = 0.0f;
                for (int r = 0; r < 4; ++r) {
                    const int i = r * WARP + lane;
                    s[r] = sycl::exp(gt[i]) * s[r];   // the decay first, as ggml
                    kr[r] = kt[i];
                    qr[r] = qt[i];
                    kv += s[r] * kr[r];
                }
                kv = sycl::reduce_over_group(sg, kv, sycl::plus<float>());
                const float delta = (v[t * TS + hs + col] - kv) * beta[t * H + head];
                float o = 0.0f;
                for (int r = 0; r < 4; ++r) {
                    s[r] += kr[r] * delta;
                    o += s[r] * qr[r];
                }
                o = sycl::reduce_over_group(sg, o, sycl::plus<float>());
                if (lane == 0) out[t * TS + hs + col] = o * scale;
                if (t < NS)   // the state after this row, to go back to (a speculative window's rows but its last)
                    for (int r = 0; r < 4; ++r) sn.p[t][((size_t) (r * WARP + lane) * H + head) * S + col] = s[r];
            }
            for (int r = 0; r < 4; ++r) state[((size_t) (r * WARP + lane) * H + head) * S + col] = s[r];
        });
    };
    const auto event = n_snaps <= kVerifyNarrowT ? launch.template operator()<kVerifyNarrowT>()
                                                 : launch.template operator()<kGlm5MaxSnaps>();
    sync_if_needed(stream, event, "glm5_kda_scan");
}

void glm5_idx_steps(const int32_t* steps_in, int64_t n_tokens, int kpool, int top_k, int32_t* steps_out, void* stream) {
    if (n_tokens <= 0) return;
    if (kpool <= 1 || top_k <= 0) throw core::DeviceError("glm5_idx_steps: kpool above 1 and a positive top_k");
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n_tokens), [=](sycl::id<1> id) {
        const int64_t t = (int64_t) id[0];
        const int32_t pos = steps_in[t * kStepCount + kStepPos];
        const int32_t nb = pos >= 0 ? (pos + 1) / kpool : 0;
        int32_t* o = steps_out + t * kStepCount;
        o[kStepPos] = pos;
        o[kStepNBid] = nb;
        o[kStepNKv] = nb * kpool;
        o[kStepWidth] = nb * kpool < top_k ? nb * kpool : top_k;
    });
    sync_if_needed(stream, event, "glm5_idx_steps");
}

void glm5_idx_append(const float* k_raw, const float* g_raw, const float* ln_w, const float* ln_b, float ln_eps,
                     const int32_t* steps, int64_t n_tokens, float* cache, void* stream) {
    constexpr int D = 128;
    if (n_tokens <= 0) return;
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) n_tokens * D, D), [=](sycl::nd_item<1> it) {
            const int64_t t = (int64_t) it.get_group(0);
            const int c = (int) it.get_local_id(0);
            const int32_t pos = steps[t * kStepCount + kStepPos];
            const float x = k_raw[t * D + c];
            // ggml_norm: the mean, then the variance of the centred values
            const float mean = sycl::reduce_over_group(it.get_group(), x, sycl::plus<float>()) / (float) D;
            const float xc = x - mean;
            const float var = sycl::reduce_over_group(it.get_group(), xc * xc, sycl::plus<float>()) / (float) D;
            if (pos < 0) return;
            float* row = cache + (int64_t) pos * 2 * D;
            row[c] = xc * (1.0f / sycl::sqrt(var + ln_eps)) * ln_w[c] + ln_b[c];
            row[D + c] = g_raw[t * D + c];
        });
    sync_if_needed(stream, event, "glm5_idx_append");
}

void glm5_idx_pool(const float* cache, const float* ape, int kpool, const int32_t* steps, int64_t n_tokens,
                   float* pooled, void* stream) {
    constexpr int D = 128, KMAX = 8;
    if (n_tokens <= 0) return;
    if (kpool <= 1 || kpool > KMAX) throw core::DeviceError("glm5_idx_pool: kpool " + std::to_string(kpool));
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n_tokens * D), [=](sycl::id<1> id) {
        const int64_t t = (int64_t) id[0] / D;
        const int c = (int) ((int64_t) id[0] % D);
        const int32_t pos = steps[t * kStepCount + kStepPos];
        if (pos < 0 || (pos + 1) % kpool != 0) return;   // no pool completes at this token
        const int64_t m = (pos + 1) / kpool - 1;
        float l[KMAX], mx = -INFINITY;
        for (int j = 0; j < kpool; ++j) {
            l[j] = cache[(m * kpool + j) * 2 * D + D + c] + ape[j * D + c];
            mx = sycl::fmax(mx, l[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < kpool; ++j) {
            l[j] = sycl::exp(l[j] - mx);
            sum += l[j];
        }
        const float inv = 1.0f / sum;
        float acc = 0.0f;
        for (int j = 0; j < kpool; ++j) acc += l[j] * inv * cache[(m * kpool + j) * 2 * D + c];
        pooled[m * D + c] = acc;
    });
    sync_if_needed(stream, event, "glm5_idx_pool");
}

void glm5_idx_scores(const float* pooled, const float* q, const float* w, int n_heads, const int32_t* steps,
                     int64_t n_tokens, int64_t max_blocks, float* scores, void* stream) {
    constexpr int D = 128, PER = D / WARP, SG = 8;   // sub-groups (blocks) a work-group
    if (n_tokens <= 0) return;
    if (n_heads <= 0 || max_blocks <= 0 || n_tokens > 65535)
        throw core::DeviceError("glm5_idx_scores: " + std::to_string(n_heads) + " heads, " + std::to_string(max_blocks) +
                                " blocks, " + std::to_string(n_tokens) + " tokens");
    const size_t gx = (size_t) ((max_blocks + SG - 1) / SG);
    const int H = n_heads;
    // the kernels' ids are ints (-fsycl-id-queries-range=int): a launch's work-items stay within one, so a long
    // context's chunk goes in slices of tokens (a 131072-cell context's 32768 blocks take 1 M work-items a token, and
    // a prompt chunk of over 2047 tokens failed)
    const int64_t row = (int64_t) (gx * SG * WARP);
    const int64_t slice = std::max<int64_t>(1, (int64_t) std::numeric_limits<int32_t>::max() / row);
    for (int64_t t0 = 0; t0 < n_tokens; t0 += slice) {
        const int64_t nt = std::min<int64_t>(slice, n_tokens - t0);
        const auto event = queue_for(stream).parallel_for(
            sycl::nd_range<2>({(size_t) nt, (size_t) row}, {1, (size_t) SG * WARP}),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int64_t t = t0 + (int64_t) it.get_group(0);
                const int64_t b = (int64_t) it.get_group(1) * SG + (int64_t) sg.get_group_linear_id();
                const int64_t n_bid = steps[t * kStepCount + kStepNBid];
                if (b > n_bid || b >= max_blocks) return;   // uniform over the sub-group
                const int lane = (int) sg.get_local_linear_id();
                float score = 0.0f;
                if (b < n_bid) {
                    float k[PER];
                    for (int i = 0; i < PER; ++i) k[i] = pooled[b * D + (int64_t) lane * PER + i];
                    for (int h = 0; h < H; ++h) {
                        const float* qh = q + (t * H + h) * D + (int64_t) lane * PER;
                        float d = 0.0f;
                        for (int i = 0; i < PER; ++i) d += qh[i] * k[i];
                        d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                        score += (d > 0.0f ? d : 0.0f) * w[t * H + h];
                    }
                }
                if (lane == 0) scores[t * max_blocks + b] = score;
            });
        sync_if_needed(stream, event, "glm5_idx_scores");
    }
}

void glm5_latent_store(const float* kv, const int32_t* steps, int64_t n_tokens, int64_t rank, uint16_t* latent,
                       void* stream) {
    if (n_tokens <= 0 || rank <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * rank)), [=](sycl::id<1> id) {
        const int64_t t = (int64_t) id[0] / rank, l = (int64_t) id[0] % rank;
        const int32_t pos = steps[t * kStepCount + kStepPos];
        if (pos >= 0) latent[(int64_t) pos * rank + l] = f16_from_f32(kv[t * rank + l]);
    });
    sync_if_needed(stream, event, "glm5_latent_store");
}

void glm5_mla_attn(const float* q, const uint16_t* latent, const int32_t* ids, int64_t cap, const int32_t* steps,
                   int n_heads, float scale, int64_t n_tokens, float* out, void* stream, int tail_kpool, int64_t compact) {
    constexpr int L = 512, WG = 256, NSG = WG / WARP, PER = L / WARP;
    if (n_tokens <= 0) return;
    if (n_heads <= 0 || cap <= 0 || n_tokens * n_heads > 0x7fffffff || tail_kpool < 0 || tail_kpool == 1)
        throw core::DeviceError("glm5_mla_attn: " + std::to_string(n_heads) + " heads, cap " + std::to_string(cap) +
                                ", tail pool " + std::to_string(tail_kpool));
    const int H = n_heads, KP = tail_kpool;
    const auto event = queue_for(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>((size_t) (cap + (KP > 0 ? KP - 1 : 0))), hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) (n_tokens * H) * WG, WG),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int64_t th = (int64_t) it.get_group(0), t = th / H;
            const int lid = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int warp = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int64_t chosen = sycl::min<int64_t>(steps[t * kStepCount + kStepWidth], cap);
            // the unfinished pool's cells up to the token, after the chosen ones: cell j - chosen + tail0
            const int32_t pos = steps[t * kStepCount + kStepPos];
            const int64_t tail0 = KP > 0 && pos >= 0 ? (int64_t) ((pos + 1) / KP) * KP : 0;
            const int64_t width = chosen + (KP > 0 && pos >= 0 ? (int64_t) pos + 1 - tail0 : 0);
            const float* qh = q + th * L;
            float* oh = out + th * L;
            if (width <= 0) {   // uniform over the work-group
                for (int d = lid; d < L; d += WG) oh[d] = 0.0f;
                return;
            }
            float qr[PER];
            for (int i = 0; i < PER; ++i) qr[i] = qh[lane * PER + i];
            const int32_t* sel = ids + t * cap;
            auto cell = [&](int64_t j) {
                return compact > 0 ? t * compact + j : j < chosen ? (int64_t) sel[j] : tail0 + (j - chosen);
            };
            for (int64_t j = warp; j < width; j += NSG) {   // a sub-group a cell
                const uint16_t* row = latent + cell(j) * L + (int64_t) lane * PER;
                float d = 0.0f;
                for (int i = 0; i < PER; ++i) d += qr[i] * f32_from_f16(row[i]);
                d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                if (lane == 0) sc[j] = d * scale;
            }
            sycl::group_barrier(it.get_group());
            float mx = -INFINITY;
            for (int64_t j = lid; j < width; j += WG) mx = sycl::fmax(mx, sc[j]);
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            float sum = 0.0f;
            for (int64_t j = lid; j < width; j += WG) sum += sycl::exp(sc[j] - mx);
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float inv = 1.0f / sum;
            for (int64_t j = lid; j < width; j += WG) sc[j] = sycl::exp(sc[j] - mx) * inv;   // the probabilities
            sycl::group_barrier(it.get_group());
            for (int d = lid; d < L; d += WG) {
                float acc = 0.0f;
                for (int64_t j = 0; j < width; ++j) acc += sc[j] * f32_from_f16(latent[cell(j) * L + d]);
                oh[d] = acc;
            }
        });
    });
    sync_if_needed(stream, event, "glm5_mla_attn");
}

void glm5_latent_gather(const uint16_t* latent, const int32_t* ids, int64_t cap, const int32_t* steps,
                        int64_t n_tokens, int tail_kpool, int64_t stride, uint16_t* out, void* stream) {
    constexpr int L = 512, W = L / 8;   // a work-item a 16-byte quarter-row
    if (n_tokens <= 0) return;
    if (cap <= 0 || stride < cap + (tail_kpool > 1 ? tail_kpool - 1 : 0) || tail_kpool < 0 || tail_kpool == 1)
        throw core::DeviceError("glm5_latent_gather: cap " + std::to_string(cap) + ", stride " + std::to_string(stride));
    const int KP = tail_kpool;
    const auto event = for_rows_cols(queue_for(stream),
        n_tokens * stride, W, [=](int64_t row, int64_t q) {
            const int64_t t = row / stride, j = row % stride;
            const int64_t chosen = sycl::min<int64_t>(steps[t * kStepCount + kStepWidth], cap);
            const int32_t pos = steps[t * kStepCount + kStepPos];
            const int64_t tail0 = KP > 0 && pos >= 0 ? (int64_t) ((pos + 1) / KP) * KP : 0;
            const int64_t width = chosen + (KP > 0 && pos >= 0 ? (int64_t) pos + 1 - tail0 : 0);
            if (j >= width) return;
            const int64_t c = j < chosen ? (int64_t) ids[t * cap + j] : tail0 + (j - chosen);
            const auto* src = reinterpret_cast<const sycl::uint4*>(latent + c * L);
            auto* dst = reinterpret_cast<sycl::uint4*>(out + (t * stride + j) * L);
            dst[q] = src[q];
        });
    sync_if_needed(stream, event, "glm5_latent_gather");
}

void glm5_swiglu_clamp(const float* gate, const float* up, float* out, int64_t n, float limit, void* stream) {
    if (n <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float gv = sycl::fmin(gate[i], limit);
        const float uv = sycl::clamp(up[i], -limit, limit);
        out[i] = gv / (1.0f + sycl::exp(-gv)) * uv;
    });
    sync_if_needed(stream, event, "glm5_swiglu_clamp");
}

}  // namespace strata::kernels
