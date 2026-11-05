// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_kernels.cpp - see include/strata/image/hunyuanimage3_kernels.hpp.
//
// The shapes are the model's (head_dim 128, hidden 4096, an intermediate of 3072 = 12 groups of 256) but nothing
// here is tuned: these are the plain forms, held to the host's numbers first (strata-image-hunyuanimage3-probe's
// `gpu` and `gpulayer`), and tile/local-memory versions come once the values agree.

#include "strata/image/hunyuanimage3_kernels.hpp"
#include "strata/image/qi_kernels.hpp"

#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/qk_norm_rope.hpp"
#include "strata/kernels/rows_cols.hpp"

#include <sycl/sycl.hpp>

#include <stdexcept>

namespace strata::image {

// The kernels' helpers in a namespace of this file's own: SYCL names a lambda's kernel by its mangled type, and
// an anonymous namespace mangles the same in every file.
namespace hunyuanimage3_detail {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

// The GroupNorm(32) statistics (hunyuanimage3_group_part and -stats): a work-group a chunk of kGnChunk positions and
// one of the 32 groups, the chunk sums reduced over at most kGnMaxChunk of them a work-group.  The VAE's 32 groups
// and the entry and exit's are the same reduction; only eps differs.
constexpr int64_t kGnGroups = 32;      // every GroupNorm here: nn.GroupNorm(32, C)
constexpr int64_t kGnChunk = 256;      // positions a chunk
constexpr int64_t kGnMaxChunk = 256;   // chunks a reducer's work-group

// The router: the lanes a dot product is split over, each summing every kRouterLanes-th term (FP32 throughout, the
// reference's own type, and no FP64, which not every GPU has), and the most experts a token takes.
constexpr size_t kRouterLanes = 32;
constexpr int64_t kRouterMaxTopk = 32;

void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

// An FP8 e4m3 byte as a float: 4 exponent bits with a bias of 7, 3 mantissa bits, no inf and no NaN in the
// files.  e=0 is a subnormal m/8 * 2^-6 (the dequantization was calibrated against the BF16 weights,
// CALIB-2026-10-23.md).
float e4m3(uint8_t b) {
    const uint32_t s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    // a normal one is the float with the same mantissa bits and the exponent rebiased (exact, no pow)
    const float v = e == 0 ? (float) m * 0.001953125f                 // m / 8 * 2^-6
                           : sycl::bit_cast<float>(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}

// One group of 256 values times the normalized regular Hadamard: four butterflies on the group's base-4 digits
// (strides 64, 16, 4, 1), H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], by a work-group of 256 on the values in
// local memory, the work-item `lid` writing value `lid` (each sum in the order a single work-item's loop had them,
// so the bits are the same).  Returns the value.
float hadamard256_lane(float* t, int lid, const sycl::nd_item<1>& it) {
    constexpr int h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    float scale = 1.0f;
    for (int n = 256; n > 1; n /= 4) scale *= 0.5f;
    for (int stride = 64; stride >= 1; stride /= 4) {
        const int i = (lid / stride) % 4, first = lid - i * stride;
        sycl::group_barrier(it.get_group());
        float sum = 0.f;
        for (int j = 0; j < 4; ++j) sum += (float) h4[i][j] * t[first + j * stride];
        sycl::group_barrier(it.get_group());
        t[lid] = sum;
    }
    return t[lid] * scale;
}

}  // namespace hunyuanimage3_detail

using hunyuanimage3_detail::e4m3;
using hunyuanimage3_detail::f16_from_f32;
using hunyuanimage3_detail::f32_from_f16;
using hunyuanimage3_detail::finish;
using hunyuanimage3_detail::hadamard256_lane;
using hunyuanimage3_detail::kGnChunk;
using hunyuanimage3_detail::kGnGroups;
using hunyuanimage3_detail::kGnMaxChunk;
using hunyuanimage3_detail::kRouterLanes;
using hunyuanimage3_detail::kRouterMaxTopk;
using hunyuanimage3_detail::queue_for;

void hunyuanimage3_rotate16_from32(const float* x32, int64_t rows, int64_t K, int64_t block, uint16_t* out16, void* stream) {
    if (rows <= 0 || block <= 1) return;
    if (block != 256) throw std::runtime_error("hunyuanimage3_rotate16_from32: only the 256-wide group is implemented");
    const int64_t groups = rows * (K / 256);
    // a work-group a group of 256 (hadamard256_lane): one value a work-item
    const auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> t(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<1>((size_t) groups * 256, 256), [=](sycl::nd_item<1> it) {
            const int64_t g = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const int64_t at = (g / (K / 256)) * K + (g % (K / 256)) * 256 + lid;
            float* tp = t.get_multi_ptr<sycl::access::decorated::no>().get();
            tp[lid] = x32[at];
            out16[at] = f16_from_f32(hadamard256_lane(tp, lid, it));   // the one rounding, after the spread
        });
    });
    finish(stream, e, "hunyuanimage3_rotate16_from32");
}

void hunyuanimage3_rotate16(const uint16_t* x16, int64_t rows, int64_t K, int64_t block, uint16_t* out16, void* stream) {
    if (rows <= 0 || block <= 1) return;
    if (block != 256) throw std::runtime_error("hunyuanimage3_rotate16: only the 256-wide group is implemented");
    const int64_t groups = rows * (K / 256);
    const auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> t(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<1>((size_t) groups * 256, 256), [=](sycl::nd_item<1> it) {
            const int64_t g = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const int64_t at = (g / (K / 256)) * K + (g % (K / 256)) * 256 + lid;
            float* tp = t.get_multi_ptr<sycl::access::decorated::no>().get();
            tp[lid] = f32_from_f16(x16[at]);
            out16[at] = f16_from_f32(hadamard256_lane(tp, lid, it));
        });
    });
    finish(stream, e, "hunyuanimage3_rotate16");
}

void hunyuanimage3_w4a8_dequant16(const int8_t* codes, const uint8_t* s_rel, const float* s_channel, const float* codebook,
                        int64_t r0, int64_t r1, int64_t N, int64_t K, int64_t group, uint16_t* out16,
                        void* stream) {
    if (r1 <= r0) return;
    (void) N;
    const int64_t rows = r1 - r0;
    if (K % group != 0 || group % 2 != 0)
        throw std::runtime_error("hunyuanimage3_w4a8_dequant16: " + std::to_string(K) + " columns in groups of " +
                                 std::to_string(group));
    // a work-item a group: its scale read and decoded once for the group's values
    const int64_t groups = K / group;
    const auto e = kernels::for_rows_cols(queue_for(stream), rows, groups, [=](int64_t r, int64_t g) {
            const float sg = e4m3(s_rel[r * groups + g]);
            const float sc = s_channel[r];
            const uint8_t* const src = reinterpret_cast<const uint8_t*>(codes) + r * (K / 2) + g * (group / 2);
            uint16_t* const dst = out16 + r * K + g * group;
            for (int64_t c = 0; c < group / 2; ++c) {
                // the even column (2c) in the low nibble, the odd (2c+1) in the high: measured against the BF16
                // weights (the other order gives cos 0.02)
                const uint8_t byte = src[c];
                dst[2 * c] = f16_from_f32(codebook[byte & 0xF] * sg * sc);
                dst[2 * c + 1] = f16_from_f32(codebook[byte >> 4] * sg * sc);
            }
        });
    finish(stream, e, "hunyuanimage3_w4a8_dequant16");
}

void hunyuanimage3_int8_dequant16(const int8_t* w, const float* row_scale, int64_t r0, int64_t r1, int64_t N, int64_t K,
                        uint16_t* out16, void* stream) {
    if (r1 <= r0) return;
    (void) N;
    const int64_t rows = r1 - r0;
    const auto e = kernels::for_rows_cols(queue_for(stream), rows, K, [=](int64_t r, int64_t c) {
            out16[r * K + c] = f16_from_f32((float) w[r * K + c] * row_scale[r]);
        });
    finish(stream, e, "hunyuanimage3_int8_dequant16");
}

void hunyuanimage3_rope_angles(const int32_t* pos_y, const int32_t* pos_x, int64_t T, int64_t head_dim, float theta,
                     float* ang, void* stream) {
    if (T <= 0) return;
    const int64_t half = head_dim / 2;
    const auto e = kernels::for_rows_cols(queue_for(stream), T, half, [=](int64_t t, int64_t j) {
        // FP32 as the reference's `1.0 / (base ** (arange(0, n, 2).float() / n))` (and no FP64, which not every
        // GPU has)
        const float pos = (float) ((j % 2) ? pos_x[t] : pos_y[t]);
        ang[t * half + j] = pos * (1.0f / sycl::pow(theta, (float) (2 * j) / (float) head_dim));
    });
    finish(stream, e, "hunyuanimage3_rope_angles");
}

void hunyuanimage3_qk_rope_norm16(const float* qkv, const float* wq, const float* wk, const float* ang, int64_t T,
                        int64_t q_heads, int64_t kv_heads, int64_t head_dim, float eps, uint16_t* q16,
                        uint16_t* k16, uint16_t* v16, void* stream) {
    if (T <= 0) return;
    const int64_t groups = q_heads / kv_heads;      // the reference's num_key_value_groups
    // the file's order: [kv head][q head of that group, then k, then v][head_dim]
    struct In {
        const float* qkv;
        int64_t kv_heads, groups, hd;
        float at(int which, int64_t t, int64_t h, int64_t d) const {
            const int64_t kv = which == 0 ? h / groups : h;
            const int64_t slot = which == 0 ? h % groups : which == 1 ? groups : groups + 1;
            return qkv[((t * kv_heads + kv) * (groups + 2) + slot) * hd + d];
        }
    };
    // rotate_half with the SAME angle for both halves, then the norm (the reference's order): the rotation is
    // orthogonal on each pair, so the head's sum of squares - the norm's denominator - is the same before and after
    // it.  k and v written to each of the q heads their KV head serves.
    kernels::qk::Shape sh{T, q_heads, kv_heads, head_dim};
    sh.wq = wq;
    sh.wk = wk;
    sh.eps = eps;
    sh.rope_first = true;
    const auto e = kernels::qk::qk_norm_rope(queue_for(stream), sh, In{qkv, kv_heads, groups, head_dim},
                                             kernels::qk::Angles{ang, head_dim / 2},
                                             kernels::qk::Rows16{q16, k16, v16, q_heads, q_heads, head_dim, groups});
    finish(stream, e, "hunyuanimage3_qk_rope_norm16");
}

void hunyuanimage3_swiglu16_rotated(const float* gu, int64_t T, int64_t inter, uint16_t* out16, void* stream) {
    if (T <= 0) return;
    if (inter % 256 != 0) throw std::runtime_error("hunyuanimage3_swiglu16_rotated: the intermediate is not in 256 groups");
    const int64_t groups = T * (inter / 256);
    // a work-group a group of 256 (hadamard256_lane): one value a work-item
    const auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> v(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<1>((size_t) groups * 256, 256), [=](sycl::nd_item<1> it) {
            const int64_t grp = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const int64_t t = grp / (inter / 256), off = (grp % (inter / 256)) * 256 + lid;
            const float gate = gu[t * 2 * inter + off], up = gu[t * 2 * inter + inter + off];
            float* vp = v.get_multi_ptr<sycl::access::decorated::no>().get();
            vp[lid] = gate * (up / (1.f + sycl::exp(-up)));     // x1 * silu(x2), the SECOND half silu'd
            out16[t * inter + off] = f16_from_f32(hadamard256_lane(vp, lid, it));
        });
    });
    finish(stream, e, "hunyuanimage3_swiglu16_rotated");
}

void hunyuanimage3_gather16(const uint16_t* x16, const int32_t* rows, int64_t n_rows, int64_t K, uint16_t* out16,
                  void* stream) {
    if (n_rows <= 0) return;
    const auto e = kernels::for_rows_cols(queue_for(stream), n_rows, K, [=](int64_t r, int64_t c) {
        out16[r * K + c] = x16[(int64_t) rows[r] * K + c];
    });
    finish(stream, e, "hunyuanimage3_gather16");
}

void hunyuanimage3_scatter_scale_add(float* x, const float* y, const float* w, const int32_t* rows, int64_t n_rows,
                           int64_t C, void* stream) {
    if (n_rows <= 0) return;
    const auto e = kernels::for_rows_cols(queue_for(stream), n_rows, C, [=](int64_t r, int64_t c) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> target(
                x[(int64_t) rows[r] * C + c]);
            target += w[r] * y[r * C + c];
        });
    finish(stream, e, "hunyuanimage3_scatter_scale_add");
}

void hunyuanimage3_add(float* x, const float* v, int64_t n, void* stream) { kernels::add_inplace(x, v, n, stream); }

void hunyuanimage3_copy16(const uint16_t* src16, uint16_t* dst16, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        dst16[i[0]] = src16[i[0]];
    });
    finish(stream, e, "hunyuanimage3_copy16");
}

void hunyuanimage3_router_logits(const float* x32, const float* w, int64_t T, int64_t C, int64_t E, float* logits,
                       void* stream) {
    if (T <= 0) return;
    // a work-group a (token, expert): its lanes read neighbouring columns, each sums every L-th term, and the
    // group adds the lanes' sums
    constexpr size_t L = kRouterLanes;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * E) * L), sycl::range<1>(L)), [=](sycl::nd_item<1> it) {
            const int64_t g = (int64_t) it.get_group(0), t = g / E, ex = g - t * E;
            const int64_t lane = (int64_t) it.get_local_id(0);
            const float* xr = x32 + t * C;
            const float* wr = w + ex * C;
            float part = 0.f;
            for (int64_t c = lane; c < C; c += (int64_t) L) part += xr[c] * wr[c];
            const float sum = sycl::reduce_over_group(it.get_group(), part, sycl::plus<float>());
            if (lane == 0) logits[t * E + ex] = sum;
        });
    finish(stream, e, "hunyuanimage3_router_logits");
}

void hunyuanimage3_router_topk(const float* logits, int64_t T, int64_t E, int64_t K, int32_t* idx, float* wt,
                               void* stream) {
    if (T <= 0) return;
    if (K <= 0 || K > kRouterMaxTopk || K > E)
        throw std::runtime_error("hunyuanimage3_router_topk: " + std::to_string(K) + " of " + std::to_string(E) +
                                 " experts");
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) T), [=](sycl::id<1> id) {
        const int64_t t = (int64_t) id[0];
        const float* lg = logits + t * E;
        int32_t pick[kRouterMaxTopk];
        for (int64_t j = 0; j < K; ++j) {
            int32_t best = -1;
            for (int64_t ex = 0; ex < E; ++ex) {
                bool taken = false;
                for (int64_t i = 0; i < j; ++i) taken = taken || pick[i] == (int32_t) ex;
                if (!taken && (best < 0 || lg[ex] > lg[best])) best = (int32_t) ex;   // the lower expert on a tie
            }
            pick[j] = best;
        }
        // softmax over the K picked (norm_topk_prob: the full softmax's denominator cancels), from the largest
        const float m = lg[pick[0]];
        float s = 0.f;
        for (int64_t j = 0; j < K; ++j) s += sycl::exp(lg[pick[j]] - m);
        for (int64_t j = 0; j < K; ++j) {
            idx[t * K + j] = pick[j];
            wt[t * K + j] = sycl::exp(lg[pick[j]] - m) / s;
        }
    });
    finish(stream, e, "hunyuanimage3_router_topk");
}

void hunyuanimage3_route_lists(const int32_t* idx, const float* wt, int64_t T, int64_t K, int64_t E, int32_t* bounds,
                               int32_t* rows, float* w, int32_t* cursor, void* stream) {
    if (T <= 0) return;
    sycl::queue& q = queue_for(stream);
    const int64_t n = T * K;
    using ref = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    // the counts (an entry a work-item), their prefix, then each entry into its expert's next place: the order
    // within an expert is the atomics', which changes no row's product (each row is its own)
    q.memset(bounds, 0, (size_t) (E + 1) * sizeof(int32_t));
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { ref(bounds[idx[id[0]] + 1]).fetch_add(1); });
    q.single_task([=]() {
        for (int64_t ex = 0; ex < E; ++ex) {
            bounds[ex + 1] += bounds[ex];
            cursor[ex] = bounds[ex];
        }
    });
    const auto e = q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int32_t at = ref(cursor[idx[i]]).fetch_add(1);
        rows[at] = (int32_t) (i / K);
        w[at] = wt[i];
    });
    finish(stream, e, "hunyuanimage3_route_lists");
}

void hunyuanimage3_grid_in16(const float* x, int64_t C, int64_t P, uint16_t* out16, void* stream, uint16_t* lo16) {
    if (C <= 0 || P <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (P * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C;
        const float v = x[c * P + p];
        out16[i] = f16_from_f32(v);
        if (lo16 != nullptr) lo16[i] = f16_from_f32(v - f32_from_f16(out16[i]));
    });
    finish(stream, e, "hunyuanimage3_grid_in16");
}

void hunyuanimage3_to16(const float* x, int64_t n, uint16_t* y16, void* stream, uint16_t* lo16) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = x[id[0]];
        y16[id[0]] = f16_from_f32(v);
        if (lo16 != nullptr) lo16[id[0]] = f16_from_f32(v - f32_from_f16(y16[id[0]]));
    });
    finish(stream, e, "hunyuanimage3_to16");
}

void hunyuanimage3_im2col16(const uint16_t* in16, int64_t H, int64_t W, int64_t C, int64_t p0, int64_t rows,
                            uint16_t* col16, void* stream) {
    qi_im2col16(in16, 1, H, W, C, 1, 3, 1, 1, H, W, p0, rows, col16, stream);
}

void hunyuanimage3_bias_add32(float* y, const float* b, int64_t rows, int64_t N, void* stream) {
    qi_bias_add(y, b, rows, N, stream);
}

int64_t hunyuanimage3_group_ws_floats(int64_t P) {
    const int64_t B = (P + kGnChunk - 1) / kGnChunk, G = (B + kGnMaxChunk - 1) / kGnMaxChunk;
    return 2 * kGnGroups + (B + G) * kGnGroups;
}

void hunyuanimage3_group_part(const float* x, const float* mean, int64_t P, int64_t C, bool sq, float* part,
                              void* stream) {
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
                const float* r = x + p * C + g * cg;
                for (int64_t c = 0; c < cg; ++c) {
                    const float d = r[c] - m;
                    const float y = (sq ? d * d : d) - comp, t = a + y;
                    comp = (t - a) - y;
                    a = t;
                }
            }
            a = sycl::reduce_over_group(it.get_group(), a, sycl::plus<float>());
            if (lane == 0) part[blk] = a;
        });
    finish(stream, e, "hunyuanimage3_group_part");
}

void hunyuanimage3_group_stats(const float* part, int64_t B, int64_t P, int64_t CG, float eps, bool sq,
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
        finish(stream, e, "hunyuanimage3_group_stats (the reducers' level)");
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
    finish(stream, e, "hunyuanimage3_group_stats");
}

void hunyuanimage3_group_apply16(const float* x, const float* mean, const float* inv, const float* w,
                                 const float* b, const float* scale, const float* shift, int64_t P, int64_t C,
                                 bool silu, uint16_t* y16, void* stream, uint16_t* lo16) {
    if (P <= 0) return;
    const int64_t cg = C / kGnGroups;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (P * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, g = c / cg;
        float v = (x[i] - mean[g]) * inv[g] * w[c] + b[c];
        if (scale != nullptr) v = v * (1.f + scale[c]) + shift[c];
        if (silu) v = v / (1.f + sycl::exp(-v));
        y16[i] = f16_from_f32(v);
        if (lo16 != nullptr) lo16[i] = f16_from_f32(v - f32_from_f16(y16[i]));
    });
    finish(stream, e, "hunyuanimage3_group_apply16");
}

void hunyuanimage3_scale_rows16(const float* x, const float* rs, int64_t n_rows, int64_t C, uint16_t* y16, void* stream) {
    if (n_rows <= 0) return;
    const auto e = kernels::for_rows_cols(queue_for(stream), n_rows, C, [=](int64_t r, int64_t c) {
            const float s = rs != nullptr ? rs[r] : 1.f;
            y16[r * C + c] = f16_from_f32(x[r * C + c] / s);
        });
    finish(stream, e, "hunyuanimage3_scale_rows16");
}

}  // namespace strata::image
