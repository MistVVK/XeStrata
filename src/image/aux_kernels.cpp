// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/aux_kernels.cpp - see include/strata/image/aux_kernels.hpp.

#include "strata/image/aux_kernels.hpp"

#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>

namespace strata::image {

namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

}  // namespace

void aux_rgb_norm16(const uint8_t* rgba, int64_t W, int64_t H, const float* mean, const float* std, uint16_t* y16,
                    void* stream) {
    const int64_t n = W * H;
    if (n <= 0) return;
    const float m0 = mean[0], m1 = mean[1], m2 = mean[2], s0 = 1.f / std[0], s1 = 1.f / std[1], s2 = 1.f / std[2];
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        y16[3 * i] = f16_from_f32(((float) rgba[4 * i] / 255.f - m0) * s0);
        y16[3 * i + 1] = f16_from_f32(((float) rgba[4 * i + 1] / 255.f - m1) * s1);
        y16[3 * i + 2] = f16_from_f32(((float) rgba[4 * i + 2] / 255.f - m2) * s2);
    });
    finish(stream, e, "aux_rgb_norm16");
}

void aux_im2col16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int k, int stride, int pad, int64_t OW,
                  int64_t p0, int64_t n, uint16_t* col16, void* stream) {
    const int64_t K = (int64_t) k * k * C;
    if (n <= 0 || K <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n * K)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / K, q = i - r * K, tap = q / C, c = q - tap * C;
        const int64_t p = p0 + r, oy = p / OW, ox = p - oy * OW;
        const int64_t y = oy * stride - pad + tap / k, x = ox * stride - pad + tap % k;
        col16[i] = y >= 0 && y < H && x >= 0 && x < W ? x16[(y * W + x) * C + c] : (uint16_t) 0;
    });
    finish(stream, e, "aux_im2col16");
}

void aux_dwconv3_add(const float* x, const float* w, const float* b, int64_t H, int64_t W, int64_t C, float* y,
                     void* stream) {
    const int64_t n = H * W * C;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, p = i / C, py = p / W, px = p - py * W;
        float s = b[c];
        for (int64_t dy = -1; dy <= 1; ++dy) {
            const int64_t yy = py + dy;
            if (yy < 0 || yy >= H) continue;
            for (int64_t dx = -1; dx <= 1; ++dx) {
                const int64_t xx = px + dx;
                if (xx < 0 || xx >= W) continue;
                s += w[c * 9 + (dy + 1) * 3 + dx + 1] * x[(yy * W + xx) * C + c];
            }
        }
        y[i] = x[i] + s;
    });
    finish(stream, e, "aux_dwconv3_add");
}

void aux_layer_norm(const float* x, const float* r, const float* w, const float* b, int64_t T, int64_t C, float eps,
                    float* y, uint16_t* y16, void* stream) {
    if (T <= 0) return;
    constexpr size_t L = 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) T * L), sycl::range<1>(L)), [=](sycl::nd_item<1> it) {
            const auto t = static_cast<int64_t>(it.get_group_linear_id());
            const auto l = static_cast<int64_t>(it.get_local_linear_id());
            const float* xr = x + t * C;
            const float* rr = r != nullptr ? r + t * C : nullptr;
            float s = 0.f;
            for (int64_t c = l; c < C; c += (int64_t) L) s += xr[c] + (rr ? rr[c] : 0.f);
            const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
            float s2 = 0.f;   // two passes: the variance about the mean (the residual streams' large values)
            for (int64_t c = l; c < C; c += (int64_t) L) {
                const float d = xr[c] + (rr ? rr[c] : 0.f) - mean;
                s2 += d * d;
            }
            const float inv =
                1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) C + eps);
            for (int64_t c = l; c < C; c += (int64_t) L) {
                const float v = (xr[c] + (rr ? rr[c] : 0.f) - mean) * inv * w[c] + b[c];
                if (y16 != nullptr) y16[t * C + c] = f16_from_f32(v);
                if (y != nullptr) y[t * C + c] = v;
            }
        });
    finish(stream, e, "aux_layer_norm");
}

void aux_window16(const uint16_t* src16, int64_t H, int64_t W, int64_t C, int64_t ws, int64_t shift, int64_t Hp,
                  int64_t Wp, uint16_t* dst16, void* stream) {
    const int64_t n = Hp * Wp * C;
    if (n <= 0) return;
    const int64_t nwx = Wp / ws;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, t = i / C, win = t / (ws * ws), r = t - win * ws * ws;
        const int64_t y = ((win / nwx) * ws + r / ws + shift) % Hp, x = ((win % nwx) * ws + r % ws + shift) % Wp;
        dst16[i] = y < H && x < W ? src16[(y * W + x) * C + c] : (uint16_t) 0;
    });
    finish(stream, e, "aux_window16");
}

void aux_unwindow_add(float* x, const float* y, int64_t H, int64_t W, int64_t C, int64_t ws, int64_t shift, int64_t Hp,
                      int64_t Wp, void* stream) {
    const int64_t n = Hp * Wp * C;
    if (n <= 0) return;
    const int64_t nwx = Wp / ws;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, t = i / C, win = t / (ws * ws), r = t - win * ws * ws;
        const int64_t py = ((win / nwx) * ws + r / ws + shift) % Hp, px = ((win % nwx) * ws + r % ws + shift) % Wp;
        if (py < H && px < W) x[(py * W + px) * C + c] += y[i];
    });
    finish(stream, e, "aux_unwindow_add");
}

void aux_channel_split(const float* qkv, int64_t N, int64_t C, int64_t G, float scale, uint16_t* qT16, uint16_t* kT16,
                       uint16_t* v16, void* stream) {
    const int64_t n = N * C;
    if (n <= 0) return;
    const int64_t Cg = C / G;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        // i over [G][Cg][N]: the transposed operands' writes in order
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i % N, gc = i / N, g = gc / Cg, c = gc - g * Cg;
        const float* row = qkv + t * 3 * C;
        qT16[i] = f16_from_f32(row[g * Cg + c] * scale);
        kT16[i] = f16_from_f32(row[C + g * Cg + c]);
        v16[(g * N + t) * Cg + c] = f16_from_f32(row[2 * C + g * Cg + c]);
    });
    finish(stream, e, "aux_channel_split");
}

void aux_softmax16(const float* s, int64_t rows, int64_t cols, uint16_t* p16, void* stream) {
    if (rows <= 0) return;
    constexpr size_t L = 128;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * L), sycl::range<1>(L)), [=](sycl::nd_item<1> it) {
            const auto r = static_cast<int64_t>(it.get_group_linear_id());
            const auto l = static_cast<int64_t>(it.get_local_linear_id());
            const float* x = s + r * cols;
            float m = -INFINITY;
            for (int64_t c = l; c < cols; c += (int64_t) L) m = sycl::fmax(m, x[c]);
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            float z = 0.f;
            for (int64_t c = l; c < cols; c += (int64_t) L) z += sycl::exp(x[c] - m);
            z = sycl::reduce_over_group(it.get_group(), z, sycl::plus<float>());
            const float inv = 1.f / z;
            for (int64_t c = l; c < cols; c += (int64_t) L) p16[r * cols + c] = f16_from_f32(sycl::exp(x[c] - m) * inv);
        });
    finish(stream, e, "aux_softmax16");
}

void aux_embed(const int32_t* ids, const int32_t* pos_ids, const uint16_t* tok, const uint16_t* pos, int64_t n,
               int64_t C, float scale, float* y, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / C, c = i - t * C;
        const float p = f32_from_f16(pos[(int64_t) pos_ids[t] * C + c]);
        y[i] = tok != nullptr ? f32_from_f16(tok[(int64_t) ids[t] * C + c]) * scale + p : y[i] + p;
    });
    finish(stream, e, "aux_embed");
}

void aux_split3_16(const float* x, int64_t T, int64_t C, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream) {
    if (T <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / C, c = i - t * C;
        q16[i] = f16_from_f32(x[t * 3 * C + c]);
        k16[i] = f16_from_f32(x[t * 3 * C + C + c]);
        v16[i] = f16_from_f32(x[t * 3 * C + 2 * C + c]);
    });
    finish(stream, e, "aux_split3_16");
}

void aux_cache_put(const uint16_t* src16, int64_t B, int64_t ld, int64_t off, int64_t C, uint16_t* cache, int64_t cap,
                   int64_t at, void* stream) {
    if (B <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (B * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t b = i / C, c = i - b * C;
        cache[(b * cap + at) * C + c] = src16[b * ld + off + c];
    });
    finish(stream, e, "aux_cache_put");
}

void aux_cache_gather(const uint16_t* src, const int32_t* from, int64_t B, int64_t cap, int64_t len, int64_t C,
                      uint16_t* dst, void* stream) {
    if (B <= 0 || len <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (B * len * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, t = (i / C) % len, b = i / (C * len);
        dst[(b * cap + t) * C + c] = src[((int64_t) from[b] * cap + t) * C + c];
    });
    finish(stream, e, "aux_cache_gather");
}

void aux_polygon_fill(const float* xy, const int32_t* starts, int64_t n, int64_t W, int64_t H, uint8_t* mask,
                      void* stream) {
    if (W <= 0 || H <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (W * H)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t row = i / W;
        const float px = (float) (i - row * W), py = (float) row;
        bool on = false;
        for (int64_t p = 0; p < n && !on; ++p) {
            const int64_t a = starts[p], z = starts[p + 1];
            if (z - a < 3) continue;
            bool in = false;
            for (int64_t v = a, u = z - 1; v < z; u = v++) {
                const float x0 = xy[2 * u], y0 = xy[2 * u + 1], x1 = xy[2 * v], y1 = xy[2 * v + 1];
                // even-odd crossing of the row through the pixel's center
                if ((y1 > py) != (y0 > py) && px < (x0 - x1) * (py - y1) / (y0 - y1) + x1) in = !in;
                // the outline: within half a pixel of the edge
                const float ex = x0 - x1, ey = y0 - y1, l2 = ex * ex + ey * ey;
                float tt = l2 > 0.f ? ((px - x1) * ex + (py - y1) * ey) / l2 : 0.f;
                tt = sycl::clamp(tt, 0.f, 1.f);
                const float dx = x1 + tt * ex - px, dy = y1 + tt * ey - py;
                if (dx * dx + dy * dy <= 0.25f) in = true, on = true;
            }
            on = on || in;
        }
        mask[i] = on ? (uint8_t) 255 : (uint8_t) 0;
    });
    finish(stream, e, "aux_polygon_fill");
}

void aux_relu(const float* x, int64_t n, float* y, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const float v = sycl::fmax(x[i], 0.f);
        if (y != nullptr) y[i] = v;
        if (y16 != nullptr) y16[i] = f16_from_f32(v);
    });
    finish(stream, e, "aux_relu");
}

void aux_add16(const float* a, const float* b, int64_t n, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        y16[i] = f16_from_f32(a[i] + b[i]);
    });
    finish(stream, e, "aux_add16");
}

void aux_tile_add(float* x, const float* table, int64_t H, int64_t W, int64_t C, int64_t th, int64_t tw, void* stream) {
    const int64_t n = H * W * C;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, p = i / C, py = p / W, px = p - py * W;
        x[i] += table[((py % th) * tw + px % tw) * C + c];
    });
    finish(stream, e, "aux_tile_add");
}

void aux_rope2d_split16(const float* qkv, int64_t N, int64_t heads, int64_t hd, int64_t grid, float pscale, float theta,
                        uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream) {
    const int64_t C = heads * hd, half = hd / 2, quarter = hd / 4;
    if (N <= 0 || half <= 0) return;
    const float lt = sycl::log(theta);   // f_i = exp(-4 i / hd * ln theta)
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (N * heads * half)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i % half, th = i / half, t = th / heads, h = th - t * heads;
        const int64_t li = t % (grid * grid), px = li % grid, py = li / grid;
        const int64_t f = p < quarter ? p : p - quarter;
        const float pos = (float) (p < quarter ? px : py) * pscale;
        const float ang = pos * sycl::exp(-(float) (4 * f) / (float) hd * lt);
        const float cs = sycl::cos(ang), sn = sycl::sin(ang);
        const float* row = qkv + t * 3 * C + h * hd + 2 * p;
        const int64_t o = t * C + h * hd + 2 * p;
        const float q0 = row[0], q1 = row[1], k0 = row[C], k1 = row[C + 1];
        q16[o] = f16_from_f32(q0 * cs - q1 * sn);
        q16[o + 1] = f16_from_f32(q1 * cs + q0 * sn);
        k16[o] = f16_from_f32(k0 * cs - k1 * sn);
        k16[o + 1] = f16_from_f32(k1 * cs + k0 * sn);
        v16[o] = f16_from_f32(row[2 * C]);
        v16[o + 1] = f16_from_f32(row[2 * C + 1]);
    });
    finish(stream, e, "aux_rope2d_split16");
}

void aux_deconv2_scatter(const float* y, int64_t H, int64_t W, int64_t O, const float* bias, bool gelu, float* out,
                         uint16_t* out16, void* stream) {
    const int64_t n = 4 * H * W * O;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        // i over out's [2H][2W][O]
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t o = i % O, p = i / O, oy = p / (2 * W), ox = p - oy * (2 * W);
        const int64_t src = ((oy / 2) * W + ox / 2) * 4 * O + ((oy % 2) * 2 + ox % 2) * O + o;
        float v = y[src] + (bias != nullptr ? bias[o] : 0.f);
        if (gelu) v = 0.5f * v * (1.f + sycl::erf(v * 0.70710678118654752f));
        if (out != nullptr) out[i] = v;
        if (out16 != nullptr) out16[i] = f16_from_f32(v);
    });
    finish(stream, e, "aux_deconv2_scatter");
}

void aux_up_add16(const float* prev, int64_t h, int64_t w, const float* feat, int64_t H, int64_t W, int64_t C,
                  uint16_t* y16, void* stream) {
    const int64_t n = H * W * C;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, p = i / C, py = p / W, px = p - py * W;
        const int64_t sy = sycl::min(py * h / H, h - 1), sx = sycl::min(px * w / W, w - 1);
        y16[i] = f16_from_f32(feat[i] + prev[(sy * w + sx) * C + c]);
    });
    finish(stream, e, "aux_up_add16");
}

void aux_group_norm(const float* x, int64_t N, int64_t C, int64_t G, const float* w, const float* b, float eps,
                    bool relu, float* y, uint16_t* y16, float* part, void* stream) {
    if (N <= 0 || G <= 0) return;
    constexpr int64_t P = 64;      // chunks of the pixels a group's statistics are gathered over
    constexpr size_t L = 256;
    const int64_t Cg = C / G, chunk = (N + P - 1) / P;
    auto& q = queue_for(stream);
    // each chunk's count, mean and squared deviations about it (two passes over its values)
    auto e = q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (G * P) * L), sycl::range<1>(L)),
                            [=](sycl::nd_item<1> it) {
        const auto gid = static_cast<int64_t>(it.get_group_linear_id());
        const auto l = static_cast<int64_t>(it.get_local_linear_id());
        const int64_t g = gid / P, ch = gid - g * P, p0 = ch * chunk, p1 = sycl::min(N, p0 + chunk);
        const int64_t cnt = p1 > p0 ? (p1 - p0) * Cg : 0;
        float s = 0.f;
        for (int64_t k = l; k < cnt; k += (int64_t) L) s += x[(p0 + k / Cg) * C + g * Cg + k % Cg];
        const float tot = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float mean = cnt > 0 ? tot / (float) cnt : 0.f;
        float s2 = 0.f;
        for (int64_t k = l; k < cnt; k += (int64_t) L) {
            const float d = x[(p0 + k / Cg) * C + g * Cg + k % Cg] - mean;
            s2 += d * d;
        }
        const float m2 = sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>());
        if (l == 0) {
            part[3 * gid] = (float) cnt;
            part[3 * gid + 1] = mean;
            part[3 * gid + 2] = m2;
        }
    });
    if (!stream) core::Runtime::get().wait(e, "aux_group_norm");
    const int64_t n = N * C;
    e = q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, g = c / Cg;
        // the chunks' statistics combined (Chan et al.'s pairwise update)
        float cnt = 0.f, mean = 0.f, m2 = 0.f;
        for (int64_t ch = 0; ch < P; ++ch) {
            const float nb = part[3 * (g * P + ch)];
            if (nb <= 0.f) continue;
            const float mb = part[3 * (g * P + ch) + 1], d = mb - mean, tot = cnt + nb;
            mean += d * nb / tot;
            m2 += part[3 * (g * P + ch) + 2] + d * d * cnt * nb / tot;
            cnt = tot;
        }
        const float inv = 1.f / sycl::sqrt(m2 / cnt + eps);
        float v = (x[i] - mean) * inv * w[c] + b[c];
        if (relu) v = sycl::fmax(v, 0.f);
        if (y != nullptr) y[i] = v;
        if (y16 != nullptr) y16[i] = f16_from_f32(v);
    });
    finish(stream, e, "aux_group_norm");
}

void aux_mask_resize(const float* logits, int64_t K, int64_t h, int64_t w, int64_t H, int64_t W, float threshold,
                     uint8_t* masks, void* stream) {
    const int64_t n = K * H * W;
    if (n <= 0) return;
    const float sy = (float) h / (float) H, sx = (float) w / (float) W;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t k = i / (H * W), p = i - k * H * W, py = p / W, px = p - py * W;
        const float fy = sycl::fmax(((float) py + 0.5f) * sy - 0.5f, 0.f);
        const float fx = sycl::fmax(((float) px + 0.5f) * sx - 0.5f, 0.f);
        const int64_t y0 = sycl::min((int64_t) fy, h - 1), x0 = sycl::min((int64_t) fx, w - 1);
        const int64_t y1 = sycl::min(y0 + 1, h - 1), x1 = sycl::min(x0 + 1, w - 1);
        const float ay = fy - (float) y0, ax = fx - (float) x0;
        const float* l = logits + k * h * w;
        auto sg = [](float v) { return 1.f / (1.f + sycl::exp(-v)); };
        const float v = (1.f - ay) * ((1.f - ax) * sg(l[y0 * w + x0]) + ax * sg(l[y0 * w + x1])) +
                        ay * ((1.f - ax) * sg(l[y1 * w + x0]) + ax * sg(l[y1 * w + x1]));
        masks[i] = v > threshold ? (uint8_t) 255 : (uint8_t) 0;
    });
    finish(stream, e, "aux_mask_resize");
}

void aux_box_bias(const float* boxes, int64_t Q, int64_t H, int64_t W, const float* mlp, int64_t hidden, int64_t heads,
                  float* tmp, float* bias, void* stream) {
    if (Q <= 0) return;
    auto& q = queue_for(stream);
    const int64_t R = H + W, per = hidden * 2 + hidden + heads * hidden + heads;
    // tmp [Q][R][heads]: the rows' MLP (i < H) and the columns' (H + j)
    auto e = q.parallel_for(sycl::range<1>((size_t) (Q * R)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t qq = i / R, r = i - qq * R;
        const bool rows = r < H;
        const float c = rows ? (float) r / (float) H : (float) (r - H) / (float) W;
        const float* bx = boxes + qq * 4;
        float d[2] = {c - (rows ? bx[1] : bx[0]), c - (rows ? bx[3] : bx[2])};
        for (float& v : d) {
            v *= 8.f;
            v = (v > 0.f ? 1.f : v < 0.f ? -1.f : 0.f) * sycl::log2(sycl::fabs(v) + 1.f) / 3.f;
        }
        const float* m = mlp + (rows ? 0 : per);
        const float *w1 = m, *b1 = m + hidden * 2, *w2 = b1 + hidden, *b2 = w2 + heads * hidden;
        float out[16];
        for (int64_t o = 0; o < heads && o < 16; ++o) out[o] = b2[o];
        for (int64_t k = 0; k < hidden; ++k) {
            const float a = sycl::fmax(w1[2 * k] * d[0] + w1[2 * k + 1] * d[1] + b1[k], 0.f);
            for (int64_t o = 0; o < heads && o < 16; ++o) out[o] += w2[o * hidden + k] * a;
        }
        for (int64_t o = 0; o < heads && o < 16; ++o) tmp[i * heads + o] = out[o];
    });
    if (!stream) core::Runtime::get().wait(e, "aux_box_bias");
    const int64_t HW = H * W, n = heads * (Q + 1) * HW;
    e = q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i % HW, hq = i / HW, row = hq % (Q + 1), h = hq / (Q + 1);
        if (row == 0) {
            bias[i] = 0.f;
            return;
        }
        const int64_t qq = row - 1, py = p / W, px = p - py * W;
        bias[i] = tmp[(qq * R + py) * heads + h] + tmp[(qq * R + H + px) * heads + h];
    });
    finish(stream, e, "aux_box_bias");
}

void aux_merge2(const float* x, int64_t H, int64_t W, int64_t C, float* y, void* stream) {
    const int64_t h = H / 2, w = W / 2, n = h * w * 4 * C;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c4 = i % (4 * C), p = i / (4 * C), part = c4 / C, c = c4 - part * C;
        const int64_t py = p / w, px = p - py * w;
        const int64_t sy = 2 * py + part % 2, sx = 2 * px + part / 2;   // x0 (0, 0), x1 (1, 0), x2 (0, 1), x3 (1, 1)
        y[i] = x[(sy * W + sx) * C + c];
    });
    finish(stream, e, "aux_merge2");
}

void aux_resize_into(const float* src, int64_t h, int64_t w, int64_t C, float* dst, int64_t H, int64_t W, int64_t ld,
                     int64_t c0, bool align_corners, bool add, void* stream) {
    const int64_t n = H * W * C;
    if (n <= 0) return;
    const bool same = h == H && w == W;
    // the source coordinate of a destination pixel: align_corners maps the corners' centers onto each other
    const float ry = align_corners ? (H > 1 ? (float) (h - 1) / (float) (H - 1) : 0.f) : (float) h / (float) H;
    const float rx = align_corners ? (W > 1 ? (float) (w - 1) / (float) (W - 1) : 0.f) : (float) w / (float) W;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, p = i / C, py = p / W, px = p - py * W;
        float v;
        if (same) {
            v = src[i];
        } else {
            const float fy = align_corners ? (float) py * ry : sycl::fmax(((float) py + 0.5f) * ry - 0.5f, 0.f);
            const float fx = align_corners ? (float) px * rx : sycl::fmax(((float) px + 0.5f) * rx - 0.5f, 0.f);
            const int64_t y0 = sycl::min((int64_t) fy, h - 1), x0 = sycl::min((int64_t) fx, w - 1);
            const int64_t y1 = sycl::min(y0 + 1, h - 1), x1 = sycl::min(x0 + 1, w - 1);
            const float ay = fy - (float) y0, ax = fx - (float) x0;
            v = (1.f - ay) * ((1.f - ax) * src[(y0 * w + x0) * C + c] + ax * src[(y0 * w + x1) * C + c]) +
                ay * ((1.f - ax) * src[(y1 * w + x0) * C + c] + ax * src[(y1 * w + x1) * C + c]);
        }
        float& d = dst[p * ld + c0 + c];
        d = add ? d + v : v;
    });
    finish(stream, e, "aux_resize_into");
}

void aux_deform_col16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int k, int pad, const float* om,
                      int64_t p0, int64_t n, uint16_t* col16, void* stream) {
    const int64_t k2 = (int64_t) k * k, K = k2 * C;
    if (n <= 0 || K <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n * K)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / K, q = i - r * K, tap = q / C, c = q - tap * C;
        const int64_t p = p0 + r, oy = p / W, ox = p - oy * W;
        const float* o = om + p * 3 * k2;
        const int64_t ty = tap / k, tx = tap - ty * k;   // the tap's row and column in the kernel
        const float y = (float) (oy - pad + ty) + o[2 * tap], x = (float) (ox - pad + tx) + o[2 * tap + 1];
        const float m = 2.f / (1.f + sycl::exp(-o[2 * k2 + tap]));
        float v = 0.f;
        // torchvision's bilinear_interpolate: nothing outside (-1, H) x (-1, W), the corners outside it zero
        if (y > -1.f && y < (float) H && x > -1.f && x < (float) W) {
            const float fy = sycl::floor(y), fx = sycl::floor(x);
            const int64_t yl = (int64_t) fy, xl = (int64_t) fx, yh = yl + 1, xh = xl + 1;
            const float ly = y - fy, lx = x - fx, hy = 1.f - ly, hx = 1.f - lx;
            auto at = [&](int64_t yy, int64_t xx) {
                return yy >= 0 && yy < H && xx >= 0 && xx < W ? f32_from_f16(x16[(yy * W + xx) * C + c]) : 0.f;
            };
            v = hy * hx * at(yl, xl) + hy * lx * at(yl, xh) + ly * hx * at(yh, xl) + ly * lx * at(yh, xh);
        }
        col16[i] = f16_from_f32(v * m);
    });
    finish(stream, e, "aux_deform_col16");
}

void aux_bias_act(float* y, int64_t rows, int64_t n, int64_t ld, const float* bias, bool relu, void* stream) {
    if (rows <= 0 || n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (rows * n)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / n, c = i - r * n;
        float& v = y[r * ld + c];
        float t = v + (bias != nullptr ? bias[c] : 0.f);
        if (relu) t = sycl::fmax(t, 0.f);
        v = t;
    });
    finish(stream, e, "aux_bias_act");
}

void aux_col_mean(const float* x, int64_t P, int64_t C, float* mean, void* stream) {
    if (P <= 0 || C <= 0) return;
    constexpr size_t L = 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) C * L), sycl::range<1>(L)), [=](sycl::nd_item<1> it) {
            const auto c = static_cast<int64_t>(it.get_group_linear_id());
            const auto l = static_cast<int64_t>(it.get_local_linear_id());
            float s = 0.f;
            for (int64_t p = l; p < P; p += (int64_t) L) s += x[p * C + c];
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            if (l == 0) mean[c] = s / (float) P;
        });
    finish(stream, e, "aux_col_mean");
}

void aux_scale_by_sigmoid(float* x, int64_t P, int64_t C, const float* logit, void* stream) {
    if (P <= 0 || C <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (P * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        x[i] *= 1.f / (1.f + sycl::exp(-logit[i / C]));
    });
    finish(stream, e, "aux_scale_by_sigmoid");
}

void aux_image_patches16(const float* img, int64_t S, int64_t g, uint16_t* y16, void* stream) {
    const int64_t s = S / g, ch = 3 * g * g, n = s * s * ch;
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t q = i % ch, p = i / ch, py = p / s, px = p - py * s;
        const int64_t c = q / (g * g), gy = (q / g) % g, gx = q % g;
        y16[i] = f16_from_f32(img[((gy * s + py) * S + gx * s + px) * 3 + c]);
    });
    finish(stream, e, "aux_image_patches16");
}

void aux_rgb_norm(const uint8_t* rgba, int64_t W, int64_t H, const float* mean, const float* std, float* y,
                  void* stream) {
    const int64_t n = W * H;
    if (n <= 0) return;
    const float m0 = mean[0], m1 = mean[1], m2 = mean[2], s0 = 1.f / std[0], s1 = 1.f / std[1], s2 = 1.f / std[2];
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        y[3 * i] = ((float) rgba[4 * i] / 255.f - m0) * s0;
        y[3 * i + 1] = ((float) rgba[4 * i + 1] / 255.f - m1) * s1;
        y[3 * i + 2] = ((float) rgba[4 * i + 2] / 255.f - m2) * s2;
    });
    finish(stream, e, "aux_rgb_norm");
}

void aux_alpha_bicubic(const float* logits, int64_t h, int64_t w, int64_t W, int64_t H, uint8_t* alpha, void* stream) {
    const int64_t n = W * H;
    if (n <= 0) return;
    const float sy = (float) h / (float) H, sx = (float) w / (float) W;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t py = i / W, px = i - py * W;
        // torch's upsample_bicubic2d: the source coordinate unclamped, the four taps' indices clamped to the edge
        const float fy = ((float) py + 0.5f) * sy - 0.5f, fx = ((float) px + 0.5f) * sx - 0.5f;
        const float by = sycl::floor(fy), bx = sycl::floor(fx), ty = fy - by, tx = fx - bx;
        auto cubic = [](float t, float* wt) {
            constexpr float A = -0.75f;
            auto c1 = [](float x) { return ((A + 2.f) * x - (A + 3.f)) * x * x + 1.f; };
            auto c2 = [](float x) { return ((A * x - 5.f * A) * x + 8.f * A) * x - 4.f * A; };
            wt[0] = c2(t + 1.f), wt[1] = c1(t), wt[2] = c1(1.f - t), wt[3] = c2(2.f - t);
        };
        float wy[4], wx[4];
        cubic(ty, wy);
        cubic(tx, wx);
        float v = 0.f;
        for (int a = 0; a < 4; ++a) {
            const int64_t yy = sycl::clamp((int64_t) by - 1 + a, (int64_t) 0, h - 1);
            float r = 0.f;
            for (int b = 0; b < 4; ++b) {
                const int64_t xx = sycl::clamp((int64_t) bx - 1 + b, (int64_t) 0, w - 1);
                r += wx[b] * logits[yy * w + xx];
            }
            v += wy[a] * r;
        }
        alpha[i] = (uint8_t) sycl::round(255.f / (1.f + sycl::exp(-v)));
    });
    finish(stream, e, "aux_alpha_bicubic");
}

}  // namespace strata::image
