// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 1997-2011 Secret Labs AB
// SPDX-FileCopyrightText: 1995-2011 Fredrik Lundh and contributors
// SPDX-FileCopyrightText: 2010 Jeffrey A. Clark and contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT-CMU
// src/image/img_crop.cpp - see include/strata/image/img_crop.hpp.  The host resize (pillow_resize_planes) is
// transcribed from Pillow's libImaging/Resample.c (https://github.com/python-pillow/Pillow): that code stays under
// Pillow's license (MIT-CMU), the file as a whole is under the LGPL.

#include "strata/image/img_crop.hpp"

#include "strata/core/gpu.hpp"
#include "strata/image/img_color.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/cpu/resize_rows.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace strata::image {

namespace {

sycl::queue& queue() { return core::Runtime::get().stream(nullptr); }

void done(const sycl::event& e, const char* what) { core::Runtime::get().wait(e, what); }

// the device buffers of one call, freed with it
struct Buffers {
    std::vector<void*> held;
    ~Buffers() {
        for (void* p : held) gpu::free(p);
    }
    template <class T> T* get(size_t n) {
        T* p = nullptr;
        if (!gpu::alloc_device(&p, n * sizeof(T)))
            throw std::runtime_error(std::string("image engine: GPU memory (the cut and its return): ") + gpu::last_error());
        held.push_back(p);
        return p;
    }
    template <class T> T* put(const std::vector<T>& v) {
        T* p = get<T>(v.size());
        gpu::copy(p, v.data(), v.size() * sizeof(T));
        return p;
    }
};

// Pillow's resampling weights (precompute_coeffs) from `in` samples to `out`: each output's first input `lo` and its
// `k` weights, the filter's support widened by the scale when it shrinks; Lanczos (a = 3), the triangle (bilinear),
// the cubic (bicubic, a = -0.5) or OpenCV's cubic (cv_cubic, a = -0.75): the weights divided by their sum, as floats
struct Taps {
    std::vector<int32_t> lo;
    std::vector<float> w;
    int k = 0;
};

double sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= 3.14159265358979323846;
    return std::sin(x) / x;
}

// Pillow's bicubic_filter
double cubic(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

// OpenCV's cubic (a = -0.75)
double cv_cubic(double x) {
    constexpr double a = -0.75;
    x = std::abs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

Taps taps(int64_t in, int64_t out, Filter f) {
    const double scale = (double) in / (double) out, fs = std::max(scale, 1.0);
    if (f == Filter::cv_cubic) {
        // the 4 samples around (x + 0.5) scale - 0.5, an index past an edge taken as the edge's (its weight
        // joining the edge's slot, which is what repeating the edge's sample means)
        Taps t;
        t.k = 4;
        t.lo.resize((size_t) out);
        t.w.assign((size_t) (out * t.k), 0.0f);
        for (int64_t x = 0; x < out; ++x) {
            const double c = ((double) x + 0.5) * scale - 0.5;
            const auto i0 = (int64_t) std::floor(c);
            const double fr = c - (double) i0;
            const int64_t lo = std::clamp<int64_t>(i0 - 1, 0, std::max<int64_t>(in - 4, 0));
            t.lo[(size_t) x] = (int32_t) lo;
            for (int64_t j = 0; j < 4; ++j) {
                const int64_t at = std::clamp<int64_t>(i0 - 1 + j, 0, in - 1);
                const int64_t slot = std::clamp<int64_t>(at - lo, 0, 3);
                t.w[(size_t) (x * 4 + slot)] += (float) cv_cubic(fr + 1.0 - (double) j);
            }
        }
        return t;
    }
    const double support = (f == Filter::lanczos ? 3.0 : f == Filter::bicubic ? 2.0 : 1.0) * fs;
    Taps t;
    t.k = (int) std::ceil(support) * 2 + 1;
    t.lo.resize((size_t) out);
    t.w.assign((size_t) (out * t.k), 0.0f);
    std::vector<double> ww((size_t) t.k);
    for (int64_t x = 0; x < out; ++x) {
        const double center = ((double) x + 0.5) * scale;
        // Pillow's (int) casts: its truncation, floor here (a negative start is 0 either way)
        const int64_t lo = std::max<int64_t>((int64_t) std::floor(center - support + 0.5), 0);
        const int64_t hi = std::min<int64_t>((int64_t) std::floor(center + support + 0.5), in);
        double sum = 0.0;
        for (int64_t j = 0; j < hi - lo; ++j) {
            const double u = ((double) (j + lo) - center + 0.5) / fs;
            const double v = f == Filter::lanczos ? (u > -3.0 && u < 3.0 ? sinc(u) * sinc(u / 3.0) : 0.0)
                             : f == Filter::bicubic ? cubic(u)
                                                    : std::max(0.0, 1.0 - std::abs(u));
            ww[(size_t) j] = v;
            sum += v;
        }
        t.lo[(size_t) x] = (int32_t) lo;
        for (int64_t j = 0; j < hi - lo; ++j)
            t.w[(size_t) (x * t.k + j)] = (float) (sum != 0.0 ? ww[(size_t) j] / sum : 0.0);
    }
    return t;
}

// The edit share before its growth, on the host (a flood is a walk, not a GPU's work): 1 - alpha / 255, its strokes
// fainter than kFaint left out, then its holes filled in grey: a pixel's fill is the lowest level a 4-connected way from
// outside the mask's box to it has to climb over (a flood from a frame around the box, lowest first), and the pixel
// takes the higher of its share and its fill - a hole takes the level of what encloses it, so a ring painted at 0.97
// redraws its middle at 0.97, and a binary mask stays binary.  The fill stays inside the mask's box, so the server's
// box (serve/images.py, crop_reach) need not count it.
constexpr float kFaint = 0.05f;

std::vector<float> edit_share(const QiRef& mask) {
    const int64_t W = mask.width, H = mask.height;
    std::vector<float> m((size_t) (W * H));
    int64_t x0 = W, y0 = H, x1 = -1, y1 = -1;   // the mask's box
    for (int64_t i = 0; i < W * H; ++i) {
        const float v = 1.0f - (float) mask.rgba[(size_t) (i * 4 + 3)] / 255.0f;
        m[(size_t) i] = v < kFaint ? 0.0f : v;
        if (v >= kFaint) {
            x0 = std::min(x0, i % W), x1 = std::max(x1, i % W);
            y0 = std::min(y0, i / W), y1 = std::max(y1, i / W);
        }
    }
    if (x1 - x0 < 2 || y1 - y0 < 2) return m;   // no room for a hole
    // the box on a frame of one cell a side, the frame's cells the flood's sources at level 0
    const int64_t gw = x1 - x0 + 3, gh = y1 - y0 + 3;
    std::vector<float> level((size_t) (gw * gh), 2.0f);
    const auto share = [&](int64_t i) {
        const int64_t x = i % gw + x0 - 1, y = i / gw + y0 - 1;
        return x >= x0 && x <= x1 && y >= y0 && y <= y1 ? m[(size_t) (y * W + x)] : 0.0f;
    };
    using Step = std::pair<float, int64_t>;   // a cell's level so far, the cell
    std::priority_queue<Step, std::vector<Step>, std::greater<Step>> open;
    for (int64_t i = 0; i < gw * gh; ++i)
        if (i % gw == 0 || i / gw == 0 || i % gw == gw - 1 || i / gw == gh - 1) {
            level[(size_t) i] = 0.0f;
            open.emplace(0.0f, i);
        }
    while (!open.empty()) {
        const auto [lv, i] = open.top();
        open.pop();
        if (lv > level[(size_t) i]) continue;   // reached lower already
        const int64_t x = i % gw, y = i / gw;
        const int64_t next[4] = {x > 0 ? i - 1 : -1, x + 1 < gw ? i + 1 : -1, y > 0 ? i - gw : -1,
                                 y + 1 < gh ? i + gw : -1};
        for (const int64_t j : next) {
            if (j < 0) continue;
            const float up = std::max(lv, share(j));
            if (up < level[(size_t) j]) {
                level[(size_t) j] = up;
                open.emplace(up, j);
            }
        }
    }
    for (int64_t y = y0; y <= y1; ++y)
        for (int64_t x = x0; x <= x1; ++x) {
            float& v = m[(size_t) (y * W + x)];
            v = std::max(v, level[(size_t) ((y - y0 + 1) * gw + x - x0 + 1)]);
        }
    return m;
}

// The mask grown by a disc of radius `r` (each pixel the largest value within r of it, the picture's own pixels only)
void grow_disc(float* m, float* tmp, int64_t W, int64_t H, int64_t r) {
    done(queue().parallel_for(sycl::range<2>((size_t) H, (size_t) W), [=](sycl::id<2> id) {
        const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
        float v = 0.0f;
        for (int64_t dy = -r; dy <= r; ++dy) {
            const int64_t yy = y + dy;
            if (yy < 0 || yy >= H) continue;
            // the disc's half width on this row
            const int64_t span = (int64_t) sycl::floor(sycl::sqrt((float) (r * r - dy * dy)));
            for (int64_t xx = sycl::max(x - span, (int64_t) 0); xx <= sycl::min(x + span, W - 1); ++xx)
                v = sycl::fmax(v, m[yy * W + xx]);
        }
        tmp[y * W + x] = v;
    }), "crop: grow");
    done(queue().memcpy(m, tmp, (size_t) (W * H) * sizeof(float)), "crop: grow");
}

// The mask softened by three passes of a box mean of 2 r + 1 pixels each way (nearly a Gaussian of sigma about r; its
// reach 3 r), the picture's edge pixels repeated past it, then kept within [0, 1]
void soften(float* m, float* tmp, int64_t W, int64_t H, int64_t r) {
    const float inv = 1.0f / (float) (2 * r + 1);
    for (int pass = 0; pass < 3; ++pass) {
        done(queue().parallel_for(sycl::range<2>((size_t) H, (size_t) W), [=](sycl::id<2> id) {
            const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
            float v = 0.0f;
            for (int64_t d = -r; d <= r; ++d) v += m[y * W + sycl::clamp(x + d, (int64_t) 0, W - 1)];
            tmp[y * W + x] = v * inv;
        }), "crop: soften rows");
        done(queue().parallel_for(sycl::range<2>((size_t) H, (size_t) W), [=](sycl::id<2> id) {
            const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
            float v = 0.0f;
            for (int64_t d = -r; d <= r; ++d) v += tmp[sycl::clamp(y + d, (int64_t) 0, H - 1) * W + x];
            m[y * W + x] = sycl::clamp(v * inv, 0.0f, 1.0f);
        }), "crop: soften columns");
    }
}

// the rows' pass of a resize: an RGBA8 picture of W x H seen through a box (x0, y0, its rows bh; its parts past the
// edges the edges' copies) to planar floats [4][bh][ow] by the taps over the box's width
void rows_rgba8(const uint8_t* src, int64_t W, int64_t H, int64_t x0, int64_t y0, int64_t bh, const Taps& t,
                const int32_t* lo, const float* w, float* out) {
    const int64_t ow = (int64_t) t.lo.size(), k = t.k;
    done(queue().parallel_for(sycl::range<2>((size_t) bh, (size_t) ow), [=](sycl::id<2> id) {
        const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
        const int64_t sy = sycl::clamp(y0 + y, (int64_t) 0, H - 1);
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int64_t j = 0; j < k; ++j) {
            const float c = w[x * k + j];
            if (c == 0.0f) continue;
            const int64_t sx = sycl::clamp(x0 + lo[x] + j, (int64_t) 0, W - 1);
            const uint8_t* p = src + (sy * W + sx) * 4;
            for (int ch = 0; ch < 4; ++ch) v[ch] += c * (float) p[ch];
        }
        for (int ch = 0; ch < 4; ++ch) out[(ch * bh + y) * ow + x] = v[ch];
    }), "crop: resize rows");
}

// the same for a mask of floats [H][W] (1 past the picture's edges, as the canvas the box reaches past has)
void rows_mask(const float* src, int64_t W, int64_t H, int64_t x0, int64_t y0, int64_t bh, const Taps& t,
               const int32_t* lo, const float* w, float* out) {
    const int64_t ow = (int64_t) t.lo.size(), k = t.k;
    done(queue().parallel_for(sycl::range<2>((size_t) bh, (size_t) ow), [=](sycl::id<2> id) {
        const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
        const int64_t sy = y0 + y;
        float v = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            const float c = w[x * k + j];
            if (c == 0.0f) continue;
            const int64_t sx = x0 + lo[x] + j;
            v += c * (sy < 0 || sy >= H || sx < 0 || sx >= W ? 1.0f : src[sy * W + sx]);
        }
        out[y * ow + x] = v;
    }), "crop: resize mask rows");
}

// the columns' pass: planar floats [C][ih][ow] to [C][oh][ow] by the taps over ih; `lo_v`, `hi_v` the values' bounds
void columns(const float* in, int64_t C, int64_t ih, int64_t ow, const Taps& t, const int32_t* lo, const float* w,
             float* out, float lo_v, float hi_v) {
    const int64_t oh = (int64_t) t.lo.size(), k = t.k;
    done(queue().parallel_for(sycl::range<3>((size_t) C, (size_t) oh, (size_t) ow), [=](sycl::id<3> id) {
        const int64_t c = (int64_t) id[0], y = (int64_t) id[1], x = (int64_t) id[2];
        float v = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            const float a = w[y * k + j];
            if (a != 0.0f) v += a * in[(c * ih + lo[y] + j) * ow + x];
        }
        out[(c * oh + y) * ow + x] = sycl::clamp(v, lo_v, hi_v);
    }), "crop: resize columns");
}

// a W x H resize of what `rows` puts in `mid` (planar [C][bh][ow]) to [C][oh][ow]
struct Resize {
    Taps tx, ty;
    int32_t *lox = nullptr, *loy = nullptr;
    float *wx = nullptr, *wy = nullptr;
    Resize(Buffers& b, int64_t iw, int64_t ih, int64_t ow, int64_t oh, bool lanczos)
        : Resize(b, iw, ih, ow, oh, lanczos ? Filter::lanczos : Filter::triangle) {}
    Resize(Buffers& b, int64_t iw, int64_t ih, int64_t ow, int64_t oh, Filter f) : tx(taps(iw, ow, f)), ty(taps(ih, oh, f)) {
        lox = b.put(tx.lo);
        wx = b.put(tx.w);
        loy = b.put(ty.lo);
        wy = b.put(ty.w);
    }
};

uint8_t* rgba8_device(Buffers& b, const std::vector<uint8_t>& v) { return b.put(v); }

}  // namespace

namespace {

// src (RGBA8, device) at w x h by Pillow's filter into dst (device); `pillow8`: the rows' pass rounded and clipped to
// 8 bits before the columns', as Pillow's resampling of an 8-bit picture does (with Bicubic's and Lanczos's negative
// lobes the float pass differs from Pillow's by up to 6 levels: Florence-2's 768 x 768 input, 2026-11-07)
void resize_device(const uint8_t* src, int64_t sw, int64_t sh, uint8_t* dst, int64_t w, int64_t h, Filter filter,
                   bool pillow8 = false) {
    if (sw <= 0 || sh <= 0 || w <= 0 || h <= 0)
        throw std::invalid_argument("image engine: a resize of an empty image, or to no size");
    Buffers b;
    const int64_t n = w * h;
    float* planes = b.get<float>((size_t) (4 * n));
    {
        Resize rz(b, sw, sh, w, h, filter);
        float* mid = b.get<float>((size_t) (4 * sh * w));
        rows_rgba8(src, sw, sh, 0, 0, sh, rz.tx, rz.lox, rz.wx, mid);
        if (pillow8)
            done(queue().parallel_for(sycl::range<1>((size_t) (4 * sh * w)), [=](sycl::id<1> id) {
                float& v = mid[id[0]];
                v = sycl::clamp(sycl::floor(v + 0.5f), 0.0f, 255.0f);
            }), "resize: rows to 8 bits");
        columns(mid, 4, sh, w, rz.ty, rz.loy, rz.wy, planes, 0.0f, 255.0f);
    }
    done(queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        for (int c = 0; c < 4; ++c) dst[i * 4 + c] = (uint8_t) sycl::floor(planes[c * n + i] + 0.5f);
    }), "resize: outputs");
}

}  // namespace

void lanczos_resize_device(const uint8_t* src, int64_t sw, int64_t sh, uint8_t* dst, int64_t w, int64_t h) {
    resize_device(src, sw, sh, dst, w, h, Filter::lanczos);
}

uint64_t lanczos_resize_bytes(int64_t sw, int64_t sh, int64_t w, int64_t h) {
    (void) sw;
    const auto taps = [](int64_t in, int64_t out) {   // a side's weights: a = 3, widened by the scale when it shrinks
        const int64_t k = 2 * (int64_t) std::ceil(3.0 * std::max(1.0, (double) in / (double) out)) + 1;
        return (uint64_t) out * (uint64_t) (k * 4 + 4);
    };
    return (uint64_t) (4 * w * h) * 4 + (uint64_t) (4 * sh * w) * 4 + taps(sw, w) + taps(sh, h);
}

QiRef lanczos_resize(const QiRef& src, int64_t w, int64_t h) {
    if (src.width <= 0 || src.height <= 0 || src.rgba.size() != (size_t) (src.width * src.height * 4) || w <= 0 ||
        h <= 0)
        throw std::invalid_argument("image engine: a resize of an empty image, or to no size");
    Buffers b;
    const uint8_t* pic = rgba8_device(b, src.rgba);
    uint8_t* out = b.get<uint8_t>((size_t) (4 * w * h));
    resize_device(pic, src.width, src.height, out, w, h, Filter::lanczos);
    QiRef r;
    r.width = w;
    r.height = h;
    r.rgba.resize((size_t) (4 * w * h));
    gpu::copy(r.rgba.data(), out, r.rgba.size());
    return r;
}

QiRef pillow_resize(const QiRef& src, int64_t w, int64_t h, Filter filter) {
    if (src.width <= 0 || src.height <= 0 || src.rgba.size() != (size_t) (src.width * src.height * 4) || w <= 0 ||
        h <= 0)
        throw std::invalid_argument("image engine: a resize of an empty image, or to no size");
    Buffers b;
    const uint8_t* pic = rgba8_device(b, src.rgba);
    uint8_t* out = b.get<uint8_t>((size_t) (4 * w * h));
    resize_device(pic, src.width, src.height, out, w, h, filter, true);
    QiRef r;
    r.width = w;
    r.height = h;
    r.rgba.resize((size_t) (4 * w * h));
    gpu::copy(r.rgba.data(), out, r.rgba.size());
    return r;
}

namespace {

// Pillow's 8-bit resample on the host (libImaging/Resample.c: precompute_coeffs, normalize_coeffs_8bpc,
// ImagingResampleHorizontal_8bpc / Vertical_8bpc, ImagingResampleInner), the arithmetic as Pillow has it: each
// output's taps the filter's values times 1 / the scale, divided by their sum in doubles, then fixed point with 22
// bits below the point (rounded half away from 0); an output the int32 total of its taps times the 8-bit samples from
// half a unit, shifted down and clipped to 0 .. 255.  The rows' pass then the columns', each skipped where its axis
// keeps its size.
constexpr int kPilBits = 32 - 8 - 2;   // Pillow's PRECISION_BITS

// Pillow's filters (bicubic_filter: a = -0.5, lanczos_filter, bilinear_filter) and their supports
double pil_filter(Filter f, double x) {
    if (f == Filter::lanczos) {
        auto sinc = [](double v) {
            if (v == 0.0) return 1.0;
            v = v * 3.14159265358979323846;
            return std::sin(v) / v;
        };
        return -3.0 <= x && x < 3.0 ? sinc(x) * sinc(x / 3) : 0.0;
    }
    if (x < 0.0) x = -x;
    if (f == Filter::triangle) return x < 1.0 ? 1.0 - x : 0.0;
    constexpr double a = -0.5;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

// one axis's taps: output o reads n[o] inputs from lo[o], its fixed-point weights k[o * stride ..] (Pillow's ksize of
// them, 0 past n[o] up to `stride`, a multiple of 16) and their int16 halves for the rows' vector copies
// (kernels/cpu/resize_rows.hpp)
struct PilTaps {
    std::vector<int32_t> lo, n, k;
    std::vector<int16_t> kh, kl;
    int64_t ksize = 0, stride = 0;
    int32_t n_max = 0;
};

PilTaps pil_taps(int64_t in_size, int64_t out_size, Filter f) {
    const double support0 = f == Filter::lanczos ? 3.0 : f == Filter::bicubic ? 2.0 : 1.0;
    double filterscale, scale;
    filterscale = scale = (double) in_size / (double) out_size;
    if (filterscale < 1.0) filterscale = 1.0;
    const double support = support0 * filterscale;
    PilTaps t;
    t.ksize = (int64_t) std::ceil(support) * 2 + 1;
    t.lo.resize((size_t) out_size);
    t.n.resize((size_t) out_size);
    t.stride = (t.ksize + 15) / 16 * 16;
    t.k.assign((size_t) (out_size * t.stride), 0);
    t.kh.assign(t.k.size(), 0);
    t.kl.assign(t.k.size(), 0);
    std::vector<double> kk((size_t) t.ksize);
    for (int64_t xx = 0; xx < out_size; ++xx) {
        const double center = ((double) xx + 0.5) * scale, ss = 1.0 / filterscale;
        double ww = 0.0;
        // Pillow's (int) casts: truncation, not rounding (lround would move the taps off Pillow's); a negative start
        // is 0 either way
        int64_t xmin = (int64_t) (center - support + 0.5);   // NOLINT(bugprone-incorrect-roundings): Pillow's
        if (xmin < 0) xmin = 0;
        int64_t xmax = (int64_t) (center + support + 0.5);   // NOLINT(bugprone-incorrect-roundings): Pillow's
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        for (int64_t x = 0; x < xmax; ++x) {
            const double w = pil_filter(f, ((double) (x + xmin) - center + 0.5) * ss);
            kk[(size_t) x] = w;
            ww += w;
        }
        for (int64_t x = 0; x < xmax; ++x) {
            if (ww != 0.0) kk[(size_t) x] /= ww;
            const double v = kk[(size_t) x] * (1 << kPilBits);
            const int32_t q = (int32_t) (v < 0 ? -0.5 + v : 0.5 + v);
            t.k[(size_t) (xx * t.stride + x)] = q;
            t.kh[(size_t) (xx * t.stride + x)] = (int16_t) (q >> 15);
            t.kl[(size_t) (xx * t.stride + x)] = (int16_t) (q & 0x7fff);
        }
        t.lo[(size_t) xx] = (int32_t) xmin;
        t.n[(size_t) xx] = (int32_t) xmax;
        t.n_max = std::max(t.n_max, (int32_t) xmax);
    }
    return t;
}

// Pillow's clip8: the total shifted down to whole levels and clipped
uint8_t pil_clip8(int32_t v) { return (uint8_t) std::clamp<int32_t>(v >> kPilBits, 0, 255); }

// The rows' pass: `lines` rows of `n_in` bytes (readable 16 bytes past the end) to rows of `n_out`: the CPU's vector
// copy (kernels/cpu/resize_rows.hpp), else four rows at a time here (a tap's weight and index read once for the four,
// the four totals side by side).  The totals are Pillow's integers either way.
void pil_rows(const uint8_t* src, int64_t lines, int64_t n_in, int64_t n_out, const PilTaps& t, uint8_t* dst) {
    namespace kc = strata::kernels::cpu;
    kc::ResizeRowsTaps v;
    v.lo = t.lo.data();
    v.n = t.n.data();
    v.k = t.k.data();
    v.kh = t.kh.data();
    v.kl = t.kl.data();
    v.stride = t.stride;
    v.n_max = t.n_max;
    if (kc::resize_rows_u8(kc::resize_rows_isa(), src, lines, n_in, n_out, v, kPilBits, dst)) return;
    constexpr int32_t half = 1 << (kPilBits - 1);
    int64_t l = 0;
    for (; l + 4 <= lines; l += 4) {
        const uint8_t* r0 = src + (size_t) (l * n_in);
        const uint8_t *r1 = r0 + n_in, *r2 = r1 + n_in, *r3 = r2 + n_in;
        uint8_t* d0 = dst + (size_t) (l * n_out);
        for (int64_t o = 0; o < n_out; ++o) {
            const int32_t* k = t.k.data() + (size_t) (o * t.stride);
            const int64_t lo = t.lo[(size_t) o], n = t.n[(size_t) o];
            int32_t v0 = half, v1 = half, v2 = half, v3 = half;
            for (int64_t x = 0; x < n; ++x) {
                v0 += (int32_t) r0[lo + x] * k[x];
                v1 += (int32_t) r1[lo + x] * k[x];
                v2 += (int32_t) r2[lo + x] * k[x];
                v3 += (int32_t) r3[lo + x] * k[x];
            }
            d0[o] = pil_clip8(v0);
            d0[n_out + o] = pil_clip8(v1);
            d0[2 * n_out + o] = pil_clip8(v2);
            d0[3 * n_out + o] = pil_clip8(v3);
        }
    }
    for (; l < lines; ++l) {
        const uint8_t* r = src + (size_t) (l * n_in);
        for (int64_t o = 0; o < n_out; ++o) {
            const int32_t* k = t.k.data() + (size_t) (o * t.stride);
            const int64_t lo = t.lo[(size_t) o], n = t.n[(size_t) o];
            int32_t v = half;
            for (int64_t x = 0; x < n; ++x) v += (int32_t) r[lo + x] * k[x];
            dst[(size_t) (l * n_out + o)] = pil_clip8(v);
        }
    }
}

// The columns' pass: a plane of `n_in` rows of `lines` bytes to `n_out` rows, an output row's totals for all its
// samples at once (each tap's row added across them, side by side in the vector registers).
void pil_columns(const uint8_t* src, int64_t lines, int64_t n_out, const PilTaps& t, uint8_t* dst) {
    std::vector<int32_t> acc((size_t) lines);
    for (int64_t o = 0; o < n_out; ++o) {
        std::fill(acc.begin(), acc.end(), 1 << (kPilBits - 1));
        const int32_t* k = t.k.data() + (size_t) (o * t.stride);
        const int64_t lo = t.lo[(size_t) o], n = t.n[(size_t) o];
        for (int64_t y = 0; y < n; ++y) {
            const int32_t c = k[y];
            const uint8_t* row = src + (size_t) ((lo + y) * lines);
            for (int64_t l = 0; l < lines; ++l) acc[(size_t) l] += (int32_t) row[l] * c;
        }
        uint8_t* d = dst + (size_t) (o * lines);
        for (int64_t l = 0; l < lines; ++l) d[l] = pil_clip8(acc[(size_t) l]);
    }
}

}  // namespace

std::vector<float> pillow_resize_planes(const std::vector<float>& planes, int64_t c, int64_t w, int64_t h, int64_t w2,
                                        int64_t h2, Filter filter) {
    if (c <= 0 || w <= 0 || h <= 0 || w2 <= 0 || h2 <= 0 || planes.size() != (size_t) (c * w * h))
        throw std::invalid_argument("image engine: a resize of an empty image, or to no size");
    if (filter == Filter::cv_cubic)
        throw std::invalid_argument("image engine: the host resize takes Pillow's filters (not OpenCV's cubic)");
    if (w == w2 && h == h2) return planes;
    // the bytes, each pass's input 16 bytes longer (the rows' vector copies read whole vectors).  Rounded to the
    // nearest, ties to even, by 2^23's add and subtract (nearbyint on 0 .. 255): nearbyint itself is a call per value
    // on the baseline x86-64, the most of this resize's time before
    const size_t pad = 16, n_out = (size_t) (c * h2 * w2);
    std::vector<uint8_t> in(planes.size() + pad), mid, out(n_out + pad);
    for (size_t i = 0; i < planes.size(); ++i)
        in[i] = (uint8_t) (int32_t) ((std::clamp(planes[i], 0.0f, 255.0f) + 0x1p23f) - 0x1p23f);
    auto rows = [&](const uint8_t* src, int64_t lines, uint8_t* dst) {
        pil_rows(src, lines, w, w2, pil_taps(w, w2, filter), dst);
    };
    auto columns = [&](const uint8_t* src, int64_t width, uint8_t* dst) {
        const PilTaps ty = pil_taps(h, h2, filter);
        for (int64_t ch = 0; ch < c; ++ch)
            pil_columns(src + (size_t) (ch * h * width), width, h2, ty, dst + (size_t) (ch * h2 * width));
    };
    // Image.resize's order: the columns first where the picture is over a hundred times taller than wide and loses
    // height (Pillow 12's Image.py), else the rows first; a pass is skipped where its axis keeps its size
    if (h > w * 100 && h2 < h) {
        mid.resize((size_t) (c * h2 * w) + pad);
        columns(in.data(), w, mid.data());
        if (w2 != w) rows(mid.data(), c * h2, out.data());
        else out = std::move(mid);
    } else if (w2 != w) {
        mid.resize((size_t) (c * h * w2) + pad);
        rows(in.data(), c * h, mid.data());
        if (h2 != h) columns(mid.data(), w2, out.data());
        else out = std::move(mid);
    } else {
        columns(in.data(), w, out.data());
    }
    return std::vector<float>(out.begin(), out.begin() + (std::ptrdiff_t) n_out);
}

CropBox parse_crop(const std::string& word) {
    CropBox c;
    std::vector<int64_t> v;
    std::stringstream ss(word.substr(word.rfind("CROP:", 0) == 0 ? 5 : 0));
    for (std::string tok; std::getline(ss, tok, ',');) v.push_back(std::stoll(tok));
    if ((v.size() != 7 && v.size() != 9) || v[2] <= 0 || v[3] <= 0 || v[4] < 0 || v[5] <= 0 || v[6] <= 0 ||
        (v.size() == 9 && (v[7] <= 0 || v[8] <= 0)))
        throw std::invalid_argument("CROP:<x>,<y>,<w>,<h>,<blend>,<width>,<height>[,<ref_w>,<ref_h>]: " + word);
    c.x = v[0];
    c.y = v[1];
    c.w = v[2];
    c.h = v[3];
    c.blend = v[4];
    c.width = v[5];
    c.height = v[6];
    if (v.size() == 9) {
        c.ref_w = v[7];
        c.ref_h = v[8];
    }
    return c;
}

uint64_t CropStitch::bytes(int64_t W, int64_t H, int64_t w, int64_t h, int64_t gw, int64_t gh, int64_t rw,
                           int64_t rh) {
    const uint64_t pic = (uint64_t) (W * H), box = (uint64_t) (w * h), gen = (uint64_t) (gw * gh);
    // prepare: the picture, its mask and a scratch; the rows' passes (the picture's 4 planes and the mask's) at the
    // drawing's width and the box's rows; the drawing's planes, its mask and a scratch, the outputs; the reference's
    // rows' pass, planes and output
    const uint64_t prep = pic * (4 + 4 + 4) + (uint64_t) (h * gw) * 5 * 4 + gen * (4 * 4 + 4 + 4 + 4 + 4) +
                          (uint64_t) (h * rw) * 4 * 4 + (uint64_t) (rw * rh) * (4 * 4 + 4);
    // stitch: the picture, the drawing, the rows' passes at the box's width and the drawing's rows, the box's planes
    // and its blend mask; the color match's copy of the cut and the drawing's Lab
    const uint64_t back = pic * 4 + gen * (4 + 4 + 4 + 12) + (uint64_t) (gh * w) * 5 * 4 + box * 5 * 4;
    return std::max(prep, back) + (1ull << 20);   // and the taps
}

CropStitch::~CropStitch() { gpu::free(crop_dev_); }

QiRef CropStitch::cut(const QiRef& base, const CropBox& box) {
    const int64_t W = base.width, H = base.height;
    if (W != box.width || H != box.height || base.rgba.size() != (size_t) (W * H * 4))
        throw std::invalid_argument("image engine: CROP: the picture is not the size the box was made for");
    QiRef out;
    out.width = box.w;
    out.height = box.h;
    out.rgba.resize((size_t) (box.w * box.h * 4));
    for (int64_t y = 0; y < box.h; ++y) {
        const int64_t sy = std::clamp<int64_t>(box.y + y, 0, H - 1);
        for (int64_t x = 0; x < box.w; ++x) {
            const int64_t sx = std::clamp<int64_t>(box.x + x, 0, W - 1);
            std::memcpy(&out.rgba[(size_t) ((y * box.w + x) * 4)], &base.rgba[(size_t) ((sy * W + sx) * 4)], 4);
        }
    }
    return out;
}

void CropStitch::prepare(const QiRef& base, const QiRef& mask, const CropBox& box, int64_t gw, int64_t gh, bool keep,
                         const QiRef* scaled) {
    const int64_t W = base.width, H = base.height;
    if (W != box.width || H != box.height || base.rgba.size() != (size_t) (W * H * 4))
        throw std::invalid_argument("image engine: CROP: the picture is not the size the box was made for");
    if (mask.width != W || mask.height != H || mask.rgba.size() != (size_t) (W * H * 4))
        throw std::invalid_argument("image engine: CROP: the mask is not the picture's size");
    if (gw <= 0 || gh <= 0) throw std::invalid_argument("image engine: CROP: no size to draw at");
    base_ = base.rgba;
    box_ = box;
    gw_ = gw;
    gh_ = gh;
    Buffers b;
    const uint8_t* pic = rgba8_device(b, base.rgba);
    float* m = b.put(edit_share(mask));
    float* tmp = b.get<float>((size_t) (W * H));
    // the mask grown by an eighth of the blend width, then softened by three box means of an eighth each (crop_reach's
    // sizes: the mask reaches half the blend width past the paint, no further than a square window and a Gaussian of
    // the same blend width would)
    if (box.blend > 0) {
        grow_disc(m, tmp, W, H, box.blend / 8);
        soften(m, tmp, W, H, box.blend / 8);
    }
    if (scaled != nullptr &&
        (scaled->width != gw || scaled->height != gh || scaled->rgba.size() != (size_t) (gw * gh * 4)))
        throw std::invalid_argument("image engine: CROP: the upscaled cut is not the drawing's size");
    // the box at the drawing's size: the picture (Lanczos, or the upscaler's `scaled`) and the mask (bilinear)
    const int64_t n = gw * gh;
    float* planes = b.get<float>((size_t) (4 * n));
    if (scaled != nullptr) {
        const uint8_t* up = rgba8_device(b, scaled->rgba);
        done(queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const int64_t i = (int64_t) id[0];
            for (int c = 0; c < 4; ++c) planes[c * n + i] = (float) up[i * 4 + c];
        }), "crop: the upscaled cut");
    } else {
        Resize rz(b, box.w, box.h, gw, gh, true);
        float* mid = b.get<float>((size_t) (4 * box.h * gw));
        rows_rgba8(pic, W, H, box.x, box.y, box.h, rz.tx, rz.lox, rz.wx, mid);
        columns(mid, 4, box.h, gw, rz.ty, rz.loy, rz.wy, planes, 0.0f, 255.0f);
    }
    float* mt = b.get<float>((size_t) n);
    {
        Resize rz(b, box.w, box.h, gw, gh, false);
        float* mid = b.get<float>((size_t) (box.h * gw));
        rows_mask(m, W, H, box.x, box.y, box.h, rz.tx, rz.lox, rz.wx, mid);
        columns(mid, 1, box.h, gw, rz.ty, rz.loy, rz.wy, mt, 0.0f, 1.0f);
    }
    share_.resize((size_t) n);
    gpu::copy(share_.data(), mt, (size_t) n * sizeof(float));
    // the drawing's base image and its mask (the model redraws where the blend mask is above 0)
    uint8_t* out = b.get<uint8_t>((size_t) (4 * n));
    uint8_t* reg = b.get<uint8_t>((size_t) (4 * n));
    done(queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        for (int c = 0; c < 4; ++c) out[i * 4 + c] = (uint8_t) sycl::round(planes[c * n + i]);
        const uint8_t a = mt[i] > 0.0f ? 0 : 255;
        reg[i * 4] = reg[i * 4 + 1] = reg[i * 4 + 2] = 0;
        reg[i * 4 + 3] = a;
    }), "crop: outputs");
    crop_.width = regen_.width = gw;
    crop_.height = regen_.height = gh;
    crop_.rgba.resize((size_t) (4 * n));
    regen_.rgba.resize((size_t) (4 * n));
    blend_.resize((size_t) n);
    gpu::copy(crop_.rgba.data(), out, (size_t) (4 * n));
    gpu::free(crop_dev_);
    crop_dev_ = nullptr;
    if (keep) {
        if (!gpu::alloc_device(&crop_dev_, (size_t) (4 * n)))
            throw std::runtime_error(std::string("image engine: GPU memory (the cut and its return): ") + gpu::last_error());
        gpu::copy(crop_dev_, out, (size_t) (4 * n));
    }
    gpu::copy(regen_.rgba.data(), reg, (size_t) (4 * n));
    gpu::copy(blend_.data(), mt, (size_t) n * sizeof(float));
    // the reference beside the base, when the box asks for one: the box at its size from the picture as well
    ref_ = QiRef{};
    if (box.ref_w > 0) {
        const int64_t rw = box.ref_w, rh = box.ref_h, rn = rw * rh;
        float* rp = b.get<float>((size_t) (4 * rn));
        Resize rz(b, box.w, box.h, rw, rh, true);
        float* mid = b.get<float>((size_t) (4 * box.h * rw));
        rows_rgba8(pic, W, H, box.x, box.y, box.h, rz.tx, rz.lox, rz.wx, mid);
        columns(mid, 4, box.h, rw, rz.ty, rz.loy, rz.wy, rp, 0.0f, 255.0f);
        uint8_t* ro = b.get<uint8_t>((size_t) (4 * rn));
        done(queue().parallel_for(sycl::range<1>((size_t) rn), [=](sycl::id<1> id) {
            const int64_t i = (int64_t) id[0];
            for (int c = 0; c < 4; ++c) ro[i * 4 + c] = (uint8_t) sycl::round(rp[c * rn + i]);
        }), "crop: reference");
        ref_.width = rw;
        ref_.height = rh;
        ref_.rgba.resize((size_t) (4 * rn));
        gpu::copy(ref_.rgba.data(), ro, (size_t) (4 * rn));
    }
}

namespace {

// the drawing's Lab channels moved to the picture's mean and spread (Reinhard): the statistics of the pixels the blend
// mask keeps (below kKept, both opaque; `whole`: every opaque one); none where too few pixels or a flat channel give
// no statistics
void color_match(uint8_t* drawn, const uint8_t* orig, const float* blend, int64_t n, bool whole) {
    // the pixels the stitch keeps as the picture's, nearly: the blend mask's Gaussian reaches most of a cut 1.2 times
    // the paint, and at exactly 0 a face's cut kept none (1120 x 928, 2026-11-05)
    constexpr float kKept = 0.05f;
    const float* m = whole ? nullptr : blend;
    const color::Stats d = color::stats(drawn, n, color::Space::lab, m, kKept, orig);
    const color::Stats o = color::stats(orig, n, color::Space::lab, m, kKept, drawn);
    color::transfer(drawn, n, color::Space::lab, d, o, 1024.0);
}

}  // namespace

std::vector<uint8_t> CropStitch::stitch(const std::vector<uint8_t>& made, int color) const {
    const int64_t W = box_.width, H = box_.height, bw = box_.w, bh = box_.h, n = gw_ * gh_;
    if (made.size() != (size_t) (4 * n))
        throw std::invalid_argument("image engine: CROP: the drawing is not the size it was asked for");
    Buffers b;
    uint8_t* pic = rgba8_device(b, base_);
    uint8_t* drawn = rgba8_device(b, made);
    const float* bm = b.put(blend_);
    if (color > 0) color_match(drawn, crop_dev_ != nullptr ? crop_dev_ : rgba8_device(b, crop_.rgba), bm, n, color == 2);
    // the drawing at the box's size (Lanczos) and the blend mask with it (bilinear)
    float* planes = b.get<float>((size_t) (4 * bw * bh));
    {
        Resize rz(b, gw_, gh_, bw, bh, true);
        float* mid = b.get<float>((size_t) (4 * gh_ * bw));
        rows_rgba8(drawn, gw_, gh_, 0, 0, gh_, rz.tx, rz.lox, rz.wx, mid);
        columns(mid, 4, gh_, bw, rz.ty, rz.loy, rz.wy, planes, 0.0f, 255.0f);
    }
    float* mb = b.get<float>((size_t) (bw * bh));
    {
        Resize rz(b, gw_, gh_, bw, bh, false);
        float* mid = b.get<float>((size_t) (gh_ * bw));
        rows_mask(bm, gw_, gh_, 0, 0, gh_, rz.tx, rz.lox, rz.wx, mid);
        columns(mid, 1, gh_, bw, rz.ty, rz.loy, rz.wy, mb, 0.0f, 1.0f);
    }
    // into the picture: blended by the mask where it is above 0, the picture's own bytes elsewhere
    const int64_t x0 = box_.x, y0 = box_.y;
    done(queue().parallel_for(sycl::range<2>((size_t) bh, (size_t) bw), [=](sycl::id<2> id) {
        const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
        const int64_t px = x0 + x, py = y0 + y;
        if (px < 0 || px >= W || py < 0 || py >= H) return;
        const float a = mb[y * bw + x];
        if (a <= 0.0f) return;
        uint8_t* p = pic + (py * W + px) * 4;
        for (int c = 0; c < 4; ++c) {
            const float v = a * planes[(c * bh + y) * bw + x] + (1.0f - a) * (float) p[c];
            p[c] = (uint8_t) sycl::clamp(sycl::round(v), 0.0f, 255.0f);
        }
    }), "crop: stitch");
    std::vector<uint8_t> out(base_.size());
    gpu::copy(out.data(), pic, out.size());
    return out;
}

}  // namespace strata::image
