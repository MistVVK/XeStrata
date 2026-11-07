// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/img_crop.cpp - see include/strata/image/img_crop.hpp.

#include "strata/image/img_crop.hpp"

#include "strata/core/gpu.hpp"
#include "strata/image/img_color.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

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
            throw std::runtime_error(std::string("image engine: GPU memory (crop and stitch): ") + gpu::last_error());
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
// `k` weights, the filter's support widened by the scale when it shrinks; Lanczos (a = 3), the triangle (bilinear) or
// the cubic (bicubic, a = -0.5)
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

Taps taps(int64_t in, int64_t out, Filter f) {
    const double scale = (double) in / (double) out, fs = std::max(scale, 1.0);
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

// scipy's gaussian_filter1d weights: sigma, cut at int(4 sigma + 0.5)
std::vector<float> gauss(double sigma) {
    const int64_t r = (int64_t) std::floor(4.0 * sigma + 0.5);
    std::vector<double> e((size_t) (2 * r + 1));
    double sum = 0.0;
    for (size_t k = 0; k < e.size(); ++k) {
        const double x = (double) k - (double) r;
        e[k] = std::exp(-0.5 * x * x / (sigma * sigma));
        sum += e[k];
    }
    std::vector<float> w(e.size());
    for (size_t k = 0; k < e.size(); ++k) w[k] = (float) (e[k] / sum);
    return w;
}

// scipy's "reflect" edge: d c b a | a b c d | d c b a
inline int64_t reflect(int64_t i, int64_t n) {
    for (;;) {
        if (i < 0) i = -i - 1;
        else if (i >= n) i = 2 * n - i - 1;
        else return i;
    }
}

// InpaintCropImproved's mask before its growth, on the host (a fill is a walk, not a GPU's work): the edit share
// (1 - alpha / 255), its values below 0.1 out (mask_hipass_filter), then its holes filled at each of
// fillholes_iterative_hipass_fill_m's steps - the share at least t set, closed by the 3 x 3 square with the picture's
// outside counted as set (scipy's binary_closing, border_value 1), its holes filled (binary_fill_holes: the unset
// pixels, 4-connected, that the edges do not reach), and the filled pixels at least t.  The closing and the fill
// change nothing more than 2 pixels past the mask's box (InpaintCropImproved's, checked against scipy), so they run in
// that box: the server's box (serve/images.py, crop_reach) counts the 2 pixels.
std::vector<float> filled_share(const QiRef& mask) {
    static const float kSteps[] = {1.0f, 0.99f, 0.97f, 0.95f, 0.93f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    const int64_t W = mask.width, H = mask.height;
    std::vector<float> m((size_t) (W * H));
    int64_t x0 = W, y0 = H, x1 = -1, y1 = -1;
    for (int64_t y = 0; y < H; ++y)
        for (int64_t x = 0; x < W; ++x) {
            float v = 1.0f - (float) mask.rgba[(size_t) ((y * W + x) * 4 + 3)] / 255.0f;
            if (v < 0.1f) v = 0.0f;
            m[(size_t) (y * W + x)] = v;
            if (v > 0.0f) {
                x0 = std::min(x0, x), x1 = std::max(x1, x);
                y0 = std::min(y0, y), y1 = std::max(y1, y);
            }
        }
    if (x1 < 0) return m;
    // the box R (the mask's, 2 pixels wider) and R1 (1 more), within the picture
    const int64_t rx0 = std::max<int64_t>(0, x0 - 2), ry0 = std::max<int64_t>(0, y0 - 2);
    const int64_t rx1 = std::min(W, x1 + 3), ry1 = std::min(H, y1 + 3), rw = rx1 - rx0, rh = ry1 - ry0;
    const int64_t ax0 = std::max<int64_t>(0, rx0 - 1), ay0 = std::max<int64_t>(0, ry0 - 1);
    const int64_t ax1 = std::min(W, rx1 + 1), ay1 = std::min(H, ry1 + 1), aw = ax1 - ax0;
    std::vector<uint8_t> b((size_t) (rw * rh)), last, dil((size_t) (aw * (ay1 - ay0))), grid((size_t) ((rw + 2) * (rh + 2)));
    std::vector<int64_t> todo;
    const auto set = [&](int64_t x, int64_t y) {   // b at a pixel of the picture (unset outside R)
        return x >= rx0 && x < rx1 && y >= ry0 && y < ry1 && b[(size_t) ((y - ry0) * rw + x - rx0)] != 0;
    };
    for (const float t : kSteps) {
        for (int64_t y = 0; y < rh; ++y)
            for (int64_t x = 0; x < rw; ++x) b[(size_t) (y * rw + x)] = m[(size_t) ((y + ry0) * W + x + rx0)] >= t;
        if (b == last) continue;   // the same set: the same fill, its pixels already at a higher step
        last = b;
        for (int64_t y = ay0; y < ay1; ++y)   // dilated, the outside set
            for (int64_t x = ax0; x < ax1; ++x) {
                bool v = false;
                for (int64_t dy = -1; dy <= 1 && !v; ++dy)
                    for (int64_t dx = -1; dx <= 1 && !v; ++dx) {
                        const int64_t nx = x + dx, ny = y + dy;
                        v = nx < 0 || ny < 0 || nx >= W || ny >= H || set(nx, ny);
                    }
                dil[(size_t) ((y - ay0) * aw + x - ax0)] = v;
            }
        // eroded (closed), on a frame of unset pixels the fill starts from
        std::fill(grid.begin(), grid.end(), 0);
        for (int64_t y = 0; y < rh; ++y)
            for (int64_t x = 0; x < rw; ++x) {
                bool v = true;
                for (int64_t dy = -1; dy <= 1 && v; ++dy)
                    for (int64_t dx = -1; dx <= 1 && v; ++dx) {
                        const int64_t nx = x + rx0 + dx, ny = y + ry0 + dy;
                        if (nx >= 0 && ny >= 0 && nx < W && ny < H) v = dil[(size_t) ((ny - ay0) * aw + nx - ax0)] != 0;
                    }
                grid[(size_t) ((y + 1) * (rw + 2) + x + 1)] = v;
            }
        // the unset pixels the frame reaches (2); the rest of the unset ones are holes
        const int64_t gw = rw + 2, gh = rh + 2;
        todo.assign(1, 0);
        grid[0] = 2;
        while (!todo.empty()) {
            const int64_t i = todo.back();
            todo.pop_back();
            const int64_t x = i % gw, y = i / gw;
            const int64_t next[4] = {x > 0 ? i - 1 : -1, x + 1 < gw ? i + 1 : -1, y > 0 ? i - gw : -1,
                                     y + 1 < gh ? i + gw : -1};
            for (const int64_t j : next)
                if (j >= 0 && grid[(size_t) j] == 0) {
                    grid[(size_t) j] = 2;
                    todo.push_back(j);
                }
        }
        for (int64_t y = 0; y < rh; ++y)
            for (int64_t x = 0; x < rw; ++x)
                if (grid[(size_t) ((y + 1) * gw + x + 1)] != 2) {
                    float& v = m[(size_t) ((y + ry0) * W + x + rx0)];
                    v = std::max(v, t);
                }
    }
    return m;
}

// scipy's grey_dilation by a square window of k (reflected edges: the window clipped to the picture), rows then
// columns; the window's offsets are [-(k - 1 - k / 2), k / 2], scipy's footprint mirrored
void dilate(float* m, float* tmp, int64_t W, int64_t H, int64_t k) {
    const int64_t left = k - 1 - k / 2, right = k / 2;
    done(queue().parallel_for(sycl::range<2>((size_t) H, (size_t) W), [=](sycl::id<2> id) {
        const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
        const int64_t a = sycl::max(x - left, (int64_t) 0), b = sycl::min(x + right, W - 1);
        float v = 0.0f;
        for (int64_t j = a; j <= b; ++j) v = sycl::fmax(v, m[y * W + j]);
        tmp[y * W + x] = v;
    }), "crop: dilate rows");
    done(queue().parallel_for(sycl::range<2>((size_t) H, (size_t) W), [=](sycl::id<2> id) {
        const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
        const int64_t a = sycl::max(y - left, (int64_t) 0), b = sycl::min(y + right, H - 1);
        float v = 0.0f;
        for (int64_t j = a; j <= b; ++j) v = sycl::fmax(v, tmp[j * W + x]);
        m[y * W + x] = sycl::fmin(v, 1.0f);
    }), "crop: dilate columns");
}

// scipy's gaussian_filter (reflected edges), along the rows' axis first, then clipped to [0, 1]
void blur(float* m, float* tmp, int64_t W, int64_t H, const float* w, int r) {
    done(queue().parallel_for(sycl::range<2>((size_t) H, (size_t) W), [=](sycl::id<2> id) {
        const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
        float v = 0.0f;
        for (int i = -r; i <= r; ++i) v += w[i + r] * m[reflect(y + i, H) * W + x];
        tmp[y * W + x] = v;
    }), "crop: blur columns");
    done(queue().parallel_for(sycl::range<2>((size_t) H, (size_t) W), [=](sycl::id<2> id) {
        const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
        float v = 0.0f;
        for (int i = -r; i <= r; ++i) v += w[i + r] * tmp[y * W + reflect(x + i, W)];
        m[y * W + x] = sycl::clamp(v, 0.0f, 1.0f);
    }), "crop: blur rows");
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

// src (RGBA8, device) at w x h by Pillow's filter into dst (device)
void resize_device(const uint8_t* src, int64_t sw, int64_t sh, uint8_t* dst, int64_t w, int64_t h, Filter filter) {
    if (sw <= 0 || sh <= 0 || w <= 0 || h <= 0)
        throw std::invalid_argument("image engine: a resize of an empty image, or to no size");
    Buffers b;
    const int64_t n = w * h;
    float* planes = b.get<float>((size_t) (4 * n));
    {
        Resize rz(b, sw, sh, w, h, filter);
        float* mid = b.get<float>((size_t) (4 * sh * w));
        rows_rgba8(src, sw, sh, 0, 0, sh, rz.tx, rz.lox, rz.wx, mid);
        columns(mid, 4, sh, w, rz.ty, rz.loy, rz.wy, planes, 0.0f, 255.0f);
    }
    done(queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        for (int c = 0; c < 4; ++c) dst[i * 4 + c] = (uint8_t) sycl::round(planes[c * n + i]);
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

QiRef lanczos_resize(const QiRef& src, int64_t w, int64_t h) { return pillow_resize(src, w, h, Filter::lanczos); }

QiRef pillow_resize(const QiRef& src, int64_t w, int64_t h, Filter filter) {
    if (src.width <= 0 || src.height <= 0 || src.rgba.size() != (size_t) (src.width * src.height * 4) || w <= 0 ||
        h <= 0)
        throw std::invalid_argument("image engine: a resize of an empty image, or to no size");
    Buffers b;
    const uint8_t* pic = rgba8_device(b, src.rgba);
    uint8_t* out = b.get<uint8_t>((size_t) (4 * w * h));
    resize_device(pic, src.width, src.height, out, w, h, filter);
    QiRef r;
    r.width = w;
    r.height = h;
    r.rgba.resize((size_t) (4 * w * h));
    gpu::copy(r.rgba.data(), out, r.rgba.size());
    return r;
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

void CropStitch::prepare(const QiRef& base, const QiRef& mask, const CropBox& box, int64_t gw, int64_t gh, bool keep) {
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
    float* m = b.put(filled_share(mask));
    float* tmp = b.get<float>((size_t) (W * H));
    // the mask grown and blurred over the picture (InpaintCropImproved's mask_blend_pixels)
    const double sigma = (double) box.blend * 0.5 / 4.0;
    std::vector<float> gw8;
    float* gwd = nullptr;
    int r = 0;
    if (box.blend > 0) {
        dilate(m, tmp, W, H, (int64_t) std::ceil((double) box.blend / 4.0 * 1.5 + 1.0));
        gw8 = gauss(sigma);
        r = (int) (gw8.size() / 2);
        gwd = b.put(gw8);
        blur(m, tmp, W, H, gwd, r);
    }
    // the box at the drawing's size: the picture (Lanczos) and the mask (bilinear), the mask blurred again there
    const int64_t n = gw * gh;
    float* planes = b.get<float>((size_t) (4 * n));
    {
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
    if (box.blend > 0) blur(mt, b.get<float>((size_t) n), gw, gh, gwd, r);
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
            throw std::runtime_error(std::string("image engine: GPU memory (crop and stitch): ") + gpu::last_error());
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
