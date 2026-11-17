// tools/vision/own/image_prep.cpp - see image_prep.h
//
// The resize below is a port of llama.cpp's tools/mtmd/mtmd-image.cpp (resize_pillow, calc_size_preserved_ratio, img_tool::resize),
// itself adapted from Pillow's Resample.c.  llama.cpp: MIT, Copyright (c) 2023-2026 The ggml authors.  Pillow: HPND.
#include "image_prep.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#define STBI_NO_STDIO
#define STB_IMAGE_STATIC            // private to this file: the mtmd build of strata-vision has its own copy of stb_image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace strata_vit {

bool image_decode(const uint8_t* data, size_t n, ImageU8& out, std::string& err) {
    int nx = 0, ny = 0, nc = 0;
    if (n == 0 || n > 0x7FFFFFFFu) { err = "the picture is empty or too large"; return false; }
    unsigned char* px = stbi_load_from_memory(data, (int) n, &nx, &ny, &nc, 3);
    if (!px) { err = std::string("cannot decode the picture: ") + stbi_failure_reason(); return false; }
    out.w = nx;
    out.h = ny;
    out.rgb.assign(px, px + (size_t) nx * ny * 3);
    stbi_image_free(px);
    return true;
}

bool image_load_file(const std::string& path, ImageU8& out, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<uint8_t> buf;
    uint8_t tmp[1 << 16];
    size_t k;
    while ((k = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + k);
    std::fclose(f);
    return image_decode(buf.data(), buf.size(), out, err);
}

Size2 smart_resize(int width, int height, int align, int min_pixels, int max_pixels) {
    if (width <= 0 || height <= 0 || align <= 0) return {0, 0};
    // the same float arithmetic as mtmd (calc_size_preserved_ratio with longest_edge off)
    auto round_by_factor = [align](float x) { return static_cast<int>(std::round(x / static_cast<float>(align))) * align; };
    auto ceil_by_factor = [align](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(align))) * align; };
    auto floor_by_factor = [align](float x) { return static_cast<int>(std::floor(x / static_cast<float>(align))) * align; };
    int w_bar = std::max(align, round_by_factor((float) width));
    int h_bar = std::max(align, round_by_factor((float) height));
    if (max_pixels > 0 && h_bar * w_bar > max_pixels) {
        const auto beta = std::sqrt(static_cast<float>(height) * width / max_pixels);
        h_bar = std::max(align, floor_by_factor(height / beta));
        w_bar = std::max(align, floor_by_factor(width / beta));
    } else if (min_pixels > 0 && h_bar * w_bar < min_pixels) {
        const auto beta = std::sqrt(static_cast<float>(min_pixels) / (static_cast<float>(height) * width));
        h_bar = ceil_by_factor(height * beta);
        w_bar = ceil_by_factor(width * beta);
    }
    return {w_bar, h_bar};
}

namespace {

const int PRECISION_BITS = 32 - 8 - 2;   // 22 fractional bits: 32 - 8 (uint8 pixels) - 2 (headroom for the accumulation)

double bicubic(double x) {               // Pillow's a = -0.5
    if (x < 0.0) x = -x;
    constexpr double a = -0.5;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

uint8_t clip8(int v) { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t) v; }

// the filter coefficients of one dimension: bounds[2 i] first input pixel, bounds[2 i + 1] their count; weights in fixed point; returns the kernel size
int precompute_weights(int in_size, int out_size, std::vector<int>& bounds, std::vector<int32_t>& weights) {
    double filterscale, scale;
    filterscale = scale = (double) in_size / out_size;
    if (filterscale < 1.0) filterscale = 1.0;
    const double support = 2.0 * filterscale;
    const int ksize = (int) std::ceil(support) * 2 + 1;
    std::vector<double> pre((size_t) out_size * ksize);
    bounds.resize((size_t) out_size * 2);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        double ww = 0.0;
        const double ss = 1.0 / filterscale;
        int xmin = (int) (center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int) (center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        int x = 0;
        for (; x < xmax; ++x) {
            const double w = bicubic((x + xmin - center + 0.5) * ss);
            pre[(size_t) xx * ksize + x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; ++x)
            if (ww != 0.0) pre[(size_t) xx * ksize + x] /= ww;
        for (; x < ksize; ++x) pre[(size_t) xx * ksize + x] = 0;
        bounds[(size_t) xx * 2 + 0] = xmin;
        bounds[(size_t) xx * 2 + 1] = xmax;
    }
    weights.resize((size_t) out_size * ksize);
    const double fxp_scale = std::ldexp(1.0, PRECISION_BITS);
    for (size_t i = 0; i < weights.size(); ++i) {
        const double rounded = pre[i] * fxp_scale + (pre[i] < 0 ? -0.5 : 0.5);   // Pillow: add 0.5 then truncate toward zero
        weights[i] = (int32_t) rounded;
    }
    return ksize;
}

std::vector<uint8_t> resample_horizontal(const uint8_t* src, int in_nx, int in_ny, int out_nx, int ksize, const std::vector<int>& bounds,
                                         const std::vector<int32_t>& weights) {
    std::vector<uint8_t> out((size_t) out_nx * in_ny * 3);
    for (int yy = 0; yy < in_ny; ++yy) {
        const uint8_t* src_row = src + (size_t) yy * in_nx * 3;
        uint8_t* dst_row = out.data() + (size_t) yy * out_nx * 3;
        for (int xx = 0; xx < out_nx; ++xx) {
            const int xmin = bounds[(size_t) xx * 2 + 0], xcnt = bounds[(size_t) xx * 2 + 1];
            const int32_t* k = &weights[(size_t) xx * ksize];
            const uint8_t* p = src_row + (size_t) xmin * 3;
            int32_t ss0 = 1 << (PRECISION_BITS - 1), ss1 = ss0, ss2 = ss0;
            for (int x = 0; x < xcnt; ++x) {
                ss0 += p[0] * k[x];
                ss1 += p[1] * k[x];
                ss2 += p[2] * k[x];
                p += 3;
            }
            dst_row[xx * 3 + 0] = clip8(ss0 >> PRECISION_BITS);
            dst_row[xx * 3 + 1] = clip8(ss1 >> PRECISION_BITS);
            dst_row[xx * 3 + 2] = clip8(ss2 >> PRECISION_BITS);
        }
    }
    return out;
}

std::vector<uint8_t> resample_vertical(const uint8_t* src, int in_nx, int out_ny, int ksize, const std::vector<int>& bounds,
                                       const std::vector<int32_t>& weight) {
    const size_t row_elems = (size_t) in_nx * 3;
    std::vector<uint8_t> out(row_elems * out_ny);
    std::vector<int32_t> acc(row_elems);
    for (int yy = 0; yy < out_ny; ++yy) {
        const int ymin = bounds[(size_t) yy * 2 + 0], ycnt = bounds[(size_t) yy * 2 + 1];
        const int32_t* k = &weight[(size_t) yy * ksize];
        std::fill(acc.begin(), acc.end(), 1 << (PRECISION_BITS - 1));
        for (int y = 0; y < ycnt; ++y) {
            const uint8_t* src_row = src + (size_t) (ymin + y) * row_elems;
            const int32_t w = k[y];
            for (size_t i = 0; i < row_elems; ++i) acc[i] += src_row[i] * w;
        }
        uint8_t* dst_row = out.data() + (size_t) yy * row_elems;
        for (size_t i = 0; i < row_elems; ++i) dst_row[i] = clip8(acc[i] >> PRECISION_BITS);
    }
    return out;
}

}  // namespace

void resize_bicubic(const ImageU8& img, ImageU8& dst, int target_w, int target_h) {
    const bool need_h = target_w != img.w, need_v = target_h != img.h;
    std::vector<int> bh, bv;
    std::vector<int32_t> wh, wv;
    int kh = 0, kv = 0;
    if (need_h) kh = precompute_weights(img.w, target_w, bh, wh);
    if (need_v) kv = precompute_weights(img.h, target_h, bv, wv);
    std::vector<uint8_t> res;
    if (need_h && need_v) {
        auto tmp = resample_horizontal(img.rgb.data(), img.w, img.h, target_w, kh, bh, wh);
        res = resample_vertical(tmp.data(), target_w, target_h, kv, bv, wv);
    } else if (need_h) {
        res = resample_horizontal(img.rgb.data(), img.w, img.h, target_w, kh, bh, wh);
    } else if (need_v) {
        res = resample_vertical(img.rgb.data(), img.w, target_h, kv, bv, wv);
    } else {
        res = img.rgb;
    }
    dst.w = target_w;
    dst.h = target_h;
    dst.rgb = std::move(res);
}

void resize_pad_ceil(const ImageU8& src, ImageU8& dst, Size2 target) {
    if (src.w == target.w && src.h == target.h) { dst = src; return; }
    const float scale_w = static_cast<float>(target.w) / src.w;
    const float scale_h = static_cast<float>(target.h) / src.h;
    const float scale = std::min(scale_w, scale_h);
    const int new_w = std::min(static_cast<int>(std::ceil(src.w * scale)), target.w);
    const int new_h = std::min(static_cast<int>(std::ceil(src.h * scale)), target.h);
    ImageU8 resized;
    resize_bicubic(src, resized, new_w, new_h);
    ImageU8 out;
    out.w = target.w;
    out.h = target.h;
    out.rgb.assign((size_t) target.w * target.h * 3, 0);   // black canvas
    const int ox = (target.w - new_w) / 2, oy = (target.h - new_h) / 2;
    for (int y = 0; y < new_h; ++y) {
        std::memcpy(&out.rgb[((size_t) (y + oy) * target.w + ox) * 3], &resized.rgb[(size_t) y * new_w * 3], (size_t) new_w * 3);
    }
    dst = std::move(out);
}

Size2 prepared_size(int w, int h, const VitGeometry& g, int min_tokens, int max_tokens) {
    const int patch_area = g.patch * g.patch * g.merge * g.merge;
    const int min_px = (min_tokens > 0 ? min_tokens : g.min_tokens) * patch_area;
    const int max_px = (max_tokens > 0 ? max_tokens : g.max_tokens) * patch_area;
    return smart_resize(w, h, g.align(), min_px, max_px);
}

bool prepare_image(const ImageU8& src, const VitGeometry& g, int min_tokens, int max_tokens, PreparedImage& out, std::string& err, ImageU8* resized) {
    if (src.w <= 0 || src.h <= 0 || src.rgb.size() != (size_t) src.w * src.h * 3) { err = "the picture is empty"; return false; }
    const Size2 t = prepared_size(src.w, src.h, g, min_tokens, max_tokens);
    if (t.w <= 0 || t.h <= 0) { err = "the picture has no usable size"; return false; }
    ImageU8 r;
    resize_pad_ceil(src, r, t);
    out.w = t.w;
    out.h = t.h;
    const size_t n = (size_t) t.w * t.h;
    out.chw.resize(3 * n);
    for (size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) out.chw[c * n + i] = ((float) r.rgb[i * 3 + c] / 255.0f - g.mean[c]) / g.stdv[c];
    if (resized) *resized = std::move(r);
    return true;
}

}  // namespace strata_vit
