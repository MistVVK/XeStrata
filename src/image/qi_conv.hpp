// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_conv.hpp - the image engine's convolutions (the VAEs', RealESRGAN's) through a GPU maker's library where
// the build has one for the GPU (oneDNN on Intel GPUs, cuDNN on NVIDIA GPUs in contrib-llvm), false where it does not
// take the convolution (the engine's im2col and products then).
#pragma once

#include "strata/core/device.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace strata::image::conv {

/// Where a convolution reads and writes beyond the plain form (the VAEs'): RealESRGAN's dense blocks read a prefix of a
/// wider tensor's channels and write their output into its next channels, LeakyReLU'd, in FP16.
struct Form {
    int64_t ldx = 0;       // x's elements between two pixels (0: C)
    int64_t ldy = 0;       // y's (0: O)
    bool y16 = false;      // y FP16 (uint16_t bits) rather than FP32
    float leaky = -1.0f;   // 0 or more: LeakyReLU of that slope after the bias (not with `add`)
    bool gelu = false;     // GELU (erf's) after the bias (not with `add` or `leaky`; oneDNN alone)
};

/// y [OH][OW][O] (FP32) = the k x k convolution of stride `stride` of x [H][W][C] (FP16) by w [O][k][k][C] (FP16)
/// plus `bias` [O] (null: none), zero padding pt, pl before the rows and columns and pb, pr after them; `add`: added to
/// y (y += ...) instead of written.
bool onednn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
            int stride, int pt, int pl, int pb, int pr, const float* bias, bool add, float* y);
/// The same in `f`'s form: y is FP32 or FP16 by f.y16.
bool onednn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
            int stride, int pt, int pl, int pb, int pr, const float* bias, bool add, void* y, const Form& f);

/// The same through cuDNN (contrib-llvm, NVIDIA GPUs): stride 1, padding k / 2.
bool cudnn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
           const float* bias, bool add, float* y);
/// The same in `f`'s form.
bool cudnn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
           const float* bias, bool add, void* y, const Form& f);
/// y [D - kt + 1][H][W][O] (FP32, FP16 with y16) = the kt x k x k convolution of x [D][H][W][C] (FP16) by w
/// [O][kt][k][k][C] (FP16), stride 1, padding k / 2 in space and none in time, plus `bias`; `add`: added to y.  One
/// pass for the taps, where a 2D convolution a tap writes y as many times.
bool cudnn3(sycl::queue& q, const uint16_t* x, int64_t D, int64_t H, int64_t W, int64_t C, const uint16_t* w,
            int64_t O, int kt, int k, const float* bias, bool add, void* y, bool y16 = false);

/// The common choice of a maker's convolution.  The caller supplies the actual output form; dilation and temporal
/// taps the maker interface cannot express go through the engine's unfolded products instead.
struct Request {
    const uint16_t* x;
    int64_t h, w, c;
    const uint16_t* weights;
    int64_t outputs;
    int k, stride, pt, pl, pb, pr, dilation = 1, temporal = 1;
    const float* bias = nullptr;
    bool add = false;
    void* out = nullptr;
    Form form{};
};

enum class Library { none, onednn, cudnn };

static inline Library library(sycl::queue& q, const Request& r, void* cudnn_out = nullptr, bool cudnn_separate = false) {
    if (r.dilation != 1 || r.temporal != 1) return Library::none;
    // A plain 1x1 without a residual gained nothing over a product on the B70 (2026-11-02).  An activation can make
    // a library worthwhile because it avoids a pass; this is also the rule for every other image model.
    if (r.k == 1 && r.stride == 1 && r.pt == 0 && r.pl == 0 && r.pb == 0 && r.pr == 0 && !r.add &&
        r.form.leaky < 0 && !r.form.gelu) return Library::none;
#if STRATA_ONEDNN
    if (onednn(q, r.x, r.h, r.w, r.c, r.weights, r.outputs, r.k, r.stride, r.pt, r.pl, r.pb, r.pr, r.bias, r.add,
               r.out, r.form)) return Library::onednn;
#endif
#if STRATA_CUDNN
    // On the RTX 4070 small FP32 3x3 maps at 280 and 1120 pixels ran faster unfolded than through cuDNN:
    // ten warm runs were 5.40 vs 5.62 ms (face) and 5.93 vs 6.25 ms (person), 2026-11-11.  The same exclusion
    // on oneDNN slowed the B70 (face 3.5 to 4.2 ms, BiRefNet decoder 10.3 to 12.6 ms), so it applies to cuDNN only.
    if (r.k == 3 && r.stride == 1 && r.pt == 1 && r.pl == 1 && r.pb == 1 && r.pr == 1 && !r.add &&
        !r.form.y16 && r.form.leaky < 0 && !r.form.gelu && r.h * r.w <= 1120) return Library::none;
    if (q.get_device().get_backend() == sycl::backend::ext_oneapi_cuda && r.stride == 1 && r.k % 2 == 1 &&
        r.pt == r.k / 2 && r.pl == r.pt && r.pb == r.pt && r.pr == r.pt &&
        (!cudnn_separate || cudnn_out != nullptr) &&
        cudnn(q, r.x, r.h, r.w, r.c, r.weights, r.outputs, r.k, cudnn_separate ? nullptr : r.bias,
              cudnn_separate ? false : r.add, cudnn_separate ? cudnn_out : r.out, r.form)) return Library::cudnn;
#endif
    (void) q; (void) r; (void) cudnn_out; (void) cudnn_separate;
    return Library::none;
}

/// A small fraction of device memory, bounded by half its largest allocation (including the engine's limits), so the
/// scratch never exceeds that allocation.  On RX 7900 XTX the former /8 allocation bound split a 3456-wide product
/// at 19,418 rows instead of the earlier 32,768-row band; this bound gives 58,216 and brought MOSS-Image decode from
/// 68 to 51 ms and Anima from 58 to 43 ms (512 square, 2026-11-11).  The actual band follows the device and room.
inline int64_t col_values(sycl::queue& q) {
    const auto dev = q.get_device();
    uint64_t total = dev.get_info<sycl::info::device::global_mem_size>();
    uint64_t largest = dev.get_info<sycl::info::device::max_mem_alloc_size>();
    if (const uint64_t limit = core::vram_limit_bytes()) total = std::min(total, limit);
    if (const uint64_t limit = core::max_alloc_limit_bytes()) largest = std::min(largest, limit);
    return (int64_t) (std::min(total / 64, largest / 2) / sizeof(uint16_t));
}

/// The rows of a convolution's band, at most `most`: the pixels split into equal bands of a multiple of 64 rows, so
/// that the last band is not a short remainder.  On the RX 9060 XT bands of `most` rows and a remainder took the warm
/// 512-square decodes of MOSS-Image 144 ms, Anima 120 ms and Qwen-Image 257 ms, and SeedVR2's 1024-square decode
/// 798 ms; these bands 96, 82, 197 and 561 ms, at half and twice this `most` as well (2026-11-11).  Equal bands not of
/// a multiple of 64 left SeedVR2 at 782 ms; multiples of 256 and 1024 rows took as long as 64.  A band is at least
/// (n - 1) / n of `most`, so the wide bands the RX 7900 XTX gained from (col_values) stay about as wide.
inline int64_t band_rows(int64_t pixels, int64_t most) {
    constexpr int64_t kAlign = 64;
    if (most >= pixels || most < kAlign) return most;
    for (int64_t n = (pixels + most - 1) / most;; ++n) {
        const int64_t rows = ((pixels + n - 1) / n + kAlign - 1) / kAlign * kAlign;
        if (rows <= most) return rows;
    }
}

/// Shared product driver: point products read the input directly; the other convolutions unfold bands that fit the
/// caller's allocation.  Unfolding and storing retain the different tensor forms until their own common parts exist.
template<class Unfold, class Product, class Finish>
void products(int64_t pixels, int64_t k, int64_t col_values, bool point, Unfold&& unfold, Product&& product,
              Finish&& finish) {
    if (point) product(0, pixels, true);
    else {
        if (col_values < k) throw std::runtime_error("image engine: convolution band cannot hold one row");
        const int64_t rows = band_rows(pixels, col_values / k);
        for (int64_t p = 0; p < pixels; p += rows) {
            const int64_t n = std::min(rows, pixels - p);
            unfold(p, n);
            product(p, n, false);
        }
    }
    finish();
}

template<class Unfold, class Product, class Finish>
static Library drive(sycl::queue& q, const Request& r, int64_t pixels, int64_t k, int64_t col_cap, bool point,
              Unfold&& unfold, Product&& product, Finish&& finish) {
    const Library chosen = library(q, r);
    if (chosen == Library::none)
        products(pixels, k, std::min(col_cap, conv::col_values(q)), point, unfold, product, finish);
    return chosen;
}

}  // namespace strata::image::conv
