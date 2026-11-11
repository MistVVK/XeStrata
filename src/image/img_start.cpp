// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/img_start.cpp - see include/strata/image/img_start.hpp.

#include "strata/image/img_start.hpp"

#include "strata/core/gpu.hpp"
#include "strata/image/qi_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::image {

namespace gpu = strata::gpu;

const QiRef* base_image(const QiRequest& r) {
    if (!r.init.rgba.empty()) return &r.init;
    if ((r.strength < 1.0 || !r.mask.rgba.empty()) && !r.refs.empty()) return &r.refs[0];
    return nullptr;
}

std::string refs_key(const std::vector<QiRef>& refs) {
    std::string key;
    for (const QiRef& f : refs) {
        uint64_t h = 1469598103934665603ull;
        for (uint8_t b : f.rgba) h = (h ^ b) * 1099511628211ull;
        key += std::to_string(f.width) + "x" + std::to_string(f.height) + ":" + std::to_string(h) + ";";
    }
    return key;
}

int64_t strength_skip(int64_t steps, double strength) {
    if (!(strength > 0.0 && strength <= 1.0))
        throw std::invalid_argument("image engine: the strength must be above 0 and at most 1");
    const double run = std::min((double) steps * strength, (double) steps);
    const auto skip = (int64_t) std::max((double) steps - run, 0.0);
    if (skip >= steps)
        throw std::invalid_argument("image engine: a strength of " + std::to_string(strength) + " runs none of the " +
                                    std::to_string(steps) + " steps");
    return skip;
}

std::vector<float> strength_sigmas(const std::vector<float>& sigmas, double strength) {
    if (strength >= 1.0 || sigmas.size() < 2) return sigmas;
    const int64_t skip = strength_skip((int64_t) sigmas.size() - 1, strength);
    return std::vector<float>(sigmas.begin() + skip, sigmas.end());
}

std::vector<uint8_t> resize_nearest(const QiRef& f, int64_t W, int64_t H) {
    if (f.width <= 0 || f.height <= 0 || f.rgba.size() != (size_t) (f.width * f.height * 4))
        throw std::invalid_argument("image engine: an empty image");
    std::vector<uint8_t> out((size_t) (W * H * 4));
    for (int64_t y = 0; y < H; ++y)
        for (int64_t x = 0; x < W; ++x)
            std::copy_n(f.rgba.data() + ((y * f.height / H) * f.width + x * f.width / W) * 4, 4,
                        out.data() + (y * W + x) * 4);
    return out;
}

namespace {

/// One 8-bit value's place in the planar floats an encoder reads, the alpha `a` (0 .. 1) over white where `alpha` says
/// it.  WhiteUnit's composite is the 0 .. 1 floats' and unrounded, the others' the whole byte a picture would save.
float plane_value(int v, float a, PlaneAlpha alpha, PlaneRange range) {
    if (alpha == PlaneAlpha::WhiteUnit) {
        const float u = (float) v / 255.f * a + 1.f - a;
        return range == PlaneRange::Signed ? 2.f * u - 1.f : u;
    }
    const float b = alpha == PlaneAlpha::White ? std::nearbyint((float) v * a + 255.f * (1.f - a)) : (float) v;
    switch (range) {
        case PlaneRange::Byte: return b;
        case PlaneRange::Unit: return b / 255.f;
        case PlaneRange::Signed: return b / 127.5f - 1.f;
    }
    return b;
}

}  // namespace

std::vector<float> image_planes(const std::vector<uint8_t>& px, int64_t W, int64_t H, int64_t channels,
                                PlaneAlpha alpha, PlaneRange range) {
    const int64_t n = W * H;
    std::vector<float> out((size_t) (channels * n));
    for (int64_t q = 0; q < n; ++q) {
        const uint8_t* p = px.data() + (size_t) (q * 4);
        const float a = (float) p[3] / 255.f;
        for (int64_t c = 0; c < channels; ++c)
            out[(size_t) (c * n + q)] =
                plane_value(c == 3 && alpha != PlaneAlpha::Keep ? 255 : p[c], a, alpha, range);
    }
    return out;
}

std::vector<float> image_planes(const QiRef& f, int64_t W, int64_t H, int64_t channels, PlaneAlpha alpha,
                                PlaneRange range) {
    return image_planes(resize_nearest(f, W, H), W, H, channels, alpha, range);
}

void noise_to_tokens(const float* src, int64_t C, int64_t H, int64_t W, int64_t patch, float* out, int64_t stride) {
    const int64_t plane = stride != 0 ? stride : H * W, tw = W / patch, TV = C * patch * patch, cc = patch * patch;
    for (int64_t c = 0; c < C; ++c)
        for (int64_t y = 0; y < H; ++y)
            for (int64_t x = 0; x < W; ++x)
                out[(size_t) (((y / patch) * tw + x / patch) * TV + c * cc + (y % patch) * patch + x % patch)] =
                    src[(size_t) (c * plane + y * W + x)];
}

std::vector<float> latent_mask(const std::vector<uint8_t>& px, int64_t W, int64_t H, int64_t patch) {
    const int64_t w = W / patch;
    std::vector<float> m((size_t) ((H / patch) * w), 0.f);
    for (int64_t y = 0; y < H; ++y)
        for (int64_t x = 0; x < W; ++x)
            if (px[(size_t) ((y * W + x) * 4 + 3)] < 255) m[(size_t) ((y / patch) * w + x / patch)] = 1.f;
    return m;
}

std::vector<float> request_latent_mask(const QiRequest& r, const std::vector<uint8_t>& px, int64_t W, int64_t H,
                                       int64_t patch) {
    if (r.mask_share.empty()) return latent_mask(px, W, H, patch);
    if (r.mask_share.size() != (size_t) (W * H))
        throw std::invalid_argument("image engine: the mask's share is not the image's size");
    const int64_t w = W / patch, h = H / patch;
    std::vector<float> m((size_t) (h * w), 0.f);
    for (int64_t y = 0; y < h * patch; ++y)
        for (int64_t x = 0; x < w * patch; ++x) m[(size_t) ((y / patch) * w + x / patch)] += r.mask_share[(size_t) (y * W + x)];
    for (float& v : m) v /= (float) (patch * patch);
    return m;
}

std::vector<uint8_t> rgba8(const std::vector<float>& planar, int64_t HW) {
    std::vector<uint8_t> img((size_t) (4 * HW));
    for (int64_t q = 0; q < HW; ++q)
        for (int c = 0; c < 4; ++c)
            img[(size_t) (q * 4 + c)] =
                (uint8_t) std::lround(std::clamp(planar[(size_t) (c * HW + q)], 0.0f, 1.0f) * 255.0f);
    return img;
}

void keep_pixels(std::vector<uint8_t>& img, const std::vector<uint8_t>& base, const std::vector<uint8_t>& mask) {
    for (size_t q = 0; q + 3 < img.size(); q += 4) {
        const int a = mask[q + 3];
        for (size_t c = 0; c < 4; ++c) img[q + c] = (uint8_t) ((a * base[q + c] + (255 - a) * img[q + c] + 127) / 255);
    }
}

LatentStart::~LatentStart() {
    for (float* p : {z0_, noise_, mask_}) gpu::free(p);
}

float* LatentStart::base(int64_t n) {
    if (z0_ == nullptr || n != n_) {
        gpu::free(z0_);
        z0_ = nullptr;
        if (!gpu::alloc_device(&z0_, (size_t) n * 4))
            throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
        n_ = n;
    }
    return z0_;
}

void LatentStart::set_mask(const std::vector<float>& mask, const std::vector<float>& noise, int64_t C, int64_t HW,
                           bool tokens) {
    if (mask.empty()) return;
    if ((int64_t) mask.size() != HW || (int64_t) noise.size() != C * HW || C * HW != n_)
        throw std::logic_error("image engine: the mask, the noise and the base latent differ in size");
    if (!gpu::alloc_device(&noise_, noise.size() * 4) || !gpu::alloc_device(&mask_, mask.size() * 4))
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    gpu::copy(noise_, noise.data(), noise.size() * 4);
    gpu::copy(mask_, mask.data(), mask.size() * 4);
    C_ = C;
    HW_ = HW;
    tokens_ = tokens;
}

void LatentStart::start(float* x, float s, void* stream) const {
    qi_noise_mix(x, z0_, x, s, n_, stream);   // each value reads its own noise before it is written
}

std::function<void(float*, float)> LatentStart::after(void* stream) const {
    if (mask_ == nullptr) return {};
    const float *z0 = z0_, *nz = noise_, *mk = mask_;
    const int64_t C = C_, HW = HW_;
    const bool tokens = tokens_;
    return [z0, nz, mk, C, HW, tokens, stream](float* xs, float sigma) {
        qi_inpaint_blend(xs, z0, nz, mk, sigma, C, HW, stream, tokens);
    };
}

}  // namespace strata::image
