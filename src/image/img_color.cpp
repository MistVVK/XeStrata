// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/img_color.cpp - see include/strata/image/img_color.hpp.

#include "strata/image/img_color.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::image::color {

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
            throw std::runtime_error(std::string("image engine: GPU memory (color): ") + gpu::last_error());
        held.push_back(p);
        return p;
    }
};

// sRGB (0 .. 255) <-> CIE Lab, D65, as kornia's rgb_to_lab and lab_to_rgb (ComfyUI's ColorTransfer)
inline void to_lab(float r, float g, float b, float* lab) {
    auto lin = [](float v) { v /= 255.0f; return v > 0.04045f ? sycl::pow((v + 0.055f) / 1.055f, 2.4f) : v / 12.92f; };
    const float R = lin(r), G = lin(g), B = lin(b);
    const float x = (0.412453f * R + 0.357580f * G + 0.180423f * B) / 0.95047f;
    const float y = 0.212671f * R + 0.715160f * G + 0.072169f * B;
    const float z = (0.019334f * R + 0.119193f * G + 0.950227f * B) / 1.08883f;
    auto f = [](float t) { return t > 0.008856f ? sycl::cbrt(t) : 7.787f * t + 4.0f / 29.0f; };
    const float fx = f(x), fy = f(y), fz = f(z);
    lab[0] = 116.0f * fy - 16.0f;
    lab[1] = 500.0f * (fx - fy);
    lab[2] = 200.0f * (fy - fz);
}

inline void from_lab(const float* lab, float* rgb) {
    const float fy = (lab[0] + 16.0f) / 116.0f, fx = lab[1] / 500.0f + fy, fz = sycl::fmax(fy - lab[2] / 200.0f, 0.0f);
    auto inv = [](float t) { return t > 0.2068966f ? t * t * t : (t - 4.0f / 29.0f) / 7.787f; };
    const float x = inv(fx) * 0.95047f, y = inv(fy), z = inv(fz) * 1.08883f;
    const float R = 3.2404813432005266f * x - 1.5371515162713185f * y - 0.4985363261688878f * z;
    const float G = -0.9692549499965682f * x + 1.8759900014898907f * y + 0.0415559265582928f * z;
    const float B = 0.0556466391351772f * x - 0.2040413383665112f * y + 1.0573110696453443f * z;
    auto gam = [](float v) { return v > 0.0031308f ? 1.055f * sycl::pow(v, 1.0f / 2.4f) - 0.055f : 12.92f * v; };
    rgb[0] = sycl::clamp(gam(R), 0.0f, 1.0f) * 255.0f;
    rgb[1] = sycl::clamp(gam(G), 0.0f, 1.0f) * 255.0f;
    rgb[2] = sycl::clamp(gam(B), 0.0f, 1.0f) * 255.0f;
}

// the work-group the statistics are summed in: 256, or the device's largest where that is smaller
size_t group_size() {
    const size_t most = queue().get_device().get_info<sycl::info::device::max_work_group_size>();
    return std::min<size_t>(256, most);
}

}  // namespace

Stats stats(const uint8_t* rgba, int64_t n, Space space, const float* mask, float below, const uint8_t* partner) {
    Stats s;
    if (n <= 0) return s;
    constexpr int kV = 7;   // the 3 channels' sums and squares' sums, and the count
    const size_t wg = group_size(), groups = (size_t) ((n + (int64_t) wg - 1) / (int64_t) wg);
    Buffers b;
    float* part = b.get<float>(groups * kV);
    const bool lab = space == Space::lab;
    done(queue().parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {
        const int64_t i = (int64_t) it.get_global_id(0);
        float v[kV] = {};
        if (i < n && rgba[i * 4 + 3] == 255 && (mask == nullptr || mask[i] < below) &&
            (partner == nullptr || partner[i * 4 + 3] == 255)) {
            float c[3] = {(float) rgba[i * 4], (float) rgba[i * 4 + 1], (float) rgba[i * 4 + 2]};
            if (lab) {
                float l[3];
                to_lab(c[0], c[1], c[2], l);
                c[0] = l[0];
                c[1] = l[1];
                c[2] = l[2];
            }
            for (int k = 0; k < 3; ++k) {
                v[k] = c[k];
                v[3 + k] = c[k] * c[k];
            }
            v[6] = 1.0f;
        }
        for (int k = 0; k < kV; ++k) {
            const float t = sycl::reduce_over_group(it.get_group(), v[k], sycl::plus<float>());
            if (it.get_local_id(0) == 0) part[it.get_group(0) * kV + k] = t;
        }
    }), "color: statistics");
    std::vector<float> host(groups * kV);
    gpu::copy(host.data(), part, host.size() * sizeof(float));
    for (size_t g = 0; g < groups; ++g) {
        for (int k = 0; k < 3; ++k) {
            s.sum[k] += host[g * kV + k];
            s.sq[k] += host[g * kV + 3 + k];
        }
        s.n += host[g * kV + 6];
    }
    return s;
}

bool transfer(uint8_t* rgba, int64_t n, Space space, const Stats& from, const Stats& to, double least) {
    if (n <= 0 || from.n < least || to.n < least || from.n <= 0.0 || to.n <= 0.0) return false;
    float scale[3], offset[3];
    for (int c = 0; c < 3; ++c) {
        const double mf = from.sum[c] / from.n, mt = to.sum[c] / to.n;
        const double sf = std::sqrt(std::max(0.0, from.sq[c] / from.n - mf * mf));
        const double st = std::sqrt(std::max(0.0, to.sq[c] / to.n - mt * mt));
        if (sf < 1e-3 || st < 1e-3) return false;
        scale[c] = (float) (st / sf);
        offset[c] = (float) (mt - st / sf * mf);
    }
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2], o0 = offset[0], o1 = offset[1], o2 = offset[2];
    const bool lab = space == Space::lab;
    done(queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        float c[3] = {(float) rgba[i * 4], (float) rgba[i * 4 + 1], (float) rgba[i * 4 + 2]};
        if (lab) {
            float l[3];
            to_lab(c[0], c[1], c[2], l);
            const float m[3] = {l[0] * s0 + o0, l[1] * s1 + o1, l[2] * s2 + o2};
            from_lab(m, c);
        } else {
            c[0] = sycl::clamp(c[0] * s0 + o0, 0.0f, 255.0f);
            c[1] = sycl::clamp(c[1] * s1 + o1, 0.0f, 255.0f);
            c[2] = sycl::clamp(c[2] * s2 + o2, 0.0f, 255.0f);
        }
        for (int k = 0; k < 3; ++k) rgba[i * 4 + k] = (uint8_t) sycl::round(c[k]);
    }), "color: transfer");
    return true;
}

uint64_t wavelet_bytes(int64_t w, int64_t h) { return (uint64_t) (w * h) * 3 * 4 * 3; }

void wavelet(uint8_t* rgba, const uint8_t* ref, int64_t w, int64_t h, int levels) {
    const int64_t n = w * h;
    if (n <= 0) return;
    Buffers b;
    // the image's low part and ref's, planar [3][h][w], and a scratch for the blurs' first pass
    float* lo = b.get<float>((size_t) (3 * n));
    float* lr = b.get<float>((size_t) (3 * n));
    float* tmp = b.get<float>((size_t) (3 * n));
    done(queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        for (int c = 0; c < 3; ++c) {
            lo[c * n + i] = (float) rgba[i * 4 + c];
            lr[c * n + i] = (float) ref[i * 4 + c];
        }
    }), "color: wavelet planes");
    // the kernel is separable, and repeating the edges is a clamp of each axis' index
    const auto blur = [&](float* p, int64_t d) {
        done(queue().parallel_for(sycl::range<3>(3, (size_t) h, (size_t) w), [=](sycl::id<3> id) {
            const int64_t c = (int64_t) id[0], y = (int64_t) id[1], x = (int64_t) id[2];
            const float* row = p + (c * h + y) * w;
            const int64_t xl = std::max<int64_t>(x - d, 0), xr = std::min<int64_t>(x + d, w - 1);
            tmp[(c * h + y) * w + x] = 0.25f * row[xl] + 0.5f * row[x] + 0.25f * row[xr];
        }), "color: wavelet rows");
        done(queue().parallel_for(sycl::range<3>(3, (size_t) h, (size_t) w), [=](sycl::id<3> id) {
            const int64_t c = (int64_t) id[0], y = (int64_t) id[1], x = (int64_t) id[2];
            const float* col = tmp + c * h * w + x;
            const int64_t yu = std::max<int64_t>(y - d, 0), yd = std::min<int64_t>(y + d, h - 1);
            p[(c * h + y) * w + x] = 0.25f * col[yu * w] + 0.5f * col[y * w] + 0.25f * col[yd * w];
        }), "color: wavelet columns");
    };
    for (int l = 0; l < levels; ++l) {
        blur(lo, (int64_t) 1 << l);
        blur(lr, (int64_t) 1 << l);
    }
    done(queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        for (int c = 0; c < 3; ++c) {
            const float v = (float) rgba[i * 4 + c] - lo[c * n + i] + lr[c * n + i];
            rgba[i * 4 + c] = (uint8_t) sycl::round(sycl::clamp(v, 0.0f, 255.0f));
        }
    }), "color: wavelet");
}

}  // namespace strata::image::color
