// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
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

// sRGB (0 .. 255) <-> CIE Lab, D65, as kornia's rgb_to_lab and lab_to_rgb
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

namespace {

// the wavelet correction into planes [3][n] (0 .. 255, clamped): rgba less its low part plus ref's; `lo`, `lr`, `tmp`
// [3][n] each its scratch (the planes may be `lo`)
void wavelet_planes(const uint8_t* rgba, const uint8_t* ref, int64_t w, int64_t h, int levels, float* lo, float* lr,
                    float* tmp, float* planes) {
    const int64_t n = w * h;
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
    // the radius doubles a level, at most an eighth of the shorter side (the reference's bound: a tap stays inside)
    const int64_t limit = std::max<int64_t>(1, std::min(w, h) / 8);
    for (int l = 0; l < levels; ++l) {
        const int64_t d = std::min<int64_t>((int64_t) 1 << l, limit);
        blur(lo, d);
        blur(lr, d);
    }
    done(queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        for (int c = 0; c < 3; ++c)
            planes[c * n + i] = sycl::clamp((float) rgba[i * 4 + c] - lo[c * n + i] + lr[c * n + i], 0.0f, 255.0f);
    }), "color: wavelet");
}

// SeedVR2's sRGB (0 .. 1) <-> CIE Lab, D65 (the reference's color_fix: its matrices, CIE 15's delta 6 / 29)
constexpr float kDelta = 6.0f / 29.0f;

inline void srgb_to_lab(const float* rgb, float* lab) {
    float lin[3];
    for (int c = 0; c < 3; ++c)
        lin[c] = rgb[c] > 0.04045f ? sycl::pow((rgb[c] + 0.055f) / 1.055f, 2.4f) : rgb[c] / 12.92f;
    const float x = (0.4124564f * lin[0] + 0.3575761f * lin[1] + 0.1804375f * lin[2]) / 0.95047f;
    const float y = 0.2126729f * lin[0] + 0.7151522f * lin[1] + 0.0721750f * lin[2];
    const float z = (0.0193339f * lin[0] + 0.1191920f * lin[1] + 0.9503041f * lin[2]) / 1.08883f;
    auto f = [](float t) {
        return t > kDelta * kDelta * kDelta ? sycl::cbrt(sycl::fmax(t, 0.0f)) : t / (3.0f * kDelta * kDelta) + 4.0f / 29.0f;
    };
    const float fx = f(x), fy = f(y), fz = f(z);
    lab[0] = 116.0f * fy - 16.0f;
    lab[1] = 500.0f * (fx - fy);
    lab[2] = 200.0f * (fy - fz);
}

inline void lab_to_srgb(const float* lab, float* rgb) {
    const float fy = (lab[0] + 16.0f) / 116.0f, fx = fy + lab[1] / 500.0f, fz = fy - lab[2] / 200.0f;
    auto inv = [](float t) { return t > kDelta ? t * t * t : 3.0f * kDelta * kDelta * (t - 4.0f / 29.0f); };
    const float x = inv(fx) * 0.95047f, y = inv(fy), z = inv(fz) * 1.08883f;
    const float lin[3] = {3.2404542f * x - 1.5371385f * y - 0.4985314f * z,
                          -0.9692660f * x + 1.8760108f * y + 0.0415560f * z,
                          0.0556434f * x - 0.2040259f * y + 1.0572252f * z};
    for (int c = 0; c < 3; ++c)
        rgb[c] = sycl::clamp(lin[c] > 0.0031308f ? 1.055f * sycl::pow(sycl::fmax(lin[c], 0.0f), 1.0f / 2.4f) - 0.055f
                                                 : 12.92f * lin[c], 0.0f, 1.0f);
}

constexpr int64_t kBins = 65536;   // lab_match's histograms: a channel's range in this many bins

}  // namespace

void wavelet(uint8_t* rgba, const uint8_t* ref, int64_t w, int64_t h, int levels) {
    const int64_t n = w * h;
    if (n <= 0) return;
    Buffers b;
    float* lo = b.get<float>((size_t) (3 * n));
    float* lr = b.get<float>((size_t) (3 * n));
    float* tmp = b.get<float>((size_t) (3 * n));
    wavelet_planes(rgba, ref, w, h, levels, lo, lr, tmp, tmp);
    done(queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        for (int c = 0; c < 3; ++c) rgba[i * 4 + c] = (uint8_t) sycl::round(tmp[c * n + i]);
    }), "color: wavelet out");
}

uint64_t lab_match_bytes(int64_t w, int64_t h) {
    return (uint64_t) (w * h) * 3 * 4 * 4 + (uint64_t) (kBins * 6) * 4 + (uint64_t) (kBins * 3) * 4;
}

void lab_match(uint8_t* rgba, const uint8_t* ref, int64_t w, int64_t h, float lum_weight) {
    const int64_t n = w * h;
    if (n <= 0) return;
    Buffers b;
    float* lab = b.get<float>((size_t) (3 * n));    // the corrected image's Lab
    float* rlab = b.get<float>((size_t) (3 * n));   // ref's
    float* p0 = b.get<float>((size_t) (3 * n));
    float* p1 = b.get<float>((size_t) (3 * n));
    // the wavelet correction first (its planes in p1), then both in Lab
    wavelet_planes(rgba, ref, w, h, 5, lab, rlab, p0, p1);
    done(queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        float c[3] = {p1[i] / 255.0f, p1[n + i] / 255.0f, p1[2 * n + i] / 255.0f}, l[3];
        srgb_to_lab(c, l);
        const float r[3] = {(float) ref[i * 4] / 255.0f, (float) ref[i * 4 + 1] / 255.0f, (float) ref[i * 4 + 2] / 255.0f};
        float rl[3];
        srgb_to_lab(r, rl);
        for (int k = 0; k < 3; ++k) {
            lab[k * n + i] = l[k];
            rlab[k * n + i] = rl[k];
        }
    }), "color: lab");
    // each channel's range (the two images'), then their histograms
    std::vector<float> lo(3), hi(3);
    {
        const size_t wg = std::min<size_t>(256, queue().get_device().get_info<sycl::info::device::max_work_group_size>());
        const size_t groups = std::min<size_t>(1024, (size_t) ((2 * n + (int64_t) wg - 1) / (int64_t) wg));
        float* part = b.get<float>(groups * 6);
        done(queue().parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {
            float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
            for (int64_t i = (int64_t) it.get_global_id(0); i < 2 * n; i += (int64_t) (groups * wg)) {
                const float* src = i < n ? lab : rlab;
                const int64_t j = i < n ? i : i - n;
                for (int k = 0; k < 3; ++k) {
                    mn[k] = sycl::fmin(mn[k], src[k * n + j]);
                    mx[k] = sycl::fmax(mx[k], src[k * n + j]);
                }
            }
            for (int k = 0; k < 3; ++k) {
                const float a = sycl::reduce_over_group(it.get_group(), mn[k], sycl::minimum<float>());
                const float z = sycl::reduce_over_group(it.get_group(), mx[k], sycl::maximum<float>());
                if (it.get_local_id(0) == 0) {
                    part[it.get_group(0) * 6 + k] = a;
                    part[it.get_group(0) * 6 + 3 + k] = z;
                }
            }
        }), "color: lab range");
        std::vector<float> host(groups * 6);
        gpu::copy(host.data(), part, host.size() * 4);
        for (int k = 0; k < 3; ++k) {
            lo[(size_t) k] = 1e30f;
            hi[(size_t) k] = -1e30f;
            for (size_t g = 0; g < groups; ++g) {
                lo[(size_t) k] = std::min(lo[(size_t) k], host[g * 6 + (size_t) k]);
                hi[(size_t) k] = std::max(hi[(size_t) k], host[g * 6 + 3 + (size_t) k]);
            }
            if (hi[(size_t) k] <= lo[(size_t) k]) hi[(size_t) k] = lo[(size_t) k] + 1.0f;
        }
    }
    uint32_t* hist = b.get<uint32_t>((size_t) (kBins * 6));   // [image][channel][bin]
    done(queue().memset(hist, 0, (size_t) (kBins * 6) * 4), "color: lab histograms");
    const float l0 = lo[0], l1 = lo[1], l2 = lo[2];
    const float s0 = (float) kBins / (hi[0] - lo[0]), s1 = (float) kBins / (hi[1] - lo[1]),
                s2 = (float) kBins / (hi[2] - lo[2]);
    const auto bin = [](float v, float l, float sc) {
        return sycl::clamp((int64_t) ((v - l) * sc), (int64_t) 0, kBins - 1);
    };
    done(queue().parallel_for(sycl::range<1>((size_t) (2 * n)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], img = i < n ? 0 : 1, j = i < n ? i : i - n;
        const float* src = img == 0 ? lab : rlab;
        const float lc[3] = {l0, l1, l2}, sc[3] = {s0, s1, s2};
        for (int k = 0; k < 3; ++k) {
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> a(
                hist[(img * 3 + k) * kBins + bin(src[k * n + j], lc[k], sc[k])]);
            a.fetch_add(1u);
        }
    }), "color: lab histogram");
    // each source bin's value: ref's at the same rank (its bin's middle rank), by the reference's cumulative counts,
    // placed linearly within ref's bin; the image's pixels of that bin take it
    std::vector<uint32_t> hh((size_t) (kBins * 6));
    gpu::copy(hh.data(), hist, hh.size() * 4);
    std::vector<float> table((size_t) (3 * kBins));
    for (int k = 0; k < 3; ++k) {
        const uint32_t* hs = hh.data() + (size_t) k * kBins;
        const uint32_t* hr = hh.data() + (size_t) (3 + k) * kBins;
        const double width = (double) (hi[(size_t) k] - lo[(size_t) k]) / (double) kBins;
        double cs = 0.0, cr = 0.0;
        int64_t j = 0;
        for (int64_t bi = 0; bi < kBins; ++bi) {
            const double rank = cs + std::max(0.0, ((double) hs[bi] - 1.0) / 2.0);
            while (j < kBins - 1 && cr + hr[j] <= rank) cr += hr[j++];
            const double frac = hr[j] > 0 ? (rank - cr + 0.5) / (double) hr[j] : 0.5;
            table[(size_t) (k * kBins + bi)] = (float) ((double) lo[(size_t) k] + ((double) j + std::min(1.0, frac)) * width);
            cs += hs[bi];
        }
    }
    float* dt = b.get<float>((size_t) (3 * kBins));
    gpu::copy(dt, table.data(), table.size() * 4);
    const float lw = lum_weight;
    done(queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const float lc[3] = {l0, l1, l2}, sc[3] = {s0, s1, s2};
        float m[3];
        for (int k = 0; k < 3; ++k) m[k] = dt[k * kBins + bin(lab[k * n + i], lc[k], sc[k])];
        m[0] = lab[i] * lw + m[0] * (1.0f - lw);   // the lightness mostly the restored one's
        float rgb[3];
        lab_to_srgb(m, rgb);
        for (int c = 0; c < 3; ++c) rgba[i * 4 + c] = (uint8_t) sycl::round(rgb[c] * 255.0f);
    }), "color: lab match");
}

}  // namespace strata::image::color
