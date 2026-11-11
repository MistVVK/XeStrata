// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2024 Sahand Ahmadian
// SPDX-FileCopyrightText: 2024 Jonseed
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/image/guidance.cpp - see include/strata/image/guidance.hpp.

#include "strata/image/guidance.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/platform/env.hpp"

#include "strata/image/gpu_buf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr int kSums = 4;   // the most sums one reduction makes
constexpr double kPi = 3.14159265358979323846;   // MSVC's <cmath> has no M_PI without a define

// point k of `count` evenly spaced from `from` to `to`, both ends exact (numpy's linspace, value for value)
double even_point(double from, double to, int64_t k, int64_t count) {
    if (count <= 1 || k == 0) return from;
    if (k == count - 1) return to;
    return from + (to - from) * (double) k / (double) (count - 1);
}

// Python's round() to an int: halves to even
int64_t py_round(double x) {
    const int mode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    const double r = std::nearbyint(x);
    std::fesetround(mode);
    return (int64_t) r;
}

// the sums of K values a position adds (f(i, v)), in a work-group at a time over a fixed grid, then the groups' sums
// by one group: the same order each time, and on the device (the kernels after read them there, no wait for the host)
template <int K, class F>
void sums(void* s, int64_t n, size_t wg, size_t groups, float* part, float* out, F f) {
    sycl::queue& q = q_of(s);
    const int64_t stride = (int64_t) (wg * groups);
    q.parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {
        float v[K] = {};
        for (int64_t i = (int64_t) it.get_global_id(0); i < n; i += stride) f(i, v);
        for (int k = 0; k < K; ++k) {
            const float t = sycl::reduce_over_group(it.get_group(), v[k], sycl::plus<float>());
            if (it.get_local_id(0) == 0) part[it.get_group(0) * K + k] = t;
        }
    });
    q.parallel_for(sycl::nd_range<1>(wg, wg), [=](sycl::nd_item<1> it) {
        for (int k = 0; k < K; ++k) {
            float v = 0.0f;
            for (size_t g = it.get_local_id(0); g < groups; g += wg) v += part[g * K + k];
            const float t = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
            if (it.get_local_id(0) == 0) out[k] = t;
        }
    });
}

}  // namespace

std::vector<float> detail_daemon_schedule(const GuidanceSpec& g, int steps) {
    // Each step's multiplier on its own: the offsets before the window and after it; inside, a climb from the first
    // step to the peak and a descent from the peak to the last, each eased by half a cosine (when smooth), raised to
    // the exponent and spread from its offset to the amount; the fade lowers it.  The steps are Python's round() of
    // the window's shares and a ramp's points numpy's linspace's, in doubles, so the values are Detail Daemon's own,
    // bit for bit.
    if (steps <= 0) return {};
    const double first = std::min(g.dd_start, g.dd_end), last = g.dd_end, span = (double) (steps - 1);
    const int64_t s0 = py_round(first * span), peak = py_round((first + g.dd_bias * (last - first)) * span),
                  s1 = py_round(last * span);
    const auto ramp = [&](double t, double base) {
        const double eased = g.dd_smooth ? 0.5 * (1.0 - std::cos(t * kPi)) : t;
        return std::pow(eased, (double) g.dd_exponent) * ((double) g.dd_amount - base) + base;
    };
    std::vector<float> out((size_t) steps);
    for (int64_t k = 0; k < steps; ++k) {
        double v;
        if (k < s0)
            v = g.dd_start_offset;
        else if (k > s1)
            v = g.dd_end_offset;
        else if (k < peak)   // a climb of two points or more: it reaches 1, so it is spread
            v = ramp(even_point(0.0, 1.0, k - s0, peak - s0 + 1), g.dd_start_offset);
        else                 // the descent, from 1 at the peak (the peak's own point is the descent's)
            v = ramp(even_point(1.0, 0.0, k - peak, s1 - peak + 1), g.dd_end_offset);
        out[(size_t) k] = (float) (v * (1.0 - (double) g.dd_fade));
    }
    return out;
}

Guidance::Guidance(const GuidanceSpec& g, void* stream, int64_t n, int64_t channels, bool token_major, float cfg,
                   const std::vector<float>& sigmas)
    : g_(g), s_(stream), n_(n), ch_(channels), tm_(token_major), on_(g.any()), cfg_(cfg), sig_(sigmas) {
    if (!on_) return;
    if (g_.dd) dd_ = detail_daemon_schedule(g_, (int) sig_.size() - 1);
    const sycl::device dev = q_of(s_).get_device();
    wg_ = std::min<size_t>(256, dev.get_info<sycl::info::device::max_work_group_size>());
    const size_t cus = dev.get_info<sycl::info::device::max_compute_units>();
    groups_ = std::max<size_t>(1, std::min<size_t>(((size_t) n_ + wg_ - 1) / wg_, 4 * std::max<size_t>(cus, 1)));
    auto get = [&](float*& p, size_t count) {
        if (!gpu::alloc_device(&p, count * 4)) {
            const std::string e = gpu::last_error();
            release();
            throw std::runtime_error("image engine: GPU memory for the guidance: " + e);
        }
    };
    get(cd_, (size_t) n_);
    get(ud_, (size_t) n_);
    if (g_.apg && g_.apg_momentum != 0.0f) {   // the sum starts at 0 (one Guidance is one sampling)
        get(run_, (size_t) n_);
        q_of(s_).memset(run_, 0, (size_t) n_ * 4);
    }
    get(part_, groups_ * kSums);
    get(red_, 8);
}

Guidance::~Guidance() { release(); }

void Guidance::release() {
    if (cd_ != nullptr || ud_ != nullptr || part_ != nullptr) gpu::stream_sync(s_);
    for (float** p : {&cd_, &ud_, &run_, &part_, &red_}) {
        if (*p != nullptr) gpu::free(*p);
        *p = nullptr;
    }
}

float Guidance::model_sigma(float sigma) const {
    if (!g_.dd || dd_.size() < 2 || sig_.size() < 2) return sigma;
    // a sigma past the schedule's (above its first, or within 1e-5 of its end) is asked as it is
    if (sigma > sig_.front() || sigma < (float) ((double) sig_.back() + 1e-05)) return sigma;
    // the multiplier at the sigma: the steps' sigmas fall, so the first at or below it and the one before bound it,
    // and between the two it goes straight; below the last step's, that step's
    const size_t len = dd_.size();
    float d = 0.0f;
    if (sigma > 0.0f && sigma >= sig_.back()) {
        size_t below = 0;
        while (below < len && sig_[below] > sigma) ++below;
        if (below == len) {
            d = dd_[len - 1];
        } else if (below == 0 || sig_[below] == sigma) {
            d = dd_[below];
        } else {
            const float at = sig_[below], above = sig_[below - 1];
            const float r = std::clamp((sigma - at) / (above - at), 0.0f, 1.0f);
            d = dd_[below] + r * (dd_[below - 1] - dd_[below]);
        }
    }
    // the model sees a sigma lowered by a tenth of the multiplier per unit of guidance scale (the override, else the
    // request's), never below 1e-6 of it
    const double scale = g_.dd_cfg > 0.0f ? (double) g_.dd_cfg : (double) cfg_;
    return sigma * (float) std::max(1e-06, 1.0 - (double) d * 0.1 * scale);
}

bool Guidance::skip_pass(float sigma) const {
    return !g_.slg_layers.empty() && g_.slg_scale > 0.0f && sigma >= g_.slg_sigma_end && sigma <= g_.slg_sigma_start;
}

void Guidance::denoised(const float* x, const float* c, const float* u, const float* k, float sigma, float* den) {
    sycl::queue& q = q_of(s_);
    const int64_t n = n_;
    const float s = sigma, w = cfg_;
    float *cd = cd_, *ud = ud_, *red = red_;
    const bool apg = g_.apg && u != nullptr, zero = g_.zero_star && u != nullptr;
    // the denoised predictions; for APG also the step's difference summed (red: |r|^2, |c|^2, <r, c>), for CFGZero*
    // the velocities' products (red + 4: <v_c, v_u>, |v_u|^2)
    if (apg) {
        const float mom = g_.apg_momentum;
        float* run = run_;
        sums<3>(s_, n, wg_, groups_, part_, red, [=](int64_t i, float* v) {
            const float pc = x[i] - s * c[i], pu = x[i] - s * u[i];
            cd[i] = pc;
            ud[i] = pu;
            float r = pc - pu;
            if (run != nullptr) run[i] = r = r + mom * run[i];
            v[0] += r * r;
            v[1] += pc * pc;
            v[2] += r * pc;
        });
    } else {
        q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const size_t i = id[0];
            cd[i] = x[i] - s * c[i];
            if (u != nullptr) ud[i] = x[i] - s * u[i];
        });
    }
    if (zero)
        sums<2>(s_, n, wg_, groups_, part_, red + 4, [=](int64_t i, float* v) {
            v[0] += c[i] * u[i];
            v[1] += u[i] * u[i];
        });
    const float slg = k != nullptr ? g_.slg_scale : 0.0f;
    const float eta = g_.apg_eta, cap = g_.apg_norm_threshold;
    float* run = run_;
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float pc = cd[i];
        // CFGZero* scales the negative prediction by alpha before the combination
        const float alpha = zero ? red[4] / (red[5] + 1e-8f) : 1.0f;
        float d = pc;
        if (apg) {
            // r's length capped, then its component along c (r . c / |c|^2) weighed by eta; with CFGZero* as well,
            // the same alpha's change of the plain combination added
            const float len = sycl::sqrt(red[0]);
            const float keep = cap > 0.0f && len > cap ? cap / len : 1.0f;
            const float along = keep * red[2] / sycl::fmax(red[1], 1e-24f);
            const float r = (run != nullptr ? run[i] : pc - ud[i]) * keep;
            d = pc + w * (r + (eta - 1.0f) * along * pc) + (1.0f - w) * (alpha - 1.0f) * ud[i];
        } else if (u != nullptr) {
            const float pu = alpha * ud[i];
            d = pu + w * (pc - pu);
        }
        if (slg != 0.0f) d += slg * (pc - (x[i] - s * k[i]));
        den[i] = d;
    });
    if (k != nullptr && g_.slg_rescale != 0.0f) {   // den towards c's spread: the sample standard deviations' ratio
        sums<4>(s_, n, wg_, groups_, part_, red, [=](int64_t i, float* v) {
            v[0] += cd[i];
            v[1] += cd[i] * cd[i];
            v[2] += den[i];
            v[3] += den[i] * den[i];
        });
        const float mix = g_.slg_rescale;
        q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const float nn = (float) n;
            const auto spread = [=](float sum, float sq) {
                return sycl::sqrt(sycl::fmax(sq - sum * sum / nn, 0.0f) / (nn - 1.0f));
            };
            den[id[0]] *= 1.0f + mix * (spread(red[0], red[1]) / spread(red[2], red[3]) - 1.0f);
        });
    }
    // CFG norm: at each position, den no longer than c over the channels, then the strength
    if (g_.cfgnorm >= 0.0f) {
        const int64_t C = ch_, P = n / ch_;
        const bool tm = tm_;
        const float st = g_.cfgnorm;
        q.parallel_for(sycl::range<1>((size_t) P), [=](sycl::id<1> id) {
            const int64_t p = (int64_t) id[0];
            const int64_t base = tm ? p * C : p, step = tm ? 1 : P;
            float lc = 0.0f, ld = 0.0f;
            for (int64_t j = 0; j < C; ++j) {
                lc += cd[base + j * step] * cd[base + j * step];
                ld += den[base + j * step] * den[base + j * step];
            }
            const float f = st * (ld > lc ? sycl::sqrt(lc / ld) : 1.0f);
            for (int64_t j = 0; j < C; ++j) den[base + j * step] *= f;
        });
    }
    dump(x, c, u, k, sigma, den);
    ++calls_;
}

void Guidance::dump(const float* x, const float* c, const float* u, const float* k, float sigma, const float* den) {
    // STRATA_GUIDANCE_DUMP=<folder>: each call's inputs and result as raw FP32 files (g<run>_step<i>_<x|c|u|k|den>.f32,
    // a run one guided sampling of this process) and a line of meta.txt (the run, the step, the model's sigma, cfg, n,
    // channels, layout): what checks the methods against a reference on the same values
    const char* dir = strata::env_get("GUIDANCE_DUMP");
    if (dir == nullptr || *dir == '\0') return;
    gpu::stream_sync(s_);
    std::vector<float> h((size_t) n_);
    static int runs = 0;   // the samplings guided in this process, each a prefix of its own
    if (calls_ == 0) seq_ = runs++;
    const std::string base = std::string(dir) + "/g" + std::to_string(seq_) + "_step" + std::to_string(calls_);
    for (const auto& [name, p] : {std::pair<const char*, const float*>{"x", x}, {"c", c}, {"u", u}, {"k", k},
                                  {"den", den}}) {
        if (p == nullptr) continue;
        gpu::copy(h.data(), p, h.size() * 4);
        if (FILE* f = std::fopen((base + "_" + name + ".f32").c_str(), "wb")) {
            std::fwrite(h.data(), 4, h.size(), f);
            std::fclose(f);
        }
    }
    if (FILE* f = std::fopen((std::string(dir) + "/meta.txt").c_str(), "a")) {
        std::fprintf(f, "%d %d %.9g %.9g %lld %lld %d\n", seq_, calls_, (double) sigma, (double) cfg_, (long long) n_,
                     (long long) ch_, tm_ ? 1 : 0);
        std::fclose(f);
    }
}

}  // namespace strata::image
