// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/guidance.cpp - see include/strata/image/guidance.hpp.

#include "strata/image/guidance.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"

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

sycl::queue& q_of(void* s) { return core::Runtime::get().stream(s); }

constexpr int kSums = 4;   // the most sums one reduction makes
constexpr double kPi = 3.14159265358979323846;   // MSVC's <cmath> has no M_PI without a define

// numpy's linspace(a, b, num)
std::vector<double> linspace(double a, double b, int64_t num) {
    std::vector<double> v((size_t) std::max<int64_t>(num, 0));
    if (num == 1) v[0] = a;
    for (int64_t i = 0; num > 1 && i < num; ++i) v[(size_t) i] = a + (b - a) * (double) i / (double) (num - 1);
    if (num > 1) v.back() = b;
    return v;
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
    // make_detail_daemon_schedule, in doubles as numpy works it
    const double start = std::min(g.dd_start, g.dd_end), end = g.dd_end;
    const double mid = start + g.dd_bias * (end - start);
    std::vector<double> m((size_t) std::max(steps, 0), 0.0);
    if (steps <= 0) return {};
    const int64_t si = py_round(start * (steps - 1)), mi = py_round(mid * (steps - 1)), ei = py_round(end * (steps - 1));
    auto shape = [&](std::vector<double> v, double offset) {
        for (double& x : v) {
            if (g.dd_smooth) x = 0.5 * (1.0 - std::cos(x * kPi));
            x = std::pow(x, (double) g.dd_exponent);
        }
        if (std::any_of(v.begin(), v.end(), [](double x) { return x != 0.0; }))
            for (double& x : v) x = x * ((double) g.dd_amount - offset) + offset;
        return v;
    };
    const std::vector<double> sv = shape(linspace(0.0, 1.0, mi - si + 1), g.dd_start_offset);
    const std::vector<double> ev = shape(linspace(1.0, 0.0, ei - mi + 1), g.dd_end_offset);
    for (size_t k = 0; k < sv.size() && si + (int64_t) k < steps; ++k) m[(size_t) si + k] = sv[k];
    for (size_t k = 0; k < ev.size() && mi + (int64_t) k < steps; ++k) m[(size_t) mi + k] = ev[k];
    for (int64_t k = 0; k < std::min<int64_t>(si, steps); ++k) m[(size_t) k] = g.dd_start_offset;
    for (int64_t k = ei + 1; k < steps; ++k) m[(size_t) k] = g.dd_end_offset;
    std::vector<float> out(m.size());
    for (size_t k = 0; k < m.size(); ++k) out[k] = (float) (m[k] * (1.0 - (double) g.dd_fade));
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
    if (g_.apg && g_.apg_momentum != 0.0f) get(run_, (size_t) n_);
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
    // DetailDaemonSamplerNode's model_wrapper: outside the schedule's sigmas, the sigma as it is
    const float smax = sig_.front(), smin = (float) ((double) sig_.back() + 1e-05);
    if (!(smin <= sigma && sigma <= smax)) return sigma;
    // get_dd_schedule: the nearest of the steps' sigmas, else the two around it interpolated
    const size_t len = dd_.size();
    float d = 0.0f;
    if (sigma > 0.0f && sig_.back() <= sigma && sigma <= sig_.front()) {
        size_t idx = 0;
        float best = std::fabs(sig_[0] - sigma);
        for (size_t k = 1; k + 1 < sig_.size(); ++k) {
            const float dk = std::fabs(sig_[k] - sigma);
            if (dk < best) {
                best = dk;
                idx = k;
            }
        }
        if ((idx == 0 && sigma >= sig_[0]) || (idx == len - 1 && sigma <= sig_[sig_.size() - 2]) || best == 0.0f) {
            d = dd_[idx];
        } else {
            const size_t lo = sigma > sig_[idx] ? idx : idx + 1, hi = sigma > sig_[idx] ? idx - 1 : idx;
            const float nl = sig_[lo], nh = sig_[hi];
            if (nh - nl == 0.0f) {
                d = dd_[lo];
            } else {
                const float r = std::clamp((sigma - nl) / (nh - nl), 0.0f, 1.0f);
                d = dd_[lo] + r * (dd_[hi] - dd_[lo]);   // torch.lerp
            }
        }
    }
    const double cfg = g_.dd_cfg > 0.0f ? (double) g_.dd_cfg : (double) cfg_;
    const double adj = (double) d * 0.1;
    return sigma * (float) std::max(1e-06, 1.0 - adj * cfg);
}

bool Guidance::skip_pass(float sigma) const {
    return !g_.slg_layers.empty() && g_.slg_scale > 0.0f && sigma >= g_.slg_sigma_end && sigma <= g_.slg_sigma_start;
}

void Guidance::denoised(const float* x, const float* c, const float* u, const float* k, float sigma, float* den) {
    sycl::queue& q = q_of(s_);
    const int64_t n = n_;
    const float s = sigma, cfg = cfg_;
    float *cd = cd_, *ud = ud_, *red = red_;
    const bool apg = g_.apg && u != nullptr, zero = g_.zero_star && u != nullptr;
    // the predictions (and APG's guidance with its sums: |g|^2, |cond|^2, g . cond)
    if (apg) {
        if (prev_sigma_ >= 0.0f && sigma > prev_sigma_) have_run_ = false;   // a new sampling: its sum from 0
        prev_sigma_ = sigma;
        const float mom = g_.apg_momentum;
        float* run = run_;
        const bool first = !have_run_;
        have_run_ = true;
        sums<3>(s_, n, wg_, groups_, part_, red, [=](int64_t i, float* v) {
            const float a = x[i] - s * c[i], b = x[i] - s * u[i];
            cd[i] = a;
            ud[i] = b;
            float gd = a - b;
            if (run != nullptr) {
                gd = first ? gd : mom * run[i] + gd;
                run[i] = gd;
            }
            v[0] += gd * gd;
            v[1] += a * a;
            v[2] += gd * a;
        });
        const float eta = g_.apg_eta, thr = g_.apg_norm_threshold;
        q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const float gn = sycl::sqrt(red[0]);
            const float sc = thr > 0.0f ? sycl::fmin(1.0f, thr / gn) : 1.0f;
            const float gd = (run != nullptr ? run[i] : cd[i] - ud[i]) * sc;
            // project(guidance, cond): cond normalized (F.normalize, eps 1e-12), the guidance's part along it
            const float cn = sycl::fmax(sycl::sqrt(red[1]), 1e-12f);
            const float v1 = cd[i] / cn;
            const float par = sc * red[2] / cn * v1;
            const float mg = (gd - par) + eta * par;
            cd[i] = (ud[i] + mg) + (cd[i] - ud[i]) / cfg;
        });
    } else {
        q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const size_t i = id[0];
            cd[i] = x[i] - s * c[i];
            if (u != nullptr) ud[i] = x[i] - s * u[i];
        });
    }
    // CFGZero*'s alpha: <x - cond, x - uncond> / (|x - uncond|^2 + 1e-8)
    if (zero)
        sums<2>(s_, n, wg_, groups_, part_, red + 4, [=](int64_t i, float* v) {
            const float a = x[i] - cd[i], b = x[i] - ud[i];
            v[0] += a * b;
            v[1] += b * b;
        });
    // CFG, CFGZero*, skip layer guidance
    const float slg = k != nullptr ? g_.slg_scale : 0.0f;
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        float d = u != nullptr ? ud[i] + (cd[i] - ud[i]) * cfg : cd[i];
        if (zero) {
            const float alpha = red[4] / (red[5] + 1e-8f);
            d = d + ud[i] * (alpha - 1.0f) + cfg * ud[i] * (1.0f - alpha);
        }
        if (slg != 0.0f) d = d + (cd[i] - (x[i] - s * k[i])) * slg;
        den[i] = d;
    });
    if (k != nullptr && g_.slg_rescale != 0.0f) {   // den's spread towards cond's (torch.std: unbiased)
        sums<4>(s_, n, wg_, groups_, part_, red, [=](int64_t i, float* v) {
            v[0] += cd[i];
            v[1] += cd[i] * cd[i];
            v[2] += den[i];
            v[3] += den[i] * den[i];
        });
        const float rs = g_.slg_rescale;
        q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const float nn = (float) n, mc = red[0] / nn, md = red[2] / nn;
            const float sc = sycl::sqrt(sycl::fmax(red[1] - nn * mc * mc, 0.0f) / (nn - 1.0f));
            const float sd = sycl::sqrt(sycl::fmax(red[3] - nn * md * md, 0.0f) / (nn - 1.0f));
            const float f = rs * (sc / sd) + (1.0f - rs);
            den[id[0]] *= f;
        });
    }
    // CFGNorm: each position's norm over the channels, den's no larger than cond's, then the strength
    if (g_.cfgnorm >= 0.0f) {
        const int64_t C = ch_, P = n / ch_;
        const bool tm = tm_;
        const float st = g_.cfgnorm;
        q.parallel_for(sycl::range<1>((size_t) P), [=](sycl::id<1> id) {
            const int64_t p = (int64_t) id[0];
            auto at = [=](int64_t ch) { return tm ? p * C + ch : ch * P + p; };
            float nc = 0.0f, nd = 0.0f;
            for (int64_t ch = 0; ch < C; ++ch) {
                nc += cd[at(ch)] * cd[at(ch)];
                nd += den[at(ch)] * den[at(ch)];
            }
            const float sc = sycl::clamp(sycl::sqrt(nc) / (sycl::sqrt(nd) + 1e-8f), 0.0f, 1.0f);
            for (int64_t ch = 0; ch < C; ++ch) den[at(ch)] = den[at(ch)] * sc * st;
        });
    }
    dump(x, c, u, k, sigma, den);
    ++calls_;
}

void Guidance::dump(const float* x, const float* c, const float* u, const float* k, float sigma, const float* den) {
    // STRATA_GUIDANCE_DUMP=<folder>: each call's inputs and result as raw FP32 files (g<run>_step<i>_<x|c|u|k|den>.f32,
    // a run one guided sampling of this process) and a line of meta.txt (the run, the step, the model's sigma, cfg, n,
    // channels, layout): what checks the methods against ComfyUI's nodes on the same values
    const char* dir = std::getenv("STRATA_GUIDANCE_DUMP");
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

void guidance_euler(float* x, const float* den, float s0, float s1, int64_t n, void* stream) {
    q_of(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        x[i] = s1 == 0.0f ? den[i] : x[i] + (s1 - s0) * (x[i] - den[i]) / s0;
    });
}

}  // namespace strata::image
