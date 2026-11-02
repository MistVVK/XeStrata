// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/flow_sampler.cpp - see include/strata/image/flow_sampler.hpp.

#include "strata/image/flow_sampler.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/moss_pipeline.hpp"
#include "strata/image/pipeline.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::image {

namespace gpu = strata::gpu;

namespace flow_sampler_detail {

sycl::queue& q_of(void* s) { return core::Runtime::get().stream(s); }

// den = x - sigma * v, v the prompt's velocity c or with guidance u + cfg (c - u)
void denoised(const float* x, const float* c, const float* u, float cfg, float sigma, int64_t n, float* den, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float v = u != nullptr ? u[i] + cfg * (c[i] - u[i]) : c[i];
        den[i] = x[i] - sigma * v;
    });
}

// Euler: x += (s1 - s0) (x - den) / s0, which at s1 = 0 is den
void euler(float* x, const float* den, float s0, float s1, int64_t n, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        x[i] = s1 == 0.f ? den[i] : x[i] + (s1 - s0) * (x[i] - den[i]) / s0;
    });
}

// One er_sde step after its stage-1 coefficients: x = a x + b den, then the stage-2 and stage-3 corrections from the
// last denoised values (dd: the first divided difference, kept; du: the second), then the noise
struct ErStep {
    float a, b, c2, inv2, c3, inv3, nz;
    int stage;
};

void er_sde_step(float* x, const float* den, const float* old_den, float* old_dd, const float* noise, ErStep e, int64_t n,
                 void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        float v = e.a * x[i] + e.b * den[i];
        if (e.stage >= 2) {
            const float dd = (den[i] - old_den[i]) * e.inv2;
            v += e.c2 * dd;
            if (e.stage >= 3) v += e.c3 * (dd - old_dd[i]) * e.inv3;
            old_dd[i] = dd;
        }
        if (noise != nullptr) v += e.nz * noise[i];
        x[i] = v;
    });
}

}  // namespace flow_sampler_detail

using namespace flow_sampler_detail;

void flow_sample(void* s_, const FlowForward& forward, float* x, int64_t N, int n, const std::vector<float>& sigmas_in,
                 float cfg, bool er, float shift, uint64_t seed, const std::function<bool(int, int)>& step,
                 const std::function<void(float*, float)>& after,
                 const std::function<void(int, const std::function<void(float*)>&)>& peek) {
    std::vector<float> sig = sigmas_in;
    // er_sde's first sigma below 1 (offset_first_sigma_for_snr: the schedule's at 1e-4 of the way from the top)
    if (er && !sig.empty() && sig[0] >= 1.f) {
        const float t = 1.f - 1e-4f;
        sig[0] = shift * t / (1.f + (shift - 1.f) * t);
    }
    std::vector<float*> bufs;
    auto get = [&](int64_t count) {
        float* p = nullptr;
        if (!gpu::alloc_device(&p, (size_t) count * 4)) {
            for (float* b : bufs) gpu::free(b);
            throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
        }
        bufs.push_back(p);
        return p;
    };
    float* out = get(n * N);
    float* den = get(N);
    float* old_den = er ? get(N) : nullptr;
    float* old_dd = er ? get(N) : nullptr;
    float* noise = er ? get(N) : nullptr;
    auto lam = [](double s) { return s / (1.0 - s); };              // er_lambda: sigma / alpha
    auto scaler = [](double v) { return v * (std::exp(std::pow(v, 0.3)) + 10.0); };
    const int steps = (int) sig.size() - 1;
    try {
        for (int i = 0; i < steps; ++i) {
            const float s0 = sig[(size_t) i], s1 = sig[(size_t) i + 1];
            forward(x, s0, n, out);
            denoised(x, out, n == 2 ? out + N : nullptr, cfg, s0, N, den, s_);
            if (peek) peek(i + 1, [&](float* x0) { gpu::copy_async(x0, den, (size_t) N * 4, s_); });
            if (!er || s1 == 0.f) {
                euler(x, den, s0, s1, N, s_);
            } else {
                const int stage = std::min(3, i + 1);
                const double ls = lam(s0), lt = lam(s1), as = s0 / ls, at = s1 / lt;
                const double r = scaler(lt) / scaler(ls);
                ErStep e{};
                e.a = (float) (at / as * r);
                e.b = (float) (at * (1.0 - r));
                e.stage = stage;
                const double dt = lt - ls, stepl = -dt / 200.0;
                if (stage >= 2) {
                    double sum = 0.0, sum_u = 0.0;
                    for (int k = 0; k < 200; ++k) {
                        const double pos = lt + (double) k * stepl, sp = scaler(pos);
                        sum += 1.0 / sp;
                        sum_u += (pos - ls) / sp;
                    }
                    const double ss = sum * stepl, su = sum_u * stepl;
                    const double lprev = lam(sig[(size_t) i - 1]);
                    e.c2 = (float) (at * (dt + ss * scaler(lt)));
                    e.inv2 = (float) (1.0 / (ls - lprev));
                    if (stage >= 3) {
                        const double lprev2 = lam(sig[(size_t) i - 2]);
                        e.c3 = (float) (at * (dt * dt / 2.0 + su * scaler(lt)));
                        e.inv3 = (float) (1.0 / ((ls - lprev2) / 2.0));
                    }
                }
                const double var = lt * lt - ls * ls * r * r;
                e.nz = (float) (at * (var > 0.0 ? std::sqrt(var) : 0.0));
                const std::vector<float> z = torch_cpu_randn(seed + 1 + (uint64_t) i, (uint64_t) N);
                gpu::copy(noise, z.data(), (size_t) N * 4);
                er_sde_step(x, den, old_den, old_dd, noise, e, N, s_);
            }
            if (er) gpu::copy_async(old_den, den, (size_t) N * 4, s_);
            if (after) after(x, s1);
            if (step && !step(i + 1, steps)) throw QiStopped();
        }
    } catch (...) {
        gpu::stream_sync(s_);
        for (float* b : bufs) gpu::free(b);
        throw;
    }
    gpu::stream_sync(s_);
    for (float* b : bufs) gpu::free(b);
}

}  // namespace strata::image
