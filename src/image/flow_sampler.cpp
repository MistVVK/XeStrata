// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/flow_sampler.cpp - see include/strata/image/flow_sampler.hpp.

#include "strata/image/flow_sampler.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/moss_pipeline.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/kernels/elementwise.hpp"

#include "strata/image/gpu_buf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::image {

namespace gpu = strata::gpu;

namespace flow_sampler_detail {

// The common CFG combination. MOSS-Image's original SLG uses k as the base when no negative pass is present.
inline float velocity(float c, float u, float k, bool has_u, bool has_k, float cfg, float slg) {
    float v = has_u ? u + cfg * (c - u) : c;
    if (has_k) v = (has_u ? v : k) + slg * (c - k);
    return v;
}

void denoised(const float* x, const float* c, const float* u, const float* k, float cfg, float slg, float sigma,
              int64_t n, float* den, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float v = velocity(c[i], u != nullptr ? u[i] : 0.f, k != nullptr ? k[i] : 0.f,
                                 u != nullptr, k != nullptr, cfg, slg);
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

// Euler in velocity form, retained for the DiTs whose existing sampler steps directly by the velocity.
void velocity_euler_step(float* x, const float* c, const float* u, const float* k, float cfg, float slg, float ds,
                         int64_t n, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float v = velocity(c[i], u != nullptr ? u[i] : 0.f, k != nullptr ? k[i] : 0.f,
                                 u != nullptr, k != nullptr, cfg, slg);
        x[i] += ds * v;
    });
}

// One er_sde step after its stage-1 coefficients: x = a x + b den, then the stage-2 and stage-3 corrections from the
// last denoised values (dd: the first divided difference, kept; du: the second), then the noise
struct ErStep {
    float a, b, c2, inv2, c3, inv3, nz;
    int stage;
};

void er_sde_step(float* x, const float* den, const float* old_den, float* old_dd, const float* noise, const ErStep& e, int64_t n,
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
                 const std::function<void(int, const std::function<void(float*)>&)>& peek, Guidance* guide,
                 const std::function<void(const float*, float, float*)>& skip_forward, bool velocity_euler,
                 const FlowStepSkip* step_skip, bool unfused_euler) {
    if (step_skip != nullptr && guide != nullptr && guide->on())
        throw std::invalid_argument("image engine: step-window and sigma-window guidance cannot be combined");
    const bool guiding = guide != nullptr && guide->on();
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
    float* skipped = (guiding && skip_forward && !guide->skip_layers().empty()) || step_skip != nullptr ? get(N) : nullptr;   // skip layer guidance's pass
    auto lam = [](double s) { return s / (1.0 - s); };              // er_lambda: sigma / alpha
    auto scaler = [](double v) { return v * (std::exp(std::pow(v, 0.3)) + 10.0); };
    const int steps = (int) sig.size() - 1;
    try {
        for (int i = 0; i < steps; ++i) {
            const float s0 = sig[(size_t) i], s1 = sig[(size_t) i + 1];
            const bool step_skipping = step_skip != nullptr && step_skip->active(i);
            if (guiding) {   // the model at its own sigma (Detail Daemon's), the step by the sampler's
                const float sm = guide->model_sigma(s0);
                forward(x, sm, n, out);
                const bool skipping = skipped != nullptr && guide->skip_pass(sm);
                if (skipping) skip_forward(x, sm, skipped);
                guide->denoised(x, out, n == 2 ? out + N : nullptr, skipping ? skipped : nullptr, sm, den);
            } else {
                forward(x, s0, n, out);
                if (step_skipping) step_skip->forward(x, s0, skipped);
                if (!velocity_euler || er)
                    denoised(x, out, n == 2 ? out + N : nullptr, step_skipping ? skipped : nullptr, cfg,
                             step_skipping ? step_skip->scale : 0.f, s0, N, den, s_);
            }
            if (velocity_euler && !guiding && !er) {
                const float* v = out;
                if (unfused_euler) {
                    // Qwen-Image's old axpy kernels round each operation separately (strata_kernels disables
                    // contraction). Keep those bits, including CFG, rather than fusing them into the Euler kernel.
                    if (n == 2) {
                        kernels::add_inplace(out, out + N, N, s_, -1.f);
                        kernels::add_inplace(out + N, out, N, s_, cfg);
                        v = out + N;
                    }
                    kernels::add_inplace(x, v, N, s_, s1 - s0);
                } else {
                    velocity_euler_step(x, out, n == 2 ? out + N : nullptr, step_skipping ? skipped : nullptr, cfg,
                                        step_skipping ? step_skip->scale : 0.f, s1 - s0, N, s_);
                }
                if (after) after(x, s1);
                if (peek) peek(i + 1, [&](float* x0) {
                    if (unfused_euler) {
                        gpu::copy_async(x0, x, (size_t) N * 4, s_);
                        kernels::add_inplace(x0, v, N, s_, -s1);
                    } else {
                        denoised(x, out, n == 2 ? out + N : nullptr, step_skipping ? skipped : nullptr, cfg,
                                 step_skipping ? step_skip->scale : 0.f, s1, N, x0, s_);
                    }
                });
                if (step && !step(i + 1, steps)) throw QiStopped();
                continue;
            }
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
