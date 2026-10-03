// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/rope.cpp - the Xe ports of Strata's src/kernels/cuda/rope.cu (NEOX partial RoPE over a host-built table)
// and Strata's src/kernels/cuda/native_rope.cu (the pinned ggml text IMRoPE), with the M-RoPE table they share.
//
// native_rope follows ggml/src/ggml-cuda/rope.cu (rope_multi/rope_yarn) at llama.cpp 3cf03257; MIT License,
// Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE.  CUDA compiled it with --use_fast_math;
// here pow, cos and sin are the precise SYCL functions.
#include "strata/kernels/rope.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/core/runtime.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

// One Xe device: the per-device table array of the CUDA build collapses to one pointer.
std::atomic<const int32_t*> mrope_tab{nullptr};
std::atomic<bool> native_enabled{false};

bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}

}  // namespace

namespace {
RopeScaling g_rope_scaling;   // one writer at startup, before any table or graph (rope_scaling.hpp)
}  // namespace
void rope_scaling_set(const RopeScaling& scaling) { g_rope_scaling = scaling; }
const RopeScaling& rope_scaling() { return g_rope_scaling; }

namespace {
RopeTab g_rope_tab;
RopeScaling g_rope_tab_scaling;
bool same_scaling(const RopeScaling& a, const RopeScaling& b) {
    return a.type == b.type && a.freq_base == b.freq_base && a.factor == b.factor && a.freq_scale_in == b.freq_scale_in &&
           a.orig_ctx == b.orig_ctx && a.ext_factor == b.ext_factor && a.attn_factor == b.attn_factor &&
           a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow;
}
}  // namespace
void rope_table_set(const float* cos_tab, const float* sin_tab, int max_pos, const RopeScaling& scaling) {
    g_rope_tab = RopeTab{cos_tab, sin_tab, max_pos};
    g_rope_tab_scaling = scaling;
}
void rope_table_release(const float* cos_tab) {
    if (g_rope_tab.cos == cos_tab) g_rope_tab = RopeTab{};
}
RopeTab rope_table_for(const RopeScaling& scaling) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ROPE_TABLE");
        return v != nullptr && v[0] == '1';
    }();
    return on && same_scaling(scaling, g_rope_tab_scaling) ? g_rope_tab : RopeTab{};
}

void mrope_table_set(const int32_t* device_table) { mrope_tab.store(device_table, std::memory_order_relaxed); }
const int32_t* mrope_table() { return mrope_tab.load(std::memory_order_relaxed); }

void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            // float64 throughout, in the reference's order: inv, then ang, then cos/sin
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}

void build_rope_table(int n_rot, const RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab) {
    if (sc.type == RopeScalingType::None) {
        build_rope_table(n_rot, sc.freq_base, max_pos, cos_tab, sin_tab);   // the original loop, as it is
        return;
    }
    const int half = n_rot / 2;
    const double fs = sc.freq_scale();
    const double ms = sc.mscale();
    double cd[2];
    sc.corr_dims(n_rot, cd);
    const bool correct = sc.ext_factor != 0;   // ggml: the correction rides on ext_factor, not the type
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(sc.freq_base, -2.0 * (double) i / (double) n_rot);
            const double extrap = (double) p * inv;    // the trained angle, ggml's theta_extrap
            const double interp = fs * extrap;         // ggml's theta_interp
            double ang = interp;
            if (correct) {
                const double ramp = (double) rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            cos_tab[(size_t) p * half + i] = (float) (std::cos(ang) * ms);
            sin_tab[(size_t) p * half + i] = (float) (std::sin(ang) * ms);
        }
    }
}

// One work-item per row, as in the CUDA kernel: the tail copy and the rotation of a row never race.
void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab,
                     const float* sin_tab, const int* pos, void* stream) {
    if (rows <= 0 || n_rot <= 0) return;
    if (n_rot % 2 != 0 || n_rot > head_dim)
        throw core::DeviceError("rope_neox_apply: n_rot " + std::to_string(n_rot) + " must be even and <= head_dim " +
                                std::to_string(head_dim));
    const int32_t* mtab = mrope_table();
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) rows), [=](sycl::id<1> id) {
        const int64_t r = (int64_t) id[0];
        const int half = n_rot / 2;
        const float* xr = x + r * head_dim;
        float* orow = out + r * head_dim;
        for (int d = n_rot; d < head_dim; ++d) orow[d] = xr[d];      // the PARTIAL rotation's untouched tail
        for (int i = 0; i < half; ++i) {
            const size_t toff = (size_t) mrope_pos(mtab, pos[r], i) * half;   // the image path's per-pair position
            rope_neox_pair(xr[i], xr[half + i], cos_tab[toff + i], sin_tab[toff + i], orow[i], orow[half + i]);
        }
    });
    if (!stream) core::Runtime::get().wait(e, "rope_neox_apply");
}

void native_rope_set_enabled(bool value) { native_enabled.store(value, std::memory_order_relaxed); }
bool native_rope_enabled() { return native_enabled.load(std::memory_order_relaxed); }

void native_rope_apply(const float* x, float* out, int rows, int head_dim,
                       int n_rot, float freq_base, const int* positions, void* stream) {
    if (!x || !out || !positions || !stream || rows < 1 || rows > 65535 ||
        (head_dim != 128 && head_dim != 256) || n_rot != 64 ||
        !std::isfinite(freq_base) || freq_base <= 1.0f ||
        reinterpret_cast<uintptr_t>(x) % 4 || reinterpret_cast<uintptr_t>(out) % 4 ||
        reinterpret_cast<uintptr_t>(positions) % 4) {
        throw std::invalid_argument("native RoPE requires aligned F32 rows, width 128/256, rotation 64, valid base and explicit stream");
    }
    const size_t bytes = size_t(rows) * head_dim * sizeof(float);
    if ((x != out && overlaps(x, bytes, out, bytes)) ||
        overlaps(positions, size_t(rows) * sizeof(int), out, bytes) ||
        overlaps(positions, size_t(rows) * sizeof(int), x, bytes)) {
        throw std::invalid_argument("native RoPE buffers partially overlap");
    }
    auto& q = queue_for(stream);
    // Match pinned host-side float powf before device powf/trigonometry.
    const float theta_scale = std::pow(freq_base, -2.0f / n_rot);
    const int32_t* mtab = mrope_table();
    const int width = head_dim;
    const bool scaled = rope_scaling().type != RopeScalingType::None;
    const RopeKernelArgs ka = rope_scaling().kernel_args(n_rot);
    const RopeTab tab = rope_table_for(rope_scaling());
    q.parallel_for(sycl::range<2>((size_t) rows, (size_t) width / 2), [=](sycl::id<2> id) {
        const int row = (int) id[0], pair = (int) id[1];
        const size_t start = size_t(row) * width;
        if (pair >= n_rot / 2) {
            if (x != out) {
                out[start + 2 * pair] = x[start + 2 * pair];
                out[start + 2 * pair + 1] = x[start + 2 * pair + 1];
            }
            return;
        }
        const int p = mrope_pos(mtab, positions[row], pair);
        float c, s;
        if (!rope_tab_cs(tab, p, pair, c, s))
            rope_cos_sin((float) p * sycl::pow(theta_scale, float(pair)), scaled, ka, pair, c, s);
        const float a = x[start + pair], b = x[start + pair + n_rot / 2];
        out[start + pair] = a * c - b * s;
        out[start + pair + n_rot / 2] = a * s + b * c;
    });
}

}  // namespace strata::kernels
