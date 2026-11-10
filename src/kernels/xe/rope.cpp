// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/rope.cpp - the Xe ports of Strata's src/kernels/cuda/rope.cu (NEOX partial RoPE over a host-built table)
// and Strata's src/kernels/cuda/native_rope.cu (the pinned ggml text IMRoPE), with the M-RoPE table they share.
//
// native_rope follows ggml/src/ggml-cuda/rope.cu (rope_multi/rope_yarn) at llama.cpp 3cf03257; MIT License,
// Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE.  CUDA compiled it with --use_fast_math;
// here pow, cos and sin are the precise SYCL functions.
#include "strata/kernels/rope.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/qk_norm_rope.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_rope
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <array>
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

// The tables of each engine device, as the CUDA build's per-device array: a layer split's stages each read their own
// GPU's (one pointer for all made the B70 read the RTX 4070's image positions and hang at the first rotary layer)
constexpr int kMaxDevices = 16;
std::array<std::atomic<const int32_t*>, kMaxDevices> mrope_tab{};
// the rotation each device's table is for: n_rot, sec_h, sec_w (set with the table, before any graph)
std::array<std::array<int, 3>, kMaxDevices> mrope_shape{};
int device_slot() {
    const int d = core::current_device();
    if (d < 0 || d >= kMaxDevices) throw std::runtime_error("rope tables: engine device " + std::to_string(d));
    return d;
}

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
std::array<RopeTab, kMaxDevices> g_rope_tab;   // per engine device (set at startup, before any graph)
std::array<RopeScaling, kMaxDevices> g_rope_tab_scaling;
bool same_scaling(const RopeScaling& a, const RopeScaling& b) {
    return a.type == b.type && a.freq_base == b.freq_base && a.factor == b.factor && a.freq_scale_in == b.freq_scale_in &&
           a.orig_ctx == b.orig_ctx && a.ext_factor == b.ext_factor && a.attn_factor == b.attn_factor &&
           a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow;
}
}  // namespace
void rope_table_set(const float* cos_tab, const float* sin_tab, int max_pos, const RopeScaling& scaling) {
    const int d = device_slot();
    g_rope_tab[(size_t) d] = RopeTab{cos_tab, sin_tab, max_pos};
    g_rope_tab_scaling[(size_t) d] = scaling;
}
void rope_table_release(const float* cos_tab) {
    for (RopeTab& t : g_rope_tab)
        if (t.cos == cos_tab) t = RopeTab{};
}
RopeTab rope_table_for(const RopeScaling& scaling) {
    static const bool on = [] {
        const char* v = strata::env_get("ROPE_TABLE");
        return v != nullptr && v[0] == '1';
    }();
    if (!on) return RopeTab{};
    const int d = device_slot();
    return same_scaling(scaling, g_rope_tab_scaling[(size_t) d]) ? g_rope_tab[(size_t) d] : RopeTab{};
}

void mrope_table_set(const int32_t* device_table, int n_rot, int sec_h, int sec_w) {
    const int d = device_slot();
    mrope_shape[(size_t) d] = {n_rot, sec_h, sec_w};
    mrope_tab[(size_t) d].store(device_table, std::memory_order_relaxed);
}
const int32_t* mrope_table() { return mrope_tab[(size_t) device_slot()].load(std::memory_order_relaxed); }
MRope mrope_for(int n_rot) {
    const int d = device_slot();
    const std::array<int, 3>& sh = mrope_shape[(size_t) d];
    const int32_t* tab = mrope_tab[(size_t) d].load(std::memory_order_relaxed);
    return tab != nullptr && n_rot == sh[0] ? MRope{tab, sh[1], sh[2]} : MRope{};
}

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

namespace {
sycl::event rope_launch(sycl::queue& q, const float* x, float* out, int64_t T, int64_t heads, int head_dim,
                        int64_t ld_in, int64_t ld_out, int n_rot, const qk::Rope& ang) {
    qk::Shape sh{T, heads, 0, head_dim, n_rot};
    sh.values = false;
    sh.tail = x != out;
    const qk::Rows32 rows{x, out, ld_in, ld_out, head_dim};
    return qk::qk_norm_rope(q, sh, rows, ang, rows);
}
}  // namespace

void rope_rows(const float* x, float* out, int64_t T, int64_t heads, int head_dim, int64_t ld_in, int64_t ld_out,
               int n_rot, const RopeScaling& scaling, const int* positions, int64_t pos0, void* stream) {
    if (T <= 0) return;
    // Match pinned host-side float powf before device powf/trigonometry.
    const float theta_scale = std::pow((float) scaling.freq_base, -2.0f / (float) n_rot);
    // the float64 angle table is Qwen3.8's 64-wide rotary slice (32 pairs): a rotation over the whole head (llm-jp-4.1)
    // computes its angles; the M-RoPE table applies to the rotation it was set for (mrope_for)
    const RopeTab tab = n_rot == 64 ? rope_table_for(scaling) : RopeTab{};
    const qk::Rope ang{mrope_for(n_rot), tab.cos, tab.sin, tab.max_pos, n_rot / 2, theta_scale,
                       scaling.type != RopeScalingType::None, scaling.kernel_args(n_rot), positions, pos0};
    rope_launch(queue_for(stream), x, out, T, heads, head_dim, ld_in, ld_out, n_rot, ang);
}

void native_rope_apply(const float* x, float* out, int rows, int head_dim,
                       int n_rot, const RopeScaling& scaling, const int* positions, void* stream) {
    if (!x || !out || !positions || !stream || rows < 1 || rows > 65535 ||
        (head_dim != 128 && head_dim != 256) || (n_rot != 64 && n_rot != head_dim) ||
        rope_scaling_invalid(scaling) != nullptr ||
        reinterpret_cast<uintptr_t>(x) % 4 || reinterpret_cast<uintptr_t>(out) % 4 ||
        reinterpret_cast<uintptr_t>(positions) % 4) {
        throw std::invalid_argument("native RoPE requires aligned F32 rows, width 128/256, rotation 64 or the width, valid base and explicit stream");
    }
    const size_t bytes = size_t(rows) * head_dim * sizeof(float);
    if ((x != out && overlaps(x, bytes, out, bytes)) ||
        overlaps(positions, size_t(rows) * sizeof(int), out, bytes) ||
        overlaps(positions, size_t(rows) * sizeof(int), x, bytes)) {
        throw std::invalid_argument("native RoPE buffers partially overlap");
    }
    rope_rows(x, out, rows, 1, head_dim, head_dim, head_dim, n_rot, scaling, positions, 0, stream);
}

bool native_norm_rope_usable(int head_dim, int n_rot) {
    static const bool on = [] {
        const char* v = strata::env_get("NO_NORM_ROPE");
        return v == nullptr || std::strtol(v, nullptr, 10) == 0;
    }();
    return on && (head_dim == 128 || head_dim == 256) && (n_rot == 64 || n_rot == head_dim);
}

// upstream 088e8a82 (#783 PR-f): native_qsa_rms_norm_weighted (its 256-item form: the same partial sums, sub-group
// sums and order) and native_rope_apply in one launch, so the result is the pair's bit for bit (rope_parity).  Every
// item reads its values before the barrier, so out may be x (in_stride == head_dim).
void native_qsa_rms_norm_rope(const float* x, int in_stride, const float* gamma, float* out, int rows, int head_dim,
                              int n_rot, float epsilon, const RopeScaling& scaling, const int* positions, void* stream) {
    if (!x || !gamma || !out || !positions || !stream || rows < 1 || rows > 65535 ||
        (head_dim != 128 && head_dim != 256) || in_stride < head_dim || (n_rot != 64 && n_rot != head_dim) ||
        !std::isfinite(epsilon) ||
        epsilon < 0.0f || rope_scaling_invalid(scaling) != nullptr || (x == out && in_stride != head_dim) ||
        reinterpret_cast<uintptr_t>(x) % 4 || reinterpret_cast<uintptr_t>(gamma) % 4 ||
        reinterpret_cast<uintptr_t>(out) % 4 || reinterpret_cast<uintptr_t>(positions) % 4)
        throw std::invalid_argument("native_qsa_rms_norm_rope: invalid arguments");
    constexpr int BlockSize = 256;
    const float theta_scale = std::pow((float) scaling.freq_base, -2.0f / (float) n_rot);
    const MRope mtab = mrope_for(n_rot);   // as native_rope_apply
    const bool scaled = scaling.type != RopeScalingType::None;
    const RopeKernelArgs ka = scaling.kernel_args(n_rot);
    const RopeTab tab = n_rot == 64 ? rope_table_for(scaling) : RopeTab{};
    const int n_cols = head_dim;
    queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) rows * BlockSize, BlockSize),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(0);
            const size_t row = it.get_group(0);
            const float* in = x + row * (size_t) in_stride;
            float* o = out + row * (size_t) n_cols;
            float partial = 0.0f;
            for (size_t col = tid; col < size_t(n_cols); col += BlockSize) {
                const float v = in[col];
                partial += v * v;
            }
            const int half = n_rot / 2;
            const float x_tid = tid < n_cols ? in[tid] : 0.0f;
            const float x_hi = tid < half ? in[tid + half] : 0.0f;
            for (int offset = 16; offset; offset >>= 1) partial += sycl::permute_group_by_xor(sg, partial, offset);
            const int lane = tid % 32;
            if (lane == 0) sums[tid / 32] = partial;
            sycl::group_barrier(it.get_group());
            partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
            for (int offset = 16; offset; offset >>= 1) partial += sycl::permute_group_by_xor(sg, partial, offset);
            const float mean = partial / (float) n_cols;
            const float scale = sycl::rsqrt(mean + epsilon);
            if (tid >= n_rot && tid < n_cols) {
                o[tid] = scale * x_tid * gamma[tid];
            } else if (tid < half) {
                const float a = scale * x_tid * gamma[tid], b = scale * x_hi * gamma[tid + half];
                const int p = mrope_pos(mtab, positions[row], tid);
                float c, s;
                if (!rope_tab_cs(tab, p, tid, c, s))
                    rope_cos_sin((float) p * sycl::pow(theta_scale, float(tid)), scaled, ka, tid, c, s);
                o[tid] = a * c - b * s;
                o[tid + half] = a * s + b * c;
            }
        });
    });
}

}  // namespace strata::kernels
