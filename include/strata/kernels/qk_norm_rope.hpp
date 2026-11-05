// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/qk_norm_rope.hpp - the one kernel that RMS-norms the heads of q and k and rotates them
// (RoPE), for the image engine's DiTs and its text and vision readers alike (device code: SYCL files only).
//
// A job is one head of one row: q's heads, then k's, then (with `values`) v's, which are only copied.  A sub-group of
// 32 a job, kJobs jobs a work-group, the lanes over the head's rotated pairs: (j, j + n_rot / 2) (kNeox) or (2j, 2j + 1)
// (interleaved), the columns past n_rot normed only.  The sum of squares is each lane's pairs' a a + b b, in turn,
// then the sub-group's sum.  Every engine GPU has 32-wide sub-groups (core/device.cpp refuses one without).
//
// What differs between the models is in three small types the caller gives:
//   In:  float at(int which, int64_t t, int64_t head, int64_t d)       which 0 q, 1 k, 2 v; the input's layout
//   Ang: void cs(int64_t t, int64_t j, float& c, float& s)              the angle of pair j of row t (NoRope: none)
//   Out: void put(int which, int64_t t, int64_t head, int64_t d, float)   the output's layout, type and repeats
#pragma once

#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels::qk {

constexpr int64_t kLanes = 32, kJobs = 8;

/// no rotation: the heads normed only
struct NoRope {
    static constexpr bool on = false;
    void cs(int64_t, int64_t, float& c, float& s) const { c = 1.f; s = 0.f; }
};

/// cos and sin tables [T][half]
struct Tables {
    static constexpr bool on = true;
    const float *cos, *sin;
    int64_t half;
    void cs(int64_t t, int64_t j, float& c, float& s) const { c = cos[t * half + j]; s = sin[t * half + j]; }
};

/// an angle table [T][half] (HunyuanImage 3.0's)
struct Angles {
    static constexpr bool on = true;
    const float* ang;
    int64_t half;
    void cs(int64_t t, int64_t j, float& c, float& s) const {
        const float a = ang[t * half + j];
        c = sycl::cos(a);
        s = sycl::sin(a);
    }
};

/// pos theta^(-2 f / dim): pos the row's own index, or with `pos3` (3 rows of T: time, height, width) Qwen3-VL's
/// M-RoPE, pair j taking the height's position where j % 3 == 1 and j < 3 sec_h, the width's where j % 3 == 2 and
/// j < 3 sec_w, the time's otherwise; with `yx` ([T][2]) two halves of 2D positions, the first `quarter` pairs the
/// row's y and the rest its x, f counted from each half's start (Qwen3-VL's vision reader).  f = j otherwise.
struct Theta {
    static constexpr bool on = true;
    float theta, dim;
    const int32_t* pos3 = nullptr;
    int64_t T = 0;
    int sec_h = 0, sec_w = 0;
    const int32_t* yx = nullptr;
    int64_t quarter = 0;
    void cs(int64_t t, int64_t j, float& c, float& s) const {
        float pos = (float) t;
        int64_t f = j;
        if (pos3 != nullptr)
            pos = (float) (j % 3 == 1 && j < 3 * (int64_t) sec_h ? pos3[T + t]
                           : j % 3 == 2 && j < 3 * (int64_t) sec_w ? pos3[2 * T + t] : pos3[t]);
        else if (yx != nullptr) {
            pos = (float) (j < quarter ? yx[2 * t] : yx[2 * t + 1]);
            f = j < quarter ? j : j - quarter;
        }
        const float a = pos * sycl::pow(theta, -2.0f * (float) f / dim);
        c = sycl::cos(a);
        s = sycl::sin(a);
    }
};

/// The LLMs' angles: Qwen3.8-Flash-Next's IMRoPE sections (`mtab`) and the session's table [max_pos][half] (null: none,
/// or past it) else the angle computed under the rope scaling (rope_scaling.hpp), at positions[t] or pos0 + t
struct Rope {
    static constexpr bool on = true;
    const int32_t* mtab;
    const float *cos, *sin;
    int max_pos;
    int64_t half;
    float theta_scale;
    bool scaled;
    RopeKernelArgs ka;
    const int* positions;
    int64_t pos0;
    void cs(int64_t t, int64_t j, float& c, float& s) const {
        const int p = mrope_pos(mtab, positions != nullptr ? positions[t] : (int) (pos0 + t), (int) j);
        if (cos != nullptr && p >= 0 && p < max_pos) {
            c = cos[(int64_t) p * half + j];
            s = sin[(int64_t) p * half + j];
            return;
        }
        rope_cos_sin((float) p * sycl::pow(theta_scale, (float) j), scaled, ka, (int) j, c, s);
    }
};

/// FP32 rows [T][ld] in and out, a row's heads head_dim apart (in place where in == out)
struct Rows32 {
    const float* in;
    float* out;
    int64_t ld_in, ld_out, head_dim;
    float at(int, int64_t t, int64_t head, int64_t d) const { return in[t * ld_in + head * head_dim + d]; }
    void put(int, int64_t t, int64_t head, int64_t d, float v) const { out[t * ld_out + head * head_dim + d] = v; }
};

/// FP16 rows out: q [T][q_heads][head_dim], k and v [T][kv_out][head_dim], each k and v head written `rep` times (to
/// heads head rep .. + rep - 1: a GQA's k and v repeated for each q head they serve), clamped to FP16's range with
/// `clamp`
struct Rows16 {
    uint16_t *q16, *k16, *v16;
    int64_t q_heads, kv_out, head_dim, rep = 1;
    bool clamp = false;
    void put(int which, int64_t t, int64_t head, int64_t d, float v) const {
        const uint16_t h = f16_gpu::f16_from_f32(clamp ? sycl::clamp(v, -65504.f, 65504.f) : v);
        if (which == 0) { q16[(t * q_heads + head) * head_dim + d] = h; return; }
        uint16_t* dst = which == 1 ? k16 : v16;
        for (int64_t r = 0; r < rep; ++r) dst[(t * kv_out + head * rep + r) * head_dim + d] = h;
    }
};

/// What a launch does: q_heads and kv_heads heads of head_dim a row (n_rot of them rotated; 0: all), T rows.  wq and
/// wk: the norms' weights (null: not normed); rope_first: rotated, then normed with the weights after (HunyuanImage
/// 3.0's order; the rotation keeps the sum of squares), else normed and weighted, then rotated; values: v's heads
/// copied too; tail: the columns past n_rot written (false: in place and not normed, left as they are).
struct Shape {
    int64_t T, q_heads, kv_heads, head_dim, n_rot = 0;
    const float *wq = nullptr, *wk = nullptr;
    float eps = 1e-6f;
    bool neox = true, rope_first = false, values = true, tail = true;
};

template <class In, class Ang, class Out>
sycl::event qk_norm_rope(sycl::queue& q, const Shape& sh, const In in, const Ang ang, const Out out) {
    const int64_t hd = sh.head_dim, n_rot = sh.n_rot > 0 ? sh.n_rot : hd, half = n_rot / 2;
    const int64_t per_row = sh.q_heads + sh.kv_heads * (sh.values ? 2 : 1), n = sh.T * per_row;
    const int64_t qh = sh.q_heads, kvh = sh.kv_heads;
    const float *wq = sh.wq, *wk = sh.wk, eps = sh.eps;
    const bool neox = sh.neox, rope_first = sh.rope_first, tail = sh.tail || sh.wq != nullptr || sh.wk != nullptr;
    return q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) ((n + kJobs - 1) / kJobs * kJobs * kLanes)),
                          sycl::range<1>((size_t) (kJobs * kLanes))),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const auto sg = it.get_sub_group();
            const int64_t job = (int64_t) it.get_group_linear_id() * kJobs + (int64_t) sg.get_group_linear_id();
            if (job >= n) return;   // the whole sub-group
            const int64_t t = job / per_row, k = job - t * per_row;
            const int which = k < qh ? 0 : k < qh + kvh ? 1 : 2;
            const int64_t head = which == 0 ? k : which == 1 ? k - qh : k - qh - kvh;
            const int64_t lane = (int64_t) sg.get_local_linear_id();
            if (which == 2) {
                for (int64_t d = lane; d < hd; d += kLanes) out.put(2, t, head, d, in.at(2, t, head, d));
                return;
            }
            const float* w = which == 0 ? wq : wk;
            auto lo = [&](int64_t p) { return neox ? p : 2 * p; };
            auto hi = [&](int64_t p) { return neox ? p + half : 2 * p + 1; };
            float r = 1.f;
            if (w != nullptr) {
                float ss = 0.f;
                for (int64_t p = lane; p < half; p += kLanes) {
                    const float a = in.at(which, t, head, lo(p)), b = in.at(which, t, head, hi(p));
                    ss += a * a + b * b;
                }
                for (int64_t d = n_rot + lane; d < hd; d += kLanes) {
                    const float a = in.at(which, t, head, d);
                    ss += a * a;
                }
                r = 1.f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / (float) hd + eps);
            }
            for (int64_t p = lane; p < half; p += kLanes) {
                const int64_t i0 = lo(p), i1 = hi(p);
                float a = in.at(which, t, head, i0), b = in.at(which, t, head, i1), c, s;
                ang.cs(t, p, c, s);
                // rope.hpp's rope_neox_pair: a' = a c - b s, b' = a s + b c
                if (rope_first) {
                    const float a2 = a * c - b * s, b2 = a * s + b * c;
                    a = w != nullptr ? a2 * r * w[i0] : a2;
                    b = w != nullptr ? b2 * r * w[i1] : b2;
                } else {
                    if (w != nullptr) {
                        a = a * r * w[i0];
                        b = b * r * w[i1];
                    }
                    if (Ang::on) {
                        const float a2 = a * c - b * s;
                        b = a * s + b * c;
                        a = a2;
                    }
                }
                out.put(which, t, head, i0, a);
                out.put(which, t, head, i1, b);
            }
            if (!tail) return;
            for (int64_t d = n_rot + lane; d < hd; d += kLanes) {
                const float a = in.at(which, t, head, d);
                out.put(which, t, head, d, w != nullptr ? a * r * w[d] : a);
            }
        });
}

}  // namespace strata::kernels::qk
