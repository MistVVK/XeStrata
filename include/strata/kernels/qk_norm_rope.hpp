// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/qk_norm_rope.hpp - the one kernel that RMS-norms the heads of q and k and rotates them
// (RoPE), for the image engine's DiTs and its text and vision readers alike (device code: SYCL files only).
//
// A job is one head index of one row: its q head, then its k head and (with `values`) its v head, which is only copied
// (a GQA's k and v heads past the q heads' count have none).  A sub-group of 32 a job, kJobs jobs a work-group, the
// lanes over the head's rotated pairs: (j, j + n_rot / 2) (kNeox) or (2j, 2j + 1)
// (interleaved), the columns past n_rot normed only.  The sum of squares is each lane's pairs' a a + b b, in turn,
// then the sub-group's sum.  Every engine GPU has 32-wide sub-groups (core/device.cpp refuses one without).
//
// What differs between the models is in three small types the caller gives, each bound to a head (or a row) once and
// then read by the column, the columns counted in int - an index a column took 64-bit multiplies a value, which Intel's
// GPUs have no instruction for (MOSS-Image's QK on the B70: 369 -> 468 us; the 64-bit column counts, 381 -> 390 us):
//   In:  head(int which, int64_t t, int64_t h) -> r, float r(int64_t d)          which 0 q, 1 k, 2 v; the layout
//   Ang: row(int64_t t) -> a, void a(int64_t j, float& c, float& s)               pair j's angle (NoRope: none)
//   Out: out_head(int which, int64_t t, int64_t h) -> w, void w(int64_t d, float) the layout, type and repeats
#pragma once

#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>

namespace strata::kernels::qk {

constexpr int64_t kLanes = 32, kJobs = 8;

/// no rotation: the heads normed only
struct NoRope {
    static constexpr bool on = false;
    auto row(int64_t) const { return [](int64_t, float& c, float& s) { c = 1.f; s = 0.f; }; }
};

/// cos and sin tables [T][half]
struct Tables {
    static constexpr bool on = true;
    const float *cos, *sin;
    int64_t half;
    auto row(int64_t t) const {
        const float *c0 = cos + t * half, *s0 = sin + t * half;
        return [c0, s0](int64_t j, float& c, float& s) { c = c0[j]; s = s0[j]; };
    }
};

/// an angle table [T][half] (HunyuanImage 3.0's)
struct Angles {
    static constexpr bool on = true;
    const float* ang;
    int64_t half;
    auto row(int64_t t) const {
        const float* a0 = ang + t * half;
        return [a0](int64_t j, float& c, float& s) {
            c = sycl::cos(a0[j]);
            s = sycl::sin(a0[j]);
        };
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
    auto row(int64_t t) const {
        // the row's positions: its own index, its three M-RoPE ones, or its y and x
        const float pt = pos3 != nullptr ? (float) pos3[t] : yx != nullptr ? (float) yx[2 * t] : (float) t;
        const float ph = pos3 != nullptr ? (float) pos3[T + t] : yx != nullptr ? (float) yx[2 * t + 1] : (float) t;
        const float pw = pos3 != nullptr ? (float) pos3[2 * T + t] : pt;
        const bool m3 = pos3 != nullptr, two = yx != nullptr;
        const int64_t qu = quarter, sh = sec_h, sw = sec_w;
        const float th = theta, dm = dim;
        return [=](int64_t j, float& c, float& s) {
            float pos = pt;
            int64_t f = j;
            if (m3)
                pos = j % 3 == 1 && j < 3 * sh ? ph : j % 3 == 2 && j < 3 * sw ? pw : pt;
            else if (two) {
                pos = j < qu ? pt : ph;
                f = j < qu ? j : j - qu;
            }
            const float a = pos * sycl::pow(th, -2.0f * (float) f / dm);
            c = sycl::cos(a);
            s = sycl::sin(a);
        };
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
    auto row(int64_t t) const {
        const int pt = positions != nullptr ? positions[t] : (int) (pos0 + t);
        const Rope r = *this;
        return [r, pt](int64_t j, float& c, float& s) {
            const int p = mrope_pos(r.mtab, pt, (int) j);
            if (r.cos != nullptr && p >= 0 && p < r.max_pos) {
                c = r.cos[(int64_t) p * r.half + j];
                s = r.sin[(int64_t) p * r.half + j];
                return;
            }
            rope_cos_sin((float) p * sycl::pow(r.theta_scale, (float) j), r.scaled, r.ka, (int) j, c, s);
        };
    }
};

/// FP32 rows [T][ld] in and out, a row's heads head_dim apart (in place where in == out)
struct Rows32 {
    const float* in;
    float* out;
    int64_t ld_in, ld_out, head_dim;
    auto head(int, int64_t t, int64_t h) const {
        const float* p = in + t * ld_in + h * head_dim;
        return [p](int64_t d) { return p[d]; };
    }
    struct Put {
        float* p;
        void operator()(int64_t d, float v) const { p[d] = v; }
    };
    Put out_head(int, int64_t t, int64_t h) const { return {out + t * ld_out + h * head_dim}; }
};

/// FP16 rows out: q [T][q_heads][head_dim], k and v [T][kv_out][head_dim], each k and v head written `rep` times (to
/// heads head rep .. + rep - 1: a GQA's k and v repeated for each q head they serve)
struct Rows16 {
    uint16_t *q16, *k16, *v16;
    int64_t q_heads, kv_out, head_dim, rep = 1;
    struct Put {
        uint16_t* p;
        int64_t rep, head_dim;
        void operator()(int64_t d, float v) const {
            const uint16_t h = f16_gpu::f16_from_f32(v);
            if (rep == 1) {   // the usual: one head
                p[d] = h;
                return;
            }
            for (int64_t r = 0; r < rep; ++r) p[r * head_dim + d] = h;
        }
    };
    Put out_head(int which, int64_t t, int64_t h) const {
        if (which == 0) return {q16 + (t * q_heads + h) * head_dim, 1, head_dim};
        return {(which == 1 ? k16 : v16) + (t * kv_out + h * rep) * head_dim, rep, head_dim};
    }
};

/// What a launch does: q_heads and kv_heads heads of head_dim a row (n_rot of them rotated; 0: all), T rows.  wq and
/// wk: the norms' weights (both null: not normed; one set, the other tensor's heads none); rope_first: rotated, then
/// normed with the weights after (HunyuanImage 3.0's order; the rotation keeps the sum of squares), else normed and
/// weighted, then rotated; values: v's heads copied too; tail: the columns past n_rot written (false: in place and not
/// normed, left as they are).
struct Shape {
    int64_t T, q_heads, kv_heads, head_dim, n_rot = 0;
    const float *wq = nullptr, *wk = nullptr;
    float eps = 1e-6f;
    bool neox = true, rope_first = false, values = true, tail = true;
};

/// One head of row t, of q (WHICH 0) or k (1): the job's arithmetic, WHICH a constant (the input and output it reads
/// and writes resolved when the kernel is compiled)
template <int WHICH, bool NEOX, bool ROPE_FIRST, bool NORM, class In, class Ang, class Out>
inline void qk_head(const sycl::sub_group& sg, const In& in, const Ang& ang, const Out& out, const float* w, float eps,
                    int64_t t, int64_t head, int64_t hd64, int64_t n_rot64, bool tail) {
    constexpr bool neox = NEOX, rope_first = ROPE_FIRST;
    const int lane = (int) sg.get_local_linear_id(), hd = (int) hd64, n_rot = (int) n_rot64, half = n_rot / 2;
    const auto x = in.head(WHICH, t, head);
    const auto y = out.out_head(WHICH, t, head);
    const auto rot = ang.row(t);
    float r = 1.f;
    if (NORM) {
        float ss = 0.f;
        for (int p = lane; p < half; p += (int) kLanes) {
            const float a = x(neox ? p : 2 * p);
            const float b = x(neox ? p + half : 2 * p + 1);
            ss += a * a + b * b;
        }
        for (int d = n_rot + lane; d < hd; d += (int) kLanes) {
            const float a = x(d);
            ss += a * a;
        }
        r = 1.f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / (float) hd + eps);
    }
    for (int p = lane; p < half; p += (int) kLanes) {
        const int i0 = neox ? p : 2 * p, i1 = neox ? p + half : 2 * p + 1;
        float a = x(i0), b = x(i1), c, s;
        rot(p, c, s);
        // rope.hpp's rope_neox_pair: a' = a c - b s, b' = a s + b c
        if (rope_first) {
            const float a2 = a * c - b * s, b2 = a * s + b * c;
            a = NORM ? a2 * r * w[i0] : a2;
            b = NORM ? b2 * r * w[i1] : b2;
        } else {
            if (NORM) {
                a = a * r * w[i0];
                b = b * r * w[i1];
            }
            if (Ang::on) {
                const float a2 = a * c - b * s;
                b = a * s + b * c;
                a = a2;
            }
        }
        y(i0, a);
        y(i1, b);
    }
    if (!tail) return;
    for (int d = n_rot + lane; d < hd; d += (int) kLanes) {
        const float a = x(d);
        y(d, NORM ? a * r * w[d] : a);
    }
}

template <bool NEOX, bool ROPE_FIRST, bool NORM, class In, class Ang, class Out>
sycl::event qk_norm_rope_t(sycl::queue& q, const Shape& sh, const In in, const Ang ang, const Out out) {
    const int64_t hd = sh.head_dim, n_rot = sh.n_rot > 0 ? sh.n_rot : hd;
    const int64_t qh = sh.q_heads, kvh = sh.kv_heads, nh = std::max(qh, kvh), n = sh.T * nh;
    const float *wq = sh.wq, *wk = sh.wk, eps = sh.eps;
    const bool values = sh.values, tail = sh.tail || NORM;
    // a job a row's head index h: its q head (h < q_heads), its k head and v head (h < kv_heads), one after the other
    // in the same sub-group - jobs of one tensor each ran the RTX 4070's Qwen-Image 4.0 -> 5.1 s an image
    return q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) ((n + kJobs - 1) / kJobs * kJobs * kLanes)),
                          sycl::range<1>((size_t) (kJobs * kLanes))),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const auto sg = it.get_sub_group();
            const int64_t job = (int64_t) it.get_group_linear_id() * kJobs + (int64_t) sg.get_group_linear_id();
            if (job >= n) return;   // the whole sub-group
            const int64_t t = job / nh, h = job - t * nh;
            if (h < qh) qk_head<0, NEOX, ROPE_FIRST, NORM>(sg, in, ang, out, wq, eps, t, h, hd, n_rot, tail);
            if (h < kvh) {
                qk_head<1, NEOX, ROPE_FIRST, NORM>(sg, in, ang, out, wk, eps, t, h, hd, n_rot, tail);
                if (values) {
                    const auto x = in.head(2, t, h);
                    const auto y = out.out_head(2, t, h);
                    for (int d = (int) sg.get_local_linear_id(); d < (int) hd; d += (int) kLanes) y(d, x(d));
                }
            }
        });
}

/// The kernel's forms: the pairs' layout, the order of the rotation and the norm, and whether there is a norm, as
/// constants (the per-head kernels this replaced had them so; as run-time flags in the loops they cost MOSS-Image's
/// QK on the B70 a quarter more)
template <class In, class Ang, class Out>
sycl::event qk_norm_rope(sycl::queue& q, const Shape& sh, const In in, const Ang ang, const Out out) {
    const bool norm = sh.wq != nullptr || sh.wk != nullptr;
    if (sh.neox) {
        if (sh.rope_first) return norm ? qk_norm_rope_t<true, true, true>(q, sh, in, ang, out)
                                       : qk_norm_rope_t<true, true, false>(q, sh, in, ang, out);
        return norm ? qk_norm_rope_t<true, false, true>(q, sh, in, ang, out)
                    : qk_norm_rope_t<true, false, false>(q, sh, in, ang, out);
    }
    if (sh.rope_first) return norm ? qk_norm_rope_t<false, true, true>(q, sh, in, ang, out)
                                   : qk_norm_rope_t<false, true, false>(q, sh, in, ang, out);
    return norm ? qk_norm_rope_t<false, false, true>(q, sh, in, ang, out)
                : qk_norm_rope_t<false, false, false>(q, sh, in, ang, out);
}

}  // namespace strata::kernels::qk
