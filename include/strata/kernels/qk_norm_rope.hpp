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

// Shared axial pair selection and theta angle. The power function keeps each caller's precision and math path.
inline int64_t axial_frequency(int64_t j, int64_t quarter) { return j < quarter ? j : j - quarter; }

template <class T, class Power>
inline T theta_angle(T pos, int64_t f, T dim, int factor, Power power) {
    return pos * power(-(T) (factor * f) / dim);
}

/// pos theta^(-2 f / dim): pos the row's own index, or with `yx` ([T][2]) two halves of 2D positions, the first
/// `quarter` pairs the row's y and the rest its x, f counted from each half's start (Qwen3-VL's vision reader).
/// f = j otherwise.
struct Theta {
    static constexpr bool on = true;
    float theta, dim;
    const int32_t* yx = nullptr;
    int64_t quarter = 0;
    auto row(int64_t t) const {
        // the row's positions: its own index, or its y and x
        const float pt = yx != nullptr ? (float) yx[2 * t] : (float) t;
        const float ph = yx != nullptr ? (float) yx[2 * t + 1] : (float) t;
        const bool two = yx != nullptr;
        const int64_t qu = quarter;
        const float th = theta, dm = dim;
        return [=](int64_t j, float& c, float& s) {
            float pos = pt;
            int64_t f = j;
            if (two) {
                pos = j < qu ? pt : ph;
                f = axial_frequency(j, qu);
            }
            const float a = theta_angle(pos, f, dm, 2, [=](float e) { return sycl::pow(th, e); });
            c = sycl::cos(a);
            s = sycl::sin(a);
        };
    }
};

/// The LLMs' angles: Qwen3.8-Flash-Next's IMRoPE sections (`mtab`) and the session's table [max_pos][half] (null: none,
/// or past it) else the angle computed under the rope scaling (rope_scaling.hpp), at positions[t] or pos0 + t
struct Rope {
    static constexpr bool on = true;
    MRope mtab;
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
        void w2(int64_t d, float a, float b) const {   // d and d + 1 (d even)
            const uint32_t h = (uint32_t) f16_gpu::f16_from_f32(a) | (uint32_t) f16_gpu::f16_from_f32(b) << 16;
            for (int64_t r = 0; r < rep; ++r) *reinterpret_cast<uint32_t*>(p + r * head_dim + d) = h;
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

/// Columns d and d + 1 (d even) in one access where the head's reader or writer has it (r2, w2), else one at a time
template <class X> inline void read2(const X& x, int d, float& a, float& b) {
    if constexpr (requires { x.r2(int64_t{}, a, b); }) {
        x.r2(d, a, b);
    } else {
        a = x(d);
        b = x(d + 1);
    }
}
template <class Y> inline void write2(const Y& y, int d, float a, float b) {
    if constexpr (requires { y.w2(int64_t{}, a, b); }) {
        y.w2(d, a, b);
    } else {
        y(d, a);
        y(d + 1, b);
    }
}

/// The kernel for a head of up to 128 rotated columns and 64 more not, each an even count (kernel's HELD): a lane holds
/// two adjacent pairs and two adjacent columns past n_rot, read once into registers and two at a time, q's, k's and v's
/// reads issued before the first is used.  Read where used, twice and one at a time, MOSS-Image's QK on the RTX 4070
/// (8221 rows of 24 heads of 128, 2026-11-06) waited on them at 40% of the memory's bandwidth, 0.70 ms
struct Held {
    float a[2], b[2], e[2];   // pairs 2 lane and + 1 (j and j + half, or 2j and 2j + 1); columns n_rot + 2 lane, + 1
};

template <int WHICH, bool NEOX, class In>
inline Held qk_load(int lane, const In& in, int64_t t, int64_t head, int hd, int n_rot, bool rest) {
    const int half = n_rot / 2, p = 2 * lane, d = n_rot + 2 * lane;
    const auto x = in.head(WHICH, t, head);
    Held v{};
    if (p < half) {
        if (NEOX) {
            read2(x, p, v.a[0], v.a[1]);
            read2(x, p + half, v.b[0], v.b[1]);
        } else {
            read2(x, 2 * p, v.a[0], v.b[0]);
            read2(x, 2 * p + 2, v.a[1], v.b[1]);
        }
    }
    if (rest && d < hd) read2(x, d, v.e[0], v.e[1]);
    return v;
}

template <int WHICH, bool NEOX, bool ROPE_FIRST, bool NORM, class Ang, class Out>
inline void qk_store(const sycl::sub_group& sg, const Held& v, const Ang& ang, const Out& out, const float* w,
                     float eps, int64_t t, int64_t head, int hd, int n_rot, bool tail) {
    const int lane = (int) sg.get_local_linear_id(), half = n_rot / 2, p = 2 * lane, d = n_rot + 2 * lane;
    const auto y = out.out_head(WHICH, t, head);
    float r = 1.f;
    if (NORM) {
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 2; ++i) ss += v.a[i] * v.a[i] + v.b[i] * v.b[i] + v.e[i] * v.e[i];
        r = 1.f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / (float) hd + eps);
    }
    if (p < half) {
        const auto rot = ang.row(t);
        float a[2], b[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int i0 = NEOX ? p + i : 2 * (p + i), i1 = NEOX ? p + i + half : 2 * (p + i) + 1;
            float c, s;
            rot(p + i, c, s);
            a[i] = v.a[i];
            b[i] = v.b[i];
            if (ROPE_FIRST) {
                const float a2 = a[i] * c - b[i] * s, b2 = a[i] * s + b[i] * c;
                a[i] = NORM ? a2 * r * w[i0] : a2;
                b[i] = NORM ? b2 * r * w[i1] : b2;
            } else {
                if (NORM) {
                    a[i] = a[i] * r * w[i0];
                    b[i] = b[i] * r * w[i1];
                }
                if (Ang::on) {
                    const float a2 = a[i] * c - b[i] * s;
                    b[i] = a[i] * s + b[i] * c;
                    a[i] = a2;
                }
            }
        }
        if (NEOX) {
            write2(y, p, a[0], a[1]);
            write2(y, p + half, b[0], b[1]);
        } else {
            write2(y, 2 * p, a[0], b[0]);
            write2(y, 2 * p + 2, a[1], b[1]);
        }
    }
    if (tail && d < hd)
        write2(y, d, NORM ? v.e[0] * r * w[d] : v.e[0], NORM ? v.e[1] * r * w[d + 1] : v.e[1]);
}

template <bool NEOX, bool ROPE_FIRST, bool NORM, bool HELD, class In, class Ang, class Out>
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
            if constexpr (HELD) {
                const int lane = (int) sg.get_local_linear_id(), hdi = (int) hd, nr = (int) n_rot;
                Held vq{}, vk{}, vv{};
                if (h < qh) vq = qk_load<0, NEOX>(lane, in, t, h, hdi, nr, tail);
                if (h < kvh) {
                    vk = qk_load<1, NEOX>(lane, in, t, h, hdi, nr, tail);
                    if (values) vv = qk_load<2, true>(lane, in, t, h, hdi, nr, true);
                }
                if (h < qh) qk_store<0, NEOX, ROPE_FIRST, NORM>(sg, vq, ang, out, wq, eps, t, h, hdi, nr, tail);
                if (h < kvh) {
                    qk_store<1, NEOX, ROPE_FIRST, NORM>(sg, vk, ang, out, wk, eps, t, h, hdi, nr, tail);
                    if (values)
                        qk_store<2, true, false, false>(sg, vv, NoRope{}, out, nullptr, eps, t, h, hdi, nr, true);
                }
                return;
            }
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

/// The kernel's forms: the pairs' layout, the order of the rotation and the norm, whether there is a norm and whether
/// the head fits Held, as constants (the per-head kernels this replaced had them so; as run-time flags in the loops
/// they cost MOSS-Image's QK on the B70 a quarter more)
template <class In, class Ang, class Out>
sycl::event qk_norm_rope(sycl::queue& q, const Shape& sh, const In in, const Ang ang, const Out out) {
    const bool norm = sh.wq != nullptr || sh.wk != nullptr;
    const int64_t n_rot = sh.n_rot > 0 ? sh.n_rot : sh.head_dim, rest = sh.head_dim - n_rot;
    const bool held = n_rot % 4 == 0 && n_rot <= 4 * kLanes && rest % 2 == 0 && rest <= 2 * kLanes;
    auto go = [&]<bool NEOX, bool ROPE_FIRST, bool NORM>() {
        return held ? qk_norm_rope_t<NEOX, ROPE_FIRST, NORM, true>(q, sh, in, ang, out)
                    : qk_norm_rope_t<NEOX, ROPE_FIRST, NORM, false>(q, sh, in, ang, out);
    };
    if (sh.neox) {
        if (sh.rope_first) return norm ? go.template operator()<true, true, true>()
                                       : go.template operator()<true, true, false>();
        return norm ? go.template operator()<true, false, true>() : go.template operator()<true, false, false>();
    }
    if (sh.rope_first) return norm ? go.template operator()<false, true, true>()
                                   : go.template operator()<false, true, false>();
    return norm ? go.template operator()<false, false, true>() : go.template operator()<false, false, false>();
}

// ---- the backward (a training step's), FP32 rows [T][heads][head_dim] in and out.  Normed first (rope_first false):
// z = x r w, y = R z, so dz = R^T dy and dx = r (dz w - x^ mean(dz w x^)) with x^ = x r, and dw += dz x^.  Rotated
// first: z = R x, y = z r w (r the same: the rotation keeps the sum of squares), so dz = r (dy w - z^ mean(dy w z^)),
// dx = R^T dz and dw += dy z^.  R^T turns a pair back: a = a' c + b' s, b = b' c - a' s.

/// a pair's values turned forth (FWD: a' = a c - b s, b' = a s + b c) or back
template <bool FWD> inline void qk_turn(float& a, float& b, float c, float s) {
    const float a2 = FWD ? a * c - b * s : a * c + b * s;
    b = FWD ? a * s + b * c : b * c - a * s;
    a = a2;
}

/// column d's pair among the rotated columns (-1 past them), its partner and whether d is the pair's first
template <bool NEOX> inline int qk_pair(int d, int half, int& partner, bool& first) {
    if (d >= 2 * half) return -1;
    const int p = NEOX ? (d < half ? d : d - half) : d / 2;
    first = NEOX ? d < half : d % 2 == 0;
    partner = NEOX ? (first ? d + half : d - half) : (first ? d + 1 : d - 1);
    return p;
}

/// column d of one head's normed value (x^, or rotated first z^) and of the gradient its weight multiplies (dz, or dy)
template <bool NEOX, bool ROPE_FIRST, class Rot>
inline void qk_bwd_col(const float* x, const float* dy, const Rot& rot, int d, int half, float r, float& xh, float& g) {
    int partner = 0;
    bool first = true;
    const int p = qk_pair<NEOX>(d, half, partner, first);
    if (p < 0) {
        xh = x[d] * r;
        g = dy[d];
        return;
    }
    float c, s;
    rot(p, c, s);
    float a = first ? x[d] : x[partner], b = first ? x[partner] : x[d];
    float ga = first ? dy[d] : dy[partner], gb = first ? dy[partner] : dy[d];
    if (ROPE_FIRST) qk_turn<true>(a, b, c, s);
    else qk_turn<false>(ga, gb, c, s);
    xh = (first ? a : b) * r;
    g = first ? ga : gb;
}

template <bool NEOX, bool ROPE_FIRST, bool NORM, class Ang>
sycl::event qk_norm_rope_bwd_t(sycl::queue& q, const Shape& sh, const float* xq, const float* xk, const float* dyq,
                               const float* dyk, const Ang ang, float* dxq, float* dxk, float* dwq, float* dwk,
                               float* rs) {
    const int64_t hd = sh.head_dim, n_rot = sh.n_rot > 0 ? sh.n_rot : hd, half = n_rot / 2, T = sh.T;
    const int64_t qh = sh.q_heads, kvh = sh.kv_heads, nh = std::max(qh, kvh), n = T * nh;
    const float *wq = sh.wq, *wk = sh.wk, eps = sh.eps;
    sycl::event e = q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) ((n + kJobs - 1) / kJobs * kJobs * kLanes)),
                          sycl::range<1>((size_t) (kJobs * kLanes))),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const auto sg = it.get_sub_group();
            const int64_t job = (int64_t) it.get_group_linear_id() * kJobs + (int64_t) sg.get_group_linear_id();
            if (job >= n) return;
            const int64_t t = job / nh, h = job - t * nh;
            const int lane = (int) sg.get_local_linear_id(), hdi = (int) hd, hf = (int) half;
            const auto rot = ang.row(t);
            for (int which = 0; which < 2; ++which) {
                const int64_t heads = which == 0 ? qh : kvh;
                const float* DY = which == 0 ? dyq : dyk;
                float* DX = which == 0 ? dxq : dxk;
                const float* w = which == 0 ? wq : wk;
                if (h >= heads || DY == nullptr || DX == nullptr) continue;
                const float* x = (which == 0 ? xq : xk) + (t * heads + h) * hd;
                const float* dy = DY + (t * heads + h) * hd;
                float* dx = DX + (t * heads + h) * hd;
                const bool normed = NORM && w != nullptr;
                float r = 1.f, sum = 0.f;
                if (normed) {
                    float ss = 0.f;
                    for (int d = lane; d < hdi; d += (int) kLanes) ss += x[d] * x[d];
                    r = 1.f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / (float) hd + eps);
                    // the norm's mean over the head of (the weighted gradient) x^
                    for (int d = lane; d < hdi; d += (int) kLanes) {
                        float xh, g;
                        qk_bwd_col<NEOX, ROPE_FIRST>(x, dy, rot, d, hf, r, xh, g);
                        sum += g * w[d] * xh;
                    }
                    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>()) / (float) hd;
                }
                for (int p = lane; p < hf; p += (int) kLanes) {
                    const int i0 = NEOX ? p : 2 * p, i1 = NEOX ? p + hf : 2 * p + 1;
                    float c, s;
                    rot(p, c, s);
                    float da, db;
                    if (ROPE_FIRST) {   // dz on the rotated columns, then turned back
                        float a = x[i0], b = x[i1];
                        qk_turn<true>(a, b, c, s);
                        da = normed ? r * (dy[i0] * w[i0] - a * r * sum) : dy[i0];
                        db = normed ? r * (dy[i1] * w[i1] - b * r * sum) : dy[i1];
                        qk_turn<false>(da, db, c, s);
                    } else {            // dy turned back, then the norm's backward
                        float ga = dy[i0], gb = dy[i1];
                        qk_turn<false>(ga, gb, c, s);
                        da = normed ? r * (ga * w[i0] - x[i0] * r * sum) : ga;
                        db = normed ? r * (gb * w[i1] - x[i1] * r * sum) : gb;
                    }
                    dx[i0] = da;
                    dx[i1] = db;
                }
                for (int d = 2 * hf + lane; d < hdi; d += (int) kLanes)
                    dx[d] = normed ? r * (dy[d] * w[d] - x[d] * r * sum) : dy[d];
                if (rs != nullptr && lane == 0) rs[2 * job + which] = r;
            }
        });
    if (!NORM || rs == nullptr || (dwq == nullptr && dwk == nullptr)) return e;
    // the weights' gradients: a work-group a column of q's or k's, summed over every (row, head)
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(e);
        constexpr int WG = 256;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (2 * hd * WG)), sycl::range<1>(WG)),
                         [=](sycl::nd_item<1> it) {
                             const int which = (int) ((int64_t) it.get_group_linear_id() / hd);
                             const int d = (int) ((int64_t) it.get_group_linear_id() % hd);
                             float* dw = which == 0 ? dwq : dwk;
                             const int64_t heads = which == 0 ? qh : kvh;
                             const float* X = which == 0 ? xq : xk;
                             const float* DY = which == 0 ? dyq : dyk;
                             if (dw == nullptr || DY == nullptr) return;
                             float v = 0.f;
                             for (int64_t i = (int64_t) it.get_local_linear_id(); i < T * heads; i += WG) {
                                 const int64_t t = i / heads, job = t * nh + (i - t * heads);
                                 float xh, g;
                                 qk_bwd_col<NEOX, ROPE_FIRST>(X + i * hd, DY + i * hd, ang.row(t), d, (int) half,
                                                              rs[2 * job + which], xh, g);
                                 v += g * xh;
                             }
                             v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
                             if (it.get_local_linear_id() == 0) dw[d] += v;
                         });
    });
}

/// The backward of qk_norm_rope for FP32 rows, `sh` as the forward's (q_heads, kv_heads, head_dim, n_rot, neox,
/// rope_first, wq, wk, eps; v's rows are copied, so its gradient is its dy as it is): xq and xk the forward's input
/// rows [T][q_heads or kv_heads][head_dim], dyq and dyk the gradients of its output, dxq and dxk written (a null dy or
/// dx: that tensor not taken), dwq and dwk the norms' weights' gradients added into (null: not taken).  rs: 2 T
/// max(q_heads, kv_heads) floats of scratch, kept until the returned event ends (null: no weights' gradients).
template <class Ang>
sycl::event qk_norm_rope_bwd(sycl::queue& q, const Shape& sh, const float* xq, const float* xk, const float* dyq,
                             const float* dyk, const Ang ang, float* dxq, float* dxk, float* dwq, float* dwk,
                             float* rs) {
    const bool norm = sh.wq != nullptr || sh.wk != nullptr;
    auto go = [&]<bool NEOX, bool ROPE_FIRST, bool NORM>() {
        return qk_norm_rope_bwd_t<NEOX, ROPE_FIRST, NORM>(q, sh, xq, xk, dyq, dyk, ang, dxq, dxk, dwq, dwk, rs);
    };
    if (sh.neox) {
        if (sh.rope_first) return norm ? go.template operator()<true, true, true>()
                                       : go.template operator()<true, true, false>();
        return norm ? go.template operator()<true, false, true>() : go.template operator()<true, false, false>();
    }
    if (sh.rope_first) return norm ? go.template operator()<false, true, true>()
                                   : go.template operator()<false, true, false>();
    return norm ? go.template operator()<false, false, true>() : go.template operator()<false, false, false>();
}

}  // namespace strata::kernels::qk
