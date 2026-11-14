// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/kernels/xe/yskn.cpp - the products of a .yskn matrix (strata/kernels/yskn.hpp), written anew in SYCL from
// YASAKANI's a8_prologue.cuh (the activations' rotation and int8), lnord.cu (the codes through the int8 table, the
// integer sums) and q35_kernels.cu (the output's rotation).  The Hadamard butterflies pair the same elements in the
// same order as the CPU reference (artifact/yskn.cpp), so the rotations give the same floats.
#include "strata/kernels/yskn.hpp"

#include "cuda_intrinsics.hpp"
#include "hadamard_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/trellis.hpp"
#include "strata/kernels/yskn_sum.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_yskn
#include "strata/kernels/device_arch_tag.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

namespace ys = strata::kernels::yskn;

constexpr int kGroup = 512;
constexpr int kStates = 4096;
constexpr float kInvSqrt512 = 0.044194173824159216f;
constexpr int kRowsWg = 64;   // a gemv work-group's rows: the device order's rows are a multiple of 64

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }
void finish_launch(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

void check(const YsknDev& w, const char* what) {
    if (!w.words || !w.lut8 || !w.tau || !w.sign_in || w.rows <= 0 || w.rows % kRowsWg != 0 || w.cols <= 0 ||
        w.cols % kGroup != 0 || w.rows_used <= 0 || w.rows_used > w.rows || w.bits < 4 || w.bits > 8 ||
        w.n_workers <= 0 || (w.cols / kGroup) % w.n_workers != 0 || (w.sign_out && w.rows % kGroup != 0))
        throw core::DeviceError(std::string(what) + ": invalid .yskn matrix");
}

// the orthonormal Hadamard transform of buf[0..511]: the shared butterfly (hadamard_device.hpp, the APR experts'
// hadamard_rows and the CPU's hadamard_rotate), 256 work-items each one pair a stage; the trailing barrier leaves the
// result ready to read (the caller's barrier after the load is covered by the first stage's own)
inline void fwht512_local(float* buf, int lane, const sycl::nd_item<1>& it) {
    xe::hadamard_block(buf, kGroup, lane, 256, it.get_group());
    sycl::group_barrier(it.get_group());
}

// the state of step j: the shared trellis window (kernels/trellis.hpp, the reader's unit_state)
using strata::kernels::trellis::state_of;

template<int K>
sycl::event gemv_launch(sycl::queue& q, const YsknDev& w, const int8_t* xq, const float* alpha, int T, int64_t* part) {
    const int rows = w.rows, cols = w.cols, G = cols / kGroup, nw = w.n_workers, gpw = G / nw;
    const int lw = nw % 4 == 0 ? 4 : nw % 2 == 0 ? 2 : 1;   // workers a work-group: one table load for more lanes
    const uint32_t* words = w.words;
    const uint16_t* lut8 = w.lut8;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint16_t, 1> lut(sycl::range<1>(kStates), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) rows, (size_t) nw), sycl::range<2>(kRowsWg, (size_t) lw)),
                       [=](sycl::nd_item<2> it) {
            const int lid = (int) (it.get_local_id(0) * lw + it.get_local_id(1));
            for (int i = lid; i < kStates; i += kRowsWg * lw) lut[i] = lut8[i];
            sycl::group_barrier(it.get_group());
            const int r = (int) it.get_global_id(0), wk = (int) it.get_global_id(1);
            float fac[kYsknMaxRows];
            for (int t = 0; t < kYsknMaxRows; ++t) {
                float m = 0.f;
                if (t < T)
                    for (int g = 0; g < G; ++g) m = sycl::fmax(m, alpha[t * G + g]);
                fac[t] = ys::fixed_factor(m);
            }
            int64_t acc[kYsknMaxRows] = {};
            for (int g = wk * gpw; g < (wk + 1) * gpw; ++g) {
                int dot[kYsknMaxRows] = {};
                for (int s = 2 * g; s < 2 * g + 2; ++s) {
                    uint32_t u[4 * K];
#pragma unroll
                    for (int c = 0; c < K; ++c) {
                        const sycl::vec<uint32_t, 4> v =
                            *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(words + (((size_t) s * K + c) * rows + r) * 4);
                        u[4 * c] = v[0];
                        u[4 * c + 1] = v[1];
                        u[4 * c + 2] = v[2];
                        u[4 * c + 3] = v[3];
                    }
                    const int* xs = reinterpret_cast<const int*>(xq + (size_t) s * 256);
#pragma unroll
                    for (int m = 0; m < 64; ++m) {
                        const int wv = (int) ((uint32_t) lut[state_of<K>(u, 2 * m)] |
                                              ((uint32_t) lut[state_of<K>(u, 2 * m + 1)] << 16));
#pragma unroll
                        for (int t = 0; t < kYsknMaxRows; ++t)
                            if (t < T) dot[t] = xe::dp4a(wv, xs[(size_t) t * (cols / 4) + m], dot[t]);
                    }
                }
#pragma unroll
                for (int t = 0; t < kYsknMaxRows; ++t)
                    if (t < T) acc[t] += ys::group_term(dot[t], ys::fixed_scale(alpha[t * G + g], fac[t]));
            }
#pragma unroll
            for (int t = 0; t < kYsknMaxRows; ++t)
                if (t < T) part[((size_t) wk * T + t) * rows + r] = acc[t];
        });
    });
}

// the output's rotation of T rows of `rows` values: value(t, r) in the rotated basis, then H over each 512 rows, the
// signs, the engine's rows (perm_out); without signs, the values as they are
template<class Value>
sycl::event rotate_out_launch(sycl::queue& q, const YsknDev& w, int T, float* y, int64_t ldy, Value value) {
    const int rows = w.rows, used = w.rows_used;
    const int8_t* so = w.sign_out;
    const int32_t* perm = w.perm_out;
    if (!so)
        return q.parallel_for(sycl::nd_range<1>((size_t) T * rows, kRowsWg), [=](sycl::nd_item<1> it) {
            const int i = (int) it.get_global_id(0), t = i / rows, r = i % rows;
            if (r < used) y[(size_t) t * ldy + (perm ? perm[r] : r)] = value(t, r);
        });
    const int nb = rows / kGroup;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(kGroup), h);
        h.parallel_for(sycl::nd_range<1>((size_t) T * nb * 256, 256), [=](sycl::nd_item<1> it) {
            const int lane = (int) it.get_local_id(0), blk = (int) it.get_group(0), t = blk / nb, b = blk % nb;
            float* x = &buf[0];
            for (int i = lane; i < kGroup; i += 256) x[i] = value(t, b * kGroup + i);
            sycl::group_barrier(it.get_group());
            fwht512_local(x, lane, it);
            for (int i = lane; i < kGroup; i += 256) {
                const int r = b * kGroup + i;
                if (r < used) y[(size_t) t * ldy + (perm ? perm[r] : r)] = (x[i] * kInvSqrt512) * (float) so[r];
            }
        });
    });
}

}  // namespace

void yskn_act(const YsknDev& w, const float* x, int64_t ldx, int T, int8_t* xq, float* alpha, void* stream) {
    check(w, "yskn_act");
    if (!x || !xq || !alpha || T <= 0) throw core::DeviceError("yskn_act: invalid arguments");
    auto& q = queue_for(stream);
    const int cols = w.cols, G = cols / kGroup;
    const int8_t* si = w.sign_in;
    const int32_t* perm = w.perm_in;
    const auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(kGroup), h);
        h.parallel_for(sycl::nd_range<1>((size_t) T * G * 256, 256), [=](sycl::nd_item<1> it) {
            const int lane = (int) it.get_local_id(0), blk = (int) it.get_group(0), t = blk / G, g = blk % G;
            float* v = &buf[0];
            for (int i = lane; i < kGroup; i += 256) {
                const int c = g * kGroup + i;
                v[i] = x[(size_t) t * ldx + (perm ? perm[c] : c)] * (float) si[c];
            }
            sycl::group_barrier(it.get_group());
            fwht512_local(v, lane, it);
            float a0 = f32_from_f16(f16_from_f32(v[lane] * kInvSqrt512));
            float a1 = f32_from_f16(f16_from_f32(v[lane + 256] * kInvSqrt512));
            const float amax = sycl::reduce_over_group(it.get_group(), sycl::fmax(sycl::fabs(a0), sycl::fabs(a1)),
                                                       sycl::maximum<float>());
            const float a = amax / 127.f;
            if (lane == 0) alpha[t * G + g] = a;
            auto qz = [a](float f) {
                const int i = a > 0.f ? (int) sycl::rint(f / a) : 0;
                return (int8_t) sycl::clamp(i, -127, 127);
            };
            int8_t* out = xq + (size_t) t * cols + (size_t) g * kGroup;
            out[lane] = qz(a0);
            out[lane + 256] = qz(a1);
        });
    });
    finish_launch(stream, e, "yskn_act");
}

void yskn_gemv(const YsknDev& w, const int8_t* xq, const float* alpha, int T, int64_t* part, void* stream) {
    check(w, "yskn_gemv");
    if (!xq || !alpha || !part || T <= 0 || T > kYsknMaxRows) throw core::DeviceError("yskn_gemv: invalid arguments");
    auto& q = queue_for(stream);
    sycl::event e;
    switch (w.bits) {
        case 4: e = gemv_launch<4>(q, w, xq, alpha, T, part); break;
        case 5: e = gemv_launch<5>(q, w, xq, alpha, T, part); break;
        case 6: e = gemv_launch<6>(q, w, xq, alpha, T, part); break;
        case 7: e = gemv_launch<7>(q, w, xq, alpha, T, part); break;
        default: e = gemv_launch<8>(q, w, xq, alpha, T, part); break;
    }
    finish_launch(stream, e, "yskn_gemv");
}

void yskn_finish(const YsknDev& w, const int64_t* part, const float* alpha, int T, float* y, int64_t ldy, void* stream) {
    check(w, "yskn_finish");
    if (!part || !alpha || !y || T <= 0 || T > kWindowMaxT) throw core::DeviceError("yskn_finish: invalid arguments");
    auto& q = queue_for(stream);
    const int rows = w.rows, G = w.cols / kGroup, nw = w.n_workers;
    const float* tau = w.tau;
    const auto e = rotate_out_launch(q, w, T, y, ldy, [=](int t, int r) {
        float m = 0.f;
        for (int g = 0; g < G; ++g) m = sycl::fmax(m, alpha[t * G + g]);
        int64_t acc = 0;
        for (int k = 0; k < nw; ++k) acc += part[((size_t) k * T + t) * rows + r];
        return ys::finish(acc, m, tau[r]);
    });
    finish_launch(stream, e, "yskn_finish");
}

size_t yskn_scratch_bytes(const YsknDev& w, int T) {
    (void) T;   // sized for the widest window (kYsknMmaRows rows), either form reads it
    const size_t rows = (size_t) kYsknMmaRows;
    const size_t q = (rows * (size_t) w.cols + 255) / 256 * 256;
    const size_t a = (rows * (size_t) (w.cols / kGroup) * 4 + 255) / 256 * 256;
    return q + a + (size_t) w.n_workers * rows * (size_t) w.rows * 8;
}

void yskn_proj(const YsknDev& w, const float* x, int64_t ldx, int T, float* y, int64_t ldy, void* scratch, void* stream) {
    const YsknScratch s = yskn_scratch_parts(w, scratch);
    if (yskn_mma_take(T, stream)) {   // the matrix form: the whole window a launch (up to kYsknMmaRows)
        for (int t0 = 0; t0 < T; t0 += kYsknMmaRows) {
            const int n = std::min(kYsknMmaRows, T - t0);
            yskn_mma_proj(w, x + (size_t) t0 * ldx, ldx, n, y + (size_t) t0 * ldy, ldy, scratch, stream);
        }
        return;
    }
    for (int t0 = 0; t0 < T; t0 += kYsknMaxRows) {
        const int n = std::min(kYsknMaxRows, T - t0);
        yskn_act(w, x + (size_t) t0 * ldx, ldx, n, s.xq, s.alpha, stream);
        yskn_gemv(w, s.xq, s.alpha, n, s.part, stream);
        yskn_finish(w, s.part, s.alpha, n, y + (size_t) t0 * ldy, ldy, stream);
    }
}

void yskn_dequant_f16(const YsknDev& w, uint16_t* out, void* stream) {
    check(w, "yskn_dequant_f16");
    if (!w.lut16 || !w.scale16 || !out) throw core::DeviceError("yskn_dequant_f16: invalid arguments");
    auto& q = queue_for(stream);
    const int rows = w.rows, cols = w.cols, K = w.bits, S = cols / 256;
    const uint32_t* words = w.words;
    const uint16_t* lut16 = w.lut16;
    const uint16_t* sc16 = w.scale16;
    // a work-item a step: (row, unit, step), the step fastest so a work-group's writes are consecutive
    const auto e = q.parallel_for(sycl::range<1>((size_t) rows * S * 128), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int j = (int) (i % 128), s = (int) (i / 128 % S), r = (int) (i / 128 / S);
        auto word = [&](int k) { return words[(((size_t) s * K + (size_t) (k >> 2)) * rows + r) * 4 + (size_t) (k & 3)]; };
        const int en = K * (j + 1);
        uint32_t st;
        if (en < 12) {
            st = (word(0) & ((1u << en) - 1u)) << (12 - en);
        } else {
            const int start = en - 12, wd = start >> 5, sh = start & 31;
            uint32_t v = word(wd) >> sh;
            if (sh + 12 > 32) v |= word(wd + 1) << (32 - sh);
            st = v & 0xFFFu;
        }
        const float sc = f32_from_f16(sc16[r]);
        const uint16_t h0 = f16_from_f32(f32_from_f16(lut16[2 * st]) * sc);
        const uint16_t h1 = f16_from_f32(f32_from_f16(lut16[2 * st + 1]) * sc);
        reinterpret_cast<uint32_t*>(out)[((size_t) r * cols + (size_t) s * 256) / 2 + (size_t) j] = (uint32_t) h0 | ((uint32_t) h1 << 16);
    });
    finish_launch(stream, e, "yskn_dequant_f16");
}

void yskn_rotate_in_f16(const YsknDev& w, const uint16_t* x16, int64_t ldx, int T, uint16_t* out, void* stream) {
    check(w, "yskn_rotate_in_f16");
    if (!x16 || !out || T <= 0) throw core::DeviceError("yskn_rotate_in_f16: invalid arguments");
    auto& q = queue_for(stream);
    const int cols = w.cols, G = cols / kGroup;
    const int8_t* si = w.sign_in;
    const int32_t* perm = w.perm_in;
    const auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(kGroup), h);
        h.parallel_for(sycl::nd_range<1>((size_t) T * G * 256, 256), [=](sycl::nd_item<1> it) {
            const int lane = (int) it.get_local_id(0), blk = (int) it.get_group(0), t = blk / G, g = blk % G;
            float* v = &buf[0];
            for (int i = lane; i < kGroup; i += 256) {
                const int c = g * kGroup + i;
                v[i] = f32_from_f16(x16[(size_t) t * ldx + (perm ? perm[c] : c)]) * (float) si[c];
            }
            sycl::group_barrier(it.get_group());
            fwht512_local(v, lane, it);
            for (int i = lane; i < kGroup; i += 256)
                out[(size_t) t * cols + (size_t) g * kGroup + i] = f16_from_f32(v[i] * kInvSqrt512);
        });
    });
    finish_launch(stream, e, "yskn_rotate_in_f16");
}

void yskn_rotate_out(const YsknDev& w, const float* y_rot, int T, float* y, int64_t ldy, void* stream) {
    check(w, "yskn_rotate_out");
    if (!y_rot || !y || T <= 0) throw core::DeviceError("yskn_rotate_out: invalid arguments");
    auto& q = queue_for(stream);
    const int rows = w.rows;
    const auto e = rotate_out_launch(q, w, T, y, ldy, [=](int t, int r) { return y_rot[(size_t) t * rows + r]; });
    finish_launch(stream, e, "yskn_rotate_out");
}

}  // namespace strata::kernels
