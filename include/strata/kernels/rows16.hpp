// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Row operations that prepare scaled FP16 inputs for products.
#pragma once

#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/row_stats.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::kernels::rows16 {
namespace syclex = sycl::ext::oneapi::experimental;
using f16_gpu::f16_from_f32;

// The same SwiGLU value is used by scaled image rows and the LLMs' saturating FP16 rows.
inline float swiglu(float gate, float up) { return gate / (1.f + sycl::exp(-gate)) * up; }

template<class Value>
inline void saturated(sycl::queue& q, int64_t n, uint16_t* out, Value value) {
    if (n <= 0) return;
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = value((int64_t) i[0]);
        out[i] = f16_from_f32(sycl::isnan(v) ? v : sycl::fmin(sycl::fmax(v, -65504.f), 65504.f));
    });
}

constexpr float kRowLim = 16384.f;

inline float pow2_scale(float amax) {
    return amax > kRowLim ? sycl::exp2(sycl::ceil(sycl::log2(amax / kRowLim))) : 1.f;
}

// A training step's gradient rows, small or large: the power of two that brings the row's largest value into
// [2^13, 2^14), up as well as down.  A gradient is often far below FP16's smallest normal value (6.1e-5: the LoRA
// factors' gradients of Qwen-Image-2.1's real blocks sit near 1e-6), where FP16 keeps a few bits or none.
// The exponent is read off the value's bits (a subnormal's as -127): sycl::ilogb inside this row kernel gave
// 2^-144 for a row whose largest value was 2 (intel/llvm v7.2.0, the B70, 2026-11-09).
inline float pow2_full(float amax) {
    if (!(amax > 0.f)) return 1.f;
    const int e = (int) ((sycl::bit_cast<uint32_t>(amax) >> 23) & 0xffu) - 127;
    return sycl::bit_cast<float>((uint32_t) (sycl::clamp(e - 13, -126, 127) + 127) << 23);
}

// A row-wise kernel writing FP16 with each row's scale: Op (LnMod, Gelu, Heads below) gives a row's statistics and its
// values; a device-selected sub-group a row, 256 work-items a work-group.
constexpr int64_t kSG = 32, kRows = 8;

// the row's scale is taken from a bound of |op.value| where the op has a cheaper one than the value itself
template<class Op, class St> float bound(const Op& op, int64_t t, int64_t c, const St& st) {
    if constexpr (requires { op.bound(t, c); }) return op.bound(t, c);
    else return sycl::fabs(op.value(t, c, st));
}

template<int SG, class Op> struct Rows16 {
    static constexpr int64_t Rows = 256 / SG;
    Op op;
    int64_t T, C;
    uint16_t* y16;
    float* rs;
    float *keep_mean = nullptr, *keep_inv = nullptr;   // the rows' statistics kept (a training step's backward)
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t t = static_cast<int64_t>(it.get_group_linear_id()) * Rows + sg.get_group_linear_id();
        const int64_t lane = sg.get_local_linear_id();
        if (t >= T) return;
        const auto sum = [&](float v) { return sycl::reduce_over_group(sg, v, sycl::plus<float>()); };
        const auto st = op.stats(t, lane, SG, sum);
        float m = 0.f;
        for (int64_t c = lane; c < C; c += SG) m = sycl::fmax(m, bound(op, t, c, st));
        const float rmax = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
        float sc;
        if constexpr (requires { Op::kFullRange; }) sc = pow2_full(rmax);
        else sc = pow2_scale(rmax);
        const float inv = 1.f / sc;
        for (int64_t c = lane; c < C; c += SG) y16[t * C + c] = f16_from_f32(op.value(t, c, st) * inv);
        if (lane == 0) {
            // A zero gradient row contributes nothing.  Its scale must be zero as well: a LoRA's transposed
            // backward folds this scale into the other operand before finding that operand's global FP16 scale.
            if (rs != nullptr) {
                if constexpr (requires { Op::kFullRange; }) rs[t] = rmax == 0.f ? 0.f : sc;
                else rs[t] = sc;
            }
            if (keep_mean != nullptr) keep_mean[t] = st.mean;
            if (keep_inv != nullptr) keep_inv[t] = st.inv;
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<SG>}; }
};

// mean and inv (null: not kept): the rows' statistics (Stats) the Op gave, which a training step's backward reads
template<class Op> inline void rows16(const Op& op, int64_t T, int64_t C, uint16_t* y16, float* rs, sycl::queue& q,
                                      float* mean = nullptr, float* inv = nullptr) {
    if (T <= 0) return;
    auto launch = [&]<int SG>() {
        constexpr int64_t Rows = 256 / SG;
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((T + Rows - 1) / Rows * Rows * SG)),
                                         sycl::range<1>(256)), Rows16<SG, Op>{op, T, C, y16, rs, mean, inv});
    };
    // B70, 512-square generation, 16 -> 32: Mage-Flow 0.246 -> 0.217 s, MOSS-Image 0.546 -> 0.500 s,
    // Anima 0.433 -> 0.402 s (warm sampling medians, 3 runs). Prefer 32 when the device supports it.
    const auto widths = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(widths.begin(), widths.end(), 32) != widths.end()) launch.template operator()<32>();
    else launch.template operator()<16>();
}


}  // namespace strata::kernels::rows16
