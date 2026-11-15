// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/q8_act.hpp - the activations' int8 blocks of 32 values: one kernel for every layout and
// rounding the engine's int8 products read.
//
// A block's scale d comes from the largest |x| of its values and each value's int8 from x and d by one of the rules
// below; which rule is a contract with what reads the bytes (ggml's reference, the CPU's expert kernels, the decode's
// q8_1), so each keeps its own and none is rounded like another.  The layouts: q8_1 (36 bytes: d and the values' sum
// as an fp16 pair, then the 32 values), Q8_0 (34 bytes: d as fp16, then the values; optionally d, and d times the
// int8 values' sum, as FP32 beside them) and the MMQ rows (each row's int8 values, then a FP32 scale a block, then d
// times the block's int8 sum: iq_mmq_act_bytes).  A sub-group of 32 lanes takes V blocks, each block's 32 / V lanes
// V values of it side by side, and the block's lanes reduce its largest |x| (and its sums) among themselves.  Src
// gives value j of a block (of a row, where it reads rows: kRows), Out stores the values and the block's scale.
//
// The kernels are named types in this namespace: SYCL names a kernel by its type, and a source or an output in an
// anonymous namespace would mangle alike in two files (_GLOBAL__N_1).
#pragma once

#include "strata/kernels/canon_math.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels::q8act {

namespace syclex = sycl::ext::oneapi::experimental;

constexpr int kBlock = 32;
constexpr int kSubGroups = 8;    // sub-groups a work-group
using u32x2 = uint32_t __attribute__((ext_vector_type(2)));   // one 8-byte store (a sycl::vec of bytes is split on NVPTX)

/// #606: a q8_1 block's fp16 scale and sum held to the largest finite half.  A sum past 65504 (one massive-activation
/// dimension among 32 values is enough) rounds to inf, and d does past amax = 8.3M; the dot products then read
/// inf * 0 = NaN, and a NaN in the residual ends as one token repeated (ggml-org/llama.cpp#23606 is the same defect).
/// Clamped, such a block gives an approximate product instead of NaN.  Every block that was finite before is stored
/// bit for bit as before: values in (65504, 65520) round to 65504 anyway.  A NaN stays NaN.
inline float finite(const float v) { return sycl::fabs(v) > canon::kHalfMax ? sycl::copysign(canon::kHalfMax, v) : v; }

// ---- the rules: scale(amax), inv(d) where value uses it, value(x, d, inv, amax)

/// The decode's q8_1 (native_mmvq.cpp, from llama.cpp's quantize_q8_1): d = amax / 127 held finite, round(x / d) held
/// to +-127 (past them only where d was clamped), 0 for an all-zero block.
struct Ggml {
    static constexpr int kGroup = kBlock;   // values a scale
    static float scale(float amax) { return finite(amax / 127.0f); }
    static float inv(float) { return 0.0f; }
    static int8_t value(float x, float d, float, float amax) {
        if (amax == 0.0f) return 0;
        const float q = sycl::round(x / d);
        return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
    }
};
/// ggml's reference rows (quantize_row_q8_0 and MMQ's q8_1): d = amax / 127, round(x times 1 / d); the image engine's
/// int8 rows and MMQ's activations.  A d that underflows to 0 gives zeros.
struct Mul {
    static constexpr int kGroup = kBlock;
    static float scale(float amax) { return amax / 127.0f; }
    static float inv(float d) { return d != 0.0f ? 1.0f / d : 0.0f; }
    static int8_t value(float x, float, float id, float) { return (int8_t) (int) sycl::round(x * id); }
};
/// The CPU experts' rule (cpu/expert.cpp's AVX2 rows, the same bits on the GPU): x times 1 / d rounded half away from
/// zero by adding +-0.5 and truncating, held to +-127.
struct Half {
    static constexpr int kGroup = kBlock;
    static float scale(float amax) { return amax > 0.0f ? amax / 127.0f : 0.0f; }
    static float inv(float d) { return d > 0.0f ? 1.0f / d : 0.0f; }
    static int8_t value(float x, float, float inv, float) {
        STRATA_CANON_NO_CONTRACT
        const float t = x * inv;
        const int v = (int) (t + (t >= 0.0f ? 0.5f : -0.5f));
        return (int8_t) (v < -127 ? -127 : (v > 127 ? 127 : v));
    }
};
/// canon_math.hpp's scale and value (the experts' rows on the CPU and every GPU alike), G values a scale: 32, or 256
/// for the rows of whole 256s (an i-quant's block of 256 then shares it).
template<int G>
struct Canon {
    static constexpr int kGroup = G;
    static float scale(float amax) { return canon::q8_scale(amax); }
    static float inv(float d) { return canon::q8_inv(d); }
    static int8_t value(float x, float, float id, float) { return canon::q8_value(x, id); }
};
/// ggml's quantize_row_q8_0 to its bytes (moe_hit_parity checks them): x / d in D (double where the device has FP64)
/// rounded to the nearest even, held to [-128, 127]; d = amax / 127, 0 for an all-zero block.
template<typename D>
struct Ref {
    static constexpr int kGroup = kBlock;
    static float scale(float amax) { return amax / 127.0f; }
    static float inv(float) { return 0.0f; }
    static int8_t value(float x, float d, float, float amax) {
        if (amax == 0.0f) return 0;
        D q = sycl::rint((D) x / (D) d);
        if (q > (D) 127) q = (D) 127;
        if (q < (D) -128) q = (D) -128;
        return (int8_t) q;
    }
};

// ---- the layouts: put(b, r, c, j0, q) stores values j0 .. j0 + V - 1 of block b (block c of row r); head(b, r, c,
// d, xsum, qsum) its scale, the block's first lane only.  kXSum, kQSum: whether head reads the values' sum (FP32) or
// the int8 values' sum.

/// q8_1 blocks of 36 bytes, block b at y + 36 b: d and the values' sum (finite) as an fp16 pair, then the values
struct Q8_1 {
    static constexpr bool kXSum = true, kQSum = false;
    uint8_t* y;
    template<int V> void put(int64_t b, int64_t, int64_t, int j0, const int8_t (&q)[V]) const {
        int8_t* qs = reinterpret_cast<int8_t*>(y + b * 36 + 4) + j0;
#pragma unroll
        for (int k = 0; k < V; ++k) qs[k] = q[k];
    }
    void head(int64_t b, int64_t, int64_t, float d, float xsum, int) const {
        *reinterpret_cast<sycl::half2*>(y + b * 36) = sycl::half2(sycl::half(d), sycl::half(finite(xsum)));
    }
};
/// Q8_0 blocks of 34 bytes: d as fp16, then the values; `scales` (null: none) d as FP32 a block, and with QSUM `sums`
/// d times the int8 values' sum
template<bool QSUM = false>
struct Q8_0 {
    static constexpr bool kXSum = false, kQSum = QSUM;
    uint8_t* blocks;
    float* scales = nullptr;
    float* sums = nullptr;
    template<int V> void put(int64_t b, int64_t, int64_t, int j0, const int8_t (&q)[V]) const {
        int8_t* qs = reinterpret_cast<int8_t*>(blocks + b * 34 + 2) + j0;
#pragma unroll
        for (int k = 0; k < V; ++k) qs[k] = q[k];
    }
    void head(int64_t b, int64_t, int64_t, float d, float, int qsum) const {
        *reinterpret_cast<uint16_t*>(blocks + b * 34) = f16_from_f32(d);
        if (scales != nullptr) scales[b] = d;
        if constexpr (QSUM) sums[b] = d * (float) qsum;
    }
};
/// MMQ's rows (iq_mmq_quantize): row r's int8 values at x8 + r oc, then a FP32 scale a block (onb a row), then, with
/// SUMS, d times the block's int8 sum; each block written `rep` times (Q2_K's dup: rows of twice the values).
/// SCATTER: row r is a token's, stored at rows slot[r k_used + k] for each of its k_used.
template<bool SUMS, bool SCATTER = false>
struct Rows {
    static constexpr bool kXSum = false, kQSum = SUMS;
    int8_t* x8;
    float *xs, *xu;
    int64_t oc, onb;
    int rep = 1;
    const int32_t* slot = nullptr;
    int k_used = 1;
    int64_t row_of(int64_t r, int k) const {
        if constexpr (SCATTER) return slot[r * k_used + k];
        else return r;
    }
    template<int V> void put(int64_t, int64_t r, int64_t c, int j0, const int8_t (&q)[V]) const {
        uint32_t w[2] = {};
        if constexpr (V == 8)
#pragma unroll
            for (int k = 0; k < V; ++k) w[k / 4] |= (uint32_t) (uint8_t) q[k] << (8 * (k % 4));
        for (int k = 0; k < (SCATTER ? k_used : 1); ++k) {
            const int64_t row = row_of(r, k);
            for (int e = 0; e < rep; ++e) {
                int8_t* o = x8 + row * oc + (c * rep + e) * kBlock + j0;
                if constexpr (V == 8) *reinterpret_cast<u32x2*>(o) = u32x2{w[0], w[1]};
                else
#pragma unroll
                    for (int j = 0; j < V; ++j) o[j] = q[j];
            }
        }
    }
    void head(int64_t, int64_t r, int64_t c, float d, float, int qsum) const {
        for (int k = 0; k < (SCATTER ? k_used : 1); ++k) {
            const int64_t row = row_of(r, k);
            for (int e = 0; e < rep; ++e) {
                const int64_t ob = row * onb + c * rep + e;
                xs[ob] = d;
                if constexpr (SUMS) xu[ob] = d * (float) qsum;
            }
        }
    }
};

// ---- the sources: value j of block c of row r (kRows: rows of `blocks` blocks; else r is 0 and c the block)

/// FP32 values, block c at x + 32 c
struct Flat {
    static constexpr bool kRows = false;
    const float* x;
    float operator()(int64_t, int64_t c, int j) const { return x[c * kBlock + j]; }
};
/// FP32 rows, `ld` values apart, row r of the output from row ids[r] (null: r)
struct FloatRows {
    static constexpr bool kRows = true;
    const float* x;
    const int32_t* ids;
    int64_t ld;
    float operator()(int64_t r, int64_t c, int j) const {
        return x[(int64_t) (ids != nullptr ? ids[r] : r) * ld + c * kBlock + j];
    }
};
/// FloatRows with each column scaled (x * scale[column]) as it is read: the latent pack's shared column scales
/// folded into the quantization, the same product the separate pass over the rows wrote
struct ScaledFloatRows {
    static constexpr bool kRows = true;
    const float* x;
    const int32_t* ids;
    int64_t ld;
    const float* scale;
    float operator()(int64_t r, int64_t c, int j) const {
        const int64_t k = c * kBlock + j;
        return x[(int64_t) (ids != nullptr ? ids[r] : r) * ld + k] * scale[k];
    }
};
/// FP16 rows of `cols` values
struct HalfRows {
    static constexpr bool kRows = true;
    const uint16_t* x;
    int64_t cols;
    float operator()(int64_t r, int64_t c, int j) const {
        return (float) sycl::bit_cast<sycl::half>(x[r * cols + c * kBlock + j]);
    }
};
/// BF16 values, block c at x + 32 c
struct FlatBf16 {
    static constexpr bool kRows = false;
    const uint16_t* x;
    float operator()(int64_t, int64_t c, int j) const {
        return sycl::bit_cast<float>((uint32_t) x[c * kBlock + j] << 16);
    }
};

// ---- the kernel

/// A block of 32 values a sub-group, lane l's x: its largest |x| and sum by butterflies over the lanes (16 down to 1:
/// the decode's warp_max and warp_sum, the sum from the lane's own value), its scale and the lane's int8 value.
struct Lane {
    float amax, d, xsum;
    int8_t q;
};
template<class Rule, bool XSUM = true>
inline Lane block32(const sycl::sub_group& sg, const float x) {
    float a = sycl::fabs(x), s = x;
#pragma unroll
    for (int o = kBlock / 2; o > 0; o >>= 1) {
        a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, o));
        if constexpr (XSUM) s += sycl::permute_group_by_xor(sg, s, o);
    }
    const float d = Rule::scale(a);
    return {a, d, s, Rule::value(x, d, Rule::inv(d), a)};
}
/// That block as q8_1 at blk (36 bytes), inside another kernel (the norms and the SwiGLUs that write their output's
/// q8_1 themselves): the bytes Blocks<1, Rule, ..., Q8_1> writes
template<class Rule = Ggml>
inline void q8_1_block(const sycl::sub_group& sg, const float x, uint8_t* blk, const int lane) {
    const Lane l = block32<Rule>(sg, x);
    blk[4 + lane] = (uint8_t) l.q;
    if (lane == 0) *reinterpret_cast<sycl::half2*>(blk) = sycl::half2(sycl::half(l.d), sycl::half(finite(l.xsum)));
}

template<int V, class Rule, class Src, class Out>
struct Blocks {
    static_assert(V == 1 || V == 8);
    static_assert(Rule::kGroup == kBlock || Rule::kGroup == kBlock * V, "a scale over a block or the sub-group");
    static constexpr int L = kBlock / V;   // lanes a block
    Src src;
    Out out;
    int64_t nb, blocks;   // blocks in all, and a row (kRows)
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t first = ((int64_t) it.get_group(0) * kSubGroups + (int64_t) sg.get_group_linear_id()) * V;
        if (first >= nb) return;   // the whole sub-group
        const int64_t b = first + lane / L;
        const bool in = b < nb;
        const int j0 = lane % L * V;
        int64_t r = 0, c = b;
        if constexpr (Src::kRows) {
            r = b / blocks;
            c = b - r * blocks;
        }
        if constexpr (V == 1) {   // a block a sub-group (V 1: b is the sub-group's)
            const Lane l = block32<Rule, Out::kXSum>(sg, src(r, c, lane));
            int qs = 0;
            if constexpr (Out::kQSum) qs = sycl::reduce_over_group(sg, (int) l.q, sycl::plus<int>());
            const int8_t q[1] = {l.q};
            out.template put<1>(b, r, c, lane, q);
            if (lane == 0) out.head(b, r, c, l.d, l.xsum, qs);
            return;
        }
        // a block of L lanes: the largest |x| and the sum among them up from 1 (the CPU's canon_expert.cpp adds its
        // sums in that order), each lane's V values first, from 0
        float v[V], a = 0.0f, s = 0.0f;
#pragma unroll
        for (int k = 0; k < V; ++k) {
            v[k] = in ? src(r, c, j0 + k) : 0.0f;
            a = sycl::fmax(a, sycl::fabs(v[k]));
            if constexpr (Out::kXSum) s += v[k];
        }
#pragma unroll
        for (int o = 1; o < L; o <<= 1) {
            if constexpr (Rule::kGroup == kBlock) a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, o));
            if constexpr (Out::kXSum) s += sycl::permute_group_by_xor(sg, s, o);
        }
        if constexpr (Rule::kGroup != kBlock) a = sycl::reduce_over_group(sg, a, sycl::maximum<float>());
        const float d = Rule::scale(a), id = Rule::inv(d);
        int8_t q[V];
        int qs = 0;
#pragma unroll
        for (int k = 0; k < V; ++k) {
            q[k] = Rule::value(v[k], d, id, a);
            if constexpr (Out::kQSum) qs += q[k];
        }
        if constexpr (Out::kQSum)
#pragma unroll
            for (int o = 1; o < L; o <<= 1) qs += sycl::permute_group_by_xor(sg, qs, o);
        if (!in) return;
        out.template put<V>(b, r, c, j0, q);
        if (j0 == 0) out.head(b, r, c, d, s, qs);
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<32>}; }
};

/// nb blocks of src into out, V values a lane; `blocks`: the blocks a row where Src reads rows
template<int V, class Rule, class Src, class Out>
sycl::event quantize(sycl::queue& q, int64_t nb, int64_t blocks, const Src& src, const Out& out) {
    const int64_t sgs = (nb + V - 1) / V;
    const size_t groups = (size_t) ((sgs + kSubGroups - 1) / kSubGroups);
    return q.parallel_for(sycl::nd_range<1>(groups * (size_t) (kSubGroups * 32), (size_t) (kSubGroups * 32)),
                          Blocks<V, Rule, Src, Out>{src, out, nb, blocks});
}

}  // namespace strata::kernels::q8act
