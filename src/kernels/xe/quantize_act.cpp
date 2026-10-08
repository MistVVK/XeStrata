// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/quantize_act.cpp - the Xe port of Strata's src/kernels/cuda/quantize_act.cu (activation Q8_0 and Q8_K).
//
// One work-item per block, as in the CUDA kernels; the rounding rules are theirs (see the header and the CUDA file).
// The library compiles with -ffp-contract=off, which keeps Q8_K's `iscale * x` the separately rounded multiply that
// CUDA's __fmul_rn spells.
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"

#include <cstring>
#include <string>

namespace strata::kernels {
namespace {

constexpr int QK8_0 = 32;
constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}
void require_multiple(int64_t n, int block, const char* what) {
    if (n % block != 0)
        throw core::DeviceError(std::string(what) + ": n " + std::to_string(n) + " is not a multiple of " +
                                std::to_string(block));
}

inline int nearest_int_dev(float fval) {
    const float val = fval + 12582912.0f;
    int i;
    std::memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}

// D: double where the device has FP64 (the reference's rounding), float elsewhere (device_caps.hpp)
template <typename D>
sycl::event quantize_q8_0_launch(sycl::queue& queue, const float* x, uint8_t* blocks, int64_t n) {
    return queue.parallel_for(sycl::range<1>((size_t) (n / QK8_0)), [=](sycl::id<1> id) {
        const int64_t b = (int64_t) id[0];
        const float* xb = x + b * QK8_0;
        uint8_t* out = blocks + b * 34;                 // { fp16 d ; int8 qs[32] }
        float amax = 0.0f;
        for (int i = 0; i < QK8_0; ++i) amax = sycl::fmax(amax, sycl::fabs(xb[i]));
        if (amax == 0.0f) {
            const uint16_t zb = f16_from_f32(0.0f);
            out[0] = (uint8_t) (zb & 0xFF);
            out[1] = (uint8_t) (zb >> 8);
            for (int i = 0; i < QK8_0; ++i) out[2 + i] = 0;
            return;
        }
        const float d32 = amax / 127.0f;
        const uint16_t d16bits = f16_from_f32(d32);
        out[0] = (uint8_t) (d16bits & 0xFF);
        out[1] = (uint8_t) (d16bits >> 8);
        for (int i = 0; i < QK8_0; ++i) {
            D q = sycl::rint((D) xb[i] / (D) d32);
            if (q > (D) 127) q = (D) 127;
            if (q < (D) -128) q = (D) -128;
            out[2 + i] = (uint8_t) (int8_t) q;
        }
    });
}
}  // namespace

void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    require_multiple(n, QK8_0, "quantize_q8_0");
    auto& q = queue_for(stream);
    const auto e = xe::has_fp64(q) ? quantize_q8_0_launch<double>(q, x, blocks, n)
                                   : quantize_q8_0_launch<float>(q, x, blocks, n);
    finish(stream, e, "quantize_q8_0");
}

void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream) {
    if (n <= 0) return;
    require_multiple(n, QK8_0, "quantize_q8_0_scaled");
    if (scales == nullptr) throw core::DeviceError("quantize_q8_0_scaled: scales is null");
    // one sub-group a block (upstream 3281ac32): lane i holds value i, the block's max by xor shuffles (fmax ignores
    // NaN as the sequential walk did, and the max does not depend on the order)
    const int64_t n_blocks = n / QK8_0;
    const size_t groups = (size_t) ((n_blocks + 7) / 8);
    const auto e = queue_for(stream).parallel_for(sycl::nd_range<1>(groups * 256, 256),
                                                  [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int64_t b = (int64_t) it.get_group(0) * 8 + (int64_t) sg.get_group_linear_id();
        if (b >= n_blocks) return;   // uniform over the sub-group
        const int lane = (int) sg.get_local_linear_id();
        const float xv = x[b * QK8_0 + lane];
        uint8_t* out = blocks + b * 34;
        float amax = sycl::fabs(xv);
        for (int o = 16; o > 0; o >>= 1) amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        if (lane == 0) {
            scales[b] = s;
            const uint16_t d16bits = f16_from_f32(s);
            out[0] = (uint8_t) (d16bits & 0xFF);
            out[1] = (uint8_t) (d16bits >> 8);
        }
        const float t = xv * inv;
        const float r = t + (t >= 0.f ? 0.5f : -0.5f);
        int v = (int) r;
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + lane] = (uint8_t) (int8_t) v;
    });
    finish(stream, e, "quantize_q8_0_scaled");
}

void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n / QK8_0)), [=](sycl::id<1> id) {
        const int64_t b = (int64_t) id[0];
        const uint8_t* blk = blocks + b * 34;
        const float d = f32_from_f16((uint16_t) (blk[0] | (blk[1] << 8)));
        float* out = x + b * QK8_0;
        for (int i = 0; i < QK8_0; ++i) out[i] = (float) (int8_t) blk[2 + i] * d;
    });
    finish(stream, e, "dequant_q8_0");
}

void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    require_multiple(n, QK_K, "quantize_q8_K");
    // d is stored as one 4-byte word: with only byte alignment known, the NVPTX backend stores its low byte straight
    // from the rcp.rn.f32 register (st.global.b8) and CUDA 12.4's ptxas writes 0 there (quantize_act_parity, RTX 4070)
    if ((uintptr_t) blocks % 4 != 0) throw core::DeviceError("quantize_q8_K: blocks is not 4-byte aligned");
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n / QK_K)), [=](sycl::id<1> id) {
        const int64_t b = (int64_t) id[0];
        const float* xb = x + b * QK_K;
        uint8_t* out = static_cast<uint8_t*>(__builtin_assume_aligned(blocks + b * Q8K_BYTES, 4));
        int8_t* qs = (int8_t*) (out + 4);
        uint8_t* bsums = out + 4 + QK_K;
        float max = 0.0f, amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            const float ax = sycl::fabs(xb[j]);
            if (ax > amax) {
                amax = ax;
                max = xb[j];
            }
        }
        if (amax == 0.0f) {
            const float zero = 0.0f;
            std::memcpy(out, &zero, 4);
            for (int j = 0; j < QK_K; ++j) qs[j] = 0;
            for (int j = 0; j < 2 * QK_K / 16; ++j) bsums[j] = 0;
            return;
        }
        const float iscale = -127.0f / max;          // -127, NOT -128; see the header
        for (int j = 0; j < QK_K; ++j) {
            const int v = nearest_int_dev(iscale * xb[j]);
            qs[j] = (int8_t) sycl::min(127, v);      // MIN only - the source has no lower clamp
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
            const int16_t s16 = (int16_t) sum;
            std::memcpy(bsums + 2 * j, &s16, 2);
        }
        const float d = 1.0f / iscale;
        std::memcpy(out, &d, 4);
    });
    finish(stream, e, "quantize_q8_K");
}

void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n / QK_K)), [=](sycl::id<1> id) {
        const int64_t b = (int64_t) id[0];
        const uint8_t* blk = blocks + b * Q8K_BYTES;
        float d;
        std::memcpy(&d, blk, 4);
        const int8_t* qs = (const int8_t*) (blk + 4);
        float* out = x + b * QK_K;
        for (int i = 0; i < QK_K; ++i) out[i] = (float) qs[i] * d;
    });
    finish(stream, e, "dequant_q8_K");
}

}  // namespace strata::kernels
