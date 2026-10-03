// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/s2_gemv.cpp - the Xe ports of Strata's src/kernels/cuda/dequant_s2.cu, s2_gemv.cu and s2_gemv_q8.cu: the
// S2 (Q2_0 canonical form) decode, its naive fp16-activation GEMV, and the GEMV over Q8_0 activations.  Work shapes
// and summation orders are the CUDA kernels'.
#include "strata/kernels/dequant_s2.hpp"
#include "strata/kernels/s2_gemv.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"

#include <string>

namespace strata::kernels {
namespace {

constexpr int QK = 64;             // S2 group = Q2_0 block = 64 elements
constexpr int CODES_PER_BYTE = 4;
constexpr int QK8_0 = 32;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

}  // namespace

void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t n_blocks) {
    if (n_blocks <= 0) return;
    const long long nb = n_blocks;
    const auto e = queue_for(nullptr).parallel_for(sycl::range<1>((size_t) nb), [=](sycl::id<1> id) {
        const long long b = (long long) id[0];
        const float d = scales[b];
        const uint8_t* c = codes + b * (QK / CODES_PER_BYTE);
        float* y = out + b * QK;
        for (int j = 0; j < QK; ++j) {
            const int code = (c[j / CODES_PER_BYTE] >> ((j % CODES_PER_BYTE) * 2)) & 0x03;
            y[j] = (float) (code - 1) * d;      // the -1 is applied to the CODE, then ONE multiply
        }
    });
    finish(nullptr, e, "dequant_s2");
}

void s2_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
             int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK != 0)
        throw core::DeviceError("s2_gemv: n_in " + std::to_string(n_in) + " is not a multiple of 64");
    const long long ni = n_in;
    const auto e = queue_for(nullptr).parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> id) {
        const long long o = (long long) id[0];
        const long long nb = ni / QK;
        const uint8_t* c = codes + o * nb * (QK / 4);
        const float* s = scales + o * nb;
        float acc = 0.0f;
        for (long long b = 0; b < nb; ++b) {
            const float d = s[b];
            const uint8_t* cb = c + b * (QK / 4);
            const uint16_t* xb = x + b * QK;
            for (int j = 0; j < QK; ++j) {
                const int code = (cb[j >> 2] >> ((j & 3) * 2)) & 0x03;
                acc += (float) (code - 1) * d * f32_from_f16(xb[j]);
            }
        }
        y[o] = acc;
    });
    finish(nullptr, e, "s2_gemv");
}

// One work-group of threads_per_row work-items per output row, one code byte (four elements) per step; the four
// activations lie in one Q8_0 block, so its scale is read once.
void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                int64_t n_out, int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK8_0 != 0 || n_in % QK != 0)
        throw core::DeviceError("s2_gemv_q8: n_in " + std::to_string(n_in) + " must be a multiple of 64");
    const long long ni = n_in;
    const int tpr = threads_per_row;
    const auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>((size_t) tpr), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_out * tpr, tpr), [=](sycl::nd_item<1> it) {
            const long long o = (long long) it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const long long n_quads = ni / 4;
            const uint8_t* c = codes + o * n_quads;
            const float* s = scales + o * (ni / QK);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (long long q = tid; q < n_quads; q += tpr) {
                const uint8_t byte = c[q];
                const float d = s[q >> 4];
                const uint8_t* blk = act + ((q * 4) / QK8_0) * 34;
                const float dx = f32_from_f16((uint16_t) (blk[0] | (blk[1] << 8)));
                const int8_t* xq = reinterpret_cast<const int8_t*>(blk + 2);
                const int off = (int) ((q * 4) % QK8_0);
                const float w0 = (float) ((int) (byte & 3) - 1) * d;
                const float w1 = (float) ((int) ((byte >> 2) & 3) - 1) * d;
                const float w2 = (float) ((int) ((byte >> 4) & 3) - 1) * d;
                const float w3 = (float) ((int) ((byte >> 6) & 3) - 1) * d;
                a0 += w0 * ((float) xq[off + 0] * dx);
                a1 += w1 * ((float) xq[off + 1] * dx);
                a2 += w2 * ((float) xq[off + 2] * dx);
                a3 += w3 * ((float) xq[off + 3] * dx);
            }
            partial[tid] = (a0 + a1) + (a2 + a3);
            sycl::group_barrier(it.get_group());
            for (int step = tpr / 2; step > 0; step >>= 1) {
                if (tid < step) partial[tid] += partial[tid + step];
                sycl::group_barrier(it.get_group());
            }
            if (tid == 0) y[o] = partial[0];
        });
    });
    finish(stream, e, "s2_gemv_q8");
}

}  // namespace strata::kernels
