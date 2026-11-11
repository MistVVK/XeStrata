// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/rows_i8.cpp - see include/strata/kernels/rows_i8.hpp.  Moved from the image engine
// (src/image/qi_kernels.cpp's qi_rows_i8, qi_rows_i8_blocks, qi_i8_scale) so the LLMs' prompt products reach them.

#include "strata/kernels/rows_i8.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"

#include <sycl/sycl.hpp>

namespace strata::kernels {

// The W8A8 inputs' sources (RowsI8Of's ops), in a namespace of their own: SYCL names a kernel by its type, and an
// anonymous namespace mangles the same in every file.
namespace rows_i8_src {

using strata::kernels::f32_from_bf16;
using f16_gpu::f16_from_f32;   // the GPU's conversions (f16_gpu.hpp), as the image engine's quantizer took them
using f16_gpu::f32_from_f16;

// rows_i8's values: V of row r from column k, FP16 or BF16
struct I8From16 {
    static constexpr bool kRms = false, kKeep = false;
    const uint16_t* src;
    bool bf16;
    int64_t K;
    const float* rs;   // the rows' factors on their scales (null: 1)
    template<int V> void load_sg(const sycl::sub_group&, int64_t r, int64_t k, float* v) const {
        const auto raw = *reinterpret_cast<const sycl::vec<uint16_t, V>*>(src + r * K + k);
        for (int j = 0; j < V; ++j) v[j] = bf16 ? f32_from_bf16(raw[j]) : f32_from_f16(raw[j]);
    }
    float row_scale(int64_t r) const { return rs != nullptr ? rs[r] : 1.f; }
};

// rows_i8_blocks's: a row of GGUF blocks, each value as kernels::dequant_f16 writes it (rounded to FP16), so the
// int8 rows are those of rows_i8 of its output.  RowsI8Of gives a sub-group one block (256 values, 8 a lane): its
// lanes read the block's bytes 8 a lane in 16-bit loads side by side (a block is 2-byte aligned) and take the bytes
// each value needs from the lane that read them, not a byte at a time from all over the block.
template<int TYPE> struct I8FromBlocks {
    static constexpr bool kRms = false, kKeep = false;
    static constexpr int BYTES = TYPE == 12 ? 144 : TYPE == 13 ? 176 : 210;
    const uint8_t* b;
    int64_t row_bytes;
    // inlined: else its stores through v put the values on the stack (RTX 4070, 2026-11-05: 8 stores and 4 loads a
    // lane's 8 values)
    template<int V>
    __attribute__((always_inline)) void load_sg(const sycl::sub_group& sg, int64_t r, int64_t k, float* v) const {
        static_assert(V == 8, "a lane's 8 values: 32 lanes a block");
        const uint8_t* blk = b + r * row_bytes + (k / 256) * BYTES;
        const int lane = (int) sg.get_local_linear_id();
        uint32_t w0 = 0, w1 = 0;   // the block's bytes 8 lane .. 8 lane + 7
        const auto* p = reinterpret_cast<const uint16_t*>(blk + (ptrdiff_t) 8 * lane);
        if (8 * lane + 8 <= BYTES) {
            w0 = (uint32_t) p[0] | (uint32_t) p[1] << 16;
            w1 = (uint32_t) p[2] | (uint32_t) p[3] << 16;
        } else if (8 * lane < BYTES) {   // Q6_K's d, the last 2 bytes
            w0 = p[0];
        }
        auto byte = [](uint32_t lo, uint32_t hi, int i) { return (int) (((i < 4 ? lo : hi) >> (8 * (i & 3))) & 0xFF); };
        auto f16 = [](uint32_t bits) { return f32_from_f16((uint16_t) bits); };
        if constexpr (TYPE == 14) {   // Q6_K: ql[128] qh[64] scales[16] d, 256 values
            const int n = lane / 16, qu = lane % 16 / 4, l4 = lane % 4;
            const int lq = 8 * n + 4 * (qu & 1) + l4, lh = 16 + 4 * n + l4, ls = 24 + n;
            const uint32_t q0 = sycl::select_from_group(sg, w0, lq), q1 = sycl::select_from_group(sg, w1, lq);
            const uint32_t h0 = sycl::select_from_group(sg, w0, lh), h1 = sycl::select_from_group(sg, w1, lh);
            const uint32_t s0 = sycl::select_from_group(sg, w0, ls), s1 = sycl::select_from_group(sg, w1, ls);
            const float d = f16(sycl::select_from_group(sg, w0, 26));
            const float sc = d * (float) (int8_t) byte(s0, s1, l4 / 2 + 2 * qu);
#pragma unroll
            for (int j = 0; j < V; ++j) {
                const int ql = byte(q0, q1, j), lo = qu < 2 ? (ql & 0xF) : (ql >> 4);
                const int q = lo | ((byte(h0, h1, j) >> (2 * qu)) & 3) << 4;
                v[j] = f32_from_f16(f16_from_f32(sc * (float) (q - 32)));
            }
        } else {   // Q4_K (12): d dmin scales[12] qs[128]; Q5_K (13): d dmin scales[12] qh[32] qs[128]
            const int gi = lane / 4, j64 = gi / 2, hi = gi % 2, l4 = lane % 4;
            const uint32_t a0 = sycl::group_broadcast(sg, w0, 0), a1 = sycl::group_broadcast(sg, w1, 0);
            const uint32_t c0 = sycl::group_broadcast(sg, w0, 1), c1 = sycl::group_broadcast(sg, w1, 1);
            const int lq = (TYPE == 12 ? 2 : 6) + 4 * j64 + l4;
            const uint32_t q0 = sycl::select_from_group(sg, w0, lq), q1 = sycl::select_from_group(sg, w1, lq);
            uint32_t h0 = 0, h1 = 0;
            if constexpr (TYPE == 13) {
                h0 = sycl::select_from_group(sg, w0, 2 + l4);
                h1 = sycl::select_from_group(sg, w1, 2 + l4);
            }
            const float d = f16(a0 & 0xFFFF), dmin = f16(a0 >> 16);
            auto scb = [&](int i) { return i < 4 ? byte(a1, a1, i) : byte(c0, c1, i - 4); };   // scales[i]
            int sc, m;
            if (gi < 4) {
                sc = scb(gi) & 63;
                m = scb(gi + 4) & 63;
            } else {
                sc = (scb(gi + 4) & 0xF) | ((scb(gi - 4) >> 6) << 4);
                m = (scb(gi + 4) >> 4) | ((scb(gi) >> 6) << 4);
            }
            const float d1 = d * (float) sc, m1 = dmin * (float) m;
#pragma unroll
            for (int j = 0; j < V; ++j) {
                int nib = hi ? (byte(q0, q1, j) >> 4) : (byte(q0, q1, j) & 0xF);
                if constexpr (TYPE == 13) nib += (byte(h0, h1, j) & (1 << (2 * j64 + hi))) ? 16 : 0;
                v[j] = f32_from_f16(f16_from_f32(d1 * (float) nib - m1));
            }
        }
    }
};


}  // namespace rows_i8_src

void rows_i8(const uint16_t* src, bool bf16, int64_t rows, int64_t K, int8_t* q, float* s, const float* rs,
             void* stream) {
    rows_i8_of(rows_i8_src::I8From16{src, bf16, K, rs}, rows, K, q, s, stream);
}

bool rows_i8_blocks(int type, const void* blocks, int64_t rows, int64_t K, int8_t* q, float* s, void* stream) {
    const auto* b = static_cast<const uint8_t*>(blocks);
    switch (type) {
    case 12: rows_i8_of(rows_i8_src::I8FromBlocks<12>{b, K / 256 * 144}, rows, K, q, s, stream); return true;
    case 13: rows_i8_of(rows_i8_src::I8FromBlocks<13>{b, K / 256 * 176}, rows, K, q, s, stream); return true;
    case 14: rows_i8_of(rows_i8_src::I8FromBlocks<14>{b, K / 256 * 210}, rows, K, q, s, stream); return true;
    default: return false;
    }
}

void i8_scale(float* y, int64_t T, int64_t N, int64_t ldy, const float* sx, const float* sw, const float* bias,
                 const float* col, void* stream) {
    if (T <= 0 || N <= 0) return;
    const auto e = core::Runtime::get().stream(stream).parallel_for(sycl::range<2>((size_t) T, (size_t) N), [=](sycl::id<2> i) {
        float& v = y[(int64_t) i[0] * ldy + (int64_t) i[1]];
        // without a DoRA's factors (col) the expression the product had before them, so its rounding stays the same
        const int64_t n = (int64_t) i[1];
        const float s = (float) sycl::bit_cast<int32_t>(v) * sx[i[0]];
        v = col != nullptr ? s * sw[n] * col[n] + (bias != nullptr ? bias[n] : 0.f)
                           : s * sw[n] + (bias != nullptr ? bias[n] : 0.f);
    });
    if (!stream) core::Runtime::get().wait(e, "i8_scale");
}


}  // namespace strata::kernels
