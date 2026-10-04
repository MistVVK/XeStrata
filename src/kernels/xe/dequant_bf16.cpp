// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/dequant_bf16.cpp - the Xe port of Strata's src/kernels/cuda/dequant_bf16.cu: whole-row dequantization of
// the legacy, K-quant and IQ4 block formats (and Q2_0) to BF16, FP16 or FP32, one work-item per 32-element group,
// with the CUDA kernel's expressions.  The i-quant formats go to iq_kernels as in CUDA.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"

#include <cstdint>
#include <cstring>
#include <string>

namespace strata::kernels {
namespace {

inline float h2f(const uint8_t* p) { return f32_from_f16((uint16_t) (p[0] | (p[1] << 8))); }
inline uint16_t f2bf(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    // a NaN (a NaN scale in the block) stays a quiet NaN, as in ggml and `bf16_from_f32` (upstream 6dc9fe8)
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64u);
    u += 0x7fffu + ((u >> 16) & 1u);          // round to nearest even
    return (uint16_t) (u >> 16);
}
inline void put(uint16_t* o, int i, float v) { o[i] = f2bf(v); }
struct H16 { uint16_t v; };
inline void put(H16* o, int i, float v) { o[i].v = f16_from_f32(v); }
inline void put(float* o, int i, float v) { o[i] = v; }

constexpr int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// One 32-element group `g` (row-major over the whole slice); `out` points at that group's 32 outputs.
template <int TYPE, typename T>
inline void group32(const uint8_t* row_blocks, int gi_in_row, int groups_per_row, T* out) {
    if constexpr (TYPE == 42) {                                   // Q2_0: 64 per block of 18 B
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 2) * 18;
        const float d = h2f(b);
        const int e0 = (gi_in_row % 2) * 32;
        for (int j = 0; j < 32; ++j) {
            const int e = e0 + j;
            const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
            put(out, j, (float) (q - 1) * d);
        }
    } else if constexpr (TYPE == 2) {                              // Q4_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, (float) ((b[2 + j] & 0x0F) - 8) * d);
            put(out, j + 16, (float) ((b[2 + j] >> 4) - 8) * d);
        }
    } else if constexpr (TYPE == 6) {                              // Q5_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 22;
        const float d = h2f(b);
        const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) << 4) & 0x10;
            const int xh1 = (qh >> (j + 12)) & 0x10;
            put(out, j, (float) (((b[6 + j] & 0x0F) | xh0) - 16) * d);
            put(out, j + 16, (float) (((b[6 + j] >> 4) | xh1) - 16) * d);
        }
    } else if constexpr (TYPE == 7) {                              // Q5_1 (upstream d652cd6)
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 24;
        const float d = h2f(b), m = h2f(b + 2);
        const uint32_t qh = (uint32_t) b[4] | ((uint32_t) b[5] << 8) | ((uint32_t) b[6] << 16) | ((uint32_t) b[7] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = (int) ((qh >> j) & 1) << 4;
            const int xh1 = (int) ((qh >> (j + 16)) & 1) << 4;
            put(out, j, (float) ((b[8 + j] & 15) | xh0) * d + m);
            put(out, j + 16, (float) ((b[8 + j] >> 4) | xh1) * d + m);
        }
    } else if constexpr (TYPE == 8) {                              // Q8_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 34;
        const float d = h2f(b);
        for (int j = 0; j < 32; ++j) put(out, j, (float) (int8_t) b[2 + j] * d);
    } else if constexpr (TYPE == 20) {                             // IQ4_NL
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, d * (float) kv_iq4nl[b[2 + j] & 0xf]);
            put(out, j + 16, d * (float) kv_iq4nl[b[2 + j] >> 4]);
        }
    } else if constexpr (TYPE == 11) {                             // Q3_K: hmask[32] qs[64] scales[12] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 110;
        const int gi = gi_in_row % 8, n = gi / 4, jj = gi % 4;
        const uint8_t* hm = b;
        const uint8_t* q = b + 32 + n * 32;
        const uint8_t* sc = b + 96;
        const float d_all = h2f(b + 108);
        uint32_t aux[4];
        std::memcpy(aux, sc, 12);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        const int shift = 2 * jj;
        const uint8_t m = (uint8_t) (1u << (n * 4 + jj));
        for (int t = 0; t < 32; ++t) {
            const int is = n * 8 + jj * 2 + (t >= 16 ? 1 : 0);
            const float dl = d_all * (float) (scales[is] - 32);
            put(out, t, dl * (float) ((int) ((q[t] >> shift) & 3) - ((hm[t] & m) ? 0 : 4)));
        }
    } else if constexpr (TYPE == 12) {                             // Q4_K: d dmin scales[12] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* q = b + 16 + 32 * j64;
        for (int l = 0; l < 32; ++l) put(out, l, d1 * (float) (hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
    } else if constexpr (TYPE == 13) {                             // Q5_K: d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48 + 32 * j64;
        const uint8_t u = (uint8_t) (1u << (2 * j64 + hi));
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put(out, l, d1 * (float) (nib + ((qh[l] & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {                             // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put(out, l, d * (float) sc[is + 2 * qu] * (float) (q - 32));
        }
    } else if constexpr (TYPE == kNativeQ6KRows) {                 // Q6_K split by row (native_mmvq.hpp)
        const size_t bpr = (size_t) groups_per_row / 8, blk = (size_t) (gi_in_row / 8);
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint8_t* ql = row_blocks + blk * 128 + (size_t) 64 * n;
        const uint8_t* qh = row_blocks + bpr * 128 + blk * 64 + (size_t) 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(row_blocks + bpr * 192 + blk * 16) + (size_t) 8 * n;
        const float d = h2f(row_blocks + bpr * 208 + blk * 2);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put(out, l, d * (float) sc[is + 2 * qu] * (float) (q - 32));
        }
    } else if constexpr (TYPE == 23) {                             // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            put(out, j, dl * (float) kv_iq4nl[qs[j] & 0xf]);
            put(out, j + 16, dl * (float) kv_iq4nl[qs[j] >> 4]);
        }
    }
}

// A work-item's 32 values go out 16 bytes at a time when `out` is 16-byte aligned (its groups then are): stored one
// value at a time, neighbouring work-items' stores were 64 bytes apart and the RTX 4070 took 300-940 us a matrix.
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
inline uint32_t word(uint16_t v) { return v; }
inline uint32_t word(H16 v) { return v.v; }
inline uint32_t word(float v) { return sycl::bit_cast<uint32_t>(v); }
template <int TYPE, typename T>
sycl::event dequant_kernel(sycl::queue& q, const uint8_t* blocks, int64_t row_bytes, int64_t row0, int64_t rows,
                           int64_t groups_per_row, T* out) {
    if (reinterpret_cast<uintptr_t>(out) % 16 == 0) {
        return q.parallel_for(sycl::range<1>((size_t) (rows * groups_per_row)), [=](sycl::id<1> id) {
            const int64_t g = (int64_t) id[0];
            const int64_t r = g / groups_per_row, gi = g % groups_per_row;
            T v[32];
            group32<TYPE>(blocks + (row0 + r) * row_bytes, (int) gi, (int) groups_per_row, v);
            u32x4* o = reinterpret_cast<u32x4*>(out + r * groups_per_row * 32 + gi * 32);
            if constexpr (sizeof(T) == 4) {
                for (int i = 0; i < 8; ++i)
                    o[i] = u32x4{word(v[4 * i]), word(v[4 * i + 1]), word(v[4 * i + 2]), word(v[4 * i + 3])};
            } else {
                for (int i = 0; i < 4; ++i)
                    o[i] = u32x4{word(v[8 * i]) | word(v[8 * i + 1]) << 16, word(v[8 * i + 2]) | word(v[8 * i + 3]) << 16,
                                 word(v[8 * i + 4]) | word(v[8 * i + 5]) << 16, word(v[8 * i + 6]) | word(v[8 * i + 7]) << 16};
            }
        });
    }
    return q.parallel_for(sycl::range<1>((size_t) (rows * groups_per_row)), [=](sycl::id<1> id) {
        const int64_t g = (int64_t) id[0];
        const int64_t r = g / groups_per_row, gi = g % groups_per_row;
        group32<TYPE>(blocks + (row0 + r) * row_bytes, (int) gi, (int) groups_per_row, out + r * groups_per_row * 32 + gi * 32);
    });
}

bool geometry(int type, int& block_elems, int& block_bytes) {
    switch (type) {
    case 2: block_elems = 32; block_bytes = 18; return true;
    case 6: block_elems = 32; block_bytes = 22; return true;
    case 7: block_elems = 32; block_bytes = 24; return true;
    case 8: block_elems = 32; block_bytes = 34; return true;
    case 20: block_elems = 32; block_bytes = 18; return true;
    case 11: block_elems = 256; block_bytes = 110; return true;
    case 12: block_elems = 256; block_bytes = 144; return true;
    case 13: block_elems = 256; block_bytes = 176; return true;
    case 14: block_elems = 256; block_bytes = 210; return true;
    case kNativeQ6KRows: block_elems = 512; block_bytes = 420; return true;
    case 23: block_elems = 256; block_bytes = 136; return true;
    case 42: block_elems = 64; block_bytes = 18; return true;
    default: return false;
    }
}

template <typename T>
void launch(int type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, T* out, void* stream) {
    int be = 0, bb = 0;
    if (!geometry(type, be, bb) || cols % be != 0 || rows <= 0) {
        throw core::DeviceError("dequant: unsupported type " + std::to_string(type) + " or shape " +
                                std::to_string(rows) + " x " + std::to_string(cols));
    }
    const int64_t row_bytes = cols / be * bb, gpr = cols / 32;
    const uint8_t* p = (const uint8_t*) blocks;
    auto& q = core::Runtime::get().stream(stream);
    sycl::event e;
#define STRATA_DQ(TY) e = dequant_kernel<TY, T>(q, p, row_bytes, row0, rows, gpr, out); break
    switch (type) {
    case 2: STRATA_DQ(2);
    case 6: STRATA_DQ(6);
    case 7: STRATA_DQ(7);
    case 8: STRATA_DQ(8);
    case 11: STRATA_DQ(11);
    case 12: STRATA_DQ(12);
    case 13: STRATA_DQ(13);
    case 14: STRATA_DQ(14);
    case kNativeQ6KRows: STRATA_DQ(kNativeQ6KRows);
    case 20: STRATA_DQ(20);
    case 23: STRATA_DQ(23);
    case 42: STRATA_DQ(42);
    }
#undef STRATA_DQ
    if (!stream) core::Runtime::get().wait(e, "dequant");
}

}  // namespace

bool dequant_bf16_supported(int ggml_type) noexcept {
    int a, b;
    return geometry(ggml_type, a, b);
}

namespace {
// plan v0.3 P6: the i-quant formats (llama.cpp's dequantizers, iq_kernels.cu)
bool iq_only(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }
}  // namespace

void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream) {
    launch<uint16_t>(ggml_type, blocks, row0, rows, cols, out, stream);
}

void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f16(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<H16>(ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out), stream);
}

void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f32(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<float>(ggml_type, blocks, row0, rows, cols, out, stream);
}

}  // namespace strata::kernels
