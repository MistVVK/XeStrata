// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/iq_mmq.cpp - see include/strata/kernels/iq_mmq.hpp.
//
// The weights' values and scales follow llama.cpp's dequantizers (ggml/src/ggml-quants.c, dequantize_row_*; MIT
// license, third_party/main/ggml/LICENSE, quoted in iq_kernels.cpp): a block of 32 values (16 for IQ2_XS, IQ2_S and
// IQ1_M) is its grid or table bytes, signed, times d and the block's scale (IQ1_M's values times 8, its scales over 8).
// Q8_0, Q5_0 and Q6_K (dequantize_row_q8_0, _q5_0, _q6_K) are their bytes, Q5_0's 5-bit values less 16 and Q6_K's
// 6-bit values less 32, times d and the scale of each 16.  Q4_K and Q5_K (dequantize_row_q4_K, _q5_K) are their 4- or
// 5-bit values times d sc, less dmin m: a block's product is d sc times the int8 sum, less dmin m times the
// activations' block sum, which the activations carry beside their scales (q8_1's s).  The activations are rounded as its q8_1 (amax / 127 a block of 32).
//
// A work-group takes a tile of one expert's rows (16, 32 or 64, by the group's largest) and 128 of its outputs, K 128
// values a step: the step's activation rows copied to local memory, the weight rows decoded there (a work-item a block
// of 32), then each sub-group's tiles multiplied a block at a time into int32 accumulators, which are scaled by the
// block's two scales into FP32 ones.  joint_matrix_apply gives a lane its elements but not where they sit in the tile,
// so the scales come as tiles too: loaded with stride 0, a column of one row's scale and a row of one output's, and
// multiplied element by element with the int32 tile (iq_mmq_usable checks that an int32 tile and an FP32 one hold
// their elements in the same order).  Intel's XMX take tiles of 8 x N x 32 on N lanes instead (intel_body), RDNA3 and
// RDNA4 their WMMA instructions.
// The activations' blocks are q8_act.hpp's (iq_mmq_quantize).
//
// In a source file of its own, so its kernels are in a device image apart from the Intel-only ones.
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/nvfp4.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_act.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "device_target.hpp"
#include "amd_wmma.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/per_device.hpp"
#include "strata/platform/env.hpp"
#include "iq_bits.hpp"
#include "cuda_intrinsics.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_iq_mmq
#include "strata/kernels/device_arch_tag.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include "device_caps.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace ix = sycl::ext::intel::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using namespace iq_bits;
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));   // one 16-byte access (sycl::uint4 is split on NVPTX)

// The work-group's tile shape (NVIDIA's int8 one on a warp, which RDNA's WMMA takes too; intel_body splits it into
// Intel's) and the K step.
constexpr int TM = 16, TN = 16, TK = 16, SG = 32;
[[maybe_unused]] constexpr int NE = TM * TN / SG;   // accumulator elements a lane (the layout check)
constexpr int KC = 128;               // K a step: four blocks of 32
constexpr int NB = KC / 32;
constexpr int LDA = KC + 16;          // local row strides in bytes (16-byte multiples, rows apart in the banks)
constexpr int LDB = KC + 16;

// AMD's device compile has joint_matrix on CDNA's matrix cores only, not for this shape: the joint_matrix bodies are
// left out (their types do not depend on a template parameter, so `if constexpr` alone does not drop them).  RDNA3 and
// RDNA4 (GFX11, GFX12) take the same products through their int8 WMMA instructions instead (STRATA_IQ_MMQ_WMMA).
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
#define STRATA_IQ_MMQ_MATRIX 0
#if defined(__GFX11__) || defined(__GFX12__)
#define STRATA_IQ_MMQ_WMMA 1
#else
#define STRATA_IQ_MMQ_WMMA 0
#endif
#else
#define STRATA_IQ_MMQ_MATRIX 1
#define STRATA_IQ_MMQ_WMMA 0
#endif
// Intel's XMX (Xe2, Xe-HPG) take int8 tiles of 8 x N x 32 on N lanes, B in Intel's VNNI layout (the IS bodies): only
// in Intel's device compile, whose run-time check (intel_shape) is the only one that launches them
#if STRATA_IQ_MMQ_MATRIX && !STRATA_DEVICE_NOT_INTEL
#define STRATA_IQ_MMQ_INTEL 1
#else
#define STRATA_IQ_MMQ_INTEL 0
#endif

template<int S>
constexpr bool built() {
#if STRATA_IQ_MMQ_WMMA
    return true;
#elif !STRATA_IQ_MMQ_MATRIX
    return false;
#else
    return STRATA_NV_ARCH == 0 || STRATA_NV_ARCH >= 720;   // int8 on NVIDIA's tensor cores from sm_72 on
#endif
}

// ---------------------------------------------------------------- the weights' blocks of 32 values
// v[j]: values 4j .. 4j + 3 as signed bytes; s0, s1: the scales of values 0-15 and 16-31.
struct Sub {
    uint32_t v[8];
    float s0, s1;
    float m = 0.0f;   // the K-quants' min term: the block's values less m (Q4_K, Q5_K)
};
// the grid bytes (below 128) negated where m is 0xFF
inline uint32_t signed_bytes(uint32_t g, uint32_t m) { return (g ^ m) + (m & 0x01010101u); }

template<int TY> struct Dec;
template<> struct Dec<16> {   // IQ2_XXS: d (0.5 + ls) / 4
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq2_xxs);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq2_xxs* b = reinterpret_cast<const block_iq2_xxs*>(row) + sb / 8;
        const int ib = sb % 8;
        const uint32_t q2 = (uint32_t) get_int_b2(b->qs, 2 * ib), aux = (uint32_t) get_int_b2(b->qs, 2 * ib + 1);
        Sub r;
        for (int k = 0; k < 8; k += 2) {
            const Int2 gp = grid_pair(iq2xxs_grid, byte_of(q2, k / 2));
            const uint32_t s8 = ksigns_byte((uint8_t) (aux >> (7 * k / 2)));
            r.v[k] = signed_bytes((uint32_t) gp.x, sign_mask(s8));
            r.v[k + 1] = signed_bytes((uint32_t) gp.y, sign_mask(s8 >> 4));
        }
        r.s0 = r.s1 = (float) b->d * (float) (2 * (aux >> 28) + 1) * 0.125f;
        return r;
    }
};
template<> struct Dec<18> {   // IQ3_XXS: d (0.5 + ls) / 2
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq3_xxs);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq3_xxs* b = reinterpret_cast<const block_iq3_xxs*>(row) + sb / 8;
        const int ib = sb % 8;
        const U2 pk = get_int2_b2(b->qs, 2 * ib);
        const uint32_t packed[2] = {pk.x, pk.y};
        const uint32_t aux = (uint32_t) get_int_b2(b->qs, QK_K / 16 + ib);
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const uint32_t s8 = ksigns_byte((uint8_t) (aux >> (7 * l / 2)));
            r.v[l] = signed_bytes(iq3xxs_grid[byte_of(packed[l / 4], l % 4)], sign_mask(s8));
            r.v[l + 1] = signed_bytes(iq3xxs_grid[byte_of(packed[l / 4], (l + 1) % 4)], sign_mask(s8 >> 4));
        }
        r.s0 = r.s1 = (float) b->d * (float) (2 * (aux >> 28) + 1) * 0.25f;
        return r;
    }
};
// An i-quant gate/up in the fixed-point sums (canon_math.hpp's fx_*; iq_mmq_grouped's kIqMmqFx + its type, on canonical
// activations): Dec<T>'s values, s0 the 32's integer scale and s1 the 256's d, which the FX bodies take apart; K its
// power of two.  IQ2_XXS (2 ls + 1, K 1/8), IQ3_XXS (2 ls + 1, K 1/4), IQ3_S (1 + 2 ls, K 1).
template<int T> struct FxOf;
template<> struct FxOf<16> {
    using B = block_iq2_xxs;
    static constexpr int log2k() { return -3; }
    static int scale(const B* b, int ib) { return (int) (2 * ((uint32_t) get_int_b2(b->qs, 2 * ib + 1) >> 28) + 1); }
};
template<> struct FxOf<18> {
    using B = block_iq3_xxs;
    static constexpr int log2k() { return -2; }
    static int scale(const B* b, int ib) {
        return (int) (2 * ((uint32_t) get_int_b2(b->qs, QK_K / 16 + ib) >> 28) + 1);
    }
};
template<> struct FxOf<21> {
    using B = block_iq3_s;
    static constexpr int log2k() { return 0; }
    static int scale(const B* b, int ib) { return 1 + 2 * ((b->scales[ib / 2] >> (4 * (ib & 1))) & 0x0F); }
};
template<int T> struct DecFx {
    using F = FxOf<T>;
    static constexpr int QK = 256, BSZ = (int) sizeof(typename F::B);
    static constexpr int group32() { return 8; }   // the blocks of 32 a group (FXG)
    static constexpr int fx_log2() { return F::log2k(); }
    static Sub get(const uint8_t* row, int sb, size_t row_bytes) {
        Sub r = Dec<T>::get(row, sb, row_bytes);
        const auto* b = reinterpret_cast<const typename F::B*>(row) + sb / 8;
        r.s0 = (float) F::scale(b, sb % 8);
        r.s1 = (float) b->d;
        return r;
    }
    // the d of block b of 256 (fx_tops)
    static float block_d(const uint8_t* row, int b) {
        return (float) (reinterpret_cast<const typename F::B*>(row) + b)->d;
    }
};
template<> struct Dec<21> {   // IQ3_S: d (1 + 2 ls)
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq3_s);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq3_s* b = reinterpret_cast<const block_iq3_s*>(row) + sb / 8;
        const int ib = sb % 8;
        const U2 pk = get_int2_b2(b->qs, 2 * ib);
        const uint32_t packed[2] = {pk.x, pk.y};
        const int qh = b->qh[ib];
        const uint32_t signs = (uint32_t) get_int_b2(b->signs, ib);
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const uint32_t s8 = byte_of(signs, l / 2);
            r.v[l] = signed_bytes(iq3s_grid[byte_of(packed[l / 4], l % 4) | ((qh << (8 - l)) & 0x100)], sign_mask(s8));
            r.v[l + 1] = signed_bytes(iq3s_grid[byte_of(packed[l / 4], (l + 1) % 4) | ((qh << (7 - l)) & 0x100)],
                                      sign_mask(s8 >> 4));
        }
        r.s0 = r.s1 = (float) b->d * (float) (1 + 2 * ((b->scales[ib / 2] >> (4 * (ib & 1))) & 0x0F));
        return r;
    }
};
template<> struct Dec<17> {   // IQ2_XS: d (0.5 + ls) / 4, a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq2_xs);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq2_xs* b = reinterpret_cast<const block_iq2_xs*>(row) + sb / 8;
        const int ib = sb % 8;
        const U2 pk = get_int2_b2(b->qs, 2 * ib);
        const uint32_t packed[2] = {pk.x, pk.y};
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const uint16_t q2 = half_of(packed[l / 4], (l / 2) % 2);
            const Int2 gp = grid_pair(iq2xs_grid, q2 & 0x1FF);
            const uint32_t s8 = ksigns_byte((uint8_t) (q2 >> 9));
            r.v[l] = signed_bytes((uint32_t) gp.x, sign_mask(s8));
            r.v[l + 1] = signed_bytes((uint32_t) gp.y, sign_mask(s8 >> 4));
        }
        const float d = (float) b->d;
        r.s0 = d * (float) (2 * (b->scales[ib] & 0x0F) + 1) * 0.125f;
        r.s1 = d * (float) (2 * (b->scales[ib] >> 4) + 1) * 0.125f;
        return r;
    }
};
template<> struct Dec<22> {   // IQ2_S: d (0.5 + ls) / 4, a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq2_s);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq2_s* b = reinterpret_cast<const block_iq2_s*>(row) + sb / 8;
        const int ib = sb % 8;
        const uint32_t qs = (uint32_t) get_int_b2(b->qs, ib);
        const int qh = b->qh[ib];
        const uint32_t signs = (uint32_t) get_int_b2(b->qs, QK_K / 32 + ib);
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const Int2 gp = grid_pair(iq2s_grid, byte_of(qs, l / 2) | ((qh << (8 - l)) & 0x300));
            const uint32_t s8 = byte_of(signs, l / 2);
            r.v[l] = signed_bytes((uint32_t) gp.x, sign_mask(s8));
            r.v[l + 1] = signed_bytes((uint32_t) gp.y, sign_mask(s8 >> 4));
        }
        const float d = (float) b->d;
        r.s0 = d * (float) (2 * (b->scales[ib] & 0x0F) + 1) * 0.125f;
        r.s1 = d * (float) (2 * (b->scales[ib] >> 4) + 1) * 0.125f;
        return r;
    }
};
template<> struct Dec<20> {   // IQ4_NL: d
    static constexpr int QK = 32, BSZ = (int) sizeof(block_iq4_nl);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq4_nl* b = reinterpret_cast<const block_iq4_nl*>(row) + sb;
        Sub r;
        for (int j = 0; j < 4; ++j) {
            const Int2 t = get_int_from_table_16(get_int_b2(b->qs, j), kvalues_iq4nl);
            r.v[j] = (uint32_t) t.x;
            r.v[4 + j] = (uint32_t) t.y;
        }
        r.s0 = r.s1 = (float) b->d;
        return r;
    }
};
template<> struct Dec<23> {   // IQ4_XS: d (ls - 32)
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq4_xs);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq4_xs* b = reinterpret_cast<const block_iq4_xs*>(row) + sb / 8;
        const int ib = sb % 8;
        Sub r;
        for (int j = 0; j < 4; ++j) {
            const Int2 t = get_int_from_table_16(get_int_b4(b->qs, 4 * ib + j), kvalues_iq4nl);
            r.v[j] = (uint32_t) t.x;
            r.v[4 + j] = (uint32_t) t.y;
        }
        const int ls = (((b->scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F) | (((b->scales_h >> (2 * ib)) & 0x03) << 4)) - 32;
        r.s0 = r.s1 = (float) b->d * (float) ls;
        return r;
    }
};
template<> struct Dec<42> {   // Q2_0: d (code - 1), 64 values a block
    static constexpr int QK = 64, BSZ = (int) sizeof(block_q2_0);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q2_0* b = reinterpret_cast<const block_q2_0*>(row) + sb / 2;
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(b->qs) + (ptrdiff_t) (sb % 2) * 4;
        Sub r;
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = qs[j];
            const int k = 2 * j;
            r.v[k] = q2_0_bytes(q & 0xFFu);
            r.v[k + 1] = q2_0_bytes(q >> 8);
        }
        r.s0 = r.s1 = (float) b->d;
        return r;
    }
};

template<> struct Dec<19> {   // IQ1_S: d (2 ls + 1) (grid +- 1/8), one scale and one delta a 32 (dequantize_row_iq1_s)
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq1_s);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq1_s* b = reinterpret_cast<const block_iq1_s*>(row) + sb / 8;
        const int ib = sb % 8;
        const int qh = b->qh[ib];
        // as IQ1_M's: the grid's nibbles (the value + 1) as 8 (value +- 1/8), the delta's sign from bit 15
        const uint32_t c = qh & 0x8000 ? 0x07070707u : 0x09090909u;
        Sub r;
        for (int k = 0; k < 8; k += 2) {
            const int l = k / 2;
            const uint32_t g = iq1s_grid_gpu[b->qs[4 * ib + l] | (((qh >> 3 * l) & 0x07) << 8)];
            r.v[k] = (((((g & 0x0F0F0F0Fu) << 3) + c) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
            r.v[k + 1] = ((((((g >> 4) & 0x0F0F0F0Fu) << 3) + c) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
        }
        r.s0 = r.s1 = (float) b->d * 0.125f * (float) (2 * ((qh >> 12) & 0x07) + 1);
        return r;
    }
};

template<> struct Dec<29> {   // IQ1_M: d (2 ls + 1) (grid +- 1/8), a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq1_m);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq1_m* b = reinterpret_cast<const block_iq1_m*>(row) + sb / 8;
        const int ib = sb % 8;
        Sub r;
        for (int k = 0; k < 8; k += 2) {
            const int l = k / 2;
            const int qh = b->qh[2 * ib + l / 2] >> (4 * (l % 2));
            const uint32_t g = iq1s_grid_gpu[b->qs[4 * ib + l] | ((qh & 0x07) << 8)];
            // the grid's nibbles (0 .. 2, the value + 1) as 8 (value +- 1/8): 8 nibble + 9 or 7, then 16 off each
            // byte without a borrow crossing bytes (as q2_0_bytes)
            const uint32_t c = qh & 0x08 ? 0x07070707u : 0x09090909u;
            r.v[k] = (((((g & 0x0F0F0F0Fu) << 3) + c) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
            r.v[k + 1] = ((((((g >> 4) & 0x0F0F0F0Fu) << 3) + c) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
        }
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(b->scales);
        const float d = iq1m_scale(sc) * 0.125f;
        const int ls = sc[ib / 2] >> (6 * (ib % 2));
        r.s0 = d * (float) (2 * (ls & 0x07) + 1);
        r.s1 = d * (float) (2 * ((ls >> 3) & 0x07) + 1);
        return r;
    }
};

template<> struct Dec<8> {   // Q8_0: d
    static constexpr int QK = 32, BSZ = (int) sizeof(block_q8_0);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q8_0* b = reinterpret_cast<const block_q8_0*>(row) + sb;
        Sub r;
        for (int j = 0; j < 8; ++j) r.v[j] = (uint32_t) get_int_b2(b->qs, j);
        r.s0 = r.s1 = (float) b->d;
        return r;
    }
};
template<> struct Dec<2> {   // Q4_0: d (code - 8), element j in qs[j]'s low nibble, j+16 in its high (the latent D/U)
    static constexpr int QK = 32, BSZ = (int) sizeof(block_q4_0);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q4_0* b = reinterpret_cast<const block_q4_0*>(row) + sb;
        Sub r;
        for (int j = 0; j < 4; ++j) {
            const uint32_t w = (uint32_t) get_int_b2(b->qs, j);
            const uint32_t lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;
            // each nibble 0..15 -> the signed byte (code - 8): flip the sign bit, then sign-extend it into the byte
            const uint32_t cl = lo ^ 0x08080808u, ch = hi ^ 0x08080808u;
            r.v[j] = cl | ((cl & 0x08080808u) * 0x1Eu);
            r.v[4 + j] = ch | ((ch & 0x08080808u) * 0x1Eu);
        }
        r.s0 = r.s1 = (float) b->d;
        return r;
    }
};
template<> struct Dec<14> {   // Q6_K: d scale (6-bit value - 32), a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_q6_K);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q6_K* b = reinterpret_cast<const block_q6_K*>(row) + sb / 8;
        // block of 32 ib: half n of the 256 (ql, qh and the scales 64, 32 and 8 bytes on), quarter j of the half: the
        // low nibbles (j 0, 1) or high ones (j 2, 3) of ql's first or second 32 bytes, and qh's 2 bits at 2 j
        const int ib = sb % 8, n = ib / 4, j = ib % 4;
        Sub r;
        for (int k = 0; k < 8; ++k) {
            const uint32_t ql = (uint32_t) get_int_b2(b->ql, 16 * n + 8 * (j & 1) + k);
            const uint32_t qh = (uint32_t) get_int_b2(b->qh, 8 * n + k);
            const uint32_t v = ((ql >> (4 * (j >> 1))) & 0x0F0F0F0Fu) | (((qh >> (2 * j)) & 0x03030303u) << 4);
            r.v[k] = ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;   // less 32 in each byte, no borrow across
        }
        const float d = (float) b->d;
        r.s0 = d * (float) b->scales[8 * n + 2 * j];
        r.s1 = d * (float) b->scales[8 * n + 2 * j + 1];
        return r;
    }
};

template<> struct Dec<6> {   // Q5_0: d ((5-bit value) - 16), the high bits in qh
    static constexpr int QK = 32, BSZ = (int) sizeof(block_q5_0);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q5_0* b = reinterpret_cast<const block_q5_0*>(row) + sb;
        const uint32_t qh = (uint32_t) get_int_b2(b->qh, 0);
        Sub r;
        for (int k = 0; k < 4; ++k) {   // values 4k .. 4k + 3 in the low nibbles, 16 + 4k .. in the high ones
            const uint32_t ql = (uint32_t) get_int_b2(b->qs, k);
            uint32_t h0 = 0, h1 = 0;
            for (int t = 0; t < 4; ++t) {
                h0 |= ((qh >> (4 * k + t)) & 1u) << (8 * t + 4);
                h1 |= ((qh >> (4 * k + t + 16)) & 1u) << (8 * t + 4);
            }
            r.v[k] = ((((ql & 0x0F0F0F0Fu) | h0) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
            r.v[4 + k] = (((((ql >> 4) & 0x0F0F0F0Fu) | h1) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
        }
        r.s0 = r.s1 = (float) b->d;
        return r;
    }
};
template<> struct Dec<7> {   // Q5_1: d (5-bit value) + m: Q5_0's bits without its offset, and a min
    static constexpr int QK = 32, BSZ = (int) sizeof(block_q5_1);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q5_1* b = reinterpret_cast<const block_q5_1*>(row) + sb;
        const uint32_t qh = (uint32_t) get_int_b2(b->qh, 0);
        Sub r;
        for (int k = 0; k < 4; ++k) {   // values 4k .. 4k + 3 in the low nibbles, 16 + 4k .. in the high ones
            const uint32_t ql = (uint32_t) get_int_b2(b->qs, k);
            uint32_t h0 = 0, h1 = 0;
            for (int t = 0; t < 4; ++t) {
                h0 |= ((qh >> (4 * k + t)) & 1u) << (8 * t + 4);
                h1 |= ((qh >> (4 * k + t + 16)) & 1u) << (8 * t + 4);
            }
            r.v[k] = (ql & 0x0F0F0F0Fu) | h0;
            r.v[4 + k] = ((ql >> 4) & 0x0F0F0F0Fu) | h1;
        }
        r.s0 = r.s1 = (float) b->dm[0];
        r.m = -(float) b->dm[1];   // the products take the block's values less m
        return r;
    }
};
// Q4_K and Q5_K's 6-bit scale and min of block j (get_scale_min_k4)
inline void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
template<> struct Dec<12> {   // Q4_K: d sc value - dmin m, block ib the low (even ib) or high nibbles of qs[32 (ib / 2)..]
    static constexpr int QK = 256, BSZ = (int) sizeof(block_q4_K);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q4_K* b = reinterpret_cast<const block_q4_K*>(row) + sb / 8;
        const int ib = sb % 8;
        Sub r;
        for (int k = 0; k < 8; ++k)
            r.v[k] = ((uint32_t) get_int_b4(b->qs, 8 * (ib / 2) + k) >> (4 * (ib & 1))) & 0x0F0F0F0Fu;
        int sc, mn;
        scale_min_k4(ib, b->scales, sc, mn);
        r.s0 = r.s1 = (float) b->dm[0] * (float) sc;
        r.m = (float) b->dm[1] * (float) mn;
        return r;
    }
};
template<> struct Dec<13> {   // Q5_K: Q4_K's nibbles with bit ib of qh as the fifth
    static constexpr int QK = 256, BSZ = (int) sizeof(block_q5_K);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q5_K* b = reinterpret_cast<const block_q5_K*>(row) + sb / 8;
        const int ib = sb % 8;
        Sub r;
        for (int k = 0; k < 8; ++k) {
            const uint32_t ql = (uint32_t) get_int_b4(b->qs, 8 * (ib / 2) + k) >> (4 * (ib & 1));
            const uint32_t qh = (uint32_t) get_int_b4(b->qh, k) >> ib;
            r.v[k] = (ql & 0x0F0F0F0Fu) | ((qh & 0x01010101u) << 4);
        }
        int sc, mn;
        scale_min_k4(ib, b->scales, sc, mn);
        r.s0 = r.s1 = (float) b->dm[0] * (float) sc;
        r.m = (float) b->dm[1] * (float) mn;
        return r;
    }
};

template<> struct Dec<11> {   // Q3_K: d (6-bit scale - 32) ((2-bit value | high bit << 2) - 4), a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_q3_K);
    // scale `is` of 16 (dequantize_row_q3_K's aux unpacking): its low nibble from bytes 0-7, its high bits from 8-11
    static int scale(const uint8_t* s, int is) {
        const int lo = is < 8 ? s[is] & 0x0F : s[is - 8] >> 4;
        const int hi = (s[8 + (is & 3)] >> (2 * (is >> 2))) & 0x03;
        return (lo | (hi << 4)) - 32;
    }
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q3_K* b = reinterpret_cast<const block_q3_K*>(row) + sb / 8;
        const int ib = sb % 8, n = ib / 4, j = ib % 4;   // 128-value half n, its 2-bit field j
        Sub r;
        for (int k = 0; k < 8; ++k) {   // values 0-15 from qs[32n..] and hmask[0..], 16-31 from qs[32n + 16..]
            const int l = k < 4 ? 4 * k : 16 + 4 * (k - 4);
            const uint32_t q = ((uint32_t) get_int_b2(b->qs + (size_t) 32 * n, l / 4) >> (2 * j)) & 0x03030303u;
            const uint32_t h = ((uint32_t) get_int_b2(b->hmask, l / 4) >> (4 * n + j)) & 0x01010101u;
            r.v[k] = (((q | (h << 2)) | 0x80808080u) - 0x04040404u) ^ 0x80808080u;
        }
        const float d = (float) b->d;
        r.s0 = d * (float) scale(b->scales, 8 * n + 2 * j);
        r.s1 = d * (float) scale(b->scales, 8 * n + 2 * j + 1);
        return r;
    }
};
// NVFP4 (nvfp4.hpp): 32 values are two 16-value sub-blocks of one block of 64, each with its UE4M3 scale (halved: the
// table is the E2M1 values doubled).  Sub-block s's ints 2 s and 2 s + 1 hold its values 0-3 | 8-11 and 4-7 | 12-15
// in their low | high nibbles.
template<> struct Dec<40> {
    static constexpr int QK = 64, BSZ = (int) sizeof(block_nvfp4);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_nvfp4* b = reinterpret_cast<const block_nvfp4*>(row) + sb / 2;
        const int s0 = 2 * (sb % 2);   // the first of the two sub-blocks
        Sub r;
        for (int k = 0; k < 2; ++k) {
            const Int2 a = get_int_from_table_16(get_int_b4(b->qs, 2 * (s0 + k)), kvalues_fp4);
            const Int2 c = get_int_from_table_16(get_int_b4(b->qs, 2 * (s0 + k) + 1), kvalues_fp4);
            r.v[4 * k + 0] = (uint32_t) a.x;
            r.v[4 * k + 1] = (uint32_t) c.x;
            r.v[4 * k + 2] = (uint32_t) a.y;
            r.v[4 * k + 3] = (uint32_t) c.y;
        }
        r.s0 = nvfp4::ue4m3_half(b->d[s0]);
        r.s1 = nvfp4::ue4m3_half(b->d[s0 + 1]);
        return r;
    }
};
// Q2_K: d sc value - dmin m, sc and m a pair per 16 values.  The products have one min a block of 32, so a Q2_K row
// is read as twice its length: block 2 i is its sub-block i's values (scales d sc), block 2 i + 1 ones scaled by
// -dmin m, against activations whose blocks come twice each (iq_mmq_quantize's dup).  QK: 256 values per 512 read.
template<> struct Dec<10> {
    static constexpr int QK = 512, BSZ = (int) sizeof(block_q2_K);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const int rb = sb / 2;
        const block_q2_K* b = reinterpret_cast<const block_q2_K*>(row) + rb / 8;
        const int ib = rb % 8, n = ib / 4, j = ib % 4;
        const int sc0 = b->scales[8 * n + 2 * j], sc1 = b->scales[8 * n + 2 * j + 1];
        Sub r;
        if (sb % 2 == 0) {
            for (int k = 0; k < 8; ++k) {
                const int l = k < 4 ? 4 * k : 16 + 4 * (k - 4);
                r.v[k] = ((uint32_t) get_int_b4(b->qs + (size_t) 32 * n, l / 4) >> (2 * j)) & 0x03030303u;
            }
            const float d = (float) b->dm[0];
            r.s0 = d * (float) (sc0 & 0x0F);
            r.s1 = d * (float) (sc1 & 0x0F);
        } else {
            for (int k = 0; k < 8; ++k) r.v[k] = 0x01010101u;
            const float dmin = (float) b->dm[1];
            r.s0 = -dmin * (float) (sc0 >> 4);
            r.s1 = -dmin * (float) (sc1 >> 4);
        }
        return r;
    }
};

template<> struct Dec<kNativeQ6KRows> {   // Q6_K split by row (native_mmvq.hpp): a row's ql, then qh, scales and d
    static constexpr int QK = 256, BSZ = (int) sizeof(block_q6_K);
    static Sub get(const uint8_t* row, int sb, size_t row_bytes) {
        const size_t bpr = row_bytes / BSZ, blk = (size_t) sb / 8;
        const int ib = sb % 8, n = ib / 4, j = ib % 4;
        const uint8_t* ql = row + blk * 128 + (size_t) 64 * n;
        const uint8_t* qh = row + bpr * 128 + blk * 64 + (size_t) 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(row + bpr * 192 + blk * 16) + (size_t) 8 * n;
        Sub r;
        for (int k = 0; k < 8; ++k) {
            const uint32_t l = (uint32_t) get_int_b2(ql + 32 * (ptrdiff_t) (j & 1), k), h = (uint32_t) get_int_b2(qh, k);
            const uint32_t v = ((l >> (4 * (j >> 1))) & 0x0F0F0F0Fu) | (((h >> (2 * j)) & 0x03030303u) << 4);
            r.v[k] = ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
        }
        const float d = (float) *reinterpret_cast<const sycl::half*>(row + bpr * 208 + blk * 2);
        r.s0 = d * (float) sc[2 * (ptrdiff_t) j];
        r.s1 = d * (float) sc[2 * j + 1];
        return r;
    }
};

template<> struct Dec<kNativeQ8Rows> {   // Q8_0 split by row (native_mmvq.hpp): a row's qs, then its d
    static constexpr int QK = 32, BSZ = (int) sizeof(block_q8_0);
    static Sub get(const uint8_t* row, int sb, size_t row_bytes) {
        const size_t bpr = row_bytes / BSZ;
        const uint8_t* qs = row + (size_t) sb * 32;
        Sub r;
        for (int j = 0; j < 8; ++j) r.v[j] = (uint32_t) get_int_b4(qs, j);
        r.s0 = r.s1 = (float) *reinterpret_cast<const sycl::half*>(row + bpr * 32 + (size_t) sb * 2);
        return r;
    }
};

template<> struct Dec<kIqMmqFx + 16> : DecFx<16> {};
template<> struct Dec<kIqMmqFx + 18> : DecFx<18> {};
template<> struct Dec<kIqMmqFx + 21> : DecFx<21> {};
// IQ2_XS and IQ2_S (a scale for each 16): Dec<T>'s values, s0 and s1 the halves' integer scales 2 ls + 1, m the 256's d
// (K 1/8)
template<int T, class B> struct DecFxHalf {
    static constexpr int QK = 256, BSZ = (int) sizeof(B);
    static constexpr int group32() { return 8; }   // the blocks of 32 a group (FXG)
    static constexpr int fx_log2() { return -3; }
    static Sub get(const uint8_t* row, int sb, size_t row_bytes) {
        Sub r = Dec<T>::get(row, sb, row_bytes);
        const B* b = reinterpret_cast<const B*>(row) + sb / 8;
        r.s0 = (float) (2 * (b->scales[sb % 8] & 0x0F) + 1);
        r.s1 = (float) (2 * (b->scales[sb % 8] >> 4) + 1);
        r.m = (float) b->d;
        return r;
    }
    static float block_d(const uint8_t* row, int b) { return (float) (reinterpret_cast<const B*>(row) + b)->d; }
};
template<> struct Dec<kIqMmqFx + 17> : DecFxHalf<17, block_iq2_xs> {};
template<> struct Dec<kIqMmqFx + 22> : DecFxHalf<22, block_iq2_s> {};
// IQ4_NL (a down projection's, on activations with a scale for each 32): Dec<20>'s values, s0 1, s1 the block's d
template<> struct Dec<kIqMmqFx + 20> {
    static constexpr int QK = 32, BSZ = (int) sizeof(block_iq4_nl);
    static constexpr int group32() { return 1; }   // the blocks of 32 a group (FXG)
    static constexpr int fx_log2() { return 0; }
    static Sub get(const uint8_t* row, int sb, size_t row_bytes) {
        Sub r = Dec<20>::get(row, sb, row_bytes);
        r.s0 = 1.0f;
        r.s1 = (float) (reinterpret_cast<const block_iq4_nl*>(row) + sb)->d;
        return r;
    }
    static float block_d(const uint8_t* row, int b) {
        return (float) (reinterpret_cast<const block_iq4_nl*>(row) + b)->d;
    }
};

// Q2_0 (gate/up on a scale for each 256, down on one for each 32): Dec<42>'s values (code - 1), s0 1, s1 the 64's d; a
// group of one block of 32, whichever its activation
template<> struct Dec<kIqMmqFx + 42> {
    static constexpr int QK = 64, BSZ = (int) sizeof(block_q2_0);
    static constexpr int group32() { return 1; }   // the blocks of 32 a group (FXG)
    static constexpr int fx_log2() { return 0; }
    static Sub get(const uint8_t* row, int sb, size_t row_bytes) {
        Sub r = Dec<42>::get(row, sb, row_bytes);
        r.s0 = 1.0f;
        return r;
    }
    static float block_d(const uint8_t* row, int b) {
        return (float) (reinterpret_cast<const block_q2_0*>(row) + b)->d;
    }
};

template<int TY>
constexpr bool half_scales() {
    return TY == 17 || TY == 22 || TY == 29 || TY == 14 || TY == kNativeQ6KRows || TY == 10 || TY == 11 || TY == 40 ||
           TY == kIqMmqFx + 17 || TY == kIqMmqFx + 22;
}
template<int TY>
constexpr bool has_min() { return TY == 7 || TY == 12 || TY == 13; }
template<int TY>
constexpr bool fx_form() {
    return TY == kIqMmqFx + 16 || TY == kIqMmqFx + 17 || TY == kIqMmqFx + 18 || TY == kIqMmqFx + 20 ||
           TY == kIqMmqFx + 21 || TY == kIqMmqFx + 22 || TY == kIqMmqFx + 42;
}


// ---------------------------------------------------------------- the products
template<typename T>
inline auto local_as(sycl::multi_ptr<uint8_t, sycl::access::address_space::local_space, sycl::access::decorated::no> p) {
    using V = sycl::multi_ptr<void, sycl::access::address_space::local_space, sycl::access::decorated::no>;
    return static_cast<sycl::multi_ptr<T, sycl::access::address_space::local_space, sycl::access::decorated::no>>(V(p));
}
constexpr int up128(int b) { return (b + 127) / 128 * 128; }

// An int32 sum below 2^22 in magnitude (a block's: 32 products of int8 values) as a float, exactly: its bits added to
// those of 1.5 * 2^23, less that.  Two full-rate instructions instead of a conversion (a quarter of the rate on the
// RTX 4070, where it was a top stall).
[[maybe_unused]] inline float exact_float(int32_t c) { return sycl::bit_cast<float>(c + 0x4B400000) - 12582912.0f; }

// SGM x 2 tiles a sub-group, NSGM x 4 sub-groups a work-group: WM rows by 128 outputs.  Local memory, one buffer
// carved at 128-byte boundaries (the matrix loads take 32-byte aligned tiles): the step's activations (WM x KC int8,
// row stride LDA) and weights (WN x KC, row stride LDB: B column-major), the activations' scales as [block][row] and
// the weights' as [block][half][output] (so that a tile of either loads with stride 0: the same scale down a column or
// along a row), and a tile a sub-group on its way out.  DP4A: the same tiles and steps without matrix engines (the
// GPUs iq_mmq_usable says no for: RDNA2, Volta, Intel's without XMX), each work-item a column of outputs down
// WM / (lanes / WN) of the rows, the products by dp4a.  IS: Intel's XMX shapes (1: Xe2's 8 x 16 x 32 on 16 lanes, 2:
// Xe-HPG's 8 x 8 x 32 on 8), the same work-group and steps (intel_body); a type with a scale per 16 values keeps two
// copies of the activations, each with the other half of every block zeroed.
//
// REG (with IS): the sub-group form (intel_reg_body), a sub-group a column strip of ISG outputs down all WM rows,
// each lane decoding its own output's blocks: no work-group barrier, each sub-group reading the block's activations
// on its own (intel_sub_group_first chooses between the two forms).  The idea is arcint's
// (github.com/marfrit/arcint, Apache-2.0: its int8 expert products on the Arc B60 lost 46-59% with the decoded weights
// shared through local memory, and kept them in each lane's registers).
template<int TY, int SGM, int NSGM, bool DP4A = false, bool TAIL = false, int IS = 0, bool REG = false>
struct Kernel {
    static constexpr int SGN = 2, NSGN = 4, NSG = NSGM * NSGN;
    static constexpr int WM = NSGM * SGM * TM, WN = NSGN * SGN * TN;
    static constexpr bool HALF = half_scales<TY>();
    static constexpr bool MIN = has_min<TY>();
    // FX: the fixed-point sums (canon_math.hpp's fx_*): a group's FXG blocks' int32 sums times their integer scale s0
    // (exact), then that sum times the group's fx_scale added in int64 (any order: the CPU's canonical rows give the
    // same bits); the activations canonical (iq_mmq_quantize's canon: one fp16 scale a group), K whole
    static constexpr bool FX = fx_form<TY>();
    // the per-row / per-output state's type: the experts' canon fx exponent tops (int)
    using Top = int;
    // FX: the blocks of 32 that share one activation scale and one d (8: a 256's, gate/up; 1: IQ4_NL's, Q2_0's)
    static constexpr int FXG = [] {
        if constexpr (FX) return Dec<TY>::group32();
        else return 1;
    }();
    static_assert(!FX || !MIN, "the fixed-point sums take no min");
    // FX with a scale for each 16 (HALF): the halves' integer scales in the weights' two scale slots, the group's d in
    // the min's (wml, Sub::m)
    static constexpr bool FXH = FX && HALF;
    static constexpr int ISG = IS == 1 ? 16 : IS == 2 ? 8 : SG;   // the sub-group size, and Intel's tile width
    static constexpr int LANES = REG ? WN : NSG * SG;              // the work-group: REG, WN / ISG sub-groups
    static constexpr int A_N = IS != 0 && HALF ? 2 : 1;           // copies of the activations' tile
    // MIN: the activations' block sums as [block][row] and the weights' min terms as [block][output] after the rest
    static constexpr int A_OFF = 0, B_OFF = up128(A_OFF + A_N * WM * LDA), XS_OFF = up128(B_OFF + WN * LDB),
                         WS_OFF = up128(XS_OFF + NB * WM * 4), ST_OFF = up128(WS_OFF + 2 * NB * WN * 4),
                         XU_OFF = up128(ST_OFF + NSG * TM * TN * 4), WM_OFF = up128(XU_OFF + NB * WM * 4),
                         // REG: a sub-group's B tiles, and on 8 lanes its A tiles
                         SG_LOCAL = (HALF ? 2 : 1) * 32 * ISG + (IS == 1 ? 0 : WM * 32),
                         // FX: the rows' and the outputs' tops (fx_tops) past what the body uses (FXH: the d's
                         // in the min's slot)
                         FX_OFF = FXH ? up128(WM_OFF + NB * WN * 4) : XU_OFF,
                         LOCAL = REG ? WN / ISG * SG_LOCAL : MIN ? up128(WM_OFF + NB * WN * 4)
                                 : FX ? up128(FX_OFF + (WM + WN) * 4) : XU_OFF;
    const uint8_t* w;
    const uint8_t* const* wtab;   // expert e's weights at wtab[e] + woff when set, else at w + e expert_bytes
    size_t woff, expert_bytes, row_bytes;
    const int8_t* xq;
    const float* xs;
    const float* xu;   // the activations' block sums (d times the int8 values' sum), MIN
    int64_t K;
    const int32_t* bounds;
    const int32_t* ids;
    float* dst;
    int64_t ld_dst, tiles_m, tiles_n;
    sycl::local_accessor<uint8_t, 1> lm;
    // K in ksplit parts of kspan values (launch's), a work-group each, each part's sums into part[ks][row][col] (n_out
    // columns), which launch adds up in order; part null: one part, the sums straight into dst
    int64_t ksplit = 1, kspan = 0;
    float* part = nullptr;
    int64_t total_rows = 0, n_out = 0;

    void out(int64_t ks, int64_t row, int64_t col, float v) const {
        if (part == nullptr) dst[(int64_t) ids[row] * ld_dst + col] = v;
        else part[(ks * total_rows + row) * n_out + col] = v;
    }

    // FX: the tile's rows' tops (the largest exponent of their scales, one each FXG blocks) into ta, its outputs' (of
    // their blocks' d) into tw, over the whole of K: canon_math.hpp's top of row and output is ta + tw.  The step
    // loop's first barrier publishes them.
    void fx_tops(int lid, int lanes, int64_t row0, int64_t rows, int64_t m0, int64_t n0, const uint8_t* wx, int* ta,
                 int* tw) const {
        const int64_t kb = K / 32;
        for (int rr = lid; rr < WM; rr += lanes) {
            int t = canon::kFxNone;
            if (m0 + rr < rows)
                for (int64_t b = 0; b < kb; b += FXG) t = sycl::max(t, canon::fx_exp(xs[(row0 + m0 + rr) * kb + b]));
            ta[rr] = m0 + rr < rows ? canon::fx_row_top(canon::fx_exp(xs[(row0 + m0 + rr) * kb]), t) : t;
        }
        if (tw == nullptr) return;
        for (int nn = lid; nn < WN; nn += lanes) tw[nn] = fx_wtop(wx + (size_t) (n0 + nn) * row_bytes);
    }
    // FX: an output's top (of its blocks' d) over the whole of K, its weights' row at wr
    Top fx_wtop(const uint8_t* wr) const {
        int t = canon::kFxNone;
        for (int64_t b = 0; b < K / Dec<TY>::QK; ++b) t = sycl::max(t, canon::fx_exp(Dec<TY>::block_d(wr, (int) b)));
        return canon::fx_row_top(canon::fx_exp(Dec<TY>::block_d(wr, 0)), t);
    }
    // FX: a group's integer scale from its activation scale and its d at a row and output's top
    static int32_t fx_a(float xsv, float wd, int top) {
        return canon::fx_scale(canon::fx_part(xsv), canon::fx_part(wd), top);
    }
    // FX: block b's group d of output column `col` in the tile (FXH: the min's slot)
    static float fx_wd(const float* wsl, const float* wml, int b, int col) {
        if constexpr (FXH) return wml[b * WN + col];
        else return wsl[(2 * b + 1) * WN + col];
    }
    static float fx_out(int64_t sum, int top) {
        if constexpr (FX) return canon::fx_value(sum, top, Dec<TY>::fx_log2());
        else return 0.0f;
    }

    void dp4a_body(sycl::nd_item<1> it, int64_t ks) const {
        constexpr int lanes = NSG * SG, RSTEP = lanes / WN, RPT = WM / RSTEP;
        const int64_t gk = (int64_t) it.get_group(0), g = gk / ksplit, per = tiles_m * tiles_n, e = g / per,
                      r = g % per;
        const int64_t m0 = r % tiles_m * WM, n0 = r / tiles_m * WN;
        const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
        if (m0 >= rows) return;   // the whole work-group: this expert has fewer rows than the launch
        const int lid = (int) it.get_local_id(0), n = lid % WN, rbase = lid / WN;
        const uint8_t* wx = wtab != nullptr ? wtab[e] + woff : w + (size_t) e * expert_bytes;
        const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
        int8_t* al = local_as<int8_t>(base + A_OFF).get();
        int8_t* bl = local_as<int8_t>(base + B_OFF).get();
        float* xsl = local_as<float>(base + XS_OFF).get();
        [[maybe_unused]] float* xul = local_as<float>(base + XU_OFF).get();
        [[maybe_unused]] float* wml = local_as<float>(base + WM_OFF).get();
        float* wsl = local_as<float>(base + WS_OFF).get();
        float acc[RPT];
        for (int i = 0; i < RPT; ++i) acc[i] = 0.0f;
        [[maybe_unused]] int32_t isum[FX ? RPT : 1];
        [[maybe_unused]] int64_t acc64[FX ? RPT : 1];
        [[maybe_unused]] Top* ta = local_as<Top>(base + FX_OFF).get();
        // FX: this work-item's output's top in a register, not through local memory: in the B70's DP4a kernel (IGC
        // 2.41.16) every output's top read there differed from the same computed again from global memory, while
        // the rows' tops and the scales there were right, and every fixed-point check of iq_mmq_parity failed with
        // XESTRATA_NO_XMX=1 (2026-11-16; the RTX 4070's and the RX 9060 XT's DP4a and the B70's XMX were right).
        // Its own column's: K / QK d's
        [[maybe_unused]] Top twn = (Top) canon::kFxNone;
        if constexpr (FX) {
            for (int i = 0; i < RPT; ++i) {
                isum[i] = 0;
                acc64[i] = 0;
            }
            fx_tops(lid, lanes, row0, rows, m0, n0, wx, ta, nullptr);
            twn = fx_wtop(wx + (size_t) (n0 + n) * row_bytes);
        }
        const int64_t kb = K / 32;   // activation scales a row
        for (int64_t k0 = ks * kspan; k0 < sycl::min(K, (ks + 1) * kspan); k0 += KC) {
            sycl::group_barrier(it.get_group());   // the last step's tiles are read
            const bool last = TAIL && k0 + KC > K;   // TAIL: a last step past K, zeros past it
            for (int v = lid; v < WM * (KC / 16); v += lanes) {
                const int rr = v / (KC / 16), c = v % (KC / 16) * 16;
                const int64_t row = m0 + rr;
                u32x4 val = {0u, 0u, 0u, 0u};
                if (row < rows && (!last || k0 + c < K)) val = *reinterpret_cast<const u32x4*>(xq + (row0 + row) * K + k0 + c);
                *reinterpret_cast<u32x4*>(al + static_cast<ptrdiff_t>(rr) * LDA + c) = val;
            }
            for (int v = lid; v < WM * NB; v += lanes) {
                const int b = v / WM, rr = v % WM;
                const int64_t row = m0 + rr;
                xsl[v] = row < rows && (!last || k0 / 32 + b < kb) ? xs[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                if constexpr (MIN) xul[v] = row < rows && (!last || k0 / 32 + b < kb) ? xu[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
            }
            for (int v = lid; v < WN * NB; v += lanes) {
                const int nn = v / NB, b = v % NB;
                const Sub d = !last || k0 / 32 + b < kb ? Dec<TY>::get(wx + (size_t) (n0 + nn) * row_bytes, (int) (k0 / 32) + b, row_bytes)
                                                : Sub{{}, 0.0f, 0.0f, 0.0f};   // past K: zeros
                u32x4* o = reinterpret_cast<u32x4*>(bl + static_cast<ptrdiff_t>(nn) * LDB + static_cast<ptrdiff_t>(b) * 32);
                o[0] = u32x4{d.v[0], d.v[1], d.v[2], d.v[3]};
                o[1] = u32x4{d.v[4], d.v[5], d.v[6], d.v[7]};
                wsl[(2 * b) * WN + nn] = d.s0;
                wsl[(2 * b + 1) * WN + nn] = d.s1;
                if constexpr (MIN || FXH) wml[b * WN + nn] = d.m;
            }
            sycl::group_barrier(it.get_group());
            for (int b = 0; b < NB; ++b) {
                int wv[8];   // this work-item's output's block, read once for its rows
                const int* wp = reinterpret_cast<const int*>(bl + static_cast<ptrdiff_t>(n) * LDB + static_cast<ptrdiff_t>(b) * 32);
                for (int i = 0; i < 8; ++i) wv[i] = wp[i];
                const float ws0 = wsl[(2 * b) * WN + n], ws1 = wsl[(2 * b + 1) * WN + n];
                for (int i = 0; i < RPT; ++i) {
                    const int rr = rbase + i * RSTEP;
                    const int* ap = reinterpret_cast<const int*>(al + static_cast<ptrdiff_t>(rr) * LDA + static_cast<ptrdiff_t>(b) * 32);
                    int s0 = 0, s1 = 0;
                    for (int j = 0; j < 4; ++j) s0 = xe::dp4a(ap[j], wv[j], s0);
                    for (int j = 4; j < 8; ++j) s1 = xe::dp4a(ap[j], wv[j], s1);
                    if constexpr (FX) {   // ws0 (ws1 FXH: the second 16's) the integer scale
                        isum[i] += FXH ? (int32_t) ws0 * s0 + (int32_t) ws1 * s1 : (int32_t) ws0 * (s0 + s1);
                        if ((k0 / 32 + b) % FXG == FXG - 1) {
                            acc64[i] += (int64_t) fx_a(xsl[b * WM + rr], fx_wd(wsl, wml, b, n), ta[rr] + twn) *
                                            isum[i];
                            isum[i] = 0;
                        }
                        continue;
                    }
                    acc[i] += xsl[b * WM + rr] * (ws0 * (float) s0 + ws1 * (float) s1);
                    if constexpr (MIN) acc[i] -= wml[b * WN + n] * xul[b * WM + rr];
                }
            }
        }
        for (int i = 0; i < RPT; ++i) {
            const int rr = rbase + i * RSTEP;
            const int64_t row = m0 + rr;
            float v = acc[i];
            if constexpr (FX) v = fx_out(acc64[i], ta[rr] + twn);
            if (row < rows) out(ks, row0 + row, n0 + n, v);
        }
    }

    // Intel's XMX: the sub-groups split the WM x WN tile into 8 x ISG tiles, a column of them each (TPS tiles down its
    // row group's rows).  The weights' words go to local memory in Intel's VNNI order, [K / 4][output] (a word is 4 K
    // values of one output, as Dec gives them); a block of 32 is one product.  An 8 x N accumulator gives each lane one
    // column, its element i in row i (Xe's DPAS layout, as flash_attn's selection form takes it; intel_layout_ok checks it at the
    // start), so the scales are applied by element order: asking each element its coordinates cost more than the
    // products there.
    void intel_body(sycl::nd_item<1> it, int64_t ks) const {
#if STRATA_IQ_MMQ_INTEL
        constexpr int IM = 8, IK = 32, IN = ISG;
        constexpr int NT = WN / IN, NSUB = NSG * SG / ISG, RG = NSUB / NT, TPS = WM / IM / RG;
        static_assert(NSUB % NT == 0 && (WM / IM) % RG == 0, "the tiles split evenly over the sub-groups");
        constexpr int LDBV = WN * 4;   // a VNNI row: 4 K values of every output
        const int64_t gk = (int64_t) it.get_group(0), g = gk / ksplit, per = tiles_m * tiles_n, e = g / per,
                      r = g % per;
        const int64_t m0 = r % tiles_m * WM, n0 = r / tiles_m * WN;
        const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
        if (m0 >= rows) return;   // the whole work-group: this expert has fewer rows than the launch
        const auto sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_linear_id(), lid = (int) it.get_local_id(0);
        const int lane = (int) sg.get_local_linear_id();
        constexpr int lanes = NSG * SG;
        const int rs = sgid / NT * TPS * IM, cs = sgid % NT * IN;   // the sub-group's first row and column in the tile
        const uint8_t* wx = wtab != nullptr ? wtab[e] + woff : w + (size_t) e * expert_bytes;
        const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
        const auto ap = local_as<int8_t>(base + A_OFF);
        const auto bp = local_as<int8_t>(base + B_OFF);
        int8_t* al = ap.get();
        uint32_t* bw = local_as<uint32_t>(base + B_OFF).get();
        float* xsl = local_as<float>(base + XS_OFF).get();
        [[maybe_unused]] float* xul = local_as<float>(base + XU_OFF).get();
        [[maybe_unused]] float* wml = local_as<float>(base + WM_OFF).get();
        float* wsl = local_as<float>(base + WS_OFF).get();

        float acc[TPS][IM];
        for (int t = 0; t < TPS; ++t)
            for (int i = 0; i < IM; ++i) acc[t][i] = 0.0f;
        mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, IM, IN> ci;
        [[maybe_unused]] int32_t isum[FX ? TPS : 1][IM];
        [[maybe_unused]] int64_t acc64[FX ? TPS : 1][IM];
        [[maybe_unused]] Top* ta = local_as<Top>(base + FX_OFF).get();
        [[maybe_unused]] Top* tw = ta + WM;
        if constexpr (FX) {
            for (int t = 0; t < TPS; ++t)
                for (int i = 0; i < IM; ++i) {
                    isum[t][i] = 0;
                    acc64[t][i] = 0;
                }
            fx_tops(lid, lanes, row0, rows, m0, n0, wx, ta, tw);
        }

        const int64_t kb = K / 32;   // activation scales a row
        for (int64_t k0 = ks * kspan; k0 < sycl::min(K, (ks + 1) * kspan); k0 += KC) {
            sycl::group_barrier(it.get_group());   // the last step's tiles are read
            const bool last = TAIL && k0 + KC > K;   // TAIL: a last step past K, zeros past it
            for (int v = lid; v < WM * (KC / 16); v += lanes) {
                const int rr = v / (KC / 16), c = v % (KC / 16) * 16;
                const int64_t row = m0 + rr;
                u32x4 val = {0u, 0u, 0u, 0u};
                if (row < rows && (!last || k0 + c < K)) val = *reinterpret_cast<const u32x4*>(xq + (row0 + row) * K + k0 + c);
                if constexpr (HALF) {   // copy h keeps the half h of each block, zeros in the other
                    const u32x4 z = {0u, 0u, 0u, 0u};
                    const bool hi = c % 32 >= 16;
                    *reinterpret_cast<u32x4*>(al + static_cast<ptrdiff_t>(rr) * LDA + c) = hi ? z : val;
                    *reinterpret_cast<u32x4*>(al + static_cast<ptrdiff_t>(WM + rr) * LDA + c) = hi ? val : z;
                } else {
                    *reinterpret_cast<u32x4*>(al + static_cast<ptrdiff_t>(rr) * LDA + c) = val;
                }
            }
            for (int v = lid; v < WM * NB; v += lanes) {
                const int b = v / WM, rr = v % WM;
                const int64_t row = m0 + rr;
                xsl[v] = row < rows && (!last || k0 / 32 + b < kb) ? xs[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                if constexpr (MIN) xul[v] = row < rows && (!last || k0 / 32 + b < kb) ? xu[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
            }
            for (int v = lid; v < WN * NB; v += lanes) {
                // neighbouring work-items: neighbouring outputs of a VNNI row (their stores in local memory side by
                // side), or, where a work-item decodes four blocks or more a step (the 16-row tile), neighbouring
                // blocks of one output (their reads of the weights side by side) on Xe2's 16-lane tiles: B70, 16 rows
                // an expert, Ornith Q4_K_M 228 -> 190 us; the 64-row tile was 3% slower that way, and the A380's
                // 8-lane 16-row tile 6-23% slower
                constexpr bool by_block = IS == 1 && WN * NB >= 4 * lanes;
                const int nn = by_block ? v / NB : v % WN, b = by_block ? v % NB : v / WN;
                const Sub d = !last || k0 / 32 + b < kb ? Dec<TY>::get(wx + (size_t) (n0 + nn) * row_bytes, (int) (k0 / 32) + b, row_bytes)
                                                : Sub{{}, 0.0f, 0.0f, 0.0f};   // past K: zeros
                for (int j = 0; j < 8; ++j) bw[(b * 8 + j) * WN + nn] = d.v[j];
                wsl[(2 * b) * WN + nn] = d.s0;
                wsl[(2 * b + 1) * WN + nn] = d.s1;
                if constexpr (MIN || FXH) wml[b * WN + nn] = d.m;
            }
            sycl::group_barrier(it.get_group());
            for (int b = 0; b < NB; ++b) {
                mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, IK, IN, mx::layout::ext_intel_packed> bm;
                mx::joint_matrix_load(sg, bm, bp + static_cast<int>(b * 8 * LDBV + cs * 4), LDBV);
                for (int h = 0; h < (HALF ? 2 : 1); ++h) {
                    const float ws = wsl[(2 * b + h) * WN + cs + lane];   // this lane's column's
                    for (int t = 0; t < TPS; ++t) {
                        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, IM, IK, mx::layout::row_major> a;
                        mx::joint_matrix_load(sg, a, ap + static_cast<int>((h * WM + rs + t * IM) * LDA + b * 32), LDA);
                        mx::joint_matrix_fill(sg, ci, 0);
                        mx::joint_matrix_mad(sg, ci, a, bm, ci);
                        const float* xr = xsl + static_cast<ptrdiff_t>(b * WM + rs + t * IM);
                        int i = 0;
                        if constexpr (FX) {   // ws: the lane's column's integer scale
                            const int32_t cw = (int32_t) ws;
                            mx::joint_matrix_apply(sg, ci, [&](int32_t& c) {
                                isum[t][i] += cw * c;
                                ++i;
                            });
                            continue;
                        }
                        // the int32 sum converted by one instruction (exact_float's two are for NVIDIA's slow
                        // conversion) and added in one fma: B70, Ornith Q4_K_M at 256 rows an expert 988 -> 898 us
                        mx::joint_matrix_apply(sg, ci, [&](int32_t& c) {
                            acc[t][i] = sycl::fma(ws * xr[i], static_cast<float>(c), acc[t][i]);
                            ++i;
                        });
                    }
                }
                if constexpr (FX)
                    if ((k0 / 32 + b) % FXG == FXG - 1) {   // into int64 (lane: column, element i: row)
                        [[maybe_unused]] const float wd = fx_wd(wsl, wml, b, cs + lane);
                        [[maybe_unused]] const Top twl = tw[cs + lane];
                        for (int t = 0; t < TPS; ++t)
                            for (int i = 0; i < IM; ++i) {
                                const int rr = rs + t * IM + i;
                                acc64[t][i] += (int64_t) fx_a(xsl[b * WM + rr], wd, ta[rr] + twl) * isum[t][i];
                                isum[t][i] = 0;
                            }
                    }
                if constexpr (MIN) {   // less the weights' min of the lane's column times each row's sum
                    const float wmn = wml[b * WN + cs + lane];
                    for (int t = 0; t < TPS; ++t)
                        for (int i = 0; i < IM; ++i) acc[t][i] -= wmn * xul[b * WM + rs + t * IM + i];
                }
            }
        }
        for (int t = 0; t < TPS; ++t)
            for (int i = 0; i < IM; ++i) {
                const int64_t row = m0 + rs + (int64_t) t * IM + i;
                float v = acc[t][i];
                if constexpr (FX) v = fx_out(acc64[t][i], ta[rs + t * IM + i] + tw[cs + lane]);
                if (row < rows) out(ks, row0 + row, n0 + cs + lane, v);
            }
#else
        (void) it;
        (void) ks;
#endif
    }

    // REG: sub-group j takes the outputs [j ISG, (j + 1) ISG) of the work-group's WN, all its WM rows as TPS tiles of
    // 8.  Each lane decodes its own output's block of 32 (Dec) and writes its words into the sub-group's own B tile in
    // local memory, which the sub-group loads back (no work-group barrier; a B tile written by joint_matrix_apply lost
    // the B70); a type with a scale per 16 values takes two B tiles, each with the other half zeroed.  The activations'
    // tiles come from global memory or through local memory (below), their scales read by lane (row j ISG + l in
    // lane l) and given to each element by select_from_group.
    void intel_reg_body(sycl::nd_item<1> it, int64_t ks) const {
#if STRATA_IQ_MMQ_INTEL
        constexpr int IM = 8, IK = 32, IN = ISG, TPS = WM / IM, XR = (WM + IN - 1) / IN;
        const int64_t gk = (int64_t) it.get_group(0), g = gk / ksplit, per = tiles_m * tiles_n, e = g / per,
                      r = g % per;
        const int64_t m0 = r % tiles_m * WM, n0 = r / tiles_m * WN;
        const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
        if (m0 >= rows) return;   // the whole work-group: this expert has fewer rows than the launch
        const auto sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int64_t col = n0 + (int64_t) sgid * IN + lane;
        const uint8_t* wx = (wtab != nullptr ? wtab[e] + woff : w + (size_t) e * expert_bytes) + (size_t) col * row_bytes;
        constexpr int NH = HALF ? 2 : 1;   // B tiles a block
        // the sub-group's local memory: its B tiles, then its rows' A tiles of the block (SG_LOCAL bytes)
        const auto mine = lm.template get_multi_ptr<sycl::access::decorated::no>() + static_cast<int>(sgid * SG_LOCAL);
        const auto bp = local_as<int8_t>(mine);
        uint32_t* bw = local_as<uint32_t>(mine).get();
        const auto ap = local_as<int8_t>(mine + NH * IK * IN);
        [[maybe_unused]] u32x4* aw = local_as<u32x4>(mine + NH * IK * IN).get();
        [[maybe_unused]] const auto gx =
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                const_cast<int8_t*>(xq));
        const int64_t live = sycl::min<int64_t>(rows - m0, (int64_t) WM);   // the tile's rows of this expert
        const int tiles = (int) ((live + IM - 1) / IM);

        float acc[TPS][IM];
        for (int t = 0; t < TPS; ++t)
            for (int i = 0; i < IM; ++i) acc[t][i] = 0.0f;
        mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, IM, IN> ci;

        const int64_t kb = K / 32;
        // FX: the lane's output's top and its rows' (row j IN + l in lane l), the sums as in intel_body
        [[maybe_unused]] int32_t isum[FX ? TPS : 1][IM];
        [[maybe_unused]] int64_t acc64[FX ? TPS : 1][IM];
        [[maybe_unused]] Top ta_r[XR];
        [[maybe_unused]] Top twl = (Top) canon::kFxNone;
        if constexpr (FX) {
            for (int t = 0; t < TPS; ++t)
                for (int i = 0; i < IM; ++i) {
                    isum[t][i] = 0;
                    acc64[t][i] = 0;
                }
            for (int64_t b = 0; b < K / Dec<TY>::QK; ++b)
                twl = sycl::max(twl, (Top) canon::fx_exp(Dec<TY>::block_d(wx, (int) b)));
            twl = (Top) canon::fx_row_top(canon::fx_exp(Dec<TY>::block_d(wx, 0)), twl);
            for (int j = 0; j < XR; ++j) {
                const int64_t rr = (int64_t) j * IN + lane;
                ta_r[j] = (Top) canon::kFxNone;
                if (rr < live) {
                    for (int64_t b = 0; b < kb; b += FXG)
                        ta_r[j] = sycl::max(ta_r[j], (Top) canon::fx_exp(xs[(row0 + m0 + rr) * kb + b]));
                    ta_r[j] = (Top) canon::fx_row_top(canon::fx_exp(xs[(row0 + m0 + rr) * kb]), ta_r[j]);
                }
            }
        }
        const int64_t b0 = ks * kspan / 32, b1 = sycl::min(K, (ks + 1) * kspan) / 32;
        for (int64_t b = b0; b < b1; ++b) {
            const Sub d = Dec<TY>::get(wx, (int) b, row_bytes);
            float xr[XR];
            [[maybe_unused]] float xu_r[XR];
            for (int j = 0; j < XR; ++j) {
                const int64_t rr = (int64_t) j * IN + lane;
                const bool in = rr < live;
                xr[j] = in ? xs[(row0 + m0 + rr) * kb + b] : 0.0f;
                if constexpr (MIN) xu_r[j] = in ? xu[(row0 + m0 + rr) * kb + b] : 0.0f;
            }
            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, IK, IN, mx::layout::ext_intel_packed> bm[NH];
            sycl::group_barrier(sg);   // the last block's tiles are loaded
            for (int h = 0; h < NH; ++h)
                for (int j = 0; j < 8; ++j) bw[(h * 8 + j) * IN + lane] = !HALF || j / 4 == h ? d.v[j] : 0u;
            // A: on 16 lanes (Xe2) a bounds-checked tile load from global memory (zeros past the last row; Xe2's
            // block reads: B70, llm-jp-4.1 Q4_K_M at 32 rows an expert 426 us, against 592 with a plain load and 653
            // through local memory); on 8 (Xe-HPG), whose compile refuses the checked load (the A380: it failed the
            // build of every int8 XMX kernel), the rows' 32 values of the block copied into local memory, 16 bytes a
            // lane, zeros past the last row
            if constexpr (IS != 1)
                for (int c = lane; c < tiles * IM * 2; c += IN) {
                    const int64_t row = row0 + m0 + c / 2;
                    aw[c] = row < total_rows ? *reinterpret_cast<const u32x4*>(xq + row * K + b * IK + c % 2 * 16)
                                             : u32x4{0u, 0u, 0u, 0u};
                }
            sycl::group_barrier(sg);
            for (int h = 0; h < NH; ++h)
                mx::joint_matrix_load(sg, bm[h], bp + static_cast<int>(h * 8 * IN * 4), IN * 4);
            for (int t = 0; t < TPS; ++t) {
                if (t >= tiles) break;
                mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, IM, IK, mx::layout::row_major> a;
                if constexpr (IS == 1)
                    ix::joint_matrix_load_checked(sg, a, gx, (size_t) K, (size_t) total_rows, (size_t) K,
                                                  (size_t) (row0 + m0 + (int64_t) t * IM), (size_t) (b * IK));
                else
                    mx::joint_matrix_load(sg, a, ap + static_cast<int>(t * IM * IK), IK);
                for (int h = 0; h < NH; ++h) {
                    const float ws = h == 1 ? d.s1 : d.s0;
                    mx::joint_matrix_fill(sg, ci, 0);
                    mx::joint_matrix_mad(sg, ci, a, bm[h], ci);
                    int i = 0;
                    if constexpr (FX) {   // d.s0: the lane's output's integer scale
                        const int32_t cw = (int32_t) (h == 1 ? d.s1 : d.s0);
                        mx::joint_matrix_apply(sg, ci, [&](int32_t& c) {
                            isum[t][i] += cw * c;
                            ++i;
                        });
                        continue;
                    }
                    mx::joint_matrix_apply(sg, ci, [&](int32_t& c) {
                        const int rr = t * IM + i;
                        const float x = sycl::select_from_group(sg, xr[rr / IN], rr % IN);
                        acc[t][i] = sycl::fma(ws * x, static_cast<float>(c), acc[t][i]);
                        ++i;
                    });
                }
            }
            if constexpr (MIN) {   // less the weights' min of the lane's column times each row's sum
                for (int t = 0; t < TPS; ++t)
                    for (int i = 0; i < IM; ++i) {
                        const int rr = t * IM + i;
                        acc[t][i] -= d.m * sycl::select_from_group(sg, xu_r[rr / IN], rr % IN);
                    }
            }
            if constexpr (FX)
                if (b % FXG == FXG - 1)   // the group's sums into int64 (d.s1: its d)
                    for (int t = 0; t < TPS; ++t) {
                        if (t >= tiles) break;
                        for (int i = 0; i < IM; ++i) {
                            const int rr = t * IM + i;
                            const float x = sycl::select_from_group(sg, xr[rr / IN], rr % IN);
                            const Top tr = sycl::select_from_group(sg, ta_r[rr / IN], rr % IN);
                            acc64[t][i] += (int64_t) fx_a(x, FXH ? d.m : d.s1, tr + twl) * isum[t][i];
                            isum[t][i] = 0;
                        }
                    }
        }
        for (int t = 0; t < TPS; ++t)
            for (int i = 0; i < IM; ++i) {
                const int64_t row = m0 + (int64_t) t * IM + i;
                float v = acc[t][i];
                if constexpr (FX) {
                    const int rr = t * IM + i;
                    v = fx_out(acc64[t][i], sycl::select_from_group(sg, ta_r[rr / IN], rr % IN) + twl);
                }
                if (row < rows) out(ks, row0 + row, col, v);
            }
#else
        (void) it;
        (void) ks;
#endif
    }

    // the work-group's part of K, one a work-group.  K whole (every launch but a few rows' with the parts' buffer)
    // straight with the constant part: through the parts' index, every launch of the experts' products ran slower on
    // the B70 (Qwen3.6-35B-A3B UD-Q4_K_XL's 1K prompt 3012 -> 2885 tok/s, 2026-11-12), and the Q4_K and Q5_K WMMA
    // bodies of 32 rows a tile spilled 60 to 65 VGPRs on the RX 9060 XT (none with the constant).  Never a run of
    // parts a work-group: a loop over them around the bodies made AMD's compiler keep their addresses live across it,
    // and the WMMA bodies spilled up to 106 VGPRs (the few rows' products of MOSS-Image's text encoder at 16 and 64
    // rows 0.54 -> 0.97 ms, 2026-11-12; 0.61 without the loop).  Without it the Q2_K bodies take 203 VGPRs, not 239,
    // and run a little slower on the RX 9060 XT (Ornith-1.5-35B-A3B GSQ-RCO 3bit's 1K prompt 1519 -> 1472 tok/s,
    // 2026-11-13): the loop's registers, not its work, had made them faster
    void body(sycl::nd_item<1> it) const {
        if (ksplit == 1) part_body(it, 0);
        else part_body(it, (int64_t) it.get_group(0) % ksplit);
    }

    // Inlined at both of body's calls: with two calls NVIDIA's compiler kept it a function of its own, every kernel
    // calling it with the kernel's fields on the stack (192 to 320 bytes), and the slower int8 products made the
    // RTX 4070's prompts take FP16 from 609 rows an expert and chunks of half the tokens (Qwen3.8-Flash-Next
    // IQ3_XXS's 32K 1244 -> 886 tok/s, 2026-11-13)
    __attribute__((always_inline)) void part_body(sycl::nd_item<1> it, int64_t ks) const {
        if constexpr (DP4A) {
            dp4a_body(it, ks);
        } else if constexpr (IS != 0 && REG) {
            intel_reg_body(it, ks);
        } else if constexpr (IS != 0) {
            intel_body(it, ks);
        } else if constexpr (!built<TY>()) {
            (void) it;
            (void) ks;
        } else {
#if STRATA_IQ_MMQ_MATRIX
            const int64_t gk = (int64_t) it.get_group(0), g = gk / ksplit, per = tiles_m * tiles_n, e = g / per,
                      r = g % per;
            const int64_t m0 = r % tiles_m * WM, n0 = r / tiles_m * WN;
            const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
            if (m0 >= rows) return;   // the whole work-group: this expert has fewer rows than the launch
            const auto sg = it.get_sub_group();
            const int sgid = (int) sg.get_group_linear_id(), lid = (int) it.get_local_id(0);
            const int lane = (int) sg.get_local_linear_id();
            constexpr int lanes = NSG * SG;
            const int ms = sgid / NSGN * SGM * TM, ns = sgid % NSGN * SGN * TN;
            const uint8_t* wx = wtab != nullptr ? wtab[e] + woff : w + (size_t) e * expert_bytes;
            const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
            const auto ap = local_as<int8_t>(base + A_OFF);
            const auto bp = local_as<int8_t>(base + B_OFF);
            const auto xsp = local_as<float>(base + XS_OFF);
            const auto wsp = local_as<float>(base + WS_OFF);
            const auto stp = local_as<float>(base + ST_OFF) + static_cast<int>(sgid * TM * TN);
            int8_t* al = ap.get();
            int8_t* bl = bp.get();
            float* xsl = xsp.get();
            const auto xup = local_as<float>(base + XU_OFF);
            const auto wmp = local_as<float>(base + WM_OFF);
            [[maybe_unused]] float* xul = xup.get();
            [[maybe_unused]] float* wml = wmp.get();
            float* wsl = wsp.get();

            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[SGM][SGN];
            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, TM, TN> ci[SGM][SGN];
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) mx::joint_matrix_fill(sg, acc[i][j], 0.0f);
            // FX: a group's int32 sums as tiles (each block's times its integer scale, a tile of the outputs' loaded
            // with stride 0), then through the sub-group's staging tile into int64 sums a lane holds by position
            // (element q = lane + SG k of the tile): the tiles do not say where their elements sit
            constexpr int FQ = TM * TN / SG;
            [[maybe_unused]] mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, TM, TN>
                fxs[FX ? SGM : 1][FX ? SGN : 1];
            [[maybe_unused]] int64_t acc64[FX ? SGM : 1][FX ? SGN : 1][FX ? FQ : 1];
            [[maybe_unused]] Top* ta = local_as<Top>(base + FX_OFF).get();
            [[maybe_unused]] Top* tw = ta + WM;
            [[maybe_unused]] const auto sti = local_as<int32_t>(base + ST_OFF) + static_cast<int>(sgid * TM * TN);
            if constexpr (FX) {
                for (int i = 0; i < SGM; ++i)
                    for (int j = 0; j < SGN; ++j) {
                        mx::joint_matrix_fill(sg, fxs[i][j], 0);
                        for (int k = 0; k < FQ; ++k) acc64[i][j][k] = 0;
                    }
                fx_tops(lid, lanes, row0, rows, m0, n0, wx, ta, tw);
            }

            const int64_t kb = K / 32;   // activation scales a row
            for (int64_t k0 = ks * kspan; k0 < sycl::min(K, (ks + 1) * kspan); k0 += KC) {
                sycl::group_barrier(it.get_group());   // the last step's tiles are read
                const bool last = TAIL && k0 + KC > K;   // TAIL: a last step past K, zeros past it
                for (int v = lid; v < WM * (KC / 16); v += lanes) {
                    const int rr = v / (KC / 16), c = v % (KC / 16) * 16;
                    const int64_t row = m0 + rr;
                    u32x4 val = {0u, 0u, 0u, 0u};
                    if (row < rows && (!last || k0 + c < K)) val = *reinterpret_cast<const u32x4*>(xq + (row0 + row) * K + k0 + c);
                    *reinterpret_cast<u32x4*>(al + static_cast<int>(rr * LDA + c)) = val;
                }
                for (int v = lid; v < WM * NB; v += lanes) {
                    const int b = v / WM, rr = v % WM;
                    const int64_t row = m0 + rr;
                    xsl[v] = row < rows && (!last || k0 / 32 + b < kb) ? xs[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                    if constexpr (MIN) xul[v] = row < rows && (!last || k0 / 32 + b < kb) ? xu[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                }
                for (int v = lid; v < WN * NB; v += lanes) {
                    const int nn = v / NB, b = v % NB;
                    const Sub d = !last || k0 / 32 + b < kb ? Dec<TY>::get(wx + (size_t) (n0 + nn) * row_bytes, (int) (k0 / 32) + b, row_bytes)
                                                : Sub{{}, 0.0f, 0.0f, 0.0f};   // past K: zeros
                    u32x4* o = reinterpret_cast<u32x4*>(bl + static_cast<int>(nn * LDB + b * 32));
                    o[0] = u32x4{d.v[0], d.v[1], d.v[2], d.v[3]};
                    o[1] = u32x4{d.v[4], d.v[5], d.v[6], d.v[7]};
                    wsl[(2 * b) * WN + nn] = d.s0;
                    wsl[(2 * b + 1) * WN + nn] = d.s1;
                    if constexpr (MIN || FXH) wml[b * WN + nn] = d.m;
                }
                sycl::group_barrier(it.get_group());
                for (int b = 0; b < NB; ++b) {
                    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> xsf[SGM];
                    if constexpr (!FX)
                        for (int i = 0; i < SGM; ++i)   // element (r, c): the scale of row r
                            mx::joint_matrix_load(sg, xsf[i], xsp + static_cast<int>(b * WM + ms + i * TM), 0,
                                                  mx::layout::col_major);
                    for (int h = 0; h < (HALF ? 2 : 1); ++h) {
                        for (int i = 0; i < SGM; ++i)
                            for (int j = 0; j < SGN; ++j) mx::joint_matrix_fill(sg, ci[i][j], 0);
                        for (int kk = HALF ? h * TK : 0; kk < (HALF ? h * TK + TK : 32); kk += TK) {
                            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, TM, TK, mx::layout::row_major> a[SGM];
                            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, TK, TN, mx::layout::col_major> bm[SGN];
                            for (int i = 0; i < SGM; ++i)
                                mx::joint_matrix_load(sg, a[i], ap + static_cast<int>((ms + i * TM) * LDA + b * 32 + kk), LDA);
                            for (int j = 0; j < SGN; ++j)
                                mx::joint_matrix_load(sg, bm[j], bp + static_cast<int>((ns + j * TN) * LDB + b * 32 + kk), LDB);
                            for (int i = 0; i < SGM; ++i)
                                for (int j = 0; j < SGN; ++j) mx::joint_matrix_mad(sg, ci[i][j], a[i], bm[j], ci[i][j]);
                        }
                        if constexpr (FX) {
                            for (int j = 0; j < SGN; ++j) {
                                // element (r, c): the integer scale of column c
                                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> sc;
                                mx::joint_matrix_load(sg, sc, wsp + static_cast<int>((2 * b + h) * WN + ns + j * TN), 0,
                                                      mx::layout::row_major);
                                for (int i = 0; i < SGM; ++i) {
                                    mx::joint_matrix_apply(sg, ci[i][j], sc,
                                                           [](int32_t& c, float& w) { c *= (int32_t) w; });
                                    mx::joint_matrix_apply(sg, fxs[i][j], ci[i][j],
                                                           [](int32_t& a, int32_t& c) { a += c; });
                                }
                            }
                            continue;
                        }
                        for (int j = 0; j < SGN; ++j)
                            for (int i = 0; i < SGM; ++i) {
                                // the pair of scales (r, c): the weights' of column c along each row, times the row's
                                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> sc;
                                mx::joint_matrix_load(sg, sc, wsp + static_cast<int>((2 * b + h) * WN + ns + j * TN), 0,
                                                      mx::layout::row_major);
                                mx::joint_matrix_apply(sg, sc, xsf[i], [](float& s, float& x) { s *= x; });
                                mx::joint_matrix_apply(sg, sc, ci[i][j], [](float& s, int32_t& c) { s *= exact_float(c); });
                                mx::joint_matrix_apply(sg, acc[i][j], sc, [](float& a, float& s) { a += s; });
                            }
                    }
                    if constexpr (FX)
                        if ((k0 / 32 + b) % FXG == FXG - 1)   // the group's sums out by position, into int64
                            for (int i = 0; i < SGM; ++i)
                                for (int j = 0; j < SGN; ++j) {
                                    mx::joint_matrix_store(sg, fxs[i][j], sti, TN, mx::layout::row_major);
                                    sycl::group_barrier(sg);
                                    for (int k = 0; k < FQ; ++k) {
                                        const int q = lane + SG * k, rr = ms + i * TM + q / TN;
                                        [[maybe_unused]] const int nn = ns + j * TN + q % TN;
                                        acc64[i][j][k] += (int64_t) fx_a(xsl[b * WM + rr], fx_wd(wsl, wml, b, nn),
                                                                             ta[rr] + tw[nn]) * sti[q];
                                    }
                                    sycl::group_barrier(sg);
                                    mx::joint_matrix_fill(sg, fxs[i][j], 0);
                                }
                    if constexpr (MIN)   // less the weights' min of column c times row r's sum
                        for (int i = 0; i < SGM; ++i) {
                            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> xuf;
                            mx::joint_matrix_load(sg, xuf, xup + static_cast<int>(b * WM + ms + i * TM), 0,
                                                  mx::layout::col_major);
                            for (int j = 0; j < SGN; ++j) {
                                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> mn;
                                mx::joint_matrix_load(sg, mn, wmp + static_cast<int>(b * WN + ns + j * TN), 0,
                                                      mx::layout::row_major);
                                mx::joint_matrix_apply(sg, mn, xuf, [](float& m, float& x) { m *= x; });
                                mx::joint_matrix_apply(sg, acc[i][j], mn, [](float& a, float& m) { a -= m; });
                            }
                        }
                }
            }
            if constexpr (FX) {   // straight out of the lanes' int64 sums
                for (int i = 0; i < SGM; ++i)
                    for (int j = 0; j < SGN; ++j)
                        for (int k = 0; k < FQ; ++k) {
                            const int q = lane + SG * k, rr = ms + i * TM + q / TN, nn = ns + j * TN + q % TN;
                            float v;
                            v = fx_out(acc64[i][j][k], ta[rr] + tw[nn]);
                            if (m0 + rr < rows) out(ks, row0 + m0 + rr, n0 + nn, v);
                        }
                return;
            }
            // out through local memory, the rows past the expert's last not written
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) {
                    mx::joint_matrix_store(sg, acc[i][j], stp, TN, mx::layout::row_major);
                    sycl::group_barrier(sg);
                    const int64_t rb = m0 + ms + (int64_t) i * TM, cb = n0 + ns + (int64_t) j * TN;
                    for (int q = lane; q < TM * TN; q += SG) {
                        const int64_t row = rb + q / TN;
                        if (row < rows) out(ks, row0 + row, cb + q % TN, stp[q]);
                    }
                    sycl::group_barrier(sg);
                }
#elif STRATA_IQ_MMQ_WMMA
            // The int8 WMMA of RDNA3 (GFX11) and RDNA4 (GFX12), 16 x 16 x 16 on 32 lanes; lane l holds column l % 16 of
            // B and of the accumulator, row l % 16 of A.  GFX12: each lane 8 of the 16 K values (l / 16 picks which),
            // the accumulator's rows 8 (l / 16) + e.  GFX11: each lane all 16 K values (lanes 16-31 repeat 0-15), the
            // accumulator's rows 2 e + l / 16.  The same products and scales as the joint_matrix body above, in the
            // positions those layouts give.
            const int64_t gk = (int64_t) it.get_group(0), g = gk / ksplit, per = tiles_m * tiles_n, e = g / per,
                      r = g % per;
            const int64_t m0 = r % tiles_m * WM, n0 = r / tiles_m * WN;
            const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
            if (m0 >= rows) return;   // the whole work-group: this expert has fewer rows than the launch
            const auto sg = it.get_sub_group();
            const int sgid = (int) sg.get_group_linear_id(), lid = (int) it.get_local_id(0);
            const int lane = (int) sg.get_local_linear_id(), lo = lane & 15, hi = lane >> 4;
            constexpr int lanes = NSG * SG;
            const int ms = sgid / NSGN * SGM * TM, ns = sgid % NSGN * SGN * TN;
            const uint8_t* wx = wtab != nullptr ? wtab[e] + woff : w + (size_t) e * expert_bytes;
            const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
            int8_t* al = local_as<int8_t>(base + A_OFF).get();
            int8_t* bl = local_as<int8_t>(base + B_OFF).get();
            float* xsl = local_as<float>(base + XS_OFF).get();
            [[maybe_unused]] float* xul = local_as<float>(base + XU_OFF).get();
            [[maybe_unused]] float* wml = local_as<float>(base + WM_OFF).get();
            float* wsl = local_as<float>(base + WS_OFF).get();
            using i8v = xe::wmma::iv8;     // amd_wmma.hpp's int8 operands and layouts
            using frag = xe::wmma::s8f;
            constexpr int KOFF = xe::wmma::KOFF;   // bytes of K a lane holds past its half's start
            auto row_of = [&](int el) { return xe::wmma::row_of(hi, el); };
            float acc[SGM][SGN][8];
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j)
                    for (int el = 0; el < 8; ++el) acc[i][j][el] = 0.0f;
            // FX: as intel_body, by the elements' positions
            [[maybe_unused]] int32_t isum[FX ? SGM : 1][FX ? SGN : 1][8];
            [[maybe_unused]] int64_t acc64[FX ? SGM : 1][FX ? SGN : 1][8];
            [[maybe_unused]] Top* ta = local_as<Top>(base + FX_OFF).get();
            [[maybe_unused]] Top* tw = ta + WM;
            if constexpr (FX) {
                for (int i = 0; i < SGM; ++i)
                    for (int j = 0; j < SGN; ++j)
                        for (int el = 0; el < 8; ++el) {
                            isum[i][j][el] = 0;
                            acc64[i][j][el] = 0;
                        }
                fx_tops(lid, lanes, row0, rows, m0, n0, wx, ta, tw);
            }

            const int64_t kb = K / 32;   // activation scales a row
            for (int64_t k0 = ks * kspan; k0 < sycl::min(K, (ks + 1) * kspan); k0 += KC) {
                sycl::group_barrier(it.get_group());   // the last step's tiles are read
                const bool last = TAIL && k0 + KC > K;   // TAIL: a last step past K, zeros past it
                for (int v = lid; v < WM * (KC / 16); v += lanes) {
                    const int rr = v / (KC / 16), c = v % (KC / 16) * 16;
                    const int64_t row = m0 + rr;
                    u32x4 val = {0u, 0u, 0u, 0u};
                    if (row < rows && (!last || k0 + c < K)) val = *reinterpret_cast<const u32x4*>(xq + (row0 + row) * K + k0 + c);
                    *reinterpret_cast<u32x4*>(al + rr * LDA + c) = val;
                }
                for (int v = lid; v < WM * NB; v += lanes) {
                    const int b = v / WM, rr = v % WM;
                    const int64_t row = m0 + rr;
                    xsl[v] = row < rows && (!last || k0 / 32 + b < kb) ? xs[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                    if constexpr (MIN) xul[v] = row < rows && (!last || k0 / 32 + b < kb) ? xu[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                }
                for (int v = lid; v < WN * NB; v += lanes) {
                    const int nn = v / NB, b = v % NB;
                    const Sub d = !last || k0 / 32 + b < kb ? Dec<TY>::get(wx + (size_t) (n0 + nn) * row_bytes, (int) (k0 / 32) + b, row_bytes)
                                                : Sub{{}, 0.0f, 0.0f, 0.0f};   // past K: zeros
                    u32x4* o = reinterpret_cast<u32x4*>(bl + nn * LDB + b * 32);
                    o[0] = u32x4{d.v[0], d.v[1], d.v[2], d.v[3]};
                    o[1] = u32x4{d.v[4], d.v[5], d.v[6], d.v[7]};
                    wsl[(2 * b) * WN + nn] = d.s0;
                    wsl[(2 * b + 1) * WN + nn] = d.s1;
                    if constexpr (MIN || FXH) wml[b * WN + nn] = d.m;
                }
                sycl::group_barrier(it.get_group());
                for (int b = 0; b < NB; ++b) {
                    float xsr[SGM][8];   // the rows' scales of the block, in the accumulator's row order
                    for (int i = 0; i < SGM; ++i)
                        for (int el = 0; el < 8; ++el) xsr[i][el] = xsl[b * WM + ms + i * TM + row_of(el)];
                    for (int h = 0; h < (HALF ? 2 : 1); ++h) {
                        i8v ci[SGM][SGN];
                        for (int i = 0; i < SGM; ++i)
                            for (int j = 0; j < SGN; ++j) ci[i][j] = i8v{0, 0, 0, 0, 0, 0, 0, 0};
                        for (int kk = HALF ? h * TK : 0; kk < (HALF ? h * TK + TK : 32); kk += TK) {
                            frag a[SGM], bm[SGN];
                            for (int i = 0; i < SGM; ++i)
                                a[i] = *reinterpret_cast<const frag*>(al + (ms + i * TM + lo) * LDA + b * 32 + kk + hi * KOFF);
                            for (int j = 0; j < SGN; ++j)
                                bm[j] = *reinterpret_cast<const frag*>(bl + (ns + j * TN + lo) * LDB + b * 32 + kk + hi * KOFF);
                            for (int i = 0; i < SGM; ++i)
                                for (int j = 0; j < SGN; ++j)
                                    ci[i][j] = xe::wmma::mad_s8(a[i], bm[j], ci[i][j]);
                        }
                        for (int j = 0; j < SGN; ++j) {
                            const float ws = wsl[(2 * b + h) * WN + ns + j * TN + lo];   // this lane's column's
                            if constexpr (FX) {   // ws: the integer scale
                                for (int i = 0; i < SGM; ++i)
                                    for (int el = 0; el < 8; ++el) isum[i][j][el] += (int32_t) ws * ci[i][j][el];
                                continue;
                            }
                            for (int i = 0; i < SGM; ++i)
                                for (int el = 0; el < 8; ++el)
                                    acc[i][j][el] += ws * xsr[i][el] * exact_float(ci[i][j][el]);
                        }
                    }
                    if constexpr (FX)
                        if ((k0 / 32 + b) % FXG == FXG - 1) {   // the group's sums into int64
                            
                                // fx_scale's parts, a row's and an output's apart: m = ma mw, s = (ea - ta + kFxHead) +
                                // (ew - tw)
                                uint32_t am[SGM][8];
                                int as[SGM][8];
                                for (int i = 0; i < SGM; ++i)
                                    for (int el = 0; el < 8; ++el) {
                                        const canon::FxPart p = canon::fx_part(xsr[i][el]);
                                        am[i][el] = p.m;
                                        as[i][el] = p.e - ta[ms + i * TM + row_of(el)] + canon::kFxHead;
                                    }
                                for (int j = 0; j < SGN; ++j) {
                                    const int nn = ns + j * TN + lo;
                                    const canon::FxPart wp = canon::fx_part(fx_wd(wsl, wml, b, nn));
                                    const int ws = wp.e - tw[nn];
                                    for (int i = 0; i < SGM; ++i)
                                        for (int el = 0; el < 8; ++el) {
                                            acc64[i][j][el] += (int64_t) canon::fx_shift(am[i][el] * wp.m, as[i][el] + ws,
                                                                                         wp.neg) * isum[i][j][el];
                                            isum[i][j][el] = 0;
                                        }
                                }
                            
                        }
                    if constexpr (MIN)
                        for (int j = 0; j < SGN; ++j) {
                            const float wm = wml[b * WN + ns + j * TN + lo];
                            for (int i = 0; i < SGM; ++i)
                                for (int el = 0; el < 8; ++el)
                                    acc[i][j][el] -= wm * xul[b * WM + ms + i * TM + row_of(el)];
                        }
                }
            }
            // straight out, the rows past the expert's last not written
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) {
                    const int64_t cb = n0 + ns + (int64_t) j * TN + lo;
                    for (int el = 0; el < 8; ++el) {
                        const int64_t row = m0 + ms + (int64_t) i * TM + row_of(el);
                        float v = acc[i][j][el];
                        if constexpr (FX)
                            v = fx_out(acc64[i][j][el], ta[ms + i * TM + row_of(el)] + tw[ns + j * TN + lo]);
                        if (row < rows) out(ks, row0 + row, cb, v);
                    }
                }
#endif
        }
    }
    // The entry, with the work-group launch() takes.  intel/llvm v7.2.0 bounds the registers a wave of an AMD GPU's
    // kernel may take by its work-group from the attribute only, not from a kernel property: without it the WMMA
    // bodies were sized for 1024 lanes, 192 VGPRs on the RX 9060 XT, and 57-64 of them spilled (v7.1.1's code fit in
    // 118-150), the experts' products of a 32K prompt 1.4x slower.  A kernel takes the attributes or properties, not
    // both: Intel's XMX bodies, which need the large register file (a property), go without the work-group, which
    // as a property breaks their (empty) AMD build in v7.2.0 ("reqd_work_group_size requires
    // amdgpu-flat-work-group-size").
    [[sycl::reqd_work_group_size(NSG * SG), sycl::reqd_sub_group_size(STRATA_SUB_GROUP(ISG))]]
    void operator()(sycl::nd_item<1> it) const requires(!DP4A && IS == 0) { body(it); }
    // no sub-group operations: the device's own size
    [[sycl::reqd_work_group_size(NSG * SG)]] void operator()(sycl::nd_item<1> it) const requires(DP4A) { body(it); }
    void operator()(sycl::nd_item<1> it) const requires(!DP4A && IS != 0) { body(it); }
    auto get(syclex::properties_tag) const requires(!DP4A && IS != 0) {
        // Intel's XMX bodies in the large register file: in the default one their sums and tiles spilled, 3.1x slower
        // (B70, llm-jp-4.1's experts, 256 rows an expert: 6,464 against 2,070 us)
        return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(ISG)>,
                                  sycl::ext::intel::experimental::grf_size<256>};
    }
};

// The parts' buffer of each queue (iq_mmq_split_reserve), for one expert's few rows with K in parts (launch)
struct SplitState {
    std::mutex mu;
    std::unordered_map<const sycl::queue*, std::pair<float*, size_t>> bufs;
};
SplitState& split_state(sycl::queue& q) {
    static core::PerDevice<std::shared_ptr<SplitState>> per_device;
    return *per_device.get(q.get_device(), [] { return std::make_shared<SplitState>(); });
}

// The queue's reserved parts' buffer when it holds `floats`, else null
float* part_buffer(sycl::queue& q, size_t floats) {
    SplitState& st = split_state(q);
    std::lock_guard<std::mutex> lock(st.mu);
    const auto it = st.bufs.find(&q);
    return it != st.bufs.end() && it->second.second >= floats ? it->second.first : nullptr;
}

template<int TY, int SGM, int NSGM, bool DP4A, bool TAIL, int IS, bool REG = false>
sycl::event launch(sycl::queue& q, const void* w, const uint8_t* const* wtab, size_t woff, size_t expert_bytes,
                   int64_t w_rows, int64_t w_cols, int n, const void* xq, int64_t total_rows, const int32_t* bounds,
                   int64_t max_rows, const int32_t* ids, float* dst, int64_t ld_dst, const float* xs_in = nullptr) {
    using KT = Kernel<TY, SGM, NSGM, DP4A, TAIL, IS, REG>;
    const int64_t tiles_m = (max_rows + KT::WM - 1) / KT::WM, tiles_n = w_rows / KT::WN;
    const size_t row_bytes = (size_t) (w_cols / Dec<TY>::QK) * (size_t) Dec<TY>::BSZ;
    const auto* x8 = static_cast<const int8_t*>(xq);
    const auto* xs = xs_in != nullptr ? xs_in : reinterpret_cast<const float*>(x8 + total_rows * w_cols);
    const auto* xu = xs + total_rows * (w_cols / 32);
    const int64_t steps = (w_cols + KC - 1) / KC;
    // One expert's few rows, where the queue holds the parts' buffer: K in kIqMmqSplitMax parts (two steps a part at
    // least), a work-group each, each part's sums from zero and a pass of their own adding the parts in order.  A
    // prompt's few rows (64 tokens of PLaMo 3 Translate 2B) left the RTX 4070's 46 units mostly idle with K whole: 16
    // tiles of 64 x 128 for a 2560-output product (llama.cpp splits K the same way, its stream-K).  The parts are the
    // product's own, so the sums and a prompt's states are the same from start to start: a work-group's own share of K
    // before, as many parts as the work-groups a timing at the start aimed for (0 to 16 quarters of the compute units),
    // made them follow that choice, which a near tie took one way or the other (the RX 9060 XT, 2026-11-10: Mage-Flow's
    // text encoder 0 or 16 quarters, 0.35 ms each, and another image from the same seed).  Two parts: each part's sums
    // go through memory, and more of them cost more than the work-groups they let run (Qwen3.5-9B Q4_K_M's FFN gate at
    // 16 and 64 rows, 2026-11-11: the B70 0.40 ms with 2 parts, 0.43 with 4, 0.46 with 8; the RTX 4070 0.32, 0.34,
    // 0.40).  A work-group each, not one running both: one work-group was the slower wherever the two were timed
    // (2026-11-12: the B70 0.59 ms against 0.42 and 0.47 against 0.34, the RTX 4070 0.38 against 0.32), and its loop
    // over the parts spilled on AMD (body).  The pass: adding the second part into dst as the work-group went cost the
    // RTX 4070's products 0.37 ms against 0.33, the read in the kernels' output; the last of a tile's work-groups
    // adding them, a counter a tile, 0.40 against 0.32 (the B70's 0.24 against 0.29, the RX 9060 XT's as long as the
    // pass, 2026-11-13: MOSS-Image's text encoder at 16 and 64 rows 1.30 to 1.47 ms either way, and every launch of the
    // kernel slower with its code, the 512-row products 1.86 -> 6.6 ms).  It costs a short prompt 1-2 ms on the RTX
    // 4070 (Qwen3.5-9B's 52 tokens 1127 -> 1083 tok/s, PLaMo 3 Translate 2B's 64 3810 -> 3372, 2026-11-11).
    int64_t kspan = steps * KC, ksplit = 1;
    float* part = nullptr;
    // (the fixed-point sums take K whole: a part's int64 sums would need a pass of their own)
    if (!KT::FX && n == 1 && total_rows <= kIqMmqSplitRows && steps >= 4) {
        const int64_t want = std::min<int64_t>(kIqMmqSplitMax, steps / 2);
        const int64_t span = (steps + want - 1) / want * KC, parts = (steps * KC + span - 1) / span;
        part = part_buffer(q, (size_t) (parts * total_rows * w_rows));
        if (part != nullptr) {
            ksplit = parts;
            kspan = span;
        }
    }
    sycl::event e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>((size_t) KT::LOCAL), h);
        const size_t wg = (size_t) KT::LANES;
        KT k{static_cast<const uint8_t*>(w), wtab, woff, expert_bytes, row_bytes, x8, xs, xu, w_cols, bounds, ids, dst,
             ld_dst, tiles_m, tiles_n, lm};
        k.ksplit = ksplit;
        k.kspan = kspan;
        k.part = part;
        k.total_rows = total_rows;
        k.n_out = w_rows;
        h.parallel_for(sycl::nd_range<1>((size_t) n * (size_t) (tiles_m * tiles_n * ksplit) * wg, wg), k);
    });
    if (part == nullptr) return e;
    // the parts added in order into dst (one expert: its rows are the ids' first total_rows)
    return q.parallel_for(sycl::range<2>((size_t) total_rows, (size_t) w_rows), [=](sycl::id<2> i) {
        const int64_t row = (int64_t) i[0], col = (int64_t) i[1];
        float v = part[row * w_rows + col];
        for (int64_t ks = 1; ks < ksplit; ++ks) v += part[(ks * total_rows + row) * w_rows + col];
        dst[(int64_t) ids[row] * ld_dst + col] = v;
    });
}

// Whether an int32 accumulator of Intel's 8 x N shape holds, in lane l, column l with element i in row i (the order
// intel_body applies the scales in): 1 when it does.
template<int IS>
struct IntelLayoutCheck {
    static constexpr int N = IS == 1 ? 16 : 8;
    sycl::local_accessor<int32_t, 1> pi;
    int* out;
    void operator()(sycl::nd_item<1> it) const {
#if STRATA_IQ_MMQ_INTEL
        const auto sg = it.get_sub_group();
        const int lid = (int) it.get_local_id(0);
        for (int i = lid; i < 8 * N; i += N) pi[i] = i;
        sycl::group_barrier(it.get_group());
        mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, 8, N> a;
        mx::joint_matrix_load(sg, a, pi.template get_multi_ptr<sycl::access::decorated::no>(), N, mx::layout::row_major);
        int bad = 0, i = 0;
        mx::joint_matrix_apply(sg, a, [&](int32_t& v) { bad |= v != i * N + lid; ++i; });
        bad |= i != 8;
        bad = sycl::reduce_over_group(sg, bad, sycl::bit_or<int>());
        if (lid == 0) *out = bad ? 0 : 1;
#else
        (void) it;
#endif
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(N)>}; }
};

// Intel's int8 XMX shape the device reports at its sub-group size, its local memory and work-group size permitting
// intel_body, and its accumulators laid out as intel_body reads them: 1 Xe2's (8 x 16 x 32 on 16 lanes), 2 Xe-HPG's (8
// x 8 x 32 on 8), else 0.  STRATA_NO_XMX=1: 0.
template<int IS>
bool intel_fits(sycl::queue& q) {
    const sycl::device d = q.get_device();
    using Big = Kernel<17, 2, 2, false, false, IS>;
    const size_t local = std::max<size_t>(Big::LOCAL, Kernel<12, 2, 2, false, false, IS>::LOCAL);
    if (d.get_info<sycl::info::device::local_mem_size>() < local ||
        d.get_info<sycl::info::device::max_work_group_size>() < (size_t) Big::NSG * SG)
        return false;
    try {   // the large register file (intel_body's properties), as xmx_gemm.cpp's large_grf_ok checks it
        (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {d},
                                                                       {sycl::get_kernel_id<Big>()});
        return true;
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xestrata: the int8 XMX products do not build for this GPU (%s)\n", e.what());
        return false;
    }
}
template<int IS>
bool intel_layout_ok(sycl::queue& q) {
    int* out = sycl::malloc_device<int>(1, q);
    if (out == nullptr) return false;
    int res = 0;
    try {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<int32_t, 1> pi(sycl::range<1>((size_t) 8 * IntelLayoutCheck<IS>::N), h);
            h.parallel_for(sycl::nd_range<1>(IntelLayoutCheck<IS>::N, IntelLayoutCheck<IS>::N), IntelLayoutCheck<IS>{pi, out});
        });
        q.memcpy(&res, out, sizeof(int)).wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xestrata: the int8 XMX layout check did not run (%s)\n", e.what());
        res = 0;
    }
    sycl::free(out, q);
    return res == 1;
}
int intel_shape_of(sycl::queue& q) {
    static core::PerDevice<int> per_device;
    return per_device.get(q.get_device(), [&q] {
        if (no_xmx()) return 0;
        const sycl::device d = q.get_device();
        const auto combos = matrix_combinations(d);
        for (int is = 1; is <= 2; ++is) {
            const size_t n = is == 1 ? 16 : 8;
            if (!xe::has_sub_group(d, n)) continue;
            bool rep = false;
            for (const auto& c : combos) {
                const bool m = c.msize == 8 || (c.msize == 0 && c.max_msize >= 8);
                const bool nn = c.nsize == n || (c.nsize == 0 && c.max_nsize >= n);
                const bool k = c.ksize == 32 || (c.ksize == 0 && c.max_ksize >= 32);
                rep = rep || (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 &&
                              c.ctype == mx::matrix_type::sint32 && c.dtype == mx::matrix_type::sint32 && m && nn && k);
            }
            if (!rep || !(is == 1 ? intel_fits<1>(q) : intel_fits<2>(q))) continue;
            if (is == 1 ? intel_layout_ok<1>(q) : intel_layout_ok<2>(q)) return is;
            std::fprintf(stderr, "xestrata: the GPU's int8 XMX accumulators are laid out otherwise: the experts' prompt "
                                 "products do not take them\n");
        }
        return 0;
    });
}

// Whether a product on Intel's XMX takes the sub-group form (intel_reg_body) before the work-group one (intel_body),
// by its type, N, K, whether it is one expert's (a dense product) and the class of the rows its experts average
// (powers of two, 8 to 256).  Neither wins everywhere: B70 (2026-11-10), 16 experts of r rows, llm-jp-4.1's Q4_K_M
// and Qwen3.8's IQ2_XS, the sub-group form at 32 rows 425 and 256 us against 504 and 346, at 8 rows 292 and 156
// against 209 and 194, at 64 rows 652 and 435 against 538 and 361.  So the two are timed for each class when the
// engine warms up (iq_mmq_choose_forms), not inside a prompt, and kept for the GPU (the sub-group form 5% faster or
// more, so that a near tie does not flip from start to start); a product not timed there takes the work-group form.
// XESTRATA_IQ_MMQ_SUB_GROUP=1 or 0 takes one form.
struct SgChoice {
    std::mutex mu;
    std::map<std::tuple<int, int, bool, int64_t, int64_t>, bool> picks;
};
SgChoice& sg_choice(sycl::queue& q) {
    static core::PerDevice<std::shared_ptr<SgChoice>> per_device;
    return *per_device.get(q.get_device(), [] { return std::make_shared<SgChoice>(); });
}
// top: the last class (log2 of its rows), which takes every larger mean too
int rows_class(int64_t mean_rows, int top = 8) {
    int r = 3;   // 8 rows and fewer
    while (r < top && ((int64_t) 1 << r) < mean_rows) ++r;
    return r;
}
thread_local bool t_choosing = false;   // inside iq_mmq_choose_forms: a class not chosen yet is timed
// The choice between two forms of a product for its class (sg_choice's key), kept for the GPU: `first` timed against
// `second` inside iq_mmq_choose_forms and taken where it is 5% faster or more; a class not timed takes `second`.
bool timed_first(sycl::queue& q, int ty, int64_t mean_rows, int n, int64_t N, int64_t K, int top,
                 const std::function<sycl::event()>& first, const std::function<sycl::event()>& second) {
    SgChoice& c = sg_choice(q);
    const auto key = std::make_tuple(ty, rows_class(mean_rows, top), n == 1, N, K);
    {
        std::lock_guard<std::mutex> lock(c.mu);
        if (const auto it = c.picks.find(key); it != c.picks.end()) return it->second;
    }
    if (!t_choosing) return false;
    double a_s = 0, b_s = 0;
    const auto time = [&](const std::function<sycl::event()>& f) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 3; ++i) f();
        q.wait_and_throw();
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 3;
    };
    try {
        q.wait_and_throw();
        first();   // each once first: its kernel's build
        second();
        q.wait_and_throw();
        a_s = time(first);
        b_s = time(second);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata: timing the int8 expert products' forms: %s\n", e.what());
        a_s = b_s = 0;
    }
    const bool pick = a_s > 0 && b_s > 0 && a_s * 1.05 < b_s;
    std::lock_guard<std::mutex> lock(c.mu);
    c.picks[key] = pick;
    return pick;
}
bool intel_sub_group_first(sycl::queue& q, int ty, int64_t mean_rows, int n, int64_t N, int64_t K,
                           const std::function<sycl::event()>& sg, const std::function<sycl::event()>& wg) {
    static const char* const env = strata::env_get("IQ_MMQ_SUB_GROUP");
    if (env != nullptr && (env[0] == '0' || env[0] == '1')) return env[0] == '1';
    return timed_first(q, ty, mean_rows, n, N, K, 8, sg, wg);
}

// Whether an expert group's product on RDNA3's or RDNA4's WMMA takes the FP16 form (iq_mmq_f16_grouped: the weights
// dequantized to FP16 in local memory, the int8 activations times their scales) before the int8 one: timed for the
// classes of 512 and 1024 rows an expert (the last takes every larger mean) as Intel's two forms are
// (iq_mmq_choose_forms; the first takes 257 rows on), and taken where it is 5% faster; fewer rows keep the int8 form
// untimed.  RX 9060 XT (2026-11-13, Qwen3.8-Flash-Next IQ3_XXS, generate, three starts each): a 32K prompt (about 1000
// rows an expert) 1209-1211 tok/s with the int8 form, 1249-1259 with this choice, 1259 with the FP16 form at every
// class; a 4K prompt (about 110) 949-952, 948-950 and 915-930.  The two give other last bits (FP16 products against
// exact int8 sums), so a class may answer otherwise from GPU to GPU, not from start to start (kept for the GPU).
// XESTRATA_IQ_MMQ_F16: unset, by the timing; 0, never; 1, at every class where it runs.
constexpr int kF16TopClass = 10, kF16FirstClass = 9;
int f16_env() {
    static const int v = [] {
        const char* e = strata::env_get("IQ_MMQ_F16");
        return e == nullptr || e[0] == '\0' ? -1 : (int) std::strtol(e, nullptr, 10) != 0;
    }();
    return v;
}
bool f16_first(sycl::queue& q, int ty, int64_t mean_rows, int n, int64_t N, int64_t K,
               const std::function<sycl::event()>& f16, const std::function<sycl::event()>& i8) {
    if (f16_env() >= 0) return f16_env() == 1;
    if (rows_class(mean_rows, kF16TopClass) < kF16FirstClass) return false;
    const bool timing = t_choosing;
    const bool pick = timed_first(q, ty, mean_rows, n, N, K, kF16TopClass, f16, i8);
    if (timing && pick)
        std::fprintf(stderr, "xestrata: the experts' int8 products of type %d (%lld x %lld) take the FP16 form at %lld rows "
                             "an expert (timed faster; XESTRATA_IQ_MMQ_F16=0 keeps the int8 one)\n",
                     ty, (long long) N, (long long) K, (long long) mean_rows);
    return pick;
}

// Whether the GPU runs kernel KT with its full work-group (NSG sub-groups of SG work-items) and its local memory,
// asked once a GPU.  The A380's 64 KiB of local memory did not hold the 128-row tile of a type with a scale per 16
// (two copies of the activations), and the launch lost the device instead of failing.
template<class KT>
bool takes_work_group(sycl::queue& q) {
    static core::PerDevice<bool> per_device;
    return per_device.get(q.get_device(), [&q] {
        if ((uint64_t) KT::LOCAL > q.get_device().get_info<sycl::info::device::local_mem_size>()) return false;
        try {
            const sycl::kernel_id id = sycl::get_kernel_id<KT>();
            const auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {q.get_device()}, {id});
            const size_t most = kb.get_kernel(id).template get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device());
            return most >= (size_t) KT::NSG * SG;
        } catch (const std::exception&) {
            return false;
        }
    });
}

// The fixed-point products' tiles: a sub-group's rows halved (twice the sub-groups for the tile), so its int64 sums fit
// its registers.  XESTRATA_IQ_MMQ_FX_SPLIT=0: the int8 products' shapes (an experiment, measured on the RX 8060S only)
bool fx_split() {
    static const bool on = [] {
        const char* v = strata::env_get("IQ_MMQ_FX_SPLIT");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

template<int TY>
sycl::event by_rows(sycl::queue& q, const void* w, const uint8_t* const* wtab, size_t woff, size_t expert_bytes,
                    int64_t w_rows, int64_t w_cols, int n, const void* xq, int64_t total_rows, const int32_t* bounds,
                    int64_t max_rows, const int32_t* ids, float* dst, int64_t ld_dst, int64_t rows) {
#define STRATA_MMQ_LAUNCH(SGM, NSGM, DP4A, TAIL, IS)                                                                 \
    launch<TY, SGM, NSGM, DP4A, TAIL, IS>(q, w, wtab, woff, expert_bytes, w_rows, w_cols, n, xq, total_rows, bounds, \
                                          max_rows, ids, dst, ld_dst)
#define STRATA_MMQ_ROWS(DP4A, TAIL, IS)                                                        \
    do {                                                                                        \
        if (max_rows <= 16) return STRATA_MMQ_LAUNCH(1, 1, DP4A, TAIL, IS);                     \
        if constexpr (fx_form<TY>())                                                            \
            if (fx_split()) {                                                                   \
                if (max_rows <= 32 || mean_rows <= 32) return STRATA_MMQ_LAUNCH(1, 2, DP4A, TAIL, IS); \
                if (takes_work_group<Kernel<TY, 1, 4, DP4A, TAIL, (IS)>>(q))                      \
                    return STRATA_MMQ_LAUNCH(1, 4, DP4A, TAIL, IS);                             \
            }                                                                                   \
        if (max_rows <= 32 || mean_rows <= 32) return STRATA_MMQ_LAUNCH(2, 1, DP4A, TAIL, IS);  \
        return STRATA_MMQ_LAUNCH(2, 2, DP4A, TAIL, IS);                                         \
    } while (0)
    // the tile's rows by the group's largest, but 32 where the experts average no more: a 64-row tile's work-groups
    // then multiply mostly empty rows, and the extra tiles of the few larger experts cost less (Ornith GSQ-RCO 3bit's
    // 515-token prompt, 16 rows an expert: RX 9060 XT +8%; its 4K prompt, 111 rows, keeps the 64-row tile)
    const int64_t mean_rows = rows / std::max(n, 1);
    // dp4a where the matrix form does not run (iq_mmq_grouped is called where one of the two does); the K-tail form
    // only for a K that is no multiple of the step, so that the others keep their loads unguarded
    const bool tail = w_cols % KC != 0;
    if (!iq_mmq_usable(q)) {
        if (tail) STRATA_MMQ_ROWS(true, true, 0);
        STRATA_MMQ_ROWS(true, false, 0);
    }
    // Intel's XMX: the 32-row tile's work-group (128 lanes) decodes the step's weights at half the 64-row one's rate,
    // and took longer for 32 rows than the 64-row tile for 64 on the B70; so 16 rows, else 64.  Where the experts
    // average more than 64 rows, a 128-row tile on a work-group of twice the sub-groups (each with the 64-row tile's
    // share, so nothing more in its registers: more rows a sub-group spilled them, 3x slower) decodes the weights
    // once for 128 rows: B70, Ornith Q4_K_M at 128 and 256 rows an expert 613 -> 495 and 1209 -> 988 us.  It needs
    // the kernel to take 512 work-items (the large register file halves the work-group's limit), asked of the GPU,
    // and 16-lane tiles (Xe2): on Xe-HPG's 8-lane ones its work-group is 64 sub-groups, and the A380 lost the device
    // on it, with the kernel's limits and local memory reported as enough.
#define STRATA_MMQ_ROWS_INTEL(TAIL, IS)                                                     \
    do {                                                                                    \
        if (max_rows <= 16) return STRATA_MMQ_LAUNCH(1, 1, false, TAIL, IS);                \
        if ((IS) == 1 && mean_rows > 64 && takes_work_group<Kernel<TY, 2, 4, false, (TAIL), (IS)>>(q)) \
            return STRATA_MMQ_LAUNCH(2, 4, false, TAIL, IS);                                \
        return STRATA_MMQ_LAUNCH(2, 2, false, TAIL, IS);                                    \
    } while (0)
#define STRATA_MMQ_ROWS_SG(IS)                                                                         \
    do {                                                                                               \
        if (max_rows <= 16) return STRATA_MMQ_LAUNCH_SG(1, 1, IS);                                     \
        if (max_rows <= 32 || mean_rows <= 32) return STRATA_MMQ_LAUNCH_SG(2, 1, IS);                  \
        return STRATA_MMQ_LAUNCH_SG(2, 2, IS);                                                         \
    } while (0)
#define STRATA_MMQ_LAUNCH_SG(SGM, NSGM, IS)                                                                          \
    launch<TY, SGM, NSGM, false, false, IS, true>(q, w, wtab, woff, expert_bytes, w_rows, w_cols, n, xq, total_rows, \
                                                  bounds, max_rows, ids, dst, ld_dst)
    // the sub-group form needs no K tail of its own (it walks the blocks of 32, not the steps), nor more local memory
    // than a few KiB
    if (const int is = intel_shape_of(q); is != 0) {
        const auto wg = [&]() -> sycl::event {
            if (is == 1) {
                if (tail) STRATA_MMQ_ROWS_INTEL(true, 1);
                STRATA_MMQ_ROWS_INTEL(false, 1);
            }
            if (tail) STRATA_MMQ_ROWS_INTEL(true, 2);
            STRATA_MMQ_ROWS_INTEL(false, 2);
        };
        const auto sg = [&]() -> sycl::event {
            if (is == 1) STRATA_MMQ_ROWS_SG(1);
            STRATA_MMQ_ROWS_SG(2);
        };
        return intel_sub_group_first(q, TY, mean_rows, n, w_rows, w_cols, sg, wg) ? sg() : wg();
    }
#undef STRATA_MMQ_ROWS_SG
#undef STRATA_MMQ_LAUNCH_SG
#undef STRATA_MMQ_ROWS_INTEL
    // RDNA3 and RDNA4: an expert group's product in FP16 where it was timed faster for its class of rows (f16_first)
    if (n > 1 && f16_env() != 0 && iq_mmq_f16_ok(q, TY, w_rows, w_cols)) {
        const auto f16 = [&] {
            return iq_mmq_f16_grouped(q, TY, w, expert_bytes, w_rows, w_cols, n, xq, total_rows, bounds, max_rows, ids,
                                      dst, ld_dst, wtab, woff);
        };
        const auto i8 = [&]() -> sycl::event {
            if (tail) STRATA_MMQ_ROWS(false, true, 0);
            STRATA_MMQ_ROWS(false, false, 0);
        };
        return f16_first(q, TY, mean_rows, n, w_rows, w_cols, f16, i8) ? f16() : i8();
    }
    if (tail) STRATA_MMQ_ROWS(false, true, 0);
    STRATA_MMQ_ROWS(false, false, 0);
#undef STRATA_MMQ_ROWS
#undef STRATA_MMQ_LAUNCH
}

bool shape_reported(const sycl::device& d) {
    if (!xe::has_sub_group(d, (size_t) SG)) return false;
    for (const auto& c : matrix_combinations(d)) {
        const bool m = c.msize == (size_t) TM || (c.msize == 0 && c.max_msize >= (size_t) TM);
        const bool n = c.nsize == (size_t) TN || (c.nsize == 0 && c.max_nsize >= (size_t) TN);
        const bool k = c.ksize == (size_t) TK || (c.ksize == 0 && c.max_ksize >= (size_t) TK);
        if (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 &&
            c.ctype == mx::matrix_type::sint32 && c.dtype == mx::matrix_type::sint32 && m && n && k)
            return true;
    }
    return false;
}

// The positions a lane's int32 accumulator elements take against those of its FP32 one, walked together as the
// products walk them: 1 when every pair agrees.
struct LayoutCheck {
    sycl::local_accessor<int32_t, 1> pi;
    sycl::local_accessor<float, 1> pf;
    int* out;
    void operator()(sycl::nd_item<1> it) const {
        if constexpr (!built<0>()) {
            (void) it;
        } else {
#if STRATA_IQ_MMQ_MATRIX
            const auto sg = it.get_sub_group();
            const int lid = (int) it.get_local_id(0);
            for (int i = lid; i < TM * TN; i += SG) { pi[i] = i; pf[i] = (float) i; }
            sycl::group_barrier(it.get_group());
            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, TM, TN> a;
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> b;
            mx::joint_matrix_load(sg, a, pi.template get_multi_ptr<sycl::access::decorated::no>(), TN, mx::layout::row_major);
            mx::joint_matrix_load(sg, b, pf.template get_multi_ptr<sycl::access::decorated::no>(), TN, mx::layout::row_major);
            int bad = 0, n = 0;
            mx::joint_matrix_apply(sg, b, a, [&](float& f, int32_t& v) { bad |= (int) f != v; ++n; });
            bad |= n != NE;
            bad = sycl::reduce_over_group(sg, bad, sycl::bit_or<int>());
            if (lid == 0) *out = bad ? 0 : 1;
#endif
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>}; }
};


}  // namespace

bool iq_mmq_usable(sycl::queue& q) {
    static core::PerDevice<bool> per_device;
    return per_device.get(q.get_device(), [&q] {
        if (no_xmx()) return false;
        const sycl::device d = q.get_device();
        const bool wmma = xe::amd_wmma(d);   // its int8 WMMA: the STRATA_IQ_MMQ_WMMA body
        if (!wmma && !shape_reported(d)) return intel_shape_of(q) != 0;   // Intel's XMX shapes (intel_body)
        using Big = Kernel<16, 2, 2>;
        if (d.get_info<sycl::info::device::local_mem_size>() < (size_t) Big::LOCAL ||
            d.get_info<sycl::info::device::max_work_group_size>() < (size_t) Big::NSG * SG)
            return false;
        if (wmma) return true;   // its layout is the WMMA instructions' own, written out in the kernel
        int* out = sycl::malloc_device<int>(1, q);
        if (out == nullptr) return false;
        int res = 0;
        try {
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<int32_t, 1> pi(sycl::range<1>((size_t) TM * TN), h);
                sycl::local_accessor<float, 1> pf(sycl::range<1>((size_t) TM * TN), h);
                h.parallel_for(sycl::nd_range<1>(SG, SG), LayoutCheck{pi, pf, out});
            });
            q.memcpy(&res, out, sizeof(int)).wait();
        } catch (const sycl::exception& e) {
            std::fprintf(stderr, "xestrata: the int8 expert products' layout check did not run (%s)\n", e.what());
            res = 0;
        }
        sycl::free(out, q);
        if (!res) std::fprintf(stderr, "xestrata: the GPU's int8 and FP32 accumulators differ in layout: the experts' "
                               "prompt products stay in FP16\n");
        return res == 1;
    });
}

bool iq_mmq_dp4a_usable(sycl::queue& q) {
    static core::PerDevice<bool> per_device;
    return per_device.get(q.get_device(), [&q] {
        const sycl::device d = q.get_device();
        using Big = Kernel<16, 2, 2, true>;
        return d.get_info<sycl::info::device::local_mem_size>() >= (size_t) Big::LOCAL &&
               d.get_info<sycl::info::device::max_work_group_size>() >= (size_t) Big::NSG * SG;
    });
}

const char* iq_mmq_path(sycl::queue& q) {
    if (iq_mmq_usable(q)) {
        const sycl::device d = q.get_device();
        return xe::amd_wmma(d) ? "WMMA" : shape_reported(d) ? "joint_matrix" : "XMX";
    }
    return iq_mmq_dp4a_usable(q) ? "DP4a" : nullptr;
}

bool iq_mmq_type_sums(int t) { return t == 7 || t == 12 || t == 13; }   // has_min

bool iq_mmq_type_ok(int t) {
    switch (t) {
        case 2: case 6: case 7: case 8: case 10: case 11: case 12: case 13: case 14: case kNativeQ6KRows: case kNativeQ8Rows:
        case 16: case 17: case 18: case 19: case 20: case 21: case 22: case 23: case 29: case 40: case 42:
        case kIqMmqFx + 16: case kIqMmqFx + 17: case kIqMmqFx + 18: case kIqMmqFx + 20: case kIqMmqFx + 21:
        case kIqMmqFx + 22: case kIqMmqFx + 42: return true;
        default: return false;
    }
}

int iq_mmq_fx(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 42 ? kIqMmqFx + t : 0;
}

size_t iq_mmq_row_bytes(int t, int64_t cols) {
    switch (t) {
        case 6: return (size_t) (cols / Dec<6>::QK) * Dec<6>::BSZ;
        case 7: return (size_t) (cols / Dec<7>::QK) * Dec<7>::BSZ;
        case 10: return (size_t) (cols / 256) * Dec<10>::BSZ;   // the row's own values (read twice as long)
        case 11: return (size_t) (cols / Dec<11>::QK) * Dec<11>::BSZ;
        case 12: return (size_t) (cols / Dec<12>::QK) * Dec<12>::BSZ;
        case 13: return (size_t) (cols / Dec<13>::QK) * Dec<13>::BSZ;
        case 2: return (size_t) (cols / Dec<2>::QK) * Dec<2>::BSZ;
        case 8: return (size_t) (cols / Dec<8>::QK) * Dec<8>::BSZ;
        case 14: return (size_t) (cols / Dec<14>::QK) * Dec<14>::BSZ;
        case kNativeQ6KRows: return (size_t) (cols / Dec<kNativeQ6KRows>::QK) * Dec<kNativeQ6KRows>::BSZ;
        case kNativeQ8Rows: return (size_t) (cols / Dec<kNativeQ8Rows>::QK) * Dec<kNativeQ8Rows>::BSZ;
        case 16: case kIqMmqFx + 16: return (size_t) (cols / Dec<16>::QK) * Dec<16>::BSZ;
        case 17: case kIqMmqFx + 17: return (size_t) (cols / Dec<17>::QK) * Dec<17>::BSZ;
        case 18: case kIqMmqFx + 18: return (size_t) (cols / Dec<18>::QK) * Dec<18>::BSZ;
        case 20: case kIqMmqFx + 20: return (size_t) (cols / Dec<20>::QK) * Dec<20>::BSZ;
        case 21: case kIqMmqFx + 21: return (size_t) (cols / Dec<21>::QK) * Dec<21>::BSZ;
        case 22: case kIqMmqFx + 22: return (size_t) (cols / Dec<22>::QK) * Dec<22>::BSZ;
        case 23: return (size_t) (cols / Dec<23>::QK) * Dec<23>::BSZ;
        case 29: return (size_t) (cols / Dec<29>::QK) * Dec<29>::BSZ;
        case 19: return (size_t) (cols / Dec<19>::QK) * Dec<19>::BSZ;
        case 42: case kIqMmqFx + 42: return (size_t) (cols / Dec<42>::QK) * Dec<42>::BSZ;
        case 40: return (size_t) (cols / Dec<40>::QK) * Dec<40>::BSZ;
        default: return 0;
    }
}

size_t iq_mmq_act_bytes(int64_t rows, int64_t cols) { return (size_t) (rows * cols + 2 * rows * (cols / 32) * 4); }

bool iq_mmq_type_dup(int t) { return t == 10; }

sycl::event iq_mmq_quantize(sycl::queue& q, const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld,
                            int64_t rows, bool sums, bool dup, int canon, const float* col_scale) {
    // dup (Q2_K, Dec<10>): each block written twice, as a row of 2 cols values.  8 values a lane (a block 4 lanes):
    // 8,192 rows of 3,072 took the B70 264 us and the RTX 4070 285 where a block a work-item took 299 and 296, and a
    // block a sub-group (a value a lane) 626 and 394 (2026-11-05)
    const int64_t oc = dup ? 2 * cols : cols, nb = cols / 32;
    auto* x8 = static_cast<int8_t*>(xq);
    auto* xs = reinterpret_cast<float*>(x8 + rows * oc);
    float* xu = xs + rows * (oc / 32);
    auto run = [&](const auto& src) {
        if (canon != 0) {   // the fixed-point sums' (iq_mmq_fx): the CPU's canonical rows' values and fp16 scales
            q8act::Rows<false, false, true> out{x8, xs, xu, oc, oc / 32, 1};
            if (canon == 32) return q8act::quantize<8, q8act::Canon<32>>(q, rows * nb, nb, src, out);
            return q8act::quantize<8, q8act::Canon<256>>(q, rows * nb, nb, src, out);
        }
        if (sums) {
            q8act::Rows<true> out{x8, xs, xu, oc, oc / 32, dup ? 2 : 1};
            return q8act::quantize<8, q8act::Mul>(q, rows * nb, nb, src, out);
        }
        q8act::Rows<false> out{x8, xs, xu, oc, oc / 32, dup ? 2 : 1};
        return q8act::quantize<8, q8act::Mul>(q, rows * nb, nb, src, out);
    };
    if (col_scale != nullptr) return run(q8act::ScaledFloatRows{x, ids, ld, col_scale});
    return run(q8act::FloatRows{x, ids, ld});
}

sycl::event iq_mmq_quantize_scatter(sycl::queue& q, const float* x, const int32_t* slot, void* xq, int64_t cols,
                                    int64_t ld, int64_t tokens, int k_used, bool sums, bool dup, int canon) {
    // as iq_mmq_quantize, with the rows reached from the token: the block of token t goes to its k_used rows
    const int64_t oc = dup ? 2 * cols : cols, rows = tokens * k_used, nb = cols / 32;
    auto* x8 = static_cast<int8_t*>(xq);
    auto* xs = reinterpret_cast<float*>(x8 + rows * oc);
    float* xu = xs + rows * (oc / 32);
    const q8act::FloatRows src{x, nullptr, ld};
    if (canon != 0) {
        q8act::Rows<false, true, true> out{x8, xs, xu, oc, oc / 32, 1, slot, k_used};
        if (canon == 32) return q8act::quantize<8, q8act::Canon<32>>(q, tokens * nb, nb, src, out);
        return q8act::quantize<8, q8act::Canon<256>>(q, tokens * nb, nb, src, out);
    }
    if (sums) {
        q8act::Rows<true, true> out{x8, xs, xu, oc, oc / 32, dup ? 2 : 1, slot, k_used};
        return q8act::quantize<8, q8act::Mul>(q, tokens * nb, nb, src, out);
    }
    q8act::Rows<false, true> out{x8, xs, xu, oc, oc / 32, dup ? 2 : 1, slot, k_used};
    return q8act::quantize<8, q8act::Mul>(q, tokens * nb, nb, src, out);
}

sycl::event iq_mmq_grouped(sycl::queue& q, int t, const void* w, size_t expert_bytes, int64_t w_rows, int64_t w_cols,
                           int n, const void* xq, int64_t total_rows, const int32_t* bounds, int64_t max_rows,
                           const int32_t* ids, float* dst, int64_t ld_dst, const uint8_t* const* wtab, size_t woff,
                           int64_t rows) {
#define STRATA_MMQ_CASE(TY)                                                                                           \
    case TY:                                                                                                          \
        return by_rows<TY>(q, w, wtab, woff, expert_bytes, w_rows, w_cols, n, xq, total_rows, bounds, max_rows, ids, \
                           dst, ld_dst, rows > 0 ? rows : total_rows)
    switch (t) {
        STRATA_MMQ_CASE(2);
        STRATA_MMQ_CASE(6);
        STRATA_MMQ_CASE(7);
        STRATA_MMQ_CASE(8);
        STRATA_MMQ_CASE(10);
        STRATA_MMQ_CASE(11);
        STRATA_MMQ_CASE(12);
        STRATA_MMQ_CASE(13);
        STRATA_MMQ_CASE(14);
        STRATA_MMQ_CASE(kNativeQ6KRows);
        STRATA_MMQ_CASE(kNativeQ8Rows);
        STRATA_MMQ_CASE(16);
        STRATA_MMQ_CASE(17);
        STRATA_MMQ_CASE(18);
        STRATA_MMQ_CASE(kIqMmqFx + 16);
        STRATA_MMQ_CASE(kIqMmqFx + 17);
        STRATA_MMQ_CASE(kIqMmqFx + 18);
        STRATA_MMQ_CASE(kIqMmqFx + 20);
        STRATA_MMQ_CASE(kIqMmqFx + 21);
        STRATA_MMQ_CASE(kIqMmqFx + 22);
        STRATA_MMQ_CASE(kIqMmqFx + 42);
        STRATA_MMQ_CASE(20);
        STRATA_MMQ_CASE(21);
        STRATA_MMQ_CASE(22);
        STRATA_MMQ_CASE(23);
        STRATA_MMQ_CASE(29);
        STRATA_MMQ_CASE(19);
        STRATA_MMQ_CASE(42);
        STRATA_MMQ_CASE(40);
        default: return {};
    }
#undef STRATA_MMQ_CASE
}


void iq_mmq_choose_forms(sycl::queue& q, int t, int64_t w_rows, int64_t w_cols, int n) {
    if (!iq_mmq_type_ok(t)) return;
    if (intel_shape_of(q) != 0) {   // Intel's two forms (intel_sub_group_first)
        if (const char* env = strata::env_get("IQ_MMQ_SUB_GROUP"); env != nullptr && (env[0] == '0' || env[0] == '1'))
            return;
    } else if (f16_env() >= 0 || std::max(n, 1) == 1 || !iq_mmq_f16_ok(q, t, w_rows, w_cols)) {   // f16_first
        return;   // one form only
    }
    // the classes timed: Intel's forms 8 to 256 rows an expert, the FP16 form 512 and 1024 (f16_first)
    const bool intel = intel_shape_of(q) != 0;
    const int top = intel ? 8 : kF16TopClass;
    const int64_t kTop = (int64_t) 1 << top, kFirst = intel ? 8 : (int64_t) 1 << kF16FirstClass;
    n = std::max(n, 1);
    // the buffers hold n experts of 256 rows: a class of more rows takes fewer experts (at least 2), so that the
    // timing takes no more memory than Intel's (taken with the engine's caches already in place)
    const int64_t cols = iq_mmq_type_dup(t) ? 2 * w_cols : w_cols, R = n * std::min<int64_t>(kTop, 256);
    {
        SgChoice& c = sg_choice(q);
        std::lock_guard<std::mutex> lock(c.mu);
        if (c.picks.count(std::make_tuple(t, rows_class(kTop, top), n == 1, w_rows, cols)) != 0) return;   // timed before
    }
    const size_t eb = (size_t) w_rows * iq_mmq_row_bytes(t, w_cols);
    // the products' buffers: n experts' weights (any bits are valid blocks; the values do not change the time), the
    // rows' int8 activations (zeros) and their outputs
    auto* w = sycl::malloc_device<uint8_t>(eb * (size_t) n, q);
    auto* xq = sycl::malloc_device<uint8_t>(iq_mmq_act_bytes(R, cols), q);
    auto* dst = sycl::malloc_device<float>((size_t) (R * w_rows), q);
    auto* bounds = sycl::malloc_device<int32_t>((size_t) n + 1, q);
    auto* ids = sycl::malloc_device<int32_t>((size_t) R, q);
    if (w != nullptr && xq != nullptr && dst != nullptr && bounds != nullptr && ids != nullptr) {
        t_choosing = true;
        try {
            q.memset(w, 0x11, eb * (size_t) n);
            q.memset(xq, 0, iq_mmq_act_bytes(R, cols));
            q.parallel_for(sycl::range<1>((size_t) R), [=](sycl::id<1> i) { ids[i] = (int32_t) i[0]; });
            std::vector<int32_t> hb((size_t) n + 1);
            for (int64_t r = kFirst; r <= kTop; r *= 2) {
                const int nr = n == 1 ? 1 : (int) std::max<int64_t>(2, std::min<int64_t>(n, R / r));
                for (int e = 0; e <= nr; ++e) hb[(size_t) e] = (int32_t) (e * r);
                q.memcpy(bounds, hb.data(), (size_t) (nr + 1) * sizeof(int32_t)).wait();
                if ((int64_t) nr * r <= R)
                    iq_mmq_grouped(q, t, w, eb, w_rows, cols, nr, xq, nr * r, bounds, r, ids, dst, w_rows, nullptr, 0,
                                   nr * r);
            }
            q.wait_and_throw();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xestrata: timing the int8 expert products' forms: %s\n", e.what());
        }
        t_choosing = false;
    } else {
        std::fprintf(stderr, "xestrata: no device memory to time the int8 expert products' forms: the work-group form\n");
    }
    for (void* p : {(void*) w, (void*) xq, (void*) dst, (void*) bounds, (void*) ids})
        if (p != nullptr) sycl::free(p, q);
}

bool iq_mmq_split_reserve(sycl::queue& q, size_t floats) {
    SplitState& st = split_state(q);
    std::lock_guard<std::mutex> lock(st.mu);
    auto& b = st.bufs[&q];
    if (b.second >= floats) return true;
    q.wait();
    if (b.first != nullptr) {
        sycl::free(b.first, q);
        core::graph_epoch_bump();   // a recorded step may hold the old one
    }
    b.first = sycl::malloc_device<float>(floats, q);
    b.second = b.first != nullptr ? floats : 0;
    return b.first != nullptr;
}

}  // namespace strata::kernels
