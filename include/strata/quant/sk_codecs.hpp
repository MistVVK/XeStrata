// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/quant/sk_codecs.hpp - Native .strata v1.0 S_K codecs & AVX-512 VNNI / AVX2 / Scalar kernels
// for Hy4-Flash square [1536, 1536] LatentMoE experts (256-weight super-blocks, 4 x 64-weight sub-groups).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::quant::sk {

constexpr int QK_S = 256;
constexpr int SUB_G = 64;
constexpr int NUM_SUB = 4;

enum class SkCode : uint8_t {
    S1_25 = 0,  // 48 B  (6 B hdr + 40 B 4D-lattice 5-bit indices + 2 B fine nibble multipliers)
    S1_5  = 1,  // 58 B  (6 B hdr + 52 B base-3 ternary, 5 trits/byte)
    S2_K  = 2,  // 70 B  (6 B hdr + 64 B 2-bit symmetric half-integer q in 0..3)
    S3_K  = 3,  // 102 B (6 B hdr + 96 B 3-bit symmetric half-integer q in 0..7)
    S4_K  = 4,  // 134 B (6 B hdr + 128 B 4-bit symmetric half-integer q in 0..15)
    S5_K  = 5,  // 166 B (6 B hdr + 160 B 5-bit symmetric half-integer q in 0..31)
    S6_K  = 6,  // 198 B (6 B hdr + 192 B 6-bit symmetric half-integer q in 0..63)
    S8_K  = 8,  // 262 B (6 B hdr + 256 B 8-bit signed int8 q in [-127..127])
    SF4      = 15, // Lossless NVFP4: E2M1 nibbles + F8_E4M3/16w block scales + F32 amax/denom (.strata v1.1)
    MX8      = 16, // Lossless MXFP8: F8_E4M3 weights + UE8M0/32w block scales (.strata v1.1)
    GGUF_RAW = 20, // Lossless native GGUF block expert body (Q8_0/Q4_K/Q5_K/Q6_K/IQ3_S/IQ4_NL/IQ4_XS/MXFP4)
};

// 32-byte aligned header prepended to SkCode::GGUF_RAW expert bodies
struct GgufExpertHeader {
    uint32_t magic;       // 0x47475546u ("GGUF")
    uint32_t gu_type;     // ggml_type for gate/up weights
    uint32_t dn_type;     // ggml_type for down weights
    uint32_t flags;       // bit 0: 1 = interleaved (even=gate, odd=up), 0 = split ([gate rows 0..F-1; up rows F..2F-1])
    float    gate_scale;  // per-expert scale for gate (1.0f if absent)
    float    up_scale;    // per-expert scale for up (1.0f if absent)
    float    dn_scale;    // per-expert scale for down (1.0f if absent)
    uint32_t reserved;    // padding to 32 bytes
};
static_assert(sizeof(GgufExpertHeader) == 32, "GgufExpertHeader must be 32 B");

constexpr size_t superblock_bytes(SkCode code) noexcept {
    switch (code) {
        case SkCode::S1_25: return 48;
        case SkCode::S1_5:  return 58;
        case SkCode::S2_K:  return 70;
        case SkCode::S3_K:  return 102;
        case SkCode::S4_K:  return 134;
        case SkCode::S5_K:  return 166;
        case SkCode::S6_K:  return 198;
        case SkCode::S8_K:  return 262;
        default:            return 0;
    }
}

struct alignas(64) BlockQ8_1_G64 {
    int8_t qs[64];
    float d;
    float s;  // d * sum(qs)
};

// Quantize float32 vector [n] (n % 64 == 0) into 64-element Q8_1 blocks
void quantize_q8_1_g64(const float* x, BlockQ8_1_G64* out, size_t n) noexcept;

// Apply normalized 64-wide Block-Hadamard transform (s_xform=2) in-place or out-of-place
void fwht64_f32(const float* in, float* out, size_t n) noexcept;

// Exact dequantization of n weights (n % 256 == 0) from S_K wire bytes -> float32
void dequant_sk_row(const uint8_t* raw, SkCode code, float* out, size_t n) noexcept;

// Exact quantized dot product of 1 S_K row with Q8_1_G64 activations (dispatches AVX-512 VNNI -> AVX2 -> scalar)
float dot_sk_q8_1_row(const uint8_t* raw, SkCode code, const BlockQ8_1_G64* x, size_t n) noexcept;

// Multi-token GEMM for m tokens x nrows square [1536, 1536] expert matrix (maximizes weight reuse in L1/L2)
void gemm_sk_q8_1(
    const uint8_t* weights,
    SkCode code,
    const BlockQ8_1_G64* xs,  // [m, n/64]
    float* out,               // [m, nrows]
    size_t nrows,
    size_t ncols,
    size_t m) noexcept;

// Runtime ISA name ("avx512-vnni", "avx2", "neon-dotprod", or "scalar")
const char* active_sk_isa() noexcept;

// --- .strata v1.1 Lossless Native Codecs (SF4=NVFP4 & MX8=MXFP8) & Paired 2-Row Down (SB_0..SB_8) ---

// Decode 1 row of ncols (ncols % 16 == 0) SF4 (NVFP4) weights:
// packed_e2m1: [ncols/2] uint8 (low nibble = even col, high nibble = odd col)
// scales_e4m3: [ncols/16] uint8 (F8_E4M3FN)
// outer_scale = amax / denom
void dequant_sf4_row(
    const uint8_t* packed_e2m1,
    const uint8_t* scales_e4m3,
    float outer_scale,
    float* out,
    size_t ncols) noexcept;

// Fast dot product of 1 SF4 row against float32 x (uses 1-instruction pshufb/vtbl register LUT on AVX512/AVX2/NEON)
float dot_sf4_f32_row(
    const uint8_t* packed_e2m1,
    const uint8_t* scales_e4m3,
    float outer_scale,
    const float* x,
    size_t ncols) noexcept;

// Decode 1 row of ncols (ncols % 32 == 0) MX8 (MXFP8) weights:
// weights_e4m3: [ncols] uint8 (F8_E4M3FN)
// scales_ue8m0: [ncols/32] uint8 (2^(e - 127))
void dequant_mx8_row(
    const uint8_t* weights_e4m3,
    const uint8_t* scales_ue8m0,
    float* out,
    size_t ncols) noexcept;

float dot_mx8_f32_row(
    const uint8_t* weights_e4m3,
    const uint8_t* scales_ue8m0,
    const float* x,
    size_t ncols) noexcept;

// Paired 2-row S_K decode/GEMV/GEMM for ncols % 256 == 128 (e.g. Inkling 2 Mini w2 [2304, 1152]):
// Each pair (2r, 2r+1) packs 2 * ncols weights into (2*ncols)/256 exact super-blocks (SB_0..SB_8).
void dequant_paired2_sk_rows(
    const uint8_t* pair_raw,
    SkCode code,
    float* out_row0,
    float* out_row1,
    size_t ncols) noexcept;

void gemv_paired2_sk_q8_1(
    const uint8_t* weights,
    SkCode code,
    const BlockQ8_1_G64* x,
    float* out,
    size_t nrows,  // even
    size_t ncols) noexcept;

void gemm_paired2_sk_q8_1(
    const uint8_t* weights,
    SkCode code,
    const BlockQ8_1_G64* xs,  // [m, ncols/64]
    float* out,               // [m, nrows]
    size_t nrows,             // even
    size_t ncols,
    size_t m) noexcept;

}  // namespace strata::quant::sk
