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
};

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

// Runtime ISA name ("avx512-vnni", "avx2", or "scalar")
const char* active_sk_isa() noexcept;

}  // namespace strata::quant::sk
