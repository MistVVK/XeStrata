// Copyright (c) 2026 Strata contributors
// SPDX-License-Identifier: MIT
//
// include/strata/kernels/hip_unsloth_dp4a.hpp - Clean-room AMD Wave32 DP4A & fused EXP_MODE=8
// primitives and launchers for RDNA2 (gfx1030..1034), RDNA3/3.5 (gfx1100..1151), and RDNA4 (gfx1200/1201).
//
// Covers all primary Unsloth GGUF quants:
//   Q4_K (12), Q5_K (13), Q6_K (14), Q8_0 (8), IQ4_XS (23), IQ3_XXS (18), NVFP4 (40).
//
// Key RDNA2/3/4 improvements over generic HIP compat paths:
//   1. Direct 6-instruction `v_perm_b32` 16-entry LUT lookup on ALL RDNA2/3/4 (replaces 8x emulated
//      __byte_perm in IQ4_XS, IQ4_NL, and NVFP4).
//   2. Direct 3-instruction `hip_iq_sign_negate` (replaces emulated __vcmpne4 + __vsub4 in IQ3_XXS).
//   3. Branchless 2-instruction `hip_sub32_bytes` (replaces emulated __vsubss4 in Q6_K).
//   4. Align-2 safe `__builtin_memcpy` 32-bit loads on RDNA4 (gfx1200/1201).
//   5. Unified hoisted-decode `HipSplit<TY>` across all 7 formats (including Q6_K which lacked Split<>).
//   6. Fused EXP_MODE=8 gate+up+SwiGLU+q8_1 & LDS down kernels with:
//        - strict `had_block == 0` guard (prevents corrupting Hadamard-folded down weights)
//        - exact `s_gate`, `s_up`, `s_down` NVFP4 tail scales on every path
//        - 65535-safe gridDim.y group striding
//        - Wave32 DPP warp reduction (`update_dpp` + `permlanex16`).
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#endif

namespace strata::kernels::hip_unsloth {

/// Minimal self-contained layout descriptor matching Strata's NativeExpertLayout binary layout.
struct HipExpertLayout {
    int gu_type = -1;
    int d_type = -1;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    std::size_t gu_row = 0;
    std::size_t d_row = 0;
    std::size_t up_off = 0;
    std::size_t down_off = 0;
    std::size_t bytes = 0;
    std::size_t tail_off = 0;  // NVFP4: {s_gate, s_up, s_down, 0} at blob + tail_off; 0 = none
    int had_block = 0;         // >0 means Hadamard-folded down weights; fused EXP_MODE=8 MUST decline
};

/// Bytes of one row of `n` elements for supported Unsloth quant types (0 if unsupported/misaligned).
inline std::size_t hip_quant_row_bytes(int ggml_type, int64_t n) noexcept {
    switch (ggml_type) {
        case 8:  return (n % 32 == 0)  ? std::size_t(n / 32)  * 34u  : 0u;  // Q8_0
        case 12: return (n % 256 == 0) ? std::size_t(n / 256) * 144u : 0u;  // Q4_K
        case 13: return (n % 256 == 0) ? std::size_t(n / 256) * 176u : 0u;  // Q5_K
        case 14: return (n % 256 == 0) ? std::size_t(n / 256) * 210u : 0u;  // Q6_K
        case 18: return (n % 256 == 0) ? std::size_t(n / 256) * 98u  : 0u;  // IQ3_XXS
        case 23: return (n % 256 == 0) ? std::size_t(n / 256) * 136u : 0u;  // IQ4_XS
        case 40: return (n % 64 == 0)  ? std::size_t(n / 64)  * 36u  : 0u;  // NVFP4
        default: return 0u;
    }
}

/// Build an expert layout for [gate | up | down | optional NVFP4 tail].
inline HipExpertLayout hip_make_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff,
                                              int had_block = 0) noexcept {
    HipExpertLayout L{};
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = hip_quant_row_bytes(gu_type, n_embd);
    L.d_row = hip_quant_row_bytes(d_type, n_ff);
    L.up_off = std::size_t(n_ff) * L.gu_row;
    L.down_off = 2u * L.up_off;
    L.bytes = L.down_off + std::size_t(n_embd) * L.d_row;
    if (gu_type == 40 || d_type == 40) {
        L.tail_off = L.bytes;
        L.bytes += 16u;
    }
    L.had_block = had_block;
    return L;
}

#if defined(__HIPCC__)

// ===========================================================================
// RDNA2 / RDNA3 / RDNA3.5 / RDNA4 Wave32 device primitives
// ===========================================================================

__device__ __forceinline__ int hip_dp4a(int a, int b, int c) {
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
     defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)) && \
    __has_builtin(__builtin_amdgcn_sudot4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1034__)) && \
      __has_builtin(__builtin_amdgcn_sdot4)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    uint32_t sum = static_cast<uint32_t>(c);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const int sa = static_cast<int8_t>((ua >> (lane * 8)) & 0xffu);
        const int sb = static_cast<int8_t>((ub >> (lane * 8)) & 0xffu);
        sum += static_cast<uint32_t>(sa * sb);
    }
    return static_cast<int>(sum);
#endif
}

// Align-2 safe 32-bit load: on RDNA4 (gfx1200/1201) LLVM lowers __builtin_memcpy to a single
// unaligned 32-bit flat/global load; on older targets two 16-bit loads avoid UB/misalignment traps.
__device__ __forceinline__ int hip_ld_b2(const void* ptr, int i32) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    int v;
    __builtin_memcpy(&v, static_cast<const uint16_t*>(ptr) + 2 * i32, sizeof(int));
    return v;
#else
    const uint16_t* p = static_cast<const uint16_t*>(ptr);
    return static_cast<int>(static_cast<uint32_t>(p[2 * i32]) |
                            (static_cast<uint32_t>(p[2 * i32 + 1]) << 16));
#endif
}

__device__ __forceinline__ int hip_ld_b4(const void* ptr, int i32) {
    return static_cast<const int*>(ptr)[i32];
}

// Wave32 DPP butterfly sum (5 instructions on RDNA2/3/4, zero LDS traffic).
__device__ __forceinline__ float hip_wsum32(float v) {
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1034__) || \
    defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x161, 0xf, 0xf, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x162, 0xf, 0xf, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x164, 0xf, 0xf, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x168, 0xf, 0xf, false));
    v += __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210, 0xfedcba98, false, false));
    return v;
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
#endif
}

__device__ __forceinline__ float hip_wmax32(float v) {
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1034__) || \
    defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x161, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x162, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x164, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x168, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210, 0xfedcba98, false, false)));
    return v;
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    return v;
#endif
}

// 6-instruction v_perm_b32 16-entry LUT lookup for 8 packed nibbles in `q4` (even nibbles -> .x, odd -> .y).
// Replaces 8x emulated __byte_perm (~40 ops) on RDNA2/3/4 for IQ4_XS, IQ4_NL, and NVFP4.
__device__ __forceinline__ int2 hip_vperm_lut16(uint32_t q4, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    const uint32_t q_even = q4;
    const uint32_t q_odd = q4 >> 4;
    const uint32_t sel_e = q_even & 0x07070707u;
    const uint32_t sel_o = q_odd & 0x07070707u;
    const uint32_t el = __builtin_amdgcn_perm(t1, t0, sel_e);
    const uint32_t ol = __builtin_amdgcn_perm(t1, t0, sel_o);
    const uint32_t eh = __builtin_amdgcn_perm(t3, t2, sel_e);
    const uint32_t oh = __builtin_amdgcn_perm(t3, t2, sel_o);
    const uint32_t vx = __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1));
    const uint32_t vy = __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1));
    return make_int2(static_cast<int>(vx), static_cast<int>(vy));
}

// IQ4_NL / IQ4_XS codebook: {-127,-104,-83,-65, -49,-35,-22,-10, 1,13,25,38, 53,69,89,113}
__device__ __forceinline__ int2 hip_iq4_lut16(uint32_t q4) {
    return hip_vperm_lut16(q4, 0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u);
}

// NVFP4 doubled E2M1 codebook: {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
__device__ __forceinline__ int2 hip_fp4_lut16(uint32_t q4) {
    return hip_vperm_lut16(q4, 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu);
}

// 3-instruction sign expansion & bytewise two's complement for IQ3_XXS / IQ3_S / IQ2_S
// (all grid bytes are strictly positive in [1..62], so + t never carries across byte boundaries).
__device__ __forceinline__ int hip_iq_sign_negate(uint32_t grid, uint32_t s4) {
    const uint32_t t = ((s4 & 0xFu) * 0x00204081u) & 0x01010101u;
    return static_cast<int>((grid ^ (t * 0xFFu)) + t);
}

// 2-instruction bytewise (x - 32) for Q6_K codes in [0..63] (replaces 8-op software __vsubss4).
__device__ __forceinline__ int hip_sub32_bytes(uint32_t x) {
    return static_cast<int>((x + 0x60606060u) ^ 0x80808080u);
}

// Exact UE4M3 / 2 -> FP32 decoder for NVFP4 (matches ggml_cuda_ue4m3_to_fp32 * 0.5 bit-for-bit).
__device__ __forceinline__ float hip_ue4m3_to_f32_half(uint8_t x) {
    if (x == 0u || x == 0x7Fu) return 0.0f;
    const uint32_t exp = (x >> 3) & 0xFu;
    const uint32_t man = x & 0x7u;
    return exp == 0u ? static_cast<float>(man) * 0.0009765625f
                     : __uint_as_float(((exp + 119u) << 23) | (man << 20));
}

// Finite-clamped q8_1 helpers matching strata/kernels/q8_1_finite.hpp bit-for-bit.
__device__ __forceinline__ float hip_q8_1_finite(float v) {
    constexpr float kHMax = 65504.0f;
    return fabsf(v) > kHMax ? copysignf(kHMax, v) : v;
}

__device__ __forceinline__ int8_t hip_q8_1_quant(float xi, float d, float amax) {
    if (amax == 0.0f) return 0;
    const float q = roundf(xi / d);
    return static_cast<int8_t>(q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

__device__ __forceinline__ __half2 hip_q8_1_ds(float d, float sum) {
    return __halves2half2(__float2half(hip_q8_1_finite(d)), __float2half(hip_q8_1_finite(sum)));
}

#endif  // defined(__HIPCC__)

// ===========================================================================
// Host API (implemented in src/kernels/hip/hip_unsloth_dp4a.cu)
// ===========================================================================

/// Wave32 DP4A multi-column decode GEMV for Q4_K (12), Q5_K (13), Q6_K (14), Q8_0 (8),
/// IQ4_XS (23), IQ3_XXS (18), NVFP4 (40) on RDNA2 (gfx1030), RDNA3/3.5 (gfx11*), RDNA4 (gfx12*).
/// `global_scale` is applied to outputs (pass 1.0f for non-NVFP4 or NVFP4 s_scale).
bool hip_unsloth_dp4a_mmvq(int ggml_type, const void* W_quant, const void* X_q8_1, float* Y,
                           int n_in, int n_out, int ncols, float global_scale, void* stream);

/// Fused EXP_MODE=8 gate + up + SwiGLU + q8_1 quantization in one launch (LDS activation + grid
/// staging, hoisted weight decode, exact NVFP4 s_gate/s_up tail scales, 65535-safe group striding).
/// Declines (returns false) whenever `L.had_block != 0` or `L.n_ff % 32 != 0` or unsupported type.
bool hip_unsloth_exp8_gu_fused(const HipExpertLayout& L, const unsigned long long* grp_ptr,
                               const int32_t* grp_start, const int32_t* n_groups,
                               const int32_t* ent_tok, int64_t cap_groups,
                               const void* x_q8_1, void* h_q8_1, void* stream);

/// LDS-staged down projection for all 7 Unsloth quant types (hoisted weight decode, 8-lane sub-warp
/// path for 16/20/40-item shapes, exact NVFP4 s_down tail scale, 65535-safe group striding).
bool hip_unsloth_down_lds(const HipExpertLayout& L, const unsigned long long* grp_ptr,
                          const int32_t* grp_start, const int32_t* n_groups,
                          const int32_t* ent_dst, int64_t cap_groups,
                          const void* h_q8_1, float* out, void* stream);

}  // namespace strata::kernels::hip_unsloth
