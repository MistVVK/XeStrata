// engine/src/inkling2/hip/ink_hip_common.hip.h
// Multi-Gen AMD ROCm/HIP Primitives for Inkling-2-Mini (RDNA4 gfx120x, RDNA3/3.5 gfx110x/1103/1150/1151, RDNA2 gfx103x).
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "strata/hip_uma_guard.hpp"

// ---- Compile-time architecture guards (strictly zero WMMA intrinsics on RDNA2 gfx103x!) ----
#if defined(__gfx1200__) || defined(__gfx1201__)
#define INK_GFX12 1
#else
#define INK_GFX12 0
#endif

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define INK_GFX11 1
#else
#define INK_GFX11 0
#endif

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
#define INK_RDNA2 1
#else
#define INK_RDNA2 0
#endif

#if INK_GFX12 || INK_GFX11
#define INK_HAS_WMMA 1
#else
#define INK_HAS_WMMA 0
#endif

#if INK_RDNA2 || defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__)
#define INK_HAS_SDOT4 1
#else
#define INK_HAS_SDOT4 0
#endif

#define INK_HCK(expr)                                                                                                  \
    do {                                                                                                               \
        hipError_t _e = (expr);                                                                                        \
        if (_e != hipSuccess) {                                                                                        \
            std::fprintf(stderr, "[ink-hip] HIP error %d (%s) at %s:%d\n", (int)_e, hipGetErrorString(_e), __FILE__,   \
                         __LINE__);                                                                                    \
            std::exit(3);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace strata::inkling2::hip {

using half_t = _Float16;
typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
typedef _Float16 hv8 __attribute__((ext_vector_type(8)));
typedef _Float16 hv16 __attribute__((ext_vector_type(16)));
typedef short bf8 __attribute__((ext_vector_type(8)));
typedef short bf16v __attribute__((ext_vector_type(16)));
typedef float fv8 __attribute__((ext_vector_type(8)));
typedef int iv2 __attribute__((ext_vector_type(2)));
typedef int iv4 __attribute__((ext_vector_type(4)));
typedef int iv8 __attribute__((ext_vector_type(8)));

// ---- Fast LDS barrier (RDNA4 split dscnt wait + barrier) ----
__device__ __forceinline__ void sync_lds() {
#if INK_GFX12
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
#else
    __syncthreads();
#endif
}

// ---- Wave32 DPP reductions (single-cycle intra-wave ALU shuffles on RDNA2/3/3.5/4, zero LDS) ----
__device__ __forceinline__ float wave32_sum(float v) {
#if defined(__HIP_DEVICE_COMPILE__) && (INK_GFX12 || INK_GFX11 || INK_RDNA2)
    v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0xB1, 0xf, 0xf, true));
    v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x4E, 0xf, 0xf, true));
    v += __shfl_xor(v, 4, 32);
    v += __shfl_xor(v, 8, 32);
    v += __shfl_xor(v, 16, 32);
    return v;
#else
    v += __shfl_xor(v, 1, 32);
    v += __shfl_xor(v, 2, 32);
    v += __shfl_xor(v, 4, 32);
    v += __shfl_xor(v, 8, 32);
    v += __shfl_xor(v, 16, 32);
    return v;
#endif
}

__device__ __forceinline__ float wave32_max(float v) {
#if defined(__HIP_DEVICE_COMPILE__) && (INK_GFX12 || INK_GFX11 || INK_RDNA2)
    v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0xB1, 0xf, 0xf, true)));
    v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x4E, 0xf, 0xf, true)));
    v = fmaxf(v, __shfl_xor(v, 4, 32));
    v = fmaxf(v, __shfl_xor(v, 8, 32));
    v = fmaxf(v, __shfl_xor(v, 16, 32));
    return v;
#else
    v = fmaxf(v, __shfl_xor(v, 1, 32));
    v = fmaxf(v, __shfl_xor(v, 2, 32));
    v = fmaxf(v, __shfl_xor(v, 4, 32));
    v = fmaxf(v, __shfl_xor(v, 8, 32));
    v = fmaxf(v, __shfl_xor(v, 16, 32));
    return v;
#endif
}

// ---- Packed FP16 / BF16 / INT8 dot primitives ----
__device__ __forceinline__ uint32_t h2_pack(float a, float b) {
    const hv2 h = {(half_t)a, (half_t)b};
    return __builtin_bit_cast(uint32_t, h);
}

__device__ __forceinline__ void h2_unpack(uint32_t w, float& a, float& b) {
    const hv2 h = __builtin_bit_cast(hv2, w);
    a = (float)h[0];
    b = (float)h[1];
}

__device__ __forceinline__ uint32_t h2_splat(float f) {
    const hv2 h = {(half_t)f, (half_t)f};
    return __builtin_bit_cast(uint32_t, h);
}

__device__ __forceinline__ uint32_t h2_sub(uint32_t a, uint32_t b) {
    return __builtin_bit_cast(uint32_t, __builtin_bit_cast(hv2, a) - __builtin_bit_cast(hv2, b));
}

__device__ __forceinline__ uint32_t h2_mul(uint32_t a, uint32_t b) {
    return __builtin_bit_cast(uint32_t, __builtin_bit_cast(hv2, a) * __builtin_bit_cast(hv2, b));
}

__device__ __forceinline__ float fdot2_f32(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(hv2, a), __builtin_bit_cast(hv2, b), c, false);
}

__device__ __forceinline__ int sdot4_i32(int a, int b, int c) {
#if INK_HAS_SDOT4
    return __builtin_amdgcn_sdot4(a, b, c, false);
#elif INK_GFX11 || INK_GFX12
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    int s = c;
    s += (int)(int8_t)(a & 0xFF) * (int)(int8_t)(b & 0xFF);
    s += (int)(int8_t)((a >> 8) & 0xFF) * (int)(int8_t)((b >> 8) & 0xFF);
    s += (int)(int8_t)((a >> 16) & 0xFF) * (int)(int8_t)((b >> 16) & 0xFF);
    s += (int)(int8_t)((a >> 24) & 0xFF) * (int)(int8_t)((b >> 24) & 0xFF);
    return s;
#endif
}

__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    return __builtin_bit_cast(float, u);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    const uint32_t u = __builtin_bit_cast(uint32_t, f);
    const uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding) >> 16);
}

__device__ __forceinline__ int pack4_i8(int a, int b, int c, int d) {
    return static_cast<int>((static_cast<uint32_t>(a & 0xFF)) |
                            (static_cast<uint32_t>(b & 0xFF) << 8) |
                            (static_cast<uint32_t>(c & 0xFF) << 16) |
                            (static_cast<uint32_t>(d & 0xFF) << 24));
}

__device__ __forceinline__ int clamp_i8(float x) {
    const int v = __float2int_rn(x);
    return v < -127 ? -127 : (v > 127 ? 127 : v);
}

// ---- Exact SF4 (E2M1 NVFP4) register LUTs ----
// Standard E2M1 nibbles 0..15 -> {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6}
// Exact 2x integer representation fits in signed int8 [-12, +12] with ZERO rounding error!
//   w_exact = 0.5f * sf4_e2m1_i8_x2(nib) * f8_e4m3_to_f32(blk_scale) * outer_scale
__device__ __forceinline__ int8_t sf4_e2m1_i8_x2(uint32_t nib) {
    static constexpr int8_t kLut[16] = {
         0,  1,  2,  3,  4,  6,  8, 12,
         0, -1, -2, -3, -4, -6, -8, -12
    };
    return kLut[nib & 15u];
}

__device__ __forceinline__ half_t sf4_e2m1_f16(uint32_t nib) {
    // Exact FP16 bit patterns for {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6}
    static constexpr uint16_t kLut[16] = {
        0x0000u, 0x3800u, 0x3C00u, 0x3E00u, 0x4000u, 0x4200u, 0x4400u, 0x4600u,
        0x8000u, 0xB800u, 0xBC00u, 0xBE00u, 0xC000u, 0xC200u, 0xC400u, 0xC600u
    };
    return __builtin_bit_cast(half_t, kLut[nib & 15u]);
}

__device__ __forceinline__ float sf4_e2m1_f32(uint32_t nib) {
    static constexpr float kLut[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return kLut[nib & 15u];
}

// ---- Exact F8_E4M3FN & UE8M0 decoders (for SF4 block scales, MX8 weights/scales, and FP8 KV cache) ----
__device__ __forceinline__ float e4m3fn_to_f32(uint8_t b) {
    const uint32_t sign = static_cast<uint32_t>(b & 0x80u) << 24;
    const uint32_t exp  = (b >> 3) & 0x0Fu;
    const uint32_t mant = b & 0x07u;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // Subnormal: mant * 2^-9
            const float val = static_cast<float>(mant) * (1.0f / 512.0f);
            return (b & 0x80u) ? -val : val;
        }
    } else if (exp == 15 && mant == 7) {
        return 0.0f;  // NaN maps to 0.0f safely
    } else {
        f = sign | ((exp + (127u - 7u)) << 23) | (mant << 20);
    }
    return __builtin_bit_cast(float, f);
}

__device__ __forceinline__ uint8_t f32_to_e4m3fn(float x) {
    const uint32_t sign = (x < 0.0f) ? 0x80u : 0x00u;
    float ax = fabsf(x);
    if (ax > 448.0f) ax = 448.0f;
    if (ax < (1.0f / 1024.0f)) return static_cast<uint8_t>(sign);
    const uint32_t u = __builtin_bit_cast(uint32_t, ax);
    const int exp = static_cast<int>((u >> 23) & 0xFFu) - 127;
    const uint32_t mant = u & 0x7FFFFFu;
    if (exp < -6) {
        // Subnormal: round(ax * 512) in 0..7
        int sm = __float2int_rn(ax * 512.0f);
        if (sm >= 8) return static_cast<uint8_t>(sign | 0x08u);
        return static_cast<uint8_t>(sign | static_cast<uint32_t>(sm));
    }
    int e4 = exp + 7;
    int m3 = static_cast<int>((mant + 0x080000u) >> 20);
    if (m3 >= 8) { m3 = 0; ++e4; }
    if (e4 > 15 || (e4 == 15 && m3 >= 7)) { e4 = 15; m3 = 6; }
    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(e4) << 3) | static_cast<uint32_t>(m3));
}

__device__ __forceinline__ float ue8m0_to_f32(uint8_t e) {
    if (e == 255u) return 0.0f;
    if (e == 0u) return __builtin_bit_cast(float, 0x00400000u);  // 2^-127
    return __builtin_bit_cast(float, static_cast<uint32_t>(e) << 23);
}

// ---- SITU(beta=4.0, lb=25.0) double-softcapped MoE activation ----
__device__ __forceinline__ float situ_f32(float g, float u, float beta = 4.0f, float lb = 25.0f) {
    const float sig = 1.0f / (1.0f + __expf(-g));
    const float g_act = beta * tanhf(g / beta) * sig;
    const float u_act = lb * tanhf(u / lb);
    return g_act * u_act;
}

// ---- Multi-Gen WMMA Wrappers (strictly guarded — zero WMMA builtins compiled on RDNA2!) ----
__device__ __forceinline__ fv8 fv8_zero() { return fv8{0, 0, 0, 0, 0, 0, 0, 0}; }
__device__ __forceinline__ iv8 iv8_zero() { return iv8{0, 0, 0, 0, 0, 0, 0, 0}; }

#if INK_GFX12
using wmma_h_t = hv8;
using wmma_bf_t = bf8;
__device__ __forceinline__ wmma_h_t wmma_ld_h(const half_t* p, int hi) {
    const uint4 v = *reinterpret_cast<const uint4*>(p + hi * 8);
    return __builtin_bit_cast(hv8, v);
}
__device__ __forceinline__ fv8 wmma_f16(wmma_h_t a, wmma_h_t b, fv8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ wmma_bf_t wmma_ld_bf(const uint16_t* p, int hi) {
    const uint4 v = *reinterpret_cast<const uint4*>(p + hi * 8);
    return __builtin_bit_cast(bf8, v);
}
__device__ __forceinline__ fv8 wmma_bf16(wmma_bf_t a, wmma_bf_t b, fv8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
// RDNA4 2x-rate INT8 WMMA: 16x16x32 iu8 (iv4 = 16 int8 per half-wave!) and 16x16x16 iu8 (iv2 = 8 int8 per half-wave)
__device__ __forceinline__ iv4 wmma_ld_i8_k32(const int8_t* p, int hi) {
    const uint4 v = *reinterpret_cast<const uint4*>(p + hi * 16);
    return __builtin_bit_cast(iv4, v);
}
__device__ __forceinline__ iv8 wmma_i8_k32(iv4 a, iv4 b, iv8 c) {
#if __has_builtin(__builtin_amdgcn_wmma_i32_16x16x32_iu8_w32_gfx12)
    return __builtin_amdgcn_wmma_i32_16x16x32_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    const iv2 a0 = {a[0], a[1]}, a1 = {a[2], a[3]};
    const iv2 b0 = {b[0], b[1]}, b1 = {b[2], b[3]};
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, b0, c, false);
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, true, b1, c, false);
#endif
}
__device__ __forceinline__ iv2 wmma_ld_i8_k16(const int8_t* p, int hi) {
    const uint2 v = *reinterpret_cast<const uint2*>(p + hi * 8);
    return __builtin_bit_cast(iv2, v);
}
__device__ __forceinline__ iv8 wmma_i8_k16(iv2 a, iv2 b, iv8 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}
__device__ __forceinline__ int wmma_acc_m(int r, int lane) { return 8 * (lane >> 4) + r; }
__device__ __forceinline__ int wmma_acc_n(int r, int lane) { (void)r; return lane & 15; }
#elif INK_GFX11
using wmma_h_t = hv16;
using wmma_bf_t = bf16v;
__device__ __forceinline__ wmma_h_t wmma_ld_h(const half_t* p, int /*hi*/) {
    const uint4 a = *reinterpret_cast<const uint4*>(p);
    const uint4 b = *reinterpret_cast<const uint4*>(p + 8);
    typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
    const u32x8 v = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(hv16, v);
}
__device__ __forceinline__ fv8 wmma_f16(wmma_h_t a, wmma_h_t b, fv8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ wmma_bf_t wmma_ld_bf(const uint16_t* p, int /*hi*/) {
    const uint4 a = *reinterpret_cast<const uint4*>(p);
    const uint4 b = *reinterpret_cast<const uint4*>(p + 8);
    typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
    const u32x8 v = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(bf16v, v);
}
__device__ __forceinline__ fv8 wmma_bf16(wmma_bf_t a, wmma_bf_t b, fv8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
__device__ __forceinline__ iv4 wmma_ld_i8_k16_g11(const int8_t* p) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    return __builtin_bit_cast(iv4, v);
}
__device__ __forceinline__ iv8 wmma_i8_k16_g11(iv4 a, iv4 b, iv8 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}
__device__ __forceinline__ int wmma_acc_m(int r, int lane) { return 2 * r + (lane >> 4); }
__device__ __forceinline__ int wmma_acc_n(int r, int lane) { (void)r; return lane & 15; }
#endif

}  // namespace strata::inkling2::hip
