// Copyright (c) 2026 Strata contributors
// SPDX-License-Identifier: MIT
//
// include/strata/kernels/hip_unsloth_wmma.hpp - Clean-room AMD Wave32 WMMA traits & prefill/verify
// GEMM dispatch for RDNA3 (gfx1100..1103), RDNA3.5 / Strix Halo (gfx1150/1151), and RDNA4 (gfx1200/1201).
//
// Key architectural difference handled cleanly at compile time:
//   * RDNA3 / RDNA3.5 (gfx11*): Wave32 WMMA 16x16x16 takes DUPLICATED 16-element A/B fragments per lane
//     (lanes 16..31 mirror lanes 0..15) and interleaves C/D output rows as row = 2 * i + (lane >> 4).
//   * RDNA4 (gfx1200/1201):     Wave32 WMMA 16x16x16 takes NON-DUPLICATED 8-element A/B fragments per lane
//     (lane half hi = lane >> 4 supplies k = 8*hi .. 8*hi + 7) and packs C/D output rows contiguously as
//     row = 8 * (lane >> 4) + i.
//   * RDNA2 (gfx1030..1034):    No WMMA instructions; compiles cleanly without trapping host passes and
//     returns false at runtime so callers fall back to Wave32 DP4A / rocBLAS paths.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#endif

namespace strata::kernels::hip_unsloth {

// ===========================================================================
// Host-safe architecture classification (no HIP runtime headers required)
// ===========================================================================
inline bool gcn_is(const char* gcn, const char* name) noexcept {
    if (gcn == nullptr) return false;
    const std::size_t n = std::strlen(name);
    return std::strncmp(gcn, name, n) == 0 && (gcn[n] == '\0' || gcn[n] == ':');
}

inline bool gcn_is_rdna2(const char* gcn) noexcept {
    return gcn_is(gcn, "gfx1030") || gcn_is(gcn, "gfx1031") || gcn_is(gcn, "gfx1032") || gcn_is(gcn, "gfx1034");
}

inline bool gcn_is_gfx11_wmma(const char* gcn) noexcept {
    return gcn_is(gcn, "gfx1100") || gcn_is(gcn, "gfx1101") || gcn_is(gcn, "gfx1102") ||
           gcn_is(gcn, "gfx1103") || gcn_is(gcn, "gfx1150") || gcn_is(gcn, "gfx1151");
}

inline bool gcn_is_gfx12_wmma(const char* gcn) noexcept {
    return gcn_is(gcn, "gfx1200") || gcn_is(gcn, "gfx1201");
}

inline bool gcn_has_wmma(const char* gcn) noexcept {
    return gcn_is_gfx11_wmma(gcn) || gcn_is_gfx12_wmma(gcn);
}

// ===========================================================================
// Device compile-time WMMA traits & WGP-safe LDS barrier
// ===========================================================================
#if defined(__HIPCC__)

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_HIP_UNSLOTH_WMMA_GFX11 1
#else
#define STRATA_HIP_UNSLOTH_WMMA_GFX11 0
#endif

#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_HIP_UNSLOTH_WMMA_GFX12 1
#else
#define STRATA_HIP_UNSLOTH_WMMA_GFX12 0
#endif

#define STRATA_HIP_UNSLOTH_HAS_WMMA (STRATA_HIP_UNSLOTH_WMMA_GFX11 || STRATA_HIP_UNSLOTH_WMMA_GFX12)

using f32x8_t = float __attribute__((ext_vector_type(8)));
using i32x8_t = int __attribute__((ext_vector_type(8)));

// WGP-safe LDS-only barrier: avoids global_inv scope:SCOPE_SE on gfx12 and prevents LLVM from
// sinking LDS stores past s_barrier in WGP mode on gfx11/gfx12.
__device__ __forceinline__ void hip_lds_barrier() {
#if STRATA_HIP_UNSLOTH_WMMA_GFX12
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#elif STRATA_HIP_UNSLOTH_WMMA_GFX11
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_waitcnt(0x7 | (0x3f << 10));  // lgkmcnt(0) only
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
}

// Output row within a 16x16 C/D tile for accumulator element i (0..7) and lane half hi = (lane >> 4).
__device__ __forceinline__ int wmma_d_row(int i, int hi) {
#if STRATA_HIP_UNSLOTH_WMMA_GFX12
    return 8 * hi + i;
#else
    return 2 * i + hi;
#endif
}

// K-offset within a 16-wide K step loaded by lane half hi = (lane >> 4).
__device__ __forceinline__ int wmma_k_offset(int hi) {
#if STRATA_HIP_UNSLOTH_WMMA_GFX12
    return 8 * hi;
#else
    (void) hi;
    return 0;
#endif
}

template <typename ElemT>
struct WmmaWave32;

template <>
struct WmmaWave32<_Float16> {
#if STRATA_HIP_UNSLOTH_WMMA_GFX12
    static constexpr int kFragElems = 8;
    using frag_t = _Float16 __attribute__((ext_vector_type(8)));
    __device__ static __forceinline__ frag_t load_frag(const _Float16* p) {
        const uint4 v = *reinterpret_cast<const uint4*>(p);
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
        return __builtin_bit_cast(frag_t, w);
    }
    __device__ static __forceinline__ frag_t zero_frag() {
        return frag_t{(_Float16) 0, (_Float16) 0, (_Float16) 0, (_Float16) 0,
                      (_Float16) 0, (_Float16) 0, (_Float16) 0, (_Float16) 0};
    }
    __device__ static __forceinline__ f32x8_t mma(frag_t a, frag_t b, f32x8_t c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    }
#elif STRATA_HIP_UNSLOTH_WMMA_GFX11
    static constexpr int kFragElems = 16;
    using frag_t = _Float16 __attribute__((ext_vector_type(16)));
    __device__ static __forceinline__ frag_t load_frag(const _Float16* p) {
        const uint4 a = *reinterpret_cast<const uint4*>(p);
        const uint4 b = *reinterpret_cast<const uint4*>(p + 8);
        const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        return __builtin_bit_cast(frag_t, w);
    }
    __device__ static __forceinline__ frag_t zero_frag() {
        frag_t z;
#pragma unroll
        for (int i = 0; i < 16; ++i) z[i] = (_Float16) 0;
        return z;
    }
    __device__ static __forceinline__ f32x8_t mma(frag_t a, frag_t b, f32x8_t c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
#else
    static constexpr int kFragElems = 8;
    using frag_t = _Float16 __attribute__((ext_vector_type(8)));
    __device__ static __forceinline__ frag_t load_frag(const _Float16*) { return zero_frag(); }
    __device__ static __forceinline__ frag_t zero_frag() {
        return frag_t{(_Float16) 0, (_Float16) 0, (_Float16) 0, (_Float16) 0,
                      (_Float16) 0, (_Float16) 0, (_Float16) 0, (_Float16) 0};
    }
    __device__ static __forceinline__ f32x8_t mma(frag_t, frag_t, f32x8_t c) {
        __builtin_trap();
        return c;
    }
#endif
};

template <>
struct WmmaWave32<__bf16> {
#if STRATA_HIP_UNSLOTH_WMMA_GFX12
    static constexpr int kFragElems = 8;
    using frag_t = __bf16 __attribute__((ext_vector_type(8)));
    __device__ static __forceinline__ frag_t load_frag(const __bf16* p) {
        const uint4 v = *reinterpret_cast<const uint4*>(p);
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
        return __builtin_bit_cast(frag_t, w);
    }
    __device__ static __forceinline__ frag_t zero_frag() {
        frag_t z;
#pragma unroll
        for (int i = 0; i < 8; ++i) z[i] = (__bf16) 0.0f;
        return z;
    }
    __device__ static __forceinline__ f32x8_t mma(frag_t a, frag_t b, f32x8_t c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    }
#elif STRATA_HIP_UNSLOTH_WMMA_GFX11
    static constexpr int kFragElems = 16;
    using frag_t = __bf16 __attribute__((ext_vector_type(16)));
    __device__ static __forceinline__ frag_t load_frag(const __bf16* p) {
        const uint4 a = *reinterpret_cast<const uint4*>(p);
        const uint4 b = *reinterpret_cast<const uint4*>(p + 8);
        const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        return __builtin_bit_cast(frag_t, w);
    }
    __device__ static __forceinline__ frag_t zero_frag() {
        frag_t z;
#pragma unroll
        for (int i = 0; i < 16; ++i) z[i] = (__bf16) 0.0f;
        return z;
    }
    __device__ static __forceinline__ f32x8_t mma(frag_t a, frag_t b, f32x8_t c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
#else
    static constexpr int kFragElems = 8;
    using frag_t = __bf16 __attribute__((ext_vector_type(8)));
    __device__ static __forceinline__ frag_t load_frag(const __bf16*) { return zero_frag(); }
    __device__ static __forceinline__ frag_t zero_frag() {
        frag_t z;
#pragma unroll
        for (int i = 0; i < 8; ++i) z[i] = (__bf16) 0.0f;
        return z;
    }
    __device__ static __forceinline__ f32x8_t mma(frag_t, frag_t, f32x8_t c) {
        __builtin_trap();
        return c;
    }
#endif
};

template <>
struct WmmaWave32<int8_t> {
#if STRATA_HIP_UNSLOTH_WMMA_GFX12
    static constexpr int kFragI32 = 2;  // 8 int8 per lane
    using frag_t = int __attribute__((ext_vector_type(2)));
    __device__ static __forceinline__ i32x8_t mma(frag_t a, frag_t b, i32x8_t c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
    }
#elif STRATA_HIP_UNSLOTH_WMMA_GFX11
    static constexpr int kFragI32 = 4;  // 16 int8 per lane (duplicated)
    using frag_t = int __attribute__((ext_vector_type(4)));
    __device__ static __forceinline__ i32x8_t mma(frag_t a, frag_t b, i32x8_t c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
    }
#else
    static constexpr int kFragI32 = 2;
    using frag_t = int __attribute__((ext_vector_type(2)));
    __device__ static __forceinline__ i32x8_t mma(frag_t, frag_t, i32x8_t c) {
        __builtin_trap();
        return c;
    }
#endif
};

#endif  // defined(__HIPCC__)

// ===========================================================================
// Host API (implemented in src/kernels/hip/hip_unsloth_wmma.cu)
// ===========================================================================

/// True when the active HIP device supports Wave32 WMMA (gfx1100..1151 OR gfx1200..1201).
bool hip_wmma_device_available();

/// True when EXP_MODE=8 fused gate/up+SwiGLU+q8_1 should auto-enable on `dev`
/// (covers gfx1151, gfx1100..1103, gfx1150, AND gfx1200/1201; respects STRATA_GFX1151_EXP8=0 /
/// STRATA_GFX12_EXP8=0 / STRATA_HIP_EXP8=0).
bool hip_exp8_auto_enabled(int dev);

/// FP16 GEMM: Y[t, n] = beta * Y[t, n] + sum_k X[t, k] * W[n, k] (X: T x ldx, W: N x ldw, Y: T x ldy).
/// Supports BOTH RDNA3 (gfx11*) and RDNA4 (gfx12*). Returns false if unsupported or invalid args.
bool hip_unsloth_wmma_gemm_f16(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                               float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                               void* stream);

/// BF16 GEMM: Y[t, n] = beta * Y[t, n] + sum_k X[t, k] * W[n, k] (X: T x ldx, W: N x ldw, Y: T x ldy).
/// Supports BOTH RDNA3 (gfx11*) and RDNA4 (gfx12*). Returns false if unsupported or invalid args.
bool hip_unsloth_wmma_gemm_bf16(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                                float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                                void* stream);

/// Wave32 INT8 WMMA verify GEMM for Q8_0 weights x Q8_1 activations (T = 1..16, best for T = 8..16).
/// Supports BOTH RDNA3 (gfx11*) and RDNA4 (gfx12*).
bool hip_unsloth_wmma_q80_mmvq(const void* W_q8_0, const void* X_q8_1, float* Y,
                               int n_in, int n_out, int ncols, void* stream);

/// Fused dequant-to-LDS FP16 + Wave32 WMMA prefill GEMM:
///   Y[t, n] = global_scale * sum_k X_f16[t, k] * dequant(W[n, k])
/// Supports Q4_K (12), Q5_K (13), Q6_K (14), Q8_0 (8), IQ4_XS (23), IQ3_XXS (18), NVFP4 (40)
/// on BOTH RDNA3 (gfx11*) and RDNA4 (gfx12*). K must be a multiple of 256 (64 for NVFP4/Q8_0).
bool hip_unsloth_wmma_dequant_prefill(int ggml_type, const void* W_quant, const uint16_t* X_f16,
                                      int64_t ldx, float* Y, int64_t T, int64_t N, int64_t K,
                                      int64_t ldy, float global_scale, void* stream);

}  // namespace strata::kernels::hip_unsloth
