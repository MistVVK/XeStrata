// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/hip/kitchen_module.cpp - the module of comfy-kitchen's HIP kernels (third_party/main/comfy-kitchen,
// Apache-2.0: its WMMA int8 product with the scales and bias in its epilogue, and its int8 attention, for RDNA3 and
// RDNA4), which i8_gemm.cpp opens on an AMD GPU only: the kernels need AMD's HIP runtime, which a machine with other
// makers' GPUs need not have (libxestrata_hip_kitchen.so, xestrata_hip_kitchen.dll, beside the engine).  Its entries take
// the SYCL queue's HIP stream and the device's pointers.
#include <hip/hip_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <unordered_map>

extern "C" void launch_int8_gemm_kernel(const void* a, const void* b, void* c, const void* scale_a,
                                        const void* scale_b, int scale_b_stride, const void* bias, int bias_code,
                                        int M, int N, int K, int ldc, int out_code, hipStream_t stream);

extern "C" void launch_int8_gemm_a_tiled128_kernel(const void* a, const void* b, void* c, const void* scale_a,
                                                   const void* scale_b, int scale_b_stride, const void* bias,
                                                   int bias_code, int M, int N, int K, int ldc, int out_code,
                                                   int b_tiled, hipStream_t stream);

struct SageDenseMask16;
extern "C" void launch_sage_quant_qk_int8(const void* q, void* q_int8, void* q_scale, const void* k, void* k_int8,
                                          void* k_scale, void* anchor_indices, int B, int H_q, int Lq,
                                          int Lq_padded, int H_kv, int Lk, int groups_per_head, int head_dim,
                                          int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
                                          int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
                                          int input_dtype_code, int rotation, hipStream_t stream,
                                          const SageDenseMask16* dense_mask);
extern "C" void launch_sage_quant_v_int8(const void* v, void* v_int8, void* v_scale, int B, int H, int L,
                                         int head_dim, int padded_k, int64_t stride_b, int64_t stride_h,
                                         int64_t stride_n, int input_dtype_code, hipStream_t stream);
extern "C" void launch_sage_int8_attn(const void* q, const void* k, const void* v, void* o, const void* q_scale,
                                      const void* k_scale, const void* v_scale, const void* mask,
                                      int64_t mask_stride_b, int64_t mask_stride_h, int64_t mask_stride_q,
                                      int64_t mask_stride_k, int mask_dtype_code, int cta_k, int batch, int qo_len,
                                      int kv_len, int qo_len_padded, int num_qo_heads, int num_kv_heads,
                                      int head_dim, int k_groups_per_head, int64_t q_stride_b, int64_t q_stride_h,
                                      int64_t k_stride_b, int64_t k_stride_h, int64_t v_stride_b,
                                      int64_t v_stride_h, int64_t v_stride_d, int64_t o_stride_b,
                                      int64_t o_stride_h, int64_t o_stride_n, float sm_scale,
                                      int output_dtype_code, hipStream_t stream);

#if defined(_WIN32)
#define STRATA_KITCHEN_EXPORT extern "C" __declspec(dllexport)
#else
#define STRATA_KITCHEN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Y [T][N] (BF16, row stride ldy) = (X [T][K] . W [N][K]^T) * sx[t] * sw[n] + bias[n] (FP32; null: none) on `stream`
// (a hipStream_t); 0 where it was launched.  comfy-kitchen's codes: kF32 0 (the bias), kBF16 2 (the output).
STRATA_KITCHEN_EXPORT int strata_kitchen_i8_gemm(const int8_t* x, const int8_t* w, uint16_t* y, const float* sx,
                                                 const float* sw, const float* bias, int64_t T, int64_t N,
                                                 int64_t K, int64_t ldy, void* stream) {
    const auto s = static_cast<hipStream_t>(stream);
    int dev = 0;   // its launchers ask the current device's architecture: the stream's
    if (hipStreamGetDevice(s, &dev) != hipSuccess || hipSetDevice(dev) != hipSuccess) return 1;
    (void) hipGetLastError();   // an error an earlier call on the thread left (another module's), not this launch's
    try {
        launch_int8_gemm_kernel(x, w, y, sx, sw, 1, bias, 0, (int) T, (int) N, (int) K, (int) ldy, 2, s);
    } catch (...) {
        return 1;
    }
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

// The same with X in comfy-kitchen's tiles: [T / 128 rounded up][K / 128][128 rows][128 bytes], the last tile's rows
// past T zero (K a multiple of 128): its schedule tuned for RDNA4.
STRATA_KITCHEN_EXPORT int strata_kitchen_i8_gemm_tiled(const int8_t* x, const int8_t* w, uint16_t* y, const float* sx,
                                                       const float* sw, const float* bias, int64_t T, int64_t N,
                                                       int64_t K, int64_t ldy, void* stream) {
    const auto s = static_cast<hipStream_t>(stream);
    int dev = 0;
    if (K % 128 != 0 || hipStreamGetDevice(s, &dev) != hipSuccess || hipSetDevice(dev) != hipSuccess) return 1;
    (void) hipGetLastError();
    try {
        launch_int8_gemm_a_tiled128_kernel(x, w, y, sx, sw, 1, bias, 0, (int) T, (int) N, (int) K, (int) ldy, 2,
                                           0 /* b_tiled: W is row-major */, s);
    } catch (...) {
        return 1;
    }
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

namespace {

// A device's scratch for the attention's int8 forms, grown as needed, kept (freed with the process).
struct Scratch {
    void* p = nullptr;
    size_t bytes = 0;
};

void* scratch(int dev, size_t bytes, hipStream_t s) {
    static std::mutex mu;
    static std::unordered_map<int, Scratch> of;
    std::lock_guard<std::mutex> lock(mu);
    Scratch& b = of[dev];
    if (b.bytes < bytes) {
        if (b.p != nullptr && (hipStreamSynchronize(s) != hipSuccess || hipFree(b.p) != hipSuccess)) return nullptr;
        b.p = nullptr;
        b.bytes = 0;
        if (hipMalloc(&b.p, bytes) != hipSuccess) return nullptr;
        b.bytes = bytes;
    }
    return b.p;
}

size_t up(size_t n, size_t a) { return (n + a - 1) / a * a; }

}  // namespace

// comfy-kitchen's int8 attention (SageAttention's scheme, its HIP kernels): query rows r0 .. T - 1 of q16 against
// T keys of k16 and v16, all FP16 [rows][heads * hd] (hd 64 or 128), out16's row 0 query row `first`; its own
// quantization of Q, K (smoothed by a key it picks) and V (its columns' scales), its 64 keys a step and 128 queries.
STRATA_KITCHEN_EXPORT int strata_kitchen_attn(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16,
                                              int64_t T, int64_t heads, int64_t hd, int64_t r0, int64_t first,
                                              uint16_t* out16, void* stream) {
    constexpr int kCtaQ = 128, kCtaK = 64, kKeyGroup = 16, kF16 = 1;
    const auto s = static_cast<hipStream_t>(stream);
    int dev = 0;
    if (hipStreamGetDevice(s, &dev) != hipSuccess || hipSetDevice(dev) != hipSuccess) return 1;
    const int64_t C = heads * hd, Lq = T - r0, Lqp = (Lq + kCtaQ - 1) / kCtaQ * kCtaQ;
    const int64_t Kp = (T + kCtaK - 1) / kCtaK * kCtaK;
    // its forms: Q [heads][Lq][hd], K [heads][T][hd], V^T [heads * hd][Kp] in int8; Q's scales [heads][Lqp], K's
    // [heads][Kp / 16], V's [heads * hd]; the anchor key [heads]
    const size_t oq = 0, ok = up(oq + (size_t) (heads * Lq * hd), 256), ov = up(ok + (size_t) (heads * T * hd), 256),
                 osq = up(ov + (size_t) (C * Kp), 256), osk = up(osq + (size_t) (heads * Lqp) * 4, 256),
                 osv = up(osk + (size_t) (heads * Kp / kKeyGroup) * 4, 256), oan = up(osv + (size_t) C * 4, 256),
                 bytes = oan + (size_t) heads * 4;
    auto* b = static_cast<char*>(scratch(dev, bytes, s));
    if (b == nullptr) return 1;
    const int rotation = T <= 256 ? 4 : (hd >= 128 ? 128 : 64);   // its choice by the keys and head dim
    (void) hipGetLastError();
    try {
        launch_sage_quant_qk_int8(q16 + r0 * C, b + oq, b + osq, k16, b + ok, b + osk, b + oan, 1, (int) heads,
                                  (int) Lq, (int) Lqp, (int) heads, (int) T, (int) (Kp / kKeyGroup), (int) hd, Lq * C,
                                  hd, C, T * C, hd, C, kF16, rotation, s, nullptr);
        launch_sage_quant_v_int8(v16, b + ov, b + osv, 1, (int) heads, (int) T, (int) hd, (int) Kp, T * C, hd, C,
                                 kF16, s);
        launch_sage_int8_attn(b + oq, b + ok, b + ov, out16 + (r0 - first) * C, b + osq, b + osk, b + osv, nullptr,
                              0, 0, 0, 0, -1, kCtaK, 1, (int) Lq, (int) T, (int) Lqp, (int) heads, (int) heads,
                              (int) hd, (int) (Kp / kKeyGroup), heads * Lq * hd, Lq * hd, heads * T * hd, T * hd,
                              C * Kp, hd * Kp, Kp, Lq * C, hd, C, 1.f / std::sqrt((float) hd), kF16, s);
    } catch (...) {
        return 1;
    }
    return hipGetLastError() == hipSuccess ? 0 : 1;
}
