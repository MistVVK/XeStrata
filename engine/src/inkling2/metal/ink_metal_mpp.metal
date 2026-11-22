// ink_metal_mpp.metal - Metal 4 Neural Accelerator (mpp::tensor_ops::matmul2d) Kernels for Inkling-2-Mini
// Compiled dynamically on Apple Silicon GPUs with Metal 4 / MPP support (e.g., Apple M5 Max);
// older macOS / M1-M4 Macs gracefully fall back to simdgroup_matrix kernels in ink_metal_codecs.metal.
// Implements:
//   1. Native FP16 matmul2d Dense Trunk & Concatenated QKV/R/Gate/O/Controller/DSpark GEMMs
//   2. Hybrid Decompress-Once + Staged Grouped-Expert MoE Prefill/Batching (SF4, MX8, S2_K..S8_K + fused SITU)
//   3. Tiled matmul2d FlashAttention-2 Prefill (BQ=32, BK=64) with L1-hot RelPos bias + Deferred-Sink Online Softmax
// 100% MIT clean-room implementation.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// ---- Shared codec helpers (kept self-contained so ink_metal_mpp.metal compiles standalone!) ----
constant float kMppE2M1[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float mpp_e4m3_to_f32(uint c) {
    const uint mag = c & 0x7Fu;
    if (mag == 0x7Fu) return 0.0f;
    const uint exp = mag >> 3;
    const uint man = mag & 7u;
    float v = (exp == 0u) ? (float(man) * (1.0f / 512.0f))
                          : as_type<float>(((exp + 120u) << 23) | (man << 20));
    return (c & 0x80u) ? -v : v;
}

static inline float mpp_ue8m0_to_f32(uint e) {
    if (e == 0u) return 5.877471754111438e-39f;
    if (e == 255u) return 0.0f;
    return as_type<float>(e << 23);
}

static inline float mpp_bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16); }

static inline ushort mpp_ld16(device const uchar* p) { return ushort(p[0]) | (ushort(p[1]) << 8); }

static inline constexpr int mpp_sk_sb_bytes(int F) {
    return (F == 2) ? 70 : (F == 3) ? 102 : (F == 4) ? 134 : (F == 5) ? 166 : (F == 6) ? 198 : 262;
}
static inline constexpr float mpp_sk_half(int F) {
    return (F == 2) ? 1.5f : (F == 3) ? 3.5f : (F == 4) ? 7.5f : (F == 5) ? 15.5f : (F == 6) ? 31.5f : 0.0f;
}

constant half4 kMppS2Lut[256] = {
    half4(-1.5h,-1.5h,-1.5h,-1.5h), half4(-0.5h,-1.5h,-1.5h,-1.5h), half4(0.5h,-1.5h,-1.5h,-1.5h), half4(1.5h,-1.5h,-1.5h,-1.5h),
    half4(-1.5h,-0.5h,-1.5h,-1.5h), half4(-0.5h,-0.5h,-1.5h,-1.5h), half4(0.5h,-0.5h,-1.5h,-1.5h), half4(1.5h,-0.5h,-1.5h,-1.5h),
    half4(-1.5h,0.5h,-1.5h,-1.5h), half4(-0.5h,0.5h,-1.5h,-1.5h), half4(0.5h,0.5h,-1.5h,-1.5h), half4(1.5h,0.5h,-1.5h,-1.5h),
    half4(-1.5h,1.5h,-1.5h,-1.5h), half4(-0.5h,1.5h,-1.5h,-1.5h), half4(0.5h,1.5h,-1.5h,-1.5h), half4(1.5h,1.5h,-1.5h,-1.5h),
    half4(-1.5h,-1.5h,-0.5h,-1.5h), half4(-0.5h,-1.5h,-0.5h,-1.5h), half4(0.5h,-1.5h,-0.5h,-1.5h), half4(1.5h,-1.5h,-0.5h,-1.5h),
    half4(-1.5h,-0.5h,-0.5h,-1.5h), half4(-0.5h,-0.5h,-0.5h,-1.5h), half4(0.5h,-0.5h,-0.5h,-1.5h), half4(1.5h,-0.5h,-0.5h,-1.5h),
    half4(-1.5h,0.5h,-0.5h,-1.5h), half4(-0.5h,0.5h,-0.5h,-1.5h), half4(0.5h,0.5h,-0.5h,-1.5h), half4(1.5h,0.5h,-0.5h,-1.5h),
    half4(-1.5h,1.5h,-0.5h,-1.5h), half4(-0.5h,1.5h,-0.5h,-1.5h), half4(0.5h,1.5h,-0.5h,-1.5h), half4(1.5h,1.5h,-0.5h,-1.5h),
    half4(-1.5h,-1.5h,0.5h,-1.5h), half4(-0.5h,-1.5h,0.5h,-1.5h), half4(0.5h,-1.5h,0.5h,-1.5h), half4(1.5h,-1.5h,0.5h,-1.5h),
    half4(-1.5h,-0.5h,0.5h,-1.5h), half4(-0.5h,-0.5h,0.5h,-1.5h), half4(0.5h,-0.5h,0.5h,-1.5h), half4(1.5h,-0.5h,0.5h,-1.5h),
    half4(-1.5h,0.5h,0.5h,-1.5h), half4(-0.5h,0.5h,0.5h,-1.5h), half4(0.5h,0.5h,0.5h,-1.5h), half4(1.5h,0.5h,0.5h,-1.5h),
    half4(-1.5h,1.5h,0.5h,-1.5h), half4(-0.5h,1.5h,0.5h,-1.5h), half4(0.5h,1.5h,0.5h,-1.5h), half4(1.5h,1.5h,0.5h,-1.5h),
    half4(-1.5h,-1.5h,1.5h,-1.5h), half4(-0.5h,-1.5h,1.5h,-1.5h), half4(0.5h,-1.5h,1.5h,-1.5h), half4(1.5h,-1.5h,1.5h,-1.5h),
    half4(-1.5h,-0.5h,1.5h,-1.5h), half4(-0.5h,-0.5h,1.5h,-1.5h), half4(0.5h,-0.5h,1.5h,-1.5h), half4(1.5h,-0.5h,1.5h,-1.5h),
    half4(-1.5h,0.5h,1.5h,-1.5h), half4(-0.5h,0.5h,1.5h,-1.5h), half4(0.5h,0.5h,1.5h,-1.5h), half4(1.5h,0.5h,1.5h,-1.5h),
    half4(-1.5h,1.5h,1.5h,-1.5h), half4(-0.5h,1.5h,1.5h,-1.5h), half4(0.5h,1.5h,1.5h,-1.5h), half4(1.5h,1.5h,1.5h,-1.5h),
    half4(-1.5h,-1.5h,-1.5h,-0.5h), half4(-0.5h,-1.5h,-1.5h,-0.5h), half4(0.5h,-1.5h,-1.5h,-0.5h), half4(1.5h,-1.5h,-1.5h,-0.5h),
    half4(-1.5h,-0.5h,-1.5h,-0.5h), half4(-0.5h,-0.5h,-1.5h,-0.5h), half4(0.5h,-0.5h,-1.5h,-0.5h), half4(1.5h,-0.5h,-1.5h,-0.5h),
    half4(-1.5h,0.5h,-1.5h,-0.5h), half4(-0.5h,0.5h,-1.5h,-0.5h), half4(0.5h,0.5h,-1.5h,-0.5h), half4(1.5h,0.5h,-1.5h,-0.5h),
    half4(-1.5h,1.5h,-1.5h,-0.5h), half4(-0.5h,1.5h,-1.5h,-0.5h), half4(0.5h,1.5h,-1.5h,-0.5h), half4(1.5h,1.5h,-1.5h,-0.5h),
    half4(-1.5h,-1.5h,-0.5h,-0.5h), half4(-0.5h,-1.5h,-0.5h,-0.5h), half4(0.5h,-1.5h,-0.5h,-0.5h), half4(1.5h,-1.5h,-0.5h,-0.5h),
    half4(-1.5h,-0.5h,-0.5h,-0.5h), half4(-0.5h,-0.5h,-0.5h,-0.5h), half4(0.5h,-0.5h,-0.5h,-0.5h), half4(1.5h,-0.5h,-0.5h,-0.5h),
    half4(-1.5h,0.5h,-0.5h,-0.5h), half4(-0.5h,0.5h,-0.5h,-0.5h), half4(0.5h,0.5h,-0.5h,-0.5h), half4(1.5h,0.5h,-0.5h,-0.5h),
    half4(-1.5h,1.5h,-0.5h,-0.5h), half4(-0.5h,1.5h,-0.5h,-0.5h), half4(0.5h,1.5h,-0.5h,-0.5h), half4(1.5h,1.5h,-0.5h,-0.5h),
    half4(-1.5h,-1.5h,0.5h,-0.5h), half4(-0.5h,-1.5h,0.5h,-0.5h), half4(0.5h,-1.5h,0.5h,-0.5h), half4(1.5h,-1.5h,0.5h,-0.5h),
    half4(-1.5h,-0.5h,0.5h,-0.5h), half4(-0.5h,-0.5h,0.5h,-0.5h), half4(0.5h,-0.5h,0.5h,-0.5h), half4(1.5h,-0.5h,0.5h,-0.5h),
    half4(-1.5h,0.5h,0.5h,-0.5h), half4(-0.5h,0.5h,0.5h,-0.5h), half4(0.5h,0.5h,0.5h,-0.5h), half4(1.5h,0.5h,0.5h,-0.5h),
    half4(-1.5h,1.5h,0.5h,-0.5h), half4(-0.5h,1.5h,0.5h,-0.5h), half4(0.5h,1.5h,0.5h,-0.5h), half4(1.5h,1.5h,0.5h,-0.5h),
    half4(-1.5h,-1.5h,1.5h,-0.5h), half4(-0.5h,-1.5h,1.5h,-0.5h), half4(0.5h,-1.5h,1.5h,-0.5h), half4(1.5h,-1.5h,1.5h,-0.5h),
    half4(-1.5h,-0.5h,1.5h,-0.5h), half4(-0.5h,-0.5h,1.5h,-0.5h), half4(0.5h,-0.5h,1.5h,-0.5h), half4(1.5h,-0.5h,1.5h,-0.5h),
    half4(-1.5h,0.5h,1.5h,-0.5h), half4(-0.5h,0.5h,1.5h,-0.5h), half4(0.5h,0.5h,1.5h,-0.5h), half4(1.5h,0.5h,1.5h,-0.5h),
    half4(-1.5h,1.5h,1.5h,-0.5h), half4(-0.5h,1.5h,1.5h,-0.5h), half4(0.5h,1.5h,1.5h,-0.5h), half4(1.5h,1.5h,1.5h,-0.5h),
    half4(-1.5h,-1.5h,-1.5h,0.5h), half4(-0.5h,-1.5h,-1.5h,0.5h), half4(0.5h,-1.5h,-1.5h,0.5h), half4(1.5h,-1.5h,-1.5h,0.5h),
    half4(-1.5h,-0.5h,-1.5h,0.5h), half4(-0.5h,-0.5h,-1.5h,0.5h), half4(0.5h,-0.5h,-1.5h,0.5h), half4(1.5h,-0.5h,-1.5h,0.5h),
    half4(-1.5h,0.5h,-1.5h,0.5h), half4(-0.5h,0.5h,-1.5h,0.5h), half4(0.5h,0.5h,-1.5h,0.5h), half4(1.5h,0.5h,-1.5h,0.5h),
    half4(-1.5h,1.5h,-1.5h,0.5h), half4(-0.5h,1.5h,-1.5h,0.5h), half4(0.5h,1.5h,-1.5h,0.5h), half4(1.5h,1.5h,-1.5h,0.5h),
    half4(-1.5h,-1.5h,-0.5h,0.5h), half4(-0.5h,-1.5h,-0.5h,0.5h), half4(0.5h,-1.5h,-0.5h,0.5h), half4(1.5h,-1.5h,-0.5h,0.5h),
    half4(-1.5h,-0.5h,-0.5h,0.5h), half4(-0.5h,-0.5h,-0.5h,0.5h), half4(0.5h,-0.5h,-0.5h,0.5h), half4(1.5h,-0.5h,-0.5h,0.5h),
    half4(-1.5h,0.5h,-0.5h,0.5h), half4(-0.5h,0.5h,-0.5h,0.5h), half4(0.5h,0.5h,-0.5h,0.5h), half4(1.5h,0.5h,-0.5h,0.5h),
    half4(-1.5h,1.5h,-0.5h,0.5h), half4(-0.5h,1.5h,-0.5h,0.5h), half4(0.5h,1.5h,-0.5h,0.5h), half4(1.5h,1.5h,-0.5h,0.5h),
    half4(-1.5h,-1.5h,0.5h,0.5h), half4(-0.5h,-1.5h,0.5h,0.5h), half4(0.5h,-1.5h,0.5h,0.5h), half4(1.5h,-1.5h,0.5h,0.5h),
    half4(-1.5h,-0.5h,0.5h,0.5h), half4(-0.5h,-0.5h,0.5h,0.5h), half4(0.5h,-0.5h,0.5h,0.5h), half4(1.5h,-0.5h,0.5h,0.5h),
    half4(-1.5h,0.5h,0.5h,0.5h), half4(-0.5h,0.5h,0.5h,0.5h), half4(0.5h,0.5h,0.5h,0.5h), half4(1.5h,0.5h,0.5h,0.5h),
    half4(-1.5h,1.5h,0.5h,0.5h), half4(-0.5h,1.5h,0.5h,0.5h), half4(0.5h,1.5h,0.5h,0.5h), half4(1.5h,1.5h,0.5h,0.5h),
    half4(-1.5h,-1.5h,1.5h,0.5h), half4(-0.5h,-1.5h,1.5h,0.5h), half4(0.5h,-1.5h,1.5h,0.5h), half4(1.5h,-1.5h,1.5h,0.5h),
    half4(-1.5h,-0.5h,1.5h,0.5h), half4(-0.5h,-0.5h,1.5h,0.5h), half4(0.5h,-0.5h,1.5h,0.5h), half4(1.5h,-0.5h,1.5h,0.5h),
    half4(-1.5h,0.5h,1.5h,0.5h), half4(-0.5h,0.5h,1.5h,0.5h), half4(0.5h,0.5h,1.5h,0.5h), half4(1.5h,0.5h,1.5h,0.5h),
    half4(-1.5h,1.5h,1.5h,0.5h), half4(-0.5h,1.5h,1.5h,0.5h), half4(0.5h,1.5h,1.5h,0.5h), half4(1.5h,1.5h,1.5h,0.5h),
    half4(-1.5h,-1.5h,-1.5h,1.5h), half4(-0.5h,-1.5h,-1.5h,1.5h), half4(0.5h,-1.5h,-1.5h,1.5h), half4(1.5h,-1.5h,-1.5h,1.5h),
    half4(-1.5h,-0.5h,-1.5h,1.5h), half4(-0.5h,-0.5h,-1.5h,1.5h), half4(0.5h,-0.5h,-1.5h,1.5h), half4(1.5h,-0.5h,-1.5h,1.5h),
    half4(-1.5h,0.5h,-1.5h,1.5h), half4(-0.5h,0.5h,-1.5h,1.5h), half4(0.5h,0.5h,-1.5h,1.5h), half4(1.5h,0.5h,-1.5h,1.5h),
    half4(-1.5h,1.5h,-1.5h,1.5h), half4(-0.5h,1.5h,-1.5h,1.5h), half4(0.5h,1.5h,-1.5h,1.5h), half4(1.5h,1.5h,-1.5h,1.5h),
    half4(-1.5h,-1.5h,-0.5h,1.5h), half4(-0.5h,-1.5h,-0.5h,1.5h), half4(0.5h,-1.5h,-0.5h,1.5h), half4(1.5h,-1.5h,-0.5h,1.5h),
    half4(-1.5h,-0.5h,-0.5h,1.5h), half4(-0.5h,-0.5h,-0.5h,1.5h), half4(0.5h,-0.5h,-0.5h,1.5h), half4(1.5h,-0.5h,-0.5h,1.5h),
    half4(-1.5h,0.5h,-0.5h,1.5h), half4(-0.5h,0.5h,-0.5h,1.5h), half4(0.5h,0.5h,-0.5h,1.5h), half4(1.5h,0.5h,-0.5h,1.5h),
    half4(-1.5h,1.5h,-0.5h,1.5h), half4(-0.5h,1.5h,-0.5h,1.5h), half4(0.5h,1.5h,-0.5h,1.5h), half4(1.5h,1.5h,-0.5h,1.5h),
    half4(-1.5h,-1.5h,0.5h,1.5h), half4(-0.5h,-1.5h,0.5h,1.5h), half4(0.5h,-1.5h,0.5h,1.5h), half4(1.5h,-1.5h,0.5h,1.5h),
    half4(-1.5h,-0.5h,0.5h,1.5h), half4(-0.5h,-0.5h,0.5h,1.5h), half4(0.5h,-0.5h,0.5h,1.5h), half4(1.5h,-0.5h,0.5h,1.5h),
    half4(-1.5h,0.5h,0.5h,1.5h), half4(-0.5h,0.5h,0.5h,1.5h), half4(0.5h,0.5h,0.5h,1.5h), half4(1.5h,0.5h,0.5h,1.5h),
    half4(-1.5h,1.5h,0.5h,1.5h), half4(-0.5h,1.5h,0.5h,1.5h), half4(0.5h,1.5h,0.5h,1.5h), half4(1.5h,1.5h,0.5h,1.5h),
    half4(-1.5h,-1.5h,1.5h,1.5h), half4(-0.5h,-1.5h,1.5h,1.5h), half4(0.5h,-1.5h,1.5h,1.5h), half4(1.5h,-1.5h,1.5h,1.5h),
    half4(-1.5h,-0.5h,1.5h,1.5h), half4(-0.5h,-0.5h,1.5h,1.5h), half4(0.5h,-0.5h,1.5h,1.5h), half4(1.5h,-0.5h,1.5h,1.5h),
    half4(-1.5h,0.5h,1.5h,1.5h), half4(-0.5h,0.5h,1.5h,1.5h), half4(0.5h,0.5h,1.5h,1.5h), half4(1.5h,0.5h,1.5h,1.5h),
    half4(-1.5h,1.5h,1.5h,1.5h), half4(-0.5h,1.5h,1.5h,1.5h), half4(0.5h,1.5h,1.5h,1.5h), half4(1.5h,1.5h,1.5h,1.5h),
};
constant half4 kMppS3Hi[16] = {
    half4(-2.0h,-2.0h,-2.0h,-2.0h), half4(2.0h,-2.0h,-2.0h,-2.0h), half4(-2.0h,2.0h,-2.0h,-2.0h), half4(2.0h,2.0h,-2.0h,-2.0h),
    half4(-2.0h,-2.0h,2.0h,-2.0h), half4(2.0h,-2.0h,2.0h,-2.0h), half4(-2.0h,2.0h,2.0h,-2.0h), half4(2.0h,2.0h,2.0h,-2.0h),
    half4(-2.0h,-2.0h,-2.0h,2.0h), half4(2.0h,-2.0h,-2.0h,2.0h), half4(-2.0h,2.0h,-2.0h,2.0h), half4(2.0h,2.0h,-2.0h,2.0h),
    half4(-2.0h,-2.0h,2.0h,2.0h), half4(2.0h,-2.0h,2.0h,2.0h), half4(-2.0h,2.0h,2.0h,2.0h), half4(2.0h,2.0h,2.0h,2.0h),
};
constant half2 kMppS4Lut[256] = {
    half2(-7.5h,-7.5h), half2(-6.5h,-7.5h), half2(-5.5h,-7.5h), half2(-4.5h,-7.5h), half2(-3.5h,-7.5h), half2(-2.5h,-7.5h), half2(-1.5h,-7.5h), half2(-0.5h,-7.5h),
    half2(0.5h,-7.5h), half2(1.5h,-7.5h), half2(2.5h,-7.5h), half2(3.5h,-7.5h), half2(4.5h,-7.5h), half2(5.5h,-7.5h), half2(6.5h,-7.5h), half2(7.5h,-7.5h),
    half2(-7.5h,-6.5h), half2(-6.5h,-6.5h), half2(-5.5h,-6.5h), half2(-4.5h,-6.5h), half2(-3.5h,-6.5h), half2(-2.5h,-6.5h), half2(-1.5h,-6.5h), half2(-0.5h,-6.5h),
    half2(0.5h,-6.5h), half2(1.5h,-6.5h), half2(2.5h,-6.5h), half2(3.5h,-6.5h), half2(4.5h,-6.5h), half2(5.5h,-6.5h), half2(6.5h,-6.5h), half2(7.5h,-6.5h),
    half2(-7.5h,-5.5h), half2(-6.5h,-5.5h), half2(-5.5h,-5.5h), half2(-4.5h,-5.5h), half2(-3.5h,-5.5h), half2(-2.5h,-5.5h), half2(-1.5h,-5.5h), half2(-0.5h,-5.5h),
    half2(0.5h,-5.5h), half2(1.5h,-5.5h), half2(2.5h,-5.5h), half2(3.5h,-5.5h), half2(4.5h,-5.5h), half2(5.5h,-5.5h), half2(6.5h,-5.5h), half2(7.5h,-5.5h),
    half2(-7.5h,-4.5h), half2(-6.5h,-4.5h), half2(-5.5h,-4.5h), half2(-4.5h,-4.5h), half2(-3.5h,-4.5h), half2(-2.5h,-4.5h), half2(-1.5h,-4.5h), half2(-0.5h,-4.5h),
    half2(0.5h,-4.5h), half2(1.5h,-4.5h), half2(2.5h,-4.5h), half2(3.5h,-4.5h), half2(4.5h,-4.5h), half2(5.5h,-4.5h), half2(6.5h,-4.5h), half2(7.5h,-4.5h),
    half2(-7.5h,-3.5h), half2(-6.5h,-3.5h), half2(-5.5h,-3.5h), half2(-4.5h,-3.5h), half2(-3.5h,-3.5h), half2(-2.5h,-3.5h), half2(-1.5h,-3.5h), half2(-0.5h,-3.5h),
    half2(0.5h,-3.5h), half2(1.5h,-3.5h), half2(2.5h,-3.5h), half2(3.5h,-3.5h), half2(4.5h,-3.5h), half2(5.5h,-3.5h), half2(6.5h,-3.5h), half2(7.5h,-3.5h),
    half2(-7.5h,-2.5h), half2(-6.5h,-2.5h), half2(-5.5h,-2.5h), half2(-4.5h,-2.5h), half2(-3.5h,-2.5h), half2(-2.5h,-2.5h), half2(-1.5h,-2.5h), half2(-0.5h,-2.5h),
    half2(0.5h,-2.5h), half2(1.5h,-2.5h), half2(2.5h,-2.5h), half2(3.5h,-2.5h), half2(4.5h,-2.5h), half2(5.5h,-2.5h), half2(6.5h,-2.5h), half2(7.5h,-2.5h),
    half2(-7.5h,-1.5h), half2(-6.5h,-1.5h), half2(-5.5h,-1.5h), half2(-4.5h,-1.5h), half2(-3.5h,-1.5h), half2(-2.5h,-1.5h), half2(-1.5h,-1.5h), half2(-0.5h,-1.5h),
    half2(0.5h,-1.5h), half2(1.5h,-1.5h), half2(2.5h,-1.5h), half2(3.5h,-1.5h), half2(4.5h,-1.5h), half2(5.5h,-1.5h), half2(6.5h,-1.5h), half2(7.5h,-1.5h),
    half2(-7.5h,-0.5h), half2(-6.5h,-0.5h), half2(-5.5h,-0.5h), half2(-4.5h,-0.5h), half2(-3.5h,-0.5h), half2(-2.5h,-0.5h), half2(-1.5h,-0.5h), half2(-0.5h,-0.5h),
    half2(0.5h,-0.5h), half2(1.5h,-0.5h), half2(2.5h,-0.5h), half2(3.5h,-0.5h), half2(4.5h,-0.5h), half2(5.5h,-0.5h), half2(6.5h,-0.5h), half2(7.5h,-0.5h),
    half2(-7.5h,0.5h), half2(-6.5h,0.5h), half2(-5.5h,0.5h), half2(-4.5h,0.5h), half2(-3.5h,0.5h), half2(-2.5h,0.5h), half2(-1.5h,0.5h), half2(-0.5h,0.5h),
    half2(0.5h,0.5h), half2(1.5h,0.5h), half2(2.5h,0.5h), half2(3.5h,0.5h), half2(4.5h,0.5h), half2(5.5h,0.5h), half2(6.5h,0.5h), half2(7.5h,0.5h),
    half2(-7.5h,1.5h), half2(-6.5h,1.5h), half2(-5.5h,1.5h), half2(-4.5h,1.5h), half2(-3.5h,1.5h), half2(-2.5h,1.5h), half2(-1.5h,1.5h), half2(-0.5h,1.5h),
    half2(0.5h,1.5h), half2(1.5h,1.5h), half2(2.5h,1.5h), half2(3.5h,1.5h), half2(4.5h,1.5h), half2(5.5h,1.5h), half2(6.5h,1.5h), half2(7.5h,1.5h),
    half2(-7.5h,2.5h), half2(-6.5h,2.5h), half2(-5.5h,2.5h), half2(-4.5h,2.5h), half2(-3.5h,2.5h), half2(-2.5h,2.5h), half2(-1.5h,2.5h), half2(-0.5h,2.5h),
    half2(0.5h,2.5h), half2(1.5h,2.5h), half2(2.5h,2.5h), half2(3.5h,2.5h), half2(4.5h,2.5h), half2(5.5h,2.5h), half2(6.5h,2.5h), half2(7.5h,2.5h),
    half2(-7.5h,3.5h), half2(-6.5h,3.5h), half2(-5.5h,3.5h), half2(-4.5h,3.5h), half2(-3.5h,3.5h), half2(-2.5h,3.5h), half2(-1.5h,3.5h), half2(-0.5h,3.5h),
    half2(0.5h,3.5h), half2(1.5h,3.5h), half2(2.5h,3.5h), half2(3.5h,3.5h), half2(4.5h,3.5h), half2(5.5h,3.5h), half2(6.5h,3.5h), half2(7.5h,3.5h),
    half2(-7.5h,4.5h), half2(-6.5h,4.5h), half2(-5.5h,4.5h), half2(-4.5h,4.5h), half2(-3.5h,4.5h), half2(-2.5h,4.5h), half2(-1.5h,4.5h), half2(-0.5h,4.5h),
    half2(0.5h,4.5h), half2(1.5h,4.5h), half2(2.5h,4.5h), half2(3.5h,4.5h), half2(4.5h,4.5h), half2(5.5h,4.5h), half2(6.5h,4.5h), half2(7.5h,4.5h),
    half2(-7.5h,5.5h), half2(-6.5h,5.5h), half2(-5.5h,5.5h), half2(-4.5h,5.5h), half2(-3.5h,5.5h), half2(-2.5h,5.5h), half2(-1.5h,5.5h), half2(-0.5h,5.5h),
    half2(0.5h,5.5h), half2(1.5h,5.5h), half2(2.5h,5.5h), half2(3.5h,5.5h), half2(4.5h,5.5h), half2(5.5h,5.5h), half2(6.5h,5.5h), half2(7.5h,5.5h),
    half2(-7.5h,6.5h), half2(-6.5h,6.5h), half2(-5.5h,6.5h), half2(-4.5h,6.5h), half2(-3.5h,6.5h), half2(-2.5h,6.5h), half2(-1.5h,6.5h), half2(-0.5h,6.5h),
    half2(0.5h,6.5h), half2(1.5h,6.5h), half2(2.5h,6.5h), half2(3.5h,6.5h), half2(4.5h,6.5h), half2(5.5h,6.5h), half2(6.5h,6.5h), half2(7.5h,6.5h),
    half2(-7.5h,7.5h), half2(-6.5h,7.5h), half2(-5.5h,7.5h), half2(-4.5h,7.5h), half2(-3.5h,7.5h), half2(-2.5h,7.5h), half2(-1.5h,7.5h), half2(-0.5h,7.5h),
    half2(0.5h,7.5h), half2(1.5h,7.5h), half2(2.5h,7.5h), half2(3.5h,7.5h), half2(4.5h,7.5h), half2(5.5h,7.5h), half2(6.5h,7.5h), half2(7.5h,7.5h),
};
constant half4 kMppS5Hi[16] = {
    half4(-8.0h,-8.0h,-8.0h,-8.0h), half4(8.0h,-8.0h,-8.0h,-8.0h), half4(-8.0h,8.0h,-8.0h,-8.0h), half4(8.0h,8.0h,-8.0h,-8.0h),
    half4(-8.0h,-8.0h,8.0h,-8.0h), half4(8.0h,-8.0h,8.0h,-8.0h), half4(-8.0h,8.0h,8.0h,-8.0h), half4(8.0h,8.0h,8.0h,-8.0h),
    half4(-8.0h,-8.0h,-8.0h,8.0h), half4(8.0h,-8.0h,-8.0h,8.0h), half4(-8.0h,8.0h,-8.0h,8.0h), half4(8.0h,8.0h,-8.0h,8.0h),
    half4(-8.0h,-8.0h,8.0h,8.0h), half4(8.0h,-8.0h,8.0h,8.0h), half4(-8.0h,8.0h,8.0h,8.0h), half4(8.0h,8.0h,8.0h,8.0h),
};

// Constant-cache LUT .strata S_K decoder (10-15x fewer ALU ops on S2_K..S6_K!)
template <int F>
static inline void mpp_sk_w8(device const uchar* pl, int j, thread float4& o0, thread float4& o1) {
    if constexpr (F == 2) {
        device const uchar* p = pl + (j >> 2);
        o0 = float4(kMppS2Lut[p[0]]);
        o1 = float4(kMppS2Lut[p[1]]);
    } else if constexpr (F == 3) {
        device const uchar* p = pl + (j >> 2);
        const uint h = uint(pl[64 + (j >> 3)]);
        o0 = float4(kMppS2Lut[p[0]] + kMppS3Hi[h & 15u]);
        o1 = float4(kMppS2Lut[p[1]] + kMppS3Hi[h >> 4]);
    } else if constexpr (F == 4) {
        device const uchar* p = pl + (j >> 1);
        o0 = float4(half4(kMppS4Lut[p[0]], kMppS4Lut[p[1]]));
        o1 = float4(half4(kMppS4Lut[p[2]], kMppS4Lut[p[3]]));
    } else if constexpr (F == 5) {
        device const uchar* p = pl + (j >> 1);
        const uint h = uint(pl[128 + (j >> 3)]);
        o0 = float4(half4(kMppS4Lut[p[0]], kMppS4Lut[p[1]]) + kMppS5Hi[h & 15u]);
        o1 = float4(half4(kMppS4Lut[p[2]], kMppS4Lut[p[3]]) + kMppS5Hi[h >> 4]);
    } else if constexpr (F == 6) {
        device const uchar* p = pl + (j >> 1);
        device const uchar* hp = pl + 128 + (j >> 2);
        o0 = float4(fma(kMppS2Lut[hp[0]], half4(16.0h), half4(kMppS4Lut[p[0]], kMppS4Lut[p[1]])));
        o1 = float4(fma(kMppS2Lut[hp[1]], half4(16.0h), half4(kMppS4Lut[p[2]], kMppS4Lut[p[3]])));
    } else {
        device const char* qp = (device const char*)(pl + j);
        o0 = float4(float(qp[0]), float(qp[1]), float(qp[2]), float(qp[3]));
        o1 = float4(float(qp[4]), float(qp[5]), float(qp[6]), float(qp[7]));
    }
}

// Exact SITU(g, u, beta=4.0, lb=25.0) — matches ink_metal_codecs.metal bit-for-bit!
static inline float mpp_situ_act(float g, float u) {
    const float sig = 1.0f / (1.0f + exp(-g));
    const float g_act = 4.0f * tanh(clamp(g * 0.25f, -15.0f, 15.0f)) * sig;
    const float u_act = 25.0f * tanh(clamp(u * 0.04f, -15.0f, 15.0f));
    return g_act * u_act;
}

struct InkExpertDevMpp {
    ulong body_off;
    uint dn_off;
    uchar gu_code;
    uchar dn_code;
    ushort pad;
};

// ============================================================================
// 1. Cast / Init-Time Weight Conversion + Metal 4 matmul2d Dense Trunk GEMMs
// ============================================================================

kernel void ink_mpp_f32_to_f16(
    device const float* src [[buffer(0)]],
    device half* dst        [[buffer(1)]],
    constant int& N         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid) * 4;
    if (i + 3 < N) {
        *((device half4*)(dst + i)) = half4(*((device const float4*)(src + i)));
    } else {
        for (int k = i; k < N; ++k) dst[k] = half(src[k]);
    }
}

kernel void ink_mpp_bf16_to_f16(
    device const ushort* src [[buffer(0)]],
    device half* dst         [[buffer(1)]],
    constant int& N          [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid) * 4;
    if (i + 3 < N) {
        const ushort4 v = *((device const ushort4*)(src + i));
        *((device half4*)(dst + i)) = half4(mpp_bf16_to_f32(v.x), mpp_bf16_to_f32(v.y),
                                            mpp_bf16_to_f32(v.z), mpp_bf16_to_f32(v.w));
    } else {
        for (int k = i; k < N; ++k) dst[k] = half(mpp_bf16_to_f32(src[k]));
    }
}

kernel void ink_mpp_q8row_to_f16(
    device const uchar* W [[buffer(0)]],
    device half* dst      [[buffer(1)]],
    constant int& N       [[buffer(2)]],
    constant int& K       [[buffer(3)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int r = int(tgp.y);
    if (r >= N) return;
    const float sc = ((device const float*)W)[r];
    device const char* row = (device const char*)(W + (size_t)N * 4) + (size_t)r * K;
    const int k = int(tgp.x) * 256 + int(tid);
    if (k < K) dst[(size_t)r * K + k] = half(float(row[k]) * sc);
}

kernel void ink_mpp_sk_to_f16(
    device const uchar* W [[buffer(0)]],
    device half* dst      [[buffer(1)]],
    constant int& N       [[buffer(2)]],
    constant int& K       [[buffer(3)]],
    constant int& F       [[buffer(4)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int r = int(tgp.y);
    if (r >= N) return;
    const int SB = mpp_sk_sb_bytes(F);
    const size_t rowb = (size_t)(K >> 8) * SB;
    const int col = (int(tgp.x) * 32 + int(tid)) * 8;
    if (col >= K) return;
    const int sbi = col >> 8, j = col & 255, g = j >> 6;
    device const uchar* blk = W + (size_t)r * rowb + (size_t)sbi * SB;
    const float eff = float(as_type<half>(mpp_ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
    float4 w0 = 0.0f, w1 = 0.0f;
    if (F == 2)      mpp_sk_w8<2>(blk + 6, j, w0, w1);
    else if (F == 3) mpp_sk_w8<3>(blk + 6, j, w0, w1);
    else if (F == 4) mpp_sk_w8<4>(blk + 6, j, w0, w1);
    else if (F == 5) mpp_sk_w8<5>(blk + 6, j, w0, w1);
    else if (F == 6) mpp_sk_w8<6>(blk + 6, j, w0, w1);
    else             mpp_sk_w8<8>(blk + 6, j, w0, w1);
    *((device half4*)(dst + (size_t)r * K + col + 0)) = half4(w0 * eff);
    *((device half4*)(dst + (size_t)r * K + col + 4)) = half4(w1 * eff);
}

struct InkMppMmArgs {
    int M, N, K, out_stride;
};

// Native Metal 4 matmul2d: Y[M, out_stride] = X_h[M, K] @ W_h[N, K]^T (FP16 operands -> FP32 output)
template <int TM, int TN, int SG>
kernel void ink_mpp_mm_f16(
    device half* W          [[buffer(0)]],
    device half* X          [[buffer(1)]],
    device float* Y         [[buffer(2)]],
    constant InkMppMmArgs& a [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]]) {
    tensor<device half,  dextents<int32_t, 2>, tensor_inline> tA(X, dextents<int32_t, 2>(a.K, a.M));
    tensor<device half,  dextents<int32_t, 2>, tensor_inline> tB(W, dextents<int32_t, 2>(a.K, a.N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(a.out_stride, a.M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
    op.run(mA, mB, mY);
}

#define INK_MPP_MM(TM, TN, SG) \
    template [[host_name("ink_mpp_mm_f16_" #TM "_" #TN "_" #SG)]] kernel decltype(ink_mpp_mm_f16<TM, TN, SG>) ink_mpp_mm_f16<TM, TN, SG>;
INK_MPP_MM(16, 64, 4)
INK_MPP_MM(32, 64, 4)
INK_MPP_MM(64, 64, 4)
INK_MPP_MM(64, 128, 4)
INK_MPP_MM(128, 64, 4)

// Split concatenated [Q(3072), R(384), Sinks(24), Gate(3072)] (stride=6552) into separate buffers
// grid (M, 26, 1), 256 threads
kernel void ink_mpp_split_qrsg(
    device const float* src [[buffer(0)]],
    device float* q_out     [[buffer(1)]],
    device float* r_out     [[buffer(2)]],
    device float* s_out     [[buffer(3)]],
    device float* g_out     [[buffer(4)]],
    constant int& stride    [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x);
    const int idx = int(tgp.y) * 256 + int(tid);
    if (idx >= 6552) return;
    const float v = src[(size_t)m * stride + idx];
    if (idx < 3072) {
        q_out[(size_t)m * 3072 + idx] = v;
    } else if (idx < 3456) {
        r_out[(size_t)m * 384 + (idx - 3072)] = v;
    } else if (idx < 3480) {
        s_out[(size_t)m * 24 + (idx - 3456)] = v;
    } else {
        g_out[(size_t)m * 3072 + (idx - 3480)] = v;
    }
}

// Split concatenated [K(1024), k_dyn(32)] or [V(1024), v_dyn(32)] (stride=1056)
// grid (M, 5, 1), 256 threads
kernel void ink_mpp_split_kv_dyn(
    device const float* src [[buffer(0)]],
    device float* kv_out    [[buffer(1)]],
    device float* dyn_out   [[buffer(2)]],
    constant int& has_dyn   [[buffer(3)]],
    constant int& stride    [[buffer(4)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x);
    const int idx = int(tgp.y) * 256 + int(tid);
    if (idx < 1024) {
        kv_out[(size_t)m * 1024 + idx] = src[(size_t)m * stride + idx];
    } else if (has_dyn && idx < 1056) {
        dyn_out[(size_t)m * 32 + (idx - 1024)] = src[(size_t)m * stride + idx];
    }
}

// Fused controller apply + RMSNorm reading concatenated [gain(2304), forget(2304)] (stride=4608)!
// grid (M, 1, 1), 256 threads
kernel void ink_mpp_ctrl_fuse_gf_norm(
    device const float* cur_buf [[buffer(0)]],
    device const float* msg_buf [[buffer(1)]],
    device const float* gf_buf  [[buffer(2)]],  // [M, 4608]
    device const float* norm_w  [[buffer(3)]],
    device float* out_buf       [[buffer(4)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* cp = cur_buf + (size_t)m * 2304;
    device const float* mp = msg_buf + (size_t)m * 2304;
    device const float* gp = gf_buf + (size_t)m * 4608;
    device const float* fp = gp + 2304;
    device float* dst = out_buf + (size_t)m * 2304;
    float ss = 0.0f;
    for (uint d = tid; d < 2304u; d += 256u) {
        const float v = (1.0f - tanh(clamp(fp[d], -15.0f, 15.0f))) * cp[d]
                      + tanh(clamp(gp[d], -15.0f, 15.0f)) * mp[d];
        dst[d] = v;
        ss += v * v;
    }
    float s = simd_sum(ss);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31u) == 0u) sh[tid >> 5] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += sh[i];
    const float inv = rsqrt(tot / 2304.0f + 1e-6f);
    for (uint d = tid; d < 2304u; d += 256u) dst[d] = dst[d] * inv * norm_w[d];
}

// ============================================================================
// 2. Zero-DRAM-Write Multi-Tile Direct Metal 4 matmul2d Grouped-Expert MoE
//    - ZERO FP16 DRAM weight writes (eliminates ~140 GB DRAM traffic at P=1024!)
//    - 128-row Super-Tiles (NTILE=2 x TM=64) decode 8 KB compressed weights ONCE into SLM
//      and run 2x matmul2d(64, 64, 64) in registers -> 2x weight reuse, 0 extra DRAM!
//    - Specialized SF4 (CODEC=15), MX8 (CODEC=16), and General (CODEC=0) pipelines
//      with 16-byte vectorized loads + L2-swizzled grid (n_tiles, 36, 1)!
// ============================================================================

kernel void ink_mpp_moe_plan(
    device const int* topk_idx [[buffer(0)]],
    constant int& P            [[buffer(1)]],
    constant int& allow_128    [[buffer(2)]],  // 1 for M >= 128, 0 for M < 128
    device int* srt_pair       [[buffer(3)]],  // [total_rows] (-1 for padded rows)
    device int* pair_pos       [[buffer(4)]],  // [P] -> row index in sorted layout
    device int4* tiles_128     [[buffer(5)]],  // [n128] = (e, row0, valid_rows, 0)
    device int4* tiles_64      [[buffer(6)]],  // [n64]  = (e, row0, valid_rows, 0)
    device int4* tiles_32      [[buffer(7)]],  // [n32]  = (e, row0, valid_rows, 0)
    device int4* tiles_16      [[buffer(8)]],  // [n16]  = (e, row0, valid_rows, 0)
    device uint* meta          [[buffer(9)]],  // indirect dispatch args (24 uints)
    uint e    [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup int sg_r[4];
    threadgroup int sg_128[4];
    threadgroup int sg_64[4];
    threadgroup int sg_32[4];
    threadgroup int sg_16[4];

    int c = 0;
    for (int i = 0; i < P; ++i) if (topk_idx[i] == int(e)) ++c;

    // Exact 4-bucket binary tile decomposition (128 / 64 / 32 / 16): <= 15 wasted rows/expert (<4.5% waste at P=1024!)
    int n128 = 0, n64 = 0, n32 = 0, n16 = 0;
    if (c > 0) {
        int rem = c;
        if (allow_128) {
            n128 = rem >> 7;
            rem &= 127;
            if (rem > 112) { ++n128; rem = 0; }
        }
        if (rem > 0) {
            n64 = rem >> 6;
            rem &= 63;
            if (rem > 48) { ++n64; rem = 0; }
        }
        if (rem > 0) {
            n32 = rem >> 5;
            rem &= 31;
            if (rem > 16) { ++n32; rem = 0; }
        }
        if (rem > 0) {
            n16 = 1;
        }
    }
    const int rows = n128 * 128 + n64 * 64 + n32 * 32 + n16 * 16;

    const int ex_r   = simd_prefix_exclusive_sum(rows);
    const int ex_128 = simd_prefix_exclusive_sum(n128);
    const int ex_64  = simd_prefix_exclusive_sum(n64);
    const int ex_32  = simd_prefix_exclusive_sum(n32);
    const int ex_16  = simd_prefix_exclusive_sum(n16);
    if (lane == 31) {
        sg_r[sg]   = ex_r + rows;
        sg_128[sg] = ex_128 + n128;
        sg_64[sg]  = ex_64 + n64;
        sg_32[sg]  = ex_32 + n32;
        sg_16[sg]  = ex_16 + n16;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int br = 0, b128 = 0, b64 = 0, b32 = 0, b16 = 0;
    for (uint w = 0; w < sg; ++w) {
        br += sg_r[w]; b128 += sg_128[w]; b64 += sg_64[w]; b32 += sg_32[w]; b16 += sg_16[w];
    }
    const int roff = br + ex_r;
    const int o128 = b128 + ex_128;
    const int o64  = b64 + ex_64;
    const int o32  = b32 + ex_32;
    const int o16  = b16 + ex_16;

    int cur_r = roff, rem_c = c;
    for (int j = 0; j < n128; ++j) {
        const int v = min(128, rem_c);
        tiles_128[o128 + j] = int4(int(e), cur_r, v, 0);
        cur_r += 128; rem_c -= v;
    }
    for (int j = 0; j < n64; ++j) {
        const int v = min(64, rem_c);
        tiles_64[o64 + j] = int4(int(e), cur_r, v, 0);
        cur_r += 64; rem_c -= v;
    }
    for (int j = 0; j < n32; ++j) {
        const int v = min(32, rem_c);
        tiles_32[o32 + j] = int4(int(e), cur_r, v, 0);
        cur_r += 32; rem_c -= v;
    }
    for (int j = 0; j < n16; ++j) {
        const int v = min(16, rem_c);
        tiles_16[o16 + j] = int4(int(e), cur_r, v, 0);
        cur_r += 16; rem_c -= v;
    }

    for (int r = 0; r < rows; ++r) srt_pair[roff + r] = -1;
    if (c > 0) {
        int k = 0;
        for (int i = 0; i < P; ++i) {
            if (topk_idx[i] == int(e)) {
                const int row = roff + k++;
                srt_pair[row] = i;
                pair_pos[i] = row;
            }
        }
    }
    if (e == 127) {
        const uint t128 = uint(o128 + n128);
        const uint t64  = uint(o64 + n64);
        const uint t32  = uint(o32 + n32);
        const uint t16  = uint(o16 + n16);
        const uint tot_rows = uint(roff + rows);
        meta[0] = t128; meta[1] = t64; meta[2] = t32; meta[3] = tot_rows;
        // [4..6] gather: (tot_rows, 1, 1)
        meta[4] = tot_rows; meta[5] = 1u; meta[6] = 1u;
        // [7..9] dir_128 (L2-swizzled!): (t128, 36, 1)
        meta[7] = t128; meta[8] = 36u; meta[9] = 1u;
        // [10..12] dir_64 (L2-swizzled!): (t64, 36, 1)
        meta[10] = t64; meta[11] = 36u; meta[12] = 1u;
        // [13..15] dir_32 (L2-swizzled!): (t32, 36, 1)
        meta[13] = t32; meta[14] = 36u; meta[15] = 1u;
        // [16..18] dir_16 (L2-swizzled!): (t16, 36, 1)
        meta[16] = t16; meta[17] = 36u; meta[18] = 1u;
        meta[19] = t16;
    }
}

// Gather F32 x_in [M, 2304] into tile-aligned contiguous half XS [tot_rows, 2304]
// indirect grid (tot_rows, 1, 1), 256 threads
kernel void ink_mpp_moe_gather(
    device const float* x_in   [[buffer(0)]],
    device const int* srt_pair [[buffer(1)]],
    device half* XS            [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const int p = srt_pair[row];
    device half* dst = XS + (size_t)row * 2304;
    if (p >= 0) {
        device const float* src = x_in + (size_t)(p >> 3) * 2304;
        for (uint k4 = tid; k4 < 2304u / 4u; k4 += 256u) {
            *((device half4*)(dst + k4 * 4u)) = half4(*((device const float4*)(src + k4 * 4u)));
        }
    } else {
        for (uint k4 = tid; k4 < 2304u / 4u; k4 += 256u) {
            *((device half4*)(dst + k4 * 4u)) = half4(0.0h);
        }
    }
}

// Fast 32-weight vectorized decoders (FP32 scale multiply before half4 cast -> avoids FP16 scale underflow on MX8/SF4!)
static inline void mpp_dec_sf4_32(device const uchar* p_base, device const uchar* s_base, float outer,
                                  int row_pbytes, int row_sbytes, int gn, int k0, int half_idx, threadgroup half4* dst) {
    const int blk = (k0 >> 4) + half_idx * 2;
    const float sc0 = mpp_e4m3_to_f32(s_base[(size_t)gn * row_sbytes + blk + 0]) * outer;
    const float sc1 = mpp_e4m3_to_f32(s_base[(size_t)gn * row_sbytes + blk + 1]) * outer;
    const uint4 pk = *((device const uint4*)(p_base + (size_t)gn * row_pbytes + (size_t)blk * 8));
    dst[0] = half4(float4(kMppE2M1[pk.x & 15u], kMppE2M1[(pk.x >> 4) & 15u], kMppE2M1[(pk.x >> 8) & 15u], kMppE2M1[(pk.x >> 12) & 15u]) * sc0);
    dst[1] = half4(float4(kMppE2M1[(pk.x >> 16) & 15u], kMppE2M1[(pk.x >> 20) & 15u], kMppE2M1[(pk.x >> 24) & 15u], kMppE2M1[pk.x >> 28]) * sc0);
    dst[2] = half4(float4(kMppE2M1[pk.y & 15u], kMppE2M1[(pk.y >> 4) & 15u], kMppE2M1[(pk.y >> 8) & 15u], kMppE2M1[(pk.y >> 12) & 15u]) * sc0);
    dst[3] = half4(float4(kMppE2M1[(pk.y >> 16) & 15u], kMppE2M1[(pk.y >> 20) & 15u], kMppE2M1[(pk.y >> 24) & 15u], kMppE2M1[pk.y >> 28]) * sc0);
    dst[4] = half4(float4(kMppE2M1[pk.z & 15u], kMppE2M1[(pk.z >> 4) & 15u], kMppE2M1[(pk.z >> 8) & 15u], kMppE2M1[(pk.z >> 12) & 15u]) * sc1);
    dst[5] = half4(float4(kMppE2M1[(pk.z >> 16) & 15u], kMppE2M1[(pk.z >> 20) & 15u], kMppE2M1[(pk.z >> 24) & 15u], kMppE2M1[pk.z >> 28]) * sc1);
    dst[6] = half4(float4(kMppE2M1[pk.w & 15u], kMppE2M1[(pk.w >> 4) & 15u], kMppE2M1[(pk.w >> 8) & 15u], kMppE2M1[(pk.w >> 12) & 15u]) * sc1);
    dst[7] = half4(float4(kMppE2M1[(pk.w >> 16) & 15u], kMppE2M1[(pk.w >> 20) & 15u], kMppE2M1[(pk.w >> 24) & 15u], kMppE2M1[pk.w >> 28]) * sc1);
}

static inline void mpp_dec_mx8_32(device const uchar* w_base, device const uchar* s_base,
                                  int row_wbytes, int row_sbytes, int gn, int k0, int half_idx, threadgroup half4* dst) {
    const int col = k0 + half_idx * 32;
    const float sc = mpp_ue8m0_to_f32(s_base[(size_t)gn * row_sbytes + (col >> 5)]);
    device const uint4* wp = (device const uint4*)(w_base + (size_t)gn * row_wbytes + col);
    const uint4 a = wp[0], b = wp[1];
    dst[0] = half4(float4(mpp_e4m3_to_f32(a.x & 255u), mpp_e4m3_to_f32((a.x >> 8) & 255u), mpp_e4m3_to_f32((a.x >> 16) & 255u), mpp_e4m3_to_f32(a.x >> 24)) * sc);
    dst[1] = half4(float4(mpp_e4m3_to_f32(a.y & 255u), mpp_e4m3_to_f32((a.y >> 8) & 255u), mpp_e4m3_to_f32((a.y >> 16) & 255u), mpp_e4m3_to_f32(a.y >> 24)) * sc);
    dst[2] = half4(float4(mpp_e4m3_to_f32(a.z & 255u), mpp_e4m3_to_f32((a.z >> 8) & 255u), mpp_e4m3_to_f32((a.z >> 16) & 255u), mpp_e4m3_to_f32(a.z >> 24)) * sc);
    dst[3] = half4(float4(mpp_e4m3_to_f32(a.w & 255u), mpp_e4m3_to_f32((a.w >> 8) & 255u), mpp_e4m3_to_f32((a.w >> 16) & 255u), mpp_e4m3_to_f32(a.w >> 24)) * sc);
    dst[4] = half4(float4(mpp_e4m3_to_f32(b.x & 255u), mpp_e4m3_to_f32((b.x >> 8) & 255u), mpp_e4m3_to_f32((b.x >> 16) & 255u), mpp_e4m3_to_f32(b.x >> 24)) * sc);
    dst[5] = half4(float4(mpp_e4m3_to_f32(b.y & 255u), mpp_e4m3_to_f32((b.y >> 8) & 255u), mpp_e4m3_to_f32((b.y >> 16) & 255u), mpp_e4m3_to_f32(b.y >> 24)) * sc);
    dst[6] = half4(float4(mpp_e4m3_to_f32(b.z & 255u), mpp_e4m3_to_f32((b.z >> 8) & 255u), mpp_e4m3_to_f32((b.z >> 16) & 255u), mpp_e4m3_to_f32(b.z >> 24)) * sc);
    dst[7] = half4(float4(mpp_e4m3_to_f32(b.w & 255u), mpp_e4m3_to_f32((b.w >> 8) & 255u), mpp_e4m3_to_f32((b.w >> 16) & 255u), mpp_e4m3_to_f32(b.w >> 24)) * sc);
}

template <int F>
static inline void mpp_dec_sk_32_t(device const uchar* row_ptr, int col, threadgroup half4* dst) {
    constexpr int SB = mpp_sk_sb_bytes(F);
    const int sbi = col >> 8, j = col & 255, g = j >> 6;
    device const uchar* blk = row_ptr + (size_t)sbi * SB;
    const float eff = float(as_type<half>(mpp_ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
    for (int q = 0; q < 4; ++q) {
        float4 w0, w1;
        mpp_sk_w8<F>(blk + 6, j + q * 8, w0, w1);
        dst[q * 2 + 0] = half4(w0 * eff);
        dst[q * 2 + 1] = half4(w1 * eff);
    }
}

static inline void mpp_dec_sk_32(device const uchar* row_ptr, int code, int col, threadgroup half4* dst) {
    if (code == 3)      mpp_dec_sk_32_t<3>(row_ptr, col, dst);
    else if (code == 2) mpp_dec_sk_32_t<2>(row_ptr, col, dst);
    else if (code == 4) mpp_dec_sk_32_t<4>(row_ptr, col, dst);
    else if (code == 5) mpp_dec_sk_32_t<5>(row_ptr, col, dst);
    else if (code == 6) mpp_dec_sk_32_t<6>(row_ptr, col, dst);
    else if (code == 8) mpp_dec_sk_32_t<8>(row_ptr, col, dst);
}

// Direct Zero-DRAM-Write Grouped-Expert w13 + fused exact SITU (TK=64 -> 8 KB sB, max occupancy!)
// CODEC: 15=SF4, 16=MX8, 0=Generic | NTILE: 1 (TM rows) or 2 (2*TM=128 rows, 2x weight reuse!)
// indirect grid (n_tiles, 36, 1) [L2-swizzled!], 128 threads (4 SGs)
template <int CODEC, int TM, int NTILE>
kernel void ink_mpp_moe_gu_dir(
    device const uchar* sec4           [[buffer(0)]],
    device const InkExpertDevMpp* ex   [[buffer(1)]],
    device half* XS                    [[buffer(2)]],
    device half* HS                    [[buffer(3)]],
    device const int4* tiles           [[buffer(4)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TN = 64, TK = 64, SG = 4, NTH = 128;
    constexpr int W_HALVES = TN * TK;       // 64*64 = 8 KB
    constexpr int O_HALVES = 2 * TM * TN;   // 16 KB (TM=64) / 8 KB (TM=32) / 4 KB (TM=16)
    constexpr int S_HALVES = W_HALVES > O_HALVES ? W_HALVES : O_HALVES;
    threadgroup half sMem[S_HALVES];

    const int4 tl = tiles[tg.x];
    const int e = tl.x, row0 = tl.y;
    const int n0 = int(tg.y) * TN;
    const InkExpertDevMpp ed = ex[e];
    device const uchar* body = sec4 + ed.body_off;
    const bool is_sk_dn = (ed.dn_code != 15 && ed.dn_code != 16);
    const int gcode = (CODEC == 0) ? int(ed.gu_code) : CODEC;
    const float sf4_outer = (gcode == 15) ? (((device const float*)body)[0] / ((device const float*)body)[1]) : 0.0f;
    const size_t sk_rowb = (CODEC == 0 && gcode != 15 && gcode != 16) ? (9ul * size_t(mpp_sk_sb_bytes(gcode))) : 0ul;
    const int br = int(tid >> 1), hi = int(tid & 1u);
    const int gn = n0 + br;

    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX(XS, dextents<int32_t, 2>(2304, 1 << 24));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB(sMem, dextents<int32_t, 2>(TK, TN));
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto tA0 = tX.template slice<TK, TM>(0, row0);
    auto acc0 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    auto acc1 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
        if (acc0.is_valid_element(i)) {
            acc0[i] = 0.0f;
            if constexpr (NTILE == 2) acc1[i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k0 = 0; k0 < 2304; k0 += TK) {
        threadgroup half4* d = (threadgroup half4*)(sMem + br * TK + hi * 32);
        if constexpr (CODEC == 15) {
            mpp_dec_sf4_32(body + 16, body + 16 + 2654208, sf4_outer, 1152, 144, gn, k0, hi, d);
        } else if constexpr (CODEC == 16) {
            mpp_dec_mx8_32(body, body + 5308416, 2304, 72, gn, k0, hi, d);
        } else {
            if (gcode == 15) {
                mpp_dec_sf4_32(body + 16, body + 16 + 2654208, sf4_outer, 1152, 144, gn, k0, hi, d);
            } else if (gcode == 16) {
                mpp_dec_mx8_32(body, body + 5308416, 2304, 72, gn, k0, hi, d);
            } else {
                mpp_dec_sk_32(body + 2304 + (size_t)gn * sk_rowb, gcode, k0 + hi * 32, d);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA0 = tX.template slice<TK, TM>(k0, row0);
        op.run(mA0, tB, acc0);
        if constexpr (NTILE == 2) {
            auto mA1 = tX.template slice<TK, TM>(k0, row0 + TM);
            op.run(mA1, tB, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* ot = reinterpret_cast<threadgroup float*>(sMem);
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot, dextents<int32_t, 2>(TN, TM));
    constexpr int PAIRS = TM * 32;
    const int fbase = n0 >> 1;

    acc0.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = int(tid); idx < PAIRS; idx += NTH) {
        const int m = idx >> 5, f = idx & 31;
        const float g = ot[m * TN + 2 * f], u = ot[m * TN + 2 * f + 1];
        const float inv_sq = is_sk_dn ? (1.0f / float(as_type<half>(mpp_ld16(body + 2 * (fbase + f))))) : 1.0f;
        HS[(size_t)(row0 + m) * 1152 + fbase + f] = half(mpp_situ_act(g, u) * inv_sq);
    }
    if constexpr (NTILE == 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc1.store(tO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = int(tid); idx < PAIRS; idx += NTH) {
            const int m = idx >> 5, f = idx & 31;
            const float g = ot[m * TN + 2 * f], u = ot[m * TN + 2 * f + 1];
            const float inv_sq = is_sk_dn ? (1.0f / float(as_type<half>(mpp_ld16(body + 2 * (fbase + f))))) : 1.0f;
            HS[(size_t)(row0 + TM + m) * 1152 + fbase + f] = half(mpp_situ_act(g, u) * inv_sq);
        }
    }
}

// Direct Zero-DRAM-Write Grouped-Expert w2 (TK=64 -> 8 KB SLM, max occupancy!)
// indirect grid (n_tiles, 36, 1) [L2-swizzled!], 128 threads (4 SGs)
template <int CODEC, int TM, int NTILE>
kernel void ink_mpp_moe_dn_dir(
    device const uchar* sec4           [[buffer(0)]],
    device const InkExpertDevMpp* ex   [[buffer(1)]],
    device half* HS                    [[buffer(2)]],
    device float* YE                   [[buffer(3)]],
    device const int4* tiles           [[buffer(4)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TN = 64, TK = 64, SG = 4;
    threadgroup half sB[TN * TK];

    const int4 tl = tiles[tg.x];
    const int e = tl.x, row0 = tl.y;
    const int d0 = int(tg.y) * TN;
    const InkExpertDevMpp ed = ex[e];
    device const uchar* body = sec4 + ed.body_off;
    const int dcode = (CODEC == 0) ? int(ed.dn_code) : CODEC;
    const float sf4_outer = (dcode == 15) ? (((device const float*)body)[2] / ((device const float*)body)[3]) : 0.0f;
    const size_t sk_pairb = (CODEC == 0 && dcode != 15 && dcode != 16) ? (9ul * size_t(mpp_sk_sb_bytes(dcode))) : 0ul;
    const int br = int(tid >> 1), hi = int(tid & 1u);
    const int gn = d0 + br;

    tensor<device half,  dextents<int32_t, 2>, tensor_inline> tH(HS, dextents<int32_t, 2>(1152, 1 << 24));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(YE, dextents<int32_t, 2>(2304, 1 << 24));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB(sB, dextents<int32_t, 2>(TK, TN));
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto tA0 = tH.template slice<TK, TM>(0, row0);
    auto acc0 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    auto acc1 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
        if (acc0.is_valid_element(i)) {
            acc0[i] = 0.0f;
            if constexpr (NTILE == 2) acc1[i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k0 = 0; k0 < 1152; k0 += TK) {
        threadgroup half4* d = (threadgroup half4*)(sB + br * TK + hi * 32);
        if constexpr (CODEC == 15) {
            mpp_dec_sf4_32(body + ed.dn_off, body + ed.dn_off + 1327104, sf4_outer, 576, 72, gn, k0, hi, d);
        } else if constexpr (CODEC == 16) {
            mpp_dec_mx8_32(body + ed.dn_off, body + ed.dn_off + 2654208, 1152, 36, gn, k0, hi, d);
        } else {
            if (dcode == 15) {
                mpp_dec_sf4_32(body + ed.dn_off, body + ed.dn_off + 1327104, sf4_outer, 576, 72, gn, k0, hi, d);
            } else if (dcode == 16) {
                mpp_dec_mx8_32(body + ed.dn_off, body + ed.dn_off + 2654208, 1152, 36, gn, k0, hi, d);
            } else {
                const int rp = gn >> 1;
                const int col_p = ((gn & 1) ? 1152 : 0) + k0 + hi * 32;
                mpp_dec_sk_32(body + ed.dn_off + (size_t)rp * sk_pairb, dcode, col_p, d);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA0 = tH.template slice<TK, TM>(k0, row0);
        op.run(mA0, tB, acc0);
        if constexpr (NTILE == 2) {
            auto mA1 = tH.template slice<TK, TM>(k0, row0 + TM);
            op.run(mA1, tB, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    acc0.store(tY.slice(d0, row0));
    if constexpr (NTILE == 2) acc1.store(tY.slice(d0, row0 + TM));
}

#define INK_MPP_DIR_SET(CNAME, CODE) \
    template [[host_name("ink_mpp_moe_gu_" CNAME "_16")]]  kernel decltype(ink_mpp_moe_gu_dir<CODE, 16, 1>) ink_mpp_moe_gu_dir<CODE, 16, 1>; \
    template [[host_name("ink_mpp_moe_dn_" CNAME "_16")]]  kernel decltype(ink_mpp_moe_dn_dir<CODE, 16, 1>) ink_mpp_moe_dn_dir<CODE, 16, 1>; \
    template [[host_name("ink_mpp_moe_gu_" CNAME "_32")]]  kernel decltype(ink_mpp_moe_gu_dir<CODE, 32, 1>) ink_mpp_moe_gu_dir<CODE, 32, 1>; \
    template [[host_name("ink_mpp_moe_dn_" CNAME "_32")]]  kernel decltype(ink_mpp_moe_dn_dir<CODE, 32, 1>) ink_mpp_moe_dn_dir<CODE, 32, 1>; \
    template [[host_name("ink_mpp_moe_gu_" CNAME "_64")]]  kernel decltype(ink_mpp_moe_gu_dir<CODE, 64, 1>) ink_mpp_moe_gu_dir<CODE, 64, 1>; \
    template [[host_name("ink_mpp_moe_dn_" CNAME "_64")]]  kernel decltype(ink_mpp_moe_dn_dir<CODE, 64, 1>) ink_mpp_moe_dn_dir<CODE, 64, 1>; \
    template [[host_name("ink_mpp_moe_gu_" CNAME "_128")]] kernel decltype(ink_mpp_moe_gu_dir<CODE, 64, 2>) ink_mpp_moe_gu_dir<CODE, 64, 2>; \
    template [[host_name("ink_mpp_moe_dn_" CNAME "_128")]] kernel decltype(ink_mpp_moe_dn_dir<CODE, 64, 2>) ink_mpp_moe_dn_dir<CODE, 64, 2>;
INK_MPP_DIR_SET("sf4", 15)
INK_MPP_DIR_SET("mx8", 16)
INK_MPP_DIR_SET("gen", 0)

// Weighted reduce-8 from sorted YE rows -> moe_raw [M, 2304]
// grid (M, 9, 1), 256 threads
kernel void ink_mpp_moe_combine(
    device const float* YE       [[buffer(0)]],
    device const int* pair_pos   [[buffer(1)]],
    device const float* topk_w   [[buffer(2)]],
    device float* moe_raw        [[buffer(3)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int d = int(tgp.y) * 256 + int(tid);
    if (d >= 2304) return;
    const int m = int(tgp.x);
    float acc = 0.0f;
    for (int k = 0; k < 8; ++k) {
        const int row = pair_pos[m * 8 + k];
        acc = fma(YE[(size_t)row * 2304 + d], topk_w[m * 8 + k], acc);
    }
    moe_raw[(size_t)m * 2304 + d] = acc;
}

// ============================================================================
// 3. Metal 4 matmul2d FlashAttention-2 Prefill (BQ=32, BK=64) with RelPos & Sinks
// ============================================================================

struct InkMppAttnArgs {
    int M;
    int M_pad;
    int slot;
    int pos0;
    int is_swa;
    int swa_win;
    int rel_extent;
    int kv_stride;
    float scale;
    int pad;
};

// Pre-transpose Q [M, 24, 128] F32 -> Q_h [24, M_pad, 128] half (pre-scaled by 1/sqrt(128))
// and R [M, 24, 16] F32 -> R_h [24, M_pad, 16] half
// grid (M_pad, 24, 1), 128 threads
kernel void ink_mpp_prep_qr(
    device const float* q_in [[buffer(0)]],
    device const float* r_in [[buffer(1)]],
    device half* q_h         [[buffer(2)]],
    device half* r_h         [[buffer(3)]],
    constant int& M          [[buffer(4)]],
    constant int& M_pad      [[buffer(5)]],
    constant float& scale    [[buffer(6)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x), h = int(tgp.y);
    const float qv = (m < M) ? (q_in[((size_t)m * 24 + h) * 128 + tid] * scale) : 0.0f;
    q_h[((size_t)h * M_pad + m) * 128 + tid] = half(qv);
    if (tid < 16u) {
        const float rv = (m < M) ? r_in[((size_t)m * 24 + h) * 16 + tid] : 0.0f;
        r_h[((size_t)h * M_pad + m) * 16 + tid] = half(rv);
    }
}

// Metal 4 FlashAttention-2 Prefill: BQ=32 queries x BK=64 keys, 4 SGs = 128 threads, 21.5 KB SLM!
// grid (M_pad / 32, 24, 1), 128 threads
kernel void ink_mpp_relpos_gqa_pf(
    device half* q_h                [[buffer(0)]],  // [24, M_pad, 128] pre-scaled
    device const half* r_h          [[buffer(1)]],  // [24, M_pad, 16]
    device const half* rel_proj_t   [[buffer(2)]],  // [1025, 16] transposed half (L1-hot!)
    device const float* sinks_buf   [[buffer(3)]],  // [M, 24]
    device const float* gate_buf    [[buffer(4)]],  // [M, 24, 128]
    device const half* k_cache      [[buffer(5)]],  // [max_slots, kv_stride, 8, 128]
    device const half* v_cache      [[buffer(6)]],  // [max_slots, kv_stride, 8, 128]
    device float* attn_out          [[buffer(7)]],  // [M, 24, 128]
    constant InkMppAttnArgs& A      [[buffer(8)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup half sKV[64 * 128];   // 16 KB: holds K[64,128], then sS[32,64] (8KB float), then V[64,128], then sO[32,128] (16KB float)
    threadgroup half sP[32 * 64];     // 4 KB
    threadgroup half sR[32 * 16];     // 1 KB
    threadgroup float sm[32], sl[32], sc[32];
    threadgroup float* sF = reinterpret_cast<threadgroup float*>(sKV);

    const int q0 = int(tgp.x) * 32;
    const int h = int(tgp.y);
    const int kh = h / 3;
    const int qpos0 = A.pos0 + q0;
    const int q_valid = min(32, A.M - q0);
    if (q_valid <= 0) return;

    if (tid < 32u) {
        sm[tid] = -INFINITY;
        sl[tid] = 0.0f;
        sc[tid] = 0.0f;
    }
    // Load R[32, 16] into sR (128 threads x 4 halfs = 512 halfs)
    {
        const int r = int(tid >> 2), c4 = int((tid & 3u) * 4u);
        *((threadgroup half4*)(sR + r * 16 + c4)) =
            *((device const half4*)(r_h + ((size_t)h * A.M_pad + q0 + r) * 16 + c4));
    }

    device half* qbase = q_h + ((size_t)h * A.M_pad + q0) * 128;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(qbase, dextents<int32_t, 2>(128, 32));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tKV(sKV, dextents<int32_t, 2>(128, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(sP, dextents<int32_t, 2>(64, 32));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(sF, dextents<int32_t, 2>(64, 32));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(sF, dextents<int32_t, 2>(128, 32));

    constexpr auto dQK = matmul2d_descriptor(32, 64, 128, false, true, false);
    constexpr auto dPV = matmul2d_descriptor(32, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dQK, execution_simdgroups<4>> opQK;
    matmul2d<dPV, execution_simdgroups<4>> opPV;

    auto accO = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(tKV), float>();
    for (uint16_t i = 0; i < accO.get_capacity(); ++i)
        if (accO.is_valid_element(i)) accO[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int k_min = A.is_swa ? max(0, qpos0 - A.swa_win + 1) : 0;
    const int k_max = qpos0 + q_valid;  // exclusive
    const size_t slot_kv_base = (size_t)A.slot * A.kv_stride;

    for (int k0 = k_min; k0 < k_max; k0 += 64) {
        // 1. Stage K[64, 128] into sKV (128 threads x 8 uint4 = 64*128 halfs)
        for (int p = 0; p < 8; ++p) {
            const uint idx = uint(p * 128) + tid;
            const uint kr = idx >> 4;
            const uint kc = (idx & 15u) << 3;
            const int kpos = k0 + int(kr);
            uint4 kv = uint4(0u);
            if (kpos < k_max) {
                const int kidx = A.is_swa ? (kpos & 1023) : kpos;
                const size_t off = ((slot_kv_base + kidx) * 8 + kh) * 128 + kc;
                kv = *((device const uint4*)(k_cache + off));
            }
            *((threadgroup uint4*)(sKV + kr * 128u + kc)) = kv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. S[32, 64] = Q[32, 128] @ K[64, 128]^T in cooperative tensor registers!
        auto accS = opQK.template get_destination_cooperative_tensor<decltype(tQ), decltype(tKV), float>();
        opQK.run(tQ, tKV, accS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        accS.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Add RelPos bias + Causal/SWA mask + Deferred-Sink Online Softmax -> sP[32, 64]
        // 128 threads = 32 query rows x 4 threads/row (each thread handles 16 keys!)
        {
            const int r = int(tid >> 2);
            const int lane4 = int(tid & 3u);
            const int qpos = qpos0 + r;
            const bool r_ok = (r < q_valid);
            const half4 r0 = *((threadgroup half4*)(sR + r * 16 + 0));
            const half4 r1 = *((threadgroup half4*)(sR + r * 16 + 4));
            const half4 r2 = *((threadgroup half4*)(sR + r * 16 + 8));
            const half4 r3 = *((threadgroup half4*)(sR + r * 16 + 12));
            const float4 rf0 = float4(r0), rf1 = float4(r1), rf2 = float4(r2), rf3 = float4(r3);

            float vals[16];
            float mx = -INFINITY;
            for (int e = 0; e < 16; ++e) {
                const int j = lane4 * 16 + e;
                const int kpos = k0 + j;
                float x = -INFINITY;
                if (r_ok && kpos <= qpos && (!A.is_swa || (qpos - kpos) < A.swa_win)) {
                    const int dist = min(qpos - kpos, A.rel_extent);
                    device const half4* rp = (device const half4*)(rel_proj_t + (size_t)dist * 16);
                    const float rb = dot(rf0, float4(rp[0])) + dot(rf1, float4(rp[1]))
                                   + dot(rf2, float4(rp[2])) + dot(rf3, float4(rp[3]));
                    x = sF[r * 64 + j] + rb;
                }
                vals[e] = x;
                mx = max(mx, x);
            }
            mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(2)));

            const float m_old = sm[r];
            const float mn = max(m_old, mx);
            const float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - mn);
            float ps = 0.0f;
            for (int e = 0; e < 16; ++e) {
                const int j = lane4 * 16 + e;
                const float p = (vals[e] > -1e29f) ? exp(vals[e] - mn) : 0.0f;
                ps += p;
                sP[r * 64 + j] = half(p);
            }
            ps += simd_shuffle_xor(ps, ushort(1));
            ps += simd_shuffle_xor(ps, ushort(2));
            if (lane4 == 0) {
                sm[r] = mn;
                sl[r] = sl[r] * corr + ps;
                sc[r] = corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Rescale accO if not first block
        if (k0 > k_min) {
            for (uint16_t i = 0; i < accO.get_capacity(); ++i) {
                if (accO.is_valid_element(i)) {
                    const int row = accO.get_multidimensional_index(i)[1];
                    accO[i] *= sc[row];
                }
            }
        }

        // 5. Stage V[64, 128] into sKV and accumulate O += P @ V
        for (int p = 0; p < 8; ++p) {
            const uint idx = uint(p * 128) + tid;
            const uint vr = idx >> 4;
            const uint vc = (idx & 15u) << 3;
            const int kpos = k0 + int(vr);
            uint4 vv = uint4(0u);
            if (kpos < k_max) {
                const int kidx = A.is_swa ? (kpos & 1023) : kpos;
                const size_t off = ((slot_kv_base + kidx) * 8 + kh) * 128 + vc;
                vv = *((device const uint4*)(v_cache + off));
            }
            *((threadgroup uint4*)(sKV + vr * 128u + vc)) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        opPV.run(tP, tKV, accO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 6. Final deferred-sink normalization + fused sigmoid(gate) -> attn_out[q0+r, h, 0..127]
    if (int(tid) < q_valid) {
        const float sk = sinks_buf[(size_t)(q0 + int(tid)) * 24 + h];
        const float m_k = sm[tid];
        if (m_k == -INFINITY) {
            sc[tid] = 0.0f;
        } else {
            const float m_fin = max(m_k, sk);
            const float so = exp(m_k - m_fin);
            sc[tid] = so / (sl[tid] * so + exp(sk - m_fin));
        }
    }
    accO.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = 0; r < q_valid; ++r) {
        const float inv = sc[r];
        const size_t goff = ((size_t)(q0 + r) * 24 + h) * 128 + tid;
        const float g = gate_buf[goff];
        const float g_sig = 1.0f / (1.0f + exp(-g));
        attn_out[goff] = sF[r * 128 + tid] * inv * g_sig;
    }
}
