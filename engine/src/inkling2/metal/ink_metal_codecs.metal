// ink_metal_codecs.metal - High-speed Apple Silicon Metal kernels for Inkling-2-Mini (.strata v1.1)
// Optimizations:
//   - Zero-stack-spill register-only S_K decoders (float4 pairs, 2-row/SG linear, 8-tok MoE micro-batching)
//   - Vectorized 64-bit uint2 SF4 nibble loads & 32-bit uint MX8 loads
//   - simdgroup_matrix 8x8 FP16/BF16 GEMM for M > 8 (prefill & c=32/64 batching)
//   - GPU-side MoE binning (ink_moe_plan -> indirect dispatch grid, zero host sync)
//   - Fused w13 + SITU(beta=4.0, lb=25.0) + w2 scatter-add across Top-8/128 experts
// 100% MIT clean-room implementation.
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 1. Lossless NVFP4 (E2M1) & FP8 (E4M3FN) & UE8M0 Register-LUT Decoders
// ============================================================================

constant float kE2M1[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float e4m3_to_f32(uint c) {
    const uint mag = c & 0x7Fu;
    if (mag == 0x7Fu) return 0.0f;
    const uint exp = mag >> 3;
    const uint man = mag & 7u;
    float v = (exp == 0u) ? (float(man) * (1.0f / 512.0f))
                          : as_type<float>(((exp + 120u) << 23) | (man << 20));
    return (c & 0x80u) ? -v : v;
}

static inline float ue8m0_to_f32(uint e) {
    if (e == 0u) return 5.877471754111438e-39f;
    if (e == 255u) return 0.0f;
    return as_type<float>(e << 23);
}

static inline float bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16); }

static inline float4 bf16x4_to_f32(ushort4 b) {
    return float4(as_type<float>(uint(b.x) << 16), as_type<float>(uint(b.y) << 16),
                  as_type<float>(uint(b.z) << 16), as_type<float>(uint(b.w) << 16));
}

// Exact SITU(g, u, beta=4.0, lb=25.0)
static inline float situ_act(float g, float u) {
    const float sig = 1.0f / (1.0f + exp(-g));
    const float g_act = 4.0f * tanh(clamp(g * 0.25f, -15.0f, 15.0f)) * sig;
    const float u_act = 25.0f * tanh(clamp(u * 0.04f, -15.0f, 15.0f));
    return g_act * u_act;
}

// ============================================================================
// 2. Register-Only S_K Super-Block Decoders (ZERO thread-local array spills!)
// ============================================================================

static inline constexpr int sk_sb_bytes(int F) {
    return F == 2 ? 70 : F == 3 ? 102 : F == 4 ? 134 : F == 5 ? 166 : F == 6 ? 198 : F == 8 ? 262 : 0;
}

static inline constexpr float sk_half(int F) {
    return F == 2 ? 1.5f : F == 3 ? 3.5f : F == 4 ? 7.5f : F == 5 ? 15.5f : F == 6 ? 31.5f : 0.0f;
}

static inline ushort ld16(device const uchar* p) {
    return ushort(p[0]) | (ushort(p[1]) << 8);
}

constant half4 kS2Lut[256] = {
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
constant half4 kS3Hi[16] = {
    half4(-2.0h,-2.0h,-2.0h,-2.0h), half4(2.0h,-2.0h,-2.0h,-2.0h), half4(-2.0h,2.0h,-2.0h,-2.0h), half4(2.0h,2.0h,-2.0h,-2.0h),
    half4(-2.0h,-2.0h,2.0h,-2.0h), half4(2.0h,-2.0h,2.0h,-2.0h), half4(-2.0h,2.0h,2.0h,-2.0h), half4(2.0h,2.0h,2.0h,-2.0h),
    half4(-2.0h,-2.0h,-2.0h,2.0h), half4(2.0h,-2.0h,-2.0h,2.0h), half4(-2.0h,2.0h,-2.0h,2.0h), half4(2.0h,2.0h,-2.0h,2.0h),
    half4(-2.0h,-2.0h,2.0h,2.0h), half4(2.0h,-2.0h,2.0h,2.0h), half4(-2.0h,2.0h,2.0h,2.0h), half4(2.0h,2.0h,2.0h,2.0h),
};
constant half2 kS4Lut[256] = {
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
constant half4 kS5Hi[16] = {
    half4(-8.0h,-8.0h,-8.0h,-8.0h), half4(8.0h,-8.0h,-8.0h,-8.0h), half4(-8.0h,8.0h,-8.0h,-8.0h), half4(8.0h,8.0h,-8.0h,-8.0h),
    half4(-8.0h,-8.0h,8.0h,-8.0h), half4(8.0h,-8.0h,8.0h,-8.0h), half4(-8.0h,8.0h,8.0h,-8.0h), half4(8.0h,8.0h,8.0h,-8.0h),
    half4(-8.0h,-8.0h,-8.0h,8.0h), half4(8.0h,-8.0h,-8.0h,8.0h), half4(-8.0h,8.0h,-8.0h,8.0h), half4(8.0h,8.0h,-8.0h,8.0h),
    half4(-8.0h,-8.0h,8.0h,8.0h), half4(8.0h,-8.0h,8.0h,8.0h), half4(-8.0h,8.0h,8.0h,8.0h), half4(8.0h,8.0h,8.0h,8.0h),
};

// Constant-cache LUT S_K decoder (3.25 KB total -> 10-15x fewer ALU ops on S2_K..S6_K!)
template <int F>
static inline void sk_w8(device const uchar* pl, int j, thread float4& w0, thread float4& w1) {
    if (F == 2) {
        device const uchar* p = pl + (j >> 2);
        w0 = float4(kS2Lut[p[0]]);
        w1 = float4(kS2Lut[p[1]]);
    } else if (F == 3) {
        device const uchar* p = pl + (j >> 2);
        const uint h = uint(pl[64 + (j >> 3)]);
        w0 = float4(kS2Lut[p[0]] + kS3Hi[h & 15u]);
        w1 = float4(kS2Lut[p[1]] + kS3Hi[h >> 4]);
    } else if (F == 4) {
        device const uchar* p = pl + (j >> 1);
        w0 = float4(half4(kS4Lut[p[0]], kS4Lut[p[1]]));
        w1 = float4(half4(kS4Lut[p[2]], kS4Lut[p[3]]));
    } else if (F == 5) {
        device const uchar* p = pl + (j >> 1);
        const uint h = uint(pl[128 + (j >> 3)]);
        w0 = float4(half4(kS4Lut[p[0]], kS4Lut[p[1]]) + kS5Hi[h & 15u]);
        w1 = float4(half4(kS4Lut[p[2]], kS4Lut[p[3]]) + kS5Hi[h >> 4]);
    } else if (F == 6) {
        device const uchar* p = pl + (j >> 1);
        device const uchar* hp = pl + 128 + (j >> 2);
        w0 = float4(fma(kS2Lut[hp[0]], half4(16.0h), half4(kS4Lut[p[0]], kS4Lut[p[1]])));
        w1 = float4(fma(kS2Lut[hp[1]], half4(16.0h), half4(kS4Lut[p[2]], kS4Lut[p[3]])));
    } else {
        device const char* qp = (device const char*)(pl + j);
        w0 = float4(float(qp[0]), float(qp[1]), float(qp[2]), float(qp[3]));
        w1 = float4(float(qp[4]), float(qp[5]), float(qp[6]), float(qp[7]));
    }
}

// ============================================================================
// 3. Dense Trunk & DSpark Linear Kernels (BF16, Q8_ROW, F32, S_K)
// ============================================================================

// --- BF16 GEMV for M==1 (32 rows/TG = 4 rows/SG, vectorized 16-byte ushort4x2 loads for peak M5 Max BW!)
kernel void ink_linear_bf16_mv1(
    device const ushort* W [[buffer(0)]],
    device const float* X  [[buffer(1)]],
    device float* Y        [[buffer(2)]],
    constant int& N        [[buffer(3)]],
    constant int& K        [[buffer(4)]],
    constant int& M        [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    (void)M;
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 32 + int(sg) * 4;
    if (row0 >= N) return;
    const bool h1 = (row0 + 1) < N, h2 = (row0 + 2) < N, h3 = (row0 + 3) < N;
    const int K8 = K >> 3;
    device const ushort4* w0p = (device const ushort4*)(W + (size_t)row0 * K);
    device const ushort4* w1p = (device const ushort4*)(W + (size_t)(row0 + (h1 ? 1 : 0)) * K);
    device const ushort4* w2p = (device const ushort4*)(W + (size_t)(row0 + (h2 ? 2 : 0)) * K);
    device const ushort4* w3p = (device const ushort4*)(W + (size_t)(row0 + (h3 ? 3 : 0)) * K);
    device const float4* xp = (device const float4*)X;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int k = int(lane); k < K8; k += 32) {
        const int k2 = k * 2;
        const float4 x0 = xp[k2], x1 = xp[k2 + 1];
        a0 += dot(bf16x4_to_f32(w0p[k2]), x0) + dot(bf16x4_to_f32(w0p[k2 + 1]), x1);
        if (h1) a1 += dot(bf16x4_to_f32(w1p[k2]), x0) + dot(bf16x4_to_f32(w1p[k2 + 1]), x1);
        if (h2) a2 += dot(bf16x4_to_f32(w2p[k2]), x0) + dot(bf16x4_to_f32(w2p[k2 + 1]), x1);
        if (h3) a3 += dot(bf16x4_to_f32(w3p[k2]), x0) + dot(bf16x4_to_f32(w3p[k2 + 1]), x1);
    }
    if ((K & 4) != 0 && lane == 0) {
        const int k2 = K8 * 2;
        const float4 x0 = xp[k2];
        a0 += dot(bf16x4_to_f32(w0p[k2]), x0);
        if (h1) a1 += dot(bf16x4_to_f32(w1p[k2]), x0);
        if (h2) a2 += dot(bf16x4_to_f32(w2p[k2]), x0);
        if (h3) a3 += dot(bf16x4_to_f32(w3p[k2]), x0);
    }
    const float v0 = simd_sum(a0); if (lane == 0) Y[row0] = v0;
    if (h1) { const float v1 = simd_sum(a1); if (lane == 0) Y[row0 + 1] = v1; }
    if (h2) { const float v2 = simd_sum(a2); if (lane == 0) Y[row0 + 2] = v2; }
    if (h3) { const float v3 = simd_sum(a3); if (lane == 0) Y[row0 + 3] = v3; }
}

// --- BF16 simdgroup_matrix 8x64 GEMM for M=2..8 (ZERO M-waste for DSpark M=8 verify & draft!)
kernel void ink_linear_bf16_mm8(
    device const ushort* W [[buffer(0)]],
    device const float* X  [[buffer(1)]],
    device float* Y        [[buffer(2)]],
    constant int& N        [[buffer(3)]],
    constant int& K        [[buffer(4)]],
    constant int& M        [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[8 * 32];
    threadgroup half sB[64 * 32];
    const int n0 = int(tg) * 64;
    if (n0 >= N) return;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < K; k0 += 32) {
        const int ar = int(tid >> 5), ac = int(tid & 31u);
        sA[tid] = (ar < M && k0 + ac < K) ? half(X[(size_t)ar * K + k0 + ac]) : half(0.0h);
        for (uint idx = tid * 4; idx < 64 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gn = n0 + r;
            if (gn < N && k0 + c + 3 < K)
                *((threadgroup half4*)(sB + r * 32 + c)) = half4(bf16x4_to_f32(*((device const ushort4*)(W + (size_t)gn * K + k0 + c))));
            else
                for (int u = 0; u < 4; ++u) sB[r * 32 + c + u] = (gn < N && k0 + c + u < K) ? half(bf16_to_f32(W[(size_t)gn * K + k0 + c + u])) : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 a0, b0;
            simdgroup_load(a0, sA + kk * 8, 32);
            simdgroup_load(b0, sB + (int(sg) * 8) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a0, b0, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint qid = lane >> 2;
    const int gm = int((qid & 4u) + ((lane >> 1) & 3u));
    const int gn = n0 + int(sg) * 8 + int((qid & 2u) * 2u + (lane & 1u) * 2u);
    if (gm < M) {
        if (gn < N) Y[(size_t)gm * N + gn] = acc.thread_elements()[0];
        if (gn + 1 < N) Y[(size_t)gm * N + gn + 1] = acc.thread_elements()[1];
    }
}

// --- BF16 GEMV for M=1..8 (single DRAM weight pass, ZERO barriers, exact FP32 activations!)
kernel void ink_linear_bf16_mv(
    device const ushort* W [[buffer(0)]],
    device const float* X  [[buffer(1)]],
    device float* Y        [[buffer(2)]],
    constant int& N        [[buffer(3)]],
    constant int& K        [[buffer(4)]],
    constant int& M        [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    if (row0 >= N) return;
    const bool has1 = (row0 + 1) < N;
    const int K4 = K >> 2;
    device const ushort4* w0p = (device const ushort4*)(W + (size_t)row0 * K);
    device const ushort4* w1p = (device const ushort4*)(W + (size_t)(row0 + (has1 ? 1 : 0)) * K);

    if (M == 1) {
        float a0 = 0.0f, a1 = 0.0f;
        device const float4* xp = (device const float4*)X;
        for (int k = int(lane); k < K4; k += 32) {
            const float4 xv = xp[k];
            a0 += dot(bf16x4_to_f32(w0p[k]), xv);
            if (has1) a1 += dot(bf16x4_to_f32(w1p[k]), xv);
        }
        const float v0 = simd_sum(a0);
        if (lane == 0) Y[row0] = v0;
        if (has1) {
            const float v1 = simd_sum(a1);
            if (lane == 0) Y[row0 + 1] = v1;
        }
        return;
    }

    if (M <= 8) {
        float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0};
        for (int k = int(lane); k < K4; k += 32) {
            const float4 w0 = bf16x4_to_f32(w0p[k]);
            const float4 w1 = has1 ? bf16x4_to_f32(w1p[k]) : float4(0.f);
            for (int i = 0; i < M; ++i) {
                const float4 xv = ((device const float4*)(X + (size_t)i * K))[k];
                a0[i] += dot(w0, xv);
                if (has1) a1[i] += dot(w1, xv);
            }
        }
        for (int i = 0; i < M; ++i) {
            const float v0 = simd_sum(a0[i]);
            if (lane == 0) Y[(size_t)i * N + row0] = v0;
            if (has1) {
                const float v1 = simd_sum(a1[i]);
                if (lane == 0) Y[(size_t)i * N + row0 + 1] = v1;
            }
        }
        return;
    }

    for (int m0 = 0; m0 < M; m0 += 4) {
        const int mm = min(4, M - m0);
        float a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
        for (int k = int(lane); k < K4; k += 32) {
            const float4 w0 = bf16x4_to_f32(w0p[k]);
            const float4 w1 = has1 ? bf16x4_to_f32(w1p[k]) : float4(0.f);
            for (int i = 0; i < mm; ++i) {
                const float4 xv = ((device const float4*)(X + (size_t)(m0 + i) * K))[k];
                a0[i] += dot(w0, xv);
                if (has1) a1[i] += dot(w1, xv);
            }
        }
        for (int i = 0; i < mm; ++i) {
            const float v0 = simd_sum(a0[i]);
            if (lane == 0) Y[(size_t)(m0 + i) * N + row0] = v0;
            if (has1) {
                const float v1 = simd_sum(a1[i]);
                if (lane == 0) Y[(size_t)(m0 + i) * N + row0 + 1] = v1;
            }
        }
    }
}

// --- BF16 simdgroup_matrix 8x8 GEMM for M > 8 (BM=64, BN=64, BK=32, 8 SGs = 256 threads -> 2x W reuse!)
kernel void ink_linear_bf16_mm(
    device const ushort* W [[buffer(0)]],
    device const float* X  [[buffer(1)]],
    device float* Y        [[buffer(2)]],
    constant int& N        [[buffer(3)]],
    constant int& K        [[buffer(4)]],
    constant int& M        [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[64 * 32];
    threadgroup half sB[64 * 32];
    const int m0 = int(tgp.y) * 64;
    const int n0 = int(tgp.x) * 64;
    if (m0 >= M || n0 >= N) return;

    const int sgm = int(sg >> 2), sgn = int(sg & 3u);
    simdgroup_float8x8 acc[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += 32) {
        for (uint idx = tid * 4; idx < 64 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gm = m0 + r;
            if (gm < M && k0 + c + 3 < K) {
                *((threadgroup half4*)(sA + r * 32 + c)) = half4(*((device const float4*)(X + (size_t)gm * K + k0 + c)));
            } else {
                for (int u = 0; u < 4; ++u)
                    sA[r * 32 + c + u] = (gm < M && k0 + c + u < K) ? half(X[(size_t)gm * K + k0 + c + u]) : half(0.0h);
            }
        }
        for (uint idx = tid * 4; idx < 64 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gn = n0 + r;
            if (gn < N && k0 + c + 3 < K) {
                *((threadgroup half4*)(sB + r * 32 + c)) = half4(bf16x4_to_f32(*((device const ushort4*)(W + (size_t)gn * K + k0 + c))));
            } else {
                for (int u = 0; u < 4; ++u)
                    sB[r * 32 + c + u] = (gn < N && k0 + c + u < K) ? half(bf16_to_f32(W[(size_t)gn * K + k0 + c + u])) : half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 a0, a1, a2, a3, b0, b1;
            simdgroup_load(a0, sA + (sgm * 32 + 0) * 32 + kk * 8, 32);
            simdgroup_load(a1, sA + (sgm * 32 + 8) * 32 + kk * 8, 32);
            simdgroup_load(a2, sA + (sgm * 32 + 16) * 32 + kk * 8, 32);
            simdgroup_load(a3, sA + (sgm * 32 + 24) * 32 + kk * 8, 32);
            simdgroup_load(b0, sB + (sgn * 16 + 0) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_load(b1, sB + (sgn * 16 + 8) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[2][0], a2, b0, acc[2][0]);
            simdgroup_multiply_accumulate(acc[2][1], a2, b1, acc[2][1]);
            simdgroup_multiply_accumulate(acc[3][0], a3, b0, acc[3][0]);
            simdgroup_multiply_accumulate(acc[3][1], a3, b1, acc[3][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    for (int i = 0; i < 4; ++i) {
        const int gm = m0 + sgm * 32 + i * 8 + int(fm);
        if (gm < M) {
            for (int j = 0; j < 2; ++j) {
                const int gn = n0 + sgn * 16 + j * 8 + int(fn);
                if (gn < N) Y[(size_t)gm * N + gn] = acc[i][j].thread_elements()[0];
                if (gn + 1 < N) Y[(size_t)gm * N + gn + 1] = acc[i][j].thread_elements()[1];
            }
        }
    }
}

// --- Q8_ROW GEMV for M==1 (32 rows/TG = 4 rows/SG, vectorized 16-byte char4x4 loads!)
kernel void ink_linear_q8row_mv1(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    (void)M;
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 32 + int(sg) * 4;
    if (row0 >= N) return;
    const bool h1 = (row0 + 1) < N, h2 = (row0 + 2) < N, h3 = (row0 + 3) < N;
    device const float* scales = (device const float*)W;
    const float sc0 = scales[row0], sc1 = h1 ? scales[row0 + 1] : 0.0f;
    const float sc2 = h2 ? scales[row0 + 2] : 0.0f, sc3 = h3 ? scales[row0 + 3] : 0.0f;
    device const uchar* qbase = W + (size_t)N * 4;
    device const char4* q0p = (device const char4*)(qbase + (size_t)row0 * K);
    device const char4* q1p = (device const char4*)(qbase + (size_t)(row0 + (h1 ? 1 : 0)) * K);
    device const char4* q2p = (device const char4*)(qbase + (size_t)(row0 + (h2 ? 2 : 0)) * K);
    device const char4* q3p = (device const char4*)(qbase + (size_t)(row0 + (h3 ? 3 : 0)) * K);
    device const float4* xp = (device const float4*)X;
    const int K16 = K >> 4;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int k = int(lane); k < K16; k += 32) {
        const int k4 = k * 4;
        const float4 x0 = xp[k4], x1 = xp[k4 + 1], x2 = xp[k4 + 2], x3 = xp[k4 + 3];
        a0 += dot(float4(q0p[k4]), x0) + dot(float4(q0p[k4 + 1]), x1) + dot(float4(q0p[k4 + 2]), x2) + dot(float4(q0p[k4 + 3]), x3);
        if (h1) a1 += dot(float4(q1p[k4]), x0) + dot(float4(q1p[k4 + 1]), x1) + dot(float4(q1p[k4 + 2]), x2) + dot(float4(q1p[k4 + 3]), x3);
        if (h2) a2 += dot(float4(q2p[k4]), x0) + dot(float4(q2p[k4 + 1]), x1) + dot(float4(q2p[k4 + 2]), x2) + dot(float4(q2p[k4 + 3]), x3);
        if (h3) a3 += dot(float4(q3p[k4]), x0) + dot(float4(q3p[k4 + 1]), x1) + dot(float4(q3p[k4 + 2]), x2) + dot(float4(q3p[k4 + 3]), x3);
    }
    const float v0 = simd_sum(a0) * sc0; if (lane == 0) Y[row0] = v0;
    if (h1) { const float v1 = simd_sum(a1) * sc1; if (lane == 0) Y[row0 + 1] = v1; }
    if (h2) { const float v2 = simd_sum(a2) * sc2; if (lane == 0) Y[row0 + 2] = v2; }
    if (h3) { const float v3 = simd_sum(a3) * sc3; if (lane == 0) Y[row0 + 3] = v3; }
}

// --- Q8_ROW simdgroup_matrix 8x64 GEMM for M=2..8
kernel void ink_linear_q8row_mm8(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[8 * 32];
    threadgroup half sB[64 * 32];
    const int n0 = int(tg) * 64;
    if (n0 >= N) return;
    device const float* scales = (device const float*)W;
    device const char* qw = (device const char*)(W + (size_t)N * 4);
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < K; k0 += 32) {
        const int ar = int(tid >> 5), ac = int(tid & 31u);
        sA[tid] = (ar < M && k0 + ac < K) ? half(X[(size_t)ar * K + k0 + ac]) : half(0.0h);
        for (uint idx = tid * 4; idx < 64 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gn = n0 + r;
            if (gn < N && k0 + c + 3 < K)
                *((threadgroup half4*)(sB + r * 32 + c)) = half4(float4(*((device const char4*)(qw + (size_t)gn * K + k0 + c))));
            else
                for (int u = 0; u < 4; ++u) sB[r * 32 + c + u] = (gn < N && k0 + c + u < K) ? half(float(qw[(size_t)gn * K + k0 + c + u])) : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 a0, b0;
            simdgroup_load(a0, sA + kk * 8, 32);
            simdgroup_load(b0, sB + (int(sg) * 8) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a0, b0, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint qid = lane >> 2;
    const int gm = int((qid & 4u) + ((lane >> 1) & 3u));
    const int gn = n0 + int(sg) * 8 + int((qid & 2u) * 2u + (lane & 1u) * 2u);
    if (gm < M) {
        if (gn < N) Y[(size_t)gm * N + gn] = acc.thread_elements()[0] * scales[gn];
        if (gn + 1 < N) Y[(size_t)gm * N + gn + 1] = acc.thread_elements()[1] * scales[gn + 1];
    }
}

// --- Fused Unembed + Per-TG Argmax (ZERO 800 KB bd_logits DRAM write/read for DSpark Markov steps!)
kernel void ink_unembed_argmax_q8row(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* part_v  [[buffer(2)]],
    device int* part_i    [[buffer(3)]],
    constant int& N       [[buffer(4)]],
    constant int& K       [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[8];
    threadgroup int sh_i[8];
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 32 + int(sg) * 4;
    float bv = -1e30f; int bi = 0;
    if (row0 < N) {
        const bool h1 = (row0 + 1) < N, h2 = (row0 + 2) < N, h3 = (row0 + 3) < N;
        device const float* scales = (device const float*)W;
        device const uchar* qbase = W + (size_t)N * 4;
        device const char4* q0p = (device const char4*)(qbase + (size_t)row0 * K);
        device const char4* q1p = (device const char4*)(qbase + (size_t)(row0 + (h1 ? 1 : 0)) * K);
        device const char4* q2p = (device const char4*)(qbase + (size_t)(row0 + (h2 ? 2 : 0)) * K);
        device const char4* q3p = (device const char4*)(qbase + (size_t)(row0 + (h3 ? 3 : 0)) * K);
        device const float4* xp = (device const float4*)X;
        const int K16 = K >> 4;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int k = int(lane); k < K16; k += 32) {
            const int k4 = k * 4;
            const float4 x0 = xp[k4], x1 = xp[k4 + 1], x2 = xp[k4 + 2], x3 = xp[k4 + 3];
            a0 += dot(float4(q0p[k4]), x0) + dot(float4(q0p[k4 + 1]), x1) + dot(float4(q0p[k4 + 2]), x2) + dot(float4(q0p[k4 + 3]), x3);
            if (h1) a1 += dot(float4(q1p[k4]), x0) + dot(float4(q1p[k4 + 1]), x1) + dot(float4(q1p[k4 + 2]), x2) + dot(float4(q1p[k4 + 3]), x3);
            if (h2) a2 += dot(float4(q2p[k4]), x0) + dot(float4(q2p[k4 + 1]), x1) + dot(float4(q2p[k4 + 2]), x2) + dot(float4(q2p[k4 + 3]), x3);
            if (h3) a3 += dot(float4(q3p[k4]), x0) + dot(float4(q3p[k4 + 1]), x1) + dot(float4(q3p[k4 + 2]), x2) + dot(float4(q3p[k4 + 3]), x3);
        }
        const float v0 = simd_sum(a0) * scales[row0];
        bv = v0; bi = row0;
        if (h1) { const float v1 = simd_sum(a1) * scales[row0 + 1]; if (v1 > bv) { bv = v1; bi = row0 + 1; } }
        if (h2) { const float v2 = simd_sum(a2) * scales[row0 + 2]; if (v2 > bv) { bv = v2; bi = row0 + 2; } }
        if (h3) { const float v3 = simd_sum(a3) * scales[row0 + 3]; if (v3 > bv) { bv = v3; bi = row0 + 3; } }
    }
    if (lane == 0) { sh_v[sg] = bv; sh_i[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tv = sh_v[0]; int ti = sh_i[0];
        for (int s = 1; s < 8; ++s) {
            if (sh_v[s] > tv || (sh_v[s] == tv && sh_i[s] < ti)) { tv = sh_v[s]; ti = sh_i[s]; }
        }
        part_v[tg] = tv; part_i[tg] = ti;
    }
}

kernel void ink_unembed_argmax_bf16(
    device const ushort* W [[buffer(0)]],
    device const float* X  [[buffer(1)]],
    device float* part_v   [[buffer(2)]],
    device int* part_i     [[buffer(3)]],
    constant int& N        [[buffer(4)]],
    constant int& K        [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[8];
    threadgroup int sh_i[8];
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 32 + int(sg) * 4;
    float bv = -1e30f; int bi = 0;
    if (row0 < N) {
        const bool h1 = (row0 + 1) < N, h2 = (row0 + 2) < N, h3 = (row0 + 3) < N;
        device const ushort4* w0p = (device const ushort4*)(W + (size_t)row0 * K);
        device const ushort4* w1p = (device const ushort4*)(W + (size_t)(row0 + (h1 ? 1 : 0)) * K);
        device const ushort4* w2p = (device const ushort4*)(W + (size_t)(row0 + (h2 ? 2 : 0)) * K);
        device const ushort4* w3p = (device const ushort4*)(W + (size_t)(row0 + (h3 ? 3 : 0)) * K);
        device const float4* xp = (device const float4*)X;
        const int K8 = K >> 3;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int k = int(lane); k < K8; k += 32) {
            const int k2 = k * 2;
            const float4 x0 = xp[k2], x1 = xp[k2 + 1];
            a0 += dot(bf16x4_to_f32(w0p[k2]), x0) + dot(bf16x4_to_f32(w0p[k2 + 1]), x1);
            if (h1) a1 += dot(bf16x4_to_f32(w1p[k2]), x0) + dot(bf16x4_to_f32(w1p[k2 + 1]), x1);
            if (h2) a2 += dot(bf16x4_to_f32(w2p[k2]), x0) + dot(bf16x4_to_f32(w2p[k2 + 1]), x1);
            if (h3) a3 += dot(bf16x4_to_f32(w3p[k2]), x0) + dot(bf16x4_to_f32(w3p[k2 + 1]), x1);
        }
        bv = simd_sum(a0); bi = row0;
        if (h1) { const float v1 = simd_sum(a1); if (v1 > bv) { bv = v1; bi = row0 + 1; } }
        if (h2) { const float v2 = simd_sum(a2); if (v2 > bv) { bv = v2; bi = row0 + 2; } }
        if (h3) { const float v3 = simd_sum(a3); if (v3 > bv) { bv = v3; bi = row0 + 3; } }
    }
    if (lane == 0) { sh_v[sg] = bv; sh_i[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tv = sh_v[0]; int ti = sh_i[0];
        for (int s = 1; s < 8; ++s) {
            if (sh_v[s] > tv || (sh_v[s] == tv && sh_i[s] < ti)) { tv = sh_v[s]; ti = sh_i[s]; }
        }
        part_v[tg] = tv; part_i[tg] = ti;
    }
}

kernel void ink_unembed_argmax_sk6(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* part_v  [[buffer(2)]],
    device int* part_i    [[buffer(3)]],
    constant int& N       [[buffer(4)]],
    constant int& K       [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[8];
    threadgroup int sh_i[8];
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    float bv = -1e30f; int bi = 0;
    if (row0 < N) {
        const bool h1 = (row0 + 1) < N;
        constexpr int SB = 198;
        const int nsb = K >> 8;
        const size_t rowb = (size_t)nsb * SB;
        device const uchar* r0 = W + (size_t)row0 * rowb;
        device const uchar* r1 = h1 ? (r0 + rowb) : r0;
        float a0 = 0.0f, a1 = 0.0f;
        for (int unit = int(lane); unit < nsb * 32; unit += 32) {
            const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
            device const float4* xp = (device const float4*)(X + sbi * 256 + j);
            const float4 x0 = xp[0], x1 = xp[1];
            {
                device const uchar* blk = r0 + (size_t)sbi * SB;
                const float eff = float(as_type<half>(ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
                float4 w0, w1; sk_w8<6>(blk + 6, j, w0, w1);
                a0 = fma(eff, dot(w0, x0) + dot(w1, x1), a0);
            }
            if (h1) {
                device const uchar* blk = r1 + (size_t)sbi * SB;
                const float eff = float(as_type<half>(ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
                float4 w0, w1; sk_w8<6>(blk + 6, j, w0, w1);
                a1 = fma(eff, dot(w0, x0) + dot(w1, x1), a1);
            }
        }
        bv = simd_sum(a0); bi = row0;
        if (h1) { const float v1 = simd_sum(a1); if (v1 > bv) { bv = v1; bi = row0 + 1; } }
    }
    if (lane == 0) { sh_v[sg] = bv; sh_i[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tv = sh_v[0]; int ti = sh_i[0];
        for (int s = 1; s < 8; ++s) {
            if (sh_v[s] > tv || (sh_v[s] == tv && sh_i[s] < ti)) { tv = sh_v[s]; ti = sh_i[s]; }
        }
        part_v[tg] = tv; part_i[tg] = ti;
    }
}

kernel void ink_argmax_part_reduce(
    device const float* part_v [[buffer(0)]],
    device const int* part_i   [[buffer(1)]],
    device int* out_id         [[buffer(2)]],
    constant int& n_parts      [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[256];
    threadgroup int sh_i[256];
    float bv = -1e30f; int bi = 0;
    for (int i = int(tid); i < n_parts; i += 256) {
        const float v = part_v[i]; const int idx = part_i[i];
        if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
    }
    sh_v[tid] = bv; sh_i[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (sh_v[tid + s] > sh_v[tid] || (sh_v[tid + s] == sh_v[tid] && sh_i[tid + s] < sh_i[tid])) {
                sh_v[tid] = sh_v[tid + s]; sh_i[tid] = sh_i[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out_id[0] = sh_i[0];
}

// --- Multi-row Fused Unembed + Per-TG Argmax for M = 2..16 (ZERO logits DRAM write/read!)
// grid (nparts, 1, 1), 256 threads -> writes part_v[m * nparts + tg], part_i[m * nparts + tg]
kernel void ink_unembed_argmax_m_bf16(
    device const ushort* W [[buffer(0)]],
    device const float* X  [[buffer(1)]],
    device float* part_v   [[buffer(2)]],
    device int* part_i     [[buffer(3)]],
    constant int& N        [[buffer(4)]],
    constant int& K        [[buffer(5)]],
    constant int& M        [[buffer(6)]],
    constant int& nparts   [[buffer(7)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[8 * 8];
    threadgroup int sh_i[8 * 8];
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    const bool h0 = row0 < N, h1 = (row0 + 1) < N;
    const int K4 = K >> 2;
    device const ushort4* w0p = (device const ushort4*)(W + (size_t)(h0 ? row0 : 0) * K);
    device const ushort4* w1p = (device const ushort4*)(W + (size_t)(h1 ? (row0 + 1) : 0) * K);

    for (int m0 = 0; m0 < M; m0 += 8) {
        const int mm = min(8, M - m0);
        float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0};
        if (h0) {
            for (int k = int(lane); k < K4; k += 32) {
                const float4 w0 = bf16x4_to_f32(w0p[k]);
                const float4 w1 = h1 ? bf16x4_to_f32(w1p[k]) : float4(0.0f);
                for (int i = 0; i < mm; ++i) {
                    const float4 xv = ((device const float4*)(X + (size_t)(m0 + i) * K))[k];
                    a0[i] += dot(w0, xv);
                    if (h1) a1[i] += dot(w1, xv);
                }
            }
        }
        for (int i = 0; i < mm; ++i) {
            float bv = h0 ? simd_sum(a0[i]) : -1e30f;
            int bi = row0;
            if (h1) {
                const float v1 = simd_sum(a1[i]);
                if (v1 > bv) { bv = v1; bi = row0 + 1; }
            }
            if (lane == 0) { sh_v[i * 8 + sg] = bv; sh_i[i * 8 + sg] = bi; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(mm)) {
            const int i = int(tid);
            float tv = sh_v[i * 8 + 0]; int ti = sh_i[i * 8 + 0];
            for (int s = 1; s < 8; ++s) {
                const float sv = sh_v[i * 8 + s]; const int si = sh_i[i * 8 + s];
                if (sv > tv || (sv == tv && si < ti)) { tv = sv; ti = si; }
            }
            part_v[(size_t)(m0 + i) * nparts + tg] = tv;
            part_i[(size_t)(m0 + i) * nparts + tg] = ti;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void ink_unembed_argmax_m_q8row(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* part_v  [[buffer(2)]],
    device int* part_i    [[buffer(3)]],
    constant int& N       [[buffer(4)]],
    constant int& K       [[buffer(5)]],
    constant int& M       [[buffer(6)]],
    constant int& nparts  [[buffer(7)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[8 * 8];
    threadgroup int sh_i[8 * 8];
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    const bool h0 = row0 < N, h1 = (row0 + 1) < N;
    device const float* scales = (device const float*)W;
    const float sc0 = h0 ? scales[row0] : 0.0f;
    const float sc1 = h1 ? scales[row0 + 1] : 0.0f;
    device const uchar* qbase = W + (size_t)N * 4;
    device const char4* q0p = (device const char4*)(qbase + (size_t)(h0 ? row0 : 0) * K);
    device const char4* q1p = (device const char4*)(qbase + (size_t)(h1 ? (row0 + 1) : 0) * K);
    const int K4 = K >> 2;

    for (int m0 = 0; m0 < M; m0 += 8) {
        const int mm = min(8, M - m0);
        float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0};
        if (h0) {
            for (int k = int(lane); k < K4; k += 32) {
                const float4 w0 = float4(q0p[k]);
                const float4 w1 = h1 ? float4(q1p[k]) : float4(0.0f);
                for (int i = 0; i < mm; ++i) {
                    const float4 xv = ((device const float4*)(X + (size_t)(m0 + i) * K))[k];
                    a0[i] += dot(w0, xv);
                    if (h1) a1[i] += dot(w1, xv);
                }
            }
        }
        for (int i = 0; i < mm; ++i) {
            float bv = h0 ? (simd_sum(a0[i]) * sc0) : -1e30f;
            int bi = row0;
            if (h1) {
                const float v1 = simd_sum(a1[i]) * sc1;
                if (v1 > bv) { bv = v1; bi = row0 + 1; }
            }
            if (lane == 0) { sh_v[i * 8 + sg] = bv; sh_i[i * 8 + sg] = bi; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(mm)) {
            const int i = int(tid);
            float tv = sh_v[i * 8 + 0]; int ti = sh_i[i * 8 + 0];
            for (int s = 1; s < 8; ++s) {
                const float sv = sh_v[i * 8 + s]; const int si = sh_i[i * 8 + s];
                if (sv > tv || (sv == tv && si < ti)) { tv = sv; ti = si; }
            }
            part_v[(size_t)(m0 + i) * nparts + tg] = tv;
            part_i[(size_t)(m0 + i) * nparts + tg] = ti;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void ink_unembed_argmax_m_sk6(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* part_v  [[buffer(2)]],
    device int* part_i    [[buffer(3)]],
    constant int& N       [[buffer(4)]],
    constant int& K       [[buffer(5)]],
    constant int& M       [[buffer(6)]],
    constant int& nparts  [[buffer(7)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[8 * 8];
    threadgroup int sh_i[8 * 8];
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    const bool h0 = row0 < N, h1 = (row0 + 1) < N;
    constexpr int SB = 198;
    const int nsb = K >> 8;
    const size_t rowb = (size_t)nsb * SB;
    device const uchar* r0 = W + (size_t)(h0 ? row0 : 0) * rowb;
    device const uchar* r1 = h1 ? (r0 + rowb) : r0;

    for (int m0 = 0; m0 < M; m0 += 8) {
        const int mm = min(8, M - m0);
        float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0};
        if (h0) {
            for (int unit = int(lane); unit < nsb * 32; unit += 32) {
                const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                device const uchar* b0 = r0 + (size_t)sbi * SB;
                const float e0 = float(as_type<half>(ld16(b0))) * (float(b0[2 + g]) * (1.0f / 255.0f));
                float4 w00, w01;
                sk_w8<6>(b0 + 6, j, w00, w01);
                float e1 = 0.0f; float4 w10 = 0.0f, w11 = 0.0f;
                if (h1) {
                    device const uchar* b1 = r1 + (size_t)sbi * SB;
                    e1 = float(as_type<half>(ld16(b1))) * (float(b1[2 + g]) * (1.0f / 255.0f));
                    sk_w8<6>(b1 + 6, j, w10, w11);
                }
                for (int i = 0; i < mm; ++i) {
                    device const float4* xp = (device const float4*)(X + (size_t)(m0 + i) * K + sbi * 256 + j);
                    const float4 x0 = xp[0], x1 = xp[1];
                    a0[i] = fma(e0, dot(w00, x0) + dot(w01, x1), a0[i]);
                    if (h1) a1[i] = fma(e1, dot(w10, x0) + dot(w11, x1), a1[i]);
                }
            }
        }
        for (int i = 0; i < mm; ++i) {
            float bv = h0 ? simd_sum(a0[i]) : -1e30f;
            int bi = row0;
            if (h1) {
                const float v1 = simd_sum(a1[i]);
                if (v1 > bv) { bv = v1; bi = row0 + 1; }
            }
            if (lane == 0) { sh_v[i * 8 + sg] = bv; sh_i[i * 8 + sg] = bi; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(mm)) {
            const int i = int(tid);
            float tv = sh_v[i * 8 + 0]; int ti = sh_i[i * 8 + 0];
            for (int s = 1; s < 8; ++s) {
                const float sv = sh_v[i * 8 + s]; const int si = sh_i[i * 8 + s];
                if (sv > tv || (sv == tv && si < ti)) { tv = sv; ti = si; }
            }
            part_v[(size_t)(m0 + i) * nparts + tg] = tv;
            part_i[(size_t)(m0 + i) * nparts + tg] = ti;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Multi-row partial argmax reduction: grid (M, 1, 1), 256 threads
kernel void ink_argmax_part_reduce_m(
    device const float* part_v [[buffer(0)]],
    device const int* part_i   [[buffer(1)]],
    device int* out_ids        [[buffer(2)]],
    constant int& n_parts      [[buffer(3)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[256];
    threadgroup int sh_i[256];
    device const float* pv = part_v + (size_t)m * n_parts;
    device const int*   pi = part_i + (size_t)m * n_parts;
    float bv = -1e30f; int bi = 0;
    for (int i = int(tid); i < n_parts; i += 256) {
        const float v = pv[i]; const int idx = pi[i];
        if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
    }
    sh_v[tid] = bv; sh_i[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (sh_v[tid + s] > sh_v[tid] || (sh_v[tid + s] == sh_v[tid] && sh_i[tid + s] < sh_i[tid])) {
                sh_v[tid] = sh_v[tid + s]; sh_i[tid] = sh_i[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out_ids[m] = sh_i[0];
}

// --- Q8_ROW GEMV for M=1..8 (single DRAM weight pass, ZERO barriers, exact FP32 activations!)
kernel void ink_linear_q8row_mv(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    if (row0 >= N) return;
    const bool has1 = (row0 + 1) < N;
    device const float* scales = (device const float*)W;
    const float sc0 = scales[row0];
    const float sc1 = has1 ? scales[row0 + 1] : 0.0f;
    device const char4* q0p = (device const char4*)(W + (size_t)N * 4 + (size_t)row0 * K);
    device const char4* q1p = (device const char4*)(W + (size_t)N * 4 + (size_t)(row0 + (has1 ? 1 : 0)) * K);
    const int K4 = K >> 2;

    if (M == 1) {
        float a0 = 0.0f, a1 = 0.0f;
        device const float4* xp = (device const float4*)X;
        for (int k = int(lane); k < K4; k += 32) {
            const float4 xv = xp[k];
            a0 += dot(float4(q0p[k]), xv);
            if (has1) a1 += dot(float4(q1p[k]), xv);
        }
        const float v0 = simd_sum(a0) * sc0;
        if (lane == 0) Y[row0] = v0;
        if (has1) {
            const float v1 = simd_sum(a1) * sc1;
            if (lane == 0) Y[row0 + 1] = v1;
        }
        return;
    }

    if (M <= 8) {
        float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0};
        for (int k = int(lane); k < K4; k += 32) {
            const float4 w0 = float4(q0p[k]);
            const float4 w1 = has1 ? float4(q1p[k]) : float4(0.f);
            for (int i = 0; i < M; ++i) {
                const float4 xv = ((device const float4*)(X + (size_t)i * K))[k];
                a0[i] += dot(w0, xv);
                if (has1) a1[i] += dot(w1, xv);
            }
        }
        for (int i = 0; i < M; ++i) {
            const float v0 = simd_sum(a0[i]) * sc0;
            if (lane == 0) Y[(size_t)i * N + row0] = v0;
            if (has1) {
                const float v1 = simd_sum(a1[i]) * sc1;
                if (lane == 0) Y[(size_t)i * N + row0 + 1] = v1;
            }
        }
        return;
    }

    for (int m0 = 0; m0 < M; m0 += 4) {
        const int mm = min(4, M - m0);
        float a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
        for (int k = int(lane); k < K4; k += 32) {
            const float4 w0 = float4(q0p[k]);
            const float4 w1 = has1 ? float4(q1p[k]) : float4(0.f);
            for (int i = 0; i < mm; ++i) {
                const float4 xv = ((device const float4*)(X + (size_t)(m0 + i) * K))[k];
                a0[i] += dot(w0, xv);
                if (has1) a1[i] += dot(w1, xv);
            }
        }
        for (int i = 0; i < mm; ++i) {
            const float v0 = simd_sum(a0[i]) * sc0;
            if (lane == 0) Y[(size_t)(m0 + i) * N + row0] = v0;
            if (has1) {
                const float v1 = simd_sum(a1[i]) * sc1;
                if (lane == 0) Y[(size_t)(m0 + i) * N + row0 + 1] = v1;
            }
        }
    }
}

// --- Q8_ROW simdgroup_matrix 8x8 GEMM for M > 8
kernel void ink_linear_q8row_mm(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 32];
    threadgroup half sB[64 * 32];
    const int m0 = int(tgp.y) * 32, n0 = int(tgp.x) * 64;
    if (m0 >= M || n0 >= N) return;
    device const float* scales = (device const float*)W;
    device const char* qw = (device const char*)(W + (size_t)N * 4);
    const int sgm = int(sg >> 2), sgn = int(sg & 3u);
    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += 32) {
        for (uint idx = tid * 4; idx < 32 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gm = m0 + r;
            if (gm < M && k0 + c + 3 < K)
                *((threadgroup half4*)(sA + r * 32 + c)) = half4(*((device const float4*)(X + (size_t)gm * K + k0 + c)));
            else
                for (int u = 0; u < 4; ++u) sA[r * 32 + c + u] = (gm < M && k0 + c + u < K) ? half(X[(size_t)gm * K + k0 + c + u]) : half(0.0h);
        }
        for (uint idx = tid * 4; idx < 64 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gn = n0 + r;
            if (gn < N && k0 + c + 3 < K)
                *((threadgroup half4*)(sB + r * 32 + c)) = half4(float4(*((device const char4*)(qw + (size_t)gn * K + k0 + c))));
            else
                for (int u = 0; u < 4; ++u) sB[r * 32 + c + u] = (gn < N && k0 + c + u < K) ? half(float(qw[(size_t)gn * K + k0 + c + u])) : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, sA + (sgm * 16 + 0) * 32 + kk * 8, 32);
            simdgroup_load(a1, sA + (sgm * 16 + 8) * 32 + kk * 8, 32);
            simdgroup_load(b0, sB + (sgn * 16 + 0) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_load(b1, sB + (sgn * 16 + 8) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    for (int i = 0; i < 2; ++i) {
        const int gm = m0 + sgm * 16 + i * 8 + int(fm);
        if (gm < M) {
            for (int j = 0; j < 2; ++j) {
                const int gn = n0 + sgn * 16 + j * 8 + int(fn);
                if (gn < N) Y[(size_t)gm * N + gn] = acc[i][j].thread_elements()[0] * scales[gn];
                if (gn + 1 < N) Y[(size_t)gm * N + gn + 1] = acc[i][j].thread_elements()[1] * scales[gn + 1];
            }
        }
    }
}

// --- F32 simdgroup_matrix 8x8 GEMM for M > 8 (prefill controllers, sinks, dynamic sconv projs!)
kernel void ink_linear_f32_mm(
    device const float* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 32];
    threadgroup half sB[64 * 32];
    const int m0 = int(tgp.y) * 32, n0 = int(tgp.x) * 64;
    if (m0 >= M || n0 >= N) return;
    const int sgm = int(sg >> 2), sgn = int(sg & 3u);
    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < K; k0 += 32) {
        for (uint idx = tid * 4; idx < 32 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gm = m0 + r;
            if (gm < M && k0 + c + 3 < K)
                *((threadgroup half4*)(sA + r * 32 + c)) = half4(*((device const float4*)(X + (size_t)gm * K + k0 + c)));
            else
                for (int u = 0; u < 4; ++u) sA[r * 32 + c + u] = (gm < M && k0 + c + u < K) ? half(X[(size_t)gm * K + k0 + c + u]) : half(0.0h);
        }
        for (uint idx = tid * 4; idx < 64 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gn = n0 + r;
            if (gn < N && k0 + c + 3 < K)
                *((threadgroup half4*)(sB + r * 32 + c)) = half4(*((device const float4*)(W + (size_t)gn * K + k0 + c)));
            else
                for (int u = 0; u < 4; ++u) sB[r * 32 + c + u] = (gn < N && k0 + c + u < K) ? half(W[(size_t)gn * K + k0 + c + u]) : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, sA + (sgm * 16 + 0) * 32 + kk * 8, 32);
            simdgroup_load(a1, sA + (sgm * 16 + 8) * 32 + kk * 8, 32);
            simdgroup_load(b0, sB + (sgn * 16 + 0) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_load(b1, sB + (sgn * 16 + 8) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    for (int i = 0; i < 2; ++i) {
        const int gm = m0 + sgm * 16 + i * 8 + int(fm);
        if (gm < M) {
            for (int j = 0; j < 2; ++j) {
                const int gn = n0 + sgn * 16 + j * 8 + int(fn);
                if (gn < N) Y[(size_t)gm * N + gn] = acc[i][j].thread_elements()[0];
                if (gn + 1 < N) Y[(size_t)gm * N + gn + 1] = acc[i][j].thread_elements()[1];
            }
        }
    }
}

// --- F32 Linear (exact FP32! 2D grid ((N+3)/4, (M+3)/4) for M=1..127 -> saturates all 40 cores!)
kernel void ink_linear_f32(
    device const float* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row = int(tgp.x) * 4 + int(sg);
    if (row >= N) return;
    const int m0 = int(tgp.y) * 4;
    if (m0 >= M) return;
    device const float* wp = W + (size_t)row * K;
    const int K4 = K >> 2;
    if (M == 1) {
        float acc = 0.0f;
        device const float4* w4 = (device const float4*)wp;
        device const float4* x4 = (device const float4*)X;
        for (int k = int(lane); k < K4; k += 32) acc += dot(w4[k], x4[k]);
        const float v = simd_sum(acc);
        if (lane == 0) Y[row] = v;
        return;
    }
    const int mm = min(4, M - m0);
    float a[4] = {0, 0, 0, 0};
    for (int k = int(lane); k < K4; k += 32) {
        const float4 wv = ((device const float4*)wp)[k];
        for (int i = 0; i < mm; ++i) a[i] += dot(wv, ((device const float4*)(X + (size_t)(m0 + i) * K))[k]);
    }
    for (int i = 0; i < mm; ++i) {
        const float v = simd_sum(a[i]);
        if (lane == 0) Y[(size_t)(m0 + i) * N + row] = v;
    }
}

// --- Exact FP32 simdgroup_float8x8 Tiled GEMM (BM=32, BN=32, BK=16, 4 SGs = 128 threads, ZERO FP16 truncation!)
// Used for MoE router at M >= 64 -> ~70x faster than serial ink_linear_f32 with exact FP32 routing!
kernel void ink_linear_f32_exact_mm(
    device const float* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sA[32 * 16];
    threadgroup float sB[32 * 16];
    threadgroup float sO[32 * 32];
    const int m0 = int(tgp.y) * 32;
    const int n0 = int(tgp.x) * 32;
    if (m0 >= M || n0 >= N) return;

    const int sgm = int(sg >> 1);  // 0..1 -> 16 rows/SG
    const int sgn = int(sg & 1u);  // 0..1 -> 16 cols/SG
    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const int r = int(tid >> 2);         // 0..31
    const int c4 = int((tid & 3u) * 4u); // 0, 4, 8, 12
    const int gm = m0 + r;
    const int gn = n0 + r;
    for (int k0 = 0; k0 < K; k0 += 16) {
        if (gm < M && k0 + c4 + 3 < K) {
            *((threadgroup float4*)(sA + r * 16 + c4)) = *((device const float4*)(X + (size_t)gm * K + k0 + c4));
        } else {
            for (int u = 0; u < 4; ++u)
                sA[r * 16 + c4 + u] = (gm < M && k0 + c4 + u < K) ? X[(size_t)gm * K + k0 + c4 + u] : 0.0f;
        }
        if (gn < N && k0 + c4 + 3 < K) {
            *((threadgroup float4*)(sB + r * 16 + c4)) = *((device const float4*)(W + (size_t)gn * K + k0 + c4));
        } else {
            for (int u = 0; u < 4; ++u)
                sB[r * 16 + c4 + u] = (gn < N && k0 + c4 + u < K) ? W[(size_t)gn * K + k0 + c4 + u] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 2; ++kk) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, sA + (sgm * 16 + 0) * 16 + kk * 8, 16);
            simdgroup_load(a1, sA + (sgm * 16 + 8) * 16 + kk * 8, 16);
            simdgroup_load(b0, sB + (sgn * 16 + 0) * 16 + kk * 8, 16, ulong2(0, 0), true);
            simdgroup_load(b1, sB + (sgn * 16 + 8) * 16 + kk * 8, 16, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            simdgroup_store(acc[i][j], sO + (sgm * 16 + i * 8) * 32 + (sgn * 16 + j * 8), 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = int(tid); idx < 32 * 32; idx += 128) {
        const int om = idx >> 5, on = idx & 31;
        if (m0 + om < M && n0 + on < N) {
            Y[(size_t)(m0 + om) * N + (n0 + on)] = sO[idx];
        }
    }
}

// --- FAST 2-Row-per-SG Register-Only S_K Linear (16 rows/TG! Huge speedup on S6_K unembed & S5_K DSpark!)
template <int F>
kernel void ink_linear_sk_mv(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    if (row0 >= N) return;
    const bool has1 = (row0 + 1) < N;
    constexpr int SB = sk_sb_bytes(F);
    const int nsb = K >> 8;
    const size_t rowb = (size_t)nsb * SB;
    device const uchar* r0 = W + (size_t)row0 * rowb;
    device const uchar* r1 = has1 ? (r0 + rowb) : r0;

    if (M == 1) {
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int unit = int(lane); unit < nsb * 32; unit += 32) {
            const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
            device const float4* xp = (device const float4*)(X + sbi * 256 + j);
            const float4 x0 = xp[0], x1 = xp[1];
            {
                device const uchar* blk = r0 + (size_t)sbi * SB;
                const float eff = float(as_type<half>(ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
                float4 w0, w1;
                sk_w8<F>(blk + 6, j, w0, w1);
                acc0 = fma(eff, dot(w0, x0) + dot(w1, x1), acc0);
            }
            if (has1) {
                device const uchar* blk = r1 + (size_t)sbi * SB;
                const float eff = float(as_type<half>(ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
                float4 w0, w1;
                sk_w8<F>(blk + 6, j, w0, w1);
                acc1 = fma(eff, dot(w0, x0) + dot(w1, x1), acc1);
            }
        }
        const float v0 = simd_sum(acc0);
        if (lane == 0) Y[row0] = v0;
        if (has1) {
            const float v1 = simd_sum(acc1);
            if (lane == 0) Y[row0 + 1] = v1;
        }
        return;
    }

    if (M <= 8) {
        float a0[8] = {0, 0, 0, 0, 0, 0, 0, 0}, a1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int unit = int(lane); unit < nsb * 32; unit += 32) {
            const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
            device const uchar* b0 = r0 + (size_t)sbi * SB;
            const float e0 = float(as_type<half>(ld16(b0))) * (float(b0[2 + g]) * (1.0f / 255.0f));
            float4 w00, w01;
            sk_w8<F>(b0 + 6, j, w00, w01);
            float e1 = 0.0f; float4 w10 = 0.0f, w11 = 0.0f;
            if (has1) {
                device const uchar* b1 = r1 + (size_t)sbi * SB;
                e1 = float(as_type<half>(ld16(b1))) * (float(b1[2 + g]) * (1.0f / 255.0f));
                sk_w8<F>(b1 + 6, j, w10, w11);
            }
            for (int i = 0; i < M; ++i) {
                device const float4* xp = (device const float4*)(X + (size_t)i * K + sbi * 256 + j);
                const float4 x0 = xp[0], x1 = xp[1];
                a0[i] = fma(e0, dot(w00, x0) + dot(w01, x1), a0[i]);
                if (has1) a1[i] = fma(e1, dot(w10, x0) + dot(w11, x1), a1[i]);
            }
        }
        for (int i = 0; i < M; ++i) {
            const float v0 = simd_sum(a0[i]);
            if (lane == 0) Y[(size_t)i * N + row0] = v0;
            if (has1) {
                const float v1 = simd_sum(a1[i]);
                if (lane == 0) Y[(size_t)i * N + row0 + 1] = v1;
            }
        }
        return;
    }

    for (int m0 = 0; m0 < M; m0 += 4) {
        const int mm = min(4, M - m0);
        float a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
        for (int unit = int(lane); unit < nsb * 32; unit += 32) {
            const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
            device const uchar* b0 = r0 + (size_t)sbi * SB;
            const float e0 = float(as_type<half>(ld16(b0))) * (float(b0[2 + g]) * (1.0f / 255.0f));
            float4 w00, w01;
            sk_w8<F>(b0 + 6, j, w00, w01);
            float e1 = 0.0f; float4 w10 = 0.0f, w11 = 0.0f;
            if (has1) {
                device const uchar* b1 = r1 + (size_t)sbi * SB;
                e1 = float(as_type<half>(ld16(b1))) * (float(b1[2 + g]) * (1.0f / 255.0f));
                sk_w8<F>(b1 + 6, j, w10, w11);
            }
            for (int i = 0; i < mm; ++i) {
                device const float4* xp = (device const float4*)(X + (size_t)(m0 + i) * K + sbi * 256 + j);
                const float4 x0 = xp[0], x1 = xp[1];
                a0[i] = fma(e0, dot(w00, x0) + dot(w01, x1), a0[i]);
                if (has1) a1[i] = fma(e1, dot(w10, x0) + dot(w11, x1), a1[i]);
            }
        }
        for (int i = 0; i < mm; ++i) {
            const float v0 = simd_sum(a0[i]);
            if (lane == 0) Y[(size_t)(m0 + i) * N + row0] = v0;
            if (has1) {
                const float v1 = simd_sum(a1[i]);
                if (lane == 0) Y[(size_t)(m0 + i) * N + row0 + 1] = v1;
            }
        }
    }
}

#define INK_LINSK(F) template [[host_name("ink_linear_sk_mv_" #F)]] kernel void ink_linear_sk_mv<F>(device const uchar*, device const float*, device float*, constant int&, constant int&, constant int&, uint, uint);
INK_LINSK(2) INK_LINSK(3) INK_LINSK(4) INK_LINSK(5) INK_LINSK(6) INK_LINSK(8)

// --- S6_K simdgroup_matrix 8x64 GEMM for M=2..8 (single DRAM pass on SQ-15/SQ-10 DSpark M=8 verify unembed!)
kernel void ink_linear_sk6_mm8(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[8 * 32];
    threadgroup half sB[64 * 32];
    const int n0 = int(tg) * 64;
    if (n0 >= N) return;
    constexpr int SB = 198;
    const size_t rowb = (size_t)(K >> 8) * SB;
    const int br = int(tid >> 2), bc = int((tid & 3u) * 8u);
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < K; k0 += 32) {
        const int ar = int(tid >> 5), ac = int(tid & 31u);
        sA[tid] = (ar < M && k0 + ac < K) ? half(X[(size_t)ar * K + k0 + ac]) : half(0.0h);
        const int gn = n0 + br;
        if (gn < N) {
            const int col = k0 + bc;
            const int sbi = col >> 8, j = col & 255, g = j >> 6;
            device const uchar* blk = W + (size_t)gn * rowb + (size_t)sbi * SB;
            const float eff = float(as_type<half>(ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
            float4 w0, w1;
            sk_w8<6>(blk + 6, j, w0, w1);
            *((threadgroup half4*)(sB + br * 32 + bc + 0)) = half4(w0 * eff);
            *((threadgroup half4*)(sB + br * 32 + bc + 4)) = half4(w1 * eff);
        } else {
            *((threadgroup half4*)(sB + br * 32 + bc + 0)) = half4(0.0h);
            *((threadgroup half4*)(sB + br * 32 + bc + 4)) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 a0, b0;
            simdgroup_load(a0, sA + kk * 8, 32);
            simdgroup_load(b0, sB + (int(sg) * 8) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a0, b0, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint qid = lane >> 2;
    const int gm = int((qid & 4u) + ((lane >> 1) & 3u));
    const int gn = n0 + int(sg) * 8 + int((qid & 2u) * 2u + (lane & 1u) * 2u);
    if (gm < M) {
        if (gn < N) Y[(size_t)gm * N + gn] = acc.thread_elements()[0];
        if (gn + 1 < N) Y[(size_t)gm * N + gn + 1] = acc.thread_elements()[1];
    }
}

// --- S5_K simdgroup_matrix 8x64 GEMM for M=2..8 (3.1x less DRAM than FP16 for SQ-10/SQ-15 DSpark layers!)
kernel void ink_linear_sk5_mm8(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[8 * 32];
    threadgroup half sB[64 * 32];
    const int n0 = int(tg) * 64;
    if (n0 >= N) return;
    constexpr int SB = 166;
    const size_t rowb = (size_t)(K >> 8) * SB;
    const int br = int(tid >> 2), bc = int((tid & 3u) * 8u);
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < K; k0 += 32) {
        const int ar = int(tid >> 5), ac = int(tid & 31u);
        sA[tid] = (ar < M && k0 + ac < K) ? half(X[(size_t)ar * K + k0 + ac]) : half(0.0h);
        const int gn = n0 + br;
        if (gn < N) {
            const int col = k0 + bc;
            const int sbi = col >> 8, j = col & 255, g = j >> 6;
            device const uchar* blk = W + (size_t)gn * rowb + (size_t)sbi * SB;
            const float eff = float(as_type<half>(ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
            float4 w0, w1;
            sk_w8<5>(blk + 6, j, w0, w1);
            *((threadgroup half4*)(sB + br * 32 + bc + 0)) = half4(w0 * eff);
            *((threadgroup half4*)(sB + br * 32 + bc + 4)) = half4(w1 * eff);
        } else {
            *((threadgroup half4*)(sB + br * 32 + bc + 0)) = half4(0.0h);
            *((threadgroup half4*)(sB + br * 32 + bc + 4)) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 a0, b0;
            simdgroup_load(a0, sA + kk * 8, 32);
            simdgroup_load(b0, sB + (int(sg) * 8) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a0, b0, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint qid = lane >> 2;
    const int gm = int((qid & 4u) + ((lane >> 1) & 3u));
    const int gn = n0 + int(sg) * 8 + int((qid & 2u) * 2u + (lane & 1u) * 2u);
    if (gm < M) {
        if (gn < N) Y[(size_t)gm * N + gn] = acc.thread_elements()[0];
        if (gn + 1 < N) Y[(size_t)gm * N + gn + 1] = acc.thread_elements()[1];
    }
}

// ============================================================================
// 4. Inkling-2-Mini MoE Pipeline (Register-Only Decoders + 8-Token Micro-Batching!)
// ============================================================================

struct InkExpertDev {
    ulong body_off;
    uint dn_off;
    uchar gu_code;
    uchar dn_code;
    ushort chunk_idx;
};

static inline device const uchar* ink_sec4_ptr(
    ushort cidx,
    device const uchar* s0,
    device const uchar* s1,
    device const uchar* s2,
    device const uchar* s3) {
    if (cidx == 0u) return s0;
    if (cidx == 1u) return s1;
    if (cidx == 2u) return s2;
    return s3;
}

// ============================================================================
// 4A. Native Lossless GGUF_RAW (code=20) Decoders (All 10+ Unsloth GGUF Quants!)
//     Q4_K (12), Q5_K (13), Q6_K (14), Q8_0 (8), IQ4_XS (23), IQ3_XXS (18),
//     IQ4_NL (20), Q4_0 (2), Q3_K (11), Q2_K (10), MXFP4 (39), BF16/F16/F32
// ============================================================================
struct InkGgufExpertHeader {
    uint magic;      // 0x47475546 ('GGUF')
    uint gu_type;    // ggml_type for gate/up
    uint dn_type;    // ggml_type for down
    uint flags;      // bit 0: 1=interleaved, 0=split [gate 0..F-1, up F..2F-1]
    float gate_scale;
    float up_scale;
    float dn_scale;
    uint reserved;
};

constant float kIq4NlF32[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

constant uchar kIq3XxsSigns[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
};

constant uint kIq3XxsGrid[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u
};

static inline size_t gguf_row_bytes_m(uint t, int K) {
    if (t == 12u) return (size_t)(K >> 8) * 144ul;
    if (t == 13u) return (size_t)(K >> 8) * 176ul;
    if (t == 14u) return (size_t)(K >> 8) * 210ul;
    if (t == 8u)  return (size_t)(K >> 5) * 34ul;
    if (t == 23u) return (size_t)(K >> 8) * 136ul;
    if (t == 18u) return (size_t)(K >> 8) * 98ul;
    if (t == 20u || t == 2u) return (size_t)(K >> 5) * 18ul;
    if (t == 11u) return (size_t)(K >> 8) * 110ul;
    if (t == 10u) return (size_t)(K >> 8) * 84ul;
    if (t == 39u) return (size_t)(K >> 5) * 17ul;
    if (t == 1u || t == 30u) return (size_t)K * 2ul;
    if (t == 0u)  return (size_t)K * 4ul;
    return 0ul;
}

static inline float gguf_q3k_scale(device const uchar* sc, int is) {
    int s = 0;
    if (is < 4)       s = int(sc[is] & 0xFu) | (int((sc[8 + is] >> 0) & 3u) << 4);
    else if (is < 8)  s = int(sc[is] & 0xFu) | (int((sc[8 + (is - 4)] >> 2) & 3u) << 4);
    else if (is < 12) s = int(sc[is - 8] >> 4) | (int((sc[8 + (is - 8)] >> 4) & 3u) << 4);
    else              s = int(sc[is - 8] >> 4) | (int((sc[8 + (is - 12)] >> 6) & 3u) << 4);
    return float(s - 32);
}

// Decode 8 consecutive weights starting at 8-aligned `col` from a single GGUF row into float4 w0, w1
static inline void gguf_decode_8(uint t, device const uchar* row, int col, thread float4& w0, thread float4& w1) {
    if (t == 12u) {  // Q4_K: 256 elems = 144 B
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 144ul;
        const float d = float(as_type<half>(ld16(blk)));
        const float dmin = float(as_type<half>(ld16(blk + 2)));
        device const uchar* sc = blk + 4;
        device const uchar* qs = blk + 16;
        const int j64 = idx >> 6, in64 = idx & 63, sub = j64 * 2 + (in64 >> 5);
        uchar sd, sm;
        if (sub < 4) { sd = sc[sub] & 63u; sm = sc[sub + 4] & 63u; }
        else { sd = (sc[sub + 4] & 0xFu) | ((sc[sub - 4] >> 6) << 4); sm = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4); }
        const float dl = d * float(sd), ml = -dmin * float(sm);
        device const uchar* qp = qs + j64 * 32 + (in64 & 31);
        if (in64 < 32) {
            w0 = float4(float(qp[0] & 15u), float(qp[1] & 15u), float(qp[2] & 15u), float(qp[3] & 15u)) * dl + ml;
            w1 = float4(float(qp[4] & 15u), float(qp[5] & 15u), float(qp[6] & 15u), float(qp[7] & 15u)) * dl + ml;
        } else {
            w0 = float4(float(qp[0] >> 4), float(qp[1] >> 4), float(qp[2] >> 4), float(qp[3] >> 4)) * dl + ml;
            w1 = float4(float(qp[4] >> 4), float(qp[5] >> 4), float(qp[6] >> 4), float(qp[7] >> 4)) * dl + ml;
        }
    } else if (t == 13u) {  // Q5_K: 256 elems = 176 B
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 176ul;
        const float d = float(as_type<half>(ld16(blk)));
        const float dmin = float(as_type<half>(ld16(blk + 2)));
        device const uchar* sc = blk + 4;
        device const uchar* qh = blk + 16;
        device const uchar* qs = blk + 48;
        const int j64 = idx >> 6, in64 = idx & 63, sub = j64 * 2 + (in64 >> 5), l = in64 & 31;
        uchar sd, sm;
        if (sub < 4) { sd = sc[sub] & 63u; sm = sc[sub + 4] & 63u; }
        else { sd = (sc[sub + 4] & 0xFu) | ((sc[sub - 4] >> 6) << 4); sm = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4); }
        const float dl = d * float(sd), ml = -dmin * float(sm);
        device const uchar* qp = qs + j64 * 32 + l;
        device const uchar* hp = qh + l;
        const uint sh = (in64 < 32) ? 0u : 4u;
        float qv[8];
        for (int k = 0; k < 8; ++k) {
            const uint ql = (uint(qp[k]) >> sh) & 15u;
            const uint hb = (uint(hp[k]) >> uint(sub)) & 1u;
            qv[k] = float(ql | (hb << 4)) * dl + ml;
        }
        w0 = float4(qv[0], qv[1], qv[2], qv[3]);
        w1 = float4(qv[4], qv[5], qv[6], qv[7]);
    } else if (t == 14u) {  // Q6_K: 256 elems = 210 B
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 210ul;
        const float d = float(as_type<half>(ld16(blk + 208)));
        const int hf = idx >> 7, r = idx & 127, qg = r >> 5, l = r & 31, is = l >> 4;
        device const uchar* qlh = blk + hf * 64 + ((qg & 1) ? 32 : 0) + l;
        device const uchar* qhh = blk + 128 + hf * 32 + l;
        device const char* sch = (device const char*)(blk + 192 + hf * 8);
        const float dl = d * float(sch[is + qg * 2]);
        const uint qsh = (qg >= 2) ? 4u : 0u;
        const uint hsh = uint(qg * 2);
        float qv[8];
        for (int k = 0; k < 8; ++k) {
            const int q = int(((uint(qlh[k]) >> qsh) & 15u) | (((uint(qhh[k]) >> hsh) & 3u) << 4)) - 32;
            qv[k] = dl * float(q);
        }
        w0 = float4(qv[0], qv[1], qv[2], qv[3]);
        w1 = float4(qv[4], qv[5], qv[6], qv[7]);
    } else if (t == 8u) {  // Q8_0: 32 elems = 34 B
        const int b = col >> 5, in32 = col & 31;
        device const uchar* blk = row + (size_t)b * 34ul;
        const float d = float(as_type<half>(ld16(blk)));
        device const char* qs = (device const char*)(blk + 2 + in32);
        w0 = float4(float(qs[0]), float(qs[1]), float(qs[2]), float(qs[3])) * d;
        w1 = float4(float(qs[4]), float(qs[5]), float(qs[6]), float(qs[7])) * d;
    } else if (t == 23u) {  // IQ4_XS: 256 elems = 136 B
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 136ul;
        const float d = float(as_type<half>(ld16(blk)));
        const ushort sh = ld16(blk + 2);
        device const uchar* sl = blk + 4;
        device const uchar* qs = blk + 8;
        const int ib = idx >> 5, in32 = idx & 31;
        const int ls = int((sl[ib >> 1] >> (4 * (ib & 1))) & 15u) | (int((sh >> (2 * ib)) & 3u) << 4);
        const float dl = d * float(ls - 32);
        device const uchar* qp = qs + ib * 16 + (in32 & 15);
        if (in32 < 16) {
            w0 = float4(kIq4NlF32[qp[0] & 15u], kIq4NlF32[qp[1] & 15u], kIq4NlF32[qp[2] & 15u], kIq4NlF32[qp[3] & 15u]) * dl;
            w1 = float4(kIq4NlF32[qp[4] & 15u], kIq4NlF32[qp[5] & 15u], kIq4NlF32[qp[6] & 15u], kIq4NlF32[qp[7] & 15u]) * dl;
        } else {
            w0 = float4(kIq4NlF32[qp[0] >> 4], kIq4NlF32[qp[1] >> 4], kIq4NlF32[qp[2] >> 4], kIq4NlF32[qp[3] >> 4]) * dl;
            w1 = float4(kIq4NlF32[qp[4] >> 4], kIq4NlF32[qp[5] >> 4], kIq4NlF32[qp[6] >> 4], kIq4NlF32[qp[7] >> 4]) * dl;
        }
    } else if (t == 18u) {  // IQ3_XXS: 256 elems = 98 B
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 98ul;
        const float d = float(as_type<half>(ld16(blk)));
        device const uchar* qs = blk + 2;
        device const uchar* ss = blk + 66;
        const int ib32 = idx >> 5, l = (idx & 31) >> 3;
        const uint aux32 = uint(ss[4 * ib32 + 0]) | (uint(ss[4 * ib32 + 1]) << 8) |
                           (uint(ss[4 * ib32 + 2]) << 16) | (uint(ss[4 * ib32 + 3]) << 24);
        const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
        const uchar signs = kIq3XxsSigns[(aux32 >> uint(7 * l)) & 127u];
        const uint g1 = kIq3XxsGrid[qs[ib32 * 8 + 2 * l + 0]];
        const uint g2 = kIq3XxsGrid[qs[ib32 * 8 + 2 * l + 1]];
        w0 = float4(
            float((g1 >>  0) & 255u) * ((signs &   1u) ? -db : db),
            float((g1 >>  8) & 255u) * ((signs &   2u) ? -db : db),
            float((g1 >> 16) & 255u) * ((signs &   4u) ? -db : db),
            float((g1 >> 24) & 255u) * ((signs &   8u) ? -db : db));
        w1 = float4(
            float((g2 >>  0) & 255u) * ((signs &  16u) ? -db : db),
            float((g2 >>  8) & 255u) * ((signs &  32u) ? -db : db),
            float((g2 >> 16) & 255u) * ((signs &  64u) ? -db : db),
            float((g2 >> 24) & 255u) * ((signs & 128u) ? -db : db));
    } else if (t == 20u) {  // IQ4_NL: 32 elems = 18 B
        const int b = col >> 5, in32 = col & 31;
        device const uchar* blk = row + (size_t)b * 18ul;
        const float d = float(as_type<half>(ld16(blk)));
        device const uchar* qp = blk + 2 + (in32 & 15);
        if (in32 < 16) {
            w0 = float4(kIq4NlF32[qp[0] & 15u], kIq4NlF32[qp[1] & 15u], kIq4NlF32[qp[2] & 15u], kIq4NlF32[qp[3] & 15u]) * d;
            w1 = float4(kIq4NlF32[qp[4] & 15u], kIq4NlF32[qp[5] & 15u], kIq4NlF32[qp[6] & 15u], kIq4NlF32[qp[7] & 15u]) * d;
        } else {
            w0 = float4(kIq4NlF32[qp[0] >> 4], kIq4NlF32[qp[1] >> 4], kIq4NlF32[qp[2] >> 4], kIq4NlF32[qp[3] >> 4]) * d;
            w1 = float4(kIq4NlF32[qp[4] >> 4], kIq4NlF32[qp[5] >> 4], kIq4NlF32[qp[6] >> 4], kIq4NlF32[qp[7] >> 4]) * d;
        }
    } else if (t == 2u) {  // Q4_0: 32 elems = 18 B
        const int b = col >> 5, in32 = col & 31;
        device const uchar* blk = row + (size_t)b * 18ul;
        const float d = float(as_type<half>(ld16(blk)));
        device const uchar* qp = blk + 2 + (in32 & 15);
        if (in32 < 16) {
            w0 = (float4(float(qp[0] & 15u), float(qp[1] & 15u), float(qp[2] & 15u), float(qp[3] & 15u)) - 8.0f) * d;
            w1 = (float4(float(qp[4] & 15u), float(qp[5] & 15u), float(qp[6] & 15u), float(qp[7] & 15u)) - 8.0f) * d;
        } else {
            w0 = (float4(float(qp[0] >> 4), float(qp[1] >> 4), float(qp[2] >> 4), float(qp[3] >> 4)) - 8.0f) * d;
            w1 = (float4(float(qp[4] >> 4), float(qp[5] >> 4), float(qp[6] >> 4), float(qp[7] >> 4)) - 8.0f) * d;
        }
    } else if (t == 11u) {  // Q3_K: 256 elems = 110 B
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 110ul;
        device const uchar* hm = blk;
        device const uchar* qs = blk + 32;
        device const uchar* sc = blk + 96;
        const float d_all = float(as_type<half>(ld16(blk + 108)));
        const int n128 = idx >> 7, in128 = idx & 127, j = in128 >> 5, l32 = in128 & 31;
        const int is = n128 * 8 + j * 2 + (l32 >> 4);
        const float dl = d_all * gguf_q3k_scale(sc, is);
        const uint shift = uint(j * 2);
        const uchar mask = uchar(1u << uint(n128 * 4 + j));
        device const uchar* qp = qs + n128 * 32 + l32;
        device const uchar* hp = hm + l32;
        float qv[8];
        for (int k = 0; k < 8; ++k) {
            const int q2 = int((qp[k] >> shift) & 3u);
            const int hbit = (hp[k] & mask) ? 0 : 4;
            qv[k] = dl * float(q2 - hbit);
        }
        w0 = float4(qv[0], qv[1], qv[2], qv[3]);
        w1 = float4(qv[4], qv[5], qv[6], qv[7]);
    } else if (t == 10u) {  // Q2_K: 256 elems = 84 B
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 84ul;
        device const uchar* sc = blk;
        device const uchar* qs = blk + 16;
        const float d = float(as_type<half>(ld16(blk + 80)));
        const float dmin = float(as_type<half>(ld16(blk + 82)));
        const int n128 = idx >> 7, in128 = idx & 127, j = in128 >> 5, l32 = in128 & 31;
        const int is = n128 * 8 + j * 2 + (l32 >> 4);
        const uchar scb = sc[is];
        const float dl = d * float(scb & 15u), ml = -dmin * float(scb >> 4);
        const uint shift = uint(j * 2);
        device const uchar* qp = qs + n128 * 32 + l32;
        w0 = float4(float((qp[0] >> shift) & 3u), float((qp[1] >> shift) & 3u), float((qp[2] >> shift) & 3u), float((qp[3] >> shift) & 3u)) * dl + ml;
        w1 = float4(float((qp[4] >> shift) & 3u), float((qp[5] >> shift) & 3u), float((qp[6] >> shift) & 3u), float((qp[7] >> shift) & 3u)) * dl + ml;
    } else if (t == 39u) {  // MXFP4: 32 elems = 17 B
        const int b = col >> 5, in32 = col & 31;
        device const uchar* blk = row + (size_t)b * 17ul;
        const uint e = blk[0];
        const float d = (e < 2u) ? as_type<float>(0x00200000u << e) : as_type<float>((e - 1u) << 23);
        const float d2 = 2.0f * d;
        device const uchar* qp = blk + 1 + (in32 & 15);
        if (in32 < 16) {
            w0 = float4(kE2M1[qp[0] & 15u], kE2M1[qp[1] & 15u], kE2M1[qp[2] & 15u], kE2M1[qp[3] & 15u]) * d2;
            w1 = float4(kE2M1[qp[4] & 15u], kE2M1[qp[5] & 15u], kE2M1[qp[6] & 15u], kE2M1[qp[7] & 15u]) * d2;
        } else {
            w0 = float4(kE2M1[qp[0] >> 4], kE2M1[qp[1] >> 4], kE2M1[qp[2] >> 4], kE2M1[qp[3] >> 4]) * d2;
            w1 = float4(kE2M1[qp[4] >> 4], kE2M1[qp[5] >> 4], kE2M1[qp[6] >> 4], kE2M1[qp[7] >> 4]) * d2;
        }
    } else if (t == 30u) {  // BF16
        device const ushort* wp = (device const ushort*)(row + (size_t)col * 2ul);
        w0 = float4(as_type<float>(uint(wp[0]) << 16), as_type<float>(uint(wp[1]) << 16), as_type<float>(uint(wp[2]) << 16), as_type<float>(uint(wp[3]) << 16));
        w1 = float4(as_type<float>(uint(wp[4]) << 16), as_type<float>(uint(wp[5]) << 16), as_type<float>(uint(wp[6]) << 16), as_type<float>(uint(wp[7]) << 16));
    } else if (t == 1u) {   // F16
        device const half* wp = (device const half*)(row + (size_t)col * 2ul);
        w0 = float4(wp[0], wp[1], wp[2], wp[3]);
        w1 = float4(wp[4], wp[5], wp[6], wp[7]);
    } else if (t == 0u) {   // F32
        device const float4* wp = (device const float4*)(row + (size_t)col * 4ul);
        w0 = wp[0]; w1 = wp[1];
    } else {
        w0 = float4(0.0f); w1 = float4(0.0f);
    }
}

// Vectorized activation-reusing GGUF_RAW w13 dot (32 lanes/row, 288*8 = 2304 exact -> 9 iters/lane, ZERO IDLE LANES!)
static inline void gguf_w13_chunk(
    device const uchar* body, int r, device const float* x_in,
    thread const uint* tk, int mm, int lane, thread float* ag, thread float* au) {
    device const InkGgufExpertHeader* hdr = (device const InkGgufExpertHeader*)body;
    const uint t = hdr->gu_type;
    const size_t rb = gguf_row_bytes_m(t, 2304);
    const bool interlv = (hdr->flags & 1u) != 0u;
    device const uchar* base = body + 32;
    device const uchar* rg = base + (size_t)(interlv ? (2 * r) : r) * rb;
    device const uchar* ru = base + (size_t)(interlv ? (2 * r + 1) : (1152 + r)) * rb;
    const float gs = hdr->gate_scale, us = hdr->up_scale;
    for (int u = lane; u < 288; u += 32) {
        const int col = u << 3;
        float4 wg0, wg1, wu0, wu1;
        gguf_decode_8(t, rg, col, wg0, wg1);
        gguf_decode_8(t, ru, col, wu0, wu1);
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(x_in + ulong(tk[i]) * 2304ul + uint(col));
            const float4 x0 = xp[0], x1 = xp[1];
            ag[i] = fma(gs, dot(wg0, x0) + dot(wg1, x1), ag[i]);
            au[i] = fma(us, dot(wu0, x0) + dot(wu1, x1), au[i]);
        }
    }
}

// Vectorized activation-reusing GGUF_RAW w2 pair dot (32 lanes/pair, 144*8 = 1152)
static inline void gguf_w2_pair(
    device const uchar* body, uint dn_off, int rp, device const float* h_act,
    thread const uint* pid, int mm, int lane, thread float* a0, thread float* a1) {
    device const InkGgufExpertHeader* hdr = (device const InkGgufExpertHeader*)body;
    const uint t = hdr->dn_type;
    const size_t rb = gguf_row_bytes_m(t, 1152);
    device const uchar* base = body + dn_off;
    device const uchar* r0 = base + (size_t)(2 * rp) * rb;
    device const uchar* r1 = r0 + rb;
    const float ds = hdr->dn_scale;
    for (int u = lane; u < 144; u += 32) {
        const int col = u << 3;
        float4 u0, u1, v0, v1;
        gguf_decode_8(t, r0, col, u0, u1);
        gguf_decode_8(t, r1, col, v0, v1);
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(h_act + ulong(pid[i]) * 1152ul + uint(col));
            const float4 x0 = xp[0], x1 = xp[1];
            a0[i] = fma(ds, dot(u0, x0) + dot(u1, x1), a0[i]);
            a1[i] = fma(ds, dot(v0, x0) + dot(v1, x1), a1[i]);
        }
    }
}

static inline void decode_sb_gguf_16(device const uchar* row_ptr, uint t, float sc, int col, threadgroup half* dst) {
    float4 w0, w1, w2, w3;
    gguf_decode_8(t, row_ptr, col,     w0, w1);
    gguf_decode_8(t, row_ptr, col + 8, w2, w3);
    *((threadgroup half4*)(dst +  0)) = half4(w0 * sc);
    *((threadgroup half4*)(dst +  4)) = half4(w1 * sc);
    *((threadgroup half4*)(dst +  8)) = half4(w2 * sc);
    *((threadgroup half4*)(dst + 12)) = half4(w3 * sc);
}

kernel void ink_moe_plan(
    device const int* topk_idx [[buffer(0)]],
    constant int& P            [[buffer(1)]],
    device int* uexp           [[buffer(2)]],
    device int* uoff           [[buffer(3)]],
    device int* ucnt           [[buffer(4)]],
    device int* pair_list      [[buffer(5)]],
    device uint* meta          [[buffer(6)]],
    device uint2* tiles        [[buffer(7)]],
    uint e    [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup int sg_u[4];
    threadgroup int sg_p[4];
    threadgroup int sg_t[4];
    int c = 0;
    for (int i = 0; i < P; ++i) if (topk_idx[i] == int(e)) ++c;
    const int has = c > 0 ? 1 : 0;
    const int nt = (c + 31) >> 5;
    const int ex_u = simd_prefix_exclusive_sum(has);
    const int ex_p = simd_prefix_exclusive_sum(c);
    const int ex_t = simd_prefix_exclusive_sum(nt);
    if (lane == 31) { sg_u[sg] = ex_u + has; sg_p[sg] = ex_p + c; sg_t[sg] = ex_t + nt; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int base_u = 0, base_p = 0, base_t = 0;
    for (uint w = 0; w < sg; ++w) { base_u += sg_u[w]; base_p += sg_p[w]; base_t += sg_t[w]; }
    const int pos_u = base_u + ex_u, pos_p = base_p + ex_p, pos_t = base_t + ex_t;
    if (has) {
        uexp[pos_u] = int(e); uoff[pos_u] = pos_p; ucnt[pos_u] = c;
        int k = 0;
        for (int i = 0; i < P; ++i) if (topk_idx[i] == int(e)) pair_list[pos_p + k++] = i;
        for (int ti = 0; ti < nt; ++ti) {
            const int rem = min(32, c - ti * 32);
            tiles[pos_t + ti] = uint2((e << 16) | uint(rem), uint(pos_p + ti * 32));
        }
    }
    if (e == 127) {
        const uint n_uniq = uint(base_u + ex_u + has);
        const uint n_tiles = uint(base_t + ex_t + nt);
        meta[0] = n_uniq; meta[1] = uint(P);
        meta[2] = 144u; meta[3] = n_uniq; meta[4] = 1u;   // 8-row/TG (mx8 gu, sk/gen)
        meta[5] = 72u;  meta[6] = n_uniq; meta[7] = 1u;   // 16-row/TG (sf4 gu, mx8 dn)
        meta[8] = 36u;  meta[9] = n_tiles; meta[10] = 1u; // mm tiles
        meta[11] = 36u; meta[12] = n_uniq; meta[13] = 1u; // 32-pair/TG (sf4 dn)
    }
}

static inline float moe_sum16(float v) {
    v += simd_shuffle_xor(v, ushort(1));
    v += simd_shuffle_xor(v, ushort(2));
    v += simd_shuffle_xor(v, ushort(4));
    v += simd_shuffle_xor(v, ushort(8));
    return v;
}

static inline float moe_sum8(float v) {
    v += simd_shuffle_xor(v, ushort(1));
    v += simd_shuffle_xor(v, ushort(2));
    v += simd_shuffle_xor(v, ushort(4));
    return v;
}

// Vectorized 64-bit uint2 SF4 w13 (16 lanes/pair -> 144 = 9*16 exact iters, ZERO IDLE LANES!)
template <int STEP>
static inline void sf4_w13_chunk_s(
    device const uchar* body, int r, device const float* x_in,
    thread const uint* tk, int mm, int lane, thread float* ag, thread float* au) {
    device const float* hdr = (device const float*)body;
    const float outer = hdr[0] / hdr[1];
    device const uchar* p_base = body + 16;
    device const uchar* s_base = p_base + 2654208;
    const int rg = 2 * r, ru = 2 * r + 1;
    device const uint2* pg = (device const uint2*)(p_base + (size_t)rg * 1152);
    device const uchar* sg = s_base + (size_t)rg * 144;
    device const uint2* pu = (device const uint2*)(p_base + (size_t)ru * 1152);
    device const uchar* su = s_base + (size_t)ru * 144;

    for (int b = lane; b < 144; b += STEP) {
        const float scg = e4m3_to_f32(sg[b]) * outer;
        const float scu = e4m3_to_f32(su[b]) * outer;
        const uint2 bg = pg[b], bu = pu[b];
        const float4 wg0 = float4(kE2M1[bg.x & 15u], kE2M1[(bg.x >> 4) & 15u], kE2M1[(bg.x >> 8) & 15u], kE2M1[(bg.x >> 12) & 15u]);
        const float4 wg1 = float4(kE2M1[(bg.x >> 16) & 15u], kE2M1[(bg.x >> 20) & 15u], kE2M1[(bg.x >> 24) & 15u], kE2M1[bg.x >> 28]);
        const float4 wg2 = float4(kE2M1[bg.y & 15u], kE2M1[(bg.y >> 4) & 15u], kE2M1[(bg.y >> 8) & 15u], kE2M1[(bg.y >> 12) & 15u]);
        const float4 wg3 = float4(kE2M1[(bg.y >> 16) & 15u], kE2M1[(bg.y >> 20) & 15u], kE2M1[(bg.y >> 24) & 15u], kE2M1[bg.y >> 28]);
        const float4 wu0 = float4(kE2M1[bu.x & 15u], kE2M1[(bu.x >> 4) & 15u], kE2M1[(bu.x >> 8) & 15u], kE2M1[(bu.x >> 12) & 15u]);
        const float4 wu1 = float4(kE2M1[(bu.x >> 16) & 15u], kE2M1[(bu.x >> 20) & 15u], kE2M1[(bu.x >> 24) & 15u], kE2M1[bu.x >> 28]);
        const float4 wu2 = float4(kE2M1[bu.y & 15u], kE2M1[(bu.y >> 4) & 15u], kE2M1[(bu.y >> 8) & 15u], kE2M1[(bu.y >> 12) & 15u]);
        const float4 wu3 = float4(kE2M1[(bu.y >> 16) & 15u], kE2M1[(bu.y >> 20) & 15u], kE2M1[(bu.y >> 24) & 15u], kE2M1[bu.y >> 28]);
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(x_in + ulong(tk[i]) * 2304ul + uint(b * 16));
            const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            ag[i] = fma(scg, dot(wg0, x0) + dot(wg1, x1) + dot(wg2, x2) + dot(wg3, x3), ag[i]);
            au[i] = fma(scu, dot(wu0, x0) + dot(wu1, x1) + dot(wu2, x2) + dot(wu3, x3), au[i]);
        }
    }
}

// Vectorized 64-bit uint2 MX8 w13 (288 chunks of 8 = 9*32 exact iters, ZERO IDLE LANES!)
static inline void mx8_w13_chunk(
    device const uchar* body, int r, device const float* x_in,
    thread const uint* tk, int mm, int lane, thread float* ag, thread float* au) {
    device const uchar* w_base = body;
    device const uchar* s_base = body + 5308416;
    const int rg = 2 * r, ru = 2 * r + 1;
    device const uint2* wg = (device const uint2*)(w_base + (size_t)rg * 2304);
    device const uchar* sg = s_base + (size_t)rg * 72;
    device const uint2* wu = (device const uint2*)(w_base + (size_t)ru * 2304);
    device const uchar* su = s_base + (size_t)ru * 72;

    for (int u = lane; u < 288; u += 32) {
        const int b = u >> 2;
        const float scg = ue8m0_to_f32(sg[b]);
        const float scu = ue8m0_to_f32(su[b]);
        const uint2 gw = wg[u], uw = wu[u];
        const float4 g0 = float4(e4m3_to_f32(gw.x & 255u), e4m3_to_f32((gw.x >> 8) & 255u), e4m3_to_f32((gw.x >> 16) & 255u), e4m3_to_f32(gw.x >> 24));
        const float4 g1 = float4(e4m3_to_f32(gw.y & 255u), e4m3_to_f32((gw.y >> 8) & 255u), e4m3_to_f32((gw.y >> 16) & 255u), e4m3_to_f32(gw.y >> 24));
        const float4 u0 = float4(e4m3_to_f32(uw.x & 255u), e4m3_to_f32((uw.x >> 8) & 255u), e4m3_to_f32((uw.x >> 16) & 255u), e4m3_to_f32(uw.x >> 24));
        const float4 u1 = float4(e4m3_to_f32(uw.y & 255u), e4m3_to_f32((uw.y >> 8) & 255u), e4m3_to_f32((uw.y >> 16) & 255u), e4m3_to_f32(uw.y >> 24));
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(x_in + ulong(tk[i]) * 2304ul + uint(u * 8));
            const float4 x0 = xp[0], x1 = xp[1];
            ag[i] = fma(scg, dot(g0, x0) + dot(g1, x1), ag[i]);
            au[i] = fma(scu, dot(u0, x0) + dot(u1, x1), au[i]);
        }
    }
}

template <int F>
static inline void sk_w13_chunk(
    device const uchar* body, int r, device const float* x_in,
    thread const uint* tk, int mm, int lane, thread float* ag, thread float* au) {
    constexpr int SB = sk_sb_bytes(F);
    const size_t rowb = 9ul * SB;
    device const uchar* gu = body + 2304;
    device const uchar* rg = gu + (size_t)(2 * r) * rowb;
    device const uchar* ru = gu + (size_t)(2 * r + 1) * rowb;
    for (int unit = lane; unit < 288; unit += 32) {
        const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
        device const uchar* bg = rg + (size_t)sbi * SB;
        device const uchar* bu = ru + (size_t)sbi * SB;
        const float effg = float(as_type<half>(ld16(bg))) * (float(bg[2 + g]) * (1.0f / 255.0f));
        const float effu = float(as_type<half>(ld16(bu))) * (float(bu[2 + g]) * (1.0f / 255.0f));
        float4 wg0, wg1, wu0, wu1;
        sk_w8<F>(bg + 6, j, wg0, wg1);
        sk_w8<F>(bu + 6, j, wu0, wu1);
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(x_in + ulong(tk[i]) * 2304ul + uint(sbi * 256 + j));
            const float4 x0 = xp[0], x1 = xp[1];
            ag[i] = fma(effg, dot(wg0, x0) + dot(wg1, x1), ag[i]);
            au[i] = fma(effu, dot(wu0, x0) + dot(wu1, x1), au[i]);
        }
    }
}

// Generic mixed-codec MoE gate/up + SITU (Apple M1/M2 G13/G14 GlobalISel-safe: pure uint indexing, single mm<=4 loop!)
kernel void ink_moe_gu_situ(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* x_in        [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device float* h_act             [[buffer(8)]],
    device const uchar* sec4_1      [[buffer(9)]],
    device const uchar* sec4_2      [[buffer(10)]],
    device const uchar* sec4_3      [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int y = int(tgp.y);
    if (y >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = int(tgp.x) * 8 + int(sg);
    if (r >= 1152) return;

    const int e = uexp[y], off = uoff[y], n = ucnt[y];
    const InkExpertDev ed = ex[e];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    const bool is_sk_dn = (ed.dn_code != 15 && ed.dn_code != 16 && ed.dn_code != 20);
    const float inv_sq = is_sk_dn ? (1.0f / float(as_type<half>(ld16(body + 2 * r)))) : 1.0f;

    for (int t0 = 0; t0 < n; t0 += 4) {
        const int mm = min(4, n - t0);
        uint pid[4], tk[4];
        for (int i = 0; i < mm; ++i) {
            const uint p = uint(pair_list[off + t0 + i]);
            pid[i] = p;
            tk[i] = p >> 3u;
        }
        float ag[4] = {0,0,0,0};
        float au[4] = {0,0,0,0};
        if (ed.gu_code == 15)      sf4_w13_chunk_s<32>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 16) mx8_w13_chunk(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 20) gguf_w13_chunk(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 3)  sk_w13_chunk<3>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 2)  sk_w13_chunk<2>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 4)  sk_w13_chunk<4>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 5)  sk_w13_chunk<5>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 6)  sk_w13_chunk<6>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 8)  sk_w13_chunk<8>(body, r, x_in, tk, mm, int(lane), ag, au);

        for (int i = 0; i < mm; ++i) {
            const float g = simd_sum(ag[i]);
            const float u = simd_sum(au[i]);
            if (lane == 0) h_act[ulong(pid[i]) * 1152ul + uint(r)] = situ_act(g, u) * inv_sq;
        }
    }
}

template <int STEP>
static inline void sf4_w2_pair_s(
    device const uchar* body, uint dn_off, int rp, device const float* h_act,
    thread const uint* pid, int mm, int lane, thread float* a0, thread float* a1) {
    device const float* hdr = (device const float*)body;
    const float outer = hdr[2] / hdr[3];
    device const uchar* p_base = body + dn_off;
    device const uchar* s_base = p_base + 1327104;
    const int r0 = 2 * rp, r1 = 2 * rp + 1;
    device const uint2* p0 = (device const uint2*)(p_base + (size_t)r0 * 576);
    device const uchar* s0 = s_base + (size_t)r0 * 72;
    device const uint2* p1 = (device const uint2*)(p_base + (size_t)r1 * 576);
    device const uchar* s1 = s_base + (size_t)r1 * 72;

    for (int b = lane; b < 72; b += STEP) {
        const float sc0 = e4m3_to_f32(s0[b]) * outer;
        const float sc1 = e4m3_to_f32(s1[b]) * outer;
        const uint2 b0 = p0[b], b1 = p1[b];
        const float4 w00 = float4(kE2M1[b0.x & 15u], kE2M1[(b0.x >> 4) & 15u], kE2M1[(b0.x >> 8) & 15u], kE2M1[(b0.x >> 12) & 15u]);
        const float4 w01 = float4(kE2M1[(b0.x >> 16) & 15u], kE2M1[(b0.x >> 20) & 15u], kE2M1[(b0.x >> 24) & 15u], kE2M1[b0.x >> 28]);
        const float4 w02 = float4(kE2M1[b0.y & 15u], kE2M1[(b0.y >> 4) & 15u], kE2M1[(b0.y >> 8) & 15u], kE2M1[(b0.y >> 12) & 15u]);
        const float4 w03 = float4(kE2M1[(b0.y >> 16) & 15u], kE2M1[(b0.y >> 20) & 15u], kE2M1[(b0.y >> 24) & 15u], kE2M1[b0.y >> 28]);
        const float4 w10 = float4(kE2M1[b1.x & 15u], kE2M1[(b1.x >> 4) & 15u], kE2M1[(b1.x >> 8) & 15u], kE2M1[(b1.x >> 12) & 15u]);
        const float4 w11 = float4(kE2M1[(b1.x >> 16) & 15u], kE2M1[(b1.x >> 20) & 15u], kE2M1[(b1.x >> 24) & 15u], kE2M1[b1.x >> 28]);
        const float4 w12 = float4(kE2M1[b1.y & 15u], kE2M1[(b1.y >> 4) & 15u], kE2M1[(b1.y >> 8) & 15u], kE2M1[(b1.y >> 12) & 15u]);
        const float4 w13 = float4(kE2M1[(b1.y >> 16) & 15u], kE2M1[(b1.y >> 20) & 15u], kE2M1[(b1.y >> 24) & 15u], kE2M1[b1.y >> 28]);
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(h_act + ulong(pid[i]) * 1152ul + uint(b * 16));
            const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            a0[i] = fma(sc0, dot(w00, x0) + dot(w01, x1) + dot(w02, x2) + dot(w03, x3), a0[i]);
            a1[i] = fma(sc1, dot(w10, x0) + dot(w11, x1) + dot(w12, x2) + dot(w13, x3), a1[i]);
        }
    }
}

// Vectorized 64-bit uint2 MX8 w2 (144 chunks of 8: STEP=16 -> 9*16 exact iters, ZERO IDLE LANES!)
template <int STEP>
static inline void mx8_w2_pair_s(
    device const uchar* body, uint dn_off, int rp, device const float* h_act,
    thread const uint* pid, int mm, int lane, thread float* a0, thread float* a1) {
    device const uchar* w_base = body + dn_off;
    device const uchar* s_base = w_base + 2654208;
    const int r0 = 2 * rp, r1 = 2 * rp + 1;
    device const uint2* w0 = (device const uint2*)(w_base + (size_t)r0 * 1152);
    device const uchar* s0 = s_base + (size_t)r0 * 36;
    device const uint2* w1 = (device const uint2*)(w_base + (size_t)r1 * 1152);
    device const uchar* s1 = s_base + (size_t)r1 * 36;

    for (int u = lane; u < 144; u += STEP) {
        const int b = u >> 2;
        const float sc0 = ue8m0_to_f32(s0[b]), sc1 = ue8m0_to_f32(s1[b]);
        const uint2 uw = w0[u], vw = w1[u];
        const float4 u0 = float4(e4m3_to_f32(uw.x & 255u), e4m3_to_f32((uw.x >> 8) & 255u), e4m3_to_f32((uw.x >> 16) & 255u), e4m3_to_f32(uw.x >> 24));
        const float4 u1 = float4(e4m3_to_f32(uw.y & 255u), e4m3_to_f32((uw.y >> 8) & 255u), e4m3_to_f32((uw.y >> 16) & 255u), e4m3_to_f32(uw.y >> 24));
        const float4 v0 = float4(e4m3_to_f32(vw.x & 255u), e4m3_to_f32((vw.x >> 8) & 255u), e4m3_to_f32((vw.x >> 16) & 255u), e4m3_to_f32(vw.x >> 24));
        const float4 v1 = float4(e4m3_to_f32(vw.y & 255u), e4m3_to_f32((vw.y >> 8) & 255u), e4m3_to_f32((vw.y >> 16) & 255u), e4m3_to_f32(vw.y >> 24));
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(h_act + ulong(pid[i]) * 1152ul + uint(u * 8));
            const float4 x0 = xp[0], x1 = xp[1];
            a0[i] = fma(sc0, dot(u0, x0) + dot(u1, x1), a0[i]);
            a1[i] = fma(sc1, dot(v0, x0) + dot(v1, x1), a1[i]);
        }
    }
}

template <int F>
static inline void sk_w2_paired2(
    device const uchar* body, uint dn_off, int rp, device const float* h_act,
    thread const uint* pid, int mm, int lane, thread float* a0, thread float* a1) {
    constexpr int SB = sk_sb_bytes(F);
    device const uchar* pr = body + dn_off + (size_t)rp * 9ul * SB;
    for (int unit = lane; unit < 288; unit += 32) {
        const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
        device const uchar* blk = pr + (size_t)sbi * SB;
        const float eff = float(as_type<half>(ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
        float4 w0, w1;
        sk_w8<F>(blk + 6, j, w0, w1);
        const bool is_r1 = (unit >= 144);
        const int col = (is_r1 ? (unit - 144) : unit) << 3;
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(h_act + ulong(pid[i]) * 1152ul + uint(col));
            const float val = eff * (dot(w0, xp[0]) + dot(w1, xp[1]));
            if (is_r1) a1[i] += val; else a0[i] += val;
        }
    }
}

// Generic mixed-codec MoE down projection (Apple M1/M2 G13/G14 GlobalISel-safe: pure uint indexing, single mm<=4 loop!)
kernel void ink_moe_dn(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* topk_w      [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device const float* h_act       [[buffer(8)]],
    device float* y_pairs           [[buffer(9)]],
    device const uchar* sec4_1      [[buffer(10)]],
    device const uchar* sec4_2      [[buffer(11)]],
    device const uchar* sec4_3      [[buffer(12)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int yy = int(tgp.y);
    if (yy >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int rp = int(tgp.x) * 8 + int(sg);
    if (rp >= 1152) return;

    const int e = uexp[yy], off = uoff[yy], n = ucnt[yy];
    const InkExpertDev ed = ex[e];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;

    for (int t0 = 0; t0 < n; t0 += 4) {
        const int mm = min(4, n - t0);
        uint pid[4];
        for (int i = 0; i < mm; ++i) pid[i] = uint(pair_list[off + t0 + i]);
        float a0[4] = {0,0,0,0};
        float a1[4] = {0,0,0,0};
        if (ed.dn_code == 15)      sf4_w2_pair_s<32>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 16) mx8_w2_pair_s<32>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 20) gguf_w2_pair(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 2)  sk_w2_paired2<2>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 3)  sk_w2_paired2<3>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 4)  sk_w2_paired2<4>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 5)  sk_w2_paired2<5>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 6)  sk_w2_paired2<6>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 8)  sk_w2_paired2<8>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);

        for (int i = 0; i < mm; ++i) {
            const float v0 = simd_sum(a0[i]);
            const float v1 = simd_sum(a1[i]);
            if (lane == 0) {
                const float rw = topk_w[pid[i]];
                const ulong yb = ulong(pid[i]) * 2304ul + uint(2 * rp);
                y_pairs[yb]     = v0 * rw;
                y_pairs[yb + 1] = v1 * rw;
            }
        }
    }
}

kernel void ink_moe_reduce8(
    device const float* y_pairs [[buffer(0)]],
    device float* moe_out       [[buffer(1)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x);
    const int d = int(tgp.y) * 256 + int(tid);
    if (d >= 2304) return;
    float s = 0.0f;
    for (int k = 0; k < 8; ++k) s += y_pairs[((size_t)m * 8 + k) * 2304 + d];
    moe_out[(size_t)m * 2304 + d] = s;
}

// ---- Low-Register ZERO-IDLE-LANE Specialized SF4, MX8 & S2/S3/S4_K MoE Shaders ----
// Each shader checks its target codec on line 1 so mixed layers (SQ-15/SQ-10) can run ALL active
// codecs CONCURRENTLY with ZERO megakernel register spill!
// SF4 w13: 16 lanes/pair (16 pairs/TG -> grid.x = 72), 144 = 9*16 exact iters!
kernel void ink_moe_gu_sf4(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* x_in        [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device float* h_act             [[buffer(8)]],
    device const uchar* sec4_1      [[buffer(9)]],
    device const uchar* sec4_2      [[buffer(10)]],
    device const uchar* sec4_3      [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int y = int(tgp.y);
    if (y >= int(meta[0])) return;
    const int e = uexp[y];
    const InkExpertDev ed = ex[e];
    if (ed.gu_code != 15) return;
    const uint hid = tid >> 4, lane16 = tid & 15u;
    const int r = int(tgp.x) * 16 + int(hid);
    if (r >= 1152) return;
    const int off = uoff[y], n = ucnt[y];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    const bool is_sk_dn = (ed.dn_code != 15 && ed.dn_code != 16 && ed.dn_code != 20);
    const float inv_sq = is_sk_dn ? (1.0f / float(as_type<half>(ld16(body + 2 * r)))) : 1.0f;
    if (n == 1) {
        const uint pid0 = uint(pair_list[off]), tk0 = pid0 >> 3u;
        float ag0 = 0.0f, au0 = 0.0f;
        sf4_w13_chunk_s<16>(body, r, x_in, &tk0, 1, int(lane16), &ag0, &au0);
        const float g = moe_sum16(ag0), u = moe_sum16(au0);
        if (lane16 == 0) h_act[ulong(pid0) * 1152ul + uint(r)] = situ_act(g, u) * inv_sq;
        return;
    }
    for (int t0 = 0; t0 < n; t0 += 4) {
        const int mm = min(4, n - t0);
        uint pid[4], tk[4];
        for (int i = 0; i < mm; ++i) { pid[i] = uint(pair_list[off + t0 + i]); tk[i] = pid[i] >> 3u; }
        float ag[4] = {0,0,0,0}, au[4] = {0,0,0,0};
        sf4_w13_chunk_s<16>(body, r, x_in, tk, mm, int(lane16), ag, au);
        for (int i = 0; i < mm; ++i) {
            const float g = moe_sum16(ag[i]), u = moe_sum16(au[i]);
            if (lane16 == 0) h_act[ulong(pid[i]) * 1152ul + uint(r)] = situ_act(g, u) * inv_sq;
        }
    }
}

// SF4 w2: 8 lanes/pair (32 pairs/TG -> grid.x = 36), 72 = 9*8 exact iters!
kernel void ink_moe_dn_sf4(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* topk_w      [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device const float* h_act       [[buffer(8)]],
    device float* y_pairs           [[buffer(9)]],
    device const uchar* sec4_1      [[buffer(10)]],
    device const uchar* sec4_2      [[buffer(11)]],
    device const uchar* sec4_3      [[buffer(12)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int yy = int(tgp.y);
    if (yy >= int(meta[0])) return;
    const int e = uexp[yy];
    const InkExpertDev ed = ex[e];
    if (ed.dn_code != 15) return;
    const uint qid = tid >> 3, lane8 = tid & 7u;
    const int rp = int(tgp.x) * 32 + int(qid);
    if (rp >= 1152) return;
    const int off = uoff[yy], n = ucnt[yy];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    if (n == 1) {
        const uint pid0 = uint(pair_list[off]);
        float a0 = 0.0f, a1 = 0.0f;
        sf4_w2_pair_s<8>(body, ed.dn_off, rp, h_act, &pid0, 1, int(lane8), &a0, &a1);
        const float v0 = moe_sum8(a0), v1 = moe_sum8(a1);
        if (lane8 == 0) {
            const float rw = topk_w[pid0];
            y_pairs[ulong(pid0) * 2304ul + uint(2 * rp)]     = v0 * rw;
            y_pairs[ulong(pid0) * 2304ul + uint(2 * rp + 1)] = v1 * rw;
        }
        return;
    }
    for (int t0 = 0; t0 < n; t0 += 4) {
        const int mm = min(4, n - t0);
        uint pid[4];
        for (int i = 0; i < mm; ++i) pid[i] = uint(pair_list[off + t0 + i]);
        float a0[4] = {0,0,0,0}, a1[4] = {0,0,0,0};
        sf4_w2_pair_s<8>(body, ed.dn_off, rp, h_act, pid, mm, int(lane8), a0, a1);
        for (int i = 0; i < mm; ++i) {
            const float v0 = moe_sum8(a0[i]), v1 = moe_sum8(a1[i]);
            if (lane8 == 0) {
                const float rw = topk_w[pid[i]];
                y_pairs[ulong(pid[i]) * 2304ul + uint(2 * rp)]     = v0 * rw;
                y_pairs[ulong(pid[i]) * 2304ul + uint(2 * rp + 1)] = v1 * rw;
            }
        }
    }
}

// MX8 w13: 32 lanes/pair (8 pairs/TG -> grid.x = 144), 288 = 9*32 exact iters!
kernel void ink_moe_gu_mx8(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* x_in        [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device float* h_act             [[buffer(8)]],
    device const uchar* sec4_1      [[buffer(9)]],
    device const uchar* sec4_2      [[buffer(10)]],
    device const uchar* sec4_3      [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int y = int(tgp.y);
    if (y >= int(meta[0])) return;
    const int e = uexp[y];
    const InkExpertDev ed = ex[e];
    if (ed.gu_code != 16) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = int(tgp.x) * 8 + int(sg);
    if (r >= 1152) return;
    const int off = uoff[y], n = ucnt[y];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    if (n == 1) {
        const uint pid0 = uint(pair_list[off]), tk0 = pid0 >> 3u;
        float ag0 = 0.0f, au0 = 0.0f;
        mx8_w13_chunk(body, r, x_in, &tk0, 1, int(lane), &ag0, &au0);
        const float g = simd_sum(ag0), u = simd_sum(au0);
        if (lane == 0) h_act[ulong(pid0) * 1152ul + uint(r)] = situ_act(g, u);
        return;
    }
    for (int t0 = 0; t0 < n; t0 += 8) {
        const int mm = min(8, n - t0);
        uint pid[8], tk[8];
        for (int i = 0; i < mm; ++i) { pid[i] = uint(pair_list[off + t0 + i]); tk[i] = pid[i] >> 3u; }
        float ag[8] = {0,0,0,0,0,0,0,0}, au[8] = {0,0,0,0,0,0,0,0};
        mx8_w13_chunk(body, r, x_in, tk, mm, int(lane), ag, au);
        for (int i = 0; i < mm; ++i) {
            const float g = simd_sum(ag[i]), u = simd_sum(au[i]);
            if (lane == 0) h_act[ulong(pid[i]) * 1152ul + uint(r)] = situ_act(g, u);
        }
    }
}

// MX8 w2: 16 lanes/pair (16 pairs/TG -> grid.x = 72), 144 = 9*16 exact iters!
kernel void ink_moe_dn_mx8(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* topk_w      [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device const float* h_act       [[buffer(8)]],
    device float* y_pairs           [[buffer(9)]],
    device const uchar* sec4_1      [[buffer(10)]],
    device const uchar* sec4_2      [[buffer(11)]],
    device const uchar* sec4_3      [[buffer(12)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int yy = int(tgp.y);
    if (yy >= int(meta[0])) return;
    const int e = uexp[yy];
    const InkExpertDev ed = ex[e];
    if (ed.dn_code != 16) return;
    const uint hid = tid >> 4, lane16 = tid & 15u;
    const int rp = int(tgp.x) * 16 + int(hid);
    if (rp >= 1152) return;
    const int off = uoff[yy], n = ucnt[yy];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    if (n == 1) {
        const uint pid0 = uint(pair_list[off]);
        float a0 = 0.0f, a1 = 0.0f;
        mx8_w2_pair_s<16>(body, ed.dn_off, rp, h_act, &pid0, 1, int(lane16), &a0, &a1);
        const float v0 = moe_sum16(a0), v1 = moe_sum16(a1);
        if (lane16 == 0) {
            const float rw = topk_w[pid0];
            y_pairs[ulong(pid0) * 2304ul + uint(2 * rp)]     = v0 * rw;
            y_pairs[ulong(pid0) * 2304ul + uint(2 * rp + 1)] = v1 * rw;
        }
        return;
    }
    for (int t0 = 0; t0 < n; t0 += 8) {
        const int mm = min(8, n - t0);
        uint pid[8];
        for (int i = 0; i < mm; ++i) pid[i] = uint(pair_list[off + t0 + i]);
        float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0};
        mx8_w2_pair_s<16>(body, ed.dn_off, rp, h_act, pid, mm, int(lane16), a0, a1);
        for (int i = 0; i < mm; ++i) {
            const float v0 = moe_sum16(a0[i]), v1 = moe_sum16(a1[i]);
            if (lane16 == 0) {
                const float rw = topk_w[pid[i]];
                y_pairs[ulong(pid[i]) * 2304ul + uint(2 * rp)]     = v0 * rw;
                y_pairs[ulong(pid[i]) * 2304ul + uint(2 * rp + 1)] = v1 * rw;
            }
        }
    }
}

// Low-register specialized S2_K / S3_K / S4_K / S5_K MoE shaders for SQ-10 & SQ-15!
template <int F>
kernel void ink_moe_gu_sk_t(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* x_in        [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device float* h_act             [[buffer(8)]],
    device const uchar* sec4_1      [[buffer(9)]],
    device const uchar* sec4_2      [[buffer(10)]],
    device const uchar* sec4_3      [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int y = int(tgp.y);
    if (y >= int(meta[0])) return;
    const int e = uexp[y];
    const InkExpertDev ed = ex[e];
    if (ed.gu_code != F) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = int(tgp.x) * 8 + int(sg);
    if (r >= 1152) return;
    const int off = uoff[y], n = ucnt[y];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    const bool is_sk_dn = (ed.dn_code != 15 && ed.dn_code != 16 && ed.dn_code != 20);
    const float inv_sq = is_sk_dn ? (1.0f / float(as_type<half>(ld16(body + 2 * r)))) : 1.0f;
    if (n == 1) {
        const uint pid0 = uint(pair_list[off]), tk0 = pid0 >> 3u;
        float ag0 = 0.0f, au0 = 0.0f;
        sk_w13_chunk<F>(body, r, x_in, &tk0, 1, int(lane), &ag0, &au0);
        const float g = simd_sum(ag0), u = simd_sum(au0);
        if (lane == 0) h_act[ulong(pid0) * 1152ul + uint(r)] = situ_act(g, u) * inv_sq;
        return;
    }
    for (int t0 = 0; t0 < n; t0 += 4) {
        const int mm = min(4, n - t0);
        uint pid[4], tk[4];
        for (int i = 0; i < mm; ++i) { pid[i] = uint(pair_list[off + t0 + i]); tk[i] = pid[i] >> 3u; }
        float ag[4] = {0,0,0,0}, au[4] = {0,0,0,0};
        sk_w13_chunk<F>(body, r, x_in, tk, mm, int(lane), ag, au);
        for (int i = 0; i < mm; ++i) {
            const float g = simd_sum(ag[i]), u = simd_sum(au[i]);
            if (lane == 0) h_act[ulong(pid[i]) * 1152ul + uint(r)] = situ_act(g, u) * inv_sq;
        }
    }
}

template <int F>
kernel void ink_moe_dn_sk_t(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* topk_w      [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device const float* h_act       [[buffer(8)]],
    device float* y_pairs           [[buffer(9)]],
    device const uchar* sec4_1      [[buffer(10)]],
    device const uchar* sec4_2      [[buffer(11)]],
    device const uchar* sec4_3      [[buffer(12)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int yy = int(tgp.y);
    if (yy >= int(meta[0])) return;
    const int e = uexp[yy];
    const InkExpertDev ed = ex[e];
    if (ed.dn_code != F) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int rp = int(tgp.x) * 8 + int(sg);
    if (rp >= 1152) return;
    const int off = uoff[yy], n = ucnt[yy];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    if (n == 1) {
        const uint pid0 = uint(pair_list[off]);
        float a0 = 0.0f, a1 = 0.0f;
        sk_w2_paired2<F>(body, ed.dn_off, rp, h_act, &pid0, 1, int(lane), &a0, &a1);
        const float v0 = simd_sum(a0), v1 = simd_sum(a1);
        if (lane == 0) {
            const float rw = topk_w[pid0];
            y_pairs[ulong(pid0) * 2304ul + uint(2 * rp)]     = v0 * rw;
            y_pairs[ulong(pid0) * 2304ul + uint(2 * rp + 1)] = v1 * rw;
        }
        return;
    }
    for (int t0 = 0; t0 < n; t0 += 4) {
        const int mm = min(4, n - t0);
        uint pid[4];
        for (int i = 0; i < mm; ++i) pid[i] = uint(pair_list[off + t0 + i]);
        float a0[4] = {0,0,0,0}, a1[4] = {0,0,0,0};
        sk_w2_paired2<F>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        for (int i = 0; i < mm; ++i) {
            const float v0 = simd_sum(a0[i]), v1 = simd_sum(a1[i]);
            if (lane == 0) {
                const float rw = topk_w[pid[i]];
                y_pairs[ulong(pid[i]) * 2304ul + uint(2 * rp)]     = v0 * rw;
                y_pairs[ulong(pid[i]) * 2304ul + uint(2 * rp + 1)] = v1 * rw;
            }
        }
    }
}

template [[host_name("ink_moe_gu_sk2")]] kernel decltype(ink_moe_gu_sk_t<2>) ink_moe_gu_sk_t<2>;
template [[host_name("ink_moe_dn_sk2")]] kernel decltype(ink_moe_dn_sk_t<2>) ink_moe_dn_sk_t<2>;
template [[host_name("ink_moe_gu_sk3")]] kernel decltype(ink_moe_gu_sk_t<3>) ink_moe_gu_sk_t<3>;
template [[host_name("ink_moe_dn_sk3")]] kernel decltype(ink_moe_dn_sk_t<3>) ink_moe_dn_sk_t<3>;
template [[host_name("ink_moe_gu_sk4")]] kernel decltype(ink_moe_gu_sk_t<4>) ink_moe_gu_sk_t<4>;
template [[host_name("ink_moe_dn_sk4")]] kernel decltype(ink_moe_dn_sk_t<4>) ink_moe_dn_sk_t<4>;
template [[host_name("ink_moe_gu_sk5")]] kernel decltype(ink_moe_gu_sk_t<5>) ink_moe_gu_sk_t<5>;
template [[host_name("ink_moe_dn_sk5")]] kernel decltype(ink_moe_dn_sk_t<5>) ink_moe_dn_sk_t<5>;

// ---- Ultra-Lean SQ-10 Mixed {S2_K, S3_K, SF4} MoE Shaders (minimal registers -> max M4 occupancy!) ----
// w13: 8 rows/TG (grid.x=144), 32 lanes/row -> LUT-accelerated S3_K/S2_K + SF4 fallback!
kernel void ink_moe_gu_sk23(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* x_in        [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device float* h_act             [[buffer(8)]],
    device const uchar* sec4_1      [[buffer(9)]],
    device const uchar* sec4_2      [[buffer(10)]],
    device const uchar* sec4_3      [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int y = int(tgp.y);
    if (y >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = int(tgp.x) * 8 + int(sg);
    if (r >= 1152) return;
    const int e = uexp[y], off = uoff[y], n = ucnt[y];
    const InkExpertDev ed = ex[e];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    const bool is_sk_dn = (ed.dn_code != 15 && ed.dn_code != 16 && ed.dn_code != 20);
    const float inv_sq = is_sk_dn ? (1.0f / float(as_type<half>(ld16(body + 2 * r)))) : 1.0f;

    if (n == 1) {
        const uint pid0 = uint(pair_list[off]), tk0 = pid0 >> 3u;
        float ag0 = 0.0f, au0 = 0.0f;
        if (ed.gu_code == 3)      sk_w13_chunk<3>(body, r, x_in, &tk0, 1, int(lane), &ag0, &au0);
        else if (ed.gu_code == 2) sk_w13_chunk<2>(body, r, x_in, &tk0, 1, int(lane), &ag0, &au0);
        else                      sf4_w13_chunk_s<32>(body, r, x_in, &tk0, 1, int(lane), &ag0, &au0);
        const float g = simd_sum(ag0), u = simd_sum(au0);
        if (lane == 0) h_act[ulong(pid0) * 1152ul + uint(r)] = situ_act(g, u) * inv_sq;
        return;
    }
    for (int t0 = 0; t0 < n; t0 += 4) {
        const int mm = min(4, n - t0);
        uint pid[4], tk[4];
        for (int i = 0; i < mm; ++i) { pid[i] = uint(pair_list[off + t0 + i]); tk[i] = pid[i] >> 3u; }
        float ag[4] = {0,0,0,0}, au[4] = {0,0,0,0};
        if (ed.gu_code == 3)      sk_w13_chunk<3>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 2) sk_w13_chunk<2>(body, r, x_in, tk, mm, int(lane), ag, au);
        else                      sf4_w13_chunk_s<32>(body, r, x_in, tk, mm, int(lane), ag, au);
        for (int i = 0; i < mm; ++i) {
            const float g = simd_sum(ag[i]), u = simd_sum(au[i]);
            if (lane == 0) h_act[ulong(pid[i]) * 1152ul + uint(r)] = situ_act(g, u) * inv_sq;
        }
    }
}

// Superblock-coalesced 2-pair (4 rows)/SG paired2 S_K w2 decoder (ALL 32 lanes hit 1 contiguous SB -> 1.37x faster on M4!)
template <int F>
static inline void sk_w2_2pair_coalesced(
    device const uchar* body, uint dn_off, int rp0, device const float* h_act,
    thread const uint* pid, int mm, int lane,
    thread float* a0, thread float* a1, thread float* b0, thread float* b1) {
    constexpr int SB = sk_sb_bytes(F);
    const size_t pb = 9ul * (size_t)SB;
    device const uchar* pr0 = body + dn_off + (size_t)rp0 * pb;
    device const uchar* pr1 = pr0 + pb;
    const int g = lane >> 3, j = (g << 6) + ((lane & 7) << 3);

    // Superblocks 0..3 -> 100% row 0 (32-lane contiguous!)
    for (int sbi = 0; sbi < 4; ++sbi) {
        device const uchar* p0 = pr0 + (size_t)sbi * SB;
        device const uchar* p1 = pr1 + (size_t)sbi * SB;
        const float ef0 = float(as_type<half>(ld16(p0))) * (float(p0[2 + g]) * (1.0f / 255.0f));
        const float ef1 = float(as_type<half>(ld16(p1))) * (float(p1[2 + g]) * (1.0f / 255.0f));
        float4 w00, w01, w10, w11;
        sk_w8<F>(p0 + 6, j, w00, w01);
        sk_w8<F>(p1 + 6, j, w10, w11);
        const int col = sbi * 256 + j;
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(h_act + ulong(pid[i]) * 1152ul + uint(col));
            const float4 x0 = xp[0], x1 = xp[1];
            a0[i] = fma(ef0, dot(w00, x0) + dot(w01, x1), a0[i]);
            b0[i] = fma(ef1, dot(w10, x0) + dot(w11, x1), b0[i]);
        }
    }
    // Superblock 4 -> split at lane 16 (lanes 0..15 -> row 0 [1024..1151], lanes 16..31 -> row 1 [0..127])
    {
        device const uchar* p0 = pr0 + 4ul * SB;
        device const uchar* p1 = pr1 + 4ul * SB;
        const float ef0 = float(as_type<half>(ld16(p0))) * (float(p0[2 + g]) * (1.0f / 255.0f));
        const float ef1 = float(as_type<half>(ld16(p1))) * (float(p1[2 + g]) * (1.0f / 255.0f));
        float4 w00, w01, w10, w11;
        sk_w8<F>(p0 + 6, j, w00, w01);
        sk_w8<F>(p1 + 6, j, w10, w11);
        const bool is_r1 = (lane >= 16);
        const int col = is_r1 ? ((lane - 16) << 3) : (1024 + (lane << 3));
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(h_act + ulong(pid[i]) * 1152ul + uint(col));
            const float4 x0 = xp[0], x1 = xp[1];
            const float v0 = ef0 * (dot(w00, x0) + dot(w01, x1));
            const float v1 = ef1 * (dot(w10, x0) + dot(w11, x1));
            if (is_r1) { a1[i] += v0; b1[i] += v1; }
            else       { a0[i] += v0; b0[i] += v1; }
        }
    }
    // Superblocks 5..8 -> 100% row 1 (32-lane contiguous!)
    for (int sbi = 5; sbi < 9; ++sbi) {
        device const uchar* p0 = pr0 + (size_t)sbi * SB;
        device const uchar* p1 = pr1 + (size_t)sbi * SB;
        const float ef0 = float(as_type<half>(ld16(p0))) * (float(p0[2 + g]) * (1.0f / 255.0f));
        const float ef1 = float(as_type<half>(ld16(p1))) * (float(p1[2 + g]) * (1.0f / 255.0f));
        float4 w00, w01, w10, w11;
        sk_w8<F>(p0 + 6, j, w00, w01);
        sk_w8<F>(p1 + 6, j, w10, w11);
        const int col = (sbi - 5) * 256 + 128 + j;
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(h_act + ulong(pid[i]) * 1152ul + uint(col));
            const float4 x0 = xp[0], x1 = xp[1];
            a1[i] = fma(ef0, dot(w00, x0) + dot(w01, x1), a1[i]);
            b1[i] = fma(ef1, dot(w10, x0) + dot(w11, x1), b1[i]);
        }
    }
}

// w2: 16 pairs (32 rows)/TG (grid.x=72, ind_off=5*4) -> 2 pairs/SG, 100% 32-lane superblock-coalesced S2_K/S3_K + SF4!
kernel void ink_moe_dn_sk23(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* topk_w      [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device const float* h_act       [[buffer(8)]],
    device float* y_pairs           [[buffer(9)]],
    device const uchar* sec4_1      [[buffer(10)]],
    device const uchar* sec4_2      [[buffer(11)]],
    device const uchar* sec4_3      [[buffer(12)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int yy = int(tgp.y);
    if (yy >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int rp0 = int(tgp.x) * 16 + int(sg) * 2;
    if (rp0 >= 1152) return;

    const int e = uexp[yy], off = uoff[yy], n = ucnt[yy];
    const InkExpertDev ed = ex[e];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;

    if (ed.dn_code == 2 || ed.dn_code == 3) {
        if (n == 1) {
            const uint pid0 = uint(pair_list[off]);
            float a0 = 0.0f, a1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
            if (ed.dn_code == 2) sk_w2_2pair_coalesced<2>(body, ed.dn_off, rp0, h_act, &pid0, 1, int(lane), &a0, &a1, &b0, &b1);
            else                 sk_w2_2pair_coalesced<3>(body, ed.dn_off, rp0, h_act, &pid0, 1, int(lane), &a0, &a1, &b0, &b1);
            const float va0 = simd_sum(a0), va1 = simd_sum(a1), vb0 = simd_sum(b0), vb1 = simd_sum(b1);
            if (lane == 0) {
                const float rw = topk_w[pid0];
                device float* dst = y_pairs + ulong(pid0) * 2304ul + uint(2 * rp0);
                dst[0] = va0 * rw; dst[1] = va1 * rw; dst[2] = vb0 * rw; dst[3] = vb1 * rw;
            }
            return;
        }
        for (int t0 = 0; t0 < n; t0 += 2) {
            const int mm = min(2, n - t0);
            uint pid[2];
            for (int i = 0; i < mm; ++i) pid[i] = uint(pair_list[off + t0 + i]);
            float a0[2] = {0,0}, a1[2] = {0,0};
            float b0[2] = {0,0}, b1[2] = {0,0};
            if (ed.dn_code == 2) sk_w2_2pair_coalesced<2>(body, ed.dn_off, rp0, h_act, pid, mm, int(lane), a0, a1, b0, b1);
            else                 sk_w2_2pair_coalesced<3>(body, ed.dn_off, rp0, h_act, pid, mm, int(lane), a0, a1, b0, b1);
            for (int i = 0; i < mm; ++i) {
                const float va0 = simd_sum(a0[i]), va1 = simd_sum(a1[i]), vb0 = simd_sum(b0[i]), vb1 = simd_sum(b1[i]);
                if (lane == 0) {
                    const float rw = topk_w[pid[i]];
                    device float* dst = y_pairs + ulong(pid[i]) * 2304ul + uint(2 * rp0);
                    dst[0] = va0 * rw; dst[1] = va1 * rw; dst[2] = vb0 * rw; dst[3] = vb1 * rw;
                }
            }
        }
    } else {
        const uint hid = tid >> 4, lane16 = tid & 15u;
        const int rp = int(tgp.x) * 16 + int(hid);
        for (int t0 = 0; t0 < n; t0 += 4) {
            const int mm = min(4, n - t0);
            uint pid[4];
            for (int i = 0; i < mm; ++i) pid[i] = uint(pair_list[off + t0 + i]);
            float a0[4] = {0,0,0,0}, a1[4] = {0,0,0,0};
            sf4_w2_pair_s<16>(body, ed.dn_off, rp, h_act, pid, mm, int(lane16), a0, a1);
            for (int i = 0; i < mm; ++i) {
                const float v0 = moe_sum16(a0[i]), v1 = moe_sum16(a1[i]);
                if (lane16 == 0) {
                    const float rw = topk_w[pid[i]];
                    y_pairs[ulong(pid[i]) * 2304ul + uint(2 * rp)]     = v0 * rw;
                    y_pairs[ulong(pid[i]) * 2304ul + uint(2 * rp + 1)] = v1 * rw;
                }
            }
        }
    }
}

// ============================================================================
// 5. Grouped-Expert simdgroup_matrix 8x8 MoE GEMM (BM=16, BN=64, BK=64)
//    Indirect grid: (36, n_uniq, 1) -> 36 * 64 = 2304 rows, 256 threads (8 SGs)
//    - BK=64: 1 full 16-elem SF4 block (uint2 + 1 scale) per thread -> 2x cache-line BW, 2x fewer barriers!
//    - BM=16: zero wasted M-rows for P=256 (avg n=16), half the accumulators (2 vs 4) -> 2x occupancy!
//    - Epilogue fuses SITU(el[0]=gate, el[1]=up) directly in registers!
// ============================================================================

static inline void decode_sb_sf4_16(device const uchar* p_base, device const uchar* s_base, float outer,
                                    int row_bytes, int s_row_bytes, int gn, int k0, int b_idx, threadgroup half* dst) {
    const int blk = (k0 >> 4) + b_idx;
    const float sc = e4m3_to_f32(s_base[(size_t)gn * s_row_bytes + blk]) * outer;
    const uint2 w8 = *((device const uint2*)(p_base + (size_t)gn * row_bytes + blk * 8));
    *((threadgroup half4*)(dst + 0))  = half4(kE2M1[w8.x & 15u] * sc, kE2M1[(w8.x >> 4) & 15u] * sc,
                                              kE2M1[(w8.x >> 8) & 15u] * sc, kE2M1[(w8.x >> 12) & 15u] * sc);
    *((threadgroup half4*)(dst + 4))  = half4(kE2M1[(w8.x >> 16) & 15u] * sc, kE2M1[(w8.x >> 20) & 15u] * sc,
                                              kE2M1[(w8.x >> 24) & 15u] * sc, kE2M1[w8.x >> 28] * sc);
    *((threadgroup half4*)(dst + 8))  = half4(kE2M1[w8.y & 15u] * sc, kE2M1[(w8.y >> 4) & 15u] * sc,
                                              kE2M1[(w8.y >> 8) & 15u] * sc, kE2M1[(w8.y >> 12) & 15u] * sc);
    *((threadgroup half4*)(dst + 12)) = half4(kE2M1[(w8.y >> 16) & 15u] * sc, kE2M1[(w8.y >> 20) & 15u] * sc,
                                              kE2M1[(w8.y >> 24) & 15u] * sc, kE2M1[w8.y >> 28] * sc);
}

static inline void decode_sb_mx8_16(device const uchar* w_base, device const uchar* s_base,
                                    int row_bytes, int s_row_bytes, int gn, int k0, int b_idx, threadgroup half* dst) {
    const int col = k0 + b_idx * 16;
    const float sc = ue8m0_to_f32(s_base[(size_t)gn * s_row_bytes + (col >> 5)]);
    const uint4 ww = *((device const uint4*)(w_base + (size_t)gn * row_bytes + col));
    *((threadgroup half4*)(dst + 0))  = half4(e4m3_to_f32(ww.x & 255u) * sc, e4m3_to_f32((ww.x >> 8) & 255u) * sc,
                                              e4m3_to_f32((ww.x >> 16) & 255u) * sc, e4m3_to_f32(ww.x >> 24) * sc);
    *((threadgroup half4*)(dst + 4))  = half4(e4m3_to_f32(ww.y & 255u) * sc, e4m3_to_f32((ww.y >> 8) & 255u) * sc,
                                              e4m3_to_f32((ww.y >> 16) & 255u) * sc, e4m3_to_f32(ww.y >> 24) * sc);
    *((threadgroup half4*)(dst + 8))  = half4(e4m3_to_f32(ww.z & 255u) * sc, e4m3_to_f32((ww.z >> 8) & 255u) * sc,
                                              e4m3_to_f32((ww.z >> 16) & 255u) * sc, e4m3_to_f32(ww.z >> 24) * sc);
    *((threadgroup half4*)(dst + 12)) = half4(e4m3_to_f32(ww.w & 255u) * sc, e4m3_to_f32((ww.w >> 8) & 255u) * sc,
                                              e4m3_to_f32((ww.w >> 16) & 255u) * sc, e4m3_to_f32(ww.w >> 24) * sc);
}

static inline void decode_sb_sk_16(device const uchar* row_ptr, int code, int col, threadgroup half* dst) {
    const int SB = sk_sb_bytes(code);
    const int sbi = col >> 8, j = col & 255, g = j >> 6;
    device const uchar* blk = row_ptr + (size_t)sbi * SB;
    const float eff = float(as_type<half>(ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
    float4 w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;
    if (code == 3)      { sk_w8<3>(blk + 6, j, w0, w1); sk_w8<3>(blk + 6, j + 8, w2, w3); }
    else if (code == 4) { sk_w8<4>(blk + 6, j, w0, w1); sk_w8<4>(blk + 6, j + 8, w2, w3); }
    else if (code == 2) { sk_w8<2>(blk + 6, j, w0, w1); sk_w8<2>(blk + 6, j + 8, w2, w3); }
    else if (code == 5) { sk_w8<5>(blk + 6, j, w0, w1); sk_w8<5>(blk + 6, j + 8, w2, w3); }
    else if (code == 6) { sk_w8<6>(blk + 6, j, w0, w1); sk_w8<6>(blk + 6, j + 8, w2, w3); }
    else if (code == 8) { sk_w8<8>(blk + 6, j, w0, w1); sk_w8<8>(blk + 6, j + 8, w2, w3); }
    *((threadgroup half4*)(dst + 0))  = half4(w0 * eff);
    *((threadgroup half4*)(dst + 4))  = half4(w1 * eff);
    *((threadgroup half4*)(dst + 8))  = half4(w2 * eff);
    *((threadgroup half4*)(dst + 12)) = half4(w3 * eff);
}

// Flattened Parallel (expert, m_tile) simdgroup_matrix w13 + fused SITU (SF4 specialized, BM=32 BN=64 BK=64, 12 KB SLM!)
kernel void ink_moe_gu_mm_sf4(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* x_in        [[buffer(2)]],
    device const uint2* tiles       [[buffer(3)]],
    device const int* uoff_unused   [[buffer(4)]],
    device const int* ucnt_unused   [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device float* h_act             [[buffer(8)]],
    device const uchar* sec4_1      [[buffer(9)]],
    device const uchar* sec4_2      [[buffer(10)]],
    device const uchar* sec4_3      [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 64];
    threadgroup half sB[64 * 64];
    threadgroup int sh_pid[32];
    threadgroup int sh_tk[32];
    const int y = int(tgp.y);
    if (y >= int(meta[9])) return;
    const uint2 td = tiles[y];
    const int e = int(td.x >> 16), mm = int(td.x & 0xFFFFu), off = int(td.y);
    const int n0 = int(tgp.x) * 64;
    const InkExpertDev ed = ex[e];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    device const float* hdr = (device const float*)body;
    const float outer = hdr[0] / hdr[1];
    device const uchar* p_base = body + 16;
    device const uchar* s_base = p_base + 2654208;
    const int sgm = int(sg >> 2), sgn = int(sg & 3u);
    const int br = int(tid >> 2), bi = int(tid & 3u);
    const int ar = int(tid >> 3), ac = int((tid & 7u) * 8u);
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;

    if (tid < 32) {
        const int p = (int(tid) < mm) ? pair_list[off + int(tid)] : 0;
        sh_pid[tid] = p; sh_tk[tid] = p >> 3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < 2304; k0 += 64) {
        if (ar < mm) {
            device const float4* xp = (device const float4*)(x_in + (size_t)sh_tk[ar] * 2304 + k0 + ac);
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(xp[0]);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(xp[1]);
        } else if (ar < 16 || mm > 16) {
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(0.0h);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(0.0h);
        }
        decode_sb_sf4_16(p_base, s_base, outer, 1152, 144, n0 + br, k0, bi, sB + br * 64 + bi * 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (mm > 16) {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, a1, b0, b1;
                simdgroup_load(a0, sA + (sgm * 16 + 0) * 64 + kk * 8, 64);
                simdgroup_load(a1, sA + (sgm * 16 + 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
                simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
                simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
            }
        } else {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, b0, b1;
                simdgroup_load(a0, sA + (sgm * 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int gn0 = n0 + sgn * 16 + int(fn);
    const int lm0 = (mm > 16 ? (sgm * 16) : (sgm * 8)) + int(fm);
    const int lm1 = sgm * 16 + 8 + int(fm);
    if (lm0 < mm) {
        const int pid_m = sh_pid[lm0];
        h_act[(size_t)pid_m * 1152 + (gn0 >> 1)]       = situ_act(acc00.thread_elements()[0], acc00.thread_elements()[1]);
        h_act[(size_t)pid_m * 1152 + ((gn0 + 8) >> 1)] = situ_act(acc01.thread_elements()[0], acc01.thread_elements()[1]);
    }
    if (mm > 16 && lm1 < mm) {
        const int pid_m = sh_pid[lm1];
        h_act[(size_t)pid_m * 1152 + (gn0 >> 1)]       = situ_act(acc10.thread_elements()[0], acc10.thread_elements()[1]);
        h_act[(size_t)pid_m * 1152 + ((gn0 + 8) >> 1)] = situ_act(acc11.thread_elements()[0], acc11.thread_elements()[1]);
    }
}

kernel void ink_moe_dn_mm_sf4(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* topk_w      [[buffer(2)]],
    device const uint2* tiles       [[buffer(3)]],
    device const int* uoff_unused   [[buffer(4)]],
    device const int* ucnt_unused   [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device const float* h_act       [[buffer(8)]],
    device float* y_pairs           [[buffer(9)]],
    device const uchar* sec4_1      [[buffer(10)]],
    device const uchar* sec4_2      [[buffer(11)]],
    device const uchar* sec4_3      [[buffer(12)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 64];
    threadgroup half sB[64 * 64];
    threadgroup int sh_pid[32];
    threadgroup float sh_rw[32];
    const int yy = int(tgp.y);
    if (yy >= int(meta[9])) return;
    const uint2 td = tiles[yy];
    const int e = int(td.x >> 16), mm = int(td.x & 0xFFFFu), off = int(td.y);
    const int n0 = int(tgp.x) * 64;
    const InkExpertDev ed = ex[e];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    device const float* hdr = (device const float*)body;
    const float outer = hdr[2] / hdr[3];
    device const uchar* p_base = body + ed.dn_off;
    device const uchar* s_base = p_base + 1327104;
    const int sgm = int(sg >> 2), sgn = int(sg & 3u);
    const int br = int(tid >> 2), bi = int(tid & 3u);
    const int ar = int(tid >> 3), ac = int((tid & 7u) * 8u);
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;

    if (tid < 32) {
        const int p = (int(tid) < mm) ? pair_list[off + int(tid)] : 0;
        sh_pid[tid] = p; sh_rw[tid] = (int(tid) < mm) ? topk_w[p] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < 1152; k0 += 64) {
        if (ar < mm) {
            device const float4* xp = (device const float4*)(h_act + (size_t)sh_pid[ar] * 1152 + k0 + ac);
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(xp[0]);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(xp[1]);
        } else if (ar < 16 || mm > 16) {
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(0.0h);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(0.0h);
        }
        decode_sb_sf4_16(p_base, s_base, outer, 576, 72, n0 + br, k0, bi, sB + br * 64 + bi * 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (mm > 16) {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, a1, b0, b1;
                simdgroup_load(a0, sA + (sgm * 16 + 0) * 64 + kk * 8, 64);
                simdgroup_load(a1, sA + (sgm * 16 + 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
                simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
                simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
            }
        } else {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, b0, b1;
                simdgroup_load(a0, sA + (sgm * 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int gn0 = n0 + sgn * 16 + int(fn);
    const int lm0 = (mm > 16 ? (sgm * 16) : (sgm * 8)) + int(fm);
    const int lm1 = sgm * 16 + 8 + int(fm);
    if (lm0 < mm) {
        const int pid_m = sh_pid[lm0]; const float rw = sh_rw[lm0];
        y_pairs[(size_t)pid_m * 2304 + gn0]     = acc00.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 1] = acc00.thread_elements()[1] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 8] = acc01.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 9] = acc01.thread_elements()[1] * rw;
    }
    if (mm > 16 && lm1 < mm) {
        const int pid_m = sh_pid[lm1]; const float rw = sh_rw[lm1];
        y_pairs[(size_t)pid_m * 2304 + gn0]     = acc10.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 1] = acc10.thread_elements()[1] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 8] = acc11.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 9] = acc11.thread_elements()[1] * rw;
    }
}

// Flattened Parallel (expert, m_tile) simdgroup_matrix w13 + fused SITU (MX8 specialized, BM=32 BN=64 BK=64!)
kernel void ink_moe_gu_mm_mx8(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* x_in        [[buffer(2)]],
    device const uint2* tiles       [[buffer(3)]],
    device const int* uoff_unused   [[buffer(4)]],
    device const int* ucnt_unused   [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device float* h_act             [[buffer(8)]],
    device const uchar* sec4_1      [[buffer(9)]],
    device const uchar* sec4_2      [[buffer(10)]],
    device const uchar* sec4_3      [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 64];
    threadgroup half sB[64 * 64];
    threadgroup int sh_pid[32];
    threadgroup int sh_tk[32];
    const int y = int(tgp.y);
    if (y >= int(meta[9])) return;
    const uint2 td = tiles[y];
    const int e = int(td.x >> 16), mm = int(td.x & 0xFFFFu), off = int(td.y);
    const int n0 = int(tgp.x) * 64;
    const InkExpertDev ed = ex[e];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    device const uchar* s_base = body + 5308416;
    const int sgm = int(sg >> 2), sgn = int(sg & 3u);
    const int br = int(tid >> 2), bi = int(tid & 3u);
    const int ar = int(tid >> 3), ac = int((tid & 7u) * 8u);
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;

    if (tid < 32) {
        const int p = (int(tid) < mm) ? pair_list[off + int(tid)] : 0;
        sh_pid[tid] = p; sh_tk[tid] = p >> 3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < 2304; k0 += 64) {
        if (ar < mm) {
            device const float4* xp = (device const float4*)(x_in + (size_t)sh_tk[ar] * 2304 + k0 + ac);
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(xp[0]);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(xp[1]);
        } else if (ar < 16 || mm > 16) {
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(0.0h);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(0.0h);
        }
        decode_sb_mx8_16(body, s_base, 2304, 72, n0 + br, k0, bi, sB + br * 64 + bi * 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (mm > 16) {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, a1, b0, b1;
                simdgroup_load(a0, sA + (sgm * 16 + 0) * 64 + kk * 8, 64);
                simdgroup_load(a1, sA + (sgm * 16 + 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
                simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
                simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
            }
        } else {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, b0, b1;
                simdgroup_load(a0, sA + (sgm * 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int gn0 = n0 + sgn * 16 + int(fn);
    const int lm0 = (mm > 16 ? (sgm * 16) : (sgm * 8)) + int(fm);
    const int lm1 = sgm * 16 + 8 + int(fm);
    if (lm0 < mm) {
        const int pid_m = sh_pid[lm0];
        h_act[(size_t)pid_m * 1152 + (gn0 >> 1)]       = situ_act(acc00.thread_elements()[0], acc00.thread_elements()[1]);
        h_act[(size_t)pid_m * 1152 + ((gn0 + 8) >> 1)] = situ_act(acc01.thread_elements()[0], acc01.thread_elements()[1]);
    }
    if (mm > 16 && lm1 < mm) {
        const int pid_m = sh_pid[lm1];
        h_act[(size_t)pid_m * 1152 + (gn0 >> 1)]       = situ_act(acc10.thread_elements()[0], acc10.thread_elements()[1]);
        h_act[(size_t)pid_m * 1152 + ((gn0 + 8) >> 1)] = situ_act(acc11.thread_elements()[0], acc11.thread_elements()[1]);
    }
}

kernel void ink_moe_dn_mm_mx8(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* topk_w      [[buffer(2)]],
    device const uint2* tiles       [[buffer(3)]],
    device const int* uoff_unused   [[buffer(4)]],
    device const int* ucnt_unused   [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device const float* h_act       [[buffer(8)]],
    device float* y_pairs           [[buffer(9)]],
    device const uchar* sec4_1      [[buffer(10)]],
    device const uchar* sec4_2      [[buffer(11)]],
    device const uchar* sec4_3      [[buffer(12)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 64];
    threadgroup half sB[64 * 64];
    threadgroup int sh_pid[32];
    threadgroup float sh_rw[32];
    const int yy = int(tgp.y);
    if (yy >= int(meta[9])) return;
    const uint2 td = tiles[yy];
    const int e = int(td.x >> 16), mm = int(td.x & 0xFFFFu), off = int(td.y);
    const int n0 = int(tgp.x) * 64;
    const InkExpertDev ed = ex[e];
    device const uchar* w_base = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off + ed.dn_off;
    device const uchar* s_base = w_base + 2654208;
    const int sgm = int(sg >> 2), sgn = int(sg & 3u);
    const int br = int(tid >> 2), bi = int(tid & 3u);
    const int ar = int(tid >> 3), ac = int((tid & 7u) * 8u);
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;

    if (tid < 32) {
        const int p = (int(tid) < mm) ? pair_list[off + int(tid)] : 0;
        sh_pid[tid] = p; sh_rw[tid] = (int(tid) < mm) ? topk_w[p] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < 1152; k0 += 64) {
        if (ar < mm) {
            device const float4* xp = (device const float4*)(h_act + (size_t)sh_pid[ar] * 1152 + k0 + ac);
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(xp[0]);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(xp[1]);
        } else if (ar < 16 || mm > 16) {
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(0.0h);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(0.0h);
        }
        decode_sb_mx8_16(w_base, s_base, 1152, 36, n0 + br, k0, bi, sB + br * 64 + bi * 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (mm > 16) {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, a1, b0, b1;
                simdgroup_load(a0, sA + (sgm * 16 + 0) * 64 + kk * 8, 64);
                simdgroup_load(a1, sA + (sgm * 16 + 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
                simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
                simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
            }
        } else {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, b0, b1;
                simdgroup_load(a0, sA + (sgm * 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int gn0 = n0 + sgn * 16 + int(fn);
    const int lm0 = (mm > 16 ? (sgm * 16) : (sgm * 8)) + int(fm);
    const int lm1 = sgm * 16 + 8 + int(fm);
    if (lm0 < mm) {
        const int pid_m = sh_pid[lm0]; const float rw = sh_rw[lm0];
        y_pairs[(size_t)pid_m * 2304 + gn0]     = acc00.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 1] = acc00.thread_elements()[1] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 8] = acc01.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 9] = acc01.thread_elements()[1] * rw;
    }
    if (mm > 16 && lm1 < mm) {
        const int pid_m = sh_pid[lm1]; const float rw = sh_rw[lm1];
        y_pairs[(size_t)pid_m * 2304 + gn0]     = acc10.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 1] = acc10.thread_elements()[1] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 8] = acc11.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 9] = acc11.thread_elements()[1] * rw;
    }
}

// Flattened Parallel (expert, m_tile) simdgroup_matrix w13 + fused SITU (General: SF4 + MX8 + S2_K..S8_K, BM=32 BN=64 BK=64!)
kernel void ink_moe_gu_mm_gen(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* x_in        [[buffer(2)]],
    device const uint2* tiles       [[buffer(3)]],
    device const int* uoff_unused   [[buffer(4)]],
    device const int* ucnt_unused   [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device float* h_act             [[buffer(8)]],
    device const uchar* sec4_1      [[buffer(9)]],
    device const uchar* sec4_2      [[buffer(10)]],
    device const uchar* sec4_3      [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 64];
    threadgroup half sB[64 * 64];
    threadgroup int sh_pid[32];
    threadgroup int sh_tk[32];
    const int y = int(tgp.y);
    if (y >= int(meta[9])) return;
    const uint2 td = tiles[y];
    const int e = int(td.x >> 16), mm = int(td.x & 0xFFFFu), off = int(td.y);
    const int n0 = int(tgp.x) * 64;
    const InkExpertDev ed = ex[e];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    const int gcode = int(ed.gu_code);
    const bool is_sk_dn = (ed.dn_code != 15 && ed.dn_code != 16 && ed.dn_code != 20);
    const float sf4_outer = (gcode == 15) ? (((device const float*)body)[0] / ((device const float*)body)[1]) : 0.0f;
    const size_t sk_rowb = (gcode != 15 && gcode != 16 && gcode != 20) ? (9ul * size_t(sk_sb_bytes(gcode))) : 0ul;
    const uint gguf_t = (gcode == 20) ? ((device const InkGgufExpertHeader*)body)->gu_type : 0u;
    const size_t gguf_rb = (gcode == 20) ? gguf_row_bytes_m(gguf_t, 2304) : 0ul;
    const uint gguf_fl = (gcode == 20) ? ((device const InkGgufExpertHeader*)body)->flags : 0u;
    const float gguf_gs = (gcode == 20) ? ((device const InkGgufExpertHeader*)body)->gate_scale : 1.0f;
    const float gguf_us = (gcode == 20) ? ((device const InkGgufExpertHeader*)body)->up_scale : 1.0f;
    const int sgm = int(sg >> 2), sgn = int(sg & 3u);
    const int br = int(tid >> 2), bi = int(tid & 3u);
    const int ar = int(tid >> 3), ac = int((tid & 7u) * 8u);
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;

    if (tid < 32) {
        const int p = (int(tid) < mm) ? pair_list[off + int(tid)] : 0;
        sh_pid[tid] = p; sh_tk[tid] = p >> 3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < 2304; k0 += 64) {
        if (ar < mm) {
            device const float4* xp = (device const float4*)(x_in + (size_t)sh_tk[ar] * 2304 + k0 + ac);
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(xp[0]);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(xp[1]);
        } else if (ar < 16 || mm > 16) {
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(0.0h);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(0.0h);
        }
        if (gcode == 15) {
            decode_sb_sf4_16(body + 16, body + 16 + 2654208, sf4_outer, 1152, 144, n0 + br, k0, bi, sB + br * 64 + bi * 16);
        } else if (gcode == 16) {
            decode_sb_mx8_16(body, body + 5308416, 2304, 72, n0 + br, k0, bi, sB + br * 64 + bi * 16);
        } else if (gcode == 20) {
            const int gn = n0 + br;
            const int src_row = (gguf_fl & 1u) ? gn : (((gn & 1) ? 1152 : 0) + (gn >> 1));
            const float sc = (gn & 1) ? gguf_us : gguf_gs;
            decode_sb_gguf_16(body + 32 + (size_t)src_row * gguf_rb, gguf_t, sc, k0 + bi * 16, sB + br * 64 + bi * 16);
        } else {
            device const uchar* rp = body + 2304 + (size_t)(n0 + br) * sk_rowb;
            decode_sb_sk_16(rp, gcode, k0 + bi * 16, sB + br * 64 + bi * 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (mm > 16) {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, a1, b0, b1;
                simdgroup_load(a0, sA + (sgm * 16 + 0) * 64 + kk * 8, 64);
                simdgroup_load(a1, sA + (sgm * 16 + 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
                simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
                simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
            }
        } else {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, b0, b1;
                simdgroup_load(a0, sA + (sgm * 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int r0 = (n0 + sgn * 16 + int(fn)) >> 1;
    const int r1 = r0 + 4;
    const float isq0 = is_sk_dn ? (1.0f / float(as_type<half>(ld16(body + 2 * r0)))) : 1.0f;
    const float isq1 = is_sk_dn ? (1.0f / float(as_type<half>(ld16(body + 2 * r1)))) : 1.0f;
    const int lm0 = (mm > 16 ? (sgm * 16) : (sgm * 8)) + int(fm);
    const int lm1 = sgm * 16 + 8 + int(fm);
    if (lm0 < mm) {
        const int pid_m = sh_pid[lm0];
        h_act[(size_t)pid_m * 1152 + r0] = situ_act(acc00.thread_elements()[0], acc00.thread_elements()[1]) * isq0;
        h_act[(size_t)pid_m * 1152 + r1] = situ_act(acc01.thread_elements()[0], acc01.thread_elements()[1]) * isq1;
    }
    if (mm > 16 && lm1 < mm) {
        const int pid_m = sh_pid[lm1];
        h_act[(size_t)pid_m * 1152 + r0] = situ_act(acc10.thread_elements()[0], acc10.thread_elements()[1]) * isq0;
        h_act[(size_t)pid_m * 1152 + r1] = situ_act(acc11.thread_elements()[0], acc11.thread_elements()[1]) * isq1;
    }
}

// Flattened Parallel (expert, m_tile) simdgroup_matrix w2 (General: SF4 + MX8 + Paired-2-Row S2_K..S8_K, BM=32 BN=64 BK=64!)
kernel void ink_moe_dn_mm_gen(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* topk_w      [[buffer(2)]],
    device const uint2* tiles       [[buffer(3)]],
    device const int* uoff_unused   [[buffer(4)]],
    device const int* ucnt_unused   [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device const float* h_act       [[buffer(8)]],
    device float* y_pairs           [[buffer(9)]],
    device const uchar* sec4_1      [[buffer(10)]],
    device const uchar* sec4_2      [[buffer(11)]],
    device const uchar* sec4_3      [[buffer(12)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 64];
    threadgroup half sB[64 * 64];
    threadgroup int sh_pid[32];
    threadgroup float sh_rw[32];
    const int yy = int(tgp.y);
    if (yy >= int(meta[9])) return;
    const uint2 td = tiles[yy];
    const int e = int(td.x >> 16), mm = int(td.x & 0xFFFFu), off = int(td.y);
    const int n0 = int(tgp.x) * 64;
    const InkExpertDev ed = ex[e];
    device const uchar* body = ink_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    const int dcode = int(ed.dn_code);
    const float sf4_outer = (dcode == 15) ? (((device const float*)body)[2] / ((device const float*)body)[3]) : 0.0f;
    const size_t sk_pairb = (dcode != 15 && dcode != 16 && dcode != 20) ? (9ul * size_t(sk_sb_bytes(dcode))) : 0ul;
    const uint gguf_t = (dcode == 20) ? ((device const InkGgufExpertHeader*)body)->dn_type : 0u;
    const size_t gguf_rb = (dcode == 20) ? gguf_row_bytes_m(gguf_t, 1152) : 0ul;
    const float gguf_ds = (dcode == 20) ? ((device const InkGgufExpertHeader*)body)->dn_scale : 1.0f;
    const int sgm = int(sg >> 2), sgn = int(sg & 3u);
    const int br = int(tid >> 2), bi = int(tid & 3u);
    const int ar = int(tid >> 3), ac = int((tid & 7u) * 8u);
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;

    if (tid < 32) {
        const int p = (int(tid) < mm) ? pair_list[off + int(tid)] : 0;
        sh_pid[tid] = p; sh_rw[tid] = (int(tid) < mm) ? topk_w[p] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < 1152; k0 += 64) {
        if (ar < mm) {
            device const float4* xp = (device const float4*)(h_act + (size_t)sh_pid[ar] * 1152 + k0 + ac);
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(xp[0]);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(xp[1]);
        } else if (ar < 16 || mm > 16) {
            *((threadgroup half4*)(sA + ar * 64 + ac + 0)) = half4(0.0h);
            *((threadgroup half4*)(sA + ar * 64 + ac + 4)) = half4(0.0h);
        }
        const int gn = n0 + br;
        if (dcode == 15) {
            decode_sb_sf4_16(body + ed.dn_off, body + ed.dn_off + 1327104, sf4_outer, 576, 72, gn, k0, bi, sB + br * 64 + bi * 16);
        } else if (dcode == 16) {
            decode_sb_mx8_16(body + ed.dn_off, body + ed.dn_off + 2654208, 1152, 36, gn, k0, bi, sB + br * 64 + bi * 16);
        } else if (dcode == 20) {
            decode_sb_gguf_16(body + ed.dn_off + (size_t)gn * gguf_rb, gguf_t, gguf_ds, k0 + bi * 16, sB + br * 64 + bi * 16);
        } else {
            const int rp = gn >> 1;
            const int col_p = ((gn & 1) ? 1152 : 0) + k0 + bi * 16;
            device const uchar* pr = body + ed.dn_off + (size_t)rp * sk_pairb;
            decode_sb_sk_16(pr, dcode, col_p, sB + br * 64 + bi * 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (mm > 16) {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, a1, b0, b1;
                simdgroup_load(a0, sA + (sgm * 16 + 0) * 64 + kk * 8, 64);
                simdgroup_load(a1, sA + (sgm * 16 + 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
                simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
                simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
            }
        } else {
            for (int kk = 0; kk < 8; ++kk) {
                simdgroup_half8x8 a0, b0, b1;
                simdgroup_load(a0, sA + (sgm * 8) * 64 + kk * 8, 64);
                simdgroup_load(b0, sB + (sgn * 16 + 0) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_load(b1, sB + (sgn * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
                simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int gn0 = n0 + sgn * 16 + int(fn);
    const int lm0 = (mm > 16 ? (sgm * 16) : (sgm * 8)) + int(fm);
    const int lm1 = sgm * 16 + 8 + int(fm);
    if (lm0 < mm) {
        const int pid_m = sh_pid[lm0]; const float rw = sh_rw[lm0];
        y_pairs[(size_t)pid_m * 2304 + gn0]     = acc00.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 1] = acc00.thread_elements()[1] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 8] = acc01.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 9] = acc01.thread_elements()[1] * rw;
    }
    if (mm > 16 && lm1 < mm) {
        const int pid_m = sh_pid[lm1]; const float rw = sh_rw[lm1];
        y_pairs[(size_t)pid_m * 2304 + gn0]     = acc10.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 1] = acc10.thread_elements()[1] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 8] = acc11.thread_elements()[0] * rw;
        y_pairs[(size_t)pid_m * 2304 + gn0 + 9] = acc11.thread_elements()[1] * rw;
    }
}
