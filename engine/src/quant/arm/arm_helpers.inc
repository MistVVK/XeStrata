// arm_helpers.inc - the few ggml-cpu helpers the ported NEON code names (ggml/src/ggml-cpu/ggml-cpu-impl.h at 3cf03257f219afbe7334045ff7c6a06ac68c627d,
// Copyright (c) 2023-2026 The ggml authors, MIT), in their aarch64 form.  Included inside the neon namespace by dot_neon.cpp and act_neon.cpp.

#ifndef STRATA_ARM_HELPERS
#define STRATA_ARM_HELPERS

#define ggml_vld1q_u32(w, x, y, z) { (w), (x), (y), (z) }

#define ggml_int16x8x2_t  int16x8x2_t
#define ggml_uint8x16x2_t uint8x16x2_t
#define ggml_uint8x16x4_t uint8x16x4_t
#define ggml_int8x16x2_t  int8x16x2_t
#define ggml_int8x16x4_t  int8x16x4_t

#define ggml_vld1q_s16_x2 vld1q_s16_x2
#define ggml_vld1q_u8_x2  vld1q_u8_x2
#define ggml_vld1q_u8_x4  vld1q_u8_x4
#define ggml_vld1q_s8_x2  vld1q_s8_x2
#define ggml_vld1q_s8_x4  vld1q_s8_x4
#define ggml_vqtbl1q_s8   vqtbl1q_s8
#define ggml_vqtbl1q_u8   vqtbl1q_u8

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

static inline int32x4_t ggml_nvfp4_dot8(const int8x8_t q4_lo, const int8x8_t q8_lo, const int8x8_t q4_hi, const int8x8_t q8_hi) {
    const int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    const int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    const int32x4_t sum_lo = vpaddlq_s16(p_lo);
    const int32x4_t sum_hi = vpaddlq_s16(p_hi);
    return vaddq_s32(sum_lo, sum_hi);
}

#endif
