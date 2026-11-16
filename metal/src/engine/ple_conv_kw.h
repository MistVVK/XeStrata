// metal/src/engine/ple_conv_kw.h - the PLE conv kernel's weight (blk.1.ple_conv1d.weight) as the F16 values ple_conv_add
// reads, as plain code so it can be tested without a GPU (metal/tests/ple_conv_kw_test.cpp).
//
// The kernel reads half.  An F16 tensor is used in place; F32 (Unsloth's UD-IQ3_XXS and up, Q8_0) and BF16 (Unsloth's
// UD-IQ1_S / IQ1_M / Q2_0 / IQ2_XS / Q2_K_XL) are narrowed at load with round-to-nearest-even, as tools/iq_pack.py does
// for the host's pack.  Before #5387 only F32 was narrowed: a BF16 tensor was read as F16 bits (BF16 1.0 = 0x3F80 reads
// as F16 1.875) and the Metal engine wrote garbage from the first token (#5416, #5818); the CUDA / HIP / SYCL engines
// read the pack's F16 copy and were not affected.  Any other type is refused at load instead of being misread.
#pragma once

#include "ref_quant.h"

#include <cstddef>
#include <cstdint>

namespace strata::metal_engine {

enum class PleConvKw { file_f16, narrow_f32, narrow_bf16, unsupported };

inline PleConvKw ple_conv_kw_plan(int type) {
    switch (type) {
        case ref::F16: return PleConvKw::file_f16;
        case ref::F32: return PleConvKw::narrow_f32;
        case ref::BF16: return PleConvKw::narrow_bf16;
        default: return PleConvKw::unsupported;
    }
}

// the element count of a tensor of `bytes` bytes stored as `type` (0 for a type ple_conv_kw_plan refuses)
inline size_t ple_conv_kw_elems(int type, size_t bytes) {
    switch (ple_conv_kw_plan(type)) {
        case PleConvKw::file_f16: case PleConvKw::narrow_bf16: return bytes / 2;
        case PleConvKw::narrow_f32: return bytes / 4;
        default: return 0;
    }
}

// narrows n values of an F32 or BF16 tensor to F16 (dst holds n values); false for any other type
inline bool ple_conv_kw_narrow(int type, const void* src, size_t n, uint16_t* dst) {
    const PleConvKw p = ple_conv_kw_plan(type);
    if (p == PleConvKw::narrow_f32) {
        const float* s = (const float*) src;
        for (size_t i = 0; i < n; ++i) dst[i] = ref::f32_to_f16(s[i]);
        return true;
    }
    if (p == PleConvKw::narrow_bf16) {
        const uint16_t* s = (const uint16_t*) src;
        for (size_t i = 0; i < n; ++i) dst[i] = ref::f32_to_f16(ref::bf16_to_f32(s[i]));
        return true;
    }
    return false;
}

}  // namespace strata::metal_engine
