#pragma once
#include <cstdint>

namespace strata::kernels {
// Use bounded byte offsets for IQ2_XXS/IQ2_S/IQ1_M gate/up FP16 outputs.
// False leaves unsupported formats or outputs exceeding UINT32_MAX to the
// existing full-width path; USM base pointers always retain their full width.
bool iq_dequant_gu_f16_bounded(int type, const void* gate, const void* up,
                              int64_t n_ff, int64_t n_embd, uint16_t* output,
                              void* stream);
// SYCL-only gate/up dequantization: independent 256-value blocks share a WG.
// The quantization formulas and interleaved output layout are unchanged.
// subgroups must be 2, 4 or 8; every subgroup has exactly 32 lanes.
// static_type specializes formats 16, 22 and 29; others retain the generic path.
// static_row also fixes their 2560-wide rows to ten blocks; other widths fall back.
void iq_dequant_gu_f16_tiled(int type, const void* gate, const void* up,
                           int64_t n_ff, int64_t n_embd, uint16_t* output,
                           void* stream, int subgroups, bool static_type = false,
                           bool static_row = false);
}
