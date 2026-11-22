// SYCL port, 0.1.43: one expert's gate/up and down matrices dequantized to FP16 in ONE launch (the prompt path's FP16 expert
// path used two: iq_dequant_gu_f16 then iq_dequant_f16).  The values are the two launches' values, byte for byte; only the
// launch count changes (one kernel boundary less per expert, ~24,500 per 4K prompt).
#pragma once
#include <cstdint>
namespace strata::kernels {
/// gate/up -> dst_gu (interleaved rows 2r = gate r, 2r+1 = up r, as iq_dequant_gu_f16), down (n_embd x n_ff values, n a multiple
/// of 256) -> dst_d (as iq_dequant_f16).  Same argument rules as the two functions.
/// `tail` (device, NVFP4 only): {s_gate, s_up, s_down, 0}; gate rows scaled by tail[0], up by tail[1], down by tail[2].
void iq_dequant_expert_f16(int gu_type, int d_type, const void* gate, const void* up, const void* down, int64_t n_ff,
                           int64_t n_embd, uint16_t* dst_gu, uint16_t* dst_d, void* stream, const float* tail = nullptr);
}  // namespace strata::kernels
