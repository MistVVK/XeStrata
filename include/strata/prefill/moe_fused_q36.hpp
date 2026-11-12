// include/strata/prefill/moe_fused_q36.hpp - the fused int8 prompt-expert kernels for Qwen3.6-35B-A3B's expert shape
// (n_embd 2048, n_ff 512): the copy of moe_fused_iq.cu that src/prefill/moe_fused_iq_q36.cu compiles in its own namespace
// (CUDA).  The same API as moe_fused_iq.hpp's native entry points; the grouping (fused::group, fused::group_bytes,
// fused::act_bytes) does not depend on the shape and is moe_fused.cu's.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fused_q36 {

constexpr int kQ36Embd = 2048, kQ36FF = 512;   ///< the shape this copy is compiled for
constexpr int kQ36MaxBatch = 128;              ///< experts per launch (moe_fused.hpp's kMaxBatch)

#if !defined(STRATA_FUSED_Q36_TU)   // (moe_fused_iq_q36.cu has the copy's own definitions of these)
struct Batch {
    int e0 = 0, e1 = 0;
    const uint8_t* blob[kQ36MaxBatch] = {};
};
struct NativeGeom {
    int gu_type = -1, d_type = -1;
    size_t gu_row = 0, d_row = 0;
    size_t up_off = 0, down_off = 0;
};
/// The kernels cover this gate/up and down pair on the current device (the rules of moe_fused_iq.cu's native_supported).
bool native_supported(int gu_type, int d_type);
/// x [rows][cols] FP32 -> int8 per 32 values in natural order (fused::act_bytes(rows, cols)).
void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);
/// One layer's experts [b.e0, b.e1): see moe_fused_iq.hpp's experts_native.
void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream);
/// native_supported / experts_native plus the K-quant downs of UD-Q4_K_XL (a Q4_K or Q5_K gate/up with a Q5_K or Q6_K down: K = 512 is two 256-value
/// super-blocks), which the copy of moe_fused_iq.cu has instances for but whose dispatch is not part of it.  Every other pair goes to the two above.
bool native_supported_ext(int gu_type, int d_type);
void experts_native_ext(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                        const int32_t* src, void* ha, float* dm, void* stream);
#endif

}  // namespace strata::prefill::fused_q36
