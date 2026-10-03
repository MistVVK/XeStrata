// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Native GGUF expert blobs for the opt-in Xe fused prompt path (moe_fused.hpp).
// Gate/up: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; down: Q2_0 or IQ4_NL.
// Unsupported layers use MMQ or the FP16 path.
#pragma once

#include "strata/prefill/moe_fused.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fused {

/// One layer's native expert geometry (cpu::NativeFmt's fields the kernels read; n_embd 2560, n_ff 640).
struct NativeGeom {
    int gu_type = -1, d_type = -1;    ///< ggml types
    size_t gu_row = 0, d_row = 0;     ///< bytes per gate/up row and per down row
    size_t up_off = 0, down_off = 0;  ///< inside the blob
};

/// available(), and the kernels cover this gate/up and down pair on this device.
bool native_supported(int gu_type, int d_type);

/// x [rows][cols] FP32 -> `xa` (act_bytes(rows, cols)): int8 per 32 values in natural order, the native kernels' form.
void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);

/// experts() for native blobs: b.blob[e - e0] is the native blob of expert e (gate at 0, up at g.up_off, down at
/// g.down_off; 2-byte aligned).  `xa` from quantize_act_native; H in `ha` (act_bytes(n, 640)); down into `dm` at the
/// rows of `group`.
void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream);

}  // namespace strata::prefill::fused
