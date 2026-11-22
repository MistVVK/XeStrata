// include/strata/prefill/moe_fp8_mxf_tc.hpp - clean-room MIT Blackwell (sm_120a / sm_121a) mxf8f6f4 Tensor Core MoE prefill.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Hardware block-scaled FP8 (mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0)
// prompt experts for native GGUF formats (IQ3_S, IQ3_XXS, IQ2_S, IQ4_XS, Q4_K, Q5_K, Q8_0 gate/up; IQ4_NL, Q5_1, Q8_0 down).
// Eliminates per-32 scalar I2F + FFMA scaling inside the K-loop by folding the power-of-2 exponent into hardware UE8M0
// block scales and the [1, 2) mantissa directly into E4M3 weights, reusing Strata's existing 80-byte/64-val activation slots.
#pragma once

#include <cstdint>

#include "strata/prefill/moe_fused_iq.hpp"

namespace strata::prefill::fp8_mxf_tc {

/// True when this binary compiled the sm_120a / sm_121a mxf8f6f4 kernels (CUDA 13.0+, arch 120 or 121).
bool built();

/// True when built() and the current CUDA device is compute capability 12.0 or 12.1 with runnable mxf8f6f4 kernels.
bool available();

/// True when available(), STRATA_PF_MXFP8_MOE != 0, and (gu_type, d_type) are covered by this path.
bool wanted(int gu_type, int d_type);

/// Quantize x [rows][cols] FP32 (cols a multiple of 64) into `xa` (80 bytes per 64 values: 64 E4M3 + 2 UE8M0 + pad).
void quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);

/// Run gate+up (+ fused SwiGLU + MXFP8 H quantize into `ha`) and down (into `dm` [n][kMoeEmbd] FP32) for [b.e0, b.e1).
void experts(const strata::prefill::fused::Batch& b, const strata::prefill::fused::NativeGeom& g, int n_expert,
             int64_t n, const void* scratch, const void* xa, const int32_t* src, void* ha, float* dm, void* stream);

}  // namespace strata::prefill::fp8_mxf_tc
