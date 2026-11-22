// include/strata/prefill/moe_fp4_tc.hpp - native Blackwell (sm_120a / sm_121a) mxf4nvf4 Tensor Core prompt experts.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Clean-room MIT implementation of fused W4A4 MoE prefill for ModelOpt NVFP4 / NVFP4 expert layers on GeForce RTX 50-series
// / RTX PRO Blackwell (sm_120a) and DGX Spark GB10 (sm_121a):
//   - quantize_act: per 64 FP32 values (one 80-byte AB slot in m.Xq), computes a 64-element FP32 scale g_blk = amax / (6 * 448),
//     four UE4M3 sub-block scales d[0..3], and 32 split-half E2M1x2 bytes qs[0..31] matching block_nvfp4's layout verbatim.
//   - experts: persistent tile-scheduled gate+up and down kernels issuing PTX
//       mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3,
//     with SwiGLU + NVFP4 activation re-quantization fused directly inside the gate/up epilogue into m.H (two launches per
//     expert batch, zero extra VRAM buffers).
// Opt-in via STRATA_PREFILL_NVFP4=w4a4; default W4A8 path stays byte-identical when unset.
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/prefill/moe_fused_iq.hpp"

namespace strata::prefill::fp4_tc {

/// True when this binary compiled the sm_120a / sm_121a mxf4nvf4 kernels (CUDA 13.0+, arch 120 or 121).
bool built();

/// True when built() and the current CUDA device is compute capability 12.0 or 12.1 with runnable mxf4nvf4 kernels.
bool available();

/// True when available() and STRATA_PREFILL_NVFP4=w4a4 was requested (and the loaded model provides `<stem>.input_scale` when required).
bool wanted();

/// Record whether the loaded NVFP4 model carries companion `<stem>.input_scale` tensors.
/// When false, `wanted()` safely degrades to W4A8 instead of dispatching uncalibrated W4A4.
void set_model_has_input_scale(bool has_scale);
bool model_has_input_scale();

/// Quantize x [rows][cols] FP32 (cols a multiple of 64) into `xa` (fused::act_bytes(rows, cols): 80 bytes per 64 values).
void quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);

/// Run gate+up (+ fused SwiGLU + NVFP4 H quantize into `ha`) and down (into `dm` [n][kMoeEmbd] FP32) for NVFP4 experts [b.e0, b.e1).
void experts(const strata::prefill::fused::Batch& b, const strata::prefill::fused::NativeGeom& g, int n_expert,
             int64_t n, const void* scratch, const void* xa, const int32_t* src, void* ha, float* dm, void* stream);

}  // namespace strata::prefill::fp4_tc
