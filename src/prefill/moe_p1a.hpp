// src/prefill/moe_p1a.hpp - gfx11: the K-quant prompt experts with F16 activations and F16 H (moe_p1a.cu, gufo's
// RoutedPairF16Kernel / RoutedDownF16Kernel).  The engine takes them on gfx1151 only, behind STRATA_PF_P1A=1
// (moe_fused_iq.cu: experts_native, p1a_input); off by default.
// Internal to moe_fused_iq.cu, which owns the native experts' tables and launches.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"

namespace strata::prefill::fused::p1a {

/// The pair has kernels here: Q4_K gate/up with Q5_1 down; with `l2` (STRATA_PF_P1A_L2=1) also Q5_K gate/up and Q8_0 down.
bool takes(int gu_type, int d_type, bool l2);

/// Gate/up + SwiGLU for experts [b.e0, b.e1): `x16` the layer input as FP16 rows [token][n_embd], `src` the routed rows'
/// tokens, `ts`, `tiles`, `off` fused::group's tables (at most `work_bound` 64-row tiles in this batch); H as FP16 rows
/// [routed row][n_ff] into `h16`.
void gate_up(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t work_bound,
             const void* x16, const int32_t* src, void* h16, cudaStream_t s);
/// Down of that H: the routed rows' outputs as FP16 [routed row][n_embd] into `d16` (saturated at +-65504; moe_combine16 reads them).
void down(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t work_bound,
          const void* h16, void* d16, cudaStream_t s);

}  // namespace strata::prefill::fused::p1a
