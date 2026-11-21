// include/strata/kernels/nvfp4_decode_sm120.hpp - Clean-room MIT Blackwell sm_120a hardware NVFP4 grouped decode/verify.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>

namespace strata::kernels {

/// True when compiled with sm_120a/sm_121a hardware FP4 decode support and running on a Blackwell sm_120/sm_121 GPU
/// (STRATA_NVFP4_DEC_SM120=0 disables at runtime).
bool nvfp4_decode_sm120_available() noexcept;

/// Test/benchmark override: -1 = follow STRATA_NVFP4_DEC_SM120 env (default), 0 = disable, 1 = enable.
void nvfp4_decode_sm120_set_override(int mode) noexcept;

/// Hardware Blackwell sm_120a grouped NVFP4 (type 40) MoE decode/verify (fused gate+up+SwiGLU+Q8_1 -> down).
/// Uses native PTX cvt.rn.f16x2.e2m1x2 + K-chunked split-half FP16 shared-memory activation caching across up to
/// 8 tokens per expert group, reading weights once per group. Returns false if unsupported shape/device.
bool nvfp4_decode_sm120_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr,
                                const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                                const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                                const void* x_q8_1, void* hq_scratch, float* out, void* stream);

}  // namespace strata::kernels
