// SYCL-only sparse FP32 prompt attention; no change to the common CUDA/HIP API.
#pragma once
#include "strata/kernels/qsa_decode_attn.hpp"

namespace strata::kernels {
// Enqueues core and merge into the supplied queue. Returns false before enqueue
// for unsupported pools, geometry, alignment or device. Scratch uses the existing
// qsa_decode_attn_scratch_floats layout; weights and KV storage are unchanged.
bool qsa_prompt_attn_tla_batch(const float* q, const QsaAttnPools& pools,
                             const int32_t* ids, const int32_t* steps, int64_t cap,
                             const QsaShapes& shapes, float* scratch, float* out,
                             int64_t n_q, void* stream);
}
