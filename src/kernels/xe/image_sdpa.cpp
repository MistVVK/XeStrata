// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/image_sdpa.cpp - see include/strata/kernels/image_attn.hpp (qi_sdpa): the image attention's
// unmasked rows through
// a GPU maker's kernel, the tensors' strides the engine's [rows][heads * head_dim] as they are, a plan built once a shape
// (the query rows, the key rows) and kept:
//   * Intel GPUs: oneDNN's SDPA through its graph API (Q . K^T, the scale, the softmax, . V fused into one partition);
//   * NVIDIA GPUs (contrib-llvm): cuDNN's SDPA through cudnn-frontend, cuDNN opened at the first attention
//     (the user's installation, not free software: open_cudnn), on the queue's CUDA stream;
//   * AMD GPUs: aotriton's flash attention (MIT; its library and its kernels' images beside the engine), on the
//     queue's HIP stream, in a module the kernels open on an AMD GPU only (kernels::aotriton_sdpa).
// A GPU or a build without one, or a failure, leaves it to the engine's own kernel.

#include "strata/kernels/image_attn.hpp"

#include "strata/kernels/sdpa_cudnn_internal.hpp"
#include "strata/kernels/sdpa_select.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/sdpa_aotriton.hpp"
#include "strata/kernels/sdpa_onednn.hpp"

#include <sycl/sycl.hpp>

namespace strata::image {

bool qi_sdpa_causal(void* stream) { return kernels::aotriton_sdpa_ready(stream); }

bool qi_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1, int64_t T,
             int64_t heads, int64_t head_dim, uint16_t* out16, void* stream, bool causal, int64_t batch) {
    if (q1 <= q0 || batch <= 0) return true;
    if (batch > 1 && (q0 != 0 || causal)) return false;
    // every maker's kernel here runs on the matrix engines: STRATA_NO_XMX=1 (a GPU without them, played on one with
    // them) leaves the attention to the engine's own kernels, which take their paths without them
    sycl::queue& q = core::Runtime::get().stream(stream);
    const auto maker = kernels::sdpa_maker(q, stream, kernels::SdpaUse::image, causal, head_dim);
    if (maker == kernels::SdpaMaker::aotriton) {   // aotriton: one call a batch
        const int64_t row = heads * head_dim;
        for (int64_t b = 0; b < batch; ++b)
            if (!kernels::aotriton_sdpa(q16 + (b * (q1 - q0) + q0) * row, k16 + b * T * row, v16 + b * T * row,
                                        q1 - q0, T, heads, heads, head_dim, out16 + b * (q1 - q0) * row, stream,
                                        causal))
                return false;
        return true;
    }
#if STRATA_CUDNN
    if (maker == kernels::SdpaMaker::cudnn)
        return sdpa::cudnn(q, q16, k16, v16, q0, q1, T, heads, head_dim, out16, batch);
#endif
#if STRATA_ONEDNN
    // oneDNN's GPU kernels are Intel's: its Level Zero and OpenCL devices
    if (maker == kernels::SdpaMaker::onednn)
        return kernels::onednn_sdpa(q16 + q0 * heads * head_dim, k16, v16, q1 - q0, T, heads, heads, head_dim, out16,
                                    stream, false, batch);
#endif
    (void) k16, (void) v16, (void) T, (void) heads, (void) head_dim, (void) out16, (void) q16;
    return false;
}

}  // namespace strata::image
