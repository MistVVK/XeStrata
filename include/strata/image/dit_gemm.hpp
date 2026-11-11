// include/strata/image/dit_gemm.hpp - the matrix products of Strata's Qwen-Image-2.1 DiT (src/image/dit/dit_gemm.cu),
// CUDA and HIP: the weight's GGUF blocks decoded to f16 (te_dequant.hpp), the activations rounded to f16, cuBLAS /
// hipBLAS GemmEx with f32 accumulation and f32 output - the arithmetic of the text encoder (step 3) and of
// tools/image/gate/dit_ref.py's "f16op" mode.  stable-diffusion.cpp's MMQ rounds the activations to 8 bits instead.
// The GEMM shapes follow only from the weight and the token count: the same call gives the same bytes on every run.
#pragma once

#include <cstdint>
#include <string>

namespace strata::image {

// y[n][N] (f32, row stride ldy floats) = x[n][K] (f32 rows, stride ldx floats) . W[N][K]^T, W = `type` blocks at w
// (N rows of K values, GGUF layout), on `stream` (cudaStream_t / hipStream_t).  Asynchronous; "" on success.
// The decoded weight and the f16 activations live in buffers kept between calls (dit_gemm_release frees them).
std::string dit_gemm(void* stream, uint32_t type, const void* w, int64_t K, int64_t N, const float* x, int64_t ldx,
                     int64_t n, float* y, int64_t ldy);

// device bytes the buffers hold now
size_t dit_gemm_buffer_bytes();
void dit_gemm_release();

}  // namespace strata::image
