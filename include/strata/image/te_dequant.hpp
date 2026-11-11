// include/strata/image/te_dequant.hpp - the text encoder's weight decoders: GGUF block formats -> f16 rows, on CUDA and
// HIP (src/image/te/te_dequant.cu) and on the host (the same bits, for tests).
//
// Qwen-Image-2.1's text encoder (Qwen3-VL-8B, Unsloth UD-Q4_K_XL) stores its layer weights as Q4_K, Q5_K, Q6_K and
// IQ4_XS; the 1-D weights are F32.  Our encoder decodes one matrix at a time into an f16 buffer and multiplies it with
// cuBLAS / hipBLAS GemmEx.  The value of every weight is the one ggml's dequantize_row_* computes in f32 (the order of
// operations of include/strata/artifact/dequant.hpp, every product and difference rounded on its own, never fused),
// rounded to f16 to nearest even (include/strata/kernels/f16_bits.hpp).  The device and the host path give the same
// bits; tools/image/te test program te_dequant_check compares them over every tensor of a file.
//
// Kept separate from the engine's src/kernels/cuda/dequant_bf16.cu on purpose (coordinator 11-11): sharing one header
// of block decoders with the engine is a later step under the engine identity gate.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::image {

// ggml type ids the decoders read
enum : uint32_t { TE_F32 = 0, TE_F16 = 1, TE_Q8_0 = 8, TE_Q4_K = 12, TE_Q5_K = 13, TE_Q6_K = 14, TE_IQ4_XS = 23, TE_BF16 = 30 };

bool te_type_supported(uint32_t type);

// Bytes of one row of ne0 values; 0 when the type is not supported or ne0 is not a whole number of blocks.
size_t te_row_bytes(uint32_t type, int64_t ne0);

// rows x ne0 values (GGUF layout, rows one after the other) at d_src -> f16 bit patterns at d_dst, row-major (ld =
// ne0), on `stream` (a cudaStream_t / hipStream_t; nullptr = the default stream).  Asynchronous; "" on success.
std::string te_dequant_f16(uint32_t type, const void* d_src, int64_t ne0, int64_t rows, uint16_t* d_dst, void* stream);

// The same on the host (the reference of the device path).
std::string te_dequant_f16_host(uint32_t type, const uint8_t* src, int64_t ne0, int64_t rows, uint16_t* dst);

// The f32 values themselves on the host (ggml_get_rows of the token embedding: no f16 rounding).
std::string te_dequant_f32_host(uint32_t type, const uint8_t* src, int64_t ne0, int64_t rows, float* dst);

}  // namespace strata::image
