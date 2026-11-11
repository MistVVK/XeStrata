// include/strata/image/qwen3vl_te.hpp - Strata's own text encoder for Qwen-Image-2.1: the Qwen3-VL-8B text model run
// once over a prompt's token ids, CUDA and HIP from one source (src/image/te/qwen3vl_te.cu; HIP through
// include/strata/hip_compat).
//
// It computes what stable-diffusion.cpp 2988060's LLM::LLMRunner (arch QWEN3_VL, text only, out_layers = {36})
// returns to LLMEmbedder::encode_prompt: the residual stream after the last layer, without the final norm, one row of
// `hidden` floats per token.  The caller keeps sd.cpp's token weights and drops the system tokens, as sd.cpp does.
//
// Arithmetic (tools/image/gate/te_ref.py mode "f16op"): weights decoded to f16 (include/strata/image/te_dequant.hpp),
// GEMM inputs f16, cuBLAS / hipBLAS GemmEx with f32 accumulation, residual stream f32, Q / K / V f16, scores f32,
// probabilities f16; the rest f32.  sd.cpp's MMQ GEMMs quantize the inputs to 8 bits instead: its condition is 0.055
// (relative L2) from the f32 reference, this arithmetic 0.0012 (8 gate prompts, 11-11).
// The attention runs over fixed chunks of 256 queries and every GEMM shape follows only from the prompt length, so
// one prompt gives the same bytes on every run of one card.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::image {

struct Qwen3VlTe;

struct Qwen3VlTeStats {
    double encode_ms = 0;          // embeddings up, 36 layers, result down (the stream's events)
    size_t peak_bytes = 0;         // the encoder's own device memory at its peak (weights + work)
    size_t weight_bytes = 0;       // layer weights on the device (streamed: the two layer slots)
    bool streamed = false;
};

// Open the text encoder of a GGUF file (general.architecture "qwen3vl": block_count, embedding_length, ... from its
// metadata) on `device` (-1: the current device).  nullptr and *err on failure (an unknown architecture, a missing
// tensor, a weight type te_dequant.hpp does not read).
// mode 0 (auto): the weights stay resident on cards with at least 14 GiB of TOTAL memory (16 GB cards and up) and are
// streamed otherwise; never decided by free memory.  1: resident, 2: streamed (tests).  Streamed keeps two layers on
// the device and uploads each layer at every encode, one ahead of the compute.  Both give the same bytes.
// threads: for copying the file into the pinned upload chunks (0: up to 8); the bytes do not depend on it.
Qwen3VlTe* qwen3vl_te_open(const std::string& gguf_path, int device, std::string* err, int mode = 0, int threads = 0);

int qwen3vl_te_hidden(const Qwen3VlTe* te);

// ids: n token ids (sd.cpp's, prompt template included); out: n x hidden floats (host), row-major.  "" on success.
std::string qwen3vl_te_encode(Qwen3VlTe* te, const int32_t* ids, int n, float* out, Qwen3VlTeStats* stats = nullptr);

// "read .. ms, upload .. ms": where the open time went.
std::string qwen3vl_te_open_report(const Qwen3VlTe* te);

void qwen3vl_te_close(Qwen3VlTe* te);

}  // namespace strata::image
