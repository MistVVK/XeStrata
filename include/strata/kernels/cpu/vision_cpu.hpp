// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/cpu/vision_cpu.hpp - the vision encoder's rows on the CPU (core::Vision's CPU path,
// src/core/vision.cpp): the LayerNorm, the two GELUs, the residual add, the ViT's QKV split with its 2-D RoPE, and the
// attention over every patch, its two products through bf16_gemm.hpp.  The GPU's are image_attn.hpp's; these compute
// the same functions in FP32 on the host.
//
// THE ATTENTION.  Per head, the keys and the transposed values are packed as BF16 panels once a layer; a piece of
// query rows then takes its scores (FP32 queries against the BF16 keys), the softmax in FP32 and the weighted values
// (the FP32 probabilities against the BF16 values), and divides by the softmax's sum.  The int8 form (the default;
// XESTRATA_VISION_CPU_I8=0 takes this BF16 one, core/vision.hpp) packs them as int8 instead (the keys a scale a key,
// the values a scale per kVisionAttnBlock keys) and quantizes the queries and the probabilities the same way before
// i8_gemm_rows; keys past kI8MaxIn go in more than one chunk of the value product, summed.
//
// WHY THE INT8 DEFAULT AND WHAT =0 KEEPS: the int8 products (i8_gemm.hpp) are about twice as fast on an AVX2 CPU,
// but their embeddings stand further from llama.cpp's mtmd (mtmd on the CPU against mtmd on Vulkan: a mean cosine of
// 0.99971 per image token).  With W8A8 products everywhere the mean cosine came to 0.99637; with every product in
// BF16 it came to 0.99978.  Measured on a Ryzen 7 5700X, 8 threads, 9 pictures of 1,024 tokens, 2026-11-12: int8
// 5.3 s a picture, BF16 9.5 s, mtmd 10.9 s; the answers to 36 questions: int8 18 at 300 tokens / 28 at 1,024, BF16
// 19 / 26.  The user chose the speed for the default (2026-11-12); XESTRATA_VISION_CPU_I8=0 keeps the closer form.
#pragma once

#include "strata/kernels/cpu/cpu_ops.hpp"

#include <cstdint>
#include <vector>

namespace strata::kernels::cpu {

/// The keys one scale of the probabilities and the transposed values covers in the int8 attention.
inline constexpr int kVisionAttnBlock = 32;

/// y [n][C] = (x - mean) / sqrt(var + eps) * w + b, per row (FP32), rows on the pool.
void vision_layer_norm(ExpertPool& pool, const float* x, int64_t n, int64_t C, const float* w, const float* b, float eps,
                       float* y);
/// x[i] = GELU(x[i]), the tanh form (the ViT's MLP), on the pool.
void vision_gelu_tanh(ExpertPool& pool, float* x, int64_t n);
/// x[i] = GELU(x[i]), the erf form (the merger's MLP), on the pool.
void vision_gelu_erf(ExpertPool& pool, float* x, int64_t n);
/// x[i] += v[i], on the pool.
void vision_add(ExpertPool& pool, float* x, const float* v, int64_t n);

/// The 2-D RoPE's angles [T][hd / 2] (cos, sin) for the patches at yx [T][2] (row, column): pair j < hd / 4 turns by
/// the row, the rest by the column, j counted from each half's start, theta^(-4 f / hd) (qk_norm_rope.hpp's Theta).
void vision_rope_table(const int32_t* yx, int64_t T, int64_t hd, float theta, std::vector<float>& cos,
                       std::vector<float>& sin);

/// The attention of one layer: qkv [T][q | k | v][heads][hd] (the product's rows, bias added) -> out [T][heads * hd].
/// The pairs (j, j + hd / 2) of q and k rotated by the table, q scaled by 1 / sqrt(hd); every patch sees every patch.
/// `i8`: the int8 form on that ISA (i8_gemm_isa(), its weight levels i8_gemm_weight_levels), null the BF16 one.
class VisionAttention {
public:
    void run(ExpertPool& pool, const float* qkv, int64_t T, int64_t heads, int64_t hd, const std::vector<float>& cos,
             const std::vector<float>& sin, float* out, const I8GemmIsa* i8 = nullptr);

private:
    void run_i8(ExpertPool& pool, I8GemmIsa isa, int64_t T, int64_t heads, int64_t hd, float* out);
    std::vector<float> q_, k_, vt_;   // [heads][T][hd], [heads][T][hd], [heads][hd][T]
    std::vector<Bf16Matrix> keys_, values_;   // per head
    std::vector<I8Matrix> keys8_, values8_;   // per head; the values per head and key chunk
};

}  // namespace strata::kernels::cpu
