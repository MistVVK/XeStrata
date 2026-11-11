// src/image/dit/qi21_kernels.hpp - the Qwen-Image-2.1 DiT's element-wise and attention kernels (src/image/dit/
// qi21_kernels.cu), launched on a caller's stream.  Moved from the sd.cpp patch tools/image/sdcpp_native/qi21-native.cu
// (same kernels, same launch shapes; that patch keeps its own copy for the test builds until the sd-cli hook changes),
// plus the steps ggml still did for the native runner: the timestep sinusoid, SiLU, GELU (tanh form), the zero-centred
// RMSNorm of txt_in.  Every launcher returns 0, or 1 for an unsupported shape, 2 for a launch error.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::image::qi21 {

// x += h * tanh(g) (when h), then out = LayerNorm(x) * (1 + m) (when out); no affine, eps.  T rows of C (C <= 8192).
int res_ln_mod(void* stream, float* x, const float* h, const float* g, float* out, const float* m, int T, int C, float eps);

// per head of T tokens: q, k RMSNorm * wq | wk, RoPE from the per-token table pe ([T][64][4]), q -> qo head-major
// [H][q_tstride][128] (f16 when q_f16, else f32), k, v -> f16 head-major [H][kv_tstride][128] at row kv_off.
// q, k, v rows have stride ld floats.  H % 8 == 0, ld % 4 == 0.
int qkv_post(void* stream, const float* q, const float* k, const float* v, int ld, const float* wq, const float* wk,
             const float* pe, void* qo, int q_f16, int q_tstride, void* ko, void* vo, int kv_tstride, int kv_off, int T,
             int H, float eps);

// out[T][F] = up * silu(gate), gu = [T][gate F | up F]
int swiglu(void* stream, const float* gu, float* out, int T, int F);

// fp32 attention, q f32 head-major, k / v f16 head-major, out token-major [Lq][H*128]; causal: query i sees keys
// 0..i (the text prefix); otherwise all Lk keys (the exact reference path).  Lk * 4 bytes of shared memory <= 48 KiB.
int attn_exact(void* stream, const float* q, int q_tstride, const void* k, const void* v, int kv_tstride, float* out,
               int H, int Lq, int Lk, int causal);

// 1 when this device runs attn_fast (NVIDIA sm_80+: mma.sync; HIP gfx11: WMMA; gfx12 with STRATA_QI_GFX12_ATTN=1)
int has_fast_attn(int device);

// flash attention for the image tokens: q f16 head-major, k / v f16 head-major, out f32 token-major
int attn_fast(void* stream, const void* q, int q_tstride, const void* k, const void* v, int kv_tstride, float* out, int H,
              int nq, int nkv);

// emb[n][dim] = [cos(t * f_j) | sin(t * f_j)], f_j = exp(-ln(max_period) * j / (dim / 2))  (ggml_timestep_embedding)
int timestep_embedding(void* stream, const float* t, float* emb, int n, int dim, float max_period);

// x = x / (1 + exp(-x)) in place
int silu(void* stream, float* x, int64_t n);

// x = 0.5 x (1 + tanh(sqrt(2 / pi) x (1 + 0.044715 x^2))) in place (ggml's GELU)
int gelu_tanh(void* stream, float* x, int64_t n);

// out[r] = x[r] / sqrt(mean(x[r]^2) + eps) * (w + 1), T rows of C (txt_in's zero-centred RMSNorm)
int rms_norm_w1(void* stream, const float* x, const float* w, float* out, int T, int C, float eps);

}  // namespace strata::image::qi21
