// include/strata/image/wan_vae.hpp - Strata's own decoder for Qwen-Image-2.1's VAE (a Wan2.2-style residual decoder),
// CUDA and HIP from one source (src/image/vae/wan_vae.cu; HIP through include/strata/hip_compat).
//
// It computes what stable-diffusion.cpp 2988060's WanVAE::decode computes for one picture, with one deliberate
// difference: sd.cpp accumulates every convolution in f16 (cuBLAS COMPUTE_16F), we accumulate in f32.  The operands are
// the same (weights bf16 -> f16 round-to-nearest-even, conv inputs x 2^-7 -> f16), everything else is f32.
// The picture is always the one-pass (untiled) result: memory is bounded by running every convolution over fixed-size
// bands of pixels, and the band size never depends on the picture size or on free memory, so a pixel's value is the
// same on every card of one kind whatever its memory.
#pragma once

#include <cstddef>
#include <functional>
#include <string>

namespace strata::image {

struct WanVae;

struct WanVaeStats {
    double decode_ms = 0;          // upload + decode + download, measured with the stream's events
    size_t peak_bytes = 0;         // the decoder's own device memory at its peak (weights + activations + work)
    size_t weight_bytes = 0;
};

// Read the decoder of a Qwen-Image-2.1 VAE file (.safetensors, bf16) to the current device of `device` (-1: the
// current device).  nullptr and *err on failure.
WanVae* wan_vae_open(const std::string& path, int device, std::string* err);

// latent: the VAE-space latent (after sd.cpp's diffusion_to_vae_latents), planar [64][h][w] floats on the host.
// out: planar [4][16h][16w] floats on the host (RGBA in about [-1, 1], before sd.cpp's 0-1 scaling).
// Returns an empty string on success.
std::string wan_vae_decode(WanVae* vae, const float* latent, int h, int w, float* out, WanVaeStats* stats = nullptr);

// The diffusion-space latent (what the sampler ends with, planar [64][h][w]) to the VAE-space latent, in place:
// sd.cpp's diffusion_to_vae_latents for Qwen-Image-2.1 (x * std / 1 + mean per channel).
void wan_vae_denormalize(float* latent, int h, int w);

// Block 3's upsample, block 4 and the head computed in row bands with recomputed overlaps: the same bytes as the
// one-pass decode in less device memory.  ON for every card (coordinator 11-10: one path for small and big cards);
// off only for the one-pass reference that the byte-identity test compares against.
void wan_vae_set_streaming(WanVae* vae, bool on);

// Test hook: called with each stage's output (host copy, pixel-major [H][W][C]) during the next decodes; names as in
// tools/image/gate/vae_ref.py (conv_in, mid_attn, mid, up0..up4, out).  An empty function turns it off.
void wan_vae_set_dump(WanVae* vae, std::function<void(const char*, const float*, int, int, int)> dump);

void wan_vae_close(WanVae* vae);

}  // namespace strata::image
