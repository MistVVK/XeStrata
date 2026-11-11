// include/strata/image/qi21_dit.hpp - Strata's Qwen-Image-2.1 DiT without ggml (step 4, D4): one denoising step,
// latent + text condition + timestep -> velocity, on CUDA / HIP.
//
// The weights come from the DiT GGUF through our own reader (strata/artifact/gguf_reader.hpp) into one device buffer
// as the file's bytes (Q8_0 / Q4_K / Q6_K ... blocks; q, k and v of a block laid one after the other, so one GEMM
// reads all three); the 1-D norm weights as f32.  Every multiply is dit_gemm (the weight decoded to f16, activations
// to f16, float32 sums); the element-wise steps and the attention are src/image/dit/qi21_kernels.cu.  The text prefix
// (causal within the text, modulation row t = 0) runs once per condition and leaves each block's K / V behind, as the
// native runner in stable-diffusion.cpp does; a step with the same condition reuses it.
//
// Test program only so far (strata-dit-step); nothing in a build or in sd-cli calls it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::image {

struct Qi21Dit;

struct Qi21DitStats {
    double step_ms = 0;          // this call, the stream's events (prefix pass included when it ran)
    double prefix_ms = 0;        // the text prefix pass when it ran in this call, else 0
    bool prefix_ran = false;
    size_t weight_bytes = 0;     // weights on the device (raw blocks + f32 norms)
    size_t work_bytes = 0;       // activations, K / V, the prefix cache, the GEMM's f16 buffers
    size_t peak_bytes = 0;       // weight_bytes + work_bytes at their largest so far
};

// device < 0: the current device.  nullptr on failure (*err says why).
Qi21Dit* qi21_dit_open(const std::string& gguf_path, int device, std::string* err);

int qi21_dit_in_channels(const Qi21Dit* dit);
int qi21_dit_out_channels(const Qi21Dit* dit);
int qi21_dit_context_dim(const Qi21Dit* dit);

// latent: [in_channels][H][W] f32 (sd.cpp's DiT input, x), cond: [L][context_dim] f32, t: the timestep as sd.cpp
// passes it; velocity: [out_channels][H][W] f32.  "" on success.
std::string qi21_dit_step(Qi21Dit* dit, const float* latent, int W, int H, const float* cond, int L, float t,
                          float* velocity, Qi21DitStats* stats = nullptr);

std::string qi21_dit_open_report(const Qi21Dit* dit);

void qi21_dit_close(Qi21Dit* dit);

}  // namespace strata::image
