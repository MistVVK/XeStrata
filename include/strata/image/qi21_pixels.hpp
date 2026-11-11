// include/strata/image/qi21_pixels.hpp - the decoded picture to 8-bit pixels in Strata's own code (step 5), as
// stable-diffusion.cpp 2988060 does it (src/core/util.cpp tensor_to_sd_image -> src/runtime/preprocessing.hpp
// preprocessing_float_to_u8): per value 0 at or below 0, 255 at or above 1, else (uint8)(v * 255 + 0.5) with the
// multiply and the add rounded separately (sd.cpp's x86 build; no fused multiply-add); planar [C][H][W] in,
// interleaved [H][W][C] out.  The latent's preparation for the VAE (diffusion_to_vae_latents) is
// wan_vae_denormalize (include/strata/image/wan_vae.hpp, the step 2 decoder's).
#pragma once

#include <cstdint>

namespace strata::image {

uint8_t qi21_float_to_u8(float v);

// planar w x h x c floats (c = 3 or 4) -> interleaved bytes
void qi21_picture_to_u8(const float* planar, int w, int h, int c, uint8_t* out);

}  // namespace strata::image
