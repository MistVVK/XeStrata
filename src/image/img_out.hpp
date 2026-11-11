// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/img_out.hpp - the image engine's output-side row parts, the copies gathered here: the DiT's final
// layer from the patch's order into the tokens' (patch_to_tokens: Anima's to_tokens and SeedVR2's patch_out were the
// same transpose), and the VAE's decoded values into RGBA planes (to_rgba: Qwen-Image's and Mage-Flow's VAEs, the
// input's layout a TilePlanes as tile_sides reads it).  The input side is img_start.hpp.
#pragma once

#include "tile_sides.hpp"

#include <cstdint>

namespace strata::image {

/// in [N][4][L] (token t's patch position p, channel c at t * 4L + p * L + c) -> out [N][L][4] in the tokens' order
/// (value 4c + p): the final layer's [hw][4L] as the latent's tokens read it.
void patch_to_tokens(const float* in, int64_t N, int64_t L, float* out, void* s);

/// rgba [4][HW] = clamp((x + 1) / 2, 0, 1), the alpha 1; x the decoded values in `layout` (Planes [C][HW],
/// Interleaved [HW][C]), C the model's channels (3).
void to_rgba(const float* x, int64_t HW, int64_t C, TilePlanes layout, float* rgba, void* s);

}  // namespace strata::image
