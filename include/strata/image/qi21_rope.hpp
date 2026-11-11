// include/strata/image/qi21_rope.hpp - Qwen-Image-2.1's token positions and RoPE table, as stable-diffusion.cpp
// 2988060 builds them for a picture without reference images (src/image/dit/qi21_rope.cpp):
//   QwenImage21Layout::build (model/diffusion/qwen_image_2_1.hpp): text tokens at positions (p, p, p), p = 0..L-1;
//     then the image grid, token h*W + w at (L, h - (H - H/2), w - (W - W/2));
//   Rope::embed_nd (model/common/rope.hpp) with axes 16 / 56 / 56 and theta 10000: per token and axis, for each
//     frequency omega = 1 / powf(theta, linspace(0, (d - 2) / d, d / 2)[i]) and angle = position * omega (float),
//     the 2x2 matrix [cos, -sin; sin, cos]; per token 64 matrices = 256 floats, axis 0 first.
// Every operation is the float operation sd.cpp performs, with the C library's cosf / sinf / powf: the table is
// byte-identical to sd.cpp's built with the same C library (checked on glibc; another C library may differ in the
// last bit of a few entries).
#pragma once

#include <cstdint>
#include <vector>

namespace strata::image {

// positions[3 * i + a]: token i, axis a (text tokens first, then H x W image tokens)
std::vector<float> qi21_positions(int64_t text_len, int64_t H, int64_t W);

// the table for those positions: 256 floats a token (64 x [cos, -sin, sin, cos]), as finish_rope_pe returns it
std::vector<float> qi21_rope_table(const std::vector<float>& positions);

}  // namespace strata::image
