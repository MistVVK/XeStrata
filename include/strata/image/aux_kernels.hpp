// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/aux_kernels.hpp - the element kernels of the image engine's auxiliary models (Florence-2's
// DaViT and BART, BiRefNet's Swin and decoder): their products go through prefill::Gemm, their attention through
// qi_sdpa_ex, these do the rest on the GPU.  Activations are channels-last ([pixel or token][channel]); FP32 unless a
// name ends in 16 (FP16 bits).
#pragma once

#include <cstdint>

namespace strata::image {

/// y16 [H][W][3] = (rgba[...] / 255 - mean[c]) / std[c] of an RGBA8 picture's RGB on the GPU (its alpha unread).
void aux_rgb_norm16(const uint8_t* rgba, int64_t W, int64_t H, const float* mean, const float* std, uint16_t* y16,
                    void* stream);

/// The rows p0 .. p0 + n - 1 of a k x k convolution's patches (stride `stride`, zero padding `pad` on every side) of
/// x16 [H][W][C], to col16 [n][k][k][C] (the weights' [O][k][k][C] order); OW the output's width.
void aux_im2col16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int k, int stride, int pad, int64_t OW,
                  int64_t p0, int64_t n, uint16_t* col16, void* stream);

/// y = x + the depthwise 3 x 3 convolution of x (zero padding 1, w [C][9], bias b [C]): DaViT's position encoding
/// with its residual.  x and y [H][W][C], not the same buffer.
void aux_dwconv3_add(const float* x, const float* w, const float* b, int64_t H, int64_t W, int64_t C, float* y,
                     void* stream);

/// LayerNorm of x (+ r, where r is not null) per row of C, affine w, b: to y (FP32, may be x) and y16 (FP16), each
/// where not null.
void aux_layer_norm(const float* x, const float* r, const float* w, const float* b, int64_t T, int64_t C, float eps,
                    float* y, uint16_t* y16, void* stream);

/// Windows of ws x ws over the grid padded to Hp x Wp (multiples of ws; the padding zero), cyclically shifted by
/// `shift` (Swin's shifted windows; 0: none): dst16 [windows][ws * ws][C] (row-major windows, row-major in each) from
/// src16 [H][W][C].
void aux_window16(const uint16_t* src16, int64_t H, int64_t W, int64_t C, int64_t ws, int64_t shift, int64_t Hp,
                  int64_t Wp, uint16_t* dst16, void* stream);
/// x [H][W][C] += y back from aux_window16's order (the same windows, shift and padding; the padding's rows dropped).
void aux_unwindow_add(float* x, const float* y, int64_t H, int64_t W, int64_t C, int64_t ws, int64_t shift, int64_t Hp,
                      int64_t Wp, void* stream);

/// DaViT's channel attention's operands from qkv [N][3C] (q, k, v, each G groups of C / G): qT16 and kT16
/// [G][C / G][N] (q times `scale`), v16 [G][N][C / G].
void aux_channel_split(const float* qkv, int64_t N, int64_t C, int64_t G, float scale, uint16_t* qT16, uint16_t* kT16,
                       uint16_t* v16, void* stream);

/// p16 [rows][cols] = softmax of each row of s.
void aux_softmax16(const float* s, int64_t rows, int64_t cols, uint16_t* p16, void* stream);

/// y [n][C] = tok[ids[i]] * scale + pos[pos_ids[i]] (FP16 tables); tok null: y += pos[pos_ids[i]].
void aux_embed(const int32_t* ids, const int32_t* pos_ids, const uint16_t* tok, const uint16_t* pos, int64_t n,
               int64_t C, float scale, float* y, void* stream);

/// x [T][3C] (a fused q, k, v product) -> q16, k16, v16 [T][C] each.
void aux_split3_16(const float* x, int64_t T, int64_t C, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream);

/// The decoder's cache: rows b < B of src16 (row stride ld) from column off, C wide, to cache [B][cap][C] at row `at`.
void aux_cache_put(const uint16_t* src16, int64_t B, int64_t ld, int64_t off, int64_t C, uint16_t* cache, int64_t cap,
                   int64_t at, void* stream);
/// dst [B][cap][C] row b's first `len` rows = src's row from[b]'s (a beam search's reorder; dst is not src).
void aux_cache_gather(const uint16_t* src, const int32_t* from, int64_t B, int64_t cap, int64_t len, int64_t C,
                      uint16_t* dst, void* stream);

/// mask [H][W] = 255 where pixel (x, y) is inside one of the n polygons or within half a pixel of an edge (Pillow's
/// filled polygon with its one-pixel outline), else 0.  xy holds the vertices (x, y pairs) of polygon i from
/// starts[i] to starts[i + 1] (counts of points).
void aux_polygon_fill(const float* xy, const int32_t* starts, int64_t n, int64_t W, int64_t H, uint8_t* mask,
                      void* stream);

/// y = max(x, 0) (y may be x, or null) and y16 its FP16 copy (or null), n elements.
void aux_relu(const float* x, int64_t n, float* y, uint16_t* y16, void* stream);

/// y16 = a + b, n elements (a stream plus its positions, as a query or key's input).
void aux_add16(const float* a, const float* b, int64_t n, uint16_t* y16, void* stream);

/// x [H][W][C] += table [(y % th) * tw + x % tw][C]: a position table tiled over the grid (no interpolation).
void aux_tile_add(float* x, const float* table, int64_t H, int64_t W, int64_t C, int64_t th, int64_t tw, void* stream);

/// q16, k16, v16 [N][heads * hd] from qkv [N][3 * heads * hd], q and k turned by a 2D axial RoPE: token t's position
/// in a grid x grid square of the tokens (t mod grid^2: x = its column, y = its row), times pscale; channel pairs
/// (2p, 2p + 1) of each head turn by x * f_p for p < hd / 4, else y * f_(p - hd / 4), f_i = theta^(-4i / hd).
void aux_rope2d_split16(const float* qkv, int64_t N, int64_t heads, int64_t hd, int64_t grid, float pscale, float theta,
                        uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream);

/// A 2 x 2 transposed convolution of stride 2 from its product y [H * W][(dy, dx, O)]: out [2H][2W][O] = y + bias
/// (GELU of it with `gelu`), to out (FP32) and out16, each where not null.
void aux_deconv2_scatter(const float* y, int64_t H, int64_t W, int64_t O, const float* bias, bool gelu, float* out,
                         uint16_t* out16, void* stream);

/// y16 [H][W][C] = feat + prev [h][w][C] resized to H x W by the nearest pixel (torch's `nearest`: floor(i * h / H)).
void aux_up_add16(const float* prev, int64_t h, int64_t w, const float* feat, int64_t H, int64_t W, int64_t C,
                  uint16_t* y16, void* stream);

/// GroupNorm of x [N][C] (G groups of C / G channels over all N pixels, eps), affine w, b, then ReLU with `relu`: to
/// y (FP32, may be x) and y16, each where not null.  `part` scratch of 3 * G * 64 floats.
void aux_group_norm(const float* x, int64_t N, int64_t C, int64_t G, const float* w, const float* b, float eps,
                    bool relu, float* y, uint16_t* y16, float* part, void* stream);

/// masks [K][H][W] = 255 where the sigmoid of logits [K][h][w], resized to H x W bilinearly (torch's
/// align_corners=False, the probabilities resized), passes `threshold`, else 0.
void aux_mask_resize(const float* logits, int64_t K, int64_t h, int64_t w, int64_t H, int64_t W, float threshold,
                     uint8_t* masks, void* stream);

/// SAM 3's box relative position bias: for each of Q boxes (xyxy in [0, 1]), the rows' and the columns' distances
/// to its edges (i / H - y0, i / H - y1; j / W - x0, j / W - x1), times 8, as sign * log2(|d| + 1) / 3, through two
/// MLPs (2 -> hidden, ReLU, -> heads; `mlp`: the rows' w1 [hidden][2], b1, w2 [heads][hidden], b2, then the
/// columns'), summed into bias [heads][Q + 1][H * W] (its row 0, the presence token's, zero).  tmp: Q * (H + W) *
/// heads floats.
void aux_box_bias(const float* boxes, int64_t Q, int64_t H, int64_t W, const float* mlp, int64_t hidden, int64_t heads,
                  float* tmp, float* bias, void* stream);

/// x [H][W][C] -> y [((H + 1) / 2) ((W + 1) / 2)][4C]: Swin's patch merging's gather, each 2 x 2 of the grid in the
/// order (0, 0), (1, 0), (0, 1), (1, 1) (row offset first); an odd side padded with zeros, as Swin pads it.
void aux_merge2(const float* x, int64_t H, int64_t W, int64_t C, float* y, void* stream);

/// dst [H][W] at channels c0 .. c0 + C of rows ld wide = src [h][w][C] resized bilinearly (torch's interpolate,
/// align_corners as given; the same size: a copy), added to what is there with `add`.
void aux_resize_into(const float* src, int64_t h, int64_t w, int64_t C, float* dst, int64_t H, int64_t W, int64_t ld,
                     int64_t c0, bool align_corners, bool add, void* stream);

/// The sampled patches of a modulated deformable k x k convolution (stride 1, zero padding `pad`, torchvision's
/// deform_conv2d) of x16 [H][W][C] for the output pixels p0 .. p0 + n - 1: col16 [n][k][k][C] = 2 sigmoid(m) times
/// x at the tap's position moved by its offset (bilinear, zero outside), om [H W][3 k^2]: the taps' (dy, dx) pairs,
/// then their modulators' logits m.
void aux_deform_col16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int k, int pad, const float* om,
                      int64_t p0, int64_t n, uint16_t* col16, void* stream);

/// y [rows][n] (row stride ld) += bias (null: none), then ReLU with `relu`.
void aux_bias_act(float* y, int64_t rows, int64_t n, int64_t ld, const float* bias, bool relu, void* stream);

/// mean [C] = the average of x [P][C] over its P rows.
void aux_col_mean(const float* x, int64_t P, int64_t C, float* mean, void* stream);

/// x [P][C] *= sigmoid(logit[p]), each pixel's channels by its own gate.
void aux_scale_by_sigmoid(float* x, int64_t P, int64_t C, const float* logit, void* stream);

/// y [SH / g][SW / g][3 g g] FP16 = img [SH][SW][3] rearranged as BiRefNet's image2patches ('b c (hg h) (wg w) -> b
/// (c hg wg) h w'): channel (c, gy, gx) of pixel (y, x) the picture's (gy (SH / g) + y, gx (SW / g) + x).
void aux_image_patches16(const float* img, int64_t SH, int64_t SW, int64_t g, uint16_t* y16, void* stream);

/// y [H][W][3] (FP32) = (rgba[...] / 255 - mean[c]) / std[c] of an RGBA8 picture's RGB.
void aux_rgb_norm(const uint8_t* rgba, int64_t W, int64_t H, const float* mean, const float* std, float* y,
                  void* stream);

/// alpha [H][W] = 255 sigmoid of logits [h][w] resized bicubically (torch's interpolate, align_corners=False, A
/// -0.75, the border replicated), rounded.
void aux_alpha_bicubic(const float* logits, int64_t h, int64_t w, int64_t W, int64_t H, uint8_t* alpha, void* stream);

// BiRefNet's refine_foreground (Photoroom's blur fusion, as its image_proc.py's GPU version runs it): the foreground's
// colors estimated from the picture and the alpha, so a cut-out laid on another background carries no fringe of the
// old one.  Two rounds of box means (90, then 6 pixels), each a stack, the means and their combination.

/// x [n][7] = (a, F a, B (1 - a)) of the picture rgba (RGBA8) and its alpha (a = alpha / 255), F and B the picture's
/// colors (fb null) or fb [n][6]'s (F, then B: the first round's).
void aux_fg_stack(const uint8_t* rgba, const uint8_t* alpha, const float* fb, int64_t n, float* x, void* stream);

/// y [H][W][C] = the mean of x over a k x k box (cv2.blur's: k / 2 - 1 before and k / 2 after a pixel when k is even,
/// k / 2 each side when odd), the edges replicated; by rows into tmp, then by columns.
void aux_box_mean(const float* x, int64_t H, int64_t W, int64_t C, int64_t k, float* tmp, float* y, void* stream);

/// The round's foreground from the means y [n][7] (aux_fg_stack's, box-averaged): F = clamp(mF + a (I - a mF - (1 - a)
/// mB)), mF = y[1..3] / (y[0] + 1e-5), mB = y[4..6] / (1 - y[0] + 1e-5); into fb [n][6] (F, mB) when given, and as
/// RGB8 into rgb (255 F truncated, as torch's to(uint8)) when given.
void aux_fg_combine(const uint8_t* rgba, const uint8_t* alpha, const float* y, int64_t n, float* fb, uint8_t* rgb,
                    void* stream);

}  // namespace strata::image
