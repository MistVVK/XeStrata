// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2024 ZhengPeng
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/image/aux_kernels.hpp - the element kernels of the image engine's auxiliary models (Florence-2's
// DaViT and BART, BiRefNet's Swin and decoder): their products go through prefill::Gemm, their attention through
// qi_sdpa_ex, these do the rest on the GPU.  Activations are channels-last ([pixel or token][channel]); FP32 unless a
// name ends in 16 (FP16 bits).
#pragma once

#include "strata/image/product_store.hpp"

#include <cstdint>

namespace strata::image {

/// y16 [H][W][3] = (rgba[...] / 255 - mean[c]) / std[c] of an RGBA8 picture's RGB on the GPU (its alpha unread).
void aux_rgb_norm16(const uint8_t* rgba, int64_t W, int64_t H, const float* mean, const float* std, uint16_t* y16,
                    void* stream);

/// y [OH][OW][O] (FP32, no bias) = the k x k convolution (stride `stride`, zero padding `pad`) of x16 [H][W][C] by
/// w16 [O][k][k][C], each output pixel's sums at once (O <= kConvFewOut; the outputs past `real` 0, the padding's): a
/// convolution with few outputs, whose unfolded patches would be read for a product of O columns only.
constexpr int64_t kConvFewOut = 8;
void aux_conv_few16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, const uint16_t* w16, int64_t O,
                    int64_t real, int k, int stride, int pad, int64_t OH, int64_t OW, float* y, void* stream);

// Depthwise convolution on channels-last rows, with optional bias, activation and residual.
// Exactly one of x / x16 and w / w16 is set; output has the same type as the input.
// norm_w selects the fused FP32 3 x 3, stride 1, pad 1 residual and row LayerNorm, with an FP16 second output;
// window selects that output's window order. The ordinary residual also requires FP32 input.
struct Depthwise {
    const float* x = nullptr;
    const float* w = nullptr;
    const uint16_t* x16 = nullptr;
    const uint16_t* w16 = nullptr;
    const float* bias = nullptr;
    const float* norm_w = nullptr;
    const float* norm_b = nullptr;
    float* out = nullptr;
    uint16_t* out16 = nullptr;
    int64_t h = 0, wdim = 0, c = 0, oh = 0, ow = 0;
    int k = 3, stride = 1, pad = 1, dilation = 1;
    int64_t window = 0;
    float eps = 0.f;
    ImageAct act = ImageAct::none;
    bool residual = false;
};
void image_depthwise(const Depthwise& d, void* stream);

/// LayerNorm of x (+ r, where r is not null) per row of C, affine w, b: to y (FP32, may be x) and y16 (FP16), each
/// where not null.
void aux_layer_norm(const float* x, const float* r, const float* w, const float* b, int64_t T, int64_t C, float eps,
                    float* y, uint16_t* y16, void* stream);
/// y = x + the depthwise 3 x 3 convolution of x [H][W][C] (w [C][9], b [C]; y not x), then y16 = its LayerNorm (nw,
/// nb) in aux_window16's order of windows of o_ws (as it is at 0): DaViT's positional convolution and the norm after
/// it in one pass.
void aux_dwconv3_norm16(const float* x, const float* w, const float* b, const float* nw, const float* nb, int64_t H,
                        int64_t W, int64_t C, int64_t o_ws, float eps, float* y, uint16_t* y16, void* stream);
/// x [H][W][C] += r + rb (r's rows in aux_window16's order of windows of r_ws where r_ws > 0; rb [C], none where
/// null), then, where w is not null, y16 = its LayerNorm (w, b) in the windows' order of o_ws (as it is at 0): a
/// residual's product, its bias and the next norm in one pass.
void aux_residual_norm16(float* x, const float* r, const float* rb, const float* w, const float* b, int64_t H, int64_t W,
                         int64_t C, int64_t r_ws, int64_t o_ws, float eps, uint16_t* y16, void* stream);
/// aux_layer_norm of x + r + rb (rb [C], a product's bias left out of it; none where null): a post-norm's residual
/// and its product's bias in its pass.  y may be x.
void aux_add_norm(const float* x, const float* r, const float* rb, const float* w, const float* b, int64_t T, int64_t C,
                  float eps, float* y, uint16_t* y16, void* stream);
/// aux_cache_put of a row's keys and values (at koff and voff) into their caches in one pass, at the row the device's
/// `at` [1] says (a recorded step's position).
void aux_cache_put_kv(const uint16_t* src16, int64_t B, int64_t ld, int64_t koff, int64_t voff, int64_t C,
                      uint16_t* kc, uint16_t* vc, int64_t cap, const int32_t* at, void* stream);
/// aux_layer_norm of x [T][C] (no residual) then GELU (erf's), into y16 (FP16).
void aux_layer_norm_gelu16(const float* x, const float* w, const float* b, int64_t T, int64_t C, float eps,
                           uint16_t* y16, void* stream);
/// aux_layer_norm of x [H][W][C] into y16 in aux_window16's order (no shift, H and W whole windows of ws): the norm
/// and the windows in one pass.
void aux_layer_norm_window16(const float* x, const float* w, const float* b, int64_t H, int64_t W, int64_t C,
                             int64_t ws, float eps, uint16_t* y16, void* stream);

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
/// The same from FP16 rows (AuxNet::product16's, the bias in), C a multiple of 32: q and k transposed through local
/// memory, a tile at a time.
void aux_channel_split16(const uint16_t* qkv16, int64_t N, int64_t C, int64_t G, float scale, uint16_t* qT16,
                         uint16_t* kT16, uint16_t* v16, void* stream);
/// y16 [N][C] FP16 = yg [G][N][C / G] (the channel attention's groups' outputs, each after the last), merged.
void aux_channel_merge16(const float* yg, int64_t N, int64_t C, int64_t G, uint16_t* y16, void* stream);
void aux_channel_split(const float* qkv, int64_t N, int64_t C, int64_t G, float scale, uint16_t* qT16, uint16_t* kT16,
                       uint16_t* v16, void* stream);

/// The softmax of attention scores s [batch][heads][rows][cols] times `scale`, with optional additive terms: `bias`
/// [heads][rows][cols] (every batch's), `mask` [mask_n][rows][cols] (batch b's b % mask_n), and SAM 3's grid box bias
/// `grid` [rows - skip][2 G][heads] (key y G + x adds its y-th and its G + x-th terms, rows below `skip` none) -> p16,
/// FP16.  A null term is absent.
void aux_softmax_attn16(const float* s, int64_t batch, int64_t heads, int64_t rows, int64_t cols, float scale,
                        const float* bias, const float* mask, int64_t mask_n, const float* grid, int64_t G,
                        int64_t skip, uint16_t* p16, void* stream);
/// An attention's heads apart: x [rows][ld] (head h's hd channels at h * hd) -> y [heads][rows][hd], or (transpose)
/// [heads][hd][rows] (a product's keys' values); FP16.
void aux_heads16(const uint16_t* x, int64_t rows, int64_t heads, int64_t hd, int64_t ld, bool transpose, uint16_t* y,
                 void* stream);
/// The batched heads apart: x [batch][rows][ld] (batch stride xbatch) -> y [batch][heads][rows][hd], or (transpose)
/// [batch][heads][hd][rows]; FP16.
void aux_heads_batch16(const uint16_t* x, int64_t batch, int64_t rows, int64_t heads, int64_t hd, int64_t ld,
                       int64_t xbatch, bool transpose, uint16_t* y, void* stream);
/// The heads' outputs together: o [heads][rows][hd] (FP32) -> y [rows][heads hd] (FP16).
void aux_heads_merge16(const float* o, int64_t rows, int64_t heads, int64_t hd, uint16_t* y, void* stream);
/// The batched heads' outputs together: o [batch][heads][rows][hd] (FP32) -> y [batch][rows][ld] (head h's hd at
/// h * hd, batch stride ybatch); FP16.
void aux_heads_merge_batch16(const float* o, int64_t batch, int64_t rows, int64_t heads, int64_t hd, int64_t ld,
                             int64_t ybatch, uint16_t* y, void* stream);

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
/// aux_rope2d_split16 from FP16 rows (AuxNet::product16's), `bias` [3 heads hd] added where not null.
void aux_rope2d_split16h(const uint16_t* qkv16, const float* bias, int64_t N, int64_t heads, int64_t hd, int64_t grid,
                         float pscale, float theta, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream);
/// y16 [T][n] = act(x16 + bias[n]) (no bias where null; GELU erf's); y16 may be x16.
void aux_bias_act16(const uint16_t* x16, const float* bias, int64_t T, int64_t n, ImageAct act, uint16_t* y16,
                    void* stream);
/// y16 [T][n] = y + bias[n] (none where null), FP32 narrowed.
void aux_narrow_bias16(const float* y, const float* bias, int64_t T, int64_t n, uint16_t* y16, void* stream);

/// A 2 x 2 transposed convolution of stride 2 from its product y [H * W][(dy, dx, O)]: out [2H][2W][O] = y + bias
/// (GELU of it with `gelu`), to out (FP32) and out16, each where not null.
void aux_deconv2_scatter(const float* y, int64_t H, int64_t W, int64_t O, const float* bias, bool gelu, float* out,
                         uint16_t* out16, void* stream);

/// y16 [H][W][C] = feat + prev [h][w][C] resized to H x W by the nearest pixel (torch's `nearest`: floor(i * h / H)).
void aux_up_add16(const float* prev, int64_t h, int64_t w, const float* feat, int64_t H, int64_t W, int64_t C,
                  uint16_t* y16, void* stream);

/// masks [K][H][W] = 255 where the sigmoid of logits [K][h][w], resized to H x W bilinearly (torch's
/// align_corners=False; the probabilities resized, or with `as_logits` the logits themselves, as SAM 3's video
/// outputs are), passes `threshold` (with `as_logits`, a logit), else 0.
void aux_mask_resize(const float* logits, int64_t K, int64_t h, int64_t w, int64_t H, int64_t W, bool as_logits,
                     float threshold,
                     uint8_t* masks, void* stream);

/// SAM 3's box relative position bias: for each of Q boxes (xyxy in [0, 1]), the rows' and the columns' distances
/// to its edges (i / H - y0, i / H - y1; j / W - x0, j / W - x1), times 8, as sign * log2(|d| + 1) / 3, through two
/// MLPs (2 -> hidden, ReLU, -> heads; `mlp`: the rows' w1 [hidden][2], b1, w2 [heads][hidden], b2, then the
/// columns'), into tmp [Q][H + W][heads] (the rows' terms, then the columns': aux_softmax_attn16's grid), and, where
/// `bias` is given, summed into bias [heads][Q + 1][H * W] (its row 0, the presence token's, zero).
void aux_box_bias(const float* boxes, int64_t Q, int64_t H, int64_t W, const float* mlp, int64_t hidden, int64_t heads,
                  float* tmp, float* bias, void* stream);
/// SAM 3's decoder's query positions from boxes [Q][4] (cx, cy, w, h): sin16 [Q][4F] the sine embeddings of y, x, w
/// and h (F channels each, sin and cos interleaved at 10000^(2i/F)), xyxy [Q][4] their corners.
void aux_box_sine16(const float* boxes, int64_t Q, int64_t F, uint16_t* sin16, float* xyxy, void* stream);
/// boxes [n] = sigmoid(delta + inverse_sigmoid(boxes)) (the inverse clamped as the model's, 1e-3), in place.
void aux_box_refine(float* boxes, const float* delta, int64_t n, void* stream);

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

// BiRefNet's refine_foreground (Photoroom's blur fusion, as its image_proc.py's GPU version runs it; MIT): the
// foreground's colors estimated from the picture and the alpha, so a cut-out laid on another background carries no
// fringe of the old one.  Two rounds of box means (90, then 6 pixels), each a stack, the means and their combination.

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

// SAM 3.1's video tracker (sam3_track.cpp)

/// y [H][W][C] = the depthwise k x k convolution of x [H][W][C] (zero padding k / 2), w [C][k][k], b [C].
void aux_dwconv(const float* x, const float* w, const float* b, int64_t H, int64_t W, int64_t C, int k, float* y,
                void* stream);
/// x [n][C] += g [C] * y [n][C] (a block's scaled residual).
void aux_scale_add(float* x, const float* y, const float* g, int64_t n, int64_t C, void* stream);
/// x [n] = v.
void aux_fill(float* x, int64_t n, float v, void* stream);
/// y16 [N][heads hd] = y (+ add [P][C] at row r % P, where not null) turned by the 2D axial RoPE of
/// aux_rope2d_split16 at the position r % (grid grid): a query's rows, a bucket of P after another.
void aux_rope2d16(const float* y, const float* add, int64_t P, int64_t N, int64_t heads, int64_t hd, int64_t grid,
                  float theta, uint16_t* y16, void* stream);
/// The memory attention's keys of B buckets of N rows each: a bucket's first nsp rows the frames' memories (P rows
/// a frame) + sine [P][C] + tpos [frame][C], turned by aux_rope2d16's RoPE at r % P; the rest its object pointers +
/// ptr_pos [(r - nsp) / per][C], not turned.  y [B][N][C] -> k16.  tpos null: sine is the spatial rows' whole
/// positions [nsp][C] (the probe's).
void aux_mem_key16(const float* y, int64_t B, int64_t N, int64_t nsp, int64_t P, const float* sine, const float* tpos,
                   const float* ptr_pos, int64_t per, int64_t heads, int64_t hd, int64_t grid, float theta,
                   uint16_t* k16, void* stream);
/// The memory encoder's input of B buckets of M slots: y16 [B][H][W][2M], channel s the mask of object b M + s
/// (masks [N][h][w] logits resized bilinearly to H x W, align_corners false; its sigmoid taken before the resize
/// with `sigmoid_first`, else after) as 2 sigmoid - 1, channel M + s its cond [N] value; a slot past N, or whose
/// valid [N] is 0 (a removed object's; valid null: all valid), zeros.  With `live` (the GPU's), only those L slots
/// of one bucket: y16 [H][W][2L], channel c the mask of slot live[c], L + c its cond (the zero channels of the empty
/// slots left out, for a convolution whose weights leave them out too).
void aux_mux_masks16(const float* masks, int64_t N, int64_t h, int64_t w, int64_t H, int64_t W, bool sigmoid_first,
                     const float* cond, const float* valid, int64_t M, int64_t B, uint16_t* y16, void* stream,
                     const int32_t* live = nullptr, int64_t L = 0);
/// y [n] = 1 where x > 0, else 0.
void aux_binarize(const float* x, int64_t n, float* y, void* stream);
/// y16 [R][C] = src [T][R][C]'s rows r of block r % T (each row its own hypernetwork's output).
void aux_mod_select16(const float* src, int64_t T, int64_t R, int64_t C, uint16_t* y16, void* stream);

// The feature maps of the control images' networks (AuxMapNet: HED, line art, M-LSD, Depth Anything V2's head, YOLOX,
// DWPose): [H][W][C] FP16, C a multiple of 8 (the channels past a layer's own zero).

/// dst16 [n][O] (row stride ld, at channel off) = act(y [n][O] + bias) (+ res16 [n][O] after the activation), each of
/// bias and res16 none where null: a convolution's FP32 sums finished.
void aux_map_store16(const float* y, int64_t n, int64_t O, const float* bias, ImageAct act, const uint16_t* res16,
                     uint16_t* dst16, int64_t ld, int64_t off, void* stream);
/// y16 [OH][OW][C] = act(the depthwise k x k convolution of x16 [H][W][C] of stride s and zero padding p by w16
/// [C][k][k], + bias [C]).
void aux_dwconv16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, const uint16_t* w16, const float* bias, int k,
                  int s, int p, int64_t OH, int64_t OW, ImageAct act, uint16_t* y16, void* stream);
/// The k x k max pool of stride s and padding p (the padding never the maximum) of x16 [H][W][C] to [OH][OW][C].
void aux_maxpool16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int k, int s, int p, int64_t OH, int64_t OW,
                   uint16_t* y16, void* stream, int64_t ld = 0, int64_t off = 0, uint16_t* copy = nullptr);
/// Letterbox RGBA to normalized RGB, or RGB to YOLOX's unnormalized BGR Focus layout.
void aux_letterbox16(const uint8_t* src, int64_t w, int64_t h, int64_t nw, int64_t nh, int64_t left, int64_t top,
                     int64_t iw, int64_t ih, float ratio, bool focus, uint16_t* out, void* stream);
/// Two maps concatenated, with optional nearest-neighbour 2x upsampling of the first.
void aux_concat2_16(const uint16_t* a, int64_t ca, bool up, const uint16_t* b, int64_t cb, int64_t h, int64_t w,
                    uint16_t* dst, void* stream);
/// y16 [OH][OW][C] = x16 [H][W][C] resized bilinearly (torch's interpolate, align_corners as given); `add`: added to
/// what y16 holds.
void aux_bilinear16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int64_t OH, int64_t OW, bool align_corners,
                    bool add, uint16_t* y16, void* stream);
/// A transposed convolution whose kernel is its stride k, from its product y [H W][k][k][O]: y16 [k H][k W][O] =
/// act(y + bias).
void aux_deconv_scatter16(const float* y, int64_t H, int64_t W, int64_t k, int64_t O, const float* bias, ImageAct act,
                          uint16_t* y16, void* stream);
/// dst16 [n][ld] at channels off .. off + C = src16 [n][C] (a concatenation's part).
void aux_copy_channels16(const uint16_t* src16, int64_t n, int64_t C, uint16_t* dst16, int64_t ld, int64_t off,
                         void* stream);
/// y16 = a16 + b16, n values; y16 may be either.
void aux_add16h(const uint16_t* a16, const uint16_t* b16, int64_t n, uint16_t* y16, void* stream);
/// y16 [H + 2p][W + 2p][C] = x16 padded by reflection (torch's ReflectionPad2d: the edge not repeated).
void aux_pad_reflect16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int64_t p, uint16_t* y16, void* stream);
/// A strided transposed convolution's input for a plain one: x16 [H][W][C] with s - 1 zeros between its pixels, `lo`
/// zero rows and columns before and `hi` after.
void aux_zero_insert16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int64_t s, int64_t lo, int64_t hi,
                       uint16_t* y16, void* stream);
/// y16 [H + top + bottom][W + left + right][C] = x16 with zeros around it.
void aux_pad16(const uint16_t* x16, int64_t H, int64_t W, int64_t C, int64_t top, int64_t left, int64_t bottom,
               int64_t right, uint16_t* y16, void* stream);
/// x16 [n][C] *= gate [C] (a channel attention's weights, FP32), in place.
void aux_scale_channels16(uint16_t* x16, int64_t n, int64_t C, const float* gate, void* stream);
/// mean [C] (FP32) = the average of x16 [n][C] over its n pixels.
void aux_col_mean16(const uint16_t* x16, int64_t n, int64_t C, float* mean, void* stream);

}  // namespace strata::image
