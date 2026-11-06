// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/mf_vae.hpp - Mage-Flow's VAE (Mage-VAE) decoder on the GPU: the DiT's latent tokens [h w][128]
// to an RGBA image [4][16 h][16 w] in 0 .. 1.
//
// The decoder is the reference's DConvDenoiser run once at t = 0 from zero noise: the CoD decoder turns the latent into
// a condition [h w][384] (3 x 3 convolutions, group norms, self-attention in 32 x 32 windows), 21 DiCo blocks (their
// adaLN modulation constant at t = 0, worked out when loaded) refine a per-patch state from it, and a small MLP takes
// each pixel of a 16 x 16 patch to RGB.  The activations are channels-last ([pixel][channel], FP32) at the latent's
// size; the group norms and the blocks' channel attention take statistics over the whole latent, so it is not tiled
// (2048 x 2048 is 128 x 128 there); the per-pixel MLP runs a band of patches at a time.
//
// The encoder (an edit's reference images, loaded by open_encoder) is the reference's DConvEncoder at t = 0 from a zero
// latent: 16 x 16 patches to 768 channels, two DiCo blocks without modulation, 384 channels fused with the zero
// latent's projection, 21 DiCo blocks (constant modulation), and the posterior's mean and log-variance [h w][256].
//
// The same class decodes with Black Forest Labs' Flux2 VAE (whose latent space Mage-VAE was made to share; MageTrail's
// author decodes with it), as its LDM checkpoint names it (flux2-vae.safetensors) or as diffusers does (FLUX.2-klein's
// vae/, the same weights in bf16): each latent token's 128 values un-normalized by the VAE's batch norm
// statistics and spread over 2 x 2 positions of 32 channels, then the LDM decoder (a 1 x 1 convolution, ResNet blocks
// with group norms, one self-attention over every position in the middle, three nearest 2x upsamplings: 8 x 8 pixels
// a position).  Its activations are channels-last FP32 at each level's size: the largest, 256 channels at the full
// size after the last upsampling, holds 1 GiB at 1024 x 1024.  Its encoder (img2img's and a mask's base image, by
// open_encoder) is the LDM encoder: conv_in, four levels of ResNet blocks with a 3 x 3 convolution of stride 2 after
// each but the last (the image padded by one on the bottom and the right), the middle as the decoder's, norm_out and
// conv_out to the posterior's mean and log-variance, and the 1 x 1 quant_conv; the latent is the mean (diffusers'
// Flux2 pipelines take its argmax), each 2 x 2 positions' 32 channels gathered into a token's 128 and normalized by
// the batch norm's statistics, the decoder's input undone.  It runs on the whole image (not in tiles).
// The widths are the weights' own: MOSS-Image's Flux2 VAE decodes with 96 to 384 channels and encodes with FLUX.2's
// 128 to 512 (the convolutions' bands are the decoder's alone, so a decode is the same with the encoder loaded).
// Where the whole does not fit the room decode is given, it runs the stages up to the deepest level that fits on the
// whole latent (the middle's attention and the group norms there over all of it) and the rest in overlapping tiles (4
// tokens, 64 pixels, on each side shared), blended linearly across the overlap as ComfyUI's tiled decode does; the
// group norms of the tiled levels take each tile's statistics.
//
// SeedVR2's VAE (ByteDance-Seed/SeedVR's video_vae_v3, as Comfy-Org's ema_vae_fp16 names it, diffusers' names) is the
// same LDM VAE inflated to video; an image is a clip of one frame, where its causal convolutions (the first frame
// repeated before it) are 2D ones whose kernels are their time taps summed, as they are read.  Its latent is 16
// channels with no batch norm and no post_quant_conv (a token: 2 x 2 positions of them, 64 values, as the Flux2 VAE's
// 128), and an upsampling is a 1 x 1 convolution to 4 c whose output spread over 2 x 2 positions is the larger image
// (the frames it would add in time left out), then the convolution.
#pragma once

#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::image {

class MfVae {
public:
    MfVae();
    ~MfVae();
    MfVae(const MfVae&) = delete;
    MfVae& operator=(const MfVae&) = delete;

    /// Reads the VAE's safetensors (Mage-VAE: the reference's names, pipeline.* the decoder, student.* the encoder; or
    /// the Flux2 VAE: decoder.* and bn.*) and puts the decoder's weights on the current GPU.  `dry`: read for its sizes
    /// only, nothing put on the GPU.
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false);

    /// rgba [4][16 h][16 w] (device FP32, 0 .. 1) from the DiT's latent tokens z [h w][128] (device FP32), its
    /// scratch within `limit` bytes where it can be (the Flux2 VAE in tiles).
    void decode(const float* z, int64_t h, int64_t w, float* rgba, uint64_t limit = UINT64_MAX);
    /// The image before the clamp, [3][16 h][16 w] (device FP32, about -1 .. 1): the reference's decode output.
    void decode_raw(const float* z, int64_t h, int64_t w, float* rgb);

    /// Puts the encoder's weights (the file open read) on the GPU.  `dry`: their sizes only.
    bool open_encoder(std::string& err, bool dry = false);
    bool has_encoder() const { return f2_ ? !f2_down_.empty() : !enc_blocks_.empty(); }
    /// The file is the Flux2 VAE.
    bool flux2() const { return f2_; }
    uint64_t encoder_bytes() const { return enc_bytes_; }
    /// The posterior's mean and log-variance [h w][256] (device FP32) of rgb [3][16 h][16 w] (device FP32, -1 .. 1).
    /// Mage-VAE's only: the Flux2 VAE's encode gives its latent alone.
    void encode_moments(const float* rgb, int64_t h, int64_t w, float* moments);
    /// The latent tokens z [h w][128] (device FP32) of rgb: the posterior's mean, plus exp(logvar / 2) eps when `eps`
    /// (device [h w][128] standard normal) is given (the reference samples the posterior).  The Flux2 VAE's is the
    /// mean (eps is not read).
    /// The Flux2 VAE's within `limit` bytes of scratch where it can be: in tiles of tokens whose latents are blended
    /// across their overlap where the whole does not fit.
    void encode(const float* rgb, int64_t h, int64_t w, const float* eps, float* z, uint64_t limit = UINT64_MAX);
    /// The GPU memory an encode of an h x w latent works in.
    uint64_t encode_scratch_bytes(int64_t h, int64_t w) const;

    /// The GPU memory a decode of an h x w latent works in (its buffers, besides the weights).
    uint64_t scratch_bytes(int64_t h, int64_t w) const;
    /// The least of it: the Flux2 VAE's in its smallest tiles; Mage-VAE's whole.
    uint64_t least_scratch_bytes(int64_t h, int64_t w) const;
    /// The last decode went in tiles.
    bool last_tiled() const { return tiled_; }
    void drop_scratch();
    uint64_t scratch_held() const { return f2_ ? f2_held_ : cap_hw_ > 0 ? scratch_bytes(cap_hw_, 1) : 0; }
    /// The Flux2 VAE's stage sizes (mf_vae.cpp's f2_walk).
    struct F2Sizes {
        int64_t most = 0, pix = 0, mid = 0, rows = 0, c = 0;
    };
    uint64_t weight_bytes() const { return weight_bytes_; }

    /// The CoD decoder's output [h w][384] and the 21 blocks' output [h w][384] of the last decode (device FP32), for
    /// the probe.
    const float* last_cond() const { return cond_; }
    const float* last_state() const { return s_state_; }

private:
    struct Conv {
        int64_t out = 0, in = 0, k = 1;
        const uint16_t* w = nullptr;   // [out][k][k][in] FP16
        const float* b = nullptr;      // null: no bias
    };
    struct Res {
        const float *n1w = nullptr, *n1b = nullptr, *n2w = nullptr, *n2b = nullptr;
        Conv c1, c2;
    };
    struct Attn {
        const float *nw = nullptr, *nb = nullptr;
        Conv qkv, proj;   // qkv: q, k and v's 1 x 1 convolutions as one of 3 C outputs
    };
    struct DiCo {
        Conv c1, c4, c3, c5;
        const float *dw = nullptr, *db = nullptr;   // the depthwise 3 x 3 convolution [C][9], its bias
        const float *caw = nullptr, *cab = nullptr; // the channel attention's 1 x 1 convolution [C][C] (FP32)
        const float* mod = nullptr;                  // [6][C]: shift, scale, gate (attention), shift, scale, gate (MLP)
    };
    /// A DiCo block's buffers: t0, t1 [HW][F], a16 [HW][F] FP16, vec [2 C].
    struct DiCoBufs {
        float *t0 = nullptr, *t1 = nullptr, *vec = nullptr;
        uint16_t* a16 = nullptr;
    };
    Conv conv_w(class SafeTensors& st, const std::string& name, bool bias = true, int64_t col0 = 0,
                int64_t cols = -1, const std::vector<int64_t>* rows = nullptr);
    const float* vec(class SafeTensors& st, const std::string& name);
    const float* upload(const std::vector<float>& v);
    Attn attn_w(class SafeTensors& st, const std::string& norm, const std::string (&qkv)[3], const std::string& proj,
                int64_t C);
    // out = the convolution of in16 plus its bias, or out += that with `add` (a residual)
    void conv(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, float* out, bool add = false);
    /// A 3 x 3 convolution of stride 2 (the input padded by one on the bottom and the right): out [H / 2][W / 2].
    void conv_s2(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, float* out);
    void gn_silu16(const float* x, const float* w, const float* b, int64_t HW, int64_t C, bool silu, uint16_t* y16);
    void resblock(const Res& r, float*& x, int64_t H, int64_t W);
    void attention(const Attn& a, float* x, int64_t H, int64_t W);
    void dico(const DiCo& d, float* s, int64_t H, int64_t W, const DiCoBufs& b);
    DiCo dico_w(class SafeTensors& st, const std::string& p, const std::vector<float>& mod);
    void grow(int64_t h, int64_t w);
    // the Flux2 VAE's decoder
    struct F2Res {
        const float *n1w = nullptr, *n1b = nullptr, *n2w = nullptr, *n2b = nullptr;
        Conv c1, c2, nin;   // nin: the 1 x 1 shortcut where the channels change (out 0: none)
    };
    struct F2Level {
        std::vector<F2Res> blocks;
        Conv up;            // the convolution after the 2x upsampling (out 0: the last level, none)
        Conv shuffle;       // SeedVR2's: the 1 x 1 convolution to 4 c whose 2 x 2 spread is the upsampling (out 0:
                            // nearest)
    };
    // the Flux2 VAE's encoder: a level's blocks, then its downsampling (out 0: the last level, none)
    struct F2Down {
        std::vector<F2Res> blocks;
        Conv down;
    };
    void open_flux2(class SafeTensors& st);
    void open_flux2_encoder(class SafeTensors& st);
    void f2_others(const float* x, float*& a, float*& b) const;
    void f2_resblock(const F2Res& r, float*& x, int64_t H, int64_t W);
    void f2_attention(const Attn& a, float* x, int64_t H, int64_t W);
    F2Sizes f2_enc_walk(int64_t h, int64_t w) const;
    void f2_encode(const float* rgb, int64_t h, int64_t w, float* z);
    void f2_encode_tiled(const float* rgb, int64_t h, int64_t w, float* z, int64_t th, int64_t tw);
    void f2_front(const float* z, int64_t h, int64_t w, int k, float*& x, int64_t& H, int64_t& W, int64_t& c);
    void f2_levels(int k0, int k1, float*& x, int64_t& H, int64_t& W, int64_t& c);
    void f2_tail(const float* x, int64_t H, int64_t W, int64_t c, float* rgb);
    void f2_decode_raw(const float* z, int64_t h, int64_t w, float* rgb);
    void f2_decode_tiled(const float* z, int64_t h, int64_t w, float* rgb, int k, int64_t th, int64_t tw);
    F2Sizes f2_walk(int64_t h, int64_t w, int k0, int k1, bool front, bool tail) const;
    uint64_t f2_bytes(const F2Sizes& z) const;
    uint64_t f2_tiled_bytes(int64_t h, int64_t w, int k, int64_t th, int64_t tw) const;
    struct F2Tiling {
        int k = -1;
        int64_t th = 0, tw = 0;
    };
    F2Tiling f2_tiling(int64_t h, int64_t w, uint64_t limit) const;
    uint64_t f2_least_bytes(int64_t h, int64_t w) const;
    uint64_t f2_scratch_bytes(int64_t h, int64_t w) const;
    void f2_grow(const F2Sizes& z);

    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    uint64_t weight_bytes_ = 0;
    bool dry_ = false;
    int64_t C_ = 384, F_ = 1536;
    // the CoD decoder
    Conv cin_, cout_;
    Res res_[3];
    Attn attn_[2];
    const float *gnw_ = nullptr, *gnb_ = nullptr;
    // the denoiser
    Conv s_in_;        // s_embedder.proj2's condition columns (x is zero: proj1's part vanishes)
    std::vector<DiCo> blocks_;
    Conv yx_, ce_;     // y_embedder_x [8192][384], dec_net.cond_embed [8192][384]
    const float* px_ = nullptr;   // the per-pixel MLP's weights (FP32, mf_vae.cpp's PixW layout)
    const float* dct_ = nullptr;  // [256][64] the NerfEmbedder's DCT features of a patch's pixels
    // scratch, grown on demand
    int64_t cap_hw_ = 0, cap_band_ = 0;
    float *x_ = nullptr, *t0_ = nullptr, *t1_ = nullptr, *t2_ = nullptr, *cond_ = nullptr, *s_state_ = nullptr;
    float *gn_part_ = nullptr, *vec_ = nullptr, *yxb_ = nullptr, *ceb_ = nullptr, *scores_ = nullptr;
    uint16_t *a16_ = nullptr, *col16_ = nullptr, *w16_ = nullptr;
    int64_t col_cap_ = 0;
    // the encoder
    std::string path_;
    uint64_t enc_bytes_ = 0;
    Conv e_patch_, e_down_, e_fuse_, e_out_;   // e_patch_: [768][3 * 16 * 16] in (channel, row, column) order
    std::vector<DiCo> enc_head_, enc_blocks_;
    const float* e_norm_ = nullptr;           // norm_out as a modulation: shift (its bias), scale (its weight - 1)
    // the Flux2 VAE (f2_): its batch norm's mean and std (sqrt(var + 1e-4)) by the token's 128 values, the 1 x 1
    // convolution after it, conv_in, the middle (two ResNet blocks around the attention), the levels from the
    // smallest, and norm_out / conv_out in gnw_, gnb_ and cout_
    bool f2_ = false, tiled_ = false;
    int64_t f2_lc_ = 32;       // the latent's channels: a token holds 2 x 2 positions of them (SeedVR2's VAE: 16)
    const float *f2_mean_ = nullptr, *f2_std_ = nullptr;
    Conv f2_pq_;
    F2Res f2_mid_[2];
    Attn f2_attn_;
    std::vector<F2Level> f2_levels_;
    // its encoder: conv_in (its input widened to 8 channels), the levels from the largest, the middle, norm_out,
    // conv_out and quant_conv (out 0: none)
    Conv f2_ein_, f2_eout_, f2_qc_;
    std::vector<F2Down> f2_down_;
    F2Res f2_emid_[2];
    Attn f2_eattn_;
    const float *f2_enw_ = nullptr, *f2_enb_ = nullptr;
    F2Sizes f2_cap_;           // what the scratch holds (rows: the attention's query band)
    int64_t f2_ac_ = 0;        // the widest middle attention's channels (the decoder's, or the encoder's once loaded)
    int64_t f2_band_ = 0;      // the convolutions' unfolded band (values), by the decoder's widths
    uint64_t f2_held_ = 0;
    float *f2_x_ = nullptr, *f2_y_ = nullptr, *f2_t_ = nullptr, *f2_q_ = nullptr;
    uint16_t *f2_qk16_ = nullptr, *f2_p16_ = nullptr;
};

}  // namespace strata::image
