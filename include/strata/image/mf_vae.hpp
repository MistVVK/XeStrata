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
// tokens, 64 pixels, on each side shared), blended linearly across the overlap; the
// group norms of the tiled levels take each tile's statistics.
//
// SeedVR2's VAE (ByteDance-Seed/SeedVR's video_vae_v3, as Comfy-Org's ema_vae_fp16 names it, diffusers' names) is the
// same LDM VAE inflated to video; an image is a clip of one frame, where its causal convolutions (the first frame
// repeated before it) are 2D ones whose kernels are their time taps summed, as they are read.  Its latent is 16
// channels with no batch norm and no post_quant_conv (a token: 2 x 2 positions of them, 64 values, as the Flux2 VAE's
// 128), and an upsampling is a 1 x 1 convolution to 4 c whose output spread over 2 x 2 positions is the larger image
// (the frames it would add in time left out), then the convolution.
//
// A video (set_video before open: SeedVR2's VAE) keeps its convolutions' time taps as well, and runs slices of frames
// as the reference's causal slicing does (video_vae_v3's slicing_encode / slicing_decode): a causal convolution of kt
// taps sees the slice's frames after the frames its last slice left (kt - stride of them), the first slice's first
// frame repeated kt - 1 times before it; the encoder halves the frames in time twice (levels 1 and 2, a stride of 2)
// and the decoder doubles them twice (its first two levels' 1 x 1 convolution to 2 frames, the first slice's
// duplicated first frame dropped).  The group norms and the middle's attention are a frame's own.  A clip of 4 n + 1
// frames is n + 1 latents; its first slice 1 + 4 k frames, each next one 4 k.
#pragma once

#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
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

    /// The video form (SeedVR2's VAE): set before open() and open_encoder(), which then keep each convolution's time
    /// taps as the file has them besides their sum (the image's form).
    void set_video(bool on) { video_ = on; }
    bool video() const { return video_; }
    /// Forgets what the slices so far left (a new clip), and lets their frames' memory go.
    void video_reset();
    /// The tiles' carried frames in host memory (pinned), each convolution's copied to the GPU when its tile's turn
    /// comes and back when another tile's does, so that the GPU holds one tile's: for a frame whose tiles' carried
    /// frames do not fit the GPU together.  Set between clips (it forgets what the slices left).
    void set_video_host_cache(bool on);
    bool video_host_cache() const { return vhost_; }
    /// With the host cache: the GPU's copies of the last tiles' carried frames (both sides') back to their host copies
    /// and their memory let go, for another part's turn (the copy back the next call would make anyway: nothing more
    /// is copied while there are several tiles).
    void video_park();
    /// Whether the slices after this call read what it leaves (true until the clip's last call): false spares the
    /// host cache's copies back of the tiles' carried frames, which no slice reads again.
    void set_video_keep(bool keep) { vkeep_ = keep; }
    /// The latents a tile's slice takes in encode_video_tiled and decode_video_tiled (the encoder's 1 + 4 (n - 1)
    /// frames first, then 4 n): their work video_scratch_bytes(.., 4 n).  1 by default.
    void set_video_slice(int64_t latents) { vslice_ = std::max<int64_t>(latents, 1); }
    /// A slice of frames rgb [T][3][16 h][16 w] (device FP32, -1 .. 1) through the encoder: the latents' tokens it
    /// gives, z [n][h w][64] (device FP32, the posterior's means), their number n the return.  `first`: the clip's
    /// first slice (T = 1 + 4 k, k + 1 latents), else T = 4 k (k latents) after the slice before.
    int64_t encode_video(const float* rgb, int64_t T, int64_t h, int64_t w, bool first, float* z);
    /// A slice of latents z [n][h w][64] (device FP32) through the decoder: rgb [T][3][16 h][16 w] (device FP32, the
    /// output before the clamp, about -1 .. 1), T the return.  `first`: the clip's first slice (1 + 4 (n - 1) frames),
    /// else 4 n frames after the slice before.
    int64_t decode_video(const float* z, int64_t n, int64_t h, int64_t w, bool first, float* rgb);
    /// The same in tiles of tt x tt tokens (the last ones flush with the edges, kF2Overlap tokens shared and blended
    /// linearly, the latents' on the encoder's side, the pixels' on the decoder's), each tile its own carried frames:
    /// the layout must stay the same through a clip.  tt 0, or one the latent fits in: the whole.  Each tile goes
    /// through all the call's frames before the next, a latent's at a time (the work of one latent's, whatever the
    /// call's length), so that a tile's carried frames move to and from host memory once a call.
    int64_t encode_video_tiled(const float* rgb, int64_t T, int64_t h, int64_t w, bool first, float* z, int64_t tt);
    int64_t decode_video_tiled(const float* z, int64_t n, int64_t h, int64_t w, bool first, float* rgb, int64_t tt);
    /// The GPU memory a slice of T frames (encode) or n latents (decode) of an h x w latent works in, besides the
    /// carried frames.
    uint64_t video_scratch_bytes(int64_t h, int64_t w, int64_t frames) const;
    /// A video's tiles along a side of n tokens for tiles of at most tt (encode_video_tiled's and
    /// decode_video_tiled's): their side, as small as the count that side takes covers n with, and that count.
    static int64_t video_tile_side(int64_t n, int64_t tt);
    static int64_t video_tiles(int64_t n, int64_t tt);
    /// The frames a tile of h x w tokens carries through a clip (its causal convolutions'): the encoder's (side 0), the
    /// decoder's (1) or both (-1).  The host cache holds one side's on the GPU at a time.
    uint64_t video_carried_bytes(int64_t h, int64_t w, int side = -1) const;

    /// The CoD decoder's output [h w][384] and the 21 blocks' output [h w][384] of the last decode (device FP32), for
    /// the probe.
    const float* last_cond() const { return cond_; }
    const float* last_state() const { return s_state_; }

private:
    struct Conv {
        int64_t out = 0, in = 0, k = 1;
        const uint16_t* w = nullptr;   // [out][k][k][in] FP16 (the image form's: the time taps summed)
        const float* b = nullptr;      // null: no bias
        int64_t kt = 1;                // a video convolution's time taps
        const uint16_t* taps = nullptr;   // the video form's: [out][kt][k][k][in] FP16 (null: kt 1, or not video)
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
    /// A 3 x 3 convolution of stride 2 (the input padded by one on the bottom and the right): out [H / 2][W / 2]
    /// (+= with `add`).
    void conv_s2(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, float* out, bool add = false);
    /// conv's into FP16 out16 (+= with `add`) where a library writes FP16 (oneDNN, cuDNN): false where none takes it.
    /// side: room for the convolution's own output where the sum is made in a pass of the engine's (cuDNN's).
    bool conv16(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, uint16_t* out16, bool add,
                uint16_t* side);
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
        Conv shuffle_t;     // the video form's: all its 4 z c outputs, z frames in time (z the frames it makes)
        int64_t z = 1;
    };
    // the Flux2 VAE's encoder: a level's blocks, then its downsampling (out 0: the last level, none)
    struct F2Down {
        std::vector<F2Res> blocks;
        Conv down;
    };
    void open_flux2(class SafeTensors& st);
    void open_flux2_encoder(class SafeTensors& st);
    void f2_others(const uint16_t* x, uint16_t*& a, uint16_t*& b) const;
    void f2_resblock(const F2Res& r, uint16_t*& x, int64_t H, int64_t W);
    void f2_attention(const Attn& a, uint16_t* x, int64_t H, int64_t W);
    const uint16_t* f2_in16(const uint16_t* x, int64_t n, uint16_t* room);
    F2Sizes f2_enc_walk(int64_t h, int64_t w) const;
    void f2_encode(const float* rgb, int64_t h, int64_t w, float* z);
    void f2_encode_tiled(const float* rgb, int64_t h, int64_t w, float* z, int64_t th, int64_t tw);
    void f2_front(const float* z, int64_t h, int64_t w, int k, uint16_t*& x, int64_t& H, int64_t& W, int64_t& c);
    void f2_levels(int k0, int k1, uint16_t*& x, int64_t& H, int64_t& W, int64_t& c);
    void f2_tail(const uint16_t* x, int64_t H, int64_t W, int64_t c, float* rgb);
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
    // the video form
    struct VCache {
        uint16_t* buf = nullptr;   // the frames a causal convolution carries, [m][frame] FP16 (host memory: vhost_)
        int64_t frame = 0, m = 0, n = 0;   // values a frame, the frames it holds, the frames held now
        int owner = -1;                    // vdev_'s: the tile whose frames it holds (newer than its host copy)
    };
    // out: the activations' 16 bits (o16) or FP32; side: T output frames' room the residual's sum may use
    void vconv(const Conv& c, const uint16_t* in16, int64_t T, int64_t H, int64_t W, int64_t ts, bool s2, void* out,
               bool o16, bool add, int64_t& To, uint16_t* side = nullptr);
    void v_frame(const Conv& k, const uint16_t* in16, int64_t H, int64_t W, bool s2, void* out, bool o16, bool add,
                 uint16_t* side = nullptr);
    const uint16_t* v_in16(const uint16_t* x, int64_t n);
    void v_resblock(const F2Res& r, uint16_t*& x, int64_t T, int64_t H, int64_t W);
    void v_attention(const Attn& a, uint16_t* x, int64_t T, int64_t H, int64_t W);
    void v_grow(int64_t values, int64_t head, int64_t frame);
    void v_others(const uint16_t* x, uint16_t*& a, uint16_t*& b) const;
    void v_leave(int side);
    void v_enter(int side);
    bool finite(const float* x, int64_t n);
    int64_t v_encode_tiled(const float* rgb, int64_t T, int64_t h, int64_t w, bool first, float* z, int64_t tt);
    int64_t v_decode_tiled(const float* z, int64_t n, int64_t h, int64_t w, bool first, float* rgb, int64_t tt);
    bool overflowed(const float* x, int64_t n);
    bool video_ = false;
    struct VKey {
        const Conv* c;
        int tile;
        bool operator==(const VKey& o) const { return c == o.c && tile == o.tile; }
    };
    struct VKeyHash {
        size_t operator()(const VKey& k) const { return std::hash<const void*>()(k.c) ^ ((size_t) k.tile << 1); }
    };
    std::unordered_map<VKey, VCache, VKeyHash> vcache_[2];   // the encoder's, the decoder's, by tile
    std::unordered_map<const Conv*, VCache> vdev_[2];        // vhost_: a convolution's on the GPU, the tiles' in turn
    int vside_ = 0, vtile_ = 0;
    bool vhost_ = false, vkeep_ = true;
    bool vkept_[2] = {true, true};   // vkeep_ at each side's last call
    // a slice's activations, the largest stage's frames: 16 bits, FP16 until an output has a value past FP16's range
    // (then BF16 from that call on, the call again); the normalizations' statistics and the taps' sums in FP32
    // (vtmp_, a frame's)
    uint16_t *vx_ = nullptr, *vy_ = nullptr, *vt_ = nullptr;
    bool vbf_ = false;
    int* vbad_ = nullptr;
    int64_t vslice_ = 1;
    float* vtmp_ = nullptr;
    int64_t vtmp_cap_ = 0;
    uint16_t* vtapw_ = nullptr;   // a convolution's taps one after another, [kt][out][k][k][in], for the tap-wise path
    int64_t vtapw_cap_ = 0;
    uint16_t* va16_ = nullptr;   // after `vhead_` values of room for a convolution's carried frames before its input
    uint16_t* va16_base_ = nullptr;
    int64_t vhead_ = 0;
    int64_t vcap_ = 0;

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
    // the Flux2 VAE's activations, a frame's: 16 bits as the video form's (vbf_), the taps' sums in vtmp_
    uint16_t *f2_x_ = nullptr, *f2_y_ = nullptr, *f2_t_ = nullptr;
    float* f2_q_ = nullptr;
    uint16_t *f2_qk16_ = nullptr, *f2_p16_ = nullptr;
};

}  // namespace strata::image
