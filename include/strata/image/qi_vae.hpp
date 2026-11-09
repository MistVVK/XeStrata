// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_vae.hpp - Qwen-Image-2.1's VAE decoder on the GPU: a latent [64][h][w] to an RGBA image
// [4][16 h][16 w] in 0 .. 1.  The same part reads Qwen-Image's first VAE, which is Wan 2.1's (Anima's: its 16 channels
// at an eighth of the image's sides, the original checkpoint's names), whose causal 3-D convolutions take one frame:
// the zero frames before it leave each kernel's last time slice, a 2-D convolution, and the upsamplers' time
// convolutions are not run on a first frame.
//
// The activations are channels-last ([pixel][channel], FP32); a convolution is a product on the matrix path
// (prefill::Gemm, FP16 operands) of the input's 3 x 3 neighbourhoods (unfolded a band of rows at a time, so the scratch
// stays small at full size) by the weights (FP16, unfolded the same way when loaded).
#pragma once

#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

class QiVae {
public:
    QiVae();
    ~QiVae();
    QiVae(const QiVae&) = delete;
    QiVae& operator=(const QiVae&) = delete;

    /// Reads the VAE's safetensors (Qwen-Image-2.1's diffusers names, or Wan 2.1's original names) and puts the
    /// decoder's weights on the current GPU.
    /// `dry`: read for its sizes only (weight_bytes, scratch_bytes), nothing put on the GPU; such a part cannot run.
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false);

    /// rgba [4][16 h][16 w] (device FP32, 0 .. 1) from the DiT's latent z [64][h][w] (device FP32).  When the whole
    /// image's working memory (scratch_bytes) passes `max_scratch`, the latent is decoded in overlapping tiles (the
    /// largest that fit, a quarter of a tile apart) whose images are blended where they meet.
    void decode(const float* z, int64_t h, int64_t w, float* rgba, uint64_t max_scratch = UINT64_MAX);

    /// The encoder (a reference image for an edit): z [64][H / 16][W / 16] (device FP32, the DiT's latent, normalized
    /// as the decoder's input is) from rgba [4][H][W] (device FP32, 0 .. 1), H and W multiples of 16.  False when the
    /// VAE file has no encoder.  zstd (null: not taken) gets the posterior's standard deviation in the same normalized
    /// units, exp(logvar / 2) / the latent's sd, logvar clamped to [-30, 20] as diffusers' DiagonalGaussianDistribution
    /// does: a training step draws z + zstd eps from them, as ai-toolkit's latent_dist.sample() does.
    bool encode(const float* rgba, int64_t H, int64_t W, float* z, float* zstd = nullptr);
    bool has_encoder() const { return has_encoder_; }
    /// The GPU memory an encode of an H x W image works in.
    uint64_t encode_scratch_bytes(int64_t H, int64_t W) const;

    /// The GPU memory a decode of one h x w piece works in (its buffers, besides the weights).
    uint64_t scratch_bytes(int64_t h, int64_t w) const;
    /// The least a decode takes: an 8 x 8 tile's.
    uint64_t scratch_min() const { return scratch_bytes(8, 8); }
    /// Frees the working buffers (grown again on use).
    void drop_scratch();
    /// What the working buffers hold now.
    uint64_t scratch_held() const { return (uint64_t) cap16_ * 2 + (uint64_t) cap32_ * 4 * 3 + (uint64_t) col_cap_ * 2; }
    uint64_t weight_bytes() const { return weight_bytes_; }
    /// The image's side over the latent's (16; Wan 2.1's 8) and the latent's channels (64; Wan 2.1's 16).
    int scale() const { return scale_; }
    int64_t latent_channels() const { return pq_.in; }

private:
    struct Need {
        int64_t big16 = 0, big32 = 0, keep = 0, col = 0;
    };
    Need need(int64_t h, int64_t w) const;
    void decode_piece(const float* z, int64_t h, int64_t w, float* rgba);
    struct Conv {
        int64_t out = 0, in = 0, k = 1;
        const uint16_t* w = nullptr;   // [out][k][k][in] FP16
        const float* b = nullptr;
    };
    struct Res {
        const float *g1 = nullptr, *g2 = nullptr;
        Conv c1, c2, sc;   // sc.w null: no shortcut convolution
    };
    Conv conv_w(class SafeTensors& st, const std::string& name);
    const float* vec(class SafeTensors& st, const std::string& name);
    Res res_w(class SafeTensors& st, const std::string& p);
    // out [HW][c.out] = c(in16 [H][W][c.in]) + bias
    // out = the convolution of in16 plus its bias, or out += that with `add` (a residual)
    void conv(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, float* out, bool add = false);
    // x [HW][C] -> x + the residual block (x replaced; scratch buffers)
    void resblock(const Res& r, float*& x, int64_t H, int64_t W, int64_t cin);
    void attention(float* x, int64_t H, int64_t W, int64_t C, const float* g, const Conv& qkv, const Conv& proj);
    Need encode_need(int64_t H, int64_t W) const;
    void grow(int64_t elems16, int64_t elems32, int64_t col16);

    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    uint64_t weight_bytes_ = 0;
    bool dry_ = false;
    const float *mean_ = nullptr, *std_ = nullptr;
    Conv pq_, cin_, cout_;
    Res mid0_, mid1_;
    const float *ag_ = nullptr;
    Conv aqkv_, aproj_;
    struct Up {
        std::vector<Res> r;
        Conv up;   // up.out 0: no upsampling
        int ft = 1;   // the shortcut's frames; 0: no shortcut over the block (Wan 2.1's)
    };
    std::vector<Up> up_;
    const float* gout_ = nullptr;
    // the encoder
    bool has_encoder_ = false;
    Conv ein_, eout_, quant_;
    Res emid0_, emid1_;
    const float* eag_ = nullptr;
    Conv eaqkv_, eaproj_;
    struct Down {
        std::vector<Res> r;
        Conv down;   // down.out 0: no downsampling
        int ft = 1;  // the shortcut's frames (2 where the block also halves time); 0: none (Wan 2.1's)
    };
    std::vector<Down> down_;
    void open_wan21(class SafeTensors& st);
    bool wan21_ = false;
    int scale_ = 16;
    int64_t rgb_in_ = 4;   // the encoder's input channels taken from the image (Wan 2.1's 3: no alpha)
    const float* egout_ = nullptr;
    // scratch, grown on demand
    int64_t cap16_ = 0, cap32_ = 0;
    uint16_t *a16_ = nullptr, *col16_ = nullptr;
    float *t0_ = nullptr, *t1_ = nullptr, *t2_ = nullptr;
    int64_t col_cap_ = 0;
};

}  // namespace strata::image
