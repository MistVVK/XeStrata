// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/upscale/seedvr2.hpp - SeedVR2 (ByteDance Seed's one-step diffusion restoration, Apache-2.0), an
// upscaler of the common frame (upscale.hpp): it takes the frames at the size asked for (the driver resizes them by
// Lanczos and pads them to 16s) and restores them in one step.
//
// A frame (an image is a clip of one): its RGB in -1 .. 1 through SeedVR2's VAE encoder (MfVae's SeedVR2 form, the
// posterior's mean), times 0.9152, the condition; a noise of the latent's shape
// (torch.randn of a CPU generator seeded by the request's seed, [16][h][w]); the DiT's
// velocity of them (seedvr2_dit.hpp); the noise less the velocity, divided by 0.9152, through the decoder.  The alpha
// comes from the input as it is.
//
// The files: the DiT's (Comfy-Org's seedvr2_7b_fp16 or 7b_sharp) and the VAE's, ema_vae_fp16.safetensors, beside it.
//
// A video goes through in time (the VAE's encode of the whole clip, the latent split into overlapping chunks, a step a
// chunk, the chunks merged over their overlaps, the decode of the whole), streamed: the
// frames pushed are encoded in the VAE's slices (its causal convolutions carry what the slice before left: the same
// latents as the whole clip at once); each chunk of latents goes through the DiT once its latents are there (the noise
// the same seed's for every chunk, of the chunk's shape [16][t][h][w], as the sampler makes it for each); the chunks
// overlap by `overlap` latents, crossfaded (seedvr2.cpp, kept_share); the latents no later chunk touches are decoded, in the VAE's slices too, and their
// frames are ready.  What it holds does not grow with the clip: a chunk's latents and the frames of a slice.  The
// clip's frames are padded to 4 n + 1 (the last one repeated), the padding's output dropped.
#pragma once

#include "strata/upscale/upscale.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace strata::upscale {

class Seedvr2 final : public UpscaleModel {
public:
    /// Reads the DiT at `path` and the VAE beside it; std::runtime_error for files that are not SeedVR2's.
    explicit Seedvr2(const std::string& path);
    ~Seedvr2() override;
    Seedvr2(const Seedvr2&) = delete;
    Seedvr2& operator=(const Seedvr2&) = delete;

    const ModelInfo& info() const override { return info_; }
    uint64_t weight_bytes() const override;
    uint64_t weight_least() const override;
    void set_room(uint64_t bytes) override;
    uint64_t clip_bytes(int64_t w, int64_t h, int64_t frames) const override;
    uint64_t clip_largest(int64_t w, int64_t h, int64_t frames) const override;
    /// run_clip's least: the VAE's passes in their smallest tiles (as run_clip takes them where the room is short).
    uint64_t clip_least(int64_t w, int64_t h, int64_t frames) const override;
    void run_clip(const uint8_t* in, int64_t w, int64_t h, int64_t frames, uint8_t* out, uint64_t seed) override;
    void release(bool all) override;
    uint64_t held() const override;
    std::string parts() const override;

    /// A video of about `frames` frames of w x h (multiples of 16) from now on (the count a hint: the chunk is no
    /// larger than the clip; the clip is what is pushed before video_end): chunks of `chunk` frames (4 n + 1; 0: the
    /// largest whose activations fit the room set_room gave) through the DiT, overlapping by `overlap` latents.  Its
    /// VAE is opened in its video form (its time taps), in tiles where a slice of the frame whole does not fit.
    bool temporal() const override { return true; }
    void video_begin(int64_t w, int64_t h, int64_t frames, uint64_t seed, int64_t chunk, int64_t overlap) override;
    /// `n` frames, RGBA8 [n][h][w][4] on the GPU (the alpha not read), next in the clip.
    void video_push(const uint8_t* in, int64_t n) override;
    /// The frames restored so far and not taken, RGBA8 (the alpha 255).
    int64_t video_ready() const override;
    /// The next `n` of them into out [n][h][w][4] on the GPU.
    void video_take(uint8_t* out, int64_t n) override;
    /// After the last frame: the rest made ready (the clip's padding not among them).
    void video_end() override;
    /// The VRAM a video's chunk of `chunk` frames of w x h works in besides the weights: the DiT's activations, the
    /// VAE's slice and the frames it carries (in its smallest tiles where `least`, else the frame whole), the latents
    /// held.
    uint64_t video_bytes(int64_t w, int64_t h, int64_t chunk, bool least) const override;

    /// The VAE's file for a DiT's: ema_vae_fp16.safetensors in its folder.
    static std::string vae_path(const std::string& dit);

    static constexpr float kScaling = 0.9152f;   // the latent's scale (the reference's configs' scaling_factor)

private:
    struct Impl;
    ModelInfo info_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::upscale
