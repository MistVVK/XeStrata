// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/upscale/seedvr2.hpp - SeedVR2 (ByteDance Seed's one-step diffusion restoration, Apache-2.0), an
// upscaler of the common frame (upscale.hpp): it takes the frames at the size asked for (the driver resizes them by
// Lanczos and pads them to 16s) and restores them in one step.
//
// A frame (an image is a clip of one): its RGB in -1 .. 1 through SeedVR2's VAE encoder (MfVae's SeedVR2 form, the
// posterior's mean, as ComfyUI's encode takes it), times 0.9152, the condition; a noise of the latent's shape
// (torch.randn of a CPU generator seeded by the request's seed, [16][h][w], as ComfyUI's sampler makes it); the DiT's
// velocity of them (seedvr2_dit.hpp); the noise less the velocity, divided by 0.9152, through the decoder.  The alpha
// comes from the input as it is.
//
// The files: the DiT's (Comfy-Org's seedvr2_7b_fp16 or 7b_sharp) and the VAE's, ema_vae_fp16.safetensors, beside it.
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
    void run_clip(const uint8_t* in, int64_t w, int64_t h, int64_t frames, uint8_t* out, uint64_t seed) override;
    void release(bool all) override;
    uint64_t held() const override;
    std::string parts() const override;

    /// The VAE's file for a DiT's: ema_vae_fp16.safetensors in its folder.
    static std::string vae_path(const std::string& dit);

    static constexpr float kScaling = 0.9152f;   // the latent's scale (the reference's configs' scaling_factor)

private:
    struct Impl;
    ModelInfo info_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::upscale
