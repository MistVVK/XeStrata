// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/guidance.hpp - the guidance ComfyUI's nodes add to a flow model's sampling, worked as ComfyUI
// works them: on the denoised predictions (x0 = x - sigma v, ComfyUI's CONST calculate_denoised), not on the velocity.
//
//   APG (comfy_extras/nodes_apg.py, a pre-CFG function): the guidance cond - uncond of the denoised predictions, its
//     running sum (momentum; started again when the sigma rises), its norm over the whole latent cut to
//     norm_threshold, its part along cond scaled by eta; the conditional prediction becomes
//     uncond + guidance' + (cond - uncond) / cfg, which CFG then combines.
//   CFG: uncond + cfg (cond - uncond); without a negative pass, cond.
//   Then the post-CFG functions in this order (ComfyUI runs them in the order the nodes patch the model; this order is
//   XeStrata's, fixed):
//   CFGZero* (nodes_cfg.py CFGZeroStar): alpha = <x - cond, x - uncond> / (|x - uncond|^2 + 1e-8) over the whole
//     latent, den + uncond (alpha - 1) + cfg uncond (1 - alpha).
//   Skip Layer Guidance (nodes_slg.py SkipLayerGuidanceDiT): with the model's sigma within [sigma_end, sigma_start],
//     den + scale (cond - skipped), skipped the conditional prediction of a pass with `layers` skipped; rescaling_scale
//     moves den's spread towards cond's (torch.std, unbiased, over the whole latent).
//   CFGNorm (nodes_cfg.py CFGNorm, its default post-CFG form): den scaled at each latent position by
//     clamp(|cond| / (|den| + 1e-8), 0, 1) (norms over the channels) and by strength.
//   Detail Daemon (Jonseed's ComfyUI-Detail-Daemon, DetailDaemonSamplerNode, after muerrilla's sd-webui-detail-daemon):
//     the sampler's own sigmas and steps stay; the model is asked at sigma (1 - 0.1 d cfg), d the schedule's value
//     (make_detail_daemon_schedule over the steps, looked up and interpolated at the sigma as get_dd_schedule does),
//     cfg its override or the request's scale (1 without guidance).  Every prediction above (and the window of skip
//     layer guidance) then goes by that sigma, as ComfyUI's wrapped model does.
// CFGZero* and APG need the negative pass (they do nothing without one here; the server refuses them); CFGNorm, skip
// layer guidance and Detail Daemon work at CFG 1 as well.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::image {

struct GuidanceSpec {
    float cfgnorm = -1.0f;   // CFGNorm's strength; below 0 off
    bool zero_star = false;  // CFGZero*
    bool apg = false;
    float apg_eta = 1.0f, apg_norm_threshold = 5.0f, apg_momentum = 0.0f;
    std::vector<int> slg_layers;   // skip layer guidance's blocks; empty off
    float slg_scale = 3.0f, slg_sigma_start = 1.0f, slg_sigma_end = 0.0f, slg_rescale = 0.0f;
    bool dd = false;   // Detail Daemon
    float dd_amount = 0.1f, dd_start = 0.2f, dd_end = 0.8f, dd_bias = 0.5f, dd_exponent = 1.0f, dd_start_offset = 0.0f,
          dd_end_offset = 0.0f, dd_fade = 0.0f, dd_cfg = 0.0f;   // dd_cfg: the CFG override, 0 the request's
    bool dd_smooth = true;

    bool any() const { return cfgnorm >= 0.0f || zero_star || apg || (!slg_layers.empty() && slg_scale > 0.0f) || dd; }
};

/// Detail Daemon's schedule (make_detail_daemon_schedule): `steps` multipliers.
std::vector<float> detail_daemon_schedule(const GuidanceSpec& g, int steps);

/// One sampling's guidance: its history (APG's running sum) lives as long as it does, one per image, region or tile.
/// The latent is n values, `channels` a position: token-major ([position][channel], Mage-Flow's) or channel-major
/// ([channel][position], Qwen-Image's).  `sigmas`: the sampler's own (the steps and the terminal 0, before any change
/// of its own, as er_sde's first), which Detail Daemon's schedule is looked up by.
class Guidance {
public:
    Guidance(const GuidanceSpec& g, void* stream, int64_t n, int64_t channels, bool token_major, float cfg,
             const std::vector<float>& sigmas);
    ~Guidance();
    Guidance(const Guidance&) = delete;
    Guidance& operator=(const Guidance&) = delete;

    /// Any method on: the sampler steps by the denoised values `denoised` makes.
    bool on() const { return on_; }
    /// The sigma the model is asked at, for the sampler's sigma (Detail Daemon's; else the same).
    float model_sigma(float sigma) const;
    /// Skip layer guidance's pass at the model's sigma, and its blocks.
    bool skip_pass(float sigma) const;
    const std::vector<int>& skip_layers() const { return g_.slg_layers; }

    /// den = the guided denoised prediction at the model's sigma from the velocities c (the prompt's), u (the
    /// negative prompt's; null without guidance) and k (the skipping pass's; null outside its window), device FP32
    /// [n] each, on the stream.
    void denoised(const float* x, const float* c, const float* u, const float* k, float sigma, float* den);

private:
    void release();
    void dump(const float* x, const float* c, const float* u, const float* k, float sigma, const float* den);

    GuidanceSpec g_;
    void* s_;
    int64_t n_, ch_;
    bool tm_, on_;
    float cfg_;
    std::vector<float> sig_, dd_;
    float* cd_ = nullptr;    // the conditional prediction (APG's after it)
    float* ud_ = nullptr;    // the unconditional one
    float* run_ = nullptr;   // APG's running guidance
    float* part_ = nullptr;  // the reductions' partial sums
    float* red_ = nullptr;   // their sums
    size_t wg_ = 0, groups_ = 0;
    bool have_run_ = false;
    float prev_sigma_ = -1.0f;
    int calls_ = 0, seq_ = 0;
};

/// Euler's step from the denoised values: x += (s1 - s0) (x - den) / s0, x = den at s1 = 0.
void guidance_euler(float* x, const float* den, float s0, float s1, int64_t n, void* stream);

}  // namespace strata::image
