// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2024 Sahand Ahmadian
// SPDX-FileCopyrightText: 2024 Jonseed
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/image/guidance.hpp - the guidance added to a flow model's sampling, worked on the denoised
// predictions (x0 = x - sigma v), not on the velocity.  c and u below are the prompt's and the negative prompt's
// predictions, w the guidance scale; the result is den.
//
//   Plain CFG: den = u + w (c - u); without a negative pass, den = c.
//   APG (Sadat et al., adaptive projected guidance), in place of plain CFG: d = c - u, summed over the steps with the
//     momentum's weight on the earlier sum (r = momentum r + d), its length over the whole latent kept at most
//     norm_threshold, and its component along c weighed by eta (the orthogonal one kept):
//     den = c + w (r_ortho + eta r_par).
//   CFGZero* (Fan et al.): with alpha the projection of the prompt's velocity on the negative one's,
//     <v_c, v_u> / |v_u|^2 over the whole latent, den = alpha u + w (c - alpha u).
//   Skip Layer Guidance: while the model's sigma is within [sigma_end, sigma_start], a third pass of the prompt with
//     the blocks `layers` left out (k): den += scale (c - k); rescaling_scale then mixes den towards c's spread
//     (Lin et al.'s rescale, by the sample standard deviations over the whole latent).
//   CFG norm: at each latent position, den's norm over the channels is brought down to c's where it is larger, then
//     den is scaled by strength.
//   Detail Daemon (after muerrilla's sd-webui-detail-daemon and Jonseed's ComfyUI-Detail-Daemon, MIT): the sampler's
//     own sigmas and steps stay; the model is asked at sigma (1 - 0.1 d cfg), d the detail schedule's value (made over
//     the steps, looked up and interpolated at the sigma), cfg its override or the request's scale (1 without
//     guidance).  Every prediction above (and the window of skip layer guidance) then goes by that sigma.
// The order is fixed: plain CFG or APG, CFGZero*, Skip Layer Guidance, CFG norm; each later one reads c as the model
// gave it.  CFGZero* and APG need the negative pass (they do nothing without one here; the server refuses them);
// CFG norm, skip layer guidance and Detail Daemon work at CFG 1 as well.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::image {

struct GuidanceSpec {
    float cfgnorm = -1.0f;   // CFG norm's strength; below 0 off
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
    float* cd_ = nullptr;    // the prompt's prediction
    float* ud_ = nullptr;    // the negative prompt's
    float* run_ = nullptr;   // APG's sum of the differences over the steps
    float* part_ = nullptr;  // the reductions' partial sums
    float* red_ = nullptr;   // their sums
    size_t wg_ = 0, groups_ = 0;
    int calls_ = 0, seq_ = 0;
};

}  // namespace strata::image
