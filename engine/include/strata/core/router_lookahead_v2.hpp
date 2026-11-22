// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Harin Mani Karri
// Copyright (c) 2026 Strata Contributors
//
// include/strata/core/router_lookahead_v2.hpp - 2-Layer Router Lookahead (L+1, L+2) with
// Shared-Expert Residual Input, Folded Router Weight Helper, and Zero-Drop Latest-Submission Mailbox.

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace strata::core {

/// Combine post-attention residual `x_post_attn` with optional shared-expert output `shared_out`
/// to form the higher-accuracy prediction activation for layer L+1 and L+2 router lookahead:
///   x_pred[i] = x_post_attn[i] + shared_scale * shared_out[i]
inline void build_lookahead_residual_input(
    const float* x_post_attn, const float* shared_out, float* x_pred,
    int64_t n_embd, float shared_scale = 1.0f) noexcept {
    if (!x_pred || !x_post_attn || n_embd <= 0) return;
    if (!shared_out || shared_scale == 0.0f) {
        std::memcpy(x_pred, x_post_attn, static_cast<size_t>(n_embd) * sizeof(float));
        return;
    }
    for (int64_t i = 0; i < n_embd; ++i) {
        x_pred[i] = x_post_attn[i] + shared_scale * shared_out[i];
    }
}

/// Optional folded router weights: W'(e, d) = W(e, d) * norm_gain(d) + bias_t(e, d)
/// Pre-folds RMSNorm gain into the lookahead router matrix so prediction skips a separate norm pass.
inline void fold_router_weights_f32(
    const float* router_w, const float* norm_gain, float* out_folded,
    int64_t n_expert, int64_t n_embd) noexcept {
    if (!router_w || !out_folded || n_expert <= 0 || n_embd <= 0) return;
    for (int64_t e = 0; e < n_expert; ++e) {
        const float* src = router_w + e * n_embd;
        float* dst = out_folded + e * n_embd;
        if (norm_gain) {
            for (int64_t d = 0; d < n_embd; ++d) dst[d] = src[d] * norm_gain[d];
        } else {
            std::memcpy(dst, src, static_cast<size_t>(n_embd) * sizeof(float));
        }
    }
}

/// Predict top-k routed experts for a single target layer (L+1 or L+2) from `x_pred` using FP32
/// router weights `[n_expert, n_embd]` + optional bias `[n_expert]`. Returns experts sorted by
/// descending score along with normalized confidence scores in [0, 1].
struct LookaheadPrediction {
    int64_t layer = -1;
    int     ahead = 1;  ///< 1 for L+1, 2 for L+2
    std::vector<int64_t> experts;
    std::vector<float>   confidences;
};

inline LookaheadPrediction predict_router_topk_f32(
    int64_t target_layer, int ahead, const float* x_pred, const float* router_w, const float* router_bias,
    int64_t n_embd, int64_t n_expert, int top_k, bool use_sigmoid = false) {
    LookaheadPrediction out{};
    out.layer = target_layer;
    out.ahead = ahead;
    if (!x_pred || !router_w || n_embd <= 0 || n_expert <= 0 || top_k <= 0) return out;
    const int k = static_cast<int>(std::min<int64_t>(top_k, n_expert));

    std::vector<std::pair<float, int64_t>> sc(static_cast<size_t>(n_expert));
    float max_logit = -1e30f;
    for (int64_t e = 0; e < n_expert; ++e) {
        const float* w = router_w + e * n_embd;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int64_t d = 0;
        for (; d + 4 <= n_embd; d += 4) {
            s0 += w[d + 0] * x_pred[d + 0];
            s1 += w[d + 1] * x_pred[d + 1];
            s2 += w[d + 2] * x_pred[d + 2];
            s3 += w[d + 3] * x_pred[d + 3];
        }
        float s = (s0 + s1) + (s2 + s3);
        for (; d < n_embd; ++d) s += w[d] * x_pred[d];
        if (router_bias) s += router_bias[e];
        if (use_sigmoid) {
            s = 1.0f / (1.0f + std::exp(-s));
        }
        if (s > max_logit) max_logit = s;
        sc[static_cast<size_t>(e)] = {s, e};
    }

    std::partial_sort(sc.begin(), sc.begin() + k, sc.end(),
                      [](const auto& a, const auto& b) {
                          return (a.first != b.first) ? (a.first > b.first) : (a.second < b.second);
                      });

    out.experts.resize(static_cast<size_t>(k));
    out.confidences.resize(static_cast<size_t>(k));
    if (use_sigmoid) {
        for (int i = 0; i < k; ++i) {
            out.experts[static_cast<size_t>(i)] = sc[static_cast<size_t>(i)].second;
            out.confidences[static_cast<size_t>(i)] = std::clamp(sc[static_cast<size_t>(i)].first, 0.0f, 1.0f);
        }
    } else {
        // Softmax confidence over top-k window relative to max_logit
        float denom = 0.0f;
        for (int i = 0; i < k; ++i) denom += std::exp(sc[static_cast<size_t>(i)].first - max_logit);
        const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (int i = 0; i < k; ++i) {
            out.experts[static_cast<size_t>(i)] = sc[static_cast<size_t>(i)].second;
            out.confidences[static_cast<size_t>(i)] = std::exp(sc[static_cast<size_t>(i)].first - max_logit) * inv;
        }
    }
    return out;
}

/// Non-dropping latest-submission mailbox for asynchronous lookahead workers.
/// Instead of failing `try_lock` and dropping submissions (`skipped_`), overwrites the pending
/// slot with the newest layer's activation so the worker ALWAYS runs the freshest L+1/L+2 job.
template <typename JobT>
class LatestLookaheadMailbox {
public:
    /// Overwrites any unconsumed pending job; never blocks on worker execution and never drops the latest job.
    void submit(JobT job) {
        std::lock_guard<std::mutex> lk(mu_);
        if (has_pending_) overwritten_.fetch_add(1, std::memory_order_relaxed);
        pending_ = std::move(job);
        has_pending_ = true;
        submitted_.fetch_add(1, std::memory_order_relaxed);
    }

    bool take(JobT& out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!has_pending_) return false;
        out = std::move(pending_);
        has_pending_ = false;
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lk(mu_);
        return !has_pending_;
    }

    uint64_t submitted()   const noexcept { return submitted_.load(std::memory_order_relaxed); }
    uint64_t overwritten() const noexcept { return overwritten_.load(std::memory_order_relaxed); }

private:
    mutable std::mutex mu_;
    JobT pending_{};
    bool has_pending_ = false;
    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> overwritten_{0};
};

}  // namespace strata::core
