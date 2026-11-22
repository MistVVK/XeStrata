// SPDX-License-Identifier: MIT
// include/strata/quant/nvfp4_recipe.hpp - lossless NVFP4 (Blackwell sm_120+ W4A4 / W4A8) quantization & load guard.
//
// Why this exists:
//   1. Naive full-model NVFP4 collapses MoE routing, attention Q/K/V, GDN/SSM recurrence, and output heads; keeping
//      those sensitive tensors in >= Q8_0 / BF16 while quantizing routed MoE expert projections (ffn_gate_exps,
//      ffn_up_exps, ffn_down_exps) to NVFP4 with calibrated per-tensor `<stem>.input_scale` preserves quality while
//      enabling Blackwell `mma.sync.aligned.block_scale.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` W4A4 prefill GEMMs.
//   2. On Blackwell W4A4 prefill, activations are scaled by `1 / input_scale` into E2M1 x UE4M3 (`max = 6.0 * 448.0 =
//      2688.0`).  An NVFP4 weight tensor without a finite positive `<stem>.input_scale` companion tensor cannot run
//      calibrated W4A4 prefill safely and must either be rejected by the converter or fall back to W4A8 (`Q8_0` act).
#pragma once

#include "strata/quant/types.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>

namespace strata::quant {

inline constexpr float kNvfp4MaxRepresentable = 6.0f * 448.0f;   // E2M1 max (6.0) * UE4M3 max (448.0) = 2688.0

enum class Nvfp4Dispatch : uint8_t {
    NotNvfp4 = 0,       ///< Tensor is not Type::NVFP4 (40)
    CalibratedW4A4 = 1, ///< sm_120+ with valid <stem>.input_scale: native block_scale E2M1xE2M1 W4A4 MMA
    FallbackW4A8 = 2,   ///< Missing input_scale or pre-sm_120: dequant/LUT W4A8 dot (never uncalibrated W4A4!)
    InvalidScale = 3,   ///< Present <stem>.input_scale is <= 0, NaN, or Inf -> refuse load
};

/// Maps `"blk.3.ffn_gate_exps.weight"` -> `"blk.3.ffn_gate_exps.input_scale"`.
inline std::string nvfp4_input_scale_name(std::string_view weight_tensor_name) {
    constexpr std::string_view kWeightSuffix = ".weight";
    if (weight_tensor_name.size() > kWeightSuffix.size() &&
        weight_tensor_name.substr(weight_tensor_name.size() - kWeightSuffix.size()) == kWeightSuffix) {
        std::string out(weight_tensor_name.substr(0, weight_tensor_name.size() - kWeightSuffix.size()));
        out += ".input_scale";
        return out;
    }
    std::string out(weight_tensor_name);
    out += ".input_scale";
    return out;
}

namespace detail {

inline bool contains_sub(std::string_view haystack, std::string_view needle) noexcept {
    return haystack.find(needle) != std::string_view::npos;
}

}  // namespace detail

/// Returns true for tensors that MUST NOT be quantized to 4-bit NVFP4 in a lossless hybrid recipe:
/// embeddings, LM head, norms, MoE routers/gates, SSM/GDN state recurrence, and MLA/QSA attention projections.
inline bool nvfp4_layer_is_sensitive(std::string_view tensor_name) noexcept {
    using detail::contains_sub;
    // Embeddings, output head, and norms
    if (contains_sub(tensor_name, "token_embd") || contains_sub(tensor_name, "embed_tokens") ||
        tensor_name == "output.weight" || contains_sub(tensor_name, "lm_head") ||
        contains_sub(tensor_name, "norm")) {
        return true;
    }
    // MoE router gates and shared-expert scalar gates (must NOT match ffn_gate_exps / routed expert gate_proj!)
    if (contains_sub(tensor_name, "ffn_gate_inp") || contains_sub(tensor_name, "shared_expert_gate") ||
        (contains_sub(tensor_name, ".mlp.gate.") && !contains_sub(tensor_name, "gate_proj"))) {
        return true;
    }
    // GDN / Mamba / SSM recurrence & conv1d state
    if (contains_sub(tensor_name, "ssm_") || contains_sub(tensor_name, "conv1d") ||
        contains_sub(tensor_name, "time_mix")) {
        return true;
    }
    // Attention / MLA / QSA / Indexer / Engram / Hyper-connection projections
    if (contains_sub(tensor_name, "attn_") || contains_sub(tensor_name, "self_attn") ||
        contains_sub(tensor_name, "indexer") || contains_sub(tensor_name, "engram") ||
        contains_sub(tensor_name, "hc_")) {
        return true;
    }
    return false;
}

/// Returns true when `tensor_name` is an eligible MoE expert body or dense FFN weight for lossless NVFP4.
inline bool nvfp4_layer_eligible(std::string_view tensor_name) noexcept {
    return !nvfp4_layer_is_sensitive(tensor_name);
}

/// Validates an NVFP4 tensor's runtime dispatch path from its GGUF type, hardware SM major/minor, and `<stem>.input_scale`.
inline Nvfp4Dispatch classify_nvfp4_tensor(int gguf_type,
                                           int sm_major,
                                           bool has_input_scale,
                                           float input_scale_value = 1.0f) noexcept {
    if (gguf_type != static_cast<int>(Type::NVFP4)) {
        return Nvfp4Dispatch::NotNvfp4;
    }
    if (has_input_scale) {
        if (!std::isfinite(input_scale_value) || input_scale_value <= 0.0f) {
            return Nvfp4Dispatch::InvalidScale;
        }
        if (sm_major >= 10) {   // sm_100 / sm_120 / sm_121 Blackwell block-scaled MMA
            return Nvfp4Dispatch::CalibratedW4A4;
        }
    }
    return Nvfp4Dispatch::FallbackW4A8;
}

}  // namespace strata::quant
