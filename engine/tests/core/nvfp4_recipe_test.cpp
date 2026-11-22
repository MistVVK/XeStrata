// SPDX-License-Identifier: MIT
// tests/core/nvfp4_recipe_test.cpp - unit tests for lossless NVFP4 hybrid layer eligibility, `<stem>.input_scale`
// companion naming, and calibrated W4A4 vs W4A8 fallback classification.

#include "strata/quant/nvfp4_recipe.hpp"
#include "strata/prefill/nvfp4_prefill_mode.hpp"

#include <cassert>
#include <cstdio>
#include <limits>

namespace {

using strata::quant::Nvfp4Dispatch;
using strata::quant::Type;
using strata::quant::classify_nvfp4_tensor;
using strata::quant::nvfp4_input_scale_name;
using strata::quant::nvfp4_layer_eligible;
using strata::quant::nvfp4_layer_is_sensitive;

void test_input_scale_naming() {
    assert(nvfp4_input_scale_name("blk.0.ffn_gate_exps.weight") == "blk.0.ffn_gate_exps.input_scale");
    assert(nvfp4_input_scale_name("model.layers.12.mlp.experts.up_proj.weight") ==
           "model.layers.12.mlp.experts.up_proj.input_scale");
    assert(nvfp4_input_scale_name("blk.3.ffn_down_exps") == "blk.3.ffn_down_exps.input_scale");
}

void test_sensitive_vs_eligible_layers() {
    // Sensitive layers MUST stay >= 8-bit / BF16
    assert(nvfp4_layer_is_sensitive("token_embd.weight"));
    assert(nvfp4_layer_is_sensitive("model.embed_tokens.weight"));
    assert(nvfp4_layer_is_sensitive("output.weight"));
    assert(nvfp4_layer_is_sensitive("lm_head.weight"));
    assert(nvfp4_layer_is_sensitive("blk.5.attn_norm.weight"));
    assert(nvfp4_layer_is_sensitive("blk.5.ffn_gate_inp.weight"));
    assert(nvfp4_layer_is_sensitive("blk.5.ssm_a"));
    assert(nvfp4_layer_is_sensitive("blk.5.ssm_conv1d.weight"));
    assert(nvfp4_layer_is_sensitive("blk.5.attn_qkv.weight"));
    assert(nvfp4_layer_is_sensitive("model.layers.4.self_attn.kv_b_proj.weight"));
    assert(nvfp4_layer_is_sensitive("model.layers.4.mlp.gate.weight"));

    // Routed MoE expert projections ARE eligible for calibrated NVFP4
    assert(nvfp4_layer_eligible("blk.5.ffn_gate_exps.weight"));
    assert(nvfp4_layer_eligible("blk.5.ffn_up_exps.weight"));
    assert(nvfp4_layer_eligible("blk.5.ffn_down_exps.weight"));
    assert(nvfp4_layer_eligible("model.layers.4.mlp.experts.gate_proj.weight"));
    assert(nvfp4_layer_eligible("model.layers.4.mlp.experts.down_proj.weight"));
}

void test_dispatch_classification() {
    const int nvfp4 = static_cast<int>(Type::NVFP4);
    const int q4_k = static_cast<int>(Type::Q4_K);

    // Non-NVFP4 tensor -> NotNvfp4
    assert(classify_nvfp4_tensor(q4_k, 12, true, 0.25f) == Nvfp4Dispatch::NotNvfp4);

    // Blackwell sm_120 with finite positive <stem>.input_scale -> CalibratedW4A4
    assert(classify_nvfp4_tensor(nvfp4, 12, true, 0.015625f) == Nvfp4Dispatch::CalibratedW4A4);
    assert(classify_nvfp4_tensor(nvfp4, 10, true, 1.0f) == Nvfp4Dispatch::CalibratedW4A4);

    // Missing <stem>.input_scale on sm_120 -> MUST fall back to W4A8, NEVER uncalibrated W4A4!
    assert(classify_nvfp4_tensor(nvfp4, 12, false, 0.0f) == Nvfp4Dispatch::FallbackW4A8);

    // Pre-Blackwell (sm_80 / sm_89 / sm_90) even with input_scale -> FallbackW4A8
    assert(classify_nvfp4_tensor(nvfp4, 8, true, 0.5f) == Nvfp4Dispatch::FallbackW4A8);
    assert(classify_nvfp4_tensor(nvfp4, 9, true, 0.5f) == Nvfp4Dispatch::FallbackW4A8);

    // Corrupt / zero / negative / NaN / Inf scale -> InvalidScale
    assert(classify_nvfp4_tensor(nvfp4, 12, true, 0.0f) == Nvfp4Dispatch::InvalidScale);
    assert(classify_nvfp4_tensor(nvfp4, 12, true, -1.0f) == Nvfp4Dispatch::InvalidScale);
    assert(classify_nvfp4_tensor(nvfp4, 12, true, std::numeric_limits<float>::quiet_NaN()) ==
           Nvfp4Dispatch::InvalidScale);
    assert(classify_nvfp4_tensor(nvfp4, 12, true, std::numeric_limits<float>::infinity()) ==
           Nvfp4Dispatch::InvalidScale);
}

void test_prefill_mode_input_scale_guard() {
    using strata::prefill::Nvfp4PrefillPath;
    using strata::prefill::resolve_nvfp4_prefill_path;
    bool deg = false;
    // Calibrated W4A4 with input_scale present on sm_120 -> FusedW4A4
    assert(resolve_nvfp4_prefill_path("w4a4", true, false, &deg, true) == Nvfp4PrefillPath::FusedW4A4);
    assert(!deg);
    // Missing <stem>.input_scale on sm_120 -> degrades safely to FusedW4A8
    assert(resolve_nvfp4_prefill_path("w4a4", true, false, &deg, false) == Nvfp4PrefillPath::FusedW4A8);
    assert(deg);
    // Default env (unset) stays W4A8
    assert(resolve_nvfp4_prefill_path(nullptr, true, false, &deg, true) == Nvfp4PrefillPath::FusedW4A8);
    assert(!deg);
}

}  // namespace

int main() {
    test_input_scale_naming();
    test_sensitive_vs_eligible_layers();
    test_dispatch_classification();
    test_prefill_mode_input_scale_guard();
    std::printf("nvfp4_recipe_test: ALL 4 SUITES PASSED\n");
    return 0;
}
