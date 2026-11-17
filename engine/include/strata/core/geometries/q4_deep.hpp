// include/strata/core/geometries/q4_deep.hpp - Flash-Next with 64 layers (G4): only the layer count differs
// A synthetic target (phaseA-tests/qwen4/GEOMETRY-MAP.md section 5), not a published model.
#pragma once

namespace strata::core {

inline constexpr const char* kGeometryName = "q4_deep";
inline constexpr ModelGeometry kCompiledGeometry = [] {
    ModelGeometry g;
    g.n_embd = 2560;
    g.n_layers = 64;
    g.n_head = 24;
    g.n_head_kv = 2;
    g.ssm_v_heads = 48;
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_state_size * g.ssm_v_heads;
    g.ssm_value_dim = g.ssm_state_size * g.ssm_v_heads;
    g.hc_lr = 320;
    g.n_expert = 512;
    g.n_expert_used = 10;
    g.n_ff = 640;
    g.n_ff_shexp = 640;
    return g;
}();

}  // namespace strata::core
