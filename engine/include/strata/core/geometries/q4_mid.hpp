// include/strata/core/geometries/q4_mid.hpp - G2, the 122B-A10B class: hidden 3072, 48 layers, 32:2 heads, 64 GDN value heads, 256 experts top-8, ff 1024, hc low rank 384
// A synthetic target (phaseA-tests/qwen4/GEOMETRY-MAP.md section 5), not a published model.
#pragma once

namespace strata::core {

inline constexpr const char* kGeometryName = "q4_mid";
inline constexpr ModelGeometry kCompiledGeometry = [] {
    ModelGeometry g;
    g.n_embd = 3072;
    g.n_layers = 48;
    g.n_head = 32;
    g.n_head_kv = 2;
    g.ssm_v_heads = 64;
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_state_size * g.ssm_v_heads;
    g.ssm_value_dim = g.ssm_state_size * g.ssm_v_heads;
    g.hc_lr = 384;
    g.n_expert = 256;
    g.n_expert_used = 8;
    g.n_ff = 1024;
    g.n_ff_shexp = 1024;
    return g;
}();

}  // namespace strata::core
