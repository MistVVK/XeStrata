// include/strata/core/geometries/q4_lite.hpp - G1, the 35B-A3B class scaled onto Flash-Next's layer set: hidden 2048, 40 layers, 16:2 heads, 32 GDN value heads, 256 experts top-8, ff 512, hc low rank 256
// A synthetic target (phaseA-tests/qwen4/GEOMETRY-MAP.md section 5), not a published model.
#pragma once

namespace strata::core {

inline constexpr const char* kGeometryName = "q4_lite";
inline constexpr ModelGeometry kCompiledGeometry = [] {
    ModelGeometry g;
    g.n_embd = 2048;
    g.n_layers = 40;
    g.n_head = 16;
    g.n_head_kv = 2;
    g.ssm_v_heads = 32;
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_state_size * g.ssm_v_heads;
    g.ssm_value_dim = g.ssm_state_size * g.ssm_v_heads;
    g.hc_lr = 256;
    g.n_expert = 256;
    g.n_expert_used = 8;
    g.n_ff = 512;
    g.n_ff_shexp = 512;
    return g;
}();

}  // namespace strata::core
