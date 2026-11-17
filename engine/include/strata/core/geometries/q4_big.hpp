// include/strata/core/geometries/q4_big.hpp - G3, the 397B-A17B class: hidden 4096, 60 layers, 32:2 heads, 64 GDN value heads, 512 experts top-10, ff 1024, hc low rank 512
// A synthetic target (phaseA-tests/qwen4/GEOMETRY-MAP.md section 5), not a published model.
#pragma once

namespace strata::core {

inline constexpr const char* kGeometryName = "q4_big";
inline constexpr ModelGeometry kCompiledGeometry = [] {
    ModelGeometry g;
    g.n_embd = 4096;
    g.n_layers = 60;
    g.n_head = 32;
    g.n_head_kv = 2;
    g.ssm_v_heads = 64;
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_state_size * g.ssm_v_heads;
    g.ssm_value_dim = g.ssm_state_size * g.ssm_v_heads;
    g.hc_lr = 512;
    g.n_expert = 512;
    g.n_expert_used = 10;
    g.n_ff = 1024;
    g.n_ff_shexp = 1024;
    return g;
}();

}  // namespace strata::core
