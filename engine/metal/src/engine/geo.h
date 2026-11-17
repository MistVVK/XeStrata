// metal/src/engine/geo.h - the shape the Metal engine is compiled for, from strata::core::kCompiledGeometry (cmake -DSTRATA_GEOMETRY /
// -DSTRATA_GEOMETRY_FILE, include/strata/core/geometry.hpp), and the #define prelude that hands it to the kernels.
//
// Host side: window.cpp, prompt_chain.inc and stream.cpp take their sizes from here instead of from literals (hidden 2560, 640 wide
// experts, 48 layers, 24 query heads, ...).  For Flash-Next's header every value below is the number it replaced.
//
// Kernel side: the MSL sources read their shape from GEO_* macros with Flash-Next's value as the default (`#ifndef GEO_QH`); the
// prelude defines a macro only when this build's value differs from Flash-Next's, so a Flash-Next build hands the compiler exactly
// the prelude it always did.  Everything else in the kernels is run-time (the arguments carry n_embd, n_ff, hc_lr, the head counts of
// the GDN) or a held-fixed constant of the family (state 128, head width 256, 4 hyper-connection streams, indexer 4 x 128).
#pragma once

#include "strata/core/geometry.hpp"

#include <string>

namespace strata_metal::geo {

inline constexpr const strata::core::ModelGeometry& G = strata::core::kCompiledGeometry;

inline constexpr int N = (int) G.n_embd;             // hidden
inline constexpr int HC = (int) G.hc;                // hyper-connection streams (4)
inline constexpr int LR = (int) G.hc_lr;             // hyper-connection low rank
inline constexpr int NFF = (int) G.n_ff;             // routed (and shared) expert width
inline constexpr int TOPK = (int) G.n_expert_used;   // experts per token
inline constexpr int NEXP = (int) G.n_expert;        // routed experts per layer
inline constexpr int NL = (int) G.n_layers;          // trunk layers
inline constexpr int QI = (int) G.qsa_interval;      // every QI-th layer is full attention (QI - 1, 2 QI - 1, ...)
inline constexpr int NQSA = NL / QI;                 // full-attention layers (12)
inline constexpr int NGDN = NL - NQSA;               // GDN layers (36)

// GDN
inline constexpr int SS = (int) G.ssm_state_size;    // state width (128)
inline constexpr int HK = (int) G.ssm_k_heads;       // key heads (16)
inline constexpr int HV = (int) G.ssm_v_heads;       // value heads (48)
inline constexpr int CONV = (int) G.ssm_conv_channels;   // 2 * SS * HK + SS * HV = q|k|v channels (10240)
inline constexpr int ZV = (int) G.ssm_value_dim;     // SS * HV (6144): z, the readout y

// full attention
inline constexpr int QH = (int) G.n_head;            // query heads (24)
inline constexpr int QKV = (int) G.n_head_kv;        // kv heads (2)
inline constexpr int QD = (int) G.head_dim;          // head width (256)
inline constexpr int IH = (int) G.idx_q_heads;       // indexer heads (4)
inline constexpr int ID = (int) G.idx_key_dim;       // indexer key width (128)
inline constexpr int BUD = (int) G.idx_top_k + 3;     // selection budget in cells (2051): top_k + the tail block's 3
inline constexpr int QCH = 64;                       // cells per attention chunk (qsa.metal QCH)
inline constexpr int QCHUNKS = (BUD + QCH - 1) / QCH;   // 33
inline constexpr int QCHUNKS2 = (BUD + 31) / 32;         // 65: the same cells in chunks of 32 (qsa2.metal), the partials buffer's chunk count
inline constexpr int QPREP = QH + 2 * QKV + IH + 1;  // qsa_prep threadgroups: q heads, k, v, indexer q, and the indexer-key group

// derived sizes (floats unless said)
inline constexpr int HCD = HC * N;                   // one residual row (hc_dim) = the PLE key width (10240)
inline constexpr int QFULL = QH * QD * 2;            // attn_q: query and gate (12288)
inline constexpr int QOUT = QH * QD;                 // the attention output row = attn_output's K (6144)
inline constexpr int KV = QKV * QD;                  // attn_k / attn_v rows (512)
inline constexpr int IQR = IH * ID;                  // indexer q_proj rows (512)
inline constexpr int GDN_FUSED = CONV + ZV + 2 * HV; // the four GDN input projections as one row (16480)

// the kernels' defaults are Flash-Next's: a macro is defined only for what this build changes
inline std::string prelude() {
    const strata::core::ModelGeometry d;
    std::string s;
    auto def = [&](const char* name, int64_t v, int64_t dv) {
        if (v != dv) s += std::string("#define ") + name + " " + std::to_string((long long) v) + "\n";
    };
    def("GEO_QH", G.n_head, d.n_head);
    def("GEO_QKV", G.n_head_kv, d.n_head_kv);
    def("GEO_BUD", G.idx_top_k + 3, d.idx_top_k + 3);
    def("GEO_TOPK", G.n_expert_used, d.n_expert_used);
    def("GEO_NJ", G.n_embd / 256, d.n_embd / 256);                      // gr_norm2: elements per thread of one stream's row
    def("GEO_GRU_NU", (G.hc_lr / 32 + 3) / 4, (d.hc_lr / 32 + 3) / 4);   // gr_up prefetch: units of 32 per lane (4 lanes a row)
    return s;
}

}  // namespace strata_metal::geo
