// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/artifact/model_spec.hpp - a model's shape, read from its GGUF metadata.
//
// The engine was built for one model, and `ModelGeometry` (core/layout.hpp) holds that model's numbers as
// defaults.  This reads the shape of every architecture XeStrata is being taught, by the key names llama.cpp
// writes (`src/llama-arch.cpp` at de7fa0a3), so that a second model is a second set of read values and not a
// second set of constants:
//
//   qwen4exp   Qwen3.8-Flash-Next and its fine-tunes (the engine's model)
//   qwen35moe  Qwen3.5 MoE (Ornith-1.5-35B-A3B): gated delta-net + full attention every 4th layer, MTP
//   qwen35     Qwen3.5 (Qwen3.5-9B): qwen35moe's block with a dense SwiGLU FFN in place of the MoE, MTP
//   sensenova_loop SenseNova-Loop-9B: dense Qwen3.5 blocks visited in a loop, with mHC
//   qwen3moe   Qwen3 MoE (llm-jp-4.1-32b-a3b): full attention only, no shared expert
//   plamo3     PLaMo 3 (PLaMo 3 Translate): full attention, sliding-window layers between every 8th, fused QKV and
//              gate|up, RMSNorms after the attention and the FFN as well as before
//   glm5-next  GLM-5.3-Flash: KDA + NoPE MLA with a k-pool indexer, mHC, leading dense layers, MTP.  Also spelled
//              `glm5next` (Project Maya's quants, after unsloth's first GGUFs), with the same keys under that name.
//   inkling2_loop Inkling-2-Mini (Thinking Machines): hybrid sliding-window / relative-attention layers with short
//              convolutions, AttnRes every 6 layers, sigmoid-gated MoE and a 3-pass loop over core layers 6-17.
//
// `block_count` counts the MTP (NextN) layers too; `n_layer` here is the trunk, block_count minus
// `nextn_predict_layers`, as llama.cpp's `n_layer()`.  A key a model does not carry reads as 0 (or false), and
// the keys an architecture cannot run without are required: a missing one is an error naming the key.
//
// Header-only, like gguf_reader.hpp, and with no ggml or SYCL dependency.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata {

enum class ModelArch { Qwen4Exp, Qwen35Moe, Qwen3Moe, Glm5Next, Qwen35, Plamo3, Qwen3, Inkling2Loop, SenseNovaLoop };

/// llama.cpp's `llama_expert_gating_func_type`
enum class ExpertGating : uint32_t { None = 0, Softmax = 1, Sigmoid = 2 };

struct ModelSpec {
    ModelArch arch = ModelArch::Qwen4Exp;
    std::string arch_name;                ///< general.architecture

    int64_t n_layer = 0;                  ///< trunk layers (block_count - nextn_predict_layers)
    int64_t n_nextn = 0;                  ///< MTP layers after the trunk
    int64_t n_embd = 0;
    int64_t context_length = 0;
    double rms_eps = 0;
    double norm_eps = 0;                  ///< attention.layer_norm_epsilon: the LayerNorms' (GLM-5.3-Flash's indexer keys)

    // attention.  head_count_kv may be per layer (GLM-5.3-Flash writes 0 on its recurrent layers).
    int64_t n_head = 0;
    std::vector<int64_t> n_head_kv;       ///< per layer, block_count entries
    int64_t key_length = 0, value_length = 0;
    int64_t rope_dims = 0;                ///< rotated dimensions of a head (llama.cpp's default: the whole key)
    std::vector<int64_t> rope_sections;   ///< MRoPE sections, empty when not written
    double rope_base = 0;
    std::string rope_scaling_type;
    double rope_scaling_factor = 0, rope_scaling_orig_ctx = 0;
    double yarn_beta_fast = 0, yarn_beta_slow = 0;   ///< rope.scaling.yarn_beta_*: 0 when not written

    // sliding-window attention (plamo3).  `swa` is per layer: true for a layer that sees the last `swa_window`
    // positions only, rotated with `rope_base_swa` and no scaling (llama.cpp's freq_base_swa, freq_scale_swa 1).
    int64_t swa_window = 0;
    std::vector<bool> swa;                ///< block_count entries, all false without a window
    double rope_base_swa = 0;
    /// True when trunk layer `l` is a sliding-window layer.
    bool layer_swa(int64_t l) const { return swa_window > 0 && l >= 0 && l < (int64_t) swa.size() && swa[(size_t) l]; }
    /// A dense model: a SwiGLU FFN in every block and no experts.
    bool dense() const {
        return arch == ModelArch::Qwen35 || arch == ModelArch::Plamo3 || arch == ModelArch::Qwen3 ||
               arch == ModelArch::SenseNovaLoop;
    }

    // inkling2_loop (Inkling-2-Mini).  No RoPE: the relative-position bias carries the positions.  The GGUF's keys
    // are those mainline llama.cpp writes in src/llama-arch.cpp (PR #32694).  The loop expands the trunk's 24 physical
    // layers to 48 cache slots; the extra 24 share their weights with core layers 6-17 but keep their own K/V and
    // convolution state.  `layer_phys` is the slot -> physical layer map of llama_model_inkling2_loop::layer_phys.
    // loop_start, loop_end and loop_count are also SenseNova-Loop's (below): both write <arch>.loop.start, .end and
    // .count, the physical layers [loop_start, loop_end) run loop_count times.  Each architecture reads and checks
    // them in its own branch of read_model_spec, and each accessor answers for its own architecture only.
    int64_t loop_start = 0, loop_end = 0, loop_count = 0, loop_gate_rank = 0;
    bool loop_varying = false;
    int64_t attn_res_block = 0;                 ///< inkling2_loop.attn_res.block_size
    int64_t sconv_l_cache = 0;                  ///< inkling2_loop.shortconv.l_cache (kernel size)
    int64_t relative_length = 0;                ///< d_rel
    int64_t relative_extent = 0;                ///< the distances the bias table covers
    double relative_clip = 0;                   ///< L2 cap of the relative vectors
    double qk_clip = 0;                         ///< L2 cap of Q and K
    int64_t log_scale_floor = 0;
    double log_scale_alpha = 0;
    double situ_beta = 0, situ_linear_beta = 0; ///< beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
    double final_logit_softcap = 0;
    int64_t vocab_padded = 0, vocab_unpadded = 0;
    bool expert_gate_bias = false;              ///< blk.*.exp_probs_b.bias present

    int64_t n_slot() const {
        if (arch != ModelArch::Inkling2Loop || loop_count <= 0 || loop_end <= loop_start) return n_layer;
        return n_layer + (loop_count - 1) * (loop_end - loop_start);
    }
    int64_t layer_phys(int64_t il) const {
        if (arch != ModelArch::Inkling2Loop || il < n_layer || loop_end <= loop_start) return il;
        return loop_start + (il - n_layer) % (loop_end - loop_start);
    }
    /// True when cache slot `il` is a copy of a physical core layer (its weights are shared, its state is not).
    bool loop_slot(int64_t il) const { return arch == ModelArch::Inkling2Loop && il >= n_layer && il < n_slot(); }

    // linear attention.  `recurrent` is per layer: true for a gated delta-net / KDA layer.
    std::vector<bool> recurrent;          ///< block_count entries; the MTP layers are false
    int64_t full_attention_interval = 0;
    int64_t ssm_d_conv = 0, ssm_d_inner = 0, ssm_d_state = 0;
    int64_t ssm_dt_rank = 0;              ///< the value heads of a gated delta-net
    int64_t ssm_n_group = 0;              ///< its key heads
    int64_t kda_head_dim = 0;
    double kda_gate_lower_bound = 0;

    // SenseNova-Loop: layer numbers are visits; tensor names after the loop use the visit number.  The loop is
    // loop_start, loop_end and loop_count above; here n_layer counts the visits, inkling2_loop's the physical layers.
    int64_t mhc_start = 0, mhc_end = 0, mhc_streams = 0, mhc_iterations = 0;
    bool mhc_fixed_identity = false;
    double mhc_pre_scale = 0;
    int64_t physical_layer(int64_t visit) const {
        if (arch != ModelArch::SenseNovaLoop || visit < loop_start || visit >= n_layer) return visit;
        const int64_t width = loop_end - loop_start, extra = (loop_count - 1) * width;
        return visit < loop_end + extra ? loop_start + (visit - loop_start) % width : visit - extra;
    }
    bool mhc_active_before(int64_t visit) const {
        if (arch != ModelArch::SenseNovaLoop || visit >= n_layer) return false;
        const int64_t physical = physical_layer(visit);
        return physical > mhc_start && physical < mhc_end;
    }
    int64_t tensor_layer(int64_t visit) const {
        const int64_t physical = physical_layer(visit);
        return arch == ModelArch::SenseNovaLoop && physical < loop_end ? physical : visit;
    }

    // MoE
    int64_t n_expert = 0, n_expert_used = 0;
    int64_t n_ff_exp = 0;                 ///< a routed expert's width
    int64_t n_ff_shexp = 0;               ///< the shared expert's width (0: none)
    int64_t n_expert_shared = 0;
    int64_t n_dense_lead = 0;             ///< leading layers with a dense FFN instead of the MoE
    int64_t n_ff = 0;                     ///< the dense FFN's width
    ExpertGating gating = ExpertGating::None;
    bool expert_weights_norm = false;
    double expert_weights_scale = 0;
    std::vector<double> swiglu_clamp_exp, swiglu_clamp_shexp;   ///< per layer, empty when not written

    // hyper-connections (qwen4exp's gated residual, GLM-5.3-Flash's mHC)
    int64_t hc = 0, hc_low_rank = 0, hc_sinkhorn_iters = 0;
    double hc_eps = 0;

    // MLA and the sparse indexer
    int64_t q_lora_rank = 0, kv_lora_rank = 0, key_length_mla = 0, value_length_mla = 0;
    int64_t idx_n_head = 0, idx_key_length = 0, idx_top_k = 0, idx_kpool = 0;
    bool idx_kpool_select_tail = false;

    /// True when trunk layer `l` routes experts (the MTP layers are not counted here).
    bool layer_routes(int64_t l) const { return n_expert > 0 && l >= n_dense_lead && l < n_layer; }
    /// Trunk layers that route experts.
    int64_t n_moe_layers() const { return n_expert > 0 && n_layer > n_dense_lead ? n_layer - n_dense_lead : 0; }
    /// The 0-based index of trunk layer `l` among the layers that route, -1 when it does not route.  An expert's
    /// number in the cache is `moe_index(l) * n_expert + e`, which is `l * n_expert + e` when every layer routes.
    int64_t moe_index(int64_t l) const { return layer_routes(l) ? l - n_dense_lead : -1; }
};

namespace detail {

inline const MetaValue* spec_get(const GgufFile& g, const std::string& arch, const char* key) {
    return g.get(arch + "." + key);
}
inline int64_t spec_int(const GgufFile& g, const std::string& arch, const char* key, int64_t dflt = 0) {
    const MetaValue* v = spec_get(g, arch, key);
    if (!v || !v->is_num()) return dflt;
    return (int64_t) v->u;
}
inline double spec_num(const GgufFile& g, const std::string& arch, const char* key, double dflt = 0) {
    const MetaValue* v = spec_get(g, arch, key);
    return v && v->is_num() ? v->num() : dflt;
}
/// A key written either as one number or as one number per layer (llama.cpp's get_key_or_arr).
inline bool spec_per_layer(const GgufFile& g, const std::string& arch, const char* key, int64_t n,
                           std::vector<double>& out) {
    out.clear();
    const MetaValue* v = spec_get(g, arch, key);
    if (!v) return false;
    if (v->type == MetaType::ARRAY) {
        if ((int64_t) v->items.size() != n) return false;
        for (const MetaValue& e : v->items) out.push_back(e.num());
    } else if (v->is_num()) {
        out.assign((size_t) n, v->num());
    } else {
        return false;
    }
    return true;
}

}  // namespace detail

/// Reads the shape of the model `g` holds (its first shard, which carries the metadata).  Returns false with a
/// reason naming the key or value at fault.
inline bool read_model_spec(const GgufFile& g, ModelSpec& s, std::string& err) {
    using namespace detail;
    s = ModelSpec{};
    const MetaValue* a = g.get("general.architecture");
    if (!a || a->type != MetaType::STRING) { err = "missing general.architecture"; return false; }
    s.arch_name = a->s;
    const std::string& A = s.arch_name;
    if (A == "qwen4exp") s.arch = ModelArch::Qwen4Exp;
    else if (A == "qwen35moe") s.arch = ModelArch::Qwen35Moe;
    else if (A == "qwen35") s.arch = ModelArch::Qwen35;
    else if (A == "sensenova_loop") s.arch = ModelArch::SenseNovaLoop;
    else if (A == "qwen3moe") s.arch = ModelArch::Qwen3Moe;
    else if (A == "glm5-next" || A == "glm5next") s.arch = ModelArch::Glm5Next;
    else if (A == "plamo3") s.arch = ModelArch::Plamo3;
    // Qwen3 and Qwen3-VL's language part (the image models' text encoders): every layer full attention, dense
    else if (A == "qwen3" || A == "qwen3vl") s.arch = ModelArch::Qwen3;
    else if (A == "inkling2_loop") s.arch = ModelArch::Inkling2Loop;
    else {
        err = "architecture '" + A + "' is not supported (qwen4exp, qwen35moe, qwen35, qwen3moe, glm5-next, plamo3, "
              "qwen3, qwen3vl, inkling2_loop, sensenova_loop)";
        return false;
    }

    // the keys every architecture here needs; the experts' all but the dense models'
    const bool dense = s.dense();
    const char* required[] = {"block_count", "embedding_length", "attention.head_count", "expert_count",
                              "expert_used_count", "attention.layer_norm_rms_epsilon"};
    for (const char* k : required)
        if (!(dense && std::strncmp(k, "expert_", 7) == 0) && !spec_get(g, A, k)) {
            err = "missing " + A + "." + k;
            return false;
        }

    const int64_t n_all = spec_int(g, A, "block_count");
    s.n_nextn = spec_int(g, A, "nextn_predict_layers");
    if (n_all <= 0 || s.n_nextn < 0 || s.n_nextn >= n_all) {
        err = A + ".block_count " + std::to_string(n_all) + " with " + std::to_string(s.n_nextn) + " MTP layers";
        return false;
    }
    s.n_layer = n_all - s.n_nextn;
    s.n_embd = spec_int(g, A, "embedding_length");
    s.context_length = spec_int(g, A, "context_length");
    s.rms_eps = spec_num(g, A, "attention.layer_norm_rms_epsilon");
    s.norm_eps = spec_num(g, A, "attention.layer_norm_epsilon");

    s.n_head = spec_int(g, A, "attention.head_count");
    if (s.arch == ModelArch::Inkling2Loop && s.n_head == 0) {
        // inkling2 writes one head count per layer; the engine uses one value and checks the layers below
        std::vector<double> heads;
        if (!spec_per_layer(g, A, "attention.head_count", n_all, heads) || heads.empty()) {
            err = A + ".attention.head_count is neither a number nor one per layer";
            return false;
        }
        s.n_head = (int64_t) heads[0];
        for (size_t i = 1; i < heads.size(); ++i)
            if (heads[i] != heads[0]) {
                err = A + ".attention.head_count differs between layers";
                return false;
            }
    }
    std::vector<double> kv;
    if (spec_get(g, A, "attention.head_count_kv") && !spec_per_layer(g, A, "attention.head_count_kv", n_all, kv)) {
        err = A + ".attention.head_count_kv is neither a number nor one per layer";
        return false;
    }
    s.n_head_kv.assign((size_t) n_all, 0);
    for (size_t i = 0; i < kv.size(); ++i) s.n_head_kv[i] = (int64_t) kv[i];
    // llama.cpp's defaults (llama_model::load_hparams): a head is n_embd / n_head wide unless the file says, and
    // rotated over its whole key unless the file says (qwen3moe writes no rope.dimension_count)
    const int64_t head_w = s.n_head > 0 ? s.n_embd / s.n_head : 0;
    s.key_length = spec_int(g, A, "attention.key_length", head_w);
    s.value_length = spec_int(g, A, "attention.value_length", head_w);
    s.rope_dims = spec_int(g, A, "rope.dimension_count", s.key_length);
    if (const MetaValue* v = spec_get(g, A, "rope.dimension_sections"); v && v->type == MetaType::ARRAY)
        for (const MetaValue& e : v->items) s.rope_sections.push_back((int64_t) e.u);
    s.rope_base = spec_num(g, A, "rope.freq_base");
    if (const MetaValue* v = spec_get(g, A, "rope.scaling.type"); v && v->type == MetaType::STRING)
        s.rope_scaling_type = v->s;
    s.rope_scaling_factor = spec_num(g, A, "rope.scaling.factor");
    s.rope_scaling_orig_ctx = spec_num(g, A, "rope.scaling.original_context_length");
    s.yarn_beta_fast = spec_num(g, A, "rope.scaling.yarn_beta_fast");
    s.yarn_beta_slow = spec_num(g, A, "rope.scaling.yarn_beta_slow");
    // llama.cpp's plamo3.cpp: a window makes every layer a sliding-window one but each pattern-th (the pattern's
    // last, il % n < n - 1: set_swa_pattern without dense_first), 8 when the file writes no pattern
    s.swa.assign((size_t) n_all, false);
    if (s.arch == ModelArch::Plamo3) {
        s.swa_window = spec_int(g, A, "attention.sliding_window");
        s.rope_base_swa = spec_num(g, A, "rope.freq_base_swa", 10000.0);
        if (s.swa_window > 0) {
            std::vector<double> pat;
            const MetaValue* pv = spec_get(g, A, "attention.sliding_window_pattern");
            if (pv && pv->type == MetaType::ARRAY) {
                if (!spec_per_layer(g, A, "attention.sliding_window_pattern", n_all, pat)) {
                    err = A + ".attention.sliding_window_pattern has another length than block_count";
                    return false;
                }
                for (int64_t i = 0; i < n_all; ++i) s.swa[(size_t) i] = pat[(size_t) i] != 0;
            } else {
                const int64_t n = spec_int(g, A, "attention.sliding_window_pattern", 8);
                for (int64_t i = 0; i < n_all; ++i) s.swa[(size_t) i] = n == 0 || i % n < n - 1;
            }
        }
    } else if (s.arch == ModelArch::Inkling2Loop) {
        // inkling2.cpp reads the pattern as one flag per layer; the GGUF writes true for a sliding-window layer
        s.swa_window = spec_int(g, A, "attention.sliding_window");
        std::vector<double> pat;
        if (!spec_per_layer(g, A, "attention.sliding_window_pattern", n_all, pat)) {
            err = A + ".attention.sliding_window_pattern must be one flag per layer";
            return false;
        }
        for (int64_t i = 0; i < n_all; ++i) s.swa[(size_t) i] = pat[(size_t) i] != 0;
    }

    if (s.arch == ModelArch::Inkling2Loop) {
        s.loop_start = spec_int(g, A, "loop.start");
        s.loop_end = spec_int(g, A, "loop.end");
        s.loop_count = spec_int(g, A, "loop.count");
        s.loop_gate_rank = spec_int(g, A, "loop.gate_rank");
        s.loop_varying = spec_int(g, A, "loop.varying") != 0;
        s.attn_res_block = spec_int(g, A, "attn_res.block_size");
        s.sconv_l_cache = spec_int(g, A, "shortconv.l_cache");
        s.relative_length = spec_int(g, A, "attention.relative_length");
        s.relative_extent = spec_int(g, A, "attention.relative_extent");
        s.relative_clip = spec_num(g, A, "attention.relative_clip");
        s.qk_clip = spec_num(g, A, "attention.qk_clip");
        s.log_scale_floor = spec_int(g, A, "attention.log_scale_floor");
        s.log_scale_alpha = spec_num(g, A, "attention.log_scale_alpha");
        s.situ_beta = spec_num(g, A, "activation.situ_beta");
        s.situ_linear_beta = spec_num(g, A, "activation.situ_linear_beta");
        s.final_logit_softcap = spec_num(g, A, "final_logit_softcapping");
        s.vocab_padded = spec_int(g, A, "vocab_size");
        s.vocab_unpadded = spec_int(g, A, "vocab_size_unpadded");
        s.expert_gate_bias = spec_get(g, A, "expert_gate_bias") != nullptr ||
                             spec_get(g, A, "use_gate_bias") != nullptr;
        // llama_model_inkling2_loop::load_arch_hparams's checks, here before a slot is allocated
        const int64_t block = s.attn_res_block;
        if (s.loop_count < (s.loop_varying ? 1 : 2) || s.loop_gate_rank <= 0 || s.loop_start <= 0 ||
            s.loop_start >= s.loop_end || s.loop_end > s.n_layer || block <= 0 ||
            s.loop_start % block != 0 || s.loop_end % block != 0 || s.n_nextn != 0) {
            err = A + ": invalid loop core [" + std::to_string(s.loop_start) + ", " + std::to_string(s.loop_end) +
                  ") with AttnRes block " + std::to_string(block) + ", " + std::to_string(s.loop_count) +
                  " passes or gate rank " + std::to_string(s.loop_gate_rank);
            return false;
        }
        if (!s.relative_length || !s.relative_extent || !s.swa_window || !s.log_scale_floor ||
            s.sconv_l_cache < 2 || s.sconv_l_cache > 64 || s.vocab_unpadded <= 0 ||
            s.vocab_unpadded > s.vocab_padded || s.qk_clip <= 0 || s.relative_clip <= 0 || s.situ_beta <= 0 ||
            s.situ_linear_beta <= 0) {
            err = A + ": invalid relative attention, short convolution, softcap or vocabulary dimensions";
            return false;
        }
        if (s.n_slot() > 512) {   // llama.cpp's LLAMA_MAX_LAYERS; keep the check local to the header
            err = A + ": " + std::to_string(s.n_slot()) + " cache slots exceed the maximum of 512";
            return false;
        }
    }

    s.full_attention_interval = spec_int(g, A, "full_attention_interval");
    s.ssm_d_conv = spec_int(g, A, "ssm.conv_kernel");
    s.ssm_d_inner = spec_int(g, A, "ssm.inner_size");
    s.ssm_d_state = spec_int(g, A, "ssm.state_size");
    s.ssm_dt_rank = spec_int(g, A, "ssm.time_step_rank");
    s.ssm_n_group = spec_int(g, A, "ssm.group_count");
    s.kda_head_dim = spec_int(g, A, "kda.head_dim");
    s.kda_gate_lower_bound = spec_num(g, A, "kda.gate_lower_bound");

    // which layers are recurrent: llama.cpp's rule per architecture
    s.recurrent.assign((size_t) n_all, false);
    switch (s.arch) {
    case ModelArch::Qwen4Exp:
    case ModelArch::Qwen35Moe:
    case ModelArch::Qwen35:
    case ModelArch::SenseNovaLoop: {
        // qwen35moe.cpp: attention.recurrent_layers when written, else every layer but each interval-th
        std::vector<double> r;
        const MetaValue* rv = spec_get(g, A, "attention.recurrent_layers");
        if (s.arch == ModelArch::SenseNovaLoop && (!rv || rv->type != MetaType::ARRAY || rv->elem != MetaType::BOOL)) {
            err = A + ".attention.recurrent_layers must have one value per layer"; return false;
        }
        if (spec_per_layer(g, A, "attention.recurrent_layers", n_all, r)) {
            if (s.arch == ModelArch::SenseNovaLoop && s.n_nextn && r[(size_t) s.n_layer] != 0) {
                err = A + ": MTP attention cannot be recurrent"; return false;
            }
            for (int64_t i = 0; i < n_all; ++i) s.recurrent[(size_t) i] = i < s.n_layer && r[(size_t) i] != 0;
        } else {
            if (s.arch == ModelArch::SenseNovaLoop) {
                err = A + ".attention.recurrent_layers must have one value per layer";
                return false;
            }
            const int64_t iv = s.full_attention_interval > 0 ? s.full_attention_interval : 4;
            for (int64_t i = 0; i < s.n_layer; ++i) s.recurrent[(size_t) i] = (i + 1) % iv != 0;
        }
        if (s.ssm_d_inner <= 0 || s.ssm_d_state <= 0 || s.ssm_dt_rank <= 0 || s.ssm_n_group <= 0 || s.ssm_d_conv <= 0) {
            err = A + ": the gated delta-net needs ssm.conv_kernel, inner_size, state_size, time_step_rank and group_count";
            return false;
        }
        break;
    }
    case ModelArch::Glm5Next:
        // glm5-next.cpp: a layer without K/V heads is a KDA layer
        for (int64_t i = 0; i < s.n_layer; ++i) s.recurrent[(size_t) i] = s.n_head_kv[(size_t) i] == 0;
        break;
    case ModelArch::Qwen3Moe:
    case ModelArch::Plamo3:
    case ModelArch::Qwen3:
    case ModelArch::Inkling2Loop:
        break;
    }

    if (s.arch == ModelArch::SenseNovaLoop) {
        const char* keys[] = {"loop.start", "loop.end", "loop.count", "mhc.start", "mhc.end", "mhc.streams",
                              "mhc.iterations", "mhc.fixed_identity", "mhc.pre_scale"};
        for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
            const MetaValue* v = spec_get(g, A, keys[i]);
            const MetaType type = i == 7 ? MetaType::BOOL : i == 8 ? MetaType::F32 : MetaType::U32;
            if (!v || v->type != type) { err = "missing or invalid " + A + "." + keys[i]; return false; }
        }
        s.loop_start = spec_int(g, A, "loop.start");
        s.loop_end = spec_int(g, A, "loop.end");
        s.loop_count = spec_int(g, A, "loop.count");
        s.mhc_start = spec_int(g, A, "mhc.start");
        s.mhc_end = spec_int(g, A, "mhc.end");
        s.mhc_streams = spec_int(g, A, "mhc.streams");
        s.mhc_iterations = spec_int(g, A, "mhc.iterations");
        s.mhc_fixed_identity = spec_int(g, A, "mhc.fixed_identity") != 0;
        s.mhc_pre_scale = spec_num(g, A, "mhc.pre_scale");
        if (s.n_nextn > 1 || s.loop_start < 0 || s.loop_start >= s.loop_end || s.loop_end > s.n_layer ||
            s.loop_count < 1 || s.loop_count - 1 > (s.n_layer - s.loop_end) / (s.loop_end - s.loop_start)) {
            err = A + ": invalid loop range or count"; return false;
        }
        const int64_t extra = (s.loop_count - 1) * (s.loop_end - s.loop_start);
        const int64_t n_physical = s.n_layer - extra;
        if (s.mhc_start < 0 || s.mhc_start >= s.mhc_end || s.mhc_end > n_physical || s.mhc_streams != 2 ||
            s.mhc_iterations < 1 || s.mhc_iterations > 20 || s.mhc_fixed_identity ||
            !std::isfinite(s.mhc_pre_scale) || s.mhc_pre_scale <= 0 ||
            !((s.mhc_end <= s.loop_start) || (s.mhc_start >= s.loop_start && s.mhc_end <= s.loop_end) ||
              s.mhc_start >= s.loop_end)) {
            err = A + ": unsupported mHC configuration"; return false;
        }
        for (int64_t i = 0; i < s.n_layer; ++i) {
            if (s.recurrent[(size_t) i] != s.recurrent[(size_t) s.tensor_layer(i)]) {
                err = A + ": loop visits have different attention types"; return false;
            }
        }
    }

    s.n_expert = spec_int(g, A, "expert_count");
    s.n_expert_used = spec_int(g, A, "expert_used_count");
    s.n_ff_exp = spec_int(g, A, "expert_feed_forward_length");
    s.n_ff_shexp = spec_int(g, A, "expert_shared_feed_forward_length");
    s.n_expert_shared = spec_int(g, A, "expert_shared_count", s.n_ff_shexp > 0 ? 1 : 0);
    s.n_dense_lead = spec_int(g, A, "leading_dense_block_count");
    s.n_ff = spec_int(g, A, "feed_forward_length");
    s.gating = (ExpertGating) spec_int(g, A, "expert_gating_func");
    if (s.gating == ExpertGating::None)   // llama.cpp's defaults: GLM-5.3-Flash and inkling2 sigmoid, the Qwens softmax
        s.gating = (s.arch == ModelArch::Glm5Next || s.arch == ModelArch::Inkling2Loop) ? ExpertGating::Sigmoid
                                                                                       : ExpertGating::Softmax;
    if (s.gating != ExpertGating::Softmax && s.gating != ExpertGating::Sigmoid) {
        err = A + ".expert_gating_func " + std::to_string((uint32_t) s.gating) + " is not supported";
        return false;
    }
    s.expert_weights_norm = spec_int(g, A, "expert_weights_norm") != 0;
    s.expert_weights_scale = spec_num(g, A, "expert_weights_scale");
    spec_per_layer(g, A, "swiglu_clamp_exp", n_all, s.swiglu_clamp_exp);
    spec_per_layer(g, A, "swiglu_clamp_shexp", n_all, s.swiglu_clamp_shexp);
    if (dense) {   // llama.cpp's qwen35, plamo3 and qwen3: a SwiGLU of n_ff in every block
        if (s.n_expert != 0 || s.n_ff <= 0) {
            err = A + ": expert_count " + std::to_string(s.n_expert) + ", feed_forward_length " + std::to_string(s.n_ff);
            return false;
        }
    } else if (s.n_expert <= 0 || s.n_expert_used <= 0 || s.n_expert_used > s.n_expert || s.n_ff_exp <= 0) {
        err = A + ": expert_count " + std::to_string(s.n_expert) + ", expert_used_count " +
              std::to_string(s.n_expert_used) + ", expert_feed_forward_length " + std::to_string(s.n_ff_exp);
        return false;
    }
    if (s.n_dense_lead < 0 || s.n_dense_lead >= s.n_layer || (s.n_dense_lead > 0 && s.n_ff <= 0)) {
        err = A + ".leading_dense_block_count " + std::to_string(s.n_dense_lead) + " with feed_forward_length " +
              std::to_string(s.n_ff);
        return false;
    }

    s.hc = spec_int(g, A, "hyper_connection.count");
    s.hc_low_rank = spec_int(g, A, "hyper_connection.low_rank");
    s.hc_sinkhorn_iters = spec_int(g, A, "hyper_connection.sinkhorn_iterations");
    s.hc_eps = spec_num(g, A, "hyper_connection.epsilon");

    s.q_lora_rank = spec_int(g, A, "attention.q_lora_rank");
    s.kv_lora_rank = spec_int(g, A, "attention.kv_lora_rank");
    s.key_length_mla = spec_int(g, A, "attention.key_length_mla");
    s.value_length_mla = spec_int(g, A, "attention.value_length_mla");
    s.idx_n_head = spec_int(g, A, "attention.indexer.head_count");
    s.idx_key_length = spec_int(g, A, "attention.indexer.key_length");
    s.idx_top_k = spec_int(g, A, "attention.indexer.top_k");
    s.idx_kpool = spec_int(g, A, "attention.indexer.kpool");
    // llama.cpp's default where the file has no such key (llama-hparams.h): GLM-5.3-Flash's GGUF writes none
    s.idx_kpool_select_tail = spec_int(g, A, "attention.indexer.kpool_select_tail", 1) != 0;

    if (s.arch == ModelArch::Glm5Next) {
        // glm5-next.cpp asserts these: an mHC of 4 streams and a pool size that divides the indexer's budget
        if (s.hc != 4) { err = A + ".hyper_connection.count " + std::to_string(s.hc) + ", only 4 is supported"; return false; }
        if (s.idx_kpool <= 1 || s.idx_top_k <= 0 || s.idx_top_k % s.idx_kpool != 0) {
            err = A + ": indexer.kpool " + std::to_string(s.idx_kpool) + " must be above 1 and divide indexer.top_k " +
                  std::to_string(s.idx_top_k);
            return false;
        }
        if (s.kda_head_dim <= 0 || s.kv_lora_rank <= 0 || s.q_lora_rank <= 0) {
            err = A + ": needs kda.head_dim, attention.q_lora_rank and attention.kv_lora_rank";
            return false;
        }
    }
    return true;
}

}  // namespace strata
