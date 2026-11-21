// engine/src/inkling2/cuda/ink_manifest.hpp - .strata v1.1 manifest parser for Inkling-2-Mini + DSpark
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "hy4_json.h"  // reuse Strata's lightweight zero-dep MIT JSON parser (or local copy)
#include "ink_cuda.cuh"

namespace strata::ink {

struct TensorInfo {
    std::string name;
    int fmt = FMT_F32;
    std::vector<int64_t> shape;
    uint64_t off = 0;    // offset relative to Section 1 base
    uint64_t bytes = 0;
    int64_t rows() const {
        if (shape.empty()) return 0;
        if (shape.size() == 1) return 1;
        int64_t r = 1;
        for (size_t i = 0; i + 1 < shape.size(); ++i) r *= shape[i];
        return r;
    }
    int64_t cols() const { return shape.empty() ? 0 : shape.back(); }
};

struct InkCfg {
    int n_phys = kInkPhysLayers;
    int n_visits = kInkVisits;
    int hidden = kInkHidden;
    int inter = kInkInter;
    int vocab_padded = kInkVocabPad;
    int vocab_unpadded = kInkVocabUnpad;
    int n_head = kInkHeads;
    int n_kv = kInkKVHeads;
    int head_dim = kInkHeadDim;
    int d_rel = kInkDRel;
    int rel_extent = kInkRelExtent;
    int swa_win = kInkSwaWin;
    int block_size = kInkBlockSize;
    int n_experts = kInkExperts;
    int top_k = kInkTopK;
    int core_start = kInkCoreStart;
    int core_end = kInkCoreEnd;
    int num_passes = kInkCorePasses;
    int gate_rank = kInkGateRank;
    int eos_id = 200006;
    float rms_eps = kInkRmsEps;
    float qk_clip = kInkQkClip;
    float rel_clip = kInkRelClip;
    float log_alpha = kInkLogAlpha;
    int log_floor = kInkLogFloor;
    float route_scale = kInkRouteScale;
    float situ_beta = kInkSituBeta;
    float situ_lb = kInkSituLb;
    float logit_softcap = kInkSoftcap;
    std::vector<int> is_swa;  // [n_phys]
    // DSpark metadata
    int dspark_layers = kInkDsparkLayers;
    int dspark_block = kInkDsparkBlock;
    int dspark_inter = kInkDsparkInter;
    int dspark_markov_rank = kInkMarkovRank;
    int dspark_swa_win = 1025;
    float dspark_rope_theta = 1000000.0f;
    std::vector<int> dspark_is_swa = {1, 0, 1};
    std::vector<int> dspark_taps = {5, 23, 41, 48};
};

inline int ink_fmt_from_str(const std::string& s) {
    if (s == "F32")    return FMT_F32;
    if (s == "BF16")   return FMT_BF16;
    if (s == "F16")    return FMT_F16;
    if (s == "Q8_ROW") return FMT_Q8ROW;
    if (s == "S1.25")  return FMT_S1_25;
    if (s == "S1.5")   return FMT_S1_5;
    if (s == "S2_K")   return FMT_S2_K;
    if (s == "S3_K")   return FMT_S3_K;
    if (s == "S4_K")   return FMT_S4_K;
    if (s == "S5_K")   return FMT_S5_K;
    if (s == "S6_K")   return FMT_S6_K;
    if (s == "S8_K")   return FMT_S8_K;
    if (s == "SF4")    return FMT_SF4;
    if (s == "MX8")    return FMT_MX8;
    std::fprintf(stderr, "strata-inkling: unknown tensor format '%s'\n", s.c_str());
    std::exit(2);
}

struct InkManifest {
    InkCfg c;
    std::string build;
    std::unordered_map<std::string, TensorInfo> trunk;
    std::unordered_map<std::string, TensorInfo> dspark;
    std::string chat_template;
    std::string tokenizer_json;
    bool has_dspark = false;

    const TensorInfo& get(const std::string& name) const {
        auto it = trunk.find(name);
        if (it != trunk.end()) return it->second;
        auto it2 = dspark.find(name);
        if (it2 != dspark.end()) return it2->second;
        std::fprintf(stderr, "strata-inkling: missing tensor '%s'\n", name.c_str());
        std::exit(2);
    }
    const TensorInfo* find(const std::string& name) const {
        auto it = trunk.find(name);
        if (it != trunk.end()) return &it->second;
        auto it2 = dspark.find(name);
        if (it2 != dspark.end()) return &it2->second;
        return nullptr;
    }
};

inline void parse_ink_manifest(const std::string& json_text, InkManifest& m, bool keep_tokenizer = false) {
    hy4::JVal man = hy4::json_parse(json_text);
    if (man.get("build")) m.build = man.at("build").str();
    const hy4::JVal& cfg = man.at("config");
    const hy4::JVal& tc = cfg.get("text_config") ? cfg.at("text_config") : cfg;
    InkCfg& c = m.c;
    if (cfg.get("eos_token_id")) c.eos_id = (int)cfg.at("eos_token_id").i();
    if (tc.get("num_hidden_layers")) c.n_phys = (int)tc.at("num_hidden_layers").i();
    if (tc.get("hidden_size")) c.hidden = (int)tc.at("hidden_size").i();
    if (tc.get("intermediate_size")) c.inter = (int)tc.at("intermediate_size").i();
    if (tc.get("vocab_size")) c.vocab_padded = (int)tc.at("vocab_size").i();
    if (tc.get("unpadded_vocab_size")) c.vocab_unpadded = (int)tc.at("unpadded_vocab_size").i();
    if (tc.get("num_attention_heads")) c.n_head = (int)tc.at("num_attention_heads").i();
    if (tc.get("num_key_value_heads")) c.n_kv = (int)tc.at("num_key_value_heads").i();
    if (tc.get("head_dim")) c.head_dim = (int)tc.at("head_dim").i();
    if (tc.get("d_rel")) c.d_rel = (int)tc.at("d_rel").i();
    if (tc.get("rel_extent")) c.rel_extent = (int)tc.at("rel_extent").i();
    if (tc.get("sliding_window_size")) c.swa_win = (int)tc.at("sliding_window_size").i();
    if (tc.get("n_layers_per_attn_res_block")) c.block_size = (int)tc.at("n_layers_per_attn_res_block").i();
    if (tc.get("n_routed_experts")) c.n_experts = (int)tc.at("n_routed_experts").i();
    if (tc.get("num_experts_per_tok")) c.top_k = (int)tc.at("num_experts_per_tok").i();
    if (tc.get("rms_norm_eps")) c.rms_eps = (float)tc.at("rms_norm_eps").f();
    if (tc.get("qk_l2_clip_threshold")) c.qk_clip = (float)tc.at("qk_l2_clip_threshold").f();
    if (tc.get("rel_l2_clip_threshold")) c.rel_clip = (float)tc.at("rel_l2_clip_threshold").f();
    if (tc.get("log_scaling_alpha")) c.log_alpha = (float)tc.at("log_scaling_alpha").f();
    if (tc.get("log_scaling_n_floor")) c.log_floor = (int)tc.at("log_scaling_n_floor").i();
    if (tc.get("route_scale")) c.route_scale = (float)tc.at("route_scale").f();
    if (tc.get("mlp_gate_softcap")) c.situ_beta = (float)tc.at("mlp_gate_softcap").f();
    if (tc.get("mlp_up_softcap")) c.situ_lb = (float)tc.at("mlp_up_softcap").f();
    if (tc.get("final_logit_softcapping")) c.logit_softcap = (float)tc.at("final_logit_softcapping").f();

    if (tc.get("loop_config")) {
        const hy4::JVal& lc = tc.at("loop_config");
        c.core_start = (int)lc.at("core_start").i();
        c.core_end = (int)lc.at("core_end").i();
        c.num_passes = (int)lc.at("num_passes").i();
        c.gate_rank = (int)lc.at("gate_rank").i();
    }
    c.n_visits = c.n_phys + (c.num_passes - 1) * (c.core_end - c.core_start);

    c.is_swa.assign(c.n_phys, 0);
    if (tc.get("local_layer_ids")) {
        for (const auto& v : tc.at("local_layer_ids").a) {
            int id = (int)v.i();
            if (id >= 0 && id < c.n_phys) c.is_swa[id] = 1;
        }
    } else {
        for (int i = 0; i < c.n_phys; ++i) c.is_swa[i] = ink_is_global_phys(i) ? 0 : 1;
    }

    if (cfg.get("mtp_config")) {
        const hy4::JVal& mc = cfg.at("mtp_config");
        if (mc.get("num_hidden_layers")) c.dspark_layers = (int)mc.at("num_hidden_layers").i();
        if (mc.get("block_size")) c.dspark_block = (int)mc.at("block_size").i();
        if (mc.get("intermediate_size")) c.dspark_inter = (int)mc.at("intermediate_size").i();
        if (mc.get("markov_rank")) c.dspark_markov_rank = (int)mc.at("markov_rank").i();
        if (mc.get("sliding_window_size")) c.dspark_swa_win = (int)mc.at("sliding_window_size").i() + 1;
        if (mc.get("rope_theta")) c.dspark_rope_theta = (float)mc.at("rope_theta").f();
        if (mc.get("local_layer_ids")) {
            c.dspark_is_swa.assign(c.dspark_layers, 0);
            for (const auto& v : mc.at("local_layer_ids").a) {
                int id = (int)v.i();
                if (id >= 0 && id < c.dspark_layers) c.dspark_is_swa[id] = 1;
            }
        }
        if (mc.get("target_layer_ids")) {
            c.dspark_taps.clear();
            for (const auto& v : mc.at("target_layer_ids").a) c.dspark_taps.push_back((int)v.i());
        }
    }

    auto read_tlist = [](const hy4::JVal& arr, std::unordered_map<std::string, TensorInfo>& dst) {
        for (const auto& t : arr.a) {
            TensorInfo ti;
            ti.name = t.at("name").str();
            ti.fmt = ink_fmt_from_str(t.at("format").str());
            for (const auto& d : t.at("shape").a) ti.shape.push_back(d.i());
            ti.off = (uint64_t)t.at("offset").i();
            ti.bytes = (uint64_t)t.at("bytes").i();
            dst[ti.name] = std::move(ti);
        }
    };
    if (man.get("trunk_tensors")) read_tlist(man.at("trunk_tensors"), m.trunk);
    if (man.get("dspark_tensors")) {
        read_tlist(man.at("dspark_tensors"), m.dspark);
        m.has_dspark = !m.dspark.empty();
    }
    if (keep_tokenizer) {
        if (man.get("chat_template")) m.chat_template = man.at("chat_template").str();
        if (man.get("tokenizer_json")) m.tokenizer_json = man.at("tokenizer_json").str();
    }
}

}  // namespace strata::ink
