// hy4_manifest.h - config / tensor table / engram metadata of a Hy4-Flash .strata, parsed from the Section 0 manifest (shared by the CPU oracle and the GPU runner)
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "hy4_json.h"
#include "strata/artifact/strata_reader.hpp"

namespace hy4 {

enum WFmt { F_S2 = 2, F_S3 = 3, F_S4 = 4, F_S5 = 5, F_S6 = 6, F_S8 = 8, F_Q8ROW = 100, F_F32 = 101 };

struct TInfo {
    std::string name;
    int fmt = 0;
    std::vector<int64_t> shape;
    uint64_t off = 0, bytes = 0;  // off relative to Section 1
    int64_t rows() const { return shape.size() > 1 ? shape[0] : 1; }
    int64_t cols() const { return shape.back(); }
};

struct Cfg {
    int n_layers = 42, hidden = 3072, hc = 4, vocab = 120832;
    float rms_eps = 1e-5f, hc_eps = 1e-6f, hc_mag = 2.0f;
    int q_lora = 1024, kv_lora = 512, qk_nope = 192, qk_rope = 64, qk_head = 256, v_head = 256;
    int window = 512, kpool = 4, index_topk = 2048, n_idx_head = 16, idx_dim = 128;
    float swiglu_limit = 10.0f, routed_scale = 3.998f;
    int n_experts = 512, top_k = 16;
    int heads_sparse = 32, heads_swa = 64;
    double theta_sparse = 1e7, theta_swa = 1e4;
    std::vector<int> is_swa;
    int bos = 120000, eos = 120025;
};

struct EngramCfg {
    int n_mod = 4, n_cols = 16, n_head = 8, n_gram = 3;
    std::vector<int64_t> token_map;
    int pad_id = 120002;
    std::vector<uint64_t> mult;    // [mod][3]
    std::vector<uint64_t> primes;  // [mod][16]
    std::vector<int64_t> offs;     // [mod][16]
    std::vector<int> layers;       // 0-based layer ids
};

inline int fmt_from(const std::string& s) {
    if (s == "F32") return F_F32;
    if (s == "Q8_ROW") return F_Q8ROW;
    if (s == "S2_K") return F_S2;
    if (s == "S3_K") return F_S3;
    if (s == "S4_K") return F_S4;
    if (s == "S5_K") return F_S5;
    if (s == "S6_K") return F_S6;
    if (s == "S8_K") return F_S8;
    fprintf(stderr, "unknown fmt %s\n", s.c_str());
    exit(2);
}

struct Manifest {
    Cfg c;
    EngramCfg e;
    std::unordered_map<std::string, TInfo> T;
    std::string chat_template;
    std::string tokenizer_json;
    const TInfo& get(const std::string& n) const {
        auto it = T.find(n);
        if (it == T.end()) {
            fprintf(stderr, "missing tensor %s\n", n.c_str());
            exit(2);
        }
        return it->second;
    }
};

inline void parse_manifest(const std::string& json_text, Manifest& m, bool keep_tokenizer = false) {
    JVal man = json_parse(json_text);
    const JVal& tc = man.at("config").at("text_config");
    Cfg& c = m.c;
    c.n_layers = (int)tc.at("num_hidden_layers").i();
    c.hidden = (int)tc.at("hidden_size").i();
    c.hc = (int)tc.at("hc_mult").i();
    c.vocab = (int)tc.at("vocab_size").i();
    c.rms_eps = (float)tc.at("rms_norm_eps").f();
    c.hc_eps = (float)tc.at("hc_eps").f();
    c.hc_mag = (float)tc.at("hc_magnitude").f();
    c.q_lora = (int)tc.at("q_lora_rank").i();
    c.kv_lora = (int)tc.at("kv_lora_rank").i();
    c.qk_nope = (int)tc.at("qk_nope_head_dim").i();
    c.qk_rope = (int)tc.at("qk_rope_head_dim").i();
    c.qk_head = (int)tc.at("qk_head_dim").i();
    c.v_head = (int)tc.at("v_head_dim").i();
    c.window = (int)tc.at("sliding_window").i();
    c.kpool = (int)tc.at("compressed_indexer_compress_ratio").i();
    c.index_topk = (int)tc.at("index_topk").i();
    c.n_idx_head = (int)tc.at("compressed_indexer_num_attention_heads").i();
    c.idx_dim = (int)tc.at("compressed_indexer_head_dim").i();
    c.swiglu_limit = (float)tc.at("swiglu_limit").f();
    c.routed_scale = (float)tc.at("routed_scaling_factor").f();
    c.n_experts = (int)tc.at("n_routed_experts").i();
    c.top_k = (int)tc.at("num_experts_per_tok").i();
    c.heads_sparse = (int)tc.at("num_attention_heads").i();
    c.heads_swa = (int)tc.at("swa_num_attention_heads").i();
    c.bos = (int)tc.at("bos_token_id").i();
    c.eos = (int)tc.at("eos_token_id").i();
    c.theta_sparse = tc.at("rope_parameters").at("deepseek_sparse_attention").at("rope_theta").f();
    c.theta_swa = tc.at("rope_parameters").at("sliding_attention").at("rope_theta").f();
    const JVal& lt = tc.at("layer_types");
    c.is_swa.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; ++i) c.is_swa[i] = lt[i].str() == "sliding_attention";

    const JVal& em = man.at("engram_meta");
    EngramCfg& e = m.e;
    e.n_gram = (int)em.at("ple_ngram_size").i();
    e.n_head = (int)em.at("ple_heads_per_ngram").i();
    e.n_cols = (e.n_gram - 1) * e.n_head;
    e.pad_id = (int)em.at("engram_pad_id").i();
    for (auto& v : em.at("ple_layers").a) e.layers.push_back((int)v.i());
    e.n_mod = (int)e.layers.size();
    for (auto& v : em.at("engram_token_map").a) e.token_map.push_back(v.i());
    for (auto& v : em.at("ple_layer_multipliers").a) e.mult.push_back((uint64_t)v.i());
    for (auto& v : em.at("ple_head_vocab_sizes").a) e.primes.push_back((uint64_t)v.i());
    for (auto& v : em.at("ple_head_offsets").a) e.offs.push_back(v.i());

    for (auto& t : man.at("trunk_tensors").a) {
        TInfo ti;
        ti.name = t.at("name").str();
        ti.fmt = fmt_from(t.at("format").str());
        for (auto& d : t.at("shape").a) ti.shape.push_back(d.i());
        ti.off = (uint64_t)t.at("offset").i();
        ti.bytes = (uint64_t)t.at("bytes").i();
        m.T[ti.name] = std::move(ti);
    }
    if (keep_tokenizer) {
        m.chat_template = man.at("chat_template").str();
        m.tokenizer_json = man.at("tokenizer_json").str();
    }
}

}  // namespace hy4
