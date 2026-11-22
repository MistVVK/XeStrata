// engine/src/inkling2/ink_manifest.h - .strata v1.1 manifest, config, tensor table & SHA-256 verification for Inkling-2-Mini
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "ink_gguf_loader.h"
#include "ink_json.h"
#include "ink_safetensors_loader.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/quant/sk_codecs.hpp"

namespace ink2 {

enum WFmt : int {
    F_S1_25 = 0,
    F_S1_5  = 1,
    F_S2    = 2,
    F_S3    = 3,
    F_S4    = 4,
    F_S5    = 5,
    F_S6    = 6,
    F_S8    = 8,
    F_SF4      = 15,
    F_MX8      = 16,
    F_GGUF_RAW = 20,
    F_Q8ROW    = 100,
    F_F32      = 101,
    F_BF16     = 102,
    F_F16      = 103,
};

inline int fmt_from_str(const std::string& s) {
    if (s == "F32" || s == "float32") return F_F32;
    if (s == "BF16" || s == "bfloat16") return F_BF16;
    if (s == "F16" || s == "float16") return F_F16;
    if (s == "Q8_ROW") return F_Q8ROW;
    if (s == "S1_25") return F_S1_25;
    if (s == "S1_5") return F_S1_5;
    if (s == "S2_K") return F_S2;
    if (s == "S3_K") return F_S3;
    if (s == "S4_K") return F_S4;
    if (s == "S5_K") return F_S5;
    if (s == "S6_K") return F_S6;
    if (s == "S8_K") return F_S8;
    if (s == "SF4") return F_SF4;
    if (s == "MX8") return F_MX8;
    return -1;
}

inline const char* fmt_name(int fmt) {
    switch (fmt) {
        case F_F32: return "F32";
        case F_BF16: return "BF16";
        case F_F16: return "F16";
        case F_Q8ROW: return "Q8_ROW";
        case F_S1_25: return "S1_25";
        case F_S1_5: return "S1_5";
        case F_S2: return "S2_K";
        case F_S3: return "S3_K";
        case F_S4: return "S4_K";
        case F_S5: return "S5_K";
        case F_S6: return "S6_K";
        case F_S8: return "S8_K";
        case F_SF4: return "SF4";
        case F_MX8: return "MX8";
        default: return "UNKNOWN";
    }
}

struct TInfo {
    std::string name;
    int fmt = F_F32;
    std::vector<int64_t> shape;
    uint64_t off = 0;    // relative to Section 1
    uint64_t bytes = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
    int64_t rows() const { return shape.size() > 1 ? shape[0] : 1; }
    int64_t cols() const { return shape.empty() ? 0 : shape.back(); }
};

struct InkCfg {
    int n_phys = 24;
    int core_start = 6;
    int core_end = 18;
    int num_passes = 3;
    int gate_rank = 64;
    int block_size = 6;         // Attention-Residual block size
    int hidden = 2304;
    int inter = 1152;
    int n_head = 24;
    int n_kv = 8;
    int head_dim = 128;
    int d_rel = 16;
    int rel_extent = 1024;
    int swa_win = 1024;
    int sconv_k = 4;
    int n_experts = 128;
    int top_k = 8;
    int vocab_padded = 201024;
    int vocab_unpadded = 200058;
    int eos_token_id = 200006;  // <|content_model_end_sampling|>
    int eot_token_id = 199999;  // <|endoftext|>
    int image_token_id = 200054;
    int max_position_embeddings = 1048576;

    float rms_eps = 1e-6f;
    float qk_l2_clip = 22.627416997969522f;
    float rel_l2_clip = 8.0f;
    int   log_scale_floor = 8192;
    float log_scale_alpha = 0.1f;
    float situ_beta = 4.0f;
    float situ_lb = 25.0f;
    float route_scale = 8.0f;
    float logit_softcap = 30.0f;

    std::vector<uint8_t> is_swa;      // [n_phys], 1=SWA-1024, 0=Global
    std::vector<int> visit_to_phys;   // [n_visits]
    std::vector<int> visit_to_pass;   // [n_visits]
    std::vector<uint8_t> visit_is_swa;// [n_visits]
    std::vector<int> visit_kv_slot;   // [n_visits] -> index in SWA pool (0..39) or Global pool (0..7)
    int n_visits = 48;
    int n_swa_visits = 40;
    int n_global_visits = 8;

    void build_visit_schedule() {
        const int core_len = core_end - core_start;
        n_visits = core_start + num_passes * core_len + (n_phys - core_end);
        visit_to_phys.resize(n_visits);
        visit_to_pass.resize(n_visits);
        visit_is_swa.resize(n_visits);
        visit_kv_slot.resize(n_visits);
        int swa_idx = 0, glob_idx = 0;
        for (int v = 0; v < n_visits; ++v) {
            int ip = 0, p = 0;
            if (v < core_start) {
                ip = v; p = 0;
            } else if (v < core_start + num_passes * core_len) {
                ip = core_start + ((v - core_start) % core_len);
                p = (v - core_start) / core_len;
            } else {
                ip = core_end + (v - core_start - num_passes * core_len);
                p = 0;
            }
            visit_to_phys[v] = ip;
            visit_to_pass[v] = p;
            const uint8_t swa = (ip < static_cast<int>(is_swa.size())) ? is_swa[ip] : 1;
            visit_is_swa[v] = swa;
            visit_kv_slot[v] = swa ? swa_idx++ : glob_idx++;
        }
        n_swa_visits = swa_idx;
        n_global_visits = glob_idx;
    }
};

struct DsparkCfg {
    bool present = false;
    int n_layers = 3;
    int hidden = 2304;
    int n_head = 24;
    int n_kv = 8;
    int head_dim = 128;
    int inter = 9216;
    int block_size = 8;
    int markov_rank = 256;
    int swa_win = 1025;
    float rms_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    std::vector<uint8_t> is_swa = {1, 0, 1};
    std::vector<int> target_visits = {5, 23, 41, 48};
};

struct InkManifest {
    std::string format;
    std::string build;
    std::string arch;
    InkCfg c;
    DsparkCfg dspark;
    std::unordered_map<std::string, TInfo> trunk;
    std::unordered_map<std::string, TInfo> dspark_tensors;
    std::string chat_template;
    std::string tokenizer_json;  // optional in v1.1 (may be empty if external tokenizer.json)

    const TInfo* find_trunk(const std::string& n) const {
        auto it = trunk.find(n);
        return it == trunk.end() ? nullptr : &it->second;
    }
    const TInfo& get_trunk(const std::string& n) const {
        const TInfo* p = find_trunk(n);
        if (!p) {
            fprintf(stderr, "[inkling2] FATAL: missing trunk tensor %s\n", n.c_str());
            std::exit(2);
        }
        return *p;
    }
    const TInfo* find_dspark(const std::string& n) const {
        auto it = dspark_tensors.find(n);
        if (it != dspark_tensors.end()) return &it->second;
        auto it2 = dspark_tensors.find("dspark." + n);
        return it2 == dspark_tensors.end() ? nullptr : &it2->second;
    }
};

inline bool parse_manifest(const std::string& json_text, InkManifest& m, std::string& err, bool keep_tokenizer = false) {
    try {
        JVal man = json_parse(json_text);
        if (const JVal* f = man.get("format")) m.format = f->str();
        if (const JVal* b = man.get("build")) m.build = b->str();
        if (const JVal* a = man.get("arch")) m.arch = a->str();

        const JVal& cfg = man.at("config");
        const JVal& tc = cfg.at("text_config");
        InkCfg& c = m.c;
        c.n_phys = static_cast<int>(tc.at("num_hidden_layers").i());
        c.hidden = static_cast<int>(tc.at("hidden_size").i());
        c.inter = static_cast<int>(tc.at("intermediate_size").i());
        c.n_head = static_cast<int>(tc.at("num_attention_heads").i());
        c.n_kv = static_cast<int>(tc.at("num_key_value_heads").i());
        c.head_dim = static_cast<int>(tc.at("head_dim").i());
        if (const JVal* v = tc.get("d_rel")) c.d_rel = static_cast<int>(v->i());
        if (const JVal* v = tc.get("rel_extent")) c.rel_extent = static_cast<int>(v->i());
        if (const JVal* v = tc.get("sliding_window_size")) c.swa_win = static_cast<int>(v->i());
        if (const JVal* v = tc.get("sconv_kernel_size")) c.sconv_k = static_cast<int>(v->i());
        if (const JVal* v = tc.get("n_layers_per_attn_res_block")) c.block_size = static_cast<int>(v->i());
        if (const JVal* v = tc.get("n_routed_experts")) c.n_experts = static_cast<int>(v->i());
        if (const JVal* v = tc.get("num_experts_per_tok")) c.top_k = static_cast<int>(v->i());
        if (const JVal* v = tc.get("vocab_size")) c.vocab_padded = static_cast<int>(v->i());
        if (const JVal* v = tc.get("unpadded_vocab_size")) c.vocab_unpadded = static_cast<int>(v->i());
        if (const JVal* v = cfg.get("eos_token_id")) c.eos_token_id = static_cast<int>(v->i());
        if (const JVal* v = tc.get("eos_token_id")) c.eos_token_id = static_cast<int>(v->i());
        if (const JVal* v = cfg.get("image_token_id")) c.image_token_id = static_cast<int>(v->i());
        if (const JVal* v = tc.get("model_max_length")) c.max_position_embeddings = static_cast<int>(v->i());
        if (const JVal* v = tc.get("rms_norm_eps")) c.rms_eps = static_cast<float>(v->f());
        if (const JVal* v = tc.get("qk_l2_clip_threshold")) c.qk_l2_clip = static_cast<float>(v->f());
        if (const JVal* v = tc.get("rel_l2_clip_threshold")) c.rel_l2_clip = static_cast<float>(v->f());
        if (const JVal* v = tc.get("log_scaling_n_floor")) c.log_scale_floor = static_cast<int>(v->i());
        if (const JVal* v = tc.get("log_scaling_alpha")) c.log_scale_alpha = static_cast<float>(v->f());
        if (const JVal* v = tc.get("mlp_gate_softcap")) c.situ_beta = static_cast<float>(v->f());
        if (const JVal* v = tc.get("mlp_up_softcap")) c.situ_lb = static_cast<float>(v->f());
        if (const JVal* v = tc.get("route_scale")) c.route_scale = static_cast<float>(v->f());
        if (const JVal* v = tc.get("final_logit_softcapping")) c.logit_softcap = static_cast<float>(v->f());

        if (const JVal* lc = tc.get("loop_config")) {
            c.core_start = static_cast<int>(lc->at("core_start").i());
            c.core_end = static_cast<int>(lc->at("core_end").i());
            c.num_passes = static_cast<int>(lc->at("num_passes").i());
            c.gate_rank = static_cast<int>(lc->at("gate_rank").i());
        }
        c.is_swa.assign(c.n_phys, 0);
        if (const JVal* loc = tc.get("local_layer_ids")) {
            for (const auto& x : loc->a) {
                int id = static_cast<int>(x.i());
                if (id >= 0 && id < c.n_phys) c.is_swa[id] = 1;
            }
        }
        c.build_visit_schedule();

        if (const JVal* mtp = cfg.get("mtp_config")) {
            DsparkCfg& d = m.dspark;
            if (const JVal* v = mtp->get("num_hidden_layers")) d.n_layers = static_cast<int>(v->i());
            if (const JVal* v = mtp->get("hidden_size")) d.hidden = static_cast<int>(v->i());
            if (const JVal* v = mtp->get("num_attention_heads")) d.n_head = static_cast<int>(v->i());
            if (const JVal* v = mtp->get("num_key_value_heads")) d.n_kv = static_cast<int>(v->i());
            if (const JVal* v = mtp->get("head_dim")) d.head_dim = static_cast<int>(v->i());
            if (const JVal* v = mtp->get("intermediate_size")) d.inter = static_cast<int>(v->i());
            if (const JVal* v = mtp->get("block_size")) d.block_size = static_cast<int>(v->i());
            if (const JVal* v = mtp->get("markov_rank")) d.markov_rank = static_cast<int>(v->i());
            if (const JVal* v = mtp->get("sliding_window_size")) d.swa_win = static_cast<int>(v->i()) + 1;
            if (const JVal* v = mtp->get("rms_norm_eps")) d.rms_eps = static_cast<float>(v->f());
            if (const JVal* v = mtp->get("rope_theta")) d.rope_theta = static_cast<float>(v->f());
            if (const JVal* loc = mtp->get("local_layer_ids")) {
                d.is_swa.assign(d.n_layers, 0);
                for (const auto& x : loc->a) {
                    int id = static_cast<int>(x.i());
                    if (id >= 0 && id < d.n_layers) d.is_swa[id] = 1;
                }
            }
            if (const JVal* tv = mtp->get("target_layer_ids")) {
                d.target_visits.clear();
                for (const auto& x : tv->a) d.target_visits.push_back(static_cast<int>(x.i()));
            }
        }

        if (const JVal* tt = man.get("trunk_tensors")) {
            for (const auto& t : tt->a) {
                TInfo ti;
                ti.name = t.at("name").str();
                ti.fmt = fmt_from_str(t.at("format").str());
                if (ti.fmt < 0) {
                    err = "unknown trunk tensor format: " + t.at("format").str();
                    return false;
                }
                for (const auto& d : t.at("shape").a) ti.shape.push_back(d.i());
                ti.off = static_cast<uint64_t>(t.at("offset").i());
                ti.bytes = static_cast<uint64_t>(t.at("bytes").i());
                m.trunk[ti.name] = std::move(ti);
            }
        }
        if (const JVal* dt = man.get("dspark_tensors")) {
            for (const auto& t : dt->a) {
                TInfo ti;
                ti.name = t.at("name").str();
                ti.fmt = fmt_from_str(t.at("format").str());
                if (ti.fmt < 0) {
                    err = "unknown dspark tensor format: " + t.at("format").str();
                    return false;
                }
                for (const auto& d : t.at("shape").a) ti.shape.push_back(d.i());
                ti.off = static_cast<uint64_t>(t.at("offset").i());
                ti.bytes = static_cast<uint64_t>(t.at("bytes").i());
                m.dspark_tensors[ti.name] = std::move(ti);
            }
            m.dspark.present = !m.dspark_tensors.empty();
        }
        if (keep_tokenizer) {
            if (const JVal* ct = man.get("chat_template"); ct && ct->t == JVal::STR) m.chat_template = ct->str();
            if (const JVal* tj = man.get("tokenizer_json"); tj && tj->t == JVal::STR) m.tokenizer_json = tj->str();
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("manifest parse error: ") + e.what();
        return false;
    }
}

// Standalone SHA-256 for Section 0..5 integrity checks (--selfcheck)
struct Sha256 {
    uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    uint64_t total_bits = 0;
    uint8_t buf[64]{};
    size_t buf_len = 0;

    static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

    void compress(const uint8_t b[64]) {
        static constexpr uint32_t K[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(b[4*i]) << 24) | (uint32_t(b[4*i+1]) << 16) | (uint32_t(b[4*i+2]) << 8) | uint32_t(b[4*i+3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0], bb=h[1], c=h[2], d=h[3], e=h[4], f=h[5], g=h[6], hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & bb) ^ (a & c) ^ (bb & c);
            uint32_t t2 = S0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=bb; bb=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=bb; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }

    void update(const uint8_t* data, size_t len) {
        total_bits += uint64_t(len) * 8u;
        size_t i = 0;
        if (buf_len > 0) {
            size_t take = std::min(len, 64 - buf_len);
            std::memcpy(buf + buf_len, data, take);
            buf_len += take;
            i += take;
            if (buf_len == 64) { compress(buf); buf_len = 0; }
        }
        while (i + 64 <= len) {
            compress(data + i);
            i += 64;
        }
        if (i < len) {
            buf_len = len - i;
            std::memcpy(buf, data + i, buf_len);
        }
    }

    std::array<uint8_t, 32> finish() {
        buf[buf_len++] = 0x80;
        if (buf_len > 56) {
            std::memset(buf + buf_len, 0, 64 - buf_len);
            compress(buf);
            buf_len = 0;
        }
        std::memset(buf + buf_len, 0, 56 - buf_len);
        for (int i = 0; i < 8; ++i) buf[56 + i] = static_cast<uint8_t>((total_bits >> (56 - 8 * i)) & 0xFFu);
        compress(buf);
        std::array<uint8_t, 32> out{};
        for (int i = 0; i < 8; ++i) {
            out[4*i+0] = static_cast<uint8_t>((h[i] >> 24) & 0xFFu);
            out[4*i+1] = static_cast<uint8_t>((h[i] >> 16) & 0xFFu);
            out[4*i+2] = static_cast<uint8_t>((h[i] >> 8) & 0xFFu);
            out[4*i+3] = static_cast<uint8_t>(h[i] & 0xFFu);
        }
        return out;
    }
};

enum class ModelSourceKind : uint8_t {
    Strata      = 0,  // Flagship native .strata v1.1 (default recommended)
    Safetensors = 1,  // Official HF thinkingmachines/Inkling-2-Mini (+ mtp.safetensors)
    Gguf        = 2,  // Community / Unsloth .gguf (single or multi-shard + optional dspark)
};

inline const char* source_kind_name(ModelSourceKind k) noexcept {
    switch (k) {
        case ModelSourceKind::Strata:      return "strata";
        case ModelSourceKind::Safetensors: return "safetensors";
        case ModelSourceKind::Gguf:        return "gguf";
        default:                           return "unknown";
    }
}

// Unified model handle (.strata v1.1 + Official HF safetensors + Community/Unsloth .gguf)
// shared transparently across CPU / CUDA / HIP / Metal / SYCL backends.
struct InkModelFile {
    std::string path;
    ModelSourceKind kind = ModelSourceKind::Strata;
    strata::artifact::StrataFileReader rd;
    std::unique_ptr<SafetensorsAdapter> st_adapter;
    std::unique_ptr<GgufAdapter> gguf_adapter;
    InkManifest m;
    // Direct lookup: dir_index_of[phys_layer * n_experts + expert_id] -> index in expert_directory()
    std::vector<int> dir_index_of;
    // 3x loop-weighted priority order of directory indices (0..3071)
    std::vector<uint16_t> vram_load_order;

    bool open(const std::string& file_path, std::string& err, bool keep_tokenizer = false,
              const std::string& explicit_dspark = "") {
        path = file_path;
        st_adapter.reset();
        gguf_adapter.reset();
        dir_index_of.clear();
        vram_load_order.clear();

        if (SafetensorsAdapter::is_safetensors_path(file_path)) {
            kind = ModelSourceKind::Safetensors;
            st_adapter = std::make_unique<SafetensorsAdapter>();
            return st_adapter->open_into<InkManifest, TInfo>(
                file_path, m, dir_index_of, vram_load_order, err, keep_tokenizer, explicit_dspark);
        }
        if (GgufAdapter::is_gguf_path(file_path)) {
            kind = ModelSourceKind::Gguf;
            gguf_adapter = std::make_unique<GgufAdapter>();
            return gguf_adapter->open_into<InkManifest, TInfo>(
                file_path, m, dir_index_of, vram_load_order, err, keep_tokenizer, explicit_dspark);
        }

        kind = ModelSourceKind::Strata;
        if (!rd.open(file_path, err)) return false;
        if (!parse_manifest(rd.manifest_json(), m, err, keep_tokenizer)) return false;
        const auto& dir = rd.expert_directory();
        const int n_total = m.c.n_phys * m.c.n_experts;
        dir_index_of.assign(n_total, -1);
        if (static_cast<int>(dir.size()) == n_total) {
            for (int il = 0; il < m.c.n_phys; ++il) {
                for (int r = 0; r < m.c.n_experts; ++r) {
                    int idx = il * m.c.n_experts + r;
                    int eid = dir[idx].expert_id;
                    if (eid >= 0 && eid < m.c.n_experts) {
                        dir_index_of[il * m.c.n_experts + eid] = idx;
                    }
                }
            }
        }
        if (!rd.vram_load_order().empty()) {
            vram_load_order = rd.vram_load_order();
        } else {
            // Synthesize 3x loop-weighted order if Section 3 lacked vram_load_order: core L6..17 first, then L0..5 & L18..23
            vram_load_order.clear();
            vram_load_order.reserve(dir.size());
            for (int il = m.c.core_start; il < m.c.core_end; ++il)
                for (int r = 0; r < m.c.n_experts; ++r)
                    if (il * m.c.n_experts + r < static_cast<int>(dir.size()))
                        vram_load_order.push_back(static_cast<uint16_t>(il * m.c.n_experts + r));
            for (int il = 0; il < m.c.core_start; ++il)
                for (int r = 0; r < m.c.n_experts; ++r)
                    if (il * m.c.n_experts + r < static_cast<int>(dir.size()))
                        vram_load_order.push_back(static_cast<uint16_t>(il * m.c.n_experts + r));
            for (int il = m.c.core_end; il < m.c.n_phys; ++il)
                for (int r = 0; r < m.c.n_experts; ++r)
                    if (il * m.c.n_experts + r < static_cast<int>(dir.size()))
                        vram_load_order.push_back(static_cast<uint16_t>(il * m.c.n_experts + r));
        }
        return true;
    }

    const std::vector<strata::artifact::StrataExpertEntry>& expert_directory() const {
        if (kind == ModelSourceKind::Safetensors && st_adapter) return st_adapter->expert_dir;
        if (kind == ModelSourceKind::Gguf && gguf_adapter) return gguf_adapter->expert_dir;
        return rd.expert_directory();
    }

    bool read_section1(std::vector<uint8_t>& out, std::string& err) const {
        if (kind == ModelSourceKind::Safetensors && st_adapter) return st_adapter->read_section1(out, err);
        if (kind == ModelSourceKind::Gguf && gguf_adapter) return gguf_adapter->read_section1(out, err);
        return rd.read_section(1, out, err);
    }

    bool read_expert_body(size_t dir_index, std::vector<uint8_t>& out, std::string& err) const {
        if (kind == ModelSourceKind::Safetensors && st_adapter) return st_adapter->read_expert_body(dir_index, m.c.n_experts, out, err);
        if (kind == ModelSourceKind::Gguf && gguf_adapter) return gguf_adapter->read_expert_body(dir_index, m.c.n_experts, out, err);
        return rd.read_expert_body(dir_index, out, err);
    }

    bool verify_sections(std::string& err, bool verify_section4 = true) const {
        if (kind != ModelSourceKind::Strata) return true;  // safetensors/GGUF verified via header/bounds at open
        std::vector<uint8_t> chunk(8 * 1024 * 1024);
        for (int sid = 0; sid < strata::artifact::STRATA_NUM_SECTIONS; ++sid) {
            if (sid == 4 && !verify_section4) continue;
            const auto& sec = rd.header().sections[sid];
            if (sec.byte_length == 0) continue;
            Sha256 h;
            uint64_t rem = sec.byte_length;
            uint64_t off = sec.file_offset;
            while (rem > 0) {
                size_t n = static_cast<size_t>(std::min<uint64_t>(rem, chunk.size()));
                if (!rd.read_at(off, chunk.data(), n, err)) return false;
                h.update(chunk.data(), n);
                off += n;
                rem -= n;
            }
            auto digest = h.finish();
            if (std::memcmp(digest.data(), sec.sha256, 32) != 0) {
                err = "SHA-256 mismatch in Section " + std::to_string(sid);
                return false;
            }
        }
        return true;
    }
};

}  // namespace ink2
