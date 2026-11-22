// engine/src/inkling2/ink_safetensors_loader.h - Direct Official HF safetensors loader for thinkingmachines/Inkling-2-Mini + mtp.safetensors
// Zero-requantization adapter into InkModelFile (NVFP4 -> SF4 code_id=15, MXFP8 -> MX8 code_id=16, BF16 trunk + DSpark).
// MIT License - Copyright (c) 2026 Strata contributors, Baptiste Rajaut (adapted from PR #6520)
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../image/vae/safetensors_lite.hpp"
#include "ink_json.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/quant/sk_codecs.hpp"

namespace ink2 {

namespace detail {

inline bool read_file_text(const std::string& p, std::string& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

inline bool fseek64(std::FILE* fp, uint64_t off) noexcept {
#if defined(_WIN32)
    return _fseeki64(fp, static_cast<__int64>(off), SEEK_SET) == 0;
#else
    return fseeko(fp, static_cast<off_t>(off), SEEK_SET) == 0;
#endif
}

inline bool read_slice_from_file(const std::string& path, uint64_t abs_off, void* dst, size_t bytes, std::string& err) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        err = "safetensors: cannot open " + path;
        return false;
    }
    bool ok = fseek64(fp, abs_off) && (std::fread(dst, 1, bytes, fp) == bytes);
    std::fclose(fp);
    if (!ok) {
        err = "safetensors: short read in " + path + " at offset " + std::to_string(abs_off);
        return false;
    }
    return true;
}

inline int st_dtype_to_wfmt(const std::string& dt) {
    if (dt == "F32")  return 101; // F_F32
    if (dt == "BF16") return 102; // F_BF16
    if (dt == "F16")  return 103; // F_F16
    return -1;
}

inline bool st_is_expert_tensor(const std::string& name) {
    return name.find(".mlp.experts.") != std::string::npos;
}

inline bool st_is_vision_tensor(const std::string& name) {
    return name.rfind("model.visual.", 0) == 0;
}

}  // namespace detail

struct SafetensorsShard {
    std::string path;
    strata::image::StFile st;
};

struct SafetensorsAdapter {
    std::string base_dir;
    std::vector<SafetensorsShard> shards;
    std::unordered_map<std::string, std::pair<size_t, const strata::image::StTensor*>> by_name;

    bool has_mtp = false;
    SafetensorsShard mtp_shard;

    struct LayerExpertMeta {
        bool is_nvfp4 = true;
        // Shard index + StTensor for w13_weight, w13_weight_scale, w2_weight, w2_weight_scale
        size_t s_w13 = 0, s_s13 = 0, s_w2 = 0, s_s2 = 0;
        const strata::image::StTensor* t_w13 = nullptr;
        const strata::image::StTensor* t_s13 = nullptr;
        const strata::image::StTensor* t_w2 = nullptr;
        const strata::image::StTensor* t_s2 = nullptr;
        std::vector<float> a13, a2;  // [n_experts]
        float d13 = 1.0f, d2 = 1.0f;
        size_t w13_bytes_per_e = 0, s13_bytes_per_e = 0;
        size_t w2_bytes_per_e = 0, s2_bytes_per_e = 0;
        uint32_t dn_offset = 0;
        uint32_t body_bytes = 0;
    };
    std::vector<LayerExpertMeta> layers;
    std::vector<strata::artifact::StrataExpertEntry> expert_dir;

    struct Sec1Item {
        std::string name;
        bool is_mtp = false;
        size_t shard_idx = 0;
        const strata::image::StTensor* tensor = nullptr;
        uint64_t off = 0;
        uint64_t bytes = 0;
    };
    std::vector<Sec1Item> sec1_items;
    uint64_t sec1_total_bytes = 0;

    static bool is_safetensors_path(const std::string& path) {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (fs::is_directory(path, ec)) {
            if (fs::is_regular_file(fs::path(path) / "model.safetensors.index.json", ec)) return true;
            if (fs::is_regular_file(fs::path(path) / "model-00001-of-00001.safetensors", ec)) return true;
            if (fs::is_regular_file(fs::path(path) / "model.safetensors", ec)) return true;
            for (const auto& ent : fs::directory_iterator(path, ec)) {
                if (!ent.is_regular_file(ec)) continue;
                auto fn = ent.path().filename().string();
                if (fn.size() > 12 && fn.substr(fn.size() - 12) == ".safetensors" && fn != "mtp.safetensors") {
                    return true;
                }
            }
            return false;
        }
        if (path.size() >= 12 && path.substr(path.size() - 12) == ".safetensors") return true;
        if (path.size() >= 28 && path.substr(path.size() - 28) == "model.safetensors.index.json") return true;
        return false;
    }

    const std::pair<size_t, const strata::image::StTensor*>* find_tensor(const std::string& name) const {
        auto it = by_name.find(name);
        return it == by_name.end() ? nullptr : &it->second;
    }

    bool read_f32_tensor(const std::string& name, std::vector<float>& out, std::string& err) const {
        const auto* hit = find_tensor(name);
        if (!hit) {
            err = "safetensors: missing tensor " + name;
            return false;
        }
        const auto& sh = shards[hit->first];
        const auto& t = *hit->second;
        std::vector<uint8_t> raw(static_cast<size_t>(t.end - t.begin));
        if (!detail::read_slice_from_file(sh.path, sh.st.data_start + t.begin, raw.data(), raw.size(), err)) return false;
        if (t.dtype == "F32") {
            out.resize(raw.size() / 4);
            std::memcpy(out.data(), raw.data(), raw.size());
        } else if (t.dtype == "BF16") {
            out.resize(raw.size() / 2);
            for (size_t i = 0; i < out.size(); ++i) {
                uint32_t u = (static_cast<uint32_t>(raw[2 * i]) | (static_cast<uint32_t>(raw[2 * i + 1]) << 8)) << 16;
                std::memcpy(&out[i], &u, 4);
            }
        } else if (t.dtype == "F16") {
            out.resize(raw.size() / 2);
            for (size_t i = 0; i < out.size(); ++i) {
                uint16_t h = static_cast<uint16_t>(raw[2 * i] | (raw[2 * i + 1] << 8));
                uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
                uint32_t exp  = (h >> 10) & 0x1Fu;
                uint32_t mant = h & 0x03FFu;
                uint32_t f = 0;
                if (exp == 0) {
                    if (mant == 0) f = sign;
                    else {
                        uint32_t m = mant, e = 127 - 14;
                        while ((m & 0x0400u) == 0) { m <<= 1; --e; }
                        f = sign | (e << 23) | ((m & 0x03FFu) << 13);
                    }
                } else if (exp == 31) {
                    f = sign | 0x7F800000u | (mant << 13);
                } else {
                    f = sign | ((exp + 112u) << 23) | (mant << 13);
                }
                std::memcpy(&out[i], &f, 4);
            }
        } else {
            err = "safetensors: unsupported float dtype " + t.dtype + " for " + name;
            return false;
        }
        return true;
    }

    template <typename ManifestT, typename TInfoT>
    bool open_into(const std::string& input_path, ManifestT& m, std::vector<int>& dir_index_of,
                   std::vector<uint16_t>& vram_load_order, std::string& err,
                   bool keep_tokenizer = false, const std::string& explicit_dspark = "") {
        namespace fs = std::filesystem;
        std::error_code ec;
        std::vector<std::string> shard_paths;

        if (fs::is_directory(input_path, ec)) {
            base_dir = input_path;
            fs::path idx_path = fs::path(base_dir) / "model.safetensors.index.json";
            if (fs::is_regular_file(idx_path, ec)) {
                std::string idx_txt;
                if (!detail::read_file_text(idx_path.string(), idx_txt)) {
                    err = "cannot read " + idx_path.string();
                    return false;
                }
                try {
                    JVal idx = json_parse(idx_txt);
                    const JVal& wm = idx.at("weight_map");
                    std::vector<std::string> uniq;
                    for (const auto& kv : wm.o) {
                        std::string fp = (fs::path(base_dir) / kv.second.str()).string();
                        if (std::find(uniq.begin(), uniq.end(), fp) == uniq.end()) uniq.push_back(fp);
                    }
                    std::sort(uniq.begin(), uniq.end());
                    shard_paths = std::move(uniq);
                } catch (const std::exception& e) {
                    err = std::string("model.safetensors.index.json parse error: ") + e.what();
                    return false;
                }
            } else if (fs::is_regular_file(fs::path(base_dir) / "model-00001-of-00001.safetensors", ec)) {
                shard_paths.push_back((fs::path(base_dir) / "model-00001-of-00001.safetensors").string());
            } else if (fs::is_regular_file(fs::path(base_dir) / "model.safetensors", ec)) {
                shard_paths.push_back((fs::path(base_dir) / "model.safetensors").string());
            } else {
                for (const auto& ent : fs::directory_iterator(base_dir, ec)) {
                    if (!ent.is_regular_file(ec)) continue;
                    auto fn = ent.path().filename().string();
                    if (fn.size() > 12 && fn.substr(fn.size() - 12) == ".safetensors" && fn != "mtp.safetensors") {
                        shard_paths.push_back(ent.path().string());
                    }
                }
                std::sort(shard_paths.begin(), shard_paths.end());
            }
        } else {
            fs::path fp(input_path);
            base_dir = fp.parent_path().empty() ? "." : fp.parent_path().string();
            if (fp.filename().string() == "model.safetensors.index.json") {
                return open_into<ManifestT, TInfoT>(base_dir, m, dir_index_of, vram_load_order, err, keep_tokenizer, explicit_dspark);
            }
            shard_paths.push_back(input_path);
        }

        if (shard_paths.empty()) {
            err = "safetensors: no weight shards found in " + input_path;
            return false;
        }

        // 1. Parse config.json (if present; falls back to Inkling-2-Mini defaults)
        m = ManifestT{};
        m.format = "safetensors";
        m.build = "HF-NVFP4-MXFP8";
        m.arch = "inkling2_loop";
        {
            // Default local layers for Inkling-2-Mini (every layer except 5, 11, 17, 23)
            m.c.is_swa.assign(m.c.n_phys, 1);
            for (int l = 5; l < m.c.n_phys; l += 6) m.c.is_swa[l] = 0;

            std::string cfg_txt;
            if (detail::read_file_text((fs::path(base_dir) / "config.json").string(), cfg_txt)) {
                try {
                    JVal cfg = json_parse(cfg_txt);
                    const JVal& tc = cfg.get("text_config") ? cfg.at("text_config") : cfg;
                    auto& c = m.c;
                    if (const JVal* v = tc.get("num_hidden_layers")) c.n_phys = static_cast<int>(v->i());
                    if (const JVal* v = tc.get("hidden_size")) c.hidden = static_cast<int>(v->i());
                    if (const JVal* v = tc.get("intermediate_size")) c.inter = static_cast<int>(v->i());
                    if (const JVal* v = tc.get("num_attention_heads")) c.n_head = static_cast<int>(v->i());
                    if (const JVal* v = tc.get("num_key_value_heads")) c.n_kv = static_cast<int>(v->i());
                    if (const JVal* v = tc.get("head_dim")) c.head_dim = static_cast<int>(v->i());
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
                        c.core_end   = static_cast<int>(lc->at("core_end").i());
                        c.num_passes = static_cast<int>(lc->at("num_passes").i());
                        c.gate_rank  = static_cast<int>(lc->at("gate_rank").i());
                    }
                    if (const JVal* loc = tc.get("local_layer_ids")) {
                        c.is_swa.assign(c.n_phys, 0);
                        for (const auto& x : loc->a) {
                            int id = static_cast<int>(x.i());
                            if (id >= 0 && id < c.n_phys) c.is_swa[id] = 1;
                        }
                    }
                    if (const JVal* mtp = cfg.get("mtp_config")) {
                        auto& d = m.dspark;
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
                } catch (const std::exception& e) {
                    err = std::string("config.json parse error: ") + e.what();
                    return false;
                }
            }
            m.c.build_visit_schedule();
        }

        // 2. Read headers of all safetensors shards
        shards.resize(shard_paths.size());
        by_name.clear();
        for (size_t i = 0; i < shard_paths.size(); ++i) {
            shards[i].path = shard_paths[i];
            std::string e = strata::image::st_read_header(shard_paths[i], shards[i].st);
            if (!e.empty()) {
                err = "safetensors header: " + e;
                return false;
            }
            for (const auto& kv : shards[i].st.tensors) {
                by_name[kv.first] = {i, &kv.second};
            }
        }

        // 3. Optional mtp.safetensors (DSpark)
        std::string mtp_path = explicit_dspark;
        if (mtp_path.empty()) {
            fs::path auto_mtp = fs::path(base_dir) / "mtp.safetensors";
            if (fs::is_regular_file(auto_mtp, ec)) mtp_path = auto_mtp.string();
        }
        has_mtp = false;
        if (!mtp_path.empty() && mtp_path.size() >= 12 && mtp_path.substr(mtp_path.size() - 12) == ".safetensors") {
            mtp_shard.path = mtp_path;
            std::string e = strata::image::st_read_header(mtp_path, mtp_shard.st);
            if (!e.empty()) {
                err = "mtp.safetensors header: " + e;
                return false;
            }
            has_mtp = !mtp_shard.st.tensors.empty();
        }

        // 4. Build virtual Section 1 layout (dense trunk + DSpark tensors)
        sec1_items.clear();
        sec1_total_bytes = 0;
        auto align64 = [](uint64_t x) -> uint64_t { return (x + 63ull) & ~63ull; };

        for (const auto& kv : by_name) {
            const std::string& name = kv.first;
            if (detail::st_is_expert_tensor(name) || detail::st_is_vision_tensor(name)) continue;
            if (name.rfind("model.llm.", 0) != 0) continue;
            const auto& st_t = *kv.second.second;
            const int wfmt = detail::st_dtype_to_wfmt(st_t.dtype);
            if (wfmt < 0) continue;

            uint64_t off = align64(sec1_total_bytes);
            uint64_t sz  = st_t.end - st_t.begin;
            TInfoT ti;
            ti.name = name;
            ti.fmt = wfmt;
            ti.shape = st_t.shape;
            ti.off = off;
            ti.bytes = sz;
            m.trunk[name] = std::move(ti);
            sec1_items.push_back({name, false, kv.second.first, &st_t, off, sz});
            sec1_total_bytes = off + sz;
        }

        if (has_mtp) {
            for (const auto& kv : mtp_shard.st.tensors) {
                const std::string& name = kv.first;
                const auto& st_t = kv.second;
                const int wfmt = detail::st_dtype_to_wfmt(st_t.dtype);
                if (wfmt < 0) continue;
                uint64_t off = align64(sec1_total_bytes);
                uint64_t sz  = st_t.end - st_t.begin;
                TInfoT ti;
                ti.name = name;
                ti.fmt = wfmt;
                ti.shape = st_t.shape;
                ti.off = off;
                ti.bytes = sz;
                m.dspark_tensors[name] = std::move(ti);
                sec1_items.push_back({name, true, 0, &st_t, off, sz});
                sec1_total_bytes = off + sz;
            }
            m.dspark.present = !m.dspark_tensors.empty();
        }

        // 5. Index all routed experts (NVFP4 -> SF4 code_id=15, MXFP8 -> MX8 code_id=16) with ZERO requantization!
        const int n_phys = m.c.n_phys;
        const int n_exp  = m.c.n_experts;
        layers.assign(n_phys, LayerExpertMeta{});
        expert_dir.clear();
        expert_dir.reserve(static_cast<size_t>(n_phys) * n_exp);
        dir_index_of.assign(static_cast<size_t>(n_phys) * n_exp, -1);

        uint64_t virt_body_off = 0;
        for (int il = 0; il < n_phys; ++il) {
            const std::string q = "model.llm.layers." + std::to_string(il) + ".mlp.experts.";
            auto& lm = layers[il];
            const auto* h_w13 = find_tensor(q + "w13_weight");
            const auto* h_s13 = find_tensor(q + "w13_weight_scale");
            const auto* h_w2  = find_tensor(q + "w2_weight");
            const auto* h_s2  = find_tensor(q + "w2_weight_scale");
            if (!h_w13 || !h_s13 || !h_w2 || !h_s2) {
                err = "safetensors: missing expert tensors for layer " + std::to_string(il);
                return false;
            }
            lm.s_w13 = h_w13->first; lm.t_w13 = h_w13->second;
            lm.s_s13 = h_s13->first; lm.t_s13 = h_s13->second;
            lm.s_w2  = h_w2->first;  lm.t_w2  = h_w2->second;
            lm.s_s2  = h_s2->first;  lm.t_s2  = h_s2->second;

            lm.w13_bytes_per_e = static_cast<size_t>(lm.t_w13->end - lm.t_w13->begin) / static_cast<size_t>(n_exp);
            lm.s13_bytes_per_e = static_cast<size_t>(lm.t_s13->end - lm.t_s13->begin) / static_cast<size_t>(n_exp);
            lm.w2_bytes_per_e  = static_cast<size_t>(lm.t_w2->end  - lm.t_w2->begin)  / static_cast<size_t>(n_exp);
            lm.s2_bytes_per_e  = static_cast<size_t>(lm.t_s2->end  - lm.t_s2->begin)  / static_cast<size_t>(n_exp);

            const bool has_amax = (find_tensor(q + "w13_weight_amax") != nullptr);
            lm.is_nvfp4 = has_amax;
            if (has_amax) {
                std::vector<float> d13v, d2v;
                if (!read_f32_tensor(q + "w13_weight_amax", lm.a13, err)) return false;
                if (!read_f32_tensor(q + "w2_weight_amax", lm.a2, err)) return false;
                if (find_tensor(q + "w13_weight_outer_scale_denominator")) {
                    if (!read_f32_tensor(q + "w13_weight_outer_scale_denominator", d13v, err)) return false;
                    lm.d13 = d13v.empty() ? 1.0f : d13v[0];
                } else {
                    lm.d13 = 2688.0f;
                }
                if (find_tensor(q + "w2_weight_outer_scale_denominator")) {
                    if (!read_f32_tensor(q + "w2_weight_outer_scale_denominator", d2v, err)) return false;
                    lm.d2 = d2v.empty() ? 1.0f : d2v[0];
                } else {
                    lm.d2 = 2688.0f;
                }
                lm.dn_offset = static_cast<uint32_t>(16u + lm.w13_bytes_per_e + lm.s13_bytes_per_e);
                lm.body_bytes = static_cast<uint32_t>(lm.dn_offset + lm.w2_bytes_per_e + lm.s2_bytes_per_e);
            } else {
                lm.dn_offset = static_cast<uint32_t>(lm.w13_bytes_per_e + lm.s13_bytes_per_e);
                lm.body_bytes = static_cast<uint32_t>(lm.dn_offset + lm.w2_bytes_per_e + lm.s2_bytes_per_e);
            }

            const uint8_t code = lm.is_nvfp4
                ? static_cast<uint8_t>(strata::quant::sk::SkCode::SF4)
                : static_cast<uint8_t>(strata::quant::sk::SkCode::MX8);
            for (int e = 0; e < n_exp; ++e) {
                strata::artifact::StrataExpertEntry ent{};
                ent.expert_id   = static_cast<uint16_t>(e);
                ent.gu_code_id  = code;
                ent.dn_code_id  = code;
                ent.body_offset = virt_body_off;
                ent.body_bytes  = lm.body_bytes;
                ent.dn_offset   = lm.dn_offset;
                int idx = static_cast<int>(expert_dir.size());
                expert_dir.push_back(ent);
                dir_index_of[il * n_exp + e] = idx;
                virt_body_off += lm.body_bytes;
            }
        }

        // 6. 3x loop-weighted vram_load_order: core L6..17 first, then L0..5, then L18..23
        vram_load_order.clear();
        vram_load_order.reserve(expert_dir.size());
        for (int il = m.c.core_start; il < m.c.core_end; ++il)
            for (int e = 0; e < n_exp; ++e)
                vram_load_order.push_back(static_cast<uint16_t>(il * n_exp + e));
        for (int il = 0; il < m.c.core_start; ++il)
            for (int e = 0; e < n_exp; ++e)
                vram_load_order.push_back(static_cast<uint16_t>(il * n_exp + e));
        for (int il = m.c.core_end; il < n_phys; ++il)
            for (int e = 0; e < n_exp; ++e)
                vram_load_order.push_back(static_cast<uint16_t>(il * n_exp + e));

        // 7. Tokenizer & chat template (optional)
        if (keep_tokenizer) {
            detail::read_file_text((fs::path(base_dir) / "chat_template.jinja").string(), m.chat_template);
            detail::read_file_text((fs::path(base_dir) / "tokenizer.json").string(), m.tokenizer_json);
        }
        return true;
    }

    bool read_section1(std::vector<uint8_t>& out, std::string& err) const {
        out.assign(static_cast<size_t>(sec1_total_bytes), 0);
        for (const auto& item : sec1_items) {
            const auto& sh = item.is_mtp ? mtp_shard : shards[item.shard_idx];
            if (!detail::read_slice_from_file(sh.path, sh.st.data_start + item.tensor->begin,
                                              out.data() + item.off, static_cast<size_t>(item.bytes), err)) {
                return false;
            }
        }
        return true;
    }

    bool read_expert_body(size_t dir_index, int n_experts, std::vector<uint8_t>& out, std::string& err) const {
        if (dir_index >= expert_dir.size()) {
            err = "safetensors: expert dir_index out of bounds";
            return false;
        }
        const int il = static_cast<int>(dir_index) / n_experts;
        const int e  = static_cast<int>(dir_index) % n_experts;
        const auto& lm = layers[il];
        out.resize(lm.body_bytes);

        size_t pos = 0;
        if (lm.is_nvfp4) {
            float hdr[4] = {lm.a13[e], lm.d13, lm.a2[e], lm.d2};
            std::memcpy(out.data(), hdr, 16);
            pos = 16;
        }
        if (!detail::read_slice_from_file(shards[lm.s_w13].path,
                shards[lm.s_w13].st.data_start + lm.t_w13->begin + static_cast<uint64_t>(e) * lm.w13_bytes_per_e,
                out.data() + pos, lm.w13_bytes_per_e, err)) return false;
        pos += lm.w13_bytes_per_e;

        if (!detail::read_slice_from_file(shards[lm.s_s13].path,
                shards[lm.s_s13].st.data_start + lm.t_s13->begin + static_cast<uint64_t>(e) * lm.s13_bytes_per_e,
                out.data() + pos, lm.s13_bytes_per_e, err)) return false;
        pos += lm.s13_bytes_per_e;

        if (!detail::read_slice_from_file(shards[lm.s_w2].path,
                shards[lm.s_w2].st.data_start + lm.t_w2->begin + static_cast<uint64_t>(e) * lm.w2_bytes_per_e,
                out.data() + pos, lm.w2_bytes_per_e, err)) return false;
        pos += lm.w2_bytes_per_e;

        if (!detail::read_slice_from_file(shards[lm.s_s2].path,
                shards[lm.s_s2].st.data_start + lm.t_s2->begin + static_cast<uint64_t>(e) * lm.s2_bytes_per_e,
                out.data() + pos, lm.s2_bytes_per_e, err)) return false;
        return true;
    }
};

}  // namespace ink2
