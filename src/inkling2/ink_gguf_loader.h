// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/ink_gguf_loader.h - Direct Unsloth / community .gguf loader for Inkling-2-Mini.
// Ported from upstream v0.2.19 (MIT, Strata contributors) for the adopted src/inkling2 engine.
// XeStrata's gguf_reader, artifact/dequant and quant/sk_codecs replace upstream's copies; the
// safetensors adapter is not part of this port.
// engine/src/inkling2/ink_gguf_loader.h - Direct Community / Unsloth .gguf loader for strata-inkling
// Supports single & split multi-shard GGUFs (UD-Q4_K_XL, IQ3_S, IQ4_NL, IQ4_XS, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4)
// in BOTH llama.cpp master (Unsloth GGUF_shard_rewrite) and PR #32187 layouts + optional GGUF dspark / mmproj + embedded o200k tokenizer.
//
// MIT License
// Copyright (c) 2026 Strata contributors
// Based on Project Maya (https://github.com/mw00/project-maya)
// Copyright (c) 2026 Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors
// Adapted with contributions from Baptiste Rajaut and merbanan (PR #6520, PR #6506).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ink_batch.h"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/strata_reader.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/quant/sk_codecs.hpp"
#if defined(STRATA_NATIVE_EXPERTS)
#include "ggml.h"
#endif

namespace ink2 {

namespace gguf_detail {

inline bool has_gguf_magic(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    uint32_t magic = 0;
    size_t n = std::fread(&magic, 1, 4, fp);
    std::fclose(fp);
    return n == 4 && magic == 0x46554747u;
}


// XeStrata's artifact/dequant.hpp is the source of the scalar dequantizers.
inline bool mv_dequant_row(uint32_t t, const uint8_t* src, float* dst, int64_t n);

// Decode a single row of `cols` elements from a GGUF tensor into FP32
inline bool gguf_dequant_row_f32(uint32_t ggml_type, const uint8_t* src, float* dst, int64_t cols) {
    if (ggml_type == 0) { // F32
        std::memcpy(dst, src, static_cast<size_t>(cols) * 4);
        return true;
    }
    if (ggml_type == 1) { // F16
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
        for (int64_t i = 0; i < cols; ++i) dst[i] = fp16_to_f32(p[i]);
        return true;
    }
    if (ggml_type == 30) { // BF16
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
        for (int64_t i = 0; i < cols; ++i) dst[i] = bf16_to_f32(p[i]);
        return true;
    }
    // XeStrata's scalar dequantizers live in artifact/dequant.hpp (namespace strata), not in
    // upstream's quant/dequant.hpp.  Every encoding UD-Q4_K_XL uses in the trunk is covered here.
    return mv_dequant_row(ggml_type, src, dst, cols);
}

inline bool mv_dequant_row(uint32_t t, const uint8_t* src, float* dst, int64_t n) {
    int be = 0, bb = 0;
    if (n <= 0 || !strata::block_geometry(t, be, bb) || be <= 0 || bb <= 0 || n % be != 0) return false;
    for (int64_t i = 0; i < n; i += be, src += bb) {
        switch (t) {
        case 2:  strata::dequantize_q4_0(src, dst + i); break;   // Q4_0
        case 6:  strata::dequantize_q5_0(src, dst + i); break;   // Q5_0
        case 7:  strata::dequantize_q5_1(src, dst + i); break;   // Q5_1
        case 8:  strata::dequantize_q8_0(src, dst + i); break;   // Q8_0
        case 11: strata::dequantize_q3_K(src, dst + i); break;   // Q3_K
        case 12: strata::dequantize_q4_K(src, dst + i); break;  // Q4_K
        case 13: strata::dequantize_q5_K(src, dst + i); break;  // Q5_K
        case 14: strata::dequantize_q6_K(src, dst + i); break;  // Q6_K
        case 20: strata::dequantize_iq4_nl(src, dst + i); break;// IQ4_NL
        case 23: strata::dequantize_iq4_xs(src, dst + i); break;// IQ4_XS
        case 42: strata::dequantize_q2_0(src, dst + i); break;  // Q2_0
        default:
#if defined(STRATA_NATIVE_EXPERTS)
            // the i-quants and NVFP4 (its per-tensor scale is the caller's): ggml's own dequantizer
            if (t < GGML_TYPE_COUNT && ggml_get_type_traits(static_cast<ggml_type>(t))->to_float) {
                ggml_get_type_traits(static_cast<ggml_type>(t))->to_float(src, dst + i, be);
                break;
            }
#endif
            return false;
        }
    }
    return true;
}

inline size_t gguf_row_bytes(uint32_t ggml_type, int64_t cols) {
    int be = 0, bb = 0;
    if (!strata::block_geometry(ggml_type, be, bb) || be <= 0) return 0;
    return static_cast<size_t>(cols / be) * static_cast<size_t>(bb);
}

// Bit-exact lossless repack of one row of GGUF block_nvfp4 (64 values = 4 UE4M3 scales + 32B E2M1 split-half nibbles)
// into canonical .strata v1.1 SF4 layout (ncols/2 interleaved E2M1 nibbles + ncols/16 E4M3 scales).
// Matches dequant_nvfp4() bit-for-bit with ZERO requantization!
inline void repack_gguf_nvfp4_row_to_sf4(const uint8_t* src_row, uint8_t* dst_packed, uint8_t* dst_scales, int64_t cols) noexcept {
    const int64_t n_blk64 = cols / 64;
    for (int64_t b = 0; b < n_blk64; ++b) {
        const uint8_t* blk = src_row + static_cast<size_t>(b) * 36;
        // 4 sub-block scales (UE4M3, bit 7 == 0)
        for (int s = 0; s < 4; ++s) {
            dst_scales[b * 4 + s] = blk[s] & 0x7Fu;
        }
        const uint8_t* qs = blk + 4;
        uint8_t* out = dst_packed + static_cast<size_t>(b) * 32;
        for (int s = 0; s < 4; ++s) {
            const uint8_t* sq = qs + s * 8;
            uint8_t* so = out + s * 8;
            // In block_nvfp4: sq[0..7] low nibbles = elem 0..7, high nibbles = elem 8..15
            // In SF4: so[k] low nibble = elem 2k, high nibble = elem 2k+1
            so[0] = static_cast<uint8_t>((sq[0] & 0x0Fu) | ((sq[1] & 0x0Fu) << 4));
            so[1] = static_cast<uint8_t>((sq[2] & 0x0Fu) | ((sq[3] & 0x0Fu) << 4));
            so[2] = static_cast<uint8_t>((sq[4] & 0x0Fu) | ((sq[5] & 0x0Fu) << 4));
            so[3] = static_cast<uint8_t>((sq[6] & 0x0Fu) | ((sq[7] & 0x0Fu) << 4));
            so[4] = static_cast<uint8_t>(((sq[0] >> 4) & 0x0Fu) | (sq[1] & 0xF0u));
            so[5] = static_cast<uint8_t>(((sq[2] >> 4) & 0x0Fu) | (sq[3] & 0xF0u));
            so[6] = static_cast<uint8_t>(((sq[4] >> 4) & 0x0Fu) | (sq[5] & 0xF0u));
            so[7] = static_cast<uint8_t>(((sq[6] >> 4) & 0x0Fu) | (sq[7] & 0xF0u));
        }
    }
}

inline std::string json_escape_str(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    out.push_back('"');
    for (unsigned char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", unsigned(c));
            out += buf;
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    out.push_back('"');
    return out;
}

// Full-array GGUF tokenizer extractor (strata::GgufFile truncates array metadata to 64 items for speed,
// so --export-tokenizer re-scans shard 0's KV header directly to extract all 200K tokens & merges cleanly!)
inline bool extract_gguf_tokenizer_json(const std::string& shard0_path, std::string& out_tokenizer_json,
                                        std::string& out_chat_template, std::string& err) {
    std::FILE* fp = std::fopen(shard0_path.c_str(), "rb");
    if (!fp) { err = "cannot open GGUF for tokenizer export: " + shard0_path; return false; }

    auto rd_u32 = [&](uint32_t& v) { return std::fread(&v, 4, 1, fp) == 1; };
    auto rd_u64 = [&](uint64_t& v) { return std::fread(&v, 8, 1, fp) == 1; };
    auto rd_str = [&](std::string& s) -> bool {
        uint64_t len = 0;
        if (!rd_u64(len) || len > (128ull << 20)) return false;
        s.resize(static_cast<size_t>(len));
        return len == 0 || std::fread(s.data(), 1, static_cast<size_t>(len), fp) == static_cast<size_t>(len);
    };
    auto skip_scalar = [&](uint32_t t) -> bool {
        size_t nb = 0;
        switch (t) {
            case 0: case 1: case 7: nb = 1; break;
            case 2: case 3: nb = 2; break;
            case 4: case 5: case 6: nb = 4; break;
            case 10: case 11: case 12: nb = 8; break;
            case 8: { std::string tmp; return rd_str(tmp); }
            default: return false;
        }
        uint8_t buf[8];
        return std::fread(buf, 1, nb, fp) == nb;
    };
    auto skip_val = [&](auto&& self, uint32_t t) -> bool {
        if (t != 9) return skip_scalar(t);
        uint32_t et = 0; uint64_t cnt = 0;
        if (!rd_u32(et) || !rd_u64(cnt)) return false;
        for (uint64_t i = 0; i < cnt; ++i) if (!self(self, et)) return false;
        return true;
    };

    uint32_t magic = 0, ver = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!rd_u32(magic) || magic != 0x46554747u || !rd_u32(ver) || !rd_u64(n_tensors) || !rd_u64(n_kv)) {
        std::fclose(fp);
        err = "invalid GGUF header during tokenizer extraction";
        return false;
    }

    std::vector<std::string> tokens, merges;
    std::vector<int32_t> token_types;
    std::string pre = "gpt-4o";
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t t = 0;
        if (!rd_str(key) || !rd_u32(t)) { std::fclose(fp); err = "truncated GGUF KV"; return false; }
        if (key == "tokenizer.chat_template" && t == 8) {
            if (!rd_str(out_chat_template)) { std::fclose(fp); return false; }
        } else if (key == "tokenizer.ggml.pre" && t == 8) {
            if (!rd_str(pre)) { std::fclose(fp); return false; }
        } else if (key == "tokenizer.ggml.tokens" && t == 9) {
            uint32_t et = 0; uint64_t cnt = 0;
            if (!rd_u32(et) || !rd_u64(cnt) || et != 8) { std::fclose(fp); return false; }
            tokens.resize(static_cast<size_t>(cnt));
            for (uint64_t j = 0; j < cnt; ++j) if (!rd_str(tokens[j])) { std::fclose(fp); return false; }
        } else if (key == "tokenizer.ggml.merges" && t == 9) {
            uint32_t et = 0; uint64_t cnt = 0;
            if (!rd_u32(et) || !rd_u64(cnt) || et != 8) { std::fclose(fp); return false; }
            merges.resize(static_cast<size_t>(cnt));
            for (uint64_t j = 0; j < cnt; ++j) if (!rd_str(merges[j])) { std::fclose(fp); return false; }
        } else if (key == "tokenizer.ggml.token_type" && t == 9) {
            uint32_t et = 0; uint64_t cnt = 0;
            if (!rd_u32(et) || !rd_u64(cnt)) { std::fclose(fp); return false; }
            token_types.resize(static_cast<size_t>(cnt), 1);
            for (uint64_t j = 0; j < cnt; ++j) {
                int32_t v = 1;
                if (et == 4 || et == 5) { if (std::fread(&v, 4, 1, fp) != 1) { std::fclose(fp); return false; } }
                else { if (!skip_scalar(et)) { std::fclose(fp); return false; } }
                token_types[j] = v;
            }
        } else {
            if (!skip_val(skip_val, t)) { std::fclose(fp); return false; }
        }
    }
    std::fclose(fp);

    if (tokens.empty()) return true;

    // Synthesize Hugging Face tokenizer.json (BPE + ignore_merges=true for o200k / inkling2)
    std::string j;
    j.reserve(tokens.size() * 24 + merges.size() * 16 + 4096);
    j += "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[";
    bool first_added = true;
    for (size_t i = 0; i < tokens.size(); ++i) {
        int tt = (i < token_types.size()) ? token_types[i] : 1;
        if (tt == 3 || tt == 4) {
            if (!first_added) j.push_back(',');
            first_added = false;
            j += "{\"id\":" + std::to_string(i) + ",\"content\":" + json_escape_str(tokens[i]) +
                 ",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":" +
                 (tt == 3 ? "true" : "false") + "}";
        }
    }
    j += "],\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
         "{\"type\":\"Split\",\"pattern\":{\"Regex\":"
         "\"[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
         "|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
         "|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n/]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"},"
         "\"behavior\":\"Isolated\",\"invert\":false},"
         "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}]},"
         "\"post_processor\":null,\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true},"
         "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":null,"
         "\"end_of_word_suffix\":null,\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":true,\"vocab\":{";
    bool first_v = true;
    for (size_t i = 0; i < tokens.size(); ++i) {
        int tt = (i < token_types.size()) ? token_types[i] : 1;
        if (tt == 3 || tt == 4) continue;
        if (!first_v) j.push_back(',');
        first_v = false;
        j += json_escape_str(tokens[i]) + ":" + std::to_string(i);
    }
    j += "},\"merges\":[";
    for (size_t i = 0; i < merges.size(); ++i) {
        if (i) j.push_back(',');
        j += json_escape_str(merges[i]);
    }
    j += "]}}";
    out_tokenizer_json = std::move(j);
    return true;
}

}  // namespace gguf_detail

struct GgufAdapter {
    std::unique_ptr<strata::GgufModel> model;
    std::unique_ptr<strata::GgufModel> dspark_gguf;
    bool is_rewrite_layout = false;  // true = llama.cpp master / Unsloth (blk.0.ffn_gate_up_exps.weight)

    std::vector<uint8_t> sec1_arena; // Assembled canonical Section 1 trunk + DSpark bytes
    std::vector<strata::artifact::StrataExpertEntry> expert_dir;

    struct LayerExpertMeta {
        bool is_nvfp4_to_sf4 = false;
        bool requant_q8_0 = false;           // a type the GPU's kernels do not read: dequantized and written as Q8_0
        bool is_fused_gu = false;
        uint32_t gu_type = 0, up_type = 0, dn_type = 0;
        const uint8_t* gu_base = nullptr;    // fused gate_up OR gate
        const uint8_t* up_base = nullptr;    // nullptr if fused_gu
        const uint8_t* dn_base = nullptr;
        size_t gu_stride = 0;                // bytes per expert in gu_base
        size_t up_stride = 0;                // bytes per expert in up_base
        size_t dn_stride = 0;                // bytes per expert in dn_base
        std::vector<float> gate_scale, up_scale, dn_scale;  // [n_experts]
        uint32_t dn_offset = 0;
        uint32_t body_bytes = 0;
        int hidden = 2304, inter = 1152;
    };
    std::vector<LayerExpertMeta> layers;

    static bool is_gguf_path(const std::string& path) {
        return gguf_detail::has_gguf_magic(path);
    }

    // A GGUF of Inkling-2-Mini (general.architecture inkling2 or inkling2_loop): the engine's other GGUFs are not ours.
    static bool is_inkling_gguf(const std::string& path) {
        if (!gguf_detail::has_gguf_magic(path)) return false;
        try {
            const auto gm = strata::GgufModel::open(path);
            const auto* a = gm.meta().get("general.architecture");
            return a && (a->s == "inkling2" || a->s == "inkling2_loop");
        } catch (const std::exception&) {
            return false;
        }
    }

    const uint8_t* tensor_ptr(const strata::GgufModel& gm, const std::string& name, const strata::TensorInfo*& ti_out) const {
        size_t s = 0;
        const strata::TensorInfo* ti = gm.find(name, &s);
        if (!ti) { ti_out = nullptr; return nullptr; }
        ti_out = ti;
        return gm.shard(s).tensor_data(*ti);
    }

    bool read_tensor_f32(const strata::GgufModel& gm, const std::string& name, std::vector<float>& out, std::string& err) const {
        const strata::TensorInfo* ti = nullptr;
        const uint8_t* ptr = tensor_ptr(gm, name, ti);
        if (!ti || !ptr) { err = "GGUF: missing tensor " + name; return false; }
        const int64_t cols = static_cast<int64_t>(ti->shape[0]);
        const int64_t rows = static_cast<int64_t>(ti->elements() / static_cast<uint64_t>(cols));
        const size_t rb = gguf_detail::gguf_row_bytes(ti->type, cols);
        if (rb == 0) { err = "GGUF: unsupported tensor type " + std::string(ti->type_name()) + " for " + name; return false; }
        out.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
        for (int64_t r = 0; r < rows; ++r) {
            if (!gguf_detail::gguf_dequant_row_f32(ti->type, ptr + static_cast<size_t>(r) * rb,
                                                   out.data() + static_cast<size_t>(r) * cols, cols)) {
                err = "GGUF: dequant failed for " + name;
                return false;
            }
        }
        return true;
    }

    template <typename ManifestT, typename TInfoT>
    bool open_into(const std::string& path, ManifestT& m, std::vector<int>& dir_index_of,
                   std::vector<uint16_t>& vram_load_order, std::string& err,
                   bool keep_tokenizer = false, const std::string& explicit_dspark = "") {
        try {
            model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(path));
        } catch (const std::exception& e) {
            err = std::string("GGUF open: ") + e.what();
            return false;
        }

        const auto& meta = model->meta();
        const auto* arch_v = meta.get("general.architecture");
        if (!arch_v || (arch_v->s != "inkling2" && arch_v->s != "inkling2_loop")) {
            err = "GGUF architecture is '" + (arch_v ? arch_v->s : std::string("(missing)")) +
                  "', expected 'inkling2' or 'inkling2_loop'";
            return false;
        }

        is_rewrite_layout = (model->find("blk.0.ffn_gate_up_exps.weight") != nullptr);

        // 1. Parse InkCfg from GGUF metadata (supports BOTH llama.cpp master & PR #32187 keys)
        m = ManifestT{};
        m.format = "gguf";
        m.arch = "inkling2_loop";
        {
            const std::string ap = arch_v->s + ".";
            auto get_num = [&](const std::vector<const char*>& keys, double def) -> double {
                for (const char* k : keys) {
                    if (const auto* v = meta.get(ap + k)) {
                        if (v->type == strata::MetaType::ARRAY && !v->items.empty()) return v->items[0].num();
                        if (v->is_num()) return v->num();
                    }
                }
                return def;
            };
            auto& c = m.c;
            const int n_visits_meta = static_cast<int>(get_num({"block_count"}, 48));
            c.hidden     = static_cast<int>(get_num({"embedding_length"}, 2304));
            c.inter      = static_cast<int>(get_num({"expert_feed_forward_length"}, 1152));
            c.n_head     = static_cast<int>(get_num({"attention.head_count"}, 24));
            c.n_kv       = static_cast<int>(get_num({"attention.head_count_kv"}, 8));
            c.head_dim   = static_cast<int>(get_num({"attention.key_length"}, 128));
            c.rms_eps    = static_cast<float>(get_num({"attention.layer_norm_rms_epsilon"}, 1e-6));
            c.n_experts  = static_cast<int>(get_num({"expert_count"}, 128));
            c.top_k      = static_cast<int>(get_num({"expert_used_count"}, 8));
            c.route_scale = static_cast<float>(get_num({"expert_weights_scale"}, 8.0));
            c.swa_win    = static_cast<int>(get_num({"attention.sliding_window"}, 1024));
            c.d_rel      = static_cast<int>(get_num({"d_rel", "attention.relative_length"}, 16));
            c.rel_extent = static_cast<int>(get_num({"rel_extent", "attention.relative_extent"}, 1024));
            c.sconv_k    = static_cast<int>(get_num({"shortconv_kernel", "shortconv.l_cache"}, 4));
            c.log_scale_floor = static_cast<int>(get_num({"log_scaling_n_floor", "attention.log_scale_floor"}, 8192));
            c.log_scale_alpha = static_cast<float>(get_num({"log_scaling_alpha", "attention.log_scale_alpha"}, 0.1));
            c.qk_l2_clip = static_cast<float>(get_num({"qk_clip_threshold", "attention.qk_clip"}, 22.627416997969522));
            c.rel_l2_clip = static_cast<float>(get_num({"rel_clip_threshold", "attention.relative_clip"}, 8.0));
            c.block_size = static_cast<int>(get_num({"attn_res.block_size"}, 6));
            c.situ_beta  = static_cast<float>(get_num({"mlp_gate_softcap", "activation.situ_beta"}, 4.0));
            c.situ_lb    = static_cast<float>(get_num({"mlp_up_softcap", "activation.situ_linear_beta"}, 25.0));
            c.logit_softcap = static_cast<float>(get_num({"final_logit_softcapping"}, 30.0));
            c.vocab_unpadded = static_cast<int>(get_num({"unpadded_vocab_size", "vocab_size_unpadded"}, 200058));
            c.num_passes = static_cast<int>(get_num({"num_loops"}, 3));
            c.core_start = static_cast<int>(get_num({"loop.core_start"}, 6));
            c.core_end   = static_cast<int>(get_num({"loop.core_end"}, 18));
            c.gate_rank  = static_cast<int>(get_num({"loop.gate_rank"}, 64));
            if (const auto* tok_emb = model->find("token_embd.weight")) {
                if (tok_emb->shape.size() >= 2) c.vocab_padded = static_cast<int>(tok_emb->shape[1]);
            }
            if (const auto* eos = meta.get("tokenizer.ggml.eos_token_id")) c.eos_token_id = static_cast<int>(eos->u);

            const int core_len = (c.num_passes > 1) ? (c.core_end - c.core_start) : 0;
            c.n_phys = n_visits_meta - (c.num_passes - 1) * core_len;
            if (c.n_phys <= 0) c.n_phys = 24;

            c.is_swa.assign(c.n_phys, 1);
            if (const auto* pat = meta.get(ap + "attention.sliding_window_pattern");
                pat && pat->type == strata::MetaType::ARRAY && static_cast<int>(pat->items.size()) >= n_visits_meta) {
                const int np = static_cast<int>(pat->items.size());
                for (int l = 0; l < c.n_phys; ++l) {
                    // Mainline Unsloth files carry one flag per physical layer (24); older files
                    // carry one per visit (48).  The core layers share their weights across passes.
                    int v = (np >= c.n_phys + (c.num_passes - 1) * core_len && l >= c.core_end)
                        ? l + (c.num_passes - 1) * core_len : l;
                    bool swa_v = false;
                    if (v < np) swa_v = (pat->items[v].num() != 0.0);
                    c.is_swa[l] = swa_v ? 1 : 0;
                }
            } else {
                for (int l = 5; l < c.n_phys; l += 6) c.is_swa[l] = 0;
            }
            c.build_visit_schedule();

            // Detect quant summary for build label
            const strata::TensorInfo* ex0 = model->find(is_rewrite_layout ? "blk.0.ffn_gate_up_exps.weight" : "blk.0.ffn_gate_exps.weight");
            m.build = std::string("GGUF-") + (ex0 ? ex0->type_name() : "CUSTOM");
        }

        // 2. Assemble canonical Section 1 trunk tensors from GGUF
        sec1_arena.clear();
        // ~63u is a 32-bit mask: its zero-extended upper half wraps offsets above 4 GiB.
        auto align64 = [](size_t x) -> size_t { return (x + size_t(63)) & ~size_t(63); };

        auto add_f32 = [&](const std::string& canon_name, const std::vector<float>& vals, const std::vector<int64_t>& shape) {
            size_t off = align64(sec1_arena.size());
            size_t nb = vals.size() * sizeof(float);
            sec1_arena.resize(off + nb, 0);
            if (nb > 0) std::memcpy(sec1_arena.data() + off, vals.data(), nb);
            TInfoT ti;
            ti.name = canon_name;
            ti.fmt = 101; // F_F32
            ti.shape = shape;
            ti.off = off;
            ti.bytes = nb;
            m.trunk[canon_name] = std::move(ti);
        };

        auto copy_mat_bf16 = [&](const std::string& gguf_name, const std::string& canon_name, int64_t rows, int64_t cols) -> bool {
            const strata::TensorInfo* ti = nullptr;
            const uint8_t* ptr = tensor_ptr(*model, gguf_name, ti);
            if (!ti || !ptr) { err = "GGUF: missing tensor " + gguf_name; return false; }
            size_t off = align64(sec1_arena.size());
            if (ti->type == 8 && cols % 32 == 0) {   // Q8_0: the engine reads the rows as they are (FMT_Q8_0)
                // split for the GPU's aligned loads: the rows' FP32 block scales, then their int8 weights
                const size_t nb = static_cast<size_t>(rows) * static_cast<size_t>(cols / 32);
                const size_t qb = nb * 4 + static_cast<size_t>(rows) * static_cast<size_t>(cols);
                sec1_arena.resize(off + qb, 0);
                float* sc = reinterpret_cast<float*>(sec1_arena.data() + off);
                int8_t* q = reinterpret_cast<int8_t*>(sec1_arena.data() + off + nb * 4);
                for (size_t i = 0; i < nb; ++i) {
                    const uint8_t* blk = ptr + i * 34;
                    uint16_t h;
                    std::memcpy(&h, blk, 2);
                    sc[i] = fp16_to_f32(h);
                    std::memcpy(q + i * 32, blk + 2, 32);
                }
                TInfoT q_ti;
                q_ti.name = canon_name;
                q_ti.fmt = 103; // FMT_Q8_0
                q_ti.shape = {rows, cols};
                q_ti.off = off;
                q_ti.bytes = qb;
                m.trunk[canon_name] = std::move(q_ti);
                return true;
            }
            size_t nb = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(uint16_t);
            sec1_arena.resize(off + nb, 0);
            uint16_t* dst = reinterpret_cast<uint16_t*>(sec1_arena.data() + off);
            if (ti->type == 30) { // Already BF16!
                std::memcpy(dst, ptr, nb);
            } else {
                const size_t rb = gguf_detail::gguf_row_bytes(ti->type, cols);
                if (rb == 0) { err = "GGUF: unsupported dense type " + std::string(ti->type_name()) + " for " + gguf_name; return false; }
                std::vector<float> row(static_cast<size_t>(cols));
                for (int64_t r = 0; r < rows; ++r) {
                    if (!gguf_detail::gguf_dequant_row_f32(ti->type, ptr + static_cast<size_t>(r) * rb, row.data(), cols)) {
                        err = "GGUF: dequant failed for " + gguf_name;
                        return false;
                    }
                    uint16_t* out_r = dst + static_cast<size_t>(r) * cols;
                    for (int64_t c = 0; c < cols; ++c) out_r[c] = f32_to_bf16(row[c]);
                }
            }
            TInfoT out_ti;
            out_ti.name = canon_name;
            out_ti.fmt = 102; // F_BF16
            out_ti.shape = {rows, cols};
            out_ti.off = off;
            out_ti.bytes = nb;
            m.trunk[canon_name] = std::move(out_ti);
            return true;
        };

        // The engine reads the routers, the loop controllers, the sinks and the ShortConv tap projections in FP32
        // (tf32 and k_linear_f32), as the .strata file stores them.
        auto copy_mat_f32 = [&](const std::string& gguf_name, const std::string& canon_name, int64_t rows, int64_t cols) -> bool {
            std::vector<float> v;
            if (!read_tensor_f32(*model, gguf_name, v, err)) return false;
            if (v.size() != static_cast<size_t>(rows * cols)) { err = "GGUF: " + gguf_name + " has the wrong size"; return false; }
            add_f32(canon_name, v, {rows, cols});
            return true;
        };

        auto copy_vec_f32 = [&](const std::string& gguf_name, const std::string& canon_name, const std::vector<int64_t>& shape) -> bool {
            std::vector<float> v;
            if (!read_tensor_f32(*model, gguf_name, v, err)) return false;
            add_f32(canon_name, v, shape);
            return true;
        };

        auto nm = [&](const std::string& b, const char* pr_name) -> std::string {
            static const std::unordered_map<std::string, std::string> rw_map = {
                {"attn_r.weight", "attn_rel.weight"},
                {"attn_r_clip.weight", "attn_rel_clip.weight"},
                {"attn_rel_proj.weight", "attn_rel_bias.weight"},
                {"attn_res_score.weight", "attn_res_query.weight"},
                {"ffn_res_score.weight", "ffn_res_query.weight"},
                {"output_res_score.weight", "output_res_query.weight"},
                {"attn_q_in_norm.weight", "attn_q_inp_norm.weight"},
                {"attn_k_in_norm.weight", "attn_k_inp_norm.weight"},
                {"attn_v_in_norm.weight", "attn_v_inp_norm.weight"},
                {"shortconv_k.weight", "attn_k_conv.weight"},
                {"shortconv_v.weight", "attn_v_conv.weight"},
                {"shortconv_attn.weight", "attn_post_conv.weight"},
                {"shortconv_mlp.weight", "ffn_post_conv.weight"},
                {"shortconv_k_dyn.weight", "attn_k_conv_proj.weight"},
                {"shortconv_v_dyn.weight", "attn_v_conv_proj.weight"},
            };
            auto it = rw_map.find(pr_name);
            return b + ((is_rewrite_layout && it != rw_map.end()) ? it->second : std::string(pr_name));
        };

        const auto& c = m.c;
        const int D = c.hidden;
        const int q_dim = c.n_head * c.head_dim;
        const int kv_dim = c.n_kv * c.head_dim;

        if (!copy_mat_bf16("token_embd.weight", "model.llm.embed.weight", c.vocab_padded, D)) return false;
        if (!copy_vec_f32("token_embd_norm.weight", "model.llm.embed_norm.weight", {D})) return false;
        if (!copy_mat_bf16("output.weight", "model.llm.unembed.weight", c.vocab_padded, D)) return false;
        if (!copy_vec_f32("output_norm.weight", "model.llm.norm.weight", {D})) return false;
        if (!copy_vec_f32(nm("", "output_res_score.weight"), "model.llm.attn_res.output_query", {D})) return false;

        const int core_len = c.core_end - c.core_start;
        if (c.num_passes > 1 && core_len > 0) {
            if (model->find("loop_res_query.weight")) {
                if (!copy_vec_f32("loop_res_query.weight", "model.llm.middle_cycle_cross_pass_queries", {2 * core_len, D})) return false;
                if (!copy_vec_f32("loop_kv_res_query.weight", "model.llm.middle_cycle_cross_pass_kv_queries", {2 * core_len, D})) return false;
            } else {
                // Unsloth's Mainline_Support layout stores one [D, 4] message-query table per core layer
                // (the four messages feed the q, k, v and ffn controllers).  The engine's canonical
                // tables interleave them as q, k, v, ffn for each pass, two rows per core layer.
                std::vector<float> q(static_cast<size_t>(2 * core_len) * D), kv(q.size());
                for (int i = 0; i < core_len; ++i) {
                    const std::string n = "blk." + std::to_string(c.core_start + i) + ".loop_msg_query.weight";
                    if (model->find(n) == nullptr) {
                        err = "GGUF: neither loop_res_query.weight nor " + n;
                        return false;
                    }
                    std::vector<float> v;
                    if (!read_tensor_f32(*model, n, v, err)) return false;
                    if (static_cast<int>(v.size()) != 4 * D) {
                        err = "GGUF: " + n + " is not [D,4]";
                        return false;
                    }
                    for (int d = 0; d < D; ++d) {
                        q[(size_t)(2 * i) * D + d] = v[(size_t)0 * D + d];
                        q[(size_t)(2 * i + 1) * D + d] = v[(size_t)3 * D + d];
                        kv[(size_t)(2 * i) * D + d] = v[(size_t)1 * D + d];
                        kv[(size_t)(2 * i + 1) * D + d] = v[(size_t)2 * D + d];
                    }
                }
                add_f32("model.llm.middle_cycle_cross_pass_queries", q, {2 * core_len, D});
                add_f32("model.llm.middle_cycle_cross_pass_kv_queries", kv, {2 * core_len, D});
            }
        }

        // Helper: load a ShortConv weight [C, 4] (lag 3..0) into raw unfolded canonical convention
        auto load_raw_conv = [&](const std::string& gname, int C, bool dynamic, std::vector<float>& out) -> bool {
            if (!read_tensor_f32(*model, gname, out, err)) return false;
            if (!is_rewrite_layout) {
                // PR #32187 folded residual + output scale into the weight; unfold to match canonical .strata/HF!
                const float inv_sc = std::sqrt(static_cast<float>((dynamic ? 2 : 1) * c.sconv_k + 1));
                for (int ch = 0; ch < C; ++ch) {
                    for (int j = 0; j < c.sconv_k; ++j) out[static_cast<size_t>(ch) * c.sconv_k + j] *= inv_sc;
                    out[static_cast<size_t>(ch) * c.sconv_k + (c.sconv_k - 1)] -= 1.0f;
                }
            }
            return true;
        };

        for (int il = 0; il < c.n_phys; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            const std::string p = "model.llm.layers." + std::to_string(il) + ".";
            const bool is_swa = (c.is_swa[il] != 0);

            if (!copy_mat_bf16(b + "attn_q.weight", p + "attn.wq_du.weight", q_dim, D)) return false;
            if (!copy_mat_bf16(b + "attn_k.weight", p + "attn.wk_dv.weight", kv_dim, D)) return false;
            if (!copy_mat_bf16(b + "attn_v.weight", p + "attn.wv_dv.weight", kv_dim, D)) return false;
            if (!copy_mat_bf16(nm(b, "attn_r.weight"), p + "attn.wr_du.weight", c.n_head * c.d_rel, D)) return false;
            if (!copy_mat_f32(b + "attn_sink_proj.weight", p + "attn.attn_sink_proj.weight", c.n_head, D)) return false;
            if (!copy_mat_bf16(b + "attn_gate.weight", p + "attn.gate_attn.weight", q_dim, D)) return false;
            if (!copy_mat_bf16(b + "attn_output.weight", p + "attn.wo_ud.weight", D, q_dim)) return false;
            if (!copy_mat_f32(b + "ffn_gate_inp.weight", p + "mlp.gate.weight", c.n_experts, D)) return false;
            if (!copy_vec_f32(b + "exp_probs_b.bias", p + "mlp.gate.bias", {c.n_experts})) return false;
            if (!copy_vec_f32(b + "ffn_norm.weight", p + "mlp_norm.weight", {D})) return false;

            if (is_swa) {
                if (!copy_vec_f32(b + "attn_norm.weight", p + "attn_norm.weight", {D})) return false;
            } else {
                if (!copy_vec_f32(nm(b, "attn_q_in_norm.weight"), p + "q_input_norm.weight", {D})) return false;
                if (!copy_vec_f32(nm(b, "attn_k_in_norm.weight"), p + "k_input_norm.weight", {D})) return false;
                if (!copy_vec_f32(nm(b, "attn_v_in_norm.weight"), p + "v_input_norm.weight", {D})) return false;
            }

            // q_l2_clip.scale may be [head_dim] or [n_head, head_dim]
            {
                std::vector<float> qc;
                if (!read_tensor_f32(*model, b + "attn_q_clip.weight", qc, err)) return false;
                if (static_cast<int>(qc.size()) == c.head_dim) {
                    add_f32(p + "attn.q_l2_clip.scale", qc, {c.head_dim});
                } else {
                    add_f32(p + "attn.q_l2_clip.scale", qc, {c.n_head, c.head_dim});
                }
            }
            {
                std::vector<float> rc;
                if (!read_tensor_f32(*model, nm(b, "attn_r_clip.weight"), rc, err)) return false;
                if (static_cast<int>(rc.size()) == c.d_rel) {
                    add_f32(p + "attn.rel_l2_clip.scale", rc, {c.d_rel});
                } else {
                    add_f32(p + "attn.rel_l2_clip.scale", rc, {c.n_head, c.d_rel});
                }
            }

            // rel_logits_proj.proj: canonical is [d_rel, extent]; master stores [extent, d_rel]!
            {
                std::vector<float> rp;
                if (!read_tensor_f32(*model, nm(b, "attn_rel_proj.weight"), rp, err)) return false;
                if (is_rewrite_layout) {
                    std::vector<float> tr(rp.size());
                    for (int e = 0; e < c.rel_extent; ++e)
                        for (int r = 0; r < c.d_rel; ++r)
                            tr[static_cast<size_t>(r) * c.rel_extent + e] = rp[static_cast<size_t>(e) * c.d_rel + r];
                    add_f32(p + "attn.rel_logits_proj.proj", tr, {c.d_rel, c.rel_extent});
                } else {
                    add_f32(p + "attn.rel_logits_proj.proj", rp, {c.d_rel, c.rel_extent});
                }
            }

            // ShortConv weights
            {
                std::vector<float> ac, mc;
                if (!load_raw_conv(nm(b, "shortconv_attn.weight"), D, false, ac)) return false;
                if (!load_raw_conv(nm(b, "shortconv_mlp.weight"), D, false, mc)) return false;
                add_f32(p + "attn_sconv.weight", ac, {D, c.sconv_k});
                add_f32(p + "mlp_sconv.weight", mc, {D, c.sconv_k});
                if (is_swa) {
                    std::vector<float> kc, vc;
                    if (!load_raw_conv(nm(b, "shortconv_k.weight"), kv_dim, false, kc)) return false;
                    if (!load_raw_conv(nm(b, "shortconv_v.weight"), kv_dim, false, vc)) return false;
                    add_f32(p + "attn.k_sconv.weight", kc, {kv_dim, c.sconv_k});
                    add_f32(p + "attn.v_sconv.weight", vc, {kv_dim, c.sconv_k});
                } else {
                    // Canonical global static_w is [4, kv_dim] lag 0..3; GGUF stores [kv_dim, 4] lag 3..0!
                    std::vector<float> kc, vc;
                    if (!load_raw_conv(nm(b, "shortconv_k.weight"), kv_dim, true, kc)) return false;
                    if (!load_raw_conv(nm(b, "shortconv_v.weight"), kv_dim, true, vc)) return false;
                    std::vector<float> ks(static_cast<size_t>(c.sconv_k) * kv_dim), vs(static_cast<size_t>(c.sconv_k) * kv_dim);
                    for (int ch = 0; ch < kv_dim; ++ch) {
                        for (int lag = 0; lag < c.sconv_k; ++lag) {
                            ks[static_cast<size_t>(lag) * kv_dim + ch] = kc[static_cast<size_t>(ch) * c.sconv_k + (c.sconv_k - 1 - lag)];
                            vs[static_cast<size_t>(lag) * kv_dim + ch] = vc[static_cast<size_t>(ch) * c.sconv_k + (c.sconv_k - 1 - lag)];
                        }
                    }
                    add_f32(p + "attn.k_sconv.static_w", ks, {c.sconv_k, kv_dim});
                    add_f32(p + "attn.v_sconv.static_w", vs, {c.sconv_k, kv_dim});

                    // Dynamic tap projections [n_kv*4, D]
                    std::vector<float> kd, vd;
                    if (!read_tensor_f32(*model, nm(b, "shortconv_k_dyn.weight"), kd, err)) return false;
                    if (!read_tensor_f32(*model, nm(b, "shortconv_v_dyn.weight"), vd, err)) return false;
                    if (!is_rewrite_layout) {
                        // PR #32187 pre-multiplied dyn_proj by 1/sqrt(9) = 1/3; canonical unfolds by * 3.0!
                        for (float& x : kd) x *= 3.0f;
                        for (float& x : vd) x *= 3.0f;
                    }
                    add_f32(p + "attn.k_sconv.weight_proj", kd, {c.n_kv * c.sconv_k, D});
                    add_f32(p + "attn.v_sconv.weight_proj", vd, {c.n_kv * c.sconv_k, D});
                }
            }

            if (il > 0) {
                const int nq = is_swa ? 1 : 3;
                if (!copy_vec_f32(nm(b, "attn_res_score.weight"),
                                  "model.llm.attn_res.attn_queries." + std::to_string(il - 1), {nq, D})) return false;
            }
            if (!copy_vec_f32(nm(b, "ffn_res_score.weight"),
                              "model.llm.attn_res.mlp_queries." + std::to_string(il), {D})) return false;

            if (c.num_passes > 1 && il >= c.core_start && il < c.core_end) {
                static const char* mods_old[4] = {"attn", "k", "v", "mlp"};
                static const char* mods_new[4] = {"q", "k", "v", "ffn"};
                const char** mods = is_rewrite_layout ? mods_new : mods_old;
                const std::string ctrl_sep =
                    model->find(b + "loop_" + mods[0] + "_ctrl_down.weight") ? "_ctrl_" : "_";
                for (int mi = 0; mi < 4; ++mi) {
                    const std::string gp = b + "loop_" + mods[mi] + ctrl_sep;
                    const std::string cp = p + "controllers." + mods_old[mi] + ".";
                    if (!copy_mat_f32(gp + "down.weight", cp + "down_weight", c.gate_rank, 2 * D)) return false;
                    if (!copy_mat_f32(gp + "gain.weight", cp + "gain_weight", D, c.gate_rank)) return false;
                    if (!copy_mat_f32(gp + "forget.weight", cp + "forget_weight", D, c.gate_rank)) return false;
                }
            }
        }

        // 3. Optional external DSpark drafter (dspark-*.gguf; the safetensors adapter is not taken)
        if (!explicit_dspark.empty()) {
            if (gguf_detail::has_gguf_magic(explicit_dspark)) {
                try {
                    dspark_gguf = std::make_unique<strata::GgufModel>(strata::GgufModel::open(explicit_dspark));
                } catch (const std::exception& e) {
                    err = std::string("dspark GGUF open: ") + e.what();
                    return false;
                }
                // Map all 54 tensors from Unsloth/llama.cpp dspark-*.gguf (dflash arch) to canonical DSpark names
                auto map_dspark_name = [](const std::string& n) -> std::string {
                    if (n == "dspark_fuse.weight" || n == "dspark.fuse_weight" || n == "fuse_weight") return "fuse_weight";
                    if (n == "mask_embd.weight" || n == "dspark.mask_embed" || n == "mask_embed") return "mask_embed";
                    if (n == "output_norm.weight" || n == "dspark.output_norm.weight" || n == "norm.weight") return "norm.weight";
                    if (n == "dspark_norm.weight") return "markov_head.dflare_norm.weight";
                    if (n == "dspark_hidden_out.weight") return "markov_head.hidden_out.weight";
                    if (n == "dspark_hidden_out.bias") return "markov_head.hidden_out.bias";
                    if (n == "markov_w1.weight") return "markov_head.markov_w1.weight";
                    if (n == "joint_proj.weight") return "markov_head.rnn_proj.weight";
                    if (n == "joint_proj.bias") return "markov_head.rnn_proj.bias";
                    if (n.rfind("blk.", 0) == 0) {
                        size_t dot = n.find('.', 4);
                        if (dot != std::string::npos) {
                            std::string idx = n.substr(4, dot - 4);
                            std::string rest = n.substr(dot + 1);
                            std::string lp = "layers." + idx + ".";
                            if (rest == "dflare_norm.weight") return lp + "dflare_norm.weight";
                            if (rest == "attn_norm.weight") return lp + "input_layernorm.weight";
                            if (rest == "ffn_norm.weight") return lp + "post_attention_layernorm.weight";
                            if (rest == "ffn_down.weight") return lp + "mlp.down_proj.weight";
                            if (rest == "ffn_gate.weight") return lp + "mlp.gate_proj.weight";
                            if (rest == "ffn_up.weight") return lp + "mlp.up_proj.weight";
                            if (rest == "attn_q.weight") return lp + "self_attn.q_proj.weight";
                            if (rest == "attn_k.weight") return lp + "self_attn.k_proj.weight";
                            if (rest == "attn_v.weight") return lp + "self_attn.v_proj.weight";
                            if (rest == "attn_k_ctx.weight") return lp + "self_attn.k_proj_ctx.weight";
                            if (rest == "attn_v_ctx.weight") return lp + "self_attn.v_proj_ctx.weight";
                            if (rest == "attn_output.weight") return lp + "self_attn.o_proj.weight";
                            if (rest == "attn_q_norm.weight") return lp + "self_attn.q_norm.weight";
                            if (rest == "attn_k_norm.weight") return lp + "self_attn.k_norm.weight";
                            if (rest == "attn_k_ctx_norm.weight") return lp + "self_attn.k_norm_ctx.weight";
                        }
                    }
                    if (n.rfind("dspark.", 0) == 0) return n.substr(7);
                    return n;
                };
                for (size_t si = 0; si < dspark_gguf->size(); ++si) {
                    for (const auto& t : dspark_gguf->shard(si).tensors()) {
                        std::vector<float> v;
                        if (!read_tensor_f32(*dspark_gguf, t.name, v, err)) return false;
                        std::string cn = map_dspark_name(t.name);
                        std::vector<int64_t> sh;
                        for (auto it = t.shape.rbegin(); it != t.shape.rend(); ++it) sh.push_back(static_cast<int64_t>(*it));
                        size_t off = align64(sec1_arena.size());
                        size_t nb = v.size() * sizeof(uint16_t);
                        bool keep_f32 = (sh.size() <= 1 || cn == "fuse_weight" || cn.find("norm") != std::string::npos || cn.find("bias") != std::string::npos);
                        TInfoT ti;
                        ti.name = cn;
                        ti.shape = sh;
                        ti.off = off;
                        if (keep_f32) {
                            nb = v.size() * sizeof(float);
                            sec1_arena.resize(off + nb, 0);
                            std::memcpy(sec1_arena.data() + off, v.data(), nb);
                            ti.fmt = 101;
                        } else {
                            sec1_arena.resize(off + nb, 0);
                            uint16_t* dst = reinterpret_cast<uint16_t*>(sec1_arena.data() + off);
                            for (size_t k = 0; k < v.size(); ++k) dst[k] = f32_to_bf16(v[k]);
                            ti.fmt = 102;
                        }
                        ti.bytes = nb;
                        m.dspark_tensors[cn] = std::move(ti);
                    }
                }
                m.dspark.present = !m.dspark_tensors.empty();
            }
        }

        // 4. Index routed experts (NVFP4 -> lossless SF4 code_id=15; Q8_0 -> lossless GGUF_RAW code_id=20; other types
        //    dequantized when read and written as GGUF_RAW Q8_0)
        layers.assign(c.n_phys, LayerExpertMeta{});
        expert_dir.clear();
        expert_dir.reserve(static_cast<size_t>(c.n_phys) * c.n_experts);
        dir_index_of.assign(static_cast<size_t>(c.n_phys) * c.n_experts, -1);

        uint64_t virt_off = 0;
        for (int il = 0; il < c.n_phys; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            auto& lm = layers[il];
            lm.hidden = D;
            lm.inter = c.inter;
            lm.is_fused_gu = is_rewrite_layout;

            const strata::TensorInfo* ti_gu = nullptr;
            const strata::TensorInfo* ti_up = nullptr;
            const strata::TensorInfo* ti_dn = nullptr;
            if (is_rewrite_layout) {
                lm.gu_base = tensor_ptr(*model, b + "ffn_gate_up_exps.weight", ti_gu);
                if (!ti_gu || !lm.gu_base) { err = "GGUF: missing " + b + "ffn_gate_up_exps.weight"; return false; }
                lm.gu_type = ti_gu->type;
                lm.gu_stride = static_cast<size_t>(strata::tensor_payload_bytes(*ti_gu) / static_cast<uint64_t>(c.n_experts));
            } else {
                lm.gu_base = tensor_ptr(*model, b + "ffn_gate_exps.weight", ti_gu);
                lm.up_base = tensor_ptr(*model, b + "ffn_up_exps.weight", ti_up);
                if (!ti_gu || !lm.gu_base || !ti_up || !lm.up_base) {
                    err = "GGUF: missing gate/up expert tensors in layer " + std::to_string(il);
                    return false;
                }
                lm.gu_type = ti_gu->type;
                lm.gu_stride = static_cast<size_t>(strata::tensor_payload_bytes(*ti_gu) / static_cast<uint64_t>(c.n_experts));
                lm.up_stride = static_cast<size_t>(strata::tensor_payload_bytes(*ti_up) / static_cast<uint64_t>(c.n_experts));
            }
            lm.dn_base = tensor_ptr(*model, b + "ffn_down_exps.weight", ti_dn);
            if (!ti_dn || !lm.dn_base) { err = "GGUF: missing " + b + "ffn_down_exps.weight"; return false; }
            lm.dn_type = ti_dn->type;
            lm.dn_stride = static_cast<size_t>(strata::tensor_payload_bytes(*ti_dn) / static_cast<uint64_t>(c.n_experts));

            // Optional per-expert scales (mandatory for NVFP4, 1.0f if absent for requantized types)
            auto load_scale = [&](const std::string& sn, bool required, std::vector<float>& out) -> bool {
                if (model->find(sn)) return read_tensor_f32(*model, sn, out, err);
                if (required) { err = "GGUF: NVFP4 expert missing required scale " + sn; return false; }
                out.assign(c.n_experts, 1.0f);
                return true;
            };
            if (is_rewrite_layout) {
                if (!load_scale(b + "ffn_gate_up_exps.scale", lm.gu_type == 40, lm.gate_scale)) return false;
                lm.up_scale = lm.gate_scale;
            } else {
                if (!load_scale(b + "ffn_gate_exps.scale", lm.gu_type == 40, lm.gate_scale)) return false;
                if (!load_scale(b + "ffn_up_exps.scale",   lm.gu_type == 40, lm.up_scale)) return false;
            }
            if (!load_scale(b + "ffn_down_exps.scale", lm.dn_type == 40, lm.dn_scale)) return false;

            // Check if NVFP4 -> bit-exact SF4 repack applies!
            const bool same_gu_scale = std::equal(lm.gate_scale.begin(), lm.gate_scale.end(), lm.up_scale.begin());
            lm.is_nvfp4_to_sf4 = (lm.gu_type == 40 && lm.dn_type == 40 && (is_rewrite_layout || (ti_up && ti_up->type == 40 && same_gu_scale)));

            uint8_t code = 0;
            if (lm.is_nvfp4_to_sf4) {
                code = static_cast<uint8_t>(strata::quant::sk::SkCode::SF4);
                const size_t w13_p = static_cast<size_t>(2 * c.inter) * (D / 2);
                const size_t w13_s = static_cast<size_t>(2 * c.inter) * (D / 16);
                const size_t w2_p  = static_cast<size_t>(D) * (c.inter / 2);
                const size_t w2_s  = static_cast<size_t>(D) * (c.inter / 16);
                lm.dn_offset = static_cast<uint32_t>(16u + w13_p + w13_s);
                lm.body_bytes = static_cast<uint32_t>(lm.dn_offset + w2_p + w2_s);
            } else {
                // The GPU's expert kernels read a raw GGUF body in Q8_0 (UD-Q4_K_XL's layers 20..23); any other
                // type (Q5_K, BF16, the i-quants, a layer that mixes NVFP4 with them) is dequantized when read and
                // written as Q8_0: 8.5 bits a weight whatever the file's size.
                lm.up_type = ti_up ? ti_up->type : lm.gu_type;
                lm.requant_q8_0 = lm.gu_type != 8 || lm.dn_type != 8 || lm.up_type != 8;
                if (lm.requant_q8_0) {
                    std::vector<float> probe(256);
                    const std::vector<uint8_t> zero(1024, 0);
                    for (uint32_t t : {lm.gu_type, lm.up_type, lm.dn_type}) {
                        int be = 0, bb = 0;
                        if (!strata::block_geometry(t, be, bb) || be > 256 ||
                            !gguf_detail::gguf_dequant_row_f32(t, zero.data(), probe.data(), be)) {
                            err = "GGUF: layer " + std::to_string(il) + "'s experts are of type " + std::to_string(t) +
                                  ", which this build cannot read";
                            return false;
                        }
                    }
                }
                code = static_cast<uint8_t>(strata::quant::sk::SkCode::GGUF_RAW);
                const size_t F = static_cast<size_t>(c.inter);
                const size_t gu_total = lm.requant_q8_0 ? 2 * F * (static_cast<size_t>(D) / 32 * 34)
                                      : is_rewrite_layout ? lm.gu_stride : (lm.gu_stride + lm.up_stride);
                const size_t dn_total = lm.requant_q8_0 ? static_cast<size_t>(D) * (F / 32 * 34) : lm.dn_stride;
                lm.dn_offset = static_cast<uint32_t>((32u + gu_total + 31u) & ~31u);
                lm.body_bytes = static_cast<uint32_t>((lm.dn_offset + dn_total + 31u) & ~31u);
            }

            for (int e = 0; e < c.n_experts; ++e) {
                strata::artifact::StrataExpertEntry ent{};
                ent.expert_id   = static_cast<uint16_t>(e);
                ent.gu_code_id  = code;
                ent.dn_code_id  = code;
                ent.body_offset = virt_off;
                ent.body_bytes  = lm.body_bytes;
                ent.dn_offset   = lm.dn_offset;
                int idx = static_cast<int>(expert_dir.size());
                expert_dir.push_back(ent);
                dir_index_of[il * c.n_experts + e] = idx;
                virt_off += lm.body_bytes;
            }
        }

        // 5. 3x loop-weighted vram_load_order
        vram_load_order.clear();
        vram_load_order.reserve(expert_dir.size());
        for (int il = c.core_start; il < c.core_end; ++il)
            for (int e = 0; e < c.n_experts; ++e)
                vram_load_order.push_back(static_cast<uint16_t>(il * c.n_experts + e));
        for (int il = 0; il < c.core_start; ++il)
            for (int e = 0; e < c.n_experts; ++e)
                vram_load_order.push_back(static_cast<uint16_t>(il * c.n_experts + e));
        for (int il = c.core_end; il < c.n_phys; ++il)
            for (int e = 0; e < c.n_experts; ++e)
                vram_load_order.push_back(static_cast<uint16_t>(il * c.n_experts + e));

        // 6. Extract embedded tokenizer + chat_template if requested
        if (keep_tokenizer) {
            if (!gguf_detail::extract_gguf_tokenizer_json(model->shard(0).path(), m.tokenizer_json, m.chat_template, err)) {
                return false;
            }
        } else if (const auto* ct = meta.get("tokenizer.chat_template")) {
            m.chat_template = ct->s;
        }
        return true;
    }

    bool read_section1(std::vector<uint8_t>& out, std::string&) const {
        out = sec1_arena;
        return true;
    }

    // One expert of a layer whose types the kernels do not read, as an interleaved Q8_0 GGUF_RAW body: each row
    // dequantized (its per-expert scale applied, so the header's are 1) and quantized again as ggml's Q8_0 is.
    static bool requant_expert_q8_0(const LayerExpertMeta& lm, int e, strata::quant::sk::GgufExpertHeader hdr,
                                    std::vector<uint8_t>& out, std::string& err) {
        const size_t D = static_cast<size_t>(lm.hidden), F = static_cast<size_t>(lm.inter);
        hdr.gu_type = hdr.dn_type = 8;
        hdr.gate_scale = hdr.up_scale = hdr.dn_scale = 1.0f;
        std::memcpy(out.data(), &hdr, sizeof(hdr));
        const size_t g_rb = gguf_detail::gguf_row_bytes(lm.gu_type, static_cast<int64_t>(D));
        const size_t u_rb = gguf_detail::gguf_row_bytes(lm.up_type, static_cast<int64_t>(D));
        const size_t d_rb = gguf_detail::gguf_row_bytes(lm.dn_type, static_cast<int64_t>(F));
        const uint8_t* g_src = lm.gu_base + static_cast<size_t>(e) * lm.gu_stride;
        const uint8_t* u_src = lm.is_fused_gu ? g_src + F * g_rb : lm.up_base + static_cast<size_t>(e) * lm.up_stride;
        const uint8_t* d_src = lm.dn_base + static_cast<size_t>(e) * lm.dn_stride;
        std::vector<float> row(std::max(D, F));
        auto put = [&](uint32_t t, const uint8_t* src, size_t n, float scale, uint8_t* dst) -> bool {
            if (!gguf_detail::gguf_dequant_row_f32(t, src, row.data(), static_cast<int64_t>(n))) return false;
            for (size_t b = 0; b < n / 32; ++b, dst += 34) {
                float amax = 0.0f;
                for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(row[b * 32 + j] * scale));
                const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
                const uint16_t h = f32_to_fp16(d);
                std::memcpy(dst, &h, 2);
                for (size_t j = 0; j < 32; ++j)
                    dst[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(std::lround(row[b * 32 + j] * scale * id)));
            }
            return true;
        };
        const size_t gq = D / 32 * 34, dq = F / 32 * 34;
        for (size_t j = 0; j < F; ++j)
            if (!put(lm.gu_type, g_src + j * g_rb, D, lm.gate_scale[e], out.data() + 32 + (2 * j) * gq) ||
                !put(lm.up_type, u_src + j * u_rb, D, lm.up_scale[e], out.data() + 32 + (2 * j + 1) * gq)) {
                err = "GGUF: an expert row did not dequantize";
                return false;
            }
        for (size_t r = 0; r < D; ++r)
            if (!put(lm.dn_type, d_src + r * d_rb, F, lm.dn_scale[e], out.data() + lm.dn_offset + r * dq)) {
                err = "GGUF: an expert row did not dequantize";
                return false;
            }
        return true;
    }

    bool read_expert_body(size_t dir_index, int n_experts, std::vector<uint8_t>& out, std::string& err) const {
        if (dir_index >= expert_dir.size()) {
            err = "GGUF: expert dir_index out of bounds";
            return false;
        }
        const int il = static_cast<int>(dir_index) / n_experts;
        const int e  = static_cast<int>(dir_index) % n_experts;
        const auto& lm = layers[il];
        out.assign(lm.body_bytes, 0);

        if (lm.is_nvfp4_to_sf4) {
            // Zero-requantization bit-exact repack GGUF NVFP4 -> canonical .strata v1.1 SF4 (interleaved even=gate, odd=up)
            float hdr[4] = {lm.gate_scale[e], 1.0f, lm.dn_scale[e], 1.0f};
            std::memcpy(out.data(), hdr, 16);
            const size_t D = static_cast<size_t>(lm.hidden);
            const size_t F = static_cast<size_t>(lm.inter);
            const size_t src_gu_row_b = (D / 64) * 36;
            const size_t src_dn_row_b = (F / 64) * 36;
            const uint8_t* g_src = lm.gu_base + static_cast<size_t>(e) * lm.gu_stride;
            const uint8_t* u_src = lm.is_fused_gu
                ? (g_src + F * src_gu_row_b)
                : (lm.up_base + static_cast<size_t>(e) * lm.up_stride);
            const uint8_t* d_src = lm.dn_base + static_cast<size_t>(e) * lm.dn_stride;

            uint8_t* w13_p = out.data() + 16;
            uint8_t* w13_s = w13_p + (2 * F) * (D / 2);
            for (size_t j = 0; j < F; ++j) {
                gguf_detail::repack_gguf_nvfp4_row_to_sf4(
                    g_src + j * src_gu_row_b,
                    w13_p + (2 * j + 0) * (D / 2),
                    w13_s + (2 * j + 0) * (D / 16),
                    static_cast<int64_t>(D));
                gguf_detail::repack_gguf_nvfp4_row_to_sf4(
                    u_src + j * src_gu_row_b,
                    w13_p + (2 * j + 1) * (D / 2),
                    w13_s + (2 * j + 1) * (D / 16),
                    static_cast<int64_t>(D));
            }

            uint8_t* w2_p = out.data() + lm.dn_offset;
            uint8_t* w2_s = w2_p + D * (F / 2);
            for (size_t r = 0; r < D; ++r) {
                gguf_detail::repack_gguf_nvfp4_row_to_sf4(
                    d_src + r * src_dn_row_b,
                    w2_p + r * (F / 2),
                    w2_s + r * (F / 16),
                    static_cast<int64_t>(F));
            }
            return true;
        }

        // Lossless native GGUF_RAW expert body (Q8_0, the only raw type the kernels read; the open refuses others)
        strata::quant::sk::GgufExpertHeader hdr{};
        hdr.magic = 0x47475546u;
        hdr.gu_type = lm.gu_type;
        hdr.dn_type = lm.dn_type;
        hdr.flags = 1u;  // interleaved: row 2j = gate j, row 2j+1 = up j, as the kernels read SF4
        hdr.gate_scale = lm.gate_scale[e];
        hdr.up_scale   = lm.up_scale[e];
        hdr.dn_scale   = lm.dn_scale[e];
        hdr.reserved   = 0u;
        std::memcpy(out.data(), &hdr, sizeof(hdr));

        const size_t F = static_cast<size_t>(lm.inter);
        if (lm.requant_q8_0) return requant_expert_q8_0(lm, e, hdr, out, err);
        const size_t row_b = lm.gu_stride / (lm.is_fused_gu ? 2 * F : F);
        const uint8_t* g_src = lm.gu_base + static_cast<size_t>(e) * lm.gu_stride;
        const uint8_t* u_src = lm.is_fused_gu ? g_src + F * row_b : lm.up_base + static_cast<size_t>(e) * lm.up_stride;
        for (size_t j = 0; j < F; ++j) {
            std::memcpy(out.data() + 32 + (2 * j) * row_b, g_src + j * row_b, row_b);
            std::memcpy(out.data() + 32 + (2 * j + 1) * row_b, u_src + j * row_b, row_b);
        }
        std::memcpy(out.data() + lm.dn_offset, lm.dn_base + static_cast<size_t>(e) * lm.dn_stride, lm.dn_stride);
        return true;
    }
};

}  // namespace ink2
