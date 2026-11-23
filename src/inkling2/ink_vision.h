// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/ink_vision.h - Lossless Inkling-2-Mini-Vision.strata (23L ViT, patch=20, 2x2 merge, axial 2D RoPE, LayerScale)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "ink_batch.h"
#include "ink_json.h"
#include "strata/artifact/strata_reader.hpp"

#ifdef INKLING_DEFINE_STBI
#define STB_IMAGE_IMPLEMENTATION
#endif
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#include "stb_image.h"

namespace ink2 {

struct VitCfg {
    int hidden = 1024;
    int inter = 4096;
    int layers = 23;
    int heads = 16;
    int head_dim = 64;
    int patch = 20;
    int merge = 2;
    int posemb_grid = 32;
    int proj_hidden = 4096;
    int out_hidden = 2304;
    float ln_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int image_token = 200054;
    int min_side = 40;
    int max_side = 4096;
    int64_t max_pixels = 4096LL * 4096LL;
};

struct VitTensor {
    int fmt = F_BF16;
    std::vector<int64_t> shape;
    uint64_t off = 0;
    uint64_t bytes = 0;
};

struct VisionShardInfo {
    VitCfg c;
    std::unordered_map<std::string, VitTensor> T;
    uint64_t sec1_off = 0;
    uint64_t sec1_len = 0;
    std::string path;

    const VitTensor* find(const std::string& n) const {
        auto it = T.find(n);
        return it == T.end() ? nullptr : &it->second;
    }
};

struct ImageRec {
    int orig_w = 0, orig_h = 0;
    int padded_w = 0, padded_h = 0;   // multiples of patch*merge (40)
    int grid_h = 0, grid_w = 0;       // in merged LM tokens (= padded_h/40, padded_w/40)
    // Normalized float pixels in [-1, 1], shape [grid_h * grid_w, 40, 40, 3]
    std::vector<float> pixel_values;
    std::string file;

    int num_lm_tokens() const noexcept { return grid_h * grid_w; }
};

inline std::string pct_decode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() &&
            std::isxdigit(static_cast<unsigned char>(s[i + 1])) &&
            std::isxdigit(static_cast<unsigned char>(s[i + 2]))) {
            o += static_cast<char>(std::strtol(s.substr(i + 1, 2).c_str(), nullptr, 16));
            i += 2;
        } else {
            o += s[i];
        }
    }
    return o;
}

// Probe Inkling-2-Mini-Vision.strata without loading weights into RAM/VRAM (XeStrata reads the .strata vision shard
// alone; upstream's HF safetensors source is not taken)
inline bool vision_probe(const std::string& path, VisionShardInfo& vi, std::string& err) {
    vi = VisionShardInfo{};
    strata::artifact::StrataFileReader rd;
    if (!rd.open(path, err)) return false;
    if (rd.header().profile != strata::artifact::StrataProfile::VisionShard) {
        err = "not a Vision shard (profile " + std::to_string(static_cast<int>(rd.header().profile)) + ")";
        return false;
    }
    try {
        JVal m = json_parse(rd.manifest_json());
        if (const JVal* vc = m.get("vision_config")) {
            VitCfg& c = vi.c;
            if (const JVal* v = vc->get("hidden_size")) c.hidden = static_cast<int>(v->i());
            if (const JVal* v = vc->get("num_hidden_layers")) c.layers = static_cast<int>(v->i());
            if (const JVal* v = vc->get("num_attention_heads")) c.heads = static_cast<int>(v->i());
            if (const JVal* v = vc->get("native_patch_size")) c.patch = static_cast<int>(v->i());
            if (const JVal* v = vc->get("patch_size")) c.patch = static_cast<int>(v->i());
            if (const JVal* v = vc->get("adapter_merge_size")) c.merge = static_cast<int>(v->i());
            if (const JVal* v = vc->get("spatial_merge_size")) c.merge = static_cast<int>(v->i());
            if (const JVal* v = vc->get("posemb_grid_size")) c.posemb_grid = static_cast<int>(v->i());
            if (const JVal* v = vc->get("projection_hidden_dim")) c.proj_hidden = static_cast<int>(v->i());
            if (const JVal* v = vc->get("text_hidden_size")) c.out_hidden = static_cast<int>(v->i());
            if (const JVal* v = vc->get("layer_norm_eps")) c.ln_eps = static_cast<float>(v->f());
            if (const JVal* v = vc->get("mlp_ratio")) c.inter = static_cast<int>(c.hidden * v->f());
            c.head_dim = c.hidden / c.heads;
        }
        for (const JVal& t : m.at("tensors").a) {
            VitTensor vt;
            vt.fmt = fmt_from_str(t.at("dtype").str());
            if (vt.fmt < 0) {
                err = "unsupported vision tensor dtype " + t.at("dtype").str();
                return false;
            }
            for (const JVal& s : t.at("shape").a) vt.shape.push_back(s.i());
            vt.off = static_cast<uint64_t>(t.at("offset").i());
            vt.bytes = static_cast<uint64_t>(t.at("bytes").i());
            vi.T[t.at("name").str()] = std::move(vt);
        }
    } catch (const std::exception& e) {
        err = std::string("vision manifest: ") + e.what();
        return false;
    }
    vi.sec1_off = rd.header().sections[1].file_offset;
    vi.sec1_len = rd.header().sections[1].byte_length;
    vi.path = path;
    const char* req[] = {
        "model.visual.backbone.embeddings.patch_embedding.weight",
        "model.visual.backbone.embeddings.position_embedding",
        "model.visual.backbone.embeddings.class_embedding",
        "model.visual.backbone.embeddings.cls_position_embedding",
        "model.visual.backbone.ln_pre.weight",
        "model.visual.proj_in.weight",
        "model.visual.proj_out.weight",
    };
    for (const char* k : req) {
        if (!vi.T.count(k)) {
            err = std::string("vision shard missing tensor ") + k;
            return false;
        }
    }
    return true;
}

// Decode PNG/JPEG/BMP/GIF, composite RGBA onto Inkling2 8x8 checkerboard (204/153), clamp to [40..max_side],
// pad to multiple of 40 with fill=-1.0 (0..255 scale), normalize (x/255 - 0.5)/0.5, and tile into [grid_h*grid_w, 40, 40, 3].
inline bool prepare_picture(const std::string& path, const VitCfg& cfg, ImageRec& rec, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open image " + path;
        return false;
    }
    std::vector<uint8_t> buf;
    uint8_t tmp[65536];
    size_t nr = 0;
    while ((nr = std::fread(tmp, 1, sizeof(tmp), f)) > 0) buf.insert(buf.end(), tmp, tmp + nr);
    std::fclose(f);

    int w = 0, h = 0, comp = 0;
    uint8_t* rgba = stbi_load_from_memory(buf.data(), static_cast<int>(buf.size()), &w, &h, &comp, 4);
    if (!rgba) {
        err = std::string("cannot decode image ") + path + ": " + stbi_failure_reason();
        return false;
    }
    if (w < 1 || h < 1) {
        stbi_image_free(rgba);
        err = "empty image dimensions";
        return false;
    }

    // 1. RGBA -> RGB using Inkling2ImageProcessor's exact 8x8 checkerboard (204 / 153)
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i4 = (static_cast<size_t>(y) * w + x) * 4;
            const size_t i3 = (static_cast<size_t>(y) * w + x) * 3;
            const int a = rgba[i4 + 3];
            if (a == 255) {
                rgb[i3 + 0] = rgba[i4 + 0];
                rgb[i3 + 1] = rgba[i4 + 1];
                rgb[i3 + 2] = rgba[i4 + 2];
            } else {
                const int bg = (((y / 8) + (x / 8)) % 2 == 0) ? 204 : 153;
                for (int c = 0; c < 3; ++c) {
                    rgb[i3 + c] = static_cast<uint8_t>((rgba[i4 + c] * a + bg * (255 - a) + 127) / 255);
                }
            }
        }
    }
    stbi_image_free(rgba);

    // 2. Downscale proportionally only if exceeding max_side (4096) or max_pixels
    int sw = w, sh = h;
    if (sw > cfg.max_side || sh > cfg.max_side || static_cast<int64_t>(sw) * sh > cfg.max_pixels) {
        double s = 1.0;
        if (sw > cfg.max_side) s = std::min(s, static_cast<double>(cfg.max_side) / sw);
        if (sh > cfg.max_side) s = std::min(s, static_cast<double>(cfg.max_side) / sh);
        if (static_cast<int64_t>(sw) * sh > cfg.max_pixels) {
            s = std::min(s, std::sqrt(static_cast<double>(cfg.max_pixels) / (static_cast<double>(sw) * sh)));
        }
        int nw = std::max(1, static_cast<int>(std::floor(sw * s)));
        int nh = std::max(1, static_cast<int>(std::floor(sh * s)));
        std::vector<uint8_t> resized(static_cast<size_t>(nw) * nh * 3);
        for (int y = 0; y < nh; ++y) {
            float fy = (y + 0.5f) * (static_cast<float>(sh) / nh) - 0.5f;
            int y0 = std::max(0, std::min(sh - 1, static_cast<int>(std::floor(fy))));
            int y1 = std::min(sh - 1, y0 + 1);
            float wy = std::max(0.0f, std::min(1.0f, fy - y0));
            for (int x = 0; x < nw; ++x) {
                float fx = (x + 0.5f) * (static_cast<float>(sw) / nw) - 0.5f;
                int x0 = std::max(0, std::min(sw - 1, static_cast<int>(std::floor(fx))));
                int x1 = std::min(sw - 1, x0 + 1);
                float wx = std::max(0.0f, std::min(1.0f, fx - x0));
                for (int c = 0; c < 3; ++c) {
                    float v00 = rgb[(static_cast<size_t>(y0) * sw + x0) * 3 + c];
                    float v01 = rgb[(static_cast<size_t>(y0) * sw + x1) * 3 + c];
                    float v10 = rgb[(static_cast<size_t>(y1) * sw + x0) * 3 + c];
                    float v11 = rgb[(static_cast<size_t>(y1) * sw + x1) * 3 + c];
                    float v = (v00 * (1 - wx) + v01 * wx) * (1 - wy) + (v10 * (1 - wx) + v11 * wx) * wy;
                    resized[(static_cast<size_t>(y) * nw + x) * 3 + c] =
                        static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, std::round(v))));
                }
            }
        }
        rgb.swap(resized);
        sw = nw;
        sh = nh;
    }

    // 3. Pad to multiple of cell = patch * merge (40) with fill_value = -1.0 on 0..255 scale, then (x/255 - 0.5)/0.5
    const int cell = cfg.patch * cfg.merge;  // 40
    const int rows = std::max(1, (sh + cell - 1) / cell);
    const int cols = std::max(1, (sw + cell - 1) / cell);
    const int ph = rows * cell;
    const int pw = cols * cell;
    const float pad_norm = (-1.0f / 255.0f - 0.5f) / 0.5f;

    rec.orig_w = sw;
    rec.orig_h = sh;
    rec.padded_w = pw;
    rec.padded_h = ph;
    rec.grid_h = rows;
    rec.grid_w = cols;
    rec.file = path;
    rec.pixel_values.assign(static_cast<size_t>(rows) * cols * cell * cell * 3, pad_norm);

    for (int y = 0; y < sh; ++y) {
        const int gr = y / cell;
        const int ly = y % cell;
        for (int x = 0; x < sw; ++x) {
            const int gc = x / cell;
            const int lx = x % cell;
            const size_t tok = static_cast<size_t>(gr) * cols + gc;
            const size_t dst_off = ((tok * cell + ly) * cell + lx) * 3;
            const size_t src_off = (static_cast<size_t>(y) * sw + x) * 3;
            for (int c = 0; c < 3; ++c) {
                rec.pixel_values[dst_off + c] = (static_cast<float>(rgb[src_off + c]) / 255.0f - 0.5f) / 0.5f;
            }
        }
    }
    return true;
}

// Expand each single image placeholder (200054) into grid_h * grid_w consecutive image placeholders
inline bool expand_image_pads(
    const std::vector<int32_t>& ids,
    int image_token,
    const std::vector<ImageRec>& recs,
    std::vector<int32_t>& out_ids,
    std::vector<int32_t>& out_img_idx,
    std::string& err) {
    size_t pads = 0;
    for (int32_t t : ids) pads += (t == image_token);
    if (pads != recs.size()) {
        err = std::to_string(recs.size()) + " image file(s) for " + std::to_string(pads) + " image placeholder(s)";
        return false;
    }
    out_ids.clear();
    out_img_idx.clear();
    size_t k = 0;
    int32_t row_cursor = 0;
    for (int32_t t : ids) {
        if (t != image_token) {
            out_ids.push_back(t);
            out_img_idx.push_back(-1);
        } else {
            const int n = recs[k].num_lm_tokens();
            for (int i = 0; i < n; ++i) {
                out_ids.push_back(image_token);
                out_img_idx.push_back(row_cursor++);
            }
            ++k;
        }
    }
    return true;
}

// ---- Exact CPU 23-Layer Inkling2VisionModel (reference & universal fallback; GPU backends may override encode_image) ----
class CpuVisionEncoder {
public:
    bool loaded = false;
    VisionShardInfo info;

    bool load(const std::string& path, std::string& err) {
        if (loaded && info.path == path) return true;
        if (!vision_probe(path, info, err)) return false;
        weights_.clear();
        strata::artifact::StrataFileReader rd;
        if (!rd.open(path, err)) return false;
        std::vector<uint8_t> s1;
        if (!rd.read_section(1, s1, err)) return false;
        weights_.clear();
        for (const auto& kv : info.T) {
            const auto& vt = kv.second;
            size_t n = 1;
            for (int64_t d : vt.shape) n *= static_cast<size_t>(d);
            std::vector<float> w(n);
            const uint8_t* src = s1.data() + vt.off;
            if (vt.fmt == F_BF16) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
                for (size_t i = 0; i < n; ++i) w[i] = bf16_to_f32(p[i]);
            } else if (vt.fmt == F_F16) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
                for (size_t i = 0; i < n; ++i) w[i] = fp16_to_f32(p[i]);
            } else {
                std::memcpy(w.data(), src, n * sizeof(float));
            }
            weights_[kv.first] = std::move(w);
        }
        loaded = true;
        return true;
    }

    // Encode one prepared ImageRec -> [grid_h * grid_w, 2304] FP32 embeddings (bypasses LLM embed_norm!)
    bool encode(const ImageRec& img, std::vector<float>& out, std::string& err) const {
        if (!loaded) {
            err = "vision encoder not loaded";
            return false;
        }
        const VitCfg& c = info.c;
        const int merge = c.merge;       // 2
        const int patch = c.patch;       // 20
        const int cell = patch * merge;  // 40
        const int n_tok = img.num_lm_tokens();
        const int ph = img.grid_h * merge;
        const int pw = img.grid_w * merge;
        const int n_patches = ph * pw;
        const int S = n_patches + 1;     // CLS at index 0, patches at 1..n_patches
        const int D = c.hidden;          // 1024
        const int H = c.heads;           // 16
        const int hd = c.head_dim;       // 64

        auto W = [&](const std::string& name) -> const float* {
            auto it = weights_.find(name);
            return it == weights_.end() ? nullptr : it->second.data();
        };

        // 1. Patchify [n_tok, merge, patch, merge, patch, 3] -> transpose(2,3) -> [n_patches, 1200]
        // Patch order inside each merged cell (gr, gc): (mr=0,mc=0), (mr=0,mc=1), (mr=1,mc=0), (mr=1,mc=1)
        const int patch_in = 3 * patch * patch;  // 1200
        std::vector<float> patch_buf(static_cast<size_t>(n_patches) * patch_in);
        std::vector<int> pos_h(S, 0), pos_w(S, 0);
        int pi = 0;
        for (int gr = 0; gr < img.grid_h; ++gr) {
            for (int gc = 0; gc < img.grid_w; ++gc) {
                const size_t tok = static_cast<size_t>(gr) * img.grid_w + gc;
                const float* cell_px = img.pixel_values.data() + tok * cell * cell * 3;
                for (int mr = 0; mr < merge; ++mr) {
                    for (int mc = 0; mc < merge; ++mc) {
                        float* dst = patch_buf.data() + static_cast<size_t>(pi) * patch_in;
                        int idx = 0;
                        for (int py = 0; py < patch; ++py) {
                            const int ly = mr * patch + py;
                            for (int px = 0; px < patch; ++px) {
                                const int lx = mc * patch + px;
                                const float* rgb = cell_px + (static_cast<size_t>(ly) * cell + lx) * 3;
                                dst[idx++] = rgb[0];
                                dst[idx++] = rgb[1];
                                dst[idx++] = rgb[2];
                            }
                        }
                        pos_h[pi + 1] = gr * merge + mr + 1;  // 1-indexed for patches, 0 for CLS
                        pos_w[pi + 1] = gc * merge + mc + 1;
                        ++pi;
                    }
                }
            }
        }

        // 2. Patch linear embedding + bilinearly interpolated 32x32 position_embedding + CLS token
        std::vector<float> cur(static_cast<size_t>(S) * D, 0.0f);
        const float* w_cls = W("model.visual.backbone.embeddings.class_embedding");
        const float* w_cls_pos = W("model.visual.backbone.embeddings.cls_position_embedding");
        const float* w_patch = W("model.visual.backbone.embeddings.patch_embedding.weight");
        const float* w_pos = W("model.visual.backbone.embeddings.position_embedding");
        for (int d = 0; d < D; ++d) cur[d] = w_cls[d] + w_cls_pos[d];

        const int G = c.posemb_grid;  // 32
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < n_patches; ++p) {
            float* dst = cur.data() + static_cast<size_t>(p + 1) * D;
            const float* xp = patch_buf.data() + static_cast<size_t>(p) * patch_in;
            for (int d = 0; d < D; ++d) {
                const float* wr = w_patch + static_cast<size_t>(d) * patch_in;
                float s = 0.0f;
                for (int k = 0; k < patch_in; ++k) s += wr[k] * xp[k];
                dst[d] = s;
            }
            // Bilinear interpolation (align_corners=False) from [G, G] to [ph, pw]
            const float y = (static_cast<float>(pos_h[p + 1] - 1) + 0.5f) * (static_cast<float>(G) / ph) - 0.5f;
            const float x = (static_cast<float>(pos_w[p + 1] - 1) + 0.5f) * (static_cast<float>(G) / pw) - 0.5f;
            const int y0 = std::max(0, std::min(G - 1, static_cast<int>(std::floor(y))));
            const int y1 = std::min(G - 1, y0 + 1);
            const int x0 = std::max(0, std::min(G - 1, static_cast<int>(std::floor(x))));
            const int x1 = std::min(G - 1, x0 + 1);
            const float wy = std::max(0.0f, std::min(1.0f, y - y0));
            const float wx = std::max(0.0f, std::min(1.0f, x - x0));
            const float w00 = (1.0f - wy) * (1.0f - wx);
            const float w01 = (1.0f - wy) * wx;
            const float w10 = wy * (1.0f - wx);
            const float w11 = wy * wx;
            const float* p00 = w_pos + (static_cast<size_t>(y0) * G + x0) * D;
            const float* p01 = w_pos + (static_cast<size_t>(y0) * G + x1) * D;
            const float* p10 = w_pos + (static_cast<size_t>(y1) * G + x0) * D;
            const float* p11 = w_pos + (static_cast<size_t>(y1) * G + x1) * D;
            for (int d = 0; d < D; ++d) {
                dst[d] += w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
            }
        }

        // 3. Pre-LayerNorm
        layer_norm_rows(cur.data(), W("model.visual.backbone.ln_pre.weight"), W("model.visual.backbone.ln_pre.bias"), S, D, c.ln_eps);

        // 4. Axial 2D RoPE cos/sin table [S, hd]: [w_freqs, h_freqs] interleaved pairwise
        const int sp_dim = hd / 2;  // 32
        const int n_freq = sp_dim / 2;  // 16
        std::vector<float> inv_freq(n_freq);
        for (int i = 0; i < n_freq; ++i) {
            inv_freq[i] = 1.0f / std::pow(c.rope_theta, static_cast<float>(2 * i) / sp_dim);
        }
        std::vector<float> rope_cos(static_cast<size_t>(S) * hd);
        std::vector<float> rope_sin(static_cast<size_t>(S) * hd);
        for (int t = 0; t < S; ++t) {
            float* rc = rope_cos.data() + static_cast<size_t>(t) * hd;
            float* rs = rope_sin.data() + static_cast<size_t>(t) * hd;
            for (int i = 0; i < n_freq; ++i) {
                float aw = static_cast<float>(pos_w[t]) * inv_freq[i];
                float ah = static_cast<float>(pos_h[t]) * inv_freq[i];
                rc[2 * i] = rc[2 * i + 1] = std::cos(aw);
                rs[2 * i] = rs[2 * i + 1] = std::sin(aw);
                rc[sp_dim + 2 * i] = rc[sp_dim + 2 * i + 1] = std::cos(ah);
                rs[sp_dim + 2 * i] = rs[sp_dim + 2 * i + 1] = std::sin(ah);
            }
        }

        // 5. 23 ViT Layers (Pre-LN + Self-Attn w/ Axial 2D RoPE + LayerScale ls_1 + Pre-LN + GELU MLP + ls_2)
        std::vector<float> norm(static_cast<size_t>(S) * D);
        std::vector<float> q(static_cast<size_t>(S) * D), k(static_cast<size_t>(S) * D), v(static_cast<size_t>(S) * D);
        std::vector<float> attn_out(static_cast<size_t>(S) * D), proj(static_cast<size_t>(S) * D);
        std::vector<float> mlp_h(static_cast<size_t>(S) * c.inter);
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));

        for (int il = 0; il < c.layers; ++il) {
            const std::string p = "model.visual.backbone.layers." + std::to_string(il);
            std::memcpy(norm.data(), cur.data(), cur.size() * sizeof(float));
            layer_norm_rows(norm.data(), W(p + ".ln_1.weight"), W(p + ".ln_1.bias"), S, D, c.ln_eps);

            linear_bias(norm.data(), W(p + ".self_attn.q_proj.weight"), W(p + ".self_attn.q_proj.bias"), q.data(), S, D, D);
            linear_bias(norm.data(), W(p + ".self_attn.k_proj.weight"), W(p + ".self_attn.k_proj.bias"), k.data(), S, D, D);
            linear_bias(norm.data(), W(p + ".self_attn.v_proj.weight"), W(p + ".self_attn.v_proj.bias"), v.data(), S, D, D);

            // Apply pairwise axial 2D RoPE to Q and K
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < S; ++t) {
                const float* rc = rope_cos.data() + static_cast<size_t>(t) * hd;
                const float* rs = rope_sin.data() + static_cast<size_t>(t) * hd;
                for (int h = 0; h < H; ++h) {
                    float* qh = q.data() + (static_cast<size_t>(t) * H + h) * hd;
                    float* kh = k.data() + (static_cast<size_t>(t) * H + h) * hd;
                    for (int j = 0; j < hd; j += 2) {
                        float q0 = qh[j], q1 = qh[j + 1];
                        qh[j]     = q0 * rc[j] - q1 * rs[j];
                        qh[j + 1] = q1 * rc[j + 1] + q0 * rs[j + 1];
                        float k0 = kh[j], k1 = kh[j + 1];
                        kh[j]     = k0 * rc[j] - k1 * rs[j];
                        kh[j + 1] = k1 * rc[j + 1] + k0 * rs[j + 1];
                    }
                }
            }

            // Full non-causal self-attention across all S tokens
            #pragma omp parallel for schedule(static)
            for (int qi = 0; qi < S; ++qi) {
                std::vector<float> sc(S);
                for (int h = 0; h < H; ++h) {
                    const float* qv = q.data() + (static_cast<size_t>(qi) * H + h) * hd;
                    float mx = -1e30f;
                    for (int ki = 0; ki < S; ++ki) {
                        const float* kv = k.data() + (static_cast<size_t>(ki) * H + h) * hd;
                        float dot = 0.0f;
                        for (int j = 0; j < hd; ++j) dot += qv[j] * kv[j];
                        sc[ki] = dot * attn_scale;
                        if (sc[ki] > mx) mx = sc[ki];
                    }
                    float sum = 0.0f;
                    for (int ki = 0; ki < S; ++ki) { sc[ki] = std::exp(sc[ki] - mx); sum += sc[ki]; }
                    const float inv = 1.0f / sum;
                    float* ov = attn_out.data() + (static_cast<size_t>(qi) * H + h) * hd;
                    std::memset(ov, 0, hd * sizeof(float));
                    for (int ki = 0; ki < S; ++ki) {
                        const float prob = sc[ki] * inv;
                        const float* vv = v.data() + (static_cast<size_t>(ki) * H + h) * hd;
                        for (int j = 0; j < hd; ++j) ov[j] += prob * vv[j];
                    }
                }
            }

            linear_bias(attn_out.data(), W(p + ".self_attn.out_proj.weight"), W(p + ".self_attn.out_proj.bias"), proj.data(), S, D, D);
            const float* ls1 = W(p + ".ls_1.gamma");
            for (int t = 0; t < S; ++t) {
                float* ct = cur.data() + static_cast<size_t>(t) * D;
                const float* pt = proj.data() + static_cast<size_t>(t) * D;
                for (int d = 0; d < D; ++d) ct[d] += ls1[d] * pt[d];
            }

            // MLP sublayer
            std::memcpy(norm.data(), cur.data(), cur.size() * sizeof(float));
            layer_norm_rows(norm.data(), W(p + ".ln_2.weight"), W(p + ".ln_2.bias"), S, D, c.ln_eps);
            linear_bias(norm.data(), W(p + ".mlp.c_fc.weight"), W(p + ".mlp.c_fc.bias"), mlp_h.data(), S, c.inter, D);
            for (float& x : mlp_h) x = gelu_exact(x);
            linear_bias(mlp_h.data(), W(p + ".mlp.c_proj.weight"), W(p + ".mlp.c_proj.bias"), proj.data(), S, D, c.inter);
            const float* ls2 = W(p + ".ls_2.gamma");
            for (int t = 0; t < S; ++t) {
                float* ct = cur.data() + static_cast<size_t>(t) * D;
                const float* pt = proj.data() + static_cast<size_t>(t) * D;
                for (int d = 0; d < D; ++d) ct[d] += ls2[d] * pt[d];
            }
        }

        // 6. Drop CLS (index 0), merge 2x2 patches [n_tok, 4096] -> proj_in [4096,4096] -> GELU -> proj_out [2304,4096]
        const int merged_dim = merge * merge * D;  // 4096
        const float* patch_states = cur.data() + D;  // already in [n_tok, merge*merge, D] raster order!
        std::vector<float> mid(static_cast<size_t>(n_tok) * c.proj_hidden);
        linear_bias(patch_states, W("model.visual.proj_in.weight"), nullptr, mid.data(), n_tok, c.proj_hidden, merged_dim);
        for (float& x : mid) x = gelu_exact(x);
        out.resize(static_cast<size_t>(n_tok) * c.out_hidden);
        linear_bias(mid.data(), W("model.visual.proj_out.weight"), nullptr, out.data(), n_tok, c.out_hidden, c.proj_hidden);
        return true;
    }

private:
    std::unordered_map<std::string, std::vector<float>> weights_;

    static inline float gelu_exact(float x) noexcept {
        return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f));
    }

    static void layer_norm_rows(float* x, const float* w, const float* b, int rows, int D, float eps) {
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            float* xr = x + static_cast<size_t>(r) * D;
            float mean = 0.0f;
            for (int d = 0; d < D; ++d) mean += xr[d];
            mean /= static_cast<float>(D);
            float var = 0.0f;
            for (int d = 0; d < D; ++d) { float t = xr[d] - mean; var += t * t; }
            const float inv = 1.0f / std::sqrt(var / static_cast<float>(D) + eps);
            for (int d = 0; d < D; ++d) xr[d] = (xr[d] - mean) * inv * w[d] + b[d];
        }
    }

    static void linear_bias(const float* x, const float* w, const float* b, float* y, int M, int N, int K) {
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<size_t>(m) * K;
            float* ym = y + static_cast<size_t>(m) * N;
            for (int n = 0; n < N; ++n) {
                const float* wn = w + static_cast<size_t>(n) * K;
                float s = b ? b[n] : 0.0f;
                for (int k = 0; k < K; ++k) s += xm[k] * wn[k];
                ym[n] = s;
            }
        }
    }
};

}  // namespace ink2
