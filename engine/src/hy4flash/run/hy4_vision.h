// hy4_vision.h - host side of the Hy4-Flash image path (plain C++, no GPU): Vision shard manifest, picture decode / resize / placeholder expansion (protocol 1.6b), image geometry,
// and the 3D (t, h, w) rope positions of a prompt that carries images.
//
// Model facts (Hy4-Flash-Vision.strata manifest + transformers models/hy_v4_vl + the llama.cpp port): SigLIP-shaped ViT, 27 pre-LayerNorm blocks (hidden 1152, 16 heads, ffn 4304, biased
// q/k/v/o, gelu), no rope in the ViT: a learned 128x128 (+1 cls row) position grid bilinearly resampled to the patch grid with scale_factor (n+0.1)/128, patch 16, then the
// rmsnorm_mlp_adapter: RMSNorm -> Conv2d(1152->4608, k2 s2) -> GELU -> Conv2d(4608->3072, k1) and one learned `image_newline` row appended to every merged row.
// An image of H x W pixels (both multiples of 32) is (H/32) rows of (W/32 + 1) rows-of-the-LM: the placeholder run in the prompt has (H/32)*(W/32 + 1) image-pad tokens.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "hy4_json.h"
#include "strata/artifact/strata_reader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#include "../../../third_party/stb/stb_image.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define HY4V_HD __host__ __device__
#else
#define HY4V_HD
#endif

namespace hy4 {

struct VitCfg {
    int hidden = 1152, heads = 16, ffn = 4304, layers = 27, patch = 16, merge = 2, out_hidden = 3072, pos_edge = 128;
    float eps = 1e-5f;
    int image_token = 120120, start_token = 120118, end_token = 120119, newline_token = 120121;
    int64_t min_pixels = 4096, max_pixels = 4194304;  // processor_config.json of tencent/Hy4-flash (an engine-side cap may lower max_pixels)
    bool gelu_tanh_ffn = true;                         // bias_activation_fusion: the encoder FFN uses the tanh GELU; the projector's GELU is the exact (erf) one
};

struct VitTensor {
    bool f16 = true;
    std::vector<int64_t> shape;
    uint64_t off = 0, bytes = 0;  // relative to Section 1
};

struct VisionShardInfo {
    VitCfg c;
    std::unordered_map<std::string, VitTensor> T;
    uint64_t sec1_off = 0, sec1_len = 0;
    std::string path;
    const VitTensor& get(const std::string& n) const {
        auto it = T.find(n);
        if (it == T.end()) {
            fprintf(stderr, "[vision] missing tensor %s\n", n.c_str());
            exit(2);
        }
        return it->second;
    }
};

// Opens the Vision shard, checks profile 3 and the tensor set (441 tensors). No GPU. Used by --info / serve start and by the encoder.
inline bool vision_probe(const std::string& path, VisionShardInfo& vi, std::string& err) {
    strata::artifact::StrataFileReader rd;
    if (!rd.open(path, err)) return false;
    if (rd.header().profile != strata::artifact::StrataProfile::VisionShard) {
        err = "not a Vision shard (profile " + std::to_string((int)rd.header().profile) + ")";
        return false;
    }
    try {
        JVal m = json_parse(rd.manifest_json());
        const JVal& vc = m.at("vision_config");
        VitCfg& c = vi.c;
        c.hidden = (int)vc.at("hidden_size").i();
        c.heads = (int)vc.at("num_attention_heads").i();
        c.ffn = (int)vc.at("intermediate_size").i();
        c.layers = (int)vc.at("num_hidden_layers").i();
        c.patch = (int)vc.at("patch_size").i();
        c.merge = (int)vc.at("spatial_merge_size").i();
        c.out_hidden = (int)vc.at("out_hidden_size").i();
        c.pos_edge = (int)vc.at("max_image_size").i() / c.patch;
        c.eps = (float)vc.at("rms_norm_eps").f();
        if (const JVal* v = m.get("image_token_id")) c.image_token = (int)v->i();
        if (const JVal* v = m.get("image_start_token_id")) c.start_token = (int)v->i();
        if (const JVal* v = m.get("image_end_token_id")) c.end_token = (int)v->i();
        if (const JVal* v = m.get("image_newline_token_id")) c.newline_token = (int)v->i();
        if (vc.at("projector_type").str() != "rmsnorm_mlp_adapter" || c.hidden % c.heads != 0 || c.merge != 2) {
            err = "unsupported vision config";
            return false;
        }
        for (const JVal& t : m.at("tensors").a) {
            VitTensor vt;
            const std::string dt = t.at("dtype").str();
            if (dt != "F16" && dt != "F32") {
                err = "unsupported tensor dtype " + dt;
                return false;
            }
            vt.f16 = dt == "F16";
            for (const JVal& s : t.at("shape").a) vt.shape.push_back(s.i());
            vt.off = (uint64_t)t.at("offset").i();
            vt.bytes = (uint64_t)t.at("bytes").i();
            vi.T[t.at("name").str()] = vt;
        }
    } catch (const std::exception& e) {
        err = std::string("vision manifest: ") + e.what();
        return false;
    }
    vi.sec1_off = rd.header().sections[1].file_offset;
    vi.sec1_len = rd.header().sections[1].byte_length;
    vi.path = path;
    // sanity: the tensors every layer needs
    const char* need[] = {"vit.embeddings.patch_embedding.weight", "vit.embeddings.position_embedding.weight", "vit.perceive.norm.weight", "vit.perceive.proj.0.weight",
                          "vit.perceive.proj.2.weight", "vit.perceive.image_newline", "vit.layers.26.mlp.dense_4h_to_h.weight"};
    for (const char* n : need)
        if (!vi.T.count(n)) {
            err = std::string("vision shard lacks ") + n;
            return false;
        }
    return true;
}

// ---- image geometry (transformers image_processing_pil_hy_v4_vl.smart_resize with factor 32) ------------------------------------------------------------------
// serve resizes the picture (PIL bicubic) to this size and hands the engine the raw RGB; the engine re-checks that the size is a multiple of 32.
inline bool smart_resize(int h, int w, int factor, int64_t min_px, int64_t max_px, int& oh, int& ow) {
    if (h < 1 || w < 1 || std::max(h, w) / (double)std::min(h, w) > 200.0) return false;
    auto rnd = [](double v) { return (int)std::nearbyint(v); };  // python round(): half to even
    int hb = rnd(h / (double)factor) * factor, wb = rnd(w / (double)factor) * factor;
    if ((int64_t)hb * wb > max_px) {
        const double beta = std::sqrt((double)h * w / (double)max_px);
        hb = std::max(factor, (int)std::floor(h / beta / factor) * factor);
        wb = std::max(factor, (int)std::floor(w / beta / factor) * factor);
    } else if ((int64_t)hb * wb < min_px) {
        const double beta = std::sqrt((double)min_px / ((double)h * w));
        hb = (int)std::ceil(h * beta / factor) * factor;
        wb = (int)std::ceil(w * beta / factor) * factor;
    }
    oh = hb;
    ow = wb;
    return true;
}

// placeholder tokens of one picture of h x w pixels (multiples of 32)
inline int image_token_count(int h, int w) { return (h / 32) * (w / 32 + 1); }

// ---- pictures: decode (stb_image: JPEG, PNG, BMP, GIF), resize like the HF processor, expand the placeholder --------------------------------------------------------------
// serve hands the engine the picture file (protocol 1.6b: image=<path>). The engine decodes it (alpha dropped like PIL convert("RGB")), picks the size with smart_resize (factor 32,
// min_pixels 4096, max_pixels 4194304 unless capped) and resizes with PIL's BICUBIC algorithm (support scaled by the reduction, 22-bit fixed-point taps, an 8-bit intermediate after the
// horizontal pass: the same arithmetic as Pillow's ImagingResample, so the pixels equal PIL's).
struct ImageRec {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;  // h*w*3, row-major, resized (multiples of 32)
    std::string file;
};

inline std::string pct_decode(const std::string& s) {  // %20 -> space, %25 -> %, any %XX
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && isxdigit((unsigned char)s[i + 1]) && isxdigit((unsigned char)s[i + 2])) {
            o += (char)strtol(s.substr(i + 1, 2).c_str(), nullptr, 16);
            i += 2;
        } else
            o += s[i];
    }
    return o;
}

namespace pil {
inline double bicubic(double x) {
    const double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}
struct Taps {
    int ksize = 0;
    std::vector<int> bounds;   // [out][2] = xmin, count
    std::vector<int32_t> kk;   // [out][ksize] fixed point, 22 bits
};
inline Taps coeffs(int in_size, int out_size) {
    Taps t;
    const double scale = (double)in_size / out_size, fs = scale < 1.0 ? 1.0 : scale, support = 2.0 * fs;
    t.ksize = (int)std::ceil(support) * 2 + 1;
    t.bounds.assign((size_t)out_size * 2, 0);
    t.kk.assign((size_t)out_size * t.ksize, 0);
    std::vector<double> k(t.ksize);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale, ss = 1.0 / fs;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double ww = 0;
        for (int x = 0; x < t.ksize; ++x) k[x] = 0;
        for (int x = 0; x < xmax; ++x) {
            const double w = bicubic((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; ++x)
            if (ww != 0.0) k[x] /= ww;
        for (int x = 0; x < t.ksize; ++x) t.kk[(size_t)xx * t.ksize + x] = (int32_t)(k[x] < 0 ? -0.5 + k[x] * (1 << 22) : 0.5 + k[x] * (1 << 22));
        t.bounds[xx * 2] = xmin;
        t.bounds[xx * 2 + 1] = xmax;
    }
    return t;
}
inline uint8_t clip8(int64_t v) {
    v >>= 22;
    return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}
// RGB u8 resize, horizontal pass first then vertical (Pillow's order), a pass is skipped when that size does not change
inline void resize_rgb(const uint8_t* src, int sw, int sh, std::vector<uint8_t>& dst, int dw, int dh) {
    std::vector<uint8_t> mid;
    const uint8_t* cur = src;
    int cw = sw;
    if (dw != sw) {
        Taps t = coeffs(sw, dw);
        mid.assign((size_t)dw * sh * 3, 0);
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < dw; ++x) {
                const int xmin = t.bounds[x * 2], n = t.bounds[x * 2 + 1];
                const int32_t* k = &t.kk[(size_t)x * t.ksize];
                for (int c = 0; c < 3; ++c) {
                    int64_t s = 1 << 21;
                    for (int i = 0; i < n; ++i) s += (int64_t)src[((size_t)y * sw + xmin + i) * 3 + c] * k[i];
                    mid[((size_t)y * dw + x) * 3 + c] = clip8(s);
                }
            }
        cur = mid.data();
        cw = dw;
    }
    if (dh != sh) {
        Taps t = coeffs(sh, dh);
        dst.assign((size_t)cw * dh * 3, 0);
        for (int y = 0; y < dh; ++y) {
            const int ymin = t.bounds[y * 2], n = t.bounds[y * 2 + 1];
            const int32_t* k = &t.kk[(size_t)y * t.ksize];
            for (int x = 0; x < cw; ++x)
                for (int c = 0; c < 3; ++c) {
                    int64_t s = 1 << 21;
                    for (int i = 0; i < n; ++i) s += (int64_t)cur[((size_t)(ymin + i) * cw + x) * 3 + c] * k[i];
                    dst[((size_t)y * cw + x) * 3 + c] = clip8(s);
                }
        }
    } else {
        dst.assign(cur, cur + (size_t)cw * sh * 3);
    }
}
}  // namespace pil

// Reads a picture file and returns it resized for the ViT. Errors are one line for `ERR image: ...`.
inline bool prepare_picture(const std::string& path, int64_t min_px, int64_t max_px, ImageRec& rec, std::string& err) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::vector<uint8_t> buf;
    std::vector<uint8_t> tmp(65536);
    size_t n;
    while ((n = fread(tmp.data(), 1, tmp.size(), f)) > 0) buf.insert(buf.end(), tmp.begin(), tmp.begin() + n);
    fclose(f);
    int w = 0, h = 0, comp = 0;
    uint8_t* px = stbi_load_from_memory(buf.data(), (int)buf.size(), &w, &h, &comp, 3);
    if (!px) {
        err = std::string("cannot decode ") + path + ": " + stbi_failure_reason();
        return false;
    }
    int oh = 0, ow = 0;
    if (!smart_resize(h, w, 32, min_px, max_px, oh, ow)) {
        stbi_image_free(px);
        err = "picture " + std::to_string(w) + "x" + std::to_string(h) + ": aspect ratio above 200";
        return false;
    }
    pil::resize_rgb(px, w, h, rec.rgb, ow, oh);
    stbi_image_free(px);
    rec.w = ow;
    rec.h = oh;
    rec.file = path;
    return true;
}

// ids with ONE image-pad per picture (protocol 1.6b) -> ids with each pad replaced by the picture's run of pads. false + err (the `ERR image: ...` text) when the counts differ.
inline bool expand_image_pads(const std::vector<int32_t>& ids, int image_token, const std::vector<ImageRec>& recs, std::vector<int32_t>& out, std::string& err) {
    size_t pads = 0;
    for (int32_t t : ids) pads += t == image_token;
    if (pads != recs.size()) {
        err = std::to_string(recs.size()) + " files for " + std::to_string(pads) + " image placeholders";
        return false;
    }
    out.clear();
    out.reserve(ids.size() + 4160 * recs.size());
    size_t k = 0;
    for (int32_t t : ids) {
        if (t != image_token) {
            out.push_back(t);
            continue;
        }
        const int n = image_token_count(recs[k].h, recs[k].w);
        out.insert(out.end(), (size_t)n, image_token);
        ++k;
    }
    return true;
}

// ---- positions of a prompt with images (transformers HYV4VLModel.get_rope_index / get_vision_position_ids) ----------------------------------------------------------
// Text tokens: all three axes = running position. A picture of oh x (ow+1) cells starting at running position p: t = p, h = p + row, w = p + col; the next text token continues at
// p + max(oh, ow + 1). The cache/attention order is still the token order; only the rope angles use these positions. After the prompt every axis = index + delta.
struct ImageSpan {
    int start = 0;       // index of the first image-pad token of the run
    int n = 0;           // run length = oh * (ow + 1)
    int oh = 0, ow = 0;  // merged rows / cols of the picture
    int row0 = 0;        // first row of this picture in the request's image-row buffer
};

struct RopePos {
    std::vector<int32_t> p[3];  // per token index of the prompt (t, h, w)
    int delta = 0;              // positions of tokens at index >= p[0].size(): index + delta
    bool active() const { return !p[0].empty(); }
};

// ids: the prompt; recs: the pictures in order; fills spans (one per run of image_token) and pos. Returns false with err when the runs and the pictures do not match.
inline bool build_image_spans(const std::vector<int32_t>& ids, int image_token, const std::vector<ImageRec>& recs, std::vector<ImageSpan>& spans, RopePos& pos, std::string& err) {
    spans.clear();
    const int n = (int)ids.size();
    int row = 0;
    for (int i = 0; i < n;) {
        if (ids[i] != image_token) {
            ++i;
            continue;
        }
        int j = i;
        while (j < n && ids[j] == image_token) ++j;
        const size_t k = spans.size();
        if (k >= recs.size()) {
            err = "the prompt has more image runs than pictures";
            return false;
        }
        ImageSpan s;
        s.start = i;
        s.n = j - i;
        s.oh = recs[k].h / 32;
        s.ow = recs[k].w / 32;
        if (s.n != s.oh * (s.ow + 1)) {
            err = "image " + std::to_string(k + 1) + ": the prompt has " + std::to_string(s.n) + " image-pad tokens but a " + std::to_string(recs[k].w) + "x" + std::to_string(recs[k].h) +
                  " picture needs " + std::to_string(s.oh * (s.ow + 1));
            return false;
        }
        s.row0 = row;
        row += s.n;
        spans.push_back(s);
        i = j;
    }
    if (spans.size() != recs.size()) {
        err = "the prompt has " + std::to_string(spans.size()) + " image runs but " + std::to_string(recs.size()) + " pictures were given";
        return false;
    }
    if (n > 0 && ids[n - 1] == image_token) {
        err = "the prompt cannot end in an image";
        return false;
    }
    for (int a = 0; a < 3; ++a) pos.p[a].assign(n, 0);
    int cur = 0, k = 0;
    for (int i = 0; i < n;) {
        if (k < (int)spans.size() && spans[k].start == i) {
            const ImageSpan& s = spans[k];
            for (int r = 0; r < s.oh; ++r)
                for (int c = 0; c <= s.ow; ++c) {
                    const int idx = i + r * (s.ow + 1) + c;
                    pos.p[0][idx] = cur;
                    pos.p[1][idx] = cur + r;
                    pos.p[2][idx] = cur + c;
                }
            cur += std::max(s.oh, s.ow + 1);
            i += s.n;
            ++k;
        } else {
            pos.p[0][i] = pos.p[1][i] = pos.p[2][i] = cur++;
            ++i;
        }
    }
    pos.delta = cur - n;
    return true;
}

// the three-axis choice of rope pair i of the 32 (mrope_interleaved, mrope_section [12, 10, 10]): pairs 1, 4, ..., 28 follow h, pairs 2, 5, ..., 29 follow w, the rest t
HY4V_HD inline int mrope_axis(int i) {
    if (i < 30) {
        if (i % 3 == 1) return 1;
        if (i % 3 == 2) return 2;
    }
    return 0;
}

}  // namespace hy4
