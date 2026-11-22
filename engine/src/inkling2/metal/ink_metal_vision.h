// ink_metal_vision.h - Host + Metal ViT Encoder for Inkling-2-Mini-Vision.strata (23-layer BF16 ViT)
// 100% MIT clean-room implementation.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "hy4_json.h"
#include "ink_metal_common.h"
#include "mtl_rt.h"

#ifdef INK_DEFINE_STBI
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#endif
#ifndef STBI_INCLUDE_STB_IMAGE_H
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#include "../../../third_party/stb/stb_image.h"
#endif

namespace ink_metal {

struct InkVisTensor {
    uint64_t off = 0;
    uint64_t bytes = 0;
    std::vector<int64_t> shape;
};

struct InkMetalVision {
    int hidden = 1024;
    int layers = 23;
    int heads = 16;
    int patch = 20;
    int merge = 2;
    int proj_hidden = 4096;
    int out_hidden = 2304;
    int pos_grid = 32;
    float ln_eps = 1e-5f;

    int fd = -1;
    uint8_t* base = nullptr;
    size_t fsize = 0;
    size_t map_len = 0;
    uint64_t sec1_off = 0;
    mtl::Buffer wbuf;
    std::unordered_map<std::string, InkVisTensor> T;

    // Compiled pipelines
    mtl::Pipeline p_patchify, p_add_pos_cls, p_ln, p_bias_gelu, p_ls_res_add;
    mtl::Pipeline p_rope2d, p_attn, p_merge2x2, p_mm_bf16, p_mv_bf16;
    bool loaded = false;

    ~InkMetalVision() {
        if (base && base != MAP_FAILED) munmap(base, map_len);
        if (fd >= 0) ::close(fd);
    }

    bool load(mtl::Context& cx, int lib_codecs, int lib_vis, const std::string& path,
              bool (*read_s0_fn)(int, uint64_t, uint64_t, std::string&, std::string&),
              std::string& err) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "cannot open vision shard: " + path; return false; }
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size < 4096) { err = "bad vision shard size"; return false; }
        fsize = (size_t)st.st_size;

        uint8_t frame[4096];
        if (::pread(fd, frame, 4096, 0) != 4096) { err = "short read on vision header"; return false; }
        strata::artifact::StrataSectionEntry secs[6]{};
        for (int i = 0; i < 6; ++i) std::memcpy(&secs[i], frame + 0x10 + i * 64, 64);
        sec1_off = secs[1].file_offset;

        std::string mjson;
        if (!read_s0_fn(fd, secs[0].file_offset, secs[0].byte_length, mjson, err)) return false;
        try {
            hy4::JVal root = hy4::json_parse(mjson);
            const hy4::JVal& vc = root.at("vision_config");
            hidden = (int)vc.at("hidden_size").i();
            layers = (int)vc.at("num_hidden_layers").i();
            heads = (int)vc.at("num_attention_heads").i();
            patch = (int)vc.at("native_patch_size").i();
            merge = (int)vc.at("adapter_merge_size").i();
            proj_hidden = (int)vc.at("projection_hidden_dim").i();
            out_hidden = (int)vc.at("text_hidden_size").i();
            pos_grid = (int)vc.at("posemb_grid_size").i();
            ln_eps = (float)vc.at("layer_norm_eps").f();
            for (const auto& t : root.at("tensors").a) {
                InkVisTensor vt;
                vt.off = (uint64_t)t.at("offset").i();
                vt.bytes = (uint64_t)t.at("bytes").i();
                for (const auto& s : t.at("shape").a) vt.shape.push_back(s.i());
                T[t.at("name").str()] = std::move(vt);
            }
        } catch (const std::exception& e) {
            err = std::string("vision manifest: ") + e.what();
            return false;
        }

        const size_t pg = 16384;
        map_len = (fsize + pg - 1) / pg * pg;
        void* m = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { err = "vision mmap failed"; return false; }
        base = static_cast<uint8_t*>(m);
        wbuf = cx.wrap(base, map_len);

        p_patchify    = cx.pipeline(lib_vis, "iv_patchify");
        p_add_pos_cls = cx.pipeline(lib_vis, "iv_add_pos_cls");
        p_ln          = cx.pipeline(lib_vis, "iv_ln_bf16");
        p_bias_gelu   = cx.pipeline(lib_vis, "iv_bias_gelu");
        p_ls_res_add  = cx.pipeline(lib_vis, "iv_ls_res_add");
        p_rope2d      = cx.pipeline(lib_vis, "iv_rope2d");
        p_attn        = cx.pipeline(lib_vis, "iv_attn");
        p_merge2x2    = cx.pipeline(lib_vis, "iv_merge2x2");
        p_mm_bf16     = cx.pipeline(lib_codecs, "ink_linear_bf16_mm");
        p_mv_bf16     = cx.pipeline(lib_codecs, "ink_linear_bf16_mv");

        loaded = true;
        fprintf(stderr, "[ink_metal_vis] loaded %s (%zu BF16 tensors, %.1f MB)\n",
                path.c_str(), T.size(), fsize / 1e6);
        return true;
    }

    mtl::Arg tw(const std::string& n) const {
        auto it = T.find(n);
        if (it == T.end()) { fprintf(stderr, "[ink_metal_vis] missing %s\n", n.c_str()); exit(2); }
        return mtl::Arg::B(wbuf, sec1_off + it->second.off);
    }

    // Decode image file (PNG/JPEG/BMP/GIF), composite alpha on 8x8 204/153 checkerboard,
    // pad to multiple of 40 (fill=-1.0), run 23-layer Metal ViT -> out_embeds [n_tokens, 2304]
    bool encode_image_file(mtl::Context& cx, const std::string& img_path,
                           std::vector<float>& out_embeds, int& n_tokens, std::string& err) {
        if (!loaded) { err = "vision shard not loaded"; return false; }
        int w = 0, h = 0, ch = 0;
        unsigned char* raw = stbi_load(img_path.c_str(), &w, &h, &ch, 4);
        if (!raw) { err = "stbi_load failed: " + img_path; return false; }

        // Cap max side to 4096 (matches CpuVisionEncoder max_side)
        int tw_px = w, th_px = h;
        const int max_side = 4096;
        if (tw_px > max_side || th_px > max_side) {
            const double sc = (double)max_side / (double)std::max(tw_px, th_px);
            tw_px = std::max(1, (int)std::floor(tw_px * sc));
            th_px = std::max(1, (int)std::floor(th_px * sc));
        }
        const int factor = patch * merge;  // 40
        const int pad_w = ((tw_px + factor - 1) / factor) * factor;
        const int pad_h = ((th_px + factor - 1) / factor) * factor;
        const int gw = pad_w / patch, gh = pad_h / patch;
        const int Np = gw * gh;
        const int S = 1 + Np;
        n_tokens = (gh / merge) * (gw / merge);

        mtl::Buffer rgb_buf = cx.alloc((size_t)pad_h * pad_w * 3 * sizeof(float));
        float* rgb = rgb_buf.as<float>();
        const float pad_norm = (-1.0f / 255.0f - 0.5f) / 0.5f;
        std::fill(rgb, rgb + (size_t)pad_h * pad_w * 3, pad_norm);
        for (int y = 0; y < th_px; ++y) {
            const int sy = std::min(h - 1, (int)((int64_t)y * h / th_px));
            for (int x = 0; x < tw_px; ++x) {
                const int sx = std::min(w - 1, (int)((int64_t)x * w / tw_px));
                const unsigned char* px = raw + ((size_t)sy * w + sx) * 4;
                const float alpha = px[3] * (1.0f / 255.0f);
                const float bg = ((((x >> 3) + (y >> 3)) & 1) == 0 ? 204.0f : 153.0f) * (1.0f / 255.0f);
                for (int c = 0; c < 3; ++c) {
                    const float fg = px[c] * (1.0f / 255.0f);
                    const float comp = fg * alpha + bg * (1.0f - alpha);
                    rgb[((size_t)y * pad_w + x) * 3 + c] = (comp - 0.5f) * 2.0f;
                }
            }
        }
        stbi_image_free(raw);

        // Allocate temporary activation buffers
        mtl::Buffer pix_buf   = cx.alloc((size_t)Np * 1200 * sizeof(float));
        mtl::Buffer pemb_buf  = cx.alloc((size_t)Np * hidden * sizeof(float));
        mtl::Buffer x_buf     = cx.alloc((size_t)S * hidden * sizeof(float));
        mtl::Buffer xn_buf    = cx.alloc((size_t)S * hidden * sizeof(float));
        mtl::Buffer q_buf     = cx.alloc((size_t)S * hidden * sizeof(float));
        mtl::Buffer k_buf     = cx.alloc((size_t)S * hidden * sizeof(float));
        mtl::Buffer v_buf     = cx.alloc((size_t)S * hidden * sizeof(float));
        mtl::Buffer ao_buf    = cx.alloc((size_t)S * hidden * sizeof(float));
        mtl::Buffer proj_buf  = cx.alloc((size_t)S * hidden * sizeof(float));
        mtl::Buffer mlp_h     = cx.alloc((size_t)S * proj_hidden * sizeof(float));
        mtl::Buffer merged    = cx.alloc((size_t)n_tokens * proj_hidden * sizeof(float));
        mtl::Buffer p_in_out  = cx.alloc((size_t)n_tokens * proj_hidden * sizeof(float));
        mtl::Buffer final_out = cx.alloc((size_t)n_tokens * out_hidden * sizeof(float));

        std::vector<mtl::Dispatch> ds;
        auto push_mm = [&](const std::string& wname, const mtl::Buffer& X, const mtl::Buffer& Y, int N, int K, int M) {
            mtl::Dispatch d;
            if (M > 8) {
                d.p = &p_mm_bf16;
                d.grid[0] = (uint32_t)((N + 63) / 64);
                d.grid[1] = (uint32_t)((M + 63) / 64);
                d.tg[0] = 256;
            } else {
                d.p = &p_mv_bf16;
                d.grid[0] = (uint32_t)((N + 15) / 16);
                d.tg[0] = 256;
            }
            d.args = {tw(wname), mtl::Arg::B(X), mtl::Arg::B(Y), mtl::Arg::V(N), mtl::Arg::V(K), mtl::Arg::V(M)};
            ds.push_back(std::move(d));
        };

        // Patchify + patch_embedding + CLS + bilinear 32x32 pos_emb + ln_pre
        {
            mtl::Dispatch d;
            d.p = &p_patchify;
            d.grid[0] = (uint32_t)Np; d.tg[0] = 256;
            d.args = {mtl::Arg::B(rgb_buf), mtl::Arg::B(pix_buf), mtl::Arg::V(pad_w), mtl::Arg::V(gw)};
            ds.push_back(std::move(d));
        }
        push_mm("model.visual.backbone.embeddings.patch_embedding.weight", pix_buf, pemb_buf, hidden, 1200, Np);
        {
            mtl::Dispatch d;
            d.p = &p_add_pos_cls;
            d.grid[0] = (uint32_t)S; d.tg[0] = 256;
            d.args = {mtl::Arg::B(pemb_buf),
                      tw("model.visual.backbone.embeddings.class_embedding"),
                      tw("model.visual.backbone.embeddings.cls_position_embedding"),
                      tw("model.visual.backbone.embeddings.position_embedding"),
                      mtl::Arg::B(x_buf), mtl::Arg::V(gh), mtl::Arg::V(gw)};
            ds.push_back(std::move(d));
        }
        {
            mtl::Dispatch d;
            d.p = &p_ln;
            d.grid[0] = (uint32_t)S; d.tg[0] = 256;
            d.args = {mtl::Arg::B(x_buf), tw("model.visual.backbone.ln_pre.weight"),
                      tw("model.visual.backbone.ln_pre.bias"), mtl::Arg::B(x_buf), mtl::Arg::V(ln_eps)};
            ds.push_back(std::move(d));
        }

        // 23 Pre-LN Transformer blocks
        for (int il = 0; il < layers; ++il) {
            std::string lp = "model.visual.backbone.layers." + std::to_string(il);
            {
                mtl::Dispatch d;
                d.p = &p_ln; d.grid[0] = (uint32_t)S; d.tg[0] = 256;
                d.args = {mtl::Arg::B(x_buf), tw(lp + ".ln_1.weight"), tw(lp + ".ln_1.bias"),
                          mtl::Arg::B(xn_buf), mtl::Arg::V(ln_eps)};
                ds.push_back(std::move(d));
            }
            push_mm(lp + ".self_attn.q_proj.weight", xn_buf, q_buf, hidden, hidden, S);
            push_mm(lp + ".self_attn.k_proj.weight", xn_buf, k_buf, hidden, hidden, S);
            push_mm(lp + ".self_attn.v_proj.weight", xn_buf, v_buf, hidden, hidden, S);
            auto push_bias = [&](const mtl::Buffer& B, const std::string& bn, int M, int N, int gelu) {
                mtl::Dispatch d;
                d.p = &p_bias_gelu; d.grid[0] = (uint32_t)((M * N + 255) / 256); d.tg[0] = 256;
                int hb = 1;
                d.args = {mtl::Arg::B(B), tw(bn), mtl::Arg::V(M), mtl::Arg::V(N), mtl::Arg::V(hb), mtl::Arg::V(gelu)};
                ds.push_back(std::move(d));
            };
            push_bias(q_buf, lp + ".self_attn.q_proj.bias", S, hidden, 0);
            push_bias(k_buf, lp + ".self_attn.k_proj.bias", S, hidden, 0);
            push_bias(v_buf, lp + ".self_attn.v_proj.bias", S, hidden, 0);
            {
                mtl::Dispatch dq; dq.p = &p_rope2d; dq.grid[0] = (uint32_t)S; dq.grid[1] = (uint32_t)heads; dq.tg[0] = 32;
                dq.args = {mtl::Arg::B(q_buf), mtl::Arg::V(gw)};
                ds.push_back(std::move(dq));
                mtl::Dispatch dk; dk.p = &p_rope2d; dk.grid[0] = (uint32_t)S; dk.grid[1] = (uint32_t)heads; dk.tg[0] = 32;
                dk.args = {mtl::Arg::B(k_buf), mtl::Arg::V(gw)};
                ds.push_back(std::move(dk));
            }
            {
                mtl::Dispatch d;
                d.p = &p_attn; d.grid[0] = (uint32_t)S; d.grid[1] = (uint32_t)heads; d.tg[0] = 128;
                d.args = {mtl::Arg::B(q_buf), mtl::Arg::B(k_buf), mtl::Arg::B(v_buf), mtl::Arg::B(ao_buf), mtl::Arg::V(S)};
                ds.push_back(std::move(d));
            }
            push_mm(lp + ".self_attn.out_proj.weight", ao_buf, proj_buf, hidden, hidden, S);
            {
                mtl::Dispatch d;
                d.p = &p_ls_res_add; d.grid[0] = (uint32_t)((S * hidden + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(x_buf), mtl::Arg::B(proj_buf), tw(lp + ".self_attn.out_proj.bias"),
                          tw(lp + ".ls_1.gamma"), mtl::Arg::V(S), mtl::Arg::V(hidden)};
                ds.push_back(std::move(d));
            }
            // MLP
            {
                mtl::Dispatch d;
                d.p = &p_ln; d.grid[0] = (uint32_t)S; d.tg[0] = 256;
                d.args = {mtl::Arg::B(x_buf), tw(lp + ".ln_2.weight"), tw(lp + ".ln_2.bias"),
                          mtl::Arg::B(xn_buf), mtl::Arg::V(ln_eps)};
                ds.push_back(std::move(d));
            }
            push_mm(lp + ".mlp.c_fc.weight", xn_buf, mlp_h, proj_hidden, hidden, S);
            push_bias(mlp_h, lp + ".mlp.c_fc.bias", S, proj_hidden, 1);
            push_mm(lp + ".mlp.c_proj.weight", mlp_h, proj_buf, hidden, proj_hidden, S);
            {
                mtl::Dispatch d;
                d.p = &p_ls_res_add; d.grid[0] = (uint32_t)((S * hidden + 255) / 256); d.tg[0] = 256;
                d.args = {mtl::Arg::B(x_buf), mtl::Arg::B(proj_buf), tw(lp + ".mlp.c_proj.bias"),
                          tw(lp + ".ls_2.gamma"), mtl::Arg::V(S), mtl::Arg::V(hidden)};
                ds.push_back(std::move(d));
            }
        }

        // 2x2 merge -> proj_in -> GELU -> proj_out
        {
            mtl::Dispatch d;
            d.p = &p_merge2x2; d.grid[0] = (uint32_t)n_tokens; d.tg[0] = 256;
            d.args = {mtl::Arg::B(x_buf), mtl::Arg::B(merged), mtl::Arg::V(gw)};
            ds.push_back(std::move(d));
        }
        push_mm("model.visual.proj_in.weight", merged, p_in_out, proj_hidden, proj_hidden, n_tokens);
        {
            mtl::Dispatch d;
            d.p = &p_bias_gelu; d.grid[0] = (uint32_t)((n_tokens * proj_hidden + 255) / 256); d.tg[0] = 256;
            int hb = 0, dg = 1;
            d.args = {mtl::Arg::B(p_in_out), tw("model.visual.proj_in.weight"),
                      mtl::Arg::V(n_tokens), mtl::Arg::V(proj_hidden), mtl::Arg::V(hb), mtl::Arg::V(dg)};
            ds.push_back(std::move(d));
        }
        push_mm("model.visual.proj_out.weight", p_in_out, final_out, out_hidden, proj_hidden, n_tokens);

        double gpu_s = cx.run_graph(ds);
        fprintf(stderr, "[ink_metal_vis] encoded %dx%d -> %d tokens in %.1f ms (GPU)\n",
                pad_w, pad_h, n_tokens, gpu_s * 1e3);

        out_embeds.resize((size_t)n_tokens * out_hidden);
        std::memcpy(out_embeds.data(), final_out.ptr, out_embeds.size() * sizeof(float));

        cx.free(rgb_buf); cx.free(pix_buf); cx.free(pemb_buf); cx.free(x_buf); cx.free(xn_buf);
        cx.free(q_buf); cx.free(k_buf); cx.free(v_buf); cx.free(ao_buf); cx.free(proj_buf);
        cx.free(mlp_h); cx.free(merged); cx.free(p_in_out); cx.free(final_out);
        return true;
    }
};

}  // namespace ink_metal
