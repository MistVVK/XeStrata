// ink_sy_vision.h - Native Intel Arc SYCL SG16 23-Layer BF16 ViT for Inkling-2-Mini-Vision.strata
// CRITICAL BATTLEMAGE B70 RULE: Strictly INK_REQD_SG16 (zero SG8!).
// 100% MIT clean-room implementation.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "ink_sy_codecs.h"
#include "ink_sy_xmx.h"
#include "ink_sy_common.h"

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

namespace ink_sycl {

struct InkSyVisTensor {
    uint64_t off = 0, bytes = 0;
    const uint16_t* dev_ptr = nullptr;
};

struct InkSyVision {
    int hidden = 1024, layers = 23, heads = 16, patch = 20, merge = 2;
    int proj_hidden = 4096, out_hidden = 2304, pos_grid = 32;
    float ln_eps = 1e-5f;
    uint8_t* w_slab = nullptr;
    std::unordered_map<std::string, InkSyVisTensor> T;
    bool loaded = false;

    ~InkSyVision() { dfree(w_slab); }

    bool load(const std::string& path, std::string& err) {
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) { err = "cannot open vision shard: " + path; return false; }
        uint8_t frame[4096];
        if (std::fread(frame, 1, 4096, fp) != 4096) { std::fclose(fp); err = "short read vision header"; return false; }
        strata::artifact::StrataSectionEntry secs[6]{};
        for (int i = 0; i < 6; ++i) std::memcpy(&secs[i], frame + 0x10 + i * 64, 64);

        std::string mjson;
        if (!read_section0_zstd(fp, secs[0].file_offset, secs[0].byte_length, mjson, err)) { std::fclose(fp); return false; }
        try {
            ink2::JVal root = ink2::json_parse(mjson);
            const ink2::JVal& vc = root.at("vision_config");
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
                InkSyVisTensor vt;
                vt.off = (uint64_t)t.at("offset").i();
                vt.bytes = (uint64_t)t.at("bytes").i();
                T[t.at("name").str()] = vt;
            }
        } catch (const std::exception& e) {
            std::fclose(fp); err = std::string("vision manifest: ") + e.what(); return false;
        }

        // Vision shard is ~636 MB -> single USM device slab!
        std::vector<uint8_t> raw(secs[1].byte_length);
        if (!seek_abs(fp, secs[1].file_offset) || std::fread(raw.data(), 1, raw.size(), fp) != raw.size()) {
            std::fclose(fp); err = "short read vision Section 1"; return false;
        }
        std::fclose(fp);

        w_slab = dalloc<uint8_t>(raw.size());
        h2d(w_slab, raw.data(), raw.size());
        wait();
        for (auto& kv : T) kv.second.dev_ptr = reinterpret_cast<const uint16_t*>(w_slab + kv.second.off);
        loaded = true;
        fprintf(stderr, "[ink_sycl_vis] loaded %s (%zu BF16 tensors, %.1f MB)\n",
                path.c_str(), T.size(), raw.size() / 1e6);
        return true;
    }

    const uint16_t* tw(const std::string& n) const {
        auto it = T.find(n);
        if (it == T.end()) { fprintf(stderr, "[ink_sycl_vis] missing %s\n", n.c_str()); std::exit(2); }
        return it->second.dev_ptr;
    }

    // Encode pre-normalized [pad_h, pad_w, 3] RGB pixels (in [-1, 1]) into [n_tokens, 2304] embeddings!
    bool encode_rgb_normalized(const float* h_rgb, int pad_w, int pad_h,
                               std::vector<float>& out_embeds, int& n_tokens, std::string& err) {
        if (!loaded) { err = "vision shard not loaded"; return false; }
        const int gw = pad_w / patch, gh = pad_h / patch;
        const int Np = gw * gh, S = 1 + Np;
        n_tokens = (gh / merge) * (gw / merge);

        const double t0 = now_s();
        float* rgb_dev   = dalloc<float>((size_t)pad_h * pad_w * 3);
        float* pix_dev   = dalloc<float>((size_t)Np * 1200);
        float* pemb_dev  = dalloc<float>((size_t)Np * hidden);
        float* x_dev     = dalloc<float>((size_t)S * hidden);
        float* xn_dev    = dalloc<float>((size_t)S * hidden);
        float* q_dev     = dalloc<float>((size_t)S * hidden);
        float* k_dev     = dalloc<float>((size_t)S * hidden);
        float* v_dev     = dalloc<float>((size_t)S * hidden);
        float* ao_dev    = dalloc<float>((size_t)S * hidden);
        float* proj_dev  = dalloc<float>((size_t)S * hidden);
        float* mlp_h_dev = dalloc<float>((size_t)S * proj_hidden);
        float* mrg_dev   = dalloc<float>((size_t)n_tokens * proj_hidden);
        float* pin_dev   = dalloc<float>((size_t)n_tokens * proj_hidden);
        float* out_dev   = dalloc<float>((size_t)n_tokens * out_hidden);

        h2d(rgb_dev, h_rgb, (size_t)pad_h * pad_w * 3 * sizeof(float));

        // 1. Patchify [Np, 1200] in [20, 20, 3] HWC order (exact CpuVisionEncoder layout)
        Q().parallel_for(NR(Np, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int p = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const int py = p / gw, px = p % gw;
            float* dst = pix_dev + (size_t)p * 1200;
            for (int idx = tid; idx < 1200; idx += 256) {
                const int px_idx = idx / 3, c = idx - px_idx * 3;
                const int dy = px_idx / 20, dx = px_idx - dy * 20;
                const int y = py * 20 + dy, x = px * 20 + dx;
                dst[idx] = rgb_dev[((size_t)y * pad_w + x) * 3 + c];
            }
        });

        auto mm = [&](const std::string& wn, const float* X, float* Y, int N, int K, int M) {
            if (ink_xmx_linear_bf16(tw(wn), N, K, X, Y, M)) return;
            if (M >= 16) k_gemm_bf16_sg16(tw(wn), N, K, X, Y, M);
            else k_linear_bf16_sg16(tw(wn), N, K, X, Y, M);
        };

        auto ln = [&](const float* X, const uint16_t* W, const uint16_t* B, float* Y) {
            const int H = hidden;
            const float eps = ln_eps;
            Q().parallel_for(NR(S, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
                float* sh = lmem<float, 16>(it);
                const int s = (int)it.get_group(0), tid = (int)it.get_local_id(0);
                const float* xp = X + (size_t)s * H;
                float sum = 0.0f;
                for (int d = tid; d < H; d += 256) sum += xp[d];
                const float mu = wg_sum16(it, sum, sh) / (float)H;
                float vs = 0.0f;
                for (int d = tid; d < H; d += 256) { float t = xp[d] - mu; vs += t * t; }
                const float inv = sycl::rsqrt(wg_sum16(it, vs, sh) / (float)H + eps);
                float* yp = Y + (size_t)s * H;
                for (int d = tid; d < H; d += 256) yp[d] = (xp[d] - mu) * inv * bf16_to_f32(W[d]) + bf16_to_f32(B[d]);
            });
        };

        auto bias_gelu = [&](float* X, const uint16_t* B, int M, int N, bool has_b, bool do_g) {
            const int tot = M * N;
            Q().parallel_for(NR((tot + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
                const int i = (int)it.get_global_id(0);
                if (i >= tot) return;
                float v = X[i];
                if (has_b) v += bf16_to_f32(B[i % N]);
                if (do_g) v = 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865475f));
                X[i] = v;
            });
        };

        auto ls_res_add = [&](float* X, const float* Y, const uint16_t* B, const uint16_t* G) {
            const int tot = S * hidden, H = hidden;
            Q().parallel_for(NR((tot + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
                const int i = (int)it.get_global_id(0);
                if (i >= tot) return;
                const int c = i % H;
                X[i] += (Y[i] + bf16_to_f32(B[c])) * bf16_to_f32(G[c]);
            });
        };

        auto rope2d = [&](float* X) {
            Q().parallel_for(NR2(16, (size_t)S, 32), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
                const int s = (int)it.get_group(0), hh = (int)it.get_group(1), tid = (int)it.get_local_id(1);
                if (s == 0) return;
                const int p = s - 1;
                const float pos_h = (float)((p / gw) + 1), pos_w = (float)((p % gw) + 1);
                float* xp = X + ((size_t)s * 16 + hh) * 64;
                const int half_ax = tid < 16 ? 0 : 1;
                const int i = tid & 15;
                const float ang = (half_ax == 0 ? pos_w : pos_h) / sycl::pow(10000.0f, (float)(2 * i) / 32.0f);
                const float c = sycl::cos(ang), sn = sycl::sin(ang);
                const int j = half_ax * 32 + 2 * i;
                const float a = xp[j], b = xp[j + 1];
                xp[j] = a * c - b * sn; xp[j + 1] = b * c + a * sn;
            });
        };

        // 2. Patch embedding + CLS + bilinear 32x32 pos_emb + ln_pre
        mm("model.visual.backbone.embeddings.patch_embedding.weight", pix_dev, pemb_dev, hidden, 1200, Np);
        {
            const uint16_t* cls_w = tw("model.visual.backbone.embeddings.class_embedding");
            const uint16_t* cpos_w = tw("model.visual.backbone.embeddings.cls_position_embedding");
            const uint16_t* pos_w = tw("model.visual.backbone.embeddings.position_embedding");
            const int H = hidden;
            Q().parallel_for(NR(S, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
                const int s = (int)it.get_group(0), tid = (int)it.get_local_id(0);
                float* dst = x_dev + (size_t)s * H;
                if (s == 0) {
                    for (int d = tid; d < H; d += 256) dst[d] = bf16_to_f32(cls_w[d]) + bf16_to_f32(cpos_w[d]);
                    return;
                }
                const int p = s - 1, py = p / gw, px = p % gw;
                const float fy = ((float)py + 0.5f) * (32.0f / (float)gh) - 0.5f;
                const float fx = ((float)px + 0.5f) * (32.0f / (float)gw) - 0.5f;
                const int y0 = sycl::clamp((int)sycl::floor(fy), 0, 31), y1 = sycl::min(y0 + 1, 31);
                const int x0 = sycl::clamp((int)sycl::floor(fx), 0, 31), x1 = sycl::min(x0 + 1, 31);
                const float wy = sycl::clamp(fy - (float)y0, 0.0f, 1.0f), wx = sycl::clamp(fx - (float)x0, 0.0f, 1.0f);
                const float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx;
                const float w10 = wy * (1.0f - wx), w11 = wy * wx;
                const float* src = pemb_dev + (size_t)p * H;
                for (int d = tid; d < H; d += 256) {
                    const float pe = w00 * bf16_to_f32(pos_w[(y0 * 32 + x0) * H + d])
                                   + w01 * bf16_to_f32(pos_w[(y0 * 32 + x1) * H + d])
                                   + w10 * bf16_to_f32(pos_w[(y1 * 32 + x0) * H + d])
                                   + w11 * bf16_to_f32(pos_w[(y1 * 32 + x1) * H + d]);
                    dst[d] = src[d] + pe;
                }
            });
        }
        ln(x_dev, tw("model.visual.backbone.ln_pre.weight"), tw("model.visual.backbone.ln_pre.bias"), x_dev);

        // 3. 23 Pre-LN Transformer blocks
        for (int il = 0; il < layers; ++il) {
            std::string lp = "model.visual.backbone.layers." + std::to_string(il);
            ln(x_dev, tw(lp + ".ln_1.weight"), tw(lp + ".ln_1.bias"), xn_dev);
            mm(lp + ".self_attn.q_proj.weight", xn_dev, q_dev, hidden, hidden, S);
            mm(lp + ".self_attn.k_proj.weight", xn_dev, k_dev, hidden, hidden, S);
            mm(lp + ".self_attn.v_proj.weight", xn_dev, v_dev, hidden, hidden, S);
            bias_gelu(q_dev, tw(lp + ".self_attn.q_proj.bias"), S, hidden, true, false);
            bias_gelu(k_dev, tw(lp + ".self_attn.k_proj.bias"), S, hidden, true, false);
            bias_gelu(v_dev, tw(lp + ".self_attn.v_proj.bias"), S, hidden, true, false);
            rope2d(q_dev);
            rope2d(k_dev);

            // Bidirectional full self-attention (8 SG16 = 128 threads, 4 dims/lane)
            Q().submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 1> sh_m(8, h);
                sycl::local_accessor<float, 1> sh_l(8, h);
                sycl::local_accessor<float, 1> sh_o(8 * 64, h);
                h.parallel_for(NR2(16, (size_t)S, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
                    const int s = (int)it.get_group(0), hh = (int)it.get_group(1), tid = (int)it.get_local_id(1);
                    const int sg = tid >> 4, lane = tid & 15;
                    float qv[4], oa[4] = {};
                    const float* qp = q_dev + ((size_t)s * 16 + hh) * 64 + lane * 4;
                    for (int u = 0; u < 4; ++u) qv[u] = qp[u] * 0.125f;
                    float mmx = -1e30f, ls = 0.0f;
                    for (int k = sg; k < S; k += 8) {
                        const float* kp = k_dev + ((size_t)k * 16 + hh) * 64 + lane * 4;
                        const float* vp = v_dev + ((size_t)k * 16 + hh) * 64 + lane * 4;
                        float dt = qv[0]*kp[0] + qv[1]*kp[1] + qv[2]*kp[2] + qv[3]*kp[3];
                        const float sc = sg_sum(it, dt);
                        const float mn = sycl::fmax(mmx, sc);
                        const float c = sycl::exp(mmx - mn), p = sycl::exp(sc - mn);
                        ls = ls * c + p; mmx = mn;
                        for (int u = 0; u < 4; ++u) oa[u] = oa[u] * c + p * vp[u];
                    }
                    if (lane == 0) { sh_m[sg] = mmx; sh_l[sg] = ls; }
                    for (int u = 0; u < 4; ++u) sh_o[sg * 64 + lane * 4 + u] = oa[u];
                    sycl::group_barrier(it.get_group());
                    if (sg == 0) {
                        float gm = -1e30f;
                        for (int i = 0; i < 8; ++i) gm = sycl::fmax(gm, sh_m[i]);
                        float gl = 0.0f, go[4] = {};
                        for (int i = 0; i < 8; ++i) {
                            const float w = sycl::exp(sh_m[i] - gm);
                            gl += sh_l[i] * w;
                            for (int u = 0; u < 4; ++u) go[u] += sh_o[i * 64 + lane * 4 + u] * w;
                        }
                        const float il = 1.0f / gl;
                        float* op = ao_dev + ((size_t)s * 16 + hh) * 64 + lane * 4;
                        for (int u = 0; u < 4; ++u) op[u] = go[u] * il;
                    }
                });
            });

            mm(lp + ".self_attn.out_proj.weight", ao_dev, proj_dev, hidden, hidden, S);
            ls_res_add(x_dev, proj_dev, tw(lp + ".self_attn.out_proj.bias"), tw(lp + ".ls_1.gamma"));

            ln(x_dev, tw(lp + ".ln_2.weight"), tw(lp + ".ln_2.bias"), xn_dev);
            mm(lp + ".mlp.c_fc.weight", xn_dev, mlp_h_dev, proj_hidden, hidden, S);
            bias_gelu(mlp_h_dev, tw(lp + ".mlp.c_fc.bias"), S, proj_hidden, true, true);
            mm(lp + ".mlp.c_proj.weight", mlp_h_dev, proj_dev, hidden, proj_hidden, S);
            ls_res_add(x_dev, proj_dev, tw(lp + ".mlp.c_proj.bias"), tw(lp + ".ls_2.gamma"));
        }

        // 4. 2x2 Spatial Merge -> proj_in -> GELU -> proj_out [2304]
        Q().parallel_for(NR(n_tokens, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const int mw = gw / 2, my = m / mw, mx = m % mw;
            const int s00 = 1 + (2 * my + 0) * gw + (2 * mx + 0);
            const int s01 = 1 + (2 * my + 0) * gw + (2 * mx + 1);
            const int s10 = 1 + (2 * my + 1) * gw + (2 * mx + 0);
            const int s11 = 1 + (2 * my + 1) * gw + (2 * mx + 1);
            float* dst = mrg_dev + (size_t)m * 4096;
            for (int d = tid; d < 1024; d += 256) {
                dst[0 * 1024 + d] = x_dev[(size_t)s00 * 1024 + d];
                dst[1 * 1024 + d] = x_dev[(size_t)s01 * 1024 + d];
                dst[2 * 1024 + d] = x_dev[(size_t)s10 * 1024 + d];
                dst[3 * 1024 + d] = x_dev[(size_t)s11 * 1024 + d];
            }
        });
        mm("model.visual.proj_in.weight", mrg_dev, pin_dev, proj_hidden, proj_hidden, n_tokens);
        bias_gelu(pin_dev, nullptr, n_tokens, proj_hidden, false, true);
        mm("model.visual.proj_out.weight", pin_dev, out_dev, out_hidden, proj_hidden, n_tokens);

        out_embeds.resize((size_t)n_tokens * out_hidden);
        d2h(out_embeds.data(), out_dev, out_embeds.size() * sizeof(float));
        wait();

        fprintf(stderr, "[ink_sycl_vis] encoded %dx%d -> %d tokens in %.1f ms (SYCL SG16)\n",
                pad_w, pad_h, n_tokens, (now_s() - t0) * 1e3);

        dfree(rgb_dev); dfree(pix_dev); dfree(pemb_dev); dfree(x_dev); dfree(xn_dev);
        dfree(q_dev); dfree(k_dev); dfree(v_dev); dfree(ao_dev); dfree(proj_dev);
        dfree(mlp_h_dev); dfree(mrg_dev); dfree(pin_dev); dfree(out_dev);
        return true;
    }

    bool encode_image_file(const std::string& img_path, std::vector<float>& out_embeds, int& n_tokens, std::string& err) {
        int w = 0, h = 0, ch = 0;
        unsigned char* raw = stbi_load(img_path.c_str(), &w, &h, &ch, 4);
        if (!raw) { err = "stbi_load failed: " + img_path; return false; }
        int tw_px = w, th_px = h;
        const int max_side = 4096;
        if (tw_px > max_side || th_px > max_side) {
            const double sc = (double)max_side / (double)std::max(tw_px, th_px);
            tw_px = std::max(1, (int)std::floor(tw_px * sc));
            th_px = std::max(1, (int)std::floor(th_px * sc));
        }
        const int factor = patch * merge;
        const int pad_w = ((tw_px + factor - 1) / factor) * factor;
        const int pad_h = ((th_px + factor - 1) / factor) * factor;
        const float pad_norm = (-1.0f / 255.0f - 0.5f) / 0.5f;
        std::vector<float> rgb((size_t)pad_h * pad_w * 3, pad_norm);
        for (int y = 0; y < th_px; ++y) {
            const int sy = std::min(h - 1, (int)((int64_t)y * h / th_px));
            for (int x = 0; x < tw_px; ++x) {
                const int sx = std::min(w - 1, (int)((int64_t)x * w / tw_px));
                const unsigned char* px = raw + ((size_t)sy * w + sx) * 4;
                const float alpha = px[3] * (1.0f / 255.0f);
                const float bg = ((((x >> 3) + (y >> 3)) & 1) == 0 ? 204.0f : 153.0f) * (1.0f / 255.0f);
                for (int c = 0; c < 3; ++c) {
                    const float comp = (px[c] * (1.0f / 255.0f)) * alpha + bg * (1.0f - alpha);
                    rgb[((size_t)y * pad_w + x) * 3 + c] = (comp - 0.5f) * 2.0f;
                }
            }
        }
        stbi_image_free(raw);
        return encode_rgb_normalized(rgb.data(), pad_w, pad_h, out_embeds, n_tokens, err);
    }
};

}  // namespace ink_sycl
