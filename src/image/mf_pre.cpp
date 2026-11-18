// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_pre.cpp - strata-image-mf-pre: Mage-Flow's learning preprocessing on the GPU, the two values the
// trainer reads back from disk (docs/LORA_TRAIN.md "The latent and text-embedding disk caches").  Mage-VAE's own
// encoder (src/image/mf_vae.cpp) and the text reader on the inference's own states path (src/image/text_encoder.cpp,
// Qwen3-VL-4B with its final norm and plain positions) make them; nothing here recomputes either.
//
//   strata-image-mf-pre vae  VAE.safetensors DIR          DIR/pix.{bin,json} -> DIR/latent.{bin,json}
//   strata-image-mf-pre text TEXT_ENCODER.gguf DIR [DROP] DIR/prompt_ids.{bin,json} (+ DIR/uncond_ids.*)
//                                                          -> DIR/prompt_embeds[32].*, DIR/uncond_embeds[32].*
//
// The dumps are the image engine's own form (tools/image/qi21_ref.py's save): <name>.bin with <name>.json listing
// {"type":"f32"|"i32"|"bf16","shape":[fastest, ...]}.  `pix` is the resized image [3][H][W] FP32 in -1 .. 1 (what
// the encoder takes); the outputs are:
//
//   latent   the DiT's latent [128][h][w] FP32 - the posterior's mean (MfVae::encode with no eps).  Mage-VAE's
//            2 x 2 packing is inside its encoder; the Flux2 VAE's (MageTrail's) encode packs and normalizes by its
//            batch norm the same way, so either is already the DiT's latent space.
//   latent_sd the posterior's standard deviation [128][h][w] FP32, Mage-VAE's only (the Flux2 VAE's encode gives
//            the mean alone).
//   embeds   the states after the final norm with the system turn's DROP rows dropped, as bf16 bits in uint16 (the
//            cache's form); embeds32 is the FP32 the engine computed, so a reference can tell the cache's rounding
//            from the engine's value.
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mf_vae.hpp"
#include "strata/image/anima_dit.hpp"
#include "strata/image/anima_pipeline.hpp"
#include "strata/image/moss_dit.hpp"
#include "strata/image/qi_vae.hpp"
#include "strata/image/text_encoder.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using strata::image::MfVae;
using strata::image::TextEncoder;

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

struct Var {
    std::vector<int64_t> shape;   // outer .. fastest
    std::vector<uint8_t> raw;
    std::string type = "f32";
    const float* f() const { return reinterpret_cast<const float*>(raw.data()); }
    const int32_t* i() const { return reinterpret_cast<const int32_t*>(raw.data()); }
    int64_t dim(int64_t k) const { return shape.at((size_t) k); }
};

/// a dump read back: the json lists the fastest dim first, so the shape is reversed to outer .. fastest
Var load(const std::string& dir, const std::string& name, bool optional = false) {
    Var v;
    std::ifstream j(dir + "/" + name + ".json");
    if (!j) {
        if (optional) return v;
        throw std::runtime_error("no " + name + ".json in " + dir);
    }
    std::stringstream ss;
    ss << j.rdbuf();
    const std::string s = ss.str();
    if (const size_t t = s.find("\"type\""); t != std::string::npos) {
        const size_t q = s.find('"', s.find(':', t) + 1), e = s.find('"', q + 1);
        if (q != std::string::npos && e != std::string::npos) v.type = s.substr(q + 1, e - q - 1);
    }
    if (const size_t b = s.find("\"shape\""); b != std::string::npos) {
        const size_t o = s.find('[', b), c = s.find(']', o);
        std::stringstream ns(s.substr(o + 1, c - o - 1));
        std::string tok;
        while (std::getline(ns, tok, ','))
            if (tok.find_first_not_of(" \t\n") != std::string::npos)
                v.shape.insert(v.shape.begin(), std::stoll(tok));
    }
    std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
    if (!f) {
        if (optional) return v;
        throw std::runtime_error("no " + name + ".bin in " + dir);
    }
    v.raw.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return v;
}

void save(const std::string& dir, const std::string& name, const void* p, int64_t bytes, const std::string& type,
          const std::vector<int64_t>& fastest_first) {
    std::ofstream b(dir + "/" + name + ".bin", std::ios::binary);
    b.write(reinterpret_cast<const char*>(p), (std::streamsize) bytes);
    std::ofstream j(dir + "/" + name + ".json");
    j << "{\"type\":\"" << type << "\",\"shape\":[";
    for (size_t k = 0; k < fastest_first.size(); ++k) j << (k ? "," : "") << fastest_first[k];
    j << "]}";
}

float* put(const std::vector<float>& h) {
    float* d = nullptr;
    if (!strata::gpu::alloc_device(&d, h.size() * 4)) throw std::runtime_error("GPU memory");
    if (!strata::gpu::copy(d, h.data(), h.size() * 4)) throw std::runtime_error("GPU copy");
    return d;
}

std::vector<float> fetch(const float* d, size_t n) {
    std::vector<float> h(n);
    if (!strata::gpu::copy(h.data(), d, n * 4)) throw std::runtime_error("GPU read");
    return h;
}

/// [h w][128] (the encoder's and the DiT's token order) -> [128][h][w] (the cache's channels-first latent)
std::vector<float> channels_first(const std::vector<float>& z, int64_t hw, int64_t C) {
    std::vector<float> y(z.size());
    for (int64_t p = 0; p < hw; ++p)
        for (int64_t c = 0; c < C; ++c) y[(size_t) (c * hw + p)] = z[(size_t) (p * C + c)];
    return y;
}

int vae(const std::string& path, const std::string& dir) {
    void* s = strata::core::Runtime::get().create_stream();
    MfVae v;
    std::string err;
    if (!v.open(path, s, err)) throw std::runtime_error(err);
    if (!v.open_encoder(err)) throw std::runtime_error(err);
    const Var pix = load(dir, "pix");
    if (pix.type != "f32" || pix.shape.size() != 3 || pix.dim(0) != 3)
        throw std::runtime_error("pix must be [3][H][W] FP32 (the VAE's input), not " + pix.type + " of " +
                                 std::to_string(pix.shape.size()) + " dims");
    const int64_t H = pix.dim(1), W = pix.dim(2), C = 128;
    if (H % 16 != 0 || W % 16 != 0)
        throw std::runtime_error("pix is not a size the VAE takes: both sides must be multiples of 16, got " +
                                 std::to_string(W) + " x " + std::to_string(H));
    const int64_t h = H / 16, w = W / 16;
    std::vector<float> rgb(pix.f(), pix.f() + pix.raw.size() / 4);
    if ((int64_t) rgb.size() != 3 * H * W) throw std::runtime_error("pix's bytes are not 3*H*W floats");
    float* src = put(rgb);
    float* z = nullptr;
    if (!strata::gpu::alloc_device(&z, (size_t) h * w * C * 4)) throw std::runtime_error("GPU memory");
    v.encode(src, h, w, nullptr, z);   // warm
    strata::gpu::stream_sync(s);
    const auto t0 = std::chrono::steady_clock::now();
    v.encode(src, h, w, nullptr, z);
    strata::gpu::stream_sync(s);
    const auto y = channels_first(fetch(z, (size_t) h * w * C), h * w, C);
    std::printf("latent %lld x %lld -> 128 x %lld x %lld: %.1f ms (encoder scratch %.0f MiB)\n", (long long) W,
                (long long) H, (long long) w, (long long) h, 1e3 * seconds_since(t0),
                (double) v.encode_scratch_bytes(h, w) / 1048576.0);
    save(dir, "latent", y.data(), (int64_t) y.size() * 4, "f32", {w, h, C});
    // the posterior's standard deviation, exp(logvar / 2) as encode takes it: a training step draws its latent
    // from the posterior (ai-toolkit samples latent_dist each step), the cache keeping the mean and this.  The Flux2
    // VAE's encode (MageTrail's) gives its mean alone, so its cache has no deviation and a step reads the mean
    if (v.flux2()) {
        strata::gpu::free(z);
        strata::gpu::free(src);
        return 0;
    }
    float* mom = nullptr;
    if (!strata::gpu::alloc_device(&mom, (size_t) h * w * 2 * C * 4)) throw std::runtime_error("GPU memory");
    v.encode_moments(src, h, w, mom);
    strata::gpu::stream_sync(s);
    const auto m = fetch(mom, (size_t) h * w * 2 * C);
    std::vector<float> sd((size_t) (h * w * C));
    for (int64_t p = 0; p < h * w; ++p)
        for (int64_t c = 0; c < C; ++c)
            sd[(size_t) (p * C + c)] = std::exp(0.5f * std::clamp(m[(size_t) (p * 2 * C + C + c)], -20.f, 10.f));
    const auto sdf = channels_first(sd, h * w, C);
    save(dir, "latent_sd", sdf.data(), (int64_t) sdf.size() * 4, "f32", {w, h, C});
    strata::gpu::free(mom);
    strata::gpu::free(z);
    strata::gpu::free(src);
    return 0;
}

/// one prompt's states, the system turn's rows dropped: the cache's bf16 bits and the FP32 behind them
void text_one(TextEncoder& te, const std::string& dir, const std::string& base, int64_t drop, int64_t C) {
    const Var ids = load(dir, base + "_ids", base != "prompt");
    if (ids.raw.empty()) return;
    if (ids.type != "i32" || ids.shape.size() != 1) throw std::runtime_error(base + "_ids must be [n] i32");
    const std::vector<int32_t> v(ids.i(), ids.i() + ids.dim(0));
    if (ids.dim(0) <= drop)
        throw std::runtime_error(base + ": no rows left after the system turn's " + std::to_string(drop));
    std::string err;
    te.encode(v, err);   // warm
    const auto t0 = std::chrono::steady_clock::now();
    const auto h = te.encode(v, err);
    if (h.empty()) throw std::runtime_error(base + ": " + err);
    if ((int64_t) h.size() / C != (int64_t) v.size())
        throw std::runtime_error(base + ": the states are not one row per id");
    const std::vector<float> kept(h.begin() + (std::ptrdiff_t) (drop * C), h.end());
    const int64_t rows = (int64_t) kept.size() / C;
    std::printf("%s: %lld ids -> %lld rows x %lld: %.1f ms\n", base.c_str(), (long long) v.size(), (long long) rows,
                (long long) C, 1e3 * seconds_since(t0));
    std::vector<uint16_t> bits(kept.size());
    for (size_t k = 0; k < kept.size(); ++k) bits[k] = strata::kernels::bf16_from_f32(kept[k]);
    save(dir, base + "_embeds", bits.data(), (int64_t) bits.size() * 2, "bf16", {C, rows});
    save(dir, base + "_embeds32", kept.data(), (int64_t) kept.size() * 4, "f32", {C, rows});
}

int text(const std::string& path, const std::string& dir, int64_t drop) {
    TextEncoder te;
    std::string err;
    if (!te.open(path, 16384, err)) throw std::runtime_error(err);
    te.set_final_norm(true);      // Mage-Flow takes the states after the final RMS norm
    te.set_flat_positions(true);  // every token's position is its index, as the reference's encoder passes them
    text_one(te, dir, "prompt", drop, te.hidden());
    text_one(te, dir, "uncond", drop, te.hidden());
    return 0;
}

/// MOSS-Image's text rows as its DiT reads them: each listed work dir's prompt_ids and uncond_ids through the text
/// encoder (Qwen3.5-9B, every row: MOSS-Image drops no system turn) and then the DiT's text embedder (set_context),
/// whose output a training step takes as it is (no time in it).  Both models are opened once for the whole list, the
/// text encoder closed before the DiT opens.
int moss_text(const std::string& te_path, const std::string& dit_path, const std::string& list) {
    void* s = strata::core::Runtime::get().create_stream();
    std::vector<std::string> dirs;
    {
        std::ifstream f(list);
        std::string line;
        while (std::getline(f, line))
            if (!line.empty()) dirs.push_back(line);
    }
    if (dirs.empty()) throw std::runtime_error(list + ": no work dirs");
    std::vector<std::vector<float>> states(dirs.size() * 2);
    std::string err;
    {
        TextEncoder te;
        if (!te.open(te_path, 4096, err)) throw std::runtime_error(err);
        for (size_t d = 0; d < dirs.size(); ++d)
            for (int b = 0; b < 2; ++b) {
                const Var ids = load(dirs[d], b == 0 ? "prompt_ids" : "uncond_ids", b == 1);
                if (ids.raw.empty()) continue;
                if (ids.type != "i32" || ids.shape.size() != 1) throw std::runtime_error(dirs[d] + ": ids not [n] i32");
                const std::vector<int32_t> v(ids.i(), ids.i() + ids.dim(0));
                states[d * 2 + b] = te.encode(v, err);
                if (states[d * 2 + b].empty()) throw std::runtime_error(err);
            }
    }
    strata::image::MossDit dit;
    if (!dit.open(dit_path, s, err, false, false)) throw std::runtime_error(err);
    const int64_t C = dit.hidden(), H = 4096;
    for (size_t d = 0; d < dirs.size(); ++d)
        for (int b = 0; b < 2; ++b) {
            const std::vector<float>& st = states[d * 2 + b];
            if (st.empty()) continue;
            const int64_t rows = (int64_t) st.size() / H;
            dit.set_context(0, st.data(), rows);
            strata::gpu::stream_sync(s);
            std::vector<uint16_t> bits((size_t) (rows * C));
            if (!strata::gpu::copy(bits.data(), dit.text_rows(0), bits.size() * 2)) throw std::runtime_error("copy");
            std::vector<float> f32(bits.size());
            for (size_t k = 0; k < bits.size(); ++k) f32[k] = strata::kernels::f32_from_bf16(bits[k]);
            const std::string base = b == 0 ? "prompt" : "uncond";
            save(dirs[d], base + "_embeds", bits.data(), (int64_t) bits.size() * 2, "bf16", {C, rows});
            save(dirs[d], base + "_embeds32", f32.data(), (int64_t) f32.size() * 4, "f32", {C, rows});
            std::printf("%s %s: %lld rows x %lld\n", dirs[d].c_str(), base.c_str(), (long long) rows, (long long) C);
        }
    return 0;
}

/// Anima's latent (Wan 2.1's VAE, QiVae): the posterior's mean and standard deviation as the DiT's tokens of 2 x 2
/// patches, written channels-first [64][h][w] as `latent` and `latent_sd` (the export puts them back to tokens).
int anima_vae(const std::string& path, const std::string& dir) {
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::QiVae v;
    std::string err;
    if (!v.open(path, s, err)) throw std::runtime_error(err);
    const Var pix = load(dir, "pix");
    if (pix.type != "f32" || pix.shape.size() != 3 || pix.dim(0) != 3)
        throw std::runtime_error("pix must be [3][H][W] FP32 (-1 .. 1)");
    const int64_t H = pix.dim(1), W = pix.dim(2), h = H / 16, w = W / 16, L = 16, T = 4 * L;
    if (H % 16 != 0 || W % 16 != 0) throw std::runtime_error("pix's sides must be multiples of 16");
    std::vector<float> rgba((size_t) (4 * H * W), 1.f);   // an opaque alpha, the encoder's 0 .. 1
    for (int64_t i = 0; i < 3 * H * W; ++i) rgba[(size_t) i] = pix.f()[i] * 0.5f + 0.5f;
    float* src = put(rgba);
    float *z = nullptr, *zs = nullptr, *x = nullptr;
    if (!strata::gpu::alloc_device(&z, (size_t) (T * h * w) * 4) || !strata::gpu::alloc_device(&zs, (size_t) (T * h * w) * 4) ||
        !strata::gpu::alloc_device(&x, (size_t) (T * h * w) * 4))
        throw std::runtime_error("GPU memory");
    if (!v.encode(src, H, W, z, zs)) throw std::runtime_error("the VAE has no encoder");
    for (int k = 0; k < 2; ++k) {
        strata::image::anima_pipeline_detail::latent_to_tokens(k == 0 ? z : zs, h, w, L, x, s);
        strata::gpu::stream_sync(s);
        const auto y = channels_first(fetch(x, (size_t) (T * h * w)), h * w, T);
        save(dir, k == 0 ? "latent" : "latent_sd", y.data(), (int64_t) y.size() * 4, "f32", {w, h, T});
    }
    std::printf("anima latent %lld x %lld -> %lld x %lld x %lld\n", (long long) W, (long long) H, (long long) T,
                (long long) h, (long long) w);
    for (float* p : {z, zs, x, src}) strata::gpu::free(p);
    return 0;
}

/// Anima's context rows as its blocks' cross-attention reads them: each listed work dir's prompt and uncond (Qwen3 ids
/// through the text encoder's final norm, then the DiT's LLM adapter with the T5 ids and their weights), every one of
/// the rows the blocks take (the zero rows after the prompt's included), FP32 from the FP16 rows times their scales.
/// The text encoder is closed before the DiT opens.
int anima_text(const std::string& te_path, const std::string& dit_path, const std::string& list) {
    void* s = strata::core::Runtime::get().create_stream();
    std::vector<std::string> dirs;
    {
        std::ifstream f(list);
        for (std::string line; std::getline(f, line);)
            if (!line.empty()) dirs.push_back(line);
    }
    if (dirs.empty()) throw std::runtime_error(list + ": no work dirs");
    std::vector<std::vector<float>> states(dirs.size() * 2);
    std::string err;
    int64_t H = 0;
    {
        TextEncoder te;
        if (!te.open(te_path, 4096, err)) throw std::runtime_error(err);   // Anima's longest prompt
        H = te.hidden();
        for (size_t d = 0; d < dirs.size(); ++d)
            for (int b = 0; b < 2; ++b) {
                const Var ids = load(dirs[d], b == 0 ? "prompt_ids" : "uncond_ids", b == 1);
                if (ids.raw.empty()) continue;
                const std::vector<int32_t> v(ids.i(), ids.i() + ids.dim(0));
                states[d * 2 + b] = te.encode(v, err);
                if (states[d * 2 + b].empty()) throw std::runtime_error(dirs[d] + ": " + err);
            }
    }
    strata::image::AnimaDit dit;
    if (!dit.open(dit_path, s, err, false, false)) throw std::runtime_error(err);
    const int64_t C = dit.context_cols();
    for (size_t d = 0; d < dirs.size(); ++d)
        for (int b = 0; b < 2; ++b) {
            const std::vector<float>& st = states[d * 2 + b];
            if (st.empty()) continue;
            const std::string base = b == 0 ? "prompt" : "uncond";
            const Var t5 = load(dirs[d], base + "_t5");
            const Var t5w = load(dirs[d], base + "_t5w");
            const std::vector<int32_t> ids(t5.i(), t5.i() + t5.dim(0));
            const std::vector<float> wts(t5w.f(), t5w.f() + t5w.dim(0));
            dit.set_contexts({{0, st.data(), (int64_t) st.size() / H, &ids, &wts}});
            strata::gpu::stream_sync(s);
            const int64_t rows = dit.context_rows(0);
            std::vector<uint16_t> c16((size_t) (rows * C));
            std::vector<float> rs((size_t) rows);
            if (!strata::gpu::copy(c16.data(), dit.context16(0), c16.size() * 2) ||
                !strata::gpu::copy(rs.data(), dit.context_scales(0), rs.size() * 4))
                throw std::runtime_error("copy");
            std::vector<float> f32(c16.size());
            std::vector<uint16_t> bits(c16.size());
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < C; ++c) {
                    const size_t k = (size_t) (r * C + c);
                    f32[k] = strata::kernels::f32_from_f16(c16[k]) * rs[(size_t) r];
                    bits[k] = strata::kernels::bf16_from_f32(f32[k]);
                }
            save(dirs[d], base + "_embeds", bits.data(), (int64_t) bits.size() * 2, "bf16", {C, rows});
            save(dirs[d], base + "_embeds32", f32.data(), (int64_t) f32.size() * 4, "f32", {C, rows});
            std::printf("%s %s: %lld rows x %lld\n", dirs[d].c_str(), base.c_str(), (long long) rows, (long long) C);
        }
    return 0;
}

/// Qwen-Image-2.1's caches, as its trainer reads them back (qi_train_latent, qi_train_states): the list's cut images
/// through the VAE's encoder (the posterior's mean and standard deviation) and its prompts through the text encoder
/// (the states before its final norm, the system turn's rows dropped), each to the flat file the list names.
int qi_prep(const std::string& vae_path, const std::string& te_path, const std::string& list_path) {
    std::ifstream list(list_path);
    if (!list) throw std::runtime_error("cannot read " + list_path);
    std::vector<std::vector<std::string>> imgs, txts;
    for (std::string line; std::getline(list, line);) {
        // tab-separated: a cache path can hold spaces (a Windows user's folder), never a tab.  The last field of a
        // line written on Windows holds the CR of CRLF: whitespace splitting used to eat it, getline does not.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ls(line);
        std::vector<std::string> w;
        for (std::string t; std::getline(ls, t, '\t');) w.push_back(t);
        if (w.empty()) continue;
        if (w[0] == "img" && w.size() == 5) imgs.push_back(w);
        else if (w[0] == "txt" && w.size() == 4) txts.push_back(w);
        else throw std::runtime_error("bad list line: " + line);
    }
    void* s = strata::core::Runtime::get().create_stream();
    std::string err;
    if (!imgs.empty()) {   // the VAE first, then the text encoder: one of them on the GPU at a time
        strata::image::QiVae vae;
        if (!vae.open(vae_path, s, err)) throw std::runtime_error(err);
        for (const auto& w : imgs) {
            const int64_t W = std::stoll(w[2]), H = std::stoll(w[3]), HW = W * H;
            std::vector<uint8_t> px;
            {
                std::ifstream f(w[1], std::ios::binary);
                px.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            }
            if ((int64_t) px.size() != HW * 3) throw std::runtime_error("bad image file " + w[1]);
            std::vector<float> rgba((size_t) (4 * HW), 1.0f);   // an opaque alpha, as ai-toolkit gives an RGB image
            for (int64_t q = 0; q < HW; ++q)
                for (int c = 0; c < 3; ++c) rgba[(size_t) (c * HW + q)] = (float) px[(size_t) (q * 3 + c)] / 255.f;
            const int64_t n = 64 * HW / 256;
            float *d = nullptr, *z = nullptr, *sd = nullptr;
            if (!strata::gpu::alloc_device(&d, rgba.size() * 4) || !strata::gpu::alloc_device(&z, (size_t) n * 4) ||
                !strata::gpu::alloc_device(&sd, (size_t) n * 4))
                throw std::runtime_error("GPU memory for an image");
            strata::gpu::copy(d, rgba.data(), rgba.size() * 4);
            if (!vae.encode(d, H, W, z, sd)) throw std::runtime_error("the VAE has no encoder");
            std::vector<float> out((size_t) (2 + 2 * n));
            const int64_t gw = W / 16, gh = H / 16;   // the latent's grid
            out[0] = (float) gw;
            out[1] = (float) gh;
            strata::gpu::copy(out.data() + 2, z, (size_t) n * 4);
            strata::gpu::copy(out.data() + 2 + n, sd, (size_t) n * 4);
            strata::gpu::free(d);
            strata::gpu::free(z);
            strata::gpu::free(sd);
            vae.drop_scratch();
            std::ofstream o(w[4], std::ios::binary);
            o.write(reinterpret_cast<const char*>(out.data()), (std::streamsize) (out.size() * 4));
            if (!o) throw std::runtime_error("cannot write " + w[4]);
            std::printf("img %s %lldx%lld\n", w[4].c_str(), (long long) W, (long long) H);
        }
    }
    if (!txts.empty()) {
        strata::image::TextEncoder te;
        strata::gpu::stream_sync(s);   // the encoder's own stream: the VAE's work first
        if (!te.open(te_path, 16384, err)) throw std::runtime_error(err);
        te.set_final_norm(false);   // Qwen-Image's context: before the final norm
        for (const auto& w : txts) {
            std::vector<int32_t> ids;
            {
                std::ifstream f(w[1], std::ios::binary);
                const auto bytes = (size_t) std::filesystem::file_size(w[1]);
                ids.resize(bytes / 4);
                f.read(reinterpret_cast<char*>(ids.data()), (std::streamsize) bytes);
            }
            const int64_t drop = std::stoll(w[2]);
            std::vector<float> st = te.encode(ids, err);
            if (st.empty()) throw std::runtime_error(err);
            const int64_t D = te.hidden(), n = (int64_t) ids.size() - drop;
            if (D != 4096 || n <= 0) throw std::runtime_error("the text encoder's states do not fit " + w[1]);
            std::vector<float> out((size_t) (1 + n * D));
            out[0] = (float) n;
            std::copy(st.begin() + drop * D, st.end(), out.begin() + 1);
            std::ofstream o(w[3], std::ios::binary);
            o.write(reinterpret_cast<const char*>(out.data()), (std::streamsize) (out.size() * 4));
            if (!o) throw std::runtime_error("cannot write " + w[3]);
            std::printf("txt %s %lld rows\n", w[3].c_str(), (long long) n);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string what = argc > 1 ? argv[1] : "";
    try {
        if (what == "vae" && argc == 4) return vae(argv[2], argv[3]);
        if (what == "text" && (argc == 4 || argc == 5)) return text(argv[2], argv[3], argc == 5 ? std::stoll(argv[4]) : 34);
        if (what == "moss-text" && argc == 5) return moss_text(argv[2], argv[3], argv[4]);
        if (what == "anima-vae" && argc == 4) return anima_vae(argv[2], argv[3]);
        if (what == "anima-text" && argc == 5) return anima_text(argv[2], argv[3], argv[4]);
        if (what == "qi-prep" && argc == 5) return qi_prep(argv[2], argv[3], argv[4]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-mf-pre: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr,
                 "usage: %s vae VAE.safetensors DIR | text TEXT_ENCODER.gguf DIR [DROP] | "
                 "moss-text TEXT_ENCODER.gguf DIT_DIR LIST | anima-vae VAE.safetensors DIR | "
                 "anima-text TEXT_ENCODER.gguf DIT.safetensors LIST | qi-prep VAE TEXT_ENCODER.gguf LIST\n", argv[0]);
    return 2;
}
