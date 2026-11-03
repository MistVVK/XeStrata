// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_pre.cpp - strata-image-mf-pre: Mage-Flow's learning preprocessing on the GPU, the two values the
// trainer reads back from disk (docs/LORA_TRAIN.md "The latent and text-embedding disk caches").  Mage-VAE's own
// encoder (src/image/mf_vae.cpp) and the text reader on the inference's own states path (src/image/qi_text.cpp,
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
//   latent   the VAE's raw output [128][h][w] FP32 - the posterior's mean (MfVae::encode with no eps).  Mage-VAE's
//            2 x 2 packing is inside its encoder, so this is already the DiT's latent space; there is no batch norm
//            or statistics left for the trainer to fold in (a Flux2 VAE's raw latent is another card's).
//   embeds   the states after the final norm with the system turn's DROP rows dropped, as bf16 bits in uint16 (the
//            cache's form); embeds32 is the FP32 the engine computed, so a reference can tell the cache's rounding
//            from the engine's value.
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mf_vae.hpp"
#include "strata/image/qi_text.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using strata::image::MfVae;
using strata::image::QiText;

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
                v.shape.insert(v.shape.begin(), std::atoll(tok.c_str()));
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
    if (v.flux2())
        throw std::runtime_error("Mage-VAE only here: a Flux2 VAE's raw latent is 32 channels at 1/8, and its cache "
                                 "is MageTrail's (card 9)");
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
    strata::gpu::free(z);
    strata::gpu::free(src);
    return 0;
}

/// one prompt's states, the system turn's rows dropped: the cache's bf16 bits and the FP32 behind them
void text_one(QiText& te, const std::string& dir, const std::string& base, int64_t drop, int64_t C) {
    const Var ids = load(dir, base + "_ids", base != "prompt");
    if (ids.raw.empty()) return;
    if (ids.type != "i32" || ids.shape.size() != 1) throw std::runtime_error(base + "_ids must be [n] i32");
    const std::vector<int32_t> v(ids.i(), ids.i() + ids.dim(0));
    if (ids.dim(0) <= drop)
        throw std::runtime_error(base + ": no rows left after the system turn's " + std::to_string(drop));
    te.encode(v);   // warm
    const auto t0 = std::chrono::steady_clock::now();
    const auto h = te.encode(v);
    if ((int64_t) h.size() / C != (int64_t) v.size())
        throw std::runtime_error(base + ": the states are not one row per id");
    const std::vector<float> kept(h.begin() + (size_t) (drop * C), h.end());
    const int64_t rows = (int64_t) kept.size() / C;
    std::printf("%s: %lld ids -> %lld rows x %lld: %.1f ms\n", base.c_str(), (long long) v.size(), (long long) rows,
                (long long) C, 1e3 * seconds_since(t0));
    std::vector<uint16_t> bits(kept.size());
    for (size_t k = 0; k < kept.size(); ++k) bits[k] = strata::kernels::bf16_from_f32(kept[k]);
    save(dir, base + "_embeds", bits.data(), (int64_t) bits.size() * 2, "bf16", {C, rows});
    save(dir, base + "_embeds32", kept.data(), (int64_t) kept.size() * 4, "f32", {C, rows});
}

int text(const std::string& path, const std::string& dir, int64_t drop) {
    void* s = strata::core::Runtime::get().create_stream();
    QiText te;
    std::string err;
    if (!te.open(path, s, err)) throw std::runtime_error(err);
    te.set_final_norm(true);      // Mage-Flow takes the states after the final RMS norm
    te.set_flat_positions(true);  // every token's position is its index, as the reference's encoder passes them
    text_one(te, dir, "prompt", drop, te.hidden());
    text_one(te, dir, "uncond", drop, te.hidden());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string what = argc > 1 ? argv[1] : "";
    try {
        if (what == "vae" && argc == 4) return vae(argv[2], argv[3]);
        if (what == "text" && (argc == 4 || argc == 5)) return text(argv[2], argv[3], argc == 5 ? atoll(argv[4]) : 34);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-mf-pre: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "usage: %s vae VAE.safetensors DIR | text TEXT_ENCODER.gguf DIR [DROP]\n", argv[0]);
    return 2;
}
