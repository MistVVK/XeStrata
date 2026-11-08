// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/upscale/seedvr2_probe.cpp - strata-image-seedvr2-probe: SeedVR2's parts on the GPU against a reference run's
// outputs (.npy files: an image's FP32 run of a reference that follows ByteDance-Seed/SeedVR).
//
//   strata-image-seedvr2-probe vae VAE.safetensors GOLDEN_DIR   (pix [3][H][W] in -1 .. 1, its encoder mean
//       mean [16][H / 8][W / 8], and dec [3][H][W] the decode of that mean)
//   strata-image-seedvr2-probe dit DIT.safetensors GOLDEN_DIR   (noise and cond [16][h][w], or [16][t][h][w] a
//       clip's t latent frames, the condition latent scaled; vel alike, the DiT's velocity of them)
//   strata-image-seedvr2-probe vae-video VAE.safetensors CLIP.rgb W H T GOLDEN_DIR [K [TT [host]]]   (a clip of T
//       frames of RGB8 [T][H][W][3]; mean [16][t][H / 8][W / 8] the encoder's mean of all of it at once,
//       dec [3][T][H][W] the decode of that mean): the engine's in slices, 1 + 4 K frames first and 4 K each after (K 0: the clip
//       whole), and the decode of the run's mean in slices of K latents alike; TT: in tiles of TT x TT tokens; host:
//       their carried frames in host memory
//   strata-image-seedvr2-probe video DIT.safetensors CLIP.rgb W H T OUT.rgb CHUNK OVERLAP SEED   (a clip of RGB8
//       [T][H][W][3], its sides multiples of 16, through Seedvr2's video path: chunks of CHUNK frames overlapping by
//       OVERLAP latents; the frames out as RGB8 to OUT.rgb, and the parts' seconds)
//
// Prints how far each part's output is from the run's (cosine and relative L1) and its time.

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mf_vae.hpp"
#include "strata/upscale/seedvr2.hpp"
#include "strata/upscale/seedvr2_dit.hpp"
#include "strata/platform/env.hpp"
#include "../image/probe_npy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::vector<float> fetch(const float* d, size_t n) {
    std::vector<float> h(n);
    strata::gpu::copy(h.data(), d, n * 4);
    return h;
}

float* put(const std::vector<float>& h) {
    float* d = nullptr;
    if (!strata::gpu::alloc_device(&d, h.size() * 4)) throw std::runtime_error("GPU memory");
    strata::gpu::copy(d, h.data(), h.size() * 4);
    return d;
}

// the VAE's tokens [h][w][64] (a token 2 x 2 latent positions: value 4 c + 2 dy + dx) <-> the latent [16][2 h][2 w]
std::vector<float> latent_of(const std::vector<float>& z, int64_t h, int64_t w) {
    std::vector<float> l(z.size());
    for (int64_t Y = 0; Y < 2 * h; ++Y)
        for (int64_t X = 0; X < 2 * w; ++X)
            for (int64_t c = 0; c < 16; ++c)
                l[(size_t) ((c * 2 * h + Y) * 2 * w + X)] =
                    z[(size_t) (((Y / 2) * w + X / 2) * 64 + 4 * c + 2 * (Y % 2) + X % 2)];
    return l;
}

std::vector<float> tokens_of(const std::vector<float>& l, int64_t h, int64_t w) {
    std::vector<float> z(l.size());
    for (int64_t Y = 0; Y < 2 * h; ++Y)
        for (int64_t X = 0; X < 2 * w; ++X)
            for (int64_t c = 0; c < 16; ++c)
                z[(size_t) (((Y / 2) * w + X / 2) * 64 + 4 * c + 2 * (Y % 2) + X % 2)] =
                    l[(size_t) ((c * 2 * h + Y) * 2 * w + X)];
    return z;
}

int vae(const std::string& path, const std::string& dir, const std::string& out_path = "") {
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::MfVae v;
    std::string err;
    if (!v.open(path, s, err) || !v.open_encoder(err)) throw std::runtime_error(err);
    std::vector<int64_t> ps;
    const auto pix = npy<float>(dir + "/pix.npy", &ps);   // [3][H][W]
    const auto mean = npy<float>(dir + "/mean.npy");      // [16][H / 8][W / 8]
    const auto dec = npy<float>(dir + "/dec.npy");        // [3][H][W]
    const int64_t H = ps[1], W = ps[2], h = H / 16, w = W / 16;
    if (H % 16 != 0 || W % 16 != 0) throw std::runtime_error("the image's sides are not multiples of 16");
    float* rgb = put(pix);
    float* z = nullptr;
    if (!strata::gpu::alloc_device(&z, (size_t) (h * w * 64) * 4)) throw std::runtime_error("GPU memory");
    v.encode(rgb, h, w, nullptr, z);   // warm
    strata::gpu::stream_sync(s);
    auto t0 = std::chrono::steady_clock::now();
    v.encode(rgb, h, w, nullptr, z);
    strata::gpu::stream_sync(s);
    std::printf("encode %lld x %lld: %.1f ms\n", (long long) W, (long long) H, 1e3 * seconds_since(t0));
    const auto got = latent_of(fetch(z, (size_t) (h * w * 64)), h, w);
    compare("mean", got.data(), mean.data(), mean.size());
    float* zg = put(tokens_of(mean, h, w));
    float* out = nullptr;
    if (!strata::gpu::alloc_device(&out, (size_t) (3 * H * W) * 4)) throw std::runtime_error("GPU memory");
    v.decode_raw(zg, h, w, out);   // warm
    strata::gpu::stream_sync(s);
    t0 = std::chrono::steady_clock::now();
    v.decode_raw(zg, h, w, out);
    strata::gpu::stream_sync(s);
    std::printf("decode %lld x %lld: %.1f ms\n", (long long) W, (long long) H, 1e3 * seconds_since(t0));
    const auto d = fetch(out, (size_t) (3 * H * W));
    compare("decode", d.data(), dec.data(), dec.size());
    if (!out_path.empty()) {   // the decode, [3][H][W] FP32
        std::ofstream o(out_path, std::ios::binary);
        o.write(reinterpret_cast<const char*>(d.data()), (std::streamsize) (d.size() * 4));
    }
    return 0;
}

int vae_video(const std::string& path, const std::string& clip, int64_t W, int64_t H, int64_t T,
              const std::string& dir, int64_t K, int64_t tt, bool host) {
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::MfVae v;
    v.set_video(true);
    std::string err;
    if (!v.open(path, s, err) || !v.open_encoder(err)) throw std::runtime_error(err);
    v.set_video_host_cache(host);
    if (H % 16 != 0 || W % 16 != 0 || (T - 1) % 4 != 0) throw std::runtime_error("sides of 16s and 1 + 4 n frames");
    const int64_t h = H / 16, w = W / 16, n = (T - 1) / 4 + 1, HW = H * W;
    if (K <= 0) K = (T - 1) / 4;
    std::vector<uint8_t> raw((size_t) (T * HW * 3));
    std::ifstream(clip, std::ios::binary).read(reinterpret_cast<char*>(raw.data()), (std::streamsize) raw.size());
    std::vector<float> pix((size_t) (T * 3 * HW));   // [T][3][H][W], -1 .. 1
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < 3; ++c)
            for (int64_t p = 0; p < HW; ++p)
                pix[(size_t) ((t * 3 + c) * HW + p)] =
                    (float) raw[(size_t) ((t * HW + p) * 3 + c)] * (2.f / 255.f) - 1.f;
    const auto mean = npy<float>(dir + "/mean.npy");   // [16][n][h * 2][w * 2]
    const auto dec = npy<float>(dir + "/dec.npy");     // [3][T][H][W]
    float* rgb = put(pix);
    float* z = nullptr;
    if (!strata::gpu::alloc_device(&z, (size_t) (n * h * w * 64) * 4)) throw std::runtime_error("GPU memory");
    auto t0 = std::chrono::steady_clock::now();
    int64_t done = 0, latents = 0;
    for (bool first = true; done < T; first = false) {
        const int64_t f = first ? std::min<int64_t>(T, 1 + 4 * K) : std::min<int64_t>(T - done, 4 * K);
        latents += v.encode_video_tiled(rgb + done * 3 * HW, f, h, w, first, z + latents * h * w * 64, tt);
        done += f;
    }
    std::printf("encode %lld frames of %lld x %lld in slices of %lld: %lld latents, %.1f ms\n", (long long) T,
                (long long) W, (long long) H, 4LL * K, (long long) latents, 1e3 * seconds_since(t0));
    {   // the run's [16][n][2 h][2 w] against each latent's tokens
        const auto zt = fetch(z, (size_t) (n * h * w * 64));
        std::vector<float> got(mean.size());
        const int64_t lp = 4 * h * w;
        for (int64_t t = 0; t < n; ++t) {
            const std::vector<float> one(zt.begin() + t * h * w * 64, zt.begin() + (t + 1) * h * w * 64);
            const auto l = latent_of(one, h, w);
            for (int64_t c = 0; c < 16; ++c)
                std::copy(l.begin() + c * lp, l.begin() + (c + 1) * lp, got.begin() + (c * n + t) * lp);
        }
        compare("mean", got.data(), mean.data(), mean.size());
    }
    {   // the decode of the run's mean, in slices of K latents (the first 1 + K)
        std::vector<float> tok((size_t) (n * h * w * 64));
        const int64_t lp = 4 * h * w;
        for (int64_t t = 0; t < n; ++t) {
            std::vector<float> l((size_t) (16 * lp));
            for (int64_t c = 0; c < 16; ++c)
                std::copy(mean.begin() + (c * n + t) * lp, mean.begin() + (c * n + t + 1) * lp, l.begin() + c * lp);
            const auto tk = tokens_of(l, h, w);
            std::copy(tk.begin(), tk.end(), tok.begin() + t * h * w * 64);
        }
        float* zg = put(tok);
        float* out = nullptr;
        if (!strata::gpu::alloc_device(&out, (size_t) (T * 3 * HW) * 4)) throw std::runtime_error("GPU memory");
        t0 = std::chrono::steady_clock::now();
        int64_t lat = 0, frames = 0;
        for (bool first = true; lat < n; first = false) {
            const int64_t k = first ? std::min<int64_t>(n, 1 + K) : std::min<int64_t>(n - lat, K);
            frames += v.decode_video_tiled(zg + lat * h * w * 64, k, h, w, first, out + frames * 3 * HW, tt);
            lat += k;
        }
        std::printf("decode %lld latents in slices of %lld: %lld frames, %.1f ms\n", (long long) n, (long long) K,
                    (long long) frames, 1e3 * seconds_since(t0));
        const auto d = fetch(out, (size_t) (T * 3 * HW));   // [T][3][H][W] -> [3][T][H][W]
        std::vector<float> got(d.size());
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < 3; ++c)
                std::copy(d.begin() + (t * 3 + c) * HW, d.begin() + (t * 3 + c + 1) * HW,
                          got.begin() + (c * T + t) * HW);
        compare("decode", got.data(), dec.data(), dec.size());
        for (int64_t t = 0; t < T; ++t) {   // each frame's own
            std::vector<float> a((size_t) (3 * HW)), b((size_t) (3 * HW));
            for (int64_t c = 0; c < 3; ++c) {
                std::copy(got.begin() + (c * T + t) * HW, got.begin() + (c * T + t + 1) * HW, a.begin() + c * HW);
                std::copy(dec.begin() + (c * T + t) * HW, dec.begin() + (c * T + t + 1) * HW, b.begin() + c * HW);
            }
            char name[32];
            std::snprintf(name, sizeof name, "  frame %lld", (long long) t);
            compare(name, a.data(), b.data(), a.size());
        }
    }
    return 0;
}

int video(const std::string& path, const std::string& clip, int64_t W, int64_t H, int64_t T, const std::string& out,
          int64_t chunk, int64_t overlap, uint64_t seed) {
    strata::upscale::Seedvr2 m(path);
    const int64_t HW = W * H;
    std::vector<uint8_t> raw((size_t) (T * HW * 3));
    std::ifstream(clip, std::ios::binary).read(reinterpret_cast<char*>(raw.data()), (std::streamsize) raw.size());
    uint8_t *in = nullptr, *got = nullptr;
    if (!strata::gpu::alloc_device(&in, (size_t) (HW * 4)) || !strata::gpu::alloc_device(&got, (size_t) (HW * 4)))
        throw std::runtime_error("GPU memory");
    std::vector<uint8_t> rgba((size_t) (HW * 4)), res;
    res.reserve((size_t) (T * HW * 3));
    const auto take = [&] {
        while (m.video_ready() > 0) {
            m.video_take(got, 1);
            strata::gpu::copy(rgba.data(), got, rgba.size());
            for (int64_t p = 0; p < HW; ++p)
                for (int c = 0; c < 3; ++c) res.push_back(rgba[(size_t) (p * 4 + c)]);
        }
    };
    {   // the room: what is free now, a GiB kept (STRATA_SEEDVR2_ROOM_MIB names a smaller one)
        size_t free = 0, total = 0;
        strata::gpu::mem_info(&free, &total);
        uint64_t room = free > (1ull << 30) ? (uint64_t) free - (1ull << 30) : 0;
        if (const char* e = strata::env_get("SEEDVR2_ROOM_MIB"))
            room = std::min<uint64_t>(room, std::stoull(e) << 20);
        m.set_room(room);
    }
    // STRATA_PROBE_REPEAT: the clip run that many times, each one's time printed (the last warm), its frames the last's
    const char* rep = strata::env_get("PROBE_REPEAT");
    const long repeat = rep != nullptr ? std::max(1L, std::strtol(rep, nullptr, 10)) : 1L;
    for (long r = 0; r < repeat; ++r) {
        res.clear();
        const auto t0 = std::chrono::steady_clock::now();
        m.video_begin(W, H, T, seed, chunk, overlap);
        for (int64_t f = 0; f < T; ++f) {
            for (int64_t p = 0; p < HW; ++p) {
                for (int c = 0; c < 3; ++c) rgba[(size_t) (p * 4 + c)] = raw[(size_t) ((f * HW + p) * 3 + c)];
                rgba[(size_t) (p * 4 + 3)] = 255;
            }
            strata::gpu::copy(in, rgba.data(), rgba.size());
            m.video_push(in, 1);
            take();
        }
        m.video_end();
        take();
        std::printf("video %lld frames of %lld x %lld, chunks of %lld overlapping by %lld: %.1f s (%s), %zu frames out\n",
                    (long long) T, (long long) W, (long long) H, (long long) chunk, (long long) overlap,
                    seconds_since(t0), m.parts().c_str(), res.size() / (size_t) (HW * 3));
    }
    std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char*>(res.data()), (std::streamsize) res.size());
    return 0;
}

int dit(const std::string& path, const std::string& dir) {
    void* s = strata::core::Runtime::get().create_stream();
    strata::upscale::Seedvr2Dit d;
    std::string err;
    if (!d.open(path, s, err)) throw std::runtime_error(err);
    std::vector<int64_t> ns;
    const auto noise = npy<float>(dir + "/noise.npy", &ns);   // [16][2 h][2 w], or [16][t][2 h][2 w]
    const auto cond = npy<float>(dir + "/cond.npy");
    const auto vel = npy<float>(dir + "/vel.npy");
    const bool clip = ns.size() == 4;
    const int64_t t = clip ? ns[1] : 1, h = ns[clip ? 2 : 1] / 2, w = ns[clip ? 3 : 2] / 2, n = t * h * w * 64;
    const int64_t lp = 4 * h * w;
    // [16][t][..] <-> each frame's tokens one after another
    const auto frames_tokens = [&](const std::vector<float>& l) {
        std::vector<float> z((size_t) n);
        for (int64_t f = 0; f < t; ++f) {
            std::vector<float> one((size_t) (16 * lp));
            for (int64_t c = 0; c < 16; ++c)
                std::copy(l.begin() + (c * t + f) * lp, l.begin() + (c * t + f + 1) * lp, one.begin() + c * lp);
            const auto tk = tokens_of(one, h, w);
            std::copy(tk.begin(), tk.end(), z.begin() + f * h * w * 64);
        }
        return z;
    };
    const auto frames_latent = [&](const std::vector<float>& z) {
        std::vector<float> l((size_t) n);
        for (int64_t f = 0; f < t; ++f) {
            const std::vector<float> one(z.begin() + f * h * w * 64, z.begin() + (f + 1) * h * w * 64);
            const auto lt = latent_of(one, h, w);
            for (int64_t c = 0; c < 16; ++c)
                std::copy(lt.begin() + c * lp, lt.begin() + (c + 1) * lp, l.begin() + (c * t + f) * lp);
        }
        return l;
    };
    float* dn = put(frames_tokens(noise));
    float* dc = put(frames_tokens(cond));
    float* out = nullptr;
    if (!strata::gpu::alloc_device(&out, (size_t) n * 4)) throw std::runtime_error("GPU memory");
    {   // the blocks that fit beside the activations, a GiB kept free
        size_t free = 0, total = 0;
        strata::gpu::mem_info(&free, &total);
        const uint64_t act = d.activation_bytes(h, w, t), keep = 1ull << 30;
        const uint64_t room = (uint64_t) free + d.blocks_vram() > act + keep ? (uint64_t) free + d.blocks_vram() - act - keep : 0;
        d.set_block_budget(std::max(d.block_min(), std::min(d.block_bytes(), room)));
    }
    d.forward(dn, dc, h, w, out, t);   // warm
    const auto t0 = std::chrono::steady_clock::now();
    d.forward(dn, dc, h, w, out, t);
    std::printf("dit %lld x %lld x %lld tokens: %.1f ms (%d blocks resident, activations %.0f MiB)\n", (long long) t,
                (long long) w, (long long) h, 1e3 * seconds_since(t0), d.resident_blocks(),
                (double) d.activations_held() / 1048576.0);
    const auto got = frames_latent(fetch(out, (size_t) n));
    compare("velocity", got.data(), vel.data(), vel.size());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string what = argc > 1 ? argv[1] : "";
        if (what == "vae" && (argc == 4 || argc == 5)) return vae(argv[2], argv[3], argc == 5 ? argv[4] : "");
        if (what == "dit" && argc == 4) return dit(argv[2], argv[3]);
        if (what == "video" && argc == 11)
            return video(argv[2], argv[3], std::stoll(argv[4]), std::stoll(argv[5]), std::stoll(argv[6]), argv[7],
                         std::stoll(argv[8]), std::stoll(argv[9]), std::stoull(argv[10]));
        if (what == "vae-video" && argc >= 8)
            return vae_video(argv[2], argv[3], std::stoll(argv[4]), std::stoll(argv[5]), std::stoll(argv[6]), argv[7],
                             argc > 8 ? std::stoll(argv[8]) : 0, argc > 9 ? std::stoll(argv[9]) : 0,
                             argc > 10 && std::string(argv[10]) == "host");
        std::fprintf(stderr, "usage: %s vae VAE.safetensors GOLDEN_DIR [OUT.f32] | dit DIT.safetensors GOLDEN_DIR\n",
                     argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "seedvr2-probe: %s\n", e.what());
        return 1;
    }
}
