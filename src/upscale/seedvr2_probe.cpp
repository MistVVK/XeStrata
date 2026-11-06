// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/upscale/seedvr2_probe.cpp - strata-image-seedvr2-probe: SeedVR2's parts on the GPU against a reference run's
// outputs (.npy files: an image's FP32 run of ComfyUI's SeedVR2 core, which follows ByteDance-Seed/SeedVR).
//
//   strata-image-seedvr2-probe vae VAE.safetensors GOLDEN_DIR   (pix [3][H][W] in -1 .. 1, its encoder mean
//       mean [16][H / 8][W / 8], and dec [3][H][W] the decode of that mean)
//   strata-image-seedvr2-probe dit DIT.safetensors GOLDEN_DIR   (noise and cond [16][h][w], the condition latent
//       scaled; vel [16][h][w] the DiT's velocity of them)
//
// Prints how far each part's output is from the run's (cosine and relative L1) and its time.

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mf_vae.hpp"
#include "strata/upscale/seedvr2_dit.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// a .npy file's data as T (its dtype '<f4' or '<i8', C order), and its shape
template<class T> std::vector<T> npy(const std::string& path, std::vector<int64_t>* shape = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not a .npy file");
    uint32_t hlen = 0;
    if (magic[6] == 1) {
        uint16_t h = 0;
        f.read(reinterpret_cast<char*>(&h), 2);
        hlen = h;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);
    const char* want = sizeof(T) == 8 ? "'<i8'" : "'<f4'";
    if (header.find(want) == std::string::npos || header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error(path + ": expected " + want + " in C order: " + header);
    const size_t l = header.find('('), r = header.find(')');
    std::vector<int64_t> dims;
    size_t n = 1;
    for (size_t i = l + 1; i < r;) {
        size_t j = header.find_first_of(",)", i);
        const std::string tok = header.substr(i, j - i);
        if (tok.find_first_not_of(' ') != std::string::npos) {
            dims.push_back(std::stoll(tok));
            n *= (size_t) dims.back();
        }
        i = j + 1;
    }
    std::vector<T> v(n);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (n * sizeof(T)));
    if (!f) throw std::runtime_error(path + ": short");
    if (shape) *shape = dims;
    return v;
}

void compare(const char* what, const float* got, const float* want, size_t n) {
    double dot = 0, ng = 0, nw = 0, err = 0, mag = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double) got[i] * want[i];
        ng += (double) got[i] * got[i];
        nw += (double) want[i] * want[i];
        err += std::fabs((double) got[i] - want[i]);
        mag += std::fabs((double) want[i]);
    }
    std::printf("%-10s cos %.6f  rel %.3g  (%zu values)\n", what, dot / std::sqrt(ng * nw), err / mag, n);
}

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

int vae(const std::string& path, const std::string& dir) {
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
    return 0;
}

int dit(const std::string& path, const std::string& dir) {
    void* s = strata::core::Runtime::get().create_stream();
    strata::upscale::Seedvr2Dit d;
    std::string err;
    if (!d.open(path, s, err)) throw std::runtime_error(err);
    std::vector<int64_t> ns;
    const auto noise = npy<float>(dir + "/noise.npy", &ns);   // [16][2 h][2 w]
    const auto cond = npy<float>(dir + "/cond.npy");
    const auto vel = npy<float>(dir + "/vel.npy");
    const int64_t h = ns[1] / 2, w = ns[2] / 2, n = h * w * 64;
    float* dn = put(tokens_of(noise, h, w));
    float* dc = put(tokens_of(cond, h, w));
    float* out = nullptr;
    if (!strata::gpu::alloc_device(&out, (size_t) n * 4)) throw std::runtime_error("GPU memory");
    {   // the blocks that fit beside the activations, a GiB kept free
        size_t free = 0, total = 0;
        strata::gpu::mem_info(&free, &total);
        const uint64_t act = d.activation_bytes(h, w), keep = 1ull << 30;
        const uint64_t room = (uint64_t) free + d.blocks_vram() > act + keep ? (uint64_t) free + d.blocks_vram() - act - keep : 0;
        d.set_block_budget(std::max(d.block_min(), std::min(d.block_bytes(), room)));
    }
    d.forward(dn, dc, h, w, out);   // warm
    const auto t0 = std::chrono::steady_clock::now();
    d.forward(dn, dc, h, w, out);
    std::printf("dit %lld x %lld tokens: %.1f ms (%d blocks resident, activations %.0f MiB)\n", (long long) w,
                (long long) h, 1e3 * seconds_since(t0), d.resident_blocks(),
                (double) d.activations_held() / 1048576.0);
    const auto got = latent_of(fetch(out, (size_t) n), h, w);
    compare("velocity", got.data(), vel.data(), vel.size());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string what = argc > 1 ? argv[1] : "";
        if (what == "vae" && argc == 4) return vae(argv[2], argv[3]);
        if (what == "dit" && argc == 4) return dit(argv[2], argv[3]);
        std::fprintf(stderr, "usage: %s vae VAE.safetensors GOLDEN_DIR | dit DIT.safetensors GOLDEN_DIR\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "seedvr2-probe: %s\n", e.what());
        return 1;
    }
}
