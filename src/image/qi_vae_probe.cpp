// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_vae_probe.cpp - strata-image-vae-probe: Qwen-Image-2.1's VAE decoder on the GPU against
// stable-diffusion.cpp's image of one run (its latent_final and image, a run decoded without tiling).
//
//   strata-image-vae-probe VAE.safetensors DUMP_DIR [LATENT.f32]
//   strata-image-vae-probe --encode VAE.safetensors DUMP_DIR
//
// Prints how far the decoded image is from the run's (per 255) and the decode's time.  LATENT.f32: decode that latent
// (raw FP32 [64][h][w], strata-image-dit-probe's) instead of the run's, against the same image.  --encode: the encoder
// instead, the reference image of an edit (ref_image_0, [4][H][W] in 0 .. 1) against the run's latent (ref_latent_0,
// encoded without tiling).

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_vae.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<float> v(raw.size() / 4);
    std::copy(raw.begin(), raw.end(), reinterpret_cast<char*>(v.data()));
    return v;
}

}  // namespace

namespace {

int encode_probe(const std::string& vae_path, const std::string& dir) {
    namespace gpu = strata::gpu;
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::QiVae vae;
    std::string err;
    if (!vae.open(vae_path, s, err)) throw std::runtime_error(err);
    const auto img = load(dir + "/ref_image_0.bin");
    const auto want = load(dir + "/ref_latent_0.bin");
    const auto side = (int64_t) std::lround(std::sqrt((double) img.size() / 4.0));
    float *dimg = nullptr, *dz = nullptr;
    if (!gpu::alloc_device(&dimg, img.size() * 4) || !gpu::alloc_device(&dz, want.size() * 4))
        throw std::runtime_error("alloc");
    gpu::copy(dimg, img.data(), img.size() * 4);
    std::printf("VAE encoder working memory: %.2f GB\n", (double) vae.encode_scratch_bytes(side, side) / 1e9);
    if (!vae.encode(dimg, side, side, dz)) throw std::runtime_error("the VAE file has no encoder");   // a warm-up
    const auto t0 = std::chrono::steady_clock::now();
    vae.encode(dimg, side, side, dz);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<float> got(want.size());
    gpu::copy(got.data(), dz, got.size() * 4);
    double dot = 0, ng = 0, nw = 0, e = 0, mag = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        dot += (double) got[i] * want[i];
        ng += (double) got[i] * got[i];
        nw += (double) want[i] * want[i];
        e += std::fabs((double) got[i] - want[i]);
        mag += std::fabs((double) want[i]);
    }
    std::printf("encoder: %lld x %lld image in %.3f s; latent vs the run's: cos %.6f rel %.3g\n", (long long) side,
                (long long) side, secs, dot / std::sqrt(ng * nw), e / mag);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--encode") {
        try {
            return encode_probe(argv[2], argv[3]);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xestrata-image-vae-probe: %s\n", e.what());
            return 1;
        }
    }
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s VAE.safetensors DUMP_DIR [LATENT.f32]\n", argv[0]);
        return 2;
    }
    namespace gpu = strata::gpu;
    try {
        void* s = strata::core::Runtime::get().create_stream();
        strata::image::QiVae vae;
        std::string err;
        if (!vae.open(argv[1], s, err)) throw std::runtime_error(err);
        const std::string dir = argv[2];
        const auto z = load(argc > 3 ? std::string(argv[3]) : dir + "/latent_final.bin");
        const auto want = load(dir + "/image.bin");
        const auto side = (int64_t) std::lround(std::sqrt((double) z.size() / 64.0));
        const int64_t HW = 256 * side * side;
        float *dz = nullptr, *img = nullptr;
        if (!gpu::alloc_device(&dz, z.size() * 4) || !gpu::alloc_device(&img, (size_t) (4 * HW) * 4))
            throw std::runtime_error("alloc");
        gpu::copy(dz, z.data(), z.size() * 4);
        const char* cap = std::getenv("QI_VAE_SCRATCH_MIB");   // tiles when the whole image's buffers pass it
        const uint64_t max_scratch = cap ? std::strtoull(cap, nullptr, 10) << 20 : UINT64_MAX;
        std::printf("VAE working memory for the whole image: %.2f GB\n", (double) vae.scratch_bytes(side, side) / 1e9);
        vae.decode(dz, side, side, img, max_scratch);   // a warm-up
        const auto t0 = std::chrono::steady_clock::now();
        vae.decode(dz, side, side, img, max_scratch);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<float> got((size_t) (4 * HW));
        gpu::copy(got.data(), img, got.size() * 4);
        double sum = 0, mx = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double d = std::fabs((double) got[i] - want[i]);
            sum += d;
            mx = std::max(mx, d);
        }
        std::printf("VAE: %.2f GB, %lld x %lld image in %.2f s; vs the run's image: mean %.3f / 255, max %.1f / 255\n",
                    (double) vae.weight_bytes() / 1e9, 16LL * (long long) side, 16LL * (long long) side, secs,
                    sum / (double) got.size() * 255.0, mx * 255.0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-vae-probe: %s\n", e.what());
        return 1;
    }
    return 0;
}
