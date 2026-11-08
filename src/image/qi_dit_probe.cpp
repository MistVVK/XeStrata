// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_dit_probe.cpp - strata-image-dit-probe: Qwen-Image-2.1's whole DiT and its sampler on the GPU against
// stable-diffusion.cpp's values of one image (the dump of a run: noise, dit_x_K / dit_t_K / dit_out_K, cond,
// latent_final).
//
//   strata-image-dit-probe DIT.gguf DUMP_DIR [OUT_LATENT]
//
// Prints how far the noise from philox_randn (seed 42) is from the run's, each step's velocity (from the run's input
// of that step) from the run's, and the latent after the 8 steps (from the run's noise) from the run's, with the
// sampling's time.  OUT_LATENT: the latent written as raw FP32.  An edit's run (ref_latent_0, and its image's pads
// marked in cond_token_types): the reference's latent takes their place, as in the run.

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_dit.hpp"

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

namespace gpu = strata::gpu;

std::vector<float> load(const std::string& dir, const std::string& name) {
    std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + dir + "/" + name + ".bin");
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<float> v(raw.size() / 4);
    std::copy(raw.begin(), raw.end(), reinterpret_cast<char*>(v.data()));
    return v;
}

void compare(const std::string& what, const std::vector<float>& got, const std::vector<float>& want) {
    double dot = 0, ng = 0, nw = 0, err = 0, mag = 0, mx = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        dot += (double) got[i] * want[i];
        ng += (double) got[i] * got[i];
        nw += (double) want[i] * want[i];
        const double e = std::fabs((double) got[i] - want[i]);
        err += e;
        mag += std::fabs((double) want[i]);
        mx = std::max(mx, e);
    }
    std::printf("%-12s cos %.6f  rel %.3g  max %.3g\n", what.c_str(), dot / std::sqrt(ng * nw), err / mag, mx);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s DIT.gguf DUMP_DIR [OUT_LATENT]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    try {
        void* s = strata::core::Runtime::get().create_stream();
        strata::image::QiDit dit;
        std::string err;
        const auto t0 = std::chrono::steady_clock::now();
        if (!dit.open(argv[1], s, err)) throw std::runtime_error(err);
        if (const char* b = std::getenv("QI_BLOCK_BUDGET_MIB")) dit.set_block_budget(std::strtoull(b, nullptr, 10) << 20);
        if (const char* m = std::getenv("QI_MLP_ROWS")) dit.set_mlp_rows(std::strtoll(m, nullptr, 10));
        std::printf("DiT: %.2f GB of blocks, %d resident, %.2f GB besides; opened in %.1f s\n",
                    (double) dit.block_bytes() / 1e9, dit.resident_blocks(), (double) dit.weight_bytes() / 1e9,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        const auto noise = load(dir, "noise"), cond = load(dir, "cond"), want_latent = load(dir, "latent_final");
        const int64_t C = dit.channels(), n_text = (int64_t) cond.size() / dit.hidden();
        const auto side = (int64_t) std::lround(std::sqrt((double) noise.size() / (double) C));
        const int64_t H = side, W = side, n = C * H * W;
        compare("noise", strata::image::philox_randn(42, (uint64_t) n), noise);
        // an edit: the reference's latent in place of its pads (the run of marked context rows)
        std::vector<strata::image::QiDit::Ref> refs;
        float* ref_latent = nullptr;
        if (std::ifstream(dir + "/ref_latent_0.bin")) {
            const auto rl = load(dir, "ref_latent_0");
            std::ifstream tf(dir + "/cond_token_types.bin", std::ios::binary);
            std::vector<int32_t> types((size_t) n_text);
            tf.read(reinterpret_cast<char*>(types.data()), (std::streamsize) (types.size() * 4));
            const auto first = std::find_if(types.begin(), types.end(), [](int32_t v) { return v != 0; });
            strata::image::QiDit::Ref r;
            r.slot = first - types.begin();
            r.slots = std::count_if(types.begin(), types.end(), [](int32_t v) { return v != 0; });
            r.h = r.w = (int64_t) std::lround(std::sqrt((double) rl.size() / (double) C));
            if (!gpu::alloc_device(&ref_latent, rl.size() * 4)) throw std::runtime_error("alloc");
            gpu::copy(ref_latent, rl.data(), rl.size() * 4);
            r.latent = ref_latent;
            refs.push_back(r);
            std::printf("reference: %lld pads from %lld, a %lld x %lld latent\n", (long long) r.slots,
                        (long long) r.slot, (long long) r.h, (long long) r.w);
        }
        dit.set_context(cond.data(), n_text, refs);
        float *x = nullptr, *v = nullptr;
        if (!gpu::alloc_device(&x, n * 4) || !gpu::alloc_device(&v, n * 4)) throw std::runtime_error("alloc");
        std::vector<float> got((size_t) n);
        for (int k = 0; k < 8; ++k) {
            const auto xk = load(dir, "dit_x_" + std::to_string(k)), tk = load(dir, "dit_t_" + std::to_string(k));
            gpu::copy(x, xk.data(), n * 4);
            dit.forward(x, tk[0], H, W, k > 0, v);
            gpu::stream_sync(s);
            gpu::copy(got.data(), v, n * 4);
            compare("step " + std::to_string(k), got, load(dir, "dit_out_" + std::to_string(k)));
        }
        gpu::copy(x, noise.data(), n * 4);
        dit.set_context(cond.data(), n_text, refs);
        dit.sample(x, H, W, strata::image::turbo_sigmas());   // a warm-up
        gpu::copy(x, noise.data(), n * 4);
        dit.set_context(cond.data(), n_text, refs);
        const auto t1 = std::chrono::steady_clock::now();
        dit.sample(x, H, W, strata::image::turbo_sigmas());
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        gpu::copy(got.data(), x, n * 4);
        compare("latent", got, want_latent);
        std::printf("sampling: %lld x %lld latent, %lld text tokens, 8 steps in %.2f s\n", (long long) H,
                    (long long) W, (long long) n_text, secs);
        if (argc > 3) {
            std::ofstream o(argv[3], std::ios::binary);
            o.write(reinterpret_cast<const char*>(got.data()), (std::streamsize) (got.size() * 4));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-dit-probe: %s\n", e.what());
        return 1;
    }
    return 0;
}
