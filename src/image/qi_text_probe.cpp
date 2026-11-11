// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_text_probe.cpp - strata-image-text-probe: the image model's text encoder on the GPU against
// stable-diffusion.cpp's of one run (its te_tokens_0 and te_hidden_0: the prompt's ids and the last block's output).
//
//   strata-image-text-probe TEXT_ENCODER.gguf DUMP_DIR
//
// Prints how far the states are from the run's (all rows, and the rows after the 14 of the system turn: the DiT's
// context) and the encoder's time.  An edit's run (vit_out_0_K, the reference image's embeddings): its image pads
// (<|image_pad|>, a run of them) take those, as in the run.

#include "strata/core/runtime.hpp"
#include "strata/image/text_encoder.hpp"

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

template<class T> std::vector<T> load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<T> v(raw.size() / sizeof(T));
    std::copy(raw.begin(), raw.end(), reinterpret_cast<char*>(v.data()));
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
    std::printf("%-8s cos %.6f  rel %.3g\n", what, dot / std::sqrt(ng * nw), err / mag);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s TEXT_ENCODER.gguf DUMP_DIR\n", argv[0]);
        return 2;
    }
    try {
        strata::image::TextEncoder te;
        std::string err;
        if (!te.open(argv[1], 16384, err)) throw std::runtime_error(err);
        te.set_final_norm(false);   // Qwen-Image's context: before the final norm
        if (const char* b = std::getenv("QI_LAYER_BUDGET_MIB")) te.set_layer_budget(std::strtoull(b, nullptr, 10) << 20);
        const std::string dir = argv[2];
        const auto ids = load<int32_t>(dir + "/te_tokens_0.bin");
        const auto want = load<float>(dir + "/te_hidden_0.bin");
        // an edit's run: the image pads' run and the vision encoder's outputs
        const int32_t kImagePad = te.image_pad();
        std::vector<strata::image::TextEncoder::Picture> images;
        std::vector<std::vector<float>> vit;
        const auto first = std::find(ids.begin(), ids.end(), kImagePad);
        if (first != ids.end()) {
            strata::image::TextEncoder::Picture im;
            im.index = first - ids.begin();
            const int64_t pads = std::count(ids.begin(), ids.end(), kImagePad);
            const auto side = (int64_t) std::lround(std::sqrt((double) pads));   // a square reference
            im.nx = im.ny = side;
            for (int k = 0;; ++k) {
                std::ifstream f(dir + "/vit_out_0_" + std::to_string(k) + ".bin");
                if (!f) break;
                vit.push_back(load<float>(dir + "/vit_out_0_" + std::to_string(k) + ".bin"));
            }
            if (vit.empty()) throw std::runtime_error("image pads but no vit_out_0_0");
            im.rows = vit[0].data();
            for (size_t k = 1; k < vit.size(); ++k) im.deepstack.push_back(vit[k].data());
            images.push_back(im);
            std::printf("image: %lld pads from %lld, %zu DeepStack blocks\n", (long long) pads, (long long) im.index,
                        im.deepstack.size());
        }
        te.encode(ids, err, images);   // a warm-up
        const auto t0 = std::chrono::steady_clock::now();
        const auto got = te.encode(ids, err, images);
        if (got.empty()) throw std::runtime_error(err);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const auto C = (size_t) te.hidden();
        compare("all", got.data(), want.data(), got.size());
        if (ids.size() > 14) compare("context", got.data() + 14 * C, want.data() + 14 * C, got.size() - 14 * C);
        std::printf("text encoder: %.2f GB of layers, %d resident, %zu tokens in %.3f s\n",
                    (double) te.layer_bytes() / 1e9, te.resident_layers(), ids.size(), secs);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-text-probe: %s\n", e.what());
        return 1;
    }
    return 0;
}
