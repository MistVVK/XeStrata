// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_vision_probe.cpp - strata-image-vision-probe: the image model's vision encoder (Qwen3-VL's ViT) on the
// GPU against stable-diffusion.cpp's values of one edit (the dump of a run: vit_in_0, vit_out_0_0 .. vit_out_0_3).
//
//   strata-image-vision-probe MMPROJ.gguf DUMP_DIR
//
// Prints how far the embeddings and each DeepStack block are from the run's, and the encoder's time.

#include "strata/core/runtime.hpp"
#include "strata/core/vision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> load(const std::string& dir, const std::string& name) {
    std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + dir + "/" + name + ".bin");
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<float> v(raw.size() / 4);
    std::copy(raw.begin(), raw.end(), reinterpret_cast<char*>(v.data()));
    return v;
}

void compare(const std::string& what, const std::vector<float>& got, const std::vector<float>& want) {
    if (got.size() != want.size()) {
        std::printf("%-10s size %zu, the run's %zu\n", what.c_str(), got.size(), want.size());
        return;
    }
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
    std::printf("%-10s cos %.6f  rel %.3g  max %.3g\n", what.c_str(), dot / std::sqrt(ng * nw), err / mag, mx);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MMPROJ.gguf DUMP_DIR\n", argv[0]);
        return 2;
    }
    try {
        const std::string dir = argv[2];
        void* s = strata::core::Runtime::get().create_stream();
        strata::core::Vision vit;
        std::string err;
        if (!vit.open(argv[1], s, err)) throw std::runtime_error(err);
        const auto rgb = load(dir, "vit_in_0");
        const auto side = (int64_t) std::lround(std::sqrt((double) rgb.size() / 3));   // a square reference
        strata::core::VisionOut out = vit.encode(rgb.data(), side, side);   // warm-up
        const auto t0 = std::chrono::steady_clock::now();
        out = vit.encode(rgb.data(), side, side);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        compare("main", out.main, load(dir, "vit_out_0_0"));
        for (size_t i = 0; i < out.deepstack.size(); ++i)
            compare("deepstack " + std::to_string(i), out.deepstack[i], load(dir, "vit_out_0_" + std::to_string(i + 1)));
        std::printf("vision: %lld x %lld image, %lld tokens, %.3f s (weights %.2f GB)\n", (long long) side,
                    (long long) side, (long long) out.rows, secs, (double) vit.weight_bytes() / 1e9);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-vision-probe: %s\n", e.what());
        return 1;
    }
    return 0;
}
