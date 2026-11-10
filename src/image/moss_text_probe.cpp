// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_text_probe.cpp - strata-image-moss-text-probe: MOSS-Image's text encoder (TextEncoder) against the
// reference pipeline's states of the same prompts (diffusers' MOSS-Image pipeline, its ids-NAME.npy and te-NAME.npy:
// int32 token ids and float32 [tokens][4096] states after the final norm).
//
//   strata-image-moss-text-probe QWEN3.5-9B.gguf REF_DIR [NAME...]   (default: short text layout negative)
//   strata-image-moss-text-probe --states DIR REF_DIR [NAME...]      the chat engine's ENCODE files DIR/NAME.ste instead
//
// Prints, for each prompt, the cosine and the relative error (|got - want| / |want|) over all its states, the largest
// difference, and the encoder's time; and the load's.  MOSS_LAYER_BUDGET_MIB: the blocks' VRAM (the rest streamed).

#include "strata/core/runtime.hpp"
#include "strata/image/text_encoder.hpp"

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

// A little-endian .npy of `T` (descr '<i4' or '<f4'), C order; its shape's elements
template<class T> std::vector<T> load_npy(const std::string& path, const char* descr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not a .npy");
    uint32_t hlen = 0;
    if (magic[6] == 1) {
        uint16_t h16 = 0;
        f.read(reinterpret_cast<char*>(&h16), 2);
        hlen = h16;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);
    if (header.find(descr) == std::string::npos || header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error(path + ": not a C-order " + descr + " array");
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<T> v(raw.size() / sizeof(T));
    std::memcpy(v.data(), raw.data(), v.size() * sizeof(T));
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    const bool files = argc >= 2 && std::strcmp(argv[1], "--states") == 0;
    const int a0 = files ? 2 : 1;   // the GGUF (or the states' folder), then REF_DIR
    if (argc < a0 + 2) {
        std::fprintf(stderr, "usage: xestrata-image-moss-text-probe QWEN3.5-9B.gguf REF_DIR [NAME...]\n"
                             "       xestrata-image-moss-text-probe --states DIR REF_DIR [NAME...]\n");
        return 2;
    }
    std::vector<std::string> names;
    for (int i = a0 + 2; i < argc; ++i) names.emplace_back(argv[i]);
    if (names.empty()) names = {"short", "text", "layout", "negative"};
    try {
        strata::image::TextEncoder te;
        std::string err;
        if (!files) {
            strata::core::Runtime::get();
            const char* b = std::getenv("MOSS_LAYER_BUDGET_MIB");
            const uint64_t budget = b != nullptr && *b != '\0' ? std::strtoull(b, nullptr, 10) << 20 : UINT64_MAX;
            const auto t0 = std::chrono::steady_clock::now();
            if (!te.open(argv[a0], 4096, err, budget)) {
                std::fprintf(stderr, "xestrata-image-moss-text-probe: %s\n", err.c_str());
                return 1;
            }
            std::printf("open %.0f ms, blocks %.2f GB, %d resident, %.2f GB of VRAM\n",
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                        (double) te.layer_bytes() / 1e9, te.resident_layers(), (double) te.layers_vram() / 1e9);
        }
        int bad = 0;
        for (const std::string& n : names) {
            auto path = [&](const char* kind) {   // REF_DIR/<kind>-<name>.npy
                std::string p = argv[a0 + 1];
                p.append("/").append(kind).append("-").append(n).append(".npy");
                return p;
            };
            const std::vector<int32_t> ids = load_npy<int32_t>(path("ids"), "<i4");
            const std::vector<float> want = load_npy<float>(path("te"), "<f4");
            const auto t1 = std::chrono::steady_clock::now();
            std::vector<float> got;
            int64_t rows = (int64_t) ids.size(), width = 0;
            if (files) {
                std::string p = argv[a0];
                p.append("/").append(n).append(".ste");
                if (!strata::image::TextEncoder::read_states(p, got, rows, width, err)) got.clear();
            } else {
                got = te.encode(ids, err);
                width = te.hidden();
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
            if (got.empty()) {
                std::fprintf(stderr, "%s: %s\n", n.c_str(), err.c_str());
                ++bad;
                continue;
            }
            if (rows != (int64_t) ids.size() || want.size() != got.size()) {
                std::fprintf(stderr, "%s: %lld rows of %lld, the reference's %zu values\n", n.c_str(), (long long) rows,
                             (long long) width, want.size());
                ++bad;
                continue;
            }
            double dot = 0, ng = 0, nw = 0, nd = 0, mx = 0;
            for (size_t i = 0; i < want.size(); ++i) {
                const double g = got[i], w = want[i], d = g - w;
                dot += g * w;
                ng += g * g;
                nw += w * w;
                nd += d * d;
                mx = std::max(mx, std::fabs(d));
            }
            std::printf("%-9s tokens %4zu  cos %.6f  rel %.3e  max |d| %.3e  (%.0f ms)\n", n.c_str(), ids.size(),
                        dot / std::sqrt(ng * nw), std::sqrt(nd / nw), mx, ms);
        }
        return bad == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-moss-text-probe: %s\n", e.what());
        return 1;
    }
}
