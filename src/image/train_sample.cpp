// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/train_sample.cpp - see include/strata/image/train_sample.hpp: the disk caches a training step reads back,
// apart from the trainers that use the values (so an executable takes only the reader it needs).
#include "strata/image/train_sample.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace strata::image {

namespace {

std::vector<float> read_all_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("image engine (Qwen-Image-2.1 training): no " + path);
    const auto bytes = (int64_t) f.tellg();
    f.seekg(0);
    std::vector<float> v((size_t) (bytes / 4));
    if (!f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (v.size() * 4)))
        throw std::runtime_error("image engine (Qwen-Image-2.1 training): short " + path);
    return v;
}

}  // namespace

MfTrainSample mf_trainer_sample(const std::string& dir, const std::string& stem, int64_t txt_dim) {
    MfTrainSample s;
    auto read = [&](const std::string& name) -> std::vector<float> {
        std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
        if (!f) throw std::runtime_error("image engine (Mage-Flow training): no " + name);
        std::stringstream js;
        std::ifstream j(dir + "/" + name + ".json");
        js << j.rdbuf();
        const std::string t = js.str();
        std::vector<float> v((size_t) (std::filesystem::file_size(dir + "/" + name + ".bin") / 4));
        if (!f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (v.size() * 4)))
            throw std::runtime_error("image engine (Mage-Flow training): short " + name);
        if (t.find("\"f16\"") != std::string::npos) {
            std::vector<float> w(v.size());
            const auto* h = reinterpret_cast<const uint16_t*>(v.data());
            for (size_t i = 0; i < v.size(); ++i) w[i] = kernels::f32_from_f16(h[i]);
            return w;
        }
        return v;
    };
    s.lat = read(stem + "_lat");
    if (std::filesystem::exists(dir + "/" + stem + "_lat_sd.bin")) s.lat_sd = read(stem + "_lat_sd");
    s.te = read(stem + "_te");
    s.uncond = read(stem + "_uncond");
    // the bucket's own rows and columns come from the manifest
    std::ifstream mf(dir + "/manifest.tsv");
    std::string line;
    while (std::getline(mf, line)) {
        std::stringstream ls(line);
        std::string tok;
        std::getline(ls, tok, '\t');
        if (tok != stem) continue;
        std::getline(ls, tok, '\t');
        s.w = std::stoll(tok);
        std::getline(ls, tok, '\t');
        s.h = std::stoll(tok);
    }
    if (s.w <= 0 || s.h <= 0)
        throw std::runtime_error("image engine (Mage-Flow training): " + stem + " is not in the manifest");
    // the dump lists its shape fastest dim first: [ch][tokens] and [txt_dim][rows]
    s.tokens = s.w * s.h;
    s.ch = (int64_t) s.lat.size() / s.tokens;
    s.rows = (int64_t) (s.te.size() / (size_t) txt_dim);
    s.uncond_rows = (int64_t) (s.uncond.size() / (size_t) txt_dim);
    if ((int64_t) s.lat.size() != s.tokens * s.ch || (int64_t) s.te.size() != s.rows * txt_dim ||
        (int64_t) s.uncond.size() != s.uncond_rows * txt_dim)
        throw std::runtime_error("image engine (Mage-Flow training): " + stem + " is not this model's shape");
    return s;
}

int64_t mf_train_max_tokens(const std::string& dir) {
    std::ifstream f(dir + "/manifest.tsv");
    if (!f) throw std::runtime_error("image engine: no manifest in " + dir);
    int64_t maximum = 0, text_rows = 0;
    for (std::string line; std::getline(f, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ls(line);
        std::string stem;
        int64_t w = 0, h = 0, tokens = 0, rows = 0, uncond_rows = 0;
        if (!(ls >> stem >> w >> h >> tokens >> rows >> uncond_rows) || w <= 0 || h <= 0 || tokens != w * h ||
            rows < 0 || uncond_rows < 0)
            throw std::runtime_error("image engine: bad shape in " + dir + "/manifest.tsv");
        maximum = std::max(maximum, w * h);
        text_rows = std::max({text_rows, rows, uncond_rows});
    }
    if (maximum == 0) throw std::runtime_error("image engine: no samples in " + dir);
    return maximum + text_rows;
}

QiTrainLatent qi_train_latent(const std::string& path) {
    std::vector<float> v = read_all_f32(path);
    if (v.size() < 2) throw std::runtime_error("image engine (Qwen-Image-2.1 training): short latent cache " + path);
    QiTrainLatent l;
    l.w = (int64_t) v[0];
    l.h = (int64_t) v[1];
    const size_t n = (size_t) (64 * l.h * l.w);
    if (v.size() != 2 + 2 * n)
        throw std::runtime_error("image engine (Qwen-Image-2.1 training): latent cache of the wrong size " + path);
    l.mean.assign(v.begin() + 2, v.begin() + 2 + (long) n);
    l.sd.assign(v.begin() + 2 + (long) n, v.end());
    return l;
}

std::vector<float> qi_train_states(const std::string& path, int64_t& rows) {
    std::vector<float> v = read_all_f32(path);
    if (v.empty()) throw std::runtime_error("image engine (Qwen-Image-2.1 training): empty states cache " + path);
    rows = (int64_t) v[0];
    if ((int64_t) v.size() != 1 + rows * 4096)
        throw std::runtime_error("image engine (Qwen-Image-2.1 training): states cache of the wrong size " + path);
    return {v.begin() + 1, v.end()};
}

}  // namespace strata::image
