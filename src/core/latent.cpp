// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// src/core/latent.cpp - the latent MoE pack's readers.  See the header.
#include "strata/core/latent.hpp"

#include "strata/platform/env.hpp"
#include "strata/platform/mapped_file.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace strata::core {

LatentDense::LatentDense() = default;
LatentDense::~LatentDense() = default;

void LatentDense::close() {
    map_.reset();
    map_size_ = 0;
    n_layers_ = n_groups_ = n_embd_ = latent_dim_ = 0;
    fmt_ = 8;
}

uint64_t LatentDense::bytes_for(int64_t n_layers, int64_t n_groups, int64_t n_embd, int64_t latent_dim, int fmt) {
    const size_t row_b = fmt == 4 ? 18 : 34;
    const size_t db = (size_t) n_groups * (size_t) latent_dim * (size_t) (n_embd / 32) * row_b;
    const size_t ub = (size_t) n_groups * (size_t) n_embd * (size_t) (latent_dim / 32) * row_b;
    return (uint64_t) n_layers * (db + ub);
}

bool LatentDense::open(const std::string& path, int64_t n_layers, int64_t n_groups, int64_t n_embd,
                       int64_t latent_dim, std::string& err) {
    close();
    if (n_layers <= 0 || n_groups <= 0 || n_embd <= 0 || latent_dim <= 0 || n_embd % 32 != 0 || latent_dim % 32 != 0) {
        err = path + ": the latent dense geometry is not whole 32-value blocks";
        return false;
    }
    n_layers_ = n_layers; n_groups_ = n_groups; n_embd_ = n_embd; latent_dim_ = latent_dim;
    fmt_ = 8;
    if (const char* f = strata::env_get("LATENT_DENSE_FMT"); f != nullptr && f[0] != '\0') {
        if (f[0] == '4' && f[1] == '\0') fmt_ = 4;
        else if (f[0] == '8' && f[1] == '\0') fmt_ = 8;
        else {
            err = path + ": XESTRATA_LATENT_DENSE_FMT must be 4 or 8 (got '" + std::string(f) + "')";
            return false;
        }
    }
    const uint64_t want = (uint64_t) n_layers * (d_bytes() + u_bytes());
    try {
        map_ = std::make_unique<platform::MappedFile>(path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    map_size_ = map_->size();
    if (map_size_ != want) {
        // strict: the env's format and the file's size disagree.  Name both candidate sizes so the operator
        // sees which format the file actually is (edge0-strata's refusal, no silent fallback).
        const uint64_t alt = bytes_for(n_layers, n_groups, n_embd, latent_dim, fmt_ == 4 ? 8 : 4);
        err = path + ": size " + std::to_string(map_size_) + " != " + std::to_string(want) +
              " (fmt " + std::to_string(fmt_) + " wants " + std::to_string(want) +
              "; the other fmt wants " + std::to_string(alt) + ")";
        close();
        return false;
    }
    return true;
}

const uint8_t* LatentDense::d(int64_t layer) const {
    if (!map_ || layer < 0 || layer >= n_layers_) return nullptr;
    return map_->data() + (size_t) layer * (d_bytes() + u_bytes());
}

const uint8_t* LatentDense::u(int64_t layer) const {
    const uint8_t* p = d(layer);
    return p != nullptr ? p + d_bytes() : nullptr;
}

LatentA::LatentA() = default;
LatentA::~LatentA() = default;

void LatentA::close() {
    map_.reset();
    n_layers_ = n_expert_ = blob_bytes_ = 0;
}

bool LatentA::open(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                   std::string& err) {
    close();
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = path + ": the latent A geometry is empty";
        return false;
    }
    n_layers_ = n_layers; n_expert_ = n_expert; blob_bytes_ = blob_bytes;
    const uint64_t want = (uint64_t) n_layers * (uint64_t) n_expert * (uint64_t) blob_bytes;
    try {
        map_ = std::make_unique<platform::MappedFile>(path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    if (map_->size() != want) {
        err = path + ": size " + std::to_string(map_->size()) + " != " + std::to_string(want);
        close();
        return false;
    }
    return true;
}

const uint8_t* LatentA::blob(int64_t layer, int64_t expert) const {
    if (!map_ || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return nullptr;
    return map_->data() + ((size_t) layer * (size_t) n_expert_ + (size_t) expert) * (size_t) blob_bytes_;
}

bool latent_manifest_read(const std::string& path, LatentManifest& m, std::string& err) {
    m = LatentManifest{};
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::string line;
    bool head = false;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (!head) {
                int v = 0;
                if (std::sscanf(line.c_str(), "# strata native experts v%d:", &v) != 1) {
                    err = path + ": not a strata native experts manifest (XeStrata's xs manifests go to expert_layout_load)";
                    return false;
                }
                if (v < 3 || v > 6) {
                    err = path + ": native experts version " + std::to_string(v) + " is not one this reader knows (3..6)";
                    return false;
                }
                m.version = v;
                head = true;
                const size_t eb = line.find("expert_embd ");
                if (eb != std::string::npos) m.expert_embd = std::atoll(line.c_str() + eb + 12);
                const size_t ef = line.find("expert_ff ");
                if (ef != std::string::npos) m.expert_ff = std::atoll(line.c_str() + ef + 10);
                const size_t ne = line.find("(n_expert ");
                if (ne != std::string::npos) m.n_expert = std::atoll(line.c_str() + ne + 10);
            }
            continue;
        }
        if (!head) { err = path + ": a row before the version header"; return false; }
        LatentLayerFmt l;
        unsigned long long off = 0, blob = 0;
        int layer = -1;
        if (std::sscanf(line.c_str(), "%d %d %d %llu %llu", &layer, &l.gu_type, &l.d_type, &off, &blob) != 5) {
            err = path + ": cannot parse the row '" + line + "'";
            return false;
        }
        if (layer != (int) m.layers.size()) {
            err = path + ": layer " + std::to_string(layer) + " out of order (expected " +
                  std::to_string(m.layers.size()) + ")";
            return false;
        }
        if (l.gu_type < 0 || l.d_type < 0 || blob == 0) {
            err = path + ": layer " + std::to_string(layer) + " has a bad type or blob size";
            return false;
        }
        l.offset = off;
        l.blob_bytes = blob;
        m.layers.push_back(l);
    }
    if (!head || m.layers.empty()) { err = path + ": no layers"; return false; }
    return true;
}

bool latent_labels_read(const std::string& path, int64_t n_layers, int64_t n_expert, int n_groups,
                        std::vector<int32_t>& labels, std::string& err) {
    labels.clear();
    if (n_layers <= 0 || n_expert <= 0 || n_groups <= 0) {
        err = path + ": the labels geometry is empty";
        return false;
    }
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { err = "cannot open " + path; return false; }
    const std::streamsize want = (std::streamsize) (n_layers * n_expert * 4);
    if (f.tellg() != want) {
        err = path + ": size " + std::to_string((long long) f.tellg()) + " != " + std::to_string((long long) want) +
              " (" + std::to_string(n_layers) + " layers x " + std::to_string(n_expert) + " experts x int32)";
        return false;
    }
    f.seekg(0);
    labels.resize((size_t) (n_layers * n_expert));
    if (!f.read(reinterpret_cast<char*>(labels.data()), want)) { err = path + ": short read"; return false; }
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] < 0 || labels[i] >= n_groups) {
            err = path + ": label " + std::to_string(labels[i]) + " at (layer " + std::to_string(i / (size_t) n_expert) +
                  ", expert " + std::to_string(i % (size_t) n_expert) + ") is outside 0.." + std::to_string(n_groups - 1);
            return false;
        }
    }
    return true;
}

bool latent_colscale_read(const std::string& pack_dir, int64_t n_layers, int64_t width, std::vector<float>& inv,
                          std::string& err) {
    inv.clear();
    if (n_layers <= 0 || width <= 0) { err = "scales: the geometry is empty"; return false; }
    {
        std::ifstream probe(pack_dir + "/scales/l0.f32", std::ios::binary);
        if (!probe) return true;   // no scales: the pack's rows are not column-scaled
    }
    inv.reserve((size_t) (n_layers * width));
    for (int64_t l = 0; l < n_layers; ++l) {
        const std::string p = pack_dir + "/scales/l" + std::to_string(l) + ".f32";
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) { err = "scales: " + p + " is missing (all-or-none)"; return false; }
        const std::streamsize want = (std::streamsize) (width * 4);
        if (f.tellg() != want) {
            err = "scales: " + p + ": size " + std::to_string((long long) f.tellg()) + " != " +
                  std::to_string((long long) want);
            return false;
        }
        f.seekg(0);
        std::vector<float> s((size_t) width);
        if (!f.read(reinterpret_cast<char*>(s.data()), want)) { err = "scales: " + p + ": short read"; return false; }
        for (float v : s) inv.push_back(v > 0.0f ? 1.0f / v : 1.0f);
    }
    return true;
}

}  // namespace strata::core