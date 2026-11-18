// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// src/core/latent.cpp - the latent MoE pack's readers.  See the header.
#include "strata/core/latent.hpp"

#include "strata/platform/env.hpp"
#include "strata/platform/mapped_file.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace strata::core {
namespace {

// A manifest's number at p, read whole up to a blank, the end, or the header's ':', ',' or ')': false where atoll and
// sscanf took the digits up to any other character (or 0 for none).  p moves past it.
bool manifest_number(const char*& p, int64_t& v) {
    char* end = nullptr;
    errno = 0;
    v = std::strtoll(p, &end, 10);
    if (end == p || errno != 0 || (*end != '\0' && std::strchr(" \t:,)", *end) == nullptr)) return false;
    p = end;
    return true;
}

}  // namespace

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

int latent_dense_fmt_env(std::string& err) {
    const char* f = strata::env_get("LATENT_DENSE_FMT");
    if (f == nullptr || f[0] == '\0' || std::strcmp(f, "auto") == 0) return 0;
    if (std::strcmp(f, "4") == 0) return 4;
    if (std::strcmp(f, "8") == 0) return 8;
    err = "XESTRATA_LATENT_DENSE_FMT must be auto, 4 or 8 (got '" + std::string(f) + "')";
    return -1;
}

std::string latent_dense_sibling(const std::string& path, int fmt) {
    const size_t slash = path.find_last_of("/\\");
    const std::string dir = slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
    const std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    if (base != "latent_dense_q4.bin" && base != "latent_dense_q8.bin") return {};
    const std::string other = dir + (fmt == 4 ? "latent_dense_q4.bin" : "latent_dense_q8.bin");
    return std::ifstream(other, std::ios::binary).good() ? other : std::string();
}

std::string latent_labels_path(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return (slash == std::string::npos ? std::string() : path.substr(0, slash + 1)) + "labels.bin";
}

bool LatentDense::open(const std::string& path, int64_t n_layers, int64_t n_groups, int64_t n_embd,
                       int64_t latent_dim, std::string& err, int fmt) {
    close();
    if (n_layers <= 0 || n_groups <= 0 || n_embd <= 0 || latent_dim <= 0 || n_embd % 32 != 0 || latent_dim % 32 != 0) {
        err = path + ": the latent dense geometry is not whole 32-value blocks";
        return false;
    }
    n_layers_ = n_layers; n_groups_ = n_groups; n_embd_ = n_embd; latent_dim_ = latent_dim;
    if (fmt == 0) fmt = latent_dense_fmt_env(err);
    if (fmt < 0) {
        err = path + ": " + err;
        return false;
    }
    try {
        map_ = std::make_unique<platform::MappedFile>(path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    map_size_ = map_->size();
    // auto: the format the file's size is (the two formats' sizes differ for every geometry)
    if (fmt == 0) fmt = map_size_ == bytes_for(n_layers, n_groups, n_embd, latent_dim, 4) ? 4 : 8;
    fmt_ = fmt;
    const uint64_t want = (uint64_t) n_layers * (d_bytes() + u_bytes());
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
                static constexpr char kHead[] = "# strata native experts v";
                const char* p = line.c_str() + sizeof kHead - 1;
                int64_t v = 0;
                if (line.rfind(kHead, 0) != 0 || !manifest_number(p, v) || *p != ':') {
                    err = path + ": not a strata native experts manifest (XeStrata's xs manifests go to expert_layout_load)";
                    return false;
                }
                if (v < 3 || v > 6) {
                    err = path + ": native experts version " + std::to_string(v) + " is not one this reader knows (3..6)";
                    return false;
                }
                m.version = (int) v;
                head = true;
                // the header's "expert_embd N", "expert_ff N" and "(n_expert N, ..."; a key without its number fails
                auto key_number = [&line](const char* key, int64_t& out) {
                    const size_t at = line.find(key);
                    if (at == std::string::npos) return true;
                    const char* q = line.c_str() + at + std::strlen(key);
                    return manifest_number(q, out);
                };
                if (!key_number("expert_embd ", m.expert_embd) || !key_number("expert_ff ", m.expert_ff) ||
                    !key_number("(n_expert ", m.n_expert)) {
                    err = path + ": the header's expert_embd, expert_ff or n_expert is not a number";
                    return false;
                }
            }
            continue;
        }
        if (!head) { err = path + ": a row before the version header"; return false; }
        LatentLayerFmt l;
        int64_t row[5] = {};   // layer gu_type d_type offset blob_bytes, then columns this reader does not use
        const char* p = line.c_str();
        bool ok = true;
        for (int64_t& v : row) ok = ok && manifest_number(p, v);
        if (!ok || row[0] < 0 || row[0] > INT32_MAX || row[1] < INT32_MIN || row[1] > INT32_MAX || row[2] < INT32_MIN ||
            row[2] > INT32_MAX || row[3] < 0) {
            err = path + ": cannot parse the row '";
            err += line + "'";
            return false;
        }
        const int layer = (int) row[0];
        l.gu_type = (int) row[1];
        l.d_type = (int) row[2];
        const uint64_t off = (uint64_t) row[3], blob = row[4] > 0 ? (uint64_t) row[4] : 0;
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

}  // namespace strata::core