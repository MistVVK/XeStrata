// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/spec/static_drafter.cpp - see include/strata/spec/static_drafter.hpp.
#include "strata/spec/static_drafter.hpp"

#include "strata/platform/env.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::spec {
namespace {
uint64_t little(const unsigned char* p, int n) {
    uint64_t v = 0;
    for (int i = 0; i < n; ++i) v |= (uint64_t) p[i] << (8 * i);
    return v;
}
}

bool StaticDrafter::load(const std::string& path, int32_t vocab, std::string& err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || vocab <= 0) { err = "YD1S: cannot open table or invalid vocabulary"; return false; }
    const auto end = f.tellg();
    if (end < 20) { err = "YD1S: truncated header"; return false; }
    f.seekg(0);
    unsigned char h[20];
    if (!f.read(reinterpret_cast<char*>(h), sizeof h) || std::memcmp(h, "YD1S", 4) != 0) {
        err = "YD1S: invalid header"; return false;
    }
    const uint64_t n = little(h + 4, 8), pairs = little(h + 12, 8), bytes = (uint64_t) end;
    if (n != (uint64_t) vocab || n > (bytes - 20) / 4 || pairs > (bytes - 20 - 4 * n) / 12 ||
        20 + 4 * n + 12 * pairs != bytes || pairs > n * n) {
        err = "YD1S: invalid sizes or vocabulary"; return false;
    }
    std::vector<int32_t> uni((size_t) n);
    std::unordered_map<uint64_t, int32_t> bi;
    auto token = [vocab](uint64_t x) {
        return x == std::numeric_limits<uint32_t>::max() || x < (uint32_t) vocab;
    };
    unsigned char b[12];
    for (int32_t& v : uni) {
        if (!f.read(reinterpret_cast<char*>(b), 4) || !token(little(b, 4))) {
            err = "YD1S: invalid unigram"; return false;
        }
        const uint32_t u = (uint32_t) little(b, 4);
        v = u == UINT32_MAX ? -1 : (int32_t) u;
    }
    for (uint64_t i = 0; i < pairs; ++i) {
        if (!f.read(reinterpret_cast<char*>(b), 12)) { err = "YD1S: truncated bigram"; return false; }
        const uint64_t key = little(b, 8), value = little(b + 8, 4);
        if (key >= n * n || value >= n || !bi.emplace(key, (int32_t) value).second) {
            err = "YD1S: invalid or duplicate bigram"; return false;
        }
    }
    uni_ = std::move(uni);
    bi_ = std::move(bi);
    return true;
}

int StaticDrafter::configure(int32_t vocab) {
    const char* mode = strata::env_get("DF2_STATIC");
    int result = -1;
    if (mode && *mode && std::strcmp(mode, "auto") != 0) {
        if (std::strcmp(mode, "0") == 0) result = 0;
        else if (std::strcmp(mode, "1") == 0) result = 1;
        else throw std::runtime_error("XESTRATA_DF2_STATIC must be auto, 0 or 1");
    }
    if (result == 0) return result;
    const char* path = strata::env_get("DF2_STATIC_TABLE");
    if (!path || !*path) {
        if (result == 1) throw std::runtime_error("XESTRATA_DF2_STATIC=1 needs XESTRATA_DF2_STATIC_TABLE");
        return 0;
    }
    std::string err;
    if (!load(path, vocab, err)) throw std::runtime_error(err + ": " + path);
    return result;
}

int32_t StaticDrafter::next(int32_t prev, int32_t cur) const {
    const int64_t n = (int64_t) uni_.size();
    if (cur < 0 || cur >= n) return -1;
    if (prev >= 0 && prev < n) {
        auto it = bi_.find((uint64_t) prev * (uint64_t) n + (uint32_t) cur);
        if (it != bi_.end()) return it->second;
    }
    return uni_[(size_t) cur];
}

int StaticDrafter::propose(int32_t prev, int32_t cur, int max, int32_t* out) const {
    int n = 0;
    while (n < max) {
        const int32_t t = next(prev, cur);
        if (t < 0) break;
        out[n++] = t;
        prev = cur;
        cur = t;
    }
    return n;
}

}  // namespace strata::spec
