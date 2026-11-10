// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/platform/downgrade.cpp - see include/strata/platform/downgrade.hpp.

#include "strata/platform/downgrade.hpp"

#include <cstdio>
#include <mutex>

namespace strata::platform {

namespace {
std::mutex g_mutex;
std::vector<std::pair<std::string, int64_t>> g_all;
std::vector<std::string> g_new;
}  // namespace

void downgrade(const char* key, const std::string& why) {
    const std::lock_guard<std::mutex> lock(g_mutex);
    for (auto& [k, n] : g_all)
        if (k == key) {
            ++n;
            return;
        }
    g_all.emplace_back(key, 1);
    g_new.emplace_back(key);
    std::fprintf(stderr, "xestrata: downgrade %s: %s\n", key, why.c_str());
    std::fflush(stderr);
}

std::vector<std::string> downgrades_new() {
    const std::lock_guard<std::mutex> lock(g_mutex);
    std::vector<std::string> out;
    out.swap(g_new);
    return out;
}

void downgrades_say() {
    const std::vector<std::string> keys = downgrades_new();
    if (keys.empty()) return;
    for (const std::string& k : keys) std::printf("INFO downgrade=%s\n", k.c_str());
    std::fflush(stdout);
}

std::vector<std::pair<std::string, int64_t>> downgrades_all() {
    const std::lock_guard<std::mutex> lock(g_mutex);
    return g_all;
}

}  // namespace strata::platform
