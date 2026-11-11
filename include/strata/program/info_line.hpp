// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/info_line.hpp - the INFO line the engines report to serve at READY: one writer,
// each engine adds its own fields in its own order (the sets differ; the form and the engine= tail are the same).
// serve/server.py reads the key=value facts.
#pragma once

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace strata::program {

class InfoLine {
public:
    InfoLine& add(const char* key, const char* v) { parts_.emplace_back(key, v); return *this; }
    InfoLine& add(const char* key, const std::string& v) { return add(key, v.c_str()); }
    InfoLine& add(const char* key, long long v) { return add(key, std::to_string(v).c_str()); }
    InfoLine& add(const char* key, int v) { return add(key, (long long) v); }
    InfoLine& add(const char* key, size_t v) { return add(key, (long long) v); }
    /// the %.2f fields (pcie_frac, spec_min_p)
    InfoLine& add_f(const char* key, double v) {
        char b[64];
        std::snprintf(b, sizeof b, "%.2f", v);
        return add(key, b);
    }
    void print() const {
        std::string s = "INFO";
        for (const auto& kv : parts_) { s += ' '; s += kv.first; s += '='; s += kv.second; }
        s += " engine=" STRATA_VERSION "\n";
        std::fputs(s.c_str(), stdout);
    }

private:
    std::vector<std::pair<std::string, std::string>> parts_;
};

}  // namespace strata::program
