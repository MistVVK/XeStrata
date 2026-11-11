// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/id_list.hpp - the token ids' one reader: the two text engines and the image
// engine had a copy each.  The forms keep each one's contract:
//   Text     (plain): commas or whitespace apart, strtol, [0, 2147483647], false on a bad or empty list
//   TextErr  (generate): the same, from_chars (a leading '+' is not taken), the error wording generate.cpp used
//   Image    (the image engine): commas alone, stol (a token's leading integer, its trailing garbage ignored),
//            its exceptions on a bad id, an empty list allowed
#pragma once

#include <charconv>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::program {

enum class IdListForm { Text, TextErr, Image };

inline bool parse_id_list(const std::string& text, std::vector<int64_t>& out, IdListForm form,
                          std::string* err = nullptr) {
    out.clear();
    if (form == IdListForm::Image) {
        std::stringstream ss(text);
        for (std::string tok; std::getline(ss, tok, ',');)
            if (!tok.empty()) out.push_back((int32_t) std::stol(tok));
        return true;
    }
    if (form == IdListForm::TextErr) {
        std::string spaced(text);
        for (char& c : spaced) if (c == ',') c = ' ';
        std::istringstream input(spaced);
        std::string token;
        while (input >> token) {
            int32_t id = 0;
            const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
            if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
                if (err != nullptr) *err = "invalid token id: expected an integer in [0, 2147483647]";
                out.clear();
                return false;
            }
            out.push_back(id);
        }
        if (out.empty()) { if (err != nullptr) *err = "token list was empty"; return false; }
        return true;
    }
    // Text: commas or whitespace apart, a whole token an integer in [0, 2147483647]
    const char* p = text.c_str();
    while (*p != '\0') {
        while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
        if (*p == '\0') break;
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v < 0 || v > 0x7fffffff) return false;
        out.push_back((int32_t) v);
        p = end;
    }
    return !out.empty();
}

}  // namespace strata::program
