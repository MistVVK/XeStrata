// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/ink_json.h - Minimal self-contained JSON parser for .strata v1.1 manifests
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ink2 {

struct JVal {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false;
    double num = 0.0;
    int64_t inum = 0;
    bool is_int = false;
    std::string s;
    std::vector<JVal> a;
    std::map<std::string, JVal> o;

    int64_t i() const {
        if (t != NUM) throw std::runtime_error("JSON: expected number");
        return is_int ? inum : static_cast<int64_t>(num);
    }
    double f() const {
        if (t != NUM) throw std::runtime_error("JSON: expected number");
        return is_int ? static_cast<double>(inum) : num;
    }
    bool bv() const {
        if (t != BOOL) throw std::runtime_error("JSON: expected bool");
        return b;
    }
    const std::string& str() const {
        if (t != STR) throw std::runtime_error("JSON: expected string");
        return s;
    }
    const JVal& at(const std::string& k) const {
        if (t != OBJ) throw std::runtime_error("JSON: expected object for key " + k);
        auto it = o.find(k);
        if (it == o.end()) throw std::runtime_error("JSON: missing key " + k);
        return it->second;
    }
    const JVal* get(const std::string& k) const {
        if (t != OBJ) return nullptr;
        auto it = o.find(k);
        return it == o.end() ? nullptr : &it->second;
    }
    bool has(const std::string& k) const { return get(k) != nullptr; }
    const JVal& operator[](size_t idx) const {
        if (t != ARR || idx >= a.size()) throw std::runtime_error("JSON: array index out of range");
        return a[idx];
    }
};

namespace detail {

struct JParser {
    const char* p;
    const char* end;

    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
    }
    char peek() {
        ws();
        return p < end ? *p : '\0';
    }
    std::string parse_str() {
        ws();
        if (p >= end || *p != '"') throw std::runtime_error("JSON: expected '\"'");
        ++p;
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\') {
                ++p;
                if (p >= end) throw std::runtime_error("JSON: truncated escape");
                switch (*p++) {
                    case '"':  out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        if (end - p < 4) throw std::runtime_error("JSON: short \\u escape");
                        unsigned cp = 0;
                        for (int k = 0; k < 4; ++k) {
                            char c = *p++;
                            cp <<= 4;
                            if (c >= '0' && c <= '9') cp |= static_cast<unsigned>(c - '0');
                            else if (c >= 'a' && c <= 'f') cp |= static_cast<unsigned>(c - 'a' + 10);
                            else if (c >= 'A' && c <= 'F') cp |= static_cast<unsigned>(c - 'A' + 10);
                            else throw std::runtime_error("JSON: bad hex in \\u");
                        }
                        // Handle UTF-16 surrogate pair
                        if (cp >= 0xD800u && cp <= 0xDBFFu && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                            p += 2;
                            unsigned lo = 0;
                            for (int k = 0; k < 4; ++k) {
                                char c = *p++;
                                lo <<= 4;
                                if (c >= '0' && c <= '9') lo |= static_cast<unsigned>(c - '0');
                                else if (c >= 'a' && c <= 'f') lo |= static_cast<unsigned>(c - 'a' + 10);
                                else if (c >= 'A' && c <= 'F') lo |= static_cast<unsigned>(c - 'A' + 10);
                                else throw std::runtime_error("JSON: bad surrogate hex");
                            }
                            cp = 0x10000u + (((cp - 0xD800u) << 10) | (lo - 0xDC00u));
                        }
                        if (cp < 0x80u) {
                            out.push_back(static_cast<char>(cp));
                        } else if (cp < 0x800u) {
                            out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
                            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
                        } else if (cp < 0x10000u) {
                            out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
                            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
                            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
                        } else {
                            out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
                            out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
                            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
                            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
                        }
                        break;
                    }
                    default: throw std::runtime_error("JSON: unknown escape");
                }
            } else {
                out.push_back(*p++);
            }
        }
        if (p >= end || *p != '"') throw std::runtime_error("JSON: unterminated string");
        ++p;
        return out;
    }

    JVal parse_val() {
        char c = peek();
        if (c == '"') {
            JVal v;
            v.t = JVal::STR;
            v.s = parse_str();
            return v;
        }
        if (c == '{') {
            ++p;
            JVal v;
            v.t = JVal::OBJ;
            if (peek() == '}') { ++p; return v; }
            while (true) {
                std::string k = parse_str();
                if (peek() != ':') throw std::runtime_error("JSON: expected ':'");
                ++p;
                v.o.emplace(std::move(k), parse_val());
                char d = peek();
                if (d == ',') { ++p; continue; }
                if (d == '}') { ++p; break; }
                throw std::runtime_error("JSON: expected ',' or '}'");
            }
            return v;
        }
        if (c == '[') {
            ++p;
            JVal v;
            v.t = JVal::ARR;
            if (peek() == ']') { ++p; return v; }
            while (true) {
                v.a.push_back(parse_val());
                char d = peek();
                if (d == ',') { ++p; continue; }
                if (d == ']') { ++p; break; }
                throw std::runtime_error("JSON: expected ',' or ']'");
            }
            return v;
        }
        if (c == 't' && end - p >= 4 && std::memcmp(p, "true", 4) == 0) {
            p += 4;
            JVal v; v.t = JVal::BOOL; v.b = true; return v;
        }
        if (c == 'f' && end - p >= 5 && std::memcmp(p, "false", 5) == 0) {
            p += 5;
            JVal v; v.t = JVal::BOOL; v.b = false; return v;
        }
        if (c == 'n' && end - p >= 4 && std::memcmp(p, "null", 4) == 0) {
            p += 4;
            JVal v; v.t = JVal::NUL; return v;
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            const char* start = p;
            if (*p == '-') ++p;
            while (p < end && *p >= '0' && *p <= '9') ++p;
            bool flt = false;
            if (p < end && *p == '.') {
                flt = true;
                ++p;
                while (p < end && *p >= '0' && *p <= '9') ++p;
            }
            if (p < end && (*p == 'e' || *p == 'E')) {
                flt = true;
                ++p;
                if (p < end && (*p == '+' || *p == '-')) ++p;
                while (p < end && *p >= '0' && *p <= '9') ++p;
            }
            std::string tok(start, p);
            JVal v;
            v.t = JVal::NUM;
            if (!flt) {
                v.is_int = true;
                v.inum = std::strtoll(tok.c_str(), nullptr, 10);
                v.num = static_cast<double>(v.inum);
            } else {
                v.is_int = false;
                v.num = std::strtod(tok.c_str(), nullptr);
                v.inum = static_cast<int64_t>(v.num);
            }
            return v;
        }
        throw std::runtime_error(std::string("JSON: unexpected char '") + c + "'");
    }
};

}  // namespace detail

inline JVal json_parse(const std::string& text) {
    detail::JParser jp{text.data(), text.data() + text.size()};
    return jp.parse_val();
}

}  // namespace ink2
