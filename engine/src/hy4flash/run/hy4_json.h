// hy4_json.h - minimal JSON DOM parser for the Hy4-Flash runner (manifest is 14.7 MB incl. an 8.5 MB string)
#pragma once
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hy4 {

struct JVal {
    enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
    bool b = false;
    double num = 0;
    int64_t i64 = 0;
    bool is_int = false;
    std::string s;
    std::vector<JVal> a;
    std::vector<std::pair<std::string, JVal>> o;

    const JVal* get(const std::string& k) const {
        for (auto& kv : o) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    const JVal& at(const std::string& k) const {
        const JVal* v = get(k);
        if (!v) throw std::runtime_error("json: missing key " + k);
        return *v;
    }
    const JVal& operator[](size_t i) const { return a.at(i); }
    size_t size() const { return t == Arr ? a.size() : o.size(); }
    double f() const { return num; }
    int64_t i() const { return is_int ? i64 : (int64_t)num; }
    const std::string& str() const { return s; }
};

class JsonParser {
public:
    JsonParser(const char* p, size_t n) : p_(p), e_(p + n) {}
    JVal parse() {
        JVal v = value();
        ws();
        return v;
    }
private:
    const char* p_;
    const char* e_;
    void ws() { while (p_ < e_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_; }
    [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json parse: ") + m); }
    JVal value() {
        ws();
        if (p_ >= e_) fail("eof");
        char c = *p_;
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') { JVal v; v.t = JVal::Str; v.s = string(); return v; }
        if (c == 't') { p_ += 4; JVal v; v.t = JVal::Bool; v.b = true; return v; }
        if (c == 'f') { p_ += 5; JVal v; v.t = JVal::Bool; v.b = false; return v; }
        if (c == 'n') { p_ += 4; return JVal(); }
        return number();
    }
    JVal number() {
        const char* s = p_;
        bool isint = true;
        if (*p_ == '-') ++p_;
        while (p_ < e_ && ((*p_ >= '0' && *p_ <= '9') || *p_ == '.' || *p_ == 'e' || *p_ == 'E' || *p_ == '+' || *p_ == '-')) {
            if (*p_ == '.' || *p_ == 'e' || *p_ == 'E') isint = false;
            ++p_;
        }
        if (p_ == s) fail("bad number");
        std::string tmp(s, p_ - s);
        JVal v; v.t = JVal::Num; v.num = strtod(tmp.c_str(), nullptr); v.is_int = isint;
        if (isint) v.i64 = strtoll(tmp.c_str(), nullptr, 10);
        return v;
    }
    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
        else { out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
    }
    std::string string() {
        ++p_;  // opening quote
        std::string out;
        while (p_ < e_ && *p_ != '"') {
            if (*p_ == '\\') {
                ++p_;
                char c = *p_++;
                switch (c) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case '/': out += '/'; break;
                    case '\\': out += '\\'; break;
                    case '"': out += '"'; break;
                    case 'u': {
                        uint32_t cp = (uint32_t)strtoul(std::string(p_, 4).c_str(), nullptr, 16);
                        p_ += 4;
                        if (cp >= 0xD800 && cp < 0xDC00 && p_ + 6 <= e_ && p_[0] == '\\' && p_[1] == 'u') {
                            uint32_t lo = (uint32_t)strtoul(std::string(p_ + 2, 4).c_str(), nullptr, 16);
                            p_ += 6;
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        put_utf8(out, cp);
                        break;
                    }
                    default: fail("bad escape");
                }
            } else {
                const char* s = p_;
                while (p_ < e_ && *p_ != '"' && *p_ != '\\') ++p_;
                out.append(s, p_ - s);
            }
        }
        if (p_ >= e_) fail("unterminated string");
        ++p_;
        return out;
    }
    JVal array() {
        ++p_;
        JVal v; v.t = JVal::Arr;
        ws();
        if (*p_ == ']') { ++p_; return v; }
        for (;;) {
            v.a.push_back(value());
            ws();
            if (*p_ == ',') { ++p_; continue; }
            if (*p_ == ']') { ++p_; break; }
            fail("bad array");
        }
        return v;
    }
    JVal object() {
        ++p_;
        JVal v; v.t = JVal::Obj;
        ws();
        if (*p_ == '}') { ++p_; return v; }
        for (;;) {
            ws();
            if (*p_ != '"') fail("bad key");
            std::string k = string();
            ws();
            if (*p_ != ':') fail("no colon");
            ++p_;
            v.o.emplace_back(std::move(k), value());
            ws();
            if (*p_ == ',') { ++p_; continue; }
            if (*p_ == '}') { ++p_; break; }
            fail("bad object");
        }
        return v;
    }
};

inline JVal json_parse(const std::string& s) { return JsonParser(s.data(), s.size()).parse(); }

}  // namespace hy4
