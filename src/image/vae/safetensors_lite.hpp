// src/image/vae/safetensors_lite.hpp - read the tensor table of a .safetensors file (the format: an 8-byte little-endian
// header length, a JSON object {name: {"dtype", "shape", "data_offsets": [begin, end]}, "__metadata__": {...}}, then the
// data).  Only what a model file of ours needs: string keys, the three fields above, numbers and arrays of numbers; the
// "__metadata__" object is skipped.  No third-party JSON code.  Header-only, host-only.
#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace strata::image {

struct StTensor {
    std::string dtype;               // "BF16", "F16", "F32", ...
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;     // byte offsets from the start of the data section
};

struct StFile {
    std::map<std::string, StTensor> tensors;
    uint64_t data_start = 0;         // file offset of the data section
};

namespace st_detail {
struct Parser {
    const std::string& s;
    size_t i = 0;
    std::string err;
    explicit Parser(const std::string& t) : s(t) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
    bool eat(char c) { ws(); if (i < s.size() && s[i] == c) { ++i; return true; } return false; }
    bool str(std::string& out) {
        ws();
        if (i >= s.size() || s[i] != '"') return fail("string expected");
        ++i;
        out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                if (i + 1 >= s.size()) return fail("bad escape");
                char e = s[i + 1];
                if (e == 'u') {                          // names are ASCII; keep \uXXXX as-is (never matched)
                    out += s.substr(i, 6);
                    i += 6;
                    continue;
                }
                out += (e == 'n') ? '\n' : (e == 't') ? '\t' : e;
                i += 2;
                continue;
            }
            out += s[i++];
        }
        if (i >= s.size()) return fail("unterminated string");
        ++i;
        return true;
    }
    bool num(int64_t& v) {
        ws();
        size_t j = i;
        if (j < s.size() && s[j] == '-') ++j;
        while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
        if (j == i) return fail("number expected");
        v = std::stoll(s.substr(i, j - i));
        i = j;
        return true;
    }
    bool skip_value() {               // any JSON value (for __metadata__ and unknown fields)
        ws();
        if (i >= s.size()) return fail("value expected");
        char c = s[i];
        if (c == '"') { std::string t; return str(t); }
        if (c == '{' || c == '[') {
            char close = c == '{' ? '}' : ']';
            ++i;
            if (eat(close)) return true;
            do {
                if (c == '{') { std::string k; if (!str(k) || !eat(':')) return fail("object"); }
                if (!skip_value()) return false;
            } while (eat(','));
            return eat(close) || fail("unterminated container");
        }
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;   // number, true, false, null
        return true;
    }
    bool int_array(std::vector<int64_t>& out) {
        out.clear();
        if (!eat('[')) return fail("[ expected");
        if (eat(']')) return true;
        do { int64_t v; if (!num(v)) return false; out.push_back(v); } while (eat(','));
        return eat(']') || fail("] expected");
    }
    bool fail(const char* m) { if (err.empty()) err = std::string(m) + " at byte " + std::to_string(i); return false; }
};
}  // namespace st_detail

// Parse the header of `path` into `out`.  Returns an empty string on success, else the reason.
inline std::string st_read_header(const std::string& path, StFile& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    unsigned char len8[8];
    if (std::fread(len8, 1, 8, f) != 8) { std::fclose(f); return "short file " + path; }
    uint64_t n = 0;
    for (int k = 7; k >= 0; --k) n = (n << 8) | len8[k];
    if (n == 0 || n > (64u << 20)) { std::fclose(f); return "implausible header length in " + path; }
    std::string text(n, '\0');
    if (std::fread(&text[0], 1, n, f) != n) { std::fclose(f); return "short header in " + path; }
    std::fclose(f);
    out = StFile{};
    out.data_start = 8 + n;
    st_detail::Parser p(text);
    if (!p.eat('{')) return "header is not a JSON object";
    if (p.eat('}')) return "";
    do {
        std::string name;
        if (!p.str(name) || !p.eat(':')) return "header: " + p.err;
        if (name == "__metadata__") { if (!p.skip_value()) return "header: " + p.err; continue; }
        StTensor t;
        if (!p.eat('{')) return "header: tensor entry is not an object";
        if (!p.eat('}')) {
            do {
                std::string key;
                if (!p.str(key) || !p.eat(':')) return "header: " + p.err;
                if (key == "dtype") { if (!p.str(t.dtype)) return "header: " + p.err; }
                else if (key == "shape") { if (!p.int_array(t.shape)) return "header: " + p.err; }
                else if (key == "data_offsets") {
                    std::vector<int64_t> o;
                    if (!p.int_array(o) || o.size() != 2 || o[0] < 0 || o[1] < o[0]) return "header: bad data_offsets";
                    t.begin = (uint64_t) o[0];
                    t.end = (uint64_t) o[1];
                } else if (!p.skip_value()) return "header: " + p.err;
            } while (p.eat(','));
            if (!p.eat('}')) return "header: } expected";
        }
        out.tensors[name] = t;
    } while (p.eat(','));
    if (!p.eat('}')) return "header: } expected at the end";
    return "";
}

// Read the raw bytes of one tensor.
inline std::string st_read_tensor(const std::string& path, const StFile& file, const StTensor& t,
                                  std::vector<unsigned char>& bytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    bytes.resize(t.end - t.begin);
#if defined(_WIN32)
    int rc = _fseeki64(f, (long long) (file.data_start + t.begin), SEEK_SET);
#else
    int rc = fseeko(f, (off_t) (file.data_start + t.begin), SEEK_SET);
#endif
    size_t got = rc == 0 ? std::fread(bytes.data(), 1, bytes.size(), f) : 0;
    std::fclose(f);
    return got == bytes.size() ? "" : "short read in " + path;
}

}  // namespace strata::image
