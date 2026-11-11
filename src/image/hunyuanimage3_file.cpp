// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_file.cpp - see include/strata/image/hunyuanimage3_file.hpp.
#include "strata/image/hunyuanimage3_file.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace strata::image {

bool HunyuanImage3File::low_first_default_ = true;

namespace {

namespace st = strata::image;

// A JSON object's integer or string value, by key: the `comfy_quant` descriptions are tiny and flat, and a missing
// key is -1 / "" (the caller decides whether that is an error).  No parser is pulled in for a 90-byte object.
int64_t json_int(const std::string& j, const char* key) {
    const std::string k = std::string("\"") + key + "\"";
    const size_t p = j.find(k);
    if (p == std::string::npos) return -1;
    size_t i = j.find(':', p + k.size());
    if (i == std::string::npos) return -1;
    ++i;
    while (i < j.size() && (j[i] == ' ' || j[i] == '\t')) ++i;
    const size_t b = i;
    while (i < j.size() && ((j[i] >= '0' && j[i] <= '9') || j[i] == '-')) ++i;
    if (i == b) return -1;
    return std::stoll(j.substr(b, i - b));
}

std::string json_str(const std::string& j, const char* key) {
    const std::string k = std::string("\"") + key + "\"";
    const size_t p = j.find(k);
    if (p == std::string::npos) return {};
    size_t i = j.find(':', p + k.size());
    if (i == std::string::npos) return {};
    ++i;
    while (i < j.size() && (j[i] == ' ' || j[i] == '\t')) ++i;
    if (i >= j.size() || j[i] != '"') return {};
    const size_t b = ++i;
    while (i < j.size() && j[i] != '"') ++i;
    return j.substr(b, i - b);
}

// A tensor's bytes, checked against the file before a pointer is handed out: the engine reads whatever the file says,
// and a wrong offset here is a fault inside a kernel rather than an error at start.
const uint8_t* checked(const st::SafeTensors& s, const platform::MappedFile& f, const st::SafeTensor& t, uint64_t bytes,
                       const std::string& path, const std::string& name) {
    if (t.end < t.begin || bytes > t.end - t.begin)
        throw std::runtime_error(path + ": " + name + ": its shape does not match its bytes");
    const uint64_t at = s.data_base() + t.begin;
    if (at + bytes > f.size()) throw std::runtime_error(path + ": " + name + ": data_offsets past the file's end");
    return f.data() + at;
}

}  // namespace

bool hunyuanimage3_is_dit_file(const std::string& path) {
    try {
        const st::SafeTensors headers(path);   // the header alone: no mapping, no weights read
        if (headers.meta("format") == "hunyuan_image_3_quantized") return true;
        if (headers.find("model.wte.weight") != nullptr && headers.find("patch_embed.model.0.weight") != nullptr)
            return true;
        for (const auto& kv : headers.tensors())
            if (kv.first.size() > 12 && kv.first.compare(kv.first.size() - 12, 12, ".comfy_quant") == 0) return true;
        return false;
    } catch (const std::exception&) {
        return false;   // not a safetensors file (a GGUF), or its header cannot be read: another model's pipeline
    }
}

bool HunyuanImage3File::open(const std::string& path, std::string& err, bool header_only) {
    path_ = path;
    try {
        st_ = std::make_unique<st::SafeTensors>(path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    try {
        file_ = std::make_unique<platform::MappedFile>(path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    if (!header_only) {
        const uint64_t base = st_->data_base();
        for (const auto& kv : st_->tensors()) {
            const st::SafeTensor& t = kv.second;
            if (t.end < t.begin || base > file_->size() || t.end > file_->size() - base) {
                err = path + ": " + kv.first + ": data_offsets outside the file";
                return false;
            }
        }
    }
    return true;
}

const std::string& HunyuanImage3File::meta(const std::string& key) const {
    static const std::string none;
    auto it = st_->metadata().find(key);
    return it == st_->metadata().end() ? none : it->second;
}

bool HunyuanImage3File::quantized() const {
    if (meta("format") == "hunyuan_image_3_quantized") return true;
    for (const auto& kv : st_->tensors())
        if (kv.first.size() > 12 && kv.first.compare(kv.first.size() - 12, 12, ".comfy_quant") == 0) return true;
    return false;
}

bool HunyuanImage3File::matrix(const std::string& name, HunyuanImage3Matrix& m, std::string& err) const {
    const st::SafeTensor* t = st_->find(name);
    if (t == nullptr) {
        err = path_ + ": no tensor " + name;
        return false;
    }
    const std::string prefix = name.size() > 7 ? name.substr(0, name.size() - 7) : name;  // without ".weight"
    const st::SafeTensor* q = st_->find(prefix + ".comfy_quant");
    if (q == nullptr) {
        err = path_ + ": " + name + ": no comfy_quant (not a quantized matrix)";
        return false;
    }
    if (q->dtype != "U8") {
        err = path_ + ": " + prefix + ".comfy_quant: dtype " + q->dtype + ", not the U8 JSON";
        return false;
    }
    std::string json;
    try {
        const uint8_t* p = checked(*st_, *file_, *q, q->elements(), path_, prefix + ".comfy_quant");
        json.assign(reinterpret_cast<const char*>(p), (size_t) q->elements());
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    const std::string format = json_str(json, "format");
    HunyuanImage3Matrix out;
    out.name = name;
    // the w4a8 description has no "convrot" flag: its rotation is what convrot_groupsize names
    const int64_t convrot_group = json_int(json, "convrot_groupsize");
    const bool convrot = json.find("\"convrot\": true") != std::string::npos ||
                         json.find("\"convrot\":true") != std::string::npos ||
                         (format == "asym_w4a8_int8" && convrot_group > 0);
    if (t->shape.empty() || t->shape.size() > 3) {
        err = path_ + ": " + name + ": " + std::to_string(t->shape.size()) + " dimensions";
        return false;
    }
    out.experts = t->shape.size() == 3 ? t->shape[0] : 1;
    out.n = t->shape[t->shape.size() - 2];
    const int64_t stored_k = t->shape.back();
    if (format == "int8_tensorwise") {
        out.quant = HunyuanImage3Quant::int8_rotated;
        out.k = stored_k;
        out.rotate = convrot ? std::max<int64_t>(convrot_group, 1) : 0;
        const st::SafeTensor* s = st_->find(prefix + ".weight_scale");
        if (s == nullptr) {
            err = path_ + ": " + name + ": no weight_scale";
            return false;
        }
        out.row_scale = reinterpret_cast<const float*>(checked(*st_, *file_, *s, (uint64_t) out.experts * out.n * 4,
                                                              path_, prefix + ".weight_scale"));
    } else if (format == "asym_w4a8_int8") {
        out.quant = HunyuanImage3Quant::w4a8;
        out.k = 2 * stored_k;
        out.group = json_int(json, "group_size") > 0 ? json_int(json, "group_size") : 16;
        out.rotate = convrot ? std::max<int64_t>(convrot_group, 1) : 0;
        const st::SafeTensor* sr = st_->find(prefix + ".weight_s_rel");
        const st::SafeTensor* sc = st_->find(prefix + ".weight_s_channel");
        const st::SafeTensor* cb = st_->find(prefix + ".weight_codebook");
        if (sr == nullptr || sc == nullptr || cb == nullptr) {
            err = path_ + ": " + name + ": no weight_s_rel, weight_s_channel or weight_codebook";
            return false;
        }
        const uint64_t rows = (uint64_t) out.experts * (uint64_t) out.n;
        out.s_rel = checked(*st_, *file_, *sr, rows * (uint64_t) (out.k / out.group), path_, prefix + ".weight_s_rel");
        out.s_channel = reinterpret_cast<const float*>(
            checked(*st_, *file_, *sc, rows * 4u, path_, prefix + ".weight_s_channel"));
        // a bank holds one table an expert ([E][16]), a matrix one ([16]).  The files were measured to give every
        // expert the same sixteen levels (CALIB-2026-10-23.md), but a row is read with its own expert's table.
        if (cb->shape.empty() || cb->shape.back() != 16 || cb->elements() < (int64_t) out.experts * 16) {
            err = path_ + ": " + prefix + ".weight_codebook: " + std::to_string(cb->elements())
                  + " values, not [" + std::to_string(out.experts) + "][16]";
            return false;
        }
        out.codebook_rows = cb->elements() / 16;
        out.codebook = reinterpret_cast<const float*>(
            checked(*st_, *file_, *cb, (uint64_t) out.codebook_rows * 16 * 4, path_, prefix + ".weight_codebook"));
        if (out.k % out.group != 0 || (out.rotate != 0 && out.k % out.rotate != 0)) {
            err = path_ + ": " + name + ": K " + std::to_string(out.k) + " is not a multiple of its groups";
            return false;
        }
    } else {
        err = path_ + ": " + name + ": comfy_quant format \"" + format + "\" is not one this reader knows";
        return false;
    }
    const uint64_t want = (uint64_t) out.experts * (uint64_t) out.n * (uint64_t) (out.quant == HunyuanImage3Quant::w4a8 ? out.k / 2 : out.k);
    try {
        out.weight = reinterpret_cast<const int8_t*>(checked(*st_, *file_, *t, want, path_, name));
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    m = out;
    return true;
}

float HunyuanImage3File::fp8_e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 0xF, m = b & 0x7;
    float v;
    if (e == 0) {
        v = (float) m * (1.0f / 8.0f) * 0x1p-6f;   // subnormal: m / 8 x 2^-6
    } else if (e == 15) {
        v = m == 7 ? std::nanf("") : (1.0f + (float) m / 8.0f) * 0x1p8f;
    } else {
        v = (1.0f + (float) m / 8.0f) * std::ldexp(1.0f, e - 7);
    }
    return s ? -v : v;
}

namespace {

// A safetensors dtype's element size in bytes, 0 for one this reader does not size.
uint64_t element_bytes(const std::string& dtype) {
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
    if (dtype == "I64" || dtype == "U64") return 8;
    return 0;
}

}  // namespace

bool HunyuanImage3File::raw(const std::string& name, const uint8_t*& out, uint64_t& bytes, std::string& err) const {
    const st::SafeTensor* t = st_->find(name);
    if (t == nullptr) {
        err = path_ + ": no tensor " + name;
        return false;
    }
    const uint64_t es = element_bytes(t->dtype);
    if (es == 0) {
        err = path_ + ": " + name + ": dtype " + t->dtype + " has no size here";
        return false;
    }
    try {
        out = checked(*st_, *file_, *t, es * (uint64_t) t->elements(), path_, name);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    bytes = es * (uint64_t) t->elements();
    return true;
}

int64_t HunyuanImage3File::layer_count() const {
    int64_t n = 0;
    while (st_->find("model.layers." + std::to_string(n) + ".self_attn.qkv_proj.weight") != nullptr) ++n;
    return n;
}

bool HunyuanImage3File::bf16(const std::string& name, float* out) const {
    const st::SafeTensor* t = st_->find(name);
    if (t == nullptr) {
        throw std::runtime_error(path_ + ": no tensor " + name);
    }
    if (t->dtype != "BF16") throw std::runtime_error(path_ + ": " + name + ": dtype " + t->dtype + ", not BF16");
    const uint64_t bytes = (uint64_t) t->elements() * 2;
    const uint8_t* p = checked(*st_, *file_, *t, bytes, path_, name);
    for (int64_t i = 0; i < t->elements(); ++i) {
        const uint32_t bits = (uint32_t) p[(size_t) (2 * i)] | ((uint32_t) p[(size_t) (2 * i + 1)] << 8);
        const uint32_t wide = bits << 16;
        float v;
        std::memcpy(&v, &wide, sizeof v);
        out[i] = v;
    }
    return true;
}

void HunyuanImage3File::rows(const HunyuanImage3Matrix& m, int64_t e, int64_t r0, int64_t r1, float* out, bool low_first) const {
    if (e < 0 || e >= m.experts || r0 < 0 || r1 > m.n || r0 > r1) throw std::runtime_error("HunyuanImage3File::rows: range");
    const int64_t row_bytes = m.quant == HunyuanImage3Quant::w4a8 ? m.k / 2 : m.k;
    for (int64_t r = r0; r < r1; ++r) {
        const uint64_t flat = (uint64_t) e * (uint64_t) m.n + (uint64_t) r;
        const uint8_t* w = reinterpret_cast<const uint8_t*>(m.weight) + flat * (uint64_t) row_bytes;
        float* dst = out + (r - r0) * m.k;
        if (m.quant == HunyuanImage3Quant::int8_rotated) {
            const float s = m.row_scale[flat];
            // the bytes are two's complement: reading them as unsigned turns -28 into 228 and rescales the row
            for (int64_t j = 0; j < m.k; ++j) dst[j] = (float) (int8_t) w[j] * s;
        } else if (m.quant == HunyuanImage3Quant::w4a8) {
            const float sc = m.s_channel[flat];
            const uint8_t* sr = m.s_rel + flat * (uint64_t) (m.k / m.group);
            const float* cb = m.codebook + (m.codebook_rows > 1 ? (uint64_t) e * 16 : 0);
            for (int64_t j = 0; j < m.k; ++j) {
                const uint8_t byte = w[j / 2];
                // the two codes of a byte are the columns 2i (low) and 2i+1 (high): the files are calibrated for one
                // order, and the probe settles which against the BF16 weights
                const bool even = (j & 1) == 0;
                const int code = (even == low_first) ? (byte & 0xF) : (byte >> 4);
                dst[j] = cb[code] * fp8_e4m3(sr[j / m.group]) * sc;
            }
        }
    }
}

}  // namespace strata::image