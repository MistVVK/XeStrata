// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/artifact/yskn.cpp - the .yskn reader and the CPU reference of its products (strata/artifact/yskn.hpp).
#include "strata/artifact/yskn.hpp"

#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/yskn_sum.hpp"

#include <picojson.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::yskn {

namespace {

namespace ks = strata::kernels::yskn;

float h2f(uint16_t h) { return strata::kernels::f32_from_f16(h); }
float round_f16(float f) { return strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(f)); }

uint64_t splitmix64(uint64_t& st) {
    st += 0x9E3779B97F4A7C15ull;
    uint64_t z = st;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// picojson keeps numbers as doubles, exact to 2^53 only; a sign seed may be any u64.  Integer literals of 16 digits or
// more are quoted before the parse, and read back from their text.
std::string quote_long_integers(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 64);
    bool str = false;
    for (size_t i = 0; i < in.size(); ++i) {
        const char c = in[i];
        if (str) {
            out += c;
            if (c == '\\' && i + 1 < in.size()) out += in[++i];
            else if (c == '"') str = false;
            continue;
        }
        if (c == '"') { str = true; out += c; continue; }
        if (c >= '0' && c <= '9' && (out.empty() || std::strchr(":,[ \t\r\n", out.back()) != nullptr)) {
            size_t j = i;
            while (j < in.size() && in[j] >= '0' && in[j] <= '9') ++j;
            const bool integer = j >= in.size() || std::strchr(".eE", in[j]) == nullptr;
            if (integer && j - i >= 16) {
                out += '"';
                out.append(in, i, j - i);
                out += '"';
                i = j - 1;
                continue;
            }
        }
        out += c;
    }
    return out;
}

const picojson::value& need(const picojson::object& o, const char* key, const std::string& where) {
    const auto it = o.find(key);
    if (it == o.end()) throw std::runtime_error("yskn: " + where + " has no \"" + key + "\"");
    return it->second;
}
bool has(const picojson::object& o, const char* key) {
    const auto it = o.find(key);
    return it != o.end() && !it->second.is<picojson::null>();
}
uint64_t as_u64(const picojson::value& v, const std::string& what) {
    if (v.is<std::string>()) {
        const std::string& s = v.get<std::string>();
        if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos || s.size() > 20)
            throw std::runtime_error("yskn: " + what + " is not an unsigned integer");
        errno = 0;
        const unsigned long long u = std::strtoull(s.c_str(), nullptr, 10);
        if (errno != 0) throw std::runtime_error("yskn: " + what + " is past 2^64");
        return (uint64_t) u;
    }
    if (!v.is<double>()) throw std::runtime_error("yskn: " + what + " is not a number");
    const double d = v.get<double>();
    if (d < 0 || d != std::floor(d) || d >= 9007199254740992.0)
        throw std::runtime_error("yskn: " + what + " is not an exact unsigned integer");
    return (uint64_t) d;
}
int as_int(const picojson::value& v, const std::string& what) {
    const uint64_t u = as_u64(v, what);
    if (u > 0x7fffffffull) throw std::runtime_error("yskn: " + what + " is too large");
    return (int) u;
}
std::string as_str(const picojson::value& v, const std::string& what) {
    if (!v.is<std::string>()) throw std::runtime_error("yskn: " + what + " is not a string");
    return v.get<std::string>();
}
float as_float(const picojson::value& v, const std::string& what) {
    if (!v.is<double>()) throw std::runtime_error("yskn: " + what + " is not a number");
    return (float) v.get<double>();
}

RotDesc rot_of(const picojson::value& v, const std::string& where) {
    if (!v.is<picojson::object>()) throw std::runtime_error("yskn: " + where + " is not an object");
    const picojson::object& o = v.get<picojson::object>();
    RotDesc r;
    r.kind = as_str(need(o, "kind", where), where + ".kind");
    if (r.kind == "none") return r;
    r.m = as_int(need(o, "m", where), where + ".m");
    r.sign_seed = as_u64(need(o, "sign_seed", where), where + ".sign_seed");
    r.sign_offset = as_u64(need(o, "sign_offset", where), where + ".sign_offset");
    if (has(o, "qm_off")) r.qm_off = as_u64(o.at("qm_off"), where + ".qm_off");
    if (has(o, "rule")) r.rule = as_str(o.at("rule"), where + ".rule");
    return r;
}

// format.hpp's default rotations of a tensor that names none
RotDesc default_rot(int m, uint64_t offset) {
    RotDesc r;
    r.kind = "H512";
    r.m = m;
    r.sign_seed = 7;
    r.sign_offset = offset;
    r.rule = "mulberry32_7919_13";
    return r;
}

}  // namespace

std::vector<int8_t> sign_stream(const std::string& rule, uint64_t seed, uint64_t offset, size_t n) {
    std::vector<int8_t> out;
    out.reserve(n);
    if (rule == "splitmix64") {
        uint64_t st = seed;
        for (uint64_t k = 0; k < offset + n; ++k) {
            const uint64_t v = splitmix64(st);
            if (k >= offset) out.push_back((v >> 63) ? 1 : -1);
        }
        return out;
    }
    if (rule == "mulberry32_7919_13") {
        uint32_t s = (uint32_t) seed * 7919u + 13u;
        if (s == 0) s = 0x9e3779b9u;
        for (uint64_t k = 0; k < offset + n; ++k) {
            uint32_t t = (s += 0x6d2b79f5u);
            t = (t ^ (t >> 15)) * (t | 1u);
            t ^= t + (t ^ (t >> 7)) * (t | 61u);
            t ^= t >> 14;
            if (k >= offset) out.push_back(t < 0x80000000u ? -1 : 1);
        }
        return out;
    }
    throw std::runtime_error("yskn: unknown sign rule " + rule);
}

Container::Container(const std::string& path) : file_(std::make_unique<platform::MappedFile>(path)), path_(path) {
    base_ = file_->data();
    size_ = (size_t) file_->size();
    if (size_ < 0x20 || std::memcmp(base_, "YSKN", 4) != 0) throw std::runtime_error("yskn: " + path + " is not a YSKN container");
    uint32_t version = 0, json_len = 0;
    std::memcpy(&version, base_ + 4, 4);
    std::memcpy(&json_len, base_ + 8, 4);
    if (version != 1) throw std::runtime_error("yskn: " + path + ": version " + std::to_string(version) + " (1 is read)");
    if ((size_t) json_len + 0x20 > size_) throw std::runtime_error("yskn: " + path + ": the header is truncated");
    parse(std::string((const char*) base_ + 0x20, json_len));
}

Container::Container(const uint8_t* bytes, size_t size) : base_(bytes), size_(size), path_("(memory)") {
    if (size_ < 0x20 || std::memcmp(base_, "YSKN", 4) != 0) throw std::runtime_error("yskn: not a YSKN container");
    uint32_t version = 0, json_len = 0;
    std::memcpy(&version, base_ + 4, 4);
    std::memcpy(&json_len, base_ + 8, 4);
    if (version != 1) throw std::runtime_error("yskn: version " + std::to_string(version) + " (1 is read)");
    if ((size_t) json_len + 0x20 > size_) throw std::runtime_error("yskn: the header is truncated");
    parse(std::string((const char*) base_ + 0x20, json_len));
}

void Container::parse(const std::string& text) {
    picojson::value root;
    const std::string e = picojson::parse(root, quote_long_integers(text));
    if (!e.empty()) throw std::runtime_error("yskn: the header's JSON: " + e);
    if (!root.is<picojson::object>()) throw std::runtime_error("yskn: the header is not a JSON object");
    const picojson::object& j = root.get<picojson::object>();
    const std::string header = "the header";

    const picojson::value& mv = need(j, "model", header);
    if (!mv.is<picojson::object>()) throw std::runtime_error("yskn: \"model\" is not an object");
    const picojson::object& m = mv.get<picojson::object>();
    model_.arch = as_str(need(m, "arch", "model"), "model.arch");
    model_.layers = as_int(need(m, "layers", "model"), "model.layers");
    model_.hidden = as_int(need(m, "hidden", "model"), "model.hidden");
    model_.ffn = as_int(need(m, "ffn", "model"), "model.ffn");
    model_.heads = as_int(need(m, "heads", "model"), "model.heads");
    model_.kv_heads = as_int(need(m, "kv_heads", "model"), "model.kv_heads");
    model_.head_dim = as_int(need(m, "head_dim", "model"), "model.head_dim");
    model_.vocab = as_int(need(m, "vocab", "model"), "model.vocab");
    model_.eps = as_float(need(m, "eps", "model"), "model.eps");
    model_.rope_theta = as_float(need(m, "rope_theta", "model"), "model.rope_theta");
    if (has(m, "target_layers")) {
        const auto& taps = m.at("target_layers");
        if (!taps.is<picojson::array>()) throw std::runtime_error("yskn: model.target_layers is not an array");
        for (const auto& tap : taps.get<picojson::array>()) model_.target_layers.push_back(as_int(tap, "model.target_layers[]"));
    }
    if (has(m, "block")) model_.block = as_int(m.at("block"), "model.block");
    if (has(m, "mask_token")) model_.mask_token = as_int(m.at("mask_token"), "model.mask_token");
    if (has(m, "window")) model_.window = as_int(m.at("window"), "model.window");
    if (has(m, "top_k")) model_.top_k = as_int(m.at("top_k"), "model.top_k");
    if (has(m, "rank")) model_.rank = as_int(m.at("rank"), "model.rank");
    if (has(m, "tie_embeddings")) model_.tie_embeddings = m.at("tie_embeddings").evaluate_as_boolean();
    if (has(m, "layer_types")) {
        const picojson::value& lt = m.at("layer_types");
        if (!lt.is<picojson::array>()) throw std::runtime_error("yskn: model.layer_types is not an array");
        for (const picojson::value& v : lt.get<picojson::array>()) model_.layer_types.push_back(as_str(v, "model.layer_types[]"));
    }

    const picojson::value& tv = need(j, "tensors", header);
    if (!tv.is<picojson::array>()) throw std::runtime_error("yskn: \"tensors\" is not an array");
    for (const picojson::value& v : tv.get<picojson::array>()) {
        if (!v.is<picojson::object>()) throw std::runtime_error("yskn: a tensor is not an object");
        const picojson::object& t = v.get<picojson::object>();
        TensorDesc d;
        d.name = as_str(need(t, "name", "a tensor"), "a tensor's name");
        const std::string w = "tensor " + d.name;
        d.role = as_str(need(t, "role", w), w + ".role");
        d.rows = as_int(need(t, "rows", w), w + ".rows");
        d.cols = as_int(need(t, "cols", w), w + ".cols");
        d.physical_rows = has(t, "physical_rows") ? as_int(t.at("physical_rows"), w + ".physical_rows") : d.rows;
        if (has(t, "l")) d.L = as_int(t.at("l"), w + ".l");
        if (has(t, "d")) d.d = as_int(t.at("d"), w + ".d");
        if (has(t, "t")) d.T = as_int(t.at("t"), w + ".t");
        d.bits = as_int(need(t, "bits", w), w + ".bits");
        if (has(t, "start")) d.start = as_str(t.at("start"), w + ".start");
        if (has(t, "layout")) d.layout = as_str(t.at("layout"), w + ".layout");
        d.rot_in = has(t, "rot_in") ? rot_of(t.at("rot_in"), w + ".rot_in") : default_rot(d.cols / kGroup, 0);
        if (t.find("rot_out") != t.end()) {   // present: null or "none" means no output rotation
            if (has(t, "rot_out")) {
                d.rot_out = rot_of(t.at("rot_out"), w + ".rot_out");
                d.has_rot_out = d.rot_out.kind != "none";
            }
        } else if (d.role != "head" && d.role != "embedding") {
            d.rot_out = default_rot(d.rows / kGroup, (uint64_t) d.cols);
            d.has_rot_out = true;
        }
        d.lut_off = as_u64(need(t, "lut_off", w), w + ".lut_off");
        d.scale_off = as_u64(need(t, "scale_off", w), w + ".scale_off");
        d.stream_off = as_u64(need(t, "stream_off", w), w + ".stream_off");
        d.stream_bytes = as_u64(need(t, "stream_bytes", w), w + ".stream_bytes");
        if (has(t, "bias_off")) d.bias_off = as_u64(t.at("bias_off"), w + ".bias_off");
        d.has_rar = has(t, "rar");

        // what YASAKANI's qwen3_5 engine takes (format.hpp's checks, then engine35.cpp's and loader.cpp's)
        if (d.d != 2 || d.T != kUnit || d.L != kStateBits || d.bits < 4 || d.bits > 8 || d.cols <= 0 ||
            d.cols % kGroup != 0 || d.physical_rows < d.rows || d.rows <= 0 || d.physical_rows % 64 != 0)
            throw std::runtime_error("yskn: " + w + ": a shape or code this reader does not take (l " + std::to_string(d.L) +
                                     ", d " + std::to_string(d.d) + ", t " + std::to_string(d.T) + ", bits " +
                                     std::to_string(d.bits) + ", " + std::to_string(d.rows) + " x " + std::to_string(d.cols) + ")");
        if (d.start != "zero") throw std::runtime_error("yskn: " + w + ": start \"" + d.start + "\" (\"zero\" is read)");
        if (d.layout != "warp" && d.layout != "row") throw std::runtime_error("yskn: " + w + ": layout \"" + d.layout + "\"");
        if (d.has_rar) throw std::runtime_error("yskn: " + w + ": per-tile K (rar) is not read");
        if (d.rot_in.kind != "H512" || d.rot_in.m * kGroup != d.cols)
            throw std::runtime_error("yskn: " + w + ": its input rotation is " + d.rot_in.kind + " over " +
                                     std::to_string(d.rot_in.m) + " blocks (H512 over the columns is read)");
        if (d.has_rot_out && (d.rot_out.kind != "H512" || d.rot_out.m * kGroup != d.physical_rows))
            throw std::runtime_error("yskn: " + w + ": its output rotation is " + d.rot_out.kind + " over " +
                                     std::to_string(d.rot_out.m) + " blocks (H512 over the rows is read)");
        const uint64_t want = (uint64_t) d.physical_rows * (uint64_t) d.seqs() * (uint64_t) d.words() * 4u;
        if (d.stream_bytes != want)
            throw std::runtime_error("yskn: " + w + ": stream_bytes " + std::to_string(d.stream_bytes) + ", " +
                                     std::to_string(want) + " expected");
        (void) at(d.lut_off, (uint64_t) kStates * 4u);
        (void) at(d.scale_off, (uint64_t) d.physical_rows * 2u);
        (void) at(d.stream_off, d.stream_bytes);
        tensors_.push_back(d);
    }

    const picojson::value& vv = need(j, "vectors", header);
    if (!vv.is<picojson::array>()) throw std::runtime_error("yskn: \"vectors\" is not an array");
    for (const picojson::value& v : vv.get<picojson::array>()) {
        if (!v.is<picojson::object>()) throw std::runtime_error("yskn: a vector is not an object");
        const picojson::object& o = v.get<picojson::object>();
        VectorDesc d;
        d.name = as_str(need(o, "name", "a vector"), "a vector's name");
        d.off = as_u64(need(o, "off", "vector " + d.name), "vector " + d.name + ".off");
        d.n = (int64_t) as_u64(need(o, "n", "vector " + d.name), "vector " + d.name + ".n");
        if (has(o, "dtype")) d.dtype = as_str(o.at("dtype"), "vector " + d.name + ".dtype");
        const uint64_t es = d.dtype == "f32" || d.dtype == "i32" ? 4 : (d.dtype == "f16" || d.dtype == "bf16") ? 2 : 0;
        if (es == 0) throw std::runtime_error("yskn: vector " + d.name + ": dtype " + d.dtype);
        (void) at(d.off, (uint64_t) d.n * es);
        vectors_.push_back(d);
    }
}

const TensorDesc* Container::find_tensor(const std::string& name) const {
    for (const TensorDesc& t : tensors_)
        if (t.name == name) return &t;
    return nullptr;
}

const VectorDesc* Container::find_vector(const std::string& name) const {
    for (const VectorDesc& v : vectors_)
        if (v.name == name) return &v;
    return nullptr;
}

const uint8_t* Container::at(uint64_t off, uint64_t bytes) const {
    if (off > size_ || bytes > size_ - off)
        throw std::runtime_error("yskn: " + path_ + ": a payload at " + std::to_string(off) + " (" + std::to_string(bytes) +
                                 " bytes) lies past the file's end");
    return base_ + off;
}

const uint16_t* Container::lut16(const TensorDesc& t) const {
    return reinterpret_cast<const uint16_t*>(at(t.lut_off, (uint64_t) kStates * 4u));
}
const uint16_t* Container::scale16(const TensorDesc& t) const {
    return reinterpret_cast<const uint16_t*>(at(t.scale_off, (uint64_t) t.physical_rows * 2u));
}
const uint32_t* Container::stream(const TensorDesc& t) const {
    return reinterpret_cast<const uint32_t*>(at(t.stream_off, t.stream_bytes));
}

void Container::unit_words(const TensorDesc& t, int r, int s, uint32_t* out) const {
    const uint32_t* w = stream(t);
    const int wps = t.words();
    if (t.layout == "row") {
        std::memcpy(out, w + ((size_t) r * t.seqs() + s) * wps, (size_t) wps * 4);
        return;
    }
    // repack.cu's warp layout: pair q = s * rows + r, 32 pairs a tile, each pair's words in 16-byte chunks interleaved
    const size_t q = (size_t) s * t.physical_rows + r, tile = q / 32, lane = q % 32;
    for (int k = 0; k < wps; ++k) out[k] = w[((tile * (size_t) (wps / 4) + (size_t) (k / 4)) * 32 + lane) * 4 + (size_t) (k % 4)];
}

void lut8_tau(const Container& c, const TensorDesc& t, std::vector<uint16_t>& lut8, std::vector<float>& tau) {
    const uint16_t* l16 = c.lut16(t);
    const uint16_t* s16 = c.scale16(t);
    float mx = 0.f;
    for (int i = 0; i < 2 * kStates; ++i) mx = std::max(mx, std::fabs(h2f(l16[i])));
    const float gamma = mx / 127.f;
    lut8.assign(kStates, 0);
    for (int s = 0; s < kStates; ++s) {
        int q0 = gamma > 0.f ? (int) std::nearbyint(h2f(l16[2 * s]) / gamma) : 0;
        int q1 = gamma > 0.f ? (int) std::nearbyint(h2f(l16[2 * s + 1]) / gamma) : 0;
        q0 = std::clamp(q0, -127, 127);
        q1 = std::clamp(q1, -127, 127);
        lut8[(size_t) s] = (uint16_t) ((uint8_t) (int8_t) q0 | ((uint16_t) (uint8_t) (int8_t) q1 << 8));
    }
    tau.resize((size_t) t.physical_rows);
    for (int r = 0; r < t.physical_rows; ++r) tau[(size_t) r] = h2f(s16[r]) * gamma;
}

void row_rotated(const Container& c, const TensorDesc& t, int r, float* out) {
    const uint16_t* l16 = c.lut16(t);
    const float sc = h2f(c.scale16(t)[r]);
    std::vector<uint32_t> w((size_t) t.words());
    for (int s = 0; s < t.seqs(); ++s) {
        c.unit_words(t, r, s, w.data());
        for (int j = 0; j < kSteps; ++j) {
            const uint32_t st = unit_state(w.data(), t.bits, j);
            out[s * kUnit + 2 * j] = h2f(l16[2 * st]) * sc;
            out[s * kUnit + 2 * j + 1] = h2f(l16[2 * st + 1]) * sc;
        }
    }
}

void fwht512(float* x, size_t n) {
    // the shared CPU rotation (cpu/hadamard.hpp, the APR experts' own): the same Sylvester order and 1 / sqrt(512)
    // scale (the float this file's literal was), no signs (the caller applies them)
    strata::kernels::cpu::hadamard_rotate(x, (int) n, kGroup, nullptr);
}

void embedding_row(const Container& c, const TensorDesc& t, const std::vector<const TensorDesc*>& extra,
                   const std::vector<int8_t>& signs_in, int v, float* out) {
    row_rotated(c, t, v, out);
    if (!extra.empty()) {
        std::vector<float> r((size_t) t.cols);
        for (const TensorDesc* e : extra) {
            row_rotated(c, *e, v, r.data());
            for (int i = 0; i < t.cols; ++i) out[i] += r[(size_t) i];
        }
    }
    fwht512(out, (size_t) t.cols);
    for (int i = 0; i < t.cols; ++i) out[i] *= (float) signs_in[(size_t) i];
}

void quantize_rotated(const float* x, const std::vector<int8_t>& signs_in, int n, int8_t* q, float* alpha) {
    std::vector<float> v((size_t) n);
    for (int i = 0; i < n; ++i) v[(size_t) i] = x[i] * (float) signs_in[(size_t) i];
    fwht512(v.data(), (size_t) n);
    for (int g = 0; g < n / kGroup; ++g) {
        float amax = 0.f;
        for (int i = g * kGroup; i < (g + 1) * kGroup; ++i) {
            v[(size_t) i] = round_f16(v[(size_t) i]);
            amax = std::max(amax, std::fabs(v[(size_t) i]));
        }
        const float a = amax / 127.f;
        alpha[g] = a;
        for (int i = g * kGroup; i < (g + 1) * kGroup; ++i) {
            int qi = a > 0.f ? (int) std::nearbyint(v[(size_t) i] / a) : 0;
            q[i] = (int8_t) std::clamp(qi, -127, 127);
        }
    }
}

void gemv_reference(const Container& c, const TensorDesc& t, const float* x, float* y) {
    const int N = t.physical_rows, K = t.cols, G = K / kGroup;
    std::vector<uint16_t> lut8;
    std::vector<float> tau;
    lut8_tau(c, t, lut8, tau);
    const std::vector<int8_t> sin = sign_stream(t.rot_in.rule, t.rot_in.sign_seed, t.rot_in.sign_offset, (size_t) K);
    std::vector<int8_t> q((size_t) K);
    std::vector<float> alpha((size_t) G);
    quantize_rotated(x, sin, K, q.data(), alpha.data());
    float amax = 0.f;
    for (float a : alpha) amax = std::max(amax, a);
    const float factor = ks::fixed_factor(amax);
    std::vector<int64_t> fixed((size_t) G);
    for (int g = 0; g < G; ++g) fixed[(size_t) g] = ks::fixed_scale(alpha[(size_t) g], factor);
    std::vector<uint32_t> w((size_t) t.words());
    for (int r = 0; r < N; ++r) {
        int64_t acc = 0;
        for (int g = 0; g < G; ++g) {
            int32_t dot = 0;
            for (int s = 2 * g; s < 2 * g + 2; ++s) {
                c.unit_words(t, r, s, w.data());
                for (int j = 0; j < kSteps; ++j) {
                    const uint16_t p = lut8[unit_state(w.data(), t.bits, j)];
                    dot += (int32_t) (int8_t) (p & 0xFF) * q[(size_t) (s * kUnit + 2 * j)];
                    dot += (int32_t) (int8_t) (p >> 8) * q[(size_t) (s * kUnit + 2 * j + 1)];
                }
            }
            acc += ks::group_term(dot, fixed[(size_t) g]);
        }
        y[r] = ks::finish(acc, amax, tau[(size_t) r]);
    }
    if (t.has_rot_out) {
        const std::vector<int8_t> so = sign_stream(t.rot_out.rule, t.rot_out.sign_seed, t.rot_out.sign_offset, (size_t) N);
        fwht512(y, (size_t) N);
        for (int r = 0; r < N; ++r) y[r] *= (float) so[(size_t) r];
    }
}

std::vector<uint32_t> device_order(const Container& c, const TensorDesc& t) {
    const int N = t.physical_rows, S = t.seqs(), wps = t.words(), chunks = wps / 4;
    std::vector<uint32_t> out((size_t) N * S * wps);
    std::vector<uint32_t> w((size_t) wps);
    for (int s = 0; s < S; ++s)
        for (int r = 0; r < N; ++r) {
            c.unit_words(t, r, s, w.data());
            for (int k = 0; k < chunks; ++k)
                std::memcpy(&out[(((size_t) s * chunks + k) * N + r) * 4], &w[(size_t) k * 4], 16);
        }
    return out;
}

}  // namespace strata::yskn
