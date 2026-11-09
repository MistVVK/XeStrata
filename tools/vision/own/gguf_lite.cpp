// tools/vision/own/gguf_lite.cpp - see gguf_lite.h
#include "gguf_lite.h"

#include <cmath>
#include <cstdlib>

#if defined(_WIN32)
#define STRATA_FSEEK _fseeki64
#else
#define STRATA_FSEEK fseeko
#endif

namespace strata_vit {

float gguf_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;                                    // +-0
        } else {                                            // subnormal: normalise
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; --exp; }
            man &= 0x3FFu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);            // inf, nan
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

namespace {

struct File {
    FILE* f = nullptr;
    explicit File(const std::string& p) { f = std::fopen(p.c_str(), "rb"); }
    ~File() { if (f) std::fclose(f); }
};

struct Cursor {
    FILE* f;
    uint64_t pos = 0, size;
    std::string err;
    bool take(void* dst, uint64_t n) {
        if (n > size - pos) { err = "the GGUF header is cut short"; return false; }
        if (n && std::fread(dst, 1, (size_t) n, f) != (size_t) n) { err = "the GGUF header could not be read"; return false; }
        pos += n;
        return true;
    }
    template <class T> bool num(T& v) { return take(&v, sizeof v); }
    bool str(std::string& s, uint64_t cap = (1ull << 26)) {
        uint64_t n = 0;
        if (!num(n)) return false;
        if (n > cap) { err = "a string in the GGUF header is too long"; return false; }
        s.resize((size_t) n);
        return n == 0 || take(&s[0], n);
    }
};

size_t scalar_size(uint32_t t) {
    switch (t) {
        case GV_UINT8: case GV_INT8: case GV_BOOL: return 1;
        case GV_UINT16: case GV_INT16: return 2;
        case GV_UINT32: case GV_INT32: case GV_FLOAT32: return 4;
        case GV_UINT64: case GV_INT64: case GV_FLOAT64: return 8;
        default: return 0;
    }
}

bool is_float_type(uint32_t t) { return t == GV_FLOAT32 || t == GV_FLOAT64; }

// the value of one scalar as stored (little endian, as this reader's hosts are)
void scalar_value(uint32_t t, const unsigned char* b, int64_t& i, double& d) {
    switch (t) {
        case GV_UINT8: case GV_BOOL: i = b[0]; d = (double) i; break;
        case GV_INT8: i = (int8_t) b[0]; d = (double) i; break;
        case GV_UINT16: { uint16_t v; std::memcpy(&v, b, 2); i = v; d = (double) i; break; }
        case GV_INT16: { int16_t v; std::memcpy(&v, b, 2); i = v; d = (double) i; break; }
        case GV_UINT32: { uint32_t v; std::memcpy(&v, b, 4); i = v; d = (double) i; break; }
        case GV_INT32: { int32_t v; std::memcpy(&v, b, 4); i = v; d = (double) i; break; }
        case GV_UINT64: { uint64_t v; std::memcpy(&v, b, 8); i = (int64_t) v; d = (double) v; break; }
        case GV_INT64: { int64_t v; std::memcpy(&v, b, 8); i = v; d = (double) v; break; }
        case GV_FLOAT32: { float v; std::memcpy(&v, b, 4); d = v; i = (int64_t) v; break; }
        case GV_FLOAT64: { double v; std::memcpy(&v, b, 8); d = v; i = (int64_t) v; break; }
        default: i = 0; d = 0; break;
    }
}

uint64_t tensor_nbytes(uint32_t type, int64_t nelem, int64_t row) {
    switch (type) {
        case GT_F32: return (uint64_t) nelem * 4;
        case GT_F16: case GT_BF16: return (uint64_t) nelem * 2;
        case GT_Q8_0: return row % 32 == 0 ? (uint64_t) (nelem / 32) * 34 : 0;
        default: return 0;
    }
}

}  // namespace

Gguf::~Gguf() = default;

bool Gguf::open(const std::string& path, std::string& err) {
    kv_.clear(); tensors_.clear(); by_name_.clear();
    path_ = path;
    File file(path);
    if (!file.f) { err = "cannot open " + path; return false; }
    STRATA_FSEEK(file.f, 0, SEEK_END);
    const long long end = (long long) (
#if defined(_WIN32)
        _ftelli64(file.f)
#else
        ftello(file.f)
#endif
    );
    STRATA_FSEEK(file.f, 0, SEEK_SET);
    if (end < 24) { err = "not a GGUF file (too small)"; return false; }
    file_size_ = (uint64_t) end;
    Cursor c{file.f, 0, file_size_, {}};
    char magic[4];
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!c.take(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) { err = c.err.empty() ? "not a GGUF file" : c.err; return false; }
    if (!c.num(version) || !c.num(n_tensors) || !c.num(n_kv)) { err = c.err; return false; }
    if (version < 2 || version > 3) { err = "GGUF version " + std::to_string(version) + " is not handled"; return false; }
    if (n_kv > 100000 || n_tensors > 1000000) { err = "the GGUF header counts are not plausible"; return false; }
    for (uint64_t k = 0; k < n_kv; ++k) {
        std::string key;
        Value v;
        if (!c.str(key, 1 << 16) || !c.num(v.type)) { err = c.err; return false; }
        if (v.type == GV_STRING) {
            if (!c.str(v.s)) { err = c.err; return false; }
        } else if (v.type == GV_ARRAY) {
            if (!c.num(v.elem) || !c.num(v.n)) { err = c.err; return false; }
            if (v.n > 50000000ull) { err = "an array in the GGUF header is too long"; return false; }
            if (v.elem == GV_STRING) {
                v.as.resize((size_t) v.n);
                for (auto& s : v.as) if (!c.str(s)) { err = c.err; return false; }
            } else if (scalar_size(v.elem)) {
                const size_t sz = scalar_size(v.elem);
                std::vector<unsigned char> raw((size_t) v.n * sz);
                if (!c.take(raw.data(), raw.size())) { err = c.err; return false; }
                v.ai.resize((size_t) v.n);
                v.ad.resize((size_t) v.n);
                for (size_t j = 0; j < (size_t) v.n; ++j) scalar_value(v.elem, raw.data() + j * sz, v.ai[j], v.ad[j]);
            } else { err = "an array of arrays in the GGUF header is not handled"; return false; }
        } else if (scalar_size(v.type)) {
            unsigned char raw[8] = {0};
            if (!c.take(raw, scalar_size(v.type))) { err = c.err; return false; }
            scalar_value(v.type, raw, v.i, v.d);
        } else { err = "a value type (" + std::to_string(v.type) + ") in the GGUF header is not handled"; return false; }
        kv_[key] = std::move(v);
    }
    for (uint64_t t = 0; t < n_tensors; ++t) {
        GgufTensorInfo ti;
        uint32_t nd = 0;
        if (!c.str(ti.name, 1 << 16) || !c.num(nd)) { err = c.err; return false; }
        if (nd == 0 || nd > 4) { err = "tensor " + ti.name + " has " + std::to_string(nd) + " dimensions"; return false; }
        ti.ne.resize(nd);
        for (uint32_t j = 0; j < nd; ++j) {
            uint64_t d = 0;
            if (!c.num(d)) { err = c.err; return false; }
            if (d == 0 || d > (1ull << 40)) { err = "tensor " + ti.name + " has an implausible size"; return false; }
            ti.ne[j] = (int64_t) d;
        }
        if (!c.num(ti.type) || !c.num(ti.offset)) { err = c.err; return false; }
        ti.nbytes = tensor_nbytes(ti.type, ti.nelem(), ti.ne[0]);
        by_name_[ti.name] = tensors_.size();
        tensors_.push_back(std::move(ti));
    }
    uint32_t align = 32;
    get_u32("general.alignment", align);
    if (align == 0 || (align & (align - 1)) != 0) { err = "general.alignment is not a power of two"; return false; }
    data_start_ = (c.pos + align - 1) / align * align;
    for (const auto& t : tensors_) {
        if (t.nbytes && (t.offset > file_size_ || data_start_ + t.offset + t.nbytes > file_size_)) {
            err = "tensor " + t.name + " lies beyond the end of the file";
            return false;
        }
    }
    return true;
}

bool Gguf::get_u32(const std::string& key, uint32_t& v) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.type == GV_STRING || it->second.type == GV_ARRAY || is_float_type(it->second.type)) return false;
    if (it->second.i < 0 || it->second.i > 0xFFFFFFFFll) return false;
    v = (uint32_t) it->second.i;
    return true;
}

bool Gguf::get_f32(const std::string& key, float& v) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.type == GV_STRING || it->second.type == GV_ARRAY) return false;
    v = (float) it->second.d;
    return true;
}

bool Gguf::get_bool(const std::string& key, bool& v) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.type == GV_STRING || it->second.type == GV_ARRAY || is_float_type(it->second.type)) return false;
    v = it->second.i != 0;
    return true;
}

bool Gguf::get_str(const std::string& key, std::string& v) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.type != GV_STRING) return false;
    v = it->second.s;
    return true;
}

bool Gguf::get_f32_array(const std::string& key, std::vector<float>& v) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.type != GV_ARRAY || it->second.elem == GV_STRING) return false;
    v.assign(it->second.ad.begin(), it->second.ad.end());
    return true;
}

bool Gguf::get_int_array(const std::string& key, std::vector<int64_t>& v) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.type != GV_ARRAY || it->second.elem == GV_STRING || is_float_type(it->second.elem)) return false;
    v = it->second.ai;
    return true;
}

const GgufTensorInfo* Gguf::tensor(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

bool Gguf::read_at(uint64_t pos, void* dst, uint64_t n, std::string& err) const {
    File file(path_);
    if (!file.f) { err = "cannot open " + path_; return false; }
    if (STRATA_FSEEK(file.f, (long long) pos, SEEK_SET) != 0) { err = "cannot seek in " + path_; return false; }
    // in steps, so one huge read cannot fail on a platform that limits it
    unsigned char* p = (unsigned char*) dst;
    while (n) {
        const size_t step = (size_t) (n < (64ull << 20) ? n : (64ull << 20));
        if (std::fread(p, 1, step, file.f) != step) { err = "the tensor data is cut short in " + path_; return false; }
        p += step;
        n -= step;
    }
    return true;
}

bool Gguf::read_raw(const GgufTensorInfo& t, void* dst, std::string& err) const {
    if (t.nbytes == 0) { err = "tensor " + t.name + ": type " + std::to_string(t.type) + " is not handled by this reader (F32, F16, BF16 and Q8_0 are)"; return false; }
    return read_at(data_start_ + t.offset, dst, t.nbytes, err);
}

bool Gguf::read_f32(const GgufTensorInfo& t, float* dst, std::string& err) const {
    if (t.nbytes == 0) return read_raw(t, nullptr, err);
    const int64_t n = t.nelem();
    if (t.type == GT_F32) return read_raw(t, dst, err);
    std::vector<unsigned char> raw((size_t) t.nbytes);
    if (!read_raw(t, raw.data(), err)) return false;
    if (t.type == GT_F16) {
        for (int64_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, raw.data() + 2 * i, 2); dst[i] = gguf_f16_to_f32(h); }
    } else if (t.type == GT_BF16) {
        for (int64_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, raw.data() + 2 * i, 2); dst[i] = gguf_bf16_to_f32(h); }
    } else {   // Q8_0: blocks of {half scale, 32 x int8}
        for (int64_t b = 0; b < n / 32; ++b) {
            const unsigned char* blk = raw.data() + 34 * b;
            uint16_t h;
            std::memcpy(&h, blk, 2);
            const float d = gguf_f16_to_f32(h);
            for (int j = 0; j < 32; ++j) dst[b * 32 + j] = d * (float) (int8_t) blk[2 + j];
        }
    }
    return true;
}

}  // namespace strata_vit
