// tools/vision/own/gguf_lite.h - a small GGUF reader for Strata's own vision tower (no ggml, no llama.cpp).
//
// Reads the header (key/value pairs, tensor directory) of a GGUF v2/v3 file and the data of single tensors, converted to float32:
// F32, F16, BF16 and Q8_0 are handled (what an mmproj file comes in); any other tensor type is an error that names it.
// Shapes are kept as the file stores them (ne[0] is the fastest dimension: a Linear weight of "[in, out]" is `out` rows of `in`
// contiguous values, y = W x).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace strata_vit {

enum GgufValueType : uint32_t {
    GV_UINT8 = 0, GV_INT8 = 1, GV_UINT16 = 2, GV_INT16 = 3, GV_UINT32 = 4, GV_INT32 = 5, GV_FLOAT32 = 6, GV_BOOL = 7,
    GV_STRING = 8, GV_ARRAY = 9, GV_UINT64 = 10, GV_INT64 = 11, GV_FLOAT64 = 12,
};

// ggml's tensor type ids (the ones this reader knows)
enum GgufTensorType : uint32_t { GT_F32 = 0, GT_F16 = 1, GT_Q8_0 = 8, GT_BF16 = 30 };

struct GgufTensorInfo {
    std::string name;
    std::vector<int64_t> ne;   // as stored: ne[0] fastest
    uint32_t type = 0;
    uint64_t offset = 0;       // from the start of the data section
    uint64_t nbytes = 0;       // 0 when the type is not one this reader handles
    int64_t nelem() const {
        int64_t n = 1;
        for (int64_t d : ne) n *= d;
        return n;
    }
};

class Gguf {
public:
    Gguf() = default;
    Gguf(const Gguf&) = delete;
    Gguf& operator=(const Gguf&) = delete;
    ~Gguf();

    // Reads the header and the tensor directory (never the tensor data).  False and `err` when the file is not a GGUF this reader handles.
    bool open(const std::string& path, std::string& err);

    const std::string& path() const { return path_; }
    uint64_t file_size() const { return file_size_; }

    bool has(const std::string& key) const { return kv_.count(key) != 0; }
    bool get_u32(const std::string& key, uint32_t& v) const;       // any integer type holding a value in 0 .. 2^32-1 (bool counts)
    bool get_f32(const std::string& key, float& v) const;          // a float or an integer
    bool get_bool(const std::string& key, bool& v) const;          // a bool or an integer
    bool get_str(const std::string& key, std::string& v) const;
    bool get_f32_array(const std::string& key, std::vector<float>& v) const;     // an array of floats or integers
    bool get_int_array(const std::string& key, std::vector<int64_t>& v) const;   // an array of integers or bools

    const GgufTensorInfo* tensor(const std::string& name) const;
    const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }

    // the tensor's values as float32 (t.nelem() of them); false and `err` on a type this reader does not handle or a short file
    bool read_f32(const GgufTensorInfo& t, float* dst, std::string& err) const;
    // the tensor's bytes as stored (t.nbytes of them)
    bool read_raw(const GgufTensorInfo& t, void* dst, std::string& err) const;

private:
    struct Value {
        uint32_t type = 0, elem = 0;       // elem: the array's element type
        int64_t i = 0;
        double d = 0;
        std::string s;
        std::vector<int64_t> ai;
        std::vector<double> ad;
        std::vector<std::string> as;
        uint64_t n = 0;                    // array length
    };
    bool read_at(uint64_t pos, void* dst, uint64_t n, std::string& err) const;
    std::string path_;
    uint64_t file_size_ = 0, data_start_ = 0;
    std::map<std::string, Value> kv_;
    std::vector<GgufTensorInfo> tensors_;
    std::map<std::string, size_t> by_name_;
};

// half and bfloat16 to float, one value
float gguf_f16_to_f32(uint16_t h);
inline float gguf_bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    static_assert(sizeof f == sizeof u, "float is 32 bits");
    std::memcpy(&f, &u, sizeof f);
    return f;
}

}  // namespace strata_vit
