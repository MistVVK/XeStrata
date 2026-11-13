// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/artifact/yskn.hpp - YASAKANI's .yskn container (yasakani-dev/yasakani 0205f06: src/format.hpp,
// src/signs.hpp, src/loader.cpp, the embedding and the products of src/q35_kernels.cu and src/lnord.cu, written anew).
//
// The file: "YSKN", a u32 version (1) at 4, the JSON header's length (u32) at 8, the JSON at 0x20, the payloads at the
// offsets the JSON gives.  Each matrix is a trellis code: a row's columns in units of 256 values (T), each unit 128
// steps of two values (d = 2) and K = `bits` bits a step (4 K u32 words).  The decoder's state is the L = 12 bits of
// the unit's bit stream that end at bit K (j + 1) (zero bits below the stream's start: start "zero"), the state picks
// a pair of FP16 values from the tensor's table (lut16[2 s], lut16[2 s + 1], 4,096 pairs), and the row's FP16 scale
// multiplies them: the row's values in the rotated basis.  The rotation is a signed Hadamard transform over 512 values
// (H, orthonormal): a matrix W is stored as Wq with W = S_out H Wq H S_in, the signs S drawn from a seeded stream
// (RotDesc); a matrix without rot_out (the head) is W = Wq H S_in, and the embedding's row v is (Wq[v] H) S_in.
//
// Only what YASAKANI's qwen3_5 engine takes is accepted (L 12, start "zero", H512 rotations, no per-tile K): the reader
// refuses the rest with the tensor's name.
#pragma once

#include "strata/platform/mapped_file.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::yskn {

constexpr int kStateBits = 12;           ///< L
constexpr int kStates = 1 << kStateBits; ///< the table's pairs
constexpr int kUnit = 256;               ///< T: the values of a unit
constexpr int kSteps = kUnit / 2;        ///< two values a step
constexpr int kGroup = 512;              ///< the rotation's block, and the activations' scale group

struct RotDesc {
    std::string kind = "H512";
    int m = 1;                           ///< 512-value blocks
    uint64_t sign_seed = 0, sign_offset = 0, qm_off = 0;
    std::string rule = "splitmix64";
};

struct TensorDesc {
    std::string name, role, start = "zero", layout = "warp";
    int rows = 0, cols = 0, physical_rows = 0, L = kStateBits, d = 2, T = kUnit, bits = 6;
    RotDesc rot_in, rot_out;
    bool has_rot_out = false;
    uint64_t lut_off = 0, scale_off = 0, stream_off = 0, stream_bytes = 0, bias_off = 0;
    bool has_rar = false;
    int seqs() const { return cols / kUnit; }
    int words() const { return 4 * bits; }   ///< u32 words a unit
};

struct VectorDesc {
    std::string name;
    uint64_t off = 0;
    int64_t n = 0;
    std::string dtype = "f16";
};

struct ModelDesc {
    std::string arch;
    int layers = 0, hidden = 0, ffn = 0, heads = 0, kv_heads = 0, head_dim = 0, vocab = 0;
    float eps = 0.f, rope_theta = 0.f;
    bool tie_embeddings = false;
    std::vector<std::string> layer_types;   ///< "full_attention" / "linear_attention" (qwen3_5)
};

/// The signs of a rotation: n values from `offset` of the stream `rule` seeds (YASAKANI's splitmix64, or the
/// mulberry32 one its tensors without a rot_in take); +1 / -1.  Throws on another rule.
std::vector<int8_t> sign_stream(const std::string& rule, uint64_t seed, uint64_t offset, size_t n);

class Container {
public:
    /// Maps `path` and reads its header; throws std::runtime_error on a file this reader does not take.
    explicit Container(const std::string& path);
    /// The header of `bytes` (a whole file in memory, for the tests); `bytes` must outlive the container.
    Container(const uint8_t* bytes, size_t size);

    const ModelDesc& model() const { return model_; }
    const std::vector<TensorDesc>& tensors() const { return tensors_; }
    const std::vector<VectorDesc>& vectors() const { return vectors_; }
    const TensorDesc* find_tensor(const std::string& name) const;
    const VectorDesc* find_vector(const std::string& name) const;
    const std::string& path() const { return path_; }
    size_t size() const { return size_; }
    /// `bytes` bytes at `off`, checked to lie inside the file
    const uint8_t* at(uint64_t off, uint64_t bytes) const;

    const uint16_t* lut16(const TensorDesc& t) const;     ///< 2 * 4096 FP16 values
    const uint16_t* scale16(const TensorDesc& t) const;   ///< physical_rows FP16 values
    const uint32_t* stream(const TensorDesc& t) const;
    /// The 4 K words of row r's unit s, gathered from the file's layout ("warp": 32 (unit, row) pairs, rows fastest,
    /// interleaved by 16 bytes; "row": a row's units one after another)
    void unit_words(const TensorDesc& t, int r, int s, uint32_t* out) const;

private:
    void parse(const std::string& json);
    std::unique_ptr<platform::MappedFile> file_;
    const uint8_t* base_ = nullptr;
    size_t size_ = 0;
    std::string path_;
    ModelDesc model_;
    std::vector<TensorDesc> tensors_;
    std::vector<VectorDesc> vectors_;
};

// ---- the reference: the formulas YASAKANI's kernels compute, on the CPU (the GPU kernels are checked against it)

/// The state of step j (0..127) of a unit of K-bit steps (start "zero")
inline uint32_t unit_state(const uint32_t* w, int K, int j) {
    const int e = K * (j + 1);
    if (e < kStateBits) return (w[0] & ((1u << e) - 1u)) << (kStateBits - e);
    const int start = e - kStateBits, word = start >> 5, sh = start & 31;
    uint32_t v = w[word] >> sh;
    if (sh + kStateBits > 32) v |= w[word + 1] << (32 - sh);
    return v & (kStates - 1u);
}

/// The table as int8 pairs (YASAKANI's build_lut8_tau): one scale gamma = max |lut16| / 127 for the whole table, each
/// value rounded to nearest (ties to even) and clamped to +-127, the low byte the first value; tau[r] = scale16[r] *
/// gamma, so a weight is tau[r] * int8
void lut8_tau(const Container& c, const TensorDesc& t, std::vector<uint16_t>& lut8, std::vector<float>& tau);

/// Row r of the tensor in the rotated basis (lut16 times the row's scale, in FP32): cols values
void row_rotated(const Container& c, const TensorDesc& t, int r, float* out);

/// The orthonormal Hadamard transform of each 512-value block of x (n a multiple of 512), in place
void fwht512(float* x, size_t n);

/// The embedding's row of token v (role "embedding", layout "row"): sign_in * H(row in the rotated basis); `extra` the
/// residual tables (embed_tokens.res, .res2) whose rows add before the rotation
void embedding_row(const Container& c, const TensorDesc& t, const std::vector<const TensorDesc*>& extra,
                   const std::vector<int8_t>& signs_in, int v, float* out);

/// The int8 activations of rows x (n values each, rotated: signs, H, rounded to FP16), one scale a 512-group (amax /
/// 127); q [n], alpha [n / 512]
void quantize_rotated(const float* x, const std::vector<int8_t>& signs_in, int n, int8_t* q, float* alpha);

/// y = W x as YASAKANI's integer products: x quantized as quantize_rotated, each 512-group's int32 dot of the int8
/// weights (lut8) and activations, times the group's fixed-point scale (yskn_sum.hpp), summed in int64; then tau, the
/// output's Hadamard transform and signs (rot_out).  y has physical_rows values (the rows past `rows` are padding).
void gemv_reference(const Container& c, const TensorDesc& t, const float* x, float* y);

/// The words of the tensor in the device's reading order: unit s's words in chunks of four (16 bytes), each chunk the
/// physical rows one after another - [s][chunk][row][4] - so the lanes of a sub-group, consecutive rows, read
/// consecutive 16 bytes whatever the sub-group's width
std::vector<uint32_t> device_order(const Container& c, const TensorDesc& t);

}  // namespace strata::yskn
