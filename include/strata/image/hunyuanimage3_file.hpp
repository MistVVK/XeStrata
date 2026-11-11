// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/hunyuanimage3_file.hpp - HunyuanImage 3.0's weights as its safetensors files hold them: the quantized
// matrices and the routed experts' banks, read in place.
//
// The files are single checkpoints (PedroMarinhoDev's HunyuanImage-3.0-*-ComfyUI), in two quantized forms, and the form
// is in each quantized tensor's `comfy_quant` JSON:
//
//   * `int8_tensorwise` with `convrot` (81.9 GB, the original's weights at 8 bits): `weight` I8 [E][N][K] and
//     `weight_scale` F32 [E][N][1], one scale a row; K's groups of `convrot_groupsize` (256) were rotated when the
//     file was made (W_rot = W @ H^T a group), so the *activations* are rotated alike before the product:
//     x W^T = (x H) (W H)^T for H orthogonal.  THE ROTATION IS NOT SYLVESTER'S: it is the **regular** Hadamard,
//     kron(H4, H4, H4, H4) with H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] / 2, which XeStrata's kernels::rows_i8
//     does not use (that one is (-1)^popcount(r & c) / sqrt(b)); the two disagree, and rotating with the wrong one
//     gives plausible-looking noise (cos 0.05, measured 2026-10-23).  h5/H4 is symmetric, so H^T = H.
//   * `asym_w4a8_int8` (46.9 GB, the default): `weight` I8 [E][N][K/2] (two 4-bit codes a byte),
//     `weight_s_rel` U8 [E][N][K/16] (an FP8 e4m3 scale for each group of `group_size` (16) columns),
//     `weight_s_channel` F32 [E][N] (a scale a row) and `weight_codebook` F32 [16] for a matrix, [E][16] for a bank
//     (the 16 levels; the banks' tables were measured the same for every expert, CALIB-2026-10-23.md, but a row is
//     read with its own expert's table).  The value is codebook[code] * s_rel * s_channel, in the rotated
//     basis, so the engine either rotates its activations as above or dequantizes to FP16 and multiplies.
//
// E is 1 for a matrix (qkv, o, the shared expert) and 64 for a bank of routed experts; N and K are the matrix's.
// Everything else in the file (the embeddings, the routers, the norms, patch_embed and final_layer) is BF16.
//
// This header is the READER only: sizes, offsets and dequantization on the host.  What the engine puts on the GPU,
// and how it streams the banks it cannot hold, is hunyuanimage3_dit's.
#pragma once

#include "strata/platform/mapped_file.hpp"
#include "strata/image/safetensors.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::image {

/// How a matrix is stored (the file's `comfy_quant`).
enum class HunyuanImage3Quant { none, int8_rotated, w4a8 };

/// One quantized matrix, or a bank of `experts` of them, with the bytes still in the file.
struct HunyuanImage3Matrix {
    HunyuanImage3Quant quant = HunyuanImage3Quant::none;
    std::string name;
    int64_t experts = 1, n = 0, k = 0;
    const int8_t* weight = nullptr;    ///< int8_rotated: [E][N][K]; w4a8: [E][N][K/2]
    const float* row_scale = nullptr;  ///< int8_rotated: `weight_scale` [E][N][1]
    const uint8_t* s_rel = nullptr;    ///< w4a8: `weight_s_rel` [E][N][K/group]
    const float* s_channel = nullptr;  ///< w4a8: `weight_s_channel` [E][N]
    const float* codebook = nullptr;   ///< w4a8: `weight_codebook` [16], or [E][16] for a bank
    int64_t codebook_rows = 1;         ///< how many tables the tensor holds: expert e reads `e`'s, at e * 16
    int64_t group = 16;                ///< w4a8: the columns one s_rel value covers
    int64_t rotate = 256;              ///< the Hadamard group K was rotated in when the file was made (0: none)

    /// The bytes the weight's rows take (a bank's, all its experts): what a budget and a copy are planned with.
    uint64_t weight_bytes() const {
        return (uint64_t) experts * (uint64_t) n * (uint64_t) (quant == HunyuanImage3Quant::w4a8 ? k / 2 : k);
    }
};

/// Whether a DiT file is laid out as a HunyuanImage 3.0 model: its header says so (`format`), or a matrix carries a
/// `comfy_quant` (a quantized save's description of its form), or it holds the model's own pieces (`model.wte.weight`
/// beside `patch_embed.model.0.weight`).  The header alone is read (no mapping), so strata-image can pick its pipeline:
/// a HunyuanImage save and a Mage-Flow save share the `.safetensors` extension.  False on a file that is not a
/// safetensors one (a GGUF) or whose header cannot be read.
bool hunyuanimage3_is_dit_file(const std::string& path);

class HunyuanImage3File {
public:
    /// Reads the header and maps the file.  `header_only`: the tensors' offsets are not checked against the file's
    /// size (a file fetched by HTTP range requests for its header alone).
    bool open(const std::string& path, std::string& err, bool header_only = false);

    /// A metadata value of the header (`config`: the model's config JSON, `format`: "hunyuan_image_3_quantized"),
    /// or "" when the file has none.
    const std::string& meta(const std::string& key) const;
    /// Whether the tensors are laid out as a quantized HunyuanImage 3.0 file (its `format`).
    bool quantized() const;

    const SafeTensors& tensors() const {
        if (!st_) throw std::logic_error("HunyuanImage3File: open() has not been called");
        return *st_;
    }

    /// The quantized storage of a matrix or a bank by its tensor name (the `...weight`), e.g.
    /// "model.layers.0.mlp.experts_gate_up_proj.weight".  False with `err` when the file has no such quantized
    /// matrix (or its description disagrees with its shapes).
    bool matrix(const std::string& name, HunyuanImage3Matrix& m, std::string& err) const;

    /// Rows [r0, r1) of expert `e` (0 for a matrix) as FP32, dequantized into `out` ([r1-r0][k]).  The values are in
    /// the file's rotated basis (the caller rotates its activations alike, or rotates these rows back with H).
    /// w4a8: `low_first` picks which half of a byte holds the even column (see set_low_first).
    void rows(const HunyuanImage3Matrix& m, int64_t e, int64_t r0, int64_t r1, float* out, bool low_first) const;

    /// Which half of a w4a8 byte holds the even column 2i (the default: the low one, as the files have it; measured
    /// against the BF16 weights, 2026-10-23: the low order gives cos 0.9973, the high one 0.02).
    static void set_low_first(bool low) { low_first_default_ = low; }
    static bool low_first() { return low_first_default_; }

    /// The FP8 e4m3 value of one byte (the w4a8 groups' scales; no inf, 448 the largest).
    static float fp8_e4m3(uint8_t b);

    /// A tensor's bytes as the mapping holds them, with their length: a quantized matrix's storage, its scales and
    /// codebook, an embedding.  What a copy to the GPU is planned with; nothing is read here.  False with `err`
    /// when the file has no such tensor or its offsets fall outside it.
    bool raw(const std::string& name, const uint8_t*& out, uint64_t& bytes, std::string& err) const;

    /// How many decoder layers the file holds (the largest N with a `model.layers.N.self_attn.qkv_proj.weight`).
    int64_t layer_count() const;

    /// A BF16 tensor (a norm's weight, the router, an embedding) as FP32 into `out`, `out` = `elements` floats.  False
    /// with `err` when the file has no such BF16 tensor of that many values.
    bool bf16(const std::string& name, float* out) const;

    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::unique_ptr<SafeTensors> st_;
    std::unique_ptr<platform::MappedFile> file_;
    std::string err_;

    static bool low_first_default_;
};

}  // namespace strata::image