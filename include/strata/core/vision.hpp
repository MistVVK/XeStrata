// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/vision.hpp - the vision encoder the chat's picture reader and the image engine share: Qwen3-VL's
// ViT and its merger, from the model's mmproj GGUF, on the GPU (open) or on the CPU (open_cpu: the chat's reader on a
// machine without a GPU to spare).  A picture (RGB in -1 .. 1, sides multiples of 32) to the text model's image
// embeddings, one per 32 x 32 pixels (2 x 2 patches merged), and the DeepStack ones the text encoder adds after its
// first layers.  The two share the reading of the file, the patches and the positions; the CPU's products take int8
// weights and activations by default (W8A8, kernels/cpu/i8_gemm.hpp: about twice as fast and further from the
// reference), or with XESTRATA_VISION_CPU_I8=0 BF16 weights (kernels/cpu/bf16_gemm.hpp) and FP32 activations, the
// closer form (kernels/cpu/vision_cpu.hpp).
#pragma once

#include "strata/kernels/cpu/vision_cpu.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata {
class GgufFile;
}

namespace strata::core {

/// The encoder's output: rows tokens of dim (main), and one such block per DeepStack layer (host FP32).
struct VisionOut {
    int64_t rows = 0, dim = 0, grid_h = 0, grid_w = 0;   // grid: the merged tokens' rows and columns
    std::vector<float> main;
    std::vector<std::vector<float>> deepstack;
};

class Vision {
public:
    Vision();
    ~Vision();
    Vision(const Vision&) = delete;
    Vision& operator=(const Vision&) = delete;

    /// Reads the mmproj GGUF and puts its weights on the current GPU (`dry`: the sizes only).
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false);
    /// Reads the mmproj GGUF for the CPU: the products' weights packed as int8 (XESTRATA_VISION_CPU_I8=0: in BF16),
    /// `threads` threads (the caller included) to run on.  The CPU must have AVX2 (kernels/cpu/expert_layout.hpp's
    /// cpu_avx2_ok).
    bool open_cpu(const std::string& path, int threads, std::string& err);
    /// Whether the CPU's products are int8 (the default; XESTRATA_VISION_CPU_I8=0: the BF16 ones), after open_cpu.
    bool cpu_i8() const { return i8_; }

    /// rgb [3][height][width] (host, -1 .. 1) -> the embeddings; the tokens in rows of the merged grid.
    VisionOut encode(const float* rgb, int64_t width, int64_t height);

    uint64_t weight_bytes() const { return weight_bytes_; }
    /// The GPU memory an encode of an image of `tokens` patches works in.
    uint64_t scratch_bytes(int64_t patches) const;
    /// Frees the working buffers.
    void drop_scratch();
    /// The patch's side and the patches a merged token holds (the mmproj's: a picture's sides are multiples of their
    /// product, as the reader resizes them).
    int64_t patch() const { return patch_; }
    int64_t merge() const { return merge_; }
    /// The embeddings' width (the text model's).
    int64_t dim() const { return out_dim_; }

private:
    struct W {
        const uint16_t* w = nullptr;   // FP16 [n][k] (the GPU's)
        const float* b = nullptr;      // the GPU's, or the file's on the CPU
        int64_t n = 0, k = 0;
        kernels::cpu::Bf16Matrix b16;  // the CPU's with XESTRATA_VISION_CPU_I8=0, or
        kernels::cpu::I8Matrix q8;     // the CPU's, the default
    };
    struct Norm {
        const float *w = nullptr, *b = nullptr;   // the GPU's, or the file's on the CPU
    };
    struct Layer {
        Norm ln1, ln2;
        W qkv, o, up, down;
        int deepstack = -1;   // the DeepStack merger it feeds, -1 none
    };
    struct Merger {
        Norm norm;
        W fc1, fc2;
    };
    bool read(const std::string& path, std::string& err);
    W linear(const std::string& name);
    Norm norm(const std::string& name);
    void layout(const float* rgb, int64_t width, int64_t height, std::vector<float>& patches, std::vector<int32_t>& yx,
                std::vector<float>& pos) const;
    VisionOut encode_cpu(const float* rgb, int64_t width, int64_t height);
    void product(const uint16_t* x16, const W& w, float* y, int64_t T);
    void product_cpu(const float* x, const W& w, float* y, int64_t T);
    void grow(int64_t patches);
    void merger(const Merger& m, const float* x, int64_t rows, int64_t in_dim, bool norm_merged, std::vector<float>& out);
    void merger_cpu(const Merger& m, const float* x, int64_t rows, int64_t in_dim, bool norm_merged,
                    std::vector<float>& out);

    void* s_ = nullptr;
    bool dry_ = false;
    // the CPU's: the pool, the patch embedding and the working rows
    std::unique_ptr<kernels::cpu::ExpertPool> pool_;
    int host_cpu_ = -1;
    bool i8_ = false;   // the int8 products (the default; XESTRATA_VISION_CPU_I8=0: BF16)
    kernels::cpu::I8GemmIsa isa_ = kernels::cpu::I8GemmIsa::scalar;
    kernels::cpu::I8Rows rows8_;
    W patch16_;
    kernels::cpu::VisionAttention attn_;
    std::vector<float> cx_, ch_, cy_, cbig_;
    std::unique_ptr<GgufFile> g_;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    uint64_t weight_bytes_ = 0;
    int64_t C_ = 0, heads_ = 0, hd_ = 0, ffn_ = 0, patch_ = 16, merge_ = 2, out_dim_ = 0, grid_ = 0;
    float eps_ = 1e-6f;
    const uint16_t* patch_w_ = nullptr;   // FP16 [C][3 * patch * patch], the two temporal kernels summed
    const float* patch_b_ = nullptr;
    std::vector<float> pos_;              // the learned position embeddings [grid * grid][C] (host)
    std::vector<Layer> layers_;
    Merger final_;
    std::vector<Merger> deepstack_;
    // working buffers
    int64_t cap_ = 0;
    std::vector<void*> work_;
    float *x_ = nullptr, *y_ = nullptr, *big_ = nullptr;
    uint16_t *h16_ = nullptr, *q16_ = nullptr, *k16_ = nullptr, *v16_ = nullptr, *a16_ = nullptr, *o16_ = nullptr;
    int32_t* yx_ = nullptr;
};

}  // namespace strata::core
