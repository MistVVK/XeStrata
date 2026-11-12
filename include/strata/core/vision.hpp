// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/vision.hpp - the vision encoder the chat's picture reader and the image engine share: Qwen3-VL's
// ViT and its merger, from the model's mmproj GGUF, on the GPU.  A picture (RGB in -1 .. 1, sides multiples of 32) to
// the text model's image embeddings, one per 32 x 32 pixels (2 x 2 patches merged), and the DeepStack ones the text
// encoder adds after its first layers.
#pragma once

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

private:
    struct W {
        const uint16_t* w = nullptr;   // FP16 [n][k]
        const float* b = nullptr;
        int64_t n = 0, k = 0;
    };
    struct Norm {
        const float *w = nullptr, *b = nullptr;
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
    W linear(const std::string& name);
    Norm norm(const std::string& name);
    void product(const uint16_t* x16, const W& w, float* y, int64_t T);
    void grow(int64_t patches);
    void merger(const Merger& m, const float* x, int64_t rows, int64_t in_dim, bool norm_merged, std::vector<float>& out);

    void* s_ = nullptr;
    bool dry_ = false;
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
