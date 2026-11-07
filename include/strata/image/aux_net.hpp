// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/aux_net.hpp - what the image tools' auxiliary models (SAM 3, BiRefNet, Florence-2) share: their
// weights read from a safetensors file onto the GPU (FP16 matrices, FP32 norms and biases), scratch buffers grown on
// demand and kept between requests, and the products and convolutions through the engine's parts (prefill::Gemm,
// qi_conv's oneDNN and cuDNN, else unfolded patches and prefill::Gemm).
#pragma once

#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

class SafeTensors;

class AuxNet {
public:
    /// Opens the file's header; nothing goes to the GPU until load().
    AuxNet(const std::string& path, const char* what);
    virtual ~AuxNet();
    AuxNet(const AuxNet&) = delete;
    AuxNet& operator=(const AuxNet&) = delete;

    /// The bytes it holds on the GPU (the weights once loaded, and its scratch).
    uint64_t held() const;
    uint64_t weight_bytes() const { return weight_bytes_; }
    /// The tensors' bytes in the file (the weights' size on the GPU, about, before they are loaded).
    uint64_t file_bytes() const { return file_bytes_; }
    bool loaded() const { return loaded_; }
    /// Frees the scratch (the weights stay).
    void drop_scratch();
    /// Frees everything on the GPU (the next run loads the weights again).
    void release();

    /// The parity probes' view of the parts: a name and an FP32 host copy.
    std::function<void(const char*, const std::vector<float>&)> dump;

    /// A linear layer: y [T][n] = x [T][k] . w^T + b.
    struct Lin {
        const uint16_t* w = nullptr;   // [n][k] FP16
        const float* b = nullptr;      // [n] or null
        int64_t n = 0, k = 0;
    };
    struct Norm {
        const float *w = nullptr, *b = nullptr;
    };
    /// A k x k convolution, its weight as [o][k][k][c] FP16.
    struct Conv {
        const uint16_t* w = nullptr;
        const float* b = nullptr;
        int64_t o = 0, c = 0;
        int k = 0, stride = 1, pad = 0;
    };

protected:
    /// Puts the weights on the GPU once (the derived class's load_weights()), the stream and the products first.
    void load();
    virtual void load_weights() = 0;

    void* alloc(size_t bytes);
    /// Scratch slot `slot` of at least `bytes`, kept between requests (grown when a request needs more).
    float* scratch(int slot, size_t bytes);
    template <class T> T* scratch_as(int slot, size_t count) {
        return reinterpret_cast<T*>(scratch(slot, count * sizeof(T)));
    }

    /// The linear layer of one or more tensors stacked along n (name.weight, name.bias where there is one); `rows`
    /// takes rows r0 .. r0 + rows of the (stacked) weight only (a packed q, k, v's third).
    Lin lin(const std::vector<std::string>& names, int64_t r0 = 0, int64_t rows = -1);
    Lin lin(const std::string& name, int64_t r0 = 0, int64_t rows = -1) { return lin(std::vector<std::string>{name}, r0, rows); }
    /// A linear layer whose tensors are not name.weight and name.bias (a packed in_proj_weight, in_proj_bias).
    Lin lin_wb(const std::string& w_name, const std::string& b_name);
    /// A packed q, k, v (name.<w_suffix> [3n][k], name.<b_suffix> [3n]) as three layers.
    void lin3(const std::string& w_name, const std::string& b_name, Lin& q, Lin& k, Lin& v);
    Norm norm(const std::string& name, const char* w = ".weight", const char* b = ".bias");
    Conv conv(const std::string& name, int stride, int pad);
    /// A convolution from host weights w [o][c][k][k] and bias b ([o], or empty: none).
    Conv conv_host(const std::vector<float>& w, const std::vector<float>& b, int64_t o, int64_t c, int k, int stride,
                   int pad);
    /// A transposed 2 x 2 convolution of stride 2 (weight [c][o][2][2]) as the product to [(dy, dx, o)][c] and its bias.
    Lin deconv2(const std::string& name);
    const float* f32(const std::string& name);
    const uint16_t* f16(const std::string& name);
    std::vector<float> host_f32(const std::string& name);

    /// y = x16 . l.w^T (+ l.b); ldy: y's row stride (0: l.n).
    void product(const uint16_t* x16, const Lin& l, float* y, int64_t T, int64_t ldy = 0);
    /// y [OH][OW][O] (+= with `add`) = the convolution of x16 [H][W][C] plus its bias; `col_slot` a scratch slot for
    /// the unfolded patches where no library takes it.
    void convolve(const Conv& c, const uint16_t* x16, int64_t H, int64_t W, float* y, int col_slot, bool add = false);
    void host(const char* name, const float* dev, size_t n);

    std::string path_, what_;
    std::unique_ptr<SafeTensors> st_;
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    bool loaded_ = false;
    uint64_t weight_bytes_ = 0, file_bytes_ = 0;

private:
    std::vector<void*> weights_;
    std::vector<std::pair<void*, size_t>> work_;
};

}  // namespace strata::image
