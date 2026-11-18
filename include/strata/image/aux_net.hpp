// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/aux_net.hpp - what the image tools' auxiliary models (SAM 3, BiRefNet, Florence-2) share: their
// weights read from a safetensors file onto the GPU (FP16 matrices, FP32 norms and biases), scratch buffers grown on
// demand and kept between requests, and the products and convolutions through the engine's parts (prefill::Gemm,
// qi_conv's oneDNN and cuDNN, else unfolded patches and prefill::Gemm).
#pragma once

#include "strata/core/gpu.hpp"
#include "strata/core/graph.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <tuple>
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
    virtual void drop_scratch();
    /// Waits for the work queued on its stream (what a caller reads of its results is there after).
    void sync();
    /// Frees everything on the GPU (the next run loads the weights again).
    void release();

    /// The parity probes' view of the parts: a name and an FP32 host copy.
    std::function<void(const char*, const std::vector<float>&)> dump;
    /// Told a run's share done so far (0..1) as it goes (the parts' ends, the decoder's steps), by a net that tells
    /// one, and 0 once load() has read the weights; none: nothing told.
    std::function<void(double)> progress;

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
    /// Host values on the GPU (FP32), counted with the weights.
    const float* upload(const std::vector<float>& v);
    /// A linear layer from host weights w [n][k] and bias b ([n], or empty: none).
    Lin lin_host(const std::vector<float>& w, const std::vector<float>& b, int64_t n, int64_t k);
    const uint16_t* f16(const std::string& name);
    std::vector<float> host_f32(const std::string& name);

    /// y = x16 . l.w^T (+ l.b, unless `bias` is false: the reader adds it); ldy: y's row stride (0: l.n).
    void product(const uint16_t* x16, const Lin& l, float* y, int64_t T, int64_t ldy = 0, bool bias = true);
    /// y16 [T][l.n] FP16 = x16 . l.w^T, with `bias` + l.b, else the bias left to the reader (aux_bias_act16,
    /// aux_rope2d_split16h): half the bytes of product's FP32 rows, the bias's pass saved.  The library's half product (Gemm::native_half) or the FP32
    /// product narrowed, whichever the GPU runs the shape in less time (each warmed, then timed, at its first call):
    /// the half product took the B70 (oneMKL) 0.22 ms for SAM 3's 5184 x 3072 x 1024 against 0.29, the RTX 4070
    /// (cuBLAS) 0.35 against 0.58 (2026-11-07), and a library's FP16 output can be the slower (oneDNN's in the LLM's
    /// prompt on the B70).
    void product16(const uint16_t* x16, const Lin& l, uint16_t* y16, int64_t T, bool bias = false,
                   ImageAct act = ImageAct::none);
    /// y [OH][OW][O] (+= with `add`) = the convolution of x16 [H][W][C] plus its bias; `col_slot` a scratch slot for
    /// the unfolded patches where no library takes it.
    void convolve(const Conv& c, const uint16_t* x16, int64_t H, int64_t W, float* y, int col_slot, bool add = false);
    void host(const char* name, const float* dev, size_t n);
    /// `bytes` of host memory to dst on the stream, not waited for: through a host buffer of stage `stage`'s own, kept
    /// (its last copy waited for before it is written again: free where that copy has run).  A blocking gpu::copy
    /// waits for the stream's work before it; a frame's few small uploads so held a tracker's GPU idle.
    void upload_async(void* dst, const void* src, size_t bytes, int stage);

    std::string path_, what_;
    std::unique_ptr<SafeTensors> st_;
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    /// A step run again and again (a decoder's), with a graph or straight (STRATA_AUX_EAGER=0 / 1 fixes it); its
    /// graphs go with the scratch
    core::StepForm steps_{"AUX_EAGER"};
    bool loaded_ = false;
    uint64_t weight_bytes_ = 0, file_bytes_ = 0;

private:
    void free_weights();

    std::vector<void*> weights_;
    std::vector<std::pair<void*, size_t>> work_;
    // product16's way by (T, N, K, the activation and the bias): 0 the FP32 product narrowed, 1 the half product, 2 a
    // library's 1 x 1 convolution
    std::map<std::tuple<int64_t, int64_t, int64_t, int>, int> pick16_;
    static constexpr int kNarrowSlot = 255;                             // product16's FP32 rows when not half
    struct Stage {   // upload_async's
        std::vector<uint8_t> buf;
        gpu::Event* done = nullptr;
    };
    std::map<int, Stage> stages_;
};

/// The control images' networks (cnet_*.cpp: HED, line art, M-LSD, Depth Anything V2's head, YOLOX, DWPose): feature
/// maps [h][w][c] in FP16 (c a multiple of 8, the channels past a layer's own zero), convolutions with their batch norms
/// folded in through the engine's convolutions (oneDNN's and cuDNN's where they take the shape, the bias and a ReLU in
/// their pass; else 1 x 1 ones as products and the others unfolded, through prefill::Gemm), depthwise ones and the
/// rest through the aux_ kernels.  A run's maps take scratch slots in turn from begin_run(), kept between runs.
class AuxMapNet : public AuxNet {
public:
    using AuxNet::AuxNet;

    struct Map {
        uint16_t* p = nullptr;
        int64_t h = 0, w = 0, c = 0;
    };
    /// w [outp][k][k][inp] FP16 (its padding zero; depthwise: [outp][k][k]), b [outp] FP32.
    struct MConv {
        int64_t in = 0, inp = 0, out = 0, outp = 0;
        int k = 1, s = 1, pad = 0, dil = 1;   // dil: the taps' spacing (a dilated convolution)
        bool depthwise = false;
        const uint16_t* w = nullptr;
        const float* b = nullptr;
    };

    static int64_t up8(int64_t n) { return (n + 7) / 8 * 8; }

protected:
    void begin_run() { next_ = 0; }
    /// A run's next scratch buffer of n values of T.
    template <class T> T* buf(size_t n) { return scratch_as<T>(kFirstSlot + next_++, std::max<size_t>(n, 1)); }
    Map map(int64_t h, int64_t w, int64_t c) { return {buf<uint16_t>((size_t) (h * w * c)), h, w, c}; }

    /// A convolution (or a linear layer [out][in]) `w_name` with bias `b_name` ("": none) and the batch norm `bn`
    /// ("": none; its weight, bias, running_mean and running_var, eps) folded in, its inputs padded to `in_pad`
    /// channels (0: up8 of its own); depthwise: weight [C][1][k][k].
    MConv mconv(const std::string& w_name, const std::string& b_name, int stride, int pad, const std::string& bn = "",
                float eps = 1e-5f, int64_t in_pad = 0, bool depthwise = false);
    /// The same from host weights w [out][in][k][k] (depthwise [out][k][k]) and bias b ([out], or empty).
    MConv mconv_host(const std::vector<float>& w, std::vector<float> b, int64_t out, int64_t in, int k, int stride,
                     int pad, int64_t in_pad = 0, bool depthwise = false);
    /// A transposed convolution (torch's weight [in][out][k][k]): `general` false, its kernel its stride (a product
    /// scattered); true, any stride and padding (run_deconv_general: the flipped kernel over zeros put in).
    MConv mdeconv(const std::string& w_name, const std::string& b_name, bool general);

    /// The convolution of x, its bias and act, plus `res` (a map of the output's shape) after them: a new map.
    Map run(const MConv& c, const Map& x, ImageAct a = ImageAct::none, const Map* res = nullptr);
    Map run_deconv(const MConv& c, const Map& x, ImageAct a = ImageAct::none);
    Map run_deconv_general(const MConv& c, const Map& x, int64_t s, int64_t p, int64_t op, ImageAct a = ImageAct::none);
    Map act(const Map& x, ImageAct a);
    Map add(const Map& a, const Map& b);
    Map reflect(const Map& x, int64_t p);
    Map resize(const Map& x, int64_t oh, int64_t ow, bool align_corners);
    Map maxpool(const Map& x, int k, int s, int p);
    /// x with zero rows and columns around it.
    Map pad(const Map& x, int64_t top, int64_t left, int64_t bottom, int64_t right);
    /// The maps side by side along the channels (each of the same h and w).
    Map concat(const std::vector<Map>& parts);
    void instance_norm(Map& x, float eps, ImageAct a);
    /// The map's first channel as FP32 on the host (h w values).
    std::vector<float> host_channel0(const Map& x);

private:
    static constexpr int kFirstSlot = 256, kSumSlot = 254, kColSlot = 253;
    int next_ = 0;
};

}  // namespace strata::image
