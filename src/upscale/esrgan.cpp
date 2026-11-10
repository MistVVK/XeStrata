// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/upscale/esrgan.cpp - see include/strata/upscale/esrgan.hpp.

#include "strata/upscale/esrgan.hpp"

#include "strata/core/device.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/prefill/gemm.hpp"

#include "../image/qi_conv.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::upscale {

using image::SafeTensor;
using image::SafeTensors;
namespace conv = image::conv;

namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

constexpr float kSlope = 0.2f;              // the LeakyReLUs' negative slope
constexpr float kResidual = 0.2f;           // the dense blocks' and the RRDBs' residual scaling

// a 3 x 3 convolution: its FP16 weights [O][3][3][C] and FP32 bias on the host, then on the GPU
struct Layer {
    int64_t C = 0, O = 0;   // C: the input's channels as the GPU reads them (padded to 8), O: the outputs (padded to 8)
    std::vector<uint16_t> w;
    std::vector<float> b;
    uint16_t* dw = nullptr;
    float* db = nullptr;
};

// the tensor's FP32 values from an [O][C][3][3] (PyTorch) weight into [Op][3][3][Cp], times `scale`, zero padded
Layer read_layer(SafeTensors& st, const std::string& name, int64_t cin, int64_t cout, float scale = 1.0f) {
    const SafeTensor* t = st.find(name + ".weight");
    if (t == nullptr || st.find(name + ".bias") == nullptr)
        throw std::runtime_error("RealESRGAN: no " + name + " in the weights");
    if (t->shape.size() != 4 || t->shape[0] != cout || t->shape[1] != cin || t->shape[2] != 3 || t->shape[3] != 3)
        throw std::runtime_error("RealESRGAN: " + name + " is not a 3 x 3 convolution of " + std::to_string(cin) +
                                 " to " + std::to_string(cout) + " channels");
    const std::vector<float> w = st.f32(name + ".weight"), b = st.f32(name + ".bias");
    if ((int64_t) b.size() != cout) throw std::runtime_error("RealESRGAN: " + name + ".bias is not " + std::to_string(cout));
    Layer l;
    l.C = (cin + 7) / 8 * 8;
    l.O = (cout + 7) / 8 * 8;
    l.w.assign((size_t) (l.O * 9 * l.C), 0);
    l.b.assign((size_t) l.O, 0.0f);
    for (int64_t o = 0; o < cout; ++o) {
        l.b[(size_t) o] = b[(size_t) o] * scale;
        for (int64_t c = 0; c < cin; ++c)
            for (int64_t k = 0; k < 9; ++k)
                l.w[(size_t) ((o * 9 + k) * l.C + c)] = f16_from_f32(w[(size_t) ((o * cin + c) * 9 + k)] * scale);
    }
    return l;
}

// The device buffers of one tile shape, kept for the tiles that follow and freed with it (or on release)
struct Scratch {
    int64_t w = 0, h = 0;
    std::vector<void*> held;
    uint64_t bytes = 0;
    uint16_t *in16 = nullptr, *dense = nullptr, *a2 = nullptr, *a4 = nullptr, *b4 = nullptr;
    float *trunk = nullptr, *xr = nullptr, *xd = nullptr;
    ~Scratch() { drop(); }
    void drop() {
        for (void* p : held) gpu::free(p);
        held.clear();
        bytes = 0;
        w = h = 0;
    }
    template <class T> T* get(int64_t n) {
        T* p = nullptr;
        const uint64_t b = (uint64_t) n * sizeof(T);
        if (n > 0 && b > core::max_alloc_bytes())
            throw std::runtime_error("RealESRGAN: a buffer past the device's largest allocation");
        if (!gpu::alloc_device(&p, (size_t) b))
            throw std::runtime_error(std::string("RealESRGAN: GPU memory: ") + gpu::last_error());
        held.push_back(p);
        bytes += b;
        return p;
    }
};

}  // namespace

struct Esrgan::Impl {
    int64_t nf = 0, gc = 0, nb = 0, D = 0;   // the features, the dense growth, the RRDBs, a dense block's widest input
    Layer first, body, up1, up2, hr, last;
    std::vector<Layer> rdb;                   // nb * 3 * 5, in order
    std::vector<void*> weights;               // on the GPU: empty until the first run
    uint64_t weights_bytes = 0;
    void* s = nullptr;                        // the stream
    prefill::Gemm gemm;
    bool gemm_ready = false;
    uint16_t* col = nullptr;                  // the own path's scratch (null: not taken yet)
    float* tmp = nullptr;
    int64_t col_cap = 0, tmp_cap = 0;
    Scratch sc;

    sycl::queue& q() { return core::Runtime::get().stream(s); }

    std::vector<Layer*> layers() {
        std::vector<Layer*> v{&first, &body, &up1, &up2, &hr, &last};
        for (Layer& l : rdb) v.push_back(&l);
        return v;
    }

    void put_weights() {
        if (!weights.empty()) return;
        try {
            for (Layer* l : layers()) {
                if (!gpu::alloc_device(&l->dw, l->w.size() * 2) || !gpu::alloc_device(&l->db, l->b.size() * 4))
                    throw std::runtime_error(std::string("RealESRGAN: GPU memory (weights): ") + gpu::last_error());
                weights.push_back(l->dw);
                weights.push_back(l->db);
                gpu::copy(l->dw, l->w.data(), l->w.size() * 2);
                gpu::copy(l->db, l->b.data(), l->b.size() * 4);
            }
        } catch (...) {
            drop_weights();
            throw;
        }
    }
    void drop_scratch() {
        sc.drop();
        for (void* p : {(void*) col, (void*) tmp})
            if (p != nullptr) gpu::free(p);
        col = nullptr;
        tmp = nullptr;
        col_cap = tmp_cap = 0;
    }
    void drop_weights() {
        for (void* p : weights) gpu::free(p);
        weights.clear();
        for (Layer* l : layers()) l->dw = nullptr, l->db = nullptr;
    }

    // the scratch of a w x h tile (the one held where it is that shape)
    void shape(int64_t w, int64_t h) {
        if (sc.w == w && sc.h == h) return;
        sc.drop();
        const int64_t n = w * h;
        try {
            sc.in16 = sc.get<uint16_t>(n * 8);
            sc.trunk = sc.get<float>(n * nf);
            sc.xr = sc.get<float>(n * nf);
            sc.xd = sc.get<float>(n * nf);
            sc.dense = sc.get<uint16_t>(n * D);
            sc.a2 = sc.get<uint16_t>(4 * n * nf);
            sc.a4 = sc.get<uint16_t>(16 * n * nf);
            sc.b4 = sc.get<uint16_t>(16 * n * nf);
        } catch (...) {
            sc.drop();
            throw;
        }
        sc.w = w;
        sc.h = h;
    }

    // y (FP32 or FP16 by y16, a pixel's stride ldy) = the convolution of x [H][W] (a pixel's stride ldx, the first
    // l.C channels) plus the bias, LeakyReLU'd (`leaky`) or added to y (`add`): the GPU maker's library where it takes
    // the form, else im2col and the engine's products a band of rows at a time
    void conv(const Layer& l, const uint16_t* x, int64_t H, int64_t W, int64_t ldx, void* y, int64_t ldy, bool y16,
              bool leaky, bool add) {
        const int64_t O = l.O, K = 9 * l.C, HW = H * W, cap = conv::col_values(q());
        conv::Request r{x, H, W, l.C, l.dw, O, 3, 1, 1, 1, 1, 1};
        r.bias = l.db;
        r.add = add;
        r.out = y;
        r.form.ldx = ldx;
        r.form.ldy = ldy;
        r.form.y16 = y16;
        r.form.leaky = leaky ? kSlope : -1.0f;
        conv::drive(q(), r, HW, K, cap, false,
            [&](int64_t p0, int64_t n) {
                const int64_t rows = std::min(HW, cap / K);
                if (rows * K > col_cap) {
                    if (col != nullptr) gpu::free(col);
                    col = nullptr;
                    if (!gpu::alloc_device(&col, (size_t) (rows * K) * 2))
                        throw std::runtime_error(std::string("RealESRGAN: GPU memory (products): ") + gpu::last_error());
                    col_cap = rows * K;
                }
                image::qi_im2col16(x, 1, H, W, l.C, 1, 3, 1, 1, H, W, p0, n, col, s, ldx);
            },
            [&](int64_t p0, int64_t n, bool) {
                if (!gemm_ready) {
                    std::string err;
                    if (!gemm.init(s, 16, err)) throw std::runtime_error("RealESRGAN: the products: " + err);
                    gemm_ready = true;
                }
                if (y16 || add) {
                    if (n * O > tmp_cap) {
                        if (tmp != nullptr) gpu::free(tmp);
                        tmp = nullptr;
                        if (!gpu::alloc_device(&tmp, (size_t) (n * O) * 4))
                            throw std::runtime_error(std::string("RealESRGAN: GPU memory (products): ") + gpu::last_error());
                        tmp_cap = n * O;
                    }
                    gemm.f16(col, l.dw, tmp, n, O, K);
                    if (y16)
                        image::qi_conv_store16(tmp, n, O, O, l.db, leaky ? image::ImageAct::leaky : image::ImageAct::none,
                                               kSlope, nullptr, static_cast<uint16_t*>(y) + p0 * ldy, ldy, 0, s);
                    else image::qi_bias_add(static_cast<float*>(y) + p0 * ldy, l.db, n, O, s, tmp, ldy);
                } else {
                    if (ldy != O) throw std::logic_error("RealESRGAN: an FP32 output wider than its convolution");
                    float* yy = static_cast<float*>(y) + p0 * O;
                    gemm.f16(col, l.dw, yy, n, O, K);
                    image::qi_bias_add(yy, l.db, n, O, s);
                }
            }, [] {});
    }

    // One frame: rgba [h][w][4] (RGBA8 on the GPU) -> out [4h][4w][4] (RGBA8 on the GPU)
    void run(const uint8_t* rgba, int64_t w, int64_t h, uint8_t* out);
};

Esrgan::Esrgan(const std::string& path) : impl_(std::make_unique<Impl>()) {
    SafeTensors st(path);
    Impl& m = *impl_;
    const SafeTensor* f = st.find("conv_first.weight");
    const SafeTensor* g = st.find("body.0.rdb1.conv1.weight");
    if (f == nullptr || g == nullptr || f->shape.size() != 4 || g->shape.size() != 4)
        throw std::runtime_error(path + ": not RealESRGAN's RRDBNet weights (conv_first, body.N.rdbM.convK)");
    if (f->shape[1] != 3)   // x2's input is pixel-unshuffled (12 channels): not taken
        throw std::runtime_error(path + ": an RRDBNet of " + std::to_string(f->shape[1]) +
                                 " input channels (only RealESRGAN x4's 3)");
    if (st.find("conv_up1.weight") == nullptr || st.find("conv_up2.weight") == nullptr)
        throw std::runtime_error(path + ": not a 4x RRDBNet (conv_up1, conv_up2)");
    m.nf = f->shape[0];
    m.gc = g->shape[0];
    m.D = m.nf + 4 * m.gc;
    if (m.nf % 8 != 0 || m.gc % 8 != 0) throw std::runtime_error(path + ": features not multiples of 8");
    while (st.find("body." + std::to_string(m.nb) + ".rdb1.conv1.weight") != nullptr) ++m.nb;
    m.first = read_layer(st, "conv_first", 3, m.nf);
    for (int64_t i = 0; i < m.nb; ++i)
        for (int j = 1; j <= 3; ++j)
            for (int k = 1; k <= 5; ++k) {
                const std::string n = "body." + std::to_string(i) + ".rdb" + std::to_string(j) + ".conv" + std::to_string(k);
                // the fifth's output enters the block's residual times 0.2: its weights and bias scaled by it
                m.rdb.push_back(k < 5 ? read_layer(st, n, m.nf + (k - 1) * m.gc, m.gc)
                                      : read_layer(st, n, m.D, m.nf, kResidual));
            }
    m.body = read_layer(st, "conv_body", m.nf, m.nf);
    m.up1 = read_layer(st, "conv_up1", m.nf, m.nf);
    m.up2 = read_layer(st, "conv_up2", m.nf, m.nf);
    m.hr = read_layer(st, "conv_hr", m.nf, m.nf);
    m.last = read_layer(st, "conv_last", m.nf, 3);
    for (Layer* l : m.layers()) m.weights_bytes += l->w.size() * 2 + l->b.size() * 4;
    m.s = core::Runtime::get().create_stream();
    info_.name = "realesrgan";
    info_.scale = 4;
    info_.context = kContext;
    info_.min_tile = 64;
    info_.color = ColorFix::none;   // the model's output as it is
}

Esrgan::~Esrgan() {
    if (!impl_) return;
    try {
        impl_->q().wait();
    } catch (const std::exception& e) {   // a lost device: the buffers go with it
        std::fprintf(stderr, "xestrata-image: RealESRGAN's queue at its end: %s\n", e.what());
    }
    impl_->drop_scratch();
    impl_->drop_weights();
    core::Runtime::get().destroy_stream(static_cast<sycl::queue*>(impl_->s));
}

void Esrgan::release(bool all) {
    impl_->q().wait();
    impl_->drop_scratch();
    if (all) impl_->drop_weights();
}

uint64_t Esrgan::weight_bytes() const { return impl_->weights_bytes; }

uint64_t Esrgan::held() const {
    return (impl_->weights.empty() ? 0 : impl_->weights_bytes) + impl_->sc.bytes +
           (uint64_t) impl_->col_cap * 2 + (uint64_t) impl_->tmp_cap * 4;
}

// a frame of n = w x h pixels: the input in FP16 (8 channels); the trunk, the RRDB's and the block's inputs in FP32 and
// the dense block's channels in FP16 (nf + 4 gc); at 2x conv_up1's output, at 4x two FP16 buffers of nf channels (the
// 2x upsampled trunk and conv_last's FP32 output in the second); the own path's scratch (the unfolded band and its outputs)
// where the build has no library path for the GPU
uint64_t Esrgan::clip_bytes(int64_t w, int64_t h, int64_t frames) const {
    (void) frames;   // one frame at a time
    const Impl& m = *impl_;
    const uint64_t n = (uint64_t) (w * h), nf = (uint64_t) m.nf, D = (uint64_t) m.D;
    const uint64_t low = n * (16 + 12 * nf + 2 * D), x2 = 8 * n * nf, x4 = 64 * n * nf;
    const uint64_t col = (uint64_t) conv::col_values(core::Runtime::get().stream(nullptr)) * 2;
    const uint64_t own = col + (col / (9ll * 8 * 2)) * nf * 4;
    return low + x2 + x4 + own;
}

uint64_t Esrgan::clip_largest(int64_t w, int64_t h, int64_t frames) const {
    (void) frames;
    return (uint64_t) (16 * w * h) * (uint64_t) impl_->nf * 2;
}

void Esrgan::run_clip(const uint8_t* in, int64_t w, int64_t h, int64_t frames, uint8_t* out, uint64_t seed) {
    (void) seed;   // nothing random in it
    Impl& m = *impl_;
    m.put_weights();
    m.shape(w, h);
    for (int64_t f = 0; f < frames; ++f) m.run(in + f * w * h * 4, w, h, out + f * 16 * w * h * 4);
    m.q().wait_and_throw();
}

void Esrgan::Impl::run(const uint8_t* rgba, int64_t w, int64_t h, uint8_t* out) {
    using image::ImageAct;
    const int64_t n = w * h, NF = nf, DD = D;
    sycl::queue& qq = q();
    uint16_t* in16 = sc.in16;
    float* trunk = sc.trunk;
    float* xr = sc.xr;   // the RRDB's input
    float* xd = sc.xd;   // the dense block's input, and its output in place (conv5 adds to it)
    uint16_t* dense = sc.dense;
    uint16_t* a2 = sc.a2;   // conv_up1's output
    uint16_t* a4 = sc.a4;
    uint16_t* b4 = sc.b4;   // also the 2x upsampled trunk, and conv_last's output (FP32)
    // the input in [0, 1], three channels of eight
    qq.parallel_for(sycl::range<1>((size_t) (n * 8)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / 8, c = i - p * 8;
        in16[i] = f16_from_f32(c < 3 ? (float) rgba[p * 4 + c] * (1.0f / 255.0f) : 0.0f);
    });
    conv(first, in16, h, w, 8, trunk, NF, false, false, false);
    // xr = xd = trunk, dense's first nf channels its FP16
    const size_t fbytes = (size_t) (n * NF) * sizeof(float);
    qq.memcpy(xr, trunk, fbytes);
    qq.memcpy(xd, trunk, fbytes);
    image::qi_conv_store16(trunk, n, NF, NF, nullptr, ImageAct::none, 0.f, nullptr, dense, DD, 0, s);
    for (int64_t r = 0; r < nb; ++r)
        for (int j = 0; j < 3; ++j) {
            const Layer* L = &rdb[(size_t) ((r * 3 + j) * 5)];
            for (int k = 0; k < 4; ++k)   // x_k = lrelu(conv_k(x, x_1 .. x_{k-1})), into the dense channels
                conv(L[k], dense, h, w, DD, dense + NF + k * gc, DD, true, true, false);
            conv(L[4], dense, h, w, DD, xd, NF, false, false, true);   // xd += 0.2 conv5(...)
            if (j == 2) {   // the RRDB's residual: 0.2 of its blocks' output added to its input
                kernels::add_inplace(xr, xd, n * NF, s, kResidual);
                qq.memcpy(xd, xr, fbytes);
            }
            image::qi_conv_store16(xd, n, NF, NF, nullptr, ImageAct::none, 0.f, nullptr, dense, DD, 0, s);
        }
    conv(body, dense, h, w, DD, trunk, NF, false, false, true);   // trunk += conv_body(body(trunk))
    // the trunk at 2x and conv_up1's output at 4x (nearest) in FP16
    const int64_t w2 = 2 * w, w4 = 4 * w, h2 = 2 * h;
    image::qi_up2x16(trunk, h, w, NF, b4, s);
    conv(up1, b4, h2, w2, NF, a2, NF, true, true, false);
    image::qi_up2x16(a2, h2, w2, NF, a4, s);
    conv(up2, a4, 2 * h2, w4, NF, b4, NF, true, true, false);
    conv(hr, b4, 2 * h2, w4, NF, a4, NF, true, true, false);
    float* o32 = reinterpret_cast<float*>(b4);   // [16n][8]: 16n * 32 bytes, within b4's 16n * 2 * nf
    conv(last, a4, 2 * h2, w4, NF, o32, 8, false, false, false);
    const int64_t LO = last.O;
    qq.parallel_for(sycl::range<1>((size_t) (16 * n)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t X = i % w4, Y = i / w4;
        for (int c = 0; c < 3; ++c)
            out[i * 4 + c] = (uint8_t) sycl::round(sycl::clamp(o32[i * LO + c], 0.0f, 1.0f) * 255.0f);
        out[i * 4 + 3] = rgba[((Y / 4) * w + X / 4) * 4 + 3];
    });
}

}  // namespace strata::upscale
