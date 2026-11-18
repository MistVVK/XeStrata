// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/aux_net.cpp - see include/strata/image/aux_net.hpp.

#include "strata/image/aux_net.hpp"

#include "qi_conv.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/timed_choice.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

std::vector<uint16_t> f16_of(const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = kernels::f16_from_f32(v[i]);
    return h;
}


}  // namespace

AuxNet::AuxNet(const std::string& path, const char* what)
    : path_(path), what_(what), st_(std::make_unique<SafeTensors>(path)) {
    for (const auto& t : st_->tensors()) file_bytes_ += t.second.end - t.second.begin;
}

// The base's own scratch and the weights: a derived part's drop_scratch is gone by now (its members with it).  The
// stream goes last: the auxiliary models are let go before an image job and made again for the next tool's run, and
// a queue kept of each would pile up (Runtime::finish drains them all); its belongings (the RoPE tables, the
// attention's scratch) go with it.
AuxNet::~AuxNet() {
    AuxNet::drop_scratch();
    free_weights();
    if (s_ != nullptr) {
        try {
            core::Runtime::get().destroy_stream(static_cast<sycl::queue*>(s_));
        } catch (const std::exception& e) {   // a lost device: the queue goes with it
            std::fprintf(stderr, "image engine (%s): its queue at its end: %s\n", what_.c_str(), e.what());
        }
        s_ = nullptr;
    }
}

void* AuxNet::alloc(size_t bytes) {
    void* p = nullptr;
    if (!gpu::alloc_device(&p, std::max<size_t>(bytes, 4)))
        throw std::runtime_error("image engine (" + what_ + "): GPU memory: " + gpu::last_error());
    weights_.push_back(p);
    return p;
}

void AuxNet::drop_scratch() {
    if (s_ != nullptr) gpu::stream_sync(s_);
    steps_.clear();
    for (auto& w : work_) gpu::free(w.first);
    work_.clear();
    for (auto& st : stages_)
        if (st.second.done != nullptr) gpu::event_destroy(st.second.done);
    stages_.clear();
}

void AuxNet::upload_async(void* dst, const void* src, size_t bytes, int stage) {
    Stage& st = stages_[stage];
    if (st.done != nullptr) gpu::event_sync(st.done);
    else if (!gpu::event_create(&st.done)) throw std::runtime_error("image engine (" + what_ + "): an event");
    st.buf.resize(bytes);
    std::memcpy(st.buf.data(), src, bytes);
    gpu::copy_async(dst, st.buf.data(), bytes, s_);
    gpu::event_record(st.done, s_);
}

void AuxNet::release() {
    drop_scratch();
    free_weights();
}

void AuxNet::free_weights() {
    for (void* p : weights_) gpu::free(p);
    weights_.clear();
    weight_bytes_ = 0;
    loaded_ = false;
}

uint64_t AuxNet::held() const {
    uint64_t n = weight_bytes_;
    for (const auto& w : work_) n += w.second;
    return n;
}

void AuxNet::load() {
    if (loaded_) return;
    if (s_ == nullptr) s_ = core::Runtime::get().create_stream();
    std::string err;
    if (!gemm_.init(s_, 16, err)) throw std::runtime_error("image engine (" + what_ + "): " + err);
    try {
        load_weights();
        loaded_ = true;
    } catch (...) {
        release();
        throw;
    }
    if (progress) progress(0.0);   // read: the run starts (a caller showing the loading moves on)
}

float* AuxNet::scratch(int slot, size_t bytes) {
    if ((size_t) slot >= work_.size()) work_.resize((size_t) slot + 1, {nullptr, 0});
    auto& w = work_[(size_t) slot];
    if (w.second < bytes) {
        if (w.first != nullptr) {
            gpu::stream_sync(s_);
            gpu::free(w.first);
            w = {nullptr, 0};
        }
        if (!gpu::alloc_device(&w.first, std::max<size_t>(bytes, 4)))
            throw std::runtime_error("image engine (" + what_ + "): GPU memory: " + gpu::last_error());
        w.second = bytes;
    }
    return static_cast<float*>(w.first);
}

std::vector<float> AuxNet::host_f32(const std::string& name) {
    if (st_->find(name) == nullptr) throw std::runtime_error(path_ + ": no tensor " + name);
    return st_->f32(name);
}

const float* AuxNet::f32(const std::string& name) {
    const auto v = host_f32(name);
    auto* d = static_cast<float*>(alloc(v.size() * 4));
    gpu::copy(d, v.data(), v.size() * 4);
    weight_bytes_ += v.size() * 4;
    return d;
}

const float* AuxNet::upload(const std::vector<float>& v) {
    auto* d = static_cast<float*>(alloc(std::max<size_t>(v.size(), 1) * 4));
    gpu::copy(d, v.data(), v.size() * 4);
    weight_bytes_ += v.size() * 4;
    return d;
}

AuxNet::Lin AuxNet::lin_host(const std::vector<float>& w, const std::vector<float>& b, int64_t n, int64_t k) {
    if ((int64_t) w.size() != n * k || (!b.empty() && (int64_t) b.size() != n))
        throw std::runtime_error(path_ + ": a linear layer's size");
    Lin l;
    l.n = n;
    l.k = k;
    const auto h = f16_of(w);
    auto* dw = static_cast<uint16_t*>(alloc(h.size() * 2));
    gpu::copy(dw, h.data(), h.size() * 2);
    l.w = dw;
    weight_bytes_ += h.size() * 2;
    if (!b.empty()) l.b = upload(b);
    return l;
}

const uint16_t* AuxNet::f16(const std::string& name) {
    const auto h = f16_of(host_f32(name));
    auto* d = static_cast<uint16_t*>(alloc(h.size() * 2));
    gpu::copy(d, h.data(), h.size() * 2);
    weight_bytes_ += h.size() * 2;
    return d;
}

AuxNet::Lin AuxNet::lin(const std::vector<std::string>& names, int64_t r0, int64_t rows) {
    Lin l;
    std::vector<float> w, b;
    bool has_b = true;
    for (const std::string& n : names) {
        const SafeTensor* t = st_->find(n + ".weight");
        if (t == nullptr || t->shape.size() != 2) throw std::runtime_error(path_ + ": no linear layer " + n);
        if (l.k != 0 && t->shape[1] != l.k) throw std::runtime_error(path_ + ": " + n + "'s width");
        l.k = t->shape[1];
        const auto v = st_->f32(n + ".weight");
        w.insert(w.end(), v.begin(), v.end());
        if (st_->find(n + ".bias") != nullptr) {
            const auto vb = st_->f32(n + ".bias");
            b.insert(b.end(), vb.begin(), vb.end());
        } else {
            has_b = false;
        }
    }
    const int64_t all = (int64_t) w.size() / std::max<int64_t>(l.k, 1);
    if (rows < 0) rows = all - r0;
    if (r0 < 0 || r0 + rows > all) throw std::runtime_error(path_ + ": rows outside " + names[0]);
    l.n = rows;
    const std::vector<float> ws(w.begin() + r0 * l.k, w.begin() + (r0 + rows) * l.k);
    const auto h = f16_of(ws);
    auto* dw = static_cast<uint16_t*>(alloc(h.size() * 2));
    gpu::copy(dw, h.data(), h.size() * 2);
    l.w = dw;
    weight_bytes_ += h.size() * 2;
    if (has_b && (int64_t) b.size() == all) {
        auto* db = static_cast<float*>(alloc((size_t) rows * 4));
        gpu::copy(db, b.data() + r0, (size_t) rows * 4);
        l.b = db;
        weight_bytes_ += (size_t) rows * 4;
    }
    return l;
}

AuxNet::Lin AuxNet::lin_wb(const std::string& w_name, const std::string& b_name) {
    const SafeTensor* t = st_->find(w_name);
    if (t == nullptr || t->shape.size() != 2) throw std::runtime_error(path_ + ": no linear layer " + w_name);
    Lin l;
    l.n = t->shape[0];
    l.k = t->shape[1];
    l.w = f16(w_name);
    if (st_->find(b_name) != nullptr) l.b = f32(b_name);
    return l;
}

void AuxNet::lin3(const std::string& w_name, const std::string& b_name, Lin& q, Lin& k, Lin& v) {
    const SafeTensor* t = st_->find(w_name);
    if (t == nullptr || t->shape.size() != 2 || t->shape[0] % 3 != 0)
        throw std::runtime_error(path_ + ": no packed q, k, v " + w_name);
    const int64_t n = t->shape[0] / 3, kk = t->shape[1];
    const auto w = st_->f32(w_name);
    const auto b = st_->find(b_name) != nullptr ? st_->f32(b_name) : std::vector<float>();
    Lin* out[3] = {&q, &k, &v};
    for (int i = 0; i < 3; ++i) {
        Lin& l = *out[i];
        l.n = n;
        l.k = kk;
        const auto h = f16_of(std::vector<float>(w.begin() + i * n * kk, w.begin() + (i + 1) * n * kk));
        auto* dw = static_cast<uint16_t*>(alloc(h.size() * 2));
        gpu::copy(dw, h.data(), h.size() * 2);
        l.w = dw;
        weight_bytes_ += h.size() * 2;
        if (!b.empty()) {
            auto* db = static_cast<float*>(alloc((size_t) n * 4));
            gpu::copy(db, b.data() + i * n, (size_t) n * 4);
            l.b = db;
            weight_bytes_ += (size_t) n * 4;
        }
    }
}

AuxNet::Norm AuxNet::norm(const std::string& name, const char* w, const char* b) {
    Norm n;
    n.w = f32(name + w);
    n.b = f32(name + b);
    return n;
}

AuxNet::Conv AuxNet::conv(const std::string& name, int stride, int pad) {
    const SafeTensor* t = st_->find(name + ".weight");
    if (t == nullptr || t->shape.size() != 4 || t->shape[2] != t->shape[3])
        throw std::runtime_error(path_ + ": no convolution " + name);
    const auto b = st_->find(name + ".bias") != nullptr ? st_->f32(name + ".bias") : std::vector<float>();
    return conv_host(st_->f32(name + ".weight"), b, t->shape[0], t->shape[1], (int) t->shape[2], stride, pad);
}

AuxNet::Conv AuxNet::conv_host(const std::vector<float>& w, const std::vector<float>& b, int64_t o, int64_t c, int k,
                               int stride, int pad) {
    Conv cv;
    cv.o = o;
    cv.c = c;
    cv.k = k;
    cv.stride = stride;
    cv.pad = pad;
    const int64_t kk = (int64_t) k * k;
    if ((int64_t) w.size() != o * c * kk) throw std::runtime_error(path_ + ": a convolution's weight size");
    std::vector<uint16_t> h(w.size());   // [o][c][k][k] -> [o][k][k][c]
    for (int64_t oi = 0; oi < o; ++oi)
        for (int64_t ci = 0; ci < c; ++ci)
            for (int64_t t2 = 0; t2 < kk; ++t2)
                h[(size_t) ((oi * kk + t2) * c + ci)] = kernels::f16_from_f32(w[(size_t) ((oi * c + ci) * kk + t2)]);
    auto* dw = static_cast<uint16_t*>(alloc(h.size() * 2));
    gpu::copy(dw, h.data(), h.size() * 2);
    cv.w = dw;
    weight_bytes_ += h.size() * 2;
    if (!b.empty()) {
        auto* db = static_cast<float*>(alloc(b.size() * 4));
        gpu::copy(db, b.data(), b.size() * 4);
        cv.b = db;
        weight_bytes_ += b.size() * 4;
    }
    return cv;
}

AuxNet::Lin AuxNet::deconv2(const std::string& name) {
    const SafeTensor* t = st_->find(name + ".weight");
    if (t == nullptr || t->shape.size() != 4 || t->shape[2] != 2 || t->shape[3] != 2)
        throw std::runtime_error(path_ + ": no 2 x 2 transposed convolution " + name);
    const int64_t ci = t->shape[0], co = t->shape[1];
    const auto w = st_->f32(name + ".weight");   // [c][o][dy][dx] -> [(dy, dx, o)][c]
    std::vector<uint16_t> h(w.size());
    for (int64_t c = 0; c < ci; ++c)
        for (int64_t o = 0; o < co; ++o)
            for (int64_t t2 = 0; t2 < 4; ++t2)
                h[(size_t) ((t2 * co + o) * ci + c)] = kernels::f16_from_f32(w[(size_t) ((c * co + o) * 4 + t2)]);
    Lin l;
    l.n = 4 * co;
    l.k = ci;
    auto* dw = static_cast<uint16_t*>(alloc(h.size() * 2));
    gpu::copy(dw, h.data(), h.size() * 2);
    l.w = dw;
    weight_bytes_ += h.size() * 2;
    if (st_->find(name + ".bias") != nullptr) l.b = f32(name + ".bias");   // [o]: added after the scatter
    return l;
}

void AuxNet::product16(const uint16_t* x16, const Lin& l, uint16_t* y16, int64_t T, bool bias, ImageAct act) {
    const float* b = bias ? l.b : nullptr;
    const auto narrow = [&] {
        float* y = scratch(kNarrowSlot, (size_t) (T * l.n * 4));
        gemm_.f16(x16, l.w, y, T, l.n, l.k);
        aux_narrow_bias16(y, b, T, l.n, y16, s_);
        if (act != ImageAct::none) aux_bias_act16(y16, nullptr, T, l.n, act, y16, s_);
    };
    const auto half = [&] {
        gemm_.native_half(x16, 1, l.w, y16, T, l.n, l.k);   // type 1: F16 rows
        if (b != nullptr || act != ImageAct::none) aux_bias_act16(y16, b, T, l.n, act, y16, s_);
    };
    // a library's 1 x 1 convolution: the bias and the activation in its own pass over the sums (oneDNN's post-ops;
    // cuDNN's with ReLU only); false where it does not take the form
    const auto library = [&] {
        if (act != ImageAct::none && act != ImageAct::relu && act != ImageAct::gelu) return false;   // the forms it takes
#if STRATA_ONEDNN || STRATA_CUDNN
        conv::Form f;
        f.y16 = true;
        if (act == ImageAct::relu) f.leaky = 0.f;
        f.gelu = act == ImageAct::gelu;
        sycl::queue& q = core::Runtime::get().stream(s_);
#endif
#if STRATA_ONEDNN
        if (conv::onednn(q, x16, 1, T, l.k, l.w, l.n, 1, 1, 0, 0, 0, 0, b, false, y16, f)) return true;
#endif
#if STRATA_CUDNN
        if (conv::cudnn(q, x16, 1, T, l.k, l.w, l.n, 1, b, false, y16, f)) return true;
#endif
        return false;
    };
    const auto key = std::make_tuple(T, l.n, l.k, (int) act * 2 + (b != nullptr ? 1 : 0));
    auto it = pick16_.find(key);
    if (it == pick16_.end()) {
        const auto choice = core::timed_choice({{[&] { narrow(); return true; }},
                                                {[&] { if (!prefill::Gemm::half_ok(s_)) return false; half(); return true; }},
                                                {library}}, [&] { gpu::stream_sync(s_); });
        const int pick = choice.pick;
        it = pick16_.emplace(key, pick).first;
    }
    if (it->second == 2 && library()) return;
    if (it->second == 1) half();
    else narrow();
}

void AuxNet::product(const uint16_t* x16, const Lin& l, float* y, int64_t T, int64_t ldy, bool bias) {
    gemm_.f16(x16, l.w, y, T, l.n, l.k, ldy);
    if (bias && l.b != nullptr) {
        if (ldy != 0 && ldy != l.n) throw std::logic_error(what_ + ": a biased product with a row stride");
        qi_bias_add(y, l.b, T, l.n, s_);
    }
}

void AuxNet::convolve(const Conv& c, const uint16_t* x16, int64_t H, int64_t W, float* y, int col_slot, bool add) {
    const int64_t OH = (H + 2 * (int64_t) c.pad - c.k) / c.stride + 1, OW = (W + 2 * (int64_t) c.pad - c.k) / c.stride + 1;
    const int64_t P = OH * OW, K = (int64_t) c.k * c.k * c.c;
    sycl::queue& q = core::Runtime::get().stream(s_);
    const int64_t cap = conv::col_values(q);
    conv::Request r{x16, H, W, c.c, c.w, c.o, c.k, c.stride, c.pad, c.pad, c.pad, c.pad};
    r.bias = c.b;
    r.add = add;
    r.out = y;
    uint16_t* col = nullptr;
    conv::drive(q, r, P, K, cap, c.k == 1 && c.stride == 1 && c.pad == 0,
        [&](int64_t p0, int64_t n) {
            if (col == nullptr) col = scratch_as<uint16_t>(col_slot, (size_t) (std::min(P, cap / K) * K));
            qi_im2col16(x16, 1, H, W, c.c, 1, c.k, c.stride, c.pad, OH, OW, p0, n, col, s_);
        },
        [&](int64_t p0, int64_t n, bool point) {
            gemm_.f16(point ? x16 : col, c.w, y + p0 * c.o, n, c.o, point ? c.c : K, 0, false, add);
        },
        [&] { if (c.b != nullptr) qi_bias_add(y, c.b, P, c.o, s_); });
}

void AuxNet::sync() { gpu::stream_sync(s_); }

void AuxNet::host(const char* name, const float* dev, size_t n) {
    if (!dump) return;
    std::vector<float> h(n);
    gpu::stream_sync(s_);
    gpu::copy(h.data(), dev, n * 4);
    dump(name, h);
}

AuxMapNet::MConv AuxMapNet::mconv(const std::string& w_name, const std::string& b_name, int stride, int pad,
                                  const std::string& bn, float eps, int64_t in_pad, bool depthwise) {
    const SafeTensor* t = st_->find(w_name);
    if (t == nullptr || (t->shape.size() != 4 && t->shape.size() != 2))
        throw std::runtime_error(path_ + ": no convolution " + w_name);
    const int64_t out = t->shape[0], in = t->shape[1];
    const int k = t->shape.size() == 4 ? (int) t->shape[2] : 1;
    std::vector<float> b = b_name.empty() ? std::vector<float>() : host_f32(b_name);
    std::vector<float> w = host_f32(w_name);
    if (!bn.empty()) {   // the batch norm folded in: w * g / sqrt(v + eps), (b - m) * g / sqrt(v + eps) + beta
        const auto g = host_f32(bn + ".weight"), be = host_f32(bn + ".bias");
        const auto m = host_f32(bn + ".running_mean"), v = host_f32(bn + ".running_var");
        if ((int64_t) g.size() != out || be.size() != g.size() || m.size() != g.size() || v.size() != g.size() ||
            (!b.empty() && b.size() != g.size())) throw std::runtime_error(path_ + ": batch norm of another size");
        if (b.empty()) b.assign((size_t) out, 0.f);
        const size_t per = w.size() / (size_t) out;
        for (int64_t o = 0; o < out; ++o) {
            const float sc = g[(size_t) o] / std::sqrt(v[(size_t) o] + eps);
            for (size_t j = 0; j < per; ++j) w[(size_t) o * per + j] *= sc;
            b[(size_t) o] = (b[(size_t) o] - m[(size_t) o]) * sc + be[(size_t) o];
        }
    }
    if (depthwise && in != 1) throw std::runtime_error(path_ + ": " + w_name + " is not a depthwise convolution");
    return mconv_host(w, std::move(b), out, depthwise ? out : in, k, stride, pad, in_pad, depthwise);
}

AuxMapNet::MConv AuxMapNet::mconv_host(const std::vector<float>& w, std::vector<float> b, int64_t out, int64_t in,
                                       int k, int stride, int pad, int64_t in_pad, bool depthwise) {
    MConv c;
    c.out = out;
    c.in = in;
    c.k = k;
    c.s = stride;
    c.pad = pad;
    c.depthwise = depthwise;
    c.outp = up8(out);
    c.inp = depthwise ? c.outp : std::max(in_pad, up8(in));
    const int64_t kk = (int64_t) k * k;
    if ((int64_t) w.size() != out * (depthwise ? 1 : in) * kk)
        throw std::runtime_error(path_ + ": a convolution's weight size");
    std::vector<uint16_t> h;
    if (depthwise) {
        h.assign((size_t) (c.outp * kk), 0);
        for (size_t j = 0; j < w.size(); ++j) h[j] = kernels::f16_from_f32(w[j]);
    } else {   // [out][in][k][k] -> [outp][k][k][inp]
        h.assign((size_t) (c.outp * kk * c.inp), 0);
        for (int64_t o = 0; o < out; ++o)
            for (int64_t i = 0; i < in; ++i)
                for (int64_t t2 = 0; t2 < kk; ++t2)
                    h[(size_t) ((o * kk + t2) * c.inp + i)] = kernels::f16_from_f32(w[(size_t) ((o * in + i) * kk + t2)]);
    }
    auto* dw = static_cast<uint16_t*>(alloc(h.size() * 2));
    gpu::copy(dw, h.data(), h.size() * 2);
    weight_bytes_ += h.size() * 2;
    c.w = dw;
    b.resize((size_t) c.outp, 0.f);
    c.b = upload(b);
    return c;
}

AuxMapNet::MConv AuxMapNet::mdeconv(const std::string& w_name, const std::string& b_name, bool general) {
    const SafeTensor* t = st_->find(w_name);
    if (t == nullptr || t->shape.size() != 4 || t->shape[2] != t->shape[3])
        throw std::runtime_error(path_ + ": no transposed convolution " + w_name);
    MConv c;
    c.in = t->shape[0];
    c.out = t->shape[1];
    c.k = (int) t->shape[2];
    c.s = general ? 1 : c.k;
    c.inp = up8(c.in);
    c.outp = up8(c.out);
    const int64_t k = c.k;
    const std::vector<float> w = host_f32(w_name);
    std::vector<uint16_t> h;
    if (general) {   // the flipped kernel of a plain convolution [outp][k][k][inp]
        h.assign((size_t) (c.outp * k * k * c.inp), 0);
        for (int64_t i = 0; i < c.in; ++i)
            for (int64_t o = 0; o < c.out; ++o)
                for (int64_t a = 0; a < k; ++a)
                    for (int64_t bb = 0; bb < k; ++bb)
                        h[(size_t) (((o * k + (k - 1 - a)) * k + (k - 1 - bb)) * c.inp + i)] =
                            kernels::f16_from_f32(w[(size_t) (((i * c.out + o) * k + a) * k + bb)]);
    } else {         // a product to [k][k][outp] a pixel: [(a, b, o)][inp]
        h.assign((size_t) (k * k * c.outp * c.inp), 0);
        for (int64_t i = 0; i < c.in; ++i)
            for (int64_t o = 0; o < c.out; ++o)
                for (int64_t a = 0; a < k; ++a)
                    for (int64_t bb = 0; bb < k; ++bb)
                        h[(size_t) (((a * k + bb) * c.outp + o) * c.inp + i)] =
                            kernels::f16_from_f32(w[(size_t) (((i * c.out + o) * k + a) * k + bb)]);
    }
    auto* dw = static_cast<uint16_t*>(alloc(h.size() * 2));
    gpu::copy(dw, h.data(), h.size() * 2);
    weight_bytes_ += h.size() * 2;
    c.w = dw;
    std::vector<float> b = b_name.empty() ? std::vector<float>((size_t) c.out, 0.f) : host_f32(b_name);
    b.resize((size_t) c.outp, 0.f);
    c.b = upload(b);
    return c;
}

AuxMapNet::Map AuxMapNet::run(const MConv& c, const Map& x, ImageAct a, const Map* res) {
    if (x.c != c.inp)
        throw std::runtime_error(what_ + ": a convolution of " + std::to_string(c.inp) + " inputs on " +
                                 std::to_string(x.c) + " channels");
    const int64_t span = (int64_t) (c.k - 1) * c.dil + 1;
    const int64_t pad = c.pad, st = c.s;
    const int64_t OH = (x.h + 2 * pad - span) / st + 1, OW = (x.w + 2 * pad - span) / st + 1, P = OH * OW;
    Map out = map(OH, OW, c.outp);
    if (c.depthwise) {
        aux_dwconv16(x.p, x.h, x.w, x.c, c.w, c.b, c.k, c.s, c.pad, OH, OW, a, out.p, s_);
        if (res != nullptr) aux_add16h(out.p, res->p, P * c.outp, out.p, s_);
        return out;
    }
    const bool plain = c.k == 1 && c.s == 1 && c.pad == 0;
    const bool direct = res == nullptr && (a == ImageAct::none || a == ImageAct::relu);
    sycl::queue& q = core::Runtime::get().stream(s_);
    const int64_t K = (int64_t) c.k * c.k * c.inp, cap = conv::col_values(q);
    float* y = direct ? nullptr : scratch(kSumSlot, (size_t) (P * c.outp) * 4);
    conv::Request r{x.p, x.h, x.w, c.inp, c.w, c.outp, c.k, c.s, c.pad, c.pad, c.pad, c.pad};
    r.dilation = c.dil;
    r.bias = c.b;
    r.out = direct ? static_cast<void*>(out.p) : static_cast<void*>(y);
    if (direct) {
        r.form.y16 = true;
        if (a == ImageAct::relu) r.form.leaky = 0.f;
    }
    uint16_t* col = nullptr;
    // Few outputs: direct beats unfolded traffic.  Line art's 7 x 7 of 64 channels to one at 768 x 512 took the
    // RTX 4070 24 ms unfolded (2026-11-08).
    const bool few = !plain && c.outp <= kConvFewOut && c.dil == 1;
    const auto chosen = conv::drive(q, r, P, K, cap, plain || few,
        [&](int64_t p0, int64_t n) {
            if (col == nullptr) col = scratch_as<uint16_t>(kColSlot, (size_t) (std::min(P, cap / K) * K));
            qi_im2col16(x.p, 1, x.h, x.w, c.inp, 1, c.k, c.s, c.pad, OH, OW, p0, n, col, s_, 0, c.dil);
        },
        [&](int64_t p0, int64_t n, bool point) {
            if (y == nullptr) y = scratch(kSumSlot, (size_t) (P * c.outp) * 4);
            if (few) aux_conv_few16(x.p, x.h, x.w, c.inp, c.w, c.outp, c.out, c.k, c.s, c.pad, OH, OW, y, s_);
            else gemm_.f16(point ? x.p : col, c.w, y + p0 * c.outp, n, c.outp, point ? c.inp : K);
        },
        [&] { aux_map_store16(y, P, c.outp, c.b, a, res != nullptr ? res->p : nullptr, out.p, c.outp, 0, s_); });
    if (chosen != conv::Library::none && !direct)
        aux_map_store16(y, P, c.outp, nullptr, a, res != nullptr ? res->p : nullptr, out.p, c.outp, 0, s_);
    return out;
}

AuxMapNet::Map AuxMapNet::run_deconv(const MConv& c, const Map& x, ImageAct a) {
    if (x.c != c.inp) throw std::runtime_error(what_ + ": a transposed convolution on another width");
    float* y = scratch(kSumSlot, (size_t) (x.h * x.w * c.k * c.k * c.outp) * 4);
    gemm_.f16(x.p, c.w, y, x.h * x.w, (int64_t) c.k * c.k * c.outp, c.inp);
    Map out = map(x.h * c.k, x.w * c.k, c.outp);
    aux_deconv_scatter16(y, x.h, x.w, c.k, c.outp, c.b, a, out.p, s_);
    return out;
}

AuxMapNet::Map AuxMapNet::run_deconv_general(const MConv& c, const Map& x, int64_t s, int64_t p, int64_t op,
                                             ImageAct a) {
    const int64_t lo = c.k - 1 - p, hi = c.k - 1 - p + op;
    Map z = map((x.h - 1) * s + 1 + lo + hi, (x.w - 1) * s + 1 + lo + hi, x.c);
    aux_zero_insert16(x.p, x.h, x.w, x.c, s, lo, hi, z.p, s_);
    MConv plain = c;
    plain.s = 1;
    plain.pad = 0;
    return run(plain, z, a);
}

AuxMapNet::Map AuxMapNet::act(const Map& x, ImageAct a) {
    Map out = map(x.h, x.w, x.c);
    aux_bias_act16(x.p, nullptr, x.h * x.w, x.c, a, out.p, s_);
    return out;
}

AuxMapNet::Map AuxMapNet::add(const Map& a, const Map& b) {
    if (a.h != b.h || a.w != b.w || a.c != b.c) throw std::runtime_error(what_ + ": adding maps of two shapes");
    Map out = map(a.h, a.w, a.c);
    aux_add16h(a.p, b.p, a.h * a.w * a.c, out.p, s_);
    return out;
}

AuxMapNet::Map AuxMapNet::reflect(const Map& x, int64_t p) {
    Map out = map(x.h + 2 * p, x.w + 2 * p, x.c);
    aux_pad_reflect16(x.p, x.h, x.w, x.c, p, out.p, s_);
    return out;
}

AuxMapNet::Map AuxMapNet::resize(const Map& x, int64_t oh, int64_t ow, bool align_corners) {
    Map out = map(oh, ow, x.c);
    aux_bilinear16(x.p, x.h, x.w, x.c, oh, ow, align_corners, false, out.p, s_);
    return out;
}

AuxMapNet::Map AuxMapNet::maxpool(const Map& x, int k, int s, int p) {
    const int64_t pp = p, kk = k, ss = s;
    Map out = map((x.h + 2 * pp - kk) / ss + 1, (x.w + 2 * pp - kk) / ss + 1, x.c);
    aux_maxpool16(x.p, x.h, x.w, x.c, k, s, p, out.h, out.w, out.p, s_);
    return out;
}

AuxMapNet::Map AuxMapNet::pad(const Map& x, int64_t top, int64_t left, int64_t bottom, int64_t right) {
    Map out = map(x.h + top + bottom, x.w + left + right, x.c);
    aux_pad16(x.p, x.h, x.w, x.c, top, left, bottom, right, out.p, s_);
    return out;
}

AuxMapNet::Map AuxMapNet::concat(const std::vector<Map>& parts) {
    int64_t C = 0;
    for (const Map& m : parts) {
        if (m.h != parts[0].h || m.w != parts[0].w) throw std::runtime_error(what_ + ": concatenating maps of two sizes");
        C += m.c;
    }
    Map out = map(parts[0].h, parts[0].w, C);
    if (parts.size() == 2) {
        aux_concat2_16(parts[0].p, parts[0].c, false, parts[1].p, parts[1].c, out.h, out.w, out.p, s_);
        return out;
    }
    int64_t off = 0;
    for (const Map& m : parts) {
        aux_copy_channels16(m.p, m.h * m.w, m.c, out.p, C, off, s_);
        off += m.c;
    }
    return out;
}

void AuxMapNet::instance_norm(Map& x, float eps, ImageAct a) {
    image_group_norm(nullptr, x.p, x.h * x.w, x.c, x.c, nullptr, nullptr, eps, a, nullptr, x.p,
                     buf<float>((size_t) qi_group_norm_ws_floats(x.h * x.w, x.c)), s_);
}

std::vector<float> AuxMapNet::host_channel0(const Map& x) {
    std::vector<uint16_t> h((size_t) (x.h * x.w * x.c));
    gpu::stream_sync(s_);
    gpu::copy(h.data(), x.p, h.size() * 2);
    std::vector<float> out((size_t) (x.h * x.w));
    for (size_t i = 0; i < out.size(); ++i) out[i] = kernels::f32_from_f16(h[i * (size_t) x.c]);
    return out;
}

}  // namespace strata::image
