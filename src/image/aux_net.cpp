// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/aux_net.cpp - see include/strata/image/aux_net.hpp.

#include "strata/image/aux_net.hpp"

#if STRATA_ONEDNN || STRATA_CUDNN
#include "qi_conv.hpp"
#endif
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

std::vector<uint16_t> f16_of(const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = kernels::f16_from_f32(v[i]);
    return h;
}

constexpr int64_t kColBytes = 64ll << 20;   // the unfolded patches' slice, where no library takes a convolution

}  // namespace

AuxNet::AuxNet(const std::string& path, const char* what)
    : path_(path), what_(what), st_(std::make_unique<SafeTensors>(path)) {
    for (const auto& t : st_->tensors()) file_bytes_ += t.second.end - t.second.begin;
}

AuxNet::~AuxNet() { release(); }

void* AuxNet::alloc(size_t bytes) {
    void* p = nullptr;
    if (!gpu::alloc_device(&p, std::max<size_t>(bytes, 4)))
        throw std::runtime_error("image engine (" + what_ + "): GPU memory: " + gpu::last_error());
    weights_.push_back(p);
    return p;
}

void AuxNet::drop_scratch() {
    if (s_ != nullptr) gpu::stream_sync(s_);
    for (auto& w : work_) gpu::free(w.first);
    work_.clear();
}

void AuxNet::release() {
    drop_scratch();
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

void AuxNet::product(const uint16_t* x16, const Lin& l, float* y, int64_t T, int64_t ldy) {
    gemm_.f16(x16, l.w, y, T, l.n, l.k, ldy);
    if (l.b != nullptr) {
        if (ldy != 0 && ldy != l.n) throw std::logic_error(what_ + ": a biased product with a row stride");
        qi_bias_add(y, l.b, T, l.n, s_);
    }
}

void AuxNet::convolve(const Conv& c, const uint16_t* x16, int64_t H, int64_t W, float* y, int col_slot, bool add) {
    const int64_t pad = c.pad, k = c.k, stride = c.stride;
    const int64_t OH = (H + 2 * pad - k) / stride + 1, OW = (W + 2 * pad - k) / stride + 1;
    const int64_t P = OH * OW;
    if (c.k == 1 && c.stride == 1 && c.pad == 0) {
        gemm_.f16(x16, c.w, y, P, c.o, c.c, 0, false, add);
        if (c.b != nullptr) qi_bias_add(y, c.b, P, c.o, s_);
        return;
    }
#if STRATA_ONEDNN
    if (conv::onednn(core::Runtime::get().stream(s_), x16, H, W, c.c, c.w, c.o, c.k, c.stride, c.pad, c.pad, c.pad,
                     c.pad, c.b, add, y))
        return;
#endif
#if STRATA_CUDNN
    if (c.stride == 1 && 2 * c.pad + 1 == c.k &&
        conv::cudnn(core::Runtime::get().stream(s_), x16, H, W, c.c, c.w, c.o, c.k, c.b, add, y))
        return;
#endif
    const int64_t K = (int64_t) c.k * c.k * c.c;
    const int64_t rows = std::max<int64_t>(1, std::min<int64_t>(P, kColBytes / (K * 2)));
    auto* col = scratch_as<uint16_t>(col_slot, (size_t) (rows * K));
    for (int64_t p0 = 0; p0 < P; p0 += rows) {
        const int64_t n = std::min(rows, P - p0);
        aux_im2col16(x16, H, W, c.c, c.k, c.stride, c.pad, OW, p0, n, col, s_);
        gemm_.f16(col, c.w, y + p0 * c.o, n, c.o, K, 0, false, add);
    }
    if (c.b != nullptr) qi_bias_add(y, c.b, P, c.o, s_);
}

void AuxNet::host(const char* name, const float* dev, size_t n) {
    if (!dump) return;
    std::vector<float> h(n);
    gpu::stream_sync(s_);
    gpu::copy(h.data(), dev, n * 4);
    dump(name, h);
}

}  // namespace strata::image
