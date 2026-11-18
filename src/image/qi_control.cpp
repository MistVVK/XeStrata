// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_control.cpp - see include/strata/image/qi_control.hpp.

#include "strata/image/qi_control.hpp"

#include "dit_rows.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/platform/mapped_file.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::image {

namespace qi_control_k {   // the kernels' names (an anonymous namespace mangles the same in every file)

// tokens [hw][kp] FP16 of the conditioning cond [k][hw] (the columns past k zero)
inline void tokens(const float* cond, int64_t k, int64_t kp, int64_t hw, uint16_t* out, void* s) {
    core::Runtime::get().stream(s).parallel_for(sycl::range<1>((size_t) (hw * kp)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / kp, c = i - t * kp;
        out[i] = kernels::f16_from_f32(c < k ? cond[c * hw + t] : 0.f);
    });
}

// the stream as the DiT's first block takes the sequence: c = x + b, and the target's rows (from T - hw) + rows
inline void start(float* c, const float* x, const float* b, const float* rows, int64_t T, int64_t hw, int64_t C,
                  void* s) {
    const int64_t pre = T - hw;
    core::Runtime::get().stream(s).parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / C, n = i - t * C;
        c[i] = x[i] + b[n] + (t >= pre ? rows[(t - pre) * C + n] : 0.f);
    });
}

// x += strength (h + b): a control block's skip added to every row of the sequence
inline void skip_add(float* x, const float* h, const float* b, float strength, int64_t T, int64_t C, void* s) {
    core::Runtime::get().stream(s).parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        x[i] += strength * (h[i] + b[i % C]);
    });
}

}  // namespace qi_control_k

namespace {

constexpr int kTypeQ8_0 = 8;

template <class T> T* dev(std::vector<void*>& owned, const std::vector<T>& v) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, v.size() * sizeof(T)))
        throw std::runtime_error(std::string("image engine (ControlNet): GPU memory: ") + gpu::last_error());
    owned.push_back(p);
    if (!gpu::copy(p, v.data(), v.size() * sizeof(T)))
        throw std::runtime_error(std::string("image engine (ControlNet): copy: ") + gpu::last_error());
    return p;
}

template <class T> T* scratch(int64_t n) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) n * sizeof(T)))
        throw std::runtime_error(std::string("image engine (ControlNet): GPU memory: ") + gpu::last_error());
    return p;
}

}  // namespace

QiControl::QiControl() = default;

QiControl::~QiControl() {
    drop_activations();
    clear();
    for (void* p : owned_) gpu::free(p);
}

bool QiControl::open(const std::string& path, QiDit& dit, std::string& err, bool dry) {
    try {
        path_ = path;
        dit_ = &dit;
        dry_ = dry;
        file_ = std::make_unique<platform::MappedFile>(path);
        SafeTensors st(path);
        if (st.data_base() > file_->size()) throw std::runtime_error(path + ": data outside the file");
        const uint8_t* data = file_->data() + st.data_base();
        const int64_t C = dit.C_;
        auto need = [&](const std::string& name, std::initializer_list<int64_t> shape) -> const SafeTensor& {
            const SafeTensor* t = st.find(name);
            if (t == nullptr) throw std::runtime_error(path + ": no " + name + " (not the Fun ControlNet Union)");
            if (t->dtype != "BF16" || t->shape != std::vector<int64_t>(shape))
                throw std::runtime_error(path + ": " + name + " is not the BF16 shape the DiT takes");
            auto invalid = [&](const char* why) {
                std::string msg = path;
                msg += ": ";
                msg += name;
                msg += why;
                return std::runtime_error(msg);
            };
            uint64_t bytes = 2;
            for (int64_t d : t->shape) {
                if (d <= 0 || (uint64_t) d > std::numeric_limits<uint64_t>::max() / bytes)
                    throw invalid(": invalid shape");
                bytes *= (uint64_t) d;
            }
            if (t->end < t->begin || t->end > file_->size() - st.data_base() || t->end - t->begin != bytes)
                throw invalid(": invalid data_offsets");
            return *t;
        };
        const SafeTensor* inw = st.find("control_img_in.weight");
        if (inw == nullptr || inw->shape.size() != 2 || inw->shape[1] <= 0 || inw->shape[1] > INT64_MAX - 7) throw std::runtime_error(path + ": not the Fun ControlNet Union");
        in_k_ = inw->shape[1];
        need("control_img_in.weight", {C, in_k_});
        in_kp_ = (in_k_ + 7) / 8 * 8;
        need("control_img_in.bias", {C});
        need("control_blocks.0.before_proj.weight", {C, C});
        need("control_blocks.0.before_proj.bias", {C});
        const int64_t inter = dit.inter_;
        int nb = 0;
        while (st.find("control_blocks." + std::to_string(nb) + ".after_proj.weight") != nullptr) ++nb;
        if (nb == 0 || dit.blocks() % nb != 0)
            throw std::runtime_error(path + ": " + std::to_string(nb) + " control blocks for the DiT's " +
                                     std::to_string(dit.blocks()));
        // the DiT's block i * (blocks / nb) is followed by control block i (every second one of 32 for 16)
        pair_.assign((size_t) dit.blocks(), -1);
        const int every = dit.blocks() / nb;
        for (int i = 0; i < nb; ++i) pair_[(size_t) i * (size_t) every] = i;
        // the products in the DiT's own form (its blocks' shape): W8A8 where the DiT took it, else Q8_0 (FP16
        // products: QiDit's open says why not Q8_0's int8 rows)
        const bool w8 = dit.w8_;
        weight_bytes_ = 0;
        for (int b = 0; b < nb; ++b) {
            const std::string p = "control_blocks." + std::to_string(b) + ".";
            Block blk;
            struct Item {
                QiDit::W* w;
                std::vector<std::string> names;   // the file's matrices, stacked (gate_layer over proj)
                int64_t n, k;
            };
            std::vector<Item> items = {{&blk.q, {"attn.to_q.weight"}, C, C},
                                       {&blk.k, {"attn.to_k.weight"}, C, C},
                                       {&blk.v, {"attn.to_v.weight"}, C, C},
                                       {&blk.o, {"attn.to_out.0.weight"}, C, C},
                                       {&blk.gu, {"img_mlp.gate_layer.weight", "img_mlp.proj.weight"}, 2 * inter, C},
                                       {&blk.down, {"img_mlp.out.weight"}, C, inter},
                                       {&blk.after, {"after_proj.weight"}, C, C}};
            uint64_t bytes = 0;
            for (Item& it : items) {
                QiDit::W& w = *it.w;
                w.n = it.n;
                w.k = it.k;
                w.w8 = w8 && it.k % kQiRotate == 0;
                w.type = kTypeQ8_0;
                w.row_bytes = w.w8 ? (uint64_t) it.k : (uint64_t) (it.k / 32 * 34);
                bytes += w.row_bytes * (uint64_t) it.n + (w.w8 ? (uint64_t) it.n * 4 : 0);
            }
            const std::string nqn = p + "attn.norm_q.weight", nkn = p + "attn.norm_k.weight",
                              abn = p + "after_proj.bias";
            const int64_t hd = dit.C_ / dit.heads_;
            need(nqn, {hd});
            need(nkn, {hd});
            need(abn, {C});
            std::vector<uint8_t> host(dry ? 0 : (size_t) (bytes + (uint64_t) (2 * hd + C) * 4));
            std::vector<LayerStream::Span> spans;
            uint64_t at = 0;
            std::vector<size_t> wspan;
            uint64_t lo = UINT64_MAX, hi = 0;
            for (Item& it : items) {
                QiDit::W& w = *it.w;
                const uint64_t wb = w.row_bytes * (uint64_t) w.n + (w.w8 ? (uint64_t) w.n * 4 : 0);
                if (!dry) {
                    std::vector<uint16_t> stacked;
                    const uint16_t* src = nullptr;
                    int64_t rows = 0;
                    for (const std::string& nm : it.names) {
                        const SafeTensor& t = need(p + nm, {it.n / (int64_t) it.names.size(), it.k});
                        lo = std::min(lo, st.data_base() + t.begin);
                        hi = std::max(hi, st.data_base() + t.end);
                        if (it.names.size() == 1) {
                            src = reinterpret_cast<const uint16_t*>(data + t.begin);
                        } else {
                            stacked.resize((size_t) (it.n * it.k));
                            std::memcpy(stacked.data() + rows * it.k, data + t.begin, t.end - t.begin);
                        }
                        rows += t.shape[0];
                    }
                    if (src == nullptr) src = stacked.data();
                    // W8A8's rows as kernels::rows_i8 makes them, else Q8_0
                    dit_rows::quantize_matrix(src, w.n, w.k, host.data() + at, w.w8);
                }
                spans.push_back({host.data() + at, wb, 0});
                wspan.push_back(spans.size() - 1);
                at += wb;
            }
            size_t vspan[3];
            const std::string vecs[3] = {nqn, nkn, abn};
            for (int j = 0; j < 3; ++j) {
                const int64_t n = st.find(vecs[j])->elements();
                if (!dry) {
                    const std::vector<float> v = st.f32(vecs[j]);
                    std::memcpy(host.data() + at, v.data(), (size_t) n * 4);
                }
                spans.push_back({host.data() + at, (uint64_t) n * 4, 0});
                vspan[j] = spans.size() - 1;
                at += (uint64_t) n * 4;
            }
            const int layer = stream_.add(spans);
            for (size_t j = 0; j < items.size(); ++j)
                items[j].w->d = reinterpret_cast<const void*>(stream_.offset(layer, (int) wspan[j]));
            blk.nq = stream_.offset(layer, (int) vspan[0]);
            blk.nk = stream_.offset(layer, (int) vspan[1]);
            blk.after_b = stream_.offset(layer, (int) vspan[2]);
            if (!dry && hi > lo) file_->drop_cache(lo, hi - lo);   // the bf16 pages: read once
            host_.push_back(std::move(host));
            blocks_.push_back(blk);
        }
        // control_img_in and before_proj in FP16 (mapped once a request), their biases FP32
        weight_bytes_ = (uint64_t) (C * in_kp_ + C * C) * 2 + (uint64_t) (2 * C) * 4;
        if (dry) return true;
        auto f16_of = [&](const std::string& name, int64_t rows, int64_t k, int64_t kp) {
            const std::vector<float> v = st.f32(name);
            std::vector<uint16_t> o((size_t) (rows * kp), 0);
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < k; ++c)
                    o[(size_t) (r * kp + c)] = kernels::f16_from_f32(v[(size_t) (r * k + c)]);
            return o;
        };
        in_w_ = dev(owned_, f16_of("control_img_in.weight", C, in_k_, in_kp_));
        bp_w_ = dev(owned_, f16_of("control_blocks.0.before_proj.weight", C, C, C));
        in_b_ = dev(owned_, st.f32("control_img_in.bias"));
        bp_b_ = dev(owned_, st.f32("control_blocks.0.before_proj.bias"));
        stream_.configure(stream_.total(), dit.s_);   // all resident until a budget is set
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

uint64_t QiControl::activation_bytes(int64_t rows, int64_t hw) const {
    const int64_t C = dit_ != nullptr ? dit_->C_ : 0;
    // the stream, the mapped rows; while mapping: the tokens, control_img_in's output and its FP16
    return (uint64_t) (rows * C + hw * C) * 4 + (uint64_t) (hw * in_kp_) * 2 + (uint64_t) (hw * C) * 6;
}

void QiControl::set_input(const float* cond, int64_t h, int64_t w, float strength, float sigma_lo, float sigma_hi) {
    if (dit_ == nullptr || dry_) throw std::logic_error("image engine (ControlNet): not open");
    clear();
    QiDit& d = *dit_;
    const int64_t hw = h * w, C = d.C_;
    auto* tok = scratch<uint16_t>(hw * in_kp_);
    auto* y = scratch<float>(hw * C);
    auto* y16 = scratch<uint16_t>(hw * C);
    struct Free {
        std::vector<void*> ps;
        void* s;
        ~Free() {
            gpu::stream_sync(s);
            for (void* p : ps) gpu::free(p);
        }
    } tmp{{tok, y, y16}, d.s_};
    qi_control_k::tokens(cond, in_k_, in_kp_, hw, tok, d.s_);
    d.gemm_.f16(tok, in_w_, y, hw, C, in_kp_);
    ProductStore p;
    p.in = y; p.bias = in_b_; p.rows = hw; p.cols = C; p.out16 = y16;
    product_store(p, d.s_);
    bp_rows_ = scratch<float>(hw * C);
    d.gemm_.f16(y16, bp_w_, bp_rows_, hw, C, C);
    hw_ = hw;
    strength_ = strength;
    sigma_lo_ = std::min(sigma_lo, sigma_hi);
    sigma_hi_ = std::max(sigma_lo, sigma_hi);
}

void QiControl::clear() {
    if (bp_rows_ != nullptr) {
        if (dit_ != nullptr) gpu::stream_sync(dit_->s_);
        gpu::free(bp_rows_);
    }
    bp_rows_ = nullptr;
    hw_ = 0;
}

void QiControl::drop_activations() {
    if (dit_ != nullptr) gpu::stream_sync(dit_->s_);
    if (c_ != nullptr) gpu::free(c_);
    c_ = nullptr;
    c_rows_ = 0;
    if (kmean_.buf != nullptr) gpu::free(kmean_.buf);
    kmean_.clear();
}

void QiControl::begin(int64_t T) {
    QiDit& d = *dit_;
    if (T > c_rows_) {
        drop_activations();
        c_ = scratch<float>(T * d.C_);
        c_rows_ = T;
    }
    if (kmean_.buf == nullptr) {   // the int8 attention's K means, a block and context (zero until a step makes one)
        const int64_t n = (int64_t) blocks_.size() * 2;
        kmean_.init(scratch<float>(n * d.C_), n, d.C_);
        gpu::memset_async(kmean_.buf, 0, (size_t) (n * d.C_) * 4, d.s_);
    }
    qi_control_k::start(c_, d.x_, bp_b_, bp_rows_, T, hw_, d.C_, d.s_);
}

void QiControl::after_block(int bi, int64_t T) {
    const int k = bi < (int) pair_.size() ? pair_[(size_t) bi] : -1;
    if (k >= 0) run_block(k, T);
}

// control block k on the stream (the DiT's block as QiDit::forward runs it, every row asked: no prefix cache), then
// its skip after_proj(c) added to the DiT's sequence
void QiControl::run_block(int k, int64_t T) {
    QiDit& d = *dit_;
    const Block& b = blocks_[(size_t) k];
    const int64_t C = d.C_, hd = C / d.heads_, nt = d.n_text_;
    const uint8_t* base = stream_.begin(k);
    auto at = [base](const QiDit::W& w) {
        QiDit::W r = w;
        r.d = base + reinterpret_cast<uintptr_t>(w.d);
        return r;
    };
    const QiDit::W wq = at(b.q), wk = at(b.k), wv = at(b.v), wo = at(b.o), wgu = at(b.gu), wdown = at(b.down),
                   wa = at(b.after);
    const auto* nq = reinterpret_cast<const float*>(base + b.nq);
    const auto* nk = reinterpret_cast<const float*>(base + b.nk);
    const auto* ab = reinterpret_cast<const float*>(base + b.after_b);
    auto mod = [&](int i) { return d.mods_ + (int64_t) i * 2 * C; };
    qi_ln_mod16(c_, mod(0), nt, T, C, d.h16_, d.s_);
    DitOut qkv[3];
    const bool i8 = wq.w8 && wk.w8 && wv.w8;
    d.product(d.h16_, wq, d.q_, T, 0, 0, -1, false, i8);
    qkv[0] = d.i8_;
    d.product(d.h16_, wk, d.k_, T, 0, 0, -1, wq.w8, i8);
    qkv[1] = d.i8_;
    d.product(d.h16_, wv, d.v_, T, 0, 0, -1, wk.w8, i8);
    qkv[2] = d.i8_;
    qi_qk_norm_rope16(d.q_, d.k_, d.v_, nq, nk, d.cos_, d.sin_, T, d.heads_, hd, d.q16_, d.k16_, d.v16_, d.s_,
                      i8 ? qkv : nullptr);
    const int64_t slot = (int64_t) k * 2 + d.ctx_i_;
    const auto st = kmean_.stats(slot);
    qi_attention(d.q16_, d.k16_, d.v16_, T, d.heads_, hd, nt, d.o16_, d.s_, d.key_lim_, 0, true,
                 d.key_lim_ != nullptr ? &d.ref_spans_ : nullptr, &st);
    kmean_.mark(slot);
    const bool od = d.product(d.o16_, wo, d.h_, T, 0, 0, -1, false, true);
    qi_gate_add(c_, d.h_, mod(1), nt, T, C, d.s_, od ? &d.i8_ : nullptr);
    for (int64_t m0 = 0; m0 < T; m0 += d.mlp_rows_) {
        const int64_t mr = std::min(d.mlp_rows_, T - m0), mt = std::max<int64_t>(0, std::min(nt - m0, mr));
        qi_ln_mod16(c_ + m0 * C, mod(2), mt, mr, C, d.h16_, d.s_);
        d.mlp(wgu, wdown, c_ + m0 * C, mod(3), mt, mr);
    }
    kernels::f32_to_f16_bulk(c_, d.h16_, T * C, d.s_);
    d.product(d.h16_, wa, d.h_, T);
    qi_control_k::skip_add(d.x_, d.h_, ab, strength_, T, C, d.s_);
    stream_.end(k);
}

}  // namespace strata::image
