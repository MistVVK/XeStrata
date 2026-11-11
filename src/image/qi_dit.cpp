// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_dit.cpp - see include/strata/image/qi_dit.hpp.

#include "strata/image/qi_dit.hpp"
#include "strata/image/dit_file.hpp"
#include "strata/image/flow_sampler.hpp"
#include "strata/image/lora_names.hpp"
#include "strata/image/qi_control.hpp"
#include "strata/image/qi_gguf.hpp"

#include "dit_rows.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/image/gpu_buf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <tuple>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr int kTypeF16 = 1, kTypeF32 = 0, kTypeBF16 = 30;
constexpr int64_t kHead = 128;
const int kAxes[3] = {16, 56, 56};   // the rotary embedding's dims per axis: text position, height, width

void put(void* d, const void* h, size_t bytes) {
    if (!gpu::copy(d, h, bytes)) throw std::runtime_error(std::string("image engine: copy: ") + gpu::last_error());
}

void put_async(void* d, const void* h, size_t bytes, void* s) {
    if (!gpu::copy_async(d, h, bytes, s))
        throw std::runtime_error(std::string("image engine: copy: ") + gpu::last_error());
}

// A file's key that changes a text encoder (kohya's lora_te_, diffusers' text_encoder. prefix, the te. other tools
// write): the image engine applies adapters to the DiT only, and says so rather than leaving it out.
bool is_text_encoder_key(const std::string& k) {
    return k.rfind("lora_te", 0) == 0 || k.rfind("text_encoder", 0) == 0 || k.rfind("te.", 0) == 0;
}

// Six of the names, then "and N more": the errors' list of a file's keys the DiT does not take.
std::string name_list(const std::vector<std::string>& names) {
    std::string out;
    const size_t n = std::min<size_t>(names.size(), 6);
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) out += ", ";
        out += names[i];
    }
    if (names.size() > n) out += " (and " + std::to_string(names.size() - n) + " more)";
    return out;
}

}  // namespace

std::vector<float> philox_randn(uint64_t seed, uint64_t n, uint32_t offset) {
    std::vector<float> out(n);
    const uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u, W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    const float inv32 = 2.3283064e-10f, inv32_2pi = 2.3283064e-10f * 6.2831855f;
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t c0 = offset, c1 = 0, c2 = (uint32_t) i, c3 = 0;
        uint32_t k0 = (uint32_t) seed, k1 = (uint32_t) (seed >> 32);
        for (int r = 0; r < 10; ++r) {
            const uint64_t v1 = (uint64_t) c0 * M0, v2 = (uint64_t) c2 * M1;
            const uint32_t n0 = (uint32_t) (v2 >> 32) ^ c1 ^ k0, n2 = (uint32_t) (v1 >> 32) ^ c3 ^ k1;
            c1 = (uint32_t) v2;
            c3 = (uint32_t) v1;
            c0 = n0;
            c2 = n2;
            k0 += W0;
            k1 += W1;
        }
        const float u = (float) c0 * inv32 + inv32 / 2, v = (float) c1 * inv32_2pi + inv32_2pi / 2;
        out[i] = std::sqrt(-2.0f * std::log(u)) * std::sin(v);
    }
    return out;
}

const std::vector<float>& turbo_sigmas() {
    static const std::vector<float> s = {1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f,
                                         0.414568f, 0.0f};
    return s;
}

QiDit::QiDit() = default;

QiDit::~QiDit() {
    for (void* p : allocs_) gpu::free(p);
}

QiDit::W QiDit::upload(const std::string& name) {
    const TensorInfo* t = find_checked(g_->gguf(), "model.diffusion_model." + name);
    if (t == nullptr) throw std::runtime_error("image engine: the DiT has no " + name);
    W w;
    w.k = (int64_t) t->shape[0];
    w.n = t->shape.size() > 1 ? (int64_t) t->shape[1] : 1;
    const uint64_t bytes = tensor_payload_bytes(*t);
    if (bytes == 0) throw std::runtime_error("image engine: " + name + ": unknown type " + std::to_string(t->type));
    if ((int) t->type == kTypeF32 && w.n > 1 && dry_) {
        w.type = kTypeF16;
        w.row_bytes = (uint64_t) w.k * 2;
        weight_bytes_ += (uint64_t) (w.n * w.k) * 2;
        return w;
    }
    if ((int) t->type == kTypeF32 && w.n > 1) {   // a matrix in FP32 goes to the GPU as FP16 (the products take it)
        std::vector<uint16_t> h((size_t) (w.n * w.k));
        const auto* f = reinterpret_cast<const float*>(g_->data(*t));
        for (size_t i = 0; i < h.size(); ++i) h[i] = kernels::f16_from_f32(f[i]);
        uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size(), "image engine");
        put(d, h.data(), h.size() * 2);
        w.type = kTypeF16;
        w.d = d;
        w.row_bytes = (uint64_t) w.k * 2;
        weight_bytes_ += h.size() * 2;
        return w;
    }
    w.type = (int) t->type;
    w.row_bytes = bytes / (uint64_t) w.n;
    weight_bytes_ += bytes;
    if (dry_) return w;
    uint8_t* d = alloc<uint8_t>(allocs_, (int64_t) bytes, "image engine");
    put(d, g_->data(*t), bytes);
    w.d = d;
    return w;
}

const float* QiDit::upload_f32(const std::string& name, float add) {
    const TensorInfo* t = find_checked(g_->gguf(), "model.diffusion_model." + name);
    if (t == nullptr) throw std::runtime_error("image engine: the DiT has no " + name);
    const auto n = (int64_t) t->elements();
    if (dry_) {
        weight_bytes_ += (uint64_t) n * 4;
        return nullptr;
    }
    std::vector<float> h((size_t) n);
    const uint8_t* src = g_->data(*t);
    if ((int) t->type == kTypeF32) std::memcpy(h.data(), src, (size_t) n * 4);
    else if ((int) t->type == kTypeBF16)
        for (int64_t i = 0; i < n; ++i) h[(size_t) i] = kernels::f32_from_bf16(reinterpret_cast<const uint16_t*>(src)[i]);
    else throw std::runtime_error("image engine: " + name + ": a vector of type " + std::to_string(t->type));
    for (float& v : h) v += add;
    float* d = alloc<float>(allocs_, n, "image engine");
    put(d, h.data(), (size_t) n * 4);
    weight_bytes_ += (uint64_t) n * 4;
    return d;
}

bool QiDit::open(const std::string& path, void* stream, std::string& err, bool dry) {
    try {
        s_ = stream;
        dry_ = dry;
        g_ = std::make_unique<QiFile>(path);
        iin_ = upload("img_in.weight");
        in_ch_ = iin_.k;
        C_ = iin_.n;
        t1_ = upload("time_text_embed.timestep_embedder.linear_1.weight");
        t2_ = upload("time_text_embed.timestep_embedder.linear_2.weight");
        mod_ = upload("modulation.1.weight");
        nout_ = upload("norm_out.linear.weight");
        pout_ = upload("proj_out.weight");
        tin_ = upload("txt_in.in_layer.weight");
        tout_ = upload("txt_in.out_layer.weight");
        tnorm_ = upload_f32("txt_in.text_norm.weight", 1.0f);   // zero-centred: the weight is w + 1
        int64_t scratch = 0;
        norms32_.clear();
        for (int i = 0;; ++i) {   // the blocks: spans of the mapped file, on the GPU through blocks_stream_
            const std::string p = "transformer_blocks." + std::to_string(i) + ".";
            if (find_checked(g_->gguf(), "model.diffusion_model." + p + "attn.to_q.weight") == nullptr) break;
            Block b;
            std::vector<LayerStream::Span> spans;
            auto span = [&](const std::string& name, W& w) {
                const std::string full = std::string(p).append(name);
                const TensorInfo* t = find_checked(g_->gguf(), std::string("model.diffusion_model.").append(full));
                if (t == nullptr) throw std::runtime_error(std::string("image engine: the DiT has no ").append(full));
                w.type = (int) t->type;
                w.k = (int64_t) t->shape[0];
                w.n = t->shape.size() > 1 ? (int64_t) t->shape[1] : 1;
                const uint64_t bytes = tensor_payload_bytes(*t);
                w.row_bytes = bytes / (uint64_t) w.n;
                spans.push_back({g_->data(*t), bytes, 0});
                return spans.size() - 1;
            };
            W nq, nk;
            const size_t iq = span("attn.to_q.weight", b.q), ik = span("attn.to_k.weight", b.k),
                         iv = span("attn.to_v.weight", b.v), io = span("attn.to_out.0.weight", b.o),
                         ig = span("img_mlp.gate_up.weight", b.gu), id = span("img_mlp.out.weight", b.down),
                         inq = span("attn.norm_q.weight", nq), ink = span("attn.norm_k.weight", nk);
            // the q/k norms in FP32, as the kernels read them: an FP16 GGUF's (the F16 file the LoRA trainer reads)
            // widened once into copies of the engine's own, which the layer streams instead of the file's bytes
            for (const auto& [w, i] : {std::pair<const W*, size_t>{&nq, inq}, {&nk, ink}}) {
                if (w->type == 0) continue;
                if (w->type != 1)
                    throw std::runtime_error("image engine: the DiT's q/k norms are neither FP32 nor FP16");
                const auto* h = static_cast<const uint16_t*>(spans[i].host);
                const size_t n = (size_t) (spans[i].bytes / 2);
                std::vector<float>& v = norms32_.emplace_back(n);
                for (size_t j = 0; j < n; ++j) v[j] = kernels::f32_from_f16(h[j]);
                spans[i] = {v.data(), (uint64_t) n * 4, 0};
            }
            const int layer = blocks_stream_.add(spans);
            auto off = [&](size_t k) { return blocks_stream_.offset(layer, (int) k); };
            b.q.d = reinterpret_cast<const void*>(off(iq));
            b.k.d = reinterpret_cast<const void*>(off(ik));
            b.v.d = reinterpret_cast<const void*>(off(iv));
            b.o.d = reinterpret_cast<const void*>(off(io));
            b.gu.d = reinterpret_cast<const void*>(off(ig));
            b.down.d = reinterpret_cast<const void*>(off(id));
            b.nq = off(inq);
            b.nk = off(ink);
            for (const W* w : {&b.q, &b.k, &b.v, &b.o, &b.gu, &b.down}) scratch = std::max(scratch, w->n * w->k);
            blocks_.push_back(b);
        }
        if (blocks_.empty()) throw std::runtime_error("image engine: the DiT has no transformer blocks");
        inter_ = blocks_[0].gu.n / 2;
        heads_ = C_ / kHead;
        build_lora_targets();
        for (const W* w : {&t1_, &t2_, &mod_, &nout_, &pout_, &tin_, &tout_, &iin_})
            scratch = std::max(scratch, w->n * w->k);
        // the blocks' products in int8 where the GPU's library takes int8 products and they are not clearly the
        // slower here (W8A8, prefill::Gemm::i8; the choice every DiT makes, dit_rows::w8_at_open, once a GPU): each
        // weight dequantized to FP16 in the products' scratch and rotated and quantized again a row at a time, in
        // place, each time a block is copied to the GPU (int8 values and a scale a row take less room than any
        // block format of 8 bits a value or more); else in FP16 from the file's blocks (Gemm::native), as every DiT.
        // Not Q8_0's int8 activations a scale each 32 on NVIDIA's int8 tensor cores, which
        // this DiT took in W8A8's place until 2026-11-10: half W8A8's speed and short of FP16's (RTX 4070,
        // Qwen-Image-2.1-Turbo Q8_0, 1024 x 1024, 8 steps: 3.43 s of sampling in W8A8, 6.85 in Q8_0's, 9.26 in
        // FP16), and W8A8 is the faster form wherever those cores are (their int8 rate twice their FP16 rate)
        uint64_t w8t_bytes = 0;
        if (dit_rows::w8_at_open(C_, 2 * inter_, "Qwen-Image's DiT", s_) == 1) {
            for (Block& b : blocks_)
                for (W* w : {&b.q, &b.k, &b.v, &b.o, &b.gu, &b.down}) {
                    w->w8 = w->k % kQiRotate == 0 &&
                            (uint64_t) w->n * w->row_bytes >= (uint64_t) w->n * (uint64_t) w->k + 4 * (uint64_t) w->n;
                    // a block format of fewer bits (Q6_K, Q5_K, Q4_K...): kept as it is in VRAM and written as int8
                    // rows into a scratch at each product.  The int8 products' rate against the FP16 products' (half
                    // of it with FP32 sums on NVIDIA's GeForce cards) pays for the rewrite many times over (RTX 4070,
                    // Qwen-Image-2.1-Turbo Q6_K, 1024 x 1024, 8 steps, 2026-11-05: 9.12 s of sampling in FP16, 4.5 so
                    // with the int8 attention, 0.48 of it the rewrites)
                    if (!w->w8 && w->k % kQiRotate == 0 && w->type != kTypeF16 && w->type != kTypeF32 &&
                        kernels::dequant_bf16_supported(w->type)) {
                        w->w8 = w->w8t = true;
                        w8t_rows_ = std::max(w8t_rows_, w->n);
                        w8t_bytes = std::max(w8t_bytes, (uint64_t) (w->n * w->k));
                    }
                    w8_ = w8_ || w->w8;
                }
            if (w8t_bytes > 0) weight_bytes_ += w8t_bytes + (uint64_t) (kW8tScales * w8t_rows_) * 4;
        }
        install_prepare();
        if (dry_) return true;
        if (!gemm_.init(s_, scratch, err)) return false;
        if (w8t_bytes > 0) {
            w8t_ = alloc<int8_t>(allocs_, (int64_t) w8t_bytes, "image engine");
            w8t_s_ = alloc<float>(allocs_, kW8tScales * w8t_rows_, "image engine");
        }
        blocks_stream_.configure(blocks_stream_.total(), s_);   // all resident until a budget is set
        temb_ = alloc<float>(allocs_, 2 * C_, "image engine");
        mods_ = alloc<float>(allocs_, 8 * C_, "image engine");
        scale_ = alloc<float>(allocs_, 2 * C_, "image engine");
        small16_ = alloc<uint16_t>(allocs_, 2 * C_, "image engine");
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

int8_t* QiDit::xq8(int64_t T, int64_t K, float** sx) {
    const auto need = (size_t) (T * K);
    if (need > xq_bytes_) {
        gpu::stream_sync(s_);   // the old buffer may still be read
        if (xq_ != nullptr) {
            allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), xq_), allocs_.end());
            gpu::free(xq_);
        }
        xq_ = alloc<uint8_t>(allocs_, (int64_t) need, "image engine");
        xq_bytes_ = need;
    }
    if (T > xs_rows_) {
        gpu::stream_sync(s_);
        if (xs_ != nullptr) {
            allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), xs_), allocs_.end());
            gpu::free(xs_);
        }
        xs_ = alloc<float>(allocs_, 2 * T, "image engine");
        xs_rows_ = T;
    }
    *sx = xs_ + xs_next_ * xs_rows_;
    xs_next_ ^= 1;
    return static_cast<int8_t*>(xq_);
}

// A block's weights as each copy of it reaches the GPU: rewritten as int8 rows (w8), and the adapters baked in
// (set_loras); none of it where nothing is to be done (the stream then copies them as they are)
void QiDit::install_prepare() {
    if (!w8_ && !baked_any_) {
        blocks_stream_.set_prepare(nullptr);
        return;
    }
    blocks_stream_.set_prepare([this](int i, uint8_t* d) {
        const Block& b = blocks_[(size_t) i];
        for (const W* w : {&b.q, &b.k, &b.v, &b.o, &b.gu, &b.down}) {
            auto* p = d + reinterpret_cast<uintptr_t>(w->d);
            if (w->w8 && !w->w8t) {
                if (w->type == kTypeF16)   // already FP16: the values the dequantization would write
                    gpu::copy_async(gemm_.scratch(), p, (size_t) (w->n * w->k) * 2, s_);
                else
                    kernels::dequant_f16(w->type, p, 0, w->n, w->k, gemm_.scratch(), s_);
                kernels::rows_i8(gemm_.scratch(), false, w->n, w->k, reinterpret_cast<int8_t*>(p),
                           reinterpret_cast<float*>(p + w->n * w->k), nullptr, s_);
            }
            if (w->baked)
                w->lora->bake(w->w8 && !w->w8t ? -8 : w->type, p, w->n, w->k, gemm_, bake_tmp_, bake_at16_, s_);
        }
    });
}

bool QiDit::product(const uint16_t* x16, const W& w, float* y, int64_t T, int64_t ldy, int64_t row0, int64_t rows,
                    bool same_input, bool defer) {
    if (rows < 0) rows = w.n;
    LoraLayer* lora = w.baked ? nullptr : w.lora;
    if (lora != nullptr) defer = false;   // a delta is FP32: the int32 sums are scaled into y first
    if (ldy <= 0) ldy = rows;
    // a DoRA's factor of the weight's rows row0 .. row0 + rows - 1: the base product's, then the delta's own
    const float* col = lora != nullptr && lora->has_base() ? lora->base() + row0 : nullptr;
    const auto* base = static_cast<const uint8_t*>(w.d) + (uint64_t) row0 * w.row_bytes;
    bool deferred = false;
    if (w.w8) {
        // the activations rotated and quantized as the weights (their int8 values, a scale a token), the int32 sums
        // into y and scaled there, or by their reader (defer)
        float* sx = nullptr;
        int8_t* x8 = nullptr;
        if (same_input) {
            x8 = static_cast<int8_t*>(xq_);
            sx = xs_ + (xs_next_ ^ 1) * xs_rows_;
        } else {
            x8 = xq8(T, w.k, &sx);
            kernels::rows_i8(x16, false, T, w.k, x8, sx, nullptr, s_);
        }
        const auto* w8 = static_cast<const uint8_t*>(w.d);
        const float* sw = reinterpret_cast<const float*>(w8 + w.n * w.k) + row0;
        if (w.w8t) {   // the weight's rows as int8 into the scratch, as install_prepare writes a w8 weight's
            float* s = w8t_s_ + w8t_next_ * w8t_rows_;
            w8t_next_ = (w8t_next_ + 1) % kW8tScales;
            if (!kernels::rows_i8_blocks(w.type, base, rows, w.k, w8t_, s, s_)) {
                kernels::dequant_f16(w.type, base, 0, rows, w.k, gemm_.scratch(), s_);
                kernels::rows_i8(gemm_.scratch(), false, rows, w.k, w8t_, s, nullptr, s_);
            }
            w8 = reinterpret_cast<const uint8_t*>(w8t_) - row0 * w.k;
            sw = s;
        }
        // a reader's values as BF16 where the fused kernel is the faster (prefill::Gemm::w8a8): scaled in its
        // epilogue, half the bytes the reader reads
        if (prefill::Gemm::w8a8(x8, sx, reinterpret_cast<const int8_t*>(w8 + row0 * w.k), sw, nullptr, y, T, rows,
                                w.k, ldy, defer && col == nullptr, s_)) {
            i8_ = {nullptr, nullptr, true};
            deferred = true;
        } else {
            if (defer) {
                i8_ = {sx, sw};
                deferred = true;
            } else {
                kernels::i8_scale(y, T, rows, ldy, sx, sw, nullptr, col, s_);
            }
        }
    } else if (w.type == kTypeF16) {
        gemm_.f16(x16, reinterpret_cast<const uint16_t*>(base), y, T, rows, w.k, ldy);
    } else {
        gemm_.native(x16, w.type, base, y, T, rows, w.k, ldy);
    }
    if (deferred) return true;
    if (lora != nullptr) {
        lora_scratch(T);
        // a DoRA's base product, its rows scaled before the delta adds to them (the int8 ones were scaled above)
        if (col != nullptr && !w.w8) qi_rows_scale(y, T, rows, ldy, col, s_);
        // the delta is the input's own (not the rotated int8 one): no row scale to put back on it
        lora->add(x16, T, y, ldy, row0, rows, nullptr, gemm_, lora_f32_, lora16_, s_);
        if (lora->has_glora())   // a GLoRA's own path: the weight's product of its input, a band of the tokens at a time
            lora->add_glora(x16, T, row0, rows, nullptr, gemm_, lora_f32_, lora16_, s_,
                            [&](const uint16_t* v, int64_t t0, int64_t nt, int64_t r0, int64_t n2) {
                                glora_weight(v, w, r0, n2, nt, t0, y, ldy, row0);
                            });
    }
    return false;
}

void QiDit::mlp(const W& wgu, const W& wdown, float* x, const float* gate, int64_t n_text, int64_t mr) {
    if (wgu.w8 && wdown.w8 && wgu.lora == nullptr && wdown.lora == nullptr) {
        // SwiGLU straight from the gate_up product's sums into the down product's int8 input, and the down product's
        // sums gated into x as they are
        product(h16_, wgu, gu_, mr, 0, 0, -1, false, true);
        float* sx = nullptr;
        int8_t* x8 = xq8(mr, inter_, &sx);
        qi_swiglu_i8(reinterpret_cast<const int32_t*>(gu_), i8_, mr, inter_, x8, sx, s_);
        product(nullptr, wdown, h_, mr, 0, 0, -1, true, true);
        qi_gate_add(x, h_, gate, n_text, mr, C_, s_, &i8_);
        return;
    }
    product(h16_, wgu, gu_, mr);
    prefill::swiglu_pair(gu_, gu_ + inter_, h16_, mr, s_, inter_, 2 * inter_);
    product(h16_, wdown, h_, mr);
    qi_gate_add(x, h_, gate, n_text, mr, C_, s_);
}

void QiDit::ensure(int64_t T, int64_t HW) {
    if (T > cap_T_) {
        for (void* p : {(void*) x_, (void*) q_, (void*) k_, (void*) v_, (void*) h_, (void*) gu_, (void*) h16_,
                        (void*) q16_, (void*) k16_, (void*) v16_, (void*) o16_, (void*) cos_, (void*) sin_,
                        (void*) kmean_.buf})
            if (p != nullptr) {
                allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), p), allocs_.end());
                gpu::free(p);
            }
        const int64_t TP = qi_attn_rows(T);
        x_ = alloc<float>(allocs_, T * C_, "image engine");
        q_ = alloc<float>(allocs_, T * C_, "image engine");
        k_ = alloc<float>(allocs_, T * C_, "image engine");
        v_ = alloc<float>(allocs_, T * C_, "image engine");
        h_ = alloc<float>(allocs_, T * std::max(C_, in_ch_), "image engine");
        mlp_rows_ = std::min(T, mlp_cap_ > 0 ? mlp_cap_ : T);
        gu_ = alloc<float>(allocs_, mlp_rows_ * 2 * inter_, "image engine");
        h16_ = alloc<uint16_t>(allocs_, T * std::max(C_, inter_), "image engine");
        q16_ = alloc<uint16_t>(allocs_, TP * C_, "image engine", true);
        k16_ = alloc<uint16_t>(allocs_, TP * C_, "image engine", true);
        v16_ = alloc<uint16_t>(allocs_, TP * C_, "image engine", true);
        o16_ = alloc<uint16_t>(allocs_, T * C_, "image engine");
        cos_ = alloc<float>(allocs_, T * kHead / 2, "image engine");
        sin_ = alloc<float>(allocs_, T * kHead / 2, "image engine");
        // the int8 attention's K means, a block and context (zero until a step makes one)
        kmean_.init(alloc<float>(allocs_, (int64_t) blocks_.size() * 2 * C_, "image engine"),
                    (int64_t) blocks_.size() * 2, C_);
        gpu::memset_async(kmean_.buf, 0, (size_t) blocks_.size() * 2 * C_ * 4, s_);
        // the plain attention's FP32 output: only where the matrix engines' path is not taken
        cap_T_ = T;
        dirty_T_ = 0;
        rope_H_ = rope_W_ = rope_text_ = -1;
    }
    (void) HW;
}

void QiDit::swap_context() {
    std::swap(text_, other_.text);
    std::swap(n_text_, other_.n_text);
    std::swap(target_pos_, other_.target_pos);
    std::swap(pre_pos_, other_.pre_pos);
    std::swap(key_lim_, other_.key_lim);
    std::swap(ref_spans_, other_.ref_spans);
    other_.pk.resize(blocks_.size(), nullptr);
    other_.pv.resize(blocks_.size(), nullptr);
    for (size_t i = 0; i < blocks_.size(); ++i) {
        std::swap(blocks_[i].pk, other_.pk[i]);
        std::swap(blocks_[i].pv, other_.pv[i]);
    }
    rope_H_ = rope_W_ = rope_text_ = -1;   // the rotary tables are the context's positions
    ctx_i_ ^= 1;   // the other context's K means slot
}

void QiDit::drop_second_context() {
    gpu::stream_sync(s_);
    std::vector<void*> ps = {other_.text, other_.key_lim, other_.cos, other_.sin};
    for (uint16_t* p : other_.pk) ps.push_back(p);
    for (uint16_t* p : other_.pv) ps.push_back(p);
    for (void* p : ps)
        if (p != nullptr) {
            allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), p), allocs_.end());
            gpu::free(p);
        }
    other_ = Context{};
}

void QiDit::set_context(const float* ctx, int64_t n_ctx, const std::vector<Ref>& refs, bool second) {
    if (second) {   // the same work on the other context's state
        swap_context();
        try {
            set_context(ctx, n_ctx, refs, false);
        } catch (...) {
            swap_context();
            throw;
        }
        swap_context();
        other_.rope_H = -1;   // its positions changed: forward_pair builds its tables again
        return;
    }
    int64_t n_pre = n_ctx;
    for (const Ref& r : refs) {
        if (r.slot < 0 || r.slot + r.slots > n_ctx || r.slots * 4 != r.h * r.w || r.latent == nullptr)
            throw std::invalid_argument("image engine: a reference image's pads do not match its latent");
        n_pre += r.h * r.w - r.slots;
    }
    ensure(std::max({n_ctx, n_pre, cap_T_}), 0);
    n_text_ = n_pre;
    float* raw = q_;   // the encoder's states, then the projection's first product, then the projected rows
    put_async(raw, ctx, (size_t) (n_ctx * C_) * 4, s_);
    kernels::rms_norm_rows(raw, tnorm_, 1, nullptr, h16_, (int) (C_), (int) (n_ctx), 1e-6f, s_);
    product(h16_, tin_, k_, n_ctx);
    qi_gelu16(k_, n_ctx * C_, h16_, s_);
    product(h16_, tout_, v_, n_ctx);
    if (text_ != nullptr) {
        allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), (void*) text_), allocs_.end());
        gpu::free(text_);
    }
    text_ = alloc<float>(allocs_, n_pre * C_, "image engine");
    // the prefix: the context's rows, each reference's pads its latent's tokens
    pre_pos_.assign((size_t) (n_pre * 3), 0.0f);
    std::vector<int32_t> lim((size_t) n_pre);
    ref_spans_.clear();
    int64_t src = 0, dst = 0;
    float pos = 0;
    auto text_rows = [&](int64_t upto) {
        const int64_t n = upto - src;
        if (n > 0) gpu::copy_async(text_ + dst * C_, v_ + src * C_, (size_t) (n * C_) * 4, s_);
        for (int64_t k = 0; k < n; ++k, ++dst, pos += 1) {
            pre_pos_[(size_t) dst * 3] = pre_pos_[(size_t) dst * 3 + 1] = pre_pos_[(size_t) dst * 3 + 2] = pos;
            lim[(size_t) dst] = (int32_t) (dst + 1);
        }
        src = upto;
    };
    for (const Ref& r : refs) {
        text_rows(r.slot);
        const int64_t hw = r.h * r.w, ch = r.h - r.h / 2, cw = r.w - r.w / 2;   // the grid's centre, as the target's
        qi_latent_tokens16(r.latent, in_ch_, hw, h16_, s_);
        product(h16_, iin_, text_ + dst * C_, hw);
        for (int64_t y = 0; y < r.h; ++y)
            for (int64_t x = 0; x < r.w; ++x) {
                float* p = &pre_pos_[(size_t) (dst + y * r.w + x) * 3];
                p[0] = pos;
                p[1] = (float) (y - ch);
                p[2] = (float) (x - cw);
                lim[(size_t) (dst + y * r.w + x)] = (int32_t) (dst + hw);   // the whole image, and the prefix before it
            }
        pos += (float) std::max(r.h, r.w);
        ref_spans_.emplace_back(dst, dst + hw);
        dst += hw;
        src += r.slots;
    }
    text_rows(n_ctx);
    target_pos_ = (int64_t) pos;
    if (key_lim_ != nullptr) {
        allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), (void*) key_lim_), allocs_.end());
        gpu::free(key_lim_);
        key_lim_ = nullptr;
    }
    if (!refs.empty()) {   // without references a prefix row sees itself and before: the attention's own rule
        key_lim_ = alloc<int32_t>(allocs_, n_pre, "image engine");
        put(key_lim_, lim.data(), lim.size() * 4);
    }
    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        Block& b = blocks_[bi];
        for (uint16_t** p : {&b.pk, &b.pv}) {
            if (*p != nullptr) {
                allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), (void*) *p), allocs_.end());
                gpu::free(*p);
                *p = nullptr;
            }
            if ((int) bi < kv_vram_) {
                *p = alloc<uint16_t>(allocs_, n_pre * C_, "image engine");
            } else {   // pinned RAM: the step copies it in (the device reads it no slower than any copy)
                if (!gpu::alloc_host(p, (size_t) (n_pre * C_) * 2))
                    throw std::runtime_error(std::string("image engine: pinned RAM for the prefix cache: ") +
                                             gpu::last_error());
                allocs_.push_back(*p);
            }
        }
    }
    rope_H_ = -1;   // the positions changed
    prefix_cache_ = true;
    gpu::stream_sync(s_);
}

void QiDit::drop_host_prefix() {
    gpu::stream_sync(s_);
    std::vector<uint16_t**> ps;
    for (Block& b : blocks_) ps.insert(ps.end(), {&b.pk, &b.pv});
    for (uint16_t*& p : other_.pk) ps.push_back(&p);
    for (uint16_t*& p : other_.pv) ps.push_back(&p);
    for (uint16_t** p : ps)
        if (*p != nullptr && gpu::is_host_usm(*p)) {
            allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), (void*) *p), allocs_.end());
            gpu::free(*p);
            *p = nullptr;
        }
}

void QiDit::step_modulation(float t) {
    std::vector<float> e((size_t) 2 * 256);
    qi_time_sinusoid(t, e.data());
    qi_time_sinusoid(0.0f, e.data() + 256);
    std::vector<uint16_t> e16(e.size());
    for (size_t i = 0; i < e.size(); ++i) e16[i] = kernels::f16_from_f32(e[i]);
    put_async(small16_, e16.data(), e16.size() * 2, s_);
    gpu::stream_sync(s_);   // e16 is the host's: the copy is done before it goes
    product(small16_, t1_, temb_, 2);
    qi_silu(temb_, 2 * C_, small16_, s_);
    product(small16_, t2_, temb_, 2);
    qi_silu(temb_, 2 * C_, small16_, s_);   // the time after its SiLU: the modulation's and the final norm's input
    for (int k = 0; k < 4; ++k) product(small16_, mod_, mods_ + (int64_t) k * 2 * C_, 2, C_, (int64_t) k * C_, C_);
    product(small16_, nout_, scale_, 1);
}

void qi_time_sinusoid(float t, float* out) {
    constexpr int half = 128;
    for (int i = 0; i < half; ++i) {
        const float f = std::exp(-std::log(10000.0f) * (float) i / (float) half);
        out[i] = std::cos(t * f);
        out[half + i] = std::sin(t * f);
    }
}

void qi_rope_rows(const std::vector<float>& pos, int64_t T, std::vector<float>& c, std::vector<float>& s) {
    c.assign((size_t) T * kHead / 2, 0.f);
    s.assign(c.size(), 0.f);
    for (int64_t i = 0; i < T; ++i) {
        int j = 0;
        for (int a = 0; a < 3; ++a)
            for (int f = 0; f < kAxes[a] / 2; ++f, ++j) {
                const float omega = 1.0f / std::pow(10000.0f, (float) (2 * f) / (float) kAxes[a]);
                const float ang = pos[(size_t) i * 3 + a] * omega;
                c[(size_t) (i * kHead / 2 + j)] = std::cos(ang);
                s[(size_t) (i * kHead / 2 + j)] = std::sin(ang);
            }
    }
}

void QiDit::rope_tables(const std::vector<float>& pre_pos, int64_t n_text, int64_t target_pos, int64_t H, int64_t W,
                        float* cos, float* sin) {
    const int64_t T = n_text + H * W;
    std::vector<float> pos((size_t) T * 3);
    std::copy(pre_pos.begin(), pre_pos.end(), pos.begin());
    for (int64_t y = 0; y < H; ++y)
        for (int64_t xx = 0; xx < W; ++xx) {
            float* p = &pos[(size_t) (n_text + y * W + xx) * 3];
            p[0] = (float) target_pos;
            const int64_t hh = H - H / 2, wh = W - W / 2;   // the grid's centre, as the model was trained
            p[1] = (float) (y - hh);
            p[2] = (float) (xx - wh);
        }
    std::vector<float> c, s;
    qi_rope_rows(pos, T, c, s);
    // put is a copy of its own, not ordered after the stream: the work queued with the tables before (the other
    // context's forward, under guidance) is done first
    gpu::stream_sync(s_);
    put(cos, c.data(), c.size() * 4);
    put(sin, s.data(), s.size() * 4);
}

void QiDit::forward(const float* x, float t, int64_t H, int64_t W, bool prefix, float* out) {
    const int64_t HW = H * W, T = n_text_ + HW, C = C_;
    ensure(T, HW);
    if (rope_H_ != H || rope_W_ != W || rope_text_ != n_text_) {   // the 3-axis rotary tables [T][64]
        rope_tables(pre_pos_, n_text_, target_pos_, H, W, cos_, sin_);
        rope_H_ = H;
        rope_W_ = W;
        rope_text_ = n_text_;
    }
    step_modulation(t);
    if (t > last_t_) {   // a new image: the K means made again (the ControlNet branch's too)
        kmean_.reset();
        if (control_ != nullptr) control_->reset_means();
    }
    last_t_ = t;
    // a step after the first one of a context reads its prefix from the cache (set_loras drops it); a step with the
    // ControlNet branch on computes it again and leaves the cache as the steps without the branch made it
    const bool ctl = control_on(t);
    const bool prer = prefix && prefix_cache_ && !ctl;
    const int64_t r0 = prer ? n_text_ : 0, Tr = T - r0, ntr = n_text_ - r0;
    if (!prer) gpu::copy_async(x_, text_, (size_t) (n_text_ * C) * 4, s_);
    qi_latent_tokens16(x, in_ch_, HW, h16_, s_);
    product(h16_, iin_, x_ + n_text_ * C, HW);
    // qi_attention reads q16, k16 and v16 up to T rounded up, the rows past T zero: a longer context run before (the
    // other one of guidance) left its rows there
    if (dirty_T_ > T) {
        const int64_t pad = std::min(qi_attn_rows(T), dirty_T_) - T;
        for (uint16_t* b : {q16_, k16_, v16_}) gpu::memset_async(b + T * C, 0, (size_t) (pad * C) * 2, s_);
    }
    dirty_T_ = std::max(dirty_T_, T);
    auto mod = [&](int i) { return mods_ + (int64_t) i * 2 * C; };
    float* xr = x_ + r0 * C;
    const size_t text_bytes = (size_t) (n_text_ * C) * 2;
    if (ctl) control_->begin(T);
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        Block& b = blocks_[(size_t) bi];
        const uint8_t* base = blocks_stream_.begin(bi);   // the block's weights on the GPU (resident or a slot)
        // skip layer guidance's pass: the image's rows past the block as they came (the prefix's rows, cached, are
        // the block's own: the skip replaces the block of the image's rows only)
        if (prer && skip_ != nullptr && std::find(skip_->begin(), skip_->end(), bi) != skip_->end()) {
            blocks_stream_.end(bi);
            continue;
        }
        auto at = [base](const QiDit::W& w) {
            QiDit::W r = w;
            r.d = base + reinterpret_cast<uintptr_t>(w.d);
            return r;
        };
        const QiDit::W wq = at(b.q), wk = at(b.k), wv = at(b.v), wo = at(b.o), wgu = at(b.gu), wdown = at(b.down);
        const auto* nq = reinterpret_cast<const float*>(base + b.nq);
        const auto* nk = reinterpret_cast<const float*>(base + b.nk);
        qi_ln_mod16(xr, mod(0), ntr, Tr, C, h16_, s_);
        // W8A8: the sums scaled as qi_qk_norm_rope16 reads them (an adapter's delta is FP32: those products are not
        // deferred, and the sums are scaled here)
        QiI8 qkv[3];
        const bool i8 = wq.w8 && wk.w8 && wv.w8 && wq.lora == nullptr && wk.lora == nullptr && wv.lora == nullptr;
        product(h16_, wq, q_ + r0 * C, Tr, 0, 0, -1, false, i8);
        qkv[0] = i8_;
        product(h16_, wk, k_ + r0 * C, Tr, 0, 0, -1, wq.w8, i8);
        qkv[1] = i8_;
        product(h16_, wv, v_ + r0 * C, Tr, 0, 0, -1, wk.w8, i8);
        qkv[2] = i8_;
        qi_qk_norm_rope16(q_ + r0 * C, k_ + r0 * C, v_ + r0 * C, nq, nk, cos_ + r0 * kHead / 2,
                          sin_ + r0 * kHead / 2, Tr, heads_, kHead, q16_ + r0 * C, k16_ + r0 * C, v16_ + r0 * C, s_,
                          i8 ? qkv : nullptr);
        if (n_text_ > 0) {
            if (prer) {
                gpu::copy_async(k16_, b.pk, text_bytes, s_);
                gpu::copy_async(v16_, b.pv, text_bytes, s_);
            } else if (!ctl) {
                gpu::copy_async(b.pk, k16_, text_bytes, s_);
                gpu::copy_async(b.pv, v16_, text_bytes, s_);
            }
        }
        // the queries from r0 on: a cached prefix's rows are not asked again (an edit's references: most of them)
        const auto st = kmean_.stats((int64_t) bi * 2 + ctx_i_);
        qi_attention(q16_, k16_, v16_, T, heads_, kHead, n_text_, o16_ + r0 * C, s_, key_lim_, r0, true,
                     key_lim_ != nullptr ? &ref_spans_ : nullptr, &st);
        kmean_.mark((int64_t) bi * 2 + ctx_i_);
        const bool od = product(o16_ + r0 * C, wo, h_, Tr, 0, 0, -1, false, wo.lora == nullptr);
        qi_gate_add(xr, h_, mod(1), ntr, Tr, C, s_, od ? &i8_ : nullptr);
        for (int64_t m0 = 0; m0 < Tr; m0 += mlp_rows_) {   // the MLP a band of rows at a time (its 2 x inter wide product)
            const int64_t mr = std::min(mlp_rows_, Tr - m0), mt = std::max<int64_t>(0, std::min(ntr - m0, mr));
            qi_ln_mod16(xr + m0 * C, mod(2), mt, mr, C, h16_, s_);
            mlp(wgu, wdown, xr + m0 * C, mod(3), mt, mr);
        }
        if (ctl) control_->after_block(bi, T);
        blocks_stream_.end(bi);
    }
    qi_ln_mod16(x_ + n_text_ * C, scale_, 0, HW, C, h16_, s_);   // the final norm: row 0 of scale_ for every token
    product(h16_, pout_, h_, HW);
    qi_tokens_latent(h_, in_ch_, HW, out, s_);
}

bool QiDit::control_on(float t) const { return control_ != nullptr && control_->active(t * 0.001f); }

void QiDit::forward_pair(const float* x, float t, int64_t H, int64_t W, float* out, float* out2) {
    const int64_t HW = H * W, C = C_, na = n_text_, nb = other_.n_text, Ta = na + HW, Tb = nb + HW;
    const int64_t TPa = qi_attn_rows(Ta), TPb = qi_attn_rows(Tb), R = 2 * HW;
    ensure(TPa + TPb, HW);   // the rows: both images' (2 HW), and q16, k16, v16 for the two sequences side by side
    if (rope_H_ != H || rope_W_ != W || rope_text_ != na) {
        rope_tables(pre_pos_, na, target_pos_, H, W, cos_, sin_);
        rope_H_ = H;
        rope_W_ = W;
        rope_text_ = na;
    }
    Context& o = other_;
    if (o.rope_H != H || o.rope_W != W || o.rope_text != nb) {   // the other context's own tables
        if (o.rope_cap < Tb) {
            for (float** p : {&o.cos, &o.sin}) {
                if (*p != nullptr) {
                    allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), (void*) *p), allocs_.end());
                    gpu::free(*p);
                }
                *p = alloc<float>(allocs_, Tb * kHead / 2, "image engine");
            }
            o.rope_cap = Tb;
        }
        rope_tables(o.pre_pos, nb, o.target_pos, H, W, o.cos, o.sin);
        o.rope_H = H;
        o.rope_W = W;
        o.rope_text = nb;
    }
    step_modulation(t);
    if (t > last_t_) {   // a new image: the K means made again (the ControlNet branch's too)
        kmean_.reset();
        if (control_ != nullptr) control_->reset_means();
    }
    last_t_ = t;
    // the image's rows twice, one copy a context: [0, HW) the first's, [HW, 2 HW) the other's
    qi_latent_tokens16(x, in_ch_, HW, h16_, s_);
    product(h16_, iin_, x_, HW);
    gpu::copy_async(x_ + HW * C, x_, (size_t) (HW * C) * 4, s_);
    uint16_t *q16b = q16_ + TPa * C, *k16b = k16_ + TPa * C, *v16b = v16_ + TPa * C;
    for (const auto& [base, T, TP] : {std::tuple<int64_t, int64_t, int64_t>{0, Ta, TPa}, {TPa, Tb, TPb}})
        if (TP > T)   // the rows past each sequence zero (qi_attention reads up to T rounded up)
            for (uint16_t* b : {q16_, k16_, v16_})
                gpu::memset_async(b + (base + T) * C, 0, (size_t) ((TP - T) * C) * 2, s_);
    dirty_T_ = std::max(dirty_T_, TPa + TPb);
    auto mod = [&](int i) { return mods_ + (int64_t) i * 2 * C; };
    const size_t bytes_a = (size_t) (na * C) * 2, bytes_b = (size_t) (nb * C) * 2;
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        Block& b = blocks_[(size_t) bi];
        const uint8_t* base = blocks_stream_.begin(bi);   // the block's weights, read once for both contexts
        auto at = [base](const QiDit::W& w) {
            QiDit::W r = w;
            r.d = base + reinterpret_cast<uintptr_t>(w.d);
            return r;
        };
        const QiDit::W wq = at(b.q), wk = at(b.k), wv = at(b.v), wo = at(b.o), wgu = at(b.gu), wdown = at(b.down);
        const auto* nq = reinterpret_cast<const float*>(base + b.nq);
        const auto* nk = reinterpret_cast<const float*>(base + b.nk);
        qi_ln_mod16(x_, mod(0), 0, R, C, h16_, s_);
        // W8A8: the sums scaled as qi_qk_norm_rope16 reads them (the second context's rows from HW on)
        QiI8 qkv[3], qkv2[3];
        const bool i8 = wq.w8 && wk.w8 && wv.w8 && wq.lora == nullptr && wk.lora == nullptr && wv.lora == nullptr;
        product(h16_, wq, q_, R, 0, 0, -1, false, i8);
        qkv[0] = i8_;
        product(h16_, wk, k_, R, 0, 0, -1, wq.w8, i8);
        qkv[1] = i8_;
        product(h16_, wv, v_, R, 0, 0, -1, wk.w8, i8);
        qkv[2] = i8_;
        for (int j = 0; j < 3; ++j) qkv2[j] = {qkv[j].sx + HW, qkv[j].sw};
        qi_qk_norm_rope16(q_, k_, v_, nq, nk, cos_ + na * kHead / 2, sin_ + na * kHead / 2, HW, heads_, kHead,
                          q16_ + na * C, k16_ + na * C, v16_ + na * C, s_, i8 ? qkv : nullptr);
        qi_qk_norm_rope16(q_ + HW * C, k_ + HW * C, v_ + HW * C, nq, nk, o.cos + nb * kHead / 2,
                          o.sin + nb * kHead / 2, HW, heads_, kHead, q16b + nb * C, k16b + nb * C, v16b + nb * C, s_,
                          i8 ? qkv2 : nullptr);
        if (na > 0) {   // the prefixes' K and V from their caches
            gpu::copy_async(k16_, b.pk, bytes_a, s_);
            gpu::copy_async(v16_, b.pv, bytes_a, s_);
        }
        if (nb > 0) {
            gpu::copy_async(k16b, o.pk[(size_t) bi], bytes_b, s_);
            gpu::copy_async(v16b, o.pv[(size_t) bi], bytes_b, s_);
        }
        const auto sta = kmean_.stats((int64_t) bi * 2);
        qi_attention(q16_, k16_, v16_, Ta, heads_, kHead, na, o16_, s_, key_lim_, na, true, nullptr, &sta);
        kmean_.mark((int64_t) bi * 2);
        const auto stb = kmean_.stats((int64_t) bi * 2 + 1);
        qi_attention(q16b, k16b, v16b, Tb, heads_, kHead, nb, o16_ + HW * C, s_, o.key_lim, nb, true, nullptr, &stb);
        kmean_.mark((int64_t) bi * 2 + 1);
        const bool od = product(o16_, wo, h_, R, 0, 0, -1, false, wo.lora == nullptr);
        qi_gate_add(x_, h_, mod(1), 0, R, C, s_, od ? &i8_ : nullptr);
        for (int64_t m0 = 0; m0 < R; m0 += mlp_rows_) {   // the MLP a band of rows at a time
            const int64_t mr = std::min(mlp_rows_, R - m0);
            qi_ln_mod16(x_ + m0 * C, mod(2), 0, mr, C, h16_, s_);
            mlp(wgu, wdown, x_ + m0 * C, mod(3), 0, mr);
        }
        blocks_stream_.end(bi);
    }
    qi_ln_mod16(x_, scale_, 0, R, C, h16_, s_);   // the final norm: row 0 of scale_ for every token
    product(h16_, pout_, h_, R);
    qi_tokens_latent(h_, in_ch_, HW, out, s_);
    qi_tokens_latent(h_ + HW * in_ch_, in_ch_, HW, out2, s_);
}

uint64_t QiDit::activation_bytes(int64_t T, int64_t mlp_rows) const {
    const int64_t TP = qi_attn_rows(T), mr = std::min(T, mlp_rows > 0 ? mlp_rows : T);
    return (uint64_t) T * (uint64_t) C_ * 4 * 4 + (uint64_t) T * (uint64_t) std::max(C_, in_ch_) * 4 +
           (uint64_t) mr * 2 * (uint64_t) inter_ * 4 + (uint64_t) T * (uint64_t) std::max(C_, inter_) * 2 +
           (uint64_t) TP * (uint64_t) C_ * 2 * 3 + (uint64_t) T * (uint64_t) C_ * 2 + (uint64_t) T * kHead * 4 +
           (w8_ ? (uint64_t) std::max(T * C_, mr * inter_) + (uint64_t) T * 8 : 0) +
           (lora_cols_ > 0 ? (uint64_t) std::max(T, lora_tmp_t_) * (uint64_t) lora_cols_ * 6 : 0);
}

void QiDit::set_mlp_rows(int64_t rows) {
    if (rows != mlp_cap_) {
        mlp_cap_ = rows;
        drop_activations();
    }
}

std::vector<QiDit::LoraTarget> QiDit::lora_targets() const {
    std::vector<LoraTarget> out;
    out.reserve(lora_slots_.size());
    for (const LoraSlot& s : lora_slots_) out.push_back({s.name, s.row0, s.rows, s.w->k});
    return out;
}

void QiDit::build_lora_targets() {
    lora_slots_.clear();
    auto add = [this](std::string name, W& w, int64_t row0, int64_t rows) {
        if (w.n <= 0 || w.k <= 0 || rows <= 0 || row0 + rows > w.n) return;
        lora_slots_.push_back({std::move(name), &w, row0, rows});
    };
    // the weights outside the blocks (the text projection's norm is a norm, not a linear product: unchanged)
    const char* const* out = lora::kQiOutside;
    add(out[0], iin_, 0, iin_.n);
    add(out[1], t1_, 0, t1_.n);
    add(out[2], t2_, 0, t2_.n);
    add(out[3], mod_, 0, mod_.n);
    add(out[4], nout_, 0, nout_.n);
    add(out[5], pout_, 0, pout_.n);
    add(out[6], tin_, 0, tin_.n);
    add(out[7], tout_, 0, tout_.n);
    const char* const* n = lora::kQiBlock;
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        const std::string p = "transformer_blocks." + std::to_string(bi) + ".";
        Block& b = blocks_[(size_t) bi];
        add(p + n[0], b.q, 0, b.q.n);
        add(p + n[1], b.k, 0, b.k.n);
        add(p + n[2], b.v, 0, b.v.n);
        add(p + n[3], b.o, 0, b.o.n);
        // the MLP's first weight is one [gate | up] on the GPU: a file may hold it fused, or as diffusers' two halves
        add(p + n[4], b.gu, 0, b.gu.n);
        add(p + lora::kQiGateHalf[0], b.gu, 0, inter_);
        add(p + lora::kQiGateHalf[1], b.gu, inter_, b.gu.n - inter_);
        add(p + n[5], b.down, 0, b.down.n);
    }
}

void QiDit::lora_scratch(int64_t T) {
    if (lora_cols_ <= 0 || T <= 0) return;
    // a GLoRA's own path's buffers are a band's, not the tokens': the activations' drop frees them, this gives them back
    if (lora_glora_cols_ > 0 && lora_glora_y_ == nullptr) lora_glora_scratch();
    if (lora_f32_ != nullptr && T <= lora_tmp_t_) return;
    gpu::stream_sync(s_);   // the old buffers may still be read
    for (void** p : {reinterpret_cast<void**>(&lora_f32_), reinterpret_cast<void**>(&lora16_)})
        if (*p != nullptr) {
            allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), *p), allocs_.end());
            gpu::free(*p);
            *p = nullptr;
        }
    lora_f32_ = alloc<float>(allocs_, T * lora_cols_, "image engine");
    lora16_ = alloc<uint16_t>(allocs_, T * lora_cols_, "image engine");
    lora_tmp_t_ = T;
}

// A GLoRA's own path's buffers: the table the weight's product of its input is added from (a band of tokens, the
// weight's rows), and that input as a W8A8 product takes it (int8 values and a scale a row).  The scratch of the
// deltas (lora_scratch) holds the input's own two products.
void QiDit::lora_glora_scratch() {
    if (lora_glora_cols_ <= 0 || lora_glora_rows_ <= 0) return;
    const size_t band = (size_t) kLoraGloraRows;
    lora_glora_y_ = alloc<float>(allocs_, (int64_t) (band * (size_t) lora_glora_rows_), "image engine");
    lora_glora_x8_ = alloc<int8_t>(allocs_, (int64_t) (band * (size_t) lora_glora_cols_), "image engine");
    lora_glora_sx_ = alloc<float>(allocs_, (int64_t) band, "image engine");
}

void QiDit::glora_weight(const uint16_t* v, const W& w, int64_t r0, int64_t n, int64_t nt, int64_t t0, float* y,
                         int64_t ldy, int64_t row0) {
    const auto* d = static_cast<const uint8_t*>(w.d);
    if (w.w8 && !w.w8t) {   // a w8t weight's blocks take the native path below
        // the input's own int8 quantization (its buffer, not the product's: that one's is the other products' input)
        kernels::rows_i8(v, false, nt, w.k, lora_glora_x8_, lora_glora_sx_, nullptr, s_);
        const auto* sw = reinterpret_cast<const float*>(d + w.n * w.k) + r0;
        prefill::Gemm::w8a8(lora_glora_x8_, lora_glora_sx_, reinterpret_cast<const int8_t*>(d) + r0 * w.k, sw, nullptr,
                            lora_glora_y_, nt, n, w.k, n, false, s_);
        kernels::i8_scale(lora_glora_y_, nt, n, n, lora_glora_sx_, sw, nullptr, nullptr, s_);
    } else if (w.type == kTypeF16) {
        gemm_.f16(v, reinterpret_cast<const uint16_t*>(d + (uint64_t) r0 * w.row_bytes), lora_glora_y_, nt, n, w.k, n);
    } else {
        gemm_.native(v, w.type, d + (uint64_t) r0 * w.row_bytes, lora_glora_y_, nt, n, w.k, n);
    }
    // the product's rows are the weight's [r0, r0 + n), written to y's rows [row0, row0 + rows): the rows' place in it
    lora_output_add(y + t0 * ldy, nt, ldy, r0 - row0, lora_glora_y_, n, s_);
}

void QiDit::free_glora_scratch() {
    for (void* p : {(void*) lora_glora_y_, (void*) lora_glora_x8_, (void*) lora_glora_sx_})
        if (p != nullptr) {
            gpu::stream_sync(s_);
            allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), p), allocs_.end());
            gpu::free(p);
        }
    lora_glora_y_ = nullptr;
    lora_glora_x8_ = nullptr;
    lora_glora_sx_ = nullptr;
}

void QiDit::drop_prefix() {
    const auto free_dev = [this](void* p) {
        if (p == nullptr) return;
        allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), p), allocs_.end());
        gpu::free(p);
    };
    if (s_ != nullptr) gpu::stream_sync(s_);
    for (Block& b : blocks_)
        for (uint16_t** p : {&b.pk, &b.pv}) {
            free_dev(*p);
            *p = nullptr;
        }
    for (uint16_t*& p : other_.pk) {
        free_dev(p);
        p = nullptr;
    }
    for (uint16_t*& p : other_.pv) {
        free_dev(p);
        p = nullptr;
    }
    for (void** p : {reinterpret_cast<void**>(&text_), reinterpret_cast<void**>(&other_.text),
                     reinterpret_cast<void**>(&key_lim_), reinterpret_cast<void**>(&other_.key_lim)}) {
        free_dev(*p);
        *p = nullptr;
    }
    ref_spans_.clear();
    other_.ref_spans.clear();
    pre_pos_.clear();
    other_.pre_pos.clear();
    n_text_ = 0;
    other_.n_text = 0;
    prefix_cache_ = false;
    rope_H_ = rope_W_ = rope_text_ = -1;
    other_.rope_H = -1;
}

bool QiDit::set_loras(const std::vector<LoraUse>& loras, std::string& err) {
    // the set as the same again (its files and strengths): nothing to bind
    std::string key;
    for (const LoraUse& u : loras) {
        if (!u.file) continue;
        char num[32];
        std::snprintf(num, sizeof num, "%.7g", (double) u.scale);
        key += u.file->path();
        key += ':';
        key += num;
        key += ';';
    }
    if (key == lora_key_) return true;
    drop_loras();
    if (key.empty()) return true;
    std::vector<std::string> unknown, text_encoder;
    for (const LoraUse& u : loras) {
        if (!u.file) continue;
        for (const LoraEntry& entry : u.file->entries()) {
            if (is_text_encoder_key(entry.key)) {
                text_encoder.push_back(entry.key);
                continue;
            }
            const LoraSlot* slot = nullptr;
            for (const LoraSlot& s : lora_slots_)
                if (lora_key_names(entry.key, s.name)) {
                    slot = &s;
                    break;
                }
            if (slot == nullptr) {
                unknown.push_back(entry.key);
                continue;
            }
            // a GLoRA's four tensors may fit both of its readings: the weight's rows and columns say which
            LoraEntry alt;
            const LoraEntry& e = *lora_glora_for_weight(entry, slot->rows, slot->w->k, alt);
            if (e.kind == LoraKind::bias || !e.bias.empty()) {
                // no place for it: the DiT's products add no bias (its blocks' modulation does the shift), so a file's
                // bias's delta would change nothing while it was asked for
                err = "image engine: " + u.file->path() + ": " + e.key +
                      ": a bias's delta (diff_b/b_norm), and the DiT's weights add no bias (Qwen-Image-2.1's DiT has "
                      "none)";
                drop_loras();
                return false;
            }
            const bool oft = e.kind == LoraKind::oft;
            const bool glora = e.kind == LoraKind::glora;
            if (e.rows != slot->rows || (oft ? e.cols != 0 : e.cols != slot->w->k)) {
                err = "image engine: " + u.file->path() + ": " + e.key + " is " + std::to_string(e.cols) + " x " +
                      std::to_string(e.rows) + ", not the DiT's " + std::to_string(slot->w->k) + " x " +
                      std::to_string(slot->rows);
                drop_loras();
                return false;
            }
            if (oft && slot->name == "modulation.1") {
                // the DiT writes that weight a band of C_ rows at a time (step_modulation), and a rotation turns the
                // part's rows whole (LoraLayer::add): bound, it would be skipped on every band
                err = "image engine: " + u.file->path() + ": " + e.key + ": an OFT of the modulation, which the DiT "
                      "writes " + std::to_string(C_) + " rows at a time: the engine does not turn it";
                drop_loras();
                return false;
            }
            const float scale = u.scale * e.scale();
            // a strength of 0 changes nothing: the weight keeps its product's path (an alpha of 0 leaves a DoRA's base
            // factors, as a DoRA's decomposition scales the weight with no delta)
            if (u.scale == 0.0f || (scale == 0.0f && e.dora.empty())) continue;
            std::vector<float> coef, base;   // a DoRA's: the delta's rows' c, the base product's factors
            if (!e.dora.empty()) {
                std::vector<float> norms;
                if (!dora_norms(*slot, norms, err)) {
                    drop_loras();
                    return false;
                }
                std::string why;
                if (!lora_dora_rows(e, 0, slot->row0, slot->rows, norms, u.scale, coef, base, why)) {
                    err = "image engine: ";
                    err += u.file->path();
                    err += ": ";
                    err += e.key;
                    err += ": ";
                    err += why;
                    drop_loras();
                    return false;
                }
            }
            if (const LoraPart* clash = lora_dora_overlap(slot->w->lora, slot->row0, slot->rows, !coef.empty(), oft,
                                                          glora, e.replaces)) {
                err = "image engine: " + u.file->path() + ": " + e.key + ": the weight's rows " +
                      std::to_string(slot->row0) + " .. " + std::to_string(slot->row0 + slot->rows) +
                      " are another adapter's, and " + lora_patch_clash_why(*clash, oft, glora, e.replaces);
                drop_loras();
                return false;
            }
            if (e.replaces)   // a set_weight: the weight interpolates to the file's, W + s (S - W)
                base.assign((size_t) slot->rows, 1.0f - u.scale);
            auto dev = std::make_shared<LoraDeviceEntry>();
            if (!dev->upload(e, s_, err, coef.empty() ? nullptr : coef.data())) {
                drop_loras();
                return false;
            }
            if (slot->w->lora == nullptr) {
                lora_layers_.push_back(std::make_unique<LoraLayer>());
                slot->w->lora = lora_layers_.back().get();
            }
            LoraPart part;
            part.entry = dev;
            part.row0 = slot->row0;
            part.n = slot->rows;
            part.scale = scale;
            part.dora = !coef.empty();
            part.oft = oft;
            part.glora = glora;
            part.replaces = e.replaces;
            part.base = std::move(base);
            slot->w->lora->parts.push_back(std::move(part));
            lora_cols_ = std::max(lora_cols_, lora_scratch_cols(e));
            if (glora) {
                lora_glora_cols_ = std::max(lora_glora_cols_, e.cols);
                lora_glora_rows_ = std::max(lora_glora_rows_, e.rows);
            }
        }
    }
    if (!text_encoder.empty()) {
        err = "image engine: a text encoder's adapter: the image engine applies adapters to the DiT only (" +
              name_list(text_encoder) + ")";
        drop_loras();
        return false;
    }
    if (!unknown.empty()) {
        err = "image engine: the DiT takes no adapter for " + name_list(unknown);
        drop_loras();
        return false;
    }
    if (lora_layers_.empty()) {   // every strength 0: as no adapter at all (the products are untouched)
        drop_loras();
        return true;
    }
    for (const LoraSlot& slot : lora_slots_)   // the base products' row factors, once every part is bound
        if (slot.w->lora != nullptr && !slot.w->lora->build_base(slot.w->n, s_, err)) {
            drop_loras();
            return false;
        }
    if (lora_glora_cols_ > 0) lora_glora_scratch();   // a GLoRA's own path's buffers, once every part is bound
    if (lora_bake_) {   // the blocks' plain LoRAs folded into their weights as the blocks come back to the GPU
        int64_t nk = 0, kr = 0;
        for (Block& b : blocks_)
            for (W* w : {&b.q, &b.k, &b.v, &b.o, &b.gu, &b.down})
                if (w->lora != nullptr && w->lora->bakeable() &&
                    (w->w8 && !w->w8t ? -8 : w->type == 1 ? 1 : w->type == 8 ? 8 : 0) != 0) {
                    w->baked = true;
                    nk = std::max(nk, w->n * w->k);
                    for (const LoraPart& p : w->lora->parts) kr = std::max(kr, w->k * p.entry->rank());
                }
        if (nk > 0) {
            bake_tmp_ = alloc<float>(allocs_, nk, "image engine");
            bake_at16_ = alloc<uint16_t>(allocs_, kr, "image engine");
            baked_any_ = true;
            blocks_stream_.reload();   // the queued work drained before the prepare changes
            install_prepare();
        }
    }
    for (const LoraUse& u : loras)
        if (u.file) lora_files_.push_back(u.file);
    lora_key_ = key;
    drop_prefix();   // the prefix's rows and K and V were the other adapters'
    return true;
}

bool QiDit::tensor_norms(const std::string& key, const std::vector<float>*& out, std::string& err) {
    auto it = norm_cache_.find(key);
    if (it != norm_cache_.end()) {
        out = &it->second;
        return true;
    }
    const TensorInfo* t = g_->gguf().find(key);
    if (t == nullptr || t->shape.size() != 2) {
        err = "image engine: the DiT has no " + key;
        return false;
    }
    const int64_t k = (int64_t) t->shape[0], rows = (int64_t) t->shape[1];
    const uint8_t* p = g_->data(*t);
    std::vector<float> norms((size_t) rows);
    if (t->type == kTypeF32) {
        const auto* f = reinterpret_cast<const float*>(p);
        for (int64_t r = 0; r < rows; ++r) {
            double s = 0;
            for (int64_t c = 0; c < k; ++c) s += (double) f[r * k + c] * (double) f[r * k + c];
            norms[(size_t) r] = (float) std::sqrt(s);
        }
    } else if (t->type == kTypeF16 || t->type == kTypeBF16) {
        const auto* h = reinterpret_cast<const uint16_t*>(p);
        for (int64_t r = 0; r < rows; ++r) {
            double s = 0;
            for (int64_t c = 0; c < k; ++c) {
                const float v = t->type == kTypeF16 ? kernels::f32_from_f16(h[r * k + c])
                                                   : kernels::f32_from_bf16(h[r * k + c]);
                s += (double) v * (double) v;
            }
            norms[(size_t) r] = (float) std::sqrt(s);
        }
    } else if (t->type == 8) {   // Q8_0: 34 bytes a 32 values, an FP16 scale and then the int8 ones
        const int64_t nb = k / 32;
        for (int64_t r = 0; r < rows; ++r) {
            const uint8_t* row = p + (size_t) (r * nb) * 34;
            double s = 0;
            for (int64_t b = 0; b < nb; ++b) {
                const uint8_t* q = row + b * 34;
                const float d = kernels::f32_from_f16((uint16_t) ((uint16_t) q[0] | ((uint16_t) q[1] << 8)));
                for (int j = 0; j < 32; ++j) {
                    const double v = (double) d * (int8_t) q[2 + j];
                    s += v * v;
                }
            }
            norms[(size_t) r] = (float) std::sqrt(s);
        }
    } else if (kernels::dequant_bf16_supported((int) t->type) && gemm_.scratch() != nullptr &&
               gemm_.scratch_elems() >= 2 * k) {
        // the other block formats (Q6_K, Q4_K, the i-quants): the GPU's dequantization (the one the products use) to
        // FP32 in the products' scratch, a band of rows at a time (the scratch holds the largest weight in FP16, so
        // half its rows in FP32), the rows' sums on the host
        const uint64_t bytes = tensor_payload_bytes(*t);
        uint8_t* blocks = nullptr;
        if (!gpu::alloc_device(&blocks, (size_t) bytes)) {
            err = "image engine: " + key + ": its blocks on the GPU for a DoRA's row norms: " + gpu::last_error();
            return false;
        }
        const int64_t band = std::min<int64_t>(rows, gemm_.scratch_elems() / (2 * k));
        auto* f = reinterpret_cast<float*>(gemm_.scratch());
        std::vector<float> h((size_t) (band * k));
        bool ok = gpu::copy_async(blocks, p, (size_t) bytes, s_);
        for (int64_t r0 = 0; ok && r0 < rows; r0 += band) {
            const int64_t nr = std::min(band, rows - r0);
            kernels::dequant_f32((int) t->type, blocks, r0, nr, k, f, s_);
            ok = gpu::copy_async(h.data(), f, (size_t) (nr * k) * 4, s_) && gpu::stream_sync(s_);
            for (int64_t r = 0; ok && r < nr; ++r) {
                double s = 0;
                const float* row = h.data() + r * k;
                for (int64_t c = 0; c < k; ++c) s += (double) row[c] * (double) row[c];
                norms[(size_t) (r0 + r)] = (float) std::sqrt(s);
            }
        }
        gpu::stream_sync(s_);
        gpu::free(blocks);
        if (!ok) {
            err = "image engine: " + key + ": a DoRA's row norms on the GPU: " + gpu::last_error();
            return false;
        }
    } else {
        err = "image engine: " + key + ": a " + t->type_name() + " tensor: a DoRA's row norms are read from the "
              "file's F32, F16, BF16 and Q8_0 rows, and the GPU's dequantization does not take this format";
        return false;
    }
    out = &norm_cache_.emplace(key, std::move(norms)).first->second;
    return true;
}

bool QiDit::dora_norms(const LoraSlot& slot, std::vector<float>& out, std::string& err) {
    // the file's name of the weight: a fused weight takes diffusers' two names for its halves
    std::string name = slot.name;
    const auto ends = [&name](const char* suffix) {
        const size_t n = std::strlen(suffix);
        return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
    };
    if (ends(".img_mlp.gate_layer")) {   // diffusers' two names for the fused weight's halves
        name.resize(name.size() - 10);
        name += "gate_up";
    } else if (ends(".img_mlp.proj")) {
        name.resize(name.size() - 4);
        name += "gate_up";
    }
    const std::vector<float>* norms = nullptr;
    if (!tensor_norms("model.diffusion_model." + name + ".weight", norms, err)) return false;
    if ((int64_t) norms->size() < slot.w->n) {
        err = "image engine: " + slot.name + ": the file's rows are fewer than the weight's";
        return false;
    }
    out.assign(norms->begin(), norms->begin() + slot.w->n);
    return true;
}

uint64_t QiDit::lora_bytes() const {
    uint64_t n = 0;
    for (const std::unique_ptr<LoraLayer>& l : lora_layers_) n += l->bytes();
    if (lora_glora_y_ != nullptr)   // a GLoRA's own path's buffers
        n += (uint64_t) kLoraGloraRows * ((uint64_t) lora_glora_rows_ * 4 + (uint64_t) lora_glora_cols_ + 4);
    return n;
}

void QiDit::drop_loras() {
    const bool had = !lora_key_.empty();
    for (LoraSlot& s : lora_slots_) {
        s.w->lora = nullptr;
        s.w->baked = false;
    }
    if (baked_any_) {   // the weights as the file has them again: the blocks copied anew, without the bake
        baked_any_ = false;
        blocks_stream_.reload();
        install_prepare();
        for (void** p : {reinterpret_cast<void**>(&bake_tmp_), reinterpret_cast<void**>(&bake_at16_)})
            if (*p != nullptr) {
                release(allocs_, *p);
                *p = nullptr;
            }
    }
    lora_layers_.clear();
    lora_files_.clear();
    lora_key_.clear();
    lora_cols_ = 0;
    lora_glora_cols_ = lora_glora_rows_ = 0;
    if (lora_f32_ != nullptr || lora16_ != nullptr) {
        gpu::stream_sync(s_);
        for (void** p : {reinterpret_cast<void**>(&lora_f32_), reinterpret_cast<void**>(&lora16_)})
            if (*p != nullptr) {
                allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), *p), allocs_.end());
                gpu::free(*p);
                *p = nullptr;
            }
        lora_tmp_t_ = 0;
    }
    free_glora_scratch();
    if (had) drop_prefix();
}

void QiDit::drop_activations() {
    for (void* p : {(void*) x_, (void*) q_, (void*) k_, (void*) v_, (void*) h_, (void*) gu_, (void*) h16_, (void*) q16_,
                    (void*) k16_, (void*) v16_, (void*) o16_, (void*) cos_, (void*) sin_, xq_,
                    (void*) xs_, (void*) lora_f32_, (void*) lora16_, (void*) lora_glora_y_, (void*) lora_glora_x8_,
                    (void*) lora_glora_sx_, (void*) kmean_.buf})
        if (p != nullptr) {
            allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), p), allocs_.end());
            gpu::free(p);
        }
    x_ = q_ = k_ = v_ = h_ = gu_ = cos_ = sin_ = nullptr;
    xq_ = nullptr;
    xq_bytes_ = 0;
    xs_ = nullptr;
    xs_rows_ = 0;
    h16_ = q16_ = k16_ = v16_ = o16_ = nullptr;
    lora_f32_ = nullptr;
    lora16_ = nullptr;
    lora_tmp_t_ = 0;
    lora_glora_y_ = nullptr;
    lora_glora_x8_ = nullptr;
    lora_glora_sx_ = nullptr;
    cap_T_ = 0;
    kmean_.clear();
    rope_H_ = rope_W_ = rope_text_ = -1;
}

void QiDit::sample(float* x, int64_t H, int64_t W, const std::vector<float>& sigmas,
                   const std::function<bool(int, int)>& step, const std::function<void(float*, float)>& after,
                   float cfg, const std::function<void(int, const std::function<void(float*)>&)>& peek,
                   const GuidanceSpec* guide, bool er, float shift, uint64_t seed) {
    const GuidanceSpec spec = guide != nullptr ? *guide : GuidanceSpec{};
    for (const int b : spec.slg_layers)
        if (b < 0 || b >= (int) blocks_.size())
            throw std::invalid_argument("image engine: skip layer guidance's block " + std::to_string(b) +
                                        " is not one of the model's " + std::to_string(blocks_.size()));
    if (control_ != nullptr && control_->set() && !spec.slg_layers.empty())
        throw std::invalid_argument("image engine: skip layer guidance with ControlNet is not supported (its pass skips "
                                    "blocks the branch adds to)");
    ctx_i_ = 0;   // the first context's K means slot (guidance's swap_context turns it)
    last_t_ = -1.f;
    const int64_t N = in_ch_ * H * W;
    const int n = cfg > 1.0f && other_.n_text > 0 ? 2 : 1;
    Guidance g(spec, s_, N, in_ch_, false, n == 2 ? cfg : 1.0f, sigmas);
    bool cached = false;
    auto forward_all = [&](const float* xx, float sigma, int nn, float* out) {
        const float t = sigma * 1000.0f;
        const bool ctl = control_on(t);
        if (nn == 2 && cached && !ctl) {
            forward_pair(xx, t, H, W, out, out + N);
        } else {
            forward(xx, t, H, W, cached, out);
            if (nn == 2) {
                swap_context();
                try { forward(xx, t, H, W, false, out + N); }
                catch (...) { swap_context(); throw; }
                swap_context();
            }
        }
        cached = cached || !ctl;
    };
    auto skip = [&](const float* xx, float sigma, float* out) {
        skip_ = &g.skip_layers();
        try { forward(xx, sigma * 1000.0f, H, W, true, out); }
        catch (...) { skip_ = nullptr; throw; }
        skip_ = nullptr;
    };
    auto progress = [&](int done, int total) {
        if (!step) return true;
        gpu::stream_sync(s_);
        return step(done, total);
    };
    flow_sample(s_, forward_all, x, N, n, sigmas, cfg, er, shift, seed, progress, after, peek, &g, skip, true, nullptr, true);
}

}  // namespace strata::image
