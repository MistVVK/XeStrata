// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_dit.cpp - see include/strata/image/mf_dit.hpp.

#include "strata/image/mf_dit.hpp"
#include "strata/image/flow_sampler.hpp"
#include "strata/image/dit_loras.hpp"
#include "strata/image/lora_names.hpp"

#include "dit_rows.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/platform/mapped_file.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/platform/env.hpp"

#include "strata/image/gpu_buf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <thread>

namespace strata::image {

namespace gpu = strata::gpu;

// The kernels' helpers in a namespace of this file's own, not an anonymous one: SYCL names a lambda's kernel by its
// mangled type, and an anonymous namespace mangles the same in every file (_GLOBAL__N_1), so a helper of the same
// name and signature in another image file (qi_vae.cpp's to_rgba) gave its kernel to both.
namespace mf_dit_detail {

using namespace dit_rows;

// kTypeW8: int8 rows rotated as kernels::rows_i8 rotates them, then a scale a row (FP32), for prefill::Gemm::i8
constexpr int kTypeF16 = 1, kTypeQ8_0 = 8, kTypeW8 = -8, kTypeF32 = 32;

// y[n] = x . w[n] + b[n]: w N rows of Q8_0 blocks (K / 32 of them, 34 bytes: an FP16 scale and 32 int8 values),
// x FP32; a work-group of 32 a row, reading it as 16-bit words, neighbouring items neighbouring words (a row of 3072
// values: 1632 words, 96 of them the scales, which the items of the block's other words read again from the cache)
void q8_0_vec(const uint8_t* w, const float* x, const float* b, float* y, int64_t N, int64_t K, void* s) {
    constexpr int64_t local = 32;
    const int64_t nw = K / 32 * 17;
    q_of(s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N * local)), sycl::range<1>((size_t) local)),
                         [=](sycl::nd_item<1> it) {
        const auto n = static_cast<int64_t>(it.get_group_linear_id());
        const auto lane = static_cast<int64_t>(it.get_local_linear_id());
        const auto* row = reinterpret_cast<const uint16_t*>(w + n * nw * 2);
        float acc = 0.f;
        for (int64_t k = lane; k < nw; k += local) {
            const int64_t blk = k / 17, o = k - blk * 17;   // the word's block, and its place there (0 the scale)
            if (o == 0) continue;
            const uint16_t v = row[k];
            const float* xb = x + blk * 32 + 2 * (o - 1);
            const float d = f32_from_f16(row[blk * 17]);
            acc += d * ((float) (int8_t) (v & 0xFF) * xb[0] + (float) (int8_t) (v >> 8) * xb[1]);
        }
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        if (lane == 0) y[n] = acc + (b != nullptr ? b[n] : 0.f);
    });
}

}  // namespace mf_dit_detail

using namespace mf_dit_detail;

MfDit::MfDit() {
    loras_.block_path = "transformer_blocks.";   // build_lora_targets' blocks (a request's block weights)
    DitLoraHooks& h = loras_.hooks;
    h.drop = [this] { drop_loras(); };
    h.dora_norms = [this](const DitLoraSlot& slot, const LoraEntry& e, std::vector<float>& out, std::string& err) {
        return dora_norms(slot, e, out, err);
    };
    h.refuse = [](const DitLoraSlot& slot, const LoraEntry&, std::string& why) {
        if (slot.w->type != kTypeF32) return false;
        // MageTrail's low-rank modulation: F32 factors, read by the F32 MMVF
        why = "the DiT's modulation is the low-rank form (F32, MageTrail's); its adapters are not applied";
        return true;
    };
    h.block_weights = [this](std::vector<DitW*>& ws) {
        for (Block& b : blocks_)
            for (W* w : {&b.imod, &b.tmod, &b.q, &b.k, &b.v, &b.tq, &b.tk, &b.tv, &b.o, &b.to, &b.imlp1, &b.imlp2,
                         &b.tmlp1, &b.tmlp2})
                ws.push_back(w);
    };
    h.reprepare = [this] {
        blocks_stream_.reload();   // the queued work drained before the prepare changes
        install_prepare();
    };
}

MfDit::~MfDit() {
    blocks_stream_.drop();
    drop_refs();
    for (void* p : allocs_) gpu::free(p);
}

bool MfDit::open(const std::string& path, void* stream, std::string& err, bool dry, bool q8,
                 std::vector<std::vector<uint8_t>>* reuse) {
    try {
        s_ = stream;
        dry_ = dry;
        q8_ = q8;
        // the int8 form: W8A8 (int8 activations a token, kTypeW8) where the GPU's library takes int8 products, else
        // Q8_0 (dequantized to FP16 for each product)
        w8_ = q8 && strata::env_get("NO_W8A8") == nullptr && prefill::Gemm::i8_usable(stream);
        st_ = std::make_unique<SafeTensors>(path);
        SafeTensors& st = *st_;
        file_ = std::make_unique<platform::MappedFile>(path);
        const uint8_t* data = file_->data() + st.data_base();
        auto tensor = [&](const std::string& name) -> const SafeTensor& {
            const SafeTensor* t = st.find(name);
            if (t == nullptr) throw std::runtime_error("image engine (Mage-Flow DiT): no " + name);
            if (t->dtype != "BF16") throw std::runtime_error("image engine (Mage-Flow DiT): " + name + " is not bf16");
            return *t;
        };
        // a block's modulation in the low-rank form: F32 [6 hidden][rank]
        auto lowrank = [&](const std::string& name) -> const SafeTensor& {
            const SafeTensor* t = st.find(name);
            if (t == nullptr) throw std::runtime_error("image engine (Mage-Flow DiT): no " + name);
            if (t->dtype != "F32" || t->shape.size() != 2 || t->shape[1] != mdown_.n)
                throw std::runtime_error("image engine (Mage-Flow DiT): " + name + " is not F32 [6 hidden][rank]");
            return *t;
        };
        // the parts outside the blocks: FP16 weights and FP32 vectors on the GPU
        auto dense = [&](const std::string& name, const float** bias) {
            const SafeTensor& t = tensor(name + ".weight");
            W w;
            w.type = kTypeF16;
            w.n = t.shape[0];
            w.k = t.shape[1];
            w.row_bytes = (uint64_t) w.k * 2;
            weight_bytes_ += (uint64_t) (w.n * w.k) * 2 + (uint64_t) w.n * 4;
            if (!dry_) {
                const auto* src = reinterpret_cast<const uint16_t*>(data + t.begin);
                std::vector<uint16_t> h((size_t) (w.n * w.k));
                for (size_t i = 0; i < h.size(); ++i) h[i] = f16_from_f32(bf16_value(src[i]));
                uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size(), "image engine (Mage-Flow DiT)");
                gpu::copy(d, h.data(), h.size() * 2);
                w.d = d;
                const auto b = st.f32(name + ".bias");
                float* db = alloc<float>(allocs_, (int64_t) b.size(), "image engine (Mage-Flow DiT)");
                gpu::copy(db, b.data(), b.size() * 4);
                *bias = db;
            }
            return w;
        };
        img_in_ = dense("img_in", &b_img_in_);
        in_ch_ = img_in_.k;
        C_ = img_in_.n;
        txt_in_ = dense("txt_in", &b_txt_in_);
        txt_dim_ = txt_in_.k;
        t1_ = dense("time_text_embed.timestep_embedder.linear_1", &b_t1_);
        t2_ = dense("time_text_embed.timestep_embedder.linear_2", &b_t2_);
        nout_ = dense("norm_out.linear", &b_nout_);
        pout_ = dense("proj_out", &b_pout_);
        weight_bytes_ += (uint64_t) txt_dim_ * 4;
        if (!dry_) {
            const auto v = st.f32("txt_norm.weight");
            float* d = alloc<float>(allocs_, (int64_t) v.size(), "image engine (Mage-Flow DiT)");
            gpu::copy(d, v.data(), v.size() * 4);
            txt_norm_ = d;
        }
        // the shared low-rank modulation (MageTrail's): SiLU(temb) to a few hundred values, FP32
        if (const SafeTensor* t = st.find("modulation_down.weight"); t != nullptr) {
            if (t->dtype != "F32" || t->shape.size() != 2 || t->shape[1] != C_)
                throw std::runtime_error("image engine (Mage-Flow DiT): modulation_down is not F32 [rank][hidden]");
            mdown_.type = kTypeF32;
            mdown_.n = t->shape[0];
            mdown_.k = t->shape[1];
            weight_bytes_ += (uint64_t) (mdown_.n * mdown_.k + 2 * mdown_.n) * 4;
            if (!dry_) {
                const auto wv = st.f32("modulation_down.weight");
                const auto bv = st.f32("modulation_down.bias");
                float* dw = alloc<float>(allocs_, (int64_t) wv.size(), "image engine (Mage-Flow DiT)");
                gpu::copy(dw, wv.data(), wv.size() * 4);
                mdown_.d = dw;
                b_mdown_ = alloc<float>(allocs_, (int64_t) bv.size(), "image engine (Mage-Flow DiT)");
                gpu::copy(b_mdown_, bv.data(), bv.size() * 4);
            }
        }
        hd_ = tensor("transformer_blocks.0.attn.norm_q.weight").shape[0];
        heads_ = C_ / hd_;
        int64_t axes_sum = 0;
        for (int64_t a : axes_) axes_sum += a;
        if (axes_sum != hd_) throw std::runtime_error("image engine (Mage-Flow DiT): the RoPE's axes are not the head");
        int64_t scratch = std::max({nout_.n * nout_.k, txt_in_.n * txt_in_.k, t2_.n * t2_.k});
        for (int i = 0;; ++i) {
            const std::string p = "transformer_blocks." + std::to_string(i) + ".";
            if (st.find(p + "img_mod.1.weight") == nullptr) break;
            Block b;
            struct Item {
                W* w;
                uint64_t* bias;
                std::string name;
            };
            const std::vector<Item> items = {
                {&b.imod, &b.b_imod, "img_mod.1"}, {&b.tmod, &b.b_tmod, "txt_mod.1"},
                {&b.q, &b.b_q, "attn.to_q"}, {&b.k, &b.b_k, "attn.to_k"}, {&b.v, &b.b_v, "attn.to_v"},
                {&b.tq, &b.b_tq, "attn.add_q_proj"}, {&b.tk, &b.b_tk, "attn.add_k_proj"},
                {&b.tv, &b.b_tv, "attn.add_v_proj"}, {&b.o, &b.b_o, "attn.to_out.0"},
                {&b.to, &b.b_to, "attn.to_add_out"}, {&b.imlp1, &b.b_imlp1, "img_mlp.net.0.proj"},
                {&b.imlp2, &b.b_imlp2, "img_mlp.net.2"}, {&b.tmlp1, &b.b_tmlp1, "txt_mlp.net.0.proj"},
                {&b.tmlp2, &b.b_tmlp2, "txt_mlp.net.2"}};
            const std::vector<std::pair<uint64_t*, std::string>> norms = {
                {&b.nq, "attn.norm_q.weight"}, {&b.nk, "attn.norm_k.weight"},
                {&b.ntq, "attn.norm_added_q.weight"}, {&b.ntk, "attn.norm_added_k.weight"}};
            // the block's host bytes: the quantized weights (q8), then every FP32 vector
            uint64_t qbytes = 0, vbytes = 0;
            for (const Item& it : items) {
                const bool mod = mdown_.n > 0 && (it.w == &b.imod || it.w == &b.tmod);
                const SafeTensor& t = mod ? lowrank(p + it.name + ".weight") : tensor(p + it.name + ".weight");
                it.w->n = t.shape[0];
                it.w->k = t.shape[1];
                if (it.w->k % 32 != 0) throw std::runtime_error("image engine (Mage-Flow DiT): columns not 32s");
                if (mod) {   // the low-rank modulation's FP32 factors, as they are
                    it.w->type = kTypeF32;
                    it.w->row_bytes = (uint64_t) it.w->k * 4;
                    if (q8_) qbytes += it.w->row_bytes * (uint64_t) it.w->n;
                    vbytes += (uint64_t) it.w->n * 4;
                    continue;
                }
                // the modulations stay Q8_0: their one row (the time's embedding) scales, shifts and gates every
                // row of the block, and a scale for all of it costs the most (RTX 4070, Turbo's 4 steps, the final
                // latent's cosine to the reference: 0.991 all Q8_0, 0.961 with W8A8 modulations alone, 0.981 with all
                // the other products in W8A8)
                const bool w8 = w8_ && it.w->k % kQiRotate == 0 && it.w != &b.imod && it.w != &b.tmod;
                it.w->type = w8 ? kTypeW8 : q8_ ? kTypeQ8_0 : kTypeF16;   // FP16: the file's bf16, rewritten on the GPU
                it.w->row_bytes = w8    ? (uint64_t) it.w->k
                                  : q8_ ? (uint64_t) (it.w->k / 32 * 34)
                                        : (uint64_t) it.w->k * 2;
                if (q8_) qbytes += it.w->row_bytes * (uint64_t) it.w->n + (w8 ? (uint64_t) it.w->n * 4 : 0);
                vbytes += (uint64_t) it.w->n * 4;
                scratch = std::max(scratch, it.w->n * it.w->k);
            }
            for (const auto& [off, name] : norms) vbytes += (uint64_t) tensor(p + name).elements() * 4;
            std::vector<LayerStream::Span> spans;
            std::vector<uint8_t> host;
            // an earlier open's bytes of this block, in the same form (their size says it: W8A8's rows and Q8_0's
            // differ), else the file's quantized again
            const size_t bi = host_.size();
            const bool reused = !dry_ && q8_ && reuse != nullptr && bi < reuse->size() &&
                                (*reuse)[bi].size() == qbytes + vbytes;
            if (reused) host = std::move((*reuse)[bi]);
            else if (!dry_) host.resize((size_t) (qbytes + vbytes));
            uint64_t at = 0;
            std::vector<size_t> wspan(items.size()), bspan(items.size()), nspan(norms.size());
            for (size_t j = 0; j < items.size(); ++j) {
                const SafeTensor& t = *st.find(p + items[j].name + ".weight");
                const W& w = *items[j].w;
                const uint64_t bytes = w.row_bytes * (uint64_t) w.n + (w.type == kTypeW8 ? (uint64_t) w.n * 4 : 0);
                if (q8_ && w.type == kTypeF32) {
                    if (!dry_ && !reused) std::memcpy(host.data() + at, data + t.begin, bytes);
                    spans.push_back({host.data() + at, bytes, 0});
                    at += bytes;
                } else if (q8_) {
                    if (!dry_ && !reused)
                        quantize_matrix(reinterpret_cast<const uint16_t*>(data + t.begin), w.n, w.k, host.data() + at,
                                        w.type == kTypeW8);
                    spans.push_back({host.data() + at, bytes, 0});
                    at += bytes;
                } else {
                    spans.push_back({data + t.begin, bytes, 0});
                }
                wspan[j] = spans.size() - 1;
            }
            auto vec = [&](const std::string& name, size_t* span) {
                const SafeTensor* t = st.find(name);   // bf16, or F32 (the low-rank modulations' biases)
                if (t == nullptr) throw std::runtime_error("image engine (Mage-Flow DiT): no " + name);
                const auto v = dry_ || reused ? std::vector<float>((size_t) t->elements()) : st.f32(name);
                if (!dry_ && !reused) std::memcpy(host.data() + at, v.data(), v.size() * 4);
                spans.push_back({host.data() + at, v.size() * 4, 0});
                at += v.size() * 4;
                *span = spans.size() - 1;
            };
            for (size_t j = 0; j < items.size(); ++j) vec(p + items[j].name + ".bias", &bspan[j]);
            for (size_t j = 0; j < norms.size(); ++j) vec(p + norms[j].second, &nspan[j]);
            const int layer = blocks_stream_.add(spans);
            for (size_t j = 0; j < items.size(); ++j) {
                items[j].w->d = reinterpret_cast<const void*>(blocks_stream_.offset(layer, (int) wspan[j]));
                *items[j].bias = blocks_stream_.offset(layer, (int) bspan[j]);
            }
            for (size_t j = 0; j < norms.size(); ++j) *norms[j].first = blocks_stream_.offset(layer, (int) nspan[j]);
            if (q8_ && !dry_ && !reused) {   // the block's bf16 pages: read once, not kept in the page cache
                uint64_t lo = UINT64_MAX, hi = 0;
                for (const Item& it : items) {
                    const SafeTensor& t = *st.find(p + it.name + ".weight");
                    lo = std::min(lo, st.data_base() + t.begin);
                    hi = std::max(hi, st.data_base() + t.end);
                }
                file_->drop_cache(lo, hi - lo);
            }
            host_.push_back(std::move(host));
            blocks_.push_back(b);
            F_ = b.imlp1.n;
        }
        if (blocks_.empty()) throw std::runtime_error("image engine (Mage-Flow DiT): no transformer blocks");
        build_lora_targets();
        install_prepare();
        if (dry_) return true;
        if (!gemm_.init(s_, scratch, err)) return false;
        blocks_stream_.configure(blocks_stream_.total(), s_);   // all resident until a budget is set
        t16_ = alloc<uint16_t>(allocs_, 256 + 2 * C_, "image engine (Mage-Flow DiT)");
        temb_ = alloc<float>(allocs_, C_, "image engine (Mage-Flow DiT)");
        mod_i_ = alloc<float>(allocs_, 6 * C_, "image engine (Mage-Flow DiT)");
        mod_t_ = alloc<float>(allocs_, 6 * C_, "image engine (Mage-Flow DiT)");
        nmod_ = alloc<float>(allocs_, 2 * C_, "image engine (Mage-Flow DiT)");
        if (mdown_.n > 0) mlow_ = alloc<float>(allocs_, mdown_.n, "image engine (Mage-Flow DiT)");
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

std::pair<double, double> MfDit::time_forms(int64_t rows) const {
    return time_w8_f16(rows, C_, F_, "Mage-Flow's DiT", s_);
}

// Each block's weights, as they reach the GPU (resident, or a streamed slot): bf16 to FP16 (the FP16 blocks), and the
// adapters baked in (set_loras); none of it for 8-bit blocks without baked adapters
void MfDit::install_prepare() {
    if (q8_ && !loras_.baked_any()) {
        blocks_stream_.set_prepare(nullptr);
        return;
    }
    blocks_stream_.set_prepare([this](int i, uint8_t* d) {
        sycl::queue& q = q_of(s_);
        const Block& b = blocks_[(size_t) i];
        for (const W* w : {&b.imod, &b.tmod, &b.q, &b.k, &b.v, &b.tq, &b.tk, &b.tv, &b.o, &b.to, &b.imlp1, &b.imlp2,
                           &b.tmlp1, &b.tmlp2}) {
            uint8_t* wd = d + reinterpret_cast<uintptr_t>(w->d);
            if (!q8_ && w->type != kTypeF32) bf16_to_f16_inplace(reinterpret_cast<uint16_t*>(wd), w->n * w->k, q);
            if (w->baked) w->lora->bake(w->type, wd, w->n, w->k, gemm_, loras_.bake_tmp(), loras_.bake_at16(), s_);
        }
    });
}

void MfDit::modulation(const uint16_t* tsil, const W& w, const uint8_t* base, float* y, const float* bias) {
    if (w.type == kTypeF32) {
        // the shared F32 MMVF (bf16_gemv.hpp) and the bias as a residual add: the former f32_vec summed strided by
        // 32 lanes, the MMVF in its tuned order (the same values to rounding; bf16_gemv_parity checks the difference)
        kernels::f32_gemv_fp32_mmvf(mlow_, reinterpret_cast<const float*>(base + reinterpret_cast<uintptr_t>(w.d)),
                                    y, w.k, w.n, s_);
        if (bias != nullptr) kernels::add_inplace(y, bias, w.n, s_, 1.0f);
        return;
    }
    if (w.type != kTypeQ8_0 || w.lora != nullptr) {
        // (a weight an adapter changes takes the product: its delta is added to the dense product's output)
        product(tsil, nullptr, w, base, y, 1, bias);
        return;
    }
    // one row (the time's SiLU, FP32 in nmod_ until the final norm's product): a matrix-vector product reads the
    // blocks as they are, where the dense product would dequantize the whole matrix (6 x 3072 x 3072 values) for it.
    // The input stays FP32: as int8 blocks (the text engine's matrix-vector products) the final latent's cosine to
    // the reference fell from 0.97 to 0.95 (RTX 4070, Turbo's 4 steps), the modulations scaling every row.
    q8_0_vec(base + reinterpret_cast<uintptr_t>(w.d), nmod_, bias, y, w.n, w.k, s_);
}

uint64_t MfDit::activation_bytes(int64_t hw, int n, int64_t max_text, int64_t mlp_rows) const {
    const int64_t R = n * hw, L = n * max_text, X = std::max(R, L), TP = qi_attn_rows(hw + max_text);
    const int64_t band = std::max(L, mlp_rows > 0 ? std::min(R, mlp_rows) : R);
    return (uint64_t) (R + L) * C_ * 4 +                 // the two streams
           (uint64_t) (blocks_.size() * 2 + 1 + kColSlices) * C_ * 4 +   // the int8 attention's statistics, col_max's
           (uint64_t) X * C_ * 4 * 4 +                   // q, k, v, a
           (uint64_t) band * F_ * 4 +                    // the MLP's hidden rows
           (uint64_t) std::max(X * C_, band * F_) * 2 +  // the products' FP16 input
           (uint64_t) n * TP * C_ * 2 * 3 + (uint64_t) TP * C_ * 2 +   // the joint q, k, v (a context each), output
           (uint64_t) (R + L) * C_ * 2 +                 // the attention's outputs by stream
           (uint64_t) (hw + max_text) * hd_ * 4 +        // the rotary tables
           (uint64_t) TP * C_ * 4 +                      // the plain attention's scratch
           (w8_ ? (uint64_t) std::max(X * C_, band * F_) + (uint64_t) std::max(X, band) * 8 : 0) +   // W8A8's input
           (prod_.lora.cols > 0 ? (uint64_t) std::max(X, band) * (uint64_t) prod_.lora.cols * 6
                                 : 0);   // the deltas' scratch
}

void MfDit::ensure(int64_t hw, int n, int64_t max_text) {
    if (hw <= cap_hw_ && n <= cap_n_ && max_text <= cap_text_ && (mlp_cap_ <= 0 || mlp_rows_ <= mlp_cap_)) return;
    drop_activations();
    const int64_t R = n * hw, L = n * max_text, X = std::max(R, L), TP = qi_attn_rows(hw + max_text);
    mlp_rows_ = std::max(L, mlp_cap_ > 0 ? std::min(R, mlp_cap_) : R);
    img_ = alloc<float>(allocs_, R * C_, "image engine (Mage-Flow DiT)");
    txt_ = alloc<float>(allocs_, L * C_, "image engine (Mage-Flow DiT)");
    q_ = alloc<float>(allocs_, X * C_, "image engine (Mage-Flow DiT)");
    k_ = alloc<float>(allocs_, X * C_, "image engine (Mage-Flow DiT)");
    v_ = alloc<float>(allocs_, X * C_, "image engine (Mage-Flow DiT)");
    a_ = alloc<float>(allocs_, X * C_, "image engine (Mage-Flow DiT)");
    tq_ = alloc<float>(allocs_, L * C_, "image engine (Mage-Flow DiT)");
    tk_ = alloc<float>(allocs_, L * C_, "image engine (Mage-Flow DiT)");
    tv_ = alloc<float>(allocs_, L * C_, "image engine (Mage-Flow DiT)");
    rs_ = alloc<float>(allocs_, std::max(X, mlp_rows_), "image engine (Mage-Flow DiT)");
    rs2_ = alloc<float>(allocs_, std::max(X, mlp_rows_), "image engine (Mage-Flow DiT)");
    rs_io_ = alloc<float>(allocs_, R, "image engine (Mage-Flow DiT)");
    rs_to_ = alloc<float>(allocs_, L, "image engine (Mage-Flow DiT)");
    hs_ = alloc<float>(allocs_, 2 * heads_, "image engine (Mage-Flow DiT)");
    vpart_ = alloc<float>(allocs_, std::max((hw + max_text) * heads_, kColSlices * C_), "image engine (Mage-Flow DiT)");   // head_max's, col_max's
    // the int8 attention's statistics: a context's V scales, K's mean a block and context (zero until a step makes it)
    vsv_ = alloc<float>(allocs_, C_, "image engine (Mage-Flow DiT)");
    kmean_.init(alloc<float>(allocs_, (int64_t) blocks_.size() * 2 * C_, "image engine (Mage-Flow DiT)"),
                (int64_t) blocks_.size() * 2, C_);
    gpu::memset_async(kmean_.buf, 0, (size_t) blocks_.size() * 2 * C_ * 4, s_);
    gu_ = alloc<float>(allocs_, mlp_rows_ * F_, "image engine (Mage-Flow DiT)");
    h16_ = alloc<uint16_t>(allocs_, std::max(X * C_, mlp_rows_ * F_), "image engine (Mage-Flow DiT)");
    if (w8_) {   // W8A8's int8 input: the most rows of the widest input, then a scale a row
        DitI8Input& in = prod_.in;
        in.rows = std::max(X, mlp_rows_);
        in.cols = std::max(C_, F_);
        in.q = alloc<int8_t>(allocs_, in.rows * in.cols + 2 * in.rows * 4, "image engine (Mage-Flow DiT)");
    }
    jq16_ = alloc<uint16_t>(allocs_, n * TP * C_, "image engine (Mage-Flow DiT)", true);
    jk16_ = alloc<uint16_t>(allocs_, n * TP * C_, "image engine (Mage-Flow DiT)", true);
    jv16_ = alloc<uint16_t>(allocs_, n * TP * C_, "image engine (Mage-Flow DiT)", true);
    jo16_ = alloc<uint16_t>(allocs_, TP * C_, "image engine (Mage-Flow DiT)");
    io16_ = alloc<uint16_t>(allocs_, R * C_, "image engine (Mage-Flow DiT)");
    to16_ = alloc<uint16_t>(allocs_, L * C_, "image engine (Mage-Flow DiT)");
    cos_ = alloc<float>(allocs_, hw * hd_ / 2, "image engine (Mage-Flow DiT)");
    sin_ = alloc<float>(allocs_, hw * hd_ / 2, "image engine (Mage-Flow DiT)");
    {
        std::vector<float> one((size_t) (max_text * hd_ / 2), 1.f), zero((size_t) (max_text * hd_ / 2), 0.f);
        cos_id_ = alloc<float>(allocs_, max_text * hd_ / 2, "image engine (Mage-Flow DiT)");
        sin_id_ = alloc<float>(allocs_, max_text * hd_ / 2, "image engine (Mage-Flow DiT)");
        gpu::copy(cos_id_, one.data(), one.size() * 4);
        gpu::copy(sin_id_, zero.data(), zero.size() * 4);
    }
    b0_img_ = alloc<float>(allocs_, hw * C_, "image engine (Mage-Flow DiT)");
    xin_ = alloc<float>(allocs_, hw * in_ch_, "image engine (Mage-Flow DiT)");
    b0_txt_ = alloc<float>(allocs_, max_text * C_, "image engine (Mage-Flow DiT)");
    cap_hw_ = hw;
    cap_n_ = n;
    cap_text_ = max_text;
    dirty_ = 0;
    rope_h_ = rope_w_ = -1;
}

void MfDit::drop_activations() {
    gpu::stream_sync(s_);
    for (void* p : {(void*) tq_, (void*) tk_, (void*) tv_, (void*) rs_, (void*) rs2_, (void*) rs_io_, (void*) rs_to_,
                    (void*) hs_, (void*) vpart_, (void*) vsv_, (void*) kmean_.buf})
        release(allocs_, p);
    tq_ = tk_ = tv_ = rs_ = rs2_ = rs_io_ = rs_to_ = hs_ = vpart_ = vsv_ = nullptr;
    kmean_.clear();
    for (void* p : {(void*) img_, (void*) txt_, (void*) q_, (void*) k_, (void*) v_, (void*) a_, (void*) gu_,
                    (void*) h16_, (void*) jq16_, (void*) jk16_, (void*) jv16_, (void*) jo16_, (void*) io16_,
                    (void*) to16_, (void*) cos_, (void*) sin_, (void*) cos_id_, (void*) sin_id_,
                    (void*) b0_img_, (void*) b0_txt_, (void*) xin_, (void*) prod_.in.q})
        release(allocs_, p);
    prod_.in.q = nullptr;
    prod_.lora.drop_buffers(allocs_);
    img_ = txt_ = q_ = k_ = v_ = a_ = gu_ = cos_ = sin_ = cos_id_ = sin_id_ = nullptr;
    b0_img_ = b0_txt_ = xin_ = nullptr;
    h16_ = jq16_ = jk16_ = jv16_ = jo16_ = io16_ = to16_ = nullptr;
    cap_hw_ = cap_n_ = cap_text_ = 0;
}

std::vector<MfDit::LoraTarget> MfDit::lora_targets() const { return loras_.targets(); }

void MfDit::build_lora_targets() {
    loras_.clear_slots();
    // every weight of Mage-Flow's DiT adds a bias (an entry's bias's delta is added beside it)
    auto add = [this](std::string name, W& w) { loras_.add(std::move(name), w, true); };
    // the weights outside the blocks (the norms are norms, not linear products: unchanged)
    const char* const* out = lora::kMageOutside;
    add(out[0], img_in_);
    add(out[1], txt_in_);
    add(out[2], t1_);
    add(out[3], t2_);
    add(out[4], nout_);
    add(out[5], pout_);
    // the blocks: the two streams' modulations, the attention's projections and the MLPs
    for (int i = 0; i < (int) blocks_.size(); ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
        Block& b = blocks_[(size_t) i];
        const std::pair<const char*, W*> items[] = {
            {lora::kMageMod[0], &b.imod}, {lora::kMageMod[1], &b.tmod},
            {lora::kMageBlock[0][0], &b.q}, {lora::kMageBlock[0][1], &b.k}, {lora::kMageBlock[0][2], &b.v},
            {lora::kMageBlock[1][0], &b.tq}, {lora::kMageBlock[1][1], &b.tk}, {lora::kMageBlock[1][2], &b.tv},
            {lora::kMageBlock[0][3], &b.o}, {lora::kMageBlock[1][3], &b.to},
            {lora::kMageBlock[0][4], &b.imlp1}, {lora::kMageBlock[0][5], &b.imlp2},
            {lora::kMageBlock[1][4], &b.tmlp1}, {lora::kMageBlock[1][5], &b.tmlp2}};
        for (const auto& item : items) add(p + item.first, *item.second);
    }
}

bool MfDit::set_loras(const std::vector<LoraUse>& loras, std::string& err) {
    bool changed = false;
    return loras_.set(loras, err, changed);
}

bool MfDit::dora_norms(const LoraSlot& slot, const LoraEntry& e, std::vector<float>& out, std::string& err) {
    try {
        const std::string name = slot.name + ".weight";
        const SafeTensor* t = st_->find(name);
        if (t == nullptr || t->shape.size() != 2)
            throw std::runtime_error("image engine (Mage-Flow DiT): no " + name);
        if (t->dtype != "BF16") throw std::runtime_error("image engine (Mage-Flow DiT): " + name + " is not bf16");
        const uint8_t* data = file_->data() + st_->data_base() + t->begin;
        const int64_t rows = (int64_t) t->shape[0], k = (int64_t) t->shape[1];
        if (slot.src_row + slot.row0 + slot.rows > rows)
            throw std::runtime_error("image engine (Mage-Flow DiT): " + name + ": the file's rows are fewer than "
                                     "the weight's");
        return dit_dora_norms(e, slot.src_row, slot.src_row + slot.row0, slot.rows, k,
                              dit_bf16_rows(reinterpret_cast<const uint16_t*>(data), k), out, err);
    } catch (const std::exception& ex) {
        err = std::string("image engine (Mage-Flow DiT): a DoRA's row norms: ") + ex.what();
        return false;
    }
}

uint64_t MfDit::lora_bytes() const { return loras_.bytes(); }

void MfDit::drop_loras() { loras_.drop(); }

void mf_rope_rows(const std::vector<std::pair<int64_t, int64_t>>& images, int64_t hd, const std::vector<int64_t>& axes,
                  std::vector<float>& c, std::vector<float>& s) {
    const int64_t half = hd / 2, a0 = axes[0] / 2, a1 = axes[1] / 2;
    int64_t rows = 0;
    for (const auto& im : images) rows += im.first * im.second;
    c.assign((size_t) (rows * half), 0.f);
    s.assign(c.size(), 0.f);
    size_t row = 0;
    for (size_t f = 0; f < images.size(); ++f) {
        const int64_t ih = images[f].first, iw = images[f].second;
        const int64_t y0 = ih - ih / 2, x0 = iw - iw / 2;   // the centre's row and column
        for (int64_t y = 0; y < ih; ++y)
            for (int64_t x = 0; x < iw; ++x, ++row)
                for (int64_t j = 0; j < half; ++j) {
                    double pos = (double) f, dim = (double) axes[0], k = (double) j;
                    if (j >= a0 && j < a0 + a1) {
                        pos = (double) (y - y0);
                        dim = (double) axes[1];
                        k = (double) (j - a0);
                    } else if (j >= a0 + a1) {
                        pos = (double) (x - x0);
                        dim = (double) axes[2];
                        k = (double) (j - a0 - a1);
                    }
                    const double ang = pos * std::pow(10000.0, -2.0 * k / dim);
                    c[row * (size_t) half + (size_t) j] = (float) std::cos(ang);
                    s[row * (size_t) half + (size_t) j] = (float) std::sin(ang);
                }
    }
}

// the target's rows' tables, then each reference's
void MfDit::rope_tables(int64_t h, int64_t w) {
    if (h == rope_h_ && w == rope_w_ && refs_gen_ == rope_refs_) return;
    std::vector<std::pair<int64_t, int64_t>> images{{h, w}};
    images.insert(images.end(), ref_sizes_.begin(), ref_sizes_.end());
    std::vector<float> c, s;
    mf_rope_rows(images, hd_, axes_, c, s);
    gpu::stream_sync(s_);
    gpu::copy(cos_, c.data(), c.size() * 4);
    gpu::copy(sin_, s.data(), s.size() * 4);
    rope_h_ = h;
    rope_w_ = w;
    rope_refs_ = refs_gen_;
}

void MfDit::drop_refs() {
    if (refs_ != nullptr) {
        gpu::stream_sync(s_);
        release(allocs_, refs_);
    }
    refs_ = nullptr;
    ref_rows_ = 0;
    ref_sizes_.clear();
}

void MfDit::set_refs(const float* z, const std::vector<std::pair<int64_t, int64_t>>& sizes) {
    drop_refs();
    ++refs_gen_;
    int64_t rows = 0;
    for (const auto& [rh, rw] : sizes) rows += rh * rw;
    if (rows == 0) return;
    refs_ = alloc<float>(allocs_, rows * in_ch_, "image engine (Mage-Flow DiT)");
    gpu::copy_async(refs_, z, (size_t) (rows * in_ch_) * 4, s_);
    gpu::stream_sync(s_);
    ref_rows_ = rows;
    ref_sizes_ = sizes;
}

// the time's embedding as the reference makes it in bf16 (it was trained with this rounding): the sinusoid of the
// bf16 sigma by bf16 frequencies (flipped: cos then sin), rounded to bf16, through the two linear layers; then temb's
// SiLU as FP16 (the modulations' input) at t16_ + 256 + C
void mf_time_sinusoid(float sigma, std::vector<uint16_t>& e) {
    const float t = round_bf16(sigma);
    e.assign(256, 0);
    for (int i = 0; i < 128; ++i) {
        const float f = round_bf16(std::exp(-std::log(10000.f) * (float) i / 128.f));
        const float a = 1000.f * (t * f);
        e[(size_t) i] = f16_from_f32(round_bf16(std::cos(a)));
        e[128 + (size_t) i] = f16_from_f32(round_bf16(std::sin(a)));
    }
}

void MfDit::time_embedding(float sigma) {
    std::vector<uint16_t> e;
    mf_time_sinusoid(sigma, e);
    gpu::copy_async(t16_, e.data(), e.size() * 2, s_);
    gpu::stream_sync(s_);   // e leaves scope
    product(t16_, nullptr, t1_, nullptr, temb_, 1, b_t1_);
    qi_silu(temb_, C_, t16_ + 256, s_);
    product(t16_ + 256, nullptr, t2_, nullptr, temb_, 1, b_t2_);
    gpu::copy_async(nmod_, temb_, (size_t) C_ * 4, s_);   // temb kept; its SiLU below
    qi_silu(nmod_, C_, t16_ + 256 + C_, s_);
    if (mdown_.n > 0) {
        kernels::f32_gemv_fp32_mmvf(nmod_, static_cast<const float*>(mdown_.d), mlow_, mdown_.k, mdown_.n, s_);
        kernels::add_inplace(mlow_, b_mdown_, mdown_.n, s_, 1.0f);
    }
}

void MfDit::set_context(int i, const float* states, int64_t rows) {
    Ctx& c = ctx_[i];
    gpu::stream_sync(s_);
    if (c.txt0 != nullptr) release(allocs_, c.txt0);
    c.txt0 = nullptr;
    c.rows = rows;
    if (rows <= 0) return;
    float* x = alloc<float>(allocs_, rows * txt_dim_, "image engine (Mage-Flow DiT)");
    uint16_t* x16 = alloc<uint16_t>(allocs_, rows * txt_dim_, "image engine (Mage-Flow DiT)");
    gpu::copy(x, states, (size_t) (rows * txt_dim_) * 4);
    kernels::rms_norm_rows(x, txt_norm_, 1, nullptr, x16, (int) (txt_dim_), (int) (rows), 1e-6f, s_);
    c.txt0 = alloc<float>(allocs_, rows * C_, "image engine (Mage-Flow DiT)");
    product(x16, nullptr, txt_in_, nullptr, c.txt0, rows, b_txt_in_);
    gpu::stream_sync(s_);
    release(allocs_, x);
    release(allocs_, x16);
}

void MfDit::forward(const float* x, float sigma, int64_t h, int64_t w, int n, float* out) {
    // HW: a context's image rows, the target's (TW) and an edit's references' after them
    const int64_t TW = h * w, HW = TW + ref_rows_, C = C_, R = n * HW;
    int64_t L = 0, maxL = 0;
    for (int i = 0; i < n; ++i) {
        L += ctx_[i].rows;
        maxL = std::max(maxL, ctx_[i].rows);
    }
    ensure(HW, n, std::max(maxL, cap_text_));
    if (sigma > last_sigma_) kmean_.reset();   // a new image: K's means made again
    last_sigma_ = sigma;
    rope_tables(h, w);
    const int64_t TP = qi_attn_rows(HW + cap_text_);
    // the image's input projection (each context the same rows), the text stream's start
    if (ref_rows_ > 0) {
        gpu::copy_async(xin_, x, (size_t) (TW * in_ch_) * 4, s_);
        gpu::copy_async(xin_ + TW * in_ch_, refs_, (size_t) (ref_rows_ * in_ch_) * 4, s_);
        x = xin_;
    }
    kernels::f32_to_f16_bulk(x, h16_, HW * in_ch_, s_);
    product(h16_, nullptr, img_in_, nullptr, img_, HW, b_img_in_);
    for (int i = 1; i < n; ++i) gpu::copy_async(img_ + i * HW * C, img_, (size_t) (HW * C) * 4, s_);
    std::vector<int64_t> toff(2, 0);
    for (int i = 0; i < n; ++i) {
        toff[(size_t) i] = i == 0 ? 0 : toff[(size_t) i - 1] + ctx_[i - 1].rows;
        gpu::copy_async(txt_ + toff[(size_t) i] * C, ctx_[i].txt0, (size_t) (ctx_[i].rows * C) * 4, s_);
    }
    // the joint rows past a context's end zero (the attention reads them up to T rounded up)
    for (int i = 0; i < n; ++i) {
        const int64_t T = ctx_[i].rows + HW;
        if (dirty_ > T)
            for (uint16_t* b : {jq16_, jk16_, jv16_})
                gpu::memset_async(b + (i * TP + T) * C, 0, (size_t) ((dirty_ - T) * C) * 2, s_);
    }
    for (int i = 0; i < n; ++i) dirty_ = std::max(dirty_, ctx_[i].rows + HW);
    time_embedding(sigma);
    const uint16_t* tsil = t16_ + 256 + C;
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        const Block& b = blocks_[(size_t) bi];
        const uint8_t* base = blocks_stream_.begin(bi);
        if (skip_ != nullptr && std::find(skip_->begin(), skip_->end(), bi) != skip_->end()) {
            blocks_stream_.end(bi);
            continue;
        }
        auto f = [base](uint64_t off) { return reinterpret_cast<const float*>(base + off); };
        modulation(tsil, b.imod, base, mod_i_, f(b.b_imod));
        modulation(tsil, b.tmod, base, mod_t_, f(b.b_tmod));
        // q, k, v of the image rows and of the text rows
        rows16(LnMod{img_, mod_i_, mod_i_ + C, C}, R, C, h16_, rs_, s_);
        // (the products' sums are scaled by their readers, head_max and qk_rope16)
        const DitOut q = product(h16_, rs_, b.q, base, q_, R, f(b.b_q), false, true);
        const DitOut k = product(h16_, rs_, b.k, base, k_, R, f(b.b_k), b.q.type == kTypeW8, true);
        const DitOut v = product(h16_, rs_, b.v, base, v_, R, f(b.b_v), b.k.type == kTypeW8, true);
        rows16(LnMod{txt_, mod_t_, mod_t_ + C, C}, L, C, h16_, rs2_, s_);   // rs_ is the image rows' still
        const DitOut tq = product(h16_, rs2_, b.tq, base, tq_, L, f(b.b_tq), false, true);
        const DitOut tk = product(h16_, rs2_, b.tk, base, tk_, L, f(b.b_tk), b.tq.type == kTypeW8, true);
        const DitOut tv = product(h16_, rs2_, b.tv, base, tv_, L, f(b.b_tv), b.tk.type == kTypeW8, true);
        // each context: its v heads' scales, its rows normed (the image's rotated) into [text, image], the attention,
        // its output back to each stream's rows (FP16, scaled by row)
        for (int i = 0; i < n; ++i) {
            const int64_t o = i * TP * C, Lt = ctx_[i].rows, r0 = toff[(size_t) i], t0 = r0 * C;
            const DitOut vi = v.from(i * HW);
            // v's heads' scales by column, the int8 attention's V scales with them and K's mean kept from the last step
            col_max(tv.from(r0), Lt, vi, HW, C, vpart_, s_);
            col_scale(vpart_, C, heads_, hd_, hs_ + i * heads_, vsv_, s_);
            qk_rope16(tq.from(r0), tk.from(r0), tv.from(r0), f(b.ntq), f(b.ntk), nullptr, nullptr, hs_ + i * heads_,
                      Lt, heads_, hd_, false, jq16_ + o, jk16_ + o, jv16_ + o, s_);
            qk_rope16(q.from(i * HW), k.from(i * HW), vi, f(b.nq), f(b.nk), cos_, sin_, hs_ + i * heads_, HW, heads_,
                      hd_, false, jq16_ + o + Lt * C, jk16_ + o + Lt * C, jv16_ + o + Lt * C, s_);
            // every row sees every key (the text rows too: Mage-Flow's joint attention has no mask), so all of
            // them go to the GPU maker's attention where there is one
            // (two contexts at most kept: a third makes its own mean)
            const int64_t slot = i < 2 ? (int64_t) bi * 2 + i : kmean_.slots;
            const auto st = kmean_.stats(slot, vsv_);
            qi_attention(jq16_ + o, jk16_ + o, jv16_ + o, Lt + HW, heads_, hd_, 0, jo16_, s_, nullptr, 0, true, nullptr,
                         &st);
            kmean_.mark(slot);
            rows16(Heads{jo16_, hs_ + i * heads_, C, hd_}, Lt, C, to16_ + t0, rs_to_ + toff[(size_t) i], s_);
            rows16(Heads{jo16_ + Lt * C, hs_ + i * heads_, C, hd_}, HW, C, io16_ + i * HW * C, rs_io_ + i * HW, s_);
        }
        gate_add(img_, product(io16_, rs_io_, b.o, base, a_, R, f(b.b_o), false, true), mod_i_ + 2 * C, R, C, s_);
        gate_add(txt_, product(to16_, rs_to_, b.to, base, a_, L, f(b.b_to), false, true), mod_t_ + 2 * C, L, C, s_);
        // the MLPs, the image's a band of rows at a time
        for (int64_t m0 = 0; m0 < R; m0 += mlp_rows_) {
            const int64_t mr = std::min(mlp_rows_, R - m0);
            rows16(LnMod{img_ + m0 * C, mod_i_ + 3 * C, mod_i_ + 4 * C, C}, mr, C, h16_, rs_, s_);
            // the GELU reads the first product's sums with rs_, so its own rows' scales go to rs2_
            rows16(GeluTanh{product(h16_, rs_, b.imlp1, base, gu_, mr, f(b.b_imlp1), false, true)}, mr, F_, h16_, rs2_,
                   s_);
            gate_add(img_ + m0 * C, product(h16_, rs2_, b.imlp2, base, a_, mr, f(b.b_imlp2), false, true),
                     mod_i_ + 5 * C, mr, C, s_);
        }
        rows16(LnMod{txt_, mod_t_ + 3 * C, mod_t_ + 4 * C, C}, L, C, h16_, rs_, s_);
        rows16(GeluTanh{product(h16_, rs_, b.tmlp1, base, gu_, L, f(b.b_tmlp1), false, true)}, L, F_, h16_, rs2_, s_);
        gate_add(txt_, product(h16_, rs2_, b.tmlp2, base, a_, L, f(b.b_tmlp2), false, true), mod_t_ + 5 * C, L, C,
                 s_);
        blocks_stream_.end(bi);
        if (bi == 0 && keep0_ && skip_ == nullptr) {
            gpu::copy_async(b0_img_, img_, (size_t) (HW * C) * 4, s_);
            gpu::copy_async(b0_txt_, txt_, (size_t) (ctx_[0].rows * C) * 4, s_);
        }
    }
    // the output: AdaLayerNormContinuous (its emb: scale, then shift) and the projection to the latent's channels
    // (the target's rows of each context)
    product(tsil, nullptr, nout_, nullptr, nmod_, 1, b_nout_);
    if (ref_rows_ == 0) {
        rows16(LnMod{img_, nmod_ + C, nmod_, C}, R, C, h16_, rs_, s_);
        product(h16_, rs_, pout_, nullptr, out, R, b_pout_);
        return;
    }
    for (int i = 0; i < n; ++i) {
        rows16(LnMod{img_ + i * HW * C, nmod_ + C, nmod_, C}, TW, C, h16_, rs_, s_);
        product(h16_, rs_, pout_, nullptr, out + i * TW * in_ch_, TW, b_pout_);
    }
}

void MfDit::forward_skip(const float* x, float sigma, int64_t h, int64_t w, const std::vector<int>& blocks,
                         float* out) {
    skip_ = &blocks;
    try {
        forward(x, sigma, h, w, 1, out);
    } catch (...) {
        skip_ = nullptr;
        throw;
    }
    skip_ = nullptr;
}

void MfDit::sample(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, float cfg,
                   const std::function<bool(int, int)>& step, const std::function<void(float*, float)>& after,
                   const std::function<void(int, const std::function<void(float*)>&)>& peek,
                   const GuidanceSpec* guide, bool er, float shift, uint64_t seed) {
    const GuidanceSpec spec = guide != nullptr ? *guide : GuidanceSpec{};
    for (const int b : spec.slg_layers)
        if (b < 0 || b >= (int) blocks_.size())
            throw std::invalid_argument("image engine: skip layer guidance's block " + std::to_string(b) +
                                        " is not one of the model's " + std::to_string(blocks_.size()));
    const int n = cfg > 1.f && ctx_[1].rows > 0 ? 2 : 1;
    const int64_t N = h * w * in_ch_;
    Guidance g(spec, s_, N, in_ch_, true, n == 2 ? cfg : 1.f, sigmas);
    flow_sample(s_, [&](const float* xx, float sigma, int nn, float* out) { forward(xx, sigma, h, w, nn, out); },
                x, N, n, sigmas, cfg, er, shift, seed, step, after, peek, &g,
                [&](const float* xx, float sigma, float* out) { forward_skip(xx, sigma, h, w, g.skip_layers(), out); },
                true);
}

}  // namespace strata::image
