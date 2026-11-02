// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_dit.cpp - see include/strata/image/mf_dit.hpp.

#include "strata/image/mf_dit.hpp"

#include "dit_rows.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mapped_file.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/i8_gemm.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/platform/env.hpp"

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

// kTypeW8: int8 rows rotated as qi_rows_i8 rotates them, then a scale a row (FP32), for prefill::Gemm::i8
constexpr int kTypeF16 = 1, kTypeQ8_0 = 8, kTypeW8 = -8, kTypeF32 = 32;

template<class T> T* alloc(std::vector<void*>& owned, int64_t n, bool zero = false) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) std::max<int64_t>(n, 1) * sizeof(T)))
        throw std::runtime_error(std::string("image engine (Mage-Flow DiT): GPU memory: ") + gpu::last_error());
    if (zero && !gpu::memset(p, 0, (size_t) std::max<int64_t>(n, 1) * sizeof(T)))
        throw std::runtime_error(std::string("image engine (Mage-Flow DiT): memset: ") + gpu::last_error());
    owned.push_back(p);
    return p;
}

void release(std::vector<void*>& owned, void* p) {
    if (p == nullptr) return;
    owned.erase(std::remove(owned.begin(), owned.end(), p), owned.end());
    gpu::free(p);
}

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

// y[n] = x . w[n] + b[n] (b null: none): w N rows of K FP32, x FP32; a work-group of 32 a row
void f32_vec(const float* w, const float* x, const float* b, float* y, int64_t N, int64_t K, void* s) {
    constexpr int64_t local = 32;
    q_of(s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N * local)), sycl::range<1>((size_t) local)),
                         [=](sycl::nd_item<1> it) {
        const auto n = static_cast<int64_t>(it.get_group_linear_id());
        const auto lane = static_cast<int64_t>(it.get_local_linear_id());
        const float* row = w + n * K;
        float acc = 0.f;
        for (int64_t k = lane; k < K; k += local) acc += row[k] * x[k];
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        if (lane == 0) y[n] = acc + (b != nullptr ? b[n] : 0.f);
    });
}

// x0 = x - sigma * v, the step's prediction of the finished latent (a partial image's), v as euler's
void predicted(float* x0, const float* x, const float* out, int64_t n, int ctxs, float cfg, float sigma, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const float v = ctxs == 2 ? out[n + i] + cfg * (out[i] - out[n + i]) : out[i];
        x0[i] = x[i] - sigma * v;
    });
}

// x += ds * v, v the guided velocity: out[0] (the prompt's) alone, or out[1] + cfg (out[0] - out[1])
void euler(float* x, const float* out, int64_t n, int ctxs, float cfg, float ds, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const float v = ctxs == 2 ? out[n + i] + cfg * (out[i] - out[n + i]) : out[i];
        x[i] += ds * v;
    });
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

}  // namespace mf_dit_detail

using namespace mf_dit_detail;

MfDit::MfDit() = default;

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
        norm_cache_.clear();
        // the int8 form: W8A8 (int8 activations a token, kTypeW8) where the GPU's library takes int8 products, else
        // Q8_0 (dequantized to FP16 for each product)
        w8_ = q8 && strata::env_get("NO_W8A8") == nullptr && prefill::Gemm::i8_usable(stream);
        st_ = std::make_unique<SafeTensors>(path);
        SafeTensors& st = *st_;
        file_ = std::make_unique<MappedFile>(path);
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
                uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size());
                gpu::copy(d, h.data(), h.size() * 2);
                w.d = d;
                const auto b = st.f32(name + ".bias");
                float* db = alloc<float>(allocs_, (int64_t) b.size());
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
            float* d = alloc<float>(allocs_, (int64_t) v.size());
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
                float* dw = alloc<float>(allocs_, (int64_t) wv.size());
                gpu::copy(dw, wv.data(), wv.size() * 4);
                mdown_.d = dw;
                b_mdown_ = alloc<float>(allocs_, (int64_t) bv.size());
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
        t16_ = alloc<uint16_t>(allocs_, 256 + 2 * C_);
        temb_ = alloc<float>(allocs_, C_);
        mod_i_ = alloc<float>(allocs_, 6 * C_);
        mod_t_ = alloc<float>(allocs_, 6 * C_);
        nmod_ = alloc<float>(allocs_, 2 * C_);
        if (mdown_.n > 0) mlow_ = alloc<float>(allocs_, mdown_.n);
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
    if (q8_ && !baked_any_) {
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
            if (w->baked) w->lora->bake(w->type, wd, w->n, w->k, gemm_, bake_tmp_, bake_at16_, s_);
        }
    });
}

MfOut MfDit::product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
                     const float* bias, bool same_input, bool defer) {
    MfOut out;
    out.f = y;
    out.C = w.n;
    if (T <= 0) return out;
    LoraLayer* lora = w.baked ? nullptr : w.lora;
    if (lora != nullptr) defer = false;   // a delta is FP32: the int32 sums are scaled into y first
    // a DoRA's factor of the weight's rows: the base product's, then the delta's own
    const float* col = lora != nullptr && lora->has_base() ? lora->base() : nullptr;
    const auto* d = base != nullptr ? base + reinterpret_cast<uintptr_t>(w.d) : static_cast<const uint8_t*>(w.d);
    if (w.type == kTypeW8) {
        // the activations rotated and quantized as the weights (once for the products of one input), their scales
        // times the rows' (rs), the int32 sums into y and scaled there, or by the reader (defer)
        float* scales = reinterpret_cast<float*>(xq8_ + xq8_rows_ * xq8_cols_);
        if (!same_input) xs_next_ ^= 1;
        float* sx = scales + xs_next_ * xq8_rows_;
        if (!same_input) qi_rows_i8(x16, false, T, w.k, xq8_, sx, rs, s_);
        const auto* sw = reinterpret_cast<const float*>(d + w.n * w.k);
        // a reader's values as BF16 where the GPU takes such a product (kernels::i8_gemm_bf16): half the bytes it reads
        if (defer && kernels::i8_gemm_bf16(xq8_, sx, reinterpret_cast<const int8_t*>(d), sw, bias,
                                              reinterpret_cast<uint16_t*>(y), T, w.n, w.k, 0, s_)) {
            out.f = nullptr;
            out.bf = reinterpret_cast<const uint16_t*>(y);
            return out;
        }
        prefill::Gemm::i8(xq8_, reinterpret_cast<const int8_t*>(d), reinterpret_cast<int32_t*>(y), T, w.n, w.k, w.n,
                          s_);
        if (defer) {
            out.f = nullptr;
            out.i = reinterpret_cast<const int32_t*>(y);
            out.sx = sx;
            out.sw = sw;
            out.b = bias;
            return out;
        }
        // an OFT turns the weight's rows: the weight's own bias goes after it (the delta is the input's own, x16 =
        // x / rs, and the turn is of x16's units too)
        const bool oft = lora != nullptr && lora->has_oft();   // (a layer there: defer is off above)
        qi_i8_scale(y, T, w.n, w.n, sx, sw, oft ? nullptr : bias, col, s_);   // sx carries the rows' scales: y is in
                                                                            // the input's units
        if (oft) {
            lora_scratch(T);
            lora->add(x16, T, y, w.n, 0, w.n, rs, gemm_, lora_f32_, lora16_, s_);
            add_glora(lora, x16, rs, w, d, T, y);   // a GLoRA on the weight's other rows
            if (bias != nullptr) dit_rows::bias_add(y, bias, T, w.n, s_);
            return out;
        }
    } else {
        if (w.type == kTypeF16) gemm_.f16(x16, reinterpret_cast<const uint16_t*>(d), y, T, w.n, w.k);
        else gemm_.native(x16, w.type, d, y, T, w.n, w.k);
        if (defer) {
            out.rs = rs;
            out.b = bias;
            return out;
        }
        const bool oft = lora != nullptr && lora->has_oft();
        if (rs != nullptr || bias != nullptr || col != nullptr) rows_back(y, rs, oft ? nullptr : bias, T, w.n, s_, col);
        if (oft) {
            lora_scratch(T);
            lora->add(x16, T, y, w.n, 0, w.n, rs, gemm_, lora_f32_, lora16_, s_);
            add_glora(lora, x16, rs, w, d, T, y);   // a GLoRA on the weight's other rows
            if (bias != nullptr) dit_rows::bias_add(y, bias, T, w.n, s_);
            return out;
        }
    }
    if (lora != nullptr) {
        // the delta is the input's own (x16 = x / rs): its rows' scales put it back in y's units, as rows_back does
        lora_scratch(T);
        lora->add(x16, T, y, w.n, 0, w.n, rs, gemm_, lora_f32_, lora16_, s_);
        add_glora(lora, x16, rs, w, d, T, y);
    }
    return out;
}

void MfDit::add_glora(const LoraLayer* lora, const uint16_t* x16, const float* rs, const W& w, const uint8_t* d,
                         int64_t T, float* y) {
    if (lora == nullptr || !lora->has_glora()) return;
    // a GLoRA's own path: the weight's own product of its input, a band of tokens at a time
    lora->add_glora(x16, T, 0, w.n, rs, gemm_, lora_f32_, lora16_, s_,
                    [&](const uint16_t* v, int64_t t0, int64_t nt, int64_t r0, int64_t n2) {
                        glora_weight(v, w, d, r0, n2, nt, t0, y);
                    });
}

void MfDit::glora_weight(const uint16_t* v, const W& w, const uint8_t* d, int64_t r0, int64_t n, int64_t nt,
                         int64_t t0, float* y) {
    if (w.type == kTypeW8) {
        // the input's own int8 quantization (its buffer, not the product's: that one's is the other products' input)
        qi_rows_i8(v, false, nt, w.k, lora_glora_x8_, lora_glora_sx_, nullptr, s_);
        const auto* sw = reinterpret_cast<const float*>(d + w.n * w.k) + r0;
        prefill::Gemm::i8(lora_glora_x8_, reinterpret_cast<const int8_t*>(d) + r0 * w.k,
                          reinterpret_cast<int32_t*>(lora_glora_y_), nt, n, w.k, n, s_);
        qi_i8_scale(lora_glora_y_, nt, n, n, lora_glora_sx_, sw, nullptr, nullptr, s_);
    } else if (w.type == kTypeF16) {
        gemm_.f16(v, reinterpret_cast<const uint16_t*>(d) + r0 * w.k, lora_glora_y_, nt, n, w.k, n);
    } else {
        gemm_.native(v, w.type, d + (uint64_t) r0 * w.row_bytes, lora_glora_y_, nt, n, w.k, n);
    }
    lora_output_add(y + t0 * w.n, nt, w.n, r0, lora_glora_y_, n, s_);
}

void MfDit::modulation(const uint16_t* tsil, const W& w, const uint8_t* base, float* y, const float* bias) {
    if (w.type == kTypeF32) {
        f32_vec(reinterpret_cast<const float*>(base + reinterpret_cast<uintptr_t>(w.d)), mlow_, bias, y, w.n, w.k, s_);
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
           (lora_cols_ > 0 ? (uint64_t) std::max(X, band) * (uint64_t) lora_cols_ * 6 : 0);   // the deltas' scratch
}

void MfDit::ensure(int64_t hw, int n, int64_t max_text) {
    if (hw <= cap_hw_ && n <= cap_n_ && max_text <= cap_text_ && (mlp_cap_ <= 0 || mlp_rows_ <= mlp_cap_)) return;
    drop_activations();
    const int64_t R = n * hw, L = n * max_text, X = std::max(R, L), TP = qi_attn_rows(hw + max_text);
    mlp_rows_ = std::max(L, mlp_cap_ > 0 ? std::min(R, mlp_cap_) : R);
    img_ = alloc<float>(allocs_, R * C_);
    txt_ = alloc<float>(allocs_, L * C_);
    q_ = alloc<float>(allocs_, X * C_);
    k_ = alloc<float>(allocs_, X * C_);
    v_ = alloc<float>(allocs_, X * C_);
    a_ = alloc<float>(allocs_, X * C_);
    tq_ = alloc<float>(allocs_, L * C_);
    tk_ = alloc<float>(allocs_, L * C_);
    tv_ = alloc<float>(allocs_, L * C_);
    rs_ = alloc<float>(allocs_, std::max(X, mlp_rows_));
    rs2_ = alloc<float>(allocs_, std::max(X, mlp_rows_));
    rs_io_ = alloc<float>(allocs_, R);
    rs_to_ = alloc<float>(allocs_, L);
    hs_ = alloc<float>(allocs_, 2 * heads_);
    vpart_ = alloc<float>(allocs_, std::max((hw + max_text) * heads_, kColSlices * C_));   // head_max's, col_max's
    // the int8 attention's statistics: a context's V scales, K's mean a block and context (zero until a step makes it)
    vsv_ = alloc<float>(allocs_, C_);
    kmean_ = alloc<float>(allocs_, (int64_t) blocks_.size() * 2 * C_);
    gpu::memset_async(kmean_, 0, (size_t) blocks_.size() * 2 * C_ * 4, s_);
    kready_.assign(blocks_.size() * 2, 0);
    gu_ = alloc<float>(allocs_, mlp_rows_ * F_);
    h16_ = alloc<uint16_t>(allocs_, std::max(X * C_, mlp_rows_ * F_));
    if (w8_) {   // W8A8's int8 input: the most rows of the widest input, then a scale a row
        xq8_rows_ = std::max(X, mlp_rows_);
        xq8_cols_ = std::max(C_, F_);
        xq8_ = alloc<int8_t>(allocs_, xq8_rows_ * xq8_cols_ + 2 * xq8_rows_ * 4);
    }
    jq16_ = alloc<uint16_t>(allocs_, n * TP * C_, true);
    jk16_ = alloc<uint16_t>(allocs_, n * TP * C_, true);
    jv16_ = alloc<uint16_t>(allocs_, n * TP * C_, true);
    jo16_ = alloc<uint16_t>(allocs_, TP * C_);
    io16_ = alloc<uint16_t>(allocs_, R * C_);
    to16_ = alloc<uint16_t>(allocs_, L * C_);
    cos_ = alloc<float>(allocs_, hw * hd_ / 2);
    sin_ = alloc<float>(allocs_, hw * hd_ / 2);
    {
        std::vector<float> one((size_t) (max_text * hd_ / 2), 1.f), zero((size_t) (max_text * hd_ / 2), 0.f);
        cos_id_ = alloc<float>(allocs_, max_text * hd_ / 2);
        sin_id_ = alloc<float>(allocs_, max_text * hd_ / 2);
        gpu::copy(cos_id_, one.data(), one.size() * 4);
        gpu::copy(sin_id_, zero.data(), zero.size() * 4);
    }
    b0_img_ = alloc<float>(allocs_, hw * C_);
    xin_ = alloc<float>(allocs_, hw * in_ch_);
    b0_txt_ = alloc<float>(allocs_, max_text * C_);
    cap_hw_ = hw;
    cap_n_ = n;
    cap_text_ = max_text;
    dirty_ = 0;
    rope_h_ = rope_w_ = -1;
}

void MfDit::drop_activations() {
    gpu::stream_sync(s_);
    for (void* p : {(void*) tq_, (void*) tk_, (void*) tv_, (void*) rs_, (void*) rs2_, (void*) rs_io_, (void*) rs_to_,
                    (void*) hs_, (void*) vpart_, (void*) vsv_, (void*) kmean_})
        release(allocs_, p);
    tq_ = tk_ = tv_ = rs_ = rs2_ = rs_io_ = rs_to_ = hs_ = vpart_ = vsv_ = kmean_ = nullptr;
    kready_.clear();
    for (void* p : {(void*) img_, (void*) txt_, (void*) q_, (void*) k_, (void*) v_, (void*) a_, (void*) gu_,
                    (void*) h16_, (void*) jq16_, (void*) jk16_, (void*) jv16_, (void*) jo16_, (void*) io16_,
                    (void*) to16_, (void*) cos_, (void*) sin_, (void*) cos_id_, (void*) sin_id_,
                    (void*) b0_img_, (void*) b0_txt_, (void*) xin_, (void*) xq8_, (void*) lora_glora_y_,
                    (void*) lora_glora_x8_, (void*) lora_glora_sx_})
        release(allocs_, p);
    xq8_ = nullptr;
    img_ = txt_ = q_ = k_ = v_ = a_ = gu_ = cos_ = sin_ = cos_id_ = sin_id_ = nullptr;
    b0_img_ = b0_txt_ = xin_ = nullptr;
    h16_ = jq16_ = jk16_ = jv16_ = jo16_ = io16_ = to16_ = nullptr;
    lora_f32_ = nullptr;
    lora16_ = nullptr;
    lora_tmp_t_ = 0;
    lora_glora_y_ = nullptr;
    lora_glora_x8_ = nullptr;
    lora_glora_sx_ = nullptr;
    cap_hw_ = cap_n_ = cap_text_ = 0;
}

std::vector<MfDit::LoraTarget> MfDit::lora_targets() const {
    std::vector<LoraTarget> out;
    out.reserve(lora_slots_.size());
    for (const LoraSlot& s : lora_slots_) out.push_back({s.name, s.row0, s.rows, s.w->k, s.src_row});
    return out;
}

void MfDit::build_lora_targets() {
    lora_slots_.clear();
    auto add = [this](std::string name, W& w) {
        if (w.n <= 0 || w.k <= 0) return;
        lora_slots_.push_back({std::move(name), &w, 0, w.n, 0});
    };
    // the weights outside the blocks (the norms are norms, not linear products: unchanged)
    add("img_in", img_in_);
    add("txt_in", txt_in_);
    add("time_text_embed.timestep_embedder.linear_1", t1_);
    add("time_text_embed.timestep_embedder.linear_2", t2_);
    add("norm_out.linear", nout_);
    add("proj_out", pout_);
    // the blocks: the two streams' modulations, the attention's projections and the MLPs
    for (int i = 0; i < (int) blocks_.size(); ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
        Block& b = blocks_[(size_t) i];
        const std::pair<const char*, W*> items[] = {
            {"img_mod.1", &b.imod}, {"txt_mod.1", &b.tmod}, {"attn.to_q", &b.q}, {"attn.to_k", &b.k},
            {"attn.to_v", &b.v}, {"attn.add_q_proj", &b.tq}, {"attn.add_k_proj", &b.tk},
            {"attn.add_v_proj", &b.tv}, {"attn.to_out.0", &b.o}, {"attn.to_add_out", &b.to},
            {"img_mlp.net.0.proj", &b.imlp1}, {"img_mlp.net.2", &b.imlp2}, {"txt_mlp.net.0.proj", &b.tmlp1},
            {"txt_mlp.net.2", &b.tmlp2}};
        for (const auto& item : items) add(p + item.first, *item.second);
    }
}

void MfDit::lora_scratch(int64_t T) {
    if (lora_cols_ <= 0 || T <= 0) return;
    // a GLoRA's own path's buffers are a band's, not the tokens': the activations' drop frees them, this gives them back
    if (lora_glora_cols_ > 0 && lora_glora_y_ == nullptr) lora_glora_scratch();
    if (lora_f32_ != nullptr && T <= lora_tmp_t_) return;
    gpu::stream_sync(s_);   // the old buffers may still be read
    for (void** p : {reinterpret_cast<void**>(&lora_f32_), reinterpret_cast<void**>(&lora16_)})
        if (*p != nullptr) {
            release(allocs_, *p);
            *p = nullptr;
        }
    lora_f32_ = alloc<float>(allocs_, T * lora_cols_);
    lora16_ = alloc<uint16_t>(allocs_, T * lora_cols_);
    lora_tmp_t_ = T;
}

// A GLoRA's own path's buffers: the table the weight's product of its input is added from (a band of tokens, the
// weight's rows), and that input as a W8A8 product takes it (int8 values and a scale a row).  The deltas' scratch
// (lora_scratch) holds the input's own two products.
void MfDit::lora_glora_scratch() {
    if (lora_glora_cols_ <= 0 || lora_glora_rows_ <= 0) return;
    const size_t band = (size_t) kLoraGloraRows;
    lora_glora_y_ = alloc<float>(allocs_, (int64_t) (band * (size_t) lora_glora_rows_));
    lora_glora_x8_ = alloc<int8_t>(allocs_, (int64_t) (band * (size_t) lora_glora_cols_));
    lora_glora_sx_ = alloc<float>(allocs_, (int64_t) band);
}

void MfDit::free_glora_scratch() {
    for (void* p : {(void*) lora_glora_y_, (void*) lora_glora_x8_, (void*) lora_glora_sx_})
        if (p != nullptr) {
            gpu::stream_sync(s_);
            release(allocs_, p);
        }
    lora_glora_y_ = nullptr;
    lora_glora_x8_ = nullptr;
    lora_glora_sx_ = nullptr;
}

bool MfDit::set_loras(const std::vector<LoraUse>& loras, std::string& err) {
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
            const bool oft = e.kind == LoraKind::oft;
            const bool glora = e.kind == LoraKind::glora;
            if (e.kind == LoraKind::bias) {   // a bias's delta alone: its rows are the bias's, its columns none
                if (e.rows != slot->rows) {
                    err = "image engine: " + u.file->path() + ": " + e.key + ": a bias's delta (diff_b/b_norm) of " +
                          std::to_string(e.rows) + " values, not the weight's " + std::to_string(slot->rows) + " rows";
                    drop_loras();
                    return false;
                }
            } else if (e.rows != slot->rows || (oft ? e.cols != 0 : e.cols != slot->w->k)) {
                err = "image engine: " + u.file->path() + ": " + e.key + " is " + std::to_string(e.cols) + " x " +
                      std::to_string(e.rows) + ", not the DiT's " + std::to_string(slot->w->k) + " x " +
                      std::to_string(slot->rows);
                drop_loras();
                return false;
            }
            if (slot->w->type == kTypeF32) {   // MageTrail's low-rank modulation: F32 factors, read by f32_vec
                err = "image engine: " + u.file->path() + ": " + e.key +
                      ": the DiT's modulation is the low-rank form (F32, MageTrail's); its adapters are not applied";
                drop_loras();
                return false;
            }
            const float scale = u.scale * e.scale();
            // a strength of 0 changes nothing: the weight keeps its product's path (an alpha of 0 leaves the bias's
            // delta and a DoRA's base factors, as a DoRA's decomposition scales the weight with no delta)
            if (u.scale == 0.0f || (scale == 0.0f && e.bias.empty() && e.dora.empty())) continue;
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
            part.src_row = slot->src_row;
            part.scale = scale;
            part.bias_scale = u.scale;   // a diff patch has no alpha: the request's strength only
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
            for (W* w : {&b.imod, &b.tmod, &b.q, &b.k, &b.v, &b.tq, &b.tk, &b.tv, &b.o, &b.to, &b.imlp1, &b.imlp2, &b.tmlp1, &b.tmlp2})
                if (w->lora != nullptr && w->lora->bakeable() && (w->type == 1 || w->type == 8 || w->type == -8 ? w->type : 0) != 0) {
                    w->baked = true;
                    nk = std::max(nk, w->n * w->k);
                    for (const LoraPart& p : w->lora->parts) kr = std::max(kr, w->k * p.entry->rank());
                }
        if (nk > 0) {
            bake_tmp_ = alloc<float>(allocs_, nk);
            bake_at16_ = alloc<uint16_t>(allocs_, kr);
            baked_any_ = true;
            blocks_stream_.reload();   // the queued work drained before the prepare changes
            install_prepare();
        }
    }
    for (const LoraUse& u : loras)
        if (u.file) lora_files_.push_back(u.file);
    lora_key_ = key;
    return true;
}

bool MfDit::tensor_norms(const std::string& name, const std::vector<float>*& out, std::string& err) {
    auto it = norm_cache_.find(name);
    if (it != norm_cache_.end()) {
        out = &it->second;
        return true;
    }
    try {
        const SafeTensor* t = st_->find(name);
        if (t == nullptr || t->shape.size() != 2)
            throw std::runtime_error("image engine (Mage-Flow DiT): no " + name);
        if (t->dtype != "BF16") throw std::runtime_error("image engine (Mage-Flow DiT): " + name + " is not bf16");
        const uint8_t* data = file_->data() + st_->data_base() + t->begin;
        const int64_t rows = (int64_t) t->shape[0], k = (int64_t) t->shape[1];
        const auto* h = reinterpret_cast<const uint16_t*>(data);
        std::vector<float> norms((size_t) rows);
        for (int64_t r = 0; r < rows; ++r) {
            double s = 0;
            for (int64_t c = 0; c < k; ++c) {
                const float v = kernels::f32_from_bf16(h[r * k + c]);
                s += (double) v * (double) v;
            }
            norms[(size_t) r] = (float) std::sqrt(s);
        }
        out = &norm_cache_.emplace(name, std::move(norms)).first->second;
        return true;
    } catch (const std::exception& ex) {
        err = std::string("image engine (Mage-Flow DiT): a DoRA's row norms: ") + ex.what();
        return false;
    }
}

bool MfDit::dora_norms(const LoraSlot& slot, std::vector<float>& out, std::string& err) {
    const std::vector<float>* norms = nullptr;
    if (!tensor_norms(slot.name + ".weight", norms, err)) return false;
    if ((int64_t) norms->size() < slot.w->n) {
        err = "image engine (Mage-Flow DiT): " + slot.name + ": the file's rows are fewer than the weight's";
        return false;
    }
    out.assign(norms->begin(), norms->begin() + slot.w->n);
    return true;
}

uint64_t MfDit::lora_bytes() const {
    uint64_t n = 0;
    for (const std::unique_ptr<LoraLayer>& l : lora_layers_) n += l->bytes();
    if (lora_glora_y_ != nullptr)   // a GLoRA's own path's buffers
        n += (uint64_t) kLoraGloraRows * ((uint64_t) lora_glora_rows_ * 4 + (uint64_t) lora_glora_cols_ + 4);
    return n;
}

void MfDit::drop_loras() {
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
    free_glora_scratch();
    if (lora_f32_ != nullptr || lora16_ != nullptr) {
        gpu::stream_sync(s_);
        for (void** p : {reinterpret_cast<void**>(&lora_f32_), reinterpret_cast<void**>(&lora16_)})
            if (*p != nullptr) {
                release(allocs_, *p);
                *p = nullptr;
            }
        lora_tmp_t_ = 0;
    }
}

// the image rows' 2D multi-scale RoPE: pairs 0 .. 7 by the frame (0 the target, j an edit's reference j), then
// axes_[1] / 2 pairs by the row and axes_[2] / 2 by the column, each centred on its image (rows -(h - h / 2) ..
// h / 2 - 1); the target's rows, then each reference's
void MfDit::rope_tables(int64_t h, int64_t w) {
    if (h == rope_h_ && w == rope_w_ && refs_gen_ == rope_refs_) return;
    const int64_t half = hd_ / 2, a0 = axes_[0] / 2, a1 = axes_[1] / 2;
    std::vector<std::pair<int64_t, int64_t>> images{{h, w}};
    images.insert(images.end(), ref_sizes_.begin(), ref_sizes_.end());
    std::vector<float> c((size_t) ((h * w + ref_rows_) * half)), s(c.size());
    size_t row = 0;
    for (size_t f = 0; f < images.size(); ++f) {
        const int64_t ih = images[f].first, iw = images[f].second;
        const int64_t y0 = ih - ih / 2, x0 = iw - iw / 2;   // the centre's row and column
        for (int64_t y = 0; y < ih; ++y)
            for (int64_t x = 0; x < iw; ++x, ++row)
                for (int64_t j = 0; j < half; ++j) {
                    double pos = (double) f, dim = (double) axes_[0], k = (double) j;
                    if (j >= a0 && j < a0 + a1) {
                        pos = (double) (y - y0);
                        dim = (double) axes_[1];
                        k = (double) (j - a0);
                    } else if (j >= a0 + a1) {
                        pos = (double) (x - x0);
                        dim = (double) axes_[2];
                        k = (double) (j - a0 - a1);
                    }
                    const double ang = pos * std::pow(10000.0, -2.0 * k / dim);
                    c[row * (size_t) half + (size_t) j] = (float) std::cos(ang);
                    s[row * (size_t) half + (size_t) j] = (float) std::sin(ang);
                }
    }
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
    refs_ = alloc<float>(allocs_, rows * in_ch_);
    gpu::copy_async(refs_, z, (size_t) (rows * in_ch_) * 4, s_);
    gpu::stream_sync(s_);
    ref_rows_ = rows;
    ref_sizes_ = sizes;
}

// the time's embedding as the reference makes it in bf16 (it was trained with this rounding): the sinusoid of the
// bf16 sigma by bf16 frequencies (flipped: cos then sin), rounded to bf16, through the two linear layers; then temb's
// SiLU as FP16 (the modulations' input) at t16_ + 256 + C
void MfDit::time_embedding(float sigma) {
    const float t = round_bf16(sigma);
    std::vector<uint16_t> e(256);
    for (int i = 0; i < 128; ++i) {
        const float f = round_bf16(std::exp(-std::log(10000.f) * (float) i / 128.f));
        const float a = 1000.f * (t * f);
        e[(size_t) i] = f16_from_f32(round_bf16(std::cos(a)));
        e[128 + (size_t) i] = f16_from_f32(round_bf16(std::sin(a)));
    }
    gpu::copy_async(t16_, e.data(), e.size() * 2, s_);
    gpu::stream_sync(s_);   // e leaves scope
    product(t16_, nullptr, t1_, nullptr, temb_, 1, b_t1_);
    qi_silu(temb_, C_, t16_ + 256, s_);
    product(t16_ + 256, nullptr, t2_, nullptr, temb_, 1, b_t2_);
    gpu::copy_async(nmod_, temb_, (size_t) C_ * 4, s_);   // temb kept; its SiLU below
    qi_silu(nmod_, C_, t16_ + 256 + C_, s_);
    if (mdown_.n > 0) f32_vec(static_cast<const float*>(mdown_.d), nmod_, b_mdown_, mlow_, mdown_.n, mdown_.k, s_);
}

void MfDit::set_context(int i, const float* states, int64_t rows) {
    Ctx& c = ctx_[i];
    gpu::stream_sync(s_);
    if (c.txt0 != nullptr) release(allocs_, c.txt0);
    c.txt0 = nullptr;
    c.rows = rows;
    if (rows <= 0) return;
    float* x = alloc<float>(allocs_, rows * txt_dim_);
    uint16_t* x16 = alloc<uint16_t>(allocs_, rows * txt_dim_);
    gpu::copy(x, states, (size_t) (rows * txt_dim_) * 4);
    kernels::rms_norm_rows(x, txt_norm_, 1, nullptr, x16, (int) (txt_dim_), (int) (rows), 1e-6f, s_);
    c.txt0 = alloc<float>(allocs_, rows * C_);
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
    if (sigma > last_sigma_) std::fill(kready_.begin(), kready_.end(), 0);   // a new image: K's means made again
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
        const MfOut q = product(h16_, rs_, b.q, base, q_, R, f(b.b_q), false, true);
        const MfOut k = product(h16_, rs_, b.k, base, k_, R, f(b.b_k), b.q.type == kTypeW8, true);
        const MfOut v = product(h16_, rs_, b.v, base, v_, R, f(b.b_v), b.k.type == kTypeW8, true);
        rows16(LnMod{txt_, mod_t_, mod_t_ + C, C}, L, C, h16_, rs2_, s_);   // rs_ is the image rows' still
        const MfOut tq = product(h16_, rs2_, b.tq, base, tq_, L, f(b.b_tq), false, true);
        const MfOut tk = product(h16_, rs2_, b.tk, base, tk_, L, f(b.b_tk), b.tq.type == kTypeW8, true);
        const MfOut tv = product(h16_, rs2_, b.tv, base, tv_, L, f(b.b_tv), b.tk.type == kTypeW8, true);
        // each context: its v heads' scales, its rows normed (the image's rotated) into [text, image], the attention,
        // its output back to each stream's rows (FP16, scaled by row)
        for (int i = 0; i < n; ++i) {
            const int64_t o = i * TP * C, Lt = ctx_[i].rows, r0 = toff[(size_t) i], t0 = r0 * C;
            const MfOut vi = v.from(i * HW);
            // v's heads' scales by column, the int8 attention's V scales with them and K's mean kept from the last step
            col_max(tv.from(r0), Lt, vi, HW, C, vpart_, s_);
            col_scale(vpart_, C, heads_, hd_, hs_ + i * heads_, vsv_, s_);
            qk_rope16(tq.from(r0), tk.from(r0), tv.from(r0), f(b.ntq), f(b.ntk), nullptr, nullptr, hs_ + i * heads_,
                      Lt, heads_, hd_, false, jq16_ + o, jk16_ + o, jv16_ + o, s_);
            qk_rope16(q.from(i * HW), k.from(i * HW), vi, f(b.nq), f(b.nk), cos_, sin_, hs_ + i * heads_, HW, heads_,
                      hd_, false, jq16_ + o + Lt * C, jk16_ + o + Lt * C, jv16_ + o + Lt * C, s_);
            // every row sees every key (the text rows too: Mage-Flow's joint attention has no mask), so all of
            // them go to the GPU maker's attention where there is one
            const size_t ki = (size_t) bi * 2 + (size_t) i;   // (two contexts at most kept: a third makes its own)
            kernels::FlashAttnI8Stats st{vsv_, i < 2 ? kmean_ + (int64_t) ki * C : nullptr, i < 2 && kready_[ki] != 0};
            qi_attention(jq16_ + o, jk16_ + o, jv16_ + o, Lt + HW, heads_, hd_, 0, jo16_, s_, nullptr, 0, true, nullptr,
                         &st);
            if (i < 2) kready_[ki] = 1;
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
                   const GuidanceSpec* guide) {
    if (guide != nullptr && guide->any()) {
        sample_guided(x, h, w, sigmas, cfg, step, after, peek, *guide);
        return;
    }
    const int n = cfg > 1.f && ctx_[1].rows > 0 ? 2 : 1;
    const int64_t HW = h * w;
    float* out = alloc<float>(allocs_, n * HW * in_ch_);
    const int steps = (int) sigmas.size() - 1;
    for (int i = 0; i < steps; ++i) {
        forward(x, sigmas[(size_t) i], h, w, n, out);
        euler(x, out, HW * in_ch_, n, cfg, sigmas[(size_t) i + 1] - sigmas[(size_t) i], s_);
        if (after) after(x, sigmas[(size_t) i + 1]);
        if (peek) {
            const float s_next = sigmas[(size_t) i + 1];
            peek(i + 1, [&](float* x0) { predicted(x0, x, out, HW * in_ch_, n, cfg, s_next, s_); });
        }
        if (step && !step(i + 1, steps)) {
            gpu::stream_sync(s_);
            release(allocs_, out);
            throw QiStopped();
        }
    }
    gpu::stream_sync(s_);
    release(allocs_, out);
}

// the sampling with guidance.hpp's guidance: each step the model asked at its sigma (Detail Daemon's), the contexts'
// velocities (and skip layer guidance's pass) to the guided denoised values, Euler's step from them
void MfDit::sample_guided(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, float cfg,
                          const std::function<bool(int, int)>& step, const std::function<void(float*, float)>& after,
                          const std::function<void(int, const std::function<void(float*)>&)>& peek,
                          const GuidanceSpec& spec) {
    for (const int b : spec.slg_layers)
        if (b < 0 || b >= (int) blocks_.size())
            throw std::invalid_argument("image engine: skip layer guidance's block " + std::to_string(b) +
                                        " is not one of the model's " + std::to_string(blocks_.size()));
    const int n = cfg > 1.f && ctx_[1].rows > 0 ? 2 : 1;
    const int64_t N = h * w * in_ch_;
    Guidance g(spec, s_, N, in_ch_, true, n == 2 ? cfg : 1.f, sigmas);
    struct Free {
        MfDit& d;
        std::vector<float*> ps;
        ~Free() {
            gpu::stream_sync(d.s_);
            for (float* p : ps) release(d.allocs_, p);
        }
    } mine{*this, {}};
    auto get = [&](int64_t count) {
        mine.ps.push_back(alloc<float>(allocs_, count));
        return mine.ps.back();
    };
    float* out = get(n * N);
    float* den = get(N);
    float* k = !spec.slg_layers.empty() ? get(N) : nullptr;
    const int steps = (int) sigmas.size() - 1;
    for (int i = 0; i < steps; ++i) {
        const float s0 = sigmas[(size_t) i], s1 = sigmas[(size_t) i + 1], sm = g.model_sigma(s0);
        forward(x, sm, h, w, n, out);
        const bool skipping = k != nullptr && g.skip_pass(sm);
        if (skipping) forward_skip(x, sm, h, w, g.skip_layers(), k);
        g.denoised(x, out, n == 2 ? out + N : nullptr, skipping ? k : nullptr, sm, den);
        guidance_euler(x, den, s0, s1, N, s_);
        if (after) after(x, s1);
        if (peek) peek(i + 1, [&](float* x0) { gpu::copy_async(x0, den, (size_t) N * 4, s_); });
        if (step && !step(i + 1, steps)) throw QiStopped();
    }
}

}  // namespace strata::image
