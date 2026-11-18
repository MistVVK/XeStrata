// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/anima_dit.cpp - see include/strata/image/anima_dit.hpp.

#include "strata/image/anima_dit.hpp"
#include "strata/image/dit_file.hpp"
#include "strata/image/lora_names.hpp"

#include "dit_rows.hpp"
#include "img_out.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/platform/mapped_file.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/qi_sdpa_ex.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/image/gelu.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
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

namespace strata::image {

namespace gpu = strata::gpu;

namespace anima_dit_detail {

using namespace dit_rows;

constexpr int kTypeF16 = 1, kTypeQ8_0 = 8, kTypeW8 = -8;
constexpr float kEps = 1e-6f;
constexpr int64_t kCtxRows = 512;   // the context's rows at least: the adapter's output padded with zero rows

// A block's number in a key and where it ends: "blocks.7.attn.to_q" as the DiT names it, or "blocks_7_attn_to_q" as
// sd-scripts' and the LyCORIS form's keys write it.  -1 for a key that names no block; `after` is the separator the
// number ends at, so a remapped key keeps the spelling the file wrote.
int anima_block_of(const std::string& k, size_t& after) {
    if (k.size() < 8 || k.rfind("blocks", 0) != 0 || (k[6] != '.' && k[6] != '_')) return -1;
    after = k.find_first_of("._", 7);
    if (after == std::string::npos) return -1;
    try {
        return std::stoi(k.substr(7, after - 7));
    } catch (const std::exception&) {
        return -1;
    }
}

// RMSNorm(x[t]) * w (eps 1e-6), times the row's factor where given (the T5 ids' weights)
struct RmsW {
    const float *x, *w, *row;
    int64_t C;
    template<class R> Stats stats(int64_t t, int64_t lane, int64_t step, R reduce) const {
        const float* r = x + t * C;
        float s = 0.f;
        for (int64_t c = lane; c < C; c += step) s += r[c] * r[c];
        return {0.f, 1.f / sycl::sqrt(reduce(s) / (float) C + kEps)};
    }
    float value(int64_t t, int64_t c, const Stats& st) const {
        const float y = x[t * C + c] * st.inv * w[c];
        return row != nullptr ? y * row[t] : y;
    }
};

// GELU (erf form, PyTorch's nn.GELU()) of a product's sums
struct GeluErf {
    DitOut x;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const {
        const float v = x.at(t, c);
        return gelu_erf(v);
    }
    float bound(int64_t t, int64_t c) const { return sycl::fmax(x.at(t, c), 0.17f); }
};

// A head-major copy for the library's products: row t's head h RMS-normed over the head (w; null: not normed) and
// rotated in pairs (j, j + hd / 2) by cos[t][j], sin[t][j] (null: not rotated), into out [heads][Tp][hd] (rows past T
// left as they are: zero), or with `rows` into out [T][heads hd] (the shared attention's layout).  qk_norm_rope.hpp's
// kernel, the head's values clamped to FP16's range.
void heads16(const DitOut& x, int64_t T, int64_t Tp, int64_t heads, int64_t hd, const float* w, const float* cos,
             const float* sin, uint16_t* out, void* s, bool rows = false) {
    if (T <= 0) return;
    struct In {
        DitOut x;
        int64_t hd;
        auto head(int, int64_t t, int64_t h) const { return x.row(t, h * hd); }
    };
    struct Out {
        uint16_t* out;
        int64_t Tp, heads, hd;
        bool rows;
        auto out_head(int, int64_t t, int64_t h) const {
            uint16_t* p = out + (rows ? t * heads + h : h * Tp + t) * hd;
            return [p](int64_t d, float v) { p[d] = f16_from_f32(sycl::clamp(v, -65504.f, 65504.f)); };
        }
    };
    kernels::qk::Shape sh{T, heads, 0, hd};
    sh.wq = w;
    sh.values = false;
    const In in{x, hd};
    const Out o{out, Tp, heads, hd, rows};
    if (cos != nullptr) kernels::qk::qk_norm_rope(q_of(s), sh, in, kernels::qk::Tables{cos, sin, hd / 2}, o);
    else kernels::qk::qk_norm_rope(q_of(s), sh, in, kernels::qk::NoRope{}, o);
}

// out [T][C] = v's rows in FP16 (the fused attention's values)
void values16(const DitOut& v, int64_t T, int64_t C, uint16_t* out, void* s) {
    if (T <= 0) return;
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        out[i] = f16_from_f32(sycl::clamp(v.at(i / C, i % C), -65504.f, 65504.f));
    });
}

// out [heads][hd][Tp] = v's row t, head h, transposed (the products form's weighted values read V^T's rows)
void values_t16(const DitOut& v, int64_t T, int64_t Tp, int64_t heads, int64_t hd, uint16_t* out, void* s) {
    if (T <= 0) return;
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * heads * hd)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / (heads * hd), c = i - t * heads * hd, h = c / hd, d = c - h * hd;
        out[(h * hd + d) * Tp + t] = f16_from_f32(sycl::clamp(v.at(t, c), -65504.f, 65504.f));
    });
}

// The adapter's FP16 rows of a row-wise op without a row scale (its values, normed or a GELU's, stay well inside
// FP16; saturated at its largest), so that its products need no scale after them: a work-group of 256 a row, as its
// few rows (a prompt's ids) left Rows16's sub-group a row with too little to do (78 us for the GELU of 84 rows of 4096
// on the RX 9060 XT).
template<class Op> void rows16_plain(const Op& op, int64_t T, int64_t C, uint16_t* y16, void* s) {
    if (T <= 0) return;
    constexpr size_t kLocal = 256;
    q_of(s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * kLocal), sycl::range<1>(kLocal)),
                         [=](sycl::nd_item<1> it) {
        const auto t = static_cast<int64_t>(it.get_group_linear_id());
        const auto lane = static_cast<int64_t>(it.get_local_linear_id());
        const auto sum = [&](float v) { return sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()); };
        const auto st = op.stats(t, lane, (int64_t) kLocal, sum);
        for (int64_t c = lane; c < C; c += (int64_t) kLocal)
            y16[t * C + c] = f16_from_f32(sycl::clamp(op.value(t, c, st), -65504.f, 65504.f));
    });
}

// x[t] += y[t] (the adapter's residuals)
void add_rows(float* x, const float* y, int64_t n, void* s) { qi_axpy(x, y, 1.0f, n, s); }

// an F8 E4M3 byte (the scaled fp8 saves': no inf, 0x7f / 0xff NaN)
float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? (float) m / 8.f * std::ldexp(1.f, -6) : (1.f + (float) m / 8.f) * std::ldexp(1.f, e - 7);
    if (e == 15 && m == 7) v = NAN;
    return s ? -v : v;
}

// The RoPE angles' cosine and sine of each run's rows at positions 0 .. its rows - 1, half pairs each with frequency
// 10000^(-2j / (2 half)), rounded to bf16 as the adapter's RotaryEmbedding casts them to its activations' type
void rope_1d(const std::vector<int64_t>& rows, int64_t half, std::vector<float>& c, std::vector<float>& s) {
    int64_t all = 0;
    for (const int64_t r : rows) all += r;
    c.assign((size_t) (all * half), 0.f);
    s.assign(c.size(), 0.f);
    for (int64_t j = 0; j < half; ++j) {
        const float f = 1.f / std::pow(10000.f, (float) (2 * j) / (float) (2 * half));
        int64_t at = 0;
        for (const int64_t n : rows)
            for (int64_t r = 0; r < n; ++r, ++at) {
                const float ph = (float) r * f;
                c[(size_t) (at * half + j)] = round_bf16(std::cos(ph));
                s[(size_t) (at * half + j)] = round_bf16(std::sin(ph));
            }
    }
}

}  // namespace anima_dit_detail

using namespace anima_dit_detail;

bool anima_is_dit(const std::string& path) { return AnimaFile::is_dit(path); }

AnimaDit::AnimaDit() {
    loras_.block_path = "blocks.";   // build_lora_targets' blocks (a request's block weights)
    DitLoraHooks& h = loras_.hooks;
    h.drop = [this] { drop_loras(); };
    h.dora_norms = [this](const DitLoraSlot& slot, const LoraEntry& e, std::vector<float>& out, std::string& err) {
        return dora_norms(slot, e, out, err);
    };
    // a block's number in a key, in the DiT's own spelling or in sd-scripts' and the LyCORIS form's underscore one
    // (anima_block_of reads both and says where the number ends)
    h.file = [this](const LoraFile& f) {
        // the release's adapter on an expanded DiT: every block it names within the map's size
        lora_remap_ = !block_map_.empty() && (int) block_map_.size() < (int) blocks_.size();
        for (const LoraEntry& entry : f.entries()) {
            size_t after = 0;
            if (anima_block_of(entry.key, after) >= (int) block_map_.size()) lora_remap_ = false;
        }
    };
    h.rename = [this](const LoraEntry& raw, std::string& key) {
        size_t after = 0;
        const int b = anima_block_of(raw.key, after);
        if (!lora_remap_ || b < 0) return false;
        // the block's number changes, the file's own spelling of the name does not
        key = raw.key.substr(0, 7) + std::to_string(block_map_[(size_t) b]) + raw.key.substr(after);
        return true;
    };
    h.block_weights = [this](std::vector<DitW*>& ws) {
        for (Block& b : blocks_)
            for (W* w : {&b.sq, &b.sk, &b.sv, &b.so, &b.cq, &b.ck, &b.cv, &b.co, &b.l1, &b.l2, &b.msa1, &b.msa2,
                         &b.mca1, &b.mca2, &b.mml1, &b.mml2})
                ws.push_back(w);
    };
    h.reprepare = [this] {
        blocks_stream_.reload();   // the queued work drained before the prepare changes
        install_prepare();
    };
}

AnimaDit::~AnimaDit() {
    blocks_stream_.drop();
    for (void* p : allocs_) gpu::free(p);
    for (void* p : text_allocs_) gpu::free(p);
}

bool AnimaDit::open(const std::string& path, void* stream, std::string& err, bool dry, bool q8,
                    std::vector<std::vector<uint8_t>>* reuse) {
    try {
        s_ = stream;
        dry_ = dry;
        file_ = std::make_unique<AnimaFile>(path);
        AnimaFile& f = *file_;
        // a file whose blocks are not bf16 has no bytes the GPU could rewrite in place: 8 bits
        q8_ = q8 || f.get("blocks.0.self_attn.q_proj.weight").dtype != "BF16";
        w8_ = q8_ && strata::env_get("NO_W8A8") == nullptr && prefill::Gemm::i8_usable(stream);
        // the patch embedder: its first 64 columns (the 17th channel, the padding mask, is zero for an image)
        lat_ = f.get("final_layer.linear.weight").shape[0] / 4;
        xemb_ = dense("x_embedder.proj.1", false, nullptr, allocs_, weight_bytes_, 4 * lat_);
        C_ = xemb_.n;
        t1_ = dense("t_embedder.1.linear_1", false, nullptr, allocs_, weight_bytes_);
        t2_ = dense("t_embedder.1.linear_2", false, nullptr, allocs_, weight_bytes_);
        flin_ = dense("final_layer.linear", false, nullptr, allocs_, weight_bytes_);
        fm1_ = dense("final_layer.adaln_modulation.1", false, nullptr, allocs_, weight_bytes_);
        fm2_ = dense("final_layer.adaln_modulation.2", false, nullptr, allocs_, weight_bytes_);
        tnorm_host_ = f.f32("t_embedding_norm.weight");
        hd_ = f.get("blocks.0.self_attn.q_norm.weight").shape[0];
        heads_ = C_ / hd_;
        mod_dim_ = fm1_.n;
        if (C_ % hd_ != 0 || hd_ % 8 != 0 || t2_.n != 3 * C_ || flin_.n != 4 * lat_ || fm2_.n != 2 * C_)
            throw std::runtime_error("image engine (Anima DiT): an unexpected shape");
        load_text();
        for (int i = 0;; ++i) {
            const std::string p = "blocks." + std::to_string(i) + ".";
            if (f.find(p + "self_attn.q_proj.weight") == nullptr) break;
            Block b;
            const std::vector<std::pair<W*, std::string>> items = {
                {&b.sq, "self_attn.q_proj"}, {&b.sk, "self_attn.k_proj"}, {&b.sv, "self_attn.v_proj"},
                {&b.so, "self_attn.output_proj"}, {&b.cq, "cross_attn.q_proj"}, {&b.ck, "cross_attn.k_proj"},
                {&b.cv, "cross_attn.v_proj"}, {&b.co, "cross_attn.output_proj"}, {&b.l1, "mlp.layer1"},
                {&b.l2, "mlp.layer2"}, {&b.msa1, "adaln_modulation_self_attn.1"},
                {&b.msa2, "adaln_modulation_self_attn.2"}, {&b.mca1, "adaln_modulation_cross_attn.1"},
                {&b.mca2, "adaln_modulation_cross_attn.2"}, {&b.mml1, "adaln_modulation_mlp.1"},
                {&b.mml2, "adaln_modulation_mlp.2"}};
            const std::vector<std::pair<uint64_t*, std::string>> norms = {
                {&b.sqn, "self_attn.q_norm.weight"}, {&b.skn, "self_attn.k_norm.weight"},
                {&b.cqn, "cross_attn.q_norm.weight"}, {&b.ckn, "cross_attn.k_norm.weight"}};
            uint64_t qbytes = 0, vbytes = 0;
            for (const auto& [w, name] : items) {
                const SafeTensor& t = f.get(p + name + ".weight");
                w->n = t.shape[0];
                w->k = t.shape[1];
                if (w->k % 32 != 0) throw std::runtime_error("image engine (Anima DiT): columns not 32s");
                const bool w8 = w8_ && w->k % kQiRotate == 0;
                w->type = w8 ? kTypeW8 : q8_ ? kTypeQ8_0 : kTypeF16;
                w->row_bytes = w8 ? (uint64_t) w->k : q8_ ? (uint64_t) (w->k / 32 * 34) : (uint64_t) w->k * 2;
                if (q8_) qbytes += w->row_bytes * (uint64_t) w->n + (w8 ? (uint64_t) w->n * 4 : 0);
                if (w->type == kTypeQ8_0) scratch_ = std::max(scratch_, w->n * w->k);
            }
            for (const auto& [off, name] : norms) vbytes += (uint64_t) f.get(p + name).elements() * 4;
            std::vector<LayerStream::Span> spans;
            std::vector<uint8_t> host;
            const bool reused = !dry_ && q8_ && reuse != nullptr && (size_t) i < reuse->size() &&
                                (*reuse)[(size_t) i].size() == qbytes + vbytes;
            if (reused) host = std::move((*reuse)[(size_t) i]);
            else if (!dry_) host.resize((size_t) (qbytes + vbytes));
            uint64_t at = 0;
            std::vector<size_t> wspan, nspan;
            for (const auto& [w, name] : items) {
                const uint64_t bytes = w->row_bytes * (uint64_t) w->n + (w->type == kTypeW8 ? (uint64_t) w->n * 4 : 0);
                if (q8_) {
                    if (!dry_ && !reused) {
                        std::vector<uint16_t> tmp;
                        quantize_matrix(f.bf16(p + name + ".weight", tmp), w->n, w->k, host.data() + at,
                                        w->type == kTypeW8);
                        f.drop_cache(p + name + ".weight");
                    }
                    spans.push_back({host.data() + at, bytes, 0});
                    at += bytes;
                } else {
                    spans.push_back({f.raw(f.get(p + name + ".weight")), bytes, 0});
                }
                wspan.push_back(spans.size() - 1);
            }
            for (const auto& [off, name] : norms) {
                const int64_t n = f.get(p + name).elements();
                if (!dry_ && !reused) {
                    const auto v = f.f32(p + name);
                    std::memcpy(host.data() + at, v.data(), v.size() * 4);
                }
                spans.push_back({host.data() + at, (uint64_t) n * 4, 0});
                at += (uint64_t) n * 4;
                nspan.push_back(spans.size() - 1);
            }
            const int layer = blocks_stream_.add(spans);
            for (size_t j = 0; j < items.size(); ++j)
                items[j].first->d = reinterpret_cast<const void*>(blocks_stream_.offset(layer, (int) wspan[j]));
            for (size_t j = 0; j < norms.size(); ++j) *norms[j].first = blocks_stream_.offset(layer, (int) nspan[j]);
            host_.push_back(std::move(host));
            blocks_.push_back(b);
        }
        if (blocks_.empty()) throw std::runtime_error("image engine (Anima DiT): no blocks");
        F_ = blocks_[0].l1.n;
        src_dim_ = blocks_[0].ck.k;
        build_lora_targets();
        install_prepare();
        for (Ctx& c : ctx_) {
            c.kv.assign(blocks_.size(), nullptr);
            c.kv_lp.assign(blocks_.size(), 0);
            c.kv_ok.assign(blocks_.size(), 0);
        }
        if (dry_) return true;
        if (!gemm_.init(s_, scratch_, err)) return false;
        blocks_stream_.configure(blocks_stream_.total(), s_);
        e16_ = alloc<uint16_t>(allocs_, C_ + 4 * lat_, "image engine (Anima DiT)");
        m16_ = alloc<uint16_t>(allocs_, std::max(mod_dim_, C_), "image engine (Anima DiT)");
        alora_ = alloc<float>(allocs_, 3 * C_, "image engine (Anima DiT)");
        mh_ = alloc<float>(allocs_, std::max(mod_dim_, C_), "image engine (Anima DiT)");
        mod_ = alloc<float>(allocs_, 3 * C_, "image engine (Anima DiT)");
        fmod_ = alloc<float>(allocs_, 2 * C_, "image engine (Anima DiT)");
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// a part outside the blocks as FP16 (its first `cols` columns when given), its bias FP32
AnimaDit::W AnimaDit::dense(const std::string& name, bool biased, const float** bias, std::vector<void*>& owned,
                            uint64_t& bytes, int64_t cols) {
    AnimaFile& f = *file_;
    const SafeTensor& t = f.get(name + ".weight");
    W w;
    w.type = kTypeF16;
    w.n = t.shape[0];
    const int64_t K = t.shape[1];
    w.k = cols > 0 ? cols : K;
    w.row_bytes = (uint64_t) w.k * 2;
    bytes += (uint64_t) (w.n * w.k) * 2 + (biased ? (uint64_t) w.n * 4 : 0);
    if (dry_) return w;
    std::vector<uint16_t> tmp;
    const uint16_t* src = f.bf16(name + ".weight", tmp);
    std::vector<uint16_t> h((size_t) (w.n * w.k));
    for (int64_t r = 0; r < w.n; ++r)
        for (int64_t c = 0; c < w.k; ++c) h[(size_t) (r * w.k + c)] = f16_from_f32(bf16_value(src[r * K + c]));
    uint16_t* d = alloc<uint16_t>(owned, (int64_t) h.size(), "image engine (Anima DiT)");
    gpu::copy(d, h.data(), h.size() * 2);
    w.d = d;
    if (biased) {
        const auto v = f.f32(name + ".bias");
        float* db = alloc<float>(owned, (int64_t) v.size(), "image engine (Anima DiT)");
        gpu::copy(db, v.data(), v.size() * 4);
        *bias = db;
    }
    return w;
}

const float* AnimaDit::vec(const std::string& name, std::vector<void*>& owned, uint64_t& bytes) {
    const int64_t n = file_->get(name).elements();
    bytes += (uint64_t) n * 4;
    if (dry_) return nullptr;
    const auto v = file_->f32(name);
    float* d = alloc<float>(owned, n, "image engine (Anima DiT)");
    gpu::copy(d, v.data(), v.size() * 4);
    return d;
}

// the LLM adapter's weights, FP16 on the GPU (the embedding stays in the file: a prompt's rows are gathered on the host)
void AnimaDit::load_text() {
    text_bytes_ = 0;
    const auto keep = [](W& dst, const W& src) {
        LoraLayer* l = dst.lora;
        dst = src;
        dst.lora = l;
    };
    const SafeTensor& e = file_->get("llm_adapter.embed.weight");
    vocab_ = e.shape[0];
    ad_C_ = e.shape[1];
    for (int i = 0;; ++i) {
        const std::string p = "llm_adapter.blocks." + std::to_string(i) + ".";
        if (file_->find(p + "self_attn.q_proj.weight") == nullptr) break;
        if (ablocks_.size() <= (size_t) i) ablocks_.resize((size_t) i + 1);
        AdapterBlock& b = ablocks_[(size_t) i];
        keep(b.sq, dense(p + "self_attn.q_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.sk, dense(p + "self_attn.k_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.sv, dense(p + "self_attn.v_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.so, dense(p + "self_attn.o_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.cq, dense(p + "cross_attn.q_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.ck, dense(p + "cross_attn.k_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.cv, dense(p + "cross_attn.v_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.co, dense(p + "cross_attn.o_proj", false, nullptr, text_allocs_, text_bytes_));
        keep(b.m0, dense(p + "mlp.0", true, &b.bm0, text_allocs_, text_bytes_));
        keep(b.m2, dense(p + "mlp.2", true, &b.bm2, text_allocs_, text_bytes_));
        b.n_sa = vec(p + "norm_self_attn.weight", text_allocs_, text_bytes_);
        b.n_ca = vec(p + "norm_cross_attn.weight", text_allocs_, text_bytes_);
        b.n_ml = vec(p + "norm_mlp.weight", text_allocs_, text_bytes_);
        b.sqn = vec(p + "self_attn.q_norm.weight", text_allocs_, text_bytes_);
        b.skn = vec(p + "self_attn.k_norm.weight", text_allocs_, text_bytes_);
        b.cqn = vec(p + "cross_attn.q_norm.weight", text_allocs_, text_bytes_);
        b.ckn = vec(p + "cross_attn.k_norm.weight", text_allocs_, text_bytes_);
    }
    if (ablocks_.empty()) throw std::runtime_error("image engine (Anima DiT): the LLM adapter has no blocks");
    ad_hd_ = file_->get("llm_adapter.blocks.0.self_attn.q_norm.weight").shape[0];
    ad_heads_ = ad_C_ / ad_hd_;
    ad_F_ = ablocks_[0].m0.n;
    keep(aout_, dense("llm_adapter.out_proj", true, &b_aout_, text_allocs_, text_bytes_));
    anorm_ = vec("llm_adapter.norm.weight", text_allocs_, text_bytes_);
    text_loaded_ = !dry_;
    build_lora_targets();
}

void AnimaDit::drop_text() {
    if (!text_loaded_) return;
    gpu::stream_sync(s_);
    for (void* p : text_allocs_) gpu::free(p);
    text_allocs_.clear();
    ad_scratch_ = nullptr;
    ad_bytes_ = 0;
    text_loaded_ = false;
}

std::pair<double, double> AnimaDit::time_forms(int64_t rows) const {
    return time_w8_f16(rows, C_, F_, "Anima's DiT", s_);
}

uint64_t AnimaDit::activation_bytes(int64_t hw, int n, int64_t mlp_rows) const {
    const int64_t R = n * hw, L = qi_attn_rows(std::max(kCtxRows, cap_L_)), TP = qi_attn_rows(hw), Rk = std::max(R, L);
    const int64_t band = mlp_rows > 0 ? std::min(R, mlp_rows) : R;
    return (uint64_t) R * C_ * 4 * 3 +                          // x, q, the products form's FP32 output
           (uint64_t) Rk * C_ * 4 * 3 +                         // k, v, a
           (uint64_t) band * F_ * 4 +                           // the MLP's first sums
           (uint64_t) std::max(R * C_, band * F_) * 2 +         // the products' FP16 input
           (uint64_t) TP * C_ * 2 * 4 +                         // the self-attention's q, k, v, output
           (uint64_t) Rk * C_ * 2 * 2 +                         // the attention's FP16 output, the heads' q
           (uint64_t) hw * L * 6 +                              // the products form's scores and probabilities
           (uint64_t) hw * hd_ * 4 +                            // the rotary tables
           (uint64_t) (blocks_.size() * 2 + 1) * C_ * 4 +        // the int8 attention's K means and V scales
           (uint64_t) kColSlices * C_ * 4 +                     // col_max's parts (vpart_ at least)
           (uint64_t) TP * C_ * 4 +                             // the attention's scratch
           (uint64_t) n * blocks_.size() * L * C_ * 2 * 2 +     // the contexts' kept keys and values
           (w8_ ? (uint64_t) std::max(Rk, band) * std::max(C_, F_) + (uint64_t) std::max(Rk, band) * 8 : 0) +
           prod_.lora.bytes(Rk);
}

void AnimaDit::ensure(int64_t hw, int n) {
    const int64_t cap_rows = cap_n_ * cap_hw_;
    if (hw <= cap_hw_ && n <= cap_n_ && mlp_rows_ == (mlp_cap_ > 0 ? std::min(cap_rows, mlp_cap_) : cap_rows)) return;
    hw = std::max(hw, cap_hw_);
    n = std::max<int>(n, (int) cap_n_);
    drop_activations();
    const int64_t R = n * hw, L = qi_attn_rows(std::max(kCtxRows, cap_L_)), TP = qi_attn_rows(hw), Rk = std::max(R, L);
    mlp_rows_ = mlp_cap_ > 0 ? std::min(R, mlp_cap_) : R;
    x_ = alloc<float>(allocs_, R * C_, "image engine (Anima DiT)");
    q_ = alloc<float>(allocs_, Rk * C_, "image engine (Anima DiT)");
    k_ = alloc<float>(allocs_, Rk * C_, "image engine (Anima DiT)");
    v_ = alloc<float>(allocs_, Rk * C_, "image engine (Anima DiT)");
    a_ = alloc<float>(allocs_, Rk * C_, "image engine (Anima DiT)");
    o32_ = alloc<float>(allocs_, R * C_, "image engine (Anima DiT)");
    g_ = alloc<float>(allocs_, mlp_rows_ * F_, "image engine (Anima DiT)");
    h16_ = alloc<uint16_t>(allocs_, std::max(Rk * C_, mlp_rows_ * F_), "image engine (Anima DiT)");
    rs_ = alloc<float>(allocs_, std::max(Rk, mlp_rows_), "image engine (Anima DiT)");
    rs2_ = alloc<float>(allocs_, std::max(Rk, mlp_rows_), "image engine (Anima DiT)");
    rs_o_ = alloc<float>(allocs_, Rk, "image engine (Anima DiT)");
    hs_ = alloc<float>(allocs_, heads_, "image engine (Anima DiT)");
    vpart_ = alloc<float>(allocs_, std::max(hw * heads_, kColSlices * C_), "image engine (Anima DiT)");   // head_max's, col_max's
    // the int8 attention's statistics: a context's V scales, K's mean a block and context (zero until a step makes it)
    vsv_ = alloc<float>(allocs_, C_, "image engine (Anima DiT)");
    kmean_.init(alloc<float>(allocs_, (int64_t) blocks_.size() * 2 * C_, "image engine (Anima DiT)"),
                (int64_t) blocks_.size() * 2, C_);
    gpu::memset_async(kmean_.buf, 0, (size_t) blocks_.size() * 2 * C_ * 4, s_);
    if (w8_) {
        DitI8Input& in = prod_.in;
        in.rows = std::max(Rk, mlp_rows_);
        in.cols = std::max(C_, F_);
        in.q = alloc<int8_t>(allocs_, in.rows * in.cols + 2 * in.rows * 4, "image engine (Anima DiT)");
    }
    jrows_ = TP;
    jq16_ = alloc<uint16_t>(allocs_, TP * C_, "image engine (Anima DiT)", true);
    jk16_ = alloc<uint16_t>(allocs_, TP * C_, "image engine (Anima DiT)", true);
    jv16_ = alloc<uint16_t>(allocs_, TP * C_, "image engine (Anima DiT)", true);
    jo16_ = alloc<uint16_t>(allocs_, TP * C_, "image engine (Anima DiT)");
    o16_ = alloc<uint16_t>(allocs_, Rk * C_, "image engine (Anima DiT)");
    qh16_ = alloc<uint16_t>(allocs_, Rk * C_, "image engine (Anima DiT)");
    sc_ = alloc<float>(allocs_, std::max(hw, L) * L, "image engine (Anima DiT)");
    p16_ = alloc<uint16_t>(allocs_, std::max(hw, L) * L, "image engine (Anima DiT)");
    cos_ = alloc<float>(allocs_, hw * (hd_ / 2), "image engine (Anima DiT)");
    sin_ = alloc<float>(allocs_, hw * (hd_ / 2), "image engine (Anima DiT)");
    rope_h_ = rope_w_ = -1;
    cap_hw_ = hw;
    cap_n_ = n;
    dirty_[0] = dirty_[1] = 0;
}

void AnimaDit::drop_activations() {
    gpu::stream_sync(s_);
    for (void* p : {(void*) x_, (void*) q_, (void*) k_, (void*) v_, (void*) a_, (void*) o32_, (void*) g_, (void*) h16_,
                    (void*) rs_, (void*) rs2_, (void*) rs_o_, (void*) hs_, (void*) vpart_, (void*) prod_.in.q,
                    (void*) vsv_, (void*) kmean_.buf,
                    (void*) jq16_, (void*) jk16_, (void*) jv16_, (void*) jo16_, (void*) o16_, (void*) qh16_,
                    (void*) sc_, (void*) p16_, (void*) cos_, (void*) sin_})
        release(allocs_, p);
    prod_.lora.drop_buffers(allocs_);
    x_ = q_ = k_ = v_ = a_ = o32_ = g_ = rs_ = rs2_ = rs_o_ = hs_ = vpart_ = sc_ = cos_ = sin_ = vsv_ = nullptr;
    kmean_.clear();
    h16_ = jq16_ = jk16_ = jv16_ = jo16_ = o16_ = qh16_ = p16_ = nullptr;
    prod_.in.q = nullptr;
    cap_hw_ = cap_n_ = 0;
    rope_h_ = rope_w_ = -1;
    drop_kv();
}

void AnimaDit::drop_kv() {
    gpu::stream_sync(s_);
    for (Ctx& c : ctx_)
        for (uint16_t*& p : c.kv) {
            release(allocs_, p);
            p = nullptr;
        }
}

// The image tokens' RoPE (VideoRopePosition3DEmb with one frame): pairs 0 .. 21 by the time (0: not turned), 22 .. 42
// by the row, 43 .. 63 by the column, each axis's frequencies 10000^(-2k / its width) (widths 44, 42, 42 of the 128)
void anima_rope_tables(int64_t h, int64_t w, int64_t hd, std::vector<float>& c, std::vector<float>& s) {
    const int64_t half = hd / 2, dh = hd / 6 * 2, dt = hd - 2 * dh, nt = dt / 2, nh = dh / 2;
    std::vector<float> fr((size_t) half);
    for (int64_t j = 0; j < half; ++j) {
        const bool t = j < nt;
        const int64_t k = t ? j : (j - nt) % nh;
        const float range = (float) (2 * k) / (float) (t ? dt : dh);
        fr[(size_t) j] = 1.f / std::pow(10000.f, range);
    }
    c.assign((size_t) (h * w * half), 0.f);
    s.assign(c.size(), 0.f);
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x)
            for (int64_t j = 0; j < half; ++j) {
                const float pos = j < nt ? 0.f : j < nt + nh ? (float) y : (float) x;
                const float ph = pos * fr[(size_t) j];
                c[(size_t) ((y * w + x) * half + j)] = std::cos(ph);
                s[(size_t) ((y * w + x) * half + j)] = std::sin(ph);
            }
}

void AnimaDit::rope_tables(int64_t h, int64_t w) {
    if (rope_h_ == h && rope_w_ == w) return;
    std::vector<float> c, s;
    anima_rope_tables(h, w, hd_, c, s);
    gpu::copy(cos_, c.data(), c.size() * 4);
    gpu::copy(sin_, s.data(), s.size() * 4);
    rope_h_ = h;
    rope_w_ = w;
}

// the timestep embedder: the sinusoid of sigma rounded to bf16, its two layers (the AdaLN-LoRA term) and its RMS norm
// (the blocks' embedding, whose SiLU every modulation reads)
void anima_time_rows(float sigma, const std::vector<float>& tnorm, std::vector<uint16_t>& h) {
    const auto C = (int64_t) tnorm.size(), half = C / 2;
    std::vector<float> e((size_t) C);
    for (int64_t i = 0; i < half; ++i) {
        const float ex = std::exp(-std::log(10000.f) * (float) i / (float) half);
        const float ph = sigma * ex;
        e[(size_t) i] = round_bf16(std::cos(ph));
        e[(size_t) (half + i)] = round_bf16(std::sin(ph));
    }
    double ss = 0.0;
    for (const float v : e) ss += (double) v * v;
    const float inv = 1.f / std::sqrt((float) (ss / (double) C) + kEps);
    h.assign((size_t) (2 * C), 0);
    for (int64_t i = 0; i < C; ++i) {
        h[(size_t) i] = f16_from_f32(e[(size_t) i]);
        const float y = round_bf16(e[(size_t) i] * inv * tnorm[(size_t) i]);
        h[(size_t) (C + i)] = f16_from_f32(y / (1.f + std::exp(-y)));
    }
}

void AnimaDit::time_embedding(float sigma) {
    std::vector<uint16_t> h;
    anima_time_rows(sigma, tnorm_host_, h);
    uint16_t* s16 = m16_;   // the sinusoid, then the embedding's SiLU in e16_
    gpu::copy_async(s16, h.data(), (size_t) C_ * 2, s_);
    gpu::copy_async(e16_, h.data() + C_, (size_t) C_ * 2, s_);
    gpu::stream_sync(s_);
    product(s16, nullptr, t1_, nullptr, mh_, 1, nullptr);
    qi_silu(mh_, C_, m16_, s_);
    product(m16_, nullptr, t2_, nullptr, alora_, 1, nullptr);
}

void AnimaDit::modulation(const W& w1, const W& w2, const uint8_t* base, float* out) {
    product(e16_, nullptr, w1, base, mh_, 1, nullptr);
    kernels::f32_to_f16_bulk(mh_, m16_, w1.n, s_);
    product(m16_, nullptr, w2, base, out, 1, nullptr);
    add_rows(out, alora_, w2.n, s_);
}

// The cross-attention's keys and values for the products form: K [heads][Tp][hd] normed a head, V^T [heads][hd][Tp],
// the rows past Tk zero (the products read Tp of them)
void AnimaDit::attn_kv(const DitOut& k, const DitOut& v, int64_t Tk, int64_t heads, int64_t hd, const float* wk,
                       uint16_t* kv) {
    const int64_t Tp = qi_attn_rows(Tk);
    gpu::memset_async(kv, 0, (size_t) (2 * heads * Tp * hd) * 2, s_);
    heads16(k, Tk, Tp, heads, hd, wk, nullptr, nullptr, kv, s_, false);
    values_t16(v, Tk, Tp, heads, hd, kv + heads * Tp * hd, s_);
}

// The products form: a head at a time, the scores, their softmax and the weighted values through the shared products,
// out [Tq][C] FP32.  Kept for a GPU without FP16 matrix engines, where the shared attention's FMA form measured slower
// (Anima, XESTRATA_NO_XMX=1, warm then one run, 2026-11-11: the B70 0.132 -> 0.157 s/step at 512x512 and 1.044 ->
// 1.146 at 1024x1024 over 16 steps, the RTX 4070 0.170 -> 0.184 and 1.060 -> 1.120).  With matrix engines the shared
// entry takes it (qi_attention_cross): the maker's kernel 0.21 ms against the engine's own 0.98 at 4096 rows of 512
// keys on the B70, and the products a head at a time took 51 ms of a 421 ms step there (2026-11-02).
void AnimaDit::attn_q(const DitOut& q, int64_t Tq, int64_t heads, int64_t hd, const float* wq, const uint16_t* kv,
                      int64_t Tk, float* out) {
    const int64_t Tp = qi_attn_rows(Tk), C = heads * hd;
    heads16(q, Tq, Tq, heads, hd, wq, nullptr, nullptr, qh16_, s_, false);
    const float scale = 1.f / std::sqrt((float) hd);
    for (int64_t h = 0; h < heads; ++h) {
        gemm_.f16(qh16_ + h * Tq * hd, kv + h * Tp * hd, sc_, Tq, Tp, hd);
        qi_softmax16(sc_, Tq, Tk, scale, p16_, s_, Tp);
        gemm_.f16(p16_, kv + heads * Tp * hd + h * hd * Tp, out + h * hd, Tq, hd, Tp, C);
    }
}

// The LLM adapter of the T5 ids over the states, its rows times their weights,
// into each context's FP16 rows padded with zero rows to 512.  The contexts go through it as one run of rows (each
// attending to its own), so that a guided request's two cost the launches of one.
void AnimaDit::adapter(const std::vector<AdapterIn>& in) {
    const int64_t C = ad_C_;
    std::vector<int64_t> lt, lq;
    int64_t LT = 0, LQ = 0;
    for (const AdapterIn& a : in) {
        if (a.t5->empty() || a.rows <= 0) throw std::invalid_argument("image engine (Anima): an empty prompt");
        lt.push_back((int64_t) a.t5->size());
        lq.push_back(a.rows);
        LT += lt.back();
        LQ += lq.back();
        cap_L_ = std::max(cap_L_, std::max(kCtxRows, lt.back()));
    }
    // the second context's first rows (the first's when there is one)
    const int64_t t0 = in.size() > 1 ? lt[0] : LT, q0 = in.size() > 1 ? lq[0] : LQ;
    // the working buffers, carved out of one region kept for the next prompt (256-byte aligned pieces)
    const int64_t W = std::max(C, ad_F_), RT = std::max<int64_t>(64, std::max(LT, LQ)), half = ad_hd_ / 2;
    const uint64_t sizes[] = {(uint64_t) (RT * C) * 4,     (uint64_t) (RT * src_dim_) * 4, (uint64_t) (RT * src_dim_) * 2,
                              (uint64_t) (RT * W) * 2,     (uint64_t) (RT * C) * 4 * 3,    (uint64_t) (RT * W) * 4,
                              (uint64_t) (3 * RT * C) * 2, (uint64_t) RT * 4,              (uint64_t) (4 * RT * half) * 4};
    uint64_t need = 0;
    for (const uint64_t b : sizes) need += (b + 255) / 256 * 256;
    if (need > ad_bytes_) {
        gpu::stream_sync(s_);
        release(text_allocs_, ad_scratch_);
        ad_scratch_ = alloc<uint8_t>(text_allocs_, (int64_t) need, "image engine (Anima DiT)");
        ad_bytes_ = need;
    }
    uint8_t* at = ad_scratch_;
    auto take = [&](uint64_t b) { uint8_t* p = at; at += (b + 255) / 256 * 256; return p; };
    float* xa = reinterpret_cast<float*>(take(sizes[0]));
    float* src = reinterpret_cast<float*>(take(sizes[1]));
    uint16_t* s16 = reinterpret_cast<uint16_t*>(take(sizes[2]));
    uint16_t* x16 = reinterpret_cast<uint16_t*>(take(sizes[3]));
    float* q = reinterpret_cast<float*>(take(sizes[4]));
    float *k = q + RT * C, *v = k + RT * C;
    float* o = reinterpret_cast<float*>(take(sizes[5]));
    uint16_t* qh = reinterpret_cast<uint16_t*>(take(sizes[6]));
    uint16_t* kh = qh + RT * C;
    uint16_t* vh = kh + RT * C;
    float* wrow = reinterpret_cast<float*>(take(sizes[7]));
    float* ropes = reinterpret_cast<float*>(take(sizes[8]));
    // the ids' embedding rows from the file, their weights and the states, each context's after the other's
    {
        const SafeTensor& e = file_->get("llm_adapter.embed.weight");
        const uint8_t* base = file_->raw(e);
        // the one host lookup of the text models (native_row_to_float), by the safetensors dtype's ggml type
        const int type = e.dtype == "BF16" ? 30 : e.dtype == "F16" ? 1 : e.dtype == "F32" ? 0 : -1;
        const size_t row_bytes = type < 0 ? 0 : kernels::cpu::native_row_bytes(type, C);
        if (row_bytes == 0) throw std::runtime_error("image engine (Anima DiT): the adapter's embedding is " + e.dtype);
        std::vector<float> rowsf((size_t) (LT * C)), w((size_t) LT), st((size_t) (LQ * src_dim_));
        int64_t t = 0, r = 0;
        for (const AdapterIn& a : in) {
            for (size_t i = 0; i < a.t5->size(); ++i, ++t) {
                const int64_t id = (*a.t5)[i];
                if (id < 0 || id >= vocab_)
                    throw std::invalid_argument("image engine (Anima): a T5 id outside the vocabulary");
                w[(size_t) t] = (*a.t5w)[i];
                if (!kernels::cpu::native_row_to_float(type, base + (size_t) id * row_bytes, rowsf.data() + t * C, C))
                    throw std::runtime_error("image engine (Anima DiT): the adapter's embedding row could not be read");
            }
            std::memcpy(st.data() + r * src_dim_, a.states, (size_t) (a.rows * src_dim_) * 4);
            r += a.rows;
        }
        std::vector<float> ct, sn, cc, sc;
        rope_1d(lt, half, ct, sn);
        rope_1d(lq, half, cc, sc);
        gpu::copy_async(xa, rowsf.data(), rowsf.size() * 4, s_);
        gpu::copy_async(wrow, w.data(), w.size() * 4, s_);
        gpu::copy_async(src, st.data(), st.size() * 4, s_);
        gpu::copy_async(ropes, ct.data(), ct.size() * 4, s_);
        gpu::copy_async(ropes + RT * half, sn.data(), sn.size() * 4, s_);
        gpu::copy_async(ropes + 2 * RT * half, cc.data(), cc.size() * 4, s_);
        gpu::copy_async(ropes + 3 * RT * half, sc.data(), sc.size() * 4, s_);
        gpu::stream_sync(s_);   // the host vectors go
    }
    const float *dct = ropes, *dst = ropes + RT * half, *dcc = ropes + 2 * RT * half, *dsc = ropes + 3 * RT * half;
    const float scale = 1.f / std::sqrt((float) ad_hd_);
    // the attention of the rows' q over k and v, each context's over its own
    const auto attend = [&](const DitOut& mq, const float* wq, const float* cq, const float* sq, const DitOut& mk,
                            const DitOut& mv, const float* wk, const float* ck, const float* sk, int64_t Tk, int64_t k0) {
        heads16(mq, LT, LT, ad_heads_, ad_hd_, wq, cq, sq, qh, s_, true);
        heads16(mk, Tk, Tk, ad_heads_, ad_hd_, wk, ck, sk, kh, s_, true);
        values16(mv, Tk, C, vh, s_);
        // the shared attention of the auxiliary models (qi_sdpa_ex): the GPU maker's kernel where it is the faster,
        // else the engine's own.  Its FP16 rows are the next product's input.
        QiSdpa a;
        a.q = qh;
        a.k = kh;
        a.v = vh;
        a.out = x16;
        a.heads = ad_heads_;
        a.hd = ad_hd_;
        a.q_row = a.k_row = a.v_row = a.o_row = C;
        a.scale = scale;
        if (in.size() > 1) {   // each context's rows over its own keys: an attention each
            a.lq = t0;
            a.lk = k0;
            qi_sdpa_ex(a, s_);
            a.q = qh + t0 * C;
            a.k = kh + k0 * C;
            a.v = vh + k0 * C;
            a.out = x16 + t0 * C;
            a.lq = LT - t0;
            a.lk = Tk - k0;
        } else {
            a.lq = LT;
            a.lk = Tk;
        }
        qi_sdpa_ex(a, s_);
    };
    rows16_plain(Plain{src, src_dim_}, LQ, src_dim_, s16, s_);
    for (const AdapterBlock& b : ablocks_) {
        // self-attention among the ids
        rows16_plain(RmsW{xa, b.n_sa, nullptr, C}, LT, C, x16, s_);
        const DitOut mq = product(x16, nullptr, b.sq, nullptr, q, LT, nullptr);
        const DitOut mk = product(x16, nullptr, b.sk, nullptr, k, LT, nullptr);
        const DitOut mv = product(x16, nullptr, b.sv, nullptr, v, LT, nullptr);
        attend(mq, b.sqn, dct, dst, mk, mv, b.skn, dct, dst, LT, t0);
        product(x16, nullptr, b.so, nullptr, q, LT, nullptr);
        add_rows(xa, q, LT * C, s_);
        // cross-attention to the states, each side rotated by its own positions
        rows16_plain(RmsW{xa, b.n_ca, nullptr, C}, LT, C, x16, s_);
        const DitOut cq = product(x16, nullptr, b.cq, nullptr, q, LT, nullptr);
        const DitOut ck = product(s16, nullptr, b.ck, nullptr, k, LQ, nullptr);
        const DitOut cv = product(s16, nullptr, b.cv, nullptr, v, LQ, nullptr);
        attend(cq, b.cqn, dct, dst, ck, cv, b.ckn, dcc, dsc, LQ, q0);
        product(x16, nullptr, b.co, nullptr, q, LT, nullptr);
        add_rows(xa, q, LT * C, s_);
        // the MLP
        rows16_plain(RmsW{xa, b.n_ml, nullptr, C}, LT, C, x16, s_);
        const DitOut h = product(x16, nullptr, b.m0, nullptr, o, LT, b.bm0);
        rows16_plain(GeluErf{h}, LT, ad_F_, x16, s_);
        product(x16, nullptr, b.m2, nullptr, q, LT, b.bm2);
        add_rows(xa, q, LT * C, s_);
    }
    rows16_plain(Plain{xa, C}, LT, C, x16, s_);
    product(x16, nullptr, aout_, nullptr, q, LT, b_aout_);
    // each context's rows: the norm times the ids' weights, then the zero rows
    gpu::stream_sync(s_);
    int64_t t = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        Ctx& c = *in[i].c;
        const int64_t L = std::max(kCtxRows, lt[i]);
        if (c.c16 == nullptr || c.rows != L) {
            release(allocs_, c.c16);
            release(allocs_, c.rs);
            c.c16 = alloc<uint16_t>(allocs_, L * C, "image engine (Anima DiT)", true);
            c.rs = alloc<float>(allocs_, L, "image engine (Anima DiT)");
            std::vector<float> ones((size_t) L, 1.f);
            gpu::copy(c.rs, ones.data(), ones.size() * 4);
        } else {   // the zero rows after these, at a scale of 1 as alloc left them
            gpu::memset_async(c.c16 + lt[i] * C, 0, (size_t) ((L - lt[i]) * C) * 2, s_);
            std::vector<float> ones((size_t) L, 1.f);
            gpu::copy(c.rs, ones.data(), ones.size() * 4);
        }
        rows16(RmsW{q + t * C, anorm_, wrow + t, C}, lt[i], C, c.c16, c.rs, s_);
        c.rows = L;
        t += lt[i];
    }
    gpu::stream_sync(s_);
}

void AnimaDit::set_context(int i, const float* states, int64_t rows, const std::vector<int32_t>& t5,
                           const std::vector<float>& t5_weights) {
    set_contexts({{i, states, rows, &t5, &t5_weights}});
}

void AnimaDit::set_contexts(const std::vector<ContextIn>& in) {
    gpu::stream_sync(s_);
    std::vector<AdapterIn> run;
    for (const ContextIn& e : in) {
        Ctx& c = ctx_[e.i];
        std::fill(c.kv_ok.begin(), c.kv_ok.end(), (char) 0);
        if (e.rows <= 0) {
            for (uint16_t*& p : c.kv) {
                release(allocs_, p);
                p = nullptr;
            }
            release(allocs_, c.c16);
            release(allocs_, c.rs);
            c.c16 = nullptr;
            c.rs = nullptr;
            c.rows = 0;
            continue;
        }
        if (e.t5->size() != e.t5w->size())
            throw std::invalid_argument("image engine (Anima): a T5 weight for each id");
        run.push_back({&c, e.states, e.rows, e.t5, e.t5w});
    }
    if (run.empty()) return;
    if (!text_loaded_) load_text();
    adapter(run);
    bool longer = false;
    for (const AdapterIn& a : run) longer = longer || qi_attn_rows(a.c->rows) > qi_attn_rows(kCtxRows);
    if (longer && cap_hw_ > 0) {   // a longer context: the scores' buffers again
        const int64_t hw = cap_hw_, n = cap_n_;
        drop_activations();
        ensure(hw, (int) n);
    }
}

// A block's weights as each copy of it reaches the GPU: FP16 from the file's bf16 (the FP16 blocks), and the adapters
// baked in (set_loras); none of it for 8-bit blocks without baked adapters (the stream then copies them as they are)
void AnimaDit::install_prepare() {
    if (q8_ && !loras_.baked_any()) {
        blocks_stream_.set_prepare(nullptr);
        return;
    }
    blocks_stream_.set_prepare([this](int i, uint8_t* d) {
        sycl::queue& q = q_of(s_);
        const Block& b = blocks_[(size_t) i];
        for (const W* w : {&b.sq, &b.sk, &b.sv, &b.so, &b.cq, &b.ck, &b.cv, &b.co, &b.l1, &b.l2, &b.msa1, &b.msa2,
                           &b.mca1, &b.mca2, &b.mml1, &b.mml2}) {
            uint8_t* wd = d + reinterpret_cast<uintptr_t>(w->d);
            if (!q8_) bf16_to_f16_inplace(reinterpret_cast<uint16_t*>(wd), w->n * w->k, q);
            if (w->baked) w->lora->bake(w->type, wd, w->n, w->k, gemm_, loras_.bake_tmp(), loras_.bake_at16(), s_);
        }
    });
}

void AnimaDit::forward(const float* x, float sigma, int64_t h, int64_t w, int n, float* out,
                       const std::vector<int>* skip) {
    const int64_t HW = h * w, C = C_, R = n * HW, TV = 4 * lat_;
    for (int i = 0; i < n; ++i)
        if (ctx_[i].rows <= 0) throw std::runtime_error("image engine (Anima DiT): a context not set");
    ensure(HW, n);
    rope_tables(h, w);
    // the patches through the embedder (each context's image rows the same at the start)
    kernels::f32_to_f16_bulk(x, h16_, HW * TV, s_);
    product(h16_, nullptr, xemb_, nullptr, x_, HW, nullptr);
    for (int i = 1; i < n; ++i) gpu::copy_async(x_ + i * HW * C, x_, (size_t) (HW * C) * 4, s_);
    time_embedding(sigma);
    if (sigma > last_sigma_) kmean_.reset();   // a new image: K's means made again
    last_sigma_ = sigma;
    // the cross-attention's form, once: the shared entry where the GPU has FP16 matrix engines, the products a head
    // at a time where it has none (the shared kernel's FMA form measured slower there, attn_q's numbers)
    if (cross_form_ < 0) {
        const sycl::device dev = q_of(s_).get_device();
        cross_form_ = kernels::flash_attn_path(&dev) < 0 ? 1 : 0;
    }
    const bool products = cross_form_ == 1;
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        if (skip != nullptr && std::find(skip->begin(), skip->end(), bi) != skip->end()) continue;
        const Block& b = blocks_[(size_t) bi];
        const uint8_t* base = blocks_stream_.begin(bi);
        auto f = [base](uint64_t o) { return reinterpret_cast<const float*>(base + o); };
        // self-attention, each context's image rows among themselves
        modulation(b.msa1, b.msa2, base, mod_);
        rows16(LnMod{x_, mod_, mod_ + C, C}, R, C, h16_, rs_, s_);
        const DitOut q = product(h16_, rs_, b.sq, base, q_, R, nullptr, false, true);
        const DitOut k = product(h16_, rs_, b.sk, base, k_, R, nullptr, b.sq.type == kTypeW8, true);
        const DitOut v = product(h16_, rs_, b.sv, base, v_, R, nullptr, b.sk.type == kTypeW8, true);
        for (int i = 0; i < n; ++i) {
            const int64_t r0 = i * HW;
            if (dirty_[0] > HW)
                for (uint16_t* jb : {jq16_, jk16_, jv16_})
                    gpu::memset_async(jb + HW * C, 0, (size_t) ((dirty_[0] - HW) * C) * 2, s_);
            dirty_[0] = HW;
            // v's heads' scales by column, the int8 attention's V scales with them and K's mean kept from the last step
            col_max(v.from(r0), HW, DitOut{}, 0, C, vpart_, s_);
            col_scale(vpart_, C, heads_, hd_, hs_, vsv_, s_);
            qk_rope16(q.from(r0), k.from(r0), v.from(r0), f(b.sqn), f(b.skn), cos_, sin_, hs_, HW, heads_, hd_, true,
                      jq16_, jk16_, jv16_, s_);
            // (two contexts at most kept: a third makes its own mean)
            const int64_t slot = i < 2 ? (int64_t) bi * 2 + i : kmean_.slots;
            const auto st = kmean_.stats(slot, vsv_);
            qi_attention(jq16_, jk16_, jv16_, HW, heads_, hd_, 0, jo16_, s_, nullptr, 0, true, nullptr, &st);
            kmean_.mark(slot);
            rows16(Heads{jo16_, hs_, C, hd_}, HW, C, o16_ + r0 * C, rs_o_ + r0, s_);
        }
        gate_add(x_, product(o16_, rs_o_, b.so, base, a_, R, nullptr, false, true), mod_ + 2 * C, R, C, s_);
        // cross-attention to each context's rows (their keys and values made at the first step and kept, before the
        // queries: their products' input scales would take the deferred queries' slot)
        for (int i = 0; i < n; ++i) {
            Ctx& c = ctx_[i];
            const int64_t L = c.rows, Lp = qi_attn_rows(L);
            uint16_t*& kv = c.kv[(size_t) bi];
            if (kv != nullptr && c.kv_lp[(size_t) bi] != Lp) {   // a longer or shorter context than it was made for
                gpu::stream_sync(s_);
                release(allocs_, kv);
                kv = nullptr;
            }
            if (kv == nullptr) {
                kv = alloc<uint16_t>(allocs_, 2 * heads_ * Lp * hd_, "image engine (Anima DiT)");
                // the rows past L zero (the shared kernel reads whole chunks of keys, the products form whole Tp):
                // async, as a synchronous memset here stalled the step's queue once a block (28 of them, +0.11 s)
                gpu::memset_async(kv, 0, (size_t) (2 * heads_ * Lp * hd_) * 2, s_);
                c.kv_lp[(size_t) bi] = Lp;
                c.kv_ok[(size_t) bi] = 0;
            }
            if (!c.kv_ok[(size_t) bi]) {
                const DitOut ck = product(c.c16, c.rs, b.ck, base, k_, L, nullptr);
                const DitOut cv = product(c.c16, c.rs, b.cv, base, v_, L, nullptr, b.ck.type == kTypeW8);
                if (products) {
                    attn_kv(ck, cv, L, heads_, hd_, f(b.ckn), kv);
                } else {
                    // the shared attention's rows: K [Lp][C] normed a head, V [Lp][C] (its own kernel reads the keys
                    // in whole chunks of 64, the rows past L masked)
                    heads16(ck, L, Lp, heads_, hd_, f(b.ckn), nullptr, nullptr, kv, s_, true);
                    values16(cv, L, C, kv + Lp * C, s_);
                }
                c.kv_ok[(size_t) bi] = 1;
            }
        }
        modulation(b.mca1, b.mca2, base, mod_);
        rows16(LnMod{x_, mod_, mod_ + C, C}, R, C, h16_, rs_, s_);
        const DitOut cq = product(h16_, rs_, b.cq, base, q_, R, nullptr, false, true);
        if (products) {   // each context's rows through the products, a head at a time (attn_q's numbers)
            for (int i = 0; i < n; ++i)
                attn_q(cq.from(i * HW), HW, heads_, hd_, f(b.cqn), ctx_[i].kv[(size_t) bi], ctx_[i].rows,
                       o32_ + i * HW * C);
            rows16(Plain{o32_, C}, R, C, o16_, rs_o_, s_);
            gate_add(x_, product(o16_, rs_o_, b.co, base, a_, R, nullptr, false, true), mod_ + 2 * C, R, C, s_);
        } else {
            // each context's rows through the shared cross-attention (qi_attention_cross): the GPU maker's kernel
            // where it is the faster, else the engine's own on the matrix engines
            for (int i = 0; i < n; ++i) {
                const Ctx& c = ctx_[i];
                heads16(cq.from(i * HW), HW, HW, heads_, hd_, f(b.cqn), nullptr, nullptr, jq16_, s_, true);
                const uint16_t* kv = c.kv[(size_t) bi];
                qi_attention_cross(jq16_, kv, kv + c.kv_lp[(size_t) bi] * C, HW, c.rows, heads_, hd_,
                                   o16_ + i * HW * C, s_);
            }
            gate_add(x_, product(o16_, nullptr, b.co, base, a_, R, nullptr, false, true), mod_ + 2 * C, R, C, s_);
        }
        // the MLP, a band of rows at a time
        modulation(b.mml1, b.mml2, base, mod_);
        for (int64_t m0 = 0; m0 < R; m0 += mlp_rows_) {
            const int64_t mr = std::min(mlp_rows_, R - m0);
            rows16(LnMod{x_ + m0 * C, mod_, mod_ + C, C}, mr, C, h16_, rs_, s_);
            rows16(GeluErf{product(h16_, rs_, b.l1, base, g_, mr, nullptr, false, true)}, mr, F_, h16_, rs2_, s_);
            gate_add(x_ + m0 * C, product(h16_, rs2_, b.l2, base, a_, mr, nullptr, false, true), mod_ + 2 * C, mr, C,
                     s_);
        }
        blocks_stream_.end(bi);
    }
    // the final layer: each context's rows normed and modulated, projected to the patch's values
    modulation(fm1_, fm2_, nullptr, fmod_);
    for (int i = 0; i < n; ++i) {
        rows16(LnMod{x_ + i * HW * C, fmod_, fmod_ + C, C}, HW, C, h16_, rs_, s_);
        product(h16_, rs_, flin_, nullptr, a_, HW, nullptr);
        patch_to_tokens(a_, HW, lat_, out + i * HW * TV, s_);
    }
}

std::vector<AnimaDit::LoraTarget> AnimaDit::lora_targets() const { return loras_.targets(); }

void AnimaDit::build_lora_targets() {
    loras_.clear_slots();
    auto add = [this](std::string name, W& w, bool biased) { loras_.add(std::move(name), w, biased); };
    const char* const* out = lora::kAnimaOutside;
    add(out[0], xemb_, false);
    add(out[1], t1_, false);
    add(out[2], t2_, false);
    add(out[3], flin_, false);
    add(out[4], fm1_, false);
    add(out[5], fm2_, false);
    const char* const* ad = lora::kAnimaAdapter;
    for (int i = 0; i < (int) ablocks_.size(); ++i) {
        const std::string p = "llm_adapter.blocks." + std::to_string(i) + ".";
        AdapterBlock& b = ablocks_[(size_t) i];
        add(p + ad[0], b.sq, false);
        add(p + ad[1], b.sk, false);
        add(p + ad[2], b.sv, false);
        add(p + ad[3], b.so, false);
        add(p + ad[4], b.cq, false);
        add(p + ad[5], b.ck, false);
        add(p + ad[6], b.cv, false);
        add(p + ad[7], b.co, false);
        add(p + ad[8], b.m0, true);
        add(p + ad[9], b.m2, true);
    }
    add(out[6], aout_, true);
    const char* const* n = lora::kAnimaBlock;
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        const std::string p = "blocks." + std::to_string(bi) + ".";
        Block& b = blocks_[(size_t) bi];
        add(p + n[0], b.sq, false);
        add(p + n[1], b.sk, false);
        add(p + n[2], b.sv, false);
        add(p + n[3], b.so, false);
        add(p + n[4], b.cq, false);
        add(p + n[5], b.ck, false);
        add(p + n[6], b.cv, false);
        add(p + n[7], b.co, false);
        add(p + n[8], b.l1, false);
        add(p + n[9], b.l2, false);
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 2; ++j) {
                W* m = k == 0 ? (j == 0 ? &b.msa1 : &b.msa2)
                              : k == 1 ? (j == 0 ? &b.mca1 : &b.mca2) : (j == 0 ? &b.mml1 : &b.mml2);
                add(p + lora::kAnimaMod[k] + (j == 0 ? ".1" : ".2"), *m, false);
            }
    }
}

bool AnimaDit::set_loras(const std::vector<LoraUse>& loras, std::string& err) {
    bool changed = false;
    const bool ok = loras_.set(loras, err, changed);
    if (changed) drop_kv();   // the kept keys and values were the other adapters' weights'
    return ok;
}

bool AnimaDit::dora_norms(const LoraSlot& slot, const LoraEntry& e, std::vector<float>& out, std::string& err) {
    const std::string name = slot.name + ".weight";
    try {
        const SafeTensor& t = file_->get(name);
        const int64_t rows = (int64_t) t.shape[0], k = (int64_t) t.shape[1];
        if (slot.src_row + slot.row0 + slot.rows > rows)
            throw std::runtime_error(name + ": the file's rows are fewer than the weight's");
        std::vector<uint16_t> tmp;
        const uint16_t* h = file_->bf16(name, tmp);
        const bool ok = dit_dora_norms(e, slot.src_row, slot.src_row + slot.row0, slot.rows, k,
                                       dit_bf16_rows(h, k), out, err);
        file_->drop_cache(name);
        return ok;
    } catch (const std::exception& ex) {
        err = std::string("image engine (Anima DiT): a DoRA's row norms: ") + ex.what();
        return false;
    }
}

uint64_t AnimaDit::lora_bytes() const { return loras_.bytes(); }

void AnimaDit::drop_loras() { loras_.drop(); }

}  // namespace strata::image
