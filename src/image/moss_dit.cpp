// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_dit.cpp - see include/strata/image/moss_dit.hpp.

#include "strata/image/moss_dit.hpp"

#include "dit_rows.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mapped_file.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/i8_gemm.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/platform/env.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace moss_dit_detail {

using namespace dit_rows;

// kTypeW8: int8 rows rotated as qi_rows_i8 rotates them, then a scale a row (FP32), for prefill::Gemm::i8
constexpr int kTypeF16 = 1, kTypeQ8_0 = 8, kTypeW8 = -8;
constexpr float kEps = 1e-6f;

template<class T> T* alloc(std::vector<void*>& owned, int64_t n, bool zero = false) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) std::max<int64_t>(n, 1) * sizeof(T)))
        throw std::runtime_error(std::string("image engine (MOSS-Image DiT): GPU memory: ") + gpu::last_error());
    if (zero && !gpu::memset(p, 0, (size_t) std::max<int64_t>(n, 1) * sizeof(T)))
        throw std::runtime_error(std::string("image engine (MOSS-Image DiT): memset: ") + gpu::last_error());
    owned.push_back(p);
    return p;
}

void release(std::vector<void*>& owned, void* p) {
    if (p == nullptr) return;
    owned.erase(std::remove(owned.begin(), owned.end(), p), owned.end());
    gpu::free(p);
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

// RMSNorm(x[t]) * w (eps 1e-6), then * (1 + scale) + shift where the time gives them (null: not), alt's rows taking
// another set's: dit_rows' RmsAffine
RmsAffineOf<uint16_t> rms_mod(const uint16_t* x, const float* w, const float* shift, const float* scale, int64_t C,
                              const AltRows& alt = {}) {
    return RmsAffineOf<uint16_t>{x, w, scale, shift, C, kEps, true, alt};
}

// x as it is (the text encoder's states, for their FP16 rows' scales)
struct Plain {
    const float* x;
    int64_t C;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const { return x[t * C + c]; }
};

// SiTU of the up_gate product's sums g [T][2F] (the hidden values, then the gates): (4 tanh(g / 4) sigmoid(g)) times
// (25 tanh(h / 25)), F values a row
struct Situ {
    MfOut g;
    int64_t F;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const {
        const float h = g.at(t, c), gt = g.at(t, c + F);
        return 4.f * sycl::tanh(gt * 0.25f) / (1.f + sycl::exp(-gt)) * (25.f * sycl::tanh(h * 0.04f));
    }
    // below 4 x 25 in magnitude: FP16 takes the rows as they are (their scales 1)
    float bound(int64_t, int64_t) const { return 0.f; }
};

// RowsI8Of's ops (a W8A8 product's input without the FP16 rows): rms_mod's rows, after x += gate * add where add is a
// deferred product (gate_add, its sums written back to x); Situ's; the attention's output times its heads' scales
struct RmsI8 {
    static constexpr bool kRms = true, kKeep = true;
    uint16_t* x;   // BF16 (MossDit's x_)
    MfOut add;
    const float *gate, *w, *shift, *scale;
    int64_t C;
    AltRows rr{};   // rms_mod's, the gate's too
    float raw(int64_t t, int64_t c) const {
        const float v = resid(x[t * C + c]);
        return add.C > 0 ? v + gate[c + rr.at(t)] * add.at(t, c) : v;
    }
    // the sum as x keeps it: the norm takes the rounded value, as the next block reads it
    void keep(int64_t t, int64_t c, float& v) const {
        if (add.C == 0) return;
        x[t * C + c] = bf16_bits(v);
        v = resid(x[t * C + c]);
    }
    float inv(float ss) const { return 1.f / sycl::sqrt(ss / (float) C + kEps); }
    float out(int64_t t, int64_t c, float v, float inv) const {
        const float y = v * inv * w[c];
        const int64_t m = c + rr.at(t);
        return scale != nullptr ? y * (1.f + scale[m]) + shift[m] : y;
    }
    // the same W values side by side (RowsI8Of's vector form)
    template<int W> void raw_w(int64_t t, int64_t c, float* v) const {
        const Packed<uint16_t, W> xv = *reinterpret_cast<const Packed<uint16_t, W>*>(x + t * C + c);
        if (add.C > 0) {
            float a[W];
            add.at_w<W>(t, c, a);
            const Packed<float, W> gv = *reinterpret_cast<const Packed<float, W>*>(gate + c);
            for (int p = 0; p < W; ++p) v[p] = resid(xv.v[p]) + gv.v[p] * a[p];
        } else {
            for (int p = 0; p < W; ++p) v[p] = resid(xv.v[p]);
        }
    }
    template<int W> void keep_w(int64_t t, int64_t c, float* v) const {
        if (add.C == 0) return;
        Packed<uint16_t, W> o;
        for (int p = 0; p < W; ++p) {
            o.v[p] = bf16_bits(v[p]);
            v[p] = resid(o.v[p]);
        }
        *reinterpret_cast<Packed<uint16_t, W>*>(x + t * C + c) = o;
    }
    template<int W> void out_w(int64_t, int64_t c, float* v, float inv) const {
        const Packed<float, W> wv = *reinterpret_cast<const Packed<float, W>*>(w + c);
        if (scale != nullptr) {
            const Packed<float, W> sc = *reinterpret_cast<const Packed<float, W>*>(scale + c),
                                   sh = *reinterpret_cast<const Packed<float, W>*>(shift + c);
            for (int p = 0; p < W; ++p) v[p] = v[p] * inv * wv.v[p] * (1.f + sc.v[p]) + sh.v[p];
        } else {
            for (int p = 0; p < W; ++p) v[p] = v[p] * inv * wv.v[p];
        }
    }
};

// gate_add, rr's rows taking their gate alt floats on: the shared kernel over each run of rows of one gate
void gate_add_rows(uint16_t* x, const MfOut& h, const float* gate, const AltRows& rr, int64_t T, int64_t C, void* s) {
    if (rr.alt == 0 || gate == nullptr) {
        gate_add(x, h, gate, T, C, s);
        return;
    }
    for (int64_t t = 0; t < T;) {
        const int64_t a = rr.at(t), at = t + rr.t0;
        // the run's end: the next of rr's bounds past this row
        int64_t e = T;
        for (const int64_t bound : {rr.r0, rr.e0, rr.r1})
            if (bound > at && bound - rr.t0 < e) e = bound - rr.t0;
        gate_add(x + t * C, h.from(t), gate + a, e - t, C, s);
        t = e;
    }
}

struct SituI8 {
    static constexpr bool kRms = false, kKeep = false;
    Situ s;
    float raw(int64_t t, int64_t c) const { return s.value(t, c, Stats{}); }
    void keep(int64_t, int64_t, float) const {}
    float inv(float) const { return 1.f; }
    float out(int64_t, int64_t, float v, float) const { return v; }
    template<int W> void raw_w(int64_t t, int64_t c, float* v) const {
        float h[W], gt[W];
        s.g.at_w<W>(t, c, h);
        s.g.at_w<W>(t, c + s.F, gt);
        for (int p = 0; p < W; ++p)
            v[p] = 4.f * sycl::tanh(gt[p] * 0.25f) / (1.f + sycl::exp(-gt[p])) * (25.f * sycl::tanh(h[p] * 0.04f));
    }
};

struct HeadsI8 {
    static constexpr bool kRms = false, kKeep = false;
    Heads h;
    float raw(int64_t t, int64_t c) const { return h.value(t, c, Stats{}); }
    void keep(int64_t, int64_t, float) const {}
    float inv(float) const { return 1.f; }
    float out(int64_t, int64_t, float v, float) const { return v; }
};

// an edit's references beside a context (MossDit::set_refs): hs[h] = the larger of a context's and the references'
// scale of head h, powers of two both
void hs_max(float* hs, const float* other, int64_t heads, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) heads), [=](sycl::id<1> id) {
        hs[id[0]] = sycl::fmax(hs[id[0]], other[id[0]]);
    });
}

// the references' v16 [R][heads hd], kept at their heads' scales `from`, at the scales `to` (powers of two, `to` the
// larger: the FP16 values divided exactly)
void ref_values(const uint16_t* v16, const float* from, const float* to, int64_t R, int64_t heads, int64_t hd,
                uint16_t* out, void* s) {
    const int64_t C = heads * hd;
    q_of(s).parallel_for(sycl::range<1>((size_t) (R * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t h = (i % C) / hd;
        out[i] = f16_from_f32(f32_from_f16(v16[i]) * (from[h] / to[h]));
    });
}

}  // namespace moss_dit_detail

using namespace moss_dit_detail;

// the transformer's shards (a folder's *.safetensors, or one file), each mapped
struct MossDit::Shards {
    struct Shard {
        std::unique_ptr<SafeTensors> st;
        std::unique_ptr<MappedFile> map;
    };
    std::vector<Shard> s;
    explicit Shards(const std::string& path) {
        std::vector<std::string> files;
        if (std::filesystem::is_directory(path)) {
            for (const auto& e : std::filesystem::directory_iterator(path))
                if (e.path().extension() == ".safetensors") files.push_back(e.path().string());
            std::sort(files.begin(), files.end());
        } else {
            files.push_back(path);
        }
        if (files.empty()) throw std::runtime_error("image engine (MOSS-Image DiT): no .safetensors in " + path);
        for (const auto& f : files) s.push_back({std::make_unique<SafeTensors>(f), std::make_unique<MappedFile>(f)});
    }
    // the tensor and its shard, or null
    const SafeTensor* find(const std::string& name, const Shard** at = nullptr) const {
        for (const Shard& x : s)
            if (const SafeTensor* t = x.st->find(name); t != nullptr) {
                if (at != nullptr) *at = &x;
                return t;
            }
        return nullptr;
    }
    // a bf16 tensor: its bytes in the mapping
    const SafeTensor& bf16(const std::string& name, const uint8_t** data) const {
        const Shard* at = nullptr;
        const SafeTensor* t = find(name, &at);
        if (t == nullptr) throw std::runtime_error("image engine (MOSS-Image DiT): no " + name);
        if (t->dtype != "BF16") throw std::runtime_error("image engine (MOSS-Image DiT): " + name + " is not bf16");
        if (data != nullptr) *data = at->map->data() + at->st->data_base() + t->begin;
        return *t;
    }
    std::vector<float> f32(const std::string& name) const {
        const Shard* at = nullptr;
        if (find(name, &at) == nullptr) throw std::runtime_error("image engine (MOSS-Image DiT): no " + name);
        return at->st->f32(name);
    }
    void drop_cache(const std::string& name) const {
        const Shard* at = nullptr;
        if (const SafeTensor* t = find(name, &at); t != nullptr)
            at->map->drop_cache(at->st->data_base() + t->begin, t->end - t->begin);
    }
};

MossDit::MossDit() = default;

MossDit::~MossDit() {
    blocks_stream_.drop();
    for (void* p : ctl_allocs_) gpu::free(p);
    for (void* p : allocs_) gpu::free(p);
    for (void* p : text_allocs_) gpu::free(p);
}

bool MossDit::open(const std::string& path, void* stream, std::string& err, bool dry, bool q8,
                   std::vector<std::vector<uint8_t>>* reuse) {
    try {
        s_ = stream;
        dry_ = dry;
        q8_ = q8;
        norm_cache_.clear();
        w8_ = q8 && strata::env_get("NO_W8A8") == nullptr && prefill::Gemm::i8_usable(stream);
        files_ = std::make_unique<Shards>(path);
        const Shards& f = *files_;
        if (std::filesystem::is_directory(path)) {   // config.json's rope_base, where it differs from 10000
            std::ifstream cf(std::filesystem::path(path) / "config.json");
            const std::string cfg((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
            if (const size_t at = cfg.find("\"rope_base\""); at != std::string::npos)
                rope_base_ = std::stod(cfg.substr(cfg.find(':', at) + 1));
        }
        vin_ = dense({"video_embedder"}, true, &b_vin_, allocs_, weight_bytes_);
        in_ch_ = vin_.k;
        C_ = vin_.n;
        t1_ = dense({"time_embedding.0"}, true, &b_t1_, allocs_, weight_bytes_);
        freq_ = t1_.k;
        t2_ = dense({"time_embedding.2"}, true, &b_t2_, allocs_, weight_bytes_);
        smod_ = dense({"shared_block_modulation"}, true, &b_smod_, allocs_, weight_bytes_);
        hmod_ = dense({"video_head_modulation"}, true, &b_hmod_, allocs_, weight_bytes_);
        fout_ = dense({"final_linear_video"}, false, nullptr, allocs_, weight_bytes_);
        fnorm_ = vec("final_norm_video.weight", allocs_, weight_bytes_);
        hd_ = f.bf16("layers.0.q_norm.weight", nullptr).shape[0];
        heads_ = C_ / hd_;
        if (C_ % hd_ != 0 || hd_ % 8 != 0 || smod_.n != 6 * C_ || hmod_.n != 2 * C_)
            throw std::runtime_error("image engine (MOSS-Image DiT): an unexpected shape");
        load_text();
        int64_t& scratch = scratch_;
        for (int i = 0;; ++i) {
            const std::string p = "layers." + std::to_string(i) + ".";
            if (f.find(p + "proj.q.weight") == nullptr) break;
            Block b;
            const std::vector<std::pair<W*, std::string>> items = {
                {&b.q, "proj.q"}, {&b.k, "proj.k"}, {&b.v, "proj.v"}, {&b.o, "proj.o"},
                {&b.up_gate, "proj.ffn_up_gate"}, {&b.down, "proj.ffn_down"}};
            const std::vector<std::pair<uint64_t*, std::string>> norms = {
                {&b.n1, "norm1.weight"}, {&b.n2, "norm2.weight"}, {&b.nq, "q_norm.weight"}, {&b.nk, "k_norm.weight"}};
            // the block's host bytes: the quantized weights (q8), then every FP32 vector
            uint64_t qbytes = 0, vbytes = 0;
            for (const auto& [w, name] : items) {
                const SafeTensor& t = f.bf16(p + name + ".weight", nullptr);
                w->n = t.shape[0];
                w->k = t.shape[1];
                if (w->k % 32 != 0) throw std::runtime_error("image engine (MOSS-Image DiT): columns not 32s");
                const bool w8 = w8_ && w->k % kQiRotate == 0;
                w->type = w8 ? kTypeW8 : q8_ ? kTypeQ8_0 : kTypeF16;   // FP16: the file's bf16, rewritten on the GPU
                w->row_bytes = w8 ? (uint64_t) w->k : q8_ ? (uint64_t) (w->k / 32 * 34) : (uint64_t) w->k * 2;
                if (q8_) qbytes += w->row_bytes * (uint64_t) w->n + (w8 ? (uint64_t) w->n * 4 : 0);
                if (w->type == kTypeQ8_0) scratch = std::max(scratch, w->n * w->k);
            }
            for (const auto& [off, name] : norms) vbytes += (uint64_t) f.bf16(p + name, nullptr).elements() * 4;
            std::vector<LayerStream::Span> spans;
            std::vector<uint8_t> host;
            // an earlier open's bytes of this block, in the same form (their size says it: W8A8's rows and Q8_0's
            // differ), else the file's quantized again
            const bool reused = !dry_ && q8_ && reuse != nullptr && (size_t) i < reuse->size() &&
                                (*reuse)[(size_t) i].size() == qbytes + vbytes;
            if (reused) host = std::move((*reuse)[(size_t) i]);
            else if (!dry_) host.resize((size_t) (qbytes + vbytes));
            uint64_t at = 0;
            std::vector<size_t> wspan, nspan;
            for (const auto& [w, name] : items) {
                const uint8_t* d = nullptr;
                f.bf16(p + name + ".weight", &d);
                const uint64_t bytes = w->row_bytes * (uint64_t) w->n + (w->type == kTypeW8 ? (uint64_t) w->n * 4 : 0);
                if (q8_) {
                    if (!dry_ && !reused) {
                        quantize_matrix(reinterpret_cast<const uint16_t*>(d), w->n, w->k, host.data() + at,
                                        w->type == kTypeW8);
                        f.drop_cache(p + name + ".weight");   // read once, not kept in the page cache
                    }
                    spans.push_back({host.data() + at, bytes, 0});
                    at += bytes;
                } else {
                    spans.push_back({d, bytes, 0});
                }
                wspan.push_back(spans.size() - 1);
            }
            for (const auto& [off, name] : norms) {
                const int64_t n = f.bf16(p + name, nullptr).elements();
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
        if (blocks_.empty()) throw std::runtime_error("image engine (MOSS-Image DiT): no blocks");
        F_ = blocks_[0].down.k;
        if (blocks_[0].up_gate.n != 2 * F_)
            throw std::runtime_error("image engine (MOSS-Image DiT): ffn_up_gate is not twice ffn_down's width");
        build_lora_targets();
        install_prepare();
        if (dry_) return true;
        if (!gemm_.init(s_, scratch, err)) return false;
        blocks_stream_.configure(blocks_stream_.total(), s_);   // all resident until a budget is set
        t16_ = alloc<uint16_t>(allocs_, freq_ + 2 * C_);
        temb_ = alloc<float>(allocs_, C_);
        mod_ = alloc<float>(allocs_, 12 * C_);   // the step's six rows, then timestep 0's (set_refs)
        head_ = alloc<float>(allocs_, 2 * C_);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// a part outside the blocks: the rows of the named weights one after the other as FP16, their biases FP32 (in `owned`,
// their bytes added to `bytes`)
MossDit::W MossDit::dense(const std::vector<std::string>& names, bool biased, const float** bias,
                          std::vector<void*>& owned, uint64_t& bytes) {
    const Shards& f = *files_;
    W w;
    w.type = kTypeF16;
    std::vector<const uint8_t*> src;
    for (const auto& name : names) {
        const uint8_t* d = nullptr;
        const SafeTensor& t = f.bf16(name + ".weight", &d);
        if (w.k != 0 && t.shape[1] != w.k)
            throw std::runtime_error("image engine (MOSS-Image DiT): " + name + ": columns differ");
        w.k = t.shape[1];
        w.n += t.shape[0];
        src.push_back(d);
    }
    w.row_bytes = (uint64_t) w.k * 2;
    bytes += (uint64_t) (w.n * w.k) * 2 + (biased ? (uint64_t) w.n * 4 : 0);
    if (dry_) return w;
    std::vector<uint16_t> h((size_t) (w.n * w.k));
    std::vector<float> b;
    size_t at = 0;
    for (size_t j = 0; j < names.size(); ++j) {
        const int64_t n = f.bf16(names[j] + ".weight", nullptr).shape[0];
        const auto* s16 = reinterpret_cast<const uint16_t*>(src[j]);
        for (int64_t i = 0; i < n * w.k; ++i) h[at++] = f16_from_f32(bf16_value(s16[i]));
        if (biased) {
            const auto v = f.f32(names[j] + ".bias");
            b.insert(b.end(), v.begin(), v.end());
        }
    }
    uint16_t* d = alloc<uint16_t>(owned, (int64_t) h.size());
    gpu::copy(d, h.data(), h.size() * 2);
    w.d = d;
    if (biased) {
        float* db = alloc<float>(owned, (int64_t) b.size());
        gpu::copy(db, b.data(), b.size() * 4);
        *bias = db;
    }
    return w;
}

const float* MossDit::vec(const std::string& name, std::vector<void*>& owned, uint64_t& bytes) {
    const int64_t n = files_->bf16(name, nullptr).elements();
    bytes += (uint64_t) n * 4;
    if (dry_) return nullptr;
    const auto v = files_->f32(name);
    float* d = alloc<float>(owned, n);
    gpu::copy(d, v.data(), v.size() * 4);
    return d;
}

// the text embedder: its input projection and its blocks (their q, k and v rows of one qkv), FP16 on the GPU
void MossDit::load_text() {
    text_bytes_ = 0;
    auto rows = [](const W& w, int64_t r0, int64_t n) {
        W r = w;
        r.n = n;
        if (r.d != nullptr) r.d = static_cast<const uint16_t*>(w.d) + r0 * w.k;
        return r;
    };
    // the weight and the blocks go into the objects an earlier load made: a bound adapter's layer is the weight's own,
    // and the table of the weights adapters may take points into them (drop_text frees their bytes, not the objects)
    const auto keep = [](W& dst, const W& src) {
        LoraLayer* l = dst.lora;   // a weight the adapters are bound to keeps its layer across a load
        dst = src;
        dst.lora = l;
    };
    {
        const W tproj = dense({"text_embedder.proj"}, true, &b_tproj_, text_allocs_, text_bytes_);
        keep(tproj_, tproj);
        txt_dim_ = tproj_.k;
    }
    for (int i = 0;; ++i) {
        const std::string p = "text_embedder.blocks." + std::to_string(i) + ".";
        if (files_->find(p + "qkv.weight") == nullptr) break;
        if (tblocks_.size() <= (size_t) i) tblocks_.resize((size_t) i + 1);
        TextBlock& b = tblocks_[(size_t) i];
        const float* bqkv = nullptr;
        const W qkv = dense({p + "qkv"}, true, &bqkv, text_allocs_, text_bytes_);
        keep(b.q, rows(qkv, 0, C_));
        keep(b.k, rows(qkv, C_, C_));
        keep(b.v, rows(qkv, 2 * C_, C_));
        if (!dry_) {
            b.bq = bqkv;
            b.bk = bqkv + C_;
            b.bv = bqkv + 2 * C_;
        }
        keep(b.o, dense({p + "proj"}, true, &b.bo, text_allocs_, text_bytes_));
        keep(b.up_gate, dense({p + "ffn_up", p + "ffn_gate"}, true, &b.bug, text_allocs_, text_bytes_));
        keep(b.down, dense({p + "ffn_down"}, true, &b.bdown, text_allocs_, text_bytes_));
        b.n1 = vec(p + "norm1.weight", text_allocs_, text_bytes_);
        b.n2 = vec(p + "norm2.weight", text_allocs_, text_bytes_);
        b.nq = vec(p + "q_norm.weight", text_allocs_, text_bytes_);
        b.nk = vec(p + "k_norm.weight", text_allocs_, text_bytes_);
    }
    text_loaded_ = !dry_;
    build_lora_targets();
}

void MossDit::drop_text() {
    if (!text_loaded_) return;
    gpu::stream_sync(s_);
    for (void* p : text_allocs_) gpu::free(p);
    text_allocs_.clear();
    text_loaded_ = false;
}

std::pair<double, double> MossDit::time_forms(int64_t rows) const {
    return time_w8_f16(rows, C_, 2 * F_, "MOSS-Image's DiT", s_);
}

// Each block's weights, as they reach the GPU (resident, or a streamed slot): bf16 to FP16 (the FP16 blocks), and the
// adapters baked in (set_loras); none of it for 8-bit blocks without baked adapters
void MossDit::install_prepare() {
    if (q8_ && !baked_any_) {
        blocks_stream_.set_prepare(nullptr);
        return;
    }
    blocks_stream_.set_prepare([this](int i, uint8_t* d) {
        sycl::queue& q = q_of(s_);
        const Block& b = blocks_[(size_t) i];
        for (const W* w : {&b.q, &b.k, &b.v, &b.o, &b.up_gate, &b.down}) {
            uint8_t* wd = d + reinterpret_cast<uintptr_t>(w->d);
            if (!q8_) bf16_to_f16_inplace(reinterpret_cast<uint16_t*>(wd), w->n * w->k, q);
            if (w->baked) w->lora->bake(w->type, wd, w->n, w->k, gemm_, bake_tmp_, bake_at16_, s_);
        }
    });
}

MfOut MossDit::product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
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

void MossDit::add_glora(const LoraLayer* lora, const uint16_t* x16, const float* rs, const W& w, const uint8_t* d,
                         int64_t T, float* y) {
    if (lora == nullptr || !lora->has_glora()) return;
    // a GLoRA's own path: the weight's own product of its input, a band of tokens at a time
    lora->add_glora(x16, T, 0, w.n, rs, gemm_, lora_f32_, lora16_, s_,
                    [&](const uint16_t* v, int64_t t0, int64_t nt, int64_t r0, int64_t n2) {
                        glora_weight(v, w, d, r0, n2, nt, t0, y);
                    });
}

void MossDit::glora_weight(const uint16_t* v, const W& w, const uint8_t* d, int64_t r0, int64_t n, int64_t nt,
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

// An edit's references, a 1 MP image's rows asking after them (the image's rows are not known yet): the int8 kernel's
// prefix where qi_attention takes the int8 kernel at that shape
bool MossDit::refs_in_i8(int64_t rows) const {
    return rows > 0 && qi_attention_i8_preferred(4096, 4096 + rows, heads_, s_);
}

uint64_t MossDit::ref_bytes(int64_t rows) const {
    const uint64_t each = refs_in_i8(rows) ? (qi_attention_i8_prefix_bytes(rows, heads_) + 255) / 256 * 256
                                           : (uint64_t) rows * C_ * 2 * 2;
    return (uint64_t) blocks_.size() * (each + (uint64_t) heads_ * 4);
}

// The rows the working buffers hold: the steps' (n contexts of hw + max_text rows), or an edit's references' pass over
// the blocks (set_refs: one context of their rows), the more of the two
MossDit::Rows MossDit::rows_for(int64_t hw, int n, int64_t max_text, int64_t ref_rows, bool ref_i8) const {
    Rows r;
    r.R = std::max<int64_t>(n * (hw + max_text), ref_rows);
    r.ctx = std::max(hw + max_text, ref_rows);
    r.TP = qi_attn_rows(hw + max_text + (ref_i8 ? 0 : ref_rows));   // a context's, its references' rows before it
    r.J = std::max(n * r.TP, qi_attn_rows(ref_rows));
    r.O = std::max(r.TP, qi_attn_rows(ref_rows));
    return r;
}

uint64_t MossDit::activation_bytes(int64_t hw, int n, int64_t max_text, int64_t mlp_rows, int64_t ref_rows) const {
    const Rows rw = rows_for(hw, n, max_text, ref_rows, refs_in_i8(ref_rows));
    const int64_t R = rw.R;
    const int64_t band = mlp_rows > 0 ? std::min(R, mlp_rows) : R;
    return (uint64_t) R * C_ * 4 * 4 + (uint64_t) R * C_ * 2 +  // q, k, v, a; x (BF16)
           (uint64_t) band * 2 * F_ * 4 +                       // the MLP's up_gate sums
           (uint64_t) std::max(R * C_, band * F_) * 2 +         // the products' FP16 input
           (uint64_t) rw.J * C_ * 2 * 3 + (uint64_t) rw.O * C_ * 2 +   // q, k, v of each context, the output
           (uint64_t) R * C_ * 2 +                              // the attention's output, a row's scale
           (uint64_t) n * (hw + max_text) * hd_ * 4 +           // the rotary tables
           (uint64_t) (blocks_.size() * 2 + 1) * C_ * 4 +        // the int8 attention's K means and V scales
           (uint64_t) kColSlices * C_ * 4 +                     // col_max's parts (vpart_ at least)
           (uint64_t) rw.O * C_ * 4 +                           // the plain attention's scratch
           (w8_ ? (uint64_t) std::max(R, band) * std::max(C_, F_) + (uint64_t) std::max(R, band) * 8 + C_ * 4 : 0) +
           (lora_cols_ > 0 ? (uint64_t) std::max(R, lora_tmp_t_) * (uint64_t) lora_cols_ * 6 : 0);
}

void MossDit::ensure(int64_t hw, int n, int64_t max_text) {
    const int64_t cap_rows = rows_for(cap_hw_, (int) cap_n_, cap_text_, cap_ref_, cap_ref_i8_).R;
    if (hw <= cap_hw_ && n <= cap_n_ && max_text <= cap_text_ && ref_rows_ <= cap_ref_ &&
        (ref_rows_ == 0 || ref_i8_on_ == cap_ref_i8_) &&
        mlp_rows_ == (mlp_cap_ > 0 ? std::min(cap_rows, mlp_cap_) : cap_rows))
        return;
    hw = std::max(hw, cap_hw_);
    n = std::max<int>(n, (int) cap_n_);
    max_text = std::max(max_text, cap_text_);
    drop_activations();
    const Rows rw = rows_for(hw, n, max_text, ref_rows_, ref_i8_on_);
    const int64_t R = rw.R, TP = rw.TP;
    mlp_rows_ = mlp_cap_ > 0 ? std::min(R, mlp_cap_) : R;
    x_ = alloc<uint16_t>(allocs_, R * C_);
    q_ = alloc<float>(allocs_, R * C_);
    k_ = alloc<float>(allocs_, R * C_);
    v_ = alloc<float>(allocs_, R * C_);
    a_ = alloc<float>(allocs_, R * C_);
    gu_ = alloc<float>(allocs_, mlp_rows_ * 2 * F_);
    h16_ = alloc<uint16_t>(allocs_, std::max(R * C_, mlp_rows_ * F_));
    rs_ = alloc<float>(allocs_, std::max(R, mlp_rows_));
    rs2_ = alloc<float>(allocs_, std::max(R, mlp_rows_));
    rs_o_ = alloc<float>(allocs_, R);
    hs_ = alloc<float>(allocs_, heads_);
    // the int8 attention's statistics: a context's V scales, and K's mean a block and context (zero, no smoothing,
    // until a step makes it)
    vsv_ = alloc<float>(allocs_, C_);
    kmean_ = alloc<float>(allocs_, (int64_t) blocks_.size() * 2 * C_);
    gpu::memset_async(kmean_, 0, (size_t) blocks_.size() * 2 * C_ * 4, s_);
    kready_.assign(blocks_.size() * 2, 0);
    vpart_ = alloc<float>(allocs_, std::max(rw.ctx * heads_, kColSlices * C_));   // head_max's, col_max's
    if (w8_) {   // W8A8's int8 input: the most rows of the widest input, then two arrays of a scale a row
        xq8_rows_ = std::max(R, mlp_rows_);
        xq8_cols_ = std::max(C_, F_);
        xq8_ = alloc<int8_t>(allocs_, xq8_rows_ * xq8_cols_ + 2 * xq8_rows_ * 4);
        pend_sw_ = alloc<float>(allocs_, C_);
    }
    jrows_ = TP;   // a context's stride (the references' pass: one context, from row 0)
    jq16_ = alloc<uint16_t>(allocs_, rw.J * C_, true);
    jk16_ = alloc<uint16_t>(allocs_, rw.J * C_, true);
    jv16_ = alloc<uint16_t>(allocs_, rw.J * C_, true);
    jo16_ = alloc<uint16_t>(allocs_, rw.O * C_);
    o16_ = alloc<uint16_t>(allocs_, R * C_);
    if (keep0_) b0_ = alloc<uint16_t>(allocs_, (hw + max_text) * C_);
    cap_hw_ = hw;
    cap_n_ = n;
    cap_text_ = max_text;
    cap_ref_ = ref_rows_;
    cap_ref_i8_ = ref_i8_on_;
    dirty_[0] = dirty_[1] = 0;
}

void MossDit::drop_activations() {
    gpu::stream_sync(s_);
    for (void* p : {(void*) x_, (void*) q_, (void*) k_, (void*) v_, (void*) a_, (void*) gu_, (void*) h16_, (void*) rs_,
                    (void*) rs2_, (void*) rs_o_, (void*) hs_, (void*) vpart_, (void*) xq8_, (void*) pend_sw_,
                    (void*) vsv_, (void*) kmean_,
                    (void*) jq16_, (void*) jk16_, (void*) jv16_, (void*) jo16_, (void*) o16_,
                    (void*) b0_, (void*) lora_f32_, (void*) lora16_, (void*) lora_glora_y_,
                    (void*) lora_glora_x8_, (void*) lora_glora_sx_})
        release(allocs_, p);
    q_ = k_ = v_ = a_ = gu_ = rs_ = rs2_ = rs_o_ = hs_ = vpart_ = pend_sw_ = vsv_ = kmean_ = nullptr;
    x_ = b0_ = nullptr;
    kready_.clear();
    h16_ = jq16_ = jk16_ = jv16_ = jo16_ = o16_ = nullptr;
    xq8_ = nullptr;
    lora_f32_ = nullptr;
    lora16_ = nullptr;
    lora_tmp_t_ = 0;
    lora_glora_y_ = nullptr;
    lora_glora_x8_ = nullptr;
    lora_glora_sx_ = nullptr;
    cap_hw_ = cap_n_ = cap_text_ = cap_ref_ = 0;
}

std::vector<MossDit::LoraTarget> MossDit::lora_targets() const {
    std::vector<LoraTarget> out;
    out.reserve(lora_slots_.size());
    for (const LoraSlot& s : lora_slots_) out.push_back({s.name, s.row0, s.rows, s.w->k, s.src_row});
    return out;
}

void MossDit::build_lora_targets() {
    lora_slots_.clear();
    auto add = [this](std::string name, W& w, int64_t row0, int64_t rows, int64_t src_row, bool biased) {
        if (w.n <= 0 || w.k <= 0 || rows <= 0 || row0 + rows > w.n) return;
        lora_slots_.push_back({std::move(name), &w, row0, rows, src_row, biased});
    };
    // the weights outside the blocks (the norms are norms, not linear products: unchanged)
    add("video_embedder", vin_, 0, vin_.n, 0, true);
    add("time_embedding.0", t1_, 0, t1_.n, 0, true);
    add("time_embedding.2", t2_, 0, t2_.n, 0, true);
    add("shared_block_modulation", smod_, 0, smod_.n, 0, true);
    add("video_head_modulation", hmod_, 0, hmod_.n, 0, true);
    add("final_linear_video", fout_, 0, fout_.n, 0, false);
    // the text embedder: its projection and its blocks.  Its q, k and v rows are three weights on the GPU (b.q, b.k,
    // b.v) out of one qkv on the file's side (q the entry's rows [0, C), k [C, 2C), v [2C, 3C)); ffn_up and ffn_gate
    // are the two halves of the one [up | gate] the products take.  These products add biases; the blocks' below do not.
    add("text_embedder.proj", tproj_, 0, tproj_.n, 0, true);
    for (int i = 0; i < (int) tblocks_.size(); ++i) {
        const std::string p = "text_embedder.blocks." + std::to_string(i) + ".";
        TextBlock& b = tblocks_[(size_t) i];
        add(p + "qkv", b.q, 0, b.q.n, 0, true);
        add(p + "qkv", b.k, 0, b.k.n, b.q.n, true);
        add(p + "qkv", b.v, 0, b.v.n, b.q.n + b.k.n, true);
        add(p + "proj", b.o, 0, b.o.n, 0, true);
        add(p + "ffn_up_gate", b.up_gate, 0, b.up_gate.n, 0, true);
        add(p + "ffn_up", b.up_gate, 0, b.up_gate.n / 2, 0, true);
        add(p + "ffn_gate", b.up_gate, b.up_gate.n / 2, b.up_gate.n / 2, 0, true);
        add(p + "ffn_down", b.down, 0, b.down.n, 0, true);
    }
    // the 42 blocks: proj.q, proj.k, proj.v (or one proj.qkv, its rows q's, k's and v's, as a trainer that fuses them
    // writes it), proj.o, the [up | gate] of proj.ffn_up_gate, proj.ffn_down
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        const std::string p = "layers." + std::to_string(bi) + ".proj.";
        Block& b = blocks_[(size_t) bi];
        add(p + "q", b.q, 0, b.q.n, 0, false);
        add(p + "k", b.k, 0, b.k.n, 0, false);
        add(p + "v", b.v, 0, b.v.n, 0, false);
        add(p + "qkv", b.q, 0, b.q.n, 0, false);
        add(p + "qkv", b.k, 0, b.k.n, b.q.n, false);
        add(p + "qkv", b.v, 0, b.v.n, b.q.n + b.k.n, false);
        add(p + "o", b.o, 0, b.o.n, 0, false);
        add(p + "ffn_up_gate", b.up_gate, 0, b.up_gate.n, 0, false);
        add(p + "ffn_up", b.up_gate, 0, b.up_gate.n / 2, 0, false);
        add(p + "ffn_gate", b.up_gate, b.up_gate.n / 2, b.up_gate.n / 2, 0, false);
        add(p + "ffn_down", b.down, 0, b.down.n, 0, false);
    }
}

void MossDit::lora_scratch(int64_t T) {
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

bool MossDit::set_loras(const std::vector<LoraUse>& loras, std::string& err) {
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
            // the slots of the entry's name: one weight, or the weights a fused one is split into
            std::vector<const LoraSlot*> slots;
            for (const LoraSlot& s : lora_slots_)
                if (lora_key_names(entry.key, s.name)) slots.push_back(&s);
            if (slots.empty()) {
                unknown.push_back(entry.key);
                continue;
            }
            int64_t rows = 0;   // the entry's rows the weight takes
            for (const LoraSlot* s : slots) rows = std::max(rows, s->src_row + s->rows);
            // a GLoRA's four tensors may fit both of its readings: the weight's rows and columns say which
            LoraEntry alt;
            const LoraEntry& e = *lora_glora_for_weight(entry, rows, slots[0]->w->k, alt);
            const bool oft = e.kind == LoraKind::oft;
            const bool glora = e.kind == LoraKind::glora;
            if (oft && (slots.size() != 1 || slots[0]->src_row != 0)) {
                // a rotation is one weight's rows, block by block: a name the DiT splits across weights' rows takes none
                err = "image engine: " + u.file->path() + ": " + e.key +
                      ": an OFT, and the DiT's rows of that name are several weights or a part of one";
                drop_loras();
                return false;
            }
            if (e.kind == LoraKind::bias) {   // a bias's delta alone: its rows are the bias's, its columns none
                if (e.rows != rows) {
                    err = "image engine: " + u.file->path() + ": " + e.key + ": a bias's delta (diff_b/b_norm) of " +
                          std::to_string(e.rows) + " values, not the weight's " + std::to_string(rows) + " rows";
                    drop_loras();
                    return false;
                }
            } else if (e.rows != rows || (oft ? e.cols != 0 : e.cols != slots[0]->w->k)) {
                err = "image engine: " + u.file->path() + ": " + e.key + " is " + std::to_string(e.cols) + " x " +
                      std::to_string(e.rows) + ", not the DiT's " + std::to_string(slots[0]->w->k) + " x " +
                      std::to_string(rows);
                drop_loras();
                return false;
            }
            if (!e.bias.empty()) {   // a bias's delta is added where the weight adds its bias: the weight must have one
                for (const LoraSlot* s : slots)
                    if (!s->biased) {
                        err = "image engine: " + u.file->path() + ": " + e.key +
                              ": a bias's delta (diff_b/b_norm), and the DiT's weight adds no bias (its block's "
                              "modulation does the shift)";
                        drop_loras();
                        return false;
                    }
            }
            const float scale = u.scale * e.scale();
            // a strength of 0 changes nothing: the weight keeps its product's path (an alpha of 0 leaves the bias's
            // delta and a DoRA's base factors, as a DoRA's decomposition scales the weight with no delta)
            if (u.scale == 0.0f || (scale == 0.0f && e.bias.empty() && e.dora.empty())) continue;
            // a DoRA's: the entry's rows' delta factors (its B's, by the entry's rows), and each part's base factors
            const bool dora = !e.dora.empty();
            std::vector<float> coef((size_t) e.rows, 1.0f);
            std::vector<std::vector<float>> bases(slots.size());
            for (size_t i = 0; i < slots.size(); ++i) {
                const LoraSlot* slot = slots[i];
                std::vector<float> norms;
                if (dora && !dora_norms(*slot, norms, err)) {
                    drop_loras();
                    return false;
                }
                if (dora) {
                    std::vector<float> c;
                    std::string why;
                    if (!lora_dora_rows(e, slot->src_row, slot->row0, slot->rows, norms, u.scale, c, bases[i], why)) {
                        err = "image engine: ";
                        err += u.file->path();
                        err += ": ";
                        err += e.key;
                        err += ": ";
                        err += why;
                        drop_loras();
                        return false;
                    }
                    for (int64_t j = 0; j < slot->rows; ++j) coef[(size_t) (slot->src_row + j)] = c[(size_t) j];
                }
                if (e.replaces)   // a set_weight: the weight interpolates to the file's, W + s (S - W)
                    bases[i].assign((size_t) slot->rows, 1.0f - u.scale);
                if (const LoraPart* clash = lora_dora_overlap(slot->w->lora, slot->row0, slot->rows, dora, oft, glora,
                                                              e.replaces)) {
                    err = "image engine: " + u.file->path() + ": " + e.key + ": the weight's rows " +
                          std::to_string(slot->row0) + " .. " + std::to_string(slot->row0 + slot->rows) +
                          " are another adapter's, and " + lora_patch_clash_why(*clash, oft, glora, e.replaces);
                    drop_loras();
                    return false;
                }
            }
            auto dev = std::make_shared<LoraDeviceEntry>();
            if (!dev->upload(e, s_, err, dora ? coef.data() : nullptr)) {
                drop_loras();
                return false;
            }
            for (size_t i = 0; i < slots.size(); ++i) {
                const LoraSlot* slot = slots[i];
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
                part.dora = dora;
                part.oft = oft;
                part.glora = glora;
                part.replaces = e.replaces;
                part.base = std::move(bases[i]);
                slot->w->lora->parts.push_back(std::move(part));
            }
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
            for (W* w : {&b.q, &b.k, &b.v, &b.o, &b.up_gate, &b.down})
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

bool MossDit::tensor_norms(const std::string& name, const std::vector<float>*& out, std::string& err) {
    auto it = norm_cache_.find(name);
    if (it != norm_cache_.end()) {
        out = &it->second;
        return true;
    }
    try {
        const uint8_t* data = nullptr;
        const SafeTensor& t = files_->bf16(name, &data);
        const int64_t rows = (int64_t) t.shape[0], k = (int64_t) t.shape[1];
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
        files_->drop_cache(name);   // read once, not kept in the page cache
        out = &norm_cache_.emplace(name, std::move(norms)).first->second;
        return true;
    } catch (const std::exception& ex) {
        err = std::string("image engine (MOSS-Image DiT): a DoRA's row norms: ") + ex.what();
        return false;
    }
}

bool MossDit::dora_norms(const LoraSlot& slot, std::vector<float>& out, std::string& err) {
    const std::string& name = slot.name;
    const size_t dot = name.rfind('.');
    const std::string prefix = dot == std::string::npos ? std::string() : name.substr(0, dot + 1);
    const std::string last = dot == std::string::npos ? name : name.substr(dot + 1);
    if (last == "ffn_up_gate" || last == "ffn_up" || last == "ffn_gate") {
        // a fused [up | gate]: one tensor of the file, but the text embedder's, which is ffn_up's rows and ffn_gate's
        const std::vector<float>* a = nullptr;
        const std::vector<float>* b = nullptr;
        if (name.rfind("text_embedder.blocks.", 0) == 0) {
            if (!tensor_norms(prefix + "ffn_up.weight", a, err) || !tensor_norms(prefix + "ffn_gate.weight", b, err))
                return false;
            out.resize(a->size() + b->size());
            std::copy(a->begin(), a->end(), out.begin());
            std::copy(b->begin(), b->end(), out.begin() + (int64_t) a->size());
        } else {
            if (!tensor_norms(prefix + "ffn_up_gate.weight", a, err)) return false;
            out.assign(a->begin(), a->end());
        }
        if ((int64_t) out.size() < slot.w->n) {
            err = "image engine (MOSS-Image DiT): " + slot.name + ": the file's rows are fewer than the weight's";
            return false;
        }
        out.resize((size_t) slot.w->n);
        return true;
    }
    const std::vector<float>* norms = nullptr;
    if (!tensor_norms(name + ".weight", norms, err)) return false;
    if ((int64_t) norms->size() < slot.src_row + slot.w->n) {
        err = "image engine (MOSS-Image DiT): " + slot.name + ": the file's rows are fewer than the weight's";
        return false;
    }
    out.assign(norms->begin() + slot.src_row, norms->begin() + slot.src_row + slot.w->n);
    return true;
}

uint64_t MossDit::lora_bytes() const {
    uint64_t n = 0;
    for (const std::unique_ptr<LoraLayer>& l : lora_layers_) n += l->bytes();
    if (lora_glora_y_ != nullptr)   // a GLoRA's own path's buffers
        n += (uint64_t) kLoraGloraRows * ((uint64_t) lora_glora_rows_ * 4 + (uint64_t) lora_glora_cols_ + 4);
    return n;
}

void MossDit::drop_loras() {
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
                release(allocs_, *p);
                *p = nullptr;
            }
        lora_tmp_t_ = 0;
    }
    free_glora_scratch();
}

// A GLoRA's own path's buffers: the table the weight's product of its input is added from (a band of tokens, the
// weight's rows), and that input as a W8A8 product takes it (int8 values and a scale a row).  The deltas' scratch
// (lora_scratch) holds the input's own two products.
void MossDit::lora_glora_scratch() {
    if (lora_glora_cols_ <= 0 || lora_glora_rows_ <= 0) return;
    const size_t band = (size_t) kLoraGloraRows;
    lora_glora_y_ = alloc<float>(allocs_, (int64_t) (band * (size_t) lora_glora_rows_));
    lora_glora_x8_ = alloc<int8_t>(allocs_, (int64_t) (band * (size_t) lora_glora_cols_));
    lora_glora_sx_ = alloc<float>(allocs_, (int64_t) band);
}

void MossDit::free_glora_scratch() {
    for (void* p : {(void*) lora_glora_y_, (void*) lora_glora_x8_, (void*) lora_glora_sx_})
        if (p != nullptr) {
            gpu::stream_sync(s_);
            release(allocs_, p);
        }
    lora_glora_y_ = nullptr;
    lora_glora_x8_ = nullptr;
    lora_glora_sx_ = nullptr;
}

namespace {

// The RoPE's angles as the reference makes them: a coordinate (FP32) times each axis's hd / 8 frequencies
// 1 / base^(k / (hd / 8)) (FP32), the product in FP32; its cosine and sine rounded to bf16, as the reference casts
// them to its activations' type (and was trained with).  Row r's four coordinates from coord(r).
void rope_rows(int64_t rows, int64_t hd, double base, const std::function<void(int64_t, float*)>& coord,
               std::vector<float>& c, std::vector<float>& s) {
    const int64_t bands = hd / 8, half = hd / 2;
    std::vector<float> f((size_t) bands);
    for (int64_t k = 0; k < bands; ++k) {
        const auto e = (float) k / (float) bands;
        f[(size_t) k] = 1.f / (float) std::pow(base, (double) e);
    }
    c.assign((size_t) (rows * half), 0.f);
    s.assign(c.size(), 0.f);
    for (int64_t r = 0; r < rows; ++r) {
        float xyz[4];
        coord(r, xyz);
        for (int64_t j = 0; j < half; ++j) {
            const float ph = xyz[j / bands] * f[(size_t) (j % bands)];
            c[(size_t) (r * half + j)] = dit_rows::round_bf16((float) std::cos((double) ph));
            s[(size_t) (r * half + j)] = dit_rows::round_bf16((float) std::sin((double) ph));
        }
    }
}

}  // namespace

// a context's tables: the image's rows (0, the text's length, row and column centred), then the text's (0, i, 0, 0),
// then the references' in the stream (set_refs: index + 1, 0, row and column centred)
void MossDit::rope_tables(Ctx& c, int64_t h, int64_t w) {
    // an ai-toolkit adapter's references (set_refs): the image's and the text's f moved back by the text's rows
    const int64_t off = (ref_rows_ > 0 || stream_rows_ > 0) && !ref_text_start_ ? -c.rows : 0;
    if (c.rope_h == h && c.rope_w == w && c.rope_rows == c.rows && c.rope_off == off && c.rope_refs == stream_rows_)
        return;
    const int64_t HW = h * w, L = c.rows;
    std::vector<float> cv, sv;
    rope_rows(HW + L + stream_rows_, hd_, rope_base_, [&](int64_t r, float* p) {
        if (r < HW) {
            const int64_t y = r / w, x = r - y * w;
            p[0] = 0.f;
            p[1] = (float) (L + off);
            p[2] = (float) y - (float) (h - 1) / 2.f;
            p[3] = (float) x - (float) (w - 1) / 2.f;
        } else if (r < HW + L) {
            p[0] = 0.f;
            p[1] = (float) (r - HW + off);
            p[2] = p[3] = 0.f;
        } else {
            ref_coord(r - HW - L, p);
        }
    }, cv, sv);
    gpu::stream_sync(s_);
    release(allocs_, c.cos);
    release(allocs_, c.sin);
    c.cos = alloc<float>(allocs_, (int64_t) cv.size());
    c.sin = alloc<float>(allocs_, (int64_t) sv.size());
    gpu::copy(c.cos, cv.data(), cv.size() * 4);
    gpu::copy(c.sin, sv.data(), sv.size() * 4);
    c.rope_h = h;
    c.rope_w = w;
    c.rope_rows = L;
    c.rope_off = off;
    c.rope_refs = stream_rows_;
}

// the time's embedding as the reference makes it: the sinusoid [cos, sin] of the timestep by freq / 2 frequencies
// 10000^(-i / (freq / 2)) in FP64, rounded to bf16 (its first layer's type), through the two layers; then the shared
// modulation's six rows (mod_) and the head's two (head_)
void MossDit::set_time_path(const std::string& path) {
    if (path == tp_path_) return;
    gpu::stream_sync(s_);
    for (float* p : {tp_smod_b_, tp_hmod_b_})
        if (p != nullptr) release(allocs_, p);
    tp_smod_b_ = tp_hmod_b_ = nullptr;
    tp_w1_.clear(), tp_b1_.clear(), tp_w2_.clear(), tp_b2_.clear();
    tp_path_.clear();
    if (path.empty()) return;
    SafeTensors st(path);
    // the model's own layers' shapes: the file replaces them, or it is not a time path of this model
    const auto take = [&](const std::string& name, int64_t rows, int64_t cols) {
        const SafeTensor* t = st.find(name);
        const std::vector<int64_t> want = cols > 0 ? std::vector<int64_t>{rows, cols} : std::vector<int64_t>{rows};
        if (t == nullptr || t->shape != want)
            throw std::runtime_error(path + ": not a MOSS-Image time path (" + name + " of the model's shape)");
        return st.f32(name);
    };
    tp_w1_ = take("time_embedding.0.weight", t1_.n, t1_.k);
    tp_b1_ = take("time_embedding.0.bias", t1_.n, 0);
    tp_w2_ = take("time_embedding.2.weight", t2_.n, t2_.k);
    tp_b2_ = take("time_embedding.2.bias", t2_.n, 0);
    const std::vector<float> sb = take("shared_block_modulation.bias", smod_.n, 0);
    const std::vector<float> hb = take("video_head_modulation.bias", hmod_.n, 0);
    tp_smod_b_ = alloc<float>(allocs_, smod_.n);
    tp_hmod_b_ = alloc<float>(allocs_, hmod_.n);
    gpu::copy(tp_smod_b_, sb.data(), sb.size() * 4);
    gpu::copy(tp_hmod_b_, hb.data(), hb.size() * 4);
    tp_path_ = path;
}

void MossDit::time_embedding(float timestep) {
    const int64_t half = freq_ / 2;
    if (!tp_path_.empty()) {   // the adapter's time path, in FP32 (its training's): the sinusoid, two layers, SiLU between
        std::vector<float> e((size_t) freq_), h1((size_t) t1_.n), temb((size_t) t2_.n);
        for (int64_t i = 0; i < half; ++i) {
            const double ph = (double) timestep * std::pow(10000.0, -(double) i / (double) half);
            e[(size_t) i] = (float) std::cos(ph);
            e[(size_t) (half + i)] = (float) std::sin(ph);
        }
        const auto layer = [](const std::vector<float>& w, const std::vector<float>& b, const std::vector<float>& x,
                              std::vector<float>& y) {
            const size_t k = x.size();
            for (size_t o = 0; o < y.size(); ++o) {
                double s = b[o];
                const float* row = w.data() + o * k;
                for (size_t j = 0; j < k; ++j) s += (double) row[j] * x[j];
                y[o] = (float) s;
            }
        };
        layer(tp_w1_, tp_b1_, e, h1);
        for (float& v : h1) v = v / (1.f + std::exp(-v));
        layer(tp_w2_, tp_b2_, h1, temb);
        gpu::stream_sync(s_);   // the last step's products may still read temb_
        gpu::copy(temb_, temb.data(), temb.size() * 4);
        uint16_t* h16 = t16_ + freq_;
        kernels::f32_to_f16_bulk(temb_, h16 + C_, C_, s_);
        product(h16 + C_, nullptr, smod_, nullptr, mod_, 1, tp_smod_b_);
        product(h16 + C_, nullptr, hmod_, nullptr, head_, 1, tp_hmod_b_);
        return;
    }
    std::vector<uint16_t> e((size_t) freq_);
    for (int64_t i = 0; i < half; ++i) {
        const double ph = (double) timestep * std::pow(10000.0, -(double) i / (double) half);
        e[(size_t) i] = f16_from_f32(round_bf16((float) std::cos(ph)));
        e[(size_t) (half + i)] = f16_from_f32(round_bf16((float) std::sin(ph)));
    }
    gpu::copy_async(t16_, e.data(), e.size() * 2, s_);
    gpu::stream_sync(s_);   // e leaves scope
    uint16_t* h16 = t16_ + freq_;
    product(t16_, nullptr, t1_, nullptr, temb_, 1, b_t1_);
    qi_silu(temb_, C_, h16, s_);
    product(h16, nullptr, t2_, nullptr, temb_, 1, b_t2_);
    kernels::f32_to_f16_bulk(temb_, h16 + C_, C_, s_);
    product(h16 + C_, nullptr, smod_, nullptr, mod_, 1, b_smod_);
    product(h16 + C_, nullptr, hmod_, nullptr, head_, 1, b_hmod_);
}

void MossDit::run_block(const W* const* w, const float* const* bias, const float* n1, const float* n2,
                        const float* nq, const float* nk, const uint8_t* base, const float* mod, uint16_t* x,
                        const std::vector<int64_t>& seg, const std::vector<const float*>& cos,
                        const std::vector<const float*>& sin) {
    const int64_t C = C_;
    int64_t R = 0;
    for (int64_t r : seg) R += r;
    auto b = [&](int j) { return bias != nullptr ? bias[j] : nullptr; };
    const float* shift_a = mod;
    const float* scale_a = mod != nullptr ? mod + C : nullptr;
    const float* gate_a = mod != nullptr ? mod + 2 * C : nullptr;
    const float* shift_f = mod != nullptr ? mod + 3 * C : nullptr;
    const float* scale_f = mod != nullptr ? mod + 4 * C : nullptr;
    const float* gate_f = mod != nullptr ? mod + 5 * C : nullptr;
    // the references in the stream at timestep 0 (set_refs): the last rows of each segment, their modulation the
    // six rows after mod's
    AltRows rr;
    if (mod != nullptr && stream_t0_ && stream_rows_ > 0) {
        rr.alt = 6 * C;
        rr.e0 = seg[0];
        rr.r0 = seg[0] - stream_rows_;
        if (seg.size() > 1) rr.r1 = seg[0] + seg[1] - stream_rows_;
    }
    // W8A8 blocks take their products' int8 inputs from the row-wise kernels themselves (rows_i8), each into the
    // scales' slot the deferred sums before it do not read (i8_slot).  A block an adapter changes takes its products'
    // inputs as the FP16 rows (rows16): the delta is the input's own, and rows_i8 makes only its int8 values (and every
    // product there is one pass more, the adapter applied).  An adapter baked into the weights (set_loras) is as none
    const bool i8 = std::all_of(w, w + 6, [](const W* p) {
        return p->type == kTypeW8 && (p->lora == nullptr || p->baked);
    });
    // the block before's MLP output, added to x by this block's first kernel where it ran over the same rows
    MfOut pre;
    const float* pre_gate = nullptr;
    if (pend_.x != nullptr) {
        if (i8 && pend_.x == x && pend_.R == R && pend_.rr.alt == rr.alt && pend_.rr.r1 == rr.r1) {
            pre = pend_.add;
            pre_gate = pend_.gate;
            pend_ = {};
        } else {
            flush_pending();
        }
    }
    // q, k, v of every row (the products' sums scaled by their readers)
    if (i8) rows_i8(RmsI8{x, pre, pre_gate, n1, shift_a, scale_a, C, rr}, R, C, xq8_, i8_slot(), s_);
    else rows16(rms_mod(x, n1, shift_a, scale_a, C, rr), R, C, h16_, rs_, s_);
    const MfOut q = product(h16_, rs_, *w[0], base, q_, R, b(0), i8, true);
    const MfOut k = product(h16_, rs_, *w[1], base, k_, R, b(1), w[0]->type == kTypeW8, true);
    const MfOut v = product(h16_, rs_, *w[2], base, v_, R, b(2), w[1]->type == kTypeW8, true);
    // each segment: its v heads' scales, its q and k normed and rotated, the attention among its rows, the output
    // back as FP16 rows scaled by row.  An edit's references (set_refs) go first in each segment's keys and values,
    // the segment's rows asking after them, both at the larger of their heads' scales; while set_refs runs the blocks
    // over them, each block's are kept
    const int64_t P = ref_rows_ > 0 && !capture_ && mod != nullptr ? ref_rows_ : 0;
    const bool kept = P > 0 && ref_i8_ != nullptr;   // the references as the int8 kernel's prefix, not rows before
    const int64_t PO = kept ? 0 : P;                   // the segment's own rows' first row
    int64_t r0 = 0;
    float* so = i8 ? i8_slot() : nullptr;   // the attention's output's scales (q, k and v's sums read the other slot)
    for (size_t i = 0; i < seg.size(); ++i) {
        const int64_t T = seg[i], o = (int64_t) i * jrows_ * C, TT = PO + T;
        if (dirty_[i] > TT)
            for (uint16_t* jb : {jq16_, jk16_, jv16_})
                gpu::memset_async(jb + o + TT * C, 0, (size_t) ((dirty_[i] - TT) * C) * 2, s_);
        dirty_[i] = TT;
        // v's heads' scales; without references before the rows, by column, the int8 attention's V scales with them
        // and K's mean kept from the last step (FlashAttnI8Stats)
        if (P == 0) {
            col_max(v.from(r0), T, MfOut{}, 0, C, vpart_, s_);
            col_scale(vpart_, C, heads_, hd_, hs_, vsv_, s_);
        } else {
            head_max(v.from(r0), T, MfOut{}, 0, heads_, hd_, vpart_, s_);
            head_scale(vpart_, T, heads_, hs_, s_);
        }
        const int64_t rk = (int64_t) cur_block_ * ref_rows_ * C;
        if (P > 0) hs_max(hs_, ref_hs_ + cur_block_ * heads_, heads_, s_);
        qk_rope16(q.from(r0), k.from(r0), v.from(r0), nq, nk, cos[i], sin[i], hs_, T, heads_, hd_, true,
                  jq16_ + o + PO * C, jk16_ + o + PO * C, jv16_ + o + PO * C, s_);
        if (P > 0 && !kept) {
            gpu::copy_async(jk16_ + o, ref_k16_ + rk, (size_t) (P * C) * 2, s_);
            ref_values(ref_v16_ + rk, ref_hs_ + cur_block_ * heads_, hs_, P, heads_, hd_, jv16_ + o, s_);
        }
        if (capture_) {
            if (ref_i8_ != nullptr) {
                if (!qi_attention_i8_prefix(jk16_ + o, jv16_ + o, T, heads_, ref_i8_ + cur_block_ * ref_i8_stride_, s_))
                    throw std::runtime_error("image engine: the references' keys could not be kept in int8");
            } else {
                gpu::copy_async(ref_k16_ + rk, jk16_ + o, (size_t) (T * C) * 2, s_);
                gpu::copy_async(ref_v16_ + rk, jv16_ + o, (size_t) (T * C) * 2, s_);
            }
            gpu::copy_async(ref_hs_ + cur_block_ * heads_, hs_, (size_t) heads_ * 4, s_);
        }
        // every row sees every key: the int8 kernel may take them all where it is the faster (qi_attention's choice)
        if (kept) {
            const uint8_t* prefix = ref_i8_ + cur_block_ * ref_i8_stride_;
            if (!qi_attention_i8_prefixed(jq16_ + o, jk16_ + o, jv16_ + o, T, heads_, prefix, P,
                                          ref_hs_ + cur_block_ * heads_, hs_, jo16_, s_))
                throw std::runtime_error("image engine: the attention's int8 buffers do not fit");
        } else {
            const size_t ki = (size_t) cur_block_ * 2 + (size_t) (cur_c0_ + (int) i);
            kernels::FlashAttnI8Stats st{vsv_, kmean_ + (int64_t) ki * C, kready_[ki] != 0};
            qi_attention(jq16_ + o, jk16_ + o, jv16_ + o, TT, heads_, hd_, 0, jo16_, s_, nullptr, P, true, nullptr,
                         P == 0 ? &st : nullptr);
            if (P == 0) kready_[ki] = 1;
        }
        if (i8) rows_i8(HeadsI8{Heads{jo16_, hs_, C, hd_}}, T, C, xq8_ + r0 * C, so + r0, s_);
        else rows16(Heads{jo16_, hs_, C, hd_}, T, C, o16_ + r0 * C, rs_o_ + r0, s_);
        r0 += T;
    }
    const MfOut ao = product(o16_, rs_o_, *w[3], base, a_, R, b(3), i8, true);
    // the MLP, a band of rows at a time; in one band, the attention's gate_add in the kernel that makes its input
    const bool one_band = i8 && mlp_rows_ >= R;
    if (!one_band) gate_add_rows(x, ao, gate_a, rr, R, C, s_);
    for (int64_t m0 = 0; m0 < R; m0 += mlp_rows_) {
        const int64_t mr = std::min(mlp_rows_, R - m0);
        if (i8) {
            rows_i8(RmsI8{x + m0 * C, one_band ? ao : MfOut{}, gate_a, n2, shift_f, scale_f, C, rr.from(m0)}, mr, C,
                    xq8_, i8_slot(), s_);
            const MfOut g = product(h16_, rs_, *w[4], base, gu_, mr, b(4), true, true);
            rows_i8(SituI8{Situ{g, F_}}, mr, F_, xq8_, i8_slot(), s_);
            MfOut d = product(h16_, rs2_, *w[5], base, a_, mr, b(5), true, true);
            if (!one_band || mod == nullptr) {
                gate_add_rows(x + m0 * C, d, gate_f, rr.from(m0), mr, C, s_);
                continue;
            }
            // left to the next block's first kernel, int32 sums' weights' scales copied out of the block's region (the
            // stream may refill it before then; BF16 values carry none)
            if (d.sw != nullptr) {
                gpu::copy_async(pend_sw_, d.sw, (size_t) C * 4, s_);
                d.sw = pend_sw_;
            }
            pend_ = {d, gate_f, x, R, rr};
            continue;
        }
        rows16(rms_mod(x + m0 * C, n2, shift_f, scale_f, C, rr.from(m0)), mr, C, h16_, rs_, s_);
        // SiTU reads the first product's sums with rs_, so its own rows' scales go to rs2_
        rows16(Situ{product(h16_, rs_, *w[4], base, gu_, mr, b(4), false, true), F_}, mr, F_, h16_, rs2_, s_);
        gate_add_rows(x + m0 * C, product(h16_, rs2_, *w[5], base, a_, mr, b(5), false, true), gate_f, rr.from(m0),
                      mr, C, s_);
    }
}

void MossDit::flush_pending() {
    if (pend_.x != nullptr) gate_add_rows(pend_.x, pend_.add, pend_.gate, pend_.rr, pend_.R, C_, s_);
    pend_ = {};
}

void MossDit::set_context(int i, const float* states, int64_t rows) {
    Ctx& c = ctx_[i];
    gpu::stream_sync(s_);
    release(allocs_, c.txt0);
    c.txt0 = nullptr;
    c.rows = rows;
    c.rope_rows = -1;
    if (rows <= 0) return;
    if (!text_loaded_) load_text();
    ensure(cap_hw_, std::max<int>(1, (int) cap_n_), std::max(cap_text_, rows));
    // the input projection (the states' rows scaled into FP16's range), then the text blocks over the text's rows
    float* xs = alloc<float>(allocs_, rows * txt_dim_);
    uint16_t* x16 = alloc<uint16_t>(allocs_, rows * txt_dim_);
    gpu::copy(xs, states, (size_t) (rows * txt_dim_) * 4);
    rows16(Plain{xs, txt_dim_}, rows, txt_dim_, x16, rs_, s_);
    float* t32 = alloc<float>(allocs_, rows * C_);
    c.txt0 = alloc<uint16_t>(allocs_, rows * C_);
    product(x16, rs_, tproj_, nullptr, t32, rows, b_tproj_);
    prefill::to_bf16(t32, c.txt0, rows * C_, s_);
    std::vector<float> cv, sv;
    rope_rows(rows, hd_, rope_base_, [](int64_t r, float* p) {
        p[0] = p[2] = p[3] = 0.f;
        p[1] = (float) r;
    }, cv, sv);
    float* dc = alloc<float>(allocs_, (int64_t) cv.size());
    float* ds = alloc<float>(allocs_, (int64_t) sv.size());
    gpu::copy(dc, cv.data(), cv.size() * 4);
    gpu::copy(ds, sv.data(), sv.size() * 4);
    for (const TextBlock& tb : tblocks_) {
        const W* w[6] = {&tb.q, &tb.k, &tb.v, &tb.o, &tb.up_gate, &tb.down};
        const float* bias[6] = {tb.bq, tb.bk, tb.bv, tb.bo, tb.bug, tb.bdown};
        run_block(w, bias, tb.n1, tb.n2, tb.nq, tb.nk, nullptr, nullptr, c.txt0, {rows}, {dc}, {ds});
    }
    gpu::stream_sync(s_);
    for (void* p : {(void*) xs, (void*) x16, (void*) t32, (void*) dc, (void*) ds}) release(allocs_, p);
}

namespace moss_ctl_k {   // the kernels' names (an anonymous namespace mangles the same in every file)
// y16 [rows][3 in] = x, color and depth side by side ([rows][in] each; null: zeros), FP16
void tokens(const float* x, const float* color, const float* depth, int64_t rows, int64_t in, uint16_t* y16, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (rows * 3 * in)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / (3 * in), c = i - r * 3 * in, part = c / in, j = c - part * in;
        const float* src = part == 0 ? x : part == 1 ? color : depth;
        y16[i] = f16_from_f32(src != nullptr ? src[r * in + j] : 0.f);
    });
}
}  // namespace moss_ctl_k

void MossDit::set_control(const std::string& path) {
    if (path == ctl_path_) return;
    gpu::stream_sync(s_);
    for (void* p : ctl_allocs_) gpu::free(p);
    ctl_allocs_.clear();
    weight_bytes_ -= std::min<uint64_t>(weight_bytes_, ctl_bytes_);
    ctl_bytes_ = 0;
    ctl_path_.clear();
    ctl_b_ = nullptr;
    ctl_w_ = ctl_w0_ = W{};
    if (path.empty()) return;
    SafeTensors st(path);
    const SafeTensor* t = st.find("video_embedder.weight");
    if (t == nullptr || t->shape.size() != 2 || t->shape[0] != C_ || t->shape[1] != 3 * in_ch_ ||
        st.find("video_embedder.bias") == nullptr)
        throw std::runtime_error(path + ": not a MOSS-Image color and depth control adapter (its video_embedder [" +
                                 std::to_string(C_) + "][" + std::to_string(3 * in_ch_) + "])");
    const std::vector<float> w = st.f32("video_embedder.weight"), b = st.f32("video_embedder.bias");
    const int64_t K = 3 * in_ch_;
    std::vector<uint16_t> h((size_t) (C_ * K)), h0((size_t) (C_ * in_ch_));
    for (int64_t o = 0; o < C_; ++o)
        for (int64_t k = 0; k < K; ++k) {
            const uint16_t v = f16_from_f32(w[(size_t) (o * K + k)]);
            h[(size_t) (o * K + k)] = v;
            if (k < in_ch_) h0[(size_t) (o * in_ch_ + k)] = v;
        }
    uint16_t* d = alloc<uint16_t>(ctl_allocs_, (int64_t) h.size());
    uint16_t* d0 = alloc<uint16_t>(ctl_allocs_, (int64_t) h0.size());
    float* db = alloc<float>(ctl_allocs_, C_);
    gpu::copy(d, h.data(), h.size() * 2);
    gpu::copy(d0, h0.data(), h0.size() * 2);
    gpu::copy(db, b.data(), (size_t) C_ * 4);
    ctl_w_.type = ctl_w0_.type = kTypeF16;
    ctl_w_.n = ctl_w0_.n = C_;
    ctl_w_.k = K;
    ctl_w0_.k = in_ch_;
    ctl_w_.row_bytes = (uint64_t) K * 2;
    ctl_w0_.row_bytes = (uint64_t) in_ch_ * 2;
    ctl_w_.d = d;
    ctl_w0_.d = d0;
    ctl_w0_.lora = vin_.lora;   // an adapter of the request on the input projection: the images' columns
    ctl_b_ = db;
    ctl_bytes_ = (uint64_t) (C_ * (K + in_ch_)) * 2 + (uint64_t) C_ * 4;
    weight_bytes_ += ctl_bytes_;
    ctl_path_ = path;
}

void MossDit::set_control_input(const float* color, const float* depth, float color_end, float depth_end) {
    ctl_color_ = color;
    ctl_depth_ = depth;
    ctl_color_end_ = color_end;
    ctl_depth_end_ = depth_end;
    ctl_at_ = 0.0;
    if (color == nullptr && depth == nullptr && ctl16_ != nullptr) {
        gpu::stream_sync(s_);
        release(allocs_, ctl16_);
        ctl16_ = nullptr;
        ctl16_rows_ = 0;
    }
}

void MossDit::input_projection(const float* x, int64_t rows, float* y) {
    if (ctl_path_.empty()) {
        kernels::f32_to_f16_bulk(x, h16_, rows * in_ch_, s_);
        product(h16_, nullptr, vin_, nullptr, y, rows, b_vin_);
        return;
    }
    const float* color = ctl_color_ != nullptr && ctl_at_ < (double) ctl_color_end_ ? ctl_color_ : nullptr;
    const float* depth = ctl_depth_ != nullptr && ctl_at_ < (double) ctl_depth_end_ ? ctl_depth_ : nullptr;
    if (color == nullptr && depth == nullptr) {   // no map at this step: the zero columns add nothing
        kernels::f32_to_f16_bulk(x, h16_, rows * in_ch_, s_);
        product(h16_, nullptr, ctl_w0_, nullptr, y, rows, ctl_b_);
        return;
    }
    if (rows > ctl16_rows_) {
        gpu::stream_sync(s_);
        if (ctl16_ != nullptr) release(allocs_, ctl16_);
        ctl16_ = alloc<uint16_t>(allocs_, rows * 3 * in_ch_);
        ctl16_rows_ = rows;
    }
    moss_ctl_k::tokens(x, color, depth, rows, in_ch_, ctl16_, s_);
    product(ctl16_, nullptr, ctl_w_, nullptr, y, rows, ctl_b_);
}

void MossDit::forward(const float* x, float timestep, int64_t h, int64_t w, int n, float* out) {
    const int64_t HW = h * w, C = C_, J = stream_rows_;
    int64_t maxL = 0;
    std::vector<int64_t> off((size_t) n + 1, 0);
    for (int i = 0; i < n; ++i) {
        maxL = std::max(maxL, ctx_[i].rows);
        off[(size_t) i + 1] = off[(size_t) i] + HW + ctx_[i].rows + J;
    }
    ensure(HW, n, maxL + J);   // the references in the stream counted as text rows
    for (int i = 0; i < n; ++i) rope_tables(ctx_[i], h, w);
    // each context's rows: the image's input projection (the same rows in each), then its text embedder's rows,
    // then the references' in the stream
    input_projection(x, HW, a_);
    prefill::to_bf16(a_, x_, HW * C, s_);
    for (int i = 0; i < n; ++i) {
        if (i > 0) gpu::copy_async(x_ + off[(size_t) i] * C, x_, (size_t) (HW * C) * 2, s_);
        gpu::copy_async(x_ + (off[(size_t) i] + HW) * C, ctx_[i].txt0, (size_t) (ctx_[i].rows * C) * 2, s_);
        if (J > 0)
            gpu::copy_async(x_ + (off[(size_t) i] + HW + ctx_[i].rows) * C, ref_x_, (size_t) (J * C) * 2, s_);
    }
    time_embedding(timestep);
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        const Block& b = blocks_[(size_t) bi];
        // the contexts that take this block: a run of them (two at most), their rows one after the other
        int c0 = -1, c1 = -1;
        for (int i = 0; i < n; ++i)
            if (std::find(ctx_[i].skip.begin(), ctx_[i].skip.end(), bi) == ctx_[i].skip.end()) {
                if (c0 < 0) c0 = i;
                c1 = i + 1;
            }
        const uint8_t* base = blocks_stream_.begin(bi);
        cur_block_ = bi;
        if (c0 >= 0) {
            auto f = [base](uint64_t o) { return reinterpret_cast<const float*>(base + o); };
            std::vector<int64_t> seg;
            std::vector<const float*> cs, ss;
            for (int i = c0; i < c1; ++i) {
                seg.push_back(HW + ctx_[i].rows + J);
                cs.push_back(ctx_[i].cos);
                ss.push_back(ctx_[i].sin);
            }
            const W* ws[6] = {&b.q, &b.k, &b.v, &b.o, &b.up_gate, &b.down};
            cur_c0_ = c0;
            run_block(ws, nullptr, f(b.n1), f(b.n2), f(b.nq), f(b.nk), base, mod_, x_ + off[(size_t) c0] * C, seg,
                      cs, ss);
        }
        if (bi == 0 && keep0_) flush_pending();
        blocks_stream_.end(bi);
        if (bi == 0 && keep0_) gpu::copy_async(b0_, x_, (size_t) ((HW + ctx_[0].rows) * C) * 2, s_);
    }
    flush_pending();
    // the head: each context's image rows normed, scaled and shifted by the time, projected to the latent's channels
    for (int i = 0; i < n; ++i) {
        rows16(rms_mod(x_ + off[(size_t) i] * C, fnorm_, head_, head_ + C, C), HW, C, h16_, rs_, s_);
        product(h16_, rs_, fout_, nullptr, out + i * HW * in_ch_, HW, nullptr);
    }
}

// reference row r's coordinates (index + 1, 0, row and column centred), the rows of set_refs's references in turn
void MossDit::ref_coord(int64_t r, float* p) const {
    size_t i = 0;
    while (r >= ref_shapes_[i].first * ref_shapes_[i].second) {
        r -= ref_shapes_[i].first * ref_shapes_[i].second;
        ++i;
    }
    const int64_t h = ref_shapes_[i].first, w = ref_shapes_[i].second, y = r / w, x = r - y * w;
    p[0] = (float) (i + 1);
    p[1] = 0.f;
    p[2] = (float) y - (float) (h - 1) / 2.f;
    p[3] = (float) x - (float) (w - 1) / 2.f;
}

void MossDit::set_refs(const std::vector<RefLatent>& refs, QiRefLayout layout) {
    gpu::stream_sync(s_);
    ref_text_start_ = layout == QiRefLayout::MossEdit;
    for (void* p : {(void*) ref_k16_, (void*) ref_v16_, (void*) ref_hs_, (void*) ref_i8_, (void*) ref_x_})
        release(allocs_, p);
    ref_k16_ = ref_v16_ = nullptr;
    ref_hs_ = nullptr;
    ref_x_ = nullptr;
    ref_i8_ = nullptr;
    ref_i8_on_ = false;
    ref_rows_ = stream_rows_ = 0;
    stream_t0_ = false;
    ref_shapes_.clear();
    int64_t R = 0;
    for (const RefLatent& f : refs) {
        R += f.h * f.w;
        ref_shapes_.emplace_back(f.h, f.w);
    }
    if (R == 0) return;
    if (layout == QiRefLayout::AitkNoKv || layout == QiRefLayout::Appended) {
        // in the stream: their input projection kept, and timestep 0's modulation beside the step's
        uint16_t* x16 = alloc<uint16_t>(allocs_, R * in_ch_);
        int64_t o = 0;
        for (const RefLatent& f : refs) {
            kernels::f32_to_f16_bulk(f.tokens, x16 + o * in_ch_, f.h * f.w * in_ch_, s_);
            o += f.h * f.w;
        }
        float* rx = alloc<float>(allocs_, R * C_);
        ref_x_ = alloc<uint16_t>(allocs_, R * C_);
        product(x16, nullptr, ctl_path_.empty() ? vin_ : ctl_w0_, nullptr, rx, R, ctl_path_.empty() ? b_vin_ : ctl_b_);
        prefill::to_bf16(rx, ref_x_, R * C_, s_);
        if (layout == QiRefLayout::AitkNoKv) {
            time_embedding(0.f);
            gpu::copy_async(mod_ + 6 * C_, mod_, (size_t) (6 * C_) * 4, s_);
            stream_t0_ = true;
        }
        gpu::stream_sync(s_);
        release(allocs_, x16);
        release(allocs_, rx);
        stream_rows_ = R;
        return;
    }
    const int64_t nb = (int64_t) blocks_.size(), C = C_;
    ref_i8_on_ = refs_in_i8(R);
    if (ref_i8_on_) {
        ref_i8_stride_ = (qi_attention_i8_prefix_bytes(R, heads_) + 255) / 256 * 256;
        ref_i8_ = alloc<uint8_t>(allocs_, nb * (int64_t) ref_i8_stride_);
    } else {
        ref_k16_ = alloc<uint16_t>(allocs_, nb * R * C);
        ref_v16_ = alloc<uint16_t>(allocs_, nb * R * C);
    }
    ref_hs_ = alloc<float>(allocs_, nb * heads_);
    ref_rows_ = R;   // the attention's rows count them from here (ensure)
    ensure(cap_hw_, std::max<int>(1, (int) cap_n_), cap_text_);
    // their rows through the input projection, on the rotary axes (index + 1, 0, row, column), at timestep 0
    std::vector<float> cv, sv;
    rope_rows(R, hd_, rope_base_, [this](int64_t r, float* p) { ref_coord(r, p); }, cv, sv);
    float* dc = alloc<float>(allocs_, (int64_t) cv.size());
    float* ds = alloc<float>(allocs_, (int64_t) sv.size());
    gpu::copy(dc, cv.data(), cv.size() * 4);
    gpu::copy(ds, sv.data(), sv.size() * 4);
    int64_t o = 0;
    for (const RefLatent& f : refs) {
        kernels::f32_to_f16_bulk(f.tokens, h16_ + o * in_ch_, f.h * f.w * in_ch_, s_);
        o += f.h * f.w;
    }
    product(h16_, nullptr, ctl_path_.empty() ? vin_ : ctl_w0_, nullptr, a_, R, ctl_path_.empty() ? b_vin_ : ctl_b_);
    prefill::to_bf16(a_, x_, R * C_, s_);
    time_embedding(0.f);
    capture_ = true;
    try {
        for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
            const Block& b = blocks_[(size_t) bi];
            const uint8_t* base = blocks_stream_.begin(bi);
            auto f = [base](uint64_t off) { return reinterpret_cast<const float*>(base + off); };
            const W* ws[6] = {&b.q, &b.k, &b.v, &b.o, &b.up_gate, &b.down};
            cur_block_ = bi;
            cur_c0_ = 0;
            run_block(ws, nullptr, f(b.n1), f(b.n2), f(b.nq), f(b.nk), base, mod_, x_, {R}, {dc}, {ds});
            blocks_stream_.end(bi);
        }
        flush_pending();
    } catch (...) {
        capture_ = false;
        throw;
    }
    capture_ = false;
    gpu::stream_sync(s_);
    release(allocs_, dc);
    release(allocs_, ds);
}

namespace {

// an Euler step x += ds p in FP32, the prediction p as the reference pipeline makes it: the prompt's velocity c, with
// guidance (u: the negative pass's) u + cfg (c - u), with skip-layer guidance (k: the skipping pass's) that (or k
// without guidance) plus slg (c - k)
void moss_euler(float* x, const float* c, const float* u, const float* k, int64_t n, float cfg, float slg, float ds,
                void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float ci = c[i];
        float p = u != nullptr ? u[i] + cfg * (ci - u[i]) : ci;
        if (k != nullptr) p = (u != nullptr ? p : k[i]) + slg * (ci - k[i]);
        x[i] += ds * p;
    });
}

// x0 = x - sigma * v, the step's prediction of the finished latent (a partial image's), v as moss_euler's
void moss_predicted(float* x0, const float* x, const float* c, const float* u, const float* k, int64_t n, float cfg,
                    float slg, float sigma, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float ci = c[i];
        float p = u != nullptr ? u[i] + cfg * (ci - u[i]) : ci;
        if (k != nullptr) p = (u != nullptr ? p : k[i]) + slg * (ci - k[i]);
        x0[i] = x[i] - sigma * p;
    });
}

}  // namespace

void MossDit::sample(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, const Guide& g, int first,
                     int total, const std::function<bool(int, int)>& step,
                     const std::function<void(float*, float)>& after,
                     const std::function<void(int, const std::function<void(float*)>&)>& peek) {
    const int n = g.cfg != 1.f && ctx_[1].rows > 0 ? 2 : 1;
    const bool slg = !g.slg.empty() && g.slg_scale > 0.f;
    std::fill(kready_.begin(), kready_.end(), 0);   // K's means made again at this image's first step
    const int64_t HW = h * w, N = HW * in_ch_;
    float* out = alloc<float>(allocs_, (n + (slg ? 1 : 0)) * N);
    const std::vector<int> before = ctx_[1].skip;
    const int steps = (int) sigmas.size() - 1;
    try {
        for (int i = 0; i < steps; ++i) {
            // both skips act in the steps whose share of the whole schedule is in [start, stop)
            const double at = (double) (first + i) / (double) total;
            const bool active = g.start <= at && at < g.stop;
            ctl_at_ = at;   // the control maps' ends (input_projection)
            ctx_[1].skip = active ? g.uncond_skip : std::vector<int>{};
            const float t = sigmas[(size_t) i] * 1000.f;
            forward(x, t, h, w, n, out);
            const float* k = nullptr;
            if (slg && active) {
                ctx_[0].skip = g.slg;
                forward(x, t, h, w, 1, out + n * N);
                ctx_[0].skip.clear();
                k = out + n * N;
            }
            moss_euler(x, out, n == 2 ? out + N : nullptr, k, N, g.cfg, g.slg_scale,
                       sigmas[(size_t) i + 1] - sigmas[(size_t) i], s_);
            if (after) after(x, sigmas[(size_t) i + 1]);
            if (peek) {
                const float s_next = sigmas[(size_t) i + 1];
                peek(i + 1, [&](float* x0) {
                    moss_predicted(x0, x, out, n == 2 ? out + N : nullptr, k, N, g.cfg, g.slg_scale, s_next, s_);
                });
            }
            if (step && !step(i + 1, steps)) throw QiStopped();
        }
    } catch (...) {
        gpu::stream_sync(s_);
        ctx_[0].skip.clear();
        ctx_[1].skip = before;
        release(allocs_, out);
        throw;
    }
    gpu::stream_sync(s_);
    ctx_[1].skip = before;
    release(allocs_, out);
}

}  // namespace strata::image
