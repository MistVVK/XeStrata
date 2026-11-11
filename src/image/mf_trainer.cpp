// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_trainer.cpp - see include/strata/image/mf_trainer.hpp.
#include "strata/image/mf_trainer.hpp"
#include "strata/image/train_loop.hpp"

#include "strata/core/gpu.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/platform/mapped_file.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_qsa.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

using kernels::f16_from_f32;
using kernels::f32_from_f16;

// one device buffer, freed with the model
struct Owned {
    std::vector<void*> p;
    ~Owned() {
        for (void* q : p) gpu::free(q);
    }
    template <class T>
    T* get(int64_t n, void* s) {
        void* q = nullptr;
        if (!gpu::alloc_device(&q, (size_t) n * sizeof(T)))
            throw std::runtime_error("image engine (Mage-Flow training): GPU memory");
        if (!gpu::memset_async(q, 0, (size_t) n * sizeof(T), s))
            throw std::runtime_error("image engine (Mage-Flow training): zero");
        p.push_back(q);
        return static_cast<T*>(q);
    }
};

struct Model : MfTrainModel {
    Owned owned;
    prefill::Gemm gemm;
    train::Arena arena;   // the blocks' buffers, kept from block to block and step to step
    int64_t scratch_rows = 0;   // the most rows a step has had: the arena holds what they took
    // The blocks' twelve products through a LayerStream (Qwen-Image-2.1's trainer's): as many blocks as the memory
    // leaves stay in VRAM, the rest are copied in from RAM a block at a time.  Their rows as the products read them
    // (FP16 from the file's BF16, or an int8 base's rows and scales) are made once at open and kept in RAM; the
    // pointers of blk's products are set to the block's region as it is used (bind).
    LayerStream ws;
    std::vector<std::vector<uint8_t>> whost;
    std::vector<std::array<uint64_t, 12>> woff;
    bool int8 = false, stepped = false, budget_stepped = false;
    int64_t budget_rows = 0;
    ~Model() { ws.drop(); }
};

float* dev_dup(Model* m, const float* host, int64_t n) {
    float* d = m->owned.get<float>(n, m->stream);
    // the zero above runs on this stream; the copy below runs on the compute queue: without the sync the zero
    // can land after the copy and wipe the weights (usually sane values, sometimes a healthy-looking run)
    gpu::stream_sync(m->stream);
    if (!gpu::copy(d, host, (size_t) n * 4)) throw std::runtime_error("image engine (Mage-Flow training): copy");
    gpu::stream_sync(m->stream);
    return d;
}

uint16_t* dev_dup16(Model* m, const uint16_t* host, int64_t n) {
    uint16_t* d = m->owned.get<uint16_t>(n, m->stream);
    // as above: the zero and the copy run on different queues
    gpu::stream_sync(m->stream);
    if (!gpu::copy(d, host, (size_t) n * 2)) throw std::runtime_error("image engine (Mage-Flow training): copy");
    gpu::stream_sync(m->stream);
    return d;
}

// the file's bf16 rows as the FP16 bits the products read, and their transposed view
void load_product(Model* m, SafeTensors& st, const uint8_t* data, const std::string& name, MfTrainProduct& pr) {
    const SafeTensor* t = st.find(name + ".weight");
    if (t == nullptr) throw std::runtime_error("image engine (Mage-Flow training): no " + name);
    if (t->dtype != "BF16" || t->shape.size() != 2)
        throw std::runtime_error("image engine (Mage-Flow training): " + name + " is not a bf16 matrix");
    pr.N = t->shape[0];
    pr.K = t->shape[1];
    const auto* src = reinterpret_cast<const uint16_t*>(data + t->begin);
    std::vector<uint16_t> h((size_t) (pr.N * pr.K));
    for (int64_t i = 0; i < pr.N * pr.K; ++i) h[(size_t) i] = f16_from_f32(dit_rows::bf16_value(src[i]));
    pr.w = dev_dup16(m, h.data(), pr.N * pr.K);
    pr.wt = nullptr;   // the transposed views are read per block from wtScratch (all blocks do not fit beside w)
    pr.b = dev_dup(m, st.f32(name + ".bias").data(), pr.N);
}

void load_vec(Model* m, SafeTensors& st, const std::string& name, float*& out, int64_t want) {
    auto v = st.f32(name);
    if ((int64_t) v.size() != want)
        throw std::runtime_error("image engine (Mage-Flow training): " + name + " is not " +
                                 std::to_string(want) + " wide");
    out = dev_dup(m, v.data(), want);
}

/// the v1 target names per stream and product (SPEC D1 writes them as they are, no mapping table)
const char* const kTarget[2][6] = {{"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                                    "img_mlp.net.0.proj", "img_mlp.net.2"},
                                   {"attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
                                    "txt_mlp.net.0.proj", "txt_mlp.net.2"}};

}  // namespace

namespace {

// sd-scripts' networks/lokr.py factorization(dimension, -1): m * n = dimension, m <= n, m + n minimal.
void factorization(int64_t dim, int64_t& m, int64_t& n) {
    m = 1;
    n = dim;
    int64_t len = m + n;
    while (m < n) {
        int64_t nm = m + 1;
        while (dim % nm != 0) ++nm;
        int64_t nn = dim / nm;
        if (nm + nn > len || nm > dim) break;
        m = nm;
        n = nn;
        len = m + n;
    }
    if (m > n) std::swap(m, n);
}

}  // namespace

namespace {

// Block b's products' pointers in its region `base` (ws.begin's)
void bind(Model* m, int64_t b, const uint8_t* base) {
    MfTrainModel::Block& blk = m->blk[(size_t) b];
    const auto& off = m->woff[(size_t) b];
    for (int q = 0; q < 2; ++q)
        for (int i = 0; i < 6; ++i) {
            MfTrainProduct& pr = blk.prod[q][i];
            auto* at = const_cast<uint8_t*>(base) + off[(size_t) (q * 6 + i)];
            if (m->int8) {
                pr.w8 = reinterpret_cast<int8_t*>(at);
                pr.sw = reinterpret_cast<float*>(at + pr.N * pr.K);
                pr.w = nullptr;
            } else {
                pr.w = reinterpret_cast<uint16_t*>(at);
            }
        }
}

// The VRAM the blocks' products take (Qwen-Image-2.1's trainer's set_budget), set at open (no rows) and when a step
// brings more rows than any before, and once after the first step: the free memory and what the products hold now,
// less the step's own beside them (its boundaries in BF16 at least, the scratch's growth: train::scratch_growth,
// before a step has shown it a guess of 24 FP32 rows of C a row for the blocks' scratch and the attention's backward;
// 1.5 GiB of transients, 3 GiB more before the first step).
// XESTRATA_TRAIN_WEIGHTS_MIB sets it.
void set_budget(Model* m, int64_t Rj) {
    if (Rj <= m->budget_rows && m->budget_stepped == m->stepped && m->ws.vram() > 0) return;
    m->budget_rows = std::max(m->budget_rows, Rj);
    m->budget_stepped = m->stepped;
    const int64_t R = m->budget_rows;
    const uint64_t all = m->ws.total();
    uint64_t budget = all;
    if (const char* e = strata::env_get("TRAIN_WEIGHTS_MIB")) {
        budget = (uint64_t) std::atoll(e) << 20;
    } else {
        size_t free_b = 0, total = 0;
        gpu::stream_sync(m->stream);
        if (gpu::mem_info(&free_b, &total)) {
            const uint64_t have = (uint64_t) free_b + m->ws.vram();
            const uint64_t reserve = (uint64_t) ((m->nblk + 1) * R * m->C) * 2 +
                                     train::scratch_growth(m->arena.held(), R, m->scratch_rows,
                                                           (uint64_t) (24 * R * m->C) * 4) +
                                     (uint64_t{3} << 29) +
                                     (m->stepped ? 0 : uint64_t{3} << 30);
            budget = have > reserve ? have - reserve : 0;
        }
    }
    budget = std::min(all, std::max(budget, m->ws.min_budget()));
    if (budget < all)
        std::fprintf(stderr, "image engine (Mage-Flow training): the blocks' weights %llu of %llu MiB in VRAM, the "
                             "rest streamed from RAM\n",
                     (unsigned long long) (budget >> 20), (unsigned long long) (all >> 20));
    m->ws.configure(budget, m->stream);
}

}  // namespace

MfTrainModel* mf_trainer_open(const std::string& dit_path, int64_t rank, void* stream, std::string& err, bool int8) {
    std::unique_ptr<Model> m(new Model());
    m->stream = stream;
    try {
        if (!m->gemm.init(stream, 1 << 24, err)) return nullptr;
        SafeTensors st(dit_path);
        platform::MappedFile file(dit_path);
        const uint8_t* data = file.data() + st.data_base();
        auto shape_of = [&](const std::string& name, int dim) -> int64_t {
            const SafeTensor* t = st.find(name);
            if (t == nullptr || (int) t->shape.size() <= dim)
                throw std::runtime_error("image engine (Mage-Flow training): no " + name);
            return t->shape[(size_t) dim];
        };
        m->C = shape_of("transformer_blocks.0.attn.to_q.weight", 0);
        m->hd = shape_of("transformer_blocks.0.attn.norm_q.weight", 0);
        if (m->C % m->hd != 0) throw std::runtime_error("image engine (Mage-Flow training): heads * head_dim");
        m->heads = m->C / m->hd;
        m->F = shape_of("transformer_blocks.0.img_mlp.net.0.proj.weight", 0);
        m->txt_dim = shape_of("txt_in.weight", 1);
        m->in_ch = shape_of("img_in.weight", 1);
        m->axes = {16, 56, 56};
        load_product(m.get(), st, data, "img_in", m->img_in);
        load_product(m.get(), st, data, "txt_in", m->txt_in);
        load_product(m.get(), st, data, "time_text_embed.timestep_embedder.linear_1", m->t1);
        load_product(m.get(), st, data, "time_text_embed.timestep_embedder.linear_2", m->t2);
        load_product(m.get(), st, data, "norm_out.linear", m->nout);
        load_product(m.get(), st, data, "proj_out", m->pout);
        load_vec(m.get(), st, "txt_norm.weight", m->txt_norm, m->txt_dim);
        // MageTrail's shared low-rank modulation (bluvoll's mageflow-lowrank-modulation-v1): SiLU(temb) to `rank`
        // values once a step, FP32 as the file keeps it
        if (const SafeTensor* t = st.find("modulation_down.weight"); t != nullptr) {
            if (t->dtype != "F32" || t->shape.size() != 2 || t->shape[1] != m->C)
                throw std::runtime_error("image engine (Mage-Flow training): modulation_down is not F32 [rank][hidden]");
            m->mrank = t->shape[0];
            m->mdown_w = dev_dup(m.get(), st.f32("modulation_down.weight").data(), m->mrank * m->C);
            m->mdown_b = dev_dup(m.get(), st.f32("modulation_down.bias").data(), m->mrank);
        }
        m->int8 = int8;
        uint16_t* q16 = nullptr;   // an int8 base's rows made on the GPU a product at a time, through these
        int8_t* q8 = nullptr;
        float* qs = nullptr;
        Owned qtmp;
        int64_t qcap = 0;
        for (int b = 0;; ++b) {
            const std::string p = "transformer_blocks." + std::to_string(b) + ".";
            if (st.find(p + "img_mod.1.weight") == nullptr) break;
            MfTrainModel::Block blk;
            // the products' rows into RAM (FP16, or the int8 base's rows and scales), the biases into VRAM
            std::vector<std::vector<uint16_t>> h16(12);
            uint64_t hbytes = 0;
            for (int q = 0; q < 2; ++q)
                for (int i = 0; i < 6; ++i) {
                    const std::string name = p + kTarget[q][i];
                    const SafeTensor* t = st.find(name + ".weight");
                    if (t == nullptr || t->dtype != "BF16" || t->shape.size() != 2)
                        throw std::runtime_error("image engine (Mage-Flow training): " + name +
                                                 " is not a bf16 matrix");
                    MfTrainProduct& pr = blk.prod[q][i];
                    pr.N = t->shape[0];
                    pr.K = t->shape[1];
                    const auto* src = reinterpret_cast<const uint16_t*>(data + t->begin);
                    auto& h = h16[(size_t) (q * 6 + i)];
                    h.resize((size_t) (pr.N * pr.K));
                    for (int64_t k = 0; k < pr.N * pr.K; ++k)
                        h[(size_t) k] = f16_from_f32(dit_rows::bf16_value(src[k]));
                    pr.b = dev_dup(m.get(), st.f32(name + ".bias").data(), pr.N);
                    hbytes += int8 ? (uint64_t) (pr.N * pr.K + 4 * pr.N) : (uint64_t) (pr.N * pr.K) * 2;
                }
            m->whost.emplace_back((size_t) hbytes);
            uint8_t* hp = m->whost.back().data();
            std::vector<LayerStream::Span> spans;
            for (int q = 0; q < 2; ++q)
                for (int i = 0; i < 6; ++i) {
                    const MfTrainProduct& pr = blk.prod[q][i];
                    const auto& h = h16[(size_t) (q * 6 + i)];
                    const int64_t n = pr.N * pr.K;
                    uint64_t bytes = (uint64_t) n * 2;
                    if (int8) {
                        if (n > qcap) {
                            q16 = qtmp.get<uint16_t>(n, stream);
                            q8 = qtmp.get<int8_t>(n, stream);
                            qs = qtmp.get<float>(std::max(m->F, 6 * m->C), stream);
                            qcap = n;
                        }
                        gpu::stream_sync(stream);
                        gpu::copy(q16, h.data(), (size_t) n * 2);
                        train::w8_quantize(q16, pr.N, pr.K, q8, qs, stream);
                        gpu::stream_sync(stream);
                        gpu::copy(hp, q8, (size_t) n);
                        gpu::copy(hp + n, qs, (size_t) pr.N * 4);
                        bytes = (uint64_t) (n + 4 * pr.N);
                    } else {
                        std::memcpy(hp, h.data(), (size_t) n * 2);
                    }
                    spans.push_back({hp, bytes, 0});
                    hp += bytes;
                }
            const int layer = m->ws.add(spans);
            std::array<uint64_t, 12> off{};
            for (int k = 0; k < 12; ++k) off[(size_t) k] = m->ws.offset(layer, k);
            m->woff.push_back(off);
            if (m->mrank > 0) {   // MageTrail's low-rank modulation: each block's FP32 [6 C][rank] from the shared values
                for (int q = 0; q < 2; ++q) {
                    const std::string n = p + (q == 0 ? "img_mod.1" : "txt_mod.1");
                    const SafeTensor* t = st.find(n + ".weight");
                    if (t == nullptr || t->dtype != "F32" || t->shape.size() != 2 || t->shape[1] != m->mrank)
                        throw std::runtime_error("image engine (Mage-Flow training): " + n + " is not F32 [6 hidden][rank]");
                    blk.mod[q].N = t->shape[0];
                    blk.mod[q].K = m->mrank;
                    blk.modlow[q] = dev_dup(m.get(), st.f32(n + ".weight").data(), blk.mod[q].N * m->mrank);
                    blk.mod[q].b = dev_dup(m.get(), st.f32(n + ".bias").data(), blk.mod[q].N);
                }
            }
            if (m->mrank == 0) {
                load_product(m.get(), st, data, p + "img_mod.1", blk.mod[0]);
                load_product(m.get(), st, data, p + "txt_mod.1", blk.mod[1]);
            }
            load_vec(m.get(), st, p + "attn.norm_q.weight", blk.nq[0], m->hd);
            load_vec(m.get(), st, p + "attn.norm_k.weight", blk.nk[0], m->hd);
            load_vec(m.get(), st, p + "attn.norm_added_q.weight", blk.nq[1], m->hd);
            load_vec(m.get(), st, p + "attn.norm_added_k.weight", blk.nk[1], m->hd);
            if (rank > 0) {
                blk.adap = new MfTrainAdapter();
                blk.adap->rank = rank;
                for (int st = 0; st < 2; ++st)
                    for (int i = 0; i < 6; ++i) {
                        train::Adapter& az = blk.adap->ad[st][i];
                        const int64_t N = blk.prod[st][i].N, K = blk.prod[st][i].K;
                        az.method = train::AdapterMethod::lora;   // the trainer sets DoRA per product later
                        az.rank = rank;
                        int64_t ol, ok, im, inn;
                        factorization(N, ol, ok);
                        factorization(K, im, inn);
                        az.ol = ol;
                        az.ok = ok;
                        az.im = im;
                        az.inn = inn;
                        az.a = m->owned.get<uint16_t>(rank * K, stream);
                        az.at = m->owned.get<uint16_t>(rank * K, stream);
                        az.b = m->owned.get<uint16_t>(N * rank, stream);
                        az.bt = m->owned.get<uint16_t>(N * rank, stream);
                        az.dA = m->owned.get<float>(rank * K, stream);
                        az.dB = m->owned.get<float>(N * rank, stream);
                        az.dm = m->owned.get<float>(N, stream);
                        az.norms = m->owned.get<float>(N, stream);
                        az.w1 = m->owned.get<uint16_t>(ol * im, stream);
                        az.w1t = m->owned.get<uint16_t>(ol * im, stream);
                        az.w2a = m->owned.get<uint16_t>(ok * rank, stream);
                        az.w2b = m->owned.get<uint16_t>(rank * inn, stream);
                        az.dW1 = m->owned.get<float>(ol * im, stream);
                        az.dW2a = m->owned.get<float>(ok * rank, stream);
                        az.dW2b = m->owned.get<float>(rank * inn, stream);
                        MfTrainAdapter& tz = *blk.adap;
                        tz.A[st][i] = m->owned.get<float>(rank * K, stream);
                        tz.B[st][i] = m->owned.get<float>(N * rank, stream);
                        tz.M[st][i] = m->owned.get<float>(N, stream);
                        tz.mA[st][i] = m->owned.get<float>(rank * K, stream);
                        tz.vA[st][i] = m->owned.get<float>(rank * K, stream);
                        tz.mB[st][i] = m->owned.get<float>(N * rank, stream);
                        tz.vB[st][i] = m->owned.get<float>(N * rank, stream);
                        tz.mm[st][i] = m->owned.get<float>(N, stream);
                        tz.vm[st][i] = m->owned.get<float>(N, stream);
                        tz.W1[st][i] = m->owned.get<float>(ol * im, stream);
                        tz.W2A[st][i] = m->owned.get<float>(ok * rank, stream);
                        tz.W2B[st][i] = m->owned.get<float>(rank * inn, stream);
                        tz.mW1[st][i] = m->owned.get<float>(ol * im, stream);
                        tz.vW1[st][i] = m->owned.get<float>(ol * im, stream);
                        tz.mW2A[st][i] = m->owned.get<float>(ok * rank, stream);
                        tz.vW2A[st][i] = m->owned.get<float>(ok * rank, stream);
                        tz.mW2B[st][i] = m->owned.get<float>(rank * inn, stream);
                        tz.vW2B[st][i] = m->owned.get<float>(rank * inn, stream);
                    }
            }
            m->blk.push_back(blk);
        }
        m->nblk = (int64_t) m->blk.size();
        if (m->nblk == 0) throw std::runtime_error("image engine (Mage-Flow training): no transformer blocks");
        // one block's transposed views, reused for every block (all blocks' do not fit beside the weights)
        {
            int64_t total = 0;
            for (int q = 0; q < 2; ++q)
                for (int i = 0; i < 6; ++i) {
                    m->wtOff[q][i] = total;
                    total += m->blk[0].prod[q][i].K * m->blk[0].prod[q][i].N;
                }
            m->wtScratch = m->owned.get<uint16_t>(total, m->stream);
            if (int8) m->w8tmp = m->owned.get<uint16_t>(m->F * m->C, m->stream);
        }
        // the modulations' linears stay on the host: one row a step, read once
        (void) data;
        set_budget(m.get(), 0);
    } catch (const std::exception& e) {
        err = e.what();
        return nullptr;
    }
    return m.release();
}

void mf_trainer_close(MfTrainModel* base) {
    std::unique_ptr<Model> m(static_cast<Model*>(base));
    for (auto& b : m->blk) delete b.adap;
}

MfTrainSample mf_trainer_sample(const std::string& dir, const std::string& stem, int64_t txt_dim) {
    MfTrainSample s;
    auto read = [&](const std::string& name) -> std::vector<float> {
        std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
        if (!f) throw std::runtime_error("image engine (Mage-Flow training): no " + name);
        std::stringstream js;
        std::ifstream j(dir + "/" + name + ".json");
        js << j.rdbuf();
        const std::string t = js.str();
        const size_t b = t.find("\"shape\"");
        std::vector<int64_t> sh;
        if (b != std::string::npos) {
            const size_t o = t.find('[', b), c = t.find(']', o);
            std::stringstream ns(t.substr(o + 1, c - o - 1));
            std::string tok;
            while (std::getline(ns, tok, ',')) sh.push_back(std::stoll(tok));
        }
        std::vector<float> v((size_t) (std::filesystem::file_size(dir + "/" + name + ".bin") / 4));
        if (!f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (v.size() * 4)))
            throw std::runtime_error("image engine (Mage-Flow training): short " + name);
        if (t.find("\"f16\"") != std::string::npos) {
            std::vector<float> w(v.size());
            const auto* h = reinterpret_cast<const uint16_t*>(v.data());
            for (size_t i = 0; i < v.size(); ++i) w[i] = f32_from_f16(h[i]);
            return w;
        }
        return v;
    };
    s.lat = read(stem + "_lat");
    if (std::filesystem::exists(dir + "/" + stem + "_lat_sd.bin")) s.lat_sd = read(stem + "_lat_sd");
    s.te = read(stem + "_te");
    s.uncond = read(stem + "_uncond");
    // the bucket's own rows and columns come from the manifest
    std::ifstream mf(dir + "/manifest.tsv");
    std::string line;
    while (std::getline(mf, line)) {
        std::stringstream ls(line);
        std::string tok;
        std::getline(ls, tok, '\t');
        if (tok != stem) continue;
        std::getline(ls, tok, '\t');
        s.w = std::stoll(tok);
        std::getline(ls, tok, '\t');
        s.h = std::stoll(tok);
    }
    if (s.w <= 0 || s.h <= 0)
        throw std::runtime_error("image engine (Mage-Flow training): " + stem + " is not in the manifest");
    // the dump lists its shape fastest dim first: [ch][tokens] and [txt_dim][rows]
    s.tokens = s.w * s.h;
    s.ch = (int64_t) s.lat.size() / s.tokens;
    s.rows = (int64_t) (s.te.size() / (size_t) txt_dim);
    s.uncond_rows = (int64_t) (s.uncond.size() / (size_t) txt_dim);
    if ((int64_t) s.lat.size() != s.tokens * s.ch || (int64_t) s.te.size() != s.rows * txt_dim ||
        (int64_t) s.uncond.size() != s.uncond_rows * txt_dim)
        throw std::runtime_error("image engine (Mage-Flow training): " + stem + " is not this model's shape");
    return s;
}

namespace {

// the RoPE tables of one h x w image (rope_tables' math, a single frame): cos/sin [h*w][hd/2]
void rope_tables_host(int64_t h, int64_t w, int64_t hd, const std::vector<int64_t>& axes, std::vector<float>& c,
                      std::vector<float>& s) {
    const int64_t half = hd / 2, a0 = axes[0] / 2, a1 = axes[1] / 2;
    const int64_t y0 = h - h / 2, x0 = w - w / 2;
    c.resize((size_t) (h * w * half));
    s.resize((size_t) (h * w * half));
    int64_t row = 0;
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x, ++row)
            for (int64_t j = 0; j < half; ++j) {
                double pos = 0.0, dim = (double) axes[0], k = (double) j;
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
                c[(size_t) row * half + j] = (float) std::cos(ang);
                s[(size_t) row * half + j] = (float) std::sin(ang);
            }
}

}  // namespace

MfTrainCond mf_trainer_cond(MfTrainModel* base, float sigma, int64_t h, int64_t w, prefill::Gemm& gemm) {
    Model* m = static_cast<Model*>(base);
    void* s = m->stream;
    MfTrainCond c;
    Owned tmp;
    const int64_t C = m->C;
    // the sinusoidal row, as the engine reads it (bf16-rounded, then FP16)
    const float t = dit_rows::round_bf16(sigma);
    std::vector<uint16_t> e(256);
    for (int i = 0; i < 128; ++i) {
        const float f = dit_rows::round_bf16(std::exp(-std::log(10000.f) * (float) i / 128.f));
        const float a = 1000.f * (t * f);
        e[(size_t) i] = f16_from_f32(dit_rows::round_bf16(std::cos(a)));
        e[128 + (size_t) i] = f16_from_f32(dit_rows::round_bf16(std::sin(a)));
    }
    uint16_t* e16 = tmp.get<uint16_t>(256, s);
    float* temb = tmp.get<float>(C, s);
    uint16_t* tsil16 = tmp.get<uint16_t>(C, s);
    // the zero above runs on this stream; the copy below runs on the compute queue (see dev_dup)
    gpu::stream_sync(s);
    if (!gpu::copy(e16, e.data(), size_t{256} * 2)) throw std::runtime_error("image engine (Mage-Flow training): copy");
    gemm.f16(e16, m->t1.w, temb, 1, C, 256);
    dit_rows::rows_back(temb, nullptr, m->t1.b, 1, C, s);
    qi_silu(temb, C, tsil16, s);   // the inference's own (mf_dit.cpp's set_time)
    gemm.f16(tsil16, m->t2.w, temb, 1, C, C);
    dit_rows::rows_back(temb, nullptr, m->t2.b, 1, C, s);
    qi_silu(temb, C, tsil16, s);   // the inference's own (mf_dit.cpp's set_time)
    // the modulations and the norm_out pair read tsil's FP16 rows, unscaled (rows_back with no rs); the
    // modulations' LoRA reads the same values, kept in FP32 for its gradient
    c.tsil = m->owned.get<float>(C, s);
    {
        float* ts = c.tsil;
        const uint16_t* t16 = tsil16;
        dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) C),
                                       [=](sycl::id<1> id) { ts[id[0]] = kernels::f32_from_f16(t16[id[0]]); });
    }
    const int64_t mrank = m->blk.empty() || m->blk[0].adap == nullptr ? 1 : m->blk[0].adap->rank;
    float* hr = tmp.get<float>(mrank, s);
    // MageTrail's: the shared low-rank values from SiLU(temb) in FP32 (qi_silu left it in temb), as the engine's
    // F32 MMVF reads them
    float* mlow = nullptr;
    if (m->mrank > 0) {
        mlow = tmp.get<float>(m->mrank, s);
        train::row_down(m->mdown_w, temb, m->mrank, C, mlow, s);
        dit_rows::rows_back(mlow, nullptr, m->mdown_b, 1, m->mrank, s);
    }
    c.mod.resize((size_t) m->nblk * 2);
    for (int64_t b = 0; b < m->nblk; ++b) {
        for (int st = 0; st < 2; ++st) {
            float* mod = m->owned.get<float>(6 * C, s);
            const MfTrainProduct& mp = m->blk[(size_t) b].mod[st];
            if (m->mrank > 0) train::row_down(m->blk[(size_t) b].modlow[st], mlow, 6 * C, m->mrank, mod, s);
            else gemm.f16(tsil16, mp.w, mod, 1, 6 * C, C);
            dit_rows::rows_back(mod, nullptr, mp.b, 1, 6 * C, s);
            const MfTrainAdapter* az = m->blk[(size_t) b].adap;
            if (az != nullptr && az->modA[st] != nullptr)
                train::row_lora_fwd(az->modA[st], az->modB[st], c.tsil, az->alpha / (float) az->rank, az->rank, 6 * C, C,
                             mod, hr, s);
            c.mod[(size_t) b * 2 + st] = mod;
        }
    }
    c.nmod = m->owned.get<float>(2 * C, s);
    gemm.f16(tsil16, m->nout.w, c.nmod, 1, 2 * C, C);
    dit_rows::rows_back(c.nmod, nullptr, m->nout.b, 1, 2 * C, s);
    std::vector<float> ch, sh;
    rope_tables_host(h, w, m->hd, m->axes, ch, sh);
    c.cos = dev_dup(m, ch.data(), (int64_t) ch.size());
    c.sin = dev_dup(m, sh.data(), (int64_t) sh.size());
    gpu::stream_sync(s);
    return c;
}

void mf_trainer_free_cond(MfTrainModel* base, MfTrainCond& c) {
    Model* m = static_cast<Model*>(base);
    auto drop = [&](float* q) {
        if (q == nullptr) return;
        auto& v = m->owned.p;
        v.erase(std::remove(v.begin(), v.end(), (void*) q), v.end());
        gpu::free(q);
    };
    drop(c.temb);
    for (float* q : c.mod) drop(q);
    drop(c.nmod);
    drop(c.tsil);
    drop(c.cos);
    drop(c.sin);
    c = MfTrainCond();
}

namespace {

// x = a * x0 + b * x1 (the noised latent)
void axpby(const float* x0, const float* x1, float a, float b, int64_t n, float* y, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        y[id[0]] = a * x0[id[0]] + b * x1[id[0]];
    });
}

}  // namespace

namespace {

// one step's scratch that outlives the forward (the backward frees it): boundaries, inputs, velocity
struct KeepTmp {
    Owned tmp;   // freed with the step
};

}  // namespace

// Whether this step keeps its blocks' inputs in BF16 (XESTRATA_TRAIN_INPUTS16=0 or 1, or by the memory), as the
// MOSS-Image trainer does: when the free memory falls short of their FP32 bytes, the arena's growth to this step's Rj
// rows (in proportion to the most rows so far, 10% over; before a step has shown it, a guess of 12 FP32 rows' worth
// a boundary row: the attention's backward workspace and a block's products) and 1.5 GiB of transients.  The
// forward rounds each input to BF16 as it keeps it, so the backward reads what the forward used.
static bool mf_inputs_half(Model* m, int64_t Rj) {
    if (const char* e = strata::env_get("TRAIN_INPUTS16")) return std::atoi(e) != 0;
    size_t free_b = 0, total = 0;
    gpu::stream_sync(m->stream);
    if (!gpu::mem_info(&free_b, &total)) return false;
    const size_t row = (size_t) m->C * 4, held = m->arena.held();
    const size_t scratch =
        m->scratch_rows > 0
            ? (size_t) (1.1 * (double) held * (double) std::max(Rj, m->scratch_rows) / (double) m->scratch_rows)
            : held + (size_t) (12 * Rj) * row;
    const size_t need = (size_t) ((m->nblk + 1) * Rj) * row + (scratch > held ? scratch - held : 0) +
                        (size_t{3} << 29);
    return free_b < need;
}

static float forward_impl(Model* m, const MfTrainSample& s, const MfTrainCond& c, const float* eps, float sigma,
                          float* v_pred, MfTrainState* keep, prefill::Gemm& gemm) {
    void* st = m->stream;
    const int64_t C = m->C, R = s.tokens, Rt = s.rows, Rj = R + Rt;
    Owned tmp;
    auto keep_get = [&](int64_t n) -> float* {
        return keep != nullptr ? m->owned.get<float>(n, st) : tmp.get<float>(n, st);
    };
    // the noised latent, then the image stream's rows
    float* x0 = keep_get(R * 128);
    float* xe = keep_get(R * 128);
    float* xs = tmp.get<float>(R * 128, st);
    // the zeros above run on this stream; the copies below run on the compute queue (see dev_dup)
    gpu::stream_sync(st);
    if (!gpu::copy(x0, s.lat.data(), (size_t) R * 128 * 4) || !gpu::copy(xe, eps, (size_t) R * 128 * 4))
        throw std::runtime_error("image engine (Mage-Flow training): copy");
    axpby(x0, xe, 1.f - sigma, sigma, R * 128, xs, st);
    uint16_t* xs16 = tmp.get<uint16_t>(R * 128, st);
    float* xsrs = tmp.get<float>(R, st);
    dit_rows::rows16(dit_rows::Plain{xs, 128}, R, 128, xs16, xsrs, st);
    float* img = tmp.get<float>(R * C, st);
    gemm.f16(xs16, m->img_in.w, img, R, C, 128);
    dit_rows::rows_back(img, xsrs, m->img_in.b, R, C, st);
    // the text stream's rows: the states through their norm and projection
    float* te = tmp.get<float>(Rt * m->txt_dim, st);
    // as above: the zeros run on this stream, the copy on the compute queue
    gpu::stream_sync(st);
    if (!gpu::copy(te, s.te.data(), (size_t) Rt * m->txt_dim * 4))
        throw std::runtime_error("image engine (Mage-Flow training): copy");
    // the engines' one RMSNorm into FP16, as the inference's set_context reads the text states
    uint16_t* te16 = tmp.get<uint16_t>(Rt * m->txt_dim, st);
    kernels::rms_norm_rows(te, m->txt_norm, 1, nullptr, te16, (int) m->txt_dim, (int) Rt, 1e-6f, st);
    float* txt = tmp.get<float>(Rt * C, st);
    gemm.f16(te16, m->txt_in.w, txt, Rt, C, m->txt_dim);
    dit_rows::rows_back(txt, nullptr, m->txt_in.b, Rt, C, st);
    // the boundaries: block i reads bounds[i] and writes bounds[i + 1]; where the memory is short, in BF16
    // (mf_inputs_half), two FP32 boundaries passed between and each block's input rounded to BF16 as it is kept
    float* bounds = nullptr;
    uint16_t* bounds16 = nullptr;
    float* pass = nullptr;   // the forward's two boundaries passed between (BF16, or the forward alone)
    set_budget(m, Rj);
    if (keep != nullptr && mf_inputs_half(m, Rj)) {
        bounds16 = m->owned.get<uint16_t>((m->nblk + 1) * Rj * C, st);
        keep->bounds16 = bounds16;
        keep->last = m->owned.get<float>(R * C, st);
    } else if (keep != nullptr) {
        bounds = m->owned.get<float>((m->nblk + 1) * Rj * C, st);
        keep->bounds = bounds;
    }
    if (bounds == nullptr) pass = tmp.get<float>(2 * Rj * C, st);
    {
        float* b0 = bounds != nullptr ? bounds : pass;
        // img/txt are this stream's rows; the copies below run on the compute queue: without the sync they can
        // read ahead of the kernels that wrote them
        gpu::stream_sync(st);
        if (!gpu::copy(b0, img, (size_t) R * C * 4) || !gpu::copy(b0 + R * C, txt, (size_t) Rt * C * 4))
            throw std::runtime_error("image engine (Mage-Flow training): bounds");
        if (bounds16 != nullptr) train::bf16_keep(b0, Rj * C, bounds16, st);
    }
    if (keep != nullptr) {
        keep->R = R;
        keep->Rt = Rt;
        keep->x0 = x0;
        keep->xe = xe;
        keep->v_pred = m->owned.get<float>(R * m->in_ch, st);
        keep->dv_pred = m->owned.get<float>(R * m->in_ch, st);
        v_pred = keep->v_pred;
    }
    // the twelve blocks, joint attention between the streams (the text's rows first)
    float* bi = bounds != nullptr ? bounds : pass;
    float* bt = bi + R * C;
    for (int64_t b = 0; b < m->nblk; ++b) {
        // the next block's products fetched meanwhile; past the last, the backward's second when it follows
        const int nb = (int) m->nblk;
        bind(m, b, keep != nullptr ? m->ws.begin((int) b, (int) (b + 1 < nb ? b + 1 : nb - 2)) : m->ws.begin((int) b));
        const MfTrainModel::Block& blk = m->blk[(size_t) b];
        for (int q = 0; q < 2; ++q)
            for (int i = 0; i < 6; ++i) {
                const MfTrainProduct& pr = blk.prod[q][i];
                if (pr.w8 != nullptr)   // the straight-through weight: the int8 values rotated back
                    train::w8_dequant_t(pr.w8, pr.sw, pr.N, pr.K, m->w8tmp, m->wtScratch + m->wtOff[q][i], st);
                else train::transpose_f16(pr.w, pr.N, pr.K, m->wtScratch + m->wtOff[q][i], st);
            }
        MfBlockRun run;
        run.C = C;
        run.heads = m->heads;
        run.hd = m->hd;
        run.F = m->F;
        run.rows_img = R;
        run.rows_txt = Rt;
        run.img = bi;
        run.txt = bt;
        run.cos = c.cos;
        run.sin = c.sin;
        float* no = bounds != nullptr ? bounds + ((size_t) b + 1) * Rj * C : pass + ((b + 1) % 2) * Rj * C;
        float* nt = no + R * C;
        run.out_img = no;
        run.out_txt = nt;
        run.d_out_img = nullptr;
        run.d_out_txt = nullptr;
        run.dimg = nullptr;
        run.dtxt = nullptr;
        for (int q = 0; q < 2; ++q) {
            MfBlockW* w = q == 0 ? &run.img_w : &run.txt_w;
            for (int i = 0; i < 6; ++i) {
                const MfTrainProduct& pr = blk.prod[q][i];
                w->N[i] = pr.N;
                w->K[i] = pr.K;
                w->w[i] = pr.w;
                w->w8[i] = pr.w8;
                w->sw[i] = pr.sw;
                w->wt[i] = m->wtScratch + m->wtOff[q][i];
                w->b[i] = pr.b;
                w->ad[i] = (blk.adap != nullptr) ? &blk.adap->ad[q][i] : nullptr;
            }
            w->nq = blk.nq[q];
            w->nk = blk.nk[q];
            w->mod = c.mod[(size_t) b * 2 + q];
            w->dnq = nullptr;
            w->dnk = nullptr;
            for (int i = 0; i < 6; ++i) {
                w->dw[i] = nullptr;
                w->db[i] = nullptr;
            }
        }
        mf_block(run, gemm, st, &m->arena);
        m->ws.end((int) b);
        if (bounds16 != nullptr) train::bf16_keep(no, Rj * C, bounds16 + ((size_t) b + 1) * Rj * C, st);
        bi = no;
        bt = nt;
    }
    if (bounds16 != nullptr) {   // the head's input, and the backward's (rounded as kept)
        gpu::stream_sync(st);
        if (!gpu::copy(keep->last, bi, (size_t) R * C * 4))
            throw std::runtime_error("image engine (Mage-Flow training): copy");
    }
    // the output norm and projection to the latent's channels
    float* nsh = tmp.get<float>(R * C, st);
    float* nsc = tmp.get<float>(R * C, st);
    const float* nmod = c.nmod;
    dit_rows::q_of(st).parallel_for(sycl::range<1>((size_t) R * C), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / C, cc = i - t * C;
        nsh[id[0]] = nmod[C + cc];
        nsc[id[0]] = nmod[cc];
    });
    uint16_t* ln16 = tmp.get<uint16_t>(R * C, st);
    float* lnrs = tmp.get<float>(R, st);
    // LayerNorm(no affine) * (1 + scale) + shift, as rows16 does it (see ln_mod16)
    {
        float* mean = tmp.get<float>(R, st);
        float* inv = tmp.get<float>(R, st);
        dit_rows::q_of(st).parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) R * 256), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) {
                const int64_t t = static_cast<int64_t>(it.get_group_linear_id());
                float sum = 0.f;
                for (auto cc = static_cast<int64_t>(it.get_local_linear_id()); cc < C; cc += 256)
                    sum += bi[t * C + cc];
                const float mn = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>()) / (float) C;
                float va = 0.f;
                for (auto cc = static_cast<int64_t>(it.get_local_linear_id()); cc < C; cc += 256) {
                    const float d = bi[t * C + cc] - mn;
                    va += d * d;
                }
                const float iv =
                    1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), va, sycl::plus<float>()) / (float) C +
                                     1e-6f);
                float mx = 0.f;
                for (auto cc = static_cast<int64_t>(it.get_local_linear_id()); cc < C; cc += 256)
                    mx = sycl::fmax(mx, sycl::fabs((bi[t * C + cc] - mn) * iv * (1.f + nsc[t * C + cc]) +
                                                   nsh[t * C + cc]));
                const float sc = dit_rows::pow2_scale(sycl::reduce_over_group(it.get_group(), mx,
                                                                              sycl::maximum<float>()));
                for (auto cc = static_cast<int64_t>(it.get_local_linear_id()); cc < C; cc += 256)
                    ln16[t * C + cc] = kernels::f16_from_f32(
                        ((bi[t * C + cc] - mn) * iv * (1.f + nsc[t * C + cc]) + nsh[t * C + cc]) / sc);
                if (it.get_local_linear_id() == 0) {
                    mean[t] = mn;
                    inv[t] = iv;
                    lnrs[t] = sc;
                }
            });
    }
    gemm.f16(ln16, m->pout.w, v_pred, R, m->in_ch, C);
    dit_rows::rows_back(v_pred, lnrs, m->pout.b, R, m->in_ch, st);
    const float loss = train::flow_loss(v_pred, x0, xe, R * m->in_ch, keep != nullptr ? keep->dv_pred : nullptr,
                                    nullptr, nullptr, st);
    gpu::stream_sync(st);
    return loss;
}

float mf_trainer_forward(MfTrainModel* base, const MfTrainSample& s, const MfTrainCond& c, const float* eps,
                         float sigma, float* v_pred, std::vector<float>* bounds, prefill::Gemm& gemm) {
    (void) bounds;
    return forward_impl(static_cast<Model*>(base), s, c, eps, sigma, v_pred, nullptr, gemm);
}

float mf_trainer_forward_keep(MfTrainModel* base, const MfTrainSample& s, const MfTrainCond& c, const float* eps,
                              float sigma, MfTrainState& st, prefill::Gemm& gemm) {
    return forward_impl(static_cast<Model*>(base), s, c, eps, sigma, nullptr, &st, gemm);
}

void mf_trainer_free_state(MfTrainModel* base, MfTrainState& s) {
    Model* m = static_cast<Model*>(base);
    auto drop = [&](float* q) {
        if (q == nullptr) return;
        auto& v = m->owned.p;
        v.erase(std::remove(v.begin(), v.end(), (void*) q), v.end());
        gpu::free(q);
    };
    drop(s.x0);
    drop(s.xe);
    drop(s.v_pred);
    drop(s.dv_pred);
    drop(s.bounds);
    drop(s.last);
    if (s.bounds16 != nullptr) {
        auto& v = m->owned.p;
        v.erase(std::remove(v.begin(), v.end(), (void*) s.bounds16), v.end());
        gpu::free(s.bounds16);
    }
    s = MfTrainState();
}

namespace {

// the base weight's row norms (the DoRA divisor), over the FP16 values the products read
void rownorm_f16(const uint16_t* w, int64_t N, int64_t K, float* norms, void* s) {
    constexpr int64_t kWG = 256;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) N * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t n = static_cast<int64_t>(it.get_group_linear_id());
            float ss = 0.f;
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < K; c += kWG) {
                const float v = f32_from_f16(w[n * K + c]);
                ss += v * v;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            if (it.get_local_linear_id() == 0) norms[n] = sycl::sqrt(ss);
        });
}

}  // namespace

void mf_trainer_init_adapters(MfTrainModel* base, int method, float alpha, uint64_t seed, bool mod_targets) {
    Model* m = static_cast<Model*>(base);
    void* s = m->stream;
    std::mt19937_64 rng(seed);
    for (int64_t b = 0; b < m->nblk; ++b) {
        MfTrainModel::Block& blk = m->blk[(size_t) b];
        if (blk.adap == nullptr) continue;
        bind(m, b, m->ws.begin((int) b));   // the DoRA norms below read the base's rows
        MfTrainAdapter& tz = *blk.adap;
        tz.method = method == 0 ? train::AdapterMethod::lora : method == 1 ? train::AdapterMethod::dora : train::AdapterMethod::lokr;
        tz.alpha = alpha;
        tz.scale = alpha / (float) tz.rank;
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        for (int st = 0; st < 2; ++st)
            for (int i = 0; i < 6; ++i) {
                const int64_t N = blk.prod[st][i].N, K = blk.prod[st][i].K, rk = tz.rank;
                train::Adapter& az = tz.ad[st][i];
                az.method = tz.method;
                az.scale = tz.scale;
                if (tz.method == train::AdapterMethod::lokr) {
                    // LyCORIS LoKr init: w1 and w2a kaiming-uniform (bound 1/sqrt(fan_in)), w2b zero
                    // (the delta starts at nothing); the f16 views follow the masters
                    const int64_t ol = az.ol, ok = az.ok, im = az.im, inn = az.inn;
                    const float b1 = 1.f / std::sqrt((float) im), ba = 1.f / std::sqrt((float) rk);
                    std::vector<float> hw1((size_t) ol * im), hw2a((size_t) ok * rk);
                    for (float& v : hw1) v = uni(rng) * b1;
                    for (float& v : hw2a) v = uni(rng) * ba;
                    if (!gpu::copy(tz.W1[st][i], hw1.data(), hw1.size() * 4) ||
                        !gpu::copy(tz.W2A[st][i], hw2a.data(), hw2a.size() * 4))
                        throw std::runtime_error("image engine (Mage-Flow training): copy");
                    gpu::stream_sync(s);
                    kernels::f32_to_f16_bulk(tz.W1[st][i], az.w1, ol * im, s);
                    train::transpose_f16(az.w1, ol, im, az.w1t, s);
                    kernels::f32_to_f16_bulk(tz.W2A[st][i], az.w2a, ok * rk, s);
                    kernels::f32_to_f16_bulk(tz.W2B[st][i], az.w2b, rk * inn, s);
                    // W2B stays zero (Owned zeroes)
                    continue;
                }
                if (blk.prod[st][i].w != nullptr)   // (an int8 base takes no DoRA: the trainer refuses the pair)
                    rownorm_f16(blk.prod[st][i].w, N, K, az.norms, s);
                // A kaiming-uniform (bound 1/sqrt(fan_in), the common LoRA init), B = 0 (the delta starts at nothing),
                // m = norms (c starts at one).  A drawn N(0, 1/sqrt(rank)) instead is about 24 times larger at
                // Mage-Flow's K of 3072: B's every step moves the delta that much more, and the run diverged at lr 1e-4
                // (2026-11-09, the base model's character run: the gradient norm 1 -> 12 at update 72)
                const float ba = 1.f / std::sqrt((float) K);
                std::vector<float> ha((size_t) rk * K);
                for (float& v : ha) v = uni(rng) * ba;
                if (!gpu::copy(tz.A[st][i], ha.data(), ha.size() * 4))
                    throw std::runtime_error("image engine (Mage-Flow training): copy");
                gpu::stream_sync(s);
                az.m = tz.M[st][i];
                if (!gpu::copy(tz.M[st][i], az.norms, (size_t) N * 4))
                    throw std::runtime_error("image engine (Mage-Flow training): copy");
                kernels::f32_to_f16_bulk(tz.A[st][i], az.a, rk * K, s);
                train::transpose_f16(az.a, rk, K, az.at, s);
                // B stays zero (Owned zeroes); its views stay zero with it
            }
        if (mod_targets && m->mrank == 0)   // the modulations' plain LoRA: A kaiming-uniform, B zero, as above
                                            // (MageTrail's low-rank form takes none: the engine adds none to it)
            for (int st = 0; st < 2; ++st) {
                const int64_t C = m->C, rk = tz.rank;
                tz.modA[st] = m->owned.get<float>(rk * C, s);
                tz.modB[st] = m->owned.get<float>(6 * C * rk, s);
                tz.dmodA[st] = m->owned.get<float>(rk * C, s);
                tz.dmodB[st] = m->owned.get<float>(6 * C * rk, s);
                tz.mmodA[st] = m->owned.get<float>(rk * C, s);
                tz.vmodA[st] = m->owned.get<float>(rk * C, s);
                tz.mmodB[st] = m->owned.get<float>(6 * C * rk, s);
                tz.vmodB[st] = m->owned.get<float>(6 * C * rk, s);
                const float ba = 1.f / std::sqrt((float) C);
                std::vector<float> ha((size_t) (rk * C));
                for (float& v : ha) v = uni(rng) * ba;
                gpu::stream_sync(s);
                if (!gpu::copy(tz.modA[st], ha.data(), ha.size() * 4))
                    throw std::runtime_error("image engine (Mage-Flow training): copy");
            }
        m->ws.end((int) b);
    }
    gpu::stream_sync(s);
}


namespace {

// the norm's per-row statistics, recomputed for the backward (the forward's are gone with its temps)
void ln_stats(const float* x, int64_t R, int64_t C, float* mean, float* inv, void* s) {
    constexpr int64_t kWG = 256;
    constexpr float kEps = 1e-6f;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) R * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t t = static_cast<int64_t>(it.get_group_linear_id());
            float sum = 0.f, va = 0.f;
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < C; c += kWG) sum += x[t * C + c];
            const float mn = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>()) / (float) C;
            for (auto c = static_cast<int64_t>(it.get_local_linear_id()); c < C; c += kWG) {
                const float d = x[t * C + c] - mn;
                va += d * d;
            }
            const float iv =
                1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), va, sycl::plus<float>()) / (float) C + kEps);
            if (it.get_local_linear_id() == 0) {
                mean[t] = mn;
                inv[t] = iv;
            }
        });
}

// SPEC max grad norm 1.0: the global norm over the averaged grads, clipped before AdamW.
// The grads stay on the device; the norm reads them back once per update (about 80 MB for
// rank 16, negligible beside a 35 s update) through the same copy the save path uses.
float grad_clip_scale(Model* m, float gscale, double* norm_out) {
    struct Ref {
        const float* g;
        int64_t n;
    };
    std::vector<Ref> refs;
    refs.reserve(600);
    for (int64_t b = 0; b < m->nblk; ++b) {
        MfTrainAdapter* az = m->blk[(size_t) b].adap;
        if (az == nullptr) continue;
        for (int q = 0; q < 2; ++q)
            for (int i = 0; i < 6; ++i) {
                const int64_t N = m->blk[(size_t) b].prod[q][i].N, K = m->blk[(size_t) b].prod[q][i].K;
                const int64_t rk = az->rank;
                refs.push_back({az->ad[q][i].dA, rk * K});
                refs.push_back({az->ad[q][i].dB, N * rk});
                if (az->ad[q][i].dm != nullptr) refs.push_back({az->ad[q][i].dm, N});
                if (az->ad[q][i].method == train::AdapterMethod::lokr) {
                    const int64_t ol = az->ad[q][i].ol, ok = az->ad[q][i].ok;
                    const int64_t im = az->ad[q][i].im, inn = az->ad[q][i].inn;
                    refs.push_back({az->ad[q][i].dW1, ol * im});
                    refs.push_back({az->ad[q][i].dW2a, ok * rk});
                    refs.push_back({az->ad[q][i].dW2b, rk * inn});
                }
            }
        for (int q = 0; q < 2; ++q)
            if (az->dmodA[q] != nullptr) {
                refs.push_back({az->dmodA[q], az->rank * m->C});
                refs.push_back({az->dmodB[q], 6 * m->C * az->rank});
            }
    }
    int64_t mx = 0;
    for (auto& r : refs) mx = std::max(mx, r.n);
    std::vector<float> host((size_t) mx);
    gpu::stream_sync(m->stream);
    double total = 0.0;
    for (auto& r : refs) {
        if (r.g == nullptr || r.n == 0) continue;
        if (!gpu::copy(host.data(), r.g, (size_t) r.n * 4))
            throw std::runtime_error("image engine (Mage-Flow training): clip readback");
        for (int64_t i = 0; i < r.n; ++i) {
            const double v = (double) host[(size_t) i] * gscale;
            total += v * v;
        }
    }
    const double norm = std::sqrt(total);
    if (norm_out != nullptr) *norm_out = norm;
    return (float) (norm > 1.0 ? 1.0 / norm : 1.0);
}

// one AdamW step over the adapter's tensors (FP32 masters and states): g averaged over accum_n, then zeroed
void adamw_step(float* w, float* g, float* m, float* v, int64_t n, float lr, float beta1, float beta2, float eps,
                float wd, float bc1, float bc2, float gscale, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float gg = g[id[0]] * gscale;
        const float nm = m[id[0]] = beta1 * m[id[0]] + (1.f - beta1) * gg;
        const float nv = v[id[0]] = beta2 * v[id[0]] + (1.f - beta2) * gg * gg;
        w[id[0]] -= lr * (nm / bc1 / (sycl::sqrt(nv / bc2) + eps) + wd * w[id[0]]);
        g[id[0]] = 0.f;
    });
}

}  // namespace

void mf_trainer_backward(MfTrainModel* base, const MfTrainSample& smp, const MfTrainCond& c, MfTrainState& s,
                         float sigma, prefill::Gemm& gemm) {
    (void)smp;
    (void)sigma;
    Model* m = static_cast<Model*>(base);
    void* st = m->stream;
    const int64_t C = m->C, R = s.R, Rt = s.Rt, Rj = R + Rt;
    Owned tmp;
    // the norm and output projection backward (the base is frozen: its dgrad only, no wgrad or bias sum)
    uint16_t* pout_wt = tmp.get<uint16_t>(C * m->in_ch, st);
    train::transpose_f16(m->pout.w, m->in_ch, C, pout_wt, st);
    uint16_t* dv16 = tmp.get<uint16_t>(R * m->in_ch, st);
    float* dvrs = tmp.get<float>(R, st);
    dit_rows::rows16(dit_rows::Grad{{s.dv_pred, m->in_ch}}, R, m->in_ch, dv16, dvrs, st);
    float* dln = tmp.get<float>(R * C, st);
    float* dsums = tmp.get<float>(R * C, st);
    train::dgrad(dv16, dvrs, pout_wt, R, m->in_ch, C, dln, dsums, false, gemm, st);
    float* mean = tmp.get<float>(R, st);
    float* inv = tmp.get<float>(R, st);
    // block 11's output (the image stream)
    const float* last = s.bounds16 != nullptr ? s.last : s.bounds + (size_t) m->nblk * Rj * C;
    float* nsc = tmp.get<float>(R * C, st);   // the output norm's scale a row (its shift has no gradient here)
    const float* nmod = c.nmod;
    dit_rows::q_of(st).parallel_for(sycl::range<1>((size_t) R * C), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / C, cc = i - t * C;
        nsc[id[0]] = nmod[cc];
    });
    ln_stats(last, R, C, mean, inv, st);
    float* dimg = tmp.get<float>(R * C, st);
    dit_rows::ln_mod_bwd(last, mean, inv, nsc, dln, R, C, dimg, nullptr, nullptr, st);
    // every block from its boundary input, the output's gradient ping-ponging down
    float* dcur_i = dimg;
    float* dnxt_i = tmp.get<float>(R * C, st);
    float* dcur_t = tmp.get<float>(Rt * C, st);
    float* dnxt_t = tmp.get<float>(Rt * C, st);
    // the modulations' gradients a block (their LoRA's), and its rank scratch
    const MfTrainAdapter* az0 = m->blk.empty() ? nullptr : m->blk[0].adap;
    const bool mod_on = az0 != nullptr && az0->modA[0] != nullptr;
    float* dmod[2] = {mod_on ? tmp.get<float>(6 * C, st) : nullptr, mod_on ? tmp.get<float>(6 * C, st) : nullptr};
    float* mh = mod_on ? tmp.get<float>(az0->rank, st) : nullptr;
    float* mg = mod_on ? tmp.get<float>(az0->rank, st) : nullptr;
    // the blocks' outputs (their forwards' again: not read) once for all of them, and a BF16 input read back
    float* bo = tmp.get<float>(R * C, st);
    float* bt = tmp.get<float>(Rt * C, st);
    float* xin = s.bounds16 != nullptr ? tmp.get<float>(Rj * C, st) : nullptr;
    for (int64_t b = m->nblk - 1; b >= 0; --b) {
        bind(m, b, m->ws.begin((int) b, (int) b - 1));   // the block below fetched meanwhile
        const MfTrainModel::Block& blk = m->blk[(size_t) b];
        for (int q = 0; q < 2; ++q)
            for (int i = 0; i < 6; ++i) {
                const MfTrainProduct& pr = blk.prod[q][i];
                if (pr.w8 != nullptr)   // the straight-through weight: the int8 values rotated back
                    train::w8_dequant_t(pr.w8, pr.sw, pr.N, pr.K, m->w8tmp, m->wtScratch + m->wtOff[q][i], st);
                else train::transpose_f16(pr.w, pr.N, pr.K, m->wtScratch + m->wtOff[q][i], st);
            }
        MfBlockRun run;
        run.C = C;
        run.heads = m->heads;
        run.hd = m->hd;
        run.F = m->F;
        run.rows_img = R;
        run.rows_txt = Rt;
        if (xin != nullptr) train::bf16_load(s.bounds16 + (size_t) b * Rj * C, Rj * C, xin, st);
        run.img = xin != nullptr ? xin : s.bounds + (size_t) b * Rj * C;
        run.txt = run.img + R * C;
        run.cos = c.cos;
        run.sin = c.sin;
        run.out_img = nullptr;   // recomputed inside, read from the tape the forward does not keep: unused
        run.out_txt = nullptr;
        // the block forwards and backwards in one call; its outputs land in bo and bt, its gradients below
        run.out_img = bo;
        run.out_txt = bt;
        run.d_out_img = dcur_i;
        run.d_out_txt = dcur_t;
        run.dimg = dnxt_i;
        run.dtxt = dnxt_t;
        for (int q = 0; q < 2; ++q) {
            MfBlockW* w = q == 0 ? &run.img_w : &run.txt_w;
            for (int i = 0; i < 6; ++i) {
                const MfTrainProduct& pr = blk.prod[q][i];
                w->N[i] = pr.N;
                w->K[i] = pr.K;
                w->w[i] = pr.w;
                w->w8[i] = pr.w8;
                w->sw[i] = pr.sw;
                w->wt[i] = m->wtScratch + m->wtOff[q][i];
                w->b[i] = pr.b;
                w->ad[i] = (blk.adap != nullptr) ? &blk.adap->ad[q][i] : nullptr;
            }
            w->nq = blk.nq[q];
            w->nk = blk.nk[q];
            w->mod = c.mod[(size_t) b * 2 + q];
            w->dnq = nullptr;
            w->dnk = nullptr;
            w->dmod = (blk.adap != nullptr && blk.adap->modA[q] != nullptr) ? dmod[q] : nullptr;
            if (w->dmod != nullptr && !gpu::memset_async(w->dmod, 0, (size_t) (6 * C) * 4, st))
                throw std::runtime_error("image engine (Mage-Flow training): zero");
            for (int i = 0; i < 6; ++i) {
                w->dw[i] = nullptr;
                w->db[i] = nullptr;
            }
        }
        // mf_block adds into its input's gradient, and the buffer it gets back from the swap still holds the
        // gradient two blocks up: without the zero every block's input gradient carried all of those above it
        if (!gpu::memset_async(dnxt_i, 0, (size_t) (R * C) * 4, st) ||
            !gpu::memset_async(dnxt_t, 0, (size_t) (Rt * C) * 4, st))
            throw std::runtime_error("image engine (Mage-Flow training): zero");
        mf_block(run, gemm, st, &m->arena);
        m->ws.end((int) b);
        for (int q = 0; q < 2; ++q) {
            const MfBlockW& bw = q == 0 ? run.img_w : run.txt_w;
            if (bw.dmod == nullptr) continue;
            const MfTrainAdapter* az = blk.adap;
            train::row_lora_bwd(az->modA[q], az->modB[q], c.tsil, bw.dmod, az->alpha / (float) az->rank, az->rank, 6 * C, C,
                         az->dmodA[q], az->dmodB[q], mh, mg, st);
        }
        std::swap(dcur_i, dnxt_i);
        std::swap(dcur_t, dnxt_t);
    }
    gpu::stream_sync(st);
    m->scratch_rows = std::max(m->scratch_rows, Rj);
    m->stepped = true;
}

void mf_trainer_adamw(MfTrainModel* base, float lr, float beta1, float beta2, float eps, float wd,
                      int64_t accum_n) {
    Model* m = static_cast<Model*>(base);
    void* s = m->stream;
    m->adam_step += 1;
    const double b1 = beta1, b2 = beta2, st = (double) m->adam_step;
    const float bc1 = (float) (1.0 - std::pow(b1, st)), bc2 = (float) (1.0 - std::pow(b2, st));
    const float gscale = accum_n > 1 ? 1.f / (float) accum_n : 1.f;
    // SPEC max grad norm 1.0: scale the averaged grads down to norm 1 when they run over.
    double gnorm = 0;
    const float clip = grad_clip_scale(m, gscale, &gnorm);
    const float gscale_eff = gscale * clip;
    std::fprintf(stderr, "adamw: update %llu global-norm %.4f clip %.4f (accum %lld)\n",
                 (unsigned long long) m->adam_step, gnorm, (double) clip, (long long) accum_n);
    for (int64_t b = 0; b < m->nblk; ++b) {
        MfTrainAdapter* az = m->blk[(size_t) b].adap;
        if (az == nullptr) continue;
        for (int q = 0; q < 2; ++q)
            for (int i = 0; i < 6; ++i) {
                const int64_t N = m->blk[(size_t) b].prod[q][i].N, K = m->blk[(size_t) b].prod[q][i].K;
                const int64_t rk = az->rank;
                adamw_step(az->A[q][i], az->ad[q][i].dA, az->mA[q][i], az->vA[q][i], rk * K, lr, beta1, beta2,
                           eps, wd, bc1, bc2, gscale_eff, s);
                adamw_step(az->B[q][i], az->ad[q][i].dB, az->mB[q][i], az->vB[q][i], N * rk, lr, beta1, beta2,
                           eps, wd, bc1, bc2, gscale_eff, s);
                if (az->ad[q][i].dm != nullptr)
                    adamw_step(az->M[q][i], az->ad[q][i].dm, az->mm[q][i], az->vm[q][i], N, lr, beta1, beta2,
                               eps, wd, bc1, bc2, gscale_eff, s);
                // the FP16 working copies follow the masters
                if (az->ad[q][i].method == train::AdapterMethod::lokr) {
                    const int64_t ol = az->ad[q][i].ol, ok = az->ad[q][i].ok;
                    const int64_t im = az->ad[q][i].im, inn = az->ad[q][i].inn;
                    adamw_step(az->W1[q][i], az->ad[q][i].dW1, az->mW1[q][i], az->vW1[q][i], ol * im, lr, beta1,
                               beta2, eps, wd, bc1, bc2, gscale_eff, s);
                    adamw_step(az->W2A[q][i], az->ad[q][i].dW2a, az->mW2A[q][i], az->vW2A[q][i], ok * rk, lr,
                               beta1, beta2, eps, wd, bc1, bc2, gscale_eff, s);
                    adamw_step(az->W2B[q][i], az->ad[q][i].dW2b, az->mW2B[q][i], az->vW2B[q][i], rk * inn, lr,
                               beta1, beta2, eps, wd, bc1, bc2, gscale_eff, s);
                    kernels::f32_to_f16_bulk(az->W1[q][i], az->ad[q][i].w1, ol * im, s);
                    train::transpose_f16(az->ad[q][i].w1, ol, im, az->ad[q][i].w1t, s);
                    kernels::f32_to_f16_bulk(az->W2A[q][i], az->ad[q][i].w2a, ok * rk, s);
                    kernels::f32_to_f16_bulk(az->W2B[q][i], az->ad[q][i].w2b, rk * inn, s);
                    continue;
                }
                kernels::f32_to_f16_bulk(az->A[q][i], az->ad[q][i].a, rk * K, s);
                train::transpose_f16(az->ad[q][i].a, rk, K, az->ad[q][i].at, s);
                kernels::f32_to_f16_bulk(az->B[q][i], az->ad[q][i].b, N * rk, s);
                train::transpose_f16(az->ad[q][i].b, N, rk, az->ad[q][i].bt, s);
            }
        for (int q = 0; q < 2; ++q)
            if (az->modA[q] != nullptr) {
                const int64_t rk = az->rank, C = m->C;
                adamw_step(az->modA[q], az->dmodA[q], az->mmodA[q], az->vmodA[q], rk * C, lr, beta1, beta2, eps, wd,
                           bc1, bc2, gscale_eff, s);
                adamw_step(az->modB[q], az->dmodB[q], az->mmodB[q], az->vmodB[q], 6 * C * rk, lr, beta1, beta2, eps,
                           wd, bc1, bc2, gscale_eff, s);
            }
    }
    gpu::stream_sync(s);
}

void mf_trainer_save(MfTrainModel* base, const std::string& path, const std::string& title,
                     const std::string& trigger, uint64_t seed, float lr, int64_t max_steps) {
    Model* m = static_cast<Model*>(base);
    void* s = m->stream;
    gpu::stream_sync(s);
    std::vector<train::Tensor> ts;
    auto blob = [&](const std::string& key, const float* dev, const std::vector<int64_t>& shape) {
        train::Tensor t;
        t.key = key;
        t.shape = shape;
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        t.v.resize((size_t) n);
        if (!gpu::copy(t.v.data(), dev, (size_t) n * 4))
            throw std::runtime_error("image engine (Mage-Flow training): save readback");
        ts.push_back(std::move(t));
    };
    std::string module = "networks.lora";
    int64_t rank = 0;
    float alpha = 0.f;
    for (int64_t b = 0; b < m->nblk; ++b) {
        MfTrainAdapter* az = m->blk[(size_t) b].adap;
        if (az == nullptr) continue;
        rank = az->rank;
        alpha = az->alpha;
        for (int st = 0; st < 2; ++st)
            for (int i = 0; i < 6; ++i) {
                const int64_t N = m->blk[(size_t) b].prod[st][i].N, K = m->blk[(size_t) b].prod[st][i].K;
                const int64_t rk = az->rank;
                const std::string t =
                    "diffusion_model.transformer_blocks." + std::to_string(b) + "." + kTarget[st][i];
                const train::Adapter& ad = az->ad[st][i];
                // the per-weight alpha (0-dimensional): ai-toolkit's files leave it out at alpha = rank, LyCORIS's
                // carry it, and the engine reads it where it is
                ts.push_back(train::Tensor{t + ".alpha", {}, {az->alpha}});
                if (ad.method == train::AdapterMethod::lokr) {
                    module = "networks.lokr";
                    blob(t + ".lokr_w1", az->W1[st][i], {ad.ol, ad.im});
                    blob(t + ".lokr_w2_a", az->W2A[st][i], {ad.ok, rk});
                    blob(t + ".lokr_w2_b", az->W2B[st][i], {rk, ad.inn});
                    continue;
                }
                blob(t + ".lora_A.weight", az->A[st][i], {rk, K});
                blob(t + ".lora_B.weight", az->B[st][i], {N, rk});
                if (ad.method == train::AdapterMethod::dora) blob(t + ".dora_scale", az->M[st][i], {N});
            }
        for (int st = 0; st < 2; ++st)
            if (az->modA[st] != nullptr) {
                const std::string t = "diffusion_model.transformer_blocks." + std::to_string(b) +
                                      (st == 0 ? ".img_mod.1" : ".txt_mod.1");
                ts.push_back(train::Tensor{t + ".alpha", {}, {az->alpha}});
                blob(t + ".lora_A.weight", az->modA[st], {az->rank, m->C});
                blob(t + ".lora_B.weight", az->modB[st], {6 * m->C, az->rank});
            }
    }
    // ai-toolkit's metadata (get_meta_for_safetensors: the job's name and version, the software, format pt) and
    // what the run was, as the Qwen-Image-2.1 trainer writes them
    const std::string name = path.substr(path.find_last_of('/') + 1);
    std::ostringstream lrs;
    lrs << lr;
    train::save_safetensors_bf16(path, std::move(ts),
                                 {{"name", title.empty() ? name : title}, {"version", "1.0"},
                                  {"software", "{\"name\": \"XeStrata\"}"}, {"format", "pt"},
                                  {"ss_network_module", module}, {"ss_network_dim", std::to_string(rank)},
                                  {"ss_network_alpha", std::to_string((int64_t) alpha)},
                                  {"ss_base_model", m->mrank > 0 ? "magetrail" : "mage-flow"}, {"ss_max_steps", std::to_string(max_steps)},
                                  {"ss_seed", std::to_string(seed)}, {"ss_learning_rate", lrs.str()},
                                  {"modelspec.trigger_phrase", trigger}});
}

}  // namespace strata::image
