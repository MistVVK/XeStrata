// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_trainer.cpp - see include/strata/image/mf_trainer.hpp.
#include "strata/image/mf_trainer.hpp"
#include "strata/image/lora_names.hpp"
#include "strata/image/train_loop.hpp"
#include "strata/image/train_trainer.hpp"

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

// a step's temporary device buffers, freed with it
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

// The blocks' twelve products through the trainer's LayerStream: their rows as the products read them (FP16 from the
// file's BF16, or an int8 base's rows and scales) are made once at open and kept in RAM; the pointers of blk's products
// are set to the block's region as it is used (bind).
struct Model : MfTrainModel, train::Trainer {
    Model() : Trainer("Mage-Flow training") {}
    void sync_loaded_views() override;
    std::vector<std::array<uint64_t, 12>> woff;
    bool int8 = false;

    // the persistent pointers after the trainer's pause/resume (the values come back at new addresses)
    void remap() override {
        auto mv = [&](auto*& p) {
            using T = std::remove_reference_t<decltype(p)>;
            p = static_cast<T>(moved(p));
        };
        for (MfTrainProduct* pr : {&img_in, &txt_in, &t1, &t2, &nout, &pout}) {
            mv(pr->w);
            mv(pr->b);
        }
        mv(mdown_w);
        mv(mdown_b);
        mv(txt_norm);
        for (Block& B : blk) {
            for (int q = 0; q < 2; ++q) {
                for (int i = 0; i < 6; ++i) mv(B.prod[q][i].b);   // w/w8/sw bound again at each use (bind)
                mv(B.mod[q].w);
                mv(B.mod[q].b);
                mv(B.modlow[q]);
                mv(B.nq[q]);
                mv(B.nk[q]);
            }
            MfTrainAdapter* az = B.adap;
            if (az == nullptr) continue;
            for (int q = 0; q < 2; ++q) {
                mv(az->modA[q]);
                mv(az->modB[q]);
                mv(az->dmodA[q]);
                mv(az->dmodB[q]);
                for (int i = 0; i < 6; ++i) {
                    mv(az->A[q][i]);
                    mv(az->B[q][i]);
                    mv(az->M[q][i]);
                    mv(az->W1[q][i]);
                    mv(az->W2A[q][i]);
                    mv(az->W2B[q][i]);
                    train::Adapter& a = az->ad[q][i];
                    mv(a.a);
                    mv(a.at);
                    mv(a.b);
                    mv(a.bt);
                    mv(a.dA);
                    mv(a.dB);
                    mv(a.m);
                    mv(a.norms);
                    mv(a.norms0);   // a.w bound again at each use (bind), as the product's own
                    mv(a.dm);
                    mv(a.w1);
                    mv(a.w1t);
                    mv(a.w2a);
                    mv(a.w2b);
                    mv(a.dW1);
                    mv(a.dW2a);
                    mv(a.dW2b);
                }
            }
        }
    }
};

float* dev_dup(Model* m, const float* host, int64_t n) {
    float* d = m->get<float>(n);
    if (!gpu::copy(d, host, (size_t) n * 4)) throw std::runtime_error("image engine (Mage-Flow training): copy");
    gpu::stream_sync(m->stream);
    return d;
}

uint16_t* dev_dup16(Model* m, const uint16_t* host, int64_t n) {
    uint16_t* d = m->dev16(host, n);
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
    pr.wt = nullptr;
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
constexpr auto& kTarget = lora::kMageBlock;

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
            auto* at = const_cast<uint8_t*>(base) + off[(size_t) q * 6 + (size_t) i];
            if (m->int8) {
                pr.w8 = reinterpret_cast<int8_t*>(at);
                pr.sw = reinterpret_cast<float*>(at + pr.N * pr.K);
                pr.w = nullptr;
            } else {
                pr.w = reinterpret_cast<uint16_t*>(at);
            }
            if (blk.adap != nullptr) blk.adap->ad[q][i].w = pr.w;   // a DoRA's composed norms read the same rows
        }
}

}  // namespace

train::Trainer& mf_trainer_base(MfTrainModel* base) { return *static_cast<Model*>(base); }

MfTrainModel* mf_trainer_open(const std::string& dit_path, int64_t rank, void* stream, std::string& err, bool int8) {
    std::unique_ptr<Model> m(new Model());
    m->stream = stream;
    try {
        if (!m->init(stream, 1 << 24, err)) return nullptr;
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
                    auto& h = h16[(size_t) q * 6 + (size_t) i];
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
                    const auto& h = h16[(size_t) q * 6 + (size_t) i];
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
                        az.a = m->get<uint16_t>(rank * K);
                        az.at = m->get<uint16_t>(rank * K);
                        az.b = m->get<uint16_t>(N * rank);
                        az.bt = m->get<uint16_t>(N * rank);
                        az.dA = m->get<float>(rank * K);
                        az.dB = m->get<float>(N * rank);
                        az.dm = m->get<float>(N);
                        az.norms = m->get<float>(N);
                        az.norms0 = m->get<float>(N);
                        az.w1 = m->get<uint16_t>(ol * im);
                        az.w1t = m->get<uint16_t>(ol * im);
                        az.w2a = m->get<uint16_t>(ok * rank);
                        az.w2b = m->get<uint16_t>(rank * inn);
                        az.dW1 = m->get<float>(ol * im);
                        az.dW2a = m->get<float>(ok * rank);
                        az.dW2b = m->get<float>(rank * inn);
                        MfTrainAdapter& tz = *blk.adap;
                        tz.A[st][i] = m->get<float>(rank * K);
                        tz.B[st][i] = m->get<float>(N * rank);
                        tz.M[st][i] = m->get<float>(N);
                        tz.W1[st][i] = m->get<float>(ol * im);
                        tz.W2A[st][i] = m->get<float>(ok * rank);
                        tz.W2B[st][i] = m->get<float>(rank * inn);
                    }
            }
            m->blk.push_back(blk);
        }
        m->nblk = (int64_t) m->blk.size();
        if (m->nblk == 0) throw std::runtime_error("image engine (Mage-Flow training): no transformer blocks");
        // the modulations' linears stay on the host: one row a step, read once
        (void) data;
        m->set_budget(0, 0, 0);
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

MfTrainCond mf_trainer_cond(MfTrainModel* base, float sigma, int64_t h, int64_t w, prefill::Gemm& gemm) {
    Model* m = static_cast<Model*>(base);
    void* s = m->stream;
    MfTrainCond c;
    Owned tmp;
    const int64_t C = m->C;
    // the sinusoidal row, as the DiT reads it
    std::vector<uint16_t> e;
    mf_time_sinusoid(sigma, e);
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
    c.tsil = m->get<float>(C);
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
            float* mod = m->get<float>(6 * C);
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
    c.nmod = m->get<float>(2 * C);
    gemm.f16(tsil16, m->nout.w, c.nmod, 1, 2 * C, C);
    dit_rows::rows_back(c.nmod, nullptr, m->nout.b, 1, 2 * C, s);
    std::vector<float> ch, sh;
    mf_rope_rows({{h, w}}, m->hd, m->axes, ch, sh);
    c.cos = dev_dup(m, ch.data(), (int64_t) ch.size());
    c.sin = dev_dup(m, sh.data(), (int64_t) sh.size());
    gpu::stream_sync(s);
    return c;
}

void mf_trainer_free_cond(MfTrainModel* base, MfTrainCond& c) {
    Model* m = static_cast<Model*>(base);
    auto drop = [&](float* q) { m->release(q); };
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

// Block b's run as it is bound (bind): its two streams' products, adapters, norms and modulations; no dgrad weights,
// no outputs or gradients yet
static MfBlockRun block_run(Model* m, int64_t b, const MfTrainCond& c, int64_t R, int64_t Rt) {
    const MfTrainModel::Block& blk = m->blk[(size_t) b];
    MfBlockRun run;
    run.lazy_dgrad = true;
    run.C = m->C;
    run.heads = m->heads;
    run.hd = m->hd;
    run.F = m->F;
    run.rows_img = R;
    run.rows_txt = Rt;
    run.cos = c.cos;
    run.sin = c.sin;
    for (int q = 0; q < 2; ++q) {
        MfBlockW* w = q == 0 ? &run.img_w : &run.txt_w;
        for (int i = 0; i < 6; ++i) {
            const MfTrainProduct& pr = blk.prod[q][i];
            w->N[i] = pr.N;
            w->K[i] = pr.K;
            w->w[i] = pr.w;
            w->w8[i] = pr.w8;
            w->sw[i] = pr.sw;
            w->b[i] = pr.b;
            w->ad[i] = blk.adap != nullptr && !blk.adap->off[q][i] ? &blk.adap->ad[q][i] : nullptr;
        }
        w->nq = blk.nq[q];
        w->nk = blk.nk[q];
        w->mod = c.mod.empty() ? nullptr : c.mod[(size_t) b * 2 + q];
    }
    return run;
}

static float forward_impl(Model* m, const MfTrainSample& s, const MfTrainCond& c, const float* eps, float sigma,
                          float* v_pred, MfTrainState* keep, prefill::Gemm& gemm) {
    void* st = m->stream;
    const int64_t C = m->C, R = s.tokens, Rt = s.rows, Rj = R + Rt;
    Owned tmp;
    auto keep_get = [&](int64_t n) -> float* {
        return keep != nullptr ? m->get<float>(n) : tmp.get<float>(n, st);
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
    // (Trainer::inputs_half), two FP32 boundaries passed between and each block's input rounded to BF16 as it is kept
    float* bounds = nullptr;
    uint16_t* bounds16 = nullptr;
    float* pass = nullptr;   // the forward's two boundaries passed between (BF16, or the forward alone)
    // the weights' budget beside the boundaries in BF16 at least and the arena (a guess of 24 FP32 rows of C a row
    // for the blocks' scratch and the attention's backward before a step has shown it); the boundaries in BF16 where
    // the memory is short of them in FP32 (12 FP32 rows of C a row for the arena's growth before a step)
    m->set_budget(Rj, (uint64_t) ((m->nblk + 1) * C) * 2, (uint64_t) (24 * C) * 4);
    // and the last blocks' forwards kept for the backward as the memory takes them (keep_tapes, decided before the
    // step's own buffers are taken: every block's input, the backward's gradients and its blocks' outputs)
    const bool short32 =
        keep != nullptr && m->inputs_half(Rj, (uint64_t) ((m->nblk + 1) * Rj * C) * 4, (uint64_t) (12 * C) * 4);
    // the host form first (the inputs as FP32, byte for byte): the pinned memory had, else BF16
    const bool hostm = short32 && m->host_inputs(m->nblk + 1, Rj * C * 4);
    const bool half = short32 && !hostm;
    if (!hostm) m->host.close();
    m->set_inputs_mode(hostm ? "host" : half ? "bf16" : "fp32");
    if (keep != nullptr) {
        const size_t rc = (size_t) (Rj * C);
        // a block's tape at its largest: each product adapted where any block adapts it (train::Targets)
        MfBlockRun most = block_run(m, 0, c, R, Rt);
        for (int q = 0; q < 2; ++q)
            for (int i = 0; i < 6; ++i)
                for (int64_t b = 0; b < m->nblk && (q == 0 ? most.img_w : most.txt_w).ad[i] == nullptr; ++b) {
                    const MfTrainAdapter* az = m->blk[(size_t) b].adap;
                    if (az != nullptr && !az->off[q][i]) (q == 0 ? most.img_w : most.txt_w).ad[i] = &az->ad[q][i];
                }
        keep->first_kept = m->nblk - m->keep_tapes(m->nblk, Rj, mf_block_tape_bytes(most, gemm),
                                                   hostm ? (size_t) (6 * rc * 4)
                                                         : half ? (size_t) (m->nblk + 1) * rc * 2 + 4 * rc * 4
                                                                : (size_t) (m->nblk + 1 + 4) * rc * 4);
    }
    if (half) {
        bounds16 = m->get<uint16_t>((m->nblk + 1) * Rj * C);
        keep->bounds16 = bounds16;
        keep->last = m->get<float>(R * C);
    } else if (keep != nullptr && hostm) {
        keep->last = m->get<float>(Rj * C);   // the head's input read back from the host (the backward's)
    } else if (keep != nullptr) {
        bounds = m->get<float>((m->nblk + 1) * Rj * C);
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
        if (hostm) m->host.keep(0, b0, Rj * C * 4);
    }
    if (keep != nullptr) {
        keep->R = R;
        keep->Rt = Rt;
        keep->x0 = x0;
        keep->xe = xe;
        keep->v_pred = m->get<float>(R * m->in_ch);
        keep->dv_pred = m->get<float>(R * m->in_ch);
        v_pred = keep->v_pred;
    }
    // the twelve blocks, joint attention between the streams (the text's rows first); the last ones' forwards kept
    // for the backward (first_kept)
    float* bi = bounds != nullptr ? bounds : pass;
    float* bt = bi + R * C;
    for (int64_t b = 0; b < m->nblk; ++b) {
        // the next block's products fetched meanwhile; past the last, the backward's second when it follows
        const int nb = (int) m->nblk;
        bind(m, b, keep != nullptr ? m->ws.begin((int) b, (int) (b + 1 < nb ? b + 1 : nb - 2)) : m->ws.begin((int) b));
        MfBlockRun run = block_run(m, b, c, R, Rt);
        run.img = bi;
        run.txt = bt;
        float* no = bounds != nullptr ? bounds + ((size_t) b + 1) * Rj * C : pass + ((b + 1) % 2) * Rj * C;
        float* nt = no + R * C;
        run.out_img = no;
        run.out_txt = nt;
        if (keep != nullptr && b >= keep->first_kept) run.kept = m->tape(b);
        if (hostm && b > 0) m->host.kept_wait(b - 1);   // this block writes the buffer slot b-1 was copied from
        mf_block(run, gemm, st, &m->arena);
        m->ws.end((int) b);
        if (bounds16 != nullptr) train::bf16_keep(no, Rj * C, bounds16 + ((size_t) b + 1) * Rj * C, st);
        if (hostm) m->host.keep(b + 1, no, Rj * C * 4);
        bi = no;
        bt = nt;
    }
    if (bounds16 != nullptr) {   // the head's input, and the backward's (rounded as kept)
        gpu::stream_sync(st);
        if (!gpu::copy(keep->last, bi, (size_t) R * C * 4))
            throw std::runtime_error("image engine (Mage-Flow training): copy");
    }
    if (hostm) m->host.fetch(m->nblk, keep->last, Rj * C * 4, -1);   // the head's input for the backward
    // the output norm (LayerNorm without affine, times 1 + scale, plus shift: the DiT's) and the projection to the
    // latent's channels; its statistics kept for the backward
    uint16_t* ln16 = tmp.get<uint16_t>(R * C, st);
    float* lnrs = tmp.get<float>(R, st);
    float* mean = keep != nullptr ? m->get<float>(R) : tmp.get<float>(R, st);
    float* inv = keep != nullptr ? m->get<float>(R) : tmp.get<float>(R, st);
    if (keep != nullptr) {
        keep->mean = mean;
        keep->inv = inv;
    }
    dit_rows::rows16(dit_rows::LnMod{bi, c.nmod + C, c.nmod, C}, R, C, ln16, lnrs, st, mean, inv);
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
    auto drop = [&](float* q) { m->release(q); };
    drop(s.x0);
    drop(s.xe);
    drop(s.v_pred);
    drop(s.dv_pred);
    drop(s.bounds);
    drop(s.last);
    drop(s.mean);
    drop(s.inv);
    m->release(s.bounds16);
    s = MfTrainState();
}

namespace {

// the base weight's row norms (the DoRA composed norm's base term), over the FP16 values the products read
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

void mf_trainer_init_adapters(MfTrainModel* base, int method, float alpha, uint64_t seed, bool mod_targets,
                              const train::Targets& targets) {
    Model* m = static_cast<Model*>(base);
    void* s = m->stream;
    std::mt19937_64 rng(seed);
    bool any = false;
    for (int64_t b = 0; b < m->nblk; ++b) {
        MfTrainModel::Block& blk = m->blk[(size_t) b];
        if (blk.adap == nullptr) continue;
        bind(m, b, m->ws.begin((int) b));   // the DoRA norms below read the base's rows
        MfTrainAdapter& tz = *blk.adap;
        const std::string p = "transformer_blocks." + std::to_string(b) + ".";
        for (int st = 0; st < 2; ++st)
            for (int i = 0; i < 6; ++i) {
                tz.off[st][i] = !targets.take((int) b, p + kTarget[st][i]);
                any = any || !tz.off[st][i];
            }
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
                if (tz.off[st][i]) continue;
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
                    rownorm_f16(blk.prod[st][i].w, N, K, az.norms0, s);   // |W_r| once; each forward composes
                // ||W_r + (alpha/rank) (BA)_r|| from it (PEFT's DoRA, dora_composed_norms)
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
                if (!gpu::copy(az.norms, az.norms0, (size_t) N * 4) ||
                    !gpu::copy(tz.M[st][i], az.norms0, (size_t) N * 4))
                    throw std::runtime_error("image engine (Mage-Flow training): copy");
                kernels::f32_to_f16_bulk(tz.A[st][i], az.a, rk * K, s);
                train::transpose_f16(az.a, rk, K, az.at, s);
                // B stays zero (Owned zeroes); its views stay zero with it
            }
        // the trained tensors, in the order the global norm reads them: each product's factors (and a DoRA's
        // magnitude, a LoKr's three), then the modulations' LoRA below
        for (int st = 0; st < 2; ++st)
            for (int i = 0; i < 6; ++i) {
                const int64_t N = blk.prod[st][i].N, K = blk.prod[st][i].K, rk = tz.rank;
                train::Adapter& az = tz.ad[st][i];
                if (tz.off[st][i]) continue;
                if (tz.method == train::AdapterMethod::lokr) {
                    m->add_param(tz.W1[st][i], az.dW1, az.ol * az.im);
                    m->add_param(tz.W2A[st][i], az.dW2a, az.ok * rk);
                    m->add_param(tz.W2B[st][i], az.dW2b, rk * az.inn);
                    continue;
                }
                m->add_param(tz.A[st][i], az.dA, rk * K);
                m->add_param(tz.B[st][i], az.dB, N * rk);
                if (tz.method == train::AdapterMethod::dora) m->add_param(tz.M[st][i], az.dm, N);
            }
        if (mod_targets && m->mrank == 0)   // the modulations' plain LoRA: A kaiming-uniform, B zero, as above
                                            // (MageTrail's low-rank form takes none: the engine adds none to it)
            for (int st = 0; st < 2; ++st) {
                if (!targets.take((int) b, p + lora::kMageMod[st])) continue;
                any = true;
                const int64_t C = m->C, rk = tz.rank;
                tz.modA[st] = m->get<float>(rk * C);
                tz.modB[st] = m->get<float>(6 * C * rk);
                tz.dmodA[st] = m->get<float>(rk * C);
                tz.dmodB[st] = m->get<float>(6 * C * rk);
                const float ba = 1.f / std::sqrt((float) C);
                std::vector<float> ha((size_t) (rk * C));
                for (float& v : ha) v = uni(rng) * ba;
                gpu::stream_sync(s);
                if (!gpu::copy(tz.modA[st], ha.data(), ha.size() * 4))
                    throw std::runtime_error("image engine (Mage-Flow training): copy");
                m->add_param(tz.modA[st], tz.dmodA[st], rk * C);
                m->add_param(tz.modB[st], tz.dmodB[st], 6 * C * rk);
            }
        m->ws.end((int) b);
    }
    gpu::stream_sync(s);
    m->targets = targets.text();
    if (!any && !m->blk.empty() && m->blk[0].adap != nullptr)
        throw std::runtime_error("image engine (Mage-Flow training): the weights chosen (" + targets.text() +
                                 ") are none of the blocks' products");
}

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
    train::dgrad(dv16, dvrs, pout_wt, R, m->in_ch, C, dln, nullptr, false, gemm, st);
    // block 11's output (the image stream), through the output norm's scale (its shift has no gradient here)
    const bool hostm = s.bounds == nullptr && s.bounds16 == nullptr;   // the inputs on the host (FP32)
    if (hostm) m->host.wait(m->nblk);   // the head's input read back (the forward's fetch)
    const float* last = hostm || s.bounds16 != nullptr ? s.last : s.bounds + (size_t) m->nblk * Rj * C;
    float* dimg = tmp.get<float>(R * C, st);
    dit_rows::ln_mod_bwd(last, s.mean, s.inv, c.nmod, dln, R, C, dimg, nullptr, nullptr, st);
    // every block from its boundary input, the output's gradient ping-ponging down
    float* dcur_i = dimg;
    float* dnxt_i = tmp.get<float>(R * C, st);
    float* dcur_t = tmp.get<float>(Rt * C, st);
    float* dnxt_t = tmp.get<float>(Rt * C, st);
    // the modulations' gradients a block (their LoRA's), and its rank scratch
    const MfTrainAdapter* az0 = m->blk.empty() ? nullptr : m->blk[0].adap;
    bool mod_on = false;   // any block's modulation trained (train::Targets may leave some out)
    for (const MfTrainModel::Block& blk : m->blk)
        mod_on = mod_on || (blk.adap != nullptr && (blk.adap->modA[0] != nullptr || blk.adap->modA[1] != nullptr));
    float* dmod[2] = {mod_on ? tmp.get<float>(6 * C, st) : nullptr, mod_on ? tmp.get<float>(6 * C, st) : nullptr};
    float* mh = mod_on ? tmp.get<float>(az0->rank, st) : nullptr;
    float* mg = mod_on ? tmp.get<float>(az0->rank, st) : nullptr;
    // the blocks' outputs (their forwards' again: not read) once for all of them, and a BF16 input read back;
    // host: two read-back buffers, the inputs fetched a block ahead (the values as the forward kept them, FP32)
    float* bo = dln;   // the output norm's gradient is dead; replayed blocks write their unused outputs here
    float* bt = tmp.get<float>(Rt * C, st);
    float* xin = s.bounds16 != nullptr ? tmp.get<float>(Rj * C, st) : nullptr;
    float* lb = hostm ? tmp.get<float>(2 * Rj * C, st) : nullptr;
    if (hostm) {
        m->host.used(0);
        m->host.used(1);
        for (int64_t b = m->nblk - 1; b >= 0 && b >= m->nblk - 2; --b)
            m->host.fetch(b, lb + (b % 2) * Rj * C, Rj * C * 4, (int) (b % 2));
    }
    for (int64_t b = m->nblk - 1; b >= 0; --b) {
        bind(m, b, m->ws.begin((int) b, (int) b - 1));   // the block below fetched meanwhile
        const MfTrainModel::Block& blk = m->blk[(size_t) b];
        MfBlockRun run = block_run(m, b, c, R, Rt);
        if (hostm) {
            m->host.wait(b);
            run.img = lb + (b % 2) * Rj * C;
        } else if (xin != nullptr) {
            train::bf16_load(s.bounds16 + (size_t) b * Rj * C, Rj * C, xin, st);
            run.img = xin;
        } else {
            run.img = s.bounds + (size_t) b * Rj * C;
        }
        run.txt = run.img + R * C;
        // the block forwards (unless its forward was kept) and backwards in one call; its outputs land in bo and bt,
        // its gradients below
        run.out_img = bo;
        run.out_txt = bt;
        run.d_out_img = dcur_i;
        run.d_out_txt = dcur_t;
        run.dimg = dnxt_i;
        run.dtxt = dnxt_t;
        if (b >= s.first_kept) {
            run.kept = m->tape(b);
            run.kept_replay = true;
        }
        for (int q = 0; q < 2; ++q) {
            MfBlockW* w = q == 0 ? &run.img_w : &run.txt_w;
            w->dmod = (blk.adap != nullptr && blk.adap->modA[q] != nullptr) ? dmod[q] : nullptr;
            if (w->dmod != nullptr && !gpu::memset_async(w->dmod, 0, (size_t) (6 * C) * 4, st))
                throw std::runtime_error("image engine (Mage-Flow training): zero");
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
        if (hostm) {
            m->host.used((int) (b % 2));
            if (b > 0) m->host.fetch(b - 1, lb + ((b - 1) % 2) * Rj * C, Rj * C * 4, (int) ((b - 1) % 2));
        }
        std::swap(dcur_i, dnxt_i);
        std::swap(dcur_t, dnxt_t);
    }
    gpu::stream_sync(st);
    m->stepped_rows(Rj, (uint64_t) Rj * C * 4);
}

void mf_trainer_adamw(MfTrainModel* base, float lr, float beta1, float beta2, float eps, float wd,
                      int64_t accum_n) {
    Model* m = static_cast<Model*>(base);
    train::AdamW o;
    o.lr = lr;
    o.beta1 = beta1;
    o.beta2 = beta2;
    o.eps = eps;
    o.wd = wd;
    // the gradients summed over accum_n micro-steps, averaged: the norm and the update read them so
    const float gscale = accum_n > 1 ? 1.f / (float) accum_n : 1.f;
    const double gnorm = m->update(o, gscale);
    std::fprintf(stderr, "adamw: update %lld global-norm %.4f clip %.4f (accum %lld)\n", (long long) m->steps, gnorm,
                 std::min(1.0, 1.0 / (gnorm + 1e-6)), (long long) accum_n);
    m->sync_loaded_views();
}

void Model::sync_loaded_views() {
    Model* m = this;
    void* s = stream;
    // the FP16 working copies follow the masters
    for (int64_t b = 0; b < m->nblk; ++b) {
        MfTrainAdapter* az = m->blk[(size_t) b].adap;
        if (az == nullptr) continue;
        for (int q = 0; q < 2; ++q)
            for (int i = 0; i < 6; ++i) {
                const int64_t N = m->blk[(size_t) b].prod[q][i].N, K = m->blk[(size_t) b].prod[q][i].K;
                const train::Adapter& ad = az->ad[q][i];
                if (az->off[q][i]) continue;
                if (ad.method == train::AdapterMethod::lokr) {
                    const int64_t rk = az->rank;
                    kernels::f32_to_f16_bulk(az->W1[q][i], ad.w1, ad.ol * ad.im, s);
                    train::transpose_f16(ad.w1, ad.ol, ad.im, ad.w1t, s);
                    kernels::f32_to_f16_bulk(az->W2A[q][i], ad.w2a, ad.ok * rk, s);
                    kernels::f32_to_f16_bulk(az->W2B[q][i], ad.w2b, rk * ad.inn, s);
                    continue;
                }
                m->sync_views(ad, az->A[q][i], az->B[q][i], N, K);
            }
    }
    gpu::stream_sync(s);
}

void mf_trainer_save(MfTrainModel* base, const std::string& path, const std::string& title,
                     const std::string& trigger, uint64_t seed, float lr, int64_t steps, int64_t max_steps,
                     const std::string& kind) {
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
                if (az->off[st][i]) continue;   // a product the run does not train
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
    std::ostringstream lrs;
    lrs << lr;
    train::LoraMeta meta;
    meta.name = title.empty() ? path.substr(path.find_last_of('/') + 1) : title;
    meta.base = m->mrank > 0 ? "magetrail" : "mage-flow";
    meta.module = module;
    meta.lr = lrs.str();
    meta.trigger = trigger;
    meta.rank = rank;
    meta.alpha = alpha;
    meta.steps = steps;
    meta.max_steps = max_steps;
    meta.seed = seed;
    meta.kind = kind;
    meta.targets = m->targets;
    train::save_safetensors_bf16(path, std::move(ts), train::lora_meta(meta));
}

}  // namespace strata::image
