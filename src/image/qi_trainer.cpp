// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_trainer.cpp - see include/strata/image/qi_trainer.hpp.  The DiT's edges (the time embedding and the
// shared modulation, the text projection, img_in, the final norm and proj_out) as ai-toolkit's
// QwenImage21Transformer2DModel.forward takes them, around the shared training block (qi_block).
#include "strata/image/qi_trainer.hpp"
#include "strata/image/dit_file.hpp"
#include "strata/image/gelu.hpp"
#include "strata/image/lora_names.hpp"
#include "strata/image/qi_dit.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/image/qi_train.hpp"
#include "strata/image/train_ops.hpp"
#include "strata/image/train_trainer.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/platform/env.hpp"
#include "strata/prefill/gemm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <tuple>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

constexpr uint32_t kF16 = 1, kBF16 = 30;   // GGML's type numbers
constexpr int64_t kHead = 128, kInCh = 64, kTimeDim = 256;
static_assert(std::size(lora::kQiBlock) == kQiProducts);
const char* const* const kProducts = lora::kQiBlock;   // the DiT's target names, in kQiProducts' order

}  // namespace

struct QiTrainer::Impl : train::Trainer {
    Impl() : Trainer("Qwen-Image-2.1 training") {}
    std::unique_ptr<QiFile> g;
    int64_t C = 0, F = 0, H = 0, ctx = 0, rank = 0;
    struct Block {
        train::Product p[kQiProducts];   // N, K (the weights' pointers are set as the block is used: weights_of)
        uint64_t off[kQiProducts] = {};  // each product's place in the block's region (ws)
        float *nq = nullptr, *nk = nullptr;
        train::Lora* l[kQiProducts] = {};
    };
    // The blocks' weights through the trainer's LayerStream: as many as the memory leaves stay in VRAM, the rest are
    // copied in a block at a time from RAM (an int8 base's rows and scales, made once at open and kept in RAM; a 16-bit
    // base's from the file), the next one while this one computes.  A GPU with the memory keeps them all.
    bool int8 = false;
    std::vector<Block> blk;
    uint16_t *iin = nullptr, *tin = nullptr, *tout = nullptr;   // bf16
    float* tnorm = nullptr;                                      // the text norm's weight + 1
    uint16_t *t1 = nullptr, *t2 = nullptr, *mod = nullptr, *nout = nullptr, *pout = nullptr, *pout_t = nullptr;
    std::vector<std::pair<float*, int64_t>> texts;
    // the step's buffers, grown with the rows
    int64_t cap = 0;
    std::vector<float*> ckpt;   // [blocks + 1][R][C]: each block's input, the last one the DiT's output
    // where the memory is short (ensure): the inputs in BF16 instead, [blocks + 1][cap][C], ckpt the two FP32 rows'
    // buffers the forward passes between (the backward reads an input back into the one the head left alone)
    uint16_t* ckpt16 = nullptr;
    float *tmp_out = nullptr, *dX = nullptr, *dX2 = nullptr, *x0 = nullptr, *nz = nullptr, *pred = nullptr;
    float *dpred = nullptr, *dln = nullptr, *scratch = nullptr, *ln_rs = nullptr, *mean = nullptr, *inv = nullptr;
    float *drs = nullptr, *cos = nullptr, *sin = nullptr;
    uint16_t *xt16 = nullptr, *ln16 = nullptr, *d16 = nullptr;
    std::tuple<int64_t, int64_t, int64_t> rope_key{-1, -1, -1};
    // the time's buffers: the sinusoids, the embedder's rows, the modulation [2][4C] and as the block reads it
    uint16_t *e16 = nullptr, *s16 = nullptr;
    float *temb = nullptr, *mods = nullptr, *mb = nullptr, *scale_out = nullptr, *zero = nullptr;
    uint16_t* plain0[3] = {};           // an int8 base: block 0's q, k and v rotated back (QiBlockRun::w_plain)
    int64_t last_hw = 0, planned_rows = 0;

    // the persistent pointers after the trainer's pause/resume (the values come back at new addresses)
    void remap() override {
        auto mv = [&](auto*& p) {
            using T = std::remove_reference_t<decltype(p)>;
            p = static_cast<T>(moved(p));
        };
        mv(iin);
        mv(tin);
        mv(tout);
        mv(tnorm);
        mv(t1);
        mv(t2);
        mv(mod);
        mv(nout);
        mv(pout);
        mv(pout_t);
        mv(e16);
        mv(s16);
        mv(temb);
        mv(mods);
        mv(mb);
        mv(scale_out);
        mv(zero);
        for (auto& p : plain0) mv(p);
        for (Block& B : blk) {
            mv(B.nq);
            mv(B.nk);   // B.p's weights are bound again at each use (weights_of)
        }
        for (auto& t : texts) mv(t.first);
        for (float*& p : ckpt) mv(p);
        mv(ckpt16);
        mv(tmp_out);
        mv(dX);
        mv(dX2);
        mv(x0);
        mv(nz);
        mv(pred);
        mv(dpred);
        mv(dln);
        mv(scratch);
        mv(ln_rs);
        mv(mean);
        mv(inv);
        mv(drs);
        mv(cos);
        mv(sin);
        mv(xt16);
        mv(ln16);
        mv(d16);
    }

    // a 16-bit weight [N][K] as the file holds it (`type`: F16 or bf16), on the GPU
    uint16_t* w16(const std::string& name, uint32_t type, int64_t& N, int64_t& K) {
        const TensorInfo& t = g->get(name);
        if (t.type != type)
            throw std::runtime_error("image engine (Qwen-Image-2.1 training): " + name + " is not " +
                                     (type == kF16 ? "F16" : "BF16") + " (a 16-bit DiT file is the training's base)");
        K = (int64_t) t.shape[0];
        N = t.shape.size() > 1 ? (int64_t) t.shape[1] : 1;
        return dev16(g->data(t), N * K);
    }
    // the products' weights of block b in its region `base` (ws.begin's)
    void weights_of(int64_t b, const uint8_t* base, train::Product* p) const {
        const Block& B = blk[(size_t) b];
        for (int i = 0; i < kQiProducts; ++i) {
            p[i] = B.p[i];
            const uint8_t* at = base + B.off[i];
            if (int8) {
                p[i].w8 = reinterpret_cast<const int8_t*>(at);
                p[i].sw = reinterpret_cast<const float*>(at + B.p[i].N * B.p[i].K);
            } else {
                p[i].w = reinterpret_cast<const uint16_t*>(at);
            }
        }
    }
    // The step's buffers for R rows, taken before the weights' budget is set (so it sees them): the blocks' inputs
    // in BF16 where the free memory falls short of them in FP32 beside every block's weights resident (inputs_half,
    // the blocks' scratch 13 FP32 rows of F a row before a step has shown it) - the inputs halved before any weight
    // is streamed, which costs every step its copies.
    void ensure(int64_t R) {
        if (R <= cap && !replan_due()) return;
        host.close();
        for (float* p : ckpt) release(p);
        release(ckpt16);
        ckpt16 = nullptr;
        for (void* p : {(void*) tmp_out, (void*) dX, (void*) dX2, (void*) x0, (void*) nz, (void*) pred, (void*) dpred,
                        (void*) dln, (void*) scratch, (void*) ln_rs, (void*) mean, (void*) inv, (void*) drs,
                        (void*) cos, (void*) sin, (void*) xt16, (void*) ln16, (void*) d16})
            release(p);
        const uint64_t inputs = (uint64_t) ((int64_t) (blk.size() + 1) * R * C) * 4 + (ws.total() - ws.vram());
        const bool half = inputs_half(R, inputs, (uint64_t) (13 * F) * 4);
        // the host form first (the inputs as FP32, byte for byte): the pinned memory had, else BF16
        const bool hostm = half && host_inputs((int64_t) blk.size() + 1, R * C * 4);
        if (hostm) {
            ckpt.assign(2, nullptr);
            std::fprintf(stderr, "image engine (Qwen-Image-2.1 training): the blocks' inputs on the host (%lld rows)\n",
                         (long long) R);
        } else if (half) {
            ckpt.assign(2, nullptr);
            ckpt16 = get<uint16_t>((int64_t) (blk.size() + 1) * R * C);
            std::fprintf(stderr, "image engine (Qwen-Image-2.1 training): the blocks' inputs in BF16 (%lld rows)\n",
                         (long long) R);
        } else {
            ckpt.assign(blk.size() + 1, nullptr);
        }
        set_inputs_mode(hostm ? "host" : half ? "bf16" : "fp32");
        for (float*& p : ckpt) p = get<float>(R * C);
        tmp_out = get<float>(R * C);
        dX = get<float>(R * C);
        dX2 = get<float>(R * C);
        x0 = get<float>(R * kInCh);   // the image's rows at most: hw <= R
        nz = get<float>(R * kInCh);
        pred = get<float>(R * kInCh);
        dpred = get<float>(R * kInCh);
        dln = get<float>(R * C);
        scratch = get<float>(R * C);
        ln_rs = get<float>(R);
        mean = get<float>(R);
        inv = get<float>(R);
        drs = get<float>(R);
        cos = get<float>(R * kHead / 2);
        sin = get<float>(R * kHead / 2);
        xt16 = get<uint16_t>(R * kInCh);
        ln16 = get<uint16_t>(R * C);
        d16 = get<uint16_t>(R * kInCh);
        cap = R;
        rope_key = {-1, -1, -1};
    }
    // ai-toolkit's QwenImage21Rope: a text row's three axes its position, the image's frame axis the text's length
    // and its height and width centred on zero; the DiT's tables of those positions (qi_rope_rows)
    void rope(int64_t nt, int64_t h, int64_t w) {
        const auto key = std::make_tuple(nt, h, w);
        if (key == rope_key) return;
        const int64_t R = nt + h * w;
        std::vector<float> pos((size_t) R * 3), c, sn;
        for (int64_t r = 0; r < R; ++r) {
            float* p = &pos[(size_t) r * 3];
            if (r < nt) {
                p[0] = p[1] = p[2] = (float) r;
            } else {
                const int64_t q = r - nt, y = q / w, x = q % w, top = h - h / 2, left = w - w / 2;
                p[0] = (float) nt;
                p[1] = (float) (y - top);
                p[2] = (float) (x - left);
            }
        }
        qi_rope_rows(pos, R, c, sn);
        gpu::stream_sync(s);
        gpu::copy(cos, c.data(), c.size() * 4);
        gpu::copy(sin, sn.data(), sn.size() * 4);
        rope_key = key;
    }
    // the time's embedding and the shared modulation for the image's t (row 0) and t = 0 (row 1), as qi_block reads
    // it ([4][2][C], the gates through tanh), and the final norm's scale (row 0)
    void modulation(float timestep) {
        // ai-toolkit: t = timestep in bf16 / 1000 in bf16; the embedding's argument 1000 t in FP32
        const float tb = kernels::f32_from_bf16(kernels::bf16_from_f32(timestep));
        const float t = kernels::f32_from_bf16(kernels::bf16_from_f32(tb / 1000.0f)) * 1000.0f;
        float ef[2 * kTimeDim];   // the DiT's sinusoids of t and of 0
        qi_time_sinusoid(t, ef);
        qi_time_sinusoid(0.0f, ef + kTimeDim);
        std::vector<uint16_t> e(2 * kTimeDim);
        for (size_t i = 0; i < e.size(); ++i) e[i] = kernels::f16_from_f32(ef[i]);
        gpu::stream_sync(s);
        gpu::copy(e16, e.data(), e.size() * 2);
        gemm.f16(e16, t1, temb, 2, C, kTimeDim);
        qi_silu(temb, 2 * C, s16, s);
        gemm.f16(s16, t2, temb, 2, C, C);
        qi_silu(temb, 2 * C, s16, s);   // SiLU(temb): the modulation's and the final norm's input
        gemm.f16(s16, mod, mods, 2, 4 * C, C);
        gemm.f16(s16, nout, scale_out, 2, C, C);
        const int64_t Cc = C;
        float *src = mods, *dst = mb;
        dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (8 * C)), [=](sycl::id<1> id) {
            const auto i = static_cast<int64_t>(id[0]);
            const int64_t k = i / (2 * Cc), row = (i / Cc) % 2, c = i % Cc;
            const float v = src[row * 4 * Cc + k * Cc + c];
            dst[i] = k == 1 || k == 3 ? sycl::tanh(v) : v;
        });
    }
};

QiTrainer::QiTrainer() : m_(std::make_unique<Impl>()) {}

QiTrainer::~QiTrainer() = default;

train::Trainer& QiTrainer::base() { return *m_; }

int64_t QiTrainer::blocks() const { return (int64_t) m_->blk.size(); }
int64_t QiTrainer::width() const { return m_->C; }

bool QiTrainer::open(const std::string& dit, int64_t rank, uint64_t seed, void* stream, std::string& err,
                     int64_t max_blocks, bool int8, const train::Targets& targets) {
    Impl& m = *m_;
    try {
        m.rank = rank;
        m.g = std::make_unique<QiFile>(dit);
        if (!m.init(stream, 1 << 20, err)) return false;   // before the adapters: their views depend on it
        int64_t N = 0, K = 0;
        m.iin = m.w16("img_in.weight", kBF16, N, K);
        m.C = N;
        if (K != kInCh) throw std::runtime_error("image engine (Qwen-Image-2.1 training): img_in is not 64 wide");
        m.tin = m.w16("txt_in.in_layer.weight", kBF16, N, K);
        m.ctx = K;
        m.tout = m.w16("txt_in.out_layer.weight", kBF16, N, K);
        std::vector<float> tn = m.g->f32("txt_in.text_norm.weight");
        for (float& v : tn) v += 1.0f;   // zero-centred: the scale is weight + 1
        m.tnorm = m.dev_vec(tn);
        m.t1 = m.w16("time_text_embed.timestep_embedder.linear_1.weight", kF16, N, K);
        m.t2 = m.w16("time_text_embed.timestep_embedder.linear_2.weight", kF16, N, K);
        m.mod = m.w16("modulation.1.weight", kF16, N, K);
        m.nout = m.w16("norm_out.linear.weight", kF16, N, K);
        m.pout = m.w16("proj_out.weight", kF16, N, K);
        m.pout_t = m.get<uint16_t>(N * K);
        train::transpose_f16(m.pout, N, K, m.pout_t, stream);
        std::mt19937_64 rng(seed);
        m.int8 = int8;
        // an int8 base's rows made on the GPU a product at a time, through these (freed after the blocks)
        uint16_t* q16 = nullptr;
        int8_t* q8 = nullptr;
        float* qs = nullptr;
        int64_t qcap = 0, scap = 0;
        for (int b = 0;; ++b) {
            const std::string p = "transformer_blocks." + std::to_string(b) + ".";
            if (m.g->find(p + "attn.to_q.weight") == nullptr) break;
            if (max_blocks > 0 && b >= max_blocks) break;
            Impl::Block blk;
            std::vector<LayerStream::Span> spans;
            const TensorInfo* ts[kQiProducts];
            uint64_t hbytes = 0;
            for (int i = 0; i < kQiProducts; ++i) {
                train::Product& pr = blk.p[i];
                const TensorInfo& t = m.g->get(p + kProducts[i] + ".weight");
                if (t.type != kF16)
                    throw std::runtime_error("image engine (Qwen-Image-2.1 training): " + p + kProducts[i] +
                                             " is not F16 (a 16-bit DiT file is the training's base)");
                ts[i] = &t;
                pr.K = (int64_t) t.shape[0];
                pr.N = t.shape.size() > 1 ? (int64_t) t.shape[1] : 1;
                hbytes += (uint64_t) (pr.N * pr.K + 4 * pr.N);
            }
            if (int8) {   // the int8 base's rows and scales from the 16-bit weight, into RAM
                m.whost.emplace_back((size_t) hbytes);
                uint8_t* h = m.whost.back().data();
                for (int i = 0; i < kQiProducts; ++i) {
                    const train::Product& pr = blk.p[i];
                    const int64_t n = pr.N * pr.K;
                    if (n > qcap) {
                        for (void* x : {(void*) q16, (void*) q8}) m.release(x);
                        q16 = m.get<uint16_t>(n);
                        q8 = m.get<int8_t>(n);
                        qcap = n;
                    }
                    if (pr.N > scap) {
                        m.release(qs);
                        qs = m.get<float>(pr.N);
                        scap = pr.N;
                    }
                    gpu::copy(q16, m.g->data(*ts[i]), (size_t) n * 2);
                    float* sw = qs;
                    train::w8_quantize(q16, pr.N, pr.K, q8, sw, stream);
                    if (b == 0 && i < 3) {   // the first block's q, k and v for its unquantized input
                        m.plain0[i] = m.get<uint16_t>(n);
                        train::w8_dequant_rows(q8, sw, pr.N, pr.K, m.plain0[i], stream);
                    }
                    gpu::stream_sync(stream);
                    gpu::copy(h, q8, (size_t) n);
                    gpu::copy(h + n, sw, (size_t) pr.N * 4);
                    spans.push_back({h, (uint64_t) (n + 4 * pr.N), 0});
                    h += n + 4 * pr.N;
                }
            } else {
                for (int i = 0; i < kQiProducts; ++i)
                    spans.push_back({m.g->data(*ts[i]), (uint64_t) (blk.p[i].N * blk.p[i].K) * 2, 0});
            }
            const int layer = m.ws.add(spans);
            for (int i = 0; i < kQiProducts; ++i) blk.off[i] = m.ws.offset(layer, i);
            for (int i = 0; i < kQiProducts; ++i)
                if (targets.take(b, p + kProducts[i])) blk.l[i] = &m.add_lora(blk.p[i].N, blk.p[i].K, rank, rng);
            blk.nq = m.dev_vec(m.g->f32(p + "attn.norm_q.weight"));
            blk.nk = m.dev_vec(m.g->f32(p + "attn.norm_k.weight"));
            m.blk.push_back(blk);
        }
        for (void* x : {(void*) q16, (void*) q8, (void*) qs}) m.release(x);
        if (m.blk.empty()) throw std::runtime_error("image engine (Qwen-Image-2.1 training): the DiT has no blocks");
        if (rank > 0 && m.loras().empty())
            throw std::runtime_error("image engine (Qwen-Image-2.1 training): the weights chosen (" + targets.text() +
                                     ") are none of the blocks' products");
        m.F = m.blk[0].p[kQiGateUp].N / 2;
        m.H = m.C / kHead;
        m.dgrad_scratch(m.blk[0].p, kQiProducts, int8);
        m.e16 = m.get<uint16_t>(2 * kTimeDim);
        m.s16 = m.get<uint16_t>(2 * m.C);
        m.temb = m.get<float>(2 * m.C);
        m.mods = m.get<float>(8 * m.C);
        m.mb = m.get<float>(8 * m.C);
        m.scale_out = m.get<float>(2 * m.C);
        m.zero = m.get<float>(m.C);
        gpu::stream_sync(stream);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

int QiTrainer::add_text(const std::vector<float>& states, int64_t n) {
    Impl& m = *m_;
    const int64_t D = m.ctx, C = m.C;
    if ((int64_t) states.size() != n * D) throw std::invalid_argument("image engine (Qwen-Image-2.1 training): states");
    float* x = m.get<float>(n * D);
    gpu::copy(x, states.data(), states.size() * 4);
    auto* xb = m.get<uint16_t>(n * std::max(D, C));
    float* y = m.get<float>(n * C);
    // the zero-centred RMS norm (eps 1e-6), in FP32 as the reference computes it, then bf16 for in_layer
    const float* w = m.tnorm;
    dit_rows::q_of(m.s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n * 256), sycl::range<1>(256)),
                                     [=](sycl::nd_item<1> it) {
                                         const int64_t r = (int64_t) it.get_group_linear_id();
                                         float ss = 0.f;
                                         for (int64_t c = (int64_t) it.get_local_linear_id(); c < D; c += 256)
                                             ss += x[r * D + c] * x[r * D + c];
                                         ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                                         const float k = sycl::rsqrt(ss / (float) D + 1e-6f);
                                         for (int64_t c = (int64_t) it.get_local_linear_id(); c < D; c += 256)
                                             xb[r * D + c] = kernels::bf16_from_f32(x[r * D + c] * k * w[c]);
                                     });
    m.gemm.bf16(xb, m.tin, y, n, C, D);
    dit_rows::q_of(m.s).parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        xb[id[0]] = kernels::bf16_from_f32(gelu_tanh(y[id[0]]));
    });
    float* rows = m.get<float>(n * C);
    m.gemm.bf16(xb, m.tout, rows, n, C, C);
    gpu::stream_sync(m.s);
    m.release(x);
    m.release(xb);
    m.release(y);
    m.texts.emplace_back(rows, n);
    return (int) m.texts.size() - 1;
}

void QiTrainer::plan_rows(int64_t rows) {
    m_->planned_rows = rows;
}

float QiTrainer::step(int text, const float* x0, const float* noise, int64_t h, int64_t w, float timestep) {
    Impl& m = *m_;
    void* s = m.s;
    const int64_t C = m.C, hw = h * w, nt = m.texts.at((size_t) text).second, R = nt + hw;
    const int64_t planned = std::max(R, m.planned_rows);
    m.ensure(planned);
    m.set_budget(planned, 0, (uint64_t) (13 * m.F) * 4);
    m.rope(nt, h, w);
    m.modulation(timestep);
    // the noised latent's tokens, x_t = (1 - t) x0 + t noise with t = timestep / 1000 in FP32 (ai-toolkit's
    // add_noise), in bf16 as the DiT reads them
    const float t01 = timestep / 1000.0f;
    std::vector<uint16_t> xt((size_t) (hw * kInCh));
    for (size_t i = 0; i < xt.size(); ++i) xt[i] = kernels::bf16_from_f32((1.0f - t01) * x0[i] + t01 * noise[i]);
    gpu::stream_sync(s);
    gpu::copy(m.xt16, xt.data(), xt.size() * 2);
    gpu::copy(m.x0, x0, (size_t) (hw * kInCh) * 4);
    gpu::copy(m.nz, noise, (size_t) (hw * kInCh) * 4);
    float* X = m.ckpt[0];
    gpu::copy_async(X, m.texts[(size_t) text].first, (size_t) (nt * C) * 4, s);
    m.gemm.bf16(m.xt16, m.iin, X + nt * C, hw, C, kInCh);
    auto run_of = [&](int64_t b) {
        QiBlockRun r;
        r.C = C;
        r.heads = m.H;
        r.hd = kHead;
        r.F = m.F;
        r.rows = R;
        r.n_text = nt;
        r.cos = m.cos;
        r.sin = m.sin;
        r.mod = m.mb;
        // the MLP 1,024 rows a pass: its scratch at 4,035 rows 4.5 -> 2.5 GB, the B70's step the same (512: 4% slower,
        // 2026-11-09)
        r.mlp_rows = 1024;
        const Impl::Block& B = m.blk[(size_t) b];
        for (int i = 0; i < kQiProducts; ++i) {
            r.p[i] = B.p[i];
            r.ad[i] = B.l[i] != nullptr ? &B.l[i]->ad : nullptr;
        }
        r.nq = B.nq;
        r.nk = B.nk;
        if (b == 0)
            for (int i = 0; i < 3; ++i) r.w_plain[i] = m.plain0[i];
        r.x = (m.ckpt16 != nullptr || m.host.ok()) ? m.ckpt[(size_t) (b % 2)] : m.ckpt[(size_t) b];
        return r;
    };
    const bool half = m.ckpt16 != nullptr;
    const bool hostm = m.host.ok();
    const int64_t cap = m.cap;
    const int64_t nb = (int64_t) m.blk.size();
    // a block's tape at its largest: each product adapted where any block adapts it (a run trains some blocks' or
    // products' only, train::Targets)
    QiBlockRun most = run_of(0);
    for (int i = 0; i < kQiProducts; ++i)
        for (const Impl::Block& B : m.blk)
            if (most.ad[i] == nullptr && B.l[i] != nullptr) most.ad[i] = &B.l[i]->ad;
    const int64_t first_kept = nb - m.keep_tapes(nb, R, qi_block_tape_bytes(most), 0);
    if (hostm) m.host.keep(0, m.ckpt[0], R * C * 4);   // the head's input (the text rows and img_in's output)
    for (int64_t b = 0; b < nb; ++b) {   // the forward, each block's input kept, the last ones' tapes too
        QiBlockRun r = run_of(b);
        // the next block's weights fetched meanwhile; past the last, the backward's second (its first is this one)
        m.weights_of(b, m.ws.begin((int) b, (int) (b + 1 < nb ? b + 1 : nb - 2)), r.p);
        if (half && b == 0) train::bf16_keep(m.ckpt[0], R * C, m.ckpt16, s);
        if (hostm && b > 0) m.host.kept_wait(b - 1);   // this block writes the buffer slot b-1 was copied from
        r.out = (half || hostm) ? m.ckpt[(size_t) ((b + 1) % 2)] : m.ckpt[(size_t) b + 1];
        if (b >= first_kept) r.tape = m.tape(b);
        qi_block(r, m.gemm, s, &m.arena);
        m.ws.end((int) b);
        if (half) train::bf16_keep(r.out, R * C, m.ckpt16 + (b + 1) * cap * C, s);
        if (hostm) m.host.keep(b + 1, r.out, R * C * 4);
    }
    // the final norm (the image's rows, the t row's scale) and proj_out
    const float* xf = ((half || hostm) ? m.ckpt[(size_t) (nb % 2)] : m.ckpt[(size_t) nb]) + nt * C;
    dit_rows::rows16(dit_rows::LnMod{xf, m.zero, m.scale_out, C}, hw, C, m.ln16, m.ln_rs, s, m.mean, m.inv);
    m.gemm.f16(m.ln16, m.pout, m.pred, hw, kInCh, C);
    dit_rows::rows_back(m.pred, m.ln_rs, nullptr, hw, kInCh, s);
    const float loss = train::flow_loss(m.pred, m.x0, m.nz, hw * kInCh, m.dpred, nullptr, nullptr, s);
    m.last_hw = hw;
    // ai-toolkit's SDTrainer: a loss that is not finite is taken as zero, so the step adds no gradient (and the
    // update still runs on the zero gradients)
    if (!std::isfinite(loss)) {
        std::fprintf(stderr, "image engine (Qwen-Image-2.1 training): the loss is not finite, the step adds nothing\n");
        return loss;
    }
    // the backward: proj_out's dgrad, the final norm's, then the blocks from the last, each run again from its input
    dit_rows::rows16(dit_rows::Grad{{m.dpred, kInCh}}, hw, kInCh, m.d16, m.drs, s);
    train::dgrad(m.d16, m.drs, m.pout_t, hw, kInCh, C, m.dln, m.scratch, false, m.gemm, s);
    gpu::memset_async(m.dX, 0, (size_t) (R * C) * 4, s);
    dit_rows::ln_mod_bwd(xf, m.mean, m.inv, m.scale_out, m.dln, hw, C, m.dX + nt * C, nullptr, nullptr, s);
    if (hostm) {   // the pass buffers are the backward's read-back buffers now; the first two blocks prefetched
        m.host.used(0);
        m.host.used(1);
        for (int64_t b = nb - 1; b >= 0 && b >= nb - 2; --b)
            m.host.fetch(b, m.ckpt[(size_t) (b % 2)], R * C * 4, (int) (b % 2));
    }
    for (int64_t b = nb - 1; b >= 0; --b) {
        QiBlockRun r = run_of(b);
        if (hostm) {
            m.host.wait(b);   // r.x: this parity's pass buffer (run_of), read back a block ahead
        } else if (half) {   // its input read back into the buffer the head did not read
            float* xin = m.ckpt[(size_t) ((nb + 1) % 2)];
            train::bf16_load(m.ckpt16 + b * cap * C, R * C, xin, s);
            r.x = xin;
        }
        m.weights_of(b, m.ws.begin((int) b, (int) b - 1), r.p);   // the block below fetched meanwhile
        m.dgrad_weights(r.p, kQiProducts);   // (an int8 base's straight through)
        r.out = m.tmp_out;
        if (b >= first_kept) {   // its forward's tape: the backward alone
            r.tape = m.tape(b);
            r.tape_replay = true;
        }
        r.dout = m.dX;
        r.dx = b > 0 ? m.dX2 : nullptr;
        qi_block(r, m.gemm, s, &m.arena);
        m.ws.end((int) b);
        if (hostm) {
            m.host.used((int) (b % 2));
            if (b > 0) m.host.fetch(b - 1, m.ckpt[(size_t) ((b - 1) % 2)], R * C * 4, (int) ((b - 1) % 2));
        }
        std::swap(m.dX, m.dX2);
    }
    gpu::stream_sync(s);
    qi_block_profile_report();
    m.stepped_rows(R, (uint64_t) R * C * 4);
    if (strata::env_get("TRAIN_PROFILE") != nullptr) {   // the memory the step left
        size_t fb = 0, tb = 0;
        gpu::mem_info(&fb, &tb);
        std::fprintf(stderr, "end: free %zu MiB, main arena %zu MiB, R %lld\n", fb >> 20, m.arena.held() >> 20,
                     (long long) R);
    }
    return loss;
}

double QiTrainer::update(const train::AdamW& o) { return m_->update(o); }

std::vector<train::Tensor> QiTrainer::grads() const {
    const Impl& m = *m_;
    std::vector<train::Tensor> out;
    for (size_t b = 0; b < m.blk.size(); ++b)
        for (int i = 0; i < kQiProducts; ++i) {
            if (m.blk[b].l[i] == nullptr) continue;   // a product the run does not train
            const train::Lora& l = *m.blk[b].l[i];
            const std::string k = "diffusion_model.transformer_blocks." + std::to_string(b) + "." + kProducts[i];
            out.push_back({k + ".lora_A.weight", {m.rank, l.K}, m.read(l.ad.dA, m.rank * l.K)});
            out.push_back({k + ".lora_B.weight", {l.N, m.rank}, m.read(l.ad.dB, l.N * m.rank)});
        }
    return out;
}

void QiTrainer::save(const std::string& path, const std::vector<std::pair<std::string, std::string>>& meta) const {
    const Impl& m = *m_;
    std::vector<train::Tensor> ts;
    for (size_t b = 0; b < m.blk.size(); ++b)
        for (int i = 0; i < kQiProducts; ++i) {
            if (m.blk[b].l[i] == nullptr) continue;   // a product the run does not train
            const train::Lora& l = *m.blk[b].l[i];
            const std::string k = "diffusion_model.transformer_blocks." + std::to_string(b) + "." + kProducts[i];
            ts.push_back({k + ".lora_A.weight", {m.rank, l.K}, m.read(l.A, m.rank * l.K)});
            ts.push_back({k + ".lora_B.weight", {l.N, m.rank}, m.read(l.B, l.N * m.rank)});
        }
    train::save_safetensors_bf16(path, std::move(ts), meta);
}

void QiTrainer::set_factors(const std::vector<train::Tensor>& ts) {
    Impl& m = *m_;
    std::map<std::string, const train::Tensor*> by;
    for (const train::Tensor& t : ts) by[t.key] = &t;
    gpu::stream_sync(m.s);
    for (size_t b = 0; b < m.blk.size(); ++b)
        for (int i = 0; i < kQiProducts; ++i) {
            if (m.blk[b].l[i] == nullptr) continue;   // a product the run does not train
            const train::Lora& l = *m.blk[b].l[i];
            const std::string k = "diffusion_model.transformer_blocks." + std::to_string(b) + "." + kProducts[i];
            auto put = [&](const std::string& key, float* d, int64_t n) {
                auto it = by.find(key);
                if (it == by.end()) return;
                if ((int64_t) it->second->v.size() != n)
                    throw std::invalid_argument("image engine (Qwen-Image-2.1 training): " + key + " has the wrong size");
                gpu::copy(d, it->second->v.data(), (size_t) n * 4);
            };
            put(k + ".lora_A.weight", l.A, m.rank * l.K);
            put(k + ".lora_B.weight", l.B, l.N * m.rank);
            m.sync_views(l.ad, l.A, l.B, l.N, l.K);
        }
    gpu::stream_sync(m.s);
}

std::vector<float> QiTrainer::last_pred() const { return m_->read(m_->pred, m_->last_hw * kInCh); }

train::Schedule qi_train_schedule(int64_t h, int64_t w) {
    return train::schedule_dynamic(h * w, 256.0, 8192.0, 0.5, 0.9, 0.02);
}

}  // namespace strata::image
