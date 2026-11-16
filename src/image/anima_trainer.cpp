// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/anima_trainer.cpp - see include/strata/image/anima_trainer.hpp.  The parts outside the blocks are the
// inference's (AnimaDit::forward, time_embedding, modulation and the final layer), frozen: the patch embedder, the
// time's two layers and its norm, the final layer's modulation and projection.
#include "strata/image/anima_trainer.hpp"
#include "strata/image/anima_dit.hpp"
#include "strata/image/dit_file.hpp"
#include "strata/image/lora_names.hpp"
#include "strata/image/anima_train.hpp"
#include "strata/platform/mapped_file.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/image/train_trainer.hpp"

#include "strata/core/gpu.hpp"
#include "strata/platform/env.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iterator>
#include <random>
#include <stdexcept>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

static_assert(std::size(lora::kAnimaBlock) == kAnProducts);
const char* const* const kProduct = lora::kAnimaBlock;   // the DiT's target names, in kAnProducts' order
const char* const* const kMod = lora::kAnimaMod;

// out[k] = sum_n W[n][k] d[n] (W [N][K] FP16) + sum_r A[r][k] g[r] (A [rank][K] FP32): a one-row product's input
// gradient through its weight and its LoRA's down factor
void row_dgrad(const uint16_t* W, const float* d, int64_t N, int64_t K, const float* A, const float* g, int64_t rank,
               float* out, void* s) {
    constexpr int64_t WG = 256;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) K * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t k = static_cast<int64_t>(it.get_group_linear_id());
            float acc = 0.f;
            for (auto n = static_cast<int64_t>(it.get_local_linear_id()); n < N; n += WG)
                acc += kernels::f32_from_f16(W[n * K + k]) * d[n];
            acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
            if (it.get_local_linear_id() == 0) {
                for (int64_t r = 0; r < rank; ++r) acc += A[r * K + k] * g[r];
                out[k] = acc;
            }
        });
}

void to_f32(const uint16_t* h, int64_t n, float* f, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n),
                                   [=](sycl::id<1> id) { f[id[0]] = kernels::f32_from_f16(h[id[0]]); });
}

// the final projection's patch order (pixel (p1, p2) of channel c at (p1 * 2 + p2) * 16 + c) <-> the tokens'
// (c * 4 + p1 * 2 + p2), `back` the way back
void patch_order(const float* in, int64_t T, int64_t L, float* out, bool back, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (T * 4 * L)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), t = i / (4 * L), j = i - t * 4 * L, c = j / 4, p = j % 4;
        const int64_t a = t * 4 * L + p * L + c, b = t * 4 * L + j;
        if (back) out[a] = in[b];
        else out[b] = in[a];
    });
}

}  // namespace

struct AnimaTrainer::Impl : train::Trainer {
    Impl() : Trainer("Anima training") {}
    int64_t C = 0, F = 0, H = 0, hd = 0, lat = 0, Cc = 0, M = 0, rank = 0;
    struct Dense {
        uint16_t* w = nullptr;
        int64_t N = 0, K = 0;
    };
    Dense xemb, t1, t2, fm1, fm2, flin;
    std::vector<float> tnorm;
    struct Block {
        train::Product p[kAnProducts];
        Dense w1[3], w2[3];
        float *sqn = nullptr, *skn = nullptr, *cqn = nullptr, *ckn = nullptr;
        train::Lora* l[kAnProducts] = {};               // null: no LoRA (rank 0)
        train::Lora *m1[3] = {}, *m2[3] = {};           // the modulations' one-row LoRAs
    };
    std::vector<Block> blk;
    // The blocks' ten products through the trainer's LayerStream (their FP16 rows made once at open and kept in RAM),
    // blk's pointers set as a block is used (bind)
    std::vector<std::array<uint64_t, kAnProducts>> woff;
    void bind(int64_t b, const uint8_t* base) {
        for (int i = 0; i < kAnProducts; ++i)
            blk[(size_t) b].p[i].w = reinterpret_cast<const uint16_t*>(base + woff[(size_t) b][(size_t) i]);
    }

    // the persistent pointers after the trainer's pause/resume (the values come back at new addresses)
    void remap() override {
        auto mv = [&](auto*& p) {
            using T = std::remove_reference_t<decltype(p)>;
            p = static_cast<T>(moved(p));
        };
        for (Dense* d : {&xemb, &t1, &t2, &fm1, &fm2, &flin}) mv(d->w);
        for (Block& B : blk) {
            for (Dense* d : {&B.w1[0], &B.w1[1], &B.w1[2], &B.w2[0], &B.w2[1], &B.w2[2]}) mv(d->w);
            mv(B.sqn);
            mv(B.skn);
            mv(B.cqn);
            mv(B.ckn);   // B.p's weights are bound again at each use (bind); the LoRAs are the trainer's
        }
    }
};

AnimaTrainer::AnimaTrainer() : m_(std::make_unique<Impl>()) {}

train::Trainer& AnimaTrainer::base() { return *m_; }
AnimaTrainer::~AnimaTrainer() = default;
int64_t AnimaTrainer::text_cols() const { return m_->Cc; }

bool AnimaTrainer::open(const std::string& dit, int64_t rank, uint64_t seed, void* stream, std::string& err,
                        const train::Targets& targets) {
    Impl& m = *m_;
    try {
        m.rank = rank;
        if (!m.init(stream, 1 << 20, err)) return false;
        AnimaFile f(dit);
        // a matrix's rows as FP16 (BF16, F16 or F32 in the file), its first `cols` columns (0: all), in RAM
        auto rows16 = [&](const std::string& name, int64_t& N, int64_t& Kc, std::vector<uint16_t>& h) {
            int64_t K = 0;
            f.rows16(name, 0, h, N, K);
            Kc = K;
        };
        // the same rows on the GPU
        auto dense = [&](const std::string& name, int64_t cols = 0) {
            Impl::Dense d;
            int64_t K = 0;
            std::vector<uint16_t> h;
            f.rows16(name, cols, h, d.N, K);
            d.K = cols > 0 ? cols : K;
            d.w = m.dev16(h.data(), d.N * d.K);
            return d;
        };
        m.lat = f.get("final_layer.linear.weight").shape[0] / 4;
        m.xemb = dense("x_embedder.proj.1", 4 * m.lat);   // the padding mask's columns left out (all zero)
        m.C = m.xemb.N;
        m.t1 = dense("t_embedder.1.linear_1");
        m.t2 = dense("t_embedder.1.linear_2");
        m.flin = dense("final_layer.linear");
        m.fm1 = dense("final_layer.adaln_modulation.1");
        m.fm2 = dense("final_layer.adaln_modulation.2");
        m.tnorm = f.f32("t_embedding_norm.weight");
        m.hd = f.get("blocks.0.self_attn.q_norm.weight").shape[0];
        m.H = m.C / m.hd;
        m.M = m.fm1.N;
        if (m.t2.N != 3 * m.C || m.fm2.N != 2 * m.C) throw std::runtime_error("image engine (Anima training): shape");
        std::mt19937_64 rng(seed);
        for (int b = 0;; ++b) {
            const std::string p = "blocks." + std::to_string(b) + ".";
            if (f.find(p + "self_attn.q_proj.weight") == nullptr) break;
            Impl::Block B;
            {   // the products' FP16 rows into RAM, the stream's source
                std::vector<std::vector<uint16_t>> h(kAnProducts);
                uint64_t bytes = 0;
                for (int i = 0; i < kAnProducts; ++i) {
                    rows16(p + kProduct[i], B.p[i].N, B.p[i].K, h[(size_t) i]);
                    bytes += h[(size_t) i].size() * 2;
                }
                m.whost.emplace_back((size_t) bytes);
                uint8_t* hp = m.whost.back().data();
                std::vector<LayerStream::Span> spans;
                for (const auto& v : h) {
                    std::memcpy(hp, v.data(), v.size() * 2);
                    spans.push_back({hp, (uint64_t) (v.size() * 2), 0});
                    hp += v.size() * 2;
                }
                const int layer = m.ws.add(spans);
                std::array<uint64_t, kAnProducts> off{};
                for (int i = 0; i < kAnProducts; ++i) off[(size_t) i] = m.ws.offset(layer, i);
                m.woff.push_back(off);
            }
            for (int k = 0; k < 3; ++k) {
                B.w1[k] = dense(p + kMod[k] + ".1");
                B.w2[k] = dense(p + kMod[k] + ".2");
            }
            B.sqn = m.dev_vec(f.f32(p + "self_attn.q_norm.weight"));
            B.skn = m.dev_vec(f.f32(p + "self_attn.k_norm.weight"));
            B.cqn = m.dev_vec(f.f32(p + "cross_attn.q_norm.weight"));
            B.ckn = m.dev_vec(f.f32(p + "cross_attn.k_norm.weight"));
            if (rank > 0) {
                for (int i = 0; i < kAnProducts; ++i)
                    if (targets.take(b, p + kProduct[i])) B.l[i] = &m.add_lora(B.p[i].N, B.p[i].K, rank, rng);
                for (int k = 0; k < 3; ++k) {
                    if (targets.take(b, p + kMod[k] + ".1"))
                        B.m1[k] = &m.add_lora(B.w1[k].N, B.w1[k].K, rank, rng, false);
                    if (targets.take(b, p + kMod[k] + ".2"))
                        B.m2[k] = &m.add_lora(B.w2[k].N, B.w2[k].K, rank, rng, false);
                }
            }
            m.blk.push_back(B);
        }
        if (m.blk.empty()) throw std::runtime_error("image engine (Anima training): the DiT has no blocks");
        if (rank > 0 && m.loras().empty())
            throw std::runtime_error("image engine (Anima training): the weights chosen (" + targets.text() +
                                     ") are none of the blocks' products");
        m.F = m.blk[0].p[kAnL1].N;
        m.Cc = m.blk[0].p[kAnCk].K;
        m.dgrad_scratch(m.blk[0].p, kAnProducts, false);
        gpu::stream_sync(stream);
        m.set_budget(0, 0, 0);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

float AnimaTrainer::step(const MfTrainSample& smp, const std::vector<float>& x0, const std::vector<float>& eps,
                         float sigma, float weight, bool backward) {
    Impl& m = *m_;
    void* s = m.s;
    const int64_t C = m.C, HW = smp.tokens, TV = 4 * m.lat, L = smp.rows, Cc = m.Cc, M = m.M, rk = m.rank;
    const int64_t nb = (int64_t) m.blk.size();
    if ((int64_t) x0.size() != HW * TV || (int64_t) eps.size() != HW * TV || smp.w * smp.h != HW)
        throw std::invalid_argument("image engine (Anima training): the sample's shapes");
    if ((int64_t) smp.te.size() != L * Cc) throw std::invalid_argument("image engine (Anima training): context rows");
    train::Pool pool(s);
    // the noised tokens through the patch embedder
    std::vector<float> xs((size_t) (HW * TV));
    for (size_t i = 0; i < xs.size(); ++i) xs[i] = (1.f - sigma) * x0[i] + sigma * eps[i];
    float *dx0 = pool.f(HW * TV), *deps = pool.f(HW * TV), *dxs = pool.f(HW * TV);
    float* ctx = pool.f(L * Cc);
    gpu::stream_sync(s);
    if (!gpu::copy(dx0, x0.data(), x0.size() * 4) || !gpu::copy(deps, eps.data(), eps.size() * 4) ||
        !gpu::copy(dxs, xs.data(), xs.size() * 4) || !gpu::copy(ctx, smp.te.data(), smp.te.size() * 4))
        throw std::runtime_error("image engine (Anima training): copy");
    m.set_budget(HW, (uint64_t) ((nb + 1) * C) * 2, (uint64_t) (13 * m.F) * 4);
    // every block's input in FP32 (two passed between, the forward alone), or where the memory is short in BF16
    // (inputs_half): bounds the two FP32 passed between, the inputs in bounds16, the input rounded as it is kept
    // and the last blocks' tapes as the memory takes them (keep_tapes), both decided before the step's own buffers
    // are taken (every block's input, the backward's two gradients and its blocks' output; in BF16, the inputs at half
    // and two FP32 rows' buffers)
    bool half = false, hostm = false;
    int64_t first_kept = nb;
    if (backward) {
        AnimaBlockRun z;
        z.C = C;
        z.heads = m.H;
        z.hd = m.hd;
        z.F = m.F;
        z.rows = HW;
        z.L = L;
        z.Cc = Cc;
        for (int i = 0; i < kAnProducts; ++i) {
            z.p[i] = m.blk.back().p[i];
            z.ad[i] = nullptr;   // a block's tape at its largest: the product adapted where any block adapts it
            for (const Impl::Block& B : m.blk)
                if (B.l[i] != nullptr) {
                    z.ad[i] = &B.l[i]->ad;
                    break;
                }
        }
        const size_t rc = (size_t) (HW * C);
        half = m.inputs_half(HW, (uint64_t) (nb + 1 + 3) * rc * 4, (uint64_t) (13 * m.F) * 4);
        // the host form first (the inputs as FP32, byte for byte): the pinned memory had, else BF16
        hostm = half && m.host_inputs(nb + 1, (int64_t) rc * 4);
        first_kept = nb - m.keep_tapes(nb, HW, anima_block_tape_bytes(z, m.gemm),
                                       hostm ? (size_t) (5 * rc * 4)
                                             : half ? (size_t) (nb + 1) * rc * 2 + 4 * rc * 4
                                                    : (size_t) (nb + 1 + 3) * rc * 4);
    }
    m.set_inputs_mode(hostm ? "host" : half ? "bf16" : "fp32");
    if (!hostm) m.host.close();
    const bool all32 = backward && !half && !hostm;
    float* bounds = pool.f((all32 ? nb + 1 : 2) * HW * C);
    uint16_t* bounds16 = half && !hostm ? pool.u((nb + 1) * HW * C) : nullptr;
    uint16_t* x16 = pool.u(HW * std::max(C, TV));
    float* rs = pool.f(HW);
    kernels::f32_to_f16_bulk(dxs, x16, HW * TV, s);
    m.gemm.f16(x16, m.xemb.w, bounds, HW, C, TV);
    // the context's FP16 rows and their scales (its transposed view for the adapters' gradients)
    uint16_t *ctx16 = pool.u(L * Cc), *ctx16t = pool.u(L * Cc);
    float* ctx_rs = pool.f(L);
    dit_rows::rows16(dit_rows::Plain{ctx, Cc}, L, Cc, ctx16, ctx_rs, s);
    train::transpose_f16(ctx16, L, Cc, ctx16t, s);
    // the time (AnimaDit::time_embedding): the sinusoid of sigma rounded to bf16; its RMS norm's SiLU (FP16) every
    // modulation reads; the timestep embedder's two layers of the sinusoid, the AdaLN-LoRA term each adds
    uint16_t* e16 = pool.u(2 * C);
    float* x1 = pool.f(C);   // the modulations' input, FP32 of e16's second half
    float* alora = pool.f(3 * C);
    {
        std::vector<uint16_t> h;
        anima_time_rows(sigma, m.tnorm, h);
        gpu::stream_sync(s);   // e16's zero (the pool's, on this stream) before the copy (on the compute queue)
        if (!gpu::copy(e16, h.data(), h.size() * 2)) throw std::runtime_error("copy");
        float* mh = pool.f(C);
        uint16_t* m16 = pool.u(C);
        m.gemm.f16(e16, m.t1.w, mh, 1, C, C);
        qi_silu(mh, C, m16, s);
        m.gemm.f16(m16, m.t2.w, alora, 1, 3 * C, C);
        to_f32(e16 + C, C, x1, s);
    }
    // each block's three modulations (AnimaDit::modulation): W2 (W1 e) + the AdaLN-LoRA term, each layer's LoRA added
    float* mods = pool.f(nb * 9 * C);
    float* x2s = pool.f(nb * 3 * M);   // each modulation's hidden values (its second layer's input), FP32 of FP16
    float* hr = pool.f(std::max<int64_t>(rk, 1));
    {
        float* mh = pool.f(M);
        uint16_t* m16 = pool.u(M);
        for (int64_t b = 0; b < nb; ++b) {
            const Impl::Block& B = m.blk[(size_t) b];
            for (int k = 0; k < 3; ++k) {
                float* mod = mods + (b * 3 + k) * 3 * C;
                float* x2 = x2s + (b * 3 + k) * M;
                m.gemm.f16(e16 + C, B.w1[k].w, mh, 1, M, C);
                if (B.m1[k] != nullptr) train::row_lora_fwd(B.m1[k]->A, B.m1[k]->B, x1, 1.f, rk, M, C, mh, hr, s);
                kernels::f32_to_f16_bulk(mh, m16, M, s);
                to_f32(m16, M, x2, s);
                m.gemm.f16(m16, B.w2[k].w, mod, 1, 3 * C, M);
                if (B.m2[k] != nullptr)
                    train::row_lora_fwd(B.m2[k]->A, B.m2[k]->B, x2, 1.f, rk, 3 * C, M, mod, hr, s);
                train::add_into(mod, alora, 3 * C, s);
            }
        }
    }
    float* fmod = pool.f(2 * C);
    {
        float* mh = pool.f(M);
        uint16_t* m16 = pool.u(M);
        m.gemm.f16(e16 + C, m.fm1.w, mh, 1, M, C);
        kernels::f32_to_f16_bulk(mh, m16, M, s);
        m.gemm.f16(m16, m.fm2.w, fmod, 1, 2 * C, M);
        train::add_into(fmod, alora, 2 * C, s);
    }
    float *cs = pool.f(HW * m.hd / 2), *sn = pool.f(HW * m.hd / 2);
    {
        std::vector<float> cv, sv;
        anima_rope_tables(smp.h, smp.w, m.hd, cv, sv);
        gpu::stream_sync(s);
        if (!gpu::copy(cs, cv.data(), cv.size() * 4) || !gpu::copy(sn, sv.data(), sv.size() * 4))
            throw std::runtime_error("copy");
    }
    auto run_of = [&](int64_t b) {
        const Impl::Block& B = m.blk[(size_t) b];
        AnimaBlockRun r;
        r.C = C;
        r.heads = m.H;
        r.hd = m.hd;
        r.F = m.F;
        r.rows = HW;
        r.L = L;
        r.Cc = Cc;
        r.cos = cs;
        r.sin = sn;
        r.mod_sa = mods + (b * 3 + 0) * 3 * C;
        r.mod_ca = mods + (b * 3 + 1) * 3 * C;
        r.mod_ml = mods + (b * 3 + 2) * 3 * C;
        r.ctx16 = ctx16;
        r.ctx16t = ctx16t;
        r.ctx_rs = ctx_rs;
        r.sqn = B.sqn;
        r.skn = B.skn;
        r.cqn = B.cqn;
        r.ckn = B.ckn;
        for (int i = 0; i < kAnProducts; ++i) {
            r.p[i] = B.p[i];
            r.ad[i] = B.l[i] != nullptr ? &B.l[i]->ad : nullptr;
        }
        return r;
    };
    if (hostm) m.host.keep(0, bounds, HW * C * 4);
    for (int64_t b = 0; b < nb; ++b) {
        // the next block's products fetched meanwhile; past the last, the backward's second when it follows
        m.bind(b, backward ? m.ws.begin((int) b, (int) (b + 1 < nb ? b + 1 : nb - 2)) : m.ws.begin((int) b));
        AnimaBlockRun r = run_of(b);
        if (b >= first_kept) r.tape = m.tape(b);
        if (bounds16 != nullptr && b == 0) train::bf16_keep(bounds, HW * C, bounds16, s);
        if (hostm && b > 0) m.host.kept_wait(b - 1);   // this block writes the buffer slot b-1 was copied from
        r.x = all32 ? bounds + b * HW * C : bounds + (b % 2) * HW * C;
        r.out = all32 ? bounds + (b + 1) * HW * C : bounds + ((b + 1) % 2) * HW * C;
        anima_block(r, m.gemm, s, &m.arena);
        m.ws.end((int) b);
        if (bounds16 != nullptr) train::bf16_keep(r.out, HW * C, bounds16 + (b + 1) * HW * C, s);
        if (hostm) m.host.keep(b + 1, r.out, HW * C * 4);
    }
    float* last = all32 ? bounds + nb * HW * C : bounds + (nb % 2) * HW * C;
    // the final layer: the norm modulated by the time, the projection to the patch's values, in the tokens' order
    float *mean = pool.f(HW), *inv = pool.f(HW);
    dit_rows::rows16(dit_rows::LnMod{last, fmod, fmod + C, C}, HW, C, x16, rs, s, mean, inv);
    float *vp = pool.f(HW * TV), *v = pool.f(HW * TV);
    m.gemm.f16(x16, m.flin.w, vp, HW, TV, C);
    dit_rows::rows_back(vp, rs, nullptr, HW, TV, s);
    patch_order(vp, HW, m.lat, v, false, s);
    float* dv = pool.f(HW * TV);
    const float loss = train::flow_loss(v, dx0, deps, HW * TV, dv, nullptr, nullptr, s) * weight;
    if (!backward) return loss;
    if (weight != 1.f) {
        const float w = weight;
        dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (HW * TV)), [=](sycl::id<1> id) { dv[id[0]] *= w; });
    }
    // the final layer backwards (frozen: its input's gradient only)
    patch_order(dv, HW, m.lat, vp, true, s);
    uint16_t* dv16 = pool.u(HW * TV);
    float* dvrs = pool.f(HW);
    dit_rows::rows16(dit_rows::Grad{{vp, TV}}, HW, TV, dv16, dvrs, s);
    uint16_t* flin_t = pool.u(C * TV);
    train::transpose_f16(m.flin.w, TV, C, flin_t, s);
    float *dln = pool.f(HW * C), *sums = pool.f(HW * C);
    train::dgrad(dv16, dvrs, flin_t, HW, TV, C, dln, sums, false, m.gemm, s);
    float *dcur = pool.f(HW * C), *dnxt = pool.f(HW * C);
    dit_rows::ln_mod_bwd(last, mean, inv, fmod + C, dln, HW, C, dcur, nullptr, nullptr, s);
    if (hostm) {   // the pass buffers are the backward's read-back buffers now; the first two blocks prefetched
        m.host.used(0);
        m.host.used(1);
        for (int64_t b = nb - 1; b >= 0 && b >= nb - 2; --b)
            m.host.fetch(b, bounds + (b % 2) * HW * C, HW * C * 4, (int) (b % 2));
    }
    // the blocks from the last, each one's modulations' gradients to their adapters
    float* dmod = pool.f(9 * C);
    float *g = pool.f(std::max<int64_t>(rk, 1)), *dx2 = pool.f(M);
    // the blocks' outputs in the backward (their forwards' again: not read) once for all; BF16: the forward's other
    // buffer, and the last one (the head's, done with below) the input read back into
    float* bout = bounds16 != nullptr ? bounds + ((nb + 1) % 2) * HW * C : pool.f(HW * C);
    for (int64_t b = nb - 1; b >= 0; --b) {
        m.bind(b, m.ws.begin((int) b, (int) b - 1));   // the block below fetched meanwhile
        Impl::Block& B = m.blk[(size_t) b];
        AnimaBlockRun r = run_of(b);
        m.dgrad_weights(r.p, kAnProducts);
        if (hostm) {
            m.host.wait(b);
            r.x = bounds + (b % 2) * HW * C;
        } else if (half) {
            train::bf16_load(bounds16 + b * HW * C, HW * C, last, s);
            r.x = last;
        } else {
            r.x = bounds + b * HW * C;
        }
        r.out = bout;
        r.dout = dcur;
        r.dx = b > 0 ? dnxt : nullptr;
        if (b >= first_kept) {   // its forward's tape: the backward alone
            r.tape = m.tape(b);
            r.tape_replay = true;
        }
        if (rk > 0) {
            if (!gpu::memset_async(dmod, 0, (size_t) (9 * C) * 4, s)) throw std::runtime_error("zero");
            r.dmod_sa = dmod;
            r.dmod_ca = dmod + 3 * C;
            r.dmod_ml = dmod + 6 * C;
        }
        if (b > 0 && !gpu::memset_async(dnxt, 0, (size_t) (HW * C) * 4, s)) throw std::runtime_error("zero");
        anima_block(r, m.gemm, s, &m.arena);
        m.ws.end((int) b);
        if (rk > 0)
            for (int k = 0; k < 3; ++k) {
                const float* d = dmod + (int64_t) k * 3 * C;
                const float* x2 = x2s + (b * 3 + k) * M;
                const train::Lora *l1 = B.m1[k], *l2 = B.m2[k];   // either may be untrained (train::Targets)
                if (l2 != nullptr)
                    train::row_lora_bwd(l2->A, l2->B, x2, d, 1.f, rk, 3 * C, M, l2->ad.dA, l2->ad.dB, hr, g, s);
                if (l1 == nullptr) continue;
                row_dgrad(B.w2[k].w, d, 3 * C, M, l2 != nullptr ? l2->A : nullptr, g, l2 != nullptr ? rk : 0, dx2, s);
                train::row_lora_bwd(l1->A, l1->B, x1, dx2, 1.f, rk, M, C, l1->ad.dA, l1->ad.dB, hr, g, s);
            }
        if (hostm) {
            m.host.used((int) (b % 2));
            if (b > 0) m.host.fetch(b - 1, bounds + ((b - 1) % 2) * HW * C, HW * C * 4, (int) ((b - 1) % 2));
        }
        std::swap(dcur, dnxt);
    }
    gpu::stream_sync(s);
    m.stepped_rows(HW, (uint64_t) HW * C * 4);
    return loss;
}

double AnimaTrainer::adamw(const train::AdamW& o) { return m_->update(o); }

void AnimaTrainer::save(const std::string& path, const std::vector<std::pair<std::string, std::string>>& meta) const {
    const Impl& m = *m_;
    std::vector<train::Tensor> ts;
    for (size_t b = 0; b < m.blk.size(); ++b) {
        const Impl::Block& B = m.blk[b];
        const std::string p = "diffusion_model.blocks." + std::to_string(b) + ".";
        for (int i = 0; i < kAnProducts; ++i) {
            if (B.l[i] == nullptr) continue;   // a product the run does not train
            const train::Lora& l = *B.l[i];
            ts.push_back({p + kProduct[i] + ".lora_A.weight", {m.rank, l.K}, m.read(l.A, m.rank * l.K)});
            ts.push_back({p + kProduct[i] + ".lora_B.weight", {l.N, m.rank}, m.read(l.B, l.N * m.rank)});
        }
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 2; ++j) {
                const train::Lora* lp = j == 0 ? B.m1[k] : B.m2[k];
                if (lp == nullptr) continue;
                const train::Lora& l = *lp;
                const std::string n = p + kMod[k] + (j == 0 ? ".1" : ".2");
                ts.push_back({n + ".lora_A.weight", {m.rank, l.K}, m.read(l.A, m.rank * l.K)});
                ts.push_back({n + ".lora_B.weight", {l.N, m.rank}, m.read(l.B, l.N * m.rank)});
            }
    }
    train::save_safetensors_bf16(path, std::move(ts), meta);
}

}  // namespace strata::image
