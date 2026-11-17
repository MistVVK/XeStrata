// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_trainer.cpp - see include/strata/image/moss_trainer.hpp.  The parts outside the blocks are the
// inference's (MossDit::forward, time_embedding, rope_tables and the head), frozen: the input projection, the time's
// two layers, the shared modulation and the head's, the final norm and projection.
#include "strata/image/moss_trainer.hpp"
#include "strata/image/dit_file.hpp"
#include "strata/image/lora_names.hpp"
#include "strata/image/moss_dit.hpp"
#include "strata/image/moss_train.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/image/train_trainer.hpp"
#include "strata/platform/mapped_file.hpp"

#include "strata/core/gpu.hpp"
#include "strata/platform/env.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

constexpr float kEps = 1e-6f;
constexpr int64_t kFreq = 256;   // the time's sinusoid
static_assert(std::size(lora::kMossBlock) == kMossProducts);
const char* const* const kProduct = lora::kMossBlock;   // the DiT's target names, in kMossProducts' order

// a dense product outside the blocks: FP16 rows, its bias (null: none)
struct Dense {
    uint16_t* w = nullptr;
    float* b = nullptr;
    int64_t N = 0, K = 0;
};

}  // namespace

struct MossTrainer::Impl : train::Trainer {
    Impl() : Trainer("MOSS-Image training") {}
    int64_t C = 0, F = 0, H = 0, hd = 0, in_ch = 0, rank = 0;
    double rope_base = 10000.0;
    Dense vin, t1, t2, smod, hmod, fout;
    float* fnorm = nullptr;
    struct Block {
        train::Product p[kMossProducts];
        float *n1 = nullptr, *n2 = nullptr, *nq = nullptr, *nk = nullptr;
        train::Lora* l[kMossProducts] = {};   // null: the block is not adapted
    };
    std::vector<Block> blk;
    uint16_t* plain0 = nullptr;         // block 0's qkv rotated back (an int8 base's unquantized input)
    // The blocks' products through the trainer's LayerStream: their rows as the products read them (FP16, or an int8
    // base's rows and scales) made once at open and kept in RAM, blk's pointers set as a block is used (bind)
    std::vector<std::array<uint64_t, kMossProducts>> woff;
    bool int8 = false;
    void bind(int64_t b, const uint8_t* base) {
        Block& B = blk[(size_t) b];
        for (int i = 0; i < kMossProducts; ++i) {
            train::Product& pr = B.p[i];
            const uint8_t* at = base + woff[(size_t) b][(size_t) i];
            if (int8) {
                pr.w8 = reinterpret_cast<const int8_t*>(at);
                pr.sw = reinterpret_cast<const float*>(at + pr.N * pr.K);
            } else {
                pr.w = reinterpret_cast<const uint16_t*>(at);
            }
        }
    }
    // the step's device rows
    uint16_t *e16 = nullptr, *t16 = nullptr;
    float *temb = nullptr, *mod = nullptr, *head = nullptr;

    // the persistent pointers after the trainer's pause/resume (the values come back at new addresses)
    void remap() override {
        auto mv = [&](auto*& p) {
            using T = std::remove_reference_t<decltype(p)>;
            p = static_cast<T>(moved(p));
        };
        for (Dense* d : {&vin, &t1, &t2, &smod, &hmod, &fout}) {
            mv(d->w);
            mv(d->b);
        }
        mv(fnorm);
        mv(plain0);
        mv(e16);
        mv(t16);
        mv(temb);
        mv(mod);
        mv(head);
        for (Block& B : blk) {
            mv(B.n1);
            mv(B.n2);
            mv(B.nq);
            mv(B.nk);   // B.p's weights are bound again at each use (bind)
        }
    }
};

MossTrainer::MossTrainer() : m_(std::make_unique<Impl>()) {}

train::Trainer& MossTrainer::base() { return *m_; }
MossTrainer::~MossTrainer() = default;
int64_t MossTrainer::text_cols() const { return m_->C; }

bool MossTrainer::open(const std::string& dit, int64_t rank, bool int8, const train::Targets& targets, uint64_t seed,
                       void* stream, std::string& err) {
    Impl& m = *m_;
    try {
        m.rank = rank;
        if (!m.init(stream, 1 << 20, err)) return false;
        const MossFile f(dit);
        if (std::filesystem::is_directory(dit)) {   // config.json's rope_base, where it differs from 10000
            std::ifstream cf(std::filesystem::path(dit) / "config.json");
            const std::string cfg((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
            if (const size_t at = cfg.find("\"rope_base\""); at != std::string::npos)
                m.rope_base = std::stod(cfg.substr(cfg.find(':', at) + 1));
        }
        auto dense = [&](const std::string& name, bool bias) {
            Dense d;
            std::vector<uint16_t> h;
            f.rows16(name + ".weight", h, d.N, d.K);
            d.w = m.dev16(h.data(), d.N * d.K);
            if (bias) d.b = m.dev_vec(f.f32(name + ".bias"));
            return d;
        };
        m.vin = dense("video_embedder", true);
        m.C = m.vin.N;
        m.in_ch = m.vin.K;
        m.t1 = dense("time_embedding.0", true);
        m.t2 = dense("time_embedding.2", true);
        m.smod = dense("shared_block_modulation", true);
        m.hmod = dense("video_head_modulation", true);
        m.fout = dense("final_linear_video", false);
        m.fnorm = m.dev_vec(f.f32("final_norm_video.weight"));
        if (m.smod.N != 6 * m.C || m.hmod.N != 2 * m.C || m.t1.K != kFreq)
            throw std::runtime_error("image engine (MOSS-Image training): an unexpected shape");
        std::mt19937_64 rng(seed);
        m.int8 = int8;
        uint16_t* q16 = nullptr;   // an int8 base's rows made on the GPU a product at a time, through these
        int8_t* q8 = nullptr;
        float* qs = nullptr;
        int64_t qcap = 0, scap = 0;
        for (int b = 0;; ++b) {
            const std::string p = "layers." + std::to_string(b) + ".";
            if (!f.has(p + "proj.o.weight")) break;
            Impl::Block blk;
            std::vector<std::vector<uint8_t>> parts(kMossProducts);   // each product's bytes as the region holds them
            for (int i = 0; i < kMossProducts; ++i) {
                train::Product& pr = blk.p[i];
                std::vector<uint16_t> h;
                int64_t N = 0, K = 0;
                if (i == kMossQkv && !f.has(p + "proj.qkv.weight")) {   // the vendor's q, k and v: fused, rows q, k, v
                    for (const char* n : {"proj.q", "proj.k", "proj.v"}) {
                        int64_t n1 = 0;
                        f.rows16(p + n + ".weight", h, n1, K);
                        N += n1;
                    }
                } else {
                    f.rows16(p + kProduct[i] + ".weight", h, N, K);
                }
                pr.N = N;
                pr.K = K;
                auto& part = parts[(size_t) i];
                if (int8) {   // the int8 base's rows and scales from the 16-bit rows, made on the GPU
                    if (N * K > qcap) {
                        for (void* x : {(void*) q16, (void*) q8}) m.release(x);
                        q16 = m.get<uint16_t>(N * K);
                        q8 = m.get<int8_t>(N * K);
                        qcap = N * K;
                    }
                    if (N > scap) {
                        m.release(qs);
                        qs = m.get<float>(N);
                        scap = N;
                    }
                    gpu::stream_sync(m.s);
                    if (!gpu::copy(q16, h.data(), h.size() * 2)) throw std::runtime_error("copy");
                    train::w8_quantize(q16, N, K, q8, qs, stream);
                    if (b == 0 && i == kMossQkv) {   // block 0's qkv rotated back (its unquantized input)
                        m.plain0 = m.get<uint16_t>(N * K);
                        train::w8_dequant_rows(q8, qs, N, K, m.plain0, stream);
                    }
                    gpu::stream_sync(m.s);
                    part.resize((size_t) (N * K + 4 * N));
                    if (!gpu::copy(part.data(), q8, (size_t) (N * K)) ||
                        !gpu::copy(part.data() + N * K, qs, (size_t) N * 4))
                        throw std::runtime_error("copy");
                } else {
                    part.resize(h.size() * 2);
                    std::memcpy(part.data(), h.data(), part.size());
                }
            }
            {
                uint64_t bytes = 0;
                for (const auto& part : parts) bytes += part.size();
                m.whost.emplace_back((size_t) bytes);
                uint8_t* hp = m.whost.back().data();
                std::vector<LayerStream::Span> spans;
                for (const auto& part : parts) {
                    std::memcpy(hp, part.data(), part.size());
                    spans.push_back({hp, (uint64_t) part.size(), 0});
                    hp += part.size();
                }
                const int layer = m.ws.add(spans);
                std::array<uint64_t, kMossProducts> off{};
                for (int i = 0; i < kMossProducts; ++i) off[(size_t) i] = m.ws.offset(layer, i);
                m.woff.push_back(off);
            }
            blk.n1 = m.dev_vec(f.f32(p + "norm1.weight"));
            blk.n2 = m.dev_vec(f.f32(p + "norm2.weight"));
            blk.nq = m.dev_vec(f.f32(p + "q_norm.weight"));
            blk.nk = m.dev_vec(f.f32(p + "k_norm.weight"));
            for (int i = 0; i < kMossProducts && rank > 0; ++i)
                if (targets.take(b, p + kProduct[i])) blk.l[i] = &m.add_lora(blk.p[i].N, blk.p[i].K, rank, rng);
            m.blk.push_back(blk);
        }
        if (m.blk.empty()) throw std::runtime_error("image engine (MOSS-Image training): the DiT has no blocks");
        if (rank > 0 && m.loras().empty())
            throw std::runtime_error("image engine (MOSS-Image training): the weights chosen (" + targets.text() +
                                     ") are none of the blocks' products");
        m.hd = (int64_t) f.f32("layers.0.q_norm.weight").size();
        m.H = m.C / m.hd;
        m.F = m.blk[0].p[kMossUpGate].N / 2;
        for (void* x : {(void*) q16, (void*) q8, (void*) qs}) m.release(x);
        m.dgrad_scratch(m.blk[0].p, kMossProducts, int8);
        m.set_budget(0, 0, 0);
        m.e16 = m.get<uint16_t>(kFreq);
        m.t16 = m.get<uint16_t>(2 * m.C);
        m.temb = m.get<float>(m.C);
        m.mod = m.get<float>(6 * m.C);
        m.head = m.get<float>(2 * m.C);
        gpu::stream_sync(stream);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

float MossTrainer::step(const MfTrainSample& smp, const std::vector<float>& x0, const std::vector<float>& eps,
                        float sigma, float weight, bool backward) {
    Impl& m = *m_;
    void* s = m.s;
    const int64_t C = m.C, HW = smp.tokens, L = smp.rows, R = HW + L, IC = m.in_ch, nb = (int64_t) m.blk.size();
    if ((int64_t) x0.size() != HW * IC || (int64_t) eps.size() != HW * IC || smp.w * smp.h != HW)
        throw std::invalid_argument("image engine (MOSS-Image training): the sample's shapes");
    if ((int64_t) smp.te.size() != L * C) throw std::invalid_argument("image engine (MOSS-Image training): text rows");
    static train::StageTimes tm("moss step");
    tm.begin(s);
    train::Pool pool(s);
    // the noised latent's rows through the input projection, then the text embedder's rows
    std::vector<float> xs((size_t) (HW * IC));
    for (size_t i = 0; i < xs.size(); ++i) xs[i] = (1.f - sigma) * x0[i] + sigma * eps[i];
    float* dx0 = pool.f(HW * IC);
    float* deps = pool.f(HW * IC);
    float* dxs = pool.f(HW * IC);
    gpu::stream_sync(s);
    if (!gpu::copy(dx0, x0.data(), x0.size() * 4) || !gpu::copy(deps, eps.data(), eps.size() * 4) ||
        !gpu::copy(dxs, xs.data(), xs.size() * 4))
        throw std::runtime_error("image engine (MOSS-Image training): copy");
    // the blocks' inputs' form and the tapes kept this step, decided before the step's own buffers are taken (every
    // block's input, the backward's two gradients and its blocks' output; in BF16, the inputs at half and two FP32
    // rows' buffers the forward passes between and the backward reads an input into and writes its output to)
    int64_t first_kept = nb;
    bool half = false, hostm = false;
    m.set_budget(R, (uint64_t) ((nb + 1) * C) * 2, (uint64_t) (13 * m.F) * 4);
    if (backward) {
        MossBlockRun z;
        z.C = C;
        z.heads = m.H;
        z.hd = m.hd;
        z.F = m.F;
        z.rows = R;
        for (int i = 0; i < kMossProducts; ++i) {
            z.p[i] = m.blk.back().p[i];
            z.ad[i] = nullptr;   // a block's tape at its largest: the product adapted where any block adapts it
            for (const Impl::Block& B : m.blk)
                if (B.l[i] != nullptr) {
                    z.ad[i] = &B.l[i]->ad;
                    break;
                }
        }
        const size_t rc = (size_t) (R * C), tape = moss_block_tape_bytes(z, m.gemm);
        // before a step has shown the arena's size, a block's tape's bytes for its growth
        half = m.inputs_half(R, (size_t) (nb + 1 + 3) * rc * 4, tape / (size_t) R);
        // the host form first (the inputs as FP32, byte for byte): the pinned memory had, else BF16
        hostm = half && m.host_inputs(nb + 1, (int64_t) rc * 4);
        first_kept = nb - m.keep_tapes(nb, R, tape,
                                       hostm ? (size_t) (5 * rc * 4)
                                             : half ? (size_t) (nb + 1) * rc * 2 + 4 * rc * 4
                                                    : (size_t) (nb + 1 + 3) * rc * 4);
    }
    m.set_inputs_mode(hostm ? "host" : half ? "bf16" : "fp32");
    if (!hostm) m.host.close();
    // FP32: every block's input in bounds (two of them passed between, the forward alone); BF16: bounds the two
    // passed between, the inputs in bounds16; host: bounds the two passed between, the inputs on the host (FP32)
    const bool all32 = backward && !half && !hostm;
    float* bounds = pool.f((all32 ? nb + 1 : 2) * R * C);
    uint16_t* bounds16 = half && !hostm ? pool.u((nb + 1) * R * C) : nullptr;
    uint16_t* x16 = pool.u(R * std::max(C, IC));
    float* rs = pool.f(R);
    kernels::f32_to_f16_bulk(dxs, x16, HW * IC, s);
    m.gemm.f16(x16, m.vin.w, bounds, HW, C, IC);
    dit_rows::rows_back(bounds, nullptr, m.vin.b, HW, C, s);
    gpu::stream_sync(s);   // bounds' zero (the pool's, on this stream) before the copy (on the compute queue)
    if (!gpu::copy(bounds + HW * C, smp.te.data(), smp.te.size() * 4))
        throw std::runtime_error("image engine (MOSS-Image training): copy");
    // the time: the sinusoid of sigma * 1000 (FP64, bf16-rounded), two layers with SiLU between, then the shared
    // modulation's six rows and the head's two from temb itself
    {
        std::vector<uint16_t> e;
        moss_time_sinusoid((double) sigma * 1000.0, kFreq, e);
        if (!gpu::copy(m.e16, e.data(), e.size() * 2)) throw std::runtime_error("copy");
        m.gemm.f16(m.e16, m.t1.w, m.temb, 1, C, kFreq);
        dit_rows::rows_back(m.temb, nullptr, m.t1.b, 1, C, s);
        qi_silu(m.temb, C, m.t16, s);
        m.gemm.f16(m.t16, m.t2.w, m.temb, 1, C, C);
        dit_rows::rows_back(m.temb, nullptr, m.t2.b, 1, C, s);
        kernels::f32_to_f16_bulk(m.temb, m.t16 + C, C, s);
        m.gemm.f16(m.t16 + C, m.smod.w, m.mod, 1, 6 * C, C);
        dit_rows::rows_back(m.mod, nullptr, m.smod.b, 1, 6 * C, s);
        m.gemm.f16(m.t16 + C, m.hmod.w, m.head, 1, 2 * C, C);
        dit_rows::rows_back(m.head, nullptr, m.hmod.b, 1, 2 * C, s);
    }
    // the rotary tables: an image row (0, the text's length, its row and column centred), a text row (0, i, 0, 0)
    float *cs = pool.f(R * m.hd / 2), *sn = pool.f(R * m.hd / 2);
    {
        std::vector<float> cv, sv;
        const int64_t h = smp.h, w = smp.w;
        moss_rope_rows(R, m.hd, m.rope_base, [&](int64_t r, float* p) {
            if (r < HW) {
                const int64_t y = r / w, x = r - y * w;
                p[0] = 0.f;
                p[1] = (float) L;
                p[2] = (float) y - (float) (h - 1) / 2.f;
                p[3] = (float) x - (float) (w - 1) / 2.f;
            } else {
                p[0] = 0.f;
                p[1] = (float) (r - HW);
                p[2] = p[3] = 0.f;
            }
        }, cv, sv);
        gpu::stream_sync(s);
        if (!gpu::copy(cs, cv.data(), cv.size() * 4) || !gpu::copy(sn, sv.data(), sv.size() * 4))
            throw std::runtime_error("copy");
    }
    auto run_of = [&](int64_t b) {
        const Impl::Block& B = m.blk[(size_t) b];
        MossBlockRun r;
        r.C = C;
        r.heads = m.H;
        r.hd = m.hd;
        r.F = m.F;
        r.rows = R;
        r.cos = cs;
        r.sin = sn;
        r.mod = m.mod;
        r.n1 = B.n1;
        r.n2 = B.n2;
        r.nq = B.nq;
        r.nk = B.nk;
        for (int i = 0; i < kMossProducts; ++i) {
            r.p[i] = B.p[i];
            r.ad[i] = B.l[i] != nullptr ? &B.l[i]->ad : nullptr;
        }
        if (b == 0) r.w_plain_qkv = m.plain0;
        return r;
    };
    tm.mark("inputs, time, rope", s);
    // forward: every block's input kept for the backward, the last blocks' tapes as the memory takes them
    if (hostm) m.host.keep(0, bounds, R * C * 4);
    for (int64_t b = 0; b < nb; ++b) {
        // the next block's products fetched meanwhile; past the last, the backward's second when it follows
        m.bind(b, backward ? m.ws.begin((int) b, (int) (b + 1 < nb ? b + 1 : nb - 2)) : m.ws.begin((int) b));
        MossBlockRun r = run_of(b);
        if (b >= first_kept) r.tape = m.tape(b);
        if (bounds16 != nullptr && b == 0) train::bf16_keep(bounds, R * C, bounds16, s);
        if (hostm && b > 0) m.host.kept_wait(b - 1);   // this block writes the buffer slot b-1 was copied from
        r.x = all32 ? bounds + b * R * C : bounds + (b % 2) * R * C;
        r.out = all32 ? bounds + (b + 1) * R * C : bounds + ((b + 1) % 2) * R * C;
        moss_block(r, m.gemm, s, &m.arena);
        m.ws.end((int) b);
        if (bounds16 != nullptr) train::bf16_keep(r.out, R * C, bounds16 + (b + 1) * R * C, s);
        if (hostm) m.host.keep(b + 1, r.out, R * C * 4);
    }
    tm.mark("blocks forward", s);
    float* last = all32 ? bounds + nb * R * C : bounds + (nb % 2) * R * C;
    // the head on the image's rows: the final norm scaled and shifted by the time, the projection to the latent
    float *mean = pool.f(HW), *inv = pool.f(HW);
    dit_rows::rows16(dit_rows::RmsAffine{last, m.fnorm, m.head + C, m.head, C, kEps, true}, HW, C, x16, rs, s, mean,
                     inv);
    float* v = pool.f(HW * IC);
    m.gemm.f16(x16, m.fout.w, v, HW, IC, C);
    dit_rows::rows_back(v, rs, nullptr, HW, IC, s);
    float* dv = pool.f(HW * IC);
    const float loss = train::flow_loss(v, dx0, deps, HW * IC, dv, nullptr, nullptr, s) * weight;
    if (!backward) return loss;
    if (weight != 1.f) {
        const float w = weight;
        dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (HW * IC)), [=](sycl::id<1> id) { dv[id[0]] *= w; });
    }
    tm.mark("head forward + loss", s);
    // the head backwards (frozen: the input's gradient only), then every block from its kept input
    uint16_t* dv16 = pool.u(HW * IC);
    float* dvrs = pool.f(HW);
    dit_rows::rows16(dit_rows::Grad{{dv, IC}}, HW, IC, dv16, dvrs, s);
    uint16_t* fout_t = pool.u(C * IC);
    train::transpose_f16(m.fout.w, IC, C, fout_t, s);
    float* dln = pool.f(HW * C);
    float* sums = pool.f(HW * C);
    train::dgrad(dv16, dvrs, fout_t, HW, IC, C, dln, sums, false, m.gemm, s);
    // the blocks' outputs in the backward (its forward's again: not read); BF16: the forward's other buffer, and the
    // last one (the head's, done with) the input read back into; host: the two pass buffers, read back a block ahead
    float* out = bounds16 != nullptr ? bounds + ((nb + 1) % 2) * R * C : pool.f(R * C);
    float* dcur = pool.f(R * C);   // the text rows' gradient from the head: none
    float* dnxt = pool.f(R * C);
    dit_rows::rms_affine_bwd(last, inv, m.fnorm, m.head + C, dln, HW, C, dcur, s);
    if (hostm) {
        m.host.used(0);
        m.host.used(1);
        for (int64_t b = nb - 1; b >= 0 && b >= nb - 2; --b)
            m.host.fetch(b, bounds + (b % 2) * R * C, R * C * 4, (int) (b % 2));
    }
    tm.mark("head backward", s);
    for (int64_t b = nb - 1; b >= 0; --b) {
        m.bind(b, m.ws.begin((int) b, (int) b - 1));   // the block below fetched meanwhile
        MossBlockRun r = run_of(b);
        m.dgrad_weights(r.p, kMossProducts);   // (an int8 base's straight through)
        if (hostm) {
            m.host.wait(b);
            r.x = bounds + (b % 2) * R * C;
        } else if (half) {
            train::bf16_load(bounds16 + b * R * C, R * C, last, s);
            r.x = last;
        } else {
            r.x = bounds + b * R * C;
        }
        r.out = out;
        r.dout = dcur;
        r.dx = b > 0 ? dnxt : nullptr;   // nothing below block 0 takes a gradient
        if (b >= first_kept) {   // its forward's tape: the backward alone
            r.tape = m.tape(b);
            r.tape_replay = true;
        }
        if (b > 0 && !gpu::memset_async(dnxt, 0, (size_t) (R * C) * 4, s))
            throw std::runtime_error("image engine (MOSS-Image training): zero");
        moss_block(r, m.gemm, s, &m.arena);
        m.ws.end((int) b);
        if (hostm) {
            m.host.used((int) (b % 2));
            if (b > 0) m.host.fetch(b - 1, bounds + ((b - 1) % 2) * R * C, R * C * 4, (int) ((b - 1) % 2));
        }
        std::swap(dcur, dnxt);
    }
    gpu::stream_sync(s);
    m.stepped_rows(R, (uint64_t) R * C * 4);
    tm.mark("blocks backward", s);
    static int64_t n = 0;
    if (tm.on() && ++n % 10 == 0) {   // every 10 steps: the step's stages, then the blocks'
        tm.report();
        moss_block_profile_report();
        train::ad_profile_report();
    }
    return loss;
}

double MossTrainer::adamw(const train::AdamW& o) {
    Impl& m = *m_;
    static train::StageTimes tm("moss adamw");
    tm.begin(m.s);
    const double norm = m.update(o);
    tm.mark("clip + adamw + views", m.s);
    static int64_t n = 0;
    if (tm.on() && ++n % 10 == 0) tm.report();
    return norm;
}

void MossTrainer::save(const std::string& path, const std::vector<std::pair<std::string, std::string>>& meta) const {
    const Impl& m = *m_;
    std::vector<train::Tensor> ts;
    for (size_t b = 0; b < m.blk.size(); ++b) {
        const Impl::Block& B = m.blk[b];
        const std::string p = "diffusion_model.layers." + std::to_string(b) + ".";
        for (int i = 0; i < kMossProducts; ++i) {
            if (B.l[i] == nullptr) continue;   // a product the run does not train
            const train::Lora& l = *B.l[i];
            const std::vector<float> A = m.read(l.A, m.rank * l.K), Bv = m.read(l.B, l.N * m.rank);
            if (i == kMossQkv) {   // q, k and v: the fused A each, B's rows split in three
                const int64_t n = l.N / 3;
                const char* const* nm = lora::kMossQkv;   // q, k and v's names: the fused one is kMossQkv[3]
                for (int j = 0; j < 3; ++j) {
                    ts.push_back({p + nm[j] + ".lora_A.weight", {m.rank, l.K}, A});
                    ts.push_back({p + nm[j] + ".lora_B.weight", {n, m.rank},
                                  std::vector<float>(Bv.begin() + j * n * m.rank, Bv.begin() + (j + 1) * n * m.rank)});
                }
                continue;
            }
            ts.push_back({p + kProduct[i] + ".lora_A.weight", {m.rank, l.K}, A});
            ts.push_back({p + kProduct[i] + ".lora_B.weight", {l.N, m.rank}, Bv});
        }
    }
    train::save_safetensors_bf16(path, std::move(ts), meta);
}

}  // namespace strata::image
