// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/anima_trainer.cpp - see include/strata/image/anima_trainer.hpp.  The parts outside the blocks are the
// inference's (AnimaDit::forward, time_embedding, modulation and the final layer), frozen: the patch embedder, the
// time's two layers and its norm, the final layer's modulation and projection.
#include "strata/image/anima_trainer.hpp"
#include "strata/image/anima_dit.hpp"
#include "strata/image/anima_train.hpp"
#include "strata/image/mapped_file.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/image/safetensors.hpp"

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
#include <random>
#include <stdexcept>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

constexpr float kEps = 1e-6f;
const char* const kProduct[kAnProducts] = {"self_attn.q_proj",  "self_attn.k_proj",  "self_attn.v_proj",
                                           "self_attn.output_proj", "cross_attn.q_proj", "cross_attn.k_proj",
                                           "cross_attn.v_proj", "cross_attn.output_proj", "mlp.layer1",
                                           "mlp.layer2"};
const char* const kMod[3] = {"adaln_modulation_self_attn", "adaln_modulation_cross_attn", "adaln_modulation_mlp"};

struct Lora {
    train::Adapter ad;
    float *A = nullptr, *B = nullptr, *mA = nullptr, *vA = nullptr, *mB = nullptr, *vB = nullptr;
    int64_t N = 0, K = 0;
};

// a one-row layer's LoRA (a modulation's): FP32 masters read as they are
struct RowLora {
    float *A = nullptr, *B = nullptr, *dA = nullptr, *dB = nullptr;
    float *mA = nullptr, *vA = nullptr, *mB = nullptr, *vB = nullptr;
    int64_t N = 0, K = 0;
};

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

struct AnimaTrainer::Impl {
    void* s = nullptr;
    std::vector<void*> owned;
    prefill::Gemm gemm;
    int64_t C = 0, F = 0, H = 0, hd = 0, lat = 0, Cc = 0, M = 0, rank = 0, steps = 0;
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
        Lora l[kAnProducts];
        RowLora m1[3], m2[3];
    };
    std::vector<Block> blk;
    uint16_t* wt[kAnProducts] = {};
    train::Arena arena;   // the blocks' buffers, kept from block to block and step to step
    int64_t scratch_rows = 0;   // the most rows a step has had: the arena holds what they took
    // The blocks' ten products through a LayerStream (Qwen-Image-2.1's trainer's): as many blocks as the memory leaves
    // stay in VRAM, the rest are copied in from RAM a block at a time (their FP16 rows made once at open and kept in
    // RAM), blk's pointers set as a block is used (bind)
    LayerStream ws;
    std::vector<std::vector<uint8_t>> whost;
    std::vector<std::array<uint64_t, kAnProducts>> woff;
    bool stepped = false, budget_stepped = false;
    int64_t budget_rows = 0;
    void bind(int64_t b, const uint8_t* base) {
        for (int i = 0; i < kAnProducts; ++i)
            blk[(size_t) b].p[i].w = reinterpret_cast<const uint16_t*>(base + woff[(size_t) b][(size_t) i]);
    }
    // The VRAM the products take (Qwen-Image-2.1's trainer's set_budget), at open (no rows), when a step brings more
    // rows than any before and once after the first step: the free memory and what they hold now, less the step's
    // own beside them (its blocks' inputs in BF16 at least, the scratch's growth: train::scratch_growth, before a
    // step has shown it 13 FP32 rows of F a row as Qwen-Image-2.1's; 1.5 GiB of transients, 3 GiB more before the
    // first step).  XESTRATA_TRAIN_WEIGHTS_MIB sets it.
    void set_budget(int64_t R) {
        if (R <= budget_rows && budget_stepped == stepped && ws.vram() > 0) return;
        budget_rows = std::max(budget_rows, R);
        budget_stepped = stepped;
        R = budget_rows;
        const uint64_t all = ws.total();
        uint64_t budget = all;
        if (const char* e = strata::env_get("TRAIN_WEIGHTS_MIB")) {
            budget = (uint64_t) std::atoll(e) << 20;
        } else {
            size_t free_b = 0, total = 0;
            gpu::stream_sync(s);
            if (gpu::mem_info(&free_b, &total)) {
                const uint64_t have = (uint64_t) free_b + ws.vram();
                const uint64_t reserve = (uint64_t) ((int64_t) (blk.size() + 1) * R * C) * 2 +
                                         train::scratch_growth(arena.held(), R, scratch_rows,
                                                               (uint64_t) (13 * R * F) * 4) +
                                         (uint64_t{3} << 29) +
                                         (stepped ? 0 : uint64_t{3} << 30);
                budget = have > reserve ? have - reserve : 0;
            }
        }
        budget = std::min(all, std::max(budget, ws.min_budget()));
        if (budget < all)
            std::fprintf(stderr, "image engine (Anima training): the blocks' weights %llu of %llu MiB in VRAM, the "
                                 "rest streamed from RAM\n",
                         (unsigned long long) (budget >> 20), (unsigned long long) (all >> 20));
        ws.configure(budget, s);
    }
    // Whether this step keeps its blocks' inputs in BF16 (XESTRATA_TRAIN_INPUTS16=0 or 1, or by the memory), as the
    // MOSS-Image trainer does: when the free memory falls short of their FP32 bytes, the arena's growth to this step's
    // rows (in proportion to the most rows so far, 10% over; before a step has shown it, 13 FP32 rows of F a row) and
    // 1.5 GiB of transients
    bool inputs_half(int64_t R) {
        if (const char* e = strata::env_get("TRAIN_INPUTS16")) return std::atoi(e) != 0;
        size_t free_b = 0, total = 0;
        gpu::stream_sync(s);
        if (!gpu::mem_info(&free_b, &total)) return false;
        const size_t held = arena.held();
        const size_t scratch =
            scratch_rows > 0
                ? (size_t) (1.1 * (double) held * (double) std::max(R, scratch_rows) / (double) scratch_rows)
                : held + (size_t) (13 * R * F) * 4;
        const size_t need = (size_t) ((int64_t) (blk.size() + 1 + 3) * R * C) * 4 +
                            (scratch > held ? scratch - held : 0) + (size_t{3} << 29);
        return free_b < need;
    }

    template <class T> T* get(int64_t n) {
        void* q = nullptr;
        if (!gpu::alloc_device(&q, (size_t) n * sizeof(T)) || !gpu::memset_async(q, 0, (size_t) n * sizeof(T), s))
            throw std::runtime_error("image engine (Anima training): GPU memory");
        owned.push_back(q);
        return static_cast<T*>(q);
    }
    float* dev_vec(const std::vector<float>& v) {
        float* d = get<float>((int64_t) v.size());
        gpu::stream_sync(s);
        if (!gpu::copy(d, v.data(), v.size() * 4)) throw std::runtime_error("image engine (Anima training): copy");
        return d;
    }
    ~Impl() {
        if (s != nullptr) gpu::stream_sync(s);
        ws.drop();
        for (void* q : owned) gpu::free(q);
    }
    void sync_views(Lora& l) {
        train::f16_copy(l.A, rank * l.K, l.ad.a, s);
        if (l.ad.at != nullptr) train::transpose_f16(l.ad.a, rank, l.K, l.ad.at, s);
        train::f16_copy(l.B, l.N * rank, l.ad.b, s);
        if (l.ad.bt != nullptr) train::transpose_f16(l.ad.b, l.N, rank, l.ad.bt, s);
    }
};

AnimaTrainer::AnimaTrainer() : m_(std::make_unique<Impl>()) {}
AnimaTrainer::~AnimaTrainer() = default;
int64_t AnimaTrainer::context_cols() const { return m_->Cc; }

bool AnimaTrainer::open(const std::string& dit, int64_t rank, uint64_t seed, void* stream, std::string& err) {
    Impl& m = *m_;
    try {
        m.s = stream;
        m.rank = rank;
        if (!m.gemm.init(stream, 1 << 20, err)) return false;
        SafeTensors st(dit);
        MappedFile file(dit);
        std::string pre;
        for (const char* p : {"", "net.", "model.diffusion_model."})
            if (st.find(std::string(p) + "blocks.0.self_attn.q_proj.weight") != nullptr) pre = p;
        auto find = [&](const std::string& name) -> const SafeTensor& {
            const SafeTensor* t = st.find(pre + name);
            if (t == nullptr) throw std::runtime_error("image engine (Anima training): no " + name);
            return *t;
        };
        // a matrix's rows as FP16 (BF16, F16 or F32 in the file), its first `cols` columns (0: all), in RAM
        auto rows16 = [&](const std::string& name, int64_t& N, int64_t& Kc, std::vector<uint16_t>& h) {
            const SafeTensor& t = find(name + ".weight");
            if (t.shape.size() != 2) throw std::runtime_error("image engine (Anima training): " + name + " is not 2D");
            N = t.shape[0];
            const int64_t K = t.shape[1];
            Kc = K;
            const uint8_t* src = file.data() + st.data_base() + t.begin;
            h.resize((size_t) (N * Kc));
            for (int64_t n = 0; n < N; ++n)
                for (int64_t k = 0; k < Kc; ++k) {
                    const size_t i = (size_t) (n * K + k);
                    float v;
                    if (t.dtype == "BF16") v = dit_rows::bf16_value(reinterpret_cast<const uint16_t*>(src)[i]);
                    else if (t.dtype == "F16") v = kernels::f32_from_f16(reinterpret_cast<const uint16_t*>(src)[i]);
                    else if (t.dtype == "F32") v = reinterpret_cast<const float*>(src)[i];
                    else throw std::runtime_error("image engine (Anima training): " + name + " is " + t.dtype);
                    h[(size_t) (n * Kc + k)] = kernels::f16_from_f32(v);
                }
        };
        // a matrix's rows as FP16 on the GPU (BF16, F16 or F32 in the file), its first `cols` columns (0: all)
        auto dense = [&](const std::string& name, int64_t cols = 0) {
            const SafeTensor& t = find(name + ".weight");
            if (t.shape.size() != 2) throw std::runtime_error("image engine (Anima training): " + name + " is not 2D");
            Impl::Dense d;
            d.N = t.shape[0];
            const int64_t K = t.shape[1];
            d.K = cols > 0 ? cols : K;
            const uint8_t* src = file.data() + st.data_base() + t.begin;
            std::vector<uint16_t> h((size_t) (d.N * d.K));
            for (int64_t n = 0; n < d.N; ++n)
                for (int64_t k = 0; k < d.K; ++k) {
                    const size_t i = (size_t) (n * K + k);
                    float v;
                    if (t.dtype == "BF16") v = dit_rows::bf16_value(reinterpret_cast<const uint16_t*>(src)[i]);
                    else if (t.dtype == "F16") v = kernels::f32_from_f16(reinterpret_cast<const uint16_t*>(src)[i]);
                    else if (t.dtype == "F32") v = reinterpret_cast<const float*>(src)[i];
                    else throw std::runtime_error("image engine (Anima training): " + name + " is " + t.dtype);
                    h[(size_t) (n * d.K + k)] = kernels::f16_from_f32(v);
                }
            d.w = m.get<uint16_t>(d.N * d.K);
            gpu::stream_sync(m.s);
            if (!gpu::copy(d.w, h.data(), h.size() * 2)) throw std::runtime_error("copy");
            return d;
        };
        m.lat = find("final_layer.linear.weight").shape[0] / 4;
        m.xemb = dense("x_embedder.proj.1", 4 * m.lat);   // the padding mask's columns left out (all zero)
        m.C = m.xemb.N;
        m.t1 = dense("t_embedder.1.linear_1");
        m.t2 = dense("t_embedder.1.linear_2");
        m.flin = dense("final_layer.linear");
        m.fm1 = dense("final_layer.adaln_modulation.1");
        m.fm2 = dense("final_layer.adaln_modulation.2");
        m.tnorm = st.f32(pre + "t_embedding_norm.weight");
        m.hd = find("blocks.0.self_attn.q_norm.weight").shape[0];
        m.H = m.C / m.hd;
        m.M = m.fm1.N;
        if (m.t2.N != 3 * m.C || m.fm2.N != 2 * m.C) throw std::runtime_error("image engine (Anima training): shape");
        std::mt19937_64 rng(seed);
        auto lora = [&](Lora& l, int64_t N, int64_t K) {
            l.N = N;
            l.K = K;
            std::vector<float> A;
            train::lora_init(rng, rank, K, A);
            l.A = m.dev_vec(A);
            l.B = m.get<float>(N * rank);
            l.mA = m.get<float>(rank * K);
            l.vA = m.get<float>(rank * K);
            l.mB = m.get<float>(N * rank);
            l.vB = m.get<float>(N * rank);
            l.ad.rank = rank;
            l.ad.scale = 1.0f;   // alpha = rank
            l.ad.a = m.get<uint16_t>(rank * K);
            l.ad.b = m.get<uint16_t>(N * rank);
            if (!m.gemm.f16_batch_ex_ok()) {
                l.ad.at = m.get<uint16_t>(rank * K);
                l.ad.bt = m.get<uint16_t>(N * rank);
            }
            l.ad.dA = m.get<float>(rank * K);
            l.ad.dB = m.get<float>(N * rank);
            m.sync_views(l);
        };
        auto row_lora = [&](RowLora& l, int64_t N, int64_t K) {
            l.N = N;
            l.K = K;
            std::vector<float> A;
            train::lora_init(rng, rank, K, A);
            l.A = m.dev_vec(A);
            l.B = m.get<float>(N * rank);
            l.dA = m.get<float>(rank * K);
            l.dB = m.get<float>(N * rank);
            l.mA = m.get<float>(rank * K);
            l.vA = m.get<float>(rank * K);
            l.mB = m.get<float>(N * rank);
            l.vB = m.get<float>(N * rank);
        };
        for (int b = 0;; ++b) {
            const std::string p = "blocks." + std::to_string(b) + ".";
            if (st.find(pre + p + "self_attn.q_proj.weight") == nullptr) break;
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
            B.sqn = m.dev_vec(st.f32(pre + p + "self_attn.q_norm.weight"));
            B.skn = m.dev_vec(st.f32(pre + p + "self_attn.k_norm.weight"));
            B.cqn = m.dev_vec(st.f32(pre + p + "cross_attn.q_norm.weight"));
            B.ckn = m.dev_vec(st.f32(pre + p + "cross_attn.k_norm.weight"));
            if (rank > 0) {
                for (int i = 0; i < kAnProducts; ++i) lora(B.l[i], B.p[i].N, B.p[i].K);
                for (int k = 0; k < 3; ++k) {
                    row_lora(B.m1[k], B.w1[k].N, B.w1[k].K);
                    row_lora(B.m2[k], B.w2[k].N, B.w2[k].K);
                }
            }
            m.blk.push_back(B);
        }
        if (m.blk.empty()) throw std::runtime_error("image engine (Anima training): the DiT has no blocks");
        m.F = m.blk[0].p[kAnL1].N;
        m.Cc = m.blk[0].p[kAnCk].K;
        for (int i = 0; i < kAnProducts; ++i) m.wt[i] = m.get<uint16_t>(m.blk[0].p[i].N * m.blk[0].p[i].K);
        gpu::stream_sync(stream);
        m.set_budget(0);
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
    m.set_budget(HW);
    // every block's input in FP32 (two passed between, the forward alone), or where the memory is short in BF16
    // (inputs_half): bounds the two FP32 passed between, the inputs in bounds16, the input rounded as it is kept
    const bool half = backward && m.inputs_half(HW);
    const bool all32 = backward && !half;
    float* bounds = pool.f((all32 ? nb + 1 : 2) * HW * C);
    uint16_t* bounds16 = half ? pool.u((nb + 1) * HW * C) : nullptr;
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
        const int64_t half = C / 2;
        std::vector<float> e((size_t) C);
        for (int64_t i = 0; i < half; ++i) {
            const float ex = std::exp(-std::log(10000.f) * (float) i / (float) half);
            const float ph = sigma * ex;
            e[(size_t) i] = dit_rows::round_bf16(std::cos(ph));
            e[(size_t) (half + i)] = dit_rows::round_bf16(std::sin(ph));
        }
        double ss = 0.0;
        for (const float v : e) ss += (double) v * v;
        const float inv = 1.f / std::sqrt((float) (ss / (double) C) + kEps);
        std::vector<uint16_t> h((size_t) (2 * C));
        for (int64_t i = 0; i < C; ++i) {
            h[(size_t) i] = kernels::f16_from_f32(e[(size_t) i]);
            const float y = dit_rows::round_bf16(e[(size_t) i] * inv * m.tnorm[(size_t) i]);
            h[(size_t) (C + i)] = kernels::f16_from_f32(y / (1.f + std::exp(-y)));
        }
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
                if (rk > 0) train::row_lora_fwd(B.m1[k].A, B.m1[k].B, x1, 1.f, rk, M, C, mh, hr, s);
                kernels::f32_to_f16_bulk(mh, m16, M, s);
                to_f32(m16, M, x2, s);
                m.gemm.f16(m16, B.w2[k].w, mod, 1, 3 * C, M);
                if (rk > 0) train::row_lora_fwd(B.m2[k].A, B.m2[k].B, x2, 1.f, rk, 3 * C, M, mod, hr, s);
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
            r.ad[i] = rk > 0 ? &B.l[i].ad : nullptr;
        }
        return r;
    };
    for (int64_t b = 0; b < nb; ++b) {
        // the next block's products fetched meanwhile; past the last, the backward's second when it follows
        m.bind(b, backward ? m.ws.begin((int) b, (int) (b + 1 < nb ? b + 1 : nb - 2)) : m.ws.begin((int) b));
        AnimaBlockRun r = run_of(b);
        if (half && b == 0) train::bf16_keep(bounds, HW * C, bounds16, s);
        r.x = all32 ? bounds + b * HW * C : bounds + (b % 2) * HW * C;
        r.out = all32 ? bounds + (b + 1) * HW * C : bounds + ((b + 1) % 2) * HW * C;
        anima_block(r, m.gemm, s, &m.arena);
        m.ws.end((int) b);
        if (half) train::bf16_keep(r.out, HW * C, bounds16 + (b + 1) * HW * C, s);
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
    // the blocks from the last, each one's modulations' gradients to their adapters
    float* dmod = pool.f(9 * C);
    float *g = pool.f(std::max<int64_t>(rk, 1)), *dx2 = pool.f(M);
    // the blocks' outputs in the backward (their forwards' again: not read) once for all; BF16: the forward's other
    // buffer, and the last one (the head's, done with below) the input read back into
    float* bout = half ? bounds + ((nb + 1) % 2) * HW * C : pool.f(HW * C);
    for (int64_t b = nb - 1; b >= 0; --b) {
        m.bind(b, m.ws.begin((int) b, (int) b - 1));   // the block below fetched meanwhile
        Impl::Block& B = m.blk[(size_t) b];
        AnimaBlockRun r = run_of(b);
        for (int i = 0; i < kAnProducts; ++i) {   // the dgrads' weights, transposed
            train::transpose_f16(B.p[i].w, B.p[i].N, B.p[i].K, m.wt[i], s);
            r.p[i].wt = m.wt[i];
        }
        if (half) train::bf16_load(bounds16 + b * HW * C, HW * C, last, s);
        r.x = half ? last : bounds + b * HW * C;
        r.out = bout;
        r.dout = dcur;
        r.dx = b > 0 ? dnxt : nullptr;
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
                const float* d = dmod + k * 3 * C;
                const float* x2 = x2s + (b * 3 + k) * M;
                train::row_lora_bwd(B.m2[k].A, B.m2[k].B, x2, d, 1.f, rk, 3 * C, M, B.m2[k].dA, B.m2[k].dB, hr, g, s);
                row_dgrad(B.w2[k].w, d, 3 * C, M, B.m2[k].A, g, rk, dx2, s);
                train::row_lora_bwd(B.m1[k].A, B.m1[k].B, x1, dx2, 1.f, rk, M, C, B.m1[k].dA, B.m1[k].dB, hr, g, s);
            }
        std::swap(dcur, dnxt);
    }
    gpu::stream_sync(s);
    m.scratch_rows = std::max(m.scratch_rows, HW);
    m.stepped = true;
    return loss;
}

double AnimaTrainer::adamw(const train::AdamW& o) {
    Impl& m = *m_;
    const int64_t rk = m.rank;
    std::vector<std::pair<const float*, int64_t>> parts;
    for (const Impl::Block& B : m.blk) {
        for (const Lora& l : B.l) {
            parts.emplace_back(l.ad.dA, rk * l.K);
            parts.emplace_back(l.ad.dB, l.N * rk);
        }
        for (int k = 0; k < 3; ++k)
            for (const RowLora* l : {&B.m1[k], &B.m2[k]}) {
                parts.emplace_back(l->dA, rk * l->K);
                parts.emplace_back(l->dB, l->N * rk);
            }
    }
    const double norm = std::sqrt(train::sq_sum(parts, m.s));
    const float gscale = (float) std::min(1.0, 1.0 / (norm + 1e-6));   // torch's clip_grad_norm_ at 1
    ++m.steps;
    for (Impl::Block& B : m.blk) {
        for (Lora& l : B.l) {
            train::adamw(l.A, l.ad.dA, l.mA, l.vA, rk * l.K, o, m.steps, gscale, m.s);
            train::adamw(l.B, l.ad.dB, l.mB, l.vB, l.N * rk, o, m.steps, gscale, m.s);
            m.sync_views(l);
        }
        for (int k = 0; k < 3; ++k)
            for (RowLora* l : {&B.m1[k], &B.m2[k]}) {
                train::adamw(l->A, l->dA, l->mA, l->vA, rk * l->K, o, m.steps, gscale, m.s);
                train::adamw(l->B, l->dB, l->mB, l->vB, l->N * rk, o, m.steps, gscale, m.s);
            }
    }
    gpu::stream_sync(m.s);
    return norm;
}

void AnimaTrainer::save(const std::string& path, const std::vector<std::pair<std::string, std::string>>& meta) const {
    const Impl& m = *m_;
    auto read = [&](const float* p, int64_t n) {
        std::vector<float> v((size_t) n);
        gpu::stream_sync(m.s);
        if (!gpu::copy(v.data(), p, (size_t) n * 4)) throw std::runtime_error("image engine: a read back");
        return v;
    };
    std::vector<train::Tensor> ts;
    for (size_t b = 0; b < m.blk.size(); ++b) {
        const Impl::Block& B = m.blk[b];
        const std::string p = "diffusion_model.blocks." + std::to_string(b) + ".";
        for (int i = 0; i < kAnProducts; ++i) {
            const Lora& l = B.l[i];
            ts.push_back({p + kProduct[i] + ".lora_A.weight", {m.rank, l.K}, read(l.A, m.rank * l.K)});
            ts.push_back({p + kProduct[i] + ".lora_B.weight", {l.N, m.rank}, read(l.B, l.N * m.rank)});
        }
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 2; ++j) {
                const RowLora& l = j == 0 ? B.m1[k] : B.m2[k];
                const std::string n = p + kMod[k] + (j == 0 ? ".1" : ".2");
                ts.push_back({n + ".lora_A.weight", {m.rank, l.K}, read(l.A, m.rank * l.K)});
                ts.push_back({n + ".lora_B.weight", {l.N, m.rank}, read(l.B, l.N * m.rank)});
            }
    }
    train::save_safetensors_bf16(path, std::move(ts), meta);
}

}  // namespace strata::image
