// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_trainer.cpp - see include/strata/image/qi_trainer.hpp.  The DiT's edges (the time embedding and the
// shared modulation, the text projection, img_in, the final norm and proj_out) as ai-toolkit's
// QwenImage21Transformer2DModel.forward takes them, around the shared training block (qi_block).
#include "strata/image/qi_trainer.hpp"
#include "strata/image/dit_file.hpp"
#include "strata/image/gelu.hpp"
#include "strata/image/lora_names.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/image/qi_train.hpp"
#include "strata/image/train_ops.hpp"

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
constexpr int kAxes[3] = {16, 56, 56};               // the rotation's frame, height and width pairs: 8 + 28 + 28
static_assert(std::size(lora::kQiBlock) == kQiProducts);
const char* const* const kProducts = lora::kQiBlock;   // the DiT's target names, in kQiProducts' order

// one product's LoRA: the FP32 factors and their moments; its train::Adapter holds the FP16 views and the gradients
struct Lora {
    train::Adapter ad;
    float *A = nullptr, *B = nullptr, *mA = nullptr, *vA = nullptr, *mB = nullptr, *vB = nullptr;
    int64_t N = 0, K = 0;
};

}  // namespace

struct QiTrainer::Impl {
    void* s = nullptr;
    std::vector<void*> owned;
    std::unique_ptr<QiFile> g;
    prefill::Gemm gemm;
    int64_t C = 0, F = 0, H = 0, ctx = 0, rank = 0;
    struct Block {
        train::Product p[kQiProducts];   // N, K (the weights' pointers are set as the block is used: weights_of)
        uint64_t off[kQiProducts] = {};  // each product's place in the block's region (wstream)
        float *nq = nullptr, *nk = nullptr;
        Lora l[kQiProducts];
    };
    // The blocks' weights through a LayerStream: as many as the memory leaves stay in VRAM, the rest are copied in
    // a block at a time from RAM (an int8 base's rows and scales, made once at open and kept in RAM; a 16-bit base's
    // from the file), the next one while this one computes.  A GPU with the memory keeps them all, as before.
    LayerStream wstream;
    std::vector<std::vector<uint8_t>> w8host;   // an int8 base: each block's rows and scales, the stream's source
    bool int8 = false;
    int64_t budget_rows = 0;      // the rows the weights' budget was set for (set_budget)
    bool budget_stepped = false;  // and whether a step had run then
    int64_t scratch_rows = 0;     // the most rows a step has had: the arena holds what they took
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
    uint16_t* wt[kQiProducts] = {};   // a block's weights transposed (the backward's dgrad reads them)
    uint16_t* w8tmp = nullptr;          // an int8 base's weight rotated back, before its transpose
    uint16_t* plain0[3] = {};           // an int8 base: block 0's q, k and v rotated back (QiBlockRun::w_plain)
    int64_t steps = 0, last_hw = 0;
    train::Arena arena;   // the blocks' tape and scratch, kept from block to block and step to step
    // the last blocks' tapes kept from the forward to the backward (no forward again for them): one buffer, cut into
    // this step's tapes, as large as the free memory takes; an arena on each piece
    std::vector<std::unique_ptr<train::Arena>> tapes;
    void* tape_buf = nullptr;
    size_t tape_cap = 0;
    bool stepped = false;   // a step ran (its kernels compiled, the shared scratch grown)

    template<class T> T* get(int64_t n) {
        void* p = nullptr;
        if (!gpu::alloc_device(&p, (size_t) n * sizeof(T)))
            throw std::runtime_error("image engine (Qwen-Image-2.1 training): GPU memory (" +
                                     std::to_string(((size_t) n * sizeof(T)) >> 20) + " MiB): " + gpu::last_error());
        owned.push_back(p);
        // zeroed on the step's stream and waited for: the weights and the host's values are copied in through the
        // compute queue (gpu::copy), which is not ordered against s, and a zero that ran late wiped parts of the
        // weights, a different loss each process (2026-11-09)
        gpu::memset_async(p, 0, (size_t) n * sizeof(T), s);
        gpu::stream_sync(s);
        return static_cast<T*>(p);
    }
    // How many of the last blocks keep their tapes this step: XESTRATA_TRAIN_TAPES, or as many as the device memory
    // takes.  The tapes' buffer is the free memory and what it holds now, less a reserve: the blocks' shared
    // scratch's growth (its arena reached 4.9 tapes at 2,300 and 4,000 rows of Qwen-Image-2.1, rank 32) and 1.5 GiB
    // of transients, 3 GiB more the first step (the kernels' first runs: free memory fell 2.7 GB more than the arenas
    // took on the B70, 2026-11-09).  The buffer is made again only when it should grow or must shrink.
    int64_t tapes_kept(const QiBlockRun& r) {
        const int64_t nb = (int64_t) blk.size();
        if (tapes.size() != (size_t) nb) tapes.resize((size_t) nb);
        const size_t per = (qi_block_tape_bytes(r) + 255) / 256 * 256;
        gpu::stream_sync(s);
        size_t want = 0;
        if (const char* e = strata::env_get("TRAIN_TAPES")) {
            want = (size_t) std::clamp<int64_t>(std::atoll(e), 0, nb) * per;
        } else {
            size_t free_b = 0, total = 0;
            if (gpu::mem_info(&free_b, &total)) {
                free_b += tape_cap;
                const size_t scratch = (size_t) (5.5 * (double) per);
                const size_t reserve = (scratch > arena.held() ? scratch - arena.held() : 0) + (size_t{3} << 29) +
                                       (stepped ? 0 : size_t{3} << 30);
                want = std::min(free_b > reserve ? free_b - reserve : 0, (size_t) nb * per);
            }
        }
        // grow it when this step takes more tapes than it holds, shrink it when it holds more than the memory allows
        if (want / per > tape_cap / per || want < tape_cap) {
            for (auto& t : tapes) t.reset();
            if (tape_buf != nullptr) gpu::free(tape_buf);
            tape_buf = nullptr;
            tape_cap = 0;
            if (want >= per && gpu::alloc_device(&tape_buf, want)) tape_cap = want;
        }
        const int64_t k = std::min<int64_t>(nb, (int64_t) (tape_cap / per));
        if (strata::env_get("TRAIN_PROFILE") != nullptr) {
            size_t fb = 0, tb = 0;
            gpu::mem_info(&fb, &tb);
            std::fprintf(stderr, "kept tapes %lld of %lld (%zu MiB each, %zu MiB free of %zu)\n", (long long) k,
                         (long long) nb, per >> 20, fb >> 20, tb >> 20);
        }
        for (int64_t b = 0; b < nb; ++b) {
            auto& t = tapes[(size_t) b];
            if (b >= nb - k) {
                if (!t) t = std::make_unique<train::Arena>();
                t->attach(static_cast<uint8_t*>(tape_buf) + (size_t) (b - (nb - k)) * per, per);
            } else {
                t.reset();
            }
        }
        return k;
    }
    void release(void* p) {
        if (p == nullptr) return;
        owned.erase(std::remove(owned.begin(), owned.end(), p), owned.end());
        gpu::free(p);
    }
    // a 16-bit weight [N][K] as the file holds it (`type`: F16 or bf16), on the GPU
    uint16_t* w16(const std::string& name, uint32_t type, int64_t& N, int64_t& K) {
        const TensorInfo& t = g->get(name);
        if (t.type != type)
            throw std::runtime_error("image engine (Qwen-Image-2.1 training): " + name + " is not " +
                                     (type == kF16 ? "F16" : "BF16") + " (a 16-bit DiT file is the training's base)");
        K = (int64_t) t.shape[0];
        N = t.shape.size() > 1 ? (int64_t) t.shape[1] : 1;
        uint16_t* d = get<uint16_t>(N * K);
        gpu::copy(d, g->data(t), (size_t) (N * K) * 2);
        return d;
    }
    float* dev_vec(const std::vector<float>& v) {
        float* d = get<float>((int64_t) v.size());
        gpu::copy(d, v.data(), v.size() * 4);
        return d;
    }
    // The VRAM the blocks' weights take, set when a step brings more rows than any before (its buffers taken: ensure):
    // the free memory and what the weights hold now, less the step's own beside them (the blocks' shared scratch's
    // growth: train::scratch_growth, before a step has shown it 13 FP32 rows of F a row, 2.5 GB at 4,035 rows of
    // Qwen-Image-2.1 with the MLP in passes of 1,024 rows, the B70, 2026-11-09; 1.5 GiB of transients, 3 GiB more
    // before the first step, as tapes_kept's) - all of them where that
    // takes them, else as many as it does and the rest streamed.  XESTRATA_TRAIN_WEIGHTS_MIB sets the budget.
    void set_budget(int64_t R) {
        // again after the first step (its reserve for the kernels' first runs given back)
        if (R <= budget_rows && budget_stepped == stepped) return;
        budget_rows = std::max(budget_rows, R);
        R = budget_rows;
        budget_stepped = stepped;
        const uint64_t all = wstream.total();
        uint64_t budget = all;
        if (const char* e = strata::env_get("TRAIN_WEIGHTS_MIB")) {
            budget = (uint64_t) std::atoll(e) << 20;
        } else {
            size_t free_b = 0, total = 0;
            gpu::stream_sync(s);
            if (gpu::mem_info(&free_b, &total)) {
                const uint64_t have = (uint64_t) free_b + wstream.vram();
                const uint64_t reserve = train::scratch_growth(arena.held(), R, scratch_rows,
                                                               (uint64_t) (13 * R * F) * 4) +
                                         (uint64_t{3} << 29) +
                                         (stepped ? 0 : uint64_t{3} << 30);
                budget = have > reserve ? have - reserve : 0;
            }
        }
        budget = std::min(all, std::max(budget, wstream.min_budget()));
        if (budget < all)
            std::fprintf(stderr, "image engine (Qwen-Image-2.1 training): the blocks' weights %llu of %llu MiB in "
                                 "VRAM, the rest streamed from RAM\n",
                         (unsigned long long) (budget >> 20), (unsigned long long) (all >> 20));
        wstream.configure(budget, s);
    }
    // the products' weights of block b in its region `base` (wstream.begin's)
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
    // Whether the blocks' inputs go in BF16 (XESTRATA_TRAIN_INPUTS16=0 or 1, or by the memory, decided as the
    // buffers are taken for more rows): where the free memory and the weights' VRAM fall short of the inputs in FP32,
    // every block's weights resident, the blocks' scratch (13 FP32 rows of F a row) and 1.5 GiB (3 GiB more before
    // the first step) - the inputs halved before any weight is streamed, which costs every step its copies.  The
    // forward rounds each input to BF16 as it keeps it, so the backward reads what the forward used.
    bool inputs_half(int64_t R) {
        if (const char* e = strata::env_get("TRAIN_INPUTS16")) return std::atoi(e) != 0;
        size_t free_b = 0, total = 0;
        gpu::stream_sync(s);
        if (!gpu::mem_info(&free_b, &total)) return false;
        const uint64_t have = (uint64_t) free_b + wstream.vram();
        const uint64_t need = (uint64_t) ((int64_t) (blk.size() + 1) * R * C) * 4 + wstream.total() +
                              (uint64_t) (13 * R * F) * 4 + (uint64_t{3} << 29) + (stepped ? 0 : uint64_t{3} << 30);
        return have < need;
    }
    void ensure(int64_t R, int64_t hw) {
        if (R <= cap) return;
        for (float* p : ckpt) release(p);
        release(ckpt16);
        ckpt16 = nullptr;
        for (void* p : {(void*) tmp_out, (void*) dX, (void*) dX2, (void*) x0, (void*) nz, (void*) pred, (void*) dpred,
                        (void*) dln, (void*) scratch, (void*) ln_rs, (void*) mean, (void*) inv, (void*) drs,
                        (void*) cos, (void*) sin, (void*) xt16, (void*) ln16, (void*) d16})
            release(p);
        if (inputs_half(R)) {
            ckpt.assign(2, nullptr);
            ckpt16 = get<uint16_t>((int64_t) (blk.size() + 1) * R * C);
            std::fprintf(stderr, "image engine (Qwen-Image-2.1 training): the blocks' inputs in BF16 (%lld rows)\n",
                         (long long) R);
        } else {
            ckpt.assign(blk.size() + 1, nullptr);
        }
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
        (void) hw;
    }
    // the FP16 views the products read, from the FP32 factors
    void sync_views(Lora& l) {
        const int64_t r = rank;
        train::f16_copy(l.A, r * l.K, l.ad.a, s);
        if (l.ad.at != nullptr) train::transpose_f16(l.ad.a, r, l.K, l.ad.at, s);
        train::f16_copy(l.B, l.N * r, l.ad.b, s);
        if (l.ad.bt != nullptr) train::transpose_f16(l.ad.b, l.N, r, l.ad.bt, s);
    }
    // ai-toolkit's QwenImage21Rope: a text row's three axes its position, the image's frame axis the text's length
    // and its height and width centred on zero; each axis's pairs at 1 / 10000^(2j / dim) (FP32, as torch.outer
    // reads the long positions)
    void rope(int64_t nt, int64_t h, int64_t w) {
        const auto key = std::make_tuple(nt, h, w);
        if (key == rope_key) return;
        const int64_t R = nt + h * w;
        std::vector<float> c((size_t) (R * kHead / 2)), sn(c.size());
        float inv_f[3][28];
        for (int a = 0; a < 3; ++a)
            for (int j = 0; j < kAxes[a] / 2; ++j)
                inv_f[a][j] = 1.0f / std::pow(10000.0f, (float) (2 * j) / (float) kAxes[a]);
        for (int64_t r = 0; r < R; ++r) {
            int64_t pos[3];
            if (r < nt) {
                pos[0] = pos[1] = pos[2] = r;
            } else {
                const int64_t q = r - nt, y = q / w, x = q % w;
                pos[0] = nt;
                pos[1] = y - (h - h / 2);
                pos[2] = x - (w - w / 2);
            }
            int64_t col = 0;
            for (int a = 0; a < 3; ++a)
                for (int j = 0; j < kAxes[a] / 2; ++j, ++col) {
                    const float ang = (float) pos[a] * inv_f[a][j];
                    c[(size_t) (r * kHead / 2 + col)] = std::cos(ang);
                    sn[(size_t) (r * kHead / 2 + col)] = std::sin(ang);
                }
        }
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
        std::vector<uint16_t> e(2 * kTimeDim);
        for (int i = 0; i < kTimeDim / 2; ++i) {
            const float f = std::exp(-std::log(10000.0f) * (float) i / (float) (kTimeDim / 2));
            e[(size_t) i] = kernels::f16_from_f32(std::cos(t * f));
            e[(size_t) (i + kTimeDim / 2)] = kernels::f16_from_f32(std::sin(t * f));
            e[(size_t) (kTimeDim + i)] = kernels::f16_from_f32(1.0f);
            e[(size_t) (kTimeDim + i + kTimeDim / 2)] = kernels::f16_from_f32(0.0f);
        }
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

QiTrainer::~QiTrainer() {
    if (m_->s != nullptr) gpu::stream_sync(m_->s);
    m_->wstream.drop();
    m_->tapes.clear();
    if (m_->tape_buf != nullptr) gpu::free(m_->tape_buf);
    for (void* p : m_->owned) gpu::free(p);
}

int64_t QiTrainer::blocks() const { return (int64_t) m_->blk.size(); }
int64_t QiTrainer::width() const { return m_->C; }

bool QiTrainer::open(const std::string& dit, int64_t rank, uint64_t seed, void* stream, std::string& err,
                     int64_t max_blocks, bool int8) {
    Impl& m = *m_;
    try {
        m.s = stream;
        m.rank = rank;
        m.g = std::make_unique<QiFile>(dit);
        if (!m.gemm.init(stream, 1 << 20, err)) return false;   // before the adapters: their views depend on it
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
                m.w8host.emplace_back((size_t) hbytes);
                uint8_t* h = m.w8host.back().data();
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
            const int layer = m.wstream.add(spans);
            for (int i = 0; i < kQiProducts; ++i) blk.off[i] = m.wstream.offset(layer, i);
            for (int i = 0; i < kQiProducts; ++i) {
                train::Product& pr = blk.p[i];
                Lora& l = blk.l[i];
                l.N = pr.N;
                l.K = pr.K;
                std::vector<float> A;
                train::lora_init(rng, rank, l.K, A);
                l.A = m.dev_vec(A);
                l.B = m.get<float>(l.N * rank);
                l.mA = m.get<float>(rank * l.K);
                l.vA = m.get<float>(rank * l.K);
                l.mB = m.get<float>(l.N * rank);
                l.vB = m.get<float>(l.N * rank);
                l.ad.rank = rank;
                l.ad.scale = 1.0f;   // alpha = rank
                l.ad.a = m.get<uint16_t>(rank * l.K);
                l.ad.b = m.get<uint16_t>(l.N * rank);
                if (!m.gemm.f16_batch_ex_ok()) {   // the transposed views, where the products cannot read the rows
                    l.ad.at = m.get<uint16_t>(rank * l.K);
                    l.ad.bt = m.get<uint16_t>(l.N * rank);
                }
                l.ad.dA = m.get<float>(rank * l.K);
                l.ad.dB = m.get<float>(l.N * rank);
                m.sync_views(l);
            }
            blk.nq = m.dev_vec(m.g->f32(p + "attn.norm_q.weight"));
            blk.nk = m.dev_vec(m.g->f32(p + "attn.norm_k.weight"));
            m.blk.push_back(blk);
        }
        for (void* x : {(void*) q16, (void*) q8, (void*) qs}) m.release(x);
        if (m.blk.empty()) throw std::runtime_error("image engine (Qwen-Image-2.1 training): the DiT has no blocks");
        m.F = m.blk[0].p[kQiGateUp].N / 2;
        m.H = m.C / kHead;
        int64_t most = 0;
        for (int i = 0; i < kQiProducts; ++i) {
            m.wt[i] = m.get<uint16_t>(m.blk[0].p[i].N * m.blk[0].p[i].K);
            most = std::max(most, m.blk[0].p[i].N * m.blk[0].p[i].K);
        }
        if (int8) m.w8tmp = m.get<uint16_t>(most);
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

float QiTrainer::step(int text, const float* x0, const float* noise, int64_t h, int64_t w, float timestep) {
    Impl& m = *m_;
    void* s = m.s;
    const int64_t C = m.C, hw = h * w, nt = m.texts.at((size_t) text).second, R = nt + hw;
    m.ensure(R, hw);
    m.set_budget(R);
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
            r.ad[i] = &B.l[i].ad;
        }
        r.nq = B.nq;
        r.nk = B.nk;
        if (b == 0)
            for (int i = 0; i < 3; ++i) r.w_plain[i] = m.plain0[i];
        r.x = m.ckpt16 != nullptr ? m.ckpt[(size_t) (b % 2)] : m.ckpt[(size_t) b];
        return r;
    };
    const bool half = m.ckpt16 != nullptr;
    const int64_t cap = m.cap;
    const int64_t nb = (int64_t) m.blk.size();
    const int64_t first_kept = nb - m.tapes_kept(run_of(0));
    for (int64_t b = 0; b < nb; ++b) {   // the forward, each block's input kept, the last ones' tapes too
        QiBlockRun r = run_of(b);
        // the next block's weights fetched meanwhile; past the last, the backward's second (its first is this one)
        m.weights_of(b, m.wstream.begin((int) b, (int) (b + 1 < nb ? b + 1 : nb - 2)), r.p);
        if (half && b == 0) train::bf16_keep(m.ckpt[0], R * C, m.ckpt16, s);
        r.out = half ? m.ckpt[(size_t) ((b + 1) % 2)] : m.ckpt[(size_t) b + 1];
        if (b >= first_kept) r.tape = m.tapes[(size_t) b].get();
        qi_block(r, m.gemm, s, &m.arena);
        m.wstream.end((int) b);
        if (half) train::bf16_keep(r.out, R * C, m.ckpt16 + (b + 1) * cap * C, s);
    }
    // the final norm (the image's rows, the t row's scale) and proj_out
    const float* xf = (half ? m.ckpt[(size_t) (nb % 2)] : m.ckpt[(size_t) nb]) + nt * C;
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
    // the dgrads read the weights' rows as they are where the GPU takes transposed products (no transposed copy each
    // block each step)
    const bool rows_ok = m.gemm.f16_batch_ex_ok();
    for (int64_t b = nb - 1; b >= 0; --b) {
        QiBlockRun r = run_of(b);
        if (half) {   // its input read back into the buffer the head did not read
            float* xin = m.ckpt[(size_t) ((nb + 1) % 2)];
            train::bf16_load(m.ckpt16 + b * cap * C, R * C, xin, s);
            r.x = xin;
        }
        m.weights_of(b, m.wstream.begin((int) b, (int) b - 1), r.p);   // the block below fetched meanwhile
        for (int i = 0; i < kQiProducts; ++i) {   // the dgrads' weights (an int8 base's straight through)
            const train::Product pr = r.p[i];
            if (rows_ok) {   // read as they are: the 16-bit weight itself, an int8 base's rotated back
                if (pr.w8 != nullptr) train::w8_dequant_rows(pr.w8, pr.sw, pr.N, pr.K, m.wt[i], s);
                r.p[i].wr = pr.w8 != nullptr ? m.wt[i] : pr.w;
                r.p[i].wt = nullptr;
                continue;
            }
            if (pr.w8 != nullptr) train::w8_dequant_t(pr.w8, pr.sw, pr.N, pr.K, m.w8tmp, m.wt[i], s);
            else train::transpose_f16(pr.w, pr.N, pr.K, m.wt[i], s);
            r.p[i].wt = m.wt[i];
        }
        r.out = m.tmp_out;
        if (b >= first_kept) {   // its forward's tape: the backward alone
            r.tape = m.tapes[(size_t) b].get();
            r.tape_replay = true;
        }
        r.dout = m.dX;
        r.dx = b > 0 ? m.dX2 : nullptr;
        qi_block(r, m.gemm, s, &m.arena);
        m.wstream.end((int) b);
        std::swap(m.dX, m.dX2);
    }
    gpu::stream_sync(s);
    qi_block_profile_report();
    m.scratch_rows = std::max(m.scratch_rows, R);
    m.stepped = true;
    if (strata::env_get("TRAIN_PROFILE") != nullptr) {   // the memory the step left
        size_t fb = 0, tb = 0, held = 0;
        gpu::mem_info(&fb, &tb);
        held = m.tape_cap;
        std::fprintf(stderr, "end: free %zu MiB, tapes %zu MiB, main arena %zu MiB, R %lld\n", fb >> 20, held >> 20,
                     m.arena.held() >> 20, (long long) R);
    }
    return loss;
}

double QiTrainer::update(const train::AdamW& o) {
    Impl& m = *m_;
    std::vector<std::pair<const float*, int64_t>> parts;
    for (Impl::Block& B : m.blk)
        for (Lora& l : B.l) {
            parts.emplace_back(l.ad.dA, m.rank * l.K);
            parts.emplace_back(l.ad.dB, l.N * m.rank);
        }
    const double norm = std::sqrt(train::sq_sum(parts, m.s));
    const float gscale = (float) std::min(1.0, 1.0 / (norm + 1e-6));   // torch's clip_grad_norm_ at 1
    ++m.steps;
    for (Impl::Block& B : m.blk)
        for (Lora& l : B.l) {
            train::adamw(l.A, l.ad.dA, l.mA, l.vA, m.rank * l.K, o, m.steps, gscale, m.s);
            train::adamw(l.B, l.ad.dB, l.mB, l.vB, l.N * m.rank, o, m.steps, gscale, m.s);
            m.sync_views(l);
        }
    gpu::stream_sync(m.s);
    return norm;
}

namespace {

std::vector<float> read_dev(const float* p, int64_t n, void* s) {
    std::vector<float> v((size_t) n);
    gpu::stream_sync(s);
    if (!gpu::copy(v.data(), p, (size_t) n * 4)) throw std::runtime_error("image engine: a read back");
    return v;
}

}  // namespace

std::vector<train::Tensor> QiTrainer::grads() const {
    const Impl& m = *m_;
    std::vector<train::Tensor> out;
    for (size_t b = 0; b < m.blk.size(); ++b)
        for (int i = 0; i < kQiProducts; ++i) {
            const Lora& l = m.blk[b].l[i];
            const std::string k = "diffusion_model.transformer_blocks." + std::to_string(b) + "." + kProducts[i];
            out.push_back({k + ".lora_A.weight", {m.rank, l.K}, read_dev(l.ad.dA, m.rank * l.K, m.s)});
            out.push_back({k + ".lora_B.weight", {l.N, m.rank}, read_dev(l.ad.dB, l.N * m.rank, m.s)});
        }
    return out;
}

void QiTrainer::save(const std::string& path, const std::vector<std::pair<std::string, std::string>>& meta) const {
    const Impl& m = *m_;
    std::vector<train::Tensor> ts;
    for (size_t b = 0; b < m.blk.size(); ++b)
        for (int i = 0; i < kQiProducts; ++i) {
            const Lora& l = m.blk[b].l[i];
            const std::string k = "diffusion_model.transformer_blocks." + std::to_string(b) + "." + kProducts[i];
            ts.push_back({k + ".lora_A.weight", {m.rank, l.K}, read_dev(l.A, m.rank * l.K, m.s)});
            ts.push_back({k + ".lora_B.weight", {l.N, m.rank}, read_dev(l.B, l.N * m.rank, m.s)});
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
            Lora& l = m.blk[b].l[i];
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
            m.sync_views(l);
        }
    gpu::stream_sync(m.s);
}

std::vector<float> QiTrainer::last_pred() const { return read_dev(m_->pred, m_->last_hw * kInCh, m_->s); }

train::Schedule qi_train_schedule(int64_t h, int64_t w) {
    return train::schedule_dynamic(h * w, 256.0, 8192.0, 0.5, 0.9, 0.02);
}

}  // namespace strata::image
