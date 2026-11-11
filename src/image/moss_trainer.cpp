// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_trainer.cpp - see include/strata/image/moss_trainer.hpp.  The parts outside the blocks are the
// inference's (MossDit::forward, time_embedding, rope_tables and the head), frozen: the input projection, the time's
// two layers, the shared modulation and the head's, the final norm and projection.
#include "strata/image/moss_trainer.hpp"
#include "strata/image/moss_dit.hpp"
#include "strata/image/moss_train.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/image/safetensors.hpp"
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
const char* const kProduct[kMossProducts] = {"proj.qkv", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"};

struct Lora {
    train::Adapter ad;
    float *A = nullptr, *B = nullptr, *mA = nullptr, *vA = nullptr, *mB = nullptr, *vB = nullptr;
    int64_t N = 0, K = 0;
};

// a dense product outside the blocks: FP16 rows, its bias (null: none)
struct Dense {
    uint16_t* w = nullptr;
    float* b = nullptr;
    int64_t N = 0, K = 0;
};

}  // namespace

struct MossTrainer::Impl {
    void* s = nullptr;
    std::vector<void*> owned;
    prefill::Gemm gemm;
    int64_t C = 0, F = 0, H = 0, hd = 0, in_ch = 0, rank = 0, steps = 0;
    double rope_base = 10000.0;
    Dense vin, t1, t2, smod, hmod, fout;
    float* fnorm = nullptr;
    struct Block {
        train::Product p[kMossProducts];
        float *n1 = nullptr, *n2 = nullptr, *nq = nullptr, *nk = nullptr;
        Lora l[kMossProducts];
        bool adapted = false;
    };
    std::vector<Block> blk;
    uint16_t* wt[kMossProducts] = {};   // a block's dgrad views, made again for each block of the backward
    uint16_t* w8tmp = nullptr;
    uint16_t* plain0 = nullptr;
    train::Arena arena;   // the blocks' buffers, kept from block to block and step to step
    // the last blocks' tapes (their forwards' values, read back by their backwards instead of a forward again), in
    // one buffer of the memory the step leaves
    std::vector<std::unique_ptr<train::Arena>> tapes;
    void* tape_buf = nullptr;
    size_t tape_cap = 0;
    // The blocks' products through a LayerStream (Qwen-Image-2.1's trainer's): as many blocks as the memory leaves
    // stay in VRAM, the rest are copied in from RAM a block at a time; their rows as the products read them (FP16, or
    // an int8 base's rows and scales) made once at open and kept in RAM, blk's pointers set as a block is used (bind)
    LayerStream ws;
    std::vector<std::vector<uint8_t>> whost;
    std::vector<std::array<uint64_t, kMossProducts>> woff;
    bool int8 = false, stepped = false, budget_stepped = false;
    int64_t budget_rows = 0;
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
                const uint64_t have = (uint64_t) free_b + ws.vram() + tape_cap;
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
            std::fprintf(stderr, "image engine (MOSS-Image training): the blocks' weights %llu of %llu MiB in VRAM, "
                                 "the rest streamed from RAM\n",
                         (unsigned long long) (budget >> 20), (unsigned long long) (all >> 20));
        if (budget < ws.vram() + (all > budget ? ws.min_budget() : 0) && tape_cap > 0) {   // the tapes' room first
            for (auto& t : tapes) t.reset();
            gpu::free(tape_buf);
            tape_buf = nullptr;
            tape_cap = 0;
        }
        ws.configure(budget, s);
    }
    int64_t scratch_rows = 0;   // the most rows a step has had: the scratch arena holds what they took         // block 0's qkv rotated back (an int8 base's unquantized input)
    // the step's device rows
    uint16_t *e16 = nullptr, *t16 = nullptr;
    float *temb = nullptr, *mod = nullptr, *head = nullptr;

    template <class T> T* get(int64_t n) {
        void* q = nullptr;
        if (!gpu::alloc_device(&q, (size_t) n * sizeof(T)) || !gpu::memset_async(q, 0, (size_t) n * sizeof(T), s))
            throw std::runtime_error("image engine (MOSS-Image training): GPU memory");
        owned.push_back(q);
        return static_cast<T*>(q);
    }
    void release(void* q) {
        gpu::stream_sync(s);
        owned.erase(std::remove(owned.begin(), owned.end(), q), owned.end());
        gpu::free(q);
    }
    float* dev_vec(const std::vector<float>& v) {
        float* d = get<float>((int64_t) v.size());
        gpu::stream_sync(s);   // the zero above runs on this stream, the copy on the compute queue
        if (!gpu::copy(d, v.data(), v.size() * 4)) throw std::runtime_error("image engine (MOSS-Image training): copy");
        return d;
    }
    ~Impl() {
        if (s != nullptr) gpu::stream_sync(s);
        ws.drop();
        tapes.clear();
        if (tape_buf != nullptr) gpu::free(tape_buf);
        for (void* q : owned) gpu::free(q);
    }
    // How many of the last blocks keep their tapes this step: XESTRATA_TRAIN_TAPES, or as many as the device memory
    // takes once a step of as many rows or more has shown the scratch's size (none the first step, nor a step of more
    // rows than any before: a guess of the scratch's growth ran the RX 9060 XT out of memory): the free memory and the tapes' own
    // buffer, less the scratch's growth to this step's rows (what it holds for the most rows so far, in proportion,
    // 10% over), what the step takes of its own (step_bytes: decided before the step takes it, so a smaller step's
    // tapes are let go first) and 1.5 GiB of transients.
    // Whether this step holds its blocks' inputs in BF16 (XESTRATA_TRAIN_INPUTS16=0 or 1, or by the memory): when the
    // free memory and the tapes' buffer fall short of their FP32 bytes, the scratch's growth to this step's rows (in
    // proportion to the most rows so far, 10% over; a block's tape's bytes before any step has shown it) and 1.5 GiB
    // of transients.  The forward then rounds each input to BF16 as it keeps it, so the backward reads what the forward
    // used (the inference keeps its residual in BF16 too); a device with the memory keeps FP32 and its numbers.
    bool inputs_half(const MossBlockRun& r, size_t f32_bytes) {
        if (const char* e = strata::env_get("TRAIN_INPUTS16")) return std::atoi(e) != 0;
        size_t free_b = 0, total = 0;
        gpu::stream_sync(s);
        if (!gpu::mem_info(&free_b, &total)) return false;
        free_b += tape_cap;
        const size_t scratch =
            scratch_rows > 0
                ? (size_t) (1.1 * (double) arena.held() * (double) std::max(r.rows, scratch_rows) / (double) scratch_rows)
                : moss_block_tape_bytes(r, gemm);
        const size_t need = f32_bytes + (scratch > arena.held() ? scratch - arena.held() : 0) + (size_t{3} << 29);
        return free_b < need;
    }
    int64_t tapes_kept(const MossBlockRun& r, size_t step_bytes) {
        const int64_t nb = (int64_t) blk.size();
        if (tapes.size() != (size_t) nb) tapes.resize((size_t) nb);
        const size_t per = (moss_block_tape_bytes(r, gemm) + 255) / 256 * 256;
        gpu::stream_sync(s);
        size_t avail = 0;   // what the tapes may take this step
        if (const char* e = strata::env_get("TRAIN_TAPES")) {
            avail = (size_t) std::clamp<int64_t>(std::atoll(e), 0, nb) * per;
        } else if (scratch_rows > 0 && r.rows <= scratch_rows) {
            size_t free_b = 0, total = 0;
            if (gpu::mem_info(&free_b, &total)) {
                free_b += tape_cap;
                const size_t scratch =
                    (size_t) (1.1 * (double) arena.held() * (double) r.rows / (double) scratch_rows);
                const size_t reserve =
                    (scratch > arena.held() ? scratch - arena.held() : 0) + step_bytes + (size_t{3} << 29);
                avail = free_b > reserve ? free_b - reserve : 0;
            }
        }
        // grown when this step keeps more tapes than it holds, shrunk only when the memory does not allow it (a
        // smaller bucket's tapes fit in it as they are: no buffer of gigabytes made again for them)
        const size_t want = std::min(avail, (size_t) nb * per);
        if (want / per > tape_cap / per || avail < tape_cap) {
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
    void sync_views(Lora& l) {
        train::f16_copy(l.A, rank * l.K, l.ad.a, s);
        if (l.ad.at != nullptr) train::transpose_f16(l.ad.a, rank, l.K, l.ad.at, s);
        train::f16_copy(l.B, l.N * rank, l.ad.b, s);
        if (l.ad.bt != nullptr) train::transpose_f16(l.ad.b, l.N, rank, l.ad.bt, s);
    }
};

namespace {

// the DiT's shards: a name looked up in each
struct Shards {
    std::vector<std::unique_ptr<SafeTensors>> st;
    std::vector<std::unique_ptr<platform::MappedFile>> mf;
    explicit Shards(const std::string& path) {
        std::vector<std::string> files;
        if (std::filesystem::is_directory(path)) {
            for (const auto& e : std::filesystem::directory_iterator(path))
                if (e.path().extension() == ".safetensors") files.push_back(e.path().string());
            std::sort(files.begin(), files.end());
        } else {
            files.push_back(path);
        }
        if (files.empty()) throw std::runtime_error(path + ": no safetensors");
        for (const std::string& f : files) {
            st.push_back(std::make_unique<SafeTensors>(f));
            mf.push_back(std::make_unique<platform::MappedFile>(f));
        }
    }
    int at(const std::string& name) const {
        for (size_t i = 0; i < st.size(); ++i)
            if (st[i]->find(name) != nullptr) return (int) i;
        return -1;
    }
    bool has(const std::string& name) const { return at(name) >= 0; }
    const SafeTensor& bf16(const std::string& name) const {
        const int i = at(name);
        if (i < 0) throw std::runtime_error("image engine (MOSS-Image training): no " + name);
        const SafeTensor& t = *st[(size_t) i]->find(name);
        if (t.dtype != "BF16") throw std::runtime_error("image engine (MOSS-Image training): " + name + " is not bf16");
        return t;
    }
    // a bf16 matrix's rows as FP16 bits (host), appended to out
    void rows16(const std::string& name, std::vector<uint16_t>& out, int64_t& N, int64_t& K) const {
        const SafeTensor& t = bf16(name);
        if (t.shape.size() != 2) throw std::runtime_error("image engine (MOSS-Image training): " + name + " is not 2D");
        N = t.shape[0];
        K = t.shape[1];
        const int i = at(name);
        const auto* src = reinterpret_cast<const uint16_t*>(mf[(size_t) i]->data() + st[(size_t) i]->data_base() +
                                                            t.begin);
        const size_t o = out.size();
        out.resize(o + (size_t) (N * K));
        for (int64_t k = 0; k < N * K; ++k) out[o + (size_t) k] = kernels::f16_from_f32(dit_rows::bf16_value(src[k]));
    }
    std::vector<float> f32(const std::string& name) const {
        const int i = at(name);
        if (i < 0) throw std::runtime_error("image engine (MOSS-Image training): no " + name);
        return st[(size_t) i]->f32(name);
    }
};

}  // namespace

MossTrainer::MossTrainer() : m_(std::make_unique<Impl>()) {}
MossTrainer::~MossTrainer() = default;
int64_t MossTrainer::hidden() const { return m_->C; }

bool MossTrainer::open(const std::string& dit, int64_t rank, bool int8, const std::vector<int>& skip, uint64_t seed,
                       void* stream, std::string& err) {
    Impl& m = *m_;
    try {
        m.s = stream;
        m.rank = rank;
        if (!m.gemm.init(stream, 1 << 20, err)) return false;
        const Shards f(dit);
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
            d.w = m.get<uint16_t>(d.N * d.K);
            gpu::stream_sync(m.s);
            if (!gpu::copy(d.w, h.data(), h.size() * 2)) throw std::runtime_error("copy");
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
            blk.adapted = rank > 0 && std::find(skip.begin(), skip.end(), b) == skip.end();
            if (blk.adapted)
                for (int i = 0; i < kMossProducts; ++i) {
                    Lora& l = blk.l[i];
                    l.N = blk.p[i].N;
                    l.K = blk.p[i].K;
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
            m.blk.push_back(std::move(blk));
        }
        if (m.blk.empty()) throw std::runtime_error("image engine (MOSS-Image training): the DiT has no blocks");
        m.hd = (int64_t) f.f32("layers.0.q_norm.weight").size();
        m.H = m.C / m.hd;
        m.F = m.blk[0].p[kMossUpGate].N / 2;
        int64_t most = 0;
        for (int i = 0; i < kMossProducts; ++i) {
            m.wt[i] = m.get<uint16_t>(m.blk[0].p[i].N * m.blk[0].p[i].K);
            most = std::max(most, m.blk[0].p[i].N * m.blk[0].p[i].K);
        }
        for (void* x : {(void*) q16, (void*) q8, (void*) qs}) m.release(x);
        if (int8) m.w8tmp = m.get<uint16_t>(most);
        m.set_budget(0);
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
                        float sigma, bool backward) {
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
    bool half = false;
    m.set_budget(R);
    if (backward) {
        MossBlockRun z;
        z.C = C;
        z.heads = m.H;
        z.hd = m.hd;
        z.F = m.F;
        z.rows = R;
        for (int i = 0; i < kMossProducts; ++i) {
            z.p[i] = m.blk.back().p[i];
            z.ad[i] = m.blk.back().adapted ? &m.blk.back().l[i].ad : nullptr;
        }
        const size_t rc = (size_t) (R * C);
        half = m.inputs_half(z, (size_t) (nb + 1 + 3) * rc * 4);
        first_kept =
            nb - m.tapes_kept(z, half ? (size_t) (nb + 1) * rc * 2 + 4 * rc * 4 : (size_t) (nb + 1 + 3) * rc * 4);
    }
    // FP32: every block's input in bounds (two of them passed between, the forward alone); BF16: bounds the two
    // passed between, the inputs in bounds16
    const bool all32 = backward && !half;
    float* bounds = pool.f((all32 ? nb + 1 : 2) * R * C);
    uint16_t* bounds16 = half ? pool.u((nb + 1) * R * C) : nullptr;
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
        const int64_t half = kFreq / 2;
        const double t = (double) sigma * 1000.0;
        std::vector<uint16_t> e((size_t) kFreq);
        for (int64_t i = 0; i < half; ++i) {
            const double ph = t * std::pow(10000.0, -(double) i / (double) half);
            e[(size_t) i] = kernels::f16_from_f32(dit_rows::round_bf16((float) std::cos(ph)));
            e[(size_t) (half + i)] = kernels::f16_from_f32(dit_rows::round_bf16((float) std::sin(ph)));
        }
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
            r.ad[i] = B.adapted ? &B.l[i].ad : nullptr;
        }
        if (b == 0) r.w_plain_qkv = m.plain0;
        return r;
    };
    tm.mark("inputs, time, rope", s);
    // forward: every block's input kept for the backward, the last blocks' tapes as the memory takes them
    for (int64_t b = 0; b < nb; ++b) {
        // the next block's products fetched meanwhile; past the last, the backward's second when it follows
        m.bind(b, backward ? m.ws.begin((int) b, (int) (b + 1 < nb ? b + 1 : nb - 2)) : m.ws.begin((int) b));
        MossBlockRun r = run_of(b);
        if (b >= first_kept) r.tape = m.tapes[(size_t) b].get();
        if (half && b == 0) train::bf16_keep(bounds, R * C, bounds16, s);
        r.x = all32 ? bounds + b * R * C : bounds + (b % 2) * R * C;
        r.out = all32 ? bounds + (b + 1) * R * C : bounds + ((b + 1) % 2) * R * C;
        moss_block(r, m.gemm, s, &m.arena);
        m.ws.end((int) b);
        if (half) train::bf16_keep(r.out, R * C, bounds16 + (b + 1) * R * C, s);
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
    const float loss = train::flow_loss(v, dx0, deps, HW * IC, dv, nullptr, nullptr, s);
    if (!backward) return loss;
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
    // last one (the head's, done with) the input read back into
    float* out = half ? bounds + ((nb + 1) % 2) * R * C : pool.f(R * C);
    float* dcur = pool.f(R * C);   // the text rows' gradient from the head: none
    float* dnxt = pool.f(R * C);
    dit_rows::rms_affine_bwd(last, inv, m.fnorm, m.head + C, dln, HW, C, dcur, s);
    tm.mark("head backward", s);
    for (int64_t b = nb - 1; b >= 0; --b) {
        m.bind(b, m.ws.begin((int) b, (int) b - 1));   // the block below fetched meanwhile
        const Impl::Block& B = m.blk[(size_t) b];
        MossBlockRun r = run_of(b);
        for (int i = 0; i < kMossProducts; ++i) {   // the dgrads' weights (an int8 base's straight through)
            const train::Product& pr = B.p[i];
            if (pr.w8 != nullptr) train::w8_dequant_t(pr.w8, pr.sw, pr.N, pr.K, m.w8tmp, m.wt[i], s);
            else train::transpose_f16(pr.w, pr.N, pr.K, m.wt[i], s);
            r.p[i].wt = m.wt[i];
        }
        if (half) train::bf16_load(bounds16 + b * R * C, R * C, last, s);
        r.x = half ? last : bounds + b * R * C;
        r.out = out;
        r.dout = dcur;
        r.dx = b > 0 ? dnxt : nullptr;   // nothing below block 0 takes a gradient
        if (b >= first_kept) {   // its forward's tape: the backward alone
            r.tape = m.tapes[(size_t) b].get();
            r.tape_replay = true;
        }
        if (b > 0 && !gpu::memset_async(dnxt, 0, (size_t) (R * C) * 4, s))
            throw std::runtime_error("image engine (MOSS-Image training): zero");
        moss_block(r, m.gemm, s, &m.arena);
        m.ws.end((int) b);
        std::swap(dcur, dnxt);
    }
    gpu::stream_sync(s);
    m.scratch_rows = std::max(m.scratch_rows, R);
    m.stepped = true;
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
    std::vector<std::pair<const float*, int64_t>> parts;
    for (const Impl::Block& B : m.blk)
        if (B.adapted)
            for (const Lora& l : B.l) {
                parts.emplace_back(l.ad.dA, m.rank * l.K);
                parts.emplace_back(l.ad.dB, l.N * m.rank);
            }
    static train::StageTimes tm("moss adamw");
    tm.begin(m.s);
    const double norm = std::sqrt(train::sq_sum(parts, m.s));
    const float gscale = (float) std::min(1.0, 1.0 / (norm + 1e-6));   // torch's clip_grad_norm_ at 1
    ++m.steps;
    for (Impl::Block& B : m.blk)
        if (B.adapted)
            for (Lora& l : B.l) {
                train::adamw(l.A, l.ad.dA, l.mA, l.vA, m.rank * l.K, o, m.steps, gscale, m.s);
                train::adamw(l.B, l.ad.dB, l.mB, l.vB, l.N * m.rank, o, m.steps, gscale, m.s);
                m.sync_views(l);
            }
    gpu::stream_sync(m.s);
    tm.mark("clip + adamw + views", m.s);
    static int64_t n = 0;
    if (tm.on() && ++n % 10 == 0) tm.report();
    return norm;
}

void MossTrainer::save(const std::string& path, const std::vector<std::pair<std::string, std::string>>& meta) const {
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
        if (!B.adapted) continue;
        const std::string p = "diffusion_model.layers." + std::to_string(b) + ".";
        for (int i = 0; i < kMossProducts; ++i) {
            const Lora& l = B.l[i];
            const std::vector<float> A = read(l.A, m.rank * l.K), Bv = read(l.B, l.N * m.rank);
            if (i == kMossQkv) {   // q, k and v: the fused A each, B's rows split in three
                const int64_t n = l.N / 3;
                const char* nm[3] = {"proj.q", "proj.k", "proj.v"};
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
