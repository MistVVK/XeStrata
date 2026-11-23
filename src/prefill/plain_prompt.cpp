// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/plain_prompt.cpp - PlainModel::prompt: a plain model's prompt in chunks of batched products.
//
// The block of plain_model.cpp, T tokens at a time: every projection a matrix product of the chunk's rows by the weight
// dequantized to FP16 (Gemm::native), the GDN conv and recurrence walking the chunk inside their kernels (the prompt
// path's, kernels.hpp), and the routed experts grouped by expert - each expert's weights read once for all the tokens
// routed to it, from its VRAM slot or copied from RAM, dequantized to FP16 and multiplied in one grouped product per
// group of experts.  The norms with the rotation, the gate, the K/V append, the attention over every cell, the router
// and the combine are the decode's kernels, called over the chunk's rows.
//
// The state is left as the windows leave it: the GDN recurrent state and conv history (the same layouts), the K/V
// cells of the chunk's positions.  The decode then continues at the next position.
#include "strata/core/kv_host_dma.hpp"
#include "strata/core/plain_model.hpp"
#include "strata/core/stream_watch.hpp"
#include "strata/core/plain_cpu.hpp"
#include "strata/core/expert_tier.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/prompt_attn.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_tcq.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/platform/env.hpp"
#include "plain_prompt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

namespace strata::core {

namespace k = strata::kernels;

namespace {

bool dense_int8_env() {   // STRATA_PLAIN_DENSE_MMQ=0: every dense product through Gemm::native
    static const bool on = [] {
        const char* v = strata::env_get("PLAIN_DENSE_MMQ");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}

}  // namespace


void PlainModel::prompt_gemm_free(PromptGemm* g) { delete g; }

PlainModel::PromptGemm& PlainModel::prompt_gemm() {
    if (!prompt_gemm_) {
        prompt_gemm_ = std::unique_ptr<PromptGemm, void (*)(PromptGemm*)>(new PromptGemm, &PlainModel::prompt_gemm_free);
        prompt_gemm_->gemm.init_external(cs_, pb_.gemm_scratch, pb_.gemm_elems);
        PromptGemm& g = *prompt_gemm_;
        // the copies on a stream of their own, but not through the first Level Zero adapter (the processor's own
        // graphics, plain_generate.cpp): there a product's wait on the copy stream's event hung now and then, about one
        // prompt in ten on the UHD 770, and none in 15 without it (llm-jp-4.1 Q4_K_M, 2026-10-19)
        const char* v2 = std::getenv("SYCL_UR_USE_LEVEL_ZERO_V2");
        if (v2 == nullptr || std::strcmp(v2, "0") != 0 ||
            strata::core::Runtime::get().device().get_backend() != sycl::backend::ext_oneapi_level_zero)
            g.xs = strata::gpu::stream_create();
        for (int h = 0; h < 2; ++h) {
            if (!strata::gpu::event_create(&g.ready[h])) g.ready[h] = nullptr;
            if (!strata::gpu::event_create(&g.used[h])) g.used[h] = nullptr;
        }
        // without them the copies stay on the compute stream, in order with the products
    }
    return *prompt_gemm_;
}

bool PlainModel::prompt_borrow(int64_t T, std::string& err) {
    if (!settle_swaps(true, err)) return false;   // the tier's copies may still be writing the slots lent below
    if (dense_) {   // no tier: the buffers of its own (open_dense), in even chunks of up to their length
        if (borrow_first_ >= 0 || prompt_dev_.at.empty() || prompt_dev_T_ < 1) {
            err = "plain prompt: no buffers for the prompt's chunks";
            return false;
        }
        const int64_t pieces = (T + prompt_dev_T_ - 1) / prompt_dev_T_;
        uint64_t db = 0, hb = 0;
        borrow_short_ = false;
        borrowed_res_.clear();
        borrow_first_ = 0;
        prompt_carve((T + pieces - 1) / pieces, pb_, nullptr, (uint8_t*) prompt_host_, db, hb, &prompt_dev_);
        PromptGemm& g = prompt_gemm();
        g.gemm.rebind(pb_.gemm_scratch, pb_.gemm_elems);
        g.identity = false;
        return true;
    }
    const int64_t L = n_moe_, E = spec_.n_expert, n = cache_.slots();   // the live slots (--vram-elastic's shrinks)
    const uint64_t* off = cache_.slot_offsets();
    if (borrow_first_ >= 0 || off == nullptr || L <= 0 || n < 2) { err = "plain prompt: no tier to borrow from"; return false; }
    // the chunk's device bytes from the tier's last slots: halved until they take at most the share lend_lowest
    // allows, within its last piece (the slots lie back to back only inside one)
    borrow_short_ = false;
    uint64_t db = 0, hb = 0;
    const int64_t want = T;
    int64_t first = n;
    const int64_t lowest = lend_lowest(n);
    const int64_t least = std::clamp<int64_t>(want, 1, 16);   // a prompt under 16 tokens: one chunk of its length
    for (;; T /= 2) {
        if (T < least) break;
        prompt_carve(T, pb_, nullptr, nullptr, db, hb);
        first = n;
        while (first > lowest && off[n] - off[first] < db) --first;
        if (off[n] - off[first] >= db) break;
    }
    // the buffers of their own when their chunk is no shorter than the borrowed one: no slot lent, prompt_return
    // gives back none
    if (!prompt_dev_.at.empty() && (T < least || std::min(want, prompt_dev_T_) >= T)) {
        const int64_t pieces = (want + prompt_dev_T_ - 1) / prompt_dev_T_;   // even chunks: no short last one
        T = (want + pieces - 1) / pieces;
        borrowed_res_.clear();
        borrow_first_ = n;
        prompt_carve(T, pb_, nullptr, (uint8_t*) prompt_host_, db, hb, &prompt_dev_);
        PromptGemm& g = prompt_gemm();
        g.gemm.rebind(pb_.gemm_scratch, pb_.gemm_elems);
        g.identity = false;
        return true;
    }
    if (T < least) {
        err = "plain prompt: the VRAM tier is too small for the prompt's buffers (" + std::to_string(least) +
              " tokens need " + std::to_string(db >> 20) + " MiB, the tier lends at most " +
              std::to_string((off[n] - off[lowest]) >> 20) + " MiB of its " + std::to_string(off[n] >> 20) + ")";
        borrow_short_ = true;
        return false;
    }
    borrowed_res_.clear();
    for (int64_t l = 0; l < L; ++l)
        for (int64_t e = 0; e < E; ++e) {
            int32_t& r = res_[(size_t) (l * E + e)];
            if (r >= first) {
                borrowed_res_.emplace_back((int32_t) l, (int32_t) e);
                r = -1;
            }
        }
    for (auto& f : free_) f.erase(std::remove_if(f.begin(), f.end(), [&](int32_t x) { return x >= first; }), f.end());
    borrow_first_ = first;
    prompt_carve(T, pb_, cache_.device_slot((int32_t) first), (uint8_t*) prompt_host_, db, hb);
    PromptGemm& g = prompt_gemm();
    g.gemm.rebind(pb_.gemm_scratch, pb_.gemm_elems);
    g.identity = false;   // the identity map is rewritten where the buffers now lie
    return true;
}

bool PlainModel::prompt_return(std::string& err) {
    const int64_t LS = (int64_t) moe_local_.size(), n = cache_.slots();
    if (dense_) borrow_first_ = -1;   // its own buffers: nothing to give back
    if (borrow_first_ < 0 || LS <= 0) return true;   // a borrow needs slots, and slots need MoE layers
    const auto& lay = k::cpu::expert_layout();
    if (!strata::gpu::stream_sync(cs_)) { err = std::string("plain prompt: ") + strata::gpu::last_error(); return false; }
    for (int64_t s = borrow_first_; s < n; ++s) free_[(size_t) slot_layer_[(size_t) s]].push_back((int32_t) s);
    borrow_first_ = -1;
    // the experts the prompt displaced, back into the slots: straight from pinned RAM, else through fill_slot
    for (const auto& [l, e] : borrowed_res_) {
        auto& f = free_[(size_t) l];
        if (f.empty()) break;
        const uint8_t* hb = src_->transient(l, e) ? nullptr : src_->blob(l, e);
        if (hb != nullptr && (strata::gpu::is_host_usm(hb) || (dev_ >= 0 && src_->registered_on(dev_)))) {
            const int32_t slot = f.back();
            if (!strata::gpu::copy_async(cache_.device_slot(slot), hb, lay.blob_bytes(l), cs_)) {
                err = std::string("plain prompt: ") + strata::gpu::last_error();
                return false;
            }
            f.pop_back();
            res_[(size_t) (l * spec_.n_expert + e)] = slot;
        } else if (!fill_slot(l, e, err)) {
            return false;
        }
    }
    borrowed_res_.clear();
    if (!strata::gpu::stream_sync(cs_)) { err = std::string("plain prompt: ") + strata::gpu::last_error(); return false; }
    return true;
}

// After a prompt, as Qwen3.8-Flash-Next's tier (generate.cpp): the counts seeded from its routing at the steady state
// (seed_usage), and the free slots - which Qwen3.8's cache admits misses into as they come - take what the prompt
// routed most (an empty tier, no profile, took a hundred windows to fill a round at a time).  Unlike Qwen3.8's, a
// round's worth of the experts in VRAM (--adapt-swaps) is evicted for them here as well: without it a profiled tier
// kept the profile's experts through the first rounds, Qwen3.6 UD-Q4_K_XL in 8 GB of the B70 (256 tokens, 2026-10-26)
// 90.9 -> 92.9 tok/s with it (Ornith-1.5, llm-jp-4.1 and GLM-5.3-Flash equal).  No more than a round's: up to 2,048,
// after upstream's q36 engine, took the RAM's and the link's bandwidth from GLM-5.3-Flash's decode after a 1K prompt,
// B70 9.5 -> 13.5 tok/s with one round's (2026-10-20).
bool PlainModel::prompt_settle(std::string& err) {
    if (disp_.usage.empty()) return true;
    seed_usage(disp_.usage, prompt_routed_, spec_.n_expert_used, prompt_seen_, spec_.n_expert, 4 * (int64_t) opt_.adapt_every,
               opt_.adapt_decay);
    int64_t n_free = 0;
    for (const auto& f : free_) n_free += (int64_t) f.size();
    return (n_free == 0 && opt_.adapt_swaps <= 0) || adapt(err, std::max<int64_t>(n_free, opt_.adapt_swaps), opt_.adapt_swaps);
}

bool PlainModel::prompt_cpu_experts(int64_t ml, const std::vector<int32_t>& experts, const std::vector<int32_t>& count,
                                    const std::vector<int32_t>& start, std::string& err) {
    // On the host while the GPU computes its share: each routed row's token quantized once to the layer's native
    // activation, up to MAXT rows per job, and the outputs back to their routing entries' rows of parts.
    namespace kc = k::cpu;
    PromptBufs& P = pb_;
    const int64_t N = spec_.n_embd;
    const kc::NativeFmt& f = kc::expert_layout().fmt[(size_t) ml];
    // the tokens the CPU's rows read, each quantized once; on the pool, 32 tokens a piece (upstream 431a26db1)
    std::vector<char> quantized((size_t) P.T, 0);
    std::vector<int32_t> toks;
    for (int32_t e : experts)
        for (int32_t r = start[(size_t) e]; r < start[(size_t) e] + count[(size_t) e]; ++r)
            if (!quantized[(size_t) P.h_src[r]]) { quantized[(size_t) P.h_src[r]] = 1; toks.push_back(P.h_src[r]); }
    // the chunks pipelined over a layer split (prompt_split): the stages take the pool one at a time
    std::unique_lock<std::mutex> turn(root_->pool_mu_, std::defer_lock);
    if (root_->piping_) turn.lock();
    pool_->run_items((int) toks.size(), 32, 64, [&](int i) {
        const int32_t tok = toks[(size_t) i];
        kc::native_quant_act(f, P.h_hn + (size_t) tok * N, P.h_nact + (size_t) tok * kc::kNativeActBytes);
    });
    std::vector<kc::ExpertJobMulti> jobs;
    jobs.reserve(experts.size());
    int64_t q = 0;
    for (int32_t e : experts) {
        kc::ExpertJobMulti j;
        j.blob = src_->blob(ml, e);
        if (j.blob == nullptr) { err = "plain prompt: an expert could not be read for the CPU"; return false; }
        for (int32_t r = start[(size_t) e]; r < start[(size_t) e] + count[(size_t) e]; ++r) {
            const int32_t tok = P.h_src[r];
            uint8_t* act = P.h_nact + (size_t) tok * kc::kNativeActBytes;
            j.nact[j.nt] = act;
            j.out[j.nt] = P.h_cpu + q * N;
            P.h_cpu_row[q] = P.h_entry[r];
            ++j.nt;
            ++q;
            if (j.nt == kc::MAXT) { jobs.push_back(j); j.nt = 0; }
        }
        if (j.nt > 0) jobs.push_back(j);
    }
    for (size_t i = 0; i < jobs.size(); i += kc::ExpertPool::kMaxSplitMulti)
        pool_->run_split_multi_native(f, jobs.data() + i,
                                      (int) std::min<size_t>(kc::ExpertPool::kMaxSplitMulti, jobs.size() - i));
    if (!strata::gpu::copy_async(P.cpu, P.h_cpu, (size_t) (q * N) * 4, cs_) ||
        !strata::gpu::copy_async(P.cpu_row, P.h_cpu_row, (size_t) q * 4, cs_)) {
        err = std::string("plain prompt: the CPU's rows: ") + strata::gpu::last_error();
        return false;
    }
    prefill::scatter_rows(P.cpu, P.cpu_row, P.parts, q, N, cs_);
    return true;
}

bool PlainModel::prompt_warm(std::string& err) {
    if (cpu_ != nullptr) return true;   // the CPU's products need no warming
    for (auto& st : next_) {   // every GPU's products
        const OnDevice on(st->dev_);
        if (!st->prompt_warm(err)) return false;
    }
    if (opt_.prompt_chunk <= 0 || prompt_host_ == nullptr) return true;
    if (!prompt_borrow(std::min<int64_t>(opt_.prompt_chunk, 512), err)) return false;
    const bool ok = prompt_warm_products(err);
    // the W8A8 choice of the dense products (Gemm::llm_w8a8): measured at startup where the program called it; an
    // encoder opened in another process measures here, once a GPU.  A MoE model (dense_ false) takes neither the
    // timing nor the int8 library's load, and neither does a model whose chunks never reach the form's rows: the
    // image engine's text readers (TextEncoder's chunks of 512) paid the timing's 1.3 s in their first encode for a
    // form they never take (MOSS-Image, RX 9060 XT, 2026-11-12: the first request's text 5.37 -> 4.03 s)
    if (ok && opt_.prompt_chunk >= prefill::Gemm::kLlmW8a8Rows) prefill::Gemm::llm_w8a8(cs_, dense_);
    if (ok && dense_) prompt_dense_calibrate();
    if (ok && !dense_ && spec_.n_expert > 0 && prompt_int8()) {   // the experts' products' forms, timed for each type
        const int64_t N = spec_.n_embd, FF = spec_.n_ff_exp;
        for (const auto& f : k::cpu::expert_layout().fmt) {
            if (k::cpu::native_mixed_gu(f)) {
                prefill::mmq::choose_forms(f.gu_type, FF, N, 16, cs_);
                prefill::mmq::choose_forms(f.up_type, FF, N, 16, cs_);
            } else {
                prefill::mmq::choose_forms(f.gu_type, 2 * FF, N, 16, cs_);
            }
            prefill::mmq::choose_forms(f.d_type, N, FF, 16, cs_);
        }
    }
    std::string err2;
    if (!prompt_return(err2) && ok) { err = err2; return false; }
    return ok;
}

bool PlainModel::prompt_warm_products(std::string& err) {
    // A library's first product of a kind loads its kernels for it, inside whichever prompt comes first (rocBLAS on
    // the RX 9060 XT: 0.5 s for the first product, 0.9 s more for the first of 32 rows): each dense shape the prompt
    // takes, once, on a chunk of zeros into the buffers the prompt overwrites.
    if (glm_) return true;   // GLM-5.3-Flash's first prompt loads its products' kernels itself
    // A shape warmed once in the process on the GPU stays warm (Gemm::native_warm): a model opened again
    // (MOSS-Image's text encoder, after each image) skips it, and a streamed layer whose shapes are all warm is not
    // bound (the RTX 4070's 12 GB, the encoder with one layer resident: binding the other 31 to warm them took 3 s
    // an image, 2026-11-05).
    PromptBufs& P = pb_;
    prefill::Gemm& gm = prompt_gemm().gemm;
    const int64_t T = P.T, HV = spec_.ssm_dt_rank;
    auto cold = [&](const W& w, int64_t ldy = 0) {
        return gm.native_cold(w.type, w.n_out, w.n_in, ldy);
    };
    auto warm = [&](const W& w, float* y, int64_t ldy = 0) {
        gm.native_warm(P.x16, w.type, w.data, y, T, w.n_out, w.n_in, ldy);
    };
    try {
        if (!strata::gpu::memset_async(P.x16, 0, (size_t) (T * spec_.n_embd) * 2, cs_)) {
            err = std::string("plain prompt: warming: ") + strata::gpu::last_error();
            return false;
        }
        std::vector<const Block*> all;
        all.reserve(blocks_.size() + 1);
        for (const Block& b : blocks_) all.push_back(&b);
        if (has_mtp_) {
            all.push_back(&mtp_);
            warm(mtp_eh_, P.x);
        }
        for (const Block* bp : all) {
            const Block& b = *bp;
            const bool shared = has_shared_ || dense_;
            if (!(b.recurrent ? cold(b.qkv) || cold(b.z) || cold(b.alpha, 2 * HV) || cold(b.beta, 2 * HV) ||
                                    cold(b.ssm_out)
                              : cold(b.q) || cold(b.k) || cold(b.v) || cold(b.o)) &&
                !(shared && (cold(b.sh_gate) || cold(b.sh_up) || cold(b.sh_down))))
                continue;
            const int64_t l = bp == &mtp_ ? -1 : bp - blocks_.data();   // the MTP layer: nothing to bind
            bind_block(l);
            if (b.recurrent) {
                warm(b.qkv, P.qkv);
                warm(b.z, P.z);
                warm(b.alpha, P.ab, 2 * HV);
                warm(b.beta, P.ab + HV, 2 * HV);
                warm(b.ssm_out, P.mix);
            } else {
                warm(b.q, P.qfull);
                warm(b.k, P.kraw);
                warm(b.v, P.vcur);
                warm(b.o, P.mix);
            }
            if (shared) {   // the shared expert, or the dense FFN in its fields
                warm(b.sh_gate, P.sh_gate);
                warm(b.sh_up, P.sh_up);
                warm(b.sh_down, P.shared);
            }
            unbind_block(l);
        }
    } catch (const std::exception& e) {
        err = std::string("plain prompt: warming: ") + e.what();
        return false;
    }
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain prompt: warming: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}


bool PlainModel::dense_int8_ok(const W& w) const {
    return dense_int8_env() && prefill::mmq::chosen() && w.data != nullptr && prefill::mmq::supported(w.type) &&
           w.n_out % 128 == 0 && w.n_in % 32 == 0;
}

bool PlainModel::prompt_dense_int8(const W& w, const float* x, int64_t ldx, float* y, int64_t T, int64_t ldy) {
    if (!dense_int8_ok(w)) return false;
    PromptBufs& P = pb_;
    prefill::mmq::quantize(x, nullptr, P.xq, w.type, w.n_in, ldx, T, cs_);
    prefill::mmq::Product p;
    p.w = w.data; p.type = w.type; p.w_rows = w.n_out; p.w_cols = w.n_in; p.n = 1; p.xq = P.xq;
    p.bounds = P.dense_bounds; p.ids = P.identity; p.total_rows = T; p.max_rows = T;
    p.dst = y; p.ld_dst = ldy > 0 ? ldy : w.n_out;
    prompt_gemm().mmq.run(p, cs_);
    return true;
}

// The int8 products read the weight in its blocks a row tile at a time; Gemm::native dequantizes it to FP16 once a
// product (a time of its own, whatever the rows) and multiplies it in the products' library.  Each is timed as
// a + b rows from 16 rows and the chunk's (the best of three, after one), and dense_native_rows_ is where they cross.
// The RX 9060 XT, Qwen3.5-9B Q4_K_M (2026-10-27): int8 faster at 16 and 23 tokens, FP16 at 128 and above (2289 tok/s
// at 1K against 843).  Gemm::native is timed with its sums in FP32 and in FP16 too: the RTX 4070 takes FP16's (1K
// 2842 -> 3572 tok/s warm, 32K 1524 -> 1681); on the B70 the two are close in the timing and FP32's ahead in the
// prompt (1K 3476 against 3304; 2026-10-29: 1K 3125 against 2945, 32K 1276 against 1248), where FP32's are kept
// (below).
void PlainModel::prompt_dense_calibrate() {
    PromptBufs& P = pb_;
    if (blocks_.empty() || P.T < 32) return;
    bind_block(0);
    struct Unbind {
        PlainModel* m;
        ~Unbind() { m->unbind_block(0); }
    } unbind{this};
    const W& w = blocks_.front().sh_gate;   // the dense FFN's gate: the largest of the products a chunk takes
    if (!dense_int8_ok(w)) return;
    // timed once a GPU and a shape: a model opened again in the same process (MOSS-Image's text encoder, after each
    // image) takes what the first open chose, its timings beside the image engine's other work having moved the choice
    // and the same prompt's states with it (the B70, 2026-10-29: FP16 from 320 rows at the first open, 2816 at the next)
    struct Chosen {
        int64_t rows;
        bool half;
    };
    struct Chosens {
        std::mutex mu;
        std::map<std::tuple<int, int64_t, int64_t, int64_t>, Chosen> of;
    };
    static PerDevice<std::shared_ptr<Chosens>> chosen_of;
    Chosens& chosen = *chosen_of.get(Runtime::get().stream(cs_).get_device(), [] { return std::make_shared<Chosens>(); });
    const auto key = std::make_tuple((int) w.type, w.n_in, w.n_out, P.T);
    // the int8 products of few rows take K in parts where the stream holds their buffer (iq_mmq.cpp's launch)
    prompt_split_reserve();
    {
        std::lock_guard<std::mutex> lock(chosen.mu);
        const auto it = chosen.of.find(key);
        if (it != chosen.of.end()) {
            dense_native_rows_ = it->second.rows;
            dense_half_sums_ = it->second.half;
            return;
        }
    }
    prefill::Gemm& gm = prompt_gemm().gemm;
    PromptGemm& pg = prompt_gemm();
    const int64_t lo = 16, hi = P.T;
    using Clock = std::chrono::steady_clock;
    auto best1 = [&](const std::function<void()>& run) {   // one run's time
        const auto t0 = Clock::now();
        run();
        if (!strata::gpu::stream_sync(cs_)) return -1.0;
        return std::chrono::duration<double>(Clock::now() - t0).count();
    };
    auto best = [&](const std::function<void()>& run) {
        double t = 1e30;
        for (int i = 0; i < 4; ++i) {
            const double r = best1(run);
            if (r < 0) return -1.0;
            if (i > 0) t = std::min(t, r);
        }
        return t;
    };
    try {
        if (!strata::gpu::memset_async(P.hn, 0, (size_t) (hi * w.n_in) * 4, cs_) ||
            !strata::gpu::memset_async(P.x16, 0, (size_t) (hi * w.n_in) * 2, cs_))
            return;
        if (!pg.identity) {
            prefill::mmq::iota(P.identity, P.T, cs_);
            pg.identity = true;
        }
        // Gemm::native with its sums in FP32 and in FP16 (h), the faster at the chunk's rows taken for the products
        // from the normed rows
        double n[2], h[2], q[2];
        for (int i = 0; i < 2; ++i) {
            const int64_t T = i == 0 ? lo : hi;
            P.h_dense_bounds[0] = 0;
            P.h_dense_bounds[1] = (int32_t) T;
            if (!strata::gpu::copy_async(P.dense_bounds, P.h_dense_bounds, 2 * sizeof(int32_t), cs_)) return;
            // the two sums timed in turn, so that the GPU's clocks rising over the timings favour neither (the B70,
            // MOSS-Image's text encoder just opened, 2026-10-29: the products 2.4 times slower in one start of five,
            // FP16's timed after FP32's and taken)
            n[i] = h[i] = 1e30;
            for (int k = 0; k < 4; ++k) {
                const auto sums = [&](bool half) {
                    return best1([&] { gm.native(P.x16, w.type, w.data, P.sh_gate, T, w.n_out, w.n_in, 0, half); });
                };
                const double a = sums(false), b = sums(true);
                if (a < 0 || b < 0) return;
                if (k > 0) {
                    n[i] = std::min(n[i], a);
                    h[i] = std::min(h[i], b);
                }
            }
            q[i] = best([&] { prompt_dense_int8(w, P.hn, w.n_in, P.sh_gate, T, 0); });
            if (n[i] < 0 || h[i] < 0 || q[i] < 0) return;
        }
        // FP32's sums unless FP16's are clearly the faster (the GeForce's twice): the two form different values (the
        // GDN blocks' ssm_out takes them at every length), and a near tie (the B70, 2026-10-29: FP16's 0.80 ms at 512
        // rows, FP32's 0.83 in one start of five, 0.89 in the others) picked one or the other from start to start, the
        // same prompt's states, and an image made from them, changing with it
        dense_half_sums_ = h[1] * 1.2 < n[1];
        if (dense_half_sums_) { n[0] = h[0]; n[1] = h[1]; }
        // native - int8 = d0 + d1 (T - lo): FP16 from where it turns negative, at a multiple of 64 rows, so that the
        // timings' noise (the B70: 304 to 322 rows from start to start) leaves a prompt of a given length on one side
        const double d0 = n[0] - q[0], d1 = ((n[1] - q[1]) - d0) / (double) (hi - lo);
        if (d0 <= 0) dense_native_rows_ = 0;
        else if (d1 < 0) dense_native_rows_ = std::max<int64_t>(64, (lo + (int64_t) std::ceil(d0 / -d1) + 32) / 64 * 64);
        std::fprintf(stderr, "xestrata plain: dense products in FP16 (the products' library, %s sums) from %s rows, in "
                             "int8 below (%.2f / %.2f ms at %lld rows, %.2f / %.2f at %lld)\n",
                     dense_half_sums_ ? "FP16" : "FP32",
                     dense_native_rows_ == INT64_MAX ? "no" : std::to_string(dense_native_rows_).c_str(), 1e3 * n[0],
                     1e3 * q[0], (long long) lo, 1e3 * n[1], 1e3 * q[1], (long long) hi);
        std::lock_guard<std::mutex> lock(chosen.mu);
        chosen.of[key] = {dense_native_rows_, dense_half_sums_};
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata plain: timing the dense products: %s (int8 kept)\n", e.what());
        dense_native_rows_ = INT64_MAX;
        dense_half_sums_ = false;
    }
}

// The parts' buffer of the int8 products of few rows (prefill::mmq::split_reserve): for the widest of the dense
// products a block takes, at their few rows
bool PlainModel::prompt_split_reserve() {
    int64_t n_out = spec_.n_embd;
    const Block& b = blocks_.front();
    for (const W* w : {&b.qkv, &b.z, &b.ssm_out, &b.q, &b.k, &b.v, &b.o, &b.qkv_all, &b.gu_all, &b.sh_gate, &b.sh_up,
                       &b.sh_down})
        if (w->data != nullptr) n_out = std::max<int64_t>(n_out, w->n_out);
    return prefill::mmq::split_reserve(k::kIqMmqSplitRows, n_out, cs_);
}

namespace {
// The routed path's small tables (the routing ids back, the rows' sources, entries and bounds, each group's blob table
// and bounds up) as kernels that read or write the pinned host buffers in place, not as copies (upstream 5480e2bad,
// #6053: a 224-byte record's host copy queued behind the streamed experts' copies, so a group's products waited for
// the experts in flight; Qwen3.8-Flash-Next's prompt path does the same with HostVec, upstream 1b3bb8f8).  The bytes
// are the same.  Measured 2026-11-23 (3 alternating rounds, the same answers): RTX 4070, Ornith-1.5 GSQ-RCO-3bit, a
// 3,562-token prompt 1,580 -> 1,530 ms (-3.2%, streamed experts); Qwen3.6 UD-Q4_K_XL there and every cell on the B70
// (all experts in VRAM) and the RX 9060 XT within 10 ms.  XESTRATA_PLAIN_MAPPED_TABLES=0: the copies.
inline bool mapped_tables() {
    static const bool on = [] { const char* v = strata::env_get("PLAIN_MAPPED_TABLES"); return v == nullptr || std::strtol(v, nullptr, 10) != 0; }();
    return on;
}
// up to 8 host-to-device copies of 4-byte words: one kernel, or copies with XESTRATA_PLAIN_MAPPED_TABLES=0
bool tables_up(std::initializer_list<k::MappedCopy> cs, strata::gpu::Stream s) {
    if (mapped_tables()) {
        std::vector<k::MappedCopy> v(cs);
        k::copy_from_mapped_multi(v.data(), (int) v.size(), s);
        return true;
    }
    for (const k::MappedCopy& c : cs)
        if (!strata::gpu::copy_async(c.dst, c.src, (size_t) c.words * 4, s)) return false;
    return true;
}
}  // namespace

bool PlainModel::prompt_routed(int64_t ml, int64_t T, bool cpu_share, bool large_share, int64_t max_cpu, int32_t* routes,
                                const std::function<void(int)>& mark, PromptRouted& st, std::string& err) {
    PromptBufs& P = pb_;
    const auto& lay = k::cpu::expert_layout();
    const int64_t N = spec_.n_embd, E = spec_.n_expert, K = spec_.n_expert_used, FF = spec_.n_ff_exp;
    strata::gpu::Stream cs = cs_;
    PromptGemm& pg = prompt_gemm();
    // the routed experts grouped by expert, on the host from this layer's ids (and the normed rows, for the
    // CPU's share)
    if (mapped_tables()) k::copy_i32_from_mapped(P.h_ids, P.ids, T * K, cs);   // a kernel writes them into the pinned buffer
    if ((cpu_share && !strata::gpu::copy_async(P.h_hn, P.hn, (size_t) (T * N) * 4, cs)) ||
        (!mapped_tables() && !strata::gpu::copy_async(P.h_ids, P.ids, (size_t) (T * K) * 4, cs)) || !strata::gpu::stream_sync(cs)) {
        err = std::string("plain prompt: the routes: ") + strata::gpu::last_error();
        return false;
    }
    std::vector<int32_t> count((size_t) E, 0);
    for (int64_t i = 0; i < T * K; ++i) {
        const int32_t e = P.h_ids[i];
        if (e < 0 || e >= E) { err = "plain prompt: a routed expert is out of range"; return false; }
        ++count[(size_t) e];
        heat_[(size_t) (ml * E + e)] += 1.0;
    }
    dump_routes(ml, P.h_ids, T, K);
    // the int8 products while the experts this layer routes to average fewer rows than this GPU measured them faster
    // at (int8_below, prefill::mmq::int8_rows_below: Qwen3.8-Flash-Next's rule), else the FP16 path - where every one
    // of them is in VRAM: the FP16 path copies a missed expert on the compute queue before its dequantization, where
    // the int8 products copy the next group's beside this one's products.  Ornith-1.5-35B-A3B GSQ-RCO 3bit on the RTX
    // 4070 (about 78% of the routed experts in VRAM), a 32K prompt's chunks of 8192 tokens (256 rows an expert, where
    // the products alone time FP16 8% faster): 2,934 tok/s in int8, 2,336 to 2,441 on the FP16 path (2026-11-12)
    int64_t routed = 0, missed = 0;
    for (int64_t e = 0; e < E; ++e) {
        routed += count[(size_t) e] > 0;
        missed += count[(size_t) e] > 0 && res_[(size_t) (ml * E + e)] < 0;
    }
    const bool use_mmq = P.int8 && (!P.f16 || P.int8_below == INT64_MAX || missed > 0 ||
                                    T * K < P.int8_below * std::max<int64_t>(routed, 1));
    if (routes != nullptr) std::copy(P.h_ids, P.h_ids + T * K, routes);
    const auto& f = lay.fmt[(size_t) ml];
    // the CPU's share of the missed experts (after Project Maya's CPU lane, as the windows split them): of
    // the f experts not in VRAM, pcie_split_m's count over PCIe from the times measured at load, the rest
    // on the CPU pool beside the GPU, those with the fewest rows first; longer experts are split into jobs
    // of at most MAXT rows each.
    std::vector<char> on_cpu((size_t) E, 0);
    std::vector<int32_t> cpu_experts;
    if (cpu_share && f.gu_type != 42) {
        int64_t hits = 0;
        for (int64_t e = 0; e < E; ++e) hits += count[(size_t) e] > 0 && res_[(size_t) (ml * E + e)] >= 0;
        // the candidates and how many of them (core/expert_tier, as Qwen3.8-Flash-Next's prompt path takes them)
        const CpuShareCandidates cc = cpu_share_candidates(
            count.data(), E, ml, *src_, [&](int32_t e) { return res_[(size_t) (ml * E + e)] >= 0; },
            [](int32_t) { return false; }, large_share ? T : k::cpu::MAXT);
        // a fixed share (--prefill-cpu-share S) gives the CPU S of the missed experts, else the decode's split times:
        // the share measured on the prompt's layers (Qwen3.8-Flash-Next's) was no faster here.  Qwen3.6-35B-A3B
        // UD-Q4_K_XL on the RTX 4070 (2026-11-12, 2 rounds): the same with the split measured; with --pcie-frac 0, where
        // no split is measured and the CPU took none, 1K 519 -> 645 tok/s, 4K 1284 -> 1304, 32K 2651 -> 2371
        const double fixed = strata::prefill::cpu_share_setting();
        const int64_t want = std::min(max_cpu, cpu_share_count(cc.missed, hits, fixed > 0.0 ? fixed : -1.0, nullptr, &split_));
        cpu_share_take(cc.cand, want, P.cpu_cap, [&](int32_t e) { return src_->blob(ml, e); },
                       [&](int32_t e, int32_t, const uint8_t*) {
                           on_cpu[(size_t) e] = 1;
                           cpu_experts.push_back(e);
                       });
    }
    // The FP16 path's resident experts of at most kIqGemmTileRows rows: their products iq_gemm_grouped_f16 takes straight
    // from their GGUF blocks in the cache (dequantized in local memory to the same FP16 values: the same bits as
    // dequantizing and multiplying, without writing the dequantized weights), as Qwen3.8-Flash-Next's prompt does
    // (prefill.cpp); more rows than one work-group's would decode the weights again per group.  Gate and up of one type,
    // the rows of up right after gate's; the f16_from_blocks form.  STRATA_PREFILL_FUSED=0: every expert
    // dequantized.
    static const bool fused_on = [] {
        const char* v = strata::env_get("PREFILL_FUSED");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool fuse_res = fused_on && !use_mmq && !k::cpu::native_mixed_gu(f) && f.up_off == (size_t) FF * f.gu_row &&
                          k::has_form(k::MatrixForm::f16_from_blocks) && k::iq_gemm_grouped_ok(f.gu_type, 2 * FF, N) &&
                          k::iq_gemm_grouped_ok(f.d_type, N, FF);
    auto fusable = [&](int64_t e) {
        return fuse_res && count[(size_t) e] <= k::kIqGemmTileRows && res_[(size_t) (ml * E + e)] >= 0;
    };
    // rows in expert order, the GPU's experts first (their bounds one run; the fusable ones before the others), the
    // CPU's after them
    std::vector<int32_t> experts, start((size_t) E, 0);
    int64_t gpu_rows = 0;
    size_t nf = 0;   // experts' first nf: the fusable
    {
        int32_t at = 0;
        for (int pass = 0; pass < 3; ++pass)
            for (int64_t e = 0; e < E; ++e)
                if (count[(size_t) e] > 0 && (pass == 2 ? on_cpu[(size_t) e] != 0
                                                        : on_cpu[(size_t) e] == 0 && fusable(e) == (pass == 0))) {
                    start[(size_t) e] = at;
                    at += count[(size_t) e];
                    if (pass < 2) experts.push_back((int32_t) e);
                    if (pass == 0) ++nf;
                }
        for (int32_t e : experts) gpu_rows += count[(size_t) e];
    }
    {
        std::vector<int32_t> fill(start);
        for (int64_t i = 0; i < T * K; ++i) {
            const int32_t e = P.h_ids[i], row = fill[(size_t) e]++;
            P.h_src[row] = (int32_t) (i / K);   // the token whose row it reads
            P.h_slot[i] = row;                  // where routing entry i finds its expert's output
            P.h_entry[row] = (int32_t) i;       // the int8 down product's: the row it writes
        }
    }
    // the group bounds: the routed rows of the experts in `experts` order, absolute
    int64_t nb = 0;
    for (int32_t e : experts) P.h_bounds[nb++] = start[(size_t) e];
    P.h_bounds[nb] = (int32_t) gpu_rows;
    if (!tables_up({{P.src, P.h_src, T * K}, {use_mmq ? P.entry : P.slot, use_mmq ? P.h_entry : P.h_slot, T * K},
                    {P.bounds, P.h_bounds, nb + 1}}, cs)) {
        err = std::string("plain prompt: the groups: ") + strata::gpu::last_error();
        return false;
    }
    // a layer whose gate and up differ in type: up's rows too, into xq2 where the two repeat the
    // blocks differently (Q2_K against another type)
    const int up_type = k::cpu::native_mixed_gu(f) ? f.up_type : f.gu_type;
    const bool up_xq2 = k::iq_mmq_type_dup(up_type) != k::iq_mmq_type_dup(f.gu_type);
    if (use_mmq) {
        prefill::mmq::quantize(P.hn, P.src, P.xq, f.gu_type, N, N, T * K, cs);
        if (up_xq2) prefill::mmq::quantize(P.hn, P.src, P.xq2, up_type, N, N, T * K, cs);
    } else {
        prefill::gather_rows16(P.x16, P.src, P.xs16, T * K, N, cs);
    }
    mark(1);
    // whether a missed expert's blob is assembled into a short-lived buffer, asked of each: with a RAM copy of part
    // of the layer the first expert says nothing of the others (as the read-ahead below)
    bool transient = false;
    for (int32_t e : experts)
        if (res_[(size_t) (ml * E + e)] < 0 && src_->transient(ml, e)) {
            transient = true;
            break;
        }
    // a source that assembles its experts from the files (transient: GGUF read in place): each group's missed experts
    // read ahead on the source's threads (prefetch, its page faults side by side), the next group's while this one is
    // copied and multiplied.  Group gi of the loop below takes the gi-th kGroup of the missed experts in order.
    // (GLM-5.3-Flash UD-IQ3_XXS on the B70, a 1K-token prompt: its 11,735 missed experts took 22 s read one by one.)
    // (each expert asked: with a RAM copy of part of the layer the first expert says nothing of the others - the
    // layer's file reads went one by one whenever its lowest routed expert was in RAM)
    std::vector<std::vector<int64_t>> ahead;
    if (use_mmq) {
        int64_t missed = 0;   // the loop's groups count every missed expert: a group's read-ahead takes its file ones
        for (int32_t e : experts)
            if (res_[(size_t) (ml * E + e)] < 0) {
                if (missed++ % PromptBufs::kGroup == 0) ahead.emplace_back();
                if (src_->transient(ml, e)) ahead.back().push_back(e);
            }
    }
    std::future<void> fetch;
    auto fetch_group = [&](size_t i) {
        if (i < ahead.size() && !ahead[i].empty())
            fetch = std::async(std::launch::async, [this, ml, &ahead, i] {
                src_->prefetch(ml, ahead[i].data(), (int64_t) ahead[i].size());
            });
    };
    fetch_group(0);
    // int8: as many experts a launch as the staging lets (those in VRAM read where they lie, kGroup from
    // RAM at most), gate/up, the SwiGLU, its rows to int8, down
    // a transient source's blobs through pinned host memory (h_stage's halves, as the device staging's) with the copies
    // beside the products, instead of each copied from pageable memory before the next is read
    const bool bounce = transient && P.h_stage != nullptr && pg.overlap();
    for (size_t g0 = 0, gi = 0; use_mmq && g0 < experts.size(); ++gi) {
        if (transient && !bounce && !strata::gpu::stream_sync(cs)) {   // the staging's last readers are done
            err = std::string("plain prompt: ") + strata::gpu::last_error();
            return false;
        }
        if (fetch.valid()) fetch.wait();   // this group's missed experts are assembled
        fetch_group(gi + 1);
        size_t g1 = g0;
        int staged = 0;
        int64_t mr = 0;
        // the half of the staging this group takes; its copies wait for the products two groups back
        const bool ovl = (!transient || bounce) && pg.overlap();
        const int h = (int) (gi % 2);
        if (bounce && pg.ready_live[h] && !strata::gpu::event_sync(pg.ready[h])) {   // the half's last copies are done
            err = std::string("plain prompt: ") + strata::gpu::last_error();
            return false;
        }
        strata::gpu::Stream ccs = ovl ? pg.xs : cs;
        if (ovl && pg.used_live[h] && !strata::gpu::stream_wait_event(ccs, pg.used[h])) {
            err = std::string("plain prompt: ") + strata::gpu::last_error();
            return false;
        }
        for (; g1 < experts.size(); ++g1) {
            const int32_t e = experts[g1];
            const int32_t sl = res_[(size_t) (ml * E + e)];
            const uint8_t* blob = nullptr;
            if (sl >= 0) {
                blob = cache_.device_slot(sl);
            } else {
                if (staged == PromptBufs::kGroup) break;
                const size_t at = (size_t) ((ovl ? h * PromptBufs::kGroup : 0) + staged++) * lay.max_blob;
                uint8_t* dst = P.stage + at;
                const uint8_t* hb = src_->blob(ml, e);
                if (bounce && hb != nullptr) {
                    std::memcpy(P.h_stage + at, hb, f.bytes);
                    hb = P.h_stage + at;
                }
                const bool copied = hb != nullptr && (transient && !bounce ? strata::gpu::copy(dst, hb, f.bytes)
                                                                           : strata::gpu::copy_async(dst, hb, f.bytes, ccs));
                if (!copied) { err = "plain prompt: an expert could not be read"; return false; }
                blob = dst;
            }
            P.h_wtab[g1] = blob;
            mr = std::max<int64_t>(mr, count[(size_t) e]);
        }
        if (ovl && (!strata::gpu::event_record(pg.ready[h], ccs) || !strata::gpu::stream_wait_event(cs, pg.ready[h]))) {
            err = std::string("plain prompt: ") + strata::gpu::last_error();
            return false;
        }
        if (ovl) pg.ready_live[h] = true;
        const int G = (int) (g1 - g0);
        const int32_t r0 = P.h_bounds[g0], nr = P.h_bounds[g1] - r0;
        int32_t* hrb = P.h_rbounds + g0 + gi;
        for (int q = 0; q <= G; ++q) hrb[q] = P.h_bounds[g0 + (size_t) q] - r0;
        static_assert(sizeof(void*) == 8, "a blob pointer is two words");
        if (!tables_up({{static_cast<void*>(P.wtab + g0), static_cast<const void*>(P.h_wtab + g0), 2 * (int64_t) G},
                        {P.rbounds + g0 + gi, hrb, (int64_t) G + 1}}, cs)) {
            err = std::string("plain prompt: the groups: ") + strata::gpu::last_error();
            return false;
        }
        prefill::mmq::Product gu;
        gu.type = f.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.n = G; gu.xq = P.xq;
        gu.bounds = P.bounds + g0; gu.ids = P.identity; gu.total_rows = T * K; gu.max_rows = mr; gu.rows = nr;
        gu.dst = P.gu; gu.ld_dst = 2 * FF; gu.wtab = P.wtab + g0; gu.woff = 0;
        if (k::cpu::native_mixed_gu(f)) {   // the gate rows, then the up rows in their own type
            gu.w_rows = FF;
            pg.mmq.run(gu, cs);
            gu.type = up_type; gu.xq = up_xq2 ? P.xq2 : P.xq; gu.dst = P.gu + FF; gu.woff = f.up_off;
        }
        pg.mmq.run(gu, cs);
        prefill::mmq::swiglu(P.gu + (int64_t) r0 * 2 * FF, P.hh + (int64_t) r0 * FF, nr, FF,
                             /*interleaved=*/false, cs);
        prefill::mmq::quantize(P.hh + (int64_t) r0 * FF, nullptr, P.hq, f.d_type, FF, FF, nr, cs);
        prefill::mmq::Product dn;
        dn.type = f.d_type; dn.w_rows = N; dn.w_cols = FF; dn.n = G; dn.xq = P.hq;
        dn.bounds = P.rbounds + g0 + gi; dn.ids = P.entry + r0; dn.total_rows = nr; dn.max_rows = mr;
        dn.dst = P.parts; dn.ld_dst = N; dn.wtab = P.wtab + g0; dn.woff = f.down_off;   // routing order
        pg.mmq.run(dn, cs);
        if (ovl) {
            if (!strata::gpu::event_record(pg.used[h], cs)) {
                err = std::string("plain prompt: ") + strata::gpu::last_error();
                return false;
            }
            pg.used_live[h] = true;
        }
        g0 = g1;
    }
    // the CPU's rows go into parts after the GPU's products are queued (it computes beside them): here on
    // the int8 path, which writes parts itself; after the FP16 path's gather, which writes every routing
    // entry's row of parts (the CPU's from rows of dm no product wrote)
    if (use_mmq && !cpu_experts.empty() && !prompt_cpu_experts(ml, cpu_experts, count, start, err))
        return false;
    st.cpu_n += (int64_t) cpu_experts.size();
    for (int32_t e : cpu_experts) st.cpu_rows += count[(size_t) e];
    for (int64_t e = 0; e < E; ++e) st.miss_n += count[(size_t) e] > 0 && res_[(size_t) (ml * E + e)] < 0;
    // A layer the int8 products could take (their types) has its gate and up rows dequantized interleaved, with down's,
    // in one launch an expert (iq_dequant_expert_f16, Qwen3.8-Flash-Next's layout and launch; three launches an expert
    // before), the SwiGLU reading them so; else gate rows then up rows (the GGUF order)
    const bool il = P.int8 && !k::cpu::native_mixed_gu(f) && N % 256 == 0;
    // the fusable experts, kIqGemmGroup a launch, from their cache slots
    std::vector<const void*> fgu, fd;
    for (size_t j0 = 0; !use_mmq && j0 < nf; j0 += (size_t) k::kIqGemmGroup) {
        const size_t j1 = std::min(nf, j0 + (size_t) k::kIqGemmGroup);
        fgu.clear();
        fd.clear();
        int64_t maxr = 0;
        for (size_t j = j0; j < j1; ++j) {
            const int32_t e = experts[j];
            const uint8_t* blob = cache_.device_slot(res_[(size_t) (ml * E + e)]);
            fgu.push_back(blob);
            fd.push_back(blob + f.down_off);
            maxr = std::max<int64_t>(maxr, count[(size_t) e]);
        }
        const int G = (int) (j1 - j0);
        const int64_t r0 = P.h_bounds[j0], r1 = P.h_bounds[j1];
        k::iq_gemm_grouped_f16(f.gu_type, P.xs16, fgu.data(), il ? FF : 0, P.gu, 2 * FF, P.bounds + j0, G, maxr, 2 * FF,
                               N, cs);
        prefill::mmq::swiglu(P.gu + r0 * 2 * FF, P.hh + r0 * FF, r1 - r0, FF, il, cs);
        kernels::f32_to_f16_bulk(P.hh + r0 * FF, P.h16 + r0 * FF, (r1 - r0) * FF, cs);
        k::iq_gemm_grouped_f16(f.d_type, P.h16, fd.data(), 0, P.dm, N, P.bounds + j0, G, maxr, N, FF, cs);
    }
    for (size_t g0 = nf; !use_mmq && g0 < experts.size(); g0 += PromptBufs::kGroup) {
        const int G = (int) std::min<size_t>(PromptBufs::kGroup, experts.size() - g0);
        if (transient && !strata::gpu::stream_sync(cs)) {   // the staging's last readers are done
            err = std::string("plain prompt: ") + strata::gpu::last_error();
            return false;
        }
        for (int q = 0; q < G; ++q) {   // each expert's weights, then FP16
            const int32_t e = experts[g0 + (size_t) q];
            const int32_t sl = res_[(size_t) (ml * E + e)];
            const uint8_t* blob = nullptr;
            if (sl >= 0) {
                blob = cache_.device_slot(sl);
            } else {
                uint8_t* dst = P.stage + (size_t) q * lay.max_blob;
                const uint8_t* hb = src_->blob(ml, e);
                const bool copied = hb != nullptr &&
                    (transient ? strata::gpu::copy(dst, hb, f.bytes) : strata::gpu::copy_async(dst, hb, f.bytes, cs));
                if (!copied) { err = "plain prompt: an expert could not be read"; return false; }
                blob = dst;
            }
            uint16_t* dq = P.dq_gu + (size_t) q * P.gu_elems;
            uint16_t* dqd = P.dq_d + (size_t) q * P.d_elems;
            if (il) {
                k::iq_dequant_expert_f16(f.gu_type, blob, blob + f.up_off, FF, N, dq, f.d_type, blob + f.down_off, dqd,
                                         cs);
                continue;
            }
            if (k::cpu::native_mixed_gu(f)) {   // gate and up of different types: each by its own
                k::dequant_f16(f.gu_type, blob, 0, FF, N, dq, cs);
                k::dequant_f16(f.up_type, blob + f.up_off, 0, FF, N, dq + FF * N, cs);
            } else {
                k::dequant_f16(f.gu_type, blob, 0, 2 * FF, N, dq, cs);
            }
            k::dequant_f16(f.d_type, blob + f.down_off, 0, N, FF, dqd, cs);
        }
        const int32_t* hbnd = P.h_bounds + g0;
        const int32_t* dbnd = P.bounds + g0;
        const int64_t r0 = hbnd[0], r1 = hbnd[G];
        // the launch for the group's own largest expert (Qwen3.8-Flash-Next's): the layer's largest padded every group
        int64_t maxr = 0;
        for (int q = 0; q < G; ++q) maxr = std::max<int64_t>(maxr, count[(size_t) experts[g0 + (size_t) q]]);
        if (!prefill::Gemm::f16_groups(P.xs16, P.dq_gu, P.gu_elems, P.gu, hbnd, G, 2 * FF, N, cs))
            k::xmx_gemm_grouped(P.xs16, P.dq_gu, P.gu_elems, P.gu, dbnd, G, maxr, 2 * FF, N, cs);
        // h = silu(gate) * up, then FP16 for the down product
        prefill::mmq::swiglu(P.gu + r0 * 2 * FF, P.hh + r0 * FF, r1 - r0, FF, il, cs);
        kernels::f32_to_f16_bulk(P.hh + r0 * FF, P.h16 + r0 * FF, (r1 - r0) * FF, cs);
        if (!prefill::Gemm::f16_groups(P.h16, P.dq_d, P.d_elems, P.dm, hbnd, G, N, FF, cs))
            k::xmx_gemm_grouped(P.h16, P.dq_d, P.d_elems, P.dm, dbnd, G, maxr, N, FF, cs);
    }
    mark(2);
    // routing entry i's output into row i, then the weighted sum (and the gated shared expert)
    if (!use_mmq)
        k::gather_rows(reinterpret_cast<const uint8_t*>(P.dm), N * 4, P.slot, T * K,
                       reinterpret_cast<uint8_t*>(P.parts), cs);
    if (!use_mmq && !cpu_experts.empty() && !prompt_cpu_experts(ml, cpu_experts, count, start, err))
        return false;
    return true;
}

bool PlainModel::prompt_routed_tuned(int64_t ml, int64_t T, bool cpu_share, int32_t* routes,
                                     const std::function<void(int)>& mark, PromptRouted& st, std::string& err) {
    const int mode = cpu_share_staged_3k_mode();
    const bool eligible = cpu_share && T >= 2048 && T <= 3072 && mode != 0;
    if (eligible && mode < 0 && prompt_share_pending_) {
        float ms = 0;
        if (!strata::gpu::event_sync(prompt_share_events_[1]) ||
            !strata::gpu::event_elapsed_ms(&ms, prompt_share_events_[0], prompt_share_events_[1])) {
            err = "plain prompt: share timing event"; return false;
        }
        prompt_share_3k_.add(prompt_share_arm_, prompt_share_layer_, prompt_share_missed_, ms, "plain");
        prompt_share_pending_ = false;
    }
    const bool tune = eligible && mode < 0 && !prompt_share_3k_.decided();
    const bool large = eligible && (mode == 1 || (mode < 0 && prompt_share_3k_.next()));
    if (tune) {
        if (!prompt_share_events_[0] && (!strata::gpu::event_create(&prompt_share_events_[0], true) ||
                                          !strata::gpu::event_create(&prompt_share_events_[1], true))) {
            err = "plain prompt: share timing event"; return false;
        }
        if (!strata::gpu::event_record(prompt_share_events_[0], cs_)) { err = "plain prompt: share timing start"; return false; }
    }
    // The trial moves at most 16 experts; the chosen path takes its full share after the timing.
    if (!prompt_routed(ml, T, cpu_share, large, tune && large ? 16 : INT64_MAX, routes, mark, st, err)) return false;
    if (tune && st.miss_n > 0 && (!large || st.cpu_n > 0)) {
        if (!strata::gpu::event_record(prompt_share_events_[1], cs_)) { err = "plain prompt: share timing end"; return false; }
        prompt_share_pending_ = true; prompt_share_arm_ = large; prompt_share_layer_ = ml; prompt_share_missed_ = st.miss_n;
    }
    return true;
}

bool PlainModel::prompt(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err) {
    if (cpu_ != nullptr) return cpu_run(tokens, n, pos, 1, out, logits, err);
    // upstream 711a7e937: the stall report asks this stream (the one the prompt's waits block on) whether its work is done
    const strata::core::WatchedStream watched("the prompt path's compute stream", cs_, strata::core::current_device());
    return next_.empty() ? prompt_one(tokens, n, pos, out, logits, err) : prompt_split(tokens, n, pos, out, logits, err);
}

bool PlainModel::product_shape(int64_t layer, ProductHook::Which which, int64_t& n_out, int64_t& n_in) const {
    if (layer < 0 || layer >= (int64_t) blocks_.size()) return false;
    const Block& b = blocks_[(size_t) layer];
    const W* w = nullptr;
    switch (which) {
    case ProductHook::kQkv: w = b.recurrent ? &b.qkv : nullptr; break;
    case ProductHook::kZ: w = b.recurrent ? &b.z : nullptr; break;
    case ProductHook::kAlpha: w = b.recurrent ? &b.alpha : nullptr; break;
    case ProductHook::kBeta: w = b.recurrent ? &b.beta : nullptr; break;
    case ProductHook::kSsmOut: w = b.recurrent ? &b.ssm_out : nullptr; break;
    case ProductHook::kQ: w = b.recurrent ? nullptr : &b.q; break;
    case ProductHook::kK: w = b.recurrent ? nullptr : &b.k; break;
    case ProductHook::kV: w = b.recurrent ? nullptr : &b.v; break;
    case ProductHook::kO: w = b.recurrent ? nullptr : &b.o; break;
    case ProductHook::kGate: w = dense_ ? &b.sh_gate : nullptr; break;
    case ProductHook::kUp: w = dense_ ? &b.sh_up : nullptr; break;
    case ProductHook::kDown: w = dense_ ? &b.sh_down : nullptr; break;
    }
    if (w == nullptr || w->n_out <= 0) return false;
    n_out = w->n_out;
    n_in = w->n_in;
    return true;
}

bool PlainModel::encode(const int32_t* tokens, int64_t n, float* hidden, std::string& err, bool final_norm) {
    if (!dense_ || !next_.empty() || hidden == nullptr || n <= 0 || n > opt_.max_cells) {
        err = "plain model: encode takes 1 .. max_cells tokens of a dense model on one GPU";
        return false;
    }
    if (!reset(err)) return false;
    if (opt_.encoder && !encoder_buffers(n, err)) return false;   // the prompt in one chunk (the K/V cells are shared)
    enc_out_ = hidden;
    enc_norm_ = final_norm;
    int32_t next = 0;
    const bool ok = prompt_one(tokens, n, 0, &next, nullptr, err);
    enc_out_ = nullptr;
    enc_norm_ = true;
    if (!ok && err.empty()) err = "plain model: the device cannot lend the prompt's buffers";
    return ok;
}

// A layer split's prompt: every stage's buffers for chunks as long as its GPU's tier lends them, then each chunk of the
// longest through the stages in turn, the rows handed on through pinned memory, a stage of shorter chunks reading them
// in its own.  Chunks of the shortest everywhere made the RTX 4070's 1K the B70's too, which then read every missed
// expert four times as often: GLM-5.3-Flash's 4K prompt 78 tok/s (B70 + 4070, 2026-10-20)
bool PlainModel::prompt_split(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits,
                              std::string& err) {
    const std::vector<PlainModel*> c = chain();
    int64_t T = n;
    for (PlainModel* st : c) {
        if (st->opt_.prompt_chunk <= 0 || n <= 0 || st->prompt_host_ == nullptr ||
            (st->hand_host_ == nullptr && !(st->first() && st->last()))) {
            err.clear();
            return false;
        }
        T = std::min(T, st->opt_.prompt_chunk);
    }
    if (pos < 0 || pos + n > opt_.max_cells) { err = "plain prompt: past the context"; return false; }
    auto give_back = [&](size_t upto) {
        std::string e2;
        for (size_t i = 0; i < upto; ++i) {
            const OnDevice on(c[i]->dev_);
            (void) c[i]->prompt_return(e2);
        }
    };
    int64_t longest = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        const OnDevice on(c[i]->dev_);
        if (!c[i]->prompt_borrow(T, err)) {
            give_back(i);
            if (c[i]->borrow_short_) {   // as prompt_one: the windows read the prompt
                static bool said = false;
                if (!said) std::fprintf(stderr, "xestrata plain: %s: prompts are read through the decode windows\n", err.c_str());
                said = true;
                err.clear();
            }
            return false;
        }
        longest = std::max(longest, c[i]->pb_.T);
    }
    bool ok = true;
    if (opt_.pipeline_windows == 0) {
        for (int64_t c0 = 0; ok && c0 < n; c0 += longest) {
            const float* hand = nullptr;
            for (PlainModel* st : c) {
                const OnDevice on(st->dev_);
                if (!st->prompt_chunks(tokens, n, pos, out, logits, err, c0, c0 + longest, hand)) { ok = false; break; }
                hand = st->hand_host_;
            }
            if (ok && prompt_progress_) prompt_progress_(std::min(n, c0 + longest), n);
        }
    } else {
        // the chunks pipelined over the stages (--pipeline-windows, as Qwen3.8-Flash-Next's prefill chain): stage s
        // reads chunk j while stage s + 1 reads chunk j - 1, a thread each.  Each stage reads its chunks in order, the
        // same rows as the serial loop, so the same state; a chunk's rows handed on are copied out between the ticks
        std::vector<int64_t> starts;
        for (int64_t c0 = 0; c0 < n; c0 += longest) starts.push_back(c0);
        const size_t J = starts.size(), S = c.size();
        std::vector<std::vector<float>> hands(J);
        std::vector<std::string> errs(S);
        std::vector<char> failed(S, 0);
        auto run = [&](size_t s, size_t j) {
            PlainModel* st = c[s];
            const OnDevice on(st->dev_);
            if (!st->prompt_chunks(tokens, n, pos, out, logits, errs[s], starts[j], starts[j] + longest,
                                   s == 0 ? nullptr : hands[j].data())) {
                failed[s] = 1;
                return;
            }
            if (s + 1 < S) {
                const int64_t rows = std::min(n, starts[j] + longest) - starts[j];
                hands[j].assign(st->hand_host_, st->hand_host_ + (size_t) rows * (size_t) st->hand_w_);
            }
        };
        for (size_t tick = 0; ok && tick < J + S - 1; ++tick) {
            piping_ = true;
            {
                std::vector<std::thread> later;
                for (size_t s = 1; s < S; ++s)
                    if (tick >= s && tick - s < J) later.emplace_back(run, s, tick - s);
                if (tick < J) run(0, tick);
                for (auto& t : later) t.join();
            }
            piping_ = false;
            for (size_t s = 0; s < S && ok; ++s)
                if (failed[s]) {
                    err = errs[s];
                    ok = false;
                }
            if (ok && tick >= S - 1) {
                const size_t j = tick - (S - 1);
                std::vector<float>().swap(hands[j]);
                if (prompt_progress_) prompt_progress_(std::min(n, starts[j] + longest), n);
            }
        }
    }
    std::string err2;
    for (PlainModel* st : c) {
        const OnDevice on(st->dev_);
        if (!st->prompt_return(err2) && ok) { err = err2; ok = false; }
    }
    if (!ok) return false;
    for (PlainModel* st : c) {   // each tier's counts from the prompt, its free slots filled (prompt_one)
        const OnDevice on(st->dev_);
        if (!st->prompt_settle(err)) return false;
    }
    return true;
}

bool PlainModel::prompt_one(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits,
                            std::string& err) {
    if (opt_.prompt_chunk <= 0 || n <= 0 || prompt_host_ == nullptr) { err.clear(); return false; }
    if (pos < 0 || pos + n > opt_.max_cells) { err = "plain prompt: past the context"; return false; }
    const auto tb = std::chrono::steady_clock::now();
    // one chunk the prompt's length, up to prompt_chunk (prompt_borrow may take fewer)
    if (!prompt_borrow(std::min<int64_t>(n, opt_.prompt_chunk), err)) {
        // a report on WSL 2 (an RTX 2050, 4 GB): the tier lent too little for even a short chunk and every request
        // failed; the windows read the prompt instead (slower, but it answers), said once
        if (borrow_short_) {
            static bool said = false;
            if (!said) std::fprintf(stderr, "xestrata plain: %s: prompts are read through the decode windows\n", err.c_str());
            said = true;
            err.clear();
        }
        return false;
    }
    static const bool profile = strata::env_get("PROMPT_PROFILE") != nullptr;
    if (profile)
        std::fprintf(stderr, "xestrata plain: borrowing the tier's slots %.1f ms\n",
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb).count());
    auto since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    auto tc = std::chrono::steady_clock::now();
    const bool ok = prompt_chunks(tokens, n, pos, out, logits, err, 0, n, nullptr);
    const double chunks_ms = since(tc);
    tc = std::chrono::steady_clock::now();
    std::string err2;
    if (!prompt_return(err2) && ok) { err = err2; return false; }
    if (profile)
        std::fprintf(stderr, "xestrata plain: the chunks %.1f ms, giving the tier its slots back %.1f ms\n", chunks_ms, since(tc));
    const auto t0 = std::chrono::steady_clock::now();
    if (ok && !prompt_settle(err)) return false;
    if (profile)
        std::fprintf(stderr, "xestrata plain: the tier after the prompt: %.1f ms\n",
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return ok;
}

bool PlainModel::prompt_chunks(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits,
                               std::string& err, int64_t c_begin, int64_t c_end, const float* hand_in) {
    if (glm_) return prompt_chunks_glm5(tokens, n, pos, out, logits, err, c_begin, c_end, hand_in);
    PromptBufs& P = pb_;
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, NH = s.n_head, HD = s.key_length, NKV = nkv_, HV = s.ssm_dt_rank, HK = s.ssm_n_group;
    const int64_t E = s.n_expert, K = s.n_expert_used;
    const int64_t FS = dense_ ? s.n_ff : s.n_ff_shexp, qw = (has_gate_ ? 2 : 1) * NH * HD;
    const float eps = (float) s.rms_eps;
    const bool loop = s.arch == ModelArch::SenseNovaLoop;
    const k::QsaShapes qs = attn_shapes();
    strata::gpu::Stream cs = cs_;
    prefill::Gemm& gm = prompt_gemm().gemm;
    auto f16 = [&](const float* x, int64_t rows, int64_t cols) { kernels::f32_to_f16_bulk(x, P.x16, rows * cols, cs); };
    // a dense product from float rows x (ldx a row) in int8 below dense_native_rows_ rows where the weight is the int8
    // products'; false: the caller takes Gemm::native from the FP16 copy
    const bool dense_on = dense_int8_env() && prefill::mmq::chosen();   // as for the experts: PlainModel::prompt_int8
    // an encoder's adapters (set_product_hook): the products from their FP16 inputs, each followed by the hook
    const bool hooked = hook_ != nullptr && enc_out_ != nullptr;
    auto dense = [&](const W& w, const float* x, int64_t ldx, float* y, int64_t T, int64_t ldy = 0) {
        return !hooked && T < dense_native_rows_ && prompt_dense_int8(w, x, ldx, y, T, ldy);
    };
    auto proj = [&](const W& w, float* y, int64_t T, int64_t ldy = 0, bool half_sums = false) {
        gm.native_llm(P.x16, w.type, w.data, y, T, w.n_out, w.n_in, dense_, ldy, half_sums);
    };
    // the routed experts in their quantized blocks (int8 products, iq_mmq: on the matrix engines, else by dp4a where the
    // FP16 products have none either) where every layer's types are theirs (prompt_int8, decided with the buffers) and
    // a chunk's layer has few enough rows an expert (prompt_routed), else dequantized to FP16
    const bool use_mmq = P.int8;
    // the CPU's share of the missed experts needs the split's times (measured at load from pinned RAM, on the decode
    // window's kernels: the int8 and the FP16 path take them alike); STRATA_PLAIN_CPU_SHARE=0, or
    // --prefill-cpu-share 0 (STRATA_PREFILL_CPU_SHARE, as Qwen3.8-Flash-Next's): every expert on the GPU
    static const bool cpu_env = [] {
        const char* v = strata::env_get("PLAIN_CPU_SHARE");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const double share_setting = strata::prefill::cpu_share_setting();
    const bool cpu_share = cpu_env && share_setting != 0.0 &&
                           (share_setting > 0.0 || (split_.c > 0.0 && split_.p > 0.0)) && P.cpu_cap > 0;
    static const bool prompt_attn_on = [] {   // STRATA_PLAIN_PROMPT_ATTN=0: the decode's attention kernel
        const char* v = strata::env_get("PLAIN_PROMPT_ATTN");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    PromptGemm& pg = prompt_gemm();
    const int64_t rows_id = P.T * std::max<int64_t>(K, 1);   // the identity's rows: the routed ones, or a dense chunk's
    if (use_mmq && !pg.identity) {
        prefill::mmq::iota(P.identity, rows_id, cs);
        pg.identity = true;
    }
    // STRATA_PROMPT_PROFILE=1: the time of each stage, the stream synchronized after each (slower; for finding where it
    // goes): 0 the GDN layers, 4 the attention layers, 1 the router and the host's grouping, 2 the experts, 3 the shared
    // expert and the combine
    static const bool profile = strata::env_get("PROMPT_PROFILE") != nullptr;
    double stage_ms[10] = {};   // 5-9: the GDN layers' parts (stage 0 is their sum)
    int64_t cpu_n = 0, cpu_rows = 0, miss_n = 0;   // for the profile: the CPU's experts and rows, the missed experts
    auto mark = [&](int stage, std::chrono::steady_clock::time_point& t) {
        if (!profile) return;
        strata::gpu::stream_sync(cs);
        const auto now = std::chrono::steady_clock::now();
        stage_ms[stage] += std::chrono::duration<double, std::milli>(now - t).count();
        t = now;
    };
    auto t_mark = std::chrono::steady_clock::now();
    try {
        // the chunk's routes, every MoE layer's, for the adaptive tier's counts (below, after the chunk)
        const int64_t L = n_moe_;
        std::vector<int32_t> routes(disp_.usage.empty() ? 0 : (size_t) (L * P.T * K));
        const int64_t c_last = std::min(n, c_end);
        if (c_last > c_begin) ring_advance(pos + c_begin, c_last - c_begin);
        // the position after the last staging copies (an event of this call's)
        struct Staged {
            strata::gpu::Event* e = nullptr;
            bool set = false;
            ~Staged() { if (e != nullptr) strata::gpu::event_destroy(e); }
            bool mark(strata::gpu::Stream s) {
                if (e == nullptr && !strata::gpu::event_create(&e)) return false;
                set = strata::gpu::event_record(e, s);
                return set;
            }
            bool wait() { return !set || strata::gpu::event_sync(e); }
        } staged;
        if (!first())   // the rows handed on, into this stage's own pinned memory (and its rows out from there)
            std::memcpy(hand_host_, hand_in, (size_t) ((c_last - c_begin) * hand_w_) * 4);
        for (int64_t c0 = c_begin; c0 < c_last; c0 += P.T) {
            enc_row0_ = c0;   // the chunk's first row (block_hook_)
            const int64_t T = std::min<int64_t>(P.T, c_last - c0), p0 = pos + c0;
            float* hand = hand_host_ != nullptr ? hand_host_ + (c0 - c_begin) * hand_w_ : nullptr;
            // the staging buffers are pinned host memory the last chunk's copies read: written again only once those
            // copies ran (a dense model's chunk has nothing else that waits on the host, and its next inputs went
            // over the last chunk's before the GPU read them: PLaMo 3 Translate and Qwen3.5 through the FP16 and the
            // decode kernels read garbage from the second chunk on, 2026-11-09)
            if (!staged.wait()) {
                err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                return false;
            }
            // the chunk's inputs: embedding rows (a later stage: the rows handed on), per-token step counts and rotary
            // positions
            if (first()) embed_ahead(tokens + c0, T);
            for (int64_t t = 0; t < T; ++t) {
                const int32_t tok = tokens[c0 + t];
                if (tok < 0 || tok >= n_vocab_ || (first() && !embed(tok, p0 + t, P.emb + t * N))) {
                    err = "plain prompt: a token is out of the vocabulary";
                    return false;
                }
                k::qsa_step_fill(P.h_steps + t * k::kStepCount, p0 + t, qs);
                for (int64_t hd = 0; hd < NH; ++hd) P.h_pos[t * NH + hd] = (int32_t) (p0 + t);
                for (int64_t hd = 0; hd < NKV; ++hd) P.h_pos[T * NH + t * NKV + hd] = (int32_t) (p0 + t);
            }
            if (!strata::gpu::copy_async(loop && !first() ? P.g_hs : P.x, first() ? P.emb : hand,
                                         (size_t) (T * (loop && !first() ? 2 * N : N)) * 4, cs) ||
                !strata::gpu::copy_async(P.steps, P.h_steps, (size_t) (T * k::kStepCount) * 4, cs) ||
                !strata::gpu::copy_async(P.pos_q, P.h_pos, (size_t) (T * NH) * 4, cs) ||
                !strata::gpu::copy_async(P.pos_k, P.h_pos + T * NH, (size_t) (T * NKV) * 4, cs)) {
                err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                return false;
            }
            if (loop && !first() && !s.mhc_active_before(lb_)) k::glm5_hc_mean(P.g_hs, P.x, N, T, cs, 2);
            P.h_dense_bounds[0] = 0;
            P.h_dense_bounds[1] = (int32_t) T;
            if (!strata::gpu::copy_async(P.dense_bounds, P.h_dense_bounds, 2 * sizeof(int32_t), cs)) {
                err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                return false;
            }
            if (!staged.mark(cs)) {
                err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                return false;
            }
            if (dense_on && !pg.identity) {   // the dense products' row map (the experts' sets it too)
                prefill::mmq::iota(P.identity, rows_id, cs);
                pg.identity = true;
            }
            // from the normed rows: int8 where it can, else FP16 (x16, written once a layer if any product needs it),
            // its sums in FP16 where that was measured faster (a norm's rows keep them in FP16's range)
            auto proj_hn = [&](const W& w, float* y, int64_t ldy = 0) {
                if (!dense(w, P.hn, N, y, T, ldy)) proj(w, y, T, ldy, dense_half_sums_);
            };
            // the products whose sums are read once, by the residual's add or the SwiGLU, in FP16 as they come where
            // their FP16 sums are taken (not widened into FP32 first: a pass over every sum, the RTX 4070's Qwen3.5-9B
            // prompt 75 ms of 1.0 s at 4K, 2026-11-07); y16 in the FP32 buffer's room
            // (not with plamo3's norms after the mixer and the FFN: their inputs are the products' FP32 rows)
            const bool half_rows = dense_half_sums_ && !hooked && !plamo_ && !loop && T >= dense_native_rows_ &&
                                   gm.half_ok(cs);
            auto proj_h = [&](const W& w, void* y) {
                gm.native_half(P.x16, w.type, w.data, static_cast<uint16_t*>(y), T, w.n_out, w.n_in);
            };
            // the last token's logits, by the decode's head (an encoder's chunks write their rows instead, below)
            auto last_logits = [&] {
                if (c0 + T < n || enc_out_ != nullptr) return;
                k::native_qsa_rms_norm_weighted(P.x + (T - 1) * N, out_norm_, P.hn, (int) N, 1, eps, cs);
                if (head_.type == 30) {   // a BF16 head, as the decode reads it
                    k::bf16_gemv_fp32_mmvf_multi(P.hn, head_.n_in, static_cast<const uint16_t*>(head_.data), logits_,
                                                 head_.n_out, head_.n_in, head_.n_out, 1, cs);
                } else if (head_.type == 0) {
                    k::f32_gemv_fp32_mmvf_multi(P.hn, head_.n_in, static_cast<const float*>(head_.data), logits_,
                                                head_.n_out, head_.n_in, head_.n_out, 1, cs);
                } else {
                    k::native_quantize_q8_1(P.hn, xq_, head_.n_in, 1, cs);
                    k::native_mmvq(head_.type, head_.data, xq_, logits_, head_.n_in, head_.n_out, 1, cs);
                }
                pick(logits_, 1, p0 + T - 1, m_out_, cs);   // the last token's position, as forward counts
            };
            // the MTP layer after the trunk, its K/V a cell ahead: row t pairs the trunk's normed row t with the token
            // after it (the prompt's last row with a stand-in, rewritten by spec_start once the token is picked)
            const bool mtp_chunk = has_mtp_ && p0 + T < opt_.max_cells;
            for (int64_t l = lb_; l < le_ + (mtp_chunk ? 1 : 0); ++l) {
                const bool is_mtp = l == s.n_layer;
                if (is_mtp) {
                    last_logits();
                    k::native_qsa_rms_norm_weighted(P.x, out_norm_, P.hn, (int) N, (int) T, eps, cs);
                    if (c0 + T >= n &&
                        !strata::gpu::copy_async(mtp_h_, P.hn + (T - 1) * N, (size_t) N * 4, cs)) {
                        err = std::string("plain prompt: the MTP layer's input: ") + strata::gpu::last_error();
                        return false;
                    }
                    k::native_qsa_rms_norm_weighted(P.hn, mtp_hnorm_, P.mix, (int) N, (int) T, eps, cs);
                    if (!staged.wait()) {   // the staging again (above)
                        err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                        return false;
                    }
                    for (int64_t t = 0; t < T; ++t) {
                        const int32_t tok = c0 + t + 1 < n ? tokens[c0 + t + 1] : tokens[n - 1];
                        if (!embed(tok, p0 + 1 + t, P.emb + t * N)) {
                            err = "plain prompt: a token is out of the vocabulary";
                            return false;
                        }
                        k::qsa_step_fill(P.h_steps + t * k::kStepCount, p0 + 1 + t, qs);
                        for (int64_t hd = 0; hd < NH; ++hd) P.h_pos[t * NH + hd] = (int32_t) (p0 + 1 + t);
                        for (int64_t hd = 0; hd < NKV; ++hd) P.h_pos[T * NH + t * NKV + hd] = (int32_t) (p0 + 1 + t);
                    }
                    if (!strata::gpu::copy_async(P.x, P.emb, (size_t) (T * N) * 4, cs) ||
                        !strata::gpu::copy_async(P.steps, P.h_steps, (size_t) (T * k::kStepCount) * 4, cs) ||
                        !strata::gpu::copy_async(P.pos_q, P.h_pos, (size_t) (T * NH) * 4, cs) ||
                        !strata::gpu::copy_async(P.pos_k, P.h_pos + T * NH, (size_t) (T * NKV) * 4, cs) ||
                        !staged.mark(cs)) {
                        err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                        return false;
                    }
                    k::native_qsa_rms_norm_weighted(P.x, mtp_enorm_, P.hn, (int) N, (int) T, eps, cs);
                    k::concat_rows(P.qfull, P.hn, P.mix, N, T, cs);
                    if (!dense(mtp_eh_, P.qfull, 2 * N, P.x, T)) {
                        f16(P.qfull, T, 2 * N);
                        proj(mtp_eh_, P.x, T);
                    }
                }
                if (!is_mtp) bind_block(l);   // a streamed block's matrices
                const Block& b = is_mtp ? mtp_ : blocks_[(size_t) l];
                auto hook = [&](ProductHook::Which which, const W& w, const uint16_t* x, float* y, int64_t ldy = 0) {
                    if (hooked && !is_mtp) hook_->after(l - lb_, which, x, T, y, ldy > 0 ? ldy : w.n_out);
                };
                const int64_t pa = is_mtp ? p0 + 1 : p0;   // the rows' first position
                const bool mhc = b.hc_attn_fn.data != nullptr;
                auto route = [&](const W& fn, const float* bias) {
                    k::f32_gemm_rows(P.g_hs, 2 * N, static_cast<const float*>(fn.data), P.g_mixes, 8, 2 * N, 8, T, cs);
                    k::glm5_hc_pre(P.g_hs, P.g_mixes, nullptr, bias, 1e-6f, 0, (int) s.mhc_iterations,
                                   P.g_y, P.g_post, P.g_comb, N, T, cs, 2, (float) s.mhc_pre_scale);
                };
                if (mhc) {
                    if (s.physical_layer(l) == s.mhc_start) k::glm5_hc_init(P.x, P.g_hs, N, T, cs, 2);
                    route(b.hc_attn_fn, b.hc_attn_bias);
                }
                k::native_qsa_rms_norm_weighted(mhc ? P.g_y : P.x, b.attn_norm, P.hn, (int) N, (int) T, eps, cs);
                f16(P.hn, T, N);
                if (b.recurrent) {
                    auto part = [&](int stage) {   // the profile's GDN parts, summed into stage 0 as well
                        if (!profile) return;
                        const auto t0 = t_mark;
                        mark(stage, t_mark);
                        if (stage != 0) stage_ms[0] += std::chrono::duration<double, std::milli>(t_mark - t0).count();
                    };
                    part(0);   // the norm and its FP16 copy
                    proj_hn(b.qkv, P.qkv);
                    proj_hn(b.z, P.z);
                    hook(ProductHook::kQkv, b.qkv, P.x16, P.qkv);
                    hook(ProductHook::kZ, b.z, P.x16, P.z);
                    part(5);
                    for (const auto& [w, y] : {std::pair<const W*, float*>{&b.alpha, P.ab}, {&b.beta, P.ab + HV}}) {
                        if (w->type == 0)   // F32 (Qwen3.6), from the FP32 rows as the router
                            k::f32_gemm_rows(P.hn, N, static_cast<const float*>(w->data), y, 2 * HV, w->n_in,
                                             w->n_out, T, cs);
                        else
                            proj(*w, y, T, 2 * HV);
                        hook(w == &b.alpha ? ProductHook::kAlpha : ProductHook::kBeta, *w, P.x16, y, 2 * HV);
                    }
                    part(6);
                    prefill::gdn_gates(P.ab, b.dt, b.a, P.gate, P.beta, T, cs, HK, HV);
                    prefill::gdn_conv(gdn_conv_[(size_t) b.gdn], P.qkv, b.conv, P.h, T, eps, cs, HK, HV);
                    part(7);
                    prefill::gdn_recurrence(gdn_state_[(size_t) b.gdn], P.h, P.gate, P.beta, P.z, b.ssm_norm, eps, P.y,
                                            P.y16, T, cs, HK, HV, /*silu=*/true);
                    part(8);
                    // (y16, the normed and gated output, exists only in FP16: no float rows to round to int8; normed,
                    // its sums in FP16 as proj_hn's)
                    if (half_rows) {
                        gm.native_half(P.y16, b.ssm_out.type, b.ssm_out.data, reinterpret_cast<uint16_t*>(P.mix), T,
                                       b.ssm_out.n_out, b.ssm_out.n_in);
                    } else {
                        gm.native(P.y16, b.ssm_out.type, b.ssm_out.data, P.mix, T, b.ssm_out.n_out, b.ssm_out.n_in, 0,
                                  dense_half_sums_);
                        hook(ProductHook::kSsmOut, b.ssm_out, P.y16, P.mix);
                    }
                    part(9);
                } else {
                    const int64_t a = b.attn;
                    proj_hn(b.q, P.qfull);
                    proj_hn(b.k, P.kraw);
                    proj_hn(b.v, P.vcur);
                    hook(ProductHook::kQ, b.q, P.x16, P.qfull);
                    hook(ProductHook::kK, b.k, P.x16, P.kraw);
                    hook(ProductHook::kV, b.v, P.x16, P.vcur);
                    // the norm and RoPE take at most 65535 rows a launch: the chunk's heads in token slices
                    auto norm_rope = [&](const float* in, int64_t in_row, const float* gamma, float* out, int64_t heads,
                                         const int32_t* pos) {
                        const int64_t per = 65535 / heads;
                        for (int64_t t0 = 0; t0 < T; t0 += per)
                            k::native_qsa_rms_norm_rope(in + t0 * heads * in_row, (int) in_row, gamma, out + t0 * heads * HD,
                                                        (int) (std::min(per, T - t0) * heads), (int) HD,
                                                        (int) s.rope_dims, eps, rope_of(b), pos + t0 * heads, cs);
                    };
                    norm_rope(P.qfull, qw / NH, b.q_norm, P.qcur, NH, P.pos_q);
                    norm_rope(P.kraw, HD, b.k_norm, P.kcur, NKV, P.pos_k);
                    // --kv q4_0: K, V and the queries rotated (kv_q4.hpp); k8v4: V alone; tcq4/5: all three with
                    // the signs (kv_tcq.hpp); the output back below
                    if (opt_.kv_tcq != 0) {
                        k::fwht256_signed(P.kcur, T * NKV, false, cs);
                        k::fwht256_signed(P.vcur, T * NKV, false, cs);
                        k::fwht256_signed(P.qcur, T * NH, false, cs);
                    } else if (opt_.kv_q4 == 1) {
                        k::fwht256_inplace_cuda(P.kcur, T * NKV, cs);
                        k::fwht256_inplace_cuda(P.vcur, T * NKV, cs);
                        k::fwht256_inplace_cuda(P.qcur, T * NH, cs);
                    } else if (opt_.kv_q4 == 2) {
                        k::fwht256_inplace_cuda(P.vcur, T * NKV, cs);
                    }
                    const bool streamed = kv_streamed((size_t) a);
                    k::QsaAttnPools pools = kv_pools((size_t) a);
                    if (streamed) {
                        // --kv-resident: stage this layer's cells [0, pa) from the host copy into the staging pool
                        // (the identity layout), then the chunk's cells go to the host copy, the staging pool and the
                        // VRAM ring; the prompt attention reads the staging pool (Qwen3.8-Flash-Next's prompt path)
                        const int fmt = opt_.kv_q4 == 1 ? k::kKvQ4 : opt_.kv_q4 == 2 ? k::kKvHybrid
                                      : opt_.kv_int8 ? k::kKvInt8 : k::kKvF16;
                        k::QsaAttnPools stage;
                        stage.k_pool = P.kv_stage.k_pool; stage.v_pool = P.kv_stage.v_pool;
                        stage.k_q = P.kv_stage.k_q; stage.v_q = P.kv_stage.v_q;
                        stage.k_scale = P.kv_stage.k_scale; stage.v_scale = P.kv_stage.v_scale;
                        stage.k_q4 = P.kv_stage.k_q4; stage.v_q4 = P.kv_stage.v_q4;
                        stage.page_table = page_table_;   // the identity: the staging pool is a fully resident layout
                        k::kv_stage_from_host(stage, kv_host_[(size_t) a], fmt, (pa + qs.page_size - 1) / qs.page_size,
                                              qs, cs);
                        // the chunk's cells reach the host copy by DMA from the staging pool, not by the append
                        // kernel's stores over the link (core/kv_host_dma.hpp, XESTRATA_KV_HOST_DMA)
                        const bool host_by_dma = opt_.kv_tcq == 0 &&
                                                 strata::core::kv_host_dma_use(strata::core::kv_host_dma(), true, P.kv_stage.present());
                        kv_append_cells((size_t) a, -1, P.steps, (int) T, P.kcur, P.vcur, cs, &P.kv_stage, host_by_dma);
                        if (host_by_dma)
                            k::kv_unstage_to_host(stage, kv_host_[(size_t) a], fmt, pa / qs.page_size,
                                                  (pa + T + qs.page_size - 1) / qs.page_size, qs, cs);
                        pools = stage;
                    } else {
                        kv_append_cells((size_t) a, -1, P.steps, (int) T, P.kcur, P.vcur, cs);
                    }
                    // the prompt attention (prompt_attn: K and V read once a block of queries), else the decode's
                    // kernel: every query reads the cells up to its own; the grid as wide as the chunk's last context (not
                    // the cache's capacity: no graph holds it), as many queries a launch as the scratch takes
                    // a chunk whose last row is within the indexer's budget selects every cell: its keys for the later
                    // chunks, the attention as the dense one's
                    const bool qsa = sparse_qsa_ && (size_t) a < qsa_weights_.size() && qsa_weights_[(size_t) a].q.data;
                    if (qsa && pa + T <= qs.idx_top_k) {
                        const QsaWeights& w = qsa_weights_[(size_t) a];
                        k::f32_gemm_rows(P.hn, N, static_cast<const float*>(w.k.data), P.qsa_raw, 128, N, 128, T, cs);
                        k::native_qsa_indexer_append_batch(P.qsa_raw, T, pa, 0, w.k_norm, eps, qsa_index_[(size_t) a], qs,
                                                           opt_.max_cells, rope_of(b), cs, w.block_norm, true);
                    }
                    if (qsa && pa + T > qs.idx_top_k) {
                        const QsaWeights& w = qsa_weights_[(size_t) a];
                        const auto& ib = qsa_index_[(size_t) a];
                        k::f32_gemm_rows(P.hn, N, static_cast<const float*>(w.q.data), P.qsa_query, 512, N, 512, T, cs);
                        k::f32_gemm_rows(P.hn, N, static_cast<const float*>(w.k.data), P.qsa_raw, 128, N, 128, T, cs);
                        for (int64_t t = 0; t < T; t += 65535 / 4)
                            k::native_qsa_rms_norm_rope(P.qsa_query + t * 512, 128, w.q_norm,
                                                        P.qsa_query + t * 512, (int) (std::min<int64_t>(65535 / 4, T - t) * 4),
                                                        128, 64, eps, rope_of(b), P.pos_k + t * 4, cs, true);
                        k::native_qsa_indexer_append_batch(P.qsa_raw, T, pa, 0, w.k_norm, eps, ib, qs,
                                                           opt_.max_cells, rope_of(b), cs, w.block_norm, true);
                        const int64_t blocks = opt_.max_cells / 4 + 1;
                        const int64_t scratch = (int64_t) k::qsa_decode_attn_scratch_floats(2052, qs);
                        const int64_t per = std::max<int64_t>(1, std::min<int64_t>(64, opt_.max_t * attn_scratch_floats_ / scratch));
                        for (int64_t t = 0; t < T; t += per) {
                            const int64_t nq = std::min<int64_t>(per, T - t);
                            int32_t* steps = P.steps + t * k::kStepCount;
                            // the tiled FP32 scorer (the RX 9060 XT, a 32K prompt: 3% faster)
                            if (!k::qsa_block_scores_tc(ib.pooled, ib.dead, P.qsa_query + t * 512, steps, nq, blocks,
                                                        qs, P.qsa_scores, cs, 0))
                                k::qsa_block_scores(ib.pooled, ib.dead, P.qsa_query + t * 512, steps, nq, blocks, qs,
                                                    P.qsa_scores, cs);
                            k::qsa_block_topk_with_current(P.qsa_scores, steps, nq, blocks, 2052, qs, P.qsa_ids, cs);
                            if (!k::flash_attn_select(P.qcur + t * NH * HD, pools, P.qsa_ids, steps, 2052, qs,
                                                      P.attn + t * NH * HD, nq, cs, attn_scratch_,
                                                      (size_t) opt_.max_t * (size_t) attn_scratch_floats_ * 4))
                                k::qsa_decode_attn_batch(P.qcur + t * NH * HD, pools, P.qsa_ids, steps, 2052, qs,
                                                         attn_scratch_, P.attn + t * NH * HD, nq, cs);
                        }
                    } else {
                        const int64_t cap = std::min<int64_t>(opt_.max_cells, pa + T);
                        const int64_t per_q = (int64_t) k::qsa_decode_attn_scratch_floats(cap, qs);
                        const int64_t nq_max = std::max<int64_t>(1, std::min<int64_t>((int64_t) opt_.max_t * attn_scratch_floats_ / per_q, 4096));
                        const bool batched =
                            prompt_attn_on && k::prompt_attn(P.qcur, pools, pa, T, qs, P.attn, cs, window_of(b));
                        for (int64_t t = 0; !batched && t < T; t += nq_max) {
                            const int nq = (int) std::min<int64_t>(nq_max, T - t);
                            k::qsa_dense_attn_batch(P.qcur + t * NH * HD, pools, P.steps + t * k::kStepCount, cap, qs,
                                                    attn_scratch_, P.attn + t * NH * HD, nq, cs, window_of(b));
                        }
                    }
                    if (opt_.kv_tcq != 0) k::fwht256_signed(P.attn, T * NH, true, cs);
                    else if (opt_.kv_q4 != 0) k::fwht256_inplace_cuda(P.attn, T * NH, cs);
                    const float* ao = P.attn;
                    if (has_gate_) {
                        k::native_qsa_gate_apply(P.attn, P.qfull, P.attn32, (int) (T * NH), (int) HD, cs);
                        ao = P.attn32;
                    }
                    if (half_rows) {
                        f16(ao, T, NH * HD);
                        proj_h(b.o, P.mix);
                    } else if (!dense(b.o, ao, NH * HD, P.mix, T)) {
                        f16(ao, T, NH * HD);
                        proj(b.o, P.mix, T, 0, dense_half_sums_);   // averages of V's rows: in FP16's range as well
                        hook(ProductHook::kO, b.o, P.x16, P.mix);
                    }
                }
                if (mhc) {
                    k::glm5_hc_post(P.mix, P.g_hs, P.g_post, P.g_comb, P.g_hs, N, T, cs, 2);
                    route(b.hc_ffn_fn, b.hc_ffn_bias);
                } else if (half_rows) {
                    prefill::add_h(P.x, reinterpret_cast<const uint16_t*>(P.mix), T * N, cs);
                } else if (b.attn_post != nullptr) {   // plamo3: the mixer's output normed before the add
                    k::native_qsa_rms_norm_weighted(P.mix, b.attn_post, P.hn, (int) N, (int) T, eps, cs);
                    k::add_inplace(P.x, P.hn, T * N, cs);
                } else {
                    k::add_inplace(P.x, P.mix, T * N, cs);
                }
                mark(b.recurrent ? 0 : 4, t_mark);

                if (dense_) {   // qwen35's FFN: gate and up from the normed rows, the SwiGLU, down; int8 where it can
                    k::native_qsa_rms_norm_weighted(mhc ? P.g_y : P.x, b.post_norm, P.hn, (int) N, (int) T, eps, cs);
                    f16(P.hn, T, N);
                    if (half_rows) {   // gate, up, the SwiGLU and down's sums for the residual all in FP16
                        proj_h(b.sh_gate, P.sh_gate);
                        proj_h(b.sh_up, P.sh_up);
                        prefill::swiglu_pair_h(reinterpret_cast<const uint16_t*>(P.sh_gate),
                                               reinterpret_cast<const uint16_t*>(P.sh_up), P.x16, T, cs, FS);
                        proj_h(b.sh_down, P.moe_out);
                        prefill::add_h(P.x, reinterpret_cast<const uint16_t*>(P.moe_out), T * N, cs);
                        steer(b, T, P.x, cs);
                        if (!is_mtp) tap_rows(l, P.x, 1, T, p0, false, cs);   // DFlash2's prompt taps
                        mark(3, t_mark);
                        if (!is_mtp) unbind_block(l);
                        continue;
                    }
                    proj_hn(b.sh_gate, P.sh_gate);
                    proj_hn(b.sh_up, P.sh_up);
                    hook(ProductHook::kGate, b.sh_gate, P.x16, P.sh_gate);
                    hook(ProductHook::kUp, b.sh_up, P.x16, P.sh_up);
                    if (!hooked && T < dense_native_rows_ && dense_int8_ok(b.sh_down)) {
                        // the SwiGLU in FP32 rows for the int8 product: the chunk as one row (the kernel is
                        // element-wise, and takes at most 8 rows)
                        k::native_swiglu_f32(P.sh_gate, P.sh_up, P.sh_gate, (int) (FS * T), 1, cs);
                        dense(b.sh_down, P.sh_gate, FS, P.moe_out, T);
                    } else {   // into FP16 (x16 is free again) for Gemm::native
                        prefill::swiglu_pair(P.sh_gate, P.sh_up, P.x16, T, cs, FS);
                        // its sums in FP16 too where they were measured faster, as llama.cpp's cuBLAS products of FP16
                        // weights take every product's (its default precision): the SwiGLU's rows are in FP16 already
                        proj(b.sh_down, P.moe_out, T, 0, dense_half_sums_);
                        hook(ProductHook::kDown, b.sh_down, P.x16, P.moe_out);
                    }
                    if (mhc) {
                        k::glm5_hc_post(P.moe_out, P.g_hs, P.g_post, P.g_comb, P.g_hs, N, T, cs, 2);
                        if (s.physical_layer(l) == s.mhc_end - 1) k::glm5_hc_mean(P.g_hs, P.x, N, T, cs, 2);
                    } else if (b.ffn_post != nullptr) {   // plamo3: the FFN's output normed before the add
                        k::native_qsa_rms_norm_weighted(P.moe_out, b.ffn_post, P.hn, (int) N, (int) T, eps, cs);
                        k::add_inplace(P.x, P.hn, T * N, cs);
                    } else {
                        k::add_inplace(P.x, P.moe_out, T * N, cs);
                    }
                    if (!mhc || s.physical_layer(l) == s.mhc_end - 1) {
                        steer(b, T, P.x, cs);
                        if (!is_mtp) tap_rows(l, P.x, 1, T, p0, false, cs);
                    }
                    mark(3, t_mark);
                    if (!is_mtp) unbind_block(l);
                    continue;
                }

                // the MoE
                k::native_qsa_rms_norm_weighted(P.x, b.post_norm, P.hn, (int) N, (int) T, eps, cs);
                k::f32_gemm_rows(P.hn, N, b.router, P.rlog, E, N, E, T, cs);
                k::router_top10(P.rlog, (int) T, (int) E, (int) K, P.ids, P.w, cs);
                f16(P.hn, T, N);
                if (has_shared_) {
                    proj_hn(b.sh_gate, P.sh_gate);
                    proj_hn(b.sh_up, P.sh_up);
                    k::f32_gemm_rows(P.hn, N, b.sh_gate_inp, P.sh_g, 1, N, 1, T, cs);
                }
                // the routed experts grouped by expert (prompt_routed), into parts in routing order
                const int64_t ml = is_mtp ? n_moe_ - 1 : s.moe_index(l);
                PromptRouted rst;
                if (!prompt_routed_tuned(ml, T, cpu_share, routes.empty() ? nullptr : routes.data() + ml * P.T * K,
                                   [&](int stage) { mark(stage, t_mark); }, rst, err))
                    return false;
                cpu_n += rst.cpu_n;
                cpu_rows += rst.cpu_rows;
                miss_n += rst.miss_n;
                if (has_shared_) {   // shared = down(silu(gate) * up): the SwiGLU into FP16 (x16 is free again)
                    prefill::swiglu_pair(P.sh_gate, P.sh_up, P.x16, T, cs, FS);
                    proj(b.sh_down, P.shared, T);
                }
                k::native_moe_combine_multi(P.parts, P.w, has_shared_ ? P.shared : nullptr, P.moe_out, N, K, (int) T,
                                            cs, has_shared_ ? P.sh_g : nullptr);
                k::add_inplace(P.x, P.moe_out, T * N, cs);
                steer(b, T, P.x, cs);
                if (!is_mtp) tap_rows(l, P.x, 1, T, p0, false, cs);
                mark(3, t_mark);
            }
            if (!prompt_taps_deliver(ptap_dev_, T, p0, cs, err)) return false;   // the chunk's DFlash2 taps
            if (enc_out_ != nullptr) {   // encode: every row through the final norm (or as it is), to the host
                if (enc_norm_) k::native_qsa_rms_norm_weighted(P.x, out_norm_, P.hn, (int) N, (int) T, eps, cs);
                if (!strata::gpu::stream_sync(cs) ||
                    !strata::gpu::copy(enc_out_ + (c0 - c_begin) * N, enc_norm_ ? P.hn : P.x, (size_t) (T * N) * 4)) {
                    err = std::string("plain prompt: the encoder's rows: ") + strata::gpu::last_error();
                    return false;
                }
            } else if (cpu_tail_ != nullptr) {   // --gpu-layers: the CPU's blocks over the chunk's rows, the head here
                if (!strata::gpu::copy_async(hand_host_, P.x, (size_t) (T * N) * 4, cs) || !strata::gpu::stream_sync(cs)) {
                    err = std::string("plain prompt: the rows to the CPU: ") + strata::gpu::last_error();
                    return false;
                }
                if (!cpu_tail_->run_blocks(hand_host_, T, p0, err)) return false;
                if (c0 + T >= n) {   // the last row back, for the logits
                    if (!strata::gpu::copy_async(P.x + (T - 1) * N, hand_host_ + (T - 1) * N, (size_t) N * 4, cs)) {
                        err = std::string("plain prompt: the rows from the CPU: ") + strata::gpu::last_error();
                        return false;
                    }
                    last_logits();
                }
            } else if (!last()) {   // the rows to the next stage
                if (loop && !s.mhc_active_before(le_)) k::glm5_hc_init(P.x, P.g_hs, N, T, cs, 2);
                if (!strata::gpu::copy_async(hand, loop ? P.g_hs : P.x, (size_t) (T * hand_w_) * 4, cs)) {
                    err = std::string("plain prompt: the rows to the next GPU: ") + strata::gpu::last_error();
                    return false;
                }
            } else if (!mtp_chunk) {
                last_logits();
            }
            if (!routes.empty()) {
                // the counts start from the prompt's routing at the tier's steady state (seed_usage), as Qwen3.8-
                // Flash-Next's; the rounds themselves are left to the decode
                for (int64_t ml : moe_local_)   // this stage's layers' routes (seed_usage at the prompt's end)
                    for (int64_t i = 0; i < T * K; ++i) ++prompt_routed_[(size_t) (ml * E + routes[(size_t) (ml * P.T * K + i)])];
            }
            // the whole prompt on one GPU (a layer split's chunks are told by prompt_split, through every stage)
            if (prompt_progress_ && c_begin == 0 && c_end >= n) prompt_progress_(c0 + T, n);
        }
    } catch (const std::exception& e) {
        err = std::string("plain prompt: ") + e.what();
        return false;
    }
    if (!strata::gpu::stream_sync(cs)) {
        err = std::string("plain prompt: ") + strata::gpu::last_error();
        return false;
    }
    if (c_end < n) return true;   // a layer split's chunk: the stage's part of the prompt goes on with the next
    windows_ += (n + opt_.max_t - 1) / opt_.max_t;
    mtp_h_set_ = has_mtp_ && pos + n < opt_.max_cells;   // the trunk's last row is in mtp_h_ for spec_start
    if (profile)
        std::fprintf(stderr, "xestrata plain: prompt of %lld tokens: GDN %.1f ms, attention %.1f, router and grouping %.1f, "
                             "experts %.1f, shared and combine %.1f\n", (long long) n, stage_ms[0], stage_ms[4], stage_ms[1],
                     stage_ms[2], stage_ms[3]);
    if (profile && stage_ms[0] > 0.0)
        std::fprintf(stderr, "xestrata plain: the GDN layers: in-projections %.1f ms, alpha and beta %.1f, gates and conv %.1f, "
                             "recurrence %.1f, out-projection %.1f\n", stage_ms[5], stage_ms[6], stage_ms[7], stage_ms[8],
                     stage_ms[9]);
    if (profile)
        std::fprintf(stderr, "xestrata plain: prompt experts not in VRAM %lld, of them on the CPU %lld (%lld rows)\n",
                     (long long) miss_n, (long long) cpu_n, (long long) cpu_rows);
    if (!last() && cpu_tail_ == nullptr) return true;
    *out = ((volatile int32_t*) h_out_)[0];
    if (logits != nullptr && !strata::gpu::copy(logits, logits_, (size_t) n_vocab_ * 4)) {
        err = "plain prompt: reading the logits failed";
        return false;
    }
    return true;
}

}  // namespace strata::core
