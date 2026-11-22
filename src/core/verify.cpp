// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/verify.cpp - see include/strata/core/verify.hpp.
#include "strata/core/verify.hpp"
#include "strata/core/auto_pick.hpp"
#include "strata/core/constraint_upload.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = strata::env_get("VERIFY_DEBUG") != nullptr;
// STRATA_ROUTE_RESIDENT=<margin logits> (+ STRATA_ROUTE_RESIDENT_RANKS=lo-hi, default 6-9; upstream 5df35dcb, 8bf203f5):
// residency-biased routing (opt-in, changes the output).
struct RouteResidentCfg;
RouteResidentCfg& route_resident_cfg();
struct RouteResidentCfg {
    float margin = 0.0f;
    int lo = 6, hi = 9;
    // #1578: one counter block per GPU.  A layer split inits a verifier per stage; the Verifier asks for its device's
    // block in init (outside graph capture) and keeps the pointer, so a launch never picks the block by the host
    // thread's current device.  nullptr (the allocation failed): the kernel runs without counters.
    std::mutex mu;
    std::map<int, unsigned long long*> blocks;
    unsigned long long* stats_on(int dev) {
        if (margin <= 0.0f || dev < 0) return nullptr;
        std::lock_guard<std::mutex> lk(mu);
        const auto it = blocks.find(dev);
        if (it != blocks.end()) return it->second;
        unsigned long long* p = nullptr;
        {
            const OnDevice on(dev);
            if (!strata::gpu::alloc_device(&p, 8 * sizeof(unsigned long long)) ||
                !strata::gpu::memset(p, 0, 8 * sizeof(unsigned long long))) {
                if (p != nullptr) strata::gpu::free(p);
                p = nullptr;
                std::fprintf(stderr, "route-resident: the counters could not be allocated on GPU %d; it runs without them\n", dev);
            }
        }
        if (blocks.empty()) {
            std::atexit([] {
                RouteResidentCfg& c = route_resident_cfg();
                for (const auto& kv : c.blocks) {
                    if (kv.second == nullptr) continue;
                    unsigned long long h[8] = {};
                    const OnDevice on(kv.first);
                    if (!strata::gpu::device_sync() || !strata::gpu::copy(h, kv.second, sizeof(h))) continue;
                    char who[32] = "";
                    if (c.blocks.size() > 1) std::snprintf(who, sizeof who, " (GPU %d)", kv.first);
                    std::fprintf(stderr, "route-resident%s: margin=%g ranks=%d-%d windows T>8 (prompt reads): tail_nonres=%llu "
                                         "swaps=%llu (%.1f%%) nonres_entries before=%llu after=%llu\n",
                                 who, c.margin, c.lo, c.hi, h[0], h[1], h[0] ? 100.0 * (double) h[1] / (double) h[0] : 0.0, h[2], h[3]);
                    std::fprintf(stderr, "route-resident%s: margin=%g ranks=%d-%d windows T<=8 (decode): tail_nonres=%llu "
                                         "swaps=%llu (%.1f%%) nonres_entries before=%llu after=%llu\n",
                                 who, c.margin, c.lo, c.hi, h[4], h[5], h[4] ? 100.0 * (double) h[5] / (double) h[4] : 0.0, h[6], h[7]);
                }
            });
        }
        blocks[dev] = p;
        return p;
    }
};
// STRATA_ROUTE_PRIOR=<lambda> (upstream 2de9ae06, opt-in, changes the output; router.cpp).  STRATA_ROUTE_PRIOR_COUNT=picked
// feeds the expert adaptation the edited picks, as upstream does; by default it counts the router's own, so the cache
// follows the text and not the bias it causes (a loop that could lock a long session onto the same experts).
struct RoutePriorCfg {
    float lambda = 0.0f;
    bool count_router = true;
    std::mutex mu;
    std::map<int, unsigned long long*> blocks;   // one counter block per GPU, as RouteResidentCfg's
    unsigned long long* stats_on(int dev);
    static void report();
};
RoutePriorCfg& route_prior_cfg() {
    static RoutePriorCfg c;
    static const bool parsed = [] {
        const auto num = [](const char* name, double dflt) {
            const char* v = strata::env_get(name);
            return v != nullptr && *v != '\0' ? std::strtod(v, nullptr) : dflt;
        };
        c.lambda = (float) num("ROUTE_PRIOR", 0.0);
        const char* cnt = strata::env_get("ROUTE_PRIOR_COUNT");
        c.count_router = cnt == nullptr || std::string(cnt) != "picked";
        return true;
    }();
    (void) parsed;
    return c;
}
unsigned long long* RoutePriorCfg::stats_on(int dev) {
    if (lambda <= 0.0f || dev < 0) return nullptr;
    std::lock_guard<std::mutex> lk(mu);
    const auto it = blocks.find(dev);
    if (it != blocks.end()) return it->second;
    unsigned long long* p = nullptr;
    {
        const OnDevice on(dev);
        if (!strata::gpu::alloc_device(&p, 4 * sizeof(unsigned long long)) ||
            !strata::gpu::memset(p, 0, 4 * sizeof(unsigned long long))) {
            if (p != nullptr) strata::gpu::free(p);
            p = nullptr;
        }
    }
    if (blocks.empty()) std::atexit(report);
    blocks[dev] = p;
    return p;
}
void RoutePriorCfg::report() {
    RoutePriorCfg& c = route_prior_cfg();
    for (const auto& kv : c.blocks) {
        if (kv.second == nullptr) continue;
        unsigned long long h[4] = {};
        const OnDevice on(kv.first);
        if (!strata::gpu::device_sync() || !strata::gpu::copy(h, kv.second, sizeof(h))) continue;
        const std::string who = c.blocks.size() > 1 ? " (GPU " + std::to_string(kv.first) + ")" : "";
        std::fprintf(stderr, "route-prior%s: lambda=%g count=%s: token-layers=%llu with swaps=%llu (%.1f%%) swaps=%llu, "
                             "dropped router probability %.3f%% a token-layer\n",
                     who.c_str(), c.lambda, c.count_router ? "router" : "picked", h[0], h[1],
                     h[0] ? 100.0 * (double) h[1] / (double) h[0] : 0.0, h[2],
                     h[0] ? 100.0 * (double) h[3] * 1e-6 / (double) h[0] : 0.0);
    }
}
RouteResidentCfg& route_resident_cfg() {
    static RouteResidentCfg c;
    static const bool parsed = [] {
        if (const char* v = strata::env_get("ROUTE_RESIDENT")) c.margin = (float) std::strtod(v, nullptr);
        if (const char* v = strata::env_get("ROUTE_RESIDENT_RANKS")) {   // lo-hi
            char* end = nullptr;
            const long lo = std::strtol(v, &end, 10);
            if (end != v && *end == '-') {
                const char* h = end + 1;
                const long hi = std::strtol(h, &end, 10);
                if (end != h) { c.lo = (int) lo; c.hi = (int) hi; }
            }
        }
        return true;
    }();
    (void) parsed;
    return c;
}
// STRATA_ROUTE_DUMP=<path>: every verify window's routed experts and weights, after the router and every option that
// changes them; STRATA_ROUTE_REPLAY=<path> (opt-in, changes the output): a dump's experts and weights in place of the
// router's, so a quant is scored with the reference's routing and its KL is the rounding alone (tools/quant_quality.py
// --route-lock).  Record: int32 request, int32 position, int32 layer, int32 k, k int32 ids, k float weights; a request
// begins where a window's position does not pass the last one's (pos0 rises within one).  Windows of one request at a
// time only (not the batch slots, not the pipelined windows), and only the windows: positions the prompt path read
// have no records, and replaying them is an error.
struct RouteTrace {
    std::FILE* dump = nullptr;
    std::mutex mu;
    bool replay = false;
    std::string error;
    struct Req {
        int64_t k = 0, layers = 0;
        std::vector<int32_t> ids;
        std::vector<float> w;
        std::vector<uint8_t> have;   // [pos * layers + layer]
    };
    std::vector<Req> reqs;

    bool load(const char* path) {
        std::FILE* f = std::fopen(path, "rb");
        if (f == nullptr) { error = std::string("XESTRATA_ROUTE_REPLAY: cannot read ") + path; return false; }
        int32_t h[4];
        std::vector<int32_t> ids;
        std::vector<float> w;
        while (std::fread(h, sizeof h, 1, f) == 1) {
            const int32_t r = h[0], pos = h[1], l = h[2], k = h[3];
            if (r < 0 || r > 1 << 20 || pos < 0 || l < 0 || l >= 1024 || k < 1 || k > 64) {
                error = std::string("XESTRATA_ROUTE_REPLAY: ") + path + ": a malformed record";
                break;
            }
            ids.resize((size_t) k);
            w.resize((size_t) k);
            if (std::fread(ids.data(), 4, (size_t) k, f) != (size_t) k || std::fread(w.data(), 4, (size_t) k, f) != (size_t) k) {
                error = std::string("XESTRATA_ROUTE_REPLAY: ") + path + ": a truncated record";
                break;
            }
            if ((size_t) r >= reqs.size()) reqs.resize((size_t) r + 1);
            Req& q = reqs[(size_t) r];
            if (q.k == 0) q.k = k;
            if (q.k != k) { error = std::string("XESTRATA_ROUTE_REPLAY: ") + path + ": records of two k"; break; }
            if (l >= q.layers) {   // re-strided to the deeper layer count
                const int64_t L2 = l + 1, P = q.layers > 0 ? (int64_t) q.have.size() / q.layers : 0;
                Req n;
                n.k = k; n.layers = L2;
                n.ids.assign((size_t) (P * L2 * k), 0); n.w.assign(n.ids.size(), 0.0f); n.have.assign((size_t) (P * L2), 0);
                for (int64_t p = 0; p < P; ++p)
                    for (int64_t j = 0; j < q.layers; ++j) {
                        const size_t a = (size_t) (p * q.layers + j), b = (size_t) (p * L2 + j);
                        n.have[b] = q.have[a];
                        std::memcpy(&n.ids[b * (size_t) k], &q.ids[a * (size_t) k], (size_t) k * 4);
                        std::memcpy(&n.w[b * (size_t) k], &q.w[a * (size_t) k], (size_t) k * 4);
                    }
                q = std::move(n);
            }
            const size_t slot = (size_t) pos * (size_t) q.layers + (size_t) l;
            if (slot >= q.have.size()) {
                q.have.resize(slot + 1, 0);
                q.ids.resize((slot + 1) * (size_t) k, 0);
                q.w.resize((slot + 1) * (size_t) k, 0.0f);
            }
            q.have[slot] = 1;   // a position read again (a rejected draft) keeps its last routing
            std::memcpy(&q.ids[slot * (size_t) k], ids.data(), (size_t) k * 4);
            std::memcpy(&q.w[slot * (size_t) k], w.data(), (size_t) k * 4);
        }
        std::fclose(f);
        return error.empty();
    }

    /// rows [0, T) of layers [lb, le) of request `r` from position pos0 into the window's staging (stride MT rows)
    bool fill(int r, int64_t pos0, int T, int64_t lb, int64_t le, int64_t K, int64_t MT, int32_t* ids, float* w,
              std::string& err) const {
        const Req* q = r >= 0 && (size_t) r < reqs.size() ? &reqs[(size_t) r] : nullptr;
        if (q != nullptr && q->k != K) {
            err = "XESTRATA_ROUTE_REPLAY: the dump routes " + std::to_string(q->k) + " experts a token, this model " +
                  std::to_string(K);
            return false;
        }
        for (int64_t l = lb; l < le; ++l)
            for (int t = 0; t < T; ++t) {
                const size_t slot = (size_t) (pos0 + t) * (size_t) (q ? q->layers : 0) + (size_t) l;
                if (q == nullptr || l >= q->layers || slot >= q->have.size() || !q->have[slot]) {
                    err = "XESTRATA_ROUTE_REPLAY: no routing for request " + std::to_string(r) + ", position " +
                          std::to_string(pos0 + t) + ", layer " + std::to_string(l);
                    return false;
                }
                const size_t d = ((size_t) l * (size_t) MT + (size_t) t) * (size_t) K;
                std::memcpy(ids + d, &q->ids[slot * (size_t) K], (size_t) K * 4);
                std::memcpy(w + d, &q->w[slot * (size_t) K], (size_t) K * 4);
            }
        return true;
    }

    void write(int r, int64_t pos0, int T, int64_t lb, int64_t le, int64_t K, int64_t MT, const int32_t* ids,
               const float* w) {
        std::lock_guard<std::mutex> lk(mu);
        for (int t = 0; t < T; ++t)
            for (int64_t l = lb; l < le; ++l) {
                const size_t s = ((size_t) l * (size_t) MT + (size_t) t) * (size_t) K;
                const int32_t h[4] = {r, (int32_t) (pos0 + t), (int32_t) l, (int32_t) K};
                std::fwrite(h, sizeof h, 1, dump);
                std::fwrite((const void*) (ids + s), 4, (size_t) K, dump);
                std::fwrite((const void*) (w + s), 4, (size_t) K, dump);
            }
        std::fflush(dump);
    }
};
RouteTrace& route_trace() {
    static RouteTrace t;
    static const bool parsed = [] {
        if (const char* p = strata::env_get("ROUTE_DUMP"); p != nullptr && *p) {
            t.dump = std::fopen(p, "wb");
            if (t.dump == nullptr) t.error = std::string("XESTRATA_ROUTE_DUMP: cannot write ") + p;
        }
        if (const char* p = strata::env_get("ROUTE_REPLAY"); p != nullptr && *p) {
            t.replay = t.load(p);
            if (t.replay)
                std::fprintf(stderr, "route replay: %zu requests from %s (the reference's experts and weights)\n",
                             t.reqs.size(), p);
        }
        return true;
    }();
    (void) parsed;
    return t;
}
// A one-token window always keeps its token, so its graph advances the GDN conv history and recurrence state itself
// (its indexer append and PLE history are final already) and Verifier::commit launches no commit graph after it
// (upstream 352cad8f): the draft round then runs alone instead of beside the commit's kernels, which read and write
// each GDN layer's state.  STRATA_ONE_TOKEN_COMMIT=0: the commit graph after every window.
bool one_token_self_commit() {
    static const bool on = [] {
        const char* v = strata::env_get("ONE_TOKEN_COMMIT");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}
// the head's hyper-connection read for the window's tokens in one fused_gr_read_multi (upstream edc592b1: its sums
// are gr_read's); STRATA_HEAD_MIX_MULTI=0: gr_read per token
bool head_mix_multi_enabled() {
    static const bool on = [] {
        const char* v = strata::env_get("HEAD_MIX_MULTI");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? reinterpret_cast<T*>(base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (!strata::gpu::alloc_host(h, bytes)) return false;
    std::memset(*h, 0, bytes);
    return strata::gpu::device_pointer(d, *h);
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}

}  // namespace

namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
}  // namespace

void Verifier::diag(std::FILE* f) const {
    uint32_t at = 0, rang = 0, served = 0, fa = 0, fb = 0;
    win_.diag(at, rang, served, fa, fb);
    // outside a verify stage these are the last window's numbers (it finished), not the stalled work's (upstream
    // 9631c96)
    const char* where = progress().where.load();
    const bool current = where != nullptr && std::strncmp(where, "verify window", 13) == 0;
    std::fprintf(f, "  verify window%s: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u\n", current ? "" : " (last window, not the current stage)",
                 last_t_, (long long) last_pos0_, at, rang, served, fa, fb);
}

void Verifier::taps_off() {
    if (tap_layers_.empty() && tap_req_.empty()) return;
    if (cs_) strata::gpu::stream_sync(cs_);
    tap_req_.clear();
    tap_layers_.clear();
    std::fill(tap_slot_.begin(), tap_slot_.end(), -1);
    for (auto& e : exec_)
        if (e) { strata::gpu::graph_destroy(e); e = nullptr; }
    for (auto& e : exec_res_)
        if (e) { strata::gpu::graph_destroy(e); e = nullptr; }
    for (auto& v : segs_) {
        for (auto* e : v) strata::gpu::graph_destroy(e);
        v.clear();
    }
    for (auto& kv : exec_bm_)
        if (kv.second) strata::gpu::graph_destroy(kv.second);
    exec_bm_.clear();
}

Verifier::~Verifier() {
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    if (cs_) strata::gpu::stream_sync(cs_);
    for (auto& e : exec_)
        if (e) strata::gpu::graph_destroy(e);
    for (auto& e : exec_res_)
        if (e) strata::gpu::graph_destroy(e);
    for (auto& v : segs_)
        for (auto* e : v) strata::gpu::graph_destroy(e);
    if (commit_exec_) strata::gpu::graph_destroy(commit_exec_);
    for (auto& kv : exec_bm_)
        if (kv.second) strata::gpu::graph_destroy(kv.second);
    for (auto& kv : commit_bm_)
        if (kv.second) strata::gpu::graph_destroy(kv.second);
    if (arena_b_) strata::gpu::free(arena_b_);
    if (h_commitb_) strata::gpu::free(h_commitb_);
    for (auto* m : {&save_bm_, &dcommit_bm_})
        for (auto& kv : *m)
            if (kv.second) strata::gpu::graph_destroy(kv.second);
    if (dc_arena_) strata::gpu::free(dc_arena_);
    if (h_commitd_) strata::gpu::free(h_commitd_);
    if (logit_bias_device_) strata::gpu::free(logit_bias_device_);
    for (float* b : slot_bias_dev_)
        if (b != nullptr) strata::gpu::free(b);
    if (slot_hist_d_) strata::gpu::free(slot_hist_d_);
    if (slot_params_d_) strata::gpu::free(slot_params_d_);
    if (allowed_device_) strata::gpu::free(allowed_device_);
    if (rp_lambda_d_) strata::gpu::free(rp_lambda_d_);
    if (cs_ && cs_ != ext_stream_) strata::gpu::stream_destroy(cs_);   // set_stream: the stage's stream, shared
    if (ev_done_) strata::gpu::event_destroy(ev_done_);
    if (ev_commit_) strata::gpu::event_destroy(ev_commit_);
    if (prof_pin_) strata::gpu::free(prof_pin_);
    for (strata::gpu::Stream s : df_side_)
        if (s) strata::gpu::stream_destroy(s);
    if (fetch_side_) strata::gpu::stream_destroy(fetch_side_);
    if (commit_done_) strata::gpu::event_destroy(commit_done_);
    if (arena_) strata::gpu::free(arena_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_nf_, h_route_, h_rw_, h_rpl_ids_, h_rpl_w_};
    for (void* h : hosts)
        if (h) strata::gpu::free(h);
}

void Verifier::tune_device(const ModelGeometry& g, void* stream) {
    // the GDN step's fastest form on this device for each window size
    if (g.ssm_v_heads > 0 && g.ssm_k_heads > 0)
        strata::kernels::gdn_step_tune((int) g.ssm_conv_channels, (int) g.ssm_k_heads, (int) g.ssm_v_heads, stream);
    // and the PCIe share's fetch: the work-groups that fill this device's link
    strata::kernels::fetch_blobs_tune(stream);
    // and the multi-row BF16 GEMV's form for the window's shapes: the router, the indexer's key and query, the PLE
    // block's key and value
    for (const int64_t n_out : {(int64_t) g.n_expert, (int64_t) g.idx_key_dim, (int64_t) g.idx_q_heads * g.idx_key_dim,
                                (int64_t) strata::kernels::NG_HC_DIM, (int64_t) g.n_embd})
        strata::kernels::bf16_gemv_fp32_mmvf_multi_tune(g.n_embd, n_out, stream);
}

bool Verifier::set_logit_bias(const std::vector<float>& bias, std::string& err) {
    if (next_ && !next_->set_logit_bias(bias, err)) return false;
    if (le_ < g_->n_layers) return true;
    const OnDevice on_device(device_);
    if (!bias.empty() && bias.size() != (size_t) n_vocab_) {
        err = "logit_bias vocabulary size mismatch"; return false;
    }
    if (bias != logit_bias_host_ && !bias.empty()) {
        const size_t bytes = bias.size() * sizeof(float);
        if (!constraint_upload(logit_bias_device_, bias.data(), bytes, bytes)) {
            err = "cannot upload logit_bias"; return false;
        }
    }
    logit_bias_host_ = bias;
    sampling_.logit_bias = bias.empty() ? nullptr : logit_bias_device_;
    return true;
}

bool Verifier::set_slot_logit_bias(int slot, const std::vector<float>& bias, std::string& err) {
    if (next_ && !next_->set_slot_logit_bias(slot, bias, err)) return false;
    if (le_ < g_->n_layers) return true;          // only the stage with the head draws the rows
    if (slot < 0 || slot >= (int) slot_bias_dev_.size()) return true;
    if (bias.empty()) {   // its array stays for the slot's next bias; only this row's pointer is cleared
        slot_sp_[(size_t) slot].logit_bias = nullptr;
        return true;
    }
    if (bias.size() != (size_t) n_vocab_) { err = "logit_bias vocabulary size mismatch"; return false; }
    const OnDevice on_device(device_);
    float*& dev = slot_bias_dev_[(size_t) slot];
    const size_t bytes = bias.size() * sizeof(float);
    if (dev == nullptr && !strata::gpu::alloc_device(&dev, bytes)) {
        err = "cannot allocate the slot's logit_bias";
        return false;
    }
    if (!strata::gpu::copy(dev, bias.data(), bytes)) {   // its own array: a request's bias is its row's alone
        err = "cannot upload the slot's logit_bias";
        return false;
    }
    slot_sp_[(size_t) slot].logit_bias = dev;
    return true;
}

bool Verifier::set_allowed(const uint32_t* rows, int T, std::string& err) {
    if (next_ && !next_->set_allowed(rows, T, err)) return false;
    if (le_ < g_->n_layers) return true;
    allowed_on_ = rows != nullptr;
    if (rows != nullptr) {
        // (the mask's rows: the window's, at least the 8 a window always had)
        const int cap_rows = std::max(max_t_, (int) strata::kernels::kVerifyNarrowT);
        if (T < 1 || T > cap_rows) { err = "grammar mask: window rows out of range"; return false; }
        const OnDevice on_device(device_);
        const size_t row = (size_t) ((n_vocab_ + 31) / 32) * sizeof(uint32_t);
        if (!constraint_upload(allowed_device_, rows, row * (size_t) cap_rows, row * (size_t) T)) {
            err = "cannot upload the grammar's token masks"; return false;
        }
    }
    sampling_.allowed = allowed_on_ ? allowed_device_ : nullptr;
    return true;
}

bool Verifier::set_route_prior(bool on, std::string& err) {
    if (next_ && !next_->set_route_prior(on, err)) return false;
    if (rp_lambda_d_ == nullptr || on == rp_on_) return true;
    const float v = on ? route_prior_cfg().lambda : 0.0f;
    const OnDevice on_device(device_);
    if (!strata::gpu::copy(rp_lambda_d_, &v, sizeof(float))) { err = "cannot set the route prior"; return false; }
    rp_on_ = on;
    return true;
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    device_ = current_device();   // a layer split's stage on another GPU: init runs on it (OnDevice)
    rr_stats_ = route_resident_cfg().stats_on(device_);   // STRATA_ROUTE_RESIDENT: this GPU's counters, outside graph capture
    rp_stats_ = route_prior_cfg().stats_on(device_);
    if (route_prior_cfg().lambda > 0.0f && rp_lambda_d_ == nullptr) {   // the kernel reads lambda here (set_route_prior)
        if (!strata::gpu::alloc_device(&rp_lambda_d_, sizeof(float)) ||
            !strata::gpu::copy(rp_lambda_d_, &route_prior_cfg().lambda, sizeof(float))) {
            err = "cannot allocate the route prior's lambda"; return false;
        }
    }
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
    // (the CPU experts take a wider window than cpu::MAXT tokens as jobs of at most MAXT)
    if (max_t < 2 || max_t > strata::kernels::kVerifyMaxT) {
        err = "verify: the window must hold 2.." + std::to_string(strata::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache); with "
              "--expert-cache auto, no VRAM was left for it - the 'no VRAM is left for the expert cache' line above "
              "says how much is short and what makes room (a smaller --max-context, --kv q4_0, setup --draft-vocab en, "
              "images on the CPU)";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (le_ < 0) le_ = g.n_layers;
    if (lb_ < 0 || lb_ >= le_ || le_ > g.n_layers || (lb_ > 0 && hand_in_ == nullptr) ||
        (le_ < g.n_layers && hand_out_ == nullptr)) {
        err = "verify: the stage's layer range or its hand-off buffers are wrong";
        return false;
    }
    // the DFlash2 drafter's taps (set_taps): this stage's share of the layers, below the last
    tap_layers_.clear();
    tap_slot_.assign((size_t) g.n_layers, -1);
    for (int l : tap_req_)
        if (l >= lb_ && l < le_ && l < g.n_layers - 1) {
            tap_slot_[(size_t) l] = (int) tap_layers_.size();
            tap_layers_.push_back(l);
        }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * (NH + NKV + IQ) * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped((size_t) std::max(max_t, (int) strata::kernels::kVerifyNarrowT) *
                         strata::kernels::kNonfiniteBlocks * 4,
                     (void**) &h_nf_, (void**) &m_nf_) &&
              mapped((size_t) g.n_layers * T * K * 4, (void**) &h_route_, (void**) &m_route_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    {   // the routed experts' window (moe_window.hpp)
        MoeWindowShape ws;
        ws.n_embd = g.n_embd;
        ws.n_expert = g.n_expert;
        ws.k = ss.k;
        ws.max_t = max_t;
        MoeWindowTier tier;
        tier.cache_base = hits.cache_base;
        tier.blob = hits.blob;
        tier.slot_off = hits.slot_off;
        tier.n_slots = hits.n_slots;
        win_.set_layer_range(lb_, le_);
        if (!win_.init(ws, tier, err)) return false;
    }
    {
        RouteTrace& rt = route_trace();
        if (!rt.error.empty()) { err = rt.error; return false; }
        rt_dump_ = rt.dump != nullptr;
        rt_replay_ = rt.replay;
        const size_t lk = (size_t) g.n_layers * (size_t) T * (size_t) K * 4;
        if ((rt_dump_ && !mapped(lk, (void**) &h_rw_, (void**) &m_rw_)) ||
            (rt_replay_ && (!mapped(lk, (void**) &h_rpl_ids_, (void**) &m_rpl_ids_) ||
                            !mapped(lk, (void**) &h_rpl_w_, (void**) &m_rpl_w_)))) {
            err = "verify: the route trace's staging allocation failed";
            return false;
        }
    }
    // ---- the device arena: the same sequence counted, then carved
    // APR: the rotated copy of the expert input (nat_xr_, T x N floats) only where a layer's gate/up is
    // Hadamard-folded; its one reader is that layer's had_gu branch of the expert input in pre()
    bool had_gu_any = false;
    {
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (size_t l = 0; lay.native && l < lay.fmt.size(); ++l) had_gu_any = had_gu_any || lay.fmt[l].had_gu;
    }
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * (NH + NKV + IQ)); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N);
        nat_xr_ = had_gu_any ? b.take<float>(T * N) : nullptr;
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        one_ = b.take<int32_t>(4);
        hist_snap_ = b.take<float>(T * HS);
        ple_key_ = b.take<float>(T * (uint64_t) strata::kernels::NG_HC_DIM); ple_val_ = b.take<float>(T * N);
        tap_dev_ = tap_layers_.empty() ? nullptr : b.take<float>(T * tap_layers_.size() * N);
    };
    Bump count;
    carve(count);
    if (!strata::gpu::alloc_device(&arena_, count.used)) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    arena_bytes_ = count.used;
    strata::gpu::memset(arena_, 0, count.used);
    prof_on_ = strata::env_get("VERIFY_PROFILE") != nullptr;
    // The stamps are recorded in the window graph, so they need a clock a kernel reads: without one the profile is off
    // rather than the capture failing.  Level Zero native commands and profiling tags cannot be recorded in a graph.
    if (prof_on_ && !strata::kernels::gpu_stamp_available()) {
        std::fprintf(stderr, "xestrata verify: XESTRATA_VERIFY_PROFILE needs a device-scope clock (ext_oneapi_clock_device), "
                             "which this device does not have; the stage profile is off\n");
        prof_on_ = false;
    }
    if (prof_on_) {
        const size_t np = (size_t) g.n_layers * kProfPer + 4;
        if (!strata::gpu::alloc_device((void**) &prof_, np * 8)) { prof_on_ = false; prof_ = nullptr; }
        else { strata::gpu::memset(prof_, 0, np * 8); prof_h_.assign(np, 0); }
    }
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    {
        const int32_t one = 1;
        if (!strata::gpu::copy(one_, &one, sizeof one)) { err = "verify: the arena could not be set"; return false; }
    }
    try {   // APR (prism.hadamard, hadamard.txt): the sign vectors on this device, before any graph is recorded
        had_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) g.n_embd), (int) g.n_embd);
        had_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) g.n_ff), (int) g.n_ff);
    } catch (const std::exception& e) {
        err = std::string("verify: ") + e.what();
        return false;
    }
    (void) TS;
    if (ext_stream_ != nullptr) cs_ = ext_stream_;   // set_stream (pipelined windows): the stage's shared stream
    else cs_ = strata::gpu::stream_create();
    if (!cs_) {
        err = "verify: stream create failed";
        return false;
    }
    try {   // before any graph is recorded
        tune_device(g, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify: ") + e.what();
        return false;
    }
    {   // the mixer's side branches (upstream 2e4ddf6e) where they pay on this device: measured once (the RTX 4070
        // +0.7%; the B70 lost 4%, its graphs' forks and joins cost more than the overlap gave).  STRATA_DF_BRANCH=0|1
        // fixes the choice.
        const char* v = strata::env_get("DF_BRANCH");
        for (int i = 0; i < 2; ++i)
            if (!df_side_[i]) df_side_[i] = strata::gpu::stream_create();
        // the window waits for the CPU inside its graph: no branch where that wait would hold up the launch itself
        const bool no_branch = !strata::gpu::graph_branches_may_wait_for_host();
        if (!df_side_[0] || !df_side_[1] || no_branch) df_branch_ = false;
        else if (v != nullptr) df_branch_ = std::strtol(v, nullptr, 10) != 0;
        else {
            try {
                df_branch_ = strata::kernels::graph_branch_pays(cs_, df_side_[0], df_side_[1]);
            } catch (const std::exception&) {
                df_branch_ = false;   // a device that cannot record such a graph keeps one queue
            }
        }
        // the PCIe share's fetch on a side branch of the window graph, beside the VRAM hits (post()): the fetch waits
        // on the link, the hits on the GPU, so side by side the hits' time hides under the fetch.  Where the graph's
        // branches pay (df_branch_) unless STRATA_FETCH_BRANCH=0|1 fixes it.
        if (!fetch_side_) fetch_side_ = strata::gpu::stream_create();
        const char* fv = strata::env_get("FETCH_BRANCH");
        fetch_branch_ = fetch_side_ != nullptr && !no_branch &&
                        (fv != nullptr ? std::strtol(fv, nullptr, 10) != 0 : df_branch_);
        if (no_branch && (v != nullptr || fv != nullptr))
            std::fprintf(stderr, "xestrata verify: no graph branches in the windows on this device: its graph launch "
                                 "would wait behind the window's wait for the CPU (XESTRATA_DF_BRANCH and "
                                 "XESTRATA_FETCH_BRANCH ignored)\n");
    }
    if (!strata::gpu::event_create(&commit_done_)) {
        err = "verify: event create failed";
        return false;
    }
    {
        if (win_.segmented())
            std::fprintf(stderr, "xestrata verify: the GPU's running kernels do not see the host's writes: each window "
                                 "runs as segments the host launches (slower)\n");
    }
    // E-6: a layer whose routed experts are all resident is planned on the device (XESTRATA_VERIFY_DEVICE_PLAN=1: on;
    // exact, but neutral on RIBPC 1-2 GPUs, and the window graph keeps the choice, so auto (the default) is off and the
    // switch is the A/B's other side)
    {
        // (not with set_always_publish: the device table may lag the host's while windows are in flight)
        device_plan_ = !always_publish_ && strata::core::pick_from_env("VERIFY_DEVICE_PLAN") == strata::core::Pick::On;
    }
    res_graph_ = !always_publish_ && hits.h_res != nullptr && hits.d_res != nullptr && MoeWindow::resident_graph_on();
    if (!win_.device_plans()) { device_plan_ = false; res_graph_ = false; }
    // a latent pack keeps the host plan: the plan's activation-row rewrite reads the doorbell's mapped ids, which
    // the resident / device-plan graphs do not write (their speed is a later measurement)
    if (win_.latent()) { device_plan_ = false; res_graph_ = false; }
    std::fprintf(stderr, "xestrata verify: window up to %d tokens, %.1f MiB of device buffers\n", max_t,
                 (double) count.used / 1048576.0);
    return true;
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
bool Verifier::record_window(int T, strata::gpu::Stream cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && ple_stage();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const int G = (split_ && T >= 2 && !batch_rec_) ? 2 : 1;   // a batch window is one group
    // STRATA_DF_BRANCH: mixer work that reads only the layer's input on a side branch of the graph.  fork(i): side i
    // after everything on cs so far; join(i): cs after everything on side i - a branch never races with what came
    // before its fork or after its join.  Not in a split or batch window, nor under the profiler (its stamps are on cs).
    const bool br = df_branch_ && G == 1 && !batch_rec_ && !prof_on_;
    auto fork = [&](int i) { return strata::gpu::stream_fork(cs, df_side_[i]); };
    auto join = [&](int i) { return strata::gpu::stream_join(cs, df_side_[i]); };
    // see Verifier::commit; not in a pipelined window (always_publish_), which a misprediction rolls back
    const bool self_commit = T == 1 && !batch_rec_ && !always_publish_ && one_token_self_commit();
    static const bool dec_batch = [] { const char* v = strata::env_get("DEC_BATCH"); return v == nullptr || std::strtol(v, nullptr, 10) != 0; }();
    auto stamp = [&](int64_t l, int i, int grp) { if (prof_on_ && grp == 0) gpu_stamp(prof_, (int) (l * kProfPer + i), cs); };
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    bool gate_later_[2] = {false, false};   // the group's shared rows still to be scaled by the combine (pre -> post)
    if (!batch_rec_) groups_[T] = G;
    const int64_t nQall = g.n_qsa_layers();
    // a batch window: row t is slot t, whose state lives in its own session (upstream PR #559)
    auto slot_ss = [&](int t) -> SessionState& { return batch_rec_ ? *slots_[(size_t) brow_[t]] : ss; };

    // ---- the window's inputs, from mapped staging
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), cs);
    // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
    const int32_t* pos_k = pos_ + MT * NH;
    const int32_t* pos_i = pos_ + MT * (NH + NKV);
    // The window's PLE rows: read from the table by the host after the launch, while the GPU runs layer 0, and
    // copied in at layer 1 after the GPU's wait for layer 0's flag (upstream a36be1db reads them at layer 0's ring):
    // the reads no longer hold the window's launch back (0.35 ms a window on the RTX 4070).  Not where the host
    // raises no flag for layer 0 before layer 1: the resident graph, a batch window, a stage that starts after layer 0.
    const bool ple_late = ple_on && !recording_res_ && !batch_rec_ && lb_ == 0;
    if (ple_on && !ple_late) copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);

    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    const int64_t HB = Verifier::handoff_floats(g);
    const int hrow0 = batch_rec_ ? row_base_ : 0;   // a slot group's own hand-off rows (batch_launch)
    if (lb_ > 0) {
        for (int t = 0; t < T; ++t) {
            const float* hin = hand_in_ + (size_t) (hrow0 + t) * HB;
            copy_from_mapped(Rt(t), hin, HC * N, cs);
            copy_from_mapped(bo_ + (size_t) t * N, hin + HC * N, N, cs);
            copy_from_mapped(inj2_ + (size_t) t * HC, hin + HC * N + N, HC, cs);
        }
    } else {
        if (!embed_rows(wt, g, tok_, 0, T, emb_, cs, err)) {
            err = "verify: " + err;
            return false;
        }
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        stamp(l, 0, grp);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
        // already applied it)
        bool pending = l > 0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            if (ple_late && grp == 0) {
                wait_flag_ge(win_.served_flag(), 1, cs);   // the host answered ring 1 after the late PLE rows
                copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
            }
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            // the window's residual writes in one launch (each row its own R, block output and injection)
            if (dec_batch) gr_write_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, gs, Rt(tb), n, cs);
            // The key and value projections depend only on each row's n-gram embedding: the group's rows go through
            // the multi-row forms of the same kernels, the weights read once, each row's arithmetic the single-row
            // call's (upstream d92c9feb).  The history-dependent rest stays row by row.  STRATA_PLE_BATCH=0: row by row.
            static const bool ple_batch = [] {
                const char* v = strata::env_get("PLE_BATCH");
                return v == nullptr || std::strtol(v, nullptr, 10) != 0;
            }();
            PleWeights pw = ss.ple.w;
            const bool nkey = pw.key_native_data != nullptr && pw.key_bf16 == nullptr;
            const bool batched = ple_batch && n > 1 && (pw.key_bf16 != nullptr || nkey);
            if (batched) {
                try {
                    if (pw.key_bf16 != nullptr) {
                        bf16_gemv_fp32_mmvf_multi(ple_ + tb * N, N, pw.key_bf16, ple_key_ + (size_t) tb * NG_HC_DIM,
                                                  NG_HC_DIM, N, NG_HC_DIM, n, cs);
                    } else {
                        native_quantize_q8_1(ple_ + tb * N, xq_, (int) N, n, cs);
                        native_mmvq(pw.key_native_type, pw.key_native_data, xq_, ple_key_ + (size_t) tb * NG_HC_DIM,
                                    (int) N, NG_HC_DIM, n, cs);
                    }
                    bf16_gemv_fp32_mmvf_multi(ple_ + tb * N, N, pw.value_bf16, ple_val_ + tb * N, N, N, N, n, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE batch: ") + e.what();
                    return false;
                }
            }
            // with the batched projections and the native post-ops, the rows' post-ops in one pass that also writes
            // the commit's history snapshots: native_ple_postops_batch's arithmetic is the rows' calls with the history
            // advanced after each (upstream 2e4ddf6e's STRATA_DF_PLE).  Scratch: the group's expert rows (the
            // window's parts and hits, combined by now) and lo_ (the next read rewrites it).  STRATA_DF_PLE=0: row by row.
            static const bool df_ple = [] {
                const char* e = strata::env_get("DF_PLE");
                return e == nullptr || std::strtol(e, nullptr, 10) != 0;
            }();
            const bool ple_pass = df_ple && batched && dec_batch && !batch_rec_;
            if (ple_pass) {
                try {
                    native_ple_postops_batch_snap(ple_key_ + (size_t) tb * NG_HC_DIM, Rt(tb), ple_val_ + tb * N,
                                                  ss.ple.hist, ss.ple.w, win_.parts(tb),
                                                  win_.hit_out(tb), lo_ + (size_t) tb * g.hc_lr, n,
                                                  hist_snap_ + (size_t) tb * HS, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE (window): ") + e.what();
                    return false;
                }
            }
            for (int t = tb; t < te && !ple_pass; ++t) {
                if (batched) {
                    pw.pre_key = ple_key_ + (size_t) t * NG_HC_DIM;
                    pw.pre_value = ple_val_ + t * N;
                }
                if (!dec_batch) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                float* hist = batch_rec_ ? slot_ss(t).ple_hist : ss.ple.hist;   // the row's own history
                try {
                    ple_block(ple_ + t * N, Rt(t), hist, pw, po, ss.ple.scratch, cs);
                    ple_history_advance(hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                copy_from_mapped(hist_snap_ + (size_t) t * HS, hist, HS, cs);
            }
            pending = false;
        }
        // (in reads of at most kFusedGrMaxT rows: a wider window's rows in two, each row's arithmetic its own)
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            for (int c0 = tb; c0 < te; c0 += kFusedGrMaxT) {
            const int c1 = std::min(te, c0 + kFusedGrMaxT);
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = c0; t < c1; ++t) {
                FusedGrArgs& a = fa[t - c0];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                if (wd[half]->hc_q8 != nullptr && wu[half]->hc_q8 != nullptr) {   // STRATA_HC_Q8
                    a.q8_down = (const uint8_t*) wd[half]->hc_q8; a.q8_up = (const uint8_t*) wu[half]->hc_q8;
                    // the first half's mixed as q8_1 too, for the GDN / QSA projections (no quantize launch)
                    if (half == 0 && fused_gr_q8_fuse()) a.q8_mixed = xq_ + (size_t) (t - tb) * (N / 32) * 36;
                }
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, c1 - c0, xn_ + (size_t) c0 * HC * N, cs, (prof_on_ && grp == 0) ? prof_ : nullptr,
                                (int) (l * kProfPer + (half == 0 ? 27 : 30)));
            }
        };
        gr_read_group(0, pending, inj2_, inj_);
        // the read wrote mixed's q8_1 into xq_ itself (STRATA_HC_Q8's 32-column up)
        const bool mixed_q8 = wd[0]->hc_q8 != nullptr && wu[0]->hc_q8 != nullptr && fused_gr_q8_fuse();
        stamp(l, 1, grp);
        float* xm = mixed_ + tb * N;
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                if (!mixed_q8) native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                if (br) {   // a/b and z beside q/k/v and the conv: they read only this layer's input (xm, xq_)
                    if (!fork(0)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                    gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                                 (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N,
                                 (int) HV, n, df_side_[0]);
                    if (!fork(1)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                    native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n,
                                df_side_[1]);
                }
                native_mmvq(wqkv->native_type, wqkv->native_data, xq_, qkv + (size_t) tb * C, (int) N, (int) C, n, cs);
                stamp(l, 2, grp);
                if (batch_rec_) {   // each slot's rows from its own conv history (--batch-mtp: a slot's rows are contiguous)
                    for (int t = tb; t < te;) {
                        const int first = t;
                        while (t < te && brow_[t] == brow_[first]) ++t;
                        SessionState& sx = slot_ss(first);
                        const float* cx = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats +
                                          (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        gdn_conv_l2_multi(cx, qkv + (size_t) first * C, (const float*) wc->data, hb + (size_t) first * C,
                                          (int) C, (int) (2 * HK), EPS, t - first, cs, 0);
                    }
                } else
                // a one-token window keeps its token: it advances the conv history and the state itself (commit)
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb,
                                  self_commit);
                stamp(l, 3, grp);
                if (br) {   // gate, beta and z, before the recurrence reads them
                    if (!join(0) || !join(1)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                } else {
                    gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                                 (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N,
                                 (int) HV, n, cs);
                    stamp(l, 4, grp);
                    native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, cs);
                    stamp(l, 5, grp);
                }
                // the output norm also writes ssm_out's q8_1 input (the same bytes, a launch fewer; upstream
                // 22abb92d, 0a22c467 for the batch's rows).  STRATA_QFUSE=0: a quantize launch of its own.
                static const bool qfuse = [] {
                    const char* e = strata::env_get("QFUSE");
                    return e == nullptr || std::strtol(e, nullptr, 10) != 0;
                }();
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                if (batch_rec_) {   // each slot's recurrence over its own rows, from its own state
                    GdnGroups gg;
                    for (int t = tb; t < te; ++t) {
                        if (t == tb || brow_[t] != brow_[t - 1]) {
                            SessionState& sx = slot_ss(t);
                            gg.state[gg.n] = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats;
                            gg.first[gg.n] = t;
                            gg.ntok[gg.n] = 0;
                            gg.y_q8_1[gg.n] = qfuse ? reinterpret_cast<uint8_t*>(xq_) + (size_t) (t - tb) * (ZV / 32) * 36
                                                    : nullptr;
                            ++gg.n;
                        }
                        ++gg.ntok[gg.n - 1];
                    }
                    for (int i = 0; i < gg.n; ++i) {
                        const int first = gg.first[i];
                        gdn_step_norm_multi(gg.state[i], hb + (size_t) first * C, (int) C, gate + (size_t) first * HV,
                                            beta + (size_t) first * HV, z_ + (size_t) first * ZV,
                                            (const float*) wnm->data, EPS, y_ + (size_t) first * ZV, (int) HK,
                                            (int) HV, gg.ntok[i], nullptr, cs, 0, gg.y_q8_1[i]);
                    }
                } else
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, self_commit ? one_ : nullptr, cs, tb, qfuse ? (void*) xq_ : nullptr);
                stamp(l, 6, grp);
                if (!qfuse) native_quantize_q8_1(y_ + (size_t) tb * ZV, xq_, (int) ZV, n, cs);
                native_mmvq(wout->native_type, wout->native_data, xq_, bo_ + tb * N, (int) ZV, (int) N, n, cs);
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope_on = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos,
                                        strata::gpu::Stream on) {
                    if (native_norm_rope_usable(cols, (int) s.n_rot)) {
                        native_qsa_rms_norm_rope(data, cols, (const float*) norm->data, data, rows, cols, (int) s.n_rot, EPS,
                                                 rope_scaling(), pos, on);
                        return;
                    }
                    native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, on);
                    native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), pos, on);
                };
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    norm_rope_on(data, norm, rows, cols, pos, cs);
                };
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                // the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the
                // window's rows each - row-wise identical arithmetic (STRATA_DEC_BATCH=0: token by token)
                // a Q4_0 K/V takes the batched rows too: they rotate the query as the per-token path does (upstream
                // 2e4ddf6e's STRATA_DF_QB_Q4; =0: Q4_0 token by token)
                static const bool qb_q4 = [] {
                    const char* e = strata::env_get("DF_QB_Q4");
                    return e == nullptr || std::strtol(e, nullptr, 10) != 0;
                }();
                const bool qb = dec_batch && n > 1 && (!st.kv_q4 || qb_q4);
                if (!mixed_q8) native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                // the query side on branches (upstream 2e4ddf6e): q with its split, norm, RoPE and rotation on side 0, the
                // indexer query on side 1, beside the K/V side; side 1 joins before the block scores, side 0 before
                // attention (xq_ and mixed_ are not rewritten before then)
                const bool qbr = br && qb;
                // q out of the q|gate rows, normed and rotated: one launch where the fused kernel applies, else the
                // split copy and norm_rope (the batched rows only)
                auto q_split_norm_rope = [&](strata::gpu::Stream on) -> bool {
                    if (native_norm_rope_usable((int) HD, (int) s.n_rot)) {
                        native_qsa_rms_norm_rope(qfull_ + tb * NH * 2 * HD, (int) (2 * HD), (const float*) wqn->data,
                                                 qcur_ + tb * NH * HD, (int) (n * NH), (int) HD, (int) s.n_rot, EPS,
                                                 rope_scaling(), pos_ + tb * NH, on);
                        return true;
                    }
                    if (!strata::gpu::copy2d_async(qcur_ + tb * NH * HD, (size_t) HD * 4, qfull_ + tb * NH * 2 * HD,
                                                   (size_t) HD * 2 * 4, (size_t) HD * 4, (size_t) (n * NH), on)) {
                        err = std::string("verify: the q/gate split failed: ") + strata::gpu::last_error();
                        return false;
                    }
                    norm_rope_on(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH, on);
                    return true;
                };
                if (qbr) {
                    if (!fork(0)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                    native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N,
                                (int) (NH * 2 * HD), n, df_side_[0]);
                    if (!q_split_norm_rope(df_side_[0])) return false;
                    if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, df_side_[0]);
                    if (!fork(1)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
                                              N, IQ * ID, n, df_side_[1]);
                    norm_rope_on(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ, df_side_[1]);
                }
                if (qb) bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, cs);
                else for (int t = tb; t < te; ++t)
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, cs);
                stamp(l, 7, grp);
                // K and V in one launch where the pair's layout applies (each row's bits unchanged; the GDN layers' qkv
                // and gate are of different types in the packs, so they stay two launches)
                if (wk->native_type != wv->native_type ||
                    !native_mmvq_pair(wk->native_type, wk->native_data, wv->native_data, xq_, kcur_ + tb * NKV * HD,
                                      vcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs)) {
                    native_mmvq(wk->native_type, wk->native_data, xq_, kcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                    native_mmvq(wv->native_type, wv->native_data, xq_, vcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                }
                if (qb) norm_rope(kcur_ + tb * NKV * HD, wkn, (int) (n * NKV), (int) HD, pos_k + tb * NKV);
                else for (int t = tb; t < te; ++t) norm_rope(kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + t * NH);
                if (st.kv_rot) {   // K and V rotated before they are stored (kv_q4.hpp)
                    fwht256_inplace_cuda(kcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                } else if (st.kv_hybrid) {   // K8V4: only V is rotated
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                }
                stamp(l, 8, grp);
                if (batch_rec_) {   // every slot's indexer tail before its first row, restored by its commit
                    for (int t = tb; t < te; ++t)
                        if (t == tb || brow_[t] != brow_[t - 1])
                            copy_from_mapped(tail_snap_b_ + ((size_t) brow_[t] * nQall + qi) * TS,
                                             slot_ss(t).qsa_states[qi].idx_tail, TS, cs);
                } else
                if (grp == 0) copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, cs);
                // the window's cells in one launch per pool (a batch's rows each have their own K/V: one per row)
                const bool kv_steps = dec_batch && !batch_rec_;
                if (kv_steps) {
                    const int32_t* step_b = step_ + (ptrdiff_t) tb * kStepCount;
                    const float* kc_b = kcur_ + tb * NKV * HD;
                    const float* vc_b = vcur_ + tb * NKV * HD;
                    if (st.kv_hybrid) {   // K8V4: the used pool for both halves (layer.cpp)
                        const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                        const bool mirror = st.host.present();   // streamed: the host copy too
                        kv_append_q8_steps(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_b,
                                           (int) kStepCount, kc_b, kc_b, (int) (NKV * HD), n, s, cs,
                                           mirror ? &hk : nullptr);
                        kv_append_q4_steps(st.v_q4, st.v_q4, st.page_table, step_b, (int) kStepCount, n, vc_b, vc_b,
                                           s, cs, mirror ? &hv : nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_steps(st.k_q4, st.v_q4, st.page_table, step_b, (int) kStepCount, n, kc_b, vc_b, s,
                                           cs, &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_b,
                                           (int) kStepCount, kc_b, vc_b, (int) (NKV * HD), n, s, cs, &st.host);
                    else
                        for (int t = tb; t < te; ++t)
                            kv_append_step(st.k_pool, st.v_pool, st.page_table, step_ + (ptrdiff_t) t * kStepCount,
                                           kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                } else
                for (int t = tb; t < te; ++t) {
                    const QsaState& st = slot_ss(t).qsa_states[qi];   // the row's own K/V (ss's outside a batch)
                    const int32_t* step_t = step_ + (size_t) t * kStepCount;
                    if (st.kv_hybrid) {   // K8V4: the used pool for both halves (layer.cpp)
                        const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                        const bool mirror = st.host.present();   // streamed: the host copy too
                        kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, kcur_ + t * NKV * HD, s, cs, mirror ? &hk : nullptr);
                        kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, mirror ? &hv : nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                    else
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                       vcur_ + t * NKV * HD, s, cs, &st.host);
                }
                if (kv_steps) {
                    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    native_qsa_indexer_append_steps(idx_raw + tb * ID, step_ + (ptrdiff_t) tb * kStepCount + kStepPos,
                                                    (int) kStepCount, n, 0, (const float*) wikn->data, EPS, ib, s,
                                                    st.max_cells, rope_scaling(), cs);
                } else
                for (int t = tb; t < te; ++t) {
                    const QsaState& sx = slot_ss(t).qsa_states[qi];
                    const QsaIndexerBuffers ib{sx.idx_tail, sx.idx_dead, sx.idx_pooled, sx.idx_block_pos};
                    native_qsa_indexer_append(idx_raw + t * ID, step_ + (size_t) t * kStepCount + kStepPos, 0,
                                              (const float*) wikn->data, EPS, ib, s, sx.max_cells,
                                              rope_scaling(), cs);
                }
                stamp(l, 9, grp);
                if (!qbr)
                native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD),
                            n, cs);
                if (qbr) {
                    // the query side runs on the branches
                } else if (qb) {
                    if (!q_split_norm_rope(cs)) return false;
                    if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, cs);   // <Hq, Hk> = <q, k>
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
                                              N, IQ * ID, n, cs);
                    norm_rope(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ);
                } else {
                for (int t = tb; t < te; ++t) {
                    float* qc = qcur_ + t * NH * HD;
                    if (!strata::gpu::copy2d_async(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4, (size_t) HD * 4, (size_t) NH, cs)) {
                        err = std::string("verify: the q/gate split failed: ") + strata::gpu::last_error();
                        return false;
                    }
                    norm_rope(qc, wqn, (int) NH, (int) HD, pos_ + t * NH);
                    if (st.kv_rot) fwht256_inplace_cuda(qc, NH, cs);   // <Hq, Hk> = <q, k>
                }
                for (int t = tb; t < te; ++t) {
                    float* qx = qidx_ + t * IQ * ID;
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), cs);
                    norm_rope(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH);
                }
                }
                stamp(l, 10, grp);
                if (batch_rec_) {   // each row selects and attends over its own slot's K/V
                    const int nrows = te - tb;
                    bool same = nrows > 1;
                    for (int t = tb; same && t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        same = sx.kv_mode != 1 &&  // a streamed slot resolves its blocks before its attention
                               qsa_kv_format(sx) == qsa_kv_format(slot_ss(tb).qsa_states[qi]);
                    }
                    // every slot's rows in one launch (upstream 7a9245b6c); auto times the scores' two forms.  A
                    // launch takes kQsaRowsMax rows: a wider window's rows go in launches of that many
                    const core::Pick ar = core::pick_from_env("BATCH_ATTN_ROWS");
                    const bool one = same &&
                                     (ar == core::Pick::On ||
                                      (ar == core::Pick::Auto &&
                                       qsa_rows_pays(std::min(nrows, kQsaRowsMax), max_blocks_, s, cs)));
                    if (one)
                    for (int c0 = tb; c0 < te; c0 += kQsaRowsMax) {
                        const int c1 = std::min(te, c0 + kQsaRowsMax), cn = c1 - c0;
                        QsaScoreRows sr;
                        QsaAttnPoolsRows pr;
                        sr.n = pr.n = cn;
                        for (int t = c0; t < c1; ++t) {
                            const QsaState& sx = slot_ss(t).qsa_states[qi];
                            sr.pooled[t - c0] = sx.idx_pooled;
                            sr.dead[t - c0] = sx.idx_dead;
                            pr.p[t - c0] = qsa_attn_pools(sx);
                        }
                        qsa_block_scores_rows(sr, qidx_ + (size_t) c0 * IQ * ID, step_ + (size_t) c0 * kStepCount, cn,
                                              max_blocks_, s, scores_ + (size_t) c0 * max_blocks_, cs);
                        qsa_block_topk(scores_ + (size_t) c0 * max_blocks_, step_ + (size_t) c0 * kStepCount, cn,
                                       max_blocks_, cap_, s, sel_ + (size_t) c0 * cap_, cs);
                        qsa_decode_attn_batch_rows(qcur_ + (size_t) c0 * NH * HD, pr, sel_ + (size_t) c0 * cap_,
                                                   step_ + (size_t) c0 * kStepCount, cap_, s,
                                                   attn_scratch_ + (size_t) c0 * attn_scratch_floats_,
                                                   attn_ + (size_t) c0 * NH * HD, cn, cs);
                    }
                    else
                    for (int t = tb; t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        const int32_t* step_t = step_ + (size_t) t * kStepCount;
                        qsa_block_scores(sx.idx_pooled, sx.idx_dead, qidx_ + (size_t) t * IQ * ID, step_t, 1,
                                         max_blocks_, s, scores_ + (size_t) t * max_blocks_, cs);
                        qsa_block_topk(scores_ + (size_t) t * max_blocks_, step_t, 1, max_blocks_, cap_, s,
                                       sel_ + (size_t) t * cap_, cs);
                        qsa_kv_resolve(sx, *g_, sel_ + (size_t) t * cap_, step_t, 1, cap_, cs);
                        const QsaAttnPools px = qsa_attn_pools(sx);
                        qsa_decode_attn_batch(qcur_ + (size_t) t * NH * HD, px, sel_ + (size_t) t * cap_, step_t,
                                              cap_, s, attn_scratch_ + (size_t) t * attn_scratch_floats_,
                                              attn_ + t * NH * HD, 1, cs);
                    }
                } else {
                if (qbr && !join(1)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                // the window's rows read each block's key once (upstream 5b6c9398's STRATA_QSA_ROWS; the same bits).
                // STRATA_QSA_ROWS=0: a sub-group per row and block, as before
                static const bool qsa_rows = [] {
                    const char* v = strata::env_get("QSA_ROWS");
                    return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                }();
                const float* q_win = qidx_ + (ptrdiff_t) tb * IQ * ID;
                const int32_t* st_win = step_ + (ptrdiff_t) tb * kStepCount;
                float* sc_win = scores_ + (size_t) tb * max_blocks_;
                if (qsa_rows && n > 1)
                    qsa_block_scores_rows(st.idx_pooled, st.idx_dead, q_win, st_win, n, max_blocks_, s, sc_win, cs);
                else
                    qsa_block_scores(st.idx_pooled, st.idx_dead, q_win, st_win, n, max_blocks_, s, sc_win, cs);
                qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + (size_t) tb * kStepCount, n, max_blocks_, cap_, s,
                               sel_ + (size_t) tb * cap_, cs);
                stamp(l, 11, grp);
                // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                qsa_kv_resolve(st, *g_, sel_ + (size_t) tb * cap_, step_ + (size_t) tb * kStepCount, n, cap_, cs);
                stamp(l, 12, grp);
                const QsaAttnPools pools = qsa_attn_pools(st);
                if (qbr && !join(0)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                qsa_decode_attn_batch(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + (size_t) tb * kStepCount, cap_,
                                      s, attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n, cs);
                }
                stamp(l, 13, grp);
                if (st.kv_rot || st.kv_hybrid) fwht256_inplace_cuda(attn_ + tb * NH * HD, (int64_t) n * NH, cs);   // back: H^-1 = H
                if (qb) native_qsa_gate_apply(attn_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, attn32_ + tb * NH * HD,
                                              (int) (n * NH), (int) HD, cs);
                else
                for (int t = tb; t < te; ++t)
                    native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD,
                                          (int) NH, (int) HD, cs);
                stamp(l, 14, grp);
                native_quantize_q8_1(attn32_ + tb * NH * HD, xq_, (int) (NH * HD), n, cs);
                native_mmvq(wo->native_type, wo->native_data, xq_, bo_ + tb * N, (int) (NH * HD), (int) N, n, cs);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 16, grp);
        gr_read_group(1, true, inj_, inj2_);
        // the latent MoE's D stage (Edge0): the whole window's rows through D before the router, so the pool's
        // expert work has the router + plan + doorbell stretch of head start (the pool waits on the D doorbell)
        if (win_.latent() && grp == 0) {
            const int n_all = te_[G - 1] - tb_[0];
            if (!win_.latent_d_stage(l, tb_[0], n_all, mixed_ + (size_t) tb_[0] * N, cs, err)) return false;
        }
        // the window's rows routed in 2 launches (one router GEMV reading the weight once, one
        // top-10) instead of 2 per token; every row's arithmetic is the single-token call's (STRATA_DEC_BATCH=0: old)
        const WeightRef* w_router = v.get("ffn_gate_inp.weight");
        // STRATA_LFUSE_GATE=0: the shared expert's scalar gate in a launch of its own, and its sigmoid scale too
        static const bool lfuse_gate = [] {
            const char* e = strata::env_get("LFUSE_GATE");
            return e == nullptr || std::strtol(e, nullptr, 10) != 0;
        }();
        bool g_ready = false;   // the shared expert's raw gates came with the router's launch
        bool qdedup = false;    // the experts' q8_1 image of the input was made before the shared expert
        // STRATA_ROUTE_PRIOR=lambda (upstream 2de9ae06, opt-in, changes the output): the top-10 picked again with a
        // logit bonus for the resident experts; the weights stay the router's (router.cpp)
        const float route_prior = rp_lambda_d_ != nullptr ? route_prior_cfg().lambda : 0.0f;
        // the top-10 and the doorbell in one launch where nothing changes the routing between them (48 nodes fewer a
        // window; the plan's kernel runs after the doorbell, which the host does not read; MoeWindow::route_publish)
        bool published = false;
        if (dec_batch && n > 1 && w_router != nullptr && native_router_takes((int) NE, (int) K)) {
            try {
                // the shared expert's scalar gate as one more work-group of the router's launch (upstream b08cf3e1)
                const WeightRef* w_sg = v.get("ffn_gate_inp_shexp.weight");
                if (lfuse_gate && w_sg != nullptr && w_sg->kind == WeightKind::Bf16InF32)
                    g_ready = bf16_gemv_fp32_mmvf_multi_aux(mixed_ + tb * N, N, (const uint16_t*) w_router->data,
                                                            logits_ + tb * NE, NE, N, NE, n, (const uint16_t*) w_sg->data,
                                                            sh_g_ + tb, 1, cs);
                if (!g_ready)
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) w_router->data, logits_ + tb * NE, NE,
                                              N, NE, n, cs);
                if (!recording_res_ && route_prior <= 0.0f && route_resident_cfg().margin <= 0.0f && !rt_replay_)
                    published = win_.route_publish(tb, n, logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, xm, cs);
                if (!published) native_router_top10_multi(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n, cs, (int) NE);
            } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
            if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
        }
        const bool prior_on = route_prior > 0.0f && native_router_takes((int) NE, (int) K) && hits_.d_res != nullptr;
        // the expert adaptation counts the router's own picks, taken before the prior edits them (RoutePriorCfg)
        const bool count_before = prior_on && route_prior_cfg().count_router && route_rec_ && !rt_dump_ && !batch_rec_;
        if (count_before)
            copy_i32_from_mapped(m_route_ + ((size_t) l * (size_t) MT + (size_t) tb) * (size_t) K, ids_ + tb * K,
                                 (int64_t) n * K, cs);
        if (prior_on) {
            try {
                route_prior_top10(logits_ + tb * NE, hits_.d_res + l * g.n_expert, rp_lambda_d_, ids_ + tb * K, w_ + tb * K, n,
                                  rp_stats_, cs, (int) NE);
            } catch (const std::exception& e) { err = "verify route-prior: " + std::string(e.what()); return false; }
        }
        if (route_resident_cfg().margin > 0.0f && native_router_takes((int) NE, (int) K) && hits_.d_res != nullptr) {
            // STRATA_ROUTE_RESIDENT: after the router, before the plan/doorbell read ids_/w_ (opt-in, changes the output)
            try {
                native_route_resident(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, hits_.d_res + l * g.n_expert, n,
                                      route_resident_cfg().margin, route_resident_cfg().lo, route_resident_cfg().hi,
                                      rr_stats_ != nullptr ? rr_stats_ + (n <= kVerifyMaxT ? 4 : 0) : nullptr, cs,
                                      (int) NE);
            } catch (const std::exception& e) { err = "verify route-resident: " + std::string(e.what()); return false; }
        }
        if (rt_replay_ && !batch_rec_) {   // STRATA_ROUTE_REPLAY: the dump's experts and weights for these rows (RouteTrace)
            copy_i32_from_mapped(ids_ + tb * K, m_rpl_ids_ + ((size_t) l * (size_t) MT + (size_t) tb) * (size_t) K,
                                 (int64_t) n * K, cs);
            copy_i32_from_mapped((int32_t*) (w_ + tb * K),
                                 (const int32_t*) (m_rpl_w_ + ((size_t) l * (size_t) MT + (size_t) tb) * (size_t) K),
                                 (int64_t) n * K, cs);
        }
        if ((device_plan_ || recording_res_) && hits_.d_res != nullptr)   // E-6: every routed expert resident: this group's plan without the host
            win_.resident_plan(hits_.d_res + l * g.n_expert, grp, n, ids_ + tb * K, (uint32_t) ((l - lb_) * G + grp + 1),
                               cs);
        if (!recording_res_ && !published)   // the resident graph asks the host nothing
            win_.doorbell(tb, n, xm, ids_ + tb * K, w_ + tb * K, cs);
        if ((route_rec_ || rt_dump_) && !batch_rec_ && !count_before)   // add_routing's ids: every layer's, also where the host is not asked
            copy_i32_from_mapped(m_route_ + ((size_t) l * (size_t) MT + (size_t) tb) * (size_t) K, ids_ + tb * K,
                                 (int64_t) n * K, cs);
        if (rt_dump_ && !batch_rec_)   // STRATA_ROUTE_DUMP: the weights beside them
            copy_i32_from_mapped((int32_t*) (m_rw_ + ((size_t) l * (size_t) MT + (size_t) tb) * (size_t) K),
                                 (const int32_t*) (w_ + tb * K), (int64_t) n * K, cs);
        stamp(l, 17, grp);
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = xq_;
            // the experts' q8_1 image of the input first: the shared expert's gate and up read it instead of quantizing
            // the same rows again (the same bytes), and the BF16 copy only for a gate that reads it (upstream
            // be7889ed).  STRATA_VERIFY_QDEDUP=0: as before.
            static const bool qdedup_on = [] {
                const char* e = strata::env_get("VERIFY_QDEDUP");
                return e == nullptr || std::strtol(e, nullptr, 10) != 0;
            }();
            qdedup = qdedup_on && strata::kernels::cpu::expert_layout().native;
            if (qdedup) win_.expert_input(tb, n, xm, cs);
            try {
                // the window's combine scales the shared rows itself (upstream b08cf3e1; the same bits, a launch fewer)
                gate_later_[grp] = shared_expert_multi(n, xm, nsw, (const uint16_t*) wgi->data,
                                                       sh_gate_ + (size_t) tb * g.n_ff, sh_up_ + (size_t) tb * g.n_ff,
                                                       sh_g_ + tb, shared_ + tb * N, N, g.n_ff, cs,
                                                       lfuse_gate && dec_batch && n > 1,
                                                       g_ready, qdedup ? (const void*) win_.nat_xq(tb) : nullptr);
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
        }
        // APR: Hadamard-folded gate/up take H (s * x); xm itself stays as it is (the shared expert above and the
        // doorbell copy for the CPU pool, which rotates its own in native_quant_act).  The q8_1 image qdedup wrote for
        // the shared expert is of the unrotated input, so a folded layer rewrites it from the rotated rows (the same
        // stream, after the shared expert has read it).
        if (const auto& lay = strata::kernels::cpu::expert_layout(); lay.native && lay.fmt[(size_t) l].had_gu) {
            hadamard_rows(xm, nat_xr_ + (size_t) tb * N, n, N, lay.fmt[(size_t) l].had_block, had_sx_, cs);
            quantize_q8_1_rows(nat_xr_ + (size_t) tb * N, n, N, win_.nat_xq(tb), cs);
        } else if (!qdedup) {
            win_.expert_input(tb, n, xm, cs);
        }
        stamp(l, 18, grp);
        return true;
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) ((l - lb_) * G + grp + 1);
        MoeExpertsMode mode;
        mode.resident = recording_res_;
        mode.device_plan = device_plan_;
        mode.rows_only = dec_batch;
        mode.fetch_side = fetch_branch_ ? fetch_side_ : nullptr;   // the PCIe share's fetch beside the VRAM hits
        mode.had_signs = had_sh_;
        if (prof_on_ && grp == 0) mode.stamp = [&stamp, l, grp](int i) { stamp(l, i, grp); };
        if (!win_.experts(l, ring, grp, G, tb, n, cs, err, mode)) return false;
        if (dec_batch && n > 1) {   // one launch for the window's rows
            try {
                win_.combine(grp, tb, n, w_ + tb * K, shared_ + tb * N, bo_ + tb * N, gate_later_[grp] ? sh_g_ + tb : nullptr,
                             cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else {
            win_.add_hits(grp, tb, n, cs);
            for (int t = tb; t < te; ++t) {
                MoEBuffers mb = ss.moe;
                mb.weights = w_ + t * K; mb.shared = shared_ + t * N;
                if (!moe_combine_parts(g, l, K, mb, win_.parts(t), bo_ + t * N, cs, err)) return false;
            }
        }
        stamp(l, 24, grp);
        if (l == g.n_layers - 1) {
            if (dec_batch) gr_write_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, gs, Rt(tb), n, cs);
            else for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
            if (cvec().covers(l)) cvec_apply(Rt(tb), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(tb), l, n, HC * N, bo_ + tb * N, N, inj2_ + tb * HC, HC, true, cs);
        }
        if (!tap_layers_.empty() && !batch_rec_ && l < (int64_t) tap_slot_.size() && tap_slot_[(size_t) l] >= 0) {
            // the layer's completed output, the mean of the hc streams: with a control vector the write has landed
            // in R; without one it is still pending (the next layer's read folds it), so the tap applies it itself
            const int64_t nt = (int64_t) tap_layers_.size();
            float* dst = tap_dev_ + ((size_t) tb * nt + (size_t) tap_slot_[(size_t) l]) * N;
            if (cvec().covers(l))
                strata::kernels::df2_tap_rows(Rt(tb), dst, nt * N, N, (int) HC, n, cs);
            else
                strata::kernels::df2_tap_gr(Rt(tb), HC * N, bo_ + tb * N, N, inj2_ + tb * HC, HC, dst, nt * N, n, N,
                                            (int) HC, cs);
        }
        return true;
    };

    // segmented: a graph ends where the host must answer a ring (before each group's experts)
    auto cut = [&]() -> bool {
        if (!win_.segmented() || recording_res_) return true;
        strata::gpu::Graph* e = nullptr;
        if (!strata::gpu::end_capture(cs, &e)) {
            err = std::string("verify: segment capture: ") + strata::gpu::last_error();
            return false;
        }
        seg_out_->push_back(e);
        if (!strata::gpu::begin_capture(cs)) { err = "verify: begin capture failed"; return false; }
        return true;
    };
    for (int grp = 0; grp < G; ++grp)
        if (!pre(lb_, grp)) return false;
    for (int64_t l = lb_; l < le_; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!cut() || !post(l, grp)) return false;
            if (l + 1 < le_ && !pre(l + 1, grp)) return false;
        }
    if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        for (int t = 0; t < T; ++t) {
            float* hout = hand_out_ + (size_t) (hrow0 + t) * HB;
            copy_from_mapped(hout, Rt(t), HC * N, cs);
            copy_from_mapped(hout + HC * N, bo_ + (size_t) t * N, N, cs);
            copy_from_mapped(hout + HC * N + N, inj2_ + (size_t) t * HC, HC, cs);
        }
        return true;
    }

    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        // STRATA_HC_Q8=1: the final mixer's Q8_0 projections are read in this form only
        const bool mix_q8 = hd->hc_q8 != nullptr && hu->hc_q8 != nullptr;
        const bool mix_multi = (head_mix_multi_enabled() || mix_q8) && head_ != nullptr && head_->loaded() &&
                               fused_gr_supported(g.n_embd, g.hc, g.hc_lr);
        if (mix_multi) {
            // the final mixer, the window's tokens in one read (no pending write: the last layer's was done above)
            if (hn->kind != WeightKind::F32 || hd->kind != WeightKind::Bf16InF32 || hu->kind != WeightKind::Bf16InF32) {
                err = "verify: the output_hc_* weights have the wrong engine forms";
                return false;
            }
            try {
                for (int c0 = 0; c0 < T; c0 += kFusedGrMaxT) {   // (reads of at most kFusedGrMaxT rows)
                    const int c1 = std::min(T, c0 + kFusedGrMaxT);
                    FusedGrArgs fa[kFusedGrMaxT];
                    for (int t = c0; t < c1; ++t) {
                        FusedGrArgs& a = fa[t - c0];
                        a.R = Rt(t); a.R_out = Rt(t); a.apply = false;
                        a.w_norm = (const float*) hn->data; a.w_down = (const uint16_t*) hd->data;
                        a.w_up = (const uint16_t*) hu->data; a.eps = EPS;
                        if (mix_q8) { a.q8_down = (const uint8_t*) hd->hc_q8; a.q8_up = (const uint8_t*) hu->hc_q8; }
                        a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC; a.mixed = head_mixed_ + t * N;
                    }
                    fused_gr_read_multi(fa, c1 - c0, xn_ + (size_t) c0 * HC * N, cs);
                }
            } catch (const std::exception& e) {
                err = std::string("verify head mixer: ") + e.what();
                return false;
            }
        }
        for (int t = 0; t < T && !mix_multi; ++t) {
            BlockBuffers bb = ss.block;
            bb.R = Rt(t);
            bb.mixed = head_mixed_ + t * N;
            if (head_ != nullptr && head_->loaded()) {
                if (!lm_head_mix(wt, g, bb, cs, err)) return false;
            } else if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                return false;
            }
        }
        if (head_ != nullptr && head_->loaded()) {
            try {
                native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
        // #879: whether each row's logits are all finite, beside the pick (read after the window's own sync)
        if (nan_guard_on()) strata::kernels::logits_nonfinite_rows(head_logits_, T, (int) n_vocab_, m_nf_, cs);
    }
    stamp(g.n_layers, 1, 0);
    return true;
}

std::string Verifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {"-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
                                          "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
                                          "attention", "gate", "", "out-proj", "hc-read1+router", "shared+quant",
                                          "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
                                          "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "", "", ""};
    std::string out;
    char b[80];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN layers:" : " | QSA layers:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] / 1e6 / (double) prof_windows_);
            out += b;
        }
    }
    std::snprintf(b, sizeof b, " | total %.2f M ticks/window over %lld windows", total / 1e6 / (double) prof_windows_, (long long) prof_windows_);
    out += b;
    for (auto& r : prof_sum_) for (double& d : r) d = 0;
    prof_windows_ = 0;
    return out;
}

bool Verifier::window_fits(int T) const {
    if (T < 1 || T > max_t_) return false;
    if (exec_[T] != nullptr || !segs_[T].empty() || exec_res_[T] != nullptr || graph_bytes_max_ == 0) return true;
    size_t free_b = 0, total_b = 0;
    if (!strata::gpu::mem_info(&free_b, &total_b)) return true;
    return free_b >= graph_bytes_max_ / 2 * 3 + ((size_t) 64 << 20);
}

bool Verifier::capture(int T, std::string& err) {
    if (exec_[T] != nullptr || !segs_[T].empty()) return true;
    size_t free0 = 0;
    {   // said before the capture: a process that exits inside it leaves this line as the trace (upstream 0f931943)
        size_t total_b = 0;
        if (strata::gpu::mem_info(&free0, &total_b))
            std::fprintf(stderr, "xestrata verify: capturing the %d-token window (%zu MiB of VRAM free)\n", T, free0 >> 20);
    }
    if (!strata::gpu::begin_capture(cs_)) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    bool ok = false;
    seg_out_ = &segs_[T];
    try {
        ok = record_window(T, cs_, rerr);
    } catch (const std::exception& e) {
        rerr = std::string("verify: window capture: ") + e.what();
    }
    seg_out_ = nullptr;
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        for (auto* e : segs_[T]) strata::gpu::graph_destroy(e);
        segs_[T].clear();
        err = rerr;
        return false;
    }
    strata::gpu::Graph* last = nullptr;
    if (!strata::gpu::end_capture(cs_, &last)) {
        err = std::string("verify: end capture: ") + strata::gpu::last_error();
        return false;
    }
    if (win_.segmented()) segs_[T].push_back(last);
    else exec_[T] = last;
    if (strata::env_get("VERIFY_NODES") != nullptr)   // what the window graph holds (SYCL names no kernels)
        std::fprintf(stderr, "xestrata verify: the %d-token window graph has %zu nodes\n", T,
                     strata::gpu::graph_nodes(win_.segmented() ? segs_[T].back() : exec_[T]));
    const bool synced = strata::gpu::stream_sync(cs_);
    std::fprintf(stderr, "xestrata verify: captured the %d-token window (sync %s)\n", T,
                 synced ? "ok" : strata::gpu::last_error());
    graph_free0_ = free0;   // what the graph takes shows at its first run, measured once run() has the window's tokens
    return true;
}

bool Verifier::all_resident() const {
    const int64_t ne = g_->n_expert;
    for (int64_t i = lb_ * ne; i < le_ * ne; ++i)
        if (hits_.h_res[i] < 0) return false;
    return true;
}

bool Verifier::capture_res(int T, std::string& err) {
    if (exec_res_[T] != nullptr) return true;
    if (!strata::gpu::begin_capture(cs_)) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    bool ok = false;
    recording_res_ = true;
    try {
        ok = record_window(T, cs_, rerr);
    } catch (const std::exception& e) {
        rerr = std::string("verify: window capture: ") + e.what();
    }
    recording_res_ = false;
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        err = rerr;
        return false;
    }
    if (!strata::gpu::end_capture(cs_, &exec_res_[T])) {
        err = std::string("verify: end capture: ") + strata::gpu::last_error();
        return false;
    }
    const bool synced = strata::gpu::stream_sync(cs_);
    std::fprintf(stderr, "xestrata verify: captured the %d-token window with every expert resident (sync %s)\n", T,
                 synced ? "ok" : strata::gpu::last_error());
    return true;
}

bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (!strata::gpu::begin_capture(cs_)) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        // STRATA_COMMIT_BATCH (default on; =0 keeps the per-layer loop, upstream 6f3e3810): every GDN layer's conv
        // history and state in two launches, and every QSA layer's tail restore and indexer append in one, instead of
        // two or three per layer.  The same arithmetic per layer, bitwise.  The commit runs between a window and the
        // draft round.
        static const bool commit_batch = [] {
            const char* e = strata::env_get("COMMIT_BATCH");
            return e == nullptr || std::strtol(e, nullptr, 10) != 0;
        }();
        bool gdn_batched = false, qsa_batched = false;
        if (commit_batch) {
            int64_t n_gdn = 0;
            for (int64_t l = lb_; l < le_ && ok; ++l)   // the per-layer loop's check of each layer's ssm_norm.weight
                if (!is_qsa_layer(g, l)) {
                    ++n_gdn;
                    ok = need(LayerView(*wt_, l), "ssm_norm.weight", err) != nullptr;
                }
            if (ok && n_gdn > 0)
                gdn_batched = gdn_commit_layers(
                    ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats, (int64_t) gdn_floats,
                    (int64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size, qkv_L_ + (size_t) gdn_index * MT * C,
                    MT * C, h_L_ + (size_t) gdn_index * MT * C, MT * C, gate_L_ + (size_t) gdn_index * MT * HV,
                    beta_L_ + (size_t) gdn_index * MT * HV, MT * HV, (int) C, (int) g.ssm_k_heads, (int) HV, (int) MT,
                    (int) n_gdn, commit_, cs_);
        }
        if (commit_batch && ok) {
            QsaIndexerLayers il;
            int nq = 0;
            int64_t mc = -1;
            bool fits = true;
            for (int64_t l = lb_, qi = qsa_index; l < le_ && fits; ++l) {
                if (!is_qsa_layer(g, l)) continue;
                const WeightRef* wikn = need(LayerView(*wt_, l), "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                const QsaState& st = ss.qsa_states[qi];
                if (nq >= QsaIndexerLayers::kMax || (mc >= 0 && st.max_cells != mc)) { fits = false; break; }
                mc = st.max_cells;
                il.gamma[nq] = (const float*) wikn->data;
                il.tail_src[nq] = tail_snap_ + (size_t) qi * TS;
                il.tail[nq] = st.idx_tail;
                il.dead[nq] = st.idx_dead;
                il.pooled[nq] = st.idx_pooled;
                il.block_pos[nq] = st.idx_block_pos;
                ++nq;
                ++qi;
            }
            if (ok && fits && nq > 0) {
                native_qsa_indexer_append_steps_layers(idx_raw_L_ + (size_t) qsa_index * MT * ID, MT * ID, commit_ + 2, 1,
                                                       (int) MT, 0, EPS, il, nq, s, mc, rope_scaling(), cs_);
                qsa_batched = true;
            }
        }
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (is_qsa_layer(g, l) ? qsa_batched : gdn_batched) {
                ++(is_qsa_layer(g, l) ? qsa_index : gdn_index);
            } else if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = qkv_L_ + (size_t) gdn_index * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C, (int) C, gate_L_ + (size_t) gdn_index * MT * HV,
                                    beta_L_ + (size_t) gdn_index * MT * HV, z_, (const float*) wnm->data, EPS, y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, commit_, cs_,
                                    (int) MT);   // the state only: no token's output is read (upstream 22abb92d)
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                native_qsa_indexer_append_steps(idx_raw_L_ + (size_t) qsa_index * MT * ID, commit_ + 2, 1, (int) MT, 0,
                                                (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                rope_scaling(), cs_);
                ++qsa_index;
            }
        }
        if (ok && ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        return false;
    }
    if (!strata::gpu::end_capture(cs_, &commit_exec_)) {
        err = std::string("verify: commit capture: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    last_batch_ = false;
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) {
        err = "verify: the window runs past the context";
        return false;
    }
    if (rt_dump_ || rt_replay_) {   // RouteTrace: a window whose position does not pass the last one's begins a request
        if (pos0 <= rt_pos0_) ++rt_req_;
        rt_pos0_ = pos0;
        if (rt_replay_ && !route_trace().fill(rt_req_, pos0, T, lb_, le_, ss.k, max_t_, h_rpl_ids_, h_rpl_w_, err))
            return false;
    }
    const bool res = res_graph_ && all_resident();
    captured_last_ = res ? exec_res_[T] == nullptr : exec_[T] == nullptr && segs_[T].empty();
    if (!(res ? capture_res(T, err) : capture(T, err)) || !capture_commit(err)) return false;
    VDBG("captured; staging\n");
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + (size_t) t * kStepCount, pos0 + t, s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) (pos0 + t);
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) (pos0 + t);
    }
    const bool do_ple = ss.ple.ready() && ple_stage();
    const bool ple_late = do_ple && !res && lb_ == 0;   // read after the launch below (record_window's ple_late)
    uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];
    if (do_ple) {
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, ple_rows + (size_t) t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss.ple.table->prefetch_rows(ple_rows + (size_t) t * PLE_N_HEADS);   // the rows the drafter did not read ahead
        }
        if (!ple_late && !ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
    }
    win_.reset();
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    ms_host += ms_since(t0);
    VDBG("staged; launching\n");
    const bool le = strata::gpu::graph_launch(res ? exec_res_[T] : win_.segmented() ? segs_[T][0] : exec_[T], cs_);
    if (!le) { err = std::string("verify: launch: ") + strata::gpu::last_error(); return false; }
    if (!res && !win_.mark_done(cs_, err)) return false;   // wait_ring reads DEVICE_LOST from this event (#78)
    (void) strata::gpu::stream_idle(cs_);
    VDBG("launched\n");
    if (ple_late) {   // while the GPU runs layer 0 up to its experts; layer 1 waits for layer 0's flag
        const Clock::time_point tp = Clock::now();
        if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) {
            // the window is running: an empty plan and every flag raised let it finish before the error returns
            win_.release(cs_);
            return false;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ms_host += ms_since(tp);
    }
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    for (int64_t k = 0; !res && k < (le_ - lb_) * G; ++k) {   // the resident graph rings nothing
        const int64_t l = lb_ + k / G;
        const int grp = (int) (k % G);
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        progress_at("verify window: waiting for the GPU to reach layer", l);
        if (!win_.wait_ring(want, cs_, "verify", l, err)) return false;
        const Clock::time_point b = Clock::now();
        VDBG("layer %lld rang\n", (long long) l);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window: the CPU experts of layer", l);
        win_.answer(want, grp, G, tb, n, l, pool, user);
        VDBG("layer %lld served\n", (long long) l);
        progress_tick();
        // the PCIe share's copies too, then the next segment
        if (win_.segmented() && !win_.next_segment(want, segs_[T][(size_t) k + 1], cs_, "verify", l, err)) return false;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    const bool se = strata::gpu::stream_sync(cs_);
    if (!se) { err = std::string("verify: ") + strata::gpu::last_error(); return false; }
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    win_.sync_copies();   // no host function of this window may raise flag B in the next one
    if (rt_dump_) route_trace().write(rt_req_, pos0, T, lb_, le_, ss.k, max_t_, h_route_, h_rw_);
    if (prof_on_ && G == 1) {       // the window's GPU stage stamps
        strata::gpu::copy(prof_h_.data(), prof_, prof_h_.size() * 8);
        accumulate_profile(prof_h_.data());
    }
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the POSITION it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)
        ++windows;
        if (next_ == nullptr) return true;
        next_->rows(hand_out_, 0, T);
        return next_->run(T, tokens, pos0, pool, out, err);
    }
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr || sampling_.logit_bias != nullptr ||
                           sampling_.allowed != nullptr)) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (!strata::gpu::stream_sync(cs_)) {   // m_out_ is the mapped h_out_: synced, it is readable
            err = "verify: the head sampling failed";
            return false;
        }
    }
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(T, nullptr);
    if (static const bool dbg = strata::env_get("DBG_NAN") != nullptr; dbg) {   // debug: the first non-finite head
        static bool reported = false;
        if (!reported) {
            std::vector<float> h((size_t) T * (size_t) n_vocab_);
            strata::gpu::copy(h.data(), head_logits_, h.size() * 4);
            for (int t = 0; t < T && !reported; ++t) {
                int64_t bad = 0;
                for (int64_t v = 0; v < n_vocab_; ++v) bad += !std::isfinite(h[(size_t) t * n_vocab_ + v]);
                if (bad) {
                    reported = true;
                    std::fprintf(stderr, "xestrata dbg: verify window at position %lld, row %d: %lld of %lld logits non-finite "
                                         "(token out %d)\n", (long long) pos0, t, (long long) bad, (long long) n_vocab_, out[t]);
                }
            }
        }
    }
    VDBG("window done\n");
    if (graph_free0_ > 0) {
        size_t free1 = 0, total_b = 0;
        if (strata::gpu::mem_info(&free1, &total_b) && free1 < graph_free0_)
            graph_bytes_max_ = std::max(graph_bytes_max_, graph_free0_ - free1);
        graph_free0_ = 0;
    }
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}

void Verifier::add_routing(float* usage, int64_t n_expert, int rows) const {
    if (h_route_ == nullptr || !route_rec_) return;
    const int64_t K = ss_->k;
    rows = std::min(rows, last_t_);
    for (int64_t l = lb_; l < le_; ++l)
        for (int t = 0; t < rows; ++t) {
            const volatile int32_t* ids = h_route_ + ((size_t) l * (size_t) max_t_ + (size_t) t) * (size_t) K;
            for (int64_t j = 0; j < K; ++j) {
                const int32_t e = ids[j];
                if (e >= 0 && e < n_expert) usage[(size_t) l * (size_t) n_expert + (size_t) e] += 1.0f;
            }
        }
}

void Verifier::accumulate_profile(const unsigned long long* stamps) {
    const ModelGeometry& g = *g_;
    const int64_t L = g.n_layers;
    auto at = [&](int64_t l, int i) { return stamps[(size_t) (l * kProfPer + i)]; };
    // The derived columns read stamps the hc-read kernels write themselves or the next layer's first.  A slot no
    // kernel stamped is 0 and its unsigned difference wrapped to ~1e19 ns; a missing or out-of-order stamp now
    // contributes nothing (upstream e5b47dd).
    const auto gap = [](unsigned long long to, unsigned long long from) {
        return (from != 0 && to != 0 && to >= from) ? (double) (to - from) : 0.0;
    };
    // only this stage's layers [lb_, le_) are ever stamped (a layer split); the head only on the last stage
    for (int64_t l = lb_; l < le_; ++l) {
        const int kind = is_qsa_layer(g, l) ? 1 : 0;
        unsigned long long prev = at(l, 0);
        for (int i = 1; i <= 24; ++i) {
            const unsigned long long x = at(l, i);
            if (x == 0 || x < prev) continue;
            prof_sum_[kind][i] += (double) (x - prev);
            prev = x;
        }
        if (l + 1 < le_) prof_sum_[kind][25] += gap(at(l + 1, 0), at(l, 24));
        const double dn = gap(at(l, 27), at(l, 0)), dd = gap(at(l, 28), at(l, 27)), du = gap(at(l, 1), at(l, 28));
        if (dn > 0 && dd > 0 && du > 0) {   // the split exists: show it split, not twice
            prof_sum_[kind][27] += dn;      // hc-read0: norm
            prof_sum_[kind][28] += dd;      //           down
            prof_sum_[kind][29] += du;      //           up (through both halves, as before)
            prof_sum_[kind][1] -= gap(at(l, 1), at(l, 0));   // (hc-read0 shown split)
        }
    }
    if (le_ == L) prof_sum_[0][26] += gap(at(L, 1), at(L, 0));
    ++prof_windows_;
}

bool Verifier::read_logits(int rows, float* host, std::string& err) const {
    if (le_ < g_->n_layers) return next_ != nullptr && next_->read_logits(rows, host, err);
    const OnDevice on_device(device_);   // a layer split's last stage: its GPU's queue, not the caller's
    if (!strata::gpu::copy(host, head_logits_, (size_t) rows * (size_t) n_vocab_ * 4)) {
        err = "verify: reading the head's logits back failed";
        return false;
    }
    return true;
}

bool Verifier::wait_commit(std::string& err) {
    if (commit_pending_) {
        const OnDevice on_device(device_);
        commit_pending_ = false;
        if (!strata::gpu::event_sync(commit_done_)) {
            err = std::string("verify: commit: ") + strata::gpu::last_error();
            return false;
        }
    }
    return next_ == nullptr || next_->wait_commit(err);
}

// ================================ BATCH WINDOWS (see init_slots; upstream PR #559) ================================

bool Verifier::init_slots(const std::vector<SessionState*>& slots, std::string& err) {
    const OnDevice on_device(device_);
    if (g_ == nullptr || ss_ == nullptr) { err = "verify: init_slots before init"; return false; }
    if (win_.segmented()) { err = "verify: batch windows need a GPU whose running kernels see the host's writes"; return false; }
    if (slots.empty()) {
        err = "verify: init_slots needs at least one session";
        return false;
    }
    for (SessionState* x : slots)
        if (x == nullptr || x->max_cells != ss_->max_cells) {
            err = "verify: a slot's session is not carved like the verifier's own (context)";
            return false;
        }
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t S = (int64_t) slots.size(), CB = 2 + max_t_;
    const int64_t TS = (s.idx_block - 1) * g_->idx_key_dim, nQ = g_->n_qsa_layers();
    if (!mapped((size_t) (S * CB * 4 + 16), (void**) &h_commitb_, (void**) &m_commitb_)) {
        err = "verify: the batch commit staging failed";
        return false;
    }
    const uint64_t a = ((uint64_t) S * CB * 4 + 255) & ~255ull;
    if (!strata::gpu::alloc_device(&arena_b_, a + (uint64_t) S * std::max<int64_t>(nQ, 1) * TS * 4)) {
        err = "verify: the batch buffers do not fit";
        return false;
    }
    commitb_ = (int32_t*) arena_b_;
    tail_snap_b_ = reinterpret_cast<float*>(static_cast<uint8_t*>(arena_b_) + a);
    slots_ = slots;
    slot_sp_.assign(slots.size(), sampling_);   // greedy until set_slot_sampling
    slot_bias_dev_.assign(slots.size(), nullptr);
    slot_tail_.assign(slots.size(), {});
    std::fprintf(stderr, "xestrata verify: batch windows of up to %lld sequences\n", (long long) S);
    return true;
}

bool Verifier::capture_batch(const int* rows, int S, int hbase, std::string& err) {
    strata::gpu::Graph*& ex = exec_bm_[batch_key(rows, S, hbase)];
    if (ex != nullptr) return true;
    {   // auto: time the batch window's block scores here, before the graph holds one
        if (strata::core::pick_from_env("BATCH_ATTN_ROWS") == strata::core::Pick::Auto && g_ != nullptr &&
            g_->n_qsa_layers() > 0)
            strata::kernels::qsa_rows_pays(S, max_blocks_, shapes_of(*g_), cs_);
    }
    if (!strata::gpu::begin_capture(cs_)) { err = "verify: begin batch capture failed"; return false; }
    batch_rec_ = true;
    row_base_ = hbase;
    for (int t = 0; t < S; ++t) brow_[t] = rows[t];
    std::string rerr;
    bool ok = false;
    try {
        ok = record_window(S, cs_, rerr);
    } catch (const std::exception& e) {
        rerr = std::string("verify: batch capture: ") + e.what();
    }
    batch_rec_ = false;
    row_base_ = 0;
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        err = rerr;
        return false;
    }
    if (!strata::gpu::end_capture(cs_, &ex)) {
        err = std::string("verify: end batch capture: ") + strata::gpu::last_error();
        return false;
    }
    const bool synced = strata::gpu::stream_sync(cs_);
    std::string list;
    for (int t = 0; t < S; ++t) list += (t ? "," : "") + std::to_string(rows[t]);
    std::fprintf(stderr, "xestrata verify: captured the batch window over slots %s (sync %s)\n", list.c_str(),
                 synced ? "ok" : strata::gpu::last_error());
    return true;
}

bool Verifier::capture_commit_batch(const int* rows, int S, int hbase, std::string& err) {
    strata::gpu::Graph*& cex = commit_bm_[batch_key(rows, S, hbase)];
    if (cex != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ZV = g.ssm_value_dim, ID = g.idx_key_dim, MT = max_t_;
    const int64_t CB = 2 + MT, nQ = g.n_qsa_layers();
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (!strata::gpu::begin_capture(cs_)) { err = "verify: begin batch commit capture failed"; return false; }    bool ok = true;
    try {
        for (int t = 0; t < S; ++t)
            if (t == 0 || rows[t] != rows[t - 1])
                copy_i32_from_mapped(commitb_ + (size_t) rows[t] * CB, m_commitb_ + (size_t) rows[t] * CB, CB, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                GdnGroups gg;
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    SessionState& sx = *slots_[(size_t) rows[first]];
                    const int32_t* keep = commitb_ + (size_t) rows[first] * CB;
                    float* state = sx.gdn_state + (size_t) (gdn_index - sx.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    gdn_conv_commit(conv, qkv_L_ + (size_t) gdn_index * MT * C + (size_t) first * C, (int) C, keep, cs_);
                    gg.state[gg.n] = state;
                    gg.keep[gg.n] = keep;
                    gg.first[gg.n] = first;
                    gg.ntok[gg.n] = t - first;
                    ++gg.n;
                }
                for (int i = 0; i < gg.n; ++i)
                    gdn_step_norm_multi(gg.state[i], h_L_ + (size_t) gdn_index * MT * C + (size_t) gg.first[i] * C,
                                        (int) C, gate_L_ + (size_t) gdn_index * MT * HV + (size_t) gg.first[i] * HV,
                                        beta_L_ + (size_t) gdn_index * MT * HV + (size_t) gg.first[i] * HV,
                                        z_ + (size_t) gg.first[i] * ZV, (const float*) wnm->data, EPS,
                                        y_dummy_ + (size_t) gg.first[i] * ZV, (int) g.ssm_k_heads, (int) HV,
                                        gg.ntok[i], gg.keep[i], cs_, gg.ntok[i] > 1 ? gg.ntok[i] : 0);
                ++gdn_index;
            } else {
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    const QsaState& st = slots_[(size_t) rows[first]]->qsa_states[qsa_index];
                    copy_from_mapped(st.idx_tail, tail_snap_b_ + ((size_t) rows[first] * nQ + qsa_index) * TS, TS, cs_);
                    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    for (int u = first; u < t; ++u)
                        native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + u) * ID,
                                                  commitb_ + (size_t) rows[first] * CB + 2 + (u - first), 0,
                                                  (const float*) wikn->data, EPS, ib, s, st.max_cells, rope_scaling(), cs_);
                }
                ++qsa_index;
            }
        }
        if (ok && ss_->ple.ready() && ple_stage())
            for (int t = 0; t < S;) {
                const int first = t;
                while (t < S && rows[t] == rows[first]) ++t;
                copy_indexed(slots_[(size_t) rows[first]]->ple_hist, hist_snap_ + (size_t) first * HS, HS,
                             commitb_ + (size_t) rows[first] * CB + 1, HS, cs_);
            }
    } catch (const std::exception& e) {
        err = std::string("verify batch commit: ") + e.what();
        ok = false;
    }
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        return false;
    }
    if (!strata::gpu::end_capture(cs_, &cex)) {
        err = std::string("verify: batch commit capture: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool Verifier::run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                         std::string& err) {
    int rows[strata::kernels::kVerifyMaxT] = {};
    for (int t = 0; t < S && t < strata::kernels::kVerifyMaxT; ++t) rows[t] = t;
    return run_slot_rows(rows, S, tokens, pos, pool, user, out, err);
}

bool Verifier::stage_batch(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos,
                           std::string& err) {
    using namespace strata::kernels;
    if (rt_dump_ || rt_replay_) { err = "XESTRATA_ROUTE_DUMP / XESTRATA_ROUTE_REPLAY: one request at a time (no batch slots)"; return false; }
    // a slot may own a contiguous group of rows (--batch-mtp), also across the stages of a layer split, whose hand-off
    // buffers hold kVerifyNarrowT rows (a stage that hands rows on or takes them; upstream 48f2c47d)
    // (STRATA_PIPED_MTP: set_handoff_rows holds more)
    const int hand_rows = handoff_rows_ > 0 ? handoff_rows_ : (int) kVerifyNarrowT;
    if (S < 1 || S > max_t_ || hbase < 0 || ((next_ != nullptr || lb_ > 0) && hbase + S > hand_rows)) {
        err = "verify: batch rows out of range (init_slots)";
        return false;
    }
    const ModelGeometry& g = *g_;
    for (int t = 0; t < S; ++t) {
        if (rows[t] < 0 || rows[t] >= (int) slots_.size()) {
            err = "verify: a batch row's slot is out of range";
            return false;
        }
        for (int u = 0; u < t - 1; ++u)
            if (rows[u] == rows[t] && rows[t - 1] != rows[t]) {
                err = "verify: a slot's proposed rows must be contiguous";
                return false;
            }
        if (t > 0 && rows[t] == rows[t - 1] && pos[t] != pos[t - 1] + 1) {
            err = "verify: proposed rows must have consecutive positions";
            return false;
        }
        if (pos[t] < 0 || pos[t] + 1 > slots_[(size_t) rows[t]]->max_cells) {
            err = "verify: slot " + std::to_string(rows[t]) + " runs past its context";
            return false;
        }
    }
    // --batch-mtp (a limit): slot rotation creates new layouts; bound the captured graph pairs, evicting the least
    // recently used layout.  Without a limit every layout is kept.
    const std::vector<int> key = batch_key(rows, S, hbase);
    if (batch_graph_limit_ > 0) {
        if (exec_bm_.find(key) == exec_bm_.end() && exec_bm_.size() >= batch_graph_limit_) {
            if (!strata::gpu::stream_sync(cs_)) {
                err = "verify: synchronizing before batch graph eviction failed";
                return false;
            }
            auto old = exec_bm_.begin();
            uint64_t oldest = UINT64_MAX;
            for (auto it = exec_bm_.begin(); it != exec_bm_.end(); ++it) {
                const auto u = bm_used_.find(it->first);
                const uint64_t used = u == bm_used_.end() ? 0 : u->second;
                if (used < oldest) { oldest = used; old = it; }
            }
            const std::vector<int> old_key = old->first;
            if (old->second) strata::gpu::graph_destroy(old->second);
            exec_bm_.erase(old);
            bm_used_.erase(old_key);
            for (auto* m : {&commit_bm_, &save_bm_, &dcommit_bm_}) {   // (the last two: STRATA_PIPED_MTP's)
                const auto it = m->find(old_key);
                if (it == m->end()) continue;
                if (it->second) strata::gpu::graph_destroy(it->second);
                m->erase(it);
            }
        }
        bm_used_[key] = ++bm_tick_;
    }
    // (a pipelined window that commits after the verdict has its own commit graph: capture_deferred)
    if (!capture_batch(rows, S, hbase, err) || (!staging_deferred_ && !capture_commit_batch(rows, S, hbase, err)))
        return false;
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < S; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + (size_t) t * kStepCount, pos[t], s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) pos[t];
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) pos[t];
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) pos[t];
    }
    if (ss_->ple.ready() && ple_stage()) {
        uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];   // (not `rows`: that is the slots of the window's rows)
        for (int t = 0; t < S; ++t) {
            const SessionState& sx = *slots_[(size_t) rows[t]];
            int32_t prev[2] = {sx.ple_prev[0], sx.ple_prev[1]};
            if (t > 0 && rows[t] == rows[t - 1]) {   // a proposal row follows its slot's row before it
                prev[0] = t > 1 && rows[t - 2] == rows[t] ? tokens[t - 2] : sx.ple_prev[1];
                prev[1] = tokens[t - 1];
            }
            ngram_rows(&tokens[t], prev, 1, ss_->ple.consts, ple_rows + (size_t) t * PLE_N_HEADS);
        }
        if (!ss_->ple.table->gather_batch(ple_rows, (size_t) S, h_ple_, err)) return false;
    }
    // the commit's rows: each slot owns a contiguous group, all of it kept unless commit_slot_prefixes says less
    const int64_t CB = 2 + max_t_;
    for (int t = 0; t < S;) {
        const int first = t;
        while (t < S && rows[t] == rows[first]) ++t;
        int32_t* c = h_commitb_ + (size_t) rows[first] * CB;
        c[0] = t - first;
        c[1] = t - first - 1;
        for (int64_t j = 0; j < max_t_; ++j) c[2 + j] = j < t - first ? (int32_t) pos[first + j] : -1;
    }
    win_.reset();
    last_t_ = S;
    row_base_ = hbase;   // commit_slots' graph key until the next stage_batch
    last_batch_ = true;
    for (int t = 0; t < S; ++t) { last_tokens_[t] = tokens[t]; last_pos_b_[t] = pos[t]; last_rows_[t] = rows[t]; }
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool,
                             void* user, int32_t* out, std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    if (!stage_batch(rows, S, 0, tokens, pos, err)) return false;
    if (!strata::gpu::graph_launch(exec_bm_[batch_key(rows, S, 0)], cs_)) {
        err = std::string("verify: batch launch: ") + strata::gpu::last_error();
        return false;
    }
    if (!win_.mark_done(cs_, err)) return false;
    (void) strata::gpu::stream_idle(cs_);
    for (int64_t k = 0; k < le_ - lb_; ++k) {
        const int64_t l = lb_ + k;
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        progress_at("verify batch: waiting for the GPU to reach layer", l);
        if (!win_.wait_ring(want, cs_, "verify batch", l, err)) return false;
        const Clock::time_point b = Clock::now();
        progress_at("verify batch: the CPU experts of layer", l);
        win_.answer(want, 0, 1, 0, S, l, pool, user);
        progress_tick();
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    if (!strata::gpu::stream_sync(cs_)) { err = std::string("verify batch: ") + strata::gpu::last_error(); return false; }
    win_.sync_copies();   // no host function of this window may raise flag B in the next one
    ++windows;
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the rows go on to the next stage, as in run()
        if (next_ == nullptr) return true;
        next_->rows(hand_out_, 0, S);
        return next_->run_slot_rows(rows, S, tokens, pos, pool, out, err);
    }
    if (!sample_rows(S, err)) return false;
    for (int t = 0; t < S; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(S, last_pos_b_);
    progress_at("decode");
    progress_beat();
    return true;
}

bool Verifier::commit_slots(std::string& err) {
    std::vector<int> keep(slots_.size(), 0);
    for (int t = 0; t < last_t_; ++t) ++keep[(size_t) last_rows_[t]];
    return commit_slot_prefixes(keep.data(), err);
}

bool Verifier::commit_slot_prefixes(const int* keep, std::string& err) {
    const OnDevice on_device(device_);
    if (!last_batch_ || last_t_ < 1) { err = "verify: commit_slots without a batch window"; return false; }
    const int S = last_t_;
    const Clock::time_point t0 = Clock::now();
    const int64_t CB = 2 + max_t_;
    for (int t = 0; t < S;) {   // one prefix a slot: rejected draft rows must not enter the recurrent state
        const int first = t;
        while (t < S && last_rows_[t] == last_rows_[first]) ++t;
        const int n = keep[last_rows_[first]];
        if (n < 1 || n > t - first) {
            err = "verify: an accepted prefix outside its slot group";
            return false;
        }
        int32_t* c = h_commitb_ + (size_t) last_rows_[first] * CB;
        c[0] = n;
        c[1] = n - 1;
        for (int j = 0; j < max_t_; ++j) c[2 + j] = j < n ? (int32_t) last_pos_b_[first + j] : -1;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!strata::gpu::graph_launch(commit_bm_[batch_key(last_rows_, S, row_base_)], cs_) ||
        !strata::gpu::stream_sync(cs_)) {
        err = std::string("verify: batch commit: ") + strata::gpu::last_error();
        return false;
    }
    if (ple_stage())
        for (int t = 0; t < S;) {
            const int first = t;
            while (t < S && last_rows_[t] == last_rows_[first]) ++t;
            SessionState& sx = *slots_[(size_t) last_rows_[first]];
            for (int u = first; u < first + keep[last_rows_[first]]; ++u) {
                sx.ple_prev[0] = sx.ple_prev[1];
                sx.ple_prev[1] = last_tokens_[u];
            }
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit_slot_prefixes(keep, err);
}

bool Verifier::batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err) {
    const OnDevice on_device(device_);
    if (b_running_) { err = "verify: batch_launch while this stage is busy"; return false; }
    int rows[strata::kernels::kVerifyMaxT] = {};
    for (int t = 0; t < S && t < strata::kernels::kVerifyMaxT; ++t) rows[t] = base + t;
    if (!stage_batch(rows, S, base, tokens, pos, err)) return false;
    // the commit right behind the window: every row of a batch window is kept
    if (!strata::gpu::graph_launch(exec_bm_[batch_key(rows, S, base)], cs_) ||
        !strata::gpu::graph_launch(commit_bm_[batch_key(rows, S, base)], cs_)) {
        err = std::string("verify: batch launch: ") + strata::gpu::last_error();
        return false;
    }
    (void) strata::gpu::stream_idle(cs_);
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens)
        for (int t = 0; t < S; ++t) {
            SessionState& sx = *slots_[(size_t) rows[t]];
            sx.ple_prev[0] = sx.ple_prev[1];
            sx.ple_prev[1] = tokens[t];
        }
    b_running_ = true;
    b_k_ = 0;
    b_steps_ = le_ - lb_;
    b_last_ = Clock::now();
    return true;
}

// ---- STRATA_PIPED_MTP: commits after the verdict (see the header; upstream 17878ac57)

bool Verifier::init_deferred_commit(int rows, std::string& err) {
    const OnDevice on_device(device_);
    if (slots_.empty()) { err = "verify: init_deferred_commit before init_slots"; return false; }
    if (rows < 1) { err = "verify: init_deferred_commit needs at least one row"; return false; }
    const ModelGeometry& g = *g_;
    const uint64_t R = (uint64_t) rows, C = (uint64_t) g.ssm_conv_channels, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) std::max<int64_t>(g.n_gdn_layers(), 1), nQ = (uint64_t) std::max<int64_t>(g.n_qsa_layers(), 1);
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t CB = 2 + (uint64_t) max_t_, nS = (uint64_t) slots_.size();
    const uint64_t floats[6] = {nG * R * C, nG * R * C, nG * R * HV, nG * R * HV, nQ * R * ID, R * HS};
    uint64_t bytes = (nS * CB * 4 + 255) & ~255ull;
    for (uint64_t f : floats) bytes += (f * 4 + 255) & ~255ull;
    if (!strata::gpu::alloc_device(&dc_arena_, bytes)) {
        err = "verify: the saved commit rows (" + std::to_string(bytes >> 20) + " MiB) do not fit";
        return false;
    }
    uint8_t* p = static_cast<uint8_t*>(dc_arena_);
    commitd_ = reinterpret_cast<int32_t*>(p);
    p += (nS * CB * 4 + 255) & ~255ull;
    float** dst[6] = {&dc_qkv_, &dc_h_, &dc_gate_, &dc_beta_, &dc_idx_, &dc_hist_};
    for (int i = 0; i < 6; ++i) {
        *dst[i] = reinterpret_cast<float*>(p);
        p += (floats[i] * 4 + 255) & ~255ull;
    }
    if (!mapped((size_t) (nS * CB * 4 + 16), (void**) &h_commitd_, (void**) &m_commitd_)) {
        err = "verify: the deferred commit staging failed";
        return false;
    }
    dc_rows_ = rows;
    std::fprintf(stderr, "xestrata verify: pipelined batch windows commit after the verdict (layers [%lld, %lld), %d rows, "
                         "%.1f MiB saved rows)\n", (long long) lb_, (long long) le_, rows, (double) bytes / 1048576.0);
    return true;
}

bool Verifier::capture_deferred(const int* rows, int S, int hbase, std::string& err) {
    using namespace strata::kernels;
    const std::vector<int> key = batch_key(rows, S, hbase);
    strata::gpu::Graph*& sex = save_bm_[key];
    strata::gpu::Graph*& cex = dcommit_bm_[key];
    if (sex != nullptr && cex != nullptr) return true;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ZV = g.ssm_value_dim, ID = g.idx_key_dim, MT = max_t_;
    const int64_t R = dc_rows_, CB = 2 + MT, nQ = g.n_qsa_layers();
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const bool ple = ss_->ple.ready() && ple_stage();
    auto layer_index = [&](int64_t& qsa_index, int64_t& gdn_index) {
        qsa_index = gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
    };
    auto finish = [&](strata::gpu::Graph*& ex, bool ok, const char* what) {
        if (!ok) {
            strata::gpu::abandon_capture(cs_);
            if (err.empty()) err = std::string(what) + "failed";
            return false;
        }
        if (!strata::gpu::end_capture(cs_, &ex)) {
            err = std::string(what) + strata::gpu::last_error();
            return false;
        }
        return true;
    };
    // the save: the window's rows [0, S) of the scratch the commit replays -> the group's saved rows [hbase, +S)
    if (sex == nullptr) {
        if (!strata::gpu::begin_capture(cs_)) { err = "verify: begin save capture failed"; return false; }
        bool ok = true;
        try {
            int64_t qsa_index = 0, gdn_index = 0;
            layer_index(qsa_index, gdn_index);
            for (int64_t l = lb_; l < le_; ++l) {
                if (!is_qsa_layer(g, l)) {
                    copy_from_mapped(dc_qkv_ + (size_t) (gdn_index * R + hbase) * C, qkv_L_ + (size_t) gdn_index * MT * C,
                                     S * C, cs_);
                    copy_from_mapped(dc_h_ + (size_t) (gdn_index * R + hbase) * C, h_L_ + (size_t) gdn_index * MT * C,
                                     S * C, cs_);
                    copy_from_mapped(dc_gate_ + (size_t) (gdn_index * R + hbase) * HV,
                                     gate_L_ + (size_t) gdn_index * MT * HV, S * HV, cs_);
                    copy_from_mapped(dc_beta_ + (size_t) (gdn_index * R + hbase) * HV,
                                     beta_L_ + (size_t) gdn_index * MT * HV, S * HV, cs_);
                    ++gdn_index;
                } else {
                    copy_from_mapped(dc_idx_ + (size_t) (qsa_index * R + hbase) * ID,
                                     idx_raw_L_ + (size_t) qsa_index * MT * ID, S * ID, cs_);
                    ++qsa_index;
                }
            }
            if (ple) copy_from_mapped(dc_hist_ + (size_t) hbase * HS, hist_snap_, S * HS, cs_);
        } catch (const std::exception& e) {
            err = std::string("verify save: ") + e.what();
            ok = false;
        }
        if (!finish(sex, ok, "verify: save capture: ")) return false;
    }
    // the commit: as capture_commit_batch, replaying the saved rows; the accepted prefixes come from commitd_
    if (cex == nullptr) {
        if (!strata::gpu::begin_capture(cs_)) { err = "verify: begin deferred commit capture failed"; return false; }
        bool ok = true;
        try {
            for (int t = 0; t < S; ++t)
                if (t == 0 || rows[t] != rows[t - 1])
                    copy_i32_from_mapped(commitd_ + (size_t) rows[t] * CB, m_commitd_ + (size_t) rows[t] * CB, CB, cs_);
            int64_t qsa_index = 0, gdn_index = 0;
            layer_index(qsa_index, gdn_index);
            for (int64_t l = lb_; l < le_ && ok; ++l) {
                const LayerView v(*wt_, l);
                if (!is_qsa_layer(g, l)) {
                    const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                    if (!wnm) { ok = false; break; }
                    for (int t = 0; t < S;) {
                        const int first = t;
                        while (t < S && rows[t] == rows[first]) ++t;
                        SessionState& sx = *slots_[(size_t) rows[first]];
                        const int32_t* keep = commitd_ + (size_t) rows[first] * CB;
                        float* state = sx.gdn_state + (size_t) (gdn_index - sx.gdn_ord0) * gdn_floats;
                        float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        const int64_t row = gdn_index * R + hbase + first;
                        gdn_conv_commit(conv, dc_qkv_ + (size_t) row * C, (int) C, keep, cs_);
                        // z only feeds the norm's output, which the commit discards (y_dummy_): the window scratch will do
                        gdn_step_norm_multi(state, dc_h_ + (size_t) row * C, (int) C, dc_gate_ + (size_t) row * HV,
                                            dc_beta_ + (size_t) row * HV, z_ + (size_t) first * ZV,
                                            (const float*) wnm->data, EPS, y_dummy_ + (size_t) first * ZV,
                                            (int) g.ssm_k_heads, (int) HV, t - first, keep, cs_,
                                            t - first > 1 ? t - first : 0);
                    }
                    ++gdn_index;
                } else {
                    const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                    if (!wikn) { ok = false; break; }
                    for (int t = 0; t < S;) {
                        const int first = t;
                        while (t < S && rows[t] == rows[first]) ++t;
                        const QsaState& st = slots_[(size_t) rows[first]]->qsa_states[qsa_index];
                        copy_from_mapped(st.idx_tail, tail_snap_b_ + ((size_t) rows[first] * nQ + qsa_index) * TS, TS, cs_);
                        const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                        for (int u = first; u < t; ++u)
                            native_qsa_indexer_append(dc_idx_ + (size_t) (qsa_index * R + hbase + u) * ID,
                                                      commitd_ + (size_t) rows[first] * CB + 2 + (u - first), 0,
                                                      (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                      rope_scaling(), cs_);
                    }
                    ++qsa_index;
                }
            }
            if (ok && ple)
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    copy_indexed(slots_[(size_t) rows[first]]->ple_hist, dc_hist_ + (size_t) (hbase + first) * HS, HS,
                                 commitd_ + (size_t) rows[first] * CB + 1, HS, cs_);
                }
        } catch (const std::exception& e) {
            err = std::string("verify deferred commit: ") + e.what();
            ok = false;
        }
        if (!finish(cex, ok, "verify: deferred commit capture: ")) return false;
    }
    return true;
}

bool Verifier::batch_launch_rows(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos,
                                 std::string& err) {
    const OnDevice on_device(device_);
    if (dc_rows_ < 1) { err = "verify: batch_launch_rows without init_deferred_commit"; return false; }
    if (b_running_) { err = "verify: batch_launch_rows while this stage is busy"; return false; }
    if (S < 1 || hbase < 0 || hbase + S > dc_rows_) { err = "verify: a pipelined window outside the saved rows"; return false; }
    staging_deferred_ = true;
    const bool staged = stage_batch(rows, S, hbase, tokens, pos, err);
    staging_deferred_ = false;
    if (!staged || !capture_deferred(rows, S, hbase, err)) return false;
    const std::vector<int> key = batch_key(rows, S, hbase);
    // the save right behind the window, as batch_launch queues its commit: both before the host serves a layer
    if (!strata::gpu::graph_launch(exec_bm_[key], cs_) || !strata::gpu::graph_launch(save_bm_[key], cs_)) {
        err = std::string("verify: batch launch: ") + strata::gpu::last_error();
        return false;
    }
    (void) strata::gpu::stream_idle(cs_);
    b_running_ = true;
    b_k_ = 0;
    b_steps_ = le_ - lb_;
    b_last_ = Clock::now();
    return true;
}

bool Verifier::batch_commit_rows(const int* rows, int S, int hbase, const int* keep, const int32_t* tokens,
                                 const int64_t* pos, std::string& err) {
    const auto it = dcommit_bm_.find(batch_key(rows, S, hbase));
    if (it == dcommit_bm_.end() || it->second == nullptr) { err = "verify: a deferred commit without its window"; return false; }
    const int64_t CB = 2 + max_t_;
    // Each slot's accepted prefix.  The graph reads it from mapped memory when it runs; this slot's entry is written
    // again only at its next verdict, which comes after its next window has passed this stage - behind this commit.
    for (int t = 0; t < S;) {
        const int first = t;
        while (t < S && rows[t] == rows[first]) ++t;
        const int n = keep[rows[first]];
        if (n < 1 || n > t - first) { err = "verify: accepted prefix is outside its slot group"; return false; }
        int32_t* c = h_commitd_ + (size_t) rows[first] * CB;
        c[0] = n;
        c[1] = n - 1;
        for (int64_t j = 0; j < max_t_; ++j) c[2 + j] = j < n ? (int32_t) pos[first + j] : -1;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens), the accepted rows only
        for (int t = 0; t < S;) {
            const int first = t;
            while (t < S && rows[t] == rows[first]) ++t;
            SessionState& sx = *slots_[(size_t) rows[first]];
            for (int u = first; u < first + keep[rows[first]]; ++u) {
                sx.ple_prev[0] = sx.ple_prev[1];
                sx.ple_prev[1] = tokens[u];
            }
        }
    return true;
}

bool Verifier::batch_commit_rows_submit(const int* rows, int S, int hbase, std::string& err) {
    const OnDevice on_device(device_);
    if (b_running_) { err = "verify: a deferred commit while this stage runs a window"; return false; }
    const auto it = dcommit_bm_.find(batch_key(rows, S, hbase));
    if (it == dcommit_bm_.end() || it->second == nullptr) { err = "verify: a deferred commit without its window"; return false; }
    if (!strata::gpu::graph_launch(it->second, cs_)) {
        err = std::string("verify: deferred commit launch: ") + strata::gpu::last_error();
        return false;
    }
    (void) strata::gpu::stream_idle(cs_);
    return true;
}

int Verifier::batch_poll(PoolMultiFn pool, void* user, std::string& err) {
    if (!b_running_) return 1;
    const OnDevice on_device(device_);
    const int S = last_t_;
    while (b_k_ < b_steps_) {
        const uint32_t want = (uint32_t) (b_k_ + 1);
        if (win_.rung() < want) {
            const auto now = Clock::now();
            if (now - b_last_ > std::chrono::milliseconds(2)) {
                const strata::gpu::Status st = strata::gpu::stream_status(cs_);
                if (st == strata::gpu::Status::Failed) {
                    err = "verify batch: layer " + std::to_string(lb_ + b_k_) + ": " + strata::gpu::last_error();
                    b_running_ = false;
                    return -1;
                }
                if (st == strata::gpu::Status::Done && win_.rung() < want) {
                    err = "verify batch: layer " + std::to_string(lb_ + b_k_) + " never rang (graph finished)";
                    b_running_ = false;
                    return -1;
                }
                if (now - b_last_ > std::chrono::seconds(20)) {
                    err = "verify batch: timed out at layer " + std::to_string(lb_ + b_k_);
                    b_running_ = false;
                    return -1;
                }
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        win_.answer(want, 0, 1, 0, S, lb_ + b_k_, pool, user);
        progress_tick();
        ms_pool += ms_since(b);
        b_last_ = Clock::now();
        ++b_k_;
    }
    strata::gpu::Status st = strata::gpu::stream_status(cs_);
    if (st == strata::gpu::Status::Done) st = win_.copies_status();   // (the copies: no host function of this
    if (st == strata::gpu::Status::Failed) {                                         // window may raise flag B in the next)
        err = std::string("verify batch: the window's commit: ") + strata::gpu::last_error();
        b_running_ = false;
        return -1;
    }
    if (st == strata::gpu::Status::Pending) {
        if (Clock::now() - b_last_ > std::chrono::seconds(20)) {
            err = "verify batch: the window's commit did not finish";
            b_running_ = false;
            return -1;
        }
        return 0;
    }
    if (last_stage()) {
        if (!sample_rows(S, err)) { b_running_ = false; return -1; }
        for (int t = 0; t < S; ++t) b_out_[t] = ((volatile int32_t*) h_out_)[t];
        note_nonfinite(S, last_pos_b_);
    } else if (next_ != nullptr) {   // the next stage's rows of this group
        next_->rows(hand_out_, row_base_, S);
    }
    ++windows;
    b_running_ = false;
    progress_beat();
    return 1;
}

// A batch window's rows sampled outside the graph: a sampled slot's, a penalised one's (its reply tail and the
// tokens this window fed to it, `slot_penalty_rows`), one whose request has a logit bias of its own
// (set_slot_logit_bias), and with a grammar's masks set (set_allowed: a slot with a grammar, the engine's
// batch_step) every row, each with its own row of the masks - the graph's argmax knows no mask, and a slot without a
// grammar has a row that allows every token.  A plain greedy slot with none of these keeps the window's argmax
// token, untouched.  Every row is drawn as a solo window draws it: Philox(seed, position), its own parameters.
bool Verifier::sample_rows(int S, std::string& err) {
    const auto penalised = [](const strata::kernels::SamplerParams& sp) {
        return sp.penalty_last_n > 0 &&
               (sp.penalty_repeat != 1.0f || sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f);
    };
    bool any = false;
    const size_t words = (size_t) ((n_vocab_ + 31) / 32);
    // the penalty-history rows of this window, built once: the widest of its windows is every row's stride, and a
    // row counts only its own penalty_last_n of it
    int h = 0;
    for (int t = 0; t < S; ++t) {
        const strata::kernels::SamplerParams& sp = slot_sp_[(size_t) last_rows_[t]];
        if (penalised(sp)) h = std::max(h, std::min(sp.penalty_last_n, strata::kernels::kPenaltyWindowCap));
    }
    if (h > 0) {
        std::vector<const int32_t*> tails(slots_.size(), nullptr);
        std::vector<int64_t> n_tails(slots_.size(), 0);
        for (size_t b = 0; b < slots_.size() && b < slot_tail_.size(); ++b) {
            tails[b] = slot_tail_[b].data();
            n_tails[b] = (int64_t) slot_tail_[b].size();
        }
        std::vector<int32_t> hrows((size_t) S * (size_t) h, -1);
        strata::kernels::slot_penalty_rows(last_rows_, last_tokens_, S, tails.data(), n_tails.data(), h, hrows.data());
        const OnDevice on_hist(device_);
        const size_t need = (size_t) S * (size_t) h;
        if (slot_hist_cap_ < need) {
            if (slot_hist_d_ != nullptr) strata::gpu::free(slot_hist_d_);
            slot_hist_d_ = nullptr;
            if (!strata::gpu::alloc_device(&slot_hist_d_, need * sizeof(int32_t))) {
                err = "verify batch: the penalty history's allocation failed";
                return false;
            }
            slot_hist_cap_ = need;
        }
        if (!strata::gpu::copy(slot_hist_d_, hrows.data(), need * sizeof(int32_t))) {
            err = "verify batch: the penalty history's upload failed";
            return false;
        }
    }
    std::vector<strata::kernels::SamplerParams> params((size_t) S);
    for (int t = 0; t < S; ++t) {
        auto& sp = params[(size_t) t];
        sp = slot_sp_[(size_t) last_rows_[t]];
        const bool pen = h > 0 && penalised(sp);
        if (!allowed_on_ && !sp.logit_bias && !pen && (sp.greedy || sp.temperature <= 0.0f)) {
            sp.penalty_last_n = -1;  // keep the window's argmax
            continue;
        }
        sp.counter = (uint64_t) last_pos_b_[t];
        if (!pen) sp.penalty_last_n = 0;
        sp.allowed = allowed_on_ ? allowed_device_ + (size_t) t * words : nullptr;
        any = true;
    }
    if (any) {
        const OnDevice on_params(device_);
        if (slot_params_cap_ < (size_t) S) {
            if (slot_params_d_) strata::gpu::free(slot_params_d_);
            slot_params_d_ = nullptr;
            slot_params_cap_ = 0;
            if (!strata::gpu::alloc_device(&slot_params_d_, (size_t) S * sizeof(params[0]))) {
                err = "verify batch: the sampler parameters' allocation failed";
                return false;
            }
            slot_params_cap_ = (size_t) S;
        }
        if (!strata::gpu::copy(slot_params_d_, params.data(), (size_t) S * sizeof(params[0]))) {
            err = "verify batch: the sampler parameters' upload failed";
            return false;
        }
        strata::kernels::sample_token_rows(head_logits_, S, (int) n_vocab_, h ? slot_hist_d_ : nullptr, h,
                                           slot_params_d_, m_out_, cs_);
    }
    if (any && !strata::gpu::stream_sync(cs_)) { err = "verify batch: the row sampling failed"; return false; }
    return true;
}

// #879: STRATA_NAN_GUARD=0 turns the guard off (no kernel is recorded; the graphs are captured once, so it is read once)
bool Verifier::nan_guard_on() {
    static const bool on = [] {
        const char* v = strata::env_get("NAN_GUARD");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}

namespace {
// STRATA_NAN_INJECT=<n>[,<n>...] (debug): the windows (counted from 1 over the process) the guard reports as non-finite
const std::vector<long long>& nan_inject_list() {
    static const std::vector<long long> list = [] {
        std::vector<long long> v;
        if (const char* e = strata::env_get("NAN_INJECT"))
            for (const char* q = e; *q != 0;) {
                char* end = nullptr;
                const long long n = std::strtoll(q, &end, 10);
                if (end == q) break;
                if (n > 0) v.push_back(n);
                q = *end == ',' ? end + 1 : end;
            }
        return v;
    }();
    return list;
}
std::atomic<long long> g_nf_windows{0};
}  // namespace

void Verifier::note_nonfinite(int T, const int64_t* pos_b) {
    nf_row_ = -1;
    nf_pos_ = -1;
    nf_mask_ = 0;
    if (!nan_guard_on() || h_nf_ == nullptr || !last_stage()) return;   // (a stage without the head checks nothing)
    const volatile uint32_t* f = h_nf_;
    for (int t = 0; t < T && t < std::max(max_t_, (int) strata::kernels::kVerifyNarrowT); ++t)
        for (int b = 0; b < strata::kernels::kNonfiniteBlocks; ++b)
            if (f[t * strata::kernels::kNonfiniteBlocks + b] != 0) { nf_mask_ |= 1u << t; break; }
    const std::vector<long long>& inject = nan_inject_list();
    if (!inject.empty()) {
        const long long k = g_nf_windows.fetch_add(1) + 1;
        if (nf_mask_ == 0 && std::find(inject.begin(), inject.end(), k) != inject.end()) {
            nf_mask_ = 1u;
            std::fprintf(stderr, "xestrata verify: XESTRATA_NAN_INJECT: window %lld reported as non-finite\n", k);
        }
    }
    for (int t = 0; t < T && nf_row_ < 0; ++t)
        if ((nf_mask_ >> t) & 1u) nf_row_ = t;
    if (nf_row_ >= 0) nf_pos_ = pos_b != nullptr ? pos_b[nf_row_] : last_pos0_ + nf_row_;
}

bool Verifier::copy_logits(int t, float* host) const {
    if (next_ != nullptr) return next_->copy_logits(t, host);   // a layer split: the head is on the last stage
    if (head_logits_ == nullptr || host == nullptr || t < 0 || n_vocab_ <= 0) return false;
    const OnDevice on_device(device_);   // as read_logits
    return strata::gpu::copy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float));
}

bool Verifier::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                               std::string& err) {
    if (next_ != nullptr) return next_->window_logprobs(targets, T, pos0, extra_id, out, err);
    if (head_logits_ == nullptr || n_vocab_ <= 0 || T <= 0 || targets == nullptr || out == nullptr) {
        err = "window_logprobs: no head logits for this window";
        return false;
    }
    // run() synchronized its stream before returning, so the head of this window is complete
    std::vector<float> h((size_t) T * (size_t) n_vocab_);
    const OnDevice on_device(device_);   // as read_logits
    if (!strata::gpu::copy(h.data(), head_logits_, h.size() * sizeof(float))) {
        err = "window_logprobs: the head logits copy failed";
        return false;
    }
    // STRATA_FORCE_FAULT=<m>:<bias> (test only, off by default): adds <bias> to the logit of every token id divisible
    // by <m> before the rows are scored, so the token-forcing checker (tools/force_check.py) can be shown to reject a
    // faulty build
    struct Fault { long long every = 0; float bias = 0.0f; };
    static const Fault fault = [] {
        Fault f;
        if (const char* v = strata::env_get("FORCE_FAULT"); v != nullptr) {
            char* end = nullptr;
            const long long e = std::strtol(v, &end, 10);
            bool ok = false;
            if (end != v && *end == ':') {
                char* bend = nullptr;
                const float b = std::strtof(end + 1, &bend);
                if (bend != end + 1) { f.bias = b; ok = true; }
            }
            f.every = ok ? e : 0;
        }
        if (f.every < 0) f.every = 0;
        return f;
    }();
    if (fault.every > 0)
        for (int t = 0; t < T; ++t)
            for (int64_t v = 0; v < n_vocab_; v += fault.every)
                h[(size_t) t * (size_t) n_vocab_ + (size_t) v] += fault.bias;
    static const int topk = [] {
        const char* v = strata::env_get("LOGPOS_TOPK");
        return v != nullptr ? (int) std::clamp(std::strtol(v, nullptr, 10), 0L, 256L) : 0;
    }();
    for (int t = 0; t < T; ++t) {
        const float* row = h.data() + (size_t) t * (size_t) n_vocab_;
        const int32_t tgt = targets[t];
        if (tgt < 0 || (int64_t) tgt >= n_vocab_) continue;
        int64_t top = 0;
        for (int64_t v = 1; v < n_vocab_; ++v)
            if (row[v] > row[top]) top = v;
        const double maxv = row[top];
        const bool has_extra = extra_id >= 0 && (int64_t) extra_id < n_vocab_;
        double sum = 0.0, sum_without = 0.0;   // the second skips extra_id: no cancellation when it holds ~all mass
        for (int64_t v = 0; v < n_vocab_; ++v) {
            const double e = std::exp((double) row[v] - maxv);
            sum += e;
            if (v != (int64_t) extra_id) sum_without += e;
        }
        const double lse = maxv + std::log(std::max(sum, 1.0));   // sum holds the top token's exp(0) = 1
        const double extra = has_extra ? (double) row[extra_id] - lse : NAN;
        const double without = has_extra && tgt != extra_id && sum_without > 0.0
                                   ? (double) row[tgt] - (maxv + std::log(sum_without)) : NAN;
        std::fprintf(out, "%lld\t%d\t%.9f\t%lld\t%.9f\t%d\t%.9f\t%.9f", (long long) pos0 + t, (int) tgt,
                     (double) row[tgt] - lse, (long long) top, maxv - lse, (int) (top == (int64_t) tgt), extra,
                     without);
        if (topk > 0) {
            std::vector<int32_t> order((size_t) n_vocab_);
            for (int64_t v = 0; v < n_vocab_; ++v) order[(size_t) v] = (int32_t) v;
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                              [&](int32_t a, int32_t b) { return row[a] > row[b]; });
            for (int j = 0; j < topk; ++j)
                std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) row[order[(size_t) j]] - lse);
        }
        std::fprintf(out, "\n");
    }
    std::fflush(out);
    return true;
}

namespace { bool g_commit_async = false; }
void Verifier::set_commit_async(bool on) { g_commit_async = on && strata::env_get("COMMIT_SYNC") == nullptr; }

bool Verifier::commit(int n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    if (last_t_ == 1 && !last_batch_ && !always_publish_ && one_token_self_commit()) {
        // a one-token window has advanced the state itself (record_window): no commit graph
        if (ple_stage())
            for (int t = 0; t < n_keep; ++t) {
                ss_->ple_prev[0] = ss_->ple_prev[1];
                ss_->ple_prev[1] = last_tokens_[t];
            }
        ms_commit += ms_since(t0);
        return next_ == nullptr || next_->commit(n_keep, err);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const bool le = strata::gpu::graph_launch(commit_exec_, cs_);
    if (!le) { err = std::string("verify: commit launch: ") + strata::gpu::last_error(); return false; }
    // set_commit_async: no wait here - the next window runs on the same queue after it, and the drafter (its own
    // queue) reads only this window's final rows and its own K/V.  h_commit_ is next written after the next window's
    // results are read, i.e. after this graph has run.  Everything else waits on commit_done_ (wait_commit, which
    // goes on into a layer split's later stages).  Each stage's commit graph writes its own layers' state on its own
    // queue, so a layer split's stages commit without a wait as well (upstream kept the wait there).
    if (!g_commit_async) {
        if (!strata::gpu::stream_sync(cs_)) { err = std::string("verify: commit: ") + strata::gpu::last_error(); return false; }
    } else {
        if (!strata::gpu::event_record(commit_done_, cs_)) {
            err = std::string("verify: commit event: ") + strata::gpu::last_error();
            return false;
        }
        commit_pending_ = true;
    }
    if (ple_stage())   // stages that share one session must advance it once
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit(n_keep, err);
}

// ================================ PIPELINED WINDOWS (see verify.hpp) ================================
//
// The window is run()'s, step for step: the same graph, the same staging, the same per-layer service, the same commit
// graph.  Only the waits are split up: the host polls instead of spinning, so it can serve the other stage's window in
// between.  A one-token window does not commit itself here (always_publish_): a speculative window may be rolled back.

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
}  // namespace

void Verifier::diag_pipelined(std::FILE* f, const char* name) const {
    // host-side state only: a GPU call from the watchdog's thread can wait behind a window that waits for the host
    uint32_t at = 0, rang = 0, served = 0, fa = 0, fb = 0;
    win_.diag(at, rang, served, fa, fb);
    std::fprintf(f, "  %s: %s T=%d pos %lld served %lld/%lld; GPU rang %u, flags served %u A %u B %u; commit launched %d\n",
                 name, fl_active_ ? "IN FLIGHT" : "idle", last_t_, (long long) last_pos0_, (long long) fl_k_,
                 (long long) fl_total_, rang, served, fa, fb, (int) commit_live_);
}

bool Verifier::capture_all(std::string& err) {
    const OnDevice on_device(device_);
    if (g_ == nullptr) { err = "verify: capture_all before init"; return false; }
    if (win_.segmented()) { err = "verify: pipelined windows need a GPU whose running kernels see the host's writes"; return false; }
    if (!always_publish_) { err = "verify: pipelined windows need set_always_publish before init"; return false; }
    if (rt_dump_ || rt_replay_) { err = "XESTRATA_ROUTE_DUMP / XESTRATA_ROUTE_REPLAY: not with pipelined windows"; return false; }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture(T, err)) return false;
    if (!capture_commit(err)) return false;
    if ((ev_done_ == nullptr && !strata::gpu::event_create(&ev_done_)) ||
        (ev_commit_ == nullptr && !strata::gpu::event_create(&ev_commit_))) {
        err = "verify: event create failed";
        return false;
    }
    if (prof_on_ && prof_pin_ == nullptr && !strata::gpu::alloc_host(&prof_pin_, prof_h_.size() * 8))
        prof_pin_ = nullptr;   // the pipelined windows go unprofiled
    return true;
}

// The host staging of a pipelined window: run()'s, with the PLE rows from `ple_prev` (the window's verifier is idle,
// so its mapped rows are free to write)
bool Verifier::pl_stage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + (size_t) t * kStepCount, pos0 + t, s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) (pos0 + t);
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) (pos0 + t);
    }
    if (ss.ple.ready() && ple_stage()) {
        uint32_t rows[kVerifyMaxT * PLE_N_HEADS];
        int32_t prev[2] = {ple_prev[0], ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, rows + (size_t) t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
        }
        if (!ss.ple.table->gather_batch(rows, (size_t) T, h_ple_, err)) return false;
    }
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    pl_prev_[0] = ple_prev[0];
    pl_prev_[1] = ple_prev[1];
    return true;
}

bool Verifier::prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err) {
    if (fl_active_) { err = "verify: a window is in flight on this verifier"; return false; }
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    if (!pl_stage(T, tokens, pos0, ple_prev, err)) return false;
    pl_prestaged_ = true;
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string& err) {
    const OnDevice on_device(device_);
    if (fl_active_) { err = "verify: a window is already in flight on this verifier"; return false; }
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (exec_[T] == nullptr || commit_exec_ == nullptr || ev_done_ == nullptr) {
        err = "verify: pipelined window not prepared (capture_all)";
        return false;
    }
    const Clock::time_point t0 = Clock::now();
    last_batch_ = false;
    bool staged = pl_prestaged_ && last_t_ == T && last_pos0_ == pos0;
    for (int t = 0; staged && t < T; ++t) staged = last_tokens_[t] == tokens[t];
    if (staged && ss.ple.ready() && ple_stage())
        staged = pl_prev_[0] == ss.ple_prev[0] && pl_prev_[1] == ss.ple_prev[1];
    pl_prestaged_ = false;
    if (!staged && !pl_stage(T, tokens, pos0, ss.ple_prev, err)) return false;
    // the previous commit's words (h_commit_) are rewritten by this window's: wait for that graph here, before this
    // window is launched, never in pl_commit_async behind it (on the B70 that wait did not return while the window
    // waited for this thread)
    if (commit_live_) {
        strata::gpu::Status st;
        while ((st = strata::gpu::event_status(ev_commit_)) == strata::gpu::Status::Pending) _mm_pause();
        if (st == strata::gpu::Status::Failed) {
            err = std::string("verify: the previous commit: ") + strata::gpu::last_error();
            return false;
        }
        commit_live_ = false;
    }
    win_.reset();
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    fl_T_ = T;
    fl_k_ = 0;
    fl_total_ = (le_ - lb_) * G;
    fl_prof_ = prof_on_ && G == 1 && prof_pin_ != nullptr;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    ms_host += ms_since(t0);
    if (!strata::gpu::graph_launch(exec_[T], cs_)) {
        err = std::string("verify: launch: ") + strata::gpu::last_error();
        return false;
    }
    if (fl_prof_) strata::gpu::copy_async(prof_pin_, prof_, prof_h_.size() * 8, cs_);
    if (!strata::gpu::event_record(ev_done_, cs_)) { err = "verify: event record failed"; return false; }
    (void) strata::gpu::stream_idle(cs_);
    fl_active_ = true;
    fl_since_ms_ = fl_flush_ms_ = fl_launch_ms_ = now_ms();
    return true;
}

int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {
    if (!fl_active_ || fl_k_ >= fl_total_) return 1;
    // Nothing rang yet and neither the dead-graph check nor the timeout below is due: return before the device switch
    // (upstream 6b392641).  The pipelined loop asks both stages' windows this millions of times a second while they
    // wait on their doorbells; upstream's host stalled inside one of those device switches with both stages waiting
    // on it.  The same answer as the loop below gives.  (A pipelined window always rings: always_publish_ keeps the
    // zero-doorbell graph off, see init.)
    if (win_.rung() < (uint32_t) (fl_k_ + 1)) {
        const double now = now_ms();
        if (now - fl_flush_ms_ <= 2.0 && now - fl_since_ms_ <= 20000.0) return 0;
    }
    const OnDevice on_device(device_);
    const int T = fl_T_;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    while (fl_k_ < fl_total_) {
        const int64_t l = lb_ + fl_k_ / G;
        const uint32_t want = (uint32_t) (fl_k_ + 1);
        if (win_.rung() < want) {
            const double now = now_ms();
            if (now - fl_flush_ms_ > 2.0) {   // notice a dead graph, as run() does
                fl_flush_ms_ = now;
                const strata::gpu::Status st = strata::gpu::event_status(ev_done_);
                if (st == strata::gpu::Status::Failed) {
                    err = "verify: layer " + std::to_string(l) + ": " + strata::gpu::last_error();
                    return -1;
                }
                if (st == strata::gpu::Status::Done && win_.rung() < want) {
                    err = "verify: layer " + std::to_string(l) + " never rang (graph finished)";
                    return -1;
                }
            }
            if (now - fl_since_ms_ > 20000.0) {
                err = "verify: timed out at layer " + std::to_string(l);
                return -1;
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        ms_wait += now_ms() - fl_since_ms_;
        const int grp = (int) (fl_k_ % G);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window (pipelined): the CPU experts of layer", l);
        win_.answer(want, grp, G, tb, n, l, pool, user);
        progress_tick();
        ++fl_k_;
        ms_pool += ms_since(b);
        fl_since_ms_ = fl_flush_ms_ = now_ms();
    }
    return 1;
}

bool Verifier::done(std::string& err) {
    if (!fl_active_ || fl_k_ < fl_total_) return false;
    const OnDevice on_device(device_);
    strata::gpu::Status st = strata::gpu::event_status(ev_done_);
    if (st == strata::gpu::Status::Pending) {
        if (now_ms() - fl_since_ms_ > 20000.0) err = "verify: the window never finished";
        return false;
    }
    // no host function of this window may raise flag B in the next one
    if (st == strata::gpu::Status::Done) st = win_.copies_status();
    if (st == strata::gpu::Status::Failed) err = std::string("verify: the window: ") + strata::gpu::last_error();
    return st == strata::gpu::Status::Done;
}

bool Verifier::pl_finish(int32_t* out, std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    fl_active_ = false;
    if (fl_prof_) accumulate_profile(prof_pin_);
    ++windows;
    const int T = fl_T_;
    if (le_ < g.n_layers) {   // an earlier stage: the hand-off is written
        // the caller launches the next stage's window of this parity only after it has finished the one before,
        // which read its rows
        if (next_ != nullptr) next_->rows(hand_out_, 0, T);
        return true;
    }
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr || sampling_.logit_bias != nullptr ||
                           sampling_.allowed != nullptr)) {   // run()'s host-side sampling, Philox(seed, pos0 + t)
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) last_pos0_;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (!strata::gpu::stream_sync(cs_)) {
            err = "verify: the head sampling failed";
            return false;
        }
    }
    if (out != nullptr)
        for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(T, nullptr);
    progress_beat();
    return true;
}

bool Verifier::pl_commit_async(int n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    if (commit_exec_ == nullptr || ev_commit_ == nullptr) { err = "verify: pipelined commit not prepared"; return false; }
    const Clock::time_point t0 = Clock::now();
    if (commit_live_) {   // a replay (pl_launch waited for the one before): its graph reads the words below
        strata::gpu::Status st;
        while ((st = strata::gpu::event_status(ev_commit_)) == strata::gpu::Status::Pending) _mm_pause();
        if (st == strata::gpu::Status::Failed) {
            err = std::string("verify: the commit before: ") + strata::gpu::last_error();
            return false;
        }
    }
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!strata::gpu::graph_launch(commit_exec_, cs_)) {
        err = std::string("verify: commit launch: ") + strata::gpu::last_error();
        return false;
    }
    if (!strata::gpu::event_record(ev_commit_, cs_)) { err = "verify: event record failed"; return false; }
    commit_live_ = true;
    if (ple_stage())
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return true;
}

void Verifier::absorb_stats(Verifier& o) {
    ms_wait += o.ms_wait; ms_pool += o.ms_pool; ms_host += o.ms_host; ms_commit += o.ms_commit;
    windows += o.windows;
    o.ms_wait = o.ms_pool = o.ms_host = o.ms_commit = 0;
    o.windows = 0;
    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < kProfPer; ++i) { prof_sum_[k][i] += o.prof_sum_[k][i]; o.prof_sum_[k][i] = 0; }
    prof_windows_ += o.prof_windows_;
    o.prof_windows_ = 0;
}

// ================================ THE NEXT STAGE IN THIS PROCESS (StageLink) ================================

void Verifier::accept_rows(const float* from, int row0, int n) {
    float* in = const_cast<float*>(hand_in_);
    if (in == from) return;   // stages on one GPU share the hand-off
    const size_t HB = (size_t) handoff_floats(*g_);
    std::memcpy(in + (size_t) row0 * HB, from + (size_t) row0 * HB, (size_t) n * HB * sizeof(float));
}

void InProcessStage::rows(const float* hand_out, int row0, int n) { next_->accept_rows(hand_out, row0, n); }
bool InProcessStage::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, int32_t* out,
                         std::string& err) {
    return next_->run(T, tokens, pos0, pool, user_, out, err);
}
bool InProcessStage::run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos,
                                   PoolMultiFn pool, int32_t* out, std::string& err) {
    return next_->run_slot_rows(rows, S, tokens, pos, pool, user_, out, err);
}
bool InProcessStage::commit(int n_keep, std::string& err) { return next_->commit(n_keep, err); }
bool InProcessStage::commit_slot_prefixes(const int* keep, std::string& err) {
    return next_->commit_slot_prefixes(keep, err);
}
bool InProcessStage::wait_commit(std::string& err) { return next_->wait_commit(err); }
void InProcessStage::set_sampling(const strata::kernels::SamplerParams& sp) { next_->set_sampling(sp); }
void InProcessStage::set_slot_sampling(int slot, const strata::kernels::SamplerParams& sp) {
    next_->set_slot_sampling(slot, sp);
}

void InProcessStage::set_slot_history(int slot, const int32_t* tail, int64_t n) {
    next_->set_slot_history(slot, tail, n);
}

bool InProcessStage::set_slot_logit_bias(int slot, const std::vector<float>& bias, std::string& err) {
    return next_->set_slot_logit_bias(slot, bias, err);
}
void InProcessStage::set_head_sampling(bool on) { next_->set_head_sampling(on); }
bool InProcessStage::set_logit_bias(const std::vector<float>& bias, std::string& err) {
    return next_->set_logit_bias(bias, err);
}
bool InProcessStage::set_allowed(const uint32_t* rows, int T, std::string& err) {
    return next_->set_allowed(rows, T, err);
}
bool InProcessStage::set_route_prior(bool on, std::string& err) { return next_->set_route_prior(on, err); }
void InProcessStage::set_history(const int32_t* history, int history_len) { next_->set_history(history, history_len); }
bool InProcessStage::read_logits(int rows, float* host, std::string& err) const {
    return next_->read_logits(rows, host, err);
}
bool InProcessStage::copy_logits(int t, float* host) const { return next_->copy_logits(t, host); }
int InProcessStage::nonfinite_row() const { return next_->nonfinite_row(); }
int64_t InProcessStage::nonfinite_pos() const { return next_->nonfinite_pos(); }
bool InProcessStage::nonfinite_in(int t) const { return next_->nonfinite_in(t); }
bool InProcessStage::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                                     std::string& err) {
    return next_->window_logprobs(targets, T, pos0, extra_id, out, err);
}
const float* InProcessStage::final_R_all() const { return next_->final_R_all(); }

}  // namespace strata::core
