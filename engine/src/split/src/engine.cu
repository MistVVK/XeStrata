// The Engine (engine.hpp) for Qwen3.8-Flash-Next: its layers, its embeddings and its expert arena; what both models'
// engines share is engine_common.cu's (engine_parts.hpp).
#include <cstdlib>
#include "engine.hpp"
#include "engine_parts.hpp"
#include "env.hpp"
#include "peer_probe.hpp"
#include "pf_pick.hpp"

#include <cuda_runtime.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#if FNX_HIP
#include "hip/peer.h"
#include "hip/rccl.h"
#endif

namespace fnx {


Engine::Engine(Model& m, int max_ctx, int lanes) : m_(m), max_ctx_(max_ctx) {
    // the 27B build's settings: this Flash-Next engine has none of what they set, so one given is a mistake, not a
    // run of the default
    for (const char* k : {"FNX_PF_SKIP", "FNX_PF_SKIP_CONFIRM", "FNX_PF_SCORE_MIB", "FNX_PF_WINDOWS", "FNX_PF_SCORES",
                          "FNX_KV_PAGES", "FNX_KV_RESERVE_MIB"})
        if (std::getenv(k)) throw std::runtime_error(std::string(k) + " is the 27B's: Flash-Next has nothing it sets");
    // the rotary's scaling (choose_rope, src/rope.hpp): the model's own YaRN, FNX_ROPE_SCALE / FNX_YARN_ORIG_CTX, or
    // both when they agree; set once here, before any graph capture. Positions past what the rotary covers have angles
    // the model never saw, so a context past that is refused here rather than served
    rope_.base = m_.rope_base;
    long long env_orig = 0;
    if (std::getenv("FNX_YARN_ORIG_CTX")) {
        env_orig = env_int("FNX_YARN_ORIG_CTX", 0);
        if (env_orig <= 0 || env_orig > INT32_MAX)
            throw std::runtime_error("FNX_YARN_ORIG_CTX: want a trained context length from 1 to " + std::to_string(INT32_MAX) + ", got " +
                                     std::to_string(env_orig));
    }
    const RopeChoice rc = choose_rope(m_.yarn_factor, m_.yarn_orig, m_.ctx_train, std::getenv("FNX_ROPE_SCALE") ? rope_scale() : -1.0, env_orig);
    if (max_ctx > rc.limit)
        throw std::runtime_error("context " + std::to_string(max_ctx) + " is past the " + std::to_string(rc.limit) + " the rotary covers (" +
                                 (rc.S ? "trained " + std::to_string(rc.orig) + " x YaRN " + std::to_string(rc.S)
                                       : "the model's trained length; FNX_ROPE_SCALE=S (--rope-scale S) covers trained x S") +
                                 ")");
    if (rc.S) {
        rope_ = yarn_rope(m_.rope_base, rc.orig, rc.S, NROT);
        std::fprintf(stderr, "fnx: rope scaling YaRN x%.4f%s: n_ctx_orig %lld, pairs %.0f..%.0f of %d blended, mscale %.4f\n", rc.S,
                     rc.from_model ? " (the model's own)" : "", rc.orig, rope_.c_low, rope_.c_high, NROT / 2, rope_.mscale);
        if (!rc.from_model && max_ctx <= m_.ctx_train)
            std::fprintf(stderr, "fnx: note: context %d is within the trained %d: the scaling changes every position's rotary "
                         "and no length needs it\n", max_ctx, m_.ctx_train);
    }
    attn_i8_set_default(m_.g.iq_experts());   // the int8 prompt attention: on for the IQ3_S-type files, off for the others
    gdn_tc_set_default(!m_.g.iq_experts());   // the tensor-core GDN scan: off for the IQ3_S-type files (NVIDIA builds only compile it in)
    hc1_rows();   // FNX_HC1 checked now, not at the first hc read (inside a graph capture)
    const int extra_pages = (std::max(lanes, 1) - 1) * lane_block_pages();
    // the paged K/V pool: pages for the whole context (one more for the block a position past the last one needs)
    // and the extra lanes' blocks. Int8 rows: the context's pages as many as fit in their f16 bytes (the smallest ratio
    // of the cards': every card has the same pages)
    int ctx_pages = max_ctx / KVP_POS + 2;
    if (kv_q8()) {
        double ratio = 1e9;
        for (int cc = 0; cc < NCARD; ++cc) ratio = std::min(ratio, (double) kvp_page(cc, false) / kvp_page(cc, true));
        ctx_pages = (int) (ctx_pages * ratio);
    }
    int n_pages = ctx_pages + extra_pages;
    // prompt-path memory per card in MiB, measured (FNX_PF_TIMING "prefill buffers on card", the same for both models on 2 cards), and what must stay free beside the K/V pool: the decode buffers,
    // graphs and cuBLAS workspaces (UD-Q4_K_XL on 2 x 48 GB: 6.5 GB are free after the weights; 8192 x 2 fits but leaves 0.8 GB, 4096 x 4 leaves 2.3 GB)
    constexpr size_t PF_MARGIN_MIB = 2000;
    // the prompt path's chunk and pass. NVIDIA default 8192 rows and 4 chunks a pass (owner 11-03); a card whose free VRAM after the weights cannot hold that beside the K/V pool and
    // the rest (UD-Q4_K_XL on 2 x 48 GB keeps ~12 GB) gets the largest of 8192 x 4, 8192 x 2, 4096 x 4, 4096 x 2 that fits. FNX_PF_ROWS / FNX_PF_PASS set by hand are used as given.
    {
        long long rows = 4096;
        pf_pass_pick_ = 0;
#if !FNX_HIP
        rows = 8192;
        if (!std::getenv("FNX_PF_ROWS") && !std::getenv("FNX_PF_PASS")) {
            size_t free_min = SIZE_MAX, kv_max = 0, arena_max = 0;
            for (int c = 0; c < NCARD; ++c) {
                ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
                size_t fr = 0, tot = 0;
                ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
                // #5496: FNX_ARENA's pool is allocated after this pick (arena_fill); without the arena nothing is taken off (pf_pick.hpp)
                const size_t arena_c = pf_arena_unallocated(m.cards[c].ec, m.arena);
                free_min = std::min(free_min, pf_free_after_arena(fr, arena_c));
                arena_max = std::max(arena_max, arena_c);
                kv_max = std::max(kv_max, (size_t) n_pages * kvp_page(c) * sizeof(uint16_t));
            }
            const size_t margin = kv_max + (PF_MARGIN_MIB << 20);
            const PfPick pick = pf_pick(free_min, margin);
            rows = pick.rows;
            pf_pass_pick_ = pick.pass;
            if ((pick.rows != 8192 || pick.pass != 4) && arena_max == 0)
                std::fprintf(stderr, "fnx: prompt path: %d rows, %d chunks a pass (the cards hold %zu MiB free after the weights, the K/V pool %zu MiB)\n", pick.rows, pick.pass,
                             free_min >> 20, kv_max >> 20);
            else if (pick.rows != 8192 || pick.pass != 4)
                std::fprintf(stderr, "fnx: prompt path: %d rows, %d chunks a pass (the cards hold %zu MiB free after the weights and the arena's slots, the K/V pool %zu MiB, the arena's slots up to %zu MiB)\n",
                             pick.rows, pick.pass, free_min >> 20, kv_max >> 20, arena_max >> 20);
        }
#endif
        const long long pf_rows = env_int("FNX_PF_ROWS", rows);
        if (pf_rows < MAXT || pf_rows > 8192 || pf_rows % MAXT)
            throw std::runtime_error("FNX_PF_ROWS: a multiple of " + std::to_string(MAXT) + " from " + std::to_string(MAXT) + " to 8192, got " + std::to_string(pf_rows));
        pf_c_ = (int) pf_rows;
    }
    pf_jobs_ = pf_c_ * TOPK / PF_JOB + NEXP;   // jobs of up to PF_JOB pairs per expert, at most
    set_prompt_rows(pf_c_);
    // FNX_PEER (AMD only; hip/peer_policy.hpp has the rules: 0 host exchange, 1 try the peer push, unset the build's default): every directed pair is probed at start
    // (kernel stores into the peer, a full read-back compare on both sides, a flag round trip); any failure leaves the host exchange for the whole run with one line.
    // FNX_PEER_PARTS: 1 the window sums, 2 the attention merge, 3 both (the default)
    bool peer_on = false;
    int peer_parts = 0;
#if FNX_HIP
    {
        std::vector<int> devs;
        for (int c = 0; c < NCARD; ++c) devs.push_back(m.cards[c].dev);
        const PeerDecision pd = peer_decide(std::getenv("FNX_PEER"), PEER_DEFAULT_ON, [&](std::string& why) { return peer_enable_and_probe(devs, why); });
        if (!pd.line.empty()) std::fprintf(stderr, "fnx: %s\n", pd.line.c_str());
        peer_on = pd.use;
        peer_parts = (int) env_int("FNX_PEER_PARTS", 3);
        if (peer_on) std::fprintf(stderr, "fnx: peer push: %s%s pushed into the peers' memory\n", peer_parts & 1 ? "the window sums " : "", peer_parts & 2 ? "and the attention merge" : "");
        {   // RCCL for the prompt sums: constant RCCL_DEFAULT_ON (off until the screens say otherwise), FNX_PF_RCCL=0/1 overrides
            const RcclDecision rd = rccl_decide(std::getenv("FNX_PF_RCCL"), peer_on, RCCL_DEFAULT_ON, [&](std::string& why) {
                std::shared_ptr<RcclState> st;
                if (!rccl_open(devs, st, why)) return false;
                rccl_ = st;
                return true;
            });
            if (!rd.line.empty()) std::fprintf(stderr, "fnx: %s\n", rd.line.c_str());
            rccl_on_ = rd.use;
        }
        // the prompt chunk sums through the peers' device memory (needs the peer push; RCCL, when it is in use, does the sums instead): FNX_PF_PEER=0 keeps the host exchange
        pf_peer_ = peer_on && !rccl_on_ && env_int("FNX_PF_PEER", 1) != 0;
        if (pf_peer_) std::fprintf(stderr, "fnx: peer push: the prompt chunk sums are pushed into the peers' device memory\n");
    }
    auto pxalloc = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (hipExtMallocWithFlags(&p, bytes, hipDeviceMallocFinegrained) != hipSuccess) throw std::runtime_error("FNX_PEER: fine-grained device allocation of " + std::to_string(bytes >> 10) + " KiB failed");
        ck(FNX_ZERO_DEV(p, bytes), "peer mailbox clear");
        own_dev(p);
        return p;
    };
#endif
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m.cards[c];
        CardBuf* b = new CardBuf();
        b->dev = w.dev;
        ck(cudaSetDevice(w.dev), "cudaSetDevice");
        // FNX_HOSTRING: no peer access is ever enabled; the cards exchange through host memory
        ck(cudaStreamCreateWithFlags(&b->st, cudaStreamNonBlocking), "stream");
        own_stream(b->st);
        ck(cudaStreamCreateWithFlags(&b->st2, cudaStreamNonBlocking), "stream");
        own_stream(b->st2);
        ck(cudaEventCreateWithFlags(&b->fork, cudaEventDisableTiming), "event");
        own_event(b->fork);
        ck(cudaEventCreateWithFlags(&b->join, cudaEventDisableTiming), "event");
        own_event(b->join);
        ck(cudaEventCreate(&b->ev0), "event");
        own_event(b->ev0);
        ck(cudaEventCreate(&b->ev1), "event");
        own_event(b->ev1);
        const int P = MAXT * TOPK, ffs = w.ff1 - w.ff0;
        b->R = dalloc<float>((size_t) MAXT * D);
        b->mixed = dalloc<float>((size_t) MAXT * N);
        b->bo_a = dalloc<float>((size_t) MAXT * N);
        b->bo_f = dalloc<float>((size_t) MAXT * N);
        b->inj_a = dalloc<float>(MAXT * HC);
        b->inj_f = dalloc<float>(MAXT * HC);
        b->hcs = dalloc<float>(hc_scratch_floats(MAXT));
        if (const int g = hc_prepare(b->hcs); g < hc_blocks())
            std::fprintf(stderr, "fnx: card %d: %d SMs: the hc read runs its %d blocks of work on %d\n", c, g, hc_blocks(), g);
        if (std::getenv("FNX_HC_BAR_WRAP")) {
            const long long n = env_int("FNX_HC_BAR_WRAP", 0);
            if (n <= 0 || n > 0xffffffffLL) throw std::runtime_error("FNX_HC_BAR_WRAP: want a barrier count > 0, got " + std::to_string(n));
            hc_preset_barrier(b->hcs, (unsigned) n);
        }
        b->xq = dalloc<uint8_t>((size_t) P * q8_1_bytes(N));
        b->qkvz = dalloc<float>((size_t) MAXT * (w.conv_ch() + w.nv() * HS));
        b->ab = dalloc<float>((size_t) MAXT * 2 * w.nv());
        b->gdn_scratch = dalloc<float>(gdn_scratch_floats(w.nv()));
        b->qg = dalloc<float>((size_t) MAXT * NQH * 2 * HD);
        b->k = dalloc<float>((size_t) MAXT * NKVH * HD);
        b->v = dalloc<float>((size_t) MAXT * NKVH * HD);
        b->att = dalloc<float>((size_t) MAXT * w.nq * HD);
        b->att_part = dalloc<float>(attn_scratch_floats(w.nq));
        b->logits = dalloc<float>((size_t) MAXT * (NEXP + 1));
        b->topw = dalloc<float>(P);
        b->topi = dalloc<int>(P);
        b->njobs = dalloc<int>(2);
        b->jobs_gu = dalloc<Job>(P + 1);
        b->jobs_dn = dalloc<Job>(P + 1);
        for (int T = 1; T <= MAXT; ++T) {
            Job j{};
            j.w = 0;
            j.n = T;
            for (int t = 0; t < T; ++t) { j.y[t] = t; j.out[t] = t; }
            b->dense[T] = dalloc<Job>(1);
            ck(cudaMemcpy(b->dense[T], &j, sizeof j, cudaMemcpyHostToDevice), "dense job");
        }
        if (w.ec.n_slots) b->cache_list = dalloc<int>(1 + P);
        b->gu = dalloc<float>((size_t) (P + MAXT) * 2 * ffs);
        b->part = dalloc<float>((size_t) (P + MAXT) * N);
        b->sh_gu = dalloc<float>((size_t) MAXT * 2 * ffs);
        b->sh_hq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(ffs));
        if (!std::getenv("FNX_NO_TWIN")) {
            // the split twins: xq's rows of n <= N columns sit at r q8_1_bytes(n), so r n < P N
            q8_twin(b->xq, (size_t) P * q8_1_bytes(N), dalloc<int8_t>((size_t) P * N), dalloc<float>((size_t) P * N / 32));
            q8_twin(b->sh_hq, (size_t) MAXT * q8_1_bytes(ffs), dalloc<int8_t>((size_t) MAXT * ffs), dalloc<float>((size_t) MAXT * ffs / 32));
            for (const void* q : {(const void*) b->xq, (const void*) b->sh_hq}) own([q] { q8_untwin(q); });
        }
        b->ple_emb = dalloc<float>((size_t) MAXT * N);
        b->ple_key = dalloc<float>((size_t) 3 * MAXT * D);   // key, then room for gated and normalized rows
        b->ple_val = dalloc<float>((size_t) MAXT * N);
        b->head_logits = dalloc<float>((size_t) MAXT * (w.v1 - w.v0));
        b->best_v = dalloc<float>(MAXT);
        b->best_i = dalloc<int>(MAXT);
        b->am_keys = dalloc<unsigned long long>(MAXT);
        if (c == 0 && env_flag("FNX_PREFETCH_PROBE")) {
            b->probe_logits = dalloc<float>((size_t) MAXT * NEXP);
            b->probe_guess = dalloc<int>((size_t) NL * MAXT * TOPK);
            b->probe_cnt = dalloc<unsigned long long>(7);
        }
        if (c == 0 && env_flag("FNX_ROUTE_COUNT")) {
            b->route_cnt = dalloc<unsigned>((size_t) 2 * NL * NEXP);
            ck(cudaMemset(b->route_cnt, 0, (size_t) 2 * NL * NEXP * sizeof(unsigned)), "route counts");
        }
        if (const long long tr = env_int("FNX_ROUTE_TRACE", 0); c == 0 && tr > 0) {
            b->route_trace_cap = (unsigned long long) tr * 1000000ull;
            void* hp = nullptr;   // in mapped host memory: no VRAM for it
            ck(cudaHostAlloc(&hp, b->route_trace_cap * sizeof(unsigned), cudaHostAllocMapped), "route trace");
            own_pinned(hp);
            b->route_trace = (unsigned*) hp;
            b->route_trace_len = dalloc<unsigned long long>(1);
            ck(cudaMemset(b->route_trace_len, 0, sizeof(unsigned long long)), "route trace");
        }
        b->S.assign(NL, nullptr);
        b->chist.assign(NL, nullptr);
        b->snap.assign(NL, nullptr);
        b->hsnap.assign(NL, nullptr);
        b->kc.assign(NL, nullptr);
        b->vc.assign(NL, nullptr);
        b->qsa_ring.assign(NL, nullptr);
        b->qsa_pooled.assign(NL, nullptr);
        b->qsa_y = dalloc<float>((size_t) MAXT * qsa_rows());
        b->qsa_q = dalloc<float>((size_t) MAXT * 512);
        b->qsa_scores = dalloc<float>((size_t) MAXT * (max_ctx / 4));
        b->qsa_cells = dalloc<int>((size_t) MAXT * qsa_maxsel());
        b->qsa_ncells = dalloc<int>(MAXT);
        const size_t snap_floats = (size_t) (MAXT - 1) * w.nv() * HS * HS;
        int n_rec = 0;
        for (int l = 0; l < NL; ++l) n_rec += is_attn(l) ? 0 : 1;
        b->snap_block = dalloc<float>(snap_floats * n_rec);
        b->snap_block_bytes = snap_floats * n_rec * sizeof(float);
        for (int l = 0, k = 0; l < NL; ++l) {
            if (is_attn(l)) {
                b->qsa_ring[l] = dalloc<uint16_t>((size_t) qsa_ring_rows() * 128);
            } else {
                b->S[l] = dalloc<float>((size_t) w.nv() * HS * HS);
                b->chist[l] = dalloc<float>((size_t) 2 * 3 * w.conv_ch());
                b->snap[l] = b->snap_block + snap_floats * k++;
                b->hsnap[l] = dalloc<float>((size_t) MAXT * 3 * w.conv_ch());
            }
        }
        b->ple_hist = dalloc<float>((size_t) 9 * D);
        b->ple_hsnap = dalloc<float>((size_t) MAXT * 9 * D);
        b->ctl = dalloc<int>(2);
        b->ring.card = c;
        // sized for fp32 words (two per pair); fp16 pairs use the first half of each slot row's room
        b->ring.in = hxalloc<unsigned long long>(ring_in_words(MAXT));
        b->ring.ag = hxalloc<unsigned long long>(ring_ag_words(MAXT));
        b->ring.f32 = ring_f32();
        b->ring.err = dalloc<unsigned>(1);
        b->att_ring.card = c;
#if FNX_HIP
        if (peer_on && (peer_parts & 1)) b->ring.pk = pxalloc((size_t) 2 * NCARD * MAXT * (N / 2) * sizeof(unsigned long long));
        b->att_ring.in = peer_on && (peer_parts & 2) ? (unsigned long long*) pxalloc(att_ring_words() * sizeof(unsigned long long))
                                                     : hxalloc<unsigned long long>(att_ring_words());
#else
        b->att_ring.in = hxalloc<unsigned long long>(att_ring_words());
#endif
        b->att_ring.err = b->ring.err;
        cb_.push_back(b);
    }
    // every link carries a probe's stores before the ring's (src/peer_probe.hpp): one that drops them stops the engine
    // here, named, instead of in the first window's ring; a slow one is said and the engine goes on
    std::fprintf(stderr, "fnx: HOSTRING build: the cards exchange through page-locked host memory; peer access is never enabled, "
                         "no peer probe, IOMMU and BAR1 checks skipped\n");
    for (int c = 0; c < NCARD; ++c) {
        cb_[c]->ring.next_in = cb_[(c + 1) % NCARD]->ring.in;
        cb_[c]->ring.next_ag = cb_[(c + 1) % NCARD]->ring.ag;
        cb_[c]->att_ring.next_in = cb_[(c + 1) % NCARD]->att_ring.in;
        for (int d = 0; d < NCARD; ++d) {
            cb_[c]->ring.peer_in[d] = cb_[d]->ring.in;
            cb_[c]->att_ring.peer_in[d] = cb_[d]->att_ring.in;
        }
        // one hop for the window sums and the attention merge on 3 or more cards (FNX_ONEHOP=0: the rings); the mailboxes live in the ring's buffers
        // FNX_ONEHOP=2 (screen): the one-hop sum on two cards too (each card adds the two fp16-rounded addends in fp32: one host round trip instead of two; the sums differ in the last bits)
        const bool onehop = (NCARD > 2 || peer_on || env_int("FNX_ONEHOP", 1) == 2) && env_int("FNX_ONEHOP", 1) != 0 && !cb_[c]->ring.f32;
        cb_[c]->ring.onehop = cb_[c]->att_ring.onehop = onehop;
        for (int d = 0; d < NCARD; ++d) cb_[c]->ring.peer_pk[d] = cb_[d]->ring.pk;
        cb_[c]->ring.peer = onehop && cb_[c]->ring.pk != nullptr;
    }
    if (NCARD > 2 && env_int("FNX_ONEHOP", 1) != 0 && cb_[0]->ring.f32)
        std::fprintf(stderr, "fnx: FNX_RING_F32: the one-hop sum carries fp16 words only, the rings stay\n");
    // the prompt path's buffers now, so the arena below is sized from what is left after them
    prefill_alloc();
    // FNX_ARENA: the pool's own pages are those, or as many as every card's free VRAM holds beside the arena pool's
    // slots and FNX_ARENA_RESERVE_MIB, or FNX_ARENA_BASE_PAGES; the slots follow the own pages in the same allocation,
    // and the pool grows into them (lent coldest first) up to where every card keeps 2 MAXT TOPK slots. The buffers
    // above and the draft layer (an Mtp is made before its Engine) are already allocated, so the reserve is only what
    // comes later: the window graphs captured at first use (with batched drafts a verify graph per row total), cuBLAS's
    // workspaces and the lanes' page tables. Default 512: they took ~270 MiB a card on three cards and ~340 on two,
    // through one to six lanes, batched drafts and a 114k prompt
    if (m.arena) {
        const long long reserve_mib = env_int("FNX_ARENA_RESERVE_MIB", 512);
        if (reserve_mib < 0) throw std::runtime_error("FNX_ARENA_RESERVE_MIB: want MiB >= 0, got " + std::to_string(reserve_mib));
        const size_t reserve = (size_t) reserve_mib << 20;
        // a card with less free VRAM than the slots, the reserve and the smallest own pool (two cards, another process
        // on it, a leak): the coldest experts that many slots' worth go to RAM, placed again, and the engine goes on.
        // Too few slots left for a window is still an error (Model::place_experts says so)
        for (int round = 0;; ++round) {
            int more = 0;
            size_t short_mib = 0;
            for (int c = 0; c < NCARD; ++c) {
                const ExpertCache& ec = m.cards[c].ec;
                if (ec.arena_pool < 0) throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has no arena pool");
                ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
                size_t fr = 0, tot = 0;
                ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
                const size_t sb = ec.slot_bytes[ec.arena_pool],
                             need = (size_t) ec.n[ec.arena_pool] * sb + reserve + (size_t) (extra_pages + 1) * kvp_page(c) * sizeof(uint16_t) +
                                    2 * pf_stage_bytes(c);
                if (fr < need) {   // 256 MiB more than short, so the next engine's few MiB of drift place nothing again
                    more = std::max(more, (int) ((need - fr + (256u << 20) + sb - 1) / sb));
                    short_mib = std::max(short_mib, (need - fr) >> 20);
                }
            }
            if (!more) break;
            if (round == 8) throw std::runtime_error("FNX_ARENA: VRAM still short after placing the experts again 8 times");
            // two 32 GB cards are short by design; on three or more the experts fit unless something else holds VRAM
            std::fprintf(stderr, "fnx: VRAM short by %zu MiB on a card%s: %d more of the coldest experts to RAM\n", short_mib,
                         NCARD >= 3 ? " (another process on it?)" : "", more);
            m.free_experts();
            m.ram_extra += more;
            m.place_experts();
        }
        prefill_stages();
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            if (ec.arena_pool < 0) throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has no arena pool");
            ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
            size_t fr = 0, tot = 0;
            ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
            const size_t slots = (size_t) ec.n[ec.arena_pool] * ec.slot_bytes[ec.arena_pool], page = kvp_page(c) * sizeof(uint16_t);
            const int fit = fr > slots + reserve ? (int) ((fr - slots - reserve) / page) : 0;
            if (fit <= extra_pages)
                throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has " + std::to_string(fr >> 20) + " MiB free: the arena's " +
                                         std::to_string(slots >> 20) + " MiB of slots, " + std::to_string(reserve >> 20) +
                                         " MiB in reserve (FNX_ARENA_RESERVE_MIB) and the lanes' blocks do not fit");
            n_pages = std::min(n_pages, fit);
        }
        if (std::getenv("FNX_ARENA_BASE_PAGES")) {
            const long long want = env_int("FNX_ARENA_BASE_PAGES", 0);
            if (want <= extra_pages || want > n_pages)
                throw std::runtime_error("FNX_ARENA_BASE_PAGES: want " + std::to_string(extra_pages + 1) + " to " + std::to_string(n_pages) +
                                         " (the lanes' blocks take " + std::to_string(extra_pages) + ")");
            n_pages = (int) want;
        }
        arena_base_ = arena_w_ = n_pages;
        arena_cap_ = INT32_MAX;
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool, keep = EXPERT_WINDOW_SLOTS;
            const size_t page = kvp_page(c) * sizeof(uint16_t);
            const size_t room = (size_t) (ec.n[p] - keep) * ec.slot_bytes[p] + (arena_slot_off(c) - (size_t) n_pages * page);
            arena_cap_ = std::min(arena_cap_, n_pages + (int) (room / page));
        }
    } else {
        prefill_stages();
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf* b = cb_[c];
        ck(cudaSetDevice(b->dev), "cudaSetDevice");
        // the K/V pool and lane 0's page table (pages taken as it grows); with an arena the pool is its start and
        // n_pages counts the pages it can grow to
        if (m.arena) {
            ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool;
            b->n_pages = arena_cap_;
            char* region = (char*) dalloc<uint8_t>(arena_slot_off(c) + (size_t) ec.n[p] * ec.slot_bytes[p]);
            b->pool = (uint16_t*) region;
            ec.pool[p] = region + arena_slot_off(c);
            m.cards[c].bytes += (size_t) ec.n[p] * ec.slot_bytes[p];
            arena_fill(c, b->st);
            size_t fr = 0, tot = 0;
            ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
            std::fprintf(stderr, "fnx: arena: card %d: %zu MiB free after its arena\n", c, fr >> 20);
        } else {
            b->n_pages = n_pages;
            b->pool = dalloc<uint16_t>((size_t) b->n_pages * kvp_page(c));
        }
        b->pt = dalloc<int>((size_t) b->n_pages);
        b->pg.pt = b->pt;
        b->pg.page = kvp_page(c);
        b->pg.rows = kvp_rows(c);
        b->pg.q8 = kv_q8();
        const size_t kv_halves = kvp_rows_halves(c, kv_q8());
        for (int l = 0, slot = 0; l < NL; ++l)
            if (is_attn(l)) {
                b->kc[l] = b->pool + (size_t) slot * kvp_slot(c);
                b->vc[l] = b->kc[l] + kv_halves;
                b->qsa_pooled[l] = b->kc[l] + 2 * kv_halves;
                ++slot;
            }
        {   // lane 0: the buffers above
            LaneDev L;
            L.pt = b->pt;
            L.S = b->S;
            L.chist = b->chist;
            L.ple_hist = b->ple_hist;
            L.ring = b->qsa_ring;
            L.ctl = dalloc<int>(2);
            b->lanes.push_back(L);
        }
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "init sync"); }
    lanes_.emplace_back();
    for (int i = m.arena ? arena_base_ : cb_[0]->n_pages; i-- > 0;) free_pages_.push_back(i);   // taken from the back: low pages first
    if (m.arena) {
        std::string per;
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool;
            char buf[96];
            std::snprintf(buf, sizeof buf, "%s%d x %.2f MiB (%.1f GiB)", c ? ", " : "", ec.n[p], ec.slot_bytes[p] / 1048576.0,
                          (double) ec.n[p] * ec.slot_bytes[p] / (1ull << 30));
            per += buf;
        }
        std::fprintf(stderr, "fnx: arena: the K/V pool's own %d pages (%d positions), up to %d (%d positions) by lending the slots of %s\n",
                     arena_base_, arena_base_ * KVP_POS, arena_cap_, arena_cap_ * KVP_POS, per.c_str());
    }
    if (std::getenv("FNX_RING_TAG_START")) {
        const long long v = env_int("FNX_RING_TAG_START", 0);
        if (v < 1 || v >= (long long) TAG_END)
            throw std::runtime_error("FNX_RING_TAG_START: want 1.." + std::to_string(TAG_END - 1) + ", got " + std::to_string(v));
        tag_max_ = (unsigned) v;
    }
}

void Engine::clear_rings() {
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaMemset(b.ring.in, 0, ring_in_words(MAXT) * sizeof(unsigned long long)), "ring clear");
        ck(cudaMemset(b.ring.ag, 0, ring_ag_words(MAXT) * sizeof(unsigned long long)), "ring clear");
        ck(cudaMemset(b.att_ring.in, 0, att_ring_words() * sizeof(unsigned long long)), "ring clear");
        if (!pf_.empty()) pf_clear_ring(c);
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
    std::fprintf(stderr, "fnx: ring tags reached %u: the rings' buffers cleared, tags start again\n", tag_max_);
    tag_max_ = 1;
}

void Engine::tags_room(unsigned n) {
    if (tag_max_ + n + 1u >= TAG_END) clear_rings();
}

// an arena's K/V pool grows by at least this many pages a lend (64: ~12k positions f16), so lends stay rare
constexpr int ARENA_STEP = 64;

size_t Engine::arena_slot_off(int c) const {
    return ((size_t) arena_base_ * kvp_page(c) * sizeof(uint16_t) + 255) & ~(size_t) 255;
}

// the arena pool as loaded: the slot tables back to the model's first assignment (an earlier engine in this process
// lent, reclaimed and replaced slots), every expert of the pool's layers at its host copy, then each slot's expert
// copied in from it
void Engine::arena_fill(int c, cudaStream_t st) {
    ExpertCache& ec = m_.cards[c].ec;
    const CardW& w = m_.cards[c];
    const int p = ec.arena_pool;
    // only the arena pool's part: the other pools' slots keep what an earlier engine copied into them, and their
    // tables say so
    const int s0 = ec.s0[p], n = ec.n[p];
    ck(cudaMemcpy(ec.owner + s0, w.ec_owner0.data() + s0, (size_t) n * sizeof(int), cudaMemcpyHostToDevice), "arena: owners");
    std::vector<int> slot_of((size_t) NL * NEXP);
    ck(cudaMemcpy(slot_of.data(), ec.slot_of, slot_of.size() * sizeof(int), cudaMemcpyDeviceToHost), "arena: slots");
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e) slot_of[(size_t) l * NEXP + e] = w.ec_slot0[(size_t) l * NEXP + e];
    ck(cudaMemcpy(ec.slot_of, slot_of.data(), slot_of.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: slots");
    ck(cudaMemset(ec.ref + s0, 0, (size_t) n), "arena: refs");
    ck(cudaMemcpy(ec.hand + p, &ec.s0[p], sizeof(int), cudaMemcpyHostToDevice), "arena: hand");
    ck(cudaMemset(ec.lent + p, 0, sizeof(int)), "arena: lent");
    const std::vector<int> owner(w.ec_owner0.begin() + s0, w.ec_owner0.begin() + s0 + n);
    std::vector<const char*> epg((size_t) NL * NEXP), epd(epg.size());
    ck(cudaMemcpy(epg.data(), ec.ep_gu, epg.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    ck(cudaMemcpy(epd.data(), ec.ep_dn, epd.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e) {
                const size_t i = (size_t) l * NEXP + e;
                epg[i] = w.ec_host_gu[i];
                epd[i] = w.ec_host_dn[i];
            }
    // the slots' experts in one kernel reading their host copies (a copy call each ran the cards one after another)
    std::vector<int> list{n};
    for (int s = 0; s < ec.n[p]; ++s) {
        const int i = owner[(size_t) s];
        char* at = ec.pool[p] + (size_t) s * ec.slot_bytes[p];
        epg[(size_t) i] = at;
        epd[(size_t) i] = at + ec.dn_off[p];
        list.push_back(i);
    }
    int* dlist = dalloc<int>(list.size());
    ck(cudaMemcpy(dlist, list.data(), list.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: fill list");
    cache_copy(ec, p, dlist, st);
    ck(cudaMemcpyAsync(ec.ep_gu, epg.data(), epg.size() * sizeof(char*), cudaMemcpyHostToDevice, st), "arena: tables");
    // no wait here: the cards fill at once, and the constructor syncs every card before the engine is used (the tables
    // come from pageable vectors, copied out before cudaMemcpyAsync returns)
    ck(cudaMemcpyAsync(ec.ep_dn, epd.data(), epd.size() * sizeof(char*), cudaMemcpyHostToDevice, st), "arena: tables");
}

int Engine::arena_slots_for(int c, int w) const {
    const ExpertCache& ec = m_.cards[c].ec;
    const long need = (long) w * (long) (kvp_page(c) * sizeof(uint16_t)) - (long) arena_slot_off(c);
    if (need <= 0) return 0;
    const long sb = (long) ec.slot_bytes[ec.arena_pool];
    return (int) ((need + sb - 1) / sb);
}

std::vector<int> Engine::arena_lent() const {
    std::vector<int> v;
    if (arena_cap_)
        for (int c = 0; c < NCARD; ++c) v.push_back(arena_slots_for(c, arena_w_));
    return v;
}

// slots [to, from) of card c's arena pool, just given back empty: each takes one of the pool's experts not in a slot,
// the hottest first (the profile's score), copied in from its host copy (the card idle). Nothing lent then means every
// expert is in a slot, which the plan's early return relies on (cache_plan, moe.cuh)
void Engine::arena_refill(int c, int from, int to) {
    ExpertCache& ec = m_.cards[c].ec;
    const CardW& w = m_.cards[c];
    const int p = ec.arena_pool, s0 = ec.s0[p];
    std::vector<int> slot_of((size_t) NL * NEXP);
    ck(cudaMemcpy(slot_of.data(), ec.slot_of, slot_of.size() * sizeof(int), cudaMemcpyDeviceToHost), "arena: slots");
    std::vector<int> out;   // the pool's experts not in a slot
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e)
                if (slot_of[(size_t) l * NEXP + e] < 0) out.push_back(l * NEXP + e);
    std::stable_sort(out.begin(), out.end(), [&](int a, int b) { return m_.score[(size_t) a] > m_.score[(size_t) b]; });
    if (ec.arena_full && to == 0 && (int) out.size() > from)
        throw std::runtime_error("arena: " + std::to_string(out.size()) + " experts out of a slot, " + std::to_string(from) + " slots to take them");
    const int n = std::min((int) out.size(), from - to);
    std::vector<int> owner((size_t) (from - to), -1);
    std::vector<const char*> epg((size_t) NL * NEXP), epd(epg.size());
    ck(cudaMemcpy(epg.data(), ec.ep_gu, epg.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    ck(cudaMemcpy(epd.data(), ec.ep_dn, epd.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    for (int k = 0; k < n; ++k) {
        const int i = out[(size_t) k], s = to + k;
        const LayerW& L = w.L[(size_t) (i / NEXP)];
        char* at = ec.pool[p] + (size_t) s * ec.slot_bytes[p];
        ck(cudaMemcpy(at, w.ec_host_gu[(size_t) i], L.gu_e_stride, cudaMemcpyDefault), "arena: refill");
        ck(cudaMemcpy(at + ec.dn_off[p], w.ec_host_dn[(size_t) i], L.down_e_stride, cudaMemcpyDefault), "arena: refill");
        epg[(size_t) i] = at;
        epd[(size_t) i] = at + ec.dn_off[p];
        slot_of[(size_t) i] = s0 + s;
        owner[(size_t) k] = i;
    }
    ck(cudaMemcpy(ec.ep_gu, epg.data(), epg.size() * sizeof(char*), cudaMemcpyHostToDevice), "arena: tables");
    ck(cudaMemcpy(ec.ep_dn, epd.data(), epd.size() * sizeof(char*), cudaMemcpyHostToDevice), "arena: tables");
    ck(cudaMemcpy(ec.slot_of, slot_of.data(), slot_of.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: slots");
    ck(cudaMemcpy(ec.owner + s0 + to, owner.data(), owner.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: owners");
}

// lending: the slots' experts to RAM (their tables first), then K/V may write there; giving back: the pages are free,
// the slots come back and are filled again at once (arena_refill). Every card idle first: a window or a prompt pass in
// flight may read a slot being lent, or a page being given back
void Engine::arena_set(int w) {
    if (w == arena_w_) return;
    if (w < arena_base_ || w > arena_cap_) throw std::runtime_error("arena: top " + std::to_string(w) + " outside its range");
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const ExpertCache& ec = m_.cards[c].ec;
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaDeviceSynchronize(), "arena: cards idle");
        const int from = arena_slots_for(c, arena_w_), to = arena_slots_for(c, w);
        if (to > ec.n[ec.arena_pool] - EXPERT_WINDOW_SLOTS) throw std::runtime_error("arena: card " + std::to_string(c) + " would keep too few slots");
        arena_slots(ec, from, to, b.st);
        ck(cudaStreamSynchronize(b.st), "arena: slots");
        if (to < from) arena_refill(c, from, to);
    }
    if (w > arena_w_) {
        for (int pg = arena_w_; pg < w; ++pg) free_pages_.push_back(pg);
        // zeroed: they held expert weights, and the draft layer reads K/V it never wrote (batched windows skip it), which
        // as int8 rows and scales could be inf or NaN (the own pages only ever held finite K/V or zeros)
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            const size_t page = kvp_page(c) * sizeof(uint16_t);
            ck(cudaMemset((char*) cb_[c]->pool + (size_t) arena_w_ * page, 0, (size_t) (w - arena_w_) * page), "arena: zero");
        }
        arena_last_lend_ = std::chrono::steady_clock::now();
    } else {
        free_pages_.erase(std::remove_if(free_pages_.begin(), free_pages_.end(), [&](int pg) { return pg >= w; }), free_pages_.end());
    }
    std::sort(free_pages_.begin(), free_pages_.end(), std::greater<int>());
    const int old = arena_w_;
    arena_w_ = w;
    const std::vector<int> now = arena_lent();
    std::string per;
    double gib = 0;
    for (int c = 0; c < NCARD; ++c) {
        const ExpertCache& ec = m_.cards[c].ec;
        per += (c ? " / " : "") + std::to_string(now[(size_t) c]);
        gib = std::max(gib, (double) now[(size_t) c] * ec.slot_bytes[ec.arena_pool] / (1ull << 30));
    }
    std::fprintf(stderr, "fnx: arena: K/V pool %d -> %d pages (own %d, at most %d); slots lent %s (up to %.2f GiB of a card's experts in RAM), in %.1f ms\n",
                 old, w, arena_base_, arena_cap_, per.c_str(), gib,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

void Engine::arena_reclaim(double hold_s) {
    if (!arena_cap_ || arena_w_ == arena_base_) return;
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - arena_last_lend_).count() < hold_s) return;
    int top = arena_base_;
    for (const LaneHost& L : lanes_)
        for (int pg : L.pages) top = std::max(top, pg + 1);
    if (top < arena_w_) arena_set(top);
}

// FNX_ARENA_CHURN=1 (a test): before every window, ARENA_STEP more pages lent and the pool's top then given back to the
// highest page in use (those slots refilled), so lending and giving back run beside every kind of window
void Engine::arena_churn() {
    static const bool on = env_flag("FNX_ARENA_CHURN");
    if (!on || !arena_cap_) return;
    int top = arena_base_;
    for (const LaneHost& L : lanes_)
        for (int pg : L.pages) top = std::max(top, pg + 1);
    arena_set(std::min(arena_cap_, std::max(arena_w_, top) + ARENA_STEP));
    arena_set(top);
}

void Engine::ensure_pages(int n_pos) { ensure_pages_of(cur_, n_pos); }

void Engine::ensure_pages_of(int lane, int n_pos) {
    LaneHost& L = lanes_[(size_t) lane];
    const int need = n_pos / KVP_POS + 1;   // the page of position n_pos too: a block's pooled key can sit one page on
    if (need > cb_[0]->n_pages) throw Refused("context full: " + std::to_string(n_pos) + " positions");
    const size_t had = L.pages.size();
    // with an arena, slots lent for what the free pages lack: at least ARENA_STEP pages at a time
    if (const int short_by = need - (int) had - (int) free_pages_.size(); short_by > 0 && arena_w_ < arena_cap_)
        arena_set(std::min(arena_cap_, arena_w_ + std::max(short_by, ARENA_STEP)));
    if (const int short_by = need - (int) had - (int) free_pages_.size(); short_by > 0 && on_pool_short && !in_pool_short_ && pending_.empty()) {
        const int was = cur_;
        in_pool_short_ = true;
        try {
            on_pool_short(lane, short_by);
        } catch (...) {
            in_pool_short_ = false;
            throw;
        }
        in_pool_short_ = false;
        if (cur_ != was) throw std::runtime_error("on_pool_short left lane " + std::to_string(cur_) + " current, not " + std::to_string(was));
    }
    // all the pages or none: a lane holding pages its device page table lacks would write K/V through stale entries
    if (need - (int) had > (int) free_pages_.size())
        throw Refused("the K/V pool is full (" + std::to_string(n_pos) + " positions on this lane, " + std::to_string(need - (int) had) +
                      " more pages wanted, " + std::to_string(free_pages_.size()) + " free)");
    while ((int) L.pages.size() < need) {
        L.pages.push_back(free_pages_.back());
        free_pages_.pop_back();
    }
    if (L.pages.size() == had) return;
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        ck(cudaMemcpy(b->lanes[(size_t) lane].pt + had, L.pages.data() + had, (L.pages.size() - had) * sizeof(int), cudaMemcpyHostToDevice),
           "page table");
    }
}

std::vector<unsigned long long> Engine::probe_counts() const {
    const CardBuf& b = *cb_[0];
    if (!b.probe_cnt) return {};
    std::vector<unsigned long long> v(7);
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(v.data(), b.probe_cnt, 7 * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "probe counts");
    return v;
}

std::vector<unsigned> Engine::route_counts() const {
    const CardBuf& b = *cb_[0];
    if (!b.route_cnt) return {};
    std::vector<unsigned> v((size_t) 2 * NL * NEXP);
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(v.data(), b.route_cnt, v.size() * sizeof(unsigned), cudaMemcpyDeviceToHost), "route counts");
    return v;
}

std::vector<unsigned> Engine::route_trace(size_t from) const {
    const CardBuf& b = *cb_[0];
    if (!b.route_trace) return {};
    unsigned long long n = 0;
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(&n, b.route_trace_len, sizeof n, cudaMemcpyDeviceToHost), "route trace length");
    if (n > b.route_trace_cap) throw std::runtime_error("FNX_ROUTE_TRACE: " + std::to_string(n) + " picks, room for " +
                                                         std::to_string(b.route_trace_cap) + ": raise it");
    if (from >= n) return {};
    std::vector<unsigned> v((size_t) (n - from));
    std::memcpy(v.data(), b.route_trace + from, v.size() * sizeof(unsigned));
    return v;
}

std::vector<int> Engine::graph_key(const std::vector<int>& lanes) const {
    std::vector<int> k(lanes);
    for (int l : lanes) k.push_back(cb_[0]->lanes[(size_t) l].mrope != nullptr);
    k.push_back(cache_live());
    return k;
}

// a lane's states on each card, carved from a run of pool pages: recurrent states, conv histories (two halves), the
// PLE history, the QSA raw-key rings (every attention layer's and the draft layer's)
int Engine::lane_block_pages() const {
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    int n = 0;   // the most any card needs (pages differ in size between cards)
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m_.cards[c];
        size_t bytes = (size_t) 9 * D * 4 + ring_bytes + 256 * 64;   // the PLE history, the draft layer's ring, alignment
        for (int l = 0; l < NL; ++l)
            bytes += is_attn(l) ? ring_bytes : (size_t) w.nv() * HS * HS * 4 + (size_t) 2 * 3 * w.conv_ch() * 4;
        const size_t page = kvp_page(c) * sizeof(uint16_t);
        n = std::max(n, (int) ((bytes + page - 1) / page));
    }
    return n;
}

int Engine::add_lane() {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("add_lane: a window is not committed");
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    auto page_bytes = [](int c) { return kvp_page(c) * sizeof(uint16_t); };
    const int n = lane_block_pages();
    // a run of n free pages, highest first
    std::vector<int> fr(free_pages_);
    std::sort(fr.begin(), fr.end());
    int start = -1;
    for (int i = (int) fr.size() - n; i >= 0; --i)
        if (fr[(size_t) i + n - 1] - fr[(size_t) i] == n - 1) {
            start = fr[(size_t) i];
            break;
        }
    // an arena lends the run at its top (a lane's block stays as long as the engine: those slots stay lent)
    if (start < 0 && arena_w_ + n <= arena_cap_) {
        start = arena_w_;
        arena_set(arena_w_ + n);
        std::fprintf(stderr, "fnx: arena: lane %zu's block at pages %d..%d, lent for good (no run of %d free pages below)\n", lanes_.size(),
                     start, start + n - 1, n);
    }
    if (start < 0) throw std::runtime_error("add_lane: no run of " + std::to_string(n) + " free pages in the K/V pool");
    free_pages_.erase(std::remove_if(free_pages_.begin(), free_pages_.end(), [&](int p) { return p >= start && p < start + n; }),
                      free_pages_.end());
    LaneHost h;
    h.block0 = start;
    h.block_n = n;
    lanes_.push_back(h);
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        char* at = (char*) (b.pool + (size_t) start * kvp_page(c));
        ck(cudaMemset(at, 0, (size_t) n * page_bytes(c)), "lane block");
        auto take = [&](size_t bytes) {
            char* p = at;
            at += (bytes + 255) & ~(size_t) 255;
            return p;
        };
        LaneDev L;
        L.pt = dalloc<int>((size_t) b.n_pages);
        L.S.assign(NL, nullptr);
        L.chist.assign(NL, nullptr);
        L.ring.assign(NL, nullptr);
        for (int l = 0; l < NL; ++l) {
            if (is_attn(l)) {
                L.ring[l] = (uint16_t*) take(ring_bytes);
            } else {
                L.S[l] = (float*) take((size_t) w.nv() * HS * HS * 4);
                L.chist[l] = (float*) take((size_t) 2 * 3 * w.conv_ch() * 4);
            }
        }
        L.ple_hist = (float*) take((size_t) 9 * D * 4);
        L.mtp_ring = (uint16_t*) take(ring_bytes);
        L.ctl = dalloc<int>(2);
        b.lanes.push_back(L);
    }
    return (int) lanes_.size() - 1;
}

// the token rows into each of the HC streams, and the PLE rows
void Engine::embed_into(int r, int k, int p0, const std::function<int(int)>& tok_at, const char*) {
    if (!h_.ple) h_.ple = halloc<float>((size_t) MAXT * N);
    std::vector<float> emb((size_t) k * N);
    embed_rows(p0, k, tok_at, emb.data(), h_.ple + (size_t) r * N);
    for (int j = 0; j < k; ++j)
        for (int c = 0; c < HC; ++c) std::memcpy(h_.R0 + (size_t) (r + j) * D + c * N, emb.data() + (size_t) j * N, N * sizeof(float));
}

void Engine::rows_upload(CardBuf& b, int T) {
    ck(cudaMemcpyAsync(b.R, h_.R0, (size_t) T * D * 4, cudaMemcpyHostToDevice, b.st), "R0");
    ck(cudaMemcpyAsync(b.ple_emb, h_.ple, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "ple");
}

void Engine::embed_rows(int p0, int T, const std::function<int(int)>& tok_at_cell, float* emb, float* ple) const {
    std::memset(ple, 0, (size_t) T * N * sizeof(float));
    static const int8_t kv4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const Tensor& pt = *m_.ple_table;
    const size_t prb = pt.row_bytes();
    // the PLE n-grams read an image cell as the model's image token
    const auto tok_at = [&](int q) -> int64_t {
        const int t = tok_at_cell(q);
        return t >= IMAGE_TOK ? m_.ple_image : t;
    };
    for (int t = 0; t < T; ++t) {
        const int p = p0 + t;
        embed_token(tok_at_cell(p), p, emb + (size_t) t * N);
        // PLE rows: n-grams of this token and the PLE_ORDERS - 1 before it (EOS / sequence start cut the window)
        int64_t ctx[PLE_ORDERS];
        ctx[0] = tok_at(p);
        bool cut = false;
        for (int s = 1; s < PLE_ORDERS; ++s) {
            const int q = p - s;
            const int64_t tk = q >= 0 ? tok_at(q) : -1;
            cut = cut || tk < 0 || tk == m_.ple_eos;
            ctx[s] = cut ? m_.ple_eos : tk;
        }
        for (int n = 2; n <= PLE_ORDERS; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * m_.ple_mult[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * m_.ple_mult[j];
            for (int gi = 0; gi < 8; ++gi) {
                const int h = (n - 2) * 8 + gi;
                const uint64_t r = mixed % m_.ple_vocab[h] + m_.ple_off[h];
                const uint8_t* src = pt.data + r * prb;
                for (int b = 0; b < PLE_DIM / 32; ++b) {
                    float* dst = &ple[(size_t) t * N + h * PLE_DIM + b * 32];
                    uint16_t dh;
                    if (pt.type == T_Q8_0) {
                        std::memcpy(&dh, src + b * 34, 2);
                        const float d = h2f(dh);
                        for (int j = 0; j < 32; ++j) dst[j] = d * (float) (int8_t) src[b * 34 + 2 + j];
                        continue;
                    }
                    std::memcpy(&dh, src + b * 18, 2);   // IQ4_NL
                    const float d = h2f(dh);
                    const uint8_t* qs = src + b * 18 + 2;
                    for (int j = 0; j < 16; ++j) {
                        dst[j] = d * kv4[qs[j] & 15];
                        dst[j + 16] = d * kv4[qs[j] >> 4];
                    }
                }
            }
        }
    }
}

// a sequence's device state on one card, in a fixed order: (pointer, bytes) pieces
std::vector<std::pair<char*, size_t>> state_pieces(CardBuf& b, const CardW& w, bool mtp, int pos, bool kv, unsigned hist_half,
                                                   const std::vector<int>& pages) {
    std::vector<std::pair<char*, size_t>> v;
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    for (int l = 0; l < NL; ++l) {
        if (is_attn(l)) {
            v.emplace_back((char*) b.qsa_ring[l], ring_bytes);
        } else {
            v.emplace_back((char*) b.S[l], (size_t) w.nv() * HS * HS * sizeof(float));
            v.emplace_back((char*) (b.chist[l] + (size_t) hist_half * 3 * w.conv_ch()), (size_t) 3 * w.conv_ch() * sizeof(float));
        }
    }
    v.emplace_back((char*) b.ple_hist, (size_t) 9 * D * sizeof(float));
    if (mtp) v.emplace_back((char*) b.mtp_qsa_ring, ring_bytes);
    // the pages holding positions 0 .. pos (every layer's K/V rows and pooled keys, the draft layer's too)
    if (kv)
        for (int i = 0; i <= pos / KVP_POS && i < (int) pages.size(); ++i)
            v.emplace_back((char*) (b.pool + (size_t) pages[(size_t) i] * b.pg.page), b.pg.page * sizeof(uint16_t));
    return v;
}

// out[t][r] = W[r] . y[t]: the split Q8_0 kernel, or mmvq for other types
void Engine::dense(const DW& W, const void* y, float* out, CardBuf& b, int T) {
    if (W.s.q) q8s_gemv(W.s, W.n_out, W.n_in, y, out, W.n_out, T, b.st);
    else mmvq(W, 0, b.dense[T], nullptr, 1, y, out, W.n_out, b.st);
}

// an attention layer's mixer after its hc read (mixed in b.xq): q/k/v, the window's K/V into kc/vc at ctl[0], the
// attention, the output projection into b.bo_a
void Engine::attention(int T, CardBuf& b, const CardW& w, const LayerW& L, uint16_t* kc, uint16_t* vc, const int* ctl,
                       uint16_t* qsa_ring, uint16_t* qsa_pooled, int call) {
    // the indexer first (it reads mixed / its q8 copy, which the projections below leave as they are)
    if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
    qsa(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, L.idx_qn, L.idx_kn, qsa_ring, qsa_pooled, b.pg, b.qsa_q, b.qsa_scores, b.qsa_cells,
        b.qsa_ncells, max_ctx_ / 4, ctl, T, m_.eps, rope_of(b.mrope), b.att_ring.card, b.st);
    if (L.wq.s.q && L.wk.s.q && L.wv.s.q && L.wq.n_in == L.wk.n_in && L.wq.n_in == L.wv.n_in &&
        L.wq.n_out + L.wk.n_out + L.wv.n_out >= 8192) {   // q, k and v in one launch (compiled for 8192 rows and more)
        const Q8S ws[3] = {L.wq.s, L.wk.s, L.wv.s};
        const int rows[3] = {L.wq.n_out, L.wk.n_out, L.wv.n_out};
        float* const outs[3] = {b.qg, b.k, b.v};
        q8s_gemv_segs(ws, rows, outs, rows, 3, L.wq.n_in, b.xq, T, b.st);
    } else {
        dense(L.wq, b.xq, b.qg, b, T);
        dense(L.wk, b.xq, b.k, b, T);
        dense(L.wv, b.xq, b.v, b, T);
    }
    attn_prep_append(b.qg, b.k, b.v, L.q_norm, L.k_norm, kc, vc, b.pg, b.att_ring.card, T, ctl, m_.eps, rope_of(b.mrope), b.st);
    attn(b.qg, kc, vc, b.pg, b.att, b.att_ring, (const unsigned*) (ctl + 1), call, ctl, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells);
    quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
    dense(L.wo, b.xq, b.bo_a, b, T);
}

// the FFN half: the hc read with the attention half's write folded in, the shared expert (in the router's and the
// routed gate|up's launches on Q4_K layers, else on the second stream beside the router), the routing and the routed
// experts, the weighted sum across the cards into b.bo_f
void Engine::ffn_half(int T, int call, CardBuf& b, const CardW& w, const LayerW& L, float* R, const int* ctl) {
    const int P = T * TOPK, ffs = w.ff1 - w.ff0;
    hc_read(R, b.bo_a, b.inj_a, L.hc_norm[1], L.hc_down[1], L.hc_up[1], L.hc_inj[1], m_.eps, b.mixed, b.inj_f, b.hcs, b.ring.err, T, b.st, b.xq);
    // the shared expert (rows P + t of part): its gate|up in the router's launch and its swiglu and down in the routed
    // gate|up's (Q4_K experts), else on the second stream (a fork and a join cost ~5 us more than the launches' blocks)
    // (every expert type moe_gate_up handles; FNX_SH_FUSE=0 puts the non-Q4_K types back on the second stream)
    static const bool sh_all = env_int("FNX_SH_FUSE", 1) != 0;
    const bool sh_in = (L.gu_e.type == T_Q4_K || (sh_all && (L.gu_e.type == T_Q5_K || L.gu_e.type == T_Q5_1 || L.gu_e.type == T_Q4_0 || L.gu_e.type == T_Q8_0 || L.gu_e.type == T_IQ4_NL || L.gu_e.type == T_IQ4_XS || L.gu_e.type == T_IQ3_S || L.gu_e.type == T_IQ3_XXS || L.gu_e.type == T_IQ2_S))) && L.sh_gu.s.q && L.sh_down.s.q && ffs % 32 == 0 && ffs <= SH_MAXF;
    if (!sh_in) {
        ck(cudaEventRecord(b.fork, b.st), "fork");
        ck(cudaStreamWaitEvent(b.st2, b.fork, 0), "fork wait");
        q8s_gemv(L.sh_gu.s, 2 * ffs, N, b.xq, b.sh_gu, 2 * ffs, T, b.st2);
        swiglu_stacked_q8_1(b.sh_gu, b.sh_hq, ffs, T, b.st2);
        q8s_gemv(L.sh_down.s, N, ffs, b.sh_hq, b.part + (size_t) P * N, N, T, b.st2);
        ck(cudaEventRecord(b.join, b.st2), "join");
    }
    const int lyr = (int) (&L - w.L.data());
    // experts in RAM: the router's last block plans this window's slots (the prefetch probe scores the picks against
    // the slots as they were, so with it the plan runs after, in moe_cache)
    const bool live = cache_live() && lyr >= 0 && lyr < NL;
    const bool plan_in_router = live && !b.probe_cnt;
    router_route(L.router, b.mixed, b.logits, T, b.topi, b.topw, b.jobs_gu, b.jobs_dn, b.njobs, b.st, plan_in_router ? &w.ec : nullptr,
                 lyr, b.cache_list, sh_in ? &L.sh_gu.s : nullptr, 2 * ffs, b.xq, b.sh_gu);
    if (const int l = lyr; l >= 0 && l < NL) {
        if (b.route_cnt) route_count(b.topi, P, b.route_cnt + (size_t) l * NEXP, b.st);
        if (b.route_trace) fnx::route_trace(b.topi, P, l, b.route_trace, b.route_trace_len, b.route_trace_cap, b.st);
        if (b.probe_cnt) {
            if (l > 0)
                probe_score(b.topi, b.probe_guess + (size_t) l * MAXT * TOPK, T, w.ec.n_slots ? w.ec.slot_of + (size_t) l * NEXP : nullptr,
                            b.probe_cnt, b.st);
            if (l + 1 < NL) {
                gemv_f32(w.L[(size_t) l + 1].router, b.mixed, b.probe_logits, NEXP, N, T, b.st);
                probe_guess(b.probe_logits, T, b.probe_guess + (size_t) (l + 1) * MAXT * TOPK, b.st);
            }
        }
        // the planned experts copied into their slots before the MoE kernels read the layer's tables
        if (live) moe_cache(l, L.gu_e_stride, L.down_e_stride, w.ec, b.cache_list, b.st, plan_in_router ? nullptr : b.jobs_gu, b.njobs);
    }
    if (sh_in) moe_gate_up(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, b.xq, b.gu, b.st, &L.sh_down.s, ffs, b.sh_gu, b.part + (size_t) P * N);
    else moe_gate_up(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, b.xq, b.gu, b.st);
    moe_down(L.down_e, L.down_e_stride, b.jobs_dn, b.njobs, P, b.gu, b.part, b.st);
    if (!sh_in) ck(cudaStreamWaitEvent(b.st, b.join, 0), "join wait");
    RingMoe m;
    m.part = b.part;
    m.topw = b.topw;
    m.logits = b.logits;
    ring_sum(b.ring, b.bo_f, T, (const unsigned*) (ctl + 1), call, b.st, m);
}

// every card's work for one window of T rows, in stream order (captured into a graph unless dumping)
void Engine::enqueue(int T, bool dumps) {
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    auto dd = [&](const std::string& name, int layer, const float* p, int n) {
        if (dumps) dump_dev(name, layer, p, T, n);
    };
    for (int l = 0; l < NL; ++l) {
        const bool pending = l > 0;
        // ---- attention half
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = pending;
            if (l == PLE_LAYER) {
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                ple_post(b.R, b.ple_key, b.ple_val, w.ple_norm_key, w.ple_norm_query, w.ple_norm_conv, w.ple_conv,
                         b.ple_hist, b.ple_hsnap, T, eps, b.st, b.ple_key + (size_t) MAXT * D);
            }
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
            if (!is_attn(l)) {
                const int nv = w.nv(), ch = w.conv_ch();
                // ab beside qkvz in one launch (a fork and a join: ~5 us more; compiled for 4096..8191 rows)
                if (L.qkvz.s.q && 2 * nv <= 160 && L.qkvz.n_out >= 4096 && L.qkvz.n_out < 8192) {
                    q8s_gemv_f32(L.qkvz.s, L.qkvz.n_out, L.qkvz.n_in, b.xq, b.qkvz, L.ab, b.mixed, b.ab, 2 * nv, T, b.st);
                } else {
                    ck(cudaEventRecord(b.fork, b.st), "fork");
                    ck(cudaStreamWaitEvent(b.st2, b.fork, 0), "fork wait");
                    gemv_f32(L.ab, b.mixed, b.ab, 2 * nv, N, T, b.st2);
                    ck(cudaEventRecord(b.join, b.st2), "join");
                    dense(L.qkvz, b.xq, b.qkvz, b, T);
                    ck(cudaStreamWaitEvent(b.st, b.join, 0), "join wait");
                }
                gdn_core(b.qkvz, b.ab, b.chist[l], L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.S[l], b.xq, b.ctl, b.gdn_scratch, b.snap[l],
                         b.hsnap[l], w.nk, nv, w.vmap, T, eps, b.st);
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                attention(T, b, w, L, b.kc[l], b.vc[l], b.ctl, b.qsa_ring[l], b.qsa_pooled[l], 128 + l);
            }
            ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.ctl + 1), call, b.st);
        });
        ++call;
        dd("hc_mixed", l, cb_[0]->mixed, N);
        dd("mixer_out", l, cb_[0]->bo_a, N);
        // ---- FFN half
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        df2_tap_window(l, T);
        ++call;
        dd("ffn_mixed", l, cb_[0]->mixed, N);
        dd("ffn_out", l, cb_[0]->bo_f, N);
        dd("R_pending", l, cb_[0]->R, D);
        dd("inj_ffn", l, cb_[0]->inj_f, HC);
    }
    // ---- head: the final mixer (no injection) with the last write folded in, then this card's vocabulary
    each([&](int c, CardBuf& b, const CardW& w) {
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
    });
    dd("result_norm", -1, cb_[0]->mixed, N);
}

// enqueue()'s layers for a window of one row per lane (row r: lanes[r]): the shared work over all rows, the
// sequence's own work (the PLE block, the recurrent core, QSA, the rotation, the K/V append, the attention partials)
// per row with its lane's state and ctl
void Engine::enqueue_batch(const std::vector<int>& lanes) {
    const int T = (int) lanes.size();
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    for (int l = 0; l < NL; ++l) {
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = l > 0;
            if (l == PLE_LAYER) {
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                for (int r = 0; r < T; ++r)   // gated and normed rows past every key row (ple_key holds 3 MAXT rows)
                    ple_post(b.R + (size_t) r * D, b.ple_key + (size_t) r * D, b.ple_val + (size_t) r * N, w.ple_norm_key, w.ple_norm_query,
                             w.ple_norm_conv, w.ple_conv, b.lanes[(size_t) lanes[r]].ple_hist, b.ple_hsnap, 1, eps, b.st,
                             b.ple_key + (size_t) MAXT * D);
            }
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
            if (!is_attn(l)) {
                const int nv = w.nv();
                gemv_f32(L.ab, b.mixed, b.ab, 2 * nv, N, T, b.st);
                dense(L.qkvz, b.xq, b.qkvz, b, T);
                float** rec = b.lane_gdn[(size_t) T];
                const GdnLanes gl{rec + (size_t) l * MAXT, rec + (size_t) (NL + l) * MAXT, b.lane_rows[(size_t) T]};
                gdn_core_lanes(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, nv, w.vmap, gl, T, eps, b.st);
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
                else qsa_proj(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, T, b.st);
                dense(L.wq, b.xq, b.qg, b, T);
                dense(L.wk, b.xq, b.k, b, T);
                dense(L.wv, b.xq, b.v, b, T);
                const int mp = max_ctx_ / 4;
                for (int r = 0; r < T; ++r) {
                    const LaneDev& LD = b.lanes[(size_t) lanes[r]];
                    KvPages pg = b.pg;   // the card's page size and rows, the lane's table
                    pg.pt = LD.pt;
                    qsa_index(b.qsa_y + (size_t) r * qsa_rows(), L.idx_qn, L.idx_kn, LD.ring[l], b.qsa_pooled[l], pg, b.qsa_q + (size_t) r * 512,
                              LD.ctl, 1, eps, rope_of(LD.mrope), b.st);
                    attn_prep(b.qg + (size_t) r * NQH * 2 * HD, b.k + (size_t) r * NKVH * HD, L.q_norm, L.k_norm, NQH, NKVH, LD.ctl, 1, eps,
                              rope_of(LD.mrope), b.st);
                    kv_append(b.k + (size_t) r * NKVH * HD, b.v + (size_t) r * NKVH * HD, b.kc[l], b.vc[l], pg, b.att_ring.card, 1, LD.ctl,
                              b.st);
                }
                // every row's selection and attention partials at once, each on its lane's position and pages
                const LaneRows lr{b.lane_rows[(size_t) T], b.lane_rows[(size_t) T] + MAXT};
                qsa_select_lanes(b.qsa_q, b.qsa_pooled[l], b.pg, b.qsa_scores, b.qsa_cells, b.qsa_ncells, mp, lr, T, b.att_ring.card, b.st);
                attn_part_lanes(b.qg, b.kc[l], b.vc[l], b.pg, lr, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, b.att_ring.card);
                attn_finish(b.qg, b.att, b.att_ring, (const unsigned*) (b.ctl + 1), 128 + l, T, b.att_part, b.st);
                quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
                dense(L.wo, b.xq, b.bo_a, b, T);
            }
            ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.ctl + 1), call, b.st);
        });
        ++call;
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        ++call;
    }
    each([&](int, CardBuf& b, const CardW& w) {
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
    });
}

// enqueue_batch()'s layers for a window of several rows a lane (verify_rows): the shared work over all rows as there;
// each lane's own work over its rows as a window of its own does it (the rows chained through its recurrent state, the
// state and histories after each row kept for commit_rows), and every row's selection and attention partials at once,
// each at its own position (row_pos) on its lane's pages
void Engine::enqueue_rows(const std::vector<int>& lanes, const std::vector<int>& counts) {
    const RowSegs sg = row_segs(counts);
    const int T = sg.T, nl = (int) lanes.size();
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    for (int l = 0; l < NL; ++l) {
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = l > 0;
            if (l == PLE_LAYER) {
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                for (int i = 0; i < nl; ++i) {   // gated and normed rows past every key row (ple_key holds 3 MAXT rows)
                    const size_t r = (size_t) sg.r0[(size_t) i];
                    ple_post(b.R + r * D, b.ple_key + r * D, b.ple_val + r * N, w.ple_norm_key, w.ple_norm_query, w.ple_norm_conv,
                             w.ple_conv, b.lanes[(size_t) lanes[i]].ple_hist, b.ple_hsnap + r * 9 * D, counts[(size_t) i], eps, b.st,
                             b.ple_key + (size_t) MAXT * D);
                }
            }
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
            if (!is_attn(l)) {
                const int nv = w.nv();
                gemv_f32(L.ab, b.mixed, b.ab, 2 * nv, N, T, b.st);
                dense(L.qkvz, b.xq, b.qkvz, b, T);
                float** rec = b.lane_gdn[(size_t) T];
                for (const RowGroup& g : sg.groups) {
                    const GdnLanes gl{rec + (size_t) l * MAXT + g.s0, rec + (size_t) (NL + l) * MAXT + g.s0, b.seg_ctl[(size_t) T] + g.s0,
                                      b.seg_row0[(size_t) T] + g.s0, b.snap[l], b.hsnap[l]};
                    gdn_core_segs(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, nv, w.vmap, gl, g.n,
                                  g.count, eps, b.st);
                }
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
                else qsa_proj(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, T, b.st);
                dense(L.wq, b.xq, b.qg, b, T);
                dense(L.wk, b.xq, b.k, b, T);
                dense(L.wv, b.xq, b.v, b, T);
                const int mp = max_ctx_ / 4;
                for (int i = 0; i < nl; ++i) {
                    const LaneDev& LD = b.lanes[(size_t) lanes[i]];
                    KvPages pg = b.pg;   // the card's page size and rows, the lane's table
                    pg.pt = LD.pt;
                    const size_t r = (size_t) sg.r0[(size_t) i];
                    const int k = counts[(size_t) i];
                    qsa_index(b.qsa_y + r * qsa_rows(), L.idx_qn, L.idx_kn, LD.ring[l], b.qsa_pooled[l], pg, b.qsa_q + r * 512, LD.ctl, k, eps,
                              rope_of(LD.mrope), b.st);
                    attn_prep(b.qg + r * NQH * 2 * HD, b.k + r * NKVH * HD, L.q_norm, L.k_norm, NQH, NKVH, LD.ctl, k, eps,
                              rope_of(LD.mrope), b.st);
                    kv_append(b.k + r * NKVH * HD, b.v + r * NKVH * HD, b.kc[l], b.vc[l], pg, b.att_ring.card, k, LD.ctl, b.st);
                }
                const LaneRows lr{b.lane_rows[(size_t) T], b.lane_rows[(size_t) T] + MAXT};
                qsa_select_lanes(b.qsa_q, b.qsa_pooled[l], b.pg, b.qsa_scores, b.qsa_cells, b.qsa_ncells, mp, lr, T, b.att_ring.card, b.st);
                attn_part_lanes(b.qg, b.kc[l], b.vc[l], b.pg, lr, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, b.att_ring.card);
                attn_finish(b.qg, b.att, b.att_ring, (const unsigned*) (b.ctl + 1), 128 + l, T, b.att_part, b.st);
                quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
                dense(L.wo, b.xq, b.bo_a, b, T);
            }
            ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.ctl + 1), call, b.st);
        });
        ++call;
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        df2_tap_window(l, T);
        ++call;
    }
    each([&](int, CardBuf& b, const CardW& w) {
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
    });
}

}  // namespace fnx

namespace fnx {

#if FNX_DF2
// DFlash2 (df2_bridge.hpp): the drafter's card taps the window's rows after the FFN half of the tapped layers, inside the window's graph
void Engine::df2_tap_window(int l, int T) {
    if (!df2_ || df2_slot_[(size_t) l] < 0) return;
    CardBuf& b = *cb_[0];
    ck(cudaSetDevice(b.dev), "set");
    df2::Bridge::tap(b.R, b.bo_f, b.inj_f, T, df2_slot_[(size_t) l], df2_tapw_, (int64_t) df2::kTaps * df2::kNEmbd, b.st);
}

void Engine::df2_attach(df2::Bridge& br) {
    if (br.device() != cb_[0]->dev) throw std::runtime_error("df2_attach: the drafter is not on the first card");
    if (br.lanes() < (int) lanes_.size()) throw std::runtime_error("df2_attach: fewer drafters than lanes");
    if (br.tap_layers().size() != (size_t) df2::kTaps) throw std::runtime_error("df2_attach: the pack taps " + std::to_string(br.tap_layers().size()) + " layers, the engine expects " + std::to_string(df2::kTaps));
    df2_slot_.assign((size_t) NL, -1);
    for (size_t i = 0; i < br.tap_layers().size(); ++i) {
        const int l = br.tap_layers()[i];
        if (l < 0 || l >= NL - 1) throw std::runtime_error("df2_attach: tapped layer " + std::to_string(l) + " is outside the model");
        df2_slot_[(size_t) l] = (int) i;
    }
    ck(cudaSetDevice(cb_[0]->dev), "set");
    df2_tapw_ = dalloc<float>((size_t) MAXT * df2::kTaps * df2::kNEmbd);
    df2_lane_.assign((size_t) br.lanes(), Df2Lane{});
    df2_ = &br;
}
#else
void Engine::df2_tap_window(int, int) {}
void Engine::df2_attach(df2::Bridge&) { throw std::runtime_error("this engine was built without DFlash2"); }
#endif

void Engine::mtp_attach(const Mtp& mtp) {
    if (mtp.cards.size() != (size_t) NCARD) throw std::runtime_error("mtp_attach: the draft layer is not on every card");
    mtp_ = &mtp;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        b.mtp_emb = dalloc<float>((size_t) MAXT * N);
        b.mtp_en = dalloc<float>((size_t) MAXT * N);
        b.mtp_e2 = dalloc<float>((size_t) MAXT * N);
        b.mtp_src = dalloc<float>((size_t) MAXT * D);
        b.mtp_hn = dalloc<float>((size_t) MAXT * D);
        b.mtp_h2 = dalloc<float>((size_t) MAXT * D);
        b.mtp_R = dalloc<float>((size_t) MAXT * D);
        b.mtp_kc = b.pool + (size_t) (KVP_SLOTS - 1) * kvp_slot(c);   // the draft layer: the pool's last slot
        b.mtp_vc = b.mtp_kc + kvp_rows_halves(c, kv_q8());
        b.lanes[0].mtp_ring = dalloc<uint16_t>((size_t) qsa_ring_rows() * 128);   // lanes after 0 carry theirs in their blocks
        b.mtp_qsa_ring = b.lanes[(size_t) cur_].mtp_ring;
        b.mtp_qsa_pooled = b.mtp_kc + 2 * kvp_rows_halves(c, kv_q8());
        b.mtp_logits = dalloc<float>((size_t) MAXT * std::max(1, mc.dv1 - mc.dv0));
        b.mtp_bv = dalloc<float>(MAXT);
        b.mtp_bi = dalloc<int>(MAXT);
        b.mtp_ctl = dalloc<int>(2);
        b.mtp_lctl = dalloc<int>(2 * MAXT);
        b.mtp_row_pos = dalloc<int>(MAXT);
        b.mtp_lane_rows = dalloc<const int*>((size_t) LANE_ROW_COLS * MAXT);
        b.mtp_pen_l = dalloc<int>((size_t) MAXT * (1 + 2 * PEN_CAP));
        b.mtp_pen_par_l = dalloc<float>((size_t) MAXT * 3);
        b.mtp_tk_v_l = dalloc<float>((size_t) MAXT * TOPK_N);
        b.mtp_tk_i_l = dalloc<int>((size_t) MAXT * TOPK_N);
        b.mtp_tk_err_l = dalloc<unsigned>(MAXT);
        b.mtp_pen = dalloc<int>(1 + 2 * PEN_CAP);
        b.mtp_pen_par = dalloc<float>(3);
        b.mtp_tk_v = dalloc<float>(TOPK_N);
        b.mtp_tk_i = dalloc<int>(TOPK_N);
        b.mtp_tk_err = dalloc<unsigned>(1);
        b.mtp_zero = dalloc<int>(1);
        ck(cudaMemset(b.mtp_zero, 0, 4), "mtp zero");
    }
    if (!h_.mtp_emb) {
        h_.mtp_emb = halloc<float>((size_t) MAXT * N);
        h_.mtp_ctl = halloc<int>(2);
        h_.mtp_lctl = halloc<int>(2 * MAXT);
        h_.mtp_row_pos = halloc<int>(MAXT);
        h_.mtp_pen_l = halloc<int>((size_t) MAXT * NCARD * (1 + 2 * PEN_CAP));
        h_.mtp_pen_par_l = halloc<float>((size_t) MAXT * 3);
        h_.mtp_tk_v_l = halloc<float>((size_t) MAXT * NCARD * TOPK_N);
        h_.mtp_tk_i_l = halloc<int>((size_t) MAXT * NCARD * TOPK_N);
        h_.mtp_tk_err_l = halloc<unsigned>((size_t) MAXT * NCARD);
        h_.mtp_pen = halloc<int>((size_t) NCARD * (1 + 2 * PEN_CAP));
        h_.mtp_pen_par = halloc<float>(3);
        h_.mtp_tk_v = halloc<float>((size_t) NCARD * TOPK_N);
        h_.mtp_tk_i = halloc<int>((size_t) NCARD * TOPK_N);
        h_.mtp_tk_err = halloc<unsigned>(NCARD);
    }
    draft_index_.assign((size_t) m_.vocab, -1);
    for (size_t i = 0; i < mtp.vocab.size(); ++i) draft_index_[(size_t) mtp.vocab[i]] = (int) i;
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
}

// one draft-layer window of T rows from b.mtp_src (rows of D) and b.mtp_emb (the tokens' embeddings), cells at
// mtp_ctl[0]: R = fc_hidden(rms(src) w) per stream + fc_embedding(rms(emb) w), then the layer as a main attention
// layer, then the final mixer and this card's draft head rows
void Engine::mtp_enqueue(int T) {
    const float eps = m_.eps;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp_->cards[c];
        const LayerW& L = mc.L;
        ck(cudaSetDevice(b.dev), "set");
        rms_norm_rows(b.mtp_emb, mc.norm_emb, b.mtp_en, T, N, eps, b.st);
        quantize_q8_1(b.mtp_en, b.xq, N, T, b.st);
        dense(mc.fc_emb, b.xq, b.mtp_e2, b, T);
        rms_norm_rows(b.mtp_src, mc.norm_hid, b.mtp_hn, T, D, eps, b.st);
        quantize_q8_1(b.mtp_hn, b.xq, N, HC * T, b.st);
        for (int r0 = 0; r0 < HC * T; r0 += MAXT)   // every stream of every row through fc_hidden
            q8s_gemv(mc.fc_hid.s, N, N, (const uint8_t*) b.xq + (size_t) r0 * q8_1_bytes(N), b.mtp_h2 + (size_t) r0 * N, N,
                     std::min(MAXT, HC * T - r0), b.st);
        add_streams(b.mtp_h2, b.mtp_e2, b.mtp_R, T, b.st);
        hc_read(b.mtp_R, nullptr, nullptr, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T,
                b.st, b.xq);
        attention(T, b, w, L, b.mtp_kc, b.mtp_vc, b.mtp_ctl, b.mtp_qsa_ring, b.mtp_qsa_pooled, 128);
        ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.mtp_ctl + 1), 0, b.st);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ffn_half(T, 1, b, m_.cards[c], mtp_->cards[c].L, b.mtp_R, b.mtp_ctl);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const MtpCard& mc = mtp_->cards[c];
        ck(cudaSetDevice(b.dev), "set");
        hc_read(b.mtp_R, b.bo_f, b.inj_f, mc.mix_norm, mc.mix_down, mc.mix_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(mc.head, b.xq, b.mtp_logits, b, T);
        penalize_row(b.mtp_logits + (size_t) (T - 1) * (mc.dv1 - mc.dv0), mc.dv1 - mc.dv0, b.mtp_pen, b.mtp_pen_par, b.st);
        argmax(b.mtp_logits, mc.dv1 - mc.dv0, T, mc.dv0, b.mtp_bv, b.mtp_bi, b.am_keys, b.ring.err, b.st);
    }
}


// The draft layer for several lanes in ONE window (Engine::draft_lanes): mtp_enqueue's layer, with the attention of enqueue_rows' batched windows for the draft
// layer's own K/V slot (each lane's rows at its cells on its pages). Greedy: no penalties, argmax of every row.
void Engine::mtp_enqueue_lanes(const std::vector<DraftReq>& reqs, int T) {
    const float eps = m_.eps;
    const int nl = (int) reqs.size(), mp = max_ctx_ / 4;
    std::vector<int> off((size_t) nl);
    for (int i = 0, o = 0; i < nl; ++i) off[(size_t) i] = o, o += reqs[(size_t) i].T;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp_->cards[c];
        const LayerW& L = mc.L;
        ck(cudaSetDevice(b.dev), "set");
        rms_norm_rows(b.mtp_emb, mc.norm_emb, b.mtp_en, T, N, eps, b.st);
        quantize_q8_1(b.mtp_en, b.xq, N, T, b.st);
        dense(mc.fc_emb, b.xq, b.mtp_e2, b, T);
        rms_norm_rows(b.mtp_src, mc.norm_hid, b.mtp_hn, T, D, eps, b.st);
        quantize_q8_1(b.mtp_hn, b.xq, N, HC * T, b.st);
        for (int r0 = 0; r0 < HC * T; r0 += MAXT)
            q8s_gemv(mc.fc_hid.s, N, N, (const uint8_t*) b.xq + (size_t) r0 * q8_1_bytes(N), b.mtp_h2 + (size_t) r0 * N, N, std::min(MAXT, HC * T - r0), b.st);
        add_streams(b.mtp_h2, b.mtp_e2, b.mtp_R, T, b.st);
        hc_read(b.mtp_R, nullptr, nullptr, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        // the attention as a batched window does it, on the draft layer's slot of the pool
        if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
        else qsa_proj(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, T, b.st);
        dense(L.wq, b.xq, b.qg, b, T);
        dense(L.wk, b.xq, b.k, b, T);
        dense(L.wv, b.xq, b.v, b, T);
        for (int i = 0; i < nl; ++i) {
            const LaneDev& LD = b.lanes[(size_t) reqs[(size_t) i].lane];
            KvPages pg = b.pg;
            pg.pt = LD.pt;
            const size_t r = (size_t) off[(size_t) i];
            const int k = reqs[(size_t) i].T;
            const int* lc = b.mtp_lctl + 2 * i;
            qsa_index(b.qsa_y + r * qsa_rows(), L.idx_qn, L.idx_kn, LD.mtp_ring, b.mtp_qsa_pooled, pg, b.qsa_q + r * 512, lc, k, eps, rope_of(LD.mrope), b.st);
            attn_prep(b.qg + r * NQH * 2 * HD, b.k + r * NKVH * HD, L.q_norm, L.k_norm, NQH, NKVH, lc, k, eps, rope_of(LD.mrope), b.st);
            kv_append(b.k + r * NKVH * HD, b.v + r * NKVH * HD, b.mtp_kc, b.mtp_vc, pg, b.att_ring.card, k, lc, b.st);
        }
        const LaneRows lr{b.mtp_lane_rows, b.mtp_lane_rows + MAXT};
        qsa_select_lanes(b.qsa_q, b.mtp_qsa_pooled, b.pg, b.qsa_scores, b.qsa_cells, b.qsa_ncells, mp, lr, T, b.att_ring.card, b.st);
        attn_part_lanes(b.qg, b.mtp_kc, b.mtp_vc, b.pg, lr, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, b.att_ring.card);
        attn_finish(b.qg, b.att, b.att_ring, (const unsigned*) (b.mtp_ctl + 1), 128, T, b.att_part, b.st);
        quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
        dense(L.wo, b.xq, b.bo_a, b, T);
        ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.mtp_ctl + 1), 0, b.st);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ffn_half(T, 1, b, m_.cards[c], mtp_->cards[c].L, b.mtp_R, b.mtp_ctl);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const MtpCard& mc = mtp_->cards[c];
        ck(cudaSetDevice(b.dev), "set");
        hc_read(b.mtp_R, b.bo_f, b.inj_f, mc.mix_norm, mc.mix_down, mc.mix_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(mc.head, b.xq, b.mtp_logits, b, T);
        const int nv = mc.dv1 - mc.dv0;
        for (int i = 0; i < nl; ++i)   // each lane's last row, penalized as draft() does before the argmax
            if (reqs[(size_t) i].pen)
                penalize_row(b.mtp_logits + (size_t) (off[(size_t) i] + reqs[(size_t) i].T - 1) * nv, nv, b.mtp_pen_l + (size_t) i * (1 + 2 * PEN_CAP),
                             b.mtp_pen_par_l + 3 * i, b.st);
        argmax(b.mtp_logits, mc.dv1 - mc.dv0, T, mc.dv0, b.mtp_bv, b.mtp_bi, b.am_keys, b.ring.err, b.st);
        for (int i = 0; i < nl; ++i)   // and the candidates of the lanes that sample their drafts
            if (reqs[(size_t) i].cands && nv > 0) {
                ck(cudaMemsetAsync(b.mtp_tk_err_l + i, 0, 4, b.st), "draft top err");
                fnx::top_tokens(b.mtp_logits + (size_t) (off[(size_t) i] + reqs[(size_t) i].T - 1) * nv, nv, 1, mc.dv0, b.mtp_zero, b.mtp_zero, b.mtp_zero, 0,
                                b.mtp_pen_par_l + 3 * i, b.mtp_tk_v_l + (size_t) i * TOPK_N, b.mtp_tk_i_l + (size_t) i * TOPK_N, b.mtp_tk_err_l + i, b.st);
            }
    }
}

}  // namespace fnx
