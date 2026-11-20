// hy4_graph.hip.h - the 42-layer hy_v4_flash forward on gfx1151 (HIP): chunked prefill + decode, KV fp16, absorbed MLA, LatentMoE.
// Built from: skgemm (linear_h / linear_f32), skmoe (moe_experts), mla_attn (mla_*), hy4_kernels (norms, iHC, router, engram), hy4_engram (hash + page reads).
#pragma once
#include "hy4_gpu.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <future>
#include <string>
#include <vector>

#include "hy4_cpu_experts.h"
#include "hy4_engram.h"
#include "hy4_gpu_model.hip.h"
#include "hy4_indexer.hip.h"
#include "hy4_kernels.hip.h"
#include "hy4_vision.hip.h"  // HY4-VISION hook: image rows + 3D rope (all hooks below are no-ops while Runner::vis is null)
#include "mla_attn.hip.h"
#include "skhy4.hip.h"

#ifndef TUNE_DEFAULT
#define TUNE_DEFAULT 834  // default HY4_TUNE bitmask when the environment variable is not set (see Runner::tune_init)
#endif

namespace hy4 {

// device dequantization of kv_b_proj [NH*448][512] (any S_K fmt) into wk_b [NH][192][512] and wv_b [NH][256][512] fp16
template <int FMT>
__global__ void k_split_kvb(const uint8_t* __restrict__ W, __half* __restrict__ wk, __half* __restrict__ wv, int NH) {
    int r = blockIdx.x;  // row in [0, NH*448)
    int h = r / 448, j = r % 448;
    constexpr int SB = sk_sb_bytes(FMT);
    const uint8_t* rp = W + (size_t)r * 2 * SB;
    __half* dst = j < 192 ? wk + ((size_t)h * 192 + j) * 512 : wv + ((size_t)h * 256 + (j - 192)) * 512;
    for (int k = threadIdx.x; k < 512; k += blockDim.x) dst[k] = __float2half(skd::sk_weight<FMT>(rp + (size_t)(k >> 8) * SB, k & 255));
}

static void save_npy_f32(const std::string& path, const float* d, const std::vector<int64_t>& shape) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) sh += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
    sh += ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t tot = 10 + hdr.size() + 1;
    size_t pad = (64 - tot % 64) % 64;
    hdr += std::string(pad, ' ') + "\n";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite("\x93NUMPY", 1, 6, f);
    unsigned char ver[2] = {1, 0};
    fwrite(ver, 1, 2, f);
    uint16_t hl = (uint16_t)hdr.size();
    fwrite(&hl, 2, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    fwrite(d, 4, n, f);
    fclose(f);
}

struct EngramDev {  // per module, gathered at init
    float* lnw = nullptr;      // [16][96]
    float* kvb = nullptr;      // [5][3072]
    float* egate = nullptr;    // [4][3072] norm_key*norm_query
    float* pre_gamma = nullptr;  // [3072]
    float* hcs = nullptr;      // [4][3072]
};

class Runner {
public:
    GpuModel& g;
    const Cfg& c;
    int max_ctx, Tc;
    hipStream_t st = nullptr;
    // buffers
    int32_t* d_ids = nullptr;
    float *x = nullptr, *pre = nullptr, *post = nullptr, *cur_f = nullptr, *F1 = nullptr, *F2 = nullptr, *F3 = nullptr, *sub = nullptr, *ctx_ws = nullptr;
    __half *cur_h = nullptr, *qr_h = nullptr, *qa_h = nullptr, *o_h = nullptr, *H1 = nullptr, *z_h = nullptr, *er_h = nullptr, *hfin = nullptr;
    float *z_f = nullptr, *zout_f = nullptr, *moe_w = nullptr, *logits = nullptr;
    int32_t *moe_idx = nullptr, *d_argmax = nullptr;
    void *moe_ws = nullptr, *mla_ws = nullptr;
    size_t mla_ws_bytes = 0;
    float* kv_f = nullptr;
    uint8_t* d_eraw[4] = {};
    uint8_t* d_evalid[4] = {};
    EngramDev eng_dev[4];
    ExpertDev* d_experts = nullptr;
    __half* wk_s = nullptr;   // single-layer scratch [64][192][512]
    __half* wk_st = nullptr;  // single-layer transposed scratch [64][512][192] for mla_prep_q_t
    __half* wv_s = nullptr;   // single-layer scratch [64][256][512]
    bool want_all_logits = false;  // when false (default), logits allocates only 16 rows (~7.7 MB vs ~990 MB at Tc=2048)
    std::vector<__half*> kcache;  // per layer
    int ring = 0, slots_sparse = 0;
    // growable KV state (cards that cannot hold the whole context): the sparse-layer KV rows + Vt and the IndexPool state start in VRAM for kv_cap positions and move to pinned host memory
    // (all of it, once) when the context passes kv_cap, so short chats in a 128K context run at VRAM speed. kv_cap == 0: not growable.
    int kv_cap = 0;
    bool kv_moved = false;
    void ensure_mla_ws(size_t need) {
        if (need <= mla_ws_bytes) return;
        HCK(hipStreamSynchronize(st));
        if (mla_ws) HCK(hipFree(mla_ws));
        mla_ws_bytes = need + (need >> 2);
        HCK(hipMalloc(&mla_ws, mla_ws_bytes));
        if (getenv("HY4_POISON")) HCK(hipMemset(mla_ws, 0xFF, mla_ws_bytes));
    }
    // host side engram
    EngramMeta emeta;
    EngramIO eio;
    std::vector<uint8_t> h_eraw, h_evalid;
    bool engram_ready = false;
    // stats
    double t_engram_cpu = 0;
    bool dump = false;
    int dump_layer = -1;  // with dump: also dump the inner buffers of this layer's attention (debug)
    void dump_half(const std::string& name, const __half* dev, size_t n, int T) {
        HCK(hipStreamSynchronize(st));
        std::vector<__half> h(n);
        HCK(hipMemcpy(h.data(), dev, n * 2, hipMemcpyDeviceToHost));
        std::vector<float> f(n);
        for (size_t i = 0; i < n; ++i) f[i] = __half2float(h[i]);
        save_npy_f32(dump_dir + "/" + name + ".npy", f.data(), {(int64_t)T, (int64_t)(n / T)});
    }
    std::string dump_dir = ".";
    VisionCtx* vis = nullptr;  // HY4-VISION hook: the pictures of the running request (hy4_vision.hip.h); null for a text-only request
    bool dbg_layers = false;
    int index_on = 1;       // IndexPool selection on the sparse layers (off: they attend all keys)
    int index_topk = 2048;  // selection starts when the sequence is longer than this
    // indexer state
    std::vector<__half*> ivg, pooled;
    float *ik_f = nullptr, *iv_f = nullptr, *ig_f = nullptr, *iq_f = nullptr, *hw_f = nullptr, *score = nullptr;
    __half *ik_h = nullptr, *iq_h = nullptr;
    int32_t *keys = nullptr, *nkeys = nullptr;
    size_t ldS = 0;
    bool last_use_list = false;

    // DFly2 hook: the top-index_topk key lists of target layers 25,29,33,37,41 for the last (up to 16) rows of the most recent forward chunk (hy4_dfly2.hip.h reads them)
    bool dfly_cap = false;
    int32_t *dkeys[5] = {}, *dnkc[5] = {};  // [16][index_topk], [16]
    int cap_p0 = -1, cap_rows = 0;
    bool cap_list[5] = {};
    static int dfly_slot(int il) { return il == 25 ? 0 : il == 29 ? 1 : il == 33 ? 2 : il == 37 ? 3 : il == 41 ? 4 : -1; }
    void dfly_capture_alloc() {
        for (int d = 0; d < 5; ++d) {
            if (!dkeys[d]) {
                dkeys[d] = dalloc<int32_t>((size_t)16 * (index_topk + 8));
                dnkc[d] = dalloc<int32_t>(16);
            }
        }
        dfly_cap = true;
    }
    void dfly_capture(int il, int p0, int T, bool use_list) {
        const int d = dfly_slot(il);
        if (d < 0) return;
        const int nr = std::min(T, 16), r0 = T - nr;
        cap_p0 = p0 + r0;
        cap_rows = nr;
        cap_list[d] = use_list;
        if (use_list) {
            HCK(hipMemcpyAsync(dkeys[d], keys + (size_t)r0 * index_topk, (size_t)nr * index_topk * 4, hipMemcpyDeviceToDevice, st));
            HCK(hipMemcpyAsync(dnkc[d], nkeys + r0, (size_t)nr * 4, hipMemcpyDeviceToDevice, st));
        }
    }

    // hybrid placement: tier-2 experts live in host RAM and are computed by hy4_cpu_experts while the GPU runs the shared experts and the VRAM-resident experts
    CpuExperts cpu;
    bool hyb = false, has_cold = false;
    std::vector<uint8_t> cold_flag, h_use;  // [41*512] cold experts by (layer, id); [Tc*16] per-pair "computed by the CPU" flags of a prefill chunk
    float *h_z = nullptr, *h_w = nullptr, *h_zcpu = nullptr, *d_zcpu = nullptr;
    int32_t* h_idx = nullptr;
    hipEvent_t ev_hyb = nullptr;
    // prefill streaming of the CPU-tier experts: for chunks of >= pf_min_T tokens the experts of a layer are DMA'd from the host arena (registered/pinned in place) into a device
    // scratch buffer while the attention of that layer runs, and the grouped GEMM of the GPU runs on them (the CPU would need ~100 ms per layer and 512 tokens).
    // When a whole layer does not fit the scratch space (small cards), the layer's experts are split into groups that fit one of two alternating buffers: group k+1 is copied
    // while group k computes. Every group has its own ExpertDev table (the experts of other groups carry pad = 1 and are skipped by the routing; the VRAM-tier experts run in
    // group 0), and the partial sums are added up in fp32.
    struct PfRun { const uint8_t* host; size_t dev_off, bytes; };
    struct PfGroup { std::vector<PfRun> runs; size_t total_bytes = 0; int seq = 0; ExpertDev* table = nullptr; };
    std::vector<std::vector<PfGroup>> pf_groups;  // [41 MoE layers][groups]
    std::vector<int> pf_layer_first;              // sequence index of group 0 of each layer
    int pf_nseq = 0, pf_nbuf = 1, pf_next_copy = 0, pf_next_comp = 0;
    uint8_t* d_scratch[2] = {nullptr, nullptr};
    uint8_t* h_bounce[2] = {nullptr, nullptr};
    bool pf_bounce = false;
    hipStream_t st2 = nullptr;
    hipEvent_t ev_pf_copy[2] = {nullptr, nullptr}, ev_pf_done[2] = {nullptr, nullptr};
    bool ev_pf_done_set[2] = {false, false}, ev_pf_copy_set[2] = {false, false};
    bool pf_stream = false;
    int pf_min_T = 320;
    bool pf_active(int T) const { return pf_stream && T >= pf_min_T; }
    void pf_init() {
        auto al = [](size_t x) { return (x + 4095) & ~(size_t)4095; };
        std::vector<std::vector<std::pair<const uint8_t*, int>>> lay(41);
        std::vector<size_t> tot_l(41, 0);
        size_t maxb = 0;
        const uint8_t *lo = nullptr, *hi = nullptr;
        for (int l = 0; l < 41; ++l) {
            for (int e = 0; e < 512; ++e) {
                CpuExpertView vw = cpu.view(l, e);
                if (vw.body) {
                    lay[l].push_back({vw.body, e});
                    const size_t sz = al(vw.body_bytes);
                    tot_l[l] += sz;
                    if (!lo || vw.body < lo) lo = vw.body;
                    if (!hi || vw.body + sz > hi) hi = vw.body + sz;
                }
            }
            std::sort(lay[l].begin(), lay[l].end());
            maxb = std::max(maxb, tot_l[l]);
        }
        if (!maxb) return;
        double t0 = now_s();
        pf_bounce = false;
        if (hipHostRegister((void*)lo, (size_t)(hi - lo), 0) != hipSuccess) {
            (void)hipGetLastError();
            pf_bounce = true;
            fprintf(stderr, "[hybrid] cannot pin full CPU expert arena (%.1f GB); falling back to pinned bounce-ring prefill streaming\n", (double)(hi - lo) / 1e9);
        }
        size_t fr = 0, tot = 0;
        HCK(hipMemGetInfo(&fr, &tot));
        const size_t margin = (size_t)200 << 20, tables_est = (size_t)41 * 16 * 8192;
        size_t avail = fr > margin + tables_est ? fr - margin - tables_est : 0;
        size_t cap;
        if (pf_bounce) {
            pf_nbuf = 2;
            cap = std::min<size_t>((avail / 2) & ~(size_t)4095, (size_t)256 << 20);
            if (cap > maxb) cap = (maxb + 4095) & ~(size_t)4095;
        } else if (avail >= maxb) {
            pf_nbuf = 1;
            cap = maxb;
        } else {
            pf_nbuf = 2;
            cap = (avail / 2) & ~(size_t)4095;
        }
        if (cap < ((size_t)64 << 20)) {
            fprintf(stderr, "[hybrid] prefill streaming disabled: no room for a scratch buffer (free %.2f GB, layer %.2f GB)\n", fr / 1e9, maxb / 1e9);
            return;
        }
        if (pf_bounce) {
            for (int b = 0; b < pf_nbuf; ++b) {
                if (hipHostMalloc((void**)&h_bounce[b], cap, hipHostMallocDefault) != hipSuccess) {
                    (void)hipGetLastError();
                    for (int k = 0; k < b; ++k) { (void)hipHostFree(h_bounce[k]); h_bounce[k] = nullptr; }
                    fprintf(stderr, "[hybrid] prefill streaming disabled: cannot allocate pinned bounce buffer (%.2f GB)\n", cap / 1e9);
                    return;
                }
            }
        }
        for (int b = 0; b < pf_nbuf; ++b) d_scratch[b] = dalloc<uint8_t>(cap);
        // groups, sequence numbers and tables
        pf_groups.assign(41, {});
        pf_layer_first.assign(41, 0);
        std::vector<ExpertDev> all;
        std::vector<std::vector<std::vector<int>>> gexp(41);
        int seq = 0;
        for (int l = 0; l < 41; ++l) {
            pf_layer_first[l] = seq;
            pf_groups[l].emplace_back();
            gexp[l].emplace_back();
            size_t off = 0;
            for (auto& pe : lay[l]) {
                CpuExpertView vw = cpu.view(l, pe.second);
                const size_t sz = al(vw.body_bytes);
                if (off + sz > cap) {
                    pf_groups[l].back().total_bytes = off;
                    pf_groups[l].emplace_back();
                    gexp[l].emplace_back();
                    off = 0;
                }
                PfGroup& G = pf_groups[l].back();
                if (!G.runs.empty() && G.runs.back().host + G.runs.back().bytes == pe.first) G.runs.back().bytes += sz;
                else G.runs.push_back({pe.first, off, sz});
                gexp[l].back().push_back(pe.second);
                off += sz;
            }
            pf_groups[l].back().total_bytes = off;
            for (size_t gi = 0; gi < pf_groups[l].size(); ++gi) {
                pf_groups[l][gi].seq = seq++;
            }
        }
        pf_nseq = seq;
        // per-group tables: offsets of the group's experts inside its buffer follow the order of gexp
        for (int l = 0; l < 41; ++l)
            for (size_t gi = 0; gi < pf_groups[l].size(); ++gi) {
                const int k = pf_groups[l][gi].seq, b = k % pf_nbuf;
                std::vector<ExpertDev> hp(512);
                std::vector<char> in_group(512, 0);
                std::vector<size_t> eoff(512, 0);
                size_t off = 0;
                for (int e : gexp[l][gi]) {
                    CpuExpertView vw = cpu.view(l, e);
                    in_group[e] = 1;
                    eoff[e] = off;
                    off += al(vw.body_bytes);
                }
                for (int e = 0; e < 512; ++e) {
                    const ExpertRef& er = g.experts[l * 512 + e];
                    hp[e].dn_off = er.dn_off;
                    hp[e].gu_code = er.gu_code;
                    hp[e].dn_code = er.dn_code;
                    if (er.tier == 3) {
                        hp[e].body_off = 0;
                        hp[e].pad = 1;  // cold expert: computed by the CPU pool from the SSD
                    } else if (er.tier == 2) {
                        if (in_group[e]) {
                            hp[e].body_off = (uint64_t)(uintptr_t)(d_scratch[b] + eoff[e]) - (uint64_t)(uintptr_t)g.d_sec4;
                            hp[e].pad = 0;
                        } else {
                            hp[e].body_off = 0;
                            hp[e].pad = 1;
                        }
                    } else {
                        hp[e].body_off = er.body_off;
                        hp[e].pad = gi == 0 ? 0 : 1;  // the GPU-resident experts run once per layer, with the first group
                    }
                }
                all.insert(all.end(), hp.begin(), hp.end());
            }
        ExpertDev* dt = dalloc<ExpertDev>(all.size());
        HCK(hipMemcpy(dt, all.data(), all.size() * sizeof(ExpertDev), hipMemcpyHostToDevice));
        {
            size_t idx = 0;
            for (int l = 0; l < 41; ++l)
                for (size_t gi = 0; gi < pf_groups[l].size(); ++gi) pf_groups[l][gi].table = dt + 512 * idx++;
        }
        HCK(hipStreamCreate(&st2));
        for (int b = 0; b < 2; ++b) {
            HCK(hipEventCreateWithFlags(&ev_pf_copy[b], hipEventDisableTiming));
            HCK(hipEventCreateWithFlags(&ev_pf_done[b], hipEventDisableTiming));
        }
        if (getenv("HY4_PF_MIN_T")) pf_min_T = atoi(getenv("HY4_PF_MIN_T"));
        pf_stream = true;
        size_t ng = 0;
        for (auto& v : pf_groups) ng += v.size();
        fprintf(stderr, "[hybrid] prefill streaming on (%s): %d buffer(s) of %.2f GB, %zu groups for 41 layers (largest layer %.2f GB), init in %.1f s, used for chunks >= %d tokens\n",
                pf_bounce ? "bounce-ring" : "pinned-arena", pf_nbuf, cap / 1e9, ng, maxb / 1e9, now_s() - t0, pf_min_T);
    }
    void pf_begin_chunk() {
        pf_next_copy = 0;
        pf_next_comp = 0;
    }
    // queue the DMA of sequence entries up to `upto` (inclusive) whose buffer is free (the compute that used it nbuf entries earlier has been enqueued)
    void pf_pump(int upto) {
        while (pf_next_copy <= upto && pf_next_copy < pf_nseq) {
            const int k = pf_next_copy, b = k % pf_nbuf;
            if (k - pf_nbuf >= pf_next_comp) break;
            const PfGroup* G = pf_find(k);
            if (pf_bounce) {
                if (ev_pf_copy_set[b]) HCK(hipEventSynchronize(ev_pf_copy[b]));
                for (const PfRun& r : G->runs) std::memcpy(h_bounce[b] + r.dev_off, r.host, r.bytes);
                if (ev_pf_done_set[b]) HCK(hipStreamWaitEvent(st2, ev_pf_done[b], 0));
                if (G->total_bytes) HCK(hipMemcpyAsync(d_scratch[b], h_bounce[b], G->total_bytes, hipMemcpyHostToDevice, st2));
                HCK(hipEventRecord(ev_pf_copy[b], st2));
                ev_pf_copy_set[b] = true;
            } else {
                if (ev_pf_done_set[b]) HCK(hipStreamWaitEvent(st2, ev_pf_done[b], 0));
                for (const PfRun& r : G->runs) HCK(hipMemcpyAsync(d_scratch[b] + r.dev_off, r.host, r.bytes, hipMemcpyHostToDevice, st2));
                HCK(hipEventRecord(ev_pf_copy[b], st2));
            }
            ++pf_next_copy;
        }
    }
    const PfGroup* pf_find(int k) const {
        // layers are contiguous in the sequence: binary search the layer, then index the group
        int lo = 0, hi = 40;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (pf_layer_first[mid] <= k) lo = mid; else hi = mid - 1;
        }
        return &pf_groups[lo][k - pf_layer_first[lo]];
    }
    double t_cpu_wait = 0;  // host time spent waiting for the CPU experts after the GPU work was queued (layer sum)

    Runner(GpuModel& gm, int max_ctx_, int max_chunk) : g(gm), c(gm.man.c), max_ctx(max_ctx_), Tc(max_chunk) {}

    // big, rarely-touched state (sparse-layer KV cache, IndexPool, kv_b absorbed copies): VRAM when it fits and HY4_KV_HOST != 1, otherwise pinned host memory mapped into the
    // device address space (the kernels read it over PCIe: attention only reads the 2048 selected keys of a sparse layer, the pool scoring is served by L2 after the first block)
    size_t host_big_bytes = 0;
    // `bit` selects the class: 1 = sparse KV rows + Vt, 2 = absorbed wk_b/wv_b, 4 = IndexPool state (pooled keys, iv/ig). HY4_KV_HOST is a bit mask of the classes forced to host memory
    // (plan_memory sets it from the card size and the context); a class that is not forced still falls back to host memory when VRAM runs out.
    template <class T> T* dalloc_big(size_t n, const char* what, int bit) {
        const size_t bytes = n * sizeof(T);
        const char* e = getenv("HY4_KV_HOST");
        const int mode = e ? atoi(e) : 0;
        const bool forced = (mode & bit) != 0;
        void* p = nullptr;
        if (!forced) {
            size_t fr = 0, tot = 0;
            HCK(hipMemGetInfo(&fr, &tot));
            if (fr > bytes + ((size_t)300 << 20)) {
                if (hipMalloc(&p, bytes) == hipSuccess) {
                    HCK(hipMemset(p, 0, bytes));
                    return (T*)p;
                }
                (void)hipGetLastError();
            }
        }
        void *hp = nullptr, *dp = nullptr;
        HCK(hipHostMalloc(&hp, bytes, hipHostMallocMapped));
        memset(hp, 0, bytes);
        HCK(hipHostGetDevicePointer(&dp, hp, 0));
        host_big_bytes += bytes;
        fprintf(stderr, "[runner] %s: %.2f GB in pinned host memory (read by the GPU)\n", what, bytes / 1e9);
        return (T*)dp;
    }
    template <class T> T* dalloc(size_t n) {
        T* p = nullptr;
        HCK(hipMalloc((void**)&p, n * sizeof(T)));
        if (getenv("HY4_POISON")) HCK(hipMemset(p, 0xFF, n * sizeof(T)));  // debug: NaN / -1 fill, finds reads of uninitialised device memory
        return p;
    }
    const void* tw(const std::string& name) const { return g.tensor_dev(name); }
    const TInfo& ti(const std::string& name) const { return g.man.get(name); }
    std::string L(int il) const { return "model.layers." + std::to_string(il); }
    int heads(int il) const { return c.is_swa[il] ? c.heads_swa : c.heads_sparse; }
    double theta(int il) const { return c.is_swa[il] ? c.theta_swa : c.theta_sparse; }

    // per-kernel timing (prof2): hipEvent pairs per labelled section, summed per label at report time
    bool prof2 = false;
    struct PEv { hipEvent_t a, b; std::string name; };
    std::vector<PEv> pev;
    std::vector<std::pair<hipEvent_t, hipEvent_t>> pev_pool;
    std::map<std::string, std::pair<double, int>> pacc;
    struct PS {
        Runner* r; bool on; size_t idx = 0;
        PS(Runner* r_, const std::string& nm) : r(r_), on(r_->prof2) {
            if (!on) return;
            PEv e;
            if (!r->pev_pool.empty()) { e.a = r->pev_pool.back().first; e.b = r->pev_pool.back().second; r->pev_pool.pop_back(); }
            else { HCK(hipEventCreate(&e.a)); HCK(hipEventCreate(&e.b)); }
            e.name = nm;
            HCK(hipEventRecord(e.a, r->st));
            idx = r->pev.size();
            r->pev.push_back(e);
        }
        ~PS() { if (on) HCK(hipEventRecord(r->pev[idx].b, r->st)); }
    };
    void pflush() {
        if (!prof2 || pev.empty()) return;
        HCK(hipStreamSynchronize(st));
        for (auto& e : pev) {
            float ms = 0;
            HCK(hipEventElapsedTime(&ms, e.a, e.b));
            auto& a = pacc[e.name];
            a.first += ms;
            a.second++;
            pev_pool.push_back({e.a, e.b});
        }
        pev.clear();
    }
    void pacc_report(const char* what, int steps) {
        if (!prof2) return;
        pflush();
        std::vector<std::pair<double, std::string>> v;
        for (auto& kv : pacc) v.push_back({kv.second.first, kv.first});
        std::sort(v.rbegin(), v.rend());
        fprintf(stderr, "[prof2 %s] per-step ms by section (sum of GPU event time / %d steps):\n", what, steps);
        double tot = 0;
        for (auto& p : v) {
            tot += p.first;
            fprintf(stderr, "  %-34s %8.3f ms  (%d calls/step)\n", p.second.c_str(), p.first / steps, pacc[p.second].second / steps);
        }
        fprintf(stderr, "  TOTAL %.3f ms\n", tot / steps);
        pacc.clear();
    }
    void lin(const std::string& name, const __half* X, float* Y, int M) {
        const TInfo& t = ti(name);
        size_t dot = name.rfind('.', name.rfind('.') - 1);
        PS _p(this, std::string("lin ") + name.substr(name.find('.', 13) == std::string::npos ? 0 : name.find('.', 13) + 1));
        (void)dot;
        linear_h(st, t.fmt, tw(name), (int)t.rows(), (int)t.cols(), X, Y, M);
    }

    // ---- HY4-TUNE decode path (one token / T <= 4). HY4_TUNE = bitmask (default TUNE_DEFAULT), read once; every bit is an exact restructuring (same numbers bit for bit):
    //   2 TN_TOPK     router top-16 by rank (k_router_topk_r) instead of 16 sequential block-wide argmax rounds
    //  64 TN_NOF2H    no fp32 -> fp16 passes around moe_experts (the expert kernels read the fp32 latent z, the reduce also writes the fp16 result)
    // 256 TN_ROUTE    one token: the router kernel copies the 16 chosen experts' records, the one-token expert kernels (2 launches) replace route + gu + dn + reduce
    // 512 TN_ENG      engram layers, one token: the 5 projections in one launch
    enum { TN_TOPK = 2, TN_NOF2H = 64, TN_ROUTE = 256, TN_ENG = 512 };
    int tune = 0;
    bool route_now = false;
    ExpertDev* slot_rec = nullptr;  // [16] ExpertDev records of the chosen experts (one-token decode path)
    LinMulti lm_eng[4];             // per engram module: key_projs 0..3 + value_proj for one token
    void tune_init() {
        const char* e = getenv("HY4_TUNE");
        tune = e ? atoi(e) : TUNE_DEFAULT;
        if (tune == 0) return;
        slot_rec = dalloc<ExpertDev>(16);
        HCK(hipMemset(moe_ws, 0, moe_ws_bytes(Tc)));  // the one-token expert kernels keep their column tickets in the (otherwise unused) `fill` table of the workspace
        if (engram_ready && emeta.layers.size() >= 4) {
            for (int mo = 0; mo < 4; ++mo) {
                const std::string p = L(emeta.layers[mo]) + ".engram_layer";
                const void* W[5];
                float* Y[5];
                int N[5];
                bool ok = true;
                int fmt = -1, K = 0;
                for (int j = 0; j < 5; ++j) {
                    const std::string nm = j < 4 ? p + ".key_projs." + std::to_string(j) + ".weight" : p + ".value_proj.weight";
                    const TInfo& t = ti(nm);
                    if (j == 0) { fmt = t.fmt; K = (int)t.cols(); }
                    ok = ok && t.fmt == fmt && (int)t.cols() == K && t.fmt != FMT_Q8ROW;
                    W[j] = tw(nm);
                    Y[j] = kv_f + (size_t)j * HID;
                    N[j] = (int)t.rows();
                }
                if (ok) lm_eng[mo] = linear_multi_build(fmt, K, 5, W, Y, N);
            }
        }
        fprintf(stderr, "[tune] HY4_TUNE=%d\n", tune);
    }

    bool init(const std::string& engram_path, std::string& err) {
        HCK(hipStreamCreate(&st));
        int T = Tc;
        d_ids = dalloc<int32_t>(T);
        x = dalloc<float>((size_t)T * 4 * HID);
        pre = dalloc<float>(T * 4);
        post = dalloc<float>(T * 4);
        cur_h = dalloc<__half>((size_t)T * HID);
        cur_f = dalloc<float>((size_t)T * HID);
        sub = dalloc<float>((size_t)T * HID);
        size_t big = (size_t)T * 16384;
        F1 = dalloc<float>(big);
        F2 = dalloc<float>(big);
        F3 = dalloc<float>(big);
        H1 = dalloc<__half>(big);
        qr_h = dalloc<__half>((size_t)T * 1024);
        qa_h = dalloc<__half>((size_t)T * 64 * 576);
        o_h = dalloc<__half>((size_t)T * 16384);
        ctx_ws = dalloc<float>((size_t)T * 64 * 512);
        z_f = dalloc<float>((size_t)T * 1536);
        z_h = dalloc<__half>((size_t)T * 1536);
        zout_f = dalloc<float>((size_t)T * 1536);
        moe_idx = dalloc<int32_t>((size_t)T * 16);
        moe_w = dalloc<float>((size_t)T * 16);
        moe_ws = dalloc<uint8_t>(moe_ws_bytes(T));
        hfin = dalloc<__half>((size_t)T * HID);
        const int logits_rows = want_all_logits ? T : 16;
        logits = dalloc<float>((size_t)logits_rows * c.vocab);
        d_argmax = dalloc<int32_t>(T);
        mla_ws_bytes = 0;
        mla_ws = nullptr;
        // experts table
        {
            std::vector<ExpertDev> h(41 * 512);
            for (int i = 0; i < 41 * 512; ++i) {
                const ExpertRef& e = g.experts[i];
                h[i].body_off = e.body_off;
                h[i].dn_off = e.dn_off;
                h[i].gu_code = e.gu_code;
                h[i].dn_code = e.dn_code;
                h[i].pad = (e.tier == 2 || e.tier == 3) ? 1 : 0;  // CPU-computed experts (resident or cold) are skipped by the GPU routing
            }
            d_experts = dalloc<ExpertDev>(41 * 512);
            HCK(hipMemcpy(d_experts, h.data(), h.size() * sizeof(ExpertDev), hipMemcpyHostToDevice));
        }
        if (!g.cpu_mask.empty()) {
            const size_t nz = (size_t)Tc * 1536;
            HCK(hipHostMalloc((void**)&h_z, nz * 4, hipHostMallocDefault));
            HCK(hipHostMalloc((void**)&h_zcpu, nz * 4, hipHostMallocDefault));
            HCK(hipHostMalloc((void**)&h_w, (size_t)Tc * 16 * 4, hipHostMallocDefault));
            HCK(hipHostMalloc((void**)&h_idx, (size_t)Tc * 16 * 4, hipHostMallocDefault));
            d_zcpu = dalloc<float>(nz);
            HCK(hipEventCreateWithFlags(&ev_hyb, hipEventDisableTiming));
            int nth = getenv("HY4_CPU_THREADS") ? atoi(getenv("HY4_CPU_THREADS")) : std::max(1, (int)std::thread::hardware_concurrency() / 2);
            if (!cpu.init(g.path, nth, &g.cpu_mask, err)) return false;
            if (!g.cold_mask.empty()) {
#if HY4_COLD_API
                if (!cpu.set_cold_experts(g.cold_mask, err)) return false;
#else
                err = "cold experts need hy4_cpu_experts with set_cold_experts";
                return false;
#endif
                has_cold = true;
                cold_flag.assign(41 * 512, 0);
                for (int l = 0; l < 41; ++l)
                    for (int e = 0; e < 512; ++e) cold_flag[l * 512 + e] = g.cold_mask[l][e] ? 1 : 0;
                h_use.assign((size_t)Tc * 16, 0);
            }
            if (!getenv("STRATA_CPUEXP_ACT")) cpu.set_hi_activations(true);  // two-term int8 activations: ~GPU-exact quality (KL 1e-3 vs 7e-3) for ~1% decode cost
            skd::moe_has_cpu() = 1;
            hyb = true;
            fprintf(stderr, "[hybrid] %llu experts (%.2f GB) in host RAM computed by %d CPU threads\n", (unsigned long long)cpu.resident_count(), cpu.arena_bytes() / 1e9, cpu.n_threads());
        }
        // kv caches
        ring = ((512 + Tc + 15) / 16) * 16;
        slots_sparse = ((max_ctx + 8 + 15) / 16) * 16;
        const int devpos = getenv("HY4_KV_DEVPOS") ? atoi(getenv("HY4_KV_DEVPOS")) : 0;
        if (devpos > 0 && devpos < max_ctx) {
            kv_cap = devpos;
            slots_sparse = ((kv_cap + 8 + 15) / 16) * 16;
        }
        kcache.resize(c.n_layers);
        size_t kvbytes = 0;
        for (int il = 0; il < c.n_layers; ++il) {
            int slots = c.is_swa[il] ? ring : slots_sparse;
            size_t by = mla_cache_bytes(slots);
            if (c.is_swa[il]) {
                HCK(hipMalloc((void**)&kcache[il], by));
                HCK(hipMemset(kcache[il], 0, by));
            } else if (kv_cap) {
                HCK(hipMalloc((void**)&kcache[il], by));
                HCK(hipMemset(kcache[il], 0, by));
            } else {
                kcache[il] = (__half*)dalloc_big<uint8_t>(by, "sparse KV cache", 1);
            }
            kvbytes += by;
        }
        // indexer state (sparse layers only)
        ivg.assign(c.n_layers, nullptr);
        pooled.assign(c.n_layers, nullptr);
        for (int il = 0; il < c.n_layers; ++il)
            if (!c.is_swa[il]) {
                if (kv_cap) {
                    ivg[il] = dalloc<__half>((size_t)(kv_cap + 8) * 256);
                    pooled[il] = dalloc<__half>((size_t)(kv_cap / 4 + 8) * 128);
                    HCK(hipMemset(ivg[il], 0, (size_t)(kv_cap + 8) * 256 * 2));
                    HCK(hipMemset(pooled[il], 0, (size_t)(kv_cap / 4 + 8) * 128 * 2));
                } else {
                    ivg[il] = dalloc_big<__half>((size_t)(max_ctx + 8) * 256, "IndexPool iv/ig cache", 4);
                    pooled[il] = dalloc_big<__half>((size_t)(max_ctx / 4 + 8) * 128, "IndexPool pooled keys", 4);
                }
            }
        ik_f = dalloc<float>((size_t)Tc * 128);
        iv_f = dalloc<float>((size_t)Tc * 128);
        ig_f = dalloc<float>((size_t)Tc * 128);
        ik_h = dalloc<__half>((size_t)Tc * 128);
        iq_f = dalloc<float>((size_t)Tc * 2048);
        iq_h = dalloc<__half>((size_t)Tc * 2048);
        hw_f = dalloc<float>((size_t)Tc * 16);
        ldS = (size_t)(max_ctx / 4 + 8);
        score = dalloc<float>((size_t)Tc * ldS);
        keys = dalloc<int32_t>((size_t)Tc * (index_topk + 8));
        nkeys = dalloc<int32_t>((size_t)Tc);
        // single-layer absorbed kv_b scratch (dequantized on demand during prefill; saves ~1.02 GB vs 42-layer persistent copy)
        const int max_nh = std::max(c.heads_swa, c.heads_sparse);
        wk_s = dalloc_big<__half>((size_t)max_nh * 192 * 512, "kv_b scratch wk", 2);
        wk_st = dalloc_big<__half>((size_t)max_nh * 512 * 192, "kv_b scratch wk_t", 2);
        wv_s = dalloc_big<__half>((size_t)max_nh * 256 * 512, "kv_b scratch wv", 2);
        // engram
        if (!load_engram_meta_file(g.path, emeta, err)) return false;
        if (!engram_path.empty()) {
            // seed bound to the shard: pass the Section 2 bytes
            if (!eio.open(engram_path, g.seed.data(), g.seed.size(), ENGRAM_DEFAULT_LRU_ROWS, 16, 8, err)) return false;
            engram_ready = true;
        }
        for (int mo = 0; mo < 4; ++mo) {
            int il = emeta.layers[mo];
            std::string p = L(il) + ".engram_layer";
            EngramDev& e = eng_dev[mo];
            e.lnw = dalloc<float>(16 * 96);
            e.kvb = dalloc<float>(5 * HID);
            e.egate = dalloc<float>(4 * HID);
            e.pre_gamma = dalloc<float>(HID);
            e.hcs = dalloc<float>(4 * HID);
            for (int ch = 0; ch < 16; ++ch)
                HCK(hipMemcpy(e.lnw + ch * 96, tw(p + ".lookup_per_head_norms." + std::to_string(ch) + ".weight"), 96 * 4, hipMemcpyDeviceToDevice));
            for (int j = 0; j < 4; ++j) {
                HCK(hipMemcpy(e.kvb + j * HID, tw(p + ".key_projs." + std::to_string(j) + ".bias"), HID * 4, hipMemcpyDeviceToDevice));
                HCK(hipMemcpy(e.hcs + j * HID, tw(p + ".unified_per_hc_scales." + std::to_string(j)), HID * 4, hipMemcpyDeviceToDevice));
                // egate = norm_key * norm_query: computed on host
                std::vector<float> nk(HID), nq(HID), gg(HID);
                HCK(hipMemcpy(nk.data(), tw(p + ".norm_key." + std::to_string(j) + ".weight"), HID * 4, hipMemcpyDeviceToHost));
                HCK(hipMemcpy(nq.data(), tw(p + ".norm_query." + std::to_string(j) + ".weight"), HID * 4, hipMemcpyDeviceToHost));
                for (int d = 0; d < HID; ++d) gg[d] = nk[d] * nq[d];
                HCK(hipMemcpy(e.egate + j * HID, gg.data(), HID * 4, hipMemcpyHostToDevice));
            }
            HCK(hipMemcpy(e.kvb + 4 * HID, tw(p + ".value_proj.bias"), HID * 4, hipMemcpyDeviceToDevice));
            HCK(hipMemcpy(e.pre_gamma, tw(p + ".unified_pre_gamma"), HID * 4, hipMemcpyDeviceToDevice));
            d_eraw[mo] = dalloc<uint8_t>((size_t)Tc * 16 * 100);
            d_evalid[mo] = dalloc<uint8_t>((size_t)Tc * 16);
        }
        er_h = dalloc<__half>((size_t)Tc * 1536);
        kv_f = dalloc<float>((size_t)5 * Tc * HID);
        ep_alloc();
        if (hyb && (!getenv("HY4_PF_STREAM") || atoi(getenv("HY4_PF_STREAM")) != 0)) pf_init();  // last: the scratch takes what the other buffers left
        tune_init();
        size_t fr = 0, tot = 0;
        HCK(hipMemGetInfo(&fr, &tot));
        fprintf(stderr, "[runner] kv cache %.2f GB (ctx %d, chunk %d), device free %.1f GiB\n", kvbytes / 1e9, max_ctx, Tc, fr / 1073741824.0);
        return true;
    }

    // Engram rows of one chunk for all 4 modules (CPU hash + page reads), uploaded to the device
    struct EPrep {
        uint8_t *raw = nullptr, *valid = nullptr;  // pinned host memory: [4][Tc*16*100], [4][Tc*16] (real async uploads)
        int p0 = -1, T = 0;
        bool ok = false, pending = false;
        std::string err;
        std::future<void> fut;
        hipEvent_t used = nullptr;  // recorded after the uploads of this slot were enqueued
        bool staged = false, ok_rest = true;  // decode: module 0 is published through pm0 / fm0 as soon as it is loaded, modules 1..3 follow (fut)
        std::promise<void> pm0;
        std::future<void> fm0;
    };
    EPrep* ep_rest = nullptr;
    EPrep ep[2];
    int ep_next = 0;
    double t_engram_wait = 0, t_eng_issue = 0, t_eng_prep = 0;
    int n_eng_dec = 0;

    void ep_alloc() {
        for (int s = 0; s < 2; ++s) {
            HCK(hipHostMalloc((void**)&ep[s].raw, (size_t)4 * Tc * 16 * 100, hipHostMallocDefault));
            HCK(hipHostMalloc((void**)&ep[s].valid, (size_t)4 * Tc * 16, hipHostMallocDefault));
            HCK(hipEventCreateWithFlags(&ep[s].used, hipEventDisableTiming));
        }
    }
    void ep_compute(EPrep* e, const int32_t* tok_all, int p0, int T) {
        std::string err;
        e->ok = true;
        if (T <= 8 && e->staged) {  // decode / verify window: module 0 first (published through pm0), then modules 1..3 in one batch
            const size_t per = (size_t)T * 16;
            auto fetch_modules = [&](int m0, int m1) -> bool {
                std::vector<uint64_t> ids((size_t)(m1 - m0) * per);
                for (int mo = m0; mo < m1; ++mo) {
                    engram_hash(emeta, mo, tok_all, p0, p0 + T, ids.data() + (size_t)(mo - m0) * per, e->valid + (size_t)mo * Tc * 16);
                    if (vis) vis->mask_engram(e->valid + (size_t)mo * Tc * 16, p0, T);  // HY4-VISION hook (merge: the tuned decode path needs it too)
                }
                std::vector<uint8_t> tmp(ids.size() * 100);
                if (!eio.fetch_raw(ids.data(), ids.size(), tmp.data(), err)) return false;
                for (int mo = m0; mo < m1; ++mo) std::memcpy(e->raw + (size_t)mo * Tc * 16 * 100, tmp.data() + (size_t)(mo - m0) * per * 100, per * 100);
                return true;
            };
            if (!fetch_modules(0, 1)) {
                e->ok = false;
                e->err = err;
                e->ok_rest = false;
                e->pm0.set_value();
                return;
            }
            e->pm0.set_value();
            if (!fetch_modules(1, 4)) {
                e->ok_rest = false;
                e->err = err;
            }
            return;
        }
        if (T <= 8) {  // sync decode / verify window without a prefetch: the rows of all 4 modules in ONE fetch
            const size_t per = (size_t)T * 16;
            std::vector<uint64_t> ids(4 * per);
            for (int mo = 0; mo < 4; ++mo) {
                engram_hash(emeta, mo, tok_all, p0, p0 + T, ids.data() + mo * per, e->valid + (size_t)mo * Tc * 16);
                if (vis) vis->mask_engram(e->valid + (size_t)mo * Tc * 16, p0, T);  // HY4-VISION hook (merge: the tuned decode path needs it too)
            }
            std::vector<uint8_t> tmp(4 * per * 100);
            if (!eio.fetch_raw(ids.data(), ids.size(), tmp.data(), err)) {
                e->ok = false;
                e->err = err;
                return;
            }
            for (int mo = 0; mo < 4; ++mo) std::memcpy(e->raw + (size_t)mo * Tc * 16 * 100, tmp.data() + mo * per * 100, per * 100);
            return;
        }
        for (int mo = 0; mo < 4; ++mo) {
            std::vector<uint64_t> ids((size_t)T * 16);
            uint8_t* valid = e->valid + (size_t)mo * Tc * 16;
            engram_hash(emeta, mo, tok_all, p0, p0 + T, ids.data(), valid);
            if (vis) vis->mask_engram(valid, p0, T);  // HY4-VISION hook: no Engram contribution at image positions
            uint8_t* raw = e->raw + (size_t)mo * Tc * 16 * 100;
            if (!eio.fetch_raw(ids.data(), ids.size(), raw, err)) {
                e->ok = false;
                e->err = err;
                return;
            }
        }
    }
    // start fetching the rows of chunk [p0, p0+T) in the background (tokens must be final and stay valid)
    void prefetch_engram(const int32_t* tok_all, int p0, int T) {
        if (!engram_ready) return;
        EPrep& e = ep[ep_next];
        if (e.fut.valid()) e.fut.wait();
        HCK(hipEventSynchronize(e.used));
        e.p0 = p0;
        e.T = T;
        e.pending = true;
        e.staged = T <= 8;
        e.ok_rest = true;
        if (e.staged) {
            e.pm0 = std::promise<void>();
            e.fm0 = e.pm0.get_future();
        }
        e.fut = std::async(std::launch::async, [this, &e, tok_all, p0, T]() { ep_compute(&e, tok_all, p0, T); });
        ep_next ^= 1;
    }
    bool engram_prep(const int32_t* tok_all, int p0, int T, std::string& err) {
        double t0 = now_s();
        EPrep* e = nullptr;
        for (int s = 0; s < 2; ++s)
            if (ep[s].pending && ep[s].p0 == p0 && ep[s].T == T) e = &ep[s];
        if (e && e->staged) {
            // decode: only module 0 (needed at layer 1) is awaited here; modules 1..3 keep loading while the GPU runs layers 1..10 (engram_prep_rest, before module 1's layer)
            e->fm0.get();
            if (!e->ok) {
                err = e->err;
                return false;
            }
            HCK(hipMemcpyAsync(d_eraw[0], e->raw, (size_t)T * 16 * 100, hipMemcpyHostToDevice, st));
            HCK(hipMemcpyAsync(d_evalid[0], e->valid, (size_t)T * 16, hipMemcpyHostToDevice, st));
            HCK(hipEventRecord(e->used, st));
            ep_rest = e;
            t_engram_cpu += now_s() - t0;
            return true;
        }
        if (e) {
            e->fut.get();
            e->pending = false;
        } else {
            // no prefetch: wait for any running one, then do it synchronously in a free slot
            for (int s = 0; s < 2; ++s)
                if (ep[s].fut.valid()) ep[s].fut.wait();
            e = &ep[ep_next];
            HCK(hipEventSynchronize(e->used));
            ep_next ^= 1;
            e->staged = false;  // synchronous fetch: no module-0-first publishing
            ep_compute(e, tok_all, p0, T);
        }
        if (!e->ok) {
            err = e->err;
            return false;
        }
        for (int mo = 0; mo < 4; ++mo) {
            HCK(hipMemcpyAsync(d_eraw[mo], e->raw + (size_t)mo * Tc * 16 * 100, (size_t)T * 16 * 100, hipMemcpyHostToDevice, st));
            HCK(hipMemcpyAsync(d_evalid[mo], e->valid + (size_t)mo * Tc * 16, (size_t)T * 16, hipMemcpyHostToDevice, st));
        }
        HCK(hipEventRecord(e->used, st));
        t_engram_cpu += now_s() - t0;
        return true;
    }

    // second half of a staged decode fetch: wait for modules 1..3 and upload them
    bool engram_prep_rest(std::string& err) {
        if (!ep_rest) return true;
        EPrep* e = ep_rest;
        ep_rest = nullptr;
        e->fut.get();
        e->pending = false;
        if (!e->ok_rest) {
            err = e->err;
            return false;
        }
        for (int mo = 1; mo < 4; ++mo) {
            HCK(hipMemcpyAsync(d_eraw[mo], e->raw + (size_t)mo * Tc * 16 * 100, (size_t)e->T * 16 * 100, hipMemcpyHostToDevice, st));
            HCK(hipMemcpyAsync(d_evalid[mo], e->valid + (size_t)mo * Tc * 16, (size_t)e->T * 16, hipMemcpyHostToDevice, st));
        }
        HCK(hipEventRecord(e->used, st));
        return true;
    }

    void maybe_dump(const std::string& name, int T, int D, const float* dev, int per_tok_streams = 0) {
        if (!dump) return;
        HCK(hipStreamSynchronize(st));
        size_t n = (size_t)T * D;
        std::vector<float> h(n);
        HCK(hipMemcpy(h.data(), dev, n * 4, hipMemcpyDeviceToHost));
        if (per_tok_streams) save_npy_f32(dump_dir + "/" + name + ".npy", h.data(), {T, per_tok_streams, D / per_tok_streams});
        else save_npy_f32(dump_dir + "/" + name + ".npy", h.data(), {T, D});
    }

    void engram_layer(int mo, int il, int T) {
        const std::string p = L(il) + ".engram_layer";
        EngramDev& e = eng_dev[mo];
        hipLaunchKernelGGL(k_engram_rows, dim3(T), dim3(512), 0, st, d_eraw[mo], e.lnw, d_evalid[mo], er_h);
        if (T == 1 && (tune & TN_ENG) && lm_eng[mo].d_jobs) {  // the 5 projections of one token in ONE launch (outputs at kv_f + j * HID, as below)
            linear_h_multi(st, lm_eng[mo], er_h, 1);
        } else
        for (int j = 0; j < 5; ++j) {
            std::string nm = j < 4 ? p + ".key_projs." + std::to_string(j) + ".weight" : p + ".value_proj.weight";
            const TInfo& t = ti(nm);
            linear_h(st, t.fmt, tw(nm), (int)t.rows(), (int)t.cols(), er_h, kv_f + (size_t)j * T * HID, T);
        }
        hipLaunchKernelGGL(k_engram_apply, dim3(T, 4), dim3(256), 0, st, x, kv_f, e.kvb, e.egate, e.pre_gamma, e.hcs, d_evalid[mo], T);
    }

    float* hcacc = nullptr;
    bool det_mix = true;  // deterministic iHC mix (no float atomics): same answer every run; also set by the speculative decoder
    float* hcpart = nullptr;
    void hc_mix_run(const float* xin, const float* hfn, const float* hsc, const float* hbs, int nout, int T) {
        if (det_mix) {
            if (!hcpart) hcpart = dalloc<float>((size_t)Tc * 24 * 9);
            hipLaunchKernelGGL(k_hc_mix_part_d, dim3(T, 24), dim3(128), 0, st, xin, hfn, hcpart, nout);
            hipLaunchKernelGGL(k_hc_mix_fin_d, dim3(T), dim3(32), 0, st, hcpart, hsc, hbs, pre, post, nout, c.rms_eps, c.hc_eps, c.hc_mag);
            return;
        }
        if (!hcacc) hcacc = dalloc<float>((size_t)Tc * 9);
        hipLaunchKernelGGL(k_zero_f, dim3((T * 9 + 255) / 256), dim3(256), 0, st, hcacc, T * 9);
        hipLaunchKernelGGL(k_hc_mix_part, dim3(T, 24), dim3(128), 0, st, xin, hfn, hcacc, nout);
        hipLaunchKernelGGL(k_hc_mix_fin, dim3(T), dim3(32), 0, st, hcacc, hsc, hbs, pre, post, nout, c.rms_eps, c.hc_eps, c.hc_mag);
    }

    // iHC pre-mix + RMSNorm(nw) of cur = sum_h pre_h x_h  ->  outh (fp16) / outf (fp32, optional); post (4 per token) is produced for nout == 8
    void hc_pre(const float* xin, const float* hfn, const float* hsc, const float* hbs, const float* nw, int nout, int T, __half* outh, float* outf) {
        hc_mix_run(xin, hfn, hsc, hbs, nout, T);
        hipLaunchKernelGGL(k_hc_reduce_norm, dim3(T), dim3(256), 0, st, xin, pre, nw, outh, outf, c.rms_eps);
    }

    // IndexPool: cache iv/ig of the chunk's tokens and pool the groups completed by this chunk
    void idx_update(int il, int p0, int T) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        lin(p + ".token_key_projection.weight", cur_h, ik_f, T);  // [T][128]
        hipLaunchKernelGGL(k_rms_rows, dim3(T), dim3(128), 0, st, ik_f, (const float*)tw(p + ".token_key_norm.weight"), (float*)nullptr, ik_h, 128, c.rms_eps);
        lin(p + ".key_value_projection.weight", ik_h, iv_f, T);
        lin(p + ".key_gate_projection.weight", ik_h, ig_f, T);
        hipLaunchKernelGGL(idx::k_idx_store, dim3(T), dim3(128), 0, st, iv_f, ig_f, ivg[il], p0);
        int g_lo = p0 / 4, g_hi = (p0 + T) / 4 - 1;
        if (g_hi >= g_lo)
            hipLaunchKernelGGL(idx::k_idx_pool, dim3(g_hi - g_lo + 1), dim3(128), 0, st, ivg[il], (const float*)tw(p + ".intra_group_ape"), (const float*)tw(p + ".key_norm.weight"), pooled[il], g_lo,
                               (float)theta(il), c.rms_eps);
        if (vis && g_hi >= g_lo && vis->needs_fix(p0, T)) vis->pool_fix(st, pooled[il], g_lo, g_hi - g_lo + 1, (float)theta(il));  // HY4-VISION hook
    }

    // per-token key lists for the chunk (sparse layer, sequence longer than index_topk)
    void idx_select(int il, int p0, int T) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        lin(p + ".query_projection.weight", qr_h, iq_f, T);  // [T][16*128]
        if (vis && vis->needs_fix(p0, T)) vis->iq_rope(st, iq_f, iq_h, T, p0, (float)theta(il));  // HY4-VISION hook: 3D positions
        else hipLaunchKernelGGL(idx::k_idx_q_rope, dim3(T, 16), dim3(128), 0, st, iq_f, iq_h, p0, (float)theta(il));
        lin(p + ".head_weight_projection.weight", cur_h, hw_f, T);  // [T][16]
        int G = (p0 + T) / 4;
        if (G < 1) G = 1;
        hipLaunchKernelGGL(idx::k_idx_score, dim3((G + 63) / 64, T), dim3(64), 0, st, iq_h, hw_f, pooled[il], score, p0, G, (int)ldS, 1.0f / sqrtf((float)(128 * 16)));
        int n_top = index_topk / c.kpool - 1;
        hipLaunchKernelGGL(idx::k_idx_select, dim3(T), dim3(256), 0, st, score, (int)ldS, p0, n_top, index_topk, keys, nkeys);
    }

    void attention_layer(int il, int p0, int T) {
        const std::string pa = L(il) + ".self_attn";
        const int NH = heads(il);
        const bool swa = c.is_swa[il];
        // iHC pre + input_layernorm
        const float* hfn = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_base");
        { PS _q(this, "hc_mix"); hc_pre(x, hfn, hsc, hbs, (const float*)tw(L(il) + ".input_layernorm.weight"), 8, T, cur_h, (float*)nullptr); }
        // q path
        lin(pa + ".q_a_proj.weight", cur_h, F1, T);  // [T][1024]
        { PS _q(this, "q_a_rms"); hipLaunchKernelGGL(k_rms_rows, dim3(T), dim3(256), 0, st, F1, (const float*)tw(pa + ".q_a_layernorm.weight"), (float*)nullptr, qr_h, c.q_lora, c.rms_eps); }
        lin(pa + ".q_b_proj.weight", qr_h, F2, T);  // [T][NH*256]
        lin(pa + ".kv_a_proj_with_mqa.weight", cur_h, F1, T);  // [T][576]  (qr_h already extracted)
        __half* cache = kcache[il];
        const int cslots = swa ? ring : slots_sparse;
        { PS _q(this, "mla_write_kv"); mla_write_kv(st, F1, (const float*)tw(pa + ".kv_a_layernorm.weight"), cache, cslots, T, p0, swa ? ring : 0, (float)theta(il), c.rms_eps); }
        const bool dec = T <= 8;
        const int kvfmt = ti(pa + ".kv_b_proj.weight").fmt;
        if (!dec) {
            PS _q(this, "split_kvb");
            const uint8_t* wkv = (const uint8_t*)tw(pa + ".kv_b_proj.weight");
            dim3 grid(NH * 448), block(128);
            switch (kvfmt) {
                case FMT_S2: hipLaunchKernelGGL((k_split_kvb<FMT_S2>), grid, block, 0, st, wkv, wk_s, wv_s, NH); break;
                case FMT_S3: hipLaunchKernelGGL((k_split_kvb<FMT_S3>), grid, block, 0, st, wkv, wk_s, wv_s, NH); break;
                case FMT_S4: hipLaunchKernelGGL((k_split_kvb<FMT_S4>), grid, block, 0, st, wkv, wk_s, wv_s, NH); break;
                case FMT_S5: hipLaunchKernelGGL((k_split_kvb<FMT_S5>), grid, block, 0, st, wkv, wk_s, wv_s, NH); break;
                case FMT_S6: hipLaunchKernelGGL((k_split_kvb<FMT_S6>), grid, block, 0, st, wkv, wk_s, wv_s, NH); break;
                case FMT_S8: hipLaunchKernelGGL((k_split_kvb<FMT_S8>), grid, block, 0, st, wkv, wk_s, wv_s, NH); break;
                default: fprintf(stderr, "unsupported kv_b_proj fmt %d\n", kvfmt); exit(3);
            }
            mla_transpose_wk(st, wk_s, wk_st, NH);
        }
        { PS _q(this, "mla_prep_q"); if (dec) mla_prep_q_dec(st, kvfmt, tw(pa + ".kv_b_proj.weight"), F2, qa_h, T, NH, p0, (float)theta(il));
        else mla_prep_q_t(st, F2, wk_st, qa_h, T, NH, p0, (float)theta(il)); }
        if (vis && vis->needs_fix(p0, T)) {  // HY4-VISION hook: rope of q and of the new keys with the picture-aware (t, h, w) positions
            vis->fix_k(st, F1, cache, T, p0, swa ? ring : 0, (float)theta(il));
            vis->fix_q(st, F2, qa_h, T, NH, p0, (float)theta(il));
        }
        bool use_list = false;
        if (!swa && index_on) {
            { PS _q(this, "idx_update"); idx_update(il, p0, T); }
            if (p0 + T > index_topk) {
                { PS _q(this, "idx_select"); idx_select(il, p0, T); }
                use_list = true;
            }
        }
        last_use_list = use_list;
        if (dfly_cap && !swa) dfly_capture(il, p0, T, use_list);
        lin(pa + ".linear_gate.weight", cur_h, F3, T);  // [T][NH*256]
        MlaAttnArgs a;
        a.M = T;
        a.NH = NH;
        a.p0 = p0;
        a.mode = swa ? MLA_WINDOW : (use_list ? MLA_LIST : MLA_CAUSAL);
        a.window = c.window;
        a.ring = swa ? ring : 0;
        a.cache_slots = cslots;
        if (use_list) {
            a.keys = keys;
            a.nkeys = nkeys;
            a.kmax = index_topk;
        }
        // size the split-K workspace AFTER kmax is set: mla_workspace_bytes() takes max_keys = kmax in list mode. (Before: kmax was still 0 here, no workspace was ensured for the
        // IndexPool layers, and mla_attend fell back to the simple kernel unless another call had happened to grow the buffer: which kernel decode used depended on earlier GPU work.)
        ensure_mla_ws(mla_workspace_bytes(a));
        if (dec) {
            { PS _q(this, "mla_attend"); mla_attend(st, qa_h, cache, (const float*)tw(pa + ".learnable_sink_param"), (__half*)ctx_ws, a, mla_ws, mla_ws_bytes); }
            { PS _q(this, "mla_vup"); mla_vup_dec(st, kvfmt, tw(pa + ".kv_b_proj.weight"), (const __half*)ctx_ws, F3, o_h, T, NH); }
            if (dump && il == dump_layer) {
                dump_half("in_cur_h", cur_h, (size_t)T * HID, T);
                maybe_dump("in_qb", T, NH * 256, F2);
                dump_half("in_qa", qa_h, (size_t)T * NH * 576, T);
                dump_half("in_ctx", (const __half*)ctx_ws, (size_t)T * NH * 512, T);
                dump_half("in_o", o_h, (size_t)T * NH * 256, T);
                maybe_dump("in_gate", T, NH * 256, F3);
            }
        } else {
            { PS _q(this, "mla_attend_vup_prefill"); mla_attention(st, qa_h, cache, (const float*)tw(pa + ".learnable_sink_param"), wv_s, F3, o_h, ctx_ws, a, mla_ws, mla_ws_bytes); }
        }
        lin(pa + ".o_proj.weight", o_h, F1, T);  // [T][3072]
        if (dump && il == dump_layer) maybe_dump("in_oproj", T, HID, F1);
        { PS _q(this, "o_norm_post"); hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F1, (const float*)nullptr, (const float*)tw(pa + ".post_proj_norm.weight"), x, post, c.rms_eps); }
    }

    // MoE after the router and the latent down projection (z_f = [T][1536] fp32, moe_idx / moe_w on the device) for a layer whose experts are split between the GPU and the CPU.
    // Order: copy z / idx / w to the host, queue everything the GPU can do without the CPU experts (VRAM experts, shared experts), then run the CPU experts while the GPU works,
    // add their partial sum into zout_f and finish (up projection, combine).
    void moe_hybrid_tail(int il, int T, const std::string& pm) {
        const size_t nz = (size_t)T * 1536;
        if (pf_active(T)) {
            // large chunk: every expert (the CPU tier arrives through the scratch buffers, group by group) runs on the GPU
            if (has_cold) {  // the cold experts (SSD) are computed by the CPU pool while the GPU runs the streamed groups
                HCK(hipMemcpyAsync(h_z, z_f, nz * 4, hipMemcpyDeviceToHost, st));
                HCK(hipMemcpyAsync(h_idx, moe_idx, (size_t)T * 16 * 4, hipMemcpyDeviceToHost, st));
                HCK(hipMemcpyAsync(h_w, moe_w, (size_t)T * 16 * 4, hipMemcpyDeviceToHost, st));
                HCK(hipEventRecord(ev_hyb, st));
            }
            { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, z_f, z_h, nz); }
            const std::vector<PfGroup>& gs = pf_groups[il - 1];
            for (size_t gi = 0; gi < gs.size(); ++gi) {
                const PfGroup& G = gs[gi];
                const int k = G.seq, b = k % pf_nbuf;
                pf_pump(k);
                if (!G.runs.empty()) HCK(hipStreamWaitEvent(st, ev_pf_copy[b], 0));
                { PS _q(this, "moe_experts(pf)"); moe_experts(st, g.d_sec4, G.table, z_h, moe_idx, moe_w, gi == 0 ? zout_f : d_zcpu, T, moe_ws); }
                if (gi > 0) hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)d_zcpu, nz);
                HCK(hipEventRecord(ev_pf_done[b], st));
                ev_pf_done_set[b] = true;
                ++pf_next_comp;
                pf_pump(k + pf_nbuf);
            }
            if (has_cold) {
                double t0 = now_s();
                HCK(hipEventSynchronize(ev_hyb));
                for (size_t i = 0; i < (size_t)T * 16; ++i) h_use[i] = cold_flag[(size_t)(il - 1) * 512 + (size_t)h_idx[i]];
                cpu.submit(il - 1, T, h_z, h_idx, h_w, h_use.data(), h_zcpu);
                cpu.wait();
                t_cpu_wait += now_s() - t0;
                HCK(hipMemcpyAsync(d_zcpu, h_zcpu, nz * 4, hipMemcpyHostToDevice, st));
                hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)d_zcpu, nz);
            }
            { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, z_h, nz); }
            lin(pm + ".routed_expert_up_proj.weight", z_h, F2, T);  // [T][3072]
            lin(pm + ".shared_experts.gate_proj.weight", cur_h, F1, T);
            lin(pm + ".shared_experts.up_proj.weight", cur_h, F3, T);
            { PS _q(this, "swiglu"); hipLaunchKernelGGL(k_swiglu_h, dim3((nz + 255) / 256), dim3(256), 0, st, F1, F3, H1, nz); }
            lin(pm + ".shared_experts.down_proj.weight", H1, F3, T);
            { PS _q(this, "combine_norm_post"); hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F2, (const float*)F3, (const float*)tw(pm + ".combine_output_norm.weight"), x, post, c.rms_eps); }
            return;
        }
        HCK(hipMemcpyAsync(h_z, z_f, nz * 4, hipMemcpyDeviceToHost, st));
        HCK(hipMemcpyAsync(h_idx, moe_idx, (size_t)T * 16 * 4, hipMemcpyDeviceToHost, st));
        HCK(hipMemcpyAsync(h_w, moe_w, (size_t)T * 16 * 4, hipMemcpyDeviceToHost, st));
        HCK(hipEventRecord(ev_hyb, st));
        { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, z_f, z_h, nz); }
        { PS _q(this, "moe_experts(gpu part)"); moe_experts(st, g.d_sec4, d_experts + (size_t)(il - 1) * 512, z_h, moe_idx, moe_w, zout_f, T, moe_ws); }
        lin(pm + ".shared_experts.gate_proj.weight", cur_h, F1, T);  // [T][1536]
        lin(pm + ".shared_experts.up_proj.weight", cur_h, F3, T);
        { PS _q(this, "swiglu"); hipLaunchKernelGGL(k_swiglu_h, dim3((nz + 255) / 256), dim3(256), 0, st, F1, F3, H1, nz); }
        lin(pm + ".shared_experts.down_proj.weight", H1, F3, T);  // [T][3072]
        double t0 = now_s();
        HCK(hipEventSynchronize(ev_hyb));
        cpu.submit(il - 1, T, h_z, h_idx, h_w, nullptr, h_zcpu);
        cpu.wait();
        t_cpu_wait += now_s() - t0;
        HCK(hipMemcpyAsync(d_zcpu, h_zcpu, nz * 4, hipMemcpyHostToDevice, st));
        hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)d_zcpu, nz);
        { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, z_h, nz); }
        lin(pm + ".routed_expert_up_proj.weight", z_h, F2, T);  // [T][3072]
        { PS _q(this, "combine_norm_post"); hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F2, (const float*)F3, (const float*)tw(pm + ".combine_output_norm.weight"), x, post, c.rms_eps); }
    }

    void mlp_layer(int il, int T) {
        const std::string pm = L(il) + ".mlp";
        const float* hfn = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_base");
        { PS _q(this, "hc_mix"); hc_pre(x, hfn, hsc, hbs, (const float*)tw(L(il) + ".post_attention_layernorm.weight"), 8, T, cur_h, cur_f); }
        if (il == 0) {
            lin(pm + ".gate_proj.weight", cur_h, F1, T);  // [T][13824]
            lin(pm + ".up_proj.weight", cur_h, F2, T);
            size_t n = (size_t)T * 13824;
            { PS _q(this, "swiglu"); hipLaunchKernelGGL(k_swiglu_h, dim3((n + 255) / 256), dim3(256), 0, st, F1, F2, H1, n); }
            lin(pm + ".down_proj.weight", H1, F3, T);  // [T][3072]
            { PS _q(this, "o_norm_post"); hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F3, (const float*)nullptr, (const float*)tw(pm + ".post_fc2_norm.weight"), x, post, c.rms_eps); }
        } else {
            // router
            { PS _q(this, "router_gemv"); if (T <= 16) hipLaunchKernelGGL(k_f32_gemv, dim3(512), dim3(256), 0, st, (const float*)tw(pm + ".gate.weight"), (const float*)cur_f, F1, 512, HID, T); else hipLaunchKernelGGL(k_f32_gemm, dim3(8, (T + 63) / 64), dim3(256), 0, st, (const float*)tw(pm + ".gate.weight"), (const float*)cur_f, F1, 512, HID, T); }
            route_now = (tune & TN_ROUTE) && T == 1 && !hyb;
            {
                PS _q(this, "router_topk");
                const uint4* exl = route_now ? (const uint4*)(d_experts + (size_t)(il - 1) * 512) : nullptr;
                uint4* rec = route_now ? (uint4*)slot_rec : nullptr;
                const float* rb = (const float*)tw(pm + ".gate.e_score_correction_bias");
                if (tune & TN_TOPK) hipLaunchKernelGGL(k_router_topk_r, dim3(T), dim3(512), 0, st, F1, rb, moe_idx, moe_w, 16, c.routed_scale, exl, rec);
                else hipLaunchKernelGGL(k_router_topk, dim3(T), dim3(256), 0, st, F1, rb, moe_idx, moe_w, 512, 16, c.routed_scale, exl, rec);
            }
            lin(pm + ".routed_expert_down_proj.weight", cur_h, z_f, T);  // [T][1536]
            size_t nz = (size_t)T * 1536;
            if (hyb) { moe_hybrid_tail(il, T, pm); } else {
            const bool nof = T <= 4 && (tune & TN_NOF2H);  // fp32 latent read directly by the expert kernels, fp16 result written by their reduce
            if (nof) { PS _q(this, "moe_experts"); moe_experts(st, g.d_sec4, d_experts + (size_t)(il - 1) * 512, nullptr, moe_idx, moe_w, zout_f, T, moe_ws, z_f, z_h, route_now ? slot_rec : nullptr); }
            else {
            { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, z_f, z_h, nz); }
            { PS _q(this, "moe_experts"); moe_experts(st, g.d_sec4, d_experts + (size_t)(il - 1) * 512, z_h, moe_idx, moe_w, zout_f, T, moe_ws, nullptr, nullptr, route_now ? slot_rec : nullptr); }
            { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, z_h, nz); }
            }
            lin(pm + ".routed_expert_up_proj.weight", z_h, F2, T);  // [T][3072]
            lin(pm + ".shared_experts.gate_proj.weight", cur_h, F1, T);  // [T][1536]
            lin(pm + ".shared_experts.up_proj.weight", cur_h, F3, T);
            { PS _q(this, "swiglu"); hipLaunchKernelGGL(k_swiglu_h, dim3((nz + 255) / 256), dim3(256), 0, st, F1, F3, H1, nz); }
            lin(pm + ".shared_experts.down_proj.weight", H1, F3, T);  // [T][3072]
            { PS _q(this, "combine_norm_post"); hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F2, (const float*)F3, (const float*)tw(pm + ".combine_output_norm.weight"), x, post, c.rms_eps); }
            }
        }
    }

    // process tokens [p0, p0+T) (T <= Tc) of the sequence tok_all (host, positions 0..p0+T-1). Returns false on error.
    // the context passed the VRAM part of the KV state: move rows, Vt and the IndexPool state to pinned host memory (one time, ~0.1 s per GB)
    void kv_grow() {
        if (!kv_cap || kv_moved) return;
        const double t0 = now_s();
        HCK(hipDeviceSynchronize());
        const size_t os = (size_t)slots_sparse, ns = (size_t)((max_ctx + 8 + 15) / 16) * 16;
        const size_t row_b = (size_t)576 * 2;  // MLA_ROW halves per slot (no Vt copy)
        auto host_alloc = [&](size_t bytes, void** hp) {
            void* dp = nullptr;
            HCK(hipHostMalloc(hp, bytes, hipHostMallocMapped));
            memset(*hp, 0, bytes);
            HCK(hipHostGetDevicePointer(&dp, *hp, 0));
            host_big_bytes += bytes;
            return dp;
        };
        for (int il = 0; il < c.n_layers; ++il) {
            if (c.is_swa[il]) continue;
            void* hp = nullptr;
            uint8_t* nd = (uint8_t*)host_alloc(ns * row_b, &hp);
            uint8_t* od = (uint8_t*)kcache[il];
            HCK(hipMemcpy(nd, od, os * row_b, hipMemcpyDeviceToDevice));
            HCK(hipFree(od));
            kcache[il] = (__half*)nd;
            void* hp2 = nullptr;
            __half* niv = (__half*)host_alloc((size_t)(max_ctx + 8) * 256 * 2, &hp2);
            HCK(hipMemcpy(niv, ivg[il], (size_t)(kv_cap + 8) * 256 * 2, hipMemcpyDeviceToDevice));
            HCK(hipFree(ivg[il]));
            ivg[il] = niv;
            void* hp3 = nullptr;
            __half* npl = (__half*)host_alloc((size_t)(max_ctx / 4 + 8) * 128 * 2, &hp3);
            HCK(hipMemcpy(npl, pooled[il], (size_t)(kv_cap / 4 + 8) * 128 * 2, hipMemcpyDeviceToDevice));
            HCK(hipFree(pooled[il]));
            pooled[il] = npl;
        }
        slots_sparse = (int)ns;
        kv_moved = true;
        fprintf(stderr, "[runner] context passed %d positions: KV rows, Vt and IndexPool state moved to pinned host memory (%.2f GB, %.2f s); attention now reads them over PCIe\n", kv_cap, host_big_bytes / 1e9, now_s() - t0);
    }
    // make room for positions < need; false when need exceeds the context
    bool kv_ensure(int need) {
        if (need > max_ctx + 8) return false;
        if (kv_cap && !kv_moved && need > kv_cap) kv_grow();
        return need <= slots_sparse;
    }
    bool forward_chunk(const int32_t* tok_all, int p0, int T, std::string& err) {
        if (kv_cap && !kv_moved && p0 + T + 16 > kv_cap) kv_grow();
        HCK(hipMemcpyAsync(d_ids, tok_all + p0, (size_t)T * 4, hipMemcpyHostToDevice, st));
        hipLaunchKernelGGL(k_embed_hc, dim3(T), dim3(256), 0, st, (const uint8_t*)tw("model.embed_tokens.weight"), d_ids, x, T);
        if (vis) vis->embed(st, x, p0, T);  // HY4-VISION hook: image rows replace the embeddings of the image-pad tokens
        const bool defer_eng = engram_ready && T <= 8;
        if (pf_active(T)) pf_begin_chunk();
        if (defer_eng) {
            const double t_pf0 = now_s();
            prefetch_engram(tok_all, p0, T);
            t_eng_issue += now_s() - t_pf0;
        } else if (engram_ready && !engram_prep(tok_all, p0, T, err)) return false;
        for (int il = 0; il < c.n_layers; ++il) {
            if (defer_eng && il == 1) { double tq = now_s(); if (!engram_prep(tok_all, p0, T, err)) return false; t_eng_prep += now_s() - tq; ++n_eng_dec; }
            if (defer_eng && il == emeta.layers[1] && !engram_prep_rest(err)) return false;
            int mo = -1;
            for (int j = 0; j < 4; ++j)
                if (emeta.layers[j] == il) mo = j;
            double ta = 0;
            if (prof) { HCK(hipStreamSynchronize(st)); ta = now_s(); }
            if (mo >= 0 && engram_ready) {
                { PS _q(this, "engram_layer"); engram_layer(mo, il, T); }
                maybe_dump("x_L" + std::to_string(il) + "_engram", T, 4 * HID, x, 4);
            }
            if (prof) { HCK(hipStreamSynchronize(st)); double tb = now_s(); p_eng += tb - ta; ta = tb; }
            if (il >= 1 && pf_active(T)) pf_pump(pf_layer_first[il - 1] + pf_nbuf - 1);
            attention_layer(il, p0, T);
            maybe_dump("x_L" + std::to_string(il) + "_attn", T, 4 * HID, x, 4);
            if (prof) { HCK(hipStreamSynchronize(st)); double tb = now_s(); (c.is_swa[il] ? p_attn_swa : p_attn_sp) += tb - ta; ta = tb; }
            mlp_layer(il, T);
            maybe_dump("x_L" + std::to_string(il) + "_mlp", T, 4 * HID, x, 4);
            if (prof) { HCK(hipStreamSynchronize(st)); double tb = now_s(); (il == 0 ? p_mlp0 : p_moe) += tb - ta; }
        }
        return true;
    }
    bool prof = false;
    double p_eng = 0, p_attn_swa = 0, p_attn_sp = 0, p_mlp0 = 0, p_moe = 0;
    void prof_report(const char* what) {
        if (!prof) return;
        fprintf(stderr, "[prof %s] engram %.3f s | attn swa(30) %.3f s sparse(12) %.3f s | mlp0 %.3f s | moe(41) %.3f s\n", what, p_eng, p_attn_swa, p_attn_sp, p_mlp0, p_moe);
        p_eng = p_attn_swa = p_attn_sp = p_mlp0 = p_moe = 0;
    }

    // logits of chunk positions [first, T) into d logits [(T-first)][V]; h_final dumped if dump
    void head(int T, int first) {
        int n = T - first;
        const float* hfn = (const float*)tw("model.hc_head.hc_head_fn");
        const float* hsc = (const float*)tw("model.hc_head.hc_head_scale");
        const float* hbs = (const float*)tw("model.hc_head.hc_head_base");
        const float* xs = x + (size_t)first * 4 * HID;
        { PS _q(this, "head_hc_mix"); hc_pre(xs, hfn, hsc, hbs, (const float*)tw("model.norm.weight"), 4, n, hfin, (float*)nullptr); }
        const TInfo& lm = ti("lm_head.weight");
        linear_h(st, lm.fmt, tw("lm_head.weight"), (int)lm.rows(), (int)lm.cols(), hfin, logits, n);
    }
};

}  // namespace hy4
