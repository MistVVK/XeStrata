// hy4_graph.hip.h - the 42-layer hy_v4_flash forward on gfx1151 (HIP): chunked prefill + decode, KV fp16, absorbed MLA, LatentMoE.
// Built from: skgemm (linear_h / linear_f32), skmoe (moe_experts), mla_attn (mla_*), hy4_kernels (norms, iHC, router, engram), hy4_engram (hash + page reads).
#pragma once
#include "hy4_gpu.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
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

// device dequantization of one layer's kv_b_proj [NH*448][512] (any S_K fmt) into wk_bt [NH][512][192] (transposed for WMMA mla_prep_q_t) and wv_b [NH][256][512] fp16
template <int FMT>
__global__ void __launch_bounds__(256) k_dequant_wkt(const uint8_t* __restrict__ W, __half* __restrict__ wk_bt, int NH) {
    // grid (NH * 6, 16): block (h*6 + it, ct) transposes 32 rows (i0..i0+31) x 32 cols (c0..c0+31) via LDS
    __shared__ __half sh[32][33];
    const int h = blockIdx.x / 6, i0 = (blockIdx.x % 6) * 32, c0 = blockIdx.y * 32;
    const int tid = threadIdx.x, r0 = tid >> 5, col = tid & 31;
    constexpr int SB = sk_sb_bytes(FMT);
    const int sb = c0 >> 8, c_in = (c0 & 255) + col;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int ri = r0 + p * 8;
        const uint8_t* rp = W + ((size_t)h * 448 + i0 + ri) * (2 * SB) + (size_t)sb * SB;
        sh[ri][col] = __float2half(skd::sk_weight<FMT>(rp, c_in));
    }
    __syncthreads();
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int rc = r0 + p * 8;
        wk_bt[((size_t)h * 512 + c0 + rc) * 192 + i0 + col] = sh[col][rc];
    }
}
template <int FMT>
__global__ void __launch_bounds__(256) k_dequant_wv(const uint8_t* __restrict__ W, __half* __restrict__ wv, int NH) {
    // grid (NH * 256): one block per v row (512 cols = 2 per thread)
    const int r = blockIdx.x, h = r >> 8, j = r & 255;
    constexpr int SB = sk_sb_bytes(FMT);
    const uint8_t* rp = W + ((size_t)h * 448 + 192 + j) * (2 * SB);
    __half* dst = wv + ((size_t)h * 256 + j) * 512;
    for (int k = threadIdx.x; k < 512; k += 256) dst[k] = __float2half(skd::sk_weight<FMT>(rp + (size_t)(k >> 8) * SB, k & 255));
}
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

// Tree-8 speculative verification descriptor (M <= 8 nodes rooted at anchor node 0, topological order parent[i] < i).
struct TreeSpec {
    int M = 0;
    int32_t tok[8] = {};
    int8_t parent[8] = {};
    uint8_t depth[8] = {};
    uint8_t anc[8][8] = {};
    int max_depth() const {
        int md = 0;
        for (int i = 0; i < M; ++i) md = std::max(md, (int)depth[i]);
        return md;
    }
    void build_anc() {
        for (int i = 0; i < M; ++i) {
            int cur = i;
            for (int d = (int)depth[i]; d >= 0; --d) {
                anc[i][d] = (uint8_t)cur;
                cur = cur >= 0 ? (int)parent[cur] : -1;
            }
        }
    }
    TreePos to_mla_pos() const {
        TreePos tp;
        tp.n = M;
        for (int i = 0; i < M; ++i) tp.depth[i] = depth[i];
        return tp;
    }
    idx::IdxTree to_idx_tree() const {
        idx::IdxTree it;
        it.n = M;
        for (int i = 0; i < M; ++i) {
            it.depth[i] = depth[i];
            for (int d = 0; d < 8; ++d) it.anc[i][d] = anc[i][d];
        }
        return it;
    }
};

class Runner {
public:
    GpuModel& g;
    const Cfg& c;
    int max_ctx, Tc;
    hipStream_t st = nullptr;
    // buffers
    int32_t* d_ids = nullptr;
    float *x = nullptr, *pre = nullptr, *post = nullptr, *cur_f = nullptr, *F1 = nullptr, *F2 = nullptr, *F3 = nullptr, *F3_dec = nullptr, *sub = nullptr, *ctx_ws = nullptr;
    __half *cur_h = nullptr, *qr_h = nullptr, *qa_h = nullptr, *o_h = nullptr, *H1 = nullptr, *z_h = nullptr, *er_h = nullptr, *hfin = nullptr;
    float *z_f = nullptr, *zout_f = nullptr, *moe_w = nullptr, *logits = nullptr;
    int logits_cap = 0;
    int32_t *moe_idx = nullptr, *d_argmax = nullptr;
    void *work_arena = nullptr, *moe_ws = nullptr, *moe_ws_dec = nullptr, *mla_ws = nullptr;
    size_t mla_ws_bytes = 0;
    float* kv_f = nullptr;
    uint8_t* d_eraw[4] = {};
    uint8_t* d_evalid[4] = {};
    EngramDev eng_dev[4];
    ExpertDev* d_experts = nullptr;
    ExpertDev* h_experts = nullptr;    // pinned host mirror of d_experts [41*512] (non-blocking H2D for vcache updates)
    __half* wk_layer = nullptr;        // reusable single-layer [max_NH][192][512] (or [max_NH][512][192] on HIP) fp16 for prefill T > 8 (12.6 MiB)
    __half* wv_layer = nullptr;        // reusable single-layer [max_NH][256][512] fp16 for prefill T > 8 (16.8 MiB)
    std::vector<__half*> kcache;       // per layer
    int ring = 0, slots_sparse = 0;
    // Growable sparse KV + IndexPool state: starts at kv_alloc_pos positions (16,384 by default on discrete cards) in VRAM,
    // grows in 16K increments in VRAM up to kv_dev_limit (or max_ctx), and spills cleanly to pinned host memory if VRAM runs out.
    int kv_alloc_pos = 0;
    int kv_dev_limit = 0;
    bool kv_is_host = false;
    int mla_ws_gen = 0;
    uint64_t kv_gen = 0;
    void ensure_logits(int rows) {
        if (rows <= logits_cap) return;
        HCK(hipStreamSynchronize(st));
        if (logits) HCK(hipFree(logits));
        if (hfin) HCK(hipFree(hfin));
        logits_cap = rows;
        logits = dalloc<float>((size_t)logits_cap * c.vocab);
        hfin = dalloc<__half>((size_t)logits_cap * HID);
        ++kv_gen;
    }
    void ensure_mla_ws(size_t need) {
        if (need <= mla_ws_bytes) return;
        HCK(hipStreamSynchronize(st));
        if (mla_ws) HCK(hipFree(mla_ws));
        mla_ws_bytes = need + (need >> 2);
        HCK(hipMalloc(&mla_ws, mla_ws_bytes));
        if (getenv("HY4_POISON")) HCK(hipMemset(mla_ws, 0xFF, mla_ws_bytes));
        ++mla_ws_gen;
#if HY4_CUDA
        cg_clear();
#endif
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
    __half* tmp_pooled = nullptr;  // [16][128] per-node temporary pooled keys during Tree-8 verify
    float *ik_f = nullptr, *iv_f = nullptr, *ig_f = nullptr, *iq_f = nullptr, *hw_f = nullptr, *score = nullptr;
    __half *ik_h = nullptr, *iq_h = nullptr;
    int32_t *keys = nullptr, *nkeys = nullptr;
    size_t ldS = 0;
    bool last_use_list = false;
    // opt-in tail-skip (STRATA_HY4_TAIL_SKIP / --tail-skip): 0 = OFF by default (100% lossless); 1..15 skips missed CPU-tier experts routed only at rank >= tail_skip
    int tail_skip = 0;
    uint64_t tail_skipped_pairs = 0;

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
    // prefill streaming of the CPU-tier experts: for chunks of >= pf_min_T tokens the experts of a layer are DMA'd from the host arena into alternating device scratch buffers
    // while attention / earlier groups compute. When whole-arena hipHostRegister fails (Windows WDDM or ulimit -l containers) or HY4_PF_BOUNCE=1, a 2 x 256 MiB pinned host
    // bounce ring filled by parallel memcpy threads feeds st2 hipMemcpyAsync. For medium chunks (pf_min_T <= T < pf_gather_max_T), only active experts (cnt[e] > 0) are gathered!
    struct PfRun { const uint8_t* host; size_t dev_off, bytes; };
    struct PfGroup { std::vector<PfRun> runs; size_t total_bytes = 0; int seq = 0; ExpertDev* table = nullptr; };
    std::vector<std::vector<PfGroup>> pf_groups;  // [41 MoE layers][groups]
    std::vector<int> pf_layer_first;              // sequence index of group 0 of each layer
    int pf_nseq = 0, pf_nbuf = 1, pf_next_copy = 0, pf_next_comp = 0;
    size_t pf_cap = 0;
    uint8_t* d_scratch[2] = {nullptr, nullptr};
    uint8_t* h_bounce[2] = {nullptr, nullptr};    // 2 x cap pinned host bounce ring (used when arena is not pinned or for active-expert gather)
    ExpertDev* d_pf_dyn_table[2] = {nullptr, nullptr};
    ExpertDev* h_pf_dyn_table[2] = {nullptr, nullptr};  // persistent pinned host mirror for pf_use_gather async H2D
    bool pf_bounce = false;
    int pf_packed_seq[2] = {-1, -1};
    bool ev_pf_copy_set[2] = {false, false};
    // Persistent worker pool for zero-overhead parallel DDR5 -> pinned bounce-ring packing (zero std::thread spawns per chunk!)
    struct PfPackPool {
        static constexpr int NW = 8;
        std::mutex mu;
        std::condition_variable cv_work, cv_done;
        std::thread th[NW];
        bool started = false, stop = false;
        int gen = 0, done_cnt = 0;
        uint8_t* dst = nullptr;
        const PfGroup* cur_G = nullptr;
        hipEvent_t wait_ev = nullptr;
        bool busy = false;
        int active_buf = -1;
        void start() {
            if (started) return;
            started = true;
            for (int t = 0; t < NW; ++t) {
                th[t] = std::thread([this, t]() {
                    int my_gen = 0;
                    for (;;) {
                        const PfGroup* G = nullptr;
                        uint8_t* d = nullptr;
                        hipEvent_t ev = nullptr;
                        {
                            std::unique_lock<std::mutex> lk(mu);
                            cv_work.wait(lk, [&]() { return stop || gen != my_gen; });
                            if (stop) return;
                            my_gen = gen;
                            G = cur_G;
                            d = dst;
                            ev = wait_ev;
                        }
                        // Only worker 0 waits for the prior DMA on this bounce slot if needed, then all workers pack in parallel?
                        // Actually to avoid a barrier, kick wait_ev on the dispatcher thread or wait per-worker (CUDA/HIP events are thread-safe).
                        if (ev) HCK(hipEventSynchronize(ev));
                        if (G && !G->runs.empty()) {
                            const size_t lo = (G->total_bytes * (size_t)t) / (size_t)NW;
                            const size_t hi = (G->total_bytes * (size_t)(t + 1)) / (size_t)NW;
                            for (const PfRun& r : G->runs) {
                                const size_t r0 = r.dev_off, r1 = r.dev_off + r.bytes;
                                const size_t a = std::max(r0, lo), z = std::min(r1, hi);
                                if (a < z) std::memcpy(d + a, r.host + (a - r0), z - a);
                            }
                        }
                        {
                            std::lock_guard<std::mutex> lk(mu);
                            if (++done_cnt == NW) {
                                busy = false;
                                cv_done.notify_all();
                            }
                        }
                    }
                });
            }
        }
        void kick(int b, uint8_t* d, const PfGroup* G, hipEvent_t ev) {
            if (!started) start();
            wait();
            std::lock_guard<std::mutex> lk(mu);
            dst = d; cur_G = G; wait_ev = ev; active_buf = b;
            done_cnt = 0; busy = true; ++gen;
            cv_work.notify_all();
        }
        void wait() {
            if (!started) return;
            std::unique_lock<std::mutex> lk(mu);
            cv_done.wait(lk, [&]() { return !busy; });
        }
        void shutdown() {
            if (!started) return;
            {
                std::lock_guard<std::mutex> lk(mu);
                stop = true;
                cv_work.notify_all();
            }
            for (int t = 0; t < NW; ++t) if (th[t].joinable()) th[t].join();
            started = false;
        }
        ~PfPackPool() { shutdown(); }
    } pf_pool;
    hipStream_t st2 = nullptr;
    hipEvent_t ev_pf_copy[2] = {nullptr, nullptr}, ev_pf_done[2] = {nullptr, nullptr};
    bool ev_pf_done_set[2] = {false, false};
    bool pf_stream = false;
    int pf_min_T = 128;
    int pf_gather_max_T = 4096;  // below this T, gather only cnt[e] > 0 experts; at/above, stream full layer overlapped with attention
    ExpertDev* d_pf_tables = nullptr;
    size_t d_pf_tables_cap = 0;
    uint64_t vcache_ver = 1, pf_vcache_ver = 0;
    double pf_cpu_frac = 0.0;
    bool has_pf_cpu = false;
    std::vector<uint8_t> pf_cpu_flag;
    bool pf_active(int T) const { return pf_stream && T >= pf_min_T; }
    bool pf_use_gather(int T) const { return pf_stream && T >= pf_min_T && T < pf_gather_max_T && h_bounce[0] != nullptr; }

    void pf_pack_kick(int k) {
        if (!pf_bounce || k < 0 || k >= pf_nseq) return;
        const int b = k % pf_nbuf;
        if (pf_packed_seq[b] == k) return;
        const PfGroup* G = pf_find(k);
        pf_packed_seq[b] = k;
        pf_pool.kick(b, h_bounce[b], G, ev_pf_copy_set[b] ? ev_pf_copy[b] : nullptr);
    }
    void pf_build_groups(bool log_init = false) {
        if (!pf_cap) return;
        pf_pool.wait();
        if (st2) HCK(hipStreamSynchronize(st2));
        auto al = [](size_t x) { return (x + 4095) & ~(size_t)4095; };
        const size_t cap = pf_cap;
        pf_groups.assign(41, {});
        pf_layer_first.assign(41, 0);
        pf_cpu_flag.assign(41 * 512, 0);
        has_pf_cpu = false;
        std::vector<ExpertDev> all;
        std::vector<std::vector<std::vector<int>>> gexp(41);
        int seq = 0;
        size_t skipped_vcache = 0, total_dma_bytes = 0, maxb = 0;
        for (int l = 0; l < 41; ++l) {
            std::vector<std::pair<const uint8_t*, int>> lay;
            lay.reserve(512);
            for (int e = 0; e < 512; ++e) {
                const int gi = l * 512 + e;
                if (g.experts[gi].tier != 2 || !cpu.resident(l, e)) continue;
                if (gi < (int)vram_flag.size() && vram_flag[gi]) {
                    ++skipped_vcache;  // Already resident in AdaptiveExpertCache -> run directly from VRAM in group 0!
                    continue;
                }
                CpuExpertView vw = cpu.view(l, e);
                if (vw.body) lay.push_back({vw.body, e});
            }
            // Optional CPU/GPU parallel prefill split (HY4_PF_CPU_FRAC): peel coldest subset to CPU while PCIe+GPU runs
            if (pf_cpu_frac > 0.001 && lay.size() > 8) {
                std::sort(lay.begin(), lay.end(), [&](const auto& a, const auto& b) {
                    const int ra = (int)g.expert_rank.size() > l * 512 + a.second ? g.expert_rank[l * 512 + a.second] : a.second;
                    const int rb = (int)g.expert_rank.size() > l * 512 + b.second ? g.expert_rank[l * 512 + b.second] : b.second;
                    return ra < rb;
                });
                const size_t n_cpu = std::min(lay.size() - 1, (size_t)(lay.size() * pf_cpu_frac));
                for (size_t k = lay.size() - n_cpu; k < lay.size(); ++k) {
                    pf_cpu_flag[l * 512 + lay[k].second] = 1;
                    has_pf_cpu = true;
                }
                lay.resize(lay.size() - n_cpu);
            }
            std::sort(lay.begin(), lay.end());
            pf_layer_first[l] = seq;
            pf_groups[l].emplace_back();
            gexp[l].emplace_back();
            size_t off = 0, lbytes = 0;
            for (auto& pe : lay) {
                CpuExpertView vw = cpu.view(l, pe.second);
                const size_t sz = al(vw.body_bytes);
                lbytes += sz;
                if (off + sz > cap && off > 0) {
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
            total_dma_bytes += lbytes;
            maxb = std::max(maxb, lbytes);
            for (size_t gi = 0; gi < pf_groups[l].size(); ++gi) pf_groups[l][gi].seq = seq++;
        }
        pf_nseq = seq;
        all.reserve((size_t)seq * 512);
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
                    const int idx = l * 512 + e;
                    const ExpertRef& er = g.experts[idx];
                    if (idx < (int)vram_flag.size() && vram_flag[idx]) {
                        hp[e] = h_experts[idx];
                        hp[e].pad = (gi == 0) ? 0 : 1;  // static VRAM or AdaptiveExpertCache: run once in group 0!
                    } else if (er.tier == 3 || er.tier == 4 || pf_cpu_flag[idx]) {
                        hp[e].body_off = 0;
                        hp[e].dn_off = er.dn_off;
                        hp[e].gu_code = er.gu_code;
                        hp[e].dn_code = er.dn_code;
                        hp[e].pad = 1;  // cold SSD, peer GPU, or parallel CPU-split expert
                    } else if (er.tier == 2) {
                        hp[e].dn_off = er.dn_off;
                        hp[e].gu_code = er.gu_code;
                        hp[e].dn_code = er.dn_code;
                        if (in_group[e]) {
                            hp[e].body_off = (uint64_t)(uintptr_t)(d_scratch[b] + eoff[e]) - (uint64_t)(uintptr_t)g.d_sec4;
                            hp[e].pad = 0;
                        } else {
                            hp[e].body_off = 0;
                            hp[e].pad = 1;
                        }
                    } else {
                        hp[e] = h_experts[idx];
                        hp[e].pad = (gi == 0) ? 0 : 1;
                    }
                }
                all.insert(all.end(), hp.begin(), hp.end());
            }
        if (all.size() > d_pf_tables_cap) {
            if (d_pf_tables) HCK(hipFree(d_pf_tables));
            d_pf_tables_cap = std::max(all.size(), (size_t)41 * 12 * 512);
            d_pf_tables = dalloc<ExpertDev>(d_pf_tables_cap);
        }
        HCK(hipMemcpy(d_pf_tables, all.data(), all.size() * sizeof(ExpertDev), hipMemcpyHostToDevice));
        {
            size_t idx = 0;
            for (int l = 0; l < 41; ++l)
                for (size_t gi = 0; gi < pf_groups[l].size(); ++gi) pf_groups[l][gi].table = d_pf_tables + 512 * idx++;
        }
        pf_vcache_ver = vcache_ver;
        if (log_init) {
            fprintf(stderr, "[hybrid] prefill streaming on (%s): %d buffer(s) of %.0f MiB, %d groups for 41 layers (%.2f GB/pass, skipped %zu vcache experts, cpu_frac=%.2f, largest %.2f GB), active for T >= %d (gather < %d)\n",
                    pf_bounce ? "pinned bounce ring" : "arena pinned", pf_nbuf, cap / 1048576.0, seq, total_dma_bytes / 1e9, skipped_vcache, pf_cpu_frac, maxb / 1e9, pf_min_T, pf_gather_max_T);
        }
    }
    void pf_init() {
        auto al = [](size_t x) { return (x + 4095) & ~(size_t)4095; };
        size_t maxb = 0;
        const uint8_t *lo = nullptr, *hi = nullptr;
        for (int l = 0; l < 41; ++l) {
            size_t tot_l = 0;
            for (int e = 0; e < 512; ++e) {
                CpuExpertView vw = cpu.view(l, e);
                if (vw.body) {
                    const size_t sz = al(vw.body_bytes);
                    tot_l += sz;
                    if (!lo || vw.body < lo) lo = vw.body;
                    if (!hi || vw.body + sz > hi) hi = vw.body + sz;
                }
            }
            maxb = std::max(maxb, tot_l);
        }
        if (!maxb) return;
        bool pinned_arena = false;
#ifndef _WIN32
        const size_t arena_span = (size_t)(hi - lo);
        const char* b_env = getenv("HY4_PF_BOUNCE");
        const bool force_bounce = b_env && atoi(b_env) != 0;
        const bool force_arena  = b_env && atoi(b_env) == 0;
        const bool safe_to_pin  = ((double)arena_span < 0.48 * ram_limit_gb() * 1e9);
        if (!force_bounce && (force_arena || safe_to_pin)) {
            if (hipHostRegister((void*)lo, arena_span, 0) == hipSuccess) pinned_arena = true;
            else (void)hipGetLastError();
        }
#endif
        pf_bounce = !pinned_arena;
        size_t fr = 0, tot = 0;
        HCK(hipMemGetInfo(&fr, &tot));
        const size_t margin = (size_t)320 << 20, tables_est = (size_t)41 * 16 * 8192;
        const size_t cap_limit = getenv("HY4_PF_BUF_MB") ? ((size_t)std::max(64, atoi(getenv("HY4_PF_BUF_MB"))) << 20) : ((size_t)(tot >= (20ull << 30) ? 384 : 256) << 20);
        const size_t want_fr = margin + tables_est + 2 * cap_limit;
        if (fr < want_fr && vcache_nslots > 1) {
            vcache_release_vram(want_fr);
            HCK(hipMemGetInfo(&fr, &tot));
        }
        size_t avail = fr > margin + tables_est ? fr - margin - tables_est : 0;
        size_t cap;
        if (!pf_bounce && avail >= maxb && tot >= (48ull << 30)) { pf_nbuf = 1; cap = maxb; }
        else { pf_nbuf = 2; cap = std::min(cap_limit, (avail / 2) & ~(size_t)4095); }
        if (cap < ((size_t)64 << 20)) {
            fprintf(stderr, "[hybrid] prefill streaming disabled: no room for a scratch buffer (free %.2f GB, layer %.2f GB)\n", fr / 1e9, maxb / 1e9);
            return;
        }
        pf_cap = cap;
        for (int b = 0; b < pf_nbuf; ++b) {
            d_scratch[b] = dalloc<uint8_t>(cap);
            d_pf_dyn_table[b] = dalloc<ExpertDev>(512);
            HCK(hipHostMalloc((void**)&h_pf_dyn_table[b], 512 * sizeof(ExpertDev), hipHostMallocDefault));
        }
        for (int b = 0; b < pf_nbuf; ++b) {
            if (hipHostMalloc((void**)&h_bounce[b], cap, hipHostMallocDefault) != hipSuccess) {
                (void)hipGetLastError();
                for (int j = 0; j < b; ++j) { hipHostFree(h_bounce[j]); h_bounce[j] = nullptr; }
                if (pf_bounce) {
                    fprintf(stderr, "[hybrid] prefill streaming disabled: cannot allocate %d x %.0f MiB pinned bounce ring\n", pf_nbuf, cap / 1048576.0);
                    for (int j = 0; j < pf_nbuf; ++j) { hipFree(d_scratch[j]); d_scratch[j] = nullptr; }
                    return;
                }
                break;
            }
        }
        if (!st2) HCK(hipStreamCreate(&st2));
        for (int b = 0; b < 2; ++b) {
            HCK(hipEventCreateWithFlags(&ev_pf_copy[b], hipEventDisableTiming));
            HCK(hipEventCreateWithFlags(&ev_pf_done[b], hipEventDisableTiming));
        }
        if (getenv("HY4_PF_MIN_T")) pf_min_T = atoi(getenv("HY4_PF_MIN_T"));
        if (getenv("HY4_PF_GATHER_MAX_T")) pf_gather_max_T = atoi(getenv("HY4_PF_GATHER_MAX_T"));
        if (getenv("HY4_PF_CPU_FRAC")) pf_cpu_frac = std::max(0.0, std::min(0.45, atof(getenv("HY4_PF_CPU_FRAC"))));
        else if (pf_bounce && cpu.n_threads() >= 12) pf_cpu_frac = 0.20;
        pf_stream = true;
        pf_build_groups(true);
    }
    void pf_begin_chunk() {
        if (pf_vcache_ver != vcache_ver) pf_build_groups(false);
        pf_pool.wait();
        for (int b = 0; b < 2; ++b) {
            pf_packed_seq[b] = -1;
            ev_pf_copy_set[b] = false;
        }
        pf_next_copy = 0;
        pf_next_comp = 0;
    }
    // queue the DMA of sequence entries up to `upto` (inclusive) whose buffer is free
    void pf_pump(int upto) {
        while (pf_next_copy <= upto && pf_next_copy < pf_nseq) {
            const int k = pf_next_copy, b = k % pf_nbuf;
            if (k - pf_nbuf >= pf_next_comp) break;
            const PfGroup* G = pf_find(k);
            if (pf_bounce) {
                if (pf_packed_seq[b] != k) pf_pack_kick(k);
                pf_pool.wait();
                if (ev_pf_done_set[b]) HCK(hipStreamWaitEvent(st2, ev_pf_done[b], 0));
                if (G->total_bytes > 0) HCK(hipMemcpyAsync(d_scratch[b], h_bounce[b], G->total_bytes, hipMemcpyHostToDevice, st2));
                HCK(hipEventRecord(ev_pf_copy[b], st2));
                ev_pf_copy_set[b] = true;
            } else {
                if (ev_pf_done_set[b]) HCK(hipStreamWaitEvent(st2, ev_pf_done[b], 0));
                for (const PfRun& r : G->runs) HCK(hipMemcpyAsync(d_scratch[b] + r.dev_off, r.host, r.bytes, hipMemcpyHostToDevice, st2));
                HCK(hipEventRecord(ev_pf_copy[b], st2));
                ev_pf_copy_set[b] = true;
            }
            ++pf_next_copy;
            // lookahead pack of the next buffer while st2 transfers the current one
            if (pf_bounce && pf_nbuf > 1 && pf_next_copy < pf_nseq && (pf_next_copy - pf_nbuf < pf_next_comp)) pf_pack_kick(pf_next_copy);
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

    // ---- Adaptive Per-Layer VRAM Expert Cache (online decayed LFU + 4-slot async st2 promotion ring) ----
    // Parameters tuned from real 41-layer SQ-64 traces: core_frac=0 (100% dynamic when RAM allows), decay=0.965, max_swaps=2/layer.
    struct VSlot { int layer = -1, expert = -1; };
    struct VProm {
        uint8_t* h_stg = nullptr;
        hipEvent_t ev_done = nullptr, ev_evict = nullptr;
        bool inflight = false;
        int slot = -1, layer = -1, expert = -1;
    };
    static constexpr int VPROM_RING = 8;
    int vcache_nslots = 0;
    size_t vcache_slot_bytes = 0;
    double vcache_decay = 0.965;
    int vcache_max_swaps = 4;
    std::vector<VSlot> vslots;
    std::vector<int> vslot_of;        // [41*512]: slot index in [0, vcache_nslots) or -1
    std::vector<double> vscore;       // [41*512]: decayed routing score (scaled by vcache_inv_scale)
    std::vector<uint8_t> vram_flag;   // [41*512]: 1 = evaluated on GPU (static or vcache), 0 = evaluated on CPU
    std::vector<uint8_t> v_use_buf;   // [Tc*16]: per-pair CPU mask for CpuExperts::submit
    double vcache_inv_scale = 1.0;
    VProm vprom[VPROM_RING];
    bool vcache_frozen = false;       // frozen during --spec-check / --parity-m for deterministic verify comparison
    uint64_t vc_hits = 0, vc_misses = 0, vc_promotes = 0;

    void vcache_init() {
        vram_flag.assign(41 * 512, 0);
        for (int i = 0; i < 41 * 512; ++i)
            if (g.experts[i].tier == 0 || g.experts[i].tier == 1) vram_flag[i] = 1;
        vcache_nslots = g.vcache_nslots;
        vcache_slot_bytes = g.vcache_slot_bytes;
        if (!hyb || vcache_nslots <= 0 || vcache_slot_bytes == 0) return;
        if (getenv("HY4_VCACHE_DECAY")) vcache_decay = std::max(0.80, std::min(0.999, atof(getenv("HY4_VCACHE_DECAY"))));
        if (getenv("HY4_VCACHE_MAX_SWAPS")) vcache_max_swaps = std::max(0, std::min(VPROM_RING, atoi(getenv("HY4_VCACHE_MAX_SWAPS"))));
        vslots.assign(vcache_nslots, VSlot{});
        vslot_of.assign(41 * 512, -1);
        vscore.assign(41 * 512, 0.0);
        v_use_buf.assign((size_t)Tc * 16, 0);
        for (int i = 0; i < 41 * 512; ++i) {
            const int r = i < (int)g.expert_rank.size() ? g.expert_rank[i] : 511;
            vscore[i] = 0.5 / (1.0 + 0.05 * (double)r);  // soft rank prior so session hits rapidly adapt
        }
        if (!st2) HCK(hipStreamCreate(&st2));
        for (int p = 0; p < VPROM_RING; ++p) {
            HCK(hipHostMalloc((void**)&vprom[p].h_stg, vcache_slot_bytes, hipHostMallocDefault));
            HCK(hipEventCreateWithFlags(&vprom[p].ev_done, hipEventDisableTiming));
            HCK(hipEventCreateWithFlags(&vprom[p].ev_evict, hipEventDisableTiming));
        }
        // Pre-populate dynamic VRAM slots with the hottest tier-2 (RAM-resident) experts in rank-major order
        double t0 = now_s();
        int filled = 0;
        std::vector<std::pair<int, int>> cands;  // (rank, l*512+e)
        cands.reserve(41 * 512);
        for (int l = 0; l < 41; ++l)
            for (int e = 0; e < 512; ++e) {
                const int idx = l * 512 + e;
                if (g.experts[idx].tier == 2 && cpu.resident(l, e)) cands.push_back({g.expert_rank[idx], idx});
            }
        std::stable_sort(cands.begin(), cands.end());
        for (const auto& pr : cands) {
            if (filled >= vcache_nslots) break;
            const int idx = pr.second, l = idx / 512, e = idx % 512;
            CpuExpertView vw = cpu.view(l, e);
            if (!vw.body || vw.body_bytes > vcache_slot_bytes) continue;
            const uint64_t slot_off = g.vcache_slot_off(filled);
            const int rp = filled % VPROM_RING;
            if (filled >= VPROM_RING) HCK(hipEventSynchronize(vprom[rp].ev_done));
            std::memcpy(vprom[rp].h_stg, vw.body, vw.body_bytes);
            HCK(hipMemcpyAsync(g.vcache_slot_ptr(filled), vprom[rp].h_stg, vw.body_bytes, hipMemcpyHostToDevice, st2));
            HCK(hipEventRecord(vprom[rp].ev_done, st2));
            vslots[filled] = {l, e};
            vslot_of[idx] = filled;
            vram_flag[idx] = 1;
            h_experts[idx].body_off = slot_off;
            h_experts[idx].pad = 0;
            ++filled;
        }
        HCK(hipStreamSynchronize(st2));
        HCK(hipMemcpy(d_experts, h_experts, (size_t)41 * 512 * sizeof(ExpertDev), hipMemcpyHostToDevice));
        fprintf(stderr, "[vcache] pre-populated %d/%d adaptive VRAM slots (%.2f GB, decay=%.3f, max_swaps=%d) in %.2f s\n",
                filled, vcache_nslots, (filled * (double)vcache_slot_bytes) / 1e9, vcache_decay, vcache_max_swaps, now_s() - t0);
    }
    void vcache_poll_commit() {
        for (int p = 0; p < VPROM_RING; ++p) {
            if (!vprom[p].inflight) continue;
            if (hipEventQuery(vprom[p].ev_done) != hipSuccess) continue;
            const int idx = vprom[p].layer * 512 + vprom[p].expert;
            const uint64_t slot_off = g.vcache_slot_off(vprom[p].slot);
            vslots[vprom[p].slot] = {vprom[p].layer, vprom[p].expert};
            vslot_of[idx] = vprom[p].slot;
            vram_flag[idx] = 1;
            h_experts[idx].body_off = slot_off;
            h_experts[idx].pad = 0;
            HCK(hipMemcpyAsync(d_experts + idx, &h_experts[idx], sizeof(ExpertDev), hipMemcpyHostToDevice, st));
            vprom[p].inflight = false;
            ++vc_promotes;
            ++vcache_ver;
        }
    }
    void vcache_flush() {
        for (int p = 0; p < VPROM_RING; ++p)
            if (vprom[p].inflight) HCK(hipEventSynchronize(vprom[p].ev_done));
        vcache_poll_commit();
    }
    // Batch rebalance at end of prefill chunk: promote the prompt's hottest tier-2 experts into vcache so decode starts hot from token 1
    void vcache_rebalance_prefill() {
        if (!hyb || vcache_nslots <= 1 || vcache_frozen) return;
        vcache_flush();
        HCK(hipStreamSynchronize(st));
        std::vector<std::pair<double, int>> in_slots;
        in_slots.reserve(vcache_nslots);
        for (int s = 0; s < vcache_nslots; ++s) {
            const int l = vslots[s].layer, e = vslots[s].expert;
            const double sc = (l >= 0) ? vscore[l * 512 + e] : -1.0;
            in_slots.push_back({sc, s});
        }
        std::sort(in_slots.begin(), in_slots.end());
        std::vector<std::pair<double, int>> hot_cpu;
        const double min_in = in_slots.empty() ? 0.0 : in_slots[0].first;
        for (int l = 0; l < 41; ++l)
            for (int e = 0; e < 512; ++e) {
                const int idx = l * 512 + e;
                if (g.experts[idx].tier == 2 && !vram_flag[idx] && vscore[idx] > min_in * 1.05 && cpu.resident(l, e))
                    hot_cpu.push_back({vscore[idx], idx});
            }
        if (hot_cpu.empty()) return;
        std::sort(hot_cpu.rbegin(), hot_cpu.rend());
        const int max_swaps = std::min<int>({(int)hot_cpu.size(), (int)in_slots.size(), std::max(128, (vcache_nslots * 3) / 4)});
        int swapped = 0;
        for (int i = 0; i < max_swaps; ++i) {
            if (hot_cpu[i].first <= in_slots[i].first * 1.05) break;
            const int s = in_slots[i].second;
            const int old_l = vslots[s].layer, old_e = vslots[s].expert;
            if (old_l >= 0) {
                const int oidx = old_l * 512 + old_e;
                vslot_of[oidx] = -1;
                vram_flag[oidx] = 0;
                h_experts[oidx].pad = 1;
            }
            const int nidx = hot_cpu[i].second, nl = nidx / 512, ne = nidx % 512;
            CpuExpertView vw = cpu.view(nl, ne);
            const uint64_t slot_off = g.vcache_slot_off(s);
            const int rp = swapped % VPROM_RING;
            if (swapped >= VPROM_RING) HCK(hipEventSynchronize(vprom[rp].ev_done));
            std::memcpy(vprom[rp].h_stg, vw.body, vw.body_bytes);
            HCK(hipMemcpyAsync(g.vcache_slot_ptr(s), vprom[rp].h_stg, vw.body_bytes, hipMemcpyHostToDevice, st2));
            HCK(hipEventRecord(vprom[rp].ev_done, st2));
            vslots[s] = {nl, ne};
            vslot_of[nidx] = s;
            vram_flag[nidx] = 1;
            h_experts[nidx].body_off = slot_off;
            h_experts[nidx].pad = 0;
            ++swapped;
        }
        if (swapped > 0) {
            HCK(hipStreamSynchronize(st2));
            HCK(hipMemcpy(d_experts, h_experts, (size_t)41 * 512 * sizeof(ExpertDev), hipMemcpyHostToDevice));
            vc_promotes += (uint64_t)swapped;
            ++vcache_ver;
        }
    }
    // Online decode update: record routing hits for layer l (0..40) and asynchronously promote up to vcache_max_swaps hottest misses on st2
    void vcache_note_decode(int l, int T, const int32_t* idx_arr) {
        if (vcache_nslots <= 1) return;
        const size_t n = (size_t)T * 16;
        int cand_idx[VPROM_RING];
        double cand_sc[VPROM_RING];
        for (int k = 0; k < VPROM_RING; ++k) { cand_idx[k] = -1; cand_sc[k] = 0.0; }
        const int max_c = std::min(vcache_max_swaps, VPROM_RING);
        for (size_t i = 0; i < n; ++i) {
            const int e = idx_arr[i];
            if (e < 0 || e >= 512) continue;
            const int gi = l * 512 + e;
            if (vram_flag[gi]) ++vc_hits;
            else ++vc_misses;
            if (!vcache_frozen) {
                vscore[gi] += vcache_inv_scale;
                if (max_c > 0 && g.experts[gi].tier == 2 && !vram_flag[gi] && cpu.resident(l, e)) {
                    // Check not already inflight
                    bool infl = false;
                    for (int p = 0; p < VPROM_RING; ++p)
                        if (vprom[p].inflight && vprom[p].layer == l && vprom[p].expert == e) { infl = true; break; }
                    if (!infl) {
                        const double sc = vscore[gi];
                        int dup = -1, min_k = 0;
                        for (int k = 0; k < max_c; ++k) {
                            if (cand_idx[k] == gi) dup = k;
                            if (cand_sc[k] < cand_sc[min_k]) min_k = k;
                        }
                        if (dup >= 0) cand_sc[dup] = sc;
                        else if (sc > cand_sc[min_k]) { cand_idx[min_k] = gi; cand_sc[min_k] = sc; }
                    }
                }
            }
        }
        if (vcache_frozen || max_c <= 0) return;
        vcache_poll_commit();
        // Sort candidates descending by score
        for (int a = 0; a < max_c; ++a)
            for (int b = a + 1; b < max_c; ++b)
                if (cand_sc[b] > cand_sc[a]) { std::swap(cand_sc[a], cand_sc[b]); std::swap(cand_idx[a], cand_idx[b]); }
        for (int ci = 0; ci < max_c; ++ci) {
            const int best_miss = cand_idx[ci];
            if (best_miss < 0) break;
            // Find a free ring staging entry
            int rp = -1;
            for (int p = 0; p < VPROM_RING; ++p)
                if (!vprom[p].inflight) { rp = p; break; }
            if (rp < 0) break;
            // Exact coldest-slot scan (skips in-flight sentinel layer == -2; ~2 us for 3300 slots)
            int victim = -1;
            double vic_sc = 1e300;
            for (int s = 0; s < vcache_nslots; ++s) {
                const int sl = vslots[s].layer, se = vslots[s].expert;
                if (sl == -2) continue;  // in-flight promotion target
                const double sc = (sl >= 0) ? vscore[sl * 512 + se] : -1.0;
                if (sc < vic_sc) { vic_sc = sc; victim = s; }
            }
            if (victim < 0 || cand_sc[ci] <= vic_sc * 1.05) break;
            const int nl = best_miss / 512, ne = best_miss % 512;
            CpuExpertView vw = cpu.view(nl, ne);
            if (!vw.body || vw.body_bytes > vcache_slot_bytes) continue;
            // Evict victim from d_experts on st first, record ev_evict on st, wait on st2, then async DMA into victim slot
            const int ol = vslots[victim].layer, oe = vslots[victim].expert;
            if (ol >= 0) {
                const int oidx = ol * 512 + oe;
                vslot_of[oidx] = -1;
                vram_flag[oidx] = 0;
                h_experts[oidx].pad = 1;
                HCK(hipMemcpyAsync(d_experts + oidx, &h_experts[oidx], sizeof(ExpertDev), hipMemcpyHostToDevice, st));
            }
            vslots[victim] = {-2, -2};  // mark in-flight so no other promotion picks this slot
            HCK(hipEventRecord(vprom[rp].ev_evict, st));
            HCK(hipStreamWaitEvent(st2, vprom[rp].ev_evict, 0));
            std::memcpy(vprom[rp].h_stg, vw.body, vw.body_bytes);
            HCK(hipMemcpyAsync(g.vcache_slot_ptr(victim), vprom[rp].h_stg, vw.body_bytes, hipMemcpyHostToDevice, st2));
            HCK(hipEventRecord(vprom[rp].ev_done, st2));
            vprom[rp].inflight = true;
            vprom[rp].slot = victim;
            vprom[rp].layer = nl;
            vprom[rp].expert = ne;
        }
    }
    void vcache_decay_step(int T = 1) {
        if (vcache_nslots <= 1 || vcache_frozen) return;
        const double step = std::pow(vcache_decay, (double)std::max(1, std::min(T, 4)));
        vcache_inv_scale /= step;
        if (vcache_inv_scale > 1e12) {
            const double inv = 1.0 / vcache_inv_scale;
            for (double& s : vscore) s *= inv;
            vcache_inv_scale = 1.0;
        }
    }
    // Reclaim tail vcache slabs back to the GPU allocator so sparse KV + IndexPool can grow up to 128K 100% in VRAM without spilling to host!
    bool vcache_release_vram(size_t need_free) {
        if (g.vcache_slabs.size() <= 1 || g.vcache_slots_per_slab <= 0) return false;
        vcache_flush();
        HCK(hipDeviceSynchronize());
        int freed_slabs = 0;
        size_t freed_bytes = 0;
        size_t fr = 0, tot = 0;
        HCK(hipMemGetInfo(&fr, &tot));
        while (fr < need_free && g.vcache_slabs.size() > 1) {
            const int cut = (int)(g.vcache_slabs.size() - 1) * g.vcache_slots_per_slab;
            const int take = vcache_nslots - cut;
            for (int s = cut; s < vcache_nslots; ++s) {
                const int ol = vslots[s].layer, oe = vslots[s].expert;
                if (ol >= 0) {
                    const int oidx = ol * 512 + oe;
                    vslot_of[oidx] = -1;
                    vram_flag[oidx] = 0;
                    h_experts[oidx].pad = 1;
                }
            }
            HCK(hipFree(g.vcache_slabs.back()));
            g.vcache_slabs.pop_back();
            vcache_nslots = cut;
            g.vcache_nslots = cut;
            vslots.resize(cut);
            ++freed_slabs;
            freed_bytes += (size_t)take * vcache_slot_bytes;
            HCK(hipMemGetInfo(&fr, &tot));
        }
        if (freed_slabs > 0) {
            HCK(hipMemcpy(d_experts, h_experts, (size_t)41 * 512 * sizeof(ExpertDev), hipMemcpyHostToDevice));
            ++vcache_ver;
            fprintf(stderr, "[vcache] released %d slab(s) (%.2f GB) for VRAM KV growth -> %d adaptive slots (%.2f GB) remain\n",
                    freed_slabs, freed_bytes / 1e9, vcache_nslots, (vcache_nslots * (double)vcache_slot_bytes) / 1e9);
        }
        return fr >= need_free;
    }
    double t_cpu_wait = 0;  // host time spent waiting for the CPU experts after the GPU work was queued (layer sum)

    // Multi-GPU expert-parallel peer state (secondary GPUs holding hot experts via --gpus 0,1,...)
    struct PeerRun {
        int dev = 0;
        hipStream_t st = nullptr;
        hipEvent_t ev_in = nullptr, ev_out = nullptr;
        ExpertDev* d_experts = nullptr;
        __half* z_h = nullptr;
        int32_t* idx = nullptr;
        float *w = nullptr, *zout = nullptr;
        void* moe_ws = nullptr;
        float* d_zpeer0 = nullptr;  // on primary GPU 0
    };
    std::vector<PeerRun> peer_runs;
    void peers_launch(int il, int T) {
#if HY4_CUDA
        if (peer_runs.empty()) return;
        const size_t nz = (size_t)T * 1536;
        for (size_t p = 0; p < peer_runs.size(); ++p) {
            PeerRun& pr = peer_runs[p];
            HCK(cudaEventRecord(pr.ev_in, st));
            HCK(cudaSetDevice(pr.dev));
            HCK(cudaStreamWaitEvent(pr.st, pr.ev_in, 0));
            HCK(cudaMemcpyPeerAsync(pr.z_h, pr.dev, z_h, g.primary_gpu, nz * 2, pr.st));
            HCK(cudaMemcpyPeerAsync(pr.idx, pr.dev, moe_idx, g.primary_gpu, (size_t)T * 16 * 4, pr.st));
            HCK(cudaMemcpyPeerAsync(pr.w, pr.dev, moe_w, g.primary_gpu, (size_t)T * 16 * 4, pr.st));
            moe_experts(pr.st, g.peers[p].d_sec4, pr.d_experts + (size_t)(il - 1) * 512, pr.z_h, pr.idx, pr.w, pr.zout, T, pr.moe_ws);
            HCK(cudaMemcpyPeerAsync(pr.d_zpeer0, g.primary_gpu, pr.zout, pr.dev, nz * 4, pr.st));
            HCK(cudaEventRecord(pr.ev_out, pr.st));
        }
        HCK(cudaSetDevice(g.primary_gpu));
#else
        (void)il; (void)T;
#endif
    }
    void peers_gather(int T) {
#if HY4_CUDA
        if (peer_runs.empty()) return;
        const size_t nz = (size_t)T * 1536;
        for (auto& pr : peer_runs) {
            HCK(cudaStreamWaitEvent(st, pr.ev_out, 0));
            hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)pr.d_zpeer0, nz);
        }
#else
        (void)T;
#endif
    }

    // CUDA Graph capture & replay for decode (M=1) and verify (M=2..8, linear or Tree-8): collapses ~2,200 tiny launches into 1-3 graph launches (all-VRAM) or 84 launches (hybrid)
    struct DevStepArgs {
        int p0;
        int idx_G;
        int is_tree;
        TreePos mla_tp;
        MlaAttnArgs::TreeDev mla_tree;
        idx::IdxTree idx_tp;
    };
    DevStepArgs *h_step = nullptr, *d_step = nullptr;
    int32_t* h_ids = nullptr;  // pinned staging for token ids [<= 16]
    bool cuda_graph_on = false;
    bool capturing = false;
    int cap_score_grid = 0;
#if HY4_CUDA
    struct CgKey {
        int T, is_tree, use_list, s_causal, s_swa, score_grid, dfly_cap, route_now, nof, kv_is_host, slots_sparse, engram_ready, hyb, n_peers;
        int lin_gemv_max, moe_dec_max, int8_mma, dp4a, mla_ws_gen;
        bool operator==(const CgKey& o) const { return std::memcmp(this, &o, sizeof(CgKey)) == 0; }
    };
    struct CgEntry {
        CgKey key;
        cudaGraphExec_t g_l0 = nullptr;       // embed + L0 (or all 42 layers when !engram_ready && !hyb && n_peers==0)
        cudaGraphExec_t g_l1_10 = nullptr;    // L1 .. emeta.layers[1]-1 (when engram_ready && !hyb && n_peers==0)
        cudaGraphExec_t g_l11_41 = nullptr;   // emeta.layers[1] .. 41 (when engram_ready && !hyb && n_peers==0)
        cudaGraphExec_t g_pre[41] = {};       // hybrid / multi-GPU: per-MoE-layer pre segment
        cudaGraphExec_t g_gpuexp[41] = {};    // hybrid / multi-GPU: per-MoE-layer GPU-expert + shared segment
        cudaGraphExec_t g_tail_last = nullptr; // hybrid / multi-GPU: layer 41 post tail
        std::vector<std::array<cudaGraphExec_t, 41>> g_peers;  // multi-GPU: per-peer moe_experts graphs
        void destroy() {
            auto d = [](cudaGraphExec_t& e) { if (e) { (void)cudaGraphExecDestroy(e); e = nullptr; } };
            d(g_l0); d(g_l1_10); d(g_l11_41); d(g_tail_last);
            for (int i = 0; i < 41; ++i) { d(g_pre[i]); d(g_gpuexp[i]); }
            for (auto& arr : g_peers) for (int i = 0; i < 41; ++i) d(arr[i]);
            g_peers.clear();
        }
    };
    std::vector<CgEntry> cg_cache;
    void cg_clear() {
        for (auto& e : cg_cache) e.destroy();
        cg_cache.clear();
    }
    ~Runner() { cg_clear(); if (h_experts) { (void)hipHostFree(h_experts); h_experts = nullptr; } }
#endif

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
    void lin(const std::string& name, const __half* X, float* Y, int M, __half* w16 = nullptr) {
        const TInfo& t = ti(name);
        size_t dot = name.rfind('.', name.rfind('.') - 1);
        PS _p(this, std::string("lin ") + name.substr(name.find('.', 13) == std::string::npos ? 0 : name.find('.', 13) + 1));
        (void)dot;
#if HY4_MMA && !HY4_CUDA
        static const bool allow_w16 = !getenv("HY4_NO_W16") && skd::is_gfx12_device();
        // On RDNA4 (gfx12), dequant S_K -> FP16 once when the FP16 weight fits in the 64 MB Infinity Cache (<= 48 MiB)
        if (allow_w16 && w16 && M >= 256 && t.fmt != FMT_Q8ROW && skd::int8_wmma_mode() == 0 && (size_t)t.rows() * t.cols() * 2 <= (48ull << 20)) {
            linear_h_dequant_f16(st, t.fmt, tw(name), (int)t.rows(), (int)t.cols(), X, Y, M, w16);
            return;
        }
#else
        (void)w16;
#endif
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
    std::vector<LinMulti> lm_attn;  // per layer: q_a + kv_a + linear_gate (T <= 8, same cur_h)
    std::vector<LinMulti> lm_sh;    // per MoE layer (il-1): shared_experts.gate + up (T <= 8, same cur_h)
    LinMulti lm_mlp0;               // layer 0: gate_proj + up_proj (T <= 8, same cur_h)
    bool lm_ok_m(int T) const {
        if (T > 8 || T > skd::lin_gemv_max_m()) return false;
#if HY4_CUDA && HY4_MMA
        if (skd::lin_dp4a()) return false;
#endif
        return true;
    }
    void tune_init() {
        const char* e = getenv("HY4_TUNE");
        tune = e ? atoi(e) : TUNE_DEFAULT;
        if (tune == 0) return;
        slot_rec = dalloc<ExpertDev>(16);
        HCK(hipMemset(moe_ws_dec, 0, moe_ws_bytes(8)));  // the one-token expert kernels keep their column tickets in the (otherwise unused) `fill` table of the workspace
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
        // Fuse same-input, same-format projections for decode/verify (T <= 8): q_a+kv_a+linear_gate, shared gate+up, L0 gate+up
        lm_attn.resize(c.n_layers);
        for (int il = 0; il < c.n_layers; ++il) {
            const std::string pa = L(il) + ".self_attn";
            const std::string nms[3] = {pa + ".q_a_proj.weight", pa + ".kv_a_proj_with_mqa.weight", pa + ".linear_gate.weight"};
            const void* W[3]; float* Y[3] = {F1, F1 + 8192, F3_dec}; int N[3];
            bool ok = true; int fmt = -1, K = 0;
            for (int j = 0; j < 3; ++j) {
                const TInfo& t = ti(nms[j]);
                if (j == 0) { fmt = t.fmt; K = (int)t.cols(); }
                ok = ok && t.fmt == fmt && (int)t.cols() == K && t.fmt != FMT_Q8ROW;
                W[j] = tw(nms[j]); N[j] = (int)t.rows();
            }
            if (ok) lm_attn[il] = linear_multi_build(fmt, K, 3, W, Y, N);
        }
        lm_sh.resize(c.n_layers > 1 ? c.n_layers - 1 : 0);
        for (int il = 1; il < c.n_layers; ++il) {
            const std::string pm = L(il) + ".mlp";
            const TInfo& t0 = ti(pm + ".shared_experts.gate_proj.weight");
            const TInfo& t1 = ti(pm + ".shared_experts.up_proj.weight");
            if (t0.fmt == t1.fmt && t0.cols() == t1.cols() && t0.fmt != FMT_Q8ROW) {
                const void* W[2] = {tw(pm + ".shared_experts.gate_proj.weight"), tw(pm + ".shared_experts.up_proj.weight")};
                float* Y[2] = {F1, F3_dec};
                int N[2] = {(int)t0.rows(), (int)t1.rows()};
                lm_sh[il - 1] = linear_multi_build(t0.fmt, (int)t0.cols(), 2, W, Y, N);
            }
        }
        {
            const std::string pm0 = L(0) + ".mlp";
            const TInfo& t0 = ti(pm0 + ".gate_proj.weight");
            const TInfo& t1 = ti(pm0 + ".up_proj.weight");
            if (t0.fmt == t1.fmt && t0.cols() == t1.cols() && t0.fmt != FMT_Q8ROW) {
                const void* W[2] = {tw(pm0 + ".gate_proj.weight"), tw(pm0 + ".up_proj.weight")};
                float* Y[2] = {F1, F2};
                int N[2] = {(int)t0.rows(), (int)t1.rows()};
                lm_mlp0 = linear_multi_build(t0.fmt, (int)t0.cols(), 2, W, Y, N);
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
        ldS = (size_t)(max_ctx / 4 + 8);
        moe_ws_dec = dalloc<uint8_t>(moe_ws_bytes(8));
        F3_dec = dalloc<float>((size_t)8 * 16384);  // 512 KiB dedicated decode/verify scratch for fused lm_attn/lm_sh/CUDA Graphs (zero aliasing with work_arena/score)
        if (!dump) {
            const size_t arena_bytes = std::max({(size_t)2 * big * sizeof(float), (size_t)T * ldS * sizeof(float), moe_ws_bytes(T)});
            work_arena = dalloc<uint8_t>(arena_bytes);
            F1 = (float*)work_arena;
            F2 = F1 + big;
            F3 = F1;            // attention linear_gate and layer-0 down_proj; layers 1..41 use F1 + T*1536
            ctx_ws = F2;        // mla_attend stores __half[T][NH][512] = T*16384*4 bytes = big floats (F2 is dead before mla_attend)
            kv_f = F1;          // 5*T*3072 = T*15360 <= big (engram_layer runs when F1 is dead)
            score = (float*)work_arena;
            moe_ws = work_arena;
            H1 = dalloc<__half>(big);
            o_h = H1;           // o_h is only live in attention_layer; H1 is only live in mlp_layer
        } else {
            F1 = dalloc<float>(big);
            F2 = dalloc<float>(big);
            F3 = dalloc<float>(big);
            H1 = dalloc<__half>(big);
            o_h = dalloc<__half>((size_t)T * 16384);
            ctx_ws = dalloc<float>((size_t)T * 64 * 512);
            kv_f = dalloc<float>((size_t)5 * T * HID);
            score = dalloc<float>((size_t)T * ldS);
            moe_ws = dalloc<uint8_t>(moe_ws_bytes(T));
        }
        qr_h = dalloc<__half>((size_t)T * 1024);
        qa_h = dalloc<__half>((size_t)T * 64 * 576);
        z_f = dalloc<float>((size_t)T * 1536);
        z_h = dalloc<__half>((size_t)T * 1536);
        zout_f = dalloc<float>((size_t)T * 1536);
        moe_idx = dalloc<int32_t>((size_t)T * 16);
        moe_w = dalloc<float>((size_t)T * 16);
        ensure_logits(16);  // 7.7 MiB instead of Tc * vocab * 4 (1-2 GB); grows on demand for --all-logits
        d_argmax = dalloc<int32_t>(T);
        mla_ws_bytes = 0;
        mla_ws = nullptr;
        // experts table (pinned host mirror for non-blocking vcache H2D updates)
        {
            HCK(hipHostMalloc((void**)&h_experts, (size_t)41 * 512 * sizeof(ExpertDev), hipHostMallocDefault));
            for (int i = 0; i < 41 * 512; ++i) {
                const ExpertRef& e = g.experts[i];
                h_experts[i].body_off = e.body_off;
                h_experts[i].dn_off = e.dn_off;
                h_experts[i].gu_code = e.gu_code;
                h_experts[i].dn_code = e.dn_code;
                h_experts[i].pad = (e.tier >= 2) ? 1 : 0;  // CPU (2), cold-SSD (3), or peer-GPU (4) skipped by GPU 0; tier 0 (VRAM) & tier 1 (pinned host) run on GPU 0
            }
            d_experts = dalloc<ExpertDev>(41 * 512);
            HCK(hipMemcpy(d_experts, h_experts, (size_t)41 * 512 * sizeof(ExpertDev), hipMemcpyHostToDevice));
        }
#if HY4_CUDA
        if (!g.peers.empty()) {
            skd::moe_has_cpu() = 1;  // zero w.y on every GPU so partial sums add cleanly
            peer_runs.resize(g.peers.size());
            for (size_t p = 0; p < g.peers.size(); ++p) {
                PeerRun& pr = peer_runs[p];
                pr.dev = g.peers[p].dev;
                pr.d_zpeer0 = dalloc<float>((size_t)Tc * 1536);
                HCK(cudaEventCreateWithFlags(&pr.ev_in, cudaEventDisableTiming));
                HCK(cudaSetDevice(pr.dev));
                HCK(cudaStreamCreate(&pr.st));
                HCK(cudaEventCreateWithFlags(&pr.ev_out, cudaEventDisableTiming));
                std::vector<ExpertDev> ph(41 * 512);
                for (int i = 0; i < 41 * 512; ++i) {
                    const ExpertRef& e = g.peers[p].experts[i];
                    ph[i].body_off = e.body_off;
                    ph[i].dn_off = e.dn_off;
                    ph[i].gu_code = e.gu_code;
                    ph[i].dn_code = e.dn_code;
                    ph[i].pad = (e.tier != 0) ? 1 : 0;
                }
                HCK(cudaMalloc(&pr.d_experts, ph.size() * sizeof(ExpertDev)));
                HCK(cudaMemcpy(pr.d_experts, ph.data(), ph.size() * sizeof(ExpertDev), cudaMemcpyHostToDevice));
                HCK(cudaMalloc(&pr.z_h, (size_t)Tc * 1536 * 2));
                HCK(cudaMalloc(&pr.idx, (size_t)Tc * 16 * 4));
                HCK(cudaMalloc(&pr.w, (size_t)Tc * 16 * 4));
                HCK(cudaMalloc(&pr.zout, (size_t)Tc * 1536 * 4));
                HCK(cudaMalloc(&pr.moe_ws, moe_ws_bytes(Tc)));
            }
            HCK(cudaSetDevice(g.primary_gpu));
            fprintf(stderr, "[runner] multi-GPU expert-parallel active across %zu GPUs\n", 1 + peer_runs.size());
        }
#endif
        if (!g.cpu_mask.empty()) {
            const size_t nz = (size_t)Tc * 1536;
            HCK(hipHostMalloc((void**)&h_z, nz * 4, hipHostMallocDefault));
            HCK(hipHostMalloc((void**)&h_zcpu, nz * 4, hipHostMallocDefault));
            HCK(hipHostMalloc((void**)&h_w, (size_t)Tc * 16 * 4, hipHostMallocDefault));
            HCK(hipHostMalloc((void**)&h_idx, (size_t)Tc * 16 * 4, hipHostMallocDefault));
            d_zcpu = dalloc<float>(nz);
            HCK(hipEventCreateWithFlags(&ev_hyb, hipEventDisableTiming));
            int nth = getenv("HY4_CPU_THREADS") ? atoi(getenv("HY4_CPU_THREADS")) : CpuExperts::default_threads();
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
            if (h_use.empty()) h_use.assign((size_t)Tc * 16, 1);
            skd::moe_has_cpu() = 1;
            hyb = true;
            fprintf(stderr, "[hybrid] %llu experts (%.2f GB) in host RAM computed by %d CPU threads\n", (unsigned long long)cpu.resident_count(), cpu.arena_bytes() / 1e9, cpu.n_threads());
        }
        // STRATA_HY4_TAIL_SKIP: OFF by default (0); explicit opt-in only (1..15 skips CPU/cold-tier experts whose min rank in a verify/decode window is >= tail_skip)
        if (tail_skip <= 0) {
            const char* ts_env = std::getenv("STRATA_HY4_TAIL_SKIP");
            if (ts_env && ts_env[0] != '\0') {
                int v = std::atoi(ts_env);
                tail_skip = (v >= 1 && v <= 15) ? v : 0;
            }
        }
        if (tail_skip > 0) {
            fprintf(stderr, "[runner] STRATA_HY4_TAIL_SKIP=%d is ON (opt-in): missed CPU/cold-tier experts routed only at rank >= %d in decode/verify are skipped\n", tail_skip, tail_skip);
        }
        // kv caches (incremental 16K-token VRAM growth when HY4_KV_INIT_SLOTS is set)
        ring = ((512 + Tc + 15) / 16) * 16;
        const int kv_host_mask = getenv("HY4_KV_HOST") ? atoi(getenv("HY4_KV_HOST")) : 0;
        kv_dev_limit = getenv("HY4_KV_DEVPOS") ? atoi(getenv("HY4_KV_DEVPOS")) : 0;
        kv_is_host = (kv_host_mask & 1) != 0;
        if (kv_is_host) {
            kv_alloc_pos = max_ctx;
        } else {
            int init_p = max_ctx;
            if (getenv("HY4_KV_INIT_SLOTS")) init_p = std::min(init_p, std::max(4096, atoi(getenv("HY4_KV_INIT_SLOTS"))));
            if (kv_dev_limit > 0 && kv_dev_limit < max_ctx) init_p = std::min(init_p, kv_dev_limit);
            kv_alloc_pos = init_p;
        }
        slots_sparse = ((kv_alloc_pos + 8 + 15) / 16) * 16;
        kcache.resize(c.n_layers);
        size_t kvbytes = 0;
        for (int il = 0; il < c.n_layers; ++il) {
            int slots = c.is_swa[il] ? ring : slots_sparse;
            size_t by = mla_cache_bytes(slots);
            if (c.is_swa[il]) {
                HCK(hipMalloc((void**)&kcache[il], by));
                HCK(hipMemset(kcache[il], 0, by));
            } else if (!kv_is_host) {
                HCK(hipMalloc((void**)&kcache[il], by));
                HCK(hipMemset(kcache[il], 0, by));
            } else {
                kcache[il] = (__half*)dalloc_big<uint8_t>(by, "sparse KV cache", 1);
            }
            kvbytes += by;
        }
        // indexer state (sparse layers only; grows in lockstep with sparse KV)
        ivg.assign(c.n_layers, nullptr);
        pooled.assign(c.n_layers, nullptr);
        for (int il = 0; il < c.n_layers; ++il)
            if (!c.is_swa[il]) {
                if (!kv_is_host) {
                    ivg[il] = dalloc<__half>((size_t)(kv_alloc_pos + 8) * 256);
                    pooled[il] = dalloc<__half>((size_t)(kv_alloc_pos / 4 + 8) * 128);
                    HCK(hipMemset(ivg[il], 0, (size_t)(kv_alloc_pos + 8) * 256 * 2));
                    HCK(hipMemset(pooled[il], 0, (size_t)(kv_alloc_pos / 4 + 8) * 128 * 2));
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
        keys = dalloc<int32_t>((size_t)Tc * (index_topk + 8));
        nkeys = dalloc<int32_t>((size_t)Tc);
        tmp_pooled = dalloc<__half>((size_t)16 * 128);
        // Single-layer reusable FP16 scratch for prefill (T > 8); decode reads quantized kv_b_proj directly (saves 1.035 GB VRAM!)
        {
            const int max_NH = std::max(c.heads_swa, c.heads_sparse);
            wk_layer = dalloc<__half>((size_t)max_NH * 192 * 512);
            wv_layer = dalloc<__half>((size_t)max_NH * 256 * 512);
        }
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
        ep_alloc();
        if (hyb) vcache_init();
        if (hyb && (!getenv("HY4_PF_STREAM") || atoi(getenv("HY4_PF_STREAM")) != 0)) pf_init();  // last: the scratch takes what the other buffers left
        tune_init();
#if HY4_CUDA
        skd::cublas_pf().ensure(st);
        cuda_graph_on = getenv("HY4_CUDA_GRAPH") ? atoi(getenv("HY4_CUDA_GRAPH")) != 0 : true;
        HCK(hipHostMalloc((void**)&h_step, sizeof(DevStepArgs), hipHostMallocDefault));
        d_step = dalloc<DevStepArgs>(1);
        HCK(hipHostMalloc((void**)&h_ids, 64, hipHostMallocDefault));
        if (!hcpart) hcpart = dalloc<float>((size_t)Tc * 24 * 9);
        {
            const int nh_max = std::max(c.heads_swa, c.heads_sparse);
            const size_t need_dec_ws = std::max(mla_workspace_bytes(8, nh_max, std::max(c.window + 16, index_topk + 16)),
                                                mla_workspace_bytes(8, c.heads_sparse, max_ctx + 16));
            ensure_mla_ws(need_dec_ws);
        }
#if HY4_MMA
        if (skd::lin_int8_mma()) skd::int8_act_buf().ensure((size_t)Tc * 16384, skd::lin_int8_mma() >= 2);
#endif
        if (cuda_graph_on) fprintf(stderr, "[runner] CUDA Graph capture/replay enabled (HY4_CUDA_GRAPH=1)\n");
#endif
        size_t fr = 0, tot = 0;
        HCK(hipMemGetInfo(&fr, &tot));
        fprintf(stderr, "[runner] kv cache %.2f GB (init %d / ctx %d, chunk %d), device free %.1f GiB\n", kvbytes / 1e9, kv_alloc_pos, max_ctx, Tc, fr / 1073741824.0);
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
        TreeSpec tree{};                      // tree.M > 0: Tree-8 verify (engram_hash_tree)
        std::promise<void> pm0;
        std::future<void> fm0;
    };
    EPrep* ep_rest = nullptr;
    EPrep ep[2];
    int ep_next = 0;
    const int32_t* pf_eng_tok = nullptr;  // upfront whole-prompt Engram prefetch: kicks next chunk at Layer 11 while GPU runs Layers 11..41
    int pf_eng_total = 0;
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
        auto hash_mo = [&](int mo, uint64_t* dst_ids, uint8_t* dst_val) {
            if (e->tree.M > 0) engram_hash_tree(emeta, mo, tok_all, p0, e->tree.tok, e->tree.parent, T, dst_ids, dst_val);
            else engram_hash(emeta, mo, tok_all, p0, p0 + T, dst_ids, dst_val);
            if (vis && e->tree.M == 0) vis->mask_engram(dst_val, p0, T);
        };
        if (e->staged) {  // module 0 first (published through pm0 for Layer 1), then modules 1..3 overlapped with Layers 1..10
            const size_t per = (size_t)T * 16;
            if (T <= 8) {
                auto fetch_modules = [&](int m0, int m1) -> bool {
                    std::vector<uint64_t> ids((size_t)(m1 - m0) * per);
                    for (int mo = m0; mo < m1; ++mo) hash_mo(mo, ids.data() + (size_t)(mo - m0) * per, e->valid + (size_t)mo * Tc * 16);
                    std::vector<uint8_t> tmp(ids.size() * 100);
                    if (!eio.fetch_raw(ids.data(), ids.size(), tmp.data(), err)) return false;
                    for (int mo = m0; mo < m1; ++mo) std::memcpy(e->raw + (size_t)mo * Tc * 16 * 100, tmp.data() + (size_t)(mo - m0) * per * 100, per * 100);
                    return true;
                };
                if (!fetch_modules(0, 1)) { e->ok = false; e->err = err; e->ok_rest = false; e->pm0.set_value(); return; }
                e->pm0.set_value();
                if (!fetch_modules(1, 4)) { e->ok_rest = false; e->err = err; }
            } else {
                std::vector<uint64_t> ids(per);
                hash_mo(0, ids.data(), e->valid);
                if (!eio.fetch_raw(ids.data(), ids.size(), e->raw, err)) {
                    e->ok = false; e->err = err; e->ok_rest = false; e->pm0.set_value(); return;
                }
                e->pm0.set_value();
                for (int mo = 1; mo < 4; ++mo) {
                    hash_mo(mo, ids.data(), e->valid + (size_t)mo * Tc * 16);
                    if (!eio.fetch_raw(ids.data(), ids.size(), e->raw + (size_t)mo * Tc * 16 * 100, err)) {
                        e->ok_rest = false; e->err = err; return;
                    }
                }
            }
            return;
        }
        if (T <= 8) {  // sync decode / verify window without a prefetch: the rows of all 4 modules in ONE fetch
            const size_t per = (size_t)T * 16;
            std::vector<uint64_t> ids(4 * per);
            for (int mo = 0; mo < 4; ++mo) hash_mo(mo, ids.data() + mo * per, e->valid + (size_t)mo * Tc * 16);
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
            hash_mo(mo, ids.data(), valid);
            uint8_t* raw = e->raw + (size_t)mo * Tc * 16 * 100;
            if (!eio.fetch_raw(ids.data(), ids.size(), raw, err)) {
                e->ok = false;
                e->err = err;
                return;
            }
        }
    }
    // start fetching the rows of chunk [p0, p0+T) in the background (tokens must be final and stay valid)
    void prefetch_engram(const int32_t* tok_all, int p0, int T, const TreeSpec* tree = nullptr) {
        if (!engram_ready) return;
        EPrep& e = ep[ep_next];
        if (e.fut.valid()) e.fut.wait();
        HCK(hipEventSynchronize(e.used));
        e.p0 = p0;
        e.T = T;
        e.pending = true;
        e.staged = true;
        e.ok_rest = true;
        e.tree = tree ? *tree : TreeSpec{};
        if (e.staged) {
            e.pm0 = std::promise<void>();
            e.fm0 = e.pm0.get_future();
        }
        e.fut = std::async(std::launch::async, [this, &e, tok_all, p0, T]() { ep_compute(&e, tok_all, p0, T); });
        ep_next ^= 1;
    }
    bool engram_prep(const int32_t* tok_all, int p0, int T, std::string& err, const TreeSpec* tree = nullptr) {
        double t0 = now_s();
        EPrep* e = nullptr;
        for (int s = 0; s < 2; ++s)
            if (ep[s].pending && ep[s].p0 == p0 && ep[s].T == T && (ep[s].tree.M == (tree ? tree->M : 0))) e = &ep[s];
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
            e->tree = tree ? *tree : TreeSpec{};
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
            if (T >= 16) hipLaunchKernelGGL(k_hc_mix_part_d_batch, dim3((T + 31) / 32, 24), dim3(128), 0, st, xin, hfn, hcpart, nout, T);
            else hipLaunchKernelGGL(k_hc_mix_part_d, dim3(T, 24), dim3(128), 0, st, xin, hfn, hcpart, nout);
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
        if (det_mix) {
            if (!hcpart) hcpart = dalloc<float>((size_t)Tc * 24 * 9);
            if (T >= 16) hipLaunchKernelGGL(k_hc_mix_part_d_batch, dim3((T + 31) / 32, 24), dim3(128), 0, st, xin, hfn, hcpart, nout, T);
            else hipLaunchKernelGGL(k_hc_mix_part_d, dim3(T, 24), dim3(128), 0, st, xin, hfn, hcpart, nout);
            hipLaunchKernelGGL(k_hc_fin_reduce_norm, dim3(T), dim3(256), 0, st, hcpart, hsc, hbs, pre, post, nout, c.hc_eps, c.hc_mag, xin, nw, outh, outf, c.rms_eps);
            return;
        }
        hc_mix_run(xin, hfn, hsc, hbs, nout, T);
        hipLaunchKernelGGL(k_hc_reduce_norm, dim3(T), dim3(256), 0, st, xin, pre, nw, outh, outf, c.rms_eps);
    }

    // IndexPool: cache iv/ig of the chunk's tokens and pool the groups completed by this chunk
    void idx_update(int il, int p0, int T, const TreeSpec* tree = nullptr) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        lin(p + ".token_key_projection.weight", cur_h, ik_f, T);  // [T][128]
        hipLaunchKernelGGL(k_rms_rows, dim3(T), dim3(128), 0, st, ik_f, (const float*)tw(p + ".token_key_norm.weight"), (float*)nullptr, ik_h, 128, c.rms_eps);
        lin(p + ".key_value_projection.weight", ik_h, iv_f, T);
        lin(p + ".key_gate_projection.weight", ik_h, ig_f, T);
        const int* p0d = capturing ? &d_step->p0 : nullptr;
        hipLaunchKernelGGL(idx::k_idx_store, dim3(T), dim3(128), 0, st, iv_f, ig_f, ivg[il], p0, p0d);
        if (tree && tree->M > 0) {
            // Tree verify: compute temporary per-node pooled keys for groups >= p0/4 along each node's ancestor chain (committed later by commit_tree_path)
            const idx::IdxTree* itd = capturing ? &d_step->idx_tp : nullptr;
            hipLaunchKernelGGL(idx::k_idx_pool_tree, dim3(T, 2), dim3(128), 0, st, ivg[il], (const float*)tw(p + ".intra_group_ape"), (const float*)tw(p + ".key_norm.weight"), tmp_pooled, p0,
                               tree->to_idx_tree(), (float)theta(il), c.rms_eps, p0d, itd);
            return;
        }
        if (capturing) {
            const int ngrp = (T + 3) / 4;
            hipLaunchKernelGGL(idx::k_idx_pool, dim3(ngrp), dim3(128), 0, st, ivg[il], (const float*)tw(p + ".intra_group_ape"), (const float*)tw(p + ".key_norm.weight"), pooled[il], 0,
                               (float)theta(il), c.rms_eps, p0d, T);
        } else {
            int g_lo = p0 / 4, g_hi = (p0 + T) / 4 - 1;
            if (g_hi >= g_lo)
                hipLaunchKernelGGL(idx::k_idx_pool, dim3(g_hi - g_lo + 1), dim3(128), 0, st, ivg[il], (const float*)tw(p + ".intra_group_ape"), (const float*)tw(p + ".key_norm.weight"), pooled[il], g_lo,
                                   (float)theta(il), c.rms_eps, (const int*)nullptr, 0);
            if (vis && g_hi >= g_lo && vis->needs_fix(p0, T)) vis->pool_fix(st, pooled[il], g_lo, g_hi - g_lo + 1, (float)theta(il));  // HY4-VISION hook
        }
    }

    // per-token key lists for the chunk (sparse layer, sequence longer than index_topk)
    void idx_select(int il, int p0, int T, const TreeSpec* tree = nullptr) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        const int* p0d = capturing ? &d_step->p0 : nullptr;
        const int* Gd = capturing ? &d_step->idx_G : nullptr;
        lin(p + ".query_projection.weight", qr_h, iq_f, T);  // [T][16*128]
        if (tree && tree->M > 0) {
            const idx::IdxTree it = tree->to_idx_tree();
            const idx::IdxTree* itd = capturing ? &d_step->idx_tp : nullptr;
            hipLaunchKernelGGL(idx::k_idx_q_rope_tree, dim3(T, 16), dim3(128), 0, st, iq_f, iq_h, p0, it, (float)theta(il), p0d, itd);
            lin(p + ".head_weight_projection.weight", cur_h, hw_f, T);
            int G = (p0 + tree->max_depth() + 1) / 4;
            if (G < 1) G = 1;
            const int sg = capturing ? cap_score_grid : ((G + 63) / 64);
            hipLaunchKernelGGL(idx::k_idx_score_tree, dim3(sg, T), dim3(64), 0, st, iq_h, hw_f, pooled[il], tmp_pooled, score, p0, it, G, (int)ldS, 1.0f / sqrtf((float)(128 * 16)), p0d, Gd, itd);
            int n_top = index_topk / c.kpool - 1;
            hipLaunchKernelGGL(idx::k_idx_select_tree, dim3(T), dim3(256), 0, st, score, (int)ldS, p0, it, n_top, index_topk, keys, nkeys, p0d, itd);
            return;
        }
        if (!capturing && vis && vis->needs_fix(p0, T)) vis->iq_rope(st, iq_f, iq_h, T, p0, (float)theta(il));  // HY4-VISION hook: 3D positions
        else hipLaunchKernelGGL(idx::k_idx_q_rope, dim3(T, 16), dim3(128), 0, st, iq_f, iq_h, p0, (float)theta(il), p0d);
        lin(p + ".head_weight_projection.weight", cur_h, hw_f, T);  // [T][16]
        int G = (p0 + T) / 4;
        if (G < 1) G = 1;
        const int sg = capturing ? cap_score_grid : ((G + 63) / 64);
        hipLaunchKernelGGL(idx::k_idx_score, dim3(sg, T), dim3(64), 0, st, iq_h, hw_f, pooled[il], score, p0, G, (int)ldS, 1.0f / sqrtf((float)(128 * 16)), p0d, Gd);
        int n_top = index_topk / c.kpool - 1;
        hipLaunchKernelGGL(idx::k_idx_select, dim3(T), dim3(256), 0, st, score, (int)ldS, p0, n_top, index_topk, keys, nkeys, p0d);
    }

    void dequant_kvb_layer(int il) {
        const int NH = heads(il);
        const std::string nm = L(il) + ".self_attn.kv_b_proj.weight";
        const TInfo& t = ti(nm);
        const uint8_t* W = (const uint8_t*)tw(nm);
#if !HY4_CUDA
        dim3 gk(NH * 6, 16), gv(NH * 256), blk(256);
        switch (t.fmt) {
            case FMT_S2: hipLaunchKernelGGL((k_dequant_wkt<FMT_S2>), gk, blk, 0, st, W, wk_layer, NH); hipLaunchKernelGGL((k_dequant_wv<FMT_S2>), gv, blk, 0, st, W, wv_layer, NH); break;
            case FMT_S3: hipLaunchKernelGGL((k_dequant_wkt<FMT_S3>), gk, blk, 0, st, W, wk_layer, NH); hipLaunchKernelGGL((k_dequant_wv<FMT_S3>), gv, blk, 0, st, W, wv_layer, NH); break;
            case FMT_S4: hipLaunchKernelGGL((k_dequant_wkt<FMT_S4>), gk, blk, 0, st, W, wk_layer, NH); hipLaunchKernelGGL((k_dequant_wv<FMT_S4>), gv, blk, 0, st, W, wv_layer, NH); break;
            case FMT_S5: hipLaunchKernelGGL((k_dequant_wkt<FMT_S5>), gk, blk, 0, st, W, wk_layer, NH); hipLaunchKernelGGL((k_dequant_wv<FMT_S5>), gv, blk, 0, st, W, wv_layer, NH); break;
            case FMT_S6: hipLaunchKernelGGL((k_dequant_wkt<FMT_S6>), gk, blk, 0, st, W, wk_layer, NH); hipLaunchKernelGGL((k_dequant_wv<FMT_S6>), gv, blk, 0, st, W, wv_layer, NH); break;
            default:     hipLaunchKernelGGL((k_dequant_wkt<FMT_S8>), gk, blk, 0, st, W, wk_layer, NH); hipLaunchKernelGGL((k_dequant_wv<FMT_S8>), gv, blk, 0, st, W, wv_layer, NH); break;
        }
#else
        dim3 grid(NH * 448), block(128);
        switch (t.fmt) {
            case FMT_S2: hipLaunchKernelGGL((k_split_kvb<FMT_S2>), grid, block, 0, st, W, wk_layer, wv_layer, NH); break;
            case FMT_S3: hipLaunchKernelGGL((k_split_kvb<FMT_S3>), grid, block, 0, st, W, wk_layer, wv_layer, NH); break;
            case FMT_S4: hipLaunchKernelGGL((k_split_kvb<FMT_S4>), grid, block, 0, st, W, wk_layer, wv_layer, NH); break;
            case FMT_S5: hipLaunchKernelGGL((k_split_kvb<FMT_S5>), grid, block, 0, st, W, wk_layer, wv_layer, NH); break;
            case FMT_S6: hipLaunchKernelGGL((k_split_kvb<FMT_S6>), grid, block, 0, st, W, wk_layer, wv_layer, NH); break;
            default:     hipLaunchKernelGGL((k_split_kvb<FMT_S8>), grid, block, 0, st, W, wk_layer, wv_layer, NH); break;
        }
#endif
    }

    void attention_layer(int il, int p0, int T, const TreeSpec* tree = nullptr) {
        const std::string pa = L(il) + ".self_attn";
        const int NH = heads(il);
        const bool swa = c.is_swa[il];
        const int* p0d = capturing ? &d_step->p0 : nullptr;
        const bool is_tree = tree != nullptr && tree->M > 0;
        const TreePos tp = is_tree ? tree->to_mla_pos() : TreePos{};
        // iHC pre + input_layernorm
        const float* hfn = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_base");
        { PS _q(this, "hc_mix"); hc_pre(x, hfn, hsc, hbs, (const float*)tw(L(il) + ".input_layernorm.weight"), 8, T, cur_h, (float*)nullptr); }
        // q + kv_a + linear_gate: fuse into ONE LinMulti launch on decode/verify (T <= 8) when formats match
        const bool fused_attn = lm_ok_m(T) && il < (int)lm_attn.size() && lm_attn[il].d_jobs != nullptr;
        const float* kv_a_out = fused_attn ? (F1 + 8192) : F1;
        const bool w16_ok = !dump && Tc >= 2048;
        if (fused_attn) {
            { PS _q(this, "lin attn_qkvg_multi"); linear_h_multi(st, lm_attn[il], cur_h, T); }
            { PS _q(this, "q_a_rms"); hipLaunchKernelGGL(k_rms_rows, dim3(T), dim3(256), 0, st, F1, (const float*)tw(pa + ".q_a_layernorm.weight"), (float*)nullptr, qr_h, c.q_lora, c.rms_eps); }
            lin(pa + ".q_b_proj.weight", qr_h, F2, T);  // [T][NH*256]
        } else {
            lin(pa + ".q_a_proj.weight", cur_h, F1, T, w16_ok ? (__half*)F2 : nullptr);  // [T][1024]
            { PS _q(this, "q_a_rms"); hipLaunchKernelGGL(k_rms_rows, dim3(T), dim3(256), 0, st, F1, (const float*)tw(pa + ".q_a_layernorm.weight"), (float*)nullptr, qr_h, c.q_lora, c.rms_eps); }
            lin(pa + ".q_b_proj.weight", qr_h, F2, T, w16_ok ? (__half*)F1 : nullptr);  // [T][NH*256]
            lin(pa + ".kv_a_proj_with_mqa.weight", cur_h, F1, T, w16_ok ? (__half*)(F1 + (size_t)T * 1024) : nullptr);  // [T][576]  (qr_h already extracted)
        }
        __half* cache = kcache[il];
        const int cslots = swa ? ring : slots_sparse;
        const TreePos* tpd = (capturing && is_tree) ? &d_step->mla_tp : nullptr;
        { PS _q(this, "mla_write_kv");
          if (p0d) mla_write_kv_d(st, kv_a_out, (const float*)tw(pa + ".kv_a_layernorm.weight"), cache, cslots, T, p0d, swa ? ring : 0, (float)theta(il), c.rms_eps, tpd);
          else mla_write_kv(st, kv_a_out, (const float*)tw(pa + ".kv_a_layernorm.weight"), cache, cslots, T, p0, swa ? ring : 0, (float)theta(il), c.rms_eps, tp); }
        const bool dec = T <= 8;
        const int kvfmt = ti(pa + ".kv_b_proj.weight").fmt;
        if (!dec) { PS _q(this, "dequant_kvb"); dequant_kvb_layer(il); }
        { PS _q(this, "mla_prep_q");
          if (dec) {
              if (p0d) mla_prep_q_dec_d(st, kvfmt, tw(pa + ".kv_b_proj.weight"), F2, qa_h, T, NH, p0d, (float)theta(il), tpd);
              else mla_prep_q_dec(st, kvfmt, tw(pa + ".kv_b_proj.weight"), F2, qa_h, T, NH, p0, (float)theta(il), tp);
          } else {
#if !HY4_CUDA
              mla_prep_q_t(st, F2, wk_layer, qa_h, T, NH, p0, (float)theta(il));
#else
              mla_prep_q(st, F2, wk_layer, qa_h, T, NH, p0, (float)theta(il));
#endif
          }
        }
        if (!is_tree && !capturing && vis && vis->needs_fix(p0, T)) {  // HY4-VISION hook: rope of q and of the new keys with the picture-aware (t, h, w) positions
            vis->fix_k(st, kv_a_out, cache, T, p0, swa ? ring : 0, (float)theta(il));
            vis->fix_q(st, F2, qa_h, T, NH, p0, (float)theta(il));
        }
        bool use_list = false;
        if (!swa && index_on) {
            { PS _q(this, "idx_update"); idx_update(il, p0, T, tree); }
            const int end_pos = is_tree ? (p0 + tree->max_depth() + 1) : (p0 + T);
            if (end_pos > index_topk) {
                { PS _q(this, "idx_select"); idx_select(il, p0, T, tree); }
                use_list = true;
            }
        }
        last_use_list = use_list;
        if (dfly_cap && !swa) dfly_capture(il, p0, T, use_list);
        if (!fused_attn) lin(pa + ".linear_gate.weight", cur_h, F3, T, w16_ok ? (__half*)F2 : nullptr);  // [T][NH*256]
        const float* gate_buf = fused_attn ? F3_dec : F3;
        MlaAttnArgs a;
        a.M = T;
        a.NH = NH;
        a.p0 = p0;
        a.p0_dev = p0d;
        a.mode = swa ? MLA_WINDOW : (use_list ? MLA_LIST : MLA_CAUSAL);
        a.window = c.window;
        a.ring = swa ? ring : 0;
        a.cache_slots = cslots;
        if (is_tree) {
            a.tree_nodes = tree->M;
            if (capturing) {
                a.tree_dev = &d_step->mla_tree;
            } else {
                for (int i = 0; i < tree->M; ++i) {
                    a.tree_depth[i] = tree->depth[i];
                    for (int d = 0; d < 8; ++d) a.tree_anc[i][d] = tree->anc[i][d];
                }
            }
        }
        if (use_list) {
            a.keys = keys;
            a.nkeys = nkeys;
            a.kmax = index_topk;
        }
        if (!capturing) ensure_mla_ws(mla_workspace_bytes(a));
        if (dec) {
            { PS _q(this, "mla_attend"); mla_attend(st, qa_h, cache, (const float*)tw(pa + ".learnable_sink_param"), (__half*)ctx_ws, a, mla_ws, mla_ws_bytes); }
            { PS _q(this, "mla_vup"); mla_vup_dec(st, kvfmt, tw(pa + ".kv_b_proj.weight"), (const __half*)ctx_ws, gate_buf, o_h, T, NH); }
            if (dump && il == dump_layer) {
                dump_half("in_cur_h", cur_h, (size_t)T * HID, T);
                maybe_dump("in_qb", T, NH * 256, F2);
                dump_half("in_qa", qa_h, (size_t)T * NH * 576, T);
                dump_half("in_ctx", (const __half*)ctx_ws, (size_t)T * NH * 512, T);
                dump_half("in_o", o_h, (size_t)T * NH * 256, T);
                maybe_dump("in_gate", T, NH * 256, gate_buf);
            }
        } else {
            { PS _q(this, "mla_attend_vup_prefill"); mla_attention(st, qa_h, cache, (const float*)tw(pa + ".learnable_sink_param"), wv_layer, gate_buf, o_h, ctx_ws, a, mla_ws, mla_ws_bytes); }
        }
        lin(pa + ".o_proj.weight", o_h, F1, T, w16_ok ? (__half*)F2 : nullptr);  // [T][3072]
        if (dump && il == dump_layer) maybe_dump("in_oproj", T, HID, F1);
        { PS _q(this, "o_norm_post"); hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F1, (const float*)nullptr, (const float*)tw(pa + ".post_proj_norm.weight"), x, post, c.rms_eps); }
    }

    // MoE after the router and the latent down projection (z_f = [T][1536] fp32, moe_idx / moe_w on the device) for a layer whose experts are split between the GPU and the CPU.
    // Order: copy z / idx / w to the host, queue everything the GPU can do without the CPU experts (VRAM experts, shared experts), then run the CPU experts while the GPU works,
    // add their partial sum into zout_f and finish (up projection, combine).
    void moe_hybrid_tail(int il, int T, const std::string& pm) {
        const size_t nz = (size_t)T * 1536;
        float* F3m = dump ? F3 : (F1 + (size_t)T * 1536);
        void* mws = T <= 8 ? moe_ws_dec : moe_ws;
        if (pf_active(T)) {
            const int ml = il - 1;
            const bool do_pf_cpu = has_pf_cpu && !pf_use_gather(T);
            const bool need_cpu_pass = has_cold || do_pf_cpu;
            const bool need_idx_host = need_cpu_pass || pf_use_gather(T) || (vcache_nslots > 1 && !vcache_frozen);
            if (need_idx_host) {
                if (need_cpu_pass) {
                    HCK(hipMemcpyAsync(h_z, z_f, nz * 4, hipMemcpyDeviceToHost, st));
                    HCK(hipMemcpyAsync(h_w, moe_w, (size_t)T * 16 * 4, hipMemcpyDeviceToHost, st));
                }
                HCK(hipMemcpyAsync(h_idx, moe_idx, (size_t)T * 16 * 4, hipMemcpyDeviceToHost, st));
                HCK(hipEventRecord(ev_hyb, st));
            }
            { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, z_f, z_h, nz); }
            peers_launch(il, T);
            bool cpu_submitted = false;
            double t_cpu0 = 0.0;
            if (pf_use_gather(T)) {
                // Active-expert gather (pf_min_T <= T < pf_gather_max_T): only DMA host experts with cnt[e] > 0 that are NOT already in VRAM/vcache!
                // Uses pf_pool (8 threads) for DDR5 -> pinned bounce packing and pipelines batch bi+1 packing while GPU executes batch bi!
                HCK(hipEventSynchronize(ev_hyb));
                int ecnt[512] = {};
                for (size_t i = 0; i < (size_t)T * 16; ++i) {
                    const int e = h_idx[i];
                    if (e >= 0 && e < 512) {
                        ++ecnt[e];
                        if (vcache_nslots > 1 && !vcache_frozen) vscore[ml * 512 + e] += vcache_inv_scale;
                    }
                }
                auto al = [](size_t x) { return (x + 4095) & ~(size_t)4095; };
                std::vector<std::pair<const uint8_t*, int>> need;
                need.reserve(256);
                for (int e = 0; e < 512; ++e) {
                    if (ecnt[e] > 0 && g.experts[ml * 512 + e].tier == 2 && !vram_flag[ml * 512 + e] && cpu.resident(ml, e)) {
                        CpuExpertView vw = cpu.view(ml, e);
                        if (vw.body) need.push_back({vw.body, e});
                    }
                }
                std::sort(need.begin(), need.end());  // sequential DDR5 address order + coalesced runs
                std::vector<PfGroup> gbatches;
                std::vector<std::vector<std::pair<int, size_t>>> batch_eo;
                gbatches.emplace_back();
                batch_eo.emplace_back();
                size_t cur_off = 0;
                for (const auto& pe : need) {
                    CpuExpertView vw = cpu.view(ml, pe.second);
                    const size_t sz = al(vw.body_bytes);
                    if (cur_off + sz > pf_cap && cur_off > 0) {
                        gbatches.back().total_bytes = cur_off;
                        gbatches.emplace_back();
                        batch_eo.emplace_back();
                        cur_off = 0;
                    }
                    PfGroup& G = gbatches.back();
                    if (!G.runs.empty() && G.runs.back().host + G.runs.back().bytes == pe.first) G.runs.back().bytes += sz;
                    else G.runs.push_back({pe.first, cur_off, sz});
                    batch_eo.back().push_back({pe.second, cur_off});
                    cur_off += sz;
                }
                gbatches.back().total_bytes = cur_off;
                const int nb = (int)gbatches.size();
                pf_pool.wait();
                pf_packed_seq[0] = pf_packed_seq[1] = -1;
                if (!gbatches[0].runs.empty()) {
                    pf_pool.kick(0, h_bounce[0], &gbatches[0], ev_pf_copy_set[0] ? ev_pf_copy[0] : nullptr);
                }
                // Group 0 runs all VRAM/vcache experts + first batch of active host experts; subsequent batches add into zout_f
                for (int bi = 0; bi < nb; ++bi) {
                    const int b = bi % pf_nbuf;
                    if (!gbatches[bi].runs.empty()) {
                        pf_pool.wait();
                        if (ev_pf_done_set[b]) HCK(hipStreamWaitEvent(st2, ev_pf_done[b], 0));
                        HCK(hipMemcpyAsync(d_scratch[b], h_bounce[b], gbatches[bi].total_bytes, hipMemcpyHostToDevice, st2));
                        HCK(hipEventRecord(ev_pf_copy[b], st2));
                        ev_pf_copy_set[b] = true;
                        HCK(hipStreamWaitEvent(st, ev_pf_copy[b], 0));
                    }
                    // Lookahead parallel pack of batch bi+1 into the alternate bounce buffer while GPU executes batch bi
                    if (pf_nbuf > 1 && bi + 1 < nb && !gbatches[bi + 1].runs.empty()) {
                        const int bn = (bi + 1) % pf_nbuf;
                        pf_pool.kick(bn, h_bounce[bn], &gbatches[bi + 1], ev_pf_copy_set[bn] ? ev_pf_copy[bn] : nullptr);
                    }
                    if (ev_pf_done_set[b]) HCK(hipEventSynchronize(ev_pf_done[b]));  // safe to overwrite h_pf_dyn_table[b]
                    ExpertDev* tbl = h_pf_dyn_table[b];
                    for (int e = 0; e < 512; ++e) {
                        const int gi = ml * 512 + e;
                        tbl[e] = h_experts[gi];
                        if (bi > 0 || !vram_flag[gi]) tbl[e].pad = 1;
                    }
                    for (const auto& eo : batch_eo[bi]) {
                        tbl[eo.first].body_off = (uint64_t)(uintptr_t)(d_scratch[b] + eo.second) - (uint64_t)(uintptr_t)g.d_sec4;
                        tbl[eo.first].pad = 0;
                    }
                    HCK(hipMemcpyAsync(d_pf_dyn_table[b], tbl, 512 * sizeof(ExpertDev), hipMemcpyHostToDevice, st));
                    { PS _q(this, "moe_experts(pf_gather)"); moe_experts(st, g.d_sec4, d_pf_dyn_table[b], z_h, moe_idx, moe_w, bi == 0 ? zout_f : d_zcpu, T, mws); }
                    if (bi > 0) hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)d_zcpu, nz);
                    HCK(hipEventRecord(ev_pf_done[b], st));
                    ev_pf_done_set[b] = true;
                }
                pf_pool.wait();
            } else {
                // Full-layer pipelined streaming (T >= pf_gather_max_T, e.g. Tc=4096/8192): DMA overlaps attention + earlier groups, skipping VRAM/vcache experts!
                const std::vector<PfGroup>& gs = pf_groups[ml];
                for (size_t gi = 0; gi < gs.size(); ++gi) {
                    const PfGroup& G = gs[gi];
                    const int k = G.seq, b = k % pf_nbuf;
                    pf_pump(k);
                    if (!G.runs.empty()) HCK(hipStreamWaitEvent(st, ev_pf_copy[b], 0));
                    { PS _q(this, "moe_experts(pf)"); moe_experts(st, g.d_sec4, G.table, z_h, moe_idx, moe_w, gi == 0 ? zout_f : d_zcpu, T, mws); }
                    if (gi > 0) hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)d_zcpu, nz);
                    HCK(hipEventRecord(ev_pf_done[b], st));
                    ev_pf_done_set[b] = true;
                    ++pf_next_comp;
                    pf_pump(k + pf_nbuf);
                    // Launch parallel CPU subset (cold + pf_cpu_flag) right after group 0 is queued so CPU runs concurrently with remaining PCIe+GPU groups!
                    if (gi == 0 && need_cpu_pass) {
                        t_cpu0 = now_s();
                        HCK(hipEventSynchronize(ev_hyb));
                        for (size_t i = 0; i < (size_t)T * 16; ++i) {
                            const int e = h_idx[i];
                            uint8_t u = 0;
                            if (e >= 0 && e < 512) {
                                if (has_cold && cold_flag[(size_t)ml * 512 + (size_t)e]) u = 1;
                                if (do_pf_cpu && pf_cpu_flag[(size_t)ml * 512 + (size_t)e]) u = 1;
                                if (vcache_nslots > 1 && !vcache_frozen) vscore[ml * 512 + e] += vcache_inv_scale;
                            }
                            h_use[i] = u;
                        }
                        cpu.submit(ml, T, h_z, h_idx, h_w, h_use.data(), h_zcpu);
                        cpu_submitted = true;
                    }
                }
                if (need_idx_host && !need_cpu_pass) {
                    HCK(hipEventSynchronize(ev_hyb));
                    for (size_t i = 0; i < (size_t)T * 16; ++i) {
                        const int e = h_idx[i];
                        if (e >= 0 && e < 512) vscore[ml * 512 + e] += vcache_inv_scale;
                    }
                }
            }
            peers_gather(T);
            if (need_cpu_pass) {
                if (!cpu_submitted) {
                    t_cpu0 = now_s();
                    if (!pf_use_gather(T)) HCK(hipEventSynchronize(ev_hyb));
                    for (size_t i = 0; i < (size_t)T * 16; ++i) {
                        const int e = h_idx[i];
                        h_use[i] = (e >= 0 && e < 512 && has_cold) ? cold_flag[(size_t)ml * 512 + (size_t)e] : 0;
                        if (!pf_use_gather(T) && vcache_nslots > 1 && !vcache_frozen && e >= 0 && e < 512) vscore[ml * 512 + e] += vcache_inv_scale;
                    }
                    cpu.submit(ml, T, h_z, h_idx, h_w, h_use.data(), h_zcpu);
                }
                cpu.wait();
                t_cpu_wait += now_s() - t_cpu0;
                HCK(hipMemcpyAsync(d_zcpu, h_zcpu, nz * 4, hipMemcpyHostToDevice, st));
                hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)d_zcpu, nz);
            }
            { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, z_h, nz); }
            const bool w16_ok = !dump && Tc >= 2048;
            lin(pm + ".routed_expert_up_proj.weight", z_h, F2, T, w16_ok ? (__half*)F1 : nullptr);  // [T][3072]
            lin(pm + ".shared_experts.gate_proj.weight", cur_h, F1, T, w16_ok ? (__half*)(F1 + (size_t)T * 5120) : nullptr);
            lin(pm + ".shared_experts.up_proj.weight", cur_h, F3m, T, w16_ok ? (__half*)(F1 + (size_t)T * 5120) : nullptr);
            { PS _q(this, "swiglu"); hipLaunchKernelGGL(k_swiglu_h, dim3((nz + 255) / 256), dim3(256), 0, st, F1, F3m, H1, nz); }
            lin(pm + ".shared_experts.down_proj.weight", H1, F3m, T, w16_ok ? (__half*)(F1 + (size_t)T * 5120) : nullptr);
            { PS _q(this, "combine_norm_post"); hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F2, (const float*)F3m, (const float*)tw(pm + ".combine_output_norm.weight"), x, post, c.rms_eps); }
            return;
        }
        const int ml = il - 1;
        HCK(hipMemcpyAsync(h_z, z_f, nz * 4, hipMemcpyDeviceToHost, st));
        HCK(hipMemcpyAsync(h_idx, moe_idx, (size_t)T * 16 * 4, hipMemcpyDeviceToHost, st));
        HCK(hipMemcpyAsync(h_w, moe_w, (size_t)T * 16 * 4, hipMemcpyDeviceToHost, st));
        HCK(hipEventRecord(ev_hyb, st));
        { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, z_f, z_h, nz); }
        peers_launch(il, T);
        { PS _q(this, "moe_experts(gpu part)"); moe_experts(st, g.d_sec4, d_experts + (size_t)ml * 512, z_h, moe_idx, moe_w, zout_f, T, mws); }
        const bool use_sh = lm_ok_m(T) && ml < (int)lm_sh.size() && lm_sh[ml].d_jobs;
        float* F3s = use_sh ? F3_dec : F3m;
        if (use_sh) {
            PS _q(this, "lin shared_gu_multi"); linear_h_multi(st, lm_sh[ml], cur_h, T);
        } else {
            lin(pm + ".shared_experts.gate_proj.weight", cur_h, F1, T);  // [T][1536]
            lin(pm + ".shared_experts.up_proj.weight", cur_h, F3s, T);
        }
        { PS _q(this, "swiglu"); hipLaunchKernelGGL(k_swiglu_h, dim3((nz + 255) / 256), dim3(256), 0, st, F1, F3s, H1, nz); }
        lin(pm + ".shared_experts.down_proj.weight", H1, F3s, T);  // [T][3072]
        double t0 = now_s();
        HCK(hipEventSynchronize(ev_hyb));
        const uint8_t* use_ptr = nullptr;
        const bool do_tail = (tail_skip > 0 && T <= 16);
        if (vcache_nslots > 1 || do_tail) {
            int minr[512];
            if (do_tail) {
                for (int e = 0; e < 512; ++e) minr[e] = 99;
                for (int t = 0; t < T; ++t)
                    for (int s = 0; s < 16; ++s) {
                        const int e = h_idx[(size_t)t * 16 + s];
                        if (e >= 0 && e < 512 && s < minr[e]) minr[e] = s;
                    }
            }
            for (int t = 0; t < T; ++t)
                for (int s = 0; s < 16; ++s) {
                    const size_t i = (size_t)t * 16 + s;
                    const int e = h_idx[i];
                    uint8_t u = (e >= 0 && e < 512 && !(vcache_nslots > 1 && vram_flag[ml * 512 + e])) ? 1 : 0;
                    if (u && do_tail && g.experts[(size_t)ml * 512 + e].tier >= 2 && minr[e] >= tail_skip) {
                        u = 0;
                        ++tail_skipped_pairs;
                    }
                    h_use[i] = u;
                }
            use_ptr = h_use.data();
        }
        cpu.submit(ml, T, h_z, h_idx, h_w, use_ptr, h_zcpu);
        if (vcache_nslots > 1) vcache_note_decode(ml, T, h_idx);  // hidden behind CPU expert execution!
        cpu.wait();
        t_cpu_wait += now_s() - t0;
        peers_gather(T);
        HCK(hipMemcpyAsync(d_zcpu, h_zcpu, nz * 4, hipMemcpyHostToDevice, st));
        hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)d_zcpu, nz);
        { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, z_h, nz); }
        lin(pm + ".routed_expert_up_proj.weight", z_h, F2, T);  // [T][3072]
        { PS _q(this, "combine_norm_post"); hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F2, (const float*)F3s, (const float*)tw(pm + ".combine_output_norm.weight"), x, post, c.rms_eps); }
    }

    void mlp_layer(int il, int T) {
        const std::string pm = L(il) + ".mlp";
        const float* hfn = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_base");
        { PS _q(this, "hc_mix"); hc_pre(x, hfn, hsc, hbs, (const float*)tw(L(il) + ".post_attention_layernorm.weight"), 8, T, cur_h, cur_f); }
        const bool w16_ok = !dump && Tc >= 2048;
        if (il == 0) {
            if (lm_ok_m(T) && lm_mlp0.d_jobs) {
                PS _q(this, "lin mlp0_gu_multi"); linear_h_multi(st, lm_mlp0, cur_h, T);
            } else {
                lin(pm + ".gate_proj.weight", cur_h, F1, T, w16_ok ? (__half*)F2 : nullptr);  // [T][13824]
                lin(pm + ".up_proj.weight", cur_h, F2, T);
            }
            size_t n = (size_t)T * 13824;
            { PS _q(this, "swiglu"); hipLaunchKernelGGL(k_swiglu_h, dim3((n + 255) / 256), dim3(256), 0, st, F1, F2, H1, n); }
            lin(pm + ".down_proj.weight", H1, F3, T, w16_ok ? (__half*)F2 : nullptr);  // [T][3072]
            { PS _q(this, "o_norm_post"); hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F3, (const float*)nullptr, (const float*)tw(pm + ".post_fc2_norm.weight"), x, post, c.rms_eps); }
        } else {
            // router
            {
                PS _q(this, "router_gemv");
                const float* rw = (const float*)tw(pm + ".gate.weight");
                if (T <= 1) hipLaunchKernelGGL((k_f32_gemv_t<1>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
                else if (T <= 2) hipLaunchKernelGGL((k_f32_gemv_t<2>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
                else if (T <= 4) hipLaunchKernelGGL((k_f32_gemv_t<4>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
                else if (T <= 6) hipLaunchKernelGGL((k_f32_gemv_t<6>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
                else if (T <= 8) hipLaunchKernelGGL((k_f32_gemv_t<8>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
                else if (T <= 16) hipLaunchKernelGGL(k_f32_gemv, dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
                else hipLaunchKernelGGL(k_f32_gemm, dim3(8, (T + 63) / 64), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
            }
            route_now = (tune & TN_ROUTE) && T == 1 && !hyb && peer_runs.empty();
            {
                PS _q(this, "router_topk");
                const uint4* exl = route_now ? (const uint4*)(d_experts + (size_t)(il - 1) * 512) : nullptr;
                uint4* rec = route_now ? (uint4*)slot_rec : nullptr;
                const float* rb = (const float*)tw(pm + ".gate.e_score_correction_bias");
                if (tune & TN_TOPK) hipLaunchKernelGGL(k_router_topk_w32, dim3(T), dim3(32), 0, st, F1, rb, moe_idx, moe_w, 16, c.routed_scale, exl, rec);
                else hipLaunchKernelGGL(k_router_topk, dim3(T), dim3(256), 0, st, F1, rb, moe_idx, moe_w, 512, 16, c.routed_scale, exl, rec);
            }
            size_t nz = (size_t)T * 1536;
            lin(pm + ".routed_expert_down_proj.weight", cur_h, z_f, T, w16_ok ? (__half*)F2 : nullptr);  // [T][1536]
            if (hyb) { moe_hybrid_tail(il, T, pm); } else {
            void* mws = T <= 8 ? moe_ws_dec : moe_ws;
            const bool nof = T <= 4 && (tune & TN_NOF2H) && peer_runs.empty();  // fp32 latent read directly by the expert kernels, fp16 result written by their reduce
            if (nof) { PS _q(this, "moe_experts"); moe_experts(st, g.d_sec4, d_experts + (size_t)(il - 1) * 512, nullptr, moe_idx, moe_w, zout_f, T, mws, z_f, z_h, route_now ? slot_rec : nullptr); }
            else {
            { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, z_f, z_h, nz); }
            peers_launch(il, T);
            { PS _q(this, "moe_experts"); moe_experts(st, g.d_sec4, d_experts + (size_t)(il - 1) * 512, z_h, moe_idx, moe_w, zout_f, T, mws, nullptr, nullptr, route_now ? slot_rec : nullptr); }
            peers_gather(T);
            { PS _q(this, "f2h"); hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, z_h, nz); }
            }
            lin(pm + ".routed_expert_up_proj.weight", z_h, F2, T, w16_ok ? (__half*)F1 : nullptr);  // [T][3072]
            const bool use_sh = lm_ok_m(T) && (il - 1) < (int)lm_sh.size() && lm_sh[il - 1].d_jobs;
            float* F3m = use_sh ? F3_dec : (dump ? F3 : (F1 + (size_t)T * 1536));
            if (use_sh) {
                PS _q(this, "lin shared_gu_multi"); linear_h_multi(st, lm_sh[il - 1], cur_h, T);
            } else {
                lin(pm + ".shared_experts.gate_proj.weight", cur_h, F1, T, w16_ok ? (__half*)(F1 + (size_t)T * 5120) : nullptr);  // [T][1536]
                lin(pm + ".shared_experts.up_proj.weight", cur_h, F3m, T, w16_ok ? (__half*)(F1 + (size_t)T * 5120) : nullptr);
            }
            { PS _q(this, "swiglu"); hipLaunchKernelGGL(k_swiglu_h, dim3((nz + 255) / 256), dim3(256), 0, st, F1, F3m, H1, nz); }
            lin(pm + ".shared_experts.down_proj.weight", H1, F3m, T, w16_ok ? (__half*)(F1 + (size_t)T * 5120) : nullptr);  // [T][3072]
            { PS _q(this, "combine_norm_post"); hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F2, (const float*)F3m, (const float*)tw(pm + ".combine_output_norm.weight"), x, post, c.rms_eps); }
            }
        }
    }

    // Spill sparse KV + IndexPool to pinned mapped host memory (full max_ctx, once) when VRAM cannot grow further
    void kv_spill_to_host() {
        if (kv_is_host) return;
        const double t0 = now_s();
        HCK(hipDeviceSynchronize());
        const size_t os = (size_t)slots_sparse, ns = (size_t)((max_ctx + 8 + 15) / 16) * 16;
        const size_t row_b = (size_t)576 * 2;  // MLA_ROW halves (no Vt)
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
            HCK(hipMemcpy(niv, ivg[il], (size_t)(kv_alloc_pos + 8) * 256 * 2, hipMemcpyDeviceToDevice));
            HCK(hipFree(ivg[il]));
            ivg[il] = niv;
            void* hp3 = nullptr;
            __half* npl = (__half*)host_alloc((size_t)(max_ctx / 4 + 8) * 128 * 2, &hp3);
            HCK(hipMemcpy(npl, pooled[il], (size_t)(kv_alloc_pos / 4 + 8) * 128 * 2, hipMemcpyDeviceToDevice));
            HCK(hipFree(pooled[il]));
            pooled[il] = npl;
        }
        kv_alloc_pos = max_ctx;
        slots_sparse = (int)ns;
        kv_is_host = true;
        ++kv_gen;
#if HY4_CUDA
        cg_clear();
#endif
        fprintf(stderr, "[runner] sparse KV + IndexPool moved to pinned host memory (%d positions, %.2f GB, %.2f s)\n", max_ctx, host_big_bytes / 1e9, now_s() - t0);
    }
    // Grow sparse KV + IndexPool in 16,384-token VRAM increments; fall back cleanly to pinned host memory if VRAM runs out
    void kv_grow_to(int need) {
        if (kv_is_host || need <= kv_alloc_pos) return;
        const int target = std::min(max_ctx, ((std::max(need + 16, kv_alloc_pos + 16384) + 16383) / 16384) * 16384);
        if (target <= kv_alloc_pos) return;
        if (kv_dev_limit > 0 && target > kv_dev_limit) { kv_spill_to_host(); return; }
        const double t0 = now_s();
        HCK(hipDeviceSynchronize());
        const size_t os = (size_t)slots_sparse, ns = (size_t)((target + 8 + 15) / 16) * 16;
        const size_t row_b = (size_t)576 * 2;
        const size_t delta_per_layer = (ns - os) * row_b + (size_t)(target - kv_alloc_pos) * 256 * 2 + (size_t)(target / 4 - kv_alloc_pos / 4) * 128 * 2;
        const size_t peak_one = ns * row_b + (size_t)(target + 8) * 256 * 2 + (size_t)(target / 4 + 8) * 128 * 2;
        size_t fr = 0, tot = 0;
        HCK(hipMemGetInfo(&fr, &tot));
        const size_t need_vram = delta_per_layer * 12 + peak_one + (128ull << 20);
        if (fr < need_vram) {
            vcache_release_vram(need_vram);
            HCK(hipMemGetInfo(&fr, &tot));
        }
        if (fr < need_vram) {
            kv_spill_to_host();
            return;
        }
        for (int il = 0; il < c.n_layers; ++il) {
            if (c.is_swa[il]) continue;
            __half* nk = nullptr;
            __half* niv = nullptr;
            __half* npl = nullptr;
            if (hipMalloc((void**)&nk, ns * row_b) != hipSuccess ||
                hipMalloc((void**)&niv, (size_t)(target + 8) * 256 * 2) != hipSuccess ||
                hipMalloc((void**)&npl, (size_t)(target / 4 + 8) * 128 * 2) != hipSuccess) {
                if (nk) hipFree(nk);
                if (niv) hipFree(niv);
                if (npl) hipFree(npl);
                (void)hipGetLastError();
                kv_spill_to_host();
                return;
            }
            HCK(hipMemset(nk, 0, ns * row_b));
            HCK(hipMemcpy(nk, kcache[il], os * row_b, hipMemcpyDeviceToDevice));
            HCK(hipFree(kcache[il]));
            kcache[il] = nk;
            HCK(hipMemset(niv, 0, (size_t)(target + 8) * 256 * 2));
            HCK(hipMemcpy(niv, ivg[il], (size_t)(kv_alloc_pos + 8) * 256 * 2, hipMemcpyDeviceToDevice));
            HCK(hipFree(ivg[il]));
            ivg[il] = niv;
            HCK(hipMemset(npl, 0, (size_t)(target / 4 + 8) * 128 * 2));
            HCK(hipMemcpy(npl, pooled[il], (size_t)(kv_alloc_pos / 4 + 8) * 128 * 2, hipMemcpyDeviceToDevice));
            HCK(hipFree(pooled[il]));
            pooled[il] = npl;
        }
        const int old_p = kv_alloc_pos;
        kv_alloc_pos = target;
        slots_sparse = (int)ns;
        ++kv_gen;
#if HY4_CUDA
        cg_clear();
#endif
        fprintf(stderr, "[runner] sparse KV + IndexPool grew in VRAM: %d -> %d positions (%.2f GB total, %.1f ms)\n",
                old_p, target, (12.0 * (ns * row_b + (target + 8) * 512.0 + (target / 4 + 8) * 256.0)) / 1e9, (now_s() - t0) * 1000.0);
    }
    // make room for positions < need; false when need exceeds the context
    bool kv_ensure(int need) {
        if (need > max_ctx + 8) return false;
        if (!kv_is_host && need > kv_alloc_pos) kv_grow_to(need);
        return need <= slots_sparse;
    }
    // Commit accepted tree path `path[0..n_accepted]` (path[0] == 0, length cnt = n_accepted + 1, topological order path[k] >= k):
    // compacts KV rows + IndexPool ivg when a side branch was accepted, builds completed IndexPool groups for [p0, p0+cnt),
    // and aligns the DFly2 captured key-list row for the next draft step.
    void commit_tree_path(int p0, const int* path, int n_accepted) {
        const int cnt = n_accepted + 1;
        bool need_compact = false;
        TreePos src;
        src.n = cnt;
        idx::IdxTree isrc;
        isrc.n = cnt;
        for (int k = 0; k < cnt; ++k) {
            src.depth[k] = (uint8_t)path[k];
            isrc.depth[k] = (uint8_t)path[k];
            if (path[k] != k) need_compact = true;
        }
        if (need_compact) {
            for (int il = 0; il < c.n_layers; ++il) {
                const bool swa = c.is_swa[il];
                mla_compact_kv(st, kcache[il], swa ? ring : slots_sparse, p0, swa ? ring : 0, src);
                if (!swa && index_on) hipLaunchKernelGGL(idx::k_idx_compact_ivg, dim3(1), dim3(256), 0, st, ivg[il], p0, isrc);
            }
        }
        if (index_on) {
            const int g_lo = p0 / 4, g_hi = (p0 + cnt) / 4 - 1;
            if (g_hi >= g_lo) {
                for (int il = 0; il < c.n_layers; ++il) {
                    if (c.is_swa[il]) continue;
                    const std::string p = L(il) + ".self_attn.compressed_indexer";
                    hipLaunchKernelGGL(idx::k_idx_pool, dim3(g_hi - g_lo + 1), dim3(128), 0, st, ivg[il], (const float*)tw(p + ".intra_group_ape"), (const float*)tw(p + ".key_norm.weight"),
                                       pooled[il], g_lo, (float)theta(il), c.rms_eps, (const int*)nullptr, 0);
                }
            }
        }
        if (dfly_cap) {
            const int last_node = path[n_accepted];
            if (last_node != n_accepted) {
                for (int d = 0; d < 5; ++d) {
                    if (!cap_list[d]) continue;
                    HCK(hipMemcpyAsync(dkeys[d] + (size_t)n_accepted * index_topk, dkeys[d] + (size_t)last_node * index_topk, (size_t)index_topk * 4, hipMemcpyDeviceToDevice, st));
                    HCK(hipMemcpyAsync(dnkc[d] + n_accepted, dnkc[d] + last_node, 4, hipMemcpyDeviceToDevice, st));
                }
            }
            cap_p0 = p0;
            cap_rows = cnt;
        }
    }
#if HY4_CUDA
    template <class Fn>
    cudaGraphExec_t cg_capture_one(Fn&& fn) {
        cudaGraph_t gr = nullptr;
        cudaGraphExec_t ex = nullptr;
        HCK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        fn();
        HCK(cudaStreamEndCapture(st, &gr));
        HCK(cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
        HCK(cudaGraphDestroy(gr));
        return ex;
    }
    void cg_emit_layer_pre_hyb(int il, int p0, int T, const TreeSpec* tree = nullptr) {
        const size_t nz = (size_t)T * 1536;
        if (il > 1) {
            const std::string pm_prev = L(il - 1) + ".mlp";
            hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)d_zcpu, nz);
            hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, z_h, nz);
            lin(pm_prev + ".routed_expert_up_proj.weight", z_h, F2, T);
            hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F2, (const float*)F3_dec, (const float*)tw(pm_prev + ".combine_output_norm.weight"), x, post, c.rms_eps);
        }
        int mo = -1;
        for (int j = 0; j < 4; ++j) if (emeta.layers[j] == il) mo = j;
        if (mo >= 0 && engram_ready) engram_layer(mo, il, T);
        attention_layer(il, p0, T, tree);
        const std::string pm = L(il) + ".mlp";
        const float* hfn = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_base");
        hc_pre(x, hfn, hsc, hbs, (const float*)tw(L(il) + ".post_attention_layernorm.weight"), 8, T, cur_h, cur_f);
        const float* rw = (const float*)tw(pm + ".gate.weight");
        if (T <= 1) hipLaunchKernelGGL((k_f32_gemv_t<1>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
        else if (T <= 2) hipLaunchKernelGGL((k_f32_gemv_t<2>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
        else if (T <= 4) hipLaunchKernelGGL((k_f32_gemv_t<4>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
        else if (T <= 6) hipLaunchKernelGGL((k_f32_gemv_t<6>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
        else hipLaunchKernelGGL((k_f32_gemv_t<8>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
        const float* rb = (const float*)tw(pm + ".gate.e_score_correction_bias");
        if (tune & TN_TOPK) hipLaunchKernelGGL(k_router_topk_w32, dim3(T), dim3(32), 0, st, F1, rb, moe_idx, moe_w, 16, c.routed_scale, (const uint4*)nullptr, (uint4*)nullptr);
        else hipLaunchKernelGGL(k_router_topk, dim3(T), dim3(256), 0, st, F1, rb, moe_idx, moe_w, 512, 16, c.routed_scale, (const uint4*)nullptr, (uint4*)nullptr);
        lin(pm + ".routed_expert_down_proj.weight", cur_h, z_f, T);
        HCK(hipMemcpyAsync(h_z, z_f, nz * 4, hipMemcpyDeviceToHost, st));
        HCK(hipMemcpyAsync(h_idx, moe_idx, (size_t)T * 16 * 4, hipMemcpyDeviceToHost, st));
        HCK(hipMemcpyAsync(h_w, moe_w, (size_t)T * 16 * 4, hipMemcpyDeviceToHost, st));
    }
    void cg_emit_layer_gpuexp_hyb(int il, int T) {
        const size_t nz = (size_t)T * 1536;
        const std::string pm = L(il) + ".mlp";
        void* mws = T <= 8 ? moe_ws_dec : moe_ws;
        hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, z_f, z_h, nz);
        moe_experts(st, g.d_sec4, d_experts + (size_t)(il - 1) * 512, z_h, moe_idx, moe_w, zout_f, T, mws);
        if (lm_ok_m(T) && (il - 1) < (int)lm_sh.size() && lm_sh[il - 1].d_jobs) {
            linear_h_multi(st, lm_sh[il - 1], cur_h, T);
        } else {
            lin(pm + ".shared_experts.gate_proj.weight", cur_h, F1, T);
            lin(pm + ".shared_experts.up_proj.weight", cur_h, F3_dec, T);
        }
        hipLaunchKernelGGL(k_swiglu_h, dim3((nz + 255) / 256), dim3(256), 0, st, F1, F3_dec, H1, nz);
        lin(pm + ".shared_experts.down_proj.weight", H1, F3_dec, T);
    }
    void cg_emit_layer_pre_mgpu(int il, int p0, int T, const TreeSpec* tree = nullptr) {
        const size_t nz = (size_t)T * 1536;
        if (il > 1) {
            const std::string pm_prev = L(il - 1) + ".mlp";
            for (const auto& pr : peer_runs)
                hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)pr.d_zpeer0, nz);
            hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, z_h, nz);
            lin(pm_prev + ".routed_expert_up_proj.weight", z_h, F2, T);
            hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F2, (const float*)F3_dec, (const float*)tw(pm_prev + ".combine_output_norm.weight"), x, post, c.rms_eps);
        }
        int mo = -1;
        for (int j = 0; j < 4; ++j) if (emeta.layers[j] == il) mo = j;
        if (mo >= 0 && engram_ready) engram_layer(mo, il, T);
        attention_layer(il, p0, T, tree);
        const std::string pm = L(il) + ".mlp";
        const float* hfn = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_base");
        hc_pre(x, hfn, hsc, hbs, (const float*)tw(L(il) + ".post_attention_layernorm.weight"), 8, T, cur_h, cur_f);
        const float* rw = (const float*)tw(pm + ".gate.weight");
        if (T <= 1) hipLaunchKernelGGL((k_f32_gemv_t<1>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
        else if (T <= 2) hipLaunchKernelGGL((k_f32_gemv_t<2>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
        else if (T <= 4) hipLaunchKernelGGL((k_f32_gemv_t<4>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
        else if (T <= 6) hipLaunchKernelGGL((k_f32_gemv_t<6>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
        else hipLaunchKernelGGL((k_f32_gemv_t<8>), dim3(512), dim3(256), 0, st, rw, (const float*)cur_f, F1, 512, HID, T);
        const float* rb = (const float*)tw(pm + ".gate.e_score_correction_bias");
        if (tune & TN_TOPK) hipLaunchKernelGGL(k_router_topk_w32, dim3(T), dim3(32), 0, st, F1, rb, moe_idx, moe_w, 16, c.routed_scale, (const uint4*)nullptr, (uint4*)nullptr);
        else hipLaunchKernelGGL(k_router_topk, dim3(T), dim3(256), 0, st, F1, rb, moe_idx, moe_w, 512, 16, c.routed_scale, (const uint4*)nullptr, (uint4*)nullptr);
        lin(pm + ".routed_expert_down_proj.weight", cur_h, z_f, T);
        hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, z_f, z_h, nz);
    }
    void cg_emit_layer_gpuexp_mgpu(int il, int T) {
        const size_t nz = (size_t)T * 1536;
        const std::string pm = L(il) + ".mlp";
        void* mws = T <= 8 ? moe_ws_dec : moe_ws;
        moe_experts(st, g.d_sec4, d_experts + (size_t)(il - 1) * 512, z_h, moe_idx, moe_w, zout_f, T, mws);
        if (lm_ok_m(T) && (il - 1) < (int)lm_sh.size() && lm_sh[il - 1].d_jobs) {
            linear_h_multi(st, lm_sh[il - 1], cur_h, T);
        } else {
            lin(pm + ".shared_experts.gate_proj.weight", cur_h, F1, T);
            lin(pm + ".shared_experts.up_proj.weight", cur_h, F3_dec, T);
        }
        hipLaunchKernelGGL(k_swiglu_h, dim3((nz + 255) / 256), dim3(256), 0, st, F1, F3_dec, H1, nz);
        lin(pm + ".shared_experts.down_proj.weight", H1, F3_dec, T);
    }
    bool forward_chunk_graph(const int32_t* tok_all, int p0, int T, std::string& err, const TreeSpec* tree = nullptr) {
        const bool is_tree = tree != nullptr && tree->M > 0;
        const int end_pos = is_tree ? (p0 + tree->max_depth() + 1) : (p0 + T);
        const bool use_list = index_on && (end_pos > index_topk);
        int G = end_pos / 4;
        if (G < 1) G = 1;
        const int score_grid = use_list ? ((((G + 63) / 64) + 15) & ~15) : 0;
        MlaAttnArgs ma_sp;
        ma_sp.M = T; ma_sp.NH = c.heads_sparse; ma_sp.p0 = p0;
        ma_sp.mode = use_list ? MLA_LIST : MLA_CAUSAL;
        ma_sp.kmax = use_list ? index_topk : 0;
        ma_sp.cache_slots = slots_sparse;
        const int s_causal = use_list ? 0 : mla_choose_splits(ma_sp);
        MlaAttnArgs ma_swa;
        ma_swa.M = T; ma_swa.NH = c.heads_swa; ma_swa.p0 = p0;
        ma_swa.mode = MLA_WINDOW; ma_swa.window = c.window; ma_swa.ring = ring;
        const int s_swa = mla_choose_splits(ma_swa);
        const size_t need_ws = std::max(mla_workspace_bytes(ma_swa), mla_workspace_bytes(T, c.heads_sparse, std::max(index_topk, p0 + T)));
        ensure_mla_ws(need_ws);
        if (!hcpart) hcpart = dalloc<float>((size_t)Tc * 24 * 9);
#if HY4_MMA
        if (skd::lin_int8_mma()) skd::int8_act_buf().ensure((size_t)T * 16384, skd::lin_int8_mma() >= 2);
#endif
        CgKey key{};
        key.T = T;
        key.is_tree = is_tree ? 1 : 0;
        key.use_list = use_list ? 1 : 0;
        key.s_causal = s_causal;
        key.s_swa = s_swa;
        key.score_grid = score_grid;
        key.dfly_cap = dfly_cap ? 1 : 0;
        key.route_now = ((tune & TN_ROUTE) && T == 1 && !hyb && peer_runs.empty()) ? 1 : 0;
        key.nof = (T <= 4 && (tune & TN_NOF2H) && !hyb && peer_runs.empty()) ? 1 : 0;
        key.kv_is_host = kv_is_host ? 1 : 0;
        key.slots_sparse = slots_sparse;
        key.engram_ready = engram_ready ? 1 : 0;
        key.hyb = hyb ? 1 : 0;
        key.n_peers = (int)peer_runs.size();
        key.lin_gemv_max = skd::lin_gemv_max_m();
        key.moe_dec_max = skd::moe_dec_max_m();
#if HY4_MMA
        key.int8_mma = skd::lin_int8_mma();
        key.dp4a = skd::lin_dp4a();
#endif
        key.mla_ws_gen = mla_ws_gen;
        CgEntry* ent = nullptr;
        for (auto& e : cg_cache) if (e.key == key) { ent = &e; break; }
        if (!ent) {
            if (cg_cache.size() >= 16) {
                cg_cache.front().destroy();
                cg_cache.erase(cg_cache.begin());
            }
            cg_cache.emplace_back();
            ent = &cg_cache.back();
            ent->key = key;
            capturing = true;
            cap_score_grid = score_grid;
            if (!hyb && peer_runs.empty()) {
                if (!engram_ready) {
                    ent->g_l0 = cg_capture_one([&] {
                        hipLaunchKernelGGL(k_embed_hc, dim3(T), dim3(256), 0, st, (const uint8_t*)tw("model.embed_tokens.weight"), d_ids, x, T);
                        for (int il = 0; il < c.n_layers; ++il) {
                            attention_layer(il, p0, T, tree);
                            mlp_layer(il, T);
                        }
                    });
                } else {
                    const int l_mid = emeta.layers[1];
                    ent->g_l0 = cg_capture_one([&] {
                        hipLaunchKernelGGL(k_embed_hc, dim3(T), dim3(256), 0, st, (const uint8_t*)tw("model.embed_tokens.weight"), d_ids, x, T);
                        attention_layer(0, p0, T, tree);
                        mlp_layer(0, T);
                    });
                    ent->g_l1_10 = cg_capture_one([&] {
                        for (int il = 1; il < l_mid; ++il) {
                            int mo = -1;
                            for (int j = 0; j < 4; ++j) if (emeta.layers[j] == il) mo = j;
                            if (mo >= 0) engram_layer(mo, il, T);
                            attention_layer(il, p0, T, tree);
                            mlp_layer(il, T);
                        }
                    });
                    ent->g_l11_41 = cg_capture_one([&] {
                        for (int il = l_mid; il < c.n_layers; ++il) {
                            int mo = -1;
                            for (int j = 0; j < 4; ++j) if (emeta.layers[j] == il) mo = j;
                            if (mo >= 0) engram_layer(mo, il, T);
                            attention_layer(il, p0, T, tree);
                            mlp_layer(il, T);
                        }
                    });
                }
            } else if (!hyb && !peer_runs.empty()) {
                // Multi-GPU segmented CUDA Graphs: GPU 0 segments + per-peer moe_experts graphs (zero host sync!)
                ent->g_l0 = cg_capture_one([&] {
                    hipLaunchKernelGGL(k_embed_hc, dim3(T), dim3(256), 0, st, (const uint8_t*)tw("model.embed_tokens.weight"), d_ids, x, T);
                    attention_layer(0, p0, T, tree);
                    mlp_layer(0, T);
                });
                ent->g_peers.resize(peer_runs.size());
                for (auto& arr : ent->g_peers) arr.fill(nullptr);
                for (int il = 1; il < c.n_layers; ++il) {
                    ent->g_pre[il - 1] = cg_capture_one([&] { cg_emit_layer_pre_mgpu(il, p0, T, tree); });
                    ent->g_gpuexp[il - 1] = cg_capture_one([&] { cg_emit_layer_gpuexp_mgpu(il, T); });
                    for (size_t p = 0; p < peer_runs.size(); ++p) {
                        PeerRun& pr = peer_runs[p];
                        HCK(cudaSetDevice(pr.dev));
                        cudaGraph_t gr = nullptr;
                        HCK(cudaStreamBeginCapture(pr.st, cudaStreamCaptureModeThreadLocal));
                        moe_experts(pr.st, g.peers[p].d_sec4, pr.d_experts + (size_t)(il - 1) * 512, pr.z_h, pr.idx, pr.w, pr.zout, T, pr.moe_ws);
                        HCK(cudaStreamEndCapture(pr.st, &gr));
                        HCK(cudaGraphInstantiate(&ent->g_peers[p][il - 1], gr, nullptr, nullptr, 0));
                        HCK(cudaGraphDestroy(gr));
                    }
                    HCK(cudaSetDevice(g.primary_gpu));
                }
                const size_t nz = (size_t)T * 1536;
                const std::string pm_last = L(c.n_layers - 1) + ".mlp";
                ent->g_tail_last = cg_capture_one([&] {
                    for (const auto& pr : peer_runs)
                        hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)pr.d_zpeer0, nz);
                    hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, z_h, nz);
                    lin(pm_last + ".routed_expert_up_proj.weight", z_h, F2, T);
                    hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F2, (const float*)F3_dec, (const float*)tw(pm_last + ".combine_output_norm.weight"), x, post, c.rms_eps);
                });
            } else {
                // Hybrid segmented graphs: layer 0 + 41 (pre, gpuexp) pairs + final layer tail
                ent->g_l0 = cg_capture_one([&] {
                    hipLaunchKernelGGL(k_embed_hc, dim3(T), dim3(256), 0, st, (const uint8_t*)tw("model.embed_tokens.weight"), d_ids, x, T);
                    attention_layer(0, p0, T, tree);
                    mlp_layer(0, T);
                });
                for (int il = 1; il < c.n_layers; ++il) {
                    ent->g_pre[il - 1] = cg_capture_one([&] { cg_emit_layer_pre_hyb(il, p0, T, tree); });
                    ent->g_gpuexp[il - 1] = cg_capture_one([&] { cg_emit_layer_gpuexp_hyb(il, T); });
                }
                const size_t nz = (size_t)T * 1536;
                const std::string pm_last = L(c.n_layers - 1) + ".mlp";
                ent->g_tail_last = cg_capture_one([&] {
                    hipLaunchKernelGGL(k_add_inplace, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, (const float*)d_zcpu, nz);
                    hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, z_h, nz);
                    lin(pm_last + ".routed_expert_up_proj.weight", z_h, F2, T);
                    hipLaunchKernelGGL(k_add_rms_post, dim3(T), dim3(256), 0, st, F2, (const float*)F3_dec, (const float*)tw(pm_last + ".combine_output_norm.weight"), x, post, c.rms_eps);
                });
            }
            capturing = false;
        }
        // Update per-step inputs via async H2D from pinned memory
        std::memcpy(h_ids, is_tree ? tree->tok : (tok_all + p0), (size_t)T * 4);
        h_step->p0 = p0;
        h_step->idx_G = G;
        h_step->is_tree = is_tree ? 1 : 0;
        if (is_tree) {
            h_step->mla_tp = tree->to_mla_pos();
            h_step->mla_tree.tree_nodes = tree->M;
            for (int i = 0; i < tree->M; ++i) {
                h_step->mla_tree.tree_depth[i] = tree->depth[i];
                for (int d = 0; d < 8; ++d) h_step->mla_tree.tree_anc[i][d] = tree->anc[i][d];
            }
            h_step->idx_tp = tree->to_idx_tree();
        } else {
            h_step->mla_tp.n = 0;
            h_step->mla_tree.tree_nodes = 0;
            h_step->idx_tp.n = 0;
        }
        HCK(hipMemcpyAsync(d_ids, h_ids, (size_t)T * 4, hipMemcpyHostToDevice, st));
        HCK(hipMemcpyAsync(d_step, h_step, sizeof(DevStepArgs), hipMemcpyHostToDevice, st));
        if (engram_ready) {
            const double t_pf0 = now_s();
            prefetch_engram(tok_all, p0, T, tree);
            t_eng_issue += now_s() - t_pf0;
        }
        if (!hyb && peer_runs.empty()) {
            HCK(cudaGraphLaunch(ent->g_l0, st));
            if (engram_ready) {
                double tq = now_s();
                if (!engram_prep(tok_all, p0, T, err, tree)) return false;
                t_eng_prep += now_s() - tq; ++n_eng_dec;
                HCK(cudaGraphLaunch(ent->g_l1_10, st));
                if (!engram_prep_rest(err)) return false;
                HCK(cudaGraphLaunch(ent->g_l11_41, st));
            }
        } else if (!hyb && !peer_runs.empty()) {
            const size_t nz = (size_t)T * 1536;
            const int l_mid = engram_ready ? emeta.layers[1] : -1;
            HCK(cudaGraphLaunch(ent->g_l0, st));
            for (int il = 1; il < c.n_layers; ++il) {
                if (engram_ready && il == 1) { double tq = now_s(); if (!engram_prep(tok_all, p0, T, err, tree)) return false; t_eng_prep += now_s() - tq; ++n_eng_dec; }
                if (engram_ready && il == l_mid && !engram_prep_rest(err)) return false;
                HCK(cudaGraphLaunch(ent->g_pre[il - 1], st));
                for (size_t p = 0; p < peer_runs.size(); ++p) {
                    PeerRun& pr = peer_runs[p];
                    HCK(cudaEventRecord(pr.ev_in, st));
                    HCK(cudaSetDevice(pr.dev));
                    HCK(cudaStreamWaitEvent(pr.st, pr.ev_in, 0));
                    HCK(cudaMemcpyPeerAsync(pr.z_h, pr.dev, z_h, g.primary_gpu, nz * 2, pr.st));
                    HCK(cudaMemcpyPeerAsync(pr.idx, pr.dev, moe_idx, g.primary_gpu, (size_t)T * 16 * 4, pr.st));
                    HCK(cudaMemcpyPeerAsync(pr.w, pr.dev, moe_w, g.primary_gpu, (size_t)T * 16 * 4, pr.st));
                    HCK(cudaGraphLaunch(ent->g_peers[p][il - 1], pr.st));
                    HCK(cudaMemcpyPeerAsync(pr.d_zpeer0, g.primary_gpu, pr.zout, pr.dev, nz * 4, pr.st));
                    HCK(cudaEventRecord(pr.ev_out, pr.st));
                }
                HCK(cudaSetDevice(g.primary_gpu));
                HCK(cudaGraphLaunch(ent->g_gpuexp[il - 1], st));
                for (auto& pr : peer_runs) HCK(cudaStreamWaitEvent(st, pr.ev_out, 0));
            }
            HCK(cudaGraphLaunch(ent->g_tail_last, st));
        } else {
            const size_t nz = (size_t)T * 1536;
            const int l_mid = engram_ready ? emeta.layers[1] : -1;
            HCK(cudaGraphLaunch(ent->g_l0, st));
            for (int il = 1; il < c.n_layers; ++il) {
                if (engram_ready && il == 1) { double tq = now_s(); if (!engram_prep(tok_all, p0, T, err, tree)) return false; t_eng_prep += now_s() - tq; ++n_eng_dec; }
                if (engram_ready && il == l_mid && !engram_prep_rest(err)) return false;
                HCK(cudaGraphLaunch(ent->g_pre[il - 1], st));
                HCK(hipEventRecord(ev_hyb, st));
                HCK(cudaGraphLaunch(ent->g_gpuexp[il - 1], st));
                double t0 = now_s();
                HCK(hipEventSynchronize(ev_hyb));
                const int ml = il - 1;
                const uint8_t* use_ptr = nullptr;
                const bool do_tail = (tail_skip > 0 && T <= 16);
                if (vcache_nslots > 1 || do_tail) {
                    int minr[512];
                    if (do_tail) {
                        for (int e = 0; e < 512; ++e) minr[e] = 99;
                        for (int t = 0; t < T; ++t)
                            for (int s = 0; s < 16; ++s) {
                                const int e = h_idx[(size_t)t * 16 + s];
                                if (e >= 0 && e < 512 && s < minr[e]) minr[e] = s;
                            }
                    }
                    for (int t = 0; t < T; ++t)
                        for (int s = 0; s < 16; ++s) {
                            const size_t i = (size_t)t * 16 + s;
                            const int e = h_idx[i];
                            uint8_t u = (e >= 0 && e < 512 && !(vcache_nslots > 1 && vram_flag[ml * 512 + e])) ? 1 : 0;
                            if (u && do_tail && g.experts[(size_t)ml * 512 + e].tier >= 2 && minr[e] >= tail_skip) {
                                u = 0;
                                ++tail_skipped_pairs;
                            }
                            h_use[i] = u;
                        }
                    use_ptr = h_use.data();
                }
                cpu.submit(ml, T, h_z, h_idx, h_w, use_ptr, h_zcpu);
                if (vcache_nslots > 1) vcache_note_decode(ml, T, h_idx);
                cpu.wait();
                t_cpu_wait += now_s() - t0;
                HCK(hipMemcpyAsync(d_zcpu, h_zcpu, nz * 4, hipMemcpyHostToDevice, st));
            }
            HCK(cudaGraphLaunch(ent->g_tail_last, st));
        }
        // Update host-side capture metadata for DFly2
        last_use_list = use_list;
        if (dfly_cap) {
            const int nr = std::min(T, 16), r0 = T - nr;
            cap_p0 = p0 + r0;
            cap_rows = nr;
            for (int d = 0; d < 5; ++d) cap_list[d] = use_list;
        }
        return true;
    }
#endif
    bool forward_chunk(const int32_t* tok_all, int p0, int T, std::string& err, const TreeSpec* tree = nullptr) {
        if (!kv_is_host && p0 + T + 16 > kv_alloc_pos) kv_grow_to(p0 + T + 16);
        if (T <= 8 && vcache_nslots > 1) {
            vcache_decay_step(T);
            vcache_poll_commit();
        }
        const bool is_tree = tree != nullptr && tree->M > 0;
#if HY4_CUDA
        if (cuda_graph_on && (peer_runs.empty() || !hyb) && T <= 8 && !dump && !prof && !prof2 && !(!is_tree && vis && vis->needs_fix(p0, T)))
            return forward_chunk_graph(tok_all, p0, T, err, tree);
#endif
        HCK(hipMemcpyAsync(d_ids, is_tree ? tree->tok : (tok_all + p0), (size_t)T * 4, hipMemcpyHostToDevice, st));
        hipLaunchKernelGGL(k_embed_hc, dim3(T), dim3(256), 0, st, (const uint8_t*)tw("model.embed_tokens.weight"), d_ids, x, T);
        if (!is_tree && vis) vis->embed(st, x, p0, T);  // HY4-VISION hook: image rows replace the embeddings of the image-pad tokens
        const bool defer_eng = engram_ready;
        const bool pf_pipe = pf_active(T) && !pf_use_gather(T);
        if (pf_pipe) pf_begin_chunk();
        if (defer_eng) {
            bool already = false;
            for (int s = 0; s < 2; ++s)
                if (ep[s].pending && ep[s].p0 == p0 && ep[s].T == T && (ep[s].tree.M == (tree ? tree->M : 0))) already = true;
            if (!already) {
                const double t_pf0 = now_s();
                prefetch_engram(tok_all, p0, T, tree);
                if (T <= 8) t_eng_issue += now_s() - t_pf0;
            }
        }
        for (int il = 0; il < c.n_layers; ++il) {
            if (defer_eng && il == emeta.layers[0]) { double tq = now_s(); if (!engram_prep(tok_all, p0, T, err, tree)) return false; if (T <= 8) { t_eng_prep += now_s() - tq; ++n_eng_dec; } }
            if (defer_eng && il == emeta.layers[1]) {
                double tq = now_s();
                if (!engram_prep_rest(err)) return false;
                if (T > 8) t_engram_cpu += now_s() - tq;
                if (pf_eng_tok && p0 + T < pf_eng_total) {
                    const int np0 = p0 + T, nT = std::min(Tc, pf_eng_total - np0);
                    prefetch_engram(pf_eng_tok, np0, nT);
                }
            }
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
            if (il >= 1 && pf_pipe) pf_pump(pf_layer_first[il - 1] + pf_nbuf - 1);
            attention_layer(il, p0, T, tree);
            maybe_dump("x_L" + std::to_string(il) + "_attn", T, 4 * HID, x, 4);
            if (prof) { HCK(hipStreamSynchronize(st)); double tb = now_s(); (c.is_swa[il] ? p_attn_swa : p_attn_sp) += tb - ta; ta = tb; }
            mlp_layer(il, T);
            maybe_dump("x_L" + std::to_string(il) + "_mlp", T, 4 * HID, x, 4);
            if (prof) { HCK(hipStreamSynchronize(st)); double tb = now_s(); (il == 0 ? p_mlp0 : p_moe) += tb - ta; }
        }
        if (T >= 64 && vcache_nslots > 1 && !vcache_frozen) vcache_rebalance_prefill();
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
        ensure_logits(n);
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
