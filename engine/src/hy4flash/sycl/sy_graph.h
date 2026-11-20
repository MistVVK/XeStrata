// sy_graph.h - 42-layer hy_v4_flash forward on Intel Arc (SYCL):
//   * SYCL command_graph / command_buffer capture & replay for decode (T=1) and verify (T=2..8): ~1300 launches -> 85 replays
//   * Quantized S_K kv_b decode/verify + single-layer 29 MB prefill scratch (-0.76 GiB VRAM, 3x less decode kv_b bandwidth)
//   * Scratch buffer aliasing (o_h=H1, kv_f=F1, ctx_h=F2 -> -164 MB at Tc=1024, -327 MB at Tc=2048)
//   * Unified hy4::CpuExperts (AVX-512 VBMI/VNNI + AVX2 + two-term hi + cold SSD) + multi-slab SyVramExpertCache
//   * Deterministic two-launch hc_mix2 (f96da6f22) and batch-invariant MLA split-K (61e7db340) preserved intact
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
#include "hy4_cpu_experts.h"
#include "hy4_engram.h"
#include "sy_common.h"
#include "sy_gemm_xmx.h"
#include "sy_gpuexp.h"
#include "sy_indexer.h"
#include "sy_kernels.h"
#include "sy_linear.h"
#include "sy_mla.h"
#include "sy_model.h"
#include "sy_vision.h"
#include "sy_vram_cache.h"

namespace hy4s {

static bool load_npy_f32(const std::string& path, std::vector<float>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned char h[10];
    if (fread(h, 1, 10, f) != 10) { fclose(f); return false; }
    const size_t hl = (size_t)h[8] | ((size_t)h[9] << 8);
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, (long)(10 + hl), SEEK_SET);
    out.resize((size_t)(sz - 10 - (long)hl) / 4);
    const size_t n = fread(out.data(), 4, out.size(), f);
    fclose(f);
    return n == out.size();
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

struct EngramDev {
    float* lnw = nullptr;
    float* kvb = nullptr;
    float* egate = nullptr;
    float* pre_gamma = nullptr;
    float* hcs = nullptr;
};

// Shared placement helper for sy_main & sy_serve: partitions Section 4 experts across
//   Tier 0: SyVramExpertCache (multi-slab <= 1.4 GiB/slab, auto-sized from free VRAM or --vram-experts-gb)
//   Tier 1: hy4::CpuExperts RAM arena (AVX-512 VBMI/VNNI or AVX2 + optional two-term "hi")
//   Tier 2: hy4::CpuExperts cold SSD tier (when remaining experts exceed RAM budget)
inline bool setup_experts_unified(const std::string& model_path, int threads, double vram_experts_gb_arg, double ram_experts_gb_arg,
                                  size_t reserve_vram_bytes, size_t host_overhead_bytes, bool hi_act,
                                  SyVramExpertCache& vcache, hy4::CpuExperts& ucx, std::string& err) {
    // 1. Determine VRAM expert cache budget (caller includes DFly2 reserve in reserve_vram_bytes when !df.ok so plain & DFly2 match!)
    size_t vram_budget = 0;
    if (vram_experts_gb_arg >= 0.0) {
        vram_budget = (size_t)(vram_experts_gb_arg * 1073741824.0);
    } else if (const char* env_v = std::getenv("HY4S_VRAM_EXPERTS_GB")) {
        const double g = atof(env_v);
        vram_budget = g > 0.0 ? (size_t)(g * 1073741824.0) : 0;
    } else {
        const size_t tot = dev_caps().global_mem;
        const size_t used = dev_bytes_allocated() + reserve_vram_bytes;
        const size_t safe_cap = (size_t)((double)tot * 0.94);
        if (safe_cap > used + (256ull << 20)) vram_budget = safe_cap - used;
    }

    if (!vcache.init(model_path, vram_budget, err, 1400)) return false;
    vcache.set_q8_act(!hi_act);

    // 2. Partition remaining experts into RAM arena vs cold SSD using Section 3 rank order
    std::vector<uint16_t> eor, codes;
    std::vector<uint32_t> bbytes;
    if (!hy4::CpuExperts::read_rank_table(model_path, eor, codes, err)) return false;
    if (!hy4::CpuExperts::read_body_bytes(model_path, bbytes, err)) return false;
    const int NL = (int)(eor.size() / 512);

    std::vector<std::vector<bool>> vram_mask;
    vcache.build_vram_mask(vram_mask);

    uint64_t ram_budget = 0;
    if (ram_experts_gb_arg > 0.0) {
        ram_budget = (uint64_t)(ram_experts_gb_arg * 1e9);
    } else if (const char* env_r = std::getenv("HY4S_RAM_BUDGET_GB")) {
        ram_budget = (uint64_t)(atof(env_r) * 1e9);
    } else {
        const uint64_t sys_ram = hy4::CpuExperts::ram_budget_bytes();
        const uint64_t keep = host_overhead_bytes + (uint64_t)(8.5 * 1e9);  // OS + SYCL runtime + KV + bounce ring + Engram
        ram_budget = sys_ram > keep ? (sys_ram - keep) : 0;
    }

    std::vector<std::vector<bool>> cpu_mask(NL, std::vector<bool>(512, false));
    std::vector<std::vector<bool>> cold_mask(NL, std::vector<bool>(512, false));
    uint64_t ram_used = 0, n_cpu = 0, n_cold = 0, cold_bytes = 0;
    bool ram_full = false;
    for (int rk = 0; rk < 512; ++rk) {
        for (int L = 0; L < NL; ++L) {
            const size_t idx = (size_t)L * 512 + rk;
            const int e = eor[idx];
            const uint64_t sz = ((uint64_t)bbytes[idx] + 4095ull) & ~4095ull;
            if (vcache.active() && vram_mask[L][e]) {
                // Initially in VRAM; mark cold_mask as on-disk fallback in case adaptive cache evicts it later (0 RAM cost!)
                cold_mask[L][e] = true;
                continue;
            }
            if (!ram_full && ram_used + sz <= ram_budget) {
                cpu_mask[L][e] = true;
                ram_used += sz;
                ++n_cpu;
            } else {
                ram_full = true;
                cold_mask[L][e] = true;
                ++n_cold;
                cold_bytes += sz;
            }
        }
    }

#ifndef _WIN32
    // Evict stale file page cache before ucx allocates its anonymous arena so Linux doesn't stall on THP compaction!
    {
        int fd = ::open(model_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
        }
    }
#endif
    if (!ucx.init(model_path, threads, &cpu_mask, err)) return false;
    if (hi_act) ucx.set_hi_activations(true);

    bool any_cold = false;
    for (int L = 0; L < NL && !any_cold; ++L)
        for (int e = 0; e < 512; ++e)
            if (cold_mask[L][e]) { any_cold = true; break; }
#ifndef _WIN32
    if (any_cold) {
        // When cold tail is small (< 12 GB, e.g. 62 GB RAM box with SQ-64), default to buffered IO so Linux page cache holds it fast
        if (!std::getenv("STRATA_CPUEXP_COLD_IO") && cold_bytes < (uint64_t)(12.0 * 1e9))
            hy4s_setenv("STRATA_CPUEXP_COLD_IO", "buffered", 0);
        if (!ucx.set_cold_experts(cold_mask, err)) return false;
    }
#endif
    fprintf(stderr, "[experts] unified pool (%s, %d thr, act=%s): %d VRAM (%.2f GiB), %llu RAM (%.2f GB, load %.1f s), %llu cold SSD (%.2f GB)\n",
            ucx.kernel_name(), ucx.n_threads(), ucx.hi_activations() ? "hi" : "q8",
            vcache.num_slots(), vcache.device_bytes() / 1073741824.0,
            (unsigned long long)n_cpu, ucx.arena_bytes() / 1e9, ucx.load_seconds(),
            (unsigned long long)n_cold, cold_bytes / 1e9);
    return true;
}

class Runner {
public:
    SyModel& g;
    const hy4::Cfg& c;
    hy4::CpuExperts& ucx;
    SyVramExpertCache* vcache = nullptr;
    int max_ctx, Tc;
    int expert_act = 1;
    GpuExperts* gx = nullptr;
    int gx_min_T = 32;
    int kv_host_above = 24000;
    size_t kv_host_bytes = 0;
    int iring = 0;
    int score_rows = 256;

    // Device buffers
    int32_t* d_ids = nullptr;
    StepParams* d_step = nullptr;
    StepParams* h_step = nullptr;
    float *x = nullptr, *pre = nullptr, *post = nullptr, *cur_f = nullptr, *F1 = nullptr, *F2 = nullptr, *F3 = nullptr, *sub = nullptr;
    half *cur_h = nullptr, *qr_h = nullptr, *qa_h = nullptr, *o_h = nullptr, *H1 = nullptr, *z_h = nullptr, *er_h = nullptr, *hfin = nullptr, *ctx_h = nullptr;
    float *z_f = nullptr, *zout_f = nullptr, *zout_gpu_f = nullptr, *moe_w = nullptr, *logits = nullptr;
    int32_t *moe_idx = nullptr, *d_argmax = nullptr;
    float* d_pack = nullptr;
    float* h_pack = nullptr;
    float* hc_part = nullptr;
    uint32_t* hc_cnt = nullptr;
    float* h_z = nullptr;
    float* h_zout = nullptr;
    int32_t* h_idx = nullptr;
    float* h_w = nullptr;
    uint8_t h_use_cpu[16 * 16] = {};
    void* mla_ws = nullptr;
    size_t mla_ws_bytes = 0;
    float* kv_f = nullptr;  // aliased to F1!
    uint8_t* d_eraw[4] = {};
    uint8_t* d_evalid[4] = {};
    EngramDev eng_dev[4];
    // Single-layer reusable fp16 scratch: dq_buf (32 MiB) for oneMKL dequant-once dense prefill (aliases wk_tmp; wv_tmp independent)
    half* dq_buf = nullptr;
    half* wk_tmp = nullptr;
    half* wv_tmp = nullptr;
    std::vector<half*> kcache;

    // Vision & DFly2 capture
    VisionCtx* vis = nullptr;
    bool dfly_cap = false;
    int32_t *dkeys[5] = {}, *dnkc[5] = {};
    int cap_p0 = -1, cap_rows = 0;
    bool cap_list[5] = {};
    static int dfly_slot(int il) { return il == 25 ? 0 : il == 29 ? 1 : il == 33 ? 2 : il == 37 ? 3 : il == 41 ? 4 : -1; }
    void dfly_capture_alloc() {
        for (int d = 0; d < 5; ++d)
            if (!dkeys[d]) {
                dkeys[d] = dalloc<int32_t>((size_t)16 * (index_topk + 8));
                dnkc[d] = dalloc<int32_t>(16);
            }
        dfly_cap = true;
    }
    void dfly_capture_copy(int il, int T, bool use_list) {
        const int d = dfly_slot(il);
        if (d < 0 || !use_list) return;
        const int nr = std::min(T, 16), r0 = T - nr;
        d2d(dkeys[d], keys + (size_t)r0 * index_topk, (size_t)nr * index_topk * 4);
        d2d(dnkc[d], nkeys + r0, (size_t)nr * 4);
    }
    void dfly_capture(int il, int p0, int T, bool use_list) {
        const int d = dfly_slot(il);
        if (d < 0) return;
        const int nr = std::min(T, 16), r0 = T - nr;
        cap_p0 = p0 + r0;
        cap_rows = nr;
        cap_list[d] = use_list;
        dfly_capture_copy(il, T, use_list);
    }

    int ring = 0, slots_sparse = 0, logit_rows = 1;
    hy4::EngramMeta emeta;
    hy4::EngramIO eio;
    bool engram_ready = false;
    double t_engram_cpu = 0, t_engram_wait = 0, t_cpu_exp = 0;
    bool dump = false, dbg_layers = false;
    std::string force_route_dir;
    int cur_p0 = 0;
    std::string dump_dir = ".";
    int index_on = 1;
    int index_topk = 2048;

    // Indexer state
    std::vector<half*> ivg, pooled;
    float *ik_f = nullptr, *iv_f = nullptr, *ig_f = nullptr, *iq_f = nullptr, *hw_f = nullptr, *score = nullptr;
    half *ik_h = nullptr, *iq_h = nullptr;
    int32_t *keys = nullptr, *nkeys = nullptr;
    size_t ldS = 0;
    bool last_use_list = false;

    // SYCL command_graph state
    bool graph_enabled = false;
    uint64_t graph_replays = 0, graph_captures = 0;

#if HY4S_HAS_GRAPH
    struct GraphKey {
        int T = 0, use_list = 0, dfly_cap = 0, engram_on = 0;
        int swa_TK = 0, swa_S = 0, swa_tps = 0;
        int sp_TK = 0, sp_S = 0, sp_tps = 0;
        bool operator==(const GraphKey& o) const {
            return T == o.T && use_list == o.use_list && dfly_cap == o.dfly_cap && engram_on == o.engram_on &&
                   swa_TK == o.swa_TK && swa_S == o.swa_S && swa_tps == o.swa_tps &&
                   sp_TK == o.sp_TK && sp_S == o.sp_S && sp_tps == o.sp_tps;
        }
    };
    using ExecGraph = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
    struct GraphSet {
        GraphKey key;
        std::unique_ptr<ExecGraph> g_init;
        std::unique_ptr<ExecGraph> g_l1_pre;
        std::vector<std::unique_ptr<ExecGraph>> g_shared;     // [42]
        std::vector<std::unique_ptr<ExecGraph>> g_post_next;  // [42]
    };
    std::vector<std::unique_ptr<GraphSet>> gcache_;
    std::unique_ptr<ExecGraph> g_head_[9];  // cached head(T, 0) for T = 1..8
#endif

    Runner(SyModel& gm, hy4::CpuExperts& cxe, int max_ctx_, int max_chunk, int logit_rows_)
        : g(gm), c(gm.man.c), ucx(cxe), max_ctx(max_ctx_), Tc(max_chunk), logit_rows(logit_rows_) {}

    const void* tw(const std::string& name) const { return g.tw(name); }
    const hy4::TInfo& ti(const std::string& name) const { return g.ti(name); }
    std::string L(int il) const { return "model.layers." + std::to_string(il); }
    int heads(int il) const { return c.is_swa[il] ? c.heads_swa : c.heads_sparse; }
    double theta(int il) const { return c.is_swa[il] ? c.theta_swa : c.theta_sparse; }

    // Call strictly on committed tokens for bitwise plain == DFly2 adaptive VRAM cache promotion
    void commit_vram_adapt(int n_committed, int next_pos) {
        if (!vcache || !vcache->adaptive() || dump || tap || !force_route_dir.empty()) return;
        vcache->commit_tokens(n_committed, next_pos, [&](int L, int e) -> const uint8_t* { return ucx.view(L, e).body; });
    }
    void step_adapt() {
        if (!vcache || !vcache->adaptive() || dump || tap || !force_route_dir.empty()) return;
        vcache->step_adapt([&](int L, int e) -> const uint8_t* { return ucx.view(L, e).body; });
    }

    // Per-section GPU timeline (prof2)
    bool prof2 = false;
    std::map<std::string, std::pair<double, int>> pacc;
    struct PEv { sycl::event a, b; std::string nm; };
    std::vector<PEv> pev;
    static bool sync_all_env() {
        static const bool v = std::getenv("HY4S_SYNCALL") != nullptr;
        return v;
    }
    struct PS {
        Runner* r;
        bool on;
        PEv e;
        PS(Runner* r_, const std::string& n) : r(r_), on(r_->prof2) {
            if (sync_all_env()) wait();
            if (!on) return;
            e.nm = n;
            e.a = Q().single_task([]() {});
        }
        ~PS() {
            if (sync_all_env()) wait();
            if (!on) return;
            e.b = Q().single_task([]() {});
            r->pev.push_back(e);
        }
    };
    void pflush() {
        if (!prof2 || pev.empty()) return;
        wait();
        for (auto& e : pev) {
            const double t0 = (double)e.a.get_profiling_info<sycl::info::event_profiling::command_start>();
            const double t1 = (double)e.b.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto& a = pacc[e.nm];
            a.first += (t1 - t0) * 1e-6;
            a.second++;
        }
        pev.clear();
    }
    void pacc_report(const char* what, int steps) {
        if (!prof2) return;
        pflush();
        std::vector<std::pair<double, std::string>> v;
        for (auto& kv : pacc) v.push_back({kv.second.first, kv.first});
        std::sort(v.rbegin(), v.rend());
        fprintf(stderr, "[prof2 %s] per-step GPU ms by section (marker kernels, /%d steps):\n", what, steps);
        double tot = 0;
        for (auto& p : v) {
            tot += p.first;
            fprintf(stderr, "  %-34s %8.3f ms  (%d calls/step)\n", p.second.c_str(), p.first / steps, pacc[p.second].second / steps);
        }
        fprintf(stderr, "  TOTAL %.3f ms\n", tot / steps);
        pacc.clear();
    }
    void lin(const std::string& name, const half* X, float* Y, int M) {
        const hy4::TInfo& t = ti(name);
        PS _p(this, std::string("lin ") + name.substr(name.find('.', 13) == std::string::npos ? 0 : name.find('.', 13) + 1));
        linear_auto(t.fmt, tw(name), (int)t.rows(), (int)t.cols(), X, Y, M, g.is_rp(name));
    }
    void ensure_mla_ws(size_t need) {
        if (need <= mla_ws_bytes) return;
        wait();
        if (mla_ws) dfree(mla_ws);
        mla_ws_bytes = need + (need >> 2);
        mla_ws = dalloc<uint8_t>(mla_ws_bytes);
    }

    bool init(const std::string& engram_path, std::string& err) {
        const int T = Tc;
        d_ids = dalloc<int32_t>(T);
        d_step = dalloc<StepParams>(1);
        h_step = halloc<StepParams>(1);
        x = dalloc<float>((size_t)T * 4 * HID);
        pre = dalloc<float>(T * 4);
        post = dalloc<float>(T * 4);
        hc_part = dalloc<float>((size_t)T * 24 * 9);
        hc_cnt = dalloc<uint32_t>(T);
        dzero(hc_cnt, (size_t)T * 4);
        cur_h = dalloc<half>((size_t)T * HID);
        cur_f = dalloc<float>((size_t)T * HID);
        sub = dalloc<float>((size_t)T * HID);
        const size_t big = (size_t)T * 16384;
        F1 = dalloc<float>(big);
        F2 = dalloc<float>(big);
        F3 = dalloc<float>(big);
        H1 = dalloc<half>(big);
        qr_h = dalloc<half>((size_t)T * 1024);
        qa_h = dalloc<half>((size_t)T * 64 * 576);
        // Buffer aliasing (non-overlapping lifetimes -> saves ~164 MB at Tc=1024, ~327 MB at Tc=2048!)
        o_h = H1;                            // o_h only live in attention; H1 only live in MLP
        ctx_h = reinterpret_cast<half*>(F2); // F2 dead between mla_prep_q and mla_vup; sizeof(F2) == T*64*512*2
        kv_f = F1;                           // kv_f only live in engram_layer (5*T*3072 = 15360*T <= 16384*T)
        if (std::getenv("HY4S_HASH")) d_hash = dalloc<uint64_t>(65536);
        d_pack = dalloc<float>((size_t)T * (32 + 1536));
        h_pack = halloc<float>((size_t)T * (32 + 1536));
        z_f = d_pack + (size_t)T * 32;
        moe_idx = (int32_t*)d_pack;
        moe_w = d_pack + (size_t)T * 16;
        h_z = h_pack + (size_t)T * 32;
        h_idx = (int32_t*)h_pack;
        h_w = h_pack + (size_t)T * 16;
        z_h = dalloc<half>((size_t)T * 1536);
        zout_f = dalloc<float>((size_t)T * 1536);
        zout_gpu_f = dalloc<float>((size_t)16 * 1536);
        h_zout = halloc<float>((size_t)T * 1536);
        hfin = dalloc<half>((size_t)T * HID);
        logits = dalloc<float>((size_t)logit_rows * c.vocab);
        d_argmax = dalloc<int32_t>(16);

        // Pre-allocate split-K MLA workspace big enough for decode/verify so command_graph pointer never reallocates!
        mla_ws_bytes = (size_t)8 * 4 * 16 * (16 * 512 + 32) * sizeof(float);  // ~17 MB (covers up to M=8, NH=64, S=16)
        mla_ws = dalloc<uint8_t>(mla_ws_bytes);

        // KV caches
        ring = ((512 + Tc + 15) / 16) * 16;
        slots_sparse = ((max_ctx + 8 + 15) / 16) * 16;
        kcache.resize(c.n_layers);
        size_t kvbytes = 0;
        for (int il = 0; il < c.n_layers; ++il) {
            const int slots = c.is_swa[il] ? ring : slots_sparse;
            const size_t by = mla_cache_bytes(slots);
            // Put sparse KV in VRAM if card has >= 14 GiB and max_ctx fits comfortably; otherwise host-mapped for > kv_host_above
            const bool host_kv = !c.is_swa[il] && max_ctx > kv_host_above;
            if (host_kv) {
                kcache[il] = (half*)halloc<uint8_t>(by);
                std::memset(kcache[il], 0, by);
                kv_host_bytes += by;
            } else {
                kcache[il] = (half*)dalloc<uint8_t>(by);
                dzero(kcache[il], by);
            }
            kvbytes += by;
        }

        // Indexer state
        iring = ((Tc + 8 + 3) / 4) * 4;
        ivg.assign(c.n_layers, nullptr);
        pooled.assign(c.n_layers, nullptr);
        for (int il = 0; il < c.n_layers; ++il)
            if (!c.is_swa[il]) {
                ivg[il] = dalloc<half>((size_t)iring * 256);
                pooled[il] = dalloc<half>((size_t)(max_ctx / 4 + 8) * 128);
            }
        ik_f = dalloc<float>((size_t)Tc * 128);
        iv_f = dalloc<float>((size_t)Tc * 128);
        ig_f = dalloc<float>((size_t)Tc * 128);
        ik_h = dalloc<half>((size_t)Tc * 128);
        iq_f = dalloc<float>((size_t)Tc * 2048);
        iq_h = dalloc<half>((size_t)Tc * 2048);
        hw_f = dalloc<float>((size_t)Tc * 16);
        ldS = (size_t)(max_ctx / 4 + 8);
        score = dalloc<float>((size_t)std::min(Tc, score_rows) * ldS);
        keys = dalloc<int32_t>((size_t)Tc * (index_topk + 8));
        nkeys = dalloc<int32_t>((size_t)Tc);

        // Unified 32 MiB fp16 scratch for oneMKL dequant-once dense prefill (aliases wk_tmp=12.6 MB; wv_tmp stays independent!)
        constexpr size_t DQ_HALVES = (size_t)16 * 1024 * 1024;  // 32 MiB
        dq_buf = dalloc<half>(DQ_HALVES);
        wk_tmp = dq_buf;
        wv_tmp = dalloc<half>((size_t)64 * 256 * 512);
        register_dq_scratch(dq_buf, DQ_HALVES);

        // Engram
        if (!hy4::load_engram_meta_file(g.path, emeta, err)) return false;
        if (!engram_path.empty()) {
            if (!eio.open(engram_path, g.seed.data(), g.seed.size(), hy4::ENGRAM_DEFAULT_LRU_ROWS, 16, 8, err)) return false;
            engram_ready = true;
        }
        for (int mo = 0; mo < 4; ++mo) {
            const int il = emeta.layers[mo];
            const std::string p = L(il) + ".engram_layer";
            EngramDev& e = eng_dev[mo];
            e.lnw = dalloc<float>(16 * 96);
            e.kvb = dalloc<float>(5 * HID);
            e.egate = dalloc<float>(4 * HID);
            e.pre_gamma = dalloc<float>(HID);
            e.hcs = dalloc<float>(4 * HID);
            for (int ch = 0; ch < 16; ++ch) d2d(e.lnw + ch * 96, tw(p + ".lookup_per_head_norms." + std::to_string(ch) + ".weight"), 96 * 4);
            for (int j = 0; j < 4; ++j) {
                d2d(e.kvb + j * HID, tw(p + ".key_projs." + std::to_string(j) + ".bias"), HID * 4);
                d2d(e.hcs + j * HID, tw(p + ".unified_per_hc_scales." + std::to_string(j)), HID * 4);
                std::vector<float> nk(HID), nq(HID), gg(HID);
                d2h(nk.data(), tw(p + ".norm_key." + std::to_string(j) + ".weight"), HID * 4);
                d2h(nq.data(), tw(p + ".norm_query." + std::to_string(j) + ".weight"), HID * 4);
                wait();
                for (int d = 0; d < HID; ++d) gg[d] = nk[d] * nq[d];
                h2d(e.egate + j * HID, gg.data(), HID * 4);
                wait();
            }
            d2d(e.kvb + 4 * HID, tw(p + ".value_proj.bias"), HID * 4);
            d2d(e.pre_gamma, tw(p + ".unified_pre_gamma"), HID * 4);
            d_eraw[mo] = dalloc<uint8_t>((size_t)Tc * 16 * 100);
            d_evalid[mo] = dalloc<uint8_t>((size_t)Tc * 16);
        }
        er_h = dalloc<half>((size_t)Tc * 1536);
        ep_alloc();
        build_fuse();

        // Enable SYCL command_graph when supported by device & runtime (HY4S_GRAPH=0 disables)
#if HY4S_HAS_GRAPH
        {
            const char* eg = std::getenv("HY4S_GRAPH");
            graph_enabled = dev_caps().has_graph && !(eg && eg[0] == '0');
        }
#endif
        wait();
        fprintf(stderr, "[runner] kv cache %.2f GB (%.2f GB host), ctx %d, chunk %d, graph=%d, device allocated %.2f GiB of %.2f GiB\n",
                kvbytes / 1e9, kv_host_bytes / 1e9, max_ctx, Tc, (int)graph_enabled,
                dev_bytes_allocated() / 1073741824.0, dev_caps().global_mem / 1073741824.0);
        return true;
    }

    // Engram prefetch / upload
    struct EPrep {
        std::vector<uint8_t> raw, valid;
        int p0 = -1, T = 0;
        bool ok = false, pending = false;
        std::string err;
        std::future<void> fut;
        sycl::event used;
        bool have_used = false;
    };
    EPrep ep[3];
    int ep_next = 0;
    void ep_alloc() {
        for (int s = 0; s < 3; ++s) {
            ep[s].raw.resize((size_t)4 * Tc * 16 * 100);
            ep[s].valid.resize((size_t)4 * Tc * 16);
        }
    }
    void ep_compute(EPrep* e, const int32_t* tok_all, int p0, int T) {
        std::string err;
        e->ok = true;
        for (int mo = 0; mo < 4; ++mo) {
            std::vector<uint64_t> ids((size_t)T * 16);
            uint8_t* valid = e->valid.data() + (size_t)mo * Tc * 16;
            hy4::engram_hash(emeta, mo, tok_all, p0, p0 + T, ids.data(), valid);
            if (vis) vis->mask_engram(valid, p0, T);
            uint8_t* raw = e->raw.data() + (size_t)mo * Tc * 16 * 100;
            if (!eio.fetch_raw(ids.data(), ids.size(), raw, err)) {
                e->ok = false;
                e->err = err;
                return;
            }
        }
    }
    void prefetch_engram(const int32_t* tok_all, int p0, int T) {
        if (!engram_ready) return;
        for (int s = 0; s < 3; ++s)
            if (ep[s].pending && ep[s].p0 == p0 && ep[s].T == T) return;
        EPrep& e = ep[ep_next];
        if (e.fut.valid()) e.fut.wait();
        if (e.have_used) e.used.wait();
        e.p0 = p0;
        e.T = T;
        e.pending = true;
        e.fut = std::async(std::launch::async, [this, &e, tok_all, p0, T]() { ep_compute(&e, tok_all, p0, T); });
        ep_next = (ep_next + 1) % 3;
    }
    bool engram_prep(const int32_t* tok_all, int p0, int T, std::string& err) {
        const double t0 = now_s();
        EPrep* e = nullptr;
        for (int s = 0; s < 3; ++s)
            if (ep[s].pending && ep[s].p0 == p0 && ep[s].T == T) e = &ep[s];
        if (e) {
            e->fut.get();
            e->pending = false;
        } else {
            for (int s = 0; s < 3; ++s)
                if (ep[s].fut.valid()) ep[s].fut.wait();
            e = &ep[ep_next];
            if (e->have_used) e->used.wait();
            ep_next = (ep_next + 1) % 3;
            ep_compute(e, tok_all, p0, T);
        }
        if (!e->ok) {
            err = e->err;
            return false;
        }
        sycl::event last;
        for (int mo = 0; mo < 4; ++mo) {
            Q().memcpy(d_eraw[mo], e->raw.data() + (size_t)mo * Tc * 16 * 100, (size_t)T * 16 * 100);
            last = Q().memcpy(d_evalid[mo], e->valid.data() + (size_t)mo * Tc * 16, (size_t)T * 16);
        }
        e->used = last;
        e->have_used = true;
        t_engram_cpu += now_s() - t0;
        return true;
    }

    // Diagnostics (hash / tap / dump)
    uint64_t* d_hash = nullptr;
    std::vector<std::string> hash_names;
    void tap_hash(const std::string& name, const float* dev, size_t n) {
        if (!d_hash || hash_names.size() >= 60000) return;
        const size_t slot = hash_names.size();
        hash_names.push_back("c" + std::to_string(cur_p0) + " " + name);
        uint64_t* out = d_hash + slot;
        Q().memset(out, 0, 8);
        const size_t G = 128;
        Q().parallel_for(NR(G, 256), [=](sycl::nd_item<1> it) {
            uint64_t acc = 0;
            for (size_t i = it.get_global_id(0); i < n; i += G * 256) {
                uint32_t b;
                std::memcpy(&b, dev + i, 4);
                uint64_t v = (uint64_t)b * 0x9E3779B97F4A7C15ull + (uint64_t)i * 0xD6E8FEB86659FD93ull;
                v ^= v >> 29;
                v *= 0xBF58476D1CE4E5B9ull;
                v ^= v >> 32;
                acc += v;
            }
            acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<uint64_t>());
            if (it.get_local_id(0) == 0) {
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> a(*out);
                a.fetch_add(acc);
            }
        });
    }
    void hash_flush() {
        if (!d_hash || hash_names.empty()) return;
        wait();
        std::vector<uint64_t> h(hash_names.size());
        d2h(h.data(), d_hash, h.size() * 8);
        wait();
        for (size_t i = 0; i < h.size(); ++i) fprintf(stderr, "[hash] %s %016llx\n", hash_names[i].c_str(), (unsigned long long)h[i]);
        hash_names.clear();
    }

    std::vector<std::pair<std::string, std::vector<float>>>* tap = nullptr;
    void tap_dev(const std::string& name, const void* dev, size_t n, int kind) {
        if (!tap) return;
        wait();
        std::vector<float> h(n);
        if (kind == 0) d2h(h.data(), dev, n * 4);
        else if (kind == 1) {
            std::vector<half> hh(n);
            d2h(hh.data(), dev, n * 2);
            wait();
            for (size_t i = 0; i < n; ++i) h[i] = (float)hh[i];
        } else {
            std::vector<int32_t> hi(n);
            d2h(hi.data(), dev, n * 4);
            wait();
            for (size_t i = 0; i < n; ++i) h[i] = (float)hi[i];
        }
        wait();
        tap->push_back({name, std::move(h)});
    }
    void tap_keys(const std::string& name, int T) {
        if (!tap) return;
        wait();
        std::vector<int32_t> hk((size_t)T * index_topk), hn(T);
        d2h(hk.data(), keys, hk.size() * 4);
        d2h(hn.data(), nkeys, (size_t)T * 4);
        wait();
        std::vector<float> h((size_t)T * index_topk, 0.f), hnf(T);
        for (int t = 0; t < T; ++t) {
            hnf[t] = (float)hn[t];
            for (int j = 0; j < hn[t] && j < index_topk; ++j) h[(size_t)t * index_topk + j] = (float)hk[(size_t)t * index_topk + j];
        }
        tap->push_back({name + "_n", std::move(hnf)});
        tap->push_back({name, std::move(h)});
    }

    void maybe_dump(const std::string& name, int T, int D, const float* dev, int per_tok_streams = 0) {
        if (d_hash) tap_hash(name, dev, (size_t)T * D);
        if (tap) tap_dev(name, dev, (size_t)T * D, 0);
        if (!dump) return;
        wait();
        const size_t n = (size_t)T * D;
        std::vector<float> h(n);
        d2h(h.data(), dev, n * 4);
        wait();
        if (per_tok_streams) save_npy_f32(dump_dir + "/" + name + ".npy", h.data(), {T, per_tok_streams, D / per_tok_streams});
        else save_npy_f32(dump_dir + "/" + name + ".npy", h.data(), {T, D});
    }

    int engram_mod_of(int il) const {
        if (!engram_ready) return -1;
        for (int j = 0; j < 4; ++j)
            if (emeta.layers[j] == il) return j;
        return -1;
    }

    void engram_layer(int mo, int il, int T) {
        const std::string p = L(il) + ".engram_layer";
        EngramDev& e = eng_dev[mo];
        engram_rows(d_eraw[mo], e.lnw, d_evalid[mo], er_h, T);
        for (int j = 0; j < 5; ++j) {
            const std::string nm = j < 4 ? p + ".key_projs." + std::to_string(j) + ".weight" : p + ".value_proj.weight";
            const hy4::TInfo& t = ti(nm);
            linear_auto(t.fmt, tw(nm), (int)t.rows(), (int)t.cols(), er_h, kv_f + (size_t)j * T * HID, T, g.is_rp(nm));
        }
        engram_apply(x, kv_f, e.kvb, e.egate, e.pre_gamma, e.hcs, d_evalid[mo], T);
    }

    // Deterministic two-launch hc_mix2 + fused reduce_norm (f96da6f22 preserved: zero atomics, cuts 84 launches/step!)
    void hc_mix_run(const float* xin, const float* hfn, const float* hsc, const float* hbs, int nout, int T) {
        hc_mix2(xin, hfn, hsc, hbs, pre, post, nout, c.rms_eps, c.hc_eps, c.hc_mag, T, hc_part, hc_cnt);
    }
    void hc_pre_run(const float* hfn, const float* hsc, const float* hbs, const float* nw, int nout, int T, const float* xin, half* outh, float* outf) {
        hc_mix2_reduce_norm(xin, hfn, hsc, hbs, nw, hc_part, pre, post, outh, outf, nout, c.rms_eps, c.hc_eps, c.hc_mag, c.rms_eps, T);
    }

    bool fuse_dec = false;
    float *kva2 = nullptr, *sg_f = nullptr, *su_f = nullptr, *so_f = nullptr;
    std::vector<GemvMulti> mg_att, mg_m1, mg_shd, mg_up;
    std::vector<char> mg_att_ok, mg_moe_ok;
    bool mg_add(GemvMulti& m, const std::string& name, float* Y, int xmode, const half* Xh, const float* Xf, const float* Xg, const float* Xu) {
        const hy4::TInfo& t = ti(name);
        if (!g.is_rp(name) || t.fmt != FMT_S4 || t.shape.size() != 2 || m.n >= 3) return false;
        if (m.n == 0) m.K = (int)t.cols();
        else if (m.K != (int)t.cols()) return false;
        GemvJob& j = m.j[m.n++];
        j.W = (const uint8_t*)tw(name);
        j.Y = Y; j.Xh = Xh; j.Xf = Xf; j.Xg = Xg; j.Xu = Xu;
        j.N = (int)t.rows();
        j.row0 = m.total;
        j.xmode = xmode;
        m.total += j.N;
        return true;
    }
    void build_fuse() {
        const char* e = std::getenv("HY4S_FUSE");
        fuse_dec = !(e && e[0] == '0');  // ON by default: fuses q_a+kv_a, down+shared_g+u, swiglu+shared_dn, f2h+routed_up (zero atomics, bitwise identical!)
        const int NL = c.n_layers;
        mg_att.assign(NL, GemvMulti());
        mg_m1.assign(NL, GemvMulti());
        mg_shd.assign(NL, GemvMulti());
        mg_up.assign(NL, GemvMulti());
        mg_att_ok.assign(NL, 0);
        mg_moe_ok.assign(NL, 0);
        if (!fuse_dec) return;
        kva2 = dalloc<float>(576);
        sg_f = dalloc<float>(1536);
        su_f = dalloc<float>(1536);
        so_f = dalloc<float>(HID);
        float* z1 = d_pack + 32;
        for (int il = 0; il < NL; ++il) {
            const std::string pa = L(il) + ".self_attn", pm = L(il) + ".mlp";
            mg_att_ok[il] = mg_add(mg_att[il], pa + ".q_a_proj.weight", F1, 0, cur_h, nullptr, nullptr, nullptr) && mg_add(mg_att[il], pa + ".kv_a_proj_with_mqa.weight", kva2, 0, cur_h, nullptr, nullptr, nullptr);
            if (il == 0) continue;
            mg_moe_ok[il] = mg_add(mg_m1[il], pm + ".routed_expert_down_proj.weight", z1, 0, cur_h, nullptr, nullptr, nullptr) &&
                            mg_add(mg_m1[il], pm + ".shared_experts.gate_proj.weight", sg_f, 0, cur_h, nullptr, nullptr, nullptr) &&
                            mg_add(mg_m1[il], pm + ".shared_experts.up_proj.weight", su_f, 0, cur_h, nullptr, nullptr, nullptr) &&
                            mg_add(mg_shd[il], pm + ".shared_experts.down_proj.weight", so_f, 2, nullptr, nullptr, sg_f, su_f) &&
                            mg_add(mg_up[il], pm + ".routed_expert_up_proj.weight", F2, 1, nullptr, zout_f, nullptr, nullptr);
        }
    }

    // IndexPool helpers (both eager host-p0 and graph device-p0)
    void idx_update(int il, int p0, int T, const StepParams* dp = nullptr) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        lin(p + ".token_key_projection.weight", cur_h, ik_f, T);
        rms_rows(ik_f, (const float*)tw(p + ".token_key_norm.weight"), nullptr, ik_h, 128, c.rms_eps, T, 128);
        lin(p + ".key_value_projection.weight", ik_h, iv_f, T);
        lin(p + ".key_gate_projection.weight", ik_h, ig_f, T);
        if (dp) {
            idx::store_dev(iv_f, ig_f, ivg[il], dp, T, iring);
            idx::pool_dev(ivg[il], (const float*)tw(p + ".intra_group_ape"), (const float*)tw(p + ".key_norm.weight"), pooled[il], dp, T, (float)theta(il), c.rms_eps, iring);
        } else {
            idx::store(iv_f, ig_f, ivg[il], p0, T, iring);
            const int g_lo = p0 / 4, g_hi = (p0 + T) / 4 - 1;
            if (g_hi >= g_lo) idx::pool(ivg[il], (const float*)tw(p + ".intra_group_ape"), (const float*)tw(p + ".key_norm.weight"), pooled[il], g_lo, g_hi - g_lo + 1, (float)theta(il), c.rms_eps, iring);
            if (vis && g_hi >= g_lo && vis->needs_fix(p0, T)) vis->pool_fix(pooled[il], g_lo, g_hi - g_lo + 1, (float)theta(il));
        }
    }

    void idx_select(int il, int p0, int T, const StepParams* dp = nullptr) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        lin(p + ".query_projection.weight", qr_h, iq_f, T);
        if (dp) {
            idx::q_rope_dev(iq_f, iq_h, dp, (float)theta(il), T);
        } else {
            if (vis && vis->needs_fix(p0, T)) vis->iq_rope(iq_f, iq_h, T, p0, (float)theta(il));
            else idx::q_rope(iq_f, iq_h, p0, (float)theta(il), T);
        }
        lin(p + ".head_weight_projection.weight", cur_h, hw_f, T);
        const int n_top = index_topk / c.kpool - 1;
        if (dp) {
            idx::score_dev(iq_h, hw_f, pooled[il], score, dp, (int)ldS, 1.0f / std::sqrt((float)(128 * 16)), T);
            idx::select_dev(score, (int)ldS, dp, n_top, index_topk, keys, nkeys, T);
        } else {
            for (int t0 = 0; t0 < T; t0 += score_rows) {
                const int tb = std::min(score_rows, T - t0);
                int G = (p0 + t0 + tb) / 4;
                if (G < 1) G = 1;
                idx::score(iq_h + (size_t)t0 * 2048, hw_f + (size_t)t0 * 16, pooled[il], score, p0 + t0, G, (int)ldS, 1.0f / std::sqrt((float)(128 * 16)), tb);
                idx::select(score, (int)ldS, p0 + t0, n_top, index_topk, keys + (size_t)t0 * index_topk, nkeys + t0, tb);
            }
        }
    }

    // Core attention kernels (works for both eager host-p0 and command_graph device-p0!)
    void attention_layer_core(int il, int p0, int T, const StepParams* dp, bool force_use_list,
                              int g_TK, int g_S, int g_tps) {
        const std::string pa = L(il) + ".self_attn";
        const int NH = heads(il);
        const bool swa = c.is_swa[il];
        const float* hfn = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_base");
        { PS _q(this, "hc_pre"); hc_pre_run(hfn, hsc, hbs, (const float*)tw(L(il) + ".input_layernorm.weight"), 8, T, x, cur_h, nullptr); }
        const bool fz = T == 1 && fuse_dec && !dump && mg_att_ok[il];
        if (fz) { PS _q(this, "lin q_a+kv_a"); k_gemv_multi(mg_att[il]); }
        else lin(pa + ".q_a_proj.weight", cur_h, F1, T);
        { PS _q(this, "q_a_rms"); rms_rows(F1, (const float*)tw(pa + ".q_a_layernorm.weight"), nullptr, qr_h, c.q_lora, c.rms_eps, T, 256); }
        lin(pa + ".q_b_proj.weight", qr_h, F2, T);
        if (!fz) lin(pa + ".kv_a_proj_with_mqa.weight", cur_h, F1, T);
        half* cache = kcache[il];
        const int cslots = swa ? ring : slots_sparse;
        if (dp) {
            mla_write_kv_dev(fz ? kva2 : F1, (const float*)tw(pa + ".kv_a_layernorm.weight"), cache, cslots, T, dp, false, swa ? ring : 0, (float)theta(il), c.rms_eps);
        } else {
            PS _q(this, "mla_write_kv");
            mla_write_kv(fz ? kva2 : F1, (const float*)tw(pa + ".kv_a_layernorm.weight"), cache, cslots, T, p0, swa ? ring : 0, (float)theta(il), c.rms_eps);
        }

        const std::string kvb_nm = pa + ".kv_b_proj.weight";
        const int kvb_fmt = ti(kvb_nm).fmt;
        const void* kvb_w = tw(kvb_nm);
        if (T <= 16) {
            PS _q(this, "mla_prep_q_sk");
            mla_prep_q_sk(kvb_fmt, kvb_w, F2, qa_h, T, NH, p0, dp, false, (float)theta(il));
        } else {
            PS _q(this, "mla_prep_q_xmx");
            split_kvb(kvb_fmt, kvb_w, wk_tmp, wv_tmp, NH);
            mla_prep_q(F2, wk_tmp, qa_h, T, NH, p0, (float)theta(il));
        }
        if (!dp && vis && vis->needs_fix(p0, T)) {
            vis->fixk(fz ? kva2 : F1, cache, T, p0, swa ? ring : 0, (float)theta(il));
            vis->fixq(F2, qa_h, T, NH, p0, (float)theta(il));
        }

        bool use_list = false;
        if (!swa && index_on) {
            { PS _q(this, "idx_update"); idx_update(il, p0, T, dp); }
            const bool do_sel = dp ? force_use_list : (p0 + T > index_topk);
            if (do_sel) {
                { PS _q(this, "idx_select"); idx_select(il, p0, T, dp); }
                if (!dp && tap) tap_keys("L" + std::to_string(il) + "_keys", T);
                use_list = true;
            }
        }
        if (!dp) last_use_list = use_list;
        if (dfly_cap && !swa) {
            if (dp) dfly_capture_copy(il, T, use_list);
            else dfly_capture(il, p0, T, use_list);
        }

        lin(pa + ".linear_gate.weight", cur_h, F3, T);
        MlaAttnArgs a;
        a.M = T; a.NH = NH; a.p0 = p0;
        a.mode = swa ? MLA_WINDOW : (use_list ? MLA_LIST : MLA_CAUSAL);
        a.window = c.window;
        a.ring = swa ? ring : 0;
        a.cache_slots = cslots;
        if (use_list) { a.keys = keys; a.nkeys = nkeys; a.kmax = index_topk; }
        if (dp) {
            mla_attend_dev(qa_h, cache, (const float*)tw(pa + ".learnable_sink_param"), ctx_h, a, dp, false, g_TK, g_S, g_tps, mla_ws, mla_ws_bytes);
        } else {
            ensure_mla_ws(mla_workspace_bytes(a));
            PS _q(this, "mla_attend");
            mla_attend(qa_h, cache, (const float*)tw(pa + ".learnable_sink_param"), ctx_h, a, mla_ws, mla_ws_bytes);
        }
        if (!dp && tap) tap_dev("L" + std::to_string(il) + "_ctx", ctx_h, (size_t)T * NH * 512, 1);
        if (T <= 16) {
            PS _q(this, "mla_vup_sk");
            mla_vup_sk(kvb_fmt, kvb_w, ctx_h, F3, o_h, T, NH);
        } else {
            PS _q(this, "mla_vup_xmx");
            mla_vup(ctx_h, wv_tmp, F3, o_h, T, NH);
        }
        lin(pa + ".o_proj.weight", o_h, F1, T);
        { PS _q(this, "o_norm_post"); add_rms_post(F1, nullptr, (const float*)tw(pa + ".post_proj_norm.weight"), x, post, HID, c.rms_eps, T); }
    }

    void attention_layer(int il, int p0, int T) {
        attention_layer_core(il, p0, T, nullptr, false, 0, 0, 0);
    }

    // MoE layer sub-stages (shared between eager and command_graph!)
    void mlp_dense0(int T) {
        const std::string pm = L(0) + ".mlp";
        const float* hfn = (const float*)tw(L(0) + ".hc_mlp_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(0) + ".hc_mlp_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(0) + ".hc_mlp_layer.hc_pre.hc_base");
        { PS _q(this, "hc_pre"); hc_pre_run(hfn, hsc, hbs, (const float*)tw(L(0) + ".post_attention_layernorm.weight"), 8, T, x, cur_h, cur_f); }
        lin(pm + ".gate_proj.weight", cur_h, F1, T);
        lin(pm + ".up_proj.weight", cur_h, F2, T);
        { PS _q(this, "swiglu"); swiglu_h(F1, F2, H1, (size_t)T * 13824); }
        lin(pm + ".down_proj.weight", H1, F3, T);
        { PS _q(this, "o_norm_post"); add_rms_post(F3, nullptr, (const float*)tw(pm + ".post_fc2_norm.weight"), x, post, HID, c.rms_eps, T); }
    }

    void mlp_moe_pre(int il, int T) {
        const std::string pm = L(il) + ".mlp";
        const float* hfn = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_base");
        { PS _q(this, "hc_pre"); hc_pre_run(hfn, hsc, hbs, (const float*)tw(L(il) + ".post_attention_layernorm.weight"), 8, T, x, cur_h, cur_f); }
        moe_idx = (int32_t*)d_pack; moe_w = d_pack + (size_t)T * 16; z_f = d_pack + (size_t)T * 32;
        h_idx = (int32_t*)h_pack; h_w = h_pack + (size_t)T * 16; h_z = h_pack + (size_t)T * 32;
        { PS _q(this, "router"); linear_f32((const float*)tw(pm + ".gate.weight"), 512, HID, cur_f, F1, T); }
        { PS _q(this, "router_topk"); router_topk_sg(F1, (const float*)tw(pm + ".gate.e_score_correction_bias"), moe_idx, moe_w, 16, c.routed_scale, T); }
        if (d_hash) { tap_hash("L" + std::to_string(il) + "_router_logits", F1, (size_t)T * 512); tap_hash("L" + std::to_string(il) + "_router_w", moe_w, (size_t)T * 16); tap_hash("L" + std::to_string(il) + "_router_idx", (const float*)moe_idx, (size_t)T * 16); }
        const bool fm = T == 1 && fuse_dec && !dump && mg_moe_ok[il];
        if (fm) { PS _q(this, "lin down+shared gate/up"); k_gemv_multi(mg_m1[il]); }
        else lin(pm + ".routed_expert_down_proj.weight", cur_h, z_f, T);
        if (d_hash) tap_hash("L" + std::to_string(il) + "_routed_down_z", z_f, (size_t)T * 1536);
    }

    void mlp_moe_shared(int il, int T) {
        const std::string pm = L(il) + ".mlp";
        const size_t nz = (size_t)T * 1536;
        const bool fm = T == 1 && fuse_dec && !dump && mg_moe_ok[il];
        if (fm) {
            k_gemv_multi(mg_shd[il]);
        } else {
            lin(pm + ".shared_experts.gate_proj.weight", cur_h, F1, T);
            lin(pm + ".shared_experts.up_proj.weight", cur_h, F3, T);
            { PS _q(this, "swiglu"); swiglu_h(F1, F3, H1, nz); }
            lin(pm + ".shared_experts.down_proj.weight", H1, F3, T);
        }
        if (d_hash && !fm) tap_hash("L" + std::to_string(il) + "_shared_out", F3, (size_t)T * HID);
    }

    void mlp_moe_post(int il, int T) {
        const std::string pm = L(il) + ".mlp";
        const size_t nz = (size_t)T * 1536;
        const bool fm = T == 1 && fuse_dec && !dump && mg_moe_ok[il];
        if (d_hash) tap_hash("L" + std::to_string(il) + "_routed_zout", zout_f, nz);
        if (fm) {
            k_gemv_multi(mg_up[il]);
        } else {
            f2h(zout_f, z_h, nz);
            lin(pm + ".routed_expert_up_proj.weight", z_h, F2, T);
        }
        if (d_hash) tap_hash("L" + std::to_string(il) + "_routed_up_out", F2, (size_t)T * HID);
        { PS _q(this, "combine_norm_post"); add_rms_post(F2, fm ? so_f : F3, (const float*)tw(pm + ".combine_output_norm.weight"), x, post, HID, c.rms_eps, T); }
    }

    // Evaluate routed experts for MoE layer `il` (1..41) after `evw.wait()` (works identically for eager and graphed paths!)
    void run_routed_experts(int il, int T, bool use_gx) {
        const size_t nz = (size_t)T * 1536;
        const double tc = now_s();
        if (!force_route_dir.empty()) {
            std::vector<float> fi, fw;
            if (!load_npy_f32(force_route_dir + "/L" + std::to_string(il) + "_topk_idx.npy", fi) ||
                !load_npy_f32(force_route_dir + "/L" + std::to_string(il) + "_topk_w.npy", fw) ||
                fi.size() < (size_t)(cur_p0 + T) * 16) {
                fprintf(stderr, "force-route: cannot load layer %d\n", il);
                exit(5);
            }
            wait();
            for (size_t i = 0; i < (size_t)T * 16; ++i) { h_idx[i] = (int32_t)fi[(size_t)cur_p0 * 16 + i]; h_w[i] = fw[(size_t)cur_p0 * 16 + i]; }
        }
        if (use_gx) {
            gx->run(il - 1, T, z_f, h_idx, h_w, zout_f, nullptr);
            t_cpu_exp += now_s() - tc;
            return;
        }
        if (std::getenv("HY4S_SKIP_EXPERTS")) {
            dzero(zout_f, nz * 4);
            t_cpu_exp += now_s() - tc;
            return;
        }
        // Decode / Verify (T <= 16): overlap CPU misses (submitted FIRST!) with GPU VRAM cache hits!
        int vhits = 0;
        if (vcache && vcache->active() && T <= SyVramExpertCache::MAX_T) {
            vhits = vcache->plan_decode(il - 1, T, h_idx, h_w, h_use_cpu);
            if (vhits < T * 16) ucx.submit(il - 1, T, h_z, h_idx, h_w, h_use_cpu, h_zout, false);
            if (vhits > 0) vcache->launch_decode(z_f, zout_gpu_f);
            if (vhits < T * 16) ucx.wait();
        } else {
            ucx.submit(il - 1, T, h_z, h_idx, h_w, nullptr, h_zout, false);
            ucx.wait();
        }
        t_cpu_exp += now_s() - tc;

        if (vhits == T * 16) {
            d2d(zout_f, zout_gpu_f, nz * 4);
        } else if (vhits == 0) {
            h2d(zout_f, h_zout, nz * 4);
        } else {
            h2d(zout_f, h_zout, nz * 4);
            SyVramExpertCache::add_zout(zout_f, zout_gpu_f, nz);
        }
        if (dump && dbg_layers) {
            wait();
            const std::string nm = "L" + std::to_string(il) + "_";
            save_npy_f32(dump_dir + "/" + nm + "z_folded.npy", h_z, {T, 1536});
            std::vector<float> ti_((size_t)T * 16);
            for (size_t i = 0; i < ti_.size(); ++i) ti_[i] = (float)h_idx[i];
            save_npy_f32(dump_dir + "/" + nm + "topk_idx.npy", ti_.data(), {T, 16});
            save_npy_f32(dump_dir + "/" + nm + "topk_w.npy", h_w, {T, 16});
            std::vector<float> hzo(nz);
            d2h(hzo.data(), zout_f, nz * 4);
            wait();
            save_npy_f32(dump_dir + "/" + nm + "z_out.npy", hzo.data(), {T, 1536});
            maybe_dump(nm + "shared_out", T, HID, F3);
        }
    }

    void mlp_layer(int il, int T) {
        if (il == 0) {
            mlp_dense0(T);
            return;
        }
        const size_t nz = (size_t)T * 1536;
        mlp_moe_pre(il, T);
        const bool gx_now = gx != nullptr && T >= gx_min_T;
        sycl::event evw = Q().memcpy(h_pack, d_pack, ((size_t)T * 32 + ((gx_now && !(dump && dbg_layers)) ? 0 : nz)) * 4);
        mlp_moe_shared(il, T);
        evw.wait();
        run_routed_experts(il, T, gx_now);
        mlp_moe_post(il, T);
    }

#if HY4S_HAS_GRAPH
    GraphSet* get_or_build_graph(int p0, int T) {
        const bool use_list = index_on && (p0 + T > index_topk);
        MlaAttnArgs a_swa, a_sp;
        a_swa.M = T; a_swa.NH = c.heads_swa; a_swa.p0 = p0; a_swa.mode = MLA_WINDOW; a_swa.window = c.window; a_swa.ring = ring; a_swa.cache_slots = ring;
        a_sp.M = T; a_sp.NH = c.heads_sparse; a_sp.p0 = p0; a_sp.mode = use_list ? MLA_LIST : MLA_CAUSAL; a_sp.kmax = index_topk; a_sp.cache_slots = slots_sparse;
        GraphKey k;
        k.T = T;
        k.use_list = use_list ? 1 : 0;
        k.dfly_cap = dfly_cap ? 1 : 0;
        k.engram_on = engram_ready ? 1 : 0;
        mla_choose_splits(a_swa, k.swa_TK, k.swa_S, k.swa_tps);
        mla_choose_splits(a_sp, k.sp_TK, k.sp_S, k.sp_tps);
        ensure_mla_ws(std::max(mla_workspace_bytes(a_swa), mla_workspace_bytes(a_sp)));

        for (auto& gs : gcache_)
            if (gs->key == k) return gs.get();

        // Cap cache at 12 graph sets to bound driver command-buffer memory
        if (gcache_.size() >= 12) {
            wait();
            gcache_.clear();
        }

        auto record = [&](auto&& fn) -> std::unique_ptr<ExecGraph> {
            sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable> cg(Q().get_context(), Q().get_device());
            cg.begin_recording(Q());
            fn();
            cg.end_recording(Q());
            return std::make_unique<ExecGraph>(cg.finalize());
        };

        try {
            wait();
            auto gs = std::make_unique<GraphSet>();
            gs->key = k;
            gs->g_shared.resize(c.n_layers);
            gs->g_post_next.resize(c.n_layers);

            auto att_dev = [&](int il) {
                const bool swa = c.is_swa[il];
                attention_layer_core(il, p0, T, d_step, use_list,
                                     swa ? k.swa_TK : k.sp_TK, swa ? k.swa_S : k.sp_S, swa ? k.swa_tps : k.sp_tps);
            };

            // 1. g_init: embed + Layer 0 (engram if on L0 + attn + dense MLP)
            gs->g_init = record([&]() {
                embed_hc((const uint8_t*)tw("model.embed_tokens.weight"), d_ids, x, T);
                const int mo0 = engram_mod_of(0);
                if (mo0 >= 0) engram_layer(mo0, 0, T);
                att_dev(0);
                mlp_dense0(T);
            });

            // 2. g_l1_pre: Layer 1 (engram + attn + moe_pre)
            gs->g_l1_pre = record([&]() {
                const int mo1 = engram_mod_of(1);
                if (mo1 >= 0) engram_layer(mo1, 1, T);
                att_dev(1);
                mlp_moe_pre(1, T);
            });

            // 3. Per MoE layer il = 1..41: g_shared[il] and g_post_next[il] (post(il) fused with pre(il+1)!)
            for (int il = 1; il < c.n_layers; ++il) {
                gs->g_shared[il] = record([&]() {
                    mlp_moe_shared(il, T);
                });
                gs->g_post_next[il] = record([&]() {
                    mlp_moe_post(il, T);
                    if (il + 1 < c.n_layers) {
                        const int mo = engram_mod_of(il + 1);
                        if (mo >= 0) engram_layer(mo, il + 1, T);
                        att_dev(il + 1);
                        mlp_moe_pre(il + 1, T);
                    }
                });
            }
            ++graph_captures;
            GraphSet* ret = gs.get();
            gcache_.push_back(std::move(gs));
            return ret;
        } catch (const sycl::exception& e) {
            fprintf(stderr, "[graph] capture failed (%s), falling back to eager queue\n", e.what());
            graph_enabled = false;
            return nullptr;
        }
    }

    bool forward_chunk_graphed(const int32_t* tok_all, int p0, int T, std::string& err) {
        GraphSet* gs = get_or_build_graph(p0, T);
        if (!gs) return false;
        const size_t nz = (size_t)T * 1536;
        moe_idx = (int32_t*)d_pack; moe_w = d_pack + (size_t)T * 16; z_f = d_pack + (size_t)T * 32;
        h_idx = (int32_t*)h_pack; h_w = h_pack + (size_t)T * 16; h_z = h_pack + (size_t)T * 32;

        h_step->p0 = p0;
        h2d(d_step, h_step, sizeof(StepParams));
        h2d(d_ids, tok_all + p0, (size_t)T * 4);
        const bool defer_eng = engram_ready;
        if (defer_eng) prefetch_engram(tok_all, p0, T);

        Q().ext_oneapi_graph(*gs->g_init);

        if (defer_eng) {
            const double tw0 = now_s();
            if (!engram_prep(tok_all, p0, T, err)) return false;
            t_engram_wait += now_s() - tw0;
        }

        Q().ext_oneapi_graph(*gs->g_l1_pre);
        for (int il = 1; il < c.n_layers; ++il) {
            sycl::event evw = Q().memcpy(h_pack, d_pack, ((size_t)T * 32 + nz) * 4);
            Q().ext_oneapi_graph(*gs->g_shared[il]);
            evw.wait();
            run_routed_experts(il, T, false);
            Q().ext_oneapi_graph(*gs->g_post_next[il]);
        }

        const bool use_list = gs->key.use_list != 0;
        last_use_list = use_list;
        if (dfly_cap) {
            cap_p0 = p0;
            cap_rows = T;
            for (int d = 0; d < 5; ++d) cap_list[d] = use_list;
        }
        ++graph_replays;
        return true;
    }
#endif

    // Process tokens [p0, p0+T) (T <= Tc) of tok_all. Returns false on error.
    bool forward_chunk(const int32_t* tok_all, int p0, int T, std::string& err) {
        cur_p0 = p0;
#if HY4S_HAS_GRAPH
        if (graph_enabled && T <= 8 && !dump && !tap && !prof && !prof2 && !d_hash && vis == nullptr && force_route_dir.empty()) {
            if (forward_chunk_graphed(tok_all, p0, T, err)) return true;
            if (!err.empty()) return false;
            // Fallback to eager if capture disabled graph_enabled
        }
#endif
        h2d(d_ids, tok_all + p0, (size_t)T * 4);
        embed_hc((const uint8_t*)tw("model.embed_tokens.weight"), d_ids, x, T);
        if (vis) vis->embed(x, p0, T);
        const int eng_first_layer = engram_ready ? emeta.layers[0] : 0;
        const bool defer_eng = engram_ready && (T <= 8 || eng_first_layer > 0);
        if (defer_eng) prefetch_engram(tok_all, p0, T);
        else if (engram_ready && !engram_prep(tok_all, p0, T, err)) return false;
        for (int il = 0; il < c.n_layers; ++il) {
            if (defer_eng && il == std::max(1, eng_first_layer)) {
                double tw0 = now_s();
                if (!engram_prep(tok_all, p0, T, err)) return false;
                t_engram_wait += now_s() - tw0;
            }
            const int mo = engram_mod_of(il);
            double ta = 0;
            if (prof) { wait(); ta = now_s(); }
            if (mo >= 0 && engram_ready) {
                { PS _q(this, "engram_layer"); engram_layer(mo, il, T); }
                maybe_dump("x_L" + std::to_string(il) + "_engram", T, 4 * HID, x, 4);
            }
            if (prof) { wait(); double tb = now_s(); p_eng += tb - ta; ta = tb; }
            attention_layer(il, p0, T);
            maybe_dump("x_L" + std::to_string(il) + "_attn", T, 4 * HID, x, 4);
            if (prof) { wait(); double tb = now_s(); (c.is_swa[il] ? p_attn_swa : p_attn_sp) += tb - ta; ta = tb; }
            mlp_layer(il, T);
            maybe_dump("x_L" + std::to_string(il) + "_mlp", T, 4 * HID, x, 4);
            if (prof) { wait(); double tb = now_s(); (il == 0 ? p_mlp0 : p_moe) += tb - ta; }
        }
        hash_flush();
        return true;
    }

    bool prof = false;
    double p_eng = 0, p_attn_swa = 0, p_attn_sp = 0, p_mlp0 = 0, p_moe = 0;
    void prof_report(const char* what) {
        if (!prof) return;
        fprintf(stderr, "[prof %s] engram %.3f s | attn swa(30) %.3f s sparse(12) %.3f s | mlp0 %.3f s | moe(41) %.3f s (cpu experts %.3f s)\n",
                what, p_eng, p_attn_swa, p_attn_sp, p_mlp0, p_moe, t_cpu_exp);
        p_eng = p_attn_swa = p_attn_sp = p_mlp0 = p_moe = 0;
    }

    // logits of chunk positions [first, T) into d logits [(T-first)][V]
    void head(int T, int first) {
        const int n = T - first;
#if HY4S_HAS_GRAPH
        if (graph_enabled && first == 0 && n >= 1 && n <= 8 && !prof2 && !dump && !tap) {
            if (!g_head_[n]) {
                try {
                    wait();
                    sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable> cg(Q().get_context(), Q().get_device());
                    cg.begin_recording(Q());
                    const float* hfn = (const float*)tw("model.hc_head.hc_head_fn");
                    const float* hsc = (const float*)tw("model.hc_head.hc_head_scale");
                    const float* hbs = (const float*)tw("model.hc_head.hc_head_base");
                    hc_pre_run(hfn, hsc, hbs, (const float*)tw("model.norm.weight"), 4, n, x, hfin, nullptr);
                    const hy4::TInfo& lm = ti("lm_head.weight");
                    linear_auto(lm.fmt, tw("lm_head.weight"), (int)lm.rows(), (int)lm.cols(), hfin, logits, n, g.is_rp("lm_head.weight"));
                    cg.end_recording(Q());
                    g_head_[n] = std::make_unique<ExecGraph>(cg.finalize());
                } catch (const sycl::exception&) {
                    graph_enabled = false;
                }
            }
            if (g_head_[n]) {
                Q().ext_oneapi_graph(*g_head_[n]);
                return;
            }
        }
#endif
        const float* hfn = (const float*)tw("model.hc_head.hc_head_fn");
        const float* hsc = (const float*)tw("model.hc_head.hc_head_scale");
        const float* hbs = (const float*)tw("model.hc_head.hc_head_base");
        const float* xs = x + (size_t)first * 4 * HID;
        { PS _q(this, "head_hc_pre"); hc_pre_run(hfn, hsc, hbs, (const float*)tw("model.norm.weight"), 4, n, xs, hfin, nullptr); }
        const hy4::TInfo& lm = ti("lm_head.weight");
        linear_auto(lm.fmt, tw("lm_head.weight"), (int)lm.rows(), (int)lm.cols(), hfin, logits, n, g.is_rp("lm_head.weight"));
    }
};

}  // namespace hy4s
