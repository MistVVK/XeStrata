// sy_graph.h - the 42-layer hy_v4_flash forward on an Intel Arc GPU (SYCL): trunk + KV on the GPU, routed experts on the CPU.
// Port of run/hy4_graph.hip.h: chunked prefill + decode, KV fp16 MLA latent cache, absorbed MLA, IndexPool, LatentMoE (experts via CpuExperts), Engram.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <map>
#include <string>
#include <vector>

#include "hy4_engram.h"
#include "hy4s_cpuexp.h"
#include "sy_common.h"
#include "sy_gemm_xmx.h"
#include "sy_gpuexp.h"
#include "sy_indexer.h"
#include "sy_kernels.h"
#include "sy_linear.h"
#include "sy_mla.h"
#include "sy_model.h"

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

struct EngramDev {  // per module, gathered at init
    float* lnw = nullptr;        // [16][96]
    float* kvb = nullptr;        // [5][3072]
    float* egate = nullptr;      // [4][3072] norm_key*norm_query
    float* pre_gamma = nullptr;  // [3072]
    float* hcs = nullptr;        // [4][3072]
};

class Runner {
public:
    SyModel& g;
    const hy4::Cfg& c;
    CpuExperts& cx;
    int max_ctx, Tc;
    int expert_act = 1;  // 0 float activations (oracle math), 1 Q8_1 activations (fast integer path)
    GpuExperts* gx = nullptr;  // routed experts on the GPU for chunks of >= gx_min_T tokens (prefill); null: CPU only
    int gx_min_T = 32;
    int kv_host_above = 24000;  // max_ctx above which the sparse layers' KV caches stay in pinned host memory
    size_t kv_host_bytes = 0;
    int iring = 0;
    // buffers
    int32_t* d_ids = nullptr;
    float *x = nullptr, *pre = nullptr, *post = nullptr, *cur_f = nullptr, *F1 = nullptr, *F2 = nullptr, *F3 = nullptr, *sub = nullptr;
    half *cur_h = nullptr, *qr_h = nullptr, *qa_h = nullptr, *o_h = nullptr, *H1 = nullptr, *z_h = nullptr, *er_h = nullptr, *hfin = nullptr, *ctx_h = nullptr;
    float *z_f = nullptr, *zout_f = nullptr, *moe_w = nullptr, *logits = nullptr;
    int32_t *moe_idx = nullptr, *d_argmax = nullptr;
    float* hc_part = nullptr;
    uint32_t* hc_cnt = nullptr;
    float* h_z = nullptr;     // pinned host: z [T][1536]
    float* h_zout = nullptr;  // pinned host: expert output [T][1536]
    int32_t* h_idx = nullptr;
    float* h_w = nullptr;
    void* mla_ws = nullptr;
    size_t mla_ws_bytes = 0;
    float* kv_f = nullptr;
    uint8_t* d_eraw[4] = {};
    uint8_t* d_evalid[4] = {};
    EngramDev eng_dev[4];
    half* wk_b = nullptr;
    half* wv_b = nullptr;
    std::vector<size_t> wk_off, wv_off;
    std::vector<half*> kcache;
    int ring = 0, slots_sparse = 0, logit_rows = 1;
    // host side engram
    hy4::EngramMeta emeta;
    hy4::EngramIO eio;
    bool engram_ready = false;
    double t_engram_cpu = 0, t_engram_wait = 0, t_cpu_exp = 0;
    bool dump = false, dbg_layers = false;
    std::string force_route_dir;
    int cur_p0 = 0;  // first position of the chunk being processed (forced routing rows)
    std::string dump_dir = ".";
    int index_on = 1;
    int index_topk = 2048;
    // indexer state
    std::vector<half*> ivg, pooled;
    float *ik_f = nullptr, *iv_f = nullptr, *ig_f = nullptr, *iq_f = nullptr, *hw_f = nullptr, *score = nullptr;
    half *ik_h = nullptr, *iq_h = nullptr;
    int32_t *keys = nullptr, *nkeys = nullptr;
    size_t ldS = 0;
    bool last_use_list = false;

    Runner(SyModel& gm, CpuExperts& cxe, int max_ctx_, int max_chunk, int logit_rows_) : g(gm), c(gm.man.c), cx(cxe), max_ctx(max_ctx_), Tc(max_chunk), logit_rows(logit_rows_) {}

    const void* tw(const std::string& name) const { return g.tw(name); }
    const hy4::TInfo& ti(const std::string& name) const { return g.ti(name); }
    std::string L(int il) const { return "model.layers." + std::to_string(il); }
    int heads(int il) const { return c.is_swa[il] ? c.heads_swa : c.heads_sparse; }
    double theta(int il) const { return c.is_swa[il] ? c.theta_swa : c.theta_sparse; }

    // per-section GPU timeline (prof2, needs the profiling queue: HY4S_PROF=1): every labelled section is bracketed by two marker kernels whose GPU start times are read back at
    // the report, so the sections show real GPU time including launch gaps, without draining the queue (the queue keeps running between the markers).
    bool prof2 = false;
    std::map<std::string, std::pair<double, int>> pacc;
    struct PEv { sycl::event a, b; std::string nm; };
    std::vector<PEv> pev;
    struct PS {
        Runner* r;
        bool on;
        PEv e;
        PS(Runner* r_, const std::string& n) : r(r_), on(r_->prof2) {
            if (!on) return;
            e.nm = n;
            e.a = Q().single_task([]() {});
        }
        ~PS() {
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
        o_h = dalloc<half>((size_t)T * 16384);
        ctx_h = dalloc<half>((size_t)T * 64 * 512);
        z_f = dalloc<float>((size_t)T * 1536);
        z_h = dalloc<half>((size_t)T * 1536);
        zout_f = dalloc<float>((size_t)T * 1536);
        moe_idx = dalloc<int32_t>((size_t)T * 16);
        moe_w = dalloc<float>((size_t)T * 16);
        h_z = halloc<float>((size_t)T * 1536);
        h_zout = halloc<float>((size_t)T * 1536);
        h_idx = halloc<int32_t>((size_t)T * 16);
        h_w = halloc<float>((size_t)T * 16);
        hfin = dalloc<half>((size_t)T * HID);
        logits = dalloc<float>((size_t)logit_rows * c.vocab);
        d_argmax = dalloc<int32_t>(1);
        // kv caches
        ring = ((512 + Tc + 15) / 16) * 16;
        slots_sparse = ((max_ctx + 8 + 15) / 16) * 16;
        kcache.resize(c.n_layers);
        size_t kvbytes = 0;
        for (int il = 0; il < c.n_layers; ++il) {
            const int slots = c.is_swa[il] ? ring : slots_sparse;
            const size_t by = mla_cache_bytes(slots);
            // long contexts: the 12 sparse layers' caches (1152 B/token each) live in pinned host memory that the GPU reads over PCIe (the IndexPool attends only ~2052 keys per
            // query on these layers, ~2.4 MB per layer and token), because 128K tokens of KV (1.8 GB) do not fit next to the 6.4 GB trunk in 7.5 GB of VRAM
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
        // indexer state (sparse layers only): iv/ig rows are a ring (the groups of 4 are pooled as soon as they are complete)
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
        score = dalloc<float>((size_t)Tc * ldS);
        keys = dalloc<int32_t>((size_t)Tc * (index_topk + 8));
        nkeys = dalloc<int32_t>((size_t)Tc);
        // absorbed weights
        wk_off.resize(c.n_layers);
        wv_off.resize(c.n_layers);
        size_t tk = 0, tv = 0;
        for (int il = 0; il < c.n_layers; ++il) {
            const int NH = heads(il);
            wk_off[il] = tk;
            wv_off[il] = tv;
            tk += (size_t)NH * 192 * 512;
            tv += (size_t)NH * 256 * 512;
        }
        wk_b = dalloc<half>(tk);
        wv_b = dalloc<half>(tv);
        for (int il = 0; il < c.n_layers; ++il) {
            const std::string nm = L(il) + ".self_attn.kv_b_proj.weight";
            if (!split_kvb(ti(nm).fmt, tw(nm), wk_b + wk_off[il], wv_b + wv_off[il], heads(il))) {
                err = "kv_b_proj format not supported";
                return false;
            }
        }
        wait();
        // engram
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
        kv_f = dalloc<float>((size_t)5 * Tc * HID);
        ep_alloc();
        wait();
        fprintf(stderr, "[runner] kv cache %.2f GB (%.2f GB of it pinned host), ctx %d, chunk %d, device allocated %.2f GiB of 7.54\n", kvbytes / 1e9, kv_host_bytes / 1e9, max_ctx, Tc, dev_bytes_allocated() / 1073741824.0);
        return true;
    }

    // Engram rows of one chunk for all 4 modules (CPU hash + page reads), uploaded to the device
    struct EPrep {
        std::vector<uint8_t> raw, valid;  // [4][Tc*16*100], [4][Tc*16]
        int p0 = -1, T = 0;
        bool ok = false, pending = false;
        std::string err;
        std::future<void> fut;
        sycl::event used;
        bool have_used = false;
    };
    EPrep ep[2];
    int ep_next = 0;
    void ep_alloc() {
        for (int s = 0; s < 2; ++s) {
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
        EPrep& e = ep[ep_next];
        if (e.fut.valid()) e.fut.wait();
        if (e.have_used) e.used.wait();
        e.p0 = p0;
        e.T = T;
        e.pending = true;
        e.fut = std::async(std::launch::async, [this, &e, tok_all, p0, T]() { ep_compute(&e, tok_all, p0, T); });
        ep_next ^= 1;
    }
    bool engram_prep(const int32_t* tok_all, int p0, int T, std::string& err) {
        const double t0 = now_s();
        EPrep* e = nullptr;
        for (int s = 0; s < 2; ++s)
            if (ep[s].pending && ep[s].p0 == p0 && ep[s].T == T) e = &ep[s];
        if (e) {
            e->fut.get();
            e->pending = false;
        } else {
            for (int s = 0; s < 2; ++s)
                if (ep[s].fut.valid()) ep[s].fut.wait();
            e = &ep[ep_next];
            if (e->have_used) e->used.wait();
            ep_next ^= 1;
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

    void maybe_dump(const std::string& name, int T, int D, const float* dev, int per_tok_streams = 0) {
        if (!dump) return;
        wait();
        const size_t n = (size_t)T * D;
        std::vector<float> h(n);
        d2h(h.data(), dev, n * 4);
        wait();
        if (per_tok_streams) save_npy_f32(dump_dir + "/" + name + ".npy", h.data(), {T, per_tok_streams, D / per_tok_streams});
        else save_npy_f32(dump_dir + "/" + name + ".npy", h.data(), {T, D});
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

    void hc_mix_run(const float* xin, const float* hfn, const float* hsc, const float* hbs, int nout, int T) {
        hc_mix2(xin, hfn, hsc, hbs, pre, post, nout, c.rms_eps, c.hc_eps, c.hc_mag, T, hc_part, hc_cnt);
    }

    // IndexPool: cache iv/ig of the chunk's tokens and pool the groups completed by this chunk
    void idx_update(int il, int p0, int T) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        lin(p + ".token_key_projection.weight", cur_h, ik_f, T);
        rms_rows(ik_f, (const float*)tw(p + ".token_key_norm.weight"), nullptr, ik_h, 128, c.rms_eps, T, 128);
        lin(p + ".key_value_projection.weight", ik_h, iv_f, T);
        lin(p + ".key_gate_projection.weight", ik_h, ig_f, T);
        idx::store(iv_f, ig_f, ivg[il], p0, T, iring);
        const int g_lo = p0 / 4, g_hi = (p0 + T) / 4 - 1;
        if (g_hi >= g_lo) idx::pool(ivg[il], (const float*)tw(p + ".intra_group_ape"), (const float*)tw(p + ".key_norm.weight"), pooled[il], g_lo, g_hi - g_lo + 1, (float)theta(il), c.rms_eps, iring);
    }

    // per-token key lists for the chunk (sparse layer, sequence longer than index_topk)
    void idx_select(int il, int p0, int T) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        lin(p + ".query_projection.weight", qr_h, iq_f, T);
        idx::q_rope(iq_f, iq_h, p0, (float)theta(il), T);
        lin(p + ".head_weight_projection.weight", cur_h, hw_f, T);
        int G = (p0 + T) / 4;
        if (G < 1) G = 1;
        idx::score(iq_h, hw_f, pooled[il], score, p0, G, (int)ldS, 1.0f / std::sqrt((float)(128 * 16)), T);
        const int n_top = index_topk / c.kpool - 1;
        idx::select(score, (int)ldS, p0, n_top, index_topk, keys, nkeys, T);
    }

    void attention_layer(int il, int p0, int T) {
        const std::string pa = L(il) + ".self_attn";
        const int NH = heads(il);
        const bool swa = c.is_swa[il];
        const float* hfn = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_base");
        { PS _q(this, "hc_mix"); hc_mix_run(x, hfn, hsc, hbs, 8, T); }
        { PS _q(this, "hc_reduce_norm"); hc_reduce_norm(x, pre, (const float*)tw(L(il) + ".input_layernorm.weight"), cur_h, nullptr, c.rms_eps, T); }
        lin(pa + ".q_a_proj.weight", cur_h, F1, T);  // [T][1024]
        { PS _q(this, "q_a_rms"); rms_rows(F1, (const float*)tw(pa + ".q_a_layernorm.weight"), nullptr, qr_h, c.q_lora, c.rms_eps, T, 256); }
        lin(pa + ".q_b_proj.weight", qr_h, F2, T);  // [T][NH*256]
        lin(pa + ".kv_a_proj_with_mqa.weight", cur_h, F1, T);  // [T][576]
        half* cache = kcache[il];
        const int cslots = swa ? ring : slots_sparse;
        { PS _q(this, "mla_write_kv"); mla_write_kv(F1, (const float*)tw(pa + ".kv_a_layernorm.weight"), cache, cslots, T, p0, swa ? ring : 0, (float)theta(il), c.rms_eps); }
        { PS _q(this, "mla_prep_q"); mla_prep_q(F2, wk_b + wk_off[il], qa_h, T, NH, p0, (float)theta(il)); }
        bool use_list = false;
        if (!swa && index_on) {
            { PS _q(this, "idx_update"); idx_update(il, p0, T); }
            if (p0 + T > index_topk) {
                { PS _q(this, "idx_select"); idx_select(il, p0, T); }
                use_list = true;
            }
        }
        last_use_list = use_list;
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
        ensure_mla_ws(mla_workspace_bytes(a));
        { PS _q(this, "mla_attend"); mla_attend(qa_h, cache, (const float*)tw(pa + ".learnable_sink_param"), ctx_h, a, mla_ws, mla_ws_bytes); }
        { PS _q(this, "mla_vup"); mla_vup(ctx_h, wv_b + wv_off[il], F3, o_h, T, NH); }
        lin(pa + ".o_proj.weight", o_h, F1, T);  // [T][3072]
        { PS _q(this, "o_norm"); add_rms(F1, nullptr, (const float*)tw(pa + ".post_proj_norm.weight"), sub, HID, c.rms_eps, T); }
        { PS _q(this, "hc_post"); hc_post(x, sub, post, T); }
    }

    void mlp_layer(int il, int T) {
        const std::string pm = L(il) + ".mlp";
        const float* hfn = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_base");
        { PS _q(this, "hc_mix"); hc_mix_run(x, hfn, hsc, hbs, 8, T); }
        { PS _q(this, "hc_reduce_norm"); hc_reduce_norm(x, pre, (const float*)tw(L(il) + ".post_attention_layernorm.weight"), cur_h, cur_f, c.rms_eps, T); }
        if (il == 0) {
            lin(pm + ".gate_proj.weight", cur_h, F1, T);  // [T][13824]
            lin(pm + ".up_proj.weight", cur_h, F2, T);
            const size_t n = (size_t)T * 13824;
            { PS _q(this, "swiglu"); swiglu_h(F1, F2, H1, n); }
            lin(pm + ".down_proj.weight", H1, F3, T);  // [T][3072]
            { PS _q(this, "o_norm"); add_rms(F3, nullptr, (const float*)tw(pm + ".post_fc2_norm.weight"), sub, HID, c.rms_eps, T); }
        } else {
            const size_t nz = (size_t)T * 1536;
            { PS _q(this, "router"); linear_f32((const float*)tw(pm + ".gate.weight"), 512, HID, cur_f, F1, T); }
            { PS _q(this, "router_topk"); router_topk_sg(F1, (const float*)tw(pm + ".gate.e_score_correction_bias"), moe_idx, moe_w, 16, c.routed_scale, T); }
            lin(pm + ".routed_expert_down_proj.weight", cur_h, z_f, T);  // [T][1536] fp32
            d2h(h_z, z_f, nz * 4);
            d2h(h_idx, moe_idx, (size_t)T * 16 * 4);
            sycl::event evw = Q().memcpy(h_w, moe_w, (size_t)T * 16 * 4);
            // shared expert on the GPU while the CPU computes the routed experts
            lin(pm + ".shared_experts.gate_proj.weight", cur_h, F1, T);  // [T][1536]
            lin(pm + ".shared_experts.up_proj.weight", cur_h, F3, T);
            { PS _q(this, "swiglu"); swiglu_h(F1, F3, H1, nz); }
            lin(pm + ".shared_experts.down_proj.weight", H1, F3, T);  // [T][3072]
            evw.wait();
            double tc = now_s();
            if (!force_route_dir.empty()) {  // teacher-forced routing from an oracle dump (L<il>_topk_idx.npy / L<il>_topk_w.npy, float32 [T][16]): isolates numeric noise from routing flips
                std::vector<float> fi, fw;
                if (!load_npy_f32(force_route_dir + "/L" + std::to_string(il) + "_topk_idx.npy", fi) || !load_npy_f32(force_route_dir + "/L" + std::to_string(il) + "_topk_w.npy", fw) || fi.size() < (size_t)(cur_p0 + T) * 16) {
                    fprintf(stderr, "force-route: cannot load layer %d\n", il);
                    exit(5);
                }
                wait();
                for (size_t i = 0; i < (size_t)T * 16; ++i) { h_idx[i] = (int32_t)fi[(size_t)cur_p0 * 16 + i]; h_w[i] = fw[(size_t)cur_p0 * 16 + i]; }
            }
            const bool use_gx = gx != nullptr && T >= gx_min_T;
            if (use_gx) {
                // prefill: routed experts on the GPU (streamed from the page cache), result stays on the device
                gx->run(il - 1, T, z_f, h_idx, h_w, zout_f, nullptr);
            } else if (!std::getenv("HY4S_SKIP_EXPERTS")) {  // timing experiment: no routed experts at all (wrong output)
                cx.run(il - 1, T, h_z, h_idx, h_w, h_zout, expert_act);
            }
            if (dump && dbg_layers) {
                wait();
                const std::string nm = "L" + std::to_string(il) + "_";
                save_npy_f32(dump_dir + "/" + nm + "z_folded.npy", h_z, {T, 1536});
                std::vector<float> ti((size_t)T * 16);
                for (size_t i = 0; i < ti.size(); ++i) ti[i] = (float)h_idx[i];
                save_npy_f32(dump_dir + "/" + nm + "topk_idx.npy", ti.data(), {T, 16});
                save_npy_f32(dump_dir + "/" + nm + "topk_w.npy", h_w, {T, 16});
                save_npy_f32(dump_dir + "/" + nm + "z_out.npy", h_zout, {T, 1536});
                maybe_dump(nm + "shared_out", T, HID, F3);
            }
            t_cpu_exp += now_s() - tc;
            if (!use_gx) h2d(zout_f, h_zout, nz * 4);
            f2h(zout_f, z_h, nz);
            lin(pm + ".routed_expert_up_proj.weight", z_h, F2, T);  // [T][3072]
            { PS _q(this, "combine_norm"); add_rms(F2, F3, (const float*)tw(pm + ".combine_output_norm.weight"), sub, HID, c.rms_eps, T); }
        }
        { PS _q(this, "hc_post"); hc_post(x, sub, post, T); }
    }

    // process tokens [p0, p0+T) (T <= Tc) of the sequence tok_all (host, positions 0..p0+T-1). Returns false on error.
    bool forward_chunk(const int32_t* tok_all, int p0, int T, std::string& err) {
        cur_p0 = p0;
        h2d(d_ids, tok_all + p0, (size_t)T * 4);
        embed_hc((const uint8_t*)tw("model.embed_tokens.weight"), d_ids, x, T);
        const bool defer_eng = engram_ready && T <= 8;
        if (defer_eng) prefetch_engram(tok_all, p0, T);
        else if (engram_ready && !engram_prep(tok_all, p0, T, err)) return false;
        for (int il = 0; il < c.n_layers; ++il) {
            if (defer_eng && il == 1) {
                double tw0 = now_s();
                if (!engram_prep(tok_all, p0, T, err)) return false;
                t_engram_wait += now_s() - tw0;
            }
            int mo = -1;
            for (int j = 0; j < 4; ++j)
                if (emeta.layers[j] == il) mo = j;
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
        return true;
    }
    bool prof = false;
    double p_eng = 0, p_attn_swa = 0, p_attn_sp = 0, p_mlp0 = 0, p_moe = 0;
    void prof_report(const char* what) {
        if (!prof) return;
        fprintf(stderr, "[prof %s] engram %.3f s | attn swa(30) %.3f s sparse(12) %.3f s | mlp0 %.3f s | moe(41) %.3f s (cpu experts %.3f s)\n", what, p_eng, p_attn_swa, p_attn_sp, p_mlp0, p_moe, t_cpu_exp);
        p_eng = p_attn_swa = p_attn_sp = p_mlp0 = p_moe = 0;
    }

    // logits of chunk positions [first, T) into d logits [(T-first)][V]  (T-first <= logit_rows)
    void head(int T, int first) {
        const int n = T - first;
        const float* hfn = (const float*)tw("model.hc_head.hc_head_fn");
        const float* hsc = (const float*)tw("model.hc_head.hc_head_scale");
        const float* hbs = (const float*)tw("model.hc_head.hc_head_base");
        const float* xs = x + (size_t)first * 4 * HID;
        { PS _q(this, "head_hc_mix"); hc_mix_run(xs, hfn, hsc, hbs, 4, n); }
        { PS _q(this, "head_reduce_norm"); hc_reduce_norm(xs, pre, (const float*)tw("model.norm.weight"), hfin, nullptr, c.rms_eps, n); }
        const hy4::TInfo& lm = ti("lm_head.weight");
        linear_auto(lm.fmt, tw("lm_head.weight"), (int)lm.rows(), (int)lm.cols(), hfin, logits, n, g.is_rp("lm_head.weight"));
    }
};

}  // namespace hy4s
