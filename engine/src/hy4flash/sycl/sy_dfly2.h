// sy_dfly2.h - the DFly2 drafter of Hy4-Flash on the Arc runner (SYCL):
//   * Quantized S_K kv_b_proj directly (0 persistent fp16 wkd/wvd -> saves 73 MB VRAM + 3x less bandwidth)
//   * Full 5-layer + hidden_correction + lm_head chain captured in ONE SYCL command_graph (~160 launches -> 1 replay!)
#pragma once
#include <memory>
#include <vector>

#include "sy_graph.h"

namespace hy4s {

constexpr int DFLY_MASK_ID = 120009;
constexpr int DFLY_KMAX = 7;
constexpr int DFLY_BMAX = DFLY_KMAX + 1;
constexpr int DFLY_LAYERS[5] = {25, 29, 33, 37, 41};

// ---- kernels
inline void embed_rows_f32(const uint8_t* W, const int32_t* ids, float* out, int T) {
    Q().parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int t = (int)it.get_group(0);
        const uint8_t* row = W + (size_t)ids[t] * 12 * 262;
        for (int d = (int)it.get_local_id(0); d < HID; d += 256) {
            const uint8_t* blk = row + (d >> 8) * 262;
            const float D = h2f_bits(ld16(blk));
            const float eff = D * ((float)blk[2 + ((d & 255) >> 6)] / 255.0f);
            out[(size_t)t * HID + d] = (float)(int8_t)blk[6 + (d & 255)] * eff;
        }
    });
}
inline void dfly_cat2(const float* a, const float* wa, const float* b, const float* wb, half* out, float eps) {
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const int tid = (int)it.get_local_id(0);
        float sa = 0, sb = 0;
        for (int i = tid; i < HID; i += 256) {
            sa += a[i] * a[i];
            sb += b[i] * b[i];
        }
        sa = wg_sum(it, sa, sh);
        sb = wg_sum(it, sb, sh);
        const float ia = sycl::rsqrt(sa / HID + eps), ib = sycl::rsqrt(sb / HID + eps);
        for (int i = tid; i < HID; i += 256) {
            out[i] = (half)(a[i] * ia * wa[i]);
            out[HID + i] = (half)(b[i] * ib * wb[i]);
        }
    });
}
inline void dfly_fuse(float* h0, const float* gate, const float* f) {
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) {
        for (int i = (int)it.get_local_id(0); i < HID; i += 256) h0[i] += gate[i] * f[i];
    });
}
inline void dfly_rms_resid(const float* o, const float* w, float* H, float eps, int rows) {
    Q().parallel_for(NR(rows, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const size_t r = it.get_group(0);
        const int tid = (int)it.get_local_id(0);
        const float* x = o + r * HID;
        float ss = 0;
        for (int i = tid; i < HID; i += 256) ss += x[i] * x[i];
        ss = wg_sum(it, ss, sh);
        const float inv = sycl::rsqrt(ss / HID + eps);
        float* h = H + r * HID;
        for (int i = tid; i < HID; i += 256) h[i] += x[i] * inv * w[i];
    });
}
inline void dfly_corr(const float* Hf, const float* D, const float* cw, const float* cb, half* hc_h, float* conf) {
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const int tid = (int)it.get_local_id(0);
        float dot = 0;
        for (int i = tid; i < HID; i += 256) {
            const float v = Hf[i] + D[i];
            hc_h[i] = (half)v;
            dot += v * cw[i];
        }
        dot = wg_sum(it, dot, sh);
        if (tid == 0) *conf = 1.0f / (1.0f + sycl::exp(-(dot + cb[0])));
    });
}
inline void dfly_h2f(const half* in, float* out, int n) {
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) {
        for (int i = (int)it.get_local_id(0); i < n; i += 256) out[i] = (float)in[i];
    });
}
// per-slot key lists (supports device-resident StepParams* d_step for command_graph replay!)
inline void dfly_lists_impl(const int32_t* dkeys_base, const int32_t* dnkc_base, int index_topk, int row_host, int a_host,
                            const StepParams* d_step, int32_t* dlist, int32_t* dnk, int lstride, int B) {
    Q().parallel_for(NR(B, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int row = d_step ? d_step->cap_row : row_host;
        const int a = d_step ? d_step->a : a_host;
        const int32_t* ckeys = dkeys_base + (size_t)row * index_topk;
        const int n = dnkc_base[row];
        const int j = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        int32_t* l = dlist + (size_t)j * lstride;
        for (int i = tid; i < n; i += 256) l[i] = ckeys[i];
        for (int i = tid; i <= j; i += 256) l[n + i] = a + i;
        if (tid == 0) dnk[j] = n + j + 1;
    });
}
inline void argmax_rows(const float* logits, int V, int32_t* out, int rows) {
    Q().parallel_for(NR(rows, 1024), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sv = lmem<float, 32>(it);
        int* si = lmem<int, 32>(it);
        const int tid = (int)it.get_local_id(0);
        const float* lg = logits + (size_t)it.get_group(0) * V;
        float bv = -3.0e38f;
        int bi = 0;
        for (int i = tid; i < V; i += 1024)
            if (lg[i] > bv) { bv = lg[i]; bi = i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = shfl_xor(it, bv, o);
            const int oi = shfl_xor(it, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
        sycl::group_barrier(it.get_group());
        if (tid == 0) {
            float b2 = sv[0];
            int i2 = si[0];
            for (int q = 1; q < 32; ++q)
                if (sv[q] > b2 || (sv[q] == b2 && si[q] < i2)) { b2 = sv[q]; i2 = si[q]; }
            out[it.get_group(0)] = i2;
        }
    });
}
inline void dfly_topn(const float* logits, int V, int N, int32_t* cand, int rows) {
    Q().parallel_for(NR(rows, 1024), [=](sycl::nd_item<1> it) REQD_SG32 {
        int* red = lmem<int, 32>(it);
        int* cnt = lmem<int, 1>(it);
        uint32_t* thr_s = lmem<uint32_t, 1>(it);
        const int tid = (int)it.get_local_id(0);
        const float* lg = logits + (size_t)it.get_group(0) * V;
        int32_t* out = cand + (size_t)it.get_group(0) * N;
        uint32_t lo = 0, hi = 0xFFFFFFFFu;
        for (int itr = 0; itr < 33 && lo < hi; ++itr) {
            const uint32_t mid = (uint32_t)(lo + (((uint64_t)hi - lo + 1) >> 1));
            int c = 0;
            for (int i = tid; i < V; i += 1024) c += idx::f2key(lg[i]) >= mid ? 1 : 0;
            c = sycl::reduce_over_group(it.get_sub_group(), c, sycl::plus<int>());
            sycl::group_barrier(it.get_group());
            if ((tid & 31) == 0) red[tid >> 5] = c;
            sycl::group_barrier(it.get_group());
            int tot = 0;
            for (int w = 0; w < 32; ++w) tot += red[w];
            if (tot >= N) lo = mid; else hi = mid - 1;
            sycl::group_barrier(it.get_group());
        }
        if (tid == 0) { *thr_s = lo; *cnt = 0; }
        sycl::group_barrier(it.get_group());
        const uint32_t thr = *thr_s;
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> ctr(*cnt);
        for (int i = tid; i < V; i += 1024)
            if (idx::f2key(lg[i]) >= thr) {
                const int s = ctr.fetch_add(1);
                if (s < N) out[s] = i;
            }
    });
}
inline void dfly_cand_score_rp(const uint8_t* W, int Nrows, const int32_t* cand, const half* h, float* score, int ncand) {
    const size_t pay_off = rp_pay_off(Nrows, 12);
    Q().parallel_for(NR(ncand, 32), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int k = (int)it.get_group(0), lane = (int)it.get_local_id(0);
        const size_t row = (size_t)cand[k];
        float acc = 0;
        for (int sb = 0; sb < 12; ++sb) {
            const uint8_t* hd = W + (row * 12 + sb) * 8;
            const uint8_t* pl = W + pay_off + (row * 12 + sb) * 256;
            const float D = h2f_bits(ld16(hd));
            for (int i = lane; i < 256; i += 32) {
                const float eff = D * ((float)hd[2 + (i >> 6)] / 255.0f);
                acc += (float)(int8_t)pl[i] * eff * (float)h[sb * 256 + i];
            }
        }
        acc = sg_sum(it, acc);
        if (lane == 0) score[k] = acc;
    });
}
inline void dfly_cand_argmax(const float* score, const int32_t* cand, int N, int32_t* out) {
    Q().parallel_for(NR(1, 1024), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sv = lmem<float, 32>(it);
        int* si = lmem<int, 32>(it);
        const int tid = (int)it.get_local_id(0);
        float bv = -3.0e38f;
        int bi = 0x7fffffff;
        for (int i = tid; i < N; i += 1024) {
            const int id = cand[i];
            if (score[i] > bv || (score[i] == bv && id < bi)) { bv = score[i]; bi = id; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = shfl_xor(it, bv, o);
            const int oi = shfl_xor(it, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
        sycl::group_barrier(it.get_group());
        if (tid == 0) {
            float b2 = sv[0];
            int i2 = si[0];
            for (int q = 1; q < 32; ++q)
                if (sv[q] > b2 || (sv[q] == b2 && si[q] < i2)) { b2 = sv[q]; i2 = si[q]; }
            *out = i2;
        }
    });
}
inline void add_inplace(float* a, const float* b, size_t n) {
    Q().parallel_for(NR((n + 255) / 256, 256), [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i < n) a[i] += b[i];
    });
}

struct DFly2 {
    Runner& r;
    const hy4::Cfg& c;
    bool ok = false;
    int rv = 0;
    int32_t* d_cand = nullptr;
    float* d_cscore = nullptr;
    float *H = nullptr, *Hf = nullptr, *F1 = nullptr, *F2 = nullptr, *F3 = nullptr, *F4 = nullptr, *Ec = nullptr, *Dn = nullptr, *G = nullptr, *U = nullptr, *d_conf = nullptr, *hprev_f = nullptr;
    half *Hf_h = nullptr, *hn = nullptr, *qr_h = nullptr, *qa_h = nullptr, *ctx_h = nullptr, *o_h = nullptr, *mlp_h = nullptr, *hcin = nullptr, *act = nullptr, *hc_h = nullptr;
    half* hprev_h = nullptr;
    int32_t *d_bids = nullptr, *d_chain = nullptr, *dlist = nullptr, *dnk = nullptr;
    int32_t *h_bids = nullptr, *h_chain = nullptr;
    float* h_conf = nullptr;
    int lstride = 0;
    double t_draft = 0;
    long n_draft = 0;

#if HY4S_HAS_GRAPH
    struct DraftGraphKey {
        int K = 0, rv = 0, list_mask = 0;
        int TK[5] = {}, S[5] = {}, tps[5] = {};
        bool operator==(const DraftGraphKey& o) const {
            if (K != o.K || rv != o.rv || list_mask != o.list_mask) return false;
            for (int d = 0; d < 5; ++d)
                if (TK[d] != o.TK[d] || S[d] != o.S[d] || tps[d] != o.tps[d]) return false;
            return true;
        }
    };
    using ExecGraph = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
    struct DraftGraphEntry {
        DraftGraphKey key;
        std::unique_ptr<ExecGraph> exec;
    };
    std::vector<DraftGraphEntry> dgraphs_;
#endif

    explicit DFly2(Runner& rr) : r(rr), c(rr.c) {}

    template <class T> T* al(size_t n) { return dalloc<T>(n); }
    const float* wf(const std::string& n) const { return (const float*)r.tw(n); }

    bool init(std::string& err) {
        if (!r.g.man.T.count("model.dfly_layers.0.self_attn.q_a_proj.weight")) {
            err = "this .strata has no DFly2 drafter tensors";
            return false;
        }
        r.dfly_capture_alloc();
        const size_t B = DFLY_BMAX;
        H = al<float>(B * HID);
        Hf = al<float>(B * HID);
        F1 = al<float>(B * 16384);
        F2 = al<float>(B * 16384);
        F3 = al<float>(B * 16384);
        F4 = al<float>(B * 16384);
        Ec = al<float>(HID);
        Dn = al<float>(HID);
        G = al<float>(HID);
        U = al<float>(HID);
        d_conf = al<float>(16);
        hprev_f = al<float>(HID);
        Hf_h = al<half>(B * HID);
        hn = al<half>(B * HID);
        qr_h = al<half>(B * 1024);
        qa_h = al<half>(B * 32 * 576);
        ctx_h = al<half>(B * 32 * 512);
        o_h = al<half>(B * 8192);
        mlp_h = al<half>(B * 9216);
        hcin = al<half>(2 * HID);
        act = al<half>(HID);
        hc_h = al<half>(HID);
        hprev_h = al<half>(HID);
        d_bids = al<int32_t>(16);
        d_chain = al<int32_t>(16);
        lstride = r.index_topk + 8;
        dlist = al<int32_t>((size_t)B * lstride);
        dnk = al<int32_t>(16);
        d_cand = al<int32_t>((size_t)B * 1024);
        d_cscore = al<float>(1024);
        h_bids = halloc<int32_t>(16);
        h_chain = halloc<int32_t>(16);
        h_conf = halloc<float>(16);
        // Quantized kv_b_proj used directly on the fly -> saves 73 MB VRAM (no wkd/wvd)!
        wait();
        ok = true;
        return true;
    }

    void snap_h(int row) { d2d(hprev_h, r.hfin + (size_t)row * HID, HID * sizeof(half)); }

    bool can_draft(int a, int K) const {
        if (!ok || K < 1 || K > DFLY_KMAX) return false;
        if (a + K + 1 > r.slots_sparse) return false;
        if (r.cap_p0 < 0) return false;
        const int row = a - 1 - r.cap_p0;
        if (row < 0 || row >= r.cap_rows) return false;
        return true;
    }

    void draft_body(int a, int row, int K, const StepParams* dp, const int* g_TK, const int* g_S, const int* g_tps) {
        const int B = K + 1;
        const int V = c.vocab;
        const std::string pf = "model.dfly_layers.";
        const uint8_t* emb = (const uint8_t*)r.tw("model.embed_tokens.weight");
        embed_rows_f32(emb, d_bids, H, B);
        dfly_h2f(hprev_h, hprev_f, HID);
        dfly_cat2(H, wf(pf + "mtp_fusion.enorm.weight"), hprev_f, wf(pf + "mtp_fusion.hnorm.weight"), hcin, c.rms_eps);
        r.lin(pf + "mtp_fusion.eh_proj.weight", hcin, F1, 1);
        dfly_fuse(H, wf(pf + "mtp_fusion.gate"), F1);
        for (int d = 0; d < 5; ++d) {
            const int L = DFLY_LAYERS[d];
            const std::string pa = pf + std::to_string(d) + ".self_attn";
            const std::string pm = pf + std::to_string(d) + ".mlp";
            const bool use_list = r.cap_list[d];
            rms_rows(H, wf(pf + std::to_string(d) + ".input_layernorm.weight"), nullptr, hn, HID, c.rms_eps, B, 256);
            r.lin(pa + ".q_a_proj.weight", hn, F1, B);
            rms_rows(F1, wf(pa + ".q_a_layernorm.weight"), nullptr, qr_h, c.q_lora, c.rms_eps, B, 256);
            r.lin(pa + ".q_b_proj.weight", qr_h, F2, B);
            r.lin(pa + ".kv_a_proj_with_mqa.weight", hn, F3, B);
            if (dp) mla_write_kv_dev(F3, wf(pa + ".kv_a_layernorm.weight"), r.kcache[L], r.slots_sparse, B, dp, true, 0, (float)c.theta_sparse, c.rms_eps);
            else mla_write_kv(F3, wf(pa + ".kv_a_layernorm.weight"), r.kcache[L], r.slots_sparse, B, a, 0, (float)c.theta_sparse, c.rms_eps);
            const std::string kvb_nm = pa + ".kv_b_proj.weight";
            const int kvb_fmt = r.ti(kvb_nm).fmt;
            const void* kvb_w = r.tw(kvb_nm);
            mla_prep_q_sk(kvb_fmt, kvb_w, F2, qa_h, B, 32, a, dp, true, (float)c.theta_sparse);
            r.lin(pa + ".linear_gate.weight", hn, F4, B);
            MlaAttnArgs ma;
            ma.M = B; ma.NH = 32; ma.p0 = a; ma.cache_slots = r.slots_sparse;
            if (use_list) {
                dfly_lists_impl(r.dkeys[d], r.dnkc[d], r.index_topk, row, a, dp, dlist, dnk, lstride, B);
                ma.mode = MLA_LIST; ma.keys = dlist; ma.nkeys = dnk; ma.kmax = lstride;
            } else {
                ma.mode = MLA_CAUSAL;
            }
            if (dp) {
                mla_attend_dev(qa_h, r.kcache[L], wf(pa + ".learnable_sink_param"), ctx_h, ma, dp, true, g_TK[d], g_S[d], g_tps[d], r.mla_ws, r.mla_ws_bytes);
            } else {
                r.ensure_mla_ws(mla_workspace_bytes(ma));
                mla_attend(qa_h, r.kcache[L], wf(pa + ".learnable_sink_param"), ctx_h, ma, r.mla_ws, r.mla_ws_bytes);
            }
            mla_vup_sk(kvb_fmt, kvb_w, ctx_h, F4, o_h, B, 32);
            r.lin(pa + ".o_proj.weight", o_h, F1, B);
            dfly_rms_resid(F1, wf(pa + ".post_proj_norm.weight"), H, c.rms_eps, B);
            rms_rows(H, wf(pf + std::to_string(d) + ".post_attention_layernorm.weight"), nullptr, hn, HID, c.rms_eps, B, 256);
            r.lin(pm + ".gate_proj.weight", hn, F1, B);
            r.lin(pm + ".up_proj.weight", hn, F2, B);
            swiglu_h(F1, F2, mlp_h, (size_t)B * 9216);
            r.lin(pm + ".down_proj.weight", mlp_h, F3, B);
            add_inplace(H, F3, (size_t)B * HID);
        }
        rms_rows(H, wf(pf + "final_norm.weight"), Hf, Hf_h, HID, c.rms_eps, B, 256);
        const hy4::TInfo& lm = r.ti("lm_head.weight");
        const void* lmw = r.tw("lm_head.weight");
        const bool lm_rp = r.g.is_rp("lm_head.weight");
        const int N = std::min(rv, 1024);
        if (N > 0) {
            linear_auto(lm.fmt, lmw, (int)lm.rows(), (int)lm.cols(), Hf_h + HID, r.logits, K, lm_rp);
            dfly_topn(r.logits, V, N, d_cand, K);
        }
        for (int j = 1; j <= K; ++j) {
            embed_rows_f32(emb, d_chain + (j - 1), Ec, 1);
            dfly_cat2(Hf + (size_t)j * HID, wf(pf + "hidden_correction.hidden_norm.weight"), Ec, wf(pf + "hidden_correction.embed_norm.weight"), hcin, c.rms_eps);
            r.lin(pf + "hidden_correction.gate_proj.weight", hcin, G, 1);
            r.lin(pf + "hidden_correction.up_proj.weight", hcin, U, 1);
            swiglu_h(G, U, act, (size_t)HID);
            r.lin(pf + "hidden_correction.down_proj.weight", act, Dn, 1);
            dfly_corr(Hf + (size_t)j * HID, Dn, wf(pf + "confidence_head.proj.weight"), wf(pf + "confidence_head.proj.bias"), hc_h, d_conf + j);
            if (N > 0) {
                dfly_cand_score_rp((const uint8_t*)lmw, (int)lm.rows(), d_cand + (size_t)(j - 1) * N, hc_h, d_cscore, N);
                dfly_cand_argmax(d_cscore, d_cand + (size_t)(j - 1) * N, N, d_chain + j);
            } else {
                linear_auto(lm.fmt, lmw, (int)lm.rows(), (int)lm.cols(), hc_h, r.logits, 1, lm_rp);
                argmax_row(r.logits, V, d_chain + j);
            }
        }
    }

    bool draft(int a, int xa, int K, int* y, float* conf) {
        if (!can_draft(a, K)) return false;
        const double t0 = now_s();
        const int B = K + 1;
        const int row = a - 1 - r.cap_p0;
        h_bids[0] = xa;
        for (int i = 1; i <= K; ++i) h_bids[i] = DFLY_MASK_ID;
        h2d(d_bids, h_bids, B * 4);
        h2d(d_chain, h_bids, 4);

        bool ran_graph = false;
#if HY4S_HAS_GRAPH
        if (r.graph_enabled && !r.prof2 && !r.dump && !r.tap) {
            DraftGraphKey dk;
            dk.K = K;
            dk.rv = std::min(rv, 1024);
            dk.list_mask = 0;
            size_t max_ws = 0;
            for (int d = 0; d < 5; ++d) {
                if (r.cap_list[d]) dk.list_mask |= (1 << d);
                MlaAttnArgs ma;
                ma.M = B; ma.NH = 32; ma.p0 = a; ma.cache_slots = r.slots_sparse;
                if (r.cap_list[d]) { ma.mode = MLA_LIST; ma.kmax = lstride; }
                else ma.mode = MLA_CAUSAL;
                mla_choose_splits(ma, dk.TK[d], dk.S[d], dk.tps[d]);
                max_ws = std::max(max_ws, mla_workspace_bytes(ma));
            }
            r.ensure_mla_ws(max_ws);

            ExecGraph* eg = nullptr;
            for (auto& e : dgraphs_)
                if (e.key == dk) { eg = e.exec.get(); break; }
            if (!eg) {
                if (dgraphs_.size() >= 8) { wait(); dgraphs_.clear(); }
                try {
                    wait();
                    sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable> cg(Q().get_context(), Q().get_device());
                    cg.begin_recording(Q());
                    draft_body(a, row, K, r.d_step, dk.TK, dk.S, dk.tps);
                    cg.end_recording(Q());
                    dgraphs_.push_back(DraftGraphEntry{dk, std::make_unique<ExecGraph>(cg.finalize())});
                    eg = dgraphs_.back().exec.get();
                } catch (const sycl::exception& e) {
                    fprintf(stderr, "[dfly2-graph] capture failed (%s), eager fallback\n", e.what());
                    r.graph_enabled = false;
                }
            }
            if (eg) {
                r.h_step->a = a;
                r.h_step->cap_row = row;
                h2d(r.d_step, r.h_step, sizeof(StepParams));
                Q().ext_oneapi_graph(*eg);
                ran_graph = true;
            }
        }
#endif
        if (!ran_graph) draft_body(a, row, K, nullptr, nullptr, nullptr, nullptr);

        d2h(h_chain, d_chain, B * 4);
        d2h(h_conf, d_conf, B * 4);
        wait();
        for (int j = 1; j <= K; ++j) {
            y[j] = h_chain[j];
            conf[j] = h_conf[j];
        }
        t_draft += now_s() - t0;
        ++n_draft;
        return true;
    }
};

}  // namespace hy4s
