// sy_dfly2.h - the DFly2 drafter of Hy4-Flash on the Arc runner's kernels (SYCL port of the HY4-STRIX hy4_dfly2.hip.h, branch f-hy4-dfly2 a70c0e8fc).
// Serving procedure of the owner (phaseA-tests/hy4/DFLY2-PROCEDURE-OWNER.md):
//   block  = [x_a, 120009 x K] at positions a..a+K, rows E(.) of the target's embed_tokens
//   slot 0 : H[0] += gate * eh_proj([enorm(E(x_a)) ; hnorm(h_{a-1})])             (raw gate, no sigmoid)
//   5 layers once over the B = K+1 rows; layer d reads the cached latents of target layer 25+4d (all rows < a, or its top-2048 picks at a-1) plus the block's own latents,
//     both through the draft layer's own kv_b_proj (absorbed form); intra-block causal; per-head sink in the denominator
//   final_norm, then for j = 1..K: hc_in = [hidden_norm(Hf[j]) ; embed_norm(E(prev))], h_corr = Hf[j] + down(silu(gate(hc_in)) * up(hc_in)), y_j = argmax lm_head(h_corr),
//     prev = y_j, conf_j = sigmoid(confidence_head(h_corr))
// The own block rows are written into the TARGET layer's cache slots a..a+K (free slots: the verify pass rewrites a..a+K' before it reads them).
#pragma once
#include <memory>

#include "sy_graph.h"

namespace hy4s {

constexpr int DFLY_MASK_ID = 120009;
constexpr int DFLY_KMAX = 7;
constexpr int DFLY_BMAX = DFLY_KMAX + 1;
constexpr int DFLY_LAYERS[5] = {25, 29, 33, 37, 41};

// ---- kernels
// rows of S8_K [V][3072] raw -> fp32 [T][3072]
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
// out[0:3072) = rms(a) * wa, out[3072:6144) = rms(b) * wb ; one group of 256 threads
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
inline void dfly_fuse(float* h0, const float* gate, const float* f) {  // h0[i] += gate[i] * f[i]
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) {
        for (int i = (int)it.get_local_id(0); i < HID; i += 256) h0[i] += gate[i] * f[i];
    });
}
// H[row] += rms(o[row]) * w
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
// h = Hf + D ; hc_h = half(h) ; conf = sigmoid(dot(cw, h) + cb)    (one group of 256)
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
// per-slot key lists: slot j sees the context list (n entries, ckeys[0..n)) followed by its own slots a..a+j
inline void dfly_lists(const int32_t* ckeys, const int32_t* cnk, int a, int32_t* dlist, int32_t* dnk, int lstride, int B) {
    Q().parallel_for(NR(B, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int j = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const int n = cnk[0];
        int32_t* l = dlist + (size_t)j * lstride;
        for (int i = tid; i < n; i += 256) l[i] = ckeys[i];
        for (int i = tid; i <= j; i += 256) l[n + i] = a + i;
        if (tid == 0) dnk[j] = n + j + 1;
    });
}
// argmax of each logits row [rows][V] -> out[rows]; one group of 1024 per row
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
// reduced-vocabulary chain: top-N ids of each logits row (threshold search on the order-preserving key, then emit every id with key >= thr until N are written)
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
// score[k] = lm_head row cand[k] . h   (lm_head S8_K in the RP layout, N rows x 12 super-blocks); one sub-group per candidate
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
    int rv = 0;  // reduced-vocabulary chain: candidates per slot (0 = full lm_head per slot)
    int32_t* d_cand = nullptr;
    float* d_cscore = nullptr;
    float *H = nullptr, *Hf = nullptr, *F1 = nullptr, *F2 = nullptr, *F3 = nullptr, *F4 = nullptr, *Ec = nullptr, *Dn = nullptr, *G = nullptr, *U = nullptr, *d_conf = nullptr, *hprev_f = nullptr;
    half *Hf_h = nullptr, *hn = nullptr, *qr_h = nullptr, *qa_h = nullptr, *ctx_h = nullptr, *o_h = nullptr, *mlp_h = nullptr, *hcin = nullptr, *act = nullptr, *hc_h = nullptr;
    half* hprev_h = nullptr;
    half *wkd[5] = {}, *wvd[5] = {};  // absorbed kv_b_proj of the 5 draft layers (fp16)
    int32_t *d_bids = nullptr, *d_chain = nullptr, *dlist = nullptr, *dnk = nullptr;
    int32_t *h_bids = nullptr, *h_chain = nullptr;  // pinned
    float* h_conf = nullptr;
    int lstride = 0;
    double t_draft = 0;
    long n_draft = 0;

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
        const std::string pf = "model.dfly_layers.";
        for (int d = 0; d < 5; ++d) {
            wkd[d] = al<half>((size_t)32 * 192 * 512);
            wvd[d] = al<half>((size_t)32 * 256 * 512);
            const std::string nm = pf + std::to_string(d) + ".self_attn.kv_b_proj.weight";
            if (!split_kvb(r.ti(nm).fmt, r.tw(nm), wkd[d], wvd[d], 32)) {
                err = "draft kv_b_proj format not supported";
                return false;
            }
        }
        wait();
        ok = true;
        return true;
    }

    // keep h_final row `row` of the last r.head() (fp16 [n][3072]) as the drafter's h_{a-1}
    void snap_h(int row) { d2d(hprev_h, r.hfin + (size_t)row * HID, HID * sizeof(half)); }

    // can the block at anchor position a be drafted (a context list is needed for every layer above index_topk)
    bool can_draft(int a, int K) const {
        if (!ok || K < 1 || K > DFLY_KMAX) return false;
        if (a + K + 1 > r.slots_sparse) return false;
        if (r.cap_p0 < 0) return false;
        const int row = a - 1 - r.cap_p0;
        if (row < 0 || row >= r.cap_rows) return false;
        return true;
    }

    // Draft K tokens. y[1..K] and conf[1..K] (host) after a queue sync. Returns false when the target state does not allow it.
    bool draft(int a, int xa, int K, int* y, float* conf) {
        if (!can_draft(a, K)) return false;
        const double t0 = now_s();
        const int B = K + 1;
        const int V = c.vocab;
        const int row = a - 1 - r.cap_p0;
        h_bids[0] = xa;
        for (int i = 1; i <= K; ++i) h_bids[i] = DFLY_MASK_ID;
        h2d(d_bids, h_bids, B * 4);
        h2d(d_chain, h_bids, 4);
        const std::string pf = "model.dfly_layers.";
        const uint8_t* emb = (const uint8_t*)r.tw("model.embed_tokens.weight");
        embed_rows_f32(emb, d_bids, H, B);
        // slot 0: mtp_fusion
        dfly_h2f(hprev_h, hprev_f, HID);
        dfly_cat2(H, wf(pf + "mtp_fusion.enorm.weight"), hprev_f, wf(pf + "mtp_fusion.hnorm.weight"), hcin, c.rms_eps);
        r.lin(pf + "mtp_fusion.eh_proj.weight", hcin, F1, 1);
        dfly_fuse(H, wf(pf + "mtp_fusion.gate"), F1);
        // 5 layers
        for (int d = 0; d < 5; ++d) {
            const int L = DFLY_LAYERS[d];
            const std::string pa = pf + std::to_string(d) + ".self_attn";
            const std::string pm = pf + std::to_string(d) + ".mlp";
            const bool use_list = r.cap_list[d];
            rms_rows(H, wf(pf + std::to_string(d) + ".input_layernorm.weight"), nullptr, hn, HID, c.rms_eps, B, 256);
            r.lin(pa + ".q_a_proj.weight", hn, F1, B);  // [B][1024]
            rms_rows(F1, wf(pa + ".q_a_layernorm.weight"), nullptr, qr_h, c.q_lora, c.rms_eps, B, 256);
            r.lin(pa + ".q_b_proj.weight", qr_h, F2, B);       // [B][8192]
            r.lin(pa + ".kv_a_proj_with_mqa.weight", hn, F3, B);  // [B][576]
            mla_write_kv(F3, wf(pa + ".kv_a_layernorm.weight"), r.kcache[L], r.slots_sparse, B, a, 0, (float)c.theta_sparse, c.rms_eps);
            mla_prep_q(F2, wkd[d], qa_h, B, 32, a, (float)c.theta_sparse);
            r.lin(pa + ".linear_gate.weight", hn, F4, B);  // [B][8192]
            MlaAttnArgs ma;
            ma.M = B;
            ma.NH = 32;
            ma.p0 = a;
            ma.cache_slots = r.slots_sparse;
            if (use_list) {
                dfly_lists(r.dkeys[d] + (size_t)row * r.index_topk, r.dnkc[d] + row, a, dlist, dnk, lstride, B);
                ma.mode = MLA_LIST;
                ma.keys = dlist;
                ma.nkeys = dnk;
                ma.kmax = lstride;
            } else {
                ma.mode = MLA_CAUSAL;
            }
            r.ensure_mla_ws(mla_workspace_bytes(ma));
            mla_attend(qa_h, r.kcache[L], wf(pa + ".learnable_sink_param"), ctx_h, ma, r.mla_ws, r.mla_ws_bytes);
            mla_vup(ctx_h, wvd[d], F4, o_h, B, 32);
            r.lin(pa + ".o_proj.weight", o_h, F1, B);  // [B][3072]
            dfly_rms_resid(F1, wf(pa + ".post_proj_norm.weight"), H, c.rms_eps, B);
            // dense SwiGLU
            rms_rows(H, wf(pf + std::to_string(d) + ".post_attention_layernorm.weight"), nullptr, hn, HID, c.rms_eps, B, 256);
            r.lin(pm + ".gate_proj.weight", hn, F1, B);  // [B][9216]
            r.lin(pm + ".up_proj.weight", hn, F2, B);
            const size_t n9 = (size_t)B * 9216;
            swiglu_h(F1, F2, mlp_h, n9);
            r.lin(pm + ".down_proj.weight", mlp_h, F3, B);  // [B][3072]
            add_inplace(H, F3, (size_t)B * HID);
        }
        rms_rows(H, wf(pf + "final_norm.weight"), Hf, Hf_h, HID, c.rms_eps, B, 256);
        // hidden_correction + lm_head, unrolled
        const hy4::TInfo& lm = r.ti("lm_head.weight");
        const void* lmw = r.tw("lm_head.weight");
        const bool lm_rp = r.g.is_rp("lm_head.weight");
        const int N = std::min(rv, 1024);
        if (N > 0) {
            // base logits of slots 1..K in ONE lm_head pass, candidate ids per slot
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
