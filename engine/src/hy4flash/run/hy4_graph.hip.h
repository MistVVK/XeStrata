// hy4_graph.hip.h - the 42-layer hy_v4_flash forward on gfx1151 (HIP): chunked prefill + decode, KV fp16, absorbed MLA, LatentMoE.
// Built from: skgemm (linear_h / linear_f32), skmoe (moe_experts), mla_attn (mla_*), hy4_kernels (norms, iHC, router, engram), hy4_engram (hash + page reads).
#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "hy4_engram.h"
#include "hy4_gpu_model.hip.h"
#include "hy4_kernels.hip.h"
#include "mla_attn.hip.h"
#include "skgemm.hip.h"
#include "skmoe.hip.h"

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
    __half* wk_b = nullptr;  // per layer offsets
    __half* wv_b = nullptr;
    std::vector<size_t> wk_off, wv_off;
    std::vector<__half*> kcache;  // per layer
    int ring = 0;
    // host side engram
    EngramMeta emeta;
    EngramIO eio;
    std::vector<uint8_t> h_eraw, h_evalid;
    bool engram_ready = false;
    // stats
    double t_engram_cpu = 0;
    bool dump = false;
    std::string dump_dir = ".";
    bool dbg_layers = false;
    int index_on = 0;  // IndexPool (not yet implemented): sparse layers attend all keys

    Runner(GpuModel& gm, int max_ctx_, int max_chunk) : g(gm), c(gm.man.c), max_ctx(max_ctx_), Tc(max_chunk) {}

    template <class T> T* dalloc(size_t n) {
        T* p = nullptr;
        HCK(hipMalloc((void**)&p, n * sizeof(T)));
        return p;
    }
    const void* tw(const std::string& name) const { return g.tensor_dev(name); }
    const TInfo& ti(const std::string& name) const { return g.man.get(name); }
    std::string L(int il) const { return "model.layers." + std::to_string(il); }
    int heads(int il) const { return c.is_swa[il] ? c.heads_swa : c.heads_sparse; }
    double theta(int il) const { return c.is_swa[il] ? c.theta_swa : c.theta_sparse; }

    void lin(const std::string& name, const __half* X, float* Y, int M) {
        const TInfo& t = ti(name);
        linear_h(st, t.fmt, tw(name), (int)t.rows(), (int)t.cols(), X, Y, M);
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
        logits = dalloc<float>((size_t)T * c.vocab);
        d_argmax = dalloc<int32_t>(T);
        mla_ws_bytes = mla_workspace_bytes(T, 64, max_ctx);
        mla_ws = dalloc<uint8_t>(mla_ws_bytes ? mla_ws_bytes : 256);
        // experts table
        {
            std::vector<ExpertDev> h(41 * 512);
            for (int i = 0; i < 41 * 512; ++i) {
                const ExpertRef& e = g.experts[i];
                h[i].body_off = e.body_off;
                h[i].dn_off = e.dn_off;
                h[i].gu_code = e.gu_code;
                h[i].dn_code = e.dn_code;
                h[i].pad = 0;
            }
            d_experts = dalloc<ExpertDev>(41 * 512);
            HCK(hipMemcpy(d_experts, h.data(), h.size() * sizeof(ExpertDev), hipMemcpyHostToDevice));
        }
        // kv caches
        ring = 512 + Tc;
        kcache.resize(c.n_layers);
        size_t kvbytes = 0;
        for (int il = 0; il < c.n_layers; ++il) {
            size_t rows = c.is_swa[il] ? (size_t)ring : (size_t)max_ctx;
            kcache[il] = dalloc<__half>(rows * 576);
            kvbytes += rows * 576 * 2;
        }
        // absorbed weights
        wk_off.resize(c.n_layers);
        wv_off.resize(c.n_layers);
        size_t tk = 0, tv = 0;
        for (int il = 0; il < c.n_layers; ++il) {
            int NH = heads(il);
            wk_off[il] = tk;
            wv_off[il] = tv;
            tk += (size_t)NH * 192 * 512;
            tv += (size_t)NH * 256 * 512;
        }
        wk_b = dalloc<__half>(tk);
        wv_b = dalloc<__half>(tv);
        for (int il = 0; il < c.n_layers; ++il) {
            int NH = heads(il);
            const TInfo& t = ti(L(il) + ".self_attn.kv_b_proj.weight");
            dim3 grid(NH * 448), block(128);
            switch (t.fmt) {
                case FMT_S4: hipLaunchKernelGGL((k_split_kvb<FMT_S4>), grid, block, 0, st, (const uint8_t*)tw(L(il) + ".self_attn.kv_b_proj.weight"), wk_b + wk_off[il], wv_b + wv_off[il], NH); break;
                case FMT_S5: hipLaunchKernelGGL((k_split_kvb<FMT_S5>), grid, block, 0, st, (const uint8_t*)tw(L(il) + ".self_attn.kv_b_proj.weight"), wk_b + wk_off[il], wv_b + wv_off[il], NH); break;
                case FMT_S6: hipLaunchKernelGGL((k_split_kvb<FMT_S6>), grid, block, 0, st, (const uint8_t*)tw(L(il) + ".self_attn.kv_b_proj.weight"), wk_b + wk_off[il], wv_b + wv_off[il], NH); break;
                case FMT_S8: hipLaunchKernelGGL((k_split_kvb<FMT_S8>), grid, block, 0, st, (const uint8_t*)tw(L(il) + ".self_attn.kv_b_proj.weight"), wk_b + wk_off[il], wv_b + wv_off[il], NH); break;
                default: err = "kv_b_proj format not supported"; return false;
            }
        }
        HCK(hipStreamSynchronize(st));
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
        h_eraw.resize((size_t)4 * Tc * 16 * 100);
        h_evalid.resize((size_t)4 * Tc * 16);
        size_t fr = 0, tot = 0;
        HCK(hipMemGetInfo(&fr, &tot));
        fprintf(stderr, "[runner] kv cache %.2f GB (ctx %d, chunk %d), device free %.1f GiB\n", kvbytes / 1e9, max_ctx, Tc, fr / 1073741824.0);
        return true;
    }

    // Engram rows of one chunk for all 4 modules (CPU hash + page reads), uploaded to the device
    bool engram_prep(const int32_t* tok_all, int p0, int T, std::string& err) {
        double t0 = now_s();
        for (int mo = 0; mo < 4; ++mo) {
            std::vector<uint64_t> ids((size_t)T * 16);
            uint8_t* valid = h_evalid.data() + (size_t)mo * Tc * 16;
            engram_hash(emeta, mo, tok_all, p0, p0 + T, ids.data(), valid);
            uint8_t* raw = h_eraw.data() + (size_t)mo * Tc * 16 * 100;
            if (!eio.fetch_raw(ids.data(), ids.size(), raw, err)) return false;
            HCK(hipMemcpyAsync(d_eraw[mo], raw, (size_t)T * 16 * 100, hipMemcpyHostToDevice, st));
            HCK(hipMemcpyAsync(d_evalid[mo], valid, (size_t)T * 16, hipMemcpyHostToDevice, st));
        }
        HCK(hipStreamSynchronize(st));
        t_engram_cpu += now_s() - t0;
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
        for (int j = 0; j < 5; ++j) {
            std::string nm = j < 4 ? p + ".key_projs." + std::to_string(j) + ".weight" : p + ".value_proj.weight";
            const TInfo& t = ti(nm);
            linear_h(st, t.fmt, tw(nm), (int)t.rows(), (int)t.cols(), er_h, kv_f + (size_t)j * T * HID, T);
        }
        hipLaunchKernelGGL(k_engram_apply, dim3(T, 4), dim3(256), 0, st, x, kv_f, e.kvb, e.egate, e.pre_gamma, e.hcs, d_evalid[mo], T);
    }

    void attention_layer(int il, int p0, int T) {
        const std::string pa = L(il) + ".self_attn";
        const int NH = heads(il);
        const bool swa = c.is_swa[il];
        // iHC pre + input_layernorm
        const float* hfn = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_attn_layer.hc_pre.hc_base");
        hipLaunchKernelGGL(k_hc_mix, dim3(T), dim3(256), 0, st, x, hfn, hsc, hbs, pre, post, 8, c.rms_eps, c.hc_eps, c.hc_mag);
        hipLaunchKernelGGL(k_hc_reduce_norm, dim3(T), dim3(256), 0, st, x, pre, (const float*)tw(L(il) + ".input_layernorm.weight"), cur_h, (float*)nullptr, c.rms_eps);
        // q path
        lin(pa + ".q_a_proj.weight", cur_h, F1, T);  // [T][1024]
        hipLaunchKernelGGL(k_rms_rows, dim3(T), dim3(256), 0, st, F1, (const float*)tw(pa + ".q_a_layernorm.weight"), (float*)nullptr, qr_h, c.q_lora, c.rms_eps);
        lin(pa + ".q_b_proj.weight", qr_h, F2, T);  // [T][NH*256]
        lin(pa + ".kv_a_proj_with_mqa.weight", cur_h, F1, T);  // [T][576]  (qr_h already extracted)
        __half* cache = kcache[il];
        mla_write_kv(st, F1, (const float*)tw(pa + ".kv_a_layernorm.weight"), cache, T, p0, swa ? ring : 0, (float)theta(il), c.rms_eps);
        mla_prep_q(st, F2, wk_b + wk_off[il], qa_h, T, NH, p0, (float)theta(il));
        lin(pa + ".linear_gate.weight", cur_h, F3, T);  // [T][NH*256]
        MlaAttnArgs a;
        a.M = T;
        a.NH = NH;
        a.p0 = p0;
        a.mode = swa ? MLA_WINDOW : MLA_CAUSAL;
        a.window = c.window;
        a.ring = swa ? ring : 0;
        mla_attention(st, qa_h, cache, (const float*)tw(pa + ".learnable_sink_param"), wv_b + wv_off[il], F3, o_h, ctx_ws, a, mla_ws, mla_ws_bytes);
        lin(pa + ".o_proj.weight", o_h, F1, T);  // [T][3072]
        hipLaunchKernelGGL(k_add_rms, dim3(T), dim3(256), 0, st, F1, (const float*)nullptr, (const float*)tw(pa + ".post_proj_norm.weight"), sub, HID, c.rms_eps);
        hipLaunchKernelGGL(k_hc_post, dim3(T), dim3(256), 0, st, x, sub, post);
    }

    void mlp_layer(int il, int T) {
        const std::string pm = L(il) + ".mlp";
        const float* hfn = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_fn");
        const float* hsc = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_scale");
        const float* hbs = (const float*)tw(L(il) + ".hc_mlp_layer.hc_pre.hc_base");
        hipLaunchKernelGGL(k_hc_mix, dim3(T), dim3(256), 0, st, x, hfn, hsc, hbs, pre, post, 8, c.rms_eps, c.hc_eps, c.hc_mag);
        hipLaunchKernelGGL(k_hc_reduce_norm, dim3(T), dim3(256), 0, st, x, pre, (const float*)tw(L(il) + ".post_attention_layernorm.weight"), cur_h, cur_f, c.rms_eps);
        if (il == 0) {
            lin(pm + ".gate_proj.weight", cur_h, F1, T);  // [T][13824]
            lin(pm + ".up_proj.weight", cur_h, F2, T);
            size_t n = (size_t)T * 13824;
            hipLaunchKernelGGL(k_swiglu_h, dim3((n + 255) / 256), dim3(256), 0, st, F1, F2, H1, n);
            lin(pm + ".down_proj.weight", H1, F3, T);  // [T][3072]
            hipLaunchKernelGGL(k_add_rms, dim3(T), dim3(256), 0, st, F3, (const float*)nullptr, (const float*)tw(pm + ".post_fc2_norm.weight"), sub, HID, c.rms_eps);
        } else {
            // router
            linear_f32(st, (const float*)tw(pm + ".gate.weight"), 512, HID, cur_f, F1, T);  // [T][512]
            hipLaunchKernelGGL(k_router_topk, dim3(T), dim3(256), 0, st, F1, (const float*)tw(pm + ".gate.e_score_correction_bias"), moe_idx, moe_w, 512, 16, c.routed_scale);
            lin(pm + ".routed_expert_down_proj.weight", cur_h, z_f, T);  // [T][1536]
            size_t nz = (size_t)T * 1536;
            hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, z_f, z_h, nz);
            moe_experts(st, g.d_sec4, d_experts + (size_t)(il - 1) * 512, z_h, moe_idx, moe_w, zout_f, T, moe_ws);
            hipLaunchKernelGGL(k_f2h, dim3((nz + 255) / 256), dim3(256), 0, st, zout_f, z_h, nz);
            lin(pm + ".routed_expert_up_proj.weight", z_h, F2, T);  // [T][3072]
            lin(pm + ".shared_experts.gate_proj.weight", cur_h, F1, T);  // [T][1536]
            lin(pm + ".shared_experts.up_proj.weight", cur_h, F3, T);
            hipLaunchKernelGGL(k_swiglu_h, dim3((nz + 255) / 256), dim3(256), 0, st, F1, F3, H1, nz);
            lin(pm + ".shared_experts.down_proj.weight", H1, F3, T);  // [T][3072]
            hipLaunchKernelGGL(k_add_rms, dim3(T), dim3(256), 0, st, F2, (const float*)F3, (const float*)tw(pm + ".combine_output_norm.weight"), sub, HID, c.rms_eps);
        }
        hipLaunchKernelGGL(k_hc_post, dim3(T), dim3(256), 0, st, x, sub, post);
    }

    // process tokens [p0, p0+T) (T <= Tc) of the sequence tok_all (host, positions 0..p0+T-1). Returns false on error.
    bool forward_chunk(const int32_t* tok_all, int p0, int T, std::string& err) {
        HCK(hipMemcpyAsync(d_ids, tok_all + p0, (size_t)T * 4, hipMemcpyHostToDevice, st));
        hipLaunchKernelGGL(k_embed_hc, dim3(T), dim3(256), 0, st, (const uint8_t*)tw("model.embed_tokens.weight"), d_ids, x, T);
        if (engram_ready && !engram_prep(tok_all, p0, T, err)) return false;
        for (int il = 0; il < c.n_layers; ++il) {
            int mo = -1;
            for (int j = 0; j < 4; ++j)
                if (emeta.layers[j] == il) mo = j;
            if (mo >= 0 && engram_ready) {
                engram_layer(mo, il, T);
                maybe_dump("x_L" + std::to_string(il) + "_engram", T, 4 * HID, x, 4);
            }
            attention_layer(il, p0, T);
            maybe_dump("x_L" + std::to_string(il) + "_attn", T, 4 * HID, x, 4);
            mlp_layer(il, T);
            maybe_dump("x_L" + std::to_string(il) + "_mlp", T, 4 * HID, x, 4);
        }
        return true;
    }

    // logits of chunk positions [first, T) into d logits [(T-first)][V]; h_final dumped if dump
    void head(int T, int first) {
        int n = T - first;
        const float* hfn = (const float*)tw("model.hc_head.hc_head_fn");
        const float* hsc = (const float*)tw("model.hc_head.hc_head_scale");
        const float* hbs = (const float*)tw("model.hc_head.hc_head_base");
        const float* xs = x + (size_t)first * 4 * HID;
        hipLaunchKernelGGL(k_hc_mix, dim3(n), dim3(256), 0, st, xs, hfn, hsc, hbs, pre, post, 4, c.rms_eps, c.hc_eps, c.hc_mag);
        hipLaunchKernelGGL(k_hc_reduce_norm, dim3(n), dim3(256), 0, st, xs, pre, (const float*)tw("model.norm.weight"), hfin, (float*)nullptr, c.rms_eps);
        const TInfo& lm = ti("lm_head.weight");
        linear_h(st, lm.fmt, tw("lm_head.weight"), (int)lm.rows(), (int)lm.cols(), hfin, logits, n);
    }
};

}  // namespace hy4
