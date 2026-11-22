// engine/src/inkling2/cuda/ink_runner.cuh - Complete 48-Visit Inkling-2-Mini CUDA Runner (B=1..64, Tree-16, Full CUDA Graph, sm_70..sm_121a)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include "ink_attn.cuh"
#include "ink_cuda.cuh"
#include "ink_hybrid.cuh"
#include "ink_kernels.cuh"
#include "ink_manifest.hpp"
#include "ink_moe_cuda.cuh"

#include "strata/artifact/strata_reader.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata::ink {

// ---- Tree-32 Speculative Verification Descriptor (M <= 32 nodes rooted at anchor 0, topological order parent[i] < i) ----
struct InkTreeSpec {
    int M = 0;
    int32_t tok[kInkMaxTree] = {};
    int8_t  parent[kInkMaxTree] = {};
    uint8_t depth[kInkMaxTree] = {};
    uint8_t anc[kInkMaxTree][kInkMaxTreeDepth] = {};
    int8_t  sconv_lag[kInkMaxTree * 3] = {};  // for each node m: lag-1, lag-2, lag-3 indices (>=0 node, -1/-2/-3 history[2/1/0])

    void build() {
        for (int i = 0; i < M; ++i) {
            int cur = i;
            for (int d = (int)depth[i]; d >= 0; --d) {
                anc[i][d] = (uint8_t)cur;
                cur = (cur > 0) ? (int)parent[cur] : -1;
            }
            // ShortConv 3 lags: t-1, t-2, t-3
            int p = (i > 0) ? (int)parent[i] : -1;
            for (int l = 0; l < 3; ++l) {
                if (p >= 0) {
                    sconv_lag[i * 3 + l] = (int8_t)p;
                    p = (p > 0) ? (int)parent[p] : -1;
                } else {
                    // Remaining lags come from committed history: first miss = -1 (hist[2]), next = -2 (hist[1]), next = -3 (hist[0])
                    int miss = l - (int)depth[i];  // 0 -> -1, 1 -> -2, 2 -> -3
                    sconv_lag[i * 3 + l] = (int8_t)(-(miss + 1));
                }
            }
        }
    }
};

// (Exact 2-stage screened argmax kernels live in ink_kernels.cuh)

// ============================================================================
// Host GGUF_RAW -> canonical SF4 (E2M1 + E4M3 block-16) converter
// Used for exotic 1-2 bit GGUF quants (IQ1_S, IQ2_XXS, IQ3_XXS, Q2_0, etc.) or host-spilled GGUF experts!
// ============================================================================
inline uint8_t host_f32_to_e4m3_pos(float v) noexcept {
    if (v <= 0.0f) return 0u;
    if (v >= 448.0f) return 0x7Eu;
    // Exact round-to-nearest positive FP8 E4M3 (exp bias=7, 3 mantissa bits, subnormals exp=0 -> m * 2^-9)
    if (v < (1.0f / 64.0f)) {
        int m = (int)std::lround(v * 512.0f);
        if (m < 0) m = 0;
        if (m > 7) m = 7;
        return (uint8_t)m;
    }
    int exp = 0;
    float frac = std::frexp(v, &exp);  // v = frac * 2^exp, frac in [0.5, 1.0)
    int e_unb = exp - 1;               // v = (frac * 2) * 2^e_unb, mant in [1.0, 2.0)
    int e_bits = e_unb + 7;
    if (e_bits <= 0) {
        int m = (int)std::lround(v * 512.0f);
        return (uint8_t)std::max(0, std::min(7, m));
    }
    int m = (int)std::lround((frac * 2.0f - 1.0f) * 8.0f);
    if (m >= 8) { m = 0; ++e_bits; }
    if (e_bits >= 15) return (e_bits == 15 && m <= 6) ? (uint8_t)((15 << 3) | m) : 0x7Eu;
    return (uint8_t)((e_bits << 3) | (m & 7));
}

inline float host_e4m3_pos_to_f32(uint8_t b) noexcept {
    b &= 0x7Fu;
    int e = (b >> 3) & 0xF;
    int m = b & 7;
    if (e == 0) return (float)m * (1.0f / 512.0f);
    return std::ldexp(1.0f + (float)m * 0.125f, e - 7);
}

inline uint8_t host_f32_to_e2m1_nib(float v) noexcept {
    uint8_t sign = (v < 0.0f) ? 8u : 0u;
    float a = std::fabs(v);
    // Midpoints between {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    uint8_t mag = 0u;
    if      (a < 0.25f) mag = 0u;
    else if (a < 0.75f) mag = 1u;
    else if (a < 1.25f) mag = 2u;
    else if (a < 1.75f) mag = 3u;
    else if (a < 2.50f) mag = 4u;
    else if (a < 3.50f) mag = 5u;
    else if (a < 5.00f) mag = 6u;
    else                mag = 7u;
    return (uint8_t)(sign | mag);
}

inline void host_quant_row_sf4(const float* row, int K, float outer_scale, uint8_t* dst_p, uint8_t* dst_s) noexcept {
    const float inv_outer = (outer_scale > 0.0f) ? (1.0f / outer_scale) : 0.0f;
    const int nblk = K / 16;
    for (int b = 0; b < nblk; ++b) {
        const float* x = row + b * 16;
        float am = 0.0f;
        for (int i = 0; i < 16; ++i) am = std::max(am, std::fabs(x[i]));
        uint8_t sb = host_f32_to_e4m3_pos((am / 6.0f) * inv_outer);
        dst_s[b] = sb;
        const float sc = host_e4m3_pos_to_f32(sb) * outer_scale;
        const float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        uint8_t* p = dst_p + b * 8;
        for (int i = 0; i < 8; ++i) {
            uint8_t lo = host_f32_to_e2m1_nib(x[2 * i + 0] * inv);
            uint8_t hi = host_f32_to_e2m1_nib(x[2 * i + 1] * inv);
            p[i] = (uint8_t)((lo & 0xFu) | ((hi & 0xFu) << 4));
        }
    }
}

template <typename DequantFn, typename RowBytesFn>
inline bool repack_gguf_raw_body_to_sf4(
    const uint8_t* raw_body, uint32_t raw_dn_off,
    int D, int F,
    DequantFn&& dequant_row, RowBytesFn&& row_bytes,
    std::vector<uint8_t>& out_sf4, uint32_t& out_dn_off, std::string& err) {
    InkGgufExpertHeader hdr{};
    std::memcpy(&hdr, raw_body, sizeof(hdr));
    const size_t rb_gu = row_bytes(hdr.gu_type, D);
    const size_t rb_dn = row_bytes(hdr.dn_type, F);
    if (rb_gu == 0 || rb_dn == 0) {
        err = "SF4 repack: unsupported GGUF quant type (" + std::to_string(hdr.gu_type) + "/" + std::to_string(hdr.dn_type) + ")";
        return false;
    }

    // Dequant gate [F, D], up [F, D], down [D, F] once into FP32
    std::vector<float> g_buf((size_t)F * D), u_buf((size_t)F * D), d_buf((size_t)D * F);
    const uint8_t* gu_src = raw_body + sizeof(hdr);
    const uint8_t* dn_src = raw_body + raw_dn_off;
    float max_gu = 0.0f, max_dn = 0.0f;
    for (int f = 0; f < F; ++f) {
        float* gr = g_buf.data() + (size_t)f * D;
        float* ur = u_buf.data() + (size_t)f * D;
        if (!dequant_row(hdr.gu_type, gu_src + (size_t)f * rb_gu, gr, D) ||
            !dequant_row(hdr.gu_type, gu_src + (size_t)(F + f) * rb_gu, ur, D)) {
            err = "SF4 repack: gate/up dequant failed";
            return false;
        }
        for (int c = 0; c < D; ++c) {
            gr[c] *= hdr.gate_scale;
            ur[c] *= hdr.up_scale;
            max_gu = std::max(max_gu, std::max(std::fabs(gr[c]), std::fabs(ur[c])));
        }
    }
    for (int r = 0; r < D; ++r) {
        float* dr = d_buf.data() + (size_t)r * F;
        if (!dequant_row(hdr.dn_type, dn_src + (size_t)r * rb_dn, dr, F)) {
            err = "SF4 repack: down dequant failed";
            return false;
        }
        for (int c = 0; c < F; ++c) {
            dr[c] *= hdr.dn_scale;
            max_dn = std::max(max_dn, std::fabs(dr[c]));
        }
    }

    const float outer_gu = std::max(max_gu / (6.0f * 240.0f), 1e-6f);
    const float outer_dn = std::max(max_dn / (6.0f * 240.0f), 1e-6f);
    const size_t w13_p_bytes = (size_t)(2 * F) * (D / 2);
    const size_t w13_s_bytes = (size_t)(2 * F) * (D / 16);
    const size_t w2_p_bytes  = (size_t)D * (F / 2);
    const size_t w2_s_bytes  = (size_t)D * (F / 16);
    out_dn_off = (uint32_t)(16u + w13_p_bytes + w13_s_bytes);
    out_sf4.assign(out_dn_off + w2_p_bytes + w2_s_bytes, 0);

    float sf4_hdr[4] = {outer_gu, 1.0f, outer_dn, 1.0f};
    std::memcpy(out_sf4.data(), sf4_hdr, 16);
    uint8_t* w13_p = out_sf4.data() + 16;
    uint8_t* w13_s = w13_p + w13_p_bytes;
    for (int f = 0; f < F; ++f) {
        host_quant_row_sf4(g_buf.data() + (size_t)f * D, D, outer_gu,
                           w13_p + (size_t)(2 * f + 0) * (D / 2), w13_s + (size_t)(2 * f + 0) * (D / 16));
        host_quant_row_sf4(u_buf.data() + (size_t)f * D, D, outer_gu,
                           w13_p + (size_t)(2 * f + 1) * (D / 2), w13_s + (size_t)(2 * f + 1) * (D / 16));
    }
    uint8_t* w2_p = out_sf4.data() + out_dn_off;
    uint8_t* w2_s = w2_p + w2_p_bytes;
    for (int r = 0; r < D; ++r) {
        host_quant_row_sf4(d_buf.data() + (size_t)r * F, F, outer_dn,
                           w2_p + (size_t)r * (F / 2), w2_s + (size_t)r * (F / 16));
    }
    return true;
}

// ============================================================================
// InkCudaModel: GPU-resident .strata v1.1 / .gguf / HF Trunk + Experts (Single-GPU & Dual-GPU EP!)
// ============================================================================
struct InkCudaModel {
    InkManifest man;
    strata::artifact::StrataFileReader rd;
    std::string path;
    int primary_gpu = 0;
    int peer_gpu = -1;
    bool ep2_active = false;
    bool ep2_p2p = false;
    float ep_split = 0.5f;
    // Multi-GPU Pipeline Layer-Split (--layer-split auto | K1,K2,...) across 2..4 GPUs (0 duplicated KV/experts!)
    static constexpr int kMaxStages = 4;
    bool ls_active = false;
    int  n_stages = 1;
    int  stage_gpu[kMaxStages] = {0, -1, -1, -1};
    int  stage_cc[kMaxStages]  = {};
    int  stage_l0[kMaxStages]  = {0, 24, 24, 24};
    int  stage_l1[kMaxStages]  = {24, 24, 24, 24};
    int  layer_owner[24]       = {};
    bool stage_p2p[kMaxStages][kMaxStages] = {};
    bool stage_trunk_attn_fp16[kMaxStages] = {};
    uint8_t* stage_d_trunk[kMaxStages] = {};
    uint8_t* stage_d_sec4[kMaxStages]  = {};
    size_t   stage_sec4_bytes[kMaxStages] = {};
    size_t   stage_n_experts[kMaxStages]  = {};
    std::unordered_map<std::string, size_t> stage_trunk_off[kMaxStages];
    int sm_cc = 0;  // major * 10 + minor (e.g. 60, 61, 70, 75, 86, 89, 120, 121)
    bool trunk_attn_fp16 = false;  // true on sm_70/75: .attn.* BF16 weights converted in-place to FP16 at load!

    uint8_t* d_trunk = nullptr;
    uint8_t* d_sec4  = nullptr;
    uint8_t* d_sec4_peer = nullptr;
    uint8_t* h_sec4_pinned = nullptr;  // optional pinned overflow for 8/12/16 GB GPUs
    size_t trunk_bytes = 0, sec4_bytes = 0, sec4_peer_bytes = 0;
    size_t n_vram_experts = 0, n_peer_experts = 0, n_host_experts = 0;

    // Preprocessed per-layer F32 buffers (tiny: < 40 MB total)
    float* d_msg_queries[12] = {};      // [4, 2304] F32 for physical layers 6..17
    float* d_sconv_k[24] = {};          // [1024, 4] F32 (lag 3..0)
    float* d_sconv_v[24] = {};          // [1024, 4] F32 (lag 3..0)
    float* d_sconv_attn[24] = {};       // [2304, 4] F32
    float* d_sconv_mlp[24] = {};        // [2304, 4] F32

    // Per-physical-layer expert directory on GPU 0 + optional GPU 1 (Dual-GPU EP) + Hybrid Engine
    InkExpertDev* d_experts[24] = {};
    InkExpertDev* d_experts_peer[24] = {};
    InkLayerMoeInfo moe_info[24]{};
    InkLayerMoeInfo moe_info_peer[24]{};
    bool layer_has_host[24] = {};
    bool layer_has_peer[24] = {};
    InkHybridEngine hybrid;

    // Exact 2-stage unembed screening index (for BF16 unembed in SQ-20)
    UnembedScreenIndex unembed_screen;

    const uint8_t* trunk_ptr(const std::string& name) const {
        return d_trunk + man.get(name).off;
    }
    const uint8_t* trunk_ptr_stage(int s, const std::string& name) const {
        if (!ls_active || s <= 0) return d_trunk + man.get(name).off;
        auto it = stage_trunk_off[s].find(name);
        if (it != stage_trunk_off[s].end()) return stage_d_trunk[s] + it->second;
        return d_trunk + man.get(name).off;
    }
    const uint8_t* sec4_stage(int s) const {
        return (ls_active && s > 0) ? stage_d_sec4[s] : d_sec4;
    }
    const TensorInfo& tinfo(const std::string& name) const {
        return man.get(name);
    }

    static size_t wddm_safety_floor(size_t vtot) noexcept {
#ifdef _WIN32
        if (vtot <= (6ull << 30))  return 640ull << 20;
        if (vtot <= (10ull << 30)) return 896ull << 20;
        return 1152ull << 20;
#else
        return (vtot <= (10ull << 30)) ? (384ull << 20) : (512ull << 20);
#endif
    }

    static size_t safe_host_ram_cap(double ram_experts_gb, bool ssd_experts) noexcept {
        if (ssd_experts || ram_experts_gb == 0.0) return 0u;
        if (ram_experts_gb > 0.0) return static_cast<size_t>(ram_experts_gb * 1073741824.0);
        const uint64_t avail = ink_ram_available_bytes();
        if (avail == 0) return static_cast<size_t>(-1);
        const uint64_t os_res = std::max<uint64_t>(2ull << 30, std::min<uint64_t>(avail / 4, 4ull << 30));
        return (avail > os_res) ? static_cast<size_t>(avail - os_res) : 0u;
    }

    bool load(const std::string& strata_path, int gpu_id, bool build_screen, std::string& err,
              size_t extra_reserve_bytes = 0, double vram_experts_gb = -1.0,
              double ram_experts_gb = -1.0, bool ssd_experts = false, int cpu_threads = 0) {
        path = strata_path;
        primary_gpu = gpu_id;
        INK_CK(cudaSetDevice(primary_gpu));
        {
            cudaDeviceProp prop{};
            INK_CK(cudaGetDeviceProperties(&prop, primary_gpu));
            sm_cc = prop.major * 10 + prop.minor;
        }
        if (!rd.open(strata_path, err)) return false;
        parse_ink_manifest(rd.manifest_json(), man, true);

        const auto& hdr = rd.header();
        trunk_bytes = (size_t)hdr.sections[1].byte_length;
        sec4_bytes  = (size_t)hdr.sections[4].byte_length;

        double t0 = now_s();
        // 1. Load Section 1 (Contiguous GPU Trunk + DSpark) in one DMA transfer
        INK_CK(cudaMalloc(&d_trunk, trunk_bytes));
        {
            const size_t CH = 64ull << 20;
            uint8_t* hbuf = halloc_pinned<uint8_t>(std::min(CH, trunk_bytes));
            size_t done = 0;
            while (done < trunk_bytes) {
                size_t want = std::min(CH, trunk_bytes - done);
                if (!rd.read_at(hdr.sections[1].file_offset + done, hbuf, want, err)) {
                    cudaFreeHost(hbuf);
                    return false;
                }
                INK_CK(cudaMemcpy(d_trunk + done, hbuf, want, cudaMemcpyHostToDevice));
                done += want;
            }
            INK_CK(cudaFreeHost(hbuf));
        }
        double t_trunk = now_s() - t0;

        // 1B. On Volta sm_70 (V100) & Turing sm_75 (2080 Ti), convert .attn.* BF16 weights IN-PLACE to IEEE FP16 once!
        //     Eliminates 336+ per-visit conversion launches & unlocks 125 TFLOPS direct FP16 Tensor Core GEMMs!
        trunk_attn_fp16 = (sm_cc >= 70 && sm_cc < 80);
        if (trunk_attn_fp16) {
            for (const auto& kv : man.trunk) {
                if (kv.second.fmt == FMT_BF16 && kv.first.find(".attn.") != std::string::npos) {
                    const size_t nw = (size_t)kv.second.rows() * kv.second.cols();
                    __nv_bfloat16* ptr = reinterpret_cast<__nv_bfloat16*>(d_trunk + kv.second.off);
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256>>>(ptr, reinterpret_cast<__half*>(ptr), nw);
                }
            }
            INK_CK(cudaDeviceSynchronize());
        }

        // 2. Check free VRAM -> load Section 4 experts (100% VRAM when fits, else 3x loop-weighted vram_load_order!)
        size_t vfree = 0, vtot = 0;
        INK_CK(ink_cuda_mem_info(&vfree, &vtot));
        const auto& dir = rd.expert_directory();
        const int n_total = (int)dir.size();  // 24 * 128 = 3072
        std::vector<InkExpertDev> h_exp(n_total);
        for (int il = 0; il < man.c.n_phys; ++il) {
            moe_info[il] = {};
            for (int r = 0; r < man.c.n_experts; ++r) {
                const auto& e = dir[(size_t)il * man.c.n_experts + r];
                InkExpertDev dev{};
                dev.body_off = e.body_offset;
                dev.dn_off   = e.dn_offset;
                dev.gu_code  = e.gu_code_id;
                dev.dn_code  = e.dn_code_id;
                dev.tier     = 0;
                h_exp[(size_t)il * man.c.n_experts + e.expert_id] = dev;
                if (e.gu_code_id == FMT_SF4) moe_info[il].has_sf4 = true;
                else if (e.gu_code_id == FMT_MX8) moe_info[il].has_mx8 = true;
                else {
                    moe_info[il].has_sk = true;
                    if (e.gu_code_id < 32) moe_info[il].sk_gu_mask |= (1u << e.gu_code_id);
                }
                if (e.dn_code_id != FMT_SF4 && e.dn_code_id != FMT_MX8 && e.dn_code_id < 32) {
                    moe_info[il].has_sk = true;
                    moe_info[il].sk_dn_mask |= (1u << e.dn_code_id);
                }
            }
        }

        double t1 = now_s();
        // Guaranteed headroom BEFORE sizing VRAM experts: KV init + O(1) prefill ws (144 MiB) + DSpark + runner ws + OS/WDDM floor!
        const size_t wddm_floor = wddm_safety_floor(vtot);
        const size_t base_reserve = ((vtot >= (20ull << 30)) ? (1600ull << 20)
                                   : (vtot <= (13ull << 30)) ? (960ull << 20)
                                                             : (1280ull << 20)) + (wddm_floor > (512ull << 20) ? (wddm_floor - (512ull << 20)) : 0ull);
        const size_t reserve = std::max(base_reserve, extra_reserve_bytes + wddm_floor);
        const size_t user_vram_cap = (vram_experts_gb >= 0.0)
            ? static_cast<size_t>(vram_experts_gb * 1073741824.0)
            : static_cast<size_t>(-1);
        for (int il = 0; il < 24; ++il) layer_has_host[il] = false;
        if (!ssd_experts && vfree > sec4_bytes + reserve && sec4_bytes <= user_vram_cap) {
            // 100% VRAM resident! Single sequential streaming DMA!
            INK_CK(cudaMalloc(&d_sec4, sec4_bytes));
            const size_t CH = 64ull << 20;
            uint8_t* hbuf = halloc_pinned<uint8_t>(std::min(CH, sec4_bytes));
            size_t done = 0;
            while (done < sec4_bytes) {
                size_t want = std::min(CH, sec4_bytes - done);
                if (!rd.read_at(hdr.sections[4].file_offset + done, hbuf, want, err)) {
                    cudaFreeHost(hbuf);
                    return false;
                }
                INK_CK(cudaMemcpy(d_sec4 + done, hbuf, want, cudaMemcpyHostToDevice));
                done += want;
            }
            INK_CK(cudaFreeHost(hbuf));
            n_vram_experts = n_total;
            n_host_experts = 0;
        } else {
            // 3x Loop-Weighted Tiered Residency + Strata Predictive CPU/RAM Hybrid Engine (Doorbell + Ping-Pong DMA) + Auto-SSD!
            const auto& vorder = rd.vram_load_order();
            std::vector<uint16_t> order;
            if ((int)vorder.size() == n_total) {
                order = vorder;
            } else {
                // Guarantee 3x looped core layers (core_start..core_end-1) are ALWAYS prioritized before 1x prelude/coda!
                order.reserve(n_total);
                for (int il = man.c.core_start; il < man.c.core_end; ++il)
                    for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
                for (int il = 0; il < man.c.core_start; ++il)
                    for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
                for (int il = man.c.core_end; il < man.c.n_phys; ++il)
                    for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
            }
            // Reserve full headroom (runner + KV init + O(1) prefill + DSpark + WDDM floor + 2x pf_stage)!
            size_t max_body_all = 0;
            for (const auto& e : dir) max_body_all = std::max<size_t>(max_body_all, e.body_bytes);
            const size_t pf_bytes = 2 * InkHybridEngine::pf_stage_size(max_body_all);
            const size_t hyb_post_res = std::max(reserve, (size_t)(1408ull << 20) + wddm_floor / 2);
            const size_t hyb_reserve = hyb_post_res + pf_bytes + (64ull << 20);
            size_t vram_budget = (vfree > hyb_reserve) ? (vfree - hyb_reserve) : 0;
            if (user_vram_cap < vram_budget) vram_budget = user_vram_cap;
            const size_t ram_cap = safe_host_ram_cap(ram_experts_gb, ssd_experts);
            size_t vram_used = 0, host_used = 0, max_exp_bytes = 0, n_ssd_experts = 0;
            std::vector<uint8_t> in_vram(n_total, 0);
            std::vector<uint8_t> in_ram(n_total, 0);
            const bool force_ssd_only = ssd_experts || (ram_experts_gb == 0.0);
            for (uint16_t di : order) {
                const auto& e = dir[di];
                max_exp_bytes = std::max<size_t>(max_exp_bytes, e.body_bytes);
                const size_t al = (size_t)((e.body_bytes + 63ull) & ~63ull);
                if (vram_used + e.body_bytes <= vram_budget) {
                    in_vram[di] = 1;
                    vram_used += e.body_bytes;
                    ++n_vram_experts;
                } else if (!force_ssd_only && host_used + al <= ram_cap) {
                    in_ram[di] = 1;
                    host_used += al;
                    ++n_host_experts;
                } else {
                    ++n_ssd_experts;
                    ++n_host_experts;
                }
            }
            // #6617: Allocate ping-pong prefill staging buffers BEFORE d_sec4 and the pinned host arena so
            // pinned-RAM VRAM page tables can never starve pf_buf to 0 MB and silently skip host experts!
            if (n_host_experts > 0) {
                hybrid.start_workers(cpu_threads);
                if (!hybrid.alloc_pf_stage(max_exp_bytes)) {
                    err = "not enough free VRAM for the 2x prefill staging buffers required by host experts; close other GPU apps or use a smaller pack";
                    return false;
                }
            }
            INK_CK(cudaMalloc(&d_sec4, std::max<size_t>(256, vram_used)));
            uint8_t* h_arena_base = nullptr;
            if (n_host_experts > 0) {
                if (host_used > 0) {
                    h_arena_base = hybrid.alloc_host_arena(host_used);
                    if (!h_arena_base) { err = "cannot allocate host expert arena"; return false; }
                }
                hybrid.host_meta.assign((size_t)man.c.n_phys * man.c.n_experts, {});
                hybrid.orig_host_off.assign((size_t)man.c.n_phys * man.c.n_experts, 0);
            }
            if (n_ssd_experts > 0) {
                hybrid.ssd_only = true;
                std::vector<ink2::InkColdExpertDesc> cdescs((size_t)man.c.n_phys * man.c.n_experts);
                for (int di = 0; di < n_total; ++di) {
                    const auto& e = dir[di];
                    const int il = di / man.c.n_experts;
                    const int key = il * man.c.n_experts + e.expert_id;
                    cdescs[key].file_off   = hdr.sections[4].file_offset + e.body_offset;
                    cdescs[key].body_bytes = e.body_bytes;
                    cdescs[key].dn_off     = e.dn_offset;
                    cdescs[key].gu_code    = e.gu_code_id;
                    cdescs[key].dn_code    = e.dn_code_id;
                }
                // Bounded 1 GiB (or user ram_cap up to 2 GiB) page-locked DMA bounce/prefetch ring (0 OS swap)!
                const size_t ring_bytes = (ram_experts_gb > 0.0)
                    ? std::min<size_t>(2ull << 30, std::max<size_t>(256ull << 20, ram_cap))
                    : (512ull << 20);
                hybrid.cold_store = std::make_unique<ink2::InkColdStore>();
                auto pin_alloc = [](size_t b) -> uint8_t* {
                    uint8_t* p = nullptr;
                    if (cudaHostAlloc(&p, b, cudaHostAllocDefault) == cudaSuccess) return p;
                    return nullptr;
                };
                auto pin_free = [](uint8_t* p) { if (p) cudaFreeHost(p); };
                if (!hybrid.cold_store->open(strata_path, std::move(cdescs), max_exp_bytes, ring_bytes, 4, err, pin_alloc, pin_free)) {
                    return false;
                }
                INK_CK(cudaHostAlloc(&hybrid.h_copyback_buf, (max_exp_bytes + 255ull) & ~255ull, cudaHostAllocDefault));
                hybrid.h_copyback_bytes = (max_exp_bytes + 255ull) & ~255ull;
                std::fprintf(stderr, "[inkling-cuda] GPU+SSD streaming active: %zu VRAM (%.2f GiB), %zu RAM, %zu SSD-only "
                                     "(%d-slot pinned DMA ring = %.0f MiB, direct=%d)\n",
                             n_vram_experts, (double)vram_used / 1073741824.0,
                             n_host_experts - n_ssd_experts, n_ssd_experts,
                             hybrid.cold_store->num_slots(), (double)hybrid.cold_store->pool_bytes() / 1048576.0,
                             (int)hybrid.cold_store->is_direct());
            }

            size_t v_cur = 0, h_cur = 0;
            std::vector<uint8_t> tmp_body;
            for (int di = 0; di < n_total; ++di) {
                const auto& e = dir[di];
                const int il = di / man.c.n_experts;
                const int key = il * man.c.n_experts + e.expert_id;
                auto& dev = h_exp[key];
                if (in_vram[di]) {
                    tmp_body.resize(e.body_bytes);
                    if (!rd.read_at(hdr.sections[4].file_offset + e.body_offset, tmp_body.data(), e.body_bytes, err)) return false;
                    INK_CK(cudaMemcpy(d_sec4 + v_cur, tmp_body.data(), e.body_bytes, cudaMemcpyHostToDevice));
                    dev.body_off = v_cur;
                    dev.tier = 0;
                    v_cur += e.body_bytes;
                } else if (in_ram[di]) {
                    tmp_body.resize(e.body_bytes);
                    if (!rd.read_at(hdr.sections[4].file_offset + e.body_offset, tmp_body.data(), e.body_bytes, err)) return false;
                    uint8_t* hdst = h_arena_base + h_cur;
                    std::memcpy(hdst, tmp_body.data(), e.body_bytes);
                    hybrid.host_meta[key].body       = hdst;
                    hybrid.host_meta[key].body_bytes = e.body_bytes;
                    hybrid.host_meta[key].dn_off     = e.dn_offset;
                    hybrid.host_meta[key].gu_code    = e.gu_code_id;
                    hybrid.host_meta[key].dn_code    = e.dn_code_id;
                    hybrid.host_meta[key].cold_ssd   = false;
                    hybrid.orig_host_off[key]        = h_cur;
                    dev.body_off = h_cur;
                    dev.tier = 2;  // CPU Doorbell (M=1) + Ping-Pong DMA Streaming (M>=2)!
                    layer_has_host[il] = true;
                    h_cur += (size_t)((e.body_bytes + 63ull) & ~63ull);
                } else {
                    // Cold SSD expert: ZERO host RAM allocated at startup! Streamed via pinned InkColdStore DMA ring!
                    hybrid.host_meta[key].body       = nullptr;
                    hybrid.host_meta[key].body_bytes = e.body_bytes;
                    hybrid.host_meta[key].dn_off     = e.dn_offset;
                    hybrid.host_meta[key].gu_code    = e.gu_code_id;
                    hybrid.host_meta[key].dn_code    = e.dn_code_id;
                    hybrid.host_meta[key].cold_ssd   = true;
                    hybrid.orig_host_off[key]        = 0;
                    dev.body_off = 0;
                    dev.tier = 2;
                    layer_has_host[il] = true;
                }
            }
            if (n_host_experts > 0) {
                hybrid.start_workers(cpu_threads);
                size_t vf_now = 0, vt_now = 0;
                INK_CK(ink_cuda_mem_info(&vf_now, &vt_now));
                const size_t avail = (vf_now > hyb_post_res) ? (vf_now - hyb_post_res) : 0;
                hybrid.alloc_staging(max_exp_bytes, avail);
                if (hybrid.cold_store) {
                    std::vector<int> idle_keys;
                    for (uint16_t di : order) {
                        if (!in_vram[di] && !in_ram[di]) {
                            const int il = di / man.c.n_experts;
                            idle_keys.push_back(il * man.c.n_experts + dir[di].expert_id);
                        }
                    }
                    hybrid.cold_store->start_idle_preloader(std::move(idle_keys), 500);
                }
            }
        }
        double t_exp = now_s() - t1;

        for (int il = 0; il < man.c.n_phys; ++il) {
            d_experts[il] = dalloc<InkExpertDev>(man.c.n_experts);
            INK_CK(cudaMemcpy(d_experts[il], h_exp.data() + (size_t)il * man.c.n_experts,
                              (size_t)man.c.n_experts * sizeof(InkExpertDev), cudaMemcpyHostToDevice));
        }

        // 3. Preprocess msg_queries[6..17] and static_sconv[0..23] into exact contiguous F32 GPU arrays
        {
            const auto& ti_q  = man.get("model.llm.middle_cycle_cross_pass_queries");
            const auto& ti_kv = man.get("model.llm.middle_cycle_cross_pass_kv_queries");
            std::vector<float> h_q(24 * kInkHidden), h_kv(24 * kInkHidden);
            INK_CK(cudaMemcpy(h_q.data(), d_trunk + ti_q.off, h_q.size() * 4, cudaMemcpyDeviceToHost));
            INK_CK(cudaMemcpy(h_kv.data(), d_trunk + ti_kv.off, h_kv.size() * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < 12; ++i) {
                std::vector<float> mq(4 * kInkHidden);
                std::memcpy(mq.data() + 0 * kInkHidden, h_q.data()  + (2 * i)     * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 1 * kInkHidden, h_kv.data() + (2 * i)     * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 2 * kInkHidden, h_kv.data() + (2 * i + 1) * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 3 * kInkHidden, h_q.data()  + (2 * i + 1) * kInkHidden, kInkHidden * 4);
                d_msg_queries[i] = dalloc<float>(4 * kInkHidden);
                INK_CK(cudaMemcpy(d_msg_queries[i], mq.data(), mq.size() * 4, cudaMemcpyHostToDevice));
            }
        }
        for (int il = 0; il < man.c.n_phys; ++il) {
            const std::string p = "model.llm.layers." + std::to_string(il);
            d_sconv_attn[il] = (float*)(d_trunk + man.get(p + ".attn_sconv.weight").off);
            d_sconv_mlp[il]  = (float*)(d_trunk + man.get(p + ".mlp_sconv.weight").off);
            if (man.c.is_swa[il]) {
                d_sconv_k[il] = (float*)(d_trunk + man.get(p + ".attn.k_sconv.weight").off);
                d_sconv_v[il] = (float*)(d_trunk + man.get(p + ".attn.v_sconv.weight").off);
            } else {
                // Global static_w is [4, 1024] with lag 0 first -> flip(0).T -> [1024, 4] with lag 3..0!
                auto flip_t = [&](const std::string& tname) -> float* {
                    std::vector<float> src(4 * kInkKVDim), dst(kInkKVDim * 4);
                    INK_CK(cudaMemcpy(src.data(), d_trunk + man.get(tname).off, src.size() * 4, cudaMemcpyDeviceToHost));
                    for (int c = 0; c < kInkKVDim; ++c) {
                        for (int l = 0; l < 4; ++l) dst[c * 4 + l] = src[(3 - l) * kInkKVDim + c];
                    }
                    float* dptr = dalloc<float>(dst.size());
                    INK_CK(cudaMemcpy(dptr, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice));
                    return dptr;
                };
                d_sconv_k[il] = flip_t(p + ".attn.k_sconv.static_w");
                d_sconv_v[il] = flip_t(p + ".attn.v_sconv.static_w");
            }
        }

        // 4. Build exact 2-stage Cauchy-Schwarz screening index for BF16 or S6_K unembed (SQ-20, SQ-15, SQ-10)!
        const auto& ui = man.get("model.llm.unembed.weight");
        if (build_screen && (ui.fmt == FMT_BF16 || ui.fmt == FMT_S6_K)) {
            INK_CK(ink_cuda_mem_info(&vfree, &vtot));
            if (vfree > (2048ull << 20)) {
                unembed_screen.V = man.c.vocab_unpadded;
                unembed_screen.q8  = dalloc<int8_t>((size_t)unembed_screen.V * kInkHidden);
                unembed_screen.sc  = dalloc<float>(unembed_screen.V);
                unembed_screen.eps = dalloc<float>(unembed_screen.V);
                if (ui.fmt == FMT_BF16) {
                    k_ink_build_unembed_screen_bf16<<<(unembed_screen.V + 7) / 8, 256>>>(
                        reinterpret_cast<const uint16_t*>(d_trunk + ui.off),
                        unembed_screen.q8, unembed_screen.sc, unembed_screen.eps, unembed_screen.V);
                } else {
                    k_ink_build_unembed_screen_s6k<<<(unembed_screen.V + 7) / 8, 256>>>(
                        d_trunk + ui.off,
                        unembed_screen.q8, unembed_screen.sc, unembed_screen.eps, unembed_screen.V);
                }
                INK_CK(cudaDeviceSynchronize());
                unembed_screen.ready = true;
            }
        }

        std::fprintf(stderr, "[InkCUDA] Loaded %s (%s): trunk=%.2f GB (%.2fs), experts=%zu VRAM / %zu host (%.2fs, hyb=%d, promo=%d, pf_buf=%.0fMB), sm120_tc=%d, screen=%d\n",
                     strata_path.c_str(), man.build.c_str(), (double)trunk_bytes / 1e9, t_trunk,
                     n_vram_experts, n_host_experts, t_exp, (int)hybrid.active, hybrid.n_promo_slots,
                     (double)hybrid.pf_stage_bytes / (1024.0 * 1024.0), (int)sm120::available(), (int)unembed_screen.ready);
        return true;
    }

    // Hardware P2P roundtrip + latency probe between g0 and g1.
    // On PCIe root-complex systems without NVLink/PCIe-switch, cudaDeviceCanAccessPeer can report true
    // while peer DMA stalls for tens of milliseconds; benchmark P2P vs portable pinned-host bounce and
    // auto-select whichever is genuinely faster (honoring STRATA_CUDA_PEER_PINNED=1 / STRATA_CUDA_PEER_P2P=1).
    static bool probe_p2p(int g0, int g1) {
        if (const char* env = std::getenv("STRATA_CUDA_PEER_PINNED"); env && std::atoi(env) != 0) return false;
        const bool force_p2p = []() {
            const char* env = std::getenv("STRATA_CUDA_PEER_P2P");
            return env && std::atoi(env) != 0;
        }();
        int can01 = 0, can10 = 0;
        if (cudaDeviceCanAccessPeer(&can01, g0, g1) != cudaSuccess ||
            cudaDeviceCanAccessPeer(&can10, g1, g0) != cudaSuccess || !can01 || !can10) {
            return false;
        }
        cudaSetDevice(g0);
        cudaError_t e0 = cudaDeviceEnablePeerAccess(g1, 0);
        if (e0 != cudaSuccess && e0 != cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); return false; }
        cudaSetDevice(g1);
        cudaError_t e1 = cudaDeviceEnablePeerAccess(g0, 0);
        if (e1 != cudaSuccess && e1 != cudaErrorPeerAccessAlreadyEnabled) {
            cudaSetDevice(g0); cudaDeviceDisablePeerAccess(g1); cudaGetLastError();
            return false;
        }
        cudaGetLastError();

        // 9,216-byte (1x kInkHidden float) roundtrip verification + latency comparison
        constexpr size_t kN = kInkHidden;
        constexpr size_t kB = kN * sizeof(float);
        std::vector<uint32_t> h_tx(kN), h_rx01(kN, 0), h_rx10(kN, 0);
        for (size_t i = 0; i < kN; ++i) h_tx[i] = (uint32_t)(0x12345678u ^ (i * 2654435761u));
        uint32_t *d0 = nullptr, *d1 = nullptr, *hp = nullptr;
        cudaStream_t s0 = nullptr, s1 = nullptr;
        cudaSetDevice(g0);
        cudaMalloc(&d0, kB);
        cudaHostAlloc(&hp, kB, cudaHostAllocPortable);
        cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
        cudaMemcpy(d0, h_tx.data(), kB, cudaMemcpyHostToDevice);
        cudaSetDevice(g1);
        cudaMalloc(&d1, kB);
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cudaMemset(d1, 0, kB);

        bool ok = (cudaMemcpyPeer(d1, g1, d0, g0, kB) == cudaSuccess) &&
                  (cudaMemcpy(h_rx01.data(), d1, kB, cudaMemcpyDeviceToHost) == cudaSuccess);
        cudaSetDevice(g0);
        ok = ok && (cudaMemcpyPeer(d0, g0, d1, g1, kB) == cudaSuccess) &&
             (cudaMemcpy(h_rx10.data(), d0, kB, cudaMemcpyDeviceToHost) == cudaSuccess);
        ok = ok && (std::memcmp(h_tx.data(), h_rx01.data(), kB) == 0) &&
             (std::memcmp(h_tx.data(), h_rx10.data(), kB) == 0);

        double t_p2p_us = 1e9, t_pin_us = 1e9;
        if (ok && !force_p2p) {
            // Warmup + up to 8 timed roundtrips via P2P (early-exit if a single roundtrip stalls > 5 ms!)
            cudaSetDevice(g1);
            const double tw0 = now_s();
            cudaMemcpyPeerAsync(d1, g1, d0, g0, kB, s1);
            cudaMemcpyPeerAsync(d0, g0, d1, g1, kB, s1);
            cudaStreamSynchronize(s1);
            const double t_warm_us = (now_s() - tw0) * 1e6;
            if (t_warm_us > 5000.0) {
                t_p2p_us = t_warm_us;
            } else {
                const double tp0 = now_s();
                for (int it = 0; it < 8; ++it) {
                    cudaMemcpyPeerAsync(d1, g1, d0, g0, kB, s1);
                    cudaMemcpyPeerAsync(d0, g0, d1, g1, kB, s1);
                    cudaStreamSynchronize(s1);
                }
                t_p2p_us = (now_s() - tp0) * 1e6 / 8.0;
            }

            // Warmup + 8 timed roundtrips via portable pinned-host bounce
            cudaEvent_t ev0 = nullptr, ev1 = nullptr;
            cudaSetDevice(g0); cudaEventCreateWithFlags(&ev0, cudaEventDisableTiming);
            cudaSetDevice(g1); cudaEventCreateWithFlags(&ev1, cudaEventDisableTiming);
            auto do_pin = [&]() {
                cudaSetDevice(g0);
                cudaMemcpyAsync(hp, d0, kB, cudaMemcpyDeviceToHost, s0);
                cudaEventRecord(ev0, s0);
                cudaSetDevice(g1);
                cudaStreamWaitEvent(s1, ev0, 0);
                cudaMemcpyAsync(d1, hp, kB, cudaMemcpyHostToDevice, s1);
                cudaMemcpyAsync(hp, d1, kB, cudaMemcpyDeviceToHost, s1);
                cudaEventRecord(ev1, s1);
                cudaSetDevice(g0);
                cudaStreamWaitEvent(s0, ev1, 0);
                cudaMemcpyAsync(d0, hp, kB, cudaMemcpyHostToDevice, s0);
                cudaStreamSynchronize(s0);
            };
            do_pin();
            const double th0 = now_s();
            for (int it = 0; it < 8; ++it) do_pin();
            t_pin_us = (now_s() - th0) * 1e6 / 8.0;
            cudaSetDevice(g0); cudaEventDestroy(ev0);
            cudaSetDevice(g1); cudaEventDestroy(ev1);
        }

        cudaSetDevice(g0); cudaStreamDestroy(s0); cudaFree(d0); cudaFreeHost(hp);
        cudaSetDevice(g1); cudaStreamDestroy(s1); cudaFree(d1);
        cudaSetDevice(g0);
        const bool use_p2p = ok && (force_p2p || (t_p2p_us < t_pin_us * 0.90 && t_p2p_us < 500.0));
        if (!use_p2p) {
            cudaSetDevice(g0); cudaDeviceDisablePeerAccess(g1); cudaGetLastError();
            cudaSetDevice(g1); cudaDeviceDisablePeerAccess(g0); cudaGetLastError();
            cudaSetDevice(g0);
        }
        if (ok && !force_p2p) {
            std::fprintf(stderr, "[InkCUDA] EP2 GPU%d<->GPU%d transport: %s (p2p=%.1f us vs pinned=%.1f us)\n",
                         g0, g1, use_p2p ? "direct-p2p" : "pinned-host", t_p2p_us, t_pin_us);
        }
        return use_p2p;
    }

    // Unified fast GPU loader for .strata v1.1, .gguf (Unsloth + ggml-org), HF safetensors,
    // Dual-GPU EP (`cuda-ep2`), AND Multi-GPU Pipeline Layer-Split (`cuda-ls`, 2..4 GPUs)!
    template <typename ModelFileT, typename DequantFn, typename RowBytesFn>
    bool load_from_model_file(
        const ModelFileT& mf,
        int gpu0,
        int gpu1,
        float ep_split_in,
        bool build_screen,
        std::string& err,
        size_t extra_reserve_bytes,
        DequantFn&& dequant_row,
        RowBytesFn&& row_bytes,
        double vram_experts_gb = -1.0,
        double ram_experts_gb = -1.0,
        bool ssd_experts = false,
        const std::vector<int>& all_gpus = {},
        const std::string& layer_split_spec = "",
        int cpu_threads = 0) {
        // Fast zero-overhead path for single-GPU .strata v1.1!
        const bool is_strata_fmt = (mf.m.format.rfind("strata", 0) == 0);
        const bool single_gpu_req = (gpu1 < 0) && (all_gpus.size() <= 1) && layer_split_spec.empty();
        if (is_strata_fmt && single_gpu_req) {
            return load(mf.path, gpu0, build_screen, err, extra_reserve_bytes, vram_experts_gb, ram_experts_gb, ssd_experts, cpu_threads);
        }

        // Collect deduplicated active GPU list (1..kMaxStages)
        std::vector<int> gpus;
        if (!all_gpus.empty()) {
            for (int g : all_gpus) {
                if (g >= 0 && std::find(gpus.begin(), gpus.end(), g) == gpus.end() && (int)gpus.size() < kMaxStages) {
                    gpus.push_back(g);
                }
            }
        }
        if (gpus.empty()) {
            gpus.push_back(gpu0);
            if (gpu1 >= 0 && gpu1 != gpu0) gpus.push_back(gpu1);
        }

        path = mf.path;
        primary_gpu = gpus[0];
        peer_gpu = (gpus.size() == 2) ? gpus[1] : -1;
        ep2_active = false;
        ep2_p2p = false;
        ls_active = false;
        n_stages = 1;
        stage_gpu[0] = primary_gpu;
        for (int il = 0; il < 24; ++il) layer_owner[il] = 0;
        ep_split = std::max(0.1f, std::min(0.9f, ep_split_in));

        std::vector<int> gpu_cc(gpus.size(), 0);
        std::vector<size_t> gpu_vf(gpus.size(), 0), gpu_vt(gpus.size(), 0);
        for (size_t gi = 0; gi < gpus.size(); ++gi) {
            INK_CK(cudaSetDevice(gpus[gi]));
            cudaDeviceProp prop{};
            INK_CK(cudaGetDeviceProperties(&prop, gpus[gi]));
            gpu_cc[gi] = prop.major * 10 + prop.minor;
            INK_CK(ink_cuda_mem_info(&gpu_vf[gi], &gpu_vt[gi], gpus[gi]));
        }
        INK_CK(cudaSetDevice(primary_gpu));
        sm_cc = gpu_cc[0];
        stage_cc[0] = sm_cc;
        stage_trunk_attn_fp16[0] = (sm_cc >= 70 && sm_cc < 80);

        // Choose Multi-GPU mode: Pipeline Layer-Split (`cuda-ls`) vs Dual-GPU EP (`cuda-ep2`)!
        // Auto-select Layer-Split on: explicit --layer-split, 3-4 GPUs, mixed arch, unequal VRAM (>15%),
        // Windows WDDM (avoids 48x/tok cross-GPU EP sync stalls & #6592), or non-P2P Linux rigs (Report C 2x 3090 EPYC!).
        if (gpus.size() >= 2) {
            const bool force_ep2 = (layer_split_spec == "off" || layer_split_spec == "none" || layer_split_spec == "ep2");
            bool mixed_cc = false;
            size_t min_vt = gpu_vt[0], max_vt = gpu_vt[0];
            for (size_t gi = 1; gi < gpus.size(); ++gi) {
                if (gpu_cc[gi] != gpu_cc[0]) mixed_cc = true;
                min_vt = std::min(min_vt, gpu_vt[gi]);
                max_vt = std::max(max_vt, gpu_vt[gi]);
            }
            const bool unequal_vram = (min_vt > 0) && ((double)max_vt > (double)min_vt * 1.15);
#ifdef _WIN32
            const bool prefer_ls_os = true;
#else
            const bool prefer_ls_os = false;
#endif
            const bool explicit_ls = !layer_split_spec.empty() && !force_ep2;
            bool p2p01 = false;
            if (!force_ep2 && (explicit_ls || gpus.size() > 2 || mixed_cc || unequal_vram || prefer_ls_os)) {
                ls_active = true;
            } else if (gpus.size() == 2) {
                p2p01 = probe_p2p(gpus[0], gpus[1]);
                INK_CK(cudaSetDevice(primary_gpu));
                if (!force_ep2 && !p2p01) {
                    ls_active = true;
                    std::fprintf(stderr, "[InkCUDA] Multi-GPU non-P2P detected: auto-selecting Pipeline Layer-Split (--layer-split auto, 6 handoffs/tok vs 48x EP sync)\n");
                } else {
                    ep2_active = true;
                    ep2_p2p = p2p01;
                    peer_gpu = gpus[1];
                }
            } else {
                ls_active = true;
            }
        }
        if (ls_active) {
            n_stages = (int)gpus.size();
            for (int s = 0; s < n_stages; ++s) {
                stage_gpu[s] = gpus[s];
                stage_cc[s] = gpu_cc[s];
                stage_trunk_attn_fp16[s] = (stage_cc[s] >= 70 && stage_cc[s] < 80);
                for (int t = 0; t < n_stages; ++t) stage_p2p[s][t] = (s == t);
            }
            for (int a = 0; a < n_stages; ++a) {
                for (int b = a + 1; b < n_stages; ++b) {
                    const bool ok = probe_p2p(stage_gpu[a], stage_gpu[b]);
                    stage_p2p[a][b] = ok;
                    stage_p2p[b][a] = ok;
                }
            }
            INK_CK(cudaSetDevice(primary_gpu));
        }

        // 1. Populate InkManifest directly from unified InkModelFile
        man = InkManifest{};
        man.build = mf.m.build;
        man.c.n_phys         = mf.m.c.n_phys;
        man.c.n_visits       = mf.m.c.n_visits;
        man.c.hidden         = mf.m.c.hidden;
        man.c.inter          = mf.m.c.inter;
        man.c.vocab_padded   = mf.m.c.vocab_padded;
        man.c.vocab_unpadded = mf.m.c.vocab_unpadded;
        man.c.n_head         = mf.m.c.n_head;
        man.c.n_kv           = mf.m.c.n_kv;
        man.c.head_dim       = mf.m.c.head_dim;
        man.c.d_rel          = mf.m.c.d_rel;
        man.c.rel_extent     = mf.m.c.rel_extent;
        man.c.swa_win        = mf.m.c.swa_win;
        man.c.block_size     = mf.m.c.block_size;
        man.c.n_experts      = mf.m.c.n_experts;
        man.c.top_k          = mf.m.c.top_k;
        man.c.core_start     = mf.m.c.core_start;
        man.c.core_end       = mf.m.c.core_end;
        man.c.num_passes     = mf.m.c.num_passes;
        man.c.gate_rank      = mf.m.c.gate_rank;
        man.c.eos_id         = mf.m.c.eos_token_id;
        man.c.rms_eps        = mf.m.c.rms_eps;
        man.c.qk_clip        = mf.m.c.qk_l2_clip;
        man.c.rel_clip       = mf.m.c.rel_l2_clip;
        man.c.log_alpha      = mf.m.c.log_scale_alpha;
        man.c.log_floor      = mf.m.c.log_scale_floor;
        man.c.route_scale    = mf.m.c.route_scale;
        man.c.situ_beta      = mf.m.c.situ_beta;
        man.c.situ_lb        = mf.m.c.situ_lb;
        man.c.logit_softcap  = mf.m.c.logit_softcap;
        man.c.is_swa.assign(mf.m.c.is_swa.begin(), mf.m.c.is_swa.end());
        man.has_dspark       = mf.m.dspark.present;
        for (const auto& kv : mf.m.trunk) {
            TensorInfo ti;
            ti.name = kv.second.name;
            ti.fmt = kv.second.fmt;
            ti.shape = kv.second.shape;
            ti.off = kv.second.off;
            ti.bytes = kv.second.bytes;
            man.trunk[kv.first] = std::move(ti);
        }
        for (const auto& kv : mf.m.dspark_tensors) {
            TensorInfo ti;
            ti.name = kv.second.name;
            ti.fmt = kv.second.fmt;
            ti.shape = kv.second.shape;
            ti.off = kv.second.off;
            ti.bytes = kv.second.bytes;
            man.dspark[kv.first] = std::move(ti);
        }

        // 2. Read Section 1 trunk + DSpark & upload to GPU 0
        double t0 = now_s();
        std::vector<uint8_t> sec1;
        if (!mf.read_section1(sec1, err)) return false;
        trunk_bytes = sec1.size();
        INK_CK(cudaMalloc(&d_trunk, std::max<size_t>(256, trunk_bytes)));
        if (trunk_bytes > 0) INK_CK(cudaMemcpy(d_trunk, sec1.data(), trunk_bytes, cudaMemcpyHostToDevice));
        double t_trunk = now_s() - t0;

        trunk_attn_fp16 = stage_trunk_attn_fp16[0];
        if (trunk_attn_fp16) {
            for (const auto& kv : man.trunk) {
                if (kv.second.fmt == FMT_BF16 && kv.first.find(".attn.") != std::string::npos) {
                    const size_t nw = (size_t)kv.second.rows() * kv.second.cols();
                    __nv_bfloat16* ptr = reinterpret_cast<__nv_bfloat16*>(d_trunk + kv.second.off);
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256>>>(ptr, reinterpret_cast<__half*>(ptr), nw);
                }
            }
            INK_CK(cudaDeviceSynchronize());
        }

        // 3. Prepare all 3072 experts (native SF4/MX8/S_K/GGUF_RAW, auto-repacking exotic GGUF quants -> SF4 in parallel!)
        double t1 = now_s();
        const auto& dir = mf.expert_directory();
        const int n_total = (int)dir.size();
        const int D = man.c.hidden, F = man.c.inter;
        const size_t sf4_dn_off_const = 16u + (size_t)(2 * F) * (D / 2) + (size_t)(2 * F) * (D / 16);
        const size_t sf4_body_const   = sf4_dn_off_const + (size_t)D * (F / 2) + (size_t)D * (F / 16);

        // Estimate expert bytes & check which GGUF_RAW experts need SF4 repack (exotic 1-2 bit quants)
        std::vector<uint8_t> need_sf4_repack(n_total, 0);
        std::vector<size_t> eff_bytes(n_total, 0);
        for (int di = 0; di < n_total; ++di) {
            const auto& e = dir[di];
            if (e.gu_code_id == FMT_GGUF_RAW) {
                std::vector<uint8_t> tmp;
                if (di % man.c.n_experts == 0) {
                    if (!mf.read_expert_body(di, tmp, err)) return false;
                    InkGgufExpertHeader gh{};
                    std::memcpy(&gh, tmp.data(), sizeof(gh));
                    const bool gpu_ok = ink_gguf_has_gpu_kernel(gh.gu_type) && ink_gguf_has_gpu_kernel(gh.dn_type);
                    for (int ee = 0; ee < man.c.n_experts && (di + ee) < n_total; ++ee) {
                        need_sf4_repack[di + ee] = gpu_ok ? 0 : 1;
                    }
                }
            }
            eff_bytes[di] = need_sf4_repack[di] ? sf4_body_const : (size_t)e.body_bytes;
        }

        // If Layer-Split is active, partition 24 physical layers across stages & upload compact per-stage trunks!
        if (ls_active) {
            std::vector<size_t> layer_exp_b(man.c.n_phys, 0);
            for (int il = 0; il < man.c.n_phys; ++il) {
                for (int r = 0; r < man.c.n_experts; ++r) layer_exp_b[il] += eff_bytes[(size_t)il * man.c.n_experts + r];
            }
            // Refresh free VRAM on GPU0 after trunk upload
            INK_CK(ink_cuda_mem_info(&gpu_vf[0], &gpu_vt[0], stage_gpu[0]));

            std::vector<int> counts;
            if (!layer_split_spec.empty() && layer_split_spec != "auto") {
                size_t ppos = 0;
                while (ppos < layer_split_spec.size()) {
                    size_t comma = layer_split_spec.find(',', ppos);
                    std::string tok = (comma == std::string::npos) ? layer_split_spec.substr(ppos) : layer_split_spec.substr(ppos, comma - ppos);
                    int v = std::atoi(tok.c_str());
                    if (v > 0) counts.push_back(v);
                    if (comma == std::string::npos) break;
                    ppos = comma + 1;
                }
            }
            if ((int)counts.size() == n_stages) {
                int sum = 0;
                for (int v : counts) sum += v;
                if (sum != man.c.n_phys) {
                    // Scale proportionally to n_phys
                    int acc = 0;
                    for (int s = 0; s < n_stages - 1; ++s) {
                        counts[s] = std::max(1, (int)std::lround((double)counts[s] * man.c.n_phys / std::max(1, sum)));
                        acc += counts[s];
                    }
                    counts[n_stages - 1] = std::max(1, man.c.n_phys - acc);
                }
            } else {
                // Automatic VRAM-weighted partition (ensuring peer stages s > 0 fit 100% in VRAM!)
                counts.assign(n_stages, 0);
                std::vector<double> usable(n_stages, 0.0);
                double tot_u = 0.0;
                for (int s = 0; s < n_stages; ++s) {
                    const size_t wf = wddm_safety_floor(gpu_vt[s]);
                    const size_t stage_oh = wf + ((s == 0) ? std::max(extra_reserve_bytes, (size_t)(1280ull << 20)) : (size_t)(1024ull << 20));
                    usable[s] = (gpu_vf[s] > stage_oh) ? (double)(gpu_vf[s] - stage_oh) : 0.0;
                    tot_u += usable[s];
                }
                if (tot_u <= 1.0) {
                    const int base = man.c.n_phys / n_stages;
                    for (int s = 0; s < n_stages; ++s) counts[s] = base;
                    counts[0] += man.c.n_phys - base * n_stages;
                } else {
                    int rem = man.c.n_phys;
                    for (int s = n_stages - 1; s >= 1; --s) {
                        int want = std::max(1, (int)std::lround((double)man.c.n_phys * (usable[s] / tot_u)));
                        want = std::min(want, rem - s);
                        // Clamp stage s > 0 to layers that fit 100% in stage_gpu[s] VRAM (leaving stage 0 to handle any hybrid overflow!)
                        size_t acc_b = 0;
                        int fit = 0;
                        const size_t max_b = (size_t)usable[s];
                        for (int k = 0; k < want; ++k) {
                            const int il = rem - 1 - k;
                            if (acc_b + layer_exp_b[il] + (48ull << 20) <= max_b) {
                                acc_b += layer_exp_b[il] + (48ull << 20);
                                ++fit;
                            } else break;
                        }
                        counts[s] = fit;
                        rem -= fit;
                    }
                    counts[0] = rem;
                }
            }
            // Drop any empty peer stages if a secondary GPU had zero usable VRAM
            std::vector<int> active_stages;
            std::vector<int> active_counts;
            for (int s = 0; s < n_stages; ++s) {
                if (s == 0 || counts[s] > 0) {
                    active_stages.push_back(stage_gpu[s]);
                    active_counts.push_back(counts[s]);
                }
            }
            if (active_stages.size() < 2) {
                ls_active = false;
                n_stages = 1;
                for (int il = 0; il < 24; ++il) layer_owner[il] = 0;
            } else {
                n_stages = (int)active_stages.size();
                int cur_l = 0;
                for (int s = 0; s < n_stages; ++s) {
                    stage_gpu[s] = active_stages[s];
                    cudaDeviceProp sp{};
                    INK_CK(cudaGetDeviceProperties(&sp, stage_gpu[s]));
                    stage_cc[s] = sp.major * 10 + sp.minor;
                    stage_trunk_attn_fp16[s] = (stage_cc[s] >= 70 && stage_cc[s] < 80);
                    stage_l0[s] = cur_l;
                    stage_l1[s] = cur_l + active_counts[s];
                    for (int il = stage_l0[s]; il < stage_l1[s]; ++il) layer_owner[il] = s;
                    cur_l = stage_l1[s];
                }
                // Pack ONLY stage-owned trunk tensors onto each peer GPU s > 0 (saves ~1.5-2 GiB per peer GPU!)
                for (int s = 1; s < n_stages; ++s) {
                    stage_trunk_off[s].clear();
                    std::vector<std::pair<std::string, TensorInfo>> owned;
                    size_t pack_bytes = 0;
                    for (const auto& kv : man.trunk) {
                        const std::string& nm = kv.first;
                        bool own = false;
                        for (int il = stage_l0[s]; il < stage_l1[s]; ++il) {
                            const std::string lp = "model.llm.layers." + std::to_string(il) + ".";
                            if (nm.rfind(lp, 0) == 0 ||
                                nm == ("model.llm.attn_res.mlp_queries." + std::to_string(il)) ||
                                (il > 0 && nm == ("model.llm.attn_res.attn_queries." + std::to_string(il - 1)))) {
                                own = true;
                                break;
                            }
                        }
                        if (own) {
                            pack_bytes = (pack_bytes + 255ull) & ~255ull;
                            stage_trunk_off[s][nm] = pack_bytes;
                            owned.push_back({nm, kv.second});
                            pack_bytes += kv.second.bytes;
                        }
                    }
                    std::vector<uint8_t> hpack(std::max<size_t>(256, pack_bytes), 0);
                    for (const auto& item : owned) {
                        const size_t dst_o = stage_trunk_off[s][item.first];
                        if (item.second.off + item.second.bytes <= sec1.size()) {
                            std::memcpy(hpack.data() + dst_o, sec1.data() + item.second.off, item.second.bytes);
                        }
                    }
                    INK_CK(cudaSetDevice(stage_gpu[s]));
                    INK_CK(cudaMalloc(&stage_d_trunk[s], hpack.size()));
                    INK_CK(cudaMemcpy(stage_d_trunk[s], hpack.data(), hpack.size(), cudaMemcpyHostToDevice));
                    if (stage_trunk_attn_fp16[s]) {
                        for (const auto& item : owned) {
                            if (item.second.fmt == FMT_BF16 && item.first.find(".attn.") != std::string::npos) {
                                const size_t nw = (size_t)item.second.rows() * item.second.cols();
                                __nv_bfloat16* ptr = reinterpret_cast<__nv_bfloat16*>(stage_d_trunk[s] + stage_trunk_off[s][item.first]);
                                k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256>>>(ptr, reinterpret_cast<__half*>(ptr), nw);
                            }
                        }
                        INK_CK(cudaDeviceSynchronize());
                    }
                }
                INK_CK(cudaSetDevice(primary_gpu));
            }
        }
        sec1.clear();
        sec1.shrink_to_fit();

        // Determine per-expert target tier: 0=stage/GPU0 VRAM, 1=GPU1 (ep2), 2=Host RAM hybrid, 3=SSD-only (.strata)
        size_t vf0 = 0, vt0 = 0, vf1 = 0, vt1 = 0;
        INK_CK(ink_cuda_mem_info(&vf0, &vt0, primary_gpu));
        const size_t wddm_floor0 = wddm_safety_floor(vt0);
        const size_t base_res0 = ((vt0 >= (20ull << 30)) ? (1600ull << 20)
                                : (vt0 <= (13ull << 30)) ? (960ull << 20) : (1280ull << 20))
                               + (wddm_floor0 > (512ull << 20) ? (wddm_floor0 - (512ull << 20)) : 0ull);
        const size_t res0 = std::max(base_res0, extra_reserve_bytes + wddm_floor0);
        const size_t bud0 = (vf0 > res0) ? (vf0 - res0) : 0;
        size_t bud1 = 0;
        if (ep2_active) {
            INK_CK(cudaSetDevice(peer_gpu));
            INK_CK(ink_cuda_mem_info(&vf1, &vt1, peer_gpu));
            const size_t res1 = std::max((size_t)(768ull << 20), wddm_safety_floor(vt1));
            bud1 = (vf1 > res1) ? (vf1 - res1) : 0;
            INK_CK(cudaSetDevice(primary_gpu));
        }

        // Build 3x loop-weighted load order
        const auto& vorder = mf.vram_load_order;
        std::vector<uint16_t> order;
        if ((int)vorder.size() == n_total) {
            order = vorder;
        } else {
            order.reserve(n_total);
            for (int il = man.c.core_start; il < man.c.core_end; ++il)
                for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
            for (int il = 0; il < man.c.core_start; ++il)
                for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
            for (int il = man.c.core_end; il < man.c.n_phys; ++il)
                for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
        }

        float eff_split = ep_split;
        if (ep2_active && std::fabs(ep_split_in - 0.5f) < 1e-3f && vf1 > vf0 + (512ull << 20)) {
            const double ratio = (double)vf1 / std::max(1.0, (double)vf0 + (double)vf1);
            eff_split = (float)std::max(0.50, std::min(0.625, ratio));
        }
        std::vector<uint8_t> target_tier(n_total, 2);
        size_t u0 = 0, u1 = 0, uh = 0, max_exp_b = 0;
        size_t stage_u[kMaxStages] = {};
        size_t stage_bud[kMaxStages] = {};
        const size_t user_vram_cap = (vram_experts_gb >= 0.0)
            ? static_cast<size_t>(vram_experts_gb * 1073741824.0)
            : static_cast<size_t>(-1);
        size_t max_body_all = 0;
        for (int di = 0; di < n_total; ++di) max_body_all = std::max<size_t>(max_body_all, eff_bytes[di]);
        const size_t pf_bytes = 2 * InkHybridEngine::pf_stage_size(max_body_all);
        const size_t hyb_post_res = std::max(res0, (size_t)(1664ull << 20));
        const size_t hyb_bud0 = (vf0 > hyb_post_res + pf_bytes + (64ull << 20)) ? (vf0 - hyb_post_res - pf_bytes - (64ull << 20)) : 0;

        n_vram_experts = 0; n_peer_experts = 0; n_host_experts = 0;
        for (int s = 0; s < kMaxStages; ++s) { stage_sec4_bytes[s] = 0; stage_n_experts[s] = 0; }

        if (ls_active) {
            for (int s = 0; s < n_stages; ++s) {
                size_t sf = 0, st = 0;
                INK_CK(ink_cuda_mem_info(&sf, &st, stage_gpu[s]));
                const size_t sres = (s == 0) ? res0 : std::max<size_t>((size_t)(896ull << 20), wddm_safety_floor(st) + (size_t)(512ull << 20));
                stage_bud[s] = (sf > sres) ? (sf - sres) : 0;
                if (user_vram_cap < stage_bud[s]) stage_bud[s] = user_vram_cap;
            }
            for (uint16_t di : order) {
                const int il = (int)di / man.c.n_experts;
                const int s = layer_owner[il];
                const size_t eb = eff_bytes[di];
                if (stage_u[s] + eb <= stage_bud[s]) {
                    target_tier[di] = 0;
                    stage_u[s] += eb;
                    stage_n_experts[s]++;
                    if (s == 0) ++n_vram_experts; else ++n_peer_experts;
                } else {
                    // Preserve lossless native FMT_GGUF_RAW blocks in host RAM (zero SF4 quality loss & zero repack startup delay!)
                    const size_t hb = (eff_bytes[di] + 63ull) & ~63ull;
                    target_tier[di] = 2; uh += hb; ++n_host_experts;
                    max_exp_b = std::max(max_exp_b, eff_bytes[di]);
                }
            }
            u0 = stage_u[0];
        } else {
            const int n_peer_per_layer = ep2_active ? std::max(1, std::min(man.c.n_experts - 1, (int)std::lround(man.c.n_experts * eff_split))) : 0;
            auto want_peer = [&](int exp_id) -> bool {
                if (!ep2_active) return false;
                if (n_peer_per_layer == man.c.n_experts / 2) return (exp_id & 1) == 1;
                return ((exp_id * n_peer_per_layer) % man.c.n_experts) < n_peer_per_layer;
            };
            size_t need0_ideal = 0, need1_ideal = 0;
            for (int di = 0; di < n_total; ++di) {
                if (want_peer(dir[di].expert_id)) need1_ideal += eff_bytes[di];
                else                              need0_ideal += eff_bytes[di];
            }
            const bool all_fit_gpu = !ssd_experts && (need0_ideal <= std::min(bud0, user_vram_cap) && need1_ideal <= std::min(bud1, user_vram_cap));
            const size_t cap0 = std::min(all_fit_gpu ? bud0 : hyb_bud0, user_vram_cap);
            if (user_vram_cap < bud1) bud1 = user_vram_cap;

            for (uint16_t di : order) {
                const size_t eb = eff_bytes[di];
                const bool pref_p = want_peer(dir[di].expert_id);
                if (pref_p && u1 + eb <= bud1) {
                    target_tier[di] = 1; u1 += eb; ++n_peer_experts;
                } else if (!pref_p && u0 + eb <= cap0) {
                    target_tier[di] = 0; u0 += eb; ++n_vram_experts;
                } else if (ep2_active && u1 + eb <= bud1) {
                    target_tier[di] = 1; u1 += eb; ++n_peer_experts;
                } else if (u0 + eb <= cap0) {
                    target_tier[di] = 0; u0 += eb; ++n_vram_experts;
                } else {
                    // Preserve lossless native FMT_GGUF_RAW blocks in host RAM (evaluated via strata::quant SIMD & GPU DP4A staging!)
                    const size_t hb = (eff_bytes[di] + 63ull) & ~63ull;
                    target_tier[di] = 2; uh += hb; ++n_host_experts;
                    max_exp_b = std::max(max_exp_b, eff_bytes[di]);
                }
            }
            if (ep2_active && n_host_experts > 0) {
                const int E = man.c.n_experts;
                for (int il = 0; il < man.c.n_phys; ++il) {
                    bool has1 = false, has2 = false;
                    for (int r = 0; r < E; ++r) {
                        const uint8_t t = target_tier[il * E + r];
                        if (t == 1) has1 = true;
                        else if (t == 2) has2 = true;
                    }
                    if (has1 && has2) {
                        for (int r = 0; r < E; ++r) {
                            const int di = il * E + r;
                            if (target_tier[di] == 1) {
                                u1 -= eff_bytes[di]; --n_peer_experts;
                                const size_t hb = (eff_bytes[di] + 63ull) & ~63ull;
                                target_tier[di] = 2; uh += hb; ++n_host_experts;
                                max_exp_b = std::max(max_exp_b, eff_bytes[di]);
                            }
                        }
                    }
                }
            }
        }

        // #6617: Allocate ping-pong prefill staging buffers BEFORE d_sec4 and host pinned arena!
        INK_CK(cudaSetDevice(primary_gpu));
        if (n_host_experts > 0) {
            hybrid.start_workers(cpu_threads);
            if (!hybrid.alloc_pf_stage(max_exp_b)) {
                err = "not enough free VRAM for the 2x prefill staging buffers required by host experts; close other GPU apps or use a smaller quant";
                return false;
            }
        }
        sec4_bytes = u0;
        INK_CK(cudaMalloc(&d_sec4, std::max<size_t>(256, sec4_bytes)));
        stage_d_sec4[0] = d_sec4;
        stage_sec4_bytes[0] = sec4_bytes;
        if (ls_active) {
            for (int s = 1; s < n_stages; ++s) {
                INK_CK(cudaSetDevice(stage_gpu[s]));
                stage_sec4_bytes[s] = stage_u[s];
                INK_CK(cudaMalloc(&stage_d_sec4[s], std::max<size_t>(256, stage_sec4_bytes[s])));
            }
            INK_CK(cudaSetDevice(primary_gpu));
        } else if (ep2_active && u1 > 0) {
            INK_CK(cudaSetDevice(peer_gpu));
            sec4_peer_bytes = u1;
            INK_CK(cudaMalloc(&d_sec4_peer, std::max<size_t>(256, sec4_peer_bytes)));
            INK_CK(cudaSetDevice(primary_gpu));
        }
        uint8_t* h_arena_base = nullptr;
        if (uh > 0) {
            const size_t ram_cap = safe_host_ram_cap(ram_experts_gb, ssd_experts);
            if (ram_cap < uh && !is_strata_fmt) {
                std::fprintf(stderr, "[InkCUDA] Note: GGUF/HF host spill %.2f GiB exceeds soft RAM cap %.2f GiB (allocating pageable host arena)\n",
                             (double)uh / 1073741824.0, (double)ram_cap / 1073741824.0);
            }
            h_arena_base = hybrid.alloc_host_arena(uh);
            if (!h_arena_base) { err = "cannot allocate host expert arena"; return false; }
            hybrid.host_meta.assign((size_t)man.c.n_phys * man.c.n_experts, {});
            hybrid.orig_host_off.assign((size_t)man.c.n_phys * man.c.n_experts, 0);
        }

        // Precompute exact destination offsets so we can upload/repack in parallel chunks!
        std::vector<size_t> dst_off(n_total, 0);
        {
            size_t c0 = 0, c1 = 0, ch = 0;
            size_t cs[kMaxStages] = {};
            for (int di = 0; di < n_total; ++di) {
                if (ls_active && target_tier[di] == 0) {
                    const int s = layer_owner[di / man.c.n_experts];
                    dst_off[di] = cs[s];
                    cs[s] += eff_bytes[di];
                } else if (target_tier[di] == 0) { dst_off[di] = c0; c0 += eff_bytes[di]; }
                else if (target_tier[di] == 1)   { dst_off[di] = c1; c1 += eff_bytes[di]; }
                else                             { dst_off[di] = ch; ch += (eff_bytes[di] + 63ull) & ~63ull; }
            }
        }

        std::vector<InkExpertDev> h_exp0(n_total), h_exp1(n_total);
        for (int il = 0; il < 24; ++il) {
            moe_info[il] = {}; moe_info_peer[il] = {};
            layer_has_host[il] = false; layer_has_peer[il] = false;
        }

        // Process layer-by-layer: sequential file read, then multi-threaded CPU repack for any exotic GGUF->SF4 experts!
        for (int il = 0; il < man.c.n_phys; ++il) {
            const int E = man.c.n_experts;
            std::vector<std::vector<uint8_t>> layer_bodies(E);
            std::vector<uint32_t> layer_dn_off(E, 0);
            std::vector<uint8_t> layer_gu_code(E, 0), layer_dn_code(E, 0);
            bool any_repack = false;
            for (int r = 0; r < E; ++r) {
                const int di = il * E + r;
                if (need_sf4_repack[di]) any_repack = true;
            }
            if (any_repack) {
                std::vector<std::vector<uint8_t>> raw_bodies(E);
                for (int r = 0; r < E; ++r) {
                    const int di = il * E + r;
                    if (!mf.read_expert_body(di, raw_bodies[r], err)) return false;
                }
                std::atomic<int> next_r{0};
                std::atomic<bool> par_ok{true};
                std::mutex err_mu;
                std::string par_err;
                const unsigned n_thr = std::max(1u, std::min<unsigned>(16u, std::thread::hardware_concurrency()));
                std::vector<std::thread> workers;
                workers.reserve(n_thr);
                for (unsigned t = 0; t < n_thr; ++t) {
                    workers.emplace_back([&]() {
                        while (true) {
                            const int r = next_r.fetch_add(1);
                            if (r >= E || !par_ok.load(std::memory_order_relaxed)) break;
                            const int di = il * E + r;
                            const auto& e = dir[di];
                            if (need_sf4_repack[di]) {
                                uint32_t new_dn = 0;
                                std::string lerr;
                                if (!repack_gguf_raw_body_to_sf4(raw_bodies[r].data(), e.dn_offset, D, F, dequant_row, row_bytes, layer_bodies[r], new_dn, lerr)) {
                                    std::lock_guard<std::mutex> lk(err_mu);
                                    par_ok.store(false);
                                    if (par_err.empty()) par_err = lerr;
                                    break;
                                }
                                raw_bodies[r].clear();
                                raw_bodies[r].shrink_to_fit();
                                layer_dn_off[r] = new_dn;
                                layer_gu_code[r] = FMT_SF4;
                                layer_dn_code[r] = FMT_SF4;
                            } else {
                                layer_bodies[r] = std::move(raw_bodies[r]);
                                layer_dn_off[r] = e.dn_offset;
                                layer_gu_code[r] = e.gu_code_id;
                                layer_dn_code[r] = e.dn_code_id;
                            }
                        }
                    });
                }
                for (auto& w : workers) w.join();
                if (!par_ok.load()) { err = par_err; return false; }
            } else {
                for (int r = 0; r < E; ++r) {
                    const int di = il * E + r;
                    const auto& e = dir[di];
                    if (!mf.read_expert_body(di, layer_bodies[r], err)) return false;
                    layer_dn_off[r] = e.dn_offset;
                    layer_gu_code[r] = e.gu_code_id;
                    layer_dn_code[r] = e.dn_code_id;
                }
            }

            // Upload layer's experts to owning GPU stage / EP2 / Host hybrid (updating PR#6593 sk_gu_mask / sk_dn_mask!)
            auto mark_info = [](InkLayerMoeInfo& mi, uint8_t gu_code, uint8_t dn_code) {
                if (gu_code == FMT_SF4)           mi.has_sf4  = true;
                else if (gu_code == FMT_MX8)      mi.has_mx8  = true;
                else if (gu_code == FMT_GGUF_RAW) mi.has_gguf = true;
                else {
                    mi.has_sk = true;
                    if (gu_code < 32) mi.sk_gu_mask |= (1u << gu_code);
                }
                if (dn_code != FMT_SF4 && dn_code != FMT_MX8 && dn_code != FMT_GGUF_RAW && dn_code < 32) {
                    mi.has_sk = true;
                    mi.sk_dn_mask |= (1u << dn_code);
                }
            };
            const int own_s = ls_active ? layer_owner[il] : 0;
            const int own_gpu = ls_active ? stage_gpu[own_s] : primary_gpu;
            uint8_t* own_sec4 = ls_active ? stage_d_sec4[own_s] : d_sec4;
            for (int r = 0; r < E; ++r) {
                const int di = il * E + r;
                const auto& e = dir[di];
                const int key = il * E + e.expert_id;
                const auto& body = layer_bodies[r];
                InkExpertDev d0{}, d1{};
                d0.dn_off = d1.dn_off = layer_dn_off[r];
                d0.gu_code = d1.gu_code = layer_gu_code[r];
                d0.dn_code = d1.dn_code = layer_dn_code[r];
                if (target_tier[di] == 0) {
                    INK_CK(cudaSetDevice(own_gpu));
                    INK_CK(cudaMemcpy(own_sec4 + dst_off[di], body.data(), body.size(), cudaMemcpyHostToDevice));
                    d0.body_off = dst_off[di]; d0.tier = 0;
                    d1.body_off = 0;           d1.tier = 3;
                    mark_info(moe_info[il], layer_gu_code[r], layer_dn_code[r]);
                } else if (target_tier[di] == 1) {
                    INK_CK(cudaSetDevice(peer_gpu));
                    INK_CK(cudaMemcpy(d_sec4_peer + dst_off[di], body.data(), body.size(), cudaMemcpyHostToDevice));
                    d0.body_off = 0;           d0.tier = 3;
                    d1.body_off = dst_off[di]; d1.tier = 0;
                    layer_has_peer[il] = true;
                    mark_info(moe_info_peer[il], layer_gu_code[r], layer_dn_code[r]);
                } else {
                    uint8_t* hdst = h_arena_base + dst_off[di];
                    std::memcpy(hdst, body.data(), body.size());
                    hybrid.host_meta[key].body       = hdst;
                    hybrid.host_meta[key].body_bytes = (uint32_t)body.size();
                    hybrid.host_meta[key].dn_off     = layer_dn_off[r];
                    hybrid.host_meta[key].gu_code    = layer_gu_code[r];
                    hybrid.host_meta[key].dn_code    = layer_dn_code[r];
                    hybrid.orig_host_off[key]        = dst_off[di];
                    d0.body_off = dst_off[di]; d0.tier = 2;
                    d1.body_off = 0;           d1.tier = 3;
                    layer_has_host[il] = true;
                    mark_info(moe_info[il], layer_gu_code[r], layer_dn_code[r]);
                }
                h_exp0[key] = d0;
                h_exp1[key] = d1;
            }
        }
        INK_CK(cudaSetDevice(primary_gpu));
        if (n_host_experts > 0) {
            hybrid.start_workers(cpu_threads);
            size_t vf_now = 0, vt_now = 0;
            INK_CK(ink_cuda_mem_info(&vf_now, &vt_now, primary_gpu));
            const size_t avail = (vf_now > hyb_post_res) ? (vf_now - hyb_post_res) : 0;
            hybrid.alloc_staging(max_exp_b, avail);
        }
        double t_exp = now_s() - t1;

        for (int il = 0; il < man.c.n_phys; ++il) {
            const int own_gpu = ls_active ? stage_gpu[layer_owner[il]] : primary_gpu;
            INK_CK(cudaSetDevice(own_gpu));
            d_experts[il] = dalloc<InkExpertDev>(man.c.n_experts);
            INK_CK(cudaMemcpy(d_experts[il], h_exp0.data() + (size_t)il * man.c.n_experts,
                              (size_t)man.c.n_experts * sizeof(InkExpertDev), cudaMemcpyHostToDevice));
            if (ep2_active) {
                INK_CK(cudaSetDevice(peer_gpu));
                d_experts_peer[il] = dalloc<InkExpertDev>(man.c.n_experts);
                INK_CK(cudaMemcpy(d_experts_peer[il], h_exp1.data() + (size_t)il * man.c.n_experts,
                                  (size_t)man.c.n_experts * sizeof(InkExpertDev), cudaMemcpyHostToDevice));
            }
        }
        INK_CK(cudaSetDevice(primary_gpu));

        // 4. Preprocess msg_queries[6..17] and static_sconv[0..23] (placed on each layer's owning stage GPU!)
        {
            const auto& ti_q  = man.get("model.llm.middle_cycle_cross_pass_queries");
            const auto& ti_kv = man.get("model.llm.middle_cycle_cross_pass_kv_queries");
            std::vector<float> h_q(24 * kInkHidden), h_kv(24 * kInkHidden);
            INK_CK(cudaMemcpy(h_q.data(), d_trunk + ti_q.off, h_q.size() * 4, cudaMemcpyDeviceToHost));
            INK_CK(cudaMemcpy(h_kv.data(), d_trunk + ti_kv.off, h_kv.size() * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < 12; ++i) {
                std::vector<float> mq(4 * kInkHidden);
                std::memcpy(mq.data() + 0 * kInkHidden, h_q.data()  + (2 * i)     * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 1 * kInkHidden, h_kv.data() + (2 * i)     * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 2 * kInkHidden, h_kv.data() + (2 * i + 1) * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 3 * kInkHidden, h_q.data()  + (2 * i + 1) * kInkHidden, kInkHidden * 4);
                const int il = man.c.core_start + i;
                const int own_gpu = ls_active ? stage_gpu[layer_owner[il]] : primary_gpu;
                INK_CK(cudaSetDevice(own_gpu));
                d_msg_queries[i] = dalloc<float>(4 * kInkHidden);
                INK_CK(cudaMemcpy(d_msg_queries[i], mq.data(), mq.size() * 4, cudaMemcpyHostToDevice));
            }
            INK_CK(cudaSetDevice(primary_gpu));
        }
        for (int il = 0; il < man.c.n_phys; ++il) {
            const std::string p = "model.llm.layers." + std::to_string(il);
            const int own_s = ls_active ? layer_owner[il] : 0;
            const int own_gpu = ls_active ? stage_gpu[own_s] : primary_gpu;
            d_sconv_attn[il] = (float*)trunk_ptr_stage(own_s, p + ".attn_sconv.weight");
            d_sconv_mlp[il]  = (float*)trunk_ptr_stage(own_s, p + ".mlp_sconv.weight");
            if (man.c.is_swa[il]) {
                d_sconv_k[il] = (float*)trunk_ptr_stage(own_s, p + ".attn.k_sconv.weight");
                d_sconv_v[il] = (float*)trunk_ptr_stage(own_s, p + ".attn.v_sconv.weight");
            } else {
                auto flip_t = [&](const std::string& tname) -> float* {
                    std::vector<float> src(4 * kInkKVDim), dst(kInkKVDim * 4);
                    INK_CK(cudaSetDevice(primary_gpu));
                    INK_CK(cudaMemcpy(src.data(), d_trunk + man.get(tname).off, src.size() * 4, cudaMemcpyDeviceToHost));
                    for (int c = 0; c < kInkKVDim; ++c)
                        for (int l = 0; l < 4; ++l) dst[c * 4 + l] = src[(3 - l) * kInkKVDim + c];
                    INK_CK(cudaSetDevice(own_gpu));
                    float* dptr = dalloc<float>(dst.size());
                    INK_CK(cudaMemcpy(dptr, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice));
                    return dptr;
                };
                d_sconv_k[il] = flip_t(p + ".attn.k_sconv.static_w");
                d_sconv_v[il] = flip_t(p + ".attn.v_sconv.static_w");
            }
        }
        INK_CK(cudaSetDevice(primary_gpu));

        // 5. Optional unembed screening index
        const auto& ui = man.get("model.llm.unembed.weight");
        if (build_screen && (ui.fmt == FMT_BF16 || ui.fmt == FMT_S6_K)) {
            INK_CK(ink_cuda_mem_info(&vf0, &vt0, primary_gpu));
            if (vf0 > (2048ull << 20)) {
                unembed_screen.V = man.c.vocab_unpadded;
                unembed_screen.q8  = dalloc<int8_t>((size_t)unembed_screen.V * kInkHidden);
                unembed_screen.sc  = dalloc<float>(unembed_screen.V);
                unembed_screen.eps = dalloc<float>(unembed_screen.V);
                if (ui.fmt == FMT_BF16) {
                    k_ink_build_unembed_screen_bf16<<<(unembed_screen.V + 7) / 8, 256>>>(
                        reinterpret_cast<const uint16_t*>(d_trunk + ui.off),
                        unembed_screen.q8, unembed_screen.sc, unembed_screen.eps, unembed_screen.V);
                } else {
                    k_ink_build_unembed_screen_s6k<<<(unembed_screen.V + 7) / 8, 256>>>(
                        d_trunk + ui.off,
                        unembed_screen.q8, unembed_screen.sc, unembed_screen.eps, unembed_screen.V);
                }
                INK_CK(cudaDeviceSynchronize());
                unembed_screen.ready = true;
            }
        }

        if (ls_active) {
            std::fprintf(stderr, "[InkCUDA] Loaded %s (%s, fmt=%s) [cuda-ls %d stages]: trunk=%.2f GB (%.2fs), experts=%zu+%zu GPU / %zu host (%.2fs, hyb=%d):",
                         path.c_str(), man.build.c_str(), mf.m.format.c_str(), n_stages, (double)trunk_bytes / 1e9, t_trunk,
                         n_vram_experts, n_peer_experts, n_host_experts, t_exp, (int)hybrid.active);
            for (int s = 0; s < n_stages; ++s) {
                std::fprintf(stderr, " GPU%d(L%d..%d,%zuexp)", stage_gpu[s], stage_l0[s], stage_l1[s] - 1, stage_n_experts[s]);
            }
            std::fprintf(stderr, "\n");
        } else {
            std::fprintf(stderr, "[InkCUDA] Loaded %s (%s, fmt=%s): trunk=%.2f GB (%.2fs), experts=%zu GPU%d + %zu GPU%d / %zu host (%.2fs, ep2=%d, p2p=%d, hyb=%d), sm120_tc=%d\n",
                         path.c_str(), man.build.c_str(), mf.m.format.c_str(), (double)trunk_bytes / 1e9, t_trunk,
                         n_vram_experts, primary_gpu, n_peer_experts, peer_gpu, n_host_experts, t_exp,
                         (int)ep2_active, (int)ep2_p2p, (int)hybrid.active, (int)sm120::available());
        }
        return true;
    }

    void free_all() {
        if (ep2_active && peer_gpu >= 0) {
            cudaSetDevice(peer_gpu);
            for (int il = 0; il < 24; ++il) {
                if (d_experts_peer[il]) { cudaFree(d_experts_peer[il]); d_experts_peer[il] = nullptr; }
            }
            if (d_sec4_peer) { cudaFree(d_sec4_peer); d_sec4_peer = nullptr; }
            cudaSetDevice(primary_gpu);
        }
        ep2_active = false; ep2_p2p = false; peer_gpu = -1;
        n_peer_experts = 0; sec4_peer_bytes = 0;
        hybrid.free_all();
        unembed_screen.free_all();
        for (int i = 0; i < 12; ++i) {
            if (d_msg_queries[i]) {
                const int il = man.c.core_start + i;
                const int dev = (ls_active && il >= 0 && il < 24) ? stage_gpu[layer_owner[il]] : primary_gpu;
                cudaSetDevice(dev);
                cudaFree(d_msg_queries[i]);
                d_msg_queries[i] = nullptr;
            }
        }
        for (int il = 0; il < 24; ++il) {
            const int dev = ls_active ? stage_gpu[layer_owner[il]] : primary_gpu;
            cudaSetDevice(dev);
            if (d_experts[il]) { cudaFree(d_experts[il]); d_experts[il] = nullptr; }
            if (!man.c.is_swa.empty() && !man.c.is_swa[il]) {
                if (d_sconv_k[il]) { cudaFree(d_sconv_k[il]); d_sconv_k[il] = nullptr; }
                if (d_sconv_v[il]) { cudaFree(d_sconv_v[il]); d_sconv_v[il] = nullptr; }
            }
        }
        if (ls_active) {
            for (int s = 1; s < n_stages; ++s) {
                cudaSetDevice(stage_gpu[s]);
                if (stage_d_sec4[s])  { cudaFree(stage_d_sec4[s]); stage_d_sec4[s] = nullptr; }
                if (stage_d_trunk[s]) { cudaFree(stage_d_trunk[s]); stage_d_trunk[s] = nullptr; }
                stage_trunk_off[s].clear();
            }
        }
        cudaSetDevice(primary_gpu);
        stage_d_sec4[0] = nullptr;
        ls_active = false; n_stages = 1;
        if (d_sec4) { cudaFree(d_sec4); d_sec4 = nullptr; }
        if (h_sec4_pinned) { cudaFreeHost(h_sec4_pinned); h_sec4_pinned = nullptr; }
        if (d_trunk) { cudaFree(d_trunk); d_trunk = nullptr; }
    }
};

// ============================================================================
// InkCudaRunner: Full 48-Visit Forward, Continuous Batching (B=1..64), Tree-16, CUDA Graphs!
// ============================================================================
struct InkCudaRunner {
    InkCudaModel& m;
    cudaStream_t st = nullptr;
    cublasHandle_t cublas = nullptr;

    int max_slots = 1;
    int max_ctx = 4096;
    int kv_alloc_ctx = 0;         // Elastic global KV current allocated capacity (dev_ctx + host_ctx <= max_ctx!)
    int kv_dev_ctx = 0;           // Global KV positions resident in VRAM
    int kv_host_ctx = 0;          // Global KV positions offloaded to mapped pinned host RAM
    int kv_offload_mode = 0;      // 0 = auto, 1 = on (100% host global KV), -1 = off (VRAM-only)
    int max_batch = 512;
    int swa_ring = 1024 + 512;
    int tree_cap = 16;
    int max_passes_override = 0;  // 0 = full 3 passes (48 visits); 1 or 2 = early-pass exit probe!
    InkKvDtype kv_dtype = InkKvDtype::FP16;

    // KV Cache (48 visits: 40 SWA in VRAM, 8 Global split across VRAM + zero-copy mapped pinned host RAM!)
    InkKvVisitDesc kv_desc[kInkVisits]{};
    void* host_kv_raw_k[kInkVisits]   = {};
    void* host_kv_raw_v[kInkVisits]   = {};
    void* host_kv_raw_ksc[kInkVisits] = {};
    void* host_kv_raw_vsc[kInkVisits] = {};
    // ShortConv state pool: [48 visits] x 4 states (k [1024], v [1024], attn [2304], mlp [2304]) -> 1 contiguous buffer!
    // Layout: visit -> [max_slots, 3, 6656] F32 (or 256 MB for 64 slots)
    float* d_sconv_pool[kInkVisits] = {};

    // Activation workspaces
    float *d_cur = nullptr, *d_res = nullptr, *d_prelude = nullptr;
    float *d_blocks[8] = {};          // attention-residual block bank (max 6 needed)
    float *d_prelude_blocks[2] = {};  // saved prelude block snapshots
    float *d_bank[4] = {};            // saved previous-pass core blocks
    float *d_bank_out = nullptr;
    float *d_mixed = nullptr;         // [4, max_batch, 2304]
    float *d_msgs = nullptr;          // [4, max_batch, 2304]
    float *d_inp0 = nullptr, *d_inp1 = nullptr, *d_inp2 = nullptr;
    float *d_q = nullptr, *d_k_raw = nullptr, *d_v_raw = nullptr, *d_k = nullptr, *d_v = nullptr, *d_r = nullptr;
    float *d_k_dyn = nullptr, *d_v_dyn = nullptr;
    float *d_sinks = nullptr, *d_gate_attn = nullptr, *d_rel_table = nullptr;
    float *d_attn_out = nullptr, *d_wo_raw = nullptr, *d_wo_conv = nullptr;
    float *d_moe_in = nullptr, *d_moe_raw = nullptr, *d_moe_conv = nullptr;
    float *d_h_norm = nullptr;        // [max_batch, 2304] final output_norm hidden
    float *d_taps[4] = {};            // [max_batch, 2304] taps at visits [5, 23, 41, 48]
    float *d_taps_compact[4] = {};    // [tree_cap, 2304] compacted accepted-path taps for DSpark
    float *d_ctrl_cat = nullptr, *d_ctrl_gate = nullptr, *d_ctrl_gain = nullptr, *d_ctrl_forget = nullptr;
    float *d_attn_pm = nullptr, *d_attn_pl = nullptr, *d_attn_po = nullptr;  // Split-K FlashDecoding buffers
    static constexpr int kInkPfKvTile = 2048;  // Fixed O(1)-VRAM KV tile (~114 MiB forever, fixes #6609 & #6605!)
    __nv_bfloat16 *d_pf_q = nullptr, *d_pf_k = nullptr, *d_pf_v = nullptr, *d_pf_p = nullptr;  // TC prefill attention
    float *d_pf_s = nullptr, *d_pf_o = nullptr, *d_pf_ot = nullptr, *d_pf_m = nullptr, *d_pf_l = nullptr;
    int pf_kv_cap = 0;
    float *d_stage_k = nullptr, *d_stage_v = nullptr, *d_stage_sconv = nullptr;  // Tree-16 verify staging
    InkKvVisitDesc* d_kv_desc_arr = nullptr;
    float** d_sconv_pool_arr = nullptr;
    int32_t* d_acc_path = nullptr;
    __nv_bfloat16 *d_x_bf16 = nullptr;
    __nv_bfloat16 *d_w_scratch = nullptr;  // [max_N, max_K] BF16 scratch for Q8_ROW / S_K cuBLAS prefill

    // Logits & screened argmax buffers
    float   *d_logits = nullptr;      // [logits_cap, 200058] (allocated lazily if full logits needed)
    int      logits_cap = 0;
    float   *d_dot_hat = nullptr;     // [16, 200058] for screened argmax
    float   *d_h_norm2 = nullptr;     // [16]
    int     *d_max_lb = nullptr;      // [16]
    int32_t *d_cand_ids = nullptr;    // [16, 4096]
    int32_t *d_cand_cnt = nullptr;    // [16]
    int32_t *d_argmax = nullptr;      // [64]

    // Step metadata in device memory (CUDA-Graph capturable!) + 64-slot pinned host ring (zero async H2D overwrite!)
    int32_t *d_tok_ids = nullptr;     // [max_batch]
    int32_t *d_slots = nullptr;       // [max_batch]
    int32_t *d_pos = nullptr;         // [max_batch]
    uint8_t *d_is_img = nullptr;      // [max_batch] optional vision mask
    float   *d_img_emb = nullptr;     // [max_batch, 2304] optional vision embeddings
    bool     has_img_step = false;
    int8_t  *d_sconv_lag = nullptr;   // [16 * 3]
    uint8_t *d_tree_anc = nullptr;    // [16 * 16]
    uint8_t *d_tree_dep = nullptr;    // [16]
    int32_t *h_meta_pin = nullptr;    // 64-slot ring buffer
    size_t   meta_ring_stride = 0;
    size_t   meta_ring_idx = 0;

    InkMoeWorkspace moe_ws;
    InkMoeWorkspace* cur_moe_ws = &moe_ws;

    // Multi-GPU Pipeline Layer-Split (`cuda-ls`) per-stage execution context (stages[0] mirrors primary pointers!)
    struct StageExec {
        cudaStream_t st = nullptr;
        cublasHandle_t cublas = nullptr;
        cudaEvent_t ev_done = nullptr;
        float *d_cur = nullptr, *d_res = nullptr, *d_bank_out = nullptr;
        float *d_blocks[8] = {};
        float *d_bank[4] = {};
        float *d_mixed = nullptr, *d_msgs = nullptr, *d_inp0 = nullptr, *d_inp1 = nullptr, *d_inp2 = nullptr;
        float *d_q = nullptr, *d_k_raw = nullptr, *d_v_raw = nullptr, *d_k = nullptr, *d_v = nullptr, *d_r = nullptr;
        float *d_k_dyn = nullptr, *d_v_dyn = nullptr;
        float *d_sinks = nullptr, *d_gate_attn = nullptr, *d_rel_table = nullptr;
        float *d_attn_out = nullptr, *d_wo_raw = nullptr, *d_wo_conv = nullptr;
        float *d_moe_in = nullptr, *d_moe_raw = nullptr, *d_moe_conv = nullptr;
        float *d_ctrl_cat = nullptr, *d_ctrl_gate = nullptr, *d_ctrl_gain = nullptr, *d_ctrl_forget = nullptr;
        float *d_attn_pm = nullptr, *d_attn_pl = nullptr, *d_attn_po = nullptr;
        __nv_bfloat16 *d_pf_q = nullptr, *d_pf_k = nullptr, *d_pf_v = nullptr, *d_pf_p = nullptr;
        float *d_pf_s = nullptr, *d_pf_o = nullptr, *d_pf_ot = nullptr, *d_pf_m = nullptr, *d_pf_l = nullptr;
        int pf_kv_cap = 0;
        __nv_bfloat16 *d_x_bf16 = nullptr, *d_w_scratch = nullptr;
        int32_t *d_slots = nullptr, *d_pos = nullptr;
        InkMoeWorkspace ws;
    };
    StageExec ls_stages[InkCudaModel::kMaxStages]{};
    int cur_stage = 0;
    float* h_ls_bounce = nullptr;
    int32_t* h_ls_meta_bounce = nullptr;

    // Dual-GPU Expert Parallelism (cuda-ep2) peer workspace & async sync (dynamically sized to max_batch -> zero B=64 cap!)
    cudaStream_t peer_st = nullptr;
    cudaEvent_t ev_moe_ready = nullptr, ev_peer_done = nullptr;
    InkMoeWorkspace peer_ws;
    float *d_peer_x = nullptr, *d_peer_y = nullptr, *d_peer_y_rx = nullptr;
    float *h_ep_x = nullptr, *h_ep_w = nullptr, *h_ep_y = nullptr;
    int32_t *h_ep_idx = nullptr;

    // 48-Visit CUDA Graph cache (keyed by M, is_tree, capture_taps)
    struct GraphKey {
        int M;
        int is_tree;
        int want_taps;
        int kv_fp8;
        bool operator==(const GraphKey& o) const {
            return M == o.M && is_tree == o.is_tree && want_taps == o.want_taps && kv_fp8 == o.kv_fp8;
        }
    };
    struct GraphEntry {
        GraphKey key;
        cudaGraphExec_t exec = nullptr;
    };
    std::vector<GraphEntry> graph_cache;
    bool use_cuda_graph = true;

    // Optional per-stage profiler (active only when prof_enabled=true and graph disabled)
    bool prof_enabled = false;
    double prof_ms[8] = {};  // 0=res/ctrl/norm, 1=dense_qkv, 2=sconv_clip_kv, 3=attn, 4=dense_wo, 5=moe, 6=other, 7=unembed
    cudaEvent_t ev_p0 = nullptr, ev_p1 = nullptr;
    void prof_mark_start() { if (prof_enabled) INK_CK(cudaEventRecord(ev_p0, st)); }
    void prof_mark_end(int idx) {
        if (!prof_enabled) return;
        INK_CK(cudaEventRecord(ev_p1, st));
        INK_CK(cudaEventSynchronize(ev_p1));
        float ms = 0.0f;
        INK_CK(cudaEventElapsedTime(&ms, ev_p0, ev_p1));
        prof_ms[idx] += ms;
    }
    void prof_reset() { for (double& x : prof_ms) x = 0.0; }
    void prof_print(const char* tag, int iters) const {
        double tot = 0; for (double x : prof_ms) tot += x;
        std::printf("[PROF %s / %d iters] total=%.2f ms/step | res_ctrl=%.2f dense_qkv=%.2f sconv_kv=%.2f attn=%.2f dense_wo=%.2f moe=%.2f unembed=%.2f\n",
                    tag, iters, tot / iters,
                    prof_ms[0] / iters, prof_ms[1] / iters, prof_ms[2] / iters, prof_ms[3] / iters,
                    prof_ms[4] / iters, prof_ms[5] / iters, prof_ms[7] / iters);
    }

    static size_t kv_bytes_per_visit_pos(InkKvDtype dt) noexcept {
        if (dt == InkKvDtype::FP16) return (size_t)kInkKVDim * 4u;
        if (dt == InkKvDtype::K8V4) return (size_t)kInkKVDim + ((size_t)kInkKVDim / 2u) + (size_t)(2 * kInkKVHeads * 4);
        return (size_t)kInkKVDim * 2u + (size_t)(2 * kInkKVHeads * 4);  // FP8 or K8V8
    }

    InkCudaRunner(InkCudaModel& model) : m(model) {}

    void bind_stage(int s) {
        if (!m.ls_active || s == cur_stage) return;
        ls_stages[cur_stage].d_pf_q = d_pf_q; ls_stages[cur_stage].d_pf_k = d_pf_k;
        ls_stages[cur_stage].d_pf_v = d_pf_v; ls_stages[cur_stage].d_pf_p = d_pf_p;
        ls_stages[cur_stage].d_pf_s = d_pf_s; ls_stages[cur_stage].d_pf_o = d_pf_o;
        ls_stages[cur_stage].d_pf_ot = d_pf_ot; ls_stages[cur_stage].d_pf_m = d_pf_m;
        ls_stages[cur_stage].d_pf_l = d_pf_l; ls_stages[cur_stage].pf_kv_cap = pf_kv_cap;
        cur_stage = s;
        INK_CK(cudaSetDevice(m.stage_gpu[s]));
        StageExec& se = ls_stages[s];
        st = se.st; cublas = se.cublas;
        d_cur = se.d_cur; d_res = se.d_res; d_bank_out = se.d_bank_out;
        for (int i = 0; i < 8; ++i) d_blocks[i] = se.d_blocks[i];
        for (int i = 0; i < 4; ++i) d_bank[i] = se.d_bank[i];
        d_mixed = se.d_mixed; d_msgs = se.d_msgs; d_inp0 = se.d_inp0; d_inp1 = se.d_inp1; d_inp2 = se.d_inp2;
        d_q = se.d_q; d_k_raw = se.d_k_raw; d_v_raw = se.d_v_raw; d_k = se.d_k; d_v = se.d_v; d_r = se.d_r;
        d_k_dyn = se.d_k_dyn; d_v_dyn = se.d_v_dyn;
        d_sinks = se.d_sinks; d_gate_attn = se.d_gate_attn; d_rel_table = se.d_rel_table;
        d_attn_out = se.d_attn_out; d_wo_raw = se.d_wo_raw; d_wo_conv = se.d_wo_conv;
        d_moe_in = se.d_moe_in; d_moe_raw = se.d_moe_raw; d_moe_conv = se.d_moe_conv;
        d_ctrl_cat = se.d_ctrl_cat; d_ctrl_gate = se.d_ctrl_gate; d_ctrl_gain = se.d_ctrl_gain; d_ctrl_forget = se.d_ctrl_forget;
        d_attn_pm = se.d_attn_pm; d_attn_pl = se.d_attn_pl; d_attn_po = se.d_attn_po;
        d_pf_q = se.d_pf_q; d_pf_k = se.d_pf_k; d_pf_v = se.d_pf_v; d_pf_p = se.d_pf_p;
        d_pf_s = se.d_pf_s; d_pf_o = se.d_pf_o; d_pf_ot = se.d_pf_ot; d_pf_m = se.d_pf_m; d_pf_l = se.d_pf_l;
        pf_kv_cap = se.pf_kv_cap;
        d_x_bf16 = se.d_x_bf16; d_w_scratch = se.d_w_scratch;
        d_slots = se.d_slots; d_pos = se.d_pos;
        cur_moe_ws = (s == 0) ? &moe_ws : &se.ws;
    }

    void copy_between_stages(int from_s, int to_s, const void* src, void* dst, size_t bytes) {
        if (from_s == to_s || bytes == 0) return;
        const int g_from = m.stage_gpu[from_s];
        const int g_to   = m.stage_gpu[to_s];
        if (m.stage_p2p[from_s][to_s]) {
            INK_CK(cudaMemcpyPeerAsync(dst, g_to, src, g_from, bytes, ls_stages[to_s].st));
        } else {
            INK_CK(cudaSetDevice(g_from));
            INK_CK(cudaMemcpyAsync(h_ls_bounce, src, bytes, cudaMemcpyDeviceToHost, ls_stages[from_s].st));
            INK_CK(cudaEventRecord(ls_stages[from_s].ev_done, ls_stages[from_s].st));
            INK_CK(cudaSetDevice(g_to));
            INK_CK(cudaStreamWaitEvent(ls_stages[to_s].st, ls_stages[from_s].ev_done, 0));
            INK_CK(cudaMemcpyAsync(dst, h_ls_bounce, bytes, cudaMemcpyHostToDevice, ls_stages[to_s].st));
            INK_CK(cudaEventRecord(ls_stages[to_s].ev_done, ls_stages[to_s].st));
            INK_CK(cudaSetDevice(g_from));
            INK_CK(cudaStreamWaitEvent(ls_stages[from_s].st, ls_stages[to_s].ev_done, 0));
            INK_CK(cudaSetDevice(g_to));
        }
    }

    void switch_stage_handoff(int to_s, int M, int mode, int n_blocks, const float* (&block_ptrs)[8],
                              DepthStatePtrs* bank_stp, int n_bank) {
        if (!m.ls_active || to_s == cur_stage) return;
        const int from_s = cur_stage;
        const size_t hbytes = (size_t)M * kInkHidden * sizeof(float);
        StageExec& sf = ls_stages[from_s];
        StageExec& st_to = ls_stages[to_s];
        if (m.stage_p2p[from_s][to_s]) {
            INK_CK(cudaSetDevice(m.stage_gpu[from_s]));
            INK_CK(cudaEventRecord(sf.ev_done, sf.st));
            INK_CK(cudaSetDevice(m.stage_gpu[to_s]));
            INK_CK(cudaStreamWaitEvent(st_to.st, sf.ev_done, 0));
        }
        copy_between_stages(from_s, to_s, sf.d_cur, st_to.d_cur, hbytes);
        for (int b = 0; b < n_blocks; ++b) {
            copy_between_stages(from_s, to_s, sf.d_blocks[b], st_to.d_blocks[b], hbytes);
            block_ptrs[b] = st_to.d_blocks[b];
        }
        if (bank_stp && bank_stp->count > 0) {
            for (int b = 0; b < n_bank; ++b) {
                copy_between_stages(from_s, to_s, sf.d_bank[b], st_to.d_bank[b], hbytes);
                bank_stp->p[b] = st_to.d_bank[b];
            }
            copy_between_stages(from_s, to_s, sf.d_bank_out, st_to.d_bank_out, hbytes);
            bank_stp->p[n_bank] = st_to.d_bank_out;
        }
        copy_between_stages(from_s, to_s, sf.d_pos, st_to.d_pos, (size_t)M * sizeof(int32_t));
        if (mode == 1) {
            copy_between_stages(from_s, to_s, sf.d_slots, st_to.d_slots, (size_t)M * sizeof(int32_t));
        }
        bind_stage(to_s);
    }

    template <typename T>
    static bool alloc_mapped_host(size_t count, void*& raw_out, T*& dev_out) noexcept {
        raw_out = nullptr;
        dev_out = nullptr;
        if (count == 0) return true;
        const size_t bytes = count * sizeof(T);
        if (cudaHostAlloc(&raw_out, bytes, cudaHostAllocMapped) != cudaSuccess) {
            cudaGetLastError();
            raw_out = nullptr;
            return false;
        }
        void* dp = nullptr;
        if (cudaHostGetDevicePointer(&dp, raw_out, 0) != cudaSuccess) {
            cudaGetLastError();
            cudaFreeHost(raw_out);
            raw_out = nullptr;
            return false;
        }
        dev_out = reinterpret_cast<T*>(dp);
        return true;
    }

    void free_host_kv_visit(int v) noexcept {
        if (host_kv_raw_k[v])   { cudaFreeHost(host_kv_raw_k[v]);   host_kv_raw_k[v] = nullptr; }
        if (host_kv_raw_v[v])   { cudaFreeHost(host_kv_raw_v[v]);   host_kv_raw_v[v] = nullptr; }
        if (host_kv_raw_ksc[v]) { cudaFreeHost(host_kv_raw_ksc[v]); host_kv_raw_ksc[v] = nullptr; }
        if (host_kv_raw_vsc[v]) { cudaFreeHost(host_kv_raw_vsc[v]); host_kv_raw_vsc[v] = nullptr; }
        kv_desc[v].h_k_f16 = nullptr; kv_desc[v].h_v_f16 = nullptr;
        kv_desc[v].h_k_f8  = nullptr; kv_desc[v].h_v_f8  = nullptr;
        kv_desc[v].h_k_sc  = nullptr; kv_desc[v].h_v_sc  = nullptr;
        kv_desc[v].host_cap = 0;
    }

    void init(int slots, int ctx, int chunk, InkKvDtype kvd = InkKvDtype::FP16, int kv_offload = 0) {
        max_slots = std::max(1, slots);
        max_ctx = std::max(64, ctx);
        kv_offload_mode = kv_offload;
        max_batch = std::max(chunk, std::max(16, max_slots));
        swa_ring = (max_slots >= 16) ? (kInkSwaWin + 32) : (kInkSwaWin + std::max(max_batch, 64));
        kv_dtype = kvd;
        cur_stage = 0;
        cur_moe_ws = &moe_ws;

        INK_CK(cudaSetDevice(m.primary_gpu));
        INK_CK(cudaStreamCreate(&st));
        INK_CK(cudaEventCreate(&ev_p0));
        INK_CK(cudaEventCreate(&ev_p1));
        INK_CUBLAS_CK(cublasCreate(&cublas));
        INK_CUBLAS_CK(cublasSetStream(cublas, st));
        INK_CUBLAS_CK(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));

        // Auto-fit max_ctx & split Global KV across VRAM + zero-copy mapped pinned host RAM (--kv-offload auto|on|off)!
        {
            size_t vfree = 0, vtot = 0;
            INK_CK(ink_cuda_mem_info(&vfree, &vtot, m.primary_gpu));
            tree_cap = (max_slots <= 4 && vfree > (700ull << 20)) ? kInkMaxTree : 16;
            max_batch = std::max(max_batch, tree_cap);
            if (vfree < (2200ull << 20) && max_batch > 512) {
                max_batch = 512;
                swa_ring = (max_slots >= 16) ? (kInkSwaWin + 32) : (kInkSwaWin + max_batch);
            }
            if (vfree < (1200ull << 20) && max_batch > 256) {
                max_batch = 256;
                swa_ring = (max_slots >= 16) ? (kInkSwaWin + 32) : (kInkSwaWin + max_batch);
            }
            const size_t per_visit_pos = kv_bytes_per_visit_pos(kv_dtype);
            int n_swa_g0 = 0, n_glob_g0 = 0;
            for (int v = 0; v < kInkVisits; ++v) {
                const int ip = ink_visit_phys(v);
                if (!m.ls_active || m.layer_owner[ip] == 0) {
                    if (m.man.c.is_swa[ip]) ++n_swa_g0; else ++n_glob_g0;
                }
            }
            const size_t swa_kv_bytes = (size_t)n_swa_g0 * max_slots * swa_ring * per_visit_pos;
            const size_t sconv_bytes  = (size_t)(n_swa_g0 + n_glob_g0) * max_slots * 3 * kInkSconvCh * 4;
            const size_t wddm_fl = InkCudaModel::wddm_safety_floor(vtot);
            const size_t ws_reserve = ((max_slots >= 8) ? (1280ull << 20) : (640ull << 20))
                                    + (m.man.has_dspark ? (160ull << 20) : 0ull)
                                    + (wddm_fl / 2);
            const size_t avail_glob = (vfree > swa_kv_bytes + sconv_bytes + ws_reserve)
                ? (vfree - swa_kv_bytes - sconv_bytes - ws_reserve) : 0u;
            const size_t per_pos_glob = (size_t)std::max(1, n_glob_g0) * max_slots * per_visit_pos
                                      + (m.man.has_dspark ? (size_t)4096u : 0u);
            const int vram_fit_ctx = (n_glob_g0 > 0)
                ? (int)std::min<size_t>((size_t)max_ctx, (avail_glob / std::max<size_t>(1, per_pos_glob)) & ~(size_t)63)
                : max_ctx;

            // Check mapped host RAM budget for Host KV Offloading (8 global visits total!)
            const uint64_t host_avail = ink_ram_available_bytes();
            const uint64_t host_os_res = 2ull << 30;
            const size_t per_pos_all_glob = (size_t)8 * max_slots * per_visit_pos;
            const int host_fit_ctx = (host_avail > host_os_res)
                ? (int)std::min<size_t>((size_t)max_ctx, ((size_t)(host_avail - host_os_res) / std::max<size_t>(1, per_pos_all_glob)) & ~(size_t)63)
                : max_ctx;

            if (kv_offload_mode == 1 && host_fit_ctx >= 256) {
                if (host_fit_ctx < max_ctx) max_ctx = std::max(256, host_fit_ctx);
                kv_dev_ctx = 0;
                kv_host_ctx = max_ctx;
                kv_alloc_ctx = max_ctx;
                std::fprintf(stderr, "[InkCUDA] Host KV Offload ON (kv_fmt=%d): 8 global visits = 0 VRAM + %d mapped host tokens (%.1f MiB host RAM)\n",
                             (int)kv_dtype, kv_host_ctx, (double)((size_t)kv_host_ctx * per_pos_all_glob) / (1024.0 * 1024.0));
            } else if (vram_fit_ctx >= max_ctx) {
                kv_dev_ctx = std::max(256, std::min(max_ctx, (max_slots == 1) ? 8192 : 4096));
                kv_host_ctx = 0;
                kv_alloc_ctx = kv_dev_ctx;
            } else if (kv_offload_mode == 0 && host_fit_ctx >= 256) {
                int dev_part = (vram_fit_ctx >= 1024) ? ((vram_fit_ctx * 3 / 4) & ~255) : 0;
                int host_part = std::min(max_ctx - dev_part, host_fit_ctx);
                if (dev_part + host_part < max_ctx) {
                    max_ctx = std::max(256, dev_part + host_part);
                    host_part = std::max(0, max_ctx - dev_part);
                }
                kv_dev_ctx = dev_part;
                kv_host_ctx = host_part;
                kv_alloc_ctx = kv_dev_ctx + kv_host_ctx;
                std::fprintf(stderr, "[InkCUDA] Auto Host KV Offload (kv_fmt=%d, %.2f GiB free VRAM): global KV = %d VRAM + %d mapped host tokens (%.1f MiB host RAM, max_ctx=%d)\n",
                             (int)kv_dtype, (double)vfree / (1024.0 * 1024.0 * 1024.0),
                             kv_dev_ctx, kv_host_ctx,
                             (double)((size_t)kv_host_ctx * per_pos_all_glob) / (1024.0 * 1024.0), max_ctx);
            } else {
                const int clamped = std::max(256, vram_fit_ctx);
                if (clamped < max_ctx) {
                    std::fprintf(stderr, "[InkCUDA] Auto-sizing max_ctx %d -> %d for parallel=%d (%.2f GiB free VRAM)\n",
                                 max_ctx, clamped, max_slots, (double)vfree / (1024.0 * 1024.0 * 1024.0));
                    max_ctx = clamped;
                }
                kv_dev_ctx = std::max(256, std::min(max_ctx, (max_slots == 1) ? 8192 : 4096));
                kv_host_ctx = 0;
                kv_alloc_ctx = kv_dev_ctx;
            }
        }

        // Allocate KV caches & ShortConv state pools on each visit's owning GPU (+ mapped pinned host tail for global visits)!
        for (int v = 0; v < kInkVisits; ++v) {
            const int ip = ink_visit_phys(v);
            const int dev = m.ls_active ? m.stage_gpu[m.layer_owner[ip]] : m.primary_gpu;
            INK_CK(cudaSetDevice(dev));
            const bool swa = m.man.c.is_swa[ip] != 0;
            const int dcap = swa ? swa_ring : kv_dev_ctx;
            const int hcap = swa ? 0 : kv_host_ctx;
            kv_desc[v] = {};
            kv_desc[v].cap_pos  = dcap + hcap;
            kv_desc[v].dev_cap  = dcap;
            kv_desc[v].host_cap = hcap;
            kv_desc[v].is_swa   = swa;
            kv_desc[v].kv_fmt   = (uint8_t)kv_dtype;
            if (dcap > 0) {
                const size_t elems = (size_t)max_slots * dcap * kInkKVHeads * kInkHeadDim;
                if (kv_dtype == InkKvDtype::FP16) {
                    kv_desc[v].k_f16 = dalloc<__half>(elems);
                    kv_desc[v].v_f16 = dalloc<__half>(elems);
                } else {
                    kv_desc[v].k_f8 = dalloc<uint8_t>(elems);
                    kv_desc[v].v_f8 = dalloc<uint8_t>((kv_dtype == InkKvDtype::K8V4) ? (elems / 2u) : elems);
                    kv_desc[v].k_sc = dalloc<float>((size_t)max_slots * dcap * kInkKVHeads);
                    kv_desc[v].v_sc = dalloc<float>((size_t)max_slots * dcap * kInkKVHeads);
                }
            }
            if (hcap > 0) {
                const size_t helems = (size_t)max_slots * hcap * kInkKVHeads * kInkHeadDim;
                bool hok = true;
                if (kv_dtype == InkKvDtype::FP16) {
                    hok = alloc_mapped_host<__half>(helems, host_kv_raw_k[v], kv_desc[v].h_k_f16)
                       && alloc_mapped_host<__half>(helems, host_kv_raw_v[v], kv_desc[v].h_v_f16);
                } else {
                    const size_t v_b = (kv_dtype == InkKvDtype::K8V4) ? (helems / 2u) : helems;
                    const size_t sc_n = (size_t)max_slots * hcap * kInkKVHeads;
                    hok = alloc_mapped_host<uint8_t>(helems, host_kv_raw_k[v], kv_desc[v].h_k_f8)
                       && alloc_mapped_host<uint8_t>(v_b, host_kv_raw_v[v], kv_desc[v].h_v_f8)
                       && alloc_mapped_host<float>(sc_n, host_kv_raw_ksc[v], kv_desc[v].h_k_sc)
                       && alloc_mapped_host<float>(sc_n, host_kv_raw_vsc[v], kv_desc[v].h_v_sc);
                }
                if (!hok) {
                    free_host_kv_visit(v);
                    kv_desc[v].cap_pos = dcap;
                    kv_host_ctx = 0;
                    kv_alloc_ctx = dcap;
                    max_ctx = std::max(256, dcap);
                    std::fprintf(stderr, "[InkCUDA] Warning: mapped host KV alloc failed at visit %d; falling back to dev_cap=%d\n", v, dcap);
                }
            }
            d_sconv_pool[v] = dalloc<float>((size_t)max_slots * 3 * kInkSconvCh);
            INK_CK(cudaMemset(d_sconv_pool[v], 0, (size_t)max_slots * 3 * kInkSconvCh * sizeof(float)));
        }
        INK_CK(cudaSetDevice(m.primary_gpu));

        const size_t MB = (size_t)max_batch;
        d_cur       = dalloc<float>(MB * kInkHidden);
        d_res       = dalloc<float>(MB * kInkHidden);
        d_prelude   = dalloc<float>(MB * kInkHidden);
        d_bank_out  = dalloc<float>(MB * kInkHidden);
        for (int i = 0; i < 8; ++i) d_blocks[i] = dalloc<float>(MB * kInkHidden);
        for (int i = 0; i < 2; ++i) d_prelude_blocks[i] = dalloc<float>(MB * kInkHidden);
        for (int i = 0; i < 4; ++i) d_bank[i] = dalloc<float>(MB * kInkHidden);
        d_mixed     = dalloc<float>(4 * MB * kInkHidden);
        d_msgs      = dalloc<float>(4 * MB * kInkHidden);
        d_inp0      = dalloc<float>(MB * kInkHidden);
        d_inp1      = dalloc<float>(MB * kInkHidden);
        d_inp2      = dalloc<float>(MB * kInkHidden);
        d_q         = dalloc<float>(MB * kInkQDim);
        d_k_raw     = dalloc<float>(MB * kInkKVDim);
        d_v_raw     = dalloc<float>(MB * kInkKVDim);
        d_k         = dalloc<float>(MB * kInkKVDim);
        d_v         = dalloc<float>(MB * kInkKVDim);
        d_r         = dalloc<float>(MB * kInkHeads * kInkDRel);
        d_k_dyn     = dalloc<float>(MB * kInkKVHeads * 4);
        d_v_dyn     = dalloc<float>(MB * kInkKVHeads * 4);
        d_sinks     = dalloc<float>(MB * kInkHeads);
        d_gate_attn = dalloc<float>(MB * kInkQDim);
        d_rel_table = dalloc<float>(MB * kInkHeads * (kInkRelExtent + 1));
        d_attn_out  = dalloc<float>(MB * kInkQDim);
        d_wo_raw    = dalloc<float>(MB * kInkHidden);
        d_wo_conv   = dalloc<float>(MB * kInkHidden);
        d_moe_in    = dalloc<float>(MB * kInkHidden);
        d_moe_raw   = dalloc<float>(MB * kInkHidden);
        d_moe_conv  = dalloc<float>(MB * kInkHidden);
        d_h_norm    = dalloc<float>(MB * kInkHidden);
        for (int i = 0; i < 4; ++i) {
            d_taps[i]         = dalloc<float>(MB * kInkHidden);
            d_taps_compact[i] = dalloc<float>((size_t)tree_cap * kInkHidden);
        }
        d_ctrl_cat    = dalloc<float>(3ull * MB * 2 * kInkHidden);
        d_ctrl_gate   = dalloc<float>(3ull * MB * kInkGateRank);
        d_ctrl_gain   = dalloc<float>(MB * kInkHidden);
        d_ctrl_forget = dalloc<float>(MB * kInkHidden);
        const size_t sp_cap = std::max<size_t>((size_t)kInkAttnMaxSplits, (size_t)tree_cap * kInkAttnSplits);
        d_attn_pm     = dalloc<float>(sp_cap * kInkHeads);
        d_attn_pl     = dalloc<float>(sp_cap * kInkHeads);
        d_attn_po     = dalloc<float>(sp_cap * kInkHeads * kInkHeadDim);
        d_stage_k     = dalloc<float>((size_t)kInkVisits * tree_cap * kInkKVDim);
        d_stage_v     = dalloc<float>((size_t)kInkVisits * tree_cap * kInkKVDim);
        d_stage_sconv = dalloc<float>((size_t)kInkVisits * tree_cap * kInkSconvCh);
        d_kv_desc_arr = dalloc<InkKvVisitDesc>(kInkVisits);
        INK_CK(cudaMemcpy(d_kv_desc_arr, kv_desc, sizeof(kv_desc), cudaMemcpyHostToDevice));
        d_sconv_pool_arr = dalloc<float*>(kInkVisits);
        INK_CK(cudaMemcpy(d_sconv_pool_arr, d_sconv_pool, sizeof(d_sconv_pool), cudaMemcpyHostToDevice));
        d_acc_path    = dalloc<int32_t>(kInkMaxTree);

        d_x_bf16    = dalloc<__nv_bfloat16>(MB * std::max(kInkDsparkInter, std::max(kInkQDim, kInkHidden)));
        d_w_scratch = dalloc<__nv_bfloat16>((size_t)kInkDsparkInter * kInkHidden);

        // Warmup cuBLAS Tensor Cores once at init (eliminates 61 ms first-call JIT tax from prompt #1!)
        {
            INK_CK(cudaMemsetAsync(d_w_scratch, 0, 64 * 64 * sizeof(__nv_bfloat16), st));
            INK_CK(cudaMemsetAsync(d_x_bf16, 0, 64 * 64 * sizeof(__nv_bfloat16), st));
            INK_CK(cudaMemsetAsync(d_ctrl_cat, 0, 64 * 64 * sizeof(float), st));
            const float a = 1.0f, b = 0.0f;
            if (m.sm_cc >= 80) {
                cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, 64, &a,
                             d_w_scratch, CUDA_R_16BF, 64, d_x_bf16, CUDA_R_16BF, 64, &b,
                             d_ctrl_gain, CUDA_R_32F, 64, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            } else if (m.sm_cc >= 70) {
                cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, 64, &a,
                             d_w_scratch, CUDA_R_16F, 64, d_x_bf16, CUDA_R_16F, 64, &b,
                             d_ctrl_gain, CUDA_R_32F, 64, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
            cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, 64, &a,
                         d_ctrl_cat, CUDA_R_32F, 64, d_ctrl_cat, CUDA_R_32F, 64, &b,
                         d_ctrl_gain, CUDA_R_32F, 64,
                         m.sm_cc >= 80 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            INK_CK(cudaStreamSynchronize(st));
        }

        d_dot_hat   = dalloc<float>(8ull * kInkVocabUnpad);
        d_h_norm2   = dalloc<float>(8);
        d_max_lb    = dalloc<int>(8);
        d_cand_ids  = dalloc<int32_t>(8ull * 4096);
        d_cand_cnt  = dalloc<int32_t>(8);
        d_argmax    = dalloc<int32_t>(std::max(64, max_batch));

        d_tok_ids   = dalloc<int32_t>(MB);
        d_slots     = dalloc<int32_t>(MB);
        d_pos       = dalloc<int32_t>(MB);
        d_is_img    = dalloc<uint8_t>(MB);
        d_img_emb   = dalloc<float>(MB * kInkHidden);
        d_sconv_lag = dalloc<int8_t>(kInkMaxTree * 3);
        d_tree_anc  = dalloc<uint8_t>(kInkMaxTree * kInkMaxTreeDepth);
        d_tree_dep  = dalloc<uint8_t>(kInkMaxTree);
        meta_ring_stride = std::max<size_t>(MB * 3 + 256, 512);
        meta_ring_idx = 0;
        h_meta_pin  = halloc_pinned<int32_t>(64ull * meta_ring_stride);

        moe_ws.alloc(max_batch);
        // Pre-allocate O(1)-VRAM Tensor Core prefill attention workspace (~114 MiB fixed forever!)
        if (m.sm_cc >= 60 && max_slots <= 4 && max_batch >= 128) {
            size_t vf = 0, vt = 0;
            INK_CK(ink_cuda_mem_info(&vf, &vt, m.primary_gpu));
            if (vf > (320ull << 20)) ensure_pf_attn(kInkPfKvTile);
        }

        if (m.ls_active) {
            use_cuda_graph = false;
            StageExec& s0 = ls_stages[0];
            s0.st = st; s0.cublas = cublas;
            INK_CK(cudaEventCreateWithFlags(&s0.ev_done, cudaEventDisableTiming));
            s0.d_cur = d_cur; s0.d_res = d_res; s0.d_bank_out = d_bank_out;
            for (int i = 0; i < 8; ++i) s0.d_blocks[i] = d_blocks[i];
            for (int i = 0; i < 4; ++i) s0.d_bank[i] = d_bank[i];
            s0.d_mixed = d_mixed; s0.d_msgs = d_msgs; s0.d_inp0 = d_inp0; s0.d_inp1 = d_inp1; s0.d_inp2 = d_inp2;
            s0.d_q = d_q; s0.d_k_raw = d_k_raw; s0.d_v_raw = d_v_raw; s0.d_k = d_k; s0.d_v = d_v; s0.d_r = d_r;
            s0.d_k_dyn = d_k_dyn; s0.d_v_dyn = d_v_dyn;
            s0.d_sinks = d_sinks; s0.d_gate_attn = d_gate_attn; s0.d_rel_table = d_rel_table;
            s0.d_attn_out = d_attn_out; s0.d_wo_raw = d_wo_raw; s0.d_wo_conv = d_wo_conv;
            s0.d_moe_in = d_moe_in; s0.d_moe_raw = d_moe_raw; s0.d_moe_conv = d_moe_conv;
            s0.d_ctrl_cat = d_ctrl_cat; s0.d_ctrl_gate = d_ctrl_gate; s0.d_ctrl_gain = d_ctrl_gain; s0.d_ctrl_forget = d_ctrl_forget;
            s0.d_attn_pm = d_attn_pm; s0.d_attn_pl = d_attn_pl; s0.d_attn_po = d_attn_po;
            s0.d_pf_q = d_pf_q; s0.d_pf_k = d_pf_k; s0.d_pf_v = d_pf_v; s0.d_pf_p = d_pf_p;
            s0.d_pf_s = d_pf_s; s0.d_pf_o = d_pf_o; s0.d_pf_ot = d_pf_ot; s0.d_pf_m = d_pf_m; s0.d_pf_l = d_pf_l;
            s0.pf_kv_cap = pf_kv_cap;
            s0.d_x_bf16 = d_x_bf16; s0.d_w_scratch = d_w_scratch;
            s0.d_slots = d_slots; s0.d_pos = d_pos;

            INK_CK(cudaHostAlloc(&h_ls_bounce, MB * kInkHidden * sizeof(float), cudaHostAllocPortable));
            INK_CK(cudaHostAlloc(&h_ls_meta_bounce, MB * sizeof(int32_t), cudaHostAllocPortable));

            for (int s = 1; s < m.n_stages; ++s) {
                INK_CK(cudaSetDevice(m.stage_gpu[s]));
                StageExec& se = ls_stages[s];
                INK_CK(cudaStreamCreate(&se.st));
                INK_CK(cudaEventCreateWithFlags(&se.ev_done, cudaEventDisableTiming));
                INK_CUBLAS_CK(cublasCreate(&se.cublas));
                INK_CUBLAS_CK(cublasSetStream(se.cublas, se.st));
                INK_CUBLAS_CK(cublasSetMathMode(se.cublas, CUBLAS_DEFAULT_MATH));
                se.d_cur = dalloc<float>(MB * kInkHidden);
                se.d_res = dalloc<float>(MB * kInkHidden);
                se.d_bank_out = dalloc<float>(MB * kInkHidden);
                for (int i = 0; i < 8; ++i) se.d_blocks[i] = dalloc<float>(MB * kInkHidden);
                for (int i = 0; i < 4; ++i) se.d_bank[i] = dalloc<float>(MB * kInkHidden);
                se.d_mixed = dalloc<float>(4 * MB * kInkHidden);
                se.d_msgs  = dalloc<float>(4 * MB * kInkHidden);
                se.d_inp0  = dalloc<float>(MB * kInkHidden);
                se.d_inp1  = dalloc<float>(MB * kInkHidden);
                se.d_inp2  = dalloc<float>(MB * kInkHidden);
                se.d_q     = dalloc<float>(MB * kInkQDim);
                se.d_k_raw = dalloc<float>(MB * kInkKVDim);
                se.d_v_raw = dalloc<float>(MB * kInkKVDim);
                se.d_k     = dalloc<float>(MB * kInkKVDim);
                se.d_v     = dalloc<float>(MB * kInkKVDim);
                se.d_r     = dalloc<float>(MB * kInkHeads * kInkDRel);
                se.d_k_dyn = dalloc<float>(MB * kInkKVHeads * 4);
                se.d_v_dyn = dalloc<float>(MB * kInkKVHeads * 4);
                se.d_sinks = dalloc<float>(MB * kInkHeads);
                se.d_gate_attn = dalloc<float>(MB * kInkQDim);
                se.d_rel_table = dalloc<float>(MB * kInkHeads * (kInkRelExtent + 1));
                se.d_attn_out  = dalloc<float>(MB * kInkQDim);
                se.d_wo_raw    = dalloc<float>(MB * kInkHidden);
                se.d_wo_conv   = dalloc<float>(MB * kInkHidden);
                se.d_moe_in    = dalloc<float>(MB * kInkHidden);
                se.d_moe_raw   = dalloc<float>(MB * kInkHidden);
                se.d_moe_conv  = dalloc<float>(MB * kInkHidden);
                se.d_ctrl_cat  = dalloc<float>(3ull * MB * 2 * kInkHidden);
                se.d_ctrl_gate = dalloc<float>(3ull * MB * kInkGateRank);
                se.d_ctrl_gain = dalloc<float>(MB * kInkHidden);
                se.d_ctrl_forget = dalloc<float>(MB * kInkHidden);
                se.d_attn_pm   = dalloc<float>(sp_cap * kInkHeads);
                se.d_attn_pl   = dalloc<float>(sp_cap * kInkHeads);
                se.d_attn_po   = dalloc<float>(sp_cap * kInkHeads * kInkHeadDim);
                se.d_x_bf16    = dalloc<__nv_bfloat16>(MB * std::max(kInkDsparkInter, std::max(kInkQDim, kInkHidden)));
                se.d_w_scratch = dalloc<__nv_bfloat16>((size_t)kInkDsparkInter * kInkHidden);
                se.d_slots     = dalloc<int32_t>(MB);
                se.d_pos       = dalloc<int32_t>(MB);
                se.ws.alloc(max_batch);
            }
            INK_CK(cudaSetDevice(m.primary_gpu));
        } else if (m.ep2_active && m.peer_gpu >= 0) {
            use_cuda_graph = false;
            INK_CK(cudaSetDevice(m.primary_gpu));
            INK_CK(cudaEventCreateWithFlags(&ev_moe_ready, cudaEventDisableTiming));
            d_peer_y_rx = dalloc<float>(MB * kInkHidden);
            INK_CK(cudaSetDevice(m.peer_gpu));
            INK_CK(cudaStreamCreateWithFlags(&peer_st, cudaStreamNonBlocking));
            INK_CK(cudaEventCreateWithFlags(&ev_peer_done, cudaEventDisableTiming));
            peer_ws.alloc(max_batch);
            d_peer_x = dalloc<float>(MB * kInkHidden);
            d_peer_y = dalloc<float>(MB * kInkHidden);
            INK_CK(cudaSetDevice(m.primary_gpu));
            if (!m.ep2_p2p) {
                INK_CK(cudaHostAlloc(&h_ep_x,   MB * kInkHidden * sizeof(float),   cudaHostAllocPortable));
                INK_CK(cudaHostAlloc(&h_ep_idx, MB * kInkTopK   * sizeof(int32_t), cudaHostAllocPortable));
                INK_CK(cudaHostAlloc(&h_ep_w,   MB * kInkTopK   * sizeof(float),   cudaHostAllocPortable));
                INK_CK(cudaHostAlloc(&h_ep_y,   MB * kInkHidden * sizeof(float),   cudaHostAllocPortable));
            }
        }
    }

    // Elastic Global KV Cache Growth: starts compact (4K..8K) and grows in 8K-token chunks up to max_ctx!
    // Returns false cleanly if VRAM is exhausted (NEVER aborts with exit(3)!).
    bool ensure_global_kv_cap(int needed_pos) {
        if (needed_pos <= kv_alloc_ctx) return true;
        if (needed_pos > max_ctx) return false;
        const int old_cap = kv_alloc_ctx;
        int new_cap = std::min(max_ctx, std::max(needed_pos, ((old_cap + 8191) / 8192 + 1) * 8192));
        new_cap = std::min(max_ctx, ((new_cap + 63) & ~63));

        const size_t per_pos = kv_bytes_per_visit_pos(kv_dtype);
        for (int s = 0; s < (m.ls_active ? m.n_stages : 1); ++s) {
            int n_glob_s = 0;
            for (int v = 0; v < kInkVisits; ++v) {
                const int ip = ink_visit_phys(v);
                if (!m.man.c.is_swa[ip] && (!m.ls_active || m.layer_owner[ip] == s)) ++n_glob_s;
            }
            if (n_glob_s == 0) continue;
            size_t vf = 0, vt = 0;
            const int dev = m.ls_active ? m.stage_gpu[s] : m.primary_gpu;
            if (ink_cuda_mem_info(&vf, &vt, dev) != cudaSuccess) return false;
            const size_t floor_b = 256ull << 20;
            const size_t need_b = (size_t)n_glob_s * max_slots * new_cap * per_pos;
            if (vf <= floor_b + need_b) {
                const int tight_cap = std::min(max_ctx, ((needed_pos + 255) & ~255));
                const size_t tight_b = (size_t)n_glob_s * max_slots * tight_cap * per_pos;
                if (vf <= floor_b + tight_b) {
                    // VRAM cannot grow global KV further: seamlessly spill tail [kv_dev_ctx..new_cap-1] into mapped pinned host RAM!
                    if (kv_offload_mode >= 0 && kv_host_ctx == 0) {
                        const int hcap = new_cap - kv_dev_ctx;
                        if (hcap > 0) {
                            if (m.ls_active) {
                                for (int ss = 0; ss < m.n_stages; ++ss) {
                                    cudaSetDevice(m.stage_gpu[ss]);
                                    cudaStreamSynchronize(ls_stages[ss].st);
                                }
                            } else {
                                cudaSetDevice(m.primary_gpu);
                                cudaStreamSynchronize(st);
                            }
                            bool all_ok = true;
                            for (int v = 0; v < kInkVisits && all_ok; ++v) {
                                if (kv_desc[v].is_swa) continue;
                                const int ip = ink_visit_phys(v);
                                const int dev = m.ls_active ? m.stage_gpu[m.layer_owner[ip]] : m.primary_gpu;
                                cudaSetDevice(dev);
                                const size_t helems = (size_t)max_slots * hcap * kInkKVHeads * kInkHeadDim;
                                if (kv_dtype == InkKvDtype::FP16) {
                                    all_ok = alloc_mapped_host<__half>(helems, host_kv_raw_k[v], kv_desc[v].h_k_f16)
                                          && alloc_mapped_host<__half>(helems, host_kv_raw_v[v], kv_desc[v].h_v_f16);
                                } else {
                                    const size_t v_b = (kv_dtype == InkKvDtype::K8V4) ? (helems / 2u) : helems;
                                    const size_t sc_n = (size_t)max_slots * hcap * kInkKVHeads;
                                    all_ok = alloc_mapped_host<uint8_t>(helems, host_kv_raw_k[v], kv_desc[v].h_k_f8)
                                          && alloc_mapped_host<uint8_t>(v_b, host_kv_raw_v[v], kv_desc[v].h_v_f8)
                                          && alloc_mapped_host<float>(sc_n, host_kv_raw_ksc[v], kv_desc[v].h_k_sc)
                                          && alloc_mapped_host<float>(sc_n, host_kv_raw_vsc[v], kv_desc[v].h_v_sc);
                                }
                            }
                            cudaSetDevice(m.primary_gpu);
                            if (all_ok) {
                                for (int v = 0; v < kInkVisits; ++v) {
                                    if (kv_desc[v].is_swa) continue;
                                    kv_desc[v].dev_cap  = kv_dev_ctx;
                                    kv_desc[v].host_cap = hcap;
                                    kv_desc[v].cap_pos  = kv_dev_ctx + hcap;
                                }
                                INK_CK(cudaMemcpy(d_kv_desc_arr, kv_desc, sizeof(kv_desc), cudaMemcpyHostToDevice));
                                for (auto& e : graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
                                graph_cache.clear();
                                kv_host_ctx = hcap;
                                kv_alloc_ctx = kv_dev_ctx + hcap;
                                std::fprintf(stderr, "[InkCUDA] Elastic global KV spilled to mapped host RAM: %d VRAM + %d host = %d tokens\n",
                                             kv_dev_ctx, kv_host_ctx, kv_alloc_ctx);
                                return true;
                            }
                            for (int v = 0; v < kInkVisits; ++v) if (!kv_desc[v].is_swa) free_host_kv_visit(v);
                        }
                    }
                    return false;
                }
                new_cap = std::min(new_cap, tight_cap);
            }
        }

        if (m.ls_active) {
            for (int s = 0; s < m.n_stages; ++s) {
                cudaSetDevice(m.stage_gpu[s]);
                cudaStreamSynchronize(ls_stages[s].st);
            }
        } else {
            cudaSetDevice(m.primary_gpu);
            cudaStreamSynchronize(st);
        }

        for (int v = 0; v < kInkVisits; ++v) {
            if (kv_desc[v].is_swa) continue;
            const int ip = ink_visit_phys(v);
            const int s  = m.ls_active ? m.layer_owner[ip] : 0;
            const int dev = m.ls_active ? m.stage_gpu[s] : m.primary_gpu;
            cudaStream_t vst = m.ls_active ? ls_stages[s].st : st;
            cudaSetDevice(dev);

            const size_t old_slot_el = (size_t)old_cap * kInkKVHeads * kInkHeadDim;
            const size_t new_slot_el = (size_t)new_cap * kInkKVHeads * kInkHeadDim;
            const size_t new_tot_el  = (size_t)max_slots * new_slot_el;

            if (kv_dtype == InkKvDtype::FP16) {
                __half *nk = nullptr, *nv = nullptr;
                if (cudaMalloc(&nk, new_tot_el * sizeof(__half)) != cudaSuccess ||
                    cudaMalloc(&nv, new_tot_el * sizeof(__half)) != cudaSuccess) {
                    if (nk) cudaFree(nk);
                    if (nv) cudaFree(nv);
                    cudaGetLastError();
                    cudaSetDevice(m.primary_gpu);
                    return false;
                }
                for (int sl = 0; sl < max_slots; ++sl) {
                    cudaMemcpyAsync(nk + (size_t)sl * new_slot_el, kv_desc[v].k_f16 + (size_t)sl * old_slot_el,
                                    old_slot_el * sizeof(__half), cudaMemcpyDeviceToDevice, vst);
                    cudaMemcpyAsync(nv + (size_t)sl * new_slot_el, kv_desc[v].v_f16 + (size_t)sl * old_slot_el,
                                    old_slot_el * sizeof(__half), cudaMemcpyDeviceToDevice, vst);
                }
                cudaStreamSynchronize(vst);
                cudaFree(kv_desc[v].k_f16);
                cudaFree(kv_desc[v].v_f16);
                kv_desc[v].k_f16 = nk;
                kv_desc[v].v_f16 = nv;
            } else {
                const bool v4 = (kv_dtype == InkKvDtype::K8V4);
                const size_t old_v_b = v4 ? (old_slot_el / 2u) : old_slot_el;
                const size_t new_v_b = v4 ? (new_slot_el / 2u) : new_slot_el;
                const size_t old_sc  = (size_t)old_cap * kInkKVHeads;
                const size_t new_sc  = (size_t)new_cap * kInkKVHeads;
                uint8_t *nk = nullptr, *nv = nullptr;
                float *nks = nullptr, *nvs = nullptr;
                if (cudaMalloc(&nk, new_tot_el) != cudaSuccess ||
                    cudaMalloc(&nv, (size_t)max_slots * new_v_b) != cudaSuccess ||
                    cudaMalloc(&nks, (size_t)max_slots * new_sc * sizeof(float)) != cudaSuccess ||
                    cudaMalloc(&nvs, (size_t)max_slots * new_sc * sizeof(float)) != cudaSuccess) {
                    if (nk) cudaFree(nk); if (nv) cudaFree(nv);
                    if (nks) cudaFree(nks); if (nvs) cudaFree(nvs);
                    cudaGetLastError();
                    cudaSetDevice(m.primary_gpu);
                    return false;
                }
                for (int sl = 0; sl < max_slots; ++sl) {
                    cudaMemcpyAsync(nk + (size_t)sl * new_slot_el, kv_desc[v].k_f8 + (size_t)sl * old_slot_el,
                                    old_slot_el, cudaMemcpyDeviceToDevice, vst);
                    cudaMemcpyAsync(nv + (size_t)sl * new_v_b, kv_desc[v].v_f8 + (size_t)sl * old_v_b,
                                    old_v_b, cudaMemcpyDeviceToDevice, vst);
                    cudaMemcpyAsync(nks + (size_t)sl * new_sc, kv_desc[v].k_sc + (size_t)sl * old_sc,
                                    old_sc * sizeof(float), cudaMemcpyDeviceToDevice, vst);
                    cudaMemcpyAsync(nvs + (size_t)sl * new_sc, kv_desc[v].v_sc + (size_t)sl * old_sc,
                                    old_sc * sizeof(float), cudaMemcpyDeviceToDevice, vst);
                }
                cudaStreamSynchronize(vst);
                cudaFree(kv_desc[v].k_f8); cudaFree(kv_desc[v].v_f8);
                cudaFree(kv_desc[v].k_sc); cudaFree(kv_desc[v].v_sc);
                kv_desc[v].k_f8 = nk; kv_desc[v].v_f8 = nv;
                kv_desc[v].k_sc = nks; kv_desc[v].v_sc = nvs;
            }
            kv_desc[v].cap_pos = new_cap;
            kv_desc[v].dev_cap = new_cap;
        }
        cudaSetDevice(m.primary_gpu);
        INK_CK(cudaMemcpy(d_kv_desc_arr, kv_desc, sizeof(kv_desc), cudaMemcpyHostToDevice));
        for (auto& e : graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
        graph_cache.clear();
        kv_dev_ctx = new_cap;
        kv_alloc_ctx = new_cap;
        std::fprintf(stderr, "[InkCUDA] Elastic global KV grown: %d -> %d tokens (max_ctx=%d)\n", old_cap, new_cap, max_ctx);
        return true;
    }

    void reset_slot(int slot) {
        constexpr int c_off[4] = {0, kInkKVDim, 2 * kInkKVDim, 2 * kInkKVDim + kInkHidden};
        constexpr int c_len[4] = {kInkKVDim, kInkKVDim, kInkHidden, kInkHidden};
        for (int v = 0; v < kInkVisits; ++v) {
            const int ip = ink_visit_phys(v);
            const int s = m.ls_active ? m.layer_owner[ip] : 0;
            if (m.ls_active) cudaSetDevice(m.stage_gpu[s]);
            cudaStream_t vst = m.ls_active ? ls_stages[s].st : st;
            for (int st_i = 0; st_i < 4; ++st_i) {
                float* p = d_sconv_pool[v] + (size_t)c_off[st_i] * 3 * max_slots + (size_t)slot * 3 * c_len[st_i];
                INK_CK(cudaMemsetAsync(p, 0, (size_t)3 * c_len[st_i] * sizeof(float), vst));
            }
        }
        if (m.ls_active) cudaSetDevice(m.stage_gpu[cur_stage]);
    }

    void ensure_logits(int M) {
        if (M <= logits_cap) return;
        INK_CK(cudaStreamSynchronize(st));
        if (d_logits) INK_CK(cudaFree(d_logits));
        logits_cap = M;
        d_logits = dalloc<float>((size_t)logits_cap * m.man.c.vocab_unpadded);
    }

    // ---- High-Speed Rank-64 Loop Controller + RMSNorm Dispatcher (100x faster than single-CTA!) ----
    void run_loop_control_norm(
        const float* cur, const float* msg,
        const float* w_down, const float* w_gain, const float* w_forget,
        const float* norm_w, float* out, int M,
        bool cat_ready = false, bool exact_fp32 = false) {
        constexpr int D = kInkHidden;
        constexpr int D2 = 2 * D;
        if (!cat_ready) k_ink_ctrl_cat_rms<<<M, 256, 0, st>>>(cur, msg, d_ctrl_cat, M, D);
        if (M <= 16) {
            if (M == 1) {
                k_ink_ctrl_down_smallM<1><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<1><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else if (M <= 4) {
                k_ink_ctrl_down_smallM<4><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<4><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else if (M <= 8) {
                k_ink_ctrl_down_smallM<8><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<8><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else {
                k_ink_ctrl_down_smallM<16><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<16><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            }
            if (norm_w) k_ink_rms_norm<<<M, 256, 0, st>>>(out, norm_w, out, nullptr, M, D);
        } else {
            const float alpha = 1.0f, beta = 0.0f;
            if (!exact_fp32 && m.sm_cc >= 70 && m.sm_cc < 80) {
                // Volta sm_70 (V100) / Turing sm_75 (2080 Ti): 125 TFLOPS FP16 TC for loop controller GEMMs (4x faster than FP32!)
                __half* x_h = reinterpret_cast<__half*>(d_x_bf16);
                __half* w_h = reinterpret_cast<__half*>(d_w_scratch);
                const size_t nx_d = (size_t)M * D2;
                const size_t nw_d = (size_t)kInkGateRank * D2;
                k_ink_f32_to_fp16<<<(nx_d + 255) / 256, 256, 0, st>>>(d_ctrl_cat, x_h, nx_d);
                k_ink_f32_to_fp16<<<(nw_d + 255) / 256, 256, 0, st>>>(w_down, w_h, nw_d);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, kInkGateRank, M, D2, &alpha,
                    w_h, CUDA_R_16F, D2, x_h, CUDA_R_16F, D2, &beta,
                    d_ctrl_gate, CUDA_R_32F, kInkGateRank, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                const size_t nx_g = (size_t)M * kInkGateRank;
                const size_t nw_g = (size_t)D * kInkGateRank;
                k_ink_f32_to_fp16<<<(nx_g + 255) / 256, 256, 0, st>>>(d_ctrl_gate, x_h, nx_g);
                k_ink_f32_to_fp16<<<(nw_g + 255) / 256, 256, 0, st>>>(w_gain, w_h, nw_g);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_h, CUDA_R_16F, kInkGateRank, x_h, CUDA_R_16F, kInkGateRank, &beta,
                    d_ctrl_gain, CUDA_R_32F, D, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                k_ink_f32_to_fp16<<<(nw_g + 255) / 256, 256, 0, st>>>(w_forget, w_h, nw_g);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_h, CUDA_R_16F, kInkGateRank, x_h, CUDA_R_16F, kInkGateRank, &beta,
                    d_ctrl_forget, CUDA_R_32F, D, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else {
                const cublasComputeType_t ctype = (!exact_fp32 && m.sm_cc >= 80) ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, kInkGateRank, M, D2, &alpha,
                    w_down, CUDA_R_32F, D2, d_ctrl_cat, CUDA_R_32F, D2, &beta,
                    d_ctrl_gate, CUDA_R_32F, kInkGateRank, ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_gain, CUDA_R_32F, kInkGateRank, d_ctrl_gate, CUDA_R_32F, kInkGateRank, &beta,
                    d_ctrl_gain, CUDA_R_32F, D, ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_forget, CUDA_R_32F, kInkGateRank, d_ctrl_gate, CUDA_R_32F, kInkGateRank, &beta,
                    d_ctrl_forget, CUDA_R_32F, D, ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            k_ink_ctrl_blend_norm<<<M, 256, 0, st>>>(cur, msg, d_ctrl_gain, d_ctrl_forget, norm_w, out, M, D);
        }
    }

    // ---- Cross-Architecture cuBLAS GEMM for BF16 weights: sm_80+ native BF16 TC, sm_70/75 FP16 TC, sm_60/61 FP32 ----
    // When w_is_scratch==true, caller placed BF16 weights at d_w_scratch (or at d_w_scratch + 2*(cap/3) when sm_cc < 70).
    void gemm_bf16_compat(const __nv_bfloat16* W_bf16, const float* X, float* Y, int M, int N, int K,
                          int ldY, bool reuse_x, bool w_is_scratch) {
        const float alpha = 1.0f, beta = 0.0f;
        const int cur_cc = m.ls_active ? m.stage_cc[cur_stage] : m.sm_cc;
        if (cur_cc >= 80) {
            if (!reuse_x) {
                const size_t nx = (size_t)M * K;
                k_ink_f32_to_bf16<<<((nx + 3) / 4 + 255) / 256, 256, 0, st>>>(X, d_x_bf16, nx);
            }
            INK_CUBLAS_CK(cublasGemmEx(
                cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &alpha,
                W_bf16, CUDA_R_16BF, K,
                d_x_bf16, CUDA_R_16BF, K,
                &beta, Y, CUDA_R_32F, ldY,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else if (cur_cc >= 70) {
            __half* x_fp16 = reinterpret_cast<__half*>(d_x_bf16);
            if (!reuse_x) {
                const size_t nx = (size_t)M * K;
                k_ink_f32_to_fp16<<<(nx + 255) / 256, 256, 0, st>>>(X, x_fp16, nx);
            }
            const size_t cap = (size_t)kInkDsparkInter * kInkHidden;
            if (w_is_scratch) {
                const size_t nw = (size_t)N * K;
                __half* w_fp16 = reinterpret_cast<__half*>(d_w_scratch);
                k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16, w_fp16, nw);
                cublasStatus_t st_fp16 = cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    w_fp16, CUDA_R_16F, K,
                    x_fp16, CUDA_R_16F, K,
                    &beta, Y, CUDA_R_32F, ldY,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st_fp16 != CUBLAS_STATUS_SUCCESS) {
                    INK_CUBLAS_CK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        N, M, K, &alpha,
                        w_fp16, CUDA_R_16F, K,
                        x_fp16, CUDA_R_16F, K,
                        &beta, Y, CUDA_R_32F, ldY,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                }
            } else {
                const int tile_n = std::min(N, (int)(cap / (size_t)K));
                __half* w_fp16 = reinterpret_cast<__half*>(d_w_scratch);
                for (int n0 = 0; n0 < N; n0 += tile_n) {
                    const int nt = std::min(tile_n, N - n0);
                    const size_t nw = (size_t)nt * K;
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16 + (size_t)n0 * K, w_fp16, nw);
                    INK_CUBLAS_CK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        nt, M, K, &alpha,
                        w_fp16, CUDA_R_16F, K,
                        x_fp16, CUDA_R_16F, K,
                        &beta, Y + n0, CUDA_R_32F, ldY,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
            }
        } else {
            const size_t cap = (size_t)kInkDsparkInter * kInkHidden;
            const size_t max_elems = cap / 3;
            float* w_f32 = reinterpret_cast<float*>(d_w_scratch);
            if (w_is_scratch) {
                const size_t nw = (size_t)N * K;
                k_ink_bf16_to_f32<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16, w_f32, nw);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    w_f32, CUDA_R_32F, K,
                    X, CUDA_R_32F, K,
                    &beta, Y, CUDA_R_32F, ldY,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            } else {
                const int tile_n = std::min(N, (int)(max_elems / (size_t)K));
                for (int n0 = 0; n0 < N; n0 += tile_n) {
                    const int nt = std::min(tile_n, N - n0);
                    const size_t nw = (size_t)nt * K;
                    k_ink_bf16_to_f32<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16 + (size_t)n0 * K, w_f32, nw);
                    INK_CUBLAS_CK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        nt, M, K, &alpha,
                        w_f32, CUDA_R_32F, K,
                        X, CUDA_R_32F, K,
                        &beta, Y + n0, CUDA_R_32F, ldY,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                }
            }
        }
    }

    // ---- Dense linear projection helper: Y[M, N] = X[M, K] @ W[N, K]^T ----
    void linear_multi(const std::string* names, float* const* Ys, int count, const float* X, int M, bool exact_fp32_x = false) {
        if (count <= 0) return;
        const auto& ti0 = m.tinfo(names[0]);
        const int K = (int)ti0.cols();
        const int cur_cc = m.ls_active ? m.stage_cc[cur_stage] : m.sm_cc;
        const bool cur_attn_fp16 = m.ls_active ? m.stage_trunk_attn_fp16[cur_stage] : m.trunk_attn_fp16;
        const bool is_fp16 = cur_attn_fp16 && (names[0].find(".attn.") != std::string::npos);
        const int bf16_gemv_max = is_fp16 ? 3 : ((exact_fp32_x || cur_cc < 80) ? 16 : 7);
        bool can_fuse = (count <= 5) && ((ti0.fmt == FMT_BF16 && M <= bf16_gemv_max) || (ti0.fmt == FMT_Q8ROW && M <= 16));
        InkMultiProj mp{};
        mp.count = count;
        mp.cumN[0] = 0;
        for (int i = 0; i < count; ++i) {
            const auto& ti = m.tinfo(names[i]);
            if (ti.fmt != ti0.fmt || (int)ti.cols() != K) { can_fuse = false; break; }
            if (cur_attn_fp16 && ((names[i].find(".attn.") != std::string::npos) != is_fp16)) { can_fuse = false; break; }
            mp.W[i] = m.trunk_ptr_stage(cur_stage, names[i]);
            mp.Y[i] = Ys[i];
            mp.N[i] = (int)ti.rows();
            mp.cumN[i + 1] = mp.cumN[i] + mp.N[i];
        }
        if (can_fuse) {
            const int grid = (mp.cumN[count] + 7) / 8;
            if (ti0.fmt == FMT_BF16) {
                if (is_fp16) {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<4, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                } else {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 4) k_ink_gemv_multi_bf16<4, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 8) k_ink_gemv_multi_bf16<8, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<16, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                }
            } else if ((K % 64) == 0 && K <= kInkQDim && (cur_cc == 61 || cur_cc >= 75)) {
                if (M == 1) {
                    k_ink_gemv_multi_q8row_m1<<<grid, 128, 0, st>>>(mp, X, K);
                } else {
                    const int nblk = M * (K / 64);
                    k_ink_quant_q8_1<<<(nblk + 7) / 8, 256, 0, st>>>(X, cur_moe_ws->xq_sk, M, K);
                    if (M <= 4)      k_ink_gemv_multi_q8row_preq<4><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                    else if (M <= 8) k_ink_gemv_multi_q8row_preq<8><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                    else             k_ink_gemv_multi_q8row_preq<16><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                }
            } else {
                const int g4 = (mp.cumN[count] + 3) / 4;
                if (M == 1)      k_ink_gemv_multi_q8row<1><<<g4, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 4) k_ink_gemv_multi_q8row<4><<<g4, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 8) k_ink_gemv_multi_q8row<8><<<g4, 128, 0, st>>>(mp, X, M, K);
                else             k_ink_gemv_multi_q8row<16><<<g4, 128, 0, st>>>(mp, X, M, K);
            }
            return;
        }
        for (int i = 0; i < count; ++i) linear(names[i], X, Ys[i], M, 0, i > 0, exact_fp32_x);
    }

    void linear(const std::string& tname, const float* X, float* Y, int M, int N_override = 0, bool reuse_x_bf16 = false, bool exact_fp32_x = false) {
        const auto& ti = m.tinfo(tname);
        const int N = N_override > 0 ? N_override : (int)ti.rows();
        const int K = (int)ti.cols();
        const uint8_t* W = m.trunk_ptr_stage(cur_stage, tname);
        const int cur_cc = m.ls_active ? m.stage_cc[cur_stage] : m.sm_cc;
        const bool cur_attn_fp16 = m.ls_active ? m.stage_trunk_attn_fp16[cur_stage] : m.trunk_attn_fp16;

        if (ti.fmt == FMT_BF16) {
            const bool is_fp16 = cur_attn_fp16 && (tname.find(".attn.") != std::string::npos);
            const int bf16_gemv_max = is_fp16 ? 3 : ((exact_fp32_x || cur_cc < 80) ? 16 : 7);
            if (M <= bf16_gemv_max) {
                InkMultiProj mp{};
                mp.W[0] = W; mp.Y[0] = Y; mp.N[0] = N; mp.cumN[0] = 0; mp.cumN[1] = N; mp.count = 1;
                const int grid = (N + 7) / 8;
                if (is_fp16) {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<4, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                } else {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 4) k_ink_gemv_multi_bf16<4, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 8) k_ink_gemv_multi_bf16<8, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<16, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                }
            } else if (is_fp16) {
                __half* x_fp16 = reinterpret_cast<__half*>(d_x_bf16);
                if (!reuse_x_bf16) {
                    const size_t nx = (size_t)M * K;
                    k_ink_f32_to_fp16<<<(nx + 255) / 256, 256, 0, st>>>(X, x_fp16, nx);
                }
                const float alpha = 1.0f, beta = 0.0f;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    W, CUDA_R_16F, K,
                    x_fp16, CUDA_R_16F, K,
                    &beta, Y, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else {
                gemm_bf16_compat(reinterpret_cast<const __nv_bfloat16*>(W), X, Y, M, N, K, N, reuse_x_bf16, false);
            }
        } else if (ti.fmt == FMT_F32) {
            if (M <= 16) {
                const int grid = (N + 7) / 8;
                if (M == 1)      k_ink_gemv_f32<1><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
                else if (M <= 4) k_ink_gemv_f32<4><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
                else if (M <= 8) k_ink_gemv_f32<8><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
                else             k_ink_gemv_f32<16><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
            } else {
                const float alpha = 1.0f, beta = 0.0f;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                    W, CUDA_R_32F, K, X, CUDA_R_32F, K, &beta, Y, CUDA_R_32F, N,
                    cur_cc >= 80 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        } else if (ti.fmt == FMT_Q8ROW) {
            if (M > 16 && cur_cc == 61 && (K % 4 == 0) && (N % 4 == 0) && N_override == 0) {
                int8_t* x_i8 = reinterpret_cast<int8_t*>(d_x_bf16);
                float*  x_sc = d_ctrl_gain;
                if (!reuse_x_bf16) {
                    k_ink_quant_x_row_i8<<<M, 256, 0, st>>>(X, x_i8, x_sc, M, K);
                }
                const float*  w_sc = reinterpret_cast<const float*>(W);
                const int8_t* w_i8 = reinterpret_cast<const int8_t*>(W + (size_t)ti.rows() * 4);
                const int32_t alpha_i = 1, beta_i = 0;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha_i,
                    w_i8, CUDA_R_8I, K,
                    x_i8, CUDA_R_8I, K,
                    &beta_i, Y, CUDA_R_32I, N,
                    CUBLAS_COMPUTE_32I,
                    (cur_cc >= 75) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
                const size_t tot = (size_t)M * N;
                k_ink_i32_to_f32_scale<<<(tot + 255) / 256, 256, 0, st>>>(
                    reinterpret_cast<int32_t*>(Y), Y, w_sc, x_sc, M, N);
                return;
            }
            const size_t cap = (size_t)kInkDsparkInter * kInkHidden;
            const size_t max_tile = (cur_cc < 70) ? (cap / 3) : cap;
            if (M <= 16 || (size_t)N * K > max_tile) {
                InkMultiProj mp{};
                mp.W[0] = W; mp.Y[0] = Y; mp.N[0] = N; mp.cumN[0] = 0; mp.cumN[1] = N; mp.count = 1;
                if ((K % 64) == 0 && K <= kInkQDim && M <= 16 && (cur_cc == 61 || cur_cc >= 75)) {
                    const int grid = (N + 7) / 8;
                    if (M == 1) {
                        k_ink_gemv_multi_q8row_m1<<<grid, 128, 0, st>>>(mp, X, K);
                        return;
                    }
                    const int nblk = M * (K / 64);
                    if ((size_t)nblk <= (size_t)cur_moe_ws->cap_M * (kInkHidden / 64)) {
                        k_ink_quant_q8_1<<<(nblk + 7) / 8, 256, 0, st>>>(X, cur_moe_ws->xq_sk, M, K);
                        if (M <= 4)      k_ink_gemv_multi_q8row_preq<4><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                        else if (M <= 8) k_ink_gemv_multi_q8row_preq<8><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                        else             k_ink_gemv_multi_q8row_preq<16><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                        return;
                    }
                }
                const int grid = (N + 3) / 4;
                if (M == 1)      k_ink_gemv_multi_q8row<1><<<grid, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 4) k_ink_gemv_multi_q8row<4><<<grid, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 8) k_ink_gemv_multi_q8row<8><<<grid, 128, 0, st>>>(mp, X, M, K);
                else             k_ink_gemv_multi_q8row<16><<<grid, 128, 0, st>>>(mp, X, M, K);
            } else {
                __nv_bfloat16* w_dst = (cur_cc < 70) ? (d_w_scratch + 2 * (cap / 3)) : d_w_scratch;
                dim3 dg((K + 255) / 256, N);
                k_ink_q8row_to_bf16<<<dg, 256, 0, st>>>(W, w_dst, N, K);
                gemm_bf16_compat(w_dst, X, Y, M, N, K, N, reuse_x_bf16, true);
            }
        } else {
            const size_t cap = (size_t)kInkDsparkInter * kInkHidden;
            const size_t max_tile = (cur_cc < 70) ? (cap / 3) : cap;
            const int tile_n = std::min(N, (int)(max_tile / (size_t)K));
            const size_t row_b = (size_t)(K / 256) * ink_sb_bytes(ti.fmt);
            __nv_bfloat16* w_dst = (cur_cc < 70) ? (d_w_scratch + 2 * (cap / 3)) : d_w_scratch;
            for (int n0 = 0; n0 < N; n0 += tile_n) {
                const int nt = std::min(tile_n, N - n0);
                dim3 dg((K + 255) / 256, nt);
                const uint8_t* Wsub = W + (size_t)n0 * row_b;
                if (ti.fmt == FMT_S5_K)      k_ink_sk_to_bf16<FMT_S5_K><<<dg, 256, 0, st>>>(Wsub, w_dst, nt, K);
                else if (ti.fmt == FMT_S6_K) k_ink_sk_to_bf16<FMT_S6_K><<<dg, 256, 0, st>>>(Wsub, w_dst, nt, K);
                else                         k_ink_sk_to_bf16<FMT_S8_K><<<dg, 256, 0, st>>>(Wsub, w_dst, nt, K);
                gemm_bf16_compat(w_dst, X, Y + n0, M, nt, K, N, n0 > 0, true);
            }
        }
    }

    // Strictly O(1)-VRAM prefill attention scratch (fixed KV_TILE = 2048 -> ~114 MiB forever, fixes #6609 & #6605!).
    // Non-fatal: returns false if VRAM is tight so caller seamlessly falls back to zero-VRAM k_ink_flash_gqa_gqa3!
    bool ensure_pf_attn(int /*K_len*/) {
        constexpr int Q_CH = 256;
        if (d_pf_q && pf_kv_cap >= kInkPfKvTile) return true;
        INK_CK(cudaStreamSynchronize(st));
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_pf_q); fr(d_pf_k); fr(d_pf_v); fr(d_pf_p);
        fr(d_pf_s); fr(d_pf_o); fr(d_pf_ot); fr(d_pf_m); fr(d_pf_l);
        pf_kv_cap = 0;
        const size_t q_b  = (size_t)kInkHeads * max_batch * kInkHeadDim * sizeof(__nv_bfloat16);
        const size_t kv_b = (size_t)kInkHeads * kInkPfKvTile * kInkHeadDim * sizeof(__nv_bfloat16);
        const size_t s_b  = (size_t)kInkHeads * Q_CH * kInkPfKvTile * sizeof(float);
        const size_t p_b  = (size_t)kInkHeads * Q_CH * kInkPfKvTile * sizeof(__nv_bfloat16);
        const size_t o_b  = (size_t)kInkHeads * Q_CH * kInkHeadDim * sizeof(float);
        const size_t ml_b = (size_t)kInkHeads * Q_CH * sizeof(float);
        const size_t need = q_b + 2 * kv_b + s_b + p_b + 2 * o_b + 2 * ml_b;
        size_t vf = 0, vt = 0;
        const int dev = m.ls_active ? m.stage_gpu[cur_stage] : m.primary_gpu;
        if (ink_cuda_mem_info(&vf, &vt, dev) != cudaSuccess || vf < need + (192ull << 20)) return false;
        if (cudaMalloc(&d_pf_q, q_b) != cudaSuccess ||
            cudaMalloc(&d_pf_k, kv_b) != cudaSuccess ||
            cudaMalloc(&d_pf_v, kv_b) != cudaSuccess ||
            cudaMalloc(&d_pf_s, s_b) != cudaSuccess ||
            cudaMalloc(&d_pf_p, p_b) != cudaSuccess ||
            cudaMalloc(&d_pf_o, o_b) != cudaSuccess ||
            cudaMalloc(&d_pf_ot, o_b) != cudaSuccess ||
            cudaMalloc(&d_pf_m, ml_b) != cudaSuccess ||
            cudaMalloc(&d_pf_l, ml_b) != cudaSuccess) {
            cudaGetLastError();
            fr(d_pf_q); fr(d_pf_k); fr(d_pf_v); fr(d_pf_p);
            fr(d_pf_s); fr(d_pf_o); fr(d_pf_ot); fr(d_pf_m); fr(d_pf_l);
            return false;
        }
        pf_kv_cap = kInkPfKvTile;
        return true;
    }

    // ---- Execute 1 virtual visit (visit in 0..47, phys layer ip in 0..23) ----
    // mode: 0 = single-slot sequential (prefill or M=1 decode), 1 = batched multi-slot (B=1..64, 1 tok/slot), 2 = Tree-32 verify
    void run_visit(int visit, int ip, const DepthStatePtrs* bank_stp, int M, int mode, int slot0, int p0_val,
                   int& n_blocks, const float* (&block_ptrs)[8]) {
        const std::string p = "model.llm.layers." + std::to_string(ip);
        const bool is_swa = m.man.c.is_swa[ip] != 0;
        const bool has_msg = (bank_stp != nullptr);
        const bool exact_x = (mode == 2);
        const size_t n_hid = (size_t)M * kInkHidden;

        // 1. Attention-Residual Depth Mix (+ Fused Looped Pre-Attn Msg + CatRMS -> eliminates 48+72 launches!)
        prof_mark_start();
        const float* nw_q = (const float*)m.trunk_ptr_stage(cur_stage, is_swa ? (p + ".attn_norm.weight") : (p + ".q_input_norm.weight"));
        const float* nw_k = (const float*)m.trunk_ptr_stage(cur_stage, is_swa ? (p + ".attn_norm.weight") : (p + ".k_input_norm.weight"));
        const float* nw_v = (const float*)m.trunk_ptr_stage(cur_stage, is_swa ? (p + ".attn_norm.weight") : (p + ".v_input_norm.weight"));

        const float* in_q = d_inp0;
        const float* in_k = d_inp0;
        const float* in_v = d_inp0;
        if (ip > 0) {
            DepthStatePtrs stp{};
            for (int b = 0; b < n_blocks; ++b) stp.p[b] = block_ptrs[b];
            stp.p[n_blocks] = d_cur;
            stp.count = n_blocks + 1;
            const float* q_res = (const float*)m.trunk_ptr_stage(cur_stage, "model.llm.attn_res.attn_queries." + std::to_string(ip - 1));
            const int nq = is_swa ? 1 : 3;
            if (!has_msg) {
                k_ink_attn_res_mix<<<M, 256, 0, st>>>(stp, q_res, d_mixed, nq, M, kInkHidden, nw_q, is_swa ? nullptr : nw_k, is_swa ? nullptr : nw_v);
                in_q = d_mixed;
                in_k = is_swa ? d_mixed : (d_mixed + 1 * n_hid);
                in_v = is_swa ? d_mixed : (d_mixed + 2 * n_hid);
            } else {
                k_ink_attn_res_3x_cat_rms<<<dim3(M, 4), 256, 0, st>>>(
                    stp, q_res, is_swa, *bank_stp, m.d_msg_queries[ip - m.man.c.core_start],
                    d_mixed, d_msgs, (M <= 16) ? d_ctrl_cat : nullptr, M, kInkHidden);
                const float* src0 = d_mixed;
                const float* src1 = d_mixed + 1 * n_hid;
                const float* src2 = d_mixed + 2 * n_hid;
                const float* wd0 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.attn.down_weight");
                const float* wg0 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.attn.gain_weight");
                const float* wf0 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.attn.forget_weight");
                const float* wd1 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.k.down_weight");
                const float* wg1 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.k.gain_weight");
                const float* wf1 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.k.forget_weight");
                const float* wd2 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.v.down_weight");
                const float* wg2 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.v.gain_weight");
                const float* wf2 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.v.forget_weight");
                if (M <= 16) {
                    constexpr int D = kInkHidden, D2 = 2 * kInkHidden;
                    if (M == 1) {
                        k_ink_ctrl_3x_down_smallM<1><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<1><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    } else if (M <= 4) {
                        k_ink_ctrl_3x_down_smallM<4><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<4><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    } else if (M <= 8) {
                        k_ink_ctrl_3x_down_smallM<8><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<8><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    } else {
                        k_ink_ctrl_3x_down_smallM<16><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<16><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    }
                    k_ink_rms_norm_3x<<<dim3(M, 3), 256, 0, st>>>(d_inp0, d_inp1, d_inp2, nw_q, nw_k, nw_v, M, D);
                } else {
                    run_loop_control_norm(src0, d_msgs + 0 * n_hid, wd0, wg0, wf0, nw_q, d_inp0, M, false, exact_x);
                    run_loop_control_norm(src1, d_msgs + 1 * n_hid, wd1, wg1, wf1, nw_k, d_inp1, M, false, exact_x);
                    run_loop_control_norm(src2, d_msgs + 2 * n_hid, wd2, wg2, wf2, nw_v, d_inp2, M, false, exact_x);
                }
                in_q = d_inp0; in_k = d_inp1; in_v = d_inp2;
            }
        } else {
            k_ink_rms_norm<<<M, 256, 0, st>>>(d_cur, nw_q, d_inp0, nullptr, M, kInkHidden);
            in_q = in_k = in_v = d_inp0;
        }

        // 3. Fused Q, K, V, R, Gate_Attn, Sinks projections (1 launch on SWA-no-msg at M<=16 -> 1,500+ GB/s!)
        prof_mark_end(0); prof_mark_start();
        const bool share_qkv = (in_q == in_k && in_q == in_v);
        if (M <= 16 && share_qkv) {
            const std::string names[5] = {
                p + ".attn.wq_du.weight", p + ".attn.wr_du.weight", p + ".attn.gate_attn.weight",
                p + ".attn.wk_dv.weight", p + ".attn.wv_dv.weight"
            };
            float* ys[5] = { d_q, d_r, d_gate_attn, d_k_raw, d_v_raw };
            linear_multi(names, ys, 5, in_q, M, exact_x);
            linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M, 0, false, exact_x);
        } else if (M <= 16) {
            const std::string names[3] = {
                p + ".attn.wq_du.weight", p + ".attn.wr_du.weight", p + ".attn.gate_attn.weight"
            };
            float* ys[3] = { d_q, d_r, d_gate_attn };
            linear_multi(names, ys, 3, in_q, M, exact_x);
            linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M, 0, false, exact_x);
            linear(p + ".attn.wk_dv.weight", in_k, d_k_raw, M, 0, false, exact_x);
            linear(p + ".attn.wv_dv.weight", in_v, d_v_raw, M, 0, false, exact_x);
        } else {
            linear(p + ".attn.wq_du.weight", in_q, d_q, M, 0, false);
            linear(p + ".attn.wr_du.weight", in_q, d_r, M, 0, true);
            linear(p + ".attn.gate_attn.weight", in_q, d_gate_attn, M, 0, true);
            linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M, 0, false);
            linear(p + ".attn.wk_dv.weight", in_k, d_k_raw, M, 0, share_qkv);
            linear(p + ".attn.wv_dv.weight", in_v, d_v_raw, M, 0, share_qkv);
        }

        // 4. Dynamic K/V ShortConv tap projections on global layers
        float* k_dyn_ptr = nullptr;
        float* v_dyn_ptr = nullptr;
        if (!is_swa) {
            linear(p + ".attn.k_sconv.weight_proj", in_k, d_k_dyn, M);
            linear(p + ".attn.v_sconv.weight_proj", in_v, d_v_dyn, M);
            k_dyn_ptr = d_k_dyn;
            v_dyn_ptr = d_v_dyn;
        }
        prof_mark_end(1); prof_mark_start();
        auto run_sconv_uniform = [&](const float* src, const float* sw, const float* dyn, float* dst, int C, int ch_off, int nh, bool add_to_out = false) {
            float* hbase = d_sconv_pool[visit] + (size_t)ch_off * 3 * max_slots;
            if (mode == 0) {
                float* hslot = hbase + (size_t)slot0 * 3 * C;
                if (M > 16) {
                    dim3 g((C + 255) / 256, (M + 15) / 16);
                    k_ink_sconv4_seq_par<<<g, 256, 0, st>>>(src, sw, dyn, hslot, dst, M, C, nh, add_to_out);
                    k_ink_sconv4_save_hist<<<(C + 255) / 256, 256, 0, st>>>(src, hslot, M, C);
                } else {
                    k_ink_sconv4_seq<<<(C + 255) / 256, 256, 0, st>>>(src, sw, dyn, hslot, dst, M, C, nh, true, add_to_out);
                }
            } else if (mode == 1) {
                dim3 g((C + 255) / 256, M);
                k_ink_sconv4_batched<<<g, 256, 0, st>>>(src, sw, dyn, hbase, d_slots, dst, M, C, nh, add_to_out);
            } else {
                dim3 g((C + 255) / 256, M);
                k_ink_sconv4_tree<<<g, 256, 0, st>>>(
                    src, sw, dyn, hbase + (size_t)slot0 * 3 * C, d_sconv_lag, dst,
                    d_stage_sconv + (size_t)visit * tree_cap * kInkSconvCh, ch_off, M, C, nh, add_to_out);
            }
        };

        // 5-6. FUSED Post-QKV (K/V ShortConv4 + Q/K/V/R Clip + V RMSNorm + Global Tau + KV Write/Stage) in 1 launch for decode/batched/tree!
        const auto& ti_qc = m.tinfo(p + ".attn.q_l2_clip.scale");
        const auto& ti_rc = m.tinfo(p + ".attn.rel_l2_clip.scale");
        const float* p_qc = (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.q_l2_clip.scale");
        const float* p_rc = (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.rel_l2_clip.scale");
        const int qc_elems = (int)(ti_qc.bytes / 4);
        const int rc_elems = (int)(ti_rc.bytes / 4);
        if ((mode == 0 && M == 1) || mode == 1 || mode == 2) {
            float* hk_pool = d_sconv_pool[visit];
            float* hv_pool = d_sconv_pool[visit] + (size_t)kInkKVDim * 3 * max_slots;
            k_ink_post_qkv_fused<<<dim3(64, M), 128, 0, st>>>(
                d_q, p_qc, qc_elems,
                d_k_raw, d_v_raw, m.d_sconv_k[ip], m.d_sconv_v[ip],
                k_dyn_ptr, v_dyn_ptr, hk_pool, hv_pool,
                d_k, d_v, d_r, p_rc, rc_elems,
                (mode == 1) ? d_slots : nullptr, slot0, d_pos,
                (mode == 2) ? d_sconv_lag : nullptr,
                (mode == 2) ? (d_stage_k + (size_t)visit * tree_cap * kInkKVDim) : nullptr,
                (mode == 2) ? (d_stage_v + (size_t)visit * tree_cap * kInkKVDim) : nullptr,
                (mode == 2) ? (d_stage_sconv + (size_t)visit * tree_cap * kInkSconvCh) : nullptr,
                kv_desc[visit], !is_swa, mode, M);
        } else {
            run_sconv_uniform(d_k_raw, m.d_sconv_k[ip], k_dyn_ptr, d_k, kInkKVDim, 0, kInkKVHeads);
            run_sconv_uniform(d_v_raw, m.d_sconv_v[ip], v_dyn_ptr, d_v, kInkKVDim, kInkKVDim, kInkKVHeads);
            k_ink_qkvr_clip_tau<<<dim3(64, M), 128, 0, st>>>(
                d_q, p_qc, qc_elems,
                d_k, d_v, d_r, p_rc, rc_elems,
                d_pos, !is_swa, M);
            k_ink_kv_write<<<dim3(kInkKVHeads, M), 128, 0, st>>>(d_k, d_v, nullptr, slot0, d_pos, kv_desc[visit], M);
        }

        // 7. RelPos Table + Fused Sink Flash-GQA (Adaptive Split-K 16/32 for M==1 (#6591), O(1)-VRAM KV-Tiled TC for prefill M>=128 (#6609/#6605)!)
        prof_mark_end(2); prof_mark_start();
        {
            const int cur_cc = m.ls_active ? m.stage_cc[cur_stage] : m.sm_cc;
            dim3 gr(kInkHeads, M);
            k_ink_rel_table<<<gr, 256, 0, st>>>(d_r, (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.rel_logits_proj.proj"), d_rel_table, M);
            if (M <= 16 || mode == 2) {
                const int n_sp = (M == 1) ? (is_swa ? 16 : kInkAttnMaxSplits) : kInkAttnSplits;
                dim3 gsp(kInkKVHeads, n_sp, M);
                k_ink_flash_gqa_splitk<<<gsp, 128, 0, st>>>(
                    d_q, d_rel_table, d_sinks,
                    kv_desc[visit], (mode == 1) ? d_slots : nullptr, slot0, d_pos,
                    (mode == 2) ? d_k : nullptr, (mode == 2) ? d_v : nullptr,
                    (mode == 2) ? d_tree_anc : nullptr, (mode == 2) ? d_tree_dep : nullptr,
                    p0_val, d_attn_pm, d_attn_pl, d_attn_po, M, n_sp);
                k_ink_flash_gqa_merge<<<gr, 128, 0, st>>>(d_attn_pm, d_attn_pl, d_attn_po, d_gate_attn, d_attn_out, M, n_sp);
            } else if (mode == 0 && M >= 128 && (cur_cc >= 70 || cur_cc == 60) && ensure_pf_attn(kInkPfKvTile)) {
                // O(1)-VRAM KV-Tiled Online-Softmax Tensor Core Prefill Attention (fixed <= ~114 MiB even at 256K+ ctx!)
                const bool use_fp16_pf = (cur_cc < 80);
                const cudaDataType_t pf_dt = use_fp16_pf ? CUDA_R_16F : CUDA_R_16BF;
                const cublasGemmAlgo_t pf_algo = (cur_cc >= 70) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
                const int k_start = is_swa ? std::max(0, p0_val - kInkSwaWin + 1) : 0;
                const int K_len = (p0_val + M) - k_start;
                if (use_fp16_pf) k_ink_pf_pack_q<true><<<dim3(kInkHeads, M), 128, 0, st>>>(d_q, d_pf_q, M);
                else             k_ink_pf_pack_q<false><<<dim3(kInkHeads, M), 128, 0, st>>>(d_q, d_pf_q, M);

                constexpr int Q_CH = 256;
                const float sc_qk = 0.08838834764831845f;
                const float one = 1.0f, zero = 0.0f;

                if (K_len <= kInkPfKvTile) {
                    if (use_fp16_pf) k_ink_pf_pack_kv<true><<<dim3(kInkKVHeads, K_len), 128, 0, st>>>(kv_desc[visit], slot0, k_start, K_len, d_pf_k, d_pf_v);
                    else             k_ink_pf_pack_kv<false><<<dim3(kInkKVHeads, K_len), 128, 0, st>>>(kv_desc[visit], slot0, k_start, K_len, d_pf_k, d_pf_v);
                    for (int q0 = 0; q0 < M; q0 += Q_CH) {
                        const int C = std::min(Q_CH, M - q0);
                        const int k_sub_beg = is_swa ? std::max(k_start, p0_val + q0 - kInkSwaWin + 1) : k_start;
                        const int k_sub_end = p0_val + q0 + C;
                        const int k_off = k_sub_beg - k_start;
                        const int K_sub = std::max(1, k_sub_end - k_sub_beg);
                        INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            K_sub, C, kInkHeadDim, &sc_qk,
                            d_pf_k + (size_t)k_off * kInkHeadDim, pf_dt, kInkHeadDim, (long long)K_len * kInkHeadDim,
                            d_pf_q + (size_t)q0 * kInkHeadDim, pf_dt, kInkHeadDim, (long long)M * kInkHeadDim,
                            &zero, d_pf_s, CUDA_R_32F, K_sub, (long long)C * K_sub,
                            kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                        if (use_fp16_pf) {
                            k_ink_pf_softmax_rel_sink<true><<<dim3(C, kInkHeads), 256, 0, st>>>(
                                d_pf_s, d_rel_table, d_sinks, d_pf_p, q0, C, k_sub_beg, K_sub, p0_val, is_swa);
                        } else {
                            k_ink_pf_softmax_rel_sink<false><<<dim3(C, kInkHeads), 256, 0, st>>>(
                                d_pf_s, d_rel_table, d_sinks, d_pf_p, q0, C, k_sub_beg, K_sub, p0_val, is_swa);
                        }
                        INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                            kInkHeadDim, C, K_sub, &one,
                            d_pf_v + (size_t)k_off * kInkHeadDim, pf_dt, kInkHeadDim, (long long)K_len * kInkHeadDim,
                            d_pf_p, pf_dt, K_sub, (long long)C * K_sub,
                            &zero, d_pf_o, CUDA_R_32F, kInkHeadDim, (long long)C * kInkHeadDim,
                            kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                        k_ink_pf_unpack_o_gate<<<dim3(kInkHeads, C), 128, 0, st>>>(d_pf_o, d_gate_attn, d_attn_out, q0, C);
                    }
                } else {
                    // Long-context KV-Tiled Online-Softmax TC Attention (K_len > 2048 -> O(1) 114 MiB VRAM forever!)
                    float* d_alpha_tmp = d_ctrl_gain;  // [max_batch * 2304] >> 24 * 256 floats!
                    for (int q0 = 0; q0 < M; q0 += Q_CH) {
                        const int C = std::min(Q_CH, M - q0);
                        const int k_sub_beg = is_swa ? std::max(k_start, p0_val + q0 - kInkSwaWin + 1) : k_start;
                        const int k_sub_end = p0_val + q0 + C;
                        k_ink_pf_online_init<<<dim3(kInkHeads, C), 128, 0, st>>>(d_sinks, d_pf_m, d_pf_l, d_pf_o, q0, C);
                        for (int kt = k_sub_beg; kt < k_sub_end; kt += kInkPfKvTile) {
                            const int K_t = std::min(kInkPfKvTile, k_sub_end - kt);
                            const bool is_last = (kt + K_t >= k_sub_end);
                            if (use_fp16_pf) k_ink_pf_pack_kv<true><<<dim3(kInkKVHeads, K_t), 128, 0, st>>>(kv_desc[visit], slot0, kt, K_t, d_pf_k, d_pf_v);
                            else             k_ink_pf_pack_kv<false><<<dim3(kInkKVHeads, K_t), 128, 0, st>>>(kv_desc[visit], slot0, kt, K_t, d_pf_k, d_pf_v);
                            INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                                cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                K_t, C, kInkHeadDim, &sc_qk,
                                d_pf_k, pf_dt, kInkHeadDim, (long long)K_t * kInkHeadDim,
                                d_pf_q + (size_t)q0 * kInkHeadDim, pf_dt, kInkHeadDim, (long long)M * kInkHeadDim,
                                &zero, d_pf_s, CUDA_R_32F, K_t, (long long)C * K_t,
                                kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                            if (use_fp16_pf) {
                                k_ink_pf_softmax_tile_online<true><<<dim3(C, kInkHeads), 256, 0, st>>>(
                                    d_pf_s, d_rel_table, d_pf_p, d_pf_m, d_pf_l, d_alpha_tmp, q0, C, kt, K_t, p0_val, is_swa);
                            } else {
                                k_ink_pf_softmax_tile_online<false><<<dim3(C, kInkHeads), 256, 0, st>>>(
                                    d_pf_s, d_rel_table, d_pf_p, d_pf_m, d_pf_l, d_alpha_tmp, q0, C, kt, K_t, p0_val, is_swa);
                            }
                            INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                                cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                kInkHeadDim, C, K_t, &one,
                                d_pf_v, pf_dt, kInkHeadDim, (long long)K_t * kInkHeadDim,
                                d_pf_p, pf_dt, K_t, (long long)C * K_t,
                                &zero, d_pf_ot, CUDA_R_32F, kInkHeadDim, (long long)C * kInkHeadDim,
                                kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                            k_ink_pf_accum_tile<<<dim3(kInkHeads, C), 128, 0, st>>>(
                                d_pf_ot, d_alpha_tmp, d_pf_l, d_pf_o, d_gate_attn, d_attn_out, q0, C, is_last);
                        }
                    }
                }
            } else {
                dim3 g3(kInkKVHeads, M);
                k_ink_flash_gqa_gqa3<<<g3, 128, 0, st>>>(
                    d_q, d_rel_table, d_sinks, d_gate_attn,
                    kv_desc[visit], (mode == 1) ? d_slots : nullptr, slot0, d_pos,
                    d_attn_out, M);
            }
        }
        prof_mark_end(3); prof_mark_start();

        // 8. Output projection wo + FUSED post-attn ShortConv + residual add / block boundary!
        linear(p + ".attn.wo_ud.weight", d_attn_out, d_wo_raw, M, 0, false, (mode == 2));
        prof_mark_end(4); prof_mark_start();
        if (ip % m.man.c.block_size == 0) {
            INK_CK(cudaMemcpyAsync(d_blocks[n_blocks], d_cur, n_hid * 4, cudaMemcpyDeviceToDevice, st));
            block_ptrs[n_blocks] = d_blocks[n_blocks];
            ++n_blocks;
            run_sconv_uniform(d_wo_raw, m.d_sconv_attn[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim, 0, false);
        } else {
            run_sconv_uniform(d_wo_raw, m.d_sconv_attn[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim, 0, true);
        }

        // 9. MoE Sublayer: Fused Pre-MLP Mix+CatRMS -> SITU MoE -> Fused Reduce8 + Post-MLP ShortConv4 + Add!
        const float* mlp_nw = (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp_norm.weight");
        {
            DepthStatePtrs stp{};
            for (int b = 0; b < n_blocks; ++b) stp.p[b] = block_ptrs[b];
            stp.p[n_blocks] = d_cur;
            stp.count = n_blocks + 1;
            const float* q_mlp = (const float*)m.trunk_ptr_stage(cur_stage, "model.llm.attn_res.mlp_queries." + std::to_string(ip));
            if (has_msg) {
                k_ink_mlp_res_cat_rms<<<M, 256, 0, st>>>(stp, q_mlp, d_msgs + 3 * n_hid, d_moe_in, d_ctrl_cat, M, kInkHidden);
                run_loop_control_norm(
                    d_moe_in, d_msgs + 3 * n_hid,
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.down_weight"),
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.gain_weight"),
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.forget_weight"),
                    mlp_nw, d_moe_in, M, /*cat_ready=*/true, exact_x);
            } else {
                k_ink_attn_res_mix<<<M, 256, 0, st>>>(stp, q_mlp, d_moe_in, 1, M, kInkHidden, mlp_nw);
            }
        }
        prof_mark_end(0); prof_mark_start();

        if (m.ep2_active && m.layer_has_peer[ip]) {
            launch_ink_moe_router(
                st, cublas,
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.weight"),
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.bias"),
                d_moe_in, M, *cur_moe_ws, m.man.c.route_scale, (mode == 2));
            const size_t x_bytes   = (size_t)M * kInkHidden * sizeof(float);
            const size_t idx_bytes = (size_t)M * kInkTopK * sizeof(int32_t);
            const size_t w_bytes   = (size_t)M * kInkTopK * sizeof(float);
            if (m.ep2_p2p) {
                INK_CK(cudaEventRecord(ev_moe_ready, st));
                INK_CK(cudaSetDevice(m.peer_gpu));
                INK_CK(cudaStreamWaitEvent(peer_st, ev_moe_ready, 0));
                INK_CK(cudaMemcpyPeerAsync(d_peer_x, m.peer_gpu, d_moe_in, m.primary_gpu, x_bytes, peer_st));
                INK_CK(cudaMemcpyPeerAsync(peer_ws.topk_idx, m.peer_gpu, moe_ws.topk_idx, m.primary_gpu, idx_bytes, peer_st));
                INK_CK(cudaMemcpyPeerAsync(peer_ws.topk_w, m.peer_gpu, moe_ws.topk_w, m.primary_gpu, w_bytes, peer_st));
                launch_ink_moe_dispatch_only(peer_st, m.d_sec4_peer, m.d_experts_peer[ip], m.moe_info_peer[ip],
                                             d_peer_x, M, peer_ws, (mode == 2), false);
                k_ink_moe_reduce8_tier0<<<dim3((kInkHidden + 255) / 256, M), 256, 0, peer_st>>>(
                    m.d_experts_peer[ip], peer_ws.topk_idx, peer_ws.y_pair, d_peer_y, M);
                INK_CK(cudaMemcpyPeerAsync(d_peer_y_rx, m.primary_gpu, d_peer_y, m.peer_gpu, x_bytes, peer_st));
                INK_CK(cudaEventRecord(ev_peer_done, peer_st));
                INK_CK(cudaSetDevice(m.primary_gpu));
            } else {
                INK_CK(cudaMemcpyAsync(h_ep_x, d_moe_in, x_bytes, cudaMemcpyDeviceToHost, st));
                INK_CK(cudaMemcpyAsync(h_ep_idx, moe_ws.topk_idx, idx_bytes, cudaMemcpyDeviceToHost, st));
                INK_CK(cudaMemcpyAsync(h_ep_w, moe_ws.topk_w, w_bytes, cudaMemcpyDeviceToHost, st));
                INK_CK(cudaEventRecord(ev_moe_ready, st));
                INK_CK(cudaSetDevice(m.peer_gpu));
                INK_CK(cudaStreamWaitEvent(peer_st, ev_moe_ready, 0));
                INK_CK(cudaMemcpyAsync(d_peer_x, h_ep_x, x_bytes, cudaMemcpyHostToDevice, peer_st));
                INK_CK(cudaMemcpyAsync(peer_ws.topk_idx, h_ep_idx, idx_bytes, cudaMemcpyHostToDevice, peer_st));
                INK_CK(cudaMemcpyAsync(peer_ws.topk_w, h_ep_w, w_bytes, cudaMemcpyHostToDevice, peer_st));
                launch_ink_moe_dispatch_only(peer_st, m.d_sec4_peer, m.d_experts_peer[ip], m.moe_info_peer[ip],
                                             d_peer_x, M, peer_ws, (mode == 2), false);
                k_ink_moe_reduce8_tier0<<<dim3((kInkHidden + 255) / 256, M), 256, 0, peer_st>>>(
                    m.d_experts_peer[ip], peer_ws.topk_idx, peer_ws.y_pair, d_peer_y, M);
                INK_CK(cudaMemcpyAsync(h_ep_y, d_peer_y, x_bytes, cudaMemcpyDeviceToHost, peer_st));
                INK_CK(cudaEventRecord(ev_peer_done, peer_st));
                INK_CK(cudaSetDevice(m.primary_gpu));
            }
            launch_ink_moe_dispatch_only(st, m.d_sec4, m.d_experts[ip], m.moe_info[ip],
                                         d_moe_in, M, moe_ws, (mode == 2), false);
            k_ink_moe_reduce8_tier0<<<dim3((kInkHidden + 255) / 256, M), 256, 0, st>>>(
                m.d_experts[ip], moe_ws.topk_idx, moe_ws.y_pair, d_moe_raw, M);
            INK_CK(cudaStreamWaitEvent(st, ev_peer_done, 0));
            if (!m.ep2_p2p) {
                INK_CK(cudaMemcpyAsync(d_peer_y_rx, h_ep_y, x_bytes, cudaMemcpyHostToDevice, st));
            }
            const size_t n_add = (size_t)M * kInkHidden;
            k_ink_add_f32<<<(n_add + 255) / 256, 256, 0, st>>>(d_moe_raw, d_peer_y_rx, n_add);
            prof_mark_end(5); prof_mark_start();
            run_sconv_uniform(d_moe_raw, m.d_sconv_mlp[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim + kInkHidden, 0, true);
        } else if (m.layer_has_host[ip] && m.hybrid.active && cur_stage == 0) {
            launch_ink_moe_hybrid(
                st, cublas, m.d_sec4, m.d_experts[ip], m.moe_info[ip], ip, m.hybrid,
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.weight"),
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.bias"),
                d_moe_in, d_moe_raw, M, *cur_moe_ws, m.man.c.route_scale, (mode == 2));
            prof_mark_end(5); prof_mark_start();
            run_sconv_uniform(d_moe_raw, m.d_sconv_mlp[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim + kInkHidden, 0, true);
        } else {
            const bool fuse_moe_sconv = ((mode == 0 && M == 1) || mode == 1 || mode == 2);
            launch_ink_moe(
                st, cublas, m.sec4_stage(cur_stage), m.d_experts[ip], m.moe_info[ip],
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.weight"),
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.bias"),
                d_moe_in, d_moe_raw, M, *cur_moe_ws, m.man.c.route_scale, (mode == 2), !fuse_moe_sconv);
            prof_mark_end(5); prof_mark_start();
            if (fuse_moe_sconv) {
                constexpr int mlp_ch_off = 2 * kInkKVDim + kInkHidden;
                float* hbase = d_sconv_pool[visit] + (size_t)mlp_ch_off * 3 * max_slots;
                if (mode == 0 && M == 1) {
                    k_ink_moe_reduce8_sconv_m1<<<(kInkHidden + 255) / 256, 256, 0, st>>>(
                        cur_moe_ws->y_pair, m.d_sconv_mlp[ip], hbase + (size_t)slot0 * 3 * kInkHidden, d_cur);
                } else if (mode == 1) {
                    k_ink_moe_reduce8_sconv_batched<<<dim3((kInkHidden + 255) / 256, M), 256, 0, st>>>(
                        cur_moe_ws->y_pair, m.d_sconv_mlp[ip], hbase, d_slots, d_cur, M);
                } else {
                    k_ink_moe_reduce8_sconv_tree<<<(kInkHidden + 127) / 128, 128, 0, st>>>(
                        cur_moe_ws->y_pair, m.d_sconv_mlp[ip], hbase + (size_t)slot0 * 3 * kInkHidden,
                        d_sconv_lag, d_stage_sconv + (size_t)visit * tree_cap * kInkSconvCh,
                        mlp_ch_off, d_cur, M);
                }
            } else {
                run_sconv_uniform(d_moe_raw, m.d_sconv_mlp[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim + kInkHidden, 0, true);
            }
        }
        prof_mark_end(2);
    }

    // ---- Full 48-Visit Body (Capturable inside a SINGLE CUDA Graph on single GPU, or Pipeline Layer-Split on 2..4 GPUs!) ----
    void emit_48_visits(int M, int mode, int slot0, int tree_p0, bool want_taps) {
        if (m.ls_active && cur_stage != 0) bind_stage(0);
        const size_t n_hid = (size_t)M * kInkHidden;
        const auto& ei = m.tinfo("model.llm.embed.weight");
        k_ink_embed_norm<<<M, 256, 0, st>>>(
            m.d_trunk + ei.off, ei.fmt,
            (const float*)m.trunk_ptr("model.llm.embed_norm.weight"),
            d_tok_ids, has_img_step ? d_is_img : nullptr, has_img_step ? d_img_emb : nullptr, d_cur, M, kInkHidden);

        const float* block_ptrs[8] = {};
        int n_blocks = 0;
        int visit = 0;

        auto maybe_tap = [&](int v, const float* src) {
            if (!want_taps) return;
            float* dst_tap = nullptr;
            if (v == 5)       dst_tap = d_taps[0];
            else if (v == 23) dst_tap = d_taps[1];
            else if (v == 41) dst_tap = d_taps[2];
            if (!dst_tap) return;
            if (m.ls_active && cur_stage != 0) {
                copy_between_stages(cur_stage, 0, src, dst_tap, n_hid * 4);
                INK_CK(cudaSetDevice(m.stage_gpu[cur_stage]));
            } else {
                INK_CK(cudaMemcpyAsync(dst_tap, src, n_hid * 4, cudaMemcpyDeviceToDevice, st));
            }
        };

        // 1. Prelude: physical layers 0..5 (visits 0..5)
        for (int ip = 0; ip < m.man.c.core_start; ++ip) {
            if (m.ls_active && m.layer_owner[ip] != cur_stage) {
                switch_stage_handoff(m.layer_owner[ip], M, mode, n_blocks, block_ptrs, nullptr, 0);
            }
            maybe_tap(visit, d_cur);
            run_visit(visit, ip, nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs);
            ++visit;
        }

        // Save prelude state & prelude_blocks on Stage 0
        if (m.ls_active && cur_stage != 0) {
            switch_stage_handoff(0, M, mode, n_blocks, block_ptrs, nullptr, 0);
        }
        INK_CK(cudaMemcpyAsync(d_prelude, d_cur, n_hid * 4, cudaMemcpyDeviceToDevice, st));
        const int n_pre_blk = n_blocks;
        for (int b = 0; b < n_pre_blk; ++b) {
            INK_CK(cudaMemcpyAsync(d_prelude_blocks[b], block_ptrs[b], n_hid * 4, cudaMemcpyDeviceToDevice, st));
        }

        // 2. Looped Core: physical layers 6..17 x 3 passes (visits 6..41)
        const int n_passes = (max_passes_override > 0 && max_passes_override < m.man.c.num_passes)
                                 ? max_passes_override
                                 : m.man.c.num_passes;
        int n_bank = 0;
        for (int pass = 0; pass < n_passes; ++pass) {
            if (pass > 0) {
                if (m.ls_active && cur_stage != 0) {
                    switch_stage_handoff(0, M, mode, n_blocks, block_ptrs, nullptr, 0);
                }
                n_bank = n_blocks - (n_pre_blk + 1);
                for (int b = 0; b < n_bank; ++b) {
                    INK_CK(cudaMemcpyAsync(d_bank[b], block_ptrs[n_pre_blk + 1 + b], n_hid * 4, cudaMemcpyDeviceToDevice, st));
                }
                INK_CK(cudaMemcpyAsync(d_bank_out, d_cur, n_hid * 4, cudaMemcpyDeviceToDevice, st));
                INK_CK(cudaMemcpyAsync(d_cur, d_prelude, n_hid * 4, cudaMemcpyDeviceToDevice, st));
                n_blocks = n_pre_blk;
                for (int b = 0; b < n_pre_blk; ++b) {
                    INK_CK(cudaMemcpyAsync(d_blocks[b], d_prelude_blocks[b], n_hid * 4, cudaMemcpyDeviceToDevice, st));
                    block_ptrs[b] = d_blocks[b];
                }
            }
            DepthStatePtrs bst{};
            if (pass > 0) {
                for (int b = 0; b < n_bank; ++b) bst.p[b] = d_bank[b];
                bst.p[n_bank] = d_bank_out;
                bst.count = n_bank + 1;
            }
            for (int ip = m.man.c.core_start; ip < m.man.c.core_end; ++ip) {
                if (m.ls_active && m.layer_owner[ip] != cur_stage) {
                    switch_stage_handoff(m.layer_owner[ip], M, mode, n_blocks, block_ptrs, (pass > 0) ? &bst : nullptr, n_bank);
                }
                maybe_tap(visit, d_cur);
                run_visit(visit, ip, (pass > 0) ? &bst : nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs);
                ++visit;
            }
        }

        // 3. Coda: physical layers 18..23 (visits 42..47)
        visit = 42;
        for (int ip = m.man.c.core_end; ip < m.man.c.n_phys; ++ip) {
            if (m.ls_active && m.layer_owner[ip] != cur_stage) {
                switch_stage_handoff(m.layer_owner[ip], M, mode, n_blocks, block_ptrs, nullptr, 0);
            }
            maybe_tap(visit, d_cur);
            run_visit(visit, ip, nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs);
            ++visit;
        }

        // Return to Stage 0 for Output Depth Mix + Output RMSNorm -> d_h_norm (and tap 48!)
        if (m.ls_active && cur_stage != 0) {
            switch_stage_handoff(0, M, mode, n_blocks, block_ptrs, nullptr, 0);
        }
        {
            DepthStatePtrs stp{};
            for (int b = 0; b < n_blocks; ++b) stp.p[b] = block_ptrs[b];
            stp.p[n_blocks] = d_cur;
            stp.count = n_blocks + 1;
            k_ink_attn_res_mix<<<M, 256, 0, st>>>(stp, (const float*)m.trunk_ptr("model.llm.attn_res.output_query"), d_moe_in, 1, M, kInkHidden);
            k_ink_rms_norm<<<M, 256, 0, st>>>(d_moe_in, (const float*)m.trunk_ptr("model.llm.norm.weight"), d_h_norm, nullptr, M, kInkHidden);
            if (want_taps) INK_CK(cudaMemcpyAsync(d_taps[3], d_h_norm, n_hid * 4, cudaMemcpyDeviceToDevice, st));
        }
    }

    // ---- Launch Full 48-Visit Forward (with Safe CUDA Graph Capture -> Eager Fallback, NEVER exit(3)!) ----
    void forward_seq(const int32_t* tok_ids, int p0, int M, int slot0, bool want_taps) {
        if (m.ls_active && cur_stage != 0) bind_stage(0);
        int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        std::memcpy(slot_buf, tok_ids, (size_t)M * 4);
        int32_t* h_pos = slot_buf + M;
        for (int i = 0; i < M; ++i) h_pos[i] = p0 + i;
        INK_CK(cudaMemcpyAsync(d_tok_ids, slot_buf, (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos, h_pos, (size_t)M * 4, cudaMemcpyHostToDevice, st));

        if (use_cuda_graph && !m.ls_active && !has_img_step && max_passes_override == 0 && M == 1 && slot0 == 0) {
            GraphKey key{M, 0, want_taps ? 1 : 0, (int)kv_dtype};
            cudaGraphExec_t exec = nullptr;
            for (const auto& e : graph_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                cudaGraph_t gr = nullptr;
                cudaError_t ge = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                if (ge == cudaSuccess) {
                    emit_48_visits(M, 0, slot0, p0, want_taps);
                    ge = cudaStreamEndCapture(st, &gr);
                    if (ge == cudaSuccess && gr) {
                        ge = cudaGraphInstantiate(&exec, gr, nullptr, nullptr, 0);
                        cudaGraphDestroy(gr);
                    }
                }
                if (ge != cudaSuccess || !exec) {
                    cudaGetLastError();
                    if (gr) cudaGraphDestroy(gr);
                    use_cuda_graph = false;
                    std::fprintf(stderr, "[InkCUDA] CUDA Graph capture unavailable (err=%d): falling back to eager mode\n", (int)ge);
                    emit_48_visits(M, 0, slot0, p0, want_taps);
                    if (m.hybrid.active && m.n_host_experts > 0) m.hybrid.step_promote(m.d_sec4, m.d_experts, st);
                    return;
                }
                graph_cache.push_back({key, exec});
            }
            INK_CK(cudaGraphLaunch(exec, st));
            if (m.hybrid.active && m.n_host_experts > 0) m.hybrid.step_promote(m.d_sec4, m.d_experts, st);
            return;
        }
        emit_48_visits(M, 0, slot0, p0, want_taps);
        if (M == 1 && m.hybrid.active && m.n_host_experts > 0) m.hybrid.step_promote(m.d_sec4, m.d_experts, st);
    }

    // Speculative Tree-32 / Block-8 verify (100% CUDA-Graph capturable with safe eager fallback!)
    void forward_tree_verify(const InkTreeSpec& tree, int p0, int slot0, bool want_taps) {
        if (m.ls_active && cur_stage != 0) bind_stage(0);
        const int M = tree.M;
        int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        int32_t* h_tok = slot_buf;
        int32_t* h_pos = slot_buf + kInkMaxTree;
        uint8_t* h_bytes = reinterpret_cast<uint8_t*>(slot_buf + 2 * kInkMaxTree);
        for (int i = 0; i < M; ++i) {
            h_tok[i] = tree.tok[i];
            h_pos[i] = p0 + (int)tree.depth[i];
        }
        std::memcpy(h_bytes,       tree.sconv_lag, kInkMaxTree * 3);
        std::memcpy(h_bytes + 128, tree.anc,       kInkMaxTree * kInkMaxTreeDepth);
        std::memcpy(h_bytes + 384, tree.depth,     kInkMaxTree);

        INK_CK(cudaMemcpyAsync(d_tok_ids,   h_tok,         (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos,       h_pos,         (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_sconv_lag, h_bytes,       kInkMaxTree * 3, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_tree_anc,  h_bytes + 128, kInkMaxTree * kInkMaxTreeDepth, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_tree_dep,  h_bytes + 384, kInkMaxTree, cudaMemcpyHostToDevice, st));

        if (use_cuda_graph && !m.ls_active && max_passes_override == 0 && slot0 == 0 && M <= tree_cap && m.n_host_experts == 0) {
            GraphKey key{M, 2, want_taps ? 1 : 0, (int)kv_dtype};
            cudaGraphExec_t exec = nullptr;
            for (const auto& e : graph_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                cudaGraph_t gr = nullptr;
                cudaError_t ge = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                if (ge == cudaSuccess) {
                    emit_48_visits(M, 2, slot0, 0, want_taps);
                    ge = cudaStreamEndCapture(st, &gr);
                    if (ge == cudaSuccess && gr) {
                        ge = cudaGraphInstantiate(&exec, gr, nullptr, nullptr, 0);
                        cudaGraphDestroy(gr);
                    }
                }
                if (ge != cudaSuccess || !exec) {
                    cudaGetLastError();
                    if (gr) cudaGraphDestroy(gr);
                    use_cuda_graph = false;
                    std::fprintf(stderr, "[InkCUDA] Tree CUDA Graph capture unavailable (err=%d): falling back to eager mode\n", (int)ge);
                    emit_48_visits(M, 2, slot0, p0, want_taps);
                    return;
                }
                graph_cache.push_back({key, exec});
            }
            INK_CK(cudaGraphLaunch(exec, st));
            return;
        }
        emit_48_visits(M, 2, slot0, p0, want_taps);
    }

    // Batched multi-slot 1-token-per-slot step (B = 1..64 concurrent requests, with safe eager fallback!)
    void forward_batched(const int32_t* tok_ids, const int32_t* slot_ids, const int32_t* pos_arr, int B, bool want_taps = false) {
        if (m.ls_active && cur_stage != 0) bind_stage(0);
        int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        std::memcpy(slot_buf, tok_ids, (size_t)B * 4);
        std::memcpy(slot_buf + B, slot_ids, (size_t)B * 4);
        std::memcpy(slot_buf + 2 * B, pos_arr, (size_t)B * 4);
        INK_CK(cudaMemcpyAsync(d_tok_ids, slot_buf, (size_t)B * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_slots, slot_buf + B, (size_t)B * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos, slot_buf + 2 * B, (size_t)B * 4, cudaMemcpyHostToDevice, st));

        if (use_cuda_graph && !m.ls_active && B <= 64 && (m.n_host_experts == 0 || B == 1)) {
            GraphKey key{B, 10, want_taps ? 1 : 0, (int)kv_dtype};
            cudaGraphExec_t exec = nullptr;
            for (const auto& e : graph_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                cudaGraph_t gr = nullptr;
                cudaError_t ge = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                if (ge == cudaSuccess) {
                    emit_48_visits(B, 1, 0, 0, want_taps);
                    ge = cudaStreamEndCapture(st, &gr);
                    if (ge == cudaSuccess && gr) {
                        ge = cudaGraphInstantiate(&exec, gr, nullptr, nullptr, 0);
                        cudaGraphDestroy(gr);
                    }
                }
                if (ge != cudaSuccess || !exec) {
                    cudaGetLastError();
                    if (gr) cudaGraphDestroy(gr);
                    use_cuda_graph = false;
                    std::fprintf(stderr, "[InkCUDA] Batched CUDA Graph capture unavailable (err=%d): falling back to eager mode\n", (int)ge);
                    emit_48_visits(B, 1, 0, 0, want_taps);
                    return;
                }
                graph_cache.push_back({key, exec});
            }
            INK_CK(cudaGraphLaunch(exec, st));
            return;
        }
        emit_48_visits(B, 1, 0, 0, want_taps);
    }

    // ---- Unembed + Softcap(30.0) + Argmax ----
    // Fast exact 2-stage screened argmax (0.14 ms on RTX 5090, 100.000% mathematically bit-exact up to M <= 32!)
    void head_argmax_exact(int M, int32_t* h_out_ids) {
        prof_mark_start();
        const auto& ui = m.tinfo("model.llm.unembed.weight");
        if (m.unembed_screen.ready && (ui.fmt == FMT_BF16 || ui.fmt == FMT_S6_K) && M <= kInkMaxTree) {
            const int V = m.man.c.vocab_unpadded;
            const int g1 = (V + 7) / 8;
            for (int m0 = 0; m0 < M; m0 += 8) {
                const int mc = std::min(8, M - m0);
                const float* hm = d_h_norm + (size_t)m0 * kInkHidden;
                k_ink_row_l2norm<<<mc, 256, 0, st>>>(hm, d_h_norm2, d_max_lb, d_cand_cnt, mc);
                if (mc == 1) {
                    k_ink_screen_pass1_multiM<1><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                } else if (mc <= 4) {
                    k_ink_screen_pass1_multiM<4><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                } else {
                    k_ink_screen_pass1_multiM<8><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                }
                dim3 g2((V + 255) / 256, mc);
                k_ink_screen_pass2_bf16<<<g2, 256, 0, st>>>(
                    nullptr, m.unembed_screen.eps,
                    hm, d_h_norm2, d_dot_hat, d_max_lb, d_cand_ids, d_cand_cnt, 4096, mc, V);
                if (ui.fmt == FMT_BF16) {
                    k_ink_screen_eval_bf16<<<mc, 256, 0, st>>>(
                        (const uint16_t*)(m.d_trunk + ui.off), hm, d_cand_ids, d_cand_cnt, 4096, d_argmax + m0, mc);
                } else {
                    k_ink_screen_eval_s6k<<<mc, 256, 0, st>>>(
                        m.d_trunk + ui.off, hm, d_cand_ids, d_cand_cnt, 4096, d_argmax + m0, mc);
                }
            }
            prof_mark_end(7);
            INK_CK(cudaMemcpyAsync(h_out_ids, d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
            INK_CK(cudaStreamSynchronize(st));
            return;
        }
        head_logits_full(M);
        k_ink_argmax_rows<<<M, 1024, 0, st>>>(d_logits, m.man.c.vocab_unpadded, d_argmax);
        prof_mark_end(7);
        INK_CK(cudaMemcpyAsync(h_out_ids, d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
        INK_CK(cudaStreamSynchronize(st));
    }

    // Full exact softcapped logits [n_want, 200058] for selected hidden rows in d_h_src -> d_logits
    void head_logits_from(const float* d_h_src, int n_want) {
        ensure_logits(n_want);
        const int V = m.man.c.vocab_unpadded;
        linear("model.llm.unembed.weight", d_h_src, d_logits, n_want, V);
        const size_t n = (size_t)n_want * V;
        k_ink_softcap<<<(n + 255) / 256, 256, 0, st>>>(d_logits, n, m.man.c.logit_softcap);
    }

    // Full exact softcapped logits [M, 200058] into d_logits
    void head_logits_full(int M) {
        head_logits_from(d_h_norm, M);
    }

    void free_all() {
        if (m.ls_active && cur_stage != 0) bind_stage(0);
        for (auto& e : graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
        graph_cache.clear();
        if (m.ls_active) {
            auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
            if (ls_stages[0].ev_done) { cudaEventDestroy(ls_stages[0].ev_done); ls_stages[0].ev_done = nullptr; }
            for (int s = 1; s < m.n_stages; ++s) {
                cudaSetDevice(m.stage_gpu[s]);
                StageExec& se = ls_stages[s];
                se.ws.free_all();
                fr(se.d_cur); fr(se.d_res); fr(se.d_bank_out);
                for (int i = 0; i < 8; ++i) fr(se.d_blocks[i]);
                for (int i = 0; i < 4; ++i) fr(se.d_bank[i]);
                fr(se.d_mixed); fr(se.d_msgs); fr(se.d_inp0); fr(se.d_inp1); fr(se.d_inp2);
                fr(se.d_q); fr(se.d_k_raw); fr(se.d_v_raw); fr(se.d_k); fr(se.d_v); fr(se.d_r);
                fr(se.d_k_dyn); fr(se.d_v_dyn); fr(se.d_sinks); fr(se.d_gate_attn); fr(se.d_rel_table);
                fr(se.d_attn_out); fr(se.d_wo_raw); fr(se.d_wo_conv); fr(se.d_moe_in); fr(se.d_moe_raw); fr(se.d_moe_conv);
                fr(se.d_ctrl_cat); fr(se.d_ctrl_gate); fr(se.d_ctrl_gain); fr(se.d_ctrl_forget);
                fr(se.d_attn_pm); fr(se.d_attn_pl); fr(se.d_attn_po);
                fr(se.d_pf_q); fr(se.d_pf_k); fr(se.d_pf_v); fr(se.d_pf_p);
                fr(se.d_pf_s); fr(se.d_pf_o); fr(se.d_pf_ot); fr(se.d_pf_m); fr(se.d_pf_l);
                fr(se.d_x_bf16); fr(se.d_w_scratch); fr(se.d_slots); fr(se.d_pos);
                if (se.ev_done) { cudaEventDestroy(se.ev_done); se.ev_done = nullptr; }
                if (se.cublas) { cublasDestroy(se.cublas); se.cublas = nullptr; }
                if (se.st) { cudaStreamDestroy(se.st); se.st = nullptr; }
            }
            cudaSetDevice(m.primary_gpu);
            if (h_ls_bounce) { cudaFreeHost(h_ls_bounce); h_ls_bounce = nullptr; }
            if (h_ls_meta_bounce) { cudaFreeHost(h_ls_meta_bounce); h_ls_meta_bounce = nullptr; }
        }
        if (m.ep2_active && m.peer_gpu >= 0) {
            cudaSetDevice(m.peer_gpu);
            peer_ws.free_all();
            if (d_peer_x) { cudaFree(d_peer_x); d_peer_x = nullptr; }
            if (d_peer_y) { cudaFree(d_peer_y); d_peer_y = nullptr; }
            if (ev_peer_done) { cudaEventDestroy(ev_peer_done); ev_peer_done = nullptr; }
            if (peer_st) { cudaStreamDestroy(peer_st); peer_st = nullptr; }
            cudaSetDevice(m.primary_gpu);
        }
        if (d_peer_y_rx) { cudaFree(d_peer_y_rx); d_peer_y_rx = nullptr; }
        if (ev_moe_ready) { cudaEventDestroy(ev_moe_ready); ev_moe_ready = nullptr; }
        if (h_ep_x)   { cudaFreeHost(h_ep_x); h_ep_x = nullptr; }
        if (h_ep_idx) { cudaFreeHost(h_ep_idx); h_ep_idx = nullptr; }
        if (h_ep_w)   { cudaFreeHost(h_ep_w); h_ep_w = nullptr; }
        if (h_ep_y)   { cudaFreeHost(h_ep_y); h_ep_y = nullptr; }
        moe_ws.free_all();
        for (int v = 0; v < kInkVisits; ++v) {
            const int ip = ink_visit_phys(v);
            const int dev = m.ls_active ? m.stage_gpu[m.layer_owner[ip]] : m.primary_gpu;
            cudaSetDevice(dev);
            if (kv_desc[v].k_f16) { cudaFree(kv_desc[v].k_f16); kv_desc[v].k_f16 = nullptr; }
            if (kv_desc[v].v_f16) { cudaFree(kv_desc[v].v_f16); kv_desc[v].v_f16 = nullptr; }
            if (kv_desc[v].k_f8)  { cudaFree(kv_desc[v].k_f8);  kv_desc[v].k_f8 = nullptr; }
            if (kv_desc[v].v_f8)  { cudaFree(kv_desc[v].v_f8);  kv_desc[v].v_f8 = nullptr; }
            if (kv_desc[v].k_sc)  { cudaFree(kv_desc[v].k_sc);  kv_desc[v].k_sc = nullptr; }
            if (kv_desc[v].v_sc)  { cudaFree(kv_desc[v].v_sc);  kv_desc[v].v_sc = nullptr; }
            free_host_kv_visit(v);
            if (d_sconv_pool[v])  { cudaFree(d_sconv_pool[v]);  d_sconv_pool[v] = nullptr; }
        }
        cudaSetDevice(m.primary_gpu);
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_cur); fr(d_res); fr(d_prelude); fr(d_bank_out);
        for (int i = 0; i < 8; ++i) fr(d_blocks[i]);
        for (int i = 0; i < 2; ++i) fr(d_prelude_blocks[i]);
        for (int i = 0; i < 4; ++i) fr(d_bank[i]);
        fr(d_mixed); fr(d_msgs); fr(d_inp0); fr(d_inp1); fr(d_inp2);
        fr(d_q); fr(d_k_raw); fr(d_v_raw); fr(d_k); fr(d_v); fr(d_r);
        fr(d_k_dyn); fr(d_v_dyn); fr(d_sinks); fr(d_gate_attn); fr(d_rel_table);
        fr(d_attn_out); fr(d_wo_raw); fr(d_wo_conv); fr(d_moe_in); fr(d_moe_raw); fr(d_moe_conv);
        fr(d_h_norm);
        for (int i = 0; i < 4; ++i) { fr(d_taps[i]); fr(d_taps_compact[i]); }
        fr(d_ctrl_cat); fr(d_ctrl_gate); fr(d_ctrl_gain); fr(d_ctrl_forget);
        fr(d_attn_pm); fr(d_attn_pl); fr(d_attn_po);
        fr(d_pf_q); fr(d_pf_k); fr(d_pf_v); fr(d_pf_p); fr(d_pf_s); fr(d_pf_o); fr(d_pf_ot); fr(d_pf_m); fr(d_pf_l);
        fr(d_stage_k); fr(d_stage_v); fr(d_stage_sconv);
        fr(d_kv_desc_arr); fr(d_sconv_pool_arr); fr(d_acc_path);
        fr(d_x_bf16); fr(d_w_scratch); fr(d_logits);
        fr(d_dot_hat); fr(d_h_norm2); fr(d_max_lb); fr(d_cand_ids); fr(d_cand_cnt); fr(d_argmax);
        fr(d_tok_ids); fr(d_slots); fr(d_pos); fr(d_is_img); fr(d_img_emb); fr(d_sconv_lag); fr(d_tree_anc); fr(d_tree_dep);
        if (h_meta_pin) { cudaFreeHost(h_meta_pin); h_meta_pin = nullptr; }
        if (ev_p0) { cudaEventDestroy(ev_p0); ev_p0 = nullptr; }
        if (ev_p1) { cudaEventDestroy(ev_p1); ev_p1 = nullptr; }
        if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
        if (st) { cudaStreamDestroy(st); st = nullptr; }
    }
};

}  // namespace strata::ink
