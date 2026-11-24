// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/sycl/ink_sy_common.h - Inkling-2-Mini's GPU runtime (upstream's engine/src/inkling2/sycl, v0.2.19):
// the queue, the 16-lane primitives, dp4a, the .strata v1.1 manifest reader and the expert placement.
//
// XeStrata's changes: the queue is the engine's (core::Runtime: the GPU XeStrata selected, on any of the three
// makers'), the memory budget is the engine's (gpu::mem_info, vram_room, --vram-reserve-mib), and the kernels are
// written for 16 LOGICAL lanes that run on a sub-group of 16 (Intel's GPUs) or 32 (NVIDIA's and AMD's, which have no
// 16): every sub-group collective goes through the lane16_* helpers below, whose exchanges never cross a 16-lane half,
// so a 32-wide sub-group holds two independent logical groups.  A kernel must therefore keep both halves converged
// up to its last collective (no early return of one logical group before one).
#pragma once

#include <sycl/sycl.hpp>

#include <zstd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../ink_json.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/platform/env.hpp"
#include "../../kernels/xe/cuda_intrinsics.hpp"

// The sub-group the kernels run on: 16 where the device compile is for Intel's GPUs (every one has 16; upstream's
// choice, measured on the B70), 32 for NVIDIA's and AMD's (the only width all of them have).  INK_SY_SG32_TEST=1
// compiles 32 everywhere (the CPU device's check of the 32-wide path, tests/inkling2_kernels_test.cpp).
#if (defined(__SYCL_DEVICE_ONLY__) && (defined(__NVPTX__) || defined(__AMDGCN__))) || defined(INK_SY_SG32_TEST)
#define INK_REQD_SG [[sycl::reqd_sub_group_size(32)]]
#else
#define INK_REQD_SG [[sycl::reqd_sub_group_size(16)]]
#endif

namespace ink_sycl {

using half = sycl::half;

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline bool seek_abs(FILE* fp, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(fp, static_cast<__int64>(off), SEEK_SET) == 0;
#else
    return fseeko(fp, static_cast<off_t>(off), SEEK_SET) == 0;
#endif
}

// The engine's compute queue on the GPU it selected (in order), and its transfer queue on the same context.  A test
// sets test_queue() (an in-order queue on the CPU device: tests/inkling2_kernels_test.cpp) and runs the kernels alone.
inline sycl::queue*& test_queue() { static sycl::queue* q = nullptr; return q; }
inline sycl::queue& Q() { return test_queue() ? *test_queue() : strata::core::Runtime::get().compute(); }
inline sycl::queue& Q_copy() { return test_queue() ? *test_queue() : strata::core::Runtime::get().transfer(); }

inline void wait() { Q().wait_and_throw(); }

// The device's free and total memory: the engine's report (gpu::mem_info, XESTRATA_VRAM_LIMIT_MIB applied); under
// test_queue() the test device's total for both.
inline bool mem_now(size_t& free_b, size_t& total_b) {
    if (test_queue()) {
        free_b = total_b = Q().get_device().get_info<sycl::info::device::global_mem_size>();
        return true;
    }
    return strata::gpu::mem_info(&free_b, &total_b);
}

struct SyDevCaps {
    std::string name;       ///< for the reports only: no path is chosen by it
    std::string platform;
    size_t global_mem = 0;
    size_t max_alloc = 0;
    size_t local_mem = 0;
    uint32_t eu_count = 0;
    int sg = 16;            ///< the sub-group the kernels run on (INK_REQD_SG): 16 where the device has it, else 32
    bool matrix_f16 = false;   ///< FP16 products on matrix engines (prefill::Gemm takes them): the dequant + GEMM paths
    bool is_level_zero = false;
};

inline const SyDevCaps& dev_caps() {
    static SyDevCaps d = [] {
        SyDevCaps c;
        auto dev = Q().get_device();
        c.name = dev.get_info<sycl::info::device::name>();
        c.platform = dev.get_platform().get_info<sycl::info::platform::name>();
        size_t free_b = 0, total_b = 0;
        c.global_mem = mem_now(free_b, total_b) ? total_b : dev.get_info<sycl::info::device::global_mem_size>();
        c.max_alloc = test_queue() ? dev.get_info<sycl::info::device::max_mem_alloc_size>()
                                   : strata::core::max_alloc_bytes();
        c.local_mem = dev.get_info<sycl::info::device::local_mem_size>();
        c.eu_count = dev.get_info<sycl::info::device::max_compute_units>();
        c.is_level_zero = (dev.get_backend() == sycl::backend::ext_oneapi_level_zero);
        bool has16 = false, has32 = false;
        for (auto s : dev.get_info<sycl::info::device::sub_group_sizes>()) {
            has16 = has16 || s == 16;
            has32 = has32 || s == 32;
        }
        // the device compile chose the width (INK_REQD_SG); a device with neither cannot run the kernels at all
        if (!has16 && !has32) throw std::runtime_error("inkling2: the GPU has neither a 16- nor a 32-wide sub-group");
#if defined(INK_SY_SG32_TEST)
        c.sg = 32;
#else
        c.sg = (c.is_level_zero || dev.get_backend() == sycl::backend::opencl) && has16 ? 16 : 32;
#endif
        c.matrix_f16 = !test_queue() && strata::kernels::f16_matrix_engines();
        return c;
    }();
    return d;
}

inline size_t& dev_bytes_allocated() { static size_t b = 0; return b; }

template <class T>
inline T* dalloc(size_t n) {
    size_t cnt = n ? n : 1;
    T* p = sycl::malloc_device<T>(cnt, Q());
    if (!p) {
        throw std::runtime_error("inkling2: out of GPU memory allocating " + std::to_string(cnt * sizeof(T)) +
                                 " bytes (" + std::to_string(dev_bytes_allocated() >> 20) +
                                 " MiB held); a smaller --max-context or --parallel, or --vram-reserve-mib, leaves room");
    }
    dev_bytes_allocated() += cnt * sizeof(T);
    return p;
}

template <class T>
inline T* halloc(size_t n) {
    size_t cnt = n ? n : 1;
    T* p = sycl::malloc_host<T>(cnt, Q());
    if (!p) throw std::runtime_error("inkling2: out of pinned host memory allocating " + std::to_string(cnt * sizeof(T)) + " bytes");
    return p;
}

inline void dfree(void* p) { if (p) sycl::free(p, Q()); }
inline void h2d(void* d, const void* h, size_t b) { if (b) Q().memcpy(d, h, b); }
inline void h2d_sync(void* d, const void* h, size_t b) { if (b) Q().memcpy(d, h, b).wait(); }
inline void d2h(void* h, const void* d, size_t b) { if (b) Q().memcpy(h, d, b); }
inline void d2d(void* dst, const void* src, size_t b) { if (b) Q().memcpy(dst, src, b); }
inline void dzero(void* d, size_t b) { if (b) Q().memset(d, 0, b); }

inline sycl::nd_range<1> NR(size_t G, size_t B) { return sycl::nd_range<1>(G * B, B); }
inline sycl::nd_range<2> NR2(size_t Gx, size_t Gy, size_t Bx) { return sycl::nd_range<2>(sycl::range<2>(Gy, Gx * Bx), sycl::range<2>(1, Bx)); }

template <class T, int N, class It>
inline T* lmem(const It& it) {
    auto p = sycl::ext::oneapi::group_local_memory_for_overwrite<T[N]>(it.get_group());
    return &(*p)[0];
}

// ---- 16 logical lanes on a 16- or 32-wide sub-group ----
// The logical lane and the logical sub-group (16 work-items) of a work-item, from its local id: never from the
// sub_group's own ids, which count 32-wide groups on NVIDIA's and AMD's GPUs.
template <class It> inline int lane16(const It& it) { return (int)it.get_local_linear_id() & 15; }
template <class It> inline int group16(const It& it) { return (int)it.get_local_linear_id() >> 4; }
template <class It> inline int groups16(const It& it) { return (int)it.get_local_range().size() >> 4; }

// Butterfly reductions over the 16 logical lanes (the result in every lane, as reduce_over_group gives it): the
// exchanges (xor 8, 4, 2, 1) stay within a 16-lane half of a 32-wide sub-group.
template <class It, class T>
inline T lane16_sum(const It& it, T v) {
    auto sg = it.get_sub_group();
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) v += sycl::permute_group_by_xor(sg, v, m);
    return v;
}
template <class It, class T>
inline T lane16_max(const It& it, T v) {
    auto sg = it.get_sub_group();
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) v = sycl::max(v, sycl::permute_group_by_xor(sg, v, m));
    return v;
}
// The exclusive prefix sum over the 16 logical lanes (Hillis-Steele: the shifts reach no lane of the other half).
template <class It>
inline int lane16_exclusive_sum(const It& it, int v) {
    auto sg = it.get_sub_group();
    const int l = lane16(it);
    int inc = v;
#pragma unroll
    for (int d = 1; d < 16; d <<= 1) {
        const int o = sycl::shift_group_right(sg, inc, d);
        if (l >= d) inc += o;
    }
    return inc - v;
}

template <class It>
inline float sg_sum(const It& it, float v) { return lane16_sum(it, v); }

// Work-group sum for a group of up to 256 work-items, a multiple of 16 (up to 16 logical sub-groups)
template <class It, class P>
inline float wg_sum16(const It& it, float v, P sh) {
    v = lane16_sum(it, v);
    const int w = group16(it), l = lane16(it), nw = groups16(it);
    sycl::group_barrier(it.get_group());
    if (l == 0) sh[w] = v;
    sycl::group_barrier(it.get_group());
    float r = (l < nw) ? (float)sh[l] : 0.0f;
    return lane16_sum(it, r);
}

// The int8x4 dot product: the engine's shared one (kernels/xe/cuda_intrinsics.hpp: SPIR-V's integer dot product on
// Intel's GPUs, dp4a on NVIDIA's, the dot4 instruction on AMD's), exact integer sums as upstream's dot_acc
inline int32_t dp4a_s8(int32_t a_packed, int32_t b_packed, int32_t acc) {
    return strata::kernels::xe::dp4a(a_packed, b_packed, acc);
}

// ---- Device-side numeric decoders (bit-exact match to sk_codecs.cpp & inkling2_loop.py) ----
inline float bf16_to_f32(uint16_t b) {
    return sycl::bit_cast<float>((uint32_t)b << 16);
}

inline uint16_t ld16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

inline float h2f_bits(uint16_t b) {
    return (float)sycl::bit_cast<half>(b);
}

inline float e2m1_lut(uint32_t nib) {
    constexpr float kLut[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return kLut[nib & 15u];
}

inline float e4m3_to_f32(uint32_t c) {
    const uint32_t mag = c & 0x7Fu;
    if (mag == 0x7Fu) return 0.0f;
    const uint32_t exp = mag >> 3;
    const uint32_t man = mag & 7u;
    float v;
    if (exp == 0u) {
        v = (float)man * (1.0f / 512.0f);
    } else {
        v = sycl::bit_cast<float>(((exp + 120u) << 23) | (man << 20));
    }
    return (c & 0x80u) ? -v : v;
}

inline float ue8m0_to_f32(uint32_t e) {
    if (e == 0u) return 5.877471754111438e-39f;
    if (e == 255u) return 0.0f;
    return sycl::bit_cast<float>(e << 23);
}

inline float situ_act(float g, float u) {
    const float sig = 1.0f / (1.0f + sycl::exp(-g));
    const float g_act = 4.0f * sycl::tanh(sycl::clamp(g * 0.25f, -15.0f, 15.0f)) * sig;
    const float u_act = 25.0f * sycl::tanh(sycl::clamp(u * 0.04f, -15.0f, 15.0f));
    return g_act * u_act;
}

// ---- Format IDs & Manifest ----
enum InkFmt : int32_t {
    FMT_S2 = 2, FMT_S3 = 3, FMT_S4 = 4, FMT_S5 = 5, FMT_S6 = 6, FMT_S8 = 8,
    FMT_SF4 = 15, FMT_MX8 = 16,
    FMT_GGUF_RAW = 20, // native GGUF Q8_0 expert body (0x47475546 header)
    FMT_Q8ROW = 100, FMT_F32 = 101, FMT_BF16 = 102,
    FMT_Q8_0 = 103,    // a GGUF file's dense Q8_0 rows, split for aligned loads: N * K/32 FP32 scales, then N * K int8
};

constexpr int sk_sb_bytes(int f) {
    return f == FMT_S2 ? 70 : f == FMT_S3 ? 102 : f == FMT_S4 ? 134 : f == FMT_S5 ? 166 : f == FMT_S6 ? 198 : f == FMT_S8 ? 262 : 0;
}
constexpr float sk_half(int f) {
    return f == FMT_S2 ? 1.5f : f == FMT_S3 ? 3.5f : f == FMT_S4 ? 7.5f : f == FMT_S5 ? 15.5f : f == FMT_S6 ? 31.5f : 0.0f;
}

inline int fmt_from_name(const std::string& s) {
    if (s == "F32" || s == "float32") return FMT_F32;
    if (s == "BF16" || s == "bfloat16") return FMT_BF16;
    if (s == "Q8_ROW") return FMT_Q8ROW;
    if (s == "S2_K") return FMT_S2;
    if (s == "S3_K") return FMT_S3;
    if (s == "S4_K") return FMT_S4;
    if (s == "S5_K") return FMT_S5;
    if (s == "S6_K") return FMT_S6;
    if (s == "S8_K") return FMT_S8;
    if (s == "SF4") return FMT_SF4;
    if (s == "MX8") return FMT_MX8;
    return FMT_F32;
}

struct InkTensorInfo {
    std::string name;
    int fmt = FMT_F32;
    std::vector<int64_t> shape;
    uint64_t off = 0, bytes = 0;
    const uint8_t* dev_ptr = nullptr;
    int64_t rows() const { return shape.size() > 1 ? shape[0] : 1; }
    int64_t cols() const { return shape.empty() ? 0 : shape.back(); }
};

struct InkCfg {
    int n_phys = 24, n_visits = 48, hidden = 2304, inter = 1152;
    int vocab_padded = 201024, vocab_unpadded = 200058;
    int n_head = 24, n_kv = 8, head_dim = 128, d_rel = 16, rel_extent = 1024;
    int swa_win = 1024, block_size = 6, n_experts = 128, top_k = 8;
    int core_start = 6, core_end = 18, num_passes = 3, gate_rank = 64;
    int eos_id = 200006, image_pad_id = 200054;
    float rms_eps = 1e-6f, qk_l2_clip = 22.627416997969522f, rel_l2_clip = 8.0f;
    float log_scale_alpha = 0.1f, logit_softcap = 30.0f, route_scale = 8.0f;
    int log_scale_floor = 8192;
    std::vector<int> is_swa;

    bool has_dspark = false;
    int dspark_layers = 3, dspark_inter = 9216, dspark_block_size = 8, dspark_markov_rank = 256;
    float dspark_rope_theta = 1000000.0f;
    int dspark_swa_win = 1025;
    int dspark_taps[4] = {5, 23, 41, 48};
    int dspark_is_swa[3] = {1, 0, 1};

    int phys_for_visit(int v) const {
        const int clen = core_end - core_start;
        if (v < core_start) return v;
        if (v < core_start + num_passes * clen) return core_start + ((v - core_start) % clen);
        return core_end + (v - core_start - num_passes * clen);
    }
};

struct InkManifest {
    InkCfg c;
    std::string build;
    std::string chat_template;
    std::unordered_map<std::string, InkTensorInfo> T;
    std::unordered_map<std::string, InkTensorInfo> D;

    const InkTensorInfo& get(const std::string& n) const {
        auto it = T.find(n);
        if (it == T.end()) { fprintf(stderr, "[ink_sycl] missing trunk tensor %s\n", n.c_str()); std::exit(2); }
        return it->second;
    }
    const InkTensorInfo& get_d(const std::string& n) const {
        auto it = D.find(n);
        if (it == D.end()) { fprintf(stderr, "[ink_sycl] missing dspark tensor %s\n", n.c_str()); std::exit(2); }
        return it->second;
    }
};

// Direct USM device/host-pinned pointer per expert (supports multi-slab VRAM <= 3.5 GiB/slab AND hybrid host-pinned fallback!)
struct InkSyExpertDev {
    const uint8_t* body_ptr;  // USM device pointer (VRAM slab) OR USM host-pinned pointer (hybrid cold expert!)
    uint32_t dn_off;
    uint8_t gu_code;
    uint8_t dn_code;
    uint8_t in_vram;
    uint8_t pad;
};

inline bool read_section0_zstd(FILE* fp, uint64_t off, uint64_t len, std::string& out, std::string& err) {
    if (len == 0) { out.clear(); return true; }
    std::vector<uint8_t> raw(len);
    if (!seek_abs(fp, off) || std::fread(raw.data(), 1, len, fp) != len) {
        err = "short read Section 0";
        return false;
    }
    unsigned long long rsize = ZSTD_getFrameContentSize(raw.data(), raw.size());
    if (rsize == ZSTD_CONTENTSIZE_ERROR) {
        out.assign(reinterpret_cast<const char*>(raw.data()), raw.size());
        return true;
    }
    if (rsize != ZSTD_CONTENTSIZE_UNKNOWN) {
        out.resize((size_t)rsize);
        size_t d = ZSTD_decompress(out.data(), out.size(), raw.data(), raw.size());
        if (ZSTD_isError(d)) { err = std::string("zstd: ") + ZSTD_getErrorName(d); return false; }
        out.resize(d);
        return true;
    }
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    ZSTD_inBuffer in_b{raw.data(), raw.size(), 0};
    std::vector<char> chunk(1 << 20);
    out.clear();
    while (in_b.pos < in_b.size) {
        ZSTD_outBuffer out_b{chunk.data(), chunk.size(), 0};
        size_t ret = ZSTD_decompressStream(dctx, &out_b, &in_b);
        if (ZSTD_isError(ret)) { ZSTD_freeDCtx(dctx); err = std::string("zstd: ") + ZSTD_getErrorName(ret); return false; }
        out.append(chunk.data(), out_b.pos);
    }
    ZSTD_freeDCtx(dctx);
    return true;
}

inline bool parse_ink_manifest(const std::string& json_text, InkManifest& m, std::string& err) {
    try {
        ink2::JVal root = ink2::json_parse(json_text);
        if (const auto* b = root.get("build")) m.build = b->str();
        if (const auto* ct = root.get("chat_template")) m.chat_template = ct->str();
        const ink2::JVal& cfg = root.at("config");
        const ink2::JVal& tc = cfg.at("text_config");
        InkCfg& c = m.c;
        if (const auto* v = cfg.get("eos_token_id")) c.eos_id = (int)v->i();
        c.n_phys = (int)tc.at("num_hidden_layers").i();
        c.hidden = (int)tc.at("hidden_size").i();
        c.inter = (int)tc.at("intermediate_size").i();
        c.vocab_padded = (int)tc.at("vocab_size").i();
        if (const auto* v = tc.get("unpadded_vocab_size")) c.vocab_unpadded = (int)v->i();
        c.n_head = (int)tc.at("num_attention_heads").i();
        c.n_kv = (int)tc.at("num_key_value_heads").i();
        c.head_dim = (int)tc.at("head_dim").i();
        if (const auto* v = tc.get("d_rel")) c.d_rel = (int)v->i();
        if (const auto* v = tc.get("rel_extent")) c.rel_extent = (int)v->i();
        if (const auto* v = tc.get("sliding_window_size")) c.swa_win = (int)v->i();
        if (const auto* v = tc.get("n_layers_per_attn_res_block")) c.block_size = (int)v->i();
        if (const auto* v = tc.get("n_routed_experts")) c.n_experts = (int)v->i();
        if (const auto* v = tc.get("num_experts_per_tok")) c.top_k = (int)v->i();
        if (const auto* v = tc.get("rms_norm_eps")) c.rms_eps = (float)v->f();
        if (const auto* v = tc.get("qk_l2_clip_threshold")) c.qk_l2_clip = (float)v->f();
        if (const auto* v = tc.get("rel_l2_clip_threshold")) c.rel_l2_clip = (float)v->f();
        if (const auto* v = tc.get("log_scaling_alpha")) c.log_scale_alpha = (float)v->f();
        if (const auto* v = tc.get("log_scaling_n_floor")) c.log_scale_floor = (int)v->i();
        if (const auto* v = tc.get("final_logit_softcapping")) c.logit_softcap = (float)v->f();
        if (const auto* v = tc.get("route_scale")) c.route_scale = (float)v->f();

        c.is_swa.assign(c.n_phys, 0);
        if (const auto* locs = tc.get("local_layer_ids")) {
            for (const auto& lid : locs->a) {
                int id = (int)lid.i();
                if (id >= 0 && id < c.n_phys) c.is_swa[id] = 1;
            }
        }
        if (const auto* lc = tc.get("loop_config")) {
            c.core_start = (int)lc->at("core_start").i();
            c.core_end = (int)lc->at("core_end").i();
            c.num_passes = (int)lc->at("num_passes").i();
            c.gate_rank = (int)lc->at("gate_rank").i();
        }
        c.n_visits = c.core_start + c.num_passes * (c.core_end - c.core_start) + (c.n_phys - c.core_end);

        if (const auto* mc = cfg.get("mtp_config")) {
            c.dspark_layers = (int)mc->at("num_hidden_layers").i();
            c.dspark_inter = (int)mc->at("intermediate_size").i();
            c.dspark_block_size = (int)mc->at("block_size").i();
            c.dspark_markov_rank = (int)mc->at("markov_rank").i();
            if (const auto* v = mc->get("rope_theta")) c.dspark_rope_theta = (float)v->f();
            if (const auto* v = mc->get("sliding_window_size")) c.dspark_swa_win = (int)v->i() + 1;
            if (const auto* tl = mc->get("target_layer_ids")) {
                for (size_t i = 0; i < std::min<size_t>(4, tl->a.size()); ++i) c.dspark_taps[i] = (int)tl->a[i].i();
            }
            for (int i = 0; i < 3; ++i) c.dspark_is_swa[i] = 0;
            if (const auto* ll = mc->get("local_layer_ids")) {
                for (const auto& x : ll->a) { int id = (int)x.i(); if (id >= 0 && id < 3) c.dspark_is_swa[id] = 1; }
            }
        }
        for (const auto& t : root.at("trunk_tensors").a) {
            InkTensorInfo ti;
            ti.name = t.at("name").str();
            ti.fmt = fmt_from_name(t.at("format").str());
            for (const auto& d : t.at("shape").a) ti.shape.push_back(d.i());
            ti.off = (uint64_t)t.at("offset").i();
            ti.bytes = (uint64_t)t.at("bytes").i();
            m.T[ti.name] = std::move(ti);
        }
        if (const auto* dt = root.get("dspark_tensors")) {
            for (const auto& t : dt->a) {
                InkTensorInfo ti;
                ti.name = t.at("name").str();
                ti.fmt = fmt_from_name(t.at("format").str());
                for (const auto& d : t.at("shape").a) ti.shape.push_back(d.i());
                ti.off = (uint64_t)t.at("offset").i();
                ti.bytes = (uint64_t)t.at("bytes").i();
                m.D[ti.name] = std::move(ti);
            }
            c.has_dspark = !m.D.empty();
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("parse_ink_manifest: ") + e.what();
        return false;
    }
}

// Convert the adapter's canonical manifest into the SYCL backend's own manifest (the two share
// the format ids and the geometry, but are separate structs).
inline bool convert_ink_manifest(const ink2::InkManifest& src, InkManifest& m, std::string& err) {
    (void)err;
    const ink2::InkCfg& s = src.c;
    InkCfg& c = m.c;
    c.n_phys = s.n_phys; c.hidden = s.hidden; c.inter = s.inter;
    c.vocab_padded = s.vocab_padded; c.vocab_unpadded = s.vocab_unpadded;
    c.n_head = s.n_head; c.n_kv = s.n_kv; c.head_dim = s.head_dim;
    c.d_rel = s.d_rel; c.rel_extent = s.rel_extent; c.swa_win = s.swa_win;
    c.block_size = s.block_size; c.n_experts = s.n_experts; c.top_k = s.top_k;
    c.core_start = s.core_start; c.core_end = s.core_end;
    c.num_passes = s.num_passes; c.gate_rank = s.gate_rank;
    c.eos_id = s.eos_token_id; c.image_pad_id = s.image_token_id;
    c.rms_eps = s.rms_eps; c.qk_l2_clip = s.qk_l2_clip; c.rel_l2_clip = s.rel_l2_clip;
    c.log_scale_floor = s.log_scale_floor; c.log_scale_alpha = s.log_scale_alpha;
    c.route_scale = s.route_scale; c.logit_softcap = s.logit_softcap;
    c.is_swa.assign(s.is_swa.begin(), s.is_swa.end());
    c.n_visits = s.n_visits;
    c.has_dspark = !src.dspark_tensors.empty();
    m.build = src.build; m.chat_template = src.chat_template;
    auto add = [](const auto& from, auto& to) {
        for (const auto& kv : from) {
            InkTensorInfo ti;
            ti.name = kv.first; ti.fmt = kv.second.fmt;
            ti.shape = kv.second.shape; ti.off = kv.second.off; ti.bytes = kv.second.bytes;
            to[ti.name] = std::move(ti);
        }
    };
    add(src.trunk, m.T); add(src.dspark_tensors, m.D);
    return true;
}

// Multi-slab VRAM + 3x Loop-Weighted Expert Loader for 8/12/16/32 GB Intel Arc GPUs
struct InkSyclModel {
    InkManifest man;
    std::string path;
    uint8_t format_minor = 0;

    // Trunk + DSpark stored in <= 3.5 GiB device slabs (fits in 1 slab for SQ-15/10, 2 slabs for SQ-20)
    std::vector<uint8_t*> trunk_slabs;
    // Expert VRAM slabs (<= 3.5 GiB each) + host-pinned USM slabs for cold experts on 8/12/16 GB GPUs
    std::vector<uint8_t*> vram_exp_slabs;
    std::vector<uint8_t*> host_exp_slabs;

    InkSyExpertDev* expert_dev = nullptr;  // [24 * 128] on device
    float* msg_queries_dev = nullptr;      // [12 * 4 * 2304]
    float* static_sconv_dev = nullptr;     // [24 * kSconvLayerFloats]
    float* rel_proj_pad_dev = nullptr;     // [24 * 16 * 1025]

    int n_vram_experts = 0;
    int n_host_experts = 0;
    size_t vram_expert_bytes = 0;
    size_t max_expert_bytes = 0;
    double t_load = 0.0;

    // Host shadow metadata for Strata Predictive CPU/RAM Hybrid + Elastic VRAM Cache Ring
    std::vector<InkSyExpertDev> h_ed;
    std::vector<const uint8_t*> host_body_ptrs;
    std::vector<uint32_t> expert_body_bytes;

    static constexpr size_t kSconvLayerFloats = (1024 + 1024 + 2304 + 2304) * 4;
    static constexpr size_t kSconvKOff = 0;
    static constexpr size_t kSconvVOff = 1024 * 4;
    static constexpr size_t kSconvAttnOff = 2048 * 4;
    static constexpr size_t kSconvMlpOff = (2048 + 2304) * 4;

    ~InkSyclModel() {
        for (auto* p : trunk_slabs) dfree(p);
        for (auto* p : vram_exp_slabs) dfree(p);
        for (auto* p : host_exp_slabs) dfree(p);
        dfree(expert_dev);
        dfree(msg_queries_dev);
        dfree(static_sconv_dev);
        dfree(rel_proj_pad_dev);
    }

    bool load(const ink2::InkModelFile& mf, double vram_experts_gb_override, std::string& err, size_t extra_reserve_bytes,
              int64_t reserve_mib) {
        double t0 = now_s();
        const std::string& p = mf.path;
        path = p;
        const bool gguf = (mf.kind == ink2::ModelSourceKind::Gguf);
        FILE* fp = nullptr;
        strata::artifact::StrataSectionEntry secs[6]{};
        const std::vector<uint8_t>* sec1_host = nullptr;   // a GGUF's trunk, as its adapter assembled it
        if (gguf) {
            if (!convert_ink_manifest(mf.m, man, err)) return false;
            sec1_host = &mf.gguf_adapter->sec1_arena;
        } else {
            fp = std::fopen(p.c_str(), "rb");
            if (!fp) { err = "cannot open " + p; return false; }

            uint8_t frame[4096];
            if (std::fread(frame, 1, 4096, fp) != 4096) { std::fclose(fp); err = "short header"; return false; }
            static const uint8_t kMagic[8] = {'S', 'T', 'R', 'A', 'T', 'A', 0x00, 0x01};
            if (std::memcmp(frame, kMagic, 8) != 0) { std::fclose(fp); err = "bad .strata magic"; return false; }
            uint32_t hdr_flags = 0;
            std::memcpy(&hdr_flags, frame + 8, 4);
            format_minor = (uint8_t)(hdr_flags & 0xFFu);

            for (int i = 0; i < 6; ++i) std::memcpy(&secs[i], frame + 0x10 + i * 64, 64);

            std::string mjson;
            if (!read_section0_zstd(fp, secs[0].file_offset, secs[0].byte_length, mjson, err)) {
                std::fclose(fp);
                return false;
            }
            if (!parse_ink_manifest(mjson, man, err)) { std::fclose(fp); return false; }
        }

        // 1. Upload Section 1 (Trunk + DSpark) into <= 3.2 GiB device slabs (safe under Arc 4 GiB max_alloc!)
        // slabs of at most 7/8 of the device's largest allocation (core::max_alloc_bytes, XESTRATA_MAX_ALLOC_MIB applied)
        const size_t kMaxSlab = dev_caps().max_alloc * 7 / 8;
        auto upload_tensor_map = [&](std::unordered_map<std::string, InkTensorInfo>& mp,
                                     std::unordered_map<std::string, std::vector<uint8_t>>& host_copies) -> bool {
            // Pack tensors sequentially into device slabs without crossing a slab boundary
            std::vector<InkTensorInfo*> ordered;
            ordered.reserve(mp.size());
            for (auto& kv : mp) ordered.push_back(&kv.second);
            std::sort(ordered.begin(), ordered.end(), [](const InkTensorInfo* a, const InkTensorInfo* b) { return a->off < b->off; });

            size_t idx = 0;
            while (idx < ordered.size()) {
                size_t slab_sz = 0;
                size_t start = idx;
                while (idx < ordered.size()) {
                    size_t tb = ((ordered[idx]->bytes + 63) / 64) * 64;
                    if (slab_sz > 0 && slab_sz + tb > kMaxSlab) break;
                    slab_sz += tb;
                    ++idx;
                }
                uint8_t* dev_slab = dalloc<uint8_t>(slab_sz);
                trunk_slabs.push_back(dev_slab);
                std::vector<uint8_t> staging;
                size_t cur_off = 0;
                for (size_t j = start; j < idx; ++j) {
                    InkTensorInfo* ti = ordered[j];
                    staging.resize(ti->bytes);
                    if (gguf) {
                        if (ti->off + ti->bytes > sec1_host->size()) {
                            err = "short read trunk tensor " + ti->name;
                            return false;
                        }
                        std::memcpy(staging.data(), sec1_host->data() + ti->off, ti->bytes);
                    } else if (!seek_abs(fp, secs[1].file_offset + ti->off) || std::fread(staging.data(), 1, ti->bytes, fp) != ti->bytes) {
                        err = "short read trunk tensor " + ti->name;
                        return false;
                    }
                    h2d(dev_slab + cur_off, staging.data(), ti->bytes);
                    wait();
                    ti->dev_ptr = dev_slab + cur_off;
                    // Keep tiny pre-orientation tensors on host temporarily
                    if (ti->name.find("sconv") != std::string::npos ||
                        ti->name.find("middle_cycle_cross_pass") != std::string::npos ||
                        ti->name.find("rel_logits_proj") != std::string::npos) {
                        host_copies[ti->name] = staging;
                    }
                    cur_off += ((ti->bytes + 63) / 64) * 64;
                }
                wait();
            }
            return true;
        };

        std::unordered_map<std::string, std::vector<uint8_t>> host_tmp;
        if (!upload_tensor_map(man.T, host_tmp) || !upload_tensor_map(man.D, host_tmp)) {
            if (fp) std::fclose(fp);
            return false;
        }
        prepare_preoriented_tables(host_tmp);

        // 2. Read Section 3 expert directory + 3x loop-weighted vram_load_order[3072]
        const size_t n_total = (size_t)man.c.n_phys * (size_t)man.c.n_experts;
        std::vector<strata::artifact::StrataExpertEntry> ents_by_gid(n_total);
        std::vector<uint16_t> order;
        if (gguf) {
            const auto& dir = mf.expert_directory();
            for (size_t i = 0; i < std::min(dir.size(), n_total); ++i) {
                const size_t layer = i / (size_t)man.c.n_experts;
                const size_t gid = layer * (size_t)man.c.n_experts + dir[i].expert_id;
                if (gid < n_total) ents_by_gid[gid] = dir[i];
            }
            order = mf.vram_load_order;
        } else if (secs[3].byte_length > 0) {
            std::vector<uint8_t> s3(secs[3].byte_length);
            if (!seek_abs(fp, secs[3].file_offset) || std::fread(s3.data(), 1, s3.size(), fp) != s3.size()) {
                std::fclose(fp); err = "short read Section 3"; return false;
            }
            const bool has_order = (secs[3].flags & 1u) != 0;
            const size_t n_ent = has_order ? (s3.size() / 34) : (s3.size() / 32);
            const auto* raw_ents = reinterpret_cast<const strata::artifact::StrataExpertEntry*>(s3.data());
            for (size_t i = 0; i < std::min(n_ent, n_total); ++i) {
                const size_t layer = i / (size_t)man.c.n_experts;
                const size_t gid = layer * (size_t)man.c.n_experts + raw_ents[i].expert_id;
                if (gid < n_total) ents_by_gid[gid] = raw_ents[i];
            }
            if (has_order && s3.size() >= n_ent * 34) {
                order.resize(n_ent);
                std::memcpy(order.data(), s3.data() + n_ent * 32, n_ent * 2);
            }
        }
        if (order.size() != n_total) {
            // Fallback priority: L6..17 looped core first, then L0..5, then L18..23!
            order.clear();
            for (int l = man.c.core_start; l < man.c.core_end; ++l)
                for (int e = 0; e < man.c.n_experts; ++e) order.push_back((uint16_t)(l * man.c.n_experts + e));
            for (int l = 0; l < man.c.core_start; ++l)
                for (int e = 0; e < man.c.n_experts; ++e) order.push_back((uint16_t)(l * man.c.n_experts + e));
            for (int l = man.c.core_end; l < man.c.n_phys; ++l)
                for (int e = 0; e < man.c.n_experts; ++e) order.push_back((uint16_t)(l * man.c.n_experts + e));
        }

        // 3. The experts' VRAM: the device's free memory now (gpu::mem_info, XESTRATA_VRAM_LIMIT_MIB applied), less what
        // the runner allocates after the load (`extra_reserve_bytes`: KV, ShortConv, DSpark, activations), less the room
        // no plan counts (core::vram_room, or --vram-reserve-mib); --vram-experts-gb sets it outright.
        size_t free_b = 0, gpu_mem = 0;
        if (!mem_now(free_b, gpu_mem)) { if (fp) std::fclose(fp); err = "inkling2: the GPU's free memory is unknown"; return false; }
        const size_t room = reserve_mib >= 0 ? ((size_t)reserve_mib << 20) : (size_t)strata::core::vram_room(gpu_mem);
        size_t vram_budget = free_b > extra_reserve_bytes + room ? free_b - extra_reserve_bytes - room : 0;
        if (vram_experts_gb_override >= 0.0) {
            vram_budget = (size_t)(vram_experts_gb_override * (1024.0 * 1024.0 * 1024.0));
        }

        // Assign experts in exact 3x loop-weighted priority order!
        std::vector<uint8_t> in_vram(n_total, 0);
        std::vector<size_t> vram_gids, host_gids;
        size_t vram_acc = 0;
        for (uint16_t gid : order) {
            if (gid >= n_total) continue;
            const size_t eb = ents_by_gid[gid].body_bytes;
            if (vram_acc + eb <= vram_budget) {
                in_vram[gid] = 1;
                vram_gids.push_back(gid);
                vram_acc += eb;
            } else {
                host_gids.push_back(gid);
            }
        }
        n_vram_experts = (int)vram_gids.size();
        n_host_experts = (int)host_gids.size();
        vram_expert_bytes = vram_acc;

        h_ed.assign(n_total, InkSyExpertDev{});
        host_body_ptrs.assign(n_total, nullptr);
        expert_body_bytes.assign(n_total, 0);
        max_expert_bytes = 0;
        auto populate_experts = [&](const std::vector<size_t>& gids, bool to_device) -> bool {
            size_t i = 0;
            std::vector<uint8_t> buf;
            while (i < gids.size()) {
                size_t slab_sz = 0, start = i;
                while (i < gids.size()) {
                    size_t eb = ((ents_by_gid[gids[i]].body_bytes + 4095) / 4096) * 4096;
                    if (slab_sz > 0 && slab_sz + eb > kMaxSlab) break;
                    slab_sz += eb;
                    ++i;
                }
                uint8_t* slab = to_device ? dalloc<uint8_t>(slab_sz) : halloc<uint8_t>(slab_sz);
                if (to_device) vram_exp_slabs.push_back(slab); else host_exp_slabs.push_back(slab);
                size_t off = 0;
                for (size_t j = start; j < i; ++j) {
                    const size_t gid = gids[j];
                    const auto& e = ents_by_gid[gid];
                    if (gguf) {
                        buf.resize(e.body_bytes);
                        if (!mf.read_expert_body(gid, buf, err)) return false;
                        if (to_device) { h2d(slab + off, buf.data(), e.body_bytes); wait(); }
                        else { std::memcpy(slab + off, buf.data(), e.body_bytes); host_body_ptrs[gid] = slab + off; }
                    } else if (to_device) {
                        buf.resize(e.body_bytes);
                        if (!seek_abs(fp, secs[4].file_offset + e.body_offset) || std::fread(buf.data(), 1, e.body_bytes, fp) != e.body_bytes) {
                            err = "short read expert body"; return false;
                        }
                        h2d(slab + off, buf.data(), e.body_bytes);
                        wait();
                    } else {
                        if (!seek_abs(fp, secs[4].file_offset + e.body_offset) || std::fread(slab + off, 1, e.body_bytes, fp) != e.body_bytes) {
                            err = "short read host expert body"; return false;
                        }
                        host_body_ptrs[gid] = slab + off;
                    }
                    expert_body_bytes[gid] = e.body_bytes;
                    max_expert_bytes = std::max<size_t>(max_expert_bytes, e.body_bytes);
                    h_ed[gid] = InkSyExpertDev{slab + off, e.dn_offset, e.gu_code_id, e.dn_code_id, (uint8_t)(to_device ? 1 : 0), 0};
                    off += ((e.body_bytes + 4095) / 4096) * 4096;
                }
                if (to_device) wait();
            }
            return true;
        };

        if (!populate_experts(vram_gids, true) || !populate_experts(host_gids, false)) {
            if (fp) std::fclose(fp);
            return false;
        }
        if (fp) std::fclose(fp);

        expert_dev = dalloc<InkSyExpertDev>(n_total);
        h2d(expert_dev, h_ed.data(), n_total * sizeof(InkSyExpertDev));
        wait();

        t_load = now_s() - t0;
        fprintf(stderr,
                "[ink_sycl] loaded %s (build=%s, minor=%u, dev=%s %.1f GiB) in %.2fs | VRAM experts=%d/%zu (%.2f GiB, %zu slabs), Host-USM=%d\n",
                p.c_str(), man.build.c_str(), format_minor, dev_caps().name.c_str(),
                gpu_mem / (1024.0 * 1024.0 * 1024.0), t_load,
                n_vram_experts, n_total, vram_expert_bytes / (1024.0 * 1024.0 * 1024.0),
                vram_exp_slabs.size(), n_host_experts);
        return true;
    }

    void prepare_preoriented_tables(const std::unordered_map<std::string, std::vector<uint8_t>>& ht) {
        const int D = man.c.hidden;
        const int core_len = man.c.core_end - man.c.core_start;
        auto fptr = [&](const std::string& k) -> const float* {
            return reinterpret_cast<const float*>(ht.at(k).data());
        };

        // 1. Cross-pass message queries [12, 4, D]
        std::vector<float> mq((size_t)core_len * 4 * D);
        const float* q_cp = fptr("model.llm.middle_cycle_cross_pass_queries");
        const float* kv_cp = fptr("model.llm.middle_cycle_cross_pass_kv_queries");
        for (int i = 0; i < core_len; ++i) {
            float* dst = mq.data() + (size_t)i * 4 * D;
            std::memcpy(dst + 0 * D, q_cp + (size_t)(2 * i) * D, D * 4);
            std::memcpy(dst + 1 * D, kv_cp + (size_t)(2 * i) * D, D * 4);
            std::memcpy(dst + 2 * D, kv_cp + (size_t)(2 * i + 1) * D, D * 4);
            std::memcpy(dst + 3 * D, q_cp + (size_t)(2 * i + 1) * D, D * 4);
        }
        msg_queries_dev = dalloc<float>(mq.size());
        h2d(msg_queries_dev, mq.data(), mq.size() * 4);

        // 2. Oriented static ShortConv weights [24, kSconvLayerFloats]
        std::vector<float> sc((size_t)man.c.n_phys * kSconvLayerFloats);
        for (int il = 0; il < man.c.n_phys; ++il) {
            std::string p = "model.llm.layers." + std::to_string(il);
            float* dst = sc.data() + (size_t)il * kSconvLayerFloats;
            std::memcpy(dst + kSconvAttnOff, fptr(p + ".attn_sconv.weight"), 2304 * 4 * 4);
            std::memcpy(dst + kSconvMlpOff, fptr(p + ".mlp_sconv.weight"), 2304 * 4 * 4);
            if (man.c.is_swa[il]) {
                std::memcpy(dst + kSconvKOff, fptr(p + ".attn.k_sconv.weight"), 1024 * 4 * 4);
                std::memcpy(dst + kSconvVOff, fptr(p + ".attn.v_sconv.weight"), 1024 * 4 * 4);
            } else {
                const float* kw = fptr(p + ".attn.k_sconv.static_w");
                const float* vw = fptr(p + ".attn.v_sconv.static_w");
                for (int c = 0; c < 1024; ++c) {
                    for (int t = 0; t < 4; ++t) {
                        dst[kSconvKOff + c * 4 + t] = kw[(3 - t) * 1024 + c];
                        dst[kSconvVOff + c * 4 + t] = vw[(3 - t) * 1024 + c];
                    }
                }
            }
        }
        static_sconv_dev = dalloc<float>(sc.size());
        h2d(static_sconv_dev, sc.data(), sc.size() * 4);

        // 3. RelPos padded [24, 16, 1025]
        std::vector<float> rp((size_t)man.c.n_phys * 16 * 1025, 0.0f);
        for (int il = 0; il < man.c.n_phys; ++il) {
            std::string p = "model.llm.layers." + std::to_string(il) + ".attn.rel_logits_proj.proj";
            const float* src = fptr(p);
            float* dst = rp.data() + (size_t)il * 16 * 1025;
            for (int r = 0; r < 16; ++r) std::memcpy(dst + r * 1025, src + r * 1024, 1024 * 4);
        }
        rel_proj_pad_dev = dalloc<float>(rp.size());
        h2d(rel_proj_pad_dev, rp.data(), rp.size() * 4);
        wait();
    }
    bool probe_only(const std::string& p, std::string& err) {
        FILE* fp = std::fopen(p.c_str(), "rb");
        if (!fp) { err = "cannot open " + p; return false; }
        uint8_t frame[4096];
        if (std::fread(frame, 1, 4096, fp) != 4096) { std::fclose(fp); err = "short header"; return false; }
        strata::artifact::StrataSectionEntry s0{};
        std::memcpy(&s0, frame + 0x10, 64);
        std::string mjson;
        bool ok = read_section0_zstd(fp, s0.file_offset, s0.byte_length, mjson, err);
        std::fclose(fp);
        if (!ok) return false;
        return parse_ink_manifest(mjson, man, err);
    }
};

using InkSyModel = InkSyclModel;

}  // namespace ink_sycl
