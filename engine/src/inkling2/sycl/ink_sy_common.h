// ink_sy_common.h - Portable Linux & Windows Intel Arc SYCL runtime, SG16 primitives, dp4a dot_acc,
// .strata v1.1 manifest reader, and 3x loop-weighted multi-slab VRAM expert placement (8/12/16/32 GB).
// CRITICAL BATTLEMAGE B70 RULE: NEVER uses reqd_sub_group_size(8) (triggers IGC ICE on bmg_g31).
// Strictly uses reqd_sub_group_size(16) across Alchemist (dg2) and Battlemage (bmg_g21 / bmg_g31)!
// 100% MIT clean-room implementation.
#pragma once

#include <sycl/sycl.hpp>
#if __has_include(<sycl/ext/oneapi/dot_product.hpp>)
#include <sycl/ext/oneapi/dot_product.hpp>
#define INK_SY_HAS_DOT_ACC 1
#else
#define INK_SY_HAS_DOT_ACC 0
#endif

#include <zstd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "../ink_json.h"
#include "strata/artifact/strata_reader.hpp"
#ifdef INK_SYCL_WITH_MODEL_FILE
#include "../ink_gguf_transcode.h"
#endif

#define INK_REQD_SG16 [[sycl::reqd_sub_group_size(16)]]

namespace ink_sycl {

using half = sycl::half;

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void sy_setenv(const char* k, const char* v, int overwrite = 1) {
#ifdef _WIN32
    if (!overwrite && std::getenv(k)) return;
    _putenv_s(k, v);
#else
    setenv(k, v, overwrite);
#endif
}

inline bool seek_abs(FILE* fp, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(fp, static_cast<__int64>(off), SEEK_SET) == 0;
#else
    return fseeko(fp, static_cast<off_t>(off), SEEK_SET) == 0;
#endif
}

// Work around oneAPI 2026.1.1 Level Zero V2 + oneMKL GEMM SIGSEGV (exit 139) on Battlemage (Arc Pro B70 / bmg_g31).
// Preserves any explicit user override of UR_LOADER_USE_LEVEL_ZERO_V2.
inline void ensure_l0_v1_default() {
    if (!std::getenv("UR_LOADER_USE_LEVEL_ZERO_V2")) {
        sy_setenv("UR_LOADER_USE_LEVEL_ZERO_V2", "0", 0);
    }
}

inline int& primary_gpu_index() {
    static int idx = 0;
    return idx;
}

inline void set_primary_gpu_index(int idx) {
    primary_gpu_index() = idx < 0 ? 0 : idx;
}

inline std::vector<sycl::device> enum_gpu_devices() {
    ensure_l0_v1_default();
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        const char* be = std::getenv("INK_SYCL_BACKEND");
        sy_setenv("ONEAPI_DEVICE_SELECTOR", (be && std::string(be) == "opencl") ? "opencl:gpu" : "level_zero:gpu", 1);
    }
    std::vector<sycl::device> all = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (all.empty()) return all;
    const char* be = std::getenv("INK_SYCL_BACKEND");
    const bool want_ocl = (be && std::string(be) == "opencl");
    const sycl::backend want_be = want_ocl ? sycl::backend::ext_oneapi_level_zero : sycl::backend::ext_oneapi_level_zero;
    const sycl::backend target_be = want_ocl ? sycl::backend::opencl : sycl::backend::ext_oneapi_level_zero;
    (void)want_be;
    std::vector<sycl::device> filtered;
    for (const auto& d : all) {
        if (d.get_backend() == target_be) filtered.push_back(d);
    }
    if (!filtered.empty()) return filtered;
    const sycl::backend b0 = all[0].get_backend();
    for (const auto& d : all) {
        if (d.get_backend() == b0) filtered.push_back(d);
    }
    return filtered;
}

inline sycl::queue& Q() {
    static sycl::queue q = [] {
        ensure_l0_v1_default();
        auto devs = enum_gpu_devices();
        if (!devs.empty()) {
            const int idx = std::min(std::max(primary_gpu_index(), 0), (int)devs.size() - 1);
            return sycl::queue{devs[idx], sycl::property_list{sycl::property::queue::in_order()}};
        }
        return sycl::queue{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order()}};
    }();
    return q;
}

inline bool probe_p2p(sycl::queue& q0, sycl::queue& q1) {
    const char* force_pinned = std::getenv("INK_SYCL_PEER_PINNED");
    if (force_pinned && force_pinned[0] == '1') return false;
#if defined(SYCL_EXT_ONEAPI_PEER_ACCESS)
    try {
        auto d0 = q0.get_device();
        auto d1 = q1.get_device();
        if (!d0.ext_oneapi_can_access_peer(d1, sycl::ext::oneapi::peer_access::access_supported) ||
            !d1.ext_oneapi_can_access_peer(d0, sycl::ext::oneapi::peer_access::access_supported)) {
            return false;
        }
        d0.ext_oneapi_enable_peer_access(d1);
        d1.ext_oneapi_enable_peer_access(d0);
        uint32_t h_tx[4] = {0x12345678u, 0xCAFEBABEu, 0xA5A55A5Au, 0xDEADBEEFu};
        uint32_t h_rx01[4] = {0, 0, 0, 0};
        uint32_t h_rx10[4] = {0, 0, 0, 0};
        uint32_t* p0 = sycl::malloc_device<uint32_t>(4, q0);
        uint32_t* p1 = sycl::malloc_device<uint32_t>(4, q1);
        bool ok = false;
        if (p0 && p1) {
            q0.memcpy(p0, h_tx, sizeof(h_tx)).wait();
            q1.memset(p1, 0, sizeof(h_tx)).wait();
            q1.memcpy(p1, p0, sizeof(h_tx)).wait();
            q1.memcpy(h_rx01, p1, sizeof(h_tx)).wait();
            q0.memset(p0, 0, sizeof(h_tx)).wait();
            q0.memcpy(p0, p1, sizeof(h_tx)).wait();
            q0.memcpy(h_rx10, p0, sizeof(h_tx)).wait();
            ok = (std::memcmp(h_tx, h_rx01, sizeof(h_tx)) == 0 &&
                  std::memcmp(h_tx, h_rx10, sizeof(h_tx)) == 0);
        }
        if (p0) sycl::free(p0, q0);
        if (p1) sycl::free(p1, q1);
        return ok;
    } catch (...) {
        return false;
    }
#else
    (void)q0; (void)q1;
    return false;
#endif
}

// Secondary async DMA/prefetch queue on the same Intel Arc context & device (overlaps H2D with compute!)
inline sycl::queue& Q_copy() {
    static sycl::queue qc = [] {
        return sycl::queue{Q().get_context(), Q().get_device(), sycl::property_list{sycl::property::queue::in_order()}};
    }();
    return qc;
}

inline void wait() { Q().wait_and_throw(); }

struct SyDevCaps {
    std::string name;
    std::string platform;
    size_t global_mem = 0;
    size_t max_alloc = 0;
    size_t local_mem = 0;
    uint32_t eu_count = 0;
    bool has_sg16 = false;
    bool has_xmx_sg16 = false;
    bool is_level_zero = false;
};

inline const SyDevCaps& dev_caps() {
    static SyDevCaps d = [] {
        SyDevCaps c;
        auto dev = Q().get_device();
        c.name = dev.get_info<sycl::info::device::name>();
        c.platform = dev.get_platform().get_info<sycl::info::platform::name>();
        c.global_mem = dev.get_info<sycl::info::device::global_mem_size>();
        c.max_alloc = dev.get_info<sycl::info::device::max_mem_alloc_size>();
        c.local_mem = dev.get_info<sycl::info::device::local_mem_size>();
        c.eu_count = dev.get_info<sycl::info::device::max_compute_units>();
        c.is_level_zero = (dev.get_backend() == sycl::backend::ext_oneapi_level_zero);
        for (auto s : dev.get_info<sycl::info::device::sub_group_sizes>()) {
            if (s == 16) c.has_sg16 = true;
        }
        const bool no_xmx = std::getenv("INK_SYCL_NO_XMX") && std::getenv("INK_SYCL_NO_XMX")[0] == '1';
        if (!no_xmx && c.has_sg16) {
            // Xe2 Battlemage (B580 bmg_g21, Arc Pro B60/B70 bmg_g31, Lunar Lake) & PVC natively support SG16 XMX
            if (c.name.find("B5") != std::string::npos || c.name.find("B6") != std::string::npos ||
                c.name.find("B7") != std::string::npos || c.name.find("Pro B") != std::string::npos ||
                c.name.find("Battlemage") != std::string::npos || c.name.find("BMG") != std::string::npos ||
                c.name.find("0xe2") != std::string::npos || c.name.find("0xE2") != std::string::npos ||
                c.name.find("140V") != std::string::npos || c.name.find("Data Center") != std::string::npos) {
                c.has_xmx_sg16 = true;
            }
        }
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
        fprintf(stderr, "[ink_sycl] dalloc OOM (%zu bytes, total=%zu)\n", cnt * sizeof(T), dev_bytes_allocated());
        std::exit(3);
    }
    dev_bytes_allocated() += cnt * sizeof(T);
    return p;
}

template <class T>
inline T* halloc(size_t n) {
    size_t cnt = n ? n : 1;
    T* p = sycl::malloc_host<T>(cnt, Q());
    if (!p) {
        fprintf(stderr, "[ink_sycl] halloc OOM (%zu bytes)\n", cnt * sizeof(T));
        std::exit(3);
    }
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

// SG16 sub-group & work-group reductions
template <class It>
inline float sg_sum(const It& it, float v) {
    return sycl::reduce_over_group(it.get_sub_group(), v, sycl::plus<float>());
}

// Work-group sum for 1-D or 2-D group whose size is a multiple of 16 (up to 256 -> max 16 subgroups)
template <class It, class P>
inline float wg_sum16(const It& it, float v, P sh) {
    auto sg = it.get_sub_group();
    v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
    const int w = (int)sg.get_group_linear_id();
    const int l = (int)sg.get_local_linear_id();
    const int nw = (int)sg.get_group_linear_range();
    sycl::group_barrier(it.get_group());
    if (l == 0) sh[w] = v;
    sycl::group_barrier(it.get_group());
    float r = (l < nw) ? (float)sh[l] : 0.0f;
    return sycl::reduce_over_group(sg, r, sycl::plus<float>());
}

// Deterministic int8x4 dp4a helper using sycl::ext::oneapi::dot_acc (when available) or portable fallback
inline int32_t dp4a_s8(int32_t a_packed, int32_t b_packed, int32_t acc) {
#if INK_SY_HAS_DOT_ACC
    return sycl::ext::oneapi::dot_acc(a_packed, b_packed, acc);
#else
    const int8_t a0 = (int8_t)(a_packed & 0xFF), a1 = (int8_t)((a_packed >> 8) & 0xFF),
                 a2 = (int8_t)((a_packed >> 16) & 0xFF), a3 = (int8_t)((a_packed >> 24) & 0xFF);
    const int8_t b0 = (int8_t)(b_packed & 0xFF), b1 = (int8_t)((b_packed >> 8) & 0xFF),
                 b2 = (int8_t)((b_packed >> 16) & 0xFF), b3 = (int8_t)((b_packed >> 24) & 0xFF);
    return acc + (int32_t)a0 * b0 + (int32_t)a1 * b1 + (int32_t)a2 * b2 + (int32_t)a3 * b3;
#endif
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
    FMT_SF4 = 15, FMT_MX8 = 16, FMT_GGUF_RAW = 20,
    FMT_Q8ROW = 100, FMT_F32 = 101, FMT_BF16 = 102,
};

inline bool gguf_type_has_sycl_kernel(uint32_t t) noexcept {
    switch (t) {
        case 0:   // F32
        case 1:   // F16
        case 2:   // Q4_0
        case 8:   // Q8_0
        case 10:  // Q2_K
        case 11:  // Q3_K
        case 12:  // Q4_K
        case 13:  // Q5_K
        case 14:  // Q6_K
        case 18:  // IQ3_XXS
        case 20:  // IQ4_NL
        case 23:  // IQ4_XS
        case 30:  // BF16
        case 39:  // MXFP4
            return true;
        default:
            return false;
    }
}

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
    if (s == "GGUF_RAW") return FMT_GGUF_RAW;
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

    // Dual-GPU Expert Parallelism (--gpus 2 / --gpus 0,1 / --devices 0,1)
    bool ep2_active = false;
    int ep2_gpu0_idx = 0;
    int ep2_gpu1_idx = -1;
    double ep2_split = 0.5;
    bool ep2_p2p = false;
    std::string ep2_dev1_name;
    size_t ep2_dev1_mem = 0;
    sycl::queue* q_peer = nullptr;
    std::vector<uint8_t*> peer_vram_exp_slabs;
    InkSyExpertDev* expert_dev_g1 = nullptr;   // [24 * 128] on GPU 1 (non-null body_ptr ONLY for GPU 1 experts!)
    std::vector<InkSyExpertDev> h_ed_g1;
    int n_vram_experts_g0 = 0;
    int n_vram_experts_g1 = 0;
    size_t vram_expert_bytes_g1 = 0;

    InkSyExpertDev* expert_dev = nullptr;  // [24 * 128] on device 0
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
        if (q_peer) {
            try { q_peer->wait(); } catch (...) {}
            for (auto* p : peer_vram_exp_slabs) if (p) sycl::free(p, *q_peer);
            if (expert_dev_g1) sycl::free(expert_dev_g1, *q_peer);
            delete q_peer;
            q_peer = nullptr;
        }
        peer_vram_exp_slabs.clear();
        expert_dev_g1 = nullptr;
        for (auto* p : trunk_slabs) dfree(p);
        for (auto* p : vram_exp_slabs) dfree(p);
        for (auto* p : host_exp_slabs) dfree(p);
        dfree(expert_dev);
        dfree(msg_queries_dev);
        dfree(static_sconv_dev);
        dfree(rel_proj_pad_dev);
    }

    bool load(const std::string& p, double vram_experts_gb_override, std::string& err, size_t extra_reserve_bytes = 0,
              int req_gpus = 1, const std::vector<int>& req_gpu_ids = {}, double req_ep_split = 0.5) {
        double t0 = now_s();
        path = p;
        FILE* fp = std::fopen(p.c_str(), "rb");
        if (!fp) { err = "cannot open " + p; return false; }

        uint8_t frame[4096];
        if (std::fread(frame, 1, 4096, fp) != 4096) { std::fclose(fp); err = "short header"; return false; }
        static const uint8_t kMagic[8] = {'S', 'T', 'R', 'A', 'T', 'A', 0x00, 0x01};
        if (std::memcmp(frame, kMagic, 8) != 0) { std::fclose(fp); err = "bad .strata magic"; return false; }
        uint32_t hdr_flags = 0;
        std::memcpy(&hdr_flags, frame + 8, 4);
        format_minor = (uint8_t)(hdr_flags & 0xFFu);

        strata::artifact::StrataSectionEntry secs[6]{};
        for (int i = 0; i < 6; ++i) std::memcpy(&secs[i], frame + 0x10 + i * 64, 64);

        std::string mjson;
        if (!read_section0_zstd(fp, secs[0].file_offset, secs[0].byte_length, mjson, err)) {
            std::fclose(fp);
            return false;
        }
        if (!parse_ink_manifest(mjson, man, err)) { std::fclose(fp); return false; }

        // 1. Upload Section 1 (Trunk + DSpark) into <= 3.2 GiB device slabs (safe under Arc 4 GiB max_alloc!)
        const size_t kMaxSlab = std::min<size_t>(dev_caps().max_alloc ? (dev_caps().max_alloc * 7 / 8) : (3200ull << 20), 3200ull << 20);
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
                    if (!seek_abs(fp, secs[1].file_offset + ti->off) || std::fread(staging.data(), 1, ti->bytes, fp) != ti->bytes) {
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
            std::fclose(fp);
            return false;
        }
        prepare_preoriented_tables(host_tmp);

        // 2. Read Section 3 expert directory + 3x loop-weighted vram_load_order[3072]
        const size_t n_total = (size_t)man.c.n_phys * (size_t)man.c.n_experts;
        std::vector<strata::artifact::StrataExpertEntry> ents_by_gid(n_total);
        std::vector<uint16_t> order;
        if (secs[3].byte_length > 0) {
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

        // 3. Auto-compute VRAM expert budget (and optional Dual-GPU EP on 2 Intel Arc GPUs!)
        const size_t gpu_mem = dev_caps().global_mem;
        const size_t base_reserve = gpu_mem >= (24ull << 30) ? (2560ull << 20)
                                  : gpu_mem >= (11ull << 30) ? (1536ull << 20)
                                                             : (1024ull << 20);
        const size_t reserve = (extra_reserve_bytes > 0) ? (extra_reserve_bytes + (768ull << 20)) : base_reserve;
        const size_t used = dev_bytes_allocated();
        size_t vram_budget = (gpu_mem > used + reserve) ? (gpu_mem - used - reserve) : 0;
        if (vram_experts_gb_override >= 0.0) {
            vram_budget = (size_t)(vram_experts_gb_override * (1024.0 * 1024.0 * 1024.0));
        }

        // Optional Dual-GPU Expert Parallelism setup (--gpus 2 / --gpus 0,1 / --devices 0,1)
        ep2_active = false;
        ep2_gpu0_idx = primary_gpu_index();
        ep2_gpu1_idx = -1;
        ep2_split = std::clamp(req_ep_split, 0.1, 0.9);
        ep2_p2p = false;
        size_t vram_budget_g1 = 0;
        size_t kMaxSlab_g1 = kMaxSlab;
        if (req_gpus >= 2 || req_gpu_ids.size() >= 2) {
            auto all_devs = enum_gpu_devices();
            if (all_devs.size() >= 2) {
                int g0 = std::min(std::max(ep2_gpu0_idx, 0), (int)all_devs.size() - 1);
                int g1 = -1;
                if (req_gpu_ids.size() >= 2) {
                    for (size_t k = 1; k < req_gpu_ids.size(); ++k) {
                        if (req_gpu_ids[k] >= 0 && req_gpu_ids[k] < (int)all_devs.size() && req_gpu_ids[k] != g0) {
                            g1 = req_gpu_ids[k];
                            break;
                        }
                    }
                }
                if (g1 < 0) g1 = (g0 == 0) ? 1 : 0;
                try {
                    auto dev1 = all_devs[g1];
                    q_peer = new sycl::queue(dev1, sycl::property_list{sycl::property::queue::in_order()});
                    ep2_gpu0_idx = g0;
                    ep2_gpu1_idx = g1;
                    ep2_dev1_name = dev1.get_info<sycl::info::device::name>();
                    ep2_dev1_mem = dev1.get_info<sycl::info::device::global_mem_size>();
                    const size_t ma1 = dev1.get_info<sycl::info::device::max_mem_alloc_size>();
                    kMaxSlab_g1 = std::min<size_t>(ma1 ? (ma1 * 7 / 8) : (3200ull << 20), 3200ull << 20);
                    const size_t res1 = (768ull << 20);  // GPU 1 scratch + runtime headroom
                    vram_budget_g1 = (ep2_dev1_mem > res1) ? (ep2_dev1_mem - res1) : 0;
                    if (vram_experts_gb_override >= 0.0) {
                        vram_budget_g1 = std::min(vram_budget_g1, (size_t)(vram_experts_gb_override * (1024.0 * 1024.0 * 1024.0)));
                    }
                    if (vram_budget_g1 > 0) {
                        ep2_p2p = probe_p2p(Q(), *q_peer);
                        ep2_active = true;
                    } else {
                        delete q_peer;
                        q_peer = nullptr;
                    }
                } catch (const sycl::exception& e) {
                    fprintf(stderr, "[ink_sycl] Dual-GPU EP init on GPU%d failed (%s); falling back to 1 GPU\n", g1, e.what());
                    if (q_peer) { delete q_peer; q_peer = nullptr; }
                }
            } else {
                fprintf(stderr, "[ink_sycl] --gpus %d requested but only %zu SYCL GPU(s) visible; running single-GPU\n",
                        req_gpus, all_devs.size());
            }
        }

        // Assign experts in exact 3x loop-weighted priority order!
        std::vector<uint8_t> in_vram(n_total, 0);
        std::vector<size_t> vram0_gids, vram1_gids, host_gids;
        size_t vram0_acc = 0, vram1_acc = 0;
        for (uint16_t gid : order) {
            if (gid >= n_total) continue;
            const size_t eb = ((ents_by_gid[gid].body_bytes + 4095) / 4096) * 4096;
            if (ep2_active) {
                const int il = (int)(gid / (size_t)man.c.n_experts);
                const int eid = (int)(gid % (size_t)man.c.n_experts);
                const uint32_t hh = (static_cast<uint32_t>(eid) * 2654435761u + static_cast<uint32_t>(il) * 40503u) >> 8;
                int pref = ((hh % 10000u) < static_cast<uint32_t>(ep2_split * 10000.0)) ? 1 : 0;
                if (pref == 1) {
                    if (vram1_acc + eb <= vram_budget_g1) {
                        in_vram[gid] = 2; vram1_gids.push_back(gid); vram1_acc += eb;
                    } else if (vram0_acc + eb <= vram_budget) {
                        in_vram[gid] = 1; vram0_gids.push_back(gid); vram0_acc += eb;
                    } else {
                        host_gids.push_back(gid);
                    }
                } else {
                    if (vram0_acc + eb <= vram_budget) {
                        in_vram[gid] = 1; vram0_gids.push_back(gid); vram0_acc += eb;
                    } else if (vram1_acc + eb <= vram_budget_g1) {
                        in_vram[gid] = 2; vram1_gids.push_back(gid); vram1_acc += eb;
                    } else {
                        host_gids.push_back(gid);
                    }
                }
            } else {
                if (vram0_acc + eb <= vram_budget) {
                    in_vram[gid] = 1;
                    vram0_gids.push_back(gid);
                    vram0_acc += eb;
                } else {
                    host_gids.push_back(gid);
                }
            }
        }
        if (ep2_active && vram1_gids.empty()) {
            ep2_active = false;
            delete q_peer;
            q_peer = nullptr;
        }
        n_vram_experts_g0 = (int)vram0_gids.size();
        n_vram_experts_g1 = (int)vram1_gids.size();
        n_vram_experts = n_vram_experts_g0 + n_vram_experts_g1;
        n_host_experts = (int)host_gids.size();
        vram_expert_bytes = vram0_acc + vram1_acc;
        vram_expert_bytes_g1 = vram1_acc;

        h_ed.assign(n_total, InkSyExpertDev{});
        h_ed_g1.assign(n_total, InkSyExpertDev{});
        host_body_ptrs.assign(n_total, nullptr);
        expert_body_bytes.assign(n_total, 0);
        max_expert_bytes = 0;
        // mode: 0 = host USM, 1 = GPU 0 VRAM, 2 = GPU 1 VRAM
        auto populate_experts = [&](const std::vector<size_t>& gids, int mode) -> bool {
            size_t i = 0;
            std::vector<uint8_t> buf;
            const size_t slab_cap = (mode == 2) ? kMaxSlab_g1 : kMaxSlab;
            while (i < gids.size()) {
                size_t slab_sz = 0, start = i;
                while (i < gids.size()) {
                    size_t eb = ((ents_by_gid[gids[i]].body_bytes + 4095) / 4096) * 4096;
                    if (slab_sz > 0 && slab_sz + eb > slab_cap) break;
                    slab_sz += eb;
                    ++i;
                }
                uint8_t* slab = nullptr;
                if (mode == 1) {
                    slab = dalloc<uint8_t>(slab_sz);
                    vram_exp_slabs.push_back(slab);
                } else if (mode == 2) {
                    slab = sycl::malloc_device<uint8_t>(slab_sz ? slab_sz : 1, *q_peer);
                    if (!slab) { err = "peer GPU1 slab OOM"; return false; }
                    peer_vram_exp_slabs.push_back(slab);
                } else {
                    slab = halloc<uint8_t>(slab_sz);
                    host_exp_slabs.push_back(slab);
                }
                size_t off = 0;
                for (size_t j = start; j < i; ++j) {
                    const size_t gid = gids[j];
                    const auto& e = ents_by_gid[gid];
                    if (mode == 1 || mode == 2) {
                        buf.resize(e.body_bytes);
                        if (!seek_abs(fp, secs[4].file_offset + e.body_offset) || std::fread(buf.data(), 1, e.body_bytes, fp) != e.body_bytes) {
                            err = "short read expert body"; return false;
                        }
                        if (mode == 1) {
                            h2d(slab + off, buf.data(), e.body_bytes);
                            wait();
                        } else {
                            q_peer->memcpy(slab + off, buf.data(), e.body_bytes).wait();
                        }
                    } else {
                        if (!seek_abs(fp, secs[4].file_offset + e.body_offset) || std::fread(slab + off, 1, e.body_bytes, fp) != e.body_bytes) {
                            err = "short read host expert body"; return false;
                        }
                        host_body_ptrs[gid] = slab + off;
                    }
                    expert_body_bytes[gid] = e.body_bytes;
                    max_expert_bytes = std::max<size_t>(max_expert_bytes, e.body_bytes);
                    if (mode == 2) {
                        // GPU 1 owns this expert: GPU 0 descriptor has nullptr body_ptr & in_vram=2 so GPU0 & CPU skip it!
                        h_ed[gid] = InkSyExpertDev{nullptr, e.dn_offset, e.gu_code_id, e.dn_code_id, 2, 0};
                        h_ed_g1[gid] = InkSyExpertDev{slab + off, e.dn_offset, e.gu_code_id, e.dn_code_id, 2, 0};
                    } else {
                        h_ed[gid] = InkSyExpertDev{slab + off, e.dn_offset, e.gu_code_id, e.dn_code_id, (uint8_t)(mode == 1 ? 1 : 0), 0};
                    }
                    off += ((e.body_bytes + 4095) / 4096) * 4096;
                }
                if (mode == 1) wait();
                else if (mode == 2) q_peer->wait();
            }
            return true;
        };

        if (!populate_experts(vram0_gids, 1) || !populate_experts(vram1_gids, 2) || !populate_experts(host_gids, 0)) {
            std::fclose(fp);
            return false;
        }
        std::fclose(fp);

        expert_dev = dalloc<InkSyExpertDev>(n_total);
        h2d(expert_dev, h_ed.data(), n_total * sizeof(InkSyExpertDev));
        wait();
        if (ep2_active) {
            expert_dev_g1 = sycl::malloc_device<InkSyExpertDev>(n_total, *q_peer);
            if (!expert_dev_g1) { err = "peer GPU1 expert_dev OOM"; return false; }
            q_peer->memcpy(expert_dev_g1, h_ed_g1.data(), n_total * sizeof(InkSyExpertDev)).wait();
        }

        t_load = now_s() - t0;
        if (ep2_active) {
            fprintf(stderr,
                    "[ink_sycl] loaded %s (build=%s, minor=%u) in %.2fs | Dual-GPU EP2 (GPU%d %s %.1f GiB + GPU%d %s %.1f GiB, split=%.2f, %s) | VRAM experts=%d/%zu (GPU0=%d/%.2f GiB, GPU1=%d/%.2f GiB), Host-USM=%d\n",
                    p.c_str(), man.build.c_str(), format_minor, t_load,
                    ep2_gpu0_idx, dev_caps().name.c_str(), gpu_mem / 1073741824.0,
                    ep2_gpu1_idx, ep2_dev1_name.c_str(), ep2_dev1_mem / 1073741824.0,
                    ep2_split, ep2_p2p ? "p2p" : "pinned",
                    n_vram_experts, n_total, n_vram_experts_g0, vram0_acc / 1073741824.0,
                    n_vram_experts_g1, vram1_acc / 1073741824.0, n_host_experts);
        } else {
            fprintf(stderr,
                    "[ink_sycl] loaded %s (build=%s, minor=%u, dev=%s %.1f GiB) in %.2fs | VRAM experts=%d/%zu (%.2f GiB, %zu slabs), Host-USM=%d\n",
                    p.c_str(), man.build.c_str(), format_minor, dev_caps().name.c_str(),
                    gpu_mem / (1024.0 * 1024.0 * 1024.0), t_load,
                    n_vram_experts, n_total, vram_expert_bytes / (1024.0 * 1024.0 * 1024.0),
                    vram_exp_slabs.size(), n_host_experts);
        }
        return true;
    }

#ifdef INK_SYCL_WITH_MODEL_FILE
    // Unified loader for Community/Unsloth .gguf and Official HF .safetensors via InkModelFile.
    // Streams & transcodes GGUF_RAW experts to S4_K/S6_K/S8_K across CPU threads with zero temp files!
    bool load_from_model_file(const ink2::InkModelFile& mf, double vram_experts_gb_override,
                              std::string& err, size_t extra_reserve_bytes = 0) {
        double t0 = now_s();
        path = mf.path;
        format_minor = 1;

        // 1. Copy InkManifest & config from InkModelFile
        man = InkManifest{};
        man.build = mf.m.build;
        man.chat_template = mf.m.chat_template;
        const auto& sc = mf.m.c;
        InkCfg& c = man.c;
        c.n_phys = sc.n_phys;
        c.n_visits = sc.n_visits;
        c.hidden = sc.hidden;
        c.inter = sc.inter;
        c.vocab_padded = sc.vocab_padded;
        c.vocab_unpadded = sc.vocab_unpadded;
        c.n_head = sc.n_head;
        c.n_kv = sc.n_kv;
        c.head_dim = sc.head_dim;
        c.d_rel = sc.d_rel;
        c.rel_extent = sc.rel_extent;
        c.swa_win = sc.swa_win;
        c.block_size = sc.block_size;
        c.n_experts = sc.n_experts;
        c.top_k = sc.top_k;
        c.core_start = sc.core_start;
        c.core_end = sc.core_end;
        c.num_passes = sc.num_passes;
        c.gate_rank = sc.gate_rank;
        c.eos_id = sc.eos_token_id;
        c.image_pad_id = sc.image_token_id;
        c.rms_eps = sc.rms_eps;
        c.qk_l2_clip = sc.qk_l2_clip;
        c.rel_l2_clip = sc.rel_l2_clip;
        c.log_scale_alpha = sc.log_scale_alpha;
        c.log_scale_floor = sc.log_scale_floor;
        c.logit_softcap = sc.logit_softcap;
        c.route_scale = sc.route_scale;
        c.is_swa.assign(c.n_phys, 0);
        for (int i = 0; i < c.n_phys && i < (int)sc.is_swa.size(); ++i) c.is_swa[i] = sc.is_swa[i] ? 1 : 0;

        c.has_dspark = mf.m.dspark.present && !mf.m.dspark_tensors.empty();
        c.dspark_layers = mf.m.dspark.n_layers;
        c.dspark_inter = mf.m.dspark.inter;
        c.dspark_block_size = mf.m.dspark.block_size;
        c.dspark_markov_rank = mf.m.dspark.markov_rank;
        c.dspark_rope_theta = mf.m.dspark.rope_theta;
        c.dspark_swa_win = mf.m.dspark.swa_win;
        for (int i = 0; i < 4 && i < (int)mf.m.dspark.target_visits.size(); ++i) c.dspark_taps[i] = mf.m.dspark.target_visits[i];
        for (int i = 0; i < 3 && i < (int)mf.m.dspark.is_swa.size(); ++i) c.dspark_is_swa[i] = mf.m.dspark.is_swa[i] ? 1 : 0;

        for (const auto& kv : mf.m.trunk) {
            InkTensorInfo ti;
            ti.name = kv.second.name.empty() ? kv.first : kv.second.name;
            ti.fmt = kv.second.fmt;
            ti.shape = kv.second.shape;
            ti.off = kv.second.off;
            ti.bytes = kv.second.bytes;
            man.T[kv.first] = std::move(ti);
        }
        for (const auto& kv : mf.m.dspark_tensors) {
            InkTensorInfo ti;
            ti.name = kv.second.name.empty() ? kv.first : kv.second.name;
            ti.fmt = kv.second.fmt;
            ti.shape = kv.second.shape;
            ti.off = kv.second.off;
            ti.bytes = kv.second.bytes;
            man.D[kv.first] = std::move(ti);
        }

        // 2. Upload Section 1 (Trunk + DSpark) into <= 3.2 GiB device slabs
        std::vector<uint8_t> sec1;
        if (!mf.read_section1(sec1, err)) return false;

        const size_t kMaxSlab = std::min<size_t>(dev_caps().max_alloc ? (dev_caps().max_alloc * 7 / 8) : (3200ull << 20), 3200ull << 20);
        auto upload_tensor_map = [&](std::unordered_map<std::string, InkTensorInfo>& mp,
                                     std::unordered_map<std::string, std::vector<uint8_t>>& host_copies) -> bool {
            std::vector<InkTensorInfo*> ordered;
            ordered.reserve(mp.size());
            for (auto& kv : mp) ordered.push_back(&kv.second);
            std::sort(ordered.begin(), ordered.end(), [](const InkTensorInfo* a, const InkTensorInfo* b) { return a->off < b->off; });

            size_t idx = 0;
            while (idx < ordered.size()) {
                size_t slab_sz = 0, start = idx;
                while (idx < ordered.size()) {
                    size_t tb = ((ordered[idx]->bytes + 63) / 64) * 64;
                    if (slab_sz > 0 && slab_sz + tb > kMaxSlab) break;
                    slab_sz += tb;
                    ++idx;
                }
                uint8_t* dev_slab = dalloc<uint8_t>(slab_sz);
                trunk_slabs.push_back(dev_slab);
                size_t cur_off = 0;
                for (size_t j = start; j < idx; ++j) {
                    InkTensorInfo* ti = ordered[j];
                    if (ti->off + ti->bytes > sec1.size()) {
                        err = "section1 tensor out of bounds: " + ti->name;
                        return false;
                    }
                    const uint8_t* src = sec1.data() + ti->off;
                    h2d(dev_slab + cur_off, src, ti->bytes);
                    ti->dev_ptr = dev_slab + cur_off;
                    if (ti->name.find("sconv") != std::string::npos ||
                        ti->name.find("middle_cycle_cross_pass") != std::string::npos ||
                        ti->name.find("rel_logits_proj") != std::string::npos) {
                        host_copies[ti->name].assign(src, src + ti->bytes);
                    }
                    cur_off += ((ti->bytes + 63) / 64) * 64;
                }
                wait();
            }
            return true;
        };

        std::unordered_map<std::string, std::vector<uint8_t>> host_tmp;
        if (!upload_tensor_map(man.T, host_tmp) || !upload_tensor_map(man.D, host_tmp)) return false;
        sec1.clear();
        sec1.shrink_to_fit();
        prepare_preoriented_tables(host_tmp);

        // 3. Plan expert directory (preserving native lossless FMT_GGUF_RAW for all 14 supported GGML types!)
        //    & 3x loop-weighted VRAM vs Host-USM placement
        const size_t n_total = (size_t)man.c.n_phys * (size_t)man.c.n_experts;
        const bool force_sk_tc = (std::getenv("INK_SYCL_FORCE_SK_TRANSCODE") && std::atoi(std::getenv("INK_SYCL_FORCE_SK_TRANSCODE")) != 0);
        std::vector<strata::artifact::StrataExpertEntry> adj_dir = mf.expert_directory();
        std::vector<uint8_t> native_gguf(adj_dir.size(), 0);
        size_t n_native_gguf = 0;
        if (mf.kind == ink2::ModelSourceKind::Gguf && mf.gguf_adapter) {
            const size_t h_sz = (size_t)man.c.hidden;
            const size_t i_sz = (size_t)man.c.inter;
            for (size_t i = 0; i < adj_dir.size(); ++i) {
                auto& e = adj_dir[i];
                if (e.gu_code_id != FMT_GGUF_RAW && e.dn_code_id != FMT_GGUF_RAW) continue;
                const int il = (int)(i / (size_t)man.c.n_experts);
                if (il < 0 || il >= (int)mf.gguf_adapter->layers.size()) continue;
                const auto& lm = mf.gguf_adapter->layers[il];
                if (!force_sk_tc && gguf_type_has_sycl_kernel(lm.gu_type) && gguf_type_has_sycl_kernel(lm.dn_type)) {
                    native_gguf[i] = 1;
                    ++n_native_gguf;
                } else {
                    const auto gu_c = ink2::transcode::choose_sk_code_for_ggml(lm.gu_type);
                    const auto dn_c = ink2::transcode::choose_sk_code_for_ggml(lm.dn_type);
                    uint32_t dn_off = 0, body_b = 0;
                    ink2::transcode::transcoded_sk_expert_layout(gu_c, dn_c, h_sz, i_sz, dn_off, body_b);
                    e.gu_code_id = (uint8_t)gu_c;
                    e.dn_code_id = (uint8_t)dn_c;
                    e.dn_offset  = dn_off;
                    e.body_bytes = body_b;
                }
            }
            if (n_native_gguf > 0) {
                fprintf(stderr, "[ink_sycl] native lossless GGUF_RAW SG16+XMX kernels active for %zu/%zu experts (zero S_K transcode)\n",
                        n_native_gguf, adj_dir.size());
            }
        }
        auto read_expert_sy = [&](size_t di, std::vector<uint8_t>& out_b,
                                  uint32_t& out_dn, uint8_t& out_gu, uint8_t& out_dnc, std::string& lerr) -> bool {
            if (di < native_gguf.size() && native_gguf[di]) {
                if (!mf.read_expert_body(di, out_b, lerr)) return false;
                out_dn  = adj_dir[di].dn_offset;
                out_gu  = adj_dir[di].gu_code_id;
                out_dnc = adj_dir[di].dn_code_id;
                return true;
            }
            return ink2::transcode::read_or_transcode_expert(mf, di, out_b, out_dn, out_gu, out_dnc, lerr);
        };

        std::vector<strata::artifact::StrataExpertEntry> ents_by_gid(n_total);
        std::vector<size_t> dir_idx_by_gid(n_total, (size_t)-1);
        for (size_t i = 0; i < std::min(adj_dir.size(), n_total); ++i) {
            const size_t layer = i / (size_t)man.c.n_experts;
            const size_t gid = layer * (size_t)man.c.n_experts + adj_dir[i].expert_id;
            if (gid < n_total) {
                ents_by_gid[gid] = adj_dir[i];
                dir_idx_by_gid[gid] = i;
            }
        }

        std::vector<uint16_t> order;
        if (mf.vram_load_order.size() == n_total) {
            order.reserve(n_total);
            for (uint16_t di : mf.vram_load_order) {
                if (di < adj_dir.size()) {
                    const size_t layer = (size_t)di / (size_t)man.c.n_experts;
                    const size_t gid = layer * (size_t)man.c.n_experts + adj_dir[di].expert_id;
                    if (gid < n_total) order.push_back((uint16_t)gid);
                }
            }
        }
        if (order.size() != n_total) {
            order.clear();
            for (int l = man.c.core_start; l < man.c.core_end; ++l)
                for (int e = 0; e < man.c.n_experts; ++e) order.push_back((uint16_t)(l * man.c.n_experts + e));
            for (int l = 0; l < man.c.core_start; ++l)
                for (int e = 0; e < man.c.n_experts; ++e) order.push_back((uint16_t)(l * man.c.n_experts + e));
            for (int l = man.c.core_end; l < man.c.n_phys; ++l)
                for (int e = 0; e < man.c.n_experts; ++e) order.push_back((uint16_t)(l * man.c.n_experts + e));
        }

        const size_t gpu_mem = dev_caps().global_mem;
        const size_t base_reserve = gpu_mem >= (24ull << 30) ? (2560ull << 20)
                                  : gpu_mem >= (11ull << 30) ? (1536ull << 20)
                                                             : (1024ull << 20);
        const size_t reserve = (extra_reserve_bytes > 0) ? (extra_reserve_bytes + (768ull << 20)) : base_reserve;
        const size_t used = dev_bytes_allocated();
        size_t vram_budget = (gpu_mem > used + reserve) ? (gpu_mem - used - reserve) : 0;
        if (vram_experts_gb_override >= 0.0) {
            vram_budget = (size_t)(vram_experts_gb_override * (1024.0 * 1024.0 * 1024.0));
        }

        std::vector<size_t> vram_gids, host_gids;
        size_t vram_acc = 0;
        for (uint16_t gid : order) {
            if (gid >= n_total || dir_idx_by_gid[gid] == (size_t)-1) continue;
            const size_t eb = ents_by_gid[gid].body_bytes;
            if (vram_acc + eb <= vram_budget) {
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

        const int n_threads = ink2::transcode::default_transcode_threads();
        const size_t kBatch = 16;  // <= ~64-128 MiB transient host scratch

        auto populate_experts = [&](const std::vector<size_t>& gids, bool to_device) -> bool {
            size_t i = 0;
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

                // Precompute aligned offsets inside slab
                const size_t n_in_slab = i - start;
                std::vector<size_t> slab_offs(n_in_slab, 0);
                size_t cur = 0;
                for (size_t k = 0; k < n_in_slab; ++k) {
                    slab_offs[k] = cur;
                    cur += ((ents_by_gid[gids[start + k]].body_bytes + 4095) / 4096) * 4096;
                }

                if (!to_device) {
                    // Transcode straight into host-pinned USM slab in parallel!
                    std::vector<uint32_t> dn_offs(n_in_slab, 0);
                    std::vector<uint8_t> gu_codes(n_in_slab, 0), dn_codes(n_in_slab, 0);
                    std::vector<uint32_t> body_lens(n_in_slab, 0);
                    bool ok = ink2::transcode::parallel_run(n_in_slab, n_threads, [&](size_t k, std::string& lerr) -> bool {
                        const size_t gid = gids[start + k];
                        std::vector<uint8_t> tmp;
                        if (!read_expert_sy(dir_idx_by_gid[gid], tmp, dn_offs[k], gu_codes[k], dn_codes[k], lerr)) {
                            return false;
                        }
                        if (tmp.size() > ents_by_gid[gid].body_bytes) {
                            lerr = "transcoded expert exceeded planned slab size";
                            return false;
                        }
                        std::memcpy(slab + slab_offs[k], tmp.data(), tmp.size());
                        body_lens[k] = (uint32_t)tmp.size();
                        return true;
                    }, err);
                    if (!ok) return false;
                    for (size_t k = 0; k < n_in_slab; ++k) {
                        const size_t gid = gids[start + k];
                        host_body_ptrs[gid] = slab + slab_offs[k];
                        expert_body_bytes[gid] = body_lens[k];
                        max_expert_bytes = std::max<size_t>(max_expert_bytes, body_lens[k]);
                        h_ed[gid] = InkSyExpertDev{slab + slab_offs[k], dn_offs[k], gu_codes[k], dn_codes[k], 0, 0};
                    }
                } else {
                    // Transcode in bounded batches of kBatch experts in parallel, then stream H2D
                    for (size_t b0 = 0; b0 < n_in_slab; b0 += kBatch) {
                        const size_t bn = std::min(kBatch, n_in_slab - b0);
                        std::vector<std::vector<uint8_t>> bufs(bn);
                        std::vector<uint32_t> dn_offs(bn, 0);
                        std::vector<uint8_t> gu_codes(bn, 0), dn_codes(bn, 0);
                        bool ok = ink2::transcode::parallel_run(bn, n_threads, [&](size_t j, std::string& lerr) -> bool {
                            const size_t gid = gids[start + b0 + j];
                            return read_expert_sy(dir_idx_by_gid[gid], bufs[j], dn_offs[j], gu_codes[j], dn_codes[j], lerr);
                        }, err);
                        if (!ok) return false;
                        for (size_t j = 0; j < bn; ++j) {
                            const size_t gid = gids[start + b0 + j];
                            if (bufs[j].size() > ents_by_gid[gid].body_bytes) {
                                err = "transcoded expert exceeded planned VRAM slab size";
                                return false;
                            }
                            h2d(slab + slab_offs[b0 + j], bufs[j].data(), bufs[j].size());
                            expert_body_bytes[gid] = (uint32_t)bufs[j].size();
                            max_expert_bytes = std::max<size_t>(max_expert_bytes, bufs[j].size());
                            h_ed[gid] = InkSyExpertDev{slab + slab_offs[b0 + j], dn_offs[j], gu_codes[j], dn_codes[j], 1, 0};
                        }
                        wait();
                    }
                }
            }
            return true;
        };

        if (!populate_experts(vram_gids, true) || !populate_experts(host_gids, false)) return false;

        expert_dev = dalloc<InkSyExpertDev>(n_total);
        h2d(expert_dev, h_ed.data(), n_total * sizeof(InkSyExpertDev));
        wait();

        t_load = now_s() - t0;
        fprintf(stderr,
                "[ink_sycl] loaded %s (src=%s, build=%s, dev=%s %.1f GiB) in %.2fs | VRAM experts=%d/%zu (%.2f GiB, %zu slabs), Host-USM=%d\n",
                path.c_str(), ink2::source_kind_name(mf.kind), man.build.c_str(), dev_caps().name.c_str(),
                gpu_mem / (1024.0 * 1024.0 * 1024.0), t_load,
                n_vram_experts, n_total, vram_expert_bytes / (1024.0 * 1024.0 * 1024.0),
                vram_exp_slabs.size(), n_host_experts);
        return true;
    }
#endif

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
