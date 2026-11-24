// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/ink_batch.h - Multi-Request Continuous Batching (1..64 slots), Paged SWA/Global KV (FP16/FP8) & ShortConv Pool
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ink2 {

constexpr int kMaxParallelSlots = 64;
constexpr int kKvPageTokens = 64;  // 64-token pages for on-demand SWA-1024 & Global KV allocation

enum class KvDtype : uint8_t {
    FP16     = 0, // 32 KiB/tok Global (8 visits x 8 kv_heads x 128 dim x 2 x 2B), 160 MiB max SWA-1024
    FP8_E4M3 = 1, // 16 KiB/tok Global (64 MiB per 4K slot!), 80 MiB max SWA-1024 -> 64 slots fit in 16-32 GB GPU!
    K8V8     = 2, // 16.5 KiB/tok Global: symmetric INT8 K & INT8 V with per-64 FP16 scales (#6520 / @merbanan)
    K8V4     = 3, // 12.75 KiB/tok Global: symmetric INT8 K + 4-bit asymmetric V with per-64 FP16 scale & min (#6520 / @merbanan)
};

inline const char* kv_dtype_name(KvDtype dt) noexcept {
    switch (dt) {
        case KvDtype::FP16:     return "fp16";
        case KvDtype::FP8_E4M3: return "fp8";
        case KvDtype::K8V8:     return "k8v8";
        case KvDtype::K8V4:     return "k8v4";
        default:                return "fp16";
    }
}

// ---- Scalar FP16 / BF16 / FP8 E4M3FN conversion helpers (portable across CPU & GPU host prep) ----
inline float bf16_to_f32(uint16_t b) noexcept {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) noexcept {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

inline float fp16_to_f32(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            uint32_t m = mant, e = 127 - 14;
            while ((m & 0x0400u) == 0) { m <<= 1; --e; }
            f = sign | (e << 23) | ((m & 0x03FFu) << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline uint16_t f32_to_fp16(float f) noexcept {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000u);
    int exp = static_cast<int>((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        return static_cast<uint16_t>(sign | ((mant + (1u << (shift - 1))) >> shift));
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
}

// IEEE-style FP8 E4M3FN (max finite 448.0, zero NaN emission on clamp)
inline float fp8_e4m3_to_f32(uint8_t c) noexcept {
    static const struct Lut {
        float v[256];
        Lut() {
            for (int i = 0; i < 256; ++i) {
                int exp = (i & 0x7F) >> 3;
                int man = i & 0x07;
                float val = 0.0f;
                if ((i & 0x7F) == 0x7F) val = 0.0f;
                else if (exp == 0) val = static_cast<float>(man) * (1.0f / 512.0f);
                else val = (1.0f + static_cast<float>(man) * 0.125f) * std::ldexp(1.0f, exp - 7);
                v[i] = (i & 0x80) ? -val : val;
            }
        }
    } lut;
    return lut.v[c];
}

inline uint8_t f32_to_fp8_e4m3(float x) noexcept {
    if (!(x == x)) return 0;
    uint8_t sign = x < 0.0f ? 0x80u : 0x00u;
    float ax = std::fabs(x);
    if (ax >= 448.0f) return static_cast<uint8_t>(sign | 0x7Eu);
    if (ax < 1.0f / 1024.0f) return sign;
    uint32_t u;
    std::memcpy(&u, &ax, 4);
    int exp = static_cast<int>((u >> 23) & 0xFFu) - 127;
    if (exp < -6) {
        int man = static_cast<int>(std::lround(ax * 512.0f));
        if (man > 7) return static_cast<uint8_t>(sign | 0x08u);
        return static_cast<uint8_t>(sign | (man & 7));
    }
    int e4 = exp + 7;
    int man = static_cast<int>(((u & 0x7FFFFFu) + (1u << 19)) >> 20);
    if (man >= 8) { man = 0; ++e4; }
    if (e4 > 15 || (e4 == 15 && man >= 7)) return static_cast<uint8_t>(sign | 0x7Eu);
    return static_cast<uint8_t>(sign | (e4 << 3) | (man & 7));
}

// ---- Per-Slot Memory Footprint Estimator (exact Inkling-2-Mini geometry) ----
struct SlotMemBudget {
    size_t swa_kv_bytes = 0;     // 40 SWA visits x min(max_ctx, 1024) x 2 (K+V) x 1024 x elem_bytes
    size_t global_kv_bytes = 0;  // 8 Global visits x max_ctx x 2 (K+V) x 1024 x elem_bytes
    size_t sconv_bytes = 0;      // 48 visits x 3 lags x (1024+1024+2304+2304) x 2B = 3.83 MB
    size_t dspark_bytes = 0;     // 2 SWA-1025 + 1 Global KV + f_rnn ring
    size_t total_bytes() const { return swa_kv_bytes + global_kv_bytes + sconv_bytes + dspark_bytes; }
};

inline SlotMemBudget estimate_slot_bytes(int max_ctx, KvDtype kv_dtype, bool with_dspark) {
    const size_t kv_dim = 8u * 128u;  // 1024
    const size_t swa_len = static_cast<size_t>(std::min(max_ctx, 1024));
    // Bytes per token per visit for K + V combined
    size_t kv_tok_bytes = 2u * kv_dim * 2u;  // FP16: 4096 B
    if (kv_dtype == KvDtype::FP8_E4M3) kv_tok_bytes = 2u * kv_dim;
    else if (kv_dtype == KvDtype::K8V8) kv_tok_bytes = 2u * kv_dim + 2u * (kv_dim / 64u) * 2u;
    else if (kv_dtype == KvDtype::K8V4) kv_tok_bytes = kv_dim + (kv_dim / 2u) + (kv_dim / 64u) * 2u + (kv_dim / 64u) * 4u;
    SlotMemBudget b;
    b.swa_kv_bytes    = 40u * swa_len * kv_tok_bytes;
    b.global_kv_bytes = 8u * static_cast<size_t>(max_ctx) * kv_tok_bytes;
    b.sconv_bytes     = 48u * 3u * (1024u + 1024u + 2304u + 2304u) * 2u;  // 1,916,928 x 2 = 3,833,856 B (~3.83 MB)
    if (with_dspark) {
        b.dspark_bytes = (2u * std::min<size_t>(max_ctx, 1025u) + static_cast<size_t>(max_ctx)) * 2u * kv_dim * 2u
                       + static_cast<size_t>(max_ctx) * 2304u * 2u;
    }
    return b;
}

// ---- Paged KV Cache for a single slot (on-demand 64-token page allocation -> tiny footprint for active slots!) ----
// Supports FP16, FP8_E4M3, K8V8 (per-64 INT8 K + INT8 V), and K8V4 (per-64 INT8 K + 4-bit asymmetric V)
struct SlotKvPages {
    static constexpr int kGroupSize = 64;
    KvDtype dtype = KvDtype::FP16;
    int kv_dim = 1024;
    int swa_win = 1024;
    int n_swa_visits = 40;
    int n_glob_visits = 8;
    int seq_len = 0;

    // Each page stores kKvPageTokens rows:
    //   FP16: kv_dim * 2 B for K, kv_dim * 2 B for V
    //   FP8_E4M3: kv_dim B for K, kv_dim B for V; 1 float scale/token
    //   K8V8: kv_dim B int8 K, kv_dim B int8 V; (kv_dim/64) float scales/token
    //   K8V4: kv_dim B int8 K, (kv_dim/2) B packed u4 V; (kv_dim/64) scales for K, 2*(kv_dim/64) (scale, min) for V
    int max_swa_pages = 16;  // 1024 / 64 = 16
    std::vector<std::vector<uint8_t>> swa_k_pages;  // [n_swa_visits * max_swa_pages]
    std::vector<std::vector<uint8_t>> swa_v_pages;
    std::vector<std::vector<float>>   swa_k_scales;
    std::vector<std::vector<float>>   swa_v_scales;

    // Global pages: [n_glob_visits][num_alloc_pages] grown on demand!
    std::vector<std::vector<std::vector<uint8_t>>> glob_k_pages;
    std::vector<std::vector<std::vector<uint8_t>>> glob_v_pages;
    std::vector<std::vector<std::vector<float>>>   glob_k_scales;
    std::vector<std::vector<std::vector<float>>>   glob_v_scales;

    void init(int n_swa, int n_glob, int kv_d, int swa_w, KvDtype dt) {
        dtype = dt;
        kv_dim = kv_d;
        swa_win = swa_w;
        n_swa_visits = n_swa;
        n_glob_visits = n_glob;
        max_swa_pages = (swa_w + kKvPageTokens - 1) / kKvPageTokens;
        swa_k_pages.assign(static_cast<size_t>(n_swa) * max_swa_pages, {});
        swa_v_pages.assign(static_cast<size_t>(n_swa) * max_swa_pages, {});
        swa_k_scales.assign(static_cast<size_t>(n_swa) * max_swa_pages, {});
        swa_v_scales.assign(static_cast<size_t>(n_swa) * max_swa_pages, {});
        glob_k_pages.assign(n_glob, {});
        glob_v_pages.assign(n_glob, {});
        glob_k_scales.assign(n_glob, {});
        glob_v_scales.assign(n_glob, {});
        seq_len = 0;
    }

    void reset() { seq_len = 0; }

    size_t k_row_bytes() const noexcept {
        return (dtype == KvDtype::FP16) ? static_cast<size_t>(kv_dim) * 2u : static_cast<size_t>(kv_dim);
    }
    size_t v_row_bytes() const noexcept {
        if (dtype == KvDtype::FP16) return static_cast<size_t>(kv_dim) * 2u;
        if (dtype == KvDtype::K8V4) return static_cast<size_t>(kv_dim) / 2u;
        return static_cast<size_t>(kv_dim);
    }
    size_t k_scales_per_tok() const noexcept {
        if (dtype == KvDtype::FP16) return 0u;
        if (dtype == KvDtype::FP8_E4M3) return 1u;
        return static_cast<size_t>((kv_dim + kGroupSize - 1) / kGroupSize);
    }
    size_t v_scales_per_tok() const noexcept {
        if (dtype == KvDtype::FP16) return 0u;
        if (dtype == KvDtype::FP8_E4M3) return 1u;
        if (dtype == KvDtype::K8V4) return 2u * static_cast<size_t>((kv_dim + kGroupSize - 1) / kGroupSize);
        return static_cast<size_t>((kv_dim + kGroupSize - 1) / kGroupSize);
    }

    void write_k_vec(uint8_t* page, float* scales, int slot_in_page, const float* src) const noexcept {
        if (dtype == KvDtype::FP16) {
            uint16_t* dst = reinterpret_cast<uint16_t*>(page) + static_cast<size_t>(slot_in_page) * kv_dim;
            for (int i = 0; i < kv_dim; ++i) dst[i] = f32_to_fp16(src[i]);
        } else if (dtype == KvDtype::FP8_E4M3) {
            float amax = 0.0f;
            for (int i = 0; i < kv_dim; ++i) amax = std::max(amax, std::fabs(src[i]));
            const float sc = (amax > 1e-12f) ? (amax / 448.0f) : 1.0f;
            const float inv = 1.0f / sc;
            scales[slot_in_page] = sc;
            uint8_t* dst = page + static_cast<size_t>(slot_in_page) * kv_dim;
            for (int i = 0; i < kv_dim; ++i) dst[i] = f32_to_fp8_e4m3(src[i] * inv);
        } else {
            // K8V8 & K8V4 both store K as symmetric per-64 INT8 with FP16-rounded scale
            const int ng = (kv_dim + kGroupSize - 1) / kGroupSize;
            float* sc_row = scales + static_cast<size_t>(slot_in_page) * ng;
            int8_t* dst = reinterpret_cast<int8_t*>(page) + static_cast<size_t>(slot_in_page) * kv_dim;
            for (int g = 0; g < ng; ++g) {
                const int i0 = g * kGroupSize;
                const int i1 = std::min(kv_dim, i0 + kGroupSize);
                float amax = 0.0f;
                for (int i = i0; i < i1; ++i) amax = std::max(amax, std::fabs(src[i]));
                const float sc = fp16_to_f32(f32_to_fp16(amax / 127.0f));
                sc_row[g] = sc;
                const float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
                for (int i = i0; i < i1; ++i) {
                    int q = static_cast<int>(std::nearbyintf(src[i] * inv));
                    dst[i] = static_cast<int8_t>(std::clamp(q, -127, 127));
                }
            }
        }
    }

    void write_v_vec(uint8_t* page, float* scales, int slot_in_page, const float* src) const noexcept {
        if (dtype != KvDtype::K8V4) {
            write_k_vec(page, scales, slot_in_page, src);
            return;
        }
        // K8V4 V path: per-64 asymmetric 4-bit (0..15) with FP16-rounded (scale, min)
        const int ng = (kv_dim + kGroupSize - 1) / kGroupSize;
        float* sc_row = scales + static_cast<size_t>(slot_in_page) * (2u * ng);
        uint8_t* dst = page + static_cast<size_t>(slot_in_page) * (kv_dim / 2);
        for (int g = 0; g < ng; ++g) {
            const int i0 = g * kGroupSize;
            const int i1 = std::min(kv_dim, i0 + kGroupSize);
            float vmin = src[i0], vmax = src[i0];
            for (int i = i0 + 1; i < i1; ++i) {
                vmin = std::min(vmin, src[i]);
                vmax = std::max(vmax, src[i]);
            }
            const float sc = fp16_to_f32(f32_to_fp16((vmax - vmin) / 15.0f));
            const float mn = fp16_to_f32(f32_to_fp16(vmin));
            sc_row[2 * g + 0] = sc;
            sc_row[2 * g + 1] = mn;
            const float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
            for (int i = i0; i < i1; i += 2) {
                int q0 = std::clamp(static_cast<int>(std::nearbyintf((src[i] - mn) * inv)), 0, 15);
                int q1 = (i + 1 < i1) ? std::clamp(static_cast<int>(std::nearbyintf((src[i + 1] - mn) * inv)), 0, 15) : 0;
                dst[i / 2] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            }
        }
    }

    void read_k_vec(const uint8_t* page, const float* scales, int slot_in_page, float* dst) const noexcept {
        if (dtype == KvDtype::FP16) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(page) + static_cast<size_t>(slot_in_page) * kv_dim;
            for (int i = 0; i < kv_dim; ++i) dst[i] = fp16_to_f32(src[i]);
        } else if (dtype == KvDtype::FP8_E4M3) {
            const float sc = scales[slot_in_page];
            const uint8_t* src = page + static_cast<size_t>(slot_in_page) * kv_dim;
            for (int i = 0; i < kv_dim; ++i) dst[i] = fp8_e4m3_to_f32(src[i]) * sc;
        } else {
            const int ng = (kv_dim + kGroupSize - 1) / kGroupSize;
            const float* sc_row = scales + static_cast<size_t>(slot_in_page) * ng;
            const int8_t* src = reinterpret_cast<const int8_t*>(page) + static_cast<size_t>(slot_in_page) * kv_dim;
            for (int g = 0; g < ng; ++g) {
                const int i0 = g * kGroupSize;
                const int i1 = std::min(kv_dim, i0 + kGroupSize);
                const float sc = sc_row[g];
                for (int i = i0; i < i1; ++i) dst[i] = static_cast<float>(src[i]) * sc;
            }
        }
    }

    void read_v_vec(const uint8_t* page, const float* scales, int slot_in_page, float* dst) const noexcept {
        if (dtype != KvDtype::K8V4) {
            read_k_vec(page, scales, slot_in_page, dst);
            return;
        }
        const int ng = (kv_dim + kGroupSize - 1) / kGroupSize;
        const float* sc_row = scales + static_cast<size_t>(slot_in_page) * (2u * ng);
        const uint8_t* src = page + static_cast<size_t>(slot_in_page) * (kv_dim / 2);
        for (int g = 0; g < ng; ++g) {
            const int i0 = g * kGroupSize;
            const int i1 = std::min(kv_dim, i0 + kGroupSize);
            const float sc = sc_row[2 * g + 0];
            const float mn = sc_row[2 * g + 1];
            for (int i = i0; i < i1; i += 2) {
                const uint8_t b = src[i / 2];
                dst[i] = static_cast<float>(b & 0x0Fu) * sc + mn;
                if (i + 1 < i1) dst[i + 1] = static_cast<float>((b >> 4) & 0x0Fu) * sc + mn;
            }
        }
    }

    void store_swa(int swa_slot, int pos, const float* k, const float* v) {
        const int rpos = pos % swa_win;
        const int p_idx = swa_slot * max_swa_pages + (rpos / kKvPageTokens);
        const int off = rpos % kKvPageTokens;
        if (swa_k_pages[p_idx].empty()) {
            swa_k_pages[p_idx].resize(static_cast<size_t>(kKvPageTokens) * k_row_bytes());
            swa_v_pages[p_idx].resize(static_cast<size_t>(kKvPageTokens) * v_row_bytes());
            if (k_scales_per_tok() > 0) swa_k_scales[p_idx].assign(static_cast<size_t>(kKvPageTokens) * k_scales_per_tok(), 1.0f);
            if (v_scales_per_tok() > 0) swa_v_scales[p_idx].assign(static_cast<size_t>(kKvPageTokens) * v_scales_per_tok(), 1.0f);
        }
        write_k_vec(swa_k_pages[p_idx].data(), swa_k_scales[p_idx].empty() ? nullptr : swa_k_scales[p_idx].data(), off, k);
        write_v_vec(swa_v_pages[p_idx].data(), swa_v_scales[p_idx].empty() ? nullptr : swa_v_scales[p_idx].data(), off, v);
    }

    void load_swa(int swa_slot, int pos, float* k, float* v) const {
        const int rpos = pos % swa_win;
        const int p_idx = swa_slot * max_swa_pages + (rpos / kKvPageTokens);
        const int off = rpos % kKvPageTokens;
        read_k_vec(swa_k_pages[p_idx].data(), swa_k_scales[p_idx].empty() ? nullptr : swa_k_scales[p_idx].data(), off, k);
        read_v_vec(swa_v_pages[p_idx].data(), swa_v_scales[p_idx].empty() ? nullptr : swa_v_scales[p_idx].data(), off, v);
    }

    void store_global(int glob_slot, int pos, const float* k, const float* v) {
        const size_t pg = static_cast<size_t>(pos / kKvPageTokens);
        const int off = pos % kKvPageTokens;
        if (pg >= glob_k_pages[glob_slot].size()) {
            glob_k_pages[glob_slot].resize(pg + 1);
            glob_v_pages[glob_slot].resize(pg + 1);
            if (k_scales_per_tok() > 0) glob_k_scales[glob_slot].resize(pg + 1);
            if (v_scales_per_tok() > 0) glob_v_scales[glob_slot].resize(pg + 1);
        }
        if (glob_k_pages[glob_slot][pg].empty()) {
            glob_k_pages[glob_slot][pg].resize(static_cast<size_t>(kKvPageTokens) * k_row_bytes());
            glob_v_pages[glob_slot][pg].resize(static_cast<size_t>(kKvPageTokens) * v_row_bytes());
            if (k_scales_per_tok() > 0) glob_k_scales[glob_slot][pg].assign(static_cast<size_t>(kKvPageTokens) * k_scales_per_tok(), 1.0f);
            if (v_scales_per_tok() > 0) glob_v_scales[glob_slot][pg].assign(static_cast<size_t>(kKvPageTokens) * v_scales_per_tok(), 1.0f);
        }
        write_k_vec(glob_k_pages[glob_slot][pg].data(),
                    k_scales_per_tok() > 0 ? glob_k_scales[glob_slot][pg].data() : nullptr, off, k);
        write_v_vec(glob_v_pages[glob_slot][pg].data(),
                    v_scales_per_tok() > 0 ? glob_v_scales[glob_slot][pg].data() : nullptr, off, v);
    }

    void load_global(int glob_slot, int pos, float* k, float* v) const {
        const size_t pg = static_cast<size_t>(pos / kKvPageTokens);
        const int off = pos % kKvPageTokens;
        read_k_vec(glob_k_pages[glob_slot][pg].data(),
                   k_scales_per_tok() > 0 ? glob_k_scales[glob_slot][pg].data() : nullptr, off, k);
        read_v_vec(glob_v_pages[glob_slot][pg].data(),
                   v_scales_per_tok() > 0 ? glob_v_scales[glob_slot][pg].data() : nullptr, off, v);
    }
};

// ---- Compact 48-Visit 3-Lag ShortConv State Pool (3.83 MB/slot in BF16 -> 245 MB for all 64 slots!) ----
struct SlotShortConvState {
    int n_visits = 48;
    int kv_dim = 1024;
    int hidden = 2304;
    // Per visit: k[3, kv_dim], v[3, kv_dim], attn[3, hidden], mlp[3, hidden] stored in BF16 (compact!)
    // Snapshot ring for speculative rollback (up to 16 tokens)
    std::vector<uint16_t> k_hist;     // [n_visits * 3 * kv_dim]
    std::vector<uint16_t> v_hist;     // [n_visits * 3 * kv_dim]
    std::vector<uint16_t> attn_hist;  // [n_visits * 3 * hidden]
    std::vector<uint16_t> mlp_hist;   // [n_visits * 3 * hidden]

    void init(int nv, int kvd, int hid) {
        n_visits = nv;
        kv_dim = kvd;
        hidden = hid;
        k_hist.assign(static_cast<size_t>(nv) * 3 * kvd, 0);
        v_hist.assign(static_cast<size_t>(nv) * 3 * kvd, 0);
        attn_hist.assign(static_cast<size_t>(nv) * 3 * hid, 0);
        mlp_hist.assign(static_cast<size_t>(nv) * 3 * hid, 0);
    }

    void reset() {
        std::fill(k_hist.begin(), k_hist.end(), uint16_t(0));
        std::fill(v_hist.begin(), v_hist.end(), uint16_t(0));
        std::fill(attn_hist.begin(), attn_hist.end(), uint16_t(0));
        std::fill(mlp_hist.begin(), mlp_hist.end(), uint16_t(0));
    }

    size_t bytes() const noexcept {
        return (k_hist.size() + v_hist.size() + attn_hist.size() + mlp_hist.size()) * sizeof(uint16_t);
    }

    // Read 3 lags [3, C] into fp32 scratch, run step, write back BF16
    static void load_3lag(const uint16_t* src, float* dst, int C) noexcept {
        for (int i = 0; i < 3 * C; ++i) dst[i] = bf16_to_f32(src[i]);
    }
    static void store_3lag(const float* src, uint16_t* dst, int C) noexcept {
        for (int i = 0; i < 3 * C; ++i) dst[i] = f32_to_bf16(src[i]);
    }
};

// ---- Batched Step Descriptor (coalesces 1..64 active requests + chunked prefill + DSpark verify into one 48-visit pass!) ----
struct StepBatch {
    int n_tokens = 0;
    std::vector<int32_t> slot_ids;    // [n_tokens] slot index in [0..63]
    std::vector<int32_t> pos;         // [n_tokens] sequence position in slot
    std::vector<int32_t> token_ids;   // [n_tokens] input token id (ignored if img_embed_idx[i] >= 0)
    std::vector<int32_t> img_embed_idx; // [n_tokens] -1 for text token, else row index in img_embeds
    const float* img_embeds = nullptr;  // [n_img_rows, hidden] (bypasses embed_norm!)
    std::vector<uint8_t> want_logits; // [n_tokens] 1 if logits row should be written to logits_out
    bool capture_taps = false;        // 1 to capture DSpark taps [5, 23, 41, 48] into slot state

    void clear() {
        n_tokens = 0;
        slot_ids.clear();
        pos.clear();
        token_ids.clear();
        img_embed_idx.clear();
        want_logits.clear();
        img_embeds = nullptr;
        capture_taps = false;
    }

    void push(int32_t slot, int32_t position, int32_t tok, bool logits, int32_t img_idx = -1) {
        slot_ids.push_back(slot);
        pos.push_back(position);
        token_ids.push_back(tok);
        img_embed_idx.push_back(img_idx);
        want_logits.push_back(logits ? 1 : 0);
        ++n_tokens;
    }

    int count_logits() const noexcept {
        int n = 0;
        for (uint8_t w : want_logits) n += (w != 0);
        return n;
    }
};

struct TapCapture {
    // Optional host capture of [4, n_tokens, hidden] for visits [5, 23, 41, 48]
    std::vector<float> taps[4];
};

}  // namespace ink2
