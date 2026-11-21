// engine/src/inkling2/ink_batch.h - Multi-Request Continuous Batching (1..64 slots), Paged SWA/Global KV (FP16/FP8) & ShortConv Pool
// MIT License - Copyright (c) 2026 Strata contributors
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
    FP16 = 0,     // 32 KiB/tok Global (8 visits x 8 kv_heads x 128 dim x 2 x 2B), 160 MiB max SWA-1024
    FP8_E4M3 = 1, // 16 KiB/tok Global (64 MiB per 4K slot!), 80 MiB max SWA-1024 -> 64 slots fit in 16-32 GB GPU!
};

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
    const size_t elem = (kv_dtype == KvDtype::FP8_E4M3) ? 1u : 2u;
    const size_t kv_dim = 8u * 128u;  // 1024
    const size_t swa_len = static_cast<size_t>(std::min(max_ctx, 1024));
    SlotMemBudget b;
    b.swa_kv_bytes    = 40u * swa_len * 2u * kv_dim * elem;
    b.global_kv_bytes = 8u * static_cast<size_t>(max_ctx) * 2u * kv_dim * elem;
    b.sconv_bytes     = 48u * 3u * (1024u + 1024u + 2304u + 2304u) * 2u;  // 1,916,928 x 2 = 3,833,856 B (~3.83 MB)
    if (with_dspark) {
        b.dspark_bytes = (2u * std::min<size_t>(max_ctx, 1025u) + static_cast<size_t>(max_ctx)) * 2u * kv_dim * 2u
                       + static_cast<size_t>(max_ctx) * 2304u * 2u;
    }
    return b;
}

// ---- Paged KV Cache for a single slot (on-demand 64-token page allocation -> tiny footprint for active slots!) ----
struct SlotKvPages {
    KvDtype dtype = KvDtype::FP16;
    int kv_dim = 1024;
    int swa_win = 1024;
    int n_swa_visits = 40;
    int n_glob_visits = 8;
    int seq_len = 0;

    // Each page stores kKvPageTokens * kv_dim elements (uint16_t for FP16, uint8_t for FP8)
    // SWA pages: [n_swa_visits][max_swa_pages], ring-indexed by (pos % swa_win) / kKvPageTokens
    int max_swa_pages = 16;  // 1024 / 64 = 16
    std::vector<std::vector<uint8_t>> swa_k_pages;  // [n_swa_visits * max_swa_pages]
    std::vector<std::vector<uint8_t>> swa_v_pages;
    std::vector<std::vector<float>>   swa_k_scales; // per-token scale for FP8 [n_swa_visits * max_swa_pages][64]
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

    size_t elem_bytes() const noexcept { return (dtype == KvDtype::FP8_E4M3) ? 1u : 2u; }

    void write_vec(uint8_t* page, float* scales, int slot_in_page, const float* src) const noexcept {
        if (dtype == KvDtype::FP16) {
            uint16_t* dst = reinterpret_cast<uint16_t*>(page) + static_cast<size_t>(slot_in_page) * kv_dim;
            for (int i = 0; i < kv_dim; ++i) dst[i] = f32_to_fp16(src[i]);
        } else {
            float amax = 0.0f;
            for (int i = 0; i < kv_dim; ++i) amax = std::max(amax, std::fabs(src[i]));
            const float sc = (amax > 1e-12f) ? (amax / 448.0f) : 1.0f;
            const float inv = 1.0f / sc;
            scales[slot_in_page] = sc;
            uint8_t* dst = page + static_cast<size_t>(slot_in_page) * kv_dim;
            for (int i = 0; i < kv_dim; ++i) dst[i] = f32_to_fp8_e4m3(src[i] * inv);
        }
    }

    void read_vec(const uint8_t* page, const float* scales, int slot_in_page, float* dst) const noexcept {
        if (dtype == KvDtype::FP16) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(page) + static_cast<size_t>(slot_in_page) * kv_dim;
            for (int i = 0; i < kv_dim; ++i) dst[i] = fp16_to_f32(src[i]);
        } else {
            const float sc = scales[slot_in_page];
            const uint8_t* src = page + static_cast<size_t>(slot_in_page) * kv_dim;
            for (int i = 0; i < kv_dim; ++i) dst[i] = fp8_e4m3_to_f32(src[i]) * sc;
        }
    }

    void store_swa(int swa_slot, int pos, const float* k, const float* v) {
        const int rpos = pos % swa_win;
        const int p_idx = swa_slot * max_swa_pages + (rpos / kKvPageTokens);
        const int off = rpos % kKvPageTokens;
        const size_t pbytes = static_cast<size_t>(kKvPageTokens) * kv_dim * elem_bytes();
        if (swa_k_pages[p_idx].empty()) {
            swa_k_pages[p_idx].resize(pbytes);
            swa_v_pages[p_idx].resize(pbytes);
            if (dtype == KvDtype::FP8_E4M3) {
                swa_k_scales[p_idx].assign(kKvPageTokens, 1.0f);
                swa_v_scales[p_idx].assign(kKvPageTokens, 1.0f);
            }
        }
        write_vec(swa_k_pages[p_idx].data(), swa_k_scales[p_idx].data(), off, k);
        write_vec(swa_v_pages[p_idx].data(), swa_v_scales[p_idx].data(), off, v);
    }

    void load_swa(int swa_slot, int pos, float* k, float* v) const {
        const int rpos = pos % swa_win;
        const int p_idx = swa_slot * max_swa_pages + (rpos / kKvPageTokens);
        const int off = rpos % kKvPageTokens;
        read_vec(swa_k_pages[p_idx].data(), swa_k_scales[p_idx].data(), off, k);
        read_vec(swa_v_pages[p_idx].data(), swa_v_scales[p_idx].data(), off, v);
    }

    void store_global(int glob_slot, int pos, const float* k, const float* v) {
        const size_t pg = static_cast<size_t>(pos / kKvPageTokens);
        const int off = pos % kKvPageTokens;
        const size_t pbytes = static_cast<size_t>(kKvPageTokens) * kv_dim * elem_bytes();
        if (pg >= glob_k_pages[glob_slot].size()) {
            glob_k_pages[glob_slot].resize(pg + 1);
            glob_v_pages[glob_slot].resize(pg + 1);
            if (dtype == KvDtype::FP8_E4M3) {
                glob_k_scales[glob_slot].resize(pg + 1);
                glob_v_scales[glob_slot].resize(pg + 1);
            }
        }
        if (glob_k_pages[glob_slot][pg].empty()) {
            glob_k_pages[glob_slot][pg].resize(pbytes);
            glob_v_pages[glob_slot][pg].resize(pbytes);
            if (dtype == KvDtype::FP8_E4M3) {
                glob_k_scales[glob_slot][pg].assign(kKvPageTokens, 1.0f);
                glob_v_scales[glob_slot][pg].assign(kKvPageTokens, 1.0f);
            }
        }
        write_vec(glob_k_pages[glob_slot][pg].data(),
                  dtype == KvDtype::FP8_E4M3 ? glob_k_scales[glob_slot][pg].data() : nullptr, off, k);
        write_vec(glob_v_pages[glob_slot][pg].data(),
                  dtype == KvDtype::FP8_E4M3 ? glob_v_scales[glob_slot][pg].data() : nullptr, off, v);
    }

    void load_global(int glob_slot, int pos, float* k, float* v) const {
        const size_t pg = static_cast<size_t>(pos / kKvPageTokens);
        const int off = pos % kKvPageTokens;
        read_vec(glob_k_pages[glob_slot][pg].data(),
                 dtype == KvDtype::FP8_E4M3 ? glob_k_scales[glob_slot][pg].data() : nullptr, off, k);
        read_vec(glob_v_pages[glob_slot][pg].data(),
                 dtype == KvDtype::FP8_E4M3 ? glob_v_scales[glob_slot][pg].data() : nullptr, off, v);
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
