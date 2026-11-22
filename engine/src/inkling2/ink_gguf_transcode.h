// engine/src/inkling2/ink_gguf_transcode.h - Streaming multi-threaded GGUF_RAW -> S_K (S4_K/S6_K/S8_K)
// expert transcoder & InkModelFile loader helper shared across Metal, Intel Arc SYCL, and AMD HIP.
// Zero temp files on disk, < 64 MiB transient scratch, 100% MIT clean-room implementation.
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ink_manifest.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/quant/sk_codecs.hpp"

namespace ink2 {
namespace transcode {

inline uint16_t f32_to_fp16_bits(float f) noexcept {
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp  = (x >> 23) & 0xFFu;
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp == 0xFFu) {
        // Clamp Inf/NaN to max finite FP16 magnitude (or 0 for NaN)
        if (mant != 0) return static_cast<uint16_t>(sign);
        return static_cast<uint16_t>(sign | 0x7BFFu);
    }
    const int e = static_cast<int>(exp) - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7BFFu);
    if (e <= 0) {
        if (e < -10) return static_cast<uint16_t>(sign);
        const uint32_t m = mant | 0x800000u;
        const int shift = 14 - e;
        const uint32_t round_bit = 1u << (shift - 1);
        const uint32_t rounded = (m + round_bit - 1u + ((m >> shift) & 1u)) >> shift;
        return static_cast<uint16_t>(sign | rounded);
    }
    const uint32_t rounded = mant + 0x0FFFu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) {
        const int e2 = e + 1;
        if (e2 >= 31) return static_cast<uint16_t>(sign | 0x7BFFu);
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e2) << 10));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (rounded >> 13));
}

inline float fp16_bits_to_f32(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t e = 127u - 14u;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            f = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &f, 4);
    return out;
}

// Choose target S_K codec preserving source GGUF precision without blowing VRAM budget.
// Optional override via INK_GGUF_SK_BITS=4|6|8.
inline strata::quant::sk::SkCode choose_sk_code_for_ggml(uint32_t ggml_type) noexcept {
    using strata::quant::sk::SkCode;
    if (const char* ev = std::getenv("INK_GGUF_SK_BITS")) {
        const int b = std::atoi(ev);
        if (b == 4) return SkCode::S4_K;
        if (b == 6) return SkCode::S6_K;
        if (b == 8) return SkCode::S8_K;
    }
    switch (ggml_type) {
        // >= 8 bpw (Q8_0, Q8_1, Q8_K, F16, BF16, F32) -> near-lossless S8_K
        case 0:   // F32
        case 1:   // F16
        case 8:   // Q8_0
        case 9:   // Q8_1
        case 15:  // Q8_K
        case 30:  // BF16
            return SkCode::S8_K;
        // 5..6 bpw (Q5_0, Q5_1, Q5_K, Q6_K) -> S6_K
        case 6:   // Q5_0
        case 7:   // Q5_1
        case 13:  // Q5_K
        case 14:  // Q6_K
            return SkCode::S6_K;
        // <= 4 bpw (Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, IQ1_*, IQ2_*, IQ3_*, IQ4_*, MXFP4) -> S4_K
        default:
            return SkCode::S4_K;
    }
}

// Compute exact wire layout of an S_K Inkling-2-Mini expert body:
//   [0 .. 2*inter - 1]                  : sq_xd_dn[inter] (FP16 1.0h = 0x3C00)
//   [2*inter .. 2*inter + w13_bytes -1] : w13 [2*inter, hidden] interleaved (even=gate, odd=up)
//   [dn_offset .. dn_offset + w2_bytes) : w2  [hidden, inter] paired 2-row (SB_0..SB_8 per pair)
inline void transcoded_sk_expert_layout(
    strata::quant::sk::SkCode gu_code,
    strata::quant::sk::SkCode dn_code,
    size_t hidden,
    size_t inter,
    uint32_t& out_dn_off,
    uint32_t& out_body_bytes) noexcept {
    const size_t sb_gu = strata::quant::sk::superblock_bytes(gu_code);
    const size_t sb_dn = strata::quant::sk::superblock_bytes(dn_code);
    const size_t sq_bytes  = inter * 2u;
    const size_t w13_bytes = (2u * inter) * (hidden / 256u) * sb_gu;
    const size_t dn_off    = ((sq_bytes + w13_bytes) + 63u) & ~size_t(63u);
    const size_t w2_bytes  = (hidden / 2u) * ((2u * inter) / 256u) * sb_dn;
    out_dn_off     = static_cast<uint32_t>(dn_off);
    out_body_bytes = static_cast<uint32_t>(dn_off + w2_bytes);
}

// Encode one 256-weight super-block (4 x 64-weight sub-groups) into S4_K, S6_K, or S8_K wire bytes.
// Matches strata::quant::sk::dequant_sk_row() and all Metal/SYCL/HIP S_K GPU kernels bit-for-bit.
inline void encode_sk_superblock(const float* w, strata::quant::sk::SkCode code, uint8_t* dst) noexcept {
    using strata::quant::sk::SkCode;
    const size_t sb_bytes = strata::quant::sk::superblock_bytes(code);
    float max_q = 7.5f;
    if (code == SkCode::S6_K) max_q = 31.5f;
    else if (code == SkCode::S8_K) max_q = 127.0f;

    float s[4]{};
    float d_max = 0.0f;
    for (int g = 0; g < 4; ++g) {
        const float* wg = w + g * 64;
        float amax = 0.0f;
        for (int j = 0; j < 64; ++j) {
            const float a = std::fabs(wg[j]);
            if (a > amax) amax = a;
        }
        s[g] = amax / max_q;
        if (s[g] > d_max) d_max = s[g];
    }

    const uint16_t d_bits = f32_to_fp16_bits(d_max);
    const float D = fp16_bits_to_f32(d_bits);
    if (!(D > 0.0f)) {
        std::memset(dst, 0, sb_bytes);
        return;
    }

    std::memcpy(dst, &d_bits, 2);
    uint8_t* u = dst + 2;
    uint8_t* pl = dst + 6;
    const float inv_D_255 = 255.0f / D;

    if (code == SkCode::S4_K) {
        for (int g = 0; g < 4; ++g) {
            int ug = static_cast<int>(std::lround(s[g] * inv_D_255));
            if (s[g] > 0.0f && ug < 1) ug = 1;
            if (ug > 255) ug = 255;
            u[g] = static_cast<uint8_t>(ug);
            const float eff = D * (static_cast<float>(ug) * (1.0f / 255.0f));
            const float inv = (eff > 0.0f) ? (1.0f / eff) : 0.0f;
            const float* wg = w + g * 64;
            uint8_t* pg = pl + g * 32;
            for (int i = 0; i < 32; ++i) {
                int q0 = static_cast<int>(std::lround(wg[2 * i + 0] * inv + 7.5f));
                int q1 = static_cast<int>(std::lround(wg[2 * i + 1] * inv + 7.5f));
                q0 = std::max(0, std::min(15, q0));
                q1 = std::max(0, std::min(15, q1));
                pg[i] = static_cast<uint8_t>(q0 | (q1 << 4));
            }
        }
    } else if (code == SkCode::S6_K) {
        uint8_t qbuf[256];
        for (int g = 0; g < 4; ++g) {
            int ug = static_cast<int>(std::lround(s[g] * inv_D_255));
            if (s[g] > 0.0f && ug < 1) ug = 1;
            if (ug > 255) ug = 255;
            u[g] = static_cast<uint8_t>(ug);
            const float eff = D * (static_cast<float>(ug) * (1.0f / 255.0f));
            const float inv = (eff > 0.0f) ? (1.0f / eff) : 0.0f;
            const float* wg = w + g * 64;
            uint8_t* qg = qbuf + g * 64;
            for (int j = 0; j < 64; ++j) {
                int q = static_cast<int>(std::lround(wg[j] * inv + 31.5f));
                qg[j] = static_cast<uint8_t>(std::max(0, std::min(63, q)));
            }
        }
        uint8_t* p_lo = pl;
        uint8_t* p_hi = pl + 128;
        for (int i = 0; i < 128; ++i) {
            p_lo[i] = static_cast<uint8_t>((qbuf[2 * i + 0] & 0x0Fu) | ((qbuf[2 * i + 1] & 0x0Fu) << 4));
        }
        for (int i = 0; i < 64; ++i) {
            p_hi[i] = static_cast<uint8_t>(
                ((qbuf[4 * i + 0] >> 4) & 0x03u) |
                (((qbuf[4 * i + 1] >> 4) & 0x03u) << 2) |
                (((qbuf[4 * i + 2] >> 4) & 0x03u) << 4) |
                (((qbuf[4 * i + 3] >> 4) & 0x03u) << 6));
        }
    } else {
        // S8_K
        int8_t* qdst = reinterpret_cast<int8_t*>(pl);
        for (int g = 0; g < 4; ++g) {
            int ug = static_cast<int>(std::lround(s[g] * inv_D_255));
            if (s[g] > 0.0f && ug < 1) ug = 1;
            if (ug > 255) ug = 255;
            u[g] = static_cast<uint8_t>(ug);
            const float eff = D * (static_cast<float>(ug) * (1.0f / 255.0f));
            const float inv = (eff > 0.0f) ? (1.0f / eff) : 0.0f;
            const float* wg = w + g * 64;
            int8_t* qg = qdst + g * 64;
            for (int j = 0; j < 64; ++j) {
                int q = static_cast<int>(std::lround(wg[j] * inv));
                qg[j] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
            }
        }
    }
}

inline void encode_sk_row(const float* row, size_t n, strata::quant::sk::SkCode code, uint8_t* dst) noexcept {
    const size_t sb_bytes = strata::quant::sk::superblock_bytes(code);
    const size_t n_sb = n / 256u;
    for (size_t s = 0; s < n_sb; ++s) {
        encode_sk_superblock(row + s * 256u, code, dst + s * sb_bytes);
    }
}

// Transcode one SkCode::GGUF_RAW expert body into canonical .strata v1.1 S_K expert layout
// (sq_xd_dn=1.0h + interleaved w13 + paired-2-row w2). Caller supplies dst buffer sized via
// transcoded_sk_expert_layout() (or use vector overload below).
inline bool transcode_gguf_raw_expert_into(
    const uint8_t* raw_body,
    size_t raw_bytes,
    uint32_t raw_dn_offset,
    size_t hidden,
    size_t inter,
    uint8_t* dst,
    size_t dst_bytes,
    uint32_t& out_dn_off,
    uint8_t& out_gu_code,
    uint8_t& out_dn_code,
    std::string& err) {
    using strata::quant::sk::GgufExpertHeader;
    using strata::quant::sk::SkCode;
    if (raw_bytes < sizeof(GgufExpertHeader)) {
        err = "GGUF_RAW expert truncated (< 32 B header)";
        return false;
    }
    GgufExpertHeader hdr{};
    std::memcpy(&hdr, raw_body, sizeof(hdr));
    if (hdr.magic != 0x47475546u) {
        err = "GGUF_RAW expert bad header magic";
        return false;
    }
    if ((hidden % 256u) != 0 || ((2u * inter) % 256u) != 0 || (hidden % 2u) != 0) {
        err = "GGUF_RAW transcode requires hidden%256==0 and (2*inter)%256==0";
        return false;
    }

    const SkCode gu_code = choose_sk_code_for_ggml(hdr.gu_type);
    const SkCode dn_code = choose_sk_code_for_ggml(hdr.dn_type);
    uint32_t need_dn_off = 0, need_bytes = 0;
    transcoded_sk_expert_layout(gu_code, dn_code, hidden, inter, need_dn_off, need_bytes);
    if (dst_bytes < need_bytes) {
        err = "GGUF_RAW transcode dst buffer too small";
        return false;
    }

    size_t gu_row_b = gguf_detail::gguf_row_bytes(hdr.gu_type, static_cast<int64_t>(hidden));
    if (gu_row_b == 0) gu_row_b = strata::quant::row_bytes(static_cast<int>(hdr.gu_type), static_cast<int64_t>(hidden));
    size_t dn_row_b = gguf_detail::gguf_row_bytes(hdr.dn_type, static_cast<int64_t>(inter));
    if (dn_row_b == 0) dn_row_b = strata::quant::row_bytes(static_cast<int>(hdr.dn_type), static_cast<int64_t>(inter));
    if (gu_row_b == 0 || dn_row_b == 0) {
        err = "GGUF_RAW unsupported ggml_type in expert (gu=" + std::to_string(hdr.gu_type) +
              ", dn=" + std::to_string(hdr.dn_type) + ")";
        return false;
    }
    if (sizeof(GgufExpertHeader) + (2u * inter) * gu_row_b > raw_bytes ||
        static_cast<size_t>(raw_dn_offset) + hidden * dn_row_b > raw_bytes) {
        err = "GGUF_RAW expert body out of bounds";
        return false;
    }

    // 1. Write unit SmoothQuant header sq_xd_dn[inter] = 1.0h (0x3C00)
    uint16_t* sq_dst = reinterpret_cast<uint16_t*>(dst);
    for (size_t j = 0; j < inter; ++j) sq_dst[j] = 0x3C00u;

    // 2. Transcode w13 -> interleaved (even=gate, odd=up) S_K rows
    const uint8_t* w13_src = raw_body + sizeof(GgufExpertHeader);
    uint8_t* w13_dst = dst + inter * 2u;
    const size_t w13_row_out_b = (hidden / 256u) * strata::quant::sk::superblock_bytes(gu_code);
    const bool interleaved = (hdr.flags & 1u) != 0;

    std::vector<float> row_f32(hidden);
    auto scale_vec = [](float* v, size_t n, float sc) {
        if (sc != 1.0f) {
            for (size_t i = 0; i < n; ++i) v[i] *= sc;
        }
    };

    if (interleaved) {
        for (size_t r = 0; r < 2u * inter; ++r) {
            if (!gguf_detail::gguf_dequant_row_f32(hdr.gu_type, w13_src + r * gu_row_b, row_f32.data(), static_cast<int64_t>(hidden))) {
                err = "GGUF_RAW gate/up dequant failed";
                return false;
            }
            scale_vec(row_f32.data(), hidden, ((r & 1u) == 0) ? hdr.gate_scale : hdr.up_scale);
            encode_sk_row(row_f32.data(), hidden, gu_code, w13_dst + r * w13_row_out_b);
        }
    } else {
        for (size_t j = 0; j < inter; ++j) {
            if (!gguf_detail::gguf_dequant_row_f32(hdr.gu_type, w13_src + j * gu_row_b, row_f32.data(), static_cast<int64_t>(hidden))) {
                err = "GGUF_RAW gate dequant failed";
                return false;
            }
            scale_vec(row_f32.data(), hidden, hdr.gate_scale);
            encode_sk_row(row_f32.data(), hidden, gu_code, w13_dst + (2u * j + 0u) * w13_row_out_b);

            if (!gguf_detail::gguf_dequant_row_f32(hdr.gu_type, w13_src + (inter + j) * gu_row_b, row_f32.data(), static_cast<int64_t>(hidden))) {
                err = "GGUF_RAW up dequant failed";
                return false;
            }
            scale_vec(row_f32.data(), hidden, hdr.up_scale);
            encode_sk_row(row_f32.data(), hidden, gu_code, w13_dst + (2u * j + 1u) * w13_row_out_b);
        }
    }

    // Zero any alignment padding between w13 and w2
    const size_t w13_end = inter * 2u + (2u * inter) * w13_row_out_b;
    if (need_dn_off > w13_end) {
        std::memset(dst + w13_end, 0, need_dn_off - w13_end);
    }

    // 3. Transcode w2 -> paired 2-row S_K (each pair concatenates row 2p and row 2p+1 = 2*inter weights)
    const uint8_t* w2_src = raw_body + raw_dn_offset;
    uint8_t* w2_dst = dst + need_dn_off;
    const size_t pair_out_b = ((2u * inter) / 256u) * strata::quant::sk::superblock_bytes(dn_code);
    std::vector<float> pair_f32(2u * inter);
    for (size_t p = 0; p < hidden / 2u; ++p) {
        if (!gguf_detail::gguf_dequant_row_f32(hdr.dn_type, w2_src + (2u * p + 0u) * dn_row_b, pair_f32.data(), static_cast<int64_t>(inter)) ||
            !gguf_detail::gguf_dequant_row_f32(hdr.dn_type, w2_src + (2u * p + 1u) * dn_row_b, pair_f32.data() + inter, static_cast<int64_t>(inter))) {
            err = "GGUF_RAW down dequant failed";
            return false;
        }
        scale_vec(pair_f32.data(), 2u * inter, hdr.dn_scale);
        encode_sk_row(pair_f32.data(), 2u * inter, dn_code, w2_dst + p * pair_out_b);
    }

    out_dn_off  = need_dn_off;
    out_gu_code = static_cast<uint8_t>(gu_code);
    out_dn_code = static_cast<uint8_t>(dn_code);
    return true;
}

// Build post-transcode StrataExpertEntry directory so backends know exact VRAM/slab sizes upfront
// before reading or transcoding a single expert!
inline void plan_transcoded_expert_directory(
    const InkModelFile& model,
    std::vector<strata::artifact::StrataExpertEntry>& out_dir) {
    using strata::quant::sk::SkCode;
    out_dir = model.expert_directory();
    if (model.kind != ModelSourceKind::Gguf || !model.gguf_adapter) return;
    const size_t hidden = static_cast<size_t>(model.m.c.hidden);
    const size_t inter  = static_cast<size_t>(model.m.c.inter);
    const int n_experts = model.m.c.n_experts;
    for (size_t idx = 0; idx < out_dir.size(); ++idx) {
        auto& e = out_dir[idx];
        if (e.gu_code_id != static_cast<uint8_t>(SkCode::GGUF_RAW) &&
            e.dn_code_id != static_cast<uint8_t>(SkCode::GGUF_RAW)) {
            continue;
        }
        const int il = static_cast<int>(idx) / n_experts;
        if (il < 0 || il >= static_cast<int>(model.gguf_adapter->layers.size())) continue;
        const auto& lm = model.gguf_adapter->layers[il];
        const SkCode gu_c = choose_sk_code_for_ggml(lm.gu_type);
        const SkCode dn_c = choose_sk_code_for_ggml(lm.dn_type);
        uint32_t dn_off = 0, body_b = 0;
        transcoded_sk_expert_layout(gu_c, dn_c, hidden, inter, dn_off, body_b);
        e.gu_code_id = static_cast<uint8_t>(gu_c);
        e.dn_code_id = static_cast<uint8_t>(dn_c);
        e.dn_offset  = dn_off;
        e.body_bytes = body_b;
    }
}

// Read one expert body from InkModelFile (zero-copy passthrough for .strata / SF4 / MX8,
// streaming transcode to S4_K/S6_K/S8_K for GGUF_RAW). Thread-safe for GgufAdapter!
inline bool read_or_transcode_expert(
    const InkModelFile& model,
    size_t dir_idx,
    std::vector<uint8_t>& out_body,
    uint32_t& out_dn_off,
    uint8_t& out_gu_code,
    uint8_t& out_dn_code,
    std::string& err) {
    using strata::quant::sk::SkCode;
    const auto& raw_dir = model.expert_directory();
    if (dir_idx >= raw_dir.size()) {
        err = "expert dir_idx out of range";
        return false;
    }
    const auto& raw_ent = raw_dir[dir_idx];
    if (raw_ent.gu_code_id != static_cast<uint8_t>(SkCode::GGUF_RAW) &&
        raw_ent.dn_code_id != static_cast<uint8_t>(SkCode::GGUF_RAW)) {
        if (!model.read_expert_body(dir_idx, out_body, err)) return false;
        out_dn_off  = raw_ent.dn_offset;
        out_gu_code = raw_ent.gu_code_id;
        out_dn_code = raw_ent.dn_code_id;
        return true;
    }

    std::vector<uint8_t> raw;
    if (!model.read_expert_body(dir_idx, raw, err)) return false;
    if (raw.size() < sizeof(strata::quant::sk::GgufExpertHeader)) {
        err = "GGUF_RAW expert too short";
        return false;
    }
    strata::quant::sk::GgufExpertHeader hdr{};
    std::memcpy(&hdr, raw.data(), sizeof(hdr));
    const SkCode gu_c = choose_sk_code_for_ggml(hdr.gu_type);
    const SkCode dn_c = choose_sk_code_for_ggml(hdr.dn_type);
    uint32_t need_dn = 0, need_b = 0;
    transcoded_sk_expert_layout(
        gu_c, dn_c, static_cast<size_t>(model.m.c.hidden), static_cast<size_t>(model.m.c.inter),
        need_dn, need_b);
    out_body.resize(need_b);
    return transcode_gguf_raw_expert_into(
        raw.data(), raw.size(), raw_ent.dn_offset,
        static_cast<size_t>(model.m.c.hidden), static_cast<size_t>(model.m.c.inter),
        out_body.data(), out_body.size(),
        out_dn_off, out_gu_code, out_dn_code, err);
}

inline int default_transcode_threads() noexcept {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    return static_cast<int>(std::min<unsigned>(16u, std::max<unsigned>(1u, hw)));
}

// Run fn(item_idx) across [0..count) using up to n_threads workers; stops early on first error.
template <typename Fn>
inline bool parallel_run(size_t count, int n_threads, Fn&& fn, std::string& err) {
    if (count == 0) return true;
    const int nt = std::max(1, std::min<int>(n_threads, static_cast<int>(count)));
    if (nt == 1) {
        for (size_t i = 0; i < count; ++i) {
            if (!fn(i, err)) return false;
        }
        return true;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex err_mu;
    std::string first_err;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(nt));
    for (int t = 0; t < nt; ++t) {
        workers.emplace_back([&]() {
            std::string local_err;
            for (;;) {
                if (failed.load(std::memory_order_relaxed)) break;
                const size_t idx = next.fetch_add(1, std::memory_order_relaxed);
                if (idx >= count) break;
                if (!fn(idx, local_err)) {
                    if (!failed.exchange(true, std::memory_order_acq_rel)) {
                        std::lock_guard<std::mutex> lk(err_mu);
                        first_err = local_err;
                    }
                    break;
                }
            }
        });
    }
    for (auto& w : workers) w.join();
    if (failed.load()) {
        err = first_err;
        return false;
    }
    return true;
}

}  // namespace transcode
}  // namespace ink2
