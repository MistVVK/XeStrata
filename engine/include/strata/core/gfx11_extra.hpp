// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// include/strata/core/gfx11_extra.hpp - defaults for discrete RDNA3 (gfx1100 RX 7900 XTX / XT / GRE / W7900 / W7800,
// gfx1101 RX 7800 XT / 7700 XT / W7700, gfx1102 RX 7600 / XT) kept apart from arch_defaults.cpp's main table, following
// gfx12_extra.hpp's structure so every switch and family can be turned off independently.
//
// Off switches:
//   - STRATA_GFX11_DEFAULTS=0       turns off all discrete RDNA3 defaults (exact, WMMA prompt, Q8 HC, CPU share, chunk ceiling, PcieAuto)
//   - STRATA_GFX11_WMMA_DEFAULTS=0  turns off only the rounding-changing matrix-core / Q8 switches while keeping the bitwise-exact switches
//   - STRATA_DEFAULTS=026           restores v0.2.6 behaviour everywhere
//   - Any explicit NAME=0 / NAME=val set by the user always wins and is never overridden.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "strata/core/defaults_026.hpp"
#include "strata/kernels/gfx_arch.hpp"

namespace strata::core {

inline bool gfx11_defaults_disabled(bool defaults026) noexcept {
    if (defaults026) return true;
    const char* off = std::getenv("STRATA_GFX11_DEFAULTS");
    return off != nullptr && off[0] == '0';
}

inline bool gfx11_wmma_defaults_disabled(bool defaults026) noexcept {
    if (gfx11_defaults_disabled(defaults026)) return true;
    const char* woff = std::getenv("STRATA_GFX11_WMMA_DEFAULTS");
    return woff != nullptr && woff[0] == '0';
}

/// Bitwise-exact prompt and decode switches for discrete RDNA3 (gfx1100 / gfx1101 / gfx1102).
/// Note: STRATA_SH_STREAM is intentionally omitted on discrete RDNA3 (avoiding WDDM multi-stream submission overhead on Windows).
inline std::vector<std::pair<std::string, std::string>> gfx11_discrete_exact_env(
    const char* gcn_arch, bool defaults026, bool windows) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!strata::kernels::gfx_arch_is_discrete_rdna3(gcn_arch)) return t;
    if (gfx11_defaults_disabled(defaults026)) return t;
    t = {
        // Prompt (bitwise-exact)
        {"STRATA_GDN_HEAD", "1"},
        {"STRATA_GDN_PP", "2"},
        {"STRATA_GDN_CONVL2", "1"},
        {"STRATA_GDN_NOY", "1"},
        {"STRATA_CVEC_FUSE", "1"},
        {"STRATA_PF_PAD", "1"},
        {"STRATA_TOPK_REG_MIN_BLOCKS", "4096"},
        // Decode (bitwise-exact)
        {"STRATA_Q8_PACKED", "1"},
        {"STRATA_Q6_PACKED", "1"},
        {"STRATA_MMVF_ROWS", "1"},
        {"STRATA_ATTN_LANECELL", "1"},
        {"STRATA_ATTN_MERGE_V2", "1"},
        {"STRATA_EXPERT_V2", "1"},
        {"STRATA_EXPERT_V2K", "1"},
        {"STRATA_EXPDEC", "1"},
        {"STRATA_TSUM", "1"},
        {"STRATA_LFUSE", "1"},
        {"STRATA_GDN_SPLIT", "1"},
        {"STRATA_QFUSE", "1"},
        {"STRATA_PLE_BATCH", "1"},
        {"STRATA_HEAD_MIX_MULTI", "1"},
        {"STRATA_SMALL_UNR", "3"},
        {"STRATA_GR_Q32_G", "2"},
        {"STRATA_GR_Q8_PFS", "1"},
    };
    if (windows) {
        // Helper thread launches the verify window while the main thread serves CPU doorbells (prevents WDDM first-window launch stalls),
        // and PAL SDMA enables async DMA copy engines on Windows WDDM so H2D expert transfers overlap compute waves.
        t.push_back({"STRATA_VERIFY_ASYNC_LAUNCH", "1"});
        t.push_back({"GPU_ENABLE_PAL_SDMA", "1"});
    }
    return t;
}

inline std::vector<std::pair<std::string, std::string>> gfx11_discrete_exact_env(const char* gcn_arch, bool windows) {
    return gfx11_discrete_exact_env(gcn_arch, strata::qwen_defaults_026(), windows);
}

/// KL-validated matrix-core prompt & MoE switches for discrete RDNA3 (gfx1100 / gfx1101 / gfx1102), matching the Windows
/// gfx1151 (Strix Halo) matrix-core set plus PF_P1A_L2 (covering all 48/48 layers of UD-Q4_K_XL / UD-Q5_K_XL),
/// DFLASH_WMMA, and HIP_FAST_DEC (GDN_CHUNKED remains opt-in on gfx11 to keep 4K+ prefill KL <= 0.016).
inline std::vector<std::pair<std::string, std::string>> gfx11_discrete_wmma_env(
    const char* gcn_arch, bool defaults026) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!strata::kernels::gfx_arch_is_discrete_rdna3(gcn_arch)) return t;
    if (gfx11_wmma_defaults_disabled(defaults026)) return t;
    t = {
        {"STRATA_HIP_WMMA", "1"},
        {"STRATA_PA_FAST", "1"},
        {"STRATA_SELECT_WMMA", "1"},
        {"STRATA_PF_GEMM", "1"},
        {"STRATA_HC_UPMIX", "1"},
        {"STRATA_PF_FUSED", "1"},
        {"STRATA_PF_F16", "2"},
        {"STRATA_PF_DM16", "1"},
        {"STRATA_PF_P1A", "1"},
        {"STRATA_PF_P1A_L2", "1"},
        {"STRATA_DFLASH_WMMA", "1"},
        {"STRATA_HIP_FAST_DEC", "1"},
    };
    return t;
}

inline std::vector<std::pair<std::string, std::string>> gfx11_discrete_wmma_env(const char* gcn_arch) {
    return gfx11_discrete_wmma_env(gcn_arch, strata::qwen_defaults_026());
}

/// Bit-exact parallel batch-slot stack (--batch / --slots >= 2) on discrete RDNA3 (Linux and Windows).
inline std::vector<std::pair<std::string, std::string>> gfx11_discrete_parallel_env(
    const char* gcn_arch, int batch_slots, bool defaults026) {
    std::vector<std::pair<std::string, std::string>> t;
    if (batch_slots < 2 || !strata::kernels::gfx_arch_is_discrete_rdna3(gcn_arch)) return t;
    if (gfx11_defaults_disabled(defaults026)) return t;
    return {
        {"STRATA_CKPT_POOL", "1"},
        {"STRATA_CKPT_POOL_READY", "4"},
        {"STRATA_SLOT_CKPT_MOVE", "1"},
        {"STRATA_SLOT_COPY_DIRECT", "1"},
        {"STRATA_SLOT_COPY_BATCHSYNC", "1"},
        {"STRATA_BATCH_GDN_GROUPS", "1"},
        {"STRATA_BATCH_ATTN_ROWS", "1"},
        {"STRATA_BATCH_SELF_COMMIT", "1"},
        {"STRATA_QSA_SLOTS", "1"},
    };
}

inline std::vector<std::pair<std::string, std::string>> gfx11_discrete_parallel_env(const char* gcn_arch, int batch_slots) {
    return gfx11_discrete_parallel_env(gcn_arch, batch_slots, strata::qwen_defaults_026());
}

/// Single-GPU CPU prefill share default on discrete RDNA3 (up to share_rules::default_share_limit = 3,072 tokens).
inline bool gfx11_prefill_share_default(const char* gcn_arch, bool defaults026) noexcept {
    if (!strata::kernels::gfx_arch_is_discrete_rdna3(gcn_arch)) return false;
    return !gfx11_wmma_defaults_disabled(defaults026);
}

inline bool gfx11_prefill_share_default(const char* gcn_arch) noexcept {
    return gfx11_prefill_share_default(gcn_arch, strata::qwen_defaults_026());
}

/// Bare `--prefill auto` prompt-chunk ceiling for discrete RDNA3 on Linux (and explicit opt-in on Windows via STRATA_GFX11_CHUNK):
/// unset -> 32768 (unless defaults026 -> 0); "16" -> 16384; "0" -> 0 (engine's 8192 / Windows 6144); "1" / "32" / other -> 32768.
inline int64_t gfx11_chunk_ceiling(const char* env, bool defaults026 = false) noexcept {
    if (env == nullptr || env[0] == '\0') return defaults026 ? 0 : 32768;
    if (std::strcmp(env, "16") == 0) return 16384;
    if (env[0] == '0') return 0;
    return 32768;
}

}  // namespace strata::core
