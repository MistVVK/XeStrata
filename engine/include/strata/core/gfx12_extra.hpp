// include/strata/core/gfx12_extra.hpp - defaults for gfx12 (RDNA4: R9700, RX 9070, RX 9060) that came out of the PR-AMD pass of 2026-11-01 and are
// kept apart from the main table (arch_defaults.cpp) so each can be dropped alone.  Same off switches as that table: STRATA_GFX12_DEFAULTS=0
// and STRATA_DEFAULTS=026 give the old behaviour, and a variable the user sets is never overridden.
//
// Exact (bit for bit the same answers):
//   STRATA_HEAD_MIX_MULTI=1  the final hyper-connection mixer of the verify window and of the draft layer as one fused_gr_read_multi
//                            instead of one gr_read per token (#3264, bsorensen110).  R9700 (p620, IQ3_S, serve, static residency,
//                            8 interleaved pairs, A/A beside it): json_extract +0.82% (6/8 pairs), agent +0.63% (6/8), code_py -0.96%
//                            (3/8; that prompt's A/A moves +-3.5% a pair), greedy ids identical in 16/16 and 15/16 request sets.
//
// Changes the answers (needs the owner's yes; this is the separate, droppable commit):
//   STRATA_SELECT_WMMA=1 + STRATA_SELECT_ALT_MIN_TOKENS=28672   the gfx12 matrix-core QSA block scorer (already in the engine as an opt-in) for
//                            chunks whose context so far is 28,672 tokens or more; a read that stays below that is bit-identical to the warp
//                            kernel (first-token KL 0.00000 on twelve 8K-24K prompts).  R9700 (p620, IQ3_S, serve, interleaved pairs, prompt
//                            tok/s vs unset, 3 pairs): 32K +1.83% (3/3), 98K +5.74% (3/3, steady).  First-token KL against unset on seven 30K-96K
//                            prompts: mean 0.0018, max 0.0055, top-1 7/7; needle recall 25/25 at 8K..96K x 5 depths.  Without the floor (STRATA_SELECT_WMMA=1
//                            alone, 4 pairs): 4K +0.34%, 8K +0.94%, 16K +1.73%, 24K +3.34%, 32K +4.5..6.0%, 98K +6.9..7.3%, but the values change from 8K
//                            up (KL mean 0.0056, max 0.0193, top-1 11/12 at 8K-24K); a floor of 16,384 was no better (32K -0.27%, noisy).  The floor is
//                            the owner's trade between "ordinary prompts keep their exact answers" and a few percent more at 32K.
//                            Chosen over PR 3073's simt2 (author Vop3p), which is slower below 32K, has KL 0.0029 / 0.0095 and yields to WMMA on gfx12.
//                            Off: STRATA_SELECT_WMMA=0, STRATA_GFX12_DEFAULTS=0, STRATA_DEFAULTS=026.
//   STRATA_MMQ_KQ=1  (Linux only, owner 11-03 "do all 5")  the K-quant experts of Unsloth's UD-Q4/Q5/Q6_K_XL files (Q4_K / Q5_K / Q5_1 / Q6_K) run their prompt
//                            products on llama.cpp's MMQ kernels instead of being dequantized to FP16 and multiplied by hipBLASLt (setup already wrote it for UD-Q5/Q6).
//                            R9700 (r9700, UD-Q4_K_XL, serve, same job, interleaved): prompt 4K +10.3%, 8K +14.3%, 32K +16.8%; first-token KL against the FP16 path
//                            0.031 (24 prompts of ~2K; agent turns 0.065), top-1 23/24.  A pack without K-quant experts (IQ3_S...) never reads the variable.
//                            Off: STRATA_MMQ_KQ=0, STRATA_GFX12_DEFAULTS=0, STRATA_DEFAULTS=026.  (The fused variant, STRATA_PF_FUSED_KQ, stays a flag.)
#pragma once

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "strata/core/defaults_026.hpp"
#include "strata/core/gfx12_platform_defaults.hpp"
#include "strata/kernels/gfx_arch.hpp"

namespace strata::core {

/// `defaults026`: STRATA_DEFAULTS=026 (defaults_026.hpp reads it once, so a test passes it).
inline std::vector<std::pair<std::string, std::string>> gfx12_exact_extra_env(const char* gcn_arch, bool defaults026) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!strata::kernels::gfx_arch_is_gfx12(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX12_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    if (defaults026) return t;
    t.push_back({"STRATA_HEAD_MIX_MULTI", "1"});
    return t;
}
inline std::vector<std::pair<std::string, std::string>> gfx12_exact_extra_env(const char* gcn_arch) {
    return gfx12_exact_extra_env(gcn_arch, strata::qwen_defaults_026());
}

/// The value-changing defaults of gfx12 (see above).  `defaults026`: STRATA_DEFAULTS=026.
inline std::vector<std::pair<std::string, std::string>> gfx12_value_changing_env(const char* gcn_arch, bool defaults026) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!strata::kernels::gfx_arch_is_gfx12(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX12_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    if (defaults026) return t;
    t.push_back({"STRATA_SELECT_WMMA", "1"});
    t.push_back({"STRATA_SELECT_ALT_MIN_TOKENS", "28672"});
    if (kGfx12PromptDefaultsOn) t.push_back({"STRATA_MMQ_KQ", "1"});   // Linux only, like the other approved gfx12 prompt defaults
    return t;
}
inline std::vector<std::pair<std::string, std::string>> gfx12_value_changing_env(const char* gcn_arch) {
    return gfx12_value_changing_env(gcn_arch, strata::qwen_defaults_026());
}

}  // namespace strata::core
