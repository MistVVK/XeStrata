// include/strata/core/arch_defaults.hpp - the per-architecture defaults of switches that are exact everywhere.
//
// Every gfx1151 (Strix Halo, RDNA3.5, unified memory) switch in the table was measured on Aurora (experiments/s23 .. s26,
// the c-best3-v4.2 config) with BYTE-IDENTICAL output against the same engine with the switch off: the same token ids at
// 4K / 32K / 64K and on four models, and a bitwise harness per kernel.  They are therefore on by default on that
// architecture and nowhere else: a CUDA build and every other HIP architecture run the code they always ran.
// Switches that change bits (STRATA_PF_GEMM, STRATA_PF_FUSED, STRATA_HC_UPMIX, STRATA_PA_FAST, STRATA_HIP_WMMA,
// STRATA_SELECT_WMMA, STRATA_HC_Q8: rounding-level, KL-gated) stay opt-in in THIS table; docs/STRIX_HALO.md lists them.  Five of them
// (HIP_WMMA, PF_GEMM, PA_FAST, SELECT_WMMA, HC_UPMIX) are an opt-in set on an integrated Radeon (igpu_default_env() below).
//
// A switch the user set (to anything) is never overridden; STRATA_GFX1151_DEFAULTS=0 turns the whole table off.
#pragma once

#include "strata/kernels/gfx_arch.hpp"

#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The environment settings that are the defaults of `gcn_arch` (hipDeviceProp_t::gcnArchName; "gfx1151" or
/// "gfx1151:sramecc-:xnack-"): empty for every other architecture.  `defaults026` (STRATA_DEFAULTS=026, defaults_026.hpp): the gfx12 table of
/// v0.2.6 (the five prompt switches only, without the four 0.2.7 decode switches); the one-argument form asks the environment.
std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch, bool defaults026);
/// The same with the platform given instead of read from the build (a pure rule, so a test can check both).  0.2.7: on Windows the gfx1151
/// table leaves STRATA_SH_STREAM out (the fork is off, +6.9% decode on a Strix Halo Windows PC, same ids); Linux gfx1151 and every
/// other architecture are unchanged; `defaults026` keeps it in on both.  An explicit STRATA_SH_STREAM=1 turns the fork on anywhere.
std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch, bool defaults026, bool windows);
std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch);

/// AMD-Q (owner decision 10-31): the CPU prefill share (prefill.cpp, STRATA_PREFILL_CPU_SHARE) is armed by default for single-card serve on gfx12
/// (R9700 / RX 9070 / RX 9060, gfx1200 / gfx1201), with the CUDA change's limit (share_rules::default_share_limit: staged chunks up to 3,072 tokens).
/// Measured on an R9700 (p620, IQ3_S, 13,325 of 24,576 experts resident, 30 pool workers): prompt 500 +18.3%, 1K +30.6%, 2K -0.19%, 4K and 8K unchanged;
/// first-token KL 0.022 at 1K, top-1 29 of 29.  Pure architecture rule: false for gfx11 (Strix Halo gfx1151, 780M gfx1103, RX 7000), every other
/// family and a CUDA build; STRATA_GFX12_DEFAULTS=0 and STRATA_DEFAULTS=026 (`defaults026`) turn it off, and STRATA_PREFILL_CPU_SHARE=0 / auto /
/// a number still decide before it.  The caller adds the single-card, no-batch-slot and "experts are missing" conditions.
bool gfx12_prefill_share_default(const char* gcn_arch, bool defaults026);
bool gfx12_prefill_share_default(const char* gcn_arch);

/// The integrated Radeons (Phoenix / Hawk Point 780M gfx1103, Strix Point 890M / 880M gfx1150, Strix Halo gfx1151) whose
/// matrix-core prompt set is offered (opt-in; 021-minipc).  Pure architecture-name test; the caller adds the UMA test.
bool igpu_prompt_defaults_arch(const char* gcn_arch);

/// 021-minipc: the matrix-core prompt set changes bits.  The owner approved it as the DEFAULT of the Radeon 780M only (gfx1103,
/// integrated, Linux): prompts 1.60x (1K) / 2.03x (4K), decode within 0.1%, greedy ids identical in 6 of 6 pairs (notes-021-minipc.md).
/// gfx1150 / gfx1151 (no hardware) and Windows keep it off by default.  STRATA_IGPU_PROMPT_SET=1 turns it on for any integrated
/// arch of igpu_prompt_defaults_arch, STRATA_IGPU_PROMPT_SET=0 (or STRATA_IGPU_DEFAULTS=0) turns it off.
inline constexpr bool kIgpuPromptSetDefaultOn = true;
bool igpu_prompt_set_enabled(const char* gcn_arch, bool windows);

/// 021-minipc: UMA mode (the expert arena stays unregistered).  Exact (no bit changes) but NOT a default: measured on the 780M it was
/// 1-4% slower and did not stop the resets (4 of 20 runs against 1 of 10), so kIgpuUmaArenaDefaultOn is false.  When true it applies
/// to the Radeon 780M only (gfx1103, integrated, Linux).  EXCEPT a Strix Halo on Windows (gfx1151, integrated): there it is always on (0.2.1.2,
/// registering the arena used up all free VRAM and the engine did not start).  STRATA_UMA_REGISTER=1
/// registers the arena as before; STRATA_ARENA_PIN_GIB=none selects UMA mode per run.
inline constexpr bool kIgpuUmaArenaDefaultOn = false;
bool igpu_uma_arena_default(const char* gcn_arch, bool integrated, bool windows, bool on = kIgpuUmaArenaDefaultOn);

/// the registered expert arena asks for no transparent huge pages on an integrated gfx11 GPU under Linux (gfx1103 Radeon
/// 780M, gfx1150 880M / 890M, gfx1151 Strix Halo; detected from the device: `integrated` and the arch name).  MADV_HUGEPAGE on a registered
/// range adds direct compaction and khugepaged collapse in it; each move invalidates the range, KFD evicts the process's queues, and the
/// gc 11 MES firmware resets the GPU when a dispatch sits in a barrier (notes-minipc.md).  Radeon 780M, Q2_0 pack, registered 31.6 GiB arena,
/// one engine at a time: huge pages requested 9 starts, 8 ended in "unspecified launch failure" in the first prompt; not requested 3 of 3 ran.
/// Setup already wrote STRATA_NO_ARENA_THP=1 for its 780M opt-in; this makes it the engine's own rule.
/// STRATA_NO_ARENA_THP=0 asks for them (unset = this rule; =1 = never, as before).  Windows and discrete cards: unchanged.
bool igpu_arena_no_thp_default(const char* gcn_arch, bool integrated, bool windows);
/// the wait before READY (strata/core/kfd_settle.hpp) applies to the same class: an integrated gfx11 GPU under Linux (gfx1103,
/// gfx1150, gfx1151), where the registered expert arena is a KFD userptr and the kernel's page moves right after the load evict the
/// process's queues.  Never Windows, never a discrete card.  STRATA_NO_KFD_SETTLE=1 turns the wait off.
bool igpu_kfd_settle_default(const char* gcn_arch, bool integrated, bool windows);
/// the file-cache give-back during and after the expert load (strata/core/page_cache_drop.hpp) applies to the same class: an integrated
/// gfx11 GPU under Linux (gfx1103, gfx1150, gfx1151).  Never Windows, never a discrete card.  STRATA_NO_ARENA_CACHE_DROP=1 turns it off.
bool igpu_cache_drop_default(const char* gcn_arch, bool integrated, bool windows);
/// the prompt experts on Strata's own fused int8 kernels (STRATA_PF_FUSED=1) instead of llama.cpp's MMQ, for the same class: an integrated gfx11 GPU
/// under Linux (gfx1103, gfx1150, gfx1151).  Never Windows (gfx1151 there has it in igpu_default_env), never a discrete card.  apply_arch_defaults sets
/// STRATA_PF_FUSED=1 where the user has not set the variable; STRATA_PF_FUSED=0 is the way back to MMQ.
bool igpu_fused_prompt_default(const char* gcn_arch, bool integrated, bool windows);
/// The 16-bit prompt GEMMs FP16 in and out (STRATA_HIP_PROMPT_F16, gemm.cu prompt_f16()): `env` is the variable's text (nullptr = unset).  "1" = on, any other set
/// value = off (the way back); unset = the device rule: on for a Radeon 780M (gfx1103, integrated, Linux) and, #5151, for a discrete RX 6000 card (gfx1030 /
/// gfx1031 / gfx1032, Linux and Windows: rocBLAS has tuned kernels for FP16 in / FP16 out only there); off everywhere else (gfx11 and gfx12 cards, the integrated
/// gfx103x APUs, every other part).  RX 6700 XT (gfx1031, ROCm 10.1, IQ2_XS, 16 prompts of 2K, 64 forced tokens, truth = the exact FP32 products): KL to the truth
/// 0.00549 against the old default's 0.00621 (ratio 0.883 [0.829, 0.948]; first token 1.010 [0.672, 1.941]), prompts 11-14 % faster; 8K not measured.
/// STRATA_HIP_PROMPT_F16=0 keeps the old route.  Inline: src/prefill/gemm.cu uses it without linking arch_defaults.cpp.
inline bool prompt_f16_resolve(const char* env, const char* gcn_arch, bool integrated, bool windows) {
    if (env != nullptr) return env[0] == '1';
    if (integrated) return !windows && strata::kernels::gfx_arch_is(gcn_arch, "gfx1103");
    return strata::kernels::gfx_arch_is(gcn_arch, "gfx1030") || strata::kernels::gfx_arch_is(gcn_arch, "gfx1031") ||
           strata::kernels::gfx_arch_is(gcn_arch, "gfx1032");
}
/// Whether the expert arena skips its MADV_HUGEPAGE request: `no_arena_thp` is STRATA_NO_ARENA_THP (nullptr when unset): "0" asks for the huge
/// pages, any other value skips the request; unset leaves it to `igpu_rule` (igpu_arena_no_thp_default for this device).  `thp_always`: the
/// kernel's THP mode is "always" (the request would add nothing, #771).  Inline: src/core/pinned.cu (strata_core) uses it without
/// linking arch_defaults.cpp.
inline bool arena_skip_thp_request(const char* no_arena_thp, bool thp_always, bool igpu_rule) {
    if (no_arena_thp != nullptr) return no_arena_thp[0] != '0' || thp_always;   // the user's own value: 0 = ask for them, else skip
    return thp_always || igpu_rule;
}

/// The matrix-core prompt set an integrated GPU takes when `igpu_prompt_set_enabled()`: STRATA_HIP_WMMA, STRATA_PF_GEMM, STRATA_PA_FAST,
/// STRATA_SELECT_WMMA, STRATA_HC_UPMIX, all "1".  Empty unless `integrated` (hipDeviceProp_t::integrated: the device
/// shares the host's memory) AND the architecture is one of `igpu_prompt_defaults_arch`: a discrete card of the same
/// architecture family (RX 7900 XTX gfx1100, R9700 gfx1201, ...) and every CUDA / SYCL build get nothing.  These change
/// bits (KL-gated, notes-153-igpu.md); every switch keeps =0 as its off switch and STRATA_IGPU_DEFAULTS=0 turns the set off.
std::vector<std::pair<std::string, std::string>> igpu_default_env(const char* gcn_arch, bool integrated);

/// 0.2.10 (STRIX-2Q): the exact parallel-request stack (checkpoint pool, device slot copies, one-launch slot windows) for a Windows
/// Strix Halo (gfx1151) that runs batch slots (`--batch` / `--slots` of 2 or more).  Interleaved on the strix box (serve/server.py,
/// 4 clients, 3 rounds, IQ3_S): 51.7 / 50.7 tok/s without, 55.1 / 54.9 with, TTFT p50 2.55 -> 2.45 s, max chunk gap p95 1.36 -> 1.27 s,
/// no errors.  Every switch is bit-exact against the plain slots (QUAL-PAR on the Linux CUDA stack).  Not STRATA_BATCH_CONCURRENT_READ.
/// Empty for every other architecture / platform, below 2 slots, and with STRATA_GFX1151_DEFAULTS=0.  A switch the user set is kept.
std::vector<std::pair<std::string, std::string>> parallel_default_env(const char* gcn_arch, bool windows, int batch_slots);

/// The same nine switches for Linux CUDA (the Qwen engine of generate.cpp) with 2 or more batch slots: what serve/server.py's
/// PARALLEL_STACK_ENV set for a managed start, now the engine's own so a plain start is the same.  QUAL-PAR: bit-exact against the plain
/// slots (greedy identity at c=1, first-token KL 0).  `linux_cuda` is false on HIP, Windows, Metal and SYCL builds (empty then).  Each
/// switch keeps =0 as its off switch; a value the user set is kept.  Silent: the launcher printed nothing for these either.
std::vector<std::pair<std::string, std::string>> parallel_default_env_cuda(bool linux_cuda, int batch_slots);

/// Sets `parallel_default_env_cuda` in this process's environment where the user has not set the variable; returns the names it set.
/// Call once, before any engine code reads its switches.
std::vector<std::string> apply_parallel_defaults_cuda(bool linux_cuda, int batch_slots);

/// The tensor-core GDN prompt scan and the one-pass conv (QUAL-GDN, l620: prompt +4.4% at 4K, +5.5% at 16K, +5.7% at 32K; first-token KL mean
/// 0.0019, max 0.0125; owner yes 2026-11-07): STRATA_GDN_TC=1 and STRATA_GDN_CONVL2=2 on Linux CUDA, what serve/server.py's GDN_STACK_CUDA set.
/// The engine still gates the scan at run time (compute capability 8 or more, one SM per value head, shared memory) and falls back to the old
/// kernels.  Empty when `linux_cuda` is false (HIP, Windows, Metal, SYCL).  =0 turns each off; a value the user set is kept.  Silent.
std::vector<std::pair<std::string, std::string>> gdn_default_env_cuda(bool linux_cuda);
std::vector<std::string> apply_gdn_defaults_cuda(bool linux_cuda);

/// gfx12 (RDNA4, R9700 / RX 9000) on Linux, when EVERY visible card is gfx12 (the same rule as serve/server.py's GDN_STACK_GFX12): the chunked GDN scan
/// from the first 64-token chunk instead of from 30,000 tokens (owner yes 2026-11-07).  R9700: +3.5% prompt speed at 4K, +5.5% at 16K, first-token KL mean
/// 0.0025, 61-task suite 55 vs 56.  STRATA_GDN_CHUNKED=0 turns it off; a value the user set is kept.  Empty elsewhere.  Silent: the engine's own
/// "prompt chunks of 30,000+ tokens" line is not printed when the variable is set, as with a launcher start.
std::vector<std::pair<std::string, std::string>> gdn_chunked_default_env(bool linux_host, bool all_gfx12);
std::vector<std::string> apply_gdn_chunked_default(bool linux_host, bool all_gfx12);

/// 0.2.10 (STRIX-TDR): a Windows Strix Halo (gfx1151) launches the verify window's graph from a helper thread, so the main thread can
/// serve the GPU's doorbells while the launch call is still running.  On the strix box the launch call of the first window sometimes never
/// returned (host in the driver, GPU already past layer 0, "GPU rang 1, flags served 0": the Q4_K_XL start stall); the helper thread
/// removed it (see STRIX-2Q-RESULTS.md).  The value is STRATA_VERIFY_ASYNC_LAUNCH=1 (same switch as the CUDA instrumented-launch path).
/// Empty for every other architecture / platform and with STRATA_GFX1151_DEFAULTS=0.  A switch the user set is kept.
std::vector<std::pair<std::string, std::string>> launch_default_env(const char* gcn_arch, bool windows);

/// Sets `arch_default_env(gcn_arch)` and, when `integrated`, `igpu_default_env(gcn_arch, true)` in this process's
/// environment where the user has not set the variable, prints one line to stderr per table when it set anything, and
/// returns the names it set.  `integrated` is true when every visible GPU is an integrated one.  Call once, before any
/// engine code reads its switches (they are read lazily, at first use).  `mmq_built`: llama.cpp's MMQ prompt kernels are in the build (the caller knows:
/// STRATA_PREFILL_MMQ is defined on strata_prefill only); the gfx12 K-quant line names them only then.
std::vector<std::string> apply_arch_defaults(const char* gcn_arch, bool integrated = false, int batch_slots = 0, bool mmq_built = true);

}  // namespace strata::core
