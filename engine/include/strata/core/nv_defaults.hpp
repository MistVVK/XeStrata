// strata/core/nv_defaults.hpp - the per-architecture defaults of switches that are exact (bitwise) on NVIDIA sm_86 and sm_60.
//
// 148-nv measured every switch of the 0.1.45 bit-exact ON set, one at a time and in the stacked set, on an RTX 3060, an RTX A4000
// (sm_86) and a Tesla P100 (sm_60): a switch is in a table only when its token ids equal the default's and it was faster on that
// architecture (5+ interleaved pairs of fresh engines, noise arm, equal expert-cache slots; notes-148-nv.md).  The identity gate
// (10 prompts + exact_pp 4K/20K, fixed flags) is IDENTICAL with the whole table on.  Switches that change bits, hang or lost speed
// on a card (STRATA_EXCHANGE_ROTATE, STRATA_PCIE_BALANCE, STRATA_QSA_WARP, ...) are not in any table.
//
//   STRATA_SM86_PERF=0   turns the sm_86 table off (cards of compute capability 8.6: RTX 3050-3090, RTX A2000-A6000)
//   STRATA_SM70_PERF=0   turns the sm_70 table off (Volta: Tesla V100 / Titan V / Quadro GV100; also off under STRATA_DEFAULTS=026)
//   STRATA_SM60_PERF=0   turns the sm_60 table off (Pascal GP100: Tesla P100)
//   STRATA_SM121_PERF=0  turns the sm_121 table off (GB10: NVIDIA DGX Spark; new in 0.2.7.1, also off under STRATA_DEFAULTS=026).  Its
//                        switches are measured speed settings, not bit-exact ones: the fused prompt kernels and the FP4-activation NVFP4
//                        prompt round differently from the paths they replace (24 of 24 greedy answers identical; no KL run yet).
//
// A switch the user set (to anything, including 0) is never overridden.  Applied once at start, before any engine code reads its
// switches; the table is used only when every visible CUDA device has the same compute capability.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// STRATA_ADAPT_FETCH=2 as a default (0.2.17, owner yes 11-19; BW-9 on 2x RTX 5060 Ti 16 GB, phaseA-tests/bw/RESULTS-BW-F.md;
/// 0.2.20 extended to <= 13 GiB sm_86/sm_89 consumer cards: RTX 3060/3060 Ti/3070/3080, RTX 4060/4060 Ti 8G/4070/4070 Ti):
/// compute capability 12.x cards with at most kAdaptFetchMaxVramBytes (16 GiB), or sm_86/sm_89 cards with at most
/// kAdaptFetchLowVramBytes (13 GiB). A card with more VRAM keeps it opt-in. The switch is read once per process and only a
/// one-card run uses it (a layer split, helper cards, a peer tier, --batch, resident RAM mode and the other cases the start-up
/// line names turn it off), so the card that decides is the engine's main one (STRATA_PRIMARY_DEVICE).
constexpr uint64_t kAdaptFetchMaxVramBytes = 16ull << 30;
constexpr uint64_t kAdaptFetchLowVramBytes = 13ull << 30;
/// Whether STRATA_ADAPT_FETCH=2 is the default of a card with compute capability major.minor (raw) and `vram_bytes` of memory
/// (cudaDeviceProp::totalGlobalMem; 0 = unknown, which is no).  Not about the environment: the off switches are in nv_default_env.
bool adapt_fetch_default_card(int major, int minor, uint64_t vram_bytes);

/// The environment settings that are the defaults of a card with compute capability major.minor (raw, as the driver reports it);
/// empty for every other card and when its off switch is 0.
std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor);
/// The same with the answer of STRATA_DEFAULTS=026 given (the sm_121 table is empty then), so a test can check both without the environment.
std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor, bool defaults026);
/// The same for a main card with `main_vram_bytes` of memory: a card of compute capability 12.x with at most kAdaptFetchMaxVramBytes also gets
/// STRATA_ADAPT_FETCH=2 (adapt_fetch_default_card).  main_vram_bytes = 0 (the two overloads above) leaves it out.
std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor, bool defaults026, uint64_t main_vram_bytes);

/// Sets `nv_default_env(major, minor)` in this process's environment where the user has not set the variable, prints one line to
/// stderr when it set anything, and returns the names it set.  main_vram_bytes: the engine's main card's memory (see nv_default_env); 0 = not given.
std::vector<std::string> apply_nv_defaults(int major, int minor, uint64_t main_vram_bytes = 0);
/// Whether apply_nv_defaults put STRATA_ADAPT_FETCH in the environment (it was the default, not the user's setting); the engine's lines about the switch say so.
/// (Inline, so a build that does not link nv_defaults.cpp - Metal, SYCL - still compiles the engine's lines: it reads false there.)
inline bool& adapt_fetch_default_flag() { static bool v = false; return v; }
inline bool adapt_fetch_is_default() { return adapt_fetch_default_flag(); }

}  // namespace strata::core

namespace strata::core {

/// NVIDIA default that changes the answers (round 0212, NVSINGLE; the owner said yes 11-03): STRATA_MMQ_KQ=1, the K-quant MMQ prompt
/// kernels for the experts of Unsloth's UD-Q4/Q5/Q6_K_XL files (Q4_K / Q5_K / Q5_1 / Q6_K), which otherwise are dequantized to FP16 and multiplied by
/// cuBLAS.  A pack without K-quant experts never reads the variable.  Cards of compute capability 8.0 and up only (measured on sm_86 and sm_120).
/// On by default; STRATA_MMQ_KQ=0, STRATA_NV_KQ_DEFAULT=0 and STRATA_DEFAULTS=026 are the off switches (kNvKqDefaultOn=false would leave it behind
/// STRATA_NV_KQ_DEFAULT=1).  Measured: RTX PRO 6000 prompt +26..38%, A6000 +2..11%, first-token KL 0.024 / 0.035, 61-task suite 54 vs 53.
/// min_major / min_minor are the lowest compute capability among the visible cards.
constexpr bool kNvKqDefaultOn = true;
std::vector<std::pair<std::string, std::string>> nv_kq_default_env(int min_major, int min_minor, bool defaults026);
std::vector<std::pair<std::string, std::string>> nv_kq_default_env(int min_major, int min_minor);
/// Sets what nv_kq_default_env returns where the user has not set the variable; prints one line to stderr when it set anything and `say` is true.  fused_here: Strata's own fused
/// kernels read the K-quant experts on this machine (nv_kq_fused_here for a visible card); the line then says so, otherwise it is the earlier line, byte for byte.
/// The engine calls it with say = false before it knows the model (option time) and prints the line itself once the expert layout is loaded, only when
/// nv_kq_line_wanted says the loaded set has experts the switch decides for.
std::vector<std::string> apply_nv_kq_default(int min_major, int min_minor, bool fused_here = false, bool say = true);
/// A ggml type of an expert tensor whose prompt products STRATA_MMQ_KQ / STRATA_PF_FUSED_KQ decide: Q4_K, Q5_K, Q6_K and Q5_1 (the four types
/// prefill::mmq::supported() hands to the K-quant switch).  IQ types, Q2_0, Q8_0, Q4_0 ... are not decided by it.
bool nv_kq_expert_type(int ggml_type);
/// Whether the start-up line about K-quant experts is true of the loaded set: some layer has such a type as gate/up or as down (pairs of ggml types per layer).
/// IQ3_S / IQ2_XS sets: no.  UD-Q4_K_XL (Q4_K / Q5_K gate/up): yes.  A set whose only K-quant-family tensor is a Q5_1 down: yes, the switch decides that product.
bool nv_kq_line_wanted(const std::vector<std::pair<int, int>>& layer_gate_up_down_types);
/// Whether a CUDA card (raw compute capability, its multiprocessor count) reads UD-Q4_K_XL's K-quant experts (Q4_K / Q5_K gate/up with a Q5_1 or Q8_0 down) on Strata's
/// own fused kernels, from the environment as the engine reads it: the card rule of fused_tile::kq_takes with STRATA_PF_FUSED_KQ, STRATA_PF_FUSED_SM40 and
/// STRATA_PF_FUSED_SM86; STRATA_PF_FUSED=0 turns every fused kernel off.
bool nv_kq_fused_here(int major, int minor, int sms);
/// The line apply_nv_kq_default prints (without the newline).
std::string nv_kq_default_line(bool fused_here);
/// The same line for a build with (mmq_built) or without llama.cpp's MMQ kernels; the caller knows the build (STRATA_PREFILL_MMQ is defined on strata_prefill only).
std::string nv_kq_default_line(bool fused_here, bool mmq_built);

/// dflash2-kern: the speculative runs' faster start-up adaptation of the expert tier (opt-in on sm_86 with STRATA_SPEC_ADAPT_SM86=1 since owner round 9; STRATA_SPEC_ADAPT_EVERY / _WARM, read by
/// generate.cpp): sm_86 only (z440 A4000 Q2_0; the P100 is opt-in through the two variables, notes-dflash2-kern.md).  Not part of
/// the exact table above: the residency it changes decides which experts the GPU rounds, as the adaptive tier always does.  The user's
/// own settings and the cards' off switches (STRATA_SM86_PERF=0) is kept.
void apply_spec_adapt_default(int major, int minor);

}  // namespace strata::core
