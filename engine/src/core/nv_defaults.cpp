// src/core/nv_defaults.cpp - see include/strata/core/nv_defaults.hpp.
#include "strata/core/nv_defaults.hpp"

#include "strata/core/defaults_026.hpp"
#include "strata/core/emulate.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/prefill/fused_tile_rule.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::core {

namespace {
bool off(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '0';
}
}  // namespace

bool adapt_fetch_default_card(int major, int minor, uint64_t vram_bytes) {
    const int mj = strata::cc_major_of(major);
    (void) minor;
    return mj == 12 && vram_bytes > 0 && vram_bytes <= kAdaptFetchMaxVramBytes;
}

std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor) {
    return nv_default_env(major, minor, strata::qwen_defaults_026());
}

std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor, bool defaults026) {
    return nv_default_env(major, minor, defaults026, 0);
}

std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor, bool defaults026, uint64_t main_vram_bytes) {
    std::vector<std::pair<std::string, std::string>> t;
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    if (mj == 8 && mn == 6) {
        if (off("STRATA_SM86_PERF")) return t;
        t = {
            // 148-nv, RTX 3060 + RTX A4000 (IQ3_XXS, default flags, 8 pairs): the stack +9.6% / +14.6% (8/8 each), identity gate IDENTICAL.
            // Each is in because removing it from the full stack cost speed on a card (drop-one, 4 rounds) and saved none on the other.
            // Repeatable output (0.2.15, option 2 of phaseA-tests/par/RESULTS-REPEAT.md (d)): STRATA_ADAPT_NOWAIT=1 is no longer a default here.
            // With it, a verify window ran a swapped-in expert on the GPU or the CPU by its copy's timing, and two starts of the same request
            // differed (RTX A6000, fixed draft table: 7 of 24 requests identical; without it 24 of 24 in 3 of 3 starts).  Its cost there:
            // decode 0.988 [0.969, 1.002] over 3 pairs.  It was in the stack for "A4000 -6.3 points when removed (with ADAPT_LAG), 3060 -0.8";
            // ADAPT_LAG=2 stays, so the copies still get two windows to land.  STRATA_ADAPT_NOWAIT=1 brings it back.
            {"STRATA_ADAPT_LAG", "2"},
            {"STRATA_CPU_IQ_ALL_EXACT1", "1"},   // A4000 -3.3, 3060 -0.6
            {"STRATA_SIDE_MIXER", "2"},          // A4000 -1.2, 3060 -2.3
            {"STRATA_ATTN_DAILY", "1"},          // -1.2, -1.3
            {"STRATA_GR_UP75", "1"},             // -0.9, -1.8
            {"STRATA_GU_SWIGLU", "1"},           // -0.8, -1.4
            {"STRATA_QSA_ROWS", "1"},            // -0.5, -1.2
            {"STRATA_B6_MMVQ_ROWS", "1"},        // A4000 0, 3060 -1.7
        };
    } else if (mj == 12 && mn == 0) {
        // RTX 50 / PRO Blackwell (compute capability 12.0; a gfx12 HIP card reports 12.0 too, but this table is applied by the CUDA build only).  STRATA_SM120_PERF=0 and STRATA_DEFAULTS=026 leave it out.
        if (off("STRATA_SM120_PERF") || defaults026) return t;
        t = {
            // the hyper-connection read's fused up projection + mix (gr_upmix), chunks of 33+ tokens.  BW-13 (RTX 5090, UD-Q4_K_XL): BASE-05 identity 27 of 27 (9 prompts, 3 cold starts, up to
            // 31,817 tokens) against the baseline of the released engine; STRATA_HC_UPMIX_CHECK on an 8K prompt: mixed rel RMS 0, max |diff| 0 (the fused kernel is the default pair's bits).
            // STRATA_HC_UPMIX=0 is the way back.  (phaseA-tests/bw/RESULTS-BW-A.md)
            {"STRATA_HC_UPMIX", "1"},
            // BW-G (RTX 5090, UD-Q4_K_XL, DFlash2 spec 6, tail skip OFF, paired rounds; phaseA-tests/bw/RESULTS-BW-C.md, RESULTS-BW-G.md): three more bit-exact decode switches, each alone and the three
            // together BASE-05 27 of 27 identical to the released engine's baseline, the three together decode +2.0% (20 of 25 paired rounds up), window -2.9%.  Each has =0 as the way back.
            // the next layer's dense weights prefetched into L2 while the window waits for the CPU experts (doorbell graph only; the all-resident graph never runs it)
            {"STRATA_L2_PF", "1"},
            // the one-row-per-block Q8_0 dense kernel (each row keeps its own partial sums: the two-row kernel's bits)
            {"STRATA_Q80_ROWS1", "1"},
            // the grouped native experts' mode-8 layout (the GB10 / P100 / P40 default; exact against the CUDA layout on UD-Q4_K_XL)
            {"STRATA_EXP_MODE", "8"},
        };
        // 0.2.17 (owner yes 11-19), a card of 16 GiB or less: the PCIe share also fills the expert cache (not bit-exact: CPU and GPU round differently; the tier is not deterministic).
        // BW-9 (2x RTX 5060 Ti 16 GB, UD-Q4_K_XL, DFlash2 spec 6, tail skip OFF, product flags, 3 paired rounds; phaseA-tests/bw/RESULTS-BW-F.md): decode +4 to +43% by kind (medians 1.087 to 1.325, 3 of 3 rounds up),
        // expert cache hit rate 54.4 -> 62.0%, forced-token KL against the FP16 prompt path 0.948 [0.830, 1.063] of the default's (PASS), prompts level except json 169 at +3 to +5%.  Applies to a one-card run only
        // (the engine turns it off, with a line, for a layer split, helper cards, --batch and the other cases of the DETAILS.md row).  STRATA_ADAPT_FETCH=0 is the way back.
        if (adapt_fetch_default_card(major, minor, main_vram_bytes)) t.push_back({"STRATA_ADAPT_FETCH", "2"});
    } else if (mj == 6 && mn == 0) {
        if (off("STRATA_SM60_PERF")) return t;
        t = {
            // 148-nv, Tesla P100 (IQ3_XXS, default flags, 8 pairs): the stack +17.8% (8/8), identity gate IDENTICAL.
            {"STRATA_ADAPT_NOWAIT", "1"},        // -10 points when removed (with ADAPT_LAG)
            {"STRATA_ADAPT_LAG", "2"},
            {"STRATA_SIDE_MIXER", "2"},          // -1.8
            {"STRATA_FUSE_WAIT", "2"},           // -1.6
            {"STRATA_B6_MMVQ_ROWS", "1"},        // -1.3
            {"STRATA_FUSE_DECODE", "1"},         // -1.3
            {"STRATA_GR_UP75", "1"},             // -1.2
        };
    } else if (mj == 12 && mn == 1) {
        // GB10 (NVIDIA DGX Spark, 48 SMs, unified memory): only what was measured on that part goes in.  New in 0.2.7.1, so
        // STRATA_DEFAULTS=026 (the master switch of everything since 0.2.6) leaves the table out, like STRATA_SM121_PERF=0.
        if (off("STRATA_SM121_PERF") || defaults026) return t;
        t = {
            // the fused IQ prompt kernels.  IQ3_S, 64K context, fresh 4K/16K/32K prompts at T=0, 3 paired rounds against
            // the same tree without it: prompt 1.139 / 1.149 / 1.121, decode unchanged; 5 of 5 greedy answers byte-
            // identical with it on and off (docs/DGX_SPARK_LINUX.md; HANDOVER §6.2).  It rounds differently from MMQ,
            // so it is a default with an off switch, like the sm_86 table.
            {"STRATA_PF_FUSED", "1"},
            // the same fused prompt kernels for the K-quant experts of Unsloth's UD-Q4_K_XL files (Q4_K / Q5_K gate/up, Q5_1 / Q8_0 down) instead of llama.cpp's MMQ.
            // UD-Q4_K_XL (Flash-Next), expert cache budget 12000 (15,283 of 24,576 experts resident, so the box keeps over 16 GiB of its 121 GiB free), two starts a pair,
            // against the FP16 prompt path's text, 48 real-text prompts (24 at 2,048 and 24 at 4,096 tokens, 256 forced tokens each, 12,288 positions): forced-token
            // KL kq / MMQ 0.895 [0.808, 0.989] (0.864 [0.756, 0.976] at 2K, 0.928 [0.791, 1.088] at 4K), top-1 against the FP16 path's 96.31% / 95.74%, the same
            // text twice byte for byte on both sides (repeat floor 0); prompt speed kq / MMQ, 10 interleaved pairs, median of the per-round ratios 1.10 / 1.26 / 1.18 / 1.53 at 2K / 4K / 8K / 20K (faster in 7 / 8 / 8 / 9 of 10; the rounds scatter 0.5 to 2.0)
            // (phaseA-tests/par/RESULTS-SPARK-KQ.md).  STRATA_PF_FUSED_KQ=0 is the way back (a switch you set is kept); it is off with the rest of the table by STRATA_SM121_PERF=0.
            {"STRATA_PF_FUSED_KQ", "1"},
            // NVFP4 prompt MMQ with FP4 activations (W4A4), the Blackwell path the engine keeps opt-in.  On this part, the
            // NVFP4 pack measured prompt 1.106x at 4K and 1.229x at 16K against the default W4A8 MMQ, decode unchanged
            // (1.015x), and 24 of 24 greedy answers byte-identical with it on and off.  Opt-in upstream because it is only
            // checked on Blackwell; switched on here for the same reason as the row above, with the same off switch.
            {"STRATA_PREFILL_NVFP4", "w4a4"},
            // the hyper-connection read's fused up projection + mix (gr_upmix).  Measured on IQ3_S against the unfused read:
            // decode 1.054x [1.007, 1.089], prompt unchanged, and 24 of 24 greedy answers byte-identical (the parity test
            // src/prefill/gr_upmix_parity.cu covers it upstream).  The HC read is ~13% of a decode round here, which is
            // why a fusion on it moves the number at all.
            {"STRATA_HC_UPMIX", "1"},
            // Run the mixer's input-only QSA/GDN work on graph branches.  With the grouped experts' mode 8 selected,
            // fixed-work IQ3_S decode measured +6.68% paired median and +6.67% geometric mean over 8 balanced pairs
            // (AB +11.60%, BA +6.68%, identical 512-token output and verifier work).  The branch changes no arithmetic.
            {"STRATA_SIDE_MIXER", "2"},
            // the cost-aware MTP window: pick the verify window from what a draft costs against what it has been paying.
            // Measured on IQ3_S: decode 1.047x [1.019, 1.079], prompt unchanged, 24 of 24 greedy answers byte-identical.
            // It only changes which drafts are attempted, and every draft is verified, so the answers cannot drift.
            {"STRATA_SPEC_COST", "1"},
        };
        // a 12.1 part reports its unified memory, far above 16 GiB, so this stays off there; the rule is the same as for 12.0
        if (adapt_fetch_default_card(major, minor, main_vram_bytes)) t.push_back({"STRATA_ADAPT_FETCH", "2"});
    }
    return t;
}

std::vector<std::string> apply_nv_defaults(int major, int minor, uint64_t main_vram_bytes) {
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    std::vector<std::string> set;
    bool fetch_set = false;
    for (const auto& kv : nv_default_env(major, minor, strata::qwen_defaults_026(), main_vram_bytes)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
        if (kv.first == "STRATA_ADAPT_FETCH") fetch_set = true;
    }
    adapt_fetch_default_flag() = fetch_set;
    if (!set.empty()) {
        const bool sm86 = mj == 8;
        const bool sm121 = mj == 12 && mn == 1;
        // sm_86 / sm_60: bit-exact switches.  sm_121: the fused prompt path and the FP4-activation NVFP4 prompt round differently from the
        // paths they replace (24 of 24 greedy answers were identical on the Spark; no KL run), so the line does not say "exact".
        const bool sm120 = mj == 12 && mn == 0;
        // ADAPT_FETCH (a card of 16 GiB or less) is not bit-exact either: the line then says so and names its own way back
        std::fprintf(stderr, "strata: %s: %zu %sspeed switches on by default (%s=0 turns them off%s; a switch you set is kept%s): ",
                     sm86 ? "sm_86 (Ampere)" : sm121 ? "sm_121 (GB10)" : sm120 ? "sm_120 (Blackwell)" : "sm_60 (Pascal)", set.size(), sm121 || fetch_set ? "" : "exact ",
                     sm86 ? "STRATA_SM86_PERF" : sm121 ? "STRATA_SM121_PERF" : sm120 ? "STRATA_SM120_PERF" : "STRATA_SM60_PERF",
                     sm121 || sm120 ? ", as does STRATA_DEFAULTS=026" : "",
                     fetch_set ? "; ADAPT_FETCH is on because this card has 16 GiB of memory or less, it does not give bit-for-bit the same text, STRATA_ADAPT_FETCH=0 turns it off" : "");
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    return set;
}

std::vector<std::pair<std::string, std::string>> nv_kq_default_env(int min_major, int min_minor) {
    return nv_kq_default_env(min_major, min_minor, strata::qwen_defaults_026());
}

std::vector<std::pair<std::string, std::string>> nv_kq_default_env(int min_major, int min_minor, bool defaults026) {
    std::vector<std::pair<std::string, std::string>> t;
    if (defaults026) return t;
    if (strata::cc_major_of(min_major) < 8) return t;   // Volta / Turing / Pascal: not measured, keep the FP16 prompt path
    const char* sw = std::getenv("STRATA_NV_KQ_DEFAULT");
    if (sw != nullptr && sw[0] == '0') return t;
    if (!kNvKqDefaultOn && !(sw != nullptr && sw[0] == '1')) return t;
    t.push_back({"STRATA_MMQ_KQ", "1"});
    (void) min_minor;
    return t;
}

bool nv_kq_fused_here(int major, int minor, int sms) {
    namespace ft = strata::prefill::fused_tile;
    if (ft::pf_env(std::getenv("STRATA_PF_FUSED")) == 0) return false;
    const ft::Card c{strata::sm120_default(major, minor), major == 12 && minor == 1, sms, ft::s86_class(major, minor, off("STRATA_PF_FUSED_SM86"))};
    return ft::kq_takes(ft::pf_env(std::getenv("STRATA_PF_FUSED_KQ")), c, ft::sm40_env_of(std::getenv("STRATA_PF_FUSED_SM40")));
}

std::string nv_kq_default_line(bool fused_here) {
    return nv_kq_default_line(fused_here, true);
}

std::string nv_kq_default_line(bool fused_here, bool mmq_built) {
    if (!mmq_built) {   // a build without MMQ: Strata's own kernels read every K-quant layer
        if (fused_here)
            return "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on Strata's own kernels "
                   "(Q4_K and Q5_K gate/up with a Q5_1 or Q8_0 down); the 'prompt kernels' line below gives the counts";
        return "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on Strata's own int8 kernels (this engine has no MMQ kernels)";
    }
    if (fused_here)
        return "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on Strata's own kernels where they cover the layer "
               "(Q4_K and Q5_K gate/up with a Q5_1 or Q8_0 down) and on llama.cpp's MMQ kernels for the rest (STRATA_PF_FUSED_KQ=0 gives all of them back to MMQ)";
    return "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on llama.cpp's MMQ kernels "
           "(STRATA_MMQ_KQ=0 or STRATA_NV_KQ_DEFAULT=0 turns it off)";
}

bool nv_kq_expert_type(int ggml_type) {
    // ggml type ids: Q5_1 = 7, Q4_K = 12, Q5_K = 13, Q6_K = 14 (the cases of prefill::mmq::supported() behind kq_ok())
    return ggml_type == 7 || ggml_type == 12 || ggml_type == 13 || ggml_type == 14;
}

bool nv_kq_line_wanted(const std::vector<std::pair<int, int>>& layer_gate_up_down_types) {
    for (const auto& l : layer_gate_up_down_types)
        if (nv_kq_expert_type(l.first) || nv_kq_expert_type(l.second)) return true;
    return false;
}

std::vector<std::string> apply_nv_kq_default(int min_major, int min_minor, bool fused_here, bool say) {
    std::vector<std::string> set;
    for (const auto& kv : nv_kq_default_env(min_major, min_minor)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
    }
    if (say && !set.empty())   // the engine passes say = false and prints the line after the expert layout is loaded, only for a set with K-quant experts (nv_kq_line_wanted)
        std::fprintf(stderr, "%s\n", nv_kq_default_line(fused_here).c_str());
    return set;
}

void apply_spec_adapt_default(int major, int minor) {
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    // sm_86 only: the P100 (sm_60) gained 4-16% on JSON/code/chat/tool calls but lost 3% on the 4K agent transcript (5/5 pairs), so there it stays opt-in
    // Owner round 9 (2026-10-22): on the z440 A4000s the 5-pair A/B against 0.2.2 was 0.995 (code 0.996, json 1.019, tool 0.973,
    // agent 0.991) and the 3060 was length-confounded, so the earlier +9..17% did not reproduce: opt-in (STRATA_SPEC_ADAPT_SM86=1)
    // until a clean A/B shows it.
    const bool sm86 = mj == 8 && mn == 6;
    if (!sm86) return;
    if (const char* v = std::getenv("STRATA_SPEC_ADAPT_SM86"); v == nullptr || v[0] != '1') return;
    if (off("STRATA_SM86_PERF")) return;
    const std::pair<const char*, const char*> kv[2] = {{"STRATA_SPEC_ADAPT_EVERY", "2"}, {"STRATA_SPEC_ADAPT_WARM", "200"}};
    for (const auto& p : kv) {
        if (std::getenv(p.first) != nullptr) continue;
#if defined(_WIN32)
        _putenv_s(p.first, p.second);
#else
        setenv(p.first, p.second, 0);
#endif
    }
}

}  // namespace strata::core
