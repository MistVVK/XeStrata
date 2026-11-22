// src/core/nv_defaults_test.cpp - the NVIDIA per-architecture default tables (no GPU needed): the sm_121 (GB10, DGX Spark) table, its
// two off switches, and that no other card's table changed.
#include "strata/core/emulate.hpp"
#include "strata/core/nv_chunk.hpp"
#include "strata/core/nv_defaults.hpp"
#include "strata/prefill/nvfp4_prefill_mode.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static const std::string* find(const std::vector<std::pair<std::string, std::string>>& t, const char* k) {
    for (const auto& kv : t)
        if (kv.first == k) return &kv.second;
    return nullptr;
}

int main() {
    using namespace strata::core;
    unset("STRATA_SM121_PERF");
    unset("STRATA_SM86_PERF");
    unset("STRATA_SM70_PERF");
    unset("STRATA_SM60_PERF");
    unset("STRATA_EMULATE_CC");
    // GB10 (12.1): the five measured own-stack switches + 0.2.19 bit-exact pipeline/PLE/quant + native sm_121a mxf4nvf4
    {
        const auto t = nv_default_env(12, 1, false);
        CHECK(t.size() == 11);
        CHECK(find(t, "STRATA_PF_FUSED") && *find(t, "STRATA_PF_FUSED") == "1");
        CHECK(find(t, "STRATA_PF_FUSED_KQ") && *find(t, "STRATA_PF_FUSED_KQ") == "1");
        CHECK(find(t, "STRATA_PREFILL_NVFP4") && *find(t, "STRATA_PREFILL_NVFP4") == "w4a4");
        CHECK(find(t, "STRATA_HC_UPMIX") && *find(t, "STRATA_HC_UPMIX") == "1");
        CHECK(find(t, "STRATA_SIDE_MIXER") && *find(t, "STRATA_SIDE_MIXER") == "2");
        CHECK(find(t, "STRATA_SPEC_COST") && *find(t, "STRATA_SPEC_COST") == "1");
        CHECK(find(t, "STRATA_DEQUANT_SIDE") && *find(t, "STRATA_DEQUANT_SIDE") == "1");
        CHECK(find(t, "STRATA_QFUSE") && *find(t, "STRATA_QFUSE") == "1");
        CHECK(find(t, "STRATA_LFUSE") && *find(t, "STRATA_LFUSE") == "1");
        CHECK(find(t, "STRATA_PLE_COMPACT") && *find(t, "STRATA_PLE_COMPACT") == "1");
        CHECK(find(t, "STRATA_PLE_COPY_STREAM") && *find(t, "STRATA_PLE_COPY_STREAM") == "1");
        CHECK(!find(t, "STRATA_HC_Q8"));
        const auto t128 = nv_default_env(12, 1, false, 128ull << 30);
        CHECK(find(t128, "STRATA_VERIFY_DEVICE_HITS") && *find(t128, "STRATA_VERIFY_DEVICE_HITS") == "1");
        CHECK(!find(t128, "STRATA_ADAPT_FETCH"));
    }
    // STRATA_DEFAULTS=026 (the 0.2.7 master switch; the sm_121 table is newer than 0.2.6) empties it
    CHECK(nv_default_env(12, 1, true).empty());
    // STRATA_SM121_PERF=0 empties it; any other value keeps it
    put("STRATA_SM121_PERF", "0");
    CHECK(nv_default_env(12, 1, false).empty());
    put("STRATA_SM121_PERF", "1");
    CHECK(nv_default_env(12, 1, false).size() == 11);
    unset("STRATA_SM121_PERF");
    // STRATA_PREFILL_NVFP4 resolution in builds without MMQ: w4a4 routes to fp4_tc where available and degrades to w4a8 (never FP16 cuBLAS) elsewhere
    {
        using strata::prefill::Nvfp4PrefillPath;
        using strata::prefill::nvfp4_fused_covers;
        using strata::prefill::resolve_nvfp4_prefill_path;
        bool deg = true;
        CHECK(resolve_nvfp4_prefill_path(nullptr, false, false, &deg) == Nvfp4PrefillPath::FusedW4A8 && !deg);
        CHECK(resolve_nvfp4_prefill_path("", true, false, &deg) == Nvfp4PrefillPath::FusedW4A8 && !deg);
        CHECK(resolve_nvfp4_prefill_path("w4a8", true, false, &deg) == Nvfp4PrefillPath::FusedW4A8 && !deg);
        CHECK(resolve_nvfp4_prefill_path("w4a4", true, false, &deg) == Nvfp4PrefillPath::FusedW4A4 && !deg);
        CHECK(resolve_nvfp4_prefill_path("w4a4", false, false, &deg) == Nvfp4PrefillPath::FusedW4A8 && deg);
        CHECK(resolve_nvfp4_prefill_path("w4a4", false, true, &deg) == Nvfp4PrefillPath::LegacyMmqOrF16 && !deg);
        CHECK(resolve_nvfp4_prefill_path("fp16", true, false, &deg) == Nvfp4PrefillPath::LegacyMmqOrF16 && !deg);
        CHECK(nvfp4_fused_covers("w4a4", true, false));
        CHECK(nvfp4_fused_covers("w4a4", false, false));
        CHECK(!nvfp4_fused_covers("fp16", true, false));
    }
    // sm_120 (RTX 50 / PRO Blackwell): BW-13/BW-G (minus opt-in L2_PF per #6431) + 0.2.19 exact overlap/PLE/quant fusions, fused K-quant prefill, wide CPU share, native mxf4nvf4; STRATA_DEFAULTS=026 and STRATA_SM120_PERF=0 leave it out
    {
        unset("STRATA_SM120_PERF");
        const auto t = nv_default_env(12, 0, false);
        CHECK(t.size() == 12 && find(t, "STRATA_HC_UPMIX") && *find(t, "STRATA_HC_UPMIX") == "1");
        CHECK(!find(t, "STRATA_L2_PF"));                                                   // #6431: opt-in on sm_120
        CHECK(find(t, "STRATA_Q80_ROWS1") && *find(t, "STRATA_Q80_ROWS1") == "1");
        CHECK(find(t, "STRATA_EXP_MODE") && *find(t, "STRATA_EXP_MODE") == "8");
        CHECK(find(t, "STRATA_SIDE_MIXER") && *find(t, "STRATA_SIDE_MIXER") == "2");
        CHECK(find(t, "STRATA_DEQUANT_SIDE") && *find(t, "STRATA_DEQUANT_SIDE") == "1");
        CHECK(find(t, "STRATA_QFUSE") && *find(t, "STRATA_QFUSE") == "1");
        CHECK(find(t, "STRATA_LFUSE") && *find(t, "STRATA_LFUSE") == "1");
        CHECK(find(t, "STRATA_PLE_COMPACT") && *find(t, "STRATA_PLE_COMPACT") == "1");
        CHECK(find(t, "STRATA_PLE_COPY_STREAM") && *find(t, "STRATA_PLE_COPY_STREAM") == "1");
        CHECK(find(t, "STRATA_PF_FUSED_KQ") && *find(t, "STRATA_PF_FUSED_KQ") == "1");
        CHECK(!find(t, "STRATA_HC_Q8"));
        CHECK(find(t, "STRATA_PREFILL_CPU_SHARE_NT") && *find(t, "STRATA_PREFILL_CPU_SHARE_NT") == "32");
        CHECK(find(t, "STRATA_PREFILL_NVFP4") && *find(t, "STRATA_PREFILL_NVFP4") == "w4a4");
        CHECK(!find(t, "STRATA_VERIFY_DEVICE_HITS") && !find(t, "STRATA_ADAPT_FETCH"));
        // <= 16 GiB gets ADAPT_FETCH=2; > 16 GiB (RTX 5090 32 GB / RTX PRO 5000) gets VERIFY_DEVICE_HITS=1
        const auto t16 = nv_default_env(12, 0, false, 16ull << 30);
        CHECK(find(t16, "STRATA_ADAPT_FETCH") && *find(t16, "STRATA_ADAPT_FETCH") == "2");
        CHECK(!find(t16, "STRATA_VERIFY_DEVICE_HITS"));
        const auto t32 = nv_default_env(12, 0, false, 32ull << 30);
        CHECK(find(t32, "STRATA_VERIFY_DEVICE_HITS") && *find(t32, "STRATA_VERIFY_DEVICE_HITS") == "1");
        CHECK(!find(t32, "STRATA_ADAPT_FETCH"));
        CHECK(nv_default_env(12, 0, true).empty());
        put("STRATA_SM120_PERF", "0");
        CHECK(nv_default_env(12, 0, false).empty());
        put("STRATA_SM120_PERF", "1");
        CHECK(nv_default_env(12, 0, false).size() == 12);
        unset("STRATA_SM120_PERF");
    }
    // sm_89 (RTX 40 / Ada Lovelace): 17 switches (#6462 bit-exact + 0.2.19; L2_PF opt-in per #6431, QFUSE opt-in per #6583) + VERIFY_DEVICE_HITS for > 16 GiB (RTX 4090 24 GB)
    {
        unset("STRATA_SM89_PERF");
        const auto t = nv_default_env(8, 9, false);
        CHECK(t.size() == 17 && find(t, "STRATA_HC_UPMIX") && *find(t, "STRATA_HC_UPMIX") == "1");
        CHECK(find(t, "STRATA_ADAPT_LAG") && *find(t, "STRATA_ADAPT_LAG") == "2");
        CHECK(find(t, "STRATA_ATTN_DAILY") && *find(t, "STRATA_ATTN_DAILY") == "1");
        CHECK(find(t, "STRATA_PREFILL_CPU_SHARE_MAX") && *find(t, "STRATA_PREFILL_CPU_SHARE_MAX") == "6144");
        CHECK(find(t, "STRATA_PF_FUSED_KQ") && *find(t, "STRATA_PF_FUSED_KQ") == "1");
        CHECK(!find(t, "STRATA_L2_PF"));    // #6431: opt-in on sm_89
        CHECK(!find(t, "STRATA_QFUSE"));    // #6583: opt-in alongside GR_UP75 on sm_89
        CHECK(!find(t, "STRATA_HC_Q8"));
        CHECK(!find(t, "STRATA_VERIFY_DEVICE_HITS"));
        const auto t12 = nv_default_env(8, 9, false, 12ull << 30);
        CHECK(t12.size() == 18 && find(t12, "STRATA_ADAPT_FETCH") && *find(t12, "STRATA_ADAPT_FETCH") == "2");
        CHECK(!find(t12, "STRATA_VERIFY_DEVICE_HITS"));
        const auto t24 = nv_default_env(8, 9, false, 24ull << 30);
        CHECK(t24.size() == 18 && find(t24, "STRATA_VERIFY_DEVICE_HITS") && *find(t24, "STRATA_VERIFY_DEVICE_HITS") == "1");
        CHECK(!find(t24, "STRATA_ADAPT_FETCH"));
        CHECK(nv_default_env(8, 9, true).empty());
        put("STRATA_SM89_PERF", "0");
        CHECK(nv_default_env(8, 9, false).empty());
        unset("STRATA_SM89_PERF");
    }
    // sm_70 (Volta): 21 validated bit-exact switches; STRATA_DEFAULTS=026 and STRATA_SM70_PERF=0 leave it out
    {
        unset("STRATA_SM70_PERF");
        const auto t = nv_default_env(7, 0, false);
        CHECK(t.size() == 21);
        CHECK(find(t, "STRATA_ADAPT_OVERLAP") && *find(t, "STRATA_ADAPT_OVERLAP") == "1");
        CHECK(find(t, "STRATA_SM70_TABLE") && *find(t, "STRATA_SM70_TABLE") == "1");
        CHECK(find(t, "STRATA_SELECT_LONG") && *find(t, "STRATA_SELECT_LONG") == "1");
        CHECK(nv_default_env(7, 0, true).empty());
        put("STRATA_SM70_PERF", "0");
        CHECK(nv_default_env(7, 0, false).empty());
        put("STRATA_SM70_PERF", "1");
        CHECK(nv_default_env(7, 0, false).size() == 21);
        unset("STRATA_SM70_PERF");
    }
    // nbald/volta-forever remapping: 7.5 -> 7.0 when active, other architectures untouched
    CHECK(strata::cc_of(7, 5, false) == 75);
    CHECK(strata::cc_of(7, 5, true) == 70);
    CHECK(strata::cc_of(7, 0, true) == 70);
    CHECK(strata::cc_of(8, 6, true) == 86);
    CHECK(strata::cc_of(12, 0, true) == 120);
    // sm_100 and the older cards get nothing from this file; the sm_86 / sm_60 tables are unchanged by the flag
    CHECK(nv_default_env(10, 0, false).empty());
    CHECK(nv_default_env(7, 5, false).empty());
    {
        const auto a = nv_default_env(8, 6, false), b = nv_default_env(8, 6, true);
        CHECK(!a.empty() && a == b);               // STRATA_DEFAULTS=026 does not touch the 2026-10 sm_86 table
        const auto p = nv_default_env(6, 0, false), q = nv_default_env(6, 0, true);
        CHECK(!p.empty() && p == q);
        CHECK(!find(a, "STRATA_PF_FUSED") && !find(p, "STRATA_PF_FUSED"));
        // repeatable output: sm_86 keeps ADAPT_LAG=2 but no longer turns ADAPT_NOWAIT on (option 2); sm_60 keeps both
        CHECK(!find(a, "STRATA_ADAPT_NOWAIT"));
        CHECK(find(a, "STRATA_ADAPT_LAG") && *find(a, "STRATA_ADAPT_LAG") == "2");
        CHECK(find(a, "STRATA_GR_UP75") && *find(a, "STRATA_GR_UP75") == "1");
        CHECK(!find(a, "STRATA_QFUSE"));   // #6583: opt-in on sm_86 (conflicts with GR_UP75 + 1-token self_commit)
        CHECK(find(p, "STRATA_ADAPT_NOWAIT") && *find(p, "STRATA_ADAPT_NOWAIT") == "1");
        CHECK(find(p, "STRATA_ADAPT_LAG") && *find(p, "STRATA_ADAPT_LAG") == "2");
        // sm_86 <= 13 GiB (RTX 3060/3070/3080) gets ADAPT_FETCH=2 (not under 026); 16 GiB (A4000) gets neither; > 16 GiB (3090/A6000) gets VERIFY_DEVICE_HITS=1
        const auto a12 = nv_default_env(8, 6, false, 12ull << 30);
        const auto a16 = nv_default_env(8, 6, false, 16ull << 30);
        const auto a24 = nv_default_env(8, 6, false, 24ull << 30);
        CHECK(find(a12, "STRATA_ADAPT_FETCH") && *find(a12, "STRATA_ADAPT_FETCH") == "2");
        CHECK(!find(nv_default_env(8, 6, true, 12ull << 30), "STRATA_ADAPT_FETCH"));
        CHECK(!find(a16, "STRATA_VERIFY_DEVICE_HITS") && !find(a16, "STRATA_ADAPT_FETCH"));
        CHECK(find(a24, "STRATA_VERIFY_DEVICE_HITS") && *find(a24, "STRATA_VERIFY_DEVICE_HITS") == "1");
    }
    // round 0212: STRATA_MMQ_KQ=1 is a default for compute capability 8.0+ (owner yes 11-03), with its own off switches
    unset("STRATA_NV_KQ_DEFAULT");
    CHECK(kNvKqDefaultOn);
    for (int major : {8, 9, 10, 12}) CHECK(nv_kq_default_env(major, 0, false).size() == 1);   // on without any switch
    put("STRATA_NV_KQ_DEFAULT", "1");
    for (int major : {8, 9, 10, 12}) {
        const auto k = nv_kq_default_env(major, 0, false);
        CHECK(k.size() == 1 && find(k, "STRATA_MMQ_KQ") && *find(k, "STRATA_MMQ_KQ") == "1");
    }
    CHECK(nv_kq_default_env(7, 5, false).empty());     // Turing / Volta / Pascal keep the FP16 prompt path
    CHECK(nv_kq_default_env(6, 0, false).empty());
    CHECK(nv_kq_default_env(12, 0, true).empty());     // STRATA_DEFAULTS=026
    put("STRATA_NV_KQ_DEFAULT", "0");
    CHECK(nv_kq_default_env(12, 0, false).empty());    // its own off switch
    unset("STRATA_NV_KQ_DEFAULT");
    // the start-up line of that default: the earlier line byte for byte where Strata's own kernels do not read the K-quant experts, one that says what happens where they do
    CHECK(nv_kq_default_line(false) ==
          "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on llama.cpp's MMQ kernels (STRATA_MMQ_KQ=0 or STRATA_NV_KQ_DEFAULT=0 turns it off)");
    CHECK(nv_kq_default_line(true) ==
          "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on Strata's own kernels where they cover the layer (Q4_K and Q5_K gate/up with a Q5_1 or Q8_0 down) "
          "and on llama.cpp's MMQ kernels for the rest (STRATA_PF_FUSED_KQ=0 gives all of them back to MMQ)");
    CHECK(nv_kq_default_line(true, true) == nv_kq_default_line(true) && nv_kq_default_line(false, true) == nv_kq_default_line(false));
    // a build without MMQ names no MMQ kernel it does not have
    CHECK(nv_kq_default_line(true, false) ==
          "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on Strata's own kernels (Q4_K and Q5_K gate/up with a Q5_1 or Q8_0 down); "
          "the 'prompt kernels' line below gives the counts");
    CHECK(nv_kq_default_line(false, false) ==
          "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on Strata's own int8 kernels (this engine has no MMQ kernels)");
    {   // the line is printed only for a loaded set that has K-quant experts (ggml types: IQ2_XS 17, IQ3_S 21, IQ4_NL 20, IQ2_S 22, Q8_0 8, Q5_1 7, Q4_K 12, Q5_K 13, Q6_K 14)
        CHECK(nv_kq_expert_type(12) && nv_kq_expert_type(13) && nv_kq_expert_type(14) && nv_kq_expert_type(7));
        CHECK(!nv_kq_expert_type(8) && !nv_kq_expert_type(17) && !nv_kq_expert_type(21) && !nv_kq_expert_type(20) && !nv_kq_expert_type(22) && !nv_kq_expert_type(-1) && !nv_kq_expert_type(0));
        CHECK(!nv_kq_line_wanted({}));                                                                   // nothing loaded
        CHECK(!nv_kq_line_wanted({{21, 20}, {21, 20}, {22, 20}, {17, 20}}));                              // IQ3_S / IQ2_S / IQ2_XS with IQ4_NL down: no line
        CHECK(!nv_kq_line_wanted({{21, 8}, {17, 8}}));                                                   // Q8_0 down only: the switch decides nothing there
        CHECK(nv_kq_line_wanted({{12, 7}}));                                                             // UD-Q4_K_XL: Q4_K gate/up, Q5_1 down
        CHECK(nv_kq_line_wanted({{21, 20}, {13, 8}, {21, 20}}));                                         // one Q5_K layer among i-quant ones is enough
        CHECK(nv_kq_line_wanted({{12, 14}}));                                                            // Q6_K down
        CHECK(nv_kq_line_wanted({{21, 7}}));                                                             // only a Q5_1 down (i-quant gate/up): the switch decides that product, so the line stands
        CHECK(nv_kq_line_wanted({{21, 20}, {21, 7}}));
        // apply_nv_kq_default is not called here: its two-argument path reads strata::qwen_defaults_026(), which caches STRATA_DEFAULTS at its first call,
        // and the one-argument check at the end of main needs that first call to see STRATA_DEFAULTS=026.
    }
    {   // which cards: fused_tile::kq_takes with the engine's switches (raw compute capability, SM count)
        unset("STRATA_PF_FUSED"); unset("STRATA_PF_FUSED_KQ"); unset("STRATA_PF_FUSED_SM40"); unset("STRATA_PF_FUSED_SM86"); 
        CHECK(nv_kq_fused_here(8, 6, 84) && nv_kq_fused_here(8, 6, 80) && nv_kq_fused_here(12, 0, 110) && nv_kq_fused_here(12, 0, 80));   // RTX A6000, RTX PRO 5000 class
        CHECK(nv_kq_fused_here(8, 0, 108) && nv_kq_fused_here(8, 9, 128));                                                                  // the sm_86 class
        CHECK(nv_kq_fused_here(8, 6, 48) && nv_kq_fused_here(12, 0, 48));                                                                   // 40-79 SMs: the staged launches
        CHECK(!nv_kq_fused_here(8, 6, 39) && !nv_kq_fused_here(12, 1, 48) && !nv_kq_fused_here(7, 5, 68) && !nv_kq_fused_here(8, 7, 16) && !nv_kq_fused_here(9, 0, 132));   // under 40 SMs, GB10, Turing, Orin, Hopper
        put("STRATA_PF_FUSED_SM40", "0");
        CHECK(!nv_kq_fused_here(8, 6, 48) && nv_kq_fused_here(8, 6, 84));
        unset("STRATA_PF_FUSED_SM40");
        put("STRATA_PF_FUSED_KQ", "0");
        CHECK(!nv_kq_fused_here(8, 6, 84) && !nv_kq_fused_here(12, 0, 110));                                                                // the way back
        put("STRATA_PF_FUSED_KQ", "1");
        CHECK(nv_kq_fused_here(8, 6, 84) && nv_kq_fused_here(12, 1, 48) && nv_kq_fused_here(7, 5, 68));                                      // the opt-in: every card
        put("STRATA_PF_FUSED", "0");
        CHECK(!nv_kq_fused_here(8, 6, 84) && !nv_kq_fused_here(12, 1, 48));                                                                 // STRATA_PF_FUSED=0: every fused kernel off
        unset("STRATA_PF_FUSED"); unset("STRATA_PF_FUSED_KQ");
        put("STRATA_PF_FUSED_SM86", "0");
        CHECK(!nv_kq_fused_here(8, 6, 84) && nv_kq_fused_here(12, 0, 110));                                                                 // the sm_86 class diagnostic switch
        unset("STRATA_PF_FUSED_SM86");
    }
    // the other tables are unchanged by it
    CHECK(nv_default_env(12, 0, false).size() == 12);   // sm_120 default stack (#6431: L2_PF opt-in)
    // round 0212: the NVIDIA prompt-chunk ceiling of a bare `--prefill auto` (nv_chunk.hpp)
    CHECK(nv_chunk_ceiling(nullptr, false) == 32768);
    CHECK(nv_chunk_ceiling("1", false) == 32768);
    CHECK(nv_chunk_ceiling("16", false) == 16384);
    CHECK(nv_chunk_ceiling("0", false) == 0);          // STRATA_NV_CHUNK=0 keeps 8192
    CHECK(nv_chunk_ceiling(nullptr, true) == 0);       // STRATA_DEFAULTS=026 keeps 8192
    CHECK(nv_chunk_ceiling("16", true) == 0);
    CHECK(nv_chunk_card_ok(7, 0));                                                    // Volta (V100): measured, takes the raise
    CHECK(!nv_chunk_card_ok(7, 5));                                                   // Turing: not measured, keeps 8192
    CHECK(!nv_chunk_card_ok(6, 0) && !nv_chunk_card_ok(6, 1) && !nv_chunk_card_ok(6, 2));   // Pascal: not measured, keeps 8192
    CHECK(!nv_chunk_card_ok(5, 2) && !nv_chunk_card_ok(0, 0));
    CHECK(nv_chunk_card_ok(8, 0) && nv_chunk_card_ok(8, 6) && nv_chunk_card_ok(9, 0) && nv_chunk_card_ok(12, 0));
    CHECK(nv_chunk_step_down(32768) == 16384 && nv_chunk_step_down(16384) == 8192 && nv_chunk_step_down(8192) == 8192);
    // #5199: <= 16.5 GiB cards keep bare --prefill auto at 8192 unless STRATA_NV_CHUNK is set explicitly; > 16.5 GiB keeps 32768
    CHECK(nv_chunk_ceiling_for_vram(nullptr, false, 8ull << 30) == 0);
    CHECK(nv_chunk_ceiling_for_vram(nullptr, false, 12ull << 30) == 0);
    CHECK(nv_chunk_ceiling_for_vram(nullptr, false, 16ull << 30) == 0);
    CHECK(nv_chunk_ceiling_for_vram("16", false, 16ull << 30) == 16384);
    CHECK(nv_chunk_ceiling_for_vram("1", false, 16ull << 30) == 32768);
    CHECK(nv_chunk_ceiling_for_vram(nullptr, false, 24ull << 30) == 32768);
    CHECK(nv_chunk_ceiling_for_vram(nullptr, false, 0) == 32768);
    CHECK(nv_chunk_ceiling_for_vram(nullptr, true, 24ull << 30) == 0);
    // the one-argument form reads STRATA_DEFAULTS
    put("STRATA_DEFAULTS", "026");
    CHECK(nv_default_env(12, 1).empty());
    unset("STRATA_DEFAULTS");
    if (fails == 0) std::printf("nv_defaults_test: ok\n");
    return fails == 0 ? 0 : 1;
}
