// src/core/arch_defaults.cpp - see include/strata/core/arch_defaults.hpp.
#include "strata/core/arch_defaults.hpp"
#include "strata/core/defaults_026.hpp"
#include "strata/core/amd_community.hpp"
#include "strata/core/gfx12_extra.hpp"

#include "strata/kernels/gfx_arch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch) {
    return arch_default_env(gcn_arch, strata::qwen_defaults_026());
}

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch, bool defaults026) {
#if defined(_WIN32)
    return arch_default_env(gcn_arch, defaults026, true);
#else
    return arch_default_env(gcn_arch, defaults026, false);
#endif
}

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch, bool defaults026, bool windows) {
    std::vector<std::pair<std::string, std::string>> t;
    if (strata::kernels::gfx_arch_is_gfx12(gcn_arch)) {
        // gfx1200 / gfx1201 (RDNA4): the PROMPT half of the gfx1151 table (reported exact on an R9700 in #1478, re-checked
        // bitwise in notes-142-amd.md) plus four decode switches measured one at a time on an R9700 in 0.2.7 (p620, IQ3_S,
        // engine 0.1.51, 8K prompt, rotated pairs): ATTN_LANECELL + ATTN_MERGE_V2 +1.9%, TSUM +1.9%, EXPDEC +0.6%, GDN_SPLIT
        // 0.0%; the four together +3.5% (3/3 pairs) with the answer md5 identical to the default.  The rest of the gfx1151
        // decode half stays out (PF_GEMM/PF_FUSED are bit-changing; Q8_PACKED, SH_STREAM and QFUSE were measured on other
        // cards only, and the whole half measured slower here when it was tried as one block).
        // STRATA_HCD_EXACT is left out: it needs the gfx1151 hipBLASLt tuning table, finds none on gfx12, says so on every start
        // and does nothing (measured: with and without it identical, 0.0 to +0.1%).
        // STRATA_GDN_CONVL2 is AMD-only on purpose: it is not bitwise on CUDA 13.3 sm_120 (#1522), and this table is never
        // consulted by a CUDA build.
        const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
        if (off12 != nullptr && off12[0] == '0') return t;
        t = {
            {"STRATA_GDN_HEAD", "1"},
            {"STRATA_GDN_PP", "2"},
            {"STRATA_GDN_CONVL2", "1"},
            {"STRATA_GDN_NOY", "1"},
            {"STRATA_CVEC_FUSE", "1"},
        };
        if (defaults026) return t;   // STRATA_DEFAULTS=026: v0.2.6's table, the prompt half only
        // decode (bit-identical on the R9700 measurement above)
        t.push_back({"STRATA_ATTN_LANECELL", "1"});
        t.push_back({"STRATA_ATTN_MERGE_V2", "1"});
        t.push_back({"STRATA_TSUM", "1"});
        t.push_back({"STRATA_EXPDEC", "1"});
        // AMD-Q (p620, R9700, IQ3_S, serve, interleaved pairs): the constant-trip loads of Blackwell's BW-DENSE (bit 0 the router-shaped bf16 read,
        // bit 1 the GDN a/b projection) and two load-ordering variants of the Q8 hyper-connection read (2 rows a warp loaded together in the
        // up read, the down read's staging loads of all tokens together); each is the same arithmetic in the same order (kernel hashes equal).
        t.push_back({"STRATA_SMALL_UNR", "3"});
        t.push_back({"STRATA_GR_Q32_G", "2"});
        t.push_back({"STRATA_GR_Q8_PFS", "1"});
        return t;
    }
    if (!strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    // Exact (bitwise) on Aurora, each with its measured gain in aurora_s23.md:
    t = {
        // prompt
        {"STRATA_GDN_HEAD", "1"},        // the GDN recurrence, four lanes a column + the grid-stride norm (S23)
        {"STRATA_GDN_PP", "2"},          //   two columns per lane: 64K prompt +2.9% (stream A, round 2)
        {"STRATA_GDN_CONVL2", "1"},      // conv + q/k L2 norm in one kernel: 64K -0.3 s (stream A)
        {"STRATA_GDN_NOY", "1"},         // no dead FP32 store of the norm (stream A)
        {"STRATA_CVEC_FUSE", "1"},       // a steered layer's write, control vector and next norm in one pass (S23)
        {"STRATA_HCD_EXACT", "1"},       // the HC down GEMM in hipBLASLt solution 1176's k order: 64K +1.8% (stream S)
        {"STRATA_PF_PAD", "1"},          // padded GEMM row strides (acts with STRATA_PF_GEMM; inert without it) (S23)
        {"STRATA_TOPK_REG_MIN_BLOCKS", "4096"},   // the register top-k from 16K cells, not 28K: same ids; per 2,048 queries the ref takes 1.0 / 2.4 / 3.9 ms at 16K / 20K / 24K cells, the scan 0.87 / 0.93 / 1.00
        // decode
        {"STRATA_Q8_PACKED", "1"},       // the packed Q8_0 decode layout: +2.3% (S26)
        {"STRATA_Q6_PACKED", "1"},       // the packed Q6_K heads (S26)
        {"STRATA_MMVF_ROWS", "1"},       // bf16 multi-row GEMV, 4 rows per block (S25)
        {"STRATA_ATTN_LANECELL", "1"},   // decode attention scores, one cell per thread (S25)
        {"STRATA_EXPERT_V2", "1"},       // the grouped decode experts, IQ3_S gate/up + IQ4_NL down (S26)
        {"STRATA_TSUM", "1"},            // several warp sums as one transposed butterfly (S26)
        {"STRATA_LFUSE", "1"},           // fewer launches around the shared expert and the KV append (S26)
        {"STRATA_GDN_SPLIT", "1"},       // the GDN step over 4 blocks per head (S25/S26)
        {"STRATA_QFUSE", "1"},           // activation q8_1 images written by their producers (S26)
        {"STRATA_PLE_BATCH", "1"},       // the verify window's PLE key / value projections at once (S25)
        {"STRATA_SH_STREAM", "1"},       // the shared expert on its own stream: decode +1.8% / +6.7% (UD-Q4_K_XL) (140-m)
    };
    // 0.2.7: Windows gfx1151 leaves the shared-expert fork off (the HIP default).  Interleaved A/B on a Strix Halo Windows PC (Qwen
    // IQ3_S, engine 0.1.51, 9 chat answers each, 4K prompt): decode 43.9 / 43.6 / 43.2 tok/s with it, 48.3 / 46.0 / 46.6 without
    // (+6.9% median), prompts unchanged, the token ids identical either way (docs/STRIX_HALO_WINDOWS.md).  Linux gfx1151 keeps the
    // fork on until it is measured there.  STRATA_SH_STREAM=1 turns it back on; STRATA_DEFAULTS=026 keeps v0.2.6's table.
    if (windows && !defaults026) {
        for (auto it = t.begin(); it != t.end(); ++it) {
            if (it->first == "STRATA_SH_STREAM") { t.erase(it); break; }
        }
        // round 0215 (STRIX-PK): two load-ordering switches of the decode kernels, each the same arithmetic in the same order as without it
        // (the kernel hashes are equal; first-token logits and outputs identical in the engine).  UD-Q4_K_XL on a Strix Halo Windows PC, DFlash2
        // serve, 4 request rounds x 2 passes x 4 prompts, interleaved: window time -0.25% (25 of 32 pairs faster, the same window and token count in
        // every pair), alone +0.2% / +0.3%.  Small but consistent; Linux gfx1151 is unmeasured and keeps its table.
        t.push_back({"STRATA_SMALL_UNR", "3"});
        t.push_back({"STRATA_GR_Q8_PFS", "1"});
    }
    return t;
}

bool gfx12_prefill_share_default(const char* gcn_arch, bool defaults026) {
    if (defaults026 || !strata::kernels::gfx_arch_is_gfx12(gcn_arch)) return false;
    const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
    return !(off12 != nullptr && off12[0] == '0');
}

bool gfx12_prefill_share_default(const char* gcn_arch) {
    return gfx12_prefill_share_default(gcn_arch, strata::qwen_defaults_026());
}

bool igpu_prompt_defaults_arch(const char* gcn_arch) {
    return strata::kernels::gfx_arch_is(gcn_arch, "gfx1103") || strata::kernels::gfx_arch_is(gcn_arch, "gfx1150") ||
           strata::kernels::gfx_arch_is(gcn_arch, "gfx1151");
}

bool igpu_prompt_set_enabled(const char* gcn_arch, bool windows) {
    const char* off = std::getenv("STRATA_IGPU_DEFAULTS");
    if (off != nullptr && off[0] == '0') return false;
    const char* e = std::getenv("STRATA_IGPU_PROMPT_SET");
    if (e != nullptr && e[0] == '1') return true;
    if (e != nullptr && e[0] == '0') return false;
    // 0.2.8: Strix Halo (gfx1151) gets the matrix-core prompt set on Windows by default.  Measured on the strix box
    // (Qwen3.8-Flash-Next IQ3_S, engine 0.1.52, ids prompts of 4,095 / 8,191 tokens, interleaved pairs, one engine at a
    // time): 4K 690 -> 1,090 and 8K 785 -> 1,183 on the f-024-strix-pp branch (0.2.3 engine, the same switches as
    // environment variables), i.e. x1.58 / x1.51; the first-token KL against the accepted HC_Q8 bar was 0.0088 mean
    // (0.0228 allowed), top-1 the same on every prompt.  Linux gfx1151 stays opt-in until it is measured there; gfx1103
    // keeps its own rule below.
    if (windows && strata::kernels::gfx_arch_is(gcn_arch, "gfx1151")) return true;
    return kIgpuPromptSetDefaultOn && !windows && strata::kernels::gfx_arch_is(gcn_arch, "gfx1103");
}

bool igpu_uma_arena_default(const char* gcn_arch, bool integrated, bool windows, bool on) {
    if (!integrated) return false;
    const char* reg = std::getenv("STRATA_UMA_REGISTER");
    if (reg != nullptr && reg[0] == '1') return false;
    // 0.2.1.2: a Strix Halo (gfx1151) on Windows.  Registering the 47 GiB arena with the driver took ALL the free VRAM (91 GiB before,
    // 0 after; BIOS carve-out 64 GB + 32 GB shared): the expert cache got 0 slots and the engine exited ("the embedding buffer
    // failed"), 3 of 3 starts (notes-022-strix.md).  Unregistered, the arena is held by the working-set lock and the cache gets
    // 24,576 slots.  Independent of `on` (that is the 780M's opt-in).  STRATA_ARENA_PIN_GIB=N / 0 still registers it.
    if (windows) return strata::kernels::gfx_arch_is(gcn_arch, "gfx1151");
    return on && strata::kernels::gfx_arch_is(gcn_arch, "gfx1103");
}

bool igpu_arena_no_thp_default(const char* gcn_arch, bool integrated, bool windows) {
    if (!integrated || windows) return false;
    return strata::kernels::gfx_arch_is(gcn_arch, "gfx1103") || strata::kernels::gfx_arch_is(gcn_arch, "gfx1150") ||
           strata::kernels::gfx_arch_is(gcn_arch, "gfx1151");
}

bool igpu_kfd_settle_default(const char* gcn_arch, bool integrated, bool windows) {
    return igpu_arena_no_thp_default(gcn_arch, integrated, windows);   // the same device class
}

bool igpu_cache_drop_default(const char* gcn_arch, bool integrated, bool windows) {
    return igpu_arena_no_thp_default(gcn_arch, integrated, windows);   // the same device class
}

std::vector<std::pair<std::string, std::string>> igpu_default_env(const char* gcn_arch, bool integrated) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!integrated || !igpu_prompt_defaults_arch(gcn_arch)) return t;
#if defined(_WIN32)
    const bool win = true;
#else
    const bool win = false;
#endif
    if (!igpu_prompt_set_enabled(gcn_arch, win)) return t;
    // Measured on a Radeon 780M (gfx1103, P2006, Q2_0, notes-153-igpu.md / notes-148-amd.md): prompts 109.5 -> 214.7 tok/s
    // with the first four, 233 with HC_UPMIX, decode unchanged.  Dense projections are 63% of a prompt's GPU time there.
    t = {
        {"STRATA_HIP_WMMA", "1"},      // the prompt attention on matrix cores
        {"STRATA_PF_GEMM", "1"},       // the prompt's FP16 projections on the 128 x 256 WMMA GEMM
        {"STRATA_PA_FAST", "1"},       // the prompt attention with single FP16 q and p
        {"STRATA_SELECT_WMMA", "1"},   // the block scorer on matrix cores
        {"STRATA_HC_UPMIX", "1"},      // the hyper-connection up projection with its epilogue
    };
    // gfx1151 (Strix Halo) on Windows: SELECT_WMMA is in since 0.2.9 (x1.008 at 8K, x1.037 at 32K, x1.077 at 64K), and PF_FUSED (the
    // fused int8 gate/up prompt experts, #136) measured x1.12 there and joins the set.  The 780M and the 890M keep the
    // table above exactly as measured on them.
    if (win && strata::kernels::gfx_arch_is(gcn_arch, "gfx1151")) {
        t = {
            {"STRATA_HIP_WMMA", "1"},
            {"STRATA_PF_GEMM", "1"},
            {"STRATA_PA_FAST", "1"},
            {"STRATA_HC_UPMIX", "1"},
            {"STRATA_PF_FUSED", "1"},
            {"STRATA_SELECT_WMMA", "1"},   // 0.2.9: the block scorer on matrix cores; not bitwise (about 10 of 2,051 selected cells flip at the cutoff
                                           // for queries past ~5K tokens), x1.008 at 8K, x1.037 at 32K, x1.077 at 64K (STRIX-Q-RESULTS.md)
            // 0.2.9 (owner, 10-30: a consistent Strix Halo speed gain goes on by default): the prompt experts of every covered
            // format on the F16 matrix-core kernels (the Q8_0 down layers of the Unsloth files included).  Windows gfx1151,
            // interleaved x3, same 16 greedy ids: IQ3_S 4K 1,187 -> 1,296 and 8K 1,285 -> 1,399 tok/s (x1.09), UD-IQ4_XS 4K 1,119 ->
            // 1,221 and 8K 1,180 -> 1,270; decode unchanged.  The rounding differs from the int8 kernels (KL is information only).
            {"STRATA_PF_F16", "1"},
            // Round 0215 (STRIX-FIT): the down rows of the F16 expert kernels go to the combine as FP16 (half the bytes written and read).  Windows gfx1151,
            // interleaved 8K pairs, same engine: IQ3_S x1.022 (4 of 4 pairs), UD-IQ4_XS x1.026 (3 of 3), UD-Q4_K_XL x1.012 (3 of 4); the first-token KL
            // is 1e-4 at 8K and 0.02-0.04 on agent transcripts (information).  STRATA_PF_DM16=0 turns it off.
            {"STRATA_PF_DM16", "1"},
        };
    }
    return t;
}

static void set_default(const std::pair<std::string, std::string>& kv) {
#if defined(_WIN32)
    _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
    setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
}

static std::vector<std::pair<std::string, std::string>> parallel_stack_table() {
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

std::vector<std::pair<std::string, std::string>> parallel_default_env(const char* gcn_arch, bool windows, int batch_slots) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!windows || batch_slots < 2 || !strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    return parallel_stack_table();
}

std::vector<std::pair<std::string, std::string>> parallel_default_env_cuda(bool linux_cuda, int batch_slots) {
    if (!linux_cuda || batch_slots < 2) return {};
    return parallel_stack_table();
}

std::vector<std::string> apply_parallel_defaults_cuda(bool linux_cuda, int batch_slots) {
    std::vector<std::string> set;
    for (const auto& kv : parallel_default_env_cuda(linux_cuda, batch_slots)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, =0 included
        set_default(kv);
        set.push_back(kv.first);
    }
    return set;
}

std::vector<std::pair<std::string, std::string>> gdn_default_env_cuda(bool linux_cuda) {
    if (!linux_cuda) return {};
    return {{"STRATA_GDN_TC", "1"}, {"STRATA_GDN_CONVL2", "2"}};
}

std::vector<std::string> apply_gdn_defaults_cuda(bool linux_cuda) {
    std::vector<std::string> set;
    for (const auto& kv : gdn_default_env_cuda(linux_cuda)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, =0 included
        set_default(kv);
        set.push_back(kv.first);
    }
    return set;
}

std::vector<std::pair<std::string, std::string>> gdn_chunked_default_env(bool linux_host, bool all_gfx12) {
    if (!linux_host || !all_gfx12) return {};
    return {{"STRATA_GDN_CHUNKED", "1"}};
}

std::vector<std::string> apply_gdn_chunked_default(bool linux_host, bool all_gfx12) {
    std::vector<std::string> set;
    for (const auto& kv : gdn_chunked_default_env(linux_host, all_gfx12)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, =0 included
        set_default(kv);
        set.push_back(kv.first);
    }
    return set;
}

std::vector<std::pair<std::string, std::string>> launch_default_env(const char* gcn_arch, bool windows) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!windows || !strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    t = {{"STRATA_VERIFY_ASYNC_LAUNCH", "1"}};
    return t;
}

std::vector<std::string> apply_arch_defaults(const char* gcn_arch, bool integrated, int batch_slots) {
    std::vector<std::string> set;
    // the community-made gfx1010 / gfx1030 changes (amd_community.hpp): one line, and only on those two cards
    if (const std::string note = strata::amd_community::startup_line(gcn_arch); !note.empty())
        std::fprintf(stderr, "%s\n", note.c_str());
    for (const auto& kv : arch_default_env(gcn_arch)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
        set_default(kv);
        set.push_back(kv.first);
    }
    {   // gfx12 extras kept apart from the table (gfx12_extra.hpp); not counted in the returned list
        std::vector<std::string> extra;
        for (const auto& kv : gfx12_exact_extra_env(gcn_arch)) {
            if (std::getenv(kv.first.c_str()) != nullptr) continue;
            set_default(kv);
            extra.push_back(kv.first);
        }
        if (!extra.empty()) {
            std::fprintf(stderr, "strata: gfx12 (RDNA4): exact extra switches on by default (STRATA_GFX12_DEFAULTS=0 turns them off; a switch you set is kept): ");
            for (size_t i = 0; i < extra.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", extra[i].c_str() + 7);
            std::fprintf(stderr, "\n");
        }
    }
    {   // gfx12 defaults that change the answers (gfx12_extra.hpp): the matrix-core QSA scorer from 28,672 tokens; its own line
        const auto vc = gfx12_value_changing_env(gcn_arch);
        bool any = false, kq = false;
        for (const auto& kv : vc) {
            if (std::getenv(kv.first.c_str()) != nullptr) continue;
            set_default(kv);
            if (kv.first == "STRATA_MMQ_KQ") kq = true; else any = true;
        }
        if (kq)   // the engine says nothing more on a pack without K-quant experts: the variable is never read there
            std::fprintf(stderr, "strata: gfx12 (RDNA4): a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on llama.cpp's MMQ kernels (+10..17%% prompt speed on an R9700, "
                                 "first-token KL against the FP16 path about 0.03; STRATA_MMQ_KQ=0 or STRATA_GFX12_DEFAULTS=0 turns it off)\n");
        const char* wm = std::getenv("STRATA_SELECT_WMMA");   // the line is only true while the scorer switch is not set to 0
        if (any && !(wm != nullptr && wm[0] == '0'))
            std::fprintf(stderr, "strata: gfx12 (RDNA4): the matrix-core QSA block scorer is on once a prompt is past 28,672 tokens (+5.7%% prompt speed at 98K on an R9700, shorter prompts unchanged; "
                                 "the first-token KL against the warp kernel is 0.002 on average above 30K; STRATA_SELECT_WMMA=0 or STRATA_GFX12_DEFAULTS=0 turns it off)\n");
    }
    if (strata::kernels::gfx_arch_is_gfx12(gcn_arch) && std::getenv("STRATA_GDN_CHUNKED") == nullptr) {   // the length-gated chunked GDN scan (prefill/kernels.cu: gdn_chunked_auto)
        const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
        if (!(off12 != nullptr && off12[0] == '0'))
            std::fprintf(stderr, "strata: RDNA4 (gfx12): prompt chunks of 30,000+ tokens use the chunked GDN scan (STRATA_GDN_CHUNKED=0 turns it off)\n");
    }
    if (!set.empty()) {
        const bool g12 = strata::kernels::gfx_arch_is_gfx12(gcn_arch);
        std::fprintf(stderr, "strata: %s: %zu exact speed switches on by default (%s=0 turns "
                             "them off; a switch you set is kept): ",
                     g12 ? "gfx12 (RDNA4)" : "gfx1151 (Strix Halo)", set.size(),
                     g12 ? "STRATA_GFX12_DEFAULTS" : "STRATA_GFX1151_DEFAULTS");
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    std::vector<std::string> igpu;
    for (const auto& kv : igpu_default_env(gcn_arch, integrated)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;
        set_default(kv);
        igpu.push_back(kv.first);
    }
    if (!igpu.empty()) {
        std::fprintf(stderr, "strata: integrated GPU (%s): the matrix-core prompt set is on (1.6x to 2x faster prompts on a "
                             "Radeon 780M, the answers differ in the last bits): ", gcn_arch);
        for (size_t i = 0; i < igpu.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", igpu[i].c_str() + 7);
        std::fprintf(stderr, ".  Turn one off with NAME=0 (for example STRATA_HC_UPMIX=0), all of them with STRATA_IGPU_PROMPT_SET=0");
#if defined(_WIN32)
        std::fprintf(stderr, ".  Not yet tested on Windows");
#endif
        std::fprintf(stderr, "\n");
        set.insert(set.end(), igpu.begin(), igpu.end());
    }
    std::vector<std::string> par;
#if defined(_WIN32)
    const bool win_host = true;
#else
    const bool win_host = false;
#endif
    for (const auto& kv : parallel_default_env(gcn_arch, win_host, batch_slots)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;
        set_default(kv);
        par.push_back(kv.first);
    }
    if (!par.empty()) {
        std::fprintf(stderr, "strata: gfx1151 (Strix Halo): the exact parallel-request switches are on for %d batch slots (STRATA_GFX1151_DEFAULTS=0 turns them off; a switch you set is kept): ", batch_slots);
        for (size_t i = 0; i < par.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", par[i].c_str() + 7);
        std::fprintf(stderr, "\n");
        set.insert(set.end(), par.begin(), par.end());
    }
    std::vector<std::string> lau;
    for (const auto& kv : launch_default_env(gcn_arch, win_host)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;
        set_default(kv);
        lau.push_back(kv.first);
    }
    if (!lau.empty()) {
        std::fprintf(stderr, "strata: gfx1151 (Strix Halo): the verify window's graph is launched from a helper thread (the main thread serves the doorbells meanwhile; STRATA_GFX1151_DEFAULTS=0 or STRATA_VERIFY_ASYNC_LAUNCH=0 turns it off)\n");
        set.insert(set.end(), lau.begin(), lau.end());
    }
    return set;
}

}  // namespace strata::core
