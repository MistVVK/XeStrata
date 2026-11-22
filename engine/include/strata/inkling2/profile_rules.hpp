// Ported from strata-inkling2 by kuraneko1 (_ryu15_)
// https://github.com/kuraneko1/Strata (MIT License)
// Credit: kuraneko1 (_ryu15_), kuraneko1/Strata (MIT)
// include/strata/inkling2/profile_rules.hpp - two-axis (SM arch x VRAM residency regime) profile selection for strata-inkling2.
//
// Keeps V100 (sm_70, 32 GB FullVram) and RTX 4070 (sm_89, 12 GB CoreResidentHybrid @ 65K/131K & TightHybrid @ 262K)
// completely isolated in pure, GPU-free tables so tuning one card class or context tier can never silently regress another.
//
// Follows Strata's measured-only / recommend-never-force rules:
//  - Only measured SM architectures carry dedicated entries (sm_70, sm_89); unmeasured GPUs take conservative Generic defaults.
//  - Explicit user CLI flags (--chunk, --expert-vram) and env overrides (STRATA_PP_STAGE_MB) always win.
//  - Pure C++ header with zero CUDA dependency so every regime & card class is unit-tested on CPU.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::inkling2 {

enum class ArchId {
    Generic = 0,
    Sm70Volta,   // Tesla V100 (sm_70): measured R5 (819K Q8KV, DSpark ON)
    Sm89Ada,     // RTX 4070 (sm_89): measured R4 (65K/131K/262K Q8KV, DSpark OFF)
};

struct ArchProfile {
    ArchId id = ArchId::Generic;
    const char* name = "generic";
    bool force_bf16_x_f16 = false;        // Volta has no native BF16 Tensor Core -> convert X to FP16
    int full_vram_want_sp = 7;            // Split-KV arena reserve in FullVram (V100 80-SM needs total_sp=6 for 6 exact waves)
    int hybrid_want_sp_1024 = 5;          // Split-KV arena reserve in Hybrid at T<=1024 (Ada 46-SM: best_sp=4 + 1 tail)
    int hybrid_want_sp_2048 = 4;          // Split-KV arena reserve in Hybrid at T>=2048 (Ada 46-SM: best_sp=3 + 1 tail)
    int t1_full_split_mult = 8;           // attn_split_t1_full = clamp(sm_count * mult / n_head_kv, 40, kAttnSplitMax)
    int t1_swa_split_mult = 2;            // attn_split_t1_swa  = clamp(sm_count * mult / n_head_kv, 8, 24)
    int pp_stage_mb = 64;                 // Default pp_streaming ring slot size (MiB)
    int pp_stage_cnt_pref = 3;            // Preferred ring slot count in TightHybrid (3 x 64 MiB = 192 MiB)
    int pp_stage_cnt_core = 2;            // Preferred ring slot count in CoreResidentHybrid (2 x 64 MiB saves 64 MiB for experts)
    int kv_grow_reserve_mib = 112;        // Extra VRAM headroom during dynamic KV growth
    bool allow_tight_chunk_2048 = false;  // Measured win on sm_89 12GB TightHybrid (halves PCIe streaming bytes/tok)
    bool allow_core_chunk_2048 = false;   // Measured win on sm_89 12GB CoreResidentHybrid when tier1 still fits 128/128
};

inline ArchProfile select_arch_profile(int cc_major, int cc_minor) {
    if (cc_major == 7 && cc_minor == 0) {
        ArchProfile p;
        p.id = ArchId::Sm70Volta;
        p.name = "sm70";
        p.force_bf16_x_f16 = true;
        p.full_vram_want_sp = 7;
        p.hybrid_want_sp_1024 = 7;
        p.hybrid_want_sp_2048 = 4;
        p.t1_full_split_mult = 8;
        p.t1_swa_split_mult = 2;
        p.pp_stage_mb = 64;
        p.pp_stage_cnt_pref = 3;
        p.kv_grow_reserve_mib = 112;
        p.allow_tight_chunk_2048 = false;
        return p;
    }
    if (cc_major == 8 && cc_minor == 9) {
        ArchProfile p;
        p.id = ArchId::Sm89Ada;
        p.name = "sm89";
        p.force_bf16_x_f16 = false;
        p.full_vram_want_sp = 7;
        p.hybrid_want_sp_1024 = 5;
        p.hybrid_want_sp_2048 = 4;
        p.t1_full_split_mult = 8;
        p.t1_swa_split_mult = 2;
        p.pp_stage_mb = 64;
        p.pp_stage_cnt_pref = 3;
        p.pp_stage_cnt_core = 2;
        p.kv_grow_reserve_mib = 112;
        p.allow_tight_chunk_2048 = true;
        p.allow_core_chunk_2048 = true;
        return p;
    }
    ArchProfile p;
    p.id = ArchId::Generic;
    p.name = "generic";
    p.force_bf16_x_f16 = (cc_major == 7 && cc_minor < 5);
    p.full_vram_want_sp = 7;
    p.hybrid_want_sp_1024 = 5;
    p.hybrid_want_sp_2048 = 4;
    p.t1_full_split_mult = 8;
    p.t1_swa_split_mult = 2;
    p.pp_stage_mb = 64;
    p.pp_stage_cnt_pref = 3;
    p.kv_grow_reserve_mib = 112;
    p.allow_tight_chunk_2048 = false;
    return p;
}

enum class VramRegime {
    FullVram = 0,           // All routed experts + full max_ctx KV fit in VRAM (e.g. V100 32GB <= 819K)
    CoreResidentHybrid = 1, // Looped core experts (tier1=128/128) + full max_ctx KV fit (e.g. RTX 4070 @ 65K/131K)
    TightHybrid = 2,        // Even core experts must partially spill at max_ctx (e.g. RTX 4070 @ 262K)
};

inline const char* vram_regime_name(VramRegime r) {
    switch (r) {
        case VramRegime::FullVram: return "full_vram";
        case VramRegime::CoreResidentHybrid: return "core_resident_hybrid";
        case VramRegime::TightHybrid: return "tight_hybrid";
    }
    return "unknown";
}

struct ProfileInput {
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    size_t vram_total_bytes = 0;
    size_t vram_free_bytes = 0;       // Free VRAM after dense model (and optional draft weights) load
    int max_ctx = 65536;
    int requested_chunk = 0;          // 0 = auto; >0 = explicit user --chunk (never overridden)
    bool kv_q8 = true;
    bool has_draft = false;
    int n_full_layers = 8;
    int n_swa_layers = 40;
    int n_swa = 1024;
    size_t kv_bytes_per_tok = 2080;   // Per full/swa layer per token (Q8: 2*1024 + 2*8*2 = 2080 B; FP16: 4096 B)
    size_t tier1_experts_bytes = 0;   // All looped core experts (blk.6..17, 128/128)
    size_t all_experts_bytes = 0;     // All physical experts (all 24 layers, 128/128)
    bool expert_vram_forced = false;  // True when user passed --expert-vram >= 0
    size_t expert_vram_cap_bytes = 0; // Byte cap when expert_vram_forced is true
};

struct ResolvedProfile {
    ArchProfile arch{};
    VramRegime regime = VramRegime::TightHybrid;
    int chunk = 1024;
    int full_kv_init = 32768;
    int want_sp = 5;
    int pp_stage_mb = 64;
    int pp_stage_cnt_pref = 3;
    int kv_grow_reserve_mib = 112;
    bool use_d_hits = true;           // Track GPU routing counts in k_route_inkling2 (0 overhead when false)
    bool force_bf16_x_f16 = false;
    int t1_full_split_mult = 8;
    int t1_swa_split_mult = 2;
};

inline ResolvedProfile resolve_inkling2_profile(const ProfileInput& in) {
    ResolvedProfile out;
    out.arch = select_arch_profile(in.cc_major, in.cc_minor);
    out.force_bf16_x_f16 = out.arch.force_bf16_x_f16;
    out.kv_grow_reserve_mib = out.arch.kv_grow_reserve_mib;
    out.t1_full_split_mult = out.arch.t1_full_split_mult;
    out.t1_swa_split_mult = out.arch.t1_swa_split_mult;

    const size_t MiB = (size_t) 1 << 20;
    const size_t full_kv_total = (size_t) std::max(0, in.n_full_layers) * (size_t) std::max(1, in.max_ctx) * in.kv_bytes_per_tok;
    const size_t swa_kv_1024 = (size_t) std::max(0, in.n_swa_layers) * (size_t) std::max(1, in.n_swa) * in.kv_bytes_per_tok;
    // Steady engine scratch + layer_arena + conv + driver slack at T=1024 (~380 MiB + 128 MiB safety margin).
    const size_t engine_overhead_1024 = 512 * MiB;
    const size_t stage_bytes_pref = (size_t) out.arch.pp_stage_cnt_pref * ((size_t) out.arch.pp_stage_mb * MiB);

    const bool cap_allows_full = !in.expert_vram_forced || (in.expert_vram_cap_bytes >= in.all_experts_bytes);
    const bool cap_allows_core = !in.expert_vram_forced || (in.expert_vram_cap_bytes >= in.tier1_experts_bytes + stage_bytes_pref);

    const size_t full_vram_need = full_kv_total + swa_kv_1024 + engine_overhead_1024 + in.all_experts_bytes + 128 * MiB;
    const size_t core_hybrid_need = full_kv_total + swa_kv_1024 + engine_overhead_1024 + stage_bytes_pref + in.tier1_experts_bytes + 128 * MiB;

    if (in.all_experts_bytes > 0 && cap_allows_full && in.vram_free_bytes >= full_vram_need) {
        out.regime = VramRegime::FullVram;
    } else if (in.tier1_experts_bytes > 0 && cap_allows_core && in.vram_free_bytes >= core_hybrid_need) {
        out.regime = VramRegime::CoreResidentHybrid;
    } else {
        out.regime = VramRegime::TightHybrid;
    }

    // 1. Initial full-attention KV capacity:
    //    - FullVram & CoreResidentHybrid allocate max_ctx upfront (zero mid-run KV expansion pauses!).
    //    - TightHybrid starts at min(max_ctx, 32768) so short prompts keep maximum VRAM experts, then grows dynamically.
    if (out.regime == VramRegime::TightHybrid && in.max_ctx > 32768) {
        out.full_kv_init = 32768;
    } else {
        out.full_kv_init = std::max(1, in.max_ctx);
    }

    // 2. Prefill chunk size (user --chunk always wins):
    //    - FullVram keeps 1024 (no PCIe expert streaming to amortize).
    //    - CoreResidentHybrid on measured sm_89 Q8KV (no draft) promotes to 2048 when tier1 still fits 128/128
    //      with upfront full_kv_init + SWA-3072 ring + 2x64MiB staging, boosting prefill +25-35% without losing core_graph.
    //    - TightHybrid on measured sm_89 Q8KV (no draft) uses 2048 (SWA-3072 ring halves PCIe streaming bytes/tok).
    const size_t swa_kv_2048 = (size_t) std::max(0, in.n_swa_layers) * (size_t) (std::max(1, in.n_swa) + 2048) * in.kv_bytes_per_tok;
    const size_t stage_bytes_core = (size_t) out.arch.pp_stage_cnt_core * ((size_t) out.arch.pp_stage_mb * MiB);
    const size_t engine_overhead_2048 = 704 * MiB;
    const size_t core_hybrid_2048_need = full_kv_total + swa_kv_2048 + engine_overhead_2048 + stage_bytes_core + in.tier1_experts_bytes + 160 * MiB;

    if (in.requested_chunk > 0) {
        out.chunk = std::clamp(in.requested_chunk, 1, 4096);
    } else {
        const size_t init_kv_bytes = (size_t) std::max(0, in.n_full_layers) * (size_t) out.full_kv_init * in.kv_bytes_per_tok;
        if (out.regime == VramRegime::TightHybrid && out.arch.allow_tight_chunk_2048 &&
            in.kv_q8 && !in.has_draft && in.vram_free_bytes > init_kv_bytes + ((size_t) 4 << 30)) {
            out.chunk = 2048;
        } else if (out.regime == VramRegime::CoreResidentHybrid && out.arch.allow_core_chunk_2048 &&
                   in.kv_q8 && !in.has_draft && in.vram_free_bytes >= core_hybrid_2048_need) {
            out.chunk = 2048;
        } else if (in.vram_free_bytes > init_kv_bytes + ((size_t) 4 << 30)) {
            out.chunk = 1024;
        } else {
            out.chunk = 512;
        }
    }

    // 3. Split-KV arena reserve, PCIe staging ring, and GPU routing hit counters:
    if (out.chunk >= 2048) {
        out.want_sp = out.arch.hybrid_want_sp_2048;
    } else if (out.regime == VramRegime::FullVram) {
        out.want_sp = out.arch.full_vram_want_sp;
    } else {
        out.want_sp = out.arch.hybrid_want_sp_1024;
    }

    if (out.regime == VramRegime::FullVram) {
        out.pp_stage_mb = 0;
        out.pp_stage_cnt_pref = 0;
        out.use_d_hits = false;
    } else if (out.regime == VramRegime::CoreResidentHybrid) {
        out.pp_stage_mb = out.arch.pp_stage_mb;
        out.pp_stage_cnt_pref = out.arch.pp_stage_cnt_core;
        out.use_d_hits = true;
    } else {
        out.pp_stage_mb = out.arch.pp_stage_mb;
        out.pp_stage_cnt_pref = out.arch.pp_stage_cnt_pref;
        out.use_d_hits = true;
    }

    return out;
}

inline std::string format_profile_line(const ProfileInput& in, const ResolvedProfile& p) {
    char buf[320];
    const double vtot_gib = (double) in.vram_total_bytes / (double) (1ull << 30);
    const double vfree_gib = (double) in.vram_free_bytes / (double) (1ull << 30);
    std::snprintf(buf, sizeof buf,
                  "profile=%s+%s (cc=%d.%d sm=%d vram=%.1fGiB free=%.1fGiB ctx=%d chunk=%d kv_init=%d want_sp=%d stage=%dx%dMiB d_hits=%s)",
                  p.arch.name, vram_regime_name(p.regime), in.cc_major, in.cc_minor, in.sm_count,
                  vtot_gib, vfree_gib, in.max_ctx, p.chunk, p.full_kv_init, p.want_sp,
                  p.pp_stage_cnt_pref, p.pp_stage_mb, p.use_d_hits ? "on" : "off");
    return buf;
}

}  // namespace strata::inkling2
