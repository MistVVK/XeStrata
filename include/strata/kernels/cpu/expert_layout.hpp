// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 hiruhiru2005
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/cpu/expert_layout.hpp - plan v0.3 P6: where each routed expert lives in experts.bin.
//
// A Q2_0 pack (tools/strata_pack.py) has one blob size for every layer, `BLOB`, in the Strata expert form.  A
// native pack (tools/iq_pack.py, the IQ2_XS / IQ3_XXS files) keeps each expert's raw GGUF slices, so the blob
// size and the formats change from layer to layer; `native_experts.txt` says how.  Everything that touches an
// expert blob - the arena, the VRAM tier, the prompt path, the CPU pool, the GPU window - asks this table.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

struct ExpertLayout {
    bool native = false;
    int64_t n_layers = 0, n_expert = NE;
    std::vector<NativeFmt> fmt;           ///< per layer (native packs)
    std::vector<uint64_t> offset, bytes;  ///< per layer: where its 512 blobs start, bytes per blob
    /// Plan v0.3 P6: per layer, the absolute offsets of the gate / up / down tensors in their GGUF files, so the
    /// arena can be filled from the GGUF itself when the pack has no experts.bin (3 x n_layers, 0 = unknown).
    std::vector<uint64_t> gguf_off;
    /// Per layer, the GGUF files (names beside the --native shard) that hold its gate / up / down tensors when
    /// the model's shards split the layers (Swift's GGUFs: layers 13-47 in shard 2, and layer 13's down tensor in
    /// shard 1) (3 x n_layers).  Empty = the --native shard itself.
    std::vector<std::string> gguf_file;
    /// Per layer of this table, the model's block that holds it (`blk.<block>.ffn_*_exps`): a model whose first blocks
    /// have no routed experts (GLM-5.3's three dense ones) numbers its MoE layers apart from its blocks.  Empty: the
    /// same numbers.
    std::vector<int64_t> block;
    int version = 0;                      ///< native_experts.txt's header version (0 = none given)
    /// v6 (the latent MoE, Edge0): the experts' own widths from the header, 0 when it does not name them.
    /// They differ from the shell's hidden width and expert FF: the experts run at the latent width.
    int64_t expert_embd = 0, expert_ff = 0;
    /// Sync's v6 has a different format and size for each expert, unlike Edge0's v6.
    std::vector<NativeFmt> expert_fmt;
    std::vector<uint64_t> expert_off;
    std::vector<uint64_t> layer_size;
    bool mixed() const { return !expert_fmt.empty(); }
    size_t expert_index(int64_t layer, int64_t expert) const { return (size_t) (layer * n_expert + expert); }
    const NativeFmt& format(int64_t layer, int64_t expert) const {
        return mixed() ? expert_fmt[expert_index(layer, expert)] : fmt[(size_t) layer];
    }
    uint64_t blob_bytes(int64_t layer, int64_t expert) const {
        return mixed() ? format(layer, expert).bytes : blob_bytes(layer);
    }
    uint64_t layer_bytes(int64_t layer) const {
        return mixed() ? layer_size[(size_t) layer] : blob_bytes(layer) * (uint64_t) n_expert;
    }
    uint64_t max_blob = BLOB;
    uint64_t total = 0;                   ///< experts.bin size

    int64_t block_of(int64_t layer) const { return block.empty() ? layer : block[(size_t) layer]; }
    uint64_t blob_bytes(int64_t layer) const { return native ? bytes[(size_t) layer] : (uint64_t) BLOB; }
    uint64_t layer_offset(int64_t layer) const {
        return native ? offset[(size_t) layer] : (uint64_t) layer * (uint64_t) n_expert * (uint64_t) BLOB;
    }
    uint64_t blob_offset(int64_t layer, int64_t expert) const {
        return mixed() ? expert_off[expert_index(layer, expert)] :
                         layer_offset(layer) + (uint64_t) expert * blob_bytes(layer);
    }
};

/// Plan v0.3 P6: whether this CPU (and its OS) runs the AVX-512 kernels (F, BW, VL, VNNI, VBMI).  Probed in a
/// file compiled without AVX-512, so asking is safe everywhere; STRATA_FORCE_AVX2=1 answers no (for tests).
bool cpu_avx512_ok();
/// Whether this CPU (and its OS) runs the AVX2 kernels (AVX, AVX2, FMA, F16C): the floor of every expert kernel
/// (q2_avx2.cpp, iq_avx2.cpp, and ggml-cpu in the portable build).  STRATA_FORCE_AVX2 does not change it.
bool cpu_avx2_ok();
/// STRATA_IQ256_GATHER, the AVX-2 i-quant kernels' gathered grid decode (iq_avx2.cpp, upstream a7336175): -1 unset
/// (auto, see cpu_gather_fast_here), 0 the scalar decode on every core, 1 the gathered one on every core.
int iq256_gather_setting();
/// Whether this CPU's performance cores gather the IQ grid entries faster than they assemble them from scalar loads,
/// from what CPUID reports: an Intel CPU with AVX-VNNI (Alder Lake / Sapphire Rapids or newer; the older Intel cores
/// gather slowly or under the Downfall microcode, AMD Zen 2/3 gather slowly) that is hybrid (CPUID 7.0:EDX[15]).
/// A CPU of one core type is not told apart by CPUID from an E-core-only part, so it keeps the scalar decode.
bool cpu_gather_fast();
/// cpu_gather_fast() and the CALLING THREAD runs on a performance core: CPUID 1Ah core type 40h (the E-cores, 20h,
/// gather slower than they assemble).  Probed once per thread: the pool pins each worker to one core.
bool cpu_gather_fast_here();
/// Whether the AVX-VNNI kernels run here: the build has them (STRATA_HAVE_AVXVNNI), the CPU has AVX2 and AVX-VNNI
/// (CPUID 7.1 EAX bit 4).  STRATA_NO_AVXVNNI=1 answers no (to compare with the AVX2 kernels).
bool cpu_avxvnni_ok();
/// Whether this CPU (and its OS) runs the int8 product's AVX-512 kernel (i8_gemm.hpp): AVX512F and AVX512_VNNI, and the
/// OS saving the opmask and ZMM state (XCR0 bits 1, 2, 5, 6, 7).  Only what that kernel uses: cpu_avx512_ok's VBMI and BW
/// would leave out the Cascade Lake servers, which have AVX-512 VNNI.  XESTRATA_FORCE_AVX2=1 answers no.
bool cpu_avx512vnni_ok();
/// AVX-512 VNNI with BW, VL, AVX2, FMA and F16C, for the latent U-row kernel.
bool cpu_avx512_latent_ok();
/// Whether this CPU (and its OS) runs AVX-VNNI-INT8 (vpdpbssd: s8 x s8; CPUID 7.1 EDX bit 4, Arrow Lake, Lunar Lake and
/// later) beside AVX2.  XESTRATA_NO_AVXVNNI=1 answers no, as for AVX-VNNI.
bool cpu_avxvnniint8_ok();
/// The bytes of one core's level-2 cache, from CPUID (leaf 4 on Intel, 0x8000001D on AMD); 0 when the CPU does not say.
uint64_t cpu_l2_bytes();
/// The CPU's brand string (CPUID 0x80000002..4), for messages; "unknown" when it has none.
std::string cpu_name();
/// Q2_0 GGUF rows / activation quantizer on the kernels this CPU has.
void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1);
void act_quant_any(const float* x, int n, ActQ& a);

/// The process-wide layout (canonical Q2_0 until `expert_layout_load` finds a native pack).
const ExpertLayout& expert_layout();
/// Reads `<pack_dir>/native_experts.txt` when it exists (a native pack), else sets the canonical layout.
/// Versions up to kExpertLayoutVersion are read; a newer one is refused (a newer packer wrote it).
bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
/// Replaces the process-wide layout (expert_layout_from_gguf builds one from a model's GGUF).
void expert_layout_set(const ExpertLayout& layout);
/// The newest native_experts.txt this engine reads.  v4 = v3 plus the per-role shard column `gate,up,down`,
/// written only when some layer's roles are in different shards (every other pack stays v3, byte for byte).
/// v5 = v4 plus `hadamard.txt` beside it: Hadamard-folded experts (hadamard.hpp), written only for such models,
/// so an engine that cannot rotate the activations refuses the pack instead of running it wrong.
/// v6 = the latent MoE's A-only pack (Edge0): the header names expert_embd / expert_ff (the experts' widths),
/// there is no hadamard.txt, and scales/l*.f32 beside the manifest hold the Q3_K shared column scales.
inline constexpr int kExpertLayoutVersion = 6;

/// The layer's Q3_K column-scale reciprocals (NativeFmt::gu_colscale) and their width, for the GPU side to
/// upload.  nullptr when the pack has no scales; *width 0 then.
const float* latent_gu_colscale(int layer, int* width);

}  // namespace strata::kernels::cpu
