// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
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
    int version = 0;                      ///< native_experts.txt's header version (0 = none given)
    uint64_t max_blob = BLOB;
    uint64_t total = 0;                   ///< experts.bin size

    uint64_t blob_bytes(int64_t layer) const { return native ? bytes[(size_t) layer] : (uint64_t) BLOB; }
    uint64_t layer_offset(int64_t layer) const {
        return native ? offset[(size_t) layer] : (uint64_t) layer * (uint64_t) n_expert * (uint64_t) BLOB;
    }
    uint64_t blob_offset(int64_t layer, int64_t expert) const {
        return layer_offset(layer) + (uint64_t) expert * blob_bytes(layer);
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
/// The newest native_experts.txt this engine reads.  v4 = v3 plus the per-role shard column `gate,up,down`,
/// written only when some layer's roles are in different shards (every other pack stays v3, byte for byte).
inline constexpr int kExpertLayoutVersion = 4;

}  // namespace strata::kernels::cpu
