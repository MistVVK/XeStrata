// The block layouts behind prefill_expert_paths_bench's synthetic expert blobs: bytes and values per block, and which bytes of a block carry
// scale bits (fp16 super-block scales and mins, quantized sub-block scales, the scale nibbles some formats keep beside their quants).
//
// The bench gives each expert its own blob by changing one byte of a shared base blob.  That byte must be a quant byte: the old fixed offsets
// 4..19 lie inside the first block on the 256-value formats, but on Q2_0 (18-byte blocks) offsets 18 and 19 are block 1's fp16 scale, so a
// few experts got scales of +-812, +-12,992 or NaN and their rows measured the scale, not the path (the FP16 column's SwiGLU saturates at
// 65,504 and the reference does not).  perturb_offset() walks the bytes from offset 4 and skips every byte this table marks: a format whose
// bytes 4..19 hold no scale bits keeps the blobs of earlier runs.
//
// Header-only and free of ggml (tests/prefill/bench_blob_layout_test.cpp checks it on the CPU); the bench checks bytes and values per block
// against ggml_type_size / ggml_blck_size before it uses an entry.  The layouts are those of third_party/strata-ggml/ggml/src/ggml-common.h.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::bench_blob {

struct Span {
    int begin, end;   // [begin, end) bytes of one block
};

struct BlockLayout {
    int type = -1;             // ggml_type id
    int bytes = 0;             // bytes per block
    int values = 0;            // values per block
    std::vector<Span> scale;   // the bytes that carry scale bits
};

/// The layout of ggml type `t`; false for a type this table does not know (the bench then refuses the pair).
inline bool layout_of(int t, BlockLayout& L) {
    L = BlockLayout();
    L.type = t;
    switch (t) {
        case 2: L.bytes = 18; L.values = 32; L.scale = {{0, 2}}; break;               // Q4_0: d, qs
        case 3: L.bytes = 20; L.values = 32; L.scale = {{0, 4}}; break;               // Q4_1: d, m, qs
        case 6: L.bytes = 22; L.values = 32; L.scale = {{0, 2}}; break;               // Q5_0: d, qh[4], qs
        case 7: L.bytes = 24; L.values = 32; L.scale = {{0, 4}}; break;               // Q5_1: d, m, qh[4], qs
        case 8: L.bytes = 34; L.values = 32; L.scale = {{0, 2}}; break;               // Q8_0: d, qs
        case 10: L.bytes = 84; L.values = 256; L.scale = {{0, 16}, {80, 84}}; break;  // Q2_K: scales[16], qs[64], d, dmin
        case 11: L.bytes = 110; L.values = 256; L.scale = {{96, 110}}; break;         // Q3_K: hmask[32], qs[64], scales[12], d
        case 12: L.bytes = 144; L.values = 256; L.scale = {{0, 16}}; break;           // Q4_K: d, dmin, scales[12], qs
        case 13: L.bytes = 176; L.values = 256; L.scale = {{0, 16}}; break;           // Q5_K: d, dmin, scales[12], qh[32], qs
        case 14: L.bytes = 210; L.values = 256; L.scale = {{192, 210}}; break;        // Q6_K: ql[128], qh[64], scales[16], d
        case 16:                                                                       // IQ2_XXS: d, then 8 groups of 8 bytes whose last byte's top nibble is the group's scale
            L.bytes = 66; L.values = 256; L.scale = {{0, 2}};
            for (int g = 0; g < 8; ++g) L.scale.push_back({2 + 8 * g + 7, 2 + 8 * g + 8});
            break;
        case 17: L.bytes = 74; L.values = 256; L.scale = {{0, 2}, {66, 74}}; break;   // IQ2_XS: d, qs[32] (uint16), scales[8]
        case 18: L.bytes = 98; L.values = 256; L.scale = {{0, 2}, {66, 98}}; break;   // IQ3_XXS: d, grid indices[64], scales and signs[32]
        case 19: L.bytes = 50; L.values = 256; L.scale = {{0, 2}, {34, 50}}; break;   // IQ1_S: d, qs[32], qh[8] (uint16 with the 3-bit scales)
        case 20: L.bytes = 18; L.values = 32; L.scale = {{0, 2}}; break;              // IQ4_NL: d, qs
        case 21: L.bytes = 110; L.values = 256; L.scale = {{0, 2}, {106, 110}}; break; // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
        case 22: L.bytes = 82; L.values = 256; L.scale = {{0, 2}, {74, 82}}; break;   // IQ2_S: d, qs[64], qh[8], scales[8]
        case 23: L.bytes = 136; L.values = 256; L.scale = {{0, 8}}; break;            // IQ4_XS: d, scales_h, scales_l[4], qs
        case 29: L.bytes = 56; L.values = 256; L.scale = {{48, 56}}; break;           // IQ1_M: qs[32], qh[16], scales[8] (the fp16 scale in their top nibbles)
        case 40: L.bytes = 36; L.values = 64; L.scale = {{0, 4}}; break;              // NVFP4: d[4] (E4M3), qs[32]
        case 42: L.bytes = 18; L.values = 64; L.scale = {{0, 2}}; break;              // Q2_0: d, qs[16]
        default: return false;
    }
    return true;
}

/// Does byte `b` of a block (0 <= b < L.bytes) carry scale bits?
inline bool is_scale(const BlockLayout& L, int b) {
    for (const Span& s : L.scale)
        if (b >= s.begin && b < s.end) return true;
    return false;
}

/// The blob offset expert `e` changes: the (e % 16)-th byte from offset 4 on that carries no scale bits (the gate matrix's first row is a run of
/// whole blocks from offset 0, so a byte's place in its block is the offset modulo the block size).
inline size_t perturb_offset(const BlockLayout& L, int e) {
    int k = e % 16;
    for (size_t o = 4;; ++o) {
        if (is_scale(L, (int) (o % (size_t) L.bytes))) continue;
        if (k == 0) return o;
        --k;
    }
}

}  // namespace strata::bench_blob
