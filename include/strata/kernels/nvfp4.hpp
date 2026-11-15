// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/nvfp4.hpp - the one reading of NVFP4 (ggml type 40) that every product of it shares: the
// GPU's decode rows (Split<40>), its prompt products (Dec<40>), the FP16 dequantizer and the CPU's rows.
//
// A block is 64 values: four UE4M3 scales (one for each 16) and 32 bytes of E2M1 codes.  Within its 16 values a
// sub-block keeps value j in the low nibble of byte j and value j + 8 in the high one (Q4_0's split halves).  ggml's
// kvalues_fp4 holds the E2M1 values doubled (0, 1, 2, 3, 4, 6, 8, 12 and their negatives), so they are exact int8 and
// the products are integer sums; the scale is then halved, which `ue4m3_half` does.
//
// ModelOpt's per-tensor weight_scale_2 has no room in the block: an expert's blob ends with {s_gate, s_up, s_down, 0}
// (kTailBytes, tools/iq_pack.py, Edge0's packs), and each scale goes on its projection's FP32 output.  s_down is not
// folded into up: the hidden would sit near 1e-5 and its q8 block scale would be an FP16 subnormal.
// The scale decode is upstream Strata's (ported there from sergqwer/strata-nvfp4).
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>

namespace strata::kernels::nvfp4 {

inline constexpr int kType = 40;          ///< ggml's GGML_TYPE_NVFP4
inline constexpr int kBlock = 64;         ///< values a block
inline constexpr int kBlockBytes = 36;    ///< 4 scale bytes + 32 code bytes
inline constexpr int kSub = 16;           ///< values a scale
inline constexpr int kTailBytes = 16;     ///< {s_gate, s_up, s_down, 0} after an NVFP4 expert's blob

/// Half a UE4M3 scale (the codes are doubled): 2^(e-8) (1 + m/8), m 2^-10 for e 0, and 0 for 0x7F (NaN), as ggml's
/// ggml_ue4m3_to_fp32 halved (the top bit is not read).  Every such value is exact in FP32, so it is built from its
/// bits: the same value for all 256 inputs on every device, with no ldexp.
constexpr float ue4m3_half(uint8_t x) {
    const uint32_t e = (x >> 3) & 0xFu, m = x & 0x7u;
    if (x == 0x7F) return 0.0f;
    if (e == 0) return (float) m * 0.0009765625f;   // 0 included
    return std::bit_cast<float>(((e + 119u) << 23) | (m << 20));
}

/// ue4m3_half for every byte, for the CPU's rows (a table lookup a block).  Built at compile time: a namespace-scope
/// object with a constructor runs at program start on every CPU, and in a file compiled for a wider ISA its loop may be
/// vectorised past what the CPU has (upstream #2324: an illegal instruction before main).
struct HalfTable {
    float v[256];
    constexpr HalfTable() : v() {
        for (int x = 0; x < 256; ++x) v[x] = ue4m3_half((uint8_t) x);
    }
};
inline constexpr HalfTable kHalf{};

/// The expert's {s_gate, s_up, s_down} from its blob's tail (`tail_off` 0: none, every scale 1).
inline float tail_scale(const uint8_t* blob, uint64_t tail_off, int k) {
    if (tail_off == 0) return 1.0f;
    float f;
    std::memcpy(&f, blob + tail_off + 4 * (uint64_t) k, sizeof f);
    return f;
}

}  // namespace strata::kernels::nvfp4
