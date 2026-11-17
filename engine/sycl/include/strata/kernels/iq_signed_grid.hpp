#pragma once
#include <cstdint>

namespace strata::kernels::iq_signed_grid {
// Called only behind the observed decode-shape/capacity guards. The product
// fits in 32 bits; add it to the full-width USM base without truncating a pointer.
template <typename Block> inline const Block* block_at(const void* base, uint32_t index) {
    const uint32_t offset = index * uint32_t(sizeof(Block));
    return reinterpret_cast<const Block*>(static_cast<const uint8_t*>(base) + offset);
}
// Four consecutive sign bits become one bit in each byte. The unsigned
// multiply is modulo 2^32; the mask discards its other (non-overlapping) bits.
inline uint32_t bits(uint32_t nibble) {
    return ((nibble & 15u) * 0x00204081u) & 0x01010101u;
}
// The pinned IQ2_S/IQ2_XXS grids contain positive bytes 8, 25 and 43,
// After XOR every byte is <=254,
// so adding its sign bit cannot carry between bytes. Restricted to these
// separately validated codebooks; zero or negative grid bytes are invalid.
inline int apply(uint32_t positive, uint32_t nibble) {
    const uint32_t b = bits(nibble);
    return static_cast<int>((positive ^ (b * 255u)) + b);
}
}
