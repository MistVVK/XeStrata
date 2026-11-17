#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::kernels::iq4_decode {
// Exact existing lookup, retained as the diagnostic comparator.
inline uint32_t scalar(uint32_t q) {
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    uint32_t result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t n = (q >> (8 * i)) & 15;
        const uint32_t table = n < 4 ? t0 : n < 8 ? t1 : n < 12 ? t2 : t3;
        result |= ((table >> (8 * (n & 3))) & 255) << (8 * i);
    }
    return result;
}

// Four byte lookups in parallel through an exact bit-select tree. Whether these
// operations become native BFN instructions must be checked in the emitted ISA.
template<bool AddXorMask = false> inline uint32_t bit_select_impl(uint32_t q) {
    constexpr uint32_t table[] = {
        0x81818181u, 0x98989898u, 0xadadadadu, 0xbfbfbfbfu,
        0xcfcfcfcfu, 0xddddddddu, 0xeaeaeaeau, 0xf6f6f6f6u,
        0x01010101u, 0x0d0d0d0du, 0x19191919u, 0x26262626u,
        0x35353535u, 0x45454545u, 0x59595959u, 0x71717171u};
    uint32_t mask[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = (q >> i) & 0x01010101u;
        // A byte containing 0/1 maps to 0/255 without a carry to its neighbor.
        // The first form becomes an integer multiply by 255 in current IGC.
        if constexpr (AddXorMask) mask[i] = (b + 0x7f7f7f7fu) ^ 0x7f7f7f7fu;
        else mask[i] = (b << 8) - b;
    }
    uint32_t a[8], b[4], c[2];
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = sycl::bitselect(table[2 * i], table[2 * i + 1], mask[0]);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = sycl::bitselect(a[2 * i], a[2 * i + 1], mask[1]);
#pragma unroll
    for (int i = 0; i < 2; ++i) c[i] = sycl::bitselect(b[2 * i], b[2 * i + 1], mask[2]);
    return sycl::bitselect(c[0], c[1], mask[3]);
}
inline uint32_t bit_select(uint32_t q) { return bit_select_impl<false>(q); }
inline uint32_t bit_select_add_xor(uint32_t q) { return bit_select_impl<true>(q); }
} // namespace strata::kernels::iq4_decode

// Integration head (PR #4185 port): the same exact bit-select lookup under its first name, used by the decode kernels
// that came with that port; the same table and the same operations as iq4_decode::bit_select.
namespace strata::kernels {
inline uint32_t iq4_lut4_swar(uint32_t q4) { return iq4_decode::bit_select(q4); }
}
