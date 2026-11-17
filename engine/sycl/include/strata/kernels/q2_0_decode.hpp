#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::kernels {
// Four adjacent two-bit codes become four signed bytes {-1,0,1,2}.
// Every byte before addition is 0..3: adding 0x7f cannot carry into
// its neighbor. This replaces a byte-permute lookup, without requantizing.
inline uint32_t q2_0_decode4(uint32_t codes) {
    codes &= 255;
    codes = (codes | (codes << 12)) & 0x000f000fu;
    codes = (codes | (codes << 6)) & 0x03030303u;
    return (codes + 0x7f7f7f7fu) ^ 0x80808080u;
}
inline sycl::int2 q2_0_decode8(uint32_t codes) {
    return {(int)q2_0_decode4(codes), (int)q2_0_decode4(codes >> 8)};
}
} // namespace strata::kernels
