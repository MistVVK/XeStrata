// src/image/vae/convert.hpp - host-side weight conversion of Strata's VAE decoder, split over threads.
// Every value is converted on its own (bf16 -> f32 exact, f32 -> f16 round to nearest even with
// strata::kernels::f16_from_f32), so the result is the same bytes for any thread count
// (tools/image/gate/test_vae_convert.py checks it on every bf16 bit pattern).  Host only.
#pragma once

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace strata::image {

// fn(begin, end) over [0, n) in `threads` contiguous parts (threads <= 1: on the calling thread).
template <class F>
void parallel_ranges(size_t n, int threads, F fn) {
    const size_t t = (size_t) std::max(1, std::min<int>(threads, (int) std::max<size_t>(1, n)));
    if (t <= 1) {
        fn((size_t) 0, n);
        return;
    }
    std::vector<std::thread> pool;
    const size_t step = (n + t - 1) / t;
    for (size_t b = 0; b < n; b += step) pool.emplace_back(fn, b, std::min(n, b + step));
    for (auto& th : pool) th.join();
}

inline uint16_t bf16_at(const unsigned char* raw, size_t i) { return (uint16_t) (raw[2 * i] | (raw[2 * i + 1] << 8)); }

// bf16 bits (little-endian bytes) -> float, values [b, e).
inline void bf16_to_f32_range(const unsigned char* raw, size_t b, size_t e, float* out) {
    for (size_t i = b; i < e; ++i) out[i] = strata::kernels::f32_from_bf16(bf16_at(raw, i));
}

// A convolution's bf16 weights [oc][ic][taps] -> f16 bits [oc][taps * ic] (row index tap * ic + i), rows [ob, oe).
inline void bf16_conv_to_f16_rows(const unsigned char* raw, int ic, int taps, size_t ob, size_t oe, uint16_t* out) {
    const size_t K = (size_t) taps * ic;
    for (size_t o = ob; o < oe; ++o)
        for (int i = 0; i < ic; ++i)
            for (int t = 0; t < taps; ++t)
                out[o * K + (size_t) t * ic + i] =
                    strata::kernels::f16_from_f32(strata::kernels::f32_from_bf16(bf16_at(raw, (o * ic + i) * taps + t)));
}

// The same two over threads (the test's entry points; the decoder splits a list of jobs instead).
inline void bf16_to_f32(const unsigned char* raw, size_t n, float* out, int threads) {
    parallel_ranges(n, threads, [&](size_t b, size_t e) { bf16_to_f32_range(raw, b, e, out); });
}
inline void bf16_conv_to_f16(const unsigned char* raw, int oc, int ic, int taps, uint16_t* out, int threads) {
    parallel_ranges((size_t) oc, threads, [&](size_t b, size_t e) { bf16_conv_to_f16_rows(raw, ic, taps, b, e, out); });
}

}  // namespace strata::image
