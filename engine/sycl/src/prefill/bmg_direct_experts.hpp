#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
namespace strata::prefill::bmg_direct {
struct Expert {
    const uint16_t *x;
    const uint8_t *gate, *up, *down;
    uint16_t *hidden;
    float *output;
    int rows;
};
struct Batch {
    Expert experts[16];
    int count;
};
// Return the arithmetic layout for a group of 1..32-row experts, or -1 for the expanded-weight chain.
int layout(int type, int count, int max_rows);
void gate(sycl::queue &, const Batch &, int type, int variant);
void down(sycl::queue &, const Batch &, int variant);
} // namespace strata::prefill::bmg_direct
