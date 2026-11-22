#pragma once
#include <cstdint>
#include <sycl/sycl.hpp>

namespace strata::prefill::bmg_gate_up {
bool supported(const sycl::queue& queue);
bool run(sycl::queue& queue, const uint16_t* x, const uint16_t* weights, uint16_t* hidden, int rows);
}
