#pragma once

#include "bmg_expert_window_plan.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/gemm.hpp"
#include <sycl/sycl.hpp>
#include <functional>
#include <string>

namespace strata::prefill::bmg_window {
void gather_rows(sycl::queue& queue, const PairPiece (&pieces)[2], int chunk, const int* src, const int* slot_table,
                 float* all_down);
enum class Phase { GateUp, Down, Dequant };
bool project_group(const Batch& group, const kernels::cpu::NativeFmt& fmt, const uint8_t* const* raw,
                   const uint16_t* group_input, uint16_t* hidden, float* all_down, uint16_t* dq_gu, uint16_t* dq_down,
                   Gemm& gemm, sycl::queue& queue, const std::function<void(Phase)>& mark, std::string& err);
} // namespace strata::prefill::bmg_window
