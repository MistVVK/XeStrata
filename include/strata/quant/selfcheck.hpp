// include/strata/quant/selfcheck.hpp - start-up self-checks against our own reference (own-stack phase 2).
//
// ggml used to be the thing the engine compared its rows with at first use; now the reference is the layer's scalar code.  A synthetic row
// pair is built from a fixed PRNG with every scale a power of two and small activation values, so every float operation of every
// contract is exact (all partial sums are integers below 2^24 or dyadic fractions) and every kernel must give the reference's bits
// whatever its summation order.  That catches a wrong lane, shuffle, sign or table, which is what a miscompiled or mis-dispatched kernel does.
// The summation ORDER of each kernel is not checked here: the oracle tests (tests/quant/quant_parity.cpp) check it against ggml-cpu.
#pragma once

#include <string>

#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"

namespace strata::quant {

/// True when the kernel gives the reference's result on the synthetic rows (several row counts, odd block counts).  `why` gets one line on failure.
bool selfcheck(const DotKernel& k, std::string* why = nullptr);
/// Same for an activation quantizer: against the reference quantizer of the Generic contract on tie-free inputs (the tie behaviour is
/// the oracle test's job).
bool selfcheck(const ActKernel& k, std::string* why = nullptr);

/// dot_kernel() with the check built in: the best kernel that passes (each (type, contract, isa) is checked once per process); nullptr if none.
const DotKernel* checked_dot_kernel(Type w, Arith a) noexcept;
const ActKernel* checked_act_kernel(Type act, Arith a) noexcept;

}  // namespace strata::quant
