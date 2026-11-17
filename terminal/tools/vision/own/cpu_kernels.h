// tools/vision/own/cpu_kernels.h - the instruction-set specific kernels of the CPU tower (VitCpu), one table per ISA tier.
//
// Every tier lives in its own translation unit with its own compile flags (cpu_kernels_scalar.cpp, _avx2.cpp, _avx512.cpp, all from cpu_kernels_impl.h), so
// the portable build runs on any CPU and the dispatcher (cpu_ops()) picks the best tier the CPU has (quant::best_isa(), STRATA_FORCE_ISA caps it).
// The table type and the getters are in cpu_tier.h, which is all the wider-ISA tier files may include (see there); this header is for baseline code.
#pragma once

#include "cpu_caps.h"
#include "cpu_tier.h"

namespace strata_vit {

const CpuOps& cpu_ops(quant::Isa isa);

}  // namespace strata_vit
