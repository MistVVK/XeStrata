// tools/vision/own/cpu_ops.cpp - the dispatcher of the CPU tower's kernel tiers (ISA-neutral, no special compile flags)
#include "cpu_kernels.h"

namespace strata_vit {

const CpuOps& cpu_ops(quant::Isa isa) {
    if (isa == quant::Isa::Avx512 && quant::isa_ok(quant::Isa::Avx512)) return cpu_ops_avx512();
    if (isa >= quant::Isa::Avx2 && quant::isa_ok(quant::Isa::Avx2)) return cpu_ops_avx2();
    return cpu_ops_scalar();
}

}  // namespace strata_vit
