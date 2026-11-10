// tools/vision/own/cpu_ops.cpp - the dispatcher of the CPU tower's kernel tiers and the tiers' out-of-line library helpers (ISA-neutral, no special compile flags)
#include "cpu_kernels.h"

#include <cmath>

namespace strata_vit {

const CpuOps& cpu_ops(quant::Isa isa) {
    if (isa == quant::Isa::Neon) return quant::isa_ok(quant::Isa::Neon) ? cpu_ops_neon() : cpu_ops_scalar();
    if (isa == quant::Isa::Avx512 && quant::isa_ok(quant::Isa::Avx512)) return cpu_ops_avx512();
    if (isa >= quant::Isa::Avx2 && quant::isa_ok(quant::Isa::Avx2)) return cpu_ops_avx2();
    return cpu_ops_scalar();
}

float cpu_tier_expf(float x) { return std::exp(x); }

}  // namespace strata_vit
