// src/quant/dispatch.cpp - kernel lookup: the best kernel of an arithmetic contract that this CPU can run.
#include <algorithm>

#include "registry.hpp"
#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"

namespace strata::quant {
namespace {

template <typename K, typename T, int N>
int gather(const T (&tables)[N], Type t, Arith a, const K** out, int cap, Type K::*field) {
    int n = 0;
    for (const T& tb : tables)
        for (int i = 0; i < tb.n; ++i) {
            const K& k = tb.v[i];
            if (k.*field == t && k.arith == a && n < cap) out[n++] = &k;
        }
    // best (widest hardware) first
    std::stable_sort(out, out + n, [](const K* x, const K* y) { return (int) x->isa > (int) y->isa; });
    return n;
}

constexpr int kMaxK = 16;

int collect_dots(Type w, Arith a, const DotKernel** out, int cap) {
    const DotTable tables[] = {
        generic_dot_table(),
#ifdef STRATA_QUANT_X86_TIERS
        avx_dot_table(), avx2_dot_table(), avx512_dot_table(),
#endif
#ifdef STRATA_QUANT_ARM_TIERS
        neon_dot_table(),
#endif
    };
    return gather(tables, w, a, out, cap, &DotKernel::weight);
}

int collect_acts(Type act, Arith a, const ActKernel** out, int cap) {
    const ActTable tables[] = {
        generic_act_table(),
#ifdef STRATA_QUANT_X86_TIERS
        avx_act_table(), avx2_act_table(),
#endif
#ifdef STRATA_QUANT_ARM_TIERS
        neon_act_table(),
#endif
    };
    return gather(tables, act, a, out, cap, &ActKernel::act);
}

}  // namespace

int dot_kernels(Type w, Arith a, const DotKernel** out, int cap) noexcept { return collect_dots(w, a, out, cap); }

const DotKernel* dot_kernel(Type w, Arith a, Isa max_isa) noexcept {
    const DotKernel* v[kMaxK];
    const int n = collect_dots(w, a, v, kMaxK);
    for (int i = 0; i < n; ++i)
        if ((int) v[i]->isa <= (int) max_isa && isa_ok(v[i]->isa)) return v[i];
    return nullptr;
}

const DotKernel* dot_kernel(Type w, Arith a) noexcept { return dot_kernel(w, a, best_isa()); }

const ActKernel* act_kernel(Type act, Arith a, Isa max_isa) noexcept {
    const ActKernel* v[kMaxK];
    const int n = collect_acts(act, a, v, kMaxK);
    for (int i = 0; i < n; ++i)
        if ((int) v[i]->isa <= (int) max_isa && isa_ok(v[i]->isa)) return v[i];
    return nullptr;
}

const ActKernel* act_kernel(Type act, Arith a) noexcept { return act_kernel(act, a, best_isa()); }

}  // namespace strata::quant
