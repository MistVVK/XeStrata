// include/strata/kernels/cpu/pool_quant.hpp - what the CPU expert pool (native_expert.cpp) and the DeepSeek CPU lane (dsv41/ds4_gg_cpu.cpp) ask of a quant type:
// its dot product, the activation format that dot takes, the activation quantizer, row geometry.
//
// With STRATA_OWN_CPU_QUANT (the default build) these come from the Strata quant layer (src/quant): no ggml-cpu function is called by the pool.  Without it
// they are ggml-cpu's type traits, exactly as before (kept selectable for two releases).  The two builds expose the same names, so the callers do not care.
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>

#if defined(STRATA_OWN_CPU_QUANT)
#include "strata/quant/act.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/fp16.hpp"
#include "strata/quant/isa.hpp"
#include "strata/quant/selfcheck.hpp"
#include "strata/quant/types.hpp"
#define STRATA_POOL_LIB "the quant layer"
#else
#include "ggml-cpu.h"
#include "ggml.h"
#define STRATA_POOL_LIB "ggml-cpu"
#endif

namespace strata::kernels::cpu::pq {

// ggml's type ids of the formats the pool names
constexpr int kTQ5_0 = 6, kTIq4nl = 20, kTQ8_0 = 8, kTQ8_K = 15;

#if defined(STRATA_OWN_CPU_QUANT)
namespace sq = strata::quant;
/// ggml's ggml_type_traits_cpu in the layer's terms: the weight type's dot of the default arithmetic contract, the activation format that dot takes, and, for an
/// activation format, its quantizer.  Every kernel passes the layer's self-check once before it is handed out.
struct Tr {
    const sq::DotKernel* k = nullptr;
    sq::ActQuantFn q = nullptr;
    int vec_dot_type = -1;
    bool has_dot() const { return k != nullptr; }
    bool has_from_float() const { return q != nullptr; }
    void dot(int n, float* s, const void* w, const void* a) const { k->fn(n, s, w, a); }
    void from_float(const float* x, void* y, int64_t n) const { q(x, y, n); }
};
inline Tr traits(int type) {
    constexpr int kN = 64;   // ggml's ids below the trellis types; those are tq_cpu.hpp's and never asked here
    static Tr cache[kN];
    static std::once_flag once[kN];
    if (type < 0 || type >= kN) return Tr{};
    std::call_once(once[type], [type] {
        const sq::TypeInfo* ti = sq::type_info(type);
        if (ti == nullptr) return;
        Tr t;
        if (ti->has_dot) {
            t.k = sq::checked_dot_kernel(ti->type, sq::default_arith());
            t.vec_dot_type = (int) ti->act;
        }
        if (ti->is_act) {
            const sq::ActKernel* a = sq::checked_act_kernel(ti->type, sq::default_arith());
            t.q = a ? a->fn : nullptr;
        }
        cache[type] = t;
    });
    return cache[type];
}
inline size_t row_size_of(int type, int64_t n) { return sq::row_bytes(type, n); }
inline size_t type_size_of(int type) { const sq::TypeInfo* t = sq::type_info(type); return t ? t->bytes : 0; }
inline int blck_size_of(int type) { const sq::TypeInfo* t = sq::type_info(type); return t ? t->blck : 0; }
/// "this build's ggml-cpu is the AVX2 flavour": the layer's default contract is AVX2's whenever the CPU has AVX2 + FMA + F16C, which is what the shipped
/// ggml-cpu (STRATA_PORTABLE) gives on every CPU, an AVX-512 one included; ggml's own AVX-512 flavour does not exist here.
inline bool flavour_avx2() { return sq::default_arith() == sq::Arith::Avx2; }
inline bool flavour_f16c() { return sq::caps().avx2; }
inline bool flavour_avx512() { return false; }
inline void layer_init() {}
/// A row of blocks of `type` as floats (the DeepSeek embedding reader); false for a type the layer does not dequantize.
inline bool to_float(int type, const void* src, float* dst, int64_t n) { return sq::dequant_row((sq::Type) type, src, dst, n); }
inline uint16_t fp16_bits(float f) { return sq::fp32_to_fp16(f); }
#else
struct Tr {
    const ggml_type_traits_cpu* t = nullptr;
    int vec_dot_type = -1;
    bool has_dot() const { return t != nullptr && t->vec_dot != nullptr; }
    bool has_from_float() const { return t != nullptr && t->from_float != nullptr; }
    void dot(int n, float* s, const void* w, const void* a) const { t->vec_dot(n, s, 0, w, 0, a, 0, 1); }
    void from_float(const float* x, void* y, int64_t n) const { t->from_float(x, y, n); }
};
inline Tr traits(int type) {
    Tr r;
    r.t = ggml_get_type_traits_cpu((ggml_type) type);
    r.vec_dot_type = r.t ? (int) r.t->vec_dot_type : -1;
    return r;
}
inline size_t row_size_of(int type, int64_t n) { return ggml_row_size((ggml_type) type, n); }
inline size_t type_size_of(int type) { return ggml_type_size((ggml_type) type); }
inline int blck_size_of(int type) { return (int) ggml_blck_size((ggml_type) type); }
inline bool flavour_avx2() { return ggml_cpu_has_avx2() && ggml_cpu_has_fma(); }
inline bool flavour_f16c() { return ggml_cpu_has_f16c(); }
inline bool flavour_avx512() { return ggml_cpu_has_avx512(); }
inline void layer_init() { ggml_cpu_init(); }
inline bool to_float(int type, const void* src, float* dst, int64_t n) {
    const ggml_type_traits* t = ggml_get_type_traits((ggml_type) type);
    if (t == nullptr || t->to_float == nullptr) return false;
    t->to_float(src, dst, n);
    return true;
}
inline uint16_t fp16_bits(float f) { return ggml_fp32_to_fp16(f); }
#endif

}  // namespace strata::kernels::cpu::pq
