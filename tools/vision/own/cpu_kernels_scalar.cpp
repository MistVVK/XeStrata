// tools/vision/own/cpu_kernels_scalar.cpp - the portable tier of the CPU tower's kernels: plain C++ over a 4-float vector type (the compiler turns it into
// whatever vector unit the target has, NEON included).  Compiled without any ISA flags.
#include "cpu_kernels.h"

#include <cmath>
#include <cstring>

namespace strata_vit {
namespace {
namespace kn {

struct V {
    static constexpr int W = 4;
    float x[4];
    static V load(const float* p) { V v; std::memcpy(v.x, p, sizeof v.x); return v; }
    void store(float* p) const { std::memcpy(p, x, sizeof x); }
    static V set1(float f) { V v; for (int i = 0; i < 4; ++i) v.x[i] = f; return v; }
    static V zero() { return set1(0.0f); }
    static V fma(V a, V b, V c) { V v; for (int i = 0; i < 4; ++i) v.x[i] = a.x[i] * b.x[i] + c.x[i]; return v; }
    static float reduce_add(V a) { return (a.x[0] + a.x[2]) + (a.x[1] + a.x[3]); }
    static float reduce_max(V a) { return std::fmax(std::fmax(a.x[0], a.x[1]), std::fmax(a.x[2], a.x[3])); }
    friend V operator+(V a, V b) { V v; for (int i = 0; i < 4; ++i) v.x[i] = a.x[i] + b.x[i]; return v; }
    friend V operator-(V a, V b) { V v; for (int i = 0; i < 4; ++i) v.x[i] = a.x[i] - b.x[i]; return v; }
    friend V operator*(V a, V b) { V v; for (int i = 0; i < 4; ++i) v.x[i] = a.x[i] * b.x[i]; return v; }
    friend V operator/(V a, V b) { V v; for (int i = 0; i < 4; ++i) v.x[i] = a.x[i] / b.x[i]; return v; }
};
inline V vmax(V a, V b) { V v; for (int i = 0; i < 4; ++i) v.x[i] = a.x[i] > b.x[i] ? a.x[i] : b.x[i]; return v; }
inline V vexp(V a) {
    V v;
    for (int i = 0; i < 4; ++i) v.x[i] = std::exp(a.x[i] < -87.0f ? -87.0f : a.x[i] > 88.0f ? 88.0f : a.x[i]);
    return v;
}
inline float round_one(float f, int kind) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if (kind == 1) {
        if ((u & 0x7fffffffu) > 0x7f800000u) return f;
        u += 0x7fffu + ((u >> 16) & 1u);
        u &= 0xffff0000u;
        std::memcpy(&f, &u, 4);
        return f;
    }
    const float a = std::fabs(f);
    if (!(a < 65520.0f)) return f;
    if (a < 6.103515625e-05f) {
        const float q = std::nearbyint(a * 16777216.0f) / 16777216.0f;
        return (u & 0x80000000u) ? -q : q;
    }
    int e;
    std::frexp(a, &e);
    const float ulp = std::ldexp(1.0f, e - 11);
    const float q = std::nearbyint(a / ulp) * ulp;
    return (u & 0x80000000u) ? -q : q;
}
inline void round_buf(float* p, size_t n, int kind) {
    for (size_t i = 0; i < n; ++i) p[i] = round_one(p[i], kind);
}
inline void conv16(const void* Wm, int kind, size_t off, int n, float* out) {
    const uint16_t* p = (const uint16_t*) Wm + off;
    for (int i = 0; i < n; ++i) out[i] = kind == 1 ? quant::bf16_to_f32(p[i]) : quant::f16_to_f32(p[i]);
}

}  // namespace kn
}  // namespace
}  // namespace strata_vit

#define STRATA_KN_MR 4
#define STRATA_KN_NAME "scalar"
#define STRATA_KN_GETTER cpu_ops_scalar
#include "cpu_kernels_impl.h"
