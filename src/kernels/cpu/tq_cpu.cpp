// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/cpu/tq_cpu.cpp - see include/strata/kernels/cpu/tq_cpu.hpp: the state table, the scalar reference
// kernel, the activation unpacking and the run-time ISA choice.  Compiled for the baseline x86-64 (no AVX), so
// everything here runs on any CPU; the AVX2 / AVX-512 kernels are in their own translation units.
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/platform/env.hpp"

#include "tq_states.hpp"
#include "ggml-common-tq.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// The AVX2 / AVX-512 units (tq_avx2.cpp, tq_avx512.cpp) exist on x86-64 builds only; anything else (arm64: the Metal build)
// runs the scalar kernel.
#if defined(__x86_64__) || defined(_M_X64)
#define STRATA_TQ_X86 1
#endif

namespace strata::kernels::cpu {
namespace {

alignas(64) int16_t g_table[(tq_detail::kTqStateMask + 1) * 4];

void fill_table() {
    for (uint32_t s = 0; s <= tq_detail::kTqStateMask; ++s) {
        const uint32_t x = s * 0x9e3779b1u;
        const uint32_t pts[2] = {x >> 21, (x >> 10) & 2047u};   // the hash's two codebook points
        for (int h = 0; h < 2; ++h)
            for (int j = 0; j < 2; ++j) {
                // fp16 * 2^13 is exact in float; round half away from zero, as the fork's ggml-cpu table does
                const float v = tq_detail::tq_half_to_float(tq_lut_f16[2 * pts[h] + j]) * 8192.0f;
                g_table[4 * s + 2 * h + j] = (int16_t) (v < 0.0f ? v - 0.5f : v + 0.5f);
            }
    }
}

int env_isa_cap() {
    const char* e = strata::env_get("TQ_ISA");
    if (e == nullptr || *e == 0) return 2;
    const std::string v(e);
    if (v == "scalar") return 0;
    if (v == "avx2") return 1;
    return 2;
}

template <int TY>
void rows_scalar_t(const uint8_t* w, size_t row_bytes, int n, const tq_detail::TqAct* const* a, int nt,
                   float* const* out, int r0, int r1) {
    using F = tq_detail::TqFmt<TY>;
    constexpr int NS = F::NS, WPS = 128 / NS, SPK = 32 / WPS;   // weights per step, steps per Q8_0 sub-block
    const int nb = n / 128;
    constexpr size_t bsz = 2 + F::NB;
    const int16_t* tab = tq_state_table();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        float acc[kTqMaxTokens][4] = {};
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const float dx = tq_detail::tq_half_to_float(tq_detail::tq_ld16(blk));
            uint32_t st[NS];
            tq_detail::tq_states<TY>(blk + 2, st);
            for (int t = 0; t < nt; ++t) {
                const int16_t* q = a[t]->q + 128 * (ptrdiff_t) i;
                for (int k = 0; k < 4; ++k) {
                    int32_t s = 0;
                    for (int j = 0; j < SPK; ++j) {
                        const int16_t* wv = tab + 4 * st[SPK * k + j];   // a TQP step: the row's first two weights
                        const int16_t* qv = q + 32 * (ptrdiff_t) k + WPS * (ptrdiff_t) j;
                        for (int e = 0; e < WPS; ++e) s += wv[e] * qv[e];
                    }
                    const float sc = dx * a[t]->d[4 * i + k];
                    acc[t][k] = std::fma((float) s, sc, acc[t][k]);
                }
            }
        }
        for (int t = 0; t < nt; ++t)
            out[t][r] = ((acc[t][0] + acc[t][1]) + (acc[t][2] + acc[t][3])) * (1.0f / 8192.0f);
    }
}

}  // namespace

const int16_t* tq_state_table() {
    static std::once_flag once;
    std::call_once(once, fill_table);
    return g_table;
}

bool tq_supported(int t) noexcept { return t >= 144 && t <= 149; }

int tq_kbits(int t) noexcept { return t == 144 ? 8 : t == 145 ? 6 : t == 146 ? 7 : t >= 147 && t <= 149 ? t - 142 : 0; }

bool tq_isa_ok(TqIsa isa) {
#if defined(STRATA_TQ_X86)
    switch (isa) {
        case TqIsa::avx512: return cpu_avx512_ok();
        case TqIsa::avx2: return cpu_avx2_ok();
        default: return true;
    }
#else
    return isa == TqIsa::scalar;   // arm64 (the Metal build): the AVX units are not compiled, the scalar kernel runs
#endif
}

TqIsa tq_isa() {
    static const TqIsa isa = [] {
        const int cap = env_isa_cap();
        if (cap >= 2 && tq_isa_ok(TqIsa::avx512)) return TqIsa::avx512;
        if (cap >= 1 && tq_isa_ok(TqIsa::avx2)) return TqIsa::avx2;
        return TqIsa::scalar;
    }();
    return isa;
}

bool tq_kernels_enabled() {
    static const bool on = [] {
        if (const char* e = strata::env_get("NO_TQ_KERNELS"); e != nullptr && std::strtol(e, nullptr, 10) != 0) return false;
        // Every CPU runs them: the scalar kernel (baseline x86-64) where there is no AVX2.  A ggml built from upstream
        // has no vec_dot for the trellis types to fall back to, so a CPU without AVX2 must not be sent there.
        return true;
    }();
    return on;
}

const char* tq_isa_name(TqIsa isa) {
    return isa == TqIsa::avx512 ? "avx512" : isa == TqIsa::avx2 ? "avx2" : "scalar";
}

namespace tq_detail {

void unpack_act(const void* q8_0, int n, TqAct& a) {
    const uint8_t* p = (const uint8_t*) q8_0;
    for (int b = 0; b < n / 32; ++b, p += 34) {   // block_q8_0: fp16 d, 32 x int8
        a.d[b] = tq_half_to_float(tq_ld16(p));
        const int8_t* q = (const int8_t*) (p + 2);
        for (int j = 0; j < 32; ++j)
            a.q[32 * b + j] = q[j];  // NOLINT(bugprone-signed-char-misuse): q8_0's values are signed
    }
}

void tq_rows_scalar(int ty, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1) {
    switch (ty) {
        case 144: rows_scalar_t<144>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 145: rows_scalar_t<145>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 146: rows_scalar_t<146>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 147: rows_scalar_t<147>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 148: rows_scalar_t<148>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 149: rows_scalar_t<149>(w, row_bytes, n, a, nt, out, r0, r1); break;
        default: std::abort();   // a caller bug: tq_supported() was not asked
    }
}

}  // namespace tq_detail

namespace {

// the activations of one call, unpacked once for all its rows
struct ActSet {
    tq_detail::TqAct act[kTqMaxTokens];
    const tq_detail::TqAct* ptr[kTqMaxTokens];
};

ActSet& unpack_all(const void* const* act, int n, int nt) {
    thread_local std::unique_ptr<ActSet> set;
    if (!set) set.reset(new ActSet);
    for (int t = 0; t < nt; ++t) {
        tq_detail::unpack_act(act[t], n, set->act[t]);
        set->ptr[t] = &set->act[t];
    }
    return *set;
}

void rows_on(TqIsa isa, int ty, const uint8_t* w, size_t row_bytes, int n, const tq_detail::TqAct* const* a, int nt,
             float* const* out, int r0, int r1) {
    switch (isa) {
#if defined(STRATA_TQ_X86)
        case TqIsa::avx512: tq_detail::tq_rows_avx512(ty, w, row_bytes, n, a, nt, out, r0, r1); break;
        case TqIsa::avx2: tq_detail::tq_rows_avx2(ty, w, row_bytes, n, a, nt, out, r0, r1); break;
#endif
        default: tq_detail::tq_rows_scalar(ty, w, row_bytes, n, a, nt, out, r0, r1); break;
    }
}

}  // namespace

void tq_rows(TqIsa isa, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
             float* const* out, int r0, int r1) {
    if (!tq_isa_ok(isa)) std::abort();   // a caller bug: asked for an ISA this CPU does not have
    tq_state_table();
    const ActSet& s = unpack_all(act, n, nt);
    rows_on(isa, ggml_type, w, row_bytes, n, s.ptr, nt, out, r0, r1);
}

void tq_gu_rows(TqIsa isa, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                const void* const* act, int nt, float* const* ff, int r0, int r1) {
    if (!tq_isa_ok(isa)) std::abort();
    tq_state_table();
    const ActSet& s = unpack_all(act, n, nt);
    thread_local std::vector<float> gbuf, ubuf;
    const size_t nr = (size_t) r1;   // row r of token t at [t * r1 + r]
    if (gbuf.size() < nr * kTqMaxTokens) { gbuf.resize(nr * kTqMaxTokens); ubuf.resize(nr * kTqMaxTokens); }
    float* gp[kTqMaxTokens] = {};
    float* up[kTqMaxTokens] = {};
    for (int t = 0; t < nt; ++t) {
        gp[t] = gbuf.data() + (size_t) t * nr;
        up[t] = ubuf.data() + (size_t) t * nr;
    }
    rows_on(isa, ggml_type, blob, gu_row, n, s.ptr, nt, gp, r0, r1);
    rows_on(isa, ggml_type, blob + up_off, gu_row, n, s.ptr, nt, up, r0, r1);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) {
            const float g = gp[t][r], u = up[t][r];
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;   // native_gu_rows' SwiGLU expression
        }
}

}  // namespace strata::kernels::cpu
