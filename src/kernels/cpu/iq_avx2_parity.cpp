// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/cpu/iq_avx2_parity.cpp - the AVX-2 expert kernels' variants (iq_avx2.cpp, q2_avx2.cpp) bit for bit
// against each other and against ggml-cpu, and their time per expert.  No GPU, no model: random blocks, so every grid
// index and sign pattern occurs, and random activations quantized the way the engine quantizes them.
//
//     iq_avx2_parity [--cpu N]
//         IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS gate/up rows, IQ4_NL and Q2_0 down rows: every variant
//         this CPU runs, 1..8 tokens, each token's rows bit for bit the scalar variant's rows for that token alone (so
//         also the same at every width, #152); the scalar rows against ggml-cpu's vec_dot (Q2_0: a double
//         reference), rel <= 1e-5; and where the engine's dispatch (native_gu_rows) takes the multi-token kernel from
//         one token on (native_gu_mt_min), each token's rows the same alone and in a group: the i-quants', and the
//         K-quants' and Q8_0's (the other models' experts) whichever kernel the dispatch takes for them.
//     iq_avx2_parity --bench [--cpu N] [--nt 1,2,3,4] [--mb 256] [--reps 5] [--pairs iq3_s/iq4_nl,...]
//                    [--shape 2560x640] [--dispatch]
//         ms per expert on one thread: each variant, ggml-cpu's per-token dot, and the engine's dispatch
//         (native_gu_rows, native_down_rows / q2_rows_any) over --mb MB of expert blobs - far more than an L3 holds, so
//         the weights stream from DRAM as they do in decode.  --dispatch: only the kernels' own dispatch, ggml and the
//         engine's (the env decides: STRATA_IQ256_GATHER, STRATA_NO_AVXVNNI, STRATA_IQ_MT_MIN), for A/B runs against
//         another build.  --cpu pins the thread to that logical CPU.  --pairs takes any gate/up and down types
//         ggml-cpu multiplies (q6_k/q6_k, q4_k/q8_0, ...): a pair without kernels of its own measures ggml-cpu and the
//         engine's dispatch only, at --shape (n_embd x expert width: 2048x512 Ornith-1.5, 2560x960 llm-jp-4.1).
//
// Built with -DSTRATA_IQ_PARITY_DISPATCH_ONLY=1 it calls only the kernels' public entry points, so the same program
// builds on an engine from before the variants (--bench --dispatch against it).
#include "strata/kernels/cpu/canon_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/platform/env.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#include <sched.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

namespace {

int64_t kH = 2560, kFF = 640;   // the experts' geometry: n_embd, expert width (--shape for the bench)
constexpr int kMaxT = 8;

// ---- the variants: the public dispatch only in a DISPATCH_ONLY build
#if defined(STRATA_IQ_PARITY_DISPATCH_ONLY)
constexpr bool kVariants = false;
int gu_variants() { return 0; }
int down_variants() { return 0; }
void gu_rows_v(int, int type, const uint8_t* blob, size_t gu_row, size_t up_off, const void* const* act, int nt,
               float* const* ff) {
    cpu::iq256_gu_rows(type, blob, gu_row, up_off, (int) kH, act, nt, ff, 0, (int) kFF);
}
void gate_rows_v(int, int type, const uint8_t* w, size_t row_bytes, const void* const* act, int nt, float* const* out) {
    cpu::iq256_rows(type, w, row_bytes, (int) kH, act, nt, out, 0, (int) kFF);
}
void iq4nl_rows_v(int, const uint8_t* w, size_t row_bytes, const void* const* hq, int nt, float* const* out) {
    cpu::iq4nl256_down_rows(w, row_bytes, (int) kFF, hq, nt, out, 0, (int) kH);
}
void q2_rows_v(int, const uint8_t* w, size_t row_bytes, const cpu::ActQ* const* a, int nt, float* const* out) {
    cpu::q2_0_gguf_rows_multi_avx2(w, row_bytes, (int) (kFF / 64), a, nt, out, 0, (int) kH);
}
std::string variant_name(int) { return "dispatch"; }
bool kq_type(int) { return false; }
void kq_gu_v(int, int, const uint8_t*, size_t, size_t, const void* const*, int, float* const*) {}
void kq_dn_v(int, int, const uint8_t*, size_t, const void* const*, int, float* const*) {}
#else
constexpr bool kVariants = true;
int gu_variants() { return cpu::iq256_variants(); }
int down_variants() { return cpu::iq256_variants() & cpu::kIq256Vnni; }
void gu_rows_v(int v, int type, const uint8_t* blob, size_t gu_row, size_t up_off, const void* const* act, int nt,
               float* const* ff) {
    cpu::iq256_gu_rows_v(v, type, blob, gu_row, up_off, (int) kH, act, nt, ff, 0, (int) kFF);
}
void gate_rows_v(int v, int type, const uint8_t* w, size_t row_bytes, const void* const* act, int nt, float* const* out) {
    cpu::iq256_rows_v(v, type, w, row_bytes, (int) kH, act, nt, out, 0, (int) kFF);
}
void iq4nl_rows_v(int v, const uint8_t* w, size_t row_bytes, const void* const* hq, int nt, float* const* out) {
    cpu::iq4nl256_down_rows_v(v, w, row_bytes, (int) kFF, hq, nt, out, 0, (int) kH);
}
void q2_rows_v(int v, const uint8_t* w, size_t row_bytes, const cpu::ActQ* const* a, int nt, float* const* out) {
    cpu::q2_0_gguf_rows_multi_avx2_v((v & cpu::kIq256Vnni) != 0, w, row_bytes, (int) (kFF / 64), a, nt, out, 0, (int) kH);
}
std::string variant_name(int v) {
    if (v == 0) return "scalar";
    std::string s;
    if (v & cpu::kIq256Gather) s += "gather";
    if (v & cpu::kIq256Vnni) s += s.empty() ? "vnni" : "+vnni";
    return s;
}
// the K-quants' and Q8_0's multi-token kernels (kq256_*, q80_256_*): AVX2 or AVX-VNNI, no other variant
bool kq_type(int type) { return cpu::kq256_supported(type) || type == 8; }
void kq_gu_v(int v, int type, const uint8_t* blob, size_t gu_row, size_t up_off, const void* const* act, int nt,
             float* const* ff) {
    if (type == 8) cpu::q80_256_gu_rows_v(v, blob, gu_row, up_off, (int) kH, act, nt, ff, 0, (int) kFF);
    else cpu::kq256_gu_rows_v(v, type, blob, gu_row, up_off, (int) kH, act, nt, ff, 0, (int) kFF);
}
void kq_dn_v(int v, int type, const uint8_t* w, size_t row_bytes, const void* const* hq, int nt, float* const* out) {
    if (type == 8) cpu::q80_256_rows_v(v, w, row_bytes, (int) kFF, hq, nt, out, 0, (int) kH);
    else cpu::kq256_rows_v(v, type, w, row_bytes, (int) kFF, hq, nt, out, 0, (int) kH);
}
#endif

// every subset of a variant mask, the empty one (scalar) first
std::vector<int> subsets(int mask) {
    std::vector<int> v;
    for (int s = 0; s <= mask; ++s)
        if ((s & ~mask) == 0) v.push_back(s);
    return v;
}

uint64_t mix(uint64_t& s) {   // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

const ggml_type_traits_cpu* traits_cpu(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

// Where a block keeps its fp16 scales: the front for the i-quants, Q8_0 and Q5_0; Q4_K and Q5_K d and dmin first;
// Q6_K, Q3_K and Q2_K after their quants (ggml-common.h's block structs)
std::vector<size_t> scale_offsets(int type) {
    switch (type) {
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: return {0, 2};
        case GGML_TYPE_Q6_K: return {208};
        case GGML_TYPE_Q3_K: return {108};
        case GGML_TYPE_Q2_K: return {80, 82};
        default: return {0};
    }
}

// random bytes, then sane fp16 scales where every block keeps them
void fill_blocks(uint8_t* p, size_t bytes, int type, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) {
        const uint64_t v = mix(seed);
        std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i));
    }
    const size_t bs = ggml_type_size((ggml_type) type);
    const std::vector<size_t> offs = scale_offsets(type);
    for (size_t o = 0; o + bs <= bytes; o += bs)
        for (size_t at : offs) {
            const float d = (0.5f + (float) (mix(seed) >> 40) * (1.f / 16777216.f)) * 1e-3f;
            const ggml_fp16_t h = ggml_fp32_to_fp16(d);
            std::memcpy(p + o + at, &h, 2);
        }
}

void fill_blob(uint8_t* blob, const cpu::NativeFmt& f, uint64_t& seed) {
    fill_blocks(blob, f.down_off, f.gu_type, seed);                     // the gate and up rows
    fill_blocks(blob + f.down_off, f.bytes - f.down_off, f.d_type, seed);
}

// kMaxT tokens: the gate/up activation in ggml's vec_dot type (Q8_K for the i-quants) and the down one in ggml's form
// and in the Q2_0 kernels' form (ActQ), what the kernels here take; and both as the engine quantizes them for its
// dispatch (native_quant_act / native_quant_h: a layer of the canonical rows, canon_expert.hpp, has its own form)
struct alignas(64) HBuf {   // a down activation, 64-byte aligned as the pool's (SplitBufMulti::hq)
    uint8_t bytes[cpu::kNativeHBytes];
};
struct Acts {
    std::vector<std::vector<uint8_t>> gu, dn;
    std::vector<cpu::NativeAct> egu;   // the engine's forms, aligned as the pool's buffers
    std::vector<HBuf> eh;
    std::vector<cpu::ActQ> q2;
    const void* gup[kMaxT];
    const void* dnp[kMaxT];
    const void* egup[kMaxT];
    const void* ehp[kMaxT];
    const cpu::ActQ* q2p[kMaxT];
    Acts(const cpu::NativeFmt& f, unsigned seed)
        : gu(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes)), dn(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes)),
          egu(kMaxT), eh(kMaxT), q2(kMaxT) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) kH), h((size_t) kFF);
        const bool canon = cpu::canon_layer(f);
        for (int t = 0; t < kMaxT; ++t) {
            for (auto& v : x) v = nd(rng);
            for (auto& v : h) v = 0.05f * nd(rng);
            cpu::native_quant_act(f, x.data(), egu[(size_t) t].bytes);
            cpu::native_quant_h(f, h.data(), eh[(size_t) t].bytes);
            if (canon) {
                traits_cpu(f.gu_act)->from_float(x.data(), gu[(size_t) t].data(), kH);
                traits_cpu(f.d_act)->from_float(h.data(), dn[(size_t) t].data(), kFF);
            } else {
                gu[(size_t) t].assign(egu[(size_t) t].bytes, egu[(size_t) t].bytes + cpu::kNativeActBytes);
                dn[(size_t) t].assign(eh[(size_t) t].bytes, eh[(size_t) t].bytes + cpu::kNativeHBytes);
            }
            if (cpu::cpu_avx2_ok()) cpu::act_quant_any(h.data(), (int) kFF, q2[(size_t) t]);
            gup[t] = gu[(size_t) t].data();
            dnp[t] = dn[(size_t) t].data();
            egup[t] = egu[(size_t) t].bytes;
            ehp[t] = eh[(size_t) t].bytes;
            q2p[t] = &q2[(size_t) t];
        }
    }
};

double rel(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) { num += std::fabs((double) a[i] - b[i]); den += std::fabs((double) b[i]); }
    return num / (den + 1e-30);
}

size_t rows_differ(const float* a, const float* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += std::memcmp(static_cast<const void*>(a + i), static_cast<const void*>(b + i), 4) != 0;
    return d;
}

unsigned core_type() {   // CPUID 1Ah of the calling thread: 40h performance core, 20h E-core, 0 not reported
    unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    int x[4];
    __cpuid(x, 0);
    if ((unsigned) x[0] < 0x1A) return 0;
    __cpuidex(x, 0x1A, 0);
    r[0] = (unsigned) x[0];
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid(0, a, b, c, d);
    if (a < 0x1A) return 0;
    __cpuid_count(0x1A, 0, r[0], r[1], r[2], r[3]);
#endif
    return r[0] >> 24;
}

bool pin_to(int cpu_index) {
#if defined(_WIN32)
    if (cpu_index < 0 || cpu_index >= 64) return false;
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu_index) != 0;
#else
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu_index, &s);
    return sched_setaffinity(0, sizeof s, &s) == 0;
#endif
}

const char* type_name(int type) { return ggml_type_name((ggml_type) type); }

int type_of(const std::string& name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n != nullptr && name == n) return t;
    }
    return -1;
}

// ---- parity
int check_gate_up(int type) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(type, 20, kH, kFF, f, err)) {
        std::printf("  %-8s %s\n", type_name(type), err.c_str());
        return 1;
    }
    std::vector<uint8_t> blob(f.bytes);
    uint64_t seed = 0x5eed0000ull + (uint64_t) type;
    fill_blob(blob.data(), f, seed);
    const Acts a(f, 77u + (unsigned) type);
    const size_t R = (size_t) kFF;
    std::vector<float> ref(kMaxT * R);
    for (int t = 0; t < kMaxT; ++t)
        for (size_t r = 0; r < R; ++r)
            traits_cpu(type)->vec_dot((int) kH, &ref[t * R + r], 0, blob.data() + r * f.gu_row, 0, a.gup[t], 0, 1);
    // the first variant (scalar), each token alone: the rows every variant must give that token at every width
    std::vector<float> one_g(kMaxT * R), one_ff(kMaxT * R);
    for (int t = 0; t < kMaxT; ++t) {
        float* og = &one_g[t * R];
        float* of = &one_ff[t * R];
        gate_rows_v(0, type, blob.data(), f.gu_row, &a.gup[t], 1, &og);
        gu_rows_v(0, type, blob.data(), f.gu_row, f.up_off, &a.gup[t], 1, &of);
    }
    int failures = 0;
    const double r_ggml = rel(one_g.data(), ref.data(), one_g.size());
    std::printf("  %-8s gate rows vs ggml vec_dot: rel %.2e\n", type_name(type), r_ggml);
    if (!(r_ggml <= 1e-5)) { std::printf("  %-8s MISMATCH against ggml\n", type_name(type)); ++failures; }
    for (int v : subsets(gu_variants())) {
        size_t differ = 0;
        std::vector<float> g(kMaxT * R), ff(kMaxT * R);
        for (int nt = 1; nt <= kMaxT; ++nt) {
            float* gp[kMaxT];
            float* fp[kMaxT];
            for (int t = 0; t < nt; ++t) { gp[t] = &g[t * R]; fp[t] = &ff[t * R]; }
            gate_rows_v(v, type, blob.data(), f.gu_row, a.gup, nt, gp);
            gu_rows_v(v, type, blob.data(), f.gu_row, f.up_off, a.gup, nt, fp);
            for (int t = 0; t < nt; ++t)
                differ += rows_differ(gp[t], &one_g[t * R], R) + rows_differ(fp[t], &one_ff[t * R], R);
        }
        std::printf("  %-8s %-12s 1..8 tokens: %zu gate / gate+up rows differ from the %s one-token rows\n",
                    type_name(type), variant_name(v).c_str(), differ, variant_name(0).c_str());
        if (differ) ++failures;
    }
    return failures;
}

int check_down(int d_type) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(16, d_type, kH, kFF, f, err)) {
        std::printf("  %-8s %s\n", type_name(d_type), err.c_str());
        return 1;
    }
    std::vector<uint8_t> blob(f.bytes);
    uint64_t seed = 0xd0e50000ull + (uint64_t) d_type;
    fill_blob(blob.data(), f, seed);
    const Acts a(f, 91u + (unsigned) d_type);
    const uint8_t* w = blob.data() + f.down_off;
    const size_t R = (size_t) kH;
    std::vector<float> ref(kMaxT * R);
    if (d_type == 20) {
        for (int t = 0; t < kMaxT; ++t)
            for (size_t r = 0; r < R; ++r)
                traits_cpu(20)->vec_dot((int) kFF, &ref[t * R + r], 0, w + r * f.d_row, 0, a.dnp[t], 0, 1);
    } else {   // Q2_0 in the GGUF layout against the kernels' own activations: w = d * (code - 1), in double
        for (int t = 0; t < kMaxT; ++t)
            for (size_t r = 0; r < R; ++r) {
                const uint8_t* row = w + r * f.d_row;
                double s = 0;
                for (int b = 0; b < (int) (kFF / 64); ++b) {
                    const float d = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(row + 18 * (ptrdiff_t) b));
                    for (int i = 0; i < 64; ++i) {
                        const int code = (row[18 * b + 2 + i / 4] >> (2 * (i % 4))) & 3;
                        const int k = 64 * b + i;
                        s += (double) d * (code - 1) * a.q2[(size_t) t].scale[k / cpu::QKA] * a.q2[(size_t) t].q[k];
                    }
                }
                ref[t * R + r] = (float) s;
            }
    }
    auto run = [&](int v, int nt, float* const* out) {
        if (d_type == 20) iq4nl_rows_v(v, w, f.d_row, a.dnp, nt, out);
        else q2_rows_v(v, w, f.d_row, a.q2p, nt, out);
    };
    std::vector<float> one(kMaxT * R);
    for (int t = 0; t < kMaxT; ++t) {
        float* o = &one[t * R];
        if (d_type == 20) iq4nl_rows_v(0, w, f.d_row, &a.dnp[t], 1, &o);
        else q2_rows_v(0, w, f.d_row, &a.q2p[t], 1, &o);
    }
    int failures = 0;
    const double r_ref = rel(one.data(), ref.data(), one.size());
    std::printf("  %-8s down rows vs %s: rel %.2e\n", type_name(d_type),
                d_type == 20 ? "ggml vec_dot" : "a double reference", r_ref);
    if (!(r_ref <= 1e-5)) { std::printf("  %-8s MISMATCH against the reference\n", type_name(d_type)); ++failures; }
    for (int v : subsets(down_variants())) {
        size_t differ = 0;
        std::vector<float> o(kMaxT * R);
        for (int nt = 1; nt <= kMaxT; ++nt) {
            float* op[kMaxT];
            for (int t = 0; t < nt; ++t) op[t] = &o[t * R];
            run(v, nt, op);
            for (int t = 0; t < nt; ++t) differ += rows_differ(op[t], &one[t * R], R);
        }
        std::printf("  %-8s %-12s 1..8 tokens: %zu down rows differ from the %s one-token rows\n", type_name(d_type),
                    variant_name(v).c_str(), differ, variant_name(0).c_str());
        if (differ) ++failures;
    }
    return failures;
}

#if !defined(STRATA_IQ_PARITY_DISPATCH_ONLY)
// #152 through the engine's dispatch: where native_gu_rows gives a format the multi-token kernel from one token on
// (native_gu_mt_min 1: every format, unless STRATA_IQ_MT_MIN sets another count), each token's gate/up rows must be
// the same alone and in any group.
int check_engine_width(int type) {
    const int mt = cpu::native_gu_mt_min(type);
    if (mt != 1) {
        std::printf("  %-8s engine: ggml's dot below %d tokens, its one-token rows round differently (#152)\n",
                    type_name(type), mt);
        return 0;
    }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(type, 20, kH, kFF, f, err)) return 1;
    std::vector<uint8_t> blob(f.bytes);
    uint64_t seed = 0xe9900000ull + (uint64_t) type;
    fill_blob(blob.data(), f, seed);
    const Acts a(f, 13u + (unsigned) type);
    const size_t R = (size_t) kFF;
    std::vector<float> one(kMaxT * R), ff(kMaxT * R);
    for (int t = 0; t < kMaxT; ++t) {
        float* o = &one[t * R];
        cpu::native_gu_rows(f, blob.data(), &a.egup[t], 1, &o, 0, (int) kFF);
    }
    size_t differ = 0;
    for (int nt = 2; nt <= kMaxT; ++nt) {
        float* fp[kMaxT];
        for (int t = 0; t < nt; ++t) fp[t] = &ff[t * R];
        cpu::native_gu_rows(f, blob.data(), a.egup, nt, fp, 0, (int) kFF);
        for (int t = 0; t < nt; ++t) differ += rows_differ(fp[t], &one[t * R], R);
    }
    std::printf("  %-8s engine: multi-token from one token on, 2..8 tokens: %zu gate/up rows differ from the one-token "
                "rows\n", type_name(type), differ);
    return differ ? 1 : 0;
}
#endif

// ---- bench
struct Method {
    std::string name;
    std::function<void(const uint8_t* blob, int nt)> gu, dn;
};

std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t from = 0;
    while (from <= s.size()) {
        const size_t comma = s.find(',', from);
        const std::string tok = s.substr(from, comma == std::string::npos ? std::string::npos : comma - from);
        if (!tok.empty()) v.push_back((int) std::strtol(tok.c_str(), nullptr, 10));
        if (comma == std::string::npos) break;
        from = comma + 1;
    }
    return v;
}

int bench(const std::vector<std::string>& pairs, const std::vector<int>& nts, int mb, int reps, bool dispatch_only) {
    for (const std::string& pair : pairs) {
        const size_t slash = pair.find('/');
        const int gt = slash == std::string::npos ? -1 : type_of(pair.substr(0, slash));
        const int dt = slash == std::string::npos ? -1 : type_of(pair.substr(slash + 1));
        cpu::NativeFmt f;
        std::string err;
        if (gt < 0 || dt < 0 || !cpu::native_fmt(gt, dt, kH, kFF, f, err)) {
            std::printf("%s: not a pair of types ggml-cpu multiplies at %lld x %lld %s\n", pair.c_str(), (long long) kH,
                        (long long) kFF, err.c_str());
            return 2;
        }
        // the AVX-2 kernels' own variants for the pairs they implement; any other pair: ggml-cpu and the engine only
        const bool own = cpu::iq256_supported(gt) && (dt == 20 || dt == 42);
        const size_t E = std::max<size_t>(16, ((size_t) mb << 20) / f.bytes);
        std::unique_ptr<uint8_t[]> mem(new uint8_t[E * f.bytes + 64]);
        uint8_t* blobs = (uint8_t*) (((uintptr_t) mem.get() + 63) & ~(uintptr_t) 63);
        uint64_t seed = 0xbe9c0000ull + (uint64_t) gt * 64 + (uint64_t) dt;
        for (size_t e = 0; e < E; ++e) fill_blob(blobs + e * f.bytes, f, seed);
        const Acts a(f, 5u);
        std::vector<float> ffb(kMaxT * (size_t) kFF), outb(kMaxT * (size_t) kH);
        float* ff[kMaxT];
        float* out[kMaxT];
        for (int t = 0; t < kMaxT; ++t) { ff[t] = &ffb[(size_t) t * kFF]; out[t] = &outb[(size_t) t * kH]; }
        const ggml_vec_dot_t gdot = traits_cpu(gt)->vec_dot, ddot = traits_cpu(dt)->vec_dot;

        std::vector<Method> ms;
        auto down_v = [&, dt](int v) {
            return [&, v, dt](const uint8_t* blob, int nt) {
                if (dt == 20) iq4nl_rows_v(v, blob + f.down_off, f.d_row, a.dnp, nt, out);
                else q2_rows_v(v, blob + f.down_off, f.d_row, a.q2p, nt, out);
            };
        };
        if (own && !dispatch_only && kVariants) {
            for (int v : subsets(gu_variants() | down_variants()))
                ms.push_back({variant_name(v),
                              [&, v, gt](const uint8_t* blob, int nt) { gu_rows_v(v, gt, blob, f.gu_row, f.up_off, a.gup, nt, ff); },
                              down_v(v)});
        } else if (own) {
            ms.push_back({"kernels",
                          [&, gt](const uint8_t* blob, int nt) {
                              cpu::iq256_gu_rows(gt, blob, f.gu_row, f.up_off, (int) kH, a.gup, nt, ff, 0, (int) kFF);
                          },
                          [&, dt](const uint8_t* blob, int nt) {
                              if (dt == 20) cpu::iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) kFF, a.dnp, nt, out, 0, (int) kH);
                              else cpu::q2_0_gguf_rows_multi_avx2(blob + f.down_off, f.d_row, (int) (kFF / 64), a.q2p, nt, out, 0, (int) kH);
                          }});
        }
        // ggml-cpu's dot, token by token: the engine's one-token path (#152's rule)
        const std::function<void(const uint8_t*, int)> ggml_gu = [&](const uint8_t* blob, int nt) {
            for (int64_t r = 0; r < kFF; ++r)
                for (int t = 0; t < nt; ++t) {
                    float g = 0.f, u = 0.f;
                    gdot((int) kH, &g, 0, blob + (size_t) r * f.gu_row, 0, a.gup[t], 0, 1);
                    gdot((int) kH, &u, 0, blob + f.up_off + (size_t) r * f.gu_row, 0, a.gup[t], 0, 1);
                    ff[t][r] = (g / (1.f + std::exp(-g))) * u;
                }
        };
        const std::function<void(const uint8_t*, int)> ggml_dn = [&](const uint8_t* blob, int nt) {
            for (int64_t r = 0; r < kH; ++r)
                for (int t = 0; t < nt; ++t)
                    ddot((int) kFF, &out[t][r], 0, blob + f.down_off + (size_t) r * f.d_row, 0, a.dnp[t], 0, 1);
        };
        // the K-quant and Q8_0 kernels in each variant (scalar: the AVX2 copy), ggml-cpu for a side they do not take
        if (!own && !dispatch_only && kVariants && (kq_type(gt) || kq_type(dt)))
            for (int v : subsets(down_variants()))
                ms.push_back({variant_name(v),
                              kq_type(gt) ? std::function<void(const uint8_t*, int)>([&, v, gt](const uint8_t* blob, int nt) {
                                  kq_gu_v(v, gt, blob, f.gu_row, f.up_off, a.gup, nt, ff);
                              }) : ggml_gu,
                              kq_type(dt) ? std::function<void(const uint8_t*, int)>([&, v, dt](const uint8_t* blob, int nt) {
                                  kq_dn_v(v, dt, blob + f.down_off, f.d_row, a.dnp, nt, out);
                              }) : ggml_dn});
        ms.push_back({"ggml", ggml_gu, ggml_dn});
        ms.push_back({"engine",   // what the pool calls: native_gu_rows, native_down_rows or (Q2_0 off the canonical
                                  // rows) q2_rows_any
                      [&](const uint8_t* blob, int nt) { cpu::native_gu_rows(f, blob, a.egup, nt, ff, 0, (int) kFF); },
                      [&, dt](const uint8_t* blob, int nt) {
                          if (dt == 42 && !cpu::canon_layer(f))
                              cpu::q2_rows_any(blob + f.down_off, f.d_row, (int) (kFF / 64), a.q2p, nt, out, 0, (int) kH);
                          else cpu::native_down_rows(f, blob, a.ehp, nt, out, 0, (int) kH);
                      }});

        std::printf("\n%s: %zu experts of %.2f MB (%.0f MB), one thread\n", pair.c_str(), E, (double) f.bytes / 1048576.0,
                    (double) E * (double) f.bytes / 1048576.0);
        for (int nt : nts) {
            if (nt < 1 || nt > kMaxT) continue;
            std::vector<std::vector<double>> tg(ms.size()), td(ms.size());
            for (int rep = 0; rep < reps; ++rep)
                for (size_t k = 0; k < ms.size(); ++k) {
                    const size_t m = (k + (size_t) rep) % ms.size();   // rotate the order from rep to rep
                    double sg = 0, sd = 0;
                    for (size_t e = 0; e < E; ++e) {
                        const uint8_t* blob = blobs + e * f.bytes;
                        const auto t0 = Clock::now();
                        ms[m].gu(blob, nt);
                        const auto t1 = Clock::now();
                        ms[m].dn(blob, nt);
                        const auto t2 = Clock::now();
                        sg += std::chrono::duration<double, std::milli>(t1 - t0).count();
                        sd += std::chrono::duration<double, std::milli>(t2 - t1).count();
                    }
                    tg[m].push_back(sg / (double) E);
                    td[m].push_back(sd / (double) E);
                }
            std::printf("  %d token%s ms per expert, median of %d (min): gate/up, down, total\n", nt, nt == 1 ? "" : "s",
                        reps);
            for (size_t m = 0; m < ms.size(); ++m) {
                auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
                auto mn = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };
                std::vector<double> tot(tg[m].size());
                for (size_t i = 0; i < tot.size(); ++i) tot[i] = tg[m][i] + td[m][i];
                std::printf("    %-12s %7.3f (%.3f)  %7.3f (%.3f)  %7.3f (%.3f)\n", ms[m].name.c_str(), med(tg[m]),
                            mn(tg[m]), med(td[m]), mn(td[m]), med(tot), mn(tot));
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool do_bench = false, dispatch_only = !kVariants;
    int pin = -1, mb = 256, reps = 5;
    std::vector<int> nts = {1, 2, 3, 4};
    std::vector<std::string> pairs = {"iq3_s/iq4_nl", "iq3_xxs/iq4_nl", "iq2_s/q2_0", "iq2_xxs/q2_0"};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--bench") do_bench = true;
        else if (a == "--dispatch") dispatch_only = true;
        else if (a == "--cpu" && more) pin = (int) std::strtol(argv[++i], nullptr, 10);
        else if (a == "--mb" && more) mb = (int) std::strtol(argv[++i], nullptr, 10);
        else if (a == "--reps" && more) reps = std::max(1, (int) std::strtol(argv[++i], nullptr, 10));
        else if (a == "--nt" && more) nts = ints(argv[++i]);
        else if (a == "--shape" && more) {   // n_embd x expert width: 2048x512 (Ornith-1.5), 2560x960 (llm-jp-4.1)
            const std::string sh = argv[++i];
            const size_t x = sh.find('x');
            kH = std::strtoll(sh.c_str(), nullptr, 10);
            kFF = x == std::string::npos ? 0 : std::strtoll(sh.c_str() + x + 1, nullptr, 10);
            if (kH <= 0 || kFF <= 0 || kFF > cpu::kNativeMaxFF) {
                std::fprintf(stderr, "--shape: n_embd x expert width, the width at most %lld\n",
                             (long long) cpu::kNativeMaxFF);
                return 2;
            }
        } else if (a == "--pairs" && more) {
            pairs.clear();
            const std::string s = argv[++i];
            for (size_t from = 0;;) {
                const size_t comma = s.find(',', from);
                pairs.push_back(s.substr(from, comma == std::string::npos ? std::string::npos : comma - from));
                if (comma == std::string::npos) break;
                from = comma + 1;
            }
        } else {
            std::fprintf(stderr, "usage: iq_avx2_parity [--cpu N] | --bench [--cpu N] [--nt 1,2,3,4] [--mb 256] "
                                 "[--reps 5] [--pairs iq3_s/iq4_nl,...] [--shape 2560x640] [--dispatch]\n");
            return 2;
        }
    }
    if (pin >= 0 && !pin_to(pin)) std::printf("iq_avx2_parity: could not pin to CPU %d\n", pin);
    ggml_cpu_init();
    auto env = [](const char* k) { const char* v = strata::env_get(k); return v ? v : "-"; };
    std::printf("iq_avx2_parity: %s (avx2 %d, avx512 %d), CPU %d, core type %02Xh\n", cpu::cpu_name().c_str(),
                (int) cpu::cpu_avx2_ok(), (int) cpu::cpu_avx512_ok(), pin, core_type());
    std::printf("  XESTRATA_IQ256_GATHER %s, XESTRATA_NO_AVXVNNI %s, XESTRATA_IQ_MT_MIN %s, XESTRATA_IQ_PREFETCH %s\n",
                env("IQ256_GATHER"), env("NO_AVXVNNI"), env("IQ_MT_MIN"), env("IQ_PREFETCH"));
    if (!cpu::cpu_avx2_ok()) {
        std::printf("iq_avx2_parity: no AVX2 here, nothing to check (the engine takes ggml-cpu's dot)\n");
        return 0;
    }
#if !defined(STRATA_IQ_PARITY_DISPATCH_ONLY)
    std::printf("  variants this CPU runs: %s; this thread's: %s\n",
                variant_name(cpu::iq256_variants()).c_str(), variant_name(cpu::iq256_variant()).c_str());
#endif
    if (do_bench) return bench(pairs, nts, mb, reps, dispatch_only);
    int failures = 0;
    for (int type : {16, 17, 22, 18, 21, 23}) failures += check_gate_up(type);
    for (int type : {20, 42}) failures += check_down(type);
#if !defined(STRATA_IQ_PARITY_DISPATCH_ONLY)
    for (int type : {16, 17, 22, 18, 21, 23, 10, 11, 12, 13, 14, 8}) failures += check_engine_width(type);
#endif
    std::printf("iq_avx2_parity: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
