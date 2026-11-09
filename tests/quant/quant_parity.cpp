// tests/quant/quant_parity.cpp - the layer against its ORACLE: ggml-cpu's own dot products and activation quantizers, bit for bit.
//
// ggml-cpu is linked into this program only.  The flavour of the linked ggml-cpu decides which contract is checked (isa.hpp):
//   a default / STRATA_PORTABLE build  -> Arith::Avx2      (ggml's __AVX2__ branches)
//   -DSTRATA_ISA_FLOOR=avx             -> Arith::Avx       (its __AVX__ branches)
//   -DSTRATA_ISA_FLOOR=none            -> Arith::Generic   (its _generic C code)
// Every kernel of that contract the CPU can run is compared with ggml's, on rows made by ggml's own quantizers from random floats
// (so scales, mins and sign patterns are those of real models), at 1..8 blocks, and the activation quantizers on random rows and on rows
// built to be exact ties (where half-to-even and half-away-from-zero differ).
//
// usage: quant_parity [--rows N] [--type NAME] [--verbose]      exit 0 = every comparison bit-identical
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "strata/quant/act.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/isa.hpp"
#include "strata/quant/types.hpp"
#include "internal.hpp"   // src/quant: the fp16 helpers (test only)

namespace sq = strata::quant;

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 99) {}
    uint32_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return (uint32_t) ((s * 0x2545F4914F6CDD1Dull) >> 32);
    }
    float uni() { return (float) next() / 4294967296.0f * 2.f - 1.f; }
    float gauss() { return (uni() + uni() + uni() + uni()) * 0.5f; }
};

sq::Arith oracle_arith() {
#if defined(__aarch64__)
    return sq::Arith::Neon;
#else
    if (ggml_cpu_has_avx2() && ggml_cpu_has_fma()) return sq::Arith::Avx2;
    if (ggml_cpu_has_avx()) return sq::Arith::Avx;
    return sq::Arith::Generic;
#endif
}

int g_fail = 0, g_checked = 0;
bool g_verbose = false;

void fail(const char* what, const std::string& detail) {
    ++g_fail;
    if (g_fail <= 40) std::printf("FAIL %s: %s\n", what, detail.c_str());
}

// weights of `type`, `nrows` rows of n elements, quantized by ggml from gaussian floats of scale `amp`
std::vector<uint8_t> make_rows(ggml_type type, int n, int nrows, Rng& r, float amp) {
    ggml_quantize_init(type);
    std::vector<float> f((size_t) n * nrows), im((size_t) n, 1.0f);
    for (float& v : f) v = r.gauss() * amp;
    std::vector<uint8_t> out(ggml_row_size(type, n) * nrows);
    const size_t got = ggml_quantize_chunk(type, f.data(), out.data(), 0, nrows, n, ggml_quantize_requires_imatrix(type) ? im.data() : nullptr);
    if (got != out.size()) {
        std::printf("quantize_chunk(%s) returned %zu, expected %zu\n", ggml_type_name(type), got, out.size());
        std::exit(2);
    }
    return out;
}

bool bits_equal(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

void test_dot(sq::Type st, int rows) {
    const ggml_type gt = (ggml_type) (int) st;
    const sq::TypeInfo* ti = sq::type_info(st);
    const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu(gt);
    if (ti == nullptr || tr == nullptr || tr->vec_dot == nullptr) return;
    const sq::Arith ar = oracle_arith();
    // ggml's SSE4.2-only flavour (STRATA_ISA_FLOOR=none) has an SSSE3 branch for Q4_0 that is neither the generic C code nor AVX: not a contract of the layer
    if (ar == sq::Arith::Generic && st == sq::Type::Q4_0) {
        std::printf("  dot q4_0      skipped: ggml's SSSE3 branch is not the Generic contract\n");
        return;
    }
    const sq::DotKernel* ks[16];
    const int nk = sq::dot_kernels(st, ar, ks, 16);
    const ggml_type_traits_cpu* ta = ggml_get_type_traits_cpu(tr->vec_dot_type);
    if ((int) tr->vec_dot_type != (int) ti->act) {
        fail(ti->name, "vec_dot_type differs from the layer's type table");
        return;
    }
    if ((int) ggml_blck_size(gt) != ti->blck || ggml_type_size(gt) != ti->bytes) fail(ti->name, "block geometry differs from ggml's");
    int ran = 0;
    for (int ki = 0; ki < nk; ++ki) {
        const sq::DotKernel* k = ks[ki];
        if (!sq::isa_ok(k->isa)) continue;
        ++ran;
        const int unit = ti->blck > (int) ggml_blck_size(tr->vec_dot_type) ? ti->blck : (int) ggml_blck_size(tr->vec_dot_type);
        Rng r(1234 + (uint64_t) st);
        for (int nbk = 1; nbk <= 8; ++nbk) {
            const int n = unit * nbk;
            const float amp = nbk % 3 == 0 ? 0.02f : (nbk % 3 == 1 ? 1.0f : 40.0f);
            const std::vector<uint8_t> w = make_rows(gt, n, rows, r, amp);
            std::vector<float> xf((size_t) n);
            std::vector<uint8_t> act(ggml_row_size(tr->vec_dot_type, n));
            for (int rep = 0; rep < 3; ++rep) {
                for (float& v : xf) v = r.gauss() * (rep == 0 ? 1.f : (rep == 1 ? 0.05f : 7.f));
                ta->from_float(xf.data(), act.data(), n);
                for (int row = 0; row < rows; ++row) {
                    const uint8_t* wr = w.data() + (size_t) row * ggml_row_size(gt, n);
                    float want = 0.f, got = 0.f;
                    tr->vec_dot(n, &want, 0, wr, 0, act.data(), 0, 1);
                    k->fn(n, &got, wr, act.data());
                    ++g_checked;
                    if (!bits_equal(want, got)) {
                        char b[200];
                        std::snprintf(b, sizeof b, "kernel %s n=%d row %d: ggml %.9g (%08x) ours %.9g (%08x)", k->name, n, row, want, *(uint32_t*) &want, got,
                                      *(uint32_t*) &got);
                        fail(ti->name, b);
                        break;
                    }
                }
            }
        }
        if (g_verbose) std::printf("  dot %-10s %-18s (%s/%s) ok so far\n", ti->name, k->name, sq::arith_name(k->arith), sq::isa_name(k->isa));
    }
    if (ran == 0 && g_verbose) std::printf("  dot %-10s no kernel of contract %s runs here\n", ti->name, sq::arith_name(ar));
    if (ran == 0 && nk == 0) fail(ti->name, std::string("the layer has no kernel of contract ") + sq::arith_name(ar));
}

void test_act(sq::Type act, int rows) {
    const ggml_type gt = (ggml_type) (int) act;
    const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu(gt);
    const sq::TypeInfo* ti = sq::type_info(act);
    if (tr == nullptr || tr->from_float == nullptr || ti == nullptr) return;
    const sq::Arith ar = oracle_arith();
    const sq::ActKernel* k = sq::act_kernel(act, ar);
    if (k == nullptr || k->arith != ar) {
        fail(ti->name, std::string("no activation quantizer of contract ") + sq::arith_name(ar));
        return;
    }
    Rng r(777 + (uint64_t) act);
    for (int nbk = 1; nbk <= 6; ++nbk) {
        const int n = ti->blck * nbk;
        std::vector<float> x((size_t) n);
        std::vector<uint8_t> a(ti->bytes * nbk), b(ti->bytes * nbk);
        for (int it = 0; it < rows * 4; ++it) {
            for (float& v : x) v = r.gauss() * (it % 3 == 0 ? 1.f : (it % 3 == 1 ? 1e-3f : 300.f));
            if (it % 7 == 0) std::memset(x.data(), 0, x.size() * 4);   // an all-zero block
            if (it % 11 == 0) x[(size_t) (r.next() % n)] = 0.f;
            // ties: a block whose absolute maximum is 127 (id = 1) with half-integers elsewhere, and its negation
            if (it % 5 == 0) {
                for (int j = 0; j < n; ++j) x[(size_t) j] = ((int) (r.next() % 255) - 127) * 0.5f + (r.next() & 1 ? 0.f : 0.5f);
                for (int bI = 0; bI < nbk; ++bI) x[(size_t) bI * ti->blck + r.next() % ti->blck] = r.next() & 1 ? 127.f : -127.f;
                if (act == sq::Type::Q8_K) for (int bI = 0; bI < nbk; ++bI) x[(size_t) bI * ti->blck + r.next() % ti->blck] = 127.f;
            }
            tr->from_float(x.data(), a.data(), n);
            k->fn(x.data(), b.data(), n);
            ++g_checked;
            if (std::memcmp(a.data(), b.data(), a.size()) != 0) {
                size_t at = 0;
                while (a[at] == b[at]) ++at;
                char buf[160];
                std::snprintf(buf, sizeof buf, "kernel %s n=%d iteration %d: first differing byte %zu (ggml %02x ours %02x)", k->name, n, it, at, a[at], b[at]);
                fail(ti->name, buf);
                break;
            }
        }
    }
    if (g_verbose) std::printf("  act %-10s %-18s ok\n", ti->name, k->name);
}

// to_float of every type the layer dequantizes, and the Q4_0 requantizer (--mtp-q4's output), bit for bit against ggml-base
void test_dequant(int rows) {
    static const sq::Type kTypes[] = {sq::Type::Q2_0, sq::Type::Q4_0, sq::Type::Q4_1, sq::Type::Q5_0, sq::Type::Q5_1, sq::Type::Q8_0, sq::Type::MXFP4,
                                      sq::Type::NVFP4, sq::Type::Q2_K, sq::Type::Q3_K, sq::Type::Q4_K, sq::Type::Q5_K, sq::Type::Q6_K, sq::Type::IQ2_XXS,
                                      sq::Type::IQ2_XS, sq::Type::IQ2_S, sq::Type::IQ3_XXS, sq::Type::IQ3_S, sq::Type::IQ1_S, sq::Type::IQ1_M,
                                      sq::Type::IQ4_NL, sq::Type::IQ4_XS};
    for (sq::Type st : kTypes) {
        const ggml_type gt = (ggml_type) (int) st;
        const ggml_type_traits* tr = ggml_get_type_traits(gt);
        const sq::TypeInfo* ti = sq::type_info(st);
        if (tr == nullptr || tr->to_float == nullptr) continue;
        if (!sq::can_dequant(st)) {
            fail(ti->name, "the layer cannot dequantize it");
            continue;
        }
        Rng r(321 + (uint64_t) st);
        for (int nbk = 1; nbk <= 4; ++nbk) {
            const int n = ti->blck * nbk;
            const std::vector<uint8_t> w = make_rows(gt, n, rows, r, nbk % 2 ? 1.0f : 30.0f);
            std::vector<float> a((size_t) n), b((size_t) n);
            for (int row = 0; row < rows; ++row) {
                const uint8_t* wr = w.data() + (size_t) row * ggml_row_size(gt, n);
                tr->to_float(wr, a.data(), n);
                if (!sq::dequant_row(st, wr, b.data(), n) || std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
                    fail(ti->name, "dequant_row differs from ggml's to_float, n=" + std::to_string(n));
                    break;
                }
            }
            ++g_checked;
        }
    }
    // Q4_0 from floats: ggml_quantize_chunk(Q4_0, no imatrix) is quantize_row_q4_0_ref
    Rng r(99);
    for (int nbk = 1; nbk <= 6; ++nbk) {
        const int n = 32 * nbk;
        std::vector<float> f((size_t) n);
        std::vector<uint8_t> a(18 * nbk), b(18 * nbk);
        for (int it = 0; it < rows * 4; ++it) {
            for (float& v : f) v = r.gauss() * (it % 3 == 0 ? 1.f : (it % 3 == 1 ? 1e-3f : 200.f));
            if (it % 9 == 0) std::memset(f.data(), 0, f.size() * 4);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, f.data(), a.data(), 0, 1, n, nullptr);
            sq::quantize_q4_0_ref(f.data(), b.data(), n);
            ++g_checked;
            if (a != b) {
                fail("q4_0", "quantize_q4_0_ref differs from ggml_quantize_chunk, n=" + std::to_string(n));
                break;
            }
        }
    }
}

// Values within a few ulps of the rounding boundaries of the Q8_K quantizer (nearest_int adds the 2^23 + 2^22 magic number to iscale * x) and of the Q4_0
// requantizer (x * id + 8.5, truncated).  A compiler that contracts the multiply and the add into one FMA gives another result on exactly these inputs, and
// ggml-base does so on arm64 under GCC (not on x86), so random data finds the mismatch only after millions of values: this finds it in a few thousand.
void test_boundary(int rows) {
    Rng r(4242);
    const sq::Arith ar = oracle_arith();
    const sq::ActKernel* k8 = sq::act_kernel(sq::Type::Q8_K, ar);
    const ggml_type_traits_cpu* t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    auto jitter = [&r](float v) {
        for (uint32_t t = r.next() % 4; t > 0; --t) v = std::nextafter(v, r.next() & 1 ? INFINITY : -INFINITY);
        return v;
    };
    if (k8 != nullptr && t8 != nullptr) {
        std::vector<float> x(256);
        std::vector<uint8_t> a(sq::kBlockQ8_K), b(sq::kBlockQ8_K);
        for (int it = 0; it < rows * 300; ++it) {
            const float dd = std::ldexp(0.5f + 0.5f * std::fabs(r.uni()), (int) (r.next() % 14) - 7);
            const float x0 = (r.next() & 1 ? 127.f : -127.f) * dd;
            const float step = -1.f / (-127.f / x0);
            x[0] = x0;
            for (int j = 1; j < 256; ++j) x[(size_t) j] = jitter(((int) (r.next() % 253) - 126 + 0.5f) * step);
            t8->from_float(x.data(), a.data(), 256);
            k8->fn(x.data(), b.data(), 256);
            ++g_checked;
            if (a != b) {
                fail("q8_K", "boundary values: the quantizer differs from ggml's (a fused multiply-add on one side), iteration " + std::to_string(it));
                break;
            }
        }
    }
    std::vector<float> x(32);
    std::vector<uint8_t> a(18), b(18);
    for (int it = 0; it < rows * 300; ++it) {
        const float dd = std::ldexp(0.5f + 0.5f * std::fabs(r.uni()), (int) (r.next() % 14) - 7);
        x[0] = -8.f * dd;
        for (int j = 1; j < 32; ++j) x[(size_t) j] = jitter(((int) (r.next() % 15) + 1 - 8.5f) * dd);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, x.data(), a.data(), 0, 1, 32, nullptr);
        sq::quantize_q4_0_ref(x.data(), b.data(), 32);
        ++g_checked;
        if (a != b) {
            fail("q4_0", "boundary values: quantize_q4_0_ref differs from ggml's (a fused multiply-add on one side), iteration " + std::to_string(it));
            break;
        }
    }
}

void test_fp16() {
    // our software fp16 conversions against ggml's, every fp16 value and a spread of floats
    int bad = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        const float a = ggml_fp16_to_fp32((ggml_fp16_t) h);
        
        const float b = sq::fp16_to_fp32((uint16_t) h);
        if (std::isnan(a) ? !std::isnan(b) : !bits_equal(a, b)) ++bad;
    }
    Rng r(5);
    for (int i = 0; i < 4000000; ++i) {
        uint32_t w = r.next();
        float f;
        std::memcpy(&f, &w, 4);
        if (std::isnan(f)) continue;
        
        if (ggml_fp32_to_fp16(f) != sq::fp32_to_fp16(f)) ++bad;
    }
    ++g_checked;
    if (bad) fail("fp16", std::to_string(bad) + " conversions differ from ggml's");
}

}  // namespace

int main(int argc, char** argv) {
    int rows = 24;
    const char* only = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) rows = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--type") && i + 1 < argc) only = argv[++i];
        else if (!std::strcmp(argv[i], "--verbose")) g_verbose = true;
    }
    ggml_cpu_init();
    const sq::Arith ar = oracle_arith();
    std::printf("quant_parity: oracle flavour %s (ggml avx=%d avx2=%d fma=%d f16c=%d avx512=%d), layer best isa %s, default contract %s\n", sq::arith_name(ar),
                ggml_cpu_has_avx(), ggml_cpu_has_avx2(), ggml_cpu_has_fma(), ggml_cpu_has_f16c(), ggml_cpu_has_avx512(), sq::isa_name(sq::best_isa()),
                sq::arith_name(sq::default_arith()));
    test_fp16();
    test_dequant(rows);
    test_boundary(rows);
    for (sq::Type a : {sq::Type::Q8_0, sq::Type::Q8_1, sq::Type::Q8_K})
        if (only == nullptr || !std::strcmp(only, sq::type_info(a)->name)) test_act(a, rows);
    static const sq::Type kTypes[] = {sq::Type::Q2_0, sq::Type::Q4_0, sq::Type::Q4_1, sq::Type::Q5_0, sq::Type::Q5_1, sq::Type::Q8_0, sq::Type::IQ4_NL,
                                      sq::Type::MXFP4, sq::Type::NVFP4, sq::Type::Q2_K, sq::Type::Q3_K, sq::Type::Q4_K, sq::Type::Q5_K, sq::Type::Q6_K,
                                      sq::Type::IQ1_S, sq::Type::IQ1_M, sq::Type::IQ2_XXS, sq::Type::IQ2_XS, sq::Type::IQ2_S, sq::Type::IQ3_XXS,
                                      sq::Type::IQ3_S, sq::Type::IQ4_XS};
    for (sq::Type t : kTypes)
        if (only == nullptr || !std::strcmp(only, sq::type_info(t)->name)) test_dot(t, rows);
    std::printf("quant_parity: %d comparisons, %d failures -> %s\n", g_checked, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
