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
// TWO LEGS (OWN-2A-ORACLE, 11-10).  ggml-cpu is C that the compiler may contract into FMAs (-ffp-contract=fast, GCC's default for gnu11), and newer
// GCCs fuse where GCC 13 did not: ggml's own results move with the compiler that builds this test (measured on one Broadwell box: GCC 14.3 moves 55
// IQ4_NL rows, GCC 15.2 those and 183 Q2_0 rows, by 1-4 ulp), while the layer's results stay the same bytes under GCC 13.3, 14.3 and 15.2.  So:
//   leg 1, the gate: every group (flavour, section, type, n) of our results against tests/quant/golden_x86.txt, the digests of the reference that
//          GCC 13.3's ggml-cpu computes for the same inputs (its header says how it was made).  Any difference fails, whatever the compiler.
//   leg 2, information: our results row by row against the ggml-cpu linked here, as before.  Its summary names the compiler and the groups where
//          that ggml differs from the golden reference.  QUANT_PARITY_LIVE_STRICT=1 in the environment makes its differences fail too (the old
//          meaning of this test: right only where the local compiler builds ggml as GCC 13 does).
// Where no golden exists (arm64: the Neon contract; a run with --rows other than 24) leg 2 is the gate, as before.
//
// A failing group, debugged without a GCC 13 box at hand: the FAIL line names its key, e.g. "avx2 dot iq4_nl 96".  Run
//   quant_parity --dump-group "avx2 dot iq4_nl 96" > here.txt
// on the failing build and the same on a GCC 13 build of the same tree (or ask for its file): every row of the group, ours per kernel and ggml's, as
// output bits (hex bytes), with the row's input digest.  `diff` of the two files shows which rows moved and on which side.
//
// usage: quant_parity [--rows N] [--type NAME] [--verbose] [--golden FILE] [--dump-group KEY] [--write-golden FILE] [--regen-check]
//   --write-golden FILE  write this flavour's golden lines (the reference digests of the linked ggml) to FILE; tests/quant/make_golden.sh uses it
//   --regen-check        GCC 13.x on x86 only (exit 77 = skipped elsewhere): the linked ggml regenerates this flavour's stored lines byte for byte
// exit 0 = every comparison of the gate leg bit-identical
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
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

#ifndef STRATA_QUANT_GOLDEN_FILE
#define STRATA_QUANT_GOLDEN_FILE ""
#endif

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
    // every draw in its own statement: the operands of one expression have no evaluation order in C++, and GCC and MSVC pick different ones
    float gauss() {
        const float a = uni();
        const float b = uni();
        const float c = uni();
        const float d = uni();
        return (a + b + c + d) * 0.5f;
    }
};

// The golden digest: FNV-1a with 64-bit state (offset basis 0xcbf29ce484222325, prime 0x100000001b3), over the bytes in memory order, continued
// across the rows of a group.  Fixed here on purpose (never std::hash or a library's hash): the golden file is only meaningful while this stays.
constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;
uint64_t golden_fnv1a64(const void* p, size_t n, uint64_t h = kFnvBasis) {
    const unsigned char* b = (const unsigned char*) p;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}
// the published FNV-1a 64 test vectors
bool digest_selftest() {
    return golden_fnv1a64("", 0) == 0xcbf29ce484222325ull && golden_fnv1a64("a", 1) == 0xaf63dc4c8601ec8cull &&
           golden_fnv1a64("foobar", 6) == 0x85944171f73967e8ull;
}

std::string compiler_name() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#elif defined(_MSC_FULL_VER)
    return "MSVC " + std::to_string(_MSC_FULL_VER);
#else
    return "unknown compiler";
#endif
}

// Toolchain families of the golden file.  "gcc" lines are made by GCC 13.x on x86 and hold for every toolchain; "msvc" lines, made by MSVC 19.4x, stand
// beside them only for the groups where MSVC's arithmetic differs (MSVC does not contract the intrinsics' add-of-mul into FMA where GCC does).
// The family a build reads its own lines from, and whether --regen-check can regenerate them here:
#if defined(_MSC_VER) && !defined(__clang__)
const char* const kFamily = "msvc";
#if _MSC_VER >= 1940 && _MSC_VER < 1950 && (defined(_M_X64) || defined(_M_IX86))
constexpr bool kRegenHere = true;
#else
constexpr bool kRegenHere = false;
#endif
#else
const char* const kFamily = "gcc";
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 13 && (defined(__x86_64__) || defined(__i386__))
constexpr bool kRegenHere = true;
#else
constexpr bool kRegenHere = false;
#endif
#endif
constexpr int kGoldenRows = 24;   // the golden file holds the default --rows

sq::Arith oracle_arith() {
#if defined(__aarch64__)
    return sq::Arith::Neon;
#else
    if (ggml_cpu_has_avx2() && ggml_cpu_has_fma()) return sq::Arith::Avx2;
    if (ggml_cpu_has_avx()) return sq::Arith::Avx;
    return sq::Arith::Generic;
#endif
}

int g_fail = 0, g_checked = 0, g_live_diff = 0, g_group_checks = 0;   // rows compared (x kernels), live rows that differ, golden group checks
bool g_info_only = false;   // the oracle is a flavour the layer does not claim (arm64 with SVE): count the differences, do not fail
bool g_verbose = false;
bool g_golden_leg = false;   // leg 1 runs (a golden section for this flavour, default --rows)
bool g_live_strict = false;  // leg 2's differences fail (no golden here, or QUANT_PARITY_LIVE_STRICT=1)
std::string g_flavour;       // arith_name of the contract checked
std::string g_dump_key;      // --dump-group

void fail(const char* what, const std::string& detail) {
    ++g_fail;
    if (g_fail <= 40) std::printf("FAIL %s: %s\n", what, detail.c_str());
}

// leg 2: a row where the linked ggml and ours differ
void live_diff(const char* what, const std::string& detail) {
    ++g_live_diff;
    if (g_live_strict) fail(what, detail);
    else if (g_live_diff <= 10) std::printf("live: %s: %s\n", what, detail.c_str());
}

std::string hex_bytes(const void* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    const unsigned char* b = (const unsigned char*) p;
    for (size_t i = 0; i < n; ++i) {
        s += d[b[i] >> 4];
        s += d[b[i] & 15];
    }
    return s;
}

std::string hex64(uint64_t v) {
    char b[20];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return b;
}

// ---- golden file (tests/quant/golden_x86.txt): lines "<family> <flavour> <section> <name> <n> <rows> <input digest> <reference digest>";
// "%note <family> <flavour> <section> <name> <why>" lines say why a family has its own lines; '#' lines are comments (the header)
struct GoldenLine {
    int rows;
    uint64_t in, out;
    std::string text;   // the line without its family, as --write-golden writes it
};
std::map<std::string, GoldenLine> g_golden;       // this flavour's gcc lines, by key "<flavour> <section> <name> <n>"
std::map<std::string, GoldenLine> g_golden_own;   // this flavour's lines of this build's family when it is not gcc (the per-toolchain lines)
std::vector<std::string> g_golden_gcc_order;      // the gcc keys in stored order (--regen-check under GCC 13)
std::vector<std::string> g_notes;                 // "%note" lines of this family and flavour

bool load_golden(const std::string& path, const std::string& flavour) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string fam, fl, sec, name, n, in, out;
        if (line[0] == '%') {
            std::string tag;
            ss >> tag >> fam >> fl;
            if (fam == kFamily && fl == flavour) g_notes.push_back(line.substr(6));
            continue;
        }
        int rows = 0;
        if (!(ss >> fam >> fl >> sec >> name >> n >> rows >> in >> out)) {
            std::printf("quant_parity: malformed golden line: %s\n", line.c_str());
            std::exit(2);
        }
        if (fl != flavour) continue;
        const std::string key = fl + " " + sec + " " + name + " " + n;
        GoldenLine gl = {rows, std::strtoull(in.c_str(), nullptr, 16), std::strtoull(out.c_str(), nullptr, 16), line.substr(fam.size() + 1)};
        if (fam == "gcc") {
            g_golden[key] = gl;
            g_golden_gcc_order.push_back(key);
        } else if (fam == kFamily) {
            g_golden_own[key] = gl;
        }
    }
    return true;
}

// the line this build is held to: its family's own line where one exists, else the gcc line
const GoldenLine* golden_for(const std::string& key) {
    auto own = g_golden_own.find(key);
    if (own != g_golden_own.end()) return &own->second;
    auto it = g_golden.find(key);
    return it == g_golden.end() ? nullptr : &it->second;
}

// one group of comparisons: the inputs and the reference (the linked ggml) are digested once, each kernel's results against them
struct Group {
    std::string key, name;
    int rows = 0;
    uint64_t in = kFnvBasis, ref = kFnvBasis;
};
std::vector<Group> g_groups;                    // visiting order = the order of the golden lines
std::map<std::string, int> g_groups_by_key;
std::vector<std::string> g_ref_moved;           // leg 2: groups where the linked ggml differs from golden

Group& open_group(const char* section, const char* name, int n) {
    const std::string key = g_flavour + " " + section + " " + name + " " + std::to_string(n);
    g_groups.push_back(Group());
    Group& g = g_groups.back();
    g.key = key;
    g.name = name;
    g_groups_by_key[key] = (int) g_groups.size() - 1;
    return g;
}

// close a group: the reference against golden (leg 2's summary), and every kernel's digest against golden (leg 1)
void close_group(Group& g, const std::vector<std::pair<const char*, uint64_t>>& ours) {
    if (!g_golden_leg) return;
    const GoldenLine* gp = golden_for(g.key);
    if (gp == nullptr) {
        fail(g.name.c_str(), "group \"" + g.key + "\" has no golden line (a new case: regenerate tests/quant/golden_x86.txt, tests/quant/make_golden.sh)");
        return;
    }
    const GoldenLine& gl = *gp;
    if (gl.in != g.in || gl.rows != g.rows) {
        fail(g.name.c_str(), "group \"" + g.key + "\": the INPUTS differ from the golden file's (" + std::to_string(g.rows) + " rows, digest " + hex64(g.in) +
                                 "; golden " + std::to_string(gl.rows) + " rows, " + hex64(gl.in) + "): ggml's quantizers that make the test rows moved here");
        return;
    }
    if (gl.out != g.ref) g_ref_moved.push_back(g.key);
    for (const auto& k : ours) {
        ++g_group_checks;
        if (k.second != gl.out)
            fail(g.name.c_str(), std::string("kernel ") + k.first + ": group \"" + g.key + "\" differs from the golden reference (ours " + hex64(k.second) +
                                     ", golden " + hex64(gl.out) + "); rows: --dump-group \"" + g.key + "\"");
    }
}

void dump_row(const Group& g, const char* who, int idx, uint64_t in, const void* out, size_t n) {
    if (g.key == g_dump_key) std::printf("ROW %s %s %d %s %s\n", g.key.c_str(), who, idx, hex64(in).c_str(), hex_bytes(out, n).c_str());
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

uint32_t float_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

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
    std::vector<const sq::DotKernel*> run;
    for (int ki = 0; ki < nk; ++ki)
        if (sq::isa_ok(ks[ki]->isa)) run.push_back(ks[ki]);
    const int unit = ti->blck > (int) ggml_blck_size(tr->vec_dot_type) ? ti->blck : (int) ggml_blck_size(tr->vec_dot_type);
    Rng r(1234 + (uint64_t) st);
    std::vector<int> live_bad(run.size(), 0);
    for (int nbk = 1; nbk <= 8; ++nbk) {
        const int n = unit * nbk;
        const float amp = nbk % 3 == 0 ? 0.02f : (nbk % 3 == 1 ? 1.0f : 40.0f);
        const std::vector<uint8_t> w = make_rows(gt, n, rows, r, amp);
        const size_t wrow = ggml_row_size(gt, n), arow = ggml_row_size(tr->vec_dot_type, n);
        std::vector<float> xf((size_t) n);
        std::vector<uint8_t> act(arow * 3);
        for (int rep = 0; rep < 3; ++rep) {
            for (float& v : xf) v = r.gauss() * (rep == 0 ? 1.f : (rep == 1 ? 0.05f : 7.f));
            ta->from_float(xf.data(), act.data() + arow * rep, n);
        }
        Group& g = open_group("dot", ti->name, n);
        std::vector<float> want((size_t) 3 * rows);
        for (int rep = 0; rep < 3; ++rep)
            for (int row = 0; row < rows; ++row) {
                const uint8_t* wr = w.data() + (size_t) row * wrow;
                const uint8_t* ar_ = act.data() + arow * rep;
                float& o = want[(size_t) rep * rows + row];
                tr->vec_dot(n, &o, 0, wr, 0, ar_, 0, 1);
                const uint64_t rin = golden_fnv1a64(ar_, arow, golden_fnv1a64(wr, wrow));
                g.in = golden_fnv1a64(ar_, arow, golden_fnv1a64(wr, wrow, g.in));
                g.ref = golden_fnv1a64(&o, 4, g.ref);
                ++g.rows;
                dump_row(g, "ggml", rep * 1000 + row, rin, &o, 4);
            }
        std::vector<std::pair<const char*, uint64_t>> ours;
        for (size_t ki = 0; ki < run.size(); ++ki) {
            const sq::DotKernel* k = run[ki];
            uint64_t h = kFnvBasis;
            for (int rep = 0; rep < 3; ++rep)
                for (int row = 0; row < rows; ++row) {
                    const uint8_t* wr = w.data() + (size_t) row * wrow;
                    const uint8_t* ar_ = act.data() + arow * rep;
                    float got = 0.f;
                    k->fn(n, &got, wr, ar_);
                    h = golden_fnv1a64(&got, 4, h);
                    if (!g_dump_key.empty()) dump_row(g, k->name, rep * 1000 + row, golden_fnv1a64(ar_, arow, golden_fnv1a64(wr, wrow)), &got, 4);
                    const float wv = want[(size_t) rep * rows + row];
                    ++g_checked;
                    if (!bits_equal(wv, got) && live_bad[ki]++ < 1) {
                        char b[220];
                        std::snprintf(b, sizeof b, "kernel %s n=%d row %d: ggml %.9g (%08x) ours %.9g (%08x)", k->name, n, rep * 1000 + row, wv,
                                      float_bits(wv), got, float_bits(got));
                        live_diff(ti->name, b);
                    } else if (!bits_equal(wv, got)) {
                        live_diff(ti->name, std::string("kernel ") + k->name + " n=" + std::to_string(n) + " row " + std::to_string(rep * 1000 + row));
                    }
                }
            ours.emplace_back(k->name, h);
        }
        close_group(g, ours);
    }
    if (g_verbose)
        for (const sq::DotKernel* k : run) std::printf("  dot %-10s %-18s (%s/%s) done\n", ti->name, k->name, sq::arith_name(k->arith), sq::isa_name(k->isa));
    if (run.empty() && g_verbose) std::printf("  dot %-10s no kernel of contract %s runs here\n", ti->name, sq::arith_name(ar));
    if (run.empty() && nk == 0) fail(ti->name, std::string("the layer has no kernel of contract ") + sq::arith_name(ar));
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
        Group& g = open_group("act", ti->name, n);
        uint64_t h = kFnvBasis;
        bool reported = false;
        std::vector<float> x((size_t) n);
        std::vector<uint8_t> a(ti->bytes * nbk), b(ti->bytes * nbk);
        for (int it = 0; it < rows * 4; ++it) {
            for (float& v : x) v = r.gauss() * (it % 3 == 0 ? 1.f : (it % 3 == 1 ? 1e-3f : 300.f));
            if (it % 7 == 0) std::memset(x.data(), 0, x.size() * 4);   // an all-zero block
            if (it % 11 == 0) x[(size_t) (r.next() % n)] = 0.f;
            // ties: a block whose absolute maximum is 127 (id = 1) with half-integers elsewhere, and its negation
            if (it % 5 == 0) {
                // one draw per statement (see Rng::gauss): two r.next() in one expression gave MSVC other inputs than GCC
                for (int j = 0; j < n; ++j) {
                    const int q = (int) (r.next() % 255) - 127;
                    const float half = r.next() & 1 ? 0.f : 0.5f;
                    x[(size_t) j] = q * 0.5f + half;
                }
                for (int bI = 0; bI < nbk; ++bI) {
                    const uint32_t at = r.next() % ti->blck;
                    x[(size_t) bI * ti->blck + at] = r.next() & 1 ? 127.f : -127.f;
                }
                if (act == sq::Type::Q8_K) for (int bI = 0; bI < nbk; ++bI) x[(size_t) bI * ti->blck + r.next() % ti->blck] = 127.f;
            }
            // pre-filled: Q8_K leaves an all-zero block's 16 block sums unwritten (ggml and the layer alike); without this they carry an earlier row's bytes
            std::memset(a.data(), 0, a.size());
            std::memset(b.data(), 0, b.size());
            tr->from_float(x.data(), a.data(), n);
            k->fn(x.data(), b.data(), n);
            const uint64_t rin = golden_fnv1a64(x.data(), x.size() * 4);
            g.in = golden_fnv1a64(x.data(), x.size() * 4, g.in);
            g.ref = golden_fnv1a64(a.data(), a.size(), g.ref);
            h = golden_fnv1a64(b.data(), b.size(), h);
            ++g.rows;
            dump_row(g, "ggml", it, rin, a.data(), a.size());
            dump_row(g, k->name, it, rin, b.data(), b.size());
            ++g_checked;
            if (std::memcmp(a.data(), b.data(), a.size()) != 0 && !reported) {
                reported = true;
                size_t at = 0;
                while (a[at] == b[at]) ++at;
                char buf[160];
                std::snprintf(buf, sizeof buf, "kernel %s n=%d iteration %d: first differing byte %zu (ggml %02x ours %02x)", k->name, n, it, at, a[at], b[at]);
                live_diff(ti->name, buf);
            }
        }
        close_group(g, {{k->name, h}});
    }
    if (g_verbose) std::printf("  act %-10s %-18s done\n", ti->name, k->name);
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
            Group& g = open_group("deq", ti->name, n);
            uint64_t h = kFnvBasis;
            bool reported = false;
            std::vector<float> a((size_t) n), b((size_t) n);
            for (int row = 0; row < rows; ++row) {
                const uint8_t* wr = w.data() + (size_t) row * ggml_row_size(gt, n);
                tr->to_float(wr, a.data(), n);
                std::memset(b.data(), 0, b.size() * 4);
                const bool ok = sq::dequant_row(st, wr, b.data(), n);
                const uint64_t rin = golden_fnv1a64(wr, ggml_row_size(gt, n));
                g.in = golden_fnv1a64(wr, ggml_row_size(gt, n), g.in);
                g.ref = golden_fnv1a64(a.data(), a.size() * 4, g.ref);
                h = golden_fnv1a64(b.data(), b.size() * 4, h);
                ++g.rows;
                ++g_checked;
                dump_row(g, "ggml", row, rin, a.data(), a.size() * 4);
                dump_row(g, "dequant_row", row, rin, b.data(), b.size() * 4);
                if (!ok) {
                    fail(ti->name, "dequant_row refused n=" + std::to_string(n));
                    break;
                }
                if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0 && !reported) {
                    reported = true;
                    live_diff(ti->name, "dequant_row differs from ggml's to_float, n=" + std::to_string(n) + " row " + std::to_string(row));
                }
            }
            close_group(g, {{"dequant_row", h}});
        }
    }
    // Q4_0 from floats: ggml_quantize_chunk(Q4_0, no imatrix) is quantize_row_q4_0_ref
    Rng r(99);
    for (int nbk = 1; nbk <= 6; ++nbk) {
        const int n = 32 * nbk;
        Group& g = open_group("rq4", "q4_0", n);
        uint64_t h = kFnvBasis;
        bool reported = false;
        std::vector<float> f((size_t) n);
        std::vector<uint8_t> a(18 * nbk), b(18 * nbk);
        for (int it = 0; it < rows * 4; ++it) {
            for (float& v : f) v = r.gauss() * (it % 3 == 0 ? 1.f : (it % 3 == 1 ? 1e-3f : 200.f));
            if (it % 9 == 0) std::memset(f.data(), 0, f.size() * 4);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, f.data(), a.data(), 0, 1, n, nullptr);
            sq::quantize_q4_0_ref(f.data(), b.data(), n);
            const uint64_t rin = golden_fnv1a64(f.data(), f.size() * 4);
            g.in = golden_fnv1a64(f.data(), f.size() * 4, g.in);
            g.ref = golden_fnv1a64(a.data(), a.size(), g.ref);
            h = golden_fnv1a64(b.data(), b.size(), h);
            ++g.rows;
            dump_row(g, "ggml", it, rin, a.data(), a.size());
            dump_row(g, "quantize_q4_0_ref", it, rin, b.data(), b.size());
            ++g_checked;
            if (a != b && !reported) {
                reported = true;
                live_diff("q4_0", "quantize_q4_0_ref differs from ggml_quantize_chunk, n=" + std::to_string(n) + " iteration " + std::to_string(it));
            }
        }
        close_group(g, {{"quantize_q4_0_ref", h}});
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
        Group& g = open_group("bnd", "q8_K", 256);
        uint64_t h = kFnvBasis;
        bool reported = false;
        std::vector<float> x(256);
        std::vector<uint8_t> a(sq::kBlockQ8_K), b(sq::kBlockQ8_K);
        for (int it = 0; it < rows * 300; ++it) {
            const float m = 0.5f + 0.5f * std::fabs(r.uni());
            const int e = (int) (r.next() % 14) - 7;
            const float dd = std::ldexp(m, e);
            const float x0 = (r.next() & 1 ? 127.f : -127.f) * dd;
            const float step = -1.f / (-127.f / x0);
            x[0] = x0;
            for (int j = 1; j < 256; ++j) x[(size_t) j] = jitter(((int) (r.next() % 253) - 126 + 0.5f) * step);
            std::memset(a.data(), 0, a.size());
            std::memset(b.data(), 0, b.size());
            t8->from_float(x.data(), a.data(), 256);
            k8->fn(x.data(), b.data(), 256);
            const uint64_t rin = golden_fnv1a64(x.data(), x.size() * 4);
            g.in = golden_fnv1a64(x.data(), x.size() * 4, g.in);
            g.ref = golden_fnv1a64(a.data(), a.size(), g.ref);
            h = golden_fnv1a64(b.data(), b.size(), h);
            ++g.rows;
            dump_row(g, "ggml", it, rin, a.data(), a.size());
            dump_row(g, k8->name, it, rin, b.data(), b.size());
            ++g_checked;
            if (a != b && !reported) {
                reported = true;
                live_diff("q8_K", "boundary values: the quantizer differs from ggml's (a fused multiply-add on one side), iteration " + std::to_string(it));
            }
        }
        close_group(g, {{k8->name, h}});
    }
    Group& g = open_group("bnd", "q4_0", 32);
    uint64_t h = kFnvBasis;
    bool reported = false;
    std::vector<float> x(32);
    std::vector<uint8_t> a(18), b(18);
    for (int it = 0; it < rows * 300; ++it) {
        const float m = 0.5f + 0.5f * std::fabs(r.uni());
        const int e = (int) (r.next() % 14) - 7;
        const float dd = std::ldexp(m, e);
        x[0] = -8.f * dd;
        for (int j = 1; j < 32; ++j) x[(size_t) j] = jitter(((int) (r.next() % 15) + 1 - 8.5f) * dd);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, x.data(), a.data(), 0, 1, 32, nullptr);
        sq::quantize_q4_0_ref(x.data(), b.data(), 32);
        const uint64_t rin = golden_fnv1a64(x.data(), x.size() * 4);
        g.in = golden_fnv1a64(x.data(), x.size() * 4, g.in);
        g.ref = golden_fnv1a64(a.data(), a.size(), g.ref);
        h = golden_fnv1a64(b.data(), b.size(), h);
        ++g.rows;
        dump_row(g, "ggml", it, rin, a.data(), a.size());
        dump_row(g, "quantize_q4_0_ref", it, rin, b.data(), b.size());
        ++g_checked;
        if (a != b && !reported) {
            reported = true;
            live_diff("q4_0", "boundary values: quantize_q4_0_ref differs from ggml's (a fused multiply-add on one side), iteration " + std::to_string(it));
        }
    }
    close_group(g, {{"quantize_q4_0_ref", h}});
}

void test_fp16() {
    // our software fp16 conversions against ggml's, every fp16 value and a spread of floats (one group, two rows: the two directions)
    Group& g = open_group("fp16", "fp16", 0);
    int bad = 0;
    uint64_t h = kFnvBasis;
    for (uint32_t hv = 0; hv < 65536; ++hv) {
        const uint16_t u = (uint16_t) hv;
        const float a = ggml_fp16_to_fp32((ggml_fp16_t) u);
        const float b = sq::fp16_to_fp32(u);
        g.in = golden_fnv1a64(&u, 2, g.in);
        g.ref = golden_fnv1a64(&a, 4, g.ref);
        h = golden_fnv1a64(&b, 4, h);
        if (std::isnan(a) ? !std::isnan(b) : !bits_equal(a, b)) ++bad;
    }
    ++g.rows;
    Rng r(5);
    for (int i = 0; i < 4000000; ++i) {
        uint32_t w = r.next();
        float f;
        std::memcpy(&f, &w, 4);
        if (std::isnan(f)) continue;
        const uint16_t a = ggml_fp32_to_fp16(f), b = sq::fp32_to_fp16(f);
        g.in = golden_fnv1a64(&f, 4, g.in);
        g.ref = golden_fnv1a64(&a, 2, g.ref);
        h = golden_fnv1a64(&b, 2, h);
        if (a != b) ++bad;
    }
    ++g.rows;
    ++g_checked;
    if (bad) live_diff("fp16", std::to_string(bad) + " conversions differ from ggml's");
    close_group(g, {{"fp16", h}});
}

std::string golden_lines_of_this_run() {
    std::string s;
    for (const Group& g : g_groups) s += g.key + " " + std::to_string(g.rows) + " " + hex64(g.in) + " " + hex64(g.ref) + "\n";
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    int rows = kGoldenRows;
    const char* only = nullptr;
    std::string golden_path = STRATA_QUANT_GOLDEN_FILE, write_golden;
    bool regen_check = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) rows = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--type") && i + 1 < argc) only = argv[++i];
        else if (!std::strcmp(argv[i], "--verbose")) g_verbose = true;
        else if (!std::strcmp(argv[i], "--golden") && i + 1 < argc) golden_path = argv[++i];
        else if (!std::strcmp(argv[i], "--dump-group") && i + 1 < argc) g_dump_key = argv[++i];
        else if (!std::strcmp(argv[i], "--write-golden") && i + 1 < argc) write_golden = argv[++i];
        else if (!std::strcmp(argv[i], "--regen-check")) regen_check = true;
        else {
            std::printf("quant_parity: unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    if (!digest_selftest()) {
        std::printf("quant_parity: FAIL: the golden digest (FNV-1a 64) does not give its test vectors\n");
        return 1;
    }
    if (regen_check && !kRegenHere) {
        std::printf("quant_parity --regen-check: SKIPPED: the golden lines are made by GCC 13.x (family gcc) and MSVC 19.4x (family msvc) on x86; this is %s\n",
                    compiler_name().c_str());
        return 77;
    }
    ggml_cpu_init();
    const sq::Arith ar = oracle_arith();
    g_flavour = sq::arith_name(ar);
    std::printf("quant_parity: oracle flavour %s (ggml avx=%d avx2=%d fma=%d f16c=%d avx512=%d), layer best isa %s, default contract %s, compiler %s\n",
                sq::arith_name(ar), ggml_cpu_has_avx(), ggml_cpu_has_avx2(), ggml_cpu_has_fma(), ggml_cpu_has_f16c(), ggml_cpu_has_avx512(),
                sq::isa_name(sq::best_isa()), sq::arith_name(sq::default_arith()), compiler_name().c_str());
#if defined(__aarch64__)
    // The Neon contract is ggml-cpu's arm64 flavour with the dot product extension (the shipped STRATA_PORTABLE build and the Mac's native build).  A ggml-cpu without it
    // (GCC's -mcpu=native on the Spark's mixed X925 + A725 cores resolves to plain armv8-a) runs other code, and one with SVE other branches of the Q4_0 / Q8_0 /
    // K-quant dots: those flavours are not the contract.  The first is skipped, the second is run for information and its differences do not fail the test.
    if (!ggml_cpu_has_dotprod()) {
        std::printf("quant_parity: SKIPPED, this ggml-cpu is built without the dot product extension (not the Neon contract)\n");
        return 0;
    }
    if (ggml_cpu_has_sve()) {
        g_info_only = true;
        std::printf("quant_parity: this ggml-cpu is built with SVE (not the Neon contract): differences are reported, not failed\n");
    }
#endif
    const bool have_golden = !golden_path.empty() && load_golden(golden_path, g_flavour) && !g_golden.empty();
    g_golden_leg = have_golden && rows == kGoldenRows && write_golden.empty();
    const char* strict_env = std::getenv("QUANT_PARITY_LIVE_STRICT");
    g_live_strict = !g_golden_leg || (strict_env != nullptr && std::strcmp(strict_env, "0") != 0 && *strict_env != '\0');
    if (regen_check && !have_golden) {
        std::printf("quant_parity --regen-check: FAIL: no golden lines for flavour %s in %s\n", g_flavour.c_str(), golden_path.c_str());
        return 1;
    }
    if (regen_check && rows != kGoldenRows) {
        std::printf("quant_parity --regen-check: FAIL: the golden file holds --rows %d\n", kGoldenRows);
        return 1;
    }
    if (g_golden_leg) {
        std::printf("quant_parity: leg 1 (gate): ours against the golden reference, %zu groups of flavour %s (%s); leg 2 (%s): ours against the ggml-cpu linked here\n",
                    g_golden.size(), g_flavour.c_str(), golden_path.c_str(), g_live_strict ? "FATAL, QUANT_PARITY_LIVE_STRICT" : "information");
        if (!g_golden_own.empty()) {
            std::printf("quant_parity: toolchain family %s has its own golden lines for %zu groups of flavour %s:", kFamily, g_golden_own.size(), g_flavour.c_str());
            for (const auto& kv : g_golden_own) std::printf(" %s;", kv.first.c_str());
            std::printf("\n");
            for (const std::string& n : g_notes) std::printf("quant_parity:   why: %s\n", n.c_str());
        }
    }
    else
        std::printf("quant_parity: no golden reference for flavour %s%s: the gate is ours against the ggml-cpu linked here\n", g_flavour.c_str(),
                    have_golden ? " at this --rows" : (golden_path.empty() ? "" : (" in " + golden_path).c_str()));
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

    const std::string mine = golden_lines_of_this_run();
    if (!write_golden.empty()) {
        if (only != nullptr || rows != kGoldenRows) {
            std::printf("quant_parity --write-golden: refused with --type or --rows (the golden file holds every group at --rows %d)\n", kGoldenRows);
            return 2;
        }
        FILE* f = std::fopen(write_golden.c_str(), "wb");
        const std::string head = "# made by " + compiler_name() + ", flavour " + g_flavour + ", " + std::to_string(g_groups.size()) + " groups\n";
        if (f == nullptr || std::fwrite(head.data(), 1, head.size(), f) != head.size() || std::fwrite(mine.data(), 1, mine.size(), f) != mine.size() ||
            std::fclose(f) != 0) {
            std::printf("quant_parity --write-golden: cannot write %s\n", write_golden.c_str());
            return 2;
        }
        std::printf("quant_parity --write-golden: %zu groups of flavour %s written to %s (%s)\n", g_groups.size(), g_flavour.c_str(), write_golden.c_str(),
                    compiler_name().c_str());
        return 0;
    }
    if (regen_check) {
        if (only != nullptr) {
            std::printf("quant_parity --regen-check: refused with --type\n");
            return 2;
        }
        std::string stored;
        if (std::strcmp(kFamily, "gcc") == 0) {
            for (const std::string& k : g_golden_gcc_order) stored += g_golden[k].text + "\n";
        } else {
            // this family's own lines must regenerate, and every group without one must equal its gcc line (else MSVC differs in a new group: it needs a line)
            int own = 0;
            for (const Group& g : g_groups) {
                const GoldenLine* gl = golden_for(g.key);
                own += g_golden_own.count(g.key) ? 1 : 0;
                stored += (gl ? gl->text : std::string("(no golden line for ") + g.key + ")") + "\n";
            }
            if (own != (int) g_golden_own.size()) {
                std::printf("quant_parity --regen-check: FAIL: %zu %s lines of flavour %s stored, %d met\n", g_golden_own.size(), kFamily, g_flavour.c_str(), own);
                return 1;
            }
        }
        if (stored == mine) {
            std::printf("quant_parity --regen-check: PASS: %s regenerates the stored lines of flavour %s byte for byte (family %s: %zu own lines; %zu groups)\n",
                        compiler_name().c_str(), g_flavour.c_str(), kFamily, std::strcmp(kFamily, "gcc") == 0 ? g_golden.size() : g_golden_own.size(),
                        g_groups.size());
            return 0;
        }
        std::istringstream a(stored), b(mine);
        std::string la, lb;
        int ln = 0;
        while (true) {
            const bool ha = (bool) std::getline(a, la), hb = (bool) std::getline(b, lb);
            ++ln;
            if (!ha && !hb) break;
            if (!ha || !hb || la != lb) {
                std::printf("quant_parity --regen-check: FAIL at flavour line %d:\n  stored:      %s\n  regenerated: %s\n", ln, ha ? la.c_str() : "(end)",
                            hb ? lb.c_str() : "(end)");
                break;
            }
        }
        return 1;
    }
    if (g_golden_leg && only == nullptr)
        for (const auto& kv : g_golden)   // every stored group of this flavour was met: a case that vanished does not pass silently
            if (g_groups_by_key.find(kv.first) == g_groups_by_key.end()) fail("golden", "group \"" + kv.first + "\" of the golden file was not run");
    if (g_golden_leg) {
        std::printf("quant_parity: leg 2 (%s): %d rows differ from the ggml-cpu linked here (%s); that ggml differs from the golden reference in %zu groups%s",
                    g_live_strict ? "FATAL" : "information", g_live_diff, compiler_name().c_str(), g_ref_moved.size(), g_ref_moved.empty() ? "\n" : ":");
        for (size_t i = 0; i < g_ref_moved.size(); ++i) std::printf("%s %s", i ? "," : "", g_ref_moved[i].c_str());
        if (!g_ref_moved.empty()) std::printf("\n");
    }
    std::printf("quant_parity: %d comparisons, %d golden group checks, %d failures -> %s\n", g_checked, g_group_checks, g_fail, g_fail ? (g_info_only ? "DIFFERENCES (information only)" : "FAIL") : "PASS");
    return (g_fail && !g_info_only) ? 1 : 0;
}
