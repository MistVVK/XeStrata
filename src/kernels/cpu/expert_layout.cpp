// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/cpu/expert_layout.cpp - plan v0.3 P6: the per-layer expert table.  See the header.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/platform/env.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <fstream>
#include <sstream>

namespace strata::kernels::cpu {
namespace {
ExpertLayout g_layout;
}

const ExpertLayout& expert_layout() { return g_layout; }

void expert_layout_set(const ExpertLayout& layout) { g_layout = layout; }

bool cpu_avx512_ok() {
    static const bool ok = [] {
        if (const char* f = strata::env_get("FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1], ecx = r[2];
        return ((ebx >> 16) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u) && ((ecx >> 11) & 1u) && ((ecx >> 1) & 1u);
    }();
    return ok;
}

bool cpu_avx2_ok() {
    static const bool ok = [] {
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        const unsigned ecx1 = r[2];
        // FMA (12), OSXSAVE (27), AVX (28), F16C (29)
        if (!((ecx1 >> 12) & 1u) || !((ecx1 >> 27) & 1u) || !((ecx1 >> 28) & 1u) || !((ecx1 >> 29) & 1u)) return false;
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0x6) != 0x6) return false;            // the OS saves the SSE and AVX state
        cpuid(7, 0);
        return ((r[1] >> 5) & 1u) != 0;                    // AVX2
    }();
    return ok;
}

bool cpu_avxvnni_ok() {
#if defined(STRATA_HAVE_AVXVNNI)
    static const bool ok = [] {
        if (const char* e = strata::env_get("NO_AVXVNNI"); e != nullptr && e[0] == '1') return false;
        if (!cpu_avx2_ok()) return false;
        unsigned a = 0, b = 0, c = 0, d = 0;
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            a = (unsigned) x[0], b = (unsigned) x[1], c = (unsigned) x[2], d = (unsigned) x[3];
#else
            __cpuid_count(leaf, sub, a, b, c, d);
#endif
        };
        cpuid(0, 0);
        if (a < 7) return false;
        cpuid(7, 0);
        if (a < 1) return false;                           // no sub-leaf 1
        cpuid(7, 1);
        return ((a >> 4) & 1u) != 0;                       // AVX-VNNI
    }();
    return ok;
#else
    return false;
#endif
}

namespace {
void cpuid_regs(unsigned leaf, unsigned sub, unsigned r[4]) {
#if defined(_MSC_VER)
    int x[4];
    __cpuidex(x, (int) leaf, (int) sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

unsigned long long xcr0_bits() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long) hi << 32) | lo;
#endif
}
}  // namespace

bool cpu_avx512vnni_ok() {
    static const bool ok = [] {
        if (const char* f = strata::env_get("FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        unsigned r[4] = {0, 0, 0, 0};
        cpuid_regs(0, 0, r);
        if (r[0] < 7) return false;
        cpuid_regs(1, 0, r);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
        if ((xcr0_bits() & 0xE6) != 0xE6) return false;      // the OS saves the SSE, AVX, opmask and ZMM state
        cpuid_regs(7, 0, r);
        return ((r[1] >> 16) & 1u) && ((r[2] >> 11) & 1u);  // AVX512F, AVX512_VNNI
    }();
    return ok;
}

bool cpu_avxvnniint8_ok() {
    static const bool ok = [] {
        if (const char* e = strata::env_get("NO_AVXVNNI"); e != nullptr && e[0] == '1') return false;
        if (!cpu_avx2_ok()) return false;
        unsigned r[4] = {0, 0, 0, 0};
        cpuid_regs(0, 0, r);
        if (r[0] < 7) return false;
        cpuid_regs(7, 0, r);
        if (r[0] < 1) return false;                         // no sub-leaf 1
        cpuid_regs(7, 1, r);
        return ((r[3] >> 4) & 1u) != 0;                     // AVX-VNNI-INT8
    }();
    return ok;
}

uint64_t cpu_l2_bytes() {
    static const uint64_t bytes = [] {
        unsigned r[4] = {0, 0, 0, 0};
        cpuid_regs(0, 0, r);
        const unsigned max_leaf = r[0];
        cpuid_regs(0x80000000u, 0, r);
        const unsigned max_ext = r[0];
        // the deterministic cache parameters: Intel's leaf 4, AMD's 0x8000001D (the same layout)
        unsigned leaf = 0;
        if (max_leaf >= 4) leaf = 4;
        cpuid_regs(0, 0, r);
        const bool amd = r[1] == 0x68747541u;   // "Auth"enticAMD
        if (amd) leaf = max_ext >= 0x8000001Du ? 0x8000001Du : 0;
        if (leaf == 0) return (uint64_t) 0;
        for (unsigned sub = 0; sub < 16; ++sub) {
            cpuid_regs(leaf, sub, r);
            const unsigned type = r[0] & 31u, level = (r[0] >> 5) & 7u;
            if (type == 0) break;
            if (level == 2 && (type == 2 || type == 3)) {   // the data or unified L2
                const uint64_t ways = ((r[1] >> 22) & 1023u) + 1, parts = ((r[1] >> 12) & 1023u) + 1;
                const uint64_t line = (r[1] & 4095u) + 1, sets = (uint64_t) r[2] + 1;
                return ways * parts * line * sets;
            }
        }
        return (uint64_t) 0;
    }();
    return bytes;
}

int iq256_gather_setting() {
    static const int s = [] {
        const char* v = strata::env_get("IQ256_GATHER");
        if (v == nullptr || v[0] == '\0') return -1;
        return std::strtol(v, nullptr, 10) != 0 ? 1 : 0;
    }();
    return s;
}

bool cpu_gather_fast() {
    // From the feature bits CPUID reports, never a vendor string or a model (AGENTS.md): AVX-VNNI (Alder Lake,
    // Sapphire Rapids and later; the older cores gather slowly or under the Downfall microcode) on a hybrid CPU
    // (7.0:EDX[15]), whose performance cores cpu_gather_fast_here tells apart.  A CPU of one core type is not told
    // apart from an E-core-only part this way, so it keeps the scalar decode (STRATA_IQ256_GATHER=1 turns it on).
    static const bool ok = [] {
        if (!cpu_avxvnni_ok()) return false;
        unsigned r[4];
        cpuid_regs(7, 0, r);
        return ((r[3] >> 15) & 1u) != 0;                               // hybrid
    }();
    return ok;
}

bool cpu_gather_fast_here() {
    if (!cpu_gather_fast()) return false;
    static const unsigned max_leaf = [] { unsigned r[4]; cpuid_regs(0, 0, r); return r[0]; }();
    // A CPUID can cost a microsecond under a hypervisor, so once per thread (the pool pins its workers)
    thread_local int here = -1;
    if (here < 0) {
        unsigned type = 0;
        if (max_leaf >= 0x1A) {
            unsigned r[4];
            cpuid_regs(0x1A, 0, r);
            type = r[0] >> 24;
        }
        here = type == 0x40 ? 1 : 0;   // 40h a performance core, 20h an E-core
    }
    return here == 1;
}

std::string cpu_name() {
    unsigned r[12] = {};
#if defined(_MSC_VER)
    int x[4];
    __cpuid(x, (int) 0x80000000u);
    if ((unsigned) x[0] < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) {
        __cpuid(x, (int) (0x80000002u + i));
        for (int j = 0; j < 4; ++j) r[i * 4 + j] = (unsigned) x[j];
    }
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid(0x80000000u, a, b, c, d);
    if (a < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) __cpuid(0x80000002u + i, r[(size_t) i * 4], r[(size_t) i * 4 + 1], r[(size_t) i * 4 + 2], r[(size_t) i * 4 + 3]);
#endif
    char s[49] = {};
    std::memcpy(s, r, 48);
    std::string name(s);
    const size_t b0 = name.find_first_not_of(' '), b1 = name.find_last_not_of(' ');
    return b0 == std::string::npos ? std::string("unknown") : name.substr(b0, b1 - b0 + 1);
}

// ---- the AVX-512 probe and the oracle flag (#795) ----
//
// expert.cpp and iq_avx512.cpp are compiled for AVX-512, and a TU compiled that way may use AVX-512 in ANY of its
// code.  The code that decides whether this CPU has AVX-512 - and the startup flag every CPU, AVX2-only ones
// included, runs before that decision - therefore lives here, in a file compiled for the x86-64 baseline.
namespace {
std::atomic<bool> g_oracle_q8_0{false};
}

void expert_set_oracle_q8_0(bool enabled) { g_oracle_q8_0.store(enabled, std::memory_order_relaxed); }

bool expert_oracle_q8_0_enabled() { return g_oracle_q8_0.load(std::memory_order_relaxed); }

const char* CpuFeatures::reason() const {
    if (usable()) return "ok";
    // Named individually: "AVX-512 not supported" sends a user looking for a new CPU when the machine may have
    // AVX-512F and be missing only VNNI, which is a much narrower and more explicable gap.
    static char buf[160];
    std::snprintf(buf, sizeof buf, "missing %s%s%s%s%s", avx512f ? "" : "AVX512F ",
                  avx512bw ? "" : "AVX512BW ", avx512vl ? "" : "AVX512VL ",
                  avx512_vnni ? "" : "AVX512-VNNI ", avx512_vbmi ? "" : "AVX512-VBMI");
    return buf;
}

CpuFeatures cpu_features() {
    CpuFeatures f;
    int reg[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    __cpuid(reg, 0);
    if (reg[0] < 7) return f;
    __cpuidex(reg, 7, 0);
#else
    unsigned r[4] = {0, 0, 0, 0};
    __cpuid_count(0, 0, r[0], r[1], r[2], r[3]);
    if (r[0] < 7) return f;
    __cpuid_count(7, 0, r[0], r[1], r[2], r[3]);
    for (int i = 0; i < 4; ++i) reg[i] = (int) r[i];
#endif
    const unsigned ebx = (unsigned) reg[1], ecx = (unsigned) reg[2];
    f.avx512f = (ebx >> 16) & 1u;
    f.avx512bw = (ebx >> 30) & 1u;
    f.avx512vl = (ebx >> 31) & 1u;
    f.avx512_vnni = (ecx >> 11) & 1u;
    f.avx512_vbmi = (ecx >> 1) & 1u;
    return f;
}

void cpu_require_expert_support() {
    const CpuFeatures f = cpu_features();
    if (f.usable()) return;
    std::fprintf(stderr,
                 "xestrata: this CPU cannot run the expert kernel: %s.\n"
                 "        The pack in --pack is the AVX-512 Q2_0 pack (its folder has no native_experts.txt): its CPU kernel needs\n"
                 "        AVX512-VNNI and AVX512-VBMI (Intel Ice Lake / AMD Zen 4 or newer).\n"
                 "        An i-quant (IQ3_XXS, IQ3_S, ...) or Unsloth pack has native_experts.txt in its folder and runs on an\n"
                 "        AVX2 CPU as well: build that pack (setup does it on a CPU without AVX-512) and give its folder as --pack.\n"
                 "        The scalar fallback exists for tests only and is far too slow to decode with.\n",
                 f.reason());
    std::exit(1);
}

void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1) {
    if (cpu_avx512_ok()) q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
    else q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void act_quant_any(const float* x, int n, ActQ& a) {
    if (cpu_avx512_ok()) act_quant_q8_1(x, n, a);
    else act_quant_q8_1_avx2(x, n, a);
}

#if !defined(STRATA_NATIVE_EXPERTS)
// Without ggml-cpu no native pack loads (expert_layout_load refuses), so these are never reached.
bool native_experts_available() noexcept { return false; }
bool native_fmt(int, int, int64_t, int64_t, NativeFmt&, std::string& err, int) { err = "built without native experts"; return false; }
void native_quant_act(const NativeFmt&, const float*, void*) { std::abort(); }
void native_quant_h(const NativeFmt&, const float*, void*) { std::abort(); }
int native_gu_mt_min(int) { return 1; }
void native_gu_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
void native_down_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
bool native_row_to_float(int, const void*, float*, int64_t) { return false; }
size_t native_row_bytes(int, int64_t) { return 0; }
#endif

bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    ExpertLayout L;
    L.n_layers = n_layers;
    L.n_expert = n_expert;
    std::ifstream in(pack_dir + "/native_experts.txt");
    if (!in) {
        L.total = (uint64_t) n_layers * (uint64_t) n_expert * (uint64_t) BLOB;
        hadamard_set({});
        g_layout = L;
        return true;
    }
#if !defined(STRATA_NATIVE_EXPERTS)
    err = "this pack has native (IQ) experts but the engine was built without STRATA_NATIVE_EXPERTS";
    return false;
#else
    L.native = true;
    L.fmt.resize((size_t) n_layers);
    L.offset.assign((size_t) n_layers, ~0ull);
    L.bytes.assign((size_t) n_layers, 0);
    L.max_blob = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            if (!line.empty() && line[0] == '#') {
                // The version: upstream Strata's "# strata native experts vN" (v2-v4 known) or XeStrata's own
                // "# xestrata native experts xsN" (xs1), numbered apart so neither is taken for the other.  A
                // version this engine does not know may mean columns it would misread, so it is refused.
                static const char up_tag[] = "# strata native experts v", xs_tag[] = "# xestrata native experts xs";
                const bool up = line.compare(0, sizeof up_tag - 1, up_tag) == 0;
                const bool xs = line.compare(0, sizeof xs_tag - 1, xs_tag) == 0;
                if (up || xs) {
                    const char* num = line.c_str() + (up ? sizeof up_tag : sizeof xs_tag) - 1;
                    const int v = (int) std::strtol(num, nullptr, 10);
                    // xs2: xs1's lines with Hadamard-folded experts (hadamard.txt), what upstream's v5 says
                    L.version = up ? v : v >= 2 ? 5 : 4;
                    if (v > (up ? 5 : 2)) {
                        err = std::string("native_experts.txt is ") + (up ? "v" : "xs") + std::to_string(v) +
                              "; this engine reads upstream's v2-v5 and XeStrata's xs1-xs2 (repack with this "
                              "checkout's tools/iq_pack.py)";
                        return false;
                    }
                }
                // v3 packs record their expert count in the header; a pruned model (GSQ-RCO Coder) ships
                // fewer experts than the canonical geometry the caller passes, which is a compile-time
                // default, so the header wins.
                const size_t at = line.find("(n_expert ");
                if (at != std::string::npos) L.n_expert = std::atoll(line.c_str() + at + 10);
            }
            continue;
        }
        std::istringstream ss(line);
        long long l = -1, gt = -1, dt = -1;
        unsigned long long off = 0, blob = 0, go = 0, uo = 0, dox = 0;
        if (!(ss >> l >> gt >> dt >> off >> blob) || l < 0 || l >= n_layers) {
            err = "native_experts.txt: a malformed line: " + line;
            return false;
        }
        NativeFmt f;
        if (!native_fmt((int) gt, (int) dt, H, FF, f, err)) return false;
        if (f.bytes != blob) {
            err = "native_experts.txt: layer " + std::to_string(l) + " blob is " + std::to_string(blob) +
                  " B but its formats make " + std::to_string(f.bytes);
            return false;
        }
        if (ss >> go >> uo >> dox) {   // v2 lines: the GGUF offsets
            if (L.gguf_off.empty()) L.gguf_off.assign((size_t) (3 * n_layers), 0);
            L.gguf_off[(size_t) (3 * l)] = go;
            L.gguf_off[(size_t) (3 * l + 1)] = uo;
            L.gguf_off[(size_t) (3 * l + 2)] = dox;
            // The shards (file names beside --native): one for the layer, or one per role as "gate,up,down" (an
            // empty field for the --native shard: upstream's v4 and XeStrata's xs1), or as three names apart with
            // "-" for it (XeStrata's packs before xs1, which said v4).  A GGUF name has no comma or space.
            std::vector<std::string> files;
            for (std::string file; ss >> file;) files.push_back(file);
            if (files.size() == 1 && files[0].find(',') != std::string::npos) {
                const std::string col = files[0];
                files.clear();
                for (size_t from = 0;;) {
                    const size_t comma = col.find(',', from);
                    files.push_back(col.substr(from, comma == std::string::npos ? comma : comma - from));
                    if (comma == std::string::npos) break;
                    from = comma + 1;
                }
            }
            for (auto& f : files)
                if (f == "-") f.clear();
            if (files.size() == 1) {
                const std::string one = files[0];
                files.assign(3, one);
            }
            if (!files.empty() && files.size() != 3) {
                err = "native_experts.txt: layer " + std::to_string(l) + ": the shard column is one name, "
                      "gate,up,down or three names, not: " + line;
                return false;
            }
            if (!files.empty()) {
                if (L.gguf_file.empty()) L.gguf_file.assign((size_t) (3 * n_layers), std::string());
                for (int r = 0; r < 3; ++r) L.gguf_file[(size_t) (3 * l + r)] = files[(size_t) r];
            }
        }
        L.fmt[(size_t) l] = f;
        L.offset[(size_t) l] = off;
        L.bytes[(size_t) l] = blob;
        if (blob > L.max_blob) L.max_blob = blob;
    }
    uint64_t at = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        if (L.offset[(size_t) l] != at) {
            err = "native_experts.txt: layer " + std::to_string(l) + " is missing or not contiguous";
            return false;
        }
        at += L.bytes[(size_t) l] * (uint64_t) L.n_expert;
    }
    L.total = at;
    // v5: Agention's Hadamard-folded experts (prism.hadamard, hadamard.hpp).  The pack must say which layers are
    // rotated and with what signs; a v5 pack without hadamard.txt (or the reverse) is refused, never run unrotated.
    {
        std::ifstream probe(pack_dir + "/hadamard.txt");
        const bool have = (bool) probe;
        if (L.version >= 5 && !have) {
            err = "native_experts.txt v5 declares Hadamard-folded experts, but " + pack_dir + "/hadamard.txt is missing";
            return false;
        }
        if (have && L.version < 5) {
            err = pack_dir + "/hadamard.txt exists but native_experts.txt is v" + std::to_string(L.version) +
                  ": the pack is inconsistent (repack with tools/iq_pack.py)";
            return false;
        }
        HadamardSpec h;
        if (have && !hadamard_parse(pack_dir + "/hadamard.txt", n_layers, h, err)) return false;
        hadamard_set(std::move(h));
        const HadamardSpec& hs = hadamard_spec();
        for (int64_t l = 0; hs.any() && l < n_layers; ++l) {
            NativeFmt& f = L.fmt[(size_t) l];
            f.had_block = hs.block;
            f.had_gu = hs.gu(l);
            f.had_d = hs.down(l);
            if ((f.had_gu && f.n_embd % hs.block) || (f.had_d && f.n_ff % hs.block)) {
                err = "hadamard.txt: block " + std::to_string(hs.block) + " does not divide layer " +
                      std::to_string(l) + "'s expert widths";
                return false;
            }
            // the Q2_0 rows quantize their activations with Strata's own act_quant_any, which does not rotate
            if ((f.had_gu && f.gu_type == 42) || (f.had_d && f.d_type == 42)) {
                err = "hadamard.txt: layer " + std::to_string(l) + " folds a Q2_0 expert, which this engine cannot rotate";
                return false;
            }
            f.gu_signs = f.had_gu ? hadamard_signs((int) f.n_embd) : nullptr;
            f.d_signs = f.had_d ? hadamard_signs((int) f.n_ff) : nullptr;
            // explicit signs are per width: a width without a vector would silently mean identity signs
            if (!hs.signs.empty() && ((f.had_gu && !f.gu_signs) || (f.had_d && !f.d_signs))) {
                err = "hadamard.txt: no sign vector for layer " + std::to_string(l) + "'s expert width";
                return false;
            }
        }
    }
    g_layout = L;
    return true;
#endif
}

}  // namespace strata::kernels::cpu
