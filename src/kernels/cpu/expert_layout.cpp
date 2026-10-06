// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/expert_layout.cpp - plan v0.3 P6: the per-layer expert table.  See the header.
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cpuid.h>
#include <fstream>
#include <sstream>

namespace strata::kernels::cpu {
namespace {
ExpertLayout g_layout;
}

const ExpertLayout& expert_layout() { return g_layout; }

bool cpu_avx512_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
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
        if (const char* e = std::getenv("STRATA_NO_AVXVNNI"); e != nullptr && e[0] == '1') return false;
        if (!cpu_avx2_ok()) return false;
        unsigned a = 0, b = 0, c = 0, d = 0;
        __cpuid_count(0, 0, a, b, c, d);
        if (a < 7) return false;
        __cpuid_count(7, 0, a, b, c, d);
        if (a < 1) return false;                           // no sub-leaf 1
        __cpuid_count(7, 1, a, b, c, d);
        return ((a >> 4) & 1u) != 0;                       // AVX-VNNI
    }();
    return ok;
#else
    return false;
#endif
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

void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1) {
    if (cpu_avx512_ok()) q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
#if defined(STRATA_HAVE_AVXVNNI)
    else if (cpu_avxvnni_ok()) q2_0_gguf_rows_multi_avxvnni(w, row_bytes, nblocks, a, nt, out, r0, r1);
#endif
    else q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void act_quant_any(const float* x, int n, ActQ& a) {
    if (cpu_avx512_ok()) act_quant_q8_1(x, n, a);
    else act_quant_q8_1_avx2(x, n, a);
}

#if !defined(STRATA_NATIVE_EXPERTS)
// Without ggml-cpu no native pack loads (expert_layout_load refuses), so these are never reached.
bool native_experts_available() noexcept { return false; }
bool native_fmt(int, int, int64_t, int64_t, NativeFmt&, std::string& err) { err = "built without native experts"; return false; }
void native_quant_act(const NativeFmt&, const float*, void*) { std::abort(); }
void native_quant_h(const NativeFmt&, const float*, void*) { std::abort(); }
void native_gu_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
void native_down_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
#endif

bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    ExpertLayout L;
    L.n_layers = n_layers;
    L.n_expert = n_expert;
    std::ifstream in(pack_dir + "/native_experts.txt");
    if (!in) {
        L.total = (uint64_t) n_layers * (uint64_t) n_expert * (uint64_t) BLOB;
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
                    L.version = up ? v : 4;
                    if (v > (up ? 4 : 1)) {
                        err = std::string("native_experts.txt is ") + (up ? "v" : "xs") + std::to_string(v) +
                              "; this engine reads upstream's v2-v4 and XeStrata's xs1 (repack with this "
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
    g_layout = L;
    return true;
#endif
}

}  // namespace strata::kernels::cpu
