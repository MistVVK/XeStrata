// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/cpu/expert_parity.cpp - P2.S3's test for the CPU expert path.
//
// WHAT IS BEING CHECKED, and the two tolerances are different questions:
//
//   1. VNNI vs the scalar oracle with `quant_acts = true`.  Both sides consume the SAME INT8 activation
//      values, so the only difference is FP32 evaluation ORDER.  This is the check that proves the KERNEL -
//      the VNNI unpack, the 32-element chunk split, the scalar correction term - and P2.S3 asks for 3e-3.
//   2. VNNI vs the oracle with `quant_acts = false`.  Now the oracle uses exact FP32 activations, so the gap
//      IS the activation contract (`docs/activation-contract.md`).  It is REPORTED, not asserted tightly,
//      because it is a property of the model's quantisation and not of this code.
//
// The activation contract is asserted OBSERVABLE between the two: if rounding the activation to Q8_1 made no
// difference, then one of the two paths is not doing what it claims and check 1 would be vacuous.
//
// A SELF-CONSISTENT TEST CANNOT CATCH A WRONG FILE OFFSET.  Both sides read the same blob, so pointing at the
// wrong expert would still agree perfectly.  What is checked instead is structural: the blob must decode to
// finite, non-zero scales, and the blob at a DIFFERENT layer must be different data - which is what a wrong
// stride would break.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "synth_q2_blob.hpp"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

/// Normalised L1 difference, with the reference magnitude returned for reporting.
double rel_l1(const float* a, const float* b, int n, double* mag_out = nullptr) {
    double d = 0, mag = 0;
    for (int i = 0; i < n; ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        mag += std::fabs((double) a[i]);
    }
    if (mag_out) *mag_out = mag / n;
    return d / (mag > 1e-30 ? mag : 1e-30);
}

bool read_blob(const char* path, long long index, std::vector<uint8_t>& out) {
    out.assign(cpu::BLOB, 0);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
#if defined(_MSC_VER)
    if (_fseeki64(f, index * (long long) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#else
    if (fseeko(f, (off_t) index * (off_t) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#endif
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

/// The fp16 scale at a byte offset, decoded by the SAME rule the kernel uses.  Used only for the structural
/// sanity check, so it is deliberately a second, independent transcription of the three lines in question.
float f16_at(const std::vector<uint8_t>& b, size_t off) {
    uint16_t h;
    std::memcpy(&h, b.data() + off, 2);
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) f = sign << 31;
        else { exp = 127 - 15 + 1; while (!(man & 0x400u)) { man <<= 1; --exp; } man &= 0x3FFu;
               f = (sign << 31) | (exp << 23) | (man << 13); }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float o;
    std::memcpy(&o, &f, 4);
    return o;
}

// Different K and blob sizes in one pool batch must use each job's own format.
int check_bih3_pool() {
    using namespace strata::kernels::cpu;
    constexpr int ne = 2, nt = 3, h = 256, ff = 128;
    NativeFmt fmt[ne];
    float p2[3][4];
    for (auto& table : p2) for (float& v : table) v = 1.f;
    std::string err;
    if (!native_r1q2_fmt(182, 183, h, ff, p2, fmt[0], err) ||
        !native_r1q2_fmt(190, 191, h, ff, p2, fmt[1], err)) {
        std::printf("bih3 pool: %s\n", err.c_str()); return 1;
    }
    std::mt19937 rng(5167);   // NOLINT(bugprone-random-generator-seed): repeatable fixture
    std::vector<uint8_t> blobs[ne], acts[ne][nt], hq[ne][nt];
    std::vector<float> expected[ne][nt], result[ne][nt], intermediate[ne][nt];
    ExpertJobMulti jobs[ne] = {};
    auto scales = [](const NativeFmt& f, const uint8_t* blob, int m) {
        const size_t nr = (size_t) (m < 2 ? f.n_ff : f.n_embd);
        const size_t nb = (size_t) (m < 2 ? f.n_embd : f.n_ff) / 128;
        const uint8_t* p = blob + f.sc_off + native_r1q2_scale_offset(f, m);
        return TqR1q2Scales{p, p + nr * 2, p + nr * 2 + nb * 2, (nb + 3) / 4,
                           {f.p2[m][0], f.p2[m][1], f.p2[m][2], f.p2[m][3]}};
    };
    for (int e = 0; e < ne; ++e) {
        const NativeFmt& f = fmt[e];
        blobs[e].resize(f.bytes);
        for (uint8_t& v : blobs[e]) v = (uint8_t) rng();
        for (int m = 0; m < 3; ++m) {
            const size_t nr = (size_t) (m < 2 ? f.n_ff : f.n_embd);
            const size_t nb = (size_t) (m < 2 ? f.n_embd : f.n_ff) / 128;
            uint8_t* p = blobs[e].data() + f.sc_off + native_r1q2_scale_offset(f, m);
            const uint16_t value = 0x3000;   // FP16 0.125
            for (size_t i = 0; i < nr + nb; ++i) std::memcpy(p + 2 * i, &value, 2);
        }
        jobs[e].blob = blobs[e].data(); jobs[e].fmt = &fmt[e]; jobs[e].nt = nt;
        for (int t = 0; t < nt; ++t) {
            acts[e][t].resize(kNativeActBytes); hq[e][t].resize(kNativeHBytes);
            expected[e][t].resize(h); result[e][t].resize(h); intermediate[e][t].resize(ff);
            std::vector<float> x(h);
            for (float& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.f;
            native_quant_act(f, x.data(), acts[e][t].data());
            jobs[e].nact[t] = acts[e][t].data(); jobs[e].out[t] = result[e][t].data();
            const void* a[] = {acts[e][t].data()};
            float* mid[] = {intermediate[e][t].data()};
            tq_bih3_r1q2_gu_rows_isa(Bih3Isa::scalar, f.gu_type, blobs[e].data(), f.gu_row, f.up_off, h,
                                     a, 1, mid, 0, ff, scales(f, blobs[e].data(), 0), scales(f, blobs[e].data(), 1));
            native_quant_h(f, mid[0], hq[e][t].data());
            const void* a2[] = {hq[e][t].data()}; float* out[] = {expected[e][t].data()};
            tq_bih3_r1q2_rows_isa(Bih3Isa::scalar, f.d_type, blobs[e].data() + f.down_off, f.d_row, ff,
                                  a2, 1, out, 0, h, scales(f, blobs[e].data(), 2));
        }
    }
    ExpertPool pool(2, false, true);
    pool.run_split_multi_native(fmt[0], jobs, ne);
    int failures = 0;
    for (int e = 0; e < ne; ++e) for (int t = 0; t < nt; ++t)
        for (int i = 0; i < h; ++i)
            if (std::bit_cast<uint32_t>(expected[e][t][(size_t) i]) != std::bit_cast<uint32_t>(result[e][t][(size_t) i]))
                ++failures;
    if (pool.multi_bytes != (int64_t) (fmt[0].bytes + fmt[1].bytes)) ++failures;
    std::printf("bih3 pool: 2 formats x 3 tokens x 256 rows: %d failures\n", failures);
    return failures;
}

}  // namespace

// A malformed argument must be an error, not a silently accepted zero.
static bool parse_i64(const char* s, long long& out) {
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = v;
    return true;
}

int main(int argc, char** argv) {
    bool selftest = false;
    const char* path = "pack/full/experts.bin";
    bool path_given = false;
    long long layer = 0, expert = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--file" && i + 1 < argc) { path = argv[++i]; path_given = true; }
        else if (a == "--layer" && i + 1 < argc) {
            if (!parse_i64(argv[++i], layer)) { std::fprintf(stderr, "expert_parity: invalid --layer value\n"); return 2; }
        }
        else if (a == "--expert" && i + 1 < argc) {
            if (!parse_i64(argv[++i], expert)) { std::fprintf(stderr, "expert_parity: invalid --expert value\n"); return 2; }
        }
        else { std::fprintf(stderr, "usage: expert_parity [--selftest] [--file P] [--layer N] [--expert N]\n");
               return 2; }
    }

    int bad = check_bih3_pool();

    // ---- the CPU must be able to run the kernel, and the report must say WHICH feature is missing
    const cpu::CpuFeatures feat = cpu::cpu_features();
    std::printf("  %-44s %s\n", "AVX512 F/BW/VL/VNNI/VBMI",
                feat.usable() ? "all present" : feat.reason());
    if (!feat.usable()) {
        std::printf("      the VNNI path cannot run here; only the scalar oracle can be exercised.\n");
        std::printf("      P2.S3's kernel check is SKIPPED, not passed.\n");
        std::printf("\nexpert_parity: %d failures, 1 SKIPPED\n", bad);
        return bad ? 1 : 0;
    }

    // the pack's Q2_0 experts, or (no --file and no pack/full here) seeded blobs of the same form
    bool synth = false;
    if (!path_given) {
        if (std::FILE* f = std::fopen(path, "rb")) std::fclose(f);
        else synth = true;
    }
    if (synth) std::printf("  %-44s %s\n", "experts", "pack/full/experts.bin is not here: seeded Q2_0 blobs");
    auto blob_at = [&](long long index, std::vector<uint8_t>& out) {
        if (!synth) return read_blob(path, index, out);
        cpu::test::synth_q2_blob(index, out);
        return true;
    };
    // ---- a real blob out of the pack: layer stride is 512 experts
    const long long idx = layer * 512 + expert;
    std::vector<uint8_t> blob;
    if (!blob_at(idx, blob)) {
        std::fprintf(stderr, "cannot read expert %lld (layer %lld) from %s\n", expert, layer, path);
        return 2;
    }

    // ---- structural sanity, because the agreement checks below are self-consistent and would pass on the
    //      wrong blob.  A mis-strided read lands on a different expert's data, which is still plausible Q2_0,
    //      so the check that has teeth is the DIFFERENCE between layers.
    {
        int nonzero = 0;
        float smin = 1e30f, smax = -1e30f;
        for (int b = 0; b < cpu::SC_GU; ++b) {
            const float s = f16_at(blob, cpu::O_GU_SCALES + (size_t) b * 2);
            if (s != 0.f) ++nonzero;
            smin = std::fmin(smin, std::fabs(s));
            smax = std::fmax(smax, std::fabs(s));
        }
        const bool ok = nonzero > cpu::SC_GU / 2 && std::isfinite(smin) && std::isfinite(smax) && smax > 0.f;
        std::printf("  %-44s %s (%d/%d non-zero, |s| in [%.4g, %.4g])\n", "blob decodes to live scales",
                    ok ? "yes" : "*** NO ***", nonzero, cpu::SC_GU, (double) smin, (double) smax);
        if (!ok) ++bad;
    }
    {
        std::vector<uint8_t> other;
        if (blob_at((layer * 512 + expert) + 1, other)) {
            long long diff = 0;
            for (size_t i = 0; i < blob.size(); ++i) if (blob[i] != other[i]) ++diff;
            const double frac = (double) diff / (double) blob.size();
            const bool ok = frac > 0.1;
            std::printf("  %-44s %s (%.1f%% of bytes differ)\n", "the next expert is different data",
                        ok ? "yes" : "*** NO ***", frac * 100);
            if (!ok) ++bad;
        }
    }

    // ---- a realistic activation
    std::mt19937 rng(4242);  // NOLINT(bugprone-random-generator-seed): reproducible inputs
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(cpu::H);
    for (auto& v : x) v = gauss(rng);

    cpu::ExpertScratch ws;
    std::vector<float> vnni(cpu::H), scalar_q(cpu::H), scalar_fp(cpu::H);
    cpu::s2_expert_vnni(blob.data(), x.data(), vnni.data(), ws);
    cpu::s2_expert_scalar(blob.data(), x.data(), scalar_q.data(), true);
    cpu::s2_expert_scalar(blob.data(), x.data(), scalar_fp.data(), false);

    double mag = 0;
    const double rel_kernel = rel_l1(scalar_q.data(), vnni.data(), cpu::H, &mag);
    const double rel_contract = rel_l1(scalar_fp.data(), vnni.data(), cpu::H);
    const double trap_contract = rel_l1(scalar_fp.data(), scalar_q.data(), cpu::H);

    // ---- the activation contract must be OBSERVABLE, or check 1 is vacuous
    {
        const bool visible = trap_contract > 1e-4;
        std::printf("  %-44s %s (%.4f%% apart)\n", "the Q8_1 activation contract is observable",
                    visible ? "yes" : "*** NO ***", trap_contract * 100);
        if (!visible) ++bad;
    }

    // ---- 1. the kernel, against an oracle fed the SAME activations
    std::printf("  %-44s rel %.3e   (P2.S3 asks <= 3e-3, mean |ref| %.4f)\n", "VNNI vs scalar (INT8 acts)",
                rel_kernel, mag);
    if (!(rel_kernel <= 3e-3)) { std::printf("    *** over P2.S3's 3e-3 ***\n"); ++bad; }

    // ---- 2. the activation contract's own cost, reported rather than asserted
    std::printf("  %-44s rel %.3e   (the cost of the contract, REPORTED)\n",
                "VNNI vs scalar (FP32 acts)", rel_contract);

    std::printf("\nexpert_parity: %d failures (layer %lld expert %lld of %s)\n", bad, layer, expert, path);
    if (bad) return 1;
    if (selftest) std::printf("expert_parity OK\n");
    return 0;
}
