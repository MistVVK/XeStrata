// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/native_expert_parity.cpp - plan v0.3 P6: one native expert three ways.
//
//     build/native_expert_parity <shard.gguf> [layer ...]     real rows (a split model's shards are found by name)
//     build/native_expert_parity --synthetic GU/DOWN[@HxFF]   random weights quantized by ggml (e.g. q4_K/q5_1,
//                                                             q6_K/q6_K@2048x512); 2560x640 when no shape is given
//     build/native_expert_parity --q5_1-min                   the Q5_1 min term on crafted activations
//
// (a) float reference: ggml's own dequantizer (`to_float`) and a float SwiGLU expert, (b) the CPU path
// (ggml-cpu vec_dot with its quantized activations), (c) the GPU path (`native_expert_grouped`, q8_1
// activations).  (b) and (c) each differ from (a) by their activation rounding only (a few 1e-3 relative).  (d) a layer
// of the canonical rows (canon_expert.hpp): the CPU's rows and the GPU's are the same bits.
//
//     build/native_expert_parity --bf16-embd
//
// The BF16 token embedding (--embd-gguf) on a random table, no model needed (in CTest).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/canon_expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "ggml-cpu.h"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/nvfp4.hpp"
#include "strata/platform/env.hpp"

#include "ggml.h"

#include "parity_device.hpp"

#include <array>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

// The BF16 token embedding (--embd-gguf): embed_rows and iq_dequant_f32 on a random BF16 table against the exact
// widening (bits << 16), every bit; rows gathered out of order, with repeats
static int check_bf16_embd(void* s) {
    constexpr int kBf16 = 30;
    const int64_t H = 2560, V = 61, NT = 97;
    if (!strata::kernels::embed_type_supported(kBf16) || strata::kernels::iq_supported(kBf16) ||
        strata::kernels::iq_row_bytes(kBf16, H) != (size_t) H * 2) {
        std::printf("bf16 embedding: type support / row bytes wrong\n");
        return 1;
    }
    uint32_t state = 290;   // xorshift32: the same table on every run
    auto rng = [&state] { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    std::vector<uint16_t> table((size_t) (V * H));
    for (auto& v : table) {   // any bit pattern but NaN (a NaN's payload is not what the test is about)
        do v = (uint16_t) (rng() & 0xffff); while ((v & 0x7f80) == 0x7f80 && (v & 0x7f));
    }
    std::vector<int32_t> tok((size_t) NT);
    for (auto& t : tok) t = (int32_t) (rng() % V);
    strata::parity::Device dev;
    const uint16_t* dt = dev.upload(table);
    const int32_t* dtok = dev.upload(tok);
    const int64_t out_rows = NT > V ? NT : V;   // the gathered rows (NT) and the whole table (V) share the buffer
    float* dout = dev.alloc<float>((size_t) (out_rows * H));
    auto widen = [](uint16_t b) { return (uint32_t) b << 16; };   // the exact float bits
    auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    std::vector<float> got((size_t) (out_rows * H));
    strata::kernels::EmbedTable et;
    et.type = kBf16;
    et.rows = dt;
    et.row_bytes = (size_t) H * 2;
    et.n_rows = V;
    et.n_embd = H;
    strata::kernels::embed_rows(et, dtok, 0, NT, dout, s);
    strata::parity::copy_out(got.data(), dout, (size_t) (NT * H));
    size_t rows_differ = 0;
    for (int64_t r = 0; r < NT; ++r)
        for (int64_t d = 0; d < H; ++d) {
            if (bits(got[(size_t) (r * H + d)]) != widen(table[(size_t) (tok[(size_t) r] * H + d)])) {
                ++rows_differ;
                break;
            }
        }
    strata::kernels::iq_dequant_f32(kBf16, dt, V * H, dout, s);
    strata::parity::copy_out(got.data(), dout, table.size());
    size_t values_differ = 0;
    for (size_t i = 0; i < table.size(); ++i) values_differ += bits(got[i]) != widen(table[i]);
    std::printf("bf16 embedding: %lld gathered rows, %zu differ; dequant of %lld values, %zu differ in any bit\n",
                (long long) NT, rows_differ, (long long) V * H, values_differ);
    return rows_differ || values_differ ? 1 : 0;
}

// One expert three ways (see the top): the float reference, the CPU and the GPU, on `blob` (gate rows, up rows, down
// rows in the formats of `f`) and NT random activations; prints one line, `label` and `l` first.  Returns its failures.
static int check_blob(const char* label, int l, const std::vector<uint8_t>& blob, const cpu::NativeFmt& f, void* s) {
    const int NT = 3;
    const int64_t H = f.n_embd, FF = f.n_ff;
    int failures = 0;
    // upstream bfbe7fab2: the float reference below is ggml's, and this ggml's type table ends at GGML_TYPE_COUNT -
    // llama.cpp's own ggml has no trellis types (144-149, which XeStrata's own rows handle).  Asking it for one of
    // those reads past the table (on Windows: garbage), so the check is skipped rather than read past it.
    if (f.gu_type >= (int) GGML_TYPE_COUNT || f.d_type >= (int) GGML_TYPE_COUNT) {
        std::printf("%s %d: the expert types %d/%d are past this ggml's type table (it has no host traits for the "
                    "trellis types): skipped\n", label, l, f.gu_type, f.d_type);
        return 0;
    }
    // (a) the float reference
    const auto* tg = ggml_get_type_traits((ggml_type) f.gu_type);
    const auto* td = ggml_get_type_traits((ggml_type) f.d_type);
    std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
    for (int64_t r = 0; r < FF; ++r) {
        tg->to_float(blob.data() + r * f.gu_row, G.data() + r * H, H);
        tg->to_float(blob.data() + f.up_off + r * f.gu_row, U.data() + r * H, H);
    }
    for (int64_t r = 0; r < H; ++r) td->to_float(blob.data() + f.down_off + r * f.d_row, D.data() + r * FF, FF);
    std::mt19937 rng(11 + l);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NT * H);
    for (auto& v : x) v = nd(rng);
    std::vector<float> ref((size_t) NT * H), got_c((size_t) NT * H), got_g((size_t) NT * H);
    // APR (--synthetic-hadamard, upstream fbf045eb6): the folded expert's reference takes H (s * x) and H (s * h), as
    // llama.cpp's graph does
    std::vector<float> xr(x);
    if (f.had_gu) for (int k = 0; k < NT; ++k) cpu::hadamard_rotate(xr.data() + k * H, (int) H, f.had_block, f.gu_signs);
    // NVFP4: the expert's s_gate, s_up and s_down from the blob's tail (1 for every other type)
    const double sg = strata::kernels::nvfp4::tail_scale(blob.data(), f.tail_off, 0);
    const double su = strata::kernels::nvfp4::tail_scale(blob.data(), f.tail_off, 1);
    const double sd = strata::kernels::nvfp4::tail_scale(blob.data(), f.tail_off, 2);
    for (int k = 0; k < NT; ++k) {
        std::vector<float> h(FF);
        for (int64_t r = 0; r < FF; ++r) {
            double g = 0, u = 0;
            for (int64_t i = 0; i < H; ++i) { g += (double) G[r * H + i] * xr[k * H + i]; u += (double) U[r * H + i] * xr[k * H + i]; }
            g *= sg;
            u *= su;
            h[r] = (float) (g / (1.0 + std::exp(-g)) * u);
        }
        if (f.had_d) cpu::hadamard_rotate(h.data(), (int) FF, f.had_block, f.d_signs);
        for (int64_t r = 0; r < H; ++r) {
            double o = 0;
            for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[i];
            ref[k * H + r] = (float) (o * sd);
        }
    }
    // (b) the CPU
    {
        std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes));
        std::vector<std::vector<uint8_t>> hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
        std::vector<std::vector<float>> ff(NT, std::vector<float>(FF));
        const void* a[NT];
        float* ffp[NT];
        const void* hp[NT];
        float* op[NT];
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_act(f, x.data() + k * H, act[k].data());
            a[k] = act[k].data();
            ffp[k] = ff[k].data();
        }
        cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
        // the kernels below are checked against ggml's vec_dot on ggml's activations (vec_dot_type); a layer of the
        // canonical rows (canon_expert.hpp) quantizes its own form for native_gu_rows, so these get ggml's
        const bool canon = cpu::canon_layer(f);
        std::vector<std::vector<uint8_t>> actl(NT, std::vector<uint8_t>(cpu::kNativeActBytes));
        const void* al[NT];
        for (int k = 0; k < NT; ++k) {
            if (canon) ggml_get_type_traits_cpu((ggml_type) f.gu_act)->from_float(x.data() + k * H, actl[k].data(), H);
            al[k] = canon ? actl[k].data() : a[k];
        }
        // either multi-token kernel: IQ4_XS has an AVX-2 one and no AVX-512 one, so the gate cannot be
        // iq512_supported alone - that would leave the format untested on every CPU.
        if (cpu::iq512_supported(f.gu_type) || cpu::iq256_supported(f.gu_type)) {
            // ggml's own vec_dot, same Q8_K activations: the reference for both multi-token kernels
            // (float-order differences only)
            const auto* tc = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
            std::vector<float> gref((size_t) NT * FF);
            for (int k = 0; k < NT; ++k)
                for (int64_t r = 0; r < FF; ++r)
                    tc->vec_dot((int) H, &gref[k * FF + r], 0, blob.data() + r * f.gu_row, 0, al[k], 0, 1);
            double usg = 0.0;
            {   // ggml single-token throughput, the weights cache-resident (compute bound)
                const int it = 50;
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i)
                    for (int64_t r = 0; r < FF; ++r) {
                        float gg, uu;
                        tc->vec_dot((int) H, &gg, 0, blob.data() + r * f.gu_row, 0, al[0], 0, 1);
                        tc->vec_dot((int) H, &uu, 0, blob.data() + f.up_off + r * f.gu_row, 0, al[0], 0, 1);
                    }
                auto t1 = std::chrono::steady_clock::now();
                usg = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
            }
            // one multi-token kernel: parity against ggml's vec_dot, then one-token and NT-token throughput
            auto check = [&](const char* tag, bool is512) {
                std::vector<float> g((size_t) NT * FF);
                float* gp[NT];
                for (int k = 0; k < NT; ++k) gp[k] = g.data() + k * FF;
                (is512 ? cpu::iq512_rows : cpu::iq256_rows)(f.gu_type, blob.data(), f.gu_row, (int) H, al, NT, gp, 0, (int) FF);
                const double rg = rel(g, gref);
                std::printf("          %s %s gate rows vs ggml vec_dot: rel %.2e\n",
                            ggml_type_name((ggml_type) f.gu_type), tag, rg);
                if (rg > 1e-5) {   // ggml's own vec_dot is the reference: above 1e-5 it is a real mismatch,
                    std::printf("          %s %s gate rows MISMATCH (rel %.2e > 1e-5)\n",
                                ggml_type_name((ggml_type) f.gu_type), tag, rg);   // not rounding (measured 3e-8)
                    ++failures;
                }
                std::vector<float> fsb((size_t) NT * FF);   // the timed calls' output, apart from ff (read below)
                float* fs[NT];
                for (int k = 0; k < NT; ++k) fs[k] = fsb.data() + k * FF;
                float* f1[1] = {fs[0]};
                const int it = 50;
                const auto gu = is512 ? cpu::iq512_gu_rows : cpu::iq256_gu_rows;
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i) gu(f.gu_type, blob.data(), f.gu_row, f.up_off, (int) H, al, 1, f1, 0, (int) FF);
                auto t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i) gu(f.gu_type, blob.data(), f.gu_row, f.up_off, (int) H, al, NT, fs, 0, (int) FF);
                auto t2 = std::chrono::steady_clock::now();
                const double us1 = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
                const double usn = std::chrono::duration<double, std::micro>(t2 - t1).count() / it;
                std::printf("          gate+up one token one thread: %s %.0f us (%.2f GB/s), ggml %.0f us (%.2f GB/s)\n",
                            tag, us1, 2.0 * (double) f.up_off / us1 / 1e3, usg, 2.0 * (double) f.up_off / usg / 1e3);
                std::printf("          gate+up %d tokens one thread: %s %.0f us vs ggml %d x %.0f us\n", NT, tag, usn, NT, usg);
            };
            // guarded: the binary runs on AVX-2 CPUs too, and IQ4_XS has no AVX-512 kernel (an empty switch)
            if (cpu::cpu_avx512_ok() && cpu::iq512_supported(f.gu_type)) check("avx512", true);
            if (cpu::iq256_supported(f.gu_type)) check("avx2", false);
        }
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_h(f, ff[k].data(), hq[k].data());
            hp[k] = hq[k].data();
            op[k] = got_c.data() + k * H;
        }
        cpu::native_down_rows(f, blob.data(), hp, NT, op, 0, (int) H);
        std::vector<std::vector<uint8_t>> hql(NT, std::vector<uint8_t>(cpu::kNativeHBytes));   // ggml's, as above
        const void* hl[NT];
        for (int k = 0; k < NT; ++k) {
            if (canon) ggml_get_type_traits_cpu((ggml_type) f.d_act)->from_float(ff[k].data(), hql[k].data(), FF);
            hl[k] = canon ? hql[k].data() : hp[k];
        }
        if (f.d_type == 42) {
            // (b2) the GGUF-layout Q2_0 kernel the pool uses for Q2_0 down projections - the AVX-512 one
            // where the CPU has it, the AVX-2 one (q2_avx2.cpp) where it does not.  Calling the AVX-512
            // kernel unconditionally faults on a Zen 2/3 CPU.
            std::vector<cpu::ActQ> a2(NT);
            const cpu::ActQ* ap[NT];
            std::vector<float> alt((size_t) NT * H);
            float* altp[NT];
            for (int k = 0; k < NT; ++k) {
                cpu::act_quant_any(ff[k].data(), (int) FF, a2[k]);   // gated: AVX-512 or AVX-2 per CPU
                ap[k] = &a2[k];
                altp[k] = alt.data() + k * H;
            }
            cpu::q2_rows_any(blob.data() + f.down_off, f.d_row, (int) (FF / 64), ap, NT, altp, 0, (int) H);
            std::printf("          q2_0 %s down vs ggml down: rel %.2e\n",
                        cpu::cpu_avx512_ok() ? "AVX-512" : "AVX-2", rel(alt, got_c));
        }
        if (f.d_type == 20) {
            // (b3) the IQ4_NL multi-token AVX-2 kernel the pool now uses for IQ4_NL down projections,
            // against ggml-cpu's single-token vec_dot on the SAME Q8_0 activations (h), plus timing.
            std::vector<float> alt((size_t) NT * H), refd((size_t) NT * H);
            float* altp[NT];
            for (int k = 0; k < NT; ++k) altp[k] = alt.data() + k * H;
            cpu::iq4nl256_down_rows(blob.data() + f.down_off, f.d_row, (int) FF, hl, NT, altp, 0, (int) H);
            const auto* tdc = ggml_get_type_traits_cpu((ggml_type) f.d_type);
            for (int k = 0; k < NT; ++k)
                for (int64_t r = 0; r < H; ++r)
                    tdc->vec_dot((int) FF, refd.data() + k * H + r, 0,
                                 blob.data() + f.down_off + (size_t) r * f.d_row, 0, hl[k], 0, 1);
            const double rdn = rel(alt, refd);
            std::printf("          iq4_nl AVX-2 multi-token down vs ggml down: rel %.2e\n", rdn);
            if (rdn > 1e-5) {   // measured 1e-7; same reasoning as the gate/up rows above
                std::printf("          iq4_nl down MISMATCH (rel %.2e > 1e-5)\n", rdn);
                ++failures;
            }
            const int it = 50;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i)
                cpu::iq4nl256_down_rows(blob.data() + f.down_off, f.d_row, (int) FF, hl, NT, altp, 0, (int) H);
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i)
                for (int k = 0; k < NT; ++k)
                    for (int64_t r = 0; r < H; ++r)
                        tdc->vec_dot((int) FF, refd.data() + k * H + r, 0,
                                    blob.data() + f.down_off + (size_t) r * f.d_row, 0, hl[k], 0, 1);
            const auto t2 = std::chrono::steady_clock::now();
            const double usn = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
            const double usg = std::chrono::duration<double, std::micro>(t2 - t1).count() / it;
            std::printf("          down %d tokens one thread: avx2 %.0f us (%.2f GB/s) vs ggml %.0f us (%.2f GB/s)\n",
                        NT, usn, (double) (f.bytes - f.down_off) / usn / 1e3, usg,
                        (double) (f.bytes - f.down_off) / usg / 1e3);
        }
        // (b4) the K-quant and Q8_0 multi-token kernels (kq256_*, q80_256_*), the AVX2 copy and the AVX-VNNI one where
        // the CPU has it, against ggml-cpu's vec_dot on the same activations: gate rows and down rows, rel <= 1e-5
        auto kq = [](int t) { return cpu::kq256_supported(t) || t == 8; };
        if (cpu::cpu_avx2_ok() && (kq(f.gu_type) || kq(f.d_type))) {
            std::vector<int> variants = {0};
            if (cpu::iq256_variants() & cpu::kIq256Vnni) variants.push_back(cpu::kIq256Vnni);
            auto rows_vs_ggml = [&](const char* what, int type, const uint8_t* w, size_t row_bytes, int64_t n,
                                    int64_t rows, const void* const* acts) {
                const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
                std::vector<float> want((size_t) (NT * rows)), got((size_t) (NT * rows));
                float* gp[NT];
                for (int k = 0; k < NT; ++k) {
                    gp[k] = got.data() + k * rows;
                    for (int64_t r = 0; r < rows; ++r)
                        tc->vec_dot((int) n, &want[(size_t) (k * rows + r)], 0, w + (size_t) r * row_bytes, 0, acts[k], 0, 1);
                }
                for (int v : variants) {
                    if (type == 8) cpu::q80_256_rows_v(v, w, row_bytes, (int) n, acts, NT, gp, 0, (int) rows);
                    else cpu::kq256_rows_v(v, type, w, row_bytes, (int) n, acts, NT, gp, 0, (int) rows);
                    const double rr = rel(got, want);
                    const char* tag = v ? "avx2+vnni" : "avx2";
                    std::printf("          %s %s %s rows vs ggml vec_dot: rel %.2e\n", ggml_type_name((ggml_type) type),
                                tag, what, rr);
                    if (rr > 1e-5) {   // float-order differences only, as for the i-quants above
                        std::printf("          %s %s %s rows MISMATCH (rel %.2e > 1e-5)\n",
                                    ggml_type_name((ggml_type) type), tag, what, rr);
                        ++failures;
                    }
                }
            };
            if (kq(f.gu_type)) rows_vs_ggml("gate", f.gu_type, blob.data(), f.gu_row, H, FF, a);
            if (kq(f.d_type)) rows_vs_ggml("down", f.d_type, blob.data() + f.down_off, f.d_row, FF, H, hp);
        }
    }
    // (c) the GPU: one group holding the NT entries
    {
        auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF, f.up_type);
        if (f.had_d) { L.had_block = f.had_block; L.d_signs = strata::kernels::hadamard_device_signs(f.d_signs, (int) FF); }
        namespace par = strata::parity;
        void* dblob = par::alloc_bytes(blob.size());
        void* dx = par::alloc_bytes(x.size() * 4);
        void* dxq = par::alloc_bytes((size_t) NT * H / 32 * 36);
        void* dscr = par::alloc_bytes(strata::kernels::native_expert_scratch_bytes(NT, FF));
        auto* dout = static_cast<float*>(par::alloc_bytes((size_t) NT * H * 4));
        auto* dptr = static_cast<unsigned long long*>(par::alloc_bytes(8));
        auto* dstart = static_cast<int32_t*>(par::alloc_bytes(8));
        auto* dn = static_cast<int32_t*>(par::alloc_bytes(4));
        auto* ddst = static_cast<int32_t*>(par::alloc_bytes((size_t) NT * 4));
        auto* dtok = static_cast<int32_t*>(par::alloc_bytes((size_t) NT * 4));
        par::copy_bytes_in(dblob, blob.data(), blob.size());
        par::copy_bytes_in(dx, x.data(), x.size() * 4);
        const unsigned long long p = (unsigned long long) dblob;
        const int32_t st[2] = {0, NT}, one = 1, idx[NT] = {0, 1, 2};
        par::copy_bytes_in(dptr, &p, 8);
        par::copy_bytes_in(dstart, st, 8);
        par::copy_bytes_in(dn, &one, 4);
        par::copy_bytes_in(ddst, idx, (size_t) NT * 4);
        par::copy_bytes_in(dtok, idx, (size_t) NT * 4);
        if (f.gu_type == 30) {   // BF16 experts read the FP32 rows
            strata::kernels::native_expert_grouped_bf16(L, dptr, dstart, dn, ddst, dtok, 1, NT, (const float*) dx, dscr,
                                                        dout, s);
        } else {
            if (f.had_gu) {   // APR: the GPU rotation of x (in place here), then its floats against the CPU's
                const float* sx = strata::kernels::hadamard_device_signs(f.gu_signs, (int) H);
                strata::kernels::hadamard_rows((const float*) dx, (float*) dx, NT, H, f.had_block, sx, s);
                std::vector<float> gx(x.size());
                par::sync();
                par::copy_bytes_out(gx.data(), dx, gx.size() * 4);
                const bool same = std::memcmp(gx.data(), xr.data(), gx.size() * 4) == 0;
                std::printf("          gpu hadamard_rows vs cpu hadamard_rotate: %s (rel %.2e)\n",
                            same ? "bitwise equal" : "differ", rel(gx, xr));
                if (!(rel(gx, xr) < 1e-6)) ++failures;
                // the f16 form (the prompt path) on the unrotated x: within fp16 rounding of the float rotation
                std::vector<uint16_t> xh(x.size()), xh_out(x.size());
                for (size_t i = 0; i < x.size(); ++i) xh[i] = ggml_fp32_to_fp16(x[i]);
                auto* dxh = static_cast<uint16_t*>(par::alloc_bytes(xh.size() * 2));
                par::copy_bytes_in(dxh, xh.data(), xh.size() * 2);
                strata::kernels::hadamard_rows_f16(dxh, NT, H, f.had_block, sx, s);
                par::sync();
                par::copy_bytes_out(xh_out.data(), dxh, xh.size() * 2);
                std::vector<float> xf(x.size());
                for (size_t i = 0; i < xf.size(); ++i) xf[i] = ggml_fp16_to_fp32(xh_out[i]);
                std::printf("          gpu hadamard_rows_f16 vs cpu float rotation: rel %.2e\n", rel(xf, xr));
                if (!(rel(xf, xr) < 2e-3)) ++failures;
            }
            strata::kernels::quantize_q8_1_rows((const float*) dx, NT, H, dxq, s);
            strata::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, dscr, dout, s);
        }
        par::sync();
        par::copy_bytes_out(got_g.data(), dout, got_g.size() * 4);
        // STRATA_GROUPED_BENCH (measuring): a decode token's call, 8 groups of one entry, each a different expert out
        // of 64 copies of this one (past the GPU's last-level cache), in rotation; the time a call and the weight
        // bytes it reads a microsecond
        if (strata::env_get("GROUPED_BENCH") != nullptr && f.gu_type != 30) {
            constexpr int kCopies = 64, kG = 8, kSets = kCopies / kG, kIters = 200;
            auto* many = static_cast<uint8_t*>(par::alloc_bytes(blob.size() * kCopies));
            for (int c = 0; c < kCopies; ++c) par::copy_bytes_in(many + blob.size() * c, blob.data(), blob.size());
            std::vector<unsigned long long> ptrs((size_t) kSets * kG);
            for (int i = 0; i < kSets * kG; ++i) ptrs[(size_t) i] = (unsigned long long) (many + blob.size() * i);
            auto* dptrs = static_cast<unsigned long long*>(par::alloc_bytes(ptrs.size() * 8));
            par::copy_bytes_in(dptrs, ptrs.data(), ptrs.size() * 8);
            int32_t st[kG + 1], zero[kG], ent[kG];
            for (int g = 0; g <= kG; ++g) st[g] = g;
            for (int g = 0; g < kG; ++g) { zero[g] = 0; ent[g] = g; }
            const int32_t ng = kG;
            auto* bst = static_cast<int32_t*>(par::alloc_bytes(sizeof st));
            auto* bn = static_cast<int32_t*>(par::alloc_bytes(4));
            auto* bdst = static_cast<int32_t*>(par::alloc_bytes(sizeof ent));
            auto* btok = static_cast<int32_t*>(par::alloc_bytes(sizeof zero));
            void* bscr = par::alloc_bytes(strata::kernels::native_expert_scratch_bytes(kG, FF));
            auto* bout = static_cast<float*>(par::alloc_bytes((size_t) kG * H * 4));
            par::copy_bytes_in(bst, st, sizeof st);
            par::copy_bytes_in(bn, &ng, 4);
            par::copy_bytes_in(bdst, ent, sizeof ent);
            par::copy_bytes_in(btok, zero, sizeof zero);
            auto call = [&](int i) {
                strata::kernels::native_expert_grouped(L, dptrs + (size_t) (i % kSets) * kG, bst, bn, bdst, btok, kG, kG,
                                                       dxq, bscr, bout, s);
            };
            for (int i = 0; i < kSets; ++i) call(i);
            par::sync();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) call(i);
            par::sync();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kIters;
            std::printf("          grouped bench: %d experts of %zu bytes a call, %.1f us, %.1f GB/s\n", kG, blob.size(), us,
                        (double) blob.size() * kG / us / 1e3);
        }
    }
    // (d) the CPU's canonical rows (canon_expert.hpp) against the GPU's grouped experts: the same bits, the NT tokens at
    // once and one at a time (an expert's rows on either side of a verify window, #152)
    if (cpu::canon_layer(f)) {
        std::vector<float> got_k((size_t) NT * H), ffb((size_t) NT * FF);
        std::vector<cpu::NativeAct> aq((size_t) NT);
        std::vector<std::array<uint8_t, cpu::kNativeHBytes>> hq((size_t) NT);
        std::vector<const void*> acts((size_t) NT), hqs((size_t) NT);
        std::vector<float*> ffp((size_t) NT), outp((size_t) NT);
        for (int64_t t = 0; t < NT; ++t) {
            cpu::native_quant_act(f, x.data() + t * H, aq[(size_t) t].bytes);
            acts[(size_t) t] = aq[(size_t) t].bytes;
            ffp[(size_t) t] = ffb.data() + t * FF;
        }
        for (int one = 0; one < 2; ++one) {   // the NT tokens at once, then one at a time
            if (one) for (int64_t t = 0; t < NT; ++t) cpu::native_gu_rows(f, blob.data(), &acts[(size_t) t], 1, &ffp[(size_t) t], 0, (int) FF);
            else cpu::native_gu_rows(f, blob.data(), acts.data(), (int) NT, ffp.data(), 0, (int) FF);
            for (int64_t t = 0; t < NT; ++t) {
                cpu::native_quant_h(f, ffp[(size_t) t], hq[(size_t) t].data());
                hqs[(size_t) t] = hq[(size_t) t].data();
                outp[(size_t) t] = got_k.data() + t * H;
            }
            if (one) for (int64_t t = 0; t < NT; ++t) cpu::native_down_rows(f, blob.data(), &hqs[(size_t) t], 1, &outp[(size_t) t], 0, (int) H);
            else cpu::native_down_rows(f, blob.data(), hqs.data(), (int) NT, outp.data(), 0, (int) H);
            size_t diff = 0;
            double mx = 0;
            for (size_t i = 0; i < got_k.size(); ++i) {
                uint32_t a, b;
                std::memcpy(&a, &got_k[i], 4);
                std::memcpy(&b, &got_g[i], 4);
                diff += a != b;
                mx = std::max(mx, (double) std::fabs(got_k[i] - got_g[i]));
            }
            std::printf("          canon CPU (%s) vs GPU: %zu of %zu values differ (max abs %.3e)\n", one ? "1 token" : "NT tokens",
                        diff, got_k.size(), mx);
            if (diff != 0) ++failures;
        }
    }
    const double ec = rel(got_c, ref), eg = rel(got_g, ref), ecg = rel(got_c, got_g);
    // upstream aa678f10 (#4590-era round): Q5_0 down used to sit at 2.1e-2 .. 3.1e-2 (the offset from the original
    // activation sum); with the quantized sum it is as accurate as the rest
    const double gpu_limit = f.d_type == 6 ? 1.8e-2 : 3e-2;
    const bool ok = ec < 3e-2 && eg < gpu_limit && std::isfinite(ec) && std::isfinite(eg);
    std::printf("%s %2d  %-8s/%-7s blob %8zu  cpu rel %.2e  gpu rel %.2e  cpu-gpu %.2e  %s\n", label, l,
                ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), f.bytes, ec, eg, ecg,
                ok ? "ok" : "FAIL");
    if (!ok) ++failures;
    return failures;
}

// A fixed sequence for the synthetic weights (a std engine seeded with a constant draws a lint warning).
struct XorShift32 {
    using result_type = uint32_t;
    uint32_t state;
    static constexpr result_type min() { return 1; }
    static constexpr result_type max() { return 0xFFFFFFFFu; }
    result_type operator()() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
};
static uint32_t float_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// --synthetic GU/DOWN[@HxFF] (e.g. q4_K/q5_1, upstream efeffd8): one expert of random weights quantized by ggml into
// those formats at n_embd H and width FF (2560/640 unless given), three ways as above; then the GPU's f32 dequantizer against ggml's to_float, bit for bit, on
// the gate rows and the down rows.  A pair the GPU has no kernels for is refused (native_expert_supported): 1.
static int check_synthetic(const std::string& arg, void* s, bool had = false) {
    const size_t at = arg.find('@');
    const std::string pair = arg.substr(0, at);
    int64_t H = 2560, FF = 640;
    if (at != std::string::npos) {
        const size_t x = arg.find('x', at);
        H = std::strtoll(arg.c_str() + at + 1, nullptr, 10);
        FF = x == std::string::npos ? 0 : std::strtoll(arg.c_str() + x + 1, nullptr, 10);
        if (H <= 0 || FF <= 0) { std::printf("synthetic: %s is not GU/DOWN@HxFF\n", arg.c_str()); return 2; }
    }
    const size_t slash = pair.find('/');
    auto type_of = [](const std::string& n) {
        for (int t = 0; t < GGML_TYPE_COUNT; ++t)
            if (ggml_get_type_traits((ggml_type) t)->type_name && n == ggml_get_type_traits((ggml_type) t)->type_name)
                return t;
        return -1;
    };
    const int gu = slash == std::string::npos ? -1 : type_of(pair.substr(0, slash));
    const int dt = slash == std::string::npos ? -1 : type_of(pair.substr(slash + 1));
    if (gu < 0 || dt < 0) { std::printf("synthetic: %s is not GU/DOWN ggml type names\n", pair.c_str()); return 2; }
    if (!strata::kernels::native_expert_supported(gu, dt, H, FF)) {
        std::printf("synthetic %s: refused (no GPU expert kernels for the pair)\n", pair.c_str());
        return 1;
    }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu, dt, H, FF, f, err)) { std::printf("synthetic %s: %s\n", pair.c_str(), err.c_str()); return 1; }
    // --synthetic-hadamard: the layer Hadamard-folded (APR, prism.hadamard): random signs, block 128
    std::vector<float> sx((size_t) H), sh((size_t) FF);
    if (had) {
        XorShift32 srng{5};   // the same signs every run
        for (auto& v : sx) v = (srng() & 1) ? 1.f : -1.f;
        for (auto& v : sh) v = (srng() & 1) ? 1.f : -1.f;
        f.had_block = 128;
        f.had_gu = f.had_d = true;
        f.gu_signs = sx.data();
        f.d_signs = sh.data();
    }
    XorShift32 rng{1234};   // the same weights every run
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> w((size_t) FF * H);
    std::vector<uint8_t> blob(f.bytes);
    // the quantizer's importance matrix for the types that require one (IQ2_XXS, IQ2_XS, IQ1_S): ggml asserts on a
    // NULL there.  All 1.0: the weights are random, so every column matters the same
    const std::vector<float> imatrix((size_t) (H > FF ? H : FF), 1.0f);
    auto q = [&](ggml_type t, const float* src, void* dst, int64_t rows, int64_t cols) {
        ggml_quantize_chunk(t, src, dst, 0, rows, cols, ggml_quantize_requires_imatrix(t) ? imatrix.data() : nullptr);
    };
    for (int part = 0; part < 2; ++part) {   // gate, up
        for (auto& v : w) v = nd(rng);
        q((ggml_type) gu, w.data(), blob.data() + (size_t) part * f.up_off, FF, H);
    }
    std::vector<float> wd((size_t) H * FF);
    for (auto& v : wd) v = nd(rng);
    q((ggml_type) dt, wd.data(), blob.data() + f.down_off, H, FF);
    if (f.tail_off) {   // NVFP4: the expert's s_gate, s_up, s_down - not 1, so a kernel that drops one shows
        const float tail[4] = {1.5f, 0.75f, 1.25f, 0.0f};
        std::memcpy(blob.data() + f.tail_off, tail, sizeof tail);
    }
    int failures = check_blob(had ? "hadamard" : "synthetic", 0, blob, f, s);
    // the GPU dequantizers against ggml's to_float, every bit
    namespace par = strata::parity;
    auto dequant_check = [&](int t, const uint8_t* src, int64_t n, const char* what) {
        std::vector<float> want((size_t) n), got((size_t) n);
        ggml_get_type_traits((ggml_type) t)->to_float(src, want.data(), n);
        void* d_src = par::alloc_bytes((size_t) ggml_row_size((ggml_type) t, n));
        auto* d_out = static_cast<float*>(par::alloc_bytes((size_t) n * 4));
        par::copy_bytes_in(d_src, src, ggml_row_size((ggml_type) t, n));
        strata::kernels::iq_dequant_f32(t, d_src, n, d_out, s);
        par::sync();
        par::copy_bytes_out(got.data(), d_out, (size_t) n * 4);
        size_t differ = 0;
        for (int64_t i = 0; i < n; ++i)
            differ += float_bits(got[(size_t) i]) != float_bits(want[(size_t) i]);
        std::printf("          %s %s GPU dequant f32 vs ggml to_float: %zu of %lld values differ\n", what,
                    ggml_type_name((ggml_type) t), differ, (long long) n);
        return differ ? 1 : 0;
    };
    failures += dequant_check(gu, blob.data(), FF * H, "gate");
    failures += dequant_check(dt, blob.data() + f.down_off, H * FF, "down");
    std::printf("native_expert_parity --synthetic %s: %d failures\n", arg.c_str(), failures);
    return failures ? 1 : 0;
}

// R1Q2 has scales outside its code stream. Exercise every K and both short and long rows with identical Q8 values
// and FP16 activation scales on CPU and GPU, including the wrap at the last trellis state.
static int check_bih3_mmvq(void* stream, bool bench = false) {
    constexpr int rows = 128, cols = 3;
    auto& queue = strata::core::Runtime::get().compute();
    const char* path = strata::kernels::iq_mmq_path(queue);
    std::printf("bih3 R1Q2 matrix path: %s\n", path ? path : "scalar row fallback");
    const float p2[4] = {0.25f, 0.5f, 1.f, 2.f};
    std::mt19937 rng(931);  // NOLINT(cert-msc51-cpp,bugprone-random-generator-seed): reproducible parity input
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    double worst = 0, matrix_worst = 0;
    int failures = 0;
    for (int n : {128, 640, 2560}) for (int k = 2; k <= 14; ++k) {
        const int blocks = n / 128, qb = (blocks + 3) / 4;
        const size_t row_bytes = (size_t) blocks * 4 * (size_t) k;
        std::vector<uint8_t> codes((size_t) rows * row_bytes), rowbase((size_t) rows * 2),
                             colscale((size_t) blocks * 2), q((size_t) rows * qb);
        for (auto& v : codes) v = (uint8_t) rng();
        for (auto& v : q) v = (uint8_t) rng();
        auto half = [](std::vector<uint8_t>& v, size_t i, float value) {
            const uint16_t h = ggml_fp32_to_fp16(value);
            v[2 * i] = (uint8_t) h; v[2 * i + 1] = (uint8_t) (h >> 8);
        };
        for (int r = 0; r < rows; ++r) half(rowbase, (size_t) r, 0.5f + 0.5f * (float) (r + 1) / rows);
        for (int b = 0; b < blocks; ++b) half(colscale, (size_t) b, 0.5f + 0.5f * (float) (b + 1) / (float) blocks);
        std::vector<std::vector<uint8_t>> act(cols, std::vector<uint8_t>((size_t) n / 32 * 34));
        std::vector<uint8_t> q81((size_t) cols * n / 32 * 36);
        std::vector<float> xs((size_t) n);
        const void* ap[cols];
        for (int t = 0; t < cols; ++t) {
            for (float& x : xs) x = dist(rng);
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(xs.data(), act[t].data(), n);
            ap[t] = act[t].data();
            for (int b = 0; b < n / 32; ++b) {
                const uint8_t* src = act[t].data() + (size_t) b * 34;
                uint8_t* dst = q81.data() + ((size_t) t * (n / 32) + b) * 36;
                std::memcpy(dst, src, 2); std::memcpy(dst + 4, src + 2, 32);
            }
        }
        cpu::TqR1q2Scales sc{rowbase.data(), colscale.data(), q.data(), (size_t) qb, {p2[0], p2[1], p2[2], p2[3]}};
        std::vector<float> want((size_t) cols * rows), got(want.size());
        float* out[cols];
        for (int t = 0; t < cols; ++t) out[t] = want.data() + (size_t) t * rows;
        cpu::tq_bih3_r1q2_rows(180 + k, codes.data(), row_bytes, n, ap, cols, out, 0, rows, sc);
        strata::parity::Device dev;
        strata::kernels::Bih3R1q2Matrix mat;
        mat.codes = dev.upload(codes); mat.rowbase = dev.upload(rowbase); mat.colscale = dev.upload(colscale);
        mat.q = dev.upload(q); mat.row_bytes = row_bytes; mat.k = k; mat.n_in = n; mat.n_out = rows;
        std::copy(p2, p2 + 4, mat.p2);
        const auto* dx = dev.upload(q81);
        auto* dy = dev.alloc<float>(got.size());
        strata::kernels::bih3_r1q2_mmvq(mat, dx, dy, cols, stream);
        strata::parity::copy_out(got.data(), dy, got.size());
        const double error = rel(got, want);
        worst = std::max(worst, error);
        if (!std::isfinite(error) || error > 1e-5) {
            std::printf("bih3 R1Q2 mmvq: K=%d n=%d rel %.9g FAILED\n", k, n, error);
            ++failures;
        }
        if (path) {
            namespace K = strata::kernels;
            auto* gathered = dev.alloc<uint8_t>((size_t) rows * (size_t) (3 * n / 32) * 36);
            auto* expanded = dev.alloc<uint8_t>(K::iq_mmq_act_bytes(cols, (int64_t) 3 * n));
            const auto* bounds = dev.upload(std::vector<int32_t>{0, cols});
            const auto* ids = dev.upload(std::vector<int32_t>{0, 1, 2});
            K::bih3_r1q2_gather_mmq(mat, gathered, stream);
            K::bih3_r1q2_expand_mmq(dx, expanded, n, cols, stream);
            K::iq_mmq_grouped(queue, K::kBih3Planes, gathered, (size_t) rows * (3 * n / 32) * 36,
                              rows, (int64_t) 3 * n, 1, expanded, cols, bounds, cols, ids, dy, rows).wait_and_throw();
            strata::parity::copy_out(got.data(), dy, got.size());
            const double e = rel(got, want);
            matrix_worst = std::max(matrix_worst, e);
            if (!std::isfinite(e) || e > 1e-5) {
                std::printf("bih3 R1Q2 matrix: K=%d n=%d rel %.9g FAILED\n", k, n, e);
                ++failures;
            }
            if (bench && k == 8 && n == 2560) {
                auto matrix = [&] {
                    K::iq_mmq_grouped(queue, K::kBih3Planes, gathered, (size_t) rows * (3 * n / 32) * 36,
                                      rows, (int64_t) 3 * n, 1, expanded, cols, bounds, cols, ids, dy, rows);
                };
                auto row = [&] { K::bih3_r1q2_mmvq(mat, dx, dy, cols, stream); };
                auto gather = [&] { K::bih3_r1q2_gather_mmq(mat, gathered, stream); };
                auto time = [&](const auto& fn) {
                    for (int i = 0; i < 3; ++i) fn();
                    queue.wait_and_throw();
                    const auto start = std::chrono::steady_clock::now();
                    for (int i = 0; i < 20; ++i) fn();
                    queue.wait_and_throw();
                    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 20;
                };
                const double matrix_us = time(matrix), row_us = time(row), gather_us = time(gather);
                std::printf("bih3 R1Q2 bench: K=8 128x2560 3 columns, %s matrix %.1f us, "
                            "row %.1f us, gather %.1f us (wall per call, 20 calls)\n",
                            path, matrix_us, row_us, gather_us);
            }
        }
    }
    std::printf("bih3 R1Q2 mmvq: 39 cases, max relative error %.9g, %d failures\n", worst, failures);
    if (path) std::printf("bih3 R1Q2 %s: 39 cases, max relative error %.9g\n", path, matrix_worst);
    return failures ? 1 : 0;
}

// Two differently sized experts in the same call, with out-of-order destinations and a repeated expert.
static int check_bih3_grouped(void* stream) {
    constexpr int H = 256, FF = 128, NT = 3;
    const float p2[3][4] = {{0.25f, 0.5f, 1.f, 2.f}, {0.5f, 1.f, 1.5f, 2.f}, {0.25f, 0.5f, 1.f, 2.f}};
    cpu::NativeFmt f[2];
    std::string err;
    if (!cpu::native_r1q2_fmt(182, 183, H, FF, p2, f[0], err) ||
        !cpu::native_r1q2_fmt(188, 189, H, FF, p2, f[1], err)) {
        std::printf("bih3 grouped: %s\n", err.c_str()); return 1;
    }
    std::mt19937 rng(388);  // NOLINT(cert-msc51-cpp,bugprone-random-generator-seed): reproducible parity input
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<std::vector<uint8_t>> blobs(2);
    for (int e = 0; e < 2; ++e) {
        auto& blob = blobs[e]; blob.resize(f[e].bytes);
        for (auto& v : blob) v = (uint8_t) rng();
        for (int m = 0; m < 3; ++m) {
            const int r = m == 2 ? H : FF, blocks = (m == 2 ? FF : H) / 128;
            uint8_t* sc = blob.data() + f[e].sc_off + cpu::native_r1q2_scale_offset(f[e], m);
            const uint16_t rb = ggml_fp32_to_fp16(0.05f), cs = ggml_fp32_to_fp16(0.25f);
            for (int i = 0; i < r; ++i) { sc[(size_t) 2 * i] = (uint8_t) rb; sc[(size_t) 2 * i + 1] = (uint8_t) (rb >> 8); }
            for (int i = 0; i < blocks; ++i) { sc[2 * r + 2 * i] = (uint8_t) cs; sc[2 * r + 2 * i + 1] = (uint8_t) (cs >> 8); }
        }
    }
    std::vector<float> x((size_t) NT * H), want((size_t) NT * H), got(want.size());
    for (float& v : x) v = dist(rng);
    const int32_t route[NT] = {1, 0, 1}, dst[NT] = {1, 0, 2}, tok[NT] = {2, 0, 1};
    std::vector<cpu::NativeAct> act(NT);
    std::vector<uint8_t> q81((size_t) NT * H / 32 * 36);
    for (int t = 0; t < NT; ++t) {
        cpu::native_quant_act(f[0], x.data() + (size_t) t * H, act[t].bytes);
        for (int b = 0; b < H / 32; ++b) {
            const uint8_t* src = act[t].bytes + (size_t) b * 34;
            uint8_t* out = q81.data() + ((size_t) t * (H / 32) + b) * 36;
            std::memcpy(out, src, 2); std::memcpy(out + 4, src + 2, 32);
        }
    }
    for (int i = 0; i < NT; ++i) {
        const int expert = route[dst[i]];
        const void* a[] = {act[tok[i]].bytes};
        std::vector<float> hidden(FF);
        float* hp[] = {hidden.data()};
        cpu::native_gu_rows(f[expert], blobs[expert].data(), a, 1, hp, 0, FF);
        cpu::NativeAct hq;
        cpu::native_quant_h(f[expert], hidden.data(), hq.bytes);
        const void* ha[] = {hq.bytes};
        float* op[] = {want.data() + (size_t) dst[i] * H};
        cpu::native_down_rows(f[expert], blobs[expert].data(), ha, 1, op, 0, H);
    }
    strata::parity::Device dev;
    std::vector<unsigned long long> ptr(3);
    for (int e = 0; e < 2; ++e) ptr[(size_t) e] = (unsigned long long) dev.upload(blobs[(size_t) e]);
    const auto* dptr = dev.upload(ptr);
    const auto* dstart = dev.upload(std::vector<int32_t>{0, 1, 3, 3});
    const auto* groups = dev.upload(std::vector<int32_t>{2});
    const auto* ddst = dev.upload(std::vector<int32_t>(dst, dst + NT));
    const auto* dtok = dev.upload(std::vector<int32_t>(tok, tok + NT));
    const auto* droutes = dev.upload(std::vector<int32_t>(route, route + NT));
    const auto* dk = dev.upload(std::vector<uint8_t>{2, 3, 8, 9});
    const auto* dx = dev.upload(q81);
    auto* dy = dev.alloc<float>(got.size());
    // The PCIe copy kernel must read only each expert's own bytes, preserving the unused tail and empty slot.
    const size_t stride = std::max(f[0].bytes, f[1].bytes);
    std::vector<uint8_t> stage(3 * stride, 0xa5);
    auto* dstage = dev.upload(stage);
    strata::kernels::fetch_r1q2_blobs(dptr, groups, dstage, (int64_t) stride, 3, dstart, ddst, droutes, dk,
                                      H, FF, stream);
    strata::parity::copy_out(stage.data(), dstage, stage.size());
    bool fetch_ok = true;
    for (int e = 0; e < 2; ++e) {
        const size_t bytes = blobs[(size_t) e].size();
        fetch_ok = fetch_ok && std::memcmp(stage.data() + (size_t) e * stride, blobs[(size_t) e].data(), bytes) == 0;
        for (size_t i = bytes; i < stride; ++i) fetch_ok = fetch_ok && stage[(size_t) e * stride + i] == 0xa5;
    }
    for (size_t i = 2 * stride; i < stage.size(); ++i) fetch_ok = fetch_ok && stage[i] == 0xa5;
    if (!fetch_ok) { std::puts("bih3 grouped: mixed fetch corrupted a tail or an empty slot"); return 1; }
    const size_t scratch_bytes = strata::kernels::bih3_r1q2_grouped_scratch_bytes(NT, H, FF);
    void* scratch = dev.alloc<uint8_t>(scratch_bytes);
    strata::kernels::Bih3R1q2Layer L;
    L.n_embd = H; L.n_ff = FF; L.n_expert = 2; L.k_pairs = dk; L.route_ids = droutes;
    std::memcpy(L.p2, p2, sizeof p2);
    strata::kernels::bih3_r1q2_grouped(L, dptr, dstart, groups, ddst, dtok, 3, NT, dx, scratch, dy, stream);
    strata::parity::copy_out(got.data(), dy, got.size());
    const double error = rel(got, want);
    std::printf("bih3 grouped mixed K=2/3,8/9: rel %.9g, scratch %zu bytes\n", error, scratch_bytes);
    strata::kernels::bih3_r1q2_grouped_scalar(L, dptr, dstart, groups, ddst, dtok, 3, NT, dx, scratch, dy, stream);
    strata::parity::copy_out(got.data(), dy, got.size());
    const double scalar_error = rel(got, want);
    std::printf("bih3 grouped scalar mixed K=2/3,8/9: rel %.9g\n", scalar_error);
    const size_t prompt_bytes = strata::kernels::bih3_r1q2_prompt_scratch_bytes(NT, H, FF);
    auto* ps = dev.alloc<uint8_t>(prompt_bytes);
    auto* po = dev.alloc<float>((size_t) NT * H);
    strata::kernels::bih3_r1q2_prompt(L, reinterpret_cast<const uint8_t*>(ptr[0]), 2, 3,
                                      dtok, 1, NT, dx, ps, po, stream);
    strata::kernels::bih3_r1q2_prompt(L, reinterpret_cast<const uint8_t*>(ptr[1]), 8, 9,
                                      dtok + 1, 2, NT, dx, ps, po + H, stream);
    std::vector<float> prompt((size_t) NT * H), prompt_want(prompt.size());
    strata::parity::copy_out(prompt.data(), po, prompt.size());
    for (int i = 0; i < NT; ++i)
        std::copy_n(want.data() + (size_t) dst[i] * H, H, prompt_want.data() + (size_t) i * H);
    const double prompt_error = rel(prompt, prompt_want);
    std::printf("bih3 prompt mixed K=2/3,8/9: rel %.9g, scratch %zu bytes\n", prompt_error, prompt_bytes);
    // A layer's experts in one call: rows sorted by expert (entry 0 expert 0's, entries 1 and 2 expert 1's), a group
    // with no blob (row 1: left as it was) and a cap of 2 rows, so the groups take two passes.
    double groups_error = 0;
    {
        const unsigned long long blob0 = ptr[0], blob1 = ptr[1];
        const std::vector<int32_t> meta = {(int32_t) blob0, (int32_t) (blob0 >> 32), 0, 0,
                                           (int32_t) blob1, (int32_t) (blob1 >> 32)};
        const std::vector<int32_t> starts = {0, 1, 2, 3};
        const std::vector<int32_t> kp = {2 | 3 << 8, 8 | 9 << 8, 8 | 9 << 8};
        const auto* gptr = reinterpret_cast<const unsigned long long*>(dev.upload(meta));
        const auto* gstart = dev.upload(starts);
        const auto* gk = dev.upload(kp);
        const auto* gtok = dev.upload(std::vector<int32_t>(tok, tok + NT));
        const size_t gbytes = strata::kernels::bih3_r1q2_prompt_scratch_bytes(2, H, FF);
        auto* gs = dev.alloc<uint8_t>(gbytes);
        std::vector<float> gout((size_t) NT * H, 7.f);
        auto* go = dev.upload(gout);
        strata::kernels::Bih3R1q2Layer one = L;
        strata::kernels::bih3_r1q2_prompt_groups(one, gptr, gstart, gk, starts.data(), 3, gtok, dx, gs, 2, go, stream);
        strata::parity::copy_out(gout.data(), go, gout.size());
        std::vector<float> g_got, g_want;
        for (int r : {0, 2}) {
            g_got.insert(g_got.end(), gout.begin() + (size_t) r * H, gout.begin() + (size_t) (r + 1) * H);
            g_want.insert(g_want.end(), want.begin() + (size_t) dst[r] * H, want.begin() + (size_t) (dst[r] + 1) * H);
        }
        groups_error = rel(g_got, g_want);
        for (int i = 0; i < H; ++i) if (gout[(size_t) H + i] != 7.f) groups_error = 1;   // the group with no blob
        std::printf("bih3 prompt groups (two passes, a group without a blob): rel %.9g\n", groups_error);
        // the same rows through the matrix products of the gathered codes (kBih3Code), where the device has them
        if (strata::kernels::iq_mmq_path(strata::core::Runtime::get().compute()) != nullptr) {
            const std::vector<int32_t> mptr = {(int32_t) blob0, (int32_t) (blob0 >> 32), (int32_t) blob1,
                                               (int32_t) (blob1 >> 32)};
            const auto* m_ptr = reinterpret_cast<const unsigned long long*>(dev.upload(mptr));
            const auto* m_k = dev.upload(std::vector<int32_t>{2 | 3 << 8, 8 | 9 << 8});
            const auto* m_bnd = dev.upload(std::vector<int32_t>{0, 1, 3});
            const auto* m_rows = dev.upload(std::vector<int32_t>{0, 1, 2});
            auto* ms = dev.alloc<uint8_t>(strata::kernels::bih3_r1q2_grouped_scratch_bytes(NT, H, FF));
            auto* mw = dev.alloc<uint8_t>(2 * strata::kernels::bih3_r1q2_code_bytes(H, FF));
            auto* mo = dev.alloc<float>((size_t) NT * H);
            strata::kernels::bih3_r1q2_prompt_mmq(one, 2, m_ptr, m_k, m_bnd, 2, m_rows, gtok, NT, dx, ms, mw, mo, stream);
            std::vector<float> m_got((size_t) NT * H);
            strata::parity::copy_out(m_got.data(), mo, m_got.size());
            double mmq_error = rel(m_got, prompt_want);
            std::printf("bih3 prompt matrix pass (kBih3Code): rel %.9g\n", mmq_error);
            if (!(mmq_error < 1e-2)) groups_error = 1;
        }
    }
    // The FP16 dequantization (bih3_r1q2_dequant_f16) against the CPU's rows: a column of each matrix is its row
    // products with a one-hot Q8_0 activation (scale 1, one value 1: every other term zero, so the weight exactly)
    double dq_error = 0;
    for (int e = 0; e < 2; ++e) {
        auto* dgu = dev.alloc<uint16_t>((size_t) 2 * FF * H);
        auto* ddn = dev.alloc<uint16_t>((size_t) H * FF);
        strata::kernels::bih3_r1q2_dequant_f16(L, reinterpret_cast<const uint8_t*>(ptr[(size_t) e]),
                                              f[e].gu_type - 180, f[e].d_type - 180, dgu, ddn, stream);
        std::vector<uint16_t> hgu((size_t) 2 * FF * H), hdn((size_t) H * FF);
        strata::parity::copy_out(hgu.data(), dgu, hgu.size());
        strata::parity::copy_out(hdn.data(), ddn, hdn.size());
        std::vector<float> dq_got, dq_want;
        for (int m = 0; m < 3; ++m) {
            const int rows = m == 2 ? H : FF, width = m == 2 ? FF : H;
            const uint8_t* sp = blobs[(size_t) e].data() + f[e].sc_off + cpu::native_r1q2_scale_offset(f[e], m);
            const size_t blocks = (size_t) width / 128;
            cpu::TqR1q2Scales sc{sp, sp + 2 * (size_t) rows, sp + 2 * (size_t) rows + 2 * blocks, (blocks + 3) / 4, {}};
            for (int q = 0; q < 4; ++q) sc.p2[q] = p2[m][q];
            const uint8_t* w = blobs[(size_t) e].data() + (m == 0 ? 0 : m == 1 ? f[e].up_off : f[e].down_off);
            std::vector<uint8_t> onehot((size_t) 8 * (size_t) (width / 32) * 34);
            std::vector<float> col((size_t) 8 * (size_t) rows);
            for (int c0 = 0; c0 < width; c0 += 8) {
                std::fill(onehot.begin(), onehot.end(), 0);
                const void* a[8];
                float* o[8];
                for (int t = 0; t < 8; ++t) {
                    uint8_t* blk = onehot.data() + (size_t) t * (size_t) (width / 32) * 34 + (size_t) ((c0 + t) / 32) * 34;
                    blk[0] = 0x00; blk[1] = 0x3c;   // d = 1.0
                    blk[2 + (c0 + t) % 32] = 1;
                    a[t] = onehot.data() + (size_t) t * (size_t) (width / 32) * 34;
                    o[t] = col.data() + (size_t) t * (size_t) rows;
                }
                cpu::tq_bih3_r1q2_rows(m == 2 ? f[e].d_type : f[e].gu_type, w, m == 2 ? f[e].d_row : f[e].gu_row,
                                       width, a, 8, o, 0, rows, sc);
                for (int t = 0; t < 8; ++t)
                    for (int r = 0; r < rows; ++r) {
                        const size_t at = m == 2 ? (size_t) r * FF + (size_t) (c0 + t)
                                                 : (size_t) (2 * r + m) * H + (size_t) (c0 + t);
                        dq_want.push_back(col[(size_t) t * rows + r]);
                        dq_got.push_back(ggml_fp16_to_fp32(m == 2 ? hdn[at] : hgu[at]));
                    }
            }
        }
        dq_error = std::max(dq_error, rel(dq_got, dq_want));
    }
    std::printf("bih3 FP16 dequantization against the CPU rows: rel %.9g\n", dq_error);
    // The products with the codes decoded in the tile (bih3_r1q2_gemm_grouped_f16) against dequantizing and multiplying
    // (bih3_r1q2_dequant_f16, xmx_gemm_grouped): the same FP16 weights, so the same bits are expected
    double tile_error = 0;
    if (strata::kernels::has_form(strata::kernels::MatrixForm::f16_from_blocks)) {
        constexpr int TR = 5;   // expert 0 two rows, expert 1 three
        std::vector<uint16_t> xg((size_t) TR * H), xd((size_t) TR * FF);
        for (auto& v : xg) v = ggml_fp32_to_fp16(dist(rng));
        for (auto& v : xd) v = ggml_fp32_to_fp16(dist(rng));
        const auto* dxg = dev.upload(xg);
        const auto* dxd = dev.upload(xd);
        const auto* tb = dev.upload(std::vector<int32_t>{0, 2, TR});
        auto* dgu = dev.alloc<uint16_t>((size_t) 2 * 2 * FF * H);
        auto* ddn = dev.alloc<uint16_t>((size_t) 2 * H * FF);
        for (int e = 0; e < 2; ++e)
            strata::kernels::bih3_r1q2_dequant_f16(L, reinterpret_cast<const uint8_t*>(ptr[(size_t) e]),
                                                  f[e].gu_type - 180, f[e].d_type - 180, dgu + (size_t) e * 2 * FF * H,
                                                  ddn + (size_t) e * H * FF, stream);
        auto* want_gu = dev.alloc<float>((size_t) TR * 2 * FF);
        auto* want_dn = dev.alloc<float>((size_t) TR * H);
        auto* got_gu = dev.alloc<float>((size_t) TR * 2 * FF);
        auto* got_dn = dev.alloc<float>((size_t) TR * H);
        strata::kernels::xmx_gemm_grouped(dxg, dgu, (int64_t) 2 * FF * H, want_gu, tb, 2, 3, 2 * FF, H, stream);
        strata::kernels::xmx_gemm_grouped(dxd, ddn, (int64_t) H * FF, want_dn, tb, 2, 3, H, FF, stream);
        const void* blobs_dev[] = {reinterpret_cast<const void*>(ptr[0]), reinterpret_cast<const void*>(ptr[1])};
        const int32_t kp[] = {(f[0].gu_type - 180) | (f[0].d_type - 180) << 8,
                              (f[1].gu_type - 180) | (f[1].d_type - 180) << 8};
        strata::kernels::bih3_r1q2_gemm_grouped_f16(L, false, dxg, blobs_dev, kp, got_gu, 2 * FF, tb, 2, 3, stream);
        strata::kernels::bih3_r1q2_gemm_grouped_f16(L, true, dxd, blobs_dev, kp, got_dn, H, tb, 2, 3, stream);
        std::vector<float> a((size_t) TR * 2 * FF), b(a.size()), c((size_t) TR * H), d(c.size());
        strata::parity::copy_out(a.data(), got_gu, a.size());
        strata::parity::copy_out(b.data(), want_gu, b.size());
        strata::parity::copy_out(c.data(), got_dn, c.size());
        strata::parity::copy_out(d.data(), want_dn, d.size());
        size_t differ = 0;
        for (size_t i = 0; i < a.size(); ++i) differ += a[i] != b[i];
        for (size_t i = 0; i < c.size(); ++i) differ += c[i] != d[i];
        tile_error = std::max(rel(a, b), rel(c, d));
        std::printf("bih3 FP16 products decoded in the tile against dequantized: rel %.9g, %zu of %zu values differ\n",
                    tile_error, differ, a.size() + c.size());
    }
    std::vector<float> signs(FF);
    for (int i = 0; i < FF; ++i) signs[i] = i % 3 ? 1.f : -1.f;
    for (auto& fmt : f) { fmt.had_d = true; fmt.had_block = FF; fmt.d_signs = signs.data(); }
    for (int i = 0; i < NT; ++i) {
        const int expert = route[dst[i]];
        const void* a[] = {act[tok[i]].bytes};
        std::vector<float> hidden(FF);
        float* hp[] = {hidden.data()};
        cpu::native_gu_rows(f[expert], blobs[expert].data(), a, 1, hp, 0, FF);
        cpu::NativeAct hq;
        cpu::native_quant_h(f[expert], hidden.data(), hq.bytes);
        const void* ha[] = {hq.bytes};
        float* op[] = {want.data() + (size_t) dst[i] * H};
        cpu::native_down_rows(f[expert], blobs[expert].data(), ha, 1, op, 0, H);
    }
    L.had_block = FF; L.d_signs = dev.upload(signs);
    strata::kernels::bih3_r1q2_grouped_scalar(L, dptr, dstart, groups, ddst, dtok, 3, NT, dx, scratch, dy, stream);
    strata::parity::copy_out(got.data(), dy, got.size());
    const double had_error = rel(got, want);
    std::printf("bih3 grouped scalar with Hadamard: rel %.9g\n", had_error);
    return std::isfinite(error) && error < 1e-2 && std::isfinite(scalar_error) && scalar_error < 1e-2 &&
           std::isfinite(prompt_error) && prompt_error < 1e-2 && std::isfinite(groups_error) && groups_error < 1e-2 &&
           std::isfinite(had_error) && had_error < 1e-2 && std::isfinite(dq_error) && dq_error < 1e-3 &&
           std::isfinite(tile_error) && tile_error < 1e-5 ? 0 : 1;
}

// Isolate the product's device time; gathering and activation quantization finish before the warm-up.
static int benchmark_bih3_event() {
    namespace K = strata::kernels;
    auto& runtime = strata::core::Runtime::get().compute();
    sycl::queue q(runtime.get_context(), runtime.get_device(),
                  sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    const char* path = K::iq_mmq_path(q);
    if (!path) { std::puts("event bench: no tiled path"); return 1; }
    std::printf("event bench path: %s (one expert; gathered weights, no gather/quantize)\n", path);
    strata::parity::Device dev;
    constexpr int rounds = 100;
    for (int i = 0; i < 300; ++i) q.single_task([] {});
    q.wait_and_throw();
    std::vector<sycl::event> empty;
    empty.reserve(rounds);
    for (int i = 0; i < rounds; ++i) empty.push_back(q.single_task([] {}));
    q.wait_and_throw();
    std::vector<double> empty_us;
    for (const auto& ev : empty) {
        const auto a = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto b = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        empty_us.push_back((double) (b - a) / 1000.0);
    }
    std::sort(empty_us.begin(), empty_us.end());
    std::printf("event bench empty kernel: median %.2f us min %.2f us (%d events)\n",
                empty_us[rounds / 2], empty_us[0], rounds);
    for (const auto [n_out, n_in] : {std::pair{640, 2560}, std::pair{2560, 640}}) {
        for (int nt : {8, 16, 32, 64}) for (int ty : {K::kBih3Planes, 12}) {
            const int cols = ty == K::kBih3Planes ? 3 * n_in : n_in;
            // A 65-row activation pitch keeps split-K off without a runtime-owned scratch queue.
            const int pitch = 65;
            const size_t wb = (size_t) n_out * K::iq_mmq_row_bytes(ty, cols);
            auto* w = dev.alloc<uint8_t>(wb);
            auto* x = dev.alloc<float>((size_t) pitch * cols);
            auto* xq = dev.alloc<uint8_t>(K::iq_mmq_act_bytes(pitch, cols));
            auto* y = dev.alloc<float>((size_t) nt * n_out);
            const auto* bounds = dev.upload(std::vector<int32_t>{0, nt});
            std::vector<int32_t> hids((size_t) nt);
            for (int i = 0; i < nt; ++i) hids[(size_t) i] = i;
            const auto* ids = dev.upload(hids);
            q.memset(w, 0, wb); q.memset(x, 0, (size_t) pitch * cols * sizeof(float));
            K::iq_mmq_quantize(q, x, nullptr, xq, cols, cols, pitch, K::iq_mmq_type_sums(ty));
            auto product = [&] { return K::iq_mmq_grouped(q, ty, w, wb, n_out, cols, 1, xq, pitch,
                                                           bounds, nt, ids, y, n_out); };
            for (int i = 0; i < 300; ++i) product();
            q.wait_and_throw();
            std::vector<sycl::event> events;
            events.reserve(rounds);
            for (int i = 0; i < rounds; ++i) events.push_back(product());
            q.wait_and_throw();
            std::vector<double> us;
            for (const auto& ev : events) {
                const auto a = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                const auto b = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                us.push_back((double) (b - a) / 1000.0);
            }
            std::sort(us.begin(), us.end());
            std::printf("event bench %s %dx%d tokens=%d %s: median %.2f us min %.2f us (%d events)\n",
                        n_out == 640 ? "gate/up" : "down", n_out, n_in, nt,
                        ty == K::kBih3Planes ? "R1Q2 planes" : "Q4_K", us[rounds / 2], us[0], rounds);
        }
    }
    return 0;
}

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape): an uncaught error ends the check with its message
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep the trail on a crash
    if (argc == 2 && std::string(argv[1]) == "--bih3-event-bench") return benchmark_bih3_event();
    if (argc == 2 && std::string(argv[1]) == "--bih3-mmvq") return check_bih3_mmvq(strata::parity::stream());
    if (argc == 2 && std::string(argv[1]) == "--bih3-grouped") return check_bih3_grouped(strata::parity::stream());
    if (argc == 2 && std::string(argv[1]) == "--bih3-bench") return check_bih3_mmvq(strata::parity::stream(), true);
    if (argc == 2 && std::string(argv[1]) == "--bf16-embd") return check_bf16_embd(strata::parity::stream());
    if (argc == 3 && std::string(argv[1]) == "--synthetic") return check_synthetic(argv[2], strata::parity::stream());
    if (argc == 3 && std::string(argv[1]) == "--synthetic-hadamard")
        return check_synthetic(argv[2], strata::parity::stream(), true);
    if (argc < 2) { std::fprintf(stderr, "usage: native_expert_parity <shard1.gguf> [layer ...] | --bf16-embd | --synthetic GU/DOWN[@HxFF] | --synthetic-hadamard GU/DOWN[@HxFF]\n"); return 2; }
    strata::GgufFile gguf(argv[1]);
    std::vector<int> layers;
    for (int i = 2; i < argc; ++i) layers.push_back((int) std::strtol(argv[i], nullptr, 10));
    if (layers.empty()) layers = {0, 1, 2, 3, 20, 47};
    const int E = 7;
    int failures = 0;
    void* s = strata::parity::stream();
    for (int l : layers) {
        const strata::TensorInfo* t[3] = {};
        const char* roles[3] = {"gate", "up", "down"};
        for (const auto& ti : gguf.tensors())
            for (int r = 0; r < 3; ++r)
                if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
        if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
        // the model's own geometry: a gate tensor is [n_embd, n_ff, n_expert]
        const int64_t H = t[0]->shape.size() > 1 ? (int64_t) t[0]->shape[0] : 0;
        const int64_t FF = t[0]->shape.size() > 1 ? (int64_t) t[0]->shape[1] : 0;
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
            std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
        }
        std::vector<uint8_t> blob(f.bytes);
        std::memcpy(blob.data(), gguf.tensor_data(*t[0]) + (size_t) E * f.up_off, f.up_off);
        std::memcpy(blob.data() + f.up_off, gguf.tensor_data(*t[1]) + (size_t) E * f.up_off, f.up_off);
        std::memcpy(blob.data() + f.down_off, gguf.tensor_data(*t[2]) + (size_t) E * (f.bytes - f.down_off),
                    f.bytes - f.down_off);
        failures += check_blob("layer", l, blob, f, s);
    }
    std::printf("native_expert_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
