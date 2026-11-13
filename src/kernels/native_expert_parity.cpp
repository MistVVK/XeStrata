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
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "ggml-cpu.h"
#include "strata/kernels/iq_kernels.hpp"
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
    for (int k = 0; k < NT; ++k) {
        std::vector<float> h(FF);
        for (int64_t r = 0; r < FF; ++r) {
            double g = 0, u = 0;
            for (int64_t i = 0; i < H; ++i) { g += (double) G[r * H + i] * xr[k * H + i]; u += (double) U[r * H + i] * xr[k * H + i]; }
            h[r] = (float) (g / (1.0 + std::exp(-g)) * u);
        }
        if (f.had_d) cpu::hadamard_rotate(h.data(), (int) FF, f.had_block, f.d_signs);
        for (int64_t r = 0; r < H; ++r) {
            double o = 0;
            for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[i];
            ref[k * H + r] = (float) o;
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

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape): an uncaught error ends the check with its message
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep the trail on a crash
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
