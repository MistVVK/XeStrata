// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/cpu/native_expert.cpp - plan v0.3 P6: native (GGUF-form) experts on the CPU through ggml-cpu.
// See the header.  Nothing here is Strata arithmetic: the activation quantizers and the row dot products are
// ggml-cpu's, so an IQ expert computes what llama.cpp's CPU backend computes for it.
#include "strata/kernels/cpu/canon_expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/nvfp4.hpp"
#include "strata/kernels/swiglu_clamp.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "strata/platform/env.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <new>
#include <vector>

namespace strata::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

// Do the trellis rows take Strata's own kernels (tq_cpu.hpp)?  Always, unless STRATA_NO_TQ_KERNELS=1 asks for ggml-cpu's
// vec_dot AND this ggml has the type (the agentionai fork; an upstream ggml does not, and must never be asked about it).
bool tq_rows_own(int type) {
    if (!tq_supported(type)) return false;
    if (tq_kernels_enabled()) return true;
    return type >= (int) GGML_TYPE_COUNT || traits(type) == nullptr || traits(type)->vec_dot == nullptr;
}

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

}  // namespace

bool native_experts_available() noexcept { return true; }

// The "is this format handled by that kernel" questions, answered here and not next to the kernels: iq_avx512.cpp is
// compiled for AVX-512 and iq_avx2.cpp for AVX2, and native_gu_rows asks these on every CPU before it
// has checked what the CPU can run.  A function in a wide-ISA file may use that ISA anywhere in its body, so an
// answer that is only a comparison must not come from one (#795).
// IQ4_XS's gate/up rows in 512-bit lanes (upstream 55dc17697), on where the CPU has the AVX-512 the kernel needs
// (XESTRATA_IQ512_IQ4XS=0 turns them off).  They round differently from the ggml-cpu single-token dot an AVX-512 CPU
// takes IQ4_XS with today.  llvm-mca on the built loop (znver4, a 256-value block): 35.3 cycles at one token where
// ggml's dot is 31.8 (-11%), 41.0 at two (two dots: 63.6, +35%) and 54.3 at four (127.2, +57%): the decode windows
// (MTP or drafts: two tokens and up) win, one-token windows lose a little.  No AVX-512 machine here: the speed is an
// estimate (unverified).
static bool iq4xs_512_on() {
    static const bool on = [] {
        const char* v = strata::env_get("IQ512_IQ4XS");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || (type == 23 && iq4xs_512_on());
}

bool iq256_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 19 || type == 21 || type == 22 || type == 23;
}

bool kq256_supported(int type) noexcept { return type == 10 || type == 11 || type == 12 || type == 13 || type == 14; }

size_t native_row_bytes(int type, int64_t n) {
    if (type < 0 || type >= GGML_TYPE_COUNT || n <= 0 || n % ggml_blck_size((ggml_type) type)) return 0;
    return ggml_row_size((ggml_type) type, n);
}

size_t native_vec_act_bytes(int type, int64_t n) {
    init_once();
    const ggml_type_traits_cpu* t = type >= 0 && type < GGML_TYPE_COUNT ? traits(type) : nullptr;
    if (t == nullptr || t->vec_dot == nullptr || native_row_bytes(type, n) == 0) return 0;
    const int vt = (int) t->vec_dot_type;
    if (vt != GGML_TYPE_F32 && (traits(vt) == nullptr || traits(vt)->from_float == nullptr)) return 0;
    return native_row_bytes(vt, n);
}

void native_vec_quant(int type, const float* x, void* dst, int64_t n) {
    const int vt = (int) traits(type)->vec_dot_type;
    if (vt == GGML_TYPE_F32) std::memcpy(dst, x, (size_t) n * 4);
    else traits(vt)->from_float(x, dst, n);
}

void native_vec_rows(int type, const uint8_t* rows, size_t row_bytes, int64_t n, int64_t r0, int64_t r1,
                     const void* act, float* y) {
    const ggml_vec_dot_t dot = traits(type)->vec_dot;
    for (int64_t r = r0; r < r1; ++r) dot((int) n, y + r, 0, rows + (size_t) r * row_bytes, 0, act, 0, 1);
}

bool native_row_to_float(int type, const void* src, float* dst, int64_t n) {
    if (native_row_bytes(type, n) == 0) return false;
    if (type == GGML_TYPE_F32) {   // ggml has no to_float for F32: its rows are the floats
        std::memcpy(dst, src, (size_t) n * 4);
        return true;
    }
    const ggml_type_traits* t = ggml_get_type_traits((ggml_type) type);
    if (t == nullptr || t->to_float == nullptr) return false;
    t->to_float(src, dst, n);
    return true;
}

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err, int up_type) {
    init_once();
    if (up_type < 0) up_type = gu_type;
    // The trellis types (Gyro, ggml types 144-149) are not in a ggml built from upstream (and an upstream ggml indexes its
    // type table by the id, so asking it about 144 reads past the table): their geometry and activation format (Q8_0)
    // are known here, and their rows run on tq_cpu.hpp.
    const bool tq_g = tq_supported(gu_type), tq_d = tq_supported(d_type);
    const ggml_type_traits_cpu* tu = tq_g ? nullptr : traits(up_type);
    const ggml_type_traits_cpu* tg = tq_g ? nullptr : traits(gu_type);
    const ggml_type_traits_cpu* td = tq_d ? nullptr : traits(d_type);
    if ((!tq_g && (tg == nullptr || tg->vec_dot == nullptr)) || (!tq_d && (td == nullptr || td->vec_dot == nullptr))) {
        err = "native experts: ggml-cpu has no dot product for type " +
              std::to_string(!tq_g && !(tg && tg->vec_dot) ? gu_type : d_type);
        return false;
    }
    const ggml_type vdt_g = tq_g ? GGML_TYPE_Q8_0 : tg->vec_dot_type;
    const ggml_type vdt_d = tq_d ? GGML_TYPE_Q8_0 : td->vec_dot_type;
    const ggml_type_traits_cpu* ag = traits(vdt_g);
    const ggml_type_traits_cpu* ad = traits(vdt_d);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if (!tq_g && (tu == nullptr || tu->vec_dot == nullptr || tu->vec_dot_type != tg->vec_dot_type)) {
        err = "native experts: gate and up of types that read different activations";
        return false;
    }
    if (tq_g && up_type != gu_type) {
        err = "native experts: a trellis gate with an up of another type";
        return false;
    }
    if ((tq_g ? n_embd % kTqBlock : (n_embd % ggml_blck_size((ggml_type) gu_type) ||
                                     n_embd % ggml_blck_size((ggml_type) up_type))) ||
        (tq_d ? n_ff % kTqBlock : n_ff % ggml_blck_size((ggml_type) d_type)) ||
        n_embd % ggml_blck_size(vdt_g) || n_ff % ggml_blck_size(vdt_d)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) vdt_g;
    f.d_act = (int) vdt_d;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.up_type = up_type;
    f.gu_row = tq_g ? (size_t) (n_embd / kTqBlock) * tq_block_bytes(gu_type) : ggml_row_size((ggml_type) gu_type, n_embd);
    f.up_row = tq_g ? f.gu_row : ggml_row_size((ggml_type) up_type, n_embd);
    f.d_row = tq_d ? (size_t) (n_ff / kTqBlock) * tq_block_bytes(d_type) : ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = f.up_off + f.up_row * (size_t) n_ff;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    // NVFP4 in any projection: the expert's {s_gate, s_up, s_down, 0} after its rows (1 for a projection of another
    // type: a mixed expert's tail has 1 there)
    if (gu_type == nvfp4::kType || up_type == nvfp4::kType || d_type == nvfp4::kType) {
        f.tail_off = f.bytes;
        f.bytes += nvfp4::kTailBytes;
    }
    f.act_bytes = gu_type == 42 ? sizeof(ActQ) : ggml_row_size(vdt_g, n_embd);
    if (gu_type == 42 && n_embd > H) {
        err = "native experts: a Q2_0 layer wider than the ActQ";
        return false;
    }
    f.h_bytes = ggml_row_size(vdt_d, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes || n_ff > kNativeMaxFF) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

namespace {
// Q8_K (the activations of every i-quant row): ggml-cpu's x86 quantizer is the scalar reference, ~3 us per token
// and layer on the host before the pool can start; q8k_quant_avx2 writes the same bytes.  cpu_avx2_ok() too:
// iq_avx2.cpp is compiled for AVX2 (an AVX-only CPU, or STRATA_FORCE_ISA=avx, keeps ggml's).  STRATA_NO_Q8K_AVX2=1:
// ggml's on any CPU.
bool q8k_avx2(int type) {
    static const bool on = cpu_avx2_ok() && strata::env_get("NO_Q8K_AVX2") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
}  // namespace

static_assert(sizeof(ActQ) <= sizeof(NativeAct) && alignof(ActQ) <= alignof(NativeAct),
              "a Q2_0 layer's ActQ is kept in a NativeAct");

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    if (f.had_gu) {   // Hadamard-folded gate/up: quantize H (s * x), never x itself (hadamard.hpp)
        thread_local std::vector<float> tmp;
        tmp.assign(x, x + f.n_embd);
        hadamard_rotate(tmp.data(), (int) f.n_embd, f.had_block, f.gu_signs);
        x = tmp.data();
    }
    if (canon_layer(f)) {   // a Q2_0 gate/up's values in the order its rows decode their codes
        canon_quant(x, f.n_embd, 256, dst, f.gu_type == 42 ? CanonOrder::q2_planes : CanonOrder::plain);
        return;
    }
    // native_gu_rows reads a Q2_0 layer's ActQ, not ggml's vec_dot_type
    if (f.gu_type == 42) { act_quant_any(x, (int) f.n_embd, *new (dst) ActQ); return; }
    if (q8k_avx2(f.gu_act)) { q8k_quant_avx2(x, dst, f.n_embd); return; }
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    if (f.had_d) {   // Hadamard-folded down: quantize H (s * h) (hadamard.hpp)
        thread_local std::vector<float> tmp;
        tmp.assign(h, h + f.n_ff);
        hadamard_rotate(tmp.data(), (int) f.n_ff, f.had_block, f.d_signs);
        h = tmp.data();
    }
    if (canon_layer(f)) {
        canon_quant(h, f.n_ff, 32, dst, f.d_type == 42 ? CanonOrder::q2_planes : CanonOrder::iq4nl_calls);
        return;
    }
    if (q8k_avx2(f.d_act)) { q8k_quant_avx2(h, dst, f.n_ff); return; }
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

namespace {
const char* mt_min_env() {
    static const char* env = strata::env_get("IQ_MT_MIN");
    return env;
}
}  // namespace

int native_gu_mt_min(int) {
    // #152: a token's expert rows must round the same alone as in a group, or greedy output depends on how many drafts
    // a verify window held (2026-10-16: an RTX 4070's window of 4 and of 1 parted within 20 tokens, and the drafts'
    // acceptance on the text it parted into moved decode by up to 20%).  So the multi-token kernels from one token
    // on, on every CPU.  An expert's gate/up rows, one token, the kernel against ggml-cpu's dot (i7-14700 P-core,
    // iq_avx2_parity --bench --dispatch): IQ2_S 0.145 / 0.185 ms, IQ3_S 0.162 / 0.298, IQ4_XS 0.099 / 0.115, IQ3_XXS
    // 0.182 / 0.188, IQ2_XS 0.185 / 0.188, IQ2_XXS 0.152 / 0.147 (the one slower); its E-core IQ2_S 0.285 / 0.390.
    // AVX-512 (iq512_gu_rows) measured -1..-3% decode on IQ3_S from one token (0.1.30).  A machine-wide rule, not a
    // per-core one: every core rounds an expert the same.  STRATA_IQ_MT_MIN=N (A/B): from N tokens, and the K-quant
    // rules before 2026-10-16 (kq_mt_min).
    static const int mt_min = mt_min_env() ? (int) std::strtol(mt_min_env(), nullptr, 10) : 1;
    return mt_min;
}

namespace {
// The K-quants (Q4_K, Q5_K, Q6_K) and Q8_0 of the other models' experts on the AVX-2 multi-token kernels (iq_avx2.cpp,
// the AVX-VNNI copy where the CPU has it): every CPU with AVX2, AVX-512 ones too (ggml-cpu's dots for these are
// single-token there as well).  STRATA_NO_KQ256=1: ggml-cpu's dot for every token count.  Q8_0 only with AVX-VNNI: on
// a Ryzen 7 5700X (AVX2 alone, one thread, 2 to 4 tokens) its gate/up took 0.185 / 0.289 / 0.340 ms an expert of
// Ornith's shape against ggml-cpu's 0.170 / 0.235 / 0.309, and llm-jp's down 0.275 / 0.334 / 0.383 against 0.171 /
// 0.240 / 0.317; the K-quants were 1.3 to 1.6 times ggml-cpu's there.
bool kq_on(int type) {
    static const bool on = cpu_avx2_ok() && strata::env_get("NO_KQ256") == nullptr;
    static const bool q80 = on && cpu_avxvnni_ok();
    return type == (int) GGML_TYPE_Q8_0 ? q80 : on && kq256_supported(type);
}
// From how many tokens a K-quant or Q8_0 part takes the multi-token kernel, the same at every token count (#152, as
// native_gu_mt_min): Q4_K, Q5_K and Q6_K from one token (one thread, Ornith's shape, an expert's gate/up + down,
// i7-14700: Q4_K/Q6_K 0.100 / 0.137 ms against ggml-cpu at one token, Q5_K/Q6_K 0.112 / 0.128, Q6_K/Q6_K 0.130 /
// 0.127); Q2_K, Q3_K and Q8_0 never, ggml-cpu's dot being the faster where a verify window's CPU experts mostly serve
// one token (Q2_K 0.103 / 0.069 at one token, 0.210 / 0.230 at four; Q8_0 0.226 / 0.164 at one, 0.282 / 0.222 at two,
// 0.465 / 0.358 at four on the i7-14700; on a Ryzen 7 5700X Q2_K 0.154 / 0.147 at two, 0.266 / 0.295 at four).
// STRATA_IQ_MT_MIN set: the earlier rules (Q2_K and Q3_K from four tokens, the others from mt_min).
int kq_mt_min(int type, int mt_min) {
    const bool k23 = type == (int) GGML_TYPE_Q2_K || type == (int) GGML_TYPE_Q3_K;
    if (mt_min_env() != nullptr) return k23 ? (mt_min > 4 ? mt_min : 4) : mt_min;
    return k23 || type == (int) GGML_TYPE_Q8_0 ? MAXT + 1 : 1;
}
}  // namespace

namespace {
// out[t][r] = w_r . act[t] for rows [r0, r1) of one part of type `type` (the multi-token kernels where the type has one
// and the group is large enough, else ggml-cpu's dot)
void part_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
               int r0, int r1, int mt_min) {
    if (nt >= kq_mt_min(type, mt_min) && kq_on(type)) {
        if (type == (int) GGML_TYPE_Q8_0) q80_256_rows(w, row_bytes, n, act, nt, out, r0, r1);
        else kq256_rows(type, w, row_bytes, n, act, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(type)->vec_dot;
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) dot(n, &out[t][r], 0, w + (size_t) r * row_bytes, 0, act[t], 0, 1);
}
}  // namespace

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    if (canon_layer(f)) { canon_gu_rows(f, blob, act, nt, ff, r0, r1); return; }
    if (f.gu_type == 42) {   // Q2_0: Strata's GGUF-layout kernels on the ActQ native_quant_act wrote
        thread_local float ub[MAXT][kNativeMaxFF];
        float* up[MAXT] = {};
        const ActQ* qa[MAXT] = {};
        for (int t = 0; t < nt; ++t) {
            up[t] = ub[t];
            qa[t] = std::launder(static_cast<const ActQ*>(act[t]));
        }
        const int nbk = (int) (f.n_embd / 64);
        q2_rows_any(blob, f.gu_row, nbk, qa, nt, ff, r0, r1);
        q2_rows_any(blob + f.up_off, f.gu_row, nbk, qa, nt, up, r0, r1);
        for (int t = 0; t < nt; ++t)
            for (int r = r0; r < r1; ++r) ff[t][r] = (ff[t][r] / (1.f + std::exp(-ff[t][r]))) * up[t][r];
        return;
    }
    if (native_mixed_gu(f)) {   // gate and up of different types: each part on its own, then the SwiGLU
        thread_local float ub[MAXT][kNativeMaxFF];
        float* up[MAXT] = {};
        for (int t = 0; t < nt; ++t) up[t] = ub[t];
        const int mt = native_gu_mt_min(f.gu_type);
        part_rows(f.gu_type, blob, f.gu_row, (int) f.n_embd, act, nt, ff, r0, r1, mt);
        part_rows(f.up_type, blob + f.up_off, f.up_row, (int) f.n_embd, act, nt, up, r0, r1, mt);
        const float lim = expert_swiglu_limit();
        const float sg = nvfp4::tail_scale(blob, f.tail_off, 0), su = nvfp4::tail_scale(blob, f.tail_off, 1);
        for (int t = 0; t < nt; ++t)
            for (int r = r0; r < r1; ++r) ff[t][r] = swiglu_host(ff[t][r] * sg, up[t][r] * su, lim);
        return;
    }
    // NVFP4 (Edge0's latent experts): rows that decode a block once for every token, ggml's arithmetic, from one token
    // on (ggml-cpu's own NVFP4 dot is scalar on x86); the expert's s_gate and s_up from the blob's tail
    const float sg = nvfp4::tail_scale(blob, f.tail_off, 0), su = nvfp4::tail_scale(blob, f.tail_off, 1);
    if (f.gu_type == nvfp4::kType) {
        static const bool nv512 = cpu_avx512_ok() && strata::env_get("NO_NVFP4_512") == nullptr;
        static const bool nv256 = cpu_avx2_ok() && strata::env_get("NO_NVFP4_256") == nullptr;
        if (nv512 && nvfp4_512_fits((int) f.n_embd)) {
            nvfp4_512_gu_rows(blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1, sg, su);
            return;
        }
        if (nv256 && nvfp4_256_fits((int) f.n_embd)) {
            nvfp4_256_gu_rows(blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1, sg, su);
            return;
        }
    }
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, STRATA_NO_IQ512 changes nothing).
    static const bool avx512 = cpu_avx512_ok() && strata::env_get("NO_IQ512") == nullptr;
    static const bool avx2 = strata::env_get("NO_IQ256") == nullptr;
    const int mt_min = native_gu_mt_min(f.gu_type);   // #152
    // The trellis types (TQ2_T / TQK6 / TQK7 / TQP5-7): Strata's own kernels (tq_cpu.hpp), for every group size - they
    // decode a block once for all the tokens and give each token the same bits whatever the group (no #152 rule needed).
    if (tq_rows_own(f.gu_type)) {
        tq_gu_rows(tq_isa(), f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    // IQ4_XS has both AVX-2 and AVX-512 kernels; XESTRATA_IQ512_IQ4XS=0 disables its 512-bit rows.  Each
    // kernel only for the formats it implements: falling through an empty switch would leave ff unwritten instead of
    // falling back to ggml-cpu.
    static const bool cpu512 = cpu_avx512_ok();
    if (nt >= mt_min && (iq512_supported(f.gu_type) || (!cpu512 && iq256_supported(f.gu_type)))) {
        if (avx512 && iq512_supported(f.gu_type)) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
        if (avx2 && iq256_supported(f.gu_type)) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
    }
    if (nt >= kq_mt_min(f.gu_type, mt_min) && kq_on(f.gu_type)) {
        if (f.gu_type == (int) GGML_TYPE_Q8_0) q80_256_gu_rows(blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        else kq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const int n = (int) f.n_embd;
    const float lim = expert_swiglu_limit();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            ff[t][r] = swiglu_host(g * sg, u * su, lim);
        }
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    if (canon_layer(f)) { canon_down_rows(f, blob, hq, nt, out, r0, r1); return; }
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it.  From one
    // token on as the gate/up rows (#152): 0.062 / 0.065 ms a row set against ggml-cpu (i7-14700, one thread).
    static const bool iq4nl_mt = strata::env_get("NO_IQ4NL") == nullptr;
    const int mt_min = native_gu_mt_min(f.gu_type);
    if (tq_rows_own(f.d_type)) {   // the trellis types, as in native_gu_rows
        tq_rows(tq_isa(), f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const float sd = nvfp4::tail_scale(blob, f.tail_off, 2);   // NVFP4: the expert's s_down, on the output
    if (f.d_type == nvfp4::kType) {   // as the gate/up rows
        static const bool nv512 = cpu_avx512_ok() && strata::env_get("NO_NVFP4_512") == nullptr;
        static const bool nv256 = cpu_avx2_ok() && strata::env_get("NO_NVFP4_256") == nullptr;
        if (nv512 && nvfp4_512_fits((int) f.n_ff)) {
            nvfp4_512_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1, sd);
            return;
        }
        if (nv256 && nvfp4_256_fits((int) f.n_ff)) {
            nvfp4_256_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1, sd);
            return;
        }
    }
    if (nt >= mt_min && f.d_type == 20 && iq4nl_mt) {   // #152: the same rule as the gate/up rows
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    // the i-quant down rows (GLM-5.3-Flash's IQ3_S, IQ3_XXS, IQ2_XXS) by the gate/up rows' AVX-2 kernel on its Q8_K
    // rows, on AVX-2 CPUs only, as the gate/up rows' formats with no AVX-512 kernel (an AVX-512 CPU keeps ggml-cpu for
    // them).  i7-14700, one thread, 4096 x 2048: IQ3_S 0.770 -> 0.423 ms an expert at one token, 3.05 -> 0.66 at four
    static const bool avx2 = strata::env_get("NO_IQ256") == nullptr;
    static const bool cpu512 = cpu_avx512_ok();
    if (nt >= mt_min && avx2 && !cpu512 && iq256_supported(f.d_type)) {
        iq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (nt >= kq_mt_min(f.d_type, mt_min) && kq_on(f.d_type)) {   // the K-quant and Q8_0 down rows: the same rule
        if (f.d_type == (int) GGML_TYPE_Q8_0) q80_256_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        else kq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s * sd;
        }
    }
}

}  // namespace strata::kernels::cpu
