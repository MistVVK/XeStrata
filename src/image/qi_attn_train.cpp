// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_attn_train.cpp - a training step's attention on the engine's shared parts (qi_kernels.hpp): the
// forward through the shared attention kernel (kernels::flash_attn, the one inference runs) with its FP32 output and
// each row's log-sum-exp, the backward from that log-sum-exp through the shared products (prefill::Gemm).
//
// The backward takes every head at once and the query rows a tile at a time (FlashAttention-2's five products a tile:
// S = Q K^T, dP = dO V^T, dQ = dS K, dV += P^T dO, dK += dS^T Q), each a strided batch of the heads' products
// (Gemm::f16_batch): P comes back from S and the forward's log-sum-exp, so nothing of rows x rows is kept, and a
// tile's maps are the most it holds.  The products take FP16: dO enters at a power of two a head that brings its
// largest value near the top of FP16's range, dS at one a head and tile, each taken back out of the products' FP32
// sums; P is in [0, 1] as it is.  A head at a time took 32 launches of every kernel a tile on Qwen-Image-2.1 (6
// TFLOPS on the B70, a third of a training step's time, 2026-11-09).

#include "strata/image/qi_kernels.hpp"
#include "strata/image/train_ops.hpp"

#include "strata/core/gpu.hpp"
#include "strata/platform/env.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/kernels/flash_attn_bwd.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/sdpa_aotriton.hpp"
#include "strata/kernels/sdpa_onednn.hpp"
#include "strata/prefill/gemm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <tuple>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include "dit_rows.hpp"
#include "strata/kernels/sdpa_cudnn_internal.hpp"
#include "strata/kernels/sdpa_select.hpp"

namespace strata::image {

namespace {

int64_t setting_integer(const char* text, const char* name) {
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end != text) while (std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (end == text || *end || errno == ERANGE)
        throw std::invalid_argument(std::string("XESTRATA_") + name + ": expected an integer");
    return static_cast<int64_t>(value);
}

constexpr int64_t kWG = 256;
constexpr int kT = 32;              // the transposes' and maps' tile: 32 x 32 through local memory
constexpr int64_t kMaps = 512 << 20;   // a tile's maps (S, dP and three FP16 maps, every head): its rows from this

using dit_rows::pow2_full;

float g_of(const uint32_t* bits) { return pow2_full(sycl::bit_cast<float>(*bits)); }

// every head's rows out of the joint [rows][heads * hd] FP16 rows into [heads][rows][hd]
void gather16(const uint16_t* src, int64_t rows, int64_t heads, int64_t hd, uint16_t* dst, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (heads * rows * hd)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), h = i / (rows * hd), r = (i / hd) % rows, d = i % hd;
        dst[i] = src[(r * heads + h) * hd + d];
    });
}

// D[h][r] = dO[r] . O[r] over head h (FP32, as the forward summed O), each head's largest |dO| as bits into amax[h]
void head_dot(const float* dO, const float* O, int64_t rows, int64_t heads, int64_t hd, float* D, uint32_t* amax,
              void* s) {
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (rows * heads) * 128), sycl::range<1>(128)),
        [=](sycl::nd_item<1> it) {
            const int64_t g = static_cast<int64_t>(it.get_group_linear_id()), r = g / heads, h = g % heads;
            float v = 0.f, m = 0.f;
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += 128) {
                const float x = dO[(r * heads + h) * hd + d];
                v += x * O[(r * heads + h) * hd + d];
                m = sycl::fmax(m, sycl::fabs(x));
            }
            v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            if (it.get_local_linear_id() == 0) {
                D[h * rows + r] = v;
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(amax[h])
                    .fetch_max(sycl::bit_cast<uint32_t>(m));
            }
        });
}

// every head's dO as FP16 [heads][rows][hd] at its head's power of two
void gather_do16(const float* dO, int64_t rows, int64_t heads, int64_t hd, const uint32_t* amax, uint16_t* dst,
                 void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (heads * rows * hd)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), h = i / (rows * hd), r = (i / hd) % rows, d = i % hd;
        dst[i] = kernels::f16_from_f32(dO[(r * heads + h) * hd + d] / g_of(amax + h));
    });
}

// a batch of [rows][cols] FP16 maps at stride si into [cols][rows] at stride so, through local memory
void transpose_batch(const uint16_t* in, int64_t si, int64_t rows, int64_t cols, int64_t batch, uint16_t* out,
                     int64_t so, void* s) {
    const size_t gr = (size_t) ((rows + kT - 1) / kT), gc = (size_t) ((cols + kT - 1) / kT);
    dit_rows::q_of(s).submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint16_t, 2> t(sycl::range<2>(kT, kT + 1), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>((size_t) batch, gr * kT, gc * kT), sycl::range<3>(1, kT, kT)),
                         [=](sycl::nd_item<3> it) {
                             const int64_t b = (int64_t) it.get_group(0);
                             const int64_t r0 = (int64_t) it.get_group(1) * kT, c0 = (int64_t) it.get_group(2) * kT;
                             const int i = (int) it.get_local_id(1), j = (int) it.get_local_id(2);
                             if (r0 + i < rows && c0 + j < cols) t[i][j] = in[b * si + (r0 + i) * cols + c0 + j];
                             sycl::group_barrier(it.get_group());
                             if (c0 + i < cols && r0 + j < rows) out[b * so + (c0 + i) * rows + r0 + j] = t[j][i];
                         });
    });
}

// a tile's dS over dP in place for every head, from S (Q K^T, unscaled) and dP (dO16 V^T, at dO's power of two):
// P = exp(S scale - lse) (0 past a causal row's own key), dS = P (dP / g - D) scale, each head's largest |dS| as bits
// into tb[h]; D [heads][rows], a cross-attention's rows not its keys
void tile_ds(const float* S, float* dP, const float* lse, const float* D, int64_t i0, int64_t n, int64_t rows,
             int64_t keys, int64_t heads, float scale, int64_t n_causal, const uint32_t* gbits, uint32_t* tb, void* s) {
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (heads * n) * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t g = static_cast<int64_t>(it.get_group_linear_id()), h = g / n, r = g % n;
            const float l = lse[(i0 + r) * heads + h], dd = D[h * rows + i0 + r], ig = g_of(gbits + h);
            const int64_t lim = i0 + r < n_causal ? i0 + r + 1 : keys;   // a causal row's keys: itself and before
            const float* Sr = S + (h * n + r) * keys;
            float* dPr = dP + (h * n + r) * keys;
            float mx = 0.f;
            for (auto j = static_cast<int64_t>(it.get_local_linear_id()); j < keys; j += kWG) {
                const float p = j < lim ? sycl::exp(Sr[j] * scale - l) : 0.f;
                const float ds = p * (dPr[j] * ig - dd) * scale;
                dPr[j] = ds;
                mx = sycl::fmax(mx, sycl::fabs(ds));
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            if (it.get_local_linear_id() == 0)
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(tb[h]).fetch_max(
                    sycl::bit_cast<uint32_t>(mx));
        });
}

// a tile's FP16 maps for every head: dS [h][n][keys] and dS^T [h][keys][n] at the head's tile power of two, and P^T
// [h][keys][n] (P again from S and lse), the transposes through local memory
void tile_maps(const float* S, const float* dS, const float* lse, int64_t i0, int64_t n, int64_t keys, int64_t heads,
               float scale, int64_t n_causal, const uint32_t* tb, uint16_t* ds16, uint16_t* ds16t, uint16_t* pt16,
               void* s) {
    const size_t gr = (size_t) ((n + kT - 1) / kT), gc = (size_t) ((keys + kT - 1) / kT);
    dit_rows::q_of(s).submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint16_t, 2> tp(sycl::range<2>(kT, kT + 1), cgh), td(sycl::range<2>(kT, kT + 1), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>((size_t) heads, gr * kT, gc * kT), sycl::range<3>(1, kT, kT)),
                         [=](sycl::nd_item<3> it) {
                             const int64_t h = (int64_t) it.get_group(0);
                             const int64_t a = (int64_t) it.get_group(1) * kT, b = (int64_t) it.get_group(2) * kT;
                             const int i = (int) it.get_local_id(1), j = (int) it.get_local_id(2);
                             const float ig = 1.f / g_of(tb + h);
                             const int64_t mo = h * n * keys;
                             if (a + i < n && b + j < keys) {
                                 const int64_t e = mo + (a + i) * keys + b + j, row = i0 + a + i;
                                 const uint16_t d16 = kernels::f16_from_f32(dS[e] * ig);
                                 ds16[e] = d16;
                                 td[i][j] = d16;
                                 tp[i][j] = row < n_causal && b + j > row
                                                    ? (uint16_t) 0
                                                    : kernels::f16_from_f32(
                                                              sycl::exp(S[e] * scale - lse[row * heads + h]));
                             }
                             sycl::group_barrier(it.get_group());
                             if (b + i < keys && a + j < n) {
                                 ds16t[mo + (b + i) * n + a + j] = td[j][i];
                                 pt16[mo + (b + i) * n + a + j] = tp[j][i];
                             }
                         });
    });
}

// dst[(r0 + r) * heads + h][d] (+)= src[h][r][d] * G(bits[h]) (bits null: 1), src [heads][n][hd]
void scatter(const float* src, int64_t r0, int64_t n, int64_t heads, int64_t hd, float* dst, bool acc,
             const uint32_t* bits, void* s) {
    if (dst == nullptr) return;
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (heads * n * hd)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), h = i / (n * hd), r = (i / hd) % n, d = i % hd;
        const float k = bits != nullptr ? g_of(bits + h) : 1.f;
        float* o = dst + ((r0 + r) * heads + h) * hd + d;
        *o = acc ? *o + src[i] * k : src[i] * k;
    });
}

// dst[h][...] += src[h][...] * G(bits[h]) (bits null: 1), per head `per` values
void add_heads(float* dst, const float* src, int64_t heads, int64_t per, const uint32_t* bits, bool first, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (heads * per)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]);
        const float v = src[i] * (bits != nullptr ? g_of(bits + i / per) : 1.f);
        dst[i] = first ? v : dst[i] + v;
    });
}

// a head's largest dO and V row norms (FP32, V's FP16 rows widened) as bits into nd[h] and nv[h].  `vrows`: V's rows,
// where they are not the queries' own (a cross-attention's keys; 0: rows)
void head_norms(const float* dO, const uint16_t* v16, int64_t rows, int64_t heads, int64_t hd, uint32_t* nd,
                uint32_t* nv, void* s, int64_t vrows = 0) {
    // a work-group takes kNormRows rows of one head, a sub-group a row at a time (its lanes along the row), and adds
    // its largest norms into the head's with one atomic each (one a row a head was 1.6 ms of the B70's 2,304 rows)
    constexpr int64_t kNormRows = 64;
    if (vrows <= 0) vrows = rows;
    const int64_t blocks = (std::max(rows, vrows) + kNormRows - 1) / kNormRows;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (blocks * heads) * 128), sycl::range<1>(128)),
        [=](sycl::nd_item<1> it) {
            const int64_t gi = static_cast<int64_t>(it.get_group_linear_id()), h = gi % heads;
            const int64_t r0 = gi / heads * kNormRows, r1 = sycl::min(rows, r0 + kNormRows),
                        v1 = sycl::min(vrows, r0 + kNormRows);
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t lane = sg.get_local_linear_id(), w = sg.get_local_linear_range();
            float ma = 0.f, mb = 0.f;
            for (int64_t r = r0 + sg.get_group_linear_id(); r < r1; r += sg.get_group_linear_range()) {
                float a = 0.f;
                for (int64_t d = lane; d < hd; d += w) {
                    const float x = dO[(r * heads + h) * hd + d];
                    a += x * x;
                }
                ma = sycl::fmax(ma, sycl::reduce_over_group(sg, a, sycl::plus<float>()));
            }
            for (int64_t r = r0 + sg.get_group_linear_id(); r < v1; r += sg.get_group_linear_range()) {
                float b = 0.f;
                for (int64_t d = lane; d < hd; d += w) {
                    const float y = kernels::f32_from_f16(v16[(r * heads + h) * hd + d]);
                    b += y * y;
                }
                mb = sycl::fmax(mb, sycl::reduce_over_group(sg, b, sycl::plus<float>()));
            }
            ma = sycl::reduce_over_group(it.get_group(), ma, sycl::maximum<float>());
            mb = sycl::reduce_over_group(it.get_group(), mb, sycl::maximum<float>());
            if (it.get_local_linear_id() == 0) {
                using ref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;
                ref(nd[h]).fetch_max(sycl::bit_cast<uint32_t>(sycl::sqrt(ma)));
                ref(nv[h]).fetch_max(sycl::bit_cast<uint32_t>(sycl::sqrt(mb)));
            }
        });
}

// each head's g, the power of two at or above nd max(1, 2 scale nv) / 2^14: |dS| = |P (dP - D)| scale is at most
// 2 scale |dO| |V| (Cauchy-Schwarz, and O is a mean of V's rows), so dS / g stays within FP16's range, as dO / g does
void head_g(const uint32_t* nd, const uint32_t* nv, int64_t heads, float scale, float* g, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) heads), [=](sycl::id<1> id) {
        const float nvh = sycl::bit_cast<float>(nv[id[0]]);
        const float x = sycl::bit_cast<float>(nd[id[0]]) * sycl::fmax(1.f, 2.f * scale * nvh) / 16384.f;
        if (!(x > 0.f)) {
            g[id[0]] = 1.f;
            return;
        }
        const uint32_t b = sycl::bit_cast<uint32_t>(x);
        int e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
        e = sycl::clamp(e, -126, 127);
        g[id[0]] = sycl::bit_cast<float>((uint32_t) (e + 127) << 23);
    });
}

// dO / g as FP16 [rows][heads * hd] and D [h][r] = dO . O / g: a sub-group a (row, head), its lanes along the row
void scaled_do(const float* dO, const float* O, int64_t rows, int64_t heads, int64_t hd, const float* g, uint16_t* doh,
               float* D, void* s) {
    constexpr int64_t kPer = 8;   // (row, head) pairs a work-group of 128
    const int64_t n = rows * heads;
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) ((n + kPer - 1) / kPer) * 128), sycl::range<1>(128)),
        [=](sycl::nd_item<1> it) {
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t lane = sg.get_local_linear_id(), w = sg.get_local_linear_range();
            const int64_t p0 = static_cast<int64_t>(it.get_group_linear_id()) * kPer;
            for (int64_t p = p0 + sg.get_group_linear_id(); p < sycl::min(n, p0 + kPer);
                 p += sg.get_group_linear_range()) {
                const int64_t r = p / heads, h = p % heads;
                const float ig = 1.f / g[h];
                float v = 0.f;
                for (int64_t d = lane; d < hd; d += w) {
                    const int64_t i = p * hd + d;
                    v += dO[i] * O[i];
                    doh[i] = kernels::f16_from_f32(dO[i] * ig);
                }
                v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
                if (lane == 0) D[h * rows + r] = v * ig;
            }
        });
}

// the training attention's workspaces, kept between the calls: (stream, slot) -> its buffer and size
std::mutex g_ws_mu;
std::map<std::pair<void*, int>, std::pair<uint8_t*, size_t>> g_ws_kept;

// a stream's workspace of at least `bytes`, kept between the calls (`slot` one of a few, each its own): the fused
// form's (0), the library forward's (1) and backward's (2), the products' two forms' (3, 4)
uint8_t* workspace(void* stream, int slot, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_ws_mu);
    auto& [p, cap] = g_ws_kept[{stream, slot}];
    if (cap < bytes) {
        if (p != nullptr) {
            gpu::stream_sync(stream);
            gpu::free(p);
            p = nullptr;
            cap = 0;
        }
        if (!gpu::alloc_device(reinterpret_cast<void**>(&p), bytes))
            throw std::runtime_error("image engine: the attention's workspace (" + std::to_string(bytes >> 20) +
                                     " MiB)");
        cap = bytes;
    }
    return p;
}

// the fused backward (kernels::flash_attn_bwd) where the GPU takes it: false, nothing done, where it does not
bool fused_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const float* o32, const float* lse,
               const float* dO, int64_t T, int64_t H, int64_t hd, float* dQ, float* dK, float* dV, bool accumulate,
               void* stream, int64_t n_causal, int64_t q_rows = 0, int64_t keys = 0) {
    // the fused kernel walks the rows as their own keys: a cross-attention's keys are another count (the products or
    // the maker's library take it)
    if (hd != 128 || dQ == nullptr || dK == nullptr || dV == nullptr || (keys > 0 && keys != T)) return false;
    const int64_t Tp = qi_attn_rows(T);   // the chunks' rows, read whole: dO's past T zero, as q, k and v's are
    const size_t n16 = (size_t) (Tp * H * hd), n32 = (size_t) (H * T + H);
    const size_t bytes = 256 + n32 * 4 + n16 * 2;
    uint8_t* ws = workspace(stream, 0, bytes);
    auto* bits = reinterpret_cast<uint32_t*>(ws);   // nd [H], nv [H] (H up to 32)
    if (2 * H * 4 > 256) return false;
    auto* D = reinterpret_cast<float*>(ws + 256);
    float* g = D + H * T;
    auto* doh = reinterpret_cast<uint16_t*>(g + H);
    const float scale = 1.f / std::sqrt((float) hd);
    gpu::memset_async(bits, 0, 256, stream);
    head_norms(dO, v16, T, H, hd, bits, bits + H, stream);
    head_g(bits, bits + H, H, scale, g, stream);
    if (Tp > T) gpu::memset_async(doh + T * H * hd, 0, (size_t) ((Tp - T) * H * hd) * 2, stream);
    scaled_do(dO, o32, T, H, hd, g, doh, D, stream);
    kernels::FlashAttnBwdArgs a;
    a.q = q16;
    a.k = k16;
    a.v = v16;
    a.dO = doh;
    a.ld = H * hd;
    a.lse = lse;
    a.D = D;
    a.g = g;
    a.rows = T;
    a.n_head = H;
    a.head_dim = hd;
    a.n_causal = n_causal;
    a.scale = scale;
    a.dq = dQ;
    a.dk = dK;
    a.dv = dV;
    a.accumulate = accumulate;
    a.q_rows = q_rows;
    return kernels::flash_attn_bwd(a, stream);
}

// The GPU maker's library's attention for training (oneDNN's SDPA on Intel GPUs, cuDNN's on NVIDIA ones in the
// contrib-llvm mode, aotriton's on AMD ones), no mask:
// the forward into o16 FP16 and st [H][rows] (the libraries' layout), the backward into FP16 gradients in do16's unit;
// false where none takes the GPU
bool maker_fwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys, int64_t H,
               int64_t hd, uint16_t* o16, float* st, void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    const auto maker = kernels::sdpa_maker(q, stream, kernels::SdpaUse::train, false, hd);
#if STRATA_ONEDNN
    if (maker == kernels::SdpaMaker::onednn && kernels::onednn_sdpa_train_fwd(q16, k16, v16, rows, keys, H, hd, o16, st, stream)) return true;
#endif
#if STRATA_CUDNN
    if (maker == kernels::SdpaMaker::cudnn)
        return sdpa::cudnn_train(q, false, q16, k16, v16, nullptr, nullptr, st, rows, keys, H, hd, o16, nullptr,
                                 nullptr, nullptr);
#endif
    return maker == kernels::SdpaMaker::aotriton && kernels::aotriton_sdpa_train(false, q16, k16, v16, nullptr, nullptr, st, rows, keys, H, hd, o16, nullptr,
                                        nullptr, nullptr, nullptr, stream);
}
// ws: rows * H * (hd + 1) floats of the library's own
bool maker_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const uint16_t* o16, float* st,
               const uint16_t* do16, int64_t rows, int64_t keys, int64_t H, int64_t hd, uint16_t* dq16, uint16_t* dk16,
               uint16_t* dv16, float* ws, void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    const auto maker = kernels::sdpa_maker(q, stream, kernels::SdpaUse::train, false, hd);
#if STRATA_ONEDNN
    if (maker == kernels::SdpaMaker::onednn && kernels::onednn_sdpa_train_bwd(q16, k16, v16, o16, st, do16, rows, keys, H, hd, dq16, dk16, dv16, stream))
        return true;
#endif
#if STRATA_CUDNN
    if (maker == kernels::SdpaMaker::cudnn)
        return sdpa::cudnn_train(q, true, q16, k16, v16, o16, do16, st, rows, keys, H, hd, nullptr, dq16, dk16, dv16);
#endif
    return maker == kernels::SdpaMaker::aotriton && kernels::aotriton_sdpa_train(true, q16, k16, v16, o16, do16, st, rows, keys, H, hd, nullptr, dq16, dk16,
                                        dv16, ws, stream);
}

// The library's attention for training: the image rows (every key seen) through it, the text rows before them
// (n_causal: causal) through the engine's own kernels, each the rows its own.

bool library_fwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t H, int64_t hd,
                 float* out32, float* lse, void* stream, int64_t nt, int64_t keys) {
    const int64_t C = H * hd, ri = T - nt;
    if (ri <= 0) return false;
    if (keys <= 0) keys = T;
    uint8_t* ws = workspace(stream, 1, (size_t) (ri * C) * 2 + (size_t) (ri * H) * 4);
    auto* o16 = reinterpret_cast<uint16_t*>(ws);
    auto* st = reinterpret_cast<float*>(ws + (size_t) (ri * C) * 2);   // the library's [H][ri]
    if (!maker_fwd(q16 + nt * C, k16, v16, ri, keys, H, hd, o16, st, stream)) return false;
    float* o = out32 + nt * C;
    float* l = lse + nt * H;
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) (ri * C)), [=](sycl::id<1> id) {
        o[id[0]] = kernels::f32_from_f16(o16[id[0]]);
        if (id[0] < (size_t) (ri * H)) {   // its statistics into the engine's [rows][H]
            const int64_t t = (int64_t) id[0] / H, h = (int64_t) id[0] % H;
            l[id[0]] = st[h * ri + t];
        }
    });
    if (nt > 0) {   // the text rows: the engine's own kernel, its rows alone
        kernels::FlashAttnArgs a;
        a.q = q16;
        a.q_ld = a.kv_ld = a.out_ld = C;
        a.k = k16;
        a.v = v16;
        a.n_head = a.n_kv = H;
        a.head_dim = hd;
        a.rows = a.keys = T;
        a.n_causal = nt;
        a.first = 0;
        a.row_end = nt;
        a.out = out32;
        a.out_f32 = true;
        a.lse = lse;
        a.scale = 1.0f / std::sqrt((float) hd);
        if (!kernels::flash_attn(a, stream)) return false;
    }
    return true;
}

// the image rows' gradients through the library, FP16 in the head's unit g (below), then times g into dQ, dK and dV
// (written or, dK and dV, added into); false, nothing written, where the library refuses or a gradient came out past
// FP16's range; the text rows' then through the fused kernel, their queries alone, added into dK and dV
bool library_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const float* o32, const float* lse,
                 const float* dO, int64_t T, int64_t H, int64_t hd, float* dQ, float* dK, float* dV, bool accumulate,
                 void* stream, int64_t nt, int64_t keys) {
    const int64_t C = H * hd, ri = T - nt;
    constexpr size_t kHead = 512;   // nd [H], nv [H] and the flag
    if (ri <= 0 || (size_t) (2 * H * 4 + 4) > kHead) return false;
    if (nt > 0) {
        // the text rows' backward is the fused kernel's: where it cannot run (a GPU without FP16 matrix engines, a
        // head width other than 128), refuse before anything is written, so the caller's products take the whole
        // step - a return false after the library's rows would leave half-written gradients for the next form
        const auto dev = core::Runtime::get().stream(stream).get_device();
        if (hd != 128 || dQ == nullptr || dK == nullptr || dV == nullptr || 2 * H * 4 > 256 ||
            kernels::flash_attn_path(&dev) < 0)
            return false;
    }
    if (keys <= 0) keys = T;
    const size_t n = (size_t) (T * C), nk = (size_t) (keys * C), nc = std::max(n, nk);
    uint8_t* ws = workspace(stream, 2, kHead + (size_t) (T * H + H) * 4 + (3 * n + 2 * nk) * 2 +
                                           (size_t) (ri * H) * 4 + (size_t) (ri * H * (hd + 1)) * 4);
    auto* bits = reinterpret_cast<uint32_t*>(ws);   // nd [H], nv [H], the non-finite flag
    uint32_t* bad = bits + 2 * H;
    auto* D = reinterpret_cast<float*>(ws + kHead);
    float* g = D + H * T;
    auto* doh = reinterpret_cast<uint16_t*>(g + H);
    uint16_t *o16 = doh + n, *dq16 = o16 + n, *dk16 = dq16 + n, *dv16 = dk16 + nk;
    auto* st = reinterpret_cast<float*>(dv16 + nk);   // the image rows' statistics as the library takes them [H][ri]
    float* lws = st + ri * H;                         // the library's own room
    gpu::memset_async(bits, 0, kHead, stream);
    head_norms(dO, v16, T, H, hd, bits, bits + H, stream, keys);
    // the fused kernel's g (dS within FP16's range) times 16: dS 2^-4 of it, so that the products' sums over the keys
    // and the queries stay within it too (a sum past it is caught below).  The block's gradients came out the same
    // from 2^-8 to 2^8 of it (Qwen-Image-2.1, 32 heads of 1,048 rows, 2026-11-09)
    head_g(bits, bits + H, H, 1.f / std::sqrt((float) hd), g, stream);
    dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) H), [=](sycl::id<1> id) { g[id[0]] *= 16.f; });
    scaled_do(dO, o32, T, H, hd, g, doh, D, stream);
    const float* li = lse + nt * H;
    dit_rows::q_of(stream).parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        o16[id[0]] = kernels::f16_from_f32(o32[id[0]]);
        if (id[0] < (size_t) (ri * H)) {
            const int64_t h = (int64_t) id[0] / ri, t = (int64_t) id[0] % ri;
            st[id[0]] = li[t * H + h];
        }
    });
    if (!maker_bwd(q16 + nt * C, k16, v16, o16 + nt * C, st, doh + nt * C, ri, keys, H, hd, dq16, dk16, dv16, lws,
                   stream))
        return false;
    // a gradient past FP16's range: nothing written, the caller's other forms take the step
    const int64_t nq = ri * C;
    dit_rows::q_of(stream).parallel_for(sycl::range<1>(nc), [=](sycl::id<1> id) {
        const size_t i = id[0];
        bool ok = true;
        if (i < nk) ok = std::isfinite(kernels::f32_from_f16(dk16[i])) && std::isfinite(kernels::f32_from_f16(dv16[i]));
        if (ok && (int64_t) i < nq) ok = std::isfinite(kernels::f32_from_f16(dq16[i]));
        if (!ok) sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*bad).store(1u);
    });
    uint32_t flag = 0;
    gpu::stream_sync(stream);
    if (!gpu::copy(&flag, bad, 4)) throw std::runtime_error("image engine: the attention's check read back");
    if (flag != 0) {
        std::fprintf(stderr, "xestrata-image: the library's attention gradients passed FP16's range; the engine's own "
                             "kernels take the step\n");
        return false;
    }
    float* dq = dQ + nt * C;
    dit_rows::q_of(stream).parallel_for(sycl::range<1>(nc), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float gh = g[(i % C) / hd];
        if ((int64_t) i < nq) dq[i] = kernels::f32_from_f16(dq16[i]) * gh;
        if (i < nk) {
            const float k = kernels::f32_from_f16(dk16[i]) * gh, v = kernels::f32_from_f16(dv16[i]) * gh;
            dK[i] = accumulate ? dK[i] + k : k;
            dV[i] = accumulate ? dV[i] + v : v;
        }
    });
    if (nt > 0 && !fused_bwd(q16, k16, v16, o32, lse, dO, T, H, hd, dQ, dK, dV, true, stream, nt, nt))
        throw std::runtime_error("image engine: the attention's text rows have no backward kernel here");
    return true;
}


// P (FP16) and dS (FP16, in g's units) of a tile of n query rows from i0 for every head, from S and dP [h][n][keys]
// (FP32: Q K^T unscaled, dO / g . V^T): masked above a causal row's own key; D [H][rows], a cross-attention's rows
// not its keys
void tile_pds(const float* S, const float* dP, const float* lse, const float* D, int64_t i0, int64_t n, int64_t rows,
              int64_t keys, int64_t H, float scale, int64_t n_causal, uint16_t* p16, uint16_t* ds16, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (H * n * keys)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], h = i / (n * keys), r = (i / keys) % n, j = i % keys, q = i0 + r;
        float p = 0.f, ds = 0.f;
        if (!(q < n_causal && j > q)) {
            p = sycl::exp(S[i] * scale - lse[q * H + h]);
            ds = p * (dP[i] - D[h * rows + q]) * scale;
        }
        p16[i] = kernels::f16_from_f32(p);
        ds16[i] = kernels::f16_from_f32(ds);
    });
}

// out[(r0 + r) * H * hd + h * hd + d] (+)= sum[h][r][d] * g[h], sum [H][n][hd] (accumulate: added into)
void put_scaled(const float* sum, const float* g, int64_t r0, int64_t n, int64_t H, int64_t hd, float* out, bool acc,
                void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (n * H * hd)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], h = i / (n * hd), r = (i / hd) % n, d = i % hd;
        float* o = out + (r0 + r) * H * hd + h * hd + d;
        const float v = sum[i] * g[h];
        *o = acc ? *o + v : v;
    });
}

// The shared products' backward without transposed maps, where oneMath takes the GPU's FP16 products
// (Gemm::f16_batch_ex): every head a batch at the joint rows' strides, a tile of query rows at a time; dO in g's
// units (the fused form's choice), so dS needs no tile's scale and dK and dV sum in the products themselves.  A
// (query, key, head) moves about 26 bytes against the maps' 54: S and dP written and read once, P and dS written once
// and read by the three products.  False, nothing done, where oneMath does not take them.
bool products_v2(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const float* o32, const float* lse,
                 const float* dO, int64_t T, int64_t H, int64_t hd, float* dQ, float* dK, float* dV, bool accumulate,
                 prefill::Gemm& gemm, void* stream, int64_t n_causal, int64_t keys) {
    if (strata::env_get("ATTN_BWD_MAPS") != nullptr || dQ == nullptr || dK == nullptr || dV == nullptr ||
        2 * H * 4 > 256)
        return false;
    if (keys <= 0) keys = T;
    const int64_t C = H * hd;
    // a tile's query rows: S and dP FP32, P and dS FP16 (12 bytes a key a row a head) within kMaps
    const int64_t tile = std::clamp<int64_t>(kMaps / (12 * H * std::max<int64_t>(keys, 1)) / kT * kT, kT,
                                             (T + kT - 1) / kT * kT);
    const size_t maps = (size_t) (H * tile * keys);
    const size_t bytes = 256 + ((size_t) (H * T + H) + 2 * (size_t) (keys * C) + 2 * maps + (size_t) (tile * C)) * 4 +
                         (size_t) (T * C) * 2 + 2 * maps * 2;
    uint8_t* ws = workspace(stream, 3, bytes);
    auto* bits = reinterpret_cast<uint32_t*>(ws);
    auto* D = reinterpret_cast<float*>(ws + 256);
    float* g = D + H * T;
    float* dk_sum = g + H;
    float* dv_sum = dk_sum + keys * C;
    float* S = dv_sum + keys * C;
    float* dP = S + maps;
    float* dq = dP + maps;   // a tile's dQ [H][tile][hd]
    auto* doh = reinterpret_cast<uint16_t*>(dq + tile * C);
    uint16_t* p16 = doh + T * C;
    uint16_t* ds16 = p16 + maps;
    const float scale = 1.f / std::sqrt((float) hd);
    gpu::memset_async(bits, 0, 256, stream);
    head_norms(dO, v16, T, H, hd, bits, bits + H, stream, keys);
    head_g(bits, bits + H, H, scale, g, stream);
    scaled_do(dO, o32, T, H, hd, g, doh, D, stream);
    for (int64_t i0 = 0; i0 < T; i0 += tile) {
        const int64_t n = std::min(tile, T - i0);
        const float beta = i0 == 0 ? 0.f : 1.f;
        // S = Q K^T and dP = dO V^T: [h][n][keys]
        if (!gemm.f16_batch_ex(false, true, q16 + i0 * C, k16, S, n, keys, hd, H, hd, hd, n * keys, C, C, keys, 0.f) ||
            !gemm.f16_batch_ex(false, true, doh + i0 * C, v16, dP, n, keys, hd, H, hd, hd, n * keys, C, C, keys, 0.f))
            return false;   // (the first product: before anything was written)
        tile_pds(S, dP, lse, D, i0, n, T, keys, H, scale, n_causal, p16, ds16, stream);
        // dQ = dS K, dV += P^T dO, dK += dS^T Q: each head's output a block of its own ([H][rows][hd]: oneMKL takes
        // no batch whose outputs interleave)
        if (!gemm.f16_batch_ex(false, false, ds16, k16, dq, n, hd, keys, H, n * keys, hd, n * hd, keys, C, hd, 0.f) ||
            !gemm.f16_batch_ex(true, false, p16, doh + i0 * C, dv_sum, keys, hd, n, H, n * keys, hd, keys * hd, keys,
                               C, hd, beta) ||
            !gemm.f16_batch_ex(true, false, ds16, q16 + i0 * C, dk_sum, keys, hd, n, H, n * keys, hd, keys * hd, keys,
                               C, hd, beta))
            throw std::runtime_error(
                "image engine: oneMath failed in the attention's backward after its first product");
        put_scaled(dq, g, i0, n, H, hd, dQ, false, stream);   // dQ in g's units: out to its rows
    }
    put_scaled(dk_sum, g, 0, keys, H, hd, dK, accumulate, stream);
    put_scaled(dv_sum, g, 0, keys, H, hd, dV, accumulate, stream);
    return true;
}

}  // namespace

void qi_attention_train_release(void* stream) {
    std::lock_guard<std::mutex> lock(g_ws_mu);
    gpu::stream_sync(stream);
    for (auto it = g_ws_kept.begin(); it != g_ws_kept.end();) {
        if (it->first.first != stream) {
            ++it;
            continue;
        }
        if (it->second.first != nullptr) gpu::free(it->second.first);
        it = g_ws_kept.erase(it);
    }
}

void qi_attention_train(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                        int64_t head_dim, float* out32, float* lse, void* stream, int64_t n_causal, int64_t keys) {
    if (keys <= 0) keys = T;
    kernels::FlashAttnArgs a;
    a.q = q16;
    a.q_ld = heads * head_dim;
    a.k = k16;
    a.v = v16;
    a.kv_ld = heads * head_dim;
    a.n_head = a.n_kv = heads;
    a.head_dim = head_dim;
    a.rows = T;
    a.keys = keys;
    a.n_causal = n_causal;
    a.first = 0;
    a.row_end = T;
    a.out = out32;
    a.out_ld = heads * head_dim;
    a.out_f32 = true;
    a.lse = lse;
    a.scale = 1.0f / std::sqrt((float) head_dim);
    // Keep the original training choice: the common multi-sample trial and sub-group tuning raised four-step
    // Qwen-Image training from 2.17 to 2.29 s/step on the B70 and 3.88 to 4.06 s/step on the RTX 4070 (2026-11-11).
    // the engine's own kernel (0) or the GPU maker's library (1, library_fwd): XESTRATA_ATTN_TRAIN_FWD (own or
    // library), or the faster the first time a GPU takes the shape (rows in steps of 512, keys, heads), both run
    // twice into the outputs then and the second runs timed (the same outputs either way).  On the B70 at 2,304 rows
    // of 32 heads oneDNN's took 2.0 ms and the engine's 3.1; on the A380 12.4 and 189 (2026-11-09)
    auto run = [&](int form) {
        if (form == 1) return library_fwd(q16, k16, v16, T, heads, head_dim, out32, lse, stream, n_causal, keys);
        return kernels::flash_attn(a, stream);
    };
    int form = -1;
    if (const char* e = strata::env_get("ATTN_TRAIN_FWD")) {
        form = std::string(e) == "library" ? 1 : 0;
    } else {
        static std::mutex mu;
        static std::map<std::tuple<void*, int64_t, int64_t, int64_t>, int> chose;
        const auto k = std::make_tuple(stream, (T + 511) / 512, keys, heads);
        {
            std::lock_guard<std::mutex> lock(mu);
            auto it = chose.find(k);
            if (it != chose.end()) form = it->second;
        }
        if (form < 0) {
            double best = 1e30;
            form = 0;
            for (int f = 0; f < 2; ++f) {
                if (!run(f)) continue;
                gpu::stream_sync(stream);
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = run(f);
                gpu::stream_sync(stream);
                const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (ok && t < best) {
                    best = t;
                    form = f;
                }
            }
            std::lock_guard<std::mutex> lock(mu);
            chose[k] = form;
        }
    }
    if ((form == 1 && run(1)) || run(0)) return;
    throw std::runtime_error("image engine: no attention kernel takes a training step's shape (head_dim " +
                             std::to_string(head_dim) + ")");
}

namespace {

// the shared products' backward (below): every head at once, a tile of query rows at a time
void products_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const float* o32, const float* lse,
                  const float* dO, int64_t T, int64_t heads, int64_t head_dim, float* dQ, float* dK, float* dV,
                  bool accumulate, prefill::Gemm& gemm, void* stream, int64_t n_causal, int64_t keys);

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

void qi_attention_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const float* o32,
                      const float* lse, const float* dO, int64_t T, int64_t heads, int64_t head_dim, float* dQ,
                      float* dK, float* dV, bool accumulate, prefill::Gemm& gemm, void* stream, int64_t n_causal,
                      int64_t keys) {
    // Three forms: FlashAttention-2's fused backward on the matrix engines (Intel's, NVIDIA's, AMD's), the shared
    // products, and the GPU maker's library (oneDNN's SDPA on Intel GPUs, the text rows the fused kernel's): whichever
    // ran fastest the first time a GPU took the shape (rows in steps of 512, keys, heads), each timed then into
    // scratch.  On the B70 at 2,304 rows of 32 heads the library took 6.1 ms, the fused form 8.4 and the products 12;
    // on the A380 the products 57 ms, the library 145 and the fused form 509 (2026-11-09).  XESTRATA_ATTN_BWD_FORM
    // (fused, products or library) takes one always where it runs.  A cross-attention (keys not the rows' own) has no
    // fused form: its kernel walks the rows as their own keys.
    if (keys <= 0) keys = T;
    const auto dev = core::Runtime::get().stream(stream).get_device();
    const size_t n = (size_t) (T * heads * head_dim), nk = (size_t) (keys * heads * head_dim);
    auto run = [&](int form, float* q_, float* k_, float* v_, bool acc) {
        switch (form) {
        case 0: return fused_bwd(q16, k16, v16, o32, lse, dO, T, heads, head_dim, q_, k_, v_, acc, stream, n_causal, 0,
                                 keys);
        case 2: return library_bwd(q16, k16, v16, o32, lse, dO, T, heads, head_dim, q_, k_, v_, acc, stream, n_causal,
                                   keys);
        default:
            if (!products_v2(q16, k16, v16, o32, lse, dO, T, heads, head_dim, q_, k_, v_, acc, gemm, stream, n_causal,
                             keys))
                products_bwd(q16, k16, v16, o32, lse, dO, T, heads, head_dim, q_, k_, v_, acc, gemm, stream,
                             n_causal, keys);
            return true;
        }
    };
    static const char* const kName[3] = {"fused", "products", "library"};
    int form = -1;
    if (const char* e = strata::env_get("ATTN_BWD_FORM")) {
        for (int f = 0; f < 3; ++f)
            if (std::string(e) == kName[f]) form = f;
    } else {
        static std::mutex mu;
        static std::map<std::tuple<void*, int64_t, int64_t, int64_t>, int> chose;   // (stream: its GPU, rows / 512, keys, heads)
        const auto k = std::make_tuple(stream, (T + 511) / 512, keys, heads);
        {
            std::lock_guard<std::mutex> lock(mu);
            auto it = chose.find(k);
            if (it != chose.end()) form = it->second;
        }
        if (form < 0) {
            // each form into scratch twice, the second run timed (the first compiles its kernels)
            float* tmp = nullptr;
            form = 1;
            if (gpu::alloc_device(&tmp, (n + 2 * nk) * 4)) {
                double best = 1e30;
                for (int f = 0; f < 3; ++f) {
                    if (!run(f, tmp, tmp + n, tmp + n + nk, false)) continue;
                    gpu::stream_sync(stream);
                    const auto t0 = std::chrono::steady_clock::now();
                    const bool ok = run(f, tmp, tmp + n, tmp + n + nk, false);
                    gpu::stream_sync(stream);
                    const double t = seconds_since(t0);
                    if (ok && t < best) {
                        best = t;
                        form = f;
                    }
                }
                gpu::free(tmp);
            }
            std::lock_guard<std::mutex> lock(mu);
            chose[k] = form;
        }
    }
    // the form taken, or where it does not run (the library's gradients past FP16's range) the fused form, then the
    // products
    bool done = form >= 0 && run(form, dQ, dK, dV, accumulate);
    int ran = form;
    if (!done && form != 0) done = run(ran = 0, dQ, dK, dV, accumulate);
    if (!done) run(ran = 1, dQ, dK, dV, accumulate);
    if (strata::env_get("ATTN_BWD_REPORT") != nullptr) {   // which path ran, once a process (a check's)
        static bool said = false;
        if (!said) {
            std::fprintf(stderr, "attention backward: %s (matrix path %d)\n", kName[ran],
                         kernels::flash_attn_path(&dev));
            said = true;
        }
    }
}

namespace {

void products_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const float* o32, const float* lse,
                  const float* dO, int64_t T, int64_t heads, int64_t head_dim, float* dQ, float* dK, float* dV,
                  bool accumulate, prefill::Gemm& gemm, void* stream, int64_t n_causal, int64_t keys) {
    const int64_t hd = head_dim, H = heads;
    if (keys <= 0) keys = T;
    // a tile's query rows: its maps (S and dP FP32, dS, dS^T and P^T FP16: 14 bytes a key a row a head) within kMaps
    int64_t tile = std::clamp<int64_t>(kMaps / (14 * H * std::max<int64_t>(keys, 1)) / kT * kT, kT,
                                       (T + kT - 1) / kT * kT);
    // XESTRATA_ATTN_BWD_TILE: the tile's rows set (a multiple of 32), so a check with few rows takes several tiles
    if (const char* e = strata::env_get("ATTN_BWD_TILE")) tile = std::max<int64_t>(kT, setting_integer(e, "ATTN_BWD_TILE") / kT * kT);
    const int64_t ntiles = (T + tile - 1) / tile;
    const float scale = 1.f / std::sqrt((float) hd);
    // every head's Q and dO (FP16 [H][T][hd]), K, V and K^T ([H][keys][hd], K^T [H][hd][keys]), D [H][T], dK and dV
    // sums and a tile's parts ([H][keys][hd] FP32), a tile's maps, Q^T and dO^T ([H][hd][tile]) and dQ
    // ([H][tile][hd]); the powers of two: dO's a head, then dS's a head and tile
    const int64_t n16 = 2 * H * T * hd + 3 * H * keys * hd + 3 * H * tile * keys + 2 * H * hd * tile;
    const int64_t n32 = H * T + 4 * H * keys * hd + 2 * H * tile * keys + H * tile * hd;
    const size_t bits_bytes = (size_t) (H * (1 + ntiles)) * 4;
    const size_t bytes = (bits_bytes + 255) / 256 * 256 + (size_t) n32 * 4 + (size_t) n16 * 2;
    // the workspace, kept for the stream's next call (every block of a step takes the same shape): the stream runs in
    // order, so the last call's kernels are done with it when this one's run
    uint8_t* ws = workspace(stream, 4, bytes);
    auto* bits = reinterpret_cast<uint32_t*>(ws);   // [H] dO's largest, then [ntiles][H] each tile's dS's largest
    auto* f = reinterpret_cast<float*>(ws + (bits_bytes + 255) / 256 * 256);
    float* D = f;
    float* dk_sum = D + H * T;
    float* dv_sum = dk_sum + H * keys * hd;
    float* kpart = dv_sum + H * keys * hd;
    float* vpart = kpart + H * keys * hd;
    float* S = vpart + H * keys * hd;
    float* dP = S + H * tile * keys;
    float* dq = dP + H * tile * keys;
    auto* h16 = reinterpret_cast<uint16_t*>(dq + H * tile * hd);
    uint16_t *qh = h16, *kh = qh + H * T * hd, *vh = kh + H * keys * hd, *doh = vh + H * keys * hd,
             *kt = doh + H * T * hd;
    uint16_t *ds16 = kt + H * keys * hd, *ds16t = ds16 + H * tile * keys, *pt16 = ds16t + H * tile * keys;
    uint16_t *qt = pt16 + H * tile * keys, *dot = qt + H * hd * tile;
    gpu::memset_async(bits, 0, bits_bytes, stream);
    gather16(q16, T, H, hd, qh, stream);
    gather16(k16, keys, H, hd, kh, stream);
    gather16(v16, keys, H, hd, vh, stream);
    head_dot(dO, o32, T, H, hd, D, bits, stream);
    gather_do16(dO, T, H, hd, bits, doh, stream);
    transpose_batch(kh, keys * hd, keys, hd, H, kt, hd * keys, stream);
    for (int64_t ti = 0; ti < ntiles; ++ti) {
        const int64_t i0 = ti * tile, n = std::min(tile, T - i0);
        uint32_t* tb = bits + H * (1 + ti);
        gemm.f16_batch(qh + i0 * hd, kh, S, n, keys, hd, H, T * hd, keys * hd, n * keys);    // S = Q K^T
        gemm.f16_batch(doh + i0 * hd, vh, dP, n, keys, hd, H, T * hd, keys * hd, n * keys);  // dP = dO V^T (dO's scale)
        tile_ds(S, dP, lse, D, i0, n, T, keys, H, scale, n_causal, bits, tb, stream);
        tile_maps(S, dP, lse, i0, n, keys, H, scale, n_causal, tb, ds16, ds16t, pt16, stream);
        gemm.f16_batch(ds16, kt, dq, n, hd, keys, H, n * keys, hd * keys, n * hd);           // dQ = dS K
        scatter(dq, i0, n, H, hd, dQ, false, tb, stream);
        transpose_batch(qh + i0 * hd, T * hd, n, hd, H, qt, hd * n, stream);
        transpose_batch(doh + i0 * hd, T * hd, n, hd, H, dot, hd * n, stream);
        gemm.f16_batch(pt16, dot, vpart, keys, hd, n, H, keys * n, hd * n, keys * hd);       // dV's part: P^T dO
        add_heads(dv_sum, vpart, H, keys * hd, nullptr, ti == 0, stream);              // (dO's scale)
        gemm.f16_batch(ds16t, qt, kpart, keys, hd, n, H, keys * n, hd * n, keys * hd);       // dK's part: dS^T Q
        add_heads(dk_sum, kpart, H, keys * hd, tb, ti == 0, stream);                   // (the tile's scale out)
    }
    scatter(dk_sum, 0, keys, H, hd, dK, accumulate, nullptr, stream);
    scatter(dv_sum, 0, keys, H, hd, dV, accumulate, bits, stream);
}

}  // namespace

}  // namespace strata::image
