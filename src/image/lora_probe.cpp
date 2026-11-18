// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/lora_probe.cpp - strata-image-lora-probe: a LoRA adapter file read (its keys brought to the DiT's names,
// its entries and metadata), the delta's arithmetic on the GPU against a CPU reference, and the deltas a DiT binds.
//
//   strata-image-lora-probe list LORA.safetensors|LORA.gguf      the entries and the metadata
//   strata-image-lora-probe selftest                     y += scale (x A^T) B^T, a LoKr's kron(w1, w2), a DoRA's
//                                                               rows (the base product scaled, the delta's rows by c),
//                                                               a whole delta's (diff, set_weight) and a LoHa's (its
//                                                               factors' product made when it is bound) against FP64
//                                                               (random)
//   strata-image-lora-probe map DIT.gguf [LORA [scale]]          the weights adapters may take, and where a file's keys
//                                                                land (or that they do not)
//   strata-image-lora-probe step DIT.gguf OUT.bin [LORA [scale]] one forward of a fixed context and noise, its
//                                                                velocity written as raw FP32 (a base run has no LORA)
//   strata-image-lora-probe moss-map DIR [LORA [scale]]          the same two for MOSS-Image's DiT (MossDit)
//   strata-image-lora-probe moss-step DIR OUT.bin [LORA [scale]]
//   strata-image-lora-probe mf-map DIT.safetensors [LORA [scale]]  the same two for Mage-Flow's DiT (MfDit)
//   strata-image-lora-probe mf-step DIT.safetensors OUT.bin [LORA [scale]]
//
// `list` and `map` read no GPU: they are the readers' checks.  `selftest` builds small adapters and products, with
// and without a per-row scale, and prints how far the GPU's output is from the reference's.  `step` is how a LoRA's
// delta against the same delta baked into the weights is compared (the parity test of the catalogue's records): the
// same C++ forward on the same noise, the model loaded twice.  MOSS_*_HW, MOSS_*_TEXT: the latent's and the text's
// rows (2 and 4); MOSS_STEP_Q8=1: the DiT's blocks in 8 bits (else FP16, the same on both sides of a parity);
// MOSS_STEP_DROP_TEXT=1: the text embedder's weights dropped and read again, a forward each (the same velocity, an
// adapter's layer kept on the weight through the load); MOSS_STEP_BLOCK_MIB: the blocks' VRAM (the rest streamed).
// MF_STEP_HW, MF_STEP_TEXT: the latent's rows a side (2) and the text's (4); MF_STEP_Q8=1: the blocks in 8 bits (else
// FP16); MF_STEP_BLOCK_MIB: the blocks' VRAM (the rest streamed from RAM); MF_STEP_REFS: an edit's reference images
// (one, at the latent size, its own rows after the target's).  LORA_STEP_NOBAKE=1: every step keeps the plain LoRAs
// beside the products instead of baking them in (a DoRA is never baked; the same path on both sides of a comparison).
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/dit_loras.hpp"
#include "strata/image/image_lora.hpp"
#include "strata/image/mf_dit.hpp"
#include "strata/image/moss_dit.hpp"
#include "strata/image/moss_pipeline.hpp"
#include "strata/image/qi_dit.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace gpu = strata::gpu;

std::string clip(const std::string& s, size_t n = 160) {
    std::string o;
    for (char c : s) {
        if (o.size() >= n) {
            o += "...";
            break;
        }
        o += (c == '\n' || c == '\r' || c == '\t') ? ' ' : c;
    }
    return o;
}

int list(const std::string& path) {
    const auto f = strata::image::LoraFile::open(path);
    std::printf("%s: %zu entries, %.1f MiB on the host\n", path.c_str(), f->entries().size(),
                (double) f->host_bytes() / 1048576.0);
    for (const auto& [k, v] : f->metadata()) std::printf("  meta %s = %s\n", k.c_str(), clip(v).c_str());
    for (const auto& e : f->entries()) {
        char oft[64] = "";
        if (e.kind == strata::image::LoraKind::oft && !e.oft.empty())
            std::snprintf(oft, sizeof oft, "%s %lld x %lld%s", e.oft.stages > 1 ? " boft" : " oft",
                          (long long) e.oft.blocks, (long long) e.oft.bs,
                          e.oft.stages > 1 ? (" x " + std::to_string(e.oft.stages)).c_str() : "");
        std::printf("  %-56s %-5s rank %-4lld %s %lld x %lld scale %.4f%s%s%s%s%s\n", e.key.c_str(),
                    strata::image::lora_kind_name(e.kind), (long long) e.rank, "in/out", (long long) e.cols,
                    (long long) e.rows, e.scale(), e.dora.empty() ? "" : (e.dora_over_rows ? " dora" : " dora(cols)"),
                    e.delta.empty() ? "" : (e.replaces ? " set_weight" : " diff"),
                    e.bias.empty() ? "" : " +bias", e.extra.empty() ? "" : " +tensors", oft);
    }
    return 0;
}

struct Ref {
    double dot = 0, ng = 0, nw = 0, err = 0, mag = 0, mx = 0;
};

// The cosine of the GPU's output and the reference's (an accumulator of 4096 values and more stays FP32 in both), and
// the relative error; true when the cosine is above 0.9999, the selftest's pass.
bool compare(const char* what, const std::vector<float>& got, const std::vector<double>& want) {
    Ref r;
    for (size_t i = 0; i < got.size(); ++i) {
        r.dot += (double) got[i] * want[i];
        r.ng += (double) got[i] * got[i];
        r.nw += want[i] * want[i];
        const double e = std::fabs((double) got[i] - want[i]);
        r.err += e;
        r.mag += std::fabs(want[i]);
        r.mx = std::max(r.mx, e);
    }
    const double c = r.dot / std::sqrt(r.ng * r.nw + 1e-30);
    std::printf("%-22s cos %.7f  rel %.3g  max %.3g%s\n", what, c, r.err / r.mag, r.mx,
                c > 0.9999 ? "" : "  WRONG");
    return c > 0.9999;
}

// one entry's delta: y[t][row0 + c] += scale * row_scale[t] * sum_r (x[t] . a[r]) b[src_row + c][r], the entry's
// rows [src_row, src_row + n) at the weight's [row0, row0 + n)
void selftest_case(void* s, int64_t T, int64_t K, int64_t R, int64_t N, bool rows_scaled, int64_t src_row = 0,
                   int64_t row0 = 0, int64_t n = 0) {
    namespace si = strata::image;
    if (n <= 0) n = N - src_row;
    std::mt19937 rng((uint32_t) (rows_scaled ? 7u : 3u) ^ (uint32_t) (T * 31 + K * 17 + R * 13 + N * 7));
    std::normal_distribution<float> nd(0.0f, 0.5f);
    si::LoraEntry e;
    e.key = "selftest";
    e.kind = si::LoraKind::lora;
    e.cols = K;
    e.rows = N;
    e.rank = R;
    std::vector<float> a((size_t) (R * K)), b((size_t) (N * R)), x((size_t) (T * K)), rs((size_t) T, 1.0f);
    for (float& v : a) v = nd(rng);
    for (float& v : b) v = nd(rng);
    for (float& v : x) v = nd(rng);
    for (int64_t t = 0; t < T; ++t) rs[(size_t) t] = std::exp2((float) (t % 3));   // 1, 2, 4
    si::lora_to_f16(a, e.a);
    si::lora_to_f16(b, e.b);
    const float scale = 0.8f;

    si::LoraDeviceEntry dev;
    std::string err;
    if (!dev.upload(e, s, err)) throw std::runtime_error(err);
    strata::prefill::Gemm gemm;
    if (!gemm.init(s, 0, err)) throw std::runtime_error(err);
    auto* dx = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * K) * 2));
    auto* dy = static_cast<float*>(gpu::alloc_device((size_t) (T * N) * 4));
    auto* drs = static_cast<float*>(gpu::alloc_device((size_t) T * 4));
    auto* tf = static_cast<float*>(gpu::alloc_device((size_t) (T * R) * 4));
    auto* t16 = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * R) * 2));
    std::vector<uint16_t> x16((size_t) (T * K));
    si::lora_to_f16(x, x16);
    std::vector<float> zero((size_t) (T * N), 0.0f);
    gpu::copy(dx, x16.data(), x16.size() * 2);
    gpu::copy(dy, zero.data(), zero.size() * 4);
    gpu::copy(drs, rs.data(), rs.size() * 4);
    dev.add(dx, T, dy, N, src_row, row0, n, scale, rows_scaled ? drs : nullptr, gemm, tf, t16, s);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) (T * N));
    gpu::copy(got.data(), dy, got.size() * 4);
    // the reference, in FP64 from the same FP16 inputs the GPU took
    std::vector<double> want((size_t) (T * N), 0.0);
    std::vector<double> xd((size_t) (T * K)), ad((size_t) (R * K)), bd((size_t) (N * R));
    for (int64_t i = 0; i < T * K; ++i) xd[(size_t) i] = strata::kernels::f32_from_f16(x16[(size_t) i]);
    for (int64_t i = 0; i < R * K; ++i) ad[(size_t) i] = strata::kernels::f32_from_f16(e.a[(size_t) i]);
    for (int64_t i = 0; i < N * R; ++i) bd[(size_t) i] = strata::kernels::f32_from_f16(e.b[(size_t) i]);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t r = 0; r < R; ++r) {
            double v = 0;
            for (int64_t k = 0; k < K; ++k) v += xd[(size_t) (t * K + k)] * ad[(size_t) (r * K + k)];
            for (int64_t c = 0; c < n; ++c)
                want[(size_t) (t * N + row0 + c)] +=
                    v * bd[(size_t) ((src_row + c) * R + r)] * scale * (rows_scaled ? rs[(size_t) t] : 1.0);
        }
    char label[64];
    std::snprintf(label, sizeof label, "%s %lldx%lld rank %lld", rows_scaled ? "row-scaled" : "plain",
                  (long long) T, (long long) N, (long long) R);
    const bool ok = compare(label, got, want);
    gpu::free(dx);
    gpu::free(dy);
    gpu::free(drs);
    gpu::free(tf);
    gpu::free(t16);
    if (!ok) throw std::runtime_error(std::string("the delta of ") + label + " is not the reference's");
}

// one LoKr entry's delta: y[t][row0 + c] += scale * row_scale[t] * sum_j w1[i][j] u[t][j][k], the weight's row
// src_row + c = i * out_k + k (the Kronecker product of w1 [out_l][in_m] and w2 [out_k][in_n])
void selftest_lokr(void* s, int64_t T, int64_t out_l, int64_t in_m, int64_t out_k, int64_t in_n, bool rows_scaled,
                   int64_t src_row = 0, int64_t row0 = 0, int64_t n = 0) {
    namespace si = strata::image;
    const int64_t rows = out_l * out_k, cols = in_m * in_n;
    if (n <= 0) n = rows - src_row;
    std::mt19937 rng((uint32_t) (rows_scaled ? 11u : 5u) ^
                     (uint32_t) (out_l * 97 + in_m * 31 + out_k * 17 + in_n * 7));
    std::normal_distribution<float> nd(0.0f, 0.5f);
    si::LoraEntry e;
    e.key = "selftest-lokr";
    e.kind = si::LoraKind::lokr;
    e.lokr = {out_l, in_m, out_k, in_n};
    e.rows = rows;
    e.cols = cols;
    e.rank = 0;   // both matrices whole: the file's alpha is not divided
    e.extra["lokr_w1"].resize((size_t) (out_l * in_m));
    e.extra["lokr_w2"].resize((size_t) (out_k * in_n));
    for (float& v : e.extra["lokr_w1"]) v = nd(rng);
    for (float& v : e.extra["lokr_w2"]) v = nd(rng);
    std::vector<float> x((size_t) (T * cols)), rs((size_t) T, 1.0f);
    for (float& v : x) v = nd(rng);
    for (int64_t t = 0; t < T; ++t) rs[(size_t) t] = std::exp2((float) (t % 3));   // 1, 2, 4
    const float scale = 0.8f;

    si::LoraDeviceEntry dev;
    std::string err;
    if (!dev.upload(e, s, err)) throw std::runtime_error(err);
    strata::prefill::Gemm gemm;
    if (!gemm.init(s, 0, err)) throw std::runtime_error(err);
    auto* dx = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * cols) * 2));
    auto* dy = static_cast<float*>(gpu::alloc_device((size_t) (T * rows) * 4));
    auto* drs = static_cast<float*>(gpu::alloc_device((size_t) T * 4));
    auto* tf = static_cast<float*>(gpu::alloc_device((size_t) (T * in_m * out_k) * 4));
    auto* t16 = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * si::lora_scratch_cols(e)) * 2));
    std::vector<uint16_t> x16((size_t) (T * cols));
    si::lora_to_f16(x, x16);
    std::vector<float> zero((size_t) (T * rows), 0.0f);
    gpu::copy(dx, x16.data(), x16.size() * 2);
    gpu::copy(dy, zero.data(), zero.size() * 4);
    gpu::copy(drs, rs.data(), rs.size() * 4);
    dev.add(dx, T, dy, rows, src_row, row0, n, scale, rows_scaled ? drs : nullptr, gemm, tf, t16, s);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) (T * rows));
    gpu::copy(got.data(), dy, got.size() * 4);
    // the reference, in FP64 from the same FP16 inputs the GPU took (w1 and w2 go to the GPU as FP16)
    std::vector<uint16_t> h1((size_t) (out_l * in_m)), h2((size_t) (out_k * in_n));
    si::lora_to_f16(e.extra["lokr_w1"], h1);
    si::lora_to_f16(e.extra["lokr_w2"], h2);
    std::vector<double> want((size_t) (T * rows), 0.0), xd((size_t) (T * cols));
    for (int64_t i = 0; i < T * cols; ++i) xd[(size_t) i] = strata::kernels::f32_from_f16(x16[(size_t) i]);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < n; ++c) {
            const int64_t row = src_row + c;
            const int64_t oi = row / out_k, ok = row - oi * out_k;
            double v = 0;
            for (int64_t j = 0; j < cols; ++j) {
                const double w =
                    (double) strata::kernels::f32_from_f16(h1[(size_t) (oi * in_m + j / in_n)]) *
                    (double) strata::kernels::f32_from_f16(h2[(size_t) (ok * in_n + j % in_n)]);
                v += xd[(size_t) (t * cols + j)] * w;
            }
            want[(size_t) (t * rows + row0 + c)] += v * scale * (rows_scaled ? rs[(size_t) t] : 1.0);
        }
    char label[96];
    std::snprintf(label, sizeof label, "lokr %lldx%lld (%lldx%lld (x) %lldx%lld)", (long long) rows, (long long) cols,
                  (long long) out_l, (long long) in_m, (long long) out_k, (long long) in_n);
    const bool ok = compare(label, got, want);
    gpu::free(dx);
    gpu::free(dy);
    gpu::free(drs);
    gpu::free(tf);
    gpu::free(t16);
    if (!ok) throw std::runtime_error(std::string("the delta of ") + label + " is not the reference's");
}

// one whole delta (diff, set_weight): y[t][row0 + c] += scale * row_scale[t] * sum_k x[t][k] Delta[src_row + c][k],
// the weight's own product (a set_weight's, the weight interpolated towards: `base` scales the base product's rows)
void selftest_diff(void* s, int64_t T, int64_t K, int64_t N, bool rows_scaled, float base = 0.0f,
                   int64_t src_row = 0, int64_t row0 = 0, int64_t n = 0) {
    namespace si = strata::image;
    if (n <= 0) n = N - src_row;
    std::mt19937 rng((uint32_t) (rows_scaled ? 23u : 19u) ^ (uint32_t) (T * 29 + K * 11 + N * 5));
    std::normal_distribution<float> nd(0.0f, 0.5f);
    si::LoraEntry e;
    e.key = "selftest-diff";
    e.kind = si::LoraKind::diff;
    e.cols = K;
    e.rows = N;
    e.replaces = base > 0.0f;   // a set_weight: its base factor is 1 - the strength
    std::vector<float> w((size_t) (N * K)), w0((size_t) (N * K)), x((size_t) (T * K)), rs((size_t) T, 1.0f);
    for (float& v : w) v = nd(rng);
    for (float& v : w0) v = nd(rng);
    for (float& v : x) v = nd(rng);
    for (int64_t t = 0; t < T; ++t) rs[(size_t) t] = std::exp2((float) (t % 3));   // 1, 2, 4
    si::lora_to_f16(w, e.delta);
    const float scale = 0.8f;

    si::LoraLayer layer;
    si::LoraPart part;
    auto dev = std::make_shared<si::LoraDeviceEntry>();
    std::string err;
    if (!dev->upload(e, s, err)) throw std::runtime_error(err);
    part.entry = dev;
    part.row0 = row0;
    part.n = n;
    part.src_row = src_row;
    part.scale = scale;
    if (e.replaces) part.base.assign((size_t) n, 1.0f - scale);
    layer.parts.push_back(std::move(part));
    if (!layer.build_base(N, s, err)) throw std::runtime_error(err);

    strata::prefill::Gemm gemm;
    if (!gemm.init(s, 0, err)) throw std::runtime_error(err);
    auto* dx = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * K) * 2));
    auto* dy = static_cast<float*>(gpu::alloc_device((size_t) (T * N) * 4));
    auto* dw = static_cast<uint16_t*>(gpu::alloc_device((size_t) (N * K) * 2));
    auto* drs = static_cast<float*>(gpu::alloc_device((size_t) T * 4));
    auto* tf = static_cast<float*>(gpu::alloc_device((size_t) (T * si::lora_scratch_cols(e)) * 4));
    auto* t16 = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * si::lora_scratch_cols(e)) * 2));
    std::vector<uint16_t> x16((size_t) (T * K)), w016((size_t) (N * K));
    si::lora_to_f16(x, x16);
    si::lora_to_f16(w0, w016);
    std::vector<float> zero((size_t) (T * N), 0.0f);
    gpu::copy(dx, x16.data(), x16.size() * 2);
    gpu::copy(dw, w016.data(), w016.size() * 2);
    gpu::copy(dy, zero.data(), zero.size() * 4);
    gpu::copy(drs, rs.data(), rs.size() * 4);
    if (e.replaces) {   // the weight's own product first, its rows scaled by 1 - the strength (a DiT's product)
        gemm.f16(dx, dw, dy, T, N, K);
        si::qi_rows_scale(dy, T, N, N, layer.base(), s);
    }
    layer.add(dx, T, dy, N, 0, N, rows_scaled ? drs : nullptr, gemm, tf, t16, s);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) (T * N));
    gpu::copy(got.data(), dy, got.size() * 4);
    // the reference, FP64 from the FP16 values the GPU took
    std::vector<uint16_t> hd((size_t) (N * K));
    si::lora_to_f16(w, hd);
    std::vector<double> want((size_t) (T * N), 0.0), xd((size_t) (T * K));
    for (int64_t i = 0; i < T * K; ++i) xd[(size_t) i] = strata::kernels::f32_from_f16(x16[(size_t) i]);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < n; ++c) {
            const int64_t r = row0 + c;
            double& y = want[(size_t) (t * N + r)];
            if (e.replaces)
                for (int64_t k = 0; k < K; ++k)
                    y += (double) (1.0f - scale) * strata::kernels::f32_from_f16(w016[(size_t) (r * K + k)]) *
                         xd[(size_t) (t * K + k)];
            for (int64_t k = 0; k < K; ++k)
                y += (double) strata::kernels::f32_from_f16(hd[(size_t) ((src_row + c) * K + k)]) *
                     xd[(size_t) (t * K + k)] * scale * (rows_scaled ? rs[(size_t) t] : 1.0);
        }
    char label[96];
    std::snprintf(label, sizeof label, "%s %lldx%lld", e.replaces ? "set_weight" : "diff", (long long) T,
                  (long long) N);
    const bool ok = compare(label, got, want);
    gpu::free(dx);
    gpu::free(dy);
    gpu::free(dw);
    gpu::free(drs);
    gpu::free(tf);
    gpu::free(t16);
    if (!ok) throw std::runtime_error(std::string("the delta of ") + label + " is not the reference's");
}

// one LoHa entry: its four factors' delta ((w1a w1b) * (w2a w2b), made when it is bound) and the weight's own
// product from there, with the strength and a W8A8 product's row scale; `tucker`: the sides' cores (a linear product's,
// 1 x 1) in the factors' place, the factors then written the core's way ([core][out])
void selftest_loha(void* s, int64_t T, int64_t K, int64_t R1, int64_t R2, int64_t N, bool rows_scaled, bool tucker,
                   int64_t src_row = 0, int64_t row0 = 0, int64_t n = 0) {
    namespace si = strata::image;
    if (n <= 0) n = N - src_row;
    const int64_t c1 = tucker ? 3 : 0, c2 = tucker ? 4 : 0;   // the cores' first dimensions (their rank)
    std::mt19937 rng((uint32_t) (rows_scaled ? 37u : 29u) ^
                     (uint32_t) (T * 41 + K * 23 + R1 * 13 + R2 * 7 + N * 3 + (tucker ? 1u : 0u)));
    std::normal_distribution<float> nd(0.0f, 0.5f);
    si::LoraEntry e;
    e.key = "selftest-loha";
    e.kind = si::LoraKind::loha;
    e.cols = K;
    e.rows = N;
    e.rank = R1;   // the rank: w1_b's rows
    const auto fill = [&](const char* role, std::vector<int64_t> shape) {
        size_t n_values = 1;
        for (int64_t d : shape) n_values *= (size_t) d;
        e.shapes[role] = std::move(shape);
        e.extra[role].resize(n_values);
        for (float& v : e.extra[role]) v = nd(rng);
    };
    if (tucker) {   // the factors the core's way: w_a [core][out], w_b [rank][in]
        fill("hada_w1_a", {c1, N});
        fill("hada_w1_b", {R1, K});
        fill("hada_w2_a", {c2, N});
        fill("hada_w2_b", {R2, K});
        fill("hada_t1", {c1, R1, 1, 1});
        fill("hada_t2", {c2, R2, 1, 1});
    } else {
        fill("hada_w1_a", {N, R1});
        fill("hada_w1_b", {R1, K});
        fill("hada_w2_a", {N, R2});
        fill("hada_w2_b", {R2, K});
    }
    std::vector<float> x((size_t) (T * K)), rs((size_t) T, 1.0f);
    for (float& v : x) v = nd(rng);
    for (int64_t t = 0; t < T; ++t) rs[(size_t) t] = std::exp2((float) (t % 3));   // 1, 2, 4
    const float scale = 0.8f;

    si::LoraDeviceEntry dev;
    std::string err;
    if (!dev.upload(e, s, err)) throw std::runtime_error(err);
    strata::prefill::Gemm gemm;
    if (!gemm.init(s, 0, err)) throw std::runtime_error(err);
    auto* dx = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * K) * 2));
    auto* dy = static_cast<float*>(gpu::alloc_device((size_t) (T * N) * 4));
    auto* drs = static_cast<float*>(gpu::alloc_device((size_t) T * 4));
    auto* tf = static_cast<float*>(gpu::alloc_device((size_t) (T * si::lora_scratch_cols(e)) * 4));
    auto* t16 = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * si::lora_scratch_cols(e)) * 2));
    std::vector<uint16_t> x16((size_t) (T * K));
    si::lora_to_f16(x, x16);
    std::vector<float> zero((size_t) (T * N), 0.0f);
    gpu::copy(dx, x16.data(), x16.size() * 2);
    gpu::copy(dy, zero.data(), zero.size() * 4);
    gpu::copy(drs, rs.data(), rs.size() * 4);
    dev.add(dx, T, dy, N, src_row, row0, n, scale, rows_scaled ? drs : nullptr, gemm, tf, t16, s);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) (T * N));
    gpu::copy(got.data(), dy, got.size() * 4);
    // the delta as the engine makes it, in FP64 from the FP32 factors: each side's product (under its core), their
    // element-wise product, then the FP16 the GPU took
    const auto matrix = [&](const char* a_role, const char* b_role, const char* t_role) {
        const std::vector<int64_t>& sa = e.shapes[a_role];
        std::vector<double> m((size_t) (N * K), 0.0);
        if (t_role == nullptr) {
            for (int64_t p = 0; p < N; ++p)
                for (int64_t j = 0; j < sa[1]; ++j) {
                    const double v = e.extra[a_role][(size_t) (p * sa[1] + j)];
                    for (int64_t k = 0; k < K; ++k) m[(size_t) (p * K + k)] += v * e.extra[b_role][(size_t) (j * K + k)];
                }
            return m;
        }
        const std::vector<int64_t>& st = e.shapes[t_role];
        for (int64_t i = 0; i < st[0]; ++i)
            for (int64_t j = 0; j < st[1]; ++j) {
                const double w = e.extra[t_role][(size_t) ((i * st[1] + j) * st[2] * st[3])];
                if (w == 0.0) continue;
                for (int64_t p = 0; p < N; ++p) {
                    const double v = w * e.extra[a_role][(size_t) (i * N + p)];
                    for (int64_t k = 0; k < K; ++k) m[(size_t) (p * K + k)] += v * e.extra[b_role][(size_t) (j * K + k)];
                }
            }
        return m;
    };
    const std::vector<double> m1 = matrix("hada_w1_a", "hada_w1_b", tucker ? "hada_t1" : nullptr);
    const std::vector<double> m2 = matrix("hada_w2_a", "hada_w2_b", tucker ? "hada_t2" : nullptr);
    std::vector<uint16_t> hd((size_t) (N * K));
    for (size_t i = 0; i < hd.size(); ++i) hd[i] = strata::kernels::f16_from_f32((float) (m1[i] * m2[i]));
    std::vector<double> want((size_t) (T * N), 0.0), xd((size_t) (T * K));
    for (int64_t i = 0; i < T * K; ++i) xd[(size_t) i] = strata::kernels::f32_from_f16(x16[(size_t) i]);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < n; ++c) {
            const int64_t row = src_row + c;
            double v = 0;
            for (int64_t k = 0; k < K; ++k)
                v += (double) strata::kernels::f32_from_f16(hd[(size_t) (row * K + k)]) * xd[(size_t) (t * K + k)];
            want[(size_t) (t * N + row0 + c)] += v * scale * (rows_scaled ? rs[(size_t) t] : 1.0);
        }
    char label[96];
    std::snprintf(label, sizeof label, "loha%s %lldx%lld rank %lld", tucker ? " (tucker)" : "", (long long) T,
                  (long long) N, (long long) R1);
    const bool ok = compare(label, got, want);
    gpu::free(dx);
    gpu::free(dy);
    gpu::free(drs);
    gpu::free(tf);
    gpu::free(t16);
    if (!ok) throw std::runtime_error(std::string("the delta of ") + label + " is not the reference's");
}

// one DoRA entry: the base product's rows scaled by 1 + strength (c - 1), then the delta's rows by strength (file
// scale) * c; c = dora_scale / (||W + (alpha/rank) BA|| + eps), PEFT's composed row norm (the weight and the factors
// as the product takes them, FP16).  The shared expansion (dit_dora_norms) is checked against PEFT's own form: the
// composed weight made and its rows' norm taken directly.
void selftest_dora(void* s, int64_t T, int64_t K, int64_t R, int64_t N) {
    namespace si = strata::image;
    std::mt19937 rng((uint32_t) (T * 31 + K * 17 + R * 13 + N * 7 + 1));
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::normal_distribution<float> md(1.0f, 0.25f);
    si::LoraEntry e;
    e.key = "selftest-dora";
    e.kind = si::LoraKind::lora;
    e.cols = K;
    e.rows = N;
    e.rank = R;
    e.alpha = 2.0f;
    e.alpha_set = true;
    e.dora.resize((size_t) N);
    for (float& v : e.dora) v = std::fabs(md(rng)) + 0.05f;
    std::vector<float> a((size_t) (R * K)), b((size_t) (N * R)), x((size_t) (T * K)), w((size_t) (N * K));
    for (float& v : a) v = nd(rng);
    for (float& v : b) v = nd(rng);
    for (float& v : x) v = nd(rng);
    for (float& v : w) v = nd(rng) * 0.1f;
    si::lora_to_f16(a, e.a);
    si::lora_to_f16(b, e.b);
    const float strength = 0.7f, scale = strength * e.scale();
    // the composed norms ||W_r + (alpha/rank) (BA)_r||, from the FP16 values the product takes
    std::vector<uint16_t> w16((size_t) (N * K));
    si::lora_to_f16(w, w16);
    const auto read = [&w16, K](int64_t r0, int64_t n, float* out, std::string&) {
        for (int64_t r = 0; r < n; ++r)
            for (int64_t k = 0; k < K; ++k)
                out[r * K + k] = strata::kernels::f32_from_f16(w16[(size_t) ((r0 + r) * K + k)]);
        return true;
    };
    std::vector<float> norms;
    std::string err;
    if (!si::dit_dora_norms(e, 0, 0, N, K, read, norms, err)) throw std::runtime_error(err);
    // PEFT's own form (tuners/lora/dora.py's get_weight_norm): the composed weight made, its rows' norm directly
    const float fs = e.scale();
    for (int64_t r = 0; r < N; ++r) {
        double n2 = 0;
        for (int64_t k = 0; k < K; ++k) {
            double v = strata::kernels::f32_from_f16(w16[(size_t) (r * K + k)]);
            for (int64_t j = 0; j < R; ++j)
                v += (double) fs * strata::kernels::f32_from_f16(e.b[(size_t) (r * R + j)]) *
                     strata::kernels::f32_from_f16(e.a[(size_t) (j * K + k)]);
            n2 += v * v;
        }
        const double want = std::sqrt(n2), got = norms[(size_t) r];
        if (std::fabs(got - want) > 2e-5 * std::max(want, 1e-6))
            throw std::runtime_error("the DoRA's composed norm is not PEFT's direct form's");
    }
    std::vector<float> coef, base;
    if (!si::lora_dora_rows(e, 0, N, norms, strength, coef, base, err)) throw std::runtime_error(err);

    si::LoraLayer layer;
    si::LoraPart part;
    auto dev = std::make_shared<si::LoraDeviceEntry>();
    if (!dev->upload(e, s, err, coef.data())) throw std::runtime_error(err);
    part.entry = dev;
    part.row0 = 0;
    part.n = N;
    part.scale = scale;
    part.base = base;
    layer.parts.push_back(std::move(part));
    if (!layer.build_base(N, s, err)) throw std::runtime_error(err);

    strata::prefill::Gemm gemm;
    if (!gemm.init(s, 0, err)) throw std::runtime_error(err);
    auto* dx = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * K) * 2));
    auto* dw = static_cast<uint16_t*>(gpu::alloc_device((size_t) (N * K) * 2));
    auto* dy = static_cast<float*>(gpu::alloc_device((size_t) (T * N) * 4));
    auto* tf = static_cast<float*>(gpu::alloc_device((size_t) (T * R) * 4));
    auto* t16 = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * R) * 2));
    std::vector<uint16_t> x16((size_t) (T * K));
    si::lora_to_f16(x, x16);
    gpu::copy(dx, x16.data(), x16.size() * 2);
    gpu::copy(dw, w16.data(), w16.size() * 2);
    gemm.f16(dx, dw, dy, T, N, K);                       // the base product, W x
    si::qi_rows_scale(dy, T, N, N, layer.base(), s);     // a DoRA's rows, as a DiT's product does
    layer.add(dx, T, dy, N, 0, N, nullptr, gemm, tf, t16, s);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) (T * N));
    gpu::copy(got.data(), dy, got.size() * 4);
    // the reference, FP64: the folded B (as upload made it), A and x FP16, the base product FP16-rounded
    std::vector<double> want((size_t) (T * N), 0.0), xd((size_t) (T * K)), ad((size_t) (R * K));
    std::vector<double> bd((size_t) (N * R)), wd((size_t) (N * K));
    for (int64_t i = 0; i < T * K; ++i) xd[(size_t) i] = strata::kernels::f32_from_f16(x16[(size_t) i]);
    for (int64_t i = 0; i < R * K; ++i) ad[(size_t) i] = strata::kernels::f32_from_f16(e.a[(size_t) i]);
    for (int64_t i = 0; i < N * K; ++i) wd[(size_t) i] = strata::kernels::f32_from_f16(w16[(size_t) i]);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t j = 0; j < R; ++j) {
            const uint16_t h = strata::kernels::f16_from_f32(
                strata::kernels::f32_from_f16(e.b[(size_t) (r * R + j)]) * coef[(size_t) r]);
            bd[(size_t) (r * R + j)] = strata::kernels::f32_from_f16(h);
        }
    for (int64_t t = 0; t < T; ++t) {
        std::vector<double> y1((size_t) R);
        for (int64_t r = 0; r < R; ++r) {
            double v = 0;
            for (int64_t k = 0; k < K; ++k) v += xd[(size_t) (t * K + k)] * ad[(size_t) (r * K + k)];
            y1[(size_t) r] = v;
        }
        for (int64_t r = 0; r < N; ++r) {
            double v = 0;
            for (int64_t k = 0; k < K; ++k) v += wd[(size_t) (r * K + k)] * xd[(size_t) (t * K + k)];
            double d = 0;
            for (int64_t j = 0; j < R; ++j) d += y1[(size_t) j] * bd[(size_t) (r * R + j)];
            want[(size_t) (t * N + r)] = (double) base[(size_t) r] * v + (double) scale * d;
        }
    }
    const bool ok = compare("dora", got, want);
    gpu::free(dx);
    gpu::free(dw);
    gpu::free(dy);
    gpu::free(tf);
    gpu::free(t16);
    if (!ok) throw std::runtime_error("the DoRA's product is not the reference's");
}

// a bias's delta (a "diff" patch of a weight's bias): y[t][row0 + c] += scale * bias[src_row + c], alone
// (LoraKind::bias: the file holds no weight's delta) and beside a plain LoRA's, both into the rows of a fused weight's
// second half where that is asked
void selftest_bias(void* s, int64_t T, int64_t N, bool with_weight, int64_t src_row = 0, int64_t row0 = 0,
                   int64_t n = 0) {
    namespace si = strata::image;
    if (n <= 0) n = N - src_row;
    const int64_t K = 64, R = 8;
    std::mt19937 rng((uint32_t) (with_weight ? 23u : 29u) ^ (uint32_t) (T * 31 + N * 7 + src_row * 13));
    std::normal_distribution<float> nd(0.0f, 0.5f);
    si::LoraEntry e;
    e.key = "selftest-bias";
    e.kind = with_weight ? si::LoraKind::lora : si::LoraKind::bias;
    e.rows = N;
    e.cols = with_weight ? K : 0;
    e.rank = with_weight ? R : 0;
    e.bias.resize((size_t) N);
    for (float& v : e.bias) v = nd(rng);
    std::vector<float> a((size_t) (R * K)), b((size_t) (N * R)), x((size_t) (T * K));
    if (with_weight) {
        for (float& v : a) v = nd(rng);
        for (float& v : b) v = nd(rng);
        si::lora_to_f16(a, e.a);
        si::lora_to_f16(b, e.b);
    }
    for (float& v : x) v = nd(rng);

    si::LoraDeviceEntry dev;
    std::string err;
    if (!dev.upload(e, s, err)) throw std::runtime_error(err);
    strata::prefill::Gemm gemm;
    if (!gemm.init(s, 0, err)) throw std::runtime_error(err);
    const int64_t cols = std::max<int64_t>(R, 1);
    auto* dx = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * K) * 2));
    auto* dy = static_cast<float*>(gpu::alloc_device((size_t) (T * N) * 4));
    auto* tf = static_cast<float*>(gpu::alloc_device((size_t) (T * cols) * 4));
    auto* t16 = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * cols) * 2));
    std::vector<uint16_t> x16((size_t) (T * K));
    si::lora_to_f16(x, x16);
    std::vector<float> zero((size_t) (T * N), 0.0f);
    gpu::copy(dx, x16.data(), x16.size() * 2);
    gpu::copy(dy, zero.data(), zero.size() * 4);
    const float wscale = 1.3f, bscale = 0.7f;
    if (with_weight) dev.add(dx, T, dy, N, src_row, row0, n, wscale, nullptr, gemm, tf, t16, s);
    dev.add_bias(dy, T, N, src_row, row0, n, bscale, s);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) (T * N));
    gpu::copy(got.data(), dy, got.size() * 4);
    // the reference, FP64 from the same FP16 inputs the GPU took
    std::vector<double> want((size_t) (T * N), 0.0), xd((size_t) (T * K)), ad((size_t) (R * K));
    std::vector<double> bd((size_t) (N * R));
    for (int64_t i = 0; i < T * K; ++i) xd[(size_t) i] = strata::kernels::f32_from_f16(x16[(size_t) i]);
    if (with_weight) {
        for (int64_t i = 0; i < R * K; ++i) ad[(size_t) i] = strata::kernels::f32_from_f16(e.a[(size_t) i]);
        for (int64_t i = 0; i < N * R; ++i) bd[(size_t) i] = strata::kernels::f32_from_f16(e.b[(size_t) i]);
    }
    for (int64_t t = 0; t < T; ++t) {
        if (with_weight)
            for (int64_t r = 0; r < R; ++r) {
                double v = 0;
                for (int64_t k = 0; k < K; ++k) v += xd[(size_t) (t * K + k)] * ad[(size_t) (r * K + k)];
                for (int64_t c = 0; c < n; ++c)
                    want[(size_t) (t * N + row0 + c)] += v * bd[(size_t) ((src_row + c) * R + r)] * wscale;
            }
        for (int64_t c = 0; c < n; ++c)
            want[(size_t) (t * N + row0 + c)] += (double) bscale * e.bias[(size_t) (src_row + c)];
    }
    char label[64];
    std::snprintf(label, sizeof label, "%s %lldx%lld", with_weight ? "lora+bias" : "bias", (long long) T,
                  (long long) N);
    const bool ok = compare(label, got, want);
    gpu::free(dx);
    gpu::free(dy);
    gpu::free(tf);
    gpu::free(t16);
    if (!ok) throw std::runtime_error(std::string("the bias's delta of ") + label + " is not the reference's");
}

// an OFT's (a BOFT's) turn: y = (1 - s) y + s * rescale * (the stages' rotations of y), the rotations' blocks over the
// weight's rows (an OFT file's one stage reads them as they are, a BOFT's four dimensions in 2^i-row groups)
void selftest_oft(void* s, int64_t T, int64_t N, int64_t bs, int64_t stages, bool boft, bool with_rescale,
                  float strength) {
    namespace si = strata::image;
    const int64_t blocks = N / bs;
    if (bs < 2 || blocks < 2 || N % bs != 0) throw std::runtime_error("selftest_oft: a block size the rows do not take");
    std::mt19937 rng((uint32_t) (T * 131 + N * 17 + bs * 7 + stages * 3 + (with_rescale ? 1 : 0)));
    std::normal_distribution<float> nd(0.0f, 0.5f);
    si::LoraEntry e;
    e.key = "selftest-oft";
    e.kind = si::LoraKind::oft;
    e.oft.stages = stages;
    e.oft.blocks = blocks;
    e.oft.bs = bs;
    e.oft.boft = boft;
    e.rows = N;
    e.cols = 0;
    // the rotations: a random matrix a block, its rows orthonormalized in FP64 (a rotation, as the Cayley transform's is)
    std::vector<float> rot((size_t) (stages * blocks * bs * bs));
    std::vector<double> g((size_t) (bs * bs));
    for (int64_t b = 0; b < stages * blocks; ++b) {
        for (double& v : g) v = (double) nd(rng);
        for (int64_t i = 0; i < bs; ++i) {
            for (int64_t j = 0; j < i; ++j) {
                double d = 0;
                for (int64_t k = 0; k < bs; ++k) d += g[(size_t) (i * bs + k)] * g[(size_t) (j * bs + k)];
                for (int64_t k = 0; k < bs; ++k) g[(size_t) (i * bs + k)] -= d * g[(size_t) (j * bs + k)];
            }
            double norm = 0;
            for (int64_t k = 0; k < bs; ++k) norm += g[(size_t) (i * bs + k)] * g[(size_t) (i * bs + k)];
            norm = std::sqrt(norm);
            for (int64_t k = 0; k < bs; ++k) g[(size_t) (i * bs + k)] /= norm;
        }
        for (int64_t i = 0; i < bs * bs; ++i) rot[(size_t) (b * bs * bs + i)] = (float) g[(size_t) i];
    }
    e.extra["oft_r"] = rot;
    std::vector<float> rescale((size_t) N);
    for (int64_t i = 0; i < N; ++i) rescale[(size_t) i] = std::exp2((float) (i % 3)) * 0.5f;   // 0.5, 1, 2
    if (with_rescale) e.rescale = rescale;

    si::LoraDeviceEntry dev;
    std::string err;
    if (!dev.upload(e, s, err)) throw std::runtime_error(err);
    auto* dy = static_cast<float*>(gpu::alloc_device((size_t) (T * N) * 4));
    auto* tf = static_cast<float*>(gpu::alloc_device((size_t) (T * N) * 4));
    std::vector<float> y0((size_t) (T * N));
    for (float& v : y0) v = nd(rng);
    gpu::copy(dy, y0.data(), y0.size() * 4);
    dev.rotate(dy, T, N, 0, 0, N, strength, tf, s);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) (T * N));
    gpu::copy(got.data(), dy, got.size() * 4);
    // the reference, FP64: the stages one after another, then the one interpolation and the rescale
    std::vector<double> cur((size_t) N), next((size_t) N);
    std::vector<double> want((size_t) (T * N));
    const int64_t group = boft ? 2 : 1, r_b = boft ? bs / 2 : bs;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < N; ++i) cur[(size_t) i] = y0[(size_t) (t * N + i)];
        for (int64_t st = 0; st < stages; ++st) {
            const int64_t k = boft ? (int64_t(1) << st) * (bs / 2) : bs;
            const int64_t steps = k / r_b;
            for (int64_t b = 0; b < blocks; ++b) {
                const int64_t m1 = b / steps, j = b - m1 * steps, base = m1 * group * k + j * r_b;
                for (int64_t i = 0; i < bs; ++i) {
                    double v = 0;
                    for (int64_t m = 0; m < bs; ++m)
                        v += (double) rot[(size_t) ((st * blocks + b) * bs * bs + i * bs + m)] *
                             cur[(size_t) (base + m / group + (m % group) * k)];
                    next[(size_t) (base + i / group + (i % group) * k)] = v;
                }
            }
            cur.swap(next);
        }
        for (int64_t i = 0; i < N; ++i) {
            const double r = with_rescale ? rescale[(size_t) i] : 1.0;
            want[(size_t) (t * N + i)] =
                (1.0 - (double) strength) * y0[(size_t) (t * N + i)] + (double) strength * r * cur[(size_t) i];
        }
    }
    char label[64];
    std::snprintf(label, sizeof label, "%s x%lld %lldx%lld bs %lld s %.2f%s", boft ? "boft" : "oft", (long long) stages,
                  (long long) T,
                  (long long) N, (long long) bs, (double) strength, with_rescale ? " rescale" : "");
    const bool ok = compare(label, got, want);
    gpu::free(dy);
    gpu::free(tf);
    if (!ok) throw std::runtime_error(std::string("the turn of ") + label + " is not the reference's");
}

// a GLoRA: y = W x + scale (W v + B (A x)), v = p2 (p1 x) - the plain form's pair (A, B) and the weight's own product of
// the input its two factors make, the second run by the caller (`output`, a band of tokens at a time: the engine's
// add_glora), on the weight's FP16 rows the way a DiT's F16 product would.  The reference rounds where the engine does
// (the plain pair's first product scaled into FP16, v into FP16), in FP64, from the same FP16 tensors.
void selftest_glora(void* s, int64_t T, int64_t K, int64_t R, int64_t N, bool rows_scaled, bool lycoris, float scale,
                    int64_t src_row = 0, int64_t row0 = 0, int64_t n = 0) {
    namespace si = strata::image;
    if (n <= 0) n = N - src_row;
    std::mt19937 rng((uint32_t) (T * 131 + K * 17 + R * 7 + N * 3 + (lycoris ? 1u : 0u) + (rows_scaled ? 2u : 0u)));
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> w((size_t) (N * K)), x((size_t) (T * K));
    for (float& v : w) v = nd(rng);
    for (float& v : x) v = nd(rng);
    std::vector<float> rs((size_t) T);
    for (int64_t t = 0; t < T; ++t) rs[(size_t) t] = std::exp2((float) (t % 3)) * 0.5f;   // 0.5, 1, 2
    // the file's four tensors, in the reading its shapes take: LyCORIS's a1 [rank][cols], a2 [cols][rank], b1
    // [rank][cols], b2 [rows][rank]; the newer a1 [cols][rank], a2 [rank][cols], b1 [rows][rank], b2 [rank][cols]
    si::LoraEntry e;
    e.key = "selftest-glora";
    e.kind = si::LoraKind::glora;
    e.rows = N;
    e.cols = K;
    e.rank = R;
    const auto make = [&rng, &nd](size_t n) {
        std::vector<float> v(n);
        for (float& x : v) x = nd(rng) * 0.1f;
        return v;
    };
    // the file's four, the reading's shapes: the newer a1 [cols][rank], a2 [rank][cols], b1 [rows][rank], b2
    // [rank][cols]; LyCORIS's a1 [rank][cols], a2 [cols][rank], b1 [rank][cols], b2 [rows][rank]
    std::vector<float> t1 = make(lycoris ? (size_t) (R * K) : (size_t) (K * R));
    std::vector<float> t2 = make(lycoris ? (size_t) (K * R) : (size_t) (R * K));
    std::vector<float> t3 = make(lycoris ? (size_t) (R * K) : (size_t) (N * R));
    std::vector<float> t4 = make(lycoris ? (size_t) (N * R) : (size_t) (R * K));
    // (the input's first factor [rank][cols], its second [cols][rank], A [rank][cols], B [rows][rank]), the engine's
    // four: LyCORIS's reading applies a1 [rank][cols] then a2 [cols][rank], the newer one a2 then a1
    const std::vector<float>& in = lycoris ? t1 : t2;
    const std::vector<float>& out = lycoris ? t2 : t1;
    const std::vector<float>& A = lycoris ? t3 : t4;
    const std::vector<float>& B = lycoris ? t4 : t3;
    si::lora_to_f16(A, e.glora.a);
    si::lora_to_f16(B, e.glora.b);
    e.glora.rows = e.rows;
    e.glora.cols = e.cols;
    e.glora.rank = e.rank;
    si::lora_to_f16(in, e.glora.in);    // [rank][cols], applied to x first
    si::lora_to_f16(out, e.glora.out);  // [cols][rank]
    auto entry = std::make_shared<si::LoraDeviceEntry>();
    std::string err;
    if (!entry->upload(e, s, err)) throw std::runtime_error(err);
    strata::prefill::Gemm gemm;
    if (!gemm.init(s, 0, err)) throw std::runtime_error(err);
    const int64_t cols = std::max(R, K);   // the layers' scratch: the input's two products, then the weight's input
    auto* dx = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * K) * 2));
    auto* dw = static_cast<uint16_t*>(gpu::alloc_device((size_t) (N * K) * 2));
    auto* dy = static_cast<float*>(gpu::alloc_device((size_t) (T * N) * 4));
    auto* dg = static_cast<float*>(gpu::alloc_device((size_t) (T * N) * 4));
    auto* tf = static_cast<float*>(gpu::alloc_device((size_t) (T * cols) * 4));
    auto* t16 = static_cast<uint16_t*>(gpu::alloc_device((size_t) (T * cols) * 2));
    auto* drs = rows_scaled ? static_cast<float*>(gpu::alloc_device((size_t) T * 4)) : nullptr;
    std::vector<uint16_t> x16((size_t) (T * K)), w16((size_t) (N * K));
    si::lora_to_f16(x, x16);
    si::lora_to_f16(w, w16);
    std::vector<float> zero((size_t) (T * N), 0.0f);
    gpu::copy(dx, x16.data(), x16.size() * 2);
    gpu::copy(dw, w16.data(), w16.size() * 2);
    gpu::copy(dy, zero.data(), zero.size() * 4);
    if (drs != nullptr) gpu::copy(drs, rs.data(), rs.size() * 4);
    si::LoraLayer layer;
    si::LoraPart part;
    part.entry = entry;
    part.row0 = row0;
    part.n = n;
    part.src_row = src_row;
    part.scale = scale;
    part.glora = true;
    layer.parts.push_back(std::move(part));
    // the whole weight's rows are the product's: the part's rows are its [row0, row0 + n)
    layer.add(dx, T, dy, N, 0, N, drs, gemm, tf, t16, s);
    layer.add_glora(dx, T, 0, N, drs, gemm, tf, t16, s,
                    [&](const uint16_t* v, int64_t t0, int64_t nt, int64_t r0, int64_t n2) {
                        gemm.f16(v, dw + r0 * K, dg, nt, n2, K, n2);
                        si::lora_output_add(dy + t0 * N, nt, N, r0, dg, n2, s);   // the band is the whole weight
                    });
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) (T * N));
    gpu::copy(got.data(), dy, got.size() * 4);
    // the reference, FP64 from the FP16 tensors, the roundings where the engine's are
    std::vector<double> want((size_t) (T * N), 0.0), xd((size_t) (T * K));
    std::vector<double> u((size_t) R), q((size_t) R), u16((size_t) R), q16((size_t) R), v16((size_t) K);
    const auto f = [](uint16_t h) { return (double) strata::kernels::f32_from_f16(h); };
    for (int64_t i = 0; i < T * K; ++i) xd[(size_t) i] = f(x16[(size_t) i]);
    for (int64_t t = 0; t < T; ++t) {
        const double r = rows_scaled ? (double) rs[(size_t) t] : 1.0;
        // the plain pair: (x . A^T) scaled into FP16, then (it . B^T)
        for (int64_t j = 0; j < R; ++j) {
            double a = 0;
            for (int64_t k = 0; k < K; ++k) a += xd[(size_t) (t * K + k)] * f(e.glora.a[(size_t) (j * K + k)]);
            q16[(size_t) j] = f(strata::kernels::f16_from_f32((float) (a * (double) scale * r)));
        }
        // the input's own path: (x . p1^T) scaled into FP16, (it . p2^T) into FP16, then the weight's product
        for (int64_t j = 0; j < R; ++j) {
            double a = 0;
            for (int64_t k = 0; k < K; ++k) a += xd[(size_t) (t * K + k)] * f(e.glora.in[(size_t) (j * K + k)]);
            u16[(size_t) j] = f(strata::kernels::f16_from_f32((float) (a * (double) scale * r)));
        }
        for (int64_t k = 0; k < K; ++k) {
            double a = 0;
            for (int64_t j = 0; j < R; ++j) a += u16[(size_t) j] * f(e.glora.out[(size_t) (k * R + j)]);
            v16[(size_t) k] = f(strata::kernels::f16_from_f32((float) a));
        }
        for (int64_t c = 0; c < n; ++c) {
            const int64_t row = row0 + c;
            double b = 0;
            for (int64_t j = 0; j < R; ++j) b += q16[(size_t) j] * f(e.glora.b[(size_t) ((src_row + c) * R + j)]);
            double a = 0;
            for (int64_t k = 0; k < K; ++k) a += v16[(size_t) k] * f(w16[(size_t) (row * K + k)]);
            want[(size_t) (t * N + row)] = a + b;
        }
    }
    char label[96];
    std::snprintf(label, sizeof label, "glora%s x%lld %lldx%lld r%lld s %.2f%s", lycoris ? " lycoris" : "", (long long) T,
                  (long long) N, (long long) K, (long long) R, (double) scale, rows_scaled ? " rs" : "");
    const bool ok = compare(label, got, want);
    gpu::free(dx);
    gpu::free(dw);
    gpu::free(dy);
    gpu::free(dg);
    gpu::free(tf);
    gpu::free(t16);
    if (drs != nullptr) gpu::free(drs);
    if (!ok) throw std::runtime_error(std::string("the GLoRA's delta of ") + label + " is not the reference's");
}

int selftest() {
    void* s = strata::core::Runtime::get().create_stream();
    try {
        selftest_case(s, 8, 64, 8, 32, false);
        selftest_case(s, 8, 64, 8, 32, true);
        // the rows of a fused weight: the entry's second half into the weight's second half
        selftest_case(s, 8, 64, 8, 64, false, 32, 32, 32);
        // a rank the matrix engines' accumulating product takes (K a multiple of 32)
        selftest_case(s, 64, 64, 64, 4096, false);
        // a LoKr: both matrices whole, a rank-less entry (the file's alpha is not divided)
        selftest_lokr(s, 8, 4, 8, 8, 4, false);
        // its row scale (a W8A8 product's input rows) and the rows of a fused weight's second half
        selftest_lokr(s, 8, 4, 8, 8, 4, true);
        selftest_lokr(s, 8, 4, 8, 16, 4, false, 32, 16, 16);
        // a block-shaped product, as a DiT's attention is (out_l * out_k = 64 rows of 64)
        selftest_lokr(s, 16, 8, 8, 8, 8, false);
        // a DoRA: the base product's rows scaled by 1 + strength (c - 1), the delta's by strength (file scale) c
        selftest_dora(s, 8, 64, 8, 32);
        // a whole delta: the weight's own product, a strength that scales the input and a W8A8 product's row scale
        selftest_diff(s, 8, 64, 32, false);
        selftest_diff(s, 8, 64, 32, true);
        selftest_diff(s, 8, 64, 32, true, 0.8f);   // a set_weight's base factors, 1 - strength
        // the rows of a fused weight: the entry's second half into the weight's second half
        selftest_diff(s, 8, 64, 64, false, 0.0f, 32, 32, 32);
        // a LoHa: the factors' delta, its row scale and a Tucker core (a linear product's)
        selftest_loha(s, 8, 64, 8, 4, 32, false, false);
        selftest_loha(s, 8, 64, 8, 4, 32, true, false);
        selftest_loha(s, 8, 64, 8, 4, 32, false, true);
        // the rows of a fused weight, the side written the core's way
        selftest_loha(s, 8, 64, 8, 4, 64, false, false, 32, 32, 32);
        selftest_loha(s, 8, 64, 8, 4, 64, true, true, 32, 32, 32);
        // a bias's delta: alone (the file holds no weight's delta) and beside a LoRA's, the second into the rows of a
        // fused weight's second half
        selftest_bias(s, 8, 32, false);
        selftest_bias(s, 8, 32, true);
        selftest_bias(s, 8, 64, true, 32, 32, 32);
        // an OFT's turn: the rows turned in place (a block a work-group), the one interpolation, its rescale; a BOFT's
        // stages one after another, an m = 1 one's rows crossed too (the four dimensions)
        selftest_oft(s, 8, 64, 4, 1, false, false, 1.0f);
        selftest_oft(s, 8, 64, 4, 1, false, false, 0.5f);
        selftest_oft(s, 8, 64, 4, 1, false, false, 0.0f);
        selftest_oft(s, 8, 64, 16, 1, false, true, 0.8f);
        selftest_oft(s, 8, 64, 32, 1, true, false, 0.6f);
        selftest_oft(s, 8, 64, 4, 2, true, false, 1.0f);
        selftest_oft(s, 16, 64, 8, 3, true, true, 0.7f);
        // a GLoRA: the plain form's pair and the weight's own product of the input its two factors make (the newer
        // reading, then LyCORIS's older one), a W8A8 product's row scale, and the rows of a fused weight's half
        selftest_glora(s, 8, 64, 8, 32, false, false, 0.7f);
        selftest_glora(s, 8, 64, 8, 32, false, true, 1.0f);
        selftest_glora(s, 8, 64, 8, 32, true, false, 0.7f);
        selftest_glora(s, 8, 64, 8, 64, false, false, 1.0f, 32, 32, 32);
        // the tokens a band at a time (the input's table, kLoraGloraRows)
        selftest_glora(s, 200, 16, 4, 16, false, false, 0.6f);
    } catch (...) {
        strata::core::Runtime::get().destroy_stream(static_cast<sycl::queue*>(s));
        throw;
    }
    strata::core::Runtime::get().destroy_stream(static_cast<sycl::queue*>(s));
    return 0;
}

// The DiT's table of the weights adapters may take, and where each entry of `lora` lands on it (a line a mapped
// weight, MISSING a key the DiT does not take, SHAPE a weight whose rows and columns are not the entry's).
int map(const std::string& dit_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    si::QiDit dit;
    std::string err;
    if (!dit.open(dit_path, nullptr, err, true)) throw std::runtime_error(err);
    const auto targets = dit.lora_targets();
    std::printf("targets %zu\n", targets.size());
    for (const auto& t : targets)
        std::printf("  %-52s rows %lld cols %lld%s\n", t.name.c_str(), (long long) t.rows, (long long) t.cols,
                    t.row0 != 0 ? (" at " + std::to_string(t.row0)).c_str() : "");
    if (lora_path.empty()) return 0;
    const auto f = si::LoraFile::open(lora_path);
    size_t mapped = 0;
    for (const auto& entry : f->entries()) {
        const si::QiDit::LoraTarget* t = nullptr;
        for (const auto& c : targets)
            if (si::lora_key_names(entry.key, c.name)) {
                t = &c;
                break;
            }
        if (t == nullptr) {
            std::printf("MISSING %-46s %-5s rank %-4lld %lld x %lld\n", entry.key.c_str(),
                        si::lora_kind_name(entry.kind), (long long) entry.rank, (long long) entry.cols,
                        (long long) entry.rows);
            continue;
        }
        si::LoraEntry alt;   // a GLoRA's four tensors may fit both readings: the weight's rows and columns choose
        const si::LoraEntry& e = *si::lora_glora_for_weight(entry, t->rows, t->cols, alt);
        if (e.rows != t->rows ||
            (e.kind != si::LoraKind::bias && e.kind != si::LoraKind::oft &&
             e.cols != t->cols)) {   // a bias's delta alone, an OFT's: their columns are none
            std::printf("SHAPE   %-46s %lld x %lld, the weight %lld x %lld\n", e.key.c_str(), (long long) e.cols,
                        (long long) e.rows, (long long) t->cols, (long long) t->rows);
        } else {
            ++mapped;
            std::printf("mapped  %-46s %-5s rank %-4lld rows %lld at the weight's %lld, scale %.4f\n", e.key.c_str(),
                        si::lora_kind_name(e.kind), (long long) e.rank, (long long) e.rows, (long long) t->row0,
                        e.scale() * scale);
        }
    }
    std::printf("%zu of %zu entries mapped\n", mapped, f->entries().size());
    return mapped == f->entries().size() ? 0 : 1;
}

// One forward of a fixed context and noise (philox_randn of seed 42), the velocity written as raw FP32: the same
// numbers as an adapterless run mean the adapter's delta compares equal to the same delta baked into the weights.
int step(const std::string& dit_path, const std::string& out_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    void* s = strata::core::Runtime::get().create_stream();
    si::QiDit dit;
    std::string err;
    if (!dit.open(dit_path, s, err)) throw std::runtime_error(err);
    if (std::getenv("LORA_STEP_NOBAKE") != nullptr) dit.set_lora_bake(false);   // the adapters added to each product
    if (!lora_path.empty()) {
        std::vector<si::LoraUse> loras;
        loras.push_back({si::LoraFile::open(lora_path), scale, {}});
        if (!dit.set_loras(loras, err)) throw std::runtime_error(err);
    }
    const char* hw_env = std::getenv("QI_STEP_HW");
    const char* text_env = std::getenv("QI_STEP_TEXT");
    const int64_t hw = hw_env != nullptr ? std::strtoll(hw_env, nullptr, 10) : 8;
    const int64_t n_text = text_env != nullptr ? std::strtoll(text_env, nullptr, 10) : 4;
    std::vector<float> ctx((size_t) (n_text * dit.hidden()));
    for (size_t i = 0; i < ctx.size(); ++i) ctx[i] = (float) ((int64_t) ((i * 37) % 101) - 50) / 50.0f;   // exact both sides
    dit.set_context(ctx.data(), n_text);
    const int64_t n = dit.channels() * hw * hw;
    const auto noise = si::philox_randn(42, (uint64_t) n);
    float *x = nullptr, *v = nullptr;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&v, (size_t) n * 4)) throw std::runtime_error("alloc");
    gpu::copy(x, noise.data(), noise.size() * 4);
    dit.forward(x, 500.0f, hw, hw, false, v);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) n);
    gpu::copy(got.data(), v, got.size() * 4);
    // the prefix cache: a step after the first one reads the context's K and V from it (the adapters' included)
    float* vp = nullptr;
    if (!gpu::alloc_device(&vp, (size_t) n * 4)) throw std::runtime_error("alloc");
    dit.forward(x, 500.0f, hw, hw, true, vp);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> cached((size_t) n);
    gpu::copy(cached.data(), vp, cached.size() * 4);
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; ++i) {
        dot += (double) got[(size_t) i] * cached[(size_t) i];
        na += (double) got[(size_t) i] * got[(size_t) i];
        nb += (double) cached[(size_t) i] * cached[(size_t) i];
    }
    std::printf("prefix cache vs the whole sequence: cos %.7f\n", dot / std::sqrt(na * nb));
    gpu::free(vp);
    std::ofstream fo(out_path, std::ios::binary);
    if (!fo) throw std::runtime_error("cannot write " + out_path);
    fo.write(reinterpret_cast<const char*>(got.data()), (std::streamsize) (got.size() * 4));
    double norm = 0;
    for (float g : got) norm += (double) g * g;
    std::printf("velocity %lld values, norm %.6f, first %.6f %.6f %.6f\n", (long long) n, std::sqrt(norm), got[0],
                got[1], got[2]);
    std::printf("lora %.1f MiB on the GPU%s%s\n", (double) dit.lora_bytes() / 1048576.0,
                dit.lora_key().empty() ? " (none)" : "", dit.lora_key().empty() ? "" : dit.lora_key().c_str());
    std::printf("products: %s\n", strata::prefill::Gemm::i8_usable(s) ? "int8 (W8A8) where the weights allow it"
                                                                     : "FP16/native (no int8 products)");
    gpu::free(x);
    gpu::free(v);
    return 0;
}

}  // namespace

namespace {

// n BF16 values on the device (MossDit's rows) as floats
std::vector<float> bf16_rows(const uint16_t* d, size_t n) {
    std::vector<uint16_t> b(n);
    strata::gpu::copy(b.data(), d, n * 2);
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = strata::kernels::f32_from_bf16(b[i]);
    return h;
}

// MOSS-Image's table of the weights adapters may take (one line a name: its entry's rows and columns), and where each
// entry of `lora` lands on it (a line a mapped weight, MISSING a key the DiT does not take, SHAPE a weight whose rows
// and columns are not the entry's).
int moss_map(const std::string& dit_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    si::MossDit dit;
    std::string err;
    if (!dit.open(dit_path, nullptr, err, true)) throw std::runtime_error(err);
    const auto targets = dit.lora_targets();
    std::vector<si::MossDit::LoraTarget> groups;   // one per name: the rows and columns of the weights it covers
    for (const auto& t : targets) {
        auto g = std::find_if(groups.begin(), groups.end(), [&](const auto& x) { return x.name == t.name; });
        if (g == groups.end()) {
            groups.push_back(t);
        } else {
            g->rows = std::max(g->rows, t.src_row + t.rows);
        }
    }
    std::printf("targets %zu\n", groups.size());
    for (const auto& g : groups)
        std::printf("  %-52s rows %lld cols %lld\n", g.name.c_str(), (long long) g.rows, (long long) g.cols);
    if (lora_path.empty()) return 0;
    const auto f = si::LoraFile::open(lora_path);
    size_t mapped = 0;
    for (const auto& entry : f->entries()) {
        const si::MossDit::LoraTarget* g = nullptr;
        for (const auto& c : groups)
            if (si::lora_key_names(entry.key, c.name)) {
                g = &c;
                break;
            }
        if (g == nullptr) {
            std::printf("MISSING %-46s %-5s rank %-4lld %lld x %lld\n", entry.key.c_str(),
                        si::lora_kind_name(entry.kind), (long long) entry.rank, (long long) entry.cols,
                        (long long) entry.rows);
            continue;
        }
        si::LoraEntry alt;   // a GLoRA's four tensors may fit both readings: the weight's rows and columns choose
        const si::LoraEntry& e = *si::lora_glora_for_weight(entry, g->rows, g->cols, alt);
        if (e.rows != g->rows ||
                   (e.kind != si::LoraKind::bias && e.kind != si::LoraKind::oft &&
                    e.cols != g->cols)) {   // a bias's delta alone, an OFT's: their columns are none
            std::printf("SHAPE   %-46s %lld x %lld, the weight %lld x %lld\n", e.key.c_str(), (long long) e.cols,
                        (long long) e.rows, (long long) g->cols, (long long) g->rows);
        } else {
            ++mapped;
            std::printf("mapped  %-46s %-5s rank %-4lld rows %lld, scale %.4f\n", e.key.c_str(),
                        si::lora_kind_name(e.kind), (long long) e.rank, (long long) e.rows, e.scale() * scale);
        }
    }
    std::printf("%zu of %zu entries mapped\n", mapped, f->entries().size());
    return mapped == f->entries().size() ? 0 : 1;
}

// One forward of a fixed context and noise (torch_cpu_randn of seed 42), the velocity written as raw FP32: the same
// numbers as an adapterless run of a model with the same delta in its weights mean the delta is applied as baked.
int moss_step(const std::string& dit_path, const std::string& out_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    void* s = strata::core::Runtime::get().create_stream();
    si::MossDit dit;
    std::string err;
    const bool q8 = std::getenv("MOSS_STEP_Q8") != nullptr;
    if (!dit.open(dit_path, s, err, false, q8)) throw std::runtime_error(err);
    if (std::getenv("LORA_STEP_NOBAKE") != nullptr) dit.set_lora_bake(false);   // the adapters added to each product
    if (const char* mib = std::getenv("MOSS_STEP_BLOCK_MIB"); mib != nullptr && *mib != 0) {   // VRAM of the blocks
        dit.set_block_budget((uint64_t) std::strtoull(mib, nullptr, 10) << 20);
        std::printf("the blocks' VRAM %s MiB, %d of %d resident\n", mib, dit.resident_blocks(), dit.blocks());
    }
    if (!lora_path.empty()) {
        std::vector<si::LoraUse> loras;
        loras.push_back({si::LoraFile::open(lora_path), scale, {}});
        if (!dit.set_loras(loras, err)) throw std::runtime_error(err);
    }
    const char* hw_env = std::getenv("MOSS_STEP_HW");
    const char* text_env = std::getenv("MOSS_STEP_TEXT");
    const int64_t hw = hw_env != nullptr ? std::strtoll(hw_env, nullptr, 10) : 2;
    const int64_t n_text = text_env != nullptr ? std::strtoll(text_env, nullptr, 10) : 4;
    std::vector<float> ctx((size_t) (n_text * dit.text_dim()));
    for (size_t i = 0; i < ctx.size(); ++i) ctx[i] = (float) ((int64_t) ((i * 37) % 101) - 50) / 50.0f;   // exact both sides
    dit.set_context(0, ctx.data(), n_text);
    const int64_t Cl = dit.latent_channels(), n = Cl * hw * hw;
    const auto noise = si::torch_cpu_randn(42, (uint64_t) n);
    std::vector<float> tokens((size_t) n);
    for (int64_t c = 0; c < Cl; ++c)
        for (int64_t q = 0; q < hw * hw; ++q) tokens[(size_t) (q * Cl + c)] = noise[(size_t) (c * hw * hw + q)];
    float *x = nullptr, *v = nullptr;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&v, (size_t) n * 4)) throw std::runtime_error("alloc");
    gpu::copy(x, tokens.data(), tokens.size() * 4);
    dit.forward(x, 500.0f, hw, hw, 1, v);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) n);
    gpu::copy(got.data(), v, got.size() * 4);
    if (const char* drop = std::getenv("MOSS_STEP_DROP_TEXT"); drop != nullptr) {
        // the text embedder's weights go (the pipeline gives them up for the blocks' room) and come back: a weight an
        // adapter is bound to keeps its layer across the load, and the velocity is the same again
        const uint64_t before = dit.lora_bytes();
        std::vector<float> rows_before = bf16_rows(dit.text_rows(0), (size_t) (n_text * dit.hidden()));
        if (std::strcmp(drop, "1") == 0) dit.drop_text();
        dit.set_context(0, ctx.data(), n_text);
        std::vector<float> rows_after = bf16_rows(dit.text_rows(0), (size_t) (n_text * dit.hidden()));
        double rd = 0, ra = 0, rb = 0;
        for (size_t i = 0; i < rows_after.size(); ++i) {
            rd += (double) rows_before[i] * rows_after[i];
            ra += (double) rows_before[i] * rows_before[i];
            rb += (double) rows_after[i] * rows_after[i];
        }
        std::printf("the context's text rows: cos %.7f, the same bytes %s\n", rd / std::sqrt(ra * rb + 1e-30),
                    rows_before == rows_after ? "yes" : "no");
        std::printf("the adapter's A and B: %llu bytes before the text embedder's weights went, %llu after them\n",
                    (unsigned long long) before, (unsigned long long) dit.lora_bytes());
        dit.forward(x, 500.0f, hw, hw, 1, v);
        if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
        std::vector<float> again((size_t) n);
        gpu::copy(again.data(), v, again.size() * 4);
        double dot = 0, na = 0, nb = 0;
        for (int64_t i = 0; i < n; ++i) {
            dot += (double) got[(size_t) i] * again[(size_t) i];
            na += (double) got[(size_t) i] * got[(size_t) i];
            nb += (double) again[(size_t) i] * again[(size_t) i];
        }
        std::printf("the text embedder read again: cos %.7f, the same bytes %s\n", dot / std::sqrt(na * nb + 1e-30),
                    got == again ? "yes" : "no");
    }
    std::ofstream fo(out_path, std::ios::binary);
    if (!fo) throw std::runtime_error("cannot write " + out_path);
    fo.write(reinterpret_cast<const char*>(got.data()), (std::streamsize) (got.size() * 4));
    double norm = 0;
    for (float g : got) norm += (double) g * g;
    std::printf("velocity %lld values, norm %.6f, first %.6f %.6f %.6f\n", (long long) n, std::sqrt(norm), got[0],
                got[1], got[2]);
    std::printf("lora %.1f MiB on the GPU, %d files\n", (double) dit.lora_bytes() / 1048576.0, dit.lora_count());
    std::printf("products: %s; blocks %s\n",
                strata::prefill::Gemm::i8_usable(s) ? "int8 (W8A8) where the weights allow it"
                                                    : "FP16/native (no int8 products)",
                q8 ? "8 bits" : "16 bits");
    gpu::free(x);
    gpu::free(v);
    return 0;
}

// Mage-Flow's table of the weights adapters may take (one line a name: its rows and columns), and where each entry of
// `lora` lands on it (a line a mapped weight, MISSING a key the DiT does not take, SHAPE a weight whose rows and
// columns are not the entry's).
int mf_map(const std::string& dit_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    si::MfDit dit;
    std::string err;
    if (!dit.open(dit_path, nullptr, err, true)) throw std::runtime_error(err);
    const auto targets = dit.lora_targets();
    std::printf("targets %zu\n", targets.size());
    for (const auto& t : targets) std::printf("  %-52s rows %lld cols %lld\n", t.name.c_str(), (long long) t.rows,
                                              (long long) t.cols);
    if (lora_path.empty()) return 0;
    const auto f = si::LoraFile::open(lora_path);
    size_t mapped = 0;
    for (const auto& entry : f->entries()) {
        const si::MfDit::LoraTarget* t = nullptr;
        for (const auto& c : targets)
            if (si::lora_key_names(entry.key, c.name)) {
                t = &c;
                break;
            }
        if (t == nullptr) {
            std::printf("MISSING %-46s %-5s rank %-4lld %lld x %lld\n", entry.key.c_str(),
                        si::lora_kind_name(entry.kind), (long long) entry.rank, (long long) entry.cols,
                        (long long) entry.rows);
            continue;
        }
        si::LoraEntry alt;   // a GLoRA's four tensors may fit both readings: the weight's rows and columns choose
        const si::LoraEntry& e = *si::lora_glora_for_weight(entry, t->rows, t->cols, alt);
        if (e.rows != t->rows ||
                   (e.kind != si::LoraKind::bias && e.kind != si::LoraKind::oft &&
                    e.cols != t->cols)) {   // a bias's delta alone, an OFT's: their columns are none
            std::printf("SHAPE   %-46s %lld x %lld, the weight %lld x %lld\n", e.key.c_str(), (long long) e.cols,
                        (long long) e.rows, (long long) t->cols, (long long) t->rows);
        } else {
            ++mapped;
            std::printf("mapped  %-46s %-5s rank %-4lld rows %lld, scale %.4f\n", e.key.c_str(),
                        si::lora_kind_name(e.kind), (long long) e.rank, (long long) e.rows, e.scale() * scale);
        }
    }
    std::printf("%zu of %zu entries mapped\n", mapped, f->entries().size());
    return mapped == f->entries().size() ? 0 : 1;
}

// One forward of a fixed context and noise (torch_cpu_randn of seed 42), the velocity written as raw FP32: the same
// numbers as an adapterless run of a model with the same delta in its weights mean the delta is applied as baked.
int mf_step(const std::string& dit_path, const std::string& out_path, const std::string& lora_path, float scale) {
    namespace si = strata::image;
    void* s = strata::core::Runtime::get().create_stream();
    si::MfDit dit;
    std::string err;
    const bool q8 = std::getenv("MF_STEP_Q8") != nullptr;
    if (!dit.open(dit_path, s, err, false, q8)) throw std::runtime_error(err);
    if (std::getenv("LORA_STEP_NOBAKE") != nullptr) dit.set_lora_bake(false);   // the adapters added to each product
    if (const char* mib = std::getenv("MF_STEP_BLOCK_MIB"); mib != nullptr && *mib != 0) {   // VRAM of the blocks
        dit.set_block_budget((uint64_t) std::strtoull(mib, nullptr, 10) << 20);
        std::printf("the blocks' VRAM %s MiB, %d of %d resident\n", mib, dit.resident_blocks(), (int) dit.blocks());
    }
    if (!lora_path.empty()) {
        std::vector<si::LoraUse> loras;
        loras.push_back({si::LoraFile::open(lora_path), scale, {}});
        if (!dit.set_loras(loras, err)) throw std::runtime_error(err);
    }
    const char* hw_env = std::getenv("MF_STEP_HW");
    const char* text_env = std::getenv("MF_STEP_TEXT");
    const int64_t hw = hw_env != nullptr ? std::strtoll(hw_env, nullptr, 10) : 2;
    const int64_t n_text = text_env != nullptr ? std::strtoll(text_env, nullptr, 10) : 4;
    std::vector<float> ctx((size_t) (n_text * dit.text_dim()));
    for (size_t i = 0; i < ctx.size(); ++i) ctx[i] = (float) ((int64_t) ((i * 37) % 101) - 50) / 50.0f;   // exact both sides
    dit.set_context(0, ctx.data(), n_text);
    // an edit: a reference the target's size, after its rows (the same on both sides)
    std::vector<float> rz;
    std::vector<std::pair<int64_t, int64_t>> rsz;
    if (const char* refs = std::getenv("MF_STEP_REFS"); refs != nullptr && *refs != 0) {
        for (int64_t j = 0; j < std::strtoll(refs, nullptr, 10); ++j) {
            const auto z = si::torch_cpu_randn(43 + (uint64_t) j, (uint64_t) (hw * hw * dit.latent_channels()));
            rz.insert(rz.end(), z.begin(), z.end());
            rsz.emplace_back(hw, hw);
        }
    }
    const int64_t Cl = dit.latent_channels(), n = Cl * hw * hw;
    if (!rz.empty()) dit.set_refs(rz.data(), rsz);
    const auto noise = si::torch_cpu_randn(42, (uint64_t) n);
    std::vector<float> tokens((size_t) n);
    for (int64_t c = 0; c < Cl; ++c)
        for (int64_t q = 0; q < hw * hw; ++q) tokens[(size_t) (q * Cl + c)] = noise[(size_t) (c * hw * hw + q)];
    float *x = nullptr, *v = nullptr;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&v, (size_t) n * 4)) throw std::runtime_error("alloc");
    gpu::copy(x, tokens.data(), tokens.size() * 4);
    dit.forward(x, 0.5f, hw, hw, 1, v);
    if (!gpu::stream_sync(s)) throw std::runtime_error(gpu::last_error());
    std::vector<float> got((size_t) n);
    gpu::copy(got.data(), v, got.size() * 4);
    std::ofstream fo(out_path, std::ios::binary);
    if (!fo) throw std::runtime_error("cannot write " + out_path);
    fo.write(reinterpret_cast<const char*>(got.data()), (std::streamsize) (got.size() * 4));
    double norm = 0;
    for (float g : got) norm += (double) g * g;
    std::printf("velocity %lld values, norm %.6f, first %.6f %.6f %.6f\n", (long long) n, std::sqrt(norm), got[0],
                got[1], got[2]);
    std::printf("lora %.1f MiB on the GPU, %d files\n", (double) dit.lora_bytes() / 1048576.0, dit.lora_count());
    std::printf("products: %s; blocks %s\n",
                strata::prefill::Gemm::i8_usable(s) ? "int8 (W8A8) where the weights allow it"
                                                    : "FP16/native (no int8 products)",
                q8 ? "8 bits" : "16 bits");
    gpu::free(x);
    gpu::free(v);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string what = argc > 1 ? argv[1] : "";
        if (what == "list" && argc == 3) return list(argv[2]);
        if (what == "selftest" && argc == 2) return selftest();
        if (what == "map" && (argc == 3 || argc == 4 || argc == 5))
            return map(argv[2], argc > 3 ? argv[3] : "", argc == 5 ? std::strtof(argv[4], nullptr) : 1.0f);
        if (what == "step" && (argc == 4 || argc == 5 || argc == 6))
            return step(argv[2], argv[3], argc > 4 ? argv[4] : "", argc == 6 ? std::strtof(argv[5], nullptr) : 1.0f);
        if (what == "moss-map" && (argc == 3 || argc == 4 || argc == 5))
            return moss_map(argv[2], argc > 3 ? argv[3] : "", argc == 5 ? std::strtof(argv[4], nullptr) : 1.0f);
        if (what == "moss-step" && (argc == 4 || argc == 5 || argc == 6))
            return moss_step(argv[2], argv[3], argc > 4 ? argv[4] : "", argc == 6 ? std::strtof(argv[5], nullptr) : 1.0f);
        if (what == "mf-map" && (argc == 3 || argc == 4 || argc == 5))
            return mf_map(argv[2], argc > 3 ? argv[3] : "", argc == 5 ? std::strtof(argv[4], nullptr) : 1.0f);
        if (what == "mf-step" && (argc == 4 || argc == 5 || argc == 6))
            return mf_step(argv[2], argv[3], argc > 4 ? argv[4] : "", argc == 6 ? std::strtof(argv[5], nullptr) : 1.0f);
        std::fprintf(stderr,
                     "usage: %s list LORA.safetensors|LORA.gguf | %s selftest | %s map DIT.gguf [LORA [scale]] | "
                     "%s step DIT.gguf OUT.bin [LORA [scale]] | %s moss-map DIR [LORA [scale]] | "
                     "%s moss-step DIR OUT.bin [LORA [scale]] | %s mf-map DIT [LORA [scale]] | "
                     "%s mf-step DIT OUT.bin [LORA [scale]]\n",
                     argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-lora-probe: %s\n", e.what());
        return 1;
    }
}