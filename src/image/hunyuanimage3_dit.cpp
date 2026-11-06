// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_dit.cpp - see include/strata/image/hunyuanimage3_dit.hpp.

#include "strata/image/hunyuanimage3_dit.hpp"

#include "strata/core/gpu.hpp"
#include "strata/image/hunyuanimage3_kernels.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/prefill/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::image {

namespace {

namespace gpu = strata::gpu;

/// The qkv product's rows: head_dim * kv_heads * (groups + 2), each KV head's block being [q0..q(groups-1), k, v].
constexpr int64_t kQkvRows(int64_t heads, int64_t kv_heads, int64_t head_dim) {
    return head_dim * kv_heads * (heads / kv_heads + 2);
}

std::string oom(const char* what) {
    return std::string("HunyuanImage 3.0 (GPU DiT): ") + what + ": " + gpu::last_error();
}

void* take_dev(std::vector<void*>& owned, size_t bytes, const char* what) {
    void* p = nullptr;
    if (!gpu::alloc_device(&p, bytes)) throw std::runtime_error(oom(what));
    owned.push_back(p);
    return p;
}

template <class T> T* take_as(std::vector<void*>& owned, int64_t count, const char* what) {
    return reinterpret_cast<T*>(take_dev(owned, (size_t) std::max<int64_t>(count, 1) * sizeof(T), what));
}

void up(void* dst, const void* src, size_t bytes) {
    if (!gpu::copy(dst, src, bytes)) throw std::runtime_error(oom("copy H2D"));
}

void down(void* dst, const void* src, size_t bytes) {
    if (!gpu::copy(dst, src, bytes)) throw std::runtime_error(oom("copy D2H"));
}

void zero(void* dst, size_t bytes) {
    if (!gpu::memset(dst, 0, bytes)) throw std::runtime_error(oom("memset"));
}

}  // namespace

// --- the layer's scratch

uint64_t hunyuanimage3_scratch_bytes(int64_t T, const HunyuanImage3Dims& d) {
    const int64_t rows = qi_attn_rows(T);
    const int64_t qkv_n = kQkvRows(d.heads, d.kv_heads, d.head_dim);
    uint64_t b = 0;
    auto f16 = [&](int64_t n) { b += (uint64_t) std::max<int64_t>(n, 1) * 2; };
    auto f32 = [&](int64_t n) { b += (uint64_t) std::max<int64_t>(n, 1) * 4; };
    f16(5 * T * d.hidden);                    // xr16, ar16, xr216, a16, gx16
    f16(3 * rows * d.heads * d.head_dim);     // q16, k16, v16
    f16(2 * T * d.inter);                     // act16, gact16
    f16(2 * d.inter * d.hidden);              // egu16: one expert's FP16 gate_up
    f16(d.hidden * d.inter);                  // edn16: one expert's FP16 down
    f32(T * qkv_n);                           // qkv32
    f32(2 * T * d.hidden);                    // o32, ed32
    f32(2 * T * 2 * d.inter);                 // gu32 (the shared expert, then its output), eg32
    f32(T * d.experts);                       // logits
    f32(2 * T * d.hidden);                    // xn32, xn232 (the norms, FP32 up to the rotation)
    f32(T * (d.head_dim / 2));                // ang
    f32(2 * T * d.topk);                      // d_w, d_tw
    b += (uint64_t) std::max<int64_t>(2 * T * d.topk, 1) * sizeof(int32_t);   // d_rows, d_idx
    b += (uint64_t) (2 * d.experts + 1) * sizeof(int32_t);                 // d_bounds, d_cursor
    return b;
}

bool hunyuanimage3_scratch_alloc(int64_t T, const HunyuanImage3Dims& d, HunyuanImage3Scratch& sc, void* stream, std::string& err) {
    (void) stream;
    std::vector<void*> owned;
    try {
        const int64_t rows = qi_attn_rows(T);
        const int64_t qkv_n = kQkvRows(d.heads, d.kv_heads, d.head_dim);
        const int64_t hdots = rows * d.heads * d.head_dim;
        sc.T = T;
        sc.rows = rows;
        sc.xr16 = take_as<uint16_t>(owned, T * d.hidden, "xr16");
        sc.ar16 = take_as<uint16_t>(owned, T * d.hidden, "ar16");
        sc.xr216 = take_as<uint16_t>(owned, T * d.hidden, "xr216");
        sc.a16 = take_as<uint16_t>(owned, T * d.hidden, "a16");
        sc.gx16 = take_as<uint16_t>(owned, T * d.hidden, "gx16");
        sc.q16 = take_as<uint16_t>(owned, hdots, "q16");
        sc.k16 = take_as<uint16_t>(owned, hdots, "k16");
        sc.v16 = take_as<uint16_t>(owned, hdots, "v16");
        zero(sc.q16, (size_t) hdots * 2);   // the rows past T are the attention's padding: they stay zero
        zero(sc.k16, (size_t) hdots * 2);
        zero(sc.v16, (size_t) hdots * 2);
        sc.act16 = take_as<uint16_t>(owned, T * d.inter, "act16");
        sc.gact16 = take_as<uint16_t>(owned, T * d.inter, "gact16");
        sc.egu16 = take_as<uint16_t>(owned, 2 * d.inter * d.hidden, "egu16");
        sc.edn16 = take_as<uint16_t>(owned, d.hidden * d.inter, "edn16");
        sc.qkv32 = take_as<float>(owned, T * qkv_n, "qkv32");
        sc.o32 = take_as<float>(owned, T * d.hidden, "o32");
        sc.gu32 = take_as<float>(owned, T * 2 * d.inter, "gu32");
        sc.eg32 = take_as<float>(owned, T * 2 * d.inter, "eg32");
        sc.ed32 = take_as<float>(owned, T * d.hidden, "ed32");
        sc.logits = take_as<float>(owned, T * d.experts, "logits");
        sc.xn32 = take_as<float>(owned, T * d.hidden, "xn32");
        sc.xn232 = take_as<float>(owned, T * d.hidden, "xn232");
        sc.ang = take_as<float>(owned, T * (d.head_dim / 2), "ang");
        sc.d_rows = take_as<int32_t>(owned, T * d.topk, "d_rows");
        sc.d_w = take_as<float>(owned, T * d.topk, "d_w");
        sc.d_tw = take_as<float>(owned, T * d.topk, "d_tw");
        sc.d_idx = take_as<int32_t>(owned, T * d.topk, "d_idx");
        sc.d_bounds = take_as<int32_t>(owned, d.experts + 1, "d_bounds");
        sc.d_cursor = take_as<int32_t>(owned, d.experts, "d_cursor");
        return true;
    } catch (const std::exception& e) {
        for (size_t i = owned.size(); i-- > 0;) gpu::free(owned[i]);
        sc = HunyuanImage3Scratch{};
        err = e.what();
        return false;
    }
}

void hunyuanimage3_scratch_free(HunyuanImage3Scratch& sc) {
    const void* ps[] = {sc.xr16, sc.ar16, sc.xr216, sc.a16, sc.gx16, sc.q16, sc.k16, sc.v16, sc.act16, sc.gact16,
                        sc.egu16, sc.edn16, sc.qkv32, sc.o32, sc.gu32, sc.eg32, sc.ed32, sc.logits, sc.ang,
                        sc.d_rows, sc.d_w, sc.xn32, sc.xn232, sc.d_tw, sc.d_idx, sc.d_bounds, sc.d_cursor};
    for (const void* p : ps)
        if (p != nullptr) gpu::free(p);
    sc = HunyuanImage3Scratch{};
}

// --- one layer: the arithmetic the direct path and the streamed one share

bool hunyuanimage3_layer_run(const HunyuanImage3LayerWeights& w, HunyuanImage3Experts& experts, HunyuanImage3Scratch& sc, const float* x, float* out,
                   int64_t T, const int32_t* pos_y, const int32_t* pos_x, int64_t n_img, const HunyuanImage3Dims& d,
                   prefill::Gemm& gemm, void* stream, std::string& err, const int32_t* key_lim) {
    if (T <= 0) {
        err = "hunyuanimage3_layer_run: no tokens";
        return false;
    }
    if (sc.T < T) {
        err = "hunyuanimage3_layer_run: the scratch holds " + std::to_string(sc.T) + " tokens, not " + std::to_string(T);
        return false;
    }
    // the products' shapes, against the model's: the scratch is sized for the model's
    if (w.qkv_n != kQkvRows(d.heads, d.kv_heads, d.head_dim) || w.qkv_k != d.hidden || w.o_n != d.hidden ||
        w.o_k != d.hidden || w.sgu_n != 2 * d.inter || w.sgu_k != d.hidden || w.sdn_n != d.hidden ||
        w.sdn_k != d.inter || w.egu_n != 2 * d.inter || w.egu_k != d.hidden || w.edn_n != d.hidden ||
        w.edn_k != d.inter) {
        err = "hunyuanimage3_layer_run: the layer's shapes are not the model's (" + std::to_string(d.heads) + " heads of " +
              std::to_string(d.head_dim) + ", hidden " + std::to_string(d.hidden) + ", intermediate " +
              std::to_string(d.inter) + ")";
        return false;
    }
    const int64_t rows = qi_attn_rows(T);
    if (sc.rows < rows) {
        err = "hunyuanimage3_layer_run: the scratch is not padded to qi_attn_rows(" + std::to_string(T) + ")";
        return false;
    }
    try {
        // --- attention: the norm, the save's rotation, qkv, the rope and the qk-norm, the block's mask, o
        kernels::rms_norm_rows(x, w.n_in, 1, sc.xn32, nullptr, (int) (d.hidden), (int) (T), d.eps, stream);
        hunyuanimage3_rotate16_from32(sc.xn32, T, d.hidden, d.rotate, sc.xr16, stream);
        gemm.f16(sc.xr16, w.qkv, sc.qkv32, T, w.qkv_n, w.qkv_k);
        hunyuanimage3_rope_angles(pos_y, pos_x, T, d.head_dim, d.theta, sc.ang, stream);
        hunyuanimage3_qk_rope_norm16(sc.qkv32, w.n_q, w.n_k, sc.ang, T, d.heads, d.kv_heads, d.head_dim, d.eps, sc.q16, sc.k16,
                           sc.v16, stream);
        // the rows T .. qi_attn_rows(T) of q, k and v are the attention's padding: hunyuanimage3_scratch_alloc zeroed them,
        // and a sequence changes T only by going through a new scratch
        qi_attention(sc.q16, sc.k16, sc.v16, T, d.heads, d.head_dim, T - n_img, sc.a16, stream,
                     key_lim, 0, true);
        hunyuanimage3_rotate16(sc.a16, T, d.hidden, d.rotate, sc.ar16, stream);
        gemm.f16(sc.ar16, w.o, sc.o32, T, w.o_n, w.o_k);
        if (out != x) up(out, x, (size_t) T * d.hidden * sizeof(float));
        hunyuanimage3_add(out, sc.o32, T * d.hidden, stream);

        // --- the MLP: the shared expert, then the routed ones on the router's FP32 logits
        kernels::rms_norm_rows(out, w.n_post, 1, sc.xn232, nullptr, (int) (d.hidden), (int) (T), d.eps, stream);
        hunyuanimage3_rotate16_from32(sc.xn232, T, d.hidden, d.rotate, sc.xr216, stream);
        gemm.f16(sc.xr216, w.sgu, sc.gu32, T, w.sgu_n, w.sgu_k);
        hunyuanimage3_swiglu16_rotated(sc.gu32, T, d.inter, sc.act16, stream);
        gemm.f16(sc.act16, w.sdn, sc.gu32, T, w.sdn_n, w.sdn_k);   // gu32 reused: the shared expert's output
        // The shared expert is part of every token's output, not one of the routed ones: it goes into the residual
        // here, before the routed experts' scatter adds to the same rows (`moe32` held it here until 2026-10-24 and
        // nothing added it - every layer lost it, which is the uniform, form-independent 7% the walk showed).
        hunyuanimage3_add(out, sc.gu32, T * d.hidden, stream);
        // The router on the same FP32 norm (the gate runs float32 in the reference, and which experts a token
        // takes is a discrete choice no later rounding can undo).
        kernels::f32_gemm_rows(sc.xn232, d.hidden, w.wg, sc.logits, d.experts, d.hidden, d.experts, T, stream);
        // softmax, the top `topk`, renormalized over them (norm_topk_prob), and the rows grouped by expert, on the
        // device: only the experts' bounds come back, for the products' sizes
        kernels::router_top10(sc.logits, (int) T, (int) d.experts, (int) d.topk, sc.d_idx, sc.d_tw, stream);
        hunyuanimage3_route_lists(sc.d_idx, sc.d_tw, T, d.topk, d.experts, sc.d_bounds, sc.d_rows, sc.d_w, sc.d_cursor,
                                  stream);
        gpu::device_sync();
        std::vector<int32_t> begin((size_t) d.experts + 1);
        down(begin.data(), sc.d_bounds, begin.size() * sizeof(int32_t));

        for (int64_t e = 0; e < d.experts; ++e) {
            const int64_t cnt = begin[(size_t) e + 1] - begin[(size_t) e];
            if (cnt == 0) continue;
            if (!experts.fill(e, sc.egu16, sc.edn16, err)) return false;
            prefill::gather_rows16(sc.xr216, sc.d_rows + begin[(size_t) e], sc.gx16, cnt, d.hidden, stream);
            gemm.f16(sc.gx16, sc.egu16, sc.eg32, cnt, w.egu_n, w.egu_k);
            hunyuanimage3_swiglu16_rotated(sc.eg32, cnt, d.inter, sc.gact16, stream);
            gemm.f16(sc.gact16, sc.edn16, sc.ed32, cnt, w.edn_n, w.edn_k);
            // the expert's rows back into their tokens by their weights: the LLMs' scatter of a group's rows
            // (each token once an expert, so no two work-items add to one value)
            prefill::moe_scatter_add(sc.ed32, sc.d_rows + begin[(size_t) e], nullptr, nullptr, sc.d_w + begin[(size_t) e],
                                     cnt, out, stream, d.hidden);
        }
        gpu::device_sync();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// --- a bank that sits in the device's region: the file's bytes, dequantized one expert at a time

bool HunyuanImage3Dit::RegionExperts::fill(int64_t expert, uint16_t* gate_up16, uint16_t* down16,
                                           std::string& err) {
    const Bank& b = *bank;
    const uint8_t* egu_w = region + dit->bank_offset(layer, b.spans[Bank::kEguW]);
    const uint8_t* edn_w = region + dit->bank_offset(layer, b.spans[Bank::kEdnW]);
    if (b.egu.quant == HunyuanImage3Quant::w4a8) {
        const uint8_t* egu_sr = region + dit->bank_offset(layer, b.spans[Bank::kEguSrel]);
        const float* egu_sc =
            reinterpret_cast<const float*>(region + dit->bank_offset(layer, b.spans[Bank::kEguSchan]));
        const float* egu_cb =
            reinterpret_cast<const float*>(region + dit->bank_offset(layer, b.spans[Bank::kEguCb]));
        const uint8_t* edn_sr = region + dit->bank_offset(layer, b.spans[Bank::kEdnSrel]);
        const float* edn_sc =
            reinterpret_cast<const float*>(region + dit->bank_offset(layer, b.spans[Bank::kEdnSchan]));
        const float* edn_cb =
            reinterpret_cast<const float*>(region + dit->bank_offset(layer, b.spans[Bank::kEdnCb]));
        const int64_t cb_e = b.egu.codebook_rows > 1 ? 16 : 0;   // a bank holds one table an expert
        const int64_t cb_d = b.edn.codebook_rows > 1 ? 16 : 0;
        hunyuanimage3_w4a8_dequant16(reinterpret_cast<const int8_t*>(egu_w) + expert * b.egu.n * (b.egu.k / 2),
                           egu_sr + expert * b.egu.n * (b.egu.k / b.egu.group), egu_sc + expert * b.egu.n,
                           egu_cb + expert * cb_e, 0, b.egu.n, b.egu.n, b.egu.k, b.egu.group, gate_up16, s);
        hunyuanimage3_w4a8_dequant16(reinterpret_cast<const int8_t*>(edn_w) + expert * b.edn.n * (b.edn.k / 2),
                           edn_sr + expert * b.edn.n * (b.edn.k / b.edn.group), edn_sc + expert * b.edn.n,
                           edn_cb + expert * cb_d, 0, b.edn.n, b.edn.n, b.edn.k, b.edn.group, down16, s);
        return true;
    }
    if (b.egu.quant == HunyuanImage3Quant::int8_rotated) {
        const float* egu_rs =
            reinterpret_cast<const float*>(region + dit->bank_offset(layer, b.spans[Bank::kEguSchan]));
        const float* edn_rs =
            reinterpret_cast<const float*>(region + dit->bank_offset(layer, b.spans[Bank::kEdnSchan]));
        hunyuanimage3_int8_dequant16(reinterpret_cast<const int8_t*>(egu_w) + expert * b.egu.n * b.egu.k,
                           egu_rs + expert * b.egu.n, 0, b.egu.n, b.egu.n, b.egu.k, gate_up16, s);
        hunyuanimage3_int8_dequant16(reinterpret_cast<const int8_t*>(edn_w) + expert * b.edn.n * b.edn.k,
                           edn_rs + expert * b.edn.n, 0, b.edn.n, b.edn.n, b.edn.k, down16, s);
        return true;
    }
    err = "HunyuanImage 3.0 (GPU DiT): the bank is neither w4a8 nor int8_convrot";
    return false;
}

// --- the DiT

HunyuanImage3Dit::HunyuanImage3Dit() = default;

HunyuanImage3Dit::~HunyuanImage3Dit() {
    drop_sequence();
    for (size_t i = owned_.size(); i-- > 0;) gpu::free(owned_[i]);
    owned_.clear();
    dense_.clear();
}

void HunyuanImage3Dit::drop_sequence() {
    hunyuanimage3_scratch_free(sc_);
    for (void* p : seq_owned_)
        if (p != nullptr) gpu::free(p);
    seq_owned_.clear();
    h_ = h2_ = nullptr;
    d_pos_y_ = d_pos_x_ = d_key_lim_ = nullptr;
    has_key_lim_ = false;
    T_ = n_img_ = 0;
}

void HunyuanImage3Dit::set_bank_budget(uint64_t bytes) {
    bank_budget_ = bytes;
    budget_set_ = true;
    if (!dry_) banks_.configure(bytes, s_);
}

uint64_t HunyuanImage3Dit::held() const {
    uint64_t b = dense_bytes_ + banks_.vram();
    if (T_ > 0) b += activation_bytes(T_);
    return b;
}

uint64_t HunyuanImage3Dit::bank_offset(int layer, int span) const {
    if (span < 0) throw std::runtime_error("HunyuanImage 3.0 (GPU DiT): a bank's piece this form does not have");
    return banks_.offset(layer, span);
}

bool HunyuanImage3Dit::add_bank(int64_t l, Bank& b, std::vector<LayerStream::Span>& spans, std::string& err) {
    const std::string p = "model.layers." + std::to_string(l) + ".mlp.";
    if (!f_->matrix(p + "experts_gate_up_proj.weight", b.egu, err) ||
        !f_->matrix(p + "experts_down_proj.weight", b.edn, err))
        return false;
    if (b.egu.experts != dims_.experts || b.edn.experts != dims_.experts) {
        err = "model.layers." + std::to_string(l) + ": the bank holds " + std::to_string(b.egu.experts) +
              " experts, not " + std::to_string(dims_.experts);
        return false;
    }
    const std::string gu = p + "experts_gate_up_proj.weight", dn = p + "experts_down_proj.weight";
    auto span = [&](const std::string& tensor, int role) -> bool {
        const uint8_t* host = nullptr;
        uint64_t bytes = 0;
        if (!f_->raw(tensor, host, bytes, err)) return false;
        spans.push_back({host, bytes, 0});
        b.spans[role] = (int) spans.size() - 1;
        return true;
    };
    // the bank's pieces in one region: the codes, then the group scales and the codebook (w4a8) or a row's scale
    // (int8_convrot), for the gate_up matrix and then the down one
    if (!span(gu, Bank::kEguW)) return false;
    if (b.egu.quant == HunyuanImage3Quant::w4a8) {
        if (!span(gu + "_s_rel", Bank::kEguSrel) || !span(gu + "_s_channel", Bank::kEguSchan) ||
            !span(gu + "_codebook", Bank::kEguCb))
            return false;
    } else if (!span(gu + "_scale", Bank::kEguSchan)) {
        return false;
    }
    if (!span(dn, Bank::kEdnW)) return false;
    if (b.edn.quant == HunyuanImage3Quant::w4a8) {
        if (!span(dn + "_s_rel", Bank::kEdnSrel) || !span(dn + "_s_channel", Bank::kEdnSchan) ||
            !span(dn + "_codebook", Bank::kEdnCb))
            return false;
    } else if (!span(dn + "_scale", Bank::kEdnSchan)) {
        return false;
    }
    return true;
}

bool HunyuanImage3Dit::upload_dense(int64_t l, Dense& d, std::string& err) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    const char* names[] = {"self_attn.qkv_proj.weight", "self_attn.o_proj.weight",
                           "mlp.shared_mlp.gate_and_up_proj.weight", "mlp.shared_mlp.down_proj.weight"};
    HunyuanImage3Matrix m[4];
    uint64_t total = 0;
    for (int i = 0; i < 4; ++i) {
        if (!f_->matrix(p + names[i], m[i], err)) return false;
        if (m[i].experts != 1) {
            err = std::string(p) + names[i] + ": not one matrix";
            return false;
        }
        if (m[i].rotate != dims_.rotate) {
            err = std::string(p) + names[i] + ": rotated in groups of " + std::to_string(m[i].rotate) +
                  ", the walk rotates activations in " + std::to_string(dims_.rotate);
            return false;
        }
        total += m[i].weight_bytes();
        total += m[i].quant == HunyuanImage3Quant::w4a8
                     ? (uint64_t) m[i].n * (m[i].k / m[i].group) + (uint64_t) m[i].n * 4 + 64
                     : (uint64_t) m[i].n * 4;
    }
    // the file's bytes of the layer's four dense products, gathered once, up at once, dequantized on the device
    // into the FP16 buffers that stay
    std::vector<uint8_t> host((size_t) total);
    uint64_t off[4] = {0, 0, 0, 0}, sr_off[4] = {0, 0, 0, 0}, sc_off[4] = {0, 0, 0, 0}, cb_off[4] = {0, 0, 0, 0};
    uint64_t at = 0;
    auto put = [&](const std::string& tensor, uint64_t& where) -> bool {
        const uint8_t* src = nullptr;
        uint64_t bytes = 0;
        if (!f_->raw(tensor, src, bytes, err)) return false;
        if (at + bytes > total) {
            err = tensor + ": past the space planned for the layer's dense products";
            return false;
        }
        where = at;
        std::memcpy(host.data() + at, src, bytes);
        at += bytes;
        return true;
    };
    for (int i = 0; i < 4; ++i) {
        const std::string n = p + names[i];
        if (!put(n, off[i])) return false;
        if (m[i].quant == HunyuanImage3Quant::w4a8) {
            if (!put(n + "_s_rel", sr_off[i]) || !put(n + "_s_channel", sc_off[i]) || !put(n + "_codebook", cb_off[i]))
                return false;
        } else if (!put(n + "_scale", sc_off[i])) {
            return false;
        }
    }
    void* tmp = nullptr;
    if (!gpu::alloc_device(&tmp, (size_t) at)) {
        err = oom("the dense upload");
        return false;
    }
    const uint8_t* base = reinterpret_cast<const uint8_t*>(tmp);
    try {
        up(tmp, host.data(), (size_t) at);
        uint16_t* dst[4] = {nullptr, nullptr, nullptr, nullptr};
        if (d.qkv == nullptr) d.qkv = take_as<uint16_t>(owned_, m[0].n * m[0].k, "qkv");
        if (d.o == nullptr) d.o = take_as<uint16_t>(owned_, m[1].n * m[1].k, "o");
        if (d.sgu == nullptr) d.sgu = take_as<uint16_t>(owned_, m[2].n * m[2].k, "sgu");
        if (d.sdn == nullptr) d.sdn = take_as<uint16_t>(owned_, m[3].n * m[3].k, "sdn");
        dst[0] = d.qkv, dst[1] = d.o, dst[2] = d.sgu, dst[3] = d.sdn;
        d.qkv_n = m[0].n, d.qkv_k = m[0].k, d.o_n = m[1].n, d.o_k = m[1].k;
        d.sgu_n = m[2].n, d.sgu_k = m[2].k, d.sdn_n = m[3].n, d.sdn_k = m[3].k;
        for (int i = 0; i < 4; ++i) {
            if (m[i].quant == HunyuanImage3Quant::w4a8)
                hunyuanimage3_w4a8_dequant16(reinterpret_cast<const int8_t*>(base + off[i]), base + sr_off[i],
                                   reinterpret_cast<const float*>(base + sc_off[i]),
                                   reinterpret_cast<const float*>(base + cb_off[i]), 0, m[i].n, m[i].n, m[i].k,
                                   m[i].group, dst[i], s_);
            else
                hunyuanimage3_int8_dequant16(reinterpret_cast<const int8_t*>(base + off[i]),
                                   reinterpret_cast<const float*>(base + sc_off[i]), 0, m[i].n, m[i].n, m[i].k,
                                   dst[i], s_);
        }
        gpu::device_sync();
    } catch (...) {
        gpu::free(tmp);
        throw;
    }
    gpu::free(tmp);

    // the norms and the router: BF16 in the file, FP32 on the device (the router's logits are summed in double)
    try {
        auto f32up = [&](const std::string& name, float*& dst, int64_t n, const char* what) {
            std::vector<float> h((size_t) n);
            if (!f_->bf16(p + name, h.data())) throw std::runtime_error(f_->path() + ": " + name);
            if (dst == nullptr) dst = take_as<float>(owned_, n, what);
            up(dst, h.data(), (size_t) n * sizeof(float));
        };
        f32up("input_layernorm.weight", d.n_in, dims_.hidden, "input_layernorm");
        f32up("post_attention_layernorm.weight", d.n_post, dims_.hidden, "post_attention_layernorm");
        f32up("self_attn.query_layernorm.weight", d.n_q, dims_.head_dim, "query_layernorm");
        f32up("self_attn.key_layernorm.weight", d.n_k, dims_.head_dim, "key_layernorm");
        f32up("mlp.gate.wg.weight", d.wg, dims_.experts * dims_.hidden, "gate.wg");
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool HunyuanImage3Dit::open(const std::string& path, void* stream, std::string& err, int64_t max_layers, bool dry) {
    s_ = stream;
    dry_ = dry;
    f_ = std::make_unique<HunyuanImage3File>();
    if (!f_->open(path, err)) return false;
    if (!f_->quantized()) {
        err = path + ": not a HunyuanImage 3.0 quantized file";
        return false;
    }
    const int64_t all = f_->layer_count();
    if (all == 0) {
        err = path + ": no model.layers.N tensors";
        return false;
    }
    n_layers_ = max_layers > 0 ? std::min(all, max_layers) : all;

    // the model's shapes, as its first layer states them
    HunyuanImage3Matrix qkv, sgu, sdn, egu, edn;
    if (!f_->matrix("model.layers.0.self_attn.qkv_proj.weight", qkv, err) ||
        !f_->matrix("model.layers.0.mlp.shared_mlp.gate_and_up_proj.weight", sgu, err) ||
        !f_->matrix("model.layers.0.mlp.shared_mlp.down_proj.weight", sdn, err) ||
        !f_->matrix("model.layers.0.mlp.experts_gate_up_proj.weight", egu, err) ||
        !f_->matrix("model.layers.0.mlp.experts_down_proj.weight", edn, err))
        return false;
    dims_.hidden = qkv.k;
    dims_.inter = sgu.n / 2;
    dims_.experts = egu.experts;
    dims_.rotate = qkv.rotate;
    if (qkv.n != kQkvRows(dims_.heads, dims_.kv_heads, dims_.head_dim) || sgu.k != dims_.hidden ||
        sdn.n != dims_.hidden || sdn.k != dims_.inter || egu.n != 2 * dims_.inter || egu.k != dims_.hidden ||
        edn.n != dims_.hidden || edn.k != dims_.inter) {
        err = path + ": its shapes are not the model this walk is written for (qkv " + std::to_string(qkv.n) +
              ", gate_up " + std::to_string(sgu.n) + ", hidden " + std::to_string(dims_.hidden) +
              "); heads/kv_heads/head_dim are the config's";
        return false;
    }
    if (dims_.rotate != 256) {
        err = path + ": its weights are rotated in groups of " + std::to_string(dims_.rotate) +
              ", not the 256 this walk undoes";
        return false;
    }
    dense_.resize((size_t) n_layers_);
    bank_refs_.resize((size_t) n_layers_);
    std::vector<LayerStream::Span> spans;
    try {
        if (!dry && !gemm_.init(s_, 2 * dims_.inter * dims_.hidden, err)) return false;
        for (int64_t l = 0; l < n_layers_; ++l) {
            // the dry read takes the shapes out of the header and nothing else: the same numbers a budget is
            // planned with, with no byte of the model uploaded (`need` and a small card's start run this way)
            if (dry ? !dense_shapes(l, dense_[(size_t) l], err) : !upload_dense(l, dense_[(size_t) l], err))
                return false;
            Bank b;
            spans.clear();
            if (!add_bank(l, b, spans, err)) return false;
            banks_.add(spans);
            bank_refs_[(size_t) l] = b;
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    dense_bytes_ = 0;
    for (const Dense& d : dense_)
        dense_bytes_ += ((uint64_t) d.qkv_n * d.qkv_k + (uint64_t) d.o_n * d.o_k + (uint64_t) d.sgu_n * d.sgu_k +
                         (uint64_t) d.sdn_n * d.sdn_k) * 2 +
                        (uint64_t) (2 * dims_.hidden + 2 * dims_.head_dim + dims_.experts * dims_.hidden) * 4;
    if (dry) return true;   // the sizes only: no region is allocated, so nothing can run
    // nothing resident until a budget says what the card can hold: the smallest card streams every bank
    banks_.configure(budget_set_ ? bank_budget_ : banks_.min_budget(), s_);
    return true;
}

// The four dense products' shapes, which is all a budget is planned with: the same tensors `upload_dense` turns
// into FP16 on the card, read out of the header here.
bool HunyuanImage3Dit::dense_shapes(int64_t l, Dense& d, std::string& err) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    const char* names[] = {"self_attn.qkv_proj.weight", "self_attn.o_proj.weight",
                           "mlp.shared_mlp.gate_and_up_proj.weight", "mlp.shared_mlp.down_proj.weight"};
    int64_t Dense::*field[4][2] = {
        {&Dense::qkv_n, &Dense::qkv_k}, {&Dense::o_n, &Dense::o_k},
        {&Dense::sgu_n, &Dense::sgu_k}, {&Dense::sdn_n, &Dense::sdn_k}};
    for (int i = 0; i < 4; ++i) {
        HunyuanImage3Matrix m;
        if (!f_->matrix(p + names[i], m, err)) return false;
        d.*(field[i][0]) = m.n;
        d.*(field[i][1]) = m.k;
    }
    return true;
}

bool HunyuanImage3Dit::set_sequence(int64_t T, int64_t n_img, const int32_t* pos_y, const int32_t* pos_x,
                                    std::string& err, const int32_t* key_lim) {
    if (dry_) {
        err = "HunyuanImage 3.0 (GPU DiT): a dry part cannot run";
        return false;
    }
    if (T <= 0 || n_img < 0 || n_img > T) {
        err = "HunyuanImage 3.0 (GPU DiT): no tokens, or the block is larger than the sequence";
        return false;
    }
    if (T != T_) {
        drop_sequence();
        if (!hunyuanimage3_scratch_alloc(T, dims_, sc_, s_, err)) return false;
        try {
            h_ = take_as<float>(seq_owned_, T * dims_.hidden, "hidden");
            h2_ = take_as<float>(seq_owned_, T * dims_.hidden, "hidden 2");
            d_pos_y_ = take_as<int32_t>(seq_owned_, T, "positions y");
            d_pos_x_ = take_as<int32_t>(seq_owned_, T, "positions x");
            d_key_lim_ = take_as<int32_t>(seq_owned_, T, "key limits");
        } catch (const std::exception& e) {
            err = e.what();
            drop_sequence();
            return false;
        }
        T_ = T;
    }
    try {
        up(d_pos_y_, pos_y, (size_t) T * sizeof(int32_t));
        up(d_pos_x_, pos_x, (size_t) T * sizeof(int32_t));
        if (key_lim != nullptr) up(d_key_lim_, key_lim, (size_t) T * sizeof(int32_t));
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    has_key_lim_ = key_lim != nullptr;
    n_img_ = n_img;
    return true;
}

bool HunyuanImage3Dit::forward(const float* x, float* out, int64_t n,
                               const std::function<void(int64_t, const float*)>& layer_done, std::string& err) {
    if (dry_ || T_ <= 0) {
        err = "HunyuanImage 3.0 (GPU DiT): no sequence";
        return false;
    }
    if (n <= 0 || n > n_layers_) {
        err = "HunyuanImage 3.0 (GPU DiT): asked for layers 0 .. " + std::to_string(n) + " of " +
              std::to_string(n_layers_);
        return false;
    }
    float *a = h_, *b = h2_;
    try {
        up(a, x, (size_t) T_ * dims_.hidden * sizeof(float));
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    for (int64_t l = 0; l < n; ++l) {
        if (!run_layer(l, a, b, err)) return false;
        std::swap(a, b);
        if (layer_done) layer_done(l, a);
    }
    try {
        if (out != a) up(out, a, (size_t) T_ * dims_.hidden * sizeof(float));
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool HunyuanImage3Dit::layer(int64_t l, const float* x, float* out, std::string& err) {
    if (dry_ || T_ <= 0) {
        err = "HunyuanImage 3.0 (GPU DiT): no sequence";
        return false;
    }
    if (l < 0 || l >= n_layers_) {
        err = "HunyuanImage 3.0 (GPU DiT): layer " + std::to_string(l) + " of " + std::to_string(n_layers_);
        return false;
    }
    try {
        up(h_, x, (size_t) T_ * dims_.hidden * sizeof(float));
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    if (!run_layer(l, h_, h2_, err)) return false;
    try {
        if (out != h2_) up(out, h2_, (size_t) T_ * dims_.hidden * sizeof(float));
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool HunyuanImage3Dit::run_layer(int64_t l, const float* a, float* b, std::string& err) {
    const uint8_t* region = nullptr;
    try {
        region = banks_.begin((int) l);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    RegionExperts ex;
    ex.dit = this;
    ex.layer = (int) l;
    ex.bank = &bank_refs_[(size_t) l];
    ex.region = region;
    ex.s = s_;
    const Dense& d = dense_[(size_t) l];
    HunyuanImage3LayerWeights lw;
    lw.qkv = d.qkv, lw.o = d.o, lw.sgu = d.sgu, lw.sdn = d.sdn;
    lw.n_in = d.n_in, lw.n_post = d.n_post, lw.n_q = d.n_q, lw.n_k = d.n_k, lw.wg = d.wg;
    lw.qkv_n = d.qkv_n, lw.qkv_k = d.qkv_k, lw.o_n = d.o_n, lw.o_k = d.o_k;
    lw.sgu_n = d.sgu_n, lw.sgu_k = d.sgu_k, lw.sdn_n = d.sdn_n, lw.sdn_k = d.sdn_k;
    const Bank& bk = bank_refs_[(size_t) l];
    lw.egu_n = bk.egu.n, lw.egu_k = bk.egu.k, lw.edn_n = bk.edn.n, lw.edn_k = bk.edn.k;
    const bool ok = hunyuanimage3_layer_run(lw, ex, sc_, a, b, T_, d_pos_y_, d_pos_x_, n_img_, dims_, gemm_, s_, err,
                                            has_key_lim_ ? d_key_lim_ : nullptr);
    banks_.end((int) l);   // the slot is free again once the compute stream is past this layer
    return ok;
}

}  // namespace strata::image
