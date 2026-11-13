// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/plain_model.cpp - an MoE model with one residual stream, run from its GGUF (plain_model.hpp).
//
// The block follows llama.cpp's qwen35moe graph (src/models/qwen35moe.cpp at de7fa0a3) step for step:
//
//   GDN:        qkv = attn_qkv(h), z = attn_gate(h), beta = sigmoid(ssm_beta(h)), alpha = ssm_alpha(h)
//               conv over [history | qkv], SiLU on all of it; q, k l2-normalized (v head j reads q/k head j % h_k)
//               gate = softplus(alpha + dt) * ssm_a; the delta rule; y = rms_norm(o) * ssm_norm * silu(z)
//               out = ssm_out(y)
//   attention:  [q | gate] per head = attn_q(h), k = attn_k(h), v = attn_v(h); q, k rms-normed per head, then
//               rotated in their first 64 dims (text positions: every IMRoPE section at the same position)
//               softmax(q k / sqrt(d)) v over every cell, times sigmoid(gate); out = attn_output(.)
//   MoE:        softmax over the experts, top-k, the k weights renormalized; sum of w_j expert_j(h)
//               + sigmoid(ffn_gate_inp_shexp . h) * shared(h)
//
// and qwen3moe's (llm-jp-4.1, src/models/qwen3moe.cpp): every block a full attention without the gate, rotated over the
// whole head; the MoE without the shared expert; the second norm named ffn_norm.  qwen35's (Qwen3.5-9B,
// src/models/qwen35.cpp at de7fa0a3) is qwen35moe's with the MoE replaced by a dense SwiGLU, down(silu(gate(h)) * up(h)),
// in every block and in the MTP layer: no experts, so none of the tiers, the CPU pool or the window protocol.
#include "strata/core/kv_elastic.hpp"
#include "strata/core/kv_grow_hold.hpp"
#include "strata/core/plain_model.hpp"
#include "strata/core/constraint_upload.hpp"

#include "strata/core/plain_cpu.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/device.hpp"
#include "strata/core/link_probe.hpp"
#include "strata/core/split_predict.hpp"
#include "strata/core/expert_tier.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/prompt_attn.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/sampler_host.hpp"
#include "strata/kernels/draft_head.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/platform/memory.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <numeric>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits>
#include <thread>
#include <utility>

namespace strata::core {
namespace {

namespace k = strata::kernels;

// The plain engine's floor for the room no plan counts (core::vram_room): under STRATA_VRAM_LIMIT_MIB=4096 on
// the RX 9060 XT the peak stayed below the card with 128 MiB kept free - llm-jp Q4_K_M, Ornith Q4_K_M and
// GSQ-RCO 3bit, 2026-10-12 - where the 768 MiB of before left 700 MiB of the card unused.
constexpr uint64_t kTierRoomLeast = 192ull << 20;

constexpr int kQ6K = 14;
constexpr int kQ80 = 8;
constexpr int kBF16 = 30;   // a BF16 file's projections: bf16_gemv_fp32_mmvf_multi on the FP32 rows
constexpr int kF32 = 0;
constexpr int kF16 = 1;

bool mapped(size_t bytes, void** h, void** d) {
    if (!strata::gpu::alloc_host(h, bytes)) return false;
    std::memset(*h, 0, bytes);
    return strata::gpu::device_pointer(d, *h);
}

}  // namespace

PlainModel::PlainModel() = default;

PlainModel::~PlainModel() {
    if (cpu_ != nullptr) {   // --device cpu: no GPU was opened, nothing of its to give back
        if (route_dump_ != nullptr) std::fclose(route_dump_);
        return;
    }
    next_.clear();   // the later stages read the root's experts and pool
    const OnDevice on(dev_);
    if (route_dump_ != nullptr && root_ == this) std::fclose(route_dump_);
    if (swap_cs_ != nullptr) strata::gpu::stream_sync(swap_cs_);
    if (cs_ != nullptr) strata::gpu::stream_sync(cs_);
    strata::gpu::free(tap_dev_);
    for (auto* e : prompt_share_events_) if (e) strata::gpu::event_destroy(e);
    prompt_gemm_.reset();   // the products' state before the stream
    strata::gpu::free(prompt_host_);
    if (opt_.encoder)   // an encoder's prompt buffers are its own (encoder_buffers)
        for (uint8_t* p : prompt_dev_.at) strata::gpu::free(p);
    for (auto* g : exec_) strata::gpu::graph_destroy(g);
    for (auto& v : segs_)
        for (auto* g : v) strata::gpu::graph_destroy(g);
    for (auto* g : mexec_) strata::gpu::graph_destroy(g);
    for (auto* m : {&rexec_, &mrexec_})
        for (auto* g : *m) strata::gpu::graph_destroy(g);
    for (auto& v : msegs_)
        for (auto* g : v) strata::gpu::graph_destroy(g);
    for (auto* m : {&bexec_, &mbexec_})
        for (auto& [key, g] : *m) strata::gpu::graph_destroy(g);
    for (auto* m : {&bsegs_, &mbsegs_})
        for (auto& [key, v] : *m)
            for (auto* g : v) strata::gpu::graph_destroy(g);
    for (auto& h : helpers_) h.reset();   // (their streams before the memory below)
    for (void* p : allocs_) strata::gpu::free(p);
    for (void* p : {(void*) h_emb_, (void*) h_step_, (void*) h_pos_, (void*) h_out_, (void*) h_prob_, (void*) fill_tmp_,
                    (void*) hand_host_, (void*) h_rroute_})
        strata::gpu::free(p);
    if (swap_ev_ != nullptr) strata::gpu::event_destroy(swap_ev_);
    if (swap_cs_ != nullptr) strata::gpu::stream_destroy(swap_cs_);
    if (cs_ != nullptr) strata::gpu::stream_destroy(cs_);
}

bool PlainModel::upload(const GgufModel& m, const std::string& name, W& w, std::string& err, bool f32, bool widen) {
    size_t at = 0;
    const TensorInfo* t = m.find(name, &at);
    if (t == nullptr || t->shape.size() < 2 || !m.in_bounds(*t, at)) {
        err = "plain model: " + name + " is missing, not a matrix, or truncated";
        return false;
    }
    w.type = (int) t->type;
    w.n_in = (int) t->shape[0];
    uint64_t rows = 1;
    for (size_t i = 1; i < t->shape.size(); ++i) rows *= t->shape[i];
    w.n_out = (int) rows;
    const void* src = m.shard(at).tensor_data(*t);
    uint64_t bytes = tensor_payload_bytes(*t);
    std::vector<float> wide;
    if ((f32 && w.type == kF16) || (widen && w.type != kBF16 && w.type != kF32)) {
        f32 = true;
        wide.resize((size_t) w.n_in * rows);
        if (!k::cpu::native_row_to_float(w.type, src, wide.data(), (int64_t) wide.size())) {
            err = "plain model: widening " + name + " from " + t->type_name() + " failed";
            return false;
        }
        src = wide.data();
        bytes = (uint64_t) wide.size() * 4;
        w.type = kF32;
    }
    if (!k::native_mmvq_supported(w.type) && !((w.type == kBF16 || (f32 && w.type == kF32)) && w.n_in % 2 == 0)) {
        err = "plain model: " + name + " is " + t->type_name() + ", which the native matrix-vector kernels do not take";
        return false;
    }
    if (stream_into_ != nullptr) {   // a streamed block's: left in the file, laid out as below when it is copied
        if (!wide.empty()) src = widened_.emplace_back(std::move(wide)).data();
        stream_into_->push_back({&w, {src, bytes}, 0});
        if (w.type == kQ6K && k::native_q6_k_rows_ok(w.n_in)) w.type = k::kNativeQ6KRows;
        else if (w.type == kQ80 && k::native_q8_0_rows_ok(w.n_in)) w.type = k::kNativeQ8Rows;
        w.data = nullptr;
        return true;
    }
    void* d = nullptr;
    if (!strata::gpu::alloc_device(&d, bytes)) {
        err = "plain model: no VRAM for " + name;
        return false;
    }
    allocs_.push_back(d);
    if (!strata::gpu::copy(d, src, bytes)) {
        err = "plain model: uploading " + name + " failed";
        return false;
    }
    if (w.type == kQ6K && k::native_q6_k_rows_ok(w.n_in)) {   // the same results, each row's loads aligned
        k::native_q6_k_to_rows(d, w.n_in, w.n_out, cs_);
        w.type = k::kNativeQ6KRows;
    } else if (w.type == kQ80 && k::native_q8_0_rows_ok(w.n_in)) {   // as Qwen3.8's dense weights (native_dense)
        k::native_q8_0_to_rows(d, w.n_in, w.n_out, cs_);
        w.type = k::kNativeQ8Rows;
    }
    w.data = d;
    weight_at_.push_back(d);
    weight_bytes_.push_back(bytes);
    if (k::native_mmvq_supported(w.type)) tune_wait_.push_back(&w);
    return true;
}

// each shape's decode layout on this GPU, before any graph captures it, its launches reading the weights uploaded so
// far: a span of their own (256 MiB at least) allocated and freed for each shape was kept by the B70's driver (its
// free memory 256 MiB short after the load, and a prompt's chunks short of room with it, 2026-11-07)
bool PlainModel::tune_uploaded(std::string& err) {
    try {
        for (const W* w : tune_wait_)
            k::native_mmvq_tune(w->type, w->data, w->n_in, w->n_out, cs_, (int) weight_at_.size(), weight_at_.data(),
                                weight_bytes_.data());
    } catch (const std::exception& e) {
        err = std::string("plain model: timing the decode products: ") + e.what();
        return false;
    }
    tune_wait_.clear();
    return true;
}

const char* PlainModel::head_name() const {
    size_t at = 0;
    return model_->find("output.weight", &at) != nullptr ? "output.weight" : "token_embd.weight";
}

bool PlainModel::upload_f32(const GgufModel& m, const std::string& name, uint64_t elements, const float*& out,
                            std::string& err) {
    size_t at = 0;
    const TensorInfo* t = m.find(name, &at);
    if (t == nullptr || (t->type != kF32 && t->type != kBF16) || t->elements() != elements || !m.in_bounds(*t, at)) {
        err = "plain model: " + name + " is missing, not F32 or BF16 or not " + std::to_string(elements) + " values";
        return false;
    }
    const void* src = m.shard(at).tensor_data(*t);
    std::vector<float> wide;
    if (t->type == kBF16) {
        wide.resize(elements);
        const auto* h = static_cast<const uint16_t*>(src);
        for (uint64_t i = 0; i < elements; ++i) {
            const uint32_t u = (uint32_t) h[i] << 16;
            std::memcpy(&wide[i], &u, 4);
        }
        src = wide.data();
    }
    void* d = nullptr;
    if (!strata::gpu::alloc_device(&d, elements * 4)) {
        err = "plain model: no VRAM for " + name;
        return false;
    }
    allocs_.push_back(d);
    if (!strata::gpu::copy(d, src, elements * 4)) {
        err = "plain model: uploading " + name + " failed";
        return false;
    }
    out = (const float*) d;
    return true;
}

PlainModel::W PlainModel::row_view(const W& all, int64_t r0, int64_t r1) {
    // the engine's row layouts (upload) rewrite each row in place, so a row keeps its GGUF type's size
    const int gguf = all.type == k::kNativeQ8Rows ? kQ80 : all.type == k::kNativeQ6KRows ? kQ6K : all.type;
    const size_t row = gguf == kBF16 ? (size_t) all.n_in * 2
                       : gguf == kF32 ? (size_t) all.n_in * 4
                                      : k::cpu::native_row_bytes(gguf, all.n_in);
    W v = all;
    v.data = static_cast<const uint8_t*>(all.data) + (size_t) r0 * row;
    v.n_out = (int) (r1 - r0);
    return v;
}

bool PlainModel::load_block(const GgufModel& m, int64_t l, Block& b, std::string& err) {
    const ModelSpec& s = spec_;
    const bool q35 = s.arch == ModelArch::Qwen35Moe || s.arch == ModelArch::Qwen35;
    const uint64_t N = (uint64_t) s.n_embd, E = (uint64_t) s.n_expert;
    const std::string p = "blk." + std::to_string(l) + ".";
    if (!upload_f32(m, p + "attn_norm.weight", N, b.attn_norm, err) ||
        !upload_f32(m, p + (q35 ? "post_attention_norm.weight" : "ffn_norm.weight"), N, b.post_norm, err))
        return false;
    if (plamo_) {   // llama.cpp's plamo3: the norms after the attention and the FFN (their names carry no ".weight")
        b.swa = s.layer_swa(l);
        if (!upload_f32(m, p + "post_attention_norm", N, b.attn_post, err) ||
            !upload_f32(m, p + "post_ffw_norm", N, b.ffn_post, err))
            return false;
    }
    if (b.recurrent) {
        const uint64_t C = (uint64_t) (2 * s.ssm_n_group * s.ssm_d_state + s.ssm_d_inner);
        if (!upload(m, p + "attn_qkv.weight", b.qkv, err) || !upload(m, p + "attn_gate.weight", b.z, err) ||
            !upload(m, p + "ssm_alpha.weight", b.alpha, err, true) ||
            !upload(m, p + "ssm_beta.weight", b.beta, err, true) ||
            (b.alpha.type != kBF16 && b.alpha.type != kF32 &&
             (!upload(m, p + "ssm_alpha.weight", b.alpha_f, err, true, true) ||
              !upload(m, p + "ssm_beta.weight", b.beta_f, err, true, true))) ||
            !upload(m, p + "ssm_out.weight", b.ssm_out, err) ||
            !upload_f32(m, p + "ssm_conv1d.weight", C * (uint64_t) s.ssm_d_conv, b.conv, err) ||
            !upload_f32(m, p + "ssm_dt.bias", (uint64_t) s.ssm_dt_rank, b.dt, err) ||
            !upload_f32(m, p + "ssm_a", (uint64_t) s.ssm_dt_rank, b.a, err) ||
            !upload_f32(m, p + "ssm_norm.weight", (uint64_t) s.ssm_d_state, b.ssm_norm, err))
            return false;
        if ((uint64_t) b.qkv.n_out != C || b.z.n_out != s.ssm_d_inner || b.alpha.n_out != s.ssm_dt_rank ||
            b.beta.n_out != s.ssm_dt_rank || b.ssm_out.n_in != s.ssm_d_inner) {
            err = "plain model: block " + std::to_string(l) + "'s GDN projections have other shapes";
            return false;
        }
    } else {
        const int64_t nkv = s.n_head_kv[(size_t) l], hd = s.key_length;
        if (plamo_) {   // one [q | k | v] matrix, read as its three row ranges
            if (!upload(m, p + "attn_qkv.weight", b.qkv_all, err)) return false;
            if (b.qkv_all.n_out != (s.n_head + 2 * nkv) * hd) {
                err = "plain model: block " + std::to_string(l) + "'s attn_qkv has another shape";
                return false;
            }
            b.q = row_view(b.qkv_all, 0, s.n_head * hd);
            b.k = row_view(b.qkv_all, s.n_head * hd, (s.n_head + nkv) * hd);
            b.v = row_view(b.qkv_all, (s.n_head + nkv) * hd, (s.n_head + 2 * nkv) * hd);
            for (const W* w : {&b.q, &b.k, &b.v})
                if (k::native_mmvq_supported(w->type)) tune_wait_.push_back(w);
        }
        if ((!plamo_ && (!upload(m, p + "attn_q.weight", b.q, err) || !upload(m, p + "attn_k.weight", b.k, err) ||
                         !upload(m, p + "attn_v.weight", b.v, err))) ||
            !upload(m, p + "attn_output.weight", b.o, err) ||
            !upload_f32(m, p + "attn_q_norm.weight", (uint64_t) hd, b.q_norm, err) ||
            !upload_f32(m, p + "attn_k_norm.weight", (uint64_t) hd, b.k_norm, err))
            return false;
        if (b.q.n_out != (has_gate_ ? 2 : 1) * s.n_head * hd || b.k.n_out != nkv * hd || b.v.n_out != nkv * hd ||
            b.o.n_in != s.n_head * hd || nkv <= 0 || s.n_head % nkv != 0 || (nkv_ != 0 && nkv != nkv_)) {
            err = "plain model: block " + std::to_string(l) + "'s attention projections have other shapes";
            return false;
        }
        nkv_ = nkv;
    }
    if (dense_) {   // qwen35's FFN, in the shared expert's fields (the same products, without the gate or the router)
        if (plamo_) {   // one [gate | up] matrix (ffn_up, 2 n_ff rows: llama.cpp's swiglu takes the first half as gate)
            if (!upload(m, p + "ffn_up.weight", b.gu_all, err) || !upload(m, p + "ffn_down.weight", b.sh_down, err))
                return false;
            if (b.gu_all.n_out != 2 * s.n_ff) {
                err = "plain model: block " + std::to_string(l) + "'s ffn_up is not 2 x feed_forward_length";
                return false;
            }
            b.sh_gate = row_view(b.gu_all, 0, s.n_ff);
            b.sh_up = row_view(b.gu_all, s.n_ff, 2 * s.n_ff);
            for (const W* w : {&b.sh_gate, &b.sh_up})
                if (k::native_mmvq_supported(w->type)) tune_wait_.push_back(w);
        } else if (!upload(m, p + "ffn_gate.weight", b.sh_gate, err) || !upload(m, p + "ffn_up.weight", b.sh_up, err) ||
                   !upload(m, p + "ffn_down.weight", b.sh_down, err)) {
            return false;
        }
        if (b.sh_gate.n_in != s.n_embd || b.sh_gate.n_out != s.n_ff || b.sh_up.n_in != s.n_embd ||
            b.sh_up.n_out != s.n_ff || b.sh_down.n_in != s.n_ff || b.sh_down.n_out != s.n_embd) {
            err = "plain model: block " + std::to_string(l) + "'s FFN has other shapes";
            return false;
        }
        return true;
    }
    if (!upload_f32(m, p + "ffn_gate_inp.weight", N * E, b.router, err)) return false;
    if (!has_shared_) return true;
    if (!upload_f32(m, p + "ffn_gate_inp_shexp.weight", N, b.sh_gate_inp, err) ||
        !upload(m, p + "ffn_gate_shexp.weight", b.sh_gate, err) ||
        !upload(m, p + "ffn_up_shexp.weight", b.sh_up, err) ||
        !upload(m, p + "ffn_down_shexp.weight", b.sh_down, err))
        return false;
    if (b.sh_gate.type != b.sh_up.type || b.sh_gate.n_out != s.n_ff_shexp || b.sh_down.n_in != s.n_ff_shexp) {
        err = "plain model: block " + std::to_string(l) + "'s shared expert has other shapes";
        return false;
    }
    return true;
}

std::vector<PlainModel::HostSpan> PlainModel::block_spans(int64_t l) const {
    std::vector<HostSpan> out;
    if (l >= 0 && l < (int64_t) streamed_.size())
        for (const Streamed& st : streamed_[(size_t) l]) out.push_back(st.span);
    return out;
}

void PlainModel::set_block_source(BlockSource* src, std::vector<std::vector<uint64_t>> offsets) {
    block_src_ = src;
    for (size_t l = 0; l < streamed_.size() && l < offsets.size(); ++l)
        for (size_t i = 0; i < streamed_[l].size() && i < offsets[l].size(); ++i) streamed_[l][i].offset = offsets[l][i];
}

void PlainModel::prepare_block(int64_t l, uint8_t* region) {
    for (const Streamed& st : streamed_[(size_t) l]) {
        void* d = region + st.offset;
        if (st.w->type == k::kNativeQ6KRows) k::native_q6_k_to_rows(d, st.w->n_in, st.w->n_out, cs_);
        else if (st.w->type == k::kNativeQ8Rows) k::native_q8_0_to_rows(d, st.w->n_in, st.w->n_out, cs_);
    }
}

void PlainModel::bind_block(int64_t l) {
    if (block_src_ == nullptr || l < 0 || l >= (int64_t) streamed_.size()) return;
    const uint8_t* region = block_src_->begin((int) l);
    for (Streamed& st : streamed_[(size_t) l]) st.w->data = region + st.offset;
}

void PlainModel::unbind_block(int64_t l) {
    if (block_src_ != nullptr && l >= 0 && l < (int64_t) streamed_.size()) block_src_->end((int) l);
}

std::vector<PlainModel*> PlainModel::chain() {
    std::vector<PlainModel*> c{this};
    for (auto& n : next_) c.push_back(n.get());
    return c;
}

bool PlainModel::load(const PlainOptions& opt, std::string& err) {
    opt_ = opt;
    if (!opt_.dump_routing.empty()) {
        route_dump_ = std::fopen(opt_.dump_routing.c_str(), "wb");
        if (route_dump_ == nullptr) {
            err = "plain model: cannot write " + opt_.dump_routing;
            return false;
        }
    }
    if (opt_.max_t < 1 || opt_.max_t > k::cpu::MAXT || opt_.max_cells < 16) {
        err = "plain model: the window or the context is out of range";
        return false;
    }
    if (opt_.cpu) return cpu_load(err);
    // the processor's own graphics through Level Zero: the first Level Zero adapter only (plain_generate.cpp's
    // linkless_gpu_adapter chooses it before the devices are listed; a user's SYCL_UR_USE_LEVEL_ZERO_V2=1 is refused)
    if (const sycl::device& dev = Runtime::get().device();
        integrated_gpu(dev) && dev.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        const char* v2 = std::getenv("SYCL_UR_USE_LEVEL_ZERO_V2");
        if (v2 == nullptr || std::strcmp(v2, "0") != 0) {
            err = "plain model: on the processor's own graphics this model needs SYCL_UR_USE_LEVEL_ZERO_V2=0 (the "
                  "driver aborts with the other Level Zero adapter)";
            return false;
        }
    }
    // On Level Zero (xe) the device's allocations need free RAM beside them, and the file cache does not count as free
    // there: the model's files left in the cache by a program that read them just before refused the VRAM tier's
    // allocation (B70, 2026-10-19).  So the model's own files leave the cache first; they are read from the drive.
    try {
        if (Runtime::get().device().get_backend() == sycl::backend::ext_oneapi_level_zero)
            for (const std::string& path : gguf_split_paths(opt_.gguf)) strata::platform::drop_file_cache(path);
        model_ = std::make_unique<GgufModel>(GgufModel::open(opt_.gguf));
    } catch (const std::exception& e) {
        err = std::string("plain model: ") + e.what();
        return false;
    }
    if (!read_model_spec(model_->meta(), spec_, err)) return false;
    // the layer split: its points (given, or from the GPUs' memory), each rising inside the trunk, a GPU each
    const int64_t nl = spec_.n_layer;
    // the GPUs a split takes, as generate.cpp's: the runtime's first, then the ones --split-device names, else the
    // other cards (the processor's own graphics only by name: its memory is the RAM, which holds the experts already)
    if (opt_.split_auto || !opt_.split_at.empty()) {
        split_devs_.assign(1, 0);
        if (!opt_.split_devices.empty())
            split_devs_.insert(split_devs_.end(), opt_.split_devices.begin(), opt_.split_devices.end());
        else
            for (int d = 1; d < Runtime::count(); ++d)
                if (!device_info(d).integrated) split_devs_.push_back(d);
        for (size_t i = 1; i < split_devs_.size(); ++i)
            if (split_devs_[i] <= 0 || split_devs_[i] >= Runtime::count() ||
                std::count(split_devs_.begin(), split_devs_.end(), split_devs_[i]) > 1) {
                err = "plain model: --split-device names GPU " + std::to_string(split_devs_[i]) +
                      ", not another usable one";
                return false;
            }
    }
    if (opt_.split_auto && !split_auto_points(err)) return false;
    const std::vector<int64_t>& at = opt_.split_at;
    for (size_t i = 0; i < at.size(); ++i)
        if (at[i] < 1 || at[i] >= nl || (i > 0 && at[i] <= at[i - 1])) {
            err = "plain model: --layer-split takes rising blocks from 1 to " + std::to_string(nl - 1);
            return false;
        }
    if (!at.empty() && at.size() + 1 > split_devs_.size()) {
        err = "plain model: --layer-split needs " + std::to_string(at.size() + 1) + " GPUs, " +
              std::to_string(split_devs_.size()) + " are there for it";
        return false;
    }
    lb_ = 0;
    le_ = at.empty() ? nl : at[0];
    dev_ = at.empty() ? -1 : split_devs_[0];
    // --gpu-layers: a dense model's blocks after the first N on the CPU (their weights in the RAM, read from the GGUF
    // and packed as --device cpu's), the head on the GPU; not given, from the free VRAM
    if (opt_.gpu_layers < 0 && spec_.dense() && at.empty() && !opt_.split_auto && opt_.batch_slots == 0 &&
        !opt_.encoder && !opt_.vision && k::cpu::cpu_avx2_ok() && !auto_gpu_layers(err))
        return false;
    if (opt_.gpu_layers >= 0 && opt_.gpu_layers < nl) {
        if (!spec_.dense() || !at.empty() || opt_.split_auto || opt_.batch_slots > 0 || opt_.encoder || opt_.vision) {
            err = "plain model: --gpu-layers splits a dense model (qwen35, plamo3) between one GPU and the CPU, with no "
                  "layer split, --batch, pictures or encoder";
            return false;
        }
        if (opt_.gpu_layers == 0) {
            opt_.cpu = true;
            return cpu_load(err);
        }
        if (!k::cpu::cpu_avx2_ok()) {
            err = "plain model: --gpu-layers puts blocks on the CPU, which needs AVX2";
            return false;
        }
        le_ = opt_.gpu_layers;
        opt_.spec = 0;   // the MTP layer reads the last block's rows on the GPU: not with a CPU tail
        cpu_tail_ = std::make_unique<PlainCpu>();
        if (!cpu_tail_->load(*model_, spec_, opt_, err, le_, nl)) return false;
        std::fprintf(stderr, "xestrata plain: blocks %lld .. %lld on the CPU (%s), the first %lld and the head on the GPU\n",
                     (long long) le_, (long long) (nl - 1), cpu_tail_->describe().c_str(), (long long) le_);
    }
    // --expert-cache-device1..3: the first GPUs no stage runs on, their contexts made before the experts' RAM copy is
    // registered with any GPU (RemoteExperts::preflight, as generate.cpp's)
    if (opt_.expert_cache_remote_placement != "stripe" && opt_.expert_cache_remote_placement != "layer") {
        err = "plain model: --expert-cache-remote-placement takes stripe or layer";
        return false;
    }
    for (int r = 0, next = 1; r < 3; ++r) {
        if (opt_.expert_cache_remote[(size_t) r] <= 0) continue;
        while (next < Runtime::count() && !at.empty() &&
               std::find(split_devs_.begin(), split_devs_.begin() + (std::ptrdiff_t) at.size() + 1, next) !=
                   split_devs_.begin() + (std::ptrdiff_t) at.size() + 1)
            ++next;
        double free_gib = 0;
        if (next >= Runtime::count() || !RemoteExperts::preflight(next, free_gib, err)) {
            err = "plain model: --expert-cache-device" + std::to_string(r + 1) + ": " +
                  (next >= Runtime::count() ? std::string("no GPU left that runs no stage") : err);
            return false;
        }
        std::fprintf(stderr, "xestrata plain: helper GPU %d (%s), %.2f GiB free before the experts' RAM copy\n", next,
                     strata::gpu::device_name(next).c_str(), free_gib);
        helper_dev_[(size_t) r] = next++;
    }
    {
        const OnDevice on(dev_);
        if (!load_arch(err)) return false;
    }
    for (size_t i = 0; i < at.size(); ++i) {
        auto st = std::make_unique<PlainModel>();
        const int64_t lb = at[i], le = i + 1 < at.size() ? at[i + 1] : nl;
        if (!st->load_stage(opt_, this, lb, le, split_devs_[i + 1], err)) return false;
        next_.push_back(std::move(st));
    }
    finish_experts();
    if (!open_helpers(err)) return false;
    strata::gpu::pin_reserve_release();   // loaded: the pinned memory's last reserve goes back
    set_adapt_period(4 * (int64_t) opt_.adapt_every);
    adapt_auto_ = adapt_period_ > 0 && !opt_.adapt_every_given;
    for (PlainModel* st : chain()) {
        const OnDevice on(st->dev_);
        if (!st->reset(err)) return false;
    }
    if (!next_.empty()) {
        std::string where;
        for (PlainModel* st : chain())
            where += (where.empty() ? "" : ", ") + strata::gpu::device_name(st->dev_) + " blocks " +
                     std::to_string(st->lb_) + ".." + std::to_string(st->le_ - 1);
        std::fprintf(stderr, "xestrata plain: a layer split: %s\n", where.c_str());
    }
    return true;
}

bool PlainModel::load_stage(const PlainOptions& opt, PlainModel* root, int64_t lb, int64_t le, int dev,
                            std::string& err) {
    opt_ = opt;
    if (opt.reserve_later_mib >= 0) opt_.reserve_mib = opt.reserve_later_mib;   // --vram-reserve-later-mib
    root_ = root;
    lb_ = lb;
    le_ = le;
    dev_ = dev;
    route_dump_ = root->route_dump_;
    const OnDevice on(dev_);
    try {
        model_ = std::make_unique<GgufModel>(GgufModel::open(opt_.gguf));
    } catch (const std::exception& e) {
        err = std::string("plain model: ") + e.what();
        return false;
    }
    if (!read_model_spec(model_->meta(), spec_, err)) return false;
    return load_arch(err);
}

bool PlainModel::load_arch(std::string& err) {
    return spec_.arch == ModelArch::Glm5Next ? load_glm5(err) : load_qwen(err);
}

// The points from the GPUs' memory and links.  Each GPU's share of the MoE layers is in proportion to the memory it has
// for them (its free memory, less its reserve, and the head on the last), so that each GPU's tier holds about the same
// part of its layers' experts; except a GPU whose link from the host is less than half the first's: it takes only the
// layers its tier holds whole, since a prompt chunk copies each of its layers' missed experts over that link.  GLM-5.3-
// Flash on a B70 and an RTX 4070 behind the chipset (Gen4 x4, 4.3x slower): a 4K prompt 100 tok/s with the 4070's
// share by memory (11 layers), 149 with the 2 layers and the MTP layer it holds whole, 165 on the B70 alone; the
// decode 12.0, 11.3 and 10.8 tok/s (2026-10-20).
bool PlainModel::auto_gpu_layers(std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t nl = s.n_layer;
    size_t f = 0, t = 0;
    {
        const OnDevice on(dev_);
        if (!strata::gpu::mem_info(&f, &t)) { err = "plain model: the device's memory is unknown"; return false; }
    }
    const uint64_t free_b = counted_free(f, f, opt_.vram_free_mib);   // (mem_info applies XESTRATA_VRAM_LIMIT_MIB)
    // each block's weights; the head and the final norm; a block's K/V (a sliding-window layer's ring: the window and
    // a prompt chunk) or recurrent state
    std::vector<uint64_t> block((size_t) nl, 0);
    uint64_t head = 0;
    for (size_t i = 0; i < model_->size(); ++i)
        for (const TensorInfo& ti : model_->shard(i).tensors()) {
            const uint64_t b = tensor_payload_bytes(ti);
            if (ti.name.rfind("blk.", 0) == 0) {
                const int64_t l = std::strtoll(ti.name.c_str() + 4, nullptr, 10);
                if (l >= 0 && l < nl) block[(size_t) l] += b;
            } else if (ti.name == head_name() || ti.name == "output_norm.weight") {
                head += b;
            }
        }
    const int64_t nkv = std::max<int64_t>(1, s.n_head_kv.empty() ? 1 : *std::max_element(s.n_head_kv.begin(), s.n_head_kv.end()));
    const int64_t chunk = std::max<int64_t>(opt_.prompt_chunk, opt_.max_t);
    for (int64_t l = 0; l < nl; ++l) {
        const int64_t cells = s.layer_swa(l) ? std::min<int64_t>(opt_.max_cells, s.swa_window + chunk) : opt_.max_cells;
        block[(size_t) l] += s.recurrent[(size_t) l]
                                 ? (uint64_t) (s.ssm_d_state * s.ssm_dt_rank * s.ssm_d_state + (2 * s.ssm_n_group * s.ssm_d_state + s.ssm_d_inner) * 3) * 4
                                 : (uint64_t) (cells * nkv * s.key_length) * 2 * (opt_.kv_int8 ? 1 : 2);
    }
    // the window's and the prompt's buffers and the driver's share: the layer split's reserve (split_auto_points)
    // - the device's room, core::vram_room, plus the 1 GiB this stage's windows and prompt path take
    const uint64_t reserve = opt_.reserve_mib >= 0
                                 ? (uint64_t) opt_.reserve_mib << 20
                                 : core::vram_room(t, /*parts*/ 1, kTierRoomLeast) + (1ull << 30);
    uint64_t used = head + reserve;
    int64_t fit = 0;
    while (fit < nl && used + block[(size_t) fit] <= free_b) used += block[(size_t) fit++];
    if (fit == nl) return true;   // every block on the GPU (gpu_layers stays -1)
    opt_.gpu_layers = head + reserve > free_b ? 0 : fit;   // below the head and the reserve: the CPU alone
    std::fprintf(stderr, "xestrata plain: %.2f GiB of VRAM free, the head and a reserve %.2f GiB, the blocks %.2f GiB: %s\n",
                 free_b / 1073741824.0, (head + reserve) / 1073741824.0,
                 std::accumulate(block.begin(), block.end(), uint64_t{0}) / 1073741824.0,
                 opt_.gpu_layers == 0 ? "the CPU alone" : ("the first " + std::to_string(fit) + " blocks on the GPU").c_str());
    return true;
}

bool PlainModel::split_auto_points(std::string& err) {
    opt_.split_at.clear();
    const int n = (int) split_devs_.size();
    if (n < 2) return true;
    const ModelSpec& s = spec_;
    const int64_t M = s.n_moe_layers();
    if (M < n) return true;
    uint64_t head = 0;
    if (size_t at = 0; const TensorInfo* t = model_->find(head_name(), &at)) head = tensor_payload_bytes(*t);
    // a MoE layer's experts (the largest layer's three tensors: a file's layers can differ in type), and whether the
    // last GPU runs the MTP layer as well
    uint64_t layer_b = 0;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        if (!s.layer_routes(l)) continue;
        uint64_t b = 0;
        for (const char* w : {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"})
            if (size_t at = 0; const TensorInfo* t = model_->find("blk." + std::to_string(l) + "." + w + ".weight", &at))
                b += tensor_payload_bytes(*t);
        layer_b = std::max(layer_b, b);
    }
    const bool mtp = opt_.spec >= 2 && model_->find("blk." + std::to_string(s.n_layer) + ".nextn.eh_proj.weight") != nullptr;
    std::vector<double> room((size_t) n, 0.0), link((size_t) n, 0.0);
    for (int d = 0; d < n; ++d) {
        const OnDevice on(split_devs_[(size_t) d]);
        size_t f = 0, t = 0;
        if (!strata::gpu::mem_info(&f, &t)) { err = "plain model: a GPU's memory is unknown"; return false; }
        const uint64_t free_b = f, total_b = t;   // (mem_info applies STRATA_VRAM_LIMIT_MIB)
        const uint64_t reserve =
            core::vram_room(total_b, /*parts*/ 1, kTierRoomLeast) + (1ull << 30) + (d == n - 1 ? head : 0);
        room[(size_t) d] = free_b > reserve ? (double) (free_b - reserve) : 0.0;
        link[(size_t) d] = std::max(0.0, probe_host_to_device_gbps()) * 1e9;
    }
    const std::vector<int64_t> count = choose_split_by_room(M, layer_b, mtp, room, link);
    if (count.empty()) return true;
    int64_t seen = 0, l = 0, want = 0;
    for (int d = 0; d + 1 < n; ++d) {
        want += count[(size_t) d];
        while (l < s.n_layer && seen < want) seen += s.layer_routes(l++) ? 1 : 0;
        if (l >= s.n_layer) break;
        opt_.split_at.push_back(l);
    }
    if ((int) opt_.split_at.size() != n - 1) opt_.split_at.clear();
    for (int d = 0; d < n; ++d)
        std::fprintf(stderr, "xestrata plain: --layer-split auto: %s, %.1f GiB for experts, %.1f GB/s from the host: %lld "
                             "MoE layers\n", strata::gpu::device_name(split_devs_[(size_t) d]).c_str(),
                     room[(size_t) d] / 1073741824.0, link[(size_t) d] / 1e9, (long long) count[(size_t) d]);
    return true;
}

bool PlainModel::rope_scalings(const ModelSpec& s, const PlainOptions& opt, k::RopeScaling& full, k::RopeScaling& swa,
                               std::string& err) {
    using RST = k::RopeScalingType;
    k::RopeScaling rs;
    rs.freq_base = opt.rope_freq_base > 0 ? opt.rope_freq_base : s.rope_base;
    const std::string type = !opt.rope_scaling.empty() ? opt.rope_scaling : s.rope_scaling_type;
    rs.type = type == "yarn" ? RST::YaRN : type == "linear" ? RST::Linear : RST::None;
    if (rs.type != RST::None) {
        rs.factor = opt.rope_scale > 0 ? opt.rope_scale : s.rope_scaling_factor > 1.0 ? s.rope_scaling_factor : 1.0;
        rs.orig_ctx = opt.yarn_orig_ctx > 0 ? opt.yarn_orig_ctx
                      : s.rope_scaling_orig_ctx >= 1 ? s.rope_scaling_orig_ctx : (double) s.context_length;
        rs.ext_factor = opt.yarn_ext_factor >= 0 ? opt.yarn_ext_factor : rs.type == RST::YaRN ? 1.0 : 0.0;
    }
    rs.attn_factor = opt.yarn_attn_factor;
    // the file's ramp bounds where it writes them (plamo3), else the options'
    rs.beta_fast = s.yarn_beta_fast > 0 ? s.yarn_beta_fast : opt.yarn_beta_fast;
    rs.beta_slow = s.yarn_beta_slow > 0 ? s.yarn_beta_slow : opt.yarn_beta_slow;
    if (opt.rope_freq_scale > 0 && rs.type == RST::None) {   // llama.cpp: a linear scaling of 1 / scale
        rs.type = RST::Linear;
        rs.factor = 1.0 / opt.rope_freq_scale;
    }
    if (const char* why = k::rope_scaling_invalid(rs)) {
        err = std::string("plain model: the rope scaling: ") + why;
        return false;
    }
    full = rs;
    swa = k::RopeScaling{};   // a sliding-window layer: its own base, not scaled
    swa.freq_base = s.rope_base_swa > 0 ? s.rope_base_swa : rs.freq_base;
    return true;
}

bool PlainModel::load_qwen(std::string& err) {
    const ModelSpec& s = spec_;
    if (s.arch != ModelArch::Qwen35Moe && s.arch != ModelArch::Qwen3Moe && s.arch != ModelArch::Qwen35 &&
        s.arch != ModelArch::Plamo3 && s.arch != ModelArch::Qwen3) {
        err = "plain model: " + s.arch_name + " is not run by this engine yet";
        return false;
    }
    const bool q35 = s.arch == ModelArch::Qwen35Moe || s.arch == ModelArch::Qwen35;
    dense_ = s.dense();
    plamo_ = s.arch == ModelArch::Plamo3;
    // a dense model's decode windows with a graph or straight onto the stream, whichever is faster on this GPU
    // (StepForm, timed in stage_forward).  The other models keep their graphs: their windows answer the CPU's rings
    // between graph segments
    if (!dense_) step_form_.fix(0);
    if ((opt_.encoder || opt_.stream_blocks) &&
        (!opt_.encoder || !dense_ || !opt_.split_at.empty() || opt_.split_auto)) {
        err = "plain model: an encoder (streamed or not) is a dense model (qwen35, qwen3) on one GPU";
        return false;
    }
    if (plamo_ && (opt_.encoder || opt_.vision || opt_.kv_q4 != 0)) {
        err = "plain model: plamo3 is not an encoder, has no pictures and keeps its K/V in FP16 or int8";
        return false;
    }
    if (opt_.encoder) opt_.spec = 0;   // no decode: no MTP layer, no drafts
    has_gate_ = q35;
    has_shared_ = q35 && !dense_;
    bool shape_ok = s.key_length == s.value_length && (s.key_length == 128 || s.key_length == 256) &&
                    (s.rope_dims == 64 || s.rope_dims == s.key_length) && s.n_expert_used <= 15 && s.n_dense_lead == 0;
    if (q35)
        shape_ok = shape_ok && s.ssm_d_state == 128 && s.ssm_d_conv == 4 && s.ssm_n_group > 0 && s.n_embd % 8 == 0 &&
                   s.ssm_dt_rank % s.ssm_n_group == 0 && s.ssm_d_inner == s.ssm_dt_rank * s.ssm_d_state &&
                   (dense_ ? s.n_ff > 0 : s.n_ff_shexp > 0);
    if (!shape_ok) {
        err = "plain model: a shape this engine's kernels are not built for";
        return false;
    }
    cs_ = strata::gpu::stream_create();
    if (cs_ == nullptr) { err = "plain model: stream create failed"; return false; }
    // process constants, before any graph: the rotation (its base, and the scaling the options or the file ask for, as
    // generate.cpp resolves it), and the GDN output gate (SiLU in qwen35moe)
    if (!rope_scalings(s, opt_, rope_full_, rope_swa_, err)) return false;
    k::rope_scaling_set(rope_full_);
    k::gdn_set_silu_gate(q35);

    const GgufModel& m = *model_;
    if (!load_embedding(err)) return false;
    if (last() || cpu_tail_ != nullptr) {   // the head: the last stage's (an encoder's: none; a CPU tail's: here)
        if ((!opt_.encoder && !upload(m, head_name(), head_, err)) ||
            !upload_f32(m, "output_norm.weight", (uint64_t) s.n_embd, out_norm_, err))
            return false;
        if (!opt_.encoder && (head_.n_in != s.n_embd || head_.n_out != n_vocab_)) {
            err = std::string("plain model: ") + head_name() + "'s shape";
            return false;
        }
    }

    const uint64_t N = (uint64_t) s.n_embd, E = (uint64_t) s.n_expert;
    blocks_.resize((size_t) s.n_layer);
    // the attention's K/V heads, the same in every block (checked as the blocks load)
    for (int64_t l = 0; l < s.n_layer && nkv_ == 0; ++l)
        if (!s.recurrent[(size_t) l]) nkv_ = s.n_head_kv[(size_t) l];
    int64_t n_gdn = 0, n_attn = 0;
    for (int64_t l = 0; l < s.n_layer; ++l) {   // the indices count every block; this stage loads its own
        Block& b = blocks_[(size_t) l];
        b.recurrent = s.recurrent[(size_t) l];
        if (b.recurrent) b.gdn = n_gdn++;
        else b.attn = n_attn++;
        if (opt_.stream_blocks) {
            streamed_.resize((size_t) s.n_layer);
            stream_into_ = &streamed_[(size_t) l];
        }
        const bool ok = l < lb_ || l >= le_ || load_block(m, l, b, err);
        stream_into_ = nullptr;
        if (!ok) return false;
    }
    // the DFlash2 drafter's taps (set_taps): this stage's share of the layers, below the last
    tap_local_.clear();
    tap_slot_.assign((size_t) s.n_layer, -1);
    for (int l : tap_req_)
        if (l >= lb_ && l < le_ && l < s.n_layer - 1) {
            tap_slot_[(size_t) l] = (int) tap_local_.size();
            tap_local_.push_back(l);
        }
    if (!tap_local_.empty() && tap_dev_ == nullptr &&
        !strata::gpu::alloc_device(&tap_dev_, (size_t) opt_.max_t * tap_local_.size() * s.n_embd * 4)) {
        err = "plain model: the DFlash2 tap staging does not fit";
        return false;
    }
    // the MTP layer, when the window checks drafts and the file has it (its K/V the last attention block's): the last
    // stage's
    {
        const std::string p = "blk." + std::to_string(s.n_layer) + ".nextn.";
        model_mtp_ = opt_.spec >= 2 && q35 && s.n_nextn == 1 && m.find(p + "eh_proj.weight") != nullptr;
        has_mtp_ = model_mtp_ && last();
    }
    if (opt_.dry) {   // the sizes only (encoder_buffer_bytes takes nkv_ and the blocks' shapes, read above)
        if (!opt_.encoder || !opt_.stream_blocks) {
            err = "plain model: a dry load is a streamed encoder's";
            return false;
        }
        return true;
    }
    if (has_mtp_) {
        const std::string p = "blk." + std::to_string(s.n_layer) + ".nextn.";
        mtp_.attn = n_attn++;
        if (!load_block(m, s.n_layer, mtp_, err) || !upload(m, p + "eh_proj.weight", mtp_eh_, err) ||
            !upload_f32(m, p + "enorm.weight", N, mtp_enorm_, err) ||
            !upload_f32(m, p + "hnorm.weight", N, mtp_hnorm_, err))
            return false;
        if (mtp_eh_.n_in != 2 * s.n_embd || mtp_eh_.n_out != s.n_embd) {
            err = "plain model: the MTP layer's eh_proj has another shape";
            return false;
        }
        mtp_head_norm_ = out_norm_;
        if (m.find(p + "shared_head_norm.weight") != nullptr &&
            !upload_f32(m, p + "shared_head_norm.weight", N, mtp_head_norm_, err))
            return false;
    }

    if (nkv_ == 0) { err = "plain model: the model has no attention block"; return false; }
    if (opt_.kv_q4 != 0 && s.key_length != 256) {
        err = "plain model: --kv q4_0 and k8v4 rotate heads of 256 (kv_q4.hpp); this model's are " +
              std::to_string((long long) s.key_length) + " wide";
        return false;
    }
    if (!tune_uploaded(err)) return false;
    // the BF16 projections' two GEMV forms timed for each shape a window takes, before any graph is recorded; the
    // head is left to the default form (its timing would allocate a second copy of it)
    try {
        auto tune = [&](const Block& b) {
            for (const W* w : {&b.qkv, &b.z, &b.alpha, &b.beta, &b.ssm_out, &b.q, &b.k, &b.v, &b.o, &b.sh_gate, &b.sh_up,
                               &b.sh_down})
                if (w->type == kBF16) k::bf16_gemv_fp32_mmvf_multi_tune(w->n_in, w->n_out, cs_);
        };
        for (int64_t l = lb_; l < le_; ++l) tune(blocks_[(size_t) l]);
        // the GDN recurrence's forms (Qwen3.8's verify-window kernel), one a window size
        if (std::any_of(blocks_.begin() + lb_, blocks_.begin() + le_, [](const Block& b) { return b.recurrent; }))
            k::gdn_step_tune((int) (2 * s.ssm_n_group * s.ssm_d_state + s.ssm_d_inner), (int) s.ssm_n_group,
                             (int) s.ssm_dt_rank, cs_);
        if (has_mtp_) {
            tune(mtp_);
            if (mtp_eh_.type == kBF16) k::bf16_gemv_fp32_mmvf_multi_tune(mtp_eh_.n_in, mtp_eh_.n_out, cs_);
        }
    } catch (const std::exception& e) {
        err = std::string("plain model: timing the BF16 products and the GDN recurrence: ") + e.what();
        return false;
    }
    // per-sequence state
    const int64_t C = 2 * s.ssm_n_group * s.ssm_d_state + s.ssm_d_inner;
    state_bytes_ = (size_t) (s.ssm_d_state * s.ssm_dt_rank * s.ssm_d_state) * 4;
    conv_bytes_ = (size_t) (C * (s.ssm_d_conv - 1)) * 4;
    const k::QsaShapes qs = attn_shapes();
    gdn_state_.assign((size_t) n_gdn, nullptr);
    gdn_conv_.assign((size_t) n_gdn, nullptr);
    for (int64_t l = lb_; l < le_; ++l) {
        const Block& b = blocks_[(size_t) l];
        if (b.recurrent) {
            float *st = nullptr, *cv = nullptr;
            if (!strata::gpu::alloc_device(&st, (size_t) (s.ssm_d_state * s.ssm_dt_rank * s.ssm_d_state) * 4) ||
                !strata::gpu::alloc_device(&cv, (size_t) (C * (s.ssm_d_conv - 1)) * 4)) {
                err = "plain model: no VRAM for the recurrent states";
                return false;
            }
            allocs_.push_back(st);
            allocs_.push_back(cv);
            gdn_state_[(size_t) b.gdn] = st;
            gdn_conv_[(size_t) b.gdn] = cv;
        }
    }
    // the states after a window's rows but its last, for going back to the last accepted draft: a window checks up to
    // spec - 1 drafts (at most the window kernel's snapshots); every stage's recurrent blocks
    // (--suffix-draft, --lookup-chain: as many as a window's drafts can be, MTP layer or not)
    const int lookup = opt_.suffix_draft > 0 || opt_.lookup_chain > 0 ? std::min(opt_.max_t, k::kVerifyMaxT + 1) - 1 : 0;
    if (model_mtp_ || lookup > 0) {
        if (model_mtp_) opt_.spec = std::min({opt_.spec, opt_.max_t, k::kVerifyMaxT + 1});
        n_mtp_ = model_mtp_ ? opt_.spec - 1 : 0;
        n_snap_ = std::max(n_mtp_, lookup);
        gdn_snap_.assign(gdn_state_.size(), {});
        conv_snap_.assign(gdn_state_.size(), {});
        for (size_t g = 0; g < gdn_state_.size(); ++g)
            for (int t = 0; t < n_snap_ && gdn_state_[g] != nullptr; ++t) {
                float *st = nullptr, *cv = nullptr;
                if (!strata::gpu::alloc_device(&st, (size_t) (s.ssm_d_state * s.ssm_dt_rank * s.ssm_d_state) * 4) ||
                    !strata::gpu::alloc_device(&cv, (size_t) (C * (s.ssm_d_conv - 1)) * 4)) {
                    err = "plain model: no VRAM for the recurrent states' copies (--spec)";
                    return false;
                }
                allocs_.push_back(st);
                allocs_.push_back(cv);
                gdn_snap_[g].push_back(st);
                conv_snap_[g].push_back(cv);
            }
    }
    // --pipeline-windows 2: on a stage before the last, the copies a guessed window may have to give back (pl_save)
    if (model_mtp_ && n_snap_ > 0 && opt_.pipeline_windows >= 2 && !last()) {
        pl_bak_state_.assign(gdn_state_.size(), {});
        pl_bak_conv_.assign(gdn_state_.size(), {});
        pl_pre_state_.assign(gdn_state_.size(), nullptr);
        pl_pre_conv_.assign(gdn_state_.size(), nullptr);
        const size_t sb = (size_t) (s.ssm_d_state * s.ssm_dt_rank * s.ssm_d_state) * 4, cb = (size_t) (C * (s.ssm_d_conv - 1)) * 4;
        auto take = [&](float*& p, size_t bytes) {
            if (!strata::gpu::alloc_device(&p, bytes)) return false;
            allocs_.push_back(p);
            return true;
        };
        bool ok = true;
        for (size_t g = 0; g < gdn_state_.size() && ok; ++g) {
            if (gdn_state_[g] == nullptr) continue;
            ok = take(pl_pre_state_[g], sb) && take(pl_pre_conv_[g], cb);
            for (int t = 0; t < n_snap_ && ok; ++t) {
                float *st = nullptr, *cv = nullptr;
                ok = take(st, sb) && take(cv, cb);
                pl_bak_state_[g].push_back(st);
                pl_bak_conv_[g].push_back(cv);
            }
        }
        pl_ready_ = ok;
        if (!ok)
            std::fprintf(stderr, "xestrata plain: --pipeline-windows 2: no VRAM for the guessed windows' copies on %s; the "
                                 "decode stays serial\n", strata::gpu::device_name(dev_).c_str());
    }
    // the drafts' head over a token subset (Qwen3.8's draft head, mtp.cpp): on the RX 9060 XT the whole Q8_0 head of
    // Qwen3.6's 248,320 tokens took 1.7 ms a draft, at the card's bandwidth, 2026-10-15.  The engine's row layouts
    // (upload) rewrite each row in place, so a row keeps its GGUF type's size.
    if (has_mtp_ && n_mtp_ > 0 && !opt_.draft_vocab.empty()) {
        const int gguf_type = head_.type == k::kNativeQ8Rows    ? kQ80
                              : head_.type == k::kNativeQ6KRows ? kQ6K
                                                                : head_.type;
        const size_t row = head_.type == kBF16 ? (size_t) head_.n_in * 2
                                               : k::cpu::native_row_bytes(gguf_type, head_.n_in);
        subset_ = std::make_unique<DraftSubset>();   // the shared part (core/draft_head), as Qwen3.8's drafter's
        if (!subset_->load(opt_.draft_vocab, n_vocab_, head_.data, row, cs_, "xestrata plain", err)) return false;
        if (subset_->size() > 0) {
            dhead_ = const_cast<uint8_t*>(subset_->rows());
            dvocab_ = subset_->ids();
            n_dvocab_ = subset_->size();
        }
    }
    n_pages_ = (opt_.max_cells + qs.page_size - 1) / qs.page_size;
    opt_.max_cells = n_pages_ * qs.page_size;
    // the sliding-window layers' ring (plamo3): the window and the most cells a run writes at once (a prompt chunk, a
    // window), whole pages, so that a chunk's last writes leave the cells its first query sees; none where the
    // context is not longer than that
    attn_swa_.assign((size_t) n_attn, false);
    for (int64_t l = 0; l < s.n_layer; ++l)
        if (blocks_[(size_t) l].attn >= 0) attn_swa_[(size_t) blocks_[(size_t) l].attn] = blocks_[(size_t) l].swa;
    // XESTRATA_PLAIN_RING=0: those layers' pools as long as the others' (A/B)
    static const bool ring_env = [] {
        const char* v = strata::env_get("PLAIN_RING");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    if (s.swa_window > 0 && opt_.batch_slots == 0 && ring_env) {
        const int64_t run = std::max<int64_t>(opt_.prompt_chunk, opt_.max_t);
        const int64_t pages = (s.swa_window + run + qs.page_size - 1) / qs.page_size;
        if (pages < n_pages_) ring_pages_ = pages;
        if (ring_pages_ > 0 && first())
            std::fprintf(stderr, "xestrata plain: the sliding-window layers' K/V in a ring of %lld cells (window %lld)\n",
                         (long long) (ring_pages_ * qs.page_size), (long long) s.swa_window);
    }
    // the decode attention's form for this GPU (an fp16 cache), timed before any graph is recorded and before the cache
    // takes its VRAM (an encoder does not decode)
    if (!opt_.kv_int8 && opt_.kv_q4 == 0 && !opt_.encoder) {
        try {
            k::qsa_dense_attn_tune(qs, opt_.max_cells, cs_);
        } catch (const std::exception& e) {
            err = std::string("plain model: timing the decode attention: ") + e.what();
            return false;
        }
    }
    // and the prompt attention's kernel (prompt_attn: the matrix engines', the FP16 one or the decode one), as long
    // as the VRAM is free; GLM-5.3-Flash's prompt takes its own (plain_prompt_glm5.cpp)
    if (opt_.prompt_chunk > 0 && !glm_)
        k::prompt_attn_tune(qs, opt_.kv_q4 == 1 ? 2 : opt_.kv_q4 == 2 ? 3 : opt_.kv_int8 ? 1 : 0, opt_.max_cells, cs_);
    {
        const size_t cells = (size_t) (n_pages_ * qs.n_head_kv * qs.page_size);
        k_pool_.assign((size_t) n_attn, nullptr);
        v_pool_.assign((size_t) n_attn, nullptr);
        k_q_.assign((size_t) n_attn, nullptr);
        v_q_.assign((size_t) n_attn, nullptr);
        k_scale_.assign((size_t) n_attn, nullptr);
        v_scale_.assign((size_t) n_attn, nullptr);
        k_q4_.assign((size_t) n_attn, nullptr);
        v_q4_.assign((size_t) n_attn, nullptr);
        // --kv-grow: Qwen3.8-Flash-Next's rule for the default (the B70 reports 64 KiB pages, and that engine's cache
        // in 2 MiB segments ran its prompts 45% slower there; the RTX 4070 reports 2 MiB and was faster)
        const bool asked = opt_.kv_grow > 0 || (opt_.kv_grow < 0 && vmm_large_pages());
        const char* why = dense_ ? "a dense model has no expert tier to give VRAM up"
                        : !(first() && last()) ? "a layer split"
                        : opt_.batch_slots > 0 ? "--batch"
                        : opt_.encoder ? "an encoder"
                        : !vmm_available() || !strata::gpu::vmem_supported() ? "no virtual memory on this GPU"
                        : opt_.vram_segment_mib < 64 ? "--vram-segment-mib is below 64" : nullptr;
        const bool grow = asked && why == nullptr;
        if (opt_.kv_grow > 0 && !grow) std::fprintf(stderr, "xestrata plain: --kv-grow is off: %s\n", why);
        if (grow) opt_.vram_elastic = true;   // the tier in segments it can give up (vram)
        if (grow)
            std::fprintf(stderr, "xestrata plain: the elastic K/V: a parked conversation's cells %s\n",
                         strata::core::kv_grow_hold() ? "stay while it waits" : "go back after each request");
        for (int64_t i = 0; i < n_attn; ++i) {   // this stage's attention blocks, the MTP layer on the last
            bool mine = has_mtp_ && i == mtp_.attn;
            for (int64_t l = lb_; l < le_ && !mine; ++l) mine = blocks_[(size_t) l].attn == i;
            if (!mine) continue;
            const size_t a = (size_t) i;
            KvPtrs p;
            // an encoder reads the whole prompt in one chunk (encode), so a layer's attention reads only the K/V it
            // wrote itself: every layer shares the first's cells (Qwen3-VL-8B's 36 layers of 4096 cells: 16 MiB, not
            // 590)
            if (opt_.encoder && i > 0 && (k_pool_[0] != nullptr || k_q_[0] != nullptr || k_q4_[0] != nullptr)) {
                k_pool_[a] = k_pool_[0];
                v_pool_[a] = v_pool_[0];
                k_scale_[a] = k_scale_[0];
                v_scale_[a] = v_scale_[0];
                k_q_[a] = k_q_[0];
                v_q_[a] = v_q_[0];
                k_q4_[a] = k_q4_[0];
                v_q4_[a] = v_q4_[0];
                continue;
            }
            const bool ok = alloc_kv(p, ringed(a) ? (size_t) (ring_pages_ * qs.n_head_kv * qs.page_size) : cells, grow);
            k_pool_[a] = p.k_pool;
            v_pool_[a] = p.v_pool;
            k_scale_[a] = p.k_scale;
            v_scale_[a] = p.v_scale;
            k_q_[a] = p.k_q;
            v_q_[a] = p.v_q;
            k_q4_[a] = p.k_q4;
            v_q4_[a] = p.v_q4;
            if (!ok) {
                err = "plain model: no VRAM for the K/V cache (" + std::to_string(opt_.max_cells) + " cells)";
                return false;
            }
        }
        std::vector<int32_t> table((size_t) n_pages_);
        for (int64_t i = 0; i < n_pages_; ++i) table[(size_t) i] = (int32_t) i;
        if (!strata::gpu::alloc_device(&page_table_, table.size() * 4) ||
            !strata::gpu::copy(page_table_, table.data(), table.size() * 4)) {
            err = "plain model: the page table";
            return false;
        }
        allocs_.push_back(page_table_);
        if (ring_pages_ > 0) {   // logical page p in the ring's page p % ring_pages_
            for (int64_t i = 0; i < n_pages_; ++i) table[(size_t) i] = (int32_t) (i % ring_pages_);
            if (!strata::gpu::alloc_device(&swa_table_, table.size() * 4) ||
                !strata::gpu::copy(swa_table_, table.data(), table.size() * 4)) {
                err = "plain model: the sliding-window layers' page table";
                return false;
            }
            allocs_.push_back(swa_table_);
        }
        if (grow) {   // the first cells (STRATA_KV_GROW_INIT, 16384 by default) mapped now
            const char* iv = strata::env_get("KV_GROW_INIT");
            const long long init = iv != nullptr && std::strtoll(iv, nullptr, 10) > 0 ? std::strtoll(iv, nullptr, 10) : 16384;
            kv_cells_ = 0;
            if (!kv_map(std::min<int64_t>(opt_.max_cells, init), err)) return false;
        }
    }
    if (opt_.batch_slots > 0 && !opt_.vision && !opt_.encoder && !alloc_batch_slots(err)) return false;
    // --vision: every cell's rotary position from the table (kernels/mrope.hpp), the identity until a request with
    // pictures writes it; set before any graph is recorded, which reads the pointer
    if (opt_.vision) {
        std::vector<int32_t> pos3((size_t) opt_.max_cells * 3);
        for (int64_t c = 0; c < opt_.max_cells; ++c)
            for (int i = 0; i < 3; ++i) pos3[(size_t) (c * 3 + i)] = (int32_t) c;
        if (!strata::gpu::alloc_device(&mrope_, pos3.size() * 4) ||
            !strata::gpu::copy(mrope_, pos3.data(), pos3.size() * 4)) {
            err = "plain model: the rotary positions' table";
            return false;
        }
        allocs_.push_back(mrope_);
        // for the file's rotation and sections (Qwen3.5's 64 and 11/11/10, Qwen3-VL's 128 and 24/20/20)
        const auto& sec = s.rope_sections;
        k::mrope_table_set(mrope_, (int) s.rope_dims, sec.size() >= 3 ? (int) sec[1] : 11,
                           sec.size() >= 3 ? (int) sec[2] : 11);
    }

    // the window's buffers
    const uint64_t T = (uint64_t) opt_.max_t, NH = (uint64_t) s.n_head, HD = (uint64_t) s.key_length;
    const uint64_t NKV = (uint64_t) qs.n_head_kv, HV = (uint64_t) s.ssm_dt_rank, ZV = (uint64_t) s.ssm_d_inner;
    const uint64_t K = (uint64_t) s.n_expert_used, FS = (uint64_t) (dense_ ? s.n_ff : s.n_ff_shexp);
    hand_w_ = s.n_embd;
    if (!mapped(T * N * 4, (void**) &h_emb_, (void**) &m_emb_) ||
        !mapped(T * k::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) ||
        !mapped(T * (NH + NKV) * 4, (void**) &h_pos_, (void**) &m_pos_) ||
        !mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) || !mapped(16, (void**) &h_prob_, (void**) &m_prob_) ||
        (!first() && !mapped(T * N * 4, (void**) &h_in_, (void**) &m_in_)) ||
        (!last() && !mapped(T * N * 4, (void**) &h_hand_, (void**) &m_hand_))) {
        err = "plain model: mapped staging allocation failed";
        return false;
    }
    attn_scratch_floats_ = (int64_t) k::qsa_decode_attn_scratch_floats(opt_.max_cells, qs);
    const uint64_t max_in = std::max<uint64_t>({N, ZV, NH * HD, FS, has_mtp_ ? 2 * N : 0});
    auto dev = [&](auto** p, uint64_t bytes) {
        void* d = nullptr;
        if (!strata::gpu::alloc_device(&d, (size_t) std::max<uint64_t>(bytes, 256))) return false;
        allocs_.push_back(d);
        *p = (std::remove_reference_t<decltype(*p)>) d;
        return true;
    };
    if (!dev(&step_, T * k::kStepCount * 4) || !dev(&pos_, T * (NH + NKV) * 4) || !dev(&x_, T * N * 4) ||
        !dev(&hn_, T * N * 4) || !dev(&mix_, T * N * 4) || !dev(&xq_, k::native_q8_1_bytes((int) max_in, (int) T)) ||
        !dev(&qkv_, T * (uint64_t) C * 4) || !dev(&hc_, T * (uint64_t) C * 4) || !dev(&z_, T * ZV * 4) ||
        !dev(&beta_, T * HV * 4) || !dev(&gate_, T * HV * 4) || !dev(&y_, T * ZV * 4) || !dev(&counts_, (T + 1) * 4) ||
        !dev(&qfull_, T * std::max(NH * 2 * HD, 2 * N) * 4) ||
        !dev(&qcur_, T * NH * HD * 4) || !dev(&kraw_, T * NKV * HD * 4) || !dev(&kcur_, T * NKV * HD * 4) ||
        !dev(&vcur_, T * NKV * HD * 4) || !dev(&attn_, T * NH * HD * 4) || !dev(&attn32_, T * NH * HD * 4) ||
        !dev(&attn_scratch_, T * (uint64_t) attn_scratch_floats_ * 4) || !dev(&rlog_, T * E * 4) ||
        !dev(&ids_, T * K * 4) || !dev(&w_, T * K * 4) || !dev(&shared_, T * N * 4) || !dev(&sh_g_, (T + 4) * 4) ||
        !dev(&sh_gate_, T * FS * 2 * 4) || !dev(&sh_up_, T * FS * 4) || !dev(&moe_out_, T * N * 4) ||
        !dev(&logits_, T * (uint64_t) n_vocab_ * 4) || (has_mtp_ && !dev(&mtp_h_, T * N * 4))) {
        err = "plain model: no VRAM for the window's buffers";
        return false;
    }
    std::vector<int32_t> counts((size_t) T + 1);
    for (size_t i = 0; i < counts.size(); ++i) counts[i] = (int32_t) i;
    if (!strata::gpu::copy(counts_, counts.data(), counts.size() * 4)) { err = "plain model: the uploads failed"; return false; }
    if (!strata::gpu::stream_sync(cs_)) { err = "plain model: the uploads failed"; return false; }
    setup_int2();   // before the expert tier takes what VRAM is left
    if (!(dense_ ? open_dense(err) : open_experts(err))) return false;
    exec_.assign((size_t) opt_.max_t + 1, nullptr);
    segs_.assign((size_t) opt_.max_t + 1, {});
    mexec_.assign((size_t) opt_.max_t + 1, nullptr);
    msegs_.assign((size_t) opt_.max_t + 1, {});
    rexec_.assign((size_t) opt_.max_t + 1, nullptr);
    mrexec_.assign((size_t) opt_.max_t + 1, nullptr);
    return true;
}

bool PlainModel::prompt_int8() const {
    const char* v = strata::env_get("PLAIN_MMQ");
    if (v != nullptr && std::strtol(v, nullptr, 10) == 0) return false;
    if (!prefill::mmq::chosen()) return false;
    // the products' outputs in 128s (2 n_ff and n_embd), their K in blocks of 32; gate's rows right before up's
    const int64_t N = spec_.n_embd, FF = spec_.n_ff_exp;
    if (N % 128 != 0 || (2 * FF) % 128 != 0 || FF % 32 != 0) return false;
    for (const auto& f : k::cpu::expert_layout().fmt) {
        if (!k::iq_mmq_type_ok(f.gu_type) || !k::iq_mmq_type_ok(f.d_type)) return false;
        if (k::cpu::native_mixed_gu(f) ? !k::iq_mmq_type_ok(f.up_type)   // gate and up: a product each
                                       : f.up_off != (size_t) FF * k::iq_mmq_row_bytes(f.gu_type, N))
            return false;
    }
    return true;
}

int64_t PlainModel::prompt_int8_below() const {
    // STRATA_PLAIN_MMQ=1: the int8 products at every size, untimed
    if (const char* v = strata::env_get("PLAIN_MMQ"); v != nullptr && v[0] != '\0') return INT64_MAX;
    const auto& lay = k::cpu::expert_layout();
    if (dense_ || spec_.n_expert <= 0 || !lay.native || lay.fmt.empty()) return INT64_MAX;
    // as prompt_routed launches them: the experts in VRAM in one int8 launch, kGroup a launch on the FP16 path
    prefill::mmq::ExpertLaunches launches;
    launches.n_expert = spec_.n_expert;
    launches.int8_group = (int) spec_.n_expert;
    launches.f16_group = PromptBufs::kGroup;
    return prefill::mmq::int8_rows_below(lay.fmt.front(), (size_t) lay.blob_bytes(0), INT64_MAX, launches);
}

void PlainModel::prompt_carve(int64_t T, PromptBufs& b, uint8_t* dev, uint8_t* host, uint64_t& dev_bytes,
                              uint64_t& host_bytes, PromptPieces* pieces) const {
    const ModelSpec& s = spec_;
    const auto& lay = k::cpu::expert_layout();
    const uint64_t N = (uint64_t) s.n_embd, NH = (uint64_t) s.n_head, HD = (uint64_t) s.key_length, NKV = (uint64_t) nkv_;
    const uint64_t HV = (uint64_t) s.ssm_dt_rank, ZV = (uint64_t) s.ssm_d_inner;
    const uint64_t C = (uint64_t) (2 * s.ssm_n_group * s.ssm_d_state) + ZV, E = (uint64_t) s.n_expert;
    // FS: the shared expert's width, or qwen35's dense FFN's (its products go through the same buffers)
    const uint64_t K = (uint64_t) s.n_expert_used, FS = (uint64_t) (dense_ ? s.n_ff : s.n_ff_shexp);
    const uint64_t FF = (uint64_t) s.n_ff_exp;
    // the MTP layer's input rows ([embedding | trunk row], 2 N wide) go through qfull and x16
    const uint64_t U = (uint64_t) T, TK = U * K, QW = std::max<uint64_t>((has_gate_ ? 2u : 1u) * NH * HD, has_mtp_ ? 2 * N : 0);
    // GLM-5.3-Flash's widest inputs: the four streams (the mHC's mixes), the MLA's output, the dense FFN's down, KDA's
    const uint64_t gw = glm_ ? std::max<uint64_t>({4 * N, NH * (uint64_t) s.value_length_mla, (uint64_t) s.n_ff,
                                                   NH * (uint64_t) s.kda_head_dim})
                             : 0;
    const uint64_t max_in = std::max<uint64_t>({N, ZV, NH * HD, FS, FF, has_mtp_ ? 2 * N : 0, gw});
    b.T = T;
    b.gu_elems = (int64_t) (2 * FF * N);
    b.d_elems = (int64_t) (N * FF);
    b.int8 = prompt_int8();
    b.int8_below = b.int8 ? prompt_int8_below() : INT64_MAX;
    if (b.int8_below == 0) {   // FP16 at every size
        b.int8 = false;
        b.int8_below = INT64_MAX;
    }
    // the largest dense weight, dequantized at once (Gemm::native slices past 2^25 values)
    b.gemm_elems = (int64_t) mtp_eh_.n_in * mtp_eh_.n_out;   // 0 without the MTP layer
    std::vector<const Block*> all;
    all.reserve(blocks_.size() + 1);
    for (const Block& bl : blocks_) all.push_back(&bl);
    if (has_mtp_) all.push_back(&mtp_);
    for (const Block* bl : all)
        for (const W* w : {&bl->qkv, &bl->z, &bl->alpha, &bl->beta, &bl->ssm_out, &bl->q, &bl->k, &bl->v, &bl->o,
                           &bl->sh_gate, &bl->sh_up, &bl->sh_down})
            b.gemm_elems = std::max<int64_t>(b.gemm_elems, (int64_t) w->n_in * w->n_out);
    for (const GBlock& g : gblocks_)
        for (const W* w : {&g.hc_attn_fn, &g.hc_ffn_fn, &g.q, &g.k, &g.v, &g.f_a, &g.f_b, &g.beta, &g.g_a, &g.g_b, &g.q_a,
                           &g.q_b, &g.kv_a, &g.idx_q_b, &g.idx_k, &g.idx_gate, &g.o, &g.ffn_gate, &g.ffn_up, &g.ffn_down})
            b.gemm_elems = std::max<int64_t>(b.gemm_elems, (int64_t) w->n_in * w->n_out);
    b.gemm_elems = std::min<int64_t>(b.gemm_elems, (int64_t) 1 << 25);
    uint64_t at = 0, in = 0;   // in: the bytes taken in piece pi
    size_t pi = 0;
    const bool sizing = pieces != nullptr && pieces->at.empty();
    const uint64_t cap = sizing ? max_alloc_bytes() : 0;
    if (sizing) pieces->bytes.clear();
    auto take = [&](auto*& ptr, uint64_t bytes) {
        using P = std::remove_reference_t<decltype(ptr)>;
        const uint64_t n = (std::max<uint64_t>(bytes, 1) + 255) & ~255ull;
        if (pieces == nullptr) {
            ptr = dev ? (P) (dev + at) : nullptr;
        } else {   // the next piece where this one is full (sizing: at the cap; laying out: at the piece's bytes)
            const uint64_t room = sizing ? cap : pi < pieces->bytes.size() ? pieces->bytes[pi] : 0;
            if (in > 0 && in + n > room) {
                ++pi;
                in = 0;
            }
            if (sizing) {
                if (pieces->bytes.size() <= pi) pieces->bytes.push_back(0);
                pieces->bytes[pi] = in + n;
            }
            ptr = !sizing && pi < pieces->at.size() ? (P) (pieces->at[pi] + in) : nullptr;
            in += n;
        }
        at += n;
    };
    take(b.x, U * N * 4); take(b.hn, U * N * 4); take(b.mix, U * N * 4); take(b.x16, U * max_in * 2);
    take(b.qkv, U * C * 4); take(b.z, U * ZV * 4); take(b.ab, U * 2 * HV * 4); take(b.gate, U * HV * 4);
    take(b.beta, U * HV * 4); take(b.h, U * C * 4); take(b.y, U * ZV * 4); take(b.y16, U * ZV * 2);
    take(b.qfull, U * QW * 4); take(b.qcur, U * NH * HD * 4); take(b.kraw, U * NKV * HD * 4);
    take(b.kcur, U * NKV * HD * 4); take(b.vcur, U * NKV * HD * 4); take(b.attn, U * NH * HD * 4);
    take(b.attn32, U * NH * HD * 4);
    take(b.steps, U * k::kStepCount * 4); take(b.pos_q, U * NH * 4); take(b.pos_k, U * NKV * 4);
    take(b.rlog, U * E * 4); take(b.ids, TK * 4); take(b.w, TK * 4); take(b.shared, U * N * 4);
    take(b.sh_g, (U + 4) * 4); take(b.sh_gate, U * FS * 4); take(b.sh_up, U * FS * 4);
    // the FP16 path's: the routed rows and the SwiGLU in FP16, kGroup experts' FP16 weights, the down product's rows
    // in grouped order (the int8 one writes them in routing order, straight into parts); beside the int8 products'
    // where a chunk of T tokens reaches int8_below rows an expert spread over every expert.  They are 0.9 GB at 8192
    // tokens of Qwen3.6-35B-A3B, and taken where a layer could not reach it they halved the chunk the tier lends (the
    // RTX 4070, 32K: 16384 -> 8192 tokens, its missed experts read 1.4 times as often, 2622 -> 1963 tok/s).  Only
    // where every expert has its slot (tier_whole_): elsewhere a layer's routed experts are seldom all in VRAM, which
    // the FP16 path needs (prompt_routed), and the buffers the prompt borrows from the tier left more of them out
    // (Ornith-1.5-35B-A3B GSQ-RCO 3bit on the RTX 4070, 197 slots a layer, a 32K prompt: 23,714 experts read from RAM
    // with the int8 products' buffers alone, 27,187 with these, 2,934 -> 2,824 tok/s, 2026-11-12)
    b.f16 = !b.int8 ||
            (tier_whole_ && b.int8_below != INT64_MAX && (int64_t) TK >= b.int8_below * (int64_t) E);
    const uint64_t f16 = b.f16 ? 1 : 0;
    take(b.xs16, f16 * TK * N * 2); take(b.gu, TK * 2 * FF * 4); take(b.hh, TK * FF * 4); take(b.h16, f16 * TK * FF * 2);
    take(b.dm, f16 * TK * N * 4); take(b.parts, TK * N * 4); take(b.moe_out, U * N * 4);
    take(b.src, TK * 4); take(b.slot, TK * 4); take(b.bounds, (E + 1) * 4); take(b.entry, TK * 4);
    take(b.dq_gu, f16 * (uint64_t) PromptBufs::kGroup * (uint64_t) b.gu_elems * 2);
    take(b.dq_d, f16 * (uint64_t) PromptBufs::kGroup * (uint64_t) b.d_elems * 2);
    take(b.stage, 2 * (uint64_t) PromptBufs::kGroup * lay.max_blob);   // two halves: one copied while one is read
    // the int8 activations: values, then a scale and a sum per 32 (iq_mmq_act_bytes); twice as many where a product
    // reads them with every block repeated (Q2_K, iq_mmq_type_dup), and a second copy of the routed rows for a layer
    // whose gate and up differ in that
    bool dup = false, dup_mixed = false;
    for (const auto& f : lay.fmt) {
        const int up = f.up_type >= 0 ? f.up_type : f.gu_type;
        dup = dup || k::iq_mmq_type_dup(f.gu_type) || k::iq_mmq_type_dup(up) || k::iq_mmq_type_dup(f.d_type);
        dup_mixed = dup_mixed || k::iq_mmq_type_dup(f.gu_type) != k::iq_mmq_type_dup(up);
    }
    for (const Block* bl : all)
        for (const W* w : {&bl->qkv, &bl->z, &bl->ssm_out, &bl->q, &bl->k, &bl->v, &bl->o})
            dup = dup || k::iq_mmq_type_dup(w->type);
    for (const GBlock& g : gblocks_)
        for (const W* w : {&g.q, &g.k, &g.v, &g.o, &g.q_a, &g.q_b, &g.ffn_gate, &g.ffn_down})
            dup = dup || k::iq_mmq_type_dup(w->type);
    dup = dup || (has_mtp_ && k::iq_mmq_type_dup(mtp_eh_.type));
    const uint64_t rep = dup ? 2 : 1;
    const uint64_t xq_cols = rep * std::max<uint64_t>(TK * N, U * max_in);   // the routed rows', or a dense input's
    take(b.xq, xq_cols + 2 * (xq_cols / 32) * 4); take(b.hq, rep * (TK * FF + 2 * TK * (FF / 32) * 4));
    take(b.xq2, dup_mixed && b.int8 ? rep * (TK * N + 2 * TK * (N / 32) * 4) : 0);
    if (!(dup_mixed && b.int8) && (dev != nullptr || (pieces != nullptr && !sizing))) b.xq2 = nullptr;
    take(b.identity, std::max(TK, U) * 4);   // a dense product's T rows too (a dense model routes none)
    take(b.wtab, E * sizeof(void*)); take(b.rbounds, (2 * E + 2) * 4); take(b.dense_bounds, 2 * sizeof(int32_t));
    // the CPU's share of the missed experts (plain_prompt.cpp): at most a window's tokens an expert, so 8 E rows
    b.cpu_cap = std::min<int64_t>((int64_t) TK, (int64_t) k::cpu::MAXT * (int64_t) E);
    take(b.cpu, (uint64_t) b.cpu_cap * N * 4); take(b.cpu_row, (uint64_t) b.cpu_cap * 4);
    take(b.gemm_scratch, (uint64_t) b.gemm_elems * 2);
    if (glm_) {   // plain_prompt_glm5.cpp
        const uint64_t HC = 4, MIX = (2 + HC) * HC, S = (uint64_t) s.kda_head_dim, GC = NH * S;
        const uint64_t QLR = (uint64_t) s.q_lora_rank, KVR = (uint64_t) s.kv_lora_rank;
        const uint64_t QK = (uint64_t) s.key_length_mla, VD = (uint64_t) s.value_length_mla;
        const uint64_t IH = (uint64_t) s.idx_n_head, ID = (uint64_t) s.idx_key_length;
        const uint64_t FW = (uint64_t) std::max<int64_t>(s.n_ff, s.n_ff_shexp);
        const uint64_t blocks = (uint64_t) (opt_.max_cells / s.idx_kpool + 1);
        k::QsaShapes qs;
        qs.idx_block = s.idx_kpool;
        qs.idx_top_k = s.idx_top_k;
        const uint64_t cap = (uint64_t) k::qsa_selection_width(k::kTopkMaxCells, qs);
        take(b.g_hs, U * HC * N * 4); take(b.g_mixes, U * MIX * 4); take(b.g_post, U * HC * 4);
        take(b.g_comb, U * HC * HC * 4); take(b.g_y, U * N * 4); take(b.g_steps, U * k::kStepCount * 4);
        for (float** p : {&b.g_q, &b.g_k, &b.g_v, &b.g_qc, &b.g_kc, &b.g_vc, &b.g_f, &b.g_gate, &b.g_g2, &b.g_o, &b.g_oy})
            take(*p, U * GC * 4);
        take(b.g_fa, U * S * 4); take(b.g_ga, U * S * 4); take(b.g_beta, U * NH * 4);
        take(b.g_qa, U * QLR * 4); take(b.g_qr, U * QLR * 4); take(b.g_qb, U * NH * QK * 4); take(b.g_qbT, U * NH * QK * 4);
        for (float** p : {&b.g_absT, &b.g_abs, &b.g_lat, &b.g_latT}) take(*p, U * NH * KVR * 4);
        take(b.g_kv, U * KVR * 4); take(b.g_voT, U * NH * VD * 4); take(b.g_vo, U * NH * VD * 4);
        take(b.g_iq, U * IH * ID * 4); take(b.g_ik, U * ID * 4); take(b.g_ig, U * ID * 4); take(b.g_iw, U * IH * 4);
        take(b.g_scores, U * blocks * 4); take(b.g_ids, U * cap * 4);
        take(b.g_ffg, U * FW * 4); take(b.g_ffu, U * FW * 4);
    }
    dev_bytes = at;
    at = 0;
    auto hold = [&](auto*& ptr, uint64_t bytes) {
        ptr = host ? (std::remove_reference_t<decltype(ptr)>) (host + at) : nullptr;
        at += (std::max<uint64_t>(bytes, 1) + 255) & ~255ull;
    };
    hold(b.emb, U * N * 4); hold(b.h_steps, U * k::kStepCount * 4); hold(b.h_pos, U * (NH + NKV) * 4);
    hold(b.h_ids, TK * 4); hold(b.h_src, TK * 4); hold(b.h_slot, TK * 4); hold(b.h_bounds, (E + 1) * 4);
    hold(b.h_entry, TK * 4);
    hold(b.h_wtab, E * sizeof(void*)); hold(b.h_rbounds, (2 * E + 2) * 4);
    hold(b.h_hn, U * N * 4); hold(b.h_cpu, (uint64_t) b.cpu_cap * N * 4); hold(b.h_cpu_row, (uint64_t) b.cpu_cap * 4);
    hold(b.h_nact, U * k::cpu::kNativeActBytes); hold(b.h_dense_bounds, 2 * sizeof(int32_t));
    hold(b.h_stage, 2 * (uint64_t) PromptBufs::kGroup * lay.max_blob);
    host_bytes = at;
}

bool PlainModel::alloc_prompt_dev(int64_t T, uint64_t room) {
    PromptPieces p;
    uint64_t db = 0, hb = 0;
    PromptBufs sized = pb_;
    prompt_carve(T, sized, nullptr, nullptr, db, hb, &p);
    if (db > room) return false;
    for (const uint64_t n : p.bytes) {
        void* d = nullptr;
        if (n > max_alloc_bytes() || !strata::gpu::alloc_device(&d, (size_t) n)) {
            for (uint8_t* q : p.at) strata::gpu::free(q);
            return false;
        }
        p.at.push_back(static_cast<uint8_t*>(d));
    }
    prompt_dev_ = std::move(p);
    prompt_dev_T_ = T;
    return true;
}

bool PlainModel::encoder_buffers(int64_t T, std::string& err) {
    if (!prompt_dev_.at.empty() && prompt_dev_T_ >= T && enc_host_T_ >= T) return true;
    encoder_release();
    if (enc_host_T_ < T) {   // the pinned staging for the longer chunk too
        uint64_t db = 0, hb = 0;
        prompt_carve(T, pb_, nullptr, nullptr, db, hb);
        void* h = nullptr;
        if (!strata::gpu::alloc_host(&h, hb)) {
            err = "plain model: no pinned memory for the encoder's " + std::to_string(T) + " tokens";
            return false;
        }
        strata::gpu::free(prompt_host_);
        prompt_host_ = h;
        enc_host_T_ = T;
    }
    if (!alloc_prompt_dev(T, UINT64_MAX)) {
        err = "plain model: no VRAM for the encoder's buffers of " + std::to_string(T) + " tokens (" +
              std::to_string(encoder_buffer_bytes(T) >> 20) + " MiB)";
        return false;
    }
    uint64_t db = 0, hb = 0;
    prompt_carve(T, pb_, nullptr, (uint8_t*) prompt_host_, db, hb);
    opt_.prompt_chunk = T;
    return true;
}

void PlainModel::encoder_release() {
    if (!opt_.encoder || prompt_dev_.at.empty()) return;
    strata::gpu::stream_sync(cs_);
    for (uint8_t* p : prompt_dev_.at) strata::gpu::free(p);
    prompt_dev_ = PromptPieces();
    prompt_dev_T_ = 0;
}

uint64_t PlainModel::encoder_buffer_bytes(int64_t T) {
    PromptPieces p;
    PromptBufs sized = pb_;
    uint64_t db = 0, hb = 0;
    prompt_carve(T, sized, nullptr, nullptr, db, hb, &p);
    return db;
}

bool PlainModel::open_experts(std::string& err) {
    const ModelSpec& s = spec_;
    n_moe_ = s.n_moe_layers() + (model_mtp_ ? 1 : 0);
    const int64_t L = n_moe_, E = s.n_expert;
    // this stage's MoE layers: its blocks', then the MTP layer's (the tiers' last MoE layer)
    moe_local_.clear();
    for (int64_t l = lb_; l < le_; ++l)
        if (s.layer_routes(l)) moe_local_.push_back(s.moe_index(l));
    moe_trunk_ = (int64_t) moe_local_.size();
    if (has_mtp_) moe_local_.push_back(L - 1);
    const int64_t LS = (int64_t) moe_local_.size();
    level_zero_ = Runtime::get().device().get_backend() == sycl::backend::ext_oneapi_level_zero;
    if (root_ == this) {
        std::vector<int64_t> blocks;
        for (int64_t l = 0; l < s.n_layer; ++l)
            if (s.layer_routes(l)) blocks.push_back(l);
        if (model_mtp_) blocks.push_back(s.n_layer);   // the MTP layer's experts: the tiers' last MoE layer
        if (!expert_layout_from_gguf(opt_.gguf, blocks, E, s.n_embd, s.n_ff_exp, err)) return false;
    }
    const auto& lay = k::cpu::expert_layout();
    if (const std::vector<std::string> bad = native_layer_problems(true); !bad.empty()) {
        err = "plain model: " + bad.front();
        return false;
    }
    // the RAM tier: every expert loaded into pinned RAM when the host has the room (the GPU reads them over PCIe and
    // the pool from RAM), else the GGUF read in place through the page cache.  The room is the resident copy's (#11,
    // resident_pin_fits): the pinned experts and 8 GiB beside them
    // On Level Zero the device's own allocations also need RAM beside the pinned experts: on the B70 (xe, 32 GiB) with
    // 100 GiB of host memory registered and 14 GiB left, 2 GiB device allocations were refused or the device was lost;
    // with 85 GiB and 29 GiB left, 24 GiB of them ran (2026-10-19).  So the room there counts the VRAM as well (every
    // Level Zero GPU's, with a layer split).  The root opens the tier; the later stages read from it.
    if (root_ == this) {
        in_ram_ = !opt_.mmap_experts && opt_.resident_gib == 0;
        // a shared arena an engine before this one filled (the server's, across an image job's unload) is in RAM
        // already, and counted as used: its RAM is not asked for again
        if (in_ram_ && !(!opt_.shared_arena.empty() && PinnedArena::shared_filled(opt_.shared_arena))) {
            detail::HostMemory hm;
            const uint64_t need = lay.total + lay.max_blob;
            uint64_t beside = 8ull << 30;
            const int n_dev = opt_.split_at.empty() ? 1 : (int) opt_.split_at.size() + 1;
            for (int d = 0; d < n_dev; ++d) {
                const OnDevice on(opt_.split_at.empty() ? -1 : split_devs_[(size_t) d]);
                size_t f = 0, t = 0;
                if (Runtime::get().device().get_backend() == sycl::backend::ext_oneapi_level_zero &&
                    strata::gpu::mem_info(&f, &t))
                    beside += (uint64_t) t;
            }
            beside = detail::ram_headroom(beside);   // the RAM the user asked to leave free (--ram-reserve-mib)
            std::string why;
            const uint64_t cma = strata::platform::cma_free_bytes();   // not for pinned pages (upstream #3698)
            if (detail::host_available_memory(hm) &&
                !detail::resident_pin_fits(need, 0, beside, std::min(hm.available - std::min(hm.available, cma),
                                                                     hm.cgroup_limit), why)) {
                std::fprintf(stderr, "xestrata plain: the experts need %.1f GiB of RAM and %.1f GiB beside them, %.1f GiB "
                                     "is available: read in place\n",
                             (double) need / 1073741824.0, (double) beside / 1073741824.0,
                             (double) std::min(hm.available, hm.cgroup_limit) / 1073741824.0);
                in_ram_ = false;
            }
        }
        if (in_ram_) {
            auto arena = std::make_unique<ArenaExpertSource>();
            arena->set_gguf(opt_.gguf);
            strata::gpu::pin_reserve_take();   // the pinned buffers after the arena keep their room (gpu.hpp)
            const bool arena_ok = arena->open(std::string(), L, E, /*threads=*/6, err, opt_.shared_arena);
            strata::gpu::pin_reserve_release_soft();
            if (!arena_ok) return false;
            if (!arena->ram_warning().empty())
                std::fprintf(stderr, "xestrata plain: WARNING: %s\n", arena->ram_warning().c_str());
            std::fprintf(stderr, "xestrata plain: experts in RAM, %.2f GiB at %.2f GiB/s\n",
                         (double) lay.total / 1073741824.0, arena->load_gib_per_second());
            src_own_ = std::move(arena);
        } else {
            auto file = std::make_unique<FileExpertSource>();
            file->set_gguf(opt_.gguf);
            if (const char* v = strata::env_get("FETCH_THREADS"))   // the prompt's read-ahead threads, as generate.cpp
                if (const long n = std::strtol(v, nullptr, 10); n > 0) file->set_fetch_threads((int) std::min<long>(n, 256));
            if (!file->open(std::string(), L, E, err)) return false;
            src_own_ = std::move(file);
        }
        src_ = src_own_.get();
        pool_own_ = std::make_unique<k::cpu::ExpertPool>(opt_.pool_workers, /*pin=*/true, opt_.pool_host_works,
                                                         /*first_core=*/0, opt_.pool_tasks);
        pool_ = pool_own_.get();
    } else {
        in_ram_ = root_->in_ram_;
        src_ = root_->src_;
        pool_ = root_->pool_;
    }

    // the VRAM tier: what the device has left, less the window's arena and a reserve, in slots of each layer's blob,
    // the same number for every layer of this stage
    uint64_t free_b = 0;
    {
        size_t f = 0, t = 0;
        if (!strata::gpu::mem_info(&f, &t)) { err = "plain model: the device's memory is unknown"; return false; }
        free_b = f;
        total_b_ = t;
        free_b = counted_free(free_b, free_b, opt_.vram_free_mib);   // (mem_info applies STRATA_VRAM_LIMIT_MIB)
    }
    // the prompt path's host buffers, for its longest chunk; the device ones are borrowed from the tier a prompt at a
    // time (prompt_borrow), so the tier is sized from all the free memory
    // pinned memory can be short (WSL 2 caps it): the chunk is halved until its buffers fit, and with none the prompt
    // is read through the decode windows (--prompt-chunk 0), said
    if (opt_.prompt_chunk > 0) {
        const int64_t asked = opt_.prompt_chunk;
        uint64_t db = 0, hb = 0;
        for (; opt_.prompt_chunk >= 16; opt_.prompt_chunk /= 2) {
            prompt_carve(opt_.prompt_chunk, pb_, nullptr, nullptr, db, hb);
            if (strata::gpu::alloc_host(&prompt_host_, hb)) break;
            prompt_host_ = nullptr;
        }
        // a layer split's: the rows between the stages, for the longest chunk (the stages' chunks may be shorter)
        if (prompt_host_ != nullptr && !(first() && last()) &&
            !strata::gpu::alloc_host(&hand_host_, (size_t) (asked * hand_w_) * 4)) {
            strata::gpu::free(prompt_host_);
            prompt_host_ = nullptr;
        }
        if (prompt_host_ == nullptr) {
            std::fprintf(stderr, "xestrata plain: no pinned memory for the prompt path's chunks (%lld tokens asked): "
                                 "prompts are read through the decode windows\n", (long long) asked);
            opt_.prompt_chunk = 0;
        } else {
            if (opt_.prompt_chunk != asked)
                std::fprintf(stderr, "xestrata plain: the prompt path's chunks are %lld tokens (the pinned memory for %lld "
                                     "was not there)\n", (long long) opt_.prompt_chunk, (long long) asked);
            prompt_carve(opt_.prompt_chunk, pb_, nullptr, (uint8_t*) prompt_host_, db, hb);
        }
    }
    const uint64_t window_b = (uint64_t) opt_.max_t * (uint64_t) s.n_expert_used * (uint64_t) s.n_embd * 4 * 3 +
                              16 * lay.max_blob + (64ull << 20);
    // what is allocated after the tier (the products' libraries, the graphs) and the driver's margin: the room of
    // the device's memory (core::vram_room; its floor kTierRoomLeast above, its 768 MiB ceiling past 24 GiB)
    const uint64_t reserve = opt_.reserve_mib >= 0 ? (uint64_t) opt_.reserve_mib << 20
                                                   : core::vram_room(total_b_, /*parts*/ 1, kTierRoomLeast);
    const uint64_t room = free_b > window_b + reserve ? free_b - window_b - reserve : 0;
    uint64_t per_layer_b = 0;
    for (int64_t l : moe_local_) per_layer_b += lay.bytes[(size_t) l];
    int64_t q = per_layer_b > 0 ? (int64_t) (room / per_layer_b) : 0;
    q = std::min<int64_t>(q, E);
    if (opt_.expert_slots >= 0) q = std::min<int64_t>(q, opt_.expert_slots / std::max<int64_t>(L, 1));
    if (q < 1 && LS > 0) {
        err = "plain model: no VRAM is left for the expert tier (" + std::to_string(free_b >> 20) + " MiB free)";
        return false;
    }
    // the xe driver gives VRAM only with as much RAM free beside it, and the file cache does not count as free there
    // (78 GiB of pinned experts, a model file in the cache: the tier's 24 GiB refused with 17 GiB free, B70,
    // 2026-10-19): the OS is made to give the cache back first
    if (level_zero_) strata::platform::make_free_ram(per_layer_b * (uint64_t) q + (2ull << 30));
    // the slots each layer gets: with a profile, as Qwen3.8-Flash-Next's tier fills from one (generate.cpp), the
    // profile's pairs in its order over all the layers while the tier's bytes last; without one, the layers alike
    // (a profile of the trunk's layers serves a load with the MTP layer as well, and one with the MTP layer a load
    // without it: its pairs dropped)
    std::vector<int64_t> quota((size_t) L, 0);
    if (!opt_.profile.empty()) {
        int64_t ranked_slots = 0;
        std::string err2;
        if (!read_expert_profile(opt_.profile, L, E, prior_, ranked_slots, err)) {
            const int64_t other = model_mtp_ ? L - 1 : L + 1;
            if (!read_expert_profile(opt_.profile, other, E, prior_, ranked_slots, err2)) return false;
            prior_.erase(std::remove_if(prior_.begin(), prior_.end(), [&](const auto& p) { return p.first >= L; }),
                         prior_.end());
            err.clear();
        }
        std::vector<char> local((size_t) L, 0);
        for (int64_t l : moe_local_) local[(size_t) l] = 1;
        // Ornith-1.5 (qwen35moe) the layers alike: the order over all the layers lost hits there (RTX 4070, Q6_K,
        // 256 tokens: 72.8-73.1% -> 70.8%, 91.1 / 90.8 -> 87.5 / 87.6 tok/s, 2026-10-26)
        const bool alike = s.arch == ModelArch::Qwen35Moe;
        quota = plan_profile_fill(prior_, L, E, per_layer_b * (uint64_t) q, [&](int64_t l) { return lay.bytes[(size_t) l]; },
                                  [&](int64_t l) { return local[(size_t) l] != 0; }, alike)
                    .quota;
    } else {
        for (int64_t l : moe_local_) quota[(size_t) l] = q;
    }
    std::vector<int64_t> slot_bytes;
    free_.assign((size_t) L, {});
    slot_layer_.clear();
    const int64_t most = *std::max_element(quota.begin(), quota.end());
    for (int64_t i = 0; i < most; ++i)   // the layers in turn while each has slots (prompt_borrow takes the last)
        for (int64_t l : moe_local_)
            if (i < quota[(size_t) l]) {
                free_[(size_t) l].push_back((int32_t) slot_bytes.size());
                slot_layer_.push_back(l);
                slot_bytes.push_back((int64_t) lay.bytes[(size_t) l]);
            }
    for (auto& f : free_) std::reverse(f.begin(), f.end());   // taken from the back: the lowest slot first
    cache_.set_max_piece(max_alloc_bytes());   // in pieces where the tier passes the device's largest allocation
    if (opt_.vram_elastic) {   // #533: in segments, as generate.cpp's cache; one GPU only (a split has a tier per GPU)
        const char* why = !(first() && last()) ? "a layer split has a tier per GPU"
                        : opt_.vram_segment_mib < 64 ? "--vram-segment-mib is below 64"
                        : !strata::gpu::vmem_supported() ? "this GPU's runtime has no virtual memory" : nullptr;
        if (why != nullptr) {
            std::fprintf(stderr, "xestrata plain: --vram-elastic is off: %s\n", why);
            opt_.vram_elastic = false;
        } else {
            cache_.set_segment_bytes(opt_.vram_segment_mib << 20);
        }
    }
    if (!slot_bytes.empty() && !cache_.open_sized(slot_bytes, L, E, err)) return false;
    n_slots_ = (int64_t) slot_bytes.size();
    tier_whole_ = q == E;
    if (!opt_.prompt_chunk_given && opt_.prompt_chunk > 0 && prompt_host_ != nullptr && first() && last())
        auto_prompt_chunk(q == E ? room - per_layer_b * (uint64_t) q : 0);
    // every expert has its slot and VRAM is left beside the tier: the prompt's buffers there, for the longest chunk
    // they fit (prompt_borrow then lends no slot, so the prompt reads no expert from RAM)
    if (q == E && opt_.prompt_chunk > 0 && prompt_host_ != nullptr) {
        const uint64_t left = room - per_layer_b * (uint64_t) q;
        for (int64_t T = opt_.prompt_chunk; T >= 16; T /= 2)
            if (alloc_prompt_dev(T, left)) {
                for (uint8_t* p : prompt_dev_.at) allocs_.push_back(p);
                break;
            }
    }
    res_.assign((size_t) (L * E), -1);
    heat_.assign((size_t) (L * E), 0.0);
    if (!strata::gpu::alloc_host(&fill_tmp_, (size_t) lay.max_blob)) {
        err = "plain model: pinned staging for the slot fills";
        return false;
    }

    for (size_t l = 0; lay.native && l < lay.fmt.size(); ++l)
        if (lay.fmt[l].had_gu || lay.fmt[l].had_d) {   // APR: only Qwen3.8's verify window rotates its experts' input
            err = "plain model: Hadamard-folded experts (hadamard.txt) are not supported for this model";
            return false;
        }
    if (n_slots_ <= 0) {
        err = "plain model: the decode windows need the VRAM expert tier";
        return false;
    }
    MoeWindowShape ws;
    ws.n_embd = s.n_embd;
    ws.n_expert = E;
    ws.k = s.n_expert_used;
    ws.max_t = opt_.max_t;
    MoeWindowTier tier;
    tier.cache_base = cache_.device_slot(0);
    tier.blob = (int64_t) lay.max_blob;
    tier.slot_off = cache_.slot_offsets();
    tier.n_slots = n_slots_;
    if (!win_.init(ws, tier, err)) return false;
    win_.sink()->pcie_mode = 2;   // the PCIe share staged by the copy kernel inside the window (the split's timing)
    // the windows with every routed expert in VRAM run without the host (resident_plan), as Qwen3.8-Flash-Next's:
    // the residency table on the device, and the ids the host counts after the window (res_count)
    if (win_.device_plans() && MoeWindow::resident_graph_on() && (int64_t) opt_.max_t * s.n_expert_used <= 128) {
        void* d = nullptr;
        if (strata::gpu::alloc_device(&d, res_.size() * 4) &&
            mapped((uint64_t) L * (uint64_t) opt_.max_t * (uint64_t) s.n_expert_used * 4, (void**) &h_rroute_,
                   (void**) &m_rroute_)) {
            allocs_.push_back(d);
            res_d_ = (int32_t*) d;
            res_shadow_.assign(res_.size(), -2);   // differs from every slot: the first window uploads the table
        } else if (d != nullptr) {
            strata::gpu::free(d);
        }
    }

    disp_.pool = pool_;
    disp_.src = src_;
    disp_.n_expert = E;
    disp_.cache = &cache_;
    disp_.cache_base = tier.cache_base;
    disp_.cache_blob = (int64_t) lay.max_blob;
    disp_.cache_slot_off = n_slots_ > 0 ? cache_.slot_offsets() : nullptr;
    disp_.host_res = res_.data();
    disp_.plan = win_.sink();
    disp_.pcie_num = 0;
    if (opt_.adapt_every > 0 && opt_.adapt_swaps > 0) {
        disp_.usage.assign((size_t) (L * E), 0.0f);
        // the counts from the rows a window keeps (after_rows), as Qwen3.8-Flash-Next's: the trunk's routing kept
        // here; the MTP layer's, run once a draft, counted as it runs
        disp_.usage_by_window = true;
        disp_.route_rec_layers = s.n_moe_layers();
        disp_.route_rec_t = opt_.max_t;
        disp_.route_rec.assign((size_t) (s.n_moe_layers() * opt_.max_t * s.n_expert_used), -1);
        prompt_routed_.assign((size_t) (L * E), 0);
    }
    trunk_layers_.assign(moe_local_.begin(), moe_local_.begin() + moe_trunk_);
    // (also after the RAM copy of what no VRAM tier holds is pinned, and on a later stage once the root's RAM is
    // registered on its GPU: finish_experts)
    if (in_ram_ && root_ == this) {
        // the later stages' GPUs first: the PCIe share's move into host USM (make_kernel_readable) keeps the mapping
        // where one of them is of another runtime, which could not register the moved memory (finish_experts)
        for (size_t i = 1; i < split_devs_.size() && !opt_.split_at.empty() && i <= opt_.split_at.size(); ++i) {
            std::string why;
            if (src_->register_on(split_devs_[i], why)) src_->hold_registration(split_devs_[i]);
            // (a refusal is said once the stage is loaded)
        }
        pinned_split();
    }
    // the adaptive tier's copy stream: its rounds bring experts in beside the windows (adapt)
    swap_cs_ = strata::gpu::stream_create();
    if (swap_cs_ == nullptr || !strata::gpu::event_create(&swap_ev_)) {
        err = "plain model: the adaptive tier's copy stream";
        return false;
    }
    // the tier from a profile, after the split's timing (which needs experts in RAM outside the VRAM): its ranking,
    // each of this stage's layers up to its slots.  Without a profile each layer's tier takes its first experts (the
    // adaptive tier moves them), so that the RAM copy below leaves out what VRAM holds; with the experts in RAM or read
    // in place, the round after the first prompt fills the free slots with those it routed to instead - but not when
    // no round comes (--adapt-every 0: the tier stayed empty and the CPU computed every expert, 59.1 tok/s against
    // 143.5 for Qwen3.6 UD-Q4_K_XL wholly in the B70's VRAM, 2026-11-01), nor in a layer whose slots take all its
    // experts: there is no choice to wait for, and the rounds would fill 10,496 slots 2,048 and then 96 at a time.
    std::vector<std::pair<int64_t, int32_t>> want;
    const bool rounds = opt_.adapt_every > 0 && opt_.adapt_swaps > 0;
    if (!opt_.profile.empty()) {   // prior_ read above, where it sized the layers' slots
        for (const auto& [l, e] : prior_) want.emplace_back(l, e);
    } else {
        for (int64_t l : moe_local_)
            if (!((in_ram_ || opt_.mmap_experts) && rounds && (int64_t) free_[(size_t) l].size() < E))
                for (int32_t e = 0; e < (int32_t) E; ++e) want.emplace_back(l, e);
    }
    int64_t filled = 0;
    if (!fill_tier(want, filled, err)) return false;
    if (!opt_.profile.empty())
        std::fprintf(stderr, "xestrata plain: the VRAM tier filled from %s: %lld experts\n", opt_.profile.c_str(),
                     (long long) filled);
    // the slots the prompt path lends for its longest chunk (prompt_borrow): their experts are kept in RAM too
    lend_from_ = -1;
    if (opt_.prompt_chunk > 0 && prompt_dev_.at.empty() && n_slots_ > 1) {
        uint64_t db = 0, hb = 0;
        prompt_carve(opt_.prompt_chunk, pb_, nullptr, nullptr, db, hb);
        const uint64_t* off = cache_.slot_offsets();
        const int64_t lowest = lend_lowest(n_slots_);
        lend_from_ = n_slots_;
        while (off != nullptr && lend_from_ > lowest && off[n_slots_] - off[lend_from_] < db) --lend_from_;
    }
    if (!prompt_dev_.at.empty())
        std::fprintf(stderr, "xestrata plain: the prompt's buffers beside the tier, chunks of up to %lld tokens\n",
                     (long long) prompt_dev_T_);
    free_b_ = free_b;
    return true;
}

// qwen35's part of the load in place of open_experts: every weight is in VRAM already (load_qwen uploaded them), and
// the prompt path's buffers of their own (prompt_dev_) take the longest chunk up to --prompt-chunk that fits beside
// them.  No longer chunk is chosen by itself, as the MoE models' auto_prompt_chunk does: a chunk streams no experts
// here, so a longer one saves only launches, and the VRAM stays for the picture reader and the image engine.
bool PlainModel::open_dense(std::string& err) {
    n_moe_ = 0;
    moe_local_.clear();
    moe_trunk_ = 0;
    level_zero_ = Runtime::get().device().get_backend() == sycl::backend::ext_oneapi_level_zero;
    uint64_t free_b = 0;
    {
        size_t f = 0, t = 0;
        if (!strata::gpu::mem_info(&f, &t)) { err = "plain model: the device's memory is unknown"; return false; }
        free_b = f;
        total_b_ = t;
        free_b = counted_free(free_b, free_b, opt_.vram_free_mib);   // (mem_info applies STRATA_VRAM_LIMIT_MIB)
    }
    // what is allocated after (the products' libraries, the graphs) and the driver's margin, as open_experts leaves
    const uint64_t reserve = opt_.reserve_mib >= 0 ? (uint64_t) opt_.reserve_mib << 20
                                                   : core::vram_room(total_b_, /*parts*/ 1, kTierRoomLeast);
    const uint64_t room = free_b > reserve ? free_b - reserve : 0;
    if (opt_.prompt_chunk > 0) {
        const int64_t asked = opt_.prompt_chunk;
        for (int64_t T = asked; T >= 16; T /= 2) {
            if (!alloc_prompt_dev(T, room)) continue;
            uint64_t db = 0, hb = 0;
            prompt_carve(T, pb_, nullptr, nullptr, db, hb);
            if (strata::gpu::alloc_host(&prompt_host_, hb)) {
                // an encoder's are its own, freed between prompts (encoder_release) and made again for the next
                if (!opt_.encoder)
                    for (uint8_t* p : prompt_dev_.at) allocs_.push_back(p);
                opt_.prompt_chunk = T;
                enc_host_T_ = T;
                prompt_carve(T, pb_, nullptr, (uint8_t*) prompt_host_, db, hb);
                break;
            }
            for (uint8_t* p : prompt_dev_.at) strata::gpu::free(p);
            prompt_dev_ = PromptPieces();
            prompt_host_ = nullptr;
        }
        // a layer split's: the rows between the stages
        if (!prompt_dev_.at.empty() && !(first() && last()) &&
            !strata::gpu::alloc_host(&hand_host_, (size_t) (asked * hand_w_) * 4)) {
            strata::gpu::free(prompt_host_);
            prompt_host_ = nullptr;
        }
        if (prompt_host_ == nullptr) {
            std::fprintf(stderr, "xestrata plain: no memory for the prompt path's chunks (%lld tokens asked): prompts are "
                                 "read through the decode windows\n", (long long) asked);
            opt_.prompt_chunk = 0;
        } else if (opt_.prompt_chunk != asked) {
            std::fprintf(stderr, "xestrata plain: the prompt path's chunks are %lld tokens (the memory for %lld was not "
                                 "there)\n", (long long) opt_.prompt_chunk, (long long) asked);
        }
    }
    free_b_ = free_b;
    return true;
}

// The prompt path's lending, as Qwen3.8-Flash-Next's --prefill auto (generate.cpp's plan_lend): a chunk's buffers
// borrow at most prefill::lend_pct of the tier's slots and leave 128 (half of a smaller tier), and only from its last
// piece.  85%: Qwen3.6 UD-Q4_K_XL's 32K prompt on the B70 (2026-10-26) read at 711 tok/s with the half the tier lent
// until then (2048-token chunks in 8 GB), 802 at 70-90% (4096); in 16 GB 855 in 4096-token chunks, 897 in 8192 and
// 917 in 16384 (85%: half the tier held 8192).
constexpr int64_t kPlainLendPct = 85;

int64_t PlainModel::lend_lowest(int64_t n) const {
    const int64_t pct = std::clamp<int64_t>(strata::prefill::lend_pct(kPlainLendPct), 1, 99);
    return std::max({n - n * pct / 100, std::min<int64_t>(128, n / 2), cache_.last_piece_first()});
}

// --prompt-chunk not given: the largest of prefill.hpp's kAutoChunks (as Qwen3.8's --prefill auto) whose buffers the
// tier lends (lend_lowest), or that fit in the `beside` bytes left beside a tier that holds every expert; the pinned
// host buffers made again for it.  One GPU: a layer split's stages share the rows' buffers of the chunk asked.
void PlainModel::auto_prompt_chunk(uint64_t beside) {
    const uint64_t* off = cache_.slot_offsets();
    const int64_t n = n_slots_;
    const int64_t lowest = lend_lowest(n);
    const int64_t c = strata::prefill::auto_chunk(opt_.max_cells, [&](int64_t t) {
        uint64_t db = 0, hb = 0;
        prompt_carve(t, pb_, nullptr, nullptr, db, hb);
        if (beside > 0) return db <= beside && db <= max_alloc_bytes();
        return off != nullptr && n >= 2 && off[n] - off[lowest] >= db;
    });
    // only upward: a chunk below the one asked kept the old way (prompt_borrow halves it until it fits) read faster.
    // The RX 9060 XT, llm-jp-4.1 16K at 8 GiB of VRAM and 20 GiB of RAM (2026-10-26): the chunk chosen here 2048, 310
    // tok/s; 4096 asked (borrowed at prompt time) 352, at either lending share
    if (c <= opt_.prompt_chunk) return;
    uint64_t db = 0, hb = 0;
    prompt_carve(c, pb_, nullptr, nullptr, db, hb);
    void* h = nullptr;
    if (!strata::gpu::alloc_host(&h, hb)) {
        std::fprintf(stderr, "xestrata plain: prompt chunks of %lld tokens would fit the tier, but not their %llu MiB of "
                             "pinned memory: %lld\n", (long long) c, (unsigned long long) (hb >> 20),
                     (long long) opt_.prompt_chunk);
        return;
    }
    strata::gpu::free(prompt_host_);
    prompt_host_ = h;
    opt_.prompt_chunk = c;
    prompt_carve(c, pb_, nullptr, (uint8_t*) prompt_host_, db, hb);
    std::fprintf(stderr, "xestrata plain: prompt chunks of %lld tokens (the largest the tier lends, --prompt-chunk sets "
                         "it)\n", (long long) c);
}

// The PCIe split of the missed experts between the CPU and PCIe (after Project Maya's CPU lane): the times measured
// now, while no expert is in VRAM; only from pinned RAM.
void PlainModel::pinned_split() {
    // WSL 2: an RTX 2050 there failed at this point with CUDA error 700 (an illegal address) while the experts' RAM
    // was read by the GPU's kernels; --no-pcie-split started and answered.  The cause is not known, so under WSL 2 the
    // split is off unless STRATA_WSL_PCIE_SPLIT=1 asks for it.
    static const bool wsl_split = [] {
        const char* v = strata::env_get("WSL_PCIE_SPLIT");
        return v != nullptr && std::strtol(v, nullptr, 10) != 0;
    }();
    if (moe_local_.empty()) return;
    const char* keep = !opt_.pcie_split ? "--no-pcie-split"
                       : strata::platform::under_wsl() && !wsl_split
                           ? "under WSL 2: the GPU reading the experts' RAM failed there; XESTRATA_WSL_PCIE_SPLIT=1 tries it"
                           : nullptr;
    const bool fixed = opt_.pcie_frac >= 0.0 && opt_.pcie_split;
    if (fixed) keep = "--pcie-frac is given";
    const bool measured = setup_pcie_split(disp_, n_moe_, split_, false, keep, fixed ? opt_.pcie_frac : 0.0,
                                           "xestrata plain");   // overlap: after the hits
    if (fixed) {   // --pcie-frac: that share in every window, read where the GPU can read it
        disp_.pcie_split = nullptr;
        disp_.pcie_num = (int) std::lround(std::clamp(opt_.pcie_frac, 0.0, 1.0) * 256.0);
        pcie_probe_ = -1;
        std::string why;
        if (disp_.pcie_num > 0 && !src_->make_kernel_readable(why)) {
            std::fprintf(stderr, "xestrata plain: the experts stay registered for copies (%s): the CPU takes every "
                                 "missed expert\n", why.c_str());
            disp_.pcie_num = 0;
        }
        return;
    }
    if (!measured) return;
    // the decode's PCIe share (its size from the times: pcie_split_m): chosen by the first windows (pcie_probe), which
    // Qwen3.8-Flash-Next's engine does not do; STRATA_PLAIN_DECODE_PCIE=1 / 0: on / off
    if (const char* v = strata::env_get("PLAIN_DECODE_PCIE")) {
        disp_.pcie_num = std::strtol(v, nullptr, 10) != 0 ? 256 : 0;
    } else {
        disp_.pcie_num = 256;
        pcie_probe_ = 0;
    }
    // The arena moves where the GPU can read it (Level Zero's registration is for copies) here, when the split's
    // table reads any expert over PCIe - not after the first prompt by its routing, as generate.cpp's
    // (pcie_move_if_pays): there the move's seconds went into the first answer's wait, the B70's prompt of llm-jp-4.1
    // 0.5-0.8 -> 8.5 s and of Qwen3.6 0.6 -> 5.8 s (both moved: 7.5 and 5.2 s), the load as much shorter (2026-10-26).
    bool any = false;
    for (int f = 1; f <= 16 && !any; ++f) any = pcie_split_m(split_, false, f, f) > 0;
    std::string why;
    if (any && !src_->pcie_layer(split_.layer) && !src_->make_kernel_readable(why)) {
        std::fprintf(stderr, "xestrata plain: the experts stay registered for copies (%s): the CPU takes every "
                             "missed expert\n", why.c_str());
        disp_.pcie_split = nullptr;
        disp_.pcie_num = 0;
        pcie_probe_ = -1;
    }
}

// After every stage's tier: the experts that do not all fit in RAM, the ones no VRAM tier holds pinned there, as far
// as the RAM allows, the rest read in place (generate.cpp's resident RAM mode), so that a model which fits the RAM and
// the VRAM together reads none from the drive.  On Level Zero the RAM beside them counts half the VRAM: the devices'
// allocations are made by then, and on the B70 GLM-5.3-Flash ran with 88.5 GiB pinned here and 26 GiB left
// (2026-10-19).
// --expert-cache-device1..3: each helper fills from the profile's pairs, then the others expert by expert across the
// layers, none any stage's tier holds (remote_placement), and computes the rows routed to its experts in every
// stage's windows; a dense model, or one without helpers, has none
// The free slots of each layer take `want`'s pairs in order (those the layer has none for, or holds, are passed by),
// read on several threads, each through pinned memory of its own: one at a time, GLM-5.3-Flash's 2,646 experts
// (22.9 GB) took 19.5 s from the drive, 1.2 GB/s (B70, 990 Pro, 2026-10-19).  `filled` gets how many.
bool PlainModel::fill_tier(const std::vector<std::pair<int64_t, int32_t>>& want, int64_t& filled, std::string& err) {
    const int64_t E = spec_.n_expert;
    const auto& lay = k::cpu::expert_layout();
    struct Fill { int64_t l; int32_t e, slot; };
    std::vector<Fill> todo;
    for (const auto& [l, e] : want) {
        if (l < 0 || (size_t) l >= free_.size() || e < 0 || e >= E) continue;
        auto& f = free_[(size_t) l];
        if (f.empty() || res_[(size_t) (l * E + e)] >= 0) continue;
        todo.push_back({l, e, f.back()});
        f.pop_back();
    }
    filled = (int64_t) todo.size();
    if (todo.empty()) return true;
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    auto work = [&]() {
        const OnDevice on(dev_);   // a thread's GPU is the first until it says otherwise
        uint8_t* buf = nullptr;
        if (!strata::gpu::alloc_host(&buf, (size_t) lay.max_blob)) { bad = true; return; }
        for (size_t i; !bad && (i = next.fetch_add(1)) < todo.size();) {
            const Fill& f = todo[i];
            if (!src_->copy_blob(f.l, f.e, buf) ||
                !strata::gpu::copy(cache_.device_slot(f.slot), buf, (size_t) lay.blob_bytes(f.l)))
                bad = true;
        }
        strata::gpu::free(buf);
    };
    {
        const int threads = (int) std::clamp<size_t>(todo.size() / 64, 1, 8);
        std::vector<std::thread> pool;
        for (int i = 1; i < threads; ++i) pool.emplace_back(work);
        work();
        for (auto& t : pool) t.join();
    }
    if (bad) {
        err = "plain model: an expert could not be read for the VRAM tier";
        return false;
    }
    for (const Fill& f : todo) res_[(size_t) (f.l * E + f.e)] = f.slot;
    return true;
}

bool PlainModel::open_helpers(std::string& err) {
    if (dense_ || std::all_of(helper_dev_.begin(), helper_dev_.end(), [](int d) { return d < 0; })) return true;
    const int64_t L = n_moe_, E = spec_.n_expert;
    std::vector<uint8_t> claimed((size_t) (L * E), 0);
    for (PlainModel* st : chain())
        for (size_t i = 0; i < st->res_.size() && i < claimed.size(); ++i)
            if (st->res_[i] >= 0) claimed[i] = 1;
    std::array<int, 3> caps{};
    for (int r = 0; r < 3; ++r) caps[(size_t) r] = helper_dev_[(size_t) r] >= 0 ? opt_.expert_cache_remote[(size_t) r] : 0;
    const auto by = remote_placement(remote_ranking(prior_, L, E), L, E, caps,
                                     opt_.expert_cache_remote_placement == "layer",
                                     [&](int32_t l, int32_t e) { return claimed[(size_t) (l * E + e)] != 0; });
    for (int r = 0; r < 3; ++r) {
        if (helper_dev_[(size_t) r] < 0) continue;
        auto h = std::make_unique<RemoteExperts>();
        if (!h->open(helper_dev_[(size_t) r], caps[(size_t) r], L, E, by[(size_t) r], cache_, *src_, claimed, err)) {
            err.insert(0, "plain model: ");
            return false;
        }
        std::fprintf(stderr, "xestrata plain: helper GPU %d holds %lld more experts (%.2f GiB), placed %s\n",
                     helper_dev_[(size_t) r], (long long) h->resident(), h->gib(),
                     opt_.expert_cache_remote_placement.c_str());
        for (PlainModel* st : chain())
            if (st->disp_.remote_count < 3) st->disp_.remote[st->disp_.remote_count++] = h.get();
        helpers_[(size_t) r] = std::move(h);
    }
    return true;
}

void PlainModel::finish_experts() {
    if (dense_) {   // no experts: every stage's weights in its VRAM
        for (PlainModel* st : chain())
            std::fprintf(stderr, "xestrata plain: a dense model, every weight in VRAM, %lld MiB free after it, prompt "
                                 "chunks of %lld tokens\n", (long long) (st->free_b_ >> 20), (long long) st->opt_.prompt_chunk);
        return;
    }
    const int64_t E = spec_.n_expert;
    if (!in_ram_ && (!opt_.mmap_experts || opt_.resident_gib != 0)) {
        auto* file = static_cast<FileExpertSource*>(src_);
        std::vector<std::pair<int32_t, int32_t>> held, others;
        std::vector<int32_t> slots;
        uint64_t headroom = 8ull << 30;
        for (PlainModel* st : chain()) {
            for (int64_t l : st->moe_local_)
                for (int64_t e = 0; e < E; ++e)
                    if (const int32_t r = st->res_[(size_t) (l * E + e)]; r >= 0) {
                        if (st == this) {
                            held.emplace_back((int32_t) l, (int32_t) e);
                            slots.push_back(r);
                        } else {
                            others.emplace_back((int32_t) l, (int32_t) e);
                        }
                    }
            if (st->level_zero_) headroom += st->total_b_ / 2;
        }
        // and a fifth of the RAM (up to 16 GiB) on a PC whose GPUs are not Level Zero ones: GLM-5.3-Flash UD-IQ1_S's
        // 53 GiB copy beside 8 GiB on a 62.7 GiB PC went into swap and the GPU's next wait never returned (RX 9060 XT,
        // ROCm 7.14, five starts, 2026-10-20); with 12 GiB more it ran (41 GiB copied, decode 2.6 tok/s).  Why the swap
        // with 8 GiB left is not known: dropping the files' cache before and during the copy changed nothing.
        if (!level_zero_)
            headroom += std::min<uint64_t>(strata::platform::total_physical_memory() / 5, 16ull << 30);
        headroom = detail::ram_headroom(headroom);   // the RAM the user asked to leave free (--ram-reserve-mib)
        // the order the RAM takes them in: the profile's, else each layer's experts in turn
        std::vector<std::pair<int32_t, int32_t>> order = prior_;
        if (order.empty())
            for (int32_t e = 0; e < (int32_t) E; ++e)
                for (int32_t l = 0; l < (int32_t) n_moe_; ++l) order.emplace_back(l, e);
        // STRATA_RESIDENT_PIN=0: never page-locked, 1: always; else as the RAM allows (#11), as generate.cpp
        using Pin = FileExpertSource::Pin;
        const char* pin_env = strata::env_get("RESIDENT_PIN");
        const Pin pin = opt_.resident_pageable || (pin_env != nullptr && std::strcmp(pin_env, "0") == 0) ? Pin::off
                        : pin_env != nullptr && std::strcmp(pin_env, "1") == 0                          ? Pin::on
                                                                                                         : Pin::automatic;
        std::string why;
        // --resident-budget-gib: at most that much, still within what the RAM allows
        const uint64_t budget = opt_.resident_gib > 0 ? (uint64_t) (opt_.resident_gib * 1073741824.0)
                                                      : FileExpertSource::kResidentWhatFits;
        if (!file->pin_complement(n_slots_, held, slots, why, pin, others, lend_from_, headroom, budget, &order)) {
            std::fprintf(stderr, "xestrata plain: the experts outside the VRAM tier do not fit in RAM (%s): read in "
                                 "place\n", why.c_str());
        } else {
            if (opt_.adapt_every > 0 && opt_.adapt_swaps > 0 && !file->reserve_exchanges(opt_.adapt_swaps, why))
                std::fprintf(stderr, "xestrata plain: no exchange buffers (%s): an expert the adaptive tier evicts is "
                                     "read in place after\n", why.c_str());
            std::fprintf(stderr, "xestrata plain: the experts outside the VRAM tier in RAM: %.2f GiB (%lld experts%s)\n",
                         (double) file->resident_bytes() / 1073741824.0, (long long) file->resident_count(),
                         file->complement_pinned() ? ", pinned" : "");
            for (PlainModel* st : chain()) st->complement_ = file;
        }
    }
    // a later stage's GPU copies from the root's RAM copy as from its own pinned memory once it is registered there.
    // Where it cannot be (one maker's pinned host memory on another's GPU: the B70's on the RTX 4070), each copy is a
    // pageable one that the host waits for, and that stage's tier keeps what the profile filled: its adaptive rounds
    // took GLM-5.3-Flash's decode from 14.8 to 12.6 tok/s (B70 + 4070, 2026-10-20)
    for (auto& st : next_) {
        std::string why;
        if (!src_->register_on(st->dev_, why)) {
            // no round fills it then: without a profile, its free slots take each layer's first experts now (an empty
            // tier left the RTX 4070's 21 layers of Qwen3.6-35B-A3B to the CPU: 72.1 against 188.8 tok/s, 2026-11-01)
            st->disp_.usage.clear();
            std::vector<std::pair<int64_t, int32_t>> want;
            for (int64_t l : st->moe_local_)
                for (int32_t e = 0; e < (int32_t) E; ++e) want.emplace_back(l, e);
            int64_t filled = 0;
            std::string ferr;
            const OnDevice on(st->dev_);
            const bool ok = st->fill_tier(want, filled, ferr);
            std::fprintf(stderr, "xestrata plain: %s: %s's copies from RAM are pageable, its VRAM tier stays as it is "
                                 "filled now (%lld experts%s)\n", why.c_str(), strata::gpu::device_name(st->dev_).c_str(),
                         (long long) filled, ok ? "" : ("; " + ferr).c_str());
        }
    }
    for (PlainModel* st : chain()) {   // the GPU reads its share of the misses from the pinned RAM
        const OnDevice on(st->dev_);
        if ((st->complement_ != nullptr && st->complement_->complement_pinned()) || (in_ram_ && st != this))
            st->pinned_split();
    }
    for (PlainModel* st : chain()) {
        const OnDevice on(st->dev_);
        std::fprintf(stderr, "xestrata plain: %lld MoE layers x %lld experts, %lld VRAM slots (%lld a layer, %.2f GiB), "
                             "%d CPU workers, %lld MiB free before the tier\n",
                     (long long) st->moe_local_.size(), (long long) E, (long long) st->n_slots_,
                     (long long) (st->moe_local_.empty() ? 0 : st->n_slots_ / (int64_t) st->moe_local_.size()),
                     (double) st->cache_.bytes() / 1073741824.0, st->pool_->workers(), (long long) (st->free_b_ >> 20));
    }
}

int64_t PlainModel::resident() const {
    int64_t n = 0;
    for (int32_t r : res_) n += r >= 0;
    for (const auto& st : next_) n += st->resident();
    return n;
}

std::vector<int> PlainModel::pool_worker_cpus() const {
    if (pool_ != nullptr) return pool_->worker_cpus();
    if (cpu_ != nullptr) return cpu_->worker_cpus();
    if (cpu_tail_ != nullptr) return cpu_tail_->worker_cpus();
    return {};
}

const char* PlainModel::experts_ram() const {
    return dense_ ? "none" : in_ram_ ? "arena" : complement_ != nullptr && complement_->resident_count() > 0 ? "resident" : "none";
}

uint64_t PlainModel::experts_ram_bytes() const {
    if (dense_) return 0;
    if (in_ram_) return (uint64_t) k::cpu::expert_layout().total;
    return complement_ != nullptr ? (uint64_t) complement_->resident_bytes() : 0;
}

int64_t PlainModel::slots() const {
    int64_t n = cache_.slots();   // the live ones: --vram-elastic's shrinks give the tail back
    for (const auto& st : next_) n += st->slots();
    return n;
}

int64_t PlainModel::hits() const {
    int64_t n = disp_.cache_hits;
    for (const auto& st : next_) n += st->hits();
    return n;
}

int64_t PlainModel::misses() const {
    int64_t n = disp_.cache_refused;
    for (const auto& st : next_) n += st->misses();
    return n;
}

int64_t PlainModel::swaps() const {
    int64_t n = swaps_total_;
    for (const auto& st : next_) n += st->swaps();
    return n;
}

PlainModel::KvPtrs PlainModel::kv_ptrs(size_t a, int slot) const {
    KvPtrs p;
    if (slot >= 0) {
        const SlotSession& ss = bslots_[(size_t) slot];
        p = {ss.k_pool[a], ss.v_pool[a], ss.k_scale[a], ss.v_scale[a], ss.k_q[a], ss.v_q[a], ss.k_q4[a], ss.v_q4[a]};
    } else {
        p = {k_pool_[a], v_pool_[a], k_scale_[a], v_scale_[a], k_q_[a], v_q_[a], k_q4_[a], v_q4_[a]};
    }
    return p;
}

k::QsaAttnPools PlainModel::kv_pools(size_t a, int slot) const {
    const KvPtrs p = kv_ptrs(a, slot);
    k::QsaAttnPools pools;
    pools.k_pool = p.k_pool;
    pools.v_pool = p.v_pool;
    pools.k_q = p.k_q;
    pools.v_q = p.v_q;
    pools.k_scale = p.k_scale;
    pools.v_scale = p.v_scale;
    pools.k_q4 = p.k_q4;
    pools.v_q4 = p.v_q4;
    pools.page_table = table_of(a);   // the identity (a ring's for a sliding-window layer): a slot's pools have the
                                      // live session's shape
    return pools;
}

std::vector<PlainModel::StatePart> PlainModel::kv_planes(size_t a, int slot, size_t cells) const {
    const KvPtrs p = kv_ptrs(a, slot);
    const size_t HD = (size_t) attn_shapes().head_dim;
    const size_t codes = cells * HD, scales = cells * (HD / k::KV_Q8_GROUP) * 2,
                 q4 = cells * (size_t) k::kv_q4_bytes_per_head((int) HD);
    std::vector<StatePart> v;
    if (p.k_pool != nullptr) v = {{p.k_pool, codes * 2, 2}, {p.v_pool, codes * 2, 2}};
    else if (p.k_q4 != nullptr) v = {{p.k_q4, q4}, {p.v_q4, q4}};
    else if (p.v_q4 != nullptr) v = {{p.k_q, codes}, {p.k_scale, scales}, {p.v_q4, q4}};
    else if (p.k_q != nullptr) v = {{p.k_q, codes}, {p.v_q, codes}, {p.k_scale, scales}, {p.v_scale, scales}};
    return v;
}

bool PlainModel::alloc_kv(KvPtrs& p, size_t cells, bool vmm) {
    const size_t HD = (size_t) attn_shapes().head_dim;
    const size_t codes = cells * HD, scales = cells * (HD / k::KV_Q8_GROUP) * 2,
                 q4 = cells * (size_t) k::kv_q4_bytes_per_head((int) HD);
    bool ok = true;
    auto take = [&](auto*& q, size_t bytes) {
        if (!ok) return false;
        if (vmm) {   // --kv-grow: addresses for every cell, memory mapped later (kv_map)
            auto r = std::make_unique<VmmRange>();
            if (!r->reserve(bytes)) return ok = false;
            q = (std::remove_reference_t<decltype(q)>) r->base();
            kv_vmm_page_bytes_.push_back(bytes / (size_t) n_pages_);
            kv_vmm_.push_back(std::move(r));
            return true;
        }
        if (!strata::gpu::alloc_device(&q, bytes) || !strata::gpu::memset(q, 0, bytes)) return ok = false;
        allocs_.push_back(q);
        return true;
    };
    if (opt_.kv_q4 == 1) {
        take(p.k_q4, q4);
        take(p.v_q4, q4);
    } else if (opt_.kv_q4 == 2) {
        take(p.k_q, codes);
        take(p.k_scale, scales);
        take(p.v_q4, q4);
    } else if (opt_.kv_int8) {
        take(p.k_q, codes);
        take(p.v_q, codes);
        take(p.k_scale, scales);
        take(p.v_scale, scales);
    } else {
        take(p.k_pool, codes * 2);
        take(p.v_pool, codes * 2);
    }
    return ok;
}

bool PlainModel::kv_map(int64_t cells, std::string& err) {
    const k::QsaShapes qs = attn_shapes();
    const uint64_t G = vmm_granularity();
    const int64_t pages = (cells + qs.page_size - 1) / qs.page_size;
    for (size_t i = 0; i < kv_vmm_.size(); ++i) {
        VmmRange& r = *kv_vmm_[i];
        int64_t lo = 0;
        while (lo < r.chunks() && r.mapped(lo)) ++lo;
        const int64_t hi = std::min<int64_t>(r.chunks(), (int64_t) ((pages * kv_vmm_page_bytes_[i] + G - 1) / G));
        if (hi <= lo) continue;
        if (!kv_elastic_map(r, lo, hi, [] { return (VmmChunk) 0; })) {
            err = std::string("plain model: no VRAM for the K/V's ") + std::to_string((long long) cells) + " cells (" +
                  strata::gpu::last_error() + ")";
            return false;
        }
    }
    if (!strata::gpu::device_sync()) { err = std::string("plain model: ") + strata::gpu::last_error(); return false; }
    kv_cells_ = std::max(kv_cells_, std::min(cells, pages * qs.page_size));
    return true;
}

// --kv-grow's steps (generate.cpp's): the K/V grows by whole 8,192 cells, and gives back what lies two steps past a
// request's need
constexpr int64_t kKvGrowStep = 8192;

bool PlainModel::kv_grow_to(int64_t cells, std::string& err) {
    if (kv_cells_ < 0 || cells <= kv_cells_) return true;
    const int64_t target = std::min<int64_t>(opt_.max_cells, kv_elastic_target(cells, kKvGrowStep));
    const k::QsaShapes qs = attn_shapes();
    const uint64_t G = vmm_granularity();
    const int64_t pages = (target + qs.page_size - 1) / qs.page_size;
    uint64_t need = 0;
    for (size_t i = 0; i < kv_vmm_.size(); ++i) {
        const int64_t hi = std::min<int64_t>(kv_vmm_[i]->chunks(), (int64_t) ((pages * kv_vmm_page_bytes_[i] + G - 1) / G));
        need += (uint64_t) std::max<int64_t>(0, hi - kv_vmm_[i]->mapped_count()) * G;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t slots0 = cache_.slots();
    if (cache_.segmented()) {   // the tier gives up its last segments for them, beside its usual reserve
        const int64_t reserve = opt_.reserve_mib >= 0 ? opt_.reserve_mib
                                                       : (int64_t) (core::vram_room(total_b_, 1, kTierRoomLeast) >> 20);
        std::string answer;
        if (!vram(reserve + (int64_t) (need >> 20) + 1, answer, err)) return false;
    }
    if (!kv_map(target, err)) return false;
    std::fprintf(stderr, "xestrata plain: the K/V grown to %lld cells (%.2f GiB more); the expert tier %lld -> %lld slots, "
                         "in %.0f ms\n", (long long) kv_cells_, (double) need / 1073741824.0, (long long) slots0,
                 (long long) cache_.slots(),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

bool PlainModel::kv_trim(int64_t cells, int64_t hold, std::string& err) {
    if (kv_cells_ < 0) return true;
    const int64_t target = kv_elastic_target(std::max<int64_t>(cells, hold), kKvGrowStep);
    if (kv_cells_ < target + 2 * kKvGrowStep) return true;
    if (!settle_swaps(true, err)) return false;
    if (!strata::gpu::device_sync()) { err = std::string("plain model: ") + strata::gpu::last_error(); return false; }
    const k::QsaShapes qs = attn_shapes();
    const uint64_t G = vmm_granularity();
    const int64_t pages = (target + qs.page_size - 1) / qs.page_size;
    for (size_t i = 0; i < kv_vmm_.size(); ++i) {
        VmmRange& r = *kv_vmm_[i];
        const int64_t keep = (int64_t) ((pages * kv_vmm_page_bytes_[i] + G - 1) / G);
        kv_elastic_unmap(r, keep, r.chunks(), [](VmmChunk h) { vmm_chunk_free(h); });
    }
    const int64_t from = kv_cells_, slots0 = cache_.slots();
    kv_cells_ = pages * qs.page_size;
    std::string answer;
    if (cache_.segmented() && !vram(-1, answer, err)) return false;   // the tier takes the VRAM back
    std::fprintf(stderr, "xestrata plain: the K/V trimmed from %lld to %lld cells; the expert tier %lld -> %lld slots\n",
                 (long long) from, (long long) kv_cells_, (long long) slots0, (long long) cache_.slots());
    return true;
}

void PlainModel::kv_append_cells(size_t a, int slot, const int32_t* steps, int n, const float* kc, const float* vc,
                                 strata::gpu::Stream cs) {
    const KvPtrs p = kv_ptrs(a, slot);
    const k::QsaShapes qs = attn_shapes();
    const int row = (int) (qs.n_head_kv * qs.head_dim);
    if (p.k_q4 != nullptr) {
        k::kv_append_q4_steps(p.k_q4, p.v_q4, table_of(a), steps, k::kStepCount, n, kc, vc, qs, cs);
    } else if (p.v_q4 != nullptr) {   // K8V4: K's INT8 codes and scales (one plane), V's Q4_0 (one plane)
        k::kv_append_q8_steps(p.k_q, p.k_q, p.k_scale, p.k_scale, table_of(a), steps, k::kStepCount, kc, kc, row, n, qs,
                              cs);
        k::kv_append_q4_steps(p.v_q4, p.v_q4, table_of(a), steps, k::kStepCount, n, vc, vc, qs, cs);
    } else if (n == 1) {   // a window's row (the one-cell kernels the windows always took)
        if (p.k_q != nullptr)
            k::kv_append_q8_step(p.k_q, p.v_q, p.k_scale, p.v_scale, table_of(a), steps, kc, vc, qs, cs);
        else
            k::kv_append_step(p.k_pool, p.v_pool, table_of(a), steps, kc, vc, qs, cs);
    } else if (p.k_q != nullptr) {
        k::kv_append_q8_steps(p.k_q, p.v_q, p.k_scale, p.v_scale, table_of(a), steps, k::kStepCount, kc, vc, row, n, qs,
                              cs);
    } else {
        k::kv_append_steps(p.k_pool, p.v_pool, table_of(a), steps, k::kStepCount, kc, vc, row, n, qs, cs);
    }
}

// #533, as generate.cpp's VRAM command: a shrink gives back the cache's last segments, the experts in their slots
// become misses (the CPU's or PCIe's, as any expert outside the tier) and the slots leave the layers' free lists; the
// prompt's loan takes the end of what is left (prompt_borrow).  A grow maps them again and puts each slot's expert
// back, or, when the adaptive tier already brought that one back, the layer's most-routed missing expert.
bool PlainModel::arena_drop(const std::string& line, std::string& answer, std::string& err) {
    auto* arena = dynamic_cast<ArenaExpertSource*>(src_own_.get());
    if (arena == nullptr) { err = "the experts are not in an arena"; return false; }
    return arena_drop_command(*arena, line, prior_, answer, err);
}

bool PlainModel::vram(int64_t reserve_mib, std::string& answer, std::string& err) {
    if (!cache_.segmented()) {
        err = "VRAM needs an engine started with --vram-elastic (--serve, one GPU with virtual memory)";
        return false;
    }
    if (reserve_mib < 0)
        reserve_mib = opt_.reserve_mib >= 0 ? opt_.reserve_mib
                                            : (int64_t) (core::vram_room(total_b_, 1, kTierRoomLeast) >> 20);
    if (!settle_swaps(true, err)) return false;   // the adaptive tier's copies land first
    if (!strata::gpu::device_sync()) { err = std::string("VRAM: ") + strata::gpu::last_error(); return false; }
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t E = spec_.n_expert, LS = (int64_t) moe_local_.size();
    size_t free_b = 0, total_b = 0;
    strata::gpu::mem_info(&free_b, &total_b);
    const int64_t want_free = reserve_mib << 20, mapped = cache_.mapped_bytes(), before = cache_.slots();
    // what stays at least: two slots a layer, so the prompt's loan has room (it halves its chunk down to 16 tokens)
    const int64_t floor_slots = std::min<int64_t>(cache_.full_slots(), 2 * LS);
    std::string note;
    if ((int64_t) free_b < want_free) {
        int64_t keep = mapped - (want_free - (int64_t) free_b);
        if (keep < cache_.bytes_of(floor_slots)) {
            keep = cache_.bytes_of(floor_slots);
            note = " (the tier keeps its smallest size)";
        }
        if (!cache_.shrink(keep, err)) return false;
        const int64_t live = cache_.slots();
        for (size_t i = 0; i < res_.size(); ++i)
            if (res_[i] >= live) {
                vram_evicted_.emplace_back((int32_t) i, res_[i]);
                res_[i] = -1;
            }
        for (auto& f : free_) f.erase(std::remove_if(f.begin(), f.end(), [&](int32_t s) { return s >= live; }), f.end());
    } else if (mapped < cache_.full_bytes()) {
        const int64_t had = cache_.slots();
        std::string gerr;
        if (!cache_.grow(mapped + ((int64_t) free_b - want_free), gerr)) note = " (" + gerr + ")";
        const int64_t live = cache_.slots();
        std::vector<char> taken((size_t) cache_.full_slots(), 0);
        std::vector<std::pair<int32_t, int32_t>> keep_out;
        for (const auto& [i, slot] : vram_evicted_) {
            if (slot >= live) { keep_out.emplace_back(i, slot); continue; }
            const int64_t layer = i / E;
            int64_t pick = i;
            if (res_[(size_t) i] >= 0) {   // back already (the adaptive tier): the layer's most-routed miss
                pick = -1;
                float best = -1.0f;
                for (int64_t e = 0; e < E; ++e) {
                    const size_t j = (size_t) (layer * E + e);
                    if (res_[j] >= 0) continue;
                    const float u = disp_.usage.empty() ? 0.0f : disp_.usage[j];
                    if (u > best) { best = u; pick = (int64_t) j; }
                }
            }
            if (pick < 0) continue;
            if (!src_->copy_blob(layer, pick % E, fill_tmp_) ||
                !cache_.fill_slot_queued(slot, fill_tmp_, err, (int64_t) k::cpu::expert_layout().blob_bytes(layer)) ||
                !cache_.sync_queued(err)) {   // one staging buffer: each fill lands before the next is read
                err.insert(0, err.empty() ? "VRAM: refilling the tier failed: an expert could not be read"
                                          : "VRAM: refilling the tier failed: ");
                return false;
            }
            res_[(size_t) pick] = slot;
            taken[(size_t) slot] = 1;
        }
        vram_evicted_.swap(keep_out);
        // the slots mapped again that held no expert: back to their layers' free lists
        for (int64_t s = had; s < live; ++s)
            if (!taken[(size_t) s]) free_[(size_t) slot_layer_[(size_t) s]].push_back((int32_t) s);
    }
    strata::gpu::mem_info(&free_b, &total_b);
    std::fprintf(stderr, "xestrata plain: VRAM %lld MiB kept free: the expert tier %lld -> %lld of %lld slots (%.2f of "
                         "%.2f GiB), %lld MiB free now, in %.0f ms%s\n", (long long) reserve_mib, (long long) before,
                 (long long) cache_.slots(), (long long) cache_.full_slots(),
                 (double) cache_.mapped_bytes() / 1073741824.0, (double) cache_.full_bytes() / 1073741824.0,
                 (long long) (free_b >> 20),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), note.c_str());
    char line[256];
    std::snprintf(line, sizeof line, "VRAM reserve_mib=%lld expert_slots=%lld expert_slots_full=%lld "
                  "expert_cache_mib=%lld expert_cache_full_mib=%lld vram_free_mib=%lld prompt_chunk=%lld",
                  (long long) reserve_mib, (long long) cache_.slots(), (long long) cache_.full_slots(),
                  (long long) (cache_.mapped_bytes() >> 20), (long long) (cache_.full_bytes() >> 20),
                  (long long) (free_b >> 20), (long long) opt_.prompt_chunk);
    answer = line;
    return true;
}

int64_t PlainModel::offloaded() const {
    int64_t n = disp_.offload_entries;
    for (const auto& st : next_) n += st->offloaded();
    return n;
}

// the arena holds every expert in RAM; read in place, the RAM copy and the files count apart (FileExpertSource)
int64_t PlainModel::ram_reads() const {
    if (src_ == nullptr) return 0;   // a dense model
    return in_ram_ ? src_->reads() : static_cast<const FileExpertSource*>(src_)->ram_reads();
}

int64_t PlainModel::file_reads() const {
    return in_ram_ || src_ == nullptr ? 0 : static_cast<const FileExpertSource*>(src_)->file_reads();
}

uint64_t PlainModel::file_read_bytes() const {
    return in_ram_ || src_ == nullptr ? 0 : static_cast<const FileExpertSource*>(src_)->file_read_bytes();
}

k::QsaShapes PlainModel::attn_shapes() const {
    k::QsaShapes q = k::qsa_real_shapes();
    q.n_head = spec_.n_head;
    q.n_head_kv = nkv_;
    q.head_dim = spec_.key_length;
    q.n_rot = spec_.rope_dims;
    // pages of 64 cells: a page's cells of a KV head are one run of rows, which the prompt's attention reads 64 at
    // a time where they are (flash_attn.hpp); Qwen3.8-Flash-Next's 4 are its indexer's blocks, which these lack
    q.page_size = 64;
    return q;
}

std::vector<int64_t> PlainModel::split_points() const {
    std::vector<int64_t> at(next_.size());
    for (size_t i = 0; i < next_.size(); ++i) at[i] = next_[i]->lb_;
    return at;
}

bool PlainModel::set_positions(const int32_t* pos3, int64_t n, std::string& err) {
    if (n < 0 || n > opt_.max_cells) { err = "plain model: positions past the context"; return false; }
    for (PlainModel* st : chain()) {
        const OnDevice on(st->dev_);
        if (st->mrope_ == nullptr) { err = "plain model: the engine was started without --vision"; return false; }
        if (n > 0 && !strata::gpu::copy(st->mrope_, pos3, (size_t) n * 3 * 4)) {
            err = std::string("plain model: the rotary positions: ") + strata::gpu::last_error();
            return false;
        }
    }
    return true;
}

bool PlainModel::load_embedding(std::string& err) {
    // the embedding stays in the mapping: a token's row is dequantized on the host (embed)
    const GgufModel& m = *model_;
    size_t at = 0;
    const TensorInfo* t = m.find("token_embd.weight", &at);
    if (t == nullptr || t->shape.size() != 2 || t->shape[0] != (uint64_t) spec_.n_embd || !m.in_bounds(*t, at)) {
        err = "plain model: token_embd.weight is missing or of another width";
        return false;
    }
    emb_type_ = (int) t->type;
    n_vocab_ = (int64_t) t->shape[1];
    emb_row_ = k::cpu::native_row_bytes(emb_type_, spec_.n_embd);
    if (emb_row_ == 0) { err = "plain model: token_embd.weight's type has no host dequantizer"; return false; }
    emb_ = m.shard(at).tensor_data(*t);
    return true;
}

void PlainModel::set_embeddings(const std::vector<const float*>* rows) {
    for (PlainModel* st : chain()) st->emb_rows_ = rows;
}

bool PlainModel::embed(int32_t tok, int64_t pos, float* dst) const {
    if (tok < 0 || tok >= n_vocab_) return false;
    if (emb_rows_ != nullptr && pos >= 0 && pos < (int64_t) emb_rows_->size() && (*emb_rows_)[(size_t) pos] != nullptr) {
        std::memcpy(dst, (*emb_rows_)[(size_t) pos], (size_t) spec_.n_embd * 4);
        return true;
    }
    return k::cpu::native_row_to_float(emb_type_, emb_ + (size_t) tok * emb_row_, dst, spec_.n_embd);
}

// The rows' pages read ahead (platform::advise_willneed: the system's read-ahead, every row's asked at once, read in
// parallel): a prompt's rows that are not in the page cache otherwise come from the disk one fault at a time as embed
// reads them (the B70, whose weights' upload left the mapping's pages uncached: 2694 major faults, 452 ms of a 4K
// prompt's 1.30 s, Qwen3.5-9B; the RTX 4070's 12 ms, 2026-11-07)
void PlainModel::embed_ahead(const int32_t* tokens, int64_t n) const {
    if (emb_ == nullptr || emb_row_ == 0) return;
    for (int64_t t = 0; t < n; ++t)
        if (tokens[t] >= 0 && tokens[t] < n_vocab_)
            strata::platform::advise_willneed(emb_ + (size_t) tokens[t] * emb_row_, emb_row_);
}

void PlainModel::set_sampler(const strata::kernels::SamplerParams& sp) {
    sampler_ = sp;
    for (auto& st : next_) st->sampler_ = sp;
}

// --device cpu: the model file and its spec as for a GPU, then the CPU's forward (plain_cpu.hpp) in place of the
// stages.  The CPU must run AVX2 (plain_cpu.cpp is compiled for it), checked here before any of its code runs.
bool PlainModel::cpu_load(std::string& err) {
    if (!k::cpu::cpu_avx2_ok()) {
        err = "plain model: --device cpu needs a CPU with AVX2";
        return false;
    }
    try {
        model_ = std::make_unique<GgufModel>(GgufModel::open(opt_.gguf));
    } catch (const std::exception& e) {
        err = std::string("plain model: ") + e.what();
        return false;
    }
    if (!read_model_spec(model_->meta(), spec_, err)) return false;
    if (opt_.vision || opt_.encoder || opt_.batch_slots > 0 || !opt_.split_at.empty() || opt_.split_auto) {
        err = "plain model: --device cpu runs one conversation of text: no pictures, batches or layer split";
        return false;
    }
    opt_.spec = 0;   // no MTP layer read: no drafts
    cpu_ = std::make_unique<PlainCpu>();
    if (!cpu_->load(*model_, spec_, opt_, err)) {
        cpu_.reset();
        return false;
    }
    n_vocab_ = cpu_->vocab();
    std::fprintf(stderr, "xestrata plain: the forward on the CPU: %s\n", cpu_->describe().c_str());
    return true;
}

// The CPU's run and its picks: the last `rows` tokens' logits, each row's draw from its position as pick() takes it
bool PlainModel::cpu_run(const int32_t* tokens, int64_t n, int64_t pos, int rows, int32_t* out, float* logits,
                         std::string& err) {
    cpu_logits_.resize((size_t) rows * (size_t) n_vocab_);
    if (!cpu_->run(tokens, n, pos, rows, cpu_logits_.data(), err)) return false;
    k::SamplerParams sp = sampler_;
    sp.counter = (uint64_t) (pos + n - rows);
    sp.logit_bias = cpu_bias_.empty() ? nullptr : cpu_bias_.data();
    sp.allowed = cpu_allowed_.empty() ? nullptr : cpu_allowed_.data();
    const bool hist = cpu_hist_len_ > 0 && sp.penalty_last_n > 0;
    std::vector<int> picks((size_t) rows);
    k::sample_tokens_host(cpu_logits_.data(), rows, (int) n_vocab_, hist ? cpu_hist_.data() : nullptr,
                          hist ? cpu_hist_len_ : 0, sp, picks.data());
    for (int i = 0; i < rows; ++i) out[i] = picks[(size_t) i];
    if (logits != nullptr) std::memcpy(logits, cpu_logits_.data(), cpu_logits_.size() * 4);
    progress_beat();   // the serve watchdog, as the GPU windows' (stage_forward)
    progress_tick();
    return true;
}

bool PlainModel::set_logit_bias(const std::vector<float>& bias, std::string& err) {
    if (cpu_ != nullptr) {
        if (!bias.empty() && bias.size() != (size_t) n_vocab_) {
            err = "plain model: a logit bias of another vocabulary";
            return false;
        }
        cpu_bias_ = bias;
        return true;
    }
    PlainModel& t = tail();
    const OnDevice on(t.dev_);
    t.bias_on_ = !bias.empty();
    if (!t.bias_on_) return true;
    if (bias.size() != (size_t) n_vocab_) { err = "plain model: a logit bias of another vocabulary"; return false; }
    if (!constraint_upload(t.bias_dev_, bias.data(), bias.size() * 4, bias.size() * 4)) {
        t.bias_on_ = false;
        err = std::string("plain model: the logit bias: ") + strata::gpu::last_error();
        return false;
    }
    if (t.bias_dev_ != nullptr && std::find(t.allocs_.begin(), t.allocs_.end(), (void*) t.bias_dev_) == t.allocs_.end())
        t.allocs_.push_back(t.bias_dev_);
    return true;
}

bool PlainModel::set_allowed(const uint32_t* row, std::string& err) {
    if (cpu_ != nullptr) {   // the same row for each of a window's rows
        const size_t words = (size_t) (n_vocab_ + 31) / 32;
        cpu_allowed_.clear();
        if (row != nullptr)
            for (int r = 0; r < opt_.max_t; ++r) cpu_allowed_.insert(cpu_allowed_.end(), row, row + words);
        return true;
    }
    PlainModel& t = tail();
    const OnDevice on(t.dev_);
    t.allowed_on_ = row != nullptr;
    if (!t.allowed_on_) return true;
    const size_t words = (size_t) (n_vocab_ + 31) / 32, T = (size_t) opt_.max_t;
    std::vector<uint32_t> rows(words * T);
    for (size_t r = 0; r < T; ++r) std::copy(row, row + words, rows.begin() + (std::ptrdiff_t) (r * words));
    const bool uploaded = constraint_upload(t.allowed_dev_, rows.data(), words * T * 4, rows.size() * 4);
    if (t.allowed_dev_ != nullptr &&
        std::find(t.allocs_.begin(), t.allocs_.end(), (void*) t.allowed_dev_) == t.allocs_.end())
        t.allocs_.push_back(t.allowed_dev_);   // kept even when the copy failed, as before
    if (!uploaded) {
        t.allowed_on_ = false;
        err = std::string("plain model: the grammar's mask: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool PlainModel::set_allowed_rows(const uint32_t* rows, int T, std::string& err) {
    if (cpu_ != nullptr) {
        if (rows != nullptr && (T < 1 || T > opt_.max_t)) { err = "plain model: mask rows out of range"; return false; }
        const size_t words = (size_t) (n_vocab_ + 31) / 32;
        cpu_allowed_.assign(rows, rows == nullptr ? rows : rows + words * (size_t) T);
        if (rows != nullptr) cpu_allowed_.resize(words * (size_t) opt_.max_t, ~0u);
        return true;
    }
    PlainModel& t = tail();
    const OnDevice on(t.dev_);
    t.allowed_on_ = rows != nullptr;
    if (!t.allowed_on_) return true;
    const size_t words = (size_t) (n_vocab_ + 31) / 32;
    if (T < 1 || T > opt_.max_t) { t.allowed_on_ = false; err = "plain model: mask rows out of range"; return false; }
    const bool uploaded = constraint_upload(t.allowed_dev_, rows, words * (size_t) opt_.max_t * 4, words * (size_t) T * 4);
    if (t.allowed_dev_ != nullptr &&
        std::find(t.allocs_.begin(), t.allocs_.end(), (void*) t.allowed_dev_) == t.allocs_.end())
        t.allocs_.push_back(t.allowed_dev_);   // kept even when the copy failed, as before
    if (!uploaded) {
        t.allowed_on_ = false;
        err = std::string("plain model: the window's masks: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool PlainModel::set_history(const int32_t* tokens, int n, std::string& err) {
    if (cpu_ != nullptr) {   // the same history for each of a window's rows
        cpu_hist_len_ = std::max(0, n);
        cpu_hist_.clear();
        for (int r = 0; r < opt_.max_t && cpu_hist_len_ > 0; ++r) cpu_hist_.insert(cpu_hist_.end(), tokens, tokens + n);
        return true;
    }
    PlainModel& t = tail();
    const OnDevice on(t.dev_);
    t.hist_len_ = std::max(0, n);
    if (t.hist_len_ == 0) return true;
    const size_t T = (size_t) opt_.max_t;
    if (t.hist_len_ > t.hist_cap_) {
        int32_t* d = nullptr;
        if (!strata::gpu::alloc_device(&d, (size_t) t.hist_len_ * T * 4)) {
            t.hist_len_ = 0;
            err = std::string("plain model: the penalties' history: ") + strata::gpu::last_error();
            return false;
        }
        t.allocs_.push_back(d);
        t.hist_dev_ = d;
        t.hist_cap_ = t.hist_len_;
    }
    std::vector<int32_t> rows((size_t) t.hist_len_ * T);
    for (size_t r = 0; r < T; ++r) std::copy(tokens, tokens + n, rows.begin() + (std::ptrdiff_t) (r * (size_t) n));
    if (!strata::gpu::copy(t.hist_dev_, rows.data(), rows.size() * 4)) {
        t.hist_len_ = 0;
        err = std::string("plain model: the penalties' history: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

void PlainModel::pick(const float* logits, int T, int64_t pos, int32_t* out, strata::gpu::Stream cs) const {
    k::SamplerParams sp = sampler_;
    sp.counter = (uint64_t) pos;   // row t's draw: position pos + t
    sp.logit_bias = bias_on_ ? bias_dev_ : nullptr;
    sp.allowed = allowed_on_ ? allowed_dev_ : nullptr;
    const bool hist = hist_len_ > 0 && sp.penalty_last_n > 0;
    k::sample_tokens(logits, T, (int) n_vocab_, hist ? hist_dev_ : nullptr, hist ? hist_len_ : 0, sp, out, cs);
}

void PlainModel::dump_routes(int64_t layer, const int32_t* ids, int64_t n, int64_t k) {
    if (route_dump_ == nullptr || k <= 0) return;
    const int32_t head[2] = {(int32_t) layer, (int32_t) k};
    const std::vector<float> w((size_t) k, 0.0f);
    for (int64_t t = 0; t < n; ++t) {
        std::fwrite(head, sizeof head, 1, route_dump_);
        std::fwrite(ids + t * k, sizeof(int32_t), (size_t) k, route_dump_);
        std::fwrite(w.data(), sizeof(float), (size_t) k, route_dump_);
    }
}

std::vector<PlainModel::StatePart> PlainModel::state_parts(int64_t tokens, bool recurrent_only, int slot) const {
    std::vector<StatePart> parts;
    const ModelSpec& s = spec_;
    tokens = std::clamp<int64_t>(tokens, 0, opt_.max_cells);
    const SlotSession* ss = slot >= 0 ? &bslots_[(size_t) slot] : nullptr;
    for (size_t g = 0; g < gdn_state_.size(); ++g) {   // the recurrent layers: whole (this stage's)
        if (gdn_state_[g] == nullptr) continue;
        parts.push_back({ss ? ss->gdn_state[g] : gdn_state_[g], state_bytes_, 4});
        parts.push_back({ss ? ss->gdn_conv[g] : gdn_conv_[g], conv_bytes_, 4});
    }
    if (recurrent_only && ring_pages_ == 0) return parts;
    if (glm_) {   // per DSA block: its latent and indexer rows a cell, the pools that the cells close
        const int64_t KVR = s.kv_lora_rank, ID = s.idx_key_length;
        const int64_t pools = std::min<int64_t>(g_max_blocks_, tokens / s.idx_kpool + 1);
        for (size_t d = 0; d < g_latent_.size(); ++d) {
            if (g_latent_[d] == nullptr) continue;
            parts.push_back({g_latent_[d], (size_t) (tokens * KVR) * 2, 2});
            parts.push_back({g_idx_cache_[d], (size_t) (tokens * 2 * ID) * 4, 4});
            parts.push_back({g_pooled_[d], (size_t) (pools * ID) * 4, 4});
        }
        return parts;
    }
    // the attention layers (the MTP layer's too): the pages that hold the tokens, page by page in order (the page
    // table is the identity); a sliding-window layer's ring, the pages of the window before `tokens` (the ring's
    // state, as a recurrent layer's: a checkpoint takes it too), in the ring's order
    const k::QsaShapes qs = attn_shapes();
    const int64_t pages = std::min<int64_t>(n_pages_, (tokens + qs.page_size - 1) / qs.page_size);
    const size_t cells = (size_t) (pages * qs.n_head_kv * qs.page_size);
    for (size_t a = 0; a < k_pool_.size(); ++a) {
        if (ringed(a)) {
            if (tokens <= 0) continue;
            const int64_t lp0 = std::max<int64_t>(0, tokens - s.swa_window) / qs.page_size;
            const int64_t lp1 = (tokens - 1) / qs.page_size;   // the window's logical pages lp0 .. lp1
            const std::vector<StatePart> whole =
                kv_planes(a, slot, (size_t) (ring_pages_ * qs.n_head_kv * qs.page_size));
            for (int64_t lp = lp0; lp <= lp1;) {   // runs of the ring's pages, each to its end or the window's
                const int64_t r0 = lp % ring_pages_, n = std::min(lp1 + 1 - lp, ring_pages_ - r0);
                for (const StatePart& w : whole) {
                    const size_t per = w.bytes / (size_t) ring_pages_;
                    parts.push_back({static_cast<uint8_t*>(w.dev) + (size_t) r0 * per, (size_t) n * per, w.typesize});
                }
                lp += n;
            }
            continue;
        }
        if (recurrent_only) continue;
        for (const StatePart& p : kv_planes(a, slot, cells)) parts.push_back(p);
    }
    return parts;
}

size_t PlainModel::Snapshot::bytes() const {
    size_t n = 0;
    for (const auto& p : parts) n += p.size();
    return n;
}

size_t PlainModel::state_bytes(int64_t tokens) const {
    if (cpu_ != nullptr) return cpu_->state_bytes(tokens, true);
    size_t n = 0;
    for (const StatePart& p : state_parts(tokens)) n += p.bytes;
    for (const auto& st : next_) n += st->state_bytes(tokens);
    if (cpu_tail_ != nullptr) n += cpu_tail_->state_bytes(tokens, true);
    return n;
}

// --device cpu: the CPU's parts, every one FP32
static void cpu_snapshot(const PlainCpu& c, int64_t tokens, bool kv, PlainModel::Snapshot& snap) {
    snap = PlainModel::Snapshot{};
    snap.tokens = tokens;
    c.save(tokens, kv, snap.parts);
    snap.typesize.assign(snap.parts.size(), 4);
}

bool PlainModel::recurrent() const {
    if (cpu_ != nullptr || (cpu_tail_ != nullptr && cpu_tail_->recurrent())) return true;
    for (const PlainModel* st : const_cast<PlainModel*>(this)->chain())
        if (!st->gdn_state_.empty() || st->ring_pages_ > 0) return true;
    return false;
}

bool PlainModel::can_rewind(int64_t to, int64_t end) const {
    if (cpu_ != nullptr) return !cpu_->recurrent() && to >= 0;   // its K/V keeps every cell
    if (cpu_tail_ != nullptr && cpu_tail_->recurrent()) return false;
    for (const PlainModel* st : const_cast<PlainModel*>(this)->chain()) {
        if (!st->gdn_state_.empty()) return false;
        if (st->ring_pages_ == 0) continue;
        const int64_t e = end < 0 ? st->ring_end_ : end;
        const int64_t from = end < 0 ? st->ring_from_ : std::max<int64_t>(0, end - spec_.swa_window);
        if (to > e || std::max<int64_t>(0, to - spec_.swa_window + 1) < from) return false;
    }
    return to >= 0;
}

void PlainModel::ring_advance(int64_t pos, int64_t n) {
    if (ring_pages_ == 0) return;
    if (pos == 0) ring_from_ = 0;
    ring_end_ = pos + n;
    // the cells written over: those a ring's length before the last written
    ring_from_ = std::max(ring_from_, ring_end_ - ring_pages_ * attn_shapes().page_size);
}

bool PlainModel::tail_and_head(int T, int64_t pos, std::string& err) {
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain model: the window: ") + strata::gpu::last_error();
        return false;
    }
    if (!cpu_tail_->run_blocks(h_hand_, T, pos, err)) return false;
    const int64_t N = spec_.n_embd;
    k::copy_from_mapped(x_, m_hand_, (int64_t) T * N, cs_);
    k::native_qsa_rms_norm_weighted(x_, out_norm_, hn_, (int) N, T, (float) spec_.rms_eps, cs_);
    head_logits(hn_, T, cs_);
    pick(logits_, T, pos, m_out_, cs_);
    return true;
}

bool PlainModel::save_state(int64_t tokens, Snapshot& snap, std::string& err) {
    if (cpu_ != nullptr) { cpu_snapshot(*cpu_, tokens, true, snap); return true; }
    return save_parts(tokens, false, snap, err);
}

bool PlainModel::restore_state(const Snapshot& snap, std::string& err) {
    if (cpu_ != nullptr) return cpu_->restore(snap.parts, true, err);
    return restore_parts(snap, false, err);
}

bool PlainModel::save_recurrent(int64_t tokens, Snapshot& snap, std::string& err) {
    if (cpu_ != nullptr) { cpu_snapshot(*cpu_, tokens, false, snap); return true; }
    return save_parts(tokens, true, snap, err);
}

bool PlainModel::restore_recurrent(const Snapshot& snap, std::string& err) {
    if (cpu_ != nullptr) return cpu_->restore(snap.parts, false, err);
    return restore_parts(snap, true, err);
}

int PlainModel::batch_slot_count() const {
    int n = (int) bslots_.size();
    for (const auto& st : next_) n = std::min(n, st->batch_slot_count());
    return n;
}

bool PlainModel::save_slot(int b, int64_t tokens, Snapshot& snap, std::string& err) {
    if (b < 0 || b >= batch_slot_count()) { err = "plain model: no such batch slot"; return false; }
    return save_parts(tokens, false, snap, err, b);
}

bool PlainModel::copy_session(int from, int to, int64_t tokens, bool recurrent_only, std::string& err) {
    const int n = batch_slot_count();
    if (from < -1 || to < -1 || from >= n || to >= n || from == to) { err = "plain model: no such batch slot"; return false; }
    for (PlainModel* st : chain()) {
        const OnDevice on(st->dev_);
        if (!st->settle_swaps(true, err)) return false;
        const std::vector<StatePart> src = st->state_parts(tokens, recurrent_only, from);
        const std::vector<StatePart> dst = st->state_parts(tokens, recurrent_only, to);
        for (size_t i = 0; i < src.size(); ++i)
            if (src[i].bytes > 0 && !strata::gpu::copy_async(dst[i].dev, src[i].dev, src[i].bytes, st->cs_)) {
                err = std::string("plain model: copying a session: ") + strata::gpu::last_error();
                return false;
            }
        if (!strata::gpu::stream_sync(st->cs_)) {
            err = std::string("plain model: copying a session: ") + strata::gpu::last_error();
            return false;
        }
        if (to < 0) {
            st->cur_ = -1;
            st->drafts_.clear();
            st->mtp_h_set_ = false;
        }
    }
    return true;
}

// --batch: each slot's copy of what alloc'd the live session's state above (this stage's recurrent and attention
// blocks, the MTP layer's on the last), while the slots take at most half the VRAM free now: the expert tier sizes
// itself from what is left after them (open_experts)
bool PlainModel::alloc_batch_slots(std::string& err) {
    const k::QsaShapes qs = attn_shapes();
    const int want = std::min(opt_.batch_slots, opt_.max_t);
    // --batch-mtp: each slot's recurrent states after its first row as well (a draft that fails goes back to them)
    const bool snaps = opt_.batch_mtp && model_mtp_;
    size_t one = 0;
    for (const StatePart& p : state_parts(opt_.max_cells)) one += p.bytes;
    if (snaps)
        for (const float* g : gdn_state_)
            if (g != nullptr) one += state_bytes_ + conv_bytes_;
    size_t free_b = 0, total_b = 0;
    const size_t room = strata::gpu::mem_info(&free_b, &total_b) ? free_b / 2 : 0;
    const size_t cells = (size_t) (n_pages_ * qs.n_head_kv * qs.page_size);
    for (int b = 0; b < want && (size_t) (b + 1) * one <= room; ++b) {
        SlotSession ss;
        ss.gdn_state.assign(gdn_state_.size(), nullptr);
        ss.gdn_conv.assign(gdn_conv_.size(), nullptr);
        ss.gdn_snap.assign(snaps ? gdn_state_.size() : 0, nullptr);
        ss.conv_snap.assign(snaps ? gdn_conv_.size() : 0, nullptr);
        for (auto* v : {&ss.k_pool, &ss.v_pool, &ss.k_scale, &ss.v_scale}) v->assign(k_pool_.size(), nullptr);
        for (auto* v : {&ss.k_q, &ss.v_q}) v->assign(k_pool_.size(), nullptr);
        for (auto* v : {&ss.k_q4, &ss.v_q4}) v->assign(k_pool_.size(), nullptr);
        bool ok = true;
        auto take = [&](auto*& p, size_t bytes) {
            if (!ok || !strata::gpu::alloc_device(&p, bytes) || !strata::gpu::memset(p, 0, bytes)) return ok = false;
            allocs_.push_back(p);
            return true;
        };
        for (size_t g = 0; g < gdn_state_.size(); ++g)
            if (gdn_state_[g] != nullptr) {
                take(ss.gdn_state[g], state_bytes_);
                take(ss.gdn_conv[g], conv_bytes_);
                if (snaps) {
                    take(ss.gdn_snap[g], state_bytes_);
                    take(ss.conv_snap[g], conv_bytes_);
                }
            }
        for (size_t a = 0; a < k_pool_.size(); ++a) {
            if (kv_planes(a, -1, 1).empty()) continue;   // another stage's block
            KvPtrs p;
            ok = ok && alloc_kv(p, cells);
            ss.k_pool[a] = p.k_pool;
            ss.v_pool[a] = p.v_pool;
            ss.k_scale[a] = p.k_scale;
            ss.v_scale[a] = p.v_scale;
            ss.k_q[a] = p.k_q;
            ss.v_q[a] = p.v_q;
            ss.k_q4[a] = p.k_q4;
            ss.v_q4[a] = p.v_q4;
        }
        if (!ok) break;   // what this slot took stays in allocs_ (freed with the model)
        bslots_.push_back(std::move(ss));
    }
    bslot_snaps_ = snaps;
    if (!strata::gpu::stream_sync(cs_)) { err = std::string("plain model: ") + strata::gpu::last_error(); return false; }
    std::fprintf(stderr, "xestrata plain: %d of %d batch slots (%zu MiB each%s)\n", (int) bslots_.size(),
                 opt_.batch_slots, one >> 20,
                 (int) bslots_.size() < opt_.batch_slots
                     ? opt_.batch_slots > opt_.max_t ? ", a window holds --max-t rows" : ", half the VRAM free at most"
                     : "");
    return true;
}

// Every stage's parts in order, each copied on its own GPU
bool PlainModel::save_parts(int64_t tokens, bool recurrent_only, Snapshot& snap, std::string& err, int slot) {
    snap.tokens = tokens;
    snap.parts.clear();
    snap.typesize.clear();
    for (PlainModel* st : chain()) {
        const OnDevice on(st->dev_);
        if (!st->settle_swaps(true, err)) return false;
        if (!strata::gpu::stream_sync(st->cs_)) {
            err = std::string("plain model: ") + strata::gpu::last_error();
            return false;
        }
        for (const StatePart& p : st->state_parts(tokens, recurrent_only, slot)) {
            snap.parts.emplace_back(p.bytes);
            snap.typesize.push_back((uint8_t) p.typesize);
            if (p.bytes > 0 && !strata::gpu::copy(snap.parts.back().data(), p.dev, p.bytes)) {
                err = std::string("plain model: saving the state: ") + strata::gpu::last_error();
                return false;
            }
        }
    }
    if (cpu_tail_ != nullptr) {   // the CPU's blocks after the GPU's (their K/V cells but for a checkpoint: the RAM
                                  // keeps every one)
        cpu_tail_->save(tokens, !recurrent_only, snap.parts);
        snap.typesize.resize(snap.parts.size(), 4);
    }
    return true;
}

bool PlainModel::restore_parts(const Snapshot& snap, bool recurrent_only, std::string& err) {
    size_t i = 0, n = 0;
    for (PlainModel* st : chain()) n += st->state_parts(snap.tokens, recurrent_only).size();
    if (cpu_tail_ != nullptr) {   // the CPU's blocks' parts after the GPUs'
        if (n > snap.parts.size()) { err = "plain model: a state of another model"; return false; }
        const std::vector<std::vector<uint8_t>> rest(snap.parts.begin() + (std::ptrdiff_t) n, snap.parts.end());
        if (!cpu_tail_->restore(rest, !recurrent_only, err)) return false;
    } else if (n != snap.parts.size()) {
        err = "plain model: a state of another model";
        return false;
    }
    for (PlainModel* st : chain())
        for (const StatePart& p : st->state_parts(snap.tokens, recurrent_only))
            if (p.bytes != snap.parts[i++].size()) { err = "plain model: a state of another shape"; return false; }
    i = 0;
    for (PlainModel* st : chain()) {
        const OnDevice on(st->dev_);
        if (!st->settle_swaps(true, err)) return false;
        if (!strata::gpu::stream_sync(st->cs_)) {
            err = std::string("plain model: ") + strata::gpu::last_error();
            return false;
        }
        for (const StatePart& p : st->state_parts(snap.tokens, recurrent_only)) {
            if (p.bytes > 0 && !strata::gpu::copy(p.dev, snap.parts[i].data(), p.bytes)) {
                err = std::string("plain model: restoring the state: ") + strata::gpu::last_error();
                return false;
            }
            ++i;
        }
        st->cur_ = -1;
        st->drafts_.clear();
        st->mtp_h_set_ = false;
        // the rings hold the window before the state's tokens, and nothing older
        st->ring_end_ = snap.tokens;
        st->ring_from_ = std::max<int64_t>(0, snap.tokens - spec_.swa_window);
    }
    return true;
}

bool PlainModel::reset(std::string& err) {
    if (cpu_ != nullptr) {
        cpu_->reset();
        return true;
    }
    for (auto& st : next_) {
        const OnDevice on(st->dev_);
        if (!st->reset(err)) return false;
    }
    if (glm_) return reset_glm5(err);
    if (cpu_tail_ != nullptr) cpu_tail_->reset();
    bool ok = true;
    for (float* p : gdn_state_) ok = ok && (p == nullptr || strata::gpu::memset(p, 0, state_bytes_));
    for (float* p : gdn_conv_) ok = ok && (p == nullptr || strata::gpu::memset(p, 0, conv_bytes_));
    const k::QsaShapes qs = attn_shapes();
    const size_t cells = (size_t) (n_pages_ * qs.n_head_kv * qs.page_size);
    // (--kv-grow: the cells mapped)
    const size_t live = kv_cells_ < 0 ? cells
                                      : (size_t) ((kv_cells_ + qs.page_size - 1) / qs.page_size * qs.n_head_kv * qs.page_size);
    for (size_t i = 0; i < k_pool_.size(); ++i)
        for (const StatePart& p : kv_planes(i, -1, ringed(i) ? (size_t) (ring_pages_ * qs.n_head_kv * qs.page_size) : live))
            ok = ok && strata::gpu::memset(p.dev, 0, p.bytes);
    ring_end_ = ring_from_ = 0;
    cur_ = -1;
    drafts_.clear();
    if (!ok) err = "plain model: resetting the state failed";
    return ok;
}

void PlainModel::pool_fn(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                         int64_t layer) {
    PlainModel* m = (PlainModel*) user;
    // --batch-groups: the stages' threads take the pool (and its dispatch's buffers and helpers) one at a time
    std::unique_lock<std::mutex> turn(m->root_->pool_mu_, std::defer_lock);
    if (m->root_->piping_) turn.lock();
    const int64_t E = m->spec_.n_expert;
    for (int64_t i = 0; i < n_tok * k; ++i)
        if (ids[i] >= 0 && ids[i] < E) m->heat_[(size_t) (layer * E + ids[i])] += 1.0;
    m->dump_routes(layer, ids, n_tok, k);
    m->disp_.layers = layer;
    expert_pool_dispatch_multi(m->disp_, x_f, ids, n_tok, k, out);
}

bool PlainModel::record_block(const Block& b, int64_t ml, int T, uint32_t& ring, std::vector<strata::gpu::Graph*>& segs,
                              strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, NH = s.n_head, HD = s.key_length, HV = s.ssm_dt_rank, HK = s.ssm_n_group;
    const int64_t ZV = s.ssm_d_inner, S = s.ssm_d_state, C = 2 * HK * S + ZV, E = s.n_expert;
    const int64_t FS = dense_ ? s.n_ff : s.n_ff_shexp;
    const float eps = (float) s.rms_eps;
    const k::QsaShapes qs = attn_shapes();
    const int64_t NKV = qs.n_head_kv;
    const k::RopeScaling& rs = rope_of(b);
    const int32_t* pos_q = pos_;
    const int32_t* pos_k = pos_ + (size_t) opt_.max_t * NH;
    auto proj = [&](const W& w, const float* x, float* y, int ncols, bool quantize) {
        if (w.type == kBF16) {   // from the FP32 rows x (row stride n_in), as llama.cpp's float products
            k::bf16_gemv_fp32_mmvf_multi(x, w.n_in, static_cast<const uint16_t*>(w.data), y, w.n_out, w.n_in, w.n_out,
                                         ncols, cs);
            return;
        }
        if (w.type == kF32) {
            k::f32_gemv_fp32_mmvf_multi(x, w.n_in, static_cast<const float*>(w.data), y, w.n_out, w.n_in, w.n_out,
                                        ncols, cs);
            return;
        }
        if (quantize) k::native_quantize_q8_1(x, xq_, w.n_in, ncols, cs);
        k::native_mmvq(w.type, w.data, xq_, y, w.n_in, w.n_out, ncols, cs);
    };
    // the norm writes the q8_1 rows the projections read as well (one launch; after nextsycl's Qwen3.5 norm)
    k::native_qsa_rms_norm_weighted_q8_1(x_, b.attn_norm, hn_, xq_, (int) N, T, eps, cs);
    // a batch window (row_slot_): each row reads and writes its own slot's state, the rows' products shared
    const bool rows = !row_slot_.empty();
    auto slot_of = [&](int t) -> const SlotSession& { return bslots_[(size_t) row_slot_[(size_t) t]]; };
    if (b.recurrent) {
        float* state = gdn_state_[(size_t) b.gdn];
        float* hist = gdn_conv_[(size_t) b.gdn];
        proj(b.qkv, hn_, qkv_, T, false);
        proj(b.z, hn_, z_, T, false);
        // Qwen3.8's verify-window kernels (verify_kernels.hpp), at most kVerifyMaxT rows a launch: alpha/beta with their
        // gates in one; the conv, SiLU and q/k norms of every row in one; the recurrence and the output norm in one,
        // writing ssm_out's q8_1 input as well.  The conv history and the state go on over the rows (counts_[n]: the
        // kernels' count of rows kept), and the drafts' rows to go back to are written beside (spec_step).
        const bool q8_out = b.ssm_out.type != kBF16 && b.ssm_out.type != kF32;
        const size_t q8_row = (size_t) (ZV / 32) * 36;
        const W& wa = b.alpha_f.type >= 0 ? b.alpha_f : b.alpha;
        const W& wb = b.alpha_f.type >= 0 ? b.beta_f : b.beta;
        auto ab = [&](int t0, int n) {
            if (wa.type == kBF16)
                k::gdn_ab_multi(hn_ + (size_t) t0 * N, static_cast<const uint16_t*>(wa.data),
                                static_cast<const uint16_t*>(wb.data), b.dt, b.a, gate_ + (size_t) t0 * HV,
                                beta_ + (size_t) t0 * HV, (int) N, (int) HV, n, cs);
            else
                k::gdn_ab_multi(hn_ + (size_t) t0 * N, static_cast<const float*>(wa.data),
                                static_cast<const float*>(wb.data), b.dt, b.a, gate_ + (size_t) t0 * HV,
                                beta_ + (size_t) t0 * HV, (int) N, (int) HV, n, cs);
        };
        // rows [t0, t0 + n) from the conv history `hist` and the state `st`, both left after the n rows; conv_snaps /
        // st_snaps: the history and the state after rows t0 .. (t0 + n_snaps - 1)
        auto run = [&](float* hist, float* st, int t0, int n, const k::GdnRowCopies& conv_snaps,
                       const k::GdnRowCopies& st_snaps) {
            const float* q = qkv_ + (size_t) t0 * C;
            k::gdn_conv_l2_multi(hist, q, b.conv, hc_ + (size_t) t0 * C, (int) C, (int) (2 * HK), eps, n, cs, 0, n == 1,
                                 &conv_snaps);
            if (n > 1) k::gdn_conv_commit(hist, q, (int) C, counts_ + n, cs);
            k::gdn_step_norm_multi(st, hc_ + (size_t) t0 * C, (int) C, gate_ + (size_t) t0 * HV, beta_ + (size_t) t0 * HV,
                                   z_ + (size_t) t0 * ZV, b.ssm_norm, eps, y_ + (size_t) t0 * ZV, (int) HK, (int) HV, n,
                                   counts_ + n, cs, 0, q8_out ? (void*) (xq_ + (size_t) t0 * q8_row) : nullptr, &st_snaps);
        };
        const int n_snaps = rows ? 0 : std::min(T - 1, n_snap_);   // the states after rows 0 .. n_snaps - 1, for spec_step
        for (int t0 = 0; t0 < T; t0 += k::kVerifyMaxT) ab(t0, std::min(T - t0, k::kVerifyMaxT));
        if (rows)
            for (int t = 0; t < T;) {   // a slot's rows from its own state (--batch-mtp: two, the state after the first kept)
                const SlotSession& sl = slot_of(t);
                const int n = first_of_two(t) ? 2 : 1;
                k::GdnRowCopies conv_snaps, st_snaps;
                if (n == 2) {
                    conv_snaps.p[0] = sl.conv_snap[(size_t) b.gdn];
                    st_snaps.p[0] = sl.gdn_snap[(size_t) b.gdn];
                    conv_snaps.n = st_snaps.n = 1;
                }
                run(sl.gdn_conv[(size_t) b.gdn], sl.gdn_state[(size_t) b.gdn], t, n, conv_snaps, st_snaps);
                t += n;
            }
        else
            for (int t0 = 0; t0 < T; t0 += k::kVerifyMaxT) {
                const int n = std::min(T - t0, k::kVerifyMaxT);
                k::GdnRowCopies conv_snaps, st_snaps;
                for (int t = t0; t < std::min(t0 + n, n_snaps); ++t) {
                    conv_snaps.p[t - t0] = conv_snap_[(size_t) b.gdn][(size_t) t];
                    st_snaps.p[t - t0] = gdn_snap_[(size_t) b.gdn][(size_t) t];
                    conv_snaps.n = st_snaps.n = t - t0 + 1;
                }
                run(hist, state, t0, n, conv_snaps, st_snaps);
            }
        if (q8_out) k::native_mmvq(b.ssm_out.type, b.ssm_out.data, xq_, mix_, (int) ZV, (int) N, T, cs);
        else proj(b.ssm_out, y_, mix_, T, false);
    } else {
        const int64_t a = b.attn;
        // plamo3's fused Q, K and V: one product for one row (its Q, K and V side by side in qfull_, which holds twice
        // the queries), three for more (a row's three would interleave with the next's)
        float *kr = kraw_, *vc = vcur_;
        if (b.qkv_all.data != nullptr && T == 1) {
            proj(b.qkv_all, hn_, qfull_, T, false);
            kr = qfull_ + NH * HD;
            vc = kr + NKV * HD;
        } else {
            proj(b.q, hn_, qfull_, T, false);
            proj(b.k, hn_, kraw_, T, false);
            proj(b.v, hn_, vcur_, T, false);
        }
        k::native_qsa_rms_norm_rope(qfull_, (int) ((has_gate_ ? 2 : 1) * HD), b.q_norm, qcur_, (int) (T * NH), (int) HD,
                                    (int) s.rope_dims, eps, rs, pos_q, cs);
        k::native_qsa_rms_norm_rope(kr, (int) HD, b.k_norm, kcur_, (int) (T * NKV), (int) HD,
                                    (int) s.rope_dims, eps, rs, pos_k, cs);
        // --kv q4_0: K, V and the queries rotated (kv_q4.hpp); k8v4: V alone; the output back below
        if (opt_.kv_q4 == 1) {
            k::fwht256_inplace_cuda(kcur_, (int64_t) T * NKV, cs);
            k::fwht256_inplace_cuda(vc, (int64_t) T * NKV, cs);
            k::fwht256_inplace_cuda(qcur_, (int64_t) T * NH, cs);
        } else if (opt_.kv_q4 == 2) {
            k::fwht256_inplace_cuda(vc, (int64_t) T * NKV, cs);
        }
        for (int t = 0; t < T; ++t)
            kv_append_cells((size_t) a, rows ? row_slot_[(size_t) t] : -1, step_ + (size_t) t * k::kStepCount, 1,
                            kcur_ + (size_t) t * NKV * HD, vc + (size_t) t * NKV * HD, cs);
        if (rows)   // each row over its own slot's cells
            for (int t = 0; t < T; ++t)
                k::qsa_dense_attn_batch(qcur_ + (size_t) t * NH * HD, kv_pools((size_t) a, row_slot_[(size_t) t]),
                                        step_ + (size_t) t * k::kStepCount, opt_.max_cells, qs, attn_scratch_,
                                        attn_ + (size_t) t * NH * HD, 1, cs, window_of(b));
        else
            k::qsa_dense_attn_batch(qcur_, kv_pools((size_t) a), step_, opt_.max_cells, qs, attn_scratch_, attn_, T, cs,
                                    window_of(b));
        if (opt_.kv_q4 != 0) k::fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
        if (has_gate_) {
            k::native_qsa_gate_apply(attn_, qfull_, attn32_, (int) (T * NH), (int) HD, cs);
            proj(b.o, attn32_, mix_, T, true);
        } else {
            proj(b.o, attn_, mix_, T, true);
        }
    }
    if (b.attn_post != nullptr) {   // plamo3: the mixer's output normed before the add
        k::native_qsa_rms_norm_weighted(mix_, b.attn_post, hn_, (int) N, T, eps, cs);
        k::add_inplace(x_, hn_, (int64_t) T * N, cs);
    } else {
        k::add_inplace(x_, mix_, (int64_t) T * N, cs);
    }

    if (dense_) {   // qwen35's FFN: the shared expert's products, from the norm's q8_1 rows where the weights take them
        const bool q8 = b.sh_gate.type != kBF16 && b.sh_up.type != kBF16 && b.sh_down.type != kBF16;
        if (!q8) {
            k::native_qsa_rms_norm_weighted(x_, b.post_norm, hn_, (int) N, T, eps, cs);
            proj(b.sh_gate, hn_, sh_gate_, T, true);   // (each quantizes hn_ again: a file may mix the types)
            proj(b.sh_up, hn_, sh_up_, T, true);
            k::native_swiglu_f32(sh_gate_, sh_up_, sh_gate_, (int) FS, T, cs);
            proj(b.sh_down, sh_gate_, moe_out_, T, true);
        } else {
            k::native_qsa_rms_norm_weighted_q8_1(x_, b.post_norm, hn_, xq_, (int) N, T, eps, cs);
            float* up = sh_up_;
            if (b.gu_all.data != nullptr && T == 1) {   // plamo3's fused gate and up: one product for one row
                k::native_mmvq(b.gu_all.type, b.gu_all.data, xq_, sh_gate_, (int) N, (int) (2 * FS), T, cs);
                up = sh_gate_ + FS;
            } else if (b.sh_gate.type != b.sh_up.type ||
                       !k::native_mmvq_pair(b.sh_gate.type, b.sh_gate.data, b.sh_up.data, xq_, sh_gate_, sh_up_,
                                            (int) N, (int) FS, T, cs)) {
                k::native_mmvq(b.sh_gate.type, b.sh_gate.data, xq_, sh_gate_, (int) N, (int) FS, T, cs);
                k::native_mmvq(b.sh_up.type, b.sh_up.data, xq_, sh_up_, (int) N, (int) FS, T, cs);
            }
            k::native_swiglu_q8_1(sh_gate_, up, xq_, (int) FS, T, cs);
            k::native_mmvq(b.sh_down.type, b.sh_down.data, xq_, moe_out_, (int) FS, (int) N, T, cs);
        }
        if (b.ffn_post != nullptr) {   // plamo3: the FFN's output normed before the add
            k::native_qsa_rms_norm_weighted(moe_out_, b.ffn_post, hn_, (int) N, T, eps, cs);
            k::add_inplace(x_, hn_, (int64_t) T * N, cs);
        } else {
            k::add_inplace(x_, moe_out_, (int64_t) T * N, cs);
        }
        steer(b, T, x_, cs);
        tap_block(b, T, cs);
        return true;
    }

    // the MoE: route, publish, the shared expert while the host serves, then the experts and the combine
    const bool sh_q8 = has_shared_ && b.sh_gate.type != kBF16 && b.sh_down.type != kBF16;
    if (sh_q8) k::native_qsa_rms_norm_weighted_q8_1(x_, b.post_norm, hn_, xq_, (int) N, T, eps, cs);
    else k::native_qsa_rms_norm_weighted(x_, b.post_norm, hn_, (int) N, T, eps, cs);
    k::f32_gemv_fp32_mmvf_multi(hn_, N, b.router, rlog_, E, N, E, T, cs);
    moe_ring(ml, T, ring, rlog_, cs);
    if (has_shared_ && !sh_q8) {   // in FP32 between the products
        proj(b.sh_gate, hn_, sh_gate_, T, true);
        proj(b.sh_up, hn_, sh_up_, T, false);
        k::native_swiglu_f32(sh_gate_, sh_up_, sh_gate_, (int) FS, T, cs);
        proj(b.sh_down, sh_gate_, shared_, T, true);
        k::f32_gemv_fp32_mmvf_multi(hn_, N, b.sh_gate_inp, sh_g_, 1, N, 1, T, cs);
    } else if (has_shared_) {
        if (!k::native_mmvq_pair(b.sh_gate.type, b.sh_gate.data, b.sh_up.data, xq_, sh_gate_, sh_up_, (int) N,
                                 (int) FS, T, cs)) {
            k::native_mmvq(b.sh_gate.type, b.sh_gate.data, xq_, sh_gate_, (int) N, (int) FS, T, cs);
            k::native_mmvq(b.sh_up.type, b.sh_up.data, xq_, sh_up_, (int) N, (int) FS, T, cs);
        }
        k::native_swiglu_q8_1(sh_gate_, sh_up_, xq_, (int) FS, T, cs);
        k::native_mmvq(b.sh_down.type, b.sh_down.data, xq_, shared_, (int) FS, (int) N, T, cs);
        k::f32_gemv_fp32_mmvf_multi(hn_, N, b.sh_gate_inp, sh_g_, 1, N, 1, T, cs);
    }
    if (!moe_experts(ml, T, ring, segs, cs, err)) return false;
    win_.combine(0, 0, T, w_, has_shared_ ? shared_ : nullptr, moe_out_, has_shared_ ? sh_g_ : nullptr, cs);
    k::add_inplace(x_, moe_out_, (int64_t) T * N, cs);
    steer(b, T, x_, cs);
    tap_block(b, T, cs);
    return true;
}

// A MoE layer's routing for the window's ring: published to the host (its CPU experts and plan) or, in the window's
// graph for every routed expert in VRAM (res_rec_), the plan built on the device and the ids kept for the host's counts
// (res_count); then the experts' input rows.  `rlog`: the router's logits, routed here (the softmax top k, with the
// doorbell in the same launch where that takes the shape); null: ids_ / w_ routed already (GLM's sigmoid router).
void PlainModel::moe_ring(int64_t ml, int T, uint32_t& ring, const float* rlog, strata::gpu::Stream cs) {
    const int64_t E = spec_.n_expert, K = spec_.n_expert_used;
    ++ring;
    if (res_rec_) {
        if (rlog != nullptr) k::router_top10(rlog, T, (int) E, (int) K, ids_, w_, cs);
        win_.resident_plan(res_d_ + ml * E, 0, T, ids_, ring, cs);
        k::copy_i32_from_mapped(m_rroute_ + (size_t) (ring - 1) * (size_t) (opt_.max_t * K), ids_, (int64_t) T * K, cs);
    } else if (rlog == nullptr || !win_.route_publish(0, T, rlog, ids_, w_, hn_, cs)) {
        if (rlog != nullptr) k::router_top10(rlog, T, (int) E, (int) K, ids_, w_, cs);
        win_.doorbell(0, T, hn_, ids_, w_, cs);
    }
    win_.expert_input(0, T, hn_, cs);
}

// The ring's experts into win_.parts() (a segmented window's graph ends first, where the host answers the ring)
bool PlainModel::moe_experts(int64_t ml, int T, uint32_t ring, std::vector<strata::gpu::Graph*>& segs,
                             strata::gpu::Stream cs, std::string& err) {
    if (win_.segmented() && !res_rec_) {
        strata::gpu::Graph* g = nullptr;
        if (!strata::gpu::end_capture(cs, &g)) {
            err = std::string("plain model: segment capture: ") + strata::gpu::last_error();
            return false;
        }
        segs.push_back(g);
        if (!strata::gpu::begin_capture(cs)) { err = "plain model: begin capture failed"; return false; }
    }
    MoeExpertsMode mode;
    mode.resident = res_rec_;
    return win_.experts(ml, ring, 0, 1, 0, T, cs, err, mode);
}

// the DFlash2 tap of a block: its completed output (the plain residual has one hc stream), the window's rows
void PlainModel::tap_block(const Block& b, int T, strata::gpu::Stream cs) {
    if (tap_dev_ == nullptr) return;
    const int64_t l = &b >= blocks_.data() && &b < blocks_.data() + blocks_.size() ? &b - blocks_.data() : -1;
    if (l < 0 || l >= (int64_t) tap_slot_.size() || tap_slot_[(size_t) l] < 0) return;
    const int64_t N = spec_.n_embd, nt = (int64_t) tap_local_.size();
    strata::kernels::df2_tap_rows(x_, tap_dev_ + (size_t) tap_slot_[(size_t) l] * N, nt * N, N, 1, T, cs);
}

void PlainModel::steer(const Block& b, int64_t T, float* x, strata::gpu::Stream cs) const {
    const int64_t l = &b >= blocks_.data() && &b < blocks_.data() + blocks_.size() ? &b - blocks_.data() : -1;
    if (l >= 0 && k::cvec().covers(l)) k::cvec_apply(x, l, T, spec_.n_embd, nullptr, 0, nullptr, 0, false, cs);
    if (l >= 0 && block_hook_ != nullptr && enc_out_ != nullptr) block_hook_->after(l, x, enc_row0_, T, cs);
}

bool PlainModel::replicate_cvec(std::string& err) {
    for (PlainModel* st : chain()) {
        const OnDevice on(st->dev_);
        if (!k::cvec_replicate(err)) return false;
    }
    return true;
}

bool PlainModel::record_window(int T, strata::gpu::Stream cs, std::string& err,
                               std::vector<strata::gpu::Graph*>* segs) {
    if (glm_) return record_window_glm5(T, cs, err);
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, NH = s.n_head;
    const float eps = (float) s.rms_eps;
    const int64_t NKV = attn_shapes().n_head_kv;
    k::copy_from_mapped(x_, first() ? m_emb_ : m_in_, (int64_t) T * N, cs);   // the embeddings, or the rows handed on
    k::copy_i32_from_mapped(step_, m_step_, (int64_t) T * k::kStepCount, cs);
    k::copy_i32_from_mapped(pos_, m_pos_, (int64_t) opt_.max_t * (NH + NKV), cs);
    uint32_t ring = 0;
    for (int64_t l = lb_; l < le_; ++l)
        if (!record_block(blocks_[(size_t) l], s.moe_index(l), T, ring, segs != nullptr ? *segs : segs_[(size_t) T], cs,
                          err))
            return false;
    if (!last()) {   // the rows to the next stage
        k::copy_from_mapped(m_hand_, x_, (int64_t) T * N, cs);
        return true;
    }
    k::native_qsa_rms_norm_weighted(x_, out_norm_, hn_, (int) N, T, eps, cs);
    head_logits(hn_, T, cs);   // the tokens are picked after the graph (forward), with the run's sampler
    return true;
}

void PlainModel::head_logits(const float* x, int T, strata::gpu::Stream cs) {
    if (head_.type == kBF16) {
        k::bf16_gemv_fp32_mmvf_multi(x, head_.n_in, static_cast<const uint16_t*>(head_.data), logits_, head_.n_out,
                                     head_.n_in, head_.n_out, T, cs);
        return;
    }
    k::native_quantize_q8_1(x, xq_, head_.n_in, T, cs);
    k::native_mmvq(head_.type, head_.data, xq_, logits_, head_.n_in, head_.n_out, T, cs);
}

bool PlainModel::record_mtp(int T, strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, NH = s.n_head;
    const float eps = (float) s.rms_eps;
    const int64_t NKV = attn_shapes().n_head_kv;
    k::copy_from_mapped(x_, m_emb_, (int64_t) T * N, cs);
    k::copy_i32_from_mapped(step_, m_step_, (int64_t) T * k::kStepCount, cs);
    k::copy_i32_from_mapped(pos_, m_pos_, (int64_t) opt_.max_t * (NH + NKV), cs);
    // llama.cpp's graph_mtp: the block's input is eh_proj([rms_norm(e) enorm | rms_norm(h) hnorm])
    k::native_qsa_rms_norm_weighted(x_, mtp_enorm_, hn_, (int) N, T, eps, cs);
    k::native_qsa_rms_norm_weighted(mtp_h_, mtp_hnorm_, mix_, (int) N, T, eps, cs);
    k::concat_rows(qfull_, hn_, mix_, N, T, cs);
    if (mtp_eh_.type == kBF16) {
        k::bf16_gemv_fp32_mmvf_multi(qfull_, 2 * N, static_cast<const uint16_t*>(mtp_eh_.data), x_, N, 2 * N, N, T, cs);
    } else {
        k::native_quantize_q8_1(qfull_, xq_, (int) (2 * N), T, cs);
        k::native_mmvq(mtp_eh_.type, mtp_eh_.data, xq_, x_, (int) (2 * N), (int) N, T, cs);
    }
    uint32_t ring = 0;
    if (!record_block(mtp_, n_moe_ - 1, T, ring, msegs_[(size_t) T], cs, err)) return false;
    // the last row only: its norm is the next step's trunk row, and its logits pick the draft
    k::native_qsa_rms_norm_weighted(x_ + (size_t) (T - 1) * N, mtp_head_norm_, hn_, (int) N, 1, eps, cs);
    k::copy_from_mapped(mtp_h_, hn_, N, cs);
    draft_logits(pos_ + (size_t) (T - 1) * NH, cs);
    return true;
}

// record_mtp over the batch slots' rows (row_slot_): the block reads and writes each row's slot's cells, and each
// slot's last row, normed, goes through the whole head (logits_ row j the j-th slot's); no chain after it
bool PlainModel::record_mtp_slots(int T, strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, NH = s.n_head;
    const float eps = (float) s.rms_eps;
    const int64_t NKV = attn_shapes().n_head_kv;
    k::copy_from_mapped(x_, m_emb_, (int64_t) T * N, cs);
    k::copy_i32_from_mapped(step_, m_step_, (int64_t) T * k::kStepCount, cs);
    k::copy_i32_from_mapped(pos_, m_pos_, (int64_t) opt_.max_t * (NH + NKV), cs);
    k::native_qsa_rms_norm_weighted(x_, mtp_enorm_, hn_, (int) N, T, eps, cs);
    k::native_qsa_rms_norm_weighted(mtp_h_, mtp_hnorm_, mix_, (int) N, T, eps, cs);
    k::concat_rows(qfull_, hn_, mix_, N, T, cs);
    if (mtp_eh_.type == kBF16) {
        k::bf16_gemv_fp32_mmvf_multi(qfull_, 2 * N, static_cast<const uint16_t*>(mtp_eh_.data), x_, N, 2 * N, N, T, cs);
    } else {
        k::native_quantize_q8_1(qfull_, xq_, (int) (2 * N), T, cs);
        k::native_mmvq(mtp_eh_.type, mtp_eh_.data, xq_, x_, (int) (2 * N), (int) N, T, cs);
    }
    uint32_t ring = 0;
    const uint64_t key = slots_key(row_slot_.data(), T) | (uint64_t{1} << 63);
    if (!record_block(mtp_, n_moe_ - 1, T, ring, mbsegs_[key], cs, err)) return false;
    int j = 0;
    for (int t = 0; t < T; ++t)
        if (!first_of_two(t))
            k::native_qsa_rms_norm_weighted(x_ + (size_t) t * N, mtp_head_norm_, hn_ + (size_t) (j++) * N, (int) N, 1,
                                            eps, cs);
    head_logits(hn_, j, cs);
    return true;
}

void PlainModel::draft_logits(const int32_t* pos, strata::gpu::Stream cs) {
    const int n_in = head_.n_in;
    // the head's own product of n rows in its form (the subset's, the whole head's, or int2_'s candidates)
    auto exact = [this, n_in](const void* rows, int n, float* y, void* s) {
        if (head_.type == kBF16) {
            k::bf16_gemv_fp32_mmvf_multi(hn_, n_in, static_cast<const uint16_t*>(rows), y, n, n_in, n, 1, s);
        } else {
            k::native_mmvq(head_.type, rows, xq_, y, n_in, n, 1, s);
        }
    };
    const void* rows = dhead_ != nullptr ? dhead_ : head_.data;   // the subset's rows: logits_[i] is token dvocab_[i]'s
    const int n = (int) (dhead_ != nullptr ? n_dvocab_ : head_.n_out);
    if (head_.type != kBF16) k::native_quantize_q8_1(hn_, xq_, n_in, 1, cs);
    if (int2_ != nullptr)
        int2_->logits(hn_, static_cast<const uint8_t*>(rows),
                      [&](const uint8_t* r, int m, float* y, void* s) { exact(r, m, y, s); }, logits_, cs, pos);
    else
        exact(rows, n, logits_, cs);
}

void PlainModel::setup_int2() {
    const char* v = strata::env_get("DRAFT_INT2");
    if (!has_mtp_ || n_mtp_ < 1 || head_.n_in % k::kInt2Group != 0 ||
        (v != nullptr && v[0] != 0 && std::strtol(v, nullptr, 10) == 0))
        return;
    size_t at = 0;
    const TensorInfo* t = model_->find(head_name(), &at);
    if (t == nullptr) return;
    const auto* src = static_cast<const uint8_t*>(model_->shard(at).tensor_data(*t));
    const int type = (int) t->type;
    const size_t file_row = type == kBF16 ? (size_t) head_.n_in * 2 : k::cpu::native_row_bytes(type, head_.n_in);
    // the head's rows on the GPU keep their GGUF size in the engine's layouts (upload)
    const size_t row = head_.type == kBF16 ? (size_t) head_.n_in * 2 : file_row;
    const int64_t n = dhead_ != nullptr ? n_dvocab_ : head_.n_out;
    auto h = std::make_unique<DraftHeadInt2>();
    std::string e;
    if (file_row == 0 || !h->build(type, n, head_.n_in, [&](int64_t i) {
            return src + (size_t) (dhead_ != nullptr ? dvocab_[(size_t) i] : i) * file_row;
        }, row, e)) {
        std::fprintf(stderr, "xestrata plain: the drafts' head stays as it is (%s)\n", e.c_str());
        return;
    }
    const void* rows = dhead_ != nullptr ? dhead_ : head_.data;
    const int n_in = head_.n_in;
    auto exact = [this, n_in](const uint8_t* r, int m, float* y, void* s) {
        if (head_.type == kBF16) k::bf16_gemv_fp32_mmvf_multi(hn_, n_in, reinterpret_cast<const uint16_t*>(r), y, m, n_in, m, 1, s);
        else k::native_mmvq(head_.type, r, xq_, y, n_in, m, 1, s);
    };
    // the timing reads hn_ before any window has written it: cleared, so the coarse logits are numbers (left as they
    // were, NaNs made every top-k slot empty, and an RTX 4070 start faulted 7 times in 10, 2026-11-08)
    strata::gpu::Stream s = strata::gpu::stream_create();
    const bool on = s != nullptr && strata::gpu::memset(hn_, 0, (size_t) n_in * 4) &&
                    h->choose(hn_, static_cast<const uint8_t*>(rows), exact,
                              [&](float* out, void* st) { exact(static_cast<const uint8_t*>(rows), (int) n, out, st); },
                              logits_, s, "xestrata plain");
    if (s != nullptr) strata::gpu::stream_destroy(s);
    if (on) int2_ = std::move(h);
}

// The graph of a window of T rows, the trunk's or (`mtp`) the MTP layer's, recorded once.
bool PlainModel::record(int T, bool mtp, std::string& err) {
    if (step_form_.form() == 1) return true;   // run by run_window each time
    return capture(T, mtp, (mtp ? mexec_ : exec_)[(size_t) T], (mtp ? msegs_ : segs_)[(size_t) T], err);
}

bool PlainModel::capture(int T, bool mtp, strata::gpu::Graph*& exec, std::vector<strata::gpu::Graph*>& segs,
                         std::string& err) {
    if (exec != nullptr || !segs.empty()) return true;
    if (!strata::gpu::begin_capture(cs_)) {
        err = std::string("plain model: begin capture: ") + strata::gpu::last_error();
        return false;
    }
    bool ok = false;
    try {
        ok = mtp ? (glm_ ? record_mtp_glm5(T, cs_, err) : !row_slot_.empty() ? record_mtp_slots(T, cs_, err) : record_mtp(T, cs_, err))
                 : record_window(T, cs_, err, &segs);
    } catch (const std::exception& e) {
        err = std::string("plain model: recording the window: ") + e.what();
    }
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        for (auto* g : segs) strata::gpu::graph_destroy(g);
        segs.clear();
        return false;
    }
    strata::gpu::Graph* g = nullptr;
    if (!strata::gpu::end_capture(cs_, &g)) {
        err = std::string("plain model: end capture: ") + strata::gpu::last_error();
        return false;
    }
    if (win_.segmented() && !res_rec_) segs.push_back(g);
    else exec = g;
    return true;
}

// STRATA_PLAIN_EAGER's window: what record() would have put in the graph, onto the stream as it is
bool PlainModel::run_window(int T, bool mtp, std::string& err) {
    try {
        return mtp ? (glm_ ? record_mtp_glm5(T, cs_, err) : !row_slot_.empty() ? record_mtp_slots(T, cs_, err)
                                                                              : record_mtp(T, cs_, err))
                   : record_window(T, cs_, err);
    } catch (const std::exception& e) {
        err = std::string("plain model: the window: ") + e.what();
        return false;
    }
}

// The window's inputs in the mapped staging: the tokens' embedding rows, their step records and rotary positions.
bool PlainModel::stage(const int32_t* tokens, int T, int64_t pos, std::string& err, bool emb, const int64_t* rows_pos) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, NH = s.n_head;
    const k::QsaShapes qs = attn_shapes();
    for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= n_vocab_) { err = "plain model: a token is out of the vocabulary"; return false; }
        const int64_t p = rows_pos != nullptr ? rows_pos[t] : pos + t;
        if (emb && !embed(tokens[t], p, h_emb_ + (size_t) t * N)) {
            err = "plain model: the embedding row could not be dequantized";
            return false;
        }
        k::qsa_step_fill(h_step_ + (size_t) t * k::kStepCount, p, qs);
        for (int64_t h = 0; h < NH; ++h) h_pos_[(size_t) (t * NH + h)] = (int32_t) p;
        int32_t* pk = h_pos_ + (size_t) opt_.max_t * NH;
        for (int64_t h = 0; h < qs.n_head_kv; ++h) pk[(size_t) (t * qs.n_head_kv + h)] = (int32_t) p;
    }
    return true;
}

// Every routed expert of `rings` MoE layers layers[0 ..] in VRAM: the device's residency table brought up to the host's
// first (a fill, a swap or an eviction since the last window changes it; the table is small, the compare a few us)
bool PlainModel::all_resident(int64_t rings, const int64_t* layers, std::string& err) {
    if (res_d_ == nullptr) return false;
    if (res_shadow_ != res_) {
        if (!strata::gpu::copy(res_d_, res_.data(), res_.size() * 4)) {
            err = std::string("plain model: the residency table: ") + strata::gpu::last_error();
            return false;
        }
        res_shadow_ = res_;
    }
    const int64_t E = spec_.n_expert;
    for (int64_t r = 0; r < rings; ++r)
        for (int64_t e = 0; e < E; ++e)
            if (res_[(size_t) (layers[r] * E + e)] < 0) return false;
    return true;
}

bool PlainModel::capture_res(int T, bool mtp, strata::gpu::Graph*& exec, std::string& err) {
    if (exec != nullptr) return true;
    std::vector<strata::gpu::Graph*> none;   // the resident graph is never cut into segments
    res_rec_ = true;
    const bool ok = capture(T, mtp, exec, none, err);
    res_rec_ = false;
    return ok;
}

// What the pool would have counted for the window that ran without it (pool_fn's counts, expert_pool_count)
void PlainModel::res_count() {
    if (res_rings_.empty()) return;
    const int64_t E = spec_.n_expert, K = spec_.n_expert_used;
    for (size_t r = 0; r < res_rings_.size(); ++r) {
        const int64_t layer = res_rings_[r];
        const int32_t* ids = h_rroute_ + r * (size_t) (opt_.max_t * K);
        for (int64_t i = 0; i < (int64_t) res_t_ * K; ++i)
            if (ids[i] >= 0 && ids[i] < E) heat_[(size_t) (layer * E + ids[i])] += 1.0;
        dump_routes(layer, ids, res_t_, K);
        disp_.layers = layer;
        expert_pool_count(disp_, ids, res_t_, K);
        disp_.cache_hits += (int64_t) res_t_ * K;   // every routed entry a VRAM hit, as the pool's plan counts them
    }
    res_rings_.clear();
}

// Whether this window runs its graph without the host: every routed expert of its MoE layers in VRAM (as
// Qwen3.8-Flash-Next's verify window; STRATA_VERIFY_RESIDENT_GRAPH=0: never).  Asked before the window's graph is
// recorded, so a window records only the form it runs: recording both cost the short decodes more than the graph
// without the host saved (B70, 32 tokens after a 4K prompt: Qwen3.6-35B-A3B 120 -> 114 tok/s with both recorded,
// 123 with one; 2026-11-12).
bool PlainModel::res_window(int64_t rings, const int64_t* layers, std::string& err) {
    return !dense_ && MoeWindow::resident_graph_on() && all_resident(rings, layers, err);
}

bool PlainModel::launch_res(int T, bool mtp, strata::gpu::Graph*& exec, int64_t rings, const int64_t* layers,
                            std::string& err) {
    if (!capture_res(T, mtp, exec, err)) return false;
    if (!strata::gpu::graph_launch(exec, cs_)) {
        err = std::string("plain model: launch: ") + strata::gpu::last_error();
        return false;
    }
    res_rings_.assign(layers, layers + rings);
    res_t_ = T;
    return true;
}

bool PlainModel::launch_serve(strata::gpu::Graph* g, std::vector<strata::gpu::Graph*>& segs, int64_t rings,
                              const int64_t* layers, int T, std::string& err) {
    if (dense_) {   // no rings to answer: the window's graph alone
        if (strata::gpu::graph_launch(g, cs_)) return true;
        err = std::string("plain model: launch: ") + strata::gpu::last_error();
        return false;
    }
    win_.reset();
    const bool seg = win_.segmented();
    if (!strata::gpu::graph_launch(seg ? segs[0] : g, cs_)) {
        err = std::string("plain model: launch: ") + strata::gpu::last_error();
        return false;
    }
    (void) strata::gpu::stream_idle(cs_);
    std::vector<int> grps((size_t) rings, 0), tbs((size_t) rings, 0), ns((size_t) rings, T);
    if (seg) win_.set_segments(&segs);
    return win_.serve(rings, layers, grps.data(), 1, tbs.data(), ns.data(), &PlainModel::pool_fn, this, cs_, err);
}

bool PlainModel::forward(const int32_t* tokens, int T, int64_t pos, int32_t* out, float* logits, std::string& err,
                         bool defer) {
    if (T < 1 || T > opt_.max_t || pos < 0 || pos + T > opt_.max_cells) {
        err = "plain model: the window is out of range";
        return false;
    }
    if (cpu_ != nullptr) return cpu_run(tokens, T, pos, T, out, logits, err);
    const float* hand = nullptr;   // each stage's rows to the next
    for (PlainModel* st : chain()) {
        const OnDevice on(st->dev_);
        if (!st->stage_forward(tokens, T, pos, hand, err, defer)) return false;
        hand = st->h_hand_;
    }
    PlainModel& t = tail();
    const OnDevice on(t.dev_);
    for (int i = 0; i < T; ++i) out[i] = ((volatile int32_t*) t.h_out_)[i];
    if (logits != nullptr && !strata::gpu::copy(logits, t.logits_, (size_t) T * (size_t) n_vocab_ * 4)) {
        err = "plain model: reading the logits failed";
        return false;
    }
    t.last_T_ = T;
    t.mtp_h_set_ = false;
    return true;
}

bool PlainModel::forward_run(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits,
                             std::string& err, const std::function<void(int64_t)>& progress) {
    if (cpu_ != nullptr) {   // one run; out and logits the last window's rows, as the windows would leave them
        if (n < 1) { err = "plain model: no tokens to run"; return false; }
        const int last = (int) ((n - 1) % opt_.max_t) + 1;
        if (!cpu_run(tokens, n, pos, last, out, logits, err)) return false;
        if (progress) progress(n);
        return true;
    }
    const std::vector<PlainModel*> st = chain();
    const size_t S = st.size();
    struct Win { int64_t at; int T; std::vector<float> hand; };
    std::vector<Win> wins;
    for (int64_t p = 0; p < n;) {
        const int T = (int) std::min<int64_t>(opt_.max_t, n - p);
        wins.push_back({p, T, {}});
        p += T;
    }
    if (wins.empty()) { err = "plain model: no tokens to read"; return false; }
    if (S < 2 || opt_.pipeline_windows == 0 || wins.size() < 2) {
        for (const Win& w : wins) {
            if (!forward(tokens + w.at, w.T, pos + w.at, out, &w == &wins.back() ? logits : nullptr, err)) return false;
            if (progress) progress(w.at + w.T);
        }
        return true;
    }
    if (pos < 0 || pos + n > opt_.max_cells) { err = "plain model: the window is out of range"; return false; }
    const size_t J = wins.size();
    std::vector<std::string> errs(S);
    std::vector<char> failed(S, 0);
    auto run = [&](size_t s, size_t j) {
        PlainModel* m = st[s];
        Win& w = wins[j];
        const OnDevice on(m->dev_);
        if (!m->stage_forward(tokens + w.at, w.T, pos + w.at, s == 0 ? nullptr : wins[j].hand.data(), errs[s], false)) {
            failed[s] = 1;
            return;
        }
        if (s + 1 < S) w.hand.assign(m->h_hand_, m->h_hand_ + (size_t) w.T * (size_t) m->hand_w_);
    };
    for (size_t tick = 0; tick < J + S - 1; ++tick) {
        piping_ = true;
        {
            std::vector<std::thread> others;
            for (size_t s = 1; s < S; ++s)
                if (tick >= s && tick - s < J) others.emplace_back(run, s, tick - s);
            if (tick < J) run(0, tick);
            for (auto& t : others) t.join();
        }
        piping_ = false;
        for (size_t s = 0; s < S; ++s)
            if (failed[s]) {
                err = errs[s];
                return false;
            }
        if (tick >= S - 1) {
            const Win& w = wins[tick - (S - 1)];
            if (progress) progress(w.at + w.T);
            if (!w.hand.empty()) std::vector<float>().swap(wins[tick - (S - 1)].hand);
        }
    }
    PlainModel& t = tail();
    const OnDevice on(t.dev_);
    const int T = wins.back().T;
    for (int i = 0; i < T; ++i) out[i] = ((volatile int32_t*) t.h_out_)[i];
    if (logits != nullptr && !strata::gpu::copy(logits, t.logits_, (size_t) T * (size_t) n_vocab_ * 4)) {
        err = "plain model: reading the logits failed";
        return false;
    }
    t.last_T_ = T;
    t.mtp_h_set_ = false;
    return true;
}

bool PlainModel::stage_forward(const int32_t* tokens, int T, int64_t pos, const float* hand_in, std::string& err,
                               bool defer) {
    if (!settle_window(err) || !stage(tokens, T, pos, err, first())) return false;
    if (hand_in != nullptr) std::memcpy(h_in_, hand_in, (size_t) T * (size_t) hand_w_ * 4);
    ring_advance(pos, T);
    pcie_probe_before();
    const double pool0 = disp_.ms_plan + disp_.ms_actq + disp_.ms_jobs + disp_.ms_run;
    // while the form is timed, the one-row windows take the two in turns (the same tokens either way)
    const bool timing = step_form_.form() == -1 && T == 1;
    const bool eager = step_form_.form() == 1 || (timing && step_form_.straight());
    const bool res = !eager && res_window(moe_trunk_, moe_local_.data(), err);
    if (!err.empty() || (!res && !record(T, false, err))) return false;
    const auto t0 = std::chrono::steady_clock::now();
    if (!(eager ? run_window(T, false, err)
                : res ? launch_res(T, false, rexec_[(size_t) T], moe_trunk_, moe_local_.data(), err)
                      : launch_serve(exec_[(size_t) T], segs_[(size_t) T], moe_trunk_, moe_local_.data(), T, err)))
        return false;
    if (cpu_tail_ != nullptr && !tail_and_head(T, pos, err)) return false;
    if (last()) pick(logits_, T, pos, m_out_, cs_);
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain model: the window: ") + strata::gpu::last_error();
        return false;
    }
    res_count();
    const double win_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (timing) step_form_.timed(win_ms, "xestrata plain", "the decode windows");
    pcie_probe_after(win_ms, T);
    if (disp_.failed) {
        err = std::string("plain model: the CPU experts: ") + (disp_.fail ? disp_.fail : "failed");
        return false;
    }
    pool_ms_ += disp_.ms_plan + disp_.ms_actq + disp_.ms_jobs + disp_.ms_run - pool0;
    win_ms_ += win_ms;
    ++windows_;
    progress_beat();   // the serve watchdog: a window (decode or prompt) is still moving
    progress_tick();   // plain's windows run as one graph launch: the tick counts windows, not their layers
    return defer || after_rows(T, pos, err);
}

bool PlainModel::forward_slots(const int32_t* tokens, const int* slot, const int64_t* pos,
                               const k::SamplerParams* sp, int T, int32_t* out, std::string& err) {
    const int n = batch_slot_count();
    if (T < 1 || T > opt_.max_t) { err = "plain model: the batch window is out of range"; return false; }
    for (int t = 0; t < T; ++t) {
        // a slot's second row (--batch-mtp): its draft, at the position after its token
        const bool second = t > 0 && slot[t] == slot[t - 1];
        if (slot[t] < 0 || slot[t] >= n || (t > 0 && slot[t] < slot[t - 1]) || pos[t] < 0 || pos[t] >= opt_.max_cells ||
            (second && (!batch_drafts() || pos[t] != pos[t - 1] + 1 || (t > 1 && slot[t - 2] == slot[t])))) {
            err = "plain model: a batch row's slot or position is out of range";
            return false;
        }
    }
    const float* hand = nullptr;
    for (PlainModel* st : chain()) {
        const OnDevice on(st->dev_);
        if (!st->stage_slots(tokens, slot, pos, sp, T, hand, err)) return false;
        hand = st->h_hand_;
    }
    PlainModel& t = tail();
    const OnDevice on(t.dev_);
    for (int i = 0; i < T; ++i) out[i] = ((volatile int32_t*) t.h_out_)[i];
    for (PlainModel* st : chain()) {   // the live session's drafting starts again at its next request
        st->cur_ = -1;
        st->drafts_.clear();
        st->mtp_h_set_ = false;
    }
    return true;
}

bool PlainModel::pipe_empty() const {
    return std::all_of(pipe_.begin(), pipe_.end(), [](const GroupWindow* w) { return w == nullptr; });
}

bool PlainModel::pipe_tick(GroupWindow* enter, GroupWindow*& done, std::string& err) {
    const std::vector<PlainModel*> st = chain();
    const size_t S = st.size();
    if (pipe_.size() != S) pipe_.assign(S, nullptr);
    done = nullptr;
    if (enter != nullptr) {   // a group's rows: its slots ascending, each once
        const int T = (int) enter->tokens.size();
        bool ok = T >= 1 && T <= opt_.max_t && enter->slot.size() == (size_t) T && enter->pos.size() == (size_t) T &&
                  enter->sp.size() == (size_t) T;
        for (int t = 0; ok && t < T; ++t)
            ok = enter->slot[(size_t) t] >= 0 && enter->slot[(size_t) t] < batch_slot_count() &&
                 (t == 0 || enter->slot[(size_t) t] > enter->slot[(size_t) t - 1]) && enter->pos[(size_t) t] >= 0 &&
                 enter->pos[(size_t) t] < opt_.max_cells;
        if (!ok) { err = "plain model: a group's batch window is out of range"; return false; }
    }
    for (size_t s = S; s-- > 1;) pipe_[s] = pipe_[s - 1];
    pipe_[0] = enter;
    std::vector<std::string> errs(S);
    std::vector<char> failed(S, 0);
    auto run = [&](size_t s) {
        GroupWindow* w = pipe_[s];
        if (w == nullptr) return;
        PlainModel* m = st[s];
        const OnDevice on(m->dev_);
        const int T = (int) w->tokens.size();
        if (!m->stage_slots(w->tokens.data(), w->slot.data(), w->pos.data(), w->sp.data(), T,
                            s == 0 ? nullptr : w->hand.data(), errs[s])) {
            failed[s] = 1;
            return;
        }
        if (s + 1 < S) {
            w->hand.assign(m->h_hand_, m->h_hand_ + (size_t) T * (size_t) m->hand_w_);
        } else {
            w->out.resize((size_t) T);
            for (int t = 0; t < T; ++t) w->out[(size_t) t] = ((volatile int32_t*) m->h_out_)[t];
        }
    };
    piping_ = true;
    {
        std::vector<std::thread> others;
        for (size_t s = 1; s < S; ++s)
            if (pipe_[s] != nullptr) others.emplace_back(run, s);
        run(0);
        for (auto& t : others) t.join();
    }
    piping_ = false;
    for (size_t s = 0; s < S; ++s)
        if (failed[s]) {
            err = errs[s];
            pipe_.assign(S, nullptr);
            return false;
        }
    done = pipe_[S - 1];
    pipe_[S - 1] = nullptr;
    for (PlainModel* m : st) {   // the live session's drafting starts again at its next request, as forward_slots'
        m->cur_ = -1;
        m->drafts_.clear();
        m->mtp_h_set_ = false;
    }
    return true;
}

uint64_t PlainModel::slots_key(const int* slot, int T) {
    uint64_t set = 0;
    for (int t = 0; t < T; ++t) {
        set |= uint64_t{1} << slot[t];
        if (t > 0 && slot[t] == slot[t - 1]) set |= uint64_t{1} << (32 + slot[t]);
    }
    return set;
}

bool PlainModel::batch_drafts() const {
    const PlainModel& t = tail();
    if (!opt_.batch_mtp || !t.has_mtp_ || t.n_mtp_ < 1 || t.glm_ || batch_slot_count() < 1) return false;
    if (!root_->bslot_snaps_) return false;
    for (const auto& st : root_->next_)
        if (!st->bslot_snaps_) return false;
    return true;
}

bool PlainModel::slot_rollback(int b, std::string& err) {
    if (b < 0 || b >= batch_slot_count()) { err = "plain model: no such batch slot"; return false; }
    for (PlainModel* st : chain()) {
        const OnDevice on(st->dev_);
        const SlotSession& ss = st->bslots_[(size_t) b];
        for (size_t g = 0; g < ss.gdn_snap.size(); ++g)
            if (ss.gdn_state[g] != nullptr &&
                (!strata::gpu::copy_async(ss.gdn_state[g], ss.gdn_snap[g], st->state_bytes_, st->cs_) ||
                 !strata::gpu::copy_async(ss.gdn_conv[g], ss.conv_snap[g], st->conv_bytes_, st->cs_))) {
                err = std::string("plain model: restoring a slot's recurrent states: ") + strata::gpu::last_error();
                return false;
            }
    }
    return true;
}

bool PlainModel::mtp_slots(const int* slot, const int* rows, const int32_t* tokens, const int64_t* pos,
                           const k::SamplerParams* sp, int T, int32_t* drafts, std::string& err) {
    if (!next_.empty()) {   // the MTP layer is the last stage's, as the trunk rows it reads
        const OnDevice on(tail().dev_);
        return tail().mtp_slots(slot, rows, tokens, pos, sp, T, drafts, err);
    }
    if (T < 1 || T > opt_.max_t || !batch_drafts()) { err = "plain model: the slots' drafts are out of range"; return false; }
    const int64_t N = spec_.n_embd;
    for (int i = 0; i < T; ++i)   // each row's trunk row, normed, from the batch window just run
        if (rows[i] < 0 || rows[i] >= opt_.max_t ||
            !strata::gpu::copy_async(mtp_h_ + (size_t) i * (size_t) N, hn_ + (size_t) rows[i] * (size_t) N,
                                     (size_t) N * 4, cs_)) {
            err = std::string("plain model: the MTP layer's input: ") + strata::gpu::last_error();
            return false;
        }
    const uint64_t key = slots_key(slot, T) | (uint64_t{1} << 63);   // (a key apart from a pattern of two rows a slot)
    row_slot_.assign(slot, slot + T);
    struct Clear {
        std::vector<int>& v;
        ~Clear() { v.clear(); }
    } clear{row_slot_};
    const bool eager = step_form_.form() == 1;
    if ((!eager && !capture(T, true, mbexec_[key], mbsegs_[key], err)) || !stage(tokens, T, 0, err, true, pos))
        return false;
    const int64_t ml = n_moe_ - 1;
    if (!(eager ? run_window(T, true, err) : launch_serve(mbexec_[key], mbsegs_[key], 1, &ml, T, err))) return false;
    // each slot's draft from its last row's logits (record_mtp_slots: row j of logits_ is the j-th slot's)
    int j = 0;
    for (int i = 0; i < T; ++i) {
        if (i + 1 < T && slot[i + 1] == slot[i]) continue;
        k::SamplerParams p;
        p.greedy = true;
        p.temperature = 0.0f;
        if (opt_.coupled_draft && !sp[j].greedy) {   // as mtp_run's: the window row at this position checks it
            p = sp[j];
            p.counter = (uint64_t) pos[i];
            p.logit_bias = nullptr;
            p.allowed = nullptr;
        }
        k::sample_tokens(logits_ + (size_t) j * (size_t) n_vocab_, 1, (int) n_vocab_, nullptr, 0, p, m_out_ + j, cs_);
        ++j;
    }
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain model: the slots' drafts: ") + strata::gpu::last_error();
        return false;
    }
    if (disp_.failed) {
        err = std::string("plain model: the CPU experts: ") + (disp_.fail ? disp_.fail : "failed");
        return false;
    }
    for (int i = 0; i < j; ++i) drafts[i] = ((volatile int32_t*) h_out_)[i];
    mtp_h_set_ = false;
    return true;
}

// A batch window is recorded once for each set of slots in it (their state's pointers are in the graph), the rows'
// tokens and positions staged as a window's
bool PlainModel::stage_slots(const int32_t* tokens, const int* slot, const int64_t* pos, const k::SamplerParams* sp,
                             int T, const float* hand_in, std::string& err) {
    const uint64_t set = slots_key(slot, T);
    row_slot_.assign(slot, slot + T);
    struct Clear {
        std::vector<int>& v;
        ~Clear() { v.clear(); }
    } clear{row_slot_};
    const bool eager = step_form_.form() == 1;
    if ((!eager && !capture(T, false, bexec_[set], bsegs_[set], err)) || !settle_window(err) ||
        !stage(tokens, T, 0, err, first(), pos))
        return false;
    if (hand_in != nullptr) std::memcpy(h_in_, hand_in, (size_t) T * (size_t) hand_w_ * 4);
    pcie_probe_before();
    const double pool0 = disp_.ms_plan + disp_.ms_actq + disp_.ms_jobs + disp_.ms_run;
    const auto t0 = std::chrono::steady_clock::now();
    if (!(eager ? run_window(T, false, err)
                : launch_serve(bexec_[set], bsegs_[set], moe_trunk_, moe_local_.data(), T, err)))
        return false;
    if (last())
        for (int t = 0; t < T; ++t) {   // each row with its request's sampling, its draw counted from its position
            k::SamplerParams p = sp[t];
            p.counter = (uint64_t) pos[t];
            k::sample_tokens(logits_ + (size_t) t * (size_t) n_vocab_, 1, (int) n_vocab_, nullptr, 0, p, m_out_ + t, cs_);
        }
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain model: the batch window: ") + strata::gpu::last_error();
        return false;
    }
    const double win_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    pcie_probe_after(win_ms, T);
    if (disp_.failed) {
        err = std::string("plain model: the CPU experts: ") + (disp_.fail ? disp_.fail : "failed");
        return false;
    }
    pool_ms_ += disp_.ms_plan + disp_.ms_actq + disp_.ms_jobs + disp_.ms_run - pool0;
    win_ms_ += win_ms;
    ++windows_;
    batch_rows_ += T;
    return after_rows(T, batch_rows_ - T, err);   // the adaptive rounds by the batch rows read
}

bool PlainModel::after_rows(int rows, int64_t pos0, std::string& err) {
    tokens_since_adapt_ += rows;
    if (disp_.usage.empty()) return true;
    add_route_rows(disp_.usage, disp_.route_rec, trunk_layers_, spec_.n_expert, opt_.max_t, spec_.n_expert_used, rows);
    // --batch-groups: the routing is counted, and no round starts beside the other stages' windows (as
    // Qwen3.8-Flash-Next's pipelined groups); the tier moves again once the slots are back to the serial windows
    if (root_->piping_) return true;
    if (this == root_ && adapt_auto_) adapt_rows_ += rows;
    if (this == root_ && adapt_auto_ && adapt_rows_ >= kAdaptDecideAt) {   // the whole chain's windows
        adapt_auto_ = false;
        double pool = 0, win = 0;
        for (PlainModel* st : chain()) {
            pool += st->pool_ms_;
            win += st->win_ms_;
        }
        // the experts the GPU does not hold are read from the files, some or all (no RAM copy, or one that does not
        // hold them all): a swap's blob is a file read too, and the period stays.  llm-jp-4.1 Q4_K_M at 20 GB of RAM
        // and 8 GB of VRAM on the B70 (2026-10-26, the pool's share 0.81): every 16 tokens 23.8 tok/s, every 4 20.1.
        int64_t pairs = 0, on_gpu = 0;
        for (PlainModel* st : chain()) {
            pairs += (int64_t) st->res_.size();
            on_gpu += (int64_t) std::count_if(st->res_.begin(), st->res_.end(), [](int32_t r) { return r >= 0; });
        }
        const bool from_files = !in_ram_ && (complement_ == nullptr || !complement_->complement_ready() ||
                                             complement_->resident_count() < pairs - on_gpu);
        // llm-jp-4.1 (qwen3moe) keeps every 16 tokens: after its 16K prompt on the B70 (8 GiB, the profile, 3 runs
        // each, 2026-10-26) every 4 tokens decoded 31.7 / 31.9 / 40.7 tok/s against 42.5 / 42.7 / 42.5 at 16 (the
        // engine before these rounds: 43.3 / 43.2 / 42.9); the counts the prompt seeded decay away 4 times as fast
        // a token, and scaling the decay to the period gave 36.3-37.9
        const double share = win > 0.0 ? pool / win : 0.0;
        if (spec_.arch == ModelArch::Qwen3Moe)
            std::fprintf(stderr, "xestrata plain: adaptive tier every %lld tokens: the CPU pool took %.0f%% of the verify "
                                 "windows, and this model keeps the period (--adapt-every sets it)\n",
                         (long long) adapt_period_, 100.0 * share);
        else
            set_adapt_period(auto_adapt_period(share, adapt_period_, from_files, "xestrata plain"));
    }
    // a round at fixed positions of the text (window-invariant swaps, #152), a layer split's stages in turn
    // (adapt_phase_), so that one's copies land before the next's
    const int64_t period = root_->adapt_period_;
    if (period > 0 && (pos0 + rows + adapt_phase_) / period != (pos0 + adapt_phase_) / period && !adapt(err)) return false;
    ram_adapt();
    return true;
}

void PlainModel::set_adapt_period(int64_t period) {
    adapt_period_ = period;
    for (size_t i = 0; i < next_.size(); ++i)
        next_[i]->adapt_phase_ = (int64_t) (i + 1) * std::max<int64_t>(period, 1) / (int64_t) (next_.size() + 1);
}

// STRATA_RAM_ADAPT, as generate.cpp: the RAM copy is chosen once, at load, so an expert it does not hold is read from
// the file every time the conversation routes to it.  With this, between windows and while no adaptive round's copies
// are in flight (they read the RAM copy), experts the windows keep routing to (decayed count 2 or more) take the places
// of experts of their layer that are not routed to (below 0.5), up to 4 pairs per layer, the largest difference first;
// out of a stage buffer when the incoming one is still there, else up to n of them with one blob read each, on this
// thread (ram_adapt_round).  One GPU only: a layer split's stages share the copy.
void PlainModel::ram_adapt() {
    static const int n = [] {
        const char* e = strata::env_get("RAM_ADAPT");
        return e != nullptr ? std::max(0, (int) std::strtol(e, nullptr, 10)) : 0;
    }();
    if (n <= 0 || complement_ == nullptr || !complement_->complement_ready() || disp_.usage.empty() ||
        !next_.empty() || root_ != this || !pending_.empty() || windows_ < 2)
        return;
    ram_adapt_round(disp_.usage.data(), res_.data(), n_moe_, spec_.n_expert, *complement_, n);
}

// The decode's PCIe share, measured: whether the GPU reading a share of the missed experts over PCIe beats the CPU
// computing them all depends on more than the start's times (pcie_split_m): with llm-jp-4.1 Q4_K_M's 4.6 MB experts at
// 6 GB on the B70 decode went 35.8 -> 32.9 tok/s with it, with GLM-5.3-Flash UD-IQ3_XXS's 8.5 MB ones 9.16 -> 9.46
// (2026-10-19).  So the first windows take it on and off in blocks of kBlock, kRounds times each, and the faster time a
// token stays.
namespace {
constexpr int kPcieBlock = 8, kPcieRounds = 3;
}

void PlainModel::set_pcie_frac(double f) {
    for (PlainModel* st : chain()) {
        if (f >= 0.0) {
            if (!st->pcie_req_) {   // (the decode's first windows' choice waits for the next request)
                st->pcie_split_own_ = st->disp_.pcie_split;
                st->pcie_num_own_ = st->disp_.pcie_num;
                st->pcie_probe_own_ = st->pcie_probe_;
                st->pcie_probe_ = -1;
                st->pcie_req_ = true;
            }
            st->disp_.pcie_split = nullptr;
            st->disp_.pcie_num = (int) std::lround(std::clamp(f, 0.0, 1.0) * 256.0);
        } else if (st->pcie_req_) {
            st->disp_.pcie_split = st->pcie_split_own_;
            st->disp_.pcie_num = st->pcie_num_own_;
            st->pcie_probe_ = st->pcie_probe_own_;
            st->pcie_req_ = false;
        }
    }
}

void PlainModel::pcie_probe_before() {
    if (pcie_probe_ < 0) return;
    disp_.pcie_num = (pcie_probe_ / kPcieBlock) % 2 == 0 ? 256 : 0;
}

void PlainModel::pcie_probe_after(double ms, int T) {
    if (pcie_probe_ < 0) return;
    const int on = (pcie_probe_ / kPcieBlock) % 2 == 0 ? 1 : 0;
    pcie_probe_ms_[on] += ms;
    pcie_probe_tok_[on] += T;
    if (++pcie_probe_ < 2 * kPcieBlock * kPcieRounds) return;
    const double a = pcie_probe_ms_[1] / (double) std::max<int64_t>(pcie_probe_tok_[1], 1);
    const double b = pcie_probe_ms_[0] / (double) std::max<int64_t>(pcie_probe_tok_[0], 1);
    disp_.pcie_num = a <= b ? 256 : 0;
    pcie_probe_ = -1;
    std::fprintf(stderr, "xestrata plain: the decode's PCIe share: %.2f ms a token with it, %.2f without: %s\n", a, b,
                 a <= b ? "on" : "off");
}

bool PlainModel::mtp_run(int T, const int32_t* tokens, int64_t pos, int32_t* draft, std::string& err) {
    if (!stage(tokens, T, pos, err)) return false;   // the swaps: settle_window
    const int64_t ml = n_moe_ - 1;
    const bool eager = step_form_.form() == 1;
    const bool res = !eager && res_window(1, &ml, err);
    if (!err.empty() || (!res && !record(T, true, err))) return false;
    if (!(eager ? run_window(T, true, err)
                : res ? launch_res(T, true, mrexec_[(size_t) T], 1, &ml, err)
                      : launch_serve(mexec_[(size_t) T], msegs_[(size_t) T], 1, &ml, T, err)))
        return false;
    // the drafts are greedy; --coupled-draft samples them as the window will (row t at position p + t draws with
    // counter p + t): the draft from the row at pos + T - 1 is checked by the window row at that position, with the
    // request's chain (a request with penalties, a bias or a grammar decodes without drafts)
    k::SamplerParams sp;
    sp.greedy = true;
    sp.temperature = 0.0f;
    if (opt_.coupled_draft && !sampler_.greedy) {
        sp = sampler_;
        sp.counter = (uint64_t) (pos + T - 1);
        sp.logit_bias = nullptr;
        sp.allowed = nullptr;
    }
    const int nv = (int) (dhead_ != nullptr ? n_dvocab_ : n_vocab_);
    k::sample_tokens(logits_, 1, nv, nullptr, 0, sp, m_out_, cs_);
    // the draft's probability, for the window rule and --pipeline-windows 2's estimate
    const bool probs = opt_.spec_window != "all" || pipelined_spec();
    if (probs) k::row_top_prob(logits_, 1, nv, m_out_, m_prob_, cs_);
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain model: the MTP layer: ") + strata::gpu::last_error();
        return false;
    }
    res_count();
    if (disp_.failed) {
        err = std::string("plain model: the CPU experts: ") + (disp_.fail ? disp_.fail : "failed");
        return false;
    }
    *draft = ((volatile int32_t*) h_out_)[0];
    if (dhead_ != nullptr) *draft = *draft >= 0 && *draft < n_dvocab_ ? dvocab_[(size_t) *draft] : -1;
    if (h_prob_ != nullptr) dprob_.push_back(probs ? ((volatile float*) h_prob_)[0] : 1.0f);
    return true;
}

bool PlainModel::draft_more(int n) const {
    if (n < 1 || n > (int) dprob_.size()) return true;
    if (req_spec_min_p_ >= 0.0) return dprob_[(size_t) n - 1] >= (float) req_spec_min_p_;   // the request's floor
    if (opt_.spec_window == "auto") return policy_.mtp_extend(dprob_.data(), n, n_mtp_ + 1);
    if (opt_.spec_window == "min-p") return dprob_[(size_t) n - 1] >= (float) opt_.spec_min_p;
    return true;
}

bool PlainModel::mtp_chain(int want, int64_t pos, std::string& err, bool all) {
    // each step reads the draft before it at the next position, and the previous step's normed row (mtp_h_)
    while ((int) drafts_.size() < want && (all || draft_more((int) drafts_.size()))) {
        int32_t d = -1;
        const int64_t p = pos + (int64_t) drafts_.size() - 1;
        if (p >= opt_.max_cells) return true;
        if (!mtp_run(1, &drafts_.back(), p, &d, err)) return false;
        drafts_.push_back(d);
    }
    return true;
}

bool PlainModel::spec_start(int32_t first, int64_t pos, std::string& err) {
    if (!next_.empty()) {   // the MTP layer is the last stage's
        const OnDevice on(tail().dev_);
        return tail().spec_start(first, pos, err);
    }
    if (!spec_end(err)) return false;
    drafts_.clear();
    dprob_.clear();
    cur_ = first;
    cur_pos_ = pos;
    if (!has_mtp_ || n_mtp_ < 1) return true;
    // the trunk's row before `first`: prompt() leaves it in mtp_h_, a window in hn_
    if (!mtp_h_set_ &&
        !strata::gpu::copy(mtp_h_, hn_ + (size_t) (last_T_ - 1) * (size_t) spec_.n_embd, (size_t) spec_.n_embd * 4)) {
        err = "plain model: the MTP layer's input";
        return false;
    }
    if (pos >= opt_.max_cells) return true;
    int32_t d = -1;
    if (!mtp_run(1, &first, pos, &d, err)) return false;
    drafts_.push_back(d);
    // --pipeline-windows 2: the chain reads on past a window's drafts to the next window's guess
    return pipelined_spec() ? pl_chain(pos + 1, err) : mtp_chain(n_mtp_, pos + 1, err);
}

bool PlainModel::spec_step(const std::vector<int32_t>& stop, int64_t limit, int32_t* emitted, int& n_emit, float* logits,
                           std::string& err, const int32_t* follow, const int32_t* ext, int n_ext, int match) {
    if (!next_.empty()) {   // the drafts are the last stage's; its window goes through every stage (root_->forward)
        const OnDevice on(tail().dev_);
        return tail().spec_step(stop, limit, emitted, n_emit, logits, err, follow, ext, n_ext, match);
    }
    if (pipelined_spec()) {   // --pipeline-windows 2: a window's MTP drafts alone pipeline; else a pending guess goes
        if (ext == nullptr && follow == nullptr && logits == nullptr)   // (no window masks: none left from before)
            return (!window_masks_ || set_allowed_rows(nullptr, 0, err)) &&
                   spec_step_pl(stop, limit, emitted, n_emit, err);
        if (!spec_end(err)) return false;
    }
    n_emit = 0;
    if (cur_ < 0) { err = "plain model: spec_step before spec_start"; return false; }
    // the window: the pending token and the drafts that fit the context and the limit
    int nd = (int) std::min<int64_t>({(int64_t) drafts_.size(), limit - 1, opt_.max_cells - cur_pos_ - 1});
    nd = std::max(nd, 0);
    // the window rule (--spec-window) over the drafts' probabilities
    float probs[strata::spec::DraftPolicy::kMaxT + 1] = {};
    for (int j = 0; j < nd && j < (int) dprob_.size() && j < strata::spec::DraftPolicy::kMaxT; ++j) probs[j] = dprob_[(size_t) j];
    policy_.set_context(cur_pos_);   // STRATA_SPEC_COST: the round costs of this context length
    if (opt_.spec_window == "auto") nd = std::min(nd, policy_.mtp_window(probs, n_mtp_ + 1) - 1);
    else if (opt_.spec_window == "min-p") {
        int keep = 0;
        while (keep < nd && probs[keep] >= (float) opt_.spec_min_p) ++keep;
        nd = keep;
    }
    int32_t win[k::cpu::MAXT];
    int32_t out[k::cpu::MAXT];
    win[0] = cur_;
    for (int i = 0; i < nd; ++i) win[i + 1] = drafts_[(size_t) i];
    // --suffix-draft / --lookup-chain: the lookup's drafts where they are more and agree with the MTP's first
    const int cap = (int) std::min<int64_t>({(int64_t) n_ext, (int64_t) window_drafts(), limit - 1,
                                             opt_.max_cells - cur_pos_ - 1, (int64_t) k::cpu::MAXT - 1});
    bool from_ext = ext != nullptr && cap > nd && (nd == 0 || drafts_.empty() || ext[0] == drafts_[0]);
    if (from_ext) {
        const strata::spec::DraftPolicy::Pick pk = policy_.choose(nd + 1, cap, match);
        from_ext = pk.lookup && pk.t - 1 > nd;
        if (from_ext) {
            nd = std::min(cap, pk.t - 1);
            for (int i = 0; i < nd; ++i) win[i + 1] = ext[i];
        }
    }
    const int T = nd + 1;
    if (window_masks_ && !window_masks_(win, T, err)) return false;
    const auto t0 = std::chrono::steady_clock::now();
    if (!root_->forward(win, T, cur_pos_, out, logits, err, true)) return false;
    const auto t1 = std::chrono::steady_clock::now();
    spec_ms_check_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (follow != nullptr)   // --spec-follow: the continuation in place of the window's picks
        for (int j = 0; j < T; ++j) out[j] = follow[j];
    // accepted: the drafts the window's own picks agree with, up to the first stop token among the emitted
    int n = 0;
    while (n < nd && out[n] == win[n + 1]) ++n;
    for (int j = 0; j < n; ++j)
        if (std::find(stop.begin(), stop.end(), out[j]) != stop.end()) { n = j; break; }
    ++spec_windows_;
    spec_drafted_ += nd;
    spec_accepted_ += n;
    for (PlainModel* st : root_->chain()) {   // the rows the window keeps: the accepted drafts and the token after them
        const OnDevice on(st->dev_);
        if (!st->after_rows(n + 1, cur_pos_, err)) return false;
    }
    if (opt_.spec_window == "auto" && !from_ext) policy_.observe_mtp_drafts(probs, T, n);
    for (int j = 0; j <= n; ++j) emitted[j] = out[j];
    n_emit = n + 1;
    // the recurrent states back to the last accepted row's (the attention's later cells are rewritten in turn)
    if (n < T - 1 && !root_->rollback(n, err)) return false;
    // the MTP layer catches up over the accepted rows (row t: the window's trunk row t and the token after it), whose
    // last row drafts the token after the new pending one; then the chain
    const int64_t pos = cur_pos_;
    cur_ = out[n];
    cur_pos_ = pos + n + 1;
    drafts_.clear();
    dprob_.clear();
    // the round's time (the window, the rollback, the drafts) for the window rule's costs, once the drafts are made
    // (the lookup's rounds too: the rule needs their costs and their acceptance by match length)
    auto observe = [&] {
        if (opt_.spec_window == "auto" || from_ext || ext != nullptr)
            policy_.observe(from_ext, T, n, from_ext ? match : 0,
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    };
    if (cur_pos_ >= opt_.max_cells || !has_mtp_ || n_mtp_ < 1) { observe(); return true; }   // (no MTP layer)
    if (!strata::gpu::copy_async(mtp_h_, hn_, (size_t) (n + 1) * (size_t) spec_.n_embd * 4, cs_)) {
        err = std::string("plain model: the MTP layer's input: ") + strata::gpu::last_error();
        return false;
    }
    int32_t d = -1;
    if (!mtp_run(n + 1, out, pos + 1, &d, err)) return false;
    drafts_.push_back(d);
    const bool ok = mtp_chain(n_mtp_, cur_pos_ + 1, err);
    spec_ms_draft_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    observe();
    return ok;
}

bool PlainModel::pipelined_spec() const {
    return opt_.pipeline_windows >= 2 && root_ != this && root_->next_.size() == 1 && has_mtp_ && n_mtp_ >= 1 && !glm_ &&
           root_->pl_ready_;
}

// --pipeline-windows 2: until the windows' costs are measured (pl_gate_), a guessed window starts only when the chain's
// estimate that the window before is taken whole, with the guess its next token, is at least this
// (STRATA_PIPELINE_THETA with STRATA_PIPELINE_DEBUG=1, as Qwen3.8-Flash-Next's, whose default is 0.2).  Qwen3.6-35B-A3B
// on a B70 + RTX 4070 split at layer 20, the experts' places fixed, serial 187.9 tok/s for a copy reply and 133.9 for
// a story at temperature 0.8: 240.4 and 124.5 at 0.2, 245.6 and 129.8 at 0.5 (2026-11-01; the same text each time).
// Measured on that pair only.
float PlainModel::pl_theta() {
    static const float t = [] {
        const char* dbg = strata::env_get("PIPELINE_DEBUG");
        const char* v = dbg != nullptr && std::strcmp(dbg, "1") == 0 ? strata::env_get("PIPELINE_THETA") : nullptr;
        return v != nullptr ? (float) std::strtod(v, nullptr) : 0.5f;
    }();
    return t;
}

bool PlainModel::pl_chain(int64_t pos1, std::string& err) {
    if (!mtp_chain(n_mtp_, pos1, err, true)) return false;
    float pa = 1.0f;
    for (size_t i = 0; i < (size_t) n_mtp_ && i < dprob_.size(); ++i) pa *= dprob_[i];
    if (!pl_gate_.extend(pa)) return true;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = mtp_chain(2 * n_mtp_ + 1, pos1, err, true);
    pl_gate_.observe_extend(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return ok;
}

bool PlainModel::pl_save(std::string& err) {
    for (size_t g = 0; g < gdn_state_.size(); ++g) {
        if (gdn_state_[g] == nullptr) continue;
        bool ok = strata::gpu::copy_async(pl_pre_state_[g], gdn_state_[g], state_bytes_, cs_) &&
                  strata::gpu::copy_async(pl_pre_conv_[g], gdn_conv_[g], conv_bytes_, cs_);
        for (int t = 0; t < n_snap_ && ok; ++t)
            ok = strata::gpu::copy_async(pl_bak_state_[g][(size_t) t], gdn_snap_[g][(size_t) t], state_bytes_, cs_) &&
                 strata::gpu::copy_async(pl_bak_conv_[g][(size_t) t], conv_snap_[g][(size_t) t], conv_bytes_, cs_);
        if (!ok) {
            err = std::string("plain model: keeping the recurrent states for a guess: ") + strata::gpu::last_error();
            return false;
        }
    }
    pl_bak_valid_ = true;
    return true;
}

bool PlainModel::pl_restore(int n, int T, std::string& err) {
    for (size_t g = 0; g < gdn_state_.size(); ++g) {
        if (gdn_state_[g] == nullptr) continue;
        // row n: from the copies when a guess ran over the window's own; the whole window: the state it left
        const float* st = n >= T - 1 ? (pl_bak_valid_ ? pl_pre_state_[g] : nullptr)
                                     : pl_bak_valid_ ? pl_bak_state_[g][(size_t) n] : gdn_snap_[g][(size_t) n];
        const float* cv = n >= T - 1 ? (pl_bak_valid_ ? pl_pre_conv_[g] : nullptr)
                                     : pl_bak_valid_ ? pl_bak_conv_[g][(size_t) n] : conv_snap_[g][(size_t) n];
        if (st == nullptr) continue;   // the whole window, no guess after it: the state is that window's already
        if (!strata::gpu::copy_async(gdn_state_[g], st, state_bytes_, cs_) ||
            !strata::gpu::copy_async(gdn_conv_[g], cv, conv_bytes_, cs_)) {
            err = std::string("plain model: restoring the recurrent states: ") + strata::gpu::last_error();
            return false;
        }
    }
    pl_bak_valid_ = false;
    return true;
}

bool PlainModel::spec_end(std::string& err) {
    if (!next_.empty()) {
        const OnDevice on(tail().dev_);
        return tail().spec_end(err);
    }
    if (pl_g_.empty() || root_ == this) return true;
    // the first stage is a guessed window ahead of the accepted tokens: back to the state the window before it left
    PlainModel& s0 = *root_;
    const OnDevice on(s0.dev_);
    for (size_t g = 0; g < s0.gdn_state_.size(); ++g)
        if (s0.gdn_state_[g] != nullptr &&
            (!strata::gpu::copy_async(s0.gdn_state_[g], s0.pl_pre_state_[g], s0.state_bytes_, s0.cs_) ||
             !strata::gpu::copy_async(s0.gdn_conv_[g], s0.pl_pre_conv_[g], s0.conv_bytes_, s0.cs_))) {
            err = std::string("plain model: giving a guessed window back: ") + strata::gpu::last_error();
            return false;
        }
    s0.pl_bak_valid_ = false;
    pl_g_.clear();
    pl_hand_.clear();
    pl_pos_ = -1;
    return strata::gpu::stream_sync(s0.cs_) || (err = strata::gpu::last_error(), false);
}

// --pipeline-windows 2 over two stages (the last's side; root_ is the first).  The MTP chain reads on past this
// window's drafts: when the window holds whole and the token after it is the chain's next one, the next window is
// that token and the chain's drafts after it - a guess, which the first stage computes while the last checks this
// window.  The guess held: the next window's first stage is done (and the drafter goes on from the chain's end, as
// Qwen3.8-Flash-Next's draft layer runs on through the window in flight).  It did not: the first stage goes back to
// the accepted rows (the copies pl_save kept), and the drafter catches up over them as spec_step's.  The tokens are
// the windows' own picks, as spec_step's.
bool PlainModel::spec_step_pl(const std::vector<int32_t>& stop, int64_t limit, int32_t* emitted, int& n_emit,
                              std::string& err) {
    PlainModel& s0 = *root_;
    n_emit = 0;
    if (cur_ < 0) { err = "plain model: spec_step before spec_start"; return false; }
    const int K = n_mtp_;
    // the window rule (--spec-window, a request's spec_min_p) over drafts' probabilities: how many of `avail` to check
    auto rule = [&](const float* p, int avail) {
        int keep = avail;
        if (req_spec_min_p_ >= 0.0 || opt_.spec_window == "min-p") {
            const double floor = req_spec_min_p_ >= 0.0 ? req_spec_min_p_ : opt_.spec_min_p;
            keep = 0;
            while (keep < avail && p[keep] >= (float) floor) ++keep;
        } else if (opt_.spec_window == "auto") {
            keep = std::min(avail, policy_.mtp_window(p, K + 1) - 1);
        }
        return std::max(keep, 0);
    };
    auto prob = [&](size_t i) { return i < dprob_.size() ? dprob_[i] : 1.0f; };
    const bool from_guess = !pl_g_.empty() && pl_pos_ == cur_pos_ && pl_g_[0] == cur_;
    if (!pl_g_.empty() && !from_guess && !spec_end(err)) return false;
    policy_.set_context(cur_pos_);
    int nd = 0;
    float probs[strata::spec::DraftPolicy::kMaxT + 1] = {};
    if (from_guess) {
        nd = (int) pl_g_.size() - 1;
    } else {
        nd = (int) std::min<int64_t>({(int64_t) drafts_.size(), (int64_t) K, limit - 1, opt_.max_cells - cur_pos_ - 1});
        nd = std::max(nd, 0);
        for (int j = 0; j < nd && j < strata::spec::DraftPolicy::kMaxT; ++j) probs[j] = prob((size_t) j);
        nd = rule(probs, nd);
    }
    for (int j = 0; j < nd && j < strata::spec::DraftPolicy::kMaxT; ++j) probs[j] = prob((size_t) j);
    const int T = nd + 1;
    int32_t win[k::cpu::MAXT];
    win[0] = cur_;
    for (int i = 0; i < nd; ++i) win[i + 1] = drafts_[(size_t) i];
    // the next window's guess: this one whole, then the chain's next token and its drafts after it
    std::vector<int32_t> guess;
    float p_on = 0.0f;
    if ((int) drafts_.size() > nd && limit - T >= 1 && cur_pos_ + T < opt_.max_cells - 1) {
        int kg = (int) std::min<int64_t>({(int64_t) drafts_.size() - nd - 1, (int64_t) K, limit - T - 1,
                                          opt_.max_cells - (cur_pos_ + T) - 1});
        kg = std::max(kg, 0);
        float gp[strata::spec::DraftPolicy::kMaxT + 1] = {};
        for (int j = 0; j < kg && j < strata::spec::DraftPolicy::kMaxT; ++j) gp[j] = prob((size_t) nd + 1 + (size_t) j);
        kg = rule(gp, kg);
        // the estimate that this window is taken whole and the guess's first token is its next one
        p_on = prob((size_t) nd);
        for (int j = 0; j < nd; ++j) p_on *= probs[j];
        if (pl_gate_.start(p_on)) guess.assign(drafts_.begin() + nd, drafts_.begin() + nd + 1 + kg);
    }
    const auto t0 = std::chrono::steady_clock::now();
    // the first stage's part of this window (done already when it was the guess)
    std::vector<float> hand;
    if (from_guess) {
        hand.swap(pl_hand_);
    } else {
        const OnDevice on(s0.dev_);
        if (!s0.stage_forward(win, T, cur_pos_, nullptr, err, true)) return false;
        hand.assign(s0.h_hand_, s0.h_hand_ + (size_t) T * (size_t) s0.hand_w_);
        s0.pl_bak_valid_ = false;
        pl_gate_.observe_front(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    pl_g_.clear();
    // the guess on the first stage beside this window on the last
    std::string gerr;
    bool gfail = false;
    const auto tb = std::chrono::steady_clock::now();
    s0.piping_ = true;
    std::thread front;
    if (!guess.empty())
        front = std::thread([&] {
            const OnDevice on(s0.dev_);
            pl_rec_w_ = s0.disp_.route_rec;   // this window's routing, before the guess's takes its place
            if (!s0.pl_save(gerr) || !s0.stage_forward(guess.data(), (int) guess.size(), cur_pos_ + T, nullptr, gerr, true)) {
                gfail = true;
                return;
            }
            pl_hand_.assign(s0.h_hand_, s0.h_hand_ + guess.size() * (size_t) s0.hand_w_);
        });
    int32_t out[k::cpu::MAXT];
    bool ok = true;
    {
        const OnDevice on(dev_);
        ok = stage_forward(win, T, cur_pos_, hand.data(), err, true);
        if (ok)
            for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    }
    if (front.joinable()) front.join();
    s0.piping_ = false;
    if (!ok) return false;
    if (gfail) { err = gerr; return false; }
    const auto t1 = std::chrono::steady_clock::now();
    pl_gate_.observe_back(std::chrono::duration<double, std::milli>(t1 - tb).count(), !guess.empty());
    spec_ms_check_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
    // accepted: the drafts the window's own picks agree with, up to the first stop token among the emitted
    int n = 0;
    while (n < nd && out[n] == win[n + 1]) ++n;
    for (int j = 0; j < n; ++j)
        if (std::find(stop.begin(), stop.end(), out[j]) != stop.end()) { n = j; break; }
    ++spec_windows_;
    spec_drafted_ += nd;
    spec_accepted_ += n;
    // the rows the window keeps: the accepted drafts and the token after them (the first stage's from this window's
    // routing records, which the guess's took the place of; the guess's back for when it is a window)
    if (!guess.empty()) {
        pl_rec_g_.swap(s0.disp_.route_rec);
        s0.disp_.route_rec.swap(pl_rec_w_);
    }
    for (PlainModel* st : root_->chain()) {
        const OnDevice on(st->dev_);
        if (!st->after_rows(n + 1, cur_pos_, err)) return false;
    }
    if (!guess.empty()) s0.disp_.route_rec.swap(pl_rec_g_);
    if (opt_.spec_window == "auto") policy_.observe_mtp_drafts(probs, T, n);
    for (int j = 0; j <= n; ++j) emitted[j] = out[j];
    n_emit = n + 1;
    // the last stage back to the last accepted row's states; the first to them, unless the guess holds
    if (n < T - 1)
        for (size_t g = 0; g < gdn_state_.size(); ++g)
            if (gdn_state_[g] != nullptr &&
                (!strata::gpu::copy_async(gdn_state_[g], gdn_snap_[g][(size_t) n], state_bytes_, cs_) ||
                 !strata::gpu::copy_async(gdn_conv_[g], conv_snap_[g][(size_t) n], conv_bytes_, cs_))) {
                err = std::string("plain model: restoring the recurrent states: ") + strata::gpu::last_error();
                return false;
            }
    const bool held = !guess.empty() && n == T - 1 && out[n] == guess[0];
    if (!guess.empty()) pl_gate_.observe_guess(p_on, held);
    {
        const OnDevice on(s0.dev_);
        if (held) s0.pl_bak_valid_ = false;   // its last window is the guess now (its rows' copies in its snapshots)
        else if (!s0.pl_restore(n, T, err)) return false;
    }
    const int64_t pos = cur_pos_;
    cur_ = out[n];
    cur_pos_ = pos + n + 1;
    auto observe = [&] {
        if (opt_.spec_window == "auto")
            policy_.observe(false, T, n, 0,
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    };
    if (held) {
        pl_g_ = guess;
        pl_pos_ = cur_pos_;
        const size_t drop = (size_t) nd + 1;
        drafts_.erase(drafts_.begin(), drafts_.begin() + (std::ptrdiff_t) drop);
        dprob_.erase(dprob_.begin(), dprob_.begin() + (std::ptrdiff_t) std::min(drop, dprob_.size()));
        if (cur_pos_ >= opt_.max_cells) { observe(); return true; }
        if (drafts_.empty()) {   // the chain ended at the guess's first token: it reads that one next
            int32_t d = -1;
            if (!mtp_run(1, &cur_, cur_pos_, &d, err)) return false;
            drafts_.push_back(d);
        }
        const bool okc = pl_chain(cur_pos_ + 1, err);
        spec_ms_draft_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        observe();
        return okc;
    }
    // as spec_step: the drafter catches up over the accepted rows (row t: the window's trunk row t and the token after
    // it), and drafts on past the next window to its guess
    drafts_.clear();
    dprob_.clear();
    if (cur_pos_ >= opt_.max_cells) { observe(); return true; }
    if (!strata::gpu::copy_async(mtp_h_, hn_, (size_t) (n + 1) * (size_t) spec_.n_embd * 4, cs_)) {
        err = std::string("plain model: the MTP layer's input: ") + strata::gpu::last_error();
        return false;
    }
    int32_t d = -1;
    if (!mtp_run(n + 1, out, pos + 1, &d, err)) return false;
    drafts_.push_back(d);
    const bool okc = pl_chain(cur_pos_ + 1, err);
    spec_ms_draft_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    observe();
    return okc;
}

bool PlainModel::rollback(int n, std::string& err) {
    for (PlainModel* st : chain()) {
        const OnDevice on(st->dev_);
        for (size_t g = 0; g < st->gdn_state_.size(); ++g)
            if (st->gdn_state_[g] != nullptr &&
                (!strata::gpu::copy_async(st->gdn_state_[g], st->gdn_snap_[g][(size_t) n], st->state_bytes_, st->cs_) ||
                 !strata::gpu::copy_async(st->gdn_conv_[g], st->conv_snap_[g][(size_t) n], st->conv_bytes_, st->cs_))) {
                err = std::string("plain model: restoring the recurrent states: ") + strata::gpu::last_error();
                return false;
            }
    }
    return true;
}

// One of `layer`'s free slots takes `expert`, read from the expert source (blocking: between windows, or at load).
bool PlainModel::fill_slot(int64_t layer, int32_t expert, std::string& err) {
    auto& f = free_[(size_t) layer];
    if (f.empty()) { err = "plain model: no free slot in the layer"; return false; }
    const int32_t slot = f.back();
    if (!src_->copy_blob(layer, expert, fill_tmp_) ||
        !cache_.fill_slot_blocking(slot, fill_tmp_, err, (int64_t) k::cpu::expert_layout().blob_bytes(layer))) {
        if (err.empty()) err = "plain model: an expert could not be read for the VRAM tier";
        return false;
    }
    f.pop_back();
    res_[(size_t) (layer * spec_.n_expert + expert)] = slot;
    return true;
}

bool PlainModel::save_profile(const std::string& path, std::string& err) const {
    if (dense_) { err = "plain model: a dense model has no experts to profile"; return false; }
    const int64_t L = n_moe_, E = spec_.n_expert;
    std::vector<uint8_t> resident((size_t) (L * E), 0);
    std::vector<double> heat = heat_;
    for (size_t i = 0; i < res_.size(); ++i) resident[i] = res_[i] >= 0;
    for (const auto& st : next_)   // every stage's tier and counts (each counts its own layers)
        for (size_t i = 0; i < resident.size(); ++i) {
            resident[i] = resident[i] || st->res_[i] >= 0;
            heat[i] += st->heat_[i];
        }
    return write_expert_profile(path, L, E, rank_learned_profile(L, E, resident, heat, prior_), err);
}

// XeStrata's adaptive tier (generate.cpp's `adapt`): per MoE layer, a missing expert routed at least twice (decayed
// counts) replaces the least-routed resident one when it leads by 1.5, or takes a free slot; at most adapt_swaps a
// round, the largest gains first; then the counts decay.  Runs between windows, so no window reads a slot it changes.
bool PlainModel::adapt(std::string& err, int64_t max_swaps, int64_t max_evict) {
    const int64_t L = n_moe_, E = spec_.n_expert;
    // with a layer split the stages share the RAM copy's exchange buffers, and a commit takes every staged exchange:
    // the other stages' copies land first (their rounds come a few windows apart, so they have as a rule)
    for (PlainModel* st : root_->chain())
        if (st != this && !st->pending_.empty()) {
            const OnDevice on(st->dev_);
            if (!st->settle_swaps(true, err)) return false;
        }
    if (!settle_swaps(max_swaps > 0, err)) return false;
    if (!pending_.empty()) return true;   // the last round's copies are still landing: no new round yet
    using Swap = TierSwap;   // out < 0: a free slot
    std::vector<Swap> swaps = choose_swaps(
        disp_.usage.data(), res_.data(), L, E, opt_.adapt_min_gain,
        [&](int64_t l) { return (int64_t) free_[(size_t) l].size(); }, [](int64_t, int32_t) { return false; });
    // with the RAM copy of what VRAM does not hold (complement_), an expert evicted for one the RAM has goes back into
    // the RAM in its place through an exchange buffer; past the buffers it would be read from the files from then on
    // while the one that came in is held twice, so such a swap waits for a round with room.  The round after a prompt
    // (up to 2,048 swaps against 96 buffers) sent most of GLM-5.3-Flash's profiled tier to the drive: decode 9.5 tok/s
    // against 13.4 without that round (B70, UD-IQ3_XXS, 2026-10-20); all the experts in RAM, nothing is lost.
    if (complement_ != nullptr) {
        int64_t room = complement_->exchange_capacity();
        swaps.erase(std::remove_if(swaps.begin(), swaps.end(), [&](const Swap& w) {
                        if (w.out < 0 || !complement_->has_resident(w.layer, w.in) ||
                            complement_->has_resident(w.layer, w.out))
                            return false;
                        return room-- <= 0;
                    }),
                    swaps.end());
    }
    // and at most adapt_mib_per_token of copies for each token the windows read since the last round
    const int64_t cap = max_swaps > 0 ? max_swaps
                                      : adapt_swap_budget(opt_.adapt_swaps, opt_.adapt_mib_per_token, tokens_since_adapt_);
    tokens_since_adapt_ = 0;
    if (max_evict >= 0) {
        int64_t evict = max_evict;
        swaps.erase(std::remove_if(swaps.begin(), swaps.end(), [&](const Swap& w) { return w.out >= 0 && evict-- <= 0; }),
                    swaps.end());
    }
    if ((int64_t) swaps.size() > cap) swaps.resize((size_t) cap);
    // the RAM tier of what VRAM does not hold: an evicted expert goes back into the RAM copy, in the place of the one
    // that comes in (FileExpertSource's exchanges), read back from its slot before the slot is written; the copy
    // keeps it from the exchange buffer until the new one has landed (settle_swaps commits them)
    if (complement_ != nullptr &&
        !resident_stage_swaps(*complement_, res_, E, swaps, [&](int32_t) { return SwapHome{&cache_, swap_cs_, dev_}; })) {
        err = std::string("plain model: the adaptive tier's copy back: ") + strata::gpu::last_error();
        return false;
    }
    for (const Swap& w : swaps) {
        if (w.out < 0) {
            if (free_[(size_t) w.layer].empty()) continue;
        } else {   // the evicted expert's slot becomes the layer's free one, which the fill takes
            const int32_t slot = res_[(size_t) (w.layer * E + w.out)];
            res_[(size_t) (w.layer * E + w.out)] = -1;
            free_[(size_t) w.layer].push_back(slot);
        }
        // from the pinned RAM copy on the copy stream, beside the next windows: the evicted expert is out of the plan
        // already, and the new one goes in once the copy is done (settle_swaps); else read into a slot and wait
        const uint8_t* hb = src_->transient(w.layer, w.in) ? nullptr : src_->blob(w.layer, w.in);
        if (hb != nullptr && (strata::gpu::is_host_usm(hb) || (dev_ >= 0 && src_->registered_on(dev_)))) {
            auto& f = free_[(size_t) w.layer];
            const int32_t slot = f.back();
            if (!strata::gpu::copy_async(cache_.device_slot(slot), hb, k::cpu::expert_layout().blob_bytes(w.layer),
                                         swap_cs_)) {
                err = std::string("plain model: the adaptive tier's copy: ") + strata::gpu::last_error();
                return false;
            }
            f.pop_back();
            pending_.push_back({w.layer, w.in, slot});
        } else if (!fill_slot(w.layer, w.in, err)) {
            return false;
        }
        ++swaps_total_;
    }
    pending_age_ = 0;
    if (!pending_.empty() && !strata::gpu::event_record(swap_ev_, swap_cs_)) {
        err = std::string("plain model: the adaptive tier's copies: ") + strata::gpu::last_error();
        return false;
    }
    for (float& v : disp_.usage) v *= opt_.adapt_decay;
    return opt_.adapt_async || settle_swaps(true, err);
}

// 3 windows, not Qwen3.8-Flash-Next's 2: a draft's MTP run and the drafts' head between two verify windows are short
// here, and at 2 the window waited for the copies.  RTX 4070, Ornith-1.5 Q6_K (256 tokens, 2026-10-26): 2 windows
// 84.9-86.8 tok/s, 3 90.2-91.3, the copies taken as soon as they landed (before C7, not reproducible) 90.3; Qwen3.6
// UD-Q4_K_XL in 8 GB of the B70 and llm-jp-4.1 equal at 3 with the prompt's round below (92.9 and 37.5 tok/s).
constexpr int kPlainAdaptLag = 3;

bool PlainModel::settle_window(std::string& err) {
    if (pending_.empty()) return settle_swaps(false, err);   // a round of blocking fills: its exchanges commit
    const int lag = adapt_lag(kPlainAdaptLag);
    if (++pending_age_ < lag) return true;
    // XESTRATA_ADAPT_LAG=auto: what the round's lag cost, the wait the window took for the copies or the swapped
    // experts' CPU work over the windows they waited (auto_pick.hpp)
    const size_t n_swaps = pending_.size();
    const auto lag0 = std::chrono::steady_clock::now();
    const bool ok = settle_swaps(true, err);
    const double wait_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lag0).count();
    const double per = root_ != nullptr && root_->split_.c > 0.0
                           ? std::max(0.0, root_->split_.c - root_->split_.q) : 0.04;
    if ((root_ == nullptr || root_ == this) && !serve_)   // not under --serve: a prompt reads the table again
        strata::core::adapt_lag_window(wait_ms + (lag > 1 ? (double) (lag - 1) * (double) n_swaps * per : 0.0));
    return ok;
}

bool PlainModel::settle_swaps(bool wait, std::string& err) {
    if (pending_.empty()) {   // a round of blocking fills: their copies have landed
        if (complement_ != nullptr) complement_->commit_exchanges();
        return true;
    }
    if (wait ? !strata::gpu::event_sync(swap_ev_) : !strata::gpu::event_query(swap_ev_)) {
        if (!wait) return true;
        err = std::string("plain model: the adaptive tier's copies: ") + strata::gpu::last_error();
        return false;
    }
    for (const PendingSwap& p : pending_) res_[(size_t) (p.layer * spec_.n_expert + p.expert)] = p.slot;
    pending_.clear();
    if (complement_ != nullptr) complement_->commit_exchanges();
    return true;
}

}  // namespace strata::core
