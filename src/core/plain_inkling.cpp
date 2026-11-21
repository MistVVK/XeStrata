// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/plain_inkling.cpp - Inkling-2-Mini's CPU reference forward (see plain_inkling.hpp).
#include "strata/core/plain_inkling.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace strata::core {
namespace kc = strata::kernels::cpu;

namespace {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void rms_norm(const float* x, const float* w, float* y, int64_t n, float eps) {
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float scale = 1.0f / std::sqrt((float) (ss / n) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale * w[i];
}

void rms_norm_no_gamma(const float* x, float* y, int64_t n, float eps) {
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float scale = 1.0f / std::sqrt((float) (ss / n) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale;
}

/// `cur` is [channels][n]; the row's L2 norm is capped at `threshold` after `scale` (llama.cpp's graph::clip).
void l2_clip(const float* scale, float threshold, float* cur, int64_t channels, int64_t n) {
    // a GGUF [channels][n] tensor is stored with `channels` fastest: scale[head*channels + channel]
    for (int64_t c = 0; c < channels; ++c)
        for (int64_t t = 0; t < n; ++t)
            if (scale != nullptr) cur[c * n + t] *= scale[t * channels + c];
    for (int64_t t = 0; t < n; ++t) {
        double ss = 0;
        for (int64_t c = 0; c < channels; ++c) ss += (double) cur[c * n + t] * cur[c * n + t];
        const float norm = std::sqrt((float) ss) + 1e-6f;
        const float factor = std::min(1.0f, threshold / norm);
        for (int64_t c = 0; c < channels; ++c) cur[c * n + t] *= factor;
    }
}

/// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb) (llama.cpp's LLM_FFN_SITU).
inline float situ(float gate, float up, float beta, float lb) {
    const float g = beta * std::tanh(gate / beta) * sigmoid(gate);
    return g * (lb > 0 ? lb * std::tanh(up / lb) : up);
}

}  // namespace

struct PlainInkling::Impl {
    const GgufModel* m = nullptr;
    ModelSpec s;
    int64_t N = 0, NH = 0, NKV = 0, HD = 0, NE = 0, NK = 0, FF = 0, NCONV = 0, NSLOT = 0;
    float eps = 1e-6f;

    // Dense tensors dequantized once into row-major [shape[0]][product(shape[1:])].
    std::unordered_map<std::string, std::vector<float>> dense;
    // Expert blobs stay in the file; a row is dequantized when the token routes to that expert.
    struct Expert {
        const TensorInfo* t = nullptr;
        const uint8_t* data = nullptr;
        const float* scale = nullptr;   // per-expert F32 scale, nullptr for GGUF quants
        int type = -1;
        int64_t rows = 0, cols = 0;
        size_t row_bytes = 0;
    };
    struct Layer {
        std::string q, k, v, o, gate, qclip, sink, rel, relbias, relclip, kconv, vconv, postconv, ffnconv;
        std::string attn_norm, qin, kin, vin, kproj, vproj, resq, ffnq, ffnnorm, router;
        Expert gu, down;
        const float* exp_b = nullptr;
        bool swa = false;
    };
    std::vector<Layer> layers;
    // loop controllers of physical core layers (indexed ip - loop_start)
    struct Loop {
        std::string query;
        std::string down_name[4], gain_name[4], forget_name[4];
    };
    std::vector<Loop> loops;
    std::vector<float> emb, head, out_norm, out_query, tok_norm;
    const TensorInfo* emb_t = nullptr;
    const uint8_t* emb_data = nullptr;
    int emb_type = -1;
    size_t emb_row = 0;
    const TensorInfo* head_t = nullptr;
    const uint8_t* head_data = nullptr;
    int head_type = -1;
    size_t head_row = 0;

    // state: one KV cache and one convolution history per cache slot
    struct Slot {
        std::vector<float> k, v;      // [cells][NKV][HD]
        std::vector<int32_t> pos;     // [cells]
        int64_t cells = 0;
        std::vector<float> conv;      // [NCONV-1][channels]
    };
    std::vector<Slot> slots;

    int64_t n_cells = 0;
    std::vector<float> last_logits_;

    const std::vector<float>& matrix(const std::string& name, std::string& err) {
        auto it = dense.find(name);
        if (it != dense.end()) return it->second;
        size_t at = 0;
        const TensorInfo* t = m->find(name, &at);
        if (t == nullptr || t->shape.size() < 2 || !m->in_bounds(*t, at)) {
            err = "inkling2: tensor " + name + " is missing, not a matrix or truncated";
            static const std::vector<float> empty;
            return empty;
        }
        const int64_t n_in = (int64_t) t->shape[0];
        int64_t n_out = 1;
        for (size_t d = 1; d < t->shape.size(); ++d) n_out *= (int64_t) t->shape[d];
        const size_t rb = kc::native_row_bytes((int) t->type, n_in);
        if (rb == 0) {
            err = "inkling2: tensor " + name + " has no host dequantizer";
            static const std::vector<float> empty;
            return empty;
        }
        std::vector<float> w((size_t) (n_in * n_out));
        const uint8_t* src = m->shard(at).tensor_data(*t);
        for (int64_t o = 0; o < n_out; ++o)
            if (!kc::native_row_to_float((int) t->type, src + (size_t) o * rb, w.data() + (size_t) o * n_in, n_in)) {
                err = "inkling2: cannot dequantize " + name;
                break;
            }
        return dense.emplace(name, std::move(w)).first->second;
    }

    const float* vector(const std::string& name, int64_t n, std::string& err) {
        auto it = dense.find(name);
        if (it != dense.end()) return it->second.data();
        size_t at = 0;
        const TensorInfo* t = m->find(name, &at);
        if (t == nullptr || (int64_t) t->elements() != n || !m->in_bounds(*t, at)) {
            err = "inkling2: tensor " + name + " is missing or not " + std::to_string(n) + " values";
            return nullptr;
        }
        std::vector<float> v((size_t) n);
        if (!kc::native_row_to_float((int) t->type, m->shard(at).tensor_data(*t), v.data(), n)) {
            err = "inkling2: cannot dequantize " + name;
            return nullptr;
        }
        return dense.emplace(name, std::move(v)).first->second.data();
    }

    bool expert(Expert& e, const std::string& name, int64_t n_in, int64_t n_out, const std::string& scale,
                std::string& err) {
        size_t at = 0;
        const TensorInfo* t = m->find(name, &at);
        const int64_t total = (t == nullptr || t->shape.empty()) ? 0 : (int64_t) (t->elements() / t->shape[0]);
        if (t == nullptr || t->shape.size() < 2 || (int64_t) t->shape[0] != n_in || NE <= 0 || total != n_out * NE ||
            !m->in_bounds(*t, at)) {
            err = "inkling2: expert tensor " + name + " has the wrong shape";
            return false;
        }
        e.t = t;
        e.data = m->shard(at).tensor_data(*t);
        e.type = (int) t->type;
        e.rows = n_out;
        e.cols = n_in;
        e.row_bytes = kc::native_row_bytes(e.type, n_in);
        if (e.row_bytes == 0) {
            err = "inkling2: expert tensor " + name + " has no host dequantizer";
            return false;
        }
        if (!scale.empty()) {
            std::string se;
            const float* v = vector(scale, NE, se);
            e.scale = se.empty() ? v : nullptr;   // the GGUF-quantized layers carry no scale tensor
        }
        return err.empty();
    }

    /// One expert's row range into `out` [rows][cols], with its per-expert scale applied.
    void expert_rows(const Expert& e, int ex, int64_t r0, int64_t r1, float* out) {
        const uint8_t* src = e.data + (size_t) ex * (size_t) e.rows * e.row_bytes;
        const float scale = e.scale != nullptr ? e.scale[ex] : 1.0f;
        for (int64_t r = r0; r < r1; ++r)
            kc::native_row_to_float(e.type, src + (size_t) r * e.row_bytes, out + (size_t) r * e.cols, e.cols);
        if (scale != 1.0f)
            for (int64_t i = 0; i < (r1 - r0) * e.cols; ++i) out[i] *= scale;
    }

    void matvec(const std::vector<float>& w, const float* x, float* y, int64_t n_in, int64_t n_out, int threads) {
        auto span = [&](int64_t r0, int64_t r1) {
            for (int64_t o = r0; o < r1; ++o) {
                const float* row = w.data() + (size_t) o * n_in;
                double ss = 0;
                for (int64_t i = 0; i < n_in; ++i) ss += (double) row[i] * x[i];
                y[o] = (float) ss;
            }
        };
        if (threads <= 1 || n_out < 128) { span(0, n_out); return; }
        std::vector<std::thread> ts;
        const int64_t chunk = (n_out + threads - 1) / threads;
        for (int64_t r = 0; r < n_out; r += chunk)
            ts.emplace_back([&, r, chunk] { span(r, std::min(n_out, r + chunk)); });
        for (auto& t : ts) t.join();
    }

    void mat(const std::string& name, const float* x, float* y, int64_t n_in, int64_t n_out, int threads,
             std::string& err) {
        const auto& w = matrix(name, err);
        if (!err.empty()) return;
        matvec(w, x, y, n_in, n_out, threads);
    }

    void reset_state();
    bool load(const GgufModel& model, const ModelSpec& spec, std::string& err);
    bool token(int32_t tok, int64_t pos, int threads, std::string& err);
};

bool PlainInkling::Impl::load(const GgufModel& model, const ModelSpec& spec, std::string& err) {
    m = &model;
    s = spec;
    N = s.n_embd;
    NH = s.n_head;
    NKV = s.n_head_kv.empty() ? 0 : s.n_head_kv[0];
    HD = s.key_length;
    NE = s.n_expert;
    NK = s.n_expert_used;
    FF = s.n_ff_exp;
    NSLOT = s.n_slot();
    NCONV = s.sconv_l_cache;
    eps = (float) s.rms_eps;
    if (N <= 0 || NH <= 0 || NKV <= 0 || HD <= 0 || NE <= 0 || NK <= 0 || FF <= 0 || NSLOT <= 0 ||
        N % NH != 0 || NH % NKV != 0) {
        err = "inkling2: invalid attention or expert geometry";
        return false;
    }
    for (int64_t l = 0; l < s.n_layer; ++l)
        if (s.n_head_kv[(size_t) l] != NKV || s.n_head != NH) {
            err = "inkling2: per-layer head counts differ (not supported by the CPU reference)";
            return false;
        }

    size_t at = 0;
    const TensorInfo* t = m->find("token_embd.weight", &at);
    if (t == nullptr || t->shape.size() != 2 || (int64_t) t->shape[0] != N || !m->in_bounds(*t, at)) {
        err = "inkling2: token_embd.weight is missing or of the wrong shape";
        return false;
    }
    emb_t = t;
    emb_data = m->shard(at).tensor_data(*t);
    emb_type = (int) t->type;
    emb_row = (size_t) t->shape[1];
    t = m->find("output.weight", &at);
    if (t == nullptr || t->shape.size() != 2 || (int64_t) t->shape[0] != N || !m->in_bounds(*t, at)) {
        err = "inkling2: output.weight is missing or of the wrong shape";
        return false;
    }
    head_t = t;
    head_data = m->shard(at).tensor_data(*t);
    head_type = (int) t->type;
    head_row = (size_t) t->shape[1];
    if (emb_row != (size_t) s.vocab_padded || head_row != (size_t) s.vocab_padded) {
        err = "inkling2: the embedding or head does not match the padded vocabulary";
        return false;
    }

    const auto need_vec = [&](const char* name, int64_t n) {
        std::string e;
        (void) vector(name, n, e);
        if (!e.empty()) { err = e; return false; }
        return true;
    };
    if (!need_vec("token_embd_norm.weight", N) || !need_vec("output_norm.weight", N) ||
        !need_vec("output_res_query.weight", N)) return false;
    tok_norm = dense.find("token_embd_norm.weight")->second;
    out_norm = dense.find("output_norm.weight")->second;
    out_query = dense.find("output_res_query.weight")->second;

    layers.resize((size_t) s.n_layer);
    for (int64_t l = 0; l < s.n_layer; ++l) {
        Layer& L = layers[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        L.swa = s.layer_swa(l);
        L.attn_norm = p + "attn_norm.weight";
        L.qin = p + "attn_q_inp_norm.weight";
        L.kin = p + "attn_k_inp_norm.weight";
        L.vin = p + "attn_v_inp_norm.weight";
        L.kproj = p + "attn_k_conv_proj.weight";
        L.vproj = p + "attn_v_conv_proj.weight";
        L.q = p + "attn_q.weight";
        L.k = p + "attn_k.weight";
        L.v = p + "attn_v.weight";
        L.o = p + "attn_output.weight";
        L.gate = p + "attn_gate.weight";
        L.qclip = p + "attn_q_clip.weight";
        L.sink = p + "attn_sink_proj.weight";
        L.rel = p + "attn_rel.weight";
        L.relbias = p + "attn_rel_bias.weight";
        L.relclip = p + "attn_rel_clip.weight";
        L.kconv = p + "attn_k_conv.weight";
        L.vconv = p + "attn_v_conv.weight";
        L.postconv = p + "attn_post_conv.weight";
        L.ffnconv = p + "ffn_post_conv.weight";
        L.resq = p + "attn_res_query.weight";
        L.ffnq = p + "ffn_res_query.weight";
        L.ffnnorm = p + "ffn_norm.weight";
        L.router = p + "ffn_gate_inp.weight";
        // allocate the norm and router vectors (and check every other dense tensor's presence/shape)
        std::vector<std::string> dense_names = {L.q, L.k, L.v, L.o, L.gate, L.qclip, L.sink, L.rel, L.relbias,
                                                L.relclip, L.kconv, L.vconv, L.postconv, L.ffnconv, L.ffnq,
                                                L.ffnnorm, L.router};
        if (L.swa) dense_names.push_back(L.attn_norm);
        else {
            dense_names.push_back(L.qin);
            dense_names.push_back(L.kin);
            dense_names.push_back(L.vin);
            dense_names.push_back(L.kproj);
            dense_names.push_back(L.vproj);
        }
        if (l > 0) dense_names.push_back(L.resq);
        for (const auto& name : dense_names) {
            std::string e;
            (void) matrix(name, e);
            if (!e.empty()) { err = e; return false; }
        }
        if (!need_vec(L.ffnq.c_str(), N) || !need_vec(L.ffnnorm.c_str(), N)) return false;
        const std::string b = p + "exp_probs_b.bias";
        std::string e;
        L.exp_b = vector(b, NE, e);
        if (!e.empty()) { err = e; return false; }
        const std::string gu_scale = p + "ffn_gate_up_exps.scale";
        const std::string d_scale = p + "ffn_down_exps.scale";
        if (!expert(L.gu, p + "ffn_gate_up_exps.weight", N, 2 * FF, gu_scale, err) ||
            !expert(L.down, p + "ffn_down_exps.weight", FF, N, d_scale, err)) return false;
    }

    loops.resize((size_t) (s.loop_end - s.loop_start));
    for (int64_t ip = s.loop_start; ip < s.loop_end; ++ip) {
        Loop& C = loops[(size_t) (ip - s.loop_start)];
        const std::string p = "blk." + std::to_string(ip) + ".";
        C.query = p + "loop_msg_query.weight";
        const char* ctrl[4] = {"q", "k", "v", "ffn"};
        for (int c = 0; c < 4; ++c) {
            const std::string pre = p + "loop_" + ctrl[c] + "_ctrl_";
            C.down_name[c] = pre + "down.weight";
            C.gain_name[c] = pre + "gain.weight";
            C.forget_name[c] = pre + "forget.weight";
            // down [2N, rank], gain/forget [rank, N]; kept in `dense`, not as expert blobs
            std::string e;
            (void) matrix(pre + "down.weight", e);
            (void) matrix(pre + "gain.weight", e);
            (void) matrix(pre + "forget.weight", e);
            if (!e.empty()) { err = e; return false; }
        }
    }

    slots.resize((size_t) NSLOT);
    reset_state();
    return true;
}

void PlainInkling::Impl::reset_state() {
    const int64_t channels = 2 * NKV * HD + 2 * N;
    for (auto& sl : slots) {
        sl.k.assign((size_t) s.relative_extent * NKV * HD, 0.0f);
        sl.v.assign((size_t) s.relative_extent * NKV * HD, 0.0f);
        sl.pos.assign((size_t) s.relative_extent, -1);
        sl.cells = 0;
        sl.conv.assign((size_t) (NCONV - 1) * channels, 0.0f);
    }
}

void PlainInkling::reset() {
    if (p_) p_->reset_state();
}

bool PlainInkling::Impl::token(int32_t tok, int64_t pos, int threads, std::string& err) {
    if (tok < 0 || tok >= (int32_t) s.vocab_padded) {
        err = "inkling2: token out of range";
        return false;
    }
    const int64_t nrel = s.relative_length;
    const int64_t extent = s.relative_extent;
    const int64_t channels = 2 * NKV * HD + 2 * N;
    const float inv_sqrt_hd = 1.0f / std::sqrt((float) HD);
    const float inv_sqrt_n = 1.0f / std::sqrt((float) N);
    std::vector<float> x(N), hn(N), q(NH * HD), k(NKV * HD), v(NKV * HD), r(NH * nrel), att(NH * HD);
    std::vector<float> in0(N), in1(N), in2(N), normed(N), mixed(N), tmp(N), out(N);
    std::vector<float> gu(2 * FF), hv(FF), exout(N), gate_tmp(NH * HD), msg[4];
    for (auto& m : msg) m.resize(N);
    std::vector<std::vector<float>> blocks;

    // embedding: the raw token's row is not normalized (an encoded image's row would be)
    const size_t rb = kc::native_row_bytes(emb_type, N);
    if (rb == 0 || !kc::native_row_to_float(emb_type, emb_data + (size_t) tok * rb, x.data(), N)) {
        err = "inkling2: cannot dequantize the token embedding";
        return false;
    }
    rms_norm(x.data(), tok_norm.data(), hn.data(), N, eps);

    // tau of llama.cpp's build_inputs: 1 + alpha*log(max(1, (pos+1)/floor))
    const double ratio = std::max(1.0, (double) (pos + 1) / (double) s.log_scale_floor);
    const float tau = (float) (1.0 + s.log_scale_alpha * std::log(ratio));

    // ---- the helpers of one layer ----
    auto dense_of = [&](const std::string& name) -> const float* {
        auto it = dense.find(name);
        return it == dense.end() ? nullptr : it->second.data();
    };
    auto multiply = [&](const std::string& name, const float* in, float* o, int64_t n_in, int64_t n_out) {
        std::string e;
        const auto& w = matrix(name, e);
        if (!e.empty()) { err = e; return; }
        matvec(w, in, o, n_in, n_out, threads);
    };

    // causal short convolution with a residual; `dynamic` adds per-head input-dependent taps (full-attention K/V)
    auto convolve = [&](const float* cur, int64_t n_ch, const std::string& weight, const float* dyn,
                        Slot& sl, int64_t offset, float* o) {
        std::string e;
        const auto& wt = matrix(weight, e);
        if (!e.empty()) { err = e; return; }
        const int64_t n_heads = dyn != nullptr ? NKV : 0;
        const float scale = 1.0f / std::sqrt((float) ((dyn != nullptr ? 2 : 1) * NCONV + 1));
        for (int64_t c = 0; c < n_ch; ++c) {
            double acc = 0;
            for (int64_t lag = 0; lag < NCONV; ++lag) {
                const float* src = lag == NCONV - 1 ? cur
                    : sl.conv.data() + (size_t) (NCONV - 2 - lag) * channels + (size_t) offset;
                acc += (double) wt[(size_t) lag * n_ch + c] * src[c];
            }
            o[c] = (cur[c] + (float) acc) * scale;
        }
        if (dyn != nullptr) {
            const int64_t per_head = n_ch / n_heads;
            for (int64_t hd = 0; hd < n_heads; ++hd)
                for (int64_t lag = 0; lag < NCONV; ++lag) {
                    const float* src = lag == NCONV - 1 ? cur
                        : sl.conv.data() + (size_t) (NCONV - 2 - lag) * channels + (size_t) offset;
                    const float tap = dyn[(size_t) lag * n_heads + hd];
                    for (int64_t d = 0; d < per_head; ++d) {
                        const int64_t c = hd * per_head + d;
                        o[c] += src[c] * tap * scale;
                    }
                }
        }
        for (int64_t lag = 0; lag < NCONV - 2; ++lag)
            std::memcpy(sl.conv.data() + (size_t) lag * channels + (size_t) offset,
                        sl.conv.data() + (size_t) (lag + 1) * channels + (size_t) offset,
                        (size_t) n_ch * sizeof(float));
        std::memcpy(sl.conv.data() + (size_t) (NCONV - 2) * channels + (size_t) offset, cur,
                    (size_t) n_ch * sizeof(float));
    };

    // depth attention over the stored block outputs and cur (llama.cpp's graph::mix)
    auto mix = [&](const std::vector<std::vector<float>>& b, const float* cur, const float* qry, float* o) {
        const size_t ns = b.size() + 1;
        std::vector<float> states(ns * N), scores(ns);
        for (size_t j = 0; j < b.size(); ++j) rms_norm_no_gamma(b[j].data(), states.data() + j * N, N, 1e-6f);
        rms_norm_no_gamma(cur, states.data() + (ns - 1) * N, N, 1e-6f);
        for (size_t j = 0; j < ns; ++j) {
            double ss = 0;
            for (int64_t i = 0; i < N; ++i) ss += (double) states[j * N + i] * qry[i];
            scores[j] = (float) ss * inv_sqrt_n;
        }
        const float mx = *std::max_element(scores.begin(), scores.end());
        double sum = 0;
        for (auto& z : scores) { z = std::exp(z - mx); sum += z; }
        std::fill(o, o + N, 0.0f);
        for (size_t j = 0; j < ns; ++j)
            for (int64_t i = 0; i < N; ++i) o[i] += (float) (scores[j] / sum) * states[j * N + i];
    };

    // the cross-pass control gate: (1 - forget)*cur + gain*message
    auto control = [&](const float* cur, const float* message, const Loop& C, int c, float* o) {
        std::vector<float> gate(2 * N), g(N), rank((size_t) s.loop_gate_rank);
        rms_norm_no_gamma(cur, gate.data(), N, 1e-6f);
        rms_norm_no_gamma(message, gate.data() + N, N, 1e-6f);
        std::string e;
        const auto& down = matrix(C.down_name[c], e);
        if (!e.empty()) { err = e; return; }
        matvec(down, gate.data(), rank.data(), 2 * N, s.loop_gate_rank, 1);
        const auto& gain = matrix(C.gain_name[c], e);
        const auto& forget = matrix(C.forget_name[c], e);
        if (!e.empty()) { err = e; return; }
        matvec(gain, rank.data(), g.data(), s.loop_gate_rank, N, 1);
        matvec(forget, rank.data(), o, s.loop_gate_rank, N, 1);   // o is `forget` until the last step
        for (int64_t i = 0; i < N; ++i)
            o[i] = cur[i] - std::tanh(o[i]) * cur[i] + std::tanh(g[i]) * message[i];
    };

    auto layer = [&](int64_t ip, int64_t slot_id, float* cur_in, std::vector<std::vector<float>>& bl,
                     const float* messages, const Loop* loop, bool messages_present) {
        if (!err.empty()) return;
        Layer& L = layers[(size_t) ip];
        Slot& sl = slots[(size_t) slot_id];
        std::vector<float> inpSA(cur_in, cur_in + N);
        const float* inputs[3] = {cur_in, cur_in, cur_in};
        std::vector<float> mx;
        if (dense_of(L.resq) != nullptr) {
            mx.resize(N);
            mix(bl, cur_in, dense_of(L.resq), mx.data());
            inputs[0] = inputs[1] = inputs[2] = mx.data();
            if (!L.swa) { in1 = mx; in2 = mx; inputs[1] = in1.data(); inputs[2] = in2.data(); }
        }
        if (messages_present && loop != nullptr) {
            for (int j = 0; j < 3; ++j) {
                std::vector<float> ctr(N);
                control(inputs[j], messages + (size_t) j * N, *loop, j, ctr.data());
                if (j == 0) in0 = std::move(ctr); else if (j == 1) in1 = std::move(ctr); else in2 = std::move(ctr);
                inputs[j] = j == 0 ? in0.data() : (j == 1 ? in1.data() : in2.data());
            }
        }
        if (L.swa) {
            if (messages_present) {
                for (int j = 0; j < 3; ++j) {
                    std::vector<float> nz(N);
                    rms_norm(inputs[j], dense_of(L.attn_norm), nz.data(), N, eps);
                    if (j == 0) in0 = std::move(nz); else if (j == 1) in1 = std::move(nz); else in2 = std::move(nz);
                    inputs[j] = j == 0 ? in0.data() : (j == 1 ? in1.data() : in2.data());
                }
            } else {
                rms_norm(inputs[0], dense_of(L.attn_norm), in0.data(), N, eps);
                inputs[0] = inputs[1] = inputs[2] = in0.data();
            }
        } else {
            rms_norm(inputs[0], dense_of(L.qin), in0.data(), N, eps);
            rms_norm(inputs[1], dense_of(L.kin), in1.data(), N, eps);
            rms_norm(inputs[2], dense_of(L.vin), in2.data(), N, eps);
            inputs[0] = in0.data(); inputs[1] = in1.data(); inputs[2] = in2.data();
        }
        if (!err.empty()) return;

        multiply(L.q, inputs[0], q.data(), N, NH * HD);
        multiply(L.k, inputs[1], k.data(), N, NKV * HD);
        multiply(L.v, inputs[2], v.data(), N, NKV * HD);
        multiply(L.rel, inputs[0], r.data(), N, NH * nrel);
        std::vector<float> kdyn((size_t) NCONV * NKV), vdyn((size_t) NCONV * NKV);
        if (!L.swa) {
            multiply(L.kproj, inputs[1], kdyn.data(), N, NCONV * NKV);
            multiply(L.vproj, inputs[2], vdyn.data(), N, NCONV * NKV);
        }
        convolve(k.data(), NKV * HD, L.kconv, L.swa ? nullptr : kdyn.data(), sl, 0, tmp.data());
        k.assign(tmp.begin(), tmp.begin() + NKV * HD);
        convolve(v.data(), NKV * HD, L.vconv, L.swa ? nullptr : vdyn.data(), sl, NKV * HD, tmp.data());
        v.assign(tmp.begin(), tmp.begin() + NKV * HD);
        l2_clip(dense_of(L.qclip), (float) s.qk_clip, q.data(), HD, NH);
        l2_clip(nullptr, (float) s.qk_clip, k.data(), HD, NKV);
        for (int64_t h = 0; h < NKV; ++h) rms_norm_no_gamma(v.data() + h * HD, v.data() + h * HD, HD, eps);
        l2_clip(dense_of(L.relclip), (float) s.relative_clip, r.data(), nrel, NH);
        if (!L.swa) {
            for (auto& z : q) z *= tau;
            for (auto& z : r) z *= tau;
        }
        if (!err.empty()) return;

        // append this token's K and V to the slot's cache
        const int64_t cell = sl.cells < extent ? sl.cells++ : extent - 1;
        sl.pos[(size_t) cell] = pos;
        for (int64_t h = 0; h < NKV; ++h) {
            std::memcpy(sl.k.data() + ((size_t) cell * NKV + h) * HD, k.data() + h * HD, HD * sizeof(float));
            std::memcpy(sl.v.data() + ((size_t) cell * NKV + h) * HD, v.data() + h * HD, HD * sizeof(float));
        }

        const auto& relbias = matrix(L.relbias, err);
        std::vector<float> sink(NH);
        multiply(L.sink, inputs[0], sink.data(), N, NH);
        if (!err.empty()) return;
        for (int64_t h = 0; h < NH; ++h) {
            const int64_t hkv = h / (NH / NKV);
            std::vector<float> scores((size_t) cell + 1), p((size_t) cell + 1);
            for (int64_t c = 0; c < cell; ++c) {
                const int64_t dist = pos - sl.pos[(size_t) c];
                if (dist < 0 || dist >= extent || (L.swa && dist >= s.swa_window)) { scores[(size_t) c] = -1e30f; continue; }
                double ss = 0;
                for (int64_t d = 0; d < HD; ++d)
                    ss += (double) q[h * HD + d] * sl.k.data()[((size_t) c * NKV + hkv) * HD + d];
                double bias = -(double) sink[h];
                for (int64_t d = 0; d < nrel; ++d)
                    bias += (double) r[h * nrel + d] * relbias[(size_t) dist * nrel + d];
                scores[(size_t) c] = (float) (ss * inv_sqrt_hd + bias);
            }
            scores[(size_t) cell] = 0.0f;   // the zero sink
            const float mx = *std::max_element(scores.begin(), scores.end());
            double sum = 0;
            for (auto& z : scores) { z = std::exp(z - mx); sum += z; }
            double acc = 0;
            for (int64_t c = 0; c < cell; ++c) acc += (double) (scores[(size_t) c] / sum);
            for (int64_t d = 0; d < HD; ++d)
                att[h * HD + d] = 0.0f;
            for (int64_t c = 0; c < cell; ++c) {
                if (scores[(size_t) c] < 0) continue;
                const float pw = scores[(size_t) c] / sum;
                const float* vc = sl.v.data() + ((size_t) c * NKV + hkv) * HD;
                for (int64_t d = 0; d < HD; ++d) att[h * HD + d] += pw * vc[d];
            }
            (void) acc;
        }

        multiply(L.gate, inputs[0], gate_tmp.data(), N, NH * HD);
        for (int64_t i = 0; i < NH * HD; ++i) att[i] *= sigmoid(gate_tmp[i]);
        multiply(L.o, att.data(), tmp.data(), NH * HD, N);
        convolve(tmp.data(), N, L.postconv, nullptr, sl, 2 * NKV * HD, out.data());
        if (!err.empty()) return;

        if (ip % s.attn_res_block == 0) bl.push_back(inpSA);
        else for (int64_t i = 0; i < N; ++i) cur_in[i] += out[i];

        const float* ffn_inp = cur_in;
        std::vector<float> ffn_cur(N);
        mix(bl, cur_in, dense_of(L.ffnq), ffn_cur.data());
        if (messages_present && loop != nullptr) {
            std::vector<float> ctr(N);
            control(ffn_cur.data(), messages + 3 * N, *loop, 3, ctr.data());
            ffn_cur = std::move(ctr);
        }
        if (!err.empty()) return;
        rms_norm(ffn_cur.data(), dense_of(L.ffnnorm), normed.data(), N, eps);

        // sigmoid router, top-8, the selected probabilities normalized and scaled by route_scale
        multiply(L.router, normed.data(), tmp.data(), N, NE);
        std::vector<float> prob((size_t) NE);
        for (int64_t e = 0; e < NE; ++e) prob[(size_t) e] = sigmoid(tmp[e] + (L.exp_b != nullptr ? L.exp_b[e] : 0.0f));
        std::vector<int64_t> ids(NE);
        std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(ids.begin(), ids.begin() + NK, ids.end(),
                          [&](int64_t a, int64_t b) { return prob[(size_t) a] > prob[(size_t) b]; });
        std::vector<float> w((size_t) NK);
        double wsum = 0;
        for (int64_t j = 0; j < NK; ++j) { w[(size_t) j] = prob[(size_t) ids[(size_t) j]]; wsum += w[(size_t) j]; }
        for (auto& z : w) z = (float) (z / (wsum + 1e-20) * s.expert_weights_scale);

        std::fill(out.begin(), out.end(), 0.0f);
        for (int64_t j = 0; j < NK; ++j) {
            const int64_t eid = ids[(size_t) j];
            expert_rows(L.gu, (int) eid, 0, 2 * FF, gu.data());
            for (int64_t i = 0; i < FF; ++i)
                hv[(size_t) i] = situ(gu[(size_t) i], gu[(size_t) (FF + i)], (float) s.situ_beta,
                                      (float) s.situ_linear_beta);
            expert_rows(L.down, (int) eid, 0, N, exout.data());
            for (int64_t i = 0; i < N; ++i) out[i] += w[(size_t) j] * exout[i];
        }
        convolve(out.data(), N, L.ffnconv, nullptr, sl, 2 * NKV * HD + N, tmp.data());
        for (int64_t i = 0; i < N; ++i) cur_in[i] = ffn_inp[i] + tmp[i];
    };

    // ---- the graph: prelude, 3 passes over the core, tail ----
    std::vector<float> cur(N);
    std::memcpy(cur.data(), hn.data(), N * sizeof(float));
    for (int64_t l = 0; l < s.loop_start; ++l) layer(l, l, cur.data(), blocks, nullptr, nullptr, false);
    const auto prelude_blocks = blocks;
    const std::vector<float> prelude = cur;
    std::vector<std::vector<float>> bank;
    std::vector<float> bank_out;
    for (int64_t pass = 0; pass < s.loop_count && err.empty(); ++pass) {
        if (pass > 0) {
            // the previous pass's blocks after its first core block, and its output
            bank.assign(blocks.begin() + (size_t) prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int64_t ip = s.loop_start; ip < s.loop_end && err.empty(); ++ip) {
            const int64_t il = pass == 0 ? ip : s.n_layer + (pass - 1) * (s.loop_end - s.loop_start) + (ip - s.loop_start);
            if (pass == 0) { layer(ip, ip, cur.data(), blocks, nullptr, nullptr, false); continue; }
            const Loop& C = loops[(size_t) (ip - s.loop_start)];
            std::vector<float> message(4 * N);
            mix(bank, bank_out.data(), dense_of(C.query), message.data());
            layer(ip, il, cur.data(), blocks, message.data(), &C, true);
        }
    }
    for (int64_t l = s.loop_end; l < s.n_layer && err.empty(); ++l) layer(l, l, cur.data(), blocks, nullptr, nullptr, false);
    if (!err.empty()) return false;

    // ---- the head ----
    mix(blocks, cur.data(), out_query.data(), tmp.data());
    rms_norm(tmp.data(), out_norm.data(), hn.data(), N, eps);
    std::vector<float> logits((size_t) s.vocab_padded);
    const size_t hrb = kc::native_row_bytes(head_type, N);
    if (hrb == 0) {
        err = "inkling2: the head has no host dequantizer";
        return false;
    }
    for (int64_t i = 0; i < s.vocab_padded; ++i)
        if (!kc::native_row_to_float(head_type, head_data + (size_t) i * hrb, logits.data() + i, N)) {
            err = "inkling2: cannot dequantize a head row";
            return false;
        }
    // the head's dot products, in F32 as mainline's ggml_prec_set_acc(F32)
    const float cap = (float) s.final_logit_softcap;
    for (int64_t i = 0; i < s.vocab_unpadded; ++i) {
        double ss = 0;
        for (int64_t d = 0; d < N; ++d) ss += (double) logits[(size_t) i * N + d] * hn[d];
        float z = (float) ss;
        logits[(size_t) i] = cap > 0 ? cap * std::tanh(z / cap) : z;
    }
    last_logits_ = std::move(logits);
    return true;
}

PlainInkling::PlainInkling() : p_(std::make_unique<Impl>()) {}
PlainInkling::~PlainInkling() = default;

bool PlainInkling::load(const GgufModel& m, const ModelSpec& s, std::string& err) {
    return p_->load(m, s, err);
}

bool PlainInkling::run(const int32_t* tokens, int64_t n, int64_t pos, float* logits, std::string& err) {
    if (n < 1 || tokens == nullptr) {
        err = "inkling2: no tokens to run";
        return false;
    }
    // The reference walks one token at a time.  Prompt and decode use the same path, so a token's answer is the same
    // whether it was alone in a window or at the end of a prompt.  Thread count stays modest: SYNC5-TILE is building
    // on this machine while this reference is used.
    const unsigned hw = std::thread::hardware_concurrency();
    const int threads = std::max(1, (int) hw / 8);
    for (int64_t t = 0; t < n; ++t)
        if (!p_->token(tokens[t], pos + t, threads, err)) return false;
    if (logits != nullptr)
        std::memcpy(logits, p_->last_logits_.data(), (size_t) p_->s.vocab_unpadded * sizeof(float));
    return true;
}

int64_t PlainInkling::vocab() const { return p_->s.vocab_unpadded; }

std::string PlainInkling::describe() const {
    const Impl& X = *p_;
    return "inkling2_loop: " + std::to_string(X.s.n_layer) + " layers, " + std::to_string(X.NSLOT) +
           " loop slots, " + std::to_string(X.NH) + "/" + std::to_string(X.NKV) + " heads, " +
           std::to_string(X.NE) + " experts (top " + std::to_string(X.NK) + "), relative " +
           std::to_string(X.s.relative_length) + "x" + std::to_string(X.s.relative_extent) + ", swa " +
           std::to_string(X.s.swa_window) + ", CPU reference";
}

}  // namespace strata::core
