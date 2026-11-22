// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/plain_inkling.cpp - Inkling-2-Mini's CPU reference forward (see plain_inkling.hpp).
#include "strata/core/plain_inkling.hpp"

#include "plain_dspark.hpp"
#include "plain_inkling_gpu.hpp"
#include "strata/platform/env.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_tier.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <cstdlib>
#include <cstdio>
#include <unordered_map>

namespace strata::core {
namespace kc = strata::kernels::cpu;

namespace {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void rms_norm(const float* x, const float* w, float* y, int64_t n, float eps) {
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float scale = 1.0f / std::sqrt((float) (ss / (double) n) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale * w[i];
}

void rms_norm_no_gamma(const float* x, float* y, int64_t n, float eps) {
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float scale = 1.0f / std::sqrt((float) (ss / (double) n) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale;
}

/// Cap the L2 norm of each contiguous head after applying its learned scale.
void l2_clip(const float* scale, float threshold, float* cur, int64_t width, int64_t heads) {
    for (int64_t h = 0; h < heads; ++h) {
        float* row = cur + h * width;
        double ss = 0;
        for (int64_t d = 0; d < width; ++d) {
            if (scale) row[d] *= scale[h * width + d];
            ss += (double) row[d] * row[d];
        }
        const float factor = std::min(1.0f, threshold / (std::sqrt((float) ss) + 1e-6f));
        for (int64_t d = 0; d < width; ++d) row[d] *= factor;
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
    std::unique_ptr<PlainDspark> dspark;
    std::unique_ptr<InklingProducts> products;
    std::vector<float> draft_features;
    bool taps_valid = false;
    std::vector<double> heat;

    bool embedding(int32_t tok, float* x) const {
        return emb_row && kc::native_row_to_float(emb_type, emb_data + (size_t) tok * emb_row, x, N);
    }

    bool project_head(const float* x, float* logits, int64_t rows) const {
        if (products && products->has_head()) return products->project_head(x, logits, rows);
        if (!head_row) return false;
        std::vector<float> w((size_t) N);
        for (int64_t i = 0; i < rows; ++i) {
            if (!kc::native_row_to_float(head_type, head_data + (size_t) i * head_row, w.data(), N)) return false;
            double sum = 0;
            for (int64_t d = 0; d < N; ++d) sum += (double) w[(size_t) d] * x[d];
            logits[i] = (float) sum;
        }
        return true;
    }

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

    /// One row of one expert, dequantized and dotted with `x`.  The blob is [n_in][rows][n_expert], so a row is
    /// `row_bytes` apart and expert `ex` starts `rows` rows in.  Scratch stays at one row.
    float expert_row_dot(const Expert& e, int ex, int64_t r, const float* x, std::vector<float>& buf) {
        const uint8_t* src = e.data + (size_t) ex * (size_t) e.rows * e.row_bytes + (size_t) r * e.row_bytes;
        if (!kc::native_row_to_float(e.type, src, buf.data(), e.cols)) return 0.0f;
        double ss = 0;
        for (int64_t i = 0; i < e.cols; ++i) ss += (double) buf[(size_t) i] * x[i];
        return (float) ss * (e.scale != nullptr ? e.scale[ex] : 1.0f);
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

    // matvec over several inputs in one pass of the weights; every output keeps matvec's accumulation order
    template <int R>
    static void dot_rows(const float* w, const float* xt, float* const* y, int64_t n_in, int64_t r0, int64_t r1) {
        for (int64_t o = r0; o < r1; ++o) {
            const float* row = w + (size_t) o * n_in;
            double ss[R] = {};
            for (int64_t i = 0; i < n_in; ++i) {
                const double wi = row[i];
                const float* xi = xt + (size_t) i * R;
                for (int r = 0; r < R; ++r) ss[r] += wi * xi[r];
            }
            for (int r = 0; r < R; ++r) y[r][o] = (float) ss[r];
        }
    }

    void matvec_rows(const std::vector<float>& w, const std::vector<const float*>& x, const std::vector<float*>& y,
                     int64_t n_in, int64_t n_out, int threads) {
        using Dot = void (*)(const float*, const float*, float* const*, int64_t, int64_t, int64_t);
        static constexpr Dot dots[] = {nullptr, nullptr, dot_rows<2>, dot_rows<3>, dot_rows<4>,
                                       dot_rows<5>, dot_rows<6>, dot_rows<7>, dot_rows<8>};
        std::vector<float> xt;
        for (size_t at = 0; at < x.size();) {
            const int R = (int) std::min<size_t>(8, x.size() - at);
            if (R == 1) { matvec(w, x[at], y[at], n_in, n_out, threads); ++at; continue; }
            xt.resize((size_t) n_in * R);
            for (int64_t i = 0; i < n_in; ++i)
                for (int r = 0; r < R; ++r) xt[(size_t) i * R + r] = x[at + (size_t) r][i];
            float* const* yr = y.data() + at;
            const auto span = [&](int64_t r0, int64_t r1) { dots[R](w.data(), xt.data(), yr, n_in, r0, r1); };
            if (threads <= 1 || n_out < 128) {
                span(0, n_out);
            } else {
                std::vector<std::thread> ts;
                const int64_t chunk = (n_out + threads - 1) / threads;
                for (int64_t r = 0; r < n_out; r += chunk)
                    ts.emplace_back([&, r, chunk] { span(r, std::min(n_out, r + chunk)); });
                for (auto& t : ts) t.join();
            }
            at += (size_t) R;
        }
    }

    void mat(const std::string& name, const float* x, float* y, int64_t n_in, int64_t n_out, int threads,
             std::string& err) {
        const auto& w = matrix(name, err);
        if (!err.empty()) return;
        matvec(w, x, y, n_in, n_out, threads);
    }

    struct RoutedInput {
        const float* input;
        const int64_t* ids;
        const float* weights;
        float* output;
    };
    void grouped_moe(const Layer& layer, const std::vector<RoutedInput>& rows);

    void reset_state();
    bool load(const GgufModel& model, const ModelSpec& spec, std::string& err);
    bool window(const int32_t* tokens, int count, int64_t pos0, int threads, float* logits, std::string& err,
                const std::function<int()>& keep = {});
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
    const auto emb_vocab = t->shape[1];
    emb_row = kc::native_row_bytes(emb_type, N);
    t = m->find("output.weight", &at);
    if (t == nullptr || t->shape.size() != 2 || (int64_t) t->shape[0] != N || !m->in_bounds(*t, at)) {
        err = "inkling2: output.weight is missing or of the wrong shape";
        return false;
    }
    head_t = t;
    head_data = m->shard(at).tensor_data(*t);
    head_type = (int) t->type;
    head_row = kc::native_row_bytes(head_type, N);
    if (!emb_row || !head_row || emb_vocab != (size_t) s.vocab_padded || t->shape[1] != (size_t) s.vocab_padded) {
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
        // only 2-D tensors go through matrix(); the 1-D norms and res queries are loaded with vector()
        std::vector<std::string> dense_names = {L.q, L.k, L.v, L.o, L.gate, L.qclip, L.sink, L.rel, L.relbias,
                                                L.relclip, L.kconv, L.vconv, L.postconv, L.ffnconv, L.router};
        // attn_res_query is [N] on sliding-window layers and [N,3] on full-attention layers
        if (l > 0) {
            size_t rat = 0;
            const TensorInfo* rt = m->find(L.resq, &rat);
            if (rt != nullptr && rt->shape.size() >= 2) dense_names.push_back(L.resq);
        }
        if (!L.swa) {
            dense_names.push_back(L.kproj);
            dense_names.push_back(L.vproj);
        }
        for (const auto& name : dense_names) {
            std::string e;
            (void) matrix(name, e);
            if (!e.empty()) { err = e; return false; }
        }
        std::vector<std::string> vec_names = {L.ffnq, L.ffnnorm};
        if (l > 0) {
            size_t rat = 0;
            const TensorInfo* rt = m->find(L.resq, &rat);
            if (rt != nullptr && rt->shape.size() == 1) vec_names.push_back(L.resq);
        }
        if (L.swa) vec_names.push_back(L.attn_norm);
        else {
            vec_names.push_back(L.qin);
            vec_names.push_back(L.kin);
            vec_names.push_back(L.vin);
        }
        for (const auto& name : vec_names)
            if (!need_vec(name.c_str(), N)) return false;
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
        std::string qe;
        (void) matrix(C.query, qe);   // [n_embd, 4]: one query per controller
        if (!qe.empty()) { err = qe; return false; }
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

    heat.assign((size_t) (s.n_layer * NE), 0);
    slots.resize((size_t) NSLOT);
    reset_state();
    return true;
}

void PlainInkling::Impl::reset_state() {
    const int64_t channels = 2 * NKV * HD + 2 * N;
    for (size_t i = 0; i < slots.size(); ++i) {
        auto& sl = slots[i];
        if (products) products->invalidate_kv(i);
        sl.k.assign((size_t) s.relative_extent * NKV * HD, 0.0f);
        sl.v.assign((size_t) s.relative_extent * NKV * HD, 0.0f);
        sl.pos.assign((size_t) s.relative_extent, -1);
        sl.cells = 0;
        sl.conv.assign((size_t) (NCONV - 1) * channels, 0.0f);
    }
}

void PlainInkling::reset() {
    p_->reset_state();
    p_->taps_valid = false;
    if (p_->dspark) p_->dspark->reset();
}

void PlainInkling::Impl::grouped_moe(const Layer& layer, const std::vector<RoutedInput>& rows) {
    std::vector<InklingProducts::RoutedRow> pairs;
    for (int32_t t = 0; t < (int32_t) rows.size(); ++t)
        for (int32_t j = 0; j < NK; ++j)
            pairs.push_back({rows[(size_t) t].ids[j], t, j});
    InklingProducts::group_routes(pairs);
    std::vector<float> contributions(pairs.size() * (size_t) N);
    for (size_t first = 0; first < pairs.size();) {
        std::vector<int64_t> ids;
        std::vector<int32_t> counts;
        size_t end = first;
        while (end < pairs.size() && ids.size() < (size_t) NK) {
            const int64_t id = pairs[end].expert;
            const size_t start = end;
            while (end < pairs.size() && pairs[end].expert == id) ++end;
            ids.push_back(id);
            counts.push_back((int32_t) (end - start));
        }
        const size_t n = end - first;
        std::vector<float> input(n * (size_t) N), gu(n * (size_t) (2 * FF)), hidden(n * (size_t) FF);
        std::vector<float> down(n * (size_t) N);
        for (size_t i = 0; i < n; ++i)
            std::copy_n(rows[(size_t) pairs[first + i].row].input, N, input.data() + i * (size_t) N);
        const int groups = (int) ids.size();
        products->grouped_rows(layer.gu.type, layer.gu.data, (size_t) layer.gu.rows * layer.gu.row_bytes,
                               ids.data(), groups, input.data(), N, 2 * FF, gu.data(), counts.data());
        for (size_t i = 0; i < n; ++i) {
            const int64_t id = pairs[first + i].expert;
            const float scale = layer.gu.scale ? layer.gu.scale[id] : 1.0f;
            const float* g = gu.data() + i * (size_t) (2 * FF);
            for (int64_t j = 0; j < FF; ++j)
                hidden[i * (size_t) FF + j] = situ(g[j] * scale, g[FF + j] * scale, (float) s.situ_beta,
                                                    (float) s.situ_linear_beta);
        }
        products->grouped_rows(layer.down.type, layer.down.data, (size_t) layer.down.rows * layer.down.row_bytes,
                               ids.data(), groups, hidden.data(), FF, N, down.data(), counts.data());
        for (size_t i = 0; i < n; ++i) {
            const auto& pair = pairs[first + i];
            const size_t dst = ((size_t) pair.row * (size_t) NK + (size_t) pair.rank) * (size_t) N;
            const float scale = layer.down.scale ? layer.down.scale[pair.expert] : 1.0f;
            for (int64_t j = 0; j < N; ++j) contributions[dst + j] = down[i * (size_t) N + j] * scale;
        }
        first = end;
    }
    // Keep the sum in top-k order, independent of the expert packing order.
    for (size_t t = 0; t < rows.size(); ++t) {
        std::fill_n(rows[t].output, N, 0.0f);
        for (int64_t k = 0; k < NK; ++k)
            for (int64_t j = 0; j < N; ++j)
                rows[t].output[j] += rows[t].weights[k] * contributions[(t * (size_t) NK + k) * (size_t) N + j];
    }
}

bool PlainInkling::Impl::window(const int32_t* tokens, int count, int64_t pos0, int threads,
                                float* logits, std::string& err, const std::function<int()>& keep) {
    for (int t = 0; t < count; ++t) {
        if (tokens[t] < 0 || tokens[t] >= s.vocab_padded) {
            err = "inkling2: token out of range";
            return false;
        }
    }
    const int64_t nrel = s.relative_length;
    const int64_t extent = s.relative_extent;
    const int64_t channels = 2 * NKV * HD + 2 * N;
    const float inv_sqrt_hd = 1.0f / std::sqrt((float) HD);
    const float inv_sqrt_n = 1.0f / std::sqrt((float) N);
    std::vector<float> x(N), hn(N), mixed(N), tmp(N), out(N);
    std::vector<float> hv(FF), rowbuf((size_t) std::max(N, FF));
    struct Backup {
        int64_t cells = 0;
        std::vector<int32_t> positions;
        std::vector<float> k, v, conv;
    };
    const size_t kv_width = (size_t) (NKV * HD);
    const size_t conv_width = (size_t) ((NCONV - 1) * channels);
    std::vector<Backup> backups(keep ? slots.size() : 0);
    // restore() only touches host memory and cannot throw: on the error path (Rollback) the resident copies are marked
    // stale; after a verified window the restored cells are copied to them below (refresh).
    const auto restore = [&](int kept, bool stale) {
        if (!keep) return;
        for (size_t i = 0; i < slots.size(); ++i) {
            auto& slot = slots[i];
            const auto& old = backups[i];
            if (old.positions.empty()) continue;
            slot.cells = std::min(old.cells + kept, extent);
            std::copy_n(old.conv.data() + (size_t) kept * conv_width, conv_width, slot.conv.data());
            for (int t = kept; t < count; ++t) {
                const size_t cell = (size_t) ((pos0 + t) % extent);
                slot.pos[cell] = old.positions[(size_t) t];
                std::copy_n(old.k.data() + (size_t) t * kv_width, kv_width, slot.k.data() + cell * kv_width);
                std::copy_n(old.v.data() + (size_t) t * kv_width, kv_width, slot.v.data() + cell * kv_width);
            }
            if (products && kept < count && stale) products->invalidate_kv(i);
        }
    };
    struct Rollback {
        const decltype(restore)& fn;
        bool armed;
        ~Rollback() { if (armed) fn(0, true); }
    } rollback{restore, (bool) keep};
    struct Row {
        std::vector<float> cur, prelude, bank_out, features, expert_in, expert_out, weights;
        std::vector<int64_t> ids;
        std::vector<std::vector<float>> blocks, prelude_blocks, bank;
    };
    std::vector<Row> rows((size_t) count);
    for (int t = 0; t < count; ++t) {
        auto& row = rows[(size_t) t];
        if (!embedding(tokens[t], x.data())) {
            err = "inkling2: cannot dequantize the token embedding";
            return false;
        }
        row.cur.resize((size_t) N);
        rms_norm(x.data(), tok_norm.data(), row.cur.data(), N, eps);
        row.features.resize(draft_features.size());
        row.expert_out.resize((size_t) N);
    }

    // ---- the helpers of one layer ----
    auto dense_of = [&](const std::string& name) -> const float* {
        auto it = dense.find(name);
        return it == dense.end() ? nullptr : it->second.data();
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
                acc += (double) wt[(size_t) c * NCONV + lag] * src[c];
            }
            o[c] = (cur[c] + (float) acc) * scale;
        }
        if (dyn != nullptr) {
            const int64_t per_head = n_ch / n_heads;
            for (int64_t hd = 0; hd < n_heads; ++hd)
                for (int64_t lag = 0; lag < NCONV; ++lag) {
                    const float* src = lag == NCONV - 1 ? cur
                        : sl.conv.data() + (size_t) (NCONV - 2 - lag) * channels + (size_t) offset;
                    const float tap = dyn[(size_t) hd * NCONV + (NCONV - 1 - lag)];
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

    // The draft taps the layer inputs in visit order, not the slots used for recurrent state.
    int visit = 0;
    const auto tap = [&](Row& row, int id, const float* values) {
        if (row.features.empty()) return;
        for (size_t j = 0; j < PlainDspark::taps.size(); ++j)
            if (PlainDspark::taps[j] == id) std::copy_n(values, N, row.features.data() + j * N);
    };
    // A layer's projections take every row of the window in one pass over the weights; each row keeps its own
    // accumulation order, so a row's numbers do not depend on how many rows share the window.  The short
    // convolutions, the KV append and the attention still go one row at a time, in row order.
    auto multiply_rows = [&](const std::string& name, const std::vector<const float*>& in, const std::vector<float*>& o,
                             int64_t n_in, int64_t n_out) {
        std::string e;
        const auto& w = matrix(name, e);
        if (!e.empty()) { err = e; return; }
        matvec_rows(w, in, o, n_in, n_out, threads);
    };
    struct Prep {
        std::vector<float> inpSA, mx, in0, in1, in2, q, k, v, r, kdyn, vdyn, sink, gate, att, o, normed, route;
        const float* inputs[3] = {nullptr, nullptr, nullptr};
    };
    std::vector<Prep> preps((size_t) count);
    auto prepare = [&](int64_t ip, int64_t slot_id, const std::vector<std::vector<float>>& messages, const Loop* loop,
                       Backup* old) {
        if (!err.empty()) return;
        const bool messages_present = loop != nullptr;
        Layer& L = layers[(size_t) ip];
        Slot& sl = slots[(size_t) slot_id];
        const auto& relbias = matrix(L.relbias, err);
        if (!err.empty()) return;
        std::vector<const float*> in_rows[3];
        for (int t = 0; t < count && err.empty(); ++t) {
            auto& row = rows[(size_t) t];
            auto& P = preps[(size_t) t];
            const float* msg = messages_present ? messages[(size_t) t].data() : nullptr;
            float* cur_in = row.cur.data();
            auto& bl = row.blocks;
            tap(row, visit, cur_in);
            P.inpSA.assign(cur_in, cur_in + N);
            P.inputs[0] = P.inputs[1] = P.inputs[2] = cur_in;
            if (dense_of(L.resq) != nullptr) {
                // one learned query on a sliding-window layer, three on a full-attention layer (one per Q/K/V input)
                P.mx.resize(3 * N);
                const float* rq = dense_of(L.resq);
                mix(bl, cur_in, rq, P.mx.data());
                if (L.swa) {
                    P.inputs[0] = P.inputs[1] = P.inputs[2] = P.mx.data();
                } else {
                    mix(bl, cur_in, rq + N, P.mx.data() + N);
                    mix(bl, cur_in, rq + 2 * N, P.mx.data() + 2 * N);
                    P.in0.assign(P.mx.begin(), P.mx.begin() + N);
                    P.in1.assign(P.mx.begin() + N, P.mx.begin() + 2 * N);
                    P.in2.assign(P.mx.begin() + 2 * N, P.mx.begin() + 3 * N);
                    P.inputs[0] = P.in0.data(); P.inputs[1] = P.in1.data(); P.inputs[2] = P.in2.data();
                }
            }
            if (messages_present) {
                for (int j = 0; j < 3; ++j) {
                    std::vector<float> ctr(N);
                    control(P.inputs[j], msg + (size_t) j * N, *loop, j, ctr.data());
                    if (j == 0) P.in0 = std::move(ctr); else if (j == 1) P.in1 = std::move(ctr); else P.in2 = std::move(ctr);
                    P.inputs[j] = j == 0 ? P.in0.data() : (j == 1 ? P.in1.data() : P.in2.data());
                }
            }
            if (L.swa) {
                if (messages_present) {
                    for (int j = 0; j < 3; ++j) {
                        std::vector<float> nz(N);
                        rms_norm(P.inputs[j], dense_of(L.attn_norm), nz.data(), N, eps);
                        if (j == 0) P.in0 = std::move(nz); else if (j == 1) P.in1 = std::move(nz); else P.in2 = std::move(nz);
                        P.inputs[j] = j == 0 ? P.in0.data() : (j == 1 ? P.in1.data() : P.in2.data());
                    }
                } else {
                    P.in0.resize(N);
                    rms_norm(P.inputs[0], dense_of(L.attn_norm), P.in0.data(), N, eps);
                    P.inputs[0] = P.inputs[1] = P.inputs[2] = P.in0.data();
                }
            } else {
                std::vector<float> n0(N), n1(N), n2(N);
                rms_norm(P.inputs[0], dense_of(L.qin), n0.data(), N, eps);
                rms_norm(P.inputs[1], dense_of(L.kin), n1.data(), N, eps);
                rms_norm(P.inputs[2], dense_of(L.vin), n2.data(), N, eps);
                P.in0 = std::move(n0); P.in1 = std::move(n1); P.in2 = std::move(n2);
                P.inputs[0] = P.in0.data(); P.inputs[1] = P.in1.data(); P.inputs[2] = P.in2.data();
            }
            for (int j = 0; j < 3; ++j) in_rows[j].push_back(P.inputs[j]);
            P.q.resize(NH * HD); P.k.resize(NKV * HD); P.v.resize(NKV * HD); P.r.resize(NH * nrel);
            P.sink.resize(NH); P.gate.resize(NH * HD); P.att.resize(NH * HD); P.o.resize(N);
            P.normed.resize(N); P.route.resize(NE);
            if (!L.swa) { P.kdyn.resize(NCONV * NKV); P.vdyn.resize(NCONV * NKV); }
        }
        if (!err.empty()) return;
        const auto outs = [&](std::vector<float> Prep::*field) {
            std::vector<float*> o;
            o.reserve(preps.size());
            for (auto& P : preps) o.push_back((P.*field).data());
            return o;
        };
        multiply_rows(L.q, in_rows[0], outs(&Prep::q), N, NH * HD);
        multiply_rows(L.k, in_rows[1], outs(&Prep::k), N, NKV * HD);
        multiply_rows(L.v, in_rows[2], outs(&Prep::v), N, NKV * HD);
        multiply_rows(L.rel, in_rows[0], outs(&Prep::r), N, NH * nrel);
        if (!L.swa) {
            multiply_rows(L.kproj, in_rows[1], outs(&Prep::kdyn), N, NCONV * NKV);
            multiply_rows(L.vproj, in_rows[2], outs(&Prep::vdyn), N, NCONV * NKV);
        }
        multiply_rows(L.sink, in_rows[0], outs(&Prep::sink), N, NH);
        multiply_rows(L.gate, in_rows[0], outs(&Prep::gate), N, NH * HD);
        if (!err.empty()) return;

        for (int t = 0; t < count && err.empty(); ++t) {
            auto& P = preps[(size_t) t];
            const int64_t pos = pos0 + t;
            const auto ratio = std::max(1.0, (double) (pos + 1) / (double) s.log_scale_floor);
            const float tau = (float) (1.0 + s.log_scale_alpha * std::log(ratio));
            auto& q = P.q;
            auto& k = P.k;
            auto& v = P.v;
            auto& r = P.r;
            float* att = P.att.data();
            convolve(k.data(), NKV * HD, L.kconv, L.swa ? nullptr : P.kdyn.data(), sl, 0, tmp.data());
            k.assign(tmp.begin(), tmp.begin() + NKV * HD);
            convolve(v.data(), NKV * HD, L.vconv, L.swa ? nullptr : P.vdyn.data(), sl, NKV * HD, tmp.data());
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
            const int64_t cell = pos % extent;
            const int64_t cells = sl.cells = std::min(sl.cells + 1, extent);
            sl.pos[(size_t) cell] = (int32_t) pos;
            for (int64_t h = 0; h < NKV; ++h) {
                std::memcpy(sl.k.data() + ((size_t) cell * NKV + h) * HD, k.data() + h * HD, HD * sizeof(float));
                std::memcpy(sl.v.data() + ((size_t) cell * NKV + h) * HD, v.data() + h * HD, HD * sizeof(float));
            }
            const float* sink = P.sink.data();
            if (products) {
                products->attention(q.data(), sl.k.data(), sl.v.data(), r.data(), relbias.data(), sink, sl.pos.data(),
                                    pos, cells, NH, NKV, HD, nrel, extent, L.swa ? s.swa_window : 0, att, (size_t) slot_id);
            } else {
                for (int64_t h = 0; h < NH; ++h) {
                    const int64_t hkv = h / (NH / NKV);
                    std::vector<float> scores((size_t) cells + 1);
                    for (int64_t c = 0; c < cells; ++c) {
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
                    scores[(size_t) cells] = 0.0f;   // the zero sink
                    const float mx = *std::max_element(scores.begin(), scores.end());
                    double sum = 0;
                    for (auto& z : scores) { z = std::exp(z - mx); sum += z; }
                    for (int64_t d = 0; d < HD; ++d)
                        att[h * HD + d] = 0.0f;
                    for (int64_t c = 0; c < cells; ++c) {
                        const float pw = (float) (scores[(size_t) c] / sum);
                        const float* vc = sl.v.data() + ((size_t) c * NKV + hkv) * HD;
                        for (int64_t d = 0; d < HD; ++d) att[h * HD + d] += pw * vc[d];
                    }
                }
            }
            for (int64_t i = 0; i < NH * HD; ++i) att[i] *= sigmoid(P.gate[(size_t) i]);
            // the K/V channels after this row; the later channels are copied after their own phases
            if (old) std::copy(sl.conv.begin(), sl.conv.end(), old->conv.data() + ((size_t) t + 1) * conv_width);
        }
        if (!err.empty()) return;
        std::vector<const float*> att_rows;
        att_rows.reserve(preps.size());
        for (auto& P : preps) att_rows.push_back(P.att.data());
        multiply_rows(L.o, att_rows, outs(&Prep::o), NH * HD, N);
        if (!err.empty()) return;

        std::vector<const float*> normed_rows;
        for (int t = 0; t < count && err.empty(); ++t) {
            auto& row = rows[(size_t) t];
            auto& P = preps[(size_t) t];
            const float* msg = messages_present ? messages[(size_t) t].data() : nullptr;
            convolve(P.o.data(), N, L.postconv, nullptr, sl, 2 * NKV * HD, out.data());
            if (!err.empty()) return;
            if (old)
                for (int64_t lag = 0; lag < NCONV - 1; ++lag) {
                    const size_t offset = (size_t) (lag * channels + 2 * NKV * HD);
                    std::copy_n(sl.conv.data() + offset, N, old->conv.data() + ((size_t) t + 1) * conv_width + offset);
                }
            float* cur_in = row.cur.data();
            auto& bl = row.blocks;
            if (ip % s.attn_res_block == 0) bl.push_back(P.inpSA);
            for (int64_t i = 0; i < N; ++i) cur_in[i] = ip % s.attn_res_block == 0 ? out[i] : cur_in[i] + out[i];

            std::vector<float> ffn_cur(N);
            mix(bl, cur_in, dense_of(L.ffnq), ffn_cur.data());
            if (messages_present) {
                std::vector<float> ctr(N);
                control(ffn_cur.data(), msg + 3 * N, *loop, 3, ctr.data());
                ffn_cur = std::move(ctr);
            }
            if (!err.empty()) return;
            rms_norm(ffn_cur.data(), dense_of(L.ffnnorm), P.normed.data(), N, eps);
            normed_rows.push_back(P.normed.data());
        }
        if (!err.empty()) return;
        // sigmoid router, top-8, the selected probabilities normalized and scaled by route_scale
        multiply_rows(L.router, normed_rows, outs(&Prep::route), N, NE);
        if (!err.empty()) return;

        for (int t = 0; t < count; ++t) {
            auto& row = rows[(size_t) t];
            auto& P = preps[(size_t) t];
            std::vector<float> prob((size_t) NE);
            for (int64_t e = 0; e < NE; ++e) prob[(size_t) e] = sigmoid(P.route[(size_t) e]);
            std::vector<int64_t> ids(NE);
            std::iota(ids.begin(), ids.end(), 0);
            std::partial_sort(ids.begin(), ids.begin() + NK, ids.end(),
                              [&](int64_t a, int64_t b) {
                                  const float x = prob[(size_t) a] + L.exp_b[a], y = prob[(size_t) b] + L.exp_b[b];
                                  return x == y ? a < b : x > y;
                              });
            for (int64_t j = 0; j < NK; ++j) ++heat[(size_t) (ip * NE + ids[(size_t) j])];
            std::vector<float> w((size_t) NK);
            double wsum = 0;
            for (int64_t j = 0; j < NK; ++j) { w[(size_t) j] = prob[(size_t) ids[(size_t) j]]; wsum += w[(size_t) j]; }
            for (auto& z : w) z = (float) (z / (wsum + 1e-20) * s.expert_weights_scale);

            if (products) {
                row.expert_in = P.normed;
                row.ids = std::move(ids);
                row.weights = std::move(w);
            } else {
                std::fill(row.expert_out.begin(), row.expert_out.end(), 0.0f);
                for (int64_t j = 0; j < NK; ++j) {
                    const int64_t eid = ids[(size_t) j];
                    for (int64_t i = 0; i < FF; ++i) {
                        const float g = expert_row_dot(L.gu, (int) eid, i, P.normed.data(), rowbuf);
                        const float u = expert_row_dot(L.gu, (int) eid, FF + i, P.normed.data(), rowbuf);
                        hv[(size_t) i] = situ(g, u, (float) s.situ_beta, (float) s.situ_linear_beta);
                    }
                    for (int64_t i = 0; i < N; ++i)
                        row.expert_out[(size_t) i] += w[(size_t) j] * expert_row_dot(L.down, (int) eid, i, hv.data(), rowbuf);
                }
            }
        }
    };
    const auto layer = [&](int64_t ip, int64_t slot_id, const Loop* loop) {
        auto& slot = slots[(size_t) slot_id];
        Backup* old = keep ? &backups[(size_t) slot_id] : nullptr;
        if (old) {
            old->cells = slot.cells;
            old->k.resize((size_t) count * kv_width);
            old->v.resize((size_t) count * kv_width);
            old->conv.resize(((size_t) count + 1) * conv_width);
            old->positions.resize((size_t) count);
            std::copy(slot.conv.begin(), slot.conv.end(), old->conv.begin());
            for (int t = 0; t < count; ++t) {
                const size_t cell = (size_t) ((pos0 + t) % extent);
                old->positions[(size_t) t] = slot.pos[cell];
                std::copy_n(slot.k.data() + cell * kv_width, kv_width, old->k.data() + (size_t) t * kv_width);
                std::copy_n(slot.v.data() + cell * kv_width, kv_width, old->v.data() + (size_t) t * kv_width);
            }
        }
        std::vector<std::vector<float>> messages(loop ? (size_t) count : 0);
        for (int t = 0; t < (int) messages.size(); ++t) {
            auto& row = rows[(size_t) t];
            auto& message = messages[(size_t) t];
            message.resize((size_t) (4 * N));
            const float* mq = dense_of(loop->query);
            for (int c = 0; c < 4; ++c)
                mix(row.bank, row.bank_out.data(), mq + c * N, message.data() + c * N);
        }
        prepare(ip, slot_id, messages, loop, old);
        if (!err.empty()) return;
        const auto& L = layers[(size_t) ip];
        if (products) {
            std::vector<RoutedInput> routed;
            routed.reserve(rows.size());
            for (auto& row : rows)
                routed.push_back({row.expert_in.data(), row.ids.data(), row.weights.data(), row.expert_out.data()});
            grouped_moe(L, routed);
        }
        for (int t = 0; t < count; ++t) {
            auto& row = rows[(size_t) t];
            convolve(row.expert_out.data(), N, L.ffnconv, nullptr, slot, 2 * NKV * HD + N, tmp.data());
            for (int64_t i = 0; i < N; ++i) row.cur[(size_t) i] += tmp[(size_t) i];
            if (old) {
                // Only the FFN channels changed in this phase. Keep the attention channels from prepare().
                for (int64_t lag = 0; lag < NCONV - 1; ++lag) {
                    const size_t offset = (size_t) (lag * channels + 2 * NKV * HD + N);
                    std::copy_n(slot.conv.data() + offset, N, old->conv.data() + ((size_t) t + 1) * conv_width + offset);
                }
            }
        }
        ++visit;
    };

    for (int64_t l = 0; l < s.loop_start && err.empty(); ++l) layer(l, l, nullptr);
    for (auto& row : rows) { row.prelude = row.cur; row.prelude_blocks = row.blocks; }
    for (int64_t pass = 0; pass < s.loop_count && err.empty(); ++pass) {
        if (pass > 0) {
            for (auto& row : rows) {
                row.bank.assign(row.blocks.begin() + (ptrdiff_t) row.prelude_blocks.size() + 1, row.blocks.end());
                row.bank_out = row.cur;
                row.cur = row.prelude;
                row.blocks = row.prelude_blocks;
            }
        }
        for (int64_t ip = s.loop_start; ip < s.loop_end && err.empty(); ++ip) {
            const int64_t slot = pass == 0 ? ip : s.n_layer + (pass - 1) * (s.loop_end - s.loop_start) + ip - s.loop_start;
            layer(ip, slot, pass == 0 ? nullptr : &loops[(size_t) (ip - s.loop_start)]);
        }
    }
    for (int64_t l = s.loop_end; l < s.n_layer && err.empty(); ++l) layer(l, l, nullptr);
    if (!err.empty()) return false;

    last_logits_.resize((size_t) s.vocab_unpadded);
    const float cap = (float) s.final_logit_softcap;
    for (int t = 0; t < count; ++t) {
        auto& row = rows[(size_t) t];
        mix(row.blocks, row.cur.data(), out_query.data(), tmp.data());
        rms_norm(tmp.data(), out_norm.data(), hn.data(), N, eps);
        tap(row, visit, hn.data());
        if (!project_head(hn.data(), last_logits_.data(), s.vocab_unpadded)) {
            err = "inkling2: cannot dequantize a head row";
            return false;
        }
        if (cap > 0)
            for (float& z : last_logits_) z = cap * std::tanh(z / cap);
        if (logits) std::copy(last_logits_.begin(), last_logits_.end(), logits + t * s.vocab_unpadded);
    }
    const int kept = keep ? keep() : count;
    if (kept < 1 || kept > count) { err = "inkling2: invalid verifier commit length"; return false; }
    restore(kept, false);
    rollback.armed = false;
    if (products && kept < count)
        for (size_t i = 0; i < slots.size(); ++i)
            if (!backups[i].positions.empty())
                products->refresh_kv(i, slots[i].k.data(), slots[i].v.data(), slots[i].pos.data(),
                                     (pos0 + kept) % extent, count - kept, extent, kv_width);
    for (int t = 0; t < kept; ++t) {
        auto& row = rows[(size_t) t];
        if (products) products->capture(row.features.data(), row.features.size());
        taps_valid = true;
        if (dspark) {
            if (products) products->read_taps(row.features.data(), row.features.size());
            dspark->commit(row.features.data());
        }
    }
    return true;
}

PlainInkling::PlainInkling() : p_(std::make_unique<Impl>()) {}
PlainInkling::~PlainInkling() = default;

bool PlainInkling::load(const GgufModel& m, const ModelSpec& s, std::string& err) {
    return p_->load(m, s, err);
}

bool PlainInkling::run(const int32_t* tokens, int64_t n, int64_t pos, int rows, float* logits, std::string& err) {
    p_->taps_valid = false;
    if (n < 1 || rows < 1 || rows > n || tokens == nullptr) {
        err = "inkling2: no tokens to run";
        return false;
    }
    // The reference walks one token at a time.  Prompt and decode use the same path, so a token's answer is the same
    // whether it was alone in a window or at the end of a prompt.  Thread count stays modest: SYNC5-TILE is building
    // on this machine while this reference is used.
    const unsigned hw = std::thread::hardware_concurrency();
    const int threads = std::max(1, (int) hw / 8);
    try {
        for (int64_t t = 0; t < n;) {
            const bool save = logits && t >= n - rows;
            const int64_t available = save || !logits ? n - t : n - rows - t;
            const int count = (int) std::min<int64_t>(p_->products ? PlainDspark::block : 1, available);
            float* out = save ? logits + (t - (n - rows)) * p_->s.vocab_unpadded : nullptr;
            if (!p_->window(tokens + t, count, pos + t, threads, out, err)) return false;
            t += count;
        }
    } catch (const std::exception& e) { p_->taps_valid = false; err = e.what(); return false; }
    return true;
}

bool PlainInkling::verify(const int32_t* tokens, int count, int64_t pos, float* logits,
                         const std::function<int()>& keep, std::string& err) {
    if (!p_->products || !p_->dspark || count < 2 || count > PlainDspark::block ||
        count > p_->s.relative_extent || pos < 0 || !tokens || !logits || !keep) {
        err = "inkling2: invalid GPU verifier window";
        return false;
    }
    p_->taps_valid = false;
    const int threads = std::max(1, (int) std::thread::hardware_concurrency() / 8);
    try {
        return p_->window(tokens, count, pos, threads, logits, err, keep);
    } catch (const std::exception& e) { p_->taps_valid = false; err = e.what(); return false; }
}

const float* PlainInkling::gpu_taps() const {
    return p_->products && p_->taps_valid ? p_->products->taps() : nullptr;
}

bool PlainInkling::enable_gpu(std::string& err) {
    size_t raw = 0;
    for (const auto& layer : p_->layers) {
        for (const auto* e : {&layer.gu, &layer.down}) {
            if (!kernels::iq_supported(e->type) || (e->rows * e->cols) % 256 != 0) {
                err = "inkling2: expert format has no shared GPU dequantizer";
                return false;
            }
            raw = std::max(raw, e->row_bytes * (size_t) e->rows);
        }
    }
    try {
        p_->draft_features.resize(PlainDspark::taps.size() * (size_t) p_->N);
        p_->products = std::make_unique<InklingProducts>(p_->NK, p_->N, p_->FF, raw * (size_t) p_->NK,
                                                        p_->NH, p_->NKV, p_->HD, p_->s.relative_length,
                                                        p_->s.relative_extent);
    } catch (const std::exception& e) { err = e.what(); return false; }
    return true;
}

bool PlainInkling::configure_experts(const std::string& profile, int64_t context, int64_t reserve_mib,
                                     int64_t vram_free_mib, int64_t expert_slots, std::string& err) {
    const char* mode = strata::env_get("INKLING_PLACEMENT");
    const std::string_view placement = mode ? mode : "host";
    if (placement == "host") return true;
    const char* toggle = strata::env_get("INKLING_RESIDUAL");
    const bool residual = !(toggle && std::string_view(toggle) == "0");
    try {
        if (placement == "auto" && p_->dspark && !p_->dspark->enable_gpu(err)) {
            if (!err.empty()) return false;
            std::fprintf(stderr, "xestrata: DSpark matrices do not fit or need FP64; keeping the host drafter\n");
        }
        if (!p_->products->load_head(p_->head_type, p_->head_data, p_->head_row, p_->N, p_->s.vocab_padded))
            std::fprintf(stderr, "xestrata: Inkling head does not fit the VRAM budget; keeping the host head\n");
    } catch (const std::exception& e) { err = e.what(); return false; }
    if (placement != "auto") return true;
    if (residual && p_->dspark && !p_->dspark->place_attention(context, err)) return false;
    using Pair = std::pair<int32_t, int32_t>;
    std::vector<Pair> prior;
    int64_t slots = 0;
    const auto& s = p_->s;
    if (!profile.empty() && !read_expert_profile(profile, s.n_layer, p_->NE, prior, slots, err)) return false;
    size_t free = 0, total = 0;
    if (!gpu::mem_info(&free, &total)) { err = "inkling2: cannot query expert placement budget"; return false; }
    try {
        if (residual) p_->products->configure_kv(p_->slots.size(), (size_t) s.relative_extent, (size_t) (p_->NKV * p_->HD),
                                   free - std::min<uint64_t>(free, vram_room(total)));
    } catch (const std::exception& e) { err = e.what(); return false; }
    if (!gpu::mem_info(&free, &total)) { err = "inkling2: cannot query VRAM after KV placement"; return false; }
    const auto host = sample_host_memory();
    const uint64_t hr = detail::ram_headroom(host ? host->limit / 32 : 0);
    const uint64_t dram = host ? host->available - std::min(host->available, hr) : 0;
    const auto gu_bytes = [&](int64_t l) {
        return (size_t) p_->layers[(size_t) l].gu.rows * p_->layers[(size_t) l].gu.row_bytes;
    };
    const auto down_bytes = [&](int64_t l) {
        return (size_t) p_->layers[(size_t) l].down.rows * p_->layers[(size_t) l].down.row_bytes;
    };
    uint64_t all = 0, batch = 0;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        all += (uint64_t) (gu_bytes(l) + down_bytes(l)) * (uint64_t) p_->NE;
        batch = std::max<uint64_t>(batch, gu_bytes(l) + down_bytes(l));
    }
    // The tier's room and the slots each layer takes by the tiers' one rule (expert_tier.hpp: plan_tier_room and
    // plan_tier_fill, PlainModel's), so that --vram-reserve-mib, --vram-free-mib, --expert-cache and a later change of
    // the rule reach this model as well.  This model's facts go into it: a slot is one expert's gate/up and down
    // (InklingProducts::slot_bytes), a window is kWindowRows rows of NK experts, and a layer takes slots where one
    // of its experts fits the device's largest allocation.  The residents do not move here (no adaptive rounds):
    // the profile's pairs within the layers' slots, or without a profile each layer's first experts.
    const auto slot = [&](int64_t l) { return (uint64_t) InklingProducts::slot_bytes(gu_bytes(l), down_bytes(l)); };
    const uint64_t max_alloc = max_alloc_bytes();
    const auto local = [&](int64_t l) { return slot(l) <= max_alloc; };
    uint64_t per_layer_b = 0;
    for (int64_t l = 0; l < s.n_layer; ++l)
        if (local(l)) per_layer_b += slot(l);
    const uint64_t window_b = tier_window_bytes(InklingProducts::kWindowRows, p_->NK, p_->N, batch);
    const TierRoom fits = plan_tier_room(counted_free(free, free, vram_free_mib), total, window_b, reserve_mib,
                                         expert_slots, per_layer_b, s.n_layer, p_->NE);
    auto fill = plan_tier_fill(profile.empty() ? nullptr : &prior, s.n_layer, p_->NE, fits, slot, local, false).fill;
    uint64_t hot_bytes = 0;
    try {
        if (!fill.empty()) {
            std::vector<int64_t> sizes;
            sizes.reserve(fill.size());
            for (const auto& pair : fill) sizes.push_back((int64_t) slot(pair.first));
            p_->products->open_tier(sizes, s.n_layer, p_->NE);
            // (taken in pieces after a refused allocation, the device may give the first slots only)
            fill.resize(std::min(fill.size(), (size_t) p_->products->tier_slots()));
        }
        for (const auto& [layer, expert] : fill) {
            const auto& l = p_->layers[(size_t) layer];
            const size_t gu = gu_bytes(layer), down = down_bytes(layer);
            p_->products->reserve_hot(layer, expert, l.gu.data + (size_t) expert * gu, gu,
                                      l.down.data + (size_t) expert * down, down);
            hot_bytes += gu + down;
        }
    } catch (const std::exception& e) { err = e.what(); return false; }
    const uint64_t hint = std::min(dram, batch * (uint64_t) p_->NK);
    p_->products->set_cold_hint(hint);
    std::fprintf(stderr, "xestrata: Inkling expert tier: %zu experts, %llu bytes hot of %llu bytes; "
                         "VRAM pool %llu, DRAM pool %llu, prefetch up to %llu bytes per tile\n", fill.size(),
                 (unsigned long long) hot_bytes, (unsigned long long) all, (unsigned long long) fits.room,
                 (unsigned long long) dram, (unsigned long long) hint);
    std::fprintf(stderr, "xestrata: Inkling expert tier plan: %llu bytes free, %llu kept free, %llu beside it for a "
                         "window; %lld experts a layer at most\n", (unsigned long long) free,
                 (unsigned long long) fits.reserve, (unsigned long long) window_b, (long long) fits.q);
    return true;
}

bool PlainInkling::save_profile(const std::string& path, std::string& err) const {
    auto order = rank_learned_profile(p_->s.n_layer, p_->NE, std::vector<uint8_t>(p_->heat.size(), 0), p_->heat, {});
    order.erase(std::remove_if(order.begin(), order.end(), [&](const auto& pair) {
        return p_->heat[(size_t) (pair.first * p_->NE + pair.second)] == 0;
    }), order.end());
    if (p_->products)
        std::fprintf(stderr, "xestrata: Inkling expert lookups: %llu/%llu resident\n",
                     (unsigned long long) p_->products->gpu_hits(), (unsigned long long) p_->products->gpu_lookups());
    return write_expert_profile(path, p_->s.n_layer, p_->NE, order, err);
}

bool PlainInkling::load_dspark(const std::string& path, std::string& err) {
    const auto& s = p_->s;
    if (s.n_layer != 24 || s.loop_start != 6 || s.loop_end != 18 || s.loop_count != 3 ||
        s.n_embd != 2304 || s.vocab_unpadded != 200058 || s.vocab_padded != 201024) {
        err = "DSpark: incompatible target geometry";
        return false;
    }
    auto draft = std::make_unique<PlainDspark>();
    if (!draft->load(path, *p_->m, err)) return false;
    p_->draft_features.resize(PlainDspark::taps.size() * (size_t) s.n_embd);
    p_->dspark = std::move(draft);
    return true;
}

bool PlainInkling::propose_dspark(int32_t anchor, int64_t pos, int count, std::vector<int32_t>& tokens,
                                 std::string& err) const {
    const auto embedding = [&](int32_t id, float* row) {
        if (!p_->embedding(id, row)) return false;
        rms_norm(row, p_->tok_norm.data(), row, p_->N, p_->eps);
        return true;
    };
    const auto head = [&](const float* in, float* out) { return p_->project_head(in, out, p_->s.vocab_unpadded); };
    return p_->dspark && p_->dspark->propose(anchor, pos, count, embedding, head, tokens, err);
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
