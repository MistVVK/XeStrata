// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/plain_cpu.cpp - see include/strata/core/plain_cpu.hpp.  The block follows plain_model.cpp's qwen35 path
// (llama.cpp's src/models/qwen35.cpp) step for step:
//
//   GDN:        [qkv | z] = in(h); alpha, beta = FP32 products of h; conv over [history | qkv], SiLU; q, k
//               l2-normalized per head; gate = softplus(alpha + dt) * ssm_a, beta = sigmoid(beta); the delta rule per
//               value head (value head j reads q/k head j % h_k), its output times 1/sqrt(S); y = rms_norm(o) *
//               ssm_norm * silu(z); out = ssm_out(y)
//   attention:  [q | gate] per head, k, v = in(h); q, k rms-normed per head and rotated in their first rope_dims dims
//               (NeoX halves, every IMRoPE section at the text position); softmax(q k / sqrt(d)) v over the cells up
//               to the row's own, times sigmoid(gate); out = attn_output(.)
//   FFN:        down(silu(gate(h)) * up(h))
//
// plamo3 (PLaMo 3 Translate, llama.cpp's src/models/plamo3.cpp) runs through the same code: every block an attention
// one, [q | k | v] and [gate | up] each one matrix in the file, no output gate, the attention's and the FFN's outputs
// rms-normed before their adds; the sliding-window layers see the last `swa_window` positions only and rotate with their
// own base, the others with the file's YaRN (rope_scaled_angle, the GPU kernels' formula).
//
// Compiled with -mavx2 -mfma -mf16c (CMakeLists.txt): PlainModel::load checks cpu_avx2_ok() before it makes a
// PlainCpu, and nothing here runs before (no initializer at namespace scope).
#include "strata/core/plain_cpu.hpp"

#include "strata/core/plain_model.hpp"
#include "strata/kernels/cpu/cpu_ops.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/platform/env.hpp"
#include "strata/platform/memory.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

namespace strata::core {
namespace k = strata::kernels;
namespace kc = strata::kernels::cpu;

namespace {

// the int8 products' scale block: the K-quants' sub-block.  64 was 10% faster on the 2B's prompt and moved its
// answers further from llama.cpp's (a KL of 5e-2 where 32 gave 1.2e-2; the 0.8B's story parted at token 13 instead of
// agreeing over all 64), 128 changed the 2B's verdict (i7-14700, 2026-11-08)
constexpr int kBlock = 32;

float hsum(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}

/// sum a[i] * b[i] over n (a multiple of 8), four lanes of eight
float dot(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    float r = hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; ++i) r = std::fmaf(a[i], b[i], r);
    return r;
}

inline float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// The decode windows (4 tokens at most) read the GGUF's own rows where the RAM holds them beside the int8 copy: a token
// then reads the file's 4 to 6 bits a weight instead of the int8 copy's 8.5, and the decode is the RAM's bandwidth
// (the 9B: 260 ms a token from the int8 copy on an i7-14700).  Wider windows take the int8 products.
constexpr int kNativeRows = 4;

/// A matrix of the forward: its int8 panels, and where its rows are in the GGUF (one part a tensor of a fused product)
struct Mat {
    kc::I8Matrix q;
    struct Part {
        const uint8_t* src = nullptr;
        int type = 0;
        size_t rb = 0;
        int64_t rows = 0;
    };
    std::vector<Part> parts;
    bool native = false;   ///< every part has a vec_dot in ggml-cpu
    uint64_t file_bytes() const {
        uint64_t b = 0;
        for (const Part& p : parts) b += (uint64_t) p.rb * (uint64_t) p.rows;
        return b;
    }
};

struct Layer {
    bool rec = false;
    bool swa = false;               ///< a sliding-window layer (plamo3)
    std::vector<float> attn_norm, post_norm;
    std::vector<float> attn_post, ffn_post;   ///< plamo3: the norms after the attention and the FFN (empty: none)
    Mat in, out, gu, down;
    // GDN
    std::vector<float> ab;          ///< [2 HV][N]: alpha's rows, then beta's
    std::vector<float> conv, dt, a, ssm_norm;
    int64_t gdn = -1;
    // attention
    std::vector<float> q_norm, k_norm;
    int64_t attn = -1, nkv = 0;
};

}  // namespace

struct PlainCpu::Impl {
    ModelSpec s;
    const GgufModel* m = nullptr;
    std::unique_ptr<kc::ExpertPool> pool;
    int host_cpu = -1;
    k::cpu::I8GemmIsa isa = k::cpu::I8GemmIsa::scalar;
    int levels = 0;   ///< the int8 weights' levels (kc::i8_gemm_weight_levels, or eight bits where the model needs them)
    int64_t N = 0, NH = 0, HD = 0, NROT = 0, HK = 0, HV = 0, S = 0, ZV = 0, C = 0, FF = 0;
    float eps = 1e-6f, theta_scale = 1.0f;
    bool gate = true;          ///< the attention's output gate: [q | gate] per head (qwen35; not plamo3)
    int64_t window = 0;        ///< the sliding-window layers' window (plamo3)
    // the rotations: the full-attention layers' (rope_full, scaled where the file says) and the sliding-window ones'
    k::RopeScaling rope_full, rope_swa;
    float theta_swa = 1.0f;
    int64_t max_cells = 0, chunk = 2048;
    std::vector<Layer> layers;
    std::vector<float> out_norm;
    Mat head;
    bool native = false;   ///< the decode windows read the GGUF's rows (Mat::parts)
    std::vector<std::vector<uint8_t>> act;   ///< a native product's quantized activations, a part's for each row
    // the embedding rows, read from the mapped file
    const uint8_t* emb = nullptr;
    int emb_type = 0;
    size_t emb_row = 0;
    // state
    std::vector<std::vector<float>> gdn_state;   ///< [HV][S / 32 column groups][S rows (k)][32 cols (v)] a GDN layer
    std::vector<std::vector<float>> gdn_conv;    ///< [C][3] a GDN layer
    int n_snaps = 0;                             ///< set_snap_rows: the window rows whose states are kept
    std::vector<std::vector<float>> snap_state, snap_conv;   ///< [row * gdn layers + layer]: the state after the row
    std::vector<std::vector<float>> kc_, vc_;    ///< [cells][nkv][HD] an attention layer
    std::vector<float> part_m, part_l, part_acc;   ///< a decode window's attention, a span's partials
    // scratch, kept between runs
    std::vector<float> x, hn, big, ab, mix, y, q, logit_rows;
    kc::I8Rows rows;
    // XESTRATA_CPU_PROFILE=1: each run's time by part (the products, the GDN, the attention, the rest), on stderr
    bool profile = false;
    double ms[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::chrono::steady_clock::time_point mark_at;
    void mark(int part) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        ms[part] += std::chrono::duration<double, std::milli>(now - mark_at).count();
        mark_at = now;
    }

    bool vec(const std::string& name, uint64_t n, std::vector<float>& out, std::string& err) const;
    bool mat(const std::vector<std::string>& names, Mat& out, int64_t n_in, std::string& err) const;
    bool dense(const std::string& name, int64_t n_in, int64_t n_out, float* out, std::string& err) const;
    void linear(const Mat& w, const float* x, size_t ldx, int n, float* out, size_t ldo);
    void block_gdn(Layer& L, int T);
    void block_attn(Layer& L, int T, int64_t pos);
    void ffn(Layer& L, int T);
    void run_chunk(const int32_t* tokens, int T, int64_t pos);
    void blocks(int T, int64_t pos);
};

bool PlainCpu::Impl::vec(const std::string& name, uint64_t n, std::vector<float>& out, std::string& err) const {
    size_t at = 0;
    const TensorInfo* t = m->find(name, &at);
    if (t == nullptr || t->elements() != n || !m->in_bounds(*t, at)) {
        err = "plain cpu: " + name + " is missing or not " + std::to_string(n) + " values";
        return false;
    }
    out.resize(n);
    if (!kc::native_row_to_float((int) t->type, m->shard(at).tensor_data(*t), out.data(), (int64_t) n)) {
        err = "plain cpu: " + name + "'s type has no host dequantizer";
        return false;
    }
    return true;
}

bool PlainCpu::Impl::dense(const std::string& name, int64_t n_in, int64_t n_out, float* out, std::string& err) const {
    size_t at = 0;
    const TensorInfo* t = m->find(name, &at);
    if (t == nullptr || t->shape.size() < 2 || (int64_t) t->shape[0] != n_in ||
        (int64_t) (t->elements() / t->shape[0]) != n_out || !m->in_bounds(*t, at)) {
        err = "plain cpu: " + name + " is missing or of another shape";
        return false;
    }
    const size_t rb = kc::native_row_bytes((int) t->type, n_in);
    const uint8_t* src = m->shard(at).tensor_data(*t);
    for (int64_t o = 0; o < n_out; ++o)
        if (rb == 0 || !kc::native_row_to_float((int) t->type, src + (size_t) o * rb, out + (size_t) o * n_in, n_in)) {
            err = "plain cpu: " + name + "'s type has no host dequantizer";
            return false;
        }
    return true;
}

// the matrices `names`, their rows one after another (a fused product: [qkv | z], [q | k | v], [gate | up])
bool PlainCpu::Impl::mat(const std::vector<std::string>& names, Mat& out, int64_t n_in, std::string& err) const {
    std::vector<Mat::Part>& parts = out.parts;
    parts.clear();
    int64_t n_out = 0;
    for (const std::string& name : names) {
        size_t at = 0;
        const TensorInfo* t = m->find(name, &at);
        if (t == nullptr || t->shape.size() < 2 || (int64_t) t->shape[0] != n_in || !m->in_bounds(*t, at)) {
            err = "plain cpu: " + name + " is missing or not " + std::to_string(n_in) + " wide";
            return false;
        }
        const size_t rb = kc::native_row_bytes((int) t->type, n_in);
        if (rb == 0) {
            err = "plain cpu: " + name + "'s type has no host dequantizer";
            return false;
        }
        const int64_t r = (int64_t) (t->elements() / t->shape[0]);
        parts.push_back({m->shard(at).tensor_data(*t), (int) t->type, rb, r});
        n_out += r;
    }
    out.native = true;
    for (const Mat::Part& p : parts) out.native = out.native && kc::native_vec_act_bytes(p.type, n_in) > 0;
    const bool ok = kc::i8_pack_rows(*pool, (int) n_in, (int) n_out, kBlock, levels,
                                     [&](int o, float* dst) {
        for (const Mat::Part& p : parts) {
            if (o < p.rows) return kc::native_row_to_float(p.type, p.src + (size_t) o * p.rb, dst, n_in);
            o -= (int) p.rows;
        }
        return false;
    }, out.q);
    if (!ok) err = "plain cpu: packing " + names[0] + " failed";
    return ok;
}

void PlainCpu::Impl::linear(const Mat& w, const float* xin, size_t ldx, int n, float* out, size_t ldo) {
    mark(3);
    if (!native || !w.native || n > kNativeRows) {
        kc::i8_quantize_rows(*pool, isa, xin, ldx, n, w.q.n_in, w.q.block, rows);
        kc::i8_linear(*pool, isa, w.q, rows, out, ldo);
        mark(0);
        return;
    }
    // the GGUF's rows: each part's activations once a row, then pieces of 64 rows, every token row against each
    const int64_t n_in = w.q.n_in;
    int64_t off = 0;
    for (const Mat::Part& p : w.parts) {
        const size_t ab = kc::native_vec_act_bytes(p.type, n_in);
        act.resize((size_t) n);
        for (int t = 0; t < n; ++t) {
            act[(size_t) t].resize(ab);
            kc::native_vec_quant(p.type, xin + (size_t) t * ldx, act[(size_t) t].data(), n_in);
        }
        constexpr int64_t kRows = 64;
        pool->run((int) ((p.rows + kRows - 1) / kRows), [&](int i) {
            const int64_t r0 = i * kRows, r1 = std::min(p.rows, r0 + kRows);
            float tmp[kRows];
            for (int t = 0; t < n; ++t) {
                kc::native_vec_rows(p.type, p.src + (size_t) r0 * p.rb, p.rb, n_in, 0, r1 - r0, act[(size_t) t].data(),
                                    tmp);
                std::memcpy(out + (size_t) t * ldo + off + r0, tmp, (size_t) (r1 - r0) * 4);
            }
        });
        off += p.rows;
    }
    mark(0);
}

PlainCpu::PlainCpu() = default;
PlainCpu::~PlainCpu() = default;

bool PlainCpu::load(const GgufModel& m, const ModelSpec& s, const PlainOptions& opt, std::string& err, int64_t l0,
                    int64_t l1) {
    if (l1 < 0) l1 = s.n_layer;
    const bool ends = l0 == 0 && l1 == s.n_layer;   // the embedding and the head too (else the GPU's)
    if (l0 < 0 || l1 > s.n_layer || l0 >= l1) {
        err = "plain cpu: no blocks to run";
        return false;
    }
    p_ = std::make_unique<Impl>();
    Impl& P = *p_;
    P.s = s;
    P.m = &m;
    const bool plamo = s.arch == ModelArch::Plamo3;
    if (s.arch != ModelArch::Qwen35 && !plamo) {
        err = "plain cpu: the CPU runs the dense models (qwen35, plamo3) only, not " + s.arch_name;
        return false;
    }
    if (!PlainModel::rope_scalings(s, opt, P.rope_full, P.rope_swa, err)) return false;
    P.gate = !plamo;
    P.window = plamo ? s.swa_window : 0;
    P.N = s.n_embd;
    P.NH = s.n_head;
    P.HD = s.key_length;
    P.NROT = s.rope_dims;
    P.HK = s.ssm_n_group;
    P.HV = s.ssm_dt_rank;
    P.S = s.ssm_d_state;
    P.ZV = s.ssm_d_inner;
    P.C = 2 * P.HK * P.S + P.ZV;
    P.FF = s.n_ff;
    P.eps = (float) s.rms_eps;
    if ((!plamo && (P.S != 128 || s.ssm_d_conv != 4 || P.HV % P.HK != 0 || P.ZV != P.HV * P.S)) || P.HD % 8 != 0 ||
        P.NROT % 2 != 0 || P.NROT > P.HD || P.N % kBlock != 0) {
        err = "plain cpu: a shape the CPU path is not written for";
        return false;
    }
    P.theta_scale = std::pow((float) P.rope_full.freq_base, -2.0f / (float) P.NROT);
    P.theta_swa = std::pow((float) P.rope_swa.freq_base, -2.0f / (float) P.NROT);
    P.max_cells = opt.max_cells;
    if (opt.prompt_chunk > 0) P.chunk = opt.prompt_chunk;
    const int threads = opt.cpu_threads > 0 ? opt.cpu_threads : 0;
    P.pool = std::make_unique<kc::ExpertPool>(threads, true, true, 0, 0, true);
    const std::vector<int> cores = kc::physical_cores(false);
    if (P.pool->workers() > 0 && !cores.empty()) P.host_cpu = cores[0];
    P.isa = kc::i8_gemm_isa();
    // AVX2's seven-bit weights moved plamo3 past its checks (the user's decision, 2026-11-09: not for PLaMo 3 Translate):
    // against llama.cpp's FP32 reference, 0.8B at 512 tokens, the 99th percentile's KL 0.135 where llama.cpp's own
    // default is 0.0975 and the eight-bit form 0.113, the same top token 88.2% (91.2%, 89.8%); i7-14700, 2026-11-09
    P.levels = plamo ? kc::kI8Levels : kc::i8_gemm_weight_levels(P.isa);
    const char* pr = strata::env_get("CPU_PROFILE");
    P.profile = pr != nullptr && pr[0] != 0 && pr[0] != '0';
    // the embedding: rows read from the file as they are needed
    if (ends) {
        size_t at = 0;
        const TensorInfo* t = m.find("token_embd.weight", &at);
        if (t == nullptr || t->shape.size() != 2 || (int64_t) t->shape[0] != P.N || !m.in_bounds(*t, at)) {
            err = "plain cpu: token_embd.weight is missing or of another width";
            return false;
        }
        P.emb = m.shard(at).tensor_data(*t);
        P.emb_type = (int) t->type;
        P.emb_row = kc::native_row_bytes(P.emb_type, P.N);
        n_vocab_ = (int64_t) t->shape[1];
        if (P.emb_row == 0) {
            err = "plain cpu: token_embd.weight's type has no host dequantizer";
            return false;
        }
    }
    const char* head = m.find("output.weight") != nullptr ? "output.weight" : "token_embd.weight";
    if (ends && (!P.vec("output_norm.weight", (uint64_t) P.N, P.out_norm, err) || !P.mat({head}, P.head, P.N, err)))
        return false;
    if (ends && P.head.q.n_out != n_vocab_) {
        err = std::string("plain cpu: ") + head + "'s shape";
        return false;
    }
    // blocks [l0, l1): every block, or (a GPU's tail) the ones its VRAM does not hold
    P.layers.resize((size_t) (l1 - l0));
    int64_t n_gdn = 0, n_attn = 0;
    for (int64_t l = l0; l < l1; ++l) {
        Layer& L = P.layers[(size_t) (l - l0)];
        const std::string p = "blk." + std::to_string(l) + ".";
        L.rec = s.recurrent[(size_t) l];
        L.swa = s.layer_swa(l);
        if (!P.vec(p + "attn_norm.weight", (uint64_t) P.N, L.attn_norm, err) ||
            !P.vec(p + (plamo ? "ffn_norm.weight" : "post_attention_norm.weight"), (uint64_t) P.N, L.post_norm, err) ||
            !P.mat(plamo ? std::vector<std::string>{p + "ffn_up.weight"}
                         : std::vector<std::string>{p + "ffn_gate.weight", p + "ffn_up.weight"}, L.gu, P.N, err) ||
            !P.mat({p + "ffn_down.weight"}, L.down, P.FF, err))
            return false;
        // plamo3's norms after the attention and the FFN (their names carry no ".weight")
        if (plamo && (!P.vec(p + "post_attention_norm", (uint64_t) P.N, L.attn_post, err) ||
                      !P.vec(p + "post_ffw_norm", (uint64_t) P.N, L.ffn_post, err)))
            return false;
        if (L.gu.q.n_out != 2 * P.FF || L.down.q.n_out != P.N) {
            err = "plain cpu: block " + std::to_string(l) + "'s FFN has another shape";
            return false;
        }
        if (L.rec) {
            L.gdn = n_gdn++;
            L.ab.resize((size_t) (2 * P.HV * P.N));
            if (!P.mat({p + "attn_qkv.weight", p + "attn_gate.weight"}, L.in, P.N, err) ||
                !P.mat({p + "ssm_out.weight"}, L.out, P.ZV, err) ||
                !P.dense(p + "ssm_alpha.weight", P.N, P.HV, L.ab.data(), err) ||
                !P.dense(p + "ssm_beta.weight", P.N, P.HV, L.ab.data() + (size_t) (P.HV * P.N), err) ||
                !P.vec(p + "ssm_conv1d.weight", (uint64_t) (P.C * 4), L.conv, err) ||
                !P.vec(p + "ssm_dt.bias", (uint64_t) P.HV, L.dt, err) ||
                !P.vec(p + "ssm_a", (uint64_t) P.HV, L.a, err) ||
                !P.vec(p + "ssm_norm.weight", (uint64_t) P.S, L.ssm_norm, err))
                return false;
            if (L.in.q.n_out != P.C + P.ZV || L.out.q.n_out != P.N) {
                err = "plain cpu: block " + std::to_string(l) + "'s GDN projections have other shapes";
                return false;
            }
        } else {
            L.attn = n_attn++;
            L.nkv = s.n_head_kv[(size_t) l];
            if (L.nkv <= 0 || P.NH % L.nkv != 0 ||
                !P.mat(plamo ? std::vector<std::string>{p + "attn_qkv.weight"}
                             : std::vector<std::string>{p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"},
                       L.in, P.N, err) ||
                !P.mat({p + "attn_output.weight"}, L.out, P.NH * P.HD, err) ||
                !P.vec(p + "attn_q_norm.weight", (uint64_t) P.HD, L.q_norm, err) ||
                !P.vec(p + "attn_k_norm.weight", (uint64_t) P.HD, L.k_norm, err)) {
                if (err.empty()) err = "plain cpu: block " + std::to_string(l) + "'s attention heads";
                return false;
            }
            if (L.in.q.n_out != ((P.gate ? 2 : 1) * P.NH + 2 * L.nkv) * P.HD || L.out.q.n_out != P.N) {
                err = "plain cpu: block " + std::to_string(l) + "'s attention has another shape";
                return false;
            }
        }
    }
    // the decode from the GGUF's rows where the RAM holds them beside the int8 copy: the free RAM and the file cache
    // (the packing just read the file through it) against the rows' bytes and a reserve for everything else
    {
        uint64_t file = ends ? P.head.file_bytes() : 0;
        bool all = !ends || P.head.native;
        for (const Layer& L : P.layers)
            for (const Mat* w : {&L.in, &L.out, &L.gu, &L.down}) {
                file += w->file_bytes();
                all = all && w->native;
            }
        uint64_t free = 0, cached = 0;
        const uint64_t reserve = std::max<uint64_t>(4ull << 30, strata::core::detail::ram_reserve_bytes());
        P.native = all && strata::platform::free_and_cached(free, cached) && free + cached >= file + reserve;
    }
    P.gdn_state.assign((size_t) n_gdn, std::vector<float>((size_t) (P.HV * P.S * P.S), 0.0f));
    P.gdn_conv.assign((size_t) n_gdn, std::vector<float>((size_t) (P.C * 3), 0.0f));
    P.kc_.resize((size_t) n_attn);
    P.vc_.resize((size_t) n_attn);
    for (const Layer& L : P.layers)
        if (!L.rec) {
            P.kc_[(size_t) L.attn].assign((size_t) (P.max_cells * L.nkv * P.HD), 0.0f);
            P.vc_[(size_t) L.attn].assign((size_t) (P.max_cells * L.nkv * P.HD), 0.0f);
        }
    return true;
}

std::string PlainCpu::describe() const {
    size_t bytes = p_->head.q.bytes();
    for (const Layer& L : p_->layers) bytes += L.in.q.bytes() + L.out.q.bytes() + L.gu.q.bytes() + L.down.q.bytes();
    return std::to_string(p_->pool->size()) + " threads, int8 products on " + kc::i8_gemm_isa_name(p_->isa) +
           (p_->head.q.levels == kc::kI8Levels7 ? " with 7-bit weights" : "") + " (" + std::to_string(bytes >> 20) +
           " MiB of packed weights); the decode " +
           (p_->native ? "from the model file's rows (the RAM holds both)" : "from the int8 copy");
}

std::vector<int> PlainCpu::worker_cpus() const {
    return p_ && p_->pool ? p_->pool->worker_cpus() : std::vector<int>();
}

bool PlainCpu::recurrent() const { return !p_->gdn_state.empty(); }

void PlainCpu::reset() {
    for (auto& v : p_->gdn_state) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : p_->gdn_conv) std::fill(v.begin(), v.end(), 0.0f);
}

void PlainCpu::Impl::block_gdn(Layer& L, int T) {
    const int64_t W = C + ZV;
    big.resize((size_t) T * W);
    ab.resize((size_t) T * 2 * HV);
    linear(L.in, hn.data(), (size_t) N, T, big.data(), (size_t) W);
    // alpha and beta in FP32, a row of tokens a piece
    pool->run(T, [&](int t) {
        const float* h = hn.data() + (size_t) t * N;
        for (int64_t j = 0; j < 2 * HV; ++j)
            ab[(size_t) t * 2 * HV + j] = dot(L.ab.data() + (size_t) j * N, h, (int) N);
    });
    mark(4);
    float* hist = gdn_conv[(size_t) L.gdn].data();
    // the convolution over [history | qkv] and SiLU: a piece is a range of channels, walked token by token with the
    // channels inner (a token's row is contiguous), the three inputs before each token kept beside it
    constexpr int64_t kChan = 256;
    pool->run((int) ((C + kChan - 1) / kChan), [&](int i) {
        const int64_t c0 = i * kChan, nc = std::min(C, c0 + kChan) - c0;
        float v0[kChan], v1[kChan], v2[kChan], w0[kChan], w1[kChan], w2[kChan], w3[kChan];
        for (int64_t c = 0; c < nc; ++c) {
            const float* w = L.conv.data() + (c0 + c) * 4;
            w0[c] = w[0]; w1[c] = w[1]; w2[c] = w[2]; w3[c] = w[3];
            v0[c] = hist[(c0 + c) * 3]; v1[c] = hist[(c0 + c) * 3 + 1]; v2[c] = hist[(c0 + c) * 3 + 2];
        }
        for (int t = 0; t < T; ++t) {
            float* cell = big.data() + (size_t) t * W + c0;
            for (int64_t c = 0; c < nc; ++c) {
                const float xv = cell[c];
                float sum = 0.0f;
                sum += v0[c] * w0[c];
                sum += v1[c] * w1[c];
                sum += v2[c] * w2[c];
                sum += xv * w3[c];
                cell[c] = sum / (1.0f + std::exp(-sum));
                v0[c] = v1[c]; v1[c] = v2[c]; v2[c] = xv;
            }
        }
        for (int64_t c = 0; c < nc; ++c) {
            hist[(c0 + c) * 3] = v0[c]; hist[(c0 + c) * 3 + 1] = v1[c]; hist[(c0 + c) * 3 + 2] = v2[c];
        }
    });
    mark(5);
    // q and k l2-normalized per head (plain_model's two steps: RMSNorm with eps / S, then 1 / sqrt(S)); the gates
    const float inv_s = 1.0f / std::sqrt((float) S);
    pool->run(T, [&](int t) {
        float* r = big.data() + (size_t) t * W;
        for (int64_t hh = 0; hh < 2 * HK; ++hh) {
            float* v = r + hh * S;
            const float ss = dot(v, v, (int) S);
            const float sc = 1.0f / std::sqrt(ss / (float) S + eps / (float) S);
            for (int64_t i = 0; i < S; ++i) v[i] = std::fmaf(sc * v[i], inv_s, 0.0f);
        }
        float* g = ab.data() + (size_t) t * 2 * HV;
        for (int64_t j = 0; j < HV; ++j) {
            const float v = g[j] + L.dt[(size_t) j];
            g[j] = (v > 20.0f ? v : std::log1p(std::exp(v))) * L.a[(size_t) j];
            g[HV + j] = sigm(g[HV + j]);
        }
    });
    mark(6);
    // the delta rule: each value head's columns are independent, so a piece is a head and a group of columns
    y.resize((size_t) T * ZV);
    float* st = gdn_state[(size_t) L.gdn].data();
    constexpr int64_t kCols = 32;
    const int64_t groups = S / kCols;
    pool->run((int) (HV * groups), [&](int i) {
        const int64_t h = i / groups, c0 = (i % groups) * kCols;
        const int64_t qh = h % HK;
        // a piece's columns are their own block of the state, [row][column] contiguous: 16 KiB a core walks twice
        // a token (laid out as [head][row][all columns], its rows 512 bytes apart fell on a few cache sets)
        float* s = st + ((size_t) h * groups + (size_t) (c0 / kCols)) * (size_t) S * kCols;
        constexpr int64_t V = kCols / 8;
        for (int t = 0; t < T; ++t) {
            const float* r = big.data() + (size_t) t * W;
            const float* q = r + qh * S;
            const float* kk = r + (HK + qh) * S;
            const float* v = r + 2 * HK * S + h * S + c0;
            const __m256 g = _mm256_set1_ps(std::exp(ab[(size_t) t * 2 * HV + h]));
            const __m256 b = _mm256_set1_ps(ab[(size_t) t * 2 * HV + HV + h]);
            __m256 kv[V];
            for (int64_t j = 0; j < V; ++j) kv[j] = _mm256_setzero_ps();
            for (int64_t row = 0; row < S; ++row) {
                const float* sr = s + row * kCols;
                const __m256 kr = _mm256_set1_ps(kk[row]);
                for (int64_t j = 0; j < V; ++j) kv[j] = _mm256_fmadd_ps(_mm256_loadu_ps(sr + 8 * j), kr, kv[j]);
            }
            __m256 delta[V], o[V];
            for (int64_t j = 0; j < V; ++j) {
                delta[j] = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + 8 * j), _mm256_mul_ps(g, kv[j])), b);
                o[j] = _mm256_setzero_ps();
            }
            for (int64_t row = 0; row < S; ++row) {
                float* sr = s + row * kCols;
                const __m256 kr = _mm256_set1_ps(kk[row]), qr = _mm256_set1_ps(q[row]);
                for (int64_t j = 0; j < V; ++j) {
                    const __m256 nv = _mm256_fmadd_ps(g, _mm256_loadu_ps(sr + 8 * j), _mm256_mul_ps(kr, delta[j]));
                    _mm256_storeu_ps(sr + 8 * j, nv);
                    o[j] = _mm256_fmadd_ps(nv, qr, o[j]);
                }
            }
            float* out = y.data() + (size_t) t * ZV + h * S + c0;
            const __m256 is = _mm256_set1_ps(inv_s);
            for (int64_t j = 0; j < V; ++j) _mm256_storeu_ps(out + 8 * j, _mm256_mul_ps(o[j], is));
        }
    });
    mark(7);
    // y = rms_norm(o) * ssm_norm * silu(z), a head of a token at a time
    pool->run(T, [&](int t) {
        for (int64_t h = 0; h < HV; ++h) {
            float* o = y.data() + (size_t) t * ZV + h * S;
            const float* z = big.data() + (size_t) t * W + C + h * S;
            const float sc = 1.0f / std::sqrt(dot(o, o, (int) S) / (float) S + eps);
            for (int64_t i = 0; i < S; ++i) {
                const float zv = z[i];
                o[i] = (sc * o[i]) * L.ssm_norm[(size_t) i] * (zv * sigm(zv));
            }
        }
    });
    mark(1);
    mix.resize((size_t) T * N);
    linear(L.out, y.data(), (size_t) ZV, T, mix.data(), (size_t) N);
}

void PlainCpu::Impl::block_attn(Layer& L, int T, int64_t pos) {
    const int64_t NKV = L.nkv, QH = (gate ? 2 : 1) * HD, QW = NH * QH, W = QW + 2 * NKV * HD;
    big.resize((size_t) T * W);
    linear(L.in, hn.data(), (size_t) N, T, big.data(), (size_t) W);
    float* kcache = kc_[(size_t) L.attn].data();
    float* vcache = vc_[(size_t) L.attn].data();
    q.resize((size_t) T * NH * HD);
    const int64_t half = NROT / 2;
    // the layer's rotation: a sliding-window layer's own base, else the file's (scaled where it says, as the GPU's
    // kernels: rope_scaled_angle on the trained angle)
    const k::RopeScaling& rs = L.swa ? rope_swa : rope_full;
    const float ts = L.swa ? theta_swa : theta_scale;
    const bool scaled = rs.type != k::RopeScalingType::None;
    const k::RopeKernelArgs ka = rs.kernel_args((int) NROT);
    // q and k: rms-normed per head and rotated (NeoX halves of the first NROT dims); k and v into the cache
    auto norm_rope = [&](const float* in, const float* gamma, float* out, int64_t p) {
        const float sc = 1.0f / std::sqrt(dot(in, in, (int) HD) / (float) HD + eps);
        for (int64_t i = NROT; i < HD; ++i) out[i] = sc * in[i] * gamma[i];
        for (int64_t i = 0; i < half; ++i) {
            const float a = sc * in[i] * gamma[i], b = sc * in[i + half] * gamma[i + half];
            const float th = (float) p * std::pow(ts, (float) i);
            float c, s;
            if (scaled) {
                k::rope_scaled_angle(th, ka.freq_scale, ka.corr_low, ka.corr_high, ka.ext_factor, ka.attn_factor,
                                     (int) i, c, s);
            } else {
                c = std::cos(th);
                s = std::sin(th);
            }
            out[i] = a * c - b * s;
            out[i + half] = a * s + b * c;
        }
    };
    pool->run(T, [&](int t) {
        const float* r = big.data() + (size_t) t * W;
        const int64_t p = pos + t;
        for (int64_t h = 0; h < NH; ++h)
            norm_rope(r + h * QH, L.q_norm.data(), q.data() + ((size_t) t * NH + h) * HD, p);
        for (int64_t g = 0; g < NKV; ++g) {
            norm_rope(r + QW + g * HD, L.k_norm.data(), kcache + ((size_t) p * NKV + g) * HD, p);
            std::memcpy(vcache + ((size_t) p * NKV + g) * HD, r + QW + NKV * HD + g * HD, (size_t) HD * 4);
        }
    });
    // the attention: a piece is a KV head's query heads (its cells read once for them: a full layer's 12K cells are
    // 50 MB on PLaMo 3 Translate 2B, past the L3, and its 6 heads each read them again, 18 -> 12.7 ms a token on the
    // i7-14700) over a block of queries and, in a decode window, a span of kSpan cells; the heads halved into groups
    // while the pieces number fewer than kPieces (the 0.8B's 16 heads on one KV head: a piece each span left the
    // threads idle, 2.4 -> 4.2 ms at 3K cells).  The heads and the spans by the shapes and the cells' positions, so the
    // pieces and the answer do not depend on the threads.  The keys in tiles, an online softmax per query and head,
    // the spans' partials merged in order.
    constexpr int kQ = 16, kK = 64, kPieces = 16;
    constexpr int64_t kSpan = 512;
    const int qblocks = (T + kQ - 1) / kQ;
    const float scale = 1.0f / std::sqrt((float) HD);
    const int64_t win = L.swa ? window : 0;
    const bool spans = T <= kQ;
    const int64_t lo_all = win > 0 ? std::max<int64_t>(0, pos - win + 1) : 0, hi_all = pos + T - 1;
    const int64_t s0 = spans ? lo_all / kSpan : 0, ns = spans ? hi_all / kSpan - s0 + 1 : 1;
    const int64_t G = NH / NKV;
    int64_t HG = G;
    while (HG > 1 && NKV * ((G + HG - 1) / HG) * qblocks * ns < kPieces) HG = (HG + 1) / 2;
    const int64_t hgroups = (G + HG - 1) / HG;
    y.resize((size_t) T * NH * HD);
    if (spans) {
        part_m.resize((size_t) (ns * T * NH));
        part_l.resize((size_t) (ns * T * NH));
        part_acc.resize((size_t) (ns * T * NH * HD));
    }
    // the output row of query t, head h, from its sums: the attention, times sigmoid(gate) where the model gates it
    auto finish = [&](int t, int64_t h, const float* a, float l) {
        const float inv = 1.0f / l;
        float* out = y.data() + ((size_t) t * NH + h) * HD;
        if (gate) {
            const float* gv = big.data() + (size_t) t * W + h * QH + HD;
            for (int64_t d = 0; d < HD; ++d) out[d] = a[d] * inv * sigm(gv[d]);
        } else {
            for (int64_t d = 0; d < HD; ++d) out[d] = a[d] * inv;
        }
    };
    pool->run((int) (NKV * hgroups * qblocks * ns), [&](int i) {
        const int64_t g = i / (hgroups * qblocks * ns), hg = i / (qblocks * ns) % hgroups, sp = i % ns;
        const int qb = (int) (i / ns % qblocks);
        const int64_t h0 = g * G + hg * HG, nh = std::min(HG, G - hg * HG);   // the piece's heads h0 .. h0 + nh - 1
        const int t0 = qb * kQ, t1 = std::min(T, t0 + kQ), nq = t1 - t0, R = nq * (int) nh;
        const int64_t last = pos + t1 - 1;   // the block's last query sees the cells up to its own
        // a sliding-window layer: query p sees the cells from p - window + 1 on, the block's first query the earliest
        int64_t ca = win > 0 ? std::max<int64_t>(0, pos + t0 - win + 1) : 0, cb = last + 1;
        if (spans) {
            ca = std::max(ca, (s0 + sp) * kSpan);
            cb = std::min(cb, (s0 + sp + 1) * kSpan);
        }
        std::vector<float> mrow((size_t) R, -std::numeric_limits<float>::infinity()), lrow((size_t) R, 0.0f);
        std::vector<float> acc((size_t) R * HD, 0.0f), sc((size_t) R * kK);
        for (int64_t k0 = ca; k0 < cb; k0 += kK) {
            const int64_t k1 = std::min(cb, k0 + kK);
            for (int64_t c = k0; c < k1; ++c) {
                const float* kr = kcache + ((size_t) c * NKV + g) * HD;
                for (int j = 0; j < nq; ++j) {
                    const int64_t pj = pos + t0 + j, lo = win > 0 ? pj - win + 1 : 0;
                    const bool seen = c <= pj && c >= lo;
                    for (int64_t hh = 0; hh < nh; ++hh) {
                        const float* qv = q.data() + ((size_t) (t0 + j) * NH + h0 + hh) * HD;
                        sc[(size_t) (j * nh + hh) * kK + (size_t) (c - k0)] =
                            seen ? dot(qv, kr, (int) HD) * scale : -std::numeric_limits<float>::infinity();
                    }
                }
            }
            for (int r = 0; r < R; ++r) {
                const float* sr = sc.data() + (size_t) r * kK;
                float mx = mrow[(size_t) r];
                for (int64_t c = k0; c < k1; ++c) mx = std::max(mx, sr[c - k0]);
                if (mx == -std::numeric_limits<float>::infinity()) continue;   // no cell of this tile for it
                const float corr = std::exp(mrow[(size_t) r] - mx);
                float* a = acc.data() + (size_t) r * HD;
                if (corr != 1.0f)
                    for (int64_t d = 0; d < HD; ++d) a[d] *= corr;
                float l = lrow[(size_t) r] * corr;
                for (int64_t c = k0; c < k1; ++c) {
                    const float pw = std::exp(sr[c - k0] - mx);
                    if (pw == 0.0f) continue;
                    l += pw;
                    const float* vv = vcache + ((size_t) c * NKV + g) * HD;
                    for (int64_t d = 0; d < HD; ++d) a[d] = std::fmaf(pw, vv[d], a[d]);
                }
                mrow[(size_t) r] = mx;
                lrow[(size_t) r] = l;
            }
        }
        for (int j = 0; j < nq; ++j)
            for (int64_t hh = 0; hh < nh; ++hh) {
                const int r = j * (int) nh + (int) hh, t = t0 + j;
                const int64_t h = h0 + hh;
                const float* a = acc.data() + (size_t) r * HD;
                if (!spans) {
                    finish(t, h, a, lrow[(size_t) r]);
                    continue;
                }
                const size_t at = (size_t) ((sp * T + t) * NH + h);
                part_m[at] = mrow[(size_t) r];
                part_l[at] = lrow[(size_t) r];
                std::memcpy(part_acc.data() + at * HD, a, (size_t) HD * 4);
            }
    });
    if (spans)   // the spans' partials, in order
        pool->run((int) (T * NH), [&](int i) {
            const int t = i / (int) NH;
            const int64_t h = i % NH;
            float M = -std::numeric_limits<float>::infinity();
            for (int64_t sp = 0; sp < ns; ++sp) M = std::max(M, part_m[(size_t) ((sp * T + t) * NH + h)]);
            float a[512] = {}, l = 0.0f;
            for (int64_t sp = 0; sp < ns; ++sp) {
                const size_t at = (size_t) ((sp * T + t) * NH + h);
                if (part_m[at] == -std::numeric_limits<float>::infinity()) continue;
                const float w = std::exp(part_m[at] - M);
                l = std::fmaf(part_l[at], w, l);
                const float* pa = part_acc.data() + at * HD;
                for (int64_t d = 0; d < HD; ++d) a[d] = std::fmaf(pa[d], w, a[d]);
            }
            finish(t, h, a, l);
        });
    mark(2);
    mix.resize((size_t) T * N);
    linear(L.out, y.data(), (size_t) (NH * HD), T, mix.data(), (size_t) N);
}

void PlainCpu::Impl::ffn(Layer& L, int T) {
    big.resize((size_t) T * 2 * FF);
    linear(L.gu, hn.data(), (size_t) N, T, big.data(), (size_t) (2 * FF));
    y.resize((size_t) T * FF);
    pool->run(T, [&](int t) {
        const float* gu = big.data() + (size_t) t * 2 * FF;
        float* o = y.data() + (size_t) t * FF;
        for (int64_t f = 0; f < FF; ++f) {
            const float gv = gu[f];
            o[f] = gv / (1.0f + std::exp(-gv)) * gu[FF + f];
        }
    });
    mix.resize((size_t) T * N);
    linear(L.down, y.data(), (size_t) FF, T, mix.data(), (size_t) N);
}

void PlainCpu::Impl::run_chunk(const int32_t* tokens, int T, int64_t pos) {
    x.resize((size_t) T * N);
    pool->run(T, [&](int t) {
        kc::native_row_to_float(emb_type, emb + (size_t) tokens[t] * emb_row, x.data() + (size_t) t * N, N);
    });
    blocks(T, pos);
}

// the blocks over the T rows in x
void PlainCpu::Impl::blocks(int T, int64_t pos) {
    hn.resize((size_t) T * N);
    auto norm = [&](const std::vector<float>& w) {
        const int per = std::max(1, (T + pool->size() - 1) / pool->size());
        pool->run((T + per - 1) / per, [&](int i) {
            const int t0 = i * per, t1 = std::min(T, t0 + per);
            kc::rmsnorm_rows(x.data() + (size_t) t0 * N, (size_t) N, hn.data() + (size_t) t0 * N, (size_t) N, t1 - t0,
                             (int) N, w.data(), eps);
        });
    };
    // x += mix, or (plamo3) x += rms_norm(mix) * `post`
    auto add = [&](const std::vector<float>& post) {
        pool->run(T, [&](int t) {
            float* xr = x.data() + (size_t) t * N;
            const float* mr = mix.data() + (size_t) t * N;
            if (!post.empty()) {
                float* hr = hn.data() + (size_t) t * N;
                kc::rmsnorm_rows(mr, (size_t) N, hr, (size_t) N, 1, (int) N, post.data(), eps);
                mr = hr;
            }
            for (int64_t i = 0; i < N; ++i) xr[i] += mr[i];
        });
    };
    for (Layer& L : layers) {
        norm(L.attn_norm);
        if (L.rec) block_gdn(L, T);
        else block_attn(L, T, pos);
        add(L.attn_post);
        norm(L.post_norm);
        ffn(L, T);
        add(L.ffn_post);
    }
}

bool PlainCpu::run_blocks(float* x, int64_t n, int64_t pos, std::string& err) {
    Impl& P = *p_;
    const kc::ScopedThreadPin pin(P.host_cpu);
    if (n <= 0 || pos < 0 || pos + n > P.max_cells) {
        err = "plain cpu: the rows are past the context (" + std::to_string(P.max_cells) + " cells)";
        return false;
    }
    // a decode window's rows run one at a time while their states are kept for a draft's rollback (the state after
    // each row but the last, as the GPU window kernel's copies); a prompt chunk runs in its chunks
    const bool snaps = P.n_snaps > 0 && n > 1 && n <= P.n_snaps + 1 && !P.gdn_state.empty();
    const size_t ng = P.gdn_state.size();
    for (int64_t done = 0; done < n;) {
        const int T = snaps ? 1 : (int) std::min<int64_t>(P.chunk, n - done);
        P.x.assign(x + (size_t) (done * P.N), x + (size_t) ((done + T) * P.N));
        P.blocks(T, pos + done);
        std::memcpy(x + (size_t) (done * P.N), P.x.data(), (size_t) (T * P.N) * 4);
        if (snaps && done + 1 < n && done < P.n_snaps) {
            for (size_t g = 0; g < ng; ++g) {
                std::memcpy(P.snap_state[(size_t) done * ng + g].data(), P.gdn_state[g].data(),
                            P.gdn_state[g].size() * 4);
                std::memcpy(P.snap_conv[(size_t) done * ng + g].data(), P.gdn_conv[g].data(),
                            P.gdn_conv[g].size() * 4);
            }
        }
        done += T;
    }
    return true;
}

void PlainCpu::set_snap_rows(int n) {
    Impl& P = *p_;
    if (n < 0) n = 0;
    const size_t ng = P.gdn_state.size();
    P.snap_state.resize((size_t) n * ng);
    P.snap_conv.resize((size_t) n * ng);
    for (int t = 0; t < n; ++t)
        for (size_t g = 0; g < ng; ++g) {
            P.snap_state[(size_t) t * ng + g].resize(P.gdn_state[g].size());
            P.snap_conv[(size_t) t * ng + g].resize(P.gdn_conv[g].size());
        }
    P.n_snaps = n;
}

bool PlainCpu::rollback(int n, std::string& err) {
    Impl& P = *p_;
    const size_t ng = P.gdn_state.size();
    if (n < 0 || n >= P.n_snaps || ng == 0) {
        err = "plain cpu: no recurrent state copy of row " + std::to_string(n);
        return false;
    }
    for (size_t g = 0; g < ng; ++g) {
        std::memcpy(P.gdn_state[g].data(), P.snap_state[(size_t) n * ng + g].data(), P.gdn_state[g].size() * 4);
        std::memcpy(P.gdn_conv[g].data(), P.snap_conv[(size_t) n * ng + g].data(), P.gdn_conv[g].size() * 4);
    }
    return true;
}

bool PlainCpu::run(const int32_t* tokens, int64_t n, int64_t pos, int rows, float* logits, std::string& err) {
    Impl& P = *p_;
    const kc::ScopedThreadPin pin(P.host_cpu);
    if (n <= 0 || rows < 0 || rows > n) {
        err = "plain cpu: no tokens to run";
        return false;
    }
    if (pos + n > P.max_cells) {
        err = "plain cpu: the context is full (" + std::to_string(P.max_cells) + " cells)";
        return false;
    }
    for (int64_t i = 0; i < n; ++i)
        if (tokens[i] < 0 || tokens[i] >= n_vocab_) {
            err = "plain cpu: token " + std::to_string(tokens[i]) + " is outside the vocabulary";
            return false;
        }
    if (P.profile) {
        std::fill(std::begin(P.ms), std::end(P.ms), 0.0);
        P.mark_at = std::chrono::steady_clock::now();
    }
    for (int64_t done = 0; done < n;) {
        const int T = (int) std::min<int64_t>(P.chunk, n - done);
        P.run_chunk(tokens + done, T, pos + done);
        done += T;
        // the rows asked for that fall in this chunk: the final norm and the head
        const int64_t first = n - rows;   // the first row asked for, of the whole run
        const int64_t lo = std::max(first, done - T);
        if (rows > 0 && lo < done) {
            const int cnt = (int) (done - lo);
            const int64_t from = lo - (done - T);
            P.logit_rows.resize((size_t) cnt * P.N);
            kc::rmsnorm_rows(P.x.data() + (size_t) from * P.N, (size_t) P.N, P.logit_rows.data(), (size_t) P.N, cnt,
                             (int) P.N, P.out_norm.data(), P.eps);
            P.linear(P.head, P.logit_rows.data(), (size_t) P.N, cnt, logits + (size_t) (lo - first) * n_vocab_,
                     (size_t) n_vocab_);
        }
    }
    if (P.profile) {
        P.mark(3);
        std::fprintf(stderr, "xestrata plain cpu: %lld tokens: products %.1f ms, GDN %.1f (alpha/beta %.1f, conv %.1f, "
                     "norms and gates %.1f, recurrence %.1f, output norm %.1f), attention %.1f, the rest %.1f\n",
                     (long long) n, P.ms[0], P.ms[1] + P.ms[4] + P.ms[5] + P.ms[6] + P.ms[7], P.ms[4], P.ms[5], P.ms[6],
                     P.ms[7], P.ms[1], P.ms[2], P.ms[3]);
    }
    return true;
}

void PlainCpu::save(int64_t tokens, bool kv, std::vector<std::vector<uint8_t>>& parts) const {
    const Impl& P = *p_;
    auto put = [&](const float* p, size_t n) {
        parts.emplace_back(reinterpret_cast<const uint8_t*>(p), reinterpret_cast<const uint8_t*>(p + n));
    };
    for (size_t i = 0; i < P.gdn_state.size(); ++i) {
        put(P.gdn_state[i].data(), P.gdn_state[i].size());
        put(P.gdn_conv[i].data(), P.gdn_conv[i].size());
    }
    if (!kv) return;
    for (const Layer& L : P.layers)
        if (!L.rec) {
            const size_t n = (size_t) (tokens * L.nkv * P.HD);
            put(P.kc_[(size_t) L.attn].data(), n);
            put(P.vc_[(size_t) L.attn].data(), n);
        }
}

size_t PlainCpu::state_bytes(int64_t tokens, bool kv) const {
    const Impl& P = *p_;
    size_t b = 0;
    for (size_t i = 0; i < P.gdn_state.size(); ++i) b += (P.gdn_state[i].size() + P.gdn_conv[i].size()) * 4;
    if (kv)
        for (const Layer& L : P.layers)
            if (!L.rec) b += (size_t) (tokens * L.nkv * P.HD) * 8;
    return b;
}

bool PlainCpu::restore(const std::vector<std::vector<uint8_t>>& parts, bool kv, std::string& err) {
    Impl& P = *p_;
    size_t i = 0;
    auto get = [&](float* p, size_t cap, bool exact) {
        if (i >= parts.size() || parts[i].size() % 4 != 0 || parts[i].size() / 4 > cap ||
            (exact && parts[i].size() / 4 != cap))
            return false;
        std::memcpy(p, parts[i].data(), parts[i].size());
        ++i;
        return true;
    };
    for (size_t g = 0; g < P.gdn_state.size(); ++g)
        if (!get(P.gdn_state[g].data(), P.gdn_state[g].size(), true) ||
            !get(P.gdn_conv[g].data(), P.gdn_conv[g].size(), true)) {
            err = "plain cpu: a saved state of another model";
            return false;
        }
    if (kv)
        for (const Layer& L : P.layers)
            if (!L.rec && (!get(P.kc_[(size_t) L.attn].data(), P.kc_[(size_t) L.attn].size(), false) ||
                           !get(P.vc_[(size_t) L.attn].data(), P.vc_[(size_t) L.attn].size(), false))) {
                err = "plain cpu: a saved state of another model or a longer context";
                return false;
            }
    return true;
}

}  // namespace strata::core
