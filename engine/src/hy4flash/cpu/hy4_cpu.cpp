// hy4_cpu.cpp - CPU C++ oracle forward for Hy4-Flash (hy_v4_flash) on .strata v1.0 (own code, zero llama.cpp).
// Mirrors tools/squant/hy_v4_flash.py / scale_qat_eval.py op by op (non-absorbed MLA, 42 layers, 4-stream iHC, 4 Engram modules,
// sparse IndexPool mask when T > index_topk, LatentMoE with square S_K experts). Whole prompt per forward, no KV cache: an oracle for the GPU path.
// Build (WSL): see phaseA-tests/hy4/s/build_cpu.sh
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "hy4_json.h"
#include "hy4_manifest.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/ngram/hy4_engram_reader.hpp"
#include "strata/quant/sk_codecs.hpp"

using namespace strata;
using hy4::JVal;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

using hy4::Cfg;
using hy4::fmt_from;
using hy4::TInfo;
using Engram = hy4::EngramCfg;
using hy4::F_S2; using hy4::F_S3; using hy4::F_S4; using hy4::F_S5; using hy4::F_S6; using hy4::F_S8; using hy4::F_Q8ROW; using hy4::F_F32;

struct Model {
    Cfg c;
    Engram e;
    artifact::StrataFileReader rd;
    std::unordered_map<std::string, TInfo> T;
    std::vector<uint8_t> trunk;
    uint64_t sec4_off = 0;
    int fd = -1;
    std::vector<int> rank_of;  // [41][512]
    std::unique_ptr<ngram::Hy4EngramReader> eng;
    std::vector<uint8_t> seed;
    std::string model_path;
    bool float_act = true;
};

static void die(const std::string& m) {
    fprintf(stderr, "FATAL: %s\n", m.c_str());
    exit(1);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// loading
static void load_model(Model& m, const std::string& path, int max_layers_unused) {
    (void)max_layers_unused;
    std::string err;
    m.model_path = path;
    if (!m.rd.open(path, err)) die(err);
    double t0 = now_s();
    JVal man = hy4::json_parse(m.rd.manifest_json());
    const JVal& tc = man.at("config").at("text_config");
    Cfg& c = m.c;
    c.n_layers = (int)tc.at("num_hidden_layers").i();
    c.hidden = (int)tc.at("hidden_size").i();
    c.hc = (int)tc.at("hc_mult").i();
    c.vocab = (int)tc.at("vocab_size").i();
    c.rms_eps = (float)tc.at("rms_norm_eps").f();
    c.hc_eps = (float)tc.at("hc_eps").f();
    c.hc_mag = (float)tc.at("hc_magnitude").f();
    c.q_lora = (int)tc.at("q_lora_rank").i();
    c.kv_lora = (int)tc.at("kv_lora_rank").i();
    c.qk_nope = (int)tc.at("qk_nope_head_dim").i();
    c.qk_rope = (int)tc.at("qk_rope_head_dim").i();
    c.qk_head = (int)tc.at("qk_head_dim").i();
    c.v_head = (int)tc.at("v_head_dim").i();
    c.window = (int)tc.at("sliding_window").i();
    c.kpool = (int)tc.at("compressed_indexer_compress_ratio").i();
    c.index_topk = (int)tc.at("index_topk").i();
    c.n_idx_head = (int)tc.at("compressed_indexer_num_attention_heads").i();
    c.idx_dim = (int)tc.at("compressed_indexer_head_dim").i();
    c.swiglu_limit = (float)tc.at("swiglu_limit").f();
    c.routed_scale = (float)tc.at("routed_scaling_factor").f();
    c.n_experts = (int)tc.at("n_routed_experts").i();
    c.top_k = (int)tc.at("num_experts_per_tok").i();
    c.heads_sparse = (int)tc.at("num_attention_heads").i();
    c.heads_swa = (int)tc.at("swa_num_attention_heads").i();
    c.theta_sparse = tc.at("rope_parameters").at("deepseek_sparse_attention").at("rope_theta").f();
    c.theta_swa = tc.at("rope_parameters").at("sliding_attention").at("rope_theta").f();
    const JVal& lt = tc.at("layer_types");
    c.is_swa.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; ++i) c.is_swa[i] = lt[i].str() == "sliding_attention";

    const JVal& em = man.at("engram_meta");
    Engram& e = m.e;
    e.n_gram = (int)em.at("ple_ngram_size").i();
    e.n_head = (int)em.at("ple_heads_per_ngram").i();
    e.n_cols = (e.n_gram - 1) * e.n_head;
    e.pad_id = (int)em.at("engram_pad_id").i();
    for (auto& v : em.at("ple_layers").a) e.layers.push_back((int)v.i());
    e.n_mod = (int)e.layers.size();
    for (auto& v : em.at("engram_token_map").a) e.token_map.push_back(v.i());
    for (auto& v : em.at("ple_layer_multipliers").a) e.mult.push_back((uint64_t)v.i());
    for (auto& v : em.at("ple_head_vocab_sizes").a) e.primes.push_back((uint64_t)v.i());
    for (auto& v : em.at("ple_head_offsets").a) e.offs.push_back(v.i());

    for (auto& t : man.at("trunk_tensors").a) {
        TInfo ti;
        ti.name = t.at("name").str();
        ti.fmt = fmt_from(t.at("format").str());
        for (auto& d : t.at("shape").a) ti.shape.push_back(d.i());
        ti.off = (uint64_t)t.at("offset").i();
        ti.bytes = (uint64_t)t.at("bytes").i();
        m.T[ti.name] = ti;
    }
    const auto& h = m.rd.header();
    m.sec4_off = h.sections[4].file_offset;
    // trunk (section 1) into RAM
    uint64_t s1 = h.sections[1].byte_length;
    m.trunk.resize(s1);
    if (!m.rd.read_at(h.sections[1].file_offset, m.trunk.data(), s1, err)) die(err);
    // expert directory
    const auto& dir = m.rd.expert_directory();
    m.rank_of.assign(41 * 512, -1);
    for (size_t i = 0; i < dir.size(); ++i) m.rank_of[(i / 512) * 512 + dir[i].expert_id] = (int)(i % 512);
    m.fd = open(path.c_str(), O_RDONLY);
    if (m.fd < 0) die("open model for pread");
    // engram seed
    std::vector<uint8_t> s2;
    if (!m.rd.read_section(2, s2, err)) die(err);
    m.seed = std::move(s2);
    fprintf(stderr, "[load] manifest+trunk %.1f s, trunk %.2f GB, %zu tensors, experts dir %zu\n", now_s() - t0, s1 / 1e9, m.T.size(), dir.size());
}

static void open_engram(Model& m, const std::string& shard) {
    std::string err;
    m.eng.reset(new ngram::Hy4EngramReader());
    if (!m.eng->open(shard, m.seed.data(), m.seed.size(), 200000, err)) die(err);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// primitives
static const TInfo& tinfo(Model& m, const std::string& n) {
    auto it = m.T.find(n);
    if (it == m.T.end()) die("missing tensor " + n);
    return it->second;
}
static const uint8_t* tptr(Model& m, const TInfo& t) { return m.trunk.data() + t.off; }

static void rms_rows(const float* x, const float* w, float* y, int T, int D, float eps) {
#pragma omp parallel for
    for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t)t * D;
        double ss = 0;
        for (int i = 0; i < D; ++i) ss += (double)xr[i] * xr[i];
        float inv = 1.0f / std::sqrt((float)(ss / D) + eps);
        float* yr = y + (size_t)t * D;
        for (int i = 0; i < D; ++i) yr[i] = xr[i] * inv * (w ? w[i] : 1.0f);
    }
}

static inline float siluf(float x) { return x / (1.0f + std::exp(-x)); }
static inline float sigf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// y[M][N] = x[M][K] * W[N][K]^T for a trunk tensor (or raw weight pointer + fmt)
static void linear_raw(Model& m, int fmt, const uint8_t* W, int N, int K, const float* x, float* y, int M) {
    using namespace quant::sk;
    if (fmt == F_F32) {
        const float* Wf = (const float*)W;
#pragma omp parallel for
        for (int n = 0; n < N; ++n) {
            const float* wr = Wf + (size_t)n * K;
            for (int t = 0; t < M; ++t) {
                const float* xr = x + (size_t)t * K;
                float s = 0;
                for (int k = 0; k < K; ++k) s += wr[k] * xr[k];
                y[(size_t)t * N + n] = s;
            }
        }
        return;
    }
    if (fmt == F_Q8ROW) {
        const float* sc = (const float*)W;
        const int8_t* q = (const int8_t*)(W + (size_t)N * 4);
#pragma omp parallel for
        for (int n = 0; n < N; ++n) {
            const int8_t* wr = q + (size_t)n * K;
            for (int t = 0; t < M; ++t) {
                const float* xr = x + (size_t)t * K;
                float s = 0;
                for (int k = 0; k < K; ++k) s += (float)wr[k] * xr[k];
                y[(size_t)t * N + n] = s * sc[n];
            }
        }
        return;
    }
    SkCode code = (SkCode)fmt;
    size_t row_bytes = (size_t)(K / 256) * superblock_bytes(code);
    if (m.float_act) {
#pragma omp parallel
        {
            std::vector<float> buf(K);
#pragma omp for
            for (int n = 0; n < N; ++n) {
                dequant_sk_row(W + (size_t)n * row_bytes, code, buf.data(), K);
                for (int t = 0; t < M; ++t) {
                    const float* xr = x + (size_t)t * K;
                    float s = 0;
                    for (int k = 0; k < K; ++k) s += buf[k] * xr[k];
                    y[(size_t)t * N + n] = s;
                }
            }
        }
    } else {
        std::vector<BlockQ8_1_G64> xq((size_t)M * (K / 64));
        for (int t = 0; t < M; ++t) quantize_q8_1_g64(x + (size_t)t * K, xq.data() + (size_t)t * (K / 64), K);
        const int RB = 32;
#pragma omp parallel for
        for (int r0 = 0; r0 < N; r0 += RB) {
            int nr = std::min(RB, N - r0);
            float tmp[16 * RB];
            for (int t0 = 0; t0 < M; t0 += 16) {
                int mm = std::min(16, M - t0);
                gemm_sk_q8_1(W + (size_t)r0 * row_bytes, code, xq.data() + (size_t)t0 * (K / 64), tmp, nr, K, mm);
                for (int t = 0; t < mm; ++t)
                    for (int r = 0; r < nr; ++r) y[(size_t)(t0 + t) * N + r0 + r] = tmp[t * nr + r];
            }
        }
    }
}

static void linear(Model& m, const std::string& name, const float* x, float* y, int M) {
    const TInfo& t = tinfo(m, name);
    linear_raw(m, t.fmt, tptr(m, t), (int)t.rows(), (int)t.cols(), x, y, M);
}

static void get_vec(Model& m, const std::string& name, std::vector<float>& out) {
    const TInfo& t = tinfo(m, name);
    if (t.fmt != F_F32) die("get_vec expects F32: " + name);
    size_t n = t.bytes / 4;
    out.resize(n);
    memcpy(out.data(), tptr(m, t), t.bytes);
}

// interleaved rope on D=64 slice: pairs (2i, 2i+1)
static void rope_il(float* x, int D, int pos, double theta) {
    for (int i = 0; i < D / 2; ++i) {
        double f = std::pow(theta, -(double)(2 * i) / D);
        double a = (double)pos * f;
        float c = (float)std::cos(a), s = (float)std::sin(a);
        float x0 = x[2 * i], x1 = x[2 * i + 1];
        x[2 * i] = x0 * c - x1 * s;
        x[2 * i + 1] = x0 * s + x1 * c;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// npy dump
static void save_npy(const std::string& path, const float* d, const std::vector<int64_t>& shape) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) sh += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
    sh += ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t tot = 10 + hdr.size() + 1;
    size_t pad = (64 - tot % 64) % 64;
    hdr += std::string(pad, ' ') + "\n";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    unsigned char ver[2] = {1, 0};
    fwrite(ver, 1, 2, f);
    uint16_t hl = (uint16_t)hdr.size();
    fwrite(&hl, 2, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    fwrite(d, 4, n, f);
    fclose(f);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// engram
static void engram_hash(Model& m, int mod, const std::vector<int32_t>& tok, std::vector<uint64_t>& ids, std::vector<float>& valid) {
    const Engram& e = m.e;
    int T = (int)tok.size();
    ids.assign((size_t)T * e.n_cols, 0);
    valid.assign((size_t)T * e.n_cols, 1.0f);
    const uint64_t* mult = e.mult.data() + (size_t)mod * e.n_gram;
    const uint64_t* primes = e.primes.data() + (size_t)mod * e.n_cols;
    const int64_t* offs = e.offs.data() + (size_t)mod * e.n_cols;
    const int64_t pad_cid = e.token_map[e.pad_id];
    auto cid = [&](int t) -> int64_t { return (t < 0 || t >= (int)e.token_map.size()) ? -1 : e.token_map[t]; };
    for (int i = 0; i < T; ++i) {
        int64_t c0 = cid(tok[i]);
        bool dead = c0 < 0;
        uint64_t t0 = (uint64_t)(dead ? pad_cid : c0);
        int64_t c1 = i - 1 < 0 ? -1 : cid(tok[i - 1]);
        int64_t c2 = i - 2 < 0 ? -1 : cid(tok[i - 2]);
        uint64_t t1 = (uint64_t)(c1 < 0 ? pad_cid : c1);
        uint64_t t2 = (uint64_t)(c2 < 0 ? pad_cid : c2);
        uint64_t r1 = (t0 * mult[0]) ^ (t1 * mult[1]);
        uint64_t r2 = r1 ^ (t2 * mult[2]);
        for (int h = 0; h < e.n_head; ++h) {
            ids[(size_t)i * e.n_cols + h] = r1 % primes[h] + (uint64_t)offs[h];
            ids[(size_t)i * e.n_cols + e.n_head + h] = r2 % primes[e.n_head + h] + (uint64_t)offs[e.n_head + h];
        }
        if (dead)
            for (int j = 0; j < e.n_cols; ++j) valid[(size_t)i * e.n_cols + j] = 0.0f;
    }
}

static void apply_engram(Model& m, std::vector<float>& x, int il, int mod, const std::vector<int32_t>& tok) {
    const Cfg& c = m.c;
    int T = (int)tok.size(), H = c.hidden;
    std::vector<uint64_t> ids;
    std::vector<float> valid;
    engram_hash(m, mod, tok, ids, valid);
    std::vector<float> rows((size_t)T * 16 * 96);
    std::string err;
    if (!m.eng->fetch_rows_f32(ids.data(), ids.size(), rows.data(), err)) die(err);
    const float EPS = 1.1920929e-7f;
    std::string p = "model.layers." + std::to_string(il) + ".engram_layer";
    std::vector<float> lnw(16 * 96);
    for (int ch = 0; ch < 16; ++ch) {
        std::vector<float> w;
        get_vec(m, p + ".lookup_per_head_norms." + std::to_string(ch) + ".weight", w);
        memcpy(&lnw[ch * 96], w.data(), 96 * 4);
    }
    std::vector<float> r((size_t)T * 1536);
    for (int t = 0; t < T; ++t)
        for (int ch = 0; ch < 16; ++ch) {
            const float* row = &rows[((size_t)t * 16 + ch) * 96];
            double ss = 0;
            for (int i = 0; i < 96; ++i) ss += (double)row[i] * row[i];
            float inv = 1.0f / std::sqrt((float)(ss / 96) + EPS);
            for (int i = 0; i < 96; ++i) r[(size_t)t * 1536 + ch * 96 + i] = row[i] * inv * lnw[ch * 96 + i] * valid[(size_t)t * 16 + ch];
        }
    // 4 key projections + value projection: each [3072,1536] S8_K
    std::vector<float> kv((size_t)T * 5 * H);
    for (int j = 0; j < 5; ++j) {
        std::string nm = j < 4 ? p + ".key_projs." + std::to_string(j) + ".weight" : p + ".value_proj.weight";
        std::vector<float> yj((size_t)T * H);
        linear(m, nm, r.data(), yj.data(), T);
        std::vector<float> b;
        get_vec(m, j < 4 ? p + ".key_projs." + std::to_string(j) + ".bias" : p + ".value_proj.bias", b);
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < H; ++d) kv[((size_t)t * 5 + j) * H + d] = yj[(size_t)t * H + d] + b[d];
    }
    std::vector<std::vector<float>> nk(4), nq(4), hs(4);
    for (int j = 0; j < 4; ++j) {
        get_vec(m, p + ".norm_key." + std::to_string(j) + ".weight", nk[j]);
        get_vec(m, p + ".norm_query." + std::to_string(j) + ".weight", nq[j]);
        get_vec(m, p + ".unified_per_hc_scales." + std::to_string(j), hs[j]);
    }
    std::vector<float> pre_gamma;
    get_vec(m, p + ".unified_pre_gamma", pre_gamma);
    for (int t = 0; t < T; ++t) {
        float anyv = 0;
        for (int j = 0; j < 16; ++j) anyv += valid[(size_t)t * 16 + j];
        float any = anyv > 0 ? 1.0f : 0.0f;
        // value normed
        const float* val = &kv[((size_t)t * 5 + 4) * H];
        double sv = 0;
        for (int d = 0; d < H; ++d) sv += (double)val[d] * val[d];
        float vinv = 1.0f / std::sqrt((float)(sv / H) + EPS);
        for (int j = 0; j < 4; ++j) {
            float* xs = &x[((size_t)t * 4 + j) * H];
            const float* key = &kv[((size_t)t * 5 + j) * H];
            double sx = 0, sk = 0;
            for (int d = 0; d < H; ++d) {
                sx += (double)xs[d] * xs[d];
                sk += (double)key[d] * key[d];
            }
            float xinv = 1.0f / std::sqrt((float)(sx / H) + EPS), kinv = 1.0f / std::sqrt((float)(sk / H) + EPS);
            double dot = 0;
            for (int d = 0; d < H; ++d) dot += (double)(xs[d] * xinv) * (key[d] * kinv) * (nk[j][d] * nq[j][d]);
            float dotf = (float)(dot / std::sqrt((double)H));
            float mag = std::sqrt(std::max(std::fabs(dotf), 1e-6f));
            float gate = sigf(dotf < 0 ? -mag : mag);
            for (int d = 0; d < H; ++d) xs[d] += (val[d] * vinv * pre_gamma[d]) * (gate + hs[j][d]) * any;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// attention sublayer (non-absorbed, mirrors the python reference)
static void indexpool_mask(Model& m, int il, const std::vector<float>& cur, const std::vector<float>& qr, int T, double theta, int index_topk,
                           std::vector<uint8_t>& allow) {
    const Cfg& c = m.c;
    allow.assign((size_t)T * T, 0);
    if (T <= index_topk) {
        for (int i = 0; i < T; ++i)
            for (int j = 0; j <= i; ++j) allow[(size_t)i * T + j] = 1;
        return;
    }
    std::string p = "model.layers." + std::to_string(il) + ".self_attn.compressed_indexer";
    int NH = c.n_idx_head, D = c.idx_dim, kp = c.kpool, H = c.hidden;
    std::vector<float> iq((size_t)T * NH * D);
    linear(m, p + ".query_projection.weight", qr.data(), iq.data(), T);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < NH; ++h) rope_il(&iq[((size_t)t * NH + h) * D + (D - c.qk_rope)], c.qk_rope, t, theta);
    std::vector<float> ikp((size_t)T * D), tkn;
    linear(m, p + ".token_key_projection.weight", cur.data(), ikp.data(), T);
    get_vec(m, p + ".token_key_norm.weight", tkn);
    std::vector<float> ik((size_t)T * D);
    rms_rows(ikp.data(), tkn.data(), ik.data(), T, D, c.rms_eps);
    std::vector<float> iv((size_t)T * D), ig((size_t)T * D);
    linear(m, p + ".key_value_projection.weight", ik.data(), iv.data(), T);
    linear(m, p + ".key_gate_projection.weight", ik.data(), ig.data(), T);
    std::vector<float> ape, knw;
    get_vec(m, p + ".intra_group_ape", ape);
    get_vec(m, p + ".key_norm.weight", knw);
    int n_pools = T / kp;
    std::vector<float> pooled((size_t)n_pools * D);
    for (int g = 0; g < n_pools; ++g) {
        for (int d = 0; d < D; ++d) {
            float l[8], mx = -1e30f;
            for (int j = 0; j < kp; ++j) {
                l[j] = ig[((size_t)g * kp + j) * D + d] + ape[(size_t)j * D + d];
                mx = std::max(mx, l[j]);
            }
            float s = 0;
            for (int j = 0; j < kp; ++j) {
                l[j] = std::exp(l[j] - mx);
                s += l[j];
            }
            float acc = 0;
            for (int j = 0; j < kp; ++j) acc += (l[j] / s) * iv[((size_t)g * kp + j) * D + d];
            pooled[(size_t)g * D + d] = acc;
        }
    }
    std::vector<float> pn((size_t)n_pools * D);
    rms_rows(pooled.data(), knw.data(), pn.data(), n_pools, D, c.rms_eps);
    for (int g = 0; g < n_pools; ++g) rope_il(&pn[(size_t)g * D + (D - c.qk_rope)], c.qk_rope, g * kp, theta);
    std::vector<float> hw((size_t)T * NH);
    linear(m, p + ".head_weight_projection.weight", cur.data(), hw.data(), T);
    float hscale = 1.0f / std::sqrt((float)(D * NH));
    int n_top = std::min(n_pools, index_topk / kp - 1);
    (void)H;
#pragma omp parallel for
    for (int t = 0; t < T; ++t) {
        int grp = t / kp;
        std::vector<std::pair<float, int>> sc;
        for (int g = 0; g < n_pools; ++g) {
            float s;
            if (g >= grp) {
                s = -INFINITY;
            } else {
                s = 0;
                for (int h = 0; h < NH; ++h) {
                    float d = 0;
                    const float* a = &iq[((size_t)t * NH + h) * D];
                    const float* b = &pn[(size_t)g * D];
                    for (int k = 0; k < D; ++k) d += a[k] * b[k];
                    s += std::max(d, 0.0f) * hw[(size_t)t * NH + h] * hscale;
                }
            }
            sc.push_back({s, g});
        }
        std::partial_sort(sc.begin(), sc.begin() + n_top, sc.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
        for (int k = 0; k < n_top; ++k) {
            int g = sc[k].second;
            for (int o = 0; o < kp; ++o) {
                int col = g * kp + o;
                if (col < t) allow[(size_t)t * T + col] = 1;
            }
        }
        for (int col = grp * kp; col <= t; ++col) allow[(size_t)t * T + col] = 1;
    }
}

static void attention_sublayer(Model& m, int il, const std::vector<float>& cur, int T, int index_topk, std::vector<float>& out, std::vector<std::string>* dbg_names,
                               std::vector<std::vector<float>>* dbg) {
    const Cfg& c = m.c;
    bool swa = c.is_swa[il];
    int NH = swa ? c.heads_swa : c.heads_sparse;
    double theta = swa ? c.theta_swa : c.theta_sparse;
    std::string p = "model.layers." + std::to_string(il) + ".self_attn";
    std::vector<float> w;
    std::vector<float> qa((size_t)T * c.q_lora), qr((size_t)T * c.q_lora);
    linear(m, p + ".q_a_proj.weight", cur.data(), qa.data(), T);
    get_vec(m, p + ".q_a_layernorm.weight", w);
    rms_rows(qa.data(), w.data(), qr.data(), T, c.q_lora, c.rms_eps);
    int QD = NH * c.qk_head;
    std::vector<float> q((size_t)T * QD);
    linear(m, p + ".q_b_proj.weight", qr.data(), q.data(), T);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < NH; ++h) rope_il(&q[(size_t)t * QD + (size_t)h * c.qk_head + c.qk_nope], c.qk_rope, t, theta);
    std::vector<float> kva((size_t)T * (c.kv_lora + c.qk_rope)), kvc((size_t)T * c.kv_lora);
    linear(m, p + ".kv_a_proj_with_mqa.weight", cur.data(), kva.data(), T);
    {
        std::vector<float> tmp((size_t)T * c.kv_lora);
        for (int t = 0; t < T; ++t) memcpy(&tmp[(size_t)t * c.kv_lora], &kva[(size_t)t * (c.kv_lora + c.qk_rope)], c.kv_lora * 4);
        get_vec(m, p + ".kv_a_layernorm.weight", w);
        rms_rows(tmp.data(), w.data(), kvc.data(), T, c.kv_lora, c.rms_eps);
    }
    std::vector<float> kpe((size_t)T * c.qk_rope);
    for (int t = 0; t < T; ++t) {
        memcpy(&kpe[(size_t)t * c.qk_rope], &kva[(size_t)t * (c.kv_lora + c.qk_rope) + c.kv_lora], c.qk_rope * 4);
        rope_il(&kpe[(size_t)t * c.qk_rope], c.qk_rope, t, theta);
    }
    int EX = c.qk_nope + c.v_head;
    std::vector<float> kvx((size_t)T * NH * EX);
    linear(m, p + ".kv_b_proj.weight", kvc.data(), kvx.data(), T);
    std::vector<uint8_t> allow;
    if (swa) {
        allow.assign((size_t)T * T, 0);
        for (int i = 0; i < T; ++i)
            for (int j = 0; j <= i; ++j)
                if (i - j < c.window) allow[(size_t)i * T + j] = 1;
    } else {
        indexpool_mask(m, il, cur, qr, T, theta, index_topk, allow);
    }
    std::vector<float> sinks;
    get_vec(m, p + ".learnable_sink_param", sinks);
    std::vector<float> attn((size_t)T * NH * c.v_head);
    float scale = 1.0f / std::sqrt((float)c.qk_head);
#pragma omp parallel for collapse(2)
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < NH; ++h) {
            const float* qh = &q[(size_t)t * QD + (size_t)h * c.qk_head];
            std::vector<float> s(t + 1, -INFINITY);
            float mx = sinks[h];
            for (int j = 0; j <= t; ++j) {
                if (!allow[(size_t)t * T + j]) continue;
                const float* kn = &kvx[((size_t)j * NH + h) * EX];
                float d = 0;
                for (int k = 0; k < c.qk_nope; ++k) d += qh[k] * kn[k];
                const float* kp = &kpe[(size_t)j * c.qk_rope];
                for (int k = 0; k < c.qk_rope; ++k) d += qh[c.qk_nope + k] * kp[k];
                s[j] = d * scale;
                mx = std::max(mx, s[j]);
            }
            double den = std::exp((double)sinks[h] - mx);
            for (int j = 0; j <= t; ++j) {
                if (s[j] == -INFINITY) continue;
                s[j] = std::exp(s[j] - mx);
                den += s[j];
            }
            float* o = &attn[((size_t)t * NH + h) * c.v_head];
            for (int k = 0; k < c.v_head; ++k) o[k] = 0;
            for (int j = 0; j <= t; ++j) {
                if (s[j] == -INFINITY) continue;
                float pj = (float)(s[j] / den);
                const float* v = &kvx[((size_t)j * NH + h) * EX + c.qk_nope];
                for (int k = 0; k < c.v_head; ++k) o[k] += pj * v[k];
            }
        }
    std::vector<float> gate((size_t)T * QD);
    linear(m, p + ".linear_gate.weight", cur.data(), gate.data(), T);
    for (size_t i = 0; i < attn.size(); ++i) attn[i] *= sigf(gate[i]);
    std::vector<float> o((size_t)T * c.hidden);
    linear(m, p + ".o_proj.weight", attn.data(), o.data(), T);
    get_vec(m, p + ".post_proj_norm.weight", w);
    out.resize((size_t)T * c.hidden);
    rms_rows(o.data(), w.data(), out.data(), T, c.hidden, c.rms_eps);
    if (dbg) {
        dbg_names->push_back("qr");
        dbg->push_back(qr);
        dbg_names->push_back("kv_cmpr");
        dbg->push_back(kvc);
        dbg_names->push_back("k_pe");
        dbg->push_back(kpe);
        dbg_names->push_back("attn_pre_oproj");
        dbg->push_back(attn);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// expert read
struct ExpertBuf {
    std::vector<uint8_t> body;
    int gu_code = 0, dn_code = 0;
    uint32_t dn_off = 0;
};
static void read_expert(Model& m, int moe_layer, int expert, ExpertBuf& b) {
    int rank = m.rank_of[moe_layer * 512 + expert];
    if (rank < 0) die("expert not in directory");
    const auto& ent = m.rd.expert_directory()[(size_t)moe_layer * 512 + rank];
    b.body.resize(ent.body_bytes);
    size_t got = 0;
    while (got < ent.body_bytes) {
        ssize_t r = pread(m.fd, b.body.data() + got, ent.body_bytes - got, (off_t)(m.sec4_off + ent.body_offset + got));
        if (r <= 0) die("pread expert failed");
        got += (size_t)r;
    }
    b.gu_code = ent.gu_code_id;
    b.dn_code = ent.dn_code_id;
    b.dn_off = ent.dn_offset;
}

static void mlp_sublayer(Model& m, int il, const std::vector<float>& cur, int T, std::vector<float>& out, std::vector<std::string>* dbg_names,
                         std::vector<std::vector<float>>* dbg) {
    const Cfg& c = m.c;
    int H = c.hidden;
    std::string p = "model.layers." + std::to_string(il) + ".mlp";
    std::vector<float> w;
    out.assign((size_t)T * H, 0.0f);
    if (il == 0) {
        int F = 13824;
        std::vector<float> g((size_t)T * F), u((size_t)T * F);
        linear(m, p + ".gate_proj.weight", cur.data(), g.data(), T);
        linear(m, p + ".up_proj.weight", cur.data(), u.data(), T);
        for (size_t i = 0; i < g.size(); ++i) g[i] = siluf(g[i]) * u[i];
        std::vector<float> d((size_t)T * H);
        linear(m, p + ".down_proj.weight", g.data(), d.data(), T);
        get_vec(m, p + ".post_fc2_norm.weight", w);
        rms_rows(d.data(), w.data(), out.data(), T, H, c.rms_eps);
        return;
    }
    int E = c.n_experts, K = c.top_k, L = 1536;
    std::vector<float> router_w, bias;
    std::vector<float> logits((size_t)T * E);
    linear(m, p + ".gate.weight", cur.data(), logits.data(), T);  // F32 weights, fp32 acts
    get_vec(m, p + ".gate.e_score_correction_bias", bias);
    std::vector<int> topi((size_t)T * K);
    std::vector<float> topw((size_t)T * K);
    for (int t = 0; t < T; ++t) {
        std::vector<float> pr(E), sel(E);
        for (int e = 0; e < E; ++e) {
            pr[e] = sigf(logits[(size_t)t * E + e]);
            sel[e] = pr[e] + bias[e];
        }
        std::vector<int> idx(E);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return sel[a] > sel[b] || (sel[a] == sel[b] && a < b); });
        float sum = 0;
        for (int k = 0; k < K; ++k) sum += pr[idx[k]];
        for (int k = 0; k < K; ++k) {
            topi[(size_t)t * K + k] = idx[k];
            topw[(size_t)t * K + k] = pr[idx[k]] / (sum + 1e-20f) * c.routed_scale;
        }
    }
    std::vector<float> z((size_t)T * L);
    linear(m, p + ".routed_expert_down_proj.weight", cur.data(), z.data(), T);
    std::vector<float> zout((size_t)T * L, 0.0f);
    // group tokens by expert
    std::map<int, std::vector<std::pair<int, float>>> by_e;
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) by_e[topi[(size_t)t * K + k]].push_back({t, topw[(size_t)t * K + k]});
    ExpertBuf eb;
    for (auto& kv : by_e) {
        int e = kv.first;
        auto& lst = kv.second;
        int n = (int)lst.size();
        read_expert(m, il - 1, e, eb);
        const uint16_t* sq16 = (const uint16_t*)eb.body.data();
        std::vector<float> sq(L);
        for (int i = 0; i < L; ++i) {
            uint32_t h = sq16[i];
            uint32_t sign = (h & 0x8000u) << 16, ex = (h >> 10) & 0x1F, mn = h & 0x3FF, f;
            if (ex == 0) {
                if (mn == 0) f = sign;
                else {
                    uint32_t e2 = 127 - 14, m2 = mn;
                    while (!(m2 & 0x400)) { m2 <<= 1; --e2; }
                    m2 &= 0x3FF;
                    f = sign | (e2 << 23) | (m2 << 13);
                }
            } else if (ex == 31) f = sign | 0x7F800000u | (mn << 13);
            else f = sign | ((ex + 112) << 23) | (mn << 13);
            memcpy(&sq[i], &f, 4);
        }
        std::vector<float> ze((size_t)n * L), gu((size_t)n * 2 * L), h((size_t)n * L), ye((size_t)n * L);
        for (int i = 0; i < n; ++i) memcpy(&ze[(size_t)i * L], &z[(size_t)lst[i].first * L], L * 4);
        linear_raw(m, eb.gu_code, eb.body.data() + 3072, 2 * L, L, ze.data(), gu.data(), n);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < L; ++d) {
                float g = gu[(size_t)i * 2 * L + d], u = gu[(size_t)i * 2 * L + L + d];
                float s = std::min(siluf(g), c.swiglu_limit);
                float uc = std::max(-c.swiglu_limit, std::min(u, c.swiglu_limit));
                h[(size_t)i * L + d] = (s * uc) / sq[d];
            }
        linear_raw(m, eb.dn_code, eb.body.data() + eb.dn_off, L, L, h.data(), ye.data(), n);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < L; ++d) zout[(size_t)lst[i].first * L + d] += ye[(size_t)i * L + d] * lst[i].second;
    }
    std::vector<float> moe((size_t)T * H);
    linear(m, p + ".routed_expert_up_proj.weight", zout.data(), moe.data(), T);
    std::vector<float> sg((size_t)T * L), su((size_t)T * L), sd((size_t)T * H);
    linear(m, p + ".shared_experts.gate_proj.weight", cur.data(), sg.data(), T);
    linear(m, p + ".shared_experts.up_proj.weight", cur.data(), su.data(), T);
    for (size_t i = 0; i < sg.size(); ++i) sg[i] = siluf(sg[i]) * su[i];
    linear(m, p + ".shared_experts.down_proj.weight", sg.data(), sd.data(), T);
    for (size_t i = 0; i < moe.size(); ++i) moe[i] += sd[i];
    get_vec(m, p + ".combine_output_norm.weight", w);
    rms_rows(moe.data(), w.data(), out.data(), T, H, c.rms_eps);
    if (dbg) {
        dbg_names->push_back("topk_idx");
        std::vector<float> ti(topi.begin(), topi.end());
        dbg->push_back(ti);
        dbg_names->push_back("topk_w");
        dbg->push_back(topw);
        dbg_names->push_back("z_folded");
        dbg->push_back(z);
        dbg_names->push_back("z_out");
        dbg->push_back(zout);
        dbg_names->push_back("shared_out");
        dbg->push_back(sd);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
static void hc_pre(Model& m, const std::vector<float>& x, int T, const std::string& prefix, std::vector<float>& cur, std::vector<float>& post) {
    const Cfg& c = m.c;
    int H = c.hidden, HC = c.hc;
    std::vector<float> fn, sc, base;
    get_vec(m, prefix + ".hc_pre.hc_fn", fn);
    get_vec(m, prefix + ".hc_pre.hc_scale", sc);
    get_vec(m, prefix + ".hc_pre.hc_base", base);
    std::vector<float> fl((size_t)T * HC * H);
    rms_rows(x.data(), nullptr, fl.data(), T, HC * H, c.rms_eps);
    std::vector<float> mix((size_t)T * 8);
    linear_raw(m, F_F32, (const uint8_t*)fn.data(), 8, HC * H, fl.data(), mix.data(), T);
    cur.assign((size_t)T * H, 0.0f);
    post.assign((size_t)T * HC, 0.0f);
    for (int t = 0; t < T; ++t) {
        float pre[4];
        for (int h = 0; h < HC; ++h) {
            pre[h] = sigf(mix[(size_t)t * 8 + h] * sc[0] + base[h]) + c.hc_eps;
            post[(size_t)t * HC + h] = c.hc_mag * sigf(mix[(size_t)t * 8 + HC + h] * sc[1] + base[HC + h]) + c.hc_eps;
        }
        for (int h = 0; h < HC; ++h) {
            const float* xs = &x[((size_t)t * HC + h) * H];
            float* cr = &cur[(size_t)t * H];
            for (int d = 0; d < H; ++d) cr[d] += xs[d] * pre[h];
        }
    }
}

static void hc_post(const Cfg& c, std::vector<float>& x, const std::vector<float>& sub, const std::vector<float>& post, int T) {
    int H = c.hidden, HC = c.hc;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < HC; ++h) {
            float* xs = &x[((size_t)t * HC + h) * H];
            const float* s = &sub[(size_t)t * H];
            float p = post[(size_t)t * HC + h];
            for (int d = 0; d < H; ++d) xs[d] += s[d] * p;
        }
}

int main(int argc, char** argv) {
    std::string model_path, engram_path, tok_path, out_dir = ".";
    int max_layers = 42, index_topk = -1, threads = 0;
    bool dump = false, all_logits = false, act_q8 = false, dbg_layers = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model_path = nxt();
        else if (a == "--engram") engram_path = nxt();
        else if (a == "--tokens") tok_path = nxt();
        else if (a == "--out") out_dir = nxt();
        else if (a == "--layers") max_layers = atoi(nxt().c_str());
        else if (a == "--index-topk") index_topk = atoi(nxt().c_str());
        else if (a == "--threads") threads = atoi(nxt().c_str());
        else if (a == "--dump") dump = true;
        else if (a == "--dbg-layers") dbg_layers = true;
        else if (a == "--all-logits") all_logits = true;
        else if (a == "--act") act_q8 = nxt() == "q8";
        else die("unknown arg " + a);
    }
    if (model_path.empty() || engram_path.empty() || tok_path.empty()) die("usage: hy4cpu --model f.strata --engram e.strata --tokens ids.txt --out dir [--dump] [--layers N] [--index-topk N] [--act float|q8] [--all-logits]");
    if (threads > 0) omp_set_num_threads(threads);
    Model m;
    m.float_act = !act_q8;
    load_model(m, model_path, max_layers);
    open_engram(m, engram_path);
    if (index_topk < 0) index_topk = m.c.index_topk;
    std::vector<int32_t> tok;
    {
        std::ifstream f(tok_path);
        long v;
        while (f >> v) tok.push_back((int32_t)v);
    }
    int T = (int)tok.size();
    const Cfg& c = m.c;
    int H = c.hidden, HC = c.hc;
    fprintf(stderr, "[run] T=%d layers=%d act=%s index_topk=%d threads=%d isa=%s\n", T, max_layers, m.float_act ? "float" : "q8", index_topk, omp_get_max_threads(), quant::sk::active_sk_isa());
    std::vector<float> x((size_t)T * HC * H);
    {
        const TInfo& emb = tinfo(m, "model.embed_tokens.weight");
        size_t rb = (size_t)(H / 256) * quant::sk::superblock_bytes(quant::sk::SkCode::S8_K);
        std::vector<float> row(H);
        for (int t = 0; t < T; ++t) {
            quant::sk::dequant_sk_row(tptr(m, emb) + (size_t)tok[t] * rb, quant::sk::SkCode::S8_K, row.data(), H);
            for (int h = 0; h < HC; ++h) memcpy(&x[((size_t)t * HC + h) * H], row.data(), H * 4);
        }
    }
    auto dumpx = [&](const std::string& nm, const std::vector<float>& v, std::vector<int64_t> shape) {
        if (dump) save_npy(out_dir + "/" + nm + ".npy", v.data(), shape);
    };
    double t_start = now_s();
    for (int il = 0; il < max_layers; ++il) {
        double tl = now_s();
        int mod = -1;
        for (int j = 0; j < m.e.n_mod; ++j)
            if (m.e.layers[j] == il) mod = j;
        if (mod >= 0) {
            apply_engram(m, x, il, mod, tok);
            dumpx("x_L" + std::to_string(il) + "_engram", x, {T, HC, H});
        }
        std::string L = "model.layers." + std::to_string(il);
        std::vector<float> cur, post, w, curn, sub;
        std::vector<std::string> dn;
        std::vector<std::vector<float>> dv;
        hc_pre(m, x, T, L + ".hc_attn_layer", cur, post);
        get_vec(m, L + ".input_layernorm.weight", w);
        curn.resize(cur.size());
        rms_rows(cur.data(), w.data(), curn.data(), T, H, c.rms_eps);
        if (dbg_layers) {
            dumpx("L" + std::to_string(il) + "_cur_attn", curn, {T, H});
        }
        attention_sublayer(m, il, curn, T, index_topk, sub, &dn, &dv);
        hc_post(c, x, sub, post, T);
        dumpx("x_L" + std::to_string(il) + "_attn", x, {T, HC, H});
        if (dbg_layers)
            for (size_t i = 0; i < dn.size(); ++i) dumpx("L" + std::to_string(il) + "_" + dn[i], dv[i], {T, (int64_t)(dv[i].size() / T)});
        dn.clear();
        dv.clear();
        hc_pre(m, x, T, L + ".hc_mlp_layer", cur, post);
        get_vec(m, L + ".post_attention_layernorm.weight", w);
        curn.resize(cur.size());
        rms_rows(cur.data(), w.data(), curn.data(), T, H, c.rms_eps);
        if (dbg_layers) dumpx("L" + std::to_string(il) + "_cur_mlp", curn, {T, H});
        mlp_sublayer(m, il, curn, T, sub, &dn, &dv);
        hc_post(c, x, sub, post, T);
        dumpx("x_L" + std::to_string(il) + "_mlp", x, {T, HC, H});
        if (dbg_layers)
            for (size_t i = 0; i < dn.size(); ++i) dumpx("L" + std::to_string(il) + "_" + dn[i], dv[i], {T, (int64_t)(dv[i].size() / T)});
        fprintf(stderr, "[layer %2d] %.1f s (cum %.1f s) engram ssd pages %llu\n", il, now_s() - tl, now_s() - t_start, (unsigned long long)m.eng->stats().ssd_page_reads);
    }
    if (max_layers < c.n_layers) {
        fprintf(stderr, "stopped early at %d layers\n", max_layers);
        return 0;
    }
    // head
    std::vector<float> fn, sc, base, nw;
    get_vec(m, "model.hc_head.hc_head_fn", fn);
    get_vec(m, "model.hc_head.hc_head_scale", sc);
    get_vec(m, "model.hc_head.hc_head_base", base);
    get_vec(m, "model.norm.weight", nw);
    std::vector<float> fl((size_t)T * HC * H), mix((size_t)T * HC), hh((size_t)T * H, 0.0f), hf((size_t)T * H);
    rms_rows(x.data(), nullptr, fl.data(), T, HC * H, c.rms_eps);
    linear_raw(m, F_F32, (const uint8_t*)fn.data(), HC, HC * H, fl.data(), mix.data(), T);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < HC; ++h) {
            float pre = sigf(mix[(size_t)t * HC + h] * sc[0] + base[h]) + c.hc_eps;
            const float* xs = &x[((size_t)t * HC + h) * H];
            for (int d = 0; d < H; ++d) hh[(size_t)t * H + d] += xs[d] * pre;
        }
    rms_rows(hh.data(), nw.data(), hf.data(), T, H, c.rms_eps);
    dumpx("h_final", hf, {T, H});
    int t0 = all_logits ? 0 : T - 1, nt = T - t0;
    std::vector<float> logits((size_t)nt * c.vocab);
    {
        const TInfo& lm = tinfo(m, "lm_head.weight");
        linear_raw(m, lm.fmt, tptr(m, lm), (int)lm.rows(), (int)lm.cols(), hf.data() + (size_t)t0 * H, logits.data(), nt);
    }
    dumpx("logits", logits, {nt, c.vocab});
    std::string top_out;
    for (int t = 0; t < nt; ++t) {
        std::vector<int> idx(c.vocab);
        std::iota(idx.begin(), idx.end(), 0);
        const float* lg = &logits[(size_t)t * c.vocab];
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        char buf[256];
        snprintf(buf, sizeof buf, "pos %d top5: %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f)\n", t0 + t, idx[0], lg[idx[0]], idx[1], lg[idx[1]], idx[2], lg[idx[2]], idx[3], lg[idx[3]], idx[4], lg[idx[4]]);
        top_out += buf;
    }
    printf("%s", top_out.c_str());
    {
        std::ofstream f(out_dir + "/top5.txt");
        f << top_out;
    }
    fprintf(stderr, "[done] total %.1f s; engram lookups %llu seed %llu lru %llu pages %llu\n", now_s() - t_start, (unsigned long long)m.eng->stats().total_lookups,
            (unsigned long long)m.eng->stats().seed_hits, (unsigned long long)m.eng->stats().lru_hits, (unsigned long long)m.eng->stats().ssd_page_reads);
    return 0;
}
